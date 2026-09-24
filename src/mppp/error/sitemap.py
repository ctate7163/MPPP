"""
mppp_error.sitemap -- load an overhead site mosaic and crop it to a map extent.

Used to place a real Mastcam-Z (or Navcam) overhead context image behind the
FIXED panel's otherwise-blank "improvement over itself" slot.

ALIGNMENT ASSUMPTIONS -- READ BEFORE TRUSTING PIXEL POSITIONS
---------------------------------------------------------------
This module assumes the supplied image is:
  1. North-up (image +x = East, image top = North).
  2. Square, of known real-world width `width_m`, in metres.
  3. Centred on the SAME point that the working frame's anchor is centred on.

None of these are verified against the image itself -- there is no embedded
georeferencing read here, only what you assert via `width_m`.  If the source
product carries its own registration (map-projected label, PDS4 metadata,
cartography info), use that in preference to visual alignment, and treat any
overlay from this module as approximate until checked against a known feature
(e.g. the rover deck at the anchor waypoint, a named landmark) visible in both
the image and the field maps.

Network access
---------------
Image hosts are frequently NOT on this environment's egress allowlist (this
was true for mcz-images.sese.asu.edu at the time this module was written).
`load_site_image` tries a direct fetch, and if the host is blocked, raises with
the exact host name so the caller can either add it to network settings or
download the file out of band and pass a local path instead.
"""

from __future__ import annotations

import io
import os
import numpy as np
from typing import Tuple, Optional
from urllib.parse import urlparse

__all__ = ["load_site_image", "crop_to_extent", "SiteImageError", "SiteCrop",
           "trim_border_fraction"]


class SiteCrop(np.ndarray):
    """
    ndarray subclass carrying a `partial_coverage` flag.

    Plain np.ndarray does not support arbitrary attribute assignment, so a thin
    subclass is used rather than silently dropping the flag or returning a
    tuple that would break `ax.imshow(crop, ...)` call sites.
    """
    def __new__(cls, arr, partial_coverage: bool = False):
        obj = np.asarray(arr).view(cls)
        obj.partial_coverage = partial_coverage
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.partial_coverage = getattr(obj, "partial_coverage", False)


class SiteImageError(RuntimeError):
    pass


def trim_border_fraction(image: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """
    Trim `frac` off EACH side of a square image, keeping the centre
    (1 - 2*frac) fraction in both width and height.

    Use this when the nominal footprint printed in a product's filename (e.g.
    "...50m.jpg") is smaller than the actual pixel canvas -- some vertical
    mosaic products carry a wider rendered margin than their stated footprint,
    so treating the raw file as if it were exactly that footprint stretches
    ground features to the wrong scale. frac=0.25 keeps the centre 50% of the
    canvas in each dimension, i.e. assumes the raw file's true footprint is
    width_m / (1 - 2*frac) = 2x the nominal value when frac=0.25.

    This is a manual correction, not a measurement: it is only as good as the
    frac you supply. Verify against a known real-world separation (e.g. the
    distance between two visible surface features) before trusting it
    quantitatively.
    """
    if not (0.0 <= frac < 0.5):
        raise ValueError("frac must be in [0, 0.5)")
    h, w = image.shape[:2]
    y0, y1 = int(round(h * frac)), int(round(h * (1 - frac)))
    x0, x1 = int(round(w * frac)), int(round(w * (1 - frac)))
    return image[y0:y1, x0:x1]


def load_site_image(source: str, cache_dir: str = "/tmp/mppp_sitemap_cache"):
    """
    Load an image from a local path or a URL, returning an (H,W,3) uint8 array.

    URLs are cached locally by filename so repeated runs (e.g. re-rendering a
    figure) don't re-fetch.  Requires Pillow; raises SiteImageError with an
    actionable message if the fetch fails, rather than a bare exception from
    deep inside requests/PIL.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise SiteImageError(
            "Pillow is required to load site images (pip install Pillow)") from e

    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.basename(parsed.path) or "site_image.jpg"
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            try:
                import urllib.request
                req = urllib.request.Request(
                    source, headers={"User-Agent": "mppp-sitemap/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                with open(local, "wb") as fh:
                    fh.write(data)
            except Exception as e:                                # noqa: BLE001
                raise SiteImageError(
                    f"could not fetch {source!r}: {e}. "
                    f"If this is a network/allowlist error, either add "
                    f"{parsed.netloc!r} to network egress settings, or "
                    f"download the file yourself and pass a local path to "
                    f"--site-image instead.") from e
        source = local
    elif not os.path.exists(source):
        raise SiteImageError(f"site image not found: {source!r}")

    try:
        img = Image.open(source).convert("RGB")
    except Exception as e:                                        # noqa: BLE001
        raise SiteImageError(f"could not open {source!r} as an image: {e}") from e
    return np.asarray(img)


def crop_to_extent(image: np.ndarray, image_width_m: float,
                   extent: Tuple[float, float, float, float],
                   image_center_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """
    Crop a north-up, square overhead image to a target map extent.

    image           : (H,W,3) array, assumed square in real-world coverage
                      (image_width_m x image_width_m), north-up, centred on
                      image_center_xy in the SAME relative-easting/northing
                      frame the caller's grid uses.
    image_width_m   : real-world width AND height of the full input image.
    extent          : (xmin, xmax, ymin, ymax) target extent, e.g. from
                      Grid.extent_edges().
    image_center_xy : where the image's centre pixel sits in that frame.
                      Default (0,0) assumes the image is already centred on the
                      grid's own origin (e.g. anchor='first' with the image
                      centred on the first waypoint).

    Returns an RGB array cropped (and padded with white if the extent exceeds
    the source image) to match `extent`, suitable for
    `ax.imshow(crop, extent=extent, origin='upper')`.

    If extent is not fully contained within the source image's coverage, the
    output is padded rather than upsampled, and a warning is attached as an
    attribute `.mppp_partial_coverage = True` on cases where truncation
    occurred, so the caller can flag it in the title if desired.
    """
    H, W = image.shape[:2]
    xmin, xmax, ymin, ymax = extent
    cx, cy = image_center_xy
    half = image_width_m / 2.0

    # pixel scale: image spans [cx-half, cx+half] in x, [cy-half, cy+half] in y
    # row 0 is the TOP of the image = north = +y, so row increases as y decreases
    px_per_m_x = W / image_width_m
    px_per_m_y = H / image_width_m

    def x_to_col(x):
        return (x - (cx - half)) * px_per_m_x

    def y_to_row(y):
        return ((cy + half) - y) * px_per_m_y

    col0, col1 = x_to_col(xmin), x_to_col(xmax)
    row0, row1 = y_to_row(ymax), y_to_row(ymin)   # ymax -> smaller row (top)

    c0, c1 = int(round(col0)), int(round(col1))
    r0, r1 = int(round(row0)), int(round(row1))
    out_w, out_h = c1 - c0, r1 - r0
    if out_w <= 0 or out_h <= 0:
        raise SiteImageError(
            f"requested extent {extent} does not overlap the image "
            f"(width {image_width_m} m centred at {image_center_xy})")

    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    sc0, sc1 = max(c0, 0), min(c1, W)
    sr0, sr1 = max(r0, 0), min(r1, H)
    dc0, dr0 = sc0 - c0, sr0 - r0
    if sc1 > sc0 and sr1 > sr0:
        canvas[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = \
            image[sr0:sr1, sc0:sc1]
    else:
        raise SiteImageError(
            f"requested extent {extent} falls entirely outside the "
            f"{image_width_m} m source image")

    partial = (c0 < 0 or c1 > W or r0 < 0 or r1 > H)
    return SiteCrop(canvas, partial_coverage=partial)
