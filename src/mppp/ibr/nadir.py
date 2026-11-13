"""
mppp.ibr.nadir - the mark at the bottom of a station panorama (0.97.7; Christian, 13 Nov 2026: "the ods should have a
black circle at the bottom with a white water mark that says Mars 2020 Perseverance Rover, Sol number, site name, and
'made by C. Tate' with image credits").

Straight down a panorama from the mast shows no terrain: the mast itself (masked) and, without the rover, ground
that was filled in.  The pixels within ``NADIR_MARK_DEG`` of the nadir become a light blue disc with two rows of black
text near its rim, ahead of the rover (:func:`make_mark`).  0.97.13: the disc lies on the surface below the mast
(:func:`on_ground`) - each eye sees it with that surface's disparity (until 0.97.12 it was the same picture in both
eyes: at infinity).  It goes into every colour panorama (8-bit and 16-bit, eyes, centre, stacked,
enhanced), not into the depth products; ``ods.json`` records it ("nadir_mark"), ``mppp ods --no-nadir-mark`` leaves
it out.

Image credit (Christian, 13 Nov, the last word of three): NASA/JPL-Caltech/MSSS for Navcam (and Hazcam) products,
NASA/JPL-Caltech/MSSS/ASU when Mastcam-Z frames are in.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

NADIR_MARK_DEG = 20.0                 # the disc: everything within this angle of the nadir (10 until 0.97.9)
TITLE = "Mars 2020 Perseverance Rover"
AUTHOR = "made by C. Tate"
CREDIT_NAVCAM = "NASA/JPL-Caltech/MSSS"
CREDIT_ZCAM = "NASA/JPL-Caltech/MSSS/ASU"
BACKGROUND_RGB = (176.0, 212.0, 238.0)       # 8-bit units: light blue (Christian, 13 Nov 2026: "the water mark background
#                                              should be light blue"; before: the panorama's sky colour, and black at first)
BACKGROUND_LINEAR_FACTOR = 0.8               # in a linear 16-bit panorama: this x the 8-bit colour, in 8-bit linear units
TEXT_SEPARATOR = "   -   "
TEXT_SPAN_OF_CIRCLE = 0.5                    # a row takes at most this share of the full circle
TEXT_ROWS_IN_BAND = (0.14, 0.33)             # the rows' middles, as shares of the disc's radius from its rim (outer first)
TEXT_MAX_HEIGHT_IN_BAND = (0.20, 0.09)       # a row is at most this share of the disc's radius tall; 0.97.13 (Christian,
#                                              13 Nov 2026: "The top row of the water mark should have larger font"): the
#                                              first row 0.20, the second 0.09 (0.12 both before): the first fills its half circle, 1.6 x the second
SKY_FALLBACK_RGB = BACKGROUND_RGB            # (kept for callers of sky_colour)
SKY_MIN_VALUE_8BIT = 150.0
DISC_PX = 1024                        # the text is drawn on a square of this side, then turned into the panorama's rows
FONTS = ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")


def credit(names: Sequence[str]) -> str:
    """The image credit for the frames ``names``: with a Mastcam-Z frame among them CREDIT_ZCAM, else CREDIT_NAVCAM."""
    return CREDIT_ZCAM if any(str(n)[:1].upper() == "Z" for n in names) else CREDIT_NAVCAM


def sols_of(names: Sequence[str]) -> List[int]:
    """The sols in the frames' names (``NLF_0461_...``), sorted."""
    return sorted({int(str(n)[4:8]) for n in names if len(str(n)) > 8 and str(n)[4:8].isdigit()})


def site_name(site_key: Optional[str]) -> Optional[str]:
    """A site's name for the mark from its key in sites.json: ``van_zyl_overlook_huge`` -> ``Van Zyl Overlook``."""
    if not site_key:
        return None
    parts = [p for p in str(site_key).split("_") if p]
    while parts and parts[-1].lower() in ("tiny", "small", "large", "huge", "zcam", "hazcam", "ecam"):
        parts.pop()
    return " ".join(p.capitalize() for p in parts) or None


SIZE_WORDS = ("tiny", "small", "large", "huge")


def block_name(site_key: Optional[str]) -> Optional[str]:
    """The block's name for a station's products (0.97.17; Christian, 13 Nov 2026: "the name of the ods should just
    be its block name without any large, huge, or tiny additions"): the site's label in the site definitions
    without a size word at its end - ``witch_hazel_large`` -> ``Witch Hazel Hill`` (label "Witch Hazel Hill
    Large"); a key that is not in the definitions: :func:`site_name`."""
    if not site_key:
        return None
    try:
        from ..waypoints.sites import load_site_table
        label = (load_site_table()["sites"].get(str(site_key)) or {}).get("label")
    except Exception:                                      # noqa: BLE001 - then from the key
        label = None
    if not label:
        return site_name(site_key)
    parts = str(label).split()
    while len(parts) > 1 and parts[-1].lower() in SIZE_WORDS:
        parts.pop()
    return " ".join(parts)


def mark_lines(names: Sequence[str], site: Optional[str] = None, sols: Optional[Sequence[int]] = None) -> List[str]:
    """The four lines: the rover, sol and site, the image credit, the author (0.97.16; Christian, 13 Nov 2026: "put
    'made by C. Tate' after the 'image credit: NASA/JPL...'" - the author was before the credit)."""
    s = sorted({int(x) for x in sols}) if sols else sols_of(names)
    sol = "" if not s else (f"Sol {s[0]}" if len(s) == 1 else f"Sols {s[0]}-{s[-1]}" if s[-1] - s[0] == len(s) - 1
                            else "Sols " + ", ".join(str(x) for x in s))
    where = " - ".join(x for x in (sol, site) if x)
    return [x for x in (TITLE, where, f"Image credit: {credit(names)}", AUTHOR) if x]


def _font(size: int):
    from PIL import ImageFont
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return None


def text_strip(text: str, height_px: int = 160) -> np.ndarray:
    """One line of text as (h, w) float32, 1 = ink; cropped to the ink, ``height_px`` tall."""
    from PIL import Image, ImageDraw
    import cv2
    f = _font(int(height_px))
    if f is not None:
        box = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), text, font=f)
        img = Image.new("L", (box[2] - box[0] + 8, box[3] - box[1] + 8), 0)
        ImageDraw.Draw(img).text((4 - box[0], 4 - box[1]), text, fill=255, font=f)
        a = np.asarray(img, np.float32) / 255.0
    else:                                                  # no TrueType font on this machine: OpenCV's stroke font
        fs = height_px / 30.0
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 3)
        im = np.zeros((h + base + 8, w + 8), np.uint8)
        cv2.putText(im, text, (4, h + 4), cv2.FONT_HERSHEY_DUPLEX, fs, 255, 3, cv2.LINE_AA)
        a = im.astype(np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.05)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else a


def make_mark(width: int, height: int, lines: Sequence[str], radius_deg: Optional[float] = None,
              azimuth0_deg: float = 0.0, centre_azimuth_deg: float = 0.0) -> Dict[str, Any]:
    """The mark for an equirectangular image of ``width`` x ``height`` whose column 0 looks at ``azimuth0_deg``
    (east of north): {"row0": the first row of the disc, "alpha": (height - row0, width) float32, 1 = text,
    "radius_deg", "lines", "text" (the two rows), "stretch", ...}.

    The text is two rows near the disc's rim, each over at most half the circle, centred on ``centre_azimuth_deg``
    - the rover's forward direction (Christian, 13 Nov 2026: "two lines covering no more than half the circle ...
    rotated to be read with the viewer facing the front of the rover and looking down").  The first row (the rover,
    the sol and the site) is the outer one.  In the equirectangular image they are two straight lines along the
    bottom strip, upright; looking down in a headset, facing the rover's front, they are two arcs ahead of the
    nadir with the letters upright.  A letter cannot have its own shape in both: the strip is stretched
    1 / cos(elevation) times (3.4 times at -73 degrees).  The letters are made the square root of that wider than
    they are in the flat image (``stretch``), so they are as much too wide there as they are too narrow in the
    headset."""
    import cv2
    radius_deg = NADIR_MARK_DEG if radius_deg is None else float(radius_deg)
    row0 = int(np.ceil(height * (180.0 - radius_deg) / 180.0))
    band = height - row0
    alpha = np.zeros((band, width), np.float32)
    k = (len(lines) + 1) // 2
    rows_text = [TEXT_SEPARATOR.join(lines[:k]), TEXT_SEPARATOR.join(lines[k:])]
    rows_text = [t for t in rows_text if t]
    info = []
    for text, centre, tall in zip(rows_text, TEXT_ROWS_IN_BAND, TEXT_MAX_HEIGHT_IN_BAND):
        strip = text_strip(text)
        el_text = -(90.0 - (1.0 - centre) * radius_deg)                        # the elevation of the row's middle
        stretch = float(np.sqrt(1.0 / max(np.cos(np.radians(el_text)), 1e-3)))
        span = int(round(width * TEXT_SPAN_OF_CIRCLE))                         # at most this many columns
        h = int(round(span * strip.shape[0] / (strip.shape[1] * stretch)))     # its height for that span
        h_max = max(4, int(tall * band))
        if h > h_max:                                                          # no taller than this: it then takes
            h = h_max                                                          # less of the circle
            span = min(span, int(round(h * stretch * strip.shape[1] / strip.shape[0])))
        h = max(4, h)
        t = cv2.resize(strip, (max(8, span), h), interpolation=cv2.INTER_AREA)
        y0 = int(np.clip(round(centre * band - 0.5 * h), 1, max(1, band - h - 1)))
        x_mid = ((float(centre_azimuth_deg) - float(azimuth0_deg)) % 360.0) / 360.0 * width
        cols = (int(round(x_mid - 0.5 * t.shape[1])) + np.arange(t.shape[1])) % width
        alpha[y0:y0 + t.shape[0], cols] = np.maximum(alpha[y0:y0 + t.shape[0], cols], t[:band - y0])
        info.append({"text": text, "stretch": stretch, "height_px": int(h), "elevation_deg": float(el_text),
                     "span_deg": 360.0 * t.shape[1] / width})
    return {"row0": row0, "alpha": alpha, "radius_deg": radius_deg, "lines": list(lines), "text": rows_text,
            "rows": info, "stretch": info[0]["stretch"] if info else 1.0, "centre_azimuth_deg": float(centre_azimuth_deg) % 360.0}


def forward_azimuth_deg(pose: Optional[Dict[str, Any]], north, east) -> float:
    """The azimuth (deg east of north) the rover's front points to, from its pose in the block
    (:func:`mppp.ibr.rover_deck.rover_pose`); 0 without a pose."""
    if not pose:
        return 0.0
    f = np.asarray(pose["R_block_from_rnav"], float)[:, 0]              # RNAV x: forward
    return float(np.degrees(np.arctan2(f @ np.asarray(east, float), f @ np.asarray(north, float))) % 360.0)


def sky_colour(img: np.ndarray, sky: Optional[np.ndarray], fallback, light: Optional[float] = None) -> np.ndarray:
    """The disc's colour for ``img`` (H, W, C; any units and channel order): the median of its sky pixels (``sky``:
    bool (H, W); the upper half only), else ``fallback`` (in the image's units and channel order).  ``light``: the
    brightest channel is at least this (the colour scaled up, its hue kept)."""
    c = None
    if sky is not None:
        m = np.asarray(sky, bool).copy()
        m[img.shape[0] // 2:] = False
        if int(m.sum()) >= 100:
            step = max(1, int(np.sqrt(m.sum() / 2e5)))
            c = np.median(img[::step, ::step][m[::step, ::step]].reshape(-1, img.shape[2]).astype(np.float64), axis=0)
    if c is None or not np.isfinite(c).all() or c.max() <= 0:
        c = np.asarray(fallback, np.float64)
    if light and c.max() < light:
        c = c * (light / c.max())
    return c


GROUND_MARGIN_DEG = 4.0               # an eye's mark is made for this many degrees above the disc's rim as well


def on_ground(mark: Dict[str, Any], height: int, origins: np.ndarray, dirs: np.ndarray, rows: Sequence[int], pivot,
              up, north, east, depth_m: float, azimuth0_deg: float = 0.0):
    """The mark as one eye sees it lying on the ground (0.97.13; Christian, 13 Nov 2026: "It should be placed on the
    ground so that it has the same distances as the ground"): the disc is a drawing on the horizontal plane
    ``depth_m`` below ``pivot``.  ``origins``, ``dirs`` (n, 3): the eye's rays for the image rows ``rows`` = (r0, r1)
    of a panorama ``height`` rows tall (all columns, row by row).  Each ray meets the plane at a point; the direction
    from the pivot to that point is where the point is in the mark's own drawing (:func:`make_mark`, the panorama
    from the pivot).  Returns (alpha, inside), both (r1 - r0, width): the text's share, and whether the pixel is on
    the disc.  With eyes that are apart the two pictures differ by the disparity of that plane - the mark is seen at
    the ground's distance, not at infinity; with no eye separation they are the mark itself."""
    import cv2
    a = mark["alpha"]
    band, width = a.shape
    r0, r1 = int(rows[0]), int(rows[1])
    u, n_, e = (np.asarray(v, float) for v in (up, north, east))
    P = np.asarray(pivot, float)
    o = np.broadcast_to(np.asarray(origins, float), np.asarray(dirs).shape)
    du = np.asarray(dirs, float) @ u
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-float(depth_m) - (o - P) @ u) / du
    ok = (du < -1e-6) & np.isfinite(t) & (t > 0)
    v = o + np.where(ok, t, 1.0)[:, None] * np.asarray(dirs, float) - P
    rr = np.linalg.norm(v, axis=1)
    el = np.degrees(np.arcsin(np.clip((v @ u) / np.maximum(rr, 1e-12), -1.0, 1.0)))
    az = np.degrees(np.arctan2(v @ e, v @ n_))
    x = (((az - float(azimuth0_deg)) % 360.0) / 360.0 * width - 0.5).astype(np.float32)
    y = ((90.0 - el) / 180.0 * height - 0.5 - mark["row0"]).astype(np.float32)
    inside = ok & (el <= -(90.0 - mark["radius_deg"]))
    pad = 2
    aw = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    al = cv2.remap(aw, (x + pad).reshape(r1 - r0, width), np.clip(y, 0, band - 1).reshape(r1 - r0, width), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE)
    inside = inside.reshape(r1 - r0, width)
    return np.where(inside, al, 0.0).astype(np.float32), inside


def apply_mark(img: np.ndarray, mark: Optional[Dict[str, Any]], background) -> np.ndarray:
    """The disc into ``img`` (H, W[, C]; in place): ``background`` (a number or one per channel, in the image's units
    and channel order - :func:`sky_colour`) with the text in black.  A mark with "inside" (an eye's mark from
    :func:`on_ground`; "row0" its first row) covers the pixels inside only."""
    if mark is None:
        return img
    r0, a = mark["row0"], mark["alpha"]
    if img.shape[0] - r0 != a.shape[0] or img.shape[1] != a.shape[1]:
        raise ValueError(f"the mark is for {a.shape[1]} x {r0 + a.shape[0]}, the image is {img.shape[1]} x {img.shape[0]}")
    bg = np.asarray(background, np.float64).reshape(-1)
    v = (1.0 - a)[..., None] * bg[None, None, :] if img.ndim == 3 else (1.0 - a) * float(bg.mean())
    if img.dtype.kind != "f":
        v = np.clip(np.rint(v), 0, np.iinfo(img.dtype).max)
    inside = mark.get("inside")
    if inside is None:
        img[r0:] = v.astype(img.dtype)
    else:
        img[r0:][inside] = v.astype(img.dtype)[inside]
    return img
