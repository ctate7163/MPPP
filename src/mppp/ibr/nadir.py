"""
mppp.ibr.nadir - the mark at the bottom of a station panorama (0.97.7; Christian, 13 Nov 2026: "the ods should have a
black circle at the bottom with a white water mark that says Mars 2020 Perseverance Rover, Sol number, site name, and
'made by C. Tate' with image credits").

Straight down a panorama from the mast shows no terrain: the mast itself (masked) and, without the rover, ground
that was filled in.  The pixels within ``NADIR_MARK_DEG`` of the nadir become a light blue disc with black text that
runs once around it near the rim (:func:`make_mark`), the same in both eyes (the eye separation is zero there).  It goes into every colour panorama (8-bit and 16-bit, eyes, centre, stacked,
enhanced), not into the depth products; ``ods.json`` records it ("nadir_mark"), ``mppp ods --no-nadir-mark`` leaves
it out.

Image credit (Christian, 13 Nov): "NASA/JPL/MSSS/ASU for zcam products.  For just navcam, it should be just
NASA/JPL."
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

NADIR_MARK_DEG = 10.0                 # the disc: everything within this angle of the nadir
TITLE = "Mars 2020 Perseverance Rover"
AUTHOR = "made by C. Tate"
CREDIT_NAVCAM = "NASA/JPL"
CREDIT_ZCAM = "NASA/JPL/MSSS/ASU"
BACKGROUND_RGB = (176.0, 212.0, 238.0)       # 8-bit units: light blue (Christian, 13 Nov 2026: "the water mark background
#                                              should be light blue"; before: the panorama's sky colour, and black at first)
BACKGROUND_LINEAR_FACTOR = 0.8               # in a linear 16-bit panorama: this x the 8-bit colour, in 8-bit linear units
TEXT_SEPARATOR = "   -   "
TEXT_SPAN_OF_CIRCLE = 0.94                   # the line takes this share of the full circle (the rest is the gap at north)
TEXT_CENTRE_IN_BAND = 0.36                   # the line's middle, as a share of the disc's radius from its rim
TEXT_MAX_HEIGHT_IN_BAND = 0.5                # the line is at most this share of the disc's radius tall
SKY_FALLBACK_RGB = BACKGROUND_RGB            # (kept for callers of sky_colour)
SKY_MIN_VALUE_8BIT = 150.0
DISC_PX = 1024                        # the text is drawn on a square of this side, then turned into the panorama's rows
FONTS = ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")


def credit(names: Sequence[str]) -> str:
    """The image credit for the frames ``names``: with a Mastcam-Z frame among them NASA/JPL/MSSS/ASU, else NASA/JPL."""
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


def mark_lines(names: Sequence[str], site: Optional[str] = None, sols: Optional[Sequence[int]] = None) -> List[str]:
    """The four lines: the rover, sol and site, the author, the image credit."""
    s = sorted({int(x) for x in sols}) if sols else sols_of(names)
    sol = "" if not s else (f"Sol {s[0]}" if len(s) == 1 else f"Sols {s[0]}-{s[-1]}" if s[-1] - s[0] == len(s) - 1
                            else "Sols " + ", ".join(str(x) for x in s))
    where = " - ".join(x for x in (sol, site) if x)
    return [x for x in (TITLE, where, AUTHOR, f"Image credit: {credit(names)}") if x]


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


def make_mark(width: int, height: int, lines: Sequence[str], radius_deg: float = NADIR_MARK_DEG,
              azimuth0_deg: float = 0.0) -> Dict[str, Any]:
    """The mark for an equirectangular image of ``width`` x ``height`` whose column 0 looks at ``azimuth0_deg``
    (east of north): {"row0": the first row of the disc, "alpha": (height - row0, width) float32, 1 = text,
    "radius_deg", "lines", "text", "stretch"}.

    The text is one line that runs once around the disc near its rim, starting at north and reading towards the
    east (Christian, 13 Nov 2026: "the text should wrap around the perimeter of the circle so that it can be read
    in the cylindrical projection too").  In the equirectangular image it is a straight line along the bottom strip,
    upright; looking down in a headset it is a ring with the letters' tops outwards.  A letter cannot have its own
    shape in both: near the nadir the strip is stretched 1 / cos(elevation) times (7 times at -82 degrees).  The
    letters are made ``stretch`` = the square root of that wider than they are in the flat image, so they are as much
    too wide in the image as they are too narrow in the headset (about 2.7 times each)."""
    import cv2
    row0 = int(np.ceil(height * (180.0 - float(radius_deg)) / 180.0))
    band = height - row0
    alpha = np.zeros((band, width), np.float32)
    text = TEXT_SEPARATOR.join(lines)
    strip = text_strip(text)
    el_text = -(90.0 - (1.0 - TEXT_CENTRE_IN_BAND) * float(radius_deg))        # the elevation of the line's middle
    stretch = float(np.sqrt(1.0 / max(np.cos(np.radians(el_text)), 1e-3)))
    span = int(round(width * TEXT_SPAN_OF_CIRCLE))                             # the columns the text takes
    h = int(round(span * strip.shape[0] / (strip.shape[1] * stretch)))         # its height for that stretch
    h_max = max(4, int(TEXT_MAX_HEIGHT_IN_BAND * band))
    if h > h_max:                                                              # a short text: no taller than this,
        h = h_max                                                              # and it then takes less of the circle
        span = min(span, int(round(h * stretch * strip.shape[1] / strip.shape[0])))
    h = max(4, h)
    t = cv2.resize(strip, (max(8, span), h), interpolation=cv2.INTER_AREA)
    y0 = int(np.clip(round(TEXT_CENTRE_IN_BAND * band - 0.5 * h), 1, max(1, band - h - 1)))
    x0 = int(round(((0.0 - float(azimuth0_deg)) % 360.0) / 360.0 * width + 0.5 * (width * TEXT_SPAN_OF_CIRCLE - span)
                   + 0.5 * width * (1.0 - TEXT_SPAN_OF_CIRCLE)))               # from north, centred in its span
    cols = (x0 + np.arange(t.shape[1])) % width
    alpha[y0:y0 + t.shape[0], cols] = t[:band - y0]
    return {"row0": row0, "alpha": alpha, "radius_deg": float(radius_deg), "lines": list(lines), "text": text,
            "stretch": stretch, "text_height_px": int(h), "text_elevation_deg": float(el_text)}


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


def apply_mark(img: np.ndarray, mark: Optional[Dict[str, Any]], background) -> np.ndarray:
    """The disc into ``img`` (H, W[, C]; in place): ``background`` (a number or one per channel, in the image's units
    and channel order - :func:`sky_colour`) with the text in black."""
    if mark is None:
        return img
    r0, a = mark["row0"], mark["alpha"]
    if img.shape[0] - r0 != a.shape[0] or img.shape[1] != a.shape[1]:
        raise ValueError(f"the mark is for {a.shape[1]} x {r0 + a.shape[0]}, the image is {img.shape[1]} x {img.shape[0]}")
    bg = np.asarray(background, np.float64).reshape(-1)
    v = (1.0 - a)[..., None] * bg[None, None, :] if img.ndim == 3 else (1.0 - a) * float(bg.mean())
    if img.dtype.kind != "f":
        v = np.clip(np.rint(v), 0, np.iinfo(img.dtype).max)
    img[r0:] = v.astype(img.dtype)
    return img
