"""
mppp.ibr.nadir - the mark at the bottom of a station panorama (0.97.7; Christian, 13 Nov 2026: "the ods should have a
black circle at the bottom with a white water mark that says Mars 2020 Perseverance Rover, Sol number, site name, and
'made by C. Tate' with image credits").

Straight down a panorama from the mast shows no terrain: the mast itself (masked) and, without the rover, ground
that was filled in.  The pixels within ``NADIR_MARK_DEG`` of the nadir become a disc in the colour of the panorama's
sky with black text (Christian, 13 Nov, later: "Make the water mark black text on a light background the same color
as the sky"; first a black disc with white text).  The
text is laid out as seen looking straight down with north at the top (east to the right), the same in both eyes
(the eye separation is zero there).  It goes into every colour panorama (8-bit and 16-bit, eyes, centre, stacked,
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
SKY_FALLBACK_RGB = (201.0, 172.0, 135.0)     # 8-bit units, when the panorama shows no sky: a butterscotch Mars sky
SKY_MIN_VALUE_8BIT = 150.0                    # "a light background": a darker sky colour is scaled up to this (its hue kept)
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


def disc_image(lines: Sequence[str], size: int = DISC_PX) -> np.ndarray:
    """The text as seen looking down: (size, size) float32, 1 = text, 0 = background; the disc's rim at the square's edge."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    rel = [1.0, 1.0, 0.8, 0.8][:len(lines)] + [0.8] * max(0, len(lines) - 4)       # the title and the sol line larger
    base = 0.085 * size
    fonts, boxes = [], []
    for attempt in range(12):                              # shrink until every line fits its chord of the disc
        fonts = [_font(max(8, int(round(base * r)))) for r in rel]
        if fonts[0] is None:
            break
        boxes = [dr.textbbox((0, 0), t, font=f) for t, f in zip(lines, fonts)]
        gap = 0.45 * base
        total = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
        y, fits = 0.5 * size - 0.5 * total, total < 0.80 * size
        for b in boxes:
            h = b[3] - b[1]
            far = max(abs(y - 0.5 * size), abs(y + h - 0.5 * size))                 # the line's farthest edge from the centre
            chord = 2.0 * np.sqrt(max((0.46 * size) ** 2 - far ** 2, 0.0))
            fits &= (b[2] - b[0]) <= chord
            y += h + gap
        if fits:
            break
        base *= 0.9
    if fonts and fonts[0] is not None:
        gap = 0.45 * base
        total = sum(b[3] - b[1] for b in boxes) + gap * (len(lines) - 1)
        y = 0.5 * size - 0.5 * total
        for t, f, b in zip(lines, fonts, boxes):
            dr.text((0.5 * size - 0.5 * (b[2] - b[0]) - b[0], y - b[1]), t, fill=255, font=f)
            y += (b[3] - b[1]) + gap
        return np.asarray(img, np.float32) / 255.0
    import cv2                                             # no TrueType font on this machine: OpenCV's stroke font
    a = np.zeros((size, size), np.uint8)
    y = 0.30 * size
    for t, r in zip(lines, rel):
        fs = 0.0016 * size * r
        while fs > 0.2 and cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, fs, 2)[0][0] > 0.8 * size:
            fs *= 0.9
        (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        cv2.putText(a, t, (int(0.5 * size - 0.5 * w), int(y + h)), cv2.FONT_HERSHEY_DUPLEX, fs, 255, 2, cv2.LINE_AA)
        y += h + 0.06 * size
    return a.astype(np.float32) / 255.0


def make_mark(width: int, height: int, lines: Sequence[str], radius_deg: float = NADIR_MARK_DEG,
              azimuth0_deg: float = 0.0) -> Dict[str, Any]:
    """The mark for an equirectangular image of ``width`` x ``height`` whose column 0 looks at ``azimuth0_deg``
    (east of north): {"row0": the first row of the disc, "alpha": (height - row0, width) float32, 1 = text,
    "radius_deg", "lines"}."""
    import cv2
    row0 = int(np.ceil(height * (180.0 - float(radius_deg)) / 180.0))
    rows = np.arange(row0, height)
    el = 90.0 - (rows + 0.5) * 180.0 / height                       # elevation of the row's centre (deg)
    rho = np.clip((el + 90.0) / float(radius_deg), 0.0, 1.0)        # 0 at the nadir, 1 at the disc's rim
    az = np.radians(float(azimuth0_deg) + (np.arange(width) + 0.5) * 360.0 / width)
    disc = disc_image(lines)
    half = 0.5 * disc.shape[0]
    mx = (half + half * rho[:, None] * np.sin(az)[None, :]).astype(np.float32)      # east to the right
    my = (half - half * rho[:, None] * np.cos(az)[None, :]).astype(np.float32)      # north at the top
    alpha = cv2.remap(disc, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return {"row0": row0, "alpha": alpha, "radius_deg": float(radius_deg), "lines": list(lines)}


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
