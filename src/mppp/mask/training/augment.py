"""
Online augmentation for the rover-mask model (0.96, mask run of 9 Nov 2026; `claude/mask-next-run-2026-11-09-worst-frames-and-augmentation-plan.md`).

Everything here works on one training sample - uint8 RGB (the MPPP 8-bit *linear* image, 0 = invalid) and its uint8
mask (255 = terrain, 0 = hardware or invalid) - with a ``numpy`` Generator, so the DataLoader workers draw new
parameters for every sample.  No torch in this module; :mod:`.train` wires it into :class:`MaskDataset`.

1. **Lighting invariance** (:func:`photometric`): exposure (log-uniform), gamma after the exposure, per-channel
   gains (white-balance jitter), a black-level offset, Poisson-Gaussian noise with sigma proportional to the square
   root of the signal, a synthetic soft-edged shadow (the label is unchanged: shadow is terrain), a haze veil and a
   grey <-> colour swap.  Applied in linear space and clipped to 8 bits at the end; invalid pixels stay 0, valid
   ones never become 0.  It replaces v3's one fixed tone-curve copy per file (``images_variable/``), which was 2x
   data, not invariance, and v4.1's :func:`..dataset.random_tone` (the same recipe drawn online).
2. **LMST weighting** (:func:`lmst_hours_from_stem`, :func:`lmst_weights`): 87 % of the frames are at 11-16 LMST;
   frames before 09:00 and after 16:00 are drawn 4-5x more often.  The LMST comes from the spacecraft clock in the
   product name (sol 0 midnight at SCLK :data:`SCLK_SOL0_MIDNIGHT`, the Mars sol :data:`SOL_SECONDS`): within about
   two minutes of the label's LOCAL_MEAN_SOLAR_TIME over the mission (calibrated on 39,446 Mastcam-Z frames of the
   9 Nov frame list), enough for one-hour bins.
3. **Angular-scale normalisation** (:func:`native_mrad_per_px`, :func:`angular_scale`): the hardware is 5x / 16x
   larger in Mastcam-Z 34 / 110 mm than in a Navcam frame on the 1664 px canvas (Navcam about 1.0 mrad/px at the
   canvas, Z34 0.2, Z110 0.06).  Each sample is resampled to one target angular scale drawn log-uniform from a
   shared range, so the network sees the same hardware at the same size whatever the camera: Navcam gets zoomed in
   (full-resolution crops), Z110 zoomed out.
4. **Mask-conditioned defocus** (:func:`defocus_hardware`): the hardware region blurred (Gaussian, radius 2-12 px at
   the net scale) - the near-field hardware of the 110 mm frames is heavily defocused and the worst frames of the
   v4.2 audit are there.
5. **Cross-camera copy-paste** (:func:`paste_hardware`): a hardware cutout of another frame, scaled to the sample's
   angular scale, pasted onto terrain; the mask gets the cutout's hardware.
6. **Fitted colour transfer** (:func:`color_transfer`): a 3x3 matrix plus offset (Navcam -> Mastcam-Z, fitted from
   co-registered terrain) applied forwards or inverted, so each family also sees the other's colour balance.  The
   matrix is a data product (``color_matrix.json``) - the identity until one is fitted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

# ------------------------------------------------------------------------------------------------- LMST from the clock
SOL_SECONDS = 88775.244147                  # one Mars sol in SI seconds (the SCLK runs at SI seconds to ~1e-6)
# the spacecraft clock at local mean solar midnight of sol 0 (landing site), from the 9 Nov 2026 frame list: 39,446
# Mastcam-Z frames with the label LMST; the constant fits them to a median of 0 s and 99 % within 127 s
SCLK_SOL0_MIDNIGHT = 666893526.0
_STEM_RE = re.compile(r"^([A-Z])([A-Z0-9])([A-Z0-9])_(\d{4})_(\d{10})_(\d{3})")
_ZOOM_RE = re.compile(r"ZCAM\d{5}_(\d{3})")


def lmst_hours_from_stem(stem: str) -> Optional[float]:
    """The frame's local mean solar time in hours (0-24) from the SCLK in a Mars 2020 product name, or None when the
    name has none.  About +-2 min of the label's LMST (see :data:`SCLK_SOL0_MIDNIGHT`)."""
    m = _STEM_RE.match(Path(str(stem)).name.upper())
    if not m:
        return None
    sclk = float(m.group(5)) + float(m.group(6)) / 1000.0
    frac = ((sclk - SCLK_SOL0_MIDNIGHT) / SOL_SECONDS) % 1.0
    return 24.0 * frac


def lmst_weights(stems: Sequence[str], early_h: float = 9.0, late_h: float = 16.0, upweight: float = 4.5,
                 unknown: float = 1.0) -> np.ndarray:
    """Sampling weights per stem: ``upweight`` for frames before ``early_h`` or after ``late_h`` LMST, 1 for the
    midday frames, ``unknown`` for names without a clock."""
    w = np.ones(len(stems), dtype=np.float64)
    for i, s in enumerate(stems):
        h = lmst_hours_from_stem(s)
        if h is None:
            w[i] = unknown
        elif h < early_h or h > late_h:
            w[i] = upweight
    return w


# ------------------------------------------------------------------------------------------------- angular scale
# focal lengths (px) at the native frame width, from the camera models in use (Nov 2026): the angular pixel size is
# 1000 / f mrad at the frame's native width, scaled by (native width / frame width) for downsampled products
NATIVE_FOCAL_PX: Dict[str, Tuple[float, int]] = {
    "N": (2956.0, 5120),                    # Navcam NL / NR, 5120 x 3840 full frame
    "F": (2142.0, 5120), "R": (2142.0, 5120),   # front / rear Hazcams (fisheye: the centre value)
    "Z34": (4720.0, 1648), "Z48": (6700.0, 1648), "Z63": (8615.0, 1648), "Z79": (10763.0, 1648),
    "Z100": (14758.0, 1648), "Z110": (14758.0, 1648),
}


def family_of(stem: str) -> str:
    """``"N"``, ``"F"``, ``"R"``, ``"Z"`` ... (the first letter of a Mars 2020 product name) or ``""``."""
    m = _STEM_RE.match(Path(str(stem)).name.upper())
    return m.group(1) if m else ""


def native_mrad_per_px(stem: str, width: int) -> Optional[float]:
    """The frame's angular pixel size (mrad / px) at its own width, None for a camera not in :data:`NATIVE_FOCAL_PX`
    (WATSON, SuperCam ...)."""
    fam = family_of(stem)
    key = fam
    if fam == "Z":
        m = _ZOOM_RE.search(str(stem).upper())
        if not m:
            return None
        key = f"Z{int(m.group(1))}"
    if key not in NATIVE_FOCAL_PX:
        return None
    f, nat_w = NATIVE_FOCAL_PX[key]
    return 1000.0 / f * (nat_w / float(max(1, width)))


@dataclass(frozen=True)
class AngularScaleConfig:
    prob: float = 1.0
    # the shared target range (mrad / px on the canvas): Z110 native (0.067) to the whole Navcam frame on the 1664
    # canvas (1.04)
    target_range: Tuple[float, float] = (0.08, 1.2)
    max_upscale: float = 2.5                # never enlarge native pixels more than this
    min_long_side: int = 640                # never shrink a frame below this many px on the canvas
    min_valid: float = 0.4                  # a crop with fewer valid pixels than this share is drawn again
    tries: int = 6


def angular_scale(im: np.ndarray, ms: np.ndarray, native_mrad: float, canvas: Tuple[int, int], rng,
                  cfg: AngularScaleConfig = AngularScaleConfig(), target: Optional[float] = None
                  ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Resample (image, mask) so that one canvas pixel spans ``target`` mrad (drawn from ``cfg.target_range``, clamped
    to the frame's own limits), cropped at a random place when it does not fit and zero-padded bottom / right when
    it is smaller - the letterbox's layout.  Returns (image, mask, {"target", "factor", "crop"})."""
    cw, ch = canvas
    h, w = ms.shape[:2]
    lo, hi = cfg.target_range
    lo = max(lo, native_mrad / cfg.max_upscale)                     # factor <= max_upscale
    hi = min(hi, native_mrad * max(h, w) / float(cfg.min_long_side))  # long side >= min_long_side
    if target is None:
        if hi <= lo:
            target = lo
        else:
            target = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    factor = native_mrad / float(target)
    # the crop in native pixels (the part that will fit the canvas), drawn where the frame is valid
    crop_h, crop_w = min(h, int(np.ceil(ch / factor))), min(w, int(np.ceil(cw / factor)))
    y0 = x0 = 0
    if crop_h < h or crop_w < w:
        best = None
        for _ in range(int(cfg.tries)):
            yy, xx = rng.integers(0, h - crop_h + 1), rng.integers(0, w - crop_w + 1)
            sub = im[yy:yy + crop_h:8, xx:xx + crop_w:8]
            valid = float(np.mean(np.any(sub != 0, axis=-1))) if sub.size else 0.0
            if best is None or valid > best[0]:
                best = (valid, int(yy), int(xx))
            if valid >= cfg.min_valid:
                break
        _, y0, x0 = best
    ci, cm = im[y0:y0 + crop_h, x0:x0 + crop_w], ms[y0:y0 + crop_h, x0:x0 + crop_w]
    oh, ow = min(ch, max(1, int(round(ci.shape[0] * factor)))), min(cw, max(1, int(round(ci.shape[1] * factor))))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
    ri = cv2.resize(ci, (ow, oh), interpolation=interp)
    rm = cv2.resize(cm, (ow, oh), interpolation=cv2.INTER_NEAREST)
    out_i = np.zeros((ch, cw, 3), np.uint8); out_i[:oh, :ow] = ri
    out_m = np.zeros((ch, cw), np.uint8); out_m[:oh, :ow] = rm
    return out_i, out_m, {"target": float(target), "factor": float(factor), "crop": float(crop_h * crop_w) / float(h * w)}


# ------------------------------------------------------------------------------------------------- photometric
@dataclass(frozen=True)
class PhotometricConfig:
    exposure_range: Tuple[float, float] = (0.4, 2.5)    # log-uniform multiplier
    exposure_prob: float = 0.8
    gamma_range: Tuple[float, float] = (0.6, 1.8)       # after the exposure
    gain_range: Tuple[float, float] = (0.85, 1.15)      # per channel (white-balance jitter)
    black_level: float = 0.02                           # +- this share of full scale
    noise_prob: float = 0.7
    noise_sigma_range: Tuple[float, float] = (0.002, 0.02)   # sigma at full scale: sigma(x) = s * sqrt(x) (+ s/4 read)
    shadow_prob: float = 0.3
    shadow_range: Tuple[float, float] = (0.3, 0.6)      # the multiplier inside the shadow
    shadow_edge_px: Tuple[int, int] = (8, 80)           # soft edge width
    haze_alpha: Tuple[float, float] = (0.0, 0.15)
    haze_prob: float = 0.5
    grey_prob: float = 0.25                             # grey <-> colour swap (covers mono VCE frames too)


def _shadow_mask(h: int, w: int, rng, edge_px: int) -> np.ndarray:
    """A soft-edged shadow shape in [0, 1] (1 inside): a half-plane or a convex polygon."""
    if rng.random() < 0.5:                             # half-plane at a random angle and offset
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        d = (xx - w / 2) * np.cos(th) + (yy - h / 2) * np.sin(th) - rng.uniform(-0.4, 0.4) * max(h, w)
        return (0.5 * (1.0 - np.tanh(d / max(1.0, edge_px / 2.0)))).astype(np.float32)
    n = int(rng.integers(3, 7))
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    r = rng.uniform(0.15, 0.5) * max(h, w)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    pts = np.c_[cx + r * np.cos(ang) * rng.uniform(0.6, 1.0, n), cy + r * np.sin(ang) * rng.uniform(0.6, 1.0, n)]
    hard = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(hard, np.round(pts).astype(np.int32), 1.0)
    k = max(3, (int(edge_px) // 2) * 2 + 1)
    return cv2.GaussianBlur(hard, (k, k), 0)


def photometric(rgb: np.ndarray, valid: np.ndarray, rng, cfg: PhotometricConfig = PhotometricConfig()
                ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The lighting augmentation of the module docstring on a uint8 *linear* RGB image.  ``valid``: the pixels that
    carry data (others stay 0).  Returns (uint8 RGB, the parameters drawn)."""
    x = rgb.astype(np.float32) / 255.0
    p: Dict[str, Any] = {}
    if rng.random() < cfg.exposure_prob:
        e = float(np.exp(rng.uniform(np.log(cfg.exposure_range[0]), np.log(cfg.exposure_range[1]))))
        x *= e
        p["exposure"] = e
    g = float(rng.uniform(*cfg.gamma_range))
    x = np.clip(x, 0.0, None) ** g
    p["gamma"] = g
    gains = rng.uniform(cfg.gain_range[0], cfg.gain_range[1], size=3).astype(np.float32)
    x *= gains.reshape(1, 1, 3)
    p["gains"] = [float(v) for v in gains]
    bl = float(rng.uniform(-cfg.black_level, cfg.black_level))
    x += bl
    p["black_level"] = bl
    if rng.random() < cfg.shadow_prob:
        sh = _shadow_mask(x.shape[0], x.shape[1], rng, int(rng.integers(cfg.shadow_edge_px[0], cfg.shadow_edge_px[1] + 1)))
        mult = float(rng.uniform(*cfg.shadow_range))
        x *= (1.0 - sh * (1.0 - mult))[..., None]
        p["shadow"] = mult
    if rng.random() < cfg.haze_prob:
        a = float(rng.uniform(*cfg.haze_alpha))
        veil = np.array([0.75, 0.62, 0.50], np.float32) * float(rng.uniform(0.6, 1.0))   # a warm Mars-sky veil
        x = x * (1.0 - a) + a * veil.reshape(1, 1, 3)
        p["haze"] = a
    if rng.random() < cfg.noise_prob:
        s = float(np.exp(rng.uniform(np.log(cfg.noise_sigma_range[0]), np.log(cfg.noise_sigma_range[1]))))
        sig = s * np.sqrt(np.clip(x, 0.0, None)) + s / 4.0
        x += rng.standard_normal(x.shape).astype(np.float32) * sig
        p["noise_sigma"] = s
    if rng.random() < cfg.grey_prob:
        grey = x @ np.array([0.299, 0.587, 0.114], np.float32)
        x = np.repeat(grey[..., None], 3, axis=2)
        p["grey"] = True
    out = np.clip(np.rint(np.clip(x, 0.0, 1.0) * 255.0), 1, 255).astype(np.uint8)
    out[~valid] = 0
    return out, p


# ------------------------------------------------------------------------------------------------- defocus
@dataclass(frozen=True)
class DefocusConfig:
    prob: float = 0.3
    radius_px: Tuple[float, float] = (2.0, 12.0)    # Gaussian sigma at the net scale
    rescale_prob: float = 0.3                       # a random down / up scale of the whole sample as well
    rescale_range: Tuple[float, float] = (0.5, 0.9)


def defocus_hardware(rgb: np.ndarray, ms: np.ndarray, valid: np.ndarray, rng, cfg: DefocusConfig = DefocusConfig()
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Blur the hardware (mask 0 inside the valid area) with a feathered edge; optionally down- and up-scale the
    whole image (a softer frame).  The mask is unchanged."""
    p: Dict[str, Any] = {}
    hard = (ms == 0) & valid
    if not np.any(hard):
        return rgb, p
    sigma = float(rng.uniform(*cfg.radius_px))
    k = int(2 * np.ceil(3 * sigma) + 1)
    blurred = cv2.GaussianBlur(rgb, (k, k), sigma)
    feather = cv2.GaussianBlur(hard.astype(np.float32), (k, k), sigma)[..., None]
    out = np.clip(np.rint(rgb.astype(np.float32) * (1.0 - feather) + blurred.astype(np.float32) * feather), 0, 255).astype(np.uint8)
    p["sigma"] = sigma
    if rng.random() < cfg.rescale_prob:
        f = float(rng.uniform(*cfg.rescale_range))
        h, w = out.shape[:2]
        small = cv2.resize(out, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        p["rescale"] = f
    out[~valid] = 0
    out[valid] = np.clip(out[valid], 1, 255)
    return out, p


# ------------------------------------------------------------------------------------------------- copy-paste
@dataclass(frozen=True)
class CopyPasteConfig:
    prob: float = 0.3
    min_area: float = 0.005                 # the cutout's hardware share of the donor frame, at least
    max_fill: float = 0.5                   # the paste's share of the target frame, at most
    feather_px: int = 3


def hardware_cutout(rgb: np.ndarray, ms: np.ndarray, min_area: float = 0.005
                    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The tight bounding box of the frame's hardware (mask 0 inside the valid area): (RGB crop, hardware alpha 0/1)
    or None when there is too little."""
    valid = np.any(rgb != 0, axis=-1)
    hard = (ms == 0) & valid
    if hard.mean() < min_area:
        return None
    ys, xs = np.where(hard)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rgb[y0:y1, x0:x1].copy(), hard[y0:y1, x0:x1].astype(np.uint8)


def paste_hardware(rgb: np.ndarray, ms: np.ndarray, cutout: Tuple[np.ndarray, np.ndarray], scale: float, rng,
                   cfg: CopyPasteConfig = CopyPasteConfig()) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Paste a donor's hardware cutout, resized by ``scale`` (donor angular scale / target angular scale), at a
    random place of the sample; the mask gets 0 where the cutout's hardware lands.  Returns (rgb, mask, info)."""
    crgb, calpha = cutout
    h, w = ms.shape[:2]
    ch, cw = max(1, int(round(crgb.shape[0] * scale))), max(1, int(round(crgb.shape[1] * scale)))
    f = min(1.0, np.sqrt(cfg.max_fill * h * w / float(ch * cw)))      # never more than max_fill of the frame
    ch, cw = max(1, int(ch * f)), max(1, int(cw * f))
    ch, cw = min(ch, h), min(cw, w)
    crgb = cv2.resize(crgb, (cw, ch), interpolation=cv2.INTER_AREA if scale * f < 1 else cv2.INTER_LINEAR)
    calpha = cv2.resize(calpha, (cw, ch), interpolation=cv2.INTER_NEAREST)
    if not np.any(calpha):
        return rgb, ms, {}
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    k = max(3, (int(cfg.feather_px) // 2) * 2 + 1)
    a = cv2.GaussianBlur(calpha.astype(np.float32), (k, k), 0)[..., None]
    a = np.maximum(a, calpha[..., None].astype(np.float32))
    out = rgb.copy()
    region = out[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    out[y0:y0 + ch, x0:x0 + cw] = np.clip(np.rint(region * (1.0 - a) + crgb.astype(np.float32) * a), 0, 255).astype(np.uint8)
    om = ms.copy()
    om[y0:y0 + ch, x0:x0 + cw][calpha > 0] = 0
    return out, om, {"scale": float(scale * f), "box": [y0, x0, ch, cw]}


# ------------------------------------------------------------------------------------------------- colour transfer
@dataclass
class ColorMatrix:
    """``rgb_other = M @ rgb + b`` on the [0, 1] scale, fitted Navcam -> Mastcam-Z from co-registered terrain.
    ``load`` reads ``{"M": [[...]x3], "b": [...], "from": "N", "to": "Z", "note": ...}``."""
    M: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    b: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    src: str = "N"
    dst: str = "Z"
    note: str = "identity"

    @classmethod
    def load(cls, path: PathLike) -> "ColorMatrix":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.asarray(d["M"], np.float32).reshape(3, 3), np.asarray(d.get("b", [0, 0, 0]), np.float32),
                   str(d.get("from", "N")), str(d.get("to", "Z")), str(d.get("note", "")))

    def is_identity(self) -> bool:
        return bool(np.allclose(self.M, np.eye(3)) and np.allclose(self.b, 0))

    def apply(self, rgb: np.ndarray, valid: np.ndarray, inverse: bool = False) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        if inverse:
            Mi = np.linalg.inv(self.M)
            y = (x - self.b.reshape(1, 1, 3)) @ Mi.T
        else:
            y = x @ self.M.T + self.b.reshape(1, 1, 3)
        out = np.clip(np.rint(np.clip(y, 0, 1) * 255.0), 1, 255).astype(np.uint8)
        out[~valid] = 0
        return out


def color_transfer(rgb: np.ndarray, valid: np.ndarray, family: str, cm: ColorMatrix, rng, prob: float = 0.3
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """With probability ``prob``, move a ``cm.src`` frame's colours to ``cm.dst`` and a ``cm.dst`` frame's the other
    way; other families untouched."""
    if cm.is_identity() or rng.random() >= prob:
        return rgb, {}
    if family == cm.src:
        return cm.apply(rgb, valid, inverse=False), {"color_transfer": f"{cm.src}->{cm.dst}"}
    if family == cm.dst:
        return cm.apply(rgb, valid, inverse=True), {"color_transfer": f"{cm.dst}->{cm.src}"}
    return rgb, {}


# ------------------------------------------------------------------------------------------------- the recipe
@dataclass
class AugmentConfig:
    """What :class:`..train.MaskDataset` applies to a training sample, in this order: angular scale (or the letterbox),
    copy-paste, defocus, colour transfer, photometric.  The flip stays in the dataset."""
    photometric: bool = False
    photometric_prob: float = 0.9
    photo: PhotometricConfig = field(default_factory=PhotometricConfig)
    angular_scale: bool = False
    angular: AngularScaleConfig = field(default_factory=AngularScaleConfig)
    defocus: bool = False
    defocus_cfg: DefocusConfig = field(default_factory=DefocusConfig)
    copy_paste: bool = False
    copy_paste_cfg: CopyPasteConfig = field(default_factory=CopyPasteConfig)
    color_matrix: Optional[str] = None      # path of a ColorMatrix json; None = off
    color_prob: float = 0.3

    def summary(self) -> Dict[str, Any]:
        return {"photometric": self.photometric, "angular_scale": self.angular_scale, "defocus": self.defocus,
                "copy_paste": self.copy_paste, "color_matrix": self.color_matrix}


RECIPES: Dict[str, Dict[str, Any]] = {
    # the run order of the 9 Nov plan
    "A": {"photometric": True, "lmst_sampler": True},
    "B": {"photometric": True, "lmst_sampler": True, "angular_scale": True, "defocus": True, "quad_split_above": 4000},
    "C": {"photometric": True, "lmst_sampler": True, "angular_scale": True, "defocus": True, "quad_split_above": 4000,
          "copy_paste": True, "color_matrix": "auto", "family_batches": True},
}
