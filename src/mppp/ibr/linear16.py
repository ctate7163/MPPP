"""
mppp.ibr.linear16 - the station's frames as 16-bit linear images for the panorama (0.96.1, Christian 10 Nov 2026:
"please implement a 16-bit process so that I can color correct it later ... I would like to have more control over
the color space").

A block holds 8-bit images (``images/``, from ``processed/images_png8``): the 16-bit linear product divided by 64 and
clipped at 255, i.e. at a quarter of the 16-bit range - the bright rover deck and the sky towards the sun are flat
white there although the camera measured them.  For the panorama the frames are made again from their PDS products
with the block's own processing settings (``processed/mppp_config_v*.json``), mask inference off, and kept as

    <project>/images16/<image>.png         uint16 RGB, linear:  radiance x white balance x brightness
                                           / zenith scale x ``scale_rad_to_int16`` (2e5); 0 = no data
    <project>/images16/<image>.sat.png     255 where the detector was saturated (a plateau at the top of a channel)
    <project>/images16/linear16.json       per frame: the factors above, the share saturated, the share the 8-bit
                                           image clips

The geometry is that of the block's 8-bit image (same padding and size), so poses, cameras and masks apply as they
are.  The PDS product is looked for under ``--pds`` / ``MPPP_PDS`` / the archive the block's manifest names, in the
frame's sol folder.

Saturation: a RAD product has no flag for it and its level differs per frame (exposure, flat field), so it is found
as a plateau - a channel with many pixels within ``SAT_BAND`` of its maximum, piled up against the band below.  Such pixels are white in truth but
come out tinted after the white balance (the magenta band towards the sun); the panorama makes them neutral and
prefers another frame there.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.ibr.linear16")
PathLike = Union[str, Path]

DIR = "images16"
RECORD = "linear16.json"
UNITS_PER_8BIT = 64.0            # 16-bit value / 64 = the 8-bit value (color.scale_int8_to_int16)
SAT_BAND = 0.015                 # a plateau: pixels within this fraction of the channel's maximum
SAT_MIN_SHARE = 2e-4             # ... at least this share of the frame (and SAT_MIN_PIXELS)
SAT_MIN_PIXELS = 200
SAT_PILE_UP = 3.0                # ... and this many times the pixels of the band just below (a pile-up, not a slope)


def saturation_mask(raw: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Pixels on a plateau at the top of any channel of ``raw`` (H, W, C; the radiance before the white balance)."""
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = raw[..., None]
    v = np.ones(raw.shape[:2], bool) if valid is None else np.asarray(valid, bool)
    out = np.zeros(raw.shape[:2], bool)
    n = int(v.sum())
    if not n:
        return out
    for c in range(raw.shape[2]):
        ch = raw[..., c]
        top = float(ch[v].max())
        if top <= 0:
            continue
        m = v & (ch >= (1.0 - SAT_BAND) * top)
        below = v & (ch >= (1.0 - 2.0 * SAT_BAND) * top) & ~m       # the band of the same width just under it
        if m.sum() >= max(SAT_MIN_PIXELS, SAT_MIN_SHARE * n) and m.sum() >= SAT_PILE_UP * max(1, below.sum()):
            out |= m
    return out


def linear16_of_product(path: PathLike, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(uint16 RGB linear, saturated mask, facts) of one PDS product with the block's processing settings."""
    from ..image import MPPPImage
    cfg = copy.deepcopy(config)
    cfg.setdefault("masking", {})["infer_mask"] = False            # the block's masks are used as they are
    cfg.setdefault("camera_model", {})["extrinsics_from_waypoints"] = False
    sel = cfg.setdefault("selection", {})
    for k in list(sel):                                            # the frame was selected when the block was made
        if k.startswith("max_") or k == "lmst_window_h":
            sel[k] = None
    im = MPPPImage(path, cfg, None)
    wb = np.asarray(im.white_balance, float) * float(im.brightness)
    if im.scale_zenith:
        wb = wb / float(im.scale_zenith)
    valid = im.image_int16.max(axis=2) > 0
    raw = im.image_rad.astype(np.float64) / wb.reshape(1, 1, 3)
    sat = saturation_mask(raw, valid)
    i16 = im.image_int16
    facts = {"product": Path(path).name, "white_balance": [float(v) for v in im.white_balance],
             "brightness": float(im.brightness), "scale_zenith": None if im.scale_zenith is None else float(im.scale_zenith),
             "scale_rad_to_int16": float(cfg["color"]["scale_rad_to_int16"]),
             "to_radiance_divide_by": [float(v) for v in wb * float(cfg["color"]["scale_rad_to_int16"])],
             "saturated_share": float(sat.sum() / max(1, valid.sum())),
             "clipped_in_8bit_share": float(((i16 >= 255 * UNITS_PER_8BIT).any(axis=2) & valid).sum() / max(1, valid.sum())),
             "clipped_in_16bit_share": float(((i16 >= 65535).any(axis=2) & valid).sum() / max(1, valid.sum())),
             "max_16bit": [int(v) for v in i16.reshape(-1, 3).max(axis=0)]}
    return i16, sat, facts


def block_config(project) -> Dict[str, Any]:
    """The processing settings the block's images were made with (``processed/mppp_config_v*.json``, the newest)."""
    from ..config import load_config
    for d in (Path((getattr(project, "settings", None) or {}).get("processed_dir", "")), Path(project.root).parent / "processed"):
        if d.is_dir():
            c = sorted(d.glob("mppp_config_v*.json"), key=lambda f: f.stat().st_mtime)
            if c:
                import warnings
                # 0.97.15, a bug since 0.96.1: the snapshot holds the settings under "config" (with the version and
                # the run's facts beside them); the whole file was merged over the defaults, so its "config" was an
                # unknown key and the 16-bit frames were made with TODAY'S defaults, not the block's settings
                snap = json.loads(c[-1].read_text(encoding="utf-8"))
                user = snap.get("config") if isinstance(snap.get("config"), dict) else snap
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cfg = load_config(user)
                # a block processed before the Hazcams had their own balance: its 8-bit Hazcam frames have the
                # Navcam balance, and the 16-bit ones must match them
                if "white_balance_hazcam" not in (user.get("color") or {}):
                    cfg["color"]["white_balance_hazcam"] = None
                return cfg
    raise FileNotFoundError(f"no processed/mppp_config_v*.json for {project.root}")


def archive_roots(project, pds: Optional[PathLike] = None) -> List[Path]:
    """Where the PDS products may be: ``pds``, MPPP_PDS, and the archive the block's manifest names (the parent of
    a five-digit sol folder in one of its file paths)."""
    roots: List[Path] = []
    for r in (pds, os.environ.get("MPPP_PDS")):
        if r:
            roots.append(Path(r))
    try:
        from ..sfm.workdir import load_manifest
        for d in (Path((getattr(project, "settings", None) or {}).get("processed_dir", "")), Path(project.root).parent / "processed"):
            if d.is_dir() and list(d.glob("mppp_manifest_v*.json")):
                man, _ = load_manifest(d)
                for e in (man.get("skipped") or [])[:50] + (man.get("failed") or [])[:50]:
                    parts = Path(str(e.get("file", ""))).parts
                    for i, p in enumerate(parts):
                        if re.fullmatch(r"\d{5}", p) and i:
                            roots.append(Path(*parts[:i]))
                            break
                break
    except Exception:                                              # noqa: BLE001 - a hint only
        pass
    seen, out = set(), []
    for r in roots:
        k = str(r).lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def find_product(stem: str, roots: Sequence[PathLike]) -> Optional[Path]:
    """``<stem>.IMG`` in the sol folder (``<root>/<sol, 5 digits>/**``) of one of ``roots``, or directly in one."""
    sol = stem[4:8]
    for r in roots:
        r = Path(r)
        for ext in (".IMG", ".img"):
            if (r / (stem + ext)).is_file():
                return r / (stem + ext)
        d = r / f"{int(sol):05d}" if sol.isdigit() else None
        if d is not None and d.is_dir():
            for f in d.rglob(stem + ".*"):
                if f.suffix.lower() == ".img":
                    return f
    return None


def files(project, name: str) -> Tuple[Path, Path]:
    d = Path(project.root) / DIR
    return d / name, d / (Path(name).stem + ".sat.png")


def ensure_linear16(project, names: Iterable[str], pds: Optional[PathLike] = None, force: bool = False,
                    make: Callable = linear16_of_product) -> Dict[str, Any]:
    """The 16-bit linear image and the saturation mask of every frame of ``names`` in ``<project>/images16/``
    (made where missing).  Returns {"dir", "have": [names], "missing": {name: why}, "made": n, "roots": [...]}."""
    import cv2
    out = Path(project.root) / DIR
    out.mkdir(parents=True, exist_ok=True)
    recf = out / RECORD
    record = json.loads(recf.read_text(encoding="utf-8")) if recf.is_file() else {}
    names = list(names)
    todo = [n for n in names if force or not all(f.is_file() for f in files(project, n))]
    have = [n for n in names if n not in todo]
    missing: Dict[str, str] = {}
    roots = archive_roots(project, pds) if todo else []
    cfg = None
    if todo:
        try:
            cfg = block_config(project)
        except FileNotFoundError as e:
            missing = {n: str(e) for n in todo}
            todo = []
    made = 0
    for n in todo:
        stem = Path(n).stem
        prod = find_product(stem, roots)
        if prod is None:
            missing[n] = f"no {stem}.IMG under {[str(r) for r in roots] or 'any archive (set MPPP_PDS or --pds)'}"
            continue
        try:
            i16, sat, facts = make(prod, cfg)
        except Exception as e:                                     # noqa: BLE001 - reported per frame
            missing[n] = f"{type(e).__name__}: {e}"
            continue
        ref = cv2.imread(str(Path(project.root) / "images" / n), cv2.IMREAD_UNCHANGED)
        if ref is not None and ref.shape[:2] != i16.shape[:2]:
            log.warning("linear16: %s is %dx%d, the block's image %dx%d - fitted to it", n, i16.shape[1], i16.shape[0],
                        ref.shape[1], ref.shape[0])
            h, w = ref.shape[:2]
            ys, xs = np.nonzero(i16.max(axis=2) > 0)
            if len(ys) and ys.max() < h and xs.max() < w:          # the same pixels in a larger padded frame: cut
                i16, sat = i16[:h, :w], sat[:h, :w]
                facts["cropped"] = True
            else:
                i16 = cv2.resize(i16, (w, h), interpolation=cv2.INTER_AREA)
                sat = cv2.resize(sat.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                facts["resized"] = True
        f, s = files(project, n)
        cv2.imwrite(str(f), np.ascontiguousarray(i16[..., ::-1]))
        cv2.imwrite(str(s), sat.astype(np.uint8) * 255)
        facts["source"] = str(prod)
        record[n] = facts
        have.append(n)
        made += 1
        log.info("linear16: %s  saturated %.2f %%, clipped in 8-bit %.2f %%, in 16-bit %.2f %%", n,
                 100 * facts["saturated_share"], 100 * facts["clipped_in_8bit_share"], 100 * facts["clipped_in_16bit_share"])
    if made:
        recf.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    return {"dir": str(out), "have": have, "missing": missing, "made": made, "roots": [str(r) for r in roots],
            "frames": {n: record.get(n) for n in have}}


def tone_soft(x: np.ndarray, knee: float = 190.0, top: float = 255.0) -> np.ndarray:
    """Linear values in 8-bit units (the 16-bit / 64; up to 1024) -> 0..255 for viewing: unchanged up to ``knee``
    on the brightest channel, above it rolled off towards ``top`` (reached at the 16-bit ceiling), every channel
    scaled alike so the colour keeps its hue."""
    x = np.asarray(x, np.float32)
    m = x.max(axis=-1)
    hi = 65535.0 / UNITS_PER_8BIT
    t = np.clip((m - knee) / (hi - knee), 0.0, 1.0)
    # a curve with slope 1 at the knee and 0 at the ceiling, from knee to top
    k = (hi - knee) / (top - knee)                                  # input span / output span (> 1)
    y = knee + (top - knee) * (1.0 - (1.0 - t) ** k)
    s = np.where(m > knee, y / np.maximum(m, 1e-6), 1.0)
    return np.clip(x * s[..., None], 0.0, 255.0)


PIPELINE_KNEE = 235.0            # 0.98.0: the standard 8-bit view is the pipeline's own 8-bit scale (16-bit / 64) up to here ...


def tone_pipeline(x: np.ndarray) -> np.ndarray:
    """0.98.0 (Christian, 14 Nov 2026: "the standard 8-bit version should be made from the 16 bit images and have the
    same range as the standard pipeline 8-bit images"): linear values in 8-bit units (the 16-bit / 64) -> 0..255 on
    the scale of the block's own 8-bit frames - value for value up to PIPELINE_KNEE; the 20 levels above it, where
    those frames go flat white at 255, hold what the 16 bits have beyond.  Each channel is rolled off by itself, so
    a highlight goes to white as it does in the pipeline's frames (rolled off together the channels keep the hue of
    the white balance: the deck and the sky near the sun came out blue - Van Zyl, 14 Nov)."""
    x = np.asarray(x, np.float32)
    hi = 65535.0 / UNITS_PER_8BIT
    t = np.clip((x - PIPELINE_KNEE) / (hi - PIPELINE_KNEE), 0.0, 1.0)
    k = (hi - PIPELINE_KNEE) / (255.0 - PIPELINE_KNEE)                # slope 1 at the knee, 0 at the 16-bit ceiling
    y = PIPELINE_KNEE + (255.0 - PIPELINE_KNEE) * (1.0 - (1.0 - t) ** k)
    return np.clip(np.where(x > PIPELINE_KNEE, y, x), 0.0, 255.0)


TERRAIN_GREY = 100.0             # the median of the ground lands here (8-bit), as in the 8-bit pipeline's frames
KNEE = 170.0                     # linear up to here; above it the highlights roll off towards 255 ...
DESATURATE = 0.85                # ... and lose this much of their colour by 3 shoulder widths above the knee


def auto_levels(x: np.ndarray, seen: Optional[np.ndarray] = None, low_pct: float = 0.5) -> Tuple[float, float]:
    """(black point, the ground's median) of a linear panorama, in its own units: the ``low_pct`` percentile of
    the darkest channel, and the median brightness (mean of the channels) of the lower half - below the horizon,
    mostly terrain - over the pixels a frame coloured (``seen``).  The exposure for viewing is set on the ground,
    not on the brightest things (10 Nov, Van Zyl: with the white point on the 99.7 % of the panorama the sunlit
    white deck took the range and the terrain came out at a fifth of it)."""
    x = np.asarray(x)
    H = x.shape[0]
    step = max(1, int(np.sqrt(x.shape[0] * x.shape[1] / 2e6)))
    sub = x[::step, ::step]
    ok = sub.max(axis=2) > 0 if seen is None else np.asarray(seen)[::step, ::step]
    if not ok.any():
        return 0.0, TERRAIN_GREY
    lo = float(np.percentile(sub[ok].min(axis=1), low_pct))
    low_half = np.zeros(ok.shape, bool)
    low_half[sub.shape[0] // 2:] = True
    g = ok & low_half
    med = float(np.median(sub[g if g.sum() > 100 else ok].mean(axis=1)))
    lo = min(lo, 0.5 * med)
    return lo, max(med, lo + 1e-3)


def tone_levels(x: np.ndarray, lo: float, mid: float) -> np.ndarray:
    """Linear values -> 0..255 for viewing (Christian, 10 Nov: "an 8bit version that has the high and low points
    rescaled optimally to show detail"): ``lo`` -> 0 and the ground's median ``mid`` -> TERRAIN_GREY, linear up to
    KNEE; above it the brightest channel rolls off towards 255 (never clipped) and the colour fades towards neutral,
    as an overexposed white does."""
    x = np.asarray(x, np.float32)
    k = TERRAIN_GREY / max(mid - lo, 1e-6)
    z = np.clip((x - lo) * k, 0.0, None)
    m = z.max(axis=-1)
    span = 255.0 - KNEE
    over = np.clip(m - KNEE, 0.0, None)
    y = KNEE + span * (1.0 - np.exp(-over / span))                  # slope 1 at the knee, 255 at infinity
    s_ = np.where(m > KNEE, y / np.maximum(m, 1e-6), 1.0)
    out = z * s_[..., None]
    w = (DESATURATE * np.clip(over / (3.0 * span), 0.0, 1.0))[..., None]
    out = (1.0 - w) * out + w * np.minimum(y, 255.0)[..., None]
    return np.clip(out, 0.0, 255.0)


def shared_levels(path: Optional[PathLike], compute, scale: float = 1.0) -> Tuple[Tuple[float, float], str]:
    """0.97.0 (Christian, 12 Nov 2026: "the scaling from 16 to 8 bits should be identical for the with and without
    rover versions"): the (black point, ground level) of the 8-bit view from ``path`` when that file is there, else
    from ``compute()`` - and then written to ``path``.  The file keeps them in 8-bit linear units (the 16-bit frame
    scale / 64); ``scale``: the caller's units per 8-bit unit (64 x exposure16 for the 16-bit panorama files).
    Returns (levels in the caller's units, "file" | "computed")."""
    import json as _json
    f = Path(path) if path else None
    if f is not None and f.is_file():
        d = _json.loads(f.read_text(encoding="utf-8"))
        if "levels_8bit_linear_units" in d:
            lo, mid = (float(v) for v in d["levels_8bit_linear_units"])
            return (lo * scale, mid * scale), "file"
    lo, mid = compute()
    if f is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        try:
            keep = _json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        except ValueError:
            keep = {}
        f.write_text(_json.dumps({**keep, "levels_8bit_linear_units": [lo / scale, mid / scale],
                                  "meaning": "black point and the ground's median of the linear panorama, in the "
                                             "frames' 16-bit scale / 64; view = tone_levels(linear, black, ground): "
                                             f"ground -> {TERRAIN_GREY:g} of 255, linear to {KNEE:g}, then rolled off",
                                  "note": "delete this file to have the levels found again on the next panorama"},
                                 indent=1), encoding="utf-8")
    return (float(lo), float(mid)), "computed"


# ------------------------------------------------------------------ 0.97.0 (12 Nov): exposure and enhanced views
# Christian, 12 Nov 2026: "the 16-bit images are little too dark. could we brighten them without saturating on the
# brightest parts? When generating the 8-bit version ... try a gamma stretch and min max re normalization to fit as
# much detail as possible in the 8-bits ... make sure it is roughly the same process from site to site."
EXPOSURE16_TOP = 0.95            # auto exposure: the EXPOSURE16_PCT percentile of the brightest channel lands here
EXPOSURE16_PCT = 99.99           #   (of 65535) - one linear gain per station, nothing clipped but 1 pixel in 10,000
EXPOSURE16_RANGE = (0.5, 4.0)    #   within these factors of the frames' own 16-bit scale (0.5: the value before)
# 12 Nov, third rule (Christian: "too bland ... the sky and rover deck can saturate at the expense of maximizing
# contrast on the terrain"; then "maybe gamma correction is not the right move. Maybe we need to rescale the high and
# low points of the stretch before gamma"; chosen from six treatments of the Van Zyl panorama side by side):
#   1. the display gamma first (ENH_GAMMA), on the linear values over the terrain's ENH_REF_PCT point;
#   2. black and white at ENH_LOW_PCT / ENH_HIGH_PCT of the terrain band in that encoded space - the band of the
#      panorama ENH_BAND_DEG below the horizon (past the deck, short of the sky); what is brighter clips;
#   3. an S-curve of steepness ENH_S about the band's median.
# The local version adds an unsharp mask on the lightness (radius ENH_LOCAL_SIGMA_FRAC of the width, amount
# ENH_LOCAL_AMOUNT): a fixed linear filter, so both eyes get the same processing.
ENH_BAND_DEG = (-30.0, -2.0)
ENH_GAMMA = 2.2
ENH_REF_PCT = 99.9
ENH_LOW_PCT, ENH_HIGH_PCT = 1.0, 97.0      # 99 until 0.97.24: with the rover shown the band's top 3 % is deck, not terrain
ENH_S = 5.5                      # 6.0 until 0.97.24
# 0.98.0 (Christian, 14 Nov 2026: "for the enhanced color, make sure that the bright rover and sky is compressed while
# bringing out detail from the shadows and the terrain. The shadows should not be clipped to black"): nothing clips.
# The terrain band's ENH_LOW_PCT..ENH_HIGH_PCT span takes ENH_OUT_LOW..ENH_OUT_HIGH of the 8 bits through the S-curve;
# what is darker runs down a toe to black at zero radiance only; what is brighter (the deck, the sky) rolls off
# towards white along a shoulder.  Both join the S-curve with its own slope.
ENH_OUT_LOW = 0.06
ENH_OUT_HIGH = 0.88
ENH_LOCAL_SIGMA_FRAC = 0.0025
ENH_LOCAL_AMOUNT = 0.8


def auto_exposure16(x: np.ndarray, seen: Optional[np.ndarray] = None) -> float:
    """The exposure16 (factor of the frames' 16-bit scale) for a linear panorama ``x`` in 8-bit linear units."""
    x = np.asarray(x)
    step = max(1, int(np.sqrt(x.shape[0] * x.shape[1] / 4e6)))
    sub = x[::step, ::step]
    ok = sub.max(axis=2) > 0 if seen is None else np.asarray(seen)[::step, ::step]
    if not ok.any():
        return EXPOSURE16_RANGE[0]
    top = float(np.percentile(sub[ok].max(axis=1), EXPOSURE16_PCT))
    e = EXPOSURE16_TOP * 65535.0 / max(top * UNITS_PER_8BIT, 1e-6)
    return float(np.clip(e, *EXPOSURE16_RANGE))


def shared_value(path: Optional[PathLike], key: str, compute):
    """``key`` of the station's shared levels file when it is there (the with- and without-rover panoramas use one
    value), else ``compute()`` - then added to the file.  Returns (value, "file" | "computed")."""
    import json as _json
    f = Path(path) if path else None
    d: Dict[str, Any] = {}
    if f is not None and f.is_file():
        try:
            d = _json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            d = {}
        if key in d:
            return d[key], "file"
    v = compute()
    if f is not None:
        d[key] = v
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(_json.dumps(d, indent=1), encoding="utf-8")
    return v, "computed"


def _terrain_band(x: np.ndarray, seen: Optional[np.ndarray] = None) -> np.ndarray:
    """The seen pixels (n, 3) of the terrain band of an equirectangular panorama, subsampled."""
    x = np.asarray(x)
    H = x.shape[0]
    el = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    rows = np.flatnonzero((el >= ENH_BAND_DEG[0]) & (el <= ENH_BAND_DEG[1]))
    if not len(rows):
        rows = np.arange(H // 2, H)
    step = max(1, int(np.sqrt(len(rows) * x.shape[1] / 2e6)))
    sub = x[rows[::step]][:, ::step].astype(np.float32)
    ok = sub.max(axis=2) > 0 if seen is None else np.asarray(seen)[rows[::step]][:, ::step]
    return sub[ok]


def enhanced_levels(x: np.ndarray, seen: Optional[np.ndarray] = None) -> List[float]:
    """[reference, black, white, median] of the enhanced view of a linear equirectangular panorama: the reference
    in the panorama's own units (the terrain band's ENH_REF_PCT point of the brightest channel), the others in the
    gamma-encoded space (x / reference) ** (1 / ENH_GAMMA)."""
    t = _terrain_band(x, seen)
    if not len(t):
        return [255.0, 0.0, 1.0, 0.5]
    ref = max(float(np.percentile(t.max(axis=1), ENH_REF_PCT)), 1e-6)
    g = np.clip(t / ref, 0.0, None) ** (1.0 / ENH_GAMMA)
    lo = float(np.percentile(g.min(axis=1), ENH_LOW_PCT))
    hi = max(float(np.percentile(g.max(axis=1), ENH_HIGH_PCT)), lo + 1e-3)
    med = float(np.median(np.clip((g.mean(axis=1) - lo) / (hi - lo), 0.0, 1.0)))
    return [ref, lo, hi, med]


def tone_enhanced(x: np.ndarray, ref: float, lo: float, hi: float, med: float) -> np.ndarray:
    """Linear values -> 0..255 by the rule above: gamma; the terrain's span ``lo``..``hi`` through the S-curve about
    ``med`` into ENH_OUT_LOW..ENH_OUT_HIGH of the range; a toe below it (black only at zero) and a shoulder above
    it (white only at infinity).  One curve for the three channels; monotonic, nothing clipped (0.98.0; until
    0.97.24 the span took the whole range and what lay outside it was black or white)."""
    g = np.clip(np.asarray(x, np.float32) / ref, 0.0, None) ** (1.0 / ENH_GAMMA)
    span = max(hi - lo, 1e-6)
    n = (g - lo) / span
    f = lambda t: 1.0 / (1.0 + np.exp(-ENH_S * (t - med)))            # noqa: E731
    f0, f1 = float(f(0.0)), float(f(1.0))
    A, B = ENH_OUT_LOW, ENH_OUT_HIGH
    core = A + (B - A) * (f(np.clip(n, 0.0, 1.0)) - f0) / (f1 - f0)
    s0 = (B - A) * ENH_S * f0 * (1.0 - f0) / (f1 - f0)               # the S-curve's slope at its ends
    s1 = (B - A) * ENH_S * f1 * (1.0 - f1) / (f1 - f0)
    n0 = -lo / span                                                   # where zero radiance is (<= 0)
    out = core
    if n0 < -1e-6:
        p = float(np.clip(s0 * (-n0) / A, 0.4, 4.0))                  # the toe: A ((n - n0) / -n0) ** p, slope s0 at n = 0
        toe = A * np.clip((n - n0) / (-n0), 0.0, 1.0) ** p
        out = np.where(n < 0.0, toe, out)
    sh = B + (1.0 - B) * (1.0 - np.exp(-s1 * np.clip(n - 1.0, 0.0, None) / (1.0 - B)))
    out = np.where(n > 1.0, sh, out)
    return np.clip(255.0 * out, 0.0, 255.0)


def local_contrast(bgr8: np.ndarray, sky_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """An unsharp mask on the lightness of an 8-bit BGR equirectangular image (the colour kept), continuous across
    the 360-degree seam.  The same filter for any image: both eyes are treated alike.  ``sky_weight`` (0.97.13; (H,
    W) 0-1): no sharpening where it is 1 - in the sky there is no detail to bring out, only the steps between the
    8-bit levels and the frames' seams."""
    import cv2
    W = bgr8.shape[1]
    sigma = max(1.0, ENH_LOCAL_SIGMA_FRAC * W)
    pad = int(4 * sigma) + 1
    lab = cv2.cvtColor(np.concatenate([bgr8[:, -pad:], bgr8, bgr8[:, :pad]], axis=1), cv2.COLOR_BGR2LAB).astype(np.float32)
    Lc = lab[..., 0]
    amount = ENH_LOCAL_AMOUNT
    if sky_weight is not None:
        sw = np.asarray(sky_weight, np.float32)
        amount = ENH_LOCAL_AMOUNT * (1.0 - np.concatenate([sw[:, -pad:], sw, sw[:, :pad]], axis=1))
    lab[..., 0] = np.clip(Lc + amount * (Lc - cv2.GaussianBlur(Lc, (0, 0), sigma)), 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)[:, pad:pad + W]


ENH_SKY_SIGMA_DEG = 3.0          # 0.97.13: the sky of the enhanced views is the sky blurred by this (Gaussian sigma)
ENH_SKY_WIDTH = 2048             # ... computed on a panorama of this width
ENH_SKY_DITHER = 0.5             # ... with this much noise (8-bit levels, uniform +/-) so that 8 bits show no contours


def smooth_sky(x: np.ndarray, sky_weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The linear panorama ``x`` (H, W, 3 float32; changed in place) with its sky replaced by the sky blurred over
    ENH_SKY_SIGMA_DEG (0.97.13; Christian, 13 Nov 2026: "the enhanced product should smooth the sky to avoid
    artifacts"): the enhanced views stretch the terrain's contrast, and with it the sky's - the frames' edges and the
    steps of the sky fill, invisible in the plain view, show.  ``sky_weight``: the small map the render wrote (1 =
    sky well above the skyline; it fades to 0 at the skyline, so hills and horizon stay as they are).  The blur
    averages sky pixels only (weighted by the map) and wraps around the panorama.  Returns (x, the weight at full
    size)."""
    import cv2
    H, W = x.shape[:2]
    w = min(ENH_SKY_WIDTH, W)
    h = max(2, int(round(H * w / W)))
    sw = cv2.resize(np.asarray(sky_weight, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if float(sw.max()) <= 0:
        return x, np.zeros((H, W), np.float32)
    small = cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
    sg = max(1.0, ENH_SKY_SIGMA_DEG / 360.0 * w)
    pad = int(4 * sg) + 1
    wrap = lambda a: np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)                       # noqa: E731
    blur = lambda a: cv2.GaussianBlur(wrap(a), (0, 0), sg, borderType=cv2.BORDER_REPLICATE)[:, pad:pad + w]   # noqa: E731
    den = np.maximum(blur(sw), 1e-4)
    sm = np.dstack([blur(small[..., k] * sw) / den for k in range(3)])
    full_w = np.empty((H, W), np.float32)
    for r0 in range(0, H, 1024):                                           # in strips: no full-size float copies
        r1 = min(H, r0 + 1024)
        f = H / h
        s0, s1 = max(0, int(r0 / f) - 2), min(h, int(np.ceil(r1 / f)) + 2)
        n = max(1, int(round((s1 - s0) * f)))
        off = min(max(0, int(round(r0 - s0 * f))), max(0, n - (r1 - r0)))
        wt = np.clip(cv2.resize(sw[s0:s1], (W, n), interpolation=cv2.INTER_LINEAR)[off:off + (r1 - r0)], 0.0, 1.0)
        full_w[r0:r1] = wt
        if wt.max() <= 0:
            continue
        up = cv2.resize(sm[s0:s1], (W, n), interpolation=cv2.INTER_CUBIC)[off:off + (r1 - r0)]
        x[r0:r1] = x[r0:r1] * (1.0 - wt[..., None]) + up * wt[..., None]
    return x, full_w


def write_enhanced(out_dir: PathLike, prefix: str, top: str = "left", jpeg_quality: int = 95,
                   levels_file: Optional[PathLike] = None) -> Dict[str, Any]:
    """The contrast-enhanced secondary products from the 16-bit eye panoramas of ``out_dir`` (no rendering):
      <prefix>_enh_360_3DTD.jpg        gamma, the terrain's stretch, an S-curve (:func:`tone_enhanced`)
      <prefix>_enhlocal_360_3DTD.jpg   the same, then local contrast (:func:`local_contrast`)
    Levels from the left eye (or the station's levels file), used for both eyes."""
    import cv2
    from .ods import STACK_NAME
    out = Path(out_dir)
    pre = f"{prefix}_" if prefix else ""
    k16 = UNITS_PER_8BIT * _exposure16(out)
    lv, eyes, weights = None, {}, {}
    from .ods import SKY_WEIGHT_NAME
    sky_w = cv2.imread(str(out / f"{pre}{SKY_WEIGHT_NAME}"), cv2.IMREAD_GRAYSCALE)      # None: a folder rendered before 0.97.13
    if sky_w is not None:
        sky_w = sky_w.astype(np.float32) / 255.0
    for eye in ("left", "right"):
        f = out / f"{pre}ods_{eye}_16.png"
        a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if a is None or a.dtype != np.uint16:
            raise FileNotFoundError(f"{f}: no 16-bit panorama")
        a = a[..., :3].astype(np.float32) / k16                      # 8-bit linear units, BGR
        if lv is None:
            lv, _ = shared_value(levels_file, "enhanced_gamma_stretch_s_levels", lambda: enhanced_levels(a))
        if sky_w is not None:                                        # 0.97.13: the sky smoothed, in linear units
            a, weights[eye] = smooth_sky(a, sky_w)
        t = tone_enhanced(a, *lv)
        if sky_w is not None:                                        # the same noise in both eyes: no depth of its own
            noise = np.random.default_rng(20261113).uniform(-ENH_SKY_DITHER, ENH_SKY_DITHER, t.shape[:2]).astype(np.float32)
            t = np.clip(t + (noise * weights[eye])[..., None], 0.0, 255.0)
        eyes[eye] = np.rint(t).astype(np.uint8)
        del a, t
    other = "right" if top == "left" else "left"
    files = {}
    for tag, fn in (("enh", lambda v, e: v), ("enhlocal", lambda v, e: local_contrast(v, weights.get(e)))):
        sq = np.vstack([fn(eyes[top], top), fn(eyes[other], other)])
        files[tag] = str(out / f"{pre}{tag}_{STACK_NAME}.jpg")
        cv2.imwrite(files[tag], sq, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    return {"levels": {"reference_8bit_linear_units": lv[0], "black": lv[1], "white": lv[2], "median": lv[3]}, "files": files,
            "sky": (f"smoothed: Gaussian sigma {ENH_SKY_SIGMA_DEG} deg over the sky above the skyline, no local contrast "
                    f"there, +/- {ENH_SKY_DITHER} level of noise" if sky_w is not None else "as rendered (no sky weight file)"),
            "rule": f"gamma {ENH_GAMMA} on linear / reference (the terrain band's {ENH_REF_PCT} % point; band: elevation "
                    f"{ENH_BAND_DEG} deg); the band's {ENH_LOW_PCT} to {ENH_HIGH_PCT} % span in that space takes "
                    f"{ENH_OUT_LOW:g} to {ENH_OUT_HIGH:g} of the 8 bits through an S-curve {ENH_S} about its median; darker "
                    f"values run down a toe (black at zero only), brighter ones (the deck, the sky) up a shoulder (white at "
                    f"infinity only): nothing clips; enhlocal: unsharp mask on "
                    f"the lightness, sigma {ENH_LOCAL_SIGMA_FRAC} x width, amount {ENH_LOCAL_AMOUNT}"}


def _exposure16(out_dir: PathLike) -> float:
    """The exposure16 the 16-bit panoramas of ``out_dir`` were written with (``ods.json``; 0.5 when not recorded)."""
    import json as _json
    try:
        return float((_json.loads((Path(out_dir) / "ods.json").read_text(encoding="utf-8")).get("linear16") or {})
                     .get("exposure16") or 0.5)
    except (OSError, ValueError):
        return 0.5


def retone(out_dir: PathLike, prefix: str, top: str = "left", jpeg_quality: int = 95,
           levels_file: Optional[PathLike] = None) -> Dict[str, Any]:
    """The 8-bit panoramas and the square again from the 16-bit ones in ``out_dir`` (no rendering): the levels from
    the left eye, used for both."""
    import cv2
    out = Path(out_dir)
    pre = f"{prefix}_" if prefix else ""
    levels, imgs = None, {}
    for eye in ("left", "right"):
        f = out / f"{pre}ods_{eye}_16.png"
        a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if a is None or a.dtype != np.uint16:
            raise FileNotFoundError(f"{f}: no 16-bit panorama")
        a = a[..., :3].astype(np.float32)
        if levels is None:
            k16 = 64.0 * _exposure16(out)
            levels = (0.0, 0.0)
        v = np.rint(tone_pipeline(a / k16)).astype(np.uint8)        # 0.98.0: the pipeline's 8-bit scale (auto levels before)
        cv2.imwrite(str(out / f"{pre}ods_{eye}.png"), v)
        imgs[eye] = v
        del a
    other = "right" if top == "left" else "left"
    sq = np.vstack([imgs[top], imgs[other]])
    from .ods import STACK_NAME, drop_old_stacks
    cv2.imwrite(str(out / f"{pre}{STACK_NAME}.png"), sq)
    cv2.imwrite(str(out / f"{pre}{STACK_NAME}.jpg"), sq, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    drop_old_stacks(out, pre[:-1] if pre.endswith("_") else pre)
    enh = write_enhanced(out, prefix, top=top, jpeg_quality=jpeg_quality, levels_file=levels_file)
    try:                                                                    # 0.97.4: the new files carry the XMP again
        from .xmp import tag_folder
        tag_folder(out)
    except Exception:                                                       # noqa: BLE001
        pass
    return {"levels_16bit_units": [float(v) for v in levels], "square": str(out / f"{pre}{STACK_NAME}.jpg"), "top": top,
            "enhanced": enh}
