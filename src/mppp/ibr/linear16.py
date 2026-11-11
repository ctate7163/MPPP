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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return load_config(c[-1])
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


def auto_levels(x: np.ndarray, seen: Optional[np.ndarray] = None, low_pct: float = 0.5, high_pct: float = 99.7) -> Tuple[float, float]:
    """The black and white points (8-bit units) of a linear panorama for viewing: the ``low_pct`` percentile of the
    darkest channel and the ``high_pct`` percentile of the brightest, over the pixels a frame coloured (``seen``)."""
    x = np.asarray(x)
    step = max(1, int(np.sqrt(x.shape[0] * x.shape[1] / 2e6)))
    sub = x[::step, ::step].reshape(-1, 3)
    if seen is not None:
        sub = sub[np.asarray(seen)[::step, ::step].ravel()]
    if not len(sub):
        return 0.0, 255.0
    lo = float(np.percentile(sub.min(axis=1), low_pct))
    hi = float(np.percentile(sub.max(axis=1), high_pct))
    lo = min(lo, 0.5 * hi)                                         # the black point never eats the midtones
    return lo, max(hi, lo + 1.0)


LEVELS_WHITE = 235.0             # the white point lands here; what is brighter rolls off to 255


def tone_levels(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear values -> 0..255 for viewing (Christian, 10 Nov: "an 8bit version that has the high and low points
    rescaled optimally to show detail"): ``lo`` -> 0, ``hi`` -> LEVELS_WHITE, linear between (the hue kept), the few
    pixels above ``hi`` rolled off towards 255 instead of clipped."""
    x = np.asarray(x, np.float32)
    k = LEVELS_WHITE / max(hi - lo, 1e-6)
    z = np.clip((x - lo) * k, 0.0, None)
    m = z.max(axis=-1)
    span = 255.0 - LEVELS_WHITE
    y = LEVELS_WHITE + span * (1.0 - np.exp(-(m - LEVELS_WHITE) / span))   # slope 1 at the white point, 255 at infinity
    s_ = np.where(m > LEVELS_WHITE, y / np.maximum(m, 1e-6), 1.0)
    return np.clip(z * s_[..., None], 0.0, 255.0)
