"""
rpp.mask_levels - the mask from the mask model's probability, without torch (0.99.9).

Christian, 16 Nov 2026: "Remove the dilation morphology. Use connected component to remove small islands of pixels"
and "Write out the probability map as well as the binary mask. SfM wants a generous margin, while the dense, mesh and
panorama steps want a tight edge. With the probability on disk, each step can apply its own threshold without rerunning
the network."

Processing writes ``processed/masks_prob/<stem>.png`` beside the binary ``masks/<stem>.png``: P(terrain) x 255,
8-bit, on the same padded frame as the image, 0 where there is no data or a static mask excludes.  A block made with a
mask model whose card has ``step_thresholds`` ({"sfm": 0.9, "tight": 0.5}) gets

    <project>/masks/<image>.png        P > the "sfm" threshold: feature extraction and matching (a generous margin)
    <project>/masks_tight/<image>.png  P > the "tight" threshold: dense fusion, the mesh and the panoramas
    <project>/masks_prob/<image>.png   the probability, for any other threshold (:func:`write_step_masks`)

Cards without them (rpp_mask_v3 and earlier) keep the binary mask for every step, as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
PROB_DIR, TIGHT_DIR = "masks_prob", "masks_tight"
PROB_SCALE = 255.0
STEP_KEYS = ("sfm", "tight")   # sfm: feature extraction and matching; tight: dense fusion, mesh and panorama


def remove_small_islands(include: np.ndarray, min_px: int) -> np.ndarray:
    """0.99.9 (Christian, 16 Nov 2026: "Remove the dilation morphology. Use connected component to remove small
    islands of pixels"): a bool include map with every 8-connected island of fewer than ``min_px`` pixels flipped -
    small terrain specks inside the rover and small rover specks inside the terrain (``min_px`` <= 0: unchanged)."""
    inc = np.asarray(include, bool)
    if min_px <= 0 or inc.size == 0:
        return inc
    out = inc.copy()
    for val in (True, False):
        src = (out == val).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        if n <= 1:
            continue
        small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_px) + 1
        if small.size:
            out[np.isin(lab, small)] = not val
    return out


def island_px(shape, min_island_frac: float) -> int:
    """The smallest island kept, in pixels, for an image of ``shape``: ``min_island_frac`` of its area."""
    return int(round(float(min_island_frac or 0.0) * int(shape[0]) * int(shape[1])))


def probability_to_mask(prob: np.ndarray, threshold: float, dilate_kernel: int = 0,
                        min_island_frac: float = 0.0) -> np.ndarray:
    """uint8 mask, 255 = include (terrain), 0 = exclude: ``prob > threshold``; islands smaller than
    ``min_island_frac`` of the frame removed (:func:`remove_small_islands`).  ``dilate_kernel`` > 1 dilates the include
    region (the masks of rpp_mask_v3 and earlier; 0.99.9: new cards set 0)."""
    m = remove_small_islands(prob > threshold, island_px(prob.shape, min_island_frac)).astype(np.uint8)
    if dilate_kernel and dilate_kernel > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel,) * 2))
    return m * np.uint8(255)


def prob_to_uint8(prob: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """The probability as stored: round(P x 255), uint8; 0 where ``valid`` is 0."""
    p8 = np.clip(np.rint(np.asarray(prob, np.float32) * PROB_SCALE), 0, 255).astype(np.uint8)
    if valid is not None:
        p8[np.asarray(valid) == 0] = 0
    return p8


def step_thresholds(card: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, float]]:
    """{"sfm": t, "tight": t} from the masking config (``override``) or the model card; None when neither gives one
    (rpp_mask_v3 and earlier: every step uses the binary mask, as before)."""
    src = override if override else (card or {}).get("step_thresholds")
    if not src:
        return None
    out = {k: float(src[k]) for k in STEP_KEYS if src.get(k) is not None}
    return out or None


def mask_from_probability(prob8: np.ndarray, threshold: float, min_island_frac: float = 0.0) -> np.ndarray:
    """A step's mask (uint8 0/255) from a stored probability map: ``P > threshold``, then the small islands removed;
    no dilation.  Pixels stored as 0 (no data, static exclusions) stay out at any threshold >= 0."""
    p = np.asarray(prob8, np.float32) / PROB_SCALE
    return probability_to_mask(p, float(threshold), 0, float(min_island_frac or 0.0))


def step_record(card: Optional[Dict[str, Any]], masking: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """What a processed image records for the steps' masks: {"thresholds": {...}, "min_island_frac": f}, or None."""
    m = masking or {}
    thr = step_thresholds(card, m.get("step_thresholds"))
    if not thr:
        return None
    mi = m.get("min_island_frac")
    return {"thresholds": thr, "min_island_frac": float((card or {}).get("min_island_frac", 0.0) if mi is None else mi)}


def write_step_masks(root: PathLike, names: Iterable[str], thresholds: Dict[str, float], min_island_frac: float = 0.0,
                     which: Iterable[str] = STEP_KEYS, record: bool = True) -> Dict[str, Any]:
    """Make the steps' masks of a project from its ``masks_prob/``: ``masks/<name>.png`` at the "sfm" threshold and
    ``masks_tight/<name>.png`` at the "tight" one (``which`` picks; a step without a threshold is left alone).
    A file whose content would not change is not written again (the feature database dates its masks), and a file
    that is a link (to ``processed/masks``) is replaced, never written through.  Rewriting ``masks/`` after the
    features were extracted takes effect when the block is aligned again.  ``record``: the thresholds into
    ``masks_prob/step_masks.json``.  Returns {"written": {step: n}, "kept": {step: n}, "missing": [names]}."""
    root = Path(root)
    which = tuple(which)
    rep: Dict[str, Any] = {"written": {k: 0 for k in which}, "kept": {k: 0 for k in which}, "missing": []}
    for n in names:
        p8 = cv2.imread(str(root / PROB_DIR / (n + ".png")), cv2.IMREAD_GRAYSCALE)
        if p8 is None:
            rep["missing"].append(n)
            continue
        for k in which:
            if thresholds.get(k) is None:
                continue
            d = root / ("masks" if k == "sfm" else TIGHT_DIR)
            d.mkdir(parents=True, exist_ok=True)
            f = d / (n + ".png")
            new = mask_from_probability(p8, thresholds[k], min_island_frac)
            if f.is_file() and not f.is_symlink():
                old = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if old is not None and old.shape == new.shape and np.array_equal(old, new) and f.stat().st_nlink == 1:
                    rep["kept"][k] += 1
                    continue
            if f.is_symlink() or f.exists():
                f.unlink()                                       # a link to processed/masks: never write through it
            if not cv2.imwrite(str(f), new):
                raise IOError(f)
            rep["written"][k] += 1
    if record:
        (root / PROB_DIR).mkdir(parents=True, exist_ok=True)
        (root / PROB_DIR / "step_masks.json").write_text(json.dumps({"thresholds": thresholds, "min_island_frac": min_island_frac,
                                                                     "written": rep["written"]}, indent=1), encoding="utf-8")
    return rep


def tight_mask_file(root: PathLike, name: str, fallback: Optional[PathLike] = None) -> Optional[Path]:
    """0.99.9: the mask a dense step takes (dense fusion, the mesh, the panoramas, Metashape's depth maps, the
    Nerfstudio frames): ``masks_tight/<name>.png`` where the block has one, else ``masks/<name>.png`` (``fallback``:
    another folder than ``<root>/masks``).  None if neither is there."""
    for f in (Path(root) / TIGHT_DIR / (name + ".png"),
              (Path(fallback) if fallback is not None else Path(root) / "masks") / (name + ".png")):
        if f.is_file():
            return f
    return None
