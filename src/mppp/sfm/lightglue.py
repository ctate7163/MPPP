"""
SIFT + LightGlue matching on ``database.db`` - the default matcher of MPPP.

COLMAP's SIFT keypoints and descriptors (``extract_features`` / ``build_database``) are matched with LightGlue
(Lindenberger et al., ICCV 2023; the ``sift`` weights, through ``kornia``), a learned matcher that is far more
robust than the nearest-neighbour ratio test to the illumination changes and the convergence angles of rover
image sets.  The matches go into the database like COLMAP's own and are verified geometrically by COLMAP
(``pycolmap.verify_matches``), so everything downstream (two-view geometries, reconstruction, health) is unchanged.

LightGlue attends over all keypoints of a pair; its cost grows with the product of the keypoint counts, so each
image contributes at most ``max_keypoints`` keypoints (the largest-scale ones, which are the most repeatable
between times of day); the other keypoints stay in the database but get no matches.  Use a GPU for real blocks
(``device="cuda"`` is chosen when available); the CPU path serves tests and small projects.

The weights (45 MB) are downloaded once by kornia into the torch hub cache; ``MPPP_LIGHTGLUE_WEIGHTS`` names a
local ``.pth`` file for offline use.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .project import SfmProject

PathLike = Union[str, Path]
log = logging.getLogger("mppp.sfm")

DEFAULT_MAX_KEYPOINTS = 8192
DEFAULT_FILTER_THRESHOLD = 0.1          # LightGlue's match confidence threshold
PROBE_PAIRS = 20                        # pairs timed before the block's LightGlue time is projected
PROGRESS_PAIRS = 2000                   # a log line every so many pairs (the matcher is otherwise silent for an hour)


class TooSlow(RuntimeError):
    """LightGlue would take longer than the time budget for this block (projected from the first pairs)."""


def keypoint_scale_ori(kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(scale, orientation in radians) of COLMAP keypoints: ``[x, y, scale, ori]`` rows, or the affine form
    ``[x, y, a11, a12, a21, a22]`` with ``a11 = s cos(o), a21 = s sin(o)``."""
    kp = np.asarray(kp, dtype=np.float64)
    if kp.shape[1] >= 6:
        return np.hypot(kp[:, 2], kp[:, 4]), np.arctan2(kp[:, 4], kp[:, 2])
    if kp.shape[1] >= 4:
        return kp[:, 2], kp[:, 3]
    return np.ones(len(kp)), np.zeros(len(kp))


def select_keypoints(kp: np.ndarray, max_keypoints: int) -> np.ndarray:
    """Indices of the keypoints to match: all of them, or the ``max_keypoints`` largest-scale ones (sorted)."""
    n = len(kp)
    if max_keypoints <= 0 or n <= max_keypoints:
        return np.arange(n)
    scale, _ = keypoint_scale_ori(kp)
    keep = np.argsort(-scale, kind="stable")[:max_keypoints]
    return np.sort(keep)


def _descriptors_float(desc: np.ndarray) -> np.ndarray:
    """COLMAP stores SIFT descriptors as uint8 (the unit vector x 512); LightGlue wants float unit vectors."""
    d = np.asarray(desc, dtype=np.float32)
    if desc.dtype == np.uint8:
        d = d / 512.0
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-12)


def load_model(device: str = "cpu", filter_threshold: float = DEFAULT_FILTER_THRESHOLD):
    """The SIFT LightGlue model (kornia), on ``device``."""
    import torch
    from kornia.feature import LightGlue
    weights = os.environ.get("MPPP_LIGHTGLUE_WEIGHTS")
    if weights:
        model = LightGlue(features="sift", weights=None, filter_threshold=filter_threshold)
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state, strict=False)
    else:
        model = LightGlue(features="sift", filter_threshold=filter_threshold)
    return model.eval().to(device)


def pick_device(device: Optional[str] = None) -> str:
    import torch
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class _Features:
    """Per-image tensors for LightGlue, read from the database once."""

    def __init__(self, db, image_id: int, width: int, height: int, max_keypoints: int, device: str):
        import torch
        kp = np.asarray(db.read_keypoints(image_id), dtype=np.float32)
        desc = db.read_descriptors(image_id)
        desc = np.asarray(getattr(desc, "data", desc))                 # pycolmap FeatureDescriptors -> its array
        if len(kp) != len(desc):
            raise ValueError(f"image {image_id}: {len(kp)} keypoints but {len(desc)} descriptors")
        self.index = select_keypoints(kp, max_keypoints)
        kp = kp[self.index]
        scale, ori = keypoint_scale_ori(kp)
        self.n = len(kp)
        self.data = {"keypoints": torch.from_numpy(kp[:, :2].astype(np.float32))[None].to(device),
                     "descriptors": torch.from_numpy(_descriptors_float(desc[self.index]))[None].to(device),
                     "scales": torch.from_numpy(scale.astype(np.float32))[None].to(device),
                     "oris": torch.from_numpy(ori.astype(np.float32))[None].to(device),
                     "image_size": torch.tensor([[float(width), float(height)]], device=device)}


def _image_sizes(db) -> Dict[int, Tuple[int, int]]:
    cams = {int(c.camera_id): (int(c.width), int(c.height)) for c in db.read_all_cameras()}
    return {int(im.image_id): cams[int(im.camera_id)] for im in db.read_all_images()}


def match_pairs_lightglue(project: SfmProject, pairs: Optional[Sequence[Tuple[str, str]]] = None,
                          max_keypoints: int = DEFAULT_MAX_KEYPOINTS, filter_threshold: float = DEFAULT_FILTER_THRESHOLD,
                          device: Optional[str] = None, cache_images: int = 64,
                          max_hours: Optional[float] = None) -> Dict[str, Any]:
    """Match the pairs (image names; all pairs when None) with LightGlue and write the matches into the database.
    Returns {"pairs": n, "matches": total, "pairs_file": path} - the pairs file is what the verification reads.
    ``max_hours``: after :data:`PROBE_PAIRS` pairs the block's time is projected; beyond the budget
    :class:`TooSlow` is raised (the matches written so far stay in the database and are not matched again)."""
    import time
    import pycolmap
    import torch
    device = pick_device(device)
    model = load_model(device, filter_threshold)
    db = pycolmap.Database.open(str(project.database))
    try:
        images = {im.name: int(im.image_id) for im in db.read_all_images()}
        sizes = _image_sizes(db)
        if pairs is None:
            names = sorted(images)
            pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        feats: Dict[int, _Features] = {}
        order: List[int] = []

        def features(iid: int) -> _Features:
            f = feats.get(iid)
            if f is None:
                w, h = sizes[iid]
                f = feats[iid] = _Features(db, iid, w, h, max_keypoints, device)
                order.append(iid)
                if len(order) > cache_images:                       # bounded memory on long pair lists
                    feats.pop(order.pop(0), None)
            return f

        pairs_file = project.root / "pairs_lightglue.txt"
        total = 0
        lines = []
        t0 = time.time()
        with torch.no_grad():
            for k, (a, b) in enumerate(pairs):
                if max_hours and k == PROBE_PAIRS:
                    hours = (time.time() - t0) / PROBE_PAIRS * len(pairs) / 3600.0
                    if hours > max_hours:
                        pairs_file.write_text("".join(lines), encoding="utf-8")
                        raise TooSlow(f"LightGlue on {device}: {1000 * (time.time() - t0) / PROBE_PAIRS:.0f} ms per "
                                      f"pair x {len(pairs)} pairs = {hours:.1f} h > {max_hours:g} h")
                    log.info("LightGlue on %s: %.0f ms per pair x %d pairs, about %.1f h", device,
                             1000 * (time.time() - t0) / PROBE_PAIRS, len(pairs), hours)
                if k and k % PROGRESS_PAIRS == 0:
                    left = (time.time() - t0) / k * (len(pairs) - k) / 60.0
                    log.info("LightGlue: %d of %d pairs, %d matches, about %.0f min left", k, len(pairs), total, left)
                if a not in images or b not in images or a == b:
                    continue
                ia, ib = images[a], images[b]
                fa, fb = features(ia), features(ib)
                if fa.n == 0 or fb.n == 0:
                    continue
                out = model({"image0": fa.data, "image1": fb.data})
                m = out["matches"][0].detach().cpu().numpy()
                if len(m) == 0:
                    continue
                full = np.stack([fa.index[m[:, 0]], fb.index[m[:, 1]]], axis=1).astype(np.uint32)
                if ia > ib:                                          # COLMAP stores pairs with the smaller id first
                    ia, ib, full = ib, ia, full[:, ::-1].copy()
                    a, b = b, a
                db.write_matches(ia, ib, full)
                total += len(full)
                lines.append(f"{a} {b}\n")
        pairs_file.write_text("".join(lines), encoding="utf-8")
    finally:
        db.close()
    log.info("LightGlue: %d pairs matched, %d matches (%s, <= %d keypoints per image)", len(lines), total, device,
             max_keypoints)
    return {"pairs": len(lines), "matches": total, "pairs_file": str(pairs_file), "device": device}


def verify(project: SfmProject, pairs_file: PathLike, max_error_px: float = 6.0, min_num_inliers: int = 15) -> None:
    """COLMAP's geometric verification of the matches of the listed pairs (two-view geometries into the database)."""
    import pycolmap
    vo = pycolmap.TwoViewGeometryOptions()
    vo.ransac.max_error = float(max_error_px)
    vo.min_num_inliers = int(min_num_inliers)
    pycolmap.verify_matches(str(project.database), str(pairs_file), vo)
