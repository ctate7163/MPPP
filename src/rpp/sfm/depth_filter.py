"""
The depth maps filtered before the fusion (1.0, Christian, 20 Nov 2026: "how could we implement these meshing options
without the danger of making anything worse? ... find a way to improve the post depth map filter and fusion").

COLMAP's fusion takes every pixel its patch-match kept and averages the pixels of other views that agree with it.  What
it cannot see is a pixel that is wrong in a way the other views share or do not test:

* **depth edges** - at the outline of a rock the matching window straddles the rock and the ground behind it; the
  pixels there take a depth in between (flying pixels), and the mesh stretches a skin from the rock to the ground;
* **speckles** - small islands of depth cut off from their surroundings by such edges or by holes: sky pixels that
  matched, floaters;
* **back-facing normals** - a normal that faces away from the camera that measured it: the plane fit went wrong;
* optionally, **depth beyond a range** per camera family (``max_range_m``, e.g. a Hazcam far out).

:func:`filter_depth_map` makes the mask of one map, :func:`filter_workspace` writes the filtered depth maps of a
dense workspace into ANOTHER folder: the workspace's own maps are never changed, and the filtered ones are fused and
meshed apart from the station products (:mod:`rpp.sfm.fusion_compare`) until a comparison shows they are not worse.

Thresholds (working pixels of the maps, up to 2000 px; about 0.6 mrad per pixel for a Navcam frame):

* ``edge_rel`` 0.05: a step of more than 5 % of the depth between neighbouring pixels.  Smooth ground seen from the
  2 m mast changes depth by about 0.3 % per pixel at 10 m and 2 % at 75 m (grazing), so the ground is not cut; a rock's
  outline against the ground behind it is a step of tens of percent.  The pixels on both sides of the step are
  dropped, and ``edge_dilate`` more (1 px).
* ``speckle_px`` 64: islands smaller than this (a rock of 20 cm at 30 m is about 11 px across but stands on the ground,
  without a step at its foot, so it is not an island).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, os.PathLike]

FILTER_DEFAULTS: Dict[str, Any] = {"edge_rel": 0.05, "edge_dilate": 1, "speckle_px": 64, "back_facing": True,
                                   "max_range_m": {}}
FILTER_RECORD = "rpp_depth_filter.json"


def camera_rays(width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """(height, width, 3): the ray (x/z, y/z, 1) of every pixel center of a pinhole camera (COLMAP: the pixel (0, 0)
    covers 0..1, so its center is 0.5)."""
    u = (np.arange(width, dtype=np.float32) + 0.5 - cx) / fx
    v = (np.arange(height, dtype=np.float32) + 0.5 - cy) / fy
    R = np.empty((height, width, 3), np.float32)
    R[..., 0] = u[None, :]
    R[..., 1] = v[:, None]
    R[..., 2] = 1.0
    return R


def edge_mask(D: np.ndarray, edge_rel: float) -> np.ndarray:
    """Pixels with a valid 4-neighbour whose depth differs by more than ``edge_rel`` of the nearer of the two."""
    valid = D > 0
    E = np.zeros(D.shape, bool)
    for a, b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                 ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        da, db = D[a], D[b]
        both = valid[a] & valid[b]
        step = both & (np.abs(da - db) > edge_rel * np.minimum(da, db))
        E[a] |= step
        E[b] |= step
    return E


def filter_depth_map(D: np.ndarray, N: Optional[np.ndarray] = None, rays: Optional[np.ndarray] = None,
                     max_range_m: Optional[float] = None, **opts: Any) -> Tuple[np.ndarray, Dict[str, int]]:
    """(the kept pixels (h, w) bool, the counts) of one depth map ``D`` (0: no depth), its normal map ``N`` (h, w, 3,
    in the camera frame) and the pixel ``rays`` (:func:`camera_rays`; needed for the normals and the range)."""
    import cv2
    o = {**FILTER_DEFAULTS, **opts}
    D = np.asarray(D, np.float32)
    keep = D > 0
    rep = {"valid": int(keep.sum())}
    if o["edge_rel"]:
        E = edge_mask(D, float(o["edge_rel"]))
        k = int(o["edge_dilate"] or 0)
        if k > 0:
            E = cv2.dilate(E.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        E &= keep
        rep["edges"] = int(E.sum())
        keep &= ~E
    if o["back_facing"] and N is not None and rays is not None:
        B = keep & (np.einsum("hwc,hwc->hw", np.asarray(N, np.float32), rays) > 0)
        rep["back_facing"] = int(B.sum())
        keep &= ~B
    if max_range_m and rays is not None:
        F = keep & (D * np.linalg.norm(rays, axis=2) > float(max_range_m))
        rep["beyond_range"] = int(F.sum())
        keep &= ~F
    if o["speckle_px"]:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), connectivity=4)
        small = np.zeros(n, bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < int(o["speckle_px"])
        S = small[lab]
        rep["speckles"] = int(S.sum())
        keep &= ~S
    rep["kept"] = int(keep.sum())
    return keep, rep


def _family(name: str) -> str:
    return str(name)[:1].upper()


def filter_workspace(ws: PathLike, out_dir: PathLike, input_type: str = "geometric", workers: int = 0,
                     **opts: Any) -> Dict[str, Any]:
    """The filtered depth maps of the dense workspace ``ws`` (``stereo/depth_maps/<image>.<input_type>.bin``) into
    ``out_dir/stereo/depth_maps`` under the same names; the normal maps are linked (hard links, else copied) - the
    fusion ignores a normal where the depth is 0.  ``ws`` is only read.  ``max_range_m``: {camera family letter
    (N, F, R, Z): metres}.  Returns the record also written to ``out_dir/rpp_depth_filter.json``."""
    import pycolmap
    from concurrent.futures import ThreadPoolExecutor
    from .dense import read_colmap_array
    from ..ibr.photometry import write_colmap_array
    ws, out_dir = Path(ws), Path(out_dir)
    o = {**FILTER_DEFAULTS, **opts}
    rec = pycolmap.Reconstruction(str(ws / "sparse"))
    cams = {im.name: rec.cameras[im.camera_id] for im in rec.images.values()}
    src = ws / "stereo" / "depth_maps"
    names = sorted(n for n in cams if (src / f"{n}.{input_type}.bin").is_file())
    (out_dir / "stereo" / "depth_maps").mkdir(parents=True, exist_ok=True)
    (out_dir / "stereo" / "normal_maps").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(name: str) -> Dict[str, Any]:
        D = read_colmap_array(src / f"{name}.{input_type}.bin")
        nf = ws / "stereo" / "normal_maps" / f"{name}.{input_type}.bin"
        N = read_colmap_array(nf) if nf.is_file() else None
        cam = cams[name]
        h, w = D.shape
        sx, sy = w / float(cam.width), h / float(cam.height)
        p = [float(v) for v in cam.params]
        fx, fy, cx, cy = p[:4] if len(p) >= 4 else (p[0], p[0], p[1], p[2])    # PINHOLE, else SIMPLE_PINHOLE
        rays = camera_rays(w, h, fx * sx, fy * sy, cx * sx, cy * sy)
        keep, r = filter_depth_map(D, N, rays, max_range_m=(o["max_range_m"] or {}).get(_family(name)),
                                   **{k: v for k, v in o.items() if k != "max_range_m"})
        write_colmap_array(out_dir / "stereo" / "depth_maps" / f"{name}.{input_type}.bin", np.where(keep, D, 0))
        if nf.is_file():
            dst = out_dir / "stereo" / "normal_maps" / nf.name
            if dst.exists():
                dst.unlink()
            try:
                os.link(nf, dst)
            except OSError:
                import shutil
                shutil.copyfile(nf, dst)
        return {"image": name, **r}

    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as ex:
        per = list(ex.map(one, names))
    tot = {k: int(sum(p.get(k, 0) for p in per)) for k in ("valid", "edges", "back_facing", "beyond_range", "speckles", "kept")}
    out = {"workspace": str(ws), "input_type": input_type, "options": o, "images": len(per), "totals": tot,
           "kept_fraction": round(tot["kept"] / max(tot["valid"], 1), 4), "seconds": round(time.time() - t0, 1),
           "per_image": per}
    (out_dir / FILTER_RECORD).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out
