"""
Prior-guided image pairs: which images can see the same terrain, judged from
the CAHV/waypoint poses and the initial intrinsics.

Each image's valid pixels (mask, else non-zero pixels) are sampled on a coarse
grid and cast onto a local ground plane ``camera_height_m`` below the camera,
or to ``max_range_m`` along the ray if the plane is farther or not hit.  The
footprint points of image i are projected into image j; the overlap of the pair
is the larger of the two visible fractions.  Cheap (a few seconds for ~500
images) and deliberately generous: it only has to avoid wasting time on pairs
that cannot match, not decide what matches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .project import SfmProject

PathLike = Union[str, Path]


def _valid_grid(project: SfmProject, r: Dict[str, Any], grid: Tuple[int, int]) -> np.ndarray:
    gw, gh = grid
    m = None
    if r.get("has_mask"):
        m = cv2.imread(str(project.masks_dir / (r["name"] + ".png")), cv2.IMREAD_GRAYSCALE)
    if m is None:
        im = cv2.imread(str(project.images_dir / r["name"]), cv2.IMREAD_GRAYSCALE)
        m = (im > 0).astype(np.uint8) * 255
    return cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA) > 127


def footprints(project: SfmProject, grid: Tuple[int, int] = (32, 24), max_range_m: float = 40.0,
               camera_height_m: float = 1.95) -> Dict[str, Dict[str, Any]]:
    """Per image: camera, pose, validity grid and the 3-D footprint points."""
    import pycolmap
    cams = {k: pycolmap.Camera(model=c["model"], width=c["width"], height=c["height"], params=c["params"])
            for k, c in project.cameras.items()}
    gw, gh = grid
    out = {}
    for r in project.images:
        cam = cams[r["instrument"]]
        valid = _valid_grid(project, r, grid)
        us = (np.arange(gw) + 0.5) * cam.width / gw
        vs = (np.arange(gh) + 0.5) * cam.height / gh
        uu, vv = np.meshgrid(us, vs)
        uv = np.stack([uu[valid], vv[valid]], axis=1)
        xy = cam.cam_from_img(uv.astype(np.float64))
        good = np.isfinite(xy).all(axis=1)                        # undistortion can fail in the far corners
        xy = xy[good]
        d_cam = np.hstack([xy, np.ones((len(xy), 1))])
        R = np.asarray(r["prior_R_w2c"], float)
        C = np.asarray(r["prior_C"], float)
        d = d_cam @ R                                             # R^T d  (row vectors)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t = np.full(len(d), max_range_m)
        down = d[:, 2] < -1e-3
        t[down] = np.minimum(max_range_m, -camera_height_m / d[down, 2])
        pts = C + d * t[:, None]
        # the undistorted field of view of this camera (to reject folded projections)
        t_ = np.linspace(0, 1, 41)
        border = np.concatenate([np.stack([t_ * cam.width, np.zeros_like(t_)], 1),
                                 np.stack([t_ * cam.width, np.full_like(t_, cam.height)], 1),
                                 np.stack([np.zeros_like(t_), t_ * cam.height], 1),
                                 np.stack([np.full_like(t_, cam.width), t_ * cam.height], 1)])
        fov = np.nanmax(np.abs(cam.cam_from_img(border)), axis=0) * 1.02
        out[r["name"]] = {"cam": cam, "R": R, "C": C, "valid": valid, "pts": pts, "fov": fov,
                          "frame": r.get("sclk_key"), "station": r["station"]}
    return out


def _visible_fraction(fp_i: Dict[str, Any], fp_j: Dict[str, Any], grid: Tuple[int, int]) -> float:
    P = fp_i["pts"]
    if len(P) == 0:
        return 0.0
    X = (P - fp_j["C"]) @ fp_j["R"].T
    ok = X[:, 2] > 0.05
    xy = np.zeros((len(X), 2))
    xy[ok] = X[ok, :2] / X[ok, 2:3]
    ok &= (np.abs(xy[:, 0]) <= fp_j["fov"][0]) & (np.abs(xy[:, 1]) <= fp_j["fov"][1])
    if not ok.any():
        return 0.0
    cam = fp_j["cam"]
    uv = cam.img_from_cam(np.hstack([xy[ok], np.ones((ok.sum(), 1))]))
    gw, gh = grid
    gx = np.floor(uv[:, 0] / cam.width * gw).astype(int)
    gy = np.floor(uv[:, 1] / cam.height * gh).astype(int)
    inside = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
    hit = np.zeros(inside.shape, bool)
    hit[inside] = fp_j["valid"][gy[inside], gx[inside]]
    return float(hit.sum()) / len(P)


def prior_overlap_pairs(project: SfmProject, min_overlap: float = 0.05, max_range_m: float = 40.0,
                        camera_height_m: float = 1.95, max_distance_m: float = 60.0,
                        grid: Tuple[int, int] = (32, 24), out_file: Optional[PathLike] = None
                        ) -> List[Tuple[str, str, float]]:
    """
    Pairs (name_i, name_j, overlap) with overlap >= ``min_overlap``, plus every
    stereo pair of the same exposure.  Written to ``out_file`` (default
    ``<project>/pairs_prior.txt``, the COLMAP ``match_list`` format) as well.
    """
    fps = footprints(project, grid, max_range_m, camera_height_m)
    names = list(fps)
    C = np.array([fps[n]["C"] for n in names])
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            fa, fb = fps[names[a]], fps[names[b]]
            if fa["frame"] is not None and fa["frame"] == fb["frame"]:
                pairs.append((names[a], names[b], 1.0))
                continue
            if np.linalg.norm(C[a] - C[b]) > max_distance_m:
                continue
            ov = max(_visible_fraction(fa, fb, grid), _visible_fraction(fb, fa, grid))
            if ov >= min_overlap:
                pairs.append((names[a], names[b], round(ov, 4)))
    out = Path(out_file) if out_file else project.root / "pairs_prior.txt"
    out.write_text("".join(f"{a} {b}\n" for a, b, _ in pairs), encoding="utf-8")
    project.settings["pairs_prior"] = {"file": out.name, "n_pairs": len(pairs), "min_overlap": min_overlap,
                                       "max_range_m": max_range_m, "camera_height_m": camera_height_m,
                                       "n_possible": len(names) * (len(names) - 1) // 2}
    project.save()
    return pairs
