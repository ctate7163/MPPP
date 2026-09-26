"""
mppp.colmap — COLMAP text I/O and camera-model math, shared by the whole
package (v0p13: one place for what ``mppp.colmap``, ``mppp.sfm.export`` and
``mppp.error.colmap`` each had a copy of).

* ``write_text_model(metas, out_dir)``: the prior model written by
  ``process_images`` (one camera per intrinsics group, the pose prior of every
  image, empty points, ``rig_config.json`` for the stereo pairs).
* ``write_cameras_txt`` / ``write_images_txt``: the text writers used by it and
  by ``mppp.sfm.export``.
* ``project_camera``: COLMAP projection with distortion (vectorised);
  ``scale_camera_params``: a camera at another resolution.

Conventions: COLMAP poses are world->camera (qvec w,x,y,z + tvec), camera axes
x right / y down / z forward — identical to MPPP's.  COLMAP's pixel origin is
the image corner, so 0.5 px is added to MPPP's principal point.  World = ENU
metres minus ``offset`` (kept small for numerical conditioning).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation

PathLike = Union[str, Path]


def colmap_camera_params(intr_dict: Dict[str, Any]) -> tuple:
    """-> (model_name, params) from an ``Intrinsics.to_dict()``."""
    K = np.asarray(intr_dict["K"], float)
    d = intr_dict["dist_opencv"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5
    if not any(abs(d[k]) > 0 for k in d):
        return "PINHOLE", [fx, fy, cx, cy]
    if abs(d.get("k3", 0.0)) > 0:
        return "FULL_OPENCV", [fx, fy, cx, cy, d["k1"], d["k2"], d["p1"], d["p2"], d["k3"], 0.0, 0.0, 0.0]
    return "OPENCV", [fx, fy, cx, cy, d["k1"], d["k2"], d["p1"], d["p2"]]


# ------------------------------------------------------------ camera math
_ONE_FOCAL = ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE")


def scale_camera_params(model: str, params: Sequence[float], s: float) -> np.ndarray:
    """Parameters of the same camera at ``s`` x the resolution (focal and principal point scale;
    distortion coefficients are resolution-free)."""
    p = np.array(params, float)
    if model in _ONE_FOCAL:
        p[:3] *= s
    else:
        p[:4] *= s
    return p


def project_camera(model: str, p: Sequence[float], x: np.ndarray) -> np.ndarray:
    """
    COLMAP projection of camera-frame point(s) ``x`` (..., 3) WITH lens
    distortion -> pixels (..., 2), exactly as COLMAP does.  Supported:
    SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV;
    anything else raises.  (Before v0p9 the error model ignored distortion,
    wrong by hundreds of pixels for the Navcam FULL_OPENCV model, k1 = -0.27.)
    """
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    u, v = x[..., 0] / x[..., 2], x[..., 1] / x[..., 2]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        k = list(p[3:])
    elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        k = list(p[4:])
    else:
        raise NotImplementedError(f"camera model {model} not supported by mppp.colmap.project_camera")
    r2 = u * u + v * v
    du = dv = 0.0
    if model == "SIMPLE_RADIAL":
        rad = 1 + k[0] * r2
    elif model == "RADIAL":
        rad = 1 + k[0] * r2 + k[1] * r2 * r2
    elif model == "OPENCV":
        k1, k2, p1, p2 = k[:4]
        rad = 1 + k1 * r2 + k2 * r2 * r2
        du, dv = 2 * p1 * u * v + p2 * (r2 + 2 * u * u), p1 * (r2 + 2 * v * v) + 2 * p2 * u * v
    elif model == "FULL_OPENCV":
        k1, k2, p1, p2, k3, k4, k5, k6 = k[:8]
        rad = (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
        du, dv = 2 * p1 * u * v + p2 * (r2 + 2 * u * u), p1 * (r2 + 2 * v * v) + 2 * p2 * u * v
    else:
        rad = 1.0
    ud, vd = u * rad + du, v * rad + dv
    return np.stack([fx * ud + cx, fy * vd + cy], axis=-1)


def unproject_camera(model: str, p: Sequence[float], uv: np.ndarray, iterations: int = 50,
                     tol: float = 1e-12) -> np.ndarray:
    """
    Inverse of :func:`project_camera`: pixels (..., 2) -> normalised camera
    coordinates (..., 2) (x/z, y/z).  Newton iteration with a numerical
    Jacobian from the pinhole start (v0p20; the fixed-point iteration used
    before left up to 0.15 px in the corners of the rational Navcam model).
    Round-trip error ~1e-9 px where the model is invertible; where it is not
    (a polynomial past its turning point) the result is NaN.
    """
    uv = np.asarray(uv, float)
    p = np.asarray(p, float)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    f = np.array([fx, fy])
    shape = uv.shape
    target = uv.reshape(-1, 2)
    xy = (target - np.array([cx, cy])) / f
    one = np.ones((len(xy), 1))
    proj = lambda q: project_camera(model, p, np.concatenate([q, one], axis=1))      # noqa: E731
    h = 1e-7
    for _ in range(int(iterations)):
        pr = proj(xy)
        r = target - pr
        jx = (proj(xy + [h, 0.0]) - pr) / h
        jy = (proj(xy + [0.0, h]) - pr) / h
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        det = np.where(np.abs(det) < 1e-30, np.nan, det)
        dx = (r[:, 0] * jy[:, 1] - jy[:, 0] * r[:, 1]) / det
        dy = (jx[:, 0] * r[:, 1] - r[:, 0] * jx[:, 1]) / det
        xy = xy + np.c_[dx, dy]
        if np.nanmax(np.abs(np.c_[dx, dy]), initial=0.0) < tol:
            break
    bad = ~np.all(np.isfinite(xy), axis=1) | (np.linalg.norm(proj(np.nan_to_num(xy)) - target, axis=1) > 1e-6)
    xy[bad] = np.nan
    return xy.reshape(shape)


# ------------------------------------------------------------- text writers
def write_cameras_txt(path: PathLike, cameras: Sequence[tuple], header: str = "") -> None:
    """``cameras``: (camera_id, model, width, height, params)."""
    lines = ["# Camera list with one line of data per camera:",
             "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" + (f"  ({header})" if header else ""),
             f"# Number of cameras: {len(cameras)}"]
    for cid, model, w, h, params in cameras:
        lines.append(f"{cid} {model} {int(w)} {int(h)} " + " ".join(f"{float(v):.12g}" for v in params))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_txt(path: PathLike, images: Sequence[tuple]) -> None:
    """``images``: (image_id, q_wxyz, t, camera_id, name, observations) with observations a
    sequence of (x, y, point3D_id) (empty for a priors-only model)."""
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)", f"# Number of images: {len(images)}"]
    for iid, q, t, cid, name, obs in images:
        lines.append(f"{iid} {q[0]:.15g} {q[1]:.15g} {q[2]:.15g} {q[3]:.15g} "
                     f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {cid} {name}")
        lines.append(" ".join(f"{x:.4f} {y:.4f} {pid}" for x, y, pid in obs))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text_model(metas: Sequence[Dict[str, Any]], out_dir: PathLike,
                     offset: Optional[np.ndarray] = None, image_ext: str = ".png") -> Dict[str, Any]:
    """
    ``metas``: ``MPPPImage.meta`` dicts.  One COLMAP camera per ``camera_group``;
    if intrinsics differ inside a group (e.g. Mastcam-Z focus), the group median
    is written and the spread is reported in the returned summary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    off = np.zeros(3) if offset is None else np.asarray(offset, float)

    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for m in metas:
        groups.setdefault(m["camera_group"], []).append(m)

    cam_ids, summary = {}, {"cameras": {}, "offset_enu_m": off.tolist(), "n_images": len(metas)}
    cams = []
    for cid, (g, ms) in enumerate(groups.items(), start=1):
        models = {colmap_camera_params(m["intrinsics"])[0] for m in ms}
        sizes = {(m["intrinsics"]["width"], m["intrinsics"]["height"]) for m in ms}
        if len(models) != 1 or len(sizes) != 1:
            raise ValueError(f"camera group {g}: mixed models {models} or sizes {sizes}")
        P = np.array([colmap_camera_params(m["intrinsics"])[1] for m in ms])
        w, h = sizes.pop()
        model = models.pop()
        cam_ids[g] = cid
        cams.append((cid, model, w, h, np.median(P, axis=0)))
        summary["cameras"][g] = {"camera_id": cid, "model": model, "n_images": len(ms),
                                 "focal_spread_px": float(np.ptp(P[:, 0]))}
    write_cameras_txt(out_dir / "cameras.txt", cams)

    ims = []
    for iid, m in enumerate(metas, start=1):
        R = np.asarray(m["pose"]["R_world_to_cam"], float)
        C = np.asarray(m["pose"]["C_enu_m"], float) - off
        q = Rotation.from_matrix(R).as_quat()                   # x, y, z, w
        name = Path(m["source_product"]).stem + image_ext
        ims.append((iid, (q[3], q[0], q[1], q[2]), -R @ C, cam_ids[m["camera_group"]], name, ()))
    write_images_txt(out_dir / "images.txt", ims)
    (out_dir / "points3D.txt").write_text("# 3D point list (empty: priors only)\n", encoding="utf-8")

    # stereo rigs: eyes share everything after the 3-character camera code
    codes = sorted({m["filename"]["camera_code"] for m in metas})
    rigs = []
    for c in codes:
        if c[1] == "L" and (c[0] + "R" + c[2]) in codes:
            rigs.append({"cameras": [{"image_prefix": c, "ref_sensor": True},
                                     {"image_prefix": c[0] + "R" + c[2]}]})
    (out_dir / "rig_config.json").write_text(json.dumps(rigs, indent=2), encoding="utf-8")
    summary["rigs"] = len(rigs)
    return summary
