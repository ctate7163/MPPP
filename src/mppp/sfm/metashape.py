"""
A finished block as an Agisoft Metashape project (0.95.2, Christian, 30 Oct 2026: "Convert the colmap alignment a
metashape project with just cameras, tie points, masks. The frames should be linked to the colmap images folder").

Metashape reads no COLMAP model, so the conversion has two halves:

* here (pycolmap, any Python): :func:`write_metashape_job` writes ``<project>/metashape/`` from the NATIVE-pixel
  model ``gui_native/sparse`` (the cameras fit the image files in ``images/``):

  - ``metashape_job.json`` - one Metashape sensor per COLMAP camera (calibration converted exactly, see
    :func:`metashape_calibration`), each image's pose (world from camera, 4 x 4), its file and mask, the region
    (the tie points' 2-98 % box), the block offset and frame, and a sample of tie-point observations for the check;
  - ``<block>.bundler.out`` and ``<block>.bundler_images.txt`` - COLMAP's tie points (3-D points, colours and every
    observation) in Bundler v0.3, the one tie-point format Metashape imports;
  - ``build_metashape_project.py`` (a copy of :mod:`mppp.sfm.metashape_build`) and ``make_metashape_project.bat``.

* in Metashape Professional (its Python): ``build_metashape_project.py`` creates ``<block>.psx`` beside them: the
  photos linked (not copied) from ``images/``, the Bundler import (tie points, provisional poses), then the sensors
  and poses set from the job and the calibration fixed, the masks from ``masks/``, the region, a local metric
  coordinate system (MPPP's world frame plus the block offset), and a check: every sampled observation reprojected
  by Metashape against COLMAP's (``metashape_check.json``).  No alignment and no optimisation: the geometry stays
  MPPP's.

Camera models (checked against pycolmap's projection to 1e-11 px, ``tests/test_metashape.py``):

========================  ==========================================================================================
COLMAP                    Metashape
========================  ==========================================================================================
THIN_PRISM_FISHEYE        Fisheye: f = fy, b1 = fx - fy, cx = cx - w/2, cy = cy - h/2, K1..K4 = k1, k2, k3, k4,
(Navcam, Hazcam)          P1 = p2, P2 = p1 (Metashape's P1 goes with x, COLMAP's p1 with y); exact when sx1 = sy1 = 0
                          (every MPPP model) - else the nearest calibration is fitted and its error reported
FULL_OPENCV (Mastcam-Z)   Frame: the same f, b1, cx, cy, P1/P2 swap, K1..K3 = k1, k2, k3; exact when k4 = k5 = k6 = 0
                          (the rational terms; every MPPP model) - else fitted
OPENCV, OPENCV_FISHEYE,   exact (Frame / Fisheye)
RADIAL, PINHOLE, ...
========================  ==========================================================================================

Both programs put the pixel origin at the top-left corner of the image (pixel centres at +0.5); the check in
Metashape measures it (a convention off by half a pixel shows as a 0.5 px mean offset).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .project import SfmProject

log = logging.getLogger("mppp.sfm")
PathLike = Union[str, Path]
METASHAPE_DIR = "metashape"
CHECK_SAMPLES = 20000              # tie-point observations reprojected by Metashape in its check
REGION_PERCENTILES = (2.0, 98.0)   # the region: this box of the tie points, widened by REGION_MARGIN
REGION_MARGIN = 0.10
REGION_Z_PERCENTILES = (0.5, 99.5)  # up and down (see _region)
DEFAULT_METASHAPE = r"C:\Program Files\Agisoft\Metashape Pro\metashape.exe"
REFERENCE_SIGMA_M = 1.0            # the prior's position sigma when the project records none
REFERENCE_ROTATION_SIGMA_DEG = 2.0  # the label attitude's sigma in the Reference pane (display only: nothing is optimised)

# COLMAP model -> (Metashape sensor type, {Metashape term: COLMAP parameter name}), the terms not listed are 0
_FISHEYE_MODELS = {"SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE", "FOV"}


# ====================================================================================== the camera models
def metashape_project_px(cal: Dict[str, Any], xyz: np.ndarray) -> np.ndarray:
    """Metashape's projection (manual, appendix "Camera models") of camera-frame points ``xyz`` (N x 3, z > 0) with
    the calibration ``cal`` ({"type": "frame"|"fisheye", "width", "height", "f", "cx", "cy", "b1", "b2", "k1".."k4",
    "p1".."p4"}): pixel coordinates with the origin at the image's top-left corner."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    x0, y0 = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
    if cal["type"] == "fisheye":
        r0 = np.hypot(x0, y0)
        s = np.where(r0 > 1e-12, np.arctan(r0) / np.maximum(r0, 1e-12), 1.0)
        x, y = x0 * s, y0 * s
    else:
        x, y = x0, y0
    g = lambda k: float(cal.get(k, 0.0) or 0.0)                                           # noqa: E731
    r2 = x * x + y * y
    rad = 1 + g("k1") * r2 + g("k2") * r2 ** 2 + g("k3") * r2 ** 3 + g("k4") * r2 ** 4
    t = 1 + g("p3") * r2 + g("p4") * r2 ** 2
    xp = x * rad + (g("p1") * (r2 + 2 * x * x) + 2 * g("p2") * x * y) * t
    yp = y * rad + (g("p2") * (r2 + 2 * y * y) + 2 * g("p1") * x * y) * t
    u = 0.5 * cal["width"] + g("cx") + xp * (g("f") + g("b1")) + yp * g("b2")
    v = 0.5 * cal["height"] + g("cy") + yp * g("f")
    return np.c_[u, v]


def _direct(model: str, p: Sequence[float], w: int, h: int) -> Tuple[Dict[str, Any], List[str]]:
    """The term-by-term conversion; the COLMAP terms it cannot carry (non-zero) are returned."""
    p = [float(v) for v in p]
    typ = "fisheye" if model in _FISHEYE_MODELS else "frame"
    cal: Dict[str, Any] = {"type": typ, "width": int(w), "height": int(h), "b2": 0.0,
                           "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0}
    lost: List[str] = []

    def fc(fx, fy, cx, cy):
        cal.update(f=fy, b1=fx - fy, cx=cx - 0.5 * w, cy=cy - 0.5 * h)

    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fc(p[0], p[0], p[1], p[2])
        if len(p) > 3:
            cal["k1"] = p[3]
        if len(p) > 4:
            cal["k2"] = p[4]
    elif model == "PINHOLE":
        fc(*p[:4])
    elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
        fc(*p[:4])
        if model == "OPENCV_FISHEYE":                         # fx fy cx cy k1 k2 k3 k4
            cal.update(k1=p[4], k2=p[5], k3=p[6], k4=p[7])
        else:                                                 # ... k1 k2 p1 p2 [k3 ...]
            cal.update(k1=p[4], k2=p[5], p1=p[7], p2=p[6])    # Metashape P1 <-> COLMAP p2 (and P2 <-> p1)
            if model == "FULL_OPENCV":                        # ... k3 k4 k5 k6: k4-k6 the rational denominator
                cal["k3"] = p[8]
                lost += [n for n, v in zip(("k4", "k5", "k6"), p[9:12]) if abs(v) > 0]
            elif model == "THIN_PRISM_FISHEYE":               # ... k3 k4 sx1 sy1
                cal.update(k3=p[8], k4=p[9])
                lost += [n for n, v in zip(("sx1", "sy1"), p[10:12]) if abs(v) > 0]
    else:
        raise ValueError(f"COLMAP camera model {model}: no Metashape conversion")
    return cal, lost


def _sample_rays(model: str, p, w: int, h: int, n: int = 40) -> np.ndarray:
    """Camera-frame rays through an n x n grid of the frame (COLMAP's own unprojection)."""
    import pycolmap
    cam = pycolmap.Camera(model=model, width=int(w), height=int(h), params=np.asarray(p, float))
    uu, vv = np.meshgrid(np.linspace(0.5, w - 0.5, n), np.linspace(0.5, h - 0.5, n))
    uv = np.c_[uu.ravel(), vv.ravel()]
    xy = np.asarray(cam.cam_from_img(uv), float)
    ok = np.all(np.isfinite(xy), axis=1)
    return np.c_[xy[ok], np.ones(int(ok.sum()))]


def metashape_calibration(model: str, params: Sequence[float], width: int, height: int,
                          check: bool = True) -> Dict[str, Any]:
    """A COLMAP camera as a Metashape calibration ({"type", "width", "height", "f", "cx", "cy", "b1", "b2",
    "k1".."k4", "p1".."p4"}) with ``"exact"`` (no COLMAP term lost), ``"lost"`` and - with ``check`` - the largest
    difference between the two projections over the frame (``"max_error_px"``).  Terms Metashape has no place for
    (FULL_OPENCV's k4-k6, THIN_PRISM_FISHEYE's sx1, sy1) are absorbed by a least-squares fit of K1..K4, P1, P2, B1,
    B2 on a grid of the frame."""
    cal, lost = _direct(model, params, width, height)
    if lost or check:
        import pycolmap
        cam = pycolmap.Camera(model=model, width=int(width), height=int(height), params=np.asarray(params, float))
        rays = _sample_rays(model, params, width, height)
        uv = np.asarray(cam.img_from_cam(rays), float)
        if lost:
            from scipy.optimize import least_squares
            names = ["f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2"]

            def res(x):
                c = dict(cal, **dict(zip(names, x)))
                return (metashape_project_px(c, rays) - uv).ravel()
            sol = least_squares(res, [float(cal[n]) for n in names], x_scale="jac")
            cal.update(dict(zip(names, map(float, sol.x))))
        err = np.abs(metashape_project_px(cal, rays) - uv)
        cal["max_error_px"] = float(np.nanmax(err)) if err.size else 0.0
    cal["exact"] = not lost
    cal["lost"] = lost
    cal["colmap_model"] = model
    cal["colmap_params"] = [float(v) for v in params]
    return cal


# ====================================================================================== the references
def metashape_ypr(R_w2c: np.ndarray) -> List[float]:
    """Yaw, pitch, roll (deg) of a world (east, north, up) -> camera rotation in Metashape's reference convention -
    MPPP's reference export formula (:meth:`mppp.camera.Pose.metashape_ypr_deg`); Metashape's own angles of the
    refined poses are compared with it in the build's check."""
    from ..camera import Pose
    return [float(v) for v in Pose(np.asarray(R_w2c, float), np.zeros(3), "", "").metashape_ypr_deg()]


def camera_reference(row: Dict[str, Any], offset: np.ndarray, sigma_m: Sequence[float],
                     R_w2c_refined: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """A camera's reference (Christian, 30 Oct: "the camera references are not ported over"): the PDS label prior
    MPPP aligned to - its centre in the chunk's coordinates (``prior_C`` + the block offset) with the prior sigma,
    its attitude as yaw, pitch, roll - and the yaw, pitch, roll of the refined pose (for the check)."""
    C, R = row.get("prior_C"), row.get("prior_R_w2c")
    if C is None:
        return None
    out: Dict[str, Any] = {"location": [float(v) for v in np.asarray(C, float) + offset],
                           "location_accuracy_m": [float(v) for v in sigma_m],
                           "source": row.get("position_source")}
    if R is not None:
        out["R_w2c"] = [[float(v) for v in row] for row in np.asarray(R, float)]
        out["ypr_deg"] = metashape_ypr(np.asarray(R, float))
        out["rotation_accuracy_deg"] = REFERENCE_ROTATION_SIGMA_DEG
    if R_w2c_refined is not None:
        out["ypr_refined_deg"] = metashape_ypr(R_w2c_refined)
    return out


# ====================================================================================== the job
def _block_name(project: SfmProject) -> str:
    root = Path(project.root)
    return root.parent.name if root.name.startswith("colmap") else root.name


def _sensor_labels(rec, project: SfmProject) -> Dict[int, str]:
    """A readable label per camera: the MPPP camera group of its first image and its frame size."""
    group = {r["name"]: (r.get("camera_group") or r.get("instrument") or "") for r in project.images}
    out: Dict[int, str] = {}
    for im in rec.images.values():
        cid = int(im.camera_id)
        if cid not in out:
            cam = rec.cameras[cid]
            out[cid] = f"{group.get(im.name) or 'camera'}_{cam.width}x{cam.height}_c{cid}"
    return out


def write_bundler(rec, path_out: PathLike, path_list: PathLike, order: Sequence[int],
                  max_points: Optional[int] = None, seed: int = 0) -> Dict[str, int]:
    """COLMAP's tie points as Bundler v0.3 (cameras in ``order`` - image ids - with their poses, f = fx and no
    distortion: the job's sensors replace them in Metashape; each 3-D point with its colour and every observation
    in Bundler's image coordinates: origin at the frame centre, y up).  ``max_points``: a random subset of points."""
    pos = {int(iid): k for k, iid in enumerate(order)}
    flip = np.diag([1.0, -1.0, -1.0])
    lines: List[str] = ["# Bundle file v0.3", ""]
    pids = sorted(int(p) for p in rec.points3D)
    if max_points and len(pids) > max_points:
        rng = np.random.default_rng(seed)
        pids = sorted(rng.choice(pids, int(max_points), replace=False).tolist())
    cams = []
    for iid in order:
        im = rec.images[int(iid)]
        cam = rec.cameras[im.camera_id]
        cw = im.cam_from_world()
        R = flip @ np.asarray(cw.rotation.matrix(), float)
        t = flip @ np.asarray(cw.translation, float)
        cams.append(f"{float(cam.params[0]):.10g} 0 0\n" + "\n".join(" ".join(f"{v:.15g}" for v in row) for row in R)
                    + "\n" + " ".join(f"{v:.15g}" for v in t))
    pts, n_obs, n_dup = [], 0, 0
    for pid in pids:
        pt = rec.points3D[pid]
        best: Dict[int, Tuple[float, str]] = {}
        for el in pt.track.elements:
            iid = int(el.image_id)
            if iid not in pos:
                continue
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            xy = np.asarray(im.points2D[int(el.point2D_idx)].xy, float)
            line = f"{pos[iid]} {int(el.point2D_idx)} {xy[0] - 0.5 * cam.width:.4f} {0.5 * cam.height - xy[1]:.4f}"
            if iid in best:
                # 30 Oct (Chal Rocks): COLMAP lets a track see one image twice (5,595 of 383,851 points), Bundler
                # and Metashape cannot - Metashape's import then renumbered its tracks; the observation nearer the
                # point's projection is kept
                n_dup += 1
                Xc = np.asarray(im.cam_from_world() * np.asarray(pt.xyz, float), float).reshape(1, 3)
                uv = np.asarray(cam.img_from_cam(Xc), float).reshape(2)
                err = float(np.hypot(*(uv - xy)))
                if best[iid][0] is None:
                    xy0 = np.array([float(v) for v in best[iid][1].split()[2:4]]) * [1, -1] + [0.5 * cam.width,
                                                                                               0.5 * cam.height]
                    best[iid] = (float(np.hypot(*(uv - xy0))), best[iid][1])
                if err < best[iid][0]:
                    best[iid] = (err, line)
                continue
            best[iid] = (None, line)
        obs = [v[1] for v in best.values()]
        if len(obs) < 2:
            continue
        x, y, z = (float(v) for v in pt.xyz)
        r, g, b = (int(v) for v in pt.color)
        pts.append(f"{x:.10g} {y:.10g} {z:.10g}\n{r} {g} {b}\n{len(obs)} " + " ".join(obs))
        n_obs += len(obs)
    lines[1] = f"{len(cams)} {len(pts)}"
    Path(path_out).write_text("\n".join(lines + cams + pts) + "\n", encoding="ascii")
    Path(path_list).write_text("\n".join(rec.images[int(i)].name for i in order) + "\n", encoding="utf-8")
    return {"cameras": len(cams), "points": len(pts), "observations": n_obs, "duplicate_image_observations_dropped": n_dup}


def _region(xyz: np.ndarray) -> Dict[str, List[float]]:
    """The region in internal coordinates (x east, y north, z up): the tie points' 2-98 % box across, widened by
    10 %; up and down their 0.5-99.5 % range, widened by 20 % and at least a tenth of the width (flat ground: rocks
    and slopes above the bulk of the points stay in)."""
    lo, hi = np.percentile(xyz, REGION_PERCENTILES, axis=0)
    zlo, zhi = np.percentile(xyz[:, 2], REGION_Z_PERCENTILES)
    size = (hi - lo) * (1 + 2 * REGION_MARGIN)
    size = np.maximum(size, 1.0)
    size[2] = max((zhi - zlo) * 1.2, 0.1 * max(size[0], size[1]), 1.0)
    center = 0.5 * (lo + hi)
    center[2] = 0.5 * (zlo + zhi)
    return {"center": [float(v) for v in center], "size": [float(v) for v in size],
            "percentiles": list(REGION_PERCENTILES), "z_percentiles": list(REGION_Z_PERCENTILES),
            "margin": REGION_MARGIN}


def _check_sample(rec, order: Sequence[int], n: int, seed: int = 1) -> List[List[Any]]:
    """[image name, X, Y, Z, u, v] for ``n`` random tie-point observations (native pixels, corner origin)."""
    rng = np.random.default_rng(seed)
    pids = [int(p) for p in rec.points3D]
    if not pids:
        return []
    keep = set(int(i) for i in order)
    out = []
    for pid in rng.choice(pids, min(n, len(pids)), replace=False):
        pt = rec.points3D[int(pid)]
        els = [el for el in pt.track.elements if int(el.image_id) in keep]
        if not els:
            continue
        el = els[int(rng.integers(len(els)))]
        im = rec.images[int(el.image_id)]
        xy = im.points2D[int(el.point2D_idx)].xy
        out.append([im.name, *map(float, pt.xyz), float(xy[0]), float(xy[1])])
    return out


def write_metashape_job(project: SfmProject, out_dir: Optional[PathLike] = None, model_dir: Optional[PathLike] = None,
                        max_points: Optional[int] = None, check_samples: int = CHECK_SAMPLES,
                        metashape_exe: Optional[str] = None) -> Dict[str, Any]:
    """``<project>/metashape/``: the job, the Bundler tie points, the Metashape build script and its batch file
    (see the module).  ``model_dir``: a native-pixel model (default ``gui_native/sparse``, written if missing).
    Returns the job's summary."""
    import pycolmap
    from .dense import native_model_dir
    root = Path(project.root)
    out = Path(out_dir) if out_dir else root / METASHAPE_DIR
    out.mkdir(parents=True, exist_ok=True)
    mdir = Path(model_dir) if model_dir else native_model_dir(project)
    rec = pycolmap.Reconstruction(str(mdir))
    name = _block_name(project)
    order = sorted((int(i) for i in rec.reg_image_ids()), key=lambda i: rec.images[i].name)
    labels = _sensor_labels(rec, project)
    sensors, worst = [], 0.0
    for cid, cam in sorted(rec.cameras.items()):
        if int(cid) not in labels:
            continue
        cal = metashape_calibration(cam.model.name, cam.params, cam.width, cam.height)
        worst = max(worst, cal.get("max_error_px", 0.0))
        sensors.append(dict(cal, id=int(cid), label=labels[int(cid)]))
    masks = {r["name"]: bool(r.get("has_mask")) for r in project.images}
    rows = {r["name"]: r for r in project.images}
    offset = np.asarray(project.offset or [0.0, 0.0, 0.0], float)
    sigma = (project.settings.get("prior_sigma_m") or [REFERENCE_SIGMA_M] * 3)
    images = []
    for iid in order:
        im = rec.images[iid]
        wc = im.cam_from_world().inverse()
        T = np.eye(4)
        T[:3, :3] = np.asarray(wc.rotation.matrix(), float)
        T[:3, 3] = np.asarray(wc.translation, float)
        mpath = project.masks_dir / (im.name + ".png")
        images.append({"name": im.name, "path": str(project.images_dir / im.name), "sensor_id": int(im.camera_id),
                       "world_from_camera": T.tolist(),
                       "mask": str(mpath) if (masks.get(im.name, True) and mpath.is_file()) else None,
                       "reference": camera_reference(rows.get(im.name) or {}, offset, sigma,
                                                     np.asarray(im.cam_from_world().rotation.matrix(), float))})
    bund = out / f"{name}.bundler.out"
    blist = out / f"{name}.bundler_images.txt"
    tie = write_bundler(rec, bund, blist, order, max_points=max_points)
    xyz = np.array([p.xyz for p in rec.points3D.values()], float).reshape(-1, 3)
    frame = project.settings.get("world_frame") or "world"
    job = {"mppp": _version(), "block": name, "project": str(root), "model": str(mdir),
           "psx": str(out / f"{name}.psx"), "bundler": str(bund), "bundler_images": str(blist),
           "world_frame": frame, "offset": [float(v) for v in (project.offset or [0.0, 0.0, 0.0])],
           "crs_wkt": f'LOCAL_CS["MPPP {frame} (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]',
           "note": ("internal (chunk) coordinates = MPPP's world frame of the block, x east, y north, z up, metres; "
                    "the chunk transform adds the block offset, so the chunk's coordinates are the frame's"),
           "sensors": sensors, "images": images, "region": _region(xyz) if len(xyz) else None,
           "tie_points": tie, "conversion_max_error_px": worst,
           "check": _check_sample(rec, order, check_samples)}
    (out / "metashape_job.json").write_text(json.dumps(job, indent=1), encoding="utf-8")
    from . import metashape_build
    script = out / "build_metashape_project.py"
    shutil.copyfile(metashape_build.__file__, script)
    bat = write_metashape_bat(out, metashape_exe)
    return {"folder": str(out), "psx": job["psx"], "images": len(images), "sensors": len(sensors),
            "masks": sum(1 for i in images if i["mask"]), "tie_points": tie, "conversion_max_error_px": worst,
            "inexact_sensors": [s["label"] for s in sensors if not s["exact"]], "bat": str(bat),
            "script": str(script)}


def write_metashape_bat(out_dir: PathLike, metashape_exe: Optional[str] = None) -> Path:
    """``make_metashape_project.bat`` beside the job: Metashape Professional (``MPPP_METASHAPE``, else the default
    install) runs ``build_metashape_project.py`` on ``metashape_job.json`` and leaves the project open;
    ``set MPPP_METASHAPE_HEADLESS=1`` builds it without the window."""
    exe = metashape_exe or DEFAULT_METASHAPE
    lines = ["@echo off",
             "rem MPPP: the Metashape project of this block - cameras, tie points and masks from the COLMAP alignment",
             "rem (photos linked from ..\\images). Needs Metashape Professional (its Python). Written by  mppp metashape",
             "setlocal",
             f'if not defined MPPP_METASHAPE set "MPPP_METASHAPE={exe}"',
             'if not exist "%MPPP_METASHAPE%" (echo Metashape not found: %MPPP_METASHAPE% - set MPPP_METASHAPE to metashape.exe & pause & exit /b 1)',
             'if defined MPPP_METASHAPE_HEADLESS (',
             '  "%MPPP_METASHAPE%" -platform offscreen -r "%~dp0build_metashape_project.py" "%~dp0metashape_job.json" --quit',
             ') else (',
             '  start "" "%MPPP_METASHAPE%" -r "%~dp0build_metashape_project.py" "%~dp0metashape_job.json"',
             ')',
             "exit /b 0"]
    p = Path(out_dir) / "make_metashape_project.bat"
    p.write_bytes(("\r\n".join(lines) + "\r\n").encode("ascii"))
    return p


def _version() -> str:
    try:
        from .. import __version__
        return __version__
    except Exception:                                       # noqa: BLE001
        return "?"


__all__ = ["metashape_calibration", "metashape_project_px", "write_bundler", "write_metashape_job",
           "write_metashape_bat", "METASHAPE_DIR"]
