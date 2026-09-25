"""
Reconstruction from the CAHV-initialised cameras.

``reconstruct(project)``:

1. :func:`initial_reconstruction` — cameras (XML), rig (CAHV offset), frames
   posed from CAHV + waypoints, images with their full-resolution keypoints.
2. Triangulate the verified matches with the poses fixed
   (``pycolmap.triangulate_points``; thresholds in full-resolution pixels).
3. :func:`bundle_adjust` — weighted bundle adjustment (pyceres, COLMAP cost
   functions): every observation has standard deviation ``sigma_px`` in its
   own native pixels (``sigma_px / s`` in full-resolution pixels), Cauchy loss;
   frame poses free with the waypoint position priors; the right camera's
   sensor_from_rig free (one offset for all pairs); per-camera focal lengths,
   principal point and k1-k3 free; p1, p2 held at their initial values
   (zero by default) or refined with ``refine_tangential=True`` (v0p14.3);
   k4-k6 held at zero.
4. Drop observations whose residual exceeds ``max_residual_native_px`` and
   re-triangulate with the refined poses; one round per ``schedule`` entry
   (four by default since v0p14.3), then a final adjustment.

The result is in the project's world frame (ENU metres minus ``project.offset``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .project import SfmProject

PathLike = Union[str, Path]

# FULL_OPENCV: fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 -> indices held constant
_FIXED_EXTRA = {"FULL_OPENCV": [6, 7, 9, 10, 11], "OPENCV": [6, 7]}
_TANGENTIAL = {"FULL_OPENCV": [6, 7], "OPENCV": [6, 7]}         # p1, p2

# (triangulation threshold [full-res px], Cauchy scale [sigma], maximum residual kept [native px]) per round.
# v0p14.5: the fourth round repeats the third's limits - it only shows whether another
# re-triangulation and adjustment still changes anything (v0p14.3-4 used (6, 1.5, 1.5)).
DEFAULT_SCHEDULE = ((24.0, 10.0, 8.0), (12.0, 2.0, 4.0), (8.0, 2.0, 2.0), (8.0, 2.0, 2.0))


def fixed_camera_params(model_name: str, refine_principal_point: bool = True,
                        refine_tangential: bool = False) -> List[int]:
    """Indices of the camera parameters held constant in the bundle adjustment."""
    fixed = set(_FIXED_EXTRA.get(model_name, []))
    if refine_tangential:
        fixed -= set(_TANGENTIAL.get(model_name, []))
    if not refine_principal_point:
        fixed |= {2, 3}
    return sorted(fixed)


def _rigid(R: np.ndarray, C: np.ndarray):
    import pycolmap
    R = np.asarray(R, float)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ np.asarray(C, float))


def initial_reconstruction(project: SfmProject):
    """Reconstruction with every frame posed from CAHV (no 3-D points)."""
    import pycolmap
    db = pycolmap.Database.open(str(project.database))
    rec = pycolmap.Reconstruction()
    for cam in db.read_all_cameras():
        rec.add_camera(cam)
    for rig in db.read_all_rigs():
        rec.add_rig(rig)
    by_id = {r["image_id"]: r for r in project.images if "image_id" in r}
    images = {im.image_id: im for im in db.read_all_images()}
    for fr in db.read_all_frames():
        rig = rec.rigs[fr.rig_id]
        ref_cam = rig.ref_sensor_id.id
        ref = [d.id for d in fr.data_ids if d.sensor_id.id == ref_cam]
        r = by_id[ref[0]]
        fr.rig_from_world = _rigid(r["prior_R_w2c"], r["prior_C"])
        rec.add_frame(fr)
    for iid, im in images.items():
        kp = db.read_keypoints(iid)[:, :2].astype(np.float64)
        new = pycolmap.Image(name=im.name, keypoints=kp, camera_id=im.camera_id, image_id=iid)
        new.frame_id = im.frame_id
        rec.add_image(new)
    for fid in list(rec.frames):
        rec.register_frame(fid)
    db.close()
    return rec


def triangulate(rec, project: SfmProject, max_reproj_px: float = 8.0, min_angle_deg: float = 1.5,
                out_dir: Optional[PathLike] = None):
    """Triangulate all verified matches with the current (fixed) poses and intrinsics."""
    import pycolmap
    opts = pycolmap.IncrementalPipelineOptions()
    opts.triangulation.merge_max_reproj_error = max_reproj_px
    opts.triangulation.complete_max_reproj_error = max_reproj_px
    opts.triangulation.min_angle = min_angle_deg
    opts.triangulation.ignore_two_view_tracks = False
    opts.mapper.filter_max_reproj_error = max_reproj_px
    opts.mapper.filter_min_tri_angle = min_angle_deg
    # triangulate_points runs COLMAP's own (unweighted) BA on the points; by default that
    # BA also refines the rig's sensor_from_rig (it moved the Belva baseline 4 mm) - not here
    opts.ba_refine_sensor_from_rig = False
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    out = Path(out_dir) if out_dir else project.root / "sparse" / "_triangulation"
    out.mkdir(parents=True, exist_ok=True)
    return pycolmap.triangulate_points(rec, str(project.database), str(project.images_dir), str(out),
                                       clear_points=True, options=opts, refine_intrinsics=False)


def _scales(rec, project: SfmProject) -> Dict[int, float]:
    by_name = {r["name"]: float(r["downsample_scale"]) for r in project.images}
    return {iid: by_name[im.name] for iid, im in rec.images.items()}


def bundle_adjust(rec, project: SfmProject, sigma_px: float = 0.5, loss_scale: float = 2.0,
                  refine_intrinsics: bool = True, refine_principal_point: bool = True,
                  refine_tangential: bool = False, refine_rig: Union[bool, str] = "rotation", use_priors: bool = True, max_iterations: int = 100,
                  min_frame_observations: int = 30, num_threads: int = -1, verbose: bool = False) -> Dict[str, Any]:
    """
    Weighted BA in place (see module docstring).  ``sigma_px``: keypoint
    standard deviation in native pixels; ``loss_scale``: Cauchy scale in units
    of that sigma.  ``refine_rig``: "rotation" (default) refines the right
    camera's orientation in the rig and keeps the CAHV stereo baseline vector,
    which then fixes the scale of the reconstruction; True refines the
    baseline too (scale is then set only by the position priors, weakly for
    stations metres apart - the first Belva test shrank it from 0.424 to
    0.14 m); False holds the rig fixed.  Frames with fewer than
    ``min_frame_observations`` observations are held at their current pose
    (on Belva two frames with no observations otherwise rotated freely, by 9
    and 36 deg).  ``refine_tangential`` (v0p14.3): refine p1, p2 of
    OPENCV / FULL_OPENCV cameras; otherwise they stay at their initial
    values.  Returns the solver summary as a dict.
    """
    import pycolmap
    import pycolmap.cost_functions as cf
    import pyceres

    # The problem comes from pycolmap's adjuster, which gives every frame pose its
    # quaternion x translation manifold (pyceres cannot build one).  It is created on a
    # COPY of the reconstruction holding one dummy point per frame: removing the
    # adjuster's own 3-D point blocks from the full reconstruction is O(residuals) per
    # point in Ceres (hours for 10^6 observations).  Poses and intrinsics are optimised
    # in the copy and written back; the 3-D points are the real ones.
    reg = set(rec.reg_image_ids())
    work = copy.deepcopy(rec)
    for pid in list(work.points3D):
        work.delete_point3D(pid)
    dummies = []
    for fid, fr in work.frames.items():
        ims = [d.id for d in fr.data_ids if d.id in reg]
        if not ims:
            continue
        im = work.images[ims[0]]
        if im.num_points2D() < 2:
            continue
        X = im.cam_from_world().inverse() * np.array([0.0, 0.0, 10.0])
        tr = pycolmap.Track()
        tr.add_element(im.image_id, 0)                   # COLMAP's BA wants tracks of length >= 2
        tr.add_element(im.image_id, 1)
        dummies.append(work.add_point3D(X, tr))
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = refine_intrinsics
    opts.refine_principal_point = refine_intrinsics and refine_principal_point
    opts.refine_extra_params = refine_intrinsics
    opts.refine_sensor_from_rig = False                  # the rig offset is our own block below
    opts.refine_rig_from_world = True
    opts.print_summary = False
    config = pycolmap.BundleAdjustmentConfig()
    for iid in reg:
        config.add_image(iid)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.UNSPECIFIED)
    ba = pycolmap.create_default_ceres_bundle_adjuster(opts, config, work)
    try:
        prob = ba.problem
    except TypeError as e:                                   # e.g. conda-forge (CUDA) pycolmap + pip pyceres
        raise RuntimeError(
            f"pycolmap {pycolmap.__version__} and pyceres {getattr(pyceres, '__version__', '?')} in this environment "
            f"do not share Ceres types ({str(e).splitlines()[0]}). The weighted BA needs a matching pair, e.g. the "
            f"pip wheels pycolmap 4.2 + pyceres 2.6. Keep this kernel on those and run GPU extraction and matching "
            f"in the CUDA environment with python=... (see mppp.sfm.gpu).") from e
    for pid in dummies:
        xyz = work.points3D[pid].xyz
        if prob.has_parameter_block(xyz):
            prob.remove_parameter_block(xyz)
    scale = _scales(rec, project)
    loss = pyceres.CauchyLoss(float(loss_scale))
    # Rig.sensor_from_rig() returns a COPY in pycolmap, so each non-reference sensor
    # gets one explicit parameter array (shared by all its observations), written back below.
    rig_blocks: Dict[tuple, np.ndarray] = {}
    for rid, rig in rec.rigs.items():
        for sid in rig.non_ref_sensors:
            arr = np.array(rig.sensor_from_rig(sid).params, dtype=np.float64)
            rig_blocks[(rid, sid.id)] = arr
    n_obs = 0
    pose_blocks: Dict[int, np.ndarray] = {}
    for pid, pt in rec.points3D.items():
        xyz = pt.xyz
        for el in pt.track.elements:
            if el.image_id not in reg:
                continue
            im = rec.images[el.image_id]
            fr = work.frames[im.frame_id]
            rig = rec.rigs[fr.rig_id]
            cam = work.cameras[im.camera_id]
            s = scale[el.image_id]
            cov = np.eye(2) * (sigma_px / s) ** 2
            xy = np.asarray(im.points2D[el.point2D_idx].xy, float)
            sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=im.camera_id)
            pose = pose_blocks.get(im.frame_id)
            if pose is None:
                pose = pose_blocks[im.frame_id] = fr.rig_from_world.params
            if rig.is_ref_sensor(sensor):
                prob.add_residual_block(cf.ReprojErrorCost(cam.model, cov, xy), loss,
                                        [xyz, pose, cam.params])
            else:
                prob.add_residual_block(cf.RigReprojErrorCost(cam.model, cov, xy), loss,
                                        [xyz, rig_blocks[(fr.rig_id, im.camera_id)], pose, cam.params])
            n_obs += 1

    # intrinsics: hold k4-k6 at zero, p1, p2 at their start unless refine_tangential (and the principal point if asked);
    # a camera may hold more ("fixed_params" in project.cameras, e.g. the Mastcam-Z focus bins, v0p14.4)
    from .project import FULL_OPENCV_NAMES
    key_of = {int(v): k for k, v in project.settings.get("database", {}).get("cameras", {}).items()}
    for cid, cam in work.cameras.items():
        if not prob.has_parameter_block(cam.params):
            continue
        if not refine_intrinsics:
            prob.set_parameter_block_constant(cam.params)
            continue
        fixed = fixed_camera_params(cam.model.name, refine_principal_point, refine_tangential)
        extra = project.cameras.get(key_of.get(int(cid), ""), {}).get("fixed_params") or []
        if extra and cam.model.name in ("OPENCV", "FULL_OPENCV"):
            fixed = sorted(set(fixed) | {FULL_OPENCV_NAMES.index(n) for n in extra
                                          if FULL_OPENCV_NAMES.index(n) < len(cam.params)})
        prob.set_parameter_block_variable(cam.params)
        if fixed:
            prob.set_manifold(cam.params, pyceres.SubsetManifold(len(cam.params), sorted(fixed)))

    # waypoint position priors on every frame (the reference camera's centre)
    n_prior = 0
    if use_priors:
        sig = np.asarray(project.settings.get("prior_sigma_m", [1.0, 1.0, 1.0]), float)
        cov3 = np.diag(sig ** 2)
        by_name = {r["name"]: r for r in project.images}
        for fid, fr in work.frames.items():
            rig = rec.rigs[fr.rig_id]
            ref = [d.id for d in fr.data_ids if d.sensor_id.id == rig.ref_sensor_id.id]
            if not ref or ref[0] not in rec.images:
                continue
            C = np.asarray(by_name[rec.images[ref[0]].name]["prior_C"], float)
            pose = pose_blocks.get(fid)
            if pose is None:
                continue
            prob.add_residual_block(cf.AbsolutePosePositionPriorCost(cov3, C), None, [pose])
            n_prior += 1

    # weakly observed frames: hold (their attitude is otherwise set by a handful of points)
    n_frame: Dict[int, int] = {}
    for pid, pt in rec.points3D.items():
        for el in pt.track.elements:
            if el.image_id in reg:
                f = rec.images[el.image_id].frame_id
                n_frame[f] = n_frame.get(f, 0) + 1
    held = []
    for fid, pose in pose_blocks.items():
        if n_frame.get(fid, 0) < min_frame_observations and prob.has_parameter_block(pose):
            prob.set_parameter_block_constant(pose)
            held.append(int(fid))

    for arr in rig_blocks.values():
        if not prob.has_parameter_block(arr):
            continue
        if refine_rig:
            # pyceres has no product manifold and takes no Python manifolds.  The rig
            # rotation is small (Navcam L->R ~0.13 deg), so its quaternion is refined in
            # x, y, z with w held: |q|^2 - 1 stays O(1e-6), i.e. a scale error < 0.01 px;
            # the quaternion is normalised when written back.
            prob.set_manifold(arr, pyceres.SubsetManifold(7, [3, 4, 5, 6] if refine_rig == "rotation" else [3]))
        else:
            prob.set_parameter_block_constant(arr)

    so = pyceres.SolverOptions()
    so.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    so.max_num_iterations = int(max_iterations)
    so.num_threads = int(num_threads) if num_threads > 0 else (__import__("os").cpu_count() or 1)
    so.minimizer_progress_to_stdout = bool(verbose)
    summary = pyceres.SolverSummary()
    pyceres.solve(so, prob, summary)
    for fid in pose_blocks:
        rec.frames[fid].rig_from_world = work.frames[fid].rig_from_world
    for cid, cam in work.cameras.items():
        rec.cameras[cid].params = np.array(cam.params)
    for (rid, cid), arr in rig_blocks.items():
        q = arr[:4] / np.linalg.norm(arr[:4])
        from scipy.spatial.transform import Rotation
        T = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_quat(q).as_matrix()), arr[4:7].copy())
        rec.rigs[rid].set_sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid), T)
    return {"observations": n_obs, "priors": n_prior, "frames_held": held, "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost, "iterations": summary.num_successful_steps + summary.num_unsuccessful_steps,
            "termination": str(summary.termination_type), "brief": summary.BriefReport(),
            "refine_tangential": bool(refine_tangential)}


def _verified_matches(project: SfmProject):
    import pycolmap
    db = pycolmap.Database.open(str(project.database))
    ids, geoms = db.read_two_view_geometries()
    out = []
    for pid, g in zip(ids, geoms):
        m = np.asarray(g.inlier_matches)
        if m.size:
            i1, i2 = pycolmap.pair_id_to_image_pair(pid)
            out.append((int(i1), int(i2), m))
    db.close()
    return out


def register_stations(rec, project: SfmProject, min_inliers: int = 40, max_error_px: float = 12.0,
                      reference: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Correct each station's pose as one rigid block (its CAHV-relative image poses
    kept), from 2-D/3-D correspondences with stations already placed.

    The waypoint positions put stations metres apart to within decimetres,
    which at Navcam range is tens of pixels - too far for triangulation with
    fixed poses to join tracks across stations.  Within a station, CAHV
    (mast kinematics) is good to a fraction of a pixel.  So: start from the
    ``reference`` station (default: the one with the most 3-D points), and
    repeatedly register the station with the most correspondences to the
    registered set as a generalized (multi-camera) absolute pose, RANSAC with
    ``max_error_px`` in full-resolution pixels.  Its frames and its
    station-only points are moved with it.  Stations that cannot be
    registered stay at their priors (reported).
    """
    import pycolmap
    station = {iid: project.image(im.name)["station"] for iid, im in rec.images.items()}
    st_images: Dict[str, List[int]] = {}
    for iid, s in station.items():
        st_images.setdefault(s, []).append(iid)
    # 3-D point of each (image, keypoint) and the stations of each point
    p3d: Dict[tuple, int] = {}
    pt_st: Dict[int, set] = {}
    for pid, pt in rec.points3D.items():
        sts = set()
        for el in pt.track.elements:
            p3d[(el.image_id, el.point2D_idx)] = pid
            sts.add(station[el.image_id])
        pt_st[pid] = sts
    matches = _verified_matches(project)
    # correspondences: target station -> {(image, kp): point id}, keyed by the point's station
    corr: Dict[str, Dict[tuple, tuple]] = {s: {} for s in st_images}
    for i1, i2, m in matches:
        s1, s2 = station.get(i1), station.get(i2)
        if s1 is None or s2 is None or s1 == s2:
            continue
        for (a, b), (ia, ib) in (((0, 1), (i1, i2)), ((1, 0), (i2, i1))):
            sa, sb = station[ia], station[ib]
            for k in range(m.shape[0]):
                pid = p3d.get((ia, int(m[k, a])))
                if pid is not None and pt_st[pid] == {sa}:
                    corr[sb].setdefault((ib, int(m[k, b])), (pid, sa))
    n_pts = {s: sum(1 for p in pt_st.values() if p == {s}) for s in st_images}
    ref = reference or max(n_pts, key=n_pts.get)
    placed, report = {ref}, {"reference": ref, "stations": {}}
    frames_of = {s: {rec.images[i].frame_id for i in ims} for s, ims in st_images.items()}
    while True:
        cand = {s: [(k, v) for k, v in c.items() if v[1] in placed] for s, c in corr.items() if s not in placed}
        cand = {s: v for s, v in cand.items() if len(v) >= min_inliers}
        if not cand:
            break
        s = max(cand, key=lambda k: len(cand[k]))
        ims = sorted(st_images[s])
        idx = {iid: n for n, iid in enumerate(ims)}
        pts2, pts3, cidx = [], [], []
        for (iid, kp), (pid, _) in cand[s]:
            pts2.append(rec.images[iid].points2D[kp].xy)
            pts3.append(rec.points3D[pid].xyz)
            cidx.append(idx[iid])
        cams_from_rig = [rec.images[i].cam_from_world() for i in ims]
        cams = [rec.cameras[rec.images[i].camera_id] for i in ims]
        ro = pycolmap.RANSACOptions()
        ro.max_error = float(max_error_px)
        res = pycolmap.estimate_and_refine_generalized_absolute_pose(
            np.asarray(pts2, float), np.asarray(pts3, float), cidx, cams_from_rig, cams, estimation_options=ro)
        n_in = int(np.sum(res["inlier_mask"])) if res else 0
        if not res or n_in < min_inliers:
            report["stations"][s] = {"registered": False, "correspondences": len(pts2), "inliers": n_in}
            corr[s] = {}                                    # give up on it
            continue
        T = res["rig_from_world"]                           # corrected-world -> prior-world of s
        for fid in frames_of[s]:
            fr = rec.frames[fid]
            fr.rig_from_world = fr.rig_from_world * T
        Tinv = T.inverse()
        for pid, sts in pt_st.items():
            if sts == {s}:
                rec.points3D[pid].xyz[:] = Tinv * rec.points3D[pid].xyz
        shift = float(np.linalg.norm(Tinv.translation))
        ang = float(np.degrees(T.rotation.angle()))
        report["stations"][s] = {"registered": True, "correspondences": len(pts2), "inliers": n_in,
                                 "shift_m": shift, "rotation_deg": ang}
        placed.add(s)
        if verbose:
            print(f"[sfm] station {s}: {n_in}/{len(pts2)} inliers, moved {shift:.3f} m / {ang:.3f} deg")
    for s in st_images:
        report["stations"].setdefault(s, {"registered": s in placed, "correspondences": 0})
    report["stations"][ref] = {"registered": True, "reference": True}
    report["unplaced"] = sorted(s for s in st_images if s not in placed)
    return report


def native_residuals(rec, project: SfmProject) -> Dict[str, np.ndarray]:
    """Per observation: residual in native pixels, image id, point id, 2-D index (vectorised per image)."""
    scale = _scales(rec, project)
    iid_l, pid_l, idx_l = [], [], []
    for pid, pt in rec.points3D.items():
        for el in pt.track.elements:
            iid_l.append(el.image_id); pid_l.append(pid); idx_l.append(el.point2D_idx)
    iid_a, pid_a, idx_a = np.array(iid_l, np.int64), np.array(pid_l, np.int64), np.array(idx_l, np.int64)
    res = np.full(len(iid_a), np.inf)
    if len(iid_a) == 0:
        return {"residual_native_px": res, "image_id": iid_a, "point3D_id": pid_a, "point2D_idx": idx_a}
    xyz = {pid: rec.points3D[pid].xyz for pid in np.unique(pid_a)}
    order = np.argsort(iid_a, kind="stable")
    bounds = np.flatnonzero(np.diff(iid_a[order])) + 1
    for sel in np.split(order, bounds):
        iid = int(iid_a[sel[0]])
        im = rec.images[iid]
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        R, t = T.rotation.matrix(), np.asarray(T.translation)
        X = np.array([xyz[p] for p in pid_a[sel]]) @ R.T + t
        front = X[:, 2] > 1e-9
        kp = np.array([im.points2D[int(k)].xy for k in idx_a[sel]], float)
        uv = np.full((len(sel), 2), np.nan)
        if front.any():
            uv[front] = cam.img_from_cam(X[front])
        r = np.linalg.norm(uv - kp, axis=1) * scale[iid]
        r[~np.isfinite(r)] = np.inf
        res[sel] = r
    return {"residual_native_px": res, "image_id": iid_a, "point3D_id": pid_a, "point2D_idx": idx_a}


def filter_observations(rec, project: SfmProject, max_residual_native_px: float = 2.0) -> int:
    r = native_residuals(rec, project)
    bad = r["residual_native_px"] > max_residual_native_px
    n = 0
    for iid, pid, idx in zip(r["image_id"][bad].tolist(), r["point3D_id"][bad].tolist(), r["point2D_idx"][bad].tolist()):
        # plain ints: pycolmap's maps do not recognise numpy integers ('np.int64(5) in points3D' is False)
        if pid in rec.points3D:
            if rec.points3D[pid].track.length() <= 2:
                rec.delete_point3D(int(pid))
            else:
                rec.delete_observation(int(iid), int(idx))
            n += 1
    return n


def drop_short_tracks(rec, min_track_length: int = 2) -> int:
    """Delete 3-D points observed in fewer than ``min_track_length`` images; returns the number deleted."""
    short = [int(pid) for pid, pt in rec.points3D.items() if pt.track.length() < min_track_length]
    for pid in short:
        rec.delete_point3D(pid)
    return len(short)


def track_statistics(rec, project: SfmProject) -> Dict[str, Any]:
    station = {r["name"]: r["station"] for r in project.images}
    n_cross, lens = 0, []
    for pt in rec.points3D.values():
        sts = {station[rec.images[el.image_id].name] for el in pt.track.elements}
        n_cross += len(sts) > 1
        lens.append(pt.track.length())
    n = max(1, len(lens))
    lens_a = np.asarray(lens, int)
    return {"points": len(lens), "cross_station_points": int(n_cross), "cross_station_fraction": n_cross / n,
            "mean_track_length": float(np.mean(lens)) if lens else 0.0,
            "two_view_points": int((lens_a == 2).sum()), "two_view_fraction": float((lens_a == 2).mean()) if lens else 0.0,
            "observations": int(np.sum(lens)) if lens else 0}


def reconstruct(project: SfmProject, sigma_px: float = 0.5,
                schedule: Sequence[Sequence[float]] = DEFAULT_SCHEDULE,
                refine_intrinsics: bool = True, refine_rig: Union[bool, str] = "rotation",
                register: bool = False, max_iterations: int = 50, out_name: str = "cahv_ba",
                verbose: bool = True, min_track_length: int = 2, min_tri_angle_deg: float = 0.5,
                refine_tangential: bool = False, gui_native: bool = True):
    """
    CAHV-initialised triangulation + weighted BA (see module docstring).

    ``schedule``: rounds of (triangulation threshold [full-res px], Cauchy
    scale [sigma], maximum residual kept [native px]) - a graduated schedule
    so that observations far from the initial poses can still pull them.
    Default: four rounds; the fourth repeats the third's limits (v0p14.5) to
    show whether one more re-triangulation and adjustment changes anything.
    The final adjustment uses the last round's Cauchy scale and cut-off.
    ``gui_native`` (v0p14.5): also write the native-pixel viewing copy
    ``gui_native/`` for the COLMAP GUI (``open_in_colmap.bat``).
    ``register``: first move whole stations with :func:`register_stations`
    (needed when the waypoint priors are off by more than ~20 px at range;
    at Belva they are within ~1-2 px, so it is off by default).
    ``min_track_length``: shortest track kept (v0p13).  Default 2: two-view
    tie points - e.g. seen only by the two eyes of one stereo pair - are kept,
    as COLMAP's ``ignore_two_view_tracks=False`` triangulates them; 3 drops
    them after every round.
    ``min_tri_angle_deg``: smallest triangulation angle (default 0.5 deg
    since v0p14.3; was 1.5).  With the 0.424 m Navcam baseline, 0.5 deg keeps
    stereo-only points out to about 49 m (1.5 deg: 16 m).  Their depth is
    poorly constrained, their direction is not: they tie attitudes, and the
    weighting and the residual cut-offs limit their influence.
    ``refine_tangential``: refine p1, p2 (see :func:`bundle_adjust`).  Start
    them from the Metashape calibration by creating the project with
    ``zero_terms=("b1", "b2")``; with the default ``zero_terms`` they start at 0.
    Returns the reconstruction (also written to ``sparse/<out_name>``).
    """
    if min_track_length < 2:
        raise ValueError("min_track_length must be >= 2")
    import time
    t0 = time.time()
    rec = initial_reconstruction(project)
    init = copy.deepcopy(rec)
    log: List[Dict[str, Any]] = []
    if register:
        rec = triangulate(rec, project, max_reproj_px=8.0)
        reg = register_stations(rec, project, verbose=verbose)
        log.append({"station_registration": reg})
    for k, (tpx, loss, rmax) in enumerate(schedule):
        rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg)
        st0 = track_statistics(rec, project)
        ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(loss), refine_intrinsics=refine_intrinsics,
                           refine_tangential=refine_tangential, refine_rig=refine_rig, max_iterations=max_iterations)
        n_bad = filter_observations(rec, project, float(rmax))
        n_bad += drop_short_tracks(rec, min_track_length)
        st = track_statistics(rec, project)
        res = native_residuals(rec, project)["residual_native_px"]
        entry = {"round": k + 1, "triangulation_px_full": tpx, "cauchy_scale_sigma": loss, "after_triangulation": st0,
                 "ba": ba["brief"], "filtered_observations": n_bad, "max_residual_native_px": rmax, "tracks": st,
                 "rms_native_px": float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan"),
                 "median_native_px": float(np.median(res)) if res.size else float("nan"),
                 "elapsed_s": round(time.time() - t0, 1)}
        log.append(entry)
        if verbose:
            print(f"[sfm] round {k + 1}: {st['points']} points ({st['cross_station_fraction']:.2%} cross-station), "
                  f"median {entry['median_native_px']:.3f} / rms {entry['rms_native_px']:.3f} native px, "
                  f"{n_bad} obs filtered, {entry['elapsed_s']:.0f} s", flush=True)
    ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(schedule[-1][1]),
                       refine_intrinsics=refine_intrinsics, refine_tangential=refine_tangential,
                       refine_rig=refine_rig, max_iterations=2 * max_iterations)
    # the final adjustment can push a few points behind a camera or past the limit
    n_bad = filter_observations(rec, project, float(schedule[-1][2]))
    n_bad += drop_short_tracks(rec, min_track_length)
    log.append({"final_ba": ba["brief"], "frames_held": ba["frames_held"], "filtered_after_final": n_bad,
                "tracks": track_statistics(rec, project)})
    out = project.root / "sparse" / out_name
    out.mkdir(parents=True, exist_ok=True)
    rec.write(str(out))
    idir = project.root / "sparse" / "cahv_initial"
    idir.mkdir(parents=True, exist_ok=True)
    init.write(str(idir))
    (out / "mppp_sfm_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    project.settings["reconstruction"] = {"path": str(out.relative_to(project.root)), "log": log,
                                          "sigma_px_native": sigma_px, "refine_rig": refine_rig,
                                          "min_track_length": min_track_length,
                                          "min_tri_angle_deg": min_tri_angle_deg,
                                          "refine_tangential": bool(refine_tangential),
                                          "schedule": [list(map(float, r)) for r in schedule]}
    project.save()
    if gui_native:
        try:
            from .export import write_gui_native
            g = write_gui_native(project, rec)               # native-pixel viewing copy (v0p14.5)
            if verbose:
                print(f"[sfm] COLMAP GUI copy in native pixels: {g['dir']} ({g['images']} images, "
                      f"{g['points']} points)", flush=True)
        except Exception as e:                               # noqa: BLE001  (a viewing aid; never stop the run)
            print(f"[sfm] gui_native not written: {type(e).__name__}: {e}", flush=True)
    try:
        from .database import write_gui_project
        write_gui_project(project, model=out_name)          # colmap_gui.ini, open_in_colmap*.bat (v0p14.4)
    except OSError:
        pass
    return rec
