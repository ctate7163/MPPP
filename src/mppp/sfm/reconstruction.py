"""
Reconstruction from the CAHV-initialised cameras.

``reconstruct(project)``:

1. :func:`initial_reconstruction` — cameras (from the database: the project's
   initial cameras), rig (CAHV offset), frames
   posed from CAHV + waypoints, images with their full-resolution keypoints.
2. Triangulate the verified matches with the poses fixed
   (``pycolmap.triangulate_points``; thresholds in full-resolution pixels).
3. :func:`bundle_adjust` — weighted bundle adjustment (pyceres, COLMAP cost
   functions): every observation has standard deviation ``sigma_px`` in its
   own native pixels (``sigma_px / s`` in full-resolution pixels), Cauchy loss;
   frame poses free with the waypoint position priors; the right camera's
   sensor_from_rig free (one offset for all pairs); per-camera focal lengths,
   principal point and k1-k3 free; p1, p2 refined (``refine_tangential``,
   default True since v0p20) or held; k4-k6 held at zero except where a
   camera lists them in ``free_params`` (v0p20: k4 of the rational Navcam
   cameras); a Mastcam-Z focus-bin camera holds its ``fixed_params``.
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
_FIXED_EXTRA = {"FULL_OPENCV": [6, 7, 9, 10, 11], "OPENCV": [6, 7],
                "THIN_PRISM_FISHEYE": [6, 7, 10, 11]}          # v0p30: p1, p2 (tangential) and sx1, sy1 (thin prism)
_TANGENTIAL = {"FULL_OPENCV": [6, 7], "OPENCV": [6, 7], "THIN_PRISM_FISHEYE": [6, 7]}         # p1, p2
_PARAM_NAMES = {"FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
                "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
                "THIN_PRISM_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"),
                "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")}

# (triangulation threshold [full-res px], Cauchy scale [sigma], maximum residual kept [native px]) per round.
# v0p14.5: the fourth round repeats the third's limits - it only shows whether another
# re-triangulation and adjustment still changes anything (v0p14.3-4 used (6, 1.5, 1.5)).
DEFAULT_SCHEDULE = ((24.0, 10.0, 8.0), (12.0, 2.0, 4.0), (8.0, 2.0, 2.0), (8.0, 2.0, 2.0))
ATTITUDE_PRIOR_DEG = 1.0      # v0p20: weak CAHV attitude prior per frame (holds the block's orientation)


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
                out_dir: Optional[PathLike] = None, max_transitivity: int = 1, create_max_angle_error_deg: float = 2.0,
                continue_max_angle_error_deg: float = 2.0, complete_max_transitivity: int = 5):
    """
    Triangulate all verified matches with the current (fixed) poses and intrinsics.
    v0p22 options (COLMAP's IncrementalTriangulator): ``max_transitivity`` - how
    many images a correspondence may be chained through when a track is built
    (1 = direct matches only; higher links observations at other stations that
    were only matched via an intermediate image); ``create_max_angle_error_deg``
    / ``continue_max_angle_error_deg`` - ray-angle tolerance when a track is
    created or extended; ``complete_max_transitivity`` - the same for completing
    existing tracks.
    """
    import pycolmap
    opts = pycolmap.IncrementalPipelineOptions()
    opts.triangulation.merge_max_reproj_error = max_reproj_px
    opts.triangulation.complete_max_reproj_error = max_reproj_px
    opts.triangulation.min_angle = min_angle_deg
    opts.triangulation.ignore_two_view_tracks = False
    opts.triangulation.max_transitivity = int(max_transitivity)
    opts.triangulation.complete_max_transitivity = int(complete_max_transitivity)
    opts.triangulation.create_max_angle_error = float(create_max_angle_error_deg)
    opts.triangulation.continue_max_angle_error = float(continue_max_angle_error_deg)
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
    # v0p31: images the project does not describe (deregistered, e.g. frames dropped after a rebuild) are skipped
    return {iid: by_name[im.name] for iid, im in rec.images.items() if im.name in by_name}


def _block_covariances(rec, prob, work, rig_blocks, key_of, summary, n_obs, n_prior, n_att,
                       pose_blocks: Optional[Dict[int, np.ndarray]] = None) -> Optional[Dict[str, Any]]:
    """Covariances of the rig and camera blocks of a solved :func:`bundle_adjust` problem (v0p35).  The 3-D points
    (``rec``'s) are eliminated by the Schur complement; the frame poses (the work copy's) count as further
    parameters, so each block's covariance is marginal over the poses, the other blocks and the points."""
    import pycolmap
    opts = pycolmap.BACovarianceOptions()
    opts.params = pycolmap.BACovarianceOptionsParams.ALL
    # rigged frames: COLMAP's own pose discovery wants every image to be its frame's reference, so the frame
    # poses are handed over explicitly (one pose block per frame, keyed by its first registered image)
    custom = []
    for fid, arr in (pose_blocks or {}).items():
        if not prob.has_parameter_block(arr) or prob.is_parameter_block_constant(arr):
            continue
        ims = [d.id for d in work.frames[fid].data_ids if d.id in work.images]
        pp = pycolmap.ExperimentalPoseParam()
        pp.image_id = int(ims[0]) if ims else int(fid)
        pp.cam_from_world = arr[:4]            # the block pointer (the binding checks for 4 values)
        custom.append(pp)
    opts.experimental_custom_poses = custom
    try:
        bac = pycolmap.estimate_ba_covariance_from_problem(opts, rec, prob)
    except Exception as e:                                  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if bac is None:
        return {"error": "covariance estimation failed (rank-deficient problem?)"}
    blocks: Dict[str, Any] = {}
    for (rid, cid), arr in rig_blocks.items():
        if prob.has_parameter_block(arr) and not prob.is_parameter_block_constant(arr):
            c = bac.get_other_params_cov(arr)
            if c is not None:
                blocks[f"rig:{rid}:{cid}"] = np.asarray(c)
    for cid, cam in work.cameras.items():
        if prob.has_parameter_block(cam.params) and not prob.is_parameter_block_constant(cam.params):
            c = bac.get_other_params_cov(cam.params)
            if c is not None:
                blocks[f"camera:{key_of.get(int(cid), cid)}"] = np.asarray(c)
    n_res = prob.num_residuals()
    n_par = prob.num_effective_parameters() if hasattr(prob, "num_effective_parameters") else prob.num_parameters()
    red = max(n_res - n_par, 1)
    return {"blocks": blocks, "variance_factor": float(2.0 * summary.final_cost / red), "residuals": int(n_res),
            "parameters": int(n_par)}


LOCALIZE_MIN_CROSS_POINTS = 20       # v0p35.1: an unlocalized station keeps its prior below this many cross-station points


def unlocalized_stations(project: SfmProject, min_images: int, rec=None,
                         min_cross_points: int = LOCALIZE_MIN_CROSS_POINTS) -> Dict[str, Dict[str, Any]]:
    """
    Stations (site, drive) with fewer than ``min_images`` images in the project (v0p35.1, notebook 03
    ``LOCALIZE_MIN_IMAGES``): short stops during a drive, whose waypoint positions are the least reliable.
    Returns {station: {"images", "cross_points", "prior"}} with ``prior`` "dropped" (the bundle adjustment leaves out
    its position prior) or "kept: ..." when dropping it would leave the station unconstrained: fewer than
    ``min_cross_points`` tie points shared with other stations in ``rec`` (not checked without ``rec``), or every
    station of the block below ``min_images``.
    """
    if not min_images:
        return {}
    n: Dict[str, int] = {}
    for r in project.images:
        n[r["station"]] = n.get(r["station"], 0) + 1
    small = {st for st, k in n.items() if k < int(min_images)}
    if not small:
        return {}
    out: Dict[str, Dict[str, Any]] = {st: {"images": n[st], "cross_points": None, "prior": "dropped"} for st in small}
    if len(small) == len(n):
        for v in out.values():
            v["prior"] = "kept: every station is below the minimum"
        return out
    if rec is not None:
        st_of = {}
        by_name = {r["name"]: r["station"] for r in project.images}
        for iid, im in rec.images.items():
            st_of[iid] = by_name.get(im.name)
        cross = {st: 0 for st in small}
        for pt in rec.points3D.values():
            sts = {st_of.get(el.image_id) for el in pt.track.elements}
            if len(sts - {None}) > 1:
                for st in sts & small:
                    cross[st] += 1
        for st in small:
            out[st]["cross_points"] = cross[st]
            if cross[st] < min_cross_points:
                out[st]["prior"] = f"kept: {cross[st]} cross-station points (< {min_cross_points})"
    return out


def bundle_adjust(rec, project: SfmProject, sigma_px: float = 0.5, loss_scale: float = 2.0,
                  refine_intrinsics: bool = True, refine_principal_point: bool = True,
                  refine_tangential: bool = True, refine_rig: Union[bool, str] = "rotation", use_priors: bool = True, max_iterations: int = 100,
                  min_frame_observations: int = 30, num_threads: int = -1, verbose: bool = False,
                  attitude_prior_deg: Optional[float] = ATTITUDE_PRIOR_DEG,
                  rig_translation_sigma_m: Optional[float] = None,
                  hold_cameras: Sequence[str] = (), linear_solver: str = "auto",
                  covariance: bool = False, keypoint_scale: Optional[Dict[int, float]] = None,
                  keypoint_map=None, localize_min_images: Optional[int] = None) -> Dict[str, Any]:
    """
    Weighted BA in place (see module docstring).  ``hold_cameras`` (v0p22.2):
    keys of project cameras (``"NL"``, ``"NR"``, a focus bin...) whose
    intrinsics are held entirely, whatever ``refine_intrinsics`` says.  ``sigma_px``: keypoint
    standard deviation in native pixels; ``loss_scale``: Cauchy scale in units
    of that sigma.  ``refine_rig``: "rotation" (default) refines the right
    camera's orientation in the rig and keeps the CAHV stereo baseline vector,
    which then fixes the scale of the reconstruction; True refines the
    baseline too (scale is then set only by the position priors, weakly for
    stations metres apart - the first Belva test shrank it from 0.424 to
    0.14 m); False holds the rig fixed.  Frames with fewer than
    ``min_frame_observations`` observations are held at their current pose
    (on Belva two frames with no observations otherwise rotated freely, by 9
    and 36 deg).  ``refine_tangential`` (v0p14.3; default True since v0p20):
    refine p1, p2 of OPENCV / FULL_OPENCV cameras; otherwise they stay at
    their initial values.  ``attitude_prior_deg`` (v0p20, default 1 deg):
    a weak prior on every frame's attitude (the CAHV pointing, 1-sigma per
    axis).  Without it the block's orientation is held only by the position
    priors; with few, nearly collinear stations it is free to rotate about
    the line through them (0.6 deg at Three Forks, 2.4 deg with Mastcam-Z).
    Relative orientations come from the tie points and are not affected.
    None or 0 switches it off.  ``rig_translation_sigma_m`` (v0p22): with
    ``refine_rig=True``, a prior (this 1-sigma per axis, metres) holding the
    right camera's centre in the rig near its CAHV value, so the data can move
    the stereo baseline without the scale being left to the position priors
    alone.  ``covariance`` (v0p35): after the solve, the covariance of the
    rig and camera blocks with the 3-D points eliminated
    (``pycolmap.estimate_ba_covariance_from_problem``), in the tangent space of
    their manifolds (only the refined parameters) and for the assumed
    ``sigma_px``; ``variance_factor`` (2 cost / redundancy) rescales it to the
    residuals.  Returned under ``"covariance"``.  ``keypoint_scale`` (v0p35):
    {image_id: s} - that image's keypoints are divided by ``s`` about its
    camera's principal point before the adjustment, which is exactly a camera
    whose fx and fy are ``s`` times the shared camera's (the Navcam thermal
    model f(T) = f0 (1 + b (T - T0)) with one shared camera at T0).
    ``localize_min_images`` (v0p35.1; None: ``project.settings["localize_min_images"]``, default 0 = off): stations
    (site, drive) with fewer images than this get no waypoint position prior - short stops during a drive, whose
    waypoint positions can be metres off (Seitah North: four two-image drives 3 m off pulled the block's scale by
    8.6 %) - see :func:`unlocalized_stations`.  Their attitude prior stays.
    ``keypoint_map`` (v0p35): a function (image_id, keypoints, camera) ->
    keypoints applied after the scaling (e.g. the rig's temperature
    dependence as a small rotation of the right camera's rays).  Returns
    the solver summary as a dict.
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
    # v0p30: everything that is the same for all observations of an image is looked up once per image
    # (the cost constructors and add_residual_block are what remains in the loop: ~2x faster setup)
    per_image: Dict[int, tuple] = {}
    for iid in reg:
        im = rec.images[iid]
        fr = work.frames[im.frame_id]
        rig = rec.rigs[fr.rig_id]
        cam = work.cameras[im.camera_id]
        cov = np.eye(2) * (sigma_px / scale[iid]) ** 2
        pose = pose_blocks.get(im.frame_id)
        if pose is None:
            pose = pose_blocks[im.frame_id] = fr.rig_from_world.params
        sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=im.camera_id)
        kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
        if keypoint_scale and iid in keypoint_scale and keypoint_scale[iid] != 1.0:
            c0 = np.asarray(cam.params[2:4], float)
            kps = c0 + (kps - c0) / float(keypoint_scale[iid])
        if keypoint_map is not None and len(kps):
            kps = keypoint_map(iid, kps, cam)
        if rig.is_ref_sensor(sensor):
            per_image[iid] = (cam.model, cov, kps, [pose, cam.params], None)
        else:
            per_image[iid] = (cam.model, cov, kps, [rig_blocks[(fr.rig_id, im.camera_id)], pose, cam.params], True)
    reproj, rig_reproj, add = cf.ReprojErrorCost, cf.RigReprojErrorCost, prob.add_residual_block
    for pid, pt in rec.points3D.items():
        xyz = pt.xyz
        for el in pt.track.elements:
            info = per_image.get(el.image_id)
            if info is None:
                continue
            model, cov, kps, blocks, rigged = info
            xy = kps[el.point2D_idx]
            if rigged:
                add(rig_reproj(model, cov, xy), loss, [xyz, *blocks])
            else:
                add(reproj(model, cov, xy), loss, [xyz, *blocks])
            n_obs += 1

    # intrinsics: hold k4-k6 at zero, p1, p2 at their start unless refine_tangential (and the principal point if asked);
    # a camera may hold more ("fixed_params" in project.cameras, e.g. the Mastcam-Z focus bins, v0p14.4) or
    # refine more ("free_params", e.g. k4 of the rational Navcam cameras, v0p20)
    from .project import FULL_OPENCV_NAMES
    key_of = {int(v): k for k, v in project.settings.get("database", {}).get("cameras", {}).items()}
    if not key_of and any(c.get("free_params") or c.get("fixed_params") for c in project.cameras.values()):
        raise RuntimeError("the project has no database camera mapping (run build_database first): per-camera "
                           "free_params / fixed_params (rational Navcam k4, Mastcam-Z focus bins) would be ignored")
    for cid, cam in work.cameras.items():
        if not prob.has_parameter_block(cam.params):
            continue
        if not refine_intrinsics or key_of.get(int(cid), "") in set(hold_cameras):
            prob.set_parameter_block_constant(cam.params)
            continue
        fixed = fixed_camera_params(cam.model.name, refine_principal_point, refine_tangential)
        pc = project.cameras.get(key_of.get(int(cid), ""), {})
        names = _PARAM_NAMES.get(cam.model.name, FULL_OPENCV_NAMES)
        extra = pc.get("fixed_params") or []
        if extra and cam.model.name in _PARAM_NAMES:
            fixed = sorted(set(fixed) | {names.index(n) for n in extra if n in names and names.index(n) < len(cam.params)})
        free = pc.get("free_params") or []            # v0p20: e.g. the rational Navcam k4; v0p30: any model
        if free and cam.model.name in _PARAM_NAMES:
            fixed = sorted(set(fixed) - {names.index(n) for n in free if n in names})
        prob.set_parameter_block_variable(cam.params)
        if fixed:
            prob.set_manifold(cam.params, pyceres.SubsetManifold(len(cam.params), sorted(fixed)))

    # waypoint position priors on every frame (the reference camera's centre); v0p35.1: not on the frames of
    # stations with fewer than localize_min_images images, if they are tied to the rest of the block
    n_prior = n_att = 0
    if localize_min_images is None:
        localize_min_images = int(project.settings.get("localize_min_images") or 0)
    unloc = unlocalized_stations(project, localize_min_images, rec=rec) if localize_min_images else {}
    no_prior = {st for st, v in unloc.items() if v["prior"] == "dropped"}
    n_no_prior = 0
    if use_priors:
        sig = np.asarray(project.settings.get("prior_sigma_m", [1.0, 1.0, 1.0]), float)
        cov3 = np.diag(sig ** 2)
        by_name = {r["name"]: r for r in project.images}
        for fid, fr in work.frames.items():
            rig = rec.rigs[fr.rig_id]
            ref = [d.id for d in fr.data_ids if d.sensor_id.id == rig.ref_sensor_id.id]
            if not ref or ref[0] not in rec.images:
                continue
            pose = pose_blocks.get(fid)
            r_ref = by_name.get(rec.images[ref[0]].name)
            if pose is None or r_ref is None:             # unregistered frame, or an image the project does not know
                continue
            C = np.asarray(r_ref["prior_C"], float)
            if r_ref.get("station") in no_prior:
                n_no_prior += 1
            else:
                prob.add_residual_block(cf.AbsolutePosePositionPriorCost(cov3, C), None, [pose])
                n_prior += 1
            if attitude_prior_deg:
                # 6-DoF prior: rotation block first (checked), translation effectively free (the position
                # prior above holds the centre)
                ref_r = by_name[rec.images[ref[0]].name]
                T0 = _rigid(ref_r["prior_R_w2c"], ref_r["prior_C"])
                cov6 = np.diag([np.radians(float(attitude_prior_deg)) ** 2] * 3 + [1e8] * 3)
                prob.add_residual_block(cf.AbsolutePosePriorCost(cov6, T0), None, [pose])
                n_att += 1

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

    rig_start = {k: v.copy() for k, v in rig_blocks.items()}
    n_rig_prior = 0
    for key, arr in rig_blocks.items():
        if not prob.has_parameter_block(arr):
            continue
        if refine_rig is True and rig_translation_sigma_m:
            from scipy.spatial.transform import Rotation
            q0, t0 = rig_start[key][:4], rig_start[key][4:7]
            C0 = -Rotation.from_quat(q0 / np.linalg.norm(q0)).as_matrix().T @ t0     # right centre in the rig frame
            prob.add_residual_block(cf.AbsolutePosePositionPriorCost(np.eye(3) * float(rig_translation_sigma_m) ** 2, C0),
                                    None, [arr])
            n_rig_prior += 1
        if refine_rig:
            # pyceres has no product manifold and takes no Python manifolds.  The rig
            # rotation is small (Navcam L->R ~0.13 deg), so its quaternion is refined in
            # x, y, z with w held: |q|^2 - 1 stays O(1e-6), i.e. a scale error < 0.01 px;
            # the quaternion is normalised when written back.
            prob.set_manifold(arr, pyceres.SubsetManifold(7, [3, 4, 5, 6] if refine_rig == "rotation" else [3]))
        else:
            prob.set_parameter_block_constant(arr)

    so = pyceres.SolverOptions()
    # v0p30: the linear solver.  The Schur complement has one 6-parameter block per frame plus the cameras, so
    # for the few hundred frames of a scape a dense Schur solve (multi-threaded LAPACK) beats the sparse
    # Cholesky, whose factorisation is single-threaded; large blocks (thousands of frames) get the iterative
    # Schur solver with a Schur-Jacobi preconditioner, which parallelises over the observations.
    n_frames = len(pose_blocks)
    solver = linear_solver if linear_solver != "auto" else ("dense_schur" if n_frames <= 600 else "iterative_schur")
    so.linear_solver_type = {"sparse_schur": pyceres.LinearSolverType.SPARSE_SCHUR,
                             "dense_schur": pyceres.LinearSolverType.DENSE_SCHUR,
                             "iterative_schur": pyceres.LinearSolverType.ITERATIVE_SCHUR}[solver]
    if solver == "iterative_schur":
        so.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
    so.max_num_iterations = int(max_iterations)
    so.num_threads = int(num_threads) if num_threads > 0 else (__import__("os").cpu_count() or 1)
    so.minimizer_progress_to_stdout = bool(verbose)
    summary = pyceres.SolverSummary()
    pyceres.solve(so, prob, summary)
    cov_out = None
    if covariance:
        cov_out = _block_covariances(rec, prob, work, rig_blocks, key_of, summary, n_obs, n_prior, n_att, pose_blocks)
    for fid in pose_blocks:
        rec.frames[fid].rig_from_world = work.frames[fid].rig_from_world
    for cid, cam in work.cameras.items():
        rec.cameras[cid].params = np.array(cam.params)
    for (rid, cid), arr in rig_blocks.items():
        q = arr[:4] / np.linalg.norm(arr[:4])
        from scipy.spatial.transform import Rotation
        T = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_quat(q).as_matrix()), arr[4:7].copy())
        rec.rigs[rid].set_sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid), T)
    rig_out = {}
    for (rid, cid), arr in rig_blocks.items():
        from scipy.spatial.transform import Rotation
        R1 = Rotation.from_quat(arr[:4] / np.linalg.norm(arr[:4])).as_matrix()
        R0 = Rotation.from_quat(rig_start[(rid, cid)][:4] / np.linalg.norm(rig_start[(rid, cid)][:4])).as_matrix()
        rig_out[f"{rid}:{cid}"] = {"baseline_m": float(np.linalg.norm(arr[4:7])),
                                   "baseline_start_m": float(np.linalg.norm(rig_start[(rid, cid)][4:7])),
                                   "centre_change_m": (-(R1.T @ arr[4:7]) + R0.T @ rig_start[(rid, cid)][4:7]).tolist(),
                                   "rotation_change_deg": float(np.degrees(np.linalg.norm(
                                       Rotation.from_matrix(R1 @ R0.T).as_rotvec())))}
    return {"observations": n_obs, "priors": n_prior, "attitude_priors": n_att, "rig": rig_out, "covariance": cov_out,
            "rig_translation_priors": n_rig_prior, "frames_without_position_prior": n_no_prior,
            "unlocalized_stations": unloc,
            "attitude_prior_deg": float(attitude_prior_deg or 0.0), "frames_held": held, "initial_cost": summary.initial_cost,
            "final_cost": summary.final_cost, "iterations": summary.num_successful_steps + summary.num_unsuccessful_steps,
            "termination": str(summary.termination_type), "brief": summary.BriefReport(),
            "refine_tangential": bool(refine_tangential), "linear_solver": solver, "num_threads": so.num_threads,
            "seconds": float(summary.total_time_in_seconds)}


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
                      reference: Optional[str] = None, verbose: bool = True,
                      only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
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
    registered stay at their priors (reported).  ``only`` (v0p35.1): move just
    these stations; every other station counts as placed from the start (used
    for the stations without a position prior, ``LOCALIZE_MIN_IMAGES``).
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
    if only is not None:
        placed = {s for s in st_images if s not in set(only)}
        if not placed:
            return {"reference": None, "stations": {}, "unplaced": sorted(st_images), "only": sorted(only)}
        ref = reference or max(placed, key=lambda k: n_pts.get(k, 0))
        report = {"reference": ref, "stations": {}, "only": sorted(only)}
    else:
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
        if only is not None and s not in set(only):
            continue
        report["stations"].setdefault(s, {"registered": s in placed, "correspondences": 0})
    if only is None:
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


OUTLIER_DEFAULTS = {"residual_factor": 3.0, "min_residual_px": 1.0, "shift_mad_factor": 5.0, "min_shift_m": 0.25,
                    "attitude_mad_factor": 5.0, "min_attitude_deg": 0.5, "min_observations": 30}


def find_outlier_frames(rec, project: SfmProject, **thresholds) -> List[Dict[str, Any]]:
    """
    Frames (a Navcam stereo pair, or one Mastcam-Z image) that do not align with
    the majority (v0p22).  A frame is flagged when

    * its median residual exceeds ``residual_factor`` x the median over all
      frames and ``min_residual_px`` (native px);
    * it has fewer than ``min_observations`` tie-point observations (such a
      frame is held at its prior in the bundle adjustment, i.e. not aligned);
    * its camera moved from its prior differently from the other frames of its
      station and instrument family (a rover station is one rover pose): the
      shift deviates from the station median by more than ``min_shift_m`` and
      ``shift_mad_factor`` x the station's scaled MAD, or the attitude change
      by more than ``min_attitude_deg`` and ``attitude_mad_factor`` x MAD.
      Stations with fewer than 3 frames of a family are not tested this way.
      The attitude test applies to Navcam frames only: a Mastcam-Z prior
      attitude is one mast pointing, whose error is per image (v0p22.1).

    Returns one record per flagged frame: frame id, images, station, reasons,
    and ``kind``: ``"unconstrained"`` for a frame with no tie points at all
    (e.g. a defocused focus-stack member or a calibration-target image; it was
    never aligned and is left out of the export) or ``"outlier"``.
    """
    from scipy.spatial.transform import Rotation
    th = dict(OUTLIER_DEFAULTS, **thresholds)
    by_name = {r["name"]: r for r in project.images}
    reg = set(rec.reg_image_ids())
    res = native_residuals(rec, project)
    med_img: Dict[int, float] = {}
    n_img: Dict[int, int] = {}
    order = np.argsort(res["image_id"], kind="stable")
    for sel in np.split(order, np.flatnonzero(np.diff(res["image_id"][order])) + 1):
        if sel.size:
            iid = int(res["image_id"][sel[0]])
            r = res["residual_native_px"][sel]
            med_img[iid] = float(np.median(r[np.isfinite(r)])) if np.isfinite(r).any() else float("inf")
            n_img[iid] = int(sel.size)
    frames: Dict[int, Dict[str, Any]] = {}
    for fid, fr in rec.frames.items():
        ims = [d.id for d in fr.data_ids if d.id in reg]
        if not ims:
            continue
        rows = [by_name[rec.images[i].name] for i in ims]
        ref = ims[0]
        im = rec.images[ref]
        R = im.cam_from_world().rotation.matrix()
        C = im.projection_center()
        r0 = by_name[im.name]
        dC = np.asarray(C) - np.asarray(r0["prior_C"], float)
        dR = Rotation.from_matrix(np.asarray(r0["prior_R_w2c"], float).T @ R).as_rotvec()   # world frame: a rover tilt is common to a station
        obs = sum(n_img.get(i, 0) for i in ims)
        meds = [med_img[i] for i in ims if i in med_img]
        frames[fid] = {"frame_id": int(fid), "images": [rec.images[i].name for i in ims], "station": rows[0]["station"],
                       "family": "Z" if str(rows[0]["instrument"]).startswith("Z") else "N", "observations": obs,
                       "median_residual_px": float(np.median(meds)) if meds else float("inf"), "dC": dC, "dR": dR,
                       "reasons": []}
    if not frames:
        return []
    all_med = np.median([f["median_residual_px"] for f in frames.values() if np.isfinite(f["median_residual_px"])])
    lim = max(th["residual_factor"] * all_med, th["min_residual_px"])
    for f in frames.values():
        if f["observations"] and f["median_residual_px"] > lim:
            f["reasons"].append(f"median residual {f['median_residual_px']:.2f} px > {lim:.2f}")
        if f["observations"] == 0:
            f["reasons"].append("no tie points")
        elif f["observations"] < th["min_observations"]:
            f["reasons"].append(f"{f['observations']} observations < {th['min_observations']}")
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for f in frames.values():
        groups.setdefault((f["station"], f["family"]), []).append(f)
    for fs in groups.values():
        if len(fs) < 3:
            continue
        D = np.array([f["dC"] for f in fs])
        dev = np.linalg.norm(D - np.median(D, axis=0), axis=1)
        mad = 1.4826 * np.median(dev)
        A = np.array([f["dR"] for f in fs])
        adev = np.degrees(np.linalg.norm(A - np.median(A, axis=0), axis=1))
        amad = 1.4826 * np.median(adev)
        for f, d, a in zip(fs, dev, adev):
            if d > max(th["min_shift_m"], th["shift_mad_factor"] * mad):
                f["reasons"].append(f"shift {d:.2f} m from its station's median")
            if f["family"] == "N" and a > max(th["min_attitude_deg"], th["attitude_mad_factor"] * amad):
                f["reasons"].append(f"attitude {a:.2f} deg from its station's median")
    out = []
    for f in frames.values():
        if f["reasons"]:
            rec_ = {k: v for k, v in f.items() if k not in ("dC", "dR")}
            rec_["kind"] = "unconstrained" if f["observations"] == 0 else "outlier"
            out.append(rec_)
    return sorted(out, key=lambda f: f["frame_id"])


def exclude_frames(rec, frame_ids: Sequence[int]) -> int:
    """Deregister frames (their observations and single-frame points go); returns the number removed."""
    n = 0
    reg = set(rec.reg_frame_ids())
    for fid in frame_ids:
        if int(fid) in reg:
            reg.discard(int(fid))
            rec.deregister_frame(int(fid))
            n += 1
    return n


def camera_changes(rec, project: SfmProject) -> List[Dict[str, Any]]:
    """
    Start and refined parameters of every camera of a solved block (v0p35.1), for the notebook 03 print-out.
    Cameras are found by their database id (``settings["database"]["cameras"]``), not through the images: after
    the thermal stage every image of a binned frame belongs to its bin camera, so an eye's camera can have no image
    left (it then holds the values refined before the split).  Mastcam-Z focus bins are left out (section 7b);
    thermal bins are included with ``thermal_bin`` set.  A camera missing from ``rec`` gets ``missing`` set.
    """
    db = {k: int(v) for k, v in project.settings.get("database", {}).get("cameras", {}).items()}
    for r in project.images:                      # fallback for projects without the database table
        if "camera_id" in r:
            db.setdefault(r.get("base_instrument", r["instrument"]), int(r["camera_id"]))
    reg = set(rec.reg_image_ids())
    n_img: Dict[int, int] = {}
    for iid in reg:
        cid = int(rec.images[iid].camera_id)
        n_img[cid] = n_img.get(cid, 0) + 1
    rows = []
    for k, c in project.cameras.items():
        if c.get("group") and not c.get("thermal_bin"):
            continue
        cid = db.get(k)
        if cid is None or cid not in rec.cameras:
            rows.append({"camera": k, "missing": True})
            continue
        cam = rec.cameras[cid]
        names = _PARAM_NAMES.get(cam.model.name, tuple(f"p{i}" for i in range(len(cam.params))))
        rows.append({"camera": k, "camera_id": cid, "model": cam.model.name, "names": list(names),
                     "start": [float(x) for x in c["params"]], "refined": [float(x) for x in cam.params],
                     "images": int(n_img.get(cid, 0)), "thermal_bin": bool(c.get("thermal_bin")), "missing": False})
    return rows


def print_camera_changes(rec, project: SfmProject, n_params: int = 10) -> List[Dict[str, Any]]:
    """Print :func:`camera_changes` (start -> refined, first ``n_params`` parameters).  Returns the rows."""
    rows = camera_changes(rec, project)
    for r in rows:
        if r["missing"]:
            print(f"{r['camera']}: not in the reconstruction")
            continue
        tag = ("  [temperature bin]" if r["thermal_bin"] else
               "  [no images after the thermal split: values before it]" if r["images"] == 0 else "")
        print(f"{r['camera']} ({r['images']} images){tag}: " + " ".join(
            f"{n} {a:.4g}->{b:.4g}" for n, a, b in zip(r["names"][:n_params], r["start"], r["refined"])))
    return rows


CONVERGENCE_BINS_DEG = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 180.0)


def convergence_statistics(rec, project: SfmProject, bins_deg: Sequence[float] = CONVERGENCE_BINS_DEG,
                           max_points: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """
    Tie points by convergence angle (v0p22): for each 3-D point the largest
    angle between two of its viewing rays.  Counts per bin for all points and
    for cross-station points, the number of points and observations above 10
    and 20 deg, and the median.  ``max_points``: a random sample for speed.
    """
    st = {r["name"]: r["station"] for r in project.images}
    C = {iid: np.asarray(im.projection_center()) for iid, im in rec.images.items() if im.has_pose}
    pids = list(rec.points3D)
    if max_points and len(pids) > max_points:
        pids = list(np.random.default_rng(seed).choice(pids, max_points, replace=False))
    ang, cross, nobs = [], [], []
    for pid in pids:
        pt = rec.points3D[int(pid)]
        ids = [el.image_id for el in pt.track.elements if el.image_id in C]
        if len(ids) < 2:
            continue
        d = np.array([pt.xyz - C[i] for i in ids])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = np.clip(d @ d.T, -1, 1)
        ang.append(float(np.degrees(np.arccos(c.min()))))
        cross.append(len({st[rec.images[i].name] for i in ids}) > 1)
        nobs.append(len(ids))
    ang, cross, nobs = np.array(ang), np.array(cross, bool), np.array(nobs)
    b = np.asarray(bins_deg, float)
    f = 1.0 if not max_points or len(rec.points3D) <= (max_points or 0) else len(rec.points3D) / max(len(pids), 1)
    return {"bins_deg": b.tolist(), "points": (np.histogram(ang, b)[0] * f).round().astype(int).tolist(),
            "cross_station_points": (np.histogram(ang[cross], b)[0] * f).round().astype(int).tolist(),
            "points_over_10deg": int(round(f * np.sum(ang > 10))), "points_over_20deg": int(round(f * np.sum(ang > 20))),
            "observations_over_10deg": int(round(f * nobs[ang > 10].sum())),
            "median_angle_deg": float(np.median(ang)) if ang.size else float("nan"),
            "median_angle_cross_deg": float(np.median(ang[cross])) if cross.any() else float("nan"),
            "sampled": bool(f != 1.0)}


NETWORK_MIN_STATIONS = 4        # Navcam intrinsics are refined only from networks at least this strong (v0p22.2)
NETWORK_MIN_SPAN_M = 5.0


def navcam_network(project: SfmProject, min_stations: int = NETWORK_MIN_STATIONS,
                   min_span_m: float = NETWORK_MIN_SPAN_M) -> Dict[str, Any]:
    """
    How well the Navcam network can constrain the Navcam intrinsics (v0p22.2):
    the number of stations with Navcam frames, the span of their prior
    positions, and the verdict ``"strong"`` (refine) or ``"weak"`` (hold the
    cameras at their start values).  A scape whose Navcam images all stand at
    a few stations a few metres apart cannot separate the principal point
    from the attitudes nor the distortion from the depths (methods §14).
    """
    st: Dict[str, List[np.ndarray]] = {}
    for r in project.images:
        if str(r["instrument"]).startswith("N"):
            st.setdefault(r["station"], []).append(np.asarray(r["prior_C"], float))
    n_st = sum(1 for v in st.values() if len(v) >= 2)
    C = np.array([np.mean(v, axis=0) for v in st.values()]) if st else np.zeros((0, 3))
    span = float(np.max(np.linalg.norm(C[:, None, :2] - C[None, :, :2], axis=2))) if len(C) > 1 else 0.0
    weak = n_st < min_stations or span < min_span_m
    return {"stations": n_st, "span_m": round(span, 2), "navcam_images": sum(len(v) for v in st.values()),
            "verdict": "weak" if weak else "strong", "min_stations": min_stations, "min_span_m": min_span_m}


def _frames_of_family(rec, project: SfmProject, family: str) -> List[int]:
    by_name = {r["name"]: r for r in project.images}
    out = []
    for fid, fr in rec.frames.items():
        names = [rec.images[d.id].name for d in fr.data_ids if d.id in rec.images]
        if names and str(by_name[names[0]]["instrument"]).startswith(family):
            out.append(int(fid))
    return out


def reconstruct(project: SfmProject, sigma_px: float = 0.5,
                schedule: Sequence[Sequence[float]] = DEFAULT_SCHEDULE,
                refine_intrinsics: bool = True, refine_rig: Union[bool, str] = "rotation",
                register: bool = False, max_iterations: int = 50, out_name: str = "cahv_ba",
                verbose: bool = True, min_track_length: int = 2, min_tri_angle_deg: float = 0.25,
                refine_tangential: bool = True, gui_native: bool = True,
                attitude_prior_deg: Optional[float] = ATTITUDE_PRIOR_DEG,
                exclude_outliers: bool = False, outlier_thresholds: Optional[Dict[str, float]] = None,
                exclude_after_round: int = 2, rig_translation_sigma_m: Optional[float] = None,
                triangulation_options: Optional[Dict[str, Any]] = None,
                navcam_intrinsics: str = "refine", staged: bool = False, hold_cameras: Sequence[str] = (),
                linear_solver: str = "auto", temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
                thermal_bins_deg: Optional[float] = None, thermal_min_images: int = 8,
                thermal_free: Sequence[str] = ("fx", "fy"),
                thermal_model: Optional[Dict[str, Dict[str, float]]] = None, localize_min_images: int = 0):
    """
    CAHV-initialised triangulation + weighted BA (see module docstring).

    ``localize_min_images`` (v0p35.1, notebook 03 ``LOCALIZE_MIN_IMAGES``): stations (site, drive) with fewer images
    than this are not held by their waypoint position (see :func:`unlocalized_stations`, :func:`bundle_adjust`).
    Before the first round they are moved onto the rest of the block as rigid stations
    (:func:`register_stations` with ``only``), so that triangulation can join their tracks; then every adjustment
    leaves out their position priors.  Recorded in ``project.settings["localize_min_images"]`` and
    ``project.settings["reconstruction"]["unlocalized_stations"]``.  0 (default) keeps every prior.

    ``navcam_intrinsics`` (v0p22.2): ``"refine"`` (default since v0p22.4: the
    Navcam intrinsics are always refined), ``"hold"`` (the Navcam cameras stay
    at their start values, the shipped consensus unless the project was created
    from other cameras) or ``"auto"`` (hold when :func:`navcam_network` calls
    the network weak).  The network verdict is logged whatever the setting.  ``refine_rig="auto"`` (v0p35)
    refines the rig rotation on a strong network and holds it (at the start rig, turned to the block's camera
    temperature when the start rig has a temperature model) on a weak one.  ``refine_rig=False``
    holds the rig.  ``hold_cameras``: further camera keys to hold.
    ``staged`` (v0p22.2): with Mastcam-Z in the project, solve the Navcam
    block first (Mastcam-Z frames deregistered), write it to
    ``sparse/<out_name>_navcam``, then add the Mastcam-Z frames at their
    priors and solve everything with the Navcam cameras and rig held at the
    stage-1 values.  Both solutions come from the same images, features and
    matches, so the Navcam-only and the combined results are comparable, and
    the Navcam calibration cannot be pulled by the Mastcam-Z observations.
    The stage-1 solution stays on disk (``sparse/<out_name>_navcam``) and its
    log in ``project.settings["reconstruction"]["navcam_stage"]``; export it
    with ``export_for_error(project, rec_navcam, out_dir=...)``.

    ``schedule``: rounds of (triangulation threshold [full-res px], Cauchy
    scale [sigma], maximum residual kept [native px]) - a graduated schedule
    so that observations far from the initial poses can still pull them.
    Default: four rounds; the fourth repeats the third's limits (v0p14.5) to
    show whether one more re-triangulation and adjustment changes anything.
    The final adjustment uses the last round's Cauchy scale and cut-off.
    ``attitude_prior_deg`` (v0p20, default 1 deg): weak CAHV attitude prior
    per frame; see :func:`bundle_adjust`.
    ``gui_native`` (v0p14.5): also write the native-pixel viewing copy
    ``gui_native/`` for the COLMAP GUI (``open_in_colmap.bat``).
    ``register``: first move whole stations with :func:`register_stations`
    (needed when the waypoint priors are off by more than ~20 px at range;
    at Belva they are within ~1-2 px, so it is off by default).
    ``min_track_length``: shortest track kept (v0p13).  Default 2: two-view
    tie points - e.g. seen only by the two eyes of one stereo pair - are kept,
    as COLMAP's ``ignore_two_view_tracks=False`` triangulates them; 3 drops
    them after every round.
    ``min_tri_angle_deg``: smallest triangulation angle (default 0.25 deg
    since v0p20; 0.5 from v0p14.3, 1.5 before).  With the 0.424 m Navcam
    baseline, 0.25 deg keeps stereo-only points out to about 97 m (0.5 deg:
    49 m, 1.5 deg: 16 m).  Their depth is
    poorly constrained, their direction is not: they tie attitudes, and the
    weighting and the residual cut-offs limit their influence.
    ``refine_tangential`` (default True since v0p20): refine p1, p2 (see
    :func:`bundle_adjust`); they start from the project's cameras (the
    calibration's values unless the project was created with p1, p2 in
    ``zero_terms``).
    ``exclude_outliers`` (v0p22): after round ``exclude_after_round`` and
    again before the final adjustment, frames that do not align with the
    majority (:func:`find_outlier_frames`, ``outlier_thresholds``) are
    deregistered; the rounds after that re-triangulate without them.  The
    list is in the log and ``project.settings["reconstruction"]["excluded"]``.
    ``rig_translation_sigma_m``: see :func:`bundle_adjust` (with ``refine_rig=True``).
    ``linear_solver`` (v0p30): passed to :func:`bundle_adjust`.
    ``thermal_bins_deg`` (v0p31, with ``temperatures`` from :func:`mppp.sfm.thermal.image_temperatures`): after the
    final adjustment, split the Navcam cameras into camera-temperature bins of this width
    (:func:`mppp.sfm.thermal.thermal_stage`; bins with fewer than ``thermal_min_images`` images are merged) and
    adjust once more with the bins' ``thermal_free`` parameters refined - or, when the Navcam intrinsics are held,
    with every bin held at the eye's camera scaled to the bin temperature by ``thermal_model``
    ({"NL": {"ppm_per_degC", "T0_degC"}, ...}).  The binned solution is the delivered one; the solution with one
    camera per eye stays in ``sparse/<out_name>_single`` and the bin table in
    ``project.settings["thermal"]``.
    ``triangulation_options`` (v0p22): extra keyword arguments for
    :func:`triangulate` (e.g. ``min_angle_deg`` per round is set from
    ``min_tri_angle_deg``; ``create_max_angle_error``, ``complete``...).
    Returns the reconstruction (also written to ``sparse/<out_name>``).
    """
    if min_track_length < 2:
        raise ValueError("min_track_length must be >= 2")
    if navcam_intrinsics not in ("auto", "refine", "hold"):
        raise ValueError("navcam_intrinsics must be 'auto', 'refine' or 'hold'")
    import time
    from .thermal import strip_thermal_bins
    t0 = time.time()
    n_strip = strip_thermal_bins(project)                   # v0p31: a rerun starts from one camera per eye
    if n_strip and verbose:
        print(f"[sfm] {n_strip} Navcam temperature-bin cameras of an earlier run removed; starting from one camera "
              f"per eye", flush=True)
    rec = initial_reconstruction(project)
    init = copy.deepcopy(rec)
    log: List[Dict[str, Any]] = []
    project.settings["localize_min_images"] = int(localize_min_images or 0)
    unloc = unlocalized_stations(project, localize_min_images) if localize_min_images else {}
    unloc_reg = None
    if unloc and any(v["prior"] == "dropped" for v in unloc.values()):
        # v0p35.1: place the short stops on the block before triangulating with the round-1 threshold
        rec = triangulate(rec, project, max_reproj_px=8.0, min_angle_deg=min_tri_angle_deg)
        unloc_reg = register_stations(rec, project, only=sorted(unloc), verbose=verbose)
        log.append({"unlocalized_registration": unloc_reg})
        if verbose:
            print(f"[sfm] LOCALIZE_MIN_IMAGES {localize_min_images}: {len(unloc)} stations without a position prior "
                  f"({', '.join(sorted(unloc))}); registered onto the block: "
                  f"{sum(1 for v in unloc_reg['stations'].values() if v.get('registered'))}", flush=True)
    network = navcam_network(project)
    hold = set(hold_cameras)
    if navcam_intrinsics == "hold" or (navcam_intrinsics == "auto" and network["verdict"] == "weak"):
        hold |= {k for k in project.cameras if str(k).startswith("N")}
    network["navcam_intrinsics"] = "hold" if any(str(k).startswith("N") for k in hold) else "refine"
    if refine_rig == "auto":                                # v0p35: the rig follows the intrinsics' network verdict
        refine_rig = "rotation" if network["verdict"] == "strong" else False
    network["navcam_rig"] = "rotation" if refine_rig else "hold"
    if verbose:
        print(f"[sfm] Navcam network: {network['stations']} stations over {network['span_m']} m -> {network['verdict']}; "
              f"Navcam intrinsics {network['navcam_intrinsics']}" + (f", holding {sorted(hold)}" if hold else ""), flush=True)
    log.append({"navcam_network": network})
    z_frames = _frames_of_family(rec, project, "Z")
    staged = bool(staged and z_frames and len(z_frames) < len(rec.frames))
    if staged:
        exclude_frames(rec, z_frames)
        if verbose:
            print(f"[sfm] stage 1: Navcam only ({len(z_frames)} Mastcam-Z frames set aside)", flush=True)
    if register:
        rec = triangulate(rec, project, max_reproj_px=8.0)
        reg = register_stations(rec, project, verbose=verbose)
        log.append({"station_registration": reg})
    excluded: List[Dict[str, Any]] = []
    topts = dict(triangulation_options or {})

    def _exclude(tag):
        found = find_outlier_frames(rec, project, **(outlier_thresholds or {}))
        new = [f for f in found if f["frame_id"] not in {e["frame_id"] for e in excluded}]
        if new:
            exclude_frames(rec, [f["frame_id"] for f in new])
            for f in new:
                f["when"] = tag
            excluded.extend(new)
        if verbose:
            print(f"[sfm] {tag}: {len(new)} frames excluded as outliers"
                  + ("".join(f"\n      {', '.join(f['images'])}: {'; '.join(f['reasons'])}" for f in new[:20])
                     + ("\n      ..." if len(new) > 20 else "")), flush=True)
        log.append({"excluded_frames": new, "when": tag})

    for k, (tpx, loss, rmax) in enumerate(schedule):
        rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg, **topts)
        st0 = track_statistics(rec, project)
        ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(loss), refine_intrinsics=refine_intrinsics,
                           refine_tangential=refine_tangential, refine_rig=refine_rig, max_iterations=max_iterations,
                           attitude_prior_deg=attitude_prior_deg, rig_translation_sigma_m=rig_translation_sigma_m,
                           hold_cameras=sorted(hold), linear_solver=linear_solver)
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
        if exclude_outliers and k + 1 == min(int(exclude_after_round), len(schedule)):
            _exclude(f"after round {k + 1}")
    if exclude_outliers:
        _exclude("before the final adjustment")
    ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(schedule[-1][1]),
                       refine_intrinsics=refine_intrinsics, refine_tangential=refine_tangential,
                       refine_rig=refine_rig, max_iterations=2 * max_iterations, attitude_prior_deg=attitude_prior_deg,
                       rig_translation_sigma_m=rig_translation_sigma_m, hold_cameras=sorted(hold), linear_solver=linear_solver)
    # the final adjustment can push a few points behind a camera or past the limit
    n_bad = filter_observations(rec, project, float(schedule[-1][2]))
    n_bad += drop_short_tracks(rec, min_track_length)
    log.append({"final_ba": ba["brief"], "frames_held": ba["frames_held"], "filtered_after_final": n_bad,
                "tracks": track_statistics(rec, project)})
    navcam_stage = None
    if staged:
        # stage 1 done: keep it, then bring the Mastcam-Z frames in at their priors with the Navcam block held
        out1 = project.root / "sparse" / f"{out_name}_navcam"
        out1.mkdir(parents=True, exist_ok=True)
        rec.write(str(out1))
        (out1 / "mppp_sfm_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
        navcam_stage = {"path": str(out1.relative_to(project.root)), "log": log,
                        "tracks": track_statistics(rec, project), "cameras": {}}
        log = [{"stage": 2, "note": "Mastcam-Z frames added at their priors; Navcam cameras and rig held"}]
        for fid in z_frames:
            rec.frames[fid].rig_from_world = init.frames[fid].rig_from_world
            rec.register_frame(fid)
        hold |= {k for k in project.cameras if str(k).startswith("N")}
        refine_rig = False
        if verbose:
            print(f"[sfm] stage 2: {len(z_frames)} Mastcam-Z frames added; Navcam cameras and rig held", flush=True)
        for k, (tpx, loss, rmax) in enumerate(schedule):
            rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg, **topts)
            ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(loss), refine_intrinsics=refine_intrinsics,
                               refine_tangential=refine_tangential, refine_rig=False, max_iterations=max_iterations,
                               attitude_prior_deg=attitude_prior_deg, hold_cameras=sorted(hold), linear_solver=linear_solver)
            n_bad = filter_observations(rec, project, float(rmax)) + drop_short_tracks(rec, min_track_length)
            st = track_statistics(rec, project)
            res = native_residuals(rec, project)["residual_native_px"]
            entry = {"round": k + 1, "stage": 2, "triangulation_px_full": tpx, "cauchy_scale_sigma": loss, "ba": ba["brief"],
                     "filtered_observations": n_bad, "tracks": st,
                     "rms_native_px": float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan"),
                     "median_native_px": float(np.median(res)) if res.size else float("nan"),
                     "elapsed_s": round(time.time() - t0, 1)}
            log.append(entry)
            if verbose:
                print(f"[sfm] stage 2 round {k + 1}: {st['points']} points, median {entry['median_native_px']:.3f} / "
                      f"rms {entry['rms_native_px']:.3f} native px, {n_bad} obs filtered, {entry['elapsed_s']:.0f} s", flush=True)
            if exclude_outliers and k + 1 == min(int(exclude_after_round), len(schedule)):
                _exclude(f"stage 2, after round {k + 1}")
        if exclude_outliers:
            _exclude("stage 2, before the final adjustment")
        ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(schedule[-1][1]),
                           refine_intrinsics=refine_intrinsics, refine_tangential=refine_tangential, refine_rig=False,
                           max_iterations=2 * max_iterations, attitude_prior_deg=attitude_prior_deg, hold_cameras=sorted(hold), linear_solver=linear_solver)
        n_bad = filter_observations(rec, project, float(schedule[-1][2])) + drop_short_tracks(rec, min_track_length)
        log.append({"final_ba": ba["brief"], "frames_held": ba["frames_held"], "filtered_after_final": n_bad,
                    "tracks": track_statistics(rec, project)})
    thermal = None
    if thermal_bins_deg and temperatures:
        from .thermal import rig_slopes_for_project, thermal_stage
        single = project.root / "sparse" / f"{out_name}_single"
        single.mkdir(parents=True, exist_ok=True)
        rec.write(str(single))
        hold_nav = network["navcam_intrinsics"] == "hold" or any(str(k).startswith("N") for k in hold_cameras)
        rec, thermal = thermal_stage(rec, project, temperatures, bin_deg=float(thermal_bins_deg),
                                     min_images=int(thermal_min_images), free=tuple(thermal_free), hold=hold_nav,
                                     thermal_model=thermal_model, sigma_px=sigma_px,
                                     loss_scale=float(schedule[-1][1]), max_iterations=2 * max_iterations,
                                     attitude_prior_deg=attitude_prior_deg, linear_solver=linear_solver,
                                     verbose=verbose, rig_slopes=rig_slopes_for_project(project))
        n_bad = filter_observations(rec, project, float(schedule[-1][2])) + drop_short_tracks(rec, min_track_length)
        thermal["filtered_after"] = n_bad
        thermal["single_camera_model"] = str(single.relative_to(project.root))
        log.append({"thermal": {k: v for k, v in thermal.items() if k != "rows"}})
        project.settings["thermal"] = thermal
    elif thermal_bins_deg and verbose:
        print("[sfm] thermal stage skipped: no image temperatures", flush=True)
    out = project.root / "sparse" / out_name
    out.mkdir(parents=True, exist_ok=True)
    rec.write(str(out))
    idir = project.root / "sparse" / "cahv_initial"
    idir.mkdir(parents=True, exist_ok=True)
    init.write(str(idir))
    final_unloc = unlocalized_stations(project, localize_min_images, rec=rec) if localize_min_images else {}
    if final_unloc:
        log.append({"unlocalized_stations": final_unloc})
    (out / "mppp_sfm_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    project.settings["reconstruction"] = {"path": str(out.relative_to(project.root)), "log": log,
                                          "sigma_px_native": sigma_px, "refine_rig": refine_rig,
                                          "min_track_length": min_track_length,
                                          "min_tri_angle_deg": min_tri_angle_deg,
                                          "refine_tangential": bool(refine_tangential),
                                          "attitude_prior_deg": float(attitude_prior_deg or 0.0),
                                          "schedule": [list(map(float, r)) for r in schedule],
                                          "exclude_outliers": bool(exclude_outliers),
                                          "excluded": [{k: v for k, v in e.items()} for e in excluded],
                                          "rig_translation_sigma_m": rig_translation_sigma_m,
                                          "triangulation_options": topts,
                                          "navcam_network": network, "navcam_intrinsics": network["navcam_intrinsics"],
                                          "hold_cameras": sorted(hold), "staged": bool(staged),
                                          "navcam_stage": navcam_stage,
                                          "thermal_bins_deg": float(thermal_bins_deg) if thermal and thermal_bins_deg else None,
                                          "localize_min_images": int(localize_min_images or 0),
                                          "unlocalized_stations": sorted(st for st, v in final_unloc.items()
                                                                         if v["prior"] == "dropped"),
                                          "unlocalized": final_unloc, "unlocalized_registration": unloc_reg}
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
