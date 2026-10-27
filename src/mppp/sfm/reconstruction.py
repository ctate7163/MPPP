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
   camera lists them in ``free_params`` (k4 of the rational Navcam
   cameras); a Mastcam-Z focus-bin camera holds its ``fixed_params``.
4. Drop observations whose residual exceeds ``max_residual_native_px`` and
   re-triangulate with the refined poses; one round per ``schedule`` entry
   (four by default since v0p14.3), then a final adjustment.

The result is in the project's world frame (ENU metres minus ``project.offset``).
"""
from __future__ import annotations

import contextlib
import logging

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from scipy.spatial.transform import Rotation

import numpy as np

from .project import SfmProject

PathLike = Union[str, Path]

# FULL_OPENCV: fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6 -> indices held constant
_FIXED_EXTRA = {"FULL_OPENCV": [6, 7, 9, 10, 11], "OPENCV": [6, 7],
                "THIN_PRISM_FISHEYE": [6, 7, 10, 11]}          # p1, p2 (tangential) and sx1, sy1 (thin prism)
_TANGENTIAL = {"FULL_OPENCV": [6, 7], "OPENCV": [6, 7], "THIN_PRISM_FISHEYE": [6, 7]}         # p1, p2
from .project import PARAM_NAMES as _PARAM_NAMES      # one table (mppp.sfm.project)

logger = logging.getLogger("mppp.sfm")

# (triangulation threshold [full-res px], Cauchy scale [sigma], maximum residual kept [native px]) per round.
# the fourth round repeats the third's limits - it only shows whether another
# re-triangulation and adjustment still changes anything (v0p14.3-4 used (6, 1.5, 1.5)).
DEFAULT_SCHEDULE = ((24.0, 10.0, 8.0), (12.0, 2.0, 4.0), (8.0, 2.0, 2.0), (8.0, 2.0, 2.0))
ROUND1_HOLD = True            # round 1 adjusts poses and points only (cameras and rig at the promoted models)
ATTITUDE_PRIOR_DEG = 1.0      # weak CAHV attitude prior per frame (holds the block's orientation)
# Stations without a waypoint of their own - position = the site's drive-0 waypoint + the label's onboard
# ORIGIN_OFFSET_VECTOR (mid-drive / approach stops) - get looser priors than surveyed stations: at Rochette a 1 m /
# 1 deg prior held a station whose solve wanted 0.9 m and 1.5 deg of heading, and another moved 6.3 m against its
# 1 m prior.  Project settings unsurveyed_prior_sigma_m / unsurveyed_attitude_prior_deg override these.
UNSURVEYED_PRIOR_SIGMA_M = 10.0
UNSURVEYED_ATTITUDE_PRIOR_DEG = 3.0


def is_unsurveyed(row: Optional[Dict[str, Any]]) -> bool:
    """A project image row whose position prior is not a waypoint of its own station (``position_source`` "waypoint
    site N drive 0 + label ORIGIN_OFFSET_VECTOR", :meth:`mppp.image.MPPPImage._position`)."""
    src = str((row or {}).get("position_source") or "")
    return src.startswith("waypoint") and "ORIGIN_OFFSET_VECTOR" in src
K2_LIMIT = 10.0                  # v0p85 (Christian): a camera's k2 is kept within +-10 (set to +-10 when beyond)
BOGUS_MAX_EXTRA_PARAM = 1.0e6    # COLMAP's "bogus camera" limits off (default 1.0, 10, 0.1) - see triangulate()
BOGUS_MAX_FOCAL_RATIO = 1.0e3
BOGUS_MIN_FOCAL_RATIO = 1.0e-3
PRIOR_GAUGE_RECENTRE = True   # after every adjustment with position priors the whole block is moved onto them
                              # (the mean prior residual; reprojection-invariant) - Chal Rocks Large (17 Oct) ended
                              # -0.19 m E, -0.07 m N, -0.10 m U off its priors on average, which a free gauge with
                              # Gaussian priors cannot do at convergence


def fixed_camera_params(model_name: str, refine_principal_point: bool = True,
                        refine_tangential: bool = False) -> List[int]:
    """Indices of the camera parameters held constant in the bundle adjustment."""
    fixed = set(_FIXED_EXTRA.get(model_name, []))
    if refine_tangential:
        fixed -= set(_TANGENTIAL.get(model_name, []))
    if not refine_principal_point:
        fixed |= {2, 3}
    return sorted(fixed)


def tangential_for(refine_tangential: Union[bool, Dict[str, bool]], key: str = "") -> bool:
    """
    whether p1, p2 of the camera with project key ``key`` are refined.  ``refine_tangential`` is a bool
    (all cameras) or {family: bool} with the family the first letter of the key ("N": Navcam, including its thermal
    bins; "Z": Mastcam-Z, including its focus bins) and "default" for the rest (True if not given).
    """
    if isinstance(refine_tangential, dict):
        return bool(refine_tangential.get(str(key)[:1], refine_tangential.get("default", True)))
    return bool(refine_tangential)


def tangential_setting(refine_tangential: Union[bool, Dict[str, bool]]) -> Union[bool, Dict[str, bool]]:
    """``refine_tangential`` as recorded in the settings and results (a bool or {family: bool})."""
    if isinstance(refine_tangential, dict):
        return {str(k): bool(v) for k, v in refine_tangential.items()}
    return bool(refine_tangential)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


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
                continue_max_angle_error_deg: float = 2.0, complete_max_transitivity: int = 5, database: Optional[PathLike] = None):
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
    # COLMAP treats a camera with any distortion parameter beyond +-1 (max_extra_param) or a focal length
    # beyond 10x the image size (max_focal_length_ratio) as "bogus" and triangulates none of its images.  The
    # Mastcam-Z 34 mm FULL_OPENCV k2 is about 1.0 (ZR 1.016 on its focus line, bins to 2.2: a 2-3 px effect at the
    # corners, since r <= 0.22 in a narrow field), so in the 18-19 Oct run every ZR frame lost its tie points at the
    # first triangulation after its cameras were set on the line (Rochette diagnostic, 19 Oct: the frames that kept
    # tie points were exactly those of bins with |k| < 1).  Z110 has f ~ 9 x the width.  Our own checks judge cameras.
    opts.max_extra_param = BOGUS_MAX_EXTRA_PARAM
    opts.max_focal_length_ratio = BOGUS_MAX_FOCAL_RATIO
    opts.min_focal_length_ratio = BOGUS_MIN_FOCAL_RATIO
    for sub in (opts.triangulation, opts.mapper):
        sub.max_extra_param = BOGUS_MAX_EXTRA_PARAM
        sub.max_focal_length_ratio = BOGUS_MAX_FOCAL_RATIO
        sub.min_focal_length_ratio = BOGUS_MIN_FOCAL_RATIO
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
    import copy as _copy
    source_cameras = {int(c): _copy.deepcopy(cam) for c, cam in rec.cameras.items()}    # triangulate_points may
    rec = pycolmap.triangulate_points(rec, str(database or project.database), str(project.images_dir), str(out),
                                      clear_points=True, options=opts, refine_intrinsics=False)
    restore_shared_cameras(rec, source_cameras)          # change the model it is given in place
    settings = getattr(project, "settings", None) or {}
    drop_mixed_short_tracks(rec, int(settings.get("min_mixed_track_length") or MIN_MIXED_TRACK_LENGTH))
    return rec


def restore_shared_cameras(rec, cameras: Dict[int, Any]) -> List[int]:
    """
    Put back the cameras that ``triangulate_points`` dropped although ``rec`` still uses them (0.94.4, the
    weekend run of 23 Oct: Seitah Overlook and Artuby Ridge failed).  COLMAP's tear-down removes a rig without
    registered frames together with its sensors' cameras.  A right Hazcam (or Navcam) exposure without its left
    image sits in a one-camera rig of its own (``database.build``), so when that rig's last frame is excluded as an
    outlier, the right camera disappears while the stereo rig's frames still use it: "Camera with ID 2 does not exist"
    at the next step (Linux), "Camera model does not exist" or a native crash (Windows).  ``cameras``: copies of the
    cameras before the triangulation, {id: camera}.  Returns the camera ids put back.
    """
    import pycolmap
    need = {int(rec.images[i].camera_id) for i in rec.images}
    for rig in rec.rigs.values():
        need |= {int(x.id) for x in rig.sensor_ids() if x.type == pycolmap.SensorType.CAMERA}
    back = sorted(c for c in need if c not in rec.cameras and c in cameras)
    for cid in back:
        rec.add_camera(cameras[cid])
    if back:
        logger.info(f"triangulation: camera(s) {back} put back (dropped with a rig that has no registered frame "
                    "left, still used by another rig)")
    return back


def sync_database(rec, project: SfmProject, path: Optional[PathLike] = None) -> Path:
    """
    a copy of the project database whose cameras, rigs, frames and image cameras match ``rec``.
    ``triangulate_points`` loads the frames from the database and requires every frame of the model to sit on the
    same rig as in the database ("Check failed: existing_frame.RigId() == frame.RigId()").  After the Navcam
    temperature bins of stage 1 (``thermal_after_stage1``) the Navcam frames sit on new bin rigs with new bin
    cameras, so stage 2 triangulates against this copy (default ``<project>/database_thermal.db``); the project
    database itself is not changed.  Returns the path of the copy.
    """
    import shutil
    import pycolmap
    out = Path(path) if path else project.root / "database_thermal.db"
    if out.exists():
        out.unlink()
    shutil.copyfile(project.database, out)
    db = pycolmap.Database.open(str(out))
    try:
        for cid, cam in rec.cameras.items():
            if not db.exists_camera(int(cid)):
                db.write_camera(cam, use_camera_id=True)
        for rid, rig in rec.rigs.items():
            if not db.exists_rig(int(rid)):
                db.write_rig(rig, use_rig_id=True)
        for fid, fr in rec.frames.items():
            if not db.exists_frame(int(fid)):
                continue
            old = db.read_frame(int(fid))
            want = sorted((int(d.sensor_id.id), int(d.id)) for d in fr.data_ids)
            have = sorted((int(d.sensor_id.id), int(d.id)) for d in old.data_ids)
            if int(old.rig_id) != int(fr.rig_id) or want != have:
                nf = pycolmap.Frame(frame_id=int(fid), rig_id=int(fr.rig_id))
                for d in fr.data_ids:
                    nf.add_data_id(d)
                db.update_frame(nf)
        for iid, im in rec.images.items():
            if not db.exists_image(int(iid)):
                continue
            old = db.read_image(int(iid))
            if int(old.camera_id) != int(im.camera_id):
                old.camera_id = int(im.camera_id)
                db.update_image(old)
    finally:
        db.close()
    return out


def _scales(rec, project: SfmProject) -> Dict[int, float]:
    by_name = {r["name"]: float(r["downsample_scale"]) for r in project.images}
    # images the project does not describe (deregistered, e.g. frames dropped after a rebuild) are skipped
    return {iid: by_name[im.name] for iid, im in rec.images.items() if im.name in by_name}


def _set_camera_params(rec, cid: int, params) -> None:
    """``rec``'s camera ``cid`` gets ``params``, read back and checked (a silent no-op here once left the
    thermal-bin cameras of a block at their start values while the poses had moved on)."""
    p = np.array(params, dtype=np.float64)
    rec.cameras[cid].params = p
    if np.allclose(np.asarray(rec.cameras[cid].params, float), p, rtol=0, atol=1e-9):
        return
    rec.camera(cid).params = p
    if np.allclose(np.asarray(rec.cameras[cid].params, float), p, rtol=0, atol=1e-9):
        return
    raise RuntimeError(f"camera {cid}: the refined parameters could not be written into the reconstruction")


def _block_covariances(rec, prob, work, rig_blocks, key_of, summary, n_obs, n_prior, n_att,
                       pose_blocks: Optional[Dict[int, np.ndarray]] = None,
                       cam_blocks: Optional[Dict[int, np.ndarray]] = None,
                       extra_blocks: Optional[Dict[str, np.ndarray]] = None) -> Optional[Dict[str, Any]]:
    """Covariances of the rig and camera blocks of a solved :func:`bundle_adjust` problem.  The 3-D points
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
    # the 23-block joint of 18 Oct (k3, k4 free with k1(T)) failed the LDLT ("Simplicial LDLT for computing
    # L_inv failed") - nearly collinear theta^3..theta^9 terms, not a true null space.  Retry with a small damping
    # of the (Schur-reduced) information matrix; the result says which damping it needed (sd of directions near the
    # null space come out slightly too small with damping, so it is reported, not hidden).
    bac, damping, errors = None, None, []
    d0 = float(getattr(opts, "damping", 1e-8) or 1e-8)
    for damp in COVARIANCE_DAMPING_STEPS:
        opts.damping = max(d0, damp) if damp else d0
        try:
            bac = pycolmap.estimate_ba_covariance_from_problem(opts, rec, prob)
        except Exception as e:                              # noqa: BLE001
            errors.append(f"damping {opts.damping:g}: {type(e).__name__}: {e}")
            bac = None
        if bac is not None:
            damping = float(opts.damping)
            break
        errors.append(f"damping {opts.damping:g}: LDLT failed")
    if bac is None:
        return {"error": "covariance estimation failed (rank-deficient problem?); " + "; ".join(errors)}
    blocks: Dict[str, Any] = {}
    for (rid, cid), arr in rig_blocks.items():
        if prob.has_parameter_block(arr) and not prob.is_parameter_block_constant(arr):
            c = bac.get_other_params_cov(arr)
            if c is not None:
                blocks[f"rig:{rid}:{cid}"] = np.asarray(c)
    for cid, cam in work.cameras.items():
        blk = (cam_blocks or {}).get(int(cid), cam.params)
        if prob.has_parameter_block(blk) and not prob.is_parameter_block_constant(blk):
            c = bac.get_other_params_cov(blk)
            if c is not None:
                blocks[f"camera:{key_of.get(int(cid), cid)}"] = np.asarray(c)
    for name, blk in (extra_blocks or {}).items():         # e.g. the tied joint's global parameters
        if prob.has_parameter_block(blk) and not prob.is_parameter_block_constant(blk):
            c = bac.get_other_params_cov(blk)
            if c is not None:
                blocks[f"extra:{name}"] = np.asarray(c)
    n_res = prob.num_residuals()
    n_par = prob.num_effective_parameters() if hasattr(prob, "num_effective_parameters") else prob.num_parameters()
    red = max(n_res - n_par, 1)
    return {"blocks": blocks, "variance_factor": float(2.0 * summary.final_cost / red), "residuals": int(n_res),
            "parameters": int(n_par), "damping": damping, "damped": bool(damping and damping > d0),
            "attempts": errors}


COVARIANCE_DAMPING_STEPS = (0.0, 1e-6, 1e-4)
COVARIANCE_CAMERA_VIEWS = True                    # see bundle_adjust (the covariance needs the cameras' own memory)      # the default (COLMAP's 1e-8), then damped retries
LOCALIZE_MIN_CROSS_POINTS = 20       # an unlocalized station keeps its prior below this many cross-station points


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


POSE_GUIDED_MAX_BEYOND_GROUND_RISE = 0.02    # undo the pose-guided stage above this rise (fraction of points)
ISOLATED_POINTS_FACTOR = 4.0     # 0.94.4: before the final stage-2 adjustment, two-view points this many times the median
ISOLATED_POINTS_K = 8            # neighbour spacing at their range (mean distance to the K nearest points / range) are dropped
ISOLATED_POINTS_RADII = 1.5      # ... only within this many radii of the cameras' bounding sphere (Christian, 23 Oct)
NAVCAM_ASPECT_SIGMA = 2e-5      # 1-sigma of the fy / fx prior (relative; 0.03 px of a 1500 px focal length)


@__import__("functools").lru_cache(maxsize=1)
def _aspect_prior_class():
    import pyceres

    class _AspectPrior(pyceres.CostFunction):
        """(fy / fx - a0) / sigma on a camera's parameter block."""

        def __init__(self, a0: float, sigma: float, n: int):
            super().__init__()
            self.a0, self.sigma, self.n = a0, sigma, n
            self.set_num_residuals(1)
            self.set_parameter_block_sizes([n])

        def Evaluate(self, parameters, residuals, jacobians):
            x = parameters[0]
            residuals[0] = (x[1] / x[0] - self.a0) / self.sigma
            if jacobians is not None and jacobians[0] is not None:
                j = np.zeros(self.n)
                j[0], j[1] = -x[1] / x[0] ** 2 / self.sigma, 1.0 / x[0] / self.sigma
                jacobians[0][:] = j
            return True
    return _AspectPrior


def AspectPrior(a0: float, sigma: float, n: int):
    return _aspect_prior_class()(a0, sigma, n)


@__import__("functools").lru_cache(maxsize=1)
def _translation_tie_class():
    import pyceres

    class _TranslationTie(pyceres.CostFunction):
        """(t_a - t_b) / sigma on two rig blocks (q x, y, z, w, t x, y, z): a later visit's rig keeps the common
        stereo baseline when the translation is refined (refine_rig=True, the baseline study)."""

        def __init__(self, sigma: float):
            super().__init__()
            self.sigma = float(sigma)
            self.set_num_residuals(3)
            self.set_parameter_block_sizes([7, 7])

        def Evaluate(self, parameters, residuals, jacobians):
            a, b = parameters
            for i in range(3):
                residuals[i] = (a[4 + i] - b[4 + i]) / self.sigma
            if jacobians is not None:
                for k, sgn in ((0, 1.0), (1, -1.0)):
                    if jacobians[k] is not None:
                        j = np.zeros((3, 7))
                        for i in range(3):
                            j[i, 4 + i] = sgn / self.sigma
                        jacobians[k][:] = j.ravel()
            return True
    return _TranslationTie


EPOCH_TRANSLATION_TIE_SIGMA_M = 1e-6     # the epochs' baselines are one (to a micrometre)


ZCAM_RIG_TRANSLATION_SIGMA_M = 0.01     # prior on the Mastcam-Z right centre (in the left frame) per axis, m
ZCAM_RIG_FIT = "full"                   # the Mastcam-Z rig of each zoom refined in stage 2 (notebook 03 ZCAM_RIG_FIT)


def epoch_tie_key(key: Any) -> str:
    """the camera a Navcam rig-epoch copy is tied to - the key without its ``#k``: ``NL#1`` -> ``NL``,
    ``NR#2_T-020-015`` -> ``NR_T-020-015``; other keys are their own."""
    import re
    k = str(key)
    return re.sub(r"#\d+", "", k) if k.startswith("N") else k


def _tie_epoch_cameras(cam_blocks: Dict[int, np.ndarray], work, project: SfmProject) -> Dict[str, List[str]]:
    """make the parameter arrays of the Navcam rig-epoch copies the array of the camera they copy
    (:func:`epoch_tie_key`; within a group the smallest camera id is the owner).  Off with
    ``project.settings["navcam_rig_epochs_tie"] = False``.  Returns {owner key: [tied keys]}."""
    if project.settings.get("navcam_rig_epochs_tie", True) is False:
        return {}
    key_of = {int(v): str(k) for k, v in (project.settings.get("database", {}).get("cameras") or {}).items()}
    if not any("#" in k for k in key_of.values()):
        return {}
    groups: Dict[str, List[int]] = {}
    for cid in sorted(cam_blocks):
        k = key_of.get(int(cid))
        if k is None or not k.startswith("N"):
            continue
        groups.setdefault(epoch_tie_key(k), []).append(int(cid))
    out: Dict[str, List[str]] = {}
    for g, cids in groups.items():
        if len(cids) < 2 or not any("#" in key_of[c] for c in cids):
            continue
        owner = cids[0]
        c0 = work.cameras[owner]
        for c in cids[1:]:
            ci = work.cameras[c]
            if ci.model != c0.model or len(cam_blocks[c]) != len(cam_blocks[owner]):
                continue
            cam_blocks[c] = cam_blocks[owner]
            out.setdefault(key_of[owner], []).append(key_of[c])
    return out


def bundle_adjust(rec, project: SfmProject, sigma_px: float = 0.5, loss_scale: float = 2.0,
                  refine_intrinsics: bool = True, refine_principal_point: bool = True,
                  refine_tangential: Union[bool, Dict[str, bool]] = True, refine_rig: Union[bool, str] = "rotation", use_priors: bool = True, max_iterations: int = 100,
                  min_frame_observations: Optional[int] = None, num_threads: int = -1, verbose: bool = False,
                  attitude_prior_deg: Optional[float] = ATTITUDE_PRIOR_DEG,
                  rig_translation_sigma_m: Optional[float] = None,
                  hold_cameras: Sequence[str] = (), linear_solver: str = "auto",
                  covariance: bool = False, keypoint_scale: Optional[Dict[int, float]] = None,
                  keypoint_map=None, localize_min_images: Optional[int] = None,
                  navcam_aspect: Optional[str] = None,
                  aspect_sigma: float = NAVCAM_ASPECT_SIGMA, refine_zcam_rig: Optional[str] = None,
                  zcam_rig_translation_sigma_m: float = ZCAM_RIG_TRANSLATION_SIGMA_M,
                  extra_residuals=None, refine_hazcam_rig: Optional[str] = None) -> Dict[str, Any]:
    """
    ``refine_hazcam_rig`` (0.93) - the stereo rigs of the Hazcam pairs (sensor FR, RR): "rotation" refines their
    rotation, "hold" holds them, whatever ``refine_rig`` says; None: ``project.settings["hazcam"]["rig_refine"]``
    ("hold" without it).

    ``extra_residuals`` - a callable ``f(prob, cam_blocks=..., rig_blocks=..., key_of=..., work=...)`` called
    once the problem is built (before the solve) to add residuals of its own, e.g. ties of temperature-bin cameras
    and epoch rigs to global thermal and drift parameters (:mod:`mppp.sfm.navcal_tied`); it returns a dict, kept in
    the result under ``"extra"``, whose ``"cov_blocks"`` {name: array} get a covariance too.

    ``refine_zcam_rig`` ``"rotation"`` / ``"full"`` refines each Mastcam-Z zoom's stereo rig (one per zoom and
    block, the same for every focus: the focus motion is the principal points'), ``"full"`` also its translation with a
    prior of ``zcam_rig_translation_sigma_m`` per axis on the right centre - whatever ``refine_rig`` and the rig's hold
    flag say; None: the rig follows ``refine_rig`` and its hold flag.

    Weighted BA in place (see module docstring).  ``navcam_aspect`` (v0p65; None:
    ``project.settings["navcam_aspect"]``, set by :func:`reconstruct`; default "free"): "hold" adds a prior on every Navcam
    camera whose fx and fy are both refined, holding fy / fx at its start value (the consensus pixel aspect; 1-sigma
    ``aspect_sigma``, relative) - the pixel aspect is a property of the detector, and thermal expansion is
    isotropic; at Three Forks South and Belva (Mastcam-Z) fy moved 2-4 px against fx.  ``hold_cameras``:
    keys of project cameras (``"NL"``, ``"NR"``, a focus bin...) whose
    intrinsics are held entirely, whatever ``refine_intrinsics`` says.  ``sigma_px``: keypoint
    standard deviation in native pixels; ``loss_scale``: Cauchy scale in units
    of that sigma.  ``refine_rig``: "rotation" (default) refines the right
    camera's orientation in the rig and keeps the CAHV stereo baseline vector,
    which then fixes the scale of the reconstruction; True refines the
    baseline too (scale is then set only by the position priors, weakly for
    stations metres apart - the first Belva test shrank it from 0.424 to
    0.14 m); False holds the rig fixed.  Frames with fewer than
    ``min_frame_observations`` observations (None: ``MIN_FRAME_OBSERVATIONS``, 20 since v0p43.3, was 30) are held
    at their current pose
    (on Belva two frames with no observations otherwise rotated freely, by 9
    and 36 deg).  ``refine_tangential`` (v0p14.3; default True since v0p20):
    refine p1, p2 of OPENCV / FULL_OPENCV cameras; otherwise they stay at
    their initial values.  or {family: bool}, e.g. ``{"N": True,
    "Z": False}`` refines the Navcam p1, p2 and holds the Mastcam-Z ones
    (:func:`tangential_for`).  ``attitude_prior_deg`` (v0p20, default 1 deg):
    a weak prior on every frame's attitude (the CAHV pointing, 1-sigma per
    axis).  Without it the block's orientation is held only by the position
    priors; with few, nearly collinear stations it is free to rotate about
    the line through them (0.6 deg at Three Forks, 2.4 deg with Mastcam-Z).
    Relative orientations come from the tie points and are not affected.
    None or 0 switches it off.  ``rig_translation_sigma_m``: with
    ``refine_rig=True``, a prior (this 1-sigma per axis, metres) holding the
    right camera's centre in the rig near its CAHV value, so the data can move
    the stereo baseline without the scale being left to the position priors
    alone.  ``covariance``: after the solve, the covariance of the
    rig and camera blocks with the 3-D points eliminated
    (``pycolmap.estimate_ba_covariance_from_problem``), in the tangent space of
    their manifolds (only the refined parameters) and for the assumed
    ``sigma_px``; ``variance_factor`` (2 cost / redundancy) rescales it to the
    residuals.  Returned under ``"covariance"``.  ``keypoint_scale``:
    {image_id: s} - that image's keypoints are divided by ``s`` about its
    camera's principal point before the adjustment, which is exactly a camera
    whose fx and fy are ``s`` times the shared camera's (the Navcam thermal
    model f(T) = f0 (1 + b (T - T0)) with one shared camera at T0).
    ``localize_min_images`` (v0p35.1; None: ``project.settings["localize_min_images"]``, default 0 = off): stations
    (site, drive) with fewer images than this get no waypoint position prior - short stops during a drive, whose
    waypoint positions can be metres off (Seitah North: four two-image drives 3 m off pulled the block's scale by
    8.6 %) - see :func:`unlocalized_stations`.  Their attitude prior stays.
    ``keypoint_map``: a function (image_id, keypoints, camera) ->
    keypoints applied after the scaling (e.g. the rig's temperature
    dependence as a small rotation of the right camera's rays).  Returns
    the solver summary as a dict.
    """
    min_obs_frame = int(MIN_FRAME_OBSERVATIONS if min_frame_observations is None else min_frame_observations)
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
    # the camera intrinsics, too, are explicit arrays owned here (written back below and checked).  Handing
    # Ceres pycolmap's views (camera.params) was not reliable: in the thermal stage of 4 Oct (Chal Rocks, and
    # reproduced, intermittently, on Linux) the optimiser refined the temperature-bin cameras but the reconstruction
    # kept their start values - the poses then fitted cameras the model no longer had (median 0.16 -> 0.37 px)
    cam_blocks: Dict[int, np.ndarray] = {int(cid): np.ascontiguousarray(np.array(c.params, dtype=np.float64))
                                         for cid, c in work.cameras.items()}
    # COLMAP's covariance estimator (BACovarianceOptionsParams.ALL) fails ("Simplicial LDLT ... failed",
    # every consensus since v0p74.7) when the intrinsics are arrays of our own - it works with the cameras' own
    # memory (pycolmap views), as before v0p74.7.  With ``covariance`` the problem therefore uses the views; the
    # write-back below is checked as before, and a solve in which no free camera moved is reported (the v0p74.7 case).
    views_used = bool(covariance) and COVARIANCE_CAMERA_VIEWS
    if views_used:
        cam_starts = {cid: np.array(blk) for cid, blk in cam_blocks.items()}
        cam_blocks = {int(cid): c.params for cid, c in work.cameras.items()}
    # the Navcam rig-epoch copies of an eye (NL#1, NL#2 ... and their temperature bins NL#1_T...) share one
    # parameter array with the eye's camera (NL, NL_T...): one set of intrinsics and distortion for every visit, one
    # rig rotation per visit.  (The copies exist because a COLMAP database puts a camera in one rig only.)
    tied = _tie_epoch_cameras(cam_blocks, work, project)
    # the Mastcam-Z rig of each zoom, shared by all focus bins - a right image is posed from the frame of its
    # left partner (same exposure) through this block; its own frame gets no pose block and is set afterwards
    vrig = project.settings.get("zcam_virtual_rig") or {}
    vblocks: Dict[str, np.ndarray] = {}
    vpartner: Dict[int, tuple] = {}
    if vrig:
        from scipy.spatial.transform import Rotation as _Rot
        rows_by_name = {r["name"]: r for r in project.images}
        lefts = {}
        for iid in reg:
            r = rows_by_name.get(rec.images[iid].name) or {}
            g = str(r.get("camera_group", ""))
            if g.startswith("ZL"):
                lefts[(g[2:], r.get("sclk_key"))] = rec.images[iid].frame_id
        for iid in reg:
            im = rec.images[iid]
            r = rows_by_name.get(im.name) or {}
            g = str(r.get("camera_group", ""))
            fam = "Z" + g[2:]
            if g.startswith("ZR") and fam in vrig and (g[2:], r.get("sclk_key")) in lefts:
                fr = work.frames[im.frame_id]
                if len(list(fr.data_ids)) == 1:              # a frame of its own (no COLMAP rig)
                    vpartner[iid] = (fam, lefts[(g[2:], r.get("sclk_key"))])
        for fam, v in vrig.items():
            # from the rig as refined so far in this run (R_refined, t_refined), else the start rig
            q = _Rot.from_matrix(np.asarray(v.get("R_refined") or v["R_sensor_from_ref"], float)).as_quat()
            vblocks[fam] = np.r_[q, np.asarray(v.get("t_refined") or v["t_sensor_from_ref"], float)].astype(np.float64)
    n_obs = 0
    pose_blocks: Dict[int, np.ndarray] = {}
    # everything that is the same for all observations of an image is looked up once per image
    # (the cost constructors and add_residual_block are what remains in the loop: ~2x faster setup)
    per_image: Dict[int, tuple] = {}
    for iid in reg:
        im = rec.images[iid]
        fr = work.frames[im.frame_id]
        rig = rec.rigs[fr.rig_id]
        cam = work.cameras[im.camera_id]
        cov = np.eye(2) * (sigma_px / scale[iid]) ** 2
        if iid in vpartner:                                   # posed through the Mastcam-Z rig
            fam, lf = vpartner[iid]
            lpose = pose_blocks.get(lf)
            if lpose is None:
                lpose = pose_blocks[lf] = work.frames[lf].rig_from_world.params
            kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
            if keypoint_scale and iid in keypoint_scale and np.any(np.asarray(keypoint_scale[iid], float) != 1.0):
                c0 = np.asarray(cam.params[2:4], float)
                kps = c0 + (kps - c0) / np.asarray(keypoint_scale[iid], float)      # a scalar or (sx, sy)
            if keypoint_map is not None and len(kps):
                kps = keypoint_map(iid, kps, cam)
            per_image[iid] = (cam.model, cov, kps, [vblocks[fam], lpose, cam_blocks[int(im.camera_id)]], True)
            continue
        pose = pose_blocks.get(im.frame_id)
        if pose is None:
            pose = pose_blocks[im.frame_id] = fr.rig_from_world.params
        sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=im.camera_id)
        kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
        if keypoint_scale and iid in keypoint_scale and np.any(np.asarray(keypoint_scale[iid], float) != 1.0):
            c0 = np.asarray(cam.params[2:4], float)
            kps = c0 + (kps - c0) / np.asarray(keypoint_scale[iid], float)          # a scalar or (sx, sy)
        if keypoint_map is not None and len(kps):
            kps = keypoint_map(iid, kps, cam)
        if rig.is_ref_sensor(sensor):
            per_image[iid] = (cam.model, cov, kps, [pose, cam_blocks[int(im.camera_id)]], None)
        else:
            per_image[iid] = (cam.model, cov, kps, [rig_blocks[(fr.rig_id, im.camera_id)], pose,
                                                    cam_blocks[int(im.camera_id)]], True)
    reproj, rig_reproj, add = cf.ReprojErrorCost, cf.RigReprojErrorCost, prob.add_residual_block
    point_arrs: List[np.ndarray] = []                         # 0.94.4: for the exact-tie covariance
    for pid, pt in rec.points3D.items():
        xyz = pt.xyz
        point_arrs.append(xyz)
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
    seen_blocks = set()                                       # a tied array is set up once (its group's first)
    for cid, cam in sorted(work.cameras.items()):
        blk = cam_blocks[int(cid)]
        if not prob.has_parameter_block(blk) or id(blk) in seen_blocks:
            continue
        seen_blocks.add(id(blk))
        if not refine_intrinsics or key_of.get(int(cid), "") in set(hold_cameras):
            prob.set_parameter_block_constant(blk)
            continue
        fixed = fixed_camera_params(cam.model.name, refine_principal_point,
                                    tangential_for(refine_tangential, key_of.get(int(cid), "")))
        pc = project.cameras.get(key_of.get(int(cid), ""), {})
        names = _PARAM_NAMES.get(cam.model.name, FULL_OPENCV_NAMES)
        extra = pc.get("fixed_params") or []
        if extra and cam.model.name in _PARAM_NAMES:
            fixed = sorted(set(fixed) | {names.index(n) for n in extra if n in names and names.index(n) < len(cam.params)})
        free = pc.get("free_params") or []            # e.g. the rational Navcam k4; any model
        if free and cam.model.name in _PARAM_NAMES:
            fixed = sorted(set(fixed) - {names.index(n) for n in free if n in names})
        prob.set_parameter_block_variable(blk)
        if fixed:
            prob.set_manifold(blk, pyceres.SubsetManifold(len(blk), sorted(fixed)))

    # the Navcam pixel aspect held at its start (fy / fx), a soft prior per camera
    if navcam_aspect is None:
        navcam_aspect = str(project.settings.get("navcam_aspect") or "free")
    aspect_costs = []
    if navcam_aspect == "hold" and refine_intrinsics:
        for cid, cam in work.cameras.items():
            key = key_of.get(int(cid), "")
            pc = project.cameras.get(key, {})
            if not str(pc.get("group") or key).startswith(("N", "F", "R")) or key in set(hold_cameras):
                continue                                  # 0.93: the Hazcam eyes keep their pixel aspect too
            blk = cam_blocks[int(cid)]
            if not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                continue
            if any(blk is c_.parameter_block for c_ in aspect_costs if hasattr(c_, "parameter_block")):
                continue
            names = _PARAM_NAMES.get(cam.model.name)
            if not names or names[:2] != ("fx", "fy"):
                continue
            fixed = set((pc.get("fixed_params") or [])) - set(pc.get("free_params") or [])
            if "fx" in fixed or "fy" in fixed:
                continue
            p0 = np.asarray(pc.get("params") or blk, float)
            c = AspectPrior(float(p0[1] / p0[0]), float(aspect_sigma), len(blk))
            try:
                c.parameter_block = blk                       # one prior per (tied) array
            except AttributeError:
                pass
            aspect_costs.append(c)
            prob.add_residual_block(c, None, [blk])

    # waypoint position priors on every frame (the reference camera's centre); not on the frames of
    # stations with fewer than localize_min_images images, if they are tied to the rest of the block
    n_prior = n_att = 0
    if localize_min_images is None:
        localize_min_images = int(project.settings.get("localize_min_images") or 0)
    unloc = unlocalized_stations(project, localize_min_images, rec=rec) if localize_min_images else {}
    no_prior = {st for st, v in unloc.items() if v["prior"] == "dropped"}
    # stations whose waypoints were flagged (mppp.sfm.waypoint_check; notebook 03 NO_POSITION_PRIOR_AT)
    flagged = {str(x) for x in (project.settings.get("no_position_prior_at") or [])}
    no_prior |= flagged
    prior_frames: List[Tuple[int, np.ndarray, float]] = []
    n_no_prior = 0
    n_unsurveyed = 0
    if use_priors:
        sig = np.asarray(project.settings.get("prior_sigma_m", [1.0, 1.0, 1.0]), float)
        cov3 = np.diag(sig ** 2)
        sig_u = float(project.settings.get("unsurveyed_prior_sigma_m") or UNSURVEYED_PRIOR_SIGMA_M)
        cov3_u = np.diag([max(sig_u, float(x)) ** 2 for x in sig])
        att_u = float(project.settings.get("unsurveyed_attitude_prior_deg") or UNSURVEYED_ATTITUDE_PRIOR_DEG)
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
            unsurveyed = is_unsurveyed(r_ref)               # v0p87: no waypoint of its own - looser priors
            n_unsurveyed += unsurveyed
            if r_ref.get("station") in no_prior:
                n_no_prior += 1
            else:
                c3 = cov3_u if unsurveyed else cov3
                if r_ref.get("prior_sigma_m"):              # 0.93: a Hazcam frame placed on its station's Navcam
                    c3 = np.diag(np.broadcast_to(np.asarray(r_ref["prior_sigma_m"], float), 3) ** 2)
                prob.add_residual_block(cf.AbsolutePosePositionPriorCost(c3, C), None, [pose])
                prior_frames.append((fid, C, 1.0 / float(np.mean(np.diag(c3)))))
                n_prior += 1
            if attitude_prior_deg:
                # 6-DoF prior: rotation block first (checked), translation effectively free (the position
                # prior above holds the centre)
                ref_r = by_name[rec.images[ref[0]].name]
                T0 = _rigid(ref_r["prior_R_w2c"], ref_r["prior_C"])
                att = max(float(attitude_prior_deg), att_u) if unsurveyed else float(attitude_prior_deg)
                cov6 = np.diag([np.radians(att) ** 2] * 3 + [1e8] * 3)
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
        if n_frame.get(fid, 0) < min_obs_frame and prob.has_parameter_block(pose):
            prob.set_parameter_block_constant(pose)
            held.append(int(fid))

    rig_start = {k: v.copy() for k, v in rig_blocks.items()}
    n_rig_prior = 0
    n_rig_tie = 0
    rig_yaw = str(project.settings.get("navcam_rig_yaw") or "refine")        # "refine" or "hold"
    rig_roll = str(project.settings.get("navcam_rig_roll") or "refine")      # 0.91: likewise the roll (z)
    hz_mode = refine_hazcam_rig if refine_hazcam_rig is not None else (
        (project.settings.get("hazcam") or {}).get("rig_refine") or "hold")
    for key, arr in rig_blocks.items():
        if not prob.has_parameter_block(arr):
            continue
        nav_rig = str(key_of.get(int(key[1]), "")).startswith("N") if key_of else False
        if key_of and str(key_of.get(int(key[1]), ""))[:1] in ("F", "R"):      # a Hazcam pair (0.93)
            fam_h = str(key_of.get(int(key[1]), ""))[:1]
            if hz_mode == "rotation" and fam_h not in set((project.settings.get("hazcam") or {}).get("rig_hold") or ()):
                prob.set_manifold(arr, pyceres.SubsetManifold(7, [3, 4, 5, 6]))
            else:
                prob.set_parameter_block_constant(arr)
            continue
        if refine_rig is True and rig_translation_sigma_m:
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
            held_q = [3, 4, 5, 6] if refine_rig == "rotation" else [3]
            if refine_rig is True and tied and "#" in str(key_of.get(int(key[1]), "")):
                # a later visit's rig keeps the common baseline - its translation is refined but tied to
                # the first visit's (the baseline study refines the translation; before, a later visit's stayed at CAHV)
                main = [k2 for k2 in rig_blocks if str(key_of.get(int(k2[1]), "")) == epoch_tie_key(key_of.get(int(key[1]), ""))]
                if main and prob.has_parameter_block(rig_blocks[main[0]]):
                    prob.add_residual_block(_translation_tie_class()(EPOCH_TRANSLATION_TIE_SIGMA_M), None,
                                            [arr, rig_blocks[main[0]]])
                    n_rig_tie += 1
                else:
                    held_q = [3, 4, 5, 6]
            if nav_rig and rig_yaw == "hold":
                held_q = [1] + held_q                     # the yaw stays; pitch (x) refined
            if nav_rig and rig_roll == "hold":
                held_q = sorted(set([2] + held_q))        # 0.91: the roll (z) stays too
            prob.set_manifold(arr, pyceres.SubsetManifold(7, held_q))
        else:
            prob.set_parameter_block_constant(arr)

    for fam, arr in vblocks.items():                         # the Mastcam-Z rigs
        if not prob.has_parameter_block(arr):
            continue
        zmode = refine_zcam_rig if refine_zcam_rig in ("rotation", "full") else (
            None if (vrig[fam].get("hold") or not refine_rig) else "rotation")
        if zmode is None:
            prob.set_parameter_block_constant(arr)
        elif zmode == "rotation":
            prob.set_manifold(arr, pyceres.SubsetManifold(7, [3, 4, 5, 6]))      # rotation refined, w held
        else:                                                 # rotation and translation (baseline) refined
            # the prior is on the start rig's right centre (CAHV or the focus model's), not on this call's start
            R0, t0 = np.asarray(vrig[fam]["R_sensor_from_ref"], float), np.asarray(vrig[fam]["t_sensor_from_ref"], float)
            C0 = -R0.T @ t0
            prob.add_residual_block(cf.AbsolutePosePositionPriorCost(
                np.eye(3) * float(zcam_rig_translation_sigma_m) ** 2, C0), None, [arr])
            prob.set_manifold(arr, pyceres.SubsetManifold(7, [3]))

    # a right Mastcam-Z image posed through the rig: its own frame's pose block (from COLMAP's adjuster) has no
    # residuals left - out of the problem, or the covariance is rank-deficient (the Mastcam-Z joint, 20 Oct)
    for iid in vpartner:
        arr_ = work.frames[rec.images[iid].frame_id].rig_from_world.params
        if prob.has_parameter_block(arr_) and int(rec.images[iid].frame_id) not in pose_blocks:
            prob.remove_parameter_block(arr_)
    extra_info = None
    if extra_residuals is not None:                          # e.g. the tied joint's thermal / drift ties
        extra_info = extra_residuals(prob, cam_blocks=cam_blocks, rig_blocks=rig_blocks, key_of=key_of, work=work)
    so = pyceres.SolverOptions()
    # the linear solver.  The Schur complement has one 6-parameter block per frame plus the cameras, so
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
    retried = None
    if summary.termination_type == pyceres.TerminationType.FAILURE:
        # Ceres gives up after a few linear-solver failures in a row (the dense Schur complement not
        # positive definite, e.g. a temperature-bin camera with one image) and then leaves the parameters only partly
        # updated - in the thermal stage of 4 Oct the poses moved while the cameras stayed at their start (median
        # 0.16 -> 0.37 px).  Solve again from there, with patience for invalid steps and the iterative Schur solver.
        first = summary
        so.max_num_consecutive_invalid_steps = 100
        so.linear_solver_type = pyceres.LinearSolverType.ITERATIVE_SCHUR
        so.preconditioner_type = pyceres.PreconditionerType.SCHUR_JACOBI
        summary = pyceres.SolverSummary()
        pyceres.solve(so, prob, summary)
        retried = {"first": first.BriefReport(), "second": summary.BriefReport()}
        if True:                                              # always: it matters for the record
            logger.info(f"bundle adjustment: the first solve failed ({first.BriefReport()}); solved again with the "
                  f"iterative Schur solver: {summary.BriefReport()}")
        if summary.termination_type == pyceres.TerminationType.FAILURE:
            raise RuntimeError(f"bundle adjustment failed twice: {first.BriefReport()} / {summary.BriefReport()}")
    if (extra_info or {}).get("exact_solve") and (extra_info or {}).get("tie_specs") \
            and (extra_info or {}).get("cov_blocks") and summary.termination_type != pyceres.TerminationType.FAILURE:
        # (no free global term - every global held, e.g. the pointing pass: the ties hold the units on the law already)
        # 0.94.4: Levenberg-Marquardt on with the ties as exact constraints (mppp.calib.tiesolve)
        from ..calib.tiesolve import exact_tie_refine
        free_ = list(pose_blocks.values()) + list(cam_blocks.values()) + list(rig_blocks.values()) + list(vblocks.values())
        xr = exact_tie_refine(prob, extra_info["tie_specs"], point_arrs, free_, extra_info["cov_blocks"],
                              verbose=True)
        extra_info["exact_refine"] = {k: v for k, v in xr.items() if k != "history"}
        extra_info["exact_refine"]["history"] = xr.get("history")
        extra_info["tie_ids"] = [t["id"] for t in extra_info["tie_specs"]] if extra_info.get("tie_ids") else []
    cov_out = None
    if covariance:
        cov_out = _block_covariances(rec, prob, work, rig_blocks, key_of, summary, n_obs, n_prior, n_att, pose_blocks,
                                     cam_blocks=cam_blocks, extra_blocks=(extra_info or {}).get("cov_blocks"))
        if (extra_info or {}).get("tie_ids") and (extra_info or {}).get("cov_blocks"):
            # 0.94.4: the global terms of a tied joint with the ties eliminated exactly (COLMAP's estimator gave
            # them infinite on 22-29 blocks: tie weights of 1e16 next to data weights of 1e7) - last, as it takes
            # the tie residuals out of the problem
            from ..calib.tiecov import exact_tie_covariance
            free = list(pose_blocks.values()) + list(cam_blocks.values()) + list(rig_blocks.values()) + \
                list(vblocks.values())
            ex = exact_tie_covariance(prob, extra_info["tie_ids"], point_arrs, free, extra_info["cov_blocks"])
            cov_out = dict(cov_out or {})
            cov_out["extra_method"] = ex["method"] if not ex.get("error") else "colmap (exact ties failed)"
            cov_out["extra_exact"] = {k: ex.get(k) for k in ("error", "dims", "variance_factor", "full")}
            if not ex.get("error"):
                blocks = dict(cov_out.get("blocks") or {})
                for k, c in ex["blocks"].items():         # unit variance factor: _sd scales by the problem's
                    blocks[k] = np.asarray(c)
                cov_out["blocks"] = blocks
                if not cov_out.get("variance_factor"):   # COLMAP's failed: 2 cost / (residuals - parameters)
                    d = ex.get("dims") or {}
                    n_eff = 3 * int(d.get("points", 0)) + int(d.get("free_columns", 0)) + int(d.get("global_columns", 0))
                    cov_out["variance_factor"] = float(2.0 * summary.final_cost / max(prob.num_residuals() - n_eff, 1))
                cov_out.pop("error", None)
            else:
                logger.info(f"bundle adjustment: the exact-tie covariance failed ({ex['error']}); COLMAP's kept")
    for fid in pose_blocks:
        rec.frames[fid].rig_from_world = work.frames[fid].rig_from_world
    if vblocks:                                               # write back the Mastcam-Z rigs and right poses
        from scipy.spatial.transform import Rotation as _Rot
        vT = {}
        for fam, arr in vblocks.items():
            q = arr[:4] / np.linalg.norm(arr[:4])
            vT[fam] = pycolmap.Rigid3d(pycolmap.Rotation3d(_Rot.from_quat(q).as_matrix()), arr[4:7].copy())
            vrig[fam]["R_refined"] = _Rot.from_quat(q).as_matrix().tolist()
            vrig[fam]["t_refined"] = arr[4:7].tolist()
        for iid, (fam, lf) in vpartner.items():
            rec.frames[rec.images[iid].frame_id].rig_from_world = vT[fam] * rec.frames[lf].rig_from_world
    views_unchanged = None
    if views_used:                                            # did the views take the solve's update?
        free_cids = [cid for cid, blk in cam_blocks.items()
                     if prob.has_parameter_block(blk) and not prob.is_parameter_block_constant(blk)]
        views_unchanged = bool(free_cids) and all(np.allclose(np.asarray(cam_blocks[c]), cam_starts[c], rtol=0, atol=0)
                                                  for c in free_cids)
        if views_unchanged:
            logger.info("bundle adjustment: no free camera changed in the covariance solve (camera views) - check "
                  "the cameras")
    for cid, blk in cam_blocks.items():                      # written back from the owned arrays, checked
        if cid in rec.cameras:
            _set_camera_params(rec, cid, np.array(blk, dtype=np.float64))
    k2_clamped = clamp_k2(rec)                               # |k2| <= K2_LIMIT
    for (rid, cid), arr in rig_blocks.items():
        q = arr[:4] / np.linalg.norm(arr[:4])
        T = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_quat(q).as_matrix()), arr[4:7].copy())
        rec.rigs[rid].set_sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid), T)
    rig_out = {}
    for (rid, cid), arr in rig_blocks.items():
        R1 = Rotation.from_quat(arr[:4] / np.linalg.norm(arr[:4])).as_matrix()
        R0 = Rotation.from_quat(rig_start[(rid, cid)][:4] / np.linalg.norm(rig_start[(rid, cid)][:4])).as_matrix()
        rig_out[f"{rid}:{cid}"] = {"baseline_m": float(np.linalg.norm(arr[4:7])),
                                   "baseline_start_m": float(np.linalg.norm(rig_start[(rid, cid)][4:7])),
                                   "centre_change_m": (-(R1.T @ arr[4:7]) + R0.T @ rig_start[(rid, cid)][4:7]).tolist(),
                                   "rotation_change_deg": float(np.degrees(np.linalg.norm(
                                       Rotation.from_matrix(R1 @ R0.T).as_rotvec())))}
    gauge = prior_gauge(rec, prior_frames, recentre=bool(project.settings.get("prior_gauge_recentre",
                                                                               PRIOR_GAUGE_RECENTRE)))
    return {"observations": n_obs, "priors": n_prior, "attitude_priors": n_att, "rig": rig_out, "covariance": cov_out,
            "prior_gauge": gauge, "no_position_prior_at": sorted(flagged),
            "rig_translation_priors": n_rig_prior, "rig_translation_ties": n_rig_tie,
            "frames_without_position_prior": n_no_prior, "frames_unsurveyed": int(n_unsurveyed),
            "unsurveyed_prior": {"sigma_m": sig_u, "attitude_deg": att_u} if use_priors else None,
            "tied_cameras": tied, "k2_clamped": k2_clamped,
            "unlocalized_stations": unloc, "navcam_aspect": navcam_aspect, "aspect_priors": len(aspect_costs),
            "attitude_prior_deg": float(attitude_prior_deg or 0.0), "frames_held": held, "initial_cost": summary.initial_cost,
            "final_cost": (float(extra_info["exact_refine"]["cost_end"]) if (extra_info or {}).get("exact_refine")
                           and not extra_info["exact_refine"].get("error") and "cost_end" in extra_info["exact_refine"]
                           else summary.final_cost), "iterations": summary.num_successful_steps + summary.num_unsuccessful_steps,
            "termination": str(summary.termination_type), "brief": summary.BriefReport(), "retried": retried, "extra": extra_info, "camera_views": views_used, "camera_views_unchanged": views_unchanged,
            "refine_tangential": tangential_setting(refine_tangential), "linear_solver": solver, "num_threads": so.num_threads,
            "seconds": float(summary.total_time_in_seconds)}


def clamp_k2(rec, limit: float = None) -> Dict[int, float]:
    """v0p85 (Christian, 19 Oct): every camera's ``k2`` within +-``limit`` (default :data:`K2_LIMIT`, 10): a value
    beyond it is set to +-limit.  The Mastcam-Z 34 mm bins reach k2 ~2 in a 25-degree field (r <= 0.22, so k2 r^4
    is a few pixels); Z63 and Z110 will go higher.  Returns {camera_id: value before} of the cameras changed."""
    lim = float(K2_LIMIT if limit is None else limit)
    out = {}
    for cid, cam in rec.cameras.items():
        names = _PARAM_NAMES.get(cam.model.name)
        if not names or "k2" not in names:
            continue
        i = names.index("k2")
        p = np.array(cam.params, float)
        if i < len(p) and abs(p[i]) > lim:
            out[int(cid)] = float(p[i])
            p[i] = float(np.sign(p[i]) * lim)
            cam.params = p
    return out


def prior_gauge(rec, prior_frames: Sequence[Tuple[int, Any]], recentre: bool = PRIOR_GAUGE_RECENTRE) -> Optional[Dict[str, Any]]:
    """the mean residual (prior - refined) of the frame centres that have a position prior, and with
    ``recentre`` the whole reconstruction (frames and points) moved by it.  With equal Gaussian position priors and
    nothing else holding the block's position, the least-squares solution has a zero mean residual; Ceres stops on a
    relative cost decrease, and moving a block of ~10^6 observations by 0.2 m changes the cost by a few units only,
    so that direction may be left unconverged.  The move does not change any reprojection.  Returns
    {"frames", "mean_residual_m" (E, N, U before), "shift_m" (applied), "recentred"} or None without priors."""
    import pycolmap
    if not prior_frames:
        return None
    d = []
    w = []
    for item in prior_frames:
        fid, C = item[0], item[1]
        if fid not in rec.frames:
            continue
        T = rec.frames[fid].rig_from_world
        R = np.asarray(T.rotation.matrix())
        d.append(np.asarray(C, float) - (-R.T @ np.asarray(T.translation, float)))
        w.append(float(item[2]) if len(item) > 2 else 1.0)    # v0p87: 1 / sigma^2 (unsurveyed stations count less)
    if not d:
        return None
    m = np.average(np.asarray(d), axis=0, weights=np.asarray(w))
    out = {"frames": len(d), "mean_residual_m": m.tolist(), "shift_m": [0.0, 0.0, 0.0], "recentred": False}
    if recentre and np.all(np.isfinite(m)):
        rec.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(np.eye(3)), m))
        out.update(shift_m=m.tolist(), recentred=True)
    return out


def navcam_rig_fit(ba: Dict[str, Any], rec, project: SfmProject) -> Dict[str, Any]:
    """the Navcam rig after an adjustment: per rig whose sensor is a Navcam right camera (``NR``, the epoch
    copies ``NR#k``) its rotation relative to the start (pitch, yaw, roll mdeg, rotation vector in the right camera)
    and the baseline; with a covariance (refine_rig=True and ``covariance``) the baseline's sd (translation block
    projected on the baseline direction, scaled by the variance factor).  {"baseline_m", "baseline_start_m",
    "baseline_sd_m", "epochs": [...]}."""
    from scipy.spatial.transform import Rotation
    key_of = {int(v): str(k) for k, v in (project.settings.get("database", {}).get("cameras") or {}).items()}
    out: Dict[str, Any] = {"baseline_m": None, "baseline_start_m": None, "baseline_sd_m": None, "epochs": []}
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor", 1.0) or 1.0)
    for name, r in sorted((ba.get("rig") or {}).items()):
        rid, cid = (int(x) for x in name.split(":"))
        key = key_of.get(cid, str(cid))
        if not key.startswith("NR") or "_T" in key:
            continue
        T1 = rec.rigs[rid].sensor_from_rig(__import__("pycolmap").sensor_t(type=__import__("pycolmap").SensorType.CAMERA, id=cid))
        R1 = np.asarray(T1.rotation.matrix())
        t1 = np.asarray(T1.translation, float)
        fam = "N" + key[2:]                                   # "NR#1" -> the epoch rig "N#1"
        R0 = np.asarray((project.rig.get(fam) or project.rig.get("N") or {}).get("R_sensor_from_ref") or R1, float)
        rv = 1e3 * np.degrees(Rotation.from_matrix(R1 @ R0.T).as_rotvec())
        e = {"camera": key, "rig_id": rid, "pitch_mdeg": float(rv[0]), "yaw_mdeg": float(rv[1]), "roll_mdeg": float(rv[2]),
             "baseline_m": float(r["baseline_m"]), "baseline_start_m": float(r["baseline_start_m"]),
             "t_m": t1.tolist()}
        c = (cov.get("blocks") or {}).get(f"rig:{rid}:{cid}")
        if c is not None:
            c = np.asarray(c, float)
            if c.shape[0] >= 3:
                u = t1 / max(np.linalg.norm(t1), 1e-12)
                e["baseline_sd_m"] = float(np.sqrt(max(u @ c[-3:, -3:] @ u, 0.0) * vf))
        out["epochs"].append(e)
    main = [e for e in out["epochs"] if "#" not in e["camera"]] or out["epochs"]
    if main:
        out.update(baseline_m=main[0]["baseline_m"], baseline_start_m=main[0]["baseline_start_m"],
                   baseline_sd_m=main[0].get("baseline_sd_m"))
    out["variance_factor"] = vf if cov else None
    return out


def load_alignment(project: SfmProject):
    """
    the saved alignment of ``project`` (the model ``project.settings["reconstruction"]["path"]`` names), for
    notebook 03 cells run after a kernel restart.  Raises RuntimeError with the reason when there is none: the
    reconstruction (section 6) has not finished, or its model folder is missing.
    """
    import pycolmap
    rs = (project.settings or {}).get("reconstruction") or {}
    if not rs.get("path"):
        raise RuntimeError(f"{project.root}: no finished alignment - project.json records no reconstruction (section 6 "
                           f"has not run to the end for this project, or stopped with an error): run section 6 first")
    d = project.root / str(rs["path"]).replace("\\", "/")
    if not (d / "images.bin").is_file() and not (d / "images.txt").is_file():
        raise RuntimeError(f"{d}: the alignment recorded in project.json is missing: run section 6 again")
    rec = pycolmap.Reconstruction()
    rec.read(str(d))
    return rec


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
    registered stay at their priors (reported).  ``only``: move just
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
            logger.info(f"station {s}: {n_in}/{len(pts2)} inliers, moved {shift:.3f} m / {ang:.3f} deg")
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


# v0p86 (Christian, 19 Oct): a point seen by Navcam and Mastcam-Z needs a third view.  In the 19 Oct diagnostic
# 36-47 % of the two-view Navcam + Mastcam-Z points sat more than 0.25 m off the terrain (Rochette 4 699, South Arm
# 9 706: low reprojection error, wrong place - epipolar-consistent false matches, mostly from the pose-guided
# Navcam-Mastcam-Z matching), against ~1 % of the two-view points of one camera type (the stereo pairs, kept)
MIN_MIXED_TRACK_LENGTH = 3
HAZCAM_ZCAM_NEED_NAVCAM = True   # 0.94.4: a Hazcam + Mastcam-Z point needs a Navcam image in its track too


def drop_mixed_short_tracks(rec, min_mixed_track_length: int = MIN_MIXED_TRACK_LENGTH,
                            hazcam_zcam_need_navcam: bool = HAZCAM_ZCAM_NEED_NAVCAM) -> int:
    """Delete points whose track holds both Navcam (``N...``) and Mastcam-Z (``Z...``) images and is shorter than
    ``min_mixed_track_length`` (the third look, v0p86), and - ``hazcam_zcam_need_navcam``, 0.94.4 - points whose
    track holds a Hazcam (``F`` / ``R``) and a Mastcam-Z image but no Navcam image, whatever its length.  On Chal Rocks
    tiny those Hazcam + Mastcam-Z points were 46 % above and 23 % below the Navcam terrain (two-view; 7 / 5 % with
    three looks or more), Seitah Overlook tiny 67 / 24 % - most of the scattered Mastcam-Z points - while the same
    points with a Navcam look too were on it (0.1 / 0.0 %).  Returns the number deleted.  Blocks of one camera type
    are untouched."""
    if min_mixed_track_length <= 2 and not hazcam_zcam_need_navcam:
        return 0
    fam: Dict[int, str] = {}
    bad = []
    for pid, pt in rec.points3D.items():
        short = pt.track.length() < min_mixed_track_length
        if not short and not hazcam_zcam_need_navcam:
            continue
        fs = set()
        for el in pt.track.elements:
            f = fam.get(el.image_id)
            if f is None:
                f = fam[el.image_id] = rec.images[el.image_id].name[:1]
            fs.add(f)
        if "Z" not in fs:
            continue
        if ("N" in fs and short and min_mixed_track_length > 2) or \
                (hazcam_zcam_need_navcam and "N" not in fs and fs & {"F", "R"}):
            bad.append(int(pid))
    for pid in bad:
        rec.delete_point3D(pid)
    return len(bad)


def drop_isolated_points(rec, factor: float = ISOLATED_POINTS_FACTOR, k: int = ISOLATED_POINTS_K,
                         min_points: int = 1000, two_view_only: bool = True, range_bin_ratio: float = 1.5,
                         min_bin_points: int = 200, within_radii: float = ISOLATED_POINTS_RADII) -> Dict[str, Any]:
    """
    Delete the 3-D points far from every other point (0.94.4, Christian 23 Oct: "a few points scattered above and
    below the terrain").  Isolation is the mean distance to the ``k`` nearest points divided by the point's range
    (from the nearest camera that sees it), compared with the median isolation of the points at a similar range
    (bins of ``range_bin_ratio`` in range; a bin of fewer than ``min_bin_points`` takes its neighbours too): ground
    seen at a grazing angle thins out faster than the range, so one block-wide median deleted a third of the points
    beyond 40 m (Christian's question, 14:36).  A point more than ``factor`` x its range bin's median isolation is
    deleted - only a two-view point with ``two_view_only`` (a point that a third image confirms stays, so no
    Navcam - Mastcam-Z point, which needs three, is ever deleted).  Only points within ``within_radii`` x the radius
    of the cameras' bounding sphere (centred on the mean camera centre) are judged (Christian, 14:39: the false points
    are near the rover, the far ground is left alone); 0 = everywhere.

    The scattered points are false matches at the wrong range along their rays (pose-guided matching above all;
    73-94 % two-view, median isolation 9-16x the others').  On the Chal Rocks tiny / Seitah Overlook tiny SIFT blocks
    the Mastcam-Z points above / below the Navcam terrain go from 2.81 / 1.60 % to 0.29 / 0.21 % and from 1.30 / 0.37 %
    to 0.58 / 0.10 %, deleting 5.6 / 3.7 % of the points, none of the Navcam - Mastcam-Z points and 1.3 / 6.2 % of the
    points beyond 40 m; within 1.5 radii of the camera sphere (4.5 / 4.8 m there): 0.33 / 0.24 % and 0.76 / 0.17 %,
    deleting 5.3 / 0.5 % of the points and none beyond 40 m.  Returns {"deleted", "points", "factor", "k", ...}; nothing is done with fewer than
    ``min_points`` points or ``factor`` <= 0.
    """
    out: Dict[str, Any] = {"deleted": 0, "points": int(len(rec.points3D)), "factor": float(factor), "k": int(k),
                           "two_view_only": bool(two_view_only)}
    if not factor or factor <= 0 or len(rec.points3D) < max(int(min_points), k + 2):
        return out
    from scipy.spatial import cKDTree
    centres = {int(i): np.asarray(im.projection_center(), float) for i, im in rec.images.items() if im.has_pose}
    pids = np.fromiter((int(p) for p in rec.points3D), np.int64, len(rec.points3D))
    X = np.empty((len(pids), 3))
    rng = np.empty(len(pids))
    tl = np.empty(len(pids), int)
    for n, pid in enumerate(pids):
        pt = rec.points3D[int(pid)]
        X[n] = pt.xyz
        tl[n] = pt.track.length()
        cs = [centres[int(e.image_id)] for e in pt.track.elements if int(e.image_id) in centres]
        rng[n] = min(float(np.linalg.norm(pt.xyz - c)) for c in cs) if cs else np.nan
    ok = np.isfinite(rng)
    if ok.sum() < k + 2:
        return out
    d, _ = cKDTree(X).query(X, k=k + 1)
    iso = d[:, 1:].mean(axis=1) / np.maximum(np.where(ok, rng, 1.0), 0.5)
    b = np.floor(np.log(np.maximum(np.where(ok, rng, 1.0), 0.5)) / np.log(range_bin_ratio)).astype(int)
    med = np.full(len(iso), np.nan)
    for v in np.unique(b[ok]):
        sel = ok & (b == v)
        src = sel if sel.sum() >= min_bin_points else ok & (np.abs(b - v) <= 1)
        if src.sum() >= max(k + 2, 20):
            med[sel] = np.median(iso[src])
    bad = ok & np.isfinite(med) & (iso > factor * med)
    if two_view_only:
        bad &= tl == 2
    if within_radii and within_radii > 0 and centres:
        C = np.array(list(centres.values()))
        c0 = C.mean(axis=0)
        radius = float(np.max(np.linalg.norm(C - c0, axis=1)))
        bad &= np.linalg.norm(X - c0, axis=1) <= within_radii * radius
        out.update(camera_sphere_radius_m=radius, within_m=within_radii * radius)
    for pid in pids[bad]:
        rec.delete_point3D(int(pid))
    far = ok & (rng >= 40.0)
    out.update(deleted=int(bad.sum()), median_isolation=float(np.nanmedian(med)) if np.isfinite(med).any() else None,
               deleted_beyond_40m=int((bad & far).sum()), points_beyond_40m=int(far.sum()))
    return out


def drop_short_tracks(rec, min_track_length: int = 2, min_mixed_track_length: int = MIN_MIXED_TRACK_LENGTH) -> int:
    """Delete 3-D points observed in fewer than ``min_track_length`` images, and Navcam + Mastcam-Z points in fewer
    than ``min_mixed_track_length`` (v0p86); returns the number deleted."""
    short = [int(pid) for pid, pt in rec.points3D.items() if pt.track.length() < min_track_length]
    for pid in short:
        rec.delete_point3D(pid)
    return len(short) + drop_mixed_short_tracks(rec, min_mixed_track_length)


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


HAZCAM_FAMILIES = ("F", "R")


def hazcam_coverage(rec, project: SfmProject) -> Dict[str, Any]:
    """The Hazcam frames in a solution (0.93): per eye (FL, FR, RL, RR) the project images, the registered ones, the
    ones with tie points and their observations, and the tie points shared with Navcam ({} without Hazcams)."""
    by_name = {r["name"]: r for r in project.images}
    fam = {int(i): str((by_name.get(im.name) or {}).get("instrument", "?"))[:1] for i, im in rec.images.items()}
    eye_of = {int(i): str((by_name.get(im.name) or {}).get("instrument", ""))[:2] for i, im in rec.images.items()}
    eyes = sorted({str(r["instrument"])[:2] for r in project.images if str(r["instrument"])[:1] in HAZCAM_FAMILIES})
    if not eyes:
        return {}
    out: Dict[str, Any] = {e: {"images": sum(1 for r in project.images if str(r["instrument"])[:2] == e),
                               "registered": 0, "with_points": 0, "observations": 0} for e in eyes}
    reg = set(int(i) for i in rec.reg_image_ids())
    n_obs: Dict[int, int] = {}
    nav_haz = 0
    for pt in rec.points3D.values():
        fs = set()
        for el in pt.track.elements:
            iid = int(el.image_id)
            fs.add(fam.get(iid))
            if fam.get(iid) in HAZCAM_FAMILIES:
                n_obs[iid] = n_obs.get(iid, 0) + 1
        nav_haz += ("N" in fs) and bool(fs & set(HAZCAM_FAMILIES))
    for iid, e in eye_of.items():
        if e not in out:
            continue
        out[e]["registered"] += iid in reg
        out[e]["with_points"] += n_obs.get(iid, 0) > 0
        out[e]["observations"] += n_obs.get(iid, 0)
    return {"eyes": out, "navcam_hazcam_points": int(nav_haz)}


# a Hazcam eye refines its intrinsics (hazcam_intrinsics) only with at least this many tie-point observations in the
# block; fewer, and it is held at its start (the label fit).  Rockytop (22 Oct, the example set): the front eyes had
# 3000-5000 observations and refined f by 0.25 %; the rear eyes 60-300 (quarter-resolution frames looking at the
# rover's shadow) and, refined, moved fx 2.3 % and fy 4.6 %, and one rear frame 1 m from its station
HAZCAM_MIN_REFINE_OBSERVATIONS = 500           # 0.93.5 (Christian, 22 Oct): was 1000


def hazcam_weak_eyes(cov: Dict[str, Any], min_observations: int = HAZCAM_MIN_REFINE_OBSERVATIONS) -> List[str]:
    """The Hazcam eyes of :func:`hazcam_coverage` with fewer than ``min_observations`` tie-point observations."""
    return sorted(e for e, v in ((cov or {}).get("eyes") or {}).items() if v["observations"] < int(min_observations))


def reset_cameras_to_start(rec, project: SfmProject, keys) -> List[str]:
    """Put the cameras ``keys`` of ``rec`` back to their project start parameters (a weak Hazcam eye held at its
    start, 0.93.1); returns the keys reset."""
    db = (project.settings.get("database") or {}).get("cameras") or {}
    out = []
    for k in sorted(keys):
        cid, spec = db.get(k), project.cameras.get(k)
        if cid is None or spec is None or int(cid) not in rec.cameras:
            continue
        cam = rec.cameras[int(cid)]
        p = np.asarray(spec["params"], float)
        if len(p) == len(cam.params):
            cam.params = p
            out.append(k)
    return out


# a Hazcam frame is fixed to the rover body: in stage 2 its position prior moves with the shift the Navcam block found
# for its station (the rover's localisation error) and tightens to this sigma per axis.  Rockytop (22 Oct): with the
# 1 m waypoint prior, a rear Hazcam pair tied by 48 points slid 1.3 m from its station (0.12 m for the Navcam frames)
HAZCAM_STATION_PRIOR_SIGMA_M = 0.1


def label_prior_C(r: Dict[str, Any]) -> np.ndarray:
    """The label (waypoint + CAHV) position prior of a project image: ``prior_C_label`` when stage 2 moved the
    image's prior (:func:`hazcam_station_priors`), else ``prior_C``.  Shifts are measured from it (0.94.4: the moved
    Hazcam prior made every Hazcam frame look its station's shift away from the Navcam frames)."""
    return np.asarray(r.get("prior_C_label") or r["prior_C"], float)


def hazcam_station_priors(rec, project: SfmProject, sigma_m: float = HAZCAM_STATION_PRIOR_SIGMA_M
                          ) -> Dict[str, np.ndarray]:
    """Move the position prior of every Hazcam image of ``project`` by its station's median Navcam shift (solved
    minus prior centre of the registered Navcam images of ``rec``) and give it ``sigma_m`` (``prior_sigma_m``); the
    label prior stays in ``prior_C_label``.  Stations without registered Navcam images keep the waypoint prior.
    Returns {station: shift (3,)}."""
    by_name = {r["name"]: r for r in project.images}
    d: Dict[str, List[np.ndarray]] = {}
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        r = by_name.get(im.name)
        if r is None or not str(r.get("instrument", "")).startswith("N") or r.get("prior_C") is None:
            continue
        d.setdefault(r["station"], []).append(np.asarray(im.projection_center(), float) - np.asarray(r["prior_C"], float))
    shifts = {st: np.median(np.asarray(v), axis=0) for st, v in d.items()}
    out: Dict[str, np.ndarray] = {}
    for r in project.images:
        if str(r.get("instrument", ""))[:1] not in HAZCAM_FAMILIES or r.get("station") not in shifts:
            continue
        r.setdefault("prior_C_label", list(r["prior_C"]))
        r["prior_C"] = (np.asarray(r["prior_C_label"], float) + shifts[r["station"]]).tolist()
        r["prior_sigma_m"] = float(sigma_m)
        r["prior_station_shift_m"] = shifts[r["station"]].tolist()
        out[r["station"]] = shifts[r["station"]]
    return out


def hazcam_line(cov: Dict[str, Any]) -> str:
    """One log line of :func:`hazcam_coverage`."""
    if not cov:
        return "no Hazcam frames"
    return ("Hazcam frames with tie points " + ", ".join(f"{e} {v['with_points']}/{v['images']} ({v['observations']} obs)"
                                                          for e, v in cov["eyes"].items())
            + f"; Navcam-Hazcam tie points {cov['navcam_hazcam_points']}")


OUTLIER_DEFAULTS = {"residual_factor": 3.0, "min_residual_px": 2.0,   # 2.0 px (Christian, 22 Oct 2026; 1.6 since v0p61, v0p50 1.2)
                    "shift_mad_factor": 5.0, "min_shift_m": 0.25,
                    "attitude_mad_factor": 5.0, "min_attitude_deg": 0.5, "min_observations": None,
                    "min_navcam_ties": None, "lenient_zooms": ()}
# a Mastcam-Z frame needs at least this many valid tie points with an anchor camera (points seen by the frame and by an
# anchor image with a third look, MIN_MIXED_TRACK_LENGTH) - fewer, and it is excluded like an outlier (Christian,
# 20 Oct).  The anchors (21 Oct): Navcam for 34 and 48 mm; Navcam or a Mastcam-Z image at a wider zoom (shorter focal
# length) for 63 mm and longer, whose pixels are 5-10x finer than Navcam's and rarely match it directly
ZCAM_MIN_NAVCAM_TIES = 10
ZCAM_NAVCAM_ONLY_MAX_ZOOM = 48
# a frame with fewer tie-point observations than this is held at its prior in the bundle adjustment and
# flagged by the outlier test (was 30; good frames with 20-29 observations were being dropped).  Notebook 03:
# MIN_FRAME_OBSERVATIONS; reconstruct(min_frame_observations=) sets it for every stage of one run.
MIN_FRAME_OBSERVATIONS = 20


def _zcam_terrain(rec) -> Dict[str, Any]:
    """:func:`mppp.sfm.health.zcam_terrain` (the Mastcam-Z points above / below the Navcam terrain), never raising."""
    from .health import zcam_terrain
    try:
        return zcam_terrain(rec, sample=100000)
    except Exception as e:                                        # noqa: BLE001
        return {"note": f"{type(e).__name__}: {e}"}


def _zcam_terrain_line(t: Dict[str, Any]) -> str:
    from .health import zcam_terrain_line
    return zcam_terrain_line(t)


def _zoom_of(name: str) -> Optional[int]:
    """The Mastcam-Z zoom (mm) in a PDS image name (``..._0340LUA03``: 34), None for other cameras or names."""
    s = Path(name).stem.upper()
    if not s.startswith("Z") or len(s) < 54:
        return None
    try:
        return int(s[45:48])
    except ValueError:
        return None


# 0.94.4 (Christian, 24 Oct; Chal Rocks: 136 of the 214 79 mm frames excluded with "0 tie points with Navcam or a
# wider zoom"): a tie with a wider zoom counts with two looks (a 79 mm point seen by one 34 mm image); and a frame of a
# long zoom with fewer direct ties is anchored through its own mosaic - ZCAM_MOSAIC_TIES tie points with frames of its
# zoom that are anchored (directly or so).  At Chal Rocks a 79 mm frame shares a median 13,400 points, nearly all with
# the other 79 mm frames, and 1-2 with a 34 mm one: 57 of 170 were anchored directly, the other 113 through the mosaic.
MIN_WIDER_ZOOM_TRACK_LENGTH = 2
ZCAM_MOSAIC_TIES = 50


def navcam_tie_counts(rec, min_mixed_track_length: int = MIN_MIXED_TRACK_LENGTH,
                      navcam_only_max_zoom: Optional[int] = ZCAM_NAVCAM_ONLY_MAX_ZOOM,
                      min_wider_track_length: int = MIN_WIDER_ZOOM_TRACK_LENGTH) -> Dict[int, int]:
    """Per registered Mastcam-Z image, its valid tie points with an anchor: 3-D points it observes whose track also
    holds an anchor image.  The anchors are the Navcam images (the track at least ``min_mixed_track_length`` long:
    the third look); for a zoom above ``navcam_only_max_zoom`` (48 mm) also the Mastcam-Z images at a shorter zoom (a
    63 mm frame ties through 34 or 48 mm, a 110 mm frame through 34-79 mm; 0.94.4: from ``min_wider_track_length``
    looks, 2).  ``navcam_only_max_zoom=None``: Navcam only for every zoom."""
    return zcam_anchor_ties(rec, min_mixed_track_length, navcam_only_max_zoom, min_wider_track_length)[0]


def zcam_anchor_ties(rec, min_mixed_track_length: int = MIN_MIXED_TRACK_LENGTH,
                     navcam_only_max_zoom: Optional[int] = ZCAM_NAVCAM_ONLY_MAX_ZOOM,
                     min_wider_track_length: int = MIN_WIDER_ZOOM_TRACK_LENGTH):
    """(:func:`navcam_tie_counts`, {image: {same-zoom image: shared points}} of the zooms above
    ``navcam_only_max_zoom``)."""
    fam = {int(i): rec.images[i].name[:1] for i in rec.reg_image_ids()}
    zoom = {i: _zoom_of(rec.images[i].name) for i, f in fam.items() if f == "Z"}
    out = {i: 0 for i in zoom}
    share: Dict[int, Dict[int, int]] = {}
    for pt in rec.points3D.values():
        L = pt.track.length()
        if L < min(min_mixed_track_length, min_wider_track_length):
            continue
        ids = {int(el.image_id) for el in pt.track.elements}
        zs = [i for i in ids if i in zoom]
        if not zs:
            continue
        has_nav = L >= min_mixed_track_length and any(fam.get(i) == "N" for i in ids)
        widest = min((zoom[i] for i in zs if zoom[i] is not None), default=None)
        for i in zs:
            long_zoom = navcam_only_max_zoom is not None and zoom[i] is not None and zoom[i] > navcam_only_max_zoom
            if has_nav:
                out[i] += 1
            elif long_zoom and L >= min_wider_track_length and widest is not None and widest < zoom[i]:
                out[i] += 1                                   # tied through a wider Mastcam-Z zoom
            if long_zoom:
                for j in zs:
                    if j != i and zoom[j] == zoom[i]:
                        d = share.setdefault(i, {})
                        d[j] = d.get(j, 0) + 1
    return out, share


def uncalibrated_zooms(rec) -> List[int]:
    """The Mastcam-Z zooms of a reconstruction without a promoted focus model (:func:`mppp.calib.load_zcam`)."""
    from .. import calib
    out = []
    names = [rec.images[i].name for i in rec.reg_image_ids()]
    for z in sorted({_zoom_of(n) for n in names if n[:1] == "Z"} - {None}):
        try:
            if calib.load_zcam(int(z)) is None:
                out.append(int(z))
        except Exception:                                    # noqa: BLE001 - unreadable counts as uncalibrated
            out.append(int(z))
    return out


def mosaic_anchored(ties: Dict[int, int], share: Dict[int, Dict[int, int]], min_ties: int,
                    min_mosaic: int = ZCAM_MOSAIC_TIES) -> set:
    """The long-zoom images anchored through their mosaic: with fewer than ``min_ties`` direct anchor ties but
    ``min_mosaic`` tie points with same-zoom images that are anchored (directly, or through the mosaic in turn)."""
    anchored = {i for i, n in ties.items() if n >= min_ties}
    through: set = set()
    grew = True
    while grew:
        grew = False
        for i, d in share.items():
            if i in anchored or i in through:
                continue
            if sum(c for j, c in d.items() if j in anchored or j in through) >= min_mosaic:
                through.add(i)
                grew = True
    return through


def find_outlier_frames(rec, project: SfmProject, **thresholds) -> List[Dict[str, Any]]:
    """
    Frames (a Navcam stereo pair, or one Mastcam-Z image) that do not align with
    the majority.  A frame is flagged when

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
      attitude is one mast pointing, whose error is per image;
    * (a Mastcam-Z frame, Navcam images registered) it has fewer than
      ``min_navcam_ties`` (:data:`ZCAM_MIN_NAVCAM_TIES`, 10) valid tie points
      with an anchor (:func:`navcam_tie_counts`: Navcam for 34 and 48 mm, Navcam or a
      wider Mastcam-Z zoom for 63-110 mm; a third look on each).

    Returns one record per flagged frame: frame id, images, station, reasons,
    and ``kind``: ``"unconstrained"`` for a frame with no tie points at all
    (e.g. a defocused focus-stack member or a calibration-target image; it was
    never aligned and is left out of the export) or ``"outlier"``.
    """
    from scipy.spatial.transform import Rotation
    th = dict(OUTLIER_DEFAULTS, **thresholds)
    if th.get("min_observations") is None:                   # the same limit as the bundle adjustment
        th["min_observations"] = MIN_FRAME_OBSERVATIONS
    if th.get("min_navcam_ties") is None:
        th["min_navcam_ties"] = ZCAM_MIN_NAVCAM_TIES
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
        dC = np.asarray(C) - label_prior_C(r0)            # 0.94.4: from the label prior (a Hazcam's was moved)
        dR = Rotation.from_matrix(np.asarray(r0["prior_R_w2c"], float).T @ R).as_rotvec()   # world frame: a rover tilt is common to a station
        obs = sum(n_img.get(i, 0) for i in ims)
        meds = [med_img[i] for i in ims if i in med_img]
        ins = str(rows[0]["instrument"])
        frames[fid] = {"frame_id": int(fid), "images": [rec.images[i].name for i in ims], "station": rows[0]["station"],
                       "family": "Z" if ins.startswith("Z") else "H" if ins[:1] in HAZCAM_FAMILIES else "N",
                       "observations": obs,
                       "median_residual_px": float(np.median(meds)) if meds else float("inf"), "dC": dC, "dR": dR,
                       "reasons": []}
    if not frames:
        return []
    ties, share = zcam_anchor_ties(rec) if any(f["family"] == "N" for f in frames.values()) else ({}, {})
    if ties and int(th["min_navcam_ties"]) > 0:
        through = mosaic_anchored(ties, share, int(th["min_navcam_ties"]))
        for fid, f in frames.items():
            if f["family"] != "Z" or not f["observations"]:
                continue
            n = sum(ties.get(int(d.id), 0) for d in rec.frames[fid].data_ids)
            f["navcam_ties"] = n
            if n < int(th["min_navcam_ties"]) and any(int(d.id) in through for d in rec.frames[fid].data_ids):
                f["anchored_through_mosaic"] = True             # 0.94.4: a long zoom's mosaic ties it
                continue
            if n < int(th["min_navcam_ties"]):
                z = _zoom_of(f["images"][0])
                anchor = "Navcam" if z is None or z <= ZCAM_NAVCAM_ONLY_MAX_ZOOM else "Navcam or a wider zoom"
                f["reasons"].append(f"{n} tie points with {anchor} < {int(th['min_navcam_ties'])}")
    all_med = np.median([f["median_residual_px"] for f in frames.values() if np.isfinite(f["median_residual_px"])])
    lim = max(th["residual_factor"] * all_med, th["min_residual_px"])
    lenient = {int(z) for z in (th.get("lenient_zooms") or ())}
    for f in frames.values():
        if f["family"] == "Z" and lenient and _zoom_of(f["images"][0]) in lenient:
            pass                    # 0.94.4: a zoom without a calibrated focus model, its focus lines still being fitted
        elif f["observations"] and f["median_residual_px"] > lim:
            f["reasons"].append(f"median residual {f['median_residual_px']:.2f} px > {lim:.2f}")
        if f["observations"] == 0:
            f["reasons"].append("no tie points")
        elif f["observations"] < th["min_observations"]:
            f["reasons"].append(f"{f['observations']} observations < {th['min_observations']}")
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for f in frames.values():
        groups.setdefault((f["station"], f["family"]), []).append(f)
    for (station, fam), fs in groups.items():
        # 0.94.4: the Hazcam frames of a station (one rover pose, few frames) are tested against the station's
        # Navcam frames when it has three or more, else among themselves; never on their attitude
        ref = groups.get((station, "N")) if fam == "H" else None
        ref = ref if ref is not None and len(ref) >= 3 else (fs if len(fs) >= 3 else None)
        if ref is None:
            continue
        D_ref = np.array([f["dC"] for f in ref])
        med_ref = np.median(D_ref, axis=0)
        mad = 1.4826 * np.median(np.linalg.norm(D_ref - med_ref, axis=1))
        D = np.array([f["dC"] for f in fs])
        dev = np.linalg.norm(D - med_ref, axis=1)
        A = np.array([f["dR"] for f in fs])
        adev = np.degrees(np.linalg.norm(A - np.median(A, axis=0), axis=1)) if len(fs) >= 3 else np.zeros(len(fs))
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
    Start and refined parameters of every camera of a solved block, for the notebook 03 print-out.
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
            logger.info(f"{r['camera']}: not in the reconstruction")
            continue
        tag = ("  [temperature bin]" if r["thermal_bin"] else
               "  [no images after the thermal split: values before it]" if r["images"] == 0 else "")
        logger.info(f"{r['camera']} ({r['images']} images){tag}: " + " ".join(
            f"{n} {a:.4g}->{b:.4g}" for n, a, b in zip(r["names"][:n_params], r["start"], r["refined"])))
    return rows


CONVERGENCE_BINS_DEG = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 180.0)


def convergence_statistics(rec, project: SfmProject, bins_deg: Sequence[float] = CONVERGENCE_BINS_DEG,
                           max_points: Optional[int] = None, seed: int = 0) -> Dict[str, Any]:
    """
    Tie points by convergence angle: for each 3-D point the largest
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


NETWORK_MIN_STATIONS = 4        # Navcam intrinsics are refined only from networks at least this strong
NETWORK_MIN_SPAN_M = 5.0


def navcam_network(project: SfmProject, min_stations: int = NETWORK_MIN_STATIONS,
                   min_span_m: float = NETWORK_MIN_SPAN_M) -> Dict[str, Any]:
    """
    How well the Navcam network can constrain the Navcam intrinsics:
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


def restore_frames(rec, source, frame_ids: Sequence[int]) -> int:
    """
    put frames of ``source`` that ``rec`` no longer has back into ``rec`` - with their images
    (keypoints, camera), cameras and rigs as needed - unregistered and at ``source``'s pose.  pycolmap 4 drops
    deregistered frames and their images when a model is written and read back, as ``triangulate_points`` does,
    so the Mastcam-Z frames set aside in stage 1 of a staged reconstruction were gone in stage 2
    (``KeyError`` in ``reconstruct``).  Returns the number of frames restored.
    """
    import pycolmap
    n = 0
    for fid in frame_ids:
        fid = int(fid)
        if fid in rec.frames:
            continue
        fr = source.frames[fid]
        # cameras first - every camera of the frame's images and every sensor of its rig.  COLMAP's
        # tear_down (inside triangulate_points) also drops rigs and cameras no registered frame uses, so a
        # Mastcam-Z stereo rig (ZCAM_RIG) and both its cameras are gone after stage 1, and AddRig checks that
        # all of a rig's cameras exist ("Camera 28 from rig 27 not found").
        cids = {int(source.images[d.id].camera_id) for d in fr.data_ids}
        if fr.rig_id not in rec.rigs:
            cids |= {int(s.id) for s in source.rigs[fr.rig_id].sensor_ids()
                     if s.type == pycolmap.SensorType.CAMERA}
        for cid in sorted(cids):
            if cid not in rec.cameras:
                rec.add_camera(source.cameras[cid])
        if fr.rig_id not in rec.rigs:
            rec.add_rig(source.rigs[fr.rig_id])
        nf = pycolmap.Frame(frame_id=fid, rig_id=fr.rig_id)
        for d in fr.data_ids:
            nf.add_data_id(d)
        if fr.has_pose:
            nf.rig_from_world = fr.rig_from_world
        rec.add_frame(nf)
        for d in fr.data_ids:
            if d.id in rec.images:
                continue
            im = source.images[d.id]
            kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
            ni = pycolmap.Image(name=im.name, keypoints=kps, camera_id=im.camera_id, image_id=d.id)
            ni.frame_id = fid
            rec.add_image(ni)
        n += 1
    return n


def _frames_of_family(rec, project: SfmProject, family: str) -> List[int]:
    by_name = {r["name"]: r for r in project.images}
    out = []
    for fid, fr in rec.frames.items():
        names = [rec.images[d.id].name for d in fr.data_ids if d.id in rec.images]
        if names and str(by_name[names[0]]["instrument"]).startswith(family):
            out.append(int(fid))
    return out


def _reconstruct(project: SfmProject, sigma_px: float = 0.5,
                schedule: Sequence[Sequence[float]] = DEFAULT_SCHEDULE,
                refine_intrinsics: bool = True, refine_rig: Union[bool, str] = "rotation",
                register: bool = False, max_iterations: int = 50, out_name: str = "cahv_ba",
                verbose: bool = True, min_track_length: int = 2, min_tri_angle_deg: float = 0.25,
                refine_tangential: Union[bool, Dict[str, bool]] = True, gui_native: bool = True,
                attitude_prior_deg: Optional[float] = ATTITUDE_PRIOR_DEG,
                exclude_outliers: bool = False, outlier_thresholds: Optional[Dict[str, float]] = None,
                exclude_after_round: int = 2, rig_translation_sigma_m: Optional[float] = None,
                triangulation_options: Optional[Dict[str, Any]] = None,
                navcam_intrinsics: str = "refine", staged: bool = False, hold_cameras: Sequence[str] = (),
                linear_solver: str = "auto", temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
                thermal_bins_deg: Optional[float] = None, thermal_min_images: int = 5,
                thermal_free: Sequence[str] = ("fx", "fy", "k1"),
                thermal_model: Optional[Dict[str, Dict[str, float]]] = None, localize_min_images: int = 0,
                thermal_fine_range: Any = "default", thermal_fine_deg: Any = "default",
                zcam_backlash: Optional[str] = None, zcam_backlash_z: float = 3.0,
                zcam_focus_line: bool = True, zcam_line_cycles: int = 2, thermal_after_stage1: bool = True,
                pose_guided: Union[bool, Dict[str, Any]] = True, navcam_aspect: str = "hold",
                isolated_points_factor: float = ISOLATED_POINTS_FACTOR,
                round1_hold: bool = ROUND1_HOLD):
    """
    CAHV-initialised triangulation + weighted BA (see module docstring).

    ``round1_hold`` (v0p80, notebook 03 ``ROUND1_HOLD``; the camera-model recipe of 14 Oct 2026): the first round
    (24 px triangulation, loose loss) adjusts poses and points only - every camera and the rig held at their start
    (the promoted models) - so the cameras are refined from round 2 on, on clean tracks.  With ``refine_rig=True``
    (the baseline study: translation free) the final Navcam adjustment also estimates the rig covariance, and
    ``project.settings["reconstruction"]["rig_fit"]`` holds the baseline, its sd and the per-epoch rotations.

    ``zcam_backlash`` (v0p40, notebook 03 ``ZCAM_BACKLASH``): ``"split"`` - after the final adjustment each
    Mastcam-Z focus group (eye, sol, sequence, focus count) is classified into the dominant backlash focus state
    (f about 1 % above the label) or the regular state (f about the label) from its images' own focal lengths, the
    regular images get their own cameras and the block is adjusted again (:func:`mppp.sfm.backlash.backlash_stage`;
    groups within ``zcam_backlash_z`` standard errors of the midpoint stay backlash); ``"report"`` classifies only;
    None: off.  The report is ``project.settings["zcam_backlash"]``.
    ``zcam_focus_line`` (v0p61, notebook 03 ``ZCAM_FOCUS_LINE``; staged runs): stage 2 starts with every Mastcam-Z
    camera held at its start (the focus model), then ``zcam_line_cycles`` times lets the focus-bin focal lengths free
    for one round and puts them back on one line per eye and zoom fitted to them
    (:func:`mppp.sfm.zcam.refit_focus_lines`: f0 and slope in focus count; outlier bins reported, not followed);
    the remaining rounds and the final adjustment hold them on the line.  Distortion, principal point (linear in
    focus from the model) and the zoom's rig stay one per eye / zoom.  The lines are in
    ``project.settings["zcam_focus_lines"]``.  ``pose_guided`` (v0p64, notebook 03 ``POSE_GUIDED``; staged runs with
    Mastcam-Z): after the stage-2 rounds, matches guided by the poses (:mod:`mppp.sfm.guided`; a dict overrides
    ``GUIDED_DEFAULTS``) go into ``<project>/database_guided.db``, the block is triangulated from it and adjusted
    once more; the report (Navcam-Mastcam-Z tie points before / after) is in ``project.settings["pose_guided"]``.
    ``navcam_aspect`` (v0p65, notebook 03 ``NAVCAM_ASPECT``): "hold" (default) keeps every Navcam camera's fy / fx at
    its start (the consensus pixel aspect) in all bundle adjustments of the run, the thermal stage included;
    "free" refines fx and fy independently.  ``thermal_after_stage1``: with ``staged``, the Navcam thermal
    stage runs at the end of stage 1, so the Mastcam-Z frames are added to the final Navcam cameras (temperature bins).
    ``localize_min_images`` (v0p35.1, notebook 03 ``LOCALIZE_MIN_IMAGES``): stations (site, drive) with fewer images
    than this are not held by their waypoint position (see :func:`unlocalized_stations`, :func:`bundle_adjust`).
    Before the first round they are moved onto the rest of the block as rigid stations
    (:func:`register_stations` with ``only``), so that triangulation can join their tracks; then every adjustment
    leaves out their position priors.  Recorded in ``project.settings["localize_min_images"]`` and
    ``project.settings["reconstruction"]["unlocalized_stations"]``.  0 (default) keeps every prior.

    ``navcam_intrinsics``: ``"refine"`` (default since the
    Navcam intrinsics are always refined), ``"hold"`` (the Navcam cameras stay
    at their start values, the shipped consensus unless the project was created
    from other cameras) or ``"auto"`` (hold when :func:`navcam_network` calls
    the network weak).  The network verdict is logged whatever the setting.  ``refine_rig="auto"``
    refines the rig rotation on a strong network and holds it (at the start rig, turned to the block's camera
    temperature when the start rig has a temperature model) on a weak one.  ``refine_rig=False``
    holds the rig.  ``hold_cameras``: further camera keys to hold.
    ``staged``: with Mastcam-Z in the project, solve the Navcam
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
    Default: four rounds; the fourth repeats the third's limits to
    show whether one more re-triangulation and adjustment changes anything.
    The final adjustment uses the last round's Cauchy scale and cut-off.
    ``attitude_prior_deg`` (v0p20, default 1 deg): weak CAHV attitude prior
    per frame; see :func:`bundle_adjust`.
    ``gui_native``: also write the native-pixel viewing copy
    ``gui_native/`` for the COLMAP GUI (``open_in_colmap.bat``).
    ``register``: first move whole stations with :func:`register_stations`
    (needed when the waypoint priors are off by more than ~20 px at range;
    at Belva they are within ~1-2 px, so it is off by default).
    ``min_track_length``: shortest track kept.  Default 2: two-view
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
    ``exclude_outliers``: after round ``exclude_after_round`` and
    again before the final adjustment, frames that do not align with the
    majority (:func:`find_outlier_frames`, ``outlier_thresholds``) are
    deregistered; the rounds after that re-triangulate without them.  The
    list is in the log and ``project.settings["reconstruction"]["excluded"]``.
    ``rig_translation_sigma_m``: see :func:`bundle_adjust` (with ``refine_rig=True``).
    ``linear_solver``: passed to :func:`bundle_adjust`.
    ``thermal_bins_deg`` (v0p31, with ``temperatures`` from :func:`mppp.sfm.thermal.image_temperatures`): after the
    final adjustment, split the Navcam cameras into camera-temperature bins of this width
    (:func:`mppp.sfm.thermal.thermal_stage`; bins with fewer than ``thermal_min_images`` images are merged) and
    adjust once more with the bins' ``thermal_free`` parameters refined - or, when the Navcam intrinsics are held,
    with every bin held at the eye's camera scaled to the bin temperature by ``thermal_model``
    ({"NL": {"ppm_per_degC", "T0_degC"}, ...}).  The binned solution is the delivered one; the solution with one
    camera per eye stays in ``sparse/<out_name>_single`` and the bin table in
    ``project.settings["thermal"]``.
    ``triangulation_options``: extra keyword arguments for
    :func:`triangulate` (e.g. ``min_angle_deg`` per round is set from
    ``min_tri_angle_deg``; ``create_max_angle_error``, ``complete``...).
    Returns the reconstruction (also written to ``sparse/<out_name>``).
    """
    if min_track_length < 2:
        raise ValueError("min_track_length must be >= 2")
    if navcam_intrinsics not in ("auto", "refine", "hold"):
        raise ValueError("navcam_intrinsics must be 'auto', 'refine' or 'hold'")
    if navcam_aspect not in ("free", "hold"):
        raise ValueError("navcam_aspect must be 'free' or 'hold'")
    project.settings["navcam_aspect"] = navcam_aspect          # read by every bundle adjustment of the run
    import time
    from .thermal import strip_thermal_bins
    t0 = time.time()
    # finer Navcam temperature bins where the images are dense ("default": THERMAL_FINE_RANGE_DEGC,
    # THERMAL_FINE_BIN_DEG of mppp.sfm.thermal; None or a falsy value: uniform bins of thermal_bins_deg)
    from .thermal import THERMAL_FINE_RANGE_DEGC, THERMAL_FINE_BIN_DEG
    _fine_range = THERMAL_FINE_RANGE_DEGC if isinstance(thermal_fine_range, str) else (thermal_fine_range or None)
    _fine_deg = THERMAL_FINE_BIN_DEG if isinstance(thermal_fine_deg, str) else (thermal_fine_deg or None)
    n_strip = strip_thermal_bins(project)                   # a rerun starts from one camera per eye
    if n_strip and verbose:
        logger.info(f"{n_strip} Navcam temperature-bin cameras of an earlier run removed; starting from one camera "
              f"per eye")
    rec = initial_reconstruction(project)
    init = copy.deepcopy(rec)
    log: List[Dict[str, Any]] = []
    project.settings["localize_min_images"] = int(localize_min_images or 0)
    unloc = unlocalized_stations(project, localize_min_images) if localize_min_images else {}
    unloc_reg = None
    if unloc and any(v["prior"] == "dropped" for v in unloc.values()):
        # place the short stops on the block before triangulating with the round-1 threshold
        rec = triangulate(rec, project, max_reproj_px=8.0, min_angle_deg=min_tri_angle_deg)
        unloc_reg = register_stations(rec, project, only=sorted(unloc), verbose=verbose)
        log.append({"unlocalized_registration": unloc_reg})
        if verbose:
            logger.info(f"LOCALIZE_MIN_IMAGES {localize_min_images}: {len(unloc)} stations without a position prior "
                  f"({', '.join(sorted(unloc))}); registered onto the block: "
                  f"{sum(1 for v in unloc_reg['stations'].values() if v.get('registered'))}")
    network = navcam_network(project)
    hold = set(hold_cameras)
    for _v in (project.settings.get("zcam_virtual_rig") or {}).values():      # every run starts at the start rig
        if isinstance(_v, dict):
            _v.pop("R_refined", None)
            _v.pop("t_refined", None)
    if navcam_intrinsics == "hold" or (navcam_intrinsics == "auto" and network["verdict"] == "weak"):
        hold |= {k for k in project.cameras if str(k).startswith("N")}
    network["navcam_intrinsics"] = "hold" if any(str(k).startswith("N") for k in hold) else "refine"
    hazkeys = {k for k in project.cameras if str(k)[:1] in HAZCAM_FAMILIES}       # 0.93: the Hazcam eyes
    if project.settings.get("hazcam") is not None:
        project.settings["hazcam"]["rig_hold"] = []
    if hazkeys and (project.settings.get("hazcam") or {}).get("intrinsics") == "hold":
        hold |= hazkeys
    if refine_rig == "auto":                                # the rig follows the intrinsics' network verdict
        refine_rig = "rotation" if network["verdict"] == "strong" else False
    network["navcam_rig"] = "rotation" if refine_rig else "hold"
    if verbose:
        logger.info(f"Navcam network: {network['stations']} stations over {network['span_m']} m -> {network['verdict']}; "
              f"Navcam intrinsics {network['navcam_intrinsics']}" + (f", holding {sorted(hold)}" if hold else ""))
    log.append({"navcam_network": network})
    z_frames = _frames_of_family(rec, project, "Z")
    # 0.93: the Hazcam frames come in with the Mastcam-Z ones in stage 2 (Navcam held) - their start cameras are the
    # label fits, not a calibration, and must not pull the Navcam block
    h_frames = [f for fam_ in HAZCAM_FAMILIES for f in _frames_of_family(rec, project, fam_)]
    late = z_frames + h_frames
    staged = bool(staged and late and len(late) < len(rec.frames))
    if staged:
        exclude_frames(rec, late)
        if verbose:
            logger.info(f"stage 1: Navcam only ({len(z_frames)} Mastcam-Z frames"
                        + (f", {len(h_frames)} Hazcam frames" if h_frames else "") + " set aside)")
    if register:
        rec = triangulate(rec, project, max_reproj_px=8.0)
        reg = register_stations(rec, project, verbose=verbose)
        log.append({"station_registration": reg})
    excluded: List[Dict[str, Any]] = []
    topts = dict(triangulation_options or {})

    def _exclude(tag, navcam_ties: bool = False):
        # the Mastcam-Z frames' Navcam-tie minimum only once their matches are complete (after the pose-guided
        # matching, before the final adjustment); earlier exclusions test residuals and observations only
        th_ = dict(outlier_thresholds or {})
        if not navcam_ties:
            th_["min_navcam_ties"] = 0
            # 0.94.4 (Chal Rocks 79 mm, 24 Oct): before the final test the zooms without a calibrated focus model are
            # not judged on their residuals - their focal lengths start at the labels (0.5-0.7 % off at 79 mm) and
            # are only fitted from round 2
            th_.setdefault("lenient_zooms", uncalibrated_zooms(rec))
        found = find_outlier_frames(rec, project, **th_)
        new = [f for f in found if f["frame_id"] not in {e["frame_id"] for e in excluded}]
        if new:
            exclude_frames(rec, [f["frame_id"] for f in new])
            for f in new:
                f["when"] = tag
            excluded.extend(new)
        if verbose:
            logger.info(f"{tag}: {len(new)} frames excluded as outliers"
                  + ("".join(f"\n      {', '.join(f['images'])}: {'; '.join(f['reasons'])}" for f in new[:20])
                     + ("\n      ..." if len(new) > 20 else "")))
        log.append({"excluded_frames": new, "when": tag})

    for k, (tpx, loss, rmax) in enumerate(schedule):
        rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg, **topts)
        st0 = track_statistics(rec, project)
        held_round = bool(round1_hold) and k == 0 and len(schedule) > 1      # poses and points only
        ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(loss),
                           refine_intrinsics=refine_intrinsics and not held_round,
                           refine_tangential=refine_tangential, refine_rig=False if held_round else refine_rig,
                           max_iterations=max_iterations,
                           attitude_prior_deg=attitude_prior_deg, rig_translation_sigma_m=rig_translation_sigma_m,
                           hold_cameras=sorted(hold), linear_solver=linear_solver)
        n_bad = filter_observations(rec, project, float(rmax))
        n_bad += drop_short_tracks(rec, min_track_length)
        st = track_statistics(rec, project)
        res = native_residuals(rec, project)["residual_native_px"]
        entry = {"round": k + 1, "triangulation_px_full": tpx, "cauchy_scale_sigma": loss, "after_triangulation": st0,
                 "ba": ba["brief"], "filtered_observations": n_bad, "max_residual_native_px": rmax, "tracks": st,
                 "cameras_and_rig_held": held_round, "prior_gauge": ba.get("prior_gauge"),
                 "rms_native_px": float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan"),
                 "median_native_px": float(np.median(res)) if res.size else float("nan"),
                 "elapsed_s": round(time.time() - t0, 1)}
        log.append(entry)
        if verbose:
            logger.info(f"round {k + 1}{' (cameras and rig held)' if held_round else ''}: {st['points']} points "
                  f"({st['cross_station_fraction']:.2%} cross-station), "
                  f"median {entry['median_native_px']:.3f} / rms {entry['rms_native_px']:.3f} native px, "
                  f"{n_bad} obs filtered, {entry['elapsed_s']:.0f} s")
        if exclude_outliers and k + 1 == min(int(exclude_after_round), len(schedule)):
            _exclude(f"after round {k + 1}")
    if exclude_outliers:
        _exclude("before the final adjustment")
    ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(schedule[-1][1]),
                       refine_intrinsics=refine_intrinsics, refine_tangential=refine_tangential,
                       refine_rig=refine_rig, max_iterations=2 * max_iterations, attitude_prior_deg=attitude_prior_deg,
                       rig_translation_sigma_m=rig_translation_sigma_m, hold_cameras=sorted(hold), linear_solver=linear_solver,
                       covariance=refine_rig is True)
    rig_fit = navcam_rig_fit(ba, rec, project)            # the Navcam rig of the final Navcam adjustment
    # the final adjustment can push a few points behind a camera or past the limit
    n_bad = filter_observations(rec, project, float(schedule[-1][2]))
    n_bad += drop_short_tracks(rec, min_track_length)
    log.append({"final_ba": ba["brief"], "frames_held": ba["frames_held"], "filtered_after_final": n_bad,
                "tracks": track_statistics(rec, project), "rig_fit": rig_fit, "prior_gauge": ba.get("prior_gauge"),
                "no_position_prior_at": ba.get("no_position_prior_at")})
    if verbose and ba.get("prior_gauge"):
        g = ba["prior_gauge"]
        logger.info(f"position priors: mean residual (E, N, U) {np.round(g['mean_residual_m'], 3).tolist()} m over "
              f"{g['frames']} frames" + (" - block moved onto them" if g["recentred"] else ""))
    if verbose and rig_fit.get("baseline_m") is not None:
        logger.info(f"Navcam rig: baseline {rig_fit['baseline_m']:.5f} m"
              + (f" +- {1e3 * rig_fit['baseline_sd_m']:.2f} mm" if rig_fit.get("baseline_sd_m") else "")
              + f" (start {rig_fit['baseline_start_m']:.5f} m, translation {'refined' if refine_rig is True else 'held'})"
              + "".join(f"; {e['camera']}: pitch {e['pitch_mdeg']:+.2f} roll {e['roll_mdeg']:+.2f} yaw {e['yaw_mdeg']:+.2f} mdeg"
                        for e in rig_fit.get("epochs", [])))
    navcam_stage = None
    if staged:
        # stage 1 done: keep it, then bring the Mastcam-Z frames in at their priors with the Navcam block held
        out1 = project.root / "sparse" / f"{out_name}_navcam"
        out1.mkdir(parents=True, exist_ok=True)
        rec.write(str(out1))
        (out1 / "mppp_sfm_log.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
        navcam_stage = {"path": str(out1.relative_to(project.root)), "log": log,
                        "tracks": track_statistics(rec, project), "cameras": {}}
        if thermal_after_stage1 and thermal_bins_deg and temperatures:
            # the Navcam temperature bins before the Mastcam-Z frames come in (the Navcam-only model is the
            # one-camera-per-eye model of the block: sparse/<out>_single)
            from .thermal import thermal_stage
            single = project.root / "sparse" / f"{out_name}_single"
            single.mkdir(parents=True, exist_ok=True)
            rec.write(str(single))
            hold_nav = network["navcam_intrinsics"] == "hold" or any(str(k).startswith("N") for k in hold_cameras)
            rec, thermal_early = thermal_stage(rec, project, temperatures, bin_deg=float(thermal_bins_deg),
                                               min_images=int(thermal_min_images), free=tuple(thermal_free),
                                               hold=hold_nav, thermal_model=thermal_model, sigma_px=sigma_px,
                                               loss_scale=float(schedule[-1][1]), max_iterations=2 * max_iterations,
                                               attitude_prior_deg=attitude_prior_deg, linear_solver=linear_solver,
                                               verbose=verbose, fine_range=_fine_range, fine_deg=_fine_deg)
            n_bad = filter_observations(rec, project, float(schedule[-1][2])) + drop_short_tracks(rec, min_track_length)
            thermal_early["filtered_after"] = n_bad
            thermal_early["single_camera_model"] = str(single.relative_to(project.root))
            thermal_early["when"] = "end of stage 1 (Navcam only)"
            navcam_stage["thermal"] = {k: v for k, v in thermal_early.items() if k != "rows"}
        log = [{"stage": 2, "note": "Mastcam-Z" + (" and Hazcam" if h_frames else "")
                + " frames added at their priors; Navcam cameras and rig held"}]
        n_restored = restore_frames(rec, init, late)          # pycolmap 4 dropped them in stage 1
        if n_restored:
            log[0]["frames_restored"] = n_restored
        shifts = hazcam_station_priors(rec, project) if h_frames else {}
        if shifts:
            log[0]["hazcam_station_shifts_m"] = {st: [round(float(x), 4) for x in v] for st, v in shifts.items()}
            if verbose:
                logger.info("stage 2: Hazcam priors moved with their stations' Navcam solution (+-"
                            f"{HAZCAM_STATION_PRIOR_SIGMA_M:g} m): " + ", ".join(
                                f"{st} {np.linalg.norm(v):.2f} m" for st, v in sorted(shifts.items())))
        by_name_ = {r["name"]: r for r in project.images}
        for fid in late:
            T0 = init.frames[fid].rig_from_world
            names_ = [init.images[d.id].name for d in init.frames[fid].data_ids if d.id in init.images]
            st_ = (by_name_.get(names_[0]) or {}).get("station") if names_ else None
            if st_ in shifts and str((by_name_.get(names_[0]) or {}).get("instrument", ""))[:1] in HAZCAM_FAMILIES:
                import pycolmap as _pc
                R0 = np.asarray(T0.rotation.matrix())
                C0 = -R0.T @ np.asarray(T0.translation) + shifts[st_]
                T0 = _pc.Rigid3d(T0.rotation, -R0 @ C0)
            rec.frames[fid].rig_from_world = T0
            rec.register_frame(fid)
        # the Navcam frames now sit on temperature-bin rigs: triangulate against a database copy that has them
        db2 = sync_database(rec, project) if (navcam_stage.get("thermal") and project.database.is_file()) else None
        topts2 = dict(topts, database=db2) if db2 is not None else topts
        hold |= {k for k in project.cameras if str(k).startswith("N")}
        refine_rig = False
        hazcam_weak_held: set = set()
        if verbose:
            logger.info(f"stage 2: {len(z_frames)} Mastcam-Z frames"
                        + (f" and {len(h_frames)} Hazcam frames" if h_frames else "")
                        + " added; Navcam cameras and rig held"
                        + (f"; Hazcam eyes {sorted(hazkeys)} start at their label fits, held in round 1" if h_frames else ""))
        zkeys = {k for k in project.cameras if str(k).startswith("Z")}
        cycles = max(0, min(int(zcam_line_cycles), len(schedule) - 1)) if zcam_focus_line else 0
        # the Mastcam-Z rig (rotation, and with "full" the baseline) is refined while the Mastcam-Z cameras are
        # held on their focus lines - not in the rounds that free the bins (a ZR principal point and the rig rotation
        # are the same thing in a 25-degree field)
        zfit = project.settings.get("zcam_rig_fit")
        zfit = zfit if zfit in ("rotation", "full") else None
        lines = None
        # a Mastcam-Z eye must not drop out (v0p82 run: ZR034 lost every tie point after its cameras were set
        # on the focus lines, or after the pose-guided matches) - see mppp.sfm.zcam_guard
        from . import zcam_guard as ZG
        cov_prev, line_snap, undone = None, None, set()
        poses_prev, cams_prev, reg_prev = {}, {}, set()
        ZG.trace(rec, project, "stage2_start")
        for k, (tpx, loss, rmax) in enumerate(schedule):
            rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg, **topts2)
            ZG.trace(rec, project, f"r{k + 1}_tri", {"database": topts2.get("database"), "max_reproj_px": tpx})
            cov = ZG.group_coverage(rec, project)
            if cov_prev is not None and line_snap is not None:
                bad = [g for g in ZG.collapsed(cov_prev, cov) if g not in undone]
                if bad:
                    diag = ZG.diagnose(rec, project, bad, poses_prev, cams_prev, reg_prev, topts2.get("database"))
                    logger.info(f"stage 2 round {k + 1}: Mastcam-Z frames lost at the triangulation - "
                          f"{ZG.diagnosis_line(diag)}")
                    n_rest = ZG.restore(rec, line_snap, ZG.camera_ids_of(project, bad))
                    rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg, **topts2)
                    cov2 = ZG.group_coverage(rec, project)
                    if ZG.collapsed(cov_prev, cov2):                  # not the lines: the poses of the group back too
                        n_pose = ZG.restore_poses(rec, project, bad, poses_prev)
                        rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg,
                                          **topts2)
                        cov3 = ZG.group_coverage(rec, project)
                        logger.info(f"stage 2 round {k + 1}: still lost after the cameras were restored; "
                              f"{n_pose} frame poses restored too: {ZG.brief(cov3)}")
                        diag["after_pose_restore"], diag["poses_restored"] = cov3, n_pose
                        cov2 = cov3
                    log.append({"stage": 2, "round": k + 1, "zcam_guard": {
                        "collapsed": bad, "before": cov_prev, "after_lines": cov, "restored_cameras": n_rest,
                        "after_restore": cov2, "diagnosis": diag}})
                    logger.info(f"stage 2 round {k + 1}: {', '.join(bad)} lost its frames with tie points after the "
                          f"focus lines were applied ({ZG.brief(cov_prev)} -> {ZG.brief(cov)}); the line values are "
                          f"undone for it ({n_rest} cameras back to their refined values): {ZG.brief(cov2)}")
                    undone |= set(bad)
                    cov = cov2
            line_snap = None
            # round 1 with the Mastcam-Z cameras at the focus model, then `cycles` rounds with the bins' f
            # free (each followed by the refit of one line per eye and zoom), then held on the line
            zfree = (not zcam_focus_line) or (1 <= k <= cycles)
            hold_k = hold if zfree else (hold | zkeys)
            if k == 0 and h_frames and round1_hold and len(schedule) > 1:
                hold_k = hold_k | hazkeys                   # 0.93: the Hazcam frames are posed before their cameras move
                if project.settings.get("hazcam") is not None:      # 0.93.1: and before their rigs move
                    project.settings["hazcam"]["rig_hold"] = sorted({e[0] for e in hazkeys})
            elif h_frames:
                weak_h = set(hazcam_weak_eyes(hazcam_coverage(rec, project))) & hazkeys
                if weak_h:                                  # too few ties to refine a camera: held at the label fit
                    reset_cameras_to_start(rec, project, weak_h - hazcam_weak_held)
                    hold_k = hold_k | weak_h
                    hold |= weak_h                          # 0.93.1: and in every later adjustment (pose-guided too)
                    hazcam_weak_held.update(weak_h)
                if project.settings.get("hazcam") is not None:
                    project.settings["hazcam"]["rig_hold"] = sorted(
                        {e[0] for e in hazcam_weak_held if {e[0] + "L", e[0] + "R"} & hazkeys <= hazcam_weak_held})
            ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(loss), refine_intrinsics=refine_intrinsics,
                               refine_tangential=refine_tangential, refine_rig=False, max_iterations=max_iterations,
                               attitude_prior_deg=attitude_prior_deg, hold_cameras=sorted(hold_k), linear_solver=linear_solver,
                               refine_zcam_rig=zfit if (zcam_focus_line and not zfree) else None)
            ZG.trace(rec, project, f"r{k + 1}_ba", {"zfree": zfree, "refine_zcam_rig": zfit if (zcam_focus_line and not zfree) else None})
            cov_prev = ZG.group_coverage(rec, project)              # after the adjustment, before any line
            poses_prev, cams_prev = ZG.frame_poses(rec, project), ZG.snapshot(rec, project)
            reg_prev = set(int(f) for f in rec.reg_frame_ids())
            if zcam_focus_line and zfree and zkeys:
                from .zcam import focus_line_lines, refit_focus_lines
                line_snap = ZG.snapshot(rec, project)
                lines = refit_focus_lines(rec, project)
                if undone:                                       # a group whose lines were undone stays off them
                    ZG.restore(rec, line_snap, ZG.camera_ids_of(project, undone))
                ZG.trace(rec, project, f"r{k + 1}_lines")
                log.append({"stage": 2, "round": k + 1, "zcam_focus_lines": lines})
                if verbose:
                    logger.info(f"stage 2 round {k + 1}: Mastcam-Z focus lines (bins set on them):")
                    for ln in focus_line_lines(lines):
                        logger.info(ln)
            n_bad = filter_observations(rec, project, float(rmax)) + drop_short_tracks(rec, min_track_length)
            st = track_statistics(rec, project)
            res = native_residuals(rec, project)["residual_native_px"]
            entry = {"round": k + 1, "stage": 2, "triangulation_px_full": tpx, "cauchy_scale_sigma": loss, "ba": ba["brief"],
                     "filtered_observations": n_bad, "tracks": st, "zcam_coverage": ZG.group_coverage(rec, project),
                     "hazcam_coverage": hazcam_coverage(rec, project) if h_frames else None,
                     "zcam_terrain": _zcam_terrain(rec),
                     "rms_native_px": float(np.sqrt(np.mean(res ** 2))) if res.size else float("nan"),
                     "median_native_px": float(np.median(res)) if res.size else float("nan"),
                     "elapsed_s": round(time.time() - t0, 1)}
            log.append(entry)
            if verbose:
                logger.info(f"stage 2 round {k + 1}: {st['points']} points, median {entry['median_native_px']:.3f} / "
                      f"rms {entry['rms_native_px']:.3f} native px, {n_bad} obs filtered, {entry['elapsed_s']:.0f} s"
                      + (f"; Mastcam-Z frames with tie points {ZG.brief(entry['zcam_coverage'])}" if z_frames else ""))
                if z_frames:
                    logger.info(f"stage 2 round {k + 1}: {_zcam_terrain_line(entry['zcam_terrain'])}")
                if h_frames:
                    logger.info(f"stage 2 round {k + 1}: {hazcam_line(entry['hazcam_coverage'])}")
            ZG.trace(rec, project, f"r{k + 1}_filtered", {"filtered": n_bad})
            if exclude_outliers and k + 1 == min(int(exclude_after_round), len(schedule)):
                _exclude(f"stage 2, after round {k + 1}")
                ZG.trace(rec, project, f"r{k + 1}_excluded")
        if pose_guided and zkeys and Path(topts2.get("database") or project.database).is_file():
            # matches guided by the stage-2 poses (Mastcam-Z against Navcam and the other zooms), then one more
            # triangulation from them and one adjustment at the last round's thresholds
            from .guided import _family_ties, pose_guided_matching
            from .health import beyond_ground
            before = _family_ties(rec, project)
            zcov0 = ZG.group_coverage(rec, project)                # v0p85
            zposes0, zcams0 = ZG.frame_poses(rec, project), ZG.snapshot(rec, project)
            zreg0 = set(int(f) for f in rec.reg_frame_ids())
            bg0 = beyond_ground(rec, sample=50000)["fraction"] or 0.0
            rec_before, topts_before = copy.deepcopy(rec), topts2
            dbg, prep = pose_guided_matching(rec, project, topts2.get("database") or project.database, verbose=verbose,
                                             **(pose_guided if isinstance(pose_guided, dict) else {}))
            topts2 = dict(topts, database=dbg)
            tpx, loss, rmax = schedule[-1]
            rec = triangulate(rec, project, max_reproj_px=float(tpx), min_angle_deg=min_tri_angle_deg, **topts2)
            ZG.trace(rec, project, "guided_tri", {"database": str(dbg)})
            zcov_t = ZG.group_coverage(rec, project)              # right after the guided triangulation
            zbad_t = ZG.collapsed(zcov0, zcov_t)
            if zbad_t:
                prep["zcam_diagnosis"] = ZG.diagnose(rec, project, zbad_t, zposes0, zcams0, zreg0, dbg)
                prep["zcam_diagnosis"]["database_before"] = ZG.diagnose(
                    rec, project, zbad_t, zposes0, zcams0, zreg0, topts_before.get("database")).get(
                    "db_pairs_per_image_median")
                logger.info(f"stage 2, pose-guided: Mastcam-Z frames lost at the triangulation - "
                      f"{ZG.diagnosis_line(prep['zcam_diagnosis'])} (database before: pairs per image median "
                      f"{prep['zcam_diagnosis']['database_before']})")
            ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(loss), refine_intrinsics=refine_intrinsics,
                               refine_tangential=refine_tangential, refine_rig=False, max_iterations=max_iterations,
                               attitude_prior_deg=attitude_prior_deg,
                               hold_cameras=sorted(hold | zkeys) if zcam_focus_line else sorted(hold),
                               linear_solver=linear_solver, refine_zcam_rig=zfit if zcam_focus_line else None)
            n_bad = filter_observations(rec, project, float(rmax)) + drop_short_tracks(rec, min_track_length)
            prep.update({"before": before, "after": _family_ties(rec, project), "filtered": n_bad, "ba": ba["brief"]})
            # false matches that agree with the epipolar geometry land far beyond the ground - if the guided
            # matches raise that share by more than POSE_GUIDED_MAX_BEYOND_GROUND_RISE, the stage is undone
            bg1 = beyond_ground(rec, sample=50000)["fraction"] or 0.0
            prep.update({"beyond_ground_before": bg0, "beyond_ground_after": bg1,
                         "zcam_terrain_after": _zcam_terrain(rec)})
            if verbose:
                logger.info(f"stage 2, pose-guided: {_zcam_terrain_line(prep['zcam_terrain_after'])}")
            ZG.trace(rec, project, "guided_ba")
            zcov1 = ZG.group_coverage(rec, project)
            zbad = ZG.collapsed(zcov0, zcov1)
            prep.update({"zcam_coverage_before": zcov0, "zcam_coverage_after": zcov1})
            if zbad:                                             # e.g. Rochette 18 Oct, ZR034 96 -> 0 frames
                rec, topts2 = rec_before, topts_before
                prep["reverted"] = (f"{', '.join(zbad)} lost its frames with tie points ({ZG.brief(zcov0)} -> "
                                    f"{ZG.brief(zcov1)})")
                if verbose:
                    logger.info(f"stage 2, pose-guided matches undone: {prep['reverted']}")
            elif bg1 - bg0 > POSE_GUIDED_MAX_BEYOND_GROUND_RISE:
                rec, topts2 = rec_before, topts_before
                prep["reverted"] = (f"beyond-ground share {100 * bg0:.1f} -> {100 * bg1:.1f} % "
                                    f"(> +{100 * POSE_GUIDED_MAX_BEYOND_GROUND_RISE:g} points)")
                if verbose:
                    logger.info(f"stage 2, pose-guided matches undone: {prep['reverted']}")
            del rec_before
            log.append({"stage": 2, "pose_guided": prep})
            project.settings["pose_guided"] = prep
            if verbose:
                logger.info(f"stage 2, pose-guided: Navcam-Mastcam-Z tie points {before['navcam_zcam_points']} -> "
                      f"{prep['after']['navcam_zcam_points']}, points {before['points']} -> {prep['after']['points']}")
        if isolated_points_factor and isolated_points_factor > 0:     # 0.94.4: the scattered false points
            iso = drop_isolated_points(rec, float(isolated_points_factor))
            iso["zcam_terrain_after"] = _zcam_terrain(rec) if z_frames else None
            log.append({"stage": 2, "isolated_points": iso})
            project.settings["isolated_points"] = {k: v for k, v in iso.items() if k != "zcam_terrain_after"}
            if verbose:
                logger.info(f"stage 2: {iso['deleted']} isolated two-view points deleted of {iso['points']} (more than "
                            f"{iso['factor']:g} x the median spacing to their {iso['k']} nearest points at their range, within "
                            f"{iso.get('within_m', float('nan')):.1f} m of the cameras' centre; "
                            f"{iso.get('deleted_beyond_40m', 0)} of {iso.get('points_beyond_40m', 0)} beyond 40 m)"
                            + (f"; {_zcam_terrain_line(iso['zcam_terrain_after'])}" if iso.get("zcam_terrain_after")
                               else ""))
        ZG.trace(rec, project, "before_final")
        if exclude_outliers:
            _exclude("stage 2, before the final adjustment", navcam_ties=True)
        else:                                  # the Navcam-tie minimum holds even without the outlier tests
            weak = [f for f in find_outlier_frames(rec, project, residual_factor=1e9, min_residual_px=1e9,
                                                   shift_mad_factor=1e9, min_shift_m=1e9, attitude_mad_factor=1e9,
                                                   min_attitude_deg=1e9, min_observations=0)
                    if f.get("navcam_ties") is not None and f["kind"] == "outlier"]
            if weak:
                exclude_frames(rec, [f["frame_id"] for f in weak])
                for f in weak:
                    f["when"] = "stage 2, before the final adjustment (Navcam ties)"
                excluded.extend(weak)
                log.append({"excluded_frames": weak, "when": "stage 2, Navcam ties"})
        if h_frames:
            weak_h = set(hazcam_weak_eyes(hazcam_coverage(rec, project))) & hazkeys
            reset_cameras_to_start(rec, project, weak_h - hazcam_weak_held)     # 0.93.1: held at the start
            hazcam_weak_held.update(weak_h)
            hold |= weak_h
            if hazcam_weak_held:
                # a pair whose two eyes are weak keeps its rig at the start too
                rh = sorted({e[0] for e in hazcam_weak_held if {e[0] + "L", e[0] + "R"} & hazkeys
                             <= hazcam_weak_held})
                if project.settings.get("hazcam") is not None:
                    project.settings["hazcam"]["rig_hold"] = rh
                log.append({"stage": 2, "hazcam_held_weak": sorted(hazcam_weak_held), "hazcam_rig_held": rh,
                            "min_observations": HAZCAM_MIN_REFINE_OBSERVATIONS})
                if verbose:
                    logger.info(f"stage 2: Hazcam eyes held at their start (fewer than "
                                f"{HAZCAM_MIN_REFINE_OBSERVATIONS} tie-point observations): {sorted(hazcam_weak_held)}")
        ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(schedule[-1][1]),
                           refine_intrinsics=refine_intrinsics, refine_tangential=refine_tangential, refine_rig=False,
                           max_iterations=2 * max_iterations, attitude_prior_deg=attitude_prior_deg,
                           hold_cameras=sorted(hold | zkeys) if zcam_focus_line else sorted(hold), linear_solver=linear_solver,
                           refine_zcam_rig=zfit if zcam_focus_line else None)
        if lines is not None:
            project.settings["zcam_focus_lines"] = lines
        n_bad = filter_observations(rec, project, float(schedule[-1][2])) + drop_short_tracks(rec, min_track_length)
        log.append({"final_ba": ba["brief"], "frames_held": ba["frames_held"], "filtered_after_final": n_bad,
                    "tracks": track_statistics(rec, project), "prior_gauge": ba.get("prior_gauge")})
    backlash = None
    if zcam_backlash and any(str(r.get("instrument", "")).startswith("Z") for r in project.images):
        from .backlash import backlash_stage
        rec, backlash = backlash_stage(rec, project, mode=zcam_backlash, sigma_px=sigma_px,
                                       loss_scale=float(schedule[-1][1]), max_iterations=2 * max_iterations,
                                       attitude_prior_deg=attitude_prior_deg, linear_solver=linear_solver,
                                       refine_tangential=refine_tangential,
                                       hold_f_images=int(project.settings.get("zcam_hold_f_images") or 0),
                                       z_min=zcam_backlash_z, verbose=verbose,
                                       hold_cameras=sorted(k for k in hold if not str(k).startswith("Z")))
        if backlash.get("cameras"):
            n_bad = filter_observations(rec, project, float(schedule[-1][2])) + drop_short_tracks(rec, min_track_length)
            backlash["filtered_after"] = n_bad
            if project.settings.get("zcam_line_params") and zcam_focus_line:
                # the new regular-state cameras moved freely in the backlash stage - the lines of every state
                # are fitted again, all bins set on them, and one adjustment with the Mastcam-Z cameras held
                from .zcam import focus_line_lines, refit_focus_lines
                lines = refit_focus_lines(rec, project)
                project.settings["zcam_focus_lines"] = lines
                ba = bundle_adjust(rec, project, sigma_px=sigma_px, loss_scale=float(schedule[-1][1]),
                                   refine_intrinsics=refine_intrinsics, refine_tangential=refine_tangential,
                                   refine_rig=False, max_iterations=2 * max_iterations,
                                   attitude_prior_deg=attitude_prior_deg,
                                   hold_cameras=sorted(hold | {k for k in project.cameras if str(k)[:1] in ("N", "Z")}),
                                   linear_solver=linear_solver,
                                   refine_zcam_rig=project.settings.get("zcam_rig_fit")
                                   if project.settings.get("zcam_rig_fit") in ("rotation", "full") else None)
                backlash["lines_after_split"] = {"ba": ba["brief"]}
                if verbose:
                    logger.info("Mastcam-Z focus lines after the backlash split (bins set on them):")
                    for ln in focus_line_lines(lines):
                        logger.info(ln)
        found = (project.settings.get("zcam_low_backlash") or {}).get("found_file")
        if found and backlash.get("new_regular_groups"):
            # flag the newly classified low-backlash observations for the next runs (the found list)
            from .backlash import append_found
            n_new = append_found(found, backlash["new_regular_groups"], site=project.root.parent.name)
            backlash["found_added"] = {"file": found, "lines": n_new}
            if verbose:
                logger.info(f"{n_new} low-backlash observations added to {found}")
        log.append({"zcam_backlash": {k: v for k, v in backlash.items() if k not in ("groups", "fits_before", "fits_after")}})
        project.settings["zcam_backlash"] = {k: v for k, v in backlash.items() if k not in ("fits_before", "fits_after")}
        project.settings["zcam_backlash"]["groups"] = [{k: v for k, v in g.items() if k != "names"}
                                                       for g in backlash["groups"]]
    thermal = navcam_stage.get("thermal") if navcam_stage else None
    if thermal:
        project.settings["thermal"] = thermal_early
    elif thermal_bins_deg and temperatures:
        from .thermal import thermal_stage
        single = project.root / "sparse" / f"{out_name}_single"
        single.mkdir(parents=True, exist_ok=True)
        rec.write(str(single))
        hold_nav = network["navcam_intrinsics"] == "hold" or any(str(k).startswith("N") for k in hold_cameras)
        rec, thermal = thermal_stage(rec, project, temperatures, bin_deg=float(thermal_bins_deg),
                                     min_images=int(thermal_min_images), free=tuple(thermal_free), hold=hold_nav,
                                     thermal_model=thermal_model, sigma_px=sigma_px,
                                     loss_scale=float(schedule[-1][1]), max_iterations=2 * max_iterations,
                                     attitude_prior_deg=attitude_prior_deg, linear_solver=linear_solver,
                                     verbose=verbose, fine_range=_fine_range, fine_deg=_fine_deg)
        n_bad = filter_observations(rec, project, float(schedule[-1][2])) + drop_short_tracks(rec, min_track_length)
        thermal["filtered_after"] = n_bad
        thermal["single_camera_model"] = str(single.relative_to(project.root))
        log.append({"thermal": {k: v for k, v in thermal.items() if k != "rows"}})
        project.settings["thermal"] = thermal
    elif thermal_bins_deg and verbose:
        logger.info("thermal stage skipped: no image temperatures")
    hz_cov = hazcam_coverage(rec, project) if hazkeys else None
    if hz_cov:
        log.append({"hazcam": hz_cov})
        if verbose:
            logger.info(f"final: {hazcam_line(hz_cov)}")
    out = project.root / "sparse" / out_name
    out.mkdir(parents=True, exist_ok=True)
    rec.write(str(out))
    if backlash:                                          # per image and focus group, for notebooks 04 / 05
        import json as _json
        (project.root / "zcam_focus_states.json").write_text(_json.dumps(_jsonable(backlash), indent=1), encoding="utf-8")
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
                                          "refine_tangential": tangential_setting(refine_tangential),
                                          "attitude_prior_deg": float(attitude_prior_deg or 0.0),
                                          "navcam_aspect": navcam_aspect,
                                          "schedule": [list(map(float, r)) for r in schedule],
                                          "exclude_outliers": bool(exclude_outliers),
                                          "min_frame_observations": int(MIN_FRAME_OBSERVATIONS),
                                          "excluded": [{k: v for k, v in e.items()} for e in excluded],
                                          "rig_translation_sigma_m": rig_translation_sigma_m,
                                          "round1_hold": bool(round1_hold),                  # v0p80
                                          "rig_fit": rig_fit,                                # v0p80
                                          "triangulation_options": topts,
                                          "navcam_network": network, "navcam_intrinsics": network["navcam_intrinsics"],
                                          "hold_cameras": sorted(hold), "staged": bool(staged),
                                          "navcam_stage": navcam_stage,
                                          "thermal_bins_deg": float(thermal_bins_deg) if thermal and thermal_bins_deg else None,
                                          "zcam_backlash": zcam_backlash if backlash else None,
                                          "localize_min_images": int(localize_min_images or 0),
                                          "unlocalized_stations": sorted(st for st, v in final_unloc.items()
                                                                         if v["prior"] == "dropped"),
                                          "unlocalized": final_unloc, "unlocalized_registration": unloc_reg,
                                          "hazcam": hz_cov}
    project.save()
    if gui_native:
        try:
            from .export import write_gui_native
            g = write_gui_native(project, rec)               # native-pixel viewing copy
            if verbose:
                logger.info(f"COLMAP GUI copy in native pixels: {g['dir']} ({g['images']} images, "
                      f"{g['points']} points)")
        except Exception as e:                               # noqa: BLE001  (a viewing aid; never stop the run)
            logger.info(f"gui_native not written: {type(e).__name__}: {e}")
    if project.settings.get("gui_files"):
        try:
            from .database import write_gui_project
            write_gui_project(project, model=out_name)      # colmap_gui.ini, open_in_colmap*.bat
        except OSError:
            pass
    return rec

@contextlib.contextmanager
def frame_observation_floor(value: Optional[int]):
    """Set the run-wide floor on tie-point observations per frame (``MIN_FRAME_OBSERVATIONS``) for the duration of
    the block: every bundle adjustment of the run - the staged, thermal and focus-state passes included - holds
    frames with fewer observations at their prior, and the outlier test flags them.  ``None`` leaves the default."""
    global MIN_FRAME_OBSERVATIONS
    old = MIN_FRAME_OBSERVATIONS
    if value is not None:
        MIN_FRAME_OBSERVATIONS = int(value)
    try:
        yield MIN_FRAME_OBSERVATIONS
    finally:
        MIN_FRAME_OBSERVATIONS = old


def reconstruct(project: SfmProject, *args, min_frame_observations: Optional[int] = None, **kwargs):
    """The staged reconstruction of a project (see :func:`_reconstruct` for every argument).
    ``min_frame_observations``: the floor on tie-point observations per frame for the whole run (None: 20); recorded
    in ``project.settings["reconstruction"]["min_frame_observations"]``."""
    with frame_observation_floor(min_frame_observations):
        return _reconstruct(project, *args, **kwargs)


def _reconstruct_signature():
    import inspect
    sig = inspect.signature(_reconstruct)
    return sig.replace(parameters=list(sig.parameters.values()) + [
        inspect.Parameter("min_frame_observations", inspect.Parameter.KEYWORD_ONLY, default=None,
                          annotation=Optional[int])])


reconstruct.__signature__ = _reconstruct_signature()     # _reconstruct's parameters plus the floor
