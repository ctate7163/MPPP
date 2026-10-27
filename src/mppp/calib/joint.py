"""
The joint adjustment of many solved Navcam blocks with one camera per eye and one rig: what :func:`mppp.calib.fit.
fit_navcam` runs.  Each block (``sparse/cahv_ba`` + ``project.json``) is loaded with its camera temperatures,
merged into one reconstruction (:func:`merge_scapes`) and adjusted (:func:`joint_adjust`) with the form's
per-image terms applied through keypoint maps: fx, fy scaled by the thermal slopes, the principal points moved by
theirs, k1 by its slope, the rig turned by its drift at the image's sol, the early-mission offsets before a sol.
Everything is in full-frame Navcam pixels (5120 x 3840).
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import forms
from .constrain import drift_offset_mdeg

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]
FISHEYE_T = "THIN_PRISM_FISHEYE"
BA_DEFAULTS = dict(sigma_px=0.5, loss_scale=2.0, attitude_prior_deg=5.0, refine_tangential=True)


# ================================================================ loading
@dataclass
class Scape:
    name: str
    root: Path
    project: Any                                    # mppp.sfm.project.SfmProject
    rec: Any                                        # pycolmap.Reconstruction
    temps: Dict[str, Dict[str, Any]] = field(default_factory=dict)      # image name -> {"T", "T_NL", "T_NR"}

    def image_T(self) -> Dict[int, float]:
        return {int(iid): float(self.temps[im.name]["T"]) for iid, im in self.rec.images.items()
                if im.name in self.temps}


def align_project_to_model(project, rec) -> int:
    """Make the project describe exactly the model's images (by name): image, camera and frame ids from the
    model; model images the project does not know are deregistered.  Returns the number of such frames."""
    from ..sfm.reconstruction import exclude_frames
    by_name = {r["name"]: r for r in project.images}
    missing = {im.frame_id for im in rec.images.values() if im.name not in by_name}
    if missing:
        exclude_frames(rec, sorted(missing))
    keep, cams = [], {}
    for iid, im in rec.images.items():
        r = by_name.get(im.name)
        if r is None:
            continue
        r["image_id"], r["camera_id"], r["frame_id"] = int(iid), int(im.camera_id), int(im.frame_id)
        cams.setdefault(r["instrument"], int(im.camera_id))
        keep.append(r)
    project.images = keep
    project.settings.setdefault("database", {})["cameras"] = dict(cams)
    return len(missing)


def load_scape(name: str, root: PathLike, samples: Optional[Dict[str, Dict[str, Any]]] = None,
               model: str = "cahv_ba", single: bool = True) -> Scape:
    """A solved block with its camera temperatures (project records, else label ``samples``).  A block solved with
    thermal bins is loaded from its one-camera-per-eye model ``sparse/<model>_single`` (the solution before the
    thermal stage) with the bins stripped from the project, so that every block enters the joint the same way.
    ``single`` False: the final model as it is (the Mastcam-Z joint: the Mastcam-Z frames are added after the
    thermal stage; the Navcam bins are mapped to their eye by :func:`merge_scapes`)."""
    import pycolmap
    from ..sfm.project import SfmProject
    from ..sfm.thermal import image_temperatures, strip_thermal_bins
    root = Path(root)
    if (root / "colmap").is_dir():
        root = root / "colmap"
    project = SfmProject.load(root)
    if single and any(c.get("thermal_bin") for c in project.cameras.values()) and (root / "sparse" / f"{model}_single").is_dir():
        strip_thermal_bins(project)
        model = f"{model}_single"
    rec = pycolmap.Reconstruction(str(root / "sparse" / model))
    align_project_to_model(project, rec)
    temps = image_temperatures(project, samples=samples)
    return Scape(name, root, project, rec, temps)


# ================================================================ rig geometry
def stereo_rig(rec) -> Tuple[Optional[int], Optional[int], Optional[Any]]:
    """(rig_id, right camera_id, sensor_from_rig) of the rig with a second sensor (the Navcam pair)."""
    for rid, rig in rec.rigs.items():
        for sid in rig.non_ref_sensors:
            return int(rid), int(sid.id), rig.sensor_from_rig(sid)
    return None, None, None


def rig_angles(R: np.ndarray, R_ref: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Right-from-left rotation as yaw (about the left camera's y axis - it shifts disparity), pitch (about x -
    vertical parallax) and roll (about z), mdeg; relative to ``R_ref`` if given."""
    from scipy.spatial.transform import Rotation
    M = R if R_ref is None else R @ np.asarray(R_ref).T
    rv = Rotation.from_matrix(M).as_rotvec()
    return {"yaw_mdeg": float(np.degrees(rv[1]) * 1e3), "pitch_mdeg": float(np.degrees(rv[0]) * 1e3),
            "roll_mdeg": float(np.degrees(rv[2]) * 1e3)}


def rotation_mdeg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """The angle (mdeg) between two rotations."""
    from scipy.spatial.transform import Rotation
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(np.asarray(Rb) @ np.asarray(Ra).T).as_rotvec())) * 1e3)


def stereo_offset(left, right, sensor_from_rig) -> Dict[str, float]:
    """Disparity at infinity (x_left - x_right, px) and vertical parallax (y_left - y_right) of the ray through the
    left principal point: what the rig rotation and the two principal points together do to stereo."""
    R = np.asarray(sensor_from_rig.rotation.matrix())
    pl = np.asarray(left.params[2:4], float)
    d = np.array(list(left.cam_from_img(pl)) + [1.0])
    x = R @ d
    pr = np.asarray(right.img_from_cam(x), float).ravel()
    return {"disparity_inf_px": float(pl[0] - pr[0]), "vparallax_inf_px": float(pl[1] - pr[1])}


# ================================================================ merging
MIN_TRI_ANGLE_DEG = 0.25        # 0.94.4 (Bunsen Peak, 24 Oct): below it (0.1 % of points, after the BA) 1.7 px median
MAX_TRI_ANGLE_DEG = 60.0        # above it (0.1 %) 1.3 px median - near-field mismatches
WIDE_ANGLE_DEG = 20.0           # 20-60 deg (5 %): 0.54 px median, twice the rest - drawn at half the weight
WIDE_WEIGHT = 0.5
MAX_STATION_WEIGHT = 4
OVERSAMPLE = 5                  # candidates drawn per point kept (the geometry is computed for these only)
_STATION = __import__("re").compile(r"_N(\d{7})")


def point_geometry(rec, pids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """(largest triangulation angle in deg, number of stations seen from) of the points ``pids``; a station is the
    product name's site and drive (``_N0260630``), else the frame."""
    C: Dict[int, np.ndarray] = {}
    st: Dict[int, Any] = {}
    ang = np.zeros(len(pids))
    nst = np.zeros(len(pids), int)
    for k, pid in enumerate(pids):
        pt = rec.points3D[pid]
        ids = [int(e.image_id) for e in pt.track.elements]
        for i in ids:
            if i not in C:
                im = rec.images[i]
                C[i] = np.asarray(im.projection_center(), float)
                m = _STATION.search(im.name)
                st[i] = m.group(1) if m else ("frame", int(im.frame_id))
        v = np.array([pt.xyz - C[i] for i in ids], float)
        v /= np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
        ang[k] = float(np.degrees(np.arccos(np.clip(float((v @ v.T).min()), -1.0, 1.0))))
        nst[k] = len({st[i] for i in ids})
    return ang, nst


def _subsample_points(rec, n: Optional[int], seed: int = 0, min_track: int = 2,
                      min_angle: float = MIN_TRI_ANGLE_DEG, max_angle: float = MAX_TRI_ANGLE_DEG) -> List[int]:
    """Point ids to keep: all if ``n`` is None, else ``n`` drawn without replacement.  0.94.4 (Christian 24 Oct):
    the points are drawn by the number of stations they are seen from (capped at 4; they carry the intrinsics -
    before, by track length), a point under ``min_angle`` (drifted below the blocks' cut in their adjustment) or over
    ``max_angle`` never, one of 20-60 deg at half the weight.  The geometry is computed for ``OVERSAMPLE`` x ``n``
    candidates only."""
    ids = [pid for pid, pt in rec.points3D.items() if pt.track.length() >= min_track]
    rng = np.random.default_rng(seed)
    if n is not None and len(ids) > OVERSAMPLE * n:
        ids = [ids[i] for i in sorted(rng.choice(len(ids), size=OVERSAMPLE * n, replace=False))]
    if not ids:
        return []
    ang, nst = point_geometry(rec, ids)
    ok = (ang >= float(min_angle)) & (ang <= float(max_angle))
    ids = [p for p, o in zip(ids, ok) if o]
    ang, nst = ang[ok], nst[ok]
    if n is None or len(ids) <= n:
        return sorted(ids)
    w = np.minimum(nst, MAX_STATION_WEIGHT).astype(float) * np.where(ang > WIDE_ANGLE_DEG, WIDE_WEIGHT, 1.0)
    pick = rng.choice(len(ids), size=n, replace=False, p=w / w.sum())
    return [ids[i] for i in sorted(pick)]


def thin_points(rec, n: Optional[int], seed: int = 0) -> int:
    """Keep at most ``n`` 3-D points of ``rec`` (in place; long tracks preferred).  Memory of an adjustment with
    covariances grows by ~4 kB per observation.  Returns the number of points removed."""
    if n is None or len(rec.points3D) <= n:
        return 0
    keep = set(_subsample_points(rec, n, seed=seed, min_track=2))
    drop = [pid for pid in rec.points3D if pid not in keep]
    for pid in drop:
        rec.delete_point3D(pid)
    return len(drop)


def progress(done: int, total: int, t0: float, label: str = "") -> str:
    """``[label done/total, elapsed, ~left, at HH:MM]`` for the long loops."""
    import datetime as _dt
    import time as _t
    el = _t.time() - t0
    left = el / max(done, 1) * max(total - done, 0)
    return (f"[{label + ' ' if label else ''}{done}/{total}, {el / 60:.0f} min elapsed, ~{left / 60:.0f} min left, "
            f"at {_dt.datetime.now():%H:%M}]")


def merge_scapes(scapes, cameras: Dict[str, Any], rig: Tuple[np.ndarray, np.ndarray],
                 points_per_scape: Any = 25000, seed: int = 0, exclude: Sequence[str] = (),
                 include_zcam: bool = False, zcam_points_per_scape: Optional[int] = 8000) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    One reconstruction holding every block: camera 1 = NL, camera 2 = NR (``cameras``: {"NL", "NR"} ->
    pycolmap.Camera, shared by all blocks), rig 1 = the stereo pair (``rig`` = (R, t) of NR from NL), rig 2 / 3 =
    NR / NL alone; each block's frames, images, poses and (subsampled) points with fresh ids.  The merged project
    lists every image with its own block's priors (each block keeps its own world frame).  Returns
    (rec, project, index) with ``index`` = {"scape_of_image": {image_id: scape}, "images": {scape: [image_id]},
    "frames": {scape: [frame_id]}, "temps", "duplicates"}.  An image that is in more than one block enters once per
    block, renamed ``<name>@<block>`` after its first block, with the same temperature.

    ``include_zcam`` (the Mastcam-Z joint, 20 Oct): each block's registered Mastcam-Z images enter too, one camera
    per block focus-bin camera (ids after the Navcam ones; ``index["zcams"]`` {camera_id: {"scape", "key", "spec"}}),
    one rig and frame per image, their project rows with ``sclk_key`` made unique per block (``<sclk>@<block>``: the
    Mastcam-Z rig pairs a right image with the left one of the same exposure); the points keep their Mastcam-Z
    observations, and every Navcam-Mastcam-Z point with a third look is kept besides ``points_per_scape``, plus
    up to ``zcam_points_per_scape`` points of Mastcam-Z images only.
    """
    import pycolmap
    from ..sfm.thermal import eye_key
    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))      # noqa: E731
    new = pycolmap.Reconstruction()
    for cid, key in ((1, "NL"), (2, "NR")):
        c = cameras[key]
        new.add_camera(pycolmap.Camera(camera_id=cid, model=c.model, width=c.width, height=c.height,
                                       params=np.array(c.params, float)))
    r1 = pycolmap.Rig(rig_id=1)
    r1.add_ref_sensor(sensor(1))
    r1.add_sensor(sensor(2), pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(rig[0], float)), np.asarray(rig[1], float)))
    new.add_rig(r1)
    r2 = pycolmap.Rig(rig_id=2)
    r2.add_ref_sensor(sensor(2))
    new.add_rig(r2)
    r3 = pycolmap.Rig(rig_id=3)
    r3.add_ref_sensor(sensor(1))
    new.add_rig(r3)
    proj = None
    seen_names = set()
    idx: Dict[str, Any] = {"scape_of_image": {}, "images": {}, "frames": {}, "points": {}, "temps": {},
                           "duplicates": {}, "zcams": {}, "zimages": {}}
    next_img, next_frame = 1, 1
    next_cam, next_rig = 3, 4
    for sc in scapes:
        if sc.name in exclude:
            continue
        if proj is None:
            proj = copy.deepcopy(sc.project)
            proj.images = []
            proj.cameras = {k: {"model": cameras[k].model.name, "params": list(map(float, cameras[k].params)),
                                "width": cameras[k].width, "height": cameras[k].height, "free_params": []}
                            for k in ("NL", "NR")}
            proj.settings = dict(proj.settings)
            proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
            proj.settings["navcam_rig_yaw"] = "refine"
            proj.settings["navcam_rig_roll"] = "refine"           # 0.94.4: the blocks hold it; a joint refines it
        idx["temps"].update({n: float(v["T"]) for n, v in sc.temps.items()})
        rec = sc.rec
        # a rig-epoch copy of an eye ("NL#1") is that eye here (the joint has one camera per eye)
        key_of = {int(v): eye_key(k) for k, v in sc.project.settings.get("database", {}).get("cameras", {}).items()}
        by_name = {r["name"]: r for r in sc.project.images}
        reg = set(rec.reg_image_ids())
        img_map: Dict[int, int] = {}
        idx["images"][sc.name], idx["frames"][sc.name] = [], []
        for fid, fr in rec.frames.items():
            if not fr.has_pose:
                continue
            # Navcam images only - a Navcam + Mastcam-Z block also has Mastcam-Z frames
            ims = [d.id for d in fr.data_ids if d.id in reg and rec.images[d.id].name in by_name
                   and key_of.get(int(rec.images[d.id].camera_id)) in ("NL", "NR")]
            if not ims:
                continue
            eyes = sorted({key_of.get(int(rec.images[i].camera_id), "") for i in ims})
            rid = 1 if eyes == ["NL", "NR"] else (2 if eyes == ["NR"] else 3)
            nf = pycolmap.Frame(frame_id=next_frame, rig_id=rid)
            pending = []
            for i in ims:
                im = rec.images[i]
                name = im.name
                if name in seen_names:
                    name = f"{im.name}@{sc.name}"
                    idx["duplicates"][sc.name] = idx["duplicates"].get(sc.name, 0) + 1
                    if im.name in idx["temps"]:
                        idx["temps"][name] = idx["temps"][im.name]
                seen_names.add(name)
                cid = 1 if key_of.get(int(im.camera_id)) == "NL" else 2
                img_map[i] = next_img
                nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cid), id=next_img))
                kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
                ni = pycolmap.Image(name=name, keypoints=kps, camera_id=cid, image_id=next_img)
                ni.frame_id = next_frame
                pending.append(ni)
                r = dict(by_name[im.name])
                r.update({"name": name, "image_id": next_img, "camera_id": cid, "frame_id": next_frame,
                          "scape": sc.name})
                if name != im.name:
                    r["source_name"] = im.name
                proj.images.append(r)
                idx["scape_of_image"][next_img] = sc.name
                idx["images"][sc.name].append(next_img)
                next_img += 1
            ref = [i for i in ims if key_of.get(int(rec.images[i].camera_id)) == ("NL" if rid in (1, 3) else "NR")][0]
            nf.rig_from_world = rec.images[ref].cam_from_world()
            new.add_frame(nf)
            for ni in pending:
                new.add_image(ni)
            idx["frames"][sc.name].append(next_frame)
            next_frame += 1
        if include_zcam:
            zcam_of: Dict[str, int] = {}
            idx["zimages"][sc.name] = []
            # 0.95.0 (Christian, 27 Oct): each bin's observations in its block (before the joint's point sampling)
            zobs: Dict[int, int] = {}
            for pt in rec.points3D.values():
                for el in pt.track.elements:
                    c_ = int(rec.images[el.image_id].camera_id)
                    zobs[c_] = zobs.get(c_, 0) + 1
            for fid, fr in rec.frames.items():
                if not fr.has_pose:
                    continue
                for d in fr.data_ids:
                    i = d.id
                    if i not in reg or rec.images[i].name not in by_name:
                        continue
                    key = key_of.get(int(rec.images[i].camera_id), "")
                    if not str(key).startswith("Z"):
                        continue
                    im = rec.images[i]
                    if key not in zcam_of:
                        c = rec.cameras[im.camera_id]
                        new.add_camera(pycolmap.Camera(camera_id=next_cam, model=c.model, width=c.width, height=c.height,
                                                       params=np.array(c.params, float)))
                        zr = pycolmap.Rig(rig_id=next_rig)
                        zr.add_ref_sensor(sensor(next_cam))
                        new.add_rig(zr)
                        zcam_of[key] = next_cam
                        idx["zcams"][next_cam] = {"scape": sc.name, "key": key, "rig_id": next_rig,
                                                  "spec": copy.deepcopy(sc.project.cameras.get(key) or {}),
                                                  "observations": int(zobs.get(int(im.camera_id), 0))}
                        next_cam += 1
                        next_rig += 1
                    cid = zcam_of[key]
                    name = im.name if im.name not in seen_names else f"{im.name}@{sc.name}"
                    seen_names.add(name)
                    nf = pycolmap.Frame(frame_id=next_frame, rig_id=idx["zcams"][cid]["rig_id"])
                    nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cid), id=next_img))
                    nf.rig_from_world = im.cam_from_world()
                    ni = pycolmap.Image(name=name, keypoints=np.array([q.xy for q in im.points2D], float).reshape(-1, 2),
                                        camera_id=cid, image_id=next_img)
                    ni.frame_id = next_frame
                    new.add_frame(nf)
                    new.add_image(ni)
                    r = dict(by_name[im.name])
                    r.update({"name": name, "image_id": next_img, "camera_id": cid, "frame_id": next_frame,
                              "scape": sc.name, "sclk_key": f"{r.get('sclk_key')}@{sc.name}"})
                    if name != im.name:
                        r["source_name"] = im.name
                    proj.images.append(r)
                    img_map[i] = next_img
                    idx["scape_of_image"][next_img] = sc.name
                    idx["zimages"][sc.name].append(next_img)
                    idx["frames"][sc.name].append(next_frame)
                    next_img += 1
                    next_frame += 1
        for fid in idx["frames"][sc.name]:
            new.register_frame(fid)
        n_pts = points_per_scape.get(sc.name, points_per_scape.get("*")) if isinstance(points_per_scape, dict) \
            else points_per_scape                       # 0.93: {block: points} ("*": the rest)
        keep = _subsample_points(rec, n_pts, seed=seed)
        if include_zcam:
            fam = {int(i): rec.images[i].name[:1] for i in rec.reg_image_ids()}
            mixed, zonly = [], []
            for pid, pt in rec.points3D.items():
                fs = {fam.get(int(el.image_id)) for el in pt.track.elements}
                if "Z" in fs:
                    (mixed if ("N" in fs and pt.track.length() >= 3) else zonly).append(pid)
            rz = np.random.default_rng(seed + 1)
            if zcam_points_per_scape is not None and len(zonly) > zcam_points_per_scape:
                zonly = [zonly[k] for k in sorted(rz.choice(len(zonly), zcam_points_per_scape, replace=False))]
            keep = sorted(set(keep) | set(mixed) | set(zonly))
        n_pts = 0
        for pid in keep:
            pt = rec.points3D[pid]
            tr = pycolmap.Track()
            for el in pt.track.elements:
                if el.image_id in img_map:
                    tr.add_element(img_map[el.image_id], el.point2D_idx)
            if tr.length() >= 2:
                new.add_point3D(pt.xyz, tr, pt.color)
                n_pts += 1
        idx["points"][sc.name] = n_pts
    return new, proj, idx


def subset(rec, idx: Dict[str, Any], keep: Sequence[str]):
    """A copy of a merged reconstruction with only the blocks ``keep`` registered and their points."""
    out = copy.deepcopy(rec)
    keep = set(keep)
    s_of = idx["scape_of_image"]
    drop_frames = [f for sc, fs in idx["frames"].items() if sc not in keep for f in fs]
    drop_pts = [pid for pid, pt in out.points3D.items() if s_of.get(pt.track.elements[0].image_id) not in keep]
    for pid in drop_pts:
        out.delete_point3D(pid)
    reg = set(out.reg_frame_ids())
    for f in drop_frames:
        if f in reg:
            out.deregister_frame(f)
    return out


# ================================================================ the per-image terms as keypoint maps
def keypoint_scales(proj, temps: Dict[str, float], ppm_per_degC: float, T0: float,
                    axis_ppm: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[int, Any]:
    """{image_id: (sx, sy)} = 1 + ppm 1e-6 (T - T0) per eye and axis (``axis_ppm`` {"NL": (fx ppm, fy ppm), ...}),
    else one scale from ``ppm_per_degC``."""
    out: Dict[int, Any] = {}
    for r in proj.images:
        T = temps.get(r["name"])
        eye = str(r.get("instrument", ""))[:2]
        if axis_ppm and eye in axis_ppm:
            bx, by = (float(v) for v in axis_ppm[eye])
            d = (float(T) - float(T0)) if T is not None else 0.0
            v = np.array([1.0 + 1e-6 * bx * d, 1.0 + 1e-6 * by * d])
        else:
            v = 1.0 + (1e-6 * float(ppm_per_degC) * (float(T) - float(T0)) if T is not None else 0.0)
        if T is not None or np.any(np.asarray(v) != 1.0):
            out[int(r["image_id"])] = tuple(map(float, v)) if np.ndim(v) else float(v)
    return out


def shared_state(rec) -> Dict[str, Any]:
    """The shared cameras and rig of a merged reconstruction."""
    _, _, T = stereo_rig(rec)
    R = np.asarray(T.rotation.matrix())
    t = np.asarray(T.translation)
    return {"NL": {"model": rec.cameras[1].model.name, "params": list(map(float, rec.cameras[1].params))},
            "NR": {"model": rec.cameras[2].model.name, "params": list(map(float, rec.cameras[2].params))},
            "rig_R": R.tolist(), "rig_t": t.tolist(), "rig_angles_abs": rig_angles(R), "baseline_m": float(np.linalg.norm(t))}


def thermal_keypoint_map(proj, temps: Dict[str, float], T0: float,
                         pp_slopes: Optional[Dict[int, Tuple[float, float]]] = None,
                         drift: Optional[Dict[str, Any]] = None, right_camera_ids: Sequence[int] = (2,),
                         base_of: Optional[Dict[int, int]] = None) -> Callable:
    """
    The per-image corrections of a merged reconstruction as one keypoint map for ``bundle_adjust``:

    * ``pp_slopes`` {camera id: (a_x, a_y) px/degC}: a principal point that moves with the camera temperature,
      c(T) = c(T0) + a (T - T0) (the keypoints move by -a (T - T0), the same for any COLMAP model);
    * ``drift``: the rig drift (:func:`mppp.calib.constrain.drift_offset_mdeg`): the right camera turned by the
      offset at the image's sol (pitch, yaw, roll about x, y, z of the right camera, as the rig file applies it).

    ``base_of`` {camera id: base camera id}: camera copies (:func:`split_by_unit`) follow their base camera's terms.
    """
    from scipy.spatial.transform import Rotation
    T_of = {int(r["image_id"]): temps.get(r["name"]) for r in proj.images}
    sol_of = {int(r["image_id"]): r.get("sol") for r in proj.images}
    right = set(int(c) for c in right_camera_ids)
    pp = {int(k): np.asarray(v, float) for k, v in (pp_slopes or {}).items()}
    d_cache: Dict[Any, np.ndarray] = {}

    def drift_rv(sol):
        if drift is None or sol is None:
            return np.zeros(3)
        if sol not in d_cache:
            o = drift_offset_mdeg(drift, float(sol))
            d_cache[sol] = 1e-3 * math.radians(1.0) * np.array([o["pitch_mdeg"], o["yaw_mdeg"], o["roll_mdeg"]])
        return d_cache[sol]

    bo = {int(k): int(v) for k, v in (base_of or {}).items()}

    def fmap(iid, kps, cam):
        cid = bo.get(int(cam.camera_id), int(cam.camera_id))
        T = T_of.get(int(iid))
        dT = 0.0 if T is None else float(T) - float(T0)
        if cid in pp and dT:
            kps = kps - pp[cid] * dT
        if cid in right:
            rv = drift_rv(sol_of.get(int(iid)))
            if np.any(rv):
                Rm = Rotation.from_rotvec(-rv).as_matrix()
                rays = np.array(cam.cam_from_img(kps), float)
                X = np.c_[rays[:, :2], np.ones(len(rays))] @ Rm.T
                out = np.array(cam.img_from_cam(X, check_cheirality=False), float)
                kps = np.where(np.isfinite(out), out, kps)
        return kps
    return fmap


def k1_keypoint_map(proj, temps: Dict[str, float], T0: float, k1_slopes: Dict[str, float]) -> Callable:
    """k1 that changes with the camera temperature, k1(T) = k1(T0) + s (T - T0), as a keypoint map: a keypoint seen
    with the camera at T is moved to where the shared camera (k1 at T0) puts the same ray, x' = pi_T0(pi_T^-1(x)).
    ``k1_slopes`` {"NL": s, "NR": s} per degC (eye by the image row's instrument)."""
    import pycolmap
    T_of = {int(r["image_id"]): temps.get(r["name"]) for r in proj.images}
    eye_of = {int(r["image_id"]): str(r.get("instrument", ""))[:2] for r in proj.images}
    sl = {str(k): float(v or 0.0) for k, v in k1_slopes.items()}

    def fmap(iid, kps, cam):
        T = T_of.get(int(iid))
        s = sl.get(eye_of.get(int(iid), ""), 0.0)
        if T is None or not s or not len(kps):
            return kps
        names = forms.PARAM_NAMES.get(cam.model.name)
        if not names or "k1" not in names:
            return kps
        p = np.array(cam.params, float)
        p[names.index("k1")] += s * (float(T) - float(T0))
        camT = pycolmap.Camera(model=cam.model, width=cam.width, height=cam.height, params=p)
        rays = np.array(camT.cam_from_img(kps), float)
        out = np.array(cam.img_from_cam(np.c_[rays[:, :2], np.ones(len(rays))], check_cheirality=False), float)
        return np.where(np.isfinite(out), out, kps)
    return fmap


def early_keypoint_map(proj, before_sol: float, offsets) -> Any:
    """The keypoint map that makes the images taken before ``before_sol`` see a camera with fx + dfx, fy + dfy,
    cx + dcx, cy + dcy (``offsets``, px, the same for both eyes) while they keep the shared camera: a keypoint u
    becomes c + f (u - c - dc) / (f + df) per axis - exact for any COLMAP model."""
    o = np.asarray(offsets, float)
    early = {int(r["image_id"]) for r in proj.images
             if r.get("sol") is not None and float(r["sol"]) < float(before_sol) and "image_id" in r}

    def fmap(iid, kps, cam):
        if int(iid) not in early or not np.any(o):
            return kps
        p = np.asarray(cam.params, float)
        f, c = p[0:2], p[2:4]
        return c + f * (kps - c - o[2:4]) / (f + o[0:2])
    fmap.early_images = early                      # type: ignore[attr-defined]
    fmap.offsets = o                               # type: ignore[attr-defined]
    return fmap


def split_by_unit(rec, proj, unit_of_image: Dict[int, str], free: Sequence[str] = ("fx", "fy", "cx", "cy")):
    """
    A copy of a merged reconstruction in which every camera and rig gets one copy per *unit* (e.g. block and sol
    epoch; ``unit_of_image``: image id -> unit), the rigs with the same sensor poses; each unit camera refines only
    ``free`` (the distortion stays shared at its current value).  Returns (rec, proj, units) with ``units`` =
    {unit: {"cameras": {old camera id: new id}, "rigs": {old rig id: new id}}}; the project copy maps the keys
    ``<camera>@<unit>``.
    """
    import pycolmap
    p2 = copy.deepcopy(proj)
    db = p2.settings.setdefault("database", {}).setdefault("cameras", {})
    key_of = {int(v): k for k, v in db.items()}
    new = pycolmap.Reconstruction()
    next_cid = max(rec.cameras) + 1
    next_rid = max(rec.rigs) + 1
    units: Dict[str, Dict[str, Dict[int, int]]] = {}
    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))      # noqa: E731
    for cam in rec.cameras.values():
        new.add_camera(cam)
    for rig in rec.rigs.values():
        new.add_rig(rig)
    names = forms.PARAM_NAMES.get(next(iter(rec.cameras.values())).model.name, ())
    frame_unit = {}
    for fid, fr in rec.frames.items():
        us = {unit_of_image.get(int(d.id)) for d in fr.data_ids} - {None}
        frame_unit[fid] = sorted(us)[0] if us else None
    for fid, fr in rec.frames.items():
        u = frame_unit[fid]
        if u is None:
            nf = pycolmap.Frame(frame_id=fid, rig_id=fr.rig_id)
            for d in fr.data_ids:
                nf.add_data_id(d)
        else:
            m = units.setdefault(u, {"cameras": {}, "rigs": {}})
            for d in fr.data_ids:
                c = int(d.sensor_id.id)
                if c not in m["cameras"]:
                    base = rec.cameras[c]
                    new.add_camera(pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width,
                                                   height=base.height, params=np.array(base.params, float)))
                    k = key_of.get(c, str(c))
                    pc = copy.deepcopy(p2.cameras.get(k, {}))
                    pc.update({"fixed_params": [n for n in names if n not in set(free)], "free_params": [],
                               "unit": u, "base_camera": k})
                    p2.cameras[f"{k}@{u}"] = pc
                    db[f"{k}@{u}"] = next_cid
                    m["cameras"][c] = next_cid
                    next_cid += 1
            if fr.rig_id not in m["rigs"]:
                old = rec.rigs[fr.rig_id]
                r = pycolmap.Rig(rig_id=next_rid)
                ref = old.ref_sensor_id.id
                if ref not in m["cameras"]:
                    base = rec.cameras[ref]
                    new.add_camera(pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width,
                                                   height=base.height, params=np.array(base.params, float)))
                    m["cameras"][ref] = next_cid
                    next_cid += 1
                r.add_ref_sensor(sensor(m["cameras"][ref]))
                for sid in old.non_ref_sensors:
                    if sid.id not in m["cameras"]:
                        base = rec.cameras[sid.id]
                        new.add_camera(pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width,
                                                       height=base.height, params=np.array(base.params, float)))
                        m["cameras"][sid.id] = next_cid
                        next_cid += 1
                    r.add_sensor(sensor(m["cameras"][sid.id]), old.sensor_from_rig(sid))
                new.add_rig(r)
                m["rigs"][fr.rig_id] = next_rid
                next_rid += 1
            nf = pycolmap.Frame(frame_id=fid, rig_id=m["rigs"][fr.rig_id])
            for d in fr.data_ids:
                nf.add_data_id(pycolmap.data_t(sensor_id=sensor(m["cameras"][int(d.sensor_id.id)]), id=d.id))
        if fr.has_pose:
            nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
    for iid, im in rec.images.items():
        kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
        u = frame_unit.get(im.frame_id)
        cid = units[u]["cameras"][int(im.camera_id)] if u is not None else int(im.camera_id)
        ni = pycolmap.Image(name=im.name, keypoints=kps, camera_id=cid, image_id=iid)
        ni.frame_id = im.frame_id
        new.add_image(ni)
    for fid in rec.reg_frame_ids():
        if new.frames[fid].has_pose:
            new.register_frame(fid)
    for pt in rec.points3D.values():
        tr = pycolmap.Track()
        for el in pt.track.elements:
            tr.add_element(el.image_id, el.point2D_idx)
        new.add_point3D(pt.xyz, tr, pt.color)
    return new, p2, units


def joint_adjust(rec, proj, temps: Dict[str, float], ppm_per_degC: float, T0: float,
                 covariance: bool = False, max_iterations: int = 100, refine_rig: Union[bool, str] = "rotation",
                 hold_cameras: Sequence[str] = (), pp_slopes: Optional[Dict[int, Tuple[float, float]]] = None,
                 drift: Optional[Dict[str, Any]] = None, base_of: Optional[Dict[int, int]] = None,
                 extra_map: Optional[Callable] = None, axis_ppm: Optional[Dict[str, Tuple[float, float]]] = None,
                 k1_slopes: Optional[Dict[str, float]] = None, **kw) -> Dict[str, Any]:
    """One adjustment of a merged reconstruction with the form's terms: fx, fy scaled per image by ``axis_ppm``
    (else ``ppm_per_degC``), the principal points moved by ``pp_slopes``, k1 by ``k1_slopes``, the rig turned by
    ``drift`` (:func:`thermal_keypoint_map`, :func:`k1_keypoint_map`); ``extra_map`` (e.g. the early-mission
    offsets) is applied after them."""
    from ..sfm.reconstruction import bundle_adjust
    ks = keypoint_scales(proj, temps, ppm_per_degC, T0, axis_ppm=axis_ppm)
    if pp_slopes or drift or base_of:
        right = sorted({2} | {int(k) for k, v in (base_of or {}).items() if int(v) == 2})
        kw["keypoint_map"] = thermal_keypoint_map(proj, temps, T0, pp_slopes, drift, right_camera_ids=right,
                                                  base_of=base_of)
    if k1_slopes and any(float(v or 0.0) for v in k1_slopes.values()):
        km = k1_keypoint_map(proj, temps, T0, k1_slopes)
        base0 = kw.get("keypoint_map")
        kw["keypoint_map"] = (lambda iid, k, cam: km(iid, base0(iid, k, cam), cam)) if base0 else km
    if extra_map is not None:
        base = kw.get("keypoint_map")
        kw["keypoint_map"] = (lambda iid, k, cam: extra_map(iid, base(iid, k, cam), cam)) if base else extra_map
    args = dict(BA_DEFAULTS)
    args.setdefault("linear_solver", "sparse_schur")        # blocks share only the cameras and the rig: very sparse
    args.update(kw)
    ba = bundle_adjust(rec, proj, navcam_aspect="free", refine_rig=refine_rig, max_iterations=max_iterations, covariance=covariance,
                       keypoint_scale=ks, hold_cameras=hold_cameras, **args)
    ba["ppm_per_degC"] = float(ppm_per_degC)
    ba["axis_ppm_per_degC"] = {k: list(map(float, v)) for k, v in (axis_ppm or {}).items()} or None
    ba["pp_slopes_px_per_degC"] = {int(k): list(map(float, v)) for k, v in (pp_slopes or {}).items()} or None
    ba["drift"] = bool(drift)
    ba["T0_degC"] = float(T0)
    ba["k1_slopes_per_degC"] = {k: float(v) for k, v in (k1_slopes or {}).items()} or None
    return ba


def fit_profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The minimum of a parabola through (value, cost) rows and its sd sqrt(variance factor / curvature)."""
    x = np.array([r["ppm_per_degC"] for r in rows])
    c = np.array([r["cost"] for r in rows])
    A = np.c_[np.ones_like(x), x, x ** 2]
    a0, a1, a2 = np.linalg.lstsq(A, c, rcond=None)[0]
    best = -a1 / (2 * a2) if a2 > 0 else float(x[np.argmin(c)])
    vf = float(np.median([r["variance_factor"] for r in rows]))
    sd = math.sqrt(vf / (2 * a2)) if a2 > 0 else float("nan")          # cost = 1/2 chi2: var = vf / C''
    return {"rows": rows, "best_ppm_per_degC": float(best), "sd_ppm_per_degC": sd, "curvature": float(2 * a2),
            "variance_factor": vf, "fit": [float(a0), float(a1), float(a2)]}


# ================================================================ camera comparison
def pycolmap_camera(model: str, params: Sequence[float], width: int = 5120, height: int = 3840):
    import pycolmap
    return pycolmap.Camera(model=model, width=width, height=height, params=np.asarray(params, float))


def compare(a, b, step: float = 64.0, rotation_radius: float = 0.85) -> Dict[str, float]:
    """Where camera ``b`` puts the rays of a grid of ``a``'s pixels (pycolmap cameras, any model), after the best
    rotation (fitted inside ``rotation_radius`` of the half-diagonal) is removed: rms / centre / corner pixels."""
    w, h = a.width, a.height
    u, v = np.meshgrid(np.arange(step / 2, w, step), np.arange(step / 2, h, step))
    uv = np.c_[u.ravel(), v.ravel()]
    ra = np.array(a.cam_from_img(uv), float)
    rb = np.array(b.cam_from_img(uv), float)
    da = np.c_[ra, np.ones(len(ra))]
    db = np.c_[rb, np.ones(len(rb))]
    da /= np.linalg.norm(da, axis=1)[:, None]
    db /= np.linalg.norm(db, axis=1)[:, None]
    rr = np.linalg.norm(uv - np.array([w, h]) / 2, axis=1) / (np.hypot(w, h) / 2)
    ok = np.all(np.isfinite(da), 1) & np.all(np.isfinite(db), 1)
    sel = ok & (rr <= rotation_radius)
    H = da[sel].T @ db[sel]
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Q = Vt.T @ D @ U.T                                   # da -> rotated into b's frame
    X = (da[ok] @ Q.T)
    pb = np.array(b.img_from_cam(X, check_cheirality=False), float)
    d = np.linalg.norm(pb - uv[ok], axis=1)
    r_ok = rr[ok]
    from scipy.spatial.transform import Rotation
    return {"rms_px": float(np.sqrt(np.nanmean(d ** 2))), "centre_rms_px": float(np.sqrt(np.nanmean(d[r_ok < 0.3] ** 2))),
            "corner_rms_px": float(np.sqrt(np.nanmean(d[r_ok > 0.85] ** 2))), "max_px": float(np.nanmax(d)),
            "rotation_mdeg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(Q).as_rotvec())) * 1e3)}


def residual_stats(rec, proj, keypoint_scale: Optional[Dict[int, float]] = None,
                   images: Optional[Sequence[int]] = None, keypoint_map: Optional[Callable] = None) -> Dict[str, float]:
    """Reprojection residuals in native pixels (median, rms, corner median beyond 0.85 of the half-diagonal),
    with the keypoints thermally rescaled as in the adjustment."""
    by = {int(r["image_id"]): float(r.get("downsample_scale", 1.0)) for r in proj.images}
    want = set(images) if images is not None else None
    res, rad = [], []
    cache: Dict[int, np.ndarray] = {}
    for pid, pt in rec.points3D.items():
        for el in pt.track.elements:
            iid = el.image_id
            if want is not None and iid not in want:
                continue
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            if iid not in cache:
                k = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
                if keypoint_scale and iid in keypoint_scale:
                    c0 = np.asarray(cam.params[2:4])
                    k = c0 + (k - c0) / np.asarray(keypoint_scale[iid], float)
                if keypoint_map is not None and len(k):
                    k = keypoint_map(iid, k, cam)
                cache[iid] = k
            xy = cache[iid][el.point2D_idx]
            p = im.project_point(pt.xyz)
            if p is None:
                continue
            s = by.get(iid, 1.0)
            res.append(np.linalg.norm(np.asarray(p) - xy) * s)
            rad.append(np.linalg.norm(xy - np.array([cam.width, cam.height]) / 2) / (np.hypot(cam.width, cam.height) / 2))
    res, rad = np.asarray(res), np.asarray(rad)
    return {"observations": int(res.size), "median_px": float(np.median(res)), "rms_px": float(np.sqrt(np.mean(res ** 2))),
            "corner_median_px": float(np.median(res[rad > 0.85])) if np.any(rad > 0.85) else float("nan"),
            "p95_px": float(np.percentile(res, 95))}
