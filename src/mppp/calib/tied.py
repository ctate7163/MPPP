"""
The Navcam joint as one bundle adjustment (MPPP 0.91; Christian, 20 Oct 2026: "one navcam set of params for all temps
and sols ... one offset for -20 C and one slope with temp, and the bundle adjustment of that and every other
parameter is optimized at once; afterwards, plot the per-block, per-temp-bin and per-sol values with the global
solution").

Every term of the frozen Navcam form is a parameter of one adjustment over the blocks, instead of the sequential
profiles of :func:`mppp.calib.fit.fit_navcam`:

* each eye gets one COLMAP camera per (temperature bin, era): ``bin_deg`` bins of the image temperature, era =
  before / after ``early_sol``.  The cameras are refined by the adjustment and tied (constraint residuals of
  :data:`TIE_SIGMA`, effectively exact) to the global parameters

      fx = G.fx + S.fx (T - T0) + early * E.fx      (likewise fy, cx, cy; E per eye)
      k1 = G.k1 + S.k1 (T - T0)
      k2, p1, p2, k3, k4 = G                        (one distortion for every bin; sx1 = sy1 = 0)

  with T the median temperature of the bin's images (within the bin each image's own temperature counts through
  the keypoint map of the current slopes, updated once per pass);
* each block gets one stereo rig per (rig epoch, bin, era); the epochs are the block's sol clusters cut at the
  pitch knot and split until none spans more than ``max_span`` sols.  The rig rotations are tied to one global rig
  and its drift referred to sol 0, the form of the rig file (``drift.sol0 = 0``, ``knot_ref = 0``):

      q_xyz(s) = Q0 + 1/2 [D s + H min(s, knot)]      (mdeg per 1000 sol; D = pitch, yaw, roll rates,
                                                       H = the early pitch rate minus D.pitch; H.yaw = H.roll = 0)

  so yaw and roll are linear in sol and pitch has its early-mission hinge.  Q0's yaw is held at its start (the
  runs hold the yaw; a constant yaw is the mirror of NR's cx), the translation (baseline) is held.

The global parameters, with their covariance from the one normal matrix, go into the candidate files
(:func:`write_tied`); :func:`diagnostic_pass` then holds them and frees one camera per (block, eye, bin, era) and one
rig per (block, epoch), started at the law - the per-block, per-bin and per-sol values that the figures draw about
the global solution.
"""
from __future__ import annotations

import copy
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import joint as J
from . import models as M
from .constrain import sol_epochs, thermal_terms

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]

TIED_PARAMS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4")     # THIN_PRISM_FISHEYE indices 0-9
SLOPE_TERMS = ("fx", "fy", "cx", "cy", "k1")                                  # per degC (px; k1 unitless)
EARLY_TERMS = ("fx", "fy", "cx", "cy")                                        # px, per eye
ANGLES = ("pitch", "yaw", "roll")                                             # rotation-vector x, y, z
TIE_SIGMA = {"fx": 1e-3, "fy": 1e-3, "cx": 1e-3, "cy": 1e-3, "k": 1e-7, "q": 1e-8}
# 0.94.6: the covariance of the global terms with the ties eliminated exactly (mppp.calib.tiecov) - COLMAP's estimator
# on the tied problem gave them infinite from about a dozen blocks on (20, 24, 25 Oct: 22-29 blocks)
EXACT_TIE_COVARIANCE = True
# 0.94.6 (study): after Ceres' solve, Levenberg-Marquardt with the ties as exact constraints (mppp.calib.tiesolve)
EXACT_TIE_SOLVE = False
BIN_DEG = 2.0                    # fine bins; within a bin the images' own temperatures count (keypoint map)
DIAG_BIN_MIN_IMAGES = 8          # 22 Oct: the diagnostic pass merges a block's adjacent fine bins (per era) until
DIAG_BIN_MAX_DEG = 10.0          #   each eye has this many images, at most this wide - the cold morning images
                                 #   (below -30 degC) are spread 1-3 per 2 degC bin and were all under the figure cut
FIG_BIN_MIN_IMAGES = 4           # the bin cameras drawn in the intrinsics figure (the rig epochs keep 8)
POINTS_PER_BLOCK = 8000          # Christian, 20 Oct: 8000 for the pipeline passes
MAX_SPAN_SOLS = 50.0                 # rig epochs of the joint: at most 50 sols (doubled 20 Oct; was 25)
KNOTS_TESTED = tuple(float(k) for k in np.linspace(200.0, 400.0, 25))   # the pitch transitions after the BA (20 Oct)
EARLY_SOL = 380.0
KNOT_SOL = 250.0                 # the pitch hinge (the free-knot refit of 20 Oct: 250, range 200-290)
HOLD_DEFAULT = ("Q0_yaw",)       # the runs hold the yaw: a constant yaw mirrors NR's cx
PASS_JUMP_FACTOR = 1.5           # 0.94.5: a pass starting this far above the last one's end gets one more pass
EARLY_Z_MIN = 3.0                # 0.94.5 (Christian 24 Oct): an early-mission offset is kept only where |z| reaches this
ISO_F = True    # one focal slope per eye (fy's ppm/degC = fx's): 20 Oct joint fx/fy 36.0/37.1, 35.6/35.3 ppm/degC,
                # the per-block bins fy-fx +2.0 +- 4.3, +3.7 +- 4.5 ppm/degC (Christian: "set those equal if possible")
_IDX = {n: i for i, n in enumerate(("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"))}
_MDEG = np.radians(1e-3)


# ====================================================================================== parameter blocks
def _linear_tie_class():
    import pyceres

    class LinearTie(pyceres.CostFunction):
        """r = (A x) / sigma over the concatenated parameter blocks (A constant)."""

        def __init__(self, A: np.ndarray, sigma: np.ndarray, sizes: Sequence[int]):
            super().__init__()
            self.A = np.asarray(A, float) / np.asarray(sigma, float)[:, None]
            self.sizes = list(map(int, sizes))
            self.offsets = np.cumsum([0] + self.sizes)
            self.set_num_residuals(self.A.shape[0])
            self.set_parameter_block_sizes(self.sizes)

        def Evaluate(self, parameters, residuals, jacobians):
            x = np.concatenate([np.asarray(p, float) for p in parameters])
            residuals[:] = self.A @ x
            if jacobians is not None:
                for j, jac in enumerate(jacobians):
                    if jac is not None:
                        a, b = self.offsets[j], self.offsets[j + 1]
                        jac[:] = self.A[:, a:b].ravel()
            return True

    return LinearTie


def q_xyz(R: np.ndarray) -> np.ndarray:
    """The x, y, z of the unit quaternion of ``R`` (w >= 0): half the rotation vector for small angles."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(R, float)).as_quat()          # x, y, z, w
    return (q if q[3] >= 0 else -q)[:3]


def R_of_q(qxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = np.asarray(qxyz, float)
    w = np.sqrt(max(0.0, 1.0 - float(q @ q)))
    return Rotation.from_quat([q[0], q[1], q[2], w]).as_matrix()


def rig_q_at(Q0: np.ndarray, D: np.ndarray, H: np.ndarray, sol: float, knot: float) -> np.ndarray:
    """The law: q_xyz(s) = Q0 + 1/2 mdeg [D s + H min(s, knot)] / 1000."""
    s = float(sol)
    return np.asarray(Q0, float) + 0.5 * _MDEG * (np.asarray(D, float) * s + np.asarray(H, float) * min(s, knot)) / 1000.0


def start_values(cmods_dir: Optional[PathLike] = None, knot_sol: Optional[float] = KNOT_SOL) -> Dict[str, Any]:
    """The global parameters' start from a cameras folder (default the models in use): G, S, E per eye, the rig's
    Q0 at sol 0, D (rates) and H (pitch hinge), the knot (``knot_sol``, default :data:`KNOT_SOL` = 250; None takes
    the rig file's)."""
    d = M._folder(cmods_dir)
    nl, nr, rj = M.load_camera("N", "L", d), M.load_camera("N", "R", d), M.load_rig("N", d)
    if rj is None:
        raise FileNotFoundError(f"no Navcam rig file in {d}")
    out: Dict[str, Any] = {"T0": float((nl.get("thermal") or {}).get("T0_degC", -20.0)), "G": {}, "S": {}, "E": {},
                           "model": nl["model"], "folder": str(d)}
    for eye, j in (("NL", nl), ("NR", nr)):
        p = np.asarray(j["params"], float)
        th = thermal_terms(j.get("thermal"))
        out["G"][eye] = p[:10].copy()
        out["S"][eye] = np.array([1e-6 * p[0] * th["fx_ppm"], 1e-6 * p[1] * th["fy_ppm"], th["cx_px"], th["cy_px"], th["k1"]])
        em = j.get("early_mission") or {}
        out["E"][eye] = np.array([float(em.get(f"d{k}_px", 0.0)) for k in EARLY_TERMS])
        out["early_sol"] = float(em.get("before_sol", EARLY_SOL))
    dr = rj.get("drift") or {}
    knot = float(knot_sol if knot_sol is not None else (dr.get("knot_sol") or KNOT_SOL))
    D = np.array([1e3 * float(dr.get(f"{a}_mdeg_per_sol") or 0.0) for a in ANGLES])
    early_pitch = 1e3 * float(dr.get("pitch_early_mdeg_per_sol", dr.get("pitch_mdeg_per_sol")) or 0.0)
    H = np.array([early_pitch - D[0], 0.0, 0.0])
    sol0 = float(dr.get("sol0") or 0.0)
    knot_ref = float(dr.get("knot_ref") or 0.0)
    q_file = q_xyz(np.asarray(rj["R_sensor_from_ref"], float))
    # the file's rotation is the law at sol0 (offset rate (s - sol0) + (early - rate)(min(s, knot) - knot_ref)):
    # refer it to sol 0 with knot_ref 0
    q0 = q_file - 0.5 * _MDEG * (D * sol0 + H * (min(sol0, knot) - knot_ref)) / 1000.0
    out.update({"knot": knot, "D": D, "H": H, "Q0": q0, "t": np.asarray(rj["t_sensor_from_ref_m"], float),
                "R": np.asarray(rj["R_sensor_from_ref"], float), "cameras": {"NL": nl, "NR": nr}, "rig": rj})
    return out


def camera_params(st: Dict[str, Any], eye: str, T: float, early: bool) -> np.ndarray:
    """The law's camera parameters (12, THIN_PRISM_FISHEYE) of an eye at temperature ``T`` and era."""
    p = np.zeros(12)
    p[:10] = st["G"][eye]
    dT = T - st["T0"]
    for i, k in enumerate(SLOPE_TERMS):
        p[_IDX[k]] += st["S"][eye][i] * dT
    if early:
        for i, k in enumerate(EARLY_TERMS):
            p[_IDX[k]] += st["E"][eye][i]
    return p


# ====================================================================================== the units
def frame_table(rec, proj, idx: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Per registered frame of a merged reconstruction: median temperature and sol of its images, block, rig."""
    by_id = {int(r["image_id"]): r for r in proj.images}
    temps = idx["temps"]
    frames = {}
    for fid, fr in rec.frames.items():
        ims = [d.id for d in fr.data_ids if d.id in rec.images]
        if not ims or not fr.has_pose:
            continue
        T = [float(temps[rec.images[i].name]) for i in ims if rec.images[i].name in temps]
        sols = [int(by_id[i]["sol"]) for i in ims if i in by_id and by_id[i].get("sol") is not None]
        if not T or not sols:
            continue
        frames[fid] = {"T": float(np.median(T)), "sol": int(np.median(sols)), "scape": idx["scape_of_image"][ims[0]],
                       "rig": int(fr.rig_id), "images": ims}
    return frames


def merge_bins(frames: Dict[Any, Dict[str, Any]], eye_counts: Dict[Any, Dict[str, int]], min_images: int,
               max_deg: float, bin_deg: float = BIN_DEG) -> int:
    """Relabel ``frames[k]["bin"]`` so that, within each (scape, era), adjacent fine bins (ascending temperature)
    are merged until every eye with images in the group has ``min_images`` (``eye_counts[k]``: {eye: images} of
    frame k), a group spanning at most ``max_deg``; a short remainder joins the group below it when that stays
    within ``max_deg`` (so does a thin group cut short by the width).  A merged group keeps the label of its coldest fine bin.  Returns the number of fine bins
    folded into another."""
    groups: Dict[tuple, Dict[int, List[Any]]] = {}
    for k, f in frames.items():
        groups.setdefault((f["scape"], f["early"]), {}).setdefault(f["bin"], []).append(k)
    width = max(1, int(round(max_deg / bin_deg)))
    folded = 0

    def enough(ks):
        c: Dict[str, int] = {}
        for k in ks:
            for e, n in eye_counts[k].items():
                c[e] = c.get(e, 0) + n
        return bool(c) and min(c.values()) >= min_images

    for bins in groups.values():
        out: List[List[int]] = []
        cur: List[int] = []
        for b in sorted(bins):
            if cur and b - cur[0] + 1 > width:               # too wide: a thin group joins the one below if it fits
                if out and not enough([k for x in cur for k in bins[x]]) and cur[-1] - out[-1][0] + 1 <= width:
                    out[-1] += cur
                else:
                    out.append(cur)
                cur = []
            cur.append(b)
            if enough([k for x in cur for k in bins[x]]):
                out.append(cur); cur = []
        if cur:
            if out and cur[-1] - out[-1][0] + 1 <= width:
                out[-1] += cur
            else:
                out.append(cur)
        for grp in out:
            for b in grp:
                for k in bins[b]:
                    frames[k]["bin"] = grp[0]
            folded += len(grp) - 1
    return folded


def tied_reconstruction(rec, proj, idx: Dict[str, Any], st: Dict[str, Any], bin_deg: float = BIN_DEG,
                        early_sol: Optional[float] = None, max_span: float = MAX_SPAN_SOLS,
                        gap: float = 50.0, breaks: Optional[Sequence[float]] = None, per_block: bool = False,
                        pointing: bool = False, passthrough: bool = False, merge_min_images: int = 0,
                        merge_max_deg: float = DIAG_BIN_MAX_DEG):
    """A copy of a merged reconstruction (:func:`mppp.calib.joint.merge_scapes`: cameras 1 NL, 2 NR; rigs 1 stereo,
    2 NR alone, 3 NL alone) with one camera per (eye, temperature bin, era) - per (block, eye, bin, era) with
    ``per_block`` (the diagnostic pass) - and one rig per (block, rig epoch, bin, era), every unit started at the
    law of ``st``.  Returns (rec, proj, units): units {"cameras": {cid: {...}}, "rigs": {rid: {...}}}.
    ``passthrough`` (the Mastcam-Z joint): the frames of other cameras (merged with ``include_zcam``) are copied as
    they are - their cameras renumbered after the Navcam units (``units["passthrough_cameras"]`` {old: new}), one
    rig each, the same image ids, their observations kept in the points.  ``merge_min_images`` (the diagnostic
    pass): a block's thin adjacent bins merged (:func:`merge_bins`, at most ``merge_max_deg`` wide)."""
    import pycolmap
    from ..sfm.project import navcam_distortion_terms
    early_sol = float(st.get("early_sol", EARLY_SOL) if early_sol is None else early_sol)
    knot = float(st["knot"])
    breaks = (knot,) if breaks is None else breaks
    by_id = {int(r["image_id"]): r for r in proj.images}
    frames = frame_table(rec, proj, idx)
    if passthrough:
        frames = {k: f for k, f in frames.items() if f["rig"] in (1, 2, 3)}
    ep_of: Dict[int, int] = {}
    for scape in {f["scape"] for f in frames.values()}:
        fids = [k for k, f in frames.items() if f["scape"] == scape]
        eps = sol_epochs([frames[k]["sol"] for k in fids], gap, breaks, max_span)
        for k in fids:
            s = frames[k]["sol"]
            ep_of[k] = next(i for i, (a, b) in enumerate(eps) if a <= s <= b)
    T0 = st["T0"]
    for k, f in frames.items():
        f["bin"] = 0 if pointing else int(np.floor((f["T"] - T0) / bin_deg))     # pointing: one unit per block, eye, era
        f["early"] = bool(f["sol"] < early_sol)
        f["epoch"] = ep_of[k]
    eye_of_cam = {1: "NL", 2: "NR"}
    if merge_min_images and per_block and not pointing:
        counts = {k: {} for k in frames}
        for k, f in frames.items():
            for i in f["images"]:
                e = eye_of_cam[int(rec.images[i].camera_id)]
                counts[k][e] = counts[k].get(e, 0) + 1
        nf = merge_bins(frames, counts, merge_min_images, merge_max_deg, bin_deg)
        log.info(f"diagnostic bins: {nf} thin {bin_deg:g} degC bins merged into their neighbours "
                 f"(>= {merge_min_images} images per eye, <= {merge_max_deg:g} degC)")

    def cam_key(f, eye):
        return (f["scape"] if per_block else "", eye, f["bin"], f["early"])

    cam_units: Dict[tuple, Dict[str, Any]] = {}
    for k, f in frames.items():
        for i in f["images"]:
            u = cam_units.setdefault(cam_key(f, eye_of_cam[int(rec.images[i].camera_id)]), {"T": [], "images": 0})
            u["T"].append(f["T"])
            u["images"] += 1
    new = pycolmap.Reconstruction()
    cam_id: Dict[tuple, int] = {}
    newproj = copy.deepcopy(proj)
    newproj.cameras = {}
    db: Dict[str, int] = {}
    units: Dict[str, Any] = {"cameras": {}, "rigs": {}, "per_block": bool(per_block)}
    for n, (key, u) in enumerate(sorted(cam_units.items()), start=1):
        scape, eye, b, early = key
        T = float(np.median(u["T"]))
        p = camera_params(st, eye, T, early)
        base = rec.cameras[1 if eye == "NL" else 2]
        new.add_camera(pycolmap.Camera(camera_id=n, model=base.model, width=base.width, height=base.height, params=p))
        name = f"{eye}_t{b:+d}_{'early' if early else 'late'}" + (f"@{scape}" if per_block else "")
        cam_id[key] = n
        db[name] = n
        c = {"model": base.model.name, "params": list(map(float, p)), "width": base.width, "height": base.height}
        if per_block and pointing:  # the pointing pass: fx, fy, cx, cy free, the distortion held
            c = navcam_distortion_terms(c, "hold", pp="refine")
            newproj.cameras[name] = c
        elif per_block:            # the diagnostic bins like the block runs' thermal bins: fx, fy, k1 free, the rest held
            c = navcam_distortion_terms(c, "hold", pp="hold")
            c["fixed_params"] = [n for n in c["fixed_params"] if n != "k1"]
            c["free_params"] = sorted(set(c.get("free_params") or []) | {"fx", "fy", "k1"})
            newproj.cameras[name] = c
        else:
            newproj.cameras[name] = navcam_distortion_terms(c, "refine")
        units["cameras"][n] = {"key": name, "scape": scape or None, "eye": eye, "bin": b, "early": early, "T": T,
                               "images": u["images"]}
    newproj.settings = dict(newproj.settings)
    newproj.settings["database"] = {"cameras": db}
    newproj.settings["navcam_rig_yaw"] = "refine"
    # 0.94.5: the roll too - the blocks hold it (0.91: per epoch it is poorly determined) and their settings come
    # through merge_scapes, so the joint held every unit's roll at its start and the global roll and roll rate were
    # never refitted (20-24 Oct: -56.795270 mdeg and -4.933370 mdeg/1000 sol to 7 digits, sd 6e-5 mdeg)
    newproj.settings["navcam_rig_roll"] = "refine"
    newproj.settings["navcam_rig_epochs_tie"] = False
    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))      # noqa: E731
    t0 = np.asarray(rec.rigs[1].sensor_from_rig(next(iter(rec.rigs[1].non_ref_sensors))).translation)
    rig_id: Dict[tuple, int] = {}
    rig_members: Dict[int, List[int]] = {}
    key_frames: Dict[tuple, List[int]] = {}

    def rig_key(f):
        return (("stereo", f["scape"], f["epoch"], f["bin"], f["early"]) if f["rig"] == 1 else
                ("mono", "NR" if f["rig"] == 2 else "NL", f["scape"] if per_block else "", f["bin"], f["early"]))

    for k, f in sorted(frames.items()):
        key_frames.setdefault(rig_key(f), []).append(k)
    for key, mem in key_frames.items():
        rid = len(rig_id) + 1
        r = pycolmap.Rig(rig_id=rid)
        if key[0] == "stereo":
            scape, b, early = key[1], key[3], key[4]
            s_med = float(np.median([frames[k]["sol"] for k in mem]))
            q = rig_q_at(st["Q0"], st["D"], st["H"], s_med, knot)
            ck = lambda eye: cam_key({"scape": scape, "bin": b, "early": early}, eye)      # noqa: E731
            r.add_ref_sensor(sensor(cam_id[ck("NL")]))
            r.add_sensor(sensor(cam_id[ck("NR")]), pycolmap.Rigid3d(pycolmap.Rotation3d(R_of_q(q)), t0))
        else:
            r.add_ref_sensor(sensor(cam_id[(key[2], key[1], key[3], key[4])]))
        new.add_rig(r)
        rig_id[key] = rid
        rig_members[rid] = mem
    for key, rid in rig_id.items():
        if key[0] != "stereo":
            continue
        mem = rig_members[rid]
        scape, b, early = key[1], key[3], key[4]
        units["rigs"][rid] = {"scape": scape, "epoch": key[2], "bin": b, "early": early,
                              "sol": float(np.median([frames[k]["sol"] for k in mem])),
                              "sols": [int(min(frames[k]["sol"] for k in mem)), int(max(frames[k]["sol"] for k in mem))],
                              "T": float(np.median([frames[k]["T"] for k in mem])), "frames": len(mem),
                              "stations": len({str((by_id.get(frames[k]["images"][0]) or {}).get("station")
                                                   or rec.images[frames[k]["images"][0]].name[:9]) for k in mem}),
                              "nr_camera": cam_id[cam_key({"scape": scape, "bin": b, "early": early}, "NR")]}
    for k, f in sorted(frames.items()):
        nf = pycolmap.Frame(frame_id=k, rig_id=rig_id[rig_key(f)])
        for i in f["images"]:
            eye = eye_of_cam[int(rec.images[i].camera_id)]
            nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cam_id[cam_key(f, eye)]), id=i))
        nf.rig_from_world = rec.frames[k].rig_from_world
        new.add_frame(nf)
        for i in f["images"]:
            im = rec.images[i]
            eye = eye_of_cam[int(im.camera_id)]
            ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D], float).reshape(-1, 2),
                                camera_id=cam_id[cam_key(f, eye)], image_id=i)
            ni.frame_id = k
            new.add_image(ni)
            if i in by_id:
                by_id[i]["camera_id"] = cam_id[cam_key(f, eye)]
    for k in frames:
        new.register_frame(k)
    keep = set().union(*[set(f["images"]) for f in frames.values()]) if frames else set()
    units["passthrough_cameras"] = {}
    if passthrough:
        cmap: Dict[int, int] = {}
        next_c = max(cam_id.values(), default=0) + 1
        next_r = max(rig_id.values(), default=0) + 1
        for cid in sorted(c for c in rec.cameras if c not in (1, 2)):
            c = rec.cameras[cid]
            new.add_camera(pycolmap.Camera(camera_id=next_c, model=c.model, width=c.width, height=c.height,
                                           params=np.array(c.params, float)))
            zr = pycolmap.Rig(rig_id=next_r)
            zr.add_ref_sensor(sensor(next_c))
            new.add_rig(zr)
            cmap[cid] = (next_c, next_r)
            next_c += 1
            next_r += 1
        reg = set(rec.reg_image_ids())
        for fid, fr in sorted(rec.frames.items()):
            if fid in frames or not fr.has_pose:
                continue
            ims = [d.id for d in fr.data_ids if d.id in reg]
            if not ims or any(int(rec.images[i].camera_id) not in cmap for i in ims):
                continue
            nc, nr_ = cmap[int(rec.images[ims[0]].camera_id)]
            nf = pycolmap.Frame(frame_id=fid, rig_id=nr_)
            for i in ims:
                nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cmap[int(rec.images[i].camera_id)][0]), id=i))
            nf.rig_from_world = fr.rig_from_world
            new.add_frame(nf)
            for i in ims:
                im = rec.images[i]
                ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D], float).reshape(-1, 2),
                                    camera_id=cmap[int(im.camera_id)][0], image_id=i)
                ni.frame_id = fid
                new.add_image(ni)
                if i in by_id:
                    by_id[i]["camera_id"] = cmap[int(im.camera_id)][0]
                keep.add(i)
            new.register_frame(fid)
        units["passthrough_cameras"] = {int(k): int(v[0]) for k, v in cmap.items()}
    for pt in rec.points3D.values():
        tr = pycolmap.Track()
        for el in pt.track.elements:
            if el.image_id in keep:
                tr.add_element(el.image_id, el.point2D_idx)
        if tr.length() >= 2:
            new.add_point3D(pt.xyz, tr, pt.color)
    newproj.images = [dict(r, camera_id=by_id[int(r["image_id"])]["camera_id"]) for r in proj.images
                      if int(r["image_id"]) in keep]
    units["frames"] = len(frames)
    units["epochs_max_span_sols"] = float(max_span)
    units["bin_deg"] = float(bin_deg)
    units["early_sol"] = early_sol
    units["knot"] = knot
    return new, newproj, units


# ====================================================================================== the ties
_HELD_NAMES = {"S_NL": SLOPE_TERMS, "S_NR": SLOPE_TERMS, "E_NL": EARLY_TERMS, "E_NR": EARLY_TERMS,
               "G_NL": TIED_PARAMS, "G_NR": TIED_PARAMS, "D": ANGLES, "H": ANGLES, "Q0": ANGLES}


def make_ties(st: Dict[str, Any], units: Dict[str, Any], hold: Sequence[str] = HOLD_DEFAULT, tie_rigs: bool = True,
              tie_cameras: bool = True, pp_prior: Optional[Dict[str, float]] = None, iso_f: bool = ISO_F,
              exact_solve: Optional[bool] = None):
    """The ``extra_residuals`` callable for :func:`mppp.sfm.reconstruction.bundle_adjust` and the global parameter
    arrays it ties to.  ``hold``: global terms held at their start (``"E_NR"``, ``"S_NL_cy"``, ``"Q0_yaw"``, ``"H"``
    ...).  ``H``'s yaw and roll are always held at 0 (linear yaw and roll trends); terms the data cannot see are held
    too (the early offsets without early-mission cameras, the hinge without rig units before the knot).  With
    ``tie_cameras`` / ``tie_rigs`` False the units stay free and the globals are held (the diagnostic pass).
    ``iso_f``: one relative focal slope per eye - fy's tie reads fx's slope scaled by fy/fx (the S array's fy entry
    is held and kept equal by :func:`sync_iso`)."""
    import pyceres
    LinearTie = _linear_tie_class()
    g = {"G_NL": st["G"]["NL"].copy(), "G_NR": st["G"]["NR"].copy(), "S_NL": st["S"]["NL"].copy(),
         "S_NR": st["S"]["NR"].copy(), "E_NL": st["E"]["NL"].copy(), "E_NR": st["E"]["NR"].copy(),
         "Q0": st["Q0"].copy(), "D": st["D"].copy(), "H": st["H"].copy()}
    sig = np.array([TIE_SIGMA["fx"], TIE_SIGMA["fy"], TIE_SIGMA["cx"], TIE_SIGMA["cy"]] + [TIE_SIGMA["k"]] * 6)
    knot = float(st["knot"])
    hold = list(hold) + ["H_yaw", "H_roll"]
    if iso_f and tie_cameras:
        hold += ["S_NL_fy", "S_NR_fy"]
        sync_iso(g)
    for eye in ("NL", "NR"):
        if not any(u["early"] and u["eye"] == eye for u in units["cameras"].values()):
            hold.append(f"E_{eye}")
    sols = sorted({round(float(u["sol"])) for u in units["rigs"].values()})
    if tie_rigs and (not any(s_ < knot for s_ in sols) or len(sols) < 3):
        hold.append("H")                                   # the hinge needs a unit before the knot and three sols
    if not tie_rigs:
        hold += ["Q0", "D", "H"]
    if not tie_cameras:
        hold += ["G_NL", "G_NR", "S_NL", "S_NR", "E_NL", "E_NR"]

    keep: List[np.ndarray] = []

    def hook(prob, cam_blocks=None, rig_blocks=None, key_of=None, work=None):
        n_cam = n_rig = 0
        tie_ids = []                                         # 0.94.6: for the exact-tie covariance
        specs = []                                           # 0.94.6: the ties as constraints (mppp.calib.tiesolve)
        if tie_cameras:
            for cid, u in units["cameras"].items():
                blk = cam_blocks.get(int(cid))
                if blk is None or not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                    continue
                eye = u["eye"]
                dT = u["T"] - st["T0"]
                A = np.zeros((10, 12 + 10 + 5 + 4))
                for i, k in enumerate(TIED_PARAMS):
                    A[i, _IDX[k]] = 1.0                       # the bin camera
                    A[i, 12 + i] = -1.0                        # G
                for i, k in enumerate(SLOPE_TERMS):
                    A[TIED_PARAMS.index(k), 22 + i] = -dT      # S
                if iso_f:                                      # fy follows fx's relative slope
                    A[TIED_PARAMS.index("fy"), 22 + 1] = 0.0
                    A[TIED_PARAMS.index("fy"), 22 + 0] = -dT * float(g[f"G_{eye}"][1] / g[f"G_{eye}"][0])
                if u["early"]:
                    for i, k in enumerate(EARLY_TERMS):
                        A[TIED_PARAMS.index(k), 27 + i] = -1.0  # E (this eye)
                bl = [blk, g[f"G_{eye}"], g[f"S_{eye}"], g[f"E_{eye}"]]
                tie_ids.append(prob.add_residual_block(LinearTie(A, sig, [12, 10, 5, 4]), None, bl))
                specs.append({"id": tie_ids[-1], "A": A, "sigma": sig, "sizes": [12, 10, 5, 4], "blocks": bl})
                n_cam += 1
        if tie_rigs:
            for rid, u in units["rigs"].items():
                blk = rig_blocks.get((int(rid), int(u["nr_camera"])))
                if blk is None or not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                    continue
                s = float(u["sol"])
                A = np.zeros((3, 7 + 3 + 3 + 3))
                for i in range(3):
                    A[i, i] = 1.0                                    # the rig unit's quaternion x, y, z
                    A[i, 7 + i] = -1.0                               # Q0
                    A[i, 10 + i] = -0.5 * _MDEG * s / 1000.0         # D (mdeg / 1000 sol)
                    A[i, 13 + i] = -0.5 * _MDEG * min(s, knot) / 1000.0   # H
                bl = [blk, g["Q0"], g["D"], g["H"]]
                tie_ids.append(prob.add_residual_block(LinearTie(A, np.full(3, TIE_SIGMA["q"]), [7, 3, 3, 3]), None, bl))
                specs.append({"id": tie_ids[-1], "A": A, "sigma": np.full(3, TIE_SIGMA["q"]), "sizes": [7, 3, 3, 3],
                              "blocks": bl, "quaternion": True})
                n_rig += 1
        n_pp = 0
        consts = []
        for cid, u in units["cameras"].items():             # a prior on the units' principal point about the law
            psig = (pp_prior or {}).get(u["eye"])
            blk = cam_blocks.get(int(cid))
            if not psig or blk is None or not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                continue
            law = camera_params(dict(st, G={"NL": g["G_NL"], "NR": g["G_NR"]}, S={"NL": g["S_NL"], "NR": g["S_NR"]},
                                        E={"NL": g["E_NL"], "NR": g["E_NR"]}), u["eye"], u["T"], u["early"])
            c0 = np.array([law[2], law[3]], float)
            consts.append(c0)
            A = np.zeros((2, 14)); A[0, 2] = A[1, 3] = 1.0; A[0, 12] = A[1, 13] = -1.0
            prob.add_residual_block(LinearTie(A, np.full(2, float(psig)), [12, 2]), None, [blk, c0])
            prob.set_parameter_block_constant(c0)
            n_pp += 1
        keep.extend(consts)                                  # keep the constant blocks alive through the solve
        held: Dict[str, set] = {}
        for h in hold:
            for gname in g:
                if h == gname:
                    held.setdefault(gname, set()).update(range(len(g[gname])))
                elif h.startswith(gname + "_"):
                    c = h[len(gname) + 1:]
                    names = _HELD_NAMES.get(gname, ())
                    if c in names:
                        held.setdefault(gname, set()).add(names.index(c))
        for gname, ix in held.items():
            if not prob.has_parameter_block(g[gname]):
                continue
            if len(ix) == len(g[gname]):
                prob.set_parameter_block_constant(g[gname])
            else:
                prob.set_manifold(g[gname], pyceres.SubsetManifold(len(g[gname]), sorted(ix)))
        return {"cov_blocks": {k: v for k, v in g.items() if prob.has_parameter_block(v)
                               and not prob.is_parameter_block_constant(v)},
                "ties": {"cameras": n_cam, "rigs": n_rig, "pp_priors": n_pp}, "held": {k: sorted(v) for k, v in held.items()},
                "tie_ids": tie_ids if EXACT_TIE_COVARIANCE else [], "tie_specs": specs,
                "exact_solve": bool(EXACT_TIE_SOLVE if exact_solve is None else exact_solve)}

    return hook, g


def sync_iso(g: Dict[str, np.ndarray]) -> None:
    """The isotropic focal slope: fy's slope set to fx's relative slope (same ppm/degC) in place."""
    for eye in ("NL", "NR"):
        g[f"S_{eye}"][1] = g[f"S_{eye}"][0] * g[f"G_{eye}"][1] / g[f"G_{eye}"][0]


def _sd(cov: Dict[str, Any], name: str, n: int, held: Sequence[int] = ()) -> Optional[np.ndarray]:
    c = (cov.get("blocks") or {}).get(f"extra:{name}")
    if c is None:
        return None
    d = np.sqrt(np.clip(np.diag(np.asarray(c)), 0, None) * float(cov.get("variance_factor", 1.0)))
    free = [i for i in range(n) if i not in set(held)]
    out = np.full(n, np.nan)
    if len(d) == len(free):
        out[free] = d
    return out


def keypoint_terms(units: Dict[str, Any], proj, temps: Dict[str, float], g: Dict[str, np.ndarray], iso_f: bool = False):
    """Within a bin each image's own temperature counts: its keypoints are moved by the current slopes from the bin
    camera's temperature to the image's (scale about the principal point; principal-point shift).  Returns
    (keypoint_scale, keypoint_map) for :func:`bundle_adjust` and :func:`residual_stats`."""
    T_unit = {int(cid): float(u["T"]) for cid, u in units["cameras"].items()}
    eye_unit = {int(cid): u["eye"] for cid, u in units["cameras"].items()}
    ks, shift = {}, {}
    for r in proj.images:
        iid, cid = int(r["image_id"]), int(r["camera_id"])
        T = temps.get(r["name"])
        if T is None or cid not in T_unit:
            continue
        eye, dT = eye_unit[cid], float(T) - T_unit[cid]
        Gs, Ss = g[f"G_{eye}"], g[f"S_{eye}"]
        # 0.94.5: with one relative slope (iso) fy's scale is fx's - the ties' - not the held S_fy of the last sync
        ks[iid] = (1.0 + Ss[0] / Gs[0] * dT, 1.0 + (Ss[0] / Gs[0] if iso_f else Ss[1] / Gs[1]) * dT)
        shift[iid] = np.array([Ss[2] * dT, Ss[3] * dT])
    kmap = (lambda iid, k, cam, _s=shift: k - _s[iid] if iid in _s else k)        # noqa: E731
    return ks, kmap


# ====================================================================================== the fits
def adjust_tied(rec, proj, idx: Dict[str, Any], st: Dict[str, Any], units: Dict[str, Any], hold: Sequence[str] = HOLD_DEFAULT,
                tie_rigs: bool = True, tie_cameras: bool = True, iterations: int = 100, covariance: bool = True,
                passes: int = 2, verbose: bool = True, refine_pp: bool = True,
                pp_prior: Optional[Dict[str, float]] = None, iso_f: bool = ISO_F,
                exact_solve: Optional[bool] = None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Any]:
    """The adjustment of a tied reconstruction (``passes`` solves, the slopes of the keypoint map updated between
    them).  Returns (ba result of the last pass, the global arrays, residual statistics)."""
    from ..sfm.reconstruction import bundle_adjust
    iso = bool(iso_f and tie_cameras)
    hook, g = make_ties(st, units, hold=hold, tie_rigs=tie_rigs, tie_cameras=tie_cameras, pp_prior=pp_prior, iso_f=iso,
                        exact_solve=exact_solve)
    temps = idx["temps"]
    ba = None
    n_pass = max(1, int(passes))
    costs: List[Tuple[float, float]] = []
    p_ = 0
    while p_ < n_pass:
        ks, kmap = keypoint_terms(units, proj, temps, g, iso_f=iso)
        last = p_ == n_pass - 1
        ba = bundle_adjust(rec, proj, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=False if pp_prior else refine_pp,
                           refine_principal_point=refine_pp,
                           refine_rig="rotation", max_iterations=iterations, attitude_prior_deg=5.0,
                           covariance=covariance and last, navcam_aspect="free",
                           linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap, extra_residuals=hook)
        if iso:
            sync_iso(g)
        if verbose:
            log.info(f"  pass {p_ + 1}: {ba['brief'][:160]}")
        costs.append((float(ba.get("initial_cost", np.nan)), float(ba.get("final_cost", np.nan))))
        # 0.94.5: a pass that starts far above where the last one ended (24 Oct: 2.54e6 after 1.82e5) means the
        # keypoint map or the ties moved between passes - one more pass, so the covariance is taken at a settled point
        if (last and len(costs) >= 2 and n_pass < int(passes) + 1 and np.isfinite(costs[-1][0])
                and costs[-1][0] > PASS_JUMP_FACTOR * costs[-2][1]):
            log.warning(f"  pass {p_ + 1} started at {costs[-1][0]:.4g} after {costs[-2][1]:.4g}: one more pass")
            n_pass += 1
        p_ += 1
    if ba is not None:
        ba.setdefault("extra", {})
        if isinstance(ba["extra"], dict):
            ba["extra"]["iso_f"] = iso
            ba["extra"]["pass_costs"] = costs
    ks, kmap = keypoint_terms(units, proj, temps, g, iso_f=iso)
    stats = J.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap)
    return ba, g, stats


def _sd_from_units(cov: Dict[str, Any], st: Dict[str, Any], units: Dict[str, Any], held: Dict[str, Any],
                   iso_f: bool = False) -> Dict[str, Any]:
    """The sd of the camera globals propagated from the bin cameras' own covariances (the ties are exact, so the
    globals are the weighted least-squares combination of the bins: cov = (A^T W A)^-1 with W the bins' inverse
    variances, correlations between bins ignored) - the fallback when the problem's covariance of the global
    blocks comes out infinite (22 blocks, 20 Oct)."""
    out = {}
    blocks = cov.get("blocks") or {}
    vf = float(cov.get("variance_factor", 1.0))
    for eye in ("NL", "NR"):
        rows_A, rows_w = [], []
        for cid, u in units["cameras"].items():
            if u["eye"] != eye:
                continue
            c = blocks.get(f"camera:{u['key']}")
            if c is None or not np.all(np.isfinite(np.diag(c))):
                continue
            dT = u["T"] - st["T0"]
            for i, k in enumerate(TIED_PARAMS):
                a = np.zeros(10 + 5 + 4)
                a[i] = 1.0
                if k in SLOPE_TERMS:
                    a[10 + SLOPE_TERMS.index(k)] = dT
                if iso_f and k == "fy":
                    a[11], a[10] = 0.0, dT                     # fy's slope is fx's (ratio fy/fx = 1 to 1e-5)
                if u["early"] and k in EARLY_TERMS:
                    a[15 + EARLY_TERMS.index(k)] = 1.0
                var = float(np.diag(c)[i]) * vf if i < len(np.diag(c)) else np.nan
                if np.isfinite(var) and var > 0:
                    rows_A.append(a); rows_w.append(1.0 / var)
        if len(rows_A) < 19:
            continue
        A, w = np.array(rows_A), np.array(rows_w)
        free = [j for j in range(19) if np.any(A[:, j] != 0)]
        N = (A[:, free].T * w) @ A[:, free]
        try:
            C = np.linalg.inv(N)
        except np.linalg.LinAlgError:
            continue
        sd = np.full(19, np.nan)
        sd[free] = np.sqrt(np.clip(np.diag(C), 0, None))
        if iso_f:
            sd[11] = sd[10]
        out[eye] = {"G": sd[:10], "S": sd[10:15], "E": sd[15:19]}
    return out


def global_covariance(cov: Dict[str, Any], held: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """0.94.6: the exact-tie covariance of every free global term with every other, named (``G_NL_k3``, ``D_pitch``
    ...), in the problem's units (unit variance factor x the problem's variance factor)."""
    full = (cov.get("extra_exact") or {}).get("full")
    if not full:
        return None
    vf = float(cov.get("variance_factor", 1.0))
    names = []
    for blk, j in full["columns"]:
        nm = _HELD_NAMES.get(blk)
        n_amb = len(nm) if nm else None
        free = [i for i in range(n_amb or (j + 1)) if i not in set(held.get(blk, ()))] if n_amb else list(range(j + 1))
        i_amb = free[j] if j < len(free) else j
        names.append(f"{blk}_{nm[i_amb]}" if nm and i_amb < len(nm) else f"{blk}[{j}]")
    return {"names": names, "cov": (np.asarray(full["cov"], float) * vf).tolist(), "variance_factor": vf}


def global_correlations(cov: Dict[str, Any], held: Dict[str, Any], min_abs: float = 0.8, top: int = 25
                        ) -> List[Dict[str, Any]]:
    """0.94.6: the strongest correlations between the global terms (from the exact-tie covariance): pairs with
    |r| >= ``min_abs``, strongest first - what the data cannot tell apart (e.g. a thermal slope and a drift in sol,
    when temperature follows the season)."""
    gc = global_covariance(cov, held)
    if not gc:
        return []
    C, names = np.asarray(gc["cov"], float), gc["names"]
    d = np.sqrt(np.clip(np.diag(C), 0, None))
    ok = d > 0
    R = np.zeros_like(C)
    R[np.ix_(ok, ok)] = C[np.ix_(ok, ok)] / np.outer(d[ok], d[ok])
    pairs = []
    for i in range(len(names)):
        for k in range(i + 1, len(names)):
            if ok[i] and ok[k] and abs(R[i, k]) >= min_abs:
                pairs.append({"a": names[i], "b": names[k], "r": float(R[i, k])})
    pairs.sort(key=lambda x: -abs(x["r"]))
    return pairs[:top]


def globals_result(st: Dict[str, Any], g: Dict[str, np.ndarray], ba: Dict[str, Any], units: Dict[str, Any]) -> Dict[str, Any]:
    """The global parameters with their sd, in the units of the model files (the sd from the problem's covariance
    of the global blocks, else propagated from the bin cameras' covariances, :func:`_sd_from_units`)."""
    cov = ba.get("covariance") or {}
    held = (ba.get("extra") or {}).get("held") or {}
    iso = bool((ba.get("extra") or {}).get("iso_f"))
    fallback = None
    out: Dict[str, Any] = {"T0_degC": st["T0"], "knot_sol": st["knot"], "early_sol": units["early_sol"],
                           "covariance_error": cov.get("error"), "covariance_damping": cov.get("damping"),
                           "covariance_method": cov.get("extra_method") or "colmap",
                           "covariance_exact": {k: v for k, v in (cov.get("extra_exact") or {}).items() if k != "full"},
                           "global_correlations": global_correlations(cov, held),
                           "global_covariance": global_covariance(cov, held),
                           "exact_refine": (ba.get("extra") or {}).get("exact_refine"),
                           "held": held, "ties": (ba.get("extra") or {}).get("ties"), "isotropic_f_slope": iso,
                           "pass_costs": (ba.get("extra") or {}).get("pass_costs")}
    for eye in ("NL", "NR"):
        G, S, E = g[f"G_{eye}"], g[f"S_{eye}"], g[f"E_{eye}"]
        sdG, sdS = _sd(cov, f"G_{eye}", 10, held.get(f"G_{eye}", ())), _sd(cov, f"S_{eye}", 5, held.get(f"S_{eye}", ()))
        sdE = _sd(cov, f"E_{eye}", 4, held.get(f"E_{eye}", ()))
        if iso and sdS is not None:
            sdS = sdS.copy(); sdS[1] = sdS[0] * G[1] / G[0]
        if sdS is None or not np.all(np.isfinite(sdS)):
            if fallback is None:
                fallback = _sd_from_units(cov, st, units, held, iso_f=iso)
            fb = fallback.get(eye)
            if fb is not None:
                sdG, sdS, sdE = fb["G"], fb["S"], fb["E"]
                out.setdefault("sd_source", {})[eye] = "propagated from the bin cameras' covariances"
        out[eye] = {"params_T0": dict(zip(TIED_PARAMS, map(float, G))),
                    "sd_params_T0": None if sdG is None else dict(zip(TIED_PARAMS, map(float, sdG))),
                    "fx_ppm_per_degC": float(1e6 * S[0] / G[0]), "fy_ppm_per_degC": float(1e6 * S[1] / G[1]),
                    "cx_px_per_degC": float(S[2]), "cy_px_per_degC": float(S[3]), "k1_per_degC": float(S[4]),
                    "sd": None if sdS is None else {"fx_ppm_per_degC": float(1e6 * sdS[0] / G[0]),
                                                    "fy_ppm_per_degC": float(1e6 * sdS[1] / G[1]),
                                                    "cx_px_per_degC": float(sdS[2]), "cy_px_per_degC": float(sdS[3]),
                                                    "k1_per_degC": float(sdS[4])},
                    "early_mission": {"before_sol": units["early_sol"], "offsets_px": dict(zip(EARLY_TERMS, map(float, E))),
                                      "sd_px": None if sdE is None else dict(zip(EARLY_TERMS, map(float, sdE)))}}
    R0 = R_of_q(g["Q0"])
    sdQ, sdD, sdH = (_sd(cov, "Q0", 3, held.get("Q0", ())), _sd(cov, "D", 3, held.get("D", ())),
                     _sd(cov, "H", 3, held.get("H", ())))
    mdeg_of_q = lambda v: None if v is None else dict(zip(ANGLES, map(float, 2.0 * np.degrees(v) * 1e3)))   # noqa: E731
    out["rig"] = {"R_sol0": R0.tolist(), "angles_sol0_mdeg": J.rig_angles(R0), "sd_sol0_mdeg": mdeg_of_q(sdQ),
                  "knot_sol": st["knot"], "t_m": list(map(float, st["t"])),
                  "rate_mdeg_per_1000sol": dict(zip(ANGLES, map(float, g["D"]))),
                  "sd_rate": None if sdD is None else dict(zip(ANGLES, map(float, sdD))),
                  "pitch_early_mdeg_per_1000sol": float(g["D"][0] + g["H"][0]),
                  "sd_pitch_early": None if sdH is None else float(np.hypot(sdD[0] if sdD is not None else 0.0, sdH[0]))}
    return out


def unit_rows(rec, proj, idx: Dict[str, Any], units: Dict[str, Any], st: Dict[str, Any], g: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """The units after an adjustment against the law: per camera unit the parameters and their difference from
    the law at the unit's temperature (fx, fy in ppm; cx, cy in px; k1); per rig unit the angles and their
    difference from the trend at the unit's sol (mdeg)."""
    stl = dict(st, G={"NL": g["G_NL"], "NR": g["G_NR"]}, S={"NL": g["S_NL"], "NR": g["S_NR"]},
               E={"NL": g["E_NL"], "NR": g["E_NR"]}, Q0=g["Q0"], D=g["D"], H=g["H"])
    cams = []
    for cid, u in units["cameras"].items():
        if cid not in rec.cameras:
            continue
        p = np.asarray(rec.cameras[cid].params, float)
        law = camera_params(stl, u["eye"], u["T"], u["early"])
        cams.append(dict(u, params=list(map(float, p)),
                         dfx_ppm=float(1e6 * (p[0] - law[0]) / law[0]), dfy_ppm=float(1e6 * (p[1] - law[1]) / law[1]),
                         dcx_px=float(p[2] - law[2]), dcy_px=float(p[3] - law[3]), dk1=float(p[4] - law[4])))
    rigs = []
    for rid, u in units["rigs"].items():
        if rid not in rec.rigs:
            continue
        rg = rec.rigs[rid]
        sid = next(iter(rg.non_ref_sensors), None)
        if sid is None:
            continue
        R = np.asarray(rg.sensor_from_rig(sid).rotation.matrix())
        R_law = R_of_q(rig_q_at(g["Q0"], g["D"], g["H"], u["sol"], st["knot"]))
        a = J.rig_angles(R, R_ref=R_law)
        rigs.append(dict(u, **{f"d{k}": v for k, v in a.items()}, **J.rig_angles(R)))
    return {"cameras": cams, "rigs": rigs}


def early_terms_to_hold(res: Dict[str, Any], z_min: float = EARLY_Z_MIN) -> Tuple[List[str], bool]:
    """The early-mission terms (``E_NL_fx`` ...) whose offset is not significant (|offset / sd| < ``z_min``) - judged
    only on the joint's own covariance (not the bin-camera fallback, which ignores the correlations and comes out
    3-4x small).  Returns (terms to hold at 0, judged)."""
    if not z_min or res.get("sd_source"):
        return [], False
    out = []
    for eye in ("NL", "NR"):
        em = res[eye]["early_mission"]
        sd = em.get("sd_px") or {}
        for k, v in (em.get("offsets_px") or {}).items():
            s_ = sd.get(k)
            if s_ is None or not np.isfinite(s_) or s_ <= 0:
                continue                                   # held already (no early units) or not estimable
            if abs(float(v) / float(s_)) < z_min:
                out.append(f"E_{eye}_{k}")
    return out, True


def fit_tied(scapes_cfg: Dict[str, PathLike], start_dir: Optional[PathLike] = None, points: int = POINTS_PER_BLOCK,
             bin_deg: float = BIN_DEG, max_span: float = MAX_SPAN_SOLS, early_sol: Optional[float] = None,
             knot_sol: Optional[float] = None, hold: Sequence[str] = HOLD_DEFAULT, iterations: int = 100,
             covariance: bool = True, passes: int = 2, diagnostics: bool = True, exclude: Sequence[str] = (),
             verbose: bool = True, iso_f: bool = ISO_F, early_z_min: float = EARLY_Z_MIN) -> Dict[str, Any]:
    """The one-adjustment Navcam joint over the blocks ``{name: WORK folder}`` (module docstring), then the
    diagnostic pass.  Returns the global terms with their sd (:func:`globals_result`), the residual statistics, the
    units of the joint and of the diagnostic pass (:func:`unit_rows`) - what :func:`write_tied` writes."""
    import pycolmap
    st = start_values(start_dir, knot_sol=knot_sol)
    cams = {k: pycolmap.Camera(model=st["cameras"][k]["model"], width=5120, height=3840,
                               params=np.asarray(st["cameras"][k]["params"], float)) for k in ("NL", "NR")}
    t = time.time()
    rec0, proj0, idx = J.merge_scapes((J.load_scape(n, scapes_cfg[n]) for n in scapes_cfg), cams, (st["R"], st["t"]),
                                      points_per_scape=points, seed=0, exclude=exclude)
    rec, proj, units = tied_reconstruction(rec0, proj0, idx, st, bin_deg=bin_deg, early_sol=early_sol, max_span=max_span)
    if verbose:
        log.info(f"tied joint: {len(scapes_cfg)} blocks, {rec.num_reg_images()} images, {rec.num_points3D()} points; "
                 f"{len(units['cameras'])} bin cameras, {len(units['rigs'])} rig units "
                 f"(epochs <= {max_span:g} sols, {bin_deg:g} degC bins)  ({time.time() - t:.0f} s)")
    t = time.time()
    ba, g, stats = adjust_tied(rec, proj, idx, st, units, hold=hold, iterations=iterations, covariance=covariance,
                               passes=passes, verbose=verbose, iso_f=iso_f)
    res = globals_result(st, g, ba, units)
    early_held, judged = early_terms_to_hold(res, early_z_min)
    res_first = None
    if early_held:
        # 0.94.5 (Christian 24 Oct): the offsets without significant evidence set to 0 and held, the joint again
        if verbose:
            log.info("tied joint: early-mission offsets not significant (|z| < %g), set to 0 and refitted: %s"
                     % (early_z_min, ", ".join(early_held)))
        res_first = {e: res[e]["early_mission"] for e in ("NL", "NR")}
        st = copy.deepcopy(st)
        for h in early_held:
            _, eye, k = h.split("_")
            st["E"][eye][EARLY_TERMS.index(k)] = 0.0
        hold = tuple(hold) + tuple(early_held)
        ba, g, stats = adjust_tied(rec, proj, idx, st, units, hold=hold, iterations=iterations, covariance=covariance,
                                   passes=passes, verbose=verbose, iso_f=iso_f)
        res = globals_result(st, g, ba, units)
    res["early_test"] = {"z_min": early_z_min, "judged": judged, "held": list(early_held), "first_fit": res_first}
    res.update({"blocks": list(scapes_cfg), "points_per_block": points, "bin_deg": bin_deg, "max_span_sols": max_span,
                "start_dir": st["folder"], "final_cost": float(ba["final_cost"]), "iterations": ba["iterations"],
                "brief": ba["brief"], "seconds": time.time() - t, "residuals": stats,
                "images": {s: len(v) for s, v in idx["images"].items()}, "hold": list(hold),
                "units": unit_rows(rec, proj, idx, units, st, g)})
    res["k1_significance"] = k1_significance(res)
    if verbose:
        for line in describe_result(res):
            log.info(line)
    if diagnostics:
        t = time.time()
        res["diagnostics"] = diagnostic_pass(rec0, proj0, idx, st, g, bin_deg=bin_deg, early_sol=early_sol,
                                             max_span=max_span, iterations=iterations, verbose=verbose)
        res["diagnostics"]["seconds"] = time.time() - t
        res["knot_comparison"] = knot_comparison(res)
        res["roll_profile"] = roll_profile(res)
        if verbose:
            for line in describe_rig_tests(res):
                log.info(line)
    return res


# ====================================================================================== after the BA: the rig tests
def _epoch_rows(res: Dict[str, Any], min_frames: int = 8) -> List[Dict[str, Any]]:
    """The diagnostic pass's rig epochs as :func:`mppp.calib.fit.fit_rig_drift` rows (absolute angles, a nominal
    sd of 1 mdeg at 100 images)."""
    rows = []
    for r in (res.get("diagnostics") or {}).get("rigs") or []:
        if r.get("frames", 0) < min_frames:
            continue
        sd = 1.0 * np.sqrt(100.0 / max(2 * r["frames"], 1))
        rows.append({"block": r["scape"], "epoch": r["epoch"], "sol": float(r["sol"]), "T": float(r["T"]), "images": 2 * r["frames"],
                     "sols": r.get("sols"), "pitch_mdeg": r["pitch_mdeg"], "yaw_mdeg": r["yaw_mdeg"], "roll_mdeg": r["roll_mdeg"],
                     "sd_pitch_mdeg": sd, "sd_yaw_mdeg": sd, "sd_roll_mdeg": sd})
    return rows


def knot_comparison(res: Dict[str, Any], knots: Sequence[float] = KNOTS_TESTED) -> Dict[str, Any]:
    """After the BA (Christian, 20 Oct): the pitch hinge refitted on the diagnostic pass's free epochs with the knot
    at each of ``knots`` and free, compared by the weighted residual sum of squares (Q2, with the between-epoch
    scatter tau) - which knot looks better."""
    from .fit import fit_rig_drift
    rows = _epoch_rows(res)
    out: Dict[str, Any] = {"epochs": len(rows), "knots": {}}
    if len(rows) < 5:
        out["note"] = "too few epochs"
        return out
    for k in list(knots) + ["free"]:
        try:
            d = fit_rig_drift(rows, knot_sol=k, knot_angles=("pitch",), no_drift=("yaw",))
        except Exception as e:                                        # noqa: BLE001
            out["knots"][str(k)] = {"error": f"{type(e).__name__}: {e}"}
            continue
        out["knots"][str(k)] = {"knot_sol": d.get("knot_sol"), "knot_sol_range": d.get("knot_sol_range"),
                                "pitch_Q2": d.get("pitch_Q2"), "pitch_tau_mdeg": d.get("pitch_tau_mdeg"),
                                "pitch_mdeg_per_1000sol": 1e3 * float(d.get("pitch_mdeg_per_sol") or 0.0),
                                "pitch_early_mdeg_per_1000sol": 1e3 * float(d.get("pitch_early_mdeg_per_sol") or 0.0)}
    fixed = [(k, v["pitch_Q2"]) for k, v in out["knots"].items() if k != "free" and v.get("pitch_Q2") is not None]
    if fixed:
        out["best_fixed"] = min(fixed, key=lambda t: t[1])[0]
    return out


def roll_profile(res: Dict[str, Any], step_sol: float = 900.0) -> Dict[str, Any]:
    """Is the roll linear in sol or a step near ``step_sol``?  (Christian, 20 Oct.)  Three weighted fits on the
    diagnostic pass's epochs (with the temperature term): linear, step, linear + step; Q2 and tau of each, the
    step size with its se, and which model the residual sum of squares prefers (Q2 + 2 x parameters)."""
    from .fit import wls_tau
    rows = _epoch_rows(res)
    if len(rows) < 5:
        return {"epochs": len(rows), "note": "too few epochs"}
    y = np.array([r["roll_mdeg"] for r in rows]); T = np.array([r["T"] for r in rows]); S = np.array([r["sol"] for r in rows])
    v = np.array([r["sd_roll_mdeg"] for r in rows]) ** 2
    one = np.ones(len(rows))
    models = {"linear": np.c_[one, T - T.mean(), S - S.mean()],
              "step": np.c_[one, T - T.mean(), (S > step_sol).astype(float)],
              "linear+step": np.c_[one, T - T.mean(), S - S.mean(), (S > step_sol).astype(float)]}
    out: Dict[str, Any] = {"epochs": len(rows), "step_sol": step_sol, "before": int((S <= step_sol).sum()),
                           "after": int((S > step_sol).sum()), "models": {}}
    for name, X in models.items():
        f = wls_tau(y, X, v)
        m = {"Q2": f["Q2"], "tau_mdeg": f["tau"], "dof": f["dof"], "score": f["Q2"] + 2 * X.shape[1]}
        if "linear" in name:
            m["rate_mdeg_per_1000sol"] = 1e3 * f["coef"][2]; m["rate_se"] = 1e3 * f["se"][2]
        if "step" in name:
            m["step_mdeg"] = f["coef"][-1]; m["step_se"] = f["se"][-1]; m["step_z"] = f["coef"][-1] / f["se"][-1] if f["se"][-1] else None
        out["models"][name] = m
    out["preferred"] = min(out["models"], key=lambda k: out["models"][k]["score"])
    return out


def describe_rig_tests(res: Dict[str, Any]) -> List[str]:
    lines = []
    kc = res.get("knot_comparison") or {}
    if kc.get("knots"):
        ok = {k: v for k, v in kc["knots"].items() if "error" not in v and v.get("pitch_Q2") is not None}
        fixed = {k: v for k, v in ok.items() if k != "free"}
        line = "pitch knot after the BA (diagnostic epochs), Q2 by knot: " + ", ".join(
            f"{float(k):g} {v['pitch_Q2']:.1f}" for k, v in fixed.items())
        b = kc.get("best_fixed")
        if b in ok:
            v = ok[b]
            q = v["pitch_Q2"]
            near = [float(k) for k, w in fixed.items() if w["pitch_Q2"] <= q + 1.0]
            line += (f" -> best fixed knot {float(b):g} (Q2 {q:.1f}, tau {v.get('pitch_tau_mdeg', float('nan')):.2f} mdeg, early "
                     f"{v.get('pitch_early_mdeg_per_1000sol', float('nan')):+.1f} / late {v.get('pitch_mdeg_per_1000sol', float('nan')):+.1f}"
                     f" mdeg/1000 sol; within Q2 + 1: {min(near):g}-{max(near):g})")
        if "free" in ok:
            v = ok["free"]
            line += f"; free knot {v.get('knot_sol')} (range {v.get('knot_sol_range')}, Q2 {v['pitch_Q2']:.1f})"
        lines.append(line)
    rp = res.get("roll_profile") or {}
    if rp.get("models"):
        m = rp["models"]
        lines.append(f"roll in sol ({rp['before']} epochs before sol {rp['step_sol']:g}, {rp['after']} after): linear "
                     f"{m['linear']['rate_mdeg_per_1000sol']:+.2f} +- {m['linear']['rate_se']:.2f} mdeg/1000 sol (Q2 {m['linear']['Q2']:.1f}); "
                     f"step {m['step']['step_mdeg']:+.2f} +- {m['step']['step_se']:.2f} mdeg (Q2 {m['step']['Q2']:.1f}); "
                     f"linear+step: step {m['linear+step']['step_mdeg']:+.2f} +- {m['linear+step']['step_se']:.2f}, rate "
                     f"{m['linear+step']['rate_mdeg_per_1000sol']:+.2f} (Q2 {m['linear+step']['Q2']:.1f}) -> preferred: {rp['preferred']}")
    return lines


def diagnostic_pass(rec0, proj0, idx: Dict[str, Any], st: Dict[str, Any], g: Dict[str, np.ndarray], bin_deg: float = BIN_DEG,
                    early_sol: Optional[float] = None, max_span: float = MAX_SPAN_SOLS, iterations: int = 60,
                    verbose: bool = True, pointing: bool = True) -> Dict[str, Any]:
    """With the global solution held (then :func:`pointing_pass` for the principal points): one camera per (block, eye, bin, era) and one rig per (block, epoch, bin, era),
    each started at the law and refined freely (no ties) in one adjustment with the poses and points.  Returns the
    units' values against the law (:func:`unit_rows`) - the per-block, per-bin and per-sol scatter about the
    global solution that the figures draw."""
    stl = dict(st, G={"NL": g["G_NL"], "NR": g["G_NR"]}, S={"NL": g["S_NL"], "NR": g["S_NR"]},
               E={"NL": g["E_NL"], "NR": g["E_NR"]}, Q0=g["Q0"], D=g["D"], H=g["H"])
    rec, proj, units = tied_reconstruction(rec0, proj0, idx, stl, bin_deg=bin_deg, early_sol=early_sol,
                                           max_span=max_span, per_block=True, merge_min_images=DIAG_BIN_MIN_IMAGES)
    ba, g2, stats = adjust_tied(rec, proj, idx, stl, units, hold=(), tie_rigs=False, tie_cameras=False,
                                iterations=iterations, covariance=False, passes=1, verbose=verbose, refine_pp=False)
    rows = unit_rows(rec, proj, idx, units, stl, g)
    if verbose:
        c = [r for r in rows["cameras"] if r["images"] >= 8]
        r_ = [r for r in rows["rigs"] if r["frames"] >= 8]
        log.info(f"diagnostic pass: {len(units['cameras'])} block-bin cameras, {len(units['rigs'])} rig units; "
                 f"median |dfx| {np.median([abs(x['dfx_ppm']) for x in c]) if c else float('nan'):.0f} ppm, "
                 f"|dpitch| {np.median([abs(x['dpitch_mdeg']) for x in r_]) if r_ else float('nan'):.1f} mdeg "
                 f"(units with >= 8 images / frames)")
    out = {"cameras": rows["cameras"], "rigs": rows["rigs"], "residuals": stats, "brief": ba["brief"],
           "final_cost": float(ba["final_cost"])}
    if pointing:
        out["pointing"] = pointing_pass(rec0, proj0, idx, stl, g, early_sol=early_sol, max_span=max_span,
                                        iterations=iterations, verbose=verbose)
    return out


PP_PRIOR_PX = {"NL": 1.0}        # NL's principal point slides with the frames' attitudes: a weak prior about the law


def pointing_pass(rec0, proj0, idx: Dict[str, Any], stl: Dict[str, Any], g: Dict[str, np.ndarray],
                  early_sol: Optional[float] = None, max_span: float = MAX_SPAN_SOLS, iterations: int = 60,
                  pp_prior: Optional[Dict[str, float]] = None, verbose: bool = True) -> Dict[str, Any]:
    """The principal points per block, without fitting noise (20 Oct: free bin principal points with free epoch
    rigs were degenerate - NR cx with the rig yaw, 1 px = 18.9 mdeg; cy with the pitch):

    * one camera per (block, eye, era), not per 2 degC bin: the principal point moves 0.3 px over the bins'
      20 degC, so a block measures it once; the images' own temperatures still count through the law's slopes
      (keypoint map);
    * fx, fy, cx, cy free, the distortion held at the law;
    * every rig epoch held at the global trend (tied with the globals constant): NR's principal point against NL's
      is then the stereo pointing itself, with nothing to trade against;
    * NL's absolute principal point still slides with the frames' attitudes (the label priors are 5 degrees): a
      weak prior about the law (``pp_prior``, default 1 px) keeps it from wandering, and its sd says how much the
      data itself determines it;
    * sd from the covariance of the adjustment, per unit.

    Returns rows {block, eye, era, T, images, dcx_px, dcy_px, sd_cx_px, sd_cy_px, dfx_ppm, dfy_ppm, cx, cy} and the
    stereo rows {block, era, T, dcx_stereo_px (NR - NL x fR/fL about the law), sd}."""
    rec, proj, units = tied_reconstruction(rec0, proj0, idx, stl, bin_deg=BIN_DEG, early_sol=early_sol,
                                           max_span=max_span, per_block=True, pointing=True)
    ba, _g, stats = adjust_tied(rec, proj, idx, stl, units, hold=("Q0", "D", "H"), tie_rigs=True, tie_cameras=False,
                                iterations=iterations, covariance=True, passes=1, verbose=verbose,
                                pp_prior=dict(PP_PRIOR_PX if pp_prior is None else pp_prior))
    rows = unit_rows(rec, proj, idx, units, stl, g)["cameras"]
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor", 1.0))
    for r in rows:
        c = (cov.get("blocks") or {}).get(f"camera:{r['key']}")
        d = np.sqrt(np.clip(np.diag(np.asarray(c)) * vf, 0, None)) if c is not None else None
        r["sd_cx_px"] = float(d[2]) if d is not None and len(d) > 3 else None
        r["sd_cy_px"] = float(d[3]) if d is not None and len(d) > 3 else None
    stereo = []
    for (scape, early) in sorted({(r["scape"], r["early"]) for r in rows}):
        L = next((r for r in rows if r["scape"] == scape and r["early"] == early and r["eye"] == "NL"), None)
        R = next((r for r in rows if r["scape"] == scape and r["early"] == early and r["eye"] == "NR"), None)
        if not (L and R):
            continue
        ratio = R["params"][0] / L["params"][0]
        sd = None
        if L.get("sd_cx_px") is not None and R.get("sd_cx_px") is not None:
            sd = float(np.hypot(R["sd_cx_px"], ratio * L["sd_cx_px"]))
        stereo.append({"block": scape, "early": early, "T": 0.5 * (L["T"] + R["T"]), "images": L["images"] + R["images"],
                       "dcx_stereo_px": R["dcx_px"] - ratio * L["dcx_px"], "dcy_stereo_px": R["dcy_px"] - ratio * L["dcy_px"],
                       "sd_px": sd})
    if verbose:
        v = [abs(x["dcx_stereo_px"]) for x in stereo]
        log.info(f"pointing pass: {len(rows)} block cameras, rigs at the trend; stereo cx about the law: median "
                 f"|{np.median(v) if v else float('nan'):.3f}| px over {len(stereo)} block-eras")
    return {"cameras": rows, "stereo": stereo, "residuals": stats, "brief": ba["brief"],
            "covariance_error": cov.get("error"), "pp_prior_px": dict(PP_PRIOR_PX if pp_prior is None else pp_prior)}


def k1_significance(res: Dict[str, Any], theta_corner: float = 1.05) -> Dict[str, Dict[str, float]]:
    """Is k1's thermal slope significant?  Per eye: the slope over its sd (z) and what it does to the image corner
    (fisheye: theta_d = theta (1 + k1 theta^2 ...), so dk1 moves a ray at theta by f dk1 theta^3 px) over the span of
    the bin temperatures.  ``theta_corner``: the Navcam's corner field angle, ~60 deg."""
    out = {}
    cams = ((res.get("units") or {}).get("cameras")) or []
    for eye in ("NL", "NR"):
        o = res.get(eye) or {}
        k = o.get("k1_per_degC")
        if k is None:
            continue
        sd = float((o.get("sd") or {}).get("k1_per_degC", float("nan")))
        Ts = [float(c["T"]) for c in cams if c.get("eye") == eye]
        span = (max(Ts) - min(Ts)) if Ts else float("nan")
        f = float((o.get("params_T0") or {}).get("fx", float("nan")))
        out[eye] = {"k1_per_degC": float(k), "sd": sd, "z": float(k / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan"),
                    "span_degC": float(span), "theta_corner_rad": theta_corner,
                    "corner_px": float(abs(k) * span * f * theta_corner ** 3)}
    return out


def describe_result(res: Dict[str, Any]) -> List[str]:
    lines = []
    for eye in ("NL", "NR"):
        o = res[eye]
        sd = o["sd"] or {}
        nan = float("nan")
        lines.append(f"{eye}: fx {o['fx_ppm_per_degC']:+.2f} +- {sd.get('fx_ppm_per_degC', nan):.2f}, fy "
                     f"{o['fy_ppm_per_degC']:+.2f} +- {sd.get('fy_ppm_per_degC', nan):.2f} ppm/degC; cx "
                     f"{o['cx_px_per_degC']:+.4f} +- {sd.get('cx_px_per_degC', nan):.4f}, cy {o['cy_px_per_degC']:+.4f} "
                     f"+- {sd.get('cy_px_per_degC', nan):.4f} px/degC; k1 {o['k1_per_degC']:+.2e}/degC; early offsets "
                     + ", ".join(f"{k} {v:+.3f}" for k, v in o["early_mission"]["offsets_px"].items()) + " px")
    if res.get("isotropic_f_slope"):
        lines.append("focal slope isotropic: fy's ppm/degC tied to fx's (ISO_F; --aniso-f frees it)")
    for eye, k in k1_significance(res).items():
        lines.append(f"{eye} k1(T): {k['k1_per_degC']:+.2e} +- {k['sd']:.1e}/degC (z {k['z']:+.1f}); over the bins' "
                     f"{k['span_degC']:.0f} degC it moves the corner (theta {k['theta_corner_rad']:.2f} rad) by "
                     f"{k['corner_px']:.2f} px")
    r = res["rig"]
    lines.append(f"rig at sol 0: {r['angles_sol0_mdeg']}; rates mdeg/1000 sol {r['rate_mdeg_per_1000sol']} "
                 f"(pitch before sol {r['knot_sol']:g}: {r['pitch_early_mdeg_per_1000sol']:+.2f})")
    s = res.get("residuals") or {}
    lines.append(f"tied joint: cost {res.get('final_cost', float('nan')):.1f}, it {res.get('iterations')}, median "
                 f"{s.get('median_px', float('nan')):.4f} / rms {s.get('rms_px', float('nan')):.4f} px, corners "
                 f"{s.get('corner_median_px', float('nan')):.3f} px ({res.get('seconds', 0):.0f} s); covariance "
                 f"{covariance_line(res)}")
    for c in res.get("global_correlations") or []:
        lines.append(f"  correlated: {c['a']} ~ {c['b']} r {c['r']:+.3f}")
    return lines


def covariance_line(res: Dict[str, Any]) -> str:
    """0.94.6: what the covariance is - its method, whether the global terms got their own finite sd (the log said
    "covariance ok" on 24 and 25 Oct while every rig sd was infinite)."""
    if res.get("covariance_error"):
        return f"FAILED ({res['covariance_error']})"
    bad = [k for k, v in rig_sd_free(res).items() if v is None or not np.isfinite(v)]
    src = res.get("sd_source")
    ex = res.get("covariance_exact") or {}
    cond = (ex.get("dims") or {}).get("condition")
    txt = f"{res.get('covariance_method', 'colmap')}" + (f", scaled condition {cond:.2g}" if cond else "")
    if ex.get("error"):
        txt += f" (exact ties failed: {ex['error']})"
    if bad or src:
        return txt + "; NOT USABLE: " + "; ".join(([f"rig sd not finite: {', '.join(bad)}"] if bad else []) +
                                                 ([f"camera sd {', '.join(sorted(src))} from the bin cameras"] if src else []))
    return txt + "; ok"


# ====================================================================================== the candidate files
def write_tied(res: Dict[str, Any], out_dir: PathLike, note: str = "", template_dir: Optional[PathLike] = None) -> Path:
    """The candidate folder of a :func:`fit_tied` result in the frozen form: the camera files (parameters at T0, the
    thermal terms, the early-mission offsets of each eye, sd from the covariance) and the rig file (rotation at
    sol 0, ``drift`` with ``sol0 = 0``, the three rates and the pitch hinge); ``tied_joint.json`` with everything."""
    tdir = M._folder(template_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = sorted(res["blocks"])
    stamp = datetime.date.today().isoformat()
    for eye in ("NL", "NR"):
        c = M._read(M.camera_file("N", eye[1], tdir))
        o = res[eye]
        p = o["params_T0"]
        c["params"] = [float(p.get(n, 0.0)) for n in c["param_names"]]
        c["fixed_params"] = []
        sd = o["sd"] or {}
        c["source"] = (f"MPPP Navcam joint, one adjustment (mppp.calib.tied.fit_tied, {stamp}): one {eye} camera "
                       f"shared by {len(blocks)} blocks ({', '.join(blocks)}); fisheye + tangential at {res['T0_degC']:g} "
                       f"degC, fx {o['fx_ppm_per_degC']:+.1f} / fy {o['fy_ppm_per_degC']:+.1f} ppm/degC, cx "
                       f"{o['cx_px_per_degC']:+.4f} px/degC, k1 {o['k1_per_degC']:+.2e}/degC, early offsets per eye; "
                       f"every term with the rig trend and the block poses in one bundle adjustment. {note}".strip())
        c["thermal"] = {"T0_degC": float(res["T0_degC"]), "fx_ppm_per_degC": o["fx_ppm_per_degC"],
                        "fy_ppm_per_degC": o["fy_ppm_per_degC"], "cx_px_per_degC": o["cx_px_per_degC"],
                        "cy_px_per_degC": o["cy_px_per_degC"], "k1_per_degC": o["k1_per_degC"],
                        "k1_source": "tied joint", "note": f"one adjustment of every term ({stamp})"}
        for k in ("fx_ppm_per_degC", "fy_ppm_per_degC", "cx_px_per_degC", "cy_px_per_degC", "k1_per_degC"):
            if sd.get(k) is not None and np.isfinite(sd[k]):
                c["thermal"]["sd_" + k] = float(sd[k])
        em = o["early_mission"]
        # the joint fitted all four offsets of each eye: projects apply all four (constrain.navcam_camera reads
        # applied_terms - without it the offsets were silently dropped, 20 Oct)
        c["early_mission"] = {"before_sol": float(em["before_sol"]), "per_eye": True,
                              **{f"d{k}_px": float(v) for k, v in em["offsets_px"].items()},
                              "applied_terms": [k for k in EARLY_TERMS if k in em["offsets_px"]
                                                and f"E_{eye}_{k}" not in set((res.get("early_test") or {}).get("held") or ())],
                              "note": "tied joint: the offsets of this eye for the images before before_sol (weighted by "
                                      "a block's share of early images); 0.94.5: kept only where |z| >= "
                                      f"{(res.get('early_test') or {}).get('z_min', EARLY_Z_MIN):g}, the others 0 and held"}
        if em.get("sd_px"):
            c["early_mission"]["sd_px"] = {k: float(v) for k, v in em["sd_px"].items() if np.isfinite(v)}
            c["early_mission"]["z"] = {k: float(em["offsets_px"][k] / v) for k, v in em["sd_px"].items()
                                       if np.isfinite(v) and v > 0 and k in em["offsets_px"]}
        if o.get("sd_params_T0"):
            c["sd_params"] = {k: float(v) for k, v in o["sd_params_T0"].items() if np.isfinite(v)}
        c["fit"] = {"method": "tied joint", "blocks": blocks, "date": stamp, "residuals": res.get("residuals"),
                    "covariance_error": res.get("covariance_error")}
        M.write_model(out / M.camera_file("N", eye[1], tdir).name, c)
    rj = M._read(M.rig_file("N", tdir))
    r = res["rig"]
    rj["R_sensor_from_ref"] = r["R_sol0"]
    rj["t_sensor_from_ref_m"] = list(map(float, r["t_m"]))
    rates = r["rate_mdeg_per_1000sol"]
    rj["drift"] = {"model": "linear+hinge", "sol0": 0.0, "knot_sol": float(r["knot_sol"]), "knot_ref": 0.0,
                   "pitch_mdeg_per_sol": rates["pitch"] / 1000.0, "yaw_mdeg_per_sol": rates["yaw"] / 1000.0,
                   "roll_mdeg_per_sol": rates["roll"] / 1000.0,
                   "pitch_early_mdeg_per_sol": float(r["pitch_early_mdeg_per_1000sol"]) / 1000.0,
                   "hinge_angles": ["pitch"], "blocks": len(blocks), "blocks_names": blocks,
                   "source": f"tied joint ({stamp}): yaw and roll linear in sol, pitch with the early hinge, all "
                             f"fitted with the cameras in one adjustment; sd mdeg/1000 sol {r.get('sd_rate')}"}
    a = r["angles_sol0_mdeg"]
    rj["note"] = (f"x_NR = R x_NL + t. Navcam tied joint {stamp} ({len(blocks)} blocks): rotation at sol 0, yaw "
                  f"{a['yaw_mdeg']:+.2f}, pitch {a['pitch_mdeg']:+.2f}, roll {a['roll_mdeg']:+.2f} mdeg; rates "
                  f"pitch {rates['pitch']:+.2f}, yaw {rates['yaw']:+.2f}, roll {rates['roll']:+.2f} mdeg/1000 sol "
                  f"(pitch before sol {r['knot_sol']:g}: {r['pitch_early_mdeg_per_1000sol']:+.2f}). Translation held. {note}".strip())
    rj["source"] = f"MPPP Navcam tied joint (mppp.calib.tied.fit_tied, {stamp})"
    M.write_model(out / M.rig_file("N", tdir).name, rj)
    (out / "tied_joint.json").write_text(json.dumps(res, indent=1, default=_json_default), encoding="utf-8")
    fi = figure_inputs(res)
    cons = dict(consensus_record(res), **{k: fi["consensus"][k] for k in ("NL", "NR")})
    cons["thermal"] = dict(cons["thermal"], **fi["consensus"]["thermal"])
    (out / "consensus.json").write_text(json.dumps(cons, indent=1, default=_json_default), encoding="utf-8")
    # the diagnostic pass as the figures' inputs (paper_figures.py reads <study>/rig_residuals.json next to navcam_joint/)
    for folder in {out, out.parent}:
        (folder / "rig_residuals.json").write_text(json.dumps(fi["rig_residuals"], indent=1, default=_json_default), encoding="utf-8")
        (folder / "thermal_bins.json").write_text(json.dumps(fi["bins"], indent=1, default=_json_default), encoding="utf-8")
        (folder / "pp_units.json").write_text(json.dumps(fi["pp_units"], indent=1, default=_json_default), encoding="utf-8")
    return out


# the limits from the first real run (navcal_tied_20261020_2: |dfx| p95 196 ppm, |dfy| p95 456 ppm, rig p95 7.7 mdeg;
# Christian, 20 Oct): fx 250 ppm, fy 500 ppm, rig 8 mdeg, at most 10 % of the units off
BLOCK_AGREEMENT = {"f_ppm": 250.0, "fy_ppm": 500.0, "rig_mdeg": 8.0, "min_images": 8, "max_share": 0.10}


def block_agreement(res: Dict[str, Any], limits: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """The promotion gate of the tied joint in place of the leave-one-block-out: in the diagnostic pass (every
    block's bins and epochs free about the held global solution) the share of block-bin cameras with >= min_images
    images whose fx differs from the law by more than ``f_ppm`` or fy by more than ``fy_ppm``, and of rig epochs (>= min_images frames) more
    than ``rig_mdeg`` from the trend, must be at most ``max_share`` each."""
    lim = dict(BLOCK_AGREEMENT, **(limits or {}))
    d = res.get("diagnostics") or {}
    cams = [c for c in d.get("cameras") or [] if c.get("images", 0) >= lim["min_images"]]
    rigs = [r for r in d.get("rigs") or [] if r.get("frames", 0) >= lim["min_images"]]
    fy_lim = float(lim.get("fy_ppm", lim["f_ppm"]))
    bad_c = [c for c in cams if abs(c["dfx_ppm"]) > lim["f_ppm"] or abs(c["dfy_ppm"]) > fy_lim]
    bad_r = [r for r in rigs if max(abs(r["dpitch_mdeg"]), abs(r["droll_mdeg"]), abs(r["dyaw_mdeg"])) > lim["rig_mdeg"]]
    sc = len(bad_c) / len(cams) if cams else float("nan")
    sr = len(bad_r) / len(rigs) if rigs else float("nan")
    ok = bool(cams) and bool(rigs) and sc <= lim["max_share"] and sr <= lim["max_share"]
    return {"ok": ok, "limits": lim, "cameras": len(cams), "cameras_off": len(bad_c), "share_cameras_off": sc,
            "rigs": len(rigs), "rigs_off": len(bad_r), "share_rigs_off": sr,
            "worst_camera_ppm": max((max(abs(c["dfx_ppm"]), abs(c["dfy_ppm"])) for c in cams), default=float("nan")),
            "worst_rig_mdeg": max((max(abs(r["dpitch_mdeg"]), abs(r["droll_mdeg"]), abs(r["dyaw_mdeg"])) for r in rigs),
                                  default=float("nan")),
            "blocks_off": sorted({c["scape"] for c in bad_c} | {r["scape"] for r in bad_r})}


def rig_sd_free(res: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The sd of the rig's free global terms (Q0 pitch / roll, the rates, the early pitch; held terms left out):
    {name: sd} - a non-finite one means the joint's covariance is not usable (24 Oct: pitch and yaw rates inf)."""
    r = res.get("rig") or {}
    held = res.get("held") or {}
    out: Dict[str, Optional[float]] = {}
    for i, a in enumerate(ANGLES):
        if i not in set(held.get("Q0") or ()):
            out[f"Q0_{a}"] = (r.get("sd_sol0_mdeg") or {}).get(a)
        if i not in set(held.get("D") or ()):
            out[f"D_{a}"] = (r.get("sd_rate") or {}).get(a)
    if 0 not in set(held.get("H") or ()):
        out["H_pitch"] = r.get("sd_pitch_early")
    return {k: (None if v is None else float(v)) for k, v in out.items()}


def consensus_record(res: Dict[str, Any]) -> Dict[str, Any]:
    """The ``consensus.json`` of the candidate folder in the shape :func:`mppp.calib.fit.check_navcam` reads."""
    sd = {}
    for eye in ("NL", "NR"):
        s = res[eye].get("sd_params_T0")
        sd[eye] = {k: float(v) for k, v in (s or {}).items()} if s else {"missing": True}
    em = {"converged": True, "per_eye": True, "rounds": [{"positive_definite": res.get("covariance_error") is None}],
          "offsets_px": {}, "sd_px": {}, "z": {}, "applied_terms": []}
    for eye in ("NL", "NR"):
        e = res[eye]["early_mission"]
        held_e = set((res.get("early_test") or {}).get("held") or ())
        for k, v in e["offsets_px"].items():
            em["offsets_px"][f"{eye}_{k}"] = float(v)
            if f"E_{eye}_{k}" not in held_e:
                em["applied_terms"].append(f"{eye}_{k}")
            s = (e.get("sd_px") or {}).get(k)
            if s is not None and np.isfinite(s) and s > 0:
                em["sd_px"][f"{eye}_{k}"] = float(s)
                em["z"][f"{eye}_{k}"] = float(v / s)
    r = res["rig"]
    return {"method": "tied", "blocks": {b: res.get("images", {}).get(b) for b in res["blocks"]},
            "T0_degC": res["T0_degC"], "sd": sd, "iterations": res.get("iterations"), "final_cost": res.get("final_cost"),
            "residuals": res.get("residuals"), "covariance_error": res.get("covariance_error"),
            "covariance_damping": res.get("covariance_damping"), "early_mission": em,
            "thermal": {"yaw_rate_mdeg_per_1000sol": r["rate_mdeg_per_1000sol"]["yaw"],
                        "sd": {"yaw_rate": (r.get("sd_rate") or {}).get("yaw")}},
            "sd_source": res.get("sd_source"), "rig_sd": rig_sd_free(res), "pass_costs": res.get("pass_costs"),
            "block_agreement": block_agreement(res) if res.get("diagnostics") else None,
            "rig_abs_mdeg": r["angles_sol0_mdeg"], "hold": res.get("hold"), "seconds": res.get("seconds")}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o).__name__)


# ====================================================================================== the figures
def figure_inputs(res: Dict[str, Any], min_images: int = 8, bin_min_images: Optional[int] = None) -> Dict[str, Any]:
    """What the refined figures take, from a :func:`fit_tied` result with its diagnostic pass: ``consensus`` (the
    global solution in the shape of the profile joint's ``consensus.json``: ``NL``/``NR`` params, ``thermal.axis_ppm``
    and ``pp_slopes``), ``rig_residuals`` (the diagnostic rig epochs about the trend: ``blocks`` rows with
    yaw/pitch/roll_mdeg, ``fits`` vs T and sol) and ``bins`` (the diagnostic block-bin cameras {eye, block, T, n,
    fx, fy, cx, cy, k1}; >= ``bin_min_images`` images, default the smaller of :data:`FIG_BIN_MIN_IMAGES` and
    ``min_images``; the rig epochs and pointing units >= ``min_images``) - the
    per-block, per-bin and per-sol values drawn about the one solution."""
    from .fit import ols
    from ..waypoints.sites import site_label
    cons: Dict[str, Any] = {"method": "tied", "T0_degC": res["T0_degC"], "blocks": {b: None for b in res["blocks"]}}
    for eye in ("NL", "NR"):
        cons[eye] = dict(res[eye]["params_T0"])
    cons["thermal"] = {"axis_ppm": {e: [res[e]["fx_ppm_per_degC"], res[e]["fy_ppm_per_degC"]] for e in ("NL", "NR")},
                       "pp_slopes": {e: [res[e]["cx_px_per_degC"], res[e]["cy_px_per_degC"]] for e in ("NL", "NR")},
                       "k1_slopes": {e: res[e]["k1_per_degC"] for e in ("NL", "NR")},
                       "yaw_rate_mdeg_per_1000sol": res["rig"]["rate_mdeg_per_1000sol"]["yaw"]}
    d = res.get("diagnostics") or {}
    rows = []
    for r in d.get("rigs") or []:
        if r.get("frames", 0) < min_images or not r.get("scape"):
            continue
        lab = site_label(r["scape"])
        rows.append({"block": f"{lab} [sols {r['sols'][0]}-{r['sols'][1]}]" if r.get("sols") else lab, "site": lab,
                     "T_median_degC": r["T"], "sol_median": r["sol"], "images": 2 * r["frames"],
                     "yaw_mdeg": r["dyaw_mdeg"], "pitch_mdeg": r["dpitch_mdeg"], "roll_mdeg": r["droll_mdeg"]})
    fits = {}
    for ang in ("yaw", "pitch", "roll"):
        y = np.array([r[f"{ang}_mdeg"] for r in rows], float)
        T = np.array([r["T_median_degC"] for r in rows], float)
        S = np.array([r["sol_median"] for r in rows], float)
        vt, vs = ols(T, y), ols(S / 1000.0, y)
        for f in (vt, vs):                                     # too few epochs: no slope (0 +- nan) for the figure
            if f.get("slope") is None:
                f.update({"slope": 0.0, "se": float("nan"), "z": float("nan")})
        fits[ang] = {"mean_mdeg": float(np.mean(y)) if len(y) else None, "sd_mdeg": float(np.std(y)) if len(y) > 1 else None,
                     "vs_T": vt, "vs_sol": vs}
    rr = {"blocks": rows, "fits": fits, "source": "tied joint diagnostic pass (globals held, epochs free)"}
    bins = []
    bin_min_images = min(min_images, FIG_BIN_MIN_IMAGES) if bin_min_images is None else bin_min_images
    for c in d.get("cameras") or []:
        if c.get("images", 0) < bin_min_images or not c.get("scape"):
            continue
        p = c["params"]
        bins.append({"eye": c["eye"], "block": site_label(c["scape"]), "key": c["key"], "T": c["T"], "n": c["images"],
                     "fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3], "k1": p[4]})
    pp_rows = []
    for c in (d.get("pointing") or {}).get("cameras") or []:
        if c.get("images", 0) < min_images:
            continue
        pp_rows.append({"eye": c["eye"], "block": site_label(c["scape"]), "T": c["T"], "n": c["images"], "early": c["early"],
                        "dcx": c["dcx_px"], "dcy": c["dcy_px"], "sd_cx": c.get("sd_cx_px"), "sd_cy": c.get("sd_cy_px")})
    return {"consensus": cons, "rig_residuals": rr, "bins": bins, "pp_units": pp_rows,
            "stereo": (d.get("pointing") or {}).get("stereo") or []}
