"""
The tied Navcam joint (MPPP v0p83, 18 Oct 2026; Christian: "shrink the navcam blocks used so that we can bundle adjust
the distortion params, temp and sol at once; split the rigs for ranges greater than 25 sols").

One bundle adjustment over (thinned) blocks in which every term of the camera form is a parameter, instead of the
one-at-a-time thermal profiles and the early-mission response surface of :func:`mppp.sfm.navcal_consensus.
fit_consensus` (~110 sequential adjustments, 8 h for 23 blocks, slopes still moving after two rounds):

* each eye gets one COLMAP camera per (temperature bin, era): ``bin_deg`` bins of the image temperature, era =
  before / after ``early_sol`` (the early-mission offsets).  The cameras are refined freely by the adjustment and tied
  (soft constraints of ``TIE_SIGMA``, effectively exact) to global parameters:

      fx = G.fx + S.fx (T - T0) + early * E.fx      (likewise fy, cx, cy; E common to both eyes)
      k1 = G.k1 + S.k1 (T - T0)
      k2, p1, p2, k3, k4 = G                        (one distortion for every bin; sx1, sy1 held at 0)

  with T the median temperature of the bin camera's images - so the focal-length and principal-point slopes of both
  eyes (the athermal rig: both principal points), k1(T), the distortion and the early offsets are estimated together,
  with their correlations and covariance;
* each block gets one stereo rig per (rig epoch, bin, era); epochs are the block's sol clusters cut at the pitch knot
  and split at the middle sol until none spans more than ``max_span`` sols (25).  The rig rotations are tied to one
  global rig and its drift:  q_xyz(rig) = Q0 + 1/2 [D_pre (min(s, knot) - knot) + D_post (max(s, knot) - knot)]
  (s = the rig unit's median sol; D in mdeg of pitch, yaw, roll per 1000 sols, the rotation-vector components of
  :func:`mppp.sfm.navcal.rig_angles`); no temperature term (the athermal rig).  The translation (baseline) is held.

``tie_rigs=False`` leaves the rig units free instead (one rig per epoch, the rig-residual analysis in the same solve).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

TIED_PARAMS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4")     # THIN_PRISM_FISHEYE indices 0-9
SLOPE_TERMS = ("fx", "fy", "cx", "cy", "k1")                                  # per degC (px; k1 unitless)
EARLY_TERMS = ("fx", "fy", "cx", "cy")                                        # px, both eyes
TIE_SIGMA = {"fx": 1e-3, "fy": 1e-3, "cx": 1e-3, "cy": 1e-3, "k": 1e-7, "q": 1e-8}
BIN_DEG = 2.0                    # fine bins; within a bin the images' own temperatures count (keypoint map)
POINTS_PER_BLOCK = 3000          # "shrink the blocks": ~1/3 of the profile joint's 8000
MAX_SPAN_SOLS = 25.0
EARLY_SOL = 380.0
_IDX = {n: i for i, n in enumerate(("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"))}
_MDEG = np.radians(1e-3)


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


def _q_xyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(np.asarray(R, float)).as_quat()          # x, y, z, w
    return (q if q[3] >= 0 else -q)[:3]


def _R_of_q(qxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q = np.asarray(qxyz, float)
    w = np.sqrt(max(0.0, 1.0 - float(q @ q)))
    return Rotation.from_quat([q[0], q[1], q[2], w]).as_matrix()


def start_values(cmods_dir: PathLike) -> Dict[str, Any]:
    """The global parameters' start from a cameras folder (the v0p80 form): G, S per eye, E, rig Q0 and drift."""
    d = Path(cmods_dir)
    nl, nr, rj = (json.loads((d / n).read_text(encoding="utf-8")) for n in
                  ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json"))
    out: Dict[str, Any] = {"T0": float((nl.get("thermal") or {}).get("T0_degC", -20.0)), "G": {}, "S": {}, "model": nl["model"]}
    for eye, j in (("NL", nl), ("NR", nr)):
        p = np.asarray(j["params"], float)
        th = j.get("thermal") or {}
        out["G"][eye] = p[:10].copy()
        ppm = float(th.get("ppm_per_degC") or 0.0)
        out["S"][eye] = np.array([1e-6 * p[0] * float(th.get("fx_ppm_per_degC", ppm)),
                                  1e-6 * p[1] * float(th.get("fy_ppm_per_degC", ppm)),
                                  float(th.get("cx_px_per_degC") or 0.0), float(th.get("cy_px_per_degC") or 0.0),
                                  float(th.get("k1_per_degC") or 0.0)])
    em = nl.get("early_mission") or {}
    out["E"] = np.array([float(em.get(f"d{k}_px", 0.0)) for k in EARLY_TERMS])
    dr = rj.get("drift") or {}
    knot = float(dr.get("knot_sol") or 300.0)
    post = np.array([1e3 * float(dr.get(f"{a}_mdeg_per_sol") or 0.0) for a in ("pitch", "yaw", "roll")])
    pre = np.array([1e3 * float(dr.get(f"{a}_early_mdeg_per_sol", dr.get(f"{a}_mdeg_per_sol")) or 0.0)
                    for a in ("pitch", "yaw", "roll")])
    sol0 = float(dr.get("sol0") or knot)
    q_c = _q_xyz(np.asarray(rj["R_sensor_from_ref"], float))
    out.update({"knot": knot, "D_pre": pre, "D_post": post,
                "Q0": q_c - 0.5 * _MDEG * post * (sol0 - knot) / 1000.0,
                "t": np.asarray(rj["t_sensor_from_ref_m"], float), "R": np.asarray(rj["R_sensor_from_ref"], float),
                "cameras": {"NL": nl, "NR": nr}, "rig": rj})
    return out


def _camera_params(st: Dict[str, Any], eye: str, T: float, early: bool) -> np.ndarray:
    G, S = st["G"][eye], st["S"][eye]
    p = np.zeros(12)
    p[:10] = G
    dT = T - st["T0"]
    for i, k in enumerate(SLOPE_TERMS):
        p[_IDX[k]] += S[i] * dT
    if early:
        for i, k in enumerate(EARLY_TERMS):
            p[_IDX[k]] += st["E"][i]
    return p


def tied_reconstruction(rec, proj, idx: Dict[str, Any], st: Dict[str, Any], bin_deg: float = BIN_DEG,
                        early_sol: float = EARLY_SOL, max_span: float = MAX_SPAN_SOLS,
                        gap: Optional[float] = None, breaks: Optional[Sequence[float]] = None):
    """A copy of a merged reconstruction (:func:`mppp.sfm.navcal.merge_scapes`: cameras 1 NL, 2 NR; rigs 1 stereo,
    2 NR alone, 3 NL alone) with one camera per (eye, temperature bin, era) and one rig per (block, rig epoch, bin,
    era).  Returns (rec, proj, units): units {"cameras": {cid: {...}}, "rigs": {rid: {...}}}."""
    import pycolmap
    from . import navcal as NC
    from .project import navcam_distortion_terms
    gap = NC.EPOCH_GAP_SOLS if gap is None else gap
    breaks = NC.RIG_EPOCH_BREAK_SOLS if breaks is None else breaks
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
    ep_of: Dict[int, int] = {}
    for scape in {f["scape"] for f in frames.values()}:
        fids = [k for k, f in frames.items() if f["scape"] == scape]
        eps = NC.sol_epochs([frames[k]["sol"] for k in fids], gap, breaks, max_span)
        for k in fids:
            s = frames[k]["sol"]
            ep_of[k] = next(i for i, (a, b) in enumerate(eps) if a <= s <= b)
    T0 = st["T0"]
    for k, f in frames.items():
        f["bin"] = int(np.floor((f["T"] - T0) / bin_deg))
        f["early"] = bool(f["sol"] < early_sol)
        f["epoch"] = ep_of[k]
    # camera units
    eye_of_cam = {1: "NL", 2: "NR"}
    cam_units: Dict[Tuple[str, int, bool], Dict[str, Any]] = {}
    for k, f in frames.items():
        for i in f["images"]:
            eye = eye_of_cam[int(rec.images[i].camera_id)]
            u = cam_units.setdefault((eye, f["bin"], f["early"]), {"T": [], "images": 0})
            u["T"].append(f["T"])
            u["images"] += 1
    new = pycolmap.Reconstruction()
    cam_id: Dict[Tuple[str, int, bool], int] = {}
    newproj = copy.deepcopy(proj)
    newproj.cameras = {}
    db: Dict[str, int] = {}
    units: Dict[str, Any] = {"cameras": {}, "rigs": {}}
    for n, (key, u) in enumerate(sorted(cam_units.items()), start=1):
        eye, b, early = key
        T = float(np.median(u["T"]))
        p = _camera_params(st, eye, T, early)
        base = rec.cameras[1 if eye == "NL" else 2]
        new.add_camera(pycolmap.Camera(camera_id=n, model=base.model, width=base.width, height=base.height, params=p))
        name = f"{eye}_t{b:+d}_{'early' if early else 'late'}"
        cam_id[key] = n
        db[name] = n
        c = {"model": base.model.name, "params": list(map(float, p)), "width": base.width, "height": base.height}
        newproj.cameras[name] = navcam_distortion_terms(c, "refine", k4="consensus")
        units["cameras"][n] = {"key": name, "eye": eye, "bin": b, "early": early, "T": T, "images": u["images"]}
    newproj.settings = dict(newproj.settings)
    newproj.settings["database"] = {"cameras": db}
    newproj.settings["navcam_rig_yaw"] = "refine"
    newproj.settings["navcam_rig_epochs_tie"] = False
    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))      # noqa: E731
    R0 = np.asarray(rec.rigs[1].sensor_from_rig(next(iter(rec.rigs[1].non_ref_sensors))).rotation.matrix())
    t0 = np.asarray(rec.rigs[1].sensor_from_rig(next(iter(rec.rigs[1].non_ref_sensors))).translation)
    rig_id: Dict[tuple, int] = {}
    rig_members: Dict[int, List[int]] = {}
    key_frames: Dict[tuple, List[int]] = {}
    for k, f in sorted(frames.items()):
        b, early = f["bin"], f["early"]
        key = (("stereo", f["scape"], f["epoch"], b, early) if f["rig"] == 1 else
               ("mono", "NR" if f["rig"] == 2 else "NL", b, early))
        key_frames.setdefault(key, []).append(k)
    knot = float(st["knot"])
    for key, mem in key_frames.items():
        rid = len(rig_id) + 1
        r = pycolmap.Rig(rig_id=rid)
        if key[0] == "stereo":
            b, early = key[3], key[4]
            # v0p83: each rig unit starts at the drift model's rotation for its own median sol (the ties are stiff)
            s_med = float(np.median([frames[k]["sol"] for k in mem]))
            q = st["Q0"] + 0.5 * _MDEG * (st["D_pre"] * (min(s_med, knot) - knot) + st["D_post"] * (max(s_med, knot) - knot)) / 1000.0
            r.add_ref_sensor(sensor(cam_id[("NL", b, early)]))
            r.add_sensor(sensor(cam_id[("NR", b, early)]), pycolmap.Rigid3d(pycolmap.Rotation3d(_R_of_q(q)), t0))
        else:
            b, early = key[2], key[3]
            r.add_ref_sensor(sensor(cam_id[(key[1], b, early)]))
        new.add_rig(r)
        rig_id[key] = rid
        rig_members[rid] = mem
    for key, rid in rig_id.items():
        if key[0] != "stereo":
            continue
        mem = rig_members[rid]
        units["rigs"][rid] = {"scape": key[1], "epoch": key[2], "bin": key[3], "early": key[4],
                              "sol": float(np.median([frames[k]["sol"] for k in mem])),
                              "T": float(np.median([frames[k]["T"] for k in mem])), "frames": len(mem),
                              "nr_camera": cam_id[("NR", key[3], key[4])]}
    for k, f in sorted(frames.items()):
        key = (("stereo", f["scape"], f["epoch"], f["bin"], f["early"]) if f["rig"] == 1 else
               ("mono", "NR" if f["rig"] == 2 else "NL", f["bin"], f["early"]))
        nf = pycolmap.Frame(frame_id=k, rig_id=rig_id[key])
        for i in f["images"]:
            eye = eye_of_cam[int(rec.images[i].camera_id)]
            nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cam_id[(eye, f["bin"], f["early"])]), id=i))
        nf.rig_from_world = rec.frames[k].rig_from_world
        new.add_frame(nf)
        for i in f["images"]:
            im = rec.images[i]
            eye = eye_of_cam[int(im.camera_id)]
            ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D], float).reshape(-1, 2),
                                camera_id=cam_id[(eye, f["bin"], f["early"])], image_id=i)
            ni.frame_id = k
            new.add_image(ni)
            if i in by_id:
                by_id[i]["camera_id"] = cam_id[(eye, f["bin"], f["early"])]
    for k in frames:
        new.register_frame(k)
    keep = set().union(*[set(f["images"]) for f in frames.values()]) if frames else set()
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
    return new, newproj, units


def make_ties(st: Dict[str, Any], units: Dict[str, Any], hold: Sequence[str] = (), tie_rigs: bool = True):
    """The ``extra_residuals`` callable for :func:`mppp.sfm.reconstruction.bundle_adjust` and the global parameter
    arrays it ties to.  ``hold``: global terms held at their start, e.g. ``("S_NL_cy", "S_NR_cy")``."""
    import pyceres
    LinearTie = _linear_tie_class()
    g = {"G_NL": st["G"]["NL"].copy(), "G_NR": st["G"]["NR"].copy(), "S_NL": st["S"]["NL"].copy(),
         "S_NR": st["S"]["NR"].copy(), "E": st["E"].copy(), "Q0": st["Q0"].copy(), "D_pre": st["D_pre"].copy(),
         "D_post": st["D_post"].copy()}
    sig = np.array([TIE_SIGMA["fx"], TIE_SIGMA["fy"], TIE_SIGMA["cx"], TIE_SIGMA["cy"]] + [TIE_SIGMA["k"]] * 6)
    knot = float(st["knot"])
    # terms the data cannot see are held at their start (else the problem has a null space and no covariance):
    # the early offsets without early-mission cameras, the drift before / after the knot without rig units there
    hold = list(hold)
    if not any(u["early"] for u in units["cameras"].values()):
        hold.append("E")
    if tie_rigs and not any(u["sol"] < knot for u in units["rigs"].values()):
        hold.append("D_pre")
    if tie_rigs and not any(u["sol"] > knot for u in units["rigs"].values()):
        hold.append("D_post")
    if not tie_rigs:
        hold += ["Q0", "D_pre", "D_post"]

    def hook(prob, cam_blocks=None, rig_blocks=None, key_of=None, work=None):
        n_cam = n_rig = 0
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
            if u["early"]:
                for i, k in enumerate(EARLY_TERMS):
                    A[TIED_PARAMS.index(k), 27 + i] = -1.0  # E
            prob.add_residual_block(LinearTie(A, sig, [12, 10, 5, 4]), None, [blk, g[f"G_{eye}"], g[f"S_{eye}"], g["E"]])
            n_cam += 1
        if tie_rigs:
            for rid, u in units["rigs"].items():
                blk = rig_blocks.get((int(rid), int(u["nr_camera"])))
                if blk is None or not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                    continue
                s = float(u["sol"])
                a_pre, a_post = (min(s, knot) - knot) / 1000.0, (max(s, knot) - knot) / 1000.0
                A = np.zeros((3, 7 + 3 + 3 + 3))
                for i in range(3):
                    A[i, i] = 1.0                          # the rig unit's quaternion x, y, z
                    A[i, 7 + i] = -1.0                     # Q0
                    A[i, 10 + i] = -0.5 * _MDEG * a_pre   # D_pre (mdeg / 1000 sol)
                    A[i, 13 + i] = -0.5 * _MDEG * a_post  # D_post
                prob.add_residual_block(LinearTie(A, np.full(3, TIE_SIGMA["q"]), [7, 3, 3, 3]), None,
                                        [blk, g["Q0"], g["D_pre"], g["D_post"]])
                n_rig += 1
        held = {}
        for h in hold:                                     # e.g. "S_NL_cy", "E", "D_pre_roll"
            name, _, comp = h.partition(":") if ":" in h else (h, "", "")
            for gname in g:
                if h == gname:
                    held.setdefault(gname, set()).update(range(len(g[gname])))
                elif h.startswith(gname + "_"):
                    c = h[len(gname) + 1:]
                    names = {"S_NL": SLOPE_TERMS, "S_NR": SLOPE_TERMS, "E": EARLY_TERMS, "G_NL": TIED_PARAMS,
                             "G_NR": TIED_PARAMS, "D_pre": ("pitch", "yaw", "roll"), "D_post": ("pitch", "yaw", "roll"),
                             "Q0": ("pitch", "yaw", "roll")}.get(gname, ())
                    if c in names:
                        held.setdefault(gname, set()).add(names.index(c))
        for gname, ix in held.items():
            if not prob.has_parameter_block(g[gname]):
                continue
            if len(ix) == len(g[gname]):
                prob.set_parameter_block_constant(g[gname])
            else:
                prob.set_manifold(g[gname], pyceres.SubsetManifold(len(g[gname]), sorted(ix)))
        return {"cov_blocks": {k: v for k, v in g.items() if prob.has_parameter_block(v)},
                "ties": {"cameras": n_cam, "rigs": n_rig}, "held": {k: sorted(v) for k, v in held.items()}}

    return hook, g


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


def fit_tied(scapes_cfg: Dict[str, PathLike], start_dir: Optional[PathLike] = None, points: int = POINTS_PER_BLOCK,
             bin_deg: float = BIN_DEG, max_span: float = MAX_SPAN_SOLS, early_sol: float = EARLY_SOL,
             hold: Sequence[str] = (), tie_rigs: bool = True, iterations: int = 100, covariance: bool = True,
             passes: int = 2, verbose: bool = True) -> Dict[str, Any]:
    """The tied joint over the blocks ``{name: WORK folder}`` (module docstring).  Returns the global terms with their
    sd, the per-unit rigs, residual statistics and the units."""
    import pycolmap
    from . import navcal as NC
    from .reconstruction import bundle_adjust
    from ..paths import cmods_dir
    d = Path(start_dir) if start_dir else cmods_dir()
    st = start_values(d)
    cams = {k: pycolmap.Camera(model=st["cameras"][k]["model"], width=5120, height=3840,
                               params=np.asarray(st["cameras"][k]["params"], float)) for k in ("NL", "NR")}
    t = time.time()
    rec, proj, idx = NC.merge_scapes((NC.load_scape(n, scapes_cfg[n]) for n in scapes_cfg), cams, (st["R"], st["t"]),
                                     points_per_scape=points, seed=0)
    rec, proj, units = tied_reconstruction(rec, proj, idx, st, bin_deg=bin_deg, early_sol=early_sol, max_span=max_span)
    if verbose:
        print(f"tied joint: {len(scapes_cfg)} blocks, {rec.num_reg_images()} images, {rec.num_points3D()} points; "
              f"{len(units['cameras'])} bin cameras, {len(units['rigs'])} rig units "
              f"(epochs <= {max_span:g} sols, {bin_deg:g} degC bins)  ({time.time() - t:.0f} s)", flush=True)
    hook, g = make_ties(st, units, hold=hold, tie_rigs=tie_rigs)
    t = time.time()
    # within a bin each image's own temperature still counts: its keypoints are moved by the current slopes from the
    # bin camera's temperature to the image's (scale about the principal point, principal-point shift); the slopes
    # are updated after each pass (``passes``)
    T_unit = {int(cid): float(u["T"]) for cid, u in units["cameras"].items()}
    eye_unit = {int(cid): u["eye"] for cid, u in units["cameras"].items()}
    img_T = {int(r["image_id"]): idx["temps"].get(r["name"]) for r in proj.images}
    img_cam = {int(r["image_id"]): int(r["camera_id"]) for r in proj.images}
    ba = None
    for p_ in range(max(1, int(passes))):
        ks, shift = {}, {}
        for iid, T in img_T.items():
            cid = img_cam.get(iid)
            if T is None or cid not in T_unit:
                continue
            eye, dT = eye_unit[cid], float(T) - T_unit[cid]
            Gs, Ss = g[f"G_{eye}"], g[f"S_{eye}"]
            ks[iid] = (1.0 + Ss[0] / Gs[0] * dT, 1.0 + Ss[1] / Gs[1] * dT)
            shift[iid] = np.array([Ss[2] * dT, Ss[3] * dT])
        kmap = (lambda iid, k, cam, _s=shift: k - _s[iid] if iid in _s else k)        # noqa: E731
        ba = bundle_adjust(rec, proj, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=True,
                           refine_rig="rotation", max_iterations=iterations, attitude_prior_deg=5.0,
                           covariance=covariance and p_ == max(1, int(passes)) - 1, navcam_aspect="free",
                           linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap, extra_residuals=hook)
        if verbose:
            print(f"  pass {p_ + 1}: {ba['brief'][:160]}", flush=True)
    st_res = NC.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap)
    cov = ba.get("covariance") or {}
    held = (ba.get("extra") or {}).get("held") or {}
    T0 = st["T0"]
    out: Dict[str, Any] = {"blocks": list(scapes_cfg), "points_per_block": points, "bin_deg": bin_deg,
                           "max_span_sols": max_span, "early_sol": early_sol, "T0_degC": T0, "start_dir": str(d),
                           "final_cost": float(ba["final_cost"]), "iterations": ba["iterations"], "brief": ba["brief"],
                           "seconds": time.time() - t, "residuals": st_res, "ties": (ba.get("extra") or {}).get("ties"),
                           "held": held, "covariance_error": cov.get("error"), "covariance_damping": cov.get("damping")}
    for eye in ("NL", "NR"):
        G, S = g[f"G_{eye}"], g[f"S_{eye}"]
        sdG, sdS = _sd(cov, f"G_{eye}", 10, held.get(f"G_{eye}", ())), _sd(cov, f"S_{eye}", 5, held.get(f"S_{eye}", ()))
        out[eye] = {"params_T0": dict(zip(TIED_PARAMS, map(float, G))),
                    "sd_params_T0": None if sdG is None else dict(zip(TIED_PARAMS, map(float, sdG))),
                    "fx_ppm_per_degC": float(1e6 * S[0] / G[0]), "fy_ppm_per_degC": float(1e6 * S[1] / G[1]),
                    "cx_px_per_degC": float(S[2]), "cy_px_per_degC": float(S[3]), "k1_per_degC": float(S[4]),
                    "sd": None if sdS is None else {"fx_ppm_per_degC": float(1e6 * sdS[0] / G[0]),
                                                    "fy_ppm_per_degC": float(1e6 * sdS[1] / G[1]),
                                                    "cx_px_per_degC": float(sdS[2]), "cy_px_per_degC": float(sdS[3]),
                                                    "k1_per_degC": float(sdS[4])}}
    sdE = _sd(cov, "E", 4, held.get("E", ()))
    out["early_mission"] = {"before_sol": early_sol, "offsets_px": dict(zip(EARLY_TERMS, map(float, g["E"]))),
                            "sd_px": None if sdE is None else dict(zip(EARLY_TERMS, map(float, sdE)))}
    R0 = _R_of_q(g["Q0"])
    sdDpre, sdDpost = _sd(cov, "D_pre", 3, held.get("D_pre", ())), _sd(cov, "D_post", 3, held.get("D_post", ()))
    out["rig"] = {"R_at_knot": R0.tolist(), "angles_at_knot_mdeg": NC.rig_angles(R0), "knot_sol": st["knot"],
                  "t_m": list(map(float, st["t"])), "tied": bool(tie_rigs),
                  "drift_pre_mdeg_per_1000sol": dict(zip(("pitch", "yaw", "roll"), map(float, g["D_pre"]))),
                  "drift_post_mdeg_per_1000sol": dict(zip(("pitch", "yaw", "roll"), map(float, g["D_post"]))),
                  "sd_pre": None if sdDpre is None else dict(zip(("pitch", "yaw", "roll"), map(float, sdDpre))),
                  "sd_post": None if sdDpost is None else dict(zip(("pitch", "yaw", "roll"), map(float, sdDpost)))}
    rows = []
    for rid, u in units["rigs"].items():
        if rid in rec.rigs:
            rg = rec.rigs[rid]
            sid = next(iter(rg.non_ref_sensors), None)
            if sid is not None:
                a = NC.rig_angles(np.asarray(rg.sensor_from_rig(sid).rotation.matrix()), R_ref=R0)
                rows.append(dict(u, **a))
    out["rig_units"] = rows
    out["cameras"] = [dict(u, params=list(map(float, rec.cameras[cid].params))) for cid, u in units["cameras"].items()
                      if cid in rec.cameras]
    if verbose:
        for eye in ("NL", "NR"):
            o = out[eye]
            sd = o["sd"] or {}
            print(f"{eye}: fx {o['fx_ppm_per_degC']:+.2f} +- {sd.get('fx_ppm_per_degC', float('nan')):.2f}, fy "
                  f"{o['fy_ppm_per_degC']:+.2f} +- {sd.get('fy_ppm_per_degC', float('nan')):.2f} ppm/degC; cx "
                  f"{o['cx_px_per_degC']:+.4f} +- {sd.get('cx_px_per_degC', float('nan')):.4f}, cy {o['cy_px_per_degC']:+.4f} "
                  f"+- {sd.get('cy_px_per_degC', float('nan')):.4f} px/degC; k1 {o['k1_per_degC']:+.2e}/degC", flush=True)
        e = out["early_mission"]
        print("early offsets " + ", ".join(f"{k} {v:+.3f}" for k, v in e["offsets_px"].items()) + " px", flush=True)
        r = out["rig"]
        print(f"rig drift (mdeg / 1000 sol) before sol {st['knot']:g}: {r['drift_pre_mdeg_per_1000sol']}; after: "
              f"{r['drift_post_mdeg_per_1000sol']}", flush=True)
        print(f"tied joint: cost {out['final_cost']:.1f}, it {out['iterations']}, median {st_res['median_px']:.4f} / rms "
              f"{st_res['rms_px']:.4f} px, corners {st_res['corner_median_px']:.3f} px ({out['seconds']:.0f} s); "
              f"covariance {'ok' if not out['covariance_error'] else out['covariance_error']}", flush=True)
    return out
