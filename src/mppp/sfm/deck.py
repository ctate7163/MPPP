"""
The rover deck targets inside the bundle adjustment (4 Nov 2026; Christian: "Look into ways of putting the fiducial
locations directly in the bundle adjustment ... Then build and test the deck-in-the-adjustment").

Model.  One rover pose ``T_j`` (block <- RNAV: rotation vector, translation, optionally a log scale) per rest
interval ``j`` (by default one per station; ``split="sol"`` gives each sol its own, for a rover that may have
settled).  Every (interval, target) pair is a 3-D point ``X_jk`` of the adjustment, observed by its deck pixels with
COLMAP's own reprojection costs - the same camera, frame pose, rig and intrinsic blocks as the tie points - and tied
to ``T_j P_k`` (``P_k`` the survey, :func:`mppp.calib.fiducials.survey_nav_m`) by a prior of ``survey_sigma_mm``
per axis (TDF4, which is our own ray fit: ``TDF4_SIGMA_MM``).  With a rigid ``T_j`` the survey sets the block's
metric scale, so a refined Navcam (and Hazcam) stereo baseline is measured against the survey inside the solve;
``scale_free=True`` frees a scale per interval instead (the deck's shape only).

What it gives: the baselines at the survey's scale, the rover frame of every rest interval (``T_j``: Hazcam mounts,
the mast in the rover frame), and the deck residuals as a check of the block, all from one adjustment.

Use: ``python -m mppp.sfm.deck WORK [--model cahv_ba] [--out cahv_ba_deck]`` on a solved block, or the pipeline
setting ``deck_in_ba`` (one more adjustment after the alignment; ``sparse/cahv_ba`` is kept as it is).  The block's
own settings decide what is refined (Navcam / Hazcam rig, held cameras); a translation-refined rig
(``navcam_rig_refine="full"``) is what lets the deck move the baseline.

Detections come from :func:`mppp.calib.fiducials.check_block` (the block's Navcam images); start ``T_j`` from the
label poses, refined on the deck pixels with the poses held.  The adjustment's prior gauge recentre (a shift of the
whole block, :func:`mppp.sfm.reconstruction.prior_gauge`) is applied to ``T_j`` too.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

DECK_SIGMA_PX = 0.35          # a deck target's pixel sigma, native px (the fiducial check's detection noise)
SURVEY_SIGMA_MM = 0.05        # the survey per axis: rigid (the deck is one aluminium body)
TDF4_SIGMA_MM = 0.4           # TDF4: our own ray fit (4 Nov: known to about +-0.4 mm by method)
DECK_LOSS_SCALE = 3.0         # Huber scale in units of the deck sigma
DECK_REJECT_PX = 3.0          # full-res px: a target this far (and > 4 x the median) from the start fit is dropped
                              # (Three Forks north sol 684: one AFT1 found 11 px off, a false symmetry centre)
TARGETS = ("TDF1", "TDF2", "TDF3", "AFT1", "RPFA1", "RPFA2", "TDF4")


def _rot():
    from scipy.spatial.transform import Rotation
    return Rotation


def survey_tie_class():
    """A pyceres cost ``(X - (s R(w) P + t)) / sigma`` on [X (3), T (7: w, t, log s)]."""
    import pyceres
    Rotation = _rot()

    class SurveyTie(pyceres.CostFunction):
        def __init__(self, P: Sequence[float], sigma_m: float, scale_free: bool = False):
            super().__init__()
            self.P = np.asarray(P, float)
            self.sig = float(sigma_m)
            self.sf = bool(scale_free)
            self.set_num_residuals(3)
            self.set_parameter_block_sizes([3, 7])

        def pred(self, T: np.ndarray) -> np.ndarray:
            s = float(np.exp(T[6])) if self.sf else 1.0
            return s * Rotation.from_rotvec(T[:3]).apply(self.P) + T[3:6]

        def Evaluate(self, parameters, residuals, jacobians):
            X, T = np.asarray(parameters[0], float), np.asarray(parameters[1], float)
            p0 = self.pred(T)
            residuals[:] = (X - p0) / self.sig
            if jacobians is not None:
                if jacobians[0] is not None:
                    jacobians[0][:] = (np.eye(3) / self.sig).ravel()
                if jacobians[1] is not None:
                    J = np.zeros((3, 7))
                    h = 1e-7
                    for i in (0, 1, 2) + ((6,) if self.sf else ()):
                        Tp = T.copy()
                        Tp[i] += h
                        J[:, i] = -(self.pred(Tp) - p0) / h / self.sig
                    J[:, 3:6] = -np.eye(3) / self.sig
                    jacobians[1][:] = J.ravel()
            return True
    return SurveyTie


def detections(work: Path, rec, names: Sequence[str] = TARGETS) -> List[Dict[str, Any]]:
    """The deck targets found in the block's Navcam images (full-resolution px, corner origin)."""
    from ..calib import fiducials as F
    res = F.check_block(work, rec=rec, figures=0, progress=lambda s: log.debug(s))
    by_stem = {rec.images[i].name[:-4] if rec.images[i].name.lower().endswith(".png") else rec.images[i].name: i
               for i in rec.reg_image_ids()}
    out = []
    for d in res.get("detections") or []:
        if d["target"] not in names:
            continue
        iid = by_stem.get(d["image"])
        if iid is None:
            iid = next((i for s, i in by_stem.items() if s.startswith(str(d["image"]))), None)
        if iid is None:
            continue
        out.append({"image_id": int(iid), "stem": d["image"], "target": d["target"], "sol": d.get("sol"),
                    "eye": d.get("eye"), "uv": np.array([float(d["x_full"]), float(d["y_full"])])})
    return out


Z_SEARCH_PX = 12.0            # Mastcam-Z: search around the prediction through the block and a Navcam interval's T
                              # (Three Forks north sol 692: found 1-3 px from it; a wide search finds false centres)


def z_detections(root: Path, rec, rows: Dict[str, Dict[str, Any]], T: Dict[str, np.ndarray], split: str,
                 names: Sequence[str] = TARGETS, search_px: float = Z_SEARCH_PX) -> List[Dict[str, Any]]:
    """The deck targets in the block's Mastcam-Z images (4 Nov: S2 / the Z baseline against the survey), predicted
    through the block's camera and pose and the rover pose ``T`` of the same station (any interval of it, else the
    first), and found by point symmetry within ``search_px`` (full = native px for Mastcam-Z)."""
    from ..calib import fiducials as F
    Rotation = _rot()
    P = F.survey_nav_m(list(names))
    out = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        r = rows.get(im.name) or {}
        if not str(r.get("instrument", "")).startswith("Z"):
            continue
        st = interval_of(r, "station")
        t = next((v for k, v in T.items() if k.startswith(st)), None)
        if t is None:
            continue
        cam = rec.cameras[im.camera_id]
        pj = F.Projector.from_pycolmap(cam)
        Rc, tc = _cam_pose(im)
        Rt = Rotation.from_rotvec(t[:3]).as_matrix()
        pred = {}
        for k, X in P.items():
            Xc = Rc @ (Rt @ X + t[3:6]) + tc
            if Xc[2] <= 0:
                continue
            uv = pj(Xc)[0]
            if np.all(np.isfinite(uv)) and 15 <= uv[0] < cam.width - 15 and 15 <= uv[1] < cam.height - 15:
                pred[k] = uv
        if not pred:
            continue
        img = F._read_image(root / "images" / im.name)
        if img is None:
            continue
        g = F.to_gray(img)
        s = float(r.get("downsample_scale") or 1.0)
        for k, uv in pred.items():
            nat = F.full_to_native(uv, s)
            d = F.detect_target(g, nat[0], nat[1], s, search_full_px=search_px)
            if d:
                out.append({"image_id": int(iid), "stem": r.get("stem") or im.name[:-4], "target": k, "sol": r.get("sol"),
                            "eye": str(r.get("instrument"))[1:2], "family": "Z",
                            "uv": F.native_to_full([d["x"], d["y"]], s), "interval": interval_of(r, split)})
    return out


def interval_of(row: Dict[str, Any], split: str = "station") -> str:
    """A rest interval's key: the station (S###D####), or station and sol (``split="sol"``)."""
    st = str(row.get("station") or f"S{int(row.get('site', 0)):03d}D{int(row.get('drive', 0)):04d}")
    return st if split == "station" else f"{st}_sol{int(row.get('sol', 0)):04d}"


def _cam_pose(im) -> Tuple[np.ndarray, np.ndarray]:
    cw = im.cam_from_world()
    return np.asarray(cw.rotation.matrix(), float), np.asarray(cw.translation, float)


def start_T(obs: List[Dict[str, Any]], rec, labels: Dict[str, Dict[str, Any]], scale_free: bool = False
            ) -> Tuple[np.ndarray, float]:
    """T (block <- RNAV) of one interval: the mean of the label poses, refined on the deck pixels with the
    block's poses and cameras held.  -> (T (7), rms px)."""
    from scipy.optimize import least_squares
    from ..calib import fiducials as F
    Rotation = _rot()
    P = F.survey_nav_m(sorted({o["target"] for o in obs}))
    Rs, ts = [], []
    for st in sorted({o["stem"] for o in obs}):
        cm = labels.get(st)
        if not cm:
            continue
        Rl, tl = F.label_pose(cm)
        Rc, tc = _cam_pose(rec.images[next(o["image_id"] for o in obs if o["stem"] == st)])
        Rs.append(Rc.T @ Rl)
        ts.append(Rc.T @ (tl - tc))
    if not Rs:
        raise ValueError("no label pose for the deck images of this interval")
    x0 = np.r_[Rotation.from_matrix(np.array(Rs)).mean().as_rotvec(), np.median(ts, 0), 0.0]
    pj = {cid: F.Projector.from_pycolmap(c) for cid, c in rec.cameras.items()}
    poses = {o["image_id"]: _cam_pose(rec.images[o["image_id"]]) for o in obs}

    def res(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        s = float(np.exp(x[6])) if scale_free else 1.0
        return np.concatenate([pj[rec.images[o["image_id"]].camera_id](
            poses[o["image_id"]][0] @ (s * R @ P[o["target"]] + x[3:6]) + poses[o["image_id"]][1])[0] - o["uv"]
            for o in obs])
    r = least_squares(res, x0, loss="soft_l1", f_scale=1.0)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2) * 2))


def deck_residuals(obs, rec, T: Dict[str, np.ndarray], scale_free: bool = False) -> np.ndarray:
    """Pixel residuals (n, 2) of the survey through ``T[interval]`` and the block's poses and cameras."""
    from ..calib import fiducials as F
    Rotation = _rot()
    P = F.survey_nav_m(sorted({o["target"] for o in obs}))
    pj = {cid: F.Projector.from_pycolmap(c) for cid, c in rec.cameras.items()}
    out = []
    for o in obs:
        t = T[o["interval"]]
        s = float(np.exp(t[6])) if scale_free else 1.0
        Rc, tc = _cam_pose(rec.images[o["image_id"]])
        X = s * Rotation.from_rotvec(t[:3]).apply(P[o["target"]]) + t[3:6]
        out.append(pj[rec.images[o["image_id"]].camera_id](Rc @ X + tc)[0] - o["uv"])
    return np.array(out).reshape(-1, 2)


def make_hook(obs: List[Dict[str, Any]], T: Dict[str, np.ndarray], sigma_px: float = DECK_SIGMA_PX,
              survey_sigma_mm: float = SURVEY_SIGMA_MM, tdf4_sigma_mm: float = TDF4_SIGMA_MM,
              scale_free: bool = False, scales: Optional[Dict[int, float]] = None,
              chain: Optional[Callable[..., Any]] = None, hold_poses: bool = False,
              loss_scale: float = DECK_LOSS_SCALE) -> Tuple[Callable[..., Any], Dict[str, Any]]:
    """The ``extra_residuals`` callable of :func:`mppp.sfm.reconstruction.bundle_adjust` that adds the deck, and
    the state it fills (``X``).  ``T`` {interval: array(7)} is refined in place.  ``scales`` {image_id: native /
    full-resolution scale} turns ``sigma_px`` (native) into full-resolution px.  ``chain``: another hook run
    first (e.g. the tied joint's).  ``hold_poses``: frame poses and rigs held (a test of the deck term alone)."""
    import pycolmap
    import pyceres
    import pycolmap.cost_functions as cf
    from ..calib import fiducials as F
    Rotation = _rot()
    SurveyTie = survey_tie_class()
    P = F.survey_nav_m(sorted({o["target"] for o in obs}))
    state: Dict[str, Any] = {"X": {}, "n_obs": 0}

    def hook(prob, cam_blocks=None, rig_blocks=None, key_of=None, work=None, **kw):
        info = chain(prob, cam_blocks=cam_blocks, rig_blocks=rig_blocks, key_of=key_of, work=work, **kw) if chain else None
        info = dict(info or {})
        for j, t in T.items():
            prob.add_parameter_block(t, 7)
            if not scale_free:
                prob.set_manifold(t, pyceres.SubsetManifold(7, [6]))
        X = {}
        for j in sorted(T):
            for k in sorted({o["target"] for o in obs if o["interval"] == j}):
                t = T[j]
                s = float(np.exp(t[6])) if scale_free else 1.0
                x = np.ascontiguousarray(s * Rotation.from_rotvec(t[:3]).apply(P[k]) + t[3:6], dtype=np.float64)
                X[(j, k)] = x
                prob.add_parameter_block(x, 3)
                sig = (tdf4_sigma_mm if k == "TDF4" else survey_sigma_mm) * 1e-3
                prob.add_residual_block(SurveyTie(P[k], sig, scale_free), None, [x, t])
        loss = pyceres.HuberLoss(float(loss_scale))
        n = 0
        for o in obs:
            im = work.images[o["image_id"]]
            fr = work.frames[im.frame_id]
            rig = work.rigs[fr.rig_id]
            pose = fr.rig_from_world.params
            if not prob.has_parameter_block(pose):
                continue                                  # a right Mastcam-Z image posed through its partner, or held out
            sc = float((scales or {}).get(o["image_id"], 1.0))
            cov = np.eye(2) * (float(sigma_px) / sc) ** 2
            model = work.cameras[im.camera_id].model
            cam = cam_blocks[int(im.camera_id)]
            sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=im.camera_id)
            if rig.is_ref_sensor(sensor):
                prob.add_residual_block(cf.ReprojErrorCost(model, cov, o["uv"]), loss, [X[(o["interval"], o["target"])], pose, cam])
            else:
                prob.add_residual_block(cf.RigReprojErrorCost(model, cov, o["uv"]), loss,
                                        [X[(o["interval"], o["target"])], rig_blocks[(fr.rig_id, im.camera_id)], pose, cam])
            n += 1
        if hold_poses:
            for fr in work.frames.values():
                arr = fr.rig_from_world.params
                if prob.has_parameter_block(arr):
                    prob.set_parameter_block_constant(arr)
            for arr in (rig_blocks or {}).values():
                if prob.has_parameter_block(arr):
                    prob.set_parameter_block_constant(arr)
        state["X"], state["n_obs"] = X, n
        info["deck"] = {"observations": n, "points": len(X), "intervals": sorted(T)}
        return info
    return hook, state


def _baselines(rec, key_of: Dict[int, str], proj=None) -> Dict[str, float]:
    out = {}
    for fam, v in ((proj.settings.get("zcam_virtual_rig") or {}) if proj is not None else {}).items():
        t = v.get("t_refined") or v.get("t_sensor_from_ref")      # the Mastcam-Z rig of each zoom (a block of its own)
        if t is not None:
            out[f"ZR{fam[1:]}"] = float(np.linalg.norm(np.asarray(t, float))) * 1e3
    for rid, rig in rec.rigs.items():
        for sid in rig.non_ref_sensors:
            out[key_of.get(int(sid.id), str(sid.id))] = float(np.linalg.norm(np.asarray(rig.sensor_from_rig(sid).translation))) * 1e3
    return out


def adjust(work, model: str = "cahv_ba", out_model: Optional[str] = "cahv_ba_deck", rec=None,
           split: str = "station", scale_free: bool = False, sigma_px: float = DECK_SIGMA_PX,
           survey_sigma_mm: float = SURVEY_SIGMA_MM, tdf4_sigma_mm: float = TDF4_SIGMA_MM,
           refine_rig: Any = None, refine_intrinsics: Optional[bool] = None, hold_poses: bool = False,
           report: bool = True, max_iterations: int = 100, zcam: bool = True,
           zcam_rig: Optional[str] = None) -> Dict[str, Any]:
    """One bundle adjustment of a solved block with the deck in it.  ``work``: the block or its ``colmap`` folder.
    ``refine_rig`` None: the block's own (True when it refined the Navcam baseline, else "rotation");
    ``refine_intrinsics`` None: True, with the block's held cameras held.  Writes ``sparse/<out_model>`` (None: not
    written) and ``deck/deck_<out_model or 'test'>.json``.  -> the report."""
    import pycolmap
    from ..calib import fiducials as F
    from .project import SfmProject
    from .reconstruction import bundle_adjust, _scales
    Rotation = _rot()
    w = Path(work)
    root = w / "colmap" if (w / "colmap" / "project.json").is_file() else w
    proj = SfmProject.load(root)
    if rec is None:
        rec = pycolmap.Reconstruction(str(root / "sparse" / model))
    rows = {r["name"]: r for r in proj.images}
    obs = detections(root, rec)
    if not obs:
        raise ValueError(f"{root}: no deck targets found in the block's Navcam images")
    for o in obs:
        o["interval"] = interval_of(rows.get(rec.images[o["image_id"]].name) or {}, split)
    man = F._manifest(root.parent)
    labels = {o["stem"]: (man.get(o["stem"]) or {}).get("camera_model_label") for o in obs}
    T, start, dropped = {}, {}, []
    for j in sorted({o["interval"] for o in obs}):
        sel = [o for o in obs if o["interval"] == j]
        T[j], rms = start_T(sel, rec, labels, scale_free)
        for o in sel:
            o["interval"] = j
        e0 = np.linalg.norm(deck_residuals(sel, rec, {j: T[j]}, scale_free), axis=1)
        bad = {id(o) for o, ei in zip(sel, e0) if ei > max(DECK_REJECT_PX, 4 * float(np.median(e0)))}
        if bad and len(sel) - len(bad) >= 4:
            for o, ei in zip(sel, e0):
                if id(o) in bad:
                    dropped.append({"image": o["stem"], "target": o["target"], "interval": j, "px": float(ei)})
            sel = [o for o in sel if id(o) not in bad]
            obs = [o for o in obs if id(o) not in bad]
            T[j], rms = start_T(sel, rec, labels, scale_free)
        T[j] = np.ascontiguousarray(T[j], dtype=np.float64)
        start[j] = {"rms_px": rms, "observations": len(sel), "images": len({o["image_id"] for o in sel}),
                    "T": T[j].tolist()}
        log.info("deck: interval %s, %d targets in %d images, start T on the held poses rms %.2f px",
                 j, len(sel), start[j]["images"], rms)
    if zcam:
        zobs = z_detections(root, rec, rows, T, split)
        for j in sorted({o["interval"] for o in zobs} - set(T)):
            sel = [o for o in zobs if o["interval"] == j]
            t0 = next((v for k, v in T.items() if k.split("_sol")[0] == j.split("_sol")[0]), None)
            if t0 is None or len(sel) < 4:
                zobs = [o for o in zobs if o["interval"] != j]
                continue
            T[j] = np.ascontiguousarray(np.array(t0, dtype=np.float64))
            start[j] = {"rms_px": float(np.sqrt(np.mean(deck_residuals(sel, rec, {j: T[j]}, scale_free) ** 2) * 2)),
                        "observations": len(sel), "images": len({o["image_id"] for o in sel}), "T": T[j].tolist(),
                        "from": "Mastcam-Z, started at the station's Navcam T"}
        obs = obs + zobs
        log.info("deck: %d Mastcam-Z targets in %d images", len(zobs), len({o["image_id"] for o in zobs}))
    key_of = {int(v): k for k, v in proj.settings.get("database", {}).get("cameras", {}).items()}
    b0 = _baselines(rec, key_of, proj)
    res_start = deck_residuals(obs, rec, T, scale_free)
    rs = proj.settings.get("reconstruction") or {}
    if refine_rig is None:
        # the block asked for the baseline (navcam_rig_refine "full": rig_translation_sigma_m is recorded) - the
        # deck's scale is what a one-station block lacks for it (its own stage held the rig: refine_rig False)
        refine_rig = True if rs.get("refine_rig") is True or rs.get("rig_translation_sigma_m") else "rotation"
    hold = tuple(rs.get("hold_cameras") or ())
    sc = _scales(rec, proj)
    hook, state = make_hook(obs, T, sigma_px, survey_sigma_mm, tdf4_sigma_mm, scale_free, scales=sc, hold_poses=hold_poses)
    ba = bundle_adjust(rec, proj, sigma_px=0.5, refine_intrinsics=True if refine_intrinsics is None else bool(refine_intrinsics),
                       refine_rig=refine_rig, rig_translation_sigma_m=float(proj.settings.get("rig_translation_sigma_m") or 0.005)
                       if refine_rig is True else None, hold_cameras=hold, extra_residuals=hook,
                       max_iterations=max_iterations,
                       refine_zcam_rig=zcam_rig or (proj.settings.get("zcam_rig_fit")
                                                    if proj.settings.get("zcam_rig_fit") in ("rotation", "full") else None))
    shift = np.asarray(((ba.get("prior_gauge") or {}).get("shift_m")) or [0.0, 0.0, 0.0], float)
    for t in T.values():                                  # the block was recentred after the solve: T with it
        t[3:6] += shift
    e = deck_residuals(obs, rec, T, scale_free)
    b1 = _baselines(rec, key_of, proj)
    out: Dict[str, Any] = {
        "work": str(root), "model": model, "out_model": out_model, "split": split, "scale_free": scale_free,
        "sigma_px": sigma_px, "survey_sigma_mm": survey_sigma_mm, "tdf4_sigma_mm": tdf4_sigma_mm,
        "refine_rig": refine_rig, "hold_cameras": list(hold), "hold_poses": hold_poses,
        "observations": len(obs), "images": len({o["image_id"] for o in obs}), "dropped": dropped,
        "deck_rms_px_start": float(np.sqrt(np.mean(res_start ** 2) * 2)),
        "deck_rms_px": float(np.sqrt(np.mean(e ** 2) * 2)),
        "baselines_mm_start": b0, "baselines_mm": b1,
        "baseline_change_ppm": {k: 1e6 * (b1[k] / b0[k] - 1) for k in b1 if k in b0 and b0[k]},
        "prior_gauge_shift_m": shift.tolist(),
        "solver": {k: ba.get(k) for k in ("termination", "initial_cost", "final_cost", "iterations") if k in ba},
        "intervals": {}}
    for j, t in T.items():
        sel = [i for i, o in enumerate(obs) if o["interval"] == j]
        ee = e[sel]
        out["intervals"][j] = {
            "T_block_from_rnav": {"rotvec_rad": t[:3].tolist(), "t_m": t[3:6].tolist(),
                                  "scale": float(np.exp(t[6])) if scale_free else 1.0},
            "start": start[j], "rms_px": float(np.sqrt(np.mean(ee ** 2) * 2)),
            "moved_from_start": {"rot_mdeg": (np.degrees((Rotation.from_rotvec(start[j]["T"][:3]).inv()
                                                         * Rotation.from_rotvec(t[:3])).as_rotvec()) * 1e3).tolist(),
                                 "shift_mm": ((t[3:6] - np.asarray(start[j]["T"][3:6]) - shift) * 1e3).tolist()},
            "survey_misfit_mm": {k: float(np.linalg.norm(state["X"][(jj, k)] - ((float(np.exp(t[6])) if scale_free else 1.0)
                                 * Rotation.from_rotvec(t[:3]).apply(F.survey_nav_m([k])[k]) + t[3:6] - shift)) * 1e3)
                                 for (jj, k) in state["X"] if jj == j}}
    out["residuals"] = [{"image": o["stem"], "target": o["target"], "interval": o["interval"],
                         "dx": float(v[0]), "dy": float(v[1])} for o, v in zip(obs, e)]
    errs = [p.error for p in rec.points3D.values()]
    out["tie_error_median_px"] = float(np.median(errs)) if errs else None
    if out_model:
        d = root / "sparse" / out_model
        d.mkdir(parents=True, exist_ok=True)
        rec.write(str(d))
    if report:
        rd = root / "deck"
        rd.mkdir(exist_ok=True)
        (rd / f"deck_{out_model or 'test'}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    log.info("deck in the adjustment: %d targets in %d images, rms %.2f -> %.2f px; baselines %s",
             out["observations"], out["images"], out["deck_rms_px_start"], out["deck_rms_px"],
             ", ".join(f"{k} {v:+.0f} ppm" for k, v in out["baseline_change_ppm"].items()))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mppp.sfm.deck", description=__doc__.split("\n\n")[0])
    ap.add_argument("work", help="the block folder or its colmap folder")
    ap.add_argument("--model", default="cahv_ba")
    ap.add_argument("--out", default="cahv_ba_deck", help="the model written (sparse/<out>); 'none': not written")
    ap.add_argument("--split", choices=("station", "sol"), default="station", help="rest intervals")
    ap.add_argument("--scale-free", action="store_true", help="a free scale per interval (the deck's shape only)")
    ap.add_argument("--sigma-px", type=float, default=DECK_SIGMA_PX)
    ap.add_argument("--survey-sigma-mm", type=float, default=SURVEY_SIGMA_MM)
    ap.add_argument("--rig", choices=("auto", "rotation", "full", "hold"), default="auto")
    ap.add_argument("--hold-intrinsics", action="store_true")
    ap.add_argument("--hold-poses", action="store_true", help="test: poses and rigs held")
    ap.add_argument("--zcam-rig", choices=("auto", "rotation", "full"), default="auto",
                    help="0.95.5: the Mastcam-Z rig in the deck adjustment (full: its baseline against the survey; auto: "
                         "the block's zcam_rig_fit)")
    ap.add_argument("--no-zcam", action="store_true", help="the Navcam (and Hazcam) deck targets only")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rig = {"auto": None, "rotation": "rotation", "full": True, "hold": False}[a.rig]
    out = adjust(a.work, a.model, None if a.out.lower() == "none" else a.out, split=a.split, scale_free=a.scale_free,
                 sigma_px=a.sigma_px, survey_sigma_mm=a.survey_sigma_mm, refine_rig=rig,
                 refine_intrinsics=False if a.hold_intrinsics else None, hold_poses=a.hold_poses,
                 zcam=not a.no_zcam, zcam_rig=None if a.zcam_rig == "auto" else a.zcam_rig)
    print(json.dumps({k: out[k] for k in ("observations", "images", "deck_rms_px_start", "deck_rms_px",
                                          "baselines_mm", "baseline_change_ppm")}, indent=1, default=float))
    for j, v in out["intervals"].items():
        print(j, json.dumps({k: v[k] for k in ("rms_px", "moved_from_start", "survey_misfit_mm")}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
