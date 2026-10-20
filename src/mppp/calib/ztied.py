"""
The Mastcam-Z joint (MPPP 0.91, Christian 20 Oct 2026): one bundle adjustment over the finished Navcam + Mastcam-Z
blocks of a zoom, with the Navcam held at the models in use and the Mastcam-Z focus model fitted.

* **Navcam held** - every Navcam image's camera is the promoted law at its temperature and era (one camera per eye,
  2 degC bin and era, :func:`mppp.calib.tied.tied_reconstruction`; within a bin the image's own temperature through
  the keypoint map) and every rig epoch the promoted rig trend at its sol; the cameras and rigs are constant, the
  frame poses free (position and attitude priors as in the block runs).  The Navcam fixes the scale and the
  directions the Mastcam-Z is fitted against.
* **Mastcam-Z fitted** - every block's focus-bin camera stays a camera of its own, tied by constraint residuals to
  the global focus law of its eye (the zcam-v2 form): fx = a_fx[state] + b (focus - ref), fy = a_fy[state] +
  b (focus - ref) with one slope b per eye (the regular state's slope b / regular ratio), k1, k2, p1, p2 one value
  per eye, ZL's principal point and ZR's cx held (pinned), ZR's cy = its pin + b_cy (focus - ref).  One Mastcam-Z
  rig (right from left) for all blocks: its rotation refined, its translation (the CAHV baseline) held.
* **Diagnostic pass** - the global law held, each block's bins (fx, fy) and its own rig free: the per-block focus
  lines and the per-block rig rotations about the joint - the sol and temperature trends of the rig.

The result is a candidate focus model in the zcam-v2 form (``write_zjoint``), checked with
:func:`mppp.calib.fit.check_zcam` against the one in use, promoted on request.
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
from . import tied as T

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]

ZOOM = 34
GLOBAL_TERMS = ("a_fx_backlash", "a_fy_backlash", "a_fx_regular", "a_fy_regular", "b", "k1", "k2", "p1", "p2", "b_cy")
TIE_SIGMA_Z = {"f": 1e-3, "k": 1e-7, "p": 1e-8, "cy": 1e-3}       # px; unitless - the ties are effectively exact
NAV_POINTS_PER_BLOCK = 8000
Z_POINTS_PER_BLOCK = 8000
_FX, _FY, _CX, _CY, _K1, _K2, _P1, _P2 = range(8)                     # FULL_OPENCV parameter indices


def zoom_groups(zoom: int = ZOOM) -> Tuple[str, str]:
    return f"ZL{zoom:03d}", f"ZR{zoom:03d}"


def _state(spec: Dict[str, Any], key: str) -> str:
    return "regular" if (spec.get("backlash_state") == "regular" or str(key).endswith("_reg")) else "backlash"


def start_globals(model: Dict[str, Any], zoom: int = ZOOM) -> Dict[str, Any]:
    """The global terms at the focus model in use: per eye the 10 :data:`GLOBAL_TERMS`, the reference focus, the
    regular ratio, pp0, and the rig (R, t right from left)."""
    out: Dict[str, Any] = {"eyes": {}}
    for g in zoom_groups(zoom):
        gm = (model.get("cameras") or {}).get(g) or {}
        ln = gm.get("lines") or {}
        bk, rg = ln.get("backlash") or {}, ln.get("regular") or {}
        ratio = float(gm.get("regular_ratio") or 1.0)
        f0 = float(gm.get("f0_px") or 4720.0)
        asp = float(gm.get("aspect") or 1.0)
        afx = float((bk.get("fx") or {}).get("a", f0 / np.sqrt(asp)))
        afy = float((bk.get("fy") or {}).get("a", f0 * np.sqrt(asp)))
        b = float((bk.get("fx") or {}).get("b", gm.get("slope_px_per_count") or 0.0))
        rfx = float((rg.get("fx") or {}).get("a", afx / ratio))
        rfy = float((rg.get("fy") or {}).get("a", afy / ratio))
        dist = gm.get("distortion") or {}
        dn, dp = list(dist.get("names") or []), list(dist.get("params") or [])
        dv = {n: float(dp[dn.index(n)]) if n in dn else 0.0 for n in ("k1", "k2", "p1", "p2")}
        bcy = float((gm.get("pp") or {}).get("cy_px_per_count") or 0.0) if g.startswith("ZR") else 0.0
        out["eyes"][g] = {"G": np.array([afx, afy, rfx, rfy, b, dv["k1"], dv["k2"], dv["p1"], dv["p2"], bcy], float),
                          "ref": float(gm.get("reference_focus", ln.get("reference_focus", 600.0))), "ratio": ratio,
                          "pp0": list(gm.get("pp0_px") or [])}
    rig = model.get("rig") or {}
    out["rig"] = {"R": np.asarray(rig.get("R_sensor_from_ref", np.eye(3)), float),
                  "t": np.asarray(rig.get("t_sensor_from_ref", [-0.2435, 0.0, 0.0]), float)}
    return out


# ====================================================================================== the merged problem
def build(blocks: Dict[str, PathLike], model: Dict[str, Any], navcam_dir: Optional[PathLike] = None,
          zoom: int = ZOOM, nav_points: int = NAV_POINTS_PER_BLOCK, z_points: int = Z_POINTS_PER_BLOCK,
          scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None):
    """The merged reconstruction of the blocks: Navcam on the promoted law (held), Mastcam-Z bins of ``zoom`` as
    cameras of their own.  Returns (rec, proj, idx, st, units, zunits); ``zunits`` {camera_id: {"key", "scape",
    "group", "eye", "state", "focus", "dF", "c_cy"}}.  ``scapes``: loaded blocks (tests), else read from ``blocks``;
    ``st``: the Navcam law (:func:`mppp.calib.tied.start_values` of ``navcam_dir``, default the models in use)."""
    import pycolmap
    st = T.start_values(navcam_dir) if st is None else st
    cams = {k: pycolmap.Camera(model=st["cameras"][k]["model"], width=5120, height=3840,
                               params=np.asarray(st["cameras"][k]["params"], float)) for k in ("NL", "NR")}
    if scapes is None:
        scapes = [J.load_scape(n, w, single=False) for n, w in blocks.items()]
    rec0, proj0, idx = J.merge_scapes(scapes, cams, (st["R"], st["t"]), points_per_scape=nav_points, seed=0,
                                      include_zcam=True, zcam_points_per_scape=z_points)
    rec, proj, units = T.tied_reconstruction(rec0, proj0, idx, st, passthrough=True)
    sg = start_globals(model, zoom)
    groups = zoom_groups(zoom)
    zunits: Dict[int, Dict[str, Any]] = {}
    db = dict(proj.settings["database"]["cameras"])
    for old, new in units["passthrough_cameras"].items():
        z = idx["zcams"].get(int(old))
        if z is None:
            continue
        spec = dict(z["spec"])
        g = str(spec.get("group") or z["key"])[:5]
        if g not in groups:                                  # another zoom of the block: left out
            continue
        name = f"{z['key']}@{z['scape']}"
        foc = spec.get("focus_count_median")
        e = sg["eyes"][g]
        dF = (float(foc) - e["ref"]) if foc is not None else 0.0
        p = np.asarray(rec.cameras[new].params, float)
        zunits[new] = {"key": name, "scape": z["scape"], "group": g, "eye": g[:2], "state": _state(spec, z["key"]),
                       "focus": None if foc is None else float(foc), "dF": dF,
                       # ZR cy's pin: the model's pp0 (full-frame pixels, as the padded frames), else from the bin
                       "c_cy": float(e["pp0"][1]) if len(e["pp0"]) == 2 else float(p[_CY] - e["G"][9] * dF)}
        held = ["cx", "cy", "k3", "k4", "k5", "k6"]
        spec.update({"model": rec.cameras[new].model.name, "params": list(map(float, p)), "group": g,
                     "fixed_params": held, "free_params": ["cy"] if g.startswith("ZR") else []})
        proj.cameras[name] = spec
        db[name] = int(new)
    proj.settings = dict(proj.settings)
    proj.settings["database"] = {"cameras": db}
    proj.settings["zcam_virtual_rig"] = {f"Z{zoom:03d}": {"R_sensor_from_ref": sg["rig"]["R"].tolist(),
                                                          "t_sensor_from_ref": sg["rig"]["t"].tolist(), "hold": False}}
    zcams = set(zunits)
    for r in proj.images:                        # rows of other zooms / cameras out of the problem: deregistered
        if int(r.get("camera_id", 0)) in units["passthrough_cameras"].values() and int(r["camera_id"]) not in zcams:
            fid = rec.images[int(r["image_id"])].frame_id
            if fid in rec.frames and rec.frames[fid].has_pose:
                rec.deregister_frame(fid)
    return rec, proj, idx, st, units, zunits, sg


def _navcam_globals(st: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {"G_NL": st["G"]["NL"], "G_NR": st["G"]["NR"], "S_NL": st["S"]["NL"], "S_NR": st["S"]["NR"]}


def make_zties(sg: Dict[str, Any], zunits: Dict[int, Dict[str, Any]], tie: bool = True,
               hold: Sequence[str] = ()):
    """The ``extra_residuals`` hook tying each Mastcam-Z bin camera to its eye's global terms (``tie`` False: no
    ties, the globals out of the problem - the diagnostic pass).  Returns (hook, g) with g {eye group: array}."""
    import pyceres
    LinearTie = T._linear_tie_class()
    g = {k: v["G"].copy() for k, v in sg["eyes"].items()}
    keep: List[np.ndarray] = []

    def hook(prob, cam_blocks=None, rig_blocks=None, key_of=None, work=None):
        n = 0
        has_reg = {k: False for k in g}
        if tie:
            for cid, u in zunits.items():
                blk = cam_blocks.get(int(cid))
                if blk is None or not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                    continue
                eye = u["group"]
                reg = u["state"] == "regular"
                has_reg[eye] |= reg
                ratio = float(sg["eyes"][eye]["ratio"]) if reg else 1.0
                rows, sig = [], []
                for pi, gi in ((_FX, 2 if reg else 0), (_FY, 3 if reg else 1)):
                    a = np.zeros(12 + 10); a[pi] = 1.0; a[12 + gi] = -1.0; a[12 + 4] = -u["dF"] / ratio
                    rows.append(a); sig.append(TIE_SIGMA_Z["f"])
                for pi, gi, s_ in ((_K1, 5, "k"), (_K2, 6, "k"), (_P1, 7, "p"), (_P2, 8, "p")):
                    a = np.zeros(12 + 10); a[pi] = 1.0; a[12 + gi] = -1.0
                    rows.append(a); sig.append(TIE_SIGMA_Z[s_])
                prob.add_residual_block(LinearTie(np.array(rows), np.array(sig), [12, 10]), None, [blk, g[eye]])
                if eye.startswith("ZR"):                     # cy = its pin + b_cy (focus - ref)
                    c0 = np.array([u["c_cy"]], float)
                    keep.append(c0)
                    a = np.zeros((1, 12 + 10 + 1)); a[0, _CY] = 1.0; a[0, 12 + 9] = -u["dF"]; a[0, 22] = -1.0
                    prob.add_residual_block(LinearTie(a, np.array([TIE_SIGMA_Z["cy"]]), [12, 10, 1]), None,
                                            [blk, g[eye], c0])
                    prob.set_parameter_block_constant(c0)
                n += 1
        held: Dict[str, List[int]] = {}
        for eye, arr in g.items():
            if not prob.has_parameter_block(arr):
                continue
            ix = set()
            if not has_reg[eye]:
                ix |= {2, 3}                                 # no regular-state bin: the regular offsets stay
            if not eye.startswith("ZR"):
                ix.add(9)                                    # ZL: its principal point is the reference (pinned)
            for h in hold:
                if h in GLOBAL_TERMS:
                    ix.add(GLOBAL_TERMS.index(h))
            held[eye] = sorted(ix)
            if len(ix) == len(arr):
                prob.set_parameter_block_constant(arr)
            elif ix:
                prob.set_manifold(arr, pyceres.SubsetManifold(len(arr), sorted(ix)))
        return {"cov_blocks": {k: v for k, v in g.items() if prob.has_parameter_block(v)
                               and not prob.is_parameter_block_constant(v)}, "ties": {"zcam_cameras": n},
                "held": held}

    return hook, g


def adjust_z(rec, proj, idx, st, units, zunits, sg, iterations: int = 100, covariance: bool = True, passes: int = 2,
             tie: bool = True, refine_zcam_rig: str = "rotation", verbose: bool = True):
    """The Mastcam-Z joint's adjustment: Navcam cameras and rigs held (the promoted law), frame poses free, the
    Mastcam-Z bins tied to the global terms (:func:`make_zties`), the Mastcam-Z rig rotation refined."""
    from ..sfm.reconstruction import bundle_adjust
    hook, g = make_zties(sg, zunits, tie=tie)
    navkeys = [k for k in proj.settings["database"]["cameras"] if str(k).startswith("N")]
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], _navcam_globals(st))
    ba = None
    for p_ in range(max(1, int(passes))):
        ba = bundle_adjust(rec, proj, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=True,
                           refine_principal_point=False, refine_rig=False, max_iterations=iterations,
                           attitude_prior_deg=5.0, covariance=covariance and p_ == passes - 1,
                           hold_cameras=navkeys, linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap,
                           refine_zcam_rig=refine_zcam_rig, extra_residuals=hook)
        if verbose:
            log.info(f"  pass {p_ + 1}: {ba['brief'][:160]}")
    stats = J.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap)
    return ba, g, stats


def _zstats(rec, zunits) -> Dict[str, Any]:
    """Residual statistics of the Mastcam-Z observations alone (full-frame px) and the Navcam-Mastcam-Z points."""
    zimg = {int(i) for i in rec.reg_image_ids() if int(rec.images[i].camera_id) in zunits}
    res, mixed = [], 0
    for pt in rec.points3D.values():
        ids = [int(el.image_id) for el in pt.track.elements]
        zs = [i for i in ids if i in zimg]
        if zs and len(zs) < len(ids):
            mixed += 1
        for el in pt.track.elements:
            if int(el.image_id) in zimg:
                im = rec.images[el.image_id]
                cam = rec.cameras[im.camera_id]
                uv = cam.img_from_cam(np.atleast_2d(im.cam_from_world() * pt.xyz))
                res.append(float(np.linalg.norm(np.asarray(uv)[0] - im.points2D[el.point2D_idx].xy)))
    r = np.asarray(res, float)
    return {"zcam_observations": int(r.size), "navcam_zcam_points": int(mixed),
            "zcam_median_px": float(np.median(r)) if r.size else None,
            "zcam_rms_px": float(np.sqrt(np.mean(r ** 2))) if r.size else None}


def globals_result(sg, g, ba, zunits) -> Dict[str, Any]:
    """The global terms per eye with their sd (the problem's covariance of the global arrays)."""
    cov = ba.get("covariance") or {}
    held = (ba.get("extra") or {}).get("held") or {}
    out: Dict[str, Any] = {"covariance_error": cov.get("error"), "eyes": {}}
    for eye, arr in g.items():
        sd = T._sd(cov, eye, len(arr), held.get(eye, ()))
        e = sg["eyes"][eye]
        out["eyes"][eye] = {"terms": dict(zip(GLOBAL_TERMS, map(float, arr))),
                            "sd": None if sd is None else dict(zip(GLOBAL_TERMS, map(float, sd))),
                            "held": [GLOBAL_TERMS[i] for i in held.get(eye, [])], "reference_focus": e["ref"],
                            "ratio_start": e["ratio"], "pp0_px": e["pp0"],
                            "bins": sum(1 for u in zunits.values() if u["group"] == eye),
                            "regular_bins": sum(1 for u in zunits.values() if u["group"] == eye and u["state"] == "regular")}
    return out


def rig_result(proj, sg, zoom: int = ZOOM) -> Dict[str, Any]:
    v = (proj.settings.get("zcam_virtual_rig") or {}).get(f"Z{zoom:03d}") or {}
    R = np.asarray(v.get("R_refined") or v.get("R_sensor_from_ref"), float)
    t = np.asarray(v.get("t_refined") or v.get("t_sensor_from_ref"), float)
    return {"R_sensor_from_ref": R.tolist(), "t_sensor_from_ref": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
            "change_mdeg": float(J.rotation_mdeg(sg["rig"]["R"], R)), "translation_source": "held (the rig in use)"}


# ====================================================================================== the diagnostic pass
def _block_meta(blocks: Dict[str, PathLike], proj) -> Dict[str, Dict[str, Any]]:
    """Per block: the median sol of its Mastcam-Z images and the lens temperature (block_info, if readable)."""
    from .fit import block_info
    out: Dict[str, Dict[str, Any]] = {}
    sols: Dict[str, List[int]] = {}
    for r in proj.images:
        if str(r.get("camera_group", "")).startswith("Z") and r.get("sol") is not None:
            sols.setdefault(r.get("scape"), []).append(int(r["sol"]))
    for n, w in (blocks or {}).items():
        T_opt = None
        try:
            T_opt = block_info(w).get("T_optics_degC")
        except Exception:                                   # noqa: BLE001
            pass
        out[n] = {"sol": float(np.median(sols[n])) if sols.get(n) else None, "T_optics_degC": T_opt}
    for n, s in sols.items():
        out.setdefault(n, {"sol": float(np.median(s)), "T_optics_degC": None})
    return out


def diagnostic_pass(rec, proj, idx, st, units, zunits, sg, g, blocks: Optional[Dict[str, PathLike]] = None,
                    zoom: int = ZOOM, iterations: int = 60, verbose: bool = True) -> Dict[str, Any]:
    """The global law held: every Mastcam-Z bin at the law with fx, fy free (no ties), one Mastcam-Z rig per block
    (rotation free).  Returns {"blocks": per block the lines (a, b of fx, fy per eye and state, weighted by the
    bins' observations) and the rig rotation about the joint (yaw, pitch, roll mdeg), "fits": the rig's trends with
    sol and lens temperature}."""
    from ..sfm.reconstruction import bundle_adjust
    from .fit import ols
    rec = copy.deepcopy(rec)
    proj = copy.deepcopy(proj)
    fam0 = f"Z{zoom:03d}"
    R0 = np.asarray((proj.settings["zcam_virtual_rig"][fam0]).get("R_refined")
                    or proj.settings["zcam_virtual_rig"][fam0]["R_sensor_from_ref"], float)
    t0 = np.asarray(proj.settings["zcam_virtual_rig"][fam0]["t_sensor_from_ref"], float)
    for cid, u in zunits.items():                       # the bins at the joint's law
        p = np.asarray(rec.cameras[cid].params, float).copy()
        G = g[u["group"]]
        reg = u["state"] == "regular"
        ratio = float(sg["eyes"][u["group"]]["ratio"]) if reg else 1.0
        p[_FX] = G[2 if reg else 0] + G[4] * u["dF"] / ratio
        p[_FY] = G[3 if reg else 1] + G[4] * u["dF"] / ratio
        p[_K1], p[_K2], p[_P1], p[_P2] = G[5], G[6], G[7], G[8]
        if u["group"].startswith("ZR"):
            p[_CY] = u["c_cy"] + G[9] * u["dF"]
        rec.cameras[cid].params = p
        spec = proj.cameras[u["key"]]
        spec["fixed_params"] = ["cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
        spec["free_params"] = []
    scapes = sorted({u["scape"] for u in zunits.values()})
    vr = {}
    for sc in scapes:
        vr[f"{fam0}@{sc}"] = {"R_sensor_from_ref": R0.tolist(), "t_sensor_from_ref": t0.tolist(), "hold": False}
    for r in proj.images:
        cg = str(r.get("camera_group", ""))
        if cg[:5] in zoom_groups(zoom) and r.get("scape"):
            r["camera_group"] = f"{cg[:5]}@{r['scape']}"
    proj.settings["zcam_virtual_rig"] = vr
    navkeys = [k for k in proj.settings["database"]["cameras"] if str(k).startswith("N")]
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], _navcam_globals(st))
    t = time.time()
    ba = bundle_adjust(rec, proj, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=False,
                       refine_principal_point=False, refine_rig=False, max_iterations=iterations, attitude_prior_deg=5.0,
                       hold_cameras=navkeys, linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap,
                       refine_zcam_rig="rotation")
    if verbose:
        log.info(f"Mastcam-Z diagnostic pass: {ba['brief'][:160]} ({time.time() - t:.0f} s)")
    obs = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            c = int(rec.images[el.image_id].camera_id)
            obs[c] = obs.get(c, 0) + 1
    meta = _block_meta(blocks or {}, proj)
    from scipy.spatial.transform import Rotation
    rows: List[Dict[str, Any]] = []
    for sc in scapes:
        v = proj.settings["zcam_virtual_rig"][f"{fam0}@{sc}"]
        Rb = np.asarray(v.get("R_refined") or v["R_sensor_from_ref"], float)
        rv = np.degrees(Rotation.from_matrix(Rb @ R0.T).as_rotvec()) * 1e3
        row: Dict[str, Any] = {"block": sc, "sol": meta.get(sc, {}).get("sol"),
                               "T_optics_degC": meta.get(sc, {}).get("T_optics_degC"),
                               "pitch_mdeg": float(rv[0]), "yaw_mdeg": float(rv[1]), "roll_mdeg": float(rv[2]),
                               "lines": {}}
        for eye in zoom_groups(zoom):
            for stt in ("backlash", "regular"):
                us = [(cid, u) for cid, u in zunits.items() if u["scape"] == sc and u["group"] == eye and u["state"] == stt
                      and u["focus"] is not None and obs.get(cid, 0) > 0]
                if not us:
                    continue
                F = np.array([u["dF"] for _, u in us]); w = np.array([obs[c] for c, _ in us], float)
                for pn, pi in (("fx", _FX), ("fy", _FY)):
                    y = np.array([float(rec.cameras[c].params[pi]) for c, _ in us])
                    if len(us) >= 3 and np.ptp(F) >= 300:
                        A = np.c_[np.ones_like(F), F]
                        sol = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
                        a, b = float(sol[0]), float(sol[1])
                    else:
                        bG = float(g[eye][4]) / (float(sg["eyes"][eye]["ratio"]) if stt == "regular" else 1.0)
                        a, b = float(np.average(y - bG * F, weights=w)), bG
                    row["lines"].setdefault(eye, {}).setdefault(stt, {})[pn] = {"a": a, "b": b, "bins": len(us),
                                                                                 "observations": int(w.sum())}
        rows.append(row)
    fits = {}
    for ang in ("yaw", "pitch", "roll"):
        y = np.array([r[f"{ang}_mdeg"] for r in rows], float)
        S = np.array([np.nan if r["sol"] is None else r["sol"] for r in rows], float)
        Tz = np.array([np.nan if r["T_optics_degC"] is None else r["T_optics_degC"] for r in rows], float)
        fits[ang] = {"sd_mdeg": float(np.std(y, ddof=1)) if len(y) > 1 else None, "vs_sol_per_1000": ols(S / 1000.0, y),
                     "vs_T": ols(Tz, y)}
    return {"blocks": rows, "fits": fits, "brief": ba["brief"]}


# ====================================================================================== the run and the files
def fit_zjoint(blocks: Dict[str, PathLike], model: Optional[Dict[str, Any]] = None, navcam_dir: Optional[PathLike] = None,
               zoom: int = ZOOM, nav_points: int = NAV_POINTS_PER_BLOCK, z_points: int = Z_POINTS_PER_BLOCK,
               iterations: int = 100, passes: int = 2, diagnostics: bool = True, verbose: bool = True,
               scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Mastcam-Z joint over ``blocks`` {name: WORK}: build, adjust, the diagnostic pass.  ``model``: the focus
    model to start from (default: the one in use)."""
    from ..sfm.project import zcam_focus_model_path
    if model is None:
        model = json.loads(zcam_focus_model_path().read_text(encoding="utf-8"))
    t = time.time()
    rec, proj, idx, st, units, zunits, sg = build(blocks, model, navcam_dir, zoom, nav_points, z_points, scapes=scapes, st=st)
    if verbose:
        log.info(f"Mastcam-Z joint: {len(blocks or scapes or [])} blocks, {rec.num_reg_images()} images, "
                 f"{rec.num_points3D()} points; {len(zunits)} Mastcam-Z bins, {len(units['cameras'])} Navcam bin "
                 f"cameras (held) ({time.time() - t:.0f} s)")
    if not zunits:
        raise ValueError(f"no Mastcam-Z {zoom} mm bins in the blocks")
    t = time.time()
    ba, g, stats = adjust_z(rec, proj, idx, st, units, zunits, sg, iterations=iterations, passes=passes, verbose=verbose)
    res = globals_result(sg, g, ba, zunits)
    res.update({"zoom": zoom, "blocks": sorted({u["scape"] for u in zunits.values()}), "residuals": stats,
                "zcam": _zstats(rec, zunits), "rig": rig_result(proj, sg, zoom), "final_cost": float(ba["final_cost"]),
                "iterations": ba["iterations"], "brief": ba["brief"], "seconds": time.time() - t,
                "navcam_models": str(st["folder"]), "points": {"navcam_per_block": nav_points, "zcam_per_block": z_points}})
    if verbose:
        for line in describe(res):
            log.info(line)
    if diagnostics:
        res["diagnostics"] = diagnostic_pass(rec, proj, idx, st, units, zunits, sg, g, blocks=blocks, zoom=zoom,
                                             verbose=verbose)
    return res


def describe(res: Dict[str, Any]) -> List[str]:
    out = []
    for eye, e in res["eyes"].items():
        tm, sd = e["terms"], e.get("sd") or {}
        f = lambda k, n=2: f"{tm[k]:.{n}f}" + (f" +- {sd[k]:.{n}f}" if k in sd and np.isfinite(sd[k]) else "")   # noqa: E731
        out.append(f"{eye}: fx {f('a_fx_backlash')}, fy {f('a_fy_backlash')} px at focus {e['reference_focus']:g} "
                   f"(backlash), slope {f('b', 5)} px/count; k1 {f('k1', 4)}, k2 {f('k2', 4)}, p1 {tm['p1']:+.2e}, "
                   f"p2 {tm['p2']:+.2e}" + (f"; cy slope {f('b_cy', 5)} px/count" if eye.startswith("ZR") else "")
                   + f"; {e['bins']} bins ({e['regular_bins']} regular)")
    r = res.get("rig") or {}
    z = res.get("zcam") or {}
    out.append(f"Mastcam-Z rig: {r.get('change_mdeg', float('nan')):.1f} mdeg from the rig in use, baseline "
               f"{1e3 * r.get('baseline_m', float('nan')):.1f} mm (held); Mastcam-Z residuals median "
               f"{z.get('zcam_median_px') or float('nan'):.3f} / rms {z.get('zcam_rms_px') or float('nan'):.3f} px, "
               f"{z.get('navcam_zcam_points')} Navcam-Mastcam-Z points; covariance "
               f"{'ok' if not res.get('covariance_error') else res['covariance_error']}")
    d = res.get("diagnostics") or {}
    for ang, fz in (d.get("fits") or {}).items():
        a, b = fz.get("vs_sol_per_1000") or {}, fz.get("vs_T") or {}
        sl = lambda x, u: (f"{x['slope']:+.2f} +- {x['se']:.2f} {u}" if x.get("slope") is not None else "-")   # noqa: E731
        out.append(f"rig {ang} per block: sd {fz.get('sd_mdeg') or float('nan'):.1f} mdeg; vs sol {sl(a, 'mdeg/1000 sol')}, "
                   f"vs lens T {sl(b, 'mdeg/degC')}")
    return out


def candidate_model(res: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """The focus model of the joint in the zcam-v2 form (the model in use with the joint's lines, distortion, ZR cy
    slope, regular ratio and rig)."""
    from . import forms
    new = copy.deepcopy(model)
    stamp = datetime.date.today().isoformat()
    cal: Dict[str, Any] = {"method": "zcam tied joint (mppp.calib.ztied)", "when": stamp, "blocks": res["blocks"],
                           "eyes": {}, "residuals": res.get("residuals"), "zcam": res.get("zcam"),
                           "navcam_models": res.get("navcam_models")}
    for eye, e in res["eyes"].items():
        gm = new["cameras"][eye]
        tm, sd = e["terms"], e.get("sd") or {}
        ref = float(e["reference_focus"])
        sdv = lambda k: float(sd[k]) if k in sd and np.isfinite(sd[k]) else None      # noqa: E731
        if "a_fx_regular" in e["held"]:
            ratio = float(e["ratio_start"])
        else:
            ratio = float(np.sqrt(tm["a_fx_backlash"] * tm["a_fy_backlash"] / (tm["a_fx_regular"] * tm["a_fy_regular"])))
        bk = {"fx": {"a": tm["a_fx_backlash"], "b": tm["b"], "sd_a": sdv("a_fx_backlash"), "sd_b": sdv("b"), "form": "common"},
              "fy": {"a": tm["a_fy_backlash"], "b": tm["b"], "sd_a": sdv("a_fy_backlash"), "sd_b": sdv("b"), "form": "common"}}
        for k in ("k1", "k2", "p1", "p2"):
            bk[k] = {"a": tm[k], "b": 0.0, "sd_a": sdv(k), "form": "constant"}
        pp0 = list(e.get("pp0_px") or gm.get("pp0_px") or [])
        if eye.startswith("ZR") and len(pp0) == 2:
            bk["cx"] = {"a": float(pp0[0]), "b": 0.0, "form": "pinned"}
            bk["cy"] = {"a": float(pp0[1]), "b": tm["b_cy"], "sd_b": sdv("b_cy"), "form": "pinned_slope"}
        reg = {pn: {"a": v["a"] / (ratio if pn in ("fx", "fy") else 1.0), "b": v["b"] / (ratio if pn in ("fx", "fy") else 1.0),
                    "derived": f"backlash / {ratio:.5f}" if pn in ("fx", "fy") else "backlash"} for pn, v in bk.items()}
        gm["lines"] = {"backlash": {pn: {k: v for k, v in x.items() if k in ("a", "b", "form")} for pn, x in bk.items()},
                       "regular": {pn: {k: v for k, v in x.items() if k in ("a", "b", "derived")} for pn, x in reg.items()},
                       "reference_focus": ref}
        gm["f0_px"] = float(np.sqrt(tm["a_fx_backlash"] * tm["a_fy_backlash"]))
        gm["slope_px_per_count"] = float(tm["b"])
        gm["aspect"] = float(tm["a_fy_backlash"] / tm["a_fx_backlash"])
        gm["reference_focus"] = ref
        gm["regular_ratio"] = ratio
        gm["distortion"] = {"names": ["k1", "k2", "p1", "p2"], "params": [tm["k1"], tm["k2"], tm["p1"], tm["p2"]],
                            "sd": [sdv(k) for k in ("k1", "k2", "p1", "p2")],
                            "note": "k1, k2, p1, p2 constant with focus (the Mastcam-Z joint; k3 = 0 held)"}
        if eye.startswith("ZR"):
            gm["pp"] = dict(gm.get("pp") or {}, cx_px_per_count=0.0, cy_px_per_count=float(tm["b_cy"]),
                            note="ZR principal point: cx pinned at pp0, cy pinned with its focus slope (the joint)")
        if len(pp0) == 2:
            gm["pp0_px"] = pp0
        gm["thermal"] = {"T0_degC": -15.0, "sensor": "HEAD_FPA", "f_px_per_degC": 0.0, "note": "no temperature term"}
        gm.pop("trend", None)
        cal["eyes"][eye] = {"backlash": bk, "regular": reg, "reference_focus": ref, "bins": e["bins"],
                            "regular_bins": e["regular_bins"], "held": e["held"]}
    r = res["rig"]
    d = res.get("diagnostics") or {}
    rows = d.get("blocks") or []
    new["rig"] = dict(new.get("rig") or {}, R_sensor_from_ref=r["R_sensor_from_ref"], t_sensor_from_ref=r["t_sensor_from_ref"],
                      baseline_m=r["baseline_m"], translation_source=r["translation_source"],
                      blocks=res["blocks"],
                      sd_rotvec_mdeg=[d["fits"][a]["sd_mdeg"] for a in ("pitch", "yaw", "roll")] if d.get("fits") else None,
                      residuals={"per_block": [{k: x[k] for k in ("block", "sol", "T_optics_degC", "yaw_mdeg", "pitch_mdeg",
                                                                  "roll_mdeg")} for x in rows],
                                 "fits": d.get("fits") or {}},
                      per_block=[{"block": x["block"], "sol": x["sol"], "T_optics_degC": x["T_optics_degC"]} for x in rows],
                      note="the Mastcam-Z joint: one rig for every block and focus (rotation refined, translation "
                           "held); per_block / residuals from the diagnostic pass (each block's own rig)")
    new["rig"].pop("sd_baseline_mm", None)
    new["rig"].pop("baseline_vs_cahv", None)
    cal["diagnostics"] = {"blocks": rows, "fits": d.get("fits")}
    new["calibration"] = cal
    new["state"] = "calibrated"
    new["form"] = forms.ZCAM_FORM_ID
    new["source"] = (f"MPPP Mastcam-Z joint (mppp.calib.ztied, {stamp}): one adjustment over {len(res['blocks'])} "
                     f"blocks ({', '.join(res['blocks'])}), the Navcam held at the models in use")
    return new


def block_line_rows(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The diagnostic pass's per-block lines in the rows of :func:`zcam_lines.plot_block_lines`."""
    rows = []
    for b in (res.get("diagnostics") or {}).get("blocks") or []:
        for eye, sts in (b.get("lines") or {}).items():
            for stt, prs in sts.items():
                for pn, e in prs.items():
                    rows.append({"block": b["block"], "group": eye, "state": stt, "param": pn, "a": e["a"], "b": e["b"],
                                 "sd_a": 0.0, "sd_b": 0.0, "T_optics_degC": b.get("T_optics_degC")})
    return rows


def write_zjoint(res: Dict[str, Any], out_dir: PathLike, model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The candidate folder: ``M2020_ZCAM034_focus_model.json`` (the joint's model), its check against the model in
    use (``check.json``), ``zcam_joint.json`` (everything), ``block_lines.json``, the figures and ``report.md``."""
    from ..sfm.project import zcam_focus_model_path
    from .constrain import plot_zcam_rig_trends
    from .fit import check_zcam
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    old = model if model is not None else json.loads(zcam_focus_model_path().read_text(encoding="utf-8"))
    new = candidate_model(res, old)
    name = f"M2020_ZCAM{int(res['zoom']):03d}_focus_model.json"
    (out / name).write_text(json.dumps(new, indent=1, default=float), encoding="utf-8")
    chk = check_zcam(new, old)
    (out / "check.json").write_text(json.dumps(chk, indent=1, default=float), encoding="utf-8")
    (out / "zcam_joint.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    rows = block_line_rows(res)
    (out / "block_lines.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    figs = []
    try:
        if plot_zcam_rig_trends(new, out / "zcam_rig_trends.png", label="(the Mastcam-Z joint)"):
            figs.append("zcam_rig_trends.png")
    except Exception as e:                                   # noqa: BLE001
        log.warning(f"rig figure: {e}")
    lines = [f"# Mastcam-Z {int(res['zoom'])} mm joint ({datetime.date.today().isoformat()})", "",
             f"Blocks: {', '.join(res['blocks'])}; the Navcam held at `{res.get('navcam_models')}`.", ""]
    lines += [f"- {x}" for x in describe(res)]
    lines += ["", "## Checks against the model in use", ""] + [f"- {'ok' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}"
                                                              for c in chk["checks"]]
    lines += [""] + [f"![{f}]({f})" for f in figs]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"folder": str(out), "model": str(out / name), "check": chk, "figures": figs, "block_lines": rows}
