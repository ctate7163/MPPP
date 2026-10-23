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
* **Backlash states** (0.93.2) - each state its own camera with the pose shared: the regular-state bins get the
  principal point pin + (d_cx_regular, d_cy_regular) per eye, so the state's boresight change goes into the camera,
  not into the frame rotation or the rig (chal_rocks, 22 Oct: the only block with regular images, and the outlier
  of the per-block rigs, pitch -29 / yaw -12 mdeg).
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
from .forms import is_single_state
from . import tied as T

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]

ZOOM = 34
GLOBAL_TERMS = ("a_fx_backlash", "a_fy_backlash", "a_fx_regular", "a_fy_regular", "b", "k1", "k2", "p1", "p2", "b_cy",
                "d_cx_regular", "d_cy_regular")
NG = len(GLOBAL_TERMS)
# 0.93.2 (Christian, 22 Oct): each backlash state is its own camera - the regular state has its own principal point
# (d_cx_regular, d_cy_regular from the backlash state's) - with the pose shared, so that the state does not move the
# frame rotation or the Mastcam-Z rig.  The offset is observable through the exposures in which this eye is regular
# and the other eye in the backlash state (the other eye and the rig fix the pose); with fewer it is held
MIN_MIXED_EXPOSURES = 5
REGULAR_RATIO_RANGE = (1.004, 1.016)   # 0.93.5: backlash / regular f of a fitted regular state (the states ~0.95 % apart)
TIE_SIGMA_Z = {"f": 1e-3, "k": 1e-7, "p": 1e-8, "cy": 1e-3}       # px; unitless - the ties are effectively exact
NAV_POINTS_PER_BLOCK = 8000
Z_POINTS_PER_BLOCK = 8000
_FX, _FY, _CX, _CY, _K1, _K2, _P1, _P2 = range(8)                     # FULL_OPENCV parameter indices


def zoom_groups(zoom: int = ZOOM) -> Tuple[str, str]:
    return f"ZL{zoom:03d}", f"ZR{zoom:03d}"


def _state(spec: Dict[str, Any], key: str) -> str:
    """The state whose terms a bin is tied to: ``regular`` or ``backlash`` - a single-state (110 mm) bin is on the
    zoom's one line, kept in the backlash terms (0.93.5: :func:`mppp.calib.forms.zcam_state`)."""
    from .forms import zcam_state
    return "regular" if zcam_state(key, spec) == "regular" else "backlash"


def start_globals(model: Dict[str, Any], zoom: int = ZOOM) -> Dict[str, Any]:
    """The global terms at the focus model in use: per eye the :data:`GLOBAL_TERMS`, the reference focus, the
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
        off = [float(x) for x in (gm.get("regular_pp_offset_px") or [0.0, 0.0])][:2]
        out["eyes"][g] = {"G": np.array([afx, afy, rfx, rfy, b, dv["k1"], dv["k2"], dv["p1"], dv["p2"], bcy] + off, float),
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
    split_out: List[str] = []
    db = dict(proj.settings["database"]["cameras"])
    for old, new in units["passthrough_cameras"].items():
        z = idx["zcams"].get(int(old))
        if z is None:
            continue
        spec = dict(z["spec"])
        g = str(spec.get("group") or z["key"])[:5]
        if g not in groups:                                  # another zoom of the block: left out
            continue
        if spec.get("state_split"):
            # 0.93.6: a camera a block split off by its focal fits ("list_split", before 0.93.6) is left out - its
            # state is not the list's (a frame is high unless listed) and before 0.93.4 it could hold high-state
            # frames under a low-state name (Seitah Overlook ZCAM08114)
            split_out.append(f"{z['key']}@{z['scape']}")
            continue
        name = f"{z['key']}@{z['scape']}"
        foc = spec.get("focus_count_median")
        e = sg["eyes"][g]
        dF = (float(foc) - e["ref"]) if foc is not None else 0.0
        p = np.asarray(rec.cameras[new].params, float)
        stt = _state(spec, z["key"])
        reg = stt == "regular"
        G = e["G"]
        zunits[new] = {"key": name, "scape": z["scape"], "group": g, "eye": g[:2], "state": stt,
                       "focus": None if foc is None else float(foc), "dF": dF,
                       # the pins: the model's pp0 (full-frame pixels, as the padded frames), else from the bin
                       "c_cx": float(e["pp0"][0]) if len(e["pp0"]) == 2 else float(p[_CX] - (G[10] if reg else 0.0)),
                       "c_cy": float(e["pp0"][1]) if len(e["pp0"]) == 2 else float(p[_CY] - G[9] * dF
                                                                                    - (G[11] if reg else 0.0))}
        if reg:                                              # 0.93.2: the regular state's own principal point
            u = zunits[new]
            p[_CX] = u["c_cx"] + G[10]
            p[_CY] = u["c_cy"] + G[9] * dF + G[11]
            rec.cameras[new].params = p
            held, free = ["k3", "k4", "k5", "k6"], ["cx", "cy"]
        else:
            held, free = ["cx", "cy", "k3", "k4", "k5", "k6"], (["cy"] if g.startswith("ZR") else [])
        spec.update({"model": rec.cameras[new].model.name, "params": list(map(float, p)), "group": g,
                     "fixed_params": held, "free_params": free})
        proj.cameras[name] = spec
        db[name] = int(new)
    proj.settings = dict(proj.settings)
    proj.settings["database"] = {"cameras": db}
    proj.settings["zcam_virtual_rig"] = {f"Z{zoom:03d}": {"R_sensor_from_ref": sg["rig"]["R"].tolist(),
                                                          "t_sensor_from_ref": sg["rig"]["t"].tolist(), "hold": False}}
    zcams = set(zunits)
    sg["mixed"] = mixed_exposures(proj.images, zunits)
    sg["split_cameras_left_out"] = split_out
    if split_out:
        log.info("Mastcam-Z joint: %d cameras split off by the blocks' focal fits left out (%s)", len(split_out),
                 ", ".join(split_out[:12]) + (" ..." if len(split_out) > 12 else ""))
    for r in proj.images:                        # rows of other zooms / cameras out of the problem: deregistered
        if int(r.get("camera_id", 0)) in units["passthrough_cameras"].values() and int(r["camera_id"]) not in zcams:
            fid = rec.images[int(r["image_id"])].frame_id
            if fid in rec.frames and rec.frames[fid].has_pose:
                rec.deregister_frame(fid)
    return rec, proj, idx, st, units, zunits, sg


def mixed_exposures(images: Sequence[Dict[str, Any]], zunits: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    """Per eye group, the exposures (block, SCLK) in which this eye is in the regular state and the other eye of the
    same zoom in the backlash state - what makes the regular state's principal point observable (0.93.2)."""
    ex: Dict[Tuple[Any, Any], Dict[str, str]] = {}
    for r in images:
        u = zunits.get(int(r.get("camera_id", -1)))
        if u is None:
            continue
        ex.setdefault((u["scape"], r.get("sclk_key")), {})[u["group"]] = u["state"]
    out = {g: 0 for g in {u["group"] for u in zunits.values()}}
    for v in ex.values():
        for g, st in v.items():
            other = ("ZR" if g.startswith("ZL") else "ZL") + g[2:]
            if st == "regular" and v.get(other) == "backlash":
                out[g] += 1
    return out


def _navcam_globals(st: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {"G_NL": st["G"]["NL"], "G_NR": st["G"]["NR"], "S_NL": st["S"]["NL"], "S_NR": st["S"]["NR"]}


def make_zties(sg: Dict[str, Any], zunits: Dict[int, Dict[str, Any]], tie: bool = True,
               hold: Union[Sequence[str], Dict[str, Sequence[str]]] = ()):
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
                G0 = sg["eyes"][eye]["G"]
                for pi, gi in ((_FX, 2 if reg else 0), (_FY, 2 if reg else 1)):
                    a = np.zeros(12 + NG); a[pi] = 1.0; a[12 + 4] = -u["dF"] / ratio
                    # 0.93.5: the regular state scales f, not the pixel aspect - its fy is a_fx_regular x the
                    # backlash aspect at the start (22 Oct Z34: a free a_fy_regular gave a regular fy / fx of 0.984)
                    a[12 + gi] = -(float(G0[1] / G0[0]) if (reg and pi == _FY) else 1.0)
                    rows.append(a); sig.append(TIE_SIGMA_Z["f"])
                for pi, gi, s_ in ((_K1, 5, "k"), (_K2, 6, "k"), (_P1, 7, "p"), (_P2, 8, "p")):
                    a = np.zeros(12 + NG); a[pi] = 1.0; a[12 + gi] = -1.0
                    rows.append(a); sig.append(TIE_SIGMA_Z[s_])
                prob.add_residual_block(LinearTie(np.array(rows), np.array(sig), [12, NG]), None, [blk, g[eye]])
                if reg:                                      # 0.93.2: cx = pin + d_cx, cy = pin + b_cy dF + d_cy
                    c0 = np.array([u["c_cx"], u["c_cy"]], float)
                    keep.append(c0)
                    a = np.zeros((2, 12 + NG + 2))
                    a[0, _CX] = 1.0; a[0, 12 + 10] = -1.0; a[0, 12 + NG] = -1.0
                    a[1, _CY] = 1.0; a[1, 12 + 9] = -u["dF"]; a[1, 12 + 11] = -1.0; a[1, 12 + NG + 1] = -1.0
                    prob.add_residual_block(LinearTie(a, np.array([TIE_SIGMA_Z["cy"]] * 2), [12, NG, 2]), None,
                                            [blk, g[eye], c0])
                    prob.set_parameter_block_constant(c0)
                elif eye.startswith("ZR"):                   # cy = its pin + b_cy (focus - ref)
                    c0 = np.array([u["c_cy"]], float)
                    keep.append(c0)
                    a = np.zeros((1, 12 + NG + 1)); a[0, _CY] = 1.0; a[0, 12 + 9] = -u["dF"]; a[0, 12 + NG] = -1.0
                    prob.add_residual_block(LinearTie(a, np.array([TIE_SIGMA_Z["cy"]]), [12, NG, 1]), None,
                                            [blk, g[eye], c0])
                    prob.set_parameter_block_constant(c0)
                n += 1
        held: Dict[str, List[int]] = {}
        for eye, arr in g.items():
            if not prob.has_parameter_block(arr):
                continue
            ix = {3}                                         # a_fy_regular: a_fx_regular x the aspect (0.93.5)
            if not has_reg[eye]:
                ix |= {2, 3, 10, 11}                         # no regular-state bin: the regular offsets stay
            elif int((sg.get("mixed") or {}).get(eye, 0)) < MIN_MIXED_EXPOSURES:
                ix |= {10, 11}                               # 0.93.2: the regular pp not observable apart from the pose
            if not eye.startswith("ZR"):
                ix.add(9)                                    # ZL: its principal point is the reference (pinned)
            for h in (hold.get(eye, ()) if isinstance(hold, dict) else hold):     # {eye: terms} or terms for both
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


def start_model_from_blocks(scapes: Sequence[Any], zoom: int) -> Dict[str, Any]:
    """A start focus model (zcam-v2 layout) for a zoom without one, from the blocks' refined focus-bin cameras
    (:func:`mppp.calib.joint.load_scape` scapes): per eye the median fx, fy at the median focus with one slope fitted
    over the bins (0 when the focus spans under 50 counts), the median k1, k2, p1, p2 and principal point (the pin),
    the regular state at the backlash lines (ratio 1, held by the joint without regular bins); the rig the median of
    the blocks' Mastcam-Z rigs.  The fixed terms of the zoom are set by the joint."""
    from scipy.spatial.transform import Rotation
    rows: Dict[str, List[Tuple[float, np.ndarray, float]]] = {}
    rigs, ts = [], []
    for sc in scapes:
        db = (sc.project.settings.get("database") or {}).get("cameras") or {}
        for key, spec in sc.project.cameras.items():
            g = str(spec.get("group") or key)[:5]
            if g not in zoom_groups(zoom) or spec.get("focus_count_median") is None:
                continue
            if _state(spec, key) == "regular":               # 0.93.5: the start lines are the backlash state's
                continue
            cid = db.get(key)
            if cid is None or int(cid) not in sc.rec.cameras:
                continue
            p = np.asarray(sc.rec.cameras[int(cid)].params, float)
            rows.setdefault(g, []).append((float(spec["focus_count_median"]), p, float(spec.get("n_images") or 1)))
        v = (sc.project.settings.get("zcam_virtual_rig") or {}).get(f"Z{zoom:03d}") or {}
        if v.get("R_sensor_from_ref"):
            rigs.append(Rotation.from_matrix(np.asarray(v.get("R_refined") or v["R_sensor_from_ref"], float)).as_rotvec())
            ts.append(np.asarray(v.get("t_refined") or v["t_sensor_from_ref"], float))
    cams: Dict[str, Any] = {}
    for g, rr in rows.items():
        F = np.array([r[0] for r in rr])
        P = np.array([r[1] for r in rr])
        ref = float(np.median(F))
        b = 0.0
        if np.ptp(F) >= 50.0 and len(rr) >= 3:
            b = float(np.polyfit(np.r_[F, F], np.r_[P[:, _FX], P[:, _FY]], 1)[0])
        fx = float(np.median(P[:, _FX] - b * (F - ref)))
        fy = float(np.median(P[:, _FY] - b * (F - ref)))
        k = [float(np.median(P[:, i])) for i in (_K1, _K2, _P1, _P2)]
        cams[g] = {"f0_px": float(np.sqrt(fx * fy)), "slope_px_per_count": b, "aspect": fy / fx, "reference_focus": ref,
                   "focus_range": [float(F.min()), float(F.max())], "regular_ratio": 1.0,
                   "pp0_px": [float(np.median(P[:, _CX])), float(np.median(P[:, _CY]))],
                   "lines": {"reference_focus": ref, "backlash": {"fx": {"a": fx, "b": b, "form": "common"},
                                                                  "fy": {"a": fy, "b": b, "form": "common"}}},
                   "distortion": {"names": ["k1", "k2", "p1", "p2"], "params": k,
                                  "note": "start: the median of the blocks' bins"},
                   "pp": {"cx_px_per_count": 0.0, "cy_px_per_count": 0.0},
                   "thermal": {"T0_degC": -15.0, "sensor": "HEAD_FPA", "f_px_per_degC": 0.0, "note": "no temperature term"},
                   "n_bins": len(rr)}
    if not cams:
        raise ValueError(f"no Mastcam-Z {zoom} mm bins in the blocks")
    rig: Dict[str, Any] = {}
    if rigs:
        R = Rotation.from_rotvec(np.median(np.asarray(rigs), axis=0)).as_matrix()
        t = np.median(np.asarray(ts), axis=0)
        rig = {"R_sensor_from_ref": R.tolist(), "t_sensor_from_ref": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
               "note": f"start: the median of {len(rigs)} blocks' Mastcam-Z rigs"}
    return {"cameras": cams, "rig": rig, "form": "zcam-v2", "state": "start (block results)", "pixel_mm": 0.0074,
            "source": f"start model of {zoom} mm from {len(scapes)} blocks' refined focus bins (mppp.calib.ztied)"}


def fixed_globals(sg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Set the zoom's fixed distortion terms (:data:`mppp.calib.forms.ZCAM_FIXED_TERMS`) in the start globals ``sg``
    (changed in place); returns {eye group: {term: value}} of the terms to hold."""
    from .forms import zcam_fixed_terms
    out: Dict[str, Dict[str, float]] = {}
    for eye, e in sg["eyes"].items():
        fx = {k: float(v) for k, v in zcam_fixed_terms(eye).items() if k in GLOBAL_TERMS}
        for k, v in fx.items():
            e["G"][GLOBAL_TERMS.index(k)] = v
        if fx:
            out[eye] = fx
    return out


def adjust_z(rec, proj, idx, st, units, zunits, sg, iterations: int = 100, covariance: bool = True, passes: int = 2,
             tie: bool = True, refine_zcam_rig: str = "rotation", verbose: bool = True,
             hold: Union[Sequence[str], Dict[str, Sequence[str]]] = ()):
    """The Mastcam-Z joint's adjustment: Navcam cameras and rigs held (the promoted law), frame poses free, the
    Mastcam-Z bins tied to the global terms (:func:`make_zties`), the Mastcam-Z rig rotation refined."""
    from ..sfm.reconstruction import bundle_adjust
    hook, g = make_zties(sg, zunits, tie=tie, hold=hold)
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
    ok = np.isfinite(r)                  # NaN: the point behind the camera / outside the lens model (20 Oct joint)
    f = r[ok]
    return {"zcam_observations": int(r.size), "navcam_zcam_points": int(mixed),
            "zcam_nonfinite": int((~ok).sum()), "zcam_nonfinite_share": float((~ok).mean()) if r.size else 0.0,
            "zcam_median_px": float(np.median(f)) if f.size else None,
            "zcam_rms_px": float(np.sqrt(np.mean(f ** 2))) if f.size else None,
            "zcam_p95_px": float(np.percentile(f, 95)) if f.size else None}


def globals_result(sg, g, ba, zunits) -> Dict[str, Any]:
    """The global terms per eye with their sd (the problem's covariance of the global arrays)."""
    cov = ba.get("covariance") or {}
    held = (ba.get("extra") or {}).get("held") or {}
    out: Dict[str, Any] = {"covariance_error": cov.get("error"), "eyes": {}}
    for eye, arr in g.items():
        sd = T._sd(cov, eye, len(arr), held.get(eye, ()))
        e = sg["eyes"][eye]
        if 3 in set(held.get(eye, ())):              # 0.93.5: a_fy_regular follows a_fx_regular x the start aspect
            arr[3] = arr[2] * float(e["G"][1] / e["G"][0])
        out["eyes"][eye] = {"terms": dict(zip(GLOBAL_TERMS, map(float, arr))),
                            "sd": None if sd is None else dict(zip(GLOBAL_TERMS, map(float, sd))),
                            "held": [GLOBAL_TERMS[i] for i in held.get(eye, [])], "reference_focus": e["ref"],
                            "ratio_start": e["ratio"], "pp0_px": e["pp0"],
                            "mixed_exposures": int((sg.get("mixed") or {}).get(eye, 0)),
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
        if reg:                                         # 0.93.2: the regular state's principal point
            p[_CX] = u["c_cx"] + G[10]
            p[_CY] = u["c_cy"] + G[9] * u["dF"] + G[11]
        elif u["group"].startswith("ZR"):
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
               scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None,
               k_tests: bool = True) -> Dict[str, Any]:
    """The Mastcam-Z joint over ``blocks`` {name: WORK}: build, adjust, the distortion tests (:func:`distortion_tests`,
    ``k_tests``), the diagnostic pass.  ``model``: the focus model to start from (default: the one in use)."""
    from ..sfm.project import zcam_focus_model_path
    start_from_blocks = False
    if model is None:
        f = zcam_focus_model_path(zoom=zoom)
        if f.is_file():
            model = json.loads(f.read_text(encoding="utf-8"))
        else:                                         # 0.93: a zoom without a model starts from its blocks
            if scapes is None:
                scapes = [J.load_scape(n, w, single=False) for n, w in blocks.items()]
            model = start_model_from_blocks(scapes, zoom)
            start_from_blocks = True
            if verbose:
                log.info(f"Mastcam-Z joint: no {zoom} mm focus model in use - started from the blocks' refined bins")
    t = time.time()
    rec, proj, idx, st, units, zunits, sg = build(blocks, model, navcam_dir, zoom, nav_points, z_points, scapes=scapes, st=st)
    if verbose:
        log.info(f"Mastcam-Z joint: {len(blocks or scapes or [])} blocks, {rec.num_reg_images()} images, "
                 f"{rec.num_points3D()} points; {len(zunits)} Mastcam-Z bins, {len(units['cameras'])} Navcam bin "
                 f"cameras (held) ({time.time() - t:.0f} s)")
    if not zunits:
        raise ValueError(f"no Mastcam-Z {zoom} mm bins in the blocks")
    fixed = fixed_globals(sg)                       # 0.93: the zoom's fixed distortion terms, held
    hold = {e: sorted(v) for e, v in fixed.items()}
    if verbose and fixed:
        log.info("Mastcam-Z joint: held at fixed values " + "; ".join(
            f"{e} " + ", ".join(f"{k} {v:+g}" for k, v in sorted(fx.items())) for e, fx in sorted(fixed.items())))
    t = time.time()
    ba, g, stats = adjust_z(rec, proj, idx, st, units, zunits, sg, iterations=iterations, passes=passes,
                            verbose=verbose, hold=hold)
    res = globals_result(sg, g, ba, zunits)
    res["fixed_terms"] = fixed
    if start_from_blocks:
        res["start_model"] = model
    res["split_cameras_left_out"] = list(sg.get("split_cameras_left_out") or [])
    res.update({"zoom": zoom, "blocks": sorted({u["scape"] for u in zunits.values()}), "residuals": stats,
                "zcam": _zstats(rec, zunits), "rig": rig_result(proj, sg, zoom), "final_cost": float(ba["final_cost"]),
                "iterations": ba["iterations"], "brief": ba["brief"], "seconds": time.time() - t,
                "navcam_models": str(st["folder"]), "points": {"navcam_per_block": nav_points, "zcam_per_block": z_points}})
    res["distortion"] = distortion_curves(res, ba, zunits=zunits)
    if k_tests:
        res["distortion"]["tests"] = distortion_tests(rec, proj, idx, st, units, zunits, sg, g, ba,
                                                      iterations=iterations, verbose=verbose, fixed=fixed)
    if verbose:
        for line in describe(res):
            log.info(line)
    if diagnostics:
        res["diagnostics"] = diagnostic_pass(rec, proj, idx, st, units, zunits, sg, g, blocks=blocks, zoom=zoom,
                                             verbose=verbose)
    return res


# ====================================================================================== k1, k2: curves and tests
FRAME_PX = (1648, 1200)                     # the Mastcam-Z frame (full detector)
CURVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)    # radii as a fraction of the corner radius


def _bin_k_cov(cov: Dict[str, Any], zunits: Dict[int, Dict[str, Any]], eye: str) -> Optional[np.ndarray]:
    """The 2x2 covariance of an eye's (k1, k2) combined from its bin cameras' covariances (inverse-variance sum,
    correlations between bins ignored) - the fallback when the problem's covariance of the global terms is not
    finite (exact ties: 20 Oct)."""
    blocks = cov.get("blocks") or {}
    vf = float(cov.get("variance_factor", 1.0))
    info = np.zeros((2, 2))
    n = 0
    for u in zunits.values():
        c = blocks.get(f"camera:{u['key']}")
        if u["group"] != eye or c is None:
            continue
        c = np.asarray(c, float)
        if c.shape[0] <= _K2:
            continue
        ck = c[np.ix_([_K1, _K2], [_K1, _K2])] * vf
        if not np.all(np.isfinite(ck)) or np.linalg.det(ck) <= 0:
            continue
        info += np.linalg.inv(ck)
        n += 1
    if not n:
        return None
    try:
        return np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None


def distortion_curves(res: Dict[str, Any], ba: Optional[Dict[str, Any]] = None, zunits=None,
                      frame_px: Tuple[int, int] = FRAME_PX) -> Dict[str, Any]:
    """The radial distortion of each eye as a displacement (px) at :data:`CURVE_FRACTIONS` of the corner radius,
    d(r) = f r (k1 r^2 + k2 r^4) with r the normalized radius - what the images see - and the right-minus-left
    difference.  k1 and k2 are nearly collinear over the frame (r^3 and r^5 correlate 0.98 at 34 mm), so two eyes'
    coefficients can differ a lot while their curves agree: compare curves, not coefficients.  With a covariance of
    (k1, k2) (the global terms', else :func:`_bin_k_cov`): the sd of each curve point, the k1-k2 correlation and
    the z of k1 and k2."""
    eyes = res["eyes"]
    cov = (ba or {}).get("covariance") or {}
    out: Dict[str, Any] = {"fractions": list(CURVE_FRACTIONS), "eyes": {}}
    W, H = frame_px
    curves = {}
    for eye, e in eyes.items():
        tm, sd = e["terms"], e.get("sd") or {}
        f = 0.5 * (tm["a_fx_backlash"] + tm["a_fy_backlash"])
        rmax = float(np.hypot(W / 2.0, H / 2.0) / f)
        r = np.array(CURVE_FRACTIONS) * rmax
        J_ = np.stack([f * r ** 3, f * r ** 5], axis=1)              # d(displacement)/d(k1, k2)
        d = J_ @ np.array([tm["k1"], tm["k2"]])
        C = None
        src = None
        if sd and np.isfinite(sd.get("k1", np.nan)) and np.isfinite(sd.get("k2", np.nan)):
            C, src = np.diag([sd["k1"] ** 2, sd["k2"] ** 2]), "global terms (correlation not available)"
        if C is None and zunits:
            C = _bin_k_cov(cov, zunits, eye)
            src = "bin cameras (correlations between bins ignored)" if C is not None else None
        # the collinearity over the frame (pixels spread uniformly over the frame area: density ~ r)
        rr = np.linspace(0, rmax, 400)
        w = rr / rr.sum()
        a3, a5 = rr ** 3, rr ** 5
        corr_basis = float((w * a3 * a5).sum() / np.sqrt((w * a3 * a3).sum() * (w * a5 * a5).sum()))
        rec_e: Dict[str, Any] = {"f_px": f, "corner_radius_norm": rmax, "displacement_px": d.tolist(),
                                 "k1": tm["k1"], "k2": tm["k2"], "r3_r5_correlation": corr_basis}
        if C is not None:
            rec_e.update({"cov_source": src, "sd_k1": float(np.sqrt(C[0, 0])), "sd_k2": float(np.sqrt(C[1, 1])),
                          "corr_k1_k2": float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])) if C[0, 1] else None,
                          "z_k1": float(tm["k1"] / np.sqrt(C[0, 0])), "z_k2": float(tm["k2"] / np.sqrt(C[1, 1])),
                          "sd_displacement_px": np.sqrt(np.einsum("ij,jk,ik->i", J_, C, J_)).tolist()})
        out["eyes"][eye] = rec_e
        curves[eye] = (d, rec_e.get("sd_displacement_px"))
    ks = sorted(curves)
    if len(ks) == 2:
        (dl, sl), (dr, sr) = curves[ks[0]], curves[ks[1]]
        diff = np.asarray(dr) - np.asarray(dl)
        out["difference_px"] = diff.tolist()
        out["difference_of"] = f"{ks[1]} - {ks[0]}"
        if sl is not None and sr is not None:
            sdd = np.hypot(sl, sr)
            out["difference_sd_px"] = sdd.tolist()
            out["difference_z"] = (diff / np.where(sdd > 0, sdd, np.nan)).tolist()
    return out


def distortion_tests(rec, proj, idx, st, units, zunits, sg, g, ba, iterations: int = 60,
                     verbose: bool = True, fixed: Optional[Dict[str, Dict[str, float]]] = None
                     ) -> List[Dict[str, Any]]:
    """Likelihood-ratio tests of the distortion terms: the joint again from its solution with a term held, and the
    rise of the cost against the full model, as delta chi^2 = 2 (cost - cost_full) / variance factor (1 degree of
    freedom per held term).  (1) k2 = 0 in both eyes (is k2 needed at all?); (2) ZR's k1 and k2 held at ZL's (do
    the eyes differ?).  Each also gives the largest change of the distortion curve it causes (px) - with a million
    observations a tiny effect is "significant", so read the two together - and the eyes' curve difference with its
    sd (:func:`distortion_curves`) is the direct test of (2).  With terms held at fixed values (``fixed``, 0.93:
    34 mm k2), the tests are instead (1) those terms freed - how much the fixed values cost (delta chi2 of the
    fixed against the free model) - and (2) as before, with the fixed terms held."""
    from scipy.stats import chi2
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor") or 0.0)
    cost0 = float(ba["final_cost"])
    eyes = sorted(g)
    if len(eyes) != 2 or not vf:
        return [{"test": "skipped", "note": "two eyes and a variance factor (covariance) are needed"}]
    zl, zr = eyes
    base = {e: g[e].copy() for e in eyes}
    # (a "ZR k2 = ZL k2, k1 free" case was dropped: the step it needs in k1 is ~1e-4, which the exactly tied
    # problem does not take in one pass, so its delta chi2 came out larger than the stricter case's - a variant
    # that has not converged overstates delta chi2)
    fixed = fixed or {}
    cases = []                                   # (name, terms held at a value, held terms freed)
    if any(fixed.values()):
        terms = sorted({k for v in fixed.values() for k in v})
        cases.append((f"{', '.join(terms)} free (vs the fixed values)", {}, {e: list(v) for e, v in fixed.items()}))
    if not any("k2" in fixed.get(e, {}) for e in eyes):
        cases.append(("k2 = 0 (both eyes)", {zl: {"k2": 0.0}, zr: {"k2": 0.0}}, {}))
    if not any(k in fixed.get(zr, {}) for k in ("k1", "k2")):
        cases.append((f"{zr} k1, k2 = {zl} k1, k2", {zr: {"k1": float(base[zl][5]), "k2": float(base[zl][6])}}, {}))
    elif "k1" not in fixed.get(zr, {}):
        cases.append((f"{zr} k1 = {zl} k1", {zr: {"k1": float(base[zl][5])}}, {}))
    out = []
    for name, fix, free in cases:
        t = time.time()
        sgv = copy.deepcopy(sg)
        for e in eyes:
            G = base[e].copy()
            for term, v in fix.get(e, {}).items():
                G[GLOBAL_TERMS.index(term)] = v
            sgv["eyes"][e]["G"] = G
        hold = {e: sorted((set(fixed.get(e, {})) | set(fix.get(e, {}))) - set(free.get(e, []))) for e in eyes}
        try:
            bav, gv, _ = adjust_z(copy.deepcopy(rec), copy.deepcopy(proj), idx, st, units, zunits, sgv,
                                  iterations=iterations, covariance=False, passes=1, verbose=False, hold=hold)
        except Exception as ex:                                       # noqa: BLE001
            out.append({"test": name, "error": f"{type(ex).__name__}: {ex}"})
            continue
        dof = sum(len(v) for v in fix.values()) + sum(len(v) for v in free.values())
        dchi2 = 2.0 * abs(float(bav["final_cost"]) - cost0) / vf     # freed terms lower the cost
        resv = {"eyes": {e: {"terms": dict(zip(GLOBAL_TERMS, map(float, gv[e])))} for e in eyes}}
        resb = {"eyes": {e: {"terms": dict(zip(GLOBAL_TERMS, map(float, base[e])))} for e in eyes}}
        cv, cb = distortion_curves(resv), distortion_curves(resb)
        change = max(abs(a - b) for e in eyes for a, b in zip(cv["eyes"][e]["displacement_px"],
                                                              cb["eyes"][e]["displacement_px"]))
        p = float(chi2.sf(max(dchi2, 0.0), dof))
        rec_t = {"test": name, "dof": dof, "delta_chi2": dchi2, "p": p, "significant": bool(p < 0.001),
                 "z_equivalent": float(np.sqrt(max(dchi2, 0.0))) if dof == 1 else None,
                 "max_curve_change_px": float(change), "seconds": time.time() - t,
                 "refit_terms": {e: {k: float(gv[e][GLOBAL_TERMS.index(k)]) for k in ("k1", "k2")} for e in eyes}}
        out.append(rec_t)
        if verbose:
            log.info(f"  distortion test {name}: delta chi2 {dchi2:.1f} ({dof} dof, p {p:.2g}), curve change "
                     f"<= {change:.2f} px ({time.time() - t:.0f} s)")
    return out


def distortion_lines(dist: Dict[str, Any]) -> List[str]:
    """Report lines of :func:`distortion_curves` and its tests."""
    if not dist:
        return []
    out = []
    fr = dist["fractions"]
    for eye, e in dist["eyes"].items():
        pts = ", ".join(f"{d:+.2f}" for d in e["displacement_px"])
        extra = ""
        if e.get("sd_k1") is not None:
            extra = (f"; k1 {e['k1']:+.4f} +- {e['sd_k1']:.4f} (z {e['z_k1']:.1f}), k2 {e['k2']:+.4f} +- "
                     f"{e['sd_k2']:.4f} (z {e['z_k2']:.1f})" + (f", corr {e['corr_k1_k2']:+.3f}" if e.get("corr_k1_k2") is not None else "")
                     + f" [{e['cov_source']}]")
        out.append(f"{eye} radial displacement at {', '.join(f'{x:g}' for x in fr)} of the corner radius: {pts} px"
                   f" (r^3, r^5 correlate {e['r3_r5_correlation']:.3f} over the frame){extra}")
    if "difference_px" in dist:
        z = dist.get("difference_z")
        out.append(f"{dist['difference_of']}: " + ", ".join(f"{d:+.2f}" for d in dist["difference_px"]) + " px"
                   + (" (z " + ", ".join(f"{v:.1f}" for v in z) + ")" if z else ""))
    for t in dist.get("tests") or []:
        if "delta_chi2" in t:
            out.append(f"test {t['test']}: delta chi2 {t['delta_chi2']:.1f} ({t['dof']} dof, p {t['p']:.2g}, "
                       f"{'significant' if t['significant'] else 'not significant'} at 0.001); the curve moves "
                       f"<= {t['max_curve_change_px']:.2f} px")
        else:
            out.append(f"test {t.get('test')}: {t.get('error') or t.get('note')}")
    return out


def describe(res: Dict[str, Any]) -> List[str]:
    out = []
    for eye, e in res["eyes"].items():
        tm, sd = e["terms"], e.get("sd") or {}
        f = lambda k, n=2: f"{tm[k]:.{n}f}" + (f" +- {sd[k]:.{n}f}" if k in sd and np.isfinite(sd[k]) else "")   # noqa: E731
        lo = ("" if "a_fx_regular" in e.get("held", []) else
              f"; low backlash state fx {f('a_fx_regular')} px (high / low {tm['a_fx_backlash'] / tm['a_fx_regular']:.4f})")
        out.append(f"{eye}: fx {f('a_fx_backlash')}, fy {f('a_fy_backlash')} px at focus {e['reference_focus']:g} "
                   f"(high backlash state), slope {f('b', 5)} px/count; k1 {f('k1', 4)}, k2 {f('k2', 4)}, p1 {tm['p1']:+.2e}, "
                   f"p2 {tm['p2']:+.2e}" + (f"; cy slope {f('b_cy', 5)} px/count" if eye.startswith("ZR") else "")
                   + f"; {e['bins']} bins ({e['regular_bins']} in the low backlash state)" + lo
                   + ("" if "d_cx_regular" in e.get("held", []) else
                      f"; low-state principal point {f('d_cx_regular')}, {f('d_cy_regular')} px from the high "
                      f"state's ({e.get('mixed_exposures', 0)} exposures with the other eye in the high state)"))
    if res.get("split_cameras_left_out"):
        out.append(f"left out: {len(res['split_cameras_left_out'])} cameras the blocks split off by their focal fits "
                   f"(a frame is in the high state unless it is on the low-state list)")
    r = res.get("rig") or {}
    z = res.get("zcam") or {}
    out.append(f"Mastcam-Z rig: {r.get('change_mdeg', float('nan')):.1f} mdeg from the rig in use, baseline "
               f"{1e3 * r.get('baseline_m', float('nan')):.1f} mm (held); Mastcam-Z residuals median "
               f"{z.get('zcam_median_px') or float('nan'):.3f} / rms {z.get('zcam_rms_px') or float('nan'):.3f} px "
               f"({z.get('zcam_nonfinite', 0)} of {z.get('zcam_observations', 0)} not finite), "
               f"{z.get('navcam_zcam_points')} Navcam-Mastcam-Z points; covariance "
               f"{'ok' if not res.get('covariance_error') else res['covariance_error']}")
    out += distortion_lines(res.get("distortion") or {})
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
        single = is_single_state(eye)
        ratio_note = None
        if "a_fx_regular" in e["held"]:
            ratio = float(e["ratio_start"])
        else:                                         # 0.93.5: fx only (fy follows the aspect)
            ratio = float(tm["a_fx_backlash"] / tm["a_fx_regular"])
            lo, hi = REGULAR_RATIO_RANGE
            if not lo <= ratio <= hi:                 # not a focus state: the start ratio kept
                ratio_note = (f"fitted regular ratio {ratio:.4f} outside {lo:g}..{hi:g} (the states are ~0.95 % apart): "
                              f"the start ratio {float(e['ratio_start']):.4f} kept")
                ratio = float(e["ratio_start"])
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
        # 0.93.2: the regular state's own principal point (pp0 + the offset; the pose is shared with the backlash state)
        doff = [float(tm.get("d_cx_regular", 0.0)), float(tm.get("d_cy_regular", 0.0))]
        off_held = "d_cx_regular" in e["held"]
        if len(pp0) == 2:
            reg["cx"] = {"a": float(pp0[0]) + doff[0], "b": 0.0, "derived": "pp0 + the regular-state offset"}
            reg["cy"] = {"a": float(pp0[1]) + doff[1], "b": float(tm["b_cy"]) if eye.startswith("ZR") else 0.0,
                         "derived": "pp0 + the regular-state offset" + (" (and the cy slope)" if eye.startswith("ZR") else "")}
        gm["regular_pp_offset_px"] = doff
        gm["regular_pp_offset"] = {"sd_px": [sdv("d_cx_regular"), sdv("d_cy_regular")], "held": off_held,
                                   "mixed_exposures": int(e.get("mixed_exposures") or 0),
                                   "note": ("held: fewer than %d exposures with this eye regular and the other in the "
                                            "backlash state" % MIN_MIXED_EXPOSURES) if off_held else
                                           "fitted: the regular state's principal point minus the backlash state's "
                                           "(the pose shared: the other eye and the rig fix it)"}
        gm["lines"] = {"backlash": {pn: {k: v for k, v in x.items() if k in ("a", "b", "form")} for pn, x in bk.items()},
                       "regular": {pn: {k: v for k, v in x.items() if k in ("a", "b", "derived")} for pn, x in reg.items()},
                       "reference_focus": ref}
        gm.pop("regular_ratio_note", None)                # (a note of the model in use is not this fit's)
        if ratio_note:
            gm["regular_ratio_note"] = ratio_note
        gm["f0_px"] = float(np.sqrt(tm["a_fx_backlash"] * tm["a_fy_backlash"]))
        gm["slope_px_per_count"] = float(tm["b"])
        gm["aspect"] = float(tm["a_fy_backlash"] / tm["a_fx_backlash"])
        gm["reference_focus"] = ref
        gm["regular_ratio"] = ratio
        if single:                                    # 0.93.5: 110 mm has one state - no regular state at all
            gm["lines"].pop("regular", None)
            for k_ in ("regular_ratio", "regular_pp_offset_px", "regular_pp_offset", "regular_ratio_note"):
                gm.pop(k_, None)
            gm["focus_state"] = "single"
            reg = {}
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


# the joint's own gate (20 Oct, first real run: Z residuals NaN and four blocks aligned with older models, promoted)
ZJOINT_LIMITS = {"max_zcam_median_px": 0.6, "max_zcam_nonfinite_share": 0.005}


def joint_checks(res: Dict[str, Any], chk: Dict[str, Any], limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``check_zcam``'s checks plus the joint's: the Mastcam-Z residuals finite and small, and every block aligned
    with the camera models in use (``res["block_runs"]``, from the blocks' ``run_done.json``) - a block aligned with
    other models (older Navcam models, other tie rules) is not what the joint assumes."""
    lim = dict(ZJOINT_LIMITS, **(limits or {}))
    checks = list(chk.get("checks") or [])
    z = res.get("zcam") or {}
    med, share = z.get("zcam_median_px"), float(z.get("zcam_nonfinite_share") or 0.0)
    okm = med is not None and np.isfinite(med) and med <= lim["max_zcam_median_px"]
    checks.append({"check": "Mastcam-Z residuals", "ok": bool(okm),
                   "note": (f"median {med:.3f} px" if med is not None else "none") + f" (<= {lim['max_zcam_median_px']:g})"})
    checks.append({"check": "Mastcam-Z residuals finite", "ok": share <= lim["max_zcam_nonfinite_share"],
                   "note": f"{int(z.get('zcam_nonfinite') or 0)} of {int(z.get('zcam_observations') or 0)} not finite "
                           f"({100 * share:.2f} %)"})
    navfp, bfp = res.get("navcam_fingerprint"), res.get("block_navcam_fingerprints")
    if navfp and bfp:                                  # 0.93: the Navcam models are what the joint holds
        other = sorted(n for n, f in bfp.items() if f != navfp)
        checks.append({"check": "blocks aligned with the Navcam models in use", "ok": not other,
                       "note": ("all " + str(len(bfp))) if not other else
                       f"{', '.join(other)} aligned with other Navcam models (run them again first)"})
        runs = None
    else:
        runs = res.get("block_runs")
    if runs is not None:
        cur = res.get("models_fingerprint")
        other = sorted(n for n, r in runs.items() if not r or (cur and (r.get("camera_models") or {}).get("fingerprint") != cur))
        checks.append({"check": "blocks aligned with the models in use", "ok": not other,
                       "note": ("all " + str(len(runs))) if not other else
                       f"{', '.join(other)} aligned with other camera models (run them again first)"})
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def write_zjoint(res: Dict[str, Any], out_dir: PathLike, model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The candidate folder: ``M2020_ZCAM034_focus_model.json`` (the joint's model), its check against the model in
    use (``check.json``), ``zcam_joint.json`` (everything), ``block_lines.json``, the figures and ``report.md``."""
    from ..sfm.project import zcam_focus_model_path
    from .constrain import plot_zcam_rig_trends
    from .fit import check_zcam
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if model is None and res.get("start_model"):              # a zoom without a model: the start from the blocks
        model = res["start_model"]
    old = model if model is not None else json.loads(zcam_focus_model_path(zoom=int(res.get("zoom", ZOOM))).read_text(encoding="utf-8"))
    new = candidate_model(res, old)
    name = f"M2020_ZCAM{int(res['zoom']):03d}_focus_model.json"
    (out / name).write_text(json.dumps(new, indent=1, default=float), encoding="utf-8")
    chk = check_zcam(new, old)
    chk = joint_checks(res, chk)
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
