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
from . import forms
from .forms import is_single_state, zcam_focus_below
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
# 0.95.0 (Christian, 27 Oct): ZR's regular-state offset is relative to ZL's (the reference eye, its offset held unless
# the mixed exposures make it observable) and needs only exposures with ZL in the problem, in any state
MIN_PAIRED_LOW_EXPOSURES = 10
LOW_LIST_CHECK = True                  # 0.93.7: a low-state camera enters only when its frames are on the list
# (0.94.4: the shipped list and the found list, `found` of build - "auto": <the blocks' root>/zcam_low_backlash_found.csv)
REGULAR_RATIO_RANGE = (1.004, 1.016)   # 0.93.5: backlash / regular f of a fitted regular state (the states ~0.95 % apart)
# 0.94.4 (Christian, 24 Oct): the high / low ratio may vary by block or sol - within a block the low frames give
# 1.0131-1.0153 against the block's high-state line, Three Forks North 1.010, Belva Crater 1.017.  Each group of
# low-state bins ("block": a block, "sol": a block's sol) gets its own offset on a_fx_regular (fy follows by the
# aspect), drawn to 0 with an sd of REGULAR_RATIO_GROUP_SD x f: the global regular ratio is then the groups' mean
# and a group that disagrees no longer pulls the high state's line or the distortion.  "global": one ratio (0.94.4)
# 26 Oct (the first real Mastcam-Z joint with it, Z34, 7 blocks): the exact-tie refinement found no decrease in 20
# tries and took 524 s per call (2.2 M residuals); Ceres' stiff ties hold the units on the law well enough here - off
EXACT_TIE_SOLVE = False
# 0.94.5 (Christian, 27 Oct: "tie the low backlash state's slope to the high and only fit the offset between the
# two"; the high state's focal length itself moves by sequence): one high / low offset per eye ("global"), the
# low state on the high state's slope (REGULAR_SAME_SLOPE; before: slope / ratio), and with fewer than
# MIN_LOW_IMAGES low-state images of an eye its ratio held at the 34 mm one (Z48 of 27 Oct: 6 images per eye in
# two blocks, which disagreed by 1.2 %)
REGULAR_RATIO_BY = "global"
REGULAR_SAME_SLOPE = True
MIN_LOW_IMAGES = 6                     # (Christian, 27 Oct: "under ~6"; Z63 has 5 per eye, Z48 6)
# 0.94.5 (Christian, 27 Oct: "leave out high- or low-state bins with too few observations (under ~500), and take
# the ratio across blocks"): a bin camera with fewer observations in the joint is not tied to the lines (its fx, fy
# free, the rest held at the start) - Z48 of 27 Oct: Chal Rocks' and Seitah Overlook's one high-state bins, ~150
# observations, fy / fx 1.7 % and 4.5 % off
MIN_BIN_OBSERVATIONS = 1000                 # (Christian, 27 Oct: 500 -> 1000 - Seitah Overlook's ZR048 high-state bin,
# 959 observations, fx 6547 / fy 6836, passed 500)
RATIO_HOLD_ZOOM = 34
# 0.95.0 (Christian, 27 Oct: "the focal length is not curved over focus; different sequences sit at different linear
# offsets with the same slope" - Seitah Overlook's fixed infinity focus -0.26 % from the Z34 line, Taylorfjellet's
# close look-down sequences about -0.5 %): each sequence (block, sol, sequence) of an eye gets an intercept offset
# on fx (fy x the aspect) on top of the eye's line - the way the ratio groups and the focus-from-below offset enter
# - shared by its states (a mixed sequence then measures the high / low ratio within itself), drawn to 0 with an sd
# of SEQ_OFFSET_SD x f (the eye's line is the sequences' weighted mean); a sequence with fewer than MIN_SEQ_IMAGES
# frames or MIN_SEQ_OBS observations in the joint holds its offset at 0 (the constants in mppp.calib.forms, shared with the
# blocks' line refit).  Needs sequence bins (zcam_bin_by "sequence")
SEQ_OFFSET_SD = forms.SEQ_OFFSET_SD
MIN_SEQ_IMAGES = forms.MIN_SEQ_IMAGES
MIN_SEQ_OBS = forms.MIN_SEQ_OBS
# 0.94.5 (Christian, 27 Oct): per block, the Mastcam-Z aspect (fy / fx of the diagnostic pass, the line with the
# most observations) against the Navcam's per-block dfy - dfx (the Navcam diagnostic pass over the same blocks)
BLOCK_ASPECT_MIN_OBS = 1000                # the same cut as the lines (MIN_BIN_OBSERVATIONS)
REGULAR_RATIO_GROUP_SD = 0.003
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
          scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None, found: Any = "auto"):
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
    off_list: List[str] = []
    # 0.93.7 (Christian, 23 Oct): a frame is in the low state only when it is on the shipped low-state list - a
    # block's low-state camera whose frames are not all on it (the machine-made found list, read by the blocks before
    # 0.93.6) is left out
    from ..sfm.backlash import default_found_path, is_low_backlash, load_low_backlash
    if found == "auto":                                      # 0.94.4: the found list next to the blocks
        roots = {Path(w).parent for w in (blocks or {}).values()}
        found = default_found_path(next(iter(roots)) if len(roots) == 1 else None)
    shipped = load_low_backlash(None, found=found)
    sg_found = str(found) if found else None
    by_cam: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for sc in scapes:
        for r in sc.project.images:
            by_cam.setdefault((sc.name, str(r.get("instrument"))), []).append(r)
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
        if LOW_LIST_CHECK and _state(spec, z["key"]) == "regular":
            mem = by_cam.get((z["scape"], str(z["key"]))) or []
            if not mem or not all(is_low_backlash(r, shipped) for r in mem):
                off_list.append(f"{z['key']}@{z['scape']}")
                continue
        name = f"{z['key']}@{z['scape']}"
        foc = spec.get("focus_count_median")
        e = sg["eyes"][g]
        dF = (float(foc) - e["ref"]) if foc is not None else 0.0
        p = np.asarray(rec.cameras[new].params, float)
        stt = _state(spec, z["key"])
        reg = stt == "regular"
        G = e["G"]
        seq = spec.get("sequence")                               # 0.95.0: set by sequence bins (zcam_bin_by "sequence")
        zunits[new] = {"key": name, "scape": z["scape"], "group": g, "eye": g[:2], "state": stt,
                       "fb": zcam_focus_below(z["key"]),           # 0.94.4: the focus came in from below
                       "focus": None if foc is None else float(foc), "dF": dF,
                       "observations": z.get("observations"),
                       "seq": (f"{z['scape']}|s{spec.get('sequence_sol')}|{seq}" if seq else None),
                       "sequence": seq, "sequence_sol": spec.get("sequence_sol"),
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
    sols_of: Dict[int, List[float]] = {}
    for r in proj.images:
        if int(r.get("camera_id", -1)) in zunits and r.get("sol") is not None:
            sols_of.setdefault(int(r["camera_id"]), []).append(float(r["sol"]))
    for cid, u in zunits.items():                # 0.94.4: the bin's sol (median of its images) for ratio_by "sol"
        u["sol"] = int(round(float(np.median(sols_of[cid])))) if sols_of.get(cid) else None
        u["images"] = len(sols_of.get(cid) or [])  # 0.95.0: the bin's frames in the problem
    sg["mixed"] = mixed_exposures(proj.images, zunits)
    sg["paired_low"] = paired_low_exposures(proj.images, zunits)         # 0.95.0 (Mastcam-Z joint: ZR's low-state principal-point offset fitted relative to ZL's from every exposure paired with ZL (not only the mixed ones); the site runs pin a regular-state line at pp0 + the model's offset (Christian, 27 Oct: the state shows in the centre - Seitah ZR063 0.65 px))
    sg["split_cameras_left_out"] = split_out
    sg["low_cameras_not_on_list"] = off_list
    sg["found_list"] = sg_found
    if off_list:
        log.info("Mastcam-Z joint: %d low-state cameras with frames not on the shipped or found low-state list left out (%s)",
                 len(off_list), ", ".join(off_list[:12]) + (" ..." if len(off_list) > 12 else ""))
    if split_out:
        log.info("Mastcam-Z joint: %d cameras split off by the blocks' focal fits left out (%s)", len(split_out),
                 ", ".join(split_out[:12]) + (" ..." if len(split_out) > 12 else ""))
    for r in proj.images:                        # rows of other zooms / cameras out of the problem: deregistered
        if int(r.get("camera_id", 0)) in units["passthrough_cameras"].values() and int(r["camera_id"]) not in zcams:
            fid = rec.images[int(r["image_id"])].frame_id
            if fid in rec.frames and rec.frames[fid].has_pose:
                rec.deregister_frame(fid)
    mark_thin_bins(rec, proj, zunits, sg)          # 0.94.5
    return rec, proj, idx, st, units, zunits, sg


def mark_thin_bins(rec, proj, zunits: Dict[int, Dict[str, Any]], sg: Dict[str, Any],
                   min_obs: Optional[int] = None) -> List[str]:
    """0.94.5: the bin cameras with fewer than ``min_obs`` (:data:`MIN_BIN_OBSERVATIONS`) observations in the joint
    get ``thin`` - left out of the ties (fx, fy free, the principal point and distortion held at the start); then
    ``sg["low_images"]`` {eye: low-state images of the bins kept} and ``sg["thin_bins"]`` [names]."""
    min_obs = MIN_BIN_OBSERVATIONS if min_obs is None else int(min_obs)
    obs: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            c = int(rec.images[el.image_id].camera_id)
            if c in zunits:
                obs[c] = obs.get(c, 0) + 1
    thin = []
    for cid, u in zunits.items():
        u["observations"] = int(obs.get(int(cid), 0))
        u["thin"] = bool(min_obs and u["observations"] < min_obs)
        if u["thin"]:
            thin.append(f"{u['key']} ({u['observations']})")
            spec = proj.cameras.get(u["key"])
            if spec is not None:
                spec["fixed_params"] = sorted(set(spec.get("fixed_params") or []) | {"cx", "cy", "k1", "k2", "p1", "p2"})
                spec["free_params"] = ["fx", "fy"]
    sg["thin_bins"] = thin
    sg["low_images"] = {g_: sum(1 for r in proj.images if int(r.get("camera_id", -1)) in zunits
                                and zunits[int(r["camera_id"])]["group"] == g_
                                and zunits[int(r["camera_id"])]["state"] == "regular"
                                and not zunits[int(r["camera_id"])].get("thin"))
                        for g_ in sorted({u["group"] for u in zunits.values()})}
    if thin:
        log.info("Mastcam-Z joint: %d bin cameras with fewer than %d observations left out of the lines (%s)", len(thin),
                 min_obs, ", ".join(thin[:12]) + (" ..." if len(thin) > 12 else ""))
    return thin


def mixed_exposures(images: Sequence[Dict[str, Any]], zunits: Dict[int, Dict[str, Any]],
                    other_state: Optional[str] = "backlash") -> Dict[str, int]:
    """Per eye group, the exposures (block, SCLK) in which this eye is in the regular state and the other eye of the
    same zoom in the backlash state - what makes the regular state's principal point observable (0.93.2).
    ``other_state`` None (0.95.0): the other eye in the problem in any state - the exposures that make the right
    eye's regular-state principal point observable relative to the left eye's (:func:`paired_low_exposures`)."""
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
            if st == "regular" and (v.get(other) == other_state if other_state else other in v):
                out[g] += 1
    return out


def paired_low_exposures(images: Sequence[Dict[str, Any]], zunits: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    """0.95.0 (Christian, 27 Oct: "the right eye's regular principal-point offset - the state shows in the centre"):
    per eye group, the exposures in which this eye is in the regular (low) state and the other eye is in the problem
    in any state.  With ZL the reference eye (its principal point pinned, its regular offset held unless the mixed
    exposures make it observable), ZR's regular-state offset is the right-minus-left boresight change of the low
    state, observable through the rig in every such exposure - not only in the mixed ones (0.93.2), which Z63 lacks
    (Seitah Overlook ZR063's low-state frames: 0.65 px)."""
    return mixed_exposures(images, zunits, other_state=None)


def _navcam_globals(st: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {"G_NL": st["G"]["NL"], "G_NR": st["G"]["NR"], "S_NL": st["S"]["NL"], "S_NR": st["S"]["NR"]}


def slope_div(sg: Dict[str, Any], eye: str, reg: bool) -> float:
    """What a low-state bin divides the focus slope by: 1 (:data:`REGULAR_SAME_SLOPE`, 0.94.5) or the eye's start
    ratio (the regular lines the backlash lines scaled, before 0.94.5); 1 for a backlash-state bin."""
    if not reg or REGULAR_SAME_SLOPE:
        return 1.0
    return float(sg["eyes"][eye]["ratio"])


def ratio_group(u: Dict[str, Any], by: Optional[str] = None) -> Optional[str]:
    """The group of a low-state bin whose high / low ratio is its own (:data:`REGULAR_RATIO_BY`): the block, or the
    block and sol; None for "global" (one ratio per eye)."""
    by = REGULAR_RATIO_BY if by is None else by
    if by == "global":
        return None
    if by == "sol":
        return f"{u['scape']}|sol{u.get('sol')}"
    return str(u["scape"])


def ratio_groups(zunits: Dict[int, Dict[str, Any]], by: Optional[str] = None) -> Dict[Tuple[str, str], List[int]]:
    """{(eye group, ratio group): [low-state bin camera ids]} - every eye with two or more groups (one group: its
    ratio is the eye's own, no offset)."""
    out: Dict[Tuple[str, str], List[int]] = {}
    for cid, u in zunits.items():
        k = ratio_group(u, by) if (u["state"] == "regular" and not u.get("thin")) else None
        if k is not None:
            out.setdefault((u["group"], k), []).append(int(cid))
    n_eye: Dict[str, int] = {}
    for eye, _ in out:
        n_eye[eye] = n_eye.get(eye, 0) + 1
    return {k: v for k, v in out.items() if n_eye[k[0]] >= 2}


def sequence_groups(zunits: Dict[int, Dict[str, Any]], min_images: Optional[int] = None,
                    min_obs: Optional[int] = None) -> Tuple[Dict[Tuple[str, str], List[int]], Dict[Tuple[str, str], Dict[str, Any]]]:
    """0.95.0: {(eye group, sequence key): [bin camera ids]} of the sequences whose offset is free - at least
    ``min_images`` frames and ``min_obs`` observations in the joint over the sequence's bins (:data:`MIN_SEQ_IMAGES`,
    :data:`MIN_SEQ_OBS`) - and {key: {"images", "observations", "bins", "held"}} of every sequence (the thin ones with
    ``held`` set: their offset is 0).  Bins without a sequence (focus bins, projects before 0.95.0) have none."""
    mi = MIN_SEQ_IMAGES if min_images is None else int(min_images)
    mo = MIN_SEQ_OBS if min_obs is None else int(min_obs)
    groups: Dict[Tuple[str, str], List[int]] = {}
    for cid, u in zunits.items():
        if u.get("seq") and not u.get("thin"):          # (a thin bin is not on the lines: no offset either)
            groups.setdefault((u["group"], u["seq"]), []).append(int(cid))
    info: Dict[Tuple[str, str], Dict[str, Any]] = {}
    free: Dict[Tuple[str, str], List[int]] = {}
    for k, cids in groups.items():
        us = [zunits[c] for c in cids]
        n_im = sum(int(u.get("images") or 0) for u in us)
        n_ob = sum(int(u.get("observations") or 0) for u in us)
        F = [u["dF"] for u in us if u.get("focus") is not None]
        row = {"images": n_im, "observations": n_ob, "bins": len(cids), "focus_span": float(np.ptp(F)) if F else 0.0,
               "states": sorted({u["state"] for u in us}), "held": None}
        if n_im < mi:
            row["held"] = f"held at 0: {n_im} frames (< {mi})"
        elif n_ob < mo:
            row["held"] = f"held at 0: {n_ob} observations (< {mo})"
        else:
            free[k] = cids
        info[k] = row
    return free, info


def make_zties(sg: Dict[str, Any], zunits: Dict[int, Dict[str, Any]], tie: bool = True,
               hold: Union[Sequence[str], Dict[str, Sequence[str]]] = ()):
    """The ``extra_residuals`` hook tying each Mastcam-Z bin camera to its eye's global terms (``tie`` False: no
    ties, the globals out of the problem - the diagnostic pass).  Returns (hook, g) with g {eye group: array}; the
    hook's ``ratio_offsets`` {(eye, group): array(1)} are the low-state groups' offsets on a_fx_regular (0.94.4,
    ``sg["ratio_by"]``, default :data:`REGULAR_RATIO_BY`)."""
    import pyceres
    LinearTie = T._linear_tie_class()
    g = {k: v["G"].copy() for k, v in sg["eyes"].items()}
    keep: List[np.ndarray] = []
    by = sg.get("ratio_by") or REGULAR_RATIO_BY
    rgroups = ratio_groups(zunits, by) if tie else {}
    dreg = {k: np.zeros(1) for k in rgroups}
    unit_group = {cid: k for k, cids in rgroups.items() for cid in cids}
    # 0.94.4 (Christian, 26 Oct): the focus approach - one offset per eye (px on fx; fy x the aspect) for the bins
    # whose focus came into place from below, the same focus slope ("a different y-intercept").
    # 0.94.5 / 0.95.0 (Christian, 27 Oct): only when the eye has an fb bin in the high state (not thin) - with every fb
    # bin also in the low state the offset is one with the high / low ratio (Z48 of 27 Oct: -1.5 % and -1.0 %, both
    # bins low state); then no parameter: the offset is 0 and ``hook.focus_below_held`` says why
    fb_eyes = sorted({u["group"] for u in zunits.values() if u.get("fb")})
    fb_high = {eye for eye in fb_eyes if any(u.get("fb") and u["state"] != "regular" and not u.get("thin") and u["group"] == eye
                                             for u in zunits.values())}
    dfb = {eye: np.array([float((sg.get("focus_below_px") or {}).get(eye, 0.0))])
           for eye in fb_eyes if eye in fb_high} if tie else {}
    fb_held = {eye: "held at 0: every focus-from-below bin of this eye is in the low state (the offset is one with "
                    "the high / low ratio)" for eye in fb_eyes if eye not in fb_high} if tie else {}
    # 0.95.0: the sequences' offsets (SEQ_OFFSET_SD, MIN_SEQ_IMAGES, MIN_SEQ_OBS)
    sfree, sinfo = sequence_groups(zunits) if tie else ({}, {})
    dseq = {k: np.array([float(((sg.get("sequence_offsets_px") or {}).get(f"{k[0]}|{k[1]}", 0.0)))]) for k in sfree}
    unit_seq = {cid: k for k, cids in sfree.items() for cid in cids}

    def hook(prob, cam_blocks=None, rig_blocks=None, key_of=None, work=None):
        n = 0
        tie_ids = []                                         # 0.94.4: for the exact-tie covariance
        specs = []                                           # 0.94.4: and the exact-tie solve (mppp.calib.tiesolve)

        def add_tie(A, sig, sizes, bl):
            rid = prob.add_residual_block(LinearTie(A, sig, sizes), None, bl)
            tie_ids.append(rid)
            specs.append({"id": rid, "A": A, "sigma": sig, "sizes": sizes, "blocks": bl})

        has_reg = {k: False for k in g}
        if tie:
            for cid, u in zunits.items():
                if u.get("thin"):                            # 0.94.5: too few observations - not on the lines
                    continue
                blk = cam_blocks.get(int(cid))
                if blk is None or not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                    continue
                eye = u["group"]
                reg = u["state"] == "regular"
                has_reg[eye] |= reg
                ratio = slope_div(sg, eye, reg)
                rows, sig = [], []
                G0 = sg["eyes"][eye]["G"]
                rk = unit_group.get(int(cid))
                nx = 1 if rk is not None else 0              # 0.94.4: the group's own offset on a_fx_regular
                nf = 1 if (u.get("fb") and eye in dfb) else 0    # 0.94.4: the focus-from-below offset
                sk = unit_seq.get(int(cid))
                ns = 1 if sk is not None else 0              # 0.95.0: the sequence's offset
                asp = float(G0[1] / G0[0])
                nextra = nx + nf + ns
                for pi, gi in ((_FX, 2 if reg else 0), (_FY, 2 if reg else 1)):
                    a = np.zeros(12 + NG + nextra); a[pi] = 1.0; a[12 + 4] = -u["dF"] / ratio
                    # 0.93.5: the regular state scales f, not the pixel aspect - its fy is a_fx_regular x the
                    # backlash aspect at the start (22 Oct Z34: a free a_fy_regular gave a regular fy / fx of 0.984)
                    a[12 + gi] = -(asp if (reg and pi == _FY) else 1.0)
                    if nx:
                        a[12 + NG] = -(asp if pi == _FY else 1.0)
                    if nf:
                        a[12 + NG + nx] = -(asp if pi == _FY else 1.0)
                    if ns:
                        a[12 + NG + nx + nf] = -(asp if pi == _FY else 1.0)
                    rows.append(a); sig.append(TIE_SIGMA_Z["f"])
                for pi, gi, s_ in ((_K1, 5, "k"), (_K2, 6, "k"), (_P1, 7, "p"), (_P2, 8, "p")):
                    a = np.zeros(12 + NG + nextra); a[pi] = 1.0; a[12 + gi] = -1.0
                    rows.append(a); sig.append(TIE_SIGMA_Z[s_])
                add_tie(np.array(rows), np.array(sig), [12, NG] + [1] * nextra,
                        [blk, g[eye]] + ([dreg[rk]] if nx else []) + ([dfb[eye]] if nf else []) + ([dseq[sk]] if ns else []))
                if reg:                                      # 0.93.2: cx = pin + d_cx, cy = pin + b_cy dF + d_cy
                    c0 = np.array([u["c_cx"], u["c_cy"]], float)
                    keep.append(c0)
                    a = np.zeros((2, 12 + NG + 2))
                    a[0, _CX] = 1.0; a[0, 12 + 10] = -1.0; a[0, 12 + NG] = -1.0
                    a[1, _CY] = 1.0; a[1, 12 + 9] = -u["dF"]; a[1, 12 + 11] = -1.0; a[1, 12 + NG + 1] = -1.0
                    add_tie(a, np.array([TIE_SIGMA_Z["cy"]] * 2), [12, NG, 2], [blk, g[eye], c0])
                    prob.set_parameter_block_constant(c0)
                elif eye.startswith("ZR"):                   # cy = its pin + b_cy (focus - ref)
                    c0 = np.array([u["c_cy"]], float)
                    keep.append(c0)
                    a = np.zeros((1, 12 + NG + 1)); a[0, _CY] = 1.0; a[0, 12 + 9] = -u["dF"]; a[0, 12 + NG] = -1.0
                    add_tie(a, np.array([TIE_SIGMA_Z["cy"]]), [12, NG, 1], [blk, g[eye], c0])
                    prob.set_parameter_block_constant(c0)
                n += 1
        for (eye, gk), d in dreg.items():                # the groups' offsets drawn to 0 (the eye's ratio their mean)
            if prob.has_parameter_block(d):
                sd_px = REGULAR_RATIO_GROUP_SD * float(sg["eyes"][eye]["G"][2])
                prob.add_residual_block(LinearTie(np.ones((1, 1)), np.array([sd_px]), [1]), None, [d])
        for (eye, sk), d in dseq.items():                # 0.95.0: the sequences' offsets drawn to 0 (the line their mean)
            if prob.has_parameter_block(d):
                sd_px = SEQ_OFFSET_SD * float(sg["eyes"][eye]["G"][0])
                prob.add_residual_block(LinearTie(np.ones((1, 1)), np.array([sd_px]), [1]), None, [d])
        for eye, arr in g.items():                       # 0.94.5: too few low-state images - the ratio held
            r0 = sg["eyes"][eye].get("ratio_hold")
            if tie and r0 and has_reg[eye] and prob.has_parameter_block(arr):
                a = np.zeros((1, NG)); a[0, 2] = 1.0; a[0, 0] = -1.0 / float(r0)
                add_tie(a, np.array([TIE_SIGMA_Z["f"]]), [NG], [arr])
        held: Dict[str, List[int]] = {}
        for eye, arr in g.items():
            if not prob.has_parameter_block(arr):
                continue
            ix = {3}                                         # a_fy_regular: a_fx_regular x the aspect (0.93.5)
            if not has_reg[eye]:
                ix |= {2, 3, 10, 11}                         # no regular-state bin: the regular offsets stay
            elif eye.startswith("ZR"):
                # 0.95.0: ZR's low-state principal point relative to ZL's - observable through the rig in every
                # exposure with ZL in the problem, whatever ZL's state
                if int((sg.get("paired_low") or {}).get(eye, 0)) < MIN_PAIRED_LOW_EXPOSURES:
                    ix |= {10, 11}
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
        cb = {k: v for k, v in g.items() if prob.has_parameter_block(v) and not prob.is_parameter_block_constant(v)}
        cb.update({f"dreg|{eye}|{gk}": d for (eye, gk), d in dreg.items() if prob.has_parameter_block(d)})
        cb.update({f"dfb|{eye}": d for eye, d in dfb.items() if prob.has_parameter_block(d)})
        cb.update({f"dseq|{eye}|{sk}": d for (eye, sk), d in dseq.items() if prob.has_parameter_block(d)})
        return {"cov_blocks": cb, "ties": {"zcam_cameras": n}, "held": held,
                "tie_ids": tie_ids if T.EXACT_TIE_COVARIANCE else [], "tie_specs": specs,
                "exact_solve": bool(EXACT_TIE_SOLVE)}

    hook.ratio_offsets = dreg
    hook.ratio_by = by
    hook.focus_below = dfb
    hook.focus_below_held = fb_held
    hook.sequence_offsets = dseq
    hook.sequence_info = sinfo
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
                   "focus_range": [float(F.min()), float(F.max())],
                   "regular_ratio": 1.0 if is_single_state(g) else 1.0095,   # 0.94.0: the default, not 1
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
    ba["ratio_offsets"] = {k: float(v[0]) for k, v in hook.ratio_offsets.items()}
    ba["ratio_by"] = hook.ratio_by
    ba["focus_below_px"] = {k: float(v[0]) for k, v in hook.focus_below.items()}       # 0.94.4
    ba["focus_below_held"] = dict(hook.focus_below_held)                                # 0.95.0
    sg["focus_below_px"] = dict(ba["focus_below_px"])
    ba["sequence_offsets_px"] = {f"{k[0]}|{k[1]}": float(v[0]) for k, v in hook.sequence_offsets.items()}   # 0.95.0
    ba["sequence_info"] = {f"{k[0]}|{k[1]}": dict(v) for k, v in hook.sequence_info.items()}
    sg["sequence_offsets_px"] = dict(ba["sequence_offsets_px"])
    stats = J.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap)
    return ba, g, stats


WORST_IMAGES = 20                           # 0.95.0: the frames with the largest rms in the joint's report


def _zstats(rec, zunits, images: Optional[Sequence[Dict[str, Any]]] = None, worst: int = WORST_IMAGES) -> Dict[str, Any]:
    """Residual statistics of the Mastcam-Z observations alone (full-frame px) and the Navcam-Mastcam-Z points.
    0.95.0 (Z48 of 27 Oct: rms 1.95 px against a 0.165 px median - a few frames misfit): per image too, and the
    ``worst`` frames by rms (``worst_images``: block, image, sequence, sol, bin, state, rms, observations)."""
    zimg = {int(i) for i in rec.reg_image_ids() if int(rec.images[i].camera_id) in zunits}
    res, mixed = [], 0
    per: Dict[int, List[float]] = {}
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
                d = float(np.linalg.norm(np.asarray(uv)[0] - im.points2D[el.point2D_idx].xy))
                res.append(d)
                per.setdefault(int(el.image_id), []).append(d)
    r = np.asarray(res, float)
    ok = np.isfinite(r)                  # NaN: the point behind the camera / outside the lens model (20 Oct joint)
    f = r[ok]
    meta = {int(x.get("image_id", -1)): x for x in (images or [])}
    rows = []
    for iid, ds in per.items():
        a = np.asarray(ds, float)
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        u = zunits.get(int(rec.images[iid].camera_id)) or {}
        m = meta.get(iid, {})
        rows.append({"image": str(rec.images[iid].name), "block": u.get("scape"), "bin": u.get("key"),
                     "state": u.get("state"), "sequence": m.get("sequence"), "sol": m.get("sol"),
                     "rms_px": float(np.sqrt(np.mean(a ** 2))), "median_px": float(np.median(a)), "observations": int(a.size)})
    rows.sort(key=lambda x: -x["rms_px"])
    return {"zcam_observations": int(r.size), "navcam_zcam_points": int(mixed),
            "zcam_nonfinite": int((~ok).sum()), "zcam_nonfinite_share": float((~ok).mean()) if r.size else 0.0,
            "zcam_median_px": float(np.median(f)) if f.size else None,
            "zcam_rms_px": float(np.sqrt(np.mean(f ** 2))) if f.size else None,
            "zcam_p95_px": float(np.percentile(f, 95)) if f.size else None,
            "zcam_images": len(rows), "worst_images": rows[:int(worst)],
            "images_over_1px": int(sum(1 for x in rows if x["rms_px"] > 1.0))}


def globals_result(sg, g, ba, zunits) -> Dict[str, Any]:
    """The global terms per eye with their sd (the problem's covariance of the global arrays)."""
    cov = ba.get("covariance") or {}
    held = (ba.get("extra") or {}).get("held") or {}
    out: Dict[str, Any] = {"covariance_error": cov.get("error"), "eyes": {},
                           "covariance_method": cov.get("extra_method") or "colmap",     # 0.94.4
                           "covariance_exact": {k: v for k, v in (cov.get("extra_exact") or {}).items() if k != "full"}}
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
                            "paired_low_exposures": int((sg.get("paired_low") or {}).get(eye, 0)),     # 0.95.0
                            "bins": sum(1 for u in zunits.values() if u["group"] == eye and not u.get("thin")),
                            "regular_bins": sum(1 for u in zunits.values() if u["group"] == eye and u["state"] == "regular"
                                                and not u.get("thin")),
                            "thin_bins": sum(1 for u in zunits.values() if u["group"] == eye and u.get("thin")),
                            "low_images": int((sg.get("low_images") or {}).get(eye, 0)),        # 0.94.5
                            "ratio_hold": e.get("ratio_hold"), "ratio_hold_source": e.get("ratio_hold_source")}
    out["regular_ratio_by"] = ba.get("ratio_by") or REGULAR_RATIO_BY
    out["regular_ratio_groups"] = ratio_group_rows(out, ba, zunits)
    out["focus_below"] = focus_below_rows(out, ba, zunits)
    out["sequence_offsets"] = sequence_offset_rows(out, ba, zunits)             # 0.95.0
    for eye, e in out["eyes"].items():
        rows_ = [r for r in out["sequence_offsets"] if r["eye"] == eye]
        fr = [r for r in rows_ if not r.get("held")]
        e["sequences"] = {"free": len(fr), "held": len(rows_) - len(fr),
                          "with_focus_span": sum(1 for r in fr if r["focus_span"] >= 300.0),
                          "sd_ppm": float(np.std([r["ppm"] for r in fr], ddof=1)) if len(fr) > 1 else None,
                          "max_abs_ppm": max((abs(r["ppm"]) for r in fr), default=None),
                          "prior_sd_pct": 100.0 * SEQ_OFFSET_SD}
    return out


def sequence_offset_rows(res: Dict[str, Any], ba: Dict[str, Any], zunits: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """0.95.0: per eye and sequence (block, sol, sequence) the focus offset (px on fx at the same focus count), its sd,
    z, ppm of the eye's high-state f, the frames, bins, block observations, focus span and states; the sequences held
    at 0 with their reason."""
    cov = ba.get("covariance") or {}
    rows = []
    for key, info in sorted((ba.get("sequence_info") or {}).items()):
        eye, sk = key.split("|", 1)
        e = res["eyes"].get(eye)
        f0 = float(e["terms"]["a_fx_backlash"]) if e else None
        parts = sk.split("|")
        block, sol, seq = parts[0], parts[1][1:] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None
        d = (ba.get("sequence_offsets_px") or {}).get(key)
        row = {"eye": eye, "key": sk, "block": block, "sol": (int(sol) if sol and sol != "None" else None), "sequence": seq,
               "images": info["images"], "bins": info["bins"], "observations": info["observations"],
               "focus_span": info["focus_span"], "states": info["states"], "held": info.get("held")}
        if d is None:
            row.update({"offset_px": 0.0, "sd_px": None, "z": None, "ppm": 0.0, "sd_ppm": None})
        else:
            sd = T._sd(cov, f"dseq|{key}", 1)
            s_px = float(sd[0]) if sd is not None and np.isfinite(sd[0]) else None
            row.update({"offset_px": float(d), "sd_px": s_px, "z": (float(d) / s_px) if s_px else None,
                        "ppm": (1e6 * float(d) / f0) if f0 else None, "sd_ppm": (1e6 * s_px / f0) if (f0 and s_px) else None})
        rows.append(row)
    return rows


def focus_below_rows(res: Dict[str, Any], ba: Dict[str, Any], zunits: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """0.94.4 (Christian, 26 Oct): per eye the focus-from-below offset (px on fx at the same focus count), its sd,
    z, the offset in ppm of the backlash f, and the bins it rests on (also on the low list or not)."""
    cov = ba.get("covariance") or {}
    rows = []
    for eye, d in sorted((ba.get("focus_below_px") or {}).items()):
        e = res["eyes"].get(eye)
        sd = T._sd(cov, f"dfb|{eye}", 1)
        s_px = float(sd[0]) if sd is not None and np.isfinite(sd[0]) else None
        f0 = float(e["terms"]["a_fx_backlash"]) if e else None
        us = [u for u in zunits.values() if u["group"] == eye and u.get("fb")]
        rows.append({"eye": eye, "offset_px": float(d), "sd_px": s_px, "z": (float(d) / s_px) if s_px else None,
                     "ppm": (1e6 * float(d) / f0) if f0 else None, "sd_ppm": (1e6 * s_px / f0) if (f0 and s_px) else None,
                     "bins": len(us), "bins_also_low": sum(1 for u in us if u["state"] == "regular"),
                     "blocks": sorted({u["scape"] for u in us})})
    for eye, why in sorted((ba.get("focus_below_held") or {}).items()):       # 0.95.0: held at 0, for the record
        us = [u for u in zunits.values() if u["group"] == eye and u.get("fb")]
        rows.append({"eye": eye, "offset_px": 0.0, "sd_px": None, "z": None, "ppm": 0.0, "sd_ppm": None, "held": why,
                     "bins": len(us), "bins_also_low": sum(1 for u in us if u["state"] == "regular"),
                     "blocks": sorted({u["scape"] for u in us})})
    return rows


def ratio_group_rows(res: Dict[str, Any], ba: Dict[str, Any], zunits: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """0.94.4: per eye and low-state group the high / low ratio (a_fx_backlash / (a_fx_regular + offset)), its sd
    and z against the eye's ratio, the group's bins and sols."""
    cov = ba.get("covariance") or {}
    by = ba.get("ratio_by") or REGULAR_RATIO_BY
    rows = []
    for (eye, gk), d in sorted((ba.get("ratio_offsets") or {}).items()):
        e = res["eyes"].get(eye)
        if e is None:
            continue
        tm = e["terms"]
        sd = T._sd(cov, f"dreg|{eye}|{gk}", 1)
        s_px = float(sd[0]) if sd is not None and np.isfinite(sd[0]) else None
        a_r = float(tm["a_fx_regular"])
        ratio, ratio0 = float(tm["a_fx_backlash"]) / (a_r + d), float(tm["a_fx_backlash"]) / a_r
        sd_ratio = (ratio * s_px / (a_r + d)) if s_px else None
        us = [u for u in zunits.values() if u["group"] == eye and u["state"] == "regular" and ratio_group(u, by) == gk]
        rows.append({"eye": eye, "group": gk, "offset_px": float(d), "sd_px": s_px, "ratio": ratio,
                     "sd_ratio": sd_ratio, "eye_ratio": ratio0, "z": (float(d) / s_px) if s_px else None,
                     "bins": len(us), "sols": sorted({u.get("sol") for u in us if u.get("sol") is not None})})
    return rows


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
        ratio = slope_div(sg, u["group"], reg)
        dfb_ = float((sg.get("focus_below_px") or {}).get(u["group"], 0.0)) if u.get("fb") else 0.0
        # 0.95.0: plus the bin's sequence offset (else the per-block lines absorb the offsets as aspect / level)
        dsq_ = float((sg.get("sequence_offsets_px") or {}).get(f"{u['group']}|{u['seq']}", 0.0)) if u.get("seq") else 0.0
        p[_FX] = G[2 if reg else 0] + G[4] * u["dF"] / ratio + dfb_ + dsq_
        p[_FY] = G[3 if reg else 1] + G[4] * u["dF"] / ratio + (dfb_ + dsq_) * float(G[1] / G[0])
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
                        bG = float(g[eye][4]) / slope_div(sg, eye, stt == "regular")
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


def set_ratio_holds(sg: Dict[str, Any], zoom: int, verbose: bool = True) -> Dict[str, float]:
    """0.94.5: the eyes with low-state bins but fewer than :data:`MIN_LOW_IMAGES` low-state images get
    ``sg["eyes"][eye]["ratio_hold"]`` - the 34 mm model's ratio of that side (``ZL034`` for ``ZL048``), else
    :data:`mppp.calib.fit.REGULAR_RATIO_DEFAULT`; the hook ties a_fx_regular = a_fx_backlash / that ratio."""
    from .fit import REGULAR_RATIO_DEFAULT
    out = {}
    if int(zoom) == RATIO_HOLD_ZOOM:
        return out
    ref = None
    try:
        from ..sfm.project import zcam_focus_model_path
        f = zcam_focus_model_path(zoom=RATIO_HOLD_ZOOM)
        ref = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None
    except Exception:                                # noqa: BLE001
        ref = None
    for eye, e in sg["eyes"].items():
        n = int((sg.get("low_images") or {}).get(eye, 0))
        if is_single_state(eye) or n >= MIN_LOW_IMAGES:     # (n == 0 too: no measurement - the 34 mm ratio)
            continue
        side = f"{eye[:2]}{RATIO_HOLD_ZOOM:03d}"
        r = ((ref or {}).get("cameras") or {}).get(side, {}).get("regular_ratio")
        e["ratio_hold"] = float(r) if r else float(REGULAR_RATIO_DEFAULT)
        e["ratio_hold_source"] = (f"the {RATIO_HOLD_ZOOM} mm model ({side})" if r else "the default")
        out[eye] = e["ratio_hold"]
        if verbose:
            log.info(f"Mastcam-Z joint: {eye} has {n} low-state images (< {MIN_LOW_IMAGES}): its high / low ratio held "
                     f"at {e['ratio_hold']:.4f} ({e['ratio_hold_source']})")
    return out


def navcam_block_aspect(scapes: Sequence[Any], st: Dict[str, Any], nav_points: Optional[int] = NAV_POINTS_PER_BLOCK,
                        verbose: bool = True, min_images: int = 8) -> Dict[str, Dict[str, Any]]:
    """0.94.5: per block the Navcam's dfy - dfx (ppm about the law in use): the Navcam diagnostic pass
    (:func:`mppp.calib.tied.diagnostic_pass`, one camera per block, eye and temperature bin, fx, fy, k1 free) over the
    blocks' Navcam frames, the bins with >= ``min_images`` images, weighted by images.  Returns {block: {"NL", "NR",
    "mean" (ppm), "images"}}."""
    import pycolmap
    cams = {k: pycolmap.Camera(model=st["cameras"][k]["model"], width=5120, height=3840,
                               params=np.asarray(st["cameras"][k]["params"], float)) for k in ("NL", "NR")}
    rec0, proj0, idx = J.merge_scapes(list(scapes), cams, (st["R"], st["t"]), points_per_scape=nav_points, seed=0)
    g = {"G_NL": st["G"]["NL"].copy(), "G_NR": st["G"]["NR"].copy(), "S_NL": st["S"]["NL"].copy(),
         "S_NR": st["S"]["NR"].copy(), "E_NL": st["E"]["NL"].copy(), "E_NR": st["E"]["NR"].copy(),
         "Q0": st["Q0"].copy(), "D": st["D"].copy(), "H": st["H"].copy()}
    d = T.diagnostic_pass(rec0, proj0, idx, st, g, verbose=verbose, pointing=False)
    return navcam_aspect_by_block(d.get("cameras") or [], min_images=min_images)


def navcam_aspect_by_block(cameras: Sequence[Dict[str, Any]], min_images: int = 8) -> Dict[str, Dict[str, Any]]:
    """{block: {"NL", "NR", "mean", "images"}}: the image-weighted dfy - dfx (ppm) of the diagnostic units."""
    acc: Dict[str, Dict[str, List[Tuple[float, int]]]] = {}
    for c in cameras:
        if not c.get("scape") or int(c.get("images", 0)) < min_images:
            continue
        acc.setdefault(str(c["scape"]), {}).setdefault(c["eye"], []).append(
            (float(c["dfy_ppm"]) - float(c["dfx_ppm"]), int(c["images"])))
    out = {}
    for b, eyes in acc.items():
        row: Dict[str, Any] = {"images": int(sum(n for v in eyes.values() for _, n in v))}
        for eye, v in eyes.items():
            row[eye] = float(np.average([x for x, _ in v], weights=[n for _, n in v]))
        vals = [row[e] for e in ("NL", "NR") if e in row]
        row["mean"] = float(np.mean(vals)) if vals else None
        out[b] = row
    return out


def block_aspect_rows(res: Dict[str, Any], nav: Dict[str, Dict[str, Any]],
                      min_obs: int = BLOCK_ASPECT_MIN_OBS) -> Dict[str, Any]:
    """0.94.5 (Christian, 27 Oct): per block and eye, the Mastcam-Z fy - fx of the diagnostic pass (the state line
    with the most observations, at the reference focus) about the joint's (ppm of fx and px) against the Navcam's
    per-block dfy - dfx (``nav``, :func:`navcam_block_aspect`).  Rows with fewer than ``min_obs`` observations are
    listed but not judged.  Returns {"rows", "slope" (Mastcam-Z on Navcam ppm over the judged rows), "max_diff_ppm"}."""
    rows = []
    for b in (res.get("diagnostics") or {}).get("blocks") or []:
        for eye, states in (b.get("lines") or {}).items():
            best = max(states.items(), key=lambda kv: int((kv[1].get("fx") or {}).get("observations", 0)), default=None)
            if not best or "fx" not in best[1] or "fy" not in best[1]:
                continue
            stt, ln = best
            tm = ((res.get("eyes") or {}).get(eye) or {}).get("terms") or {}
            if not tm.get("a_fx_backlash"):
                continue
            asp = float(tm["a_fy_backlash"]) / float(tm["a_fx_backlash"])
            fx, fy = float(ln["fx"]["a"]), float(ln["fy"]["a"])
            z_ppm = 1e6 * (fy / fx - asp)
            n = nav.get(b["block"]) or {}
            nv = n.get("mean")
            obs = int(ln["fx"].get("observations", 0))
            rows.append({"block": b["block"], "eye": eye, "state": stt, "observations": obs,
                         "zcam_dfy_minus_dfx_px": float((fy - fx) - fx * (asp - 1.0)), "zcam_ppm": float(z_ppm),
                         "navcam_ppm": None if nv is None else float(nv), "navcam_NL_ppm": n.get("NL"),
                         "navcam_NR_ppm": n.get("NR"), "diff_ppm": None if nv is None else float(z_ppm - nv),
                         "judged": bool(obs >= min_obs and nv is not None)})
    J_ = [r for r in rows if r["judged"]]
    slope = None
    if len(J_) >= 3 and np.ptp([r["navcam_ppm"] for r in J_]) > 0:
        slope = float(np.polyfit([r["navcam_ppm"] for r in J_], [r["zcam_ppm"] for r in J_], 1)[0])
    return {"rows": rows, "slope": slope,
            "max_diff_ppm": max((abs(r["diff_ppm"]) for r in J_), default=None), "min_observations": int(min_obs)}


# ====================================================================================== the run and the files
def fit_zjoint(blocks: Dict[str, PathLike], model: Optional[Dict[str, Any]] = None, navcam_dir: Optional[PathLike] = None,
               zoom: int = ZOOM, nav_points: int = NAV_POINTS_PER_BLOCK, z_points: int = Z_POINTS_PER_BLOCK,
               iterations: int = 100, passes: int = 2, diagnostics: bool = True, verbose: bool = True,
               scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None,
               k_tests: bool = True, ratio_by: Optional[str] = None, navcam_aspect: bool = True) -> Dict[str, Any]:
    """The Mastcam-Z joint over ``blocks`` {name: WORK}: build, adjust, the distortion tests (:func:`distortion_tests`,
    ``k_tests``), the diagnostic pass.  ``model``: the focus model to start from (default: the one in use);
    ``ratio_by``: "global" | "block" | "sol", the low-state groups with their own high / low ratio (default
    :data:`REGULAR_RATIO_BY`)."""
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
    if scapes is None and diagnostics and navcam_aspect:    # 0.94.5: kept for the Navcam's per-block aspect
        scapes = [J.load_scape(n, w, single=False) for n, w in blocks.items()]
    rec, proj, idx, st, units, zunits, sg = build(blocks, model, navcam_dir, zoom, nav_points, z_points, scapes=scapes, st=st)
    if diagnostics and navcam_aspect:
        sg["_scapes"] = scapes
    if verbose:
        log.info(f"Mastcam-Z joint: {len(blocks or scapes or [])} blocks, {rec.num_reg_images()} images, "
                 f"{rec.num_points3D()} points; {len(zunits)} Mastcam-Z bins, {len(units['cameras'])} Navcam bin "
                 f"cameras (held) ({time.time() - t:.0f} s)")
    if not zunits:
        raise ValueError(f"no Mastcam-Z {zoom} mm bins in the blocks")
    sg["ratio_by"] = ratio_by or REGULAR_RATIO_BY
    if ratio_by not in (None, "global", "block", "sol"):
        raise ValueError(f"ratio_by {ratio_by!r}: global, block or sol")
    set_ratio_holds(sg, zoom, verbose=verbose)       # 0.94.5
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
    res["low_cameras_not_on_list"] = list(sg.get("low_cameras_not_on_list") or [])
    res["thin_bins"] = list(sg.get("thin_bins") or [])          # 0.94.5
    res["min_bin_observations"] = int(MIN_BIN_OBSERVATIONS)
    res["found_list"] = sg.get("found_list")
    res.update({"zoom": zoom, "blocks": sorted({u["scape"] for u in zunits.values()}), "residuals": stats,
                "zcam": _zstats(rec, zunits, proj.images), "rig": rig_result(proj, sg, zoom), "final_cost": float(ba["final_cost"]),
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
        if navcam_aspect:                            # 0.94.5: the per-block aspect against the Navcam's
            try:
                nav = navcam_block_aspect(sg.get("_scapes") or [], st, nav_points, verbose=verbose)
                res["block_aspect"] = block_aspect_rows(res, nav)
            except Exception as e:                   # noqa: BLE001 - a diagnostic: the joint stands without it
                log.warning(f"Mastcam-Z joint: the per-block aspect against the Navcam did not run ({type(e).__name__}: {e})")
                res["block_aspect"] = {"error": f"{type(e).__name__}: {e}"}
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
                   + (f"; high / low ratio held at {e['ratio_hold']:.4f} ({e.get('ratio_hold_source')}: {e.get('low_images', 0)} "
                      f"low-state images < {MIN_LOW_IMAGES})" if e.get("ratio_hold") else "")
                   + (f"; {e['thin_bins']} bins with < {MIN_BIN_OBSERVATIONS} observations left out" if e.get("thin_bins") else "")
                   + ("" if "d_cx_regular" in e.get("held", []) else
                      f"; low-state principal point {f('d_cx_regular')}, {f('d_cy_regular')} px from the high "
                      f"state's (" + (f"{e.get('paired_low_exposures', 0)} low-state exposures paired with ZL"
                                      if eye.startswith("ZR") else
                                      f"{e.get('mixed_exposures', 0)} exposures with the other eye in the high state") + ")"))
    for r_ in res.get("focus_below") or []:          # 0.94.4: the focus approach
        if r_.get("held"):                           # 0.95.0
            out.append(f"{r_['eye']} focus from below: {r_['held']}; {r_['bins']} bins ({r_['bins_also_low']} low state) in "
                       + ", ".join(r_["blocks"]))
            continue
        sdp = f" +- {r_['sd_px']:.2f}" if r_.get("sd_px") else ""
        out.append(f"{r_['eye']} focus from below: fx {r_['offset_px']:+.2f}{sdp} px"
                   + (f" ({r_['ppm']:+.0f} ppm, z {r_['z']:+.1f})" if r_.get("z") is not None and r_.get("ppm") is not None else "")
                   + f" at the same focus count; {r_['bins']} bins ({r_['bins_also_low']} also low state) in "
                   + ", ".join(r_["blocks"]))
    for eye, e in res["eyes"].items():               # 0.95.0: the sequences' offsets
        sm = e.get("sequences") or {}
        if not (sm.get("free") or sm.get("held")):
            continue
        fr = [r for r in res.get("sequence_offsets") or [] if r["eye"] == eye and not r.get("held")]
        big = sorted(fr, key=lambda r: -abs(r["ppm"] or 0.0))[:3]
        out.append(f"{eye} sequence offsets: {sm['free']} fitted ({sm.get('with_focus_span', 0)} with a focus span >= 300 "
                   f"counts - the slope's support), {sm['held']} held at 0"
                   + (f"; sd {sm['sd_ppm']:.0f} ppm" if sm.get("sd_ppm") is not None else "")
                   + (", largest " + ", ".join(f"{r['block']} {r['sequence']} {r['ppm']:+.0f} ppm" + (f" (z {r['z']:+.1f})" if r.get("z") is not None else "")
                                               for r in big) if big else ""))
    for r_ in res.get("regular_ratio_groups") or []:
        out.append(f"{r_['eye']} high / low ratio of {r_['group']} ({r_['bins']} low-state bins, sols "
                   f"{', '.join(map(str, r_['sols'])) or '-'}): {r_['ratio']:.4f}"
                   + (f" +- {r_['sd_ratio']:.4f}" if r_.get("sd_ratio") else "")
                   + f" (the eye's {r_['eye_ratio']:.4f}" + (f", z {r_['z']:+.1f})" if r_.get("z") is not None else ")"))
    if res.get("low_cameras_not_on_list"):
        out.append(f"left out: {len(res['low_cameras_not_on_list'])} low-state cameras with frames not on the shipped "
                   f"or found low-state list" + (f" ({res['found_list']})" if res.get("found_list") else " (no found list)"))
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
    ba_ = res.get("block_aspect") or {}
    if ba_.get("error"):
        out.append(f"per-block aspect vs the Navcam: not run ({ba_['error']})")
    for r_ in ba_.get("rows") or []:                 # 0.94.5 (Christian, 27 Oct)
        nv = "-" if r_.get("navcam_ppm") is None else f"{r_['navcam_ppm']:+.0f}"
        out.append(f"{r_['block']} {r_['eye']} ({r_['state']} state, {r_['observations']} obs): Mastcam-Z d(fy - fx) "
                   f"{r_['zcam_dfy_minus_dfx_px']:+.2f} px ({r_['zcam_ppm']:+.0f} ppm), Navcam {nv} ppm"
                   + (f", difference {r_['diff_ppm']:+.0f} ppm" if r_.get("diff_ppm") is not None else "")
                   + ("" if r_["judged"] else " (not judged)"))
    if ba_.get("slope") is not None:
        out.append(f"per-block aspect: Mastcam-Z on Navcam slope {ba_['slope']:+.2f} over the judged block lines")
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
        if e.get("ratio_hold"):                       # 0.94.5: too few (or no) low-state images
            ratio = float(e["ratio_hold"])
            ratio_note = (f"held at {ratio:.4f} ({e.get('ratio_hold_source') or 'the 34 mm model'}): "
                          f"{int(e.get('low_images') or 0)} low-state images (< {MIN_LOW_IMAGES})")
        elif "a_fx_regular" in e["held"]:
            ratio = float(e["ratio_start"])
            if abs(ratio - 1.0) < 1e-9 and not single:
                # 0.94.0: a first calibration without low-state bins (its start from the blocks had ratio 1): the
                # default high / low ratio, not "no state change" (Z63 of 23 Oct failed its ratio check on 1.0000)
                from .fit import REGULAR_RATIO_DEFAULT
                ratio = float(REGULAR_RATIO_DEFAULT)
                ratio_note = (f"no low-state bins: the default high / low ratio {ratio:.4f} (no measurement)")
        else:                                         # 0.93.5: fx only (fy follows the aspect)
            ratio = float(tm["a_fx_backlash"] / tm["a_fx_regular"])
            lo, hi = REGULAR_RATIO_RANGE
            if not lo <= ratio <= hi:                 # not a focus state: the start ratio kept
                ratio_note = (f"fitted regular ratio {ratio:.4f} outside {lo:g}..{hi:g} (the states are ~0.95 % apart): "
                              f"the start ratio {float(e['ratio_start']):.4f} kept")
                ratio = float(e["ratio_start"])
                if abs(ratio - 1.0) < 1e-9:           # a start without a ratio: the default
                    from .fit import REGULAR_RATIO_DEFAULT
                    ratio = float(REGULAR_RATIO_DEFAULT)
        bk = {"fx": {"a": tm["a_fx_backlash"], "b": tm["b"], "sd_a": sdv("a_fx_backlash"), "sd_b": sdv("b"), "form": "common"},
              "fy": {"a": tm["a_fy_backlash"], "b": tm["b"], "sd_a": sdv("a_fy_backlash"), "sd_b": sdv("b"), "form": "common"}}
        for k in ("k1", "k2", "p1", "p2"):
            bk[k] = {"a": tm[k], "b": 0.0, "sd_a": sdv(k), "form": "constant"}
        pp0 = list(e.get("pp0_px") or gm.get("pp0_px") or [])
        if eye.startswith("ZR") and len(pp0) == 2:
            bk["cx"] = {"a": float(pp0[0]), "b": 0.0, "form": "pinned"}
            bk["cy"] = {"a": float(pp0[1]), "b": tm["b_cy"], "sd_b": sdv("b_cy"), "form": "pinned_slope"}
        sdiv = 1.0 if REGULAR_SAME_SLOPE else ratio    # 0.94.5: the low state on the high state's slope
        reg = {pn: {"a": v["a"] / (ratio if pn in ("fx", "fy") else 1.0), "b": v["b"] / (sdiv if pn in ("fx", "fy") else 1.0),
                    "derived": ((f"backlash a / {ratio:.5f}, the backlash slope" if REGULAR_SAME_SLOPE
                                 else f"backlash / {ratio:.5f}") if pn in ("fx", "fy") else "backlash")}
               for pn, v in bk.items()}
        # 0.93.2: the regular state's own principal point (pp0 + the offset; the pose is shared with the backlash state)
        doff = [float(tm.get("d_cx_regular", 0.0)), float(tm.get("d_cy_regular", 0.0))]
        off_held = "d_cx_regular" in e["held"]
        if len(pp0) == 2:
            reg["cx"] = {"a": float(pp0[0]) + doff[0], "b": 0.0, "derived": "pp0 + the regular-state offset"}
            reg["cy"] = {"a": float(pp0[1]) + doff[1], "b": float(tm["b_cy"]) if eye.startswith("ZR") else 0.0,
                         "derived": "pp0 + the regular-state offset" + (" (and the cy slope)" if eye.startswith("ZR") else "")}
        gm["regular_pp_offset_px"] = doff
        right = eye.startswith("ZR")
        gm["regular_pp_offset"] = {"sd_px": [sdv("d_cx_regular"), sdv("d_cy_regular")], "held": off_held,
                                   "mixed_exposures": int(e.get("mixed_exposures") or 0),
                                   "paired_low_exposures": int(e.get("paired_low_exposures") or 0),     # 0.95.0
                                   "note": (("held: fewer than %d exposures with this eye in the low state and ZL in the "
                                             "problem" % MIN_PAIRED_LOW_EXPOSURES) if right else
                                            ("held: fewer than %d exposures with this eye regular and the other in the "
                                             "backlash state" % MIN_MIXED_EXPOSURES)) if off_held else
                                           (("fitted: ZR's low-state principal point minus its high state's, relative "
                                             "to ZL's (the reference eye; the pose shared through the rig)") if right else
                                            "fitted: the regular state's principal point minus the backlash state's "
                                            "(the pose shared: the other eye and the rig fix it)")}
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
        fbr = [x for x in res.get("focus_below") or [] if x["eye"] == eye]
        gm.pop("focus_below_offset_px", None)
        gm.pop("focus_below", None)
        if fbr and fbr[0].get("held"):               # 0.95.0: no high-state fb bin - no offset (0), the reason kept
            gm["focus_below"] = {k: fbr[0][k] for k in ("bins", "bins_also_low", "blocks", "held")}
            gm["focus_below"]["note"] = "no focus-from-below offset: " + str(fbr[0]["held"])
        elif fbr:                                     # 0.94.4: the focus approach (px on fx, same slope)
            gm["focus_below_offset_px"] = float(fbr[0]["offset_px"])
            gm["focus_below"] = {k: fbr[0][k] for k in ("sd_px", "z", "ppm", "bins", "bins_also_low", "blocks")}
            gm["focus_below"]["note"] = ("the bins whose focus came into place from below (the focus-approach list): "
                                         "fx + offset, fy + offset x aspect, at the same focus count")
        sq = [x for x in res.get("sequence_offsets") or [] if x["eye"] == eye]
        gm.pop("sequence_offsets", None)
        if sq:                                        # 0.95.0: the sequences' offsets, for the record (block facts,
            sm = dict(e.get("sequences") or {})       # not model terms: the line is the eye's)
            gm["sequence_offsets"] = {
                "note": ("each sequence (block, sol, sequence) sits at its own offset on the eye's line (px on fx at the "
                         "same focus count, fy x aspect), drawn to 0 with sd %.1f %% of f; the line is the sequences' "
                         "weighted mean; sequences under %d frames or %d observations held at 0"
                         % (100 * SEQ_OFFSET_SD, MIN_SEQ_IMAGES, MIN_SEQ_OBS)),
                "free": sm.get("free"), "held": sm.get("held"), "with_focus_span": sm.get("with_focus_span"),
                "sd_ppm": sm.get("sd_ppm"), "max_abs_ppm": sm.get("max_abs_ppm"),
                "rows": [{k: x[k] for k in ("block", "sol", "sequence", "offset_px", "sd_px", "ppm", "images", "bins",
                                            "focus_span", "states", "held")} for x in sq]}
        grp = [x for x in res.get("regular_ratio_groups") or [] if x["eye"] == eye]
        gm.pop("regular_ratio_groups", None)
        if grp:                                       # 0.94.4: the groups' own ratios, for the record
            gm["regular_ratio_groups"] = {"by": res.get("regular_ratio_by"),
                                          "groups": {x["group"]: {k: x[k] for k in ("ratio", "sd_ratio", "bins", "sols")}
                                                     for x in grp},
                                          "note": "regular_ratio is the groups' mean; each low-state group's own ratio "
                                                  "(offset drawn to 0 with sd %.1f %%)" % (100 * REGULAR_RATIO_GROUP_SD)}
        if single:                                    # 0.93.5: 110 mm has one state - no regular state at all
            gm["lines"].pop("regular", None)
            for k_ in ("regular_ratio", "regular_pp_offset_px", "regular_pp_offset", "regular_ratio_note",
                       "regular_ratio_groups"):
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
ZJOINT_LIMITS = {"max_zcam_median_px": 0.6, "max_zcam_nonfinite_share": 0.005,
                 "max_zcam_rms_px": 1.5,
                 "max_block_aspect_vs_navcam_ppm": 3000.0}             # 0.94.4: Z34 0.80, Z63 0.42; Z48 of 27 Oct 3.30, Z79 of 26 Oct 29.6


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
    rms = z.get("zcam_rms_px")
    if rms is not None:                                # (results before 0.94 have no rms)
        okr = np.isfinite(rms) and rms <= lim["max_zcam_rms_px"]
        wi = z.get("worst_images") or []               # 0.95.0: the frames behind a failed rms
        checks.append({"check": "Mastcam-Z residual rms", "ok": bool(okr),
                       "note": f"rms {rms:.3f} px (<= {lim['max_zcam_rms_px']:g})"
                               + (f"; {z.get('images_over_1px', 0)} of {z.get('zcam_images', 0)} frames over 1 px, worst "
                                  + ", ".join(f"{w['image']} ({w['block']}, {w.get('sequence') or '-'}, {w['rms_px']:.2f} px)"
                                              for w in wi[:3]) if (wi and not okr) else "")})
    checks.append({"check": "Mastcam-Z residuals finite", "ok": share <= lim["max_zcam_nonfinite_share"],
                   "note": f"{int(z.get('zcam_nonfinite') or 0)} of {int(z.get('zcam_observations') or 0)} not finite "
                           f"({100 * share:.2f} %)"})
    ba_ = res.get("block_aspect") or {}
    if ba_.get("rows"):                                # 0.94.5: per-block fy - fx against the Navcam's
        bad = [r for r in ba_["rows"] if r["judged"] and abs(r["diff_ppm"]) > lim["max_block_aspect_vs_navcam_ppm"]]
        nj = sum(1 for r in ba_["rows"] if r["judged"])
        checks.append({"check": "per-block aspect vs the Navcam", "ok": not bad,
                       "note": (f"{nj} block lines judged (>= {ba_.get('min_observations')} observations), largest "
                                f"|Mastcam-Z - Navcam| d(fy - fx) {ba_['max_diff_ppm']:.0f} ppm" if nj else "no line judged")
                               + f" (<= {lim['max_block_aspect_vs_navcam_ppm']:g})"
                               + (": " + ", ".join(f"{r['block']} {r['eye']} {r['diff_ppm']:+.0f}" for r in bad) if bad else "")})
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


def write_zjoint(res: Dict[str, Any], out_dir: PathLike, model: Optional[Dict[str, Any]] = None,
                 limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    # 0.93.9: the result first - a check that fails to run (Z48 of 23 Oct: a rig scatter of one block) must not lose
    # hours of adjustment; it is then a failed check, nothing is promoted
    (out / "zcam_joint.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    try:
        chk = joint_checks(res, check_zcam(new, old, limits))
    except Exception as e:                                   # noqa: BLE001
        log.exception("Mastcam-Z joint: the checks did not run")
        chk = {"ok": False, "checks": [{"check": "checks ran", "ok": False, "note": f"{type(e).__name__}: {e}"}]}
    (out / "check.json").write_text(json.dumps(chk, indent=1, default=float), encoding="utf-8")
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
    sq = res.get("sequence_offsets") or []
    if sq:                                                   # 0.95.0: the sequences' focus offsets
        lines += ["", "## Sequence offsets (px on fx at the same focus count; drawn to 0 with sd "
                      f"{100 * SEQ_OFFSET_SD:.1f} % of f)", "",
                  "| eye | block | sol | sequence | offset px | sd | ppm | z | frames | bins | obs | focus span | states |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(sq, key=lambda x: (x["eye"], x["block"], x["sol"] or 0, x["sequence"] or "")):
            lines.append(f"| {r['eye']} | {r['block']} | {r['sol'] if r['sol'] is not None else '-'} | {r['sequence'] or '-'} | "
                         + (str(r["held"]) if r.get("held") else
                            f"{r['offset_px']:+.2f} | {r['sd_px']:.2f} | {r['ppm']:+.0f} | {r['z']:+.1f}" if r.get("sd_px")
                            else f"{r['offset_px']:+.2f} | - | {r['ppm']:+.0f} | -")
                         + (" | | |" if r.get("held") else "")
                         + f" | {r['images']} | {r['bins']} | {r['observations']} | {r['focus_span']:.0f} | {', '.join(r['states'])} |")
    wi = (res.get("zcam") or {}).get("worst_images") or []
    if wi:                                                   # 0.95.0: the frames with the largest rms
        z = res["zcam"]
        lines += ["", f"## Worst frames ({z.get('images_over_1px', 0)} of {z.get('zcam_images', 0)} over 1 px rms)", "",
                  "| image | block | sequence | sol | bin | state | rms px | median px | obs |", "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {w['image']} | {w['block']} | {w.get('sequence') or '-'} | {w.get('sol') if w.get('sol') is not None else '-'} | "
                  f"{w['bin']} | {w['state']} | {w['rms_px']:.2f} | {w['median_px']:.2f} | {w['observations']} |" for w in wi]
    lines += [""] + [f"![{f}]({f})" for f in figs]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"folder": str(out), "model": str(out / name), "check": chk, "figures": figs, "block_lines": rows}
