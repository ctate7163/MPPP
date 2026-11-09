"""
The Mastcam-Z joint (MPPP 0.91, Christian 20 Oct 2026): one bundle adjustment over the finished Navcam + Mastcam-Z
blocks of a zoom, with the Navcam held at the models in use and the Mastcam-Z focus model fitted.

* **Navcam held** - every Navcam image's camera is the promoted law at its temperature and era (one camera per eye,
  2 degC bin and era, :func:`mppp_studies.calibration.tied.tied_reconstruction`; within a bin the image's own temperature through
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
  lines and the per-block rig rotations about the joint - the sol and temperature trends of the rig.  0.96: a pooled
  pass as well (one fx scale, one fy scale and cx, cy per eye and sequence, the frames on the line at their own
  focus), which the per-block aspect gate reads; the residual direction per sequence from the final pass; a warning
  when the two eyes of a sequence sit in opposite backlash states; f0's sd with the sequence scatter folded in.

The result is a candidate focus model in the zcam-v2 form (``write_zjoint``), checked with
:func:`mppp_studies.calibration.fit.check_zcam` against the one in use, promoted on request.
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
from mppp.calib import forms
from mppp.calib.forms import is_single_state, zcam_focus_below
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
# 0.95.1: the start offset of a freed fixed term in the "free (vs the fixed values)" test: from the exactly tied
# solution with the bins at the fixed value the adjustment did not move it at all (k2 0.5 -> 0.5000013, synthetic
# 0.3 -> 0.3 with the truth at 0.40); the bins' data then pull the law to the free optimum
FREED_TERM_NUDGE = {"k1": 0.02, "k2": 0.05, "p1": 1e-4, "p2": 1e-4}
EYE_SLOPE_TIE_SD = 1e-7                   # px/count: the eye-slope tie of eye_slope_tests (exact)
# 0.95.0 (Christian, 29 Oct): the joint refines the Mastcam-Z rig translation (1 cm prior per axis on the start rig's
# right centre) instead of holding it at the CAHV value; "rotation" holds it again (the 0.94 joint)
ZJOINT_RIG_FIT = "full"
# 0.95.0 (Christian, 29 Oct: "fix the Zcam rig drift ... put this into an early mission principal point offset if it
# is significant"): the Z34 rig pitch per block was -28 and -36 mdeg at Chal Rocks (sol 102) and Seitah Overlook
# (119) and within +-7 mdeg from Rochette (182) on - a step, not a drift.  ZR's principal point gets one offset
# (cx, cy) for the bins before this sol (ZL is the reference eye); kept where |z| >= EARLY_Z_MIN
ZCAM_EARLY_SOL = 150
ZCAM_EARLY_MIN_BINS = 3                   # fewer early ZR bins on the lines: no offset (held at 0)
EARLY_Z_MIN = 3.0
REGULAR_RATIO_GROUP_SD = 0.003
TIE_SIGMA_Z = {"f": 1e-3, "k": 1e-7, "p": 1e-8, "cy": 1e-3}       # px; unitless - the ties are effectively exact
# 0.96 (Christian, 5 Nov: "yes to pooling on the block or sequence level"; the Z34 joint of 4 Nov failed only the
# per-block aspect gate on Chal Rocks large, where one 6-row pan, ZCAM08401, makes each row its own 30-count focus
# bin: fx is held by the side-by-side overlap within the row, fy only by the rows above and below, which are other
# bins, and the rows' fy drifted together by -1.5 to -1.8 %).  The diagnostic pass gets a pooled model as well: one
# pool per (eye, sequence) - DIAG_POOL "sequence" - or per (eye, block, backlash state) - "block"; None: the per-bin
# pass alone.  Every frame of a pool follows the joint's line at its own focus count (the bin's law camera, the
# frame's keypoints scaled by the line from the bin's focus to the frame's); per pool, free: one scale on fx, one on
# fy (so f and fy / fx) and cx, cy; the distortion stays on the line.  The per-block aspect gate reads the pooled
# values (the observation-weighted mean of the block's pools' dfy - dfx); the per-bin lines stay in the report
# beside them.  The pool scales are in units of 1e-3 (0.1 %) in the problem, ppm in the results
DIAG_POOL = "sequence"
POOL_SCALE_UNIT = 1e-3
POOL_SEQUENCES_REPORTED = 10            # the pools with the largest |dfy - dfx| in the report
# 0.96: a block with two visits (Chal Rocks large: sols 72-100 and 360-379) gets one aspect row per visit as well
# (report only, not gated); visits are the pools' sols split at gaps over this many sols
VISIT_GAP_SOLS = 100
# 0.96 (item C of 5 Nov; Z63 ZCAM08038, sol 78: ZL's bins sat -3,450 / -3,760 ppm in fx where its low state's step is
# -4,170 ppm, ZR's within +900 of 0 - ZL in the low state, labelled high; the mini-block of 5 Nov evening confirmed
# it): per sequence and eye, from the diagnostic pass, "possible low state" when one eye's dfx is within this
# fraction of its low-state step and the other eye's within the same fraction of 0, fx and fy moving together.  A
# warning in the report and check.json, not a failure
OPPOSITE_STATE_TOL = 0.35
# 0.96 (item B of 5 Nov): the direction of the residuals per sequence from the final joint pass - rms x and y apart,
# the scale left in fx and fy (the slope of the residual against the distance from the principal point, ppm), and
# the same per row (elevation tier) for pans; the sequences with the largest rms, and every sequence flagged by the
# pools or the states, in the report.  Rows are elevation clusters split at gaps over this fraction of the vertical
# field of view
RESIDUAL_DIR_SEQUENCES = 10
ROW_GAP_FRACTION = 0.5
# 0.96 (item F of 5 Nov; Christian, 9 Nov, the Z110 slope: fy slopes 0.11-0.13 px/count against fx 0.31 in the bins,
# the shared slope pulled 5-9 % shallow; ZCAM08038, a 2-row strip 14 deg tall, fy/fx -0.5 % with no change in the
# residuals).  A sequence is weak in the vertical when its frames' boresight elevations cover (span + one field of
# view) less than WEAK_SEQUENCE_FOV vertical fields of view: single rows and 2-row strips.  In the joint such a
# sequence's fy gets an offset of its own (px, drawn to 0 with sd WEAK_FY_SD x f), so its fy does not pull the
# slope or the aspect; its pools stay out of the per-block aspect gate's mean (reported, not judged).  0 turns it off
WEAK_SEQUENCE_FOV = 2.0
WEAK_FY_SD = 0.01
# 0.96 (Christian, 9 Nov: "remove focus breathing as a fit parameter"): the focus-approach offset (0.94.4, one offset
# on f for the bins whose focus came into place from below) is no longer fitted - held at 0 - unless asked
# (``fit_zjoint(focus_below=True)``, ``calib zcam --zcam-focus-below``).  The focus slope itself stays, of course
FOCUS_BELOW_FIT = False
NAV_POINTS_PER_BLOCK = 8000
Z_POINTS_PER_BLOCK = 8000
_FX, _FY, _CX, _CY, _K1, _K2, _P1, _P2 = range(8)                     # FULL_OPENCV parameter indices


def zoom_groups(zoom: int = ZOOM) -> Tuple[str, str]:
    return f"ZL{zoom:03d}", f"ZR{zoom:03d}"


def _state(spec: Dict[str, Any], key: str) -> str:
    """The state whose terms a bin is tied to: ``regular`` or ``backlash`` - a single-state (110 mm) bin is on the
    zoom's one line, kept in the backlash terms (0.93.5: :func:`mppp.calib.forms.zcam_state`)."""
    from mppp.calib.forms import zcam_state
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
          scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None, found: Any = "auto",
          early_sol: Optional[int] = ZCAM_EARLY_SOL):
    """The merged reconstruction of the blocks: Navcam on the promoted law (held), Mastcam-Z bins of ``zoom`` as
    cameras of their own.  Returns (rec, proj, idx, st, units, zunits); ``zunits`` {camera_id: {"key", "scape",
    "group", "eye", "state", "focus", "dF", "c_cy"}}.  ``scapes``: loaded blocks (tests), else read from ``blocks``;
    ``st``: the Navcam law (:func:`mppp_studies.calibration.tied.start_values` of ``navcam_dir``, default the models in use)."""
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
    from mppp.sfm.backlash import default_found_path, is_low_backlash, load_low_backlash
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
    mark_early_bins(proj, zunits, sg, early_sol)                                # 0.95.1
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
    sg["weak_sequences"] = weak_sequences(rec, zunits)                          # 0.96 (item F)
    return rec, proj, idx, st, units, zunits, sg


def weak_sequences(rec, zunits: Dict[int, Dict[str, Any]], fov_factor: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """0.96 (item F): {"eye|sequence key": {"eye", "sequence", "images", "elevation_span_deg", "vertical_fov_deg",
    "covered_fov", "rows", "weak"}} for every sequence with frames in the problem - the frames' boresight
    elevations (:func:`_elevation_deg`, the block alignment's poses), the covered extent (span + one field of view)
    in vertical fields of view, the number of rows (elevation tiers, :data:`ROW_GAP_FRACTION`), and ``weak`` when
    the covered extent is under ``fov_factor`` (:data:`WEAK_SEQUENCE_FOV`; 0: nothing is weak) fields of view."""
    ff = WEAK_SEQUENCE_FOV if fov_factor is None else float(fov_factor)
    per: Dict[Tuple[str, str], List[float]] = {}
    cams: Dict[Tuple[str, str], int] = {}
    for iid in rec.reg_image_ids():
        cid = int(rec.images[iid].camera_id)
        u = zunits.get(cid)
        if u is None or not u.get("seq"):
            continue
        k = (u["group"], u["seq"])
        per.setdefault(k, []).append(_elevation_deg(rec, int(iid)))
        cams.setdefault(k, cid)
    out: Dict[str, Dict[str, Any]] = {}
    for (eye, seq), els in per.items():
        cam = rec.cameras[cams[(eye, seq)]]
        fy = float(np.asarray(cam.params, float)[_FY])
        vfov = float(np.degrees(2.0 * np.arctan(float(cam.height) / (2.0 * fy)))) if fy > 0 else 0.0
        span = float(max(els) - min(els))
        srt = sorted(els)
        rows = 1 + sum(1 for a, b in zip(srt, srt[1:]) if b - a > ROW_GAP_FRACTION * vfov)
        covered = (span + vfov) / vfov if vfov > 0 else 0.0
        out[f"{eye}|{seq}"] = {"eye": eye, "sequence": seq, "images": len(els), "elevation_span_deg": span,
                               "vertical_fov_deg": vfov, "covered_fov": covered, "rows": int(rows),
                               "weak": bool(ff > 0 and covered < ff)}
    return out


def mark_early_bins(proj, zunits: Dict[int, Dict[str, Any]], sg: Dict[str, Any], early_sol: Optional[int]) -> Dict[str, Any]:
    """0.95.0: the ZR bins before ``early_sol`` (their sol, the median of their images) carry the early-mission
    principal-point offset (:func:`make_zties`); their cx is freed for it (backlash bins hold cx at the pin
    otherwise).  ``sg["early"]`` {"before_sol", "eyes": {eye: {"bins", "images"}}}; no early_sol: none."""
    out: Dict[str, Any] = {"before_sol": int(early_sol) if early_sol else None, "eyes": {}}
    for cid, u in zunits.items():
        u["early"] = bool(early_sol) and u["group"].startswith("ZR") and u.get("sol") is not None \
            and int(u["sol"]) < int(early_sol)
        if not u["early"]:
            continue
        e = out["eyes"].setdefault(u["group"], {"bins": 0, "images": 0, "blocks": set()})
        e["bins"] += 1
        e["images"] += int(u.get("images") or 0)
        e["blocks"].add(u["scape"])
        spec = proj.cameras.get(u["key"])
        if spec is not None and u["state"] != "regular":
            spec["fixed_params"] = [x for x in spec.get("fixed_params") or [] if x != "cx"]
            spec["free_params"] = sorted(set(spec.get("free_params") or []) | {"cx"})
    for e in out["eyes"].values():
        e["blocks"] = sorted(e["blocks"])
    sg["early"] = out
    return out


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
    fit_fb = bool(sg.get("focus_below_fit", FOCUS_BELOW_FIT))
    dfb = {eye: np.array([float((sg.get("focus_below_px") or {}).get(eye, 0.0))])
           for eye in fb_eyes if eye in fb_high} if (tie and fit_fb) else {}
    fb_held = ({eye: "held at 0: every focus-from-below bin of this eye is in the low state (the offset is one with "
                     "the high / low ratio)" for eye in fb_eyes if eye not in fb_high} if fit_fb else
               # 0.96 (Christian, 9 Nov): not a fit parameter any more
               {eye: "held at 0: the focus-approach offset is not fitted (0.96; --zcam-focus-below fits it)" for eye in fb_eyes}) if tie else {}
    # 0.95.0 (Christian, 29 Oct): ZR's early-mission principal-point offset (cx, cy), shared by its bins before
    # sg["early"]["before_sol"]; from ZCAM_EARLY_MIN_BINS early bins on the lines on
    early_n: Dict[str, int] = {}
    for u in zunits.values():
        if u.get("early") and not u.get("thin"):
            early_n[u["group"]] = early_n.get(u["group"], 0) + 1
    dearly = {eye: np.array(list((sg.get("early_offsets_px") or {}).get(eye, [0.0, 0.0])), float)
              for eye, n in early_n.items() if n >= ZCAM_EARLY_MIN_BINS} if tie else {}
    # 0.95.0: the sequences' offsets (SEQ_OFFSET_SD, MIN_SEQ_IMAGES, MIN_SEQ_OBS)
    sfree, sinfo = sequence_groups(zunits) if tie else ({}, {})
    dseq = {k: np.array([float(((sg.get("sequence_offsets_px") or {}).get(f"{k[0]}|{k[1]}", 0.0)))]) for k in sfree}
    unit_seq = {cid: k for k, cids in sfree.items() for cid in cids}
    # 0.96 (item F): the weak-geometry sequences' fy offsets (px, drawn to 0 with WEAK_FY_SD x f)
    weak = sg.get("weak_sequences") or {}
    dweak = {k: np.array([float(((sg.get("weak_fy_px") or {}).get(f"{k[0]}|{k[1]}", 0.0)))])
             for k in sfree if (weak.get(f"{k[0]}|{k[1]}") or {}).get("weak")} if tie else {}

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
                nw = 1 if (sk is not None and sk in dweak) else 0     # 0.96: the weak sequence's fy offset
                asp = float(G0[1] / G0[0])
                nextra = nx + nf + ns + nw
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
                    if nw and pi == _FY:                     # fy = the line + this sequence's own fy offset
                        a[12 + NG + nx + nf + ns] = -1.0
                    rows.append(a); sig.append(TIE_SIGMA_Z["f"])
                for pi, gi, s_ in ((_K1, 5, "k"), (_K2, 6, "k"), (_P1, 7, "p"), (_P2, 8, "p")):
                    a = np.zeros(12 + NG + nextra); a[pi] = 1.0; a[12 + gi] = -1.0
                    rows.append(a); sig.append(TIE_SIGMA_Z[s_])
                add_tie(np.array(rows), np.array(sig), [12, NG] + [1] * nextra,
                        [blk, g[eye]] + ([dreg[rk]] if nx else []) + ([dfb[eye]] if nf else []) + ([dseq[sk]] if ns else [])
                        + ([dweak[sk]] if nw else []))
                ne = 2 if (u.get("early") and eye in dearly) else 0     # 0.95.0: + the early-mission offset
                if reg:                                      # 0.93.2: cx = pin + d_cx, cy = pin + b_cy dF + d_cy
                    c0 = np.array([u["c_cx"], u["c_cy"]], float)
                    keep.append(c0)
                    a = np.zeros((2, 12 + NG + 2 + ne))
                    a[0, _CX] = 1.0; a[0, 12 + 10] = -1.0; a[0, 12 + NG] = -1.0
                    a[1, _CY] = 1.0; a[1, 12 + 9] = -u["dF"]; a[1, 12 + 11] = -1.0; a[1, 12 + NG + 1] = -1.0
                    if ne:
                        a[0, 12 + NG + 2] = -1.0; a[1, 12 + NG + 3] = -1.0
                    add_tie(a, np.array([TIE_SIGMA_Z["cy"]] * 2), [12, NG, 2] + ([2] if ne else []),
                            [blk, g[eye], c0] + ([dearly[eye]] if ne else []))
                    prob.set_parameter_block_constant(c0)
                elif eye.startswith("ZR"):                   # cy = its pin + b_cy (focus - ref)
                    c0 = np.array([u["c_cy"]], float)
                    keep.append(c0)
                    a = np.zeros((1, 12 + NG + 1 + ne)); a[0, _CY] = 1.0; a[0, 12 + 9] = -u["dF"]; a[0, 12 + NG] = -1.0
                    if ne:
                        a[0, 12 + NG + 2] = -1.0
                    add_tie(a, np.array([TIE_SIGMA_Z["cy"]]), [12, NG, 1] + ([2] if ne else []),
                            [blk, g[eye], c0] + ([dearly[eye]] if ne else []))
                    prob.set_parameter_block_constant(c0)
                    if ne:                                   # cx = its pin + the early offset (cx freed for it)
                        cx0 = np.array([u["c_cx"]], float)
                        keep.append(cx0)
                        a = np.zeros((1, 12 + 1 + 2)); a[0, _CX] = 1.0; a[0, 12] = -1.0; a[0, 13] = -1.0
                        add_tie(a, np.array([TIE_SIGMA_Z["cy"]]), [12, 1, 2], [blk, cx0, dearly[eye]])
                        prob.set_parameter_block_constant(cx0)
                n += 1
        for (eye, gk), d in dreg.items():                # the groups' offsets drawn to 0 (the eye's ratio their mean)
            if prob.has_parameter_block(d):
                sd_px = REGULAR_RATIO_GROUP_SD * float(sg["eyes"][eye]["G"][2])
                prob.add_residual_block(LinearTie(np.ones((1, 1)), np.array([sd_px]), [1]), None, [d])
        for (eye, sk), d in dseq.items():                # 0.95.0: the sequences' offsets drawn to 0 (the line their mean)
            if prob.has_parameter_block(d):
                sd_px = SEQ_OFFSET_SD * float(sg["eyes"][eye]["G"][0])
                prob.add_residual_block(LinearTie(np.ones((1, 1)), np.array([sd_px]), [1]), None, [d])
        for (eye, sk), d in dweak.items():               # 0.96: the weak sequences' fy offsets drawn to 0
            if prob.has_parameter_block(d):
                sd_px = WEAK_FY_SD * float(sg["eyes"][eye]["G"][1])
                prob.add_residual_block(LinearTie(np.ones((1, 1)), np.array([sd_px]), [1]), None, [d])
        if tie and sg.get("tie_eye_slopes") and len(g) == 2:
            # 0.95.0 (Christian, 29 Oct: "the efl vs focus slopes of both should be about equal"): one focus slope
            # for both eyes, b_ZR = b_ZL exactly (the eye-slope test of :func:`eye_slope_tests`)
            zl, zr = sorted(g)
            if prob.has_parameter_block(g[zl]) and prob.has_parameter_block(g[zr]):
                a = np.zeros((1, 2 * NG)); a[0, 4] = 1.0; a[0, NG + 4] = -1.0
                add_tie(a, np.array([EYE_SLOPE_TIE_SD]), [NG, NG], [g[zl], g[zr]])
        if tie and sg.get("aspect_hold") is not None:    # 6 Nov 2026: fy = aspect x fx on the backlash line
            for eye, arr in g.items():
                if prob.has_parameter_block(arr):
                    a = np.zeros((1, NG)); a[0, 1] = 1.0; a[0, 0] = -float(sg["aspect_hold"])
                    add_tie(a, np.array([TIE_SIGMA_Z["f"]]), [NG], [arr])
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
        cb.update({f"dweak|{eye}|{sk}": d for (eye, sk), d in dweak.items() if prob.has_parameter_block(d)})
        cb.update({f"dearly|{eye}": d for eye, d in dearly.items() if prob.has_parameter_block(d)})
        return {"cov_blocks": cb, "ties": {"zcam_cameras": n}, "held": held,
                "tie_ids": tie_ids if T.EXACT_TIE_COVARIANCE else [], "tie_specs": specs,
                "exact_solve": bool(EXACT_TIE_SOLVE)}

    hook.ratio_offsets = dreg
    hook.ratio_by = by
    hook.focus_below = dfb
    hook.focus_below_held = fb_held
    hook.sequence_offsets = dseq
    hook.sequence_info = sinfo
    hook.weak_fy = dweak
    hook.early_offsets = dearly
    hook.early_bins = early_n
    return hook, g


def start_model_from_blocks(scapes: Sequence[Any], zoom: int) -> Dict[str, Any]:
    """A start focus model (zcam-v2 layout) for a zoom without one, from the blocks' refined focus-bin cameras
    (:func:`mppp_studies.calibration.joint.load_scape` scapes): per eye the median fx, fy at the median focus with one slope fitted
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
            "source": f"start model of {zoom} mm from {len(scapes)} blocks' refined focus bins (mppp_studies.calibration.ztied)"}


def fixed_globals(sg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Set the zoom's fixed distortion terms (:data:`mppp.calib.forms.ZCAM_FIXED_TERMS`) in the start globals ``sg``
    (changed in place); returns {eye group: {term: value}} of the terms to hold."""
    from mppp.calib.forms import zcam_fixed_terms
    out: Dict[str, Dict[str, float]] = {}
    for eye, e in sg["eyes"].items():
        fx = {k: float(v) for k, v in zcam_fixed_terms(eye).items() if k in GLOBAL_TERMS}
        for k, v in fx.items():
            e["G"][GLOBAL_TERMS.index(k)] = v
        if fx:
            out[eye] = fx
    return out


def adjust_z(rec, proj, idx, st, units, zunits, sg, iterations: int = 100, covariance: bool = True, passes: int = 2,
             tie: bool = True, refine_zcam_rig: Optional[str] = None, verbose: bool = True,
             hold: Union[Sequence[str], Dict[str, Sequence[str]]] = ()):
    """The Mastcam-Z joint's adjustment: Navcam cameras and rigs held (the promoted law), frame poses free, the
    Mastcam-Z bins tied to the global terms (:func:`make_zties`), the Mastcam-Z rig refined - ``refine_zcam_rig``
    "full" (rotation and translation, 1 cm prior on the right centre) or "rotation"; default ``sg["rig_fit"]``, else
    "rotation".  0.95.1: the thin bins (:func:`mark_thin_bins`) are held at their start (the Z79 joint of 28 Oct: a
    free 389-observation bin went to fx 10247, fy 6701 px and the rms to 34.9 px)."""
    from mppp.sfm.reconstruction import bundle_adjust
    refine_zcam_rig = refine_zcam_rig or sg.get("rig_fit") or "rotation"
    hook, g = make_zties(sg, zunits, tie=tie, hold=hold)
    navkeys = [k for k in proj.settings["database"]["cameras"] if str(k).startswith("N")]
    navkeys += sorted({u["key"] for u in zunits.values() if u.get("thin")})
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], _navcam_globals(st))
    ba = None
    for p_ in range(max(1, int(passes))):
        ba = bundle_adjust(rec, proj, zcam_aspect="free", sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=True,
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
    ba["weak_fy_px"] = {f"{k[0]}|{k[1]}": float(v[0]) for k, v in hook.weak_fy.items()}                 # 0.96
    sg["weak_fy_px"] = dict(ba["weak_fy_px"])
    ba["early_offsets_px"] = {k: [float(v[0]), float(v[1])] for k, v in hook.early_offsets.items()}    # 0.95.0
    ba["early_bins"] = dict(hook.early_bins)
    sg["early_offsets_px"] = dict(ba["early_offsets_px"])
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
    out["weak_fy"] = weak_fy_rows(ba, sg, zunits)                                # 0.96 (item F)
    out["weak_sequences"] = {k: v for k, v in (sg.get("weak_sequences") or {}).items() if v.get("weak")}
    out["sequence_offsets"] = sequence_offset_rows(out, ba, zunits)             # 0.95.0
    for eye, e in out["eyes"].items():
        rows_ = [r for r in out["sequence_offsets"] if r["eye"] == eye]
        fr = [r for r in rows_ if not r.get("held")]
        e["sequences"] = {"free": len(fr), "held": len(rows_) - len(fr),
                          "with_focus_span": sum(1 for r in fr if r["focus_span"] >= 300.0),
                          "sd_ppm": float(np.std([r["ppm"] for r in fr], ddof=1)) if len(fr) > 1 else None,
                          "max_abs_ppm": max((abs(r["ppm"]) for r in fr), default=None),
                          "prior_sd_pct": 100.0 * SEQ_OFFSET_SD}
        e["f0_sd"] = f0_uncertainty(e, fr)                                    # 0.96 (item D)
    return out


def f0_uncertainty(e: Dict[str, Any], free_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """0.96 (item D of 5 Nov): the sd of the eye's f0 (a_fx_backlash at the reference focus).  The sequence offsets
    are random effects with the prior sd :data:`SEQ_OFFSET_SD` x f, so with few sequences the covariance's sd of f0
    is about that prior / sqrt(sequences) - Z48 of 4 Nov: +-13.39 px, the same as each offset's sd - and reflects
    the assumed scatter, not the observed one.  Here the scatter is estimated from the fitted offsets (their sample
    sd; shrunk towards 0 by the prior, so a lower bound where the prior binds), f0's sd from it (scatter /
    sqrt(sequences)), and the larger of the two adopted.  The prior stays as it is in the fit.  Returns
    {"reported_px", "free_sequences", "prior_sd_px", "sequence_scatter_px", "sequence_scatter_ppm", "from_scatter_px",
    "adopted_px", "adopted_ppm", "adopted_source", "prior_bound" (the fitted offsets' sd within 10 % of the prior:
    the prior, not the data, set the offsets), "note"}."""
    tm, sd = e.get("terms") or {}, e.get("sd") or {}
    f0 = float(tm.get("a_fx_backlash") or 0.0)
    rep = sd.get("a_fx_backlash")
    rep = float(rep) if rep is not None and np.isfinite(rep) else None
    n = len(free_rows)
    out: Dict[str, Any] = {"reported_px": rep, "free_sequences": n, "prior_sd_px": float(SEQ_OFFSET_SD * f0 / np.sqrt(n)) if (n and f0) else None,
                           "sequence_scatter_px": None, "sequence_scatter_ppm": None, "from_scatter_px": None,
                           "adopted_px": rep, "adopted_ppm": (1e6 * rep / f0) if (rep is not None and f0) else None,
                           "adopted_source": "the covariance (no sequence scatter: fewer than 2 free sequences)", "prior_bound": None}
    if n >= 2:
        offs = np.array([float(r["offset_px"]) for r in free_rows])
        scatter = float(np.std(offs, ddof=1))
        from_sc = float(scatter / np.sqrt(n))
        osd = [float(r["sd_px"]) for r in free_rows if r.get("sd_px")]
        prior_px = SEQ_OFFSET_SD * f0
        out.update({"sequence_scatter_px": scatter, "sequence_scatter_ppm": (1e6 * scatter / f0) if f0 else None,
                    "from_scatter_px": from_sc,
                    "prior_bound": bool(osd and prior_px and abs(float(np.median(osd)) - prior_px) <= 0.1 * prior_px)})
        if rep is None or from_sc > rep:
            out.update({"adopted_px": from_sc, "adopted_source": f"the sequences' scatter / sqrt({n})"})
        else:
            out["adopted_source"] = "the covariance (larger than the sequences' scatter would give)"
        out["adopted_ppm"] = (1e6 * out["adopted_px"] / f0) if f0 else None
    out["note"] = ("f0's sd: the larger of the covariance's (which rests on the sequence-offset prior, %.1f %% of f) and "
                   "the fitted sequence offsets' sample sd / sqrt(sequences); the scatter is shrunk by the prior where it binds"
                   % (100 * SEQ_OFFSET_SD))
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


def rig_result(proj, sg, zoom: int = ZOOM, ba: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The joint's Mastcam-Z rig.  0.95.1 (rig translation refined): the right centre in the left camera's frame
    (x right, y down, z along the boresight) against the start rig's (the model in use / CAHV), with its sd and z per
    axis from the covariance of the rig block ("extra:zrig|Z0nn", scaled by the variance factor)."""
    fam = f"Z{zoom:03d}"
    v = (proj.settings.get("zcam_virtual_rig") or {}).get(fam) or {}
    R = np.asarray(v.get("R_refined") or v.get("R_sensor_from_ref"), float)
    t = np.asarray(v.get("t_refined") or v.get("t_sensor_from_ref"), float)
    mode = sg.get("rig_fit") or "rotation"
    out = {"R_sensor_from_ref": R.tolist(), "t_sensor_from_ref": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
           "change_mdeg": float(J.rotation_mdeg(sg["rig"]["R"], R)), "fit": mode,
           "translation_source": ("refined in the joint (1 cm prior per axis on the start rig's right centre)"
                                  if mode == "full" else "held (the rig in use)")}
    C0 = -np.asarray(sg["rig"]["R"], float).T @ np.asarray(sg["rig"]["t"], float)
    C = -R.T @ t
    out["right_centre_m"] = C.tolist()
    out["start_right_centre_m"] = C0.tolist()
    out["baseline_change_mm"] = float(1e3 * (np.linalg.norm(C) - np.linalg.norm(C0)))
    if mode != "full":
        return out
    out["translation_change_mm"] = (1e3 * (C - C0)).tolist()
    cov = (ba or {}).get("covariance") or {}
    c = (cov.get("blocks") or {}).get(f"extra:zrig|{fam}")
    vf = float(cov.get("variance_factor") or 1.0)
    if c is not None:
        c = np.asarray(c, float)
        it = slice(4, 7) if c.shape[0] == 7 else slice(3, 6)
        St = c[it, it] * vf
        SC = R.T @ St @ R                                    # C = -R^T t (the rotation's share is microns)
        sd = np.sqrt(np.clip(np.diag(SC), 0, None)) * 1e3
        if np.all(np.isfinite(sd)) and np.all(sd > 0):
            out["translation_sd_mm"] = sd.tolist()
            out["translation_z"] = (np.asarray(out["translation_change_mm"]) / sd).tolist()
            out["translation_significant"] = bool(np.max(np.abs(out["translation_z"])) >= EARLY_Z_MIN)
            u = C / np.linalg.norm(C)
            out["baseline_sd_mm"] = float(np.sqrt(u @ SC @ u) * 1e3)
    return out


def rig_lines(r: Dict[str, Any]) -> List[str]:
    """0.95.1: report lines of the rig translation (:func:`rig_result`)."""
    if not r or r.get("fit") != "full":
        return []
    ch = r.get("translation_change_mm") or [0, 0, 0]
    out = [f"rig translation refined: the right centre moved x {ch[0]:+.2f}, y {ch[1]:+.2f}, z (along the boresight) "
           f"{ch[2]:+.2f} mm from the start rig; baseline {1e3 * r['baseline_m']:.2f} mm ({r['baseline_change_mm']:+.2f})"]
    if r.get("translation_sd_mm"):
        sd, z = r["translation_sd_mm"], r["translation_z"]
        out.append("  sd " + ", ".join(f"{a} {s:.2f} mm (z {zz:+.1f})" for a, s, zz in zip("xyz", sd, z))
                   + ("; significant" if r.get("translation_significant") else "; within 3 sd of the start rig"))
    else:
        out.append("  (no covariance of the rig block)")
    return out


def early_result(ba: Dict[str, Any], sg: Dict[str, Any]) -> Dict[str, Any]:
    """0.95.1: ZR's early-mission principal-point offset (:func:`make_zties`) per eye with its sd and z; applied in
    the candidate model where either component's |z| >= :data:`EARLY_Z_MIN`."""
    info = sg.get("early") or {}
    out: Dict[str, Any] = {"before_sol": info.get("before_sol"), "eyes": {}}
    cov = (ba or {}).get("covariance") or {}
    vf = float(cov.get("variance_factor") or 1.0)
    for eye, e in sorted((info.get("eyes") or {}).items()):
        row: Dict[str, Any] = {"bins": int(e["bins"]), "images": int(e["images"]), "blocks": list(e["blocks"]),
                               "bins_on_lines": int((ba.get("early_bins") or {}).get(eye, 0))}
        d = (ba.get("early_offsets_px") or {}).get(eye)
        if d is None:
            row.update({"fitted": False, "applied": False,
                        "note": f"fewer than {ZCAM_EARLY_MIN_BINS} early bins on the lines: no offset"})
        else:
            row.update({"fitted": True, "offset_px": [float(d[0]), float(d[1])]})
            c = (cov.get("blocks") or {}).get(f"extra:dearly|{eye}")
            if c is not None:
                sd = np.sqrt(np.clip(np.diag(np.asarray(c, float)[:2, :2]) * vf, 0, None))
                if np.all(np.isfinite(sd)) and np.all(sd > 0):
                    row["sd_px"] = sd.tolist()
                    row["z"] = (np.asarray(row["offset_px"]) / sd).tolist()
            row["applied"] = bool(row.get("z") and max(abs(x) for x in row["z"]) >= EARLY_Z_MIN)
        out["eyes"][eye] = row
    return out


def early_lines(er: Optional[Dict[str, Any]]) -> List[str]:
    if not er or not er.get("eyes"):
        return []
    out = []
    for eye, r in er["eyes"].items():
        head = f"early-mission principal point of {eye} (bins before sol {er['before_sol']}: {r['bins']} bins, " \
               f"{r['images']} images, {', '.join(r['blocks'])})"
        if not r.get("fitted"):
            out.append(f"{head}: {r['note']}")
            continue
        o = r["offset_px"]
        zs = (f" +- {r['sd_px'][0]:.2f}, {r['sd_px'][1]:.2f} (z {r['z'][0]:+.1f}, {r['z'][1]:+.1f})"
              if r.get("z") else " (no sd)")
        out.append(f"{head}: dcx {o[0]:+.2f}, dcy {o[1]:+.2f} px{zs} - "
                   + ("significant: in the candidate model" if r["applied"] else "not significant: not applied"))
    return out


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


def _law_copy(rec, proj, zunits, sg, g, zoom: int = ZOOM):
    """0.96: a deep copy of the joint's reconstruction with every Mastcam-Z bin camera at the joint's law (its
    state's line at the bin's focus, plus its sequence, focus-from-below and early-mission offsets; the distortion
    and principal point of the law) and one Mastcam-Z rig per block (``Z0nn@block``, rotation free, at the joint's
    rig) - the start of the diagnostic passes.  Returns (rec, proj, law {bin camera id: params at the law}, R0, t0)."""
    rec = copy.deepcopy(rec)
    proj = copy.deepcopy(proj)
    fam0 = f"Z{zoom:03d}"
    R0 = np.asarray((proj.settings["zcam_virtual_rig"][fam0]).get("R_refined")
                    or proj.settings["zcam_virtual_rig"][fam0]["R_sensor_from_ref"], float)
    t0 = np.asarray((proj.settings["zcam_virtual_rig"][fam0]).get("t_refined")          # 0.95.0: as refined
                    or proj.settings["zcam_virtual_rig"][fam0]["t_sensor_from_ref"], float)
    law: Dict[int, np.ndarray] = {}
    for cid, u in zunits.items():                       # the bins at the joint's law
        p = np.asarray(rec.cameras[cid].params, float).copy()
        G = g[u["group"]]
        reg = u["state"] == "regular"
        ratio = slope_div(sg, u["group"], reg)
        dfb_ = float((sg.get("focus_below_px") or {}).get(u["group"], 0.0)) if u.get("fb") else 0.0
        # 0.95.0: plus the bin's sequence offset (else the per-block lines absorb the offsets as aspect / level)
        dsq_ = float((sg.get("sequence_offsets_px") or {}).get(f"{u['group']}|{u['seq']}", 0.0)) if u.get("seq") else 0.0
        dwk_ = float((sg.get("weak_fy_px") or {}).get(f"{u['group']}|{u['seq']}", 0.0)) if u.get("seq") else 0.0   # 0.96
        p[_FX] = G[2 if reg else 0] + G[4] * u["dF"] / ratio + dfb_ + dsq_
        p[_FY] = G[3 if reg else 1] + G[4] * u["dF"] / ratio + (dfb_ + dsq_) * float(G[1] / G[0]) + dwk_
        p[_K1], p[_K2], p[_P1], p[_P2] = G[5], G[6], G[7], G[8]
        if reg:                                         # 0.93.2: the regular state's principal point
            p[_CX] = u["c_cx"] + G[10]
            p[_CY] = u["c_cy"] + G[9] * u["dF"] + G[11]
        elif u["group"].startswith("ZR"):
            p[_CY] = u["c_cy"] + G[9] * u["dF"]
        de = (sg.get("early_offsets_px") or {}).get(u["group"]) if u.get("early") else None
        if de:                                          # 0.95.0: the early-mission offset (else the rig takes it)
            if not reg:
                p[_CX] = u["c_cx"]
            p[_CX] += float(de[0])
            p[_CY] += float(de[1])
        rec.cameras[cid].params = p
        law[int(cid)] = p.copy()
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
    return rec, proj, law, R0, t0


def _bin_observations(rec, zunits: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[int, int]:
    obs: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            c = int(rec.images[el.image_id].camera_id)
            if zunits is None or c in zunits:
                obs[c] = obs.get(c, 0) + 1
    return obs


def pool_key(u: Dict[str, Any], pool: str) -> str:
    """0.96: the pool a bin belongs to - ``"sequence"``: its sequence (block | sol | sequence; a bin without one, a
    focus bin of a project before 0.95.0, pools with its block and state), ``"block"``: its block and backlash
    state."""
    if pool == "sequence" and u.get("seq"):
        return str(u["seq"])
    return f"{u['scape']}|{u['state']}"


def pooled_pass(rec, proj, idx, st, units, zunits, sg, g, law: Dict[int, np.ndarray], pool: str = DIAG_POOL,
                zoom: int = ZOOM, iterations: int = 60, verbose: bool = True) -> Dict[str, Any]:
    """0.96 (item A of 5 Nov): the pooled diagnostic pass on a law copy (:func:`_law_copy`).  One pool per (eye,
    :func:`pool_key`); every bin camera of a pool is tied to it exactly - fx = law fx (1 + s_x), fy = law fy (1 +
    s_y), cx = law cx + d_cx, cy = law cy + d_cy - so the pool's four parameters (scales in :data:`POOL_SCALE_UNIT`,
    offsets in px) are all that moves; each frame's keypoints are scaled by the eye's line from the bin's focus to
    the frame's own, so the frames follow the line at their own focus count.  The distortion stays at the law, one
    rig per block (rotation free), the Navcam held.  Returns {"pool", "pools": [per pool the eye, key, block,
    sequence, sol, states, bins, observations, images, dfx_ppm, dfy_ppm, aspect_ppm (dfy - dfx) with their sd, dcx_px,
    dcy_px], "brief", "seconds"}."""
    from mppp.sfm.reconstruction import bundle_adjust
    import pyceres                                                       # noqa: F401 (pyceres present)
    LinearTie = T._linear_tie_class()
    obs = _bin_observations(rec, zunits)
    pools: Dict[Tuple[str, str], List[int]] = {}
    for cid, u in zunits.items():
        if obs.get(int(cid), 0) <= 0 or int(cid) not in law:
            continue
        pools.setdefault((u["group"], pool_key(u, pool)), []).append(int(cid))
    pvals = {k: np.zeros(4) for k in pools}
    for cids in pools.values():                                          # the bins' fx, fy, cx, cy move with the pool
        for cid in cids:
            spec = proj.cameras[zunits[cid]["key"]]
            spec["fixed_params"] = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
            spec["free_params"] = ["cx", "cy"]
    # the frames at their own focus: the keypoints scaled by the line from the bin's focus to the frame's
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], _navcam_globals(st))
    n_scaled = 0
    for r in proj.images:
        cid = int(r.get("camera_id", -1))
        u = zunits.get(cid)
        if u is None or cid not in law or u.get("focus") is None or r.get("focus_count") is None:
            continue
        G = g[u["group"]]
        b = float(G[4]) / slope_div(sg, u["group"], u["state"] == "regular")
        dF = float(r["focus_count"]) - float(u["focus"])
        if dF == 0.0:
            continue
        L = law[cid]
        ks[int(r["image_id"])] = (1.0 + b * dF / float(L[_FX]), 1.0 + b * dF / float(L[_FY]))
        n_scaled += 1
    const = np.ones(1)
    sig = np.array([TIE_SIGMA_Z["f"], TIE_SIGMA_Z["f"], TIE_SIGMA_Z["cy"], TIE_SIGMA_Z["cy"]])

    def hook(prob, cam_blocks=None, rig_blocks=None, key_of=None, work=None):
        n = 0
        for (eye, key), cids in pools.items():
            P = pvals[(eye, key)]
            for cid in cids:
                blk = cam_blocks.get(int(cid))
                if blk is None or not prob.has_parameter_block(blk) or prob.is_parameter_block_constant(blk):
                    continue
                L = law[cid]
                A = np.zeros((4, 12 + 4 + 1))
                A[0, _FX] = 1.0; A[0, 12] = -float(L[_FX]) * POOL_SCALE_UNIT; A[0, 16] = -float(L[_FX])
                A[1, _FY] = 1.0; A[1, 13] = -float(L[_FY]) * POOL_SCALE_UNIT; A[1, 16] = -float(L[_FY])
                A[2, _CX] = 1.0; A[2, 14] = -1.0; A[2, 16] = -float(L[_CX])
                A[3, _CY] = 1.0; A[3, 15] = -1.0; A[3, 16] = -float(L[_CY])
                prob.add_residual_block(LinearTie(A, sig, [12, 4, 1]), None, [blk, P, const])
                n += 1
        if prob.has_parameter_block(const):
            prob.set_parameter_block_constant(const)
        cb = {f"pool|{eye}|{key}": P for (eye, key), P in pvals.items() if prob.has_parameter_block(P)}
        return {"cov_blocks": cb, "ties": {"pooled_bins": n, "pools": len(cb)}, "tie_ids": [], "tie_specs": [],
                "exact_solve": False}

    navkeys = [k for k in proj.settings["database"]["cameras"] if str(k).startswith("N")]
    t = time.time()
    ba = bundle_adjust(rec, proj, zcam_aspect="free", sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=False,
                       refine_principal_point=False, refine_rig=False, max_iterations=iterations, attitude_prior_deg=5.0,
                       hold_cameras=navkeys, linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap,
                       refine_zcam_rig="rotation", covariance=True, extra_residuals=hook)
    if verbose:
        log.info(f"Mastcam-Z pooled diagnostic pass ({pool}, {len(pools)} pools, {n_scaled} frames at their own focus): "
                 f"{ba['brief'][:160]} ({time.time() - t:.0f} s)")
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor") or 1.0)
    rows: List[Dict[str, Any]] = []
    for (eye, key), cids in pools.items():
        P = pvals[(eye, key)]
        us = [zunits[c] for c in cids]
        sols = [u.get("sol") for u in us if u.get("sol") is not None]
        wk = (sg.get("weak_sequences") or {}).get(f"{eye}|{key}") or {}
        row: Dict[str, Any] = {"eye": eye, "key": key, "block": us[0]["scape"],
                               "sequence": us[0].get("sequence") if (pool == "sequence" and us[0].get("seq")) else None,
                               "weak": bool(wk.get("weak")), "covered_fov": wk.get("covered_fov"), "rows": wk.get("rows"),
                               "sol": int(round(float(np.median(sols)))) if sols else None,
                               "states": sorted({u["state"] for u in us}), "bins": len(cids),
                               "thin_bins": sum(1 for u in us if u.get("thin")),
                               "observations": int(sum(obs.get(c, 0) for c in cids)),
                               "images": int(sum(int(u.get("images") or 0) for u in us)),
                               "focus_span": float(np.ptp([u["dF"] for u in us])) if len(us) > 1 else 0.0,
                               "dfx_ppm": 1e6 * float(P[0]) * POOL_SCALE_UNIT, "dfy_ppm": 1e6 * float(P[1]) * POOL_SCALE_UNIT,
                               "dcx_px": float(P[2]), "dcy_px": float(P[3])}
        row["aspect_ppm"] = row["dfy_ppm"] - row["dfx_ppm"]
        c = (cov.get("blocks") or {}).get(f"extra:pool|{eye}|{key}")
        if c is not None and np.asarray(c).shape == (4, 4) and np.all(np.isfinite(c)):
            C = np.asarray(c, float) * vf
            sd = np.sqrt(np.clip(np.diag(C), 0, None))
            k = 1e6 * POOL_SCALE_UNIT
            row.update({"sd_dfx_ppm": k * float(sd[0]), "sd_dfy_ppm": k * float(sd[1]),
                        "sd_aspect_ppm": k * float(np.sqrt(max(0.0, C[0, 0] + C[1, 1] - 2 * C[0, 1]))),
                        "sd_dcx_px": float(sd[2]), "sd_dcy_px": float(sd[3])})
        rows.append(row)
    rows.sort(key=lambda r: (r["block"], r["eye"], r["key"]))
    return {"pool": pool, "pools": rows, "brief": ba["brief"], "seconds": time.time() - t, "frames_at_own_focus": n_scaled}


def diagnostic_pass(rec, proj, idx, st, units, zunits, sg, g, blocks: Optional[Dict[str, PathLike]] = None,
                    zoom: int = ZOOM, iterations: int = 60, verbose: bool = True,
                    pool: Optional[str] = DIAG_POOL) -> Dict[str, Any]:
    """The global law held: every Mastcam-Z bin at the law with fx, fy free (no ties), one Mastcam-Z rig per block
    (rotation free).  Returns {"blocks": per block the lines (a, b of fx, fy per eye and state, weighted by the
    bins' observations) and the rig rotation about the joint (yaw, pitch, roll mdeg), "fits": the rig's trends with
    sol and lens temperature, "bins": every bin's own fx, fy (and dfx_ppm, dfy_ppm about the law)}.  0.96: with
    ``pool`` ("sequence", the default, or "block") the pooled pass as well (:func:`pooled_pass`): ``"pool"`` and
    ``"pools"``, which the per-block aspect gate reads (:func:`block_aspect_rows`); None: the per-bin pass alone."""
    from mppp.sfm.reconstruction import bundle_adjust
    from .fit import ols
    if pool not in (None, "sequence", "block"):
        raise ValueError(f"pool {pool!r}: sequence, block or None")
    pooled = None
    pooled_error = None
    if pool:
        rec_p, proj_p, law, _, _ = _law_copy(rec, proj, zunits, sg, g, zoom)
        try:
            pooled = pooled_pass(rec_p, proj_p, idx, st, units, zunits, sg, g, law, pool=pool, zoom=zoom,
                                 iterations=iterations, verbose=verbose)
        except Exception as e:                        # noqa: BLE001 - in a chain the joint stands on the per-bin pass
            pooled_error = f"{type(e).__name__}: {e}"
            log.warning(f"Mastcam-Z pooled diagnostic pass failed ({pooled_error}); the aspect gate reads the per-bin lines")
        del rec_p, proj_p
    rec, proj, law, R0, t0 = _law_copy(rec, proj, zunits, sg, g, zoom)
    fam0 = f"Z{zoom:03d}"
    scapes = sorted({u["scape"] for u in zunits.values()})
    navkeys = [k for k in proj.settings["database"]["cameras"] if str(k).startswith("N")]
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], _navcam_globals(st))
    t = time.time()
    ba = bundle_adjust(rec, proj, zcam_aspect="free", sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=False,
                       refine_principal_point=False, refine_rig=False, max_iterations=iterations, attitude_prior_deg=5.0,
                       hold_cameras=navkeys, linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap,
                       refine_zcam_rig="rotation")
    if verbose:
        log.info(f"Mastcam-Z diagnostic pass: {ba['brief'][:160]} ({time.time() - t:.0f} s)")
    obs = _bin_observations(rec)
    meta = _block_meta(blocks or {}, proj)
    weak = sg.get("weak_sequences") or {}
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
                for pn, pi in (("fx", _FX), ("fy", _FY)):
                    us_ = us
                    if pn == "fy":                       # 0.96 (item F): the weak sequences' fy stays out of the line
                        strong = [(c, u) for c, u in us if not (weak.get(f"{u['group']}|{u.get('seq')}") or {}).get("weak")]
                        us_ = strong or us
                    F = np.array([u["dF"] for _, u in us_]); w = np.array([obs[c] for c, _ in us_], float)
                    y = np.array([float(rec.cameras[c].params[pi]) for c, _ in us_])
                    if len(us_) >= 3 and np.ptp(F) >= 300:
                        A = np.c_[np.ones_like(F), F]
                        sol = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
                        a, b = float(sol[0]), float(sol[1])
                    else:
                        bG = float(g[eye][4]) / slope_div(sg, eye, stt == "regular")
                        a, b = float(np.average(y - bG * F, weights=w)), bG
                    row["lines"].setdefault(eye, {}).setdefault(stt, {})[pn] = {"a": a, "b": b, "bins": len(us_),
                                                                                 "observations": int(w.sum()),
                                                                                 "weak_bins_left_out": len(us) - len(us_)}
        rows.append(row)
    fits = {}
    for ang in ("yaw", "pitch", "roll"):
        y = np.array([r[f"{ang}_mdeg"] for r in rows], float)
        S = np.array([np.nan if r["sol"] is None else r["sol"] for r in rows], float)
        Tz = np.array([np.nan if r["T_optics_degC"] is None else r["T_optics_degC"] for r in rows], float)
        fits[ang] = {"sd_mdeg": float(np.std(y, ddof=1)) if len(y) > 1 else None, "vs_sol_per_1000": ols(S / 1000.0, y),
                     "vs_T": ols(Tz, y)}
    # 0.95.1 (Christian, 29 Oct: the efl vs focus curves): every bin's own fx, fy in the diagnostic pass (the law held,
    # fx, fy free), for zcam_efl_vs_focus.png
    bins = []
    for cid, u in zunits.items():
        if obs.get(cid, 0) <= 0 or u.get("focus") is None:
            continue
        pr = np.asarray(rec.cameras[cid].params, float)
        L = law[int(cid)]
        bins.append({"block": u["scape"], "eye": u["group"], "state": u["state"], "focus": float(u["focus"]),
                     "fx": float(pr[_FX]), "fy": float(pr[_FY]), "observations": int(obs[cid]), "sol": u.get("sol"),
                     "sequence": u.get("sequence"), "seq": u.get("seq"), "thin": bool(u.get("thin")), "early": bool(u.get("early")),
                     # 0.96: about the law (the bin's line value), for the per-sequence means and the state flags
                     "dfx_ppm": 1e6 * float(pr[_FX] / L[_FX] - 1.0), "dfy_ppm": 1e6 * float(pr[_FY] / L[_FY] - 1.0),
                     "weak": bool((weak.get(f"{u['group']}|{u.get('seq')}") or {}).get("weak"))})
    out = {"blocks": rows, "fits": fits, "brief": ba["brief"], "bins": bins, "pool": pool,
           "weak_sequences": {k: v for k, v in weak.items() if v.get("weak")}}
    if pooled_error:
        out["pooled_error"] = pooled_error
    if pooled:
        out.update({"pools": pooled["pools"], "pooled_brief": pooled["brief"], "pooled_seconds": pooled["seconds"],
                    "frames_at_own_focus": pooled["frames_at_own_focus"]})
    return out


def set_ratio_holds(sg: Dict[str, Any], zoom: int, verbose: bool = True) -> Dict[str, float]:
    """0.94.5: the eyes with low-state bins but fewer than :data:`MIN_LOW_IMAGES` low-state images get
    ``sg["eyes"][eye]["ratio_hold"]`` - the 34 mm model's ratio of that side (``ZL034`` for ``ZL048``), else
    :data:`mppp_studies.calibration.fit.REGULAR_RATIO_DEFAULT`; the hook ties a_fx_regular = a_fx_backlash / that ratio."""
    from .fit import REGULAR_RATIO_DEFAULT
    out = {}
    if int(zoom) == RATIO_HOLD_ZOOM:
        return out
    ref = None
    try:
        from mppp.sfm.project import zcam_focus_model_path
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
    (:func:`mppp_studies.calibration.tied.diagnostic_pass`, one camera per block, eye and temperature bin, fx, fy, k1 free) over the
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


def visits_of(sols: Sequence[Optional[int]], gap: int = VISIT_GAP_SOLS) -> List[Tuple[int, int]]:
    """0.96: the visits among ``sols`` - runs of sols split where two consecutive ones are more than ``gap`` apart -
    as (first sol, last sol); [] with no sols."""
    ss = sorted({int(s) for s in sols if s is not None})
    if not ss:
        return []
    out, start = [], ss[0]
    for a, b in zip(ss, ss[1:]):
        if b - a > gap:
            out.append((start, a))
            start = b
    out.append((start, ss[-1]))
    return out


def sequence_scales(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """0.96: per (block, sequence, eye) the diagnostic pass's dfx, dfy about the law (ppm) - the pool's when the
    pass was pooled by sequence, else the observation-weighted mean of the sequence's bins; and the states, bins,
    observations.  Bins without a sequence (focus bins) are grouped by block and state under sequence None."""
    rows: List[Dict[str, Any]] = []
    pools = d.get("pools") if d.get("pool") == "sequence" else None
    if pools:
        for p in pools:
            rows.append({"block": p["block"], "sequence": p.get("sequence"), "sol": p.get("sol"), "eye": p["eye"],
                         "key": p["key"], "states": list(p.get("states") or []), "bins": int(p["bins"]),
                         "observations": int(p["observations"]), "dfx_ppm": float(p["dfx_ppm"]), "dfy_ppm": float(p["dfy_ppm"]),
                         "sd_dfx_ppm": p.get("sd_dfx_ppm"), "sd_dfy_ppm": p.get("sd_dfy_ppm"), "source": "pooled"})
        return rows
    acc: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for b in d.get("bins") or []:
        if b.get("dfx_ppm") is None or int(b.get("observations") or 0) <= 0:
            continue
        key = b.get("seq") or f"{b['block']}|{b['state']}"
        acc.setdefault((b["block"], key, b["eye"]), []).append(b)
    for (blk, key, eye), bs in sorted(acc.items()):
        w = np.array([float(b["observations"]) for b in bs])
        sols = [b.get("sol") for b in bs if b.get("sol") is not None]
        rows.append({"block": blk, "sequence": bs[0].get("sequence") if bs[0].get("seq") else None,
                     "sol": int(round(float(np.median(sols)))) if sols else None, "eye": eye, "key": key,
                     "states": sorted({b["state"] for b in bs}), "bins": len(bs), "observations": int(w.sum()),
                     "dfx_ppm": float(np.average([b["dfx_ppm"] for b in bs], weights=w)),
                     "dfy_ppm": float(np.average([b["dfy_ppm"] for b in bs], weights=w)),
                     "sd_dfx_ppm": None, "sd_dfy_ppm": None, "source": "per-bin mean"})
    return rows


def block_aspect_rows(res: Dict[str, Any], nav: Dict[str, Dict[str, Any]],
                      min_obs: int = BLOCK_ASPECT_MIN_OBS) -> Dict[str, Any]:
    """0.94.5 (Christian, 27 Oct): per block and eye, the Mastcam-Z fy - fx of the diagnostic pass (the state line
    with the most observations, at the reference focus) about the joint's (ppm of fx and px) against the Navcam's
    per-block dfy - dfx (``nav``, :func:`navcam_block_aspect`).  Rows with fewer than ``min_obs`` observations are
    listed but not judged.  0.96: with a pooled pass (``diagnostics["pools"]``) the judged value is the pooled one -
    the observation-weighted mean of the block's pools' dfy - dfx (``source`` "pooled") - with the per-bin line's
    beside it (``per_bin_ppm``, ``per_bin_diff_ppm``, ``per_bin_state``); a block with several visits
    (:func:`visits_of`) gets one row per visit as well (``visit``, report only: not judged); ``sequences`` lists the
    :data:`POOL_SEQUENCES_REPORTED` pools with the largest |dfy - dfx|.  Returns {"rows", "slope" (Mastcam-Z on
    Navcam ppm over the judged rows), "max_diff_ppm", "source", "sequences"}."""
    d = res.get("diagnostics") or {}
    per_bin: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for b in d.get("blocks") or []:
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
            per_bin[(b["block"], eye)] = {"state": stt, "ppm": float(1e6 * (fy / fx - asp)),
                                          "px": float((fy - fx) - fx * (asp - 1.0)), "observations": int(ln["fx"].get("observations", 0))}
    pools = d.get("pools") or []
    rows: List[Dict[str, Any]] = []

    def _row(block: str, eye: str, z_ppm: float, obs: int, px: float, **extra) -> Dict[str, Any]:
        n = nav.get(block) or {}
        nv = n.get("mean")
        r = {"block": block, "eye": eye, "observations": int(obs), "zcam_dfy_minus_dfx_px": float(px), "zcam_ppm": float(z_ppm),
             "navcam_ppm": None if nv is None else float(nv), "navcam_NL_ppm": n.get("NL"), "navcam_NR_ppm": n.get("NR"),
             "diff_ppm": None if nv is None else float(z_ppm - nv), "judged": bool(obs >= min_obs and nv is not None),
             "visit": None}
        r.update(extra)
        return r

    if pools:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for p in pools:
            groups.setdefault((p["block"], p["eye"]), []).append(p)
        for (block, eye), ps in sorted(groups.items()):
            tm = ((res.get("eyes") or {}).get(eye) or {}).get("terms") or {}
            f0 = float(tm.get("a_fx_backlash") or 0.0)
            pb = per_bin.get((block, eye)) or {}

            def pooled(ps_):
                ps_ = [p for p in ps_ if not p.get("weak")] or ps_     # 0.96 (item F): weak sequences out of the mean
                w = np.array([float(p["observations"]) for p in ps_])
                if w.sum() <= 0:
                    return None, 0
                return float(np.average([p["aspect_ppm"] for p in ps_], weights=w)), int(w.sum())

            z_ppm, obs = pooled(ps)
            if z_ppm is None:
                continue
            n_weak = sum(1 for p in ps if p.get("weak"))
            r = _row(block, eye, z_ppm, obs, f0 * z_ppm * 1e-6, source="pooled", pools=len(ps) - n_weak, weak_pools=n_weak,
                     state="+".join(sorted({s for p in ps for s in p.get("states") or []})),
                     per_bin_ppm=pb.get("ppm"), per_bin_state=pb.get("state"), per_bin_observations=pb.get("observations"))
            r["per_bin_diff_ppm"] = (None if (pb.get("ppm") is None or r["navcam_ppm"] is None) else float(pb["ppm"] - r["navcam_ppm"]))
            rows.append(r)
            vis = visits_of([p.get("sol") for p in ps])
            if len(vis) >= 2:                            # 0.96: one row per visit, report only
                for a, b_ in vis:
                    sub = [p for p in ps if p.get("sol") is not None and a <= int(p["sol"]) <= b_]
                    zv, ov = pooled(sub)
                    if zv is None:
                        continue
                    rv = _row(block, eye, zv, ov, f0 * zv * 1e-6, source="pooled", pools=len(sub), state=r["state"],
                              per_bin_ppm=None, per_bin_state=None, per_bin_observations=None, per_bin_diff_ppm=None,
                              visit=(f"sol {a}" if a == b_ else f"sols {a}-{b_}"))
                    rv["judged"] = False
                    rows.append(rv)
    else:
        for (block, eye), pb in sorted(per_bin.items()):
            rows.append(_row(block, eye, pb["ppm"], pb["observations"], pb["px"], source="per-bin", state=pb["state"]))
    J_ = [r for r in rows if r["judged"]]
    slope = None
    if len(J_) >= 3 and np.ptp([r["navcam_ppm"] for r in J_]) > 0:
        slope = float(np.polyfit([r["navcam_ppm"] for r in J_], [r["zcam_ppm"] for r in J_], 1)[0])
    seqs = sorted(pools, key=lambda p: -abs(float(p["aspect_ppm"])))[:POOL_SEQUENCES_REPORTED]
    return {"rows": rows, "slope": slope, "source": "pooled" if pools else "per-bin", "pool": d.get("pool"),
            "max_diff_ppm": max((abs(r["diff_ppm"]) for r in J_), default=None), "min_observations": int(min_obs),
            "weak_pools": sum(1 for p in pools if p.get("weak")),
            "sequences": [{k: p.get(k) for k in ("eye", "block", "sequence", "key", "sol", "states", "bins", "observations",
                                                 "dfx_ppm", "dfy_ppm", "aspect_ppm", "sd_aspect_ppm", "weak", "rows")} for p in seqs]}


# ====================================================================================== the run and the files
def fit_zjoint(blocks: Dict[str, PathLike], model: Optional[Dict[str, Any]] = None, navcam_dir: Optional[PathLike] = None,
               zoom: int = ZOOM, nav_points: int = NAV_POINTS_PER_BLOCK, z_points: int = Z_POINTS_PER_BLOCK,
               iterations: int = 100, passes: int = 2, diagnostics: bool = True, verbose: bool = True,
               scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None,
               k_tests: bool = True, ratio_by: Optional[str] = None, navcam_aspect: bool = True,
               slope_tests: bool = False, rig_fit: str = ZJOINT_RIG_FIT,
               early_sol: Optional[int] = ZCAM_EARLY_SOL, aspect_hold: Optional[float] = None,
               pool: Optional[str] = DIAG_POOL, focus_below: bool = FOCUS_BELOW_FIT) -> Dict[str, Any]:
    """The Mastcam-Z joint over ``blocks`` {name: WORK}: build, adjust, the distortion tests (:func:`distortion_tests`,
    ``k_tests``), the diagnostic pass.  ``model``: the focus model to start from (default: the one in use);
    ``ratio_by``: "global" | "block" | "sol", the low-state groups with their own high / low ratio (default
    :data:`REGULAR_RATIO_BY`).  ``slope_tests`` (0.95.0): the eye-slope tests (:func:`eye_slope_tests`, three more
    adjustments).  0.95.1: ``rig_fit`` "full" (the rig translation refined, :data:`ZJOINT_RIG_FIT`) or "rotation"
    (held at the start rig, the 0.94 joint); ``early_sol``: ZR's early-mission principal-point offset for its bins
    before this sol (:data:`ZCAM_EARLY_SOL`; None: none).  ``aspect_hold`` (6 Nov 2026, the Navcam-Mastcam-Z
    vertical-scale study): each eye's fy / fx held at this value (1.0: square pixels) by an exact tie of the
    backlash fy line to the fx line; the regular state follows as before.  Default None: the aspect is free.
    0.96: ``pool`` - the diagnostic pass's pooled model ("sequence", :data:`DIAG_POOL`; "block"; None: per bin
    only), which the per-block aspect gate reads; the residual directions per sequence (:func:`residual_directions`)
    and the opposite-state flags (:func:`opposite_state_flags`); ``focus_below``: fit the focus-approach offset
    (:data:`FOCUS_BELOW_FIT`, off since 9 Nov)."""
    from mppp.sfm.project import zcam_focus_model_path
    if rig_fit not in ("full", "rotation"):
        raise ValueError(f"rig_fit {rig_fit!r}: full or rotation")
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
    rec, proj, idx, st, units, zunits, sg = build(blocks, model, navcam_dir, zoom, nav_points, z_points, scapes=scapes, st=st,
                                                  early_sol=early_sol)
    sg["rig_fit"] = rig_fit
    if aspect_hold is not None:                        # 6 Nov 2026: start the fy lines at the held aspect
        sg["aspect_hold"] = float(aspect_hold)
        for e in sg["eyes"].values():
            G = e["G"]
            G[1] = float(aspect_hold) * G[0]
            G[3] = float(aspect_hold) * G[2]
        if verbose:
            log.info(f"Mastcam-Z joint: fy / fx held at {float(aspect_hold):.6f} in every eye")
    sg["focus_below_fit"] = bool(focus_below)                                   # 0.96
    if verbose and sg.get("weak_sequences"):
        nw = sum(1 for v in sg["weak_sequences"].values() if v.get("weak"))
        log.info(f"Mastcam-Z joint: {nw} of {len(sg['weak_sequences'])} sequences weak in the vertical "
                 f"(under {WEAK_SEQUENCE_FOV:g} fields of view): their fy gets an offset of its own")
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
                "zcam": _zstats(rec, zunits, proj.images), "rig": rig_result(proj, sg, zoom, ba), "final_cost": float(ba["final_cost"]),
                "iterations": ba["iterations"], "max_iterations": int(iterations), "passes": int(passes),
                "brief": ba["brief"], "seconds": time.time() - t,
                "navcam_models": str(st["folder"]), "points": {"navcam_per_block": nav_points, "zcam_per_block": z_points}})
    res["early_pp"] = early_result(ba, sg)                    # 0.95.1
    try:                                                      # 0.96 (item B): the residual directions per sequence
        res["residual_directions"] = residual_directions(rec, zunits, proj.images)
    except Exception as e:                                    # noqa: BLE001 - a diagnostic: the joint stands without it
        log.warning(f"Mastcam-Z joint: the residual directions did not run ({type(e).__name__}: {e})")
        res["residual_directions"] = {"error": f"{type(e).__name__}: {e}"}
    res["distortion"] = distortion_curves(res, ba, zunits=zunits)
    if k_tests:
        res["distortion"]["tests"] = distortion_tests(rec, proj, idx, st, units, zunits, sg, g, ba,
                                                      iterations=iterations, verbose=verbose, fixed=fixed)
    if slope_tests:
        res["eye_slope_tests"] = eye_slope_tests(rec, proj, idx, st, units, zunits, sg, g, ba, zoom=zoom,
                                                 iterations=iterations, verbose=verbose, fixed=fixed)
    if verbose:
        for line in describe(res):
            log.info(line)
    if diagnostics:
        res["diagnostics"] = diagnostic_pass(rec, proj, idx, st, units, zunits, sg, g, blocks=blocks, zoom=zoom,
                                             verbose=verbose, pool=pool)
        if navcam_aspect:                            # 0.94.5: the per-block aspect against the Navcam's
            try:
                nav = navcam_block_aspect(sg.get("_scapes") or [], st, nav_points, verbose=verbose)
                res["block_aspect"] = block_aspect_rows(res, nav)
            except Exception as e:                   # noqa: BLE001 - a diagnostic: the joint stands without it
                log.warning(f"Mastcam-Z joint: the per-block aspect against the Navcam did not run ({type(e).__name__}: {e})")
                res["block_aspect"] = {"error": f"{type(e).__name__}: {e}"}
        try:                                         # 0.96 (item C): possible opposite backlash states between the eyes
            res["opposite_states"] = opposite_state_flags(res)
            if verbose:
                for line in opposite_state_lines(res["opposite_states"]):
                    log.warning(line) if res["opposite_states"].get("flags") else log.info(line)
        except Exception as e:                       # noqa: BLE001
            log.warning(f"Mastcam-Z joint: the opposite-state check did not run ({type(e).__name__}: {e})")
            res["opposite_states"] = {"error": f"{type(e).__name__}: {e}"}
    return res


# ====================================================================================== residual directions, states
def _elevation_deg(rec, image_id: int) -> float:
    """The frame's boresight elevation (deg, negative looking down) from its pose in the reconstruction: the
    camera z axis in the world frame (z up: ENU / the site frame)."""
    R = np.asarray(rec.images[int(image_id)].cam_from_world().rotation.matrix(), float)
    return float(np.degrees(np.arcsin(np.clip(R[2, 2], -1.0, 1.0))))


def _direction_stats(dx: np.ndarray, dy: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> Dict[str, Any]:
    """rms of the x and y residuals, and the scale left in fx, fy: the slope of the residual against the distance
    from the principal point, as ppm the frames want the focal length changed by (positive: a larger f than the
    model's; the residual is projected - observed, so the data's scale is 1 - slope)."""
    out = {"observations": int(dx.size), "rms_px": float(np.sqrt(np.mean(dx ** 2 + dy ** 2))) if dx.size else None,
           "rms_x_px": float(np.sqrt(np.mean(dx ** 2))) if dx.size else None,
           "rms_y_px": float(np.sqrt(np.mean(dy ** 2))) if dy.size else None}
    for nm, d, v in (("fx", dx, vx), ("fy", dy, vy)):
        s2 = float(v @ v)
        if d.size >= 20 and s2 > 0:
            slope = float(d @ v) / s2
            resid = d - slope * v
            se = float(np.sqrt(max(0.0, resid @ resid / max(1, d.size - 1)) / s2))
            out[f"{nm}_scale_left_ppm"] = -1e6 * slope
            out[f"{nm}_scale_left_sd_ppm"] = 1e6 * se
        else:
            out[f"{nm}_scale_left_ppm"] = None
            out[f"{nm}_scale_left_sd_ppm"] = None
    return out


def residual_directions(rec, zunits: Dict[int, Dict[str, Any]], images: Sequence[Dict[str, Any]],
                        row_gap_fraction: float = ROW_GAP_FRACTION) -> Dict[str, Any]:
    """0.96 (item B of 5 Nov; ZCAM08401's frames at 1.9-2.5 px rms in the 4 Nov joint against 0.17-0.25 px alone):
    per sequence (block | sol | sequence; focus bins without one: block | state) and eye, from the final joint pass,
    the rms of the x and y residuals apart and the scale left in fx and fy (:func:`_direction_stats`); for pans the
    same per row (elevation tier: the frames' boresight elevations clustered at gaps over ``row_gap_fraction`` of the
    vertical field of view).  Mostly vertical residuals growing away from the centre mean a real pull (pose or
    structure) that pooling would hide; no direction means the bins were the problem.  Returns {"sequences": rows
    sorted by rms (block, sequence, sol, eye, images, the stats, "rows": [{tier, elevation_deg, images, stats}]),
    "row_gap_deg": {eye: deg}}."""
    zimg = {int(i) for i in rec.reg_image_ids() if int(rec.images[i].camera_id) in zunits}
    meta = {int(x.get("image_id", -1)): x for x in (images or [])}
    per: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            iid = int(el.image_id)
            if iid not in zimg:
                continue
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            uv = np.asarray(cam.img_from_cam(np.atleast_2d(im.cam_from_world() * pt.xyz)))[0]
            xy = np.asarray(im.points2D[el.point2D_idx].xy, float)
            if not np.all(np.isfinite(uv)):
                continue
            p = np.asarray(cam.params, float)
            per.setdefault(iid, []).append((float(uv[0] - xy[0]), float(uv[1] - xy[1]), float(xy[0] - p[_CX]), float(xy[1] - p[_CY])))
    seqs: Dict[Tuple[str, str], List[int]] = {}
    for iid in per:
        u = zunits[int(rec.images[iid].camera_id)]
        key = u.get("seq") or f"{u['scape']}|{u['state']}"
        seqs.setdefault((u["group"], key), []).append(iid)
    gap_deg: Dict[str, float] = {}
    rows_out: List[Dict[str, Any]] = []
    for (eye, key), iids in seqs.items():
        u0 = zunits[int(rec.images[iids[0]].camera_id)]
        fy = float(np.median([float(rec.cameras[rec.images[i].camera_id].params[_FY]) for i in iids]))
        H = float(rec.cameras[rec.images[iids[0]].camera_id].height)
        vfov = float(np.degrees(2.0 * np.arctan(H / (2.0 * fy))))
        gap_deg[eye] = row_gap_fraction * vfov
        arrs = [np.asarray(per[i], float) for i in iids]
        allr = np.vstack(arrs)
        els = {i: _elevation_deg(rec, i) for i in iids}
        m0 = meta.get(iids[0], {})
        row: Dict[str, Any] = {"eye": eye, "key": key, "block": u0["scape"], "sequence": u0.get("sequence") if u0.get("seq") else None,
                               "sol": u0.get("sequence_sol") if u0.get("seq") else (m0.get("sol")), "images": len(iids),
                               "elevation_span_deg": float(max(els.values()) - min(els.values())),
                               "vertical_fov_deg": vfov}
        row.update(_direction_stats(allr[:, 0], allr[:, 1], allr[:, 2], allr[:, 3]))
        # the rows (elevation tiers) of a pan
        order = sorted(iids, key=lambda i: els[i])
        tiers: List[List[int]] = [[order[0]]]
        for a, b in zip(order, order[1:]):
            if els[b] - els[a] > gap_deg[eye]:
                tiers.append([])
            tiers[-1].append(b)
        row["rows"] = []
        if len(tiers) >= 2:
            for k, tier in enumerate(tiers):
                tr = np.vstack([np.asarray(per[i], float) for i in tier])
                trow = {"tier": k, "elevation_deg": float(np.mean([els[i] for i in tier])), "images": len(tier)}
                trow.update(_direction_stats(tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]))
                row["rows"].append(trow)
        rows_out.append(row)
    rows_out.sort(key=lambda r: -(r["rms_px"] or 0.0))
    return {"sequences": rows_out, "row_gap_deg": gap_deg, "row_gap_fraction": float(row_gap_fraction)}


def residual_direction_lines(res: Dict[str, Any], n: int = RESIDUAL_DIR_SEQUENCES) -> List[str]:
    """0.96: report lines for the ``n`` sequences with the largest rms, plus every sequence flagged by the pools
    (:func:`block_aspect_rows` ``sequences``) or the states (:func:`opposite_state_flags`)."""
    rd = res.get("residual_directions") or {}
    if rd.get("error"):
        return [f"residual directions: not run ({rd['error']})"]
    seqs = rd.get("sequences") or []
    if not seqs:
        return []
    flagged = {(s["block"], s.get("sequence") or s.get("key")) for s in (res.get("block_aspect") or {}).get("sequences") or []}
    flagged |= {(f["block"], f.get("sequence") or f.get("key")) for f in (res.get("opposite_states") or {}).get("flags") or []}
    top = seqs[:n] + [s for s in seqs[n:] if (s["block"], s.get("sequence") or s["key"]) in flagged]
    out = [f"residual direction per sequence (final joint pass; the {n} largest rms" + (", and the flagged ones" if len(top) > n else "")
           + "; scale left: ppm the frames want f changed by, from the slope of the residual against the distance from "
             "the principal point; rows: elevation tiers of a pan)"]
    fmt = lambda v: "-" if v is None else f"{v:+.0f}"      # noqa: E731
    for s in top:
        head = f"  {s['block']} {s.get('sequence') or s['key']} {s['eye']} ({s['images']} frames): rms {s['rms_px']:.2f} px " \
               f"(x {s['rms_x_px']:.2f}, y {s['rms_y_px']:.2f}); scale left fx {fmt(s.get('fx_scale_left_ppm'))}, " \
               f"fy {fmt(s.get('fy_scale_left_ppm'))} ppm"
        if s.get("rows"):
            head += "; rows: " + ", ".join(f"el {r['elevation_deg']:+.0f} deg ({r['images']}) x {r['rms_x_px']:.2f} / y {r['rms_y_px']:.2f} px, "
                                           f"fy {fmt(r.get('fy_scale_left_ppm'))} ppm" for r in s["rows"])
        out.append(head)
    return out


def weak_fy_rows(ba: Dict[str, Any], sg: Dict[str, Any], zunits: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """0.96 (item F): per weak sequence with a free fy offset: the offset (px, ppm of fy, z against its prior) and
    the sequence's geometry (covered fields of view, rows, frames)."""
    cov = (ba.get("covariance") or {}).get("blocks") or {}
    vf = float((ba.get("covariance") or {}).get("variance_factor") or 1.0)
    weak = sg.get("weak_sequences") or {}
    rows: List[Dict[str, Any]] = []
    for k, px in sorted((ba.get("weak_fy_px") or {}).items()):
        eye, _, seq = k.partition("|")
        fy = float(sg["eyes"][eye]["G"][1])
        c = cov.get(f"extra:dweak|{eye}|{seq}")
        sd = float(np.sqrt(max(0.0, float(np.asarray(c).ravel()[0]) * vf))) if c is not None else None
        w = weak.get(k) or {}
        us = [u for u in zunits.values() if u["group"] == eye and u.get("seq") == seq]
        rows.append({"eye": eye, "key": seq, "sequence": us[0].get("sequence") if us else None, "block": us[0]["scape"] if us else None,
                     "sol": us[0].get("sequence_sol") if us else None, "offset_px": float(px), "ppm": 1e6 * float(px) / fy,
                     "sd_px": sd, "z": (float(px) / sd) if sd else None, "prior_sd_px": WEAK_FY_SD * fy,
                     "covered_fov": w.get("covered_fov"), "rows": w.get("rows"), "images": w.get("images"),
                     "elevation_span_deg": w.get("elevation_span_deg")})
    return rows


def weak_sequence_lines(res: Dict[str, Any]) -> List[str]:
    rows = res.get("weak_fy") or []
    d = res.get("diagnostics") or {}
    n_weak = len(d.get("weak_sequences") or {}) if d else None
    if not rows and not n_weak:
        return []
    head = (f"weak-geometry sequences (under {WEAK_SEQUENCE_FOV:g} vertical fields of view): "
            + (f"{n_weak} in the problem; " if n_weak is not None else "")
            + f"{len(rows)} with a free fy offset in the joint (prior sd {100 * WEAK_FY_SD:g} % of fy; their fy is out of the "
              f"slope and the aspect, their pools out of the aspect gate)")
    out = [head]
    for r in sorted(rows, key=lambda r: -abs(r["ppm"]))[:POOL_SEQUENCES_REPORTED]:
        out.append(f"  {r.get('block')} {r.get('sequence') or r['key']} {r['eye']}: fy {r['offset_px']:+.1f} px ({r['ppm']:+.0f} ppm"
                   + (f", z {r['z']:+.1f}" if r.get("z") is not None else "") + f"); {r.get('rows')} row(s), "
                   f"{r['covered_fov']:.1f} FOV covered, {r.get('images')} frames" if r.get("covered_fov") is not None else
                   f"  {r.get('block')} {r.get('sequence') or r['key']} {r['eye']}: fy {r['offset_px']:+.1f} px ({r['ppm']:+.0f} ppm)")
    return out


def low_state_step_ppm(res: Dict[str, Any], eye: str) -> Optional[float]:
    """0.96: the eye's high -> low backlash step in f (ppm, negative): 1e6 (1 / ratio - 1) with the joint's fitted
    high / low ratio (a_fx_backlash / a_fx_regular), else its held ratio, else the start model's; None for a
    single-state eye (110 mm)."""
    e = (res.get("eyes") or {}).get(eye)
    if not e or is_single_state(eye):
        return None
    tm = e.get("terms") or {}
    held = set(e.get("held") or [])
    if "a_fx_regular" not in held and tm.get("a_fx_regular"):
        ratio = float(tm["a_fx_backlash"]) / float(tm["a_fx_regular"])
    elif e.get("ratio_hold"):
        ratio = float(e["ratio_hold"])
    else:
        ratio = float(e.get("ratio_start") or 1.0)
    if abs(ratio - 1.0) < 1e-9:
        from .fit import REGULAR_RATIO_DEFAULT
        ratio = float(REGULAR_RATIO_DEFAULT)
    return 1e6 * (1.0 / ratio - 1.0)


def opposite_state_flags(res: Dict[str, Any], tol: float = OPPOSITE_STATE_TOL) -> Dict[str, Any]:
    """0.96 (item C of 5 Nov): per sequence, the two eyes' dfx from the diagnostic pass (:func:`sequence_scales`)
    against the eyes' low-state steps (:func:`low_state_step_ppm`).  An eye labelled high whose dfx is within ``tol``
    of its step while the other eye's is within ``tol`` of 0, fx and fy moving together (|dfy - dfx| within the same
    tolerance), is flagged "possible low state"; the mirror (an eye labelled low at +|step|, the other at 0) "possible
    high state".  A warning, not a failure.  Returns {"tolerance", "steps_ppm": {eye: step}, "sequences_checked",
    "flags": [{block, sequence, sol, eye, flag, dfx_ppm, dfy_ppm, step_ppm, other_eye, other_dfx_ppm, observations}]}."""
    d = res.get("diagnostics") or {}
    scales = sequence_scales(d)
    steps = {eye: low_state_step_ppm(res, eye) for eye in (res.get("eyes") or {})}
    by_seq: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for s in scales:
        by_seq.setdefault((s["block"], s["key"]), {})[s["eye"]] = s
    flags: List[Dict[str, Any]] = []
    checked = 0
    for (block, key), eyes in sorted(by_seq.items()):
        if len(eyes) != 2 or any(steps.get(e) is None for e in eyes):
            continue
        checked += 1
        for eye, s in eyes.items():
            other = next(o for o in eyes if o != eye)
            so = eyes[other]
            step, step_o = float(steps[eye]), float(steps[other])
            states = set(s.get("states") or [])
            together = abs(float(s["dfy_ppm"]) - float(s["dfx_ppm"])) <= tol * abs(step)
            other_zero = abs(float(so["dfx_ppm"])) <= tol * abs(step_o)
            if not (together and other_zero):
                continue
            flag = None
            if states == {"backlash"} and abs(float(s["dfx_ppm"]) - step) <= tol * abs(step):
                flag = f"possible low state, eye {eye}"
            elif states == {"regular"} and abs(float(s["dfx_ppm"]) + step) <= tol * abs(step):
                flag = f"possible high state, eye {eye}"
            if flag:
                flags.append({"block": block, "sequence": s.get("sequence"), "key": key, "sol": s.get("sol"), "eye": eye,
                              "flag": flag, "dfx_ppm": float(s["dfx_ppm"]), "dfy_ppm": float(s["dfy_ppm"]), "step_ppm": step,
                              "labelled": sorted(states), "other_eye": other, "other_dfx_ppm": float(so["dfx_ppm"]),
                              "observations": int(s["observations"]), "source": s.get("source")})
    return {"tolerance": float(tol), "steps_ppm": steps, "sequences_checked": checked, "flags": flags,
            "source": (scales[0]["source"] if scales else None)}


def opposite_state_lines(os_: Optional[Dict[str, Any]]) -> List[str]:
    if not os_:
        return []
    if os_.get("error"):
        return [f"opposite backlash states: not checked ({os_['error']})"]
    steps = ", ".join(f"{e} {s:+.0f}" for e, s in sorted((os_.get("steps_ppm") or {}).items()) if s is not None)
    head = (f"opposite backlash states between the eyes: {os_.get('sequences_checked', 0)} sequences checked against the "
            f"low-state steps ({steps} ppm; tolerance {100 * os_.get('tolerance', OPPOSITE_STATE_TOL):.0f} %)")
    if not os_.get("flags"):
        return [head + ": none flagged"]
    out = [head + f": {len(os_['flags'])} flagged (WARNING - check the low-state list)"]
    for f in os_["flags"]:
        out.append(f"  {f['block']} {f.get('sequence') or f['key']}" + (f" sol {f['sol']}" if f.get("sol") is not None else "")
                   + f": {f['flag']} (labelled {'/'.join(f['labelled'])}): dfx {f['dfx_ppm']:+.0f}, dfy {f['dfy_ppm']:+.0f} ppm "
                     f"against a step of {f['step_ppm']:+.0f}; {f['other_eye']} dfx {f['other_dfx_ppm']:+.0f} ppm; "
                     f"{f['observations']} obs")
    return out


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
        # 1 Nov (Christian, 110 mm k1 held at 0: "test if k1 is significant"; "Does z63 have a significant k2 or
        # tangential distortion parameters?"): each part freed alone too - k1, k2, the tangential p1 and p2 - so a
        # large joint delta chi2 is attributed
        if len(terms) > 1:
            for part in (["k1"], ["k2"], ["p1", "p2"]):
                fr = {e: [t for t in part if t in v] for e, v in fixed.items()}
                fr = {e: v for e, v in fr.items() if v}
                if fr and sorted({t for v in fr.values() for t in v}) != terms:
                    cases.append((f"{', '.join(part)} free (vs the fixed value{'s' if len(part) > 1 else ''})", {}, fr))
    if not any("k2" in fixed.get(e, {}) for e in eyes):
        cases.append(("k2 = 0 (both eyes)", {zl: {"k2": 0.0}, zr: {"k2": 0.0}}, {}))
    # 0.95.1 (Christian, 29 Oct: "ZR034 k1 should not be equal to ZL034 k1. remove that test in the joint"): no
    # test of the eyes' k1 (k1, k2) held equal - each eye has its own lens; the curves' difference and its z
    # (distortion_curves) are still reported
    out = []
    for name, fix, free in cases:
        t = time.time()
        sgv = copy.deepcopy(sg)
        for e in eyes:
            G = base[e].copy()
            for term, v in fix.get(e, {}).items():
                G[GLOBAL_TERMS.index(term)] = v
            for term in free.get(e, []):            # 0.95.1: a freed term starts off its fixed value - from the
                G[GLOBAL_TERMS.index(term)] += FREED_TERM_NUDGE.get(term, 0.0)   # tied solution it did not move
            sgv["eyes"][e]["G"] = G
        hold = {e: sorted((set(fixed.get(e, {})) | set(fix.get(e, {}))) - set(free.get(e, []))) for e in eyes}
        pv = copy.deepcopy(proj)
        for u in zunits.values():                   # 0.95.1: a freed term must be free in the bins too (their specs
            fr = set(free.get(u["group"], []))      # hold the fixed terms: the 28 Oct "k2 free" refits kept k2 at
            spec = pv.cameras.get(u["key"])         # 0.5000013 and the delta chi2 was convergence noise)
            if fr and spec is not None and not u.get("thin"):
                spec["fixed_params"] = [x for x in spec.get("fixed_params") or [] if x not in fr]
                spec["free_params"] = sorted(set(spec.get("free_params") or []) | fr)
                spec.pop("fixed_terms", None)
        try:
            # 0.95.1: the rig translation held at the joint's value (refined there): a variant with it free as well
            # wandered off on the synthetic blocks (delta chi2 4774 for the eyes' equal k1, k2, 0.3 held)
            bav, gv, _ = adjust_z(copy.deepcopy(rec), pv, idx, st, units, zunits, sgv,
                                  iterations=iterations, covariance=False, passes=1, verbose=False, hold=hold,
                                  refine_zcam_rig="rotation")
        except Exception as ex:                                       # noqa: BLE001
            out.append({"test": name, "error": f"{type(ex).__name__}: {ex}"})
            continue
        dof = sum(len(v) for v in fix.values()) + sum(len(v) for v in free.values())
        # 0.95.1: signed - freed terms lower the cost, held ones raise it; the other sign is a variant that has not
        # converged (was abs(): the 28 Oct "k2 free" delta chi2 of 810 was such a case)
        d = float(bav["final_cost"]) - cost0
        dchi2 = 2.0 * (-d if free else d) / vf
        resv = {"eyes": {e: {"terms": dict(zip(GLOBAL_TERMS, map(float, gv[e])))} for e in eyes}}
        resb = {"eyes": {e: {"terms": dict(zip(GLOBAL_TERMS, map(float, base[e])))} for e in eyes}}
        cv, cb = distortion_curves(resv), distortion_curves(resb)
        change = max(abs(a - b) for e in eyes for a, b in zip(cv["eyes"][e]["displacement_px"],
                                                              cb["eyes"][e]["displacement_px"]))
        p = float(chi2.sf(max(dchi2, 0.0), dof))
        rec_t = {"test": name, "dof": dof, "delta_chi2": dchi2, "p": p, "significant": bool(p < 0.001 and dchi2 > 0),
                 "converged": bool(dchi2 >= -1.0),
                 "z_equivalent": float(np.sqrt(max(dchi2, 0.0))) if dof == 1 else None,
                 "max_curve_change_px": float(change), "seconds": time.time() - t,
                 "refit_terms": {e: {k: float(gv[e][GLOBAL_TERMS.index(k)]) for k in ("k1", "k2", "p1", "p2")}
                                 for e in eyes},
                 "max_tangential_px": float(max(_tangential_corner_px(gv[e]) for e in eyes))}
        out.append(rec_t)
        if verbose:
            log.info(f"  distortion test {name}: delta chi2 {dchi2:.1f} ({dof} dof, p {p:.2g}), curve change "
                     f"<= {change:.2f} px, tangential <= {rec_t['max_tangential_px']:.2f} px at the corners; refit "
                     + "; ".join(f"{e} " + ", ".join(f"{k} {v:+.3g}" for k, v in rec_t["refit_terms"][e].items())
                                 for e in eyes) + f" ({time.time() - t:.0f} s)")
    return out


def _tangential_corner_px(G, frame_px: Tuple[int, int] = FRAME_PX) -> float:
    """The largest tangential (p1, p2) displacement in px at the four frame corners of global terms ``G`` (OpenCV:
    dx = 2 p1 x y + p2 (r^2 + 2 x^2), dy = p1 (r^2 + 2 y^2) + 2 p2 x y, x, y normalized)."""
    f = 0.5 * (G[GLOBAL_TERMS.index("a_fx_backlash")] + G[GLOBAL_TERMS.index("a_fy_backlash")])
    p1, p2 = float(G[GLOBAL_TERMS.index("p1")]), float(G[GLOBAL_TERMS.index("p2")])
    if not f:
        return 0.0
    W, H = frame_px
    out = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * W / 2.0 / f, sy * H / 2.0 / f
            r2 = x * x + y * y
            dx, dy = 2 * p1 * x * y + p2 * (r2 + 2 * x * x), p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            out = max(out, float(f * np.hypot(dx, dy)))
    return out


def _rig_centre(proj, zoom: int) -> Optional[np.ndarray]:
    """The right Mastcam-Z centre in the left camera's frame (m) of the zoom's rig as refined in ``proj``."""
    v = (proj.settings.get("zcam_virtual_rig") or {}).get(f"Z{zoom:03d}") or {}
    R, t = v.get("R_refined") or v.get("R_sensor_from_ref"), v.get("t_refined") or v.get("t_sensor_from_ref")
    if R is None or t is None:
        return None
    return -np.asarray(R, float).T @ np.asarray(t, float)


def eye_slope_tests(rec, proj, idx, st, units, zunits, sg, g, ba, zoom: int = ZOOM, iterations: int = 60,
                    verbose: bool = True, fixed: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """0.95.0 (Christian, 29 Oct: "the efl vs focus slopes of both should be about equal"; Z34 28 Oct: ZL 0.0245, ZR
    0.0351 px/count, the difference +0.009 to +0.012 in all ten blocks and positive at every zoom).  The hypothesis:
    the rig translation is held at the CAHV value, and an error dz along the boresight scales ZR's image against ZL's by
    dz / D - with D tied to the focus count, that lands in ZR's focus slope.  Three adjustments from the joint's
    solution, each against the full model as delta chi^2 = 2 (cost change) / variance factor:

    (1) ``rig translation free``: the Mastcam-Z rig's translation refined (1 cm prior per axis on the right centre,
    :data:`mppp.sfm.reconstruction.ZCAM_RIG_TRANSLATION_SIGMA_M`) - the slopes it gives and how far the right centre
    moves (mm, in the left camera's frame: z along the boresight);
    (2) ``one slope for both eyes``: b_ZR = b_ZL, the rig as in the joint - the cost of equal slopes;
    (3) ``one slope, rig translation free``: both - against (1), the cost of equal slopes once the rig may move.

    If the rig is the cause, (1) moves the centre a few mm (mostly z) and brings the slopes together, and (3) costs
    little against (1) while (2) costs a lot."""
    from scipy.stats import chi2
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor") or 0.0)
    cost0 = float(ba["final_cost"])
    eyes = sorted(g)
    if len(eyes) != 2 or not vf:
        return {"tests": [{"test": "skipped", "note": "two eyes and a variance factor (covariance) are needed"}]}
    zl, zr = eyes
    ib = GLOBAL_TERMS.index("b")
    fixed = fixed or {}
    hold = {e: sorted(fixed.get(e, {})) for e in eyes}
    C0 = _rig_centre(proj, zoom)
    out: Dict[str, Any] = {"base": {"b": {e: float(g[e][ib]) for e in eyes}, "cost": cost0, "variance_factor": vf,
                                    "rig_centre_m": None if C0 is None else C0.tolist()}, "tests": []}
    runs = {}
    base_rig = sg.get("rig_fit") or "rotation"
    out["base"]["rig"] = base_rig
    if base_rig == "full":          # 0.95.1: the joint refines the translation - test holding it, and one slope
        cases = (("rig translation held", "rotation", False), ("one slope for both eyes", "full", True))
        refs = {"rig translation held": ("the joint", cost0, 3), "one slope for both eyes": ("the joint", cost0, 1)}
    else:
        cases = (("rig translation free", "full", False), ("one slope for both eyes", "rotation", True),
                 ("one slope, rig translation free", "full", True))
        refs = None
    for name, rig_mode, tie_b in cases:
        t = time.time()
        sgv = copy.deepcopy(sg)
        for e in eyes:
            sgv["eyes"][e]["G"] = g[e].copy()
        if tie_b:                                       # start both at the observation-weighted mean
            bm = float(np.mean([g[e][ib] for e in eyes]))
            for e in eyes:
                sgv["eyes"][e]["G"][ib] = bm
        sgv["tie_eye_slopes"] = bool(tie_b)
        pv = copy.deepcopy(proj)
        if rig_mode == "rotation" and base_rig == "full":   # 0.95.1: held at the start rig, not at the joint's
            (pv.settings.get("zcam_virtual_rig") or {}).get(f"Z{zoom:03d}", {}).pop("t_refined", None)
        try:
            bav, gv, _ = adjust_z(copy.deepcopy(rec), pv, idx, st, units, zunits, sgv, iterations=iterations,
                                  covariance=False, passes=1, verbose=False, hold=hold, refine_zcam_rig=rig_mode)
        except Exception as ex:                                       # noqa: BLE001
            out["tests"].append({"test": name, "error": f"{type(ex).__name__}: {ex}"})
            continue
        C = _rig_centre(pv, zoom)
        runs[name] = float(bav["final_cost"])
        row = {"test": name, "rig": rig_mode, "one_slope": bool(tie_b), "cost": float(bav["final_cost"]),
               "b": {e: float(gv[e][ib]) for e in eyes}, "seconds": time.time() - t,
               "rig_centre_change_mm": None if (C is None or C0 is None) else (1e3 * (C - C0)).tolist()}
        out["tests"].append(row)
    # the delta chi^2 of each against its reference: (1) and (2) against the joint, (3) against (1)
    refs = refs or {"rig translation free": ("the joint", cost0, 3), "one slope for both eyes": ("the joint", cost0, 1),
                    "one slope, rig translation free": ("rig translation free", runs.get("rig translation free"), 1)}
    for row in out["tests"]:
        if "cost" not in row:
            continue
        ref_name, ref_cost, dof = refs[row["test"]]
        if ref_cost is None:
            continue
        d = 2.0 * (row["cost"] - ref_cost) / vf               # > 0: the variant fits worse
        p = float(chi2.sf(abs(d), dof))
        row.update({"against": ref_name, "dof": dof, "delta_chi2": float(d), "p": p, "significant": bool(p < 0.001)})
        if verbose:
            ch = row.get("rig_centre_change_mm")
            log.info(f"  eye-slope test {row['test']}: b {zl} {row['b'][zl]:.5f}, {zr} {row['b'][zr]:.5f} px/count; "
                     f"delta chi2 {d:+.1f} against {ref_name} ({dof} dof)"
                     + (f"; right centre moved ({ch[0]:+.2f}, {ch[1]:+.2f}, {ch[2]:+.2f}) mm" if ch else "")
                     + f" ({row['seconds']:.0f} s)")
    return out


def eye_slope_lines(est: Optional[Dict[str, Any]]) -> List[str]:
    """Report lines of :func:`eye_slope_tests`."""
    if not est:
        return []
    base = est.get("base") or {}
    out = []
    if base.get("b"):
        (zl, bl), (zr, br) = sorted(base["b"].items())
        out.append(f"eye slopes in the joint: {zl} {bl:.5f}, {zr} {br:.5f} px/count ({zr} - {zl} {br - bl:+.5f})")
    for t in est.get("tests") or []:
        if "delta_chi2" in t:
            b = t["b"]
            ch = t.get("rig_centre_change_mm")
            out.append(f"test {t['test']}: slopes " + ", ".join(f"{e} {v:.5f}" for e, v in sorted(b.items()))
                       + f" px/count; delta chi2 {t['delta_chi2']:+.1f} against {t['against']} ({t['dof']} dof, "
                         f"{'significant' if t['significant'] else 'not significant'} at 0.001)"
                       + (f"; the right centre moved x {ch[0]:+.2f}, y {ch[1]:+.2f}, z (along the boresight) "
                          f"{ch[2]:+.2f} mm" if ch else ""))
        else:
            out.append(f"test {t.get('test')}: {t.get('error') or t.get('note')}")
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
                       f"<= {t['max_curve_change_px']:.2f} px"
                       + (f", tangential <= {t['max_tangential_px']:.2f} px at the corners"
                          if t.get("max_tangential_px") is not None else ""))
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
    for eye, e in res["eyes"].items():               # 0.96 (item D): f0's sd from the sequence scatter
        fs = e.get("f0_sd") or {}
        if fs.get("sequence_scatter_px") is None:
            continue
        rp = "-" if fs.get("reported_px") is None else f"{fs['reported_px']:.2f}"
        out.append(f"{eye} f0 sd: covariance {rp} px (prior-based: {fs['prior_sd_px']:.2f} with {fs['free_sequences']} sequences"
                   + (", the offsets' sd at the prior" if fs.get("prior_bound") else "")
                   + f"); sequence scatter {fs['sequence_scatter_px']:.2f} px ({fs['sequence_scatter_ppm']:.0f} ppm) -> "
                   f"{fs['from_scatter_px']:.2f} px; adopted {fs['adopted_px']:.2f} px ({fs['adopted_ppm']:.0f} ppm, {fs['adopted_source']})")
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
    out += rig_lines(res.get("rig") or {})                         # 0.95.1
    out += early_lines(res.get("early_pp"))
    out += eye_slope_lines(res.get("eye_slope_tests"))             # 0.95.0
    d = res.get("diagnostics") or {}
    for ang, fz in (d.get("fits") or {}).items():
        a, b = fz.get("vs_sol_per_1000") or {}, fz.get("vs_T") or {}
        sl = lambda x, u: (f"{x['slope']:+.2f} +- {x['se']:.2f} {u}" if x.get("slope") is not None else "-")   # noqa: E731
        out.append(f"rig {ang} per block: sd {fz.get('sd_mdeg') or float('nan'):.1f} mdeg; vs sol {sl(a, 'mdeg/1000 sol')}, "
                   f"vs lens T {sl(b, 'mdeg/degC')}")
    ba_ = res.get("block_aspect") or {}
    if ba_.get("error"):
        out.append(f"per-block aspect vs the Navcam: not run ({ba_['error']})")
    if ba_.get("source") == "pooled":                # 0.96
        out.append(f"per-block aspect from the pooled diagnostic pass (one pool per eye and {ba_.get('pool')}: fx, fy "
                   f"scales and cx, cy free, the frames on the line at their own focus); the per-bin line beside it")
    for r_ in ba_.get("rows") or []:                 # 0.94.5 (Christian, 27 Oct)
        nv = "-" if r_.get("navcam_ppm") is None else f"{r_['navcam_ppm']:+.0f}"
        if r_.get("source") == "pooled":             # 0.96: the pooled value, the per-bin one beside it
            head = (f"{r_['block']} {r_['eye']}" + (f" {r_['visit']}" if r_.get("visit") else "")
                    + f" ({r_.get('pools', 0)} pools, {r_['observations']} obs)")
            pb = ("" if r_.get("per_bin_ppm") is None else
                  f"; per-bin line {r_['per_bin_ppm']:+.0f} ppm ({r_.get('per_bin_state')} state"
                  + (f", difference {r_['per_bin_diff_ppm']:+.0f}" if r_.get("per_bin_diff_ppm") is not None else "") + ")")
            out.append(f"{head}: Mastcam-Z pooled d(fy - fx) {r_['zcam_ppm']:+.0f} ppm, Navcam {nv} ppm"
                       + (f", difference {r_['diff_ppm']:+.0f} ppm" if r_.get("diff_ppm") is not None else "")
                       + pb + ("" if r_["judged"] else (" (visit: report only)" if r_.get("visit") else " (not judged)")))
            continue
        out.append(f"{r_['block']} {r_['eye']} ({r_['state']} state, {r_['observations']} obs): Mastcam-Z d(fy - fx) "
                   f"{r_['zcam_dfy_minus_dfx_px']:+.2f} px ({r_['zcam_ppm']:+.0f} ppm), Navcam {nv} ppm"
                   + (f", difference {r_['diff_ppm']:+.0f} ppm" if r_.get("diff_ppm") is not None else "")
                   + ("" if r_["judged"] else " (not judged)"))
    if ba_.get("slope") is not None:
        out.append(f"per-block aspect: Mastcam-Z on Navcam slope {ba_['slope']:+.2f} over the judged block lines")
    if ba_.get("sequences"):                         # 0.96: the pools with the largest |dfy - dfx|
        out.append(f"pooled sequences with the largest |dfy - dfx| ({len(ba_['sequences'])}): " + "; ".join(
            f"{s['block']} {s.get('sequence') or s['key']} {s['eye']} dfx {s['dfx_ppm']:+.0f} / dfy {s['dfy_ppm']:+.0f} ppm"
            + (f" (+- {s['sd_aspect_ppm']:.0f})" if s.get("sd_aspect_ppm") else "") + f", {s['observations']} obs"
            + (" [weak: out of the gate]" if s.get("weak") else "")
            for s in ba_["sequences"]))
    out += weak_sequence_lines(res)                                     # 0.96 (item F)
    out += opposite_state_lines(res.get("opposite_states"))             # 0.96 (item C)
    out += residual_direction_lines(res)                                # 0.96 (item B)
    return out


def candidate_model(res: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    """The focus model of the joint in the zcam-v2 form (the model in use with the joint's lines, distortion, ZR cy
    slope, regular ratio and rig)."""
    from mppp.calib import forms
    new = copy.deepcopy(model)
    stamp = datetime.date.today().isoformat()
    cal: Dict[str, Any] = {"method": "zcam tied joint (mppp_studies.calibration.ztied)", "when": stamp, "blocks": res["blocks"],
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
        gm.pop("f0_sd_px", None)
        gm.pop("f0_uncertainty", None)
        if e.get("f0_sd"):                            # 0.96 (item D): f0's sd with the sequence scatter folded in
            gm["f0_sd_px"] = e["f0_sd"].get("adopted_px")
            gm["f0_uncertainty"] = dict(e["f0_sd"])
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
            er = ((res.get("early_pp") or {}).get("eyes") or {}).get(eye)
            gm["pp"].pop("early", None)
            if er and er.get("fitted"):                  # 0.95.1: the early-mission offset (applied if significant)
                gm["pp"]["early"] = {"before_sol": res["early_pp"]["before_sol"], "dcx_px": er["offset_px"][0],
                                     "dcy_px": er["offset_px"][1], "sd_px": er.get("sd_px"), "z": er.get("z"),
                                     "applied": bool(er["applied"]), "blocks": er["blocks"],
                                     "note": "ZR's principal point before sol before_sol: + (dcx, dcy) when applied "
                                             "(the early-mission rig step of the Mastcam-Z joint)"}
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
                      translation_change_mm=r.get("translation_change_mm"), translation_sd_mm=r.get("translation_sd_mm"),
                      note="the Mastcam-Z joint: one rig for every block and focus (rotation refined, translation "
                           + ("refined, 1 cm prior on the start rig's right centre" if r.get("fit") == "full" else "held")
                           + "); per_block / residuals from the diagnostic pass (each block's own rig, the early-mission "
                             "principal-point offset applied)")
    new["rig"].pop("sd_baseline_mm", None)
    new["rig"].pop("baseline_vs_cahv", None)
    cal["diagnostics"] = {"blocks": rows, "fits": d.get("fits")}
    new["calibration"] = cal
    new["state"] = "calibrated"
    new["form"] = forms.ZCAM_FORM_ID
    new["source"] = (f"MPPP Mastcam-Z joint (mppp_studies.calibration.ztied, {stamp}): one adjustment over {len(res['blocks'])} "
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
                 "max_block_aspect_vs_navcam_ppm": 3000.0,
                 # 0.95.0 (Christian, 29 Oct): a block line is failed on its aspect only from this many observations
                 # on - Seitah Overlook (fixed infinity focus, far field) failed Z34 at 15 k and Z63 at 9 k observations
                 # with +3100 to +5400 ppm while every line from 25 k on stayed within +-1450 ppm; the lines below are
                 # listed in the note, not failed
                 "min_block_aspect_check_observations": 20000}             # 0.94.4: Z34 0.80, Z63 0.42; Z48 of 27 Oct 3.30, Z79 of 26 Oct 29.6


def check_label(c: Dict[str, Any]) -> str:
    """0.96: ``ok`` / ``WARN`` (a passing check with ``warning`` set) / ``FAIL``."""
    return "FAIL" if not c.get("ok") else ("WARN" if c.get("warning") else "ok")


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
        floor = int(lim.get("min_block_aspect_check_observations") or 0)
        judged = [r for r in ba_["rows"] if r["judged"] and r["diff_ppm"] is not None]
        checked = [r for r in judged if int(r["observations"]) >= floor]
        below = [r for r in judged if int(r["observations"]) < floor]
        bad = [r for r in checked if abs(r["diff_ppm"]) > lim["max_block_aspect_vs_navcam_ppm"]]
        worst = max((abs(r["diff_ppm"]) for r in checked), default=None)
        checks.append({"check": "per-block aspect vs the Navcam", "ok": not bad,
                       "note": (f"{len(checked)} block lines checked (>= {floor} observations), largest "
                                f"|Mastcam-Z - Navcam| d(fy - fx) {worst:.0f} ppm" if checked else "no line checked")
                               + f" (<= {lim['max_block_aspect_vs_navcam_ppm']:g})"
                               + (": " + ", ".join(f"{r['block']} {r['eye']} {r['diff_ppm']:+.0f}" for r in bad) if bad else "")
                               + (f"; below {floor} observations, listed only: " + ", ".join(
                                   f"{r['block']} {r['eye']} {r['diff_ppm']:+.0f} ppm ({r['observations'] // 1000} k)"
                                   for r in below) if below else "")})
    os_ = res.get("opposite_states")
    if os_ and not os_.get("error"):                    # 0.96 (item C): a warning, never a failure
        fl = os_.get("flags") or []
        checks.append({"check": "opposite backlash states between the eyes", "ok": True, "warning": bool(fl),
                       "note": (f"{os_.get('sequences_checked', 0)} sequences checked; "
                                + (("WARNING: " + ", ".join(f"{f['block']} {f.get('sequence') or f['key']} {f['flag']} "
                                                            f"(dfx {f['dfx_ppm']:+.0f} vs step {f['step_ppm']:+.0f} ppm)" for f in fl))
                                   if fl else "none flagged"))})
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
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.calib.constrain import plot_zcam_rig_trends
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
    try:                                                     # 0.95.1: the efl against focus, law and bins
        from mppp.calib.constrain import plot_zcam_efl_vs_focus
        if plot_zcam_efl_vs_focus(res, out / "zcam_efl_vs_focus.png"):
            figs.append("zcam_efl_vs_focus.png")
    except Exception as e:                                   # noqa: BLE001
        log.warning(f"efl figure: {e}")
    lines = [f"# Mastcam-Z {int(res['zoom'])} mm joint ({datetime.date.today().isoformat()})", "",
             f"Blocks: {', '.join(res['blocks'])}; the Navcam held at `{res.get('navcam_models')}`.", ""]
    lines += [f"- {x}" for x in describe(res)]
    lines += ["", "## Checks against the model in use", ""] + [f"- {check_label(c)} {c['check']}: {c['note']}"
                                                              for c in chk["checks"]]
    ba_ = res.get("block_aspect") or {}
    if ba_.get("sequences"):                                 # 0.96 (item A): the pools with the largest |dfy - dfx|
        lines += ["", f"## Pooled diagnostic pass: the {len(ba_['sequences'])} sequences with the largest |dfy - fx| "
                      f"(one pool per eye and {ba_.get('pool')}; ppm about the joint's line)", "",
                  "| eye | block | sequence | sol | dfx ppm | dfy ppm | dfy - dfx | sd | bins | obs | states |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in ba_["sequences"]:
            lines.append(f"| {s['eye']} | {s['block']} | {s.get('sequence') or s['key']} | {s['sol'] if s.get('sol') is not None else '-'} | "
                         f"{s['dfx_ppm']:+.0f} | {s['dfy_ppm']:+.0f} | {s['aspect_ppm']:+.0f} | "
                         + (f"{s['sd_aspect_ppm']:.0f}" if s.get("sd_aspect_ppm") else "-")
                         + f" | {s['bins']} | {s['observations']} | {', '.join(s.get('states') or [])} |")
    rd = res.get("residual_directions") or {}
    if rd.get("sequences"):                                  # 0.96 (item B)
        n = RESIDUAL_DIR_SEQUENCES
        flagged = {(s["block"], s.get("sequence") or s.get("key")) for s in ba_.get("sequences") or []}
        flagged |= {(f["block"], f.get("sequence") or f.get("key")) for f in (res.get("opposite_states") or {}).get("flags") or []}
        top = rd["sequences"][:n] + [s for s in rd["sequences"][n:] if (s["block"], s.get("sequence") or s["key"]) in flagged]
        fm = lambda v: "-" if v is None else f"{v:+.0f}"      # noqa: E731
        lines += ["", f"## Residual direction per sequence (final joint pass; the {n} largest rms and the flagged sequences)", "",
                  "Scale left: ppm the frames want f changed by (the slope of the residual against the distance from the "
                  "principal point; positive: a larger f than the model's). Rows: the elevation tiers of a pan.", "",
                  "| eye | block | sequence | sol | frames | rms px | rms x | rms y | fx left ppm | fy left ppm | el span deg | rows (el: x / y px, fy left) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in top:
            rws = "; ".join(f"{r['elevation_deg']:+.0f}: {r['rms_x_px']:.2f} / {r['rms_y_px']:.2f}, {fm(r.get('fy_scale_left_ppm'))}"
                            for r in s.get("rows") or []) or "-"
            lines.append(f"| {s['eye']} | {s['block']} | {s.get('sequence') or s['key']} | {s['sol'] if s.get('sol') is not None else '-'} | "
                         f"{s['images']} | {s['rms_px']:.2f} | {s['rms_x_px']:.2f} | {s['rms_y_px']:.2f} | "
                         f"{fm(s.get('fx_scale_left_ppm'))} | {fm(s.get('fy_scale_left_ppm'))} | {s['elevation_span_deg']:.0f} | {rws} |")
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
