"""
rpp_studies.calibration.deck_rig - the Navcam rig of a deck joint as a camera-model candidate (0.99.3).

Christian, 15 Nov 2026: "use the best deck joint parameters to initiate the new zcam runs"; 16 Nov: "Promote the latest
deck joint parameters".  The deck joint (``D:\\rover_frame_eod_early``, ``eod_pc.py``) ends with ``all_rig.json``: every
station's NR-in-NL rotation relative to the 3 Nov Navcam joint's rig (NAVJOINT) and the fitted laws against sol.  The
freeze-3 candidate ``M2020_N_rig.json`` is made from it as the 101-station one of 15 Nov was:

- rotation at sol 0 = Rot(a) . R(NAVJOINT), a = (pitch, yaw, roll) offsets in mdeg about the sensor's x, y, z: pitch and
  yaw from the knee-250 law, roll from the linear law (its knee is not significant);
- drift ``linear+hinge``, knot sol 250: pitch and yaw before / after the knee, roll linear;
- translation: the base rig's direction (freeze-2, the DJ32 deck joint) at the joint's mean baseline, one length (the
  joint's slope with sol is reported, not applied);
- the rest (sd, conventions) from the base rig; the replaced candidate under ``previous``.

``check`` compares the new law with the candidate it replaces over sols 0-2000 (the angles the blocks would get) and
the rates and baselines against their errors; ``--promote`` writes the candidate only when the check passes
(``--force``: anyway).  Nothing else is changed: freeze-3 promotes the candidate into its model folder at its start.

0.99.11 (Christian, 17 Nov 2026: "remove the pp temp dependence and instead fit the roll pitch and yaw change with sol
and temperature ... Will do the deck joint first and use its results for the freeze3"): from a deck joint in the rig_T
form (joint_ba --nav-pp-T off; all_rig.json ``form`` rig_T, every angle fitted against the Navcam temperature) the rig
file gets ``thermal`` - pitch, yaw, roll mdeg/degC about T0_degC (-20) - and, with ``--cameras-from`` (the folder the
station blocks' Navcam law came from, cmods_freeze2), the candidate folder gets the two Navcam camera files with their
principal-point slopes set to 0 (the old slopes kept under ``thermal.pp_T_slopes``): the rig_T model set.  The check
then also compares the whole stereo temperature change (the slopes' equivalent rotation plus the rig's rates) at T0 -/+
10 degC.

    python -m rpp_studies.calibration.deck_rig --all-rig D:\\rover_frame_eod_early\\all_rig.json
        --navjoint D:\\rover_blocks\\camera_analysis\\navcal_tied_20261103\\navcam_joint\\M2020_N_rig.json
        --base D:\\rover_blocks\\cmods_freeze2\\M2020_N_rig.json
        --current "D:\\code\\RPP\\Claude outputs\\...\\cmods_candidate_rig\\M2020_N_rig.json" --out <folder> [--promote]
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

KNEE_SOL = 250.0
SOLS = np.arange(0.0, 2001.0, 10.0)
# the check: the new law against the candidate it replaces
MAX_ANGLE_CHANGE_MDEG = {"pitch": 2.0, "yaw": 4.0, "roll": 6.0}   # about each angle's station scatter (sd 1.0 / 3.4 / 5.4)
MAX_RATE_Z = 3.0                    # a rate moved by more than this many combined standard errors
MAX_BASELINE_CHANGE_UM = 60.0       # about 3x the mean's error (13 um) plus the 101-station spread
# 0.99.11: the whole stereo temperature change (pp slopes' equivalent + the rig's rates) against the one it replaces, at
# T0 -/+ 10 degC: within the station scatter (pitch 1.0, yaw 3.4, roll 5.4 mdeg) - the forms differ by roll(T) and the
# deck joint's residual yaw(T) on top of the law (-0.1 to -0.2 mdeg/degC: 2 mdeg at 10 degC)
MAX_THERMAL_CHANGE_MDEG = {"pitch": 2.0, "yaw": 4.0, "roll": 6.0}
T0_DEGC = -20.0


def _rot(a_mdeg) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(1e-3 * np.asarray(a_mdeg, float))).as_matrix()


def _rotvec(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def law(all_rig: Dict[str, Any]) -> Dict[str, Any]:
    """The candidate law of an ``all_rig.json``: offsets a (mdeg), rates (mdeg / 1000 sol) with their errors, the
    baseline (mm) - pitch and yaw kneed at sol 250, roll linear."""
    ang = all_rig["angles"]
    out: Dict[str, Any] = {"stations": int(all_rig.get("stations") or 0), "sol_range": all_rig.get("sol_range"),
                           "source": all_rig.get("source")}
    for k in ("pitch", "yaw"):
        f = ang[k]["knee250"]
        if float(f.get("knee_sol", KNEE_SOL)) != KNEE_SOL:
            raise ValueError(f"{k}: the knee is at sol {f.get('knee_sol')}, not {KNEE_SOL}")
        out[k] = {"a_mdeg": float(f["a_mdeg"]), "rate_before": float(f["rate_before"]), "se_before": float(f["se_before"]),
                  "rate_after": float(f["rate_after"]), "se_after": float(f["se_after"]), "sd_mdeg": float(f["sd_mdeg"]),
                  "n": int(f.get("n") or ang[k].get("n") or 0)}
    f = ang["roll"]["linear"]
    out["roll"] = {"a_mdeg": float(f["a_mdeg"]), "rate": float(f["rate"]), "se": float(f["se"]), "sd_mdeg": float(f["sd_mdeg"]),
                   "n": int(ang["roll"].get("n") or 0)}
    # 0.99.11: the rig_T form - every angle against the Navcam temperature (pitch, yaw from their knee-250 fits, roll
    # from its linear fit, the fits the offsets and rates come from)
    out["form"] = str(all_rig.get("form") or "pp_T")
    out["T0_degC"] = float(all_rig.get("T0_degC", T0_DEGC))
    if out["form"] == "rig_T":
        for k, src in (("pitch", ang["pitch"]["knee250"]), ("yaw", ang["yaw"]["knee250"]), ("roll", ang["roll"]["linear"])):
            if src.get("per_degC") is None:
                raise ValueError(f"{k}: no temperature term in all_rig.json (a rig_T deck joint fits every angle against T)")
            out[k]["per_degC"] = float(src["per_degC"])
            out[k]["se_per_degC"] = float(src.get("se_per_degC") or float("nan"))
    elif (ang["yaw"].get("knee250") or {}).get("per_degC") is not None:
        out["yaw"]["per_degC_not_applied"] = float(ang["yaw"]["knee250"]["per_degC"])
    b = all_rig["baseline"]
    out["baseline"] = {"mean_mm": float(b["mean_mm"]), "se_mean_um": float(b["se_mean_um"]), "b0_mm": float(b.get("b0_mm", np.nan)),
                       "um_per_100sol": float(b.get("um_per_100sol", np.nan)), "se_um_per_100sol": float(b.get("se_um_per_100sol", np.nan)),
                       "n": int(b.get("n") or 0)}
    return out


def camera_rig_correction(station_cams: Optional[Dict[str, Dict[str, Any]]],
                          target_cams: Optional[Dict[str, Dict[str, Any]]]) -> np.ndarray:
    """0.99.11: the rig rotation change (pitch, yaw, roll mdeg) that carries a rig measured with the station blocks'
    Navcam cameras over to other cameras (the candidate's): their stereo principal points at T0 differ, and a principal
    point too far right by d is a camera turned by -d / fx - so rig_target = rig_station + (cxS - cxT)_R / fx_R -
    (cxS - cxT)_L / fx_L in yaw, and minus the same of cy in pitch.  The 16 Nov deck joint's stations were made with
    D:\\scapes\\cmods (the 27 Oct law), the freeze with cmods_freeze2 (3 Nov): NR - NL cx differs by -0.08 px, about
    -1.6 mdeg of yaw.  Zeros without both."""
    if not station_cams or not target_cams:
        return np.zeros(3)
    pS = {e: station_cams[e]["params"] for e in ("NL", "NR")}
    pT = {e: target_cams[e]["params"] for e in ("NL", "NR")}
    yaw = (float(pS["NR"][2]) - float(pT["NR"][2])) / float(pT["NR"][0]) - (float(pS["NL"][2]) - float(pT["NL"][2])) / float(pT["NL"][0])
    pitch = -((float(pS["NR"][3]) - float(pT["NR"][3])) / float(pT["NR"][1]) - (float(pS["NL"][3]) - float(pT["NL"][3])) / float(pT["NL"][1]))
    return np.degrees(np.array([pitch, yaw, 0.0])) * 1e3


def station_cameras_from_project(project_json: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    """0.99.11: the Navcam cameras a station block was made with, at T0: its project's start cameras NL, NR with the
    applied thermal terms (``settings.navcam_cameras.<eye>.thermal``) and early-mission offsets taken out - what the
    deck joint's station cameras are once ``--nav-pp-T off`` took their slopes out (the folder they came from may have
    changed since).  None when the project lacks them."""
    try:
        p = _read(project_json)
    except (OSError, ValueError):
        return None
    nc = (p.get("settings") or {}).get("navcam_cameras") or {}
    out = {}
    for e in ("NL", "NR"):
        cam, info = (p.get("cameras") or {}).get(e), nc.get(e) or {}
        th = info.get("thermal") or {}
        if not cam or th.get("T_median_degC") is None:
            return None
        q = [float(x) for x in cam["params"]]
        dT = float(th["T_median_degC"]) - float(th.get("T0_degC", T0_DEGC))
        em = (info.get("early_mission") or {}).get("applied_px") or {}
        q[0] /= float(th.get("sx", 1.0))
        q[1] /= float(th.get("sy", 1.0))
        q[2] -= float(th.get("cx_px_per_degC") or 0.0) * dT + float(em.get("cx") or 0.0)
        q[3] -= float(th.get("cy_px_per_degC") or 0.0) * dT + float(em.get("cy") or 0.0)
        out[e] = {"params": q, "source": f"{project_json} ({e} at T0)"}
    return out


def build(all_rig: Dict[str, Any], navjoint: Dict[str, Any], base: Dict[str, Any], current: Optional[Dict[str, Any]] = None,
          label: str = "", day: Optional[str] = None, camera_correction_mdeg: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """The candidate ``M2020_N_rig.json`` (see the module's text).  0.99.11: ``camera_correction_mdeg`` (pitch, yaw,
    roll; :func:`camera_rig_correction`) added to the offsets - the rig for the candidate's cameras."""
    L = law(all_rig)
    day = day or datetime.date.today().isoformat()
    corr = np.zeros(3) if camera_correction_mdeg is None else np.asarray(camera_correction_mdeg, float)
    a = [L["pitch"]["a_mdeg"] + corr[0], L["yaw"]["a_mdeg"] + corr[1], L["roll"]["a_mdeg"] + corr[2]]
    R = _rot(a) @ np.asarray(navjoint["R_sensor_from_ref"], float)
    t_dir = np.asarray(base["t_sensor_from_ref_m"], float)
    t_dir = t_dir / np.linalg.norm(t_dir)
    b_m = L["baseline"]["mean_mm"] / 1000.0
    t = t_dir * b_m
    n = L["stations"]
    p, y, r, bl = L["pitch"], L["yaw"], L["roll"], L["baseline"]
    what = (f"the {n}-station deck joint{(' ' + label) if label else ''} ({day}; law_candidate, knee {KNEE_SOL:g})")
    note = (f"x_NR = R x_NL + t. freeze-3 rig ({day}, Christian: 'Promote the latest deck joint parameters'): {what}. "
            f"Rotation at sol 0 = Rot(a) . R(3 Nov NAVJOINT) with a = pitch {a[0]:+.3f}, yaw {a[1]:+.2f}, roll {a[2]:+.3f} mdeg; "
            f"drift pitch {p['rate_before']:+.2f} +- {p['se_before']:.1f} before / {p['rate_after']:+.2f} +- {p['se_after']:.2f} "
            f"after sol {KNEE_SOL:g}, yaw {y['rate_before']:+.2f} +- {y['se_before']:.1f} / {y['rate_after']:+.2f} +- {y['se_after']:.2f}, "
            f"roll {r['rate']:+.2f} +- {r['se']:.2f} mdeg/1000 sol (linear); baseline {bl['mean_mm']:.3f} +- {bl['se_mean_um'] / 1000:.3f} mm "
            f"(fit {bl['b0_mm']:.3f} at sol 0, {bl['um_per_100sol']:+.1f} +- {bl['se_um_per_100sol']:.1f} um/100 sol - not applied: one "
            f"length) along the base rig's translation direction."
            + (f" Temperature (rig_T): pitch {p['per_degC']:+.3f} +- {p['se_per_degC']:.3f}, yaw {y['per_degC']:+.3f} +- "
               f"{y['se_per_degC']:.3f}, roll {r['per_degC']:+.3f} +- {r['se_per_degC']:.3f} mdeg/degC about {L['T0_degC']:g} degC "
               f"(the deck joint with the law's principal-point slopes out of the cameras)." if L["form"] == "rig_T" else "")
            + (f" Carried over to the candidate's cameras: pitch {corr[0]:+.3f}, yaw {corr[1]:+.3f} mdeg (their stereo principal "
               f"points at T0 against the station blocks')." if np.any(corr) else ""))
    out = copy.deepcopy(base)
    out.pop("previous", None)
    out.update({
        "ref": navjoint.get("ref", "NL"), "sensor": navjoint.get("sensor", "NR"),
        "R_sensor_from_ref": R.tolist(), "rotvec_rad": _rotvec(R).tolist(),
        "t_sensor_from_ref_m": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
        "sd_mdeg": navjoint.get("sd_mdeg", base.get("sd_mdeg")), "note": note,
        "drift": {"model": "linear+hinge", "sol0": 0.0, "knot_sol": KNEE_SOL, "knot_ref": 0.0,
                  "pitch_mdeg_per_sol": p["rate_after"] / 1000.0, "pitch_early_mdeg_per_sol": p["rate_before"] / 1000.0,
                  "yaw_mdeg_per_sol": y["rate_after"] / 1000.0, "yaw_early_mdeg_per_sol": y["rate_before"] / 1000.0,
                  "roll_mdeg_per_sol": r["rate"] / 1000.0, "hinge_angles": ["pitch", "yaw"], "stations": n,
                  "source": f"deck joint {label or 'ALL'} ({n} stations, {day}): law_candidate, knee {KNEE_SOL:g}; roll linear"},
        "yaw_model": ((f"rig_T: yaw linear in the Navcam temperature ({y['per_degC']:+.3f} mdeg/degC) and kneed at sol {KNEE_SOL:g}: "
                       if L["form"] == "rig_T" else "one yaw for every temperature; yaw kneed at sol {KNEE_SOL:g}: ".format(KNEE_SOL=KNEE_SOL))
                      + f"{y['rate_before']:+.2f} +- {y['se_before']:.1f} before, {y['rate_after']:+.2f} +- {y['se_after']:.2f} "
                      f"mdeg/1000 sol after (deck joint, {n} stations)"),
        "translation_source": (f"the {n}-station deck joint ({day}): {bl['mean_mm']:.3f} +- {bl['se_mean_um'] / 1000:.3f} mm, no "
                               "temperature term; the base rig's direction (DJ32)"),
        "translation": "deck", "source": note.split("x_NR = R x_NL + t. ", 1)[1],
        "deck_law": L,
        "camera_correction_mdeg": dict(zip(("pitch", "yaw", "roll"), map(float, corr))),
    })
    if L["form"] == "rig_T":                          # 0.99.11
        out["thermal"] = {"T0_degC": L["T0_degC"], "form": "rig_T",
                          **{f"{k}_mdeg_per_degC": L[k]["per_degC"] for k in ("pitch", "yaw", "roll")},
                          **{f"sd_{k}_mdeg_per_degC": L[k]["se_per_degC"] for k in ("pitch", "yaw", "roll")},
                          "source": f"deck joint {label or 'ALL'} ({n} stations, {day}) in the rig_T form: the station rigs "
                                    "against sol and the Navcam temperature (delete-one-station jackknife errors)"}
    else:
        out.pop("thermal", None)
    prev: Dict[str, Any] = {}
    if current is not None:
        prev["replaced_source"] = current.get("source")
        prev["replaced_drift"] = current.get("drift")
        prev["replaced_baseline_m"] = current.get("baseline_m")
        prev["replaced_R_sensor_from_ref"] = current.get("R_sensor_from_ref")
        prev["replaced_t_sensor_from_ref_m"] = current.get("t_sensor_from_ref_m")
    prev["base_source"] = base.get("source")
    prev["base_drift"] = base.get("drift")
    prev["base_baseline_m"] = base.get("baseline_m")
    if isinstance(base.get("previous"), dict):
        prev["older"] = base["previous"]
    out["previous"] = prev
    return out


def angles_vs_sol(rig: Dict[str, Any], sols=SOLS, T: Optional[float] = None) -> np.ndarray:
    """(n, 3) pitch, yaw, roll (mdeg, the rotation vector of R at each sol) of a rig file (0.99.11: at camera
    temperature ``T``; None: the file's T0)."""
    from rpp.calib.constrain import rig_at
    return np.array([np.degrees(_rotvec(rig_at(rig, float(s), T)[0])) * 1e3 for s in sols])


def pp_equivalent(cams: Optional[Dict[str, Dict[str, Any]]]) -> np.ndarray:
    """0.99.11: the stereo rotation rate (pitch, yaw, roll mdeg/degC) of a pair of Navcam camera files' principal-point
    slopes (yaw = d(cx)/fx, pitch = -d(cy)/fy, NR - NL); zeros without the files."""
    if not cams:
        return np.zeros(3)
    t = {e: (cams[e].get("thermal") or {}) for e in ("NL", "NR")}
    p = {e: cams[e]["params"] for e in ("NL", "NR")}
    yaw = float(t["NR"].get("cx_px_per_degC") or 0.0) / float(p["NR"][0]) - float(t["NL"].get("cx_px_per_degC") or 0.0) / float(p["NL"][0])
    pitch = -(float(t["NR"].get("cy_px_per_degC") or 0.0) / float(p["NR"][1]) - float(t["NL"].get("cy_px_per_degC") or 0.0) / float(p["NL"][1]))
    return np.degrees(np.array([pitch, yaw, 0.0])) * 1e3


def rig_T_cameras(base_dir: Path, day: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """0.99.11: the two Navcam camera files of ``base_dir`` in the rig_T form - their principal-point slopes set to 0,
    the old ones under ``thermal.pp_T_slopes`` - {"NL": ..., "NR": ...}."""
    day = day or datetime.date.today().isoformat()
    out = {}
    for e in ("NL", "NR"):
        c = _read(Path(base_dir) / f"M2020_{e}_fisheye_tangential.json")
        th = dict(c.get("thermal") or {})
        th["pp_T_slopes"] = {"cx_px_per_degC": th.get("cx_px_per_degC"), "cy_px_per_degC": th.get("cy_px_per_degC"),
                             "sd_cx_px_per_degC": th.get("sd_cx_px_per_degC"), "sd_cy_px_per_degC": th.get("sd_cy_px_per_degC")}
        th["cx_px_per_degC"] = th["cy_px_per_degC"] = 0.0
        th.pop("sd_cx_px_per_degC", None); th.pop("sd_cy_px_per_degC", None)
        th["form"] = "rig_T"
        th["note"] = (f"{th.get('note', '')}; rig_T ({day}): cx, cy constant - the stereo temperature change is the rig's "
                      f"thermal rates (M2020_N_rig.json)").lstrip("; ")
        c["thermal"] = th
        c["source"] = f"{c.get('source', '')} [rig_T form {day}: principal-point slopes set to 0]"
        out[e] = c
    return out


def check(new: Dict[str, Any], current: Dict[str, Any], cams_new: Optional[Dict[str, Dict[str, Any]]] = None,
          cams_current: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The new candidate against the one it replaces: the largest change of each angle over sols 0-2000, the rate
    changes in combined standard errors (when both carry their law), the baseline change.  0.99.11: with the camera
    files of each (``cams_new``, ``cams_current``), the whole stereo temperature change at T0 -/+ 10 degC too (the
    principal-point slopes' equivalent rotation plus the rig's rates).  {"ok", "lines", ...}."""
    d = angles_vs_sol(new) - angles_vs_sol(current)
    lines, ok = [], True
    out: Dict[str, Any] = {"max_change_mdeg": {}}
    if cams_new is not None or cams_current is not None or new.get("thermal") or current.get("thermal"):
        en, ec = pp_equivalent(cams_new), pp_equivalent(cams_current)
        out["thermal_change_mdeg"] = {}
        for dT in (-10.0, 10.0):
            T_ = T0_DEGC + dT
            dd = ((angles_vs_sol(new, T=T_) + en * dT) - (angles_vs_sol(current, T=T_) + ec * dT)) - d
            for i, k in enumerate(("pitch", "yaw", "roll")):
                m = float(np.max(np.abs(dd[:, i])))
                out["thermal_change_mdeg"][f"{k}@{T_:g}"] = m
                good = m <= MAX_THERMAL_CHANGE_MDEG[k]
                ok &= good
                lines.append(f"{k:5s} stereo temperature change at {T_:g} degC, against the replaced model set (beyond its "
                             f"change at T0): {m:.2f} mdeg (limit {MAX_THERMAL_CHANGE_MDEG[k]:g}) {'ok' if good else 'TOO LARGE'}")
        th = new.get("thermal") or {}
        lines.append("stereo temperature rates, new: " + ", ".join(
            f"{k} {float(th.get(f'{k}_mdeg_per_degC') or 0.0) + en[i]:+.3f}" for i, k in enumerate(("pitch", "yaw", "roll")))
            + " mdeg/degC; replaced: " + ", ".join(
            f"{k} {float((current.get('thermal') or {}).get(f'{k}_mdeg_per_degC') or 0.0) + ec[i]:+.3f}"
            for i, k in enumerate(("pitch", "yaw", "roll"))) + " (rig rates plus the camera slopes' equivalent)")
    for i, k in enumerate(("pitch", "yaw", "roll")):
        m = float(np.max(np.abs(d[:, i])))
        out["max_change_mdeg"][k] = m
        good = m <= MAX_ANGLE_CHANGE_MDEG[k]
        ok &= good
        lines.append(f"{k:5s} change over sols 0-2000: {d[:, i].min():+.2f} ... {d[:, i].max():+.2f} mdeg "
                     f"(limit {MAX_ANGLE_CHANGE_MDEG[k]:g}) {'ok' if good else 'TOO LARGE'}")
    Ln, Lc = new.get("deck_law"), current.get("deck_law") or _law_from_drift(current)
    if Ln and Lc:
        if not current.get("deck_law"):
            lines.append("rates: the replaced candidate carries no deck law - its drift rates, the new law's errors only")
        for k, parts in (("pitch", ("before", "after")), ("yaw", ("before", "after")), ("roll", ("",))):
            for part in parts:
                rk, sk = (f"rate_{part}", f"se_{part}") if part else ("rate", "se")
                z = (Ln[k][rk] - Lc[k][rk]) / max(np.hypot(Ln[k][sk], Lc[k][sk]), 1e-9)
                good = abs(z) <= MAX_RATE_Z
                ok &= good
                lines.append(f"{k:5s} rate {part or 'linear':6s} {Lc[k][rk]:+7.2f} -> {Ln[k][rk]:+7.2f} mdeg/1000 sol (z {z:+.1f}) "
                             f"{'ok' if good else 'MOVED'}")
    db = 1e6 * (float(new["baseline_m"]) - float(current["baseline_m"]))
    good = abs(db) <= MAX_BASELINE_CHANGE_UM
    ok &= good
    lines.append(f"baseline {1e3 * float(current['baseline_m']):.3f} -> {1e3 * float(new['baseline_m']):.3f} mm ({db:+.1f} um, limit "
                 f"{MAX_BASELINE_CHANGE_UM:g}) {'ok' if good else 'MOVED'}")
    out.update(ok=bool(ok), lines=lines, baseline_change_um=db)
    return out


def _law_from_drift(rig: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The rates of a rig made before ``deck_law`` was written (the 101-station candidate of 15 Nov), from its
    ``drift``, with zero errors."""
    dr = rig.get("drift") or {}
    if not all(f"{k}_mdeg_per_sol" in dr for k in ("pitch", "yaw", "roll")):
        return None
    out: Dict[str, Any] = {}
    for k in ("pitch", "yaw", "roll"):
        after = 1e3 * float(dr[f"{k}_mdeg_per_sol"])
        before = 1e3 * float(dr.get(f"{k}_early_mdeg_per_sol", dr[f"{k}_mdeg_per_sol"]))
        out[k] = {"rate": after, "se": 0.0, "rate_before": before, "se_before": 0.0, "rate_after": after, "se_after": 0.0}
    return out


def _read(p) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-rig", required=True, help="the deck joint's all_rig.json")
    ap.add_argument("--navjoint", required=True, help="the 3 Nov Navcam joint's M2020_N_rig.json (the joint's reference rotation)")
    ap.add_argument("--base", required=True, help="the rig the translation direction comes from (freeze-2: cmods_freeze2)")
    ap.add_argument("--current", default=None, help="the candidate this one replaces (compared with, kept under previous)")
    ap.add_argument("--label", default="", help="the joint's name in the notes (e.g. ALL, 16 Nov)")
    ap.add_argument("--cameras-from", default=None, metavar="DIR",
                    help="0.99.11: the folder of the Navcam law the deck joint's station blocks used (cmods_freeze2): with a "
                         "rig_T all_rig.json, its two Navcam camera files go into --out with their principal-point slopes "
                         "set to 0 (the rig_T model set)")
    ap.add_argument("--station-cameras", default=None, metavar="PATH",
                    help="0.99.11: the Navcam cameras the deck joint's station blocks were made with - a station block's "
                         "colmap\\project.json (its start cameras at T0: the applied thermal and early terms taken out) or a "
                         "folder of camera files: the rig is carried over to the --cameras-from cameras by the difference of "
                         "their stereo principal points at T0 (camera_rig_correction; 16 Nov stations against cmods_freeze2: "
                         "pitch +0.60, yaw -1.61 mdeg)")
    ap.add_argument("--current-cameras", default=None, metavar="DIR",
                    help="0.99.11: the folder of the Navcam camera files that go with --current (default: --cameras-from, "
                         "else --current's folder) - the check of the whole stereo temperature change")
    ap.add_argument("--out", required=True, help="the folder M2020_N_rig.json is written into (with check.json)")
    ap.add_argument("--promote", action="store_true", help="write M2020_N_rig.json only when the check passes "
                    "(the replaced file kept as M2020_N_rig_replaced_<time>.json); without it: written as M2020_N_rig_new.json")
    ap.add_argument("--force", action="store_true", help="with --promote: write it even when the check fails")
    a = ap.parse_args(argv)
    cur = _read(a.current) if a.current else None

    def _cams0(d):
        try:
            return {e: _read(Path(d) / f"M2020_{e}_fisheye_tangential.json") for e in ("NL", "NR")}
        except (OSError, ValueError):
            return None
    corr = None
    if a.station_cameras and a.cameras_from:          # 0.99.11
        sc_ = station_cameras_from_project(Path(a.station_cameras)) if Path(a.station_cameras).is_file() else _cams0(a.station_cameras)
        tc_ = _cams0(a.cameras_from)
        if sc_ is None or tc_ is None:
            print(f"ERROR: the Navcam camera files of --station-cameras {a.station_cameras} or --cameras-from {a.cameras_from} "
                  f"are not readable", file=sys.stderr)
            return 2
        corr = camera_rig_correction(sc_, tc_)
        print(f"the station blocks' cameras ({a.station_cameras}) against the candidate's ({a.cameras_from}): the rig turned by "
              f"pitch {corr[0]:+.3f}, yaw {corr[1]:+.3f} mdeg")
    new = build(_read(a.all_rig), _read(a.navjoint), _read(a.base), cur, label=a.label, camera_correction_mdeg=corr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams_new = None
    if new.get("thermal") and a.cameras_from:         # 0.99.11: the rig_T model set
        cams_new = rig_T_cameras(Path(a.cameras_from))
    elif new.get("thermal"):
        print("WARNING: a rig_T rig without --cameras-from: the candidate has no Navcam camera files - with the slopes of the "
              "models in use the stereo temperature change would be counted twice (calib.check_form refuses that folder)")
    def _cams(d):
        try:
            return {e: _read(Path(d) / f"M2020_{e}_fisheye_tangential.json") for e in ("NL", "NR")}
        except (OSError, ValueError):
            return None
    cams_cur = _cams(a.current_cameras or a.cameras_from or (Path(a.current).parent if a.current else "")) if cur else None
    chk = (check(new, cur, cams_new=cams_new if cams_new is not None else cams_cur, cams_current=cams_cur) if cur
           else {"ok": True, "lines": ["no current candidate: nothing to compare"]})
    L = new["deck_law"]
    print(f"deck joint {a.all_rig}: {L['stations']} stations, sols {L['sol_range']}")
    for line in chk["lines"]:
        print("  " + line)
    (out / "check.json").write_text(json.dumps({k: v for k, v in chk.items()}, indent=1, default=float), encoding="utf-8")
    if a.promote and (chk["ok"] or a.force):
        dst = out / "M2020_N_rig.json"
        if dst.is_file():
            keep = out / f"M2020_N_rig_replaced_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
            shutil.copy2(dst, keep)
            print(f"the candidate it replaces: {keep}")
        dst.write_text(json.dumps(new, indent=1), encoding="utf-8")
        print(f"promoted: {dst}" + ("" if chk["ok"] else " (--force: the check failed)"))
        for e, c in (cams_new or {}).items():           # 0.99.11: the rig_T camera files beside the rig
            cf = out / f"M2020_{e}_fisheye_tangential.json"
            if cf.is_file():
                shutil.copy2(cf, out / f"M2020_{e}_fisheye_tangential_replaced_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
            cf.write_text(json.dumps(c, indent=1), encoding="utf-8")
            print(f"promoted: {cf} (rig_T: principal-point slopes 0)")
        return 0
    dst = out / "M2020_N_rig_new.json"
    dst.write_text(json.dumps(new, indent=1), encoding="utf-8")
    for e, c in (cams_new or {}).items():
        (out / f"M2020_{e}_fisheye_tangential_new.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    print(f"written: {dst} - " + ("check passed; --promote to make it the candidate" if chk["ok"]
                                  else "the check FAILED: not promoted (look at the lines above; --promote --force to use it anyway)"))
    return 0 if chk["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
