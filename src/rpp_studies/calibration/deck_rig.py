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
from typing import Any, Dict, Optional

import numpy as np

KNEE_SOL = 250.0
SOLS = np.arange(0.0, 2001.0, 10.0)
# the check: the new law against the candidate it replaces
MAX_ANGLE_CHANGE_MDEG = {"pitch": 2.0, "yaw": 4.0, "roll": 6.0}   # about each angle's station scatter (sd 1.0 / 3.4 / 5.4)
MAX_RATE_Z = 3.0                    # a rate moved by more than this many combined standard errors
MAX_BASELINE_CHANGE_UM = 60.0       # about 3x the mean's error (13 um) plus the 101-station spread


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
    b = all_rig["baseline"]
    out["baseline"] = {"mean_mm": float(b["mean_mm"]), "se_mean_um": float(b["se_mean_um"]), "b0_mm": float(b.get("b0_mm", np.nan)),
                       "um_per_100sol": float(b.get("um_per_100sol", np.nan)), "se_um_per_100sol": float(b.get("se_um_per_100sol", np.nan)),
                       "n": int(b.get("n") or 0)}
    return out


def build(all_rig: Dict[str, Any], navjoint: Dict[str, Any], base: Dict[str, Any], current: Optional[Dict[str, Any]] = None,
          label: str = "", day: Optional[str] = None) -> Dict[str, Any]:
    """The candidate ``M2020_N_rig.json`` (see the module's text)."""
    L = law(all_rig)
    day = day or datetime.date.today().isoformat()
    a = [L["pitch"]["a_mdeg"], L["yaw"]["a_mdeg"], L["roll"]["a_mdeg"]]
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
            f"length) along the base rig's translation direction.")
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
        "yaw_model": (f"one yaw for every temperature; yaw kneed at sol {KNEE_SOL:g}: {y['rate_before']:+.2f} +- {y['se_before']:.1f} "
                      f"before, {y['rate_after']:+.2f} +- {y['se_after']:.2f} mdeg/1000 sol after (deck joint, {n} stations)"),
        "translation_source": (f"the {n}-station deck joint ({day}): {bl['mean_mm']:.3f} +- {bl['se_mean_um'] / 1000:.3f} mm, no "
                               "temperature term; the base rig's direction (DJ32)"),
        "translation": "deck", "source": note.split("x_NR = R x_NL + t. ", 1)[1],
        "deck_law": L,
    })
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


def angles_vs_sol(rig: Dict[str, Any], sols=SOLS) -> np.ndarray:
    """(n, 3) pitch, yaw, roll (mdeg, the rotation vector of R at each sol) of a rig file."""
    from rpp.calib.constrain import rig_at
    return np.array([np.degrees(_rotvec(rig_at(rig, float(s))[0])) * 1e3 for s in sols])


def check(new: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """The new candidate against the one it replaces: the largest change of each angle over sols 0-2000, the rate
    changes in combined standard errors (when both carry their law), the baseline change.  {"ok", "lines", ...}."""
    d = angles_vs_sol(new) - angles_vs_sol(current)
    lines, ok = [], True
    out: Dict[str, Any] = {"max_change_mdeg": {}}
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
    ap.add_argument("--out", required=True, help="the folder M2020_N_rig.json is written into (with check.json)")
    ap.add_argument("--promote", action="store_true", help="write M2020_N_rig.json only when the check passes "
                    "(the replaced file kept as M2020_N_rig_replaced_<time>.json); without it: written as M2020_N_rig_new.json")
    ap.add_argument("--force", action="store_true", help="with --promote: write it even when the check fails")
    a = ap.parse_args(argv)
    cur = _read(a.current) if a.current else None
    new = build(_read(a.all_rig), _read(a.navjoint), _read(a.base), cur, label=a.label)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chk = check(new, cur) if cur else {"ok": True, "lines": ["no current candidate: nothing to compare"]}
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
        return 0
    dst = out / "M2020_N_rig_new.json"
    dst.write_text(json.dumps(new, indent=1), encoding="utf-8")
    print(f"written: {dst} - " + ("check passed; --promote to make it the candidate" if chk["ok"]
                                  else "the check FAILED: not promoted (look at the lines above; --promote --force to use it anyway)"))
    return 0 if chk["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
