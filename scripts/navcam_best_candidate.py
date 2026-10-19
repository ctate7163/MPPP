"""
The best Navcam models available now, as a candidate folder for promote_cmods.py (MPPP v0p80).

Starts from the models in use (``src/mppp/data/cmods``: the 4 Oct 2026 consensus, v0p73 form) and carries the
stereo temperature change in one place, as the recipe of 14 Oct 2026 asks: the residual rig yaw against block
temperature that the rig-residual study measured with these models (``rig_residuals.json``, yaw vs T -0.339 +- 0.134
mdeg/degC on 16 blocks) is moved into NR cx(T):

    NR cx slope  +=  fx_NR * radians(yaw slope)            (+yaw turns the right camera's rays toward +x)

so that the rig yaw stays one constant and NR cx(T) is the only stereo temperature term.  Nothing else changes:
fx, fy and their slopes, the distortion (k4 = 0), the rig rotation, drift and translation, the early-mission offsets.
The applied term must be at least ``--min-z`` (default 2) standard errors from zero.

    python scripts\\navcam_best_candidate.py [--rig-residuals FILE] [--out DIR] [--promote]

Default ``--rig-residuals``: the newest ``<root>/camera_analysis/navcal_*/rig_residuals.json`` made with the models in
use (its ``cameras`` entry records which).  ``--promote`` runs promote_cmods.py on the folder (with a note).
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NOTE_TAG = "v0p80 best: residual rig yaw(T) moved into NR cx(T)"


def find_rig_residuals(root: Path) -> Path:
    c = sorted((root / "camera_analysis").glob("navcal_*/rig_residuals.json"), key=lambda p: p.stat().st_mtime)
    if not c:
        raise FileNotFoundError(f"no camera_analysis/navcal_*/rig_residuals.json under {root}; give --rig-residuals")
    return c[-1]


def build(rig_residuals: Path, out: Path, min_z: float = 2.0, force: bool = False) -> dict:
    from mppp.paths import cmods_dir
    from mppp.sfm.cmods_form import NAVCAM_FILES, check_form, sha12
    src = cmods_dir()
    rr = json.loads(Path(rig_residuals).read_text(encoding="utf-8"))
    vt = ((rr.get("fits") or {}).get("yaw") or {}).get("vs_T") or {}
    if vt.get("slope") is None:
        raise ValueError(f"{rig_residuals}: no yaw vs T fit")
    slope, se = float(vt["slope"]), float(vt.get("se") or float("nan"))
    z = slope / se if se and se > 0 else float("nan")
    if not (abs(z) >= min_z) and not force:
        raise ValueError(f"yaw vs T {slope:+.3f} +- {se:.3f} mdeg/degC is only {z:+.1f} sd from 0 (< {min_z}); "
                         f"nothing to move (--force to move it anyway)")
    nr = json.loads((src / "M2020_NR_fisheye_tangential.json").read_text(encoding="utf-8"))
    nl = json.loads((src / "M2020_NL_fisheye_tangential.json").read_text(encoding="utf-8"))
    th = dict(nr.get("thermal") or {})
    if NOTE_TAG in str(th.get("note", "")):
        raise ValueError("the models in use already carry this change (promoted before) - nothing to do")
    fx = float(nr["params"][0])
    d = fx * math.radians(1e-3 * slope)
    old = float(th.get("cx_px_per_degC") or 0.0)
    th["cx_px_per_degC"] = old + d
    th["cx_px_per_degC_before_v0p80"] = old
    th["yaw_T_moved_mdeg_per_degC"] = slope
    th["yaw_T_moved_se"] = se
    if th.get("sd_cx_px_per_degC") is not None:
        th["sd_cx_px_per_degC"] = math.hypot(float(th["sd_cx_px_per_degC"]), fx * math.radians(1e-3 * se))
    stamp = datetime.date.today().isoformat()
    th["note"] = (f"{th.get('note', '')}; {NOTE_TAG} ({stamp}): {old:+.4f} {d:+.4f} = {th['cx_px_per_degC']:+.4f} "
                  f"px/degC (yaw vs T {slope:+.3f} +- {se:.3f} mdeg/degC from {rig_residuals})").strip("; ")
    nr["thermal"] = th
    nr["source"] = f"{nr.get('source', '')} [{NOTE_TAG}, {stamp}]"
    out.mkdir(parents=True, exist_ok=True)
    (out / "M2020_NR_fisheye_tangential.json").write_text(json.dumps(nr, indent=1), encoding="utf-8")
    (out / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(nl, indent=1), encoding="utf-8")
    (out / "M2020_N_rig.json").write_bytes((src / "M2020_N_rig.json").read_bytes())
    fm = check_form(out)
    rep = {"from": str(src), "rig_residuals": str(rig_residuals), "yaw_vs_T_mdeg_per_degC": slope, "se": se, "z": z,
           "fx_NR": fx, "NR_cx_slope_before": old, "NR_cx_slope_after": th["cx_px_per_degC"], "added": d,
           "form_ok": fm["ok"], "form_errors": fm["errors"], "files": {n: sha12(out / n) for n in NAVCAM_FILES}}
    (out / "CANDIDATE.md").write_text(
        f"# Navcam candidate ({stamp}): {NOTE_TAG}\n\n"
        f"- from the models in use: `{src}`\n- rig residuals: `{rig_residuals}`\n"
        f"- residual rig yaw vs block temperature: {slope:+.3f} ± {se:.3f} mdeg/°C (z {z:+.1f})\n"
        f"- NR cx slope {old:+.4f} → {th['cx_px_per_degC']:+.4f} px/°C (fx_NR {fx:.1f} px: {d:+.4f})\n"
        f"- over −50..0 °C about T0 = {th.get('T0_degC', -20)} °C: {d * -30:+.2f} … {d * 20:+.2f} px of stereo pointing moved "
        f"from the rig yaw to NR cx\n- everything else unchanged; form check: {'ok' if fm['ok'] else fm['errors']}\n",
        encoding="utf-8")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:/scapes")
    ap.add_argument("--rig-residuals", help="rig_residuals.json (default: the newest under ROOT/camera_analysis)")
    ap.add_argument("--out", help="candidate folder (default ROOT/camera_analysis/navcal_v0p80_best_<date>/navcam_joint)")
    ap.add_argument("--min-z", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--promote", action="store_true", help="then promote_cmods.py on the candidate")
    a = ap.parse_args(argv)
    root = Path(a.root)
    rrp = Path(a.rig_residuals) if a.rig_residuals else find_rig_residuals(root)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"navcal_v0p80_best_{datetime.date.today():%Y%m%d}" / "navcam_joint"
    try:
        rep = build(rrp, out, a.min_z, a.force)
    except ValueError as e:
        print(f"navcam_best_candidate: {e}", flush=True)
        return 2 if "nothing to do" not in str(e) else 0
    print(json.dumps(rep, indent=1), flush=True)
    print((out / "CANDIDATE.md").read_text(encoding="utf-8"), flush=True)
    if not rep["form_ok"]:
        print("the candidate does not pass the form check - not promoted", flush=True)
        return 1
    if a.promote:
        # promote_cmods.py (old files to cmods/history/, CHANGES.md), a local git commit of src/mppp/data/cmods and
        # _transfer/promoted.json for the next delivery from Claude
        from mppp.sfm.navcal_consensus import promote_candidate
        note = (f"{NOTE_TAG}: NR cx {rep['NR_cx_slope_before']:+.4f} -> {rep['NR_cx_slope_after']:+.4f} px/degC "
                f"(yaw vs T {rep['yaw_vs_T_mdeg_per_degC']:+.3f} +- {rep['se']:.3f} mdeg/degC)")
        r = promote_candidate(out, note, ROOT)
        return int(r.get("returncode") or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
