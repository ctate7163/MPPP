"""
Before the next batch of alignments (MPPP v0p73): make sure the right camera models are in use, in the frozen form.

    python scripts\\prepare_next_run.py                 # check, promote if needed, show what will run
    python scripts\\prepare_next_run.py --check-only    # change nothing

1. **The Navcam models in use** (``src/mppp/data/cmods``) must be the v0p72 consensus of 2 Oct 2026
   (``navcal_v0p72_20261002_2\\navcam_joint``, 16 blocks) in the v0p73 NL-reference form (the stereo temperature
   change on NR cx instead of NL cx, fx and fy their own ppm/degC, k4 = 0).  MPPP 0.73 ships them promoted, so after
   ``adopt_claude.bat`` this only checks.  If the cameras in use are something else (a later hand promotion), the
   candidate is converted (``<candidate>_v0p73``) and promoted (``promote_cmods.py``, a local commit,
   ``_transfer\\promoted.json``) - unless ``--check-only`` or ``--keep``.
2. **The form**: ``mppp.sfm.cmods_form.check_form`` - k4 = 0, NL reference, no rig temperature term.  run_sites.py
   and notebook 03 refuse to start without it.
3. **What will run**: the camera models are part of the run key (v0p73), so after a promotion every finished site
   is aligned again; ``--groups`` lists them (a dry run of run_sites.py for each group).

Nothing is deleted: the reruns reuse each site's processed images, features and matches; notebook 03 rebuilds the
project (cameras, focus bins) from the new models because their fingerprint changed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def same_parameters(cur: Path, cand: Path, tol: float = 1e-9) -> dict:
    """Whether the cameras in use carry the candidate's parameters (fx ... k4 of both eyes, rig rotation and
    translation, the stereo temperature term) - bytes differ after a conversion (notes, dates), numbers do not."""
    import numpy as np
    from mppp.sfm.cmods_form import thermal_terms
    out = {"same": True, "differences": []}
    for eye in ("NL", "NR"):
        a = json.loads((cur / f"M2020_{eye}_fisheye_tangential.json").read_text(encoding="utf-8"))
        b = json.loads((cand / f"M2020_{eye}_fisheye_tangential.json").read_text(encoding="utf-8"))
        d = float(np.max(np.abs(np.asarray(a["params"], float) - np.asarray(b["params"], float))))
        if d > tol:
            out["differences"].append(f"{eye} parameters differ (max {d:.3g})")
        ta, tb = thermal_terms(a.get("thermal")), thermal_terms(b.get("thermal"))
        if abs(0.5 * (ta["fx_ppm"] + ta["fy_ppm"]) - 0.5 * (tb["fx_ppm"] + tb["fy_ppm"])) > 1e-6:
            out["differences"].append(f"{eye} thermal ppm differs")
    ra = json.loads((cur / "M2020_N_rig.json").read_text(encoding="utf-8"))
    rb = json.loads((cand / "M2020_N_rig.json").read_text(encoding="utf-8"))
    for k in ("R_sensor_from_ref", "t_sensor_from_ref_m"):
        d = float(np.max(np.abs(np.asarray(ra[k], float) - np.asarray(rb[k], float))))
        if d > tol:
            out["differences"].append(f"rig {k} differs (max {d:.3g})")
    # the stereo temperature term: NR - NL x fR/fL (the same in both conventions)
    def stereo(d0):
        nl = json.loads((d0 / "M2020_NL_fisheye_tangential.json").read_text(encoding="utf-8"))
        nr = json.loads((d0 / "M2020_NR_fisheye_tangential.json").read_text(encoding="utf-8"))
        r = float(nr["params"][0]) / float(nl["params"][0])
        return thermal_terms(nr.get("thermal"))["cx_px"] - r * thermal_terms(nl.get("thermal"))["cx_px"]
    if abs(stereo(cur) - stereo(cand)) > 1e-6:
        out["differences"].append(f"stereo cx temperature term {stereo(cur):+.5f} vs {stereo(cand):+.5f} px/degC")
    out["same"] = not out["differences"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from mppp.sfm.cmods_form import V0P72_CANDIDATE
    ap.add_argument("--candidate", default=V0P72_CANDIDATE, help="the consensus folder that should be in use")
    ap.add_argument("--check-only", action="store_true", help="check and show; change nothing")
    ap.add_argument("--keep", action="store_true", help="keep the cameras in use even if they are not the candidate")
    ap.add_argument("--root", default="D:/scapes", help="SCAPES_ROOT (for --groups)")
    ap.add_argument("--groups", nargs="*", default=["navcam_consensus", "zcam34_consensus"],
                    help="site groups to list (a dry run of run_sites.py; zcam groups with --zcam)")
    a = ap.parse_args(argv)
    from mppp.paths import cmods_dir
    from mppp.sfm.cmods_form import check_form, convert_to_nl_reference, describe, models_fingerprint
    cur = cmods_dir()
    cand = Path(a.candidate)
    print("\n".join(describe(cur)))
    rc = 0
    if not (cand / "M2020_NL_fisheye_tangential.json").is_file():
        print(f"\nthe candidate {cand} is not on this computer: only the form of the cameras in use is checked")
    else:
        conv = cand.with_name(cand.name + "_v0p73")
        src_form = check_form(cand)
        if not src_form["ok"]:                        # a v0p72 candidate: into the NL-reference form first
            r = convert_to_nl_reference(cand, conv, note="prepare_next_run.py")
            print(f"\ncandidate converted to the v0p73 form: {conv}"
                  + "".join(f"\n  {k}: NL {v['NL_was']:+.5f} -> NR {v['NR_now']:+.5f} px/degC" for k, v in r["moved"].items()))
        else:
            conv = cand
        same = same_parameters(cur, conv)
        if same["same"]:
            print(f"\nOK: the cameras in use are the candidate's ({cand.parent.name}), in the v0p73 form")
        else:
            print("\nthe cameras in use are NOT the candidate: " + "; ".join(same["differences"]))
            if a.check_only or a.keep:
                print("  (not changed: --check-only / --keep)")
                rc = 1
            else:
                from mppp.sfm.navcal_consensus import promote_candidate
                note = f"{cand.parent.name} consensus (16 blocks) in the v0p73 NL-reference form (prepare_next_run.py)"
                res = promote_candidate(conv, note, ROOT)
                if res.get("returncode"):
                    print("ERROR: the promotion failed")
                    return 2
                print("promoted" + (f", committed {res['commit']}" if res.get("commit") else " (not committed)"))
    form = check_form(cur)
    print(f"\nform v0p73 of the cameras in use: {'OK' if form['ok'] else 'NOT OK'}; fingerprint {models_fingerprint(cur)}")
    for w in form["warnings"]:
        print("  warning:", w)
    for e in form["errors"]:
        print("  ERROR:", e)
    if not form["ok"]:
        rc = rc or 3
    for g in a.groups or []:
        cmd = [sys.executable, str(ROOT / "scripts" / "run_sites.py"), "--root", a.root, "--group", g, "--dry-run"]
        if "zcam" in g:
            cmd.append("--zcam")
        print(f"\n--- {g}: what run_sites.py would do " + "-" * 40, flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in (p.stdout + p.stderr).splitlines() if "would run" in ln or "skipped" in ln or "ERROR" in ln]
        print("\n".join(lines[-40:]) or (p.stdout + p.stderr)[-2000:])
    print("\nnext: run_sites.bat (Navcam), then run_sites_zcam34.bat - or the order in the v0p73 plan" if rc == 0 else
          "\nfix the above before the next run")
    return rc


if __name__ == "__main__":
    sys.exit(main())
