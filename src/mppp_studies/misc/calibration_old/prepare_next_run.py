"""
Before the next batch of alignments: make sure the right camera models are in use, in the frozen form.

    python studies/calibration/prepare_next_run.py                          # check the models in use, show what will run
    python studies/calibration/prepare_next_run.py --candidate <folder>     # promote a consensus candidate if it differs

1. **The models in use** (``src/mppp/data/cmods``, ``MPPP_CMODS``) are described and checked against the frozen
   forms (:func:`mppp.calib.check_form`); run_sites.py and notebook 03 refuse to start otherwise.
2. **A candidate** (``--candidate``, a folder written by ``mppp.calib.fit.write_navcam``): when its parameters
   differ from the models in use, it is gated (:func:`mppp.calib.fit.check_navcam`) and promoted
   (:func:`mppp.calib.models.promote`: the files replaced, a line in CHANGES.md; commit the folder) - unless
   ``--check-only`` or ``--keep``.
3. **What will run**: the models are part of the run key, so after a promotion every finished site is aligned
   again; ``--groups`` lists them (a dry run of scripts/run_sites.py for each group).

Nothing is deleted: the reruns reuse each site's processed images, features and matches; notebook 03 rebuilds the
project (cameras, focus bins) from the new models because their fingerprint changed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # the studies/calibration folder (its modules import as siblings)
ROOT = HERE.parents[1]                          # the repository root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def same_parameters(cur: Path, cand: Path, tol: float = 1e-9) -> dict:
    """Whether the cameras in use carry the candidate's parameters (fx ... k4 of both eyes, rig rotation and
    translation, the stereo temperature term) - bytes differ (notes, dates), numbers do not."""
    import numpy as np
    from mppp.calib.constrain import thermal_terms
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

    def stereo(d0):             # the stereo temperature term: NR - NL x fR/fL
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
    ap.add_argument("--candidate", default=None, help="a consensus candidate folder that should be in use")
    ap.add_argument("--check-only", action="store_true", help="check and show; change nothing")
    ap.add_argument("--keep", action="store_true", help="keep the cameras in use even if they are not the candidate")
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"), help="SCAPES_ROOT (for --groups)")
    ap.add_argument("--groups", nargs="*", default=["ecam_consensus", "zcam34_consensus"],
                    help="site groups to list (a dry run of run_sites.py; zcam groups with --zcam)")
    a = ap.parse_args(argv)
    from mppp.paths import cmods_dir
    from mppp.calib.models import check_form, describe, models_fingerprint
    cur = cmods_dir()
    print("\n".join(describe(cur)))
    rc = 0
    if a.candidate:
        cand = Path(a.candidate)
        if not (cand / "M2020_NL_fisheye_tangential.json").is_file():
            print(f"\nthe candidate {cand} is not on this computer: only the form of the cameras in use is checked")
        else:
            same = same_parameters(cur, cand)
            if same["same"]:
                print(f"\nOK: the cameras in use are the candidate's ({cand.parent.name})")
            else:
                print("\nthe cameras in use are NOT the candidate: " + "; ".join(same["differences"]))
                if a.check_only or a.keep:
                    print("  (not changed: --check-only / --keep)")
                    rc = 1
                else:
                    from mppp.calib.fit import check_lines, check_navcam, promote_navcam
                    rep = check_navcam(cand, cur)
                    print("\n".join(check_lines(rep)))
                    if not rep["ok"]:
                        print("ERROR: the candidate is not ready to promote")
                        return 2
                    res = promote_navcam(cand, f"{cand.parent.name} consensus (prepare_next_run.py)", check=False)
                    print(f"promoted {sorted(res['files'])} into {res['folder']}; commit src/mppp/data/cmods")
    form = check_form(cur)
    print(f"\nform of the cameras in use: {'OK' if form['ok'] else 'NOT OK'}; fingerprint {models_fingerprint(cur)}")
    for w in form["warnings"]:
        print("  warning:", w)
    for e in form["errors"]:
        print("  ERROR:", e)
    if not form["ok"]:
        rc = rc or 3
    for g in a.groups or []:
        if not a.root:
            print("\n(no --root / MPPP_SCAPES: the site groups are not listed)")
            break
        cmd = [sys.executable, str(ROOT / "scripts" / "run_sites.py"), "--root", a.root, "--group", g, "--dry-run"]
        if "zcam" in g:
            cmd.append("--zcam")
        print(f"\n--- {g}: what run_sites.py would do " + "-" * 40, flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        lines = [ln for ln in (p.stdout + p.stderr).splitlines() if "would run" in ln or "skipped" in ln or "ERROR" in ln]
        print("\n".join(lines[-40:]) or (p.stdout + p.stderr)[-2000:])
    print("\nnext: run_pipeline.bat" if rc == 0 else "\nfix the above before the next run")
    return rc


if __name__ == "__main__":
    sys.exit(main())
