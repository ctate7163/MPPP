"""
MPPP v0p80: promote the best Navcam models available now, then align Chal Rocks with the v0p80 block recipe and
diagnose it (navcam_best_chal_rocks.bat).

1. ``navcam_best_candidate.py --promote``: the 4 Oct consensus with the residual rig yaw(T) moved into NR cx(T)
   (the one carrier of the stereo temperature change); promoted with a local commit of src/mppp/data/cmods and
   _transfer/promoted.json.  Skipped when the models in use already carry it.
2. Chal Rocks Sid (sols 361-378 + 91-101; site chal_rocks, chal_rocks_sid before 17 Oct) aligned again into ``<WORK>/colmap_v0p80`` from the
   images already processed there (``--source processed``; features and matches seeded from ``colmap/``):
   v0p80 recipe - round 1 with cameras and rig held, the principal point held at the consensus, fx fy k1 per
   temperature bin (2.5 degC over -25..-5), one rig rotation per visit (sol-300 knot), intrinsics tied.
3. ``block_diagnose.py`` on it against the earlier run (``colmap/``): residuals within / across stations / across
   visits, per visit and temperature bin; registration per visit, rover site and station; rig per epoch.
4. With ``--large``: the same for Chal Rocks Large (sols 74-100 + 360-378).

The diagnosis goes to ``<WORK>/colmap_v0p80/diagnose/`` and is copied to ``<root>/error_analysis/chal_rocks_v0p80/``.
Log: ``<root>/mppp_pipeline/chal_rocks_v0p80_log.txt``.

    python scripts\\navcam_best_chal_rocks.py [--root D:\\scapes\\colmap] [--large] [--no-promote] [--source pds]
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
RIG_RESIDUALS = r"camera_analysis\navcal_v0p74_20261004_pipeline\rig_residuals.json"   # the 4 Oct study (16 blocks)
VARIANT = "v0p80"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:/scapes/colmap")
    ap.add_argument("--rig-residuals", help=f"default ROOT/{RIG_RESIDUALS}")
    ap.add_argument("--large", action="store_true", help="also Chal Rocks Large")
    ap.add_argument("--no-promote", action="store_true", help="align with the models in use as they are")
    ap.add_argument("--source", default="processed", choices=["processed", "pds"])
    ap.add_argument("--force", action="store_true", help="align again even if colmap_v0p80 is finished")
    a = ap.parse_args(argv)
    import mppp
    from mppp.runner import Log
    root = Path(a.root)
    (root / "mppp_pipeline").mkdir(parents=True, exist_ok=True)
    log = Log(root / "mppp_pipeline" / "chal_rocks_v0p80_log.txt")
    log(f"MPPP {mppp.__version__}: best Navcam models + Chal Rocks ({datetime.datetime.now():%Y-%m-%d %H:%M})")
    if tuple(int(x) for x in mppp.__version__.split(".")[:2]) < (0, 80):
        log(f"MPPP {mppp.__version__} is older than 0.80: run adopt_claude.bat with the v0p80 bundle first")
        return 1

    def run(cmd) -> int:
        log("  " + " ".join(str(c) for c in cmd))
        rc = subprocess.call([str(c) for c in cmd], cwd=str(ROOT))
        log(f"  exit {rc}")
        return rc

    if not a.no_promote:
        rr = Path(a.rig_residuals) if a.rig_residuals else root / RIG_RESIDUALS
        rc = run([PY, ROOT / "scripts" / "navcam_best_candidate.py", "--root", root, "--rig-residuals", rr, "--promote"])
        if rc not in (0,):
            log("the best-candidate step did not finish (see above): stopping before the alignments")
            return rc
    from mppp.sfm.cmods_form import describe
    for ln in describe():
        log("  " + ln)
    from mppp.sfm.sites import load_site_table, work_folder
    table = load_site_table()
    sites = ["chal_rocks"] + (["chal_rocks_large"] if a.large else [])
    out_root = root / "error_analysis" / "chal_rocks_v0p80"
    bad = 0
    for s in sites:
        work = work_folder(root, s, False, table["sites"][s]["sols"])
        if a.source == "processed" and not (work / "processed").is_dir():
            log(f"{s}: {work} has no processed/ - using --source pds for it")
            src = "pds"
        else:
            src = a.source
        cmd = [PY, ROOT / "scripts" / "run_sites.py", "--root", root, "--sites", s, "--variant", VARIANT, "--source", src]
        if a.force:
            cmd.append("--force")
        rc = run(cmd)
        proj = work / f"colmap_{VARIANT}"
        if rc != 0 or not (proj / "run_done.json").is_file():
            log(f"{s}: the alignment did not finish (exit {rc}; its log is in {work / 'runs'})")
            bad += 1
            continue
        cmp_ = [work / "colmap"] if (work / "colmap" / "error_input" / "native").is_dir() else []
        rc = run([PY, ROOT / "scripts" / "block_diagnose.py", proj, "--compare", *cmp_])
        d = proj / "diagnose"
        if d.is_dir():
            dst = out_root / s
            dst.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                shutil.copy2(f, dst / f.name)
            log(f"{s}: diagnosis in {d} (copied to {dst})")
    log("done" + (f" ({bad} site(s) failed)" if bad else "") + f"; summaries in {out_root}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
