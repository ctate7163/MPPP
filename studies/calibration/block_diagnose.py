"""
Diagnose a finished Navcam block (MPPP v0p80, mppp.sfm.diagnose): residuals within stations / across stations / across
visits, per visit, eye and temperature bin; registration to the waypoints per visit, rover site and station; the rig
per epoch; the temperature bins.  Writes <project>/diagnose/diagnose.md, .json, .png.

    python scripts\\block_diagnose.py D:\\scapes\\mars2020_sol_0361_chal_rocks_sid\\colmap_v0p80 ^
        --compare D:\\scapes\\mars2020_sol_0361_chal_rocks_sid\\colmap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # the studies/calibration folder (its modules import as siblings)
ROOT = HERE.parents[1]                          # the repository root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", help="<WORK>/colmap or <WORK>/colmap_<variant> (a WORK folder: its colmap/)")
    ap.add_argument("--compare", nargs="*", default=[], help="other runs to put beside it")
    ap.add_argument("--out", help="output folder (default <project>/diagnose)")
    a = ap.parse_args(argv)
    p = Path(a.project)
    if not (p / "project.json").is_file() and (p / "colmap" / "project.json").is_file():
        p = p / "colmap"
    from diagnose import write_report
    r = write_report(p, [Path(c) for c in a.compare], a.out)
    print(Path(r["md"]).read_text(encoding="utf-8"))
    print(f"written: {r['dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
