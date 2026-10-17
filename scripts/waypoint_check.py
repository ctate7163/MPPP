"""
Waypoint localisation check (MPPP v0p82, mppp.sfm.waypoint_check): which rover stations' waypoint positions disagree
with the bundle adjustments, over finished blocks.

    python scripts\\waypoint_check.py [--root D:\\scapes\\colmap] [--groups navcam_consensus navcam_baseline]
                                      [--variant ""] [--out DIR] [--folders WORK ...]

Each block's refined camera centres against its waypoint priors, station by station, after a robust similarity
(scale, rotation, translation; translation only for blocks under 30 m or with fewer than 5 stations).  Flagged:
horizontal residual > 1 m, or > 0.5 m and > 3 x the block's scatter.  Writes ``waypoint_flags.json`` / ``.csv`` /
``.md`` (default ``<root>/camera_analysis/waypoint_flags_<date>``); ``flagged`` there is the list for notebook 03
``NO_POSITION_PRIOR_AT``.  Exit 0 (whatever it finds), 2 when no block could be checked.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def block_folders(root: Path, groups, variant: str = ""):
    """{site: project dir} of the finished blocks of ``groups`` (the site's own WORK folder, ``colmap[_variant]``)."""
    from mppp.sfm.sites import load_site_table, site_group, work_folder
    table = load_site_table()
    out = {}
    for g in groups:
        for s in site_group(g):
            if s in out or s not in table["sites"]:
                continue
            w = work_folder(root, s, False, table["sites"][s]["sols"])
            p = w / (f"colmap_{variant}" if variant else "colmap")
            if (p / "error_input" / "poses.csv").is_file() and (p / "project.json").is_file():
                out[s] = p
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:/scapes/colmap")
    ap.add_argument("--groups", nargs="*", default=["navcam_consensus", "navcam_baseline"])
    ap.add_argument("--variant", default="", help="results of <WORK>/colmap_<variant> (default <WORK>/colmap)")
    ap.add_argument("--folders", nargs="*", help="project folders to check instead of the groups")
    ap.add_argument("--out", help="output folder (default <root>/camera_analysis/waypoint_flags_<date>)")
    a = ap.parse_args(argv)
    from mppp.sfm import waypoint_check as WC
    root = Path(a.root)
    folders = ({Path(f).parent.name if Path(f).name.startswith("colmap") else Path(f).name: Path(f) for f in a.folders}
               if a.folders else block_folders(root, a.groups, a.variant))
    results = []
    for name, p in folders.items():
        try:
            r = WC.check_block(p, name)
        except Exception as e:                                             # noqa: BLE001
            print(f"{name}: not checked ({type(e).__name__}: {e})", flush=True)
            continue
        s = r.get("similarity") or {}
        print(f"{name}: {s.get('stations', 0)} stations over {s.get('extent_m', 0):.0f} m, scale {s.get('scale_ppm', 0):+.0f} ppm, "
              f"scatter {r.get('scatter_h_m') or 0:.2f} m, flagged {', '.join(r['flagged']) or '-'}", flush=True)
        results.append(r)
    if not results:
        print("no finished block to check", flush=True)
        return 2
    m = WC.mission_flags(results)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"waypoint_flags_{datetime.date.today():%Y%m%d}"
    WC.write_flags(m, out)
    print(f"{len(m['flagged'])} flagged stations: {', '.join(m['flagged']) or 'none'} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
