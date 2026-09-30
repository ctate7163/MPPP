"""
Stop every MPPP run on this computer: process_sites, run_sites / run_all_sites, align_here / align_scape and
run_scapes, with their notebook kernels and image-processing workers.

    python scripts\\stop_runs.py            # list the runs, ask, stop them
    python scripts\\stop_runs.py --yes      # stop them without asking
    python scripts\\stop_runs.py --list     # only list them

Runs are found by their command lines (the minimised .bat windows and the Python runners); each is stopped with
everything it started (Windows: ``taskkill /T /F``).  Notebooks open in Jupyter are not touched.  The status files
under ``--root`` (``mppp_batch.json`` and each WORK folder's ``mppp_status.json``) are then marked "stopped", so a
new run can start at once.  A site stopped half-way is completed by the next run: images already processed are
reused and the rest are processed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=None,
                    help="scapes folder(s) whose status files are updated (default D:/scapes/colmap)")
    ap.add_argument("--yes", action="store_true", help="stop without asking")
    ap.add_argument("--list", action="store_true", help="only list the runs")
    a = ap.parse_args(argv)
    from mppp.runner import kill_tree, list_processes, mark_stopped, mppp_processes
    roots = [Path(r) for r in (a.root or ["D:/scapes/colmap"])]
    runs = mppp_processes(list_processes())
    if not runs:
        print("No MPPP runs are running.")
    else:
        print(f"{len(runs)} MPPP run(s):")
        for r in runs:
            print(f"  pid {r['pid']:>6}  {str(r['cmd'])[:150]}")
    if a.list:
        return 0
    if runs and not a.yes:
        try:
            ans = input("Stop them (and everything they started)? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes", "j", "ja"):
            print("Nothing stopped.")
            return 1
    killed = []
    for r in runs:
        ok = kill_tree(int(r["pid"]))
        print(f"  pid {r['pid']}: {'stopped' if ok else 'could not be stopped (already gone?)'}")
        killed.append(int(r["pid"]))
    if killed:
        time.sleep(2)
    for root in roots:
        if root.is_dir():
            for f in mark_stopped(root, killed):
                print(f"  marked stopped: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
