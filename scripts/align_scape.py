"""
Align one WORK folder with notebook 03's default settings, without Jupyter open (MPPP v0p43).

A WORK folder is a site folder made by notebook 03 (``<site>_colmap`` or ``<site>_colmap_zcam34``) or any
folder with a ``processed/`` sub-folder (images, masks and the ``mppp_manifest_v*.json`` of notebook 01 or 03).
By default the images already processed there are aligned (no PDS search, no image processing)::

    python scripts\\align_scape.py D:\\scapes\\colmap\\south_arm_colmap_zcam34

``align_here.bat`` (scripts\\windows) does the same by double-click from inside the WORK folder.

Reruns reuse what they can: the processed images and masks, the SIFT features while the images are the same,
and the matches while the features and matching settings are the same.  So:

- **leave images out**: list them in ``<WORK>\\exclude_images.txt`` (stations ``S032D1184``, ``sol:658``,
  ``seq:NCAM08111`` or file-name patterns ``ZR0_0690_*``) and run again;
- **other SfM settings**: ``--set NAME=VALUE`` (a notebook 03 setting; VALUE is JSON or Python, e.g.
  ``--set "SCHEDULE=((24,10,8),(12,4,4),(8,2,2),(8,2,2))"``), or put them in ``<WORK>\\mppp_settings.json``
  (``{"ATTITUDE_PRIOR_DEG": 1.0}``) or a file given with ``--settings``;
- **try a camera model** without losing the default results: ``--variant rational --set NAVCAM_DISTORTION="rational"
  --set NAVCAM_CAMERAS="D:/scapes/colmap/camera_analysis/navcal_v0p40/navcam_joint_rational"``; the variant goes
  to ``<WORK>\\colmap_rational`` and starts from ``colmap\\``'s features and matches.

Progress: ``<WORK>\\runs\\<time>\\log.txt`` (every cell and every [sfm] line as it is printed), the executed
notebook beside it, and ``<WORK>\\mppp_status.json`` (``--status`` prints it; a run whose heartbeat is older than
5 minutes has stopped).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def show_status(work: Path) -> int:
    from mppp.runner import STATUS_FILE, read_status
    st = read_status(work / STATUS_FILE)
    if st is None:
        print(f"{work}: no run recorded ({STATUS_FILE} missing)")
        return 1
    age = st.get("age_s")
    print(f"{work.name}: {st.get('state')}" + (f" (heartbeat {age:.0f} s ago)" if age is not None else ""))
    for k in ("variant", "source", "started", "cell", "cell_title", "cell_started", "ended", "last_error", "log"):
        if st.get(k) not in (None, ""):
            print(f"  {k:12s} {st[k]}")
    if st.get("result"):
        r = st["result"]
        print(f"  result       verdict {r.get('verdict')}, {r.get('registered_images')} of {r.get('images')} images, "
              f"rms {r.get('residual_rms_native_px')}")
    runs = work / "runs" / "runs.txt"
    if runs.is_file():
        print("  recent runs:")
        for ln in runs.read_text(encoding="utf-8").splitlines()[-5:]:
            print("   ", ln)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", help="the WORK folder (the one holding processed/)")
    ap.add_argument("--source", choices=("processed", "pds"), default="processed",
                    help="processed (default): the images already processed there; pds: select and process first")
    ap.add_argument("--variant", default="", help="results in <WORK>/colmap_<variant> instead of colmap/")
    ap.add_argument("--settings", default=None, help="a JSON file of notebook 03 settings {NAME: value}")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a notebook 03 setting")
    ap.add_argument("--sites-file", default=None, help="site definitions (default mppp/data/sites.json)")
    ap.add_argument("--force", action="store_true", help="start even if the status says a run is alive")
    ap.add_argument("--status", action="store_true", help="print the folder's run status and exit")
    ap.add_argument("--is-running", action="store_true", help="exit 0 if a run of the folder is alive, else 1")
    a = ap.parse_args(argv)
    work = Path(a.work).expanduser().resolve()
    if a.is_running:
        from mppp.runner import STATUS_FILE, read_status
        st = read_status(work / STATUS_FILE)
        return 0 if st and st.get("alive") else 1
    if a.status:
        return show_status(work)
    from mppp.runner import align, parse_set
    try:
        st = align(work, source=a.source, variant=a.variant, settings_file=a.settings, sets=parse_set(a.set),
                   sites_file=a.sites_file, force=a.force)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: st.get(k) for k in ("state", "last_error", "result", "log")}, indent=1, default=str))
    return 0 if st.get("state") == "finished" else 1


if __name__ == "__main__":
    sys.exit(main())
