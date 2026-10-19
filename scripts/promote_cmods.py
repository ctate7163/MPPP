"""
Make new camera models the models in use (``src/mppp/data/cmods``; ``MPPP_CMODS`` overrides) - a thin wrapper
around :func:`mppp.calib.models.promote`.

    python scripts/promote_cmods.py <candidate folder> --note "why these are better"
    python scripts/promote_cmods.py <path>/M2020_ZCAM034_focus_model_candidate.json --note "7-block refit"
    python scripts/promote_cmods.py --list

Accepted files (a folder is searched for them): ``M2020_NL_fisheye_tangential.json``,
``M2020_NR_fisheye_tangential.json``, ``M2020_N_rig.json`` (the Navcam cameras and rig), ``M2020_<F|R><L|R>.json``
(Hazcam), ``M2020_ZCAM<zoom>_focus_model*.json`` (a Mastcam-Z zoom, saved as ``M2020_ZCAM<zoom>_focus_model.json``).

Each file is checked against its frozen form (:mod:`mppp.calib.forms`) before anything is copied; the file in use
is replaced and ``cmods/CHANGES.md`` gets a line with the source, the note and the SHA-256 of each file.  The
replaced models are in the git history: commit the promotion (``git add src/mppp/data/cmods; git commit``).
Existing projects are rebuilt from the new models on their next notebook 03 run (the models' fingerprint is part
of the run key); features and matches are reused.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    from mppp.calib import models as M
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="folders or files with the new models")
    ap.add_argument("--note", default="", help="why these are the best models (goes into CHANGES.md)")
    ap.add_argument("--list", action="store_true", help="show the current models")
    ap.add_argument("--dry-run", action="store_true", help="check and show, copy nothing")
    a = ap.parse_args(argv)
    if a.list or not a.sources:
        print("\n".join(M.describe()))
        return 0
    try:
        todo = M.collect(a.sources)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    if not todo:
        print("ERROR: no camera model files found in", a.sources)
        return 1
    for name, f in todo.items():
        try:
            print(f"  {name:40s} <- {f}  ({M.check_file(name, f)})")
        except Exception as e:                                         # noqa: BLE001
            print(f"ERROR: {f} does not load as {name}: {e}; nothing copied")
            return 1
    if a.dry_run:
        return 0
    res = M.promote(a.sources, note=a.note)
    print(f"{len(res['files'])} file(s) now in {res['folder']} ({res['stamp']}); commit src/mppp/data/cmods")
    return 0


if __name__ == "__main__":
    sys.exit(main())
