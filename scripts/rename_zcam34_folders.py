"""
Rename the Mastcam-Z WORK folders to the v0p50 name: ``<site>_colmap_nav_zcam34`` -> ``<site>_colmap_zcam34``.

    python scripts\\rename_zcam34_folders.py                 # list what would be renamed
    python scripts\\rename_zcam34_folders.py --apply         # rename

MPPP 0.50 still finds the old names (``mppp.sfm.sites.work_folder`` uses ``<site>_colmap_nav_zcam34`` while no
``<site>_colmap_zcam34`` exists), so this is tidying, not required.  A folder with a live run (its
``mppp_status.json`` heartbeat) or with a folder of the new name beside it is left alone.  The absolute paths in the
text files directly inside each ``colmap*`` project (project.json, features.json, run_settings.json,
open_in_colmap*.bat, colmap_gui.ini, ...) are rewritten to the new folder name, so features and matches are reused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TEXT = (".json", ".bat", ".ini", ".txt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:/scapes/colmap")
    ap.add_argument("--apply", action="store_true", help="rename (default: only list)")
    a = ap.parse_args(argv)
    from mppp.runner import STATUS_FILE, read_status
    from mppp.sfm.sites import LEGACY_ZCAM_SUFFIX, ZCAM_SUFFIX
    root = Path(a.root)
    n = 0
    for old in sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(LEGACY_ZCAM_SUFFIX)):
        new = old.with_name(old.name[: -len(LEGACY_ZCAM_SUFFIX)] + ZCAM_SUFFIX)
        st = read_status(old / STATUS_FILE)
        if st and st.get("alive"):
            print(f"  skip {old.name}: a run is alive (pid {st.get('pid')})")
            continue
        if new.exists():
            print(f"  skip {old.name}: {new.name} exists")
            continue
        print(f"  {old.name} -> {new.name}" + ("" if a.apply else "   (dry run)"))
        if not a.apply:
            continue
        old.rename(new)
        n += 1
        variants = [str(old), str(old).replace("\\", "/"), str(old).replace("\\", "\\\\")]
        for proj in [p for p in new.glob("colmap*") if p.is_dir()] + [new]:
            for f in proj.iterdir():
                if f.is_file() and f.suffix.lower() in TEXT and f.stat().st_size < 50_000_000:
                    try:
                        t = f.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        continue
                    t2 = t
                    for v in variants:
                        t2 = t2.replace(v, v.replace(old.name, new.name))
                    if t2 != t:
                        f.write_text(t2, encoding="utf-8")
                        print(f"      paths updated in {f.relative_to(new)}")
    print(f"{n} folders renamed" if a.apply else "dry run: nothing renamed (add --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
