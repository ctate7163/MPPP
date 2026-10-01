"""
Check the site definitions file after editing it: ``src/mppp/data/sites.json`` (or ``--sites-file``).

    python scripts\\check_sites.py

Prints the errors that would stop the runs (bad JSON with its line, missing or reversed sol ranges, unknown sites
in a group, unreadable stations) and warnings (overlapping sol ranges, settings that are not notebook 03
settings), then the sites and groups.  Exit code 0 when there is no error.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def notebook03_settings():
    """The UPPER_CASE names set in notebook 03's parameters cell."""
    import json
    from mppp.runner import notebook
    try:
        nb = json.loads(notebook("03_colmap_alignment").read_text(encoding="utf-8"))
    except (OSError, ValueError, FileNotFoundError):
        return None
    for c in nb["cells"]:
        if "parameters" in c.get("metadata", {}).get("tags", []):
            src = "".join(c["source"])
            return set(re.findall(r"^([A-Z][A-Z0-9_]*)\s*=", src, re.M))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites-file", default=None, help="default: src/mppp/data/sites.json")
    a = ap.parse_args(argv)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mppp.sfm.sites import SITES_FILE, load_site_table, validate_site_table
    f = Path(a.sites_file) if a.sites_file else SITES_FILE
    print(f"site definitions: {f}")
    err, warn = validate_site_table(f, notebook03_settings())
    for e in err:
        print(f"  ERROR    {e}")
    for w in warn:
        print(f"  warning  {w}")
    if err:
        print(f"{len(err)} error(s): fix them before running (the file is unchanged).")
        return 1
    d = load_site_table(f)
    from mppp.sfm.sites import site_zooms, zcam_sites
    print(f"OK: {len(d['sites'])} sites ({len(zcam_sites(d))} with Mastcam-Z), groups: " +
          ", ".join(f"{g} ({len(m)})" for g, m in (d.get("groups") or {}).items()))
    for name, v in d["sites"].items():
        extra = []
        if v.get("settings"):
            extra.append(f"settings {v['settings']}")
        if v.get("no_mask_inference_at"):
            extra.append(f"no mask at {v['no_mask_inference_at']}")
        print(f"  {name:22s} sols {v['sols'][0]:>4}-{v['sols'][1]:<4} {('zcam ' + ','.join(map(str, site_zooms(v)))) if site_zooms(v) else '':14s}  "
              f"{'; '.join(extra)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
