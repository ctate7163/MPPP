"""
Check the site definitions file after editing it: ``src/mppp/data/sites.json`` (or ``--sites-file``).

    python scripts\\check_sites.py

Prints the errors that would stop the runs (bad JSON with its line, missing or reversed sol ranges, unknown sites
in a group, unreadable stations) and warnings (overlapping sol ranges, settings that are not pipeline
settings), then the sites and groups.  Exit code 0 when there is no error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path



def pipeline_settings():
    """The setting names a site's ``settings`` may use (:class:`mppp.pipeline.Settings`, either case)."""
    from mppp.pipeline import setting_names
    return set(setting_names()) | set(setting_names(upper=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites-file", default=None, help="default: src/mppp/data/sites.json")
    a = ap.parse_args(argv)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mppp.waypoints.sites import SITES_FILE, load_site_table, validate_site_table
    f = Path(a.sites_file) if a.sites_file else SITES_FILE
    print(f"site definitions: {f}")
    err, warn = validate_site_table(f, pipeline_settings())
    for e in err:
        print(f"  ERROR    {e}")
    for w in warn:
        print(f"  warning  {w}")
    if err:
        print(f"{len(err)} error(s): fix them before running (the file is unchanged).")
        return 1
    d = load_site_table(f)
    from mppp.waypoints.sites import site_zooms, zcam_sites
    print(f"OK: {len(d['sites'])} sites ({len(zcam_sites(d))} with Mastcam-Z), groups: " +
          ", ".join(f"{g} ({len(m)})" for g, m in (d.get("groups") or {}).items()))
    for name, v in d["sites"].items():
        extra = []
        if v.get("settings"):
            extra.append(f"settings {v['settings']}")
        if v.get("no_mask_inference_at"):
            extra.append(f"no mask at {v['no_mask_inference_at']}")
        from mppp.waypoints.sites import sol_ranges
        rs = sol_ranges(v["sols"])
        sols_txt = ", ".join(f"{a}-{b}" for a, b in rs) + (" (main visit first)" if len(rs) > 1 else "")
        print(f"  {name:22s} sols {sols_txt:24s} {('zcam ' + ','.join(map(str, site_zooms(v)))) if site_zooms(v) else '':14s}  "
              f"{'; '.join(extra)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
