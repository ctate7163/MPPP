"""
Check the site definitions file after editing it: ``src/mppp/data/sites.json`` (or ``--sites-file``).

    python scripts\\check_sites.py

Prints the errors that would stop the runs (bad JSON with its line, missing or reversed sol ranges, unknown sites
in a group, unreadable stations) and warnings (overlapping sol ranges, settings that are not pipeline
settings), then the sites and groups.  Exit code 0 when there is no error.

With a scapes root (``--root``, else ``MPPP_SCAPES``; 0.94.4) each site's line also says when its Navcam block and its
Navcam + Mastcam-Z block last finished, with which MPPP version, variant and verdict (``<WORK>/colmap*/run_done.json``),
or that one is running now (``mppp_status.json``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path



def pipeline_settings():
    """The setting names a site's ``settings`` may use (:class:`mppp.pipeline.Settings`, either case)."""
    from mppp.pipeline import setting_names
    return set(setting_names()) | set(setting_names(upper=True))


def last_run(work: Path) -> str:
    """When the block in ``work`` last finished: "23 Oct 11:55 0.94.0 lightglue pass", "running since ...", or "-"."""
    if not work.is_dir():
        return "-"
    try:
        st = json.loads((work / "mppp_status.json").read_text(encoding="utf-8"))
        if st.get("state") == "running":
            from mppp.runner import pid_alive
            if pid_alive(int(st.get("pid") or 0)):
                return f"running since {str(st.get('started', ''))[5:16].replace('T', ' ')} ({st.get('mppp', '?')})"
    except (OSError, ValueError, TypeError):
        pass
    best = None
    for rd in work.glob("colmap*/run_done.json"):
        try:
            d = json.loads(rd.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        when = str(d.get("finished") or "")
        if best is None or when > best[0]:
            best = (when, d, rd.parent.name)
    if best is None:
        return "not finished"
    when, d, folder = best
    import datetime
    try:
        w = datetime.datetime.fromisoformat(when).strftime("%d %b %H:%M")
    except ValueError:
        w = when or "?"
    variant = folder[len("colmap_"):] if folder.startswith("colmap_") else ""
    return " ".join(x for x in (w, str(d.get("mppp") or "?"), variant, str(d.get("verdict") or "")) if x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites-file", default=None, help="default: src/mppp/data/sites.json")
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"),
                    help="the scapes root, for the last runs (default MPPP_SCAPES; none: no run column)")
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
    root = Path(a.root) if a.root and Path(a.root).is_dir() else None
    if root:
        print(f"last runs under {root}: Navcam block | Navcam + Mastcam-Z block (finished, MPPP version, variant, verdict)")
    from mppp.waypoints.sites import work_folder
    for name, v in d["sites"].items():
        extra = []
        if root:
            try:
                runs = f"N: {last_run(work_folder(root, name, False, v['sols']))}"
                if site_zooms(v):
                    runs += f" | Z: {last_run(work_folder(root, name, True, v['sols']))}"
                extra.append(runs)
            except Exception as e:                       # noqa: BLE001  (a listing never fails on one site)
                extra.append(f"last run unknown ({type(e).__name__})")
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
