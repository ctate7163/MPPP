"""
Select and process the images of many sites (or some of them) into their WORK folders, without aligning them.

The sites and their sol ranges come from the site definitions ``src/mppp/data/sites.json`` (or ``--sites-file``).
Each site goes to ``<root>/<site>_colmap`` (Navcam) or ``<root>/<site>_colmap_nav_zcam34`` (``--zcam``, with the
Mastcam-Z 34 mm frames).  For each site, sections 1-2 of the newest notebook 03 in ``notebooks/`` run with its
default settings: one PDS product per exposure is selected from the archive, then processed into
``<WORK>/processed`` (8-bit PNG, terrain mask, CAHV + waypoint pose, manifest).  Images already processed with
the same configuration are reused.  The site's ``settings`` and ``no_mask_inference_at`` in the site
definitions, ``<WORK>/mppp_settings.json``, ``--settings`` and ``--set`` apply as in an alignment run.

Then align a site by double-clicking ``align_here.bat`` copied into its WORK folder (or
``python scripts\\align_scape.py <WORK>``): it aligns these processed images with notebook 03's defaults.

Examples (Windows, in the environment that runs the notebooks)::

    python scripts\\process_sites.py --all                          # every site, Navcam
    python scripts\\process_sites.py --group nav_zcam34 --zcam       # Navcam + Mastcam-Z 34 mm
    python scripts\\process_sites.py --sites rockytop sid south_arm
    python scripts\\process_sites.py --sites sid --set SKY_ELEVATION_DEG=20 --force
    python scripts\\process_sites.py --status                        # every WORK folder under --root

Only one batch (process_sites or run_sites) runs on a scapes folder at a time: a second start stops at once and
says which one is running (``stop_mppp.bat`` / ``scripts\\stop_runs.py`` stops it).  A site whose WORK folder
has a run going on (e.g. ``align_here.bat``) is skipped.

A site whose ``<WORK>/processed/process_done.json`` records the same settings is skipped (``--force`` processes
it again, still reusing unchanged images).  Each site has its own log in ``<WORK>/runs/<time>_process/log.txt``
and its status in ``<WORK>/mppp_status.json``; this script's log is ``<root>/process_sites_log.txt``.  A site
that fails is logged and the next one starts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def choose_sites(table, a, ap):
    if a.all:
        return list(table["sites"])
    if a.group:
        if a.group not in table.get("groups", {}):
            raise SystemExit(f"ERROR: no group {a.group!r}; groups: {sorted(table.get('groups', {}))}")
        return list(table["groups"][a.group])
    if a.sites:
        return list(a.sites)
    ap.error("choose --all, --group or --sites (or --status)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="every site in the site definitions")
    g.add_argument("--group", help="a site group of the site definitions (e.g. nav_zcam34, navcam_consensus)")
    g.add_argument("--sites", nargs="+", help="site names")
    ap.add_argument("--root", default="D:/scapes/colmap", help="the WORK folders go below it")
    ap.add_argument("--zcam", action="store_true", help="with the Mastcam-Z 34 mm frames (<site>_colmap_nav_zcam34)")
    ap.add_argument("--settings", default=None, help="a JSON file of notebook 03 settings for every site")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a notebook 03 setting")
    ap.add_argument("--sites-file", default=None, help="site definitions (default mppp/data/sites.json)")
    ap.add_argument("--force", action="store_true", help="process sites already processed with these settings again")
    ap.add_argument("--force-start", action="store_true",
                    help="start even if the batch file says another batch is running on --root")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    ap.add_argument("--status", action="store_true", help="the run status of every WORK folder under --root")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.status:
        from run_sites import status_table
        return status_table(root)

    import mppp
    from mppp.runner import (STATUS_FILE, BatchRunning, Log, align, batch_lock, notebook, parse_set, processed_done,
                             process_key, read_status, work_settings)
    from mppp.sfm.sites import load_site_table, work_folder
    table = load_site_table(a.sites_file)
    sites = choose_sites(table, a, ap)
    unknown = [s for s in sites if s not in table["sites"]]
    if unknown:
        print(f"ERROR: not in the site definitions: {unknown}", file=sys.stderr)
        return 2
    sets = parse_set(a.set)
    root.mkdir(parents=True, exist_ok=True)
    log = Log(root / "process_sites_log.txt")
    log(f"process_sites: MPPP {mppp.__version__}, {notebook('03_colmap_alignment').name}, {len(sites)} sites, "
        f"root {root}, Mastcam-Z {a.zcam}, settings {sets or '-'}")
    todo = []
    for s in sites:
        work = work_folder(root, s, a.zcam)
        key = process_key(work_settings(work, s, a.settings, sets, a.sites_file))
        done = processed_done(work, key)
        if done and not a.force:
            log(f"  {s}: skipped - processed {done.get('finished')} with these settings ({done.get('images')} images)")
            continue
        todo.append((s, work))
        log(f"  {s}: {'would run' if a.dry_run else 'queued'} -> {work / 'processed'}")
    if a.dry_run:
        return 0
    try:                                   # v0p43.3: one batch per scapes folder
        batch = batch_lock(root, "process_sites", force=a.force_start)
    except BatchRunning as e:
        log(f"not started: {e}")
        print(f"\nMPPP: {e}", file=sys.stderr)
        return 3
    results, busy = {}, []
    try:
        for s, work in todo:
            cur = read_status(work / STATUS_FILE)
            if cur and cur.get("alive"):                                 # e.g. align_here.bat running there
                busy.append(s)
                log(f"site {s}: skipped - a run of {work.name} is going on (pid {cur.get('pid')}, "
                    f"{cur.get('stage') or 'align'}, cell {cur.get('cell')})")
                continue
            t0 = time.time()
            batch.update(site=s, site_started=time.strftime("%Y-%m-%dT%H:%M:%S"))
            log(f"site {s}: start (log in {work / 'runs'})")
            try:
                st = align(work, settings_file=a.settings, sets=sets, sites_file=a.sites_file, process_only=True,
                           echo=False)
                ok = st.get("state") == "finished"
                n = (st.get("result") or {}).get("images")
                why = f"{n} images" if ok else (st.get("last_error") or "see its log")
            except Exception as e:                                       # noqa: BLE001 - next site
                ok, why = False, f"{type(e).__name__}: {e}"
            results[s] = ok
            log(f"site {s}: {'processed' if ok else 'FAILED'} after {(time.time() - t0) / 60:.1f} min - {why}")
    finally:
        n_bad = sum(not v for v in results.values())
        batch.stop("finished" if n_bad == 0 else "failed", site=None, processed=sum(results.values()), failed=n_bad)
    log(f"process_sites finished: {len(results) - n_bad} processed, {n_bad} failed, "
        f"{len(sites) - len(results) - len(busy)} skipped (done), {len(busy)} skipped (busy)")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
