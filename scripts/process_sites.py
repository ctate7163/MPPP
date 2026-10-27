"""
Select and process the images of many sites (or some of them) into their WORK folders, without aligning them.

The sites and their sol ranges come from the site definitions ``src/mppp/data/sites.json`` (or ``--sites-file``).
Each site goes to ``<root>/mars2020_sol_<first sol>_<site>`` (Navcam) or ``..._zcam`` (``--zcam``, with the
Mastcam-Z frames of the site's ``"zcam"`` zooms).  For each site :func:`mppp.pipeline.process_site` runs with the
default settings: one PDS product per exposure is selected from the archive, then processed into
``<WORK>/processed`` (8-bit PNG, terrain mask, CAHV + waypoint pose, manifest).  Images already processed with
the same configuration are reused.  The site's ``settings`` and ``no_mask_inference_at`` in the site
definitions, ``<WORK>/mppp_settings.json``, ``--settings`` and ``--set`` apply as in an alignment run.

Then align a site with ``mppp align <WORK>`` (or ``align_here.bat`` copied into its WORK folder).

Examples::

    python scripts/process_sites.py --all                          # every site, Navcam
    python scripts/process_sites.py --group zcam34_consensus --zcam   # Navcam + Mastcam-Z
    python scripts/process_sites.py --sites rockytop sid south_arm
    python scripts/process_sites.py --sites sid --set sky_elevation_deg=20 --force
    python scripts/process_sites.py --status                        # every WORK folder under --root

Only one batch (process_sites or run_sites) runs on a scapes folder at a time: a second start stops at once and
says which one is running (``stop_mppp.bat`` / ``scripts/stop_runs.py`` stops it).  A site whose WORK folder
has a run going on (e.g. ``align_here.bat``) is skipped.

A site whose ``<WORK>/processed/process_done.json`` records the same settings is skipped (``--force`` processes
it again, still reusing unchanged images).  Each site has its own log in ``<WORK>/runs/<time>_process/log.txt``
and its status in ``<WORK>/mppp_status.json``; this script's log is ``<root>/process_sites_log.txt``.  A site
that fails is logged and the next one starts.
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 27 Oct: as mppp_env.bat - two OpenMP runtimes (torch, pycolmap) abort a plain "python" start

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))                      # sibling modules (_batch)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="every site in the site definitions")
    g.add_argument("--group", help="a site group of the site definitions (e.g. ecam_consensus)")
    g.add_argument("--sites", nargs="+", help="site names")
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"), help="the WORK folders go below it (default: MPPP_SCAPES)")
    ap.add_argument("--zcam", action="store_true", help="the Navcam + Mastcam-Z blocks (..._zcam) of the sites with a \"zcam\" list")
    ap.add_argument("--settings", default=None, help="a JSON file of settings for every site")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a pipeline setting")
    ap.add_argument("--sites-file", default=None, help="site definitions (default mppp/data/sites.json)")
    ap.add_argument("--force", action="store_true", help="process sites already processed with these settings again")
    ap.add_argument("--force-start", action="store_true",
                    help="start even if the batch file says another batch is running on --root")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    ap.add_argument("--status", action="store_true", help="the run status of every WORK folder under --root")
    a = ap.parse_args(argv)
    if not a.root:
        ap.error("--root (or the MPPP_SCAPES environment variable) is required")
    root = Path(a.root)
    from _batch import BatchRunning, batch_lock, choose_sites, filter_zcam, site_settings, status_table
    if a.status:
        return status_table(root)

    import mppp
    from mppp.pipeline import process_key, process_site, processed_done
    from mppp.runner import STATUS_FILE, Log, RunAlive, parse_set, read_status, run_logged
    from mppp.waypoints.sites import load_site_table
    table = load_site_table(a.sites_file)
    sites = choose_sites(table, a, ap)
    unknown = [s for s in sites if s not in table["sites"]]
    if unknown:
        print(f"ERROR: not in the site definitions: {unknown}", file=sys.stderr)
        return 2
    sites, not_z = filter_zcam(sites, table, a.zcam, bool(a.sites))         # --zcam: sites with "zcam" zooms
    sets = parse_set(a.set)
    root.mkdir(parents=True, exist_ok=True)
    log = Log(root / "process_sites_log.txt")
    log(f"process_sites: MPPP {mppp.__version__}, {len(sites)} sites, root {root}, Mastcam-Z {a.zcam}, "
        f"settings {sets or '-'}")
    todo = []
    for s in sites:
        st, settings = site_settings(root, s, a.zcam, a, sets)
        key = process_key(settings)
        done = processed_done(st.work, key)
        if done and not a.force:
            log(f"  {s}: skipped - processed {done.get('finished')} with these settings ({done.get('images')} images)")
            continue
        todo.append((s, st.work, settings, key))
        log(f"  {s}: {'would run' if a.dry_run else 'queued'} -> {st.work / 'processed'}")
    if a.dry_run:
        return 0
    try:                                   # one batch per scapes folder
        batch = batch_lock(root, "process_sites", force=a.force_start)
    except BatchRunning as e:
        log(f"not started: {e}")
        print(f"\nMPPP: {e}", file=sys.stderr)
        return 3
    results, busy = {}, []
    try:
        for s, work, settings, key in todo:
            cur = read_status(work / STATUS_FILE)
            if cur and cur.get("alive"):                                 # e.g. align_here.bat running there
                busy.append(s)
                log(f"site {s}: skipped - a run of {work.name} is going on (pid {cur.get('pid')}, "
                    f"{cur.get('stage') or 'align'})")
                continue
            t0 = time.time()
            batch.update(site=s, site_started=time.strftime("%Y-%m-%dT%H:%M:%S"))
            log(f"site {s}: start (log in {work / 'runs'})")
            try:
                st = run_logged(process_site, work, settings, stage="process", key=key, echo=False,
                                info={"site": s})
                ok = st.get("state") == "finished"
                why = "done" if ok else (st.get("last_error") or "see its log")
            except RunAlive as e:
                ok, why = False, str(e)
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
