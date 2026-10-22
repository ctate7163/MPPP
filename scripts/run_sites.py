"""
Process and align many sites into their own WORK folders, one after the other.

The sites and their sol ranges come from the site definitions ``src/mppp/data/sites.json`` (or ``--sites-file``).
Each site goes to ``<root>/mars2020_sol_<first sol>_<site>`` (Navcam) or ``..._zcam`` (``--zcam``, with the
Mastcam-Z frames of the zooms in the site's ``"zcam"`` list): the pipeline selects the products from the PDS
archive, processes them (reusing images already processed) and aligns them.  Each site is a separate
``scripts/align_scape.py`` process (``mppp run`` / ``mppp align``), with its own log in ``<WORK>/runs/`` and status
in ``<WORK>/mppp_status.json``; this script's own log is ``<root>/run_sites_log.txt``.

Examples::

    python scripts/run_sites.py --all                                  # every site, Navcam only
    python scripts/run_sites.py --group navcam_consensus               # what run_sites.bat runs
    python scripts/run_sites.py --group zcam34_consensus --zcam        # the Navcam + Mastcam-Z blocks
    python scripts/run_sites.py --sites rockytop sid --source processed --variant tight --set attitude_prior_deg=1.0
    python scripts/run_sites.py --status                               # every WORK folder under --root

A site whose ``<project>/run_done.json`` records the same settings (run key) is skipped; ``--force`` runs it
again.  Reruns are cheap until the alignment itself: processed images, features and matches are reused.
``--jobs 2`` aligns two sites at a time (each needs several GB of memory; one at a time is the safe default).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))                      # sibling modules (_batch)

from _batch import filter_zcam, status_table  # noqa: E402,F401  (used by process_sites.py and the tests)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="every site in the site definitions")
    g.add_argument("--group", help="a site group of the site definitions (e.g. navcam_consensus)")
    g.add_argument("--sites", nargs="+", help="site names")
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"), help="the scapes root: the WORK folders go below it (default: MPPP_SCAPES)")
    ap.add_argument("--zcam", action="store_true", help="the Navcam + Mastcam-Z blocks (..._zcam) of the sites with a \"zcam\" list")
    ap.add_argument("--source", choices=("pds", "processed"), default="pds",
                    help="pds (default): select and process, then align; processed: align what is processed")
    ap.add_argument("--variant", default="", help="results in <WORK>/colmap_<variant>")
    ap.add_argument("--hazcam", default=None, choices=("front", "rear", "both", "none"),
                    help="the Hazcam frames: front, rear, both (the default since 0.93.1) or none")
    ap.add_argument("--settings", default=None, help="a JSON file of settings for every site")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a pipeline setting")
    ap.add_argument("--sites-file", default=None, help="site definitions (default mppp/data/sites.json)")
    ap.add_argument("--force", action="store_true", help="run sites that are already finished again")
    ap.add_argument("--force-start", action="store_true",
                    help="start even if the batch file says another batch is running on --root")
    ap.add_argument("--jobs", type=int, default=1, help="sites aligned at the same time (default 1)")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    ap.add_argument("--status", action="store_true", help="the run status of every WORK folder under --root")
    a = ap.parse_args(argv)

    if not a.root:
        ap.error("--root (or the MPPP_SCAPES environment variable) is required")
    root = Path(a.root)
    if a.status:
        return status_table(root)
    from _batch import BatchRunning, batch_lock, choose_sites, finished, site_settings
    from mppp.pipeline import run_key
    from mppp.runner import STATUS_FILE, Log, parse_set, read_status
    from mppp.waypoints.sites import load_site_table
    table = load_site_table(a.sites_file)
    sites = choose_sites(table, a, ap)
    unknown = [s for s in sites if s not in table["sites"]]
    if unknown:
        print(f"ERROR: not in the site definitions: {unknown}", file=sys.stderr)
        return 2
    sites, not_z = filter_zcam(sites, table, a.zcam, bool(a.sites))
    sets = parse_set(a.set)
    if a.hazcam:
        sets["hazcams"] = a.hazcam
        a.set = list(a.set) + [f"hazcams={a.hazcam}"]          # passed on to align_scape.py
    root.mkdir(parents=True, exist_ok=True)
    log = Log(root / "run_sites_log.txt")
    if not_z:
        log(f"  sites without Mastcam-Z zooms (sites.json \"zcam\": []), left out: {', '.join(not_z)}")
    log(f"run_sites: {len(sites)} sites, root {root}, Mastcam-Z {a.zcam}, source {a.source}, "
        f"variant {a.variant or '(default)'}, jobs {a.jobs}, settings {sets or '-'}")
    # which camera models the alignments will start from, and whether they follow the frozen forms
    from mppp.calib import check_form, describe, models_fingerprint
    for line in describe():
        log("  " + line)
    form = check_form()
    log(f"  camera models fingerprint {models_fingerprint()} (part of the run key); form: "
        + ("ok" if form["ok"] else "NOT OK"))
    for w in form["warnings"]:
        log(f"    warning: {w}")
    for e in form["errors"]:
        log(f"    ERROR: {e}")
    if not form["ok"] and not a.dry_run:
        log("not started: the camera models in use do not follow the frozen forms (mppp cmods list)")
        return 4

    todo = []
    for s in sites:
        st, settings = site_settings(root, s, a.zcam, a, dict(sets, source=a.source, variant=a.variant))
        work = st.work
        if a.source == "processed" and not (work / "processed").is_dir():
            log(f"  {s}: skipped - {work} has no processed/ (use --source pds)")
            continue
        key = run_key(settings)
        fin, why = finished(work, settings.variant, key)      # 0.93: a Hazcam run's own variant (colmap_hazcam)
        if fin and not a.force:
            log(f"  {s}: skipped - {why}")
            continue
        todo.append((s, work))
        log(f"  {s}: {'would run' if a.dry_run else 'queued'} ({why}) -> {work}")
    if a.dry_run:
        return 0
    try:                                   # one batch per scapes folder
        batch = batch_lock(root, "run_sites", force=a.force_start)
    except BatchRunning as e:
        log(f"not started: {e}")
        print(f"\nMPPP: {e}", file=sys.stderr)
        return 3

    def cmd(work: Path):
        c = [sys.executable, str(HERE / "align_scape.py"), str(work), "--source", a.source]
        if a.variant:
            c += ["--variant", a.variant]
        if a.settings:
            c += ["--settings", a.settings]
        if a.sites_file:
            c += ["--sites-file", a.sites_file]
        for s in a.set:
            c += ["--set", s]
        return c

    running = []
    results = {}
    queue = list(todo)
    while queue or running:
        while queue and len(running) < max(1, a.jobs):
            s, work = queue.pop(0)
            if (work / "SKIP").is_file():                                    # 21 Oct: a block taken out by hand
                log(f"site {s}: skipped - {work / 'SKIP'} exists")
                continue
            cur = read_status(work / STATUS_FILE)
            if cur and cur.get("alive"):                                 # e.g. align_here.bat there
                log(f"site {s}: skipped - a run of {work.name} is going on (pid {cur.get('pid')})")
                continue
            batch.update(site=s)
            log(f"site {s}: start (log in {work / 'runs'})")
            out = None if a.jobs <= 1 else subprocess.DEVNULL
            running.append((s, work, time.time(), subprocess.Popen(cmd(work), stdout=out, stderr=out)))
        time.sleep(2)
        for item in list(running):
            s, work, t0, p = item
            if p.poll() is not None:
                running.remove(item)
                if p.returncode == 5:                                        # align_scape: SKIP file
                    log(f"site {s}: skipped - {work / 'SKIP'} exists")
                    continue
                results[s] = p.returncode == 0
                log(f"site {s}: {'finished' if p.returncode == 0 else 'FAILED (see its log)'} after "
                    f"{(time.time() - t0) / 3600:.2f} h")

    n_bad = sum(not v for v in results.values())
    batch.stop("finished" if n_bad == 0 else "failed", site=None)
    log(f"run_sites finished: {len(results) - n_bad} ok, {n_bad} failed, {len(sites) - len(results)} skipped")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
