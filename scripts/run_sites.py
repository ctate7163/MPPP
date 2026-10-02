"""
Process and align many sites into their own WORK folders, one after the other (MPPP v0p43).

The sites and their sol ranges come from the stable site definitions ``src/mppp/data/sites.json`` (or
``--sites-file``).  Each site goes to ``<root>/mars2020_sol_<first sol>_<site>_colmap`` (Navcam) or
``..._colmap_zcam`` (``--zcam``, with the Mastcam-Z frames of the zooms in the site's ``"zcam"`` list; v0p53): notebook 03 selects the products from the PDS archive, processes
them (reusing images already processed) and aligns them.  Each site is a separate ``scripts/align_scape.py`` run,
with its own log in ``<WORK>/runs/`` and status in ``<WORK>/mppp_status.json``; this script's own log is
``<root>/run_sites_log.txt``.

Examples (Windows, in the environment that runs the notebooks)::

    python scripts\\run_sites.py --all                                  # every site, Navcam only
    python scripts\\run_sites.py --group navcam_consensus             # v0p53: what run_sites.bat runs
    python scripts\\run_sites.py --group zcam34_consensus --zcam      # the Navcam + Mastcam-Z blocks
    python scripts\\run_sites.py --sites rockytop sid --source processed --variant tight --set ATTITUDE_PRIOR_DEG=1.0
    python scripts\\run_sites.py --status                               # every WORK folder under --root

A site whose ``<project>/run_done.json`` records the same settings (run key) is skipped, and so is a folder
finished before v0p43 (no run key); ``--force`` runs them again.  Reruns are cheap until the alignment itself:
processed images, features and matches are reused.  ``--then 04 05`` runs notebooks 04 (camera models) and 05
(error analysis) over the finished sites at the end.  ``--jobs 2`` aligns two sites at a time (each needs several
GB of memory; one at a time is the safe default).
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _finished(work: Path, variant: str, key: str):
    """(finished, why) for a WORK folder and run key."""
    from mppp.sfm.workdir import project_dir
    proj = project_dir(work, variant)
    done = proj / "run_done.json"
    if done.is_file():
        try:
            d = json.loads(done.read_text(encoding="utf-8"))
        except ValueError:
            return False, "unreadable run_done.json"
        if d.get("run_key") == key:
            return True, f"finished {d.get('finished')} with these settings (verdict {d.get('verdict')})"
        return False, "finished with other settings or alignment rules"
    if (proj / "error_input" / "summary.json").is_file():
        return True, "finished before v0p43 (no run key)"
    return False, "not finished"


def filter_zcam(sites, table, zcam: bool, named: bool):
    """v0p53: with ``--zcam`` only the sites with a ``"zcam"`` list (any zoom; sites named with ``--sites`` are kept,
    with a note).  Returns (sites, left out)."""
    if not zcam:
        return list(sites), []
    from mppp.sfm.sites import zcam_sites
    z = set(zcam_sites(table))
    if named:
        for s in sites:
            if s not in z:
                print(f"note: {s} has no Mastcam-Z zooms in the site definitions; run anyway (named with --sites)")
        return list(sites), []
    return [s for s in sites if s in z], [s for s in sites if s not in z]


filter_zcam34 = filter_zcam            # before v0p53


def status_table(root: Path) -> int:
    from mppp.runner import BATCH_FILE, STATUS_FILE, read_status
    b = read_status(root / BATCH_FILE)                   # v0p43.3: the batch (process_sites / run_sites) on root
    if b:
        print(f"batch: {b.get('kind')} {'RUNNING' if b.get('alive') else b.get('state')} (pid {b.get('pid')}, "
              f"started {b.get('started')}{', now at ' + str(b.get('site')) if b.get('alive') and b.get('site') else ''})"
              + ("  - stop it with stop_mppp.bat" if b.get("alive") else ""))
    else:
        print("batch: none")
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and ((p / "processed").is_dir() or (p / STATUS_FILE).is_file())):
        st = read_status(d / STATUS_FILE)
        projs = sorted(p.name for p in d.glob("colmap*") if (p / "project.json").is_file())
        fin = [p for p in projs if (d / p / "run_done.json").is_file() or (d / p / "error_input" / "summary.json").is_file()]
        if st:
            state = st.get("state") + (" (processing)" if st.get("stage") == "process" else "")
            extra = (f"cell {st.get('cell')} since {st.get('cell_started')}" if st.get("alive")
                     else (st.get("last_error") or ""))
            age = f"{st['age_s']:.0f} s" if (st.get("alive") and st.get("age_s") is not None) else ""
            if not st.get("alive") and st.get("ended"):
                extra = f"ended {st['ended']}" + (f"; {st['last_error']}" if st.get("last_error") else "")
            if st.get("alive") and st.get("state") == "starting":          # v0p50: alive, but no cell started
                try:
                    mins = (datetime.datetime.now() - datetime.datetime.fromisoformat(st["started"])).total_seconds() / 60
                except (KeyError, ValueError, TypeError):
                    mins = 0
                if mins > 10:
                    extra = (f"STUCK? starting for {mins:.0f} min with no cell run - a paused console (press Esc in "
                             f"its window) or a kernel that does not start; else stop_mppp.bat")
        else:
            state, extra, age = "-", "", ""
        rows.append((d.name, state, age, ", ".join(fin) or "-", str(extra)[:90]))
    w = max([len(r[0]) for r in rows] + [10])
    print(f"{'folder':{w}s}  {'last run':22s} {'heartbeat':>9s}  {'finished projects':28s} note")
    for r in rows:
        print(f"{r[0]:{w}s}  {r[1]:22s} {r[2]:>9s}  {r[3]:28s} {r[4]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="every site in the site definitions")
    g.add_argument("--group", help="a site group of the site definitions (e.g. navcam_consensus)")
    g.add_argument("--sites", nargs="+", help="site names")
    ap.add_argument("--root", default="D:/scapes/colmap", help="SCAPES_ROOT: the WORK folders go below it")
    ap.add_argument("--zcam", action="store_true", help="the Navcam + Mastcam-Z blocks (..._colmap_zcam) of the sites with a \"zcam\" list")
    ap.add_argument("--source", choices=("pds", "processed"), default="pds",
                    help="pds (default): select and process, then align; processed: align what is processed")
    ap.add_argument("--variant", default="", help="results in <WORK>/colmap_<variant>")
    ap.add_argument("--settings", default=None, help="a JSON file of notebook 03 settings for every site")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a notebook 03 setting")
    ap.add_argument("--sites-file", default=None, help="site definitions (default mppp/data/sites.json)")
    ap.add_argument("--force", action="store_true", help="run sites that are already finished again")
    ap.add_argument("--force-start", action="store_true",
                    help="start even if the batch file says another batch is running on --root")
    ap.add_argument("--jobs", type=int, default=1, help="sites aligned at the same time (default 1)")
    ap.add_argument("--then", nargs="*", default=[], metavar="NB", help="then notebooks 04 and/or 05 over the finished sites")
    ap.add_argument("--dry-run", action="store_true", help="list what would run")
    ap.add_argument("--status", action="store_true", help="the run status of every WORK folder under --root")
    a = ap.parse_args(argv)

    root = Path(a.root)
    if a.status:
        return status_table(root)
    from mppp.runner import (STATUS_FILE, BatchRunning, Log, batch_lock, parse_set, read_status, run_analyses,
                             run_key, work_settings)
    from mppp.sfm.sites import load_site_table, site_label, work_folder
    table = load_site_table(a.sites_file)
    if a.all:
        sites = list(table["sites"])
    elif a.group:
        if a.group not in table.get("groups", {}):
            print(f"ERROR: no group {a.group!r}; groups: {sorted(table.get('groups', {}))}", file=sys.stderr)
            return 2
        sites = list(dict.fromkeys(table["groups"][a.group]))         # v0p53: a site listed twice runs once
    elif a.sites:
        sites = list(a.sites)
    else:
        ap.error("choose --all, --group or --sites (or --status)")
    unknown = [s for s in sites if s not in table["sites"]]
    if unknown:
        print(f"ERROR: not in the site definitions: {unknown}", file=sys.stderr)
        return 2
    sites, not_z = filter_zcam(sites, table, a.zcam, bool(a.sites))
    sets = parse_set(a.set)
    root.mkdir(parents=True, exist_ok=True)
    log = Log(root / "run_sites_log.txt")
    if not_z:
        log(f"  sites without Mastcam-Z zooms (sites.json \"zcam\": []), left out: {', '.join(not_z)}")
    log(f"run_sites: {len(sites)} sites, root {root}, Mastcam-Z {a.zcam}, source {a.source}, "
        f"variant {a.variant or '(default)'}, jobs {a.jobs}, settings {sets or '-'}")

    todo = []
    for s in sites:
        work = work_folder(root, s, a.zcam, table["sites"][s]["sols"])
        if a.source == "processed" and not (work / "processed").is_dir():
            log(f"  {s}: skipped - {work} has no processed/ (use --source pds)")
            continue
        key = run_key(work_settings(work, s, a.settings, sets, a.sites_file), a.source, a.variant)
        fin, why = _finished(work, a.variant, key)
        if fin and not a.force:
            log(f"  {s}: skipped - {why}")
            continue
        todo.append((s, work))
        log(f"  {s}: {'would run' if a.dry_run else 'queued'} ({why}) -> {work}")
    if a.dry_run:
        return 0
    try:                                   # v0p43.3: one batch per scapes folder
        batch = batch_lock(root, "run_sites", force=a.force_start)
    except BatchRunning as e:
        log(f"not started: {e}")
        print(f"\nMPPP: {e}", file=sys.stderr)
        return 3

    def cmd(work: Path):
        c = [sys.executable, str(ROOT / "scripts" / "align_scape.py"), str(work), "--source", a.source]
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
            cur = read_status(work / STATUS_FILE)
            if cur and cur.get("alive"):                                 # v0p43.3: e.g. align_here.bat there
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
                results[s] = p.returncode == 0
                log(f"site {s}: {'finished' if p.returncode == 0 else 'FAILED (see its log)'} after "
                    f"{(time.time() - t0) / 3600:.2f} h")

    if a.then:
        labels = {}
        for s in sites:
            work = work_folder(root, s, a.zcam, table["sites"][s]["sols"])
            if (work / ("colmap" if not a.variant else f"colmap_{a.variant}") / "error_input" / "summary.json").is_file():
                labels[f"{site_label(s)}{' N+Z' if a.zcam else ''}"] = str(work)
        if labels:
            run_analyses(labels, root, a.then, log)
        else:
            log("no finished site: notebooks 04 / 05 skipped")
    n_bad = sum(not v for v in results.values())
    batch.stop("finished" if n_bad == 0 else "failed", site=None)
    log(f"run_sites finished: {len(results) - n_bad} ok, {n_bad} failed, {len(sites) - len(results)} skipped")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
