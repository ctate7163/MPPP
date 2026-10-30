"""
Batch bookkeeping for the site scripts: one batch per scapes folder (``process_sites.py``, ``run_sites.py``),
and finding and stopping MPPP runs on this computer (``stop_runs.py``).  Imported by the scripts beside it
(``sys.path`` holds this folder).
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mppp.runner import STATUS_FILE, RunStatus, disable_quickedit, now, pid_alive, read_status

PathLike = Union[str, Path]

BATCH_FILE = "mppp_batch.json"           # before 0.95.0: the one batch file (still read, for an old run's status)
BATCH_DIR = "mppp_batches"               # 0.95.0 (Christian, 27 Oct): one status file per batch, <kind>_<pid>.json -
#   a one-site run beside the pipeline no longer needs --force-start and no longer takes over the pipeline's batch
#   file (stop_mppp stops both, as before); the sites protect themselves with their WORK status files


class BatchRunning(RuntimeError):
    """Another process_sites / run_sites batch is running on the same scapes folder."""


def alive_batches(root: PathLike) -> List[Dict[str, Any]]:
    """The batches alive on ``root``: the status dicts (with ``file``) of ``mppp_batches/*.json`` and the old
    ``mppp_batch.json`` whose heartbeat and process are alive."""
    root = Path(root)
    files = sorted((root / BATCH_DIR).glob("*.json")) if (root / BATCH_DIR).is_dir() else []
    files.append(root / BATCH_FILE)
    out = []
    for f in files:
        cur = read_status(f)
        if cur and cur.get("alive"):
            out.append(dict(cur, file=str(f)))
    return out


def batch_lock(root: PathLike, kind: str, argv=None, force: bool = False, sites=None) -> RunStatus:
    """Start the status (with heartbeat) of a batch over ``root`` - its own file under ``mppp_batches/``.  Refuses
    while another alive batch names one of this batch's ``sites`` (two batches on one site would see each other's
    half-written folders); ``force`` starts anyway.  Finished batch files are left for the record (the newest 20)."""
    disable_quickedit()
    root = Path(root)
    d = root / BATCH_DIR
    d.mkdir(parents=True, exist_ok=True)
    mine = set(sites or [])
    for cur in alive_batches(root):
        theirs = set(cur.get("sites") or []) | ({cur.get("site")} if cur.get("site") else set())
        clash = (mine & theirs) if (mine and theirs) else None       # unknown site sets: as before, one batch at a time
        if (clash or clash is None) and not force:
            raise BatchRunning(f"{cur.get('kind', 'a batch')} is already running on {root} (pid {cur.get('pid')}, started "
                               f"{cur.get('started')}, now at {cur.get('site') or '-'}; {cur.get('file')}) and shares "
                               f"sites with this one ({', '.join(sorted(clash or [])) or 'unknown'}). Stop it with "
                               f"stop_mppp.bat (or scripts/stop_runs.py) first, or --force-start.")
    old = sorted(d.glob("*.json"), key=lambda f: f.stat().st_mtime)
    for f in old[:-20]:
        try:
            if not (read_status(f) or {}).get("alive"):
                f.unlink()
        except OSError:
            pass
    f = d / f"{kind}_{os.getpid()}.json"
    k = 1
    while f.exists() and (read_status(f) or {}).get("alive"):      # (the same process twice: tests)
        k += 1
        f = d / f"{kind}_{os.getpid()}_{k}.json"
    st = RunStatus(f, kind=kind, argv=list(argv or sys.argv), site=None).start()
    if sites:
        st.update(sites=list(sites))
    return st


# --- finding and stopping MPPP runs ----------------------------------------------------------------------------
RUN_MARKERS = ("_run_process.bat", "_run_sites.bat", "_run_align.bat", "process_sites.py", "run_sites.py",
               "align_scape.py", "run_scapes.py", "run_pipeline.py", "mppp align", "mppp run", "mppp process",
               "_run_weekend_loop.bat", "run_weekend.py", "lightglue_ab.py")      # 0.94.4: the weekend run


def mppp_processes(rows) -> List[Dict[str, Any]]:
    """The MPPP runner processes among ``rows`` ({"pid", "ppid", "cmd"}): those whose command line names one of
    ``RUN_MARKERS`` (the minimised .bat windows and the Python runners; their image workers are their children).
    Only the top of each tree is returned (killing it with /T takes the rest)."""
    parent = {int(r["pid"]): int(r.get("ppid") or 0) for r in rows}
    mine, p = set(), os.getpid()
    while p and p not in mine:                          # this process and its ancestors (the window stopping them)
        mine.add(p)
        p = parent.get(p, 0)
    hits = {int(r["pid"]): r for r in rows if r.get("cmd") and any(m in str(r["cmd"]) for m in RUN_MARKERS)
            and int(r["pid"]) not in mine}
    return [r for pid, r in sorted(hits.items()) if int(r.get("ppid") or 0) not in hits]


def list_processes() -> List[Dict[str, Any]]:
    """All processes with command lines: [{"pid", "ppid", "name", "cmd"}] (Windows: PowerShell/CIM; else ps)."""
    import subprocess
    if os.name == "nt":
        ps = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
              "ConvertTo-Json -Compress")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=120).stdout
        data = json.loads(out) if out.strip() else []
        data = data if isinstance(data, list) else [data]
        return [{"pid": d.get("ProcessId"), "ppid": d.get("ParentProcessId"), "name": d.get("Name"),
                 "cmd": d.get("CommandLine") or ""} for d in data]
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        parts = ln.split(None, 2)
        if len(parts) >= 2:
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "name": "", "cmd": parts[2] if len(parts) > 2 else ""})
    return rows


def kill_tree(pid: int) -> bool:
    """Stop a process and everything it started (Windows: taskkill /T /F)."""
    import signal
    import subprocess
    if os.name == "nt":
        r = subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"], capture_output=True, text=True)
        return r.returncode == 0
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except OSError:
        return False


def mark_stopped(root: PathLike, pids) -> List[str]:
    """Set ``state: stopped`` in the status files under ``root`` (the batch file and every WORK folder's) whose
    run was one of ``pids`` or whose process is gone; returns the files changed."""
    root = Path(root)
    changed = []
    files = [root / BATCH_FILE, root / "mppp_pipeline" / "pipeline_status.json",
             root / "mppp_pipeline" / "weekend_status.json"] + sorted((root / BATCH_DIR).glob("*.json")) \
        + sorted(root.glob(f"*/{STATUS_FILE}"))
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("state") not in ("starting", "running"):
            continue
        pid = int(d.get("pid") or 0)
        if pid in set(map(int, pids)) or (d.get("host") == socket.gethostname() and not pid_alive(pid)):
            d.update(state="stopped", ended=now(), last_error="stopped by stop_runs")
            f.write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
            changed.append(str(f))
    return changed


def status_table(root: Path) -> int:
    """Print the run status of every WORK folder under ``root`` (``--status`` of the site scripts)."""
    import datetime
    alive = alive_batches(root)
    for b in alive:
        print(f"batch: {b.get('kind')} RUNNING (pid {b.get('pid')}, started {b.get('started')}"
              f"{', now at ' + str(b.get('site')) if b.get('site') else ''}; {Path(b['file']).name})  - stop_mppp.bat stops it")
    if not alive:
        print("batch: none running")
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and ((p / "processed").is_dir() or (p / STATUS_FILE).is_file())):
        st = read_status(d / STATUS_FILE)
        projs = sorted(p.name for p in d.glob("colmap*") if (p / "project.json").is_file())
        fin = [p for p in projs if (d / p / "run_done.json").is_file() or (d / p / "error_input" / "summary.json").is_file()]
        if st:
            state = st.get("state") + (" (processing)" if st.get("stage") == "process" else "")
            extra = f"since {st.get('step_started')}" if st.get("alive") else (st.get("last_error") or "")
            age = f"{st['age_s']:.0f} s" if (st.get("alive") and st.get("age_s") is not None) else ""
            if not st.get("alive") and st.get("ended"):
                extra = f"ended {st['ended']}" + (f"; {st['last_error']}" if st.get("last_error") else "")
            if st.get("alive") and st.get("state") == "starting":          # alive, but nothing started
                try:
                    mins = (datetime.datetime.now() - datetime.datetime.fromisoformat(st["started"])).total_seconds() / 60
                except (KeyError, ValueError, TypeError):
                    mins = 0
                if mins > 10:
                    extra = (f"STUCK? starting for {mins:.0f} min - a paused console (press Esc in its window); "
                             f"else stop_mppp.bat")
        else:
            state, extra, age = "-", "", ""
        rows.append((d.name, state, age, ", ".join(fin) or "-", str(extra)[:90]))
    w = max([len(r[0]) for r in rows] + [10])
    print(f"{'folder':{w}s}  {'last run':22s} {'heartbeat':>9s}  {'finished projects':28s} note")
    for r in rows:
        print(f"{r[0]:{w}s}  {r[1]:22s} {r[2]:>9s}  {r[3]:28s} {r[4]}")
    return 0


def finished(work: Path, variant: str, key: str):
    """``(finished, why)`` for a WORK folder and run key."""
    from mppp.pipeline import run_done
    from mppp.sfm.workdir import project_dir
    d = run_done(work, variant)
    if d is not None:
        if d.get("run_key") == key:
            return True, f"finished {d.get('finished')} with these settings (verdict {d.get('verdict')})"
        return False, "finished with other settings, camera models or alignment rules"
    if (project_dir(work, variant) / "error_input" / "summary.json").is_file():
        return True, "finished without a run key"
    return False, "not finished"


def choose_sites(table: Dict[str, Any], a, ap) -> List[str]:
    """The sites of ``--all``, ``--group`` or ``--sites``."""
    if a.all:
        return list(table["sites"])
    if a.group:                                  # 0.95.1: several groups, comma-separated (as mppp calib)
        names: List[str] = []
        for grp in str(a.group).split(","):
            grp = grp.strip()
            if grp not in table.get("groups", {}):
                raise SystemExit(f"ERROR: no group {grp!r}; groups: {sorted(table.get('groups', {}))}")
            names += list(table["groups"][grp])
        return list(dict.fromkeys(names))
    if a.sites:
        return list(a.sites)
    ap.error("choose --all, --group or --sites (or --status)")
    return []


def filter_zcam(sites, table, zcam: bool, named: bool):
    """With ``--zcam`` only the sites with a ``"zcam"`` list (any zoom; sites named with ``--sites`` are kept, with
    a note).  Returns (sites, left out)."""
    if not zcam:
        return list(sites), []
    from mppp.waypoints.sites import zcam_sites
    z = set(zcam_sites(table))
    if named:
        for s in sites:
            if s not in z:
                print(f"note: {s} has no Mastcam-Z zooms in the site definitions; run anyway (named with --sites)")
        return list(sites), []
    return [s for s in sites if s in z], [s for s in sites if s not in z]


def site_settings(root: Path, site: str, zcam: bool, a, sets: Optional[Dict[str, Any]] = None):
    """``(Site, Settings)`` of one site of a batch, from the script's arguments."""
    from mppp.pipeline import Settings, load_settings, resolve_site
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table(a.sites_file)
    work = work_folder(root, site, zcam, table["sites"][site]["sols"])
    base = Settings(scapes_root=root, include_zcam=zcam)
    s = load_settings(work, site, a.settings, sets, a.sites_file, base=base)
    return resolve_site(work, s)
