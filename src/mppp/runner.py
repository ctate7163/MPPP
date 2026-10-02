"""
Run the MPPP notebooks without Jupyter open (v0p43), with a log on disk and a status file that says whether a
run is alive.

:func:`run_notebook` executes a notebook with its ``parameters`` cell overridden (a cell inserted after it) and
writes, while it runs:

- ``<out>`` - the executed notebook, saved after every cell;
- ``log.txt`` beside it - one line per cell, plus every ``[sfm]`` / ``[mppp]`` progress line as it is printed;
- ``status`` (:class:`RunStatus`, e.g. ``<WORK>/mppp_status.json``) - state (running / finished / failed), the
  cell running and since when, and a heartbeat every 30 s.  A "running" status whose heartbeat is older than a
  few minutes means the runner was killed (window closed, computer restarted).

Used by ``scripts/align_scape.py`` (one WORK folder), ``scripts/run_sites.py`` (all or some sites) and
``scripts/run_scapes.py``.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

PathLike = Union[str, Path]

ROOT = Path(__file__).resolve().parents[2]          # the MPPP folder (src/mppp/runner.py)
NOTEBOOKS = ROOT / "notebooks"
HEARTBEAT_S = 30
STALE_S = 300                                        # a "running" status without a heartbeat for this long: dead
PATH_SETTINGS = {"PDS_DIR", "SCAPES_ROOT", "NAVCAM_CAMERAS", "ZCAM_FOCUS_MODEL", "WORK_DIR", "SITES_FILE"}
_NOISE = re.compile(r"^[IWE]\d{8} ")                 # glog lines of COLMAP
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def notebook(stem: str, folder: Optional[PathLike] = None) -> Path:
    """``notebooks/<stem>.ipynb``, else the highest-versioned ``<stem>_v0pNN[pM].ipynb`` (copies such as
    ``... - Copy.ipynb`` are ignored)."""
    nb = Path(folder) if folder else NOTEBOOKS
    plain = nb / f"{stem}.ipynb"
    if plain.is_file():
        return plain
    pat = re.compile(re.escape(stem) + r"_v(\d+)p(\d+)(?:p(\d+))?\.ipynb$")
    found = [(tuple(int(g or 0) for g in m.groups()), p) for p in nb.glob(f"{stem}_v*.ipynb")
             if (m := pat.fullmatch(p.name))]
    if not found:
        raise FileNotFoundError(f"no {stem}.ipynb or {stem}_v0pNN.ipynb in {nb}")
    return max(found)[1]


def disable_quickedit() -> bool:
    """v0p50: switch off QuickEdit mode of this process's Windows console.  A click in a console window in QuickEdit
    mode starts a text selection, and every print of the process then blocks until the selection ends - a run looks
    alive (its heartbeat thread goes on) but stops (seen 30 Sep 2026: butler_landing held at "starting" for 47 min).
    No-op elsewhere; returns True if the mode was changed."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-10)                    # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k32.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x0040, 0x0080
        new = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
        return bool(k32.SetConsoleMode(h, new)) and new != mode.value
    except Exception:                                # noqa: BLE001 - never stop a run for this
        return False


class Log:
    """Timestamped lines to a file (and the console)."""

    def __init__(self, path: PathLike, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()

    def __call__(self, msg: str) -> None:
        line = f"{now()}  {msg}"
        with self._lock:
            # v0p50: the file first - a console that blocks (Windows QuickEdit selection) must not hold the log back
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self.echo:
                try:
                    print(line, flush=True)
                except (OSError, UnicodeEncodeError):
                    pass


class RunStatus:
    """A JSON status file with a heartbeat thread (see the module docstring)."""

    def __init__(self, path: PathLike, **info: Any):
        self.path = Path(path)
        self.data: Dict[str, Any] = {"state": "starting", "pid": os.getpid(), "host": socket.gethostname(),
                                     "python": sys.executable, "started": now(), "heartbeat": now(), **info}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.write()

    def write(self) -> None:
        with self._lock:
            self.data["heartbeat"] = now()
            tmp = self.path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
                os.replace(tmp, self.path)
            except OSError:
                pass

    def update(self, **kw: Any) -> None:
        self.data.update(kw)
        self.write()

    def start(self) -> "RunStatus":
        def beat():
            while not self._stop.wait(HEARTBEAT_S):
                self.write()
        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()
        return self

    def stop(self, state: str, **kw: Any) -> None:
        self._stop.set()
        self.update(state=state, ended=now(), **kw)


def read_status(path: PathLike) -> Optional[Dict[str, Any]]:
    """A status file with ``alive`` (running and a heartbeat within ``STALE_S``) and ``age_s``; None if absent."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        age = (datetime.datetime.now() - datetime.datetime.fromisoformat(d.get("heartbeat"))).total_seconds()
    except (TypeError, ValueError):
        age = None
    d["age_s"] = age
    d["alive"] = d.get("state") in ("starting", "running") and age is not None and age < STALE_S
    # v0p43.3: a run on this computer whose process is gone is dead at once (no 5-minute wait after a stop)
    if d["alive"] and d.get("host") == socket.gethostname() and d.get("pid") and not pid_alive(int(d["pid"])):
        d["alive"] = False
    if d.get("state") in ("starting", "running") and not d["alive"]:
        d["state"] = "stopped (no heartbeat)"
    return d


def pid_alive(pid: int) -> bool:
    """Whether a process with this id runs on this computer (v0p43.3)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32                                     # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, int(pid))                     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259                        # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# --- one batch per scapes folder (v0p43.3) ----------------------------------------------------------------------
BATCH_FILE = "mppp_batch.json"


class BatchRunning(RuntimeError):
    """Another process_sites / run_sites batch is running on the same scapes folder."""


def batch_lock(root: PathLike, kind: str, argv=None, force: bool = False) -> "RunStatus":
    """Start the status (with heartbeat) of a batch over ``root``; refuses while another batch on ``root`` is alive
    (two batches would process the same sites at once and see each other's half-written folders)."""
    disable_quickedit()                                                  # v0p50
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cur = read_status(root / BATCH_FILE)
    if cur and cur.get("alive") and not force:
        raise BatchRunning(f"{cur.get('kind', 'a batch')} is already running on {root} (pid {cur.get('pid')}, started "
                           f"{cur.get('started')}, now at {cur.get('site') or '-'}). Stop it with stop_mppp.bat "
                           f"(or scripts/stop_runs.py) first.")
    return RunStatus(root / BATCH_FILE, kind=kind, argv=list(argv or sys.argv), site=None).start()


# --- finding and stopping MPPP runs (scripts/stop_runs.py, v0p43.3) ---------------------------------------------
RUN_MARKERS = ("_run_process.bat", "_run_sites.bat", "_run_align.bat", "process_sites.py", "run_sites.py",
               "align_scape.py", "run_scapes.py")


def mppp_processes(rows) -> list:
    """The MPPP runner processes among ``rows`` ({"pid", "ppid", "cmd"}): those whose command line names one of
    ``RUN_MARKERS`` (the minimised .bat windows and the Python runners; their notebook kernels and image workers
    are their children).  Only the top of each tree is returned (killing it with /T takes the rest)."""
    parent = {int(r["pid"]): int(r.get("ppid") or 0) for r in rows}
    mine, p = set(), os.getpid()
    while p and p not in mine:                          # this process and its ancestors (the window stopping them)
        mine.add(p)
        p = parent.get(p, 0)
    hits = {int(r["pid"]): r for r in rows if r.get("cmd") and any(m in str(r["cmd"]) for m in RUN_MARKERS)
            and int(r["pid"]) not in mine}
    return [r for pid, r in sorted(hits.items()) if int(r.get("ppid") or 0) not in hits]


def list_processes() -> list:
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


def mark_stopped(root: PathLike, pids) -> list:
    """Set ``state: stopped`` in the status files under ``root`` (the batch file and every WORK folder's) whose
    run was one of ``pids`` or whose process is gone; returns the files changed."""
    root = Path(root)
    changed = []
    files = [root / BATCH_FILE] + sorted(root.glob(f"*/{STATUS_FILE}"))
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


def python_literal(name: str, value: Any) -> str:
    """Python source for a notebook setting given as a JSON value (strings of path settings become ``Path``;
    a string starting with ``py:`` is taken as Python source, e.g. ``"py:dict(max_ratio=0.85)"``)."""
    if isinstance(value, str) and value.startswith("py:"):
        return value[3:].strip()
    if name in PATH_SETTINGS and isinstance(value, str):
        return f"Path(r{value!r})"
    return repr(value)


def inject(nb, overrides: Dict[str, str], tag: str = "injected-parameters") -> None:
    """Insert a cell setting ``overrides`` (name -> Python source) right after the cell tagged ``parameters``."""
    import nbformat
    idx = next((i for i, c in enumerate(nb.cells) if "parameters" in c.get("metadata", {}).get("tags", [])), None)
    if idx is None:
        raise ValueError("notebook has no cell tagged 'parameters'")
    src = "# injected by mppp.runner\n" + "\n".join(f"{k} = {v}" for k, v in overrides.items())
    cell = nbformat.v4.new_code_cell(src)
    cell.metadata["tags"] = [tag]
    nb.cells.insert(idx + 1, cell)


def _first_line(src: str) -> str:
    return next((ln for ln in src.splitlines() if ln.strip() and not ln.lstrip().startswith("#")), "")[:70]


def truncate_before(nb, heading: str) -> int:
    """Drop the notebook's cells from the first markdown cell that starts with ``heading`` (e.g. ``"## 3"``)
    on; returns the number of cells kept.  Raises if no cell starts with it."""
    idx = next((i for i, c in enumerate(nb.cells)
                if c.cell_type == "markdown" and c.source.lstrip().startswith(heading)), None)
    if idx is None:
        raise ValueError(f"no markdown cell starting with {heading!r}")
    del nb.cells[idx:]
    return idx


def run_notebook(src: PathLike, out: PathLike, overrides: Dict[str, str], log: Callable[[str], None],
                 timeout_h: float = 48.0, status: Optional[RunStatus] = None, cwd: Optional[PathLike] = None,
                 kernel_name: str = "python3", stop_before: Optional[str] = None) -> bool:
    """Execute ``src`` with ``overrides`` injected; the executed notebook goes to ``out`` (saved after every cell).
    Progress lines go to ``log``; ``status`` gets the cell running.  ``stop_before`` (v0p43.1): run only the cells
    above the first markdown cell starting with it (:func:`truncate_before`).  Returns True if every cell ran."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    src, out = Path(src), Path(out)
    nb = nbformat.read(str(src), as_version=4)
    if stop_before:
        truncate_before(nb, stop_before)
    inject(nb, dict(overrides, _kernel_python="__import__('sys').executable; print('kernel python', _kernel_python)"))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {"t": time.time(), "partial": ""}
    code_cells = sum(c.cell_type == "code" for c in nb.cells)

    class Client(NotebookClient):
        def output(self, outs, msg, display_id, cell_index):          # every printed line as it comes
            if msg.get("msg_type") == "stream":
                text = state["partial"] + msg.get("content", {}).get("text", "")
                *lines, state["partial"] = text.split("\n")
                for ln in lines:
                    ln = _ANSI.sub("", ln.rstrip("\r").split("\r")[-1])
                    if ln.strip() and not _NOISE.match(ln):
                        log(f"    | {ln[:300]}")
            return super().output(outs, msg, display_id, cell_index)

    def on_cell_start(cell, cell_index):
        if cell.cell_type != "code":
            return
        state["t"] = time.time()
        if status:
            status.update(state="running", cell=cell_index, cells=len(nb.cells), code_cells=code_cells,
                          cell_started=now(), cell_title=_first_line(cell.source))

    def on_cell_executed(cell, cell_index, execute_reply):
        if cell.cell_type != "code":
            return
        err = ""
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                err = _ANSI.sub("", f" | ERROR {o.get('ename')}: {o.get('evalue')}")[:400]
        log(f"  {src.stem} cell {cell_index} done ({time.time() - state['t']:.0f} s): {_first_line(cell.source)}{err}")
        try:
            nbformat.write(nb, str(out))
        except OSError:
            pass

    client = Client(nb, timeout=int(timeout_h * 3600), kernel_name=kernel_name,
                    resources={"metadata": {"path": str(cwd or src.parent)}},
                    on_cell_start=on_cell_start, on_cell_executed=on_cell_executed)
    ok, error = True, None
    try:
        client.execute()
    except CellExecutionError as e:
        ok = False
        error = _ANSI.sub("", str(e).strip().splitlines()[-1])[:400] if str(e).strip() else type(e).__name__
    except Exception as e:                                             # noqa: BLE001 - kernel died, timeout, ...
        ok = False
        error = f"{type(e).__name__}: {e}"[:400]
    try:
        nbformat.write(nb, str(out))
    except OSError:
        pass
    dt = (time.time() - t0) / 3600
    log(f"  {src.stem}: {'finished' if ok else 'FAILED - ' + str(error)} after {dt:.2f} h -> {out}")
    if status:
        status.update(last_error=error) if error else None
    return ok


# --- one WORK folder (scripts/align_scape.py) ------------------------------------------------------------------
STATUS_FILE = "mppp_status.json"
PROCESS_STOP = "## 3"                 # notebook 03: sections 1-2 select and process, section 3 on aligns
PROCESS_DONE = "process_done.json"    # <WORK>/processed/process_done.json: written by a finished processing run


# v0p65: bumped when the alignment code changes its results under unchanged notebook settings (a default changed in
# the code, not in the notebook), so that run_sites.py aligns every site again instead of skipping it as done.
# 65: Navcam aspect held, pose-guided matching with a local depth window (v0p65); before v0p65 not part of the key
ALIGN_RULES = 65


def run_key(settings: Dict[str, Any], source: str, variant: str) -> str:
    """A short hash of what decides a run's result (the notebook settings, source and variant)."""
    import hashlib
    d = {"settings": settings, "source": source, "variant": variant or ""}
    if not str(source).startswith("process:"):
        d["rules"] = ALIGN_RULES                  # v0p65: changed code defaults rerun the alignments
    txt = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(txt.encode()).hexdigest()[:16]


# v0p43.3: bumped when the selection / processing rules change, so that process_sites.py checks every site again
# (2: interrupted runs are completed instead of trimmed; Navcam tiles below 1/2 of the frame are left out)
PROCESS_RULES = 3                    # 3 (v0p44): Navcam tiles below 1/4 of the frame left out


def process_key(settings: Dict[str, Any]) -> str:
    """The run key of a processing-only run (``process_done.json``)."""
    return run_key(settings, f"process:{PROCESS_RULES}", "")


def work_settings(work: PathLike, site: Optional[str] = None, settings_file: Optional[PathLike] = None,
                  sets: Optional[Dict[str, Any]] = None, sites_file: Optional[PathLike] = None) -> Dict[str, Any]:
    """The notebook 03 settings of a run, later ones winning: the site's ``settings`` (and
    ``no_mask_inference_at``) in the site definitions, ``<work>/mppp_settings.json``, ``settings_file``,
    ``sets`` (e.g. ``--set NAME=VALUE``)."""
    from .sfm.sites import load_site_table, parse_work_folder, site_zooms
    from .sfm.workdir import SETTINGS_FILE, load_settings
    out: Dict[str, Any] = {}
    rec = (load_site_table(sites_file).get("sites", {}).get(site) or {}) if site else {}
    if isinstance(rec, dict) and parse_work_folder(work)[1] and site_zooms(rec):
        out["ZCAM_ZOOMS"] = site_zooms(rec)          # v0p53: the site's Mastcam-Z zooms (part of the run key)
    if isinstance(rec, dict):
        if rec.get("no_mask_inference_at"):
            out["NO_MASK_INFERENCE_AT"] = list(rec["no_mask_inference_at"])
        out.update(rec.get("settings") or {})
    out.update(load_settings(Path(work) / SETTINGS_FILE))
    if settings_file:
        out.update(load_settings(settings_file))
    out.update(sets or {})
    return out


def parse_set(items) -> Dict[str, Any]:
    """``["NAME=VALUE", ...]`` -> {NAME: value}; VALUE is JSON if it parses (``3``, ``true``, ``[1, 2]``,
    ``"text"``), else Python source (``dict(max_ratio=0.85)``, ``(8.0, 2.0, 2.0)``) passed on as ``py:``."""
    out: Dict[str, Any] = {}
    for s in items or ():
        if "=" not in s:
            raise ValueError(f"--set {s!r}: use NAME=VALUE")
        k, v = s.split("=", 1)
        v = v.strip()
        out[k.strip()] = _setting_value(v)
    return out


def _setting_value(v: str) -> Any:
    """JSON, else a Python literal, else a bare word or path as a string (Windows cmd drops the quotes of
    ``NAME="rational"``), else Python source (``py:``)."""
    import ast
    try:
        return json.loads(v)
    except ValueError:
        pass
    try:
        val = ast.literal_eval(v)
        return list(val) if isinstance(val, tuple) else val
    except (ValueError, SyntaxError):
        pass
    if re.fullmatch(r"[A-Za-z0-9_.:\\/-]+", v) and v not in ("True", "False", "None"):
        return v
    return "py:" + v


def align(work: PathLike, source: str = "processed", variant: str = "", settings_file: Optional[PathLike] = None,
          sets: Optional[Dict[str, Any]] = None, sites_file: Optional[PathLike] = None, force: bool = False,
          notebook_stem: str = "03_colmap_alignment", echo: bool = True,
          process_only: bool = False) -> Dict[str, Any]:
    """
    Run notebook 03 on one WORK folder (v0p43): ``source="processed"`` aligns the images already in
    ``<work>/processed``; ``"pds"`` selects and processes them first (the folder name
    ``mars2020_sol_<sol>_<site>_colmap[_zcam]`` names the site; v0p53).  The executed notebook and ``log.txt`` go to
    ``<work>/runs/<time>[_<variant>]/``, the status to ``<work>/mppp_status.json``.  Refuses to start while
    another run of the folder is alive (``force`` overrides).  Returns the final status.

    ``process_only`` (v0p43.1, ``scripts/process_sites.py``): select from the PDS archive and process into
    ``<work>/processed`` with notebook 03's settings, and stop before the alignment (section 3).  The run goes to
    ``<work>/runs/<time>_process/``; a finished run writes ``<work>/processed/process_done.json`` with its run key.
    """
    from .sfm.sites import parse_work_folder
    from .sfm.workdir import project_dir
    disable_quickedit()                                                  # v0p50
    work = Path(work).resolve()
    site, zcam = parse_work_folder(work)
    if process_only:
        source, variant = "pds", ""
    if source == "processed" and not (work / "processed").is_dir():
        raise FileNotFoundError(f"{work} has no processed/ folder: put this in a WORK folder next to processed/, "
                                f"or use source 'pds'")
    if source == "pds" and not site:
        raise ValueError(f"{work.name}: a PDS run needs a folder named mars2020_sol_<sol>_<site>_colmap[_zcam] (v0p53)")
    work.mkdir(parents=True, exist_ok=True)
    st_path = work / STATUS_FILE
    cur = read_status(st_path)
    if cur and cur.get("alive") and not force:
        raise RuntimeError(f"{work.name}: a run is still alive (pid {cur.get('pid')}, cell {cur.get('cell')}, "
                           f"heartbeat {cur.get('age_s', 0):.0f} s ago); wait for it or pass --force")
    settings = work_settings(work, site, settings_file, sets, sites_file)
    key = process_key(settings) if process_only else run_key(settings, source, variant)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = work / "runs" / (stamp + ("_process" if process_only else (f"_{variant}" if variant else "")))
    log = Log(run_dir / "log.txt", echo=echo)
    nb_src = notebook(notebook_stem)
    import mppp
    status = RunStatus(st_path, work=str(work), site=site, variant=variant or "", source=source, run_key=key,
                       run_dir=str(run_dir), log=str(run_dir / "log.txt"), notebook=str(nb_src),
                       mppp=mppp.__version__, project=str(project_dir(work, variant)), settings=settings,
                       stage="process" if process_only else "align").start()
    ov: Dict[str, str] = {"WORK_DIR": f"Path(r{str(work)!r})", "SOURCE": repr(source), "VARIANT": repr(variant or ""),
                          "RUN_KEY": repr(key), "SCAPES_ROOT": f"Path(r{str(work.parent)!r})"}
    if site:
        ov["SITE"] = repr(site)
        ov["INCLUDE_ZCAM"] = repr(bool(zcam))            # v0p53 (was INCLUDE_ZCAM34)
    if sites_file:
        ov["SITES_FILE"] = f"Path(r{str(Path(sites_file).resolve())!r})"
    for k, v in settings.items():
        ov[k] = python_literal(k, v)
    log(f"MPPP {mppp.__version__}: {nb_src.name}{' (select and process only)' if process_only else ''} on {work} | "
        f"source {source} | variant {variant or '(default)'} | run key {key} | python {sys.executable}")
    if settings:
        log(f"settings: {json.dumps(settings, default=str)}")
    ok = False
    t_start = time.time()
    try:
        ok = run_notebook(nb_src, run_dir / f"{notebook_stem}{'_process' if process_only else ''}_executed.ipynb",
                          ov, log, status=status, cwd=nb_src.parent,
                          stop_before=PROCESS_STOP if process_only else None)
    except Exception as e:                                               # noqa: BLE001
        log(f"runner error: {type(e).__name__}: {e}")
        status.data["last_error"] = f"{type(e).__name__}: {e}"
    if process_only:
        return _finish_process(work, key, ok, t_start, stamp, run_dir, status, log, nb_src)
    done = project_dir(work, variant) / "run_done.json"
    result = {}
    if done.is_file() and done.stat().st_mtime >= t_start - 1:          # written by this run
        try:
            result = json.loads(done.read_text(encoding="utf-8"))
        except ValueError:
            result = {}
    ok = ok and result.get("run_key") == key
    status.stop("finished" if ok else "failed", result=result)
    with (work / "runs" / "runs.txt").open("a", encoding="utf-8") as f:
        f.write(f"{stamp}  {'finished' if ok else 'FAILED  '}  variant={variant or '-'}  source={source}  "
                f"key={key}  verdict={result.get('verdict')}  rms={result.get('residual_rms_native_px')}  "
                f"{run_dir.name}\n")
    return status.data


def _finish_process(work: Path, key: str, ok: bool, t_start: float, stamp: str, run_dir: Path, status: RunStatus,
                    log: Callable[[str], None], nb_src: Path) -> Dict[str, Any]:
    """End of a ``process_only`` run: check the manifest was written by it, write ``process_done.json``."""
    import mppp
    from .sfm.workdir import load_manifest
    result: Dict[str, Any] = {}
    if ok:
        try:
            man, mf = load_manifest(work / "processed")
            fresh = Path(mf).stat().st_mtime >= t_start - 1
        except (FileNotFoundError, ValueError, OSError) as e:
            man, fresh = None, False
            log(f"no manifest in {work / 'processed'}: {e}")
        if man is not None and fresh:
            result = {"mppp": mppp.__version__, "run_key": key, "notebook": nb_src.name,
                      "finished": now(), "images": len(man.get("images", [])),
                      "failed": len(man.get("failed", [])), "skipped": len(man.get("skipped", []))}
            (work / "processed" / PROCESS_DONE).write_text(json.dumps(result, indent=1), encoding="utf-8")
            log(f"processed: {result['images']} images, {result['failed']} failed, {result['skipped']} skipped "
                f"by the selection rules -> {work / 'processed'}")
        else:
            ok = False
            if man is not None:
                log("the manifest was not rewritten by this run")
    status.stop("finished" if ok else "failed", result=result)
    with (work / "runs" / "runs.txt").open("a", encoding="utf-8") as f:
        f.write(f"{stamp}  {'finished' if ok else 'FAILED  '}  process  key={key}  "
                f"images={result.get('images')}  {run_dir.name}\n")
    return status.data


def processed_done(work: PathLike, key: str) -> Optional[Dict[str, Any]]:
    """The ``process_done.json`` of ``work`` if it records ``key``, else None."""
    p = Path(work) / "processed" / PROCESS_DONE
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if d.get("run_key") == key else None


def run_analyses(labels: Dict[str, str], root: PathLike, which, log: Callable[[str], None]) -> None:
    """Notebooks 04 (camera models) and 05 (error analysis) over finished WORK folders ``labels``
    ({label: folder}); results in ``<root>/camera_analysis/<date>`` and ``<root>/error_analysis/<date>``."""
    root = Path(root)
    today = datetime.date.today().isoformat()
    if "04" in which:
        log(f"notebook 04 (camera models) on {list(labels)}")
        run_notebook(notebook("04_camera_models"), root / "camera_analysis" / today / "04_camera_models_executed.ipynb",
                     {"SCAPES": "{" + ", ".join(f"{k!r}: Path(r{v!r})" for k, v in labels.items()) + "}",
                      "SCAPES_ROOT": f"Path(r{str(root)!r})",
                      "OUT": f"Path(r{str(root / 'camera_analysis' / today)!r})"}, log, cwd=NOTEBOOKS)
    if "05" in which:
        log(f"notebook 05 (error analysis) on {list(labels)}")
        run_notebook(notebook("05_error_analysis"), root / "error_analysis" / today / "05_error_analysis_executed.ipynb",
                     {"ALIGNMENTS": repr(dict(labels)), "MIN_TRACK_LENGTH": "3",
                      "OUT": f"Path(r{str(root / 'error_analysis' / today)!r})"}, log, cwd=NOTEBOOKS)
