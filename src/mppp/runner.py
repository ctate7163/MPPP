"""
Run a pipeline function with a log on disk and a status file that says whether the run is alive.

:func:`run_logged` calls a function of :mod:`mppp.pipeline` (``align``, ``process_site``, ``run_site``) on one WORK
folder and writes, while it runs:

- ``<WORK>/runs/<time>[_<variant>|_process]/log.txt`` - every ``mppp`` log line as it comes;
- ``<WORK>/mppp_status.json`` (:class:`RunStatus`) - state (running / finished / failed), the step running and
  since when, and a heartbeat every 30 s.  A "running" status whose heartbeat is older than a few minutes means
  the runner was killed (window closed, computer restarted).

Used by ``mppp align`` / ``mppp process`` / ``mppp run`` (:mod:`mppp.cli`), ``scripts/align_scape.py`` (one WORK
folder) and ``scripts/process_sites.py``; ``scripts/_batch.py`` adds the batch lock and the process control of
``scripts/stop_runs.py``.
"""
from __future__ import annotations

import ast
import datetime
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

PathLike = Union[str, Path]

HEARTBEAT_S = 30
STALE_S = 300                                        # a "running" status without a heartbeat for this long: dead
STATUS_FILE = "mppp_status.json"
PATH_SETTINGS = {"PDS_DIR", "SCAPES_ROOT", "NAVCAM_CAMERAS", "ZCAM_FOCUS_MODEL", "WORK_DIR", "SITES_FILE"}
_NOISE = re.compile(r"^[IWE]\d{8} ")                 # glog lines of COLMAP


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def disable_quickedit() -> bool:
    """Switch off QuickEdit mode of this process's Windows console.  A click in a console window in QuickEdit mode
    starts a text selection, and every print of the process then blocks until the selection ends - a run looks
    alive (its heartbeat thread goes on) but stops.  No-op elsewhere; returns True if the mode was changed."""
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
            # the file first - a console that blocks (Windows QuickEdit selection) must not hold the log back
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self.echo:
                try:
                    print(line, flush=True)
                except (OSError, UnicodeEncodeError):
                    pass


class LogHandler(logging.Handler):
    """A logging handler that sends the ``mppp`` log records to a :class:`Log` (COLMAP's glog noise dropped)."""

    def __init__(self, log: Log, level: int = logging.INFO):
        super().__init__(level)
        self.log = log

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:                                                  # noqa: BLE001
            return
        for ln in msg.splitlines():
            if ln.strip() and not _NOISE.match(ln):
                self.log(f"    | {ln[:300]}")


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
    # a run on this computer whose process is gone is dead at once (no 5-minute wait after a stop)
    if d["alive"] and d.get("host") == socket.gethostname() and d.get("pid") and not pid_alive(int(d["pid"])):
        d["alive"] = False
    if d.get("state") in ("starting", "running") and not d["alive"]:
        d["state"] = "stopped (no heartbeat)"
    return d


def pid_alive(pid: int) -> bool:
    """Whether a process with this id runs on this computer."""
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


# --- settings from the command line -----------------------------------------------------------------------------
def python_literal(name: str, value: Any) -> str:
    """Python source for a setting given as a JSON value (strings of path settings become ``Path``; a string
    starting with ``py:`` is taken as Python source, e.g. ``"py:dict(max_ratio=0.85)"``)."""
    if isinstance(value, str) and value.startswith("py:"):
        return value[3:].strip()
    if name.upper() in PATH_SETTINGS and isinstance(value, str):
        return f"Path(r{value!r})"
    return repr(value)


def parse_set(items) -> Dict[str, Any]:
    """``["NAME=VALUE", ...]`` -> {NAME: value}; VALUE is JSON if it parses (``3``, ``true``, ``[1, 2]``,
    ``"text"``), else a Python literal (``(8.0, 2.0, 2.0)``), else a bare word or path as a string, else Python
    source passed on as ``py:`` (``dict(max_ratio=0.85)``; :func:`mppp.pipeline.normalize` evaluates it)."""
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
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(\+[A-Za-z][A-Za-z0-9_]*)+", v):
        return v                                     # 0.94.4: a word joined by "+" (MATCH_MODE=sift+lightglue)
    return "py:" + v


# --- one logged run of a WORK folder ----------------------------------------------------------------------------
class RunAlive(RuntimeError):
    """A run of the folder is still alive (its status file has a fresh heartbeat)."""


def run_logged(fn: Callable[..., Any], work: PathLike, *args: Any, stage: str = "align", key: Optional[str] = None,
               variant: str = "", echo: bool = True, force: bool = False, info: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> Dict[str, Any]:
    """
    Call ``fn(work, *args, **kwargs)`` with a log file (``<work>/runs/<time>[_<variant>|_process]/log.txt``), the
    status file ``<work>/mppp_status.json`` (heartbeat every 30 s) and a line in ``<work>/runs/runs.txt``.
    Refuses to start while another run of the folder is alive (``force`` overrides).  Returns the final status
    (``state`` finished / failed, ``result`` what ``fn`` returned when it is JSON-like, ``last_error``).
    """
    import mppp
    disable_quickedit()
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    st_path = work / STATUS_FILE
    cur = read_status(st_path)
    if cur and cur.get("alive") and not force:
        raise RunAlive(f"{work.name}: a run is still alive (pid {cur.get('pid')}, {cur.get('stage')}, "
                       f"heartbeat {cur.get('age_s', 0):.0f} s ago); wait for it or pass --force")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = work / "runs" / (stamp + ("_process" if stage == "process" else (f"_{variant}" if variant else "")))
    log = Log(run_dir / "log.txt", echo=echo)
    status = RunStatus(st_path, work=str(work), stage=stage, variant=variant or "", run_key=key,
                       run_dir=str(run_dir), log=str(run_dir / "log.txt"), mppp=mppp.__version__,
                       **(info or {})).start()
    log(f"MPPP {mppp.__version__}: {getattr(fn, '__name__', 'run')} on {work} | variant {variant or '(default)'} | "
        f"run key {key} | python {sys.executable}")
    handler = LogHandler(log)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("mppp")
    old_level = root.level
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    status.update(state="running", step_started=now())
    t0 = time.time()
    result: Any = None
    error: Optional[str] = None
    try:
        result = fn(work, *args, **kwargs)
    except Exception as e:                                               # noqa: BLE001 - recorded, then raised
        error = f"{type(e).__name__}: {e}"[:400]
        log(f"FAILED: {error}")
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    dt = (time.time() - t0) / 3600
    ok = error is None
    log(f"{getattr(fn, '__name__', 'run')}: {'finished' if ok else 'FAILED'} after {dt:.2f} h -> {run_dir}")
    rec = _jsonable(result.get("run_done") if isinstance(result, dict) and result.get("run_done") else result)
    if isinstance(rec, dict):                        # the status file keeps the numbers, not the tables
        rec = {k: v for k, v in rec.items() if not isinstance(v, (dict, list))}
    status.stop("finished" if ok else "failed", result=rec, last_error=error)
    with (work / "runs" / "runs.txt").open("a", encoding="utf-8") as f:
        f.write(f"{stamp}  {'finished' if ok else 'FAILED  '}  {stage}  variant={variant or '-'}  key={key}  "
                f"verdict={(rec or {}).get('verdict') if isinstance(rec, dict) else None}  {run_dir.name}\n")
    return status.data


def _jsonable(x: Any) -> Any:
    """``x`` as JSON-ready data, or its string when it is not."""
    try:
        return json.loads(json.dumps(x, default=str))
    except (TypeError, ValueError):
        return str(x)
