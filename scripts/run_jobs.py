"""
v0p75: a job folder the PC watches, so that runs can be started from elsewhere (Claude's cloud session through
the Claude desktop link, which can write and read files in the scapes folder but has no shell on the PC).

    python scripts\\run_jobs.py --root D:\\scapes\\colmap          (or watch_jobs.bat: minimised window, stays up)

Every ``<root>\\_jobs\\<name>.job`` is a text file whose first non-comment line is **an MPPP script and its
arguments**, e.g.::

    run_sites.py --root D:\\scapes\\colmap --sites chal_rocks --zcam --set ZCAM_ZOOMS=[34] --force

Only scripts in MPPP's own ``scripts`` folder run (``run_sites.py``, ``run_pipeline.py``, ``zcam_calibration.py``,
``promote_cmods.py``, ``stop_runs.py``, ...); nothing else is executed, and the job file is the whole command, so
whoever runs the watcher can read every job in the folder.  Jobs run one at a time, in name order, with this
python; the output goes to ``<name>.log`` as it is produced and ``<name>.done`` holds the exit code and times.
A job whose ``.done`` exists is not run again (delete the ``.done`` to rerun it).  ``<root>\\_jobs\\stop`` ends
the watcher after the current job; ``stop_runs.py`` (as a job, or by hand) stops a running alignment.
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
SCRIPTS = ROOT / "scripts"
JOBS_DIR = "_jobs"


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def parse_job(text: str):
    """(script path, arguments) of a job file, or raises ValueError: the first non-comment line, split like a
    shell command; the script must be a ``.py`` in MPPP's scripts folder (no path, no other interpreter)."""
    import shlex
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = shlex.split(s, posix=False)
        parts = [p.strip('"') for p in parts]
        name = Path(parts[0]).name
        if name != parts[0] or not name.endswith(".py"):
            raise ValueError(f"the command must start with a script name from MPPP's scripts folder, not {parts[0]!r}")
        script = SCRIPTS / name
        if not script.is_file():
            raise ValueError(f"no such MPPP script: {name}")
        return script, parts[1:]
    raise ValueError("empty job file")


def run_job(job: Path, python: str = sys.executable) -> int:
    log = job.with_suffix(".log")
    done = job.with_suffix(".done")
    t0 = time.time()
    rec = {"job": job.name, "started": _now(), "python": python}
    try:
        script, args = parse_job(job.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        rec.update({"finished": _now(), "exit_code": 2, "error": str(e)})
        log.write_text(f"[run_jobs] not run: {e}\n", encoding="utf-8")
        done.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return 2
    cmd = [python, str(script)] + args
    rec["command"] = cmd
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"[run_jobs] {_now()} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
        rec["pid"] = p.pid
        done.write_text(json.dumps(dict(rec, running=True), indent=1), encoding="utf-8")   # visible while it runs
        rc = p.wait()
        fh.write(f"[run_jobs] {_now()} exit {rc} after {(time.time() - t0) / 60:.1f} min\n")
    rec.update({"finished": _now(), "exit_code": rc, "minutes": round((time.time() - t0) / 60, 1), "running": False})
    done.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rc


def pending(jobs: Path):
    return sorted(j for j in jobs.glob("*.job") if not j.with_suffix(".done").exists())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:/scapes/colmap", help="the scapes folder; jobs in <root>/_jobs")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between looks at the folder")
    ap.add_argument("--once", action="store_true", help="run the pending jobs and exit")
    a = ap.parse_args(argv)
    jobs = Path(a.root) / JOBS_DIR
    jobs.mkdir(parents=True, exist_ok=True)
    stop = jobs / "stop"
    if stop.exists():
        stop.unlink()
    print(f"run_jobs: watching {jobs} (python {sys.executable}); a file named 'stop' there ends it", flush=True)
    (jobs / "watcher.json").write_text(json.dumps({"started": _now(), "python": sys.executable, "pid": __import__("os").getpid(),
                                                  "root": str(Path(a.root))}, indent=1), encoding="utf-8")
    while True:
        for job in pending(jobs):
            print(f"[{_now()}] {job.name}: start", flush=True)
            rc = run_job(job)
            print(f"[{_now()}] {job.name}: exit {rc}", flush=True)
            if stop.exists():
                break
        if a.once or stop.exists():
            break
        (jobs / "watcher.json").write_text(json.dumps({"started": _now(), "alive": _now(), "python": sys.executable,
                                                      "pid": __import__("os").getpid(), "root": str(Path(a.root))},
                                                     indent=1), encoding="utf-8")
        time.sleep(a.poll)
    print("run_jobs: stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
