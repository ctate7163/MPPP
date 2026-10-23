"""
Several days unattended (0.94.4, Christian 23 Oct 2026): the matcher and pose-guided comparisons on the two tiny
blocks, then the whole pipeline - one command, restartable, keeping Windows awake.

    python studies\\calibration\\run_weekend.py               (run_weekend.bat, which restarts it after a crash)
    python studies\\calibration\\run_weekend.py --status      what is done
    python studies\\calibration\\run_weekend.py --dry-run     the plan

Steps (state in ``<root>\\mppp_pipeline\\weekend_state.json``, log in ``weekend_log.txt``; started again, it goes on
after the last finished step - a step that failed is not repeated, ``--redo STEP`` runs it again):

 1. ``wait``       - until nothing else runs on ROOT: no run_sites / process_sites batch and no alignment whose
                     status file is alive (a test_lightglue.bat going on, an align_here.bat); at most --wait-h
 2. ``ab_chal``    - scripts/lightglue_ab.py on Chal Rocks tiny, every variant again (--rerun: its masks changed,
                     so every variant must use the same images): SIFT, LightGlue, LightGlue on every octave,
                     SIFT + LightGlue, SIFT without pose-guided matching, SIFT without the isolated-point filter
                     (0.94.4); no tiny joint
 3. ``ab_seitah``  - the same on Seitah Overlook tiny; variants already finished are kept
 4. ``pipeline``   - studies/calibration/run_pipeline.py --reset (only the first time; after a restart it goes on
                     where it stopped) with the block settings of --set (default MATCH_MODE=exhaustive: the
                     calibration needs SIFT's Navcam - Mastcam-Z ties): the ecam blocks, the Navcam joint and the
                     Hazcam fit (promoted), the Mastcam-Z blocks, the Mastcam-Z joints (promoted), the error
                     analysis, REPORT.md
 5. ``summary``    - WEEKEND.md: what each step did, with the A/B tables and the pipeline report

Windows: the machine is kept from sleeping while this runs (SetThreadExecutionState); a reboot (Windows Update)
stops it - start run_weekend.bat again and it goes on.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
PY = sys.executable

STEPS = ("wait", "ab_chal", "ab_seitah", "pipeline", "summary")
AB_VARIANTS = ("sift", "lightglue", "lightglue_octaves", "sift_lightglue", "sift_pg_off", "sift_iso_off")
DEFAULT_SET = ("MATCH_MODE=exhaustive",)
STATUS_FILE = "weekend_status.json"                      # <root>/mppp_pipeline/


def keep_awake(stop: threading.Event) -> None:
    """Windows: no sleep (and no display-off sleep of the system) while the event is not set."""
    if not sys.platform.startswith("win"):
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x00000001, 0x00000040
    while not stop.is_set():
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        except Exception:                                   # noqa: BLE001
            return
        stop.wait(300)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:                                       # noqa: BLE001
        pass


class Weekend:
    def __init__(self, a):
        from mppp.runner import Log
        self.a = a
        self.root = Path(a.root)
        self.dir = self.root / "mppp_pipeline"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "weekend_state.json"
        self.log = Log(self.dir / ("weekend_dry_run_log.txt" if a.dry_run else "weekend_log.txt"))
        try:
            self.state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.state = {}

    def save(self) -> None:
        if self.a.dry_run:
            return
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.state_file)

    def mark(self, step: str, status: str, note: str = "") -> None:
        self.state[step] = {"status": status, "note": note, "when": datetime.datetime.now().isoformat(timespec="seconds")}
        self.save()
        self.log(f"{step}: {status}" + (f" - {note}" if note else ""))

    def run(self, step: str, cmd, timeout_h=None) -> int:
        """A command with its output in <dir>/weekend_<step>.log; the exit code (124: time limit, tree stopped)."""
        self.log(f"  {step}: " + " ".join(str(c) for c in cmd))
        if self.a.dry_run:
            return 0
        lf = self.dir / f"weekend_{step}.log"
        with open(lf, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(map(str, cmd))}\n")
            fh.flush()
            p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                 env=dict(os.environ, PYTHONUNBUFFERED="1", MPPP_SCAPES=str(self.root)))
            try:
                rc = p.wait(timeout=timeout_h * 3600 if timeout_h else None)
            except subprocess.TimeoutExpired:
                from _batch import kill_tree
                kill_tree(p.pid)
                p.wait(timeout=120)
                fh.write(f"\nTIME LIMIT {timeout_h} h: stopped\n")
                rc = 124
        self.log(f"  {step}: exit {rc} (log {lf})")
        return rc

    # ------------------------------------------------------------------ steps
    def busy(self):
        """What else runs on ROOT: a batch, or an alignment with a live status file (None: nothing)."""
        from _batch import BATCH_FILE
        from mppp.runner import STATUS_FILE, read_status
        cur = read_status(self.root / BATCH_FILE)
        if cur and cur.get("alive"):
            return f"batch {cur.get('kind')} (site {cur.get('site')}, pid {cur.get('pid')})"
        for st in self.root.glob(f"*/{STATUS_FILE}"):
            s = read_status(st)
            if s and s.get("alive") and int(s.get("pid") or 0) != os.getpid():
                return f"{st.parent.name} ({s.get('variant') or 'colmap'}, pid {s.get('pid')})"
        return None

    def s_wait(self):
        t0 = time.time()
        while True:
            b = self.busy()
            if not b:
                return "done", "nothing else running"
            if self.a.dry_run:
                return "done", f"dry run ({b} running)"
            if time.time() - t0 > self.a.wait_h * 3600:
                return "failed", f"still running after {self.a.wait_h} h: {b}"
            self.log(f"  wait: {b}; again in 5 min")
            time.sleep(300)

    def _ab(self, step, site, rerun):
        cmd = [PY, ROOT / "scripts" / "lightglue_ab.py", "--root", self.root, "--site", site, "--no-joint",
               "--variants", *self.a.ab_variants] + (["--rerun"] if rerun else [])
        rc = self.run(step, cmd, timeout_h=self.a.ab_timeout_h)
        md = next(iter(self.root.glob(f"*_{site}_zcam/lightglue_ab.md")), None)
        return ("done" if rc == 0 else "failed"), f"exit {rc}" + (f"; {md}" if md else "")

    def s_ab_chal(self):
        first = not self.state.get("ab_chal_started")         # after a restart: the variants done since stay
        self.state["ab_chal_started"] = True
        self.save()
        return self._ab("ab_chal", "chal_rocks_tiny", rerun=first)

    def s_ab_seitah(self):
        return self._ab("ab_seitah", "seitah_overlook_tiny", rerun=False)

    def s_pipeline(self):
        first = not self.state.get("pipeline_started")
        cmd = [PY, HERE / "run_pipeline.py", "--root", self.root] + (["--reset"] if first else [])
        for x in self.a.set:
            cmd += ["--set", x]
        cmd += ["--jobs", str(self.a.jobs)]
        self.state["pipeline_started"] = True
        self.save()
        rc = self.run("pipeline", cmd, timeout_h=self.a.pipeline_timeout_h)
        return ("done" if rc == 0 else "failed"), f"exit {rc}; {self.dir / 'REPORT.md'}"

    def s_summary(self):
        lines = [f"# Weekend run ({datetime.datetime.now():%Y-%m-%d %H:%M})", "", f"Root: {self.root}", "",
                 "| step | status | when | note |", "|---|---|---|---|"]
        for s in STEPS[:-1]:
            e = self.state.get(s) or {}
            lines.append(f"| {s} | {e.get('status', '-')} | {e.get('when', '-')} | {e.get('note', '')} |")
        for site in ("chal_rocks_tiny", "seitah_overlook_tiny"):
            md = next(iter(self.root.glob(f"*_{site}_zcam/lightglue_ab.md")), None)
            if md and md.is_file():
                txt = md.read_text(encoding="utf-8").split("## Every health check")[0]
                lines += ["", f"## {site}", "", txt.split("\n", 1)[1] if "\n" in txt else txt]
        rep = self.dir / "REPORT.md"
        if rep.is_file():
            lines += ["", "## Pipeline", "", rep.read_text(encoding="utf-8")]
        if not self.a.dry_run:
            (self.dir / "WEEKEND.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "done", str(self.dir / "WEEKEND.md")

    def go(self) -> int:
        self.log(f"run_weekend: root {self.root}, steps {', '.join(STEPS)}, block settings {list(self.a.set)}, "
                 f"jobs {self.a.jobs}, A/B variants {list(self.a.ab_variants)}")
        for s in STEPS:
            e = self.state.get(s) or {}
            if e.get("status") in ("done", "failed") and s not in (self.a.redo or []):
                self.log(f"{s}: {e['status']} before ({e.get('when')}), not run again")
                continue
            if s == "pipeline" and e.get("status") == "running":
                self.log("pipeline: started before - it goes on where it stopped")
            self.mark(s, "running")
            try:
                status, note = getattr(self, f"s_{s}")()
            except Exception as ex:                         # noqa: BLE001  (the next step still runs)
                status, note = "failed", f"{type(ex).__name__}: {ex}"
            self.mark(s, status, note)
        self.state["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.save()
        self.log("run_weekend: finished")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES") or "D:\\scapes")
    ap.add_argument("--set", action="append", default=None, metavar="NAME=VALUE",
                    help=f"block settings for the pipeline (default {' '.join(DEFAULT_SET)})")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("MPPP_JOBS", "2") or 2))
    ap.add_argument("--ab-variants", nargs="+", default=list(AB_VARIANTS))
    ap.add_argument("--wait-h", type=float, default=24.0)
    ap.add_argument("--ab-timeout-h", type=float, default=12.0, help="time limit of one tiny block's A/B")
    ap.add_argument("--pipeline-timeout-h", type=float, default=150.0)
    ap.add_argument("--redo", nargs="*", choices=STEPS, default=None, help="run these steps again")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="forget the weekend state (the pipeline starts afresh too)")
    a = ap.parse_args(argv)
    a.set = list(a.set) if a.set else list(DEFAULT_SET)
    w = Weekend(a)
    if a.status:
        print(json.dumps(w.state, indent=1))
        return 0
    if a.reset and w.state_file.exists() and not a.dry_run:
        w.state_file.replace(w.dir / f"weekend_state_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
        w.state = {}
    if a.dry_run:
        w.state = {}
    try:
        from mppp.runner import disable_quickedit
        disable_quickedit()                                 # a click in the console window must not pause the run
    except Exception:                                       # noqa: BLE001
        pass
    stop = threading.Event()
    threading.Thread(target=keep_awake, args=(stop,), daemon=True).start()
    from mppp.runner import RunStatus                     # stop_mppp.bat marks it stopped: run_weekend.bat then ends
    status = None if a.dry_run else RunStatus(w.dir / STATUS_FILE, kind="weekend", argv=sys.argv).start()
    try:
        rc = w.go()
        if status:
            status.stop("finished")
        return rc
    except BaseException as e:
        if status:
            status.stop("failed", last_error=f"{type(e).__name__}: {e}")
        raise
    finally:
        stop.set()


if __name__ == "__main__":
    sys.exit(main())
