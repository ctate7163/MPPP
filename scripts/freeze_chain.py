"""
The freeze run, unattended over several days (2 Nov 2026, Christian: "I will be away from the computer when these
finish"; 15:30: "make sure it is robust against faults. I want it to run to completion over several days, at the end
of which I want a standard set of blocks using the best camera and rig models available followed by the time
consuming tests and studies needed to really understand the results"):

1. ``root``        the blocks root (``--root``, default ``D:\\rover_blocks``): the found list and the focus-from-below
                   list copied from the old root (``--old-root``, default ``D:\\scapes``) when the new one has none;
                   ``dem\\`` and the free disk space checked;
2. ``wait_v42``    a wait for mask v4.2's training, started by hand (``train_masks.bat --tag v8_v4.2 --zoom-pairs``;
                   14:54: "don't add the mask training to the freeze, just delay the start so that it will be done by
                   the time it runs the mask comparison"): done when its release candidate or its summary is written in
                   ``Claude outputs\\mask_<checkpoint>\\``; given up when its log is quiet for ``--quiet-min`` minutes
                   or after ``--max-wait-h`` hours;
3. ``mask_ab``     the mask A/B of v3, v4.1 and v4.2 on the held-out frames (v4.2's split, v4.1 the same set) and, for
                   v3, the frames it did not train on (``--seen-split``) -> ``Claude outputs\\mask_ab_v3_v41_v42\\``;
                   v4.2 (its ``.pt``) is used for the blocks when its exclusion IoU on those frames beats v3's, else v3
                   (the default) stays - the choice is kept in the status file, so a restart keeps it (another choice
                   would change the run key and align every block again);
4. ``blocks``      ``run_pipeline.py --reset --stop-before error_analysis``: every Navcam + Hazcam block, the Navcam
                   joint and the Hazcam fit (promoted when their checks pass), every Mastcam-Z block with the models
                   just promoted, the Mastcam-Z joint of each zoom (promoted when its checks pass), the N+Z check;
5. ``final_pass``  ``run_pipeline.py --redo ecam_runs zcam_runs --stop-before error_analysis``: the blocks again with
                   the models the joints promoted - only those whose models changed (the camera-model fingerprint is
                   in the run key; features and matches are reused, so a block takes its mapping time only): the
                   standard set of blocks, every one aligned with the best models the run produced;
6. ``studies``     ``run_pipeline.py`` (no flags): error_analysis, rover_frame, lightglue_ab, zcam_states and the
                   report on those blocks;
7. ``paper_studies`` ``studies\\paper\\run_paper_studies.py`` (or ``<root>\\paper_studies\\run_paper_studies.py``) if
                   it exists when the chain gets there - it can be synced while the blocks run;
8. ``s2``          last, the point-thinning study S2 on rockytop_example at ``--s2-root`` (default the old root, where
                   its block is; 14:57: "run that at the end of the freeze run after everything else important has
                   completed") and its verdict (would a cap of 1500 do: frame centres < 1 cm p95, attitudes
                   < 0.1 mdeg p95, per-bin focal lengths < 0.05 % max from the all-points solution).

Faults.  Every step that fails is logged and the chain goes on (the pipeline itself goes on past a failed stage).  A
step's child is stopped with its whole process tree when it runs past its time limit.  The chain's own state is
``<root>\\mppp_pipeline\\freeze_chain_status.json``: started again after a crash, a reboot or a closed window
(double-click ``freeze_chain.bat`` again), it skips the steps marked done, keeps the mask choice, and continues the
pipeline without ``--reset`` (the pipeline continues after its last finished stage, and the batches skip the blocks
finished with the same run key).  A second chain on the same root does not start while one is alive
(``freeze_chain_status.json`` heartbeat).  While it runs, the chain asks Windows not to sleep (SetThreadExecutionState)
and switches the console's QuickEdit mode off (a click in the window would freeze a print).  A Windows Update restart
still ends it: start it again afterwards.  Each step's output: ``<root>\\mppp_pipeline\\freeze_chain_<step>.log``;
the chain's log: ``freeze_chain_log.txt``.

    scripts\\windows\\freeze_chain.bat                   everything (again after a stop: continues)
    scripts\\windows\\freeze_chain.bat --no-wait         no wait for the v4.2 training (the A/B with what there is)
    scripts\\windows\\freeze_chain.bat --no-final-pass   the blocks as the joints found them
    scripts\\windows\\freeze_chain.bat --no-freeze       no pipeline (the A/B and S2 only)
    scripts\\windows\\freeze_chain.bat --redo mask_ab    this step again even if done
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

CKPT_DIR = REPO / "checkpoints"
OUTPUTS = REPO / "Claude outputs"
V4 = CKPT_DIR / "convnext_tiny_s4_seg_20261030_v8.pt"
V41 = CKPT_DIR / "convnext_tiny_s4_seg_20261102_v8_v4.1.pt"
V42_TAG = "v8_v4.2"
MASK_DATA, MASK_SEEN = "D:\\masks\\masks_training_set_v8", "D:\\masks\\masks_training_set_v7"
ROOT_FILES = ("zcam_low_backlash_found.csv", "zcam_focus_from_below.csv")
S2_SITE = "rockytop_example"
S2_LIMITS = {"centre_cm_p95": 1.0, "attitude_mdeg_p95": 0.1, "focal_pct_max": 0.05}
STEPS = ("root", "wait_v42", "mask_ab", "blocks", "final_pass", "studies", "paper_studies", "s2")
FREE_GB_WARN = 300.0
STALE_S = 900                                                          # the chain's heartbeat


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Chain:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root)
        d = self.root / "mppp_pipeline"
        d.mkdir(parents=True, exist_ok=True)
        self.dir = d
        self.logf = d / "freeze_chain_log.txt"
        self.statf = d / "freeze_chain_status.json"
        self.status = {"started": now(), "steps": {}}
        self.resumed = False
        self.before: dict = {}
        if self.statf.is_file():                                       # a chain before: continue it
            try:
                old = json.loads(self.statf.read_text(encoding="utf-8"))
                if isinstance(old.get("steps"), dict):
                    self.status = old
                    self.before = dict(old)                       # as it was (alive_before looks at this)
                    self.status.setdefault("restarts", []).append(now())
                    self.resumed = True
            except (OSError, ValueError):
                pass
        self.redo = set(getattr(a, "redo", None) or [])
        self.status["state"] = "running"

    # ------------------------------------------------------------ bookkeeping
    def log(self, msg: str) -> None:
        line = f"{now()}  {msg}"
        print(line, flush=True)
        try:
            with open(self.logf, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def step(self, name: str, **kw) -> None:
        self.status["steps"].setdefault(name, {}).update(kw, updated=now())
        self.status["heartbeat"] = now()
        self.status.setdefault("state", "running")
        self.status["pid"] = os.getpid()
        tmp = self.statf.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.status, indent=1, default=str), encoding="utf-8")
            os.replace(tmp, self.statf)
        except OSError as e:
            self.log(f"status not written ({e})")

    def beat(self) -> None:
        self.step("chain")

    def end(self, state: str, **kw) -> None:
        """The chain's end state ("done", "failed", "stopped"); a start sets it back to running."""
        self.status["state"] = state
        self.step("chain", state=state, **kw)

    def done(self, name: str) -> bool:
        """This step finished before (a restart) and is not to be redone."""
        return name not in self.redo and (self.status["steps"].get(name) or {}).get("state") == "done"

    def alive_before(self) -> Optional[int]:
        """The pid of another chain alive on this root (its heartbeat within STALE_S and its process there)."""
        if not self.resumed:
            return None
        hb, pid = self.before.get("heartbeat"), self.before.get("pid")
        if self.before.get("state") != "running" or not hb or not pid or pid == os.getpid():
            return None
        try:
            age = (datetime.datetime.now() - datetime.datetime.fromisoformat(hb)).total_seconds()
        except ValueError:
            return None
        return int(pid) if age < STALE_S and pid_alive(int(pid)) else None

    # ------------------------------------------------------------ a step's command
    def run(self, name: str, cmd: Sequence, timeout_h: float) -> int:
        cmd = [str(c) for c in cmd]
        self.log(f"{name}: {' '.join(cmd)}")
        self.step(name, state="running", command=cmd, started=now())
        lf = self.dir / f"freeze_chain_{name}.log"
        t = time.time()
        rc = 127
        try:
            with open(lf, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {now()}  {' '.join(cmd)}\n")
                fh.flush()
                env = dict(os.environ, PYTHONUNBUFFERED="1", MPPP_NOPAUSE="1")
                p = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env)
                deadline = t + timeout_h * 3600
                while True:
                    try:
                        rc = p.wait(timeout=max(5.0, min(600.0, deadline - time.time())))
                        break
                    except subprocess.TimeoutExpired:
                        self.beat()
                        if time.time() > deadline:
                            self.log(f"{name}: {timeout_h} h gone - stopping it with its process tree")
                            try:
                                from _batch import kill_tree
                                kill_tree(p.pid)
                            except Exception as e:                                # noqa: BLE001
                                self.log(f"{name}: kill_tree failed ({e}); killing the child only")
                                p.kill()
                            try:
                                p.wait(timeout=120)
                            except subprocess.TimeoutExpired:
                                pass
                            fh.write(f"\nTIMEOUT after {timeout_h} h: stopped\n")
                            rc = 124
                            break
        except Exception as e:                                                        # noqa: BLE001
            self.log(f"{name}: could not run ({type(e).__name__}: {e})")
            rc = 127
        hours = round((time.time() - t) / 3600, 2)
        self.log(f"{name}: exit {rc} after {hours:.2f} h (log {lf})")
        self.step(name, state="done" if rc == 0 else f"exit {rc}", hours=hours, ended=now())
        return rc


# ------------------------------------------------------------------ the computer
def pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
            return str(pid) in r.stdout
        os.kill(pid, 0)
        return True
    except Exception:                                                                 # noqa: BLE001
        return False


def keep_awake(c: Chain) -> None:
    """Ask Windows not to sleep while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED; the display may go off)."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ok = ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        c.log("keep awake: " + ("the computer will not sleep while the chain runs" if ok else "SetThreadExecutionState failed"))
    except Exception as e:                                                            # noqa: BLE001
        c.log(f"keep awake: not set ({e})")
    try:
        from mppp.runner import disable_quickedit
        if disable_quickedit():
            c.log("console: QuickEdit mode off")
    except Exception as e:                                                            # noqa: BLE001
        c.log(f"console: QuickEdit not changed ({e})")


def prepare_root(c: Chain) -> None:
    """The root: the lists that live next to the blocks copied from the old root when missing; dem\\ and the disk."""
    import shutil
    old = Path(c.a.old_root)
    for name in ROOT_FILES:
        src, dst = old / name, c.root / name
        try:
            if dst.is_file():
                c.log(f"root: {dst} there")
            elif src.is_file():
                shutil.copy2(src, dst)
                c.log(f"root: {name} copied from {old}")
            else:
                c.log(f"root: no {name} in {old} or {c.root}")
        except OSError as e:
            c.log(f"root: {name} not copied ({e})")
    dem = c.root / "dem"
    c.log(f"root: {'DEMs in ' + str(dem) if dem.is_dir() else 'no dem folder in ' + str(c.root) + ' (maps without a DEM)'}")
    free = None
    try:
        free = shutil.disk_usage(c.root).free / 1e9
        c.log(f"root: {free:.0f} GB free on {c.root.drive or c.root}" +
              (f" - WARNING: under {FREE_GB_WARN:.0f} GB; the blocks with the full export may not fit" if free < FREE_GB_WARN else ""))
    except OSError as e:
        c.log(f"root: free space unknown ({e})")
    c.step("root", state="done", root=str(c.root), dem=dem.is_dir(), free_gb=None if free is None else round(free))


# ------------------------------------------------------------------ the mask model
def v42_release() -> Optional[Path]:
    """The newest mask v4.2 release candidate (``Claude outputs\\mask_<checkpoint>\\<checkpoint>.safetensors``)."""
    cands = sorted(OUTPUTS.glob(f"mask_*_{V42_TAG}/*_{V42_TAG}.safetensors"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def v42_summary() -> Optional[Path]:
    """The training's summary (written at the end of the training, before the export)."""
    cands = sorted(OUTPUTS.glob(f"mask_*_{V42_TAG}/summary.md"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def v42_checkpoint() -> Optional[Path]:
    cands = sorted(CKPT_DIR.glob(f"*_{V42_TAG}.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def v42_trained() -> Optional[Path]:
    """The checkpoint when the training is through (its release candidate or its summary written), else None."""
    ck = v42_checkpoint()
    if ck is None:
        return None
    sig = v42_release() or v42_summary()
    return ck if sig is not None and sig.stat().st_mtime >= ck.stat().st_mtime - 3600 else None


def wait_for_v42(c: Chain) -> None:
    """Wait for the mask v4.2 training started by hand."""
    t0 = time.time()
    c.step("wait_v42", state="waiting")
    last = 0.0
    while True:
        ck = v42_trained()
        if ck is not None:
            c.log(f"wait_v42: {ck.name} trained ({(v42_release() or v42_summary()).name} written)")
            c.step("wait_v42", state="done", checkpoint=str(ck))
            return
        logs = sorted(CKPT_DIR.glob(f"*_{V42_TAG}_debug/log.csv"), key=lambda p: p.stat().st_mtime)
        age = (time.time() - logs[-1].stat().st_mtime) / 60 if logs else None
        if age is not None and age > c.a.quiet_min and (time.time() - t0) > 600:
            c.log(f"wait_v42: the training log quiet for {age:.0f} min - going on without v4.2")
            c.step("wait_v42", state="done", note=f"log quiet {age:.0f} min")
            return
        if time.time() - t0 > c.a.max_wait_h * 3600:
            c.log(f"wait_v42: not done after {c.a.max_wait_h} h ({'no log yet' if not logs else 'log ' + str(logs[-1])}) - going on without v4.2")
            c.step("wait_v42", state="done", note="timeout")
            return
        if time.time() - last > 1800:
            last = time.time()
            c.log(f"wait_v42: waiting ({'no training log yet' if age is None else f'log {age:.0f} min old'}; "
                  f"{(time.time() - t0) / 3600:.1f} h so far)")
        c.beat()
        time.sleep(120)


def mask_choice(c: Chain, ab_dir: Path, v42: Path) -> Optional[Path]:
    """v4.2's checkpoint when its exclusion IoU on the frames no model trained on is higher than v3's."""
    import csv
    f = ab_dir / "mask_ab.csv"
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
    except OSError as e:
        c.log(f"mask choice: no {f} ({e}) - v3 stays")
        c.step("mask_choice", state="done", choice="v3", reason="no A/B")
        return None
    pick = {}
    for subset in ("not_seen", "all"):
        for r in rows:
            if r.get("subset") == subset and r.get("camera") == "all":
                try:
                    pick.setdefault(subset, {})[r["model"]] = float(r["exclusion_IoU"])
                except (KeyError, ValueError):
                    pass
    fair = pick.get("not_seen") or pick.get("all") or {}
    v3 = fair.get("mppp_mask_v3")
    v42_iou = next((v for k, v in fair.items() if k.endswith(V42_TAG)), None)
    zrec = {r["model"]: r.get("hardware_recall") for r in rows if r.get("subset") == "all" and r.get("camera") == "Mastcam-Z"}
    c.log(f"mask choice: exclusion IoU on the frames no model trained on {json.dumps(fair)}; Mastcam-Z recall on all "
          f"held-out frames {json.dumps(zrec)}")
    if v3 is not None and v42_iou is not None and v42_iou > v3:
        c.log(f"mask choice: v4.2 ({v42_iou:.4f} > v3 {v3:.4f}) - the blocks use {v42}")
        c.step("mask_choice", state="done", choice="v4.2", v3=v3, v42=v42_iou, model=str(v42))
        return v42
    c.log(f"mask choice: v3 stays ({v3} vs v4.2 {v42_iou})")
    c.step("mask_choice", state="done", choice="v3", v3=v3, v42=v42_iou)
    return None


def kept_mask(c: Chain) -> Optional[Path]:
    """The mask choice of a chain before (a restart must not change the run key)."""
    m = (c.status["steps"].get("mask_choice") or {})
    if m.get("state") == "done":
        c.log(f"mask choice: kept from before - {m.get('choice')}" + (f" ({m.get('model')})" if m.get("model") else ""))
        return Path(m["model"]) if m.get("model") else None
    return None


# ------------------------------------------------------------------ the pipeline
def wait_pipeline_idle(c: Chain) -> None:
    """No run_pipeline alive on the root (one from a chain whose window closed may still run): wait for it."""
    try:
        from mppp.runner import read_status
    except Exception as e:                                                            # noqa: BLE001
        c.log(f"pipeline status: mppp not importable ({e})")
        return
    lock = c.dir / "pipeline_status.json"
    t0 = time.time()
    while True:
        cur = read_status(lock)
        if not (cur and cur.get("alive")):
            return
        if time.time() - t0 > 24 * 7 * 3600:
            c.log(f"pipeline: still running after 7 days (pid {cur.get('pid')}) - going on")
            return
        if int(time.time() - t0) % 3600 < 600:
            c.log(f"pipeline: one is already running (pid {cur.get('pid')}, stage {cur.get('stage')}) - waiting for it")
        c.beat()
        time.sleep(600)


def pipeline_cmd(c: Chain, mask: Optional[Path], *extra: str) -> list:
    cmd = [sys.executable, REPO / "studies" / "calibration" / "run_pipeline.py", "--root", c.root, *extra]
    if c.a.cap:
        cmd += ["--set", "ba_points_cap=1500"]
    if mask is not None:
        cmd += ["--set", f"mask_checkpoint={mask}"]
    return cmd


def run_pipeline_step(c: Chain, name: str, mask: Optional[Path], *extra: str, reset: bool = False) -> int:
    wait_pipeline_idle(c)
    started = (c.status["steps"].get(name) or {}).get("started")
    flags = list(extra) + (["--reset"] if reset and not started else [])
    if reset and started:
        c.log(f"{name}: started before - the pipeline continues after its last finished stage (no --reset)")
    return c.run(name, pipeline_cmd(c, mask, *flags), timeout_h=24 * 7)


def paper_studies_script(c: Chain) -> Optional[Path]:
    for p in (REPO / "studies" / "paper" / "run_paper_studies.py", c.root / "paper_studies" / "run_paper_studies.py"):
        if p.is_file():
            return p
    return None


def s2_verdict(c: Chain, root: Optional[Path] = None) -> bool:
    from mppp.waypoints.sites import load_site_table, work_folder
    from tie_point_studies import compare_solutions
    t = load_site_table()
    work = Path(work_folder(str(root or c.root), S2_SITE, True, t["sites"][S2_SITE]["sols"]))
    f = work / "studies" / "S2.json"
    try:
        res = json.loads(f.read_text(encoding="utf-8"))
        v = res["variants"]
        cmp = compare_solutions(v["all"], v["cap1500"])
    except Exception as e:                                                            # noqa: BLE001
        c.log(f"S2 verdict: no comparison ({type(e).__name__}: {e})")
        c.step("s2_verdict", state="done", verdict="no comparison")
        return False
    got = {"centre_cm_p95": (cmp.get("centre_cm") or {}).get("p95"),
           "attitude_mdeg_p95": (cmp.get("attitude_mdeg") or {}).get("p95"),
           "focal_pct_max": (cmp.get("focal_pct") or {}).get("max")}
    ok = all(got[k] is not None and got[k] < lim for k, lim in S2_LIMITS.items())
    c.log(f"S2 verdict: {json.dumps(got)} against {json.dumps(S2_LIMITS)} -> "
          f"{'a cap of 1500 would do' if ok else 'a cap of 1500 changes the solution'}")
    c.step("s2_verdict", state="done", verdict="cap 1500" if ok else "no cap", comparison=got)
    return ok


# ------------------------------------------------------------------ main
def chain(c: Chain) -> int:
    a = c.a
    c.log(f"freeze chain {'continued' if c.resumed else 'started'} (root {c.root}, old root {a.old_root}, pid {os.getpid()})")
    if c.resumed:
        done = [s for s in STEPS if c.done(s)]
        c.log(f"steps done before: {', '.join(done) if done else 'none'}" + (f"; redo: {', '.join(sorted(c.redo))}" if c.redo else ""))
    keep_awake(c)
    prepare_root(c)                                                    # cheap: every time
    if not a.no_wait and not c.done("wait_v42"):
        wait_for_v42(c)
    else:
        c.step("wait_v42", state="done", note="--no-wait" if a.no_wait else "before")
    mask = kept_mask(c)
    if mask is None and (c.status["steps"].get("mask_choice") or {}).get("state") != "done":
        ck = v42_trained() if not a.skip_ab else None
        if a.skip_ab:
            c.log("mask_ab: --skip-ab - v3 stays")
            c.step("mask_choice", state="done", choice="v3", reason="--skip-ab")
        elif ck is None:
            c.log("mask_ab: no trained v4.2 checkpoint - v3 stays")
            c.step("mask_choice", state="done", choice="v3", reason="no v4.2")
        else:
            ab = OUTPUTS / "mask_ab_v3_v41_v42"
            if not c.done("mask_ab") or not (ab / "mask_ab.csv").is_file():
                c.run("mask_ab", [a.ab_python, HERE / "compare_mask_models.py", "--data", MASK_DATA,
                                  "--split-ckpt", ck, "--also-held-out", V41, "--models", "mppp_mask_v3", V41, ck,
                                  "--seen", MASK_SEEN, "--seen-split", "mppp_mask_v3", MASK_SEEN, "--out", ab],
                      timeout_h=3.0)
            mask = mask_choice(c, ab, ck)
    if mask is not None and not mask.is_file():
        c.log(f"mask: {mask} is gone - v3 instead (the blocks' run key changes)")
        mask = None
    if not a.no_freeze:
        if not c.done("blocks"):
            run_pipeline_step(c, "blocks", mask, "--stop-before", "error_analysis", reset=True)
        if a.no_final_pass:
            c.step("final_pass", state="done", note="--no-final-pass")
        elif not c.done("final_pass"):
            run_pipeline_step(c, "final_pass", mask, "--redo", "ecam_runs", "zcam_runs", "--stop-before", "error_analysis")
        if not c.done("studies"):
            run_pipeline_step(c, "studies", mask)
        if not c.done("paper_studies"):
            extra = paper_studies_script(c)
            if extra is not None:
                c.run("paper_studies", [sys.executable, extra, "--root", c.root], timeout_h=72)
            else:
                c.log("paper_studies: no studies\\paper\\run_paper_studies.py (nor one in the root) - nothing more")
                c.step("paper_studies", state="done", note="no script")
    else:
        c.log("--no-freeze: no pipeline")
    if not a.skip_s2 and not c.done("s2"):
        s2_root = Path(a.s2_root or a.old_root)
        c.run("s2", [sys.executable, HERE / "tie_point_studies.py", "S2", "--site", S2_SITE, "--variants", "all",
                     "cap1500", "--root", s2_root], timeout_h=10.0)
        try:
            s2_verdict(c, s2_root)
        except Exception as e:                                                        # noqa: BLE001
            c.log(f"S2 verdict: failed ({type(e).__name__}: {e})")
    failed = [s for s in STEPS if str((c.status["steps"].get(s) or {}).get("state", "")).startswith("exit")]
    c.log("freeze chain done" + (f" - steps that did not exit 0: {', '.join(failed)}" if failed else ""))
    c.end("done", failed=failed)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:\\rover_blocks", help="the root of the freeze run's blocks")
    ap.add_argument("--old-root", default="D:\\scapes", help="where the found lists and the S2 block are")
    ap.add_argument("--s2-root", default=None, help="the S2 study's root (default --old-root)")
    ap.add_argument("--ab-python", default=sys.executable, help="a Python with a CUDA torch (the mask A/B)")
    ap.add_argument("--no-wait", action="store_true", help="no wait for the v4.2 training")
    ap.add_argument("--max-wait-h", type=float, default=14.0)
    ap.add_argument("--quiet-min", type=float, default=45.0)
    ap.add_argument("--skip-ab", action="store_true", help="no mask A/B: v3")
    ap.add_argument("--no-final-pass", action="store_true", help="no second pass of the blocks with the promoted models")
    ap.add_argument("--skip-s2", action="store_true", help="no S2 study at the end")
    ap.add_argument("--cap", action="store_true", help="the blocks with ba_points_cap=1500")
    ap.add_argument("--no-freeze", action="store_true", help="no pipeline")
    ap.add_argument("--redo", nargs="*", choices=STEPS, default=None, help="these steps again even if done")
    a = ap.parse_args(argv)
    c = Chain(a)
    pid = c.alive_before()
    if pid:
        c.log(f"not started: a freeze chain is already running on {c.root} (pid {pid}) - one window only")
        return 2
    try:
        return chain(c)
    except KeyboardInterrupt:
        c.log("stopped (Ctrl-C); start freeze_chain.bat again to continue")
        c.end("stopped")
        return 130
    except Exception:                                                                 # noqa: BLE001
        c.log("the chain itself failed:\n" + traceback.format_exc() + "start freeze_chain.bat again to continue")
        c.end("failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
