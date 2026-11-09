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
7. the studies that exist (15:46: "use past studies instead of writing new ones"), on the freeze run's blocks:
   ``figures``        ``make_figures.bat`` - the standard figure set from the newest joints and the blocks;
   ``maps``           ``make_maps.bat`` - the site maps on the DTM, the orthomosaic and white;
   ``aspect_geometry`` ``zcam_aspect_geometry.py`` - fy / fx of every Mastcam-Z bin against elevation, range and
                      frame coverage (gravity sag or geometry), every zoom;
   ``s6``             ``tie_point_studies.py S6`` - the per-bin focal residuals about the focus lines against the
                      observations, over the zcam_consensus blocks;
   then the tie-point studies on the three tiny blocks in the old root (every finished variant is skipped):
   ``s2``             point thinning (``--s2-root``, default the old root; 14:57: "run that at the end of the freeze
                      run after everything else important has completed") and its verdict - would a cap of 1500 do:
                      frame centres < 1 cm p95, attitudes < 0.1 mdeg p95, per-bin focal lengths < 0.05 % max;
   ``s1`` ``s3`` ``s4`` ``s5``  the feature budget, the matcher and verification, pair selection and the pose-guided
                      step, the filters (``--skip-tie-studies`` leaves these four out; ``--skip-s2`` S2).

Faults (15:30-16:02, "make sure it is robust against faults"; 0.95.4 with an independent review's fixes):

- every step that fails - a non-zero exit, an exception in the chain's own code for it - is logged and the chain goes
  on (the pipeline itself goes on past a failed stage); a step's child is stopped with its whole process tree when it
  runs past its time limit; the free disk space is logged before every step (a warning under FREE_GB_WARN);
- the chain's own state is ``<root>\\mppp_pipeline\\freeze_chain_status.json`` (written atomically, a heartbeat
  every few minutes, tagged with CHAIN_TAG - the status of an older chain is kept aside as
  ``freeze_chain_status_before_<tag>.json`` and not used: the 14:17 chain had marked "s2" done; another version of the chain's steps gets another tag).  Started again after
  a crash, a reboot or a closed window, it skips the steps marked done, keeps the mask choice (another choice would
  change the run key and align every block again) and continues the pipeline without ``--reset`` (the pipeline
  continues after its last finished stage; the batches skip the blocks finished with the same run key; a pipeline
  left running by a closed window is waited for first);
- a second chain on the same root does not start while one is alive (the heartbeat and its process);
- while it runs the chain asks Windows not to sleep (SetThreadExecutionState) and switches the console's QuickEdit
  mode off (a click in the window would freeze a print);
- a reboot (Windows Update, a power cut with the computer set to start again): the chain registers a scheduled task
  "MPPP freeze chain" that starts ``freeze_chain.bat`` again at the next logon of this user (``schtasks /SC
  ONLOGON``; ``--no-logon-task`` leaves it out) and deletes it when the chain ends on its own (done, Ctrl-C or a
  failure of the chain's own code - a loop of restarts must not hide a bug).  The computer must log the user in
  again for that (automatic logon, or someone at the keyboard); until then the chain simply waits.
Each step's output: ``<root>\\mppp_pipeline\\freeze_chain_<step>.log``; the chain's log: ``freeze_chain_log.txt``.

    scripts\\windows\\freeze_chain.bat                   everything (again after a stop: continues)
    scripts\\windows\\freeze_chain.bat --no-wait         no wait for the v4.2 training (the A/B with what there is)
    scripts\\windows\\freeze_chain.bat --no-final-pass   the blocks as the joints found them
    scripts\\windows\\freeze_chain.bat --no-freeze       no pipeline (the A/B and S2 only)
    scripts\\windows\\freeze_chain.bat --skip-tie-studies  no S1 / S3 / S4 / S5 at the end
    scripts\\windows\\freeze_chain.bat --redo mask_ab    this step again even if done
    scripts\\windows\\freeze_chain.bat --status          the state of the chain and of the pipeline, nothing run
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

HERE = Path(__file__).resolve().parent          # mppp_studies/calibration
REPO = HERE.parents[2]                          # the repository root
WIN = REPO / "scripts" / "windows"

CKPT_DIR = REPO / "checkpoints"
OUTPUTS = REPO / "Claude outputs"
V4 = CKPT_DIR / "convnext_tiny_s4_seg_20261030_v8.pt"
V41 = CKPT_DIR / "convnext_tiny_s4_seg_20261102_v8_v4.1.pt"
V42_TAG = "v8_v4.2"
MASK_DATA, MASK_SEEN = "D:\\masks\\masks_training_set_v8", "D:\\masks\\masks_training_set_v7"
ROOT_FILES = ("zcam_low_backlash_found.csv", "zcam_focus_from_below.csv")
S2_SITE = "rockytop_example"
S2_LIMITS = {"centre_cm_p95": 1.0, "attitude_mdeg_p95": 0.1, "focal_pct_max": 0.05}
STEPS = ("root", "wait_v42", "mask_ab", "blocks", "final_pass", "studies", "figures", "maps", "aspect_geometry", "s6",
         "s2", "s1", "s3", "s4", "s5")
TIE_STUDIES = ("s1", "s3", "s4", "s5")
ZCAM_GROUP = "zcam_consensus"
FREE_GB_WARN = 300.0
STALE_S = 900                                                          # the chain's heartbeat
CHAIN_TAG = "freeze_0954"                     # the step set of the status file; another tag: kept aside, not used
LOGON_TASK = "MPPP freeze chain"              # the RunOnce value that starts the chain again after a reboot
# 3 Nov (audit): "schtasks /SC ONLOGON" needs elevation ("Access is denied" on the PC); the user's RunOnce key does not
RUNONCE_KEY = "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce"
STOP_MARKER = "mppp_stopped.txt"              # written in the root by stop_runs (stop_mppp.bat)
MIN_FREE_GB = 25.0                            # 3 Nov (audit): a step waits for disk space under this
EXTRA_STATUS_KEYS = {"chain", "mask_choice", "s2_verdict"}
WINDOWS = os.name == "nt"


class ChainStopped(BaseException):
    """stop_mppp.bat stopped a step (its marker in the root): the chain ends instead of starting the next step.  A
    BaseException, so ``guarded`` does not take it for a failure of the step's code."""


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Chain:
    def __init__(self, a, read_only: bool = False):
        self.a = a
        self.root = Path(a.root)
        d = self.root / "mppp_pipeline"
        if not read_only:
            d.mkdir(parents=True, exist_ok=True)
        self.dir = d
        self.logf = d / "freeze_chain_log.txt"
        self.statf = d / "freeze_chain_status.json"
        self.status = {"started": now(), "tag": CHAIN_TAG, "steps": {}}
        self.resumed = False
        self.before: dict = {}
        self.set_aside: Optional[Path] = None
        self.continued_from: Optional[str] = None
        if self.statf.is_file():                                       # a chain before: continue it
            try:
                old = json.loads(self.statf.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = {}
            steps = old.get("steps") if isinstance(old.get("steps"), dict) else None
            same = steps is not None and old.get("tag") == CHAIN_TAG
            # 3 Nov (audit): another version of this chain whose steps are all known is continued (a tag bump must not
            # send the blocks back to --reset); only a chain with other steps (the 14:17 one) is set aside
            compatible = steps is not None and bool(old.get("tag")) and set(steps) <= set(STEPS) | EXTRA_STATUS_KEYS
            if same or compatible:
                self.status = old
                self.before = dict(old)                           # as it was (alive_before looks at this)
                self.status.setdefault("restarts", []).append(now())
                self.resumed = True
                if not same:
                    self.continued_from = old.get("tag")
                    self.status["tag"] = CHAIN_TAG
            elif old and not read_only:                           # another chain's steps (the 14:17 one): not ours
                aside = d / f"freeze_chain_status_before_{CHAIN_TAG}.json"
                try:
                    if aside.is_file():
                        aside = d / f"freeze_chain_status_before_{CHAIN_TAG}_{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
                    self.statf.replace(aside)
                    self.set_aside = aside
                except OSError:
                    pass
        self.redo = set(getattr(a, "redo", None) or [])
        if not read_only:
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
    def run(self, name: str, cmd: Sequence, timeout_h: float, env: Optional[dict] = None) -> int:
        cmd = [str(c) for c in cmd]
        wait_for_space(self, name)
        self.log(f"{name}: {' '.join(cmd)}")
        self.step(name, state="running", command=cmd, started=now())
        lf = self.dir / f"freeze_chain_{name}.log"
        t = time.time()
        rc = 127
        p = None
        try:
            with open(lf, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== {now()}  {' '.join(cmd)}\n")
                fh.flush()
                env = dict(os.environ, PYTHONUNBUFFERED="1", MPPP_NOPAUSE="1", **(env or {}))
                p = subprocess.Popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=fh,
                                     stderr=subprocess.STDOUT, env=env)       # no stdin: a pause or input() ends
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
                                from mppp.batch import kill_tree
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
            if p is not None and p.poll() is None:            # 3 Nov (audit): no orphan beside the next step
                self.log(f"{name}: stopping its child (pid {p.pid}) with its process tree")
                try:
                    from mppp.batch import kill_tree
                    kill_tree(p.pid)
                except Exception:                                                     # noqa: BLE001
                    p.kill()
        hours = round((time.time() - t) / 3600, 2)
        self.log(f"{name}: exit {rc} after {hours:.2f} h (log {lf})")
        self.step(name, state="done" if rc == 0 else f"exit {rc}", hours=hours, ended=now())
        if rc != 0:
            mk = self.root / STOP_MARKER
            try:
                stopped = mk.is_file() and mk.stat().st_mtime >= t - 5
            except OSError:
                stopped = False
            if stopped:                                       # 3 Nov (audit): stop_mppp.bat stopped this step
                try:
                    why = mk.read_text(encoding="utf-8", errors="replace").strip()[:200]
                except OSError:
                    why = ""
                self.log(f"{name}: stopped by stop_mppp.bat ({mk.name}: {why}) - the chain ends here; "
                         "double-click freeze_chain.bat to continue")
                raise ChainStopped(name)
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


def free_gb(c: Chain, when: str = "") -> Optional[float]:
    """The free space on the root's drive, logged (a warning under FREE_GB_WARN); None when unknown."""
    import shutil
    try:
        free = shutil.disk_usage(c.root).free / 1e9
    except OSError as e:
        c.log(f"{when or 'disk'}: free space unknown ({e})")
        return None
    c.log(f"{when or 'disk'}: {free:.0f} GB free on {c.root.drive or c.root}" +
          (f" - WARNING: under {FREE_GB_WARN:.0f} GB; the blocks with the full export may not fit" if free < FREE_GB_WARN else ""))
    return free


def wait_for_space(c: Chain, name: str) -> None:
    """3 Nov (audit): under MIN_FREE_GB a step waits (logged hourly, heartbeat kept, at most 24 h) instead of starting
    into a full disk."""
    import shutil
    free = free_gb(c, name)
    if free is None or free >= MIN_FREE_GB:
        return
    t0 = last = time.time()
    c.log(f"{name}: {free:.0f} GB free (< {MIN_FREE_GB:g} GB) - waiting for space")
    while True:
        c.beat()
        time.sleep(300)
        try:
            free = shutil.disk_usage(c.root).free / 1e9
        except OSError:
            return
        if free >= MIN_FREE_GB:
            c.log(f"{name}: {free:.0f} GB free - going on")
            return
        if time.time() - t0 > 24 * 3600:
            c.log(f"{name}: still {free:.0f} GB free after 24 h - starting anyway")
            return
        if time.time() - last >= 3600:
            last = time.time()
            c.log(f"{name}: {free:.0f} GB free - still waiting for space")


def bat_path() -> Path:
    return WIN / "freeze_chain.bat"


def logon_task_register(c: Chain) -> None:
    """A RunOnce entry of this user that starts freeze_chain.bat again at the next logon (after a reboot the chain did
    not see coming; 3 Nov: a scheduled task needs elevation).  Windows removes the entry when it fires; the chain sets
    it again at every start and removes it when it ends on its own.  Best effort: Windows only, logged either way.  It
    runs a one-line launcher written next to the status file (the root and MPPP_NOPAUSE set, then the bat)."""
    if not WINDOWS or getattr(c.a, "no_logon_task", False):
        return
    bat = bat_path()
    if not bat.is_file():
        c.log(f"logon task: no {bat} - not registered")
        return
    launcher = c.dir / "freeze_chain_logon.bat"
    try:
        launcher.write_text("@echo off\r\nset \"MPPP_NOPAUSE=1\"\r\n"
                            f"set \"MPPP_FREEZE_ROOT={c.root}\"\r\ncall \"{bat}\"\r\n", encoding="ascii", newline="")
        r = subprocess.run(["reg", "add", RUNONCE_KEY, "/v", LOGON_TASK, "/t", "REG_SZ", "/d", f'"{launcher}"', "/f"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            c.log(f"logon task: RunOnce '{LOGON_TASK}' set ({launcher.name}) - after a reboot the chain continues at "
                  "the next logon")
            c.status["logon_task"] = True
        else:
            c.log(f"logon task: not set (reg exit {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}) - "
                  "after a reboot, double-click freeze_chain.bat again")
    except Exception as e:                                                            # noqa: BLE001
        c.log(f"logon task: not registered ({type(e).__name__}: {e})")


def logon_task_delete(c: Chain) -> None:
    if not WINDOWS or not c.status.get("logon_task"):
        return
    try:
        r = subprocess.run(["reg", "delete", RUNONCE_KEY, "/v", LOGON_TASK, "/f"], capture_output=True, text=True,
                           timeout=60)
        c.log(f"logon task: '{LOGON_TASK}' " + ("removed" if r.returncode == 0 else f"not removed (exit {r.returncode})"))
        if r.returncode == 0:
            c.status["logon_task"] = False
    except Exception as e:                                                            # noqa: BLE001
        c.log(f"logon task: not removed ({type(e).__name__}: {e})")


def guarded(c: Chain, name: str, fn, *args, **kw):
    """A step's own code (not its child) that raises: logged as the step's failure, the chain goes on."""
    try:
        return fn(*args, **kw)
    except KeyboardInterrupt:
        raise
    except Exception:                                                                 # noqa: BLE001
        c.log(f"{name}: the chain's code for it failed - going on:\n" + traceback.format_exc())
        st = c.status["steps"].get(name) or {}
        if st.get("state") != "done":
            c.step(name, state="exit chain-error")
        return None


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
    free = free_gb(c, "root")
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
    cmd = [sys.executable, "-m", "mppp_studies.calibration.run_pipeline", "--root", c.root, *extra]
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


def zcam_sites() -> list:
    try:
        from mppp.waypoints.sites import load_site_table
        return list(load_site_table().get("groups", {}).get(ZCAM_GROUP) or [])
    except Exception:                                                                 # noqa: BLE001
        return []


def after_studies(c: Chain) -> None:
    """The studies that exist, on the freeze run's blocks (the .bat ones on Windows only)."""
    env = {"MPPP_SCAPES": str(c.root)}
    bats = {"figures": (WIN / "make_figures.bat", 2.0), "maps": (WIN / "make_maps.bat", 3.0)}
    for name, (bat, hours) in bats.items():
        if c.done(name):
            continue
        if os.name != "nt":
            c.log(f"{name}: {bat.name} is a Windows script - skipped here")
            c.step(name, state="done", note="not Windows")
            continue
        guarded(c, name, c.run, name, ["cmd", "/c", bat], timeout_h=hours, env=env)
    if not c.done("aspect_geometry"):
        guarded(c, "aspect_geometry", c.run, "aspect_geometry",
                [sys.executable, "-m", "mppp_studies.calibration.zcam_aspect_geometry", "--root", c.root, "--group",
                 ZCAM_GROUP, "--zooms", "34", "48", "63", "79", "110"], timeout_h=3.0, env=env)
    if not c.done("s6"):
        sites = zcam_sites()
        if sites:
            guarded(c, "s6", c.run, "s6", [sys.executable, "-m", "mppp_studies.calibration.tie_point_studies", "S6", "--root", c.root,
                                           "--site", *sites], timeout_h=3.0, env=env)
        else:
            c.log(f"s6: no {ZCAM_GROUP} sites found - skipped")
            c.step("s6", state="done", note="no sites")


def s2_verdict(c: Chain, root: Optional[Path] = None) -> bool:
    from mppp.waypoints.sites import load_site_table, work_folder
    from .tie_point_studies import compare_solutions
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
    c.log(f"freeze chain {'continued' if c.resumed else 'started'} (root {c.root}, old root {a.old_root}, pid {os.getpid()}, "
          f"python {sys.executable})")
    if c.continued_from:
        c.log(f"the status of chain {c.continued_from} continued as {CHAIN_TAG} (its steps are this chain's)")
    if c.set_aside is not None:
        c.log(f"the status file of another chain kept aside as {c.set_aside.name} - its steps are not taken as done")
    if c.resumed:
        done = [s for s in STEPS if c.done(s)]
        c.log(f"steps done before: {', '.join(done) if done else 'none'}" + (f"; redo: {', '.join(sorted(c.redo))}" if c.redo else ""))
    keep_awake(c)
    logon_task_register(c)
    guarded(c, "root", prepare_root, c)                                # cheap: every time
    if not a.no_wait and not c.done("wait_v42"):
        guarded(c, "wait_v42", wait_for_v42, c)
    else:
        c.step("wait_v42", state="done", note="--no-wait" if a.no_wait else "before")
    mask = guarded(c, "mask_ab", choose_mask, c)
    if mask is not None and not mask.is_file():
        c.log(f"mask: {mask} is gone - v3 instead (the blocks' run key changes)")
        mask = None
    if not a.no_freeze:
        if not c.done("blocks"):
            guarded(c, "blocks", run_pipeline_step, c, "blocks", mask, "--stop-before", "error_analysis", reset=True)
        if a.no_final_pass:
            c.step("final_pass", state="done", note="--no-final-pass")
        elif not c.done("final_pass"):
            guarded(c, "final_pass", run_pipeline_step, c, "final_pass", mask, "--redo", "ecam_runs", "zcam_runs",
                    "--stop-before", "error_analysis")
        if not c.done("studies"):
            guarded(c, "studies", run_pipeline_step, c, "studies", mask)
    else:
        c.log("--no-freeze: no pipeline")
    guarded(c, "figures", after_studies, c)
    s2_root = Path(a.s2_root or a.old_root)
    env = {"MPPP_SCAPES": str(s2_root)}
    if not a.skip_s2 and not c.done("s2"):
        guarded(c, "s2", run_s2, c)
    if not a.skip_tie_studies:
        for st in TIE_STUDIES:
            if not c.done(st):
                guarded(c, st, c.run, st, [sys.executable, "-m", "mppp_studies.calibration.tie_point_studies", st.upper(), "--root", s2_root],
                        timeout_h=36.0, env=env)
    failed = [s for s in STEPS if str((c.status["steps"].get(s) or {}).get("state", "")).startswith("exit")]
    c.log("freeze chain done" + (f" - steps that did not exit 0: {', '.join(failed)}" if failed else ""))
    logon_task_delete(c)
    c.end("done", failed=failed)
    return 1 if failed else 0


def choose_mask(c: Chain) -> Optional[Path]:
    """The mask model of the blocks: the choice kept from a chain before, else the A/B of v3 against v4.2."""
    a = c.a
    mask = kept_mask(c)
    if mask is not None or (c.status["steps"].get("mask_choice") or {}).get("state") == "done":
        return mask
    ck = v42_trained() if not a.skip_ab else None
    if a.skip_ab:
        c.log("mask_ab: --skip-ab - v3 stays")
        c.step("mask_choice", state="done", choice="v3", reason="--skip-ab")
        return None
    if ck is None:
        c.log("mask_ab: no trained v4.2 checkpoint - v3 stays")
        c.step("mask_choice", state="done", choice="v3", reason="no v4.2")
        return None
    ab = OUTPUTS / "mask_ab_v3_v41_v42"
    if not c.done("mask_ab") or not (ab / "mask_ab.csv").is_file():
        c.run("mask_ab", [a.ab_python, "-m", "mppp_studies.mask_training.compare_mask_models", "--data", MASK_DATA,
                          "--split-ckpt", ck, "--also-held-out", V41, "--models", "mppp_mask_v3", V41, ck,
                          "--seen", MASK_SEEN, "--seen-split", "mppp_mask_v3", MASK_SEEN, "--out", ab],
              timeout_h=3.0)
    return mask_choice(c, ab, ck)


def run_s2(c: Chain) -> None:
    s2_root = Path(c.a.s2_root or c.a.old_root)
    c.run("s2", [sys.executable, "-m", "mppp_studies.calibration.tie_point_studies", "S2", "--site", S2_SITE, "--variants", "all",
                 "cap1500", "--root", s2_root], timeout_h=10.0, env={"MPPP_SCAPES": str(s2_root)})
    try:
        s2_verdict(c, s2_root)
    except Exception as e:                                                            # noqa: BLE001
        c.log(f"S2 verdict: failed ({type(e).__name__}: {e})")


def show_status(c: Chain) -> int:
    """``--status``: the chain's steps and the pipeline's stages, nothing run (the status file is not touched)."""
    st = c.status if c.resumed else {}
    print(f"freeze chain on {c.root}: {st.get('state', 'not started')}" +
          (f" (heartbeat {st.get('heartbeat')}, pid {st.get('pid')})" if st else ""))
    for s in STEPS + ("mask_choice", "s2_verdict"):
        e = (st.get("steps") or {}).get(s) or {}
        extra = {k: v for k, v in e.items() if k in ("note", "choice", "verdict", "hours", "started", "ended")}
        print(f"  {s:14s} {str(e.get('state', '-')):12s} {json.dumps(extra) if extra else ''}")
    try:
        from mppp.runner import read_status
        cur = read_status(c.dir / "pipeline_status.json")
        print(f"pipeline: {(cur or {}).get('state', 'not started')}" + (f", stage {cur.get('stage')}" if cur else ""))
    except Exception as e:                                                            # noqa: BLE001
        print(f"pipeline: unknown ({e})")
    return 0


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
    ap.add_argument("--skip-tie-studies", action="store_true", help="no S1 / S3 / S4 / S5 at the end")
    ap.add_argument("--cap", action="store_true", help="the blocks with ba_points_cap=1500")
    ap.add_argument("--no-freeze", action="store_true", help="no pipeline")
    ap.add_argument("--redo", nargs="*", choices=STEPS, default=None, help="these steps again even if done")
    ap.add_argument("--no-logon-task", action="store_true", help="no scheduled task to continue after a reboot")
    ap.add_argument("--status", action="store_true", help="the state of the chain and the pipeline; nothing run")
    a = ap.parse_args(argv)
    if a.status:
        return show_status(Chain(a, read_only=True))
    c = Chain(a)
    pid = c.alive_before()
    if pid:
        c.log(f"not started: a freeze chain is already running on {c.root} (pid {pid}) - one window only")
        return 2
    try:
        return chain(c)
    except ChainStopped:
        logon_task_delete(c)
        c.end("stopped")
        return 130
    except KeyboardInterrupt:
        c.log("stopped (Ctrl-C); start freeze_chain.bat again to continue")
        logon_task_delete(c)
        c.end("stopped")
        return 130
    except Exception:                                                                 # noqa: BLE001
        c.log("the chain itself failed:\n" + traceback.format_exc() + "start freeze_chain.bat again to continue")
        logon_task_delete(c)
        c.end("failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
