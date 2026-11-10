"""
The freeze run on MPPP 0.96 (Christian, 9 Nov 2026: "please prepare a freeze.bat that will run on 0.96"), unattended
and resumable: the freeze-2 steps (8 Nov) with the robustness of the first chain and a gate on the mask model.

The run uses its own checkout (``--code``, ``D:\\code\\MPPP_freeze``, cloned from ``D:\\code\\MPPP`` by
``scripts\\windows\\freeze.bat``), its own camera-model folder (``--cmods``, ``D:\\rover_blocks\\cmods_freeze2``:
the models in use with the 3 Nov Navcam law and the freeze-2 rig promoted into it; ``MPPP_CMODS`` for every step),
and the blocks root ``--root`` (``D:\\rover_blocks``).  ``D:\\code\\MPPP`` and ``D:\\scapes\\cmods`` are not changed.

Steps (each done once; started again after any stop, the run continues where it was):

  wait     until no pipeline runs on the root and the first chain has finished its steps there (``--now``: only
           the pipeline lock)
  mask     the mask model of the blocks (``--mask-model``):
             keep        the model the blocks on disk were processed with (default; no image is processed again)
             auto        wait for the mask training under way (``--mask-tag``, default the newest ``checkpoints\\
                         *_debug`` of ``--mask-home``) and its comparison (``train_masks.bat --compare`` writes
                         ``Claude outputs\\mask_<checkpoint>\\mask_verdict.json``; without one the freeze runs
                         ``compare_mask_models`` itself with ``--ab-python``); the new model is used when its
                         exclusion IoU beats the blocks' model by ``--mask-margin`` (0.01), else the blocks' model
                         stays.  A new model means every image is processed again and every block aligned again
                         (the mask checkpoint is in the run key): hours before the first alignment
             <file|name> that checkpoint or registry name
           The choice is kept in the status file: a restart keeps it (another choice would change the run key)
  cmods    the run's model folder: seeded from ``--cmods-from`` when it does not exist; ``--candidate DIR`` promoted
           into it (``promote_cmods``) when the folder does not yet hold a freeze rig
  zjoints  run_pipeline: every Mastcam-Z block aligned again with the Navcam models of the folder (zcam_runs), then
           the Z34, Z48, Z63, Z79 and Z110 joints, promoted into the folder when their checks pass; for the zooms
           of ``--zcam-waive-aspect`` (63 79 110) the per-block aspect gate is waived
  zfinal   the Mastcam-Z blocks again with the models the joints promoted (only those whose models changed)
  ecam     the Navcam blocks with the folder's models, then the Navcam joint and the Hazcam fit as checks (nothing
           promoted: the Navcam is frozen at the 3 Nov law and the rig, the Hazcam at the 3 Nov fit)
  nz       the N+Z34 check on the final blocks
  deck     the rover deck in the adjustment of the Mastcam-Z blocks with a Z34 deck pan (Three Forks north, Airey
           Hill, Chal Rocks large): the Navcam and Z34 baselines against the survey
  ztests   comparison joints, never promoted: Z34 and Z110 with fy / fx held at 1, Z110 without Marble Mountain
  studies  run_pipeline's error_analysis, rover_frame and report on the final blocks (no states run)
  report   ``freeze_report.md`` in the root's ``mppp_pipeline`` folder

Faults: a step that fails is logged and the run goes on; a child past its time limit is stopped with its process
tree; the status (``mppp_pipeline\\freeze_status.json``) is written atomically with a heartbeat; a second run on the
same root does not start while one is alive; Windows is asked not to sleep; a RunOnce entry starts ``freeze.bat``
again at the next logon after a reboot; a stop by the user (Ctrl-C, ``stop_mppp.bat``) is logged as such and shown
by ``--status`` as "stopped by the user" - not as running, not as a failure.

    python -m mppp_studies.calibration.freeze                      start, or continue after a stop
    python -m mppp_studies.calibration.freeze --status             where it is (nothing started)
    python -m mppp_studies.calibration.freeze --mask-model auto    wait for the mask comparison first
    python -m mppp_studies.calibration.freeze --update-code        the checkout brought up to D:\\code\\MPPP main
                                                                   (git pull --ff-only); nothing started
    python -m mppp_studies.calibration.freeze --redo zjoints       a step again
    python -m mppp_studies.calibration.freeze --only deck          one step
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

from .freeze_chain import (ChainStopped, RUNONCE_KEY, STOP_MARKER, WINDOWS, free_gb, keep_awake, now, pid_alive,
                           wait_for_space)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STEPS = ("wait", "mask", "cmods", "zjoints", "zfinal", "ecam", "nz", "deck", "ztests", "studies", "report")
TAG = "freeze_096"
ZOOMS = (34, 48, 63, 79, 110)
WAIVE_ASPECT = (63, 79, 110)          # Christian, 8 Nov: "update the z63, z79, z110 models that are good enough"
DECK_BLOCKS = {                       # the Z34 deck pans (sequence, sol, station) and their blocks
    "threeforks_north": ("ZCAM05086", 692, "S032D1184"),
    "airey_hill": ("ZCAM05086", 965, "S047D0000"),
    "chal_rocks_large": ("ZCAM08396", 364, "S011D0108"),
}
ZTESTS = (("zcam34_aspect1", ["--zoom", "34", "--zcam-aspect", "1.0"]),
          ("zcam110_aspect1", ["--zoom", "110", "--zcam-aspect", "1.0"]),
          ("zcam110_without_marble", ["--zoom", "110", "--group", "freeze2_z110_without_marble"]))
JOINT_GROUP = "zcam34_joint,zcam_consensus"
OLD_CHAIN_ROOT_STEPS = ("studies", "figures", "maps", "aspect_geometry", "s6")
STALE_S = 900
LOGON_TASK = "MPPP freeze"
REQUIRED_VERSION = "0.96"
DEFAULT_MASK = "mppp_mask_v3"         # the registry default: what a block without mask_checkpoint was processed with
STATUS_FILE, LOG_FILE, REPORT_FILE = "freeze_status.json", "freeze_log.txt", "freeze_report.md"


class Run:
    """The run's state on disk and the steps' children."""

    def __init__(self, a, read_only: bool = False):
        self.a = a
        self.root = Path(a.root)
        self.dir = self.root / "mppp_pipeline"
        if not read_only:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.dir / STATUS_FILE
        self.status = {"started": now(), "tag": TAG, "steps": {}}
        self.before: dict = {}
        self.resumed = False
        if self.status_file.is_file():
            try:
                old = json.loads(self.status_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old = {}
            if isinstance(old.get("steps"), dict):
                self.status, self.before, self.resumed = old, dict(old), True
                if not read_only:
                    self.status.setdefault("restarts", []).append(now())
        self.code = Path(a.code).resolve()
        self.cmods = Path(a.cmods)
        self.redo = set(a.redo or [])
        self.env = dict(os.environ)
        self.env.update({"MPPP_HOME": str(self.code), "MPPP_SCAPES": str(self.root), "MPPP_CMODS": str(self.cmods),
                         "PYTHONPATH": str(self.code / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                         "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                         "MPPP_NOPAUSE": "1"})
        self.python = a.python or sys.executable

    # --------------------------------------------------------------- bookkeeping
    def log(self, msg: str) -> None:
        line = f"{now()}  {msg}"
        print(line, flush=True)
        try:
            with open(self.dir / LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def save(self) -> None:
        self.status["heartbeat"] = now()
        self.status["pid"] = os.getpid()
        tmp = self.status_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.status, indent=1, default=str), encoding="utf-8")
            os.replace(tmp, self.status_file)
        except OSError as e:
            self.log(f"status not written ({e})")

    def beat(self) -> None:
        self.save()

    def step(self, name: str, **kw) -> None:
        self.status["steps"].setdefault(name, {}).update(kw, updated=now())
        self.save()

    def done(self, name: str) -> bool:
        return name not in self.redo and (self.status["steps"].get(name) or {}).get("state") == "done"

    def alive_before(self) -> Optional[int]:
        """The pid of another run of this driver alive on the root (heartbeat within STALE_S, process there)."""
        hb, pid = self.before.get("heartbeat"), self.before.get("pid")
        if self.before.get("state") != "running" or not hb or not pid or pid == os.getpid():
            return None
        try:
            age = (datetime.datetime.now() - datetime.datetime.fromisoformat(hb)).total_seconds()
        except (TypeError, ValueError):
            return None
        return int(pid) if age < STALE_S and pid_alive(int(pid)) else None

    # --------------------------------------------------------------- children
    def run(self, name: str, cmd: Sequence, timeout_h: float = 24 * 7, cwd=None) -> int:
        cmd = [str(c) for c in cmd]
        logf = self.dir / f"freeze_{name}.log"
        if self.a.dry_run:
            self.log(f"{name} (dry run): {' '.join(cmd)}")
            return 0
        wait_for_space(self, name)
        self.log(f"{name}: {' '.join(cmd)}")
        t0 = time.time()
        rc, p = 127, None
        try:
            with open(logf, "a", encoding="utf-8") as f:
                f.write(f"\n==== {now()} {' '.join(cmd)}\n")
                f.flush()
                p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, env=self.env,
                                     cwd=str(cwd or self.code))
                deadline = t0 + timeout_h * 3600
                while True:
                    try:
                        rc = p.wait(timeout=max(5.0, min(300.0, deadline - time.time())))
                        break
                    except subprocess.TimeoutExpired:
                        self.beat()
                        if time.time() > deadline:
                            self.log(f"{name}: past {timeout_h} h - stopped with its process tree")
                            kill_tree(p.pid)
                            f.write(f"\nTIMEOUT after {timeout_h} h: stopped\n")
                            rc = 124
                            break
        except KeyboardInterrupt:
            if p is not None and p.poll() is None:
                kill_tree(p.pid)
            raise
        except Exception as e:                                                        # noqa: BLE001
            self.log(f"{name}: could not run ({type(e).__name__}: {e})")
            if p is not None and p.poll() is None:
                kill_tree(p.pid)
        hours = (time.time() - t0) / 3600
        self.log(f"{name}: exit {rc} after {hours:.2f} h (log {logf})")
        if rc != 0:
            mk = self.root / STOP_MARKER
            try:
                stopped = mk.is_file() and mk.stat().st_mtime >= t0 - 5
            except OSError:
                stopped = False
            if stopped:
                self.log(f"{name}: stopped by stop_mppp.bat - the run ends here; start freeze.bat to continue")
                raise ChainStopped(name)
        return rc

    def pipeline(self, name: str, *args: str) -> int:
        cmd = [self.python, "-m", "mppp_studies.calibration.run_pipeline", "--root", self.root, "--jobs",
               str(self.a.jobs), *args]
        m = self.mask_set()
        if m:
            cmd += ["--set", f"mask_checkpoint={m}"]
        return self.run(name, cmd)

    def mppp(self, name: str, *args: str, timeout_h: float = 30) -> int:
        return self.run(name, [self.python, "-m", "mppp", *args], timeout_h=timeout_h)

    def guarded(self, step: str, fn) -> None:
        if self.done(step) or (self.a.only and step != self.a.only):
            return
        self.step(step, state="running", started=now())
        try:
            note = fn()
            self.step(step, state="done", ended=now(), note=note)
            self.log(f"{step}: done - {note}")
        except (KeyboardInterrupt, ChainStopped):
            self.step(step, state="stopped", ended=now())
            raise
        except Exception as e:                                                        # noqa: BLE001
            self.step(step, state=f"failed: {type(e).__name__}: {e}", ended=now())
            self.log(f"{step}: FAILED ({type(e).__name__}: {e}) - going on\n{traceback.format_exc()}")

    # --------------------------------------------------------------- the mask model
    def mask_set(self) -> Optional[str]:
        """The ``mask_checkpoint`` setting of every pipeline call: None when the blocks' own model stays."""
        return (self.status.get("mask") or {}).get("set") or None

    def s_mask(self) -> str:
        a = self.a
        kept = self.status.get("mask")
        if kept and kept.get("state") == "done":
            return f"kept from before: {kept.get('choice')}"
        current = blocks_mask(self.root)
        cur_label = Path(current).stem if current else DEFAULT_MASK
        spec = a.mask_model
        if spec == "keep":
            choice = {"state": "done", "choice": f"keep ({cur_label})", "set": None, "current": current}
        elif spec != "auto":
            same = spec == current or (current is None and spec == DEFAULT_MASK)
            choice = {"state": "done", "choice": spec, "set": None if same else spec, "current": current}
            if not same:
                self.log(f"mask: {spec} instead of {cur_label} - every image is processed again and every block "
                         "aligned again (the mask checkpoint is in the run key)")
        else:
            choice = self.mask_auto(current, cur_label)
        self.status["mask"] = choice
        self.save()
        return choice["choice"]

    def mask_auto(self, current: Optional[str], cur_label: str) -> dict:
        a = self.a
        home = Path(a.mask_home)
        tag = a.mask_tag or newest_training_tag(home)
        if not tag:
            self.log(f"mask auto: no training found under {home / 'checkpoints'} - {cur_label} stays")
            return {"state": "done", "choice": f"keep ({cur_label}; no training found)", "set": None, "current": current}
        self.log(f"mask auto: waiting for the training '{tag}' and its comparison")
        ck = self.wait_for_training(home, tag)
        if ck is None:
            return {"state": "done", "choice": f"keep ({cur_label}; training {tag} not finished)", "set": None,
                    "current": current, "tag": tag}
        verdict = self.ensure_verdict(home, ck, current or DEFAULT_MASK)
        scores = (verdict or {}).get("models") or {}
        new_iou, base_iou = scores.get(Path(ck).stem), scores.get(cur_label)
        self.log(f"mask auto: exclusion IoU on the {(verdict or {}).get('subset', '?')} frames {json.dumps(scores)}")
        if new_iou is not None and base_iou is not None and new_iou - base_iou >= a.mask_margin:
            self.log(f"mask auto: {ck.name} ({new_iou:.4f}) beats {cur_label} ({base_iou:.4f}) by >= {a.mask_margin}"
                     " - the blocks use it: every image is processed again, every block aligned again")
            return {"state": "done", "choice": f"new {ck.name}", "set": str(ck), "current": current, "tag": tag,
                    "iou": scores, "margin": new_iou - base_iou}
        why = "no comparison" if new_iou is None or base_iou is None else f"margin {new_iou - base_iou:+.4f} < {a.mask_margin}"
        self.log(f"mask auto: {cur_label} stays ({why})")
        return {"state": "done", "choice": f"keep ({cur_label}; {why})", "set": None, "current": current, "tag": tag,
                "iou": scores}

    def wait_for_training(self, home: Path, tag: str) -> Optional[Path]:
        """The trained checkpoint of ``tag`` once its training is through (summary or release candidate written);
        None when the training log goes quiet or the wait runs out."""
        t0, last = time.time(), 0.0
        while True:
            ck = trained_checkpoint(home, tag)
            if ck is not None:
                self.log(f"mask auto: {ck.name} trained")
                return ck
            logs = sorted((home / "checkpoints").glob(f"*_{tag}_debug/log.csv"), key=lambda p: p.stat().st_mtime)
            age = (time.time() - logs[-1].stat().st_mtime) / 60 if logs else None
            if self.a.dry_run:
                return None
            if age is not None and age > self.a.mask_quiet_min and time.time() - t0 > 600:
                self.log(f"mask auto: the training log quiet for {age:.0f} min - going on without it")
                return None
            if time.time() - t0 > self.a.mask_max_wait_h * 3600:
                self.log(f"mask auto: not trained after {self.a.mask_max_wait_h} h - going on without it")
                return None
            if time.time() - last > 1800:
                last = time.time()
                self.log("mask auto: waiting for the training (" + ("no log yet" if age is None else f"log {age:.0f} min old")
                         + f"; {(time.time() - t0) / 3600:.1f} h so far)")
            self.beat()
            time.sleep(120)

    def ensure_verdict(self, home: Path, ck: Path, baseline: str) -> Optional[dict]:
        """``mask_verdict.json`` of the training (train_masks.bat --compare wrote it), else the comparison run here."""
        out = home / "Claude outputs" / f"mask_{ck.stem}"
        vf = out / "mask_verdict.json"
        if not vf.is_file() and not self.a.dry_run:
            # give train_mask's own comparison a moment (it runs right after the training)
            t0 = time.time()
            while not vf.is_file() and time.time() - t0 < 1800:
                self.beat()
                time.sleep(60)
        if vf.is_file():
            try:
                return json.loads(vf.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                self.log(f"mask auto: {vf} unreadable ({e})")
        data = self.a.mask_data or newest_training_set(self.a.mask_root)
        if not data:
            self.log("mask auto: no training set found for the comparison")
            return None
        seen = self.a.mask_seen or previous_training_set(data)
        cmd = [self.a.ab_python, "-m", "mppp_studies.mask_training.compare_mask_models", "--data", data, "--split-ckpt",
               ck, "--models", baseline, ck, "--out", out / "mask_ab"]
        if seen:
            cmd += ["--seen", seen, "--seen-split", baseline, seen]
        rc = self.run("mask_ab", cmd, timeout_h=3.0)
        if rc != 0 and not self.a.dry_run:
            return None
        try:
            from mppp_studies.mask_training.train_mask import read_verdict_from_ab
            v = read_verdict_from_ab(out / "mask_ab", new=ck, baseline=baseline)
            vf.parent.mkdir(parents=True, exist_ok=True)
            vf.write_text(json.dumps(v, indent=1), encoding="utf-8")
            return v
        except Exception as e:                                                        # noqa: BLE001
            self.log(f"mask auto: verdict not read ({type(e).__name__}: {e})")
            return None

    # --------------------------------------------------------------- steps
    def s_wait(self) -> str:
        from mppp.runner import read_status
        t0 = time.time()
        while True:
            pl = read_status(self.dir / "pipeline_status.json") or {}
            ch = read_status(self.dir / "freeze_chain_status.json") or {}
            steps = ch.get("steps") or {}
            chain_root_done = (not ch.get("alive")) or all((steps.get(s) or {}).get("state") == "done"
                                                           for s in OLD_CHAIN_ROOT_STEPS)
            if (not pl.get("alive") and (self.a.now or chain_root_done)) or self.a.dry_run:
                return "pipeline idle; " + ("--now" if self.a.now and not chain_root_done else
                                            "the first chain's steps on this root are done or it is not running")
            if time.time() - t0 > self.a.max_wait_h * 3600:
                raise RuntimeError(f"still busy after {self.a.max_wait_h} h (pipeline alive {pl.get('alive')})")
            if int(time.time() - t0) % 3600 < 600:
                left = [s for s in OLD_CHAIN_ROOT_STEPS if (steps.get(s) or {}).get("state") != "done"]
                self.log(f"wait: pipeline {'running (' + str(pl.get('stage')) + ')' if pl.get('alive') else 'idle'}; "
                         f"first chain {'running, root steps left: ' + ', '.join(left) if ch.get('alive') else 'not running'}")
            self.beat()
            time.sleep(600)

    def s_cmods(self) -> str:
        src = Path(self.a.cmods_from)
        if not self.cmods.is_dir() or not any(self.cmods.glob("*.json")):
            if self.a.dry_run:
                return f"(dry run) {self.cmods} would be seeded from {src}"
            self.cmods.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.cmods / f.name)
            self.log(f"cmods: {self.cmods} seeded from {src}")
        rig = self.cmods / "M2020_N_rig.json"
        try:
            source = str(json.loads(rig.read_text(encoding="utf-8")).get("source", ""))
        except (OSError, ValueError):
            source = ""
        if "freeze" in source.lower():
            return f"{self.cmods}: the freeze Navcam models and rig already in it ({source[:80]})"
        if not self.a.candidate:
            return f"{self.cmods}: the models as they are (no --candidate)"
        rc = self.run("cmods", [self.python, "-m", "mppp_studies.calibration.promote_cmods", self.a.candidate, "--note",
                                self.a.candidate_note], timeout_h=0.5)
        if rc != 0:
            raise RuntimeError(f"promote_cmods exit {rc}")
        return f"{self.cmods}: {self.a.candidate} promoted into it"

    def s_zjoints(self) -> str:
        args = ["--from", "zcam_runs", "--redo", "zcam_runs", "zcam_joints", "--stop-before", "error_analysis", "--no-nz"]
        if self.a.zcam_waive_aspect:
            args += ["--zcam-waive-aspect", *map(str, self.a.zcam_waive_aspect)]
        rc = self.pipeline("zjoints", *args)
        since = (self.status["steps"].get("zjoints") or {}).get("started", "")
        return f"exit {rc}; " + "; ".join(self.zjoint_lines(since=since))

    def s_zfinal(self) -> str:
        return f"exit {self.pipeline('zfinal', '--from', 'zcam_runs', '--redo', 'zcam_runs', '--stop-before', 'zcam_joints')}"

    def s_ecam(self) -> str:
        return f"exit {self.pipeline('ecam', '--from', 'ecam_runs', '--redo', 'ecam_runs', 'ecam_joint', '--stop-before', 'zcam_runs', '--no-promote')}"

    def s_nz(self) -> str:
        out = self.root / "camera_analysis" / f"nz34_joint_freeze_{datetime.date.today():%Y%m%d}"
        rc = self.mppp("nz", "--log", f"{out}_log.txt", "calib", "nz", "--group", JOINT_GROUP, "--navcam-group",
                       "ecam_consensus", "--root", self.root, "--out", out)
        return f"exit {rc} ({out})"

    def s_deck(self) -> str:
        from mppp.waypoints.sites import load_site_table, work_folder
        table = load_site_table()
        rows = {}
        for site, (seq, sol, st) in DECK_BLOCKS.items():
            if site not in table["sites"]:
                rows[site] = {"error": "not in sites.json"}
                continue
            w = Path(work_folder(str(self.root), site, True, table["sites"][site]["sols"]))
            if not (w / "colmap" / "run_done.json").is_file() and not self.a.dry_run:
                self.log(f"deck: {site}: no finished block at {w} - skipped")
                rows[site] = {"error": "no finished block"}
                continue
            rc = self.run(f"deck_{site}", [self.python, "-m", "mppp_studies.calibration.deck", w, "--zcam-rig", "full",
                                           "--rig", "full"], timeout_h=12)
            js = w / "colmap" / "deck" / "deck_cahv_ba_deck.json"
            if rc == 0 and js.is_file():
                d = json.loads(js.read_text(encoding="utf-8"))
                rows[site] = {k: d.get(k) for k in ("observations", "images", "deck_rms_px_start", "deck_rms_px",
                                                    "baselines_mm_start", "baselines_mm", "baseline_change_ppm")}
                rows[site]["deck_pan"] = {"sequence": seq, "sol": sol, "station": st}
                rows[site]["file"] = str(js)
            else:
                rows[site] = {"error": f"exit {rc}" if not self.a.dry_run else "dry run"}
        (self.dir / "freeze_deck.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
        return "; ".join(f"{k}: " + (", ".join(f"{n} {v:.2f} mm" for n, v in (r.get("baselines_mm") or {}).items())
                                    if "error" not in r else r["error"]) for k, r in rows.items())

    def s_ztests(self) -> str:
        notes = []
        day = f"{datetime.date.today():%Y%m%d}"
        for name, args in ZTESTS:
            out = self.root / "camera_analysis" / f"{name}_freeze_{day}"
            if (out / "check.json").is_file():
                notes.append(f"{name}: done before")
                continue
            grp = [] if "--group" in args else ["--group", JOINT_GROUP]
            rc = self.mppp(f"ztest_{name}", "--log", f"{out}_log.txt", "calib", "zcam", *args, *grp, "--root", self.root,
                           "--out", out, "--no-promote", "--iterations", "300")
            notes.append(f"{name}: exit {rc}")
        return "; ".join(notes)

    def s_studies(self) -> str:
        rc = self.pipeline("studies", "--from", "error_analysis", "--redo", "error_analysis", "rover_frame", "--no-zcam-states")
        return f"exit {rc} ({self.dir / 'REPORT.md'})"

    def s_report(self) -> str:
        md = report_md(self)
        (self.dir / REPORT_FILE).write_text(md, encoding="utf-8")
        if self.a.report_copy:
            try:
                Path(self.a.report_copy).mkdir(parents=True, exist_ok=True)
                (Path(self.a.report_copy) / REPORT_FILE).write_text(md, encoding="utf-8")
            except OSError as e:
                self.log(f"report: copy not written ({e})")
        return str(self.dir / REPORT_FILE)

    # --------------------------------------------------------------- results
    def zjoint_lines(self, since: str = ""):
        out = []
        for z in ZOOMS:
            c = newest(self.root / "camera_analysis", f"zcam{z}_joint_*", "check.json", since)
            if c is None:
                out.append(f"Z{z}: no joint")
                continue
            try:
                chk = json.loads((c / "check.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                out.append(f"Z{z}: check.json unreadable ({c.name})")
                continue
            bad = [k["check"] for k in chk.get("checks", []) if not k.get("ok")]
            wv = (c / "promotion_waiver.json").is_file()
            out.append(f"Z{z}: " + ("passed" if chk.get("ok") else ("promoted, waived: " if wv else "failed: ") + ", ".join(bad))
                       + f" ({c.name})")
        return out


# ------------------------------------------------------------------ the mask model on disk
def blocks_mask(root: Path) -> Optional[str]:
    """The ``mask_checkpoint`` the finished blocks under ``root`` were processed with (the most common value of their
    ``run_settings.json``); None: the registry default."""
    counts: dict = {}
    for f in glob.glob(str(root / "*" / "colmap" / "run_settings.json")):
        try:
            s = json.loads(Path(f).read_text(encoding="utf-8")).get("settings") or {}
        except (OSError, ValueError):
            continue
        m = s.get("mask_checkpoint") or None
        counts[m] = counts.get(m, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def newest_training_tag(home: Path) -> Optional[str]:
    """The tag of the newest training under ``home/checkpoints`` (``<stem>_<tag>_debug/log.csv``)."""
    logs = sorted((home / "checkpoints").glob("*_debug/log.csv"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    m = re.match(r"convnext_\w+?_seg_\d{8}(?:_(\d+))?(?:_(.+))?_debug$", logs[-1].parent.name)
    return m.group(2) if m and m.group(2) else None


def trained_checkpoint(home: Path, tag: str) -> Optional[Path]:
    """The checkpoint of ``tag`` once its training is through: its summary or release candidate exists in
    ``Claude outputs/mask_<checkpoint>/`` or its card says so."""
    cks = sorted((home / "checkpoints").glob(f"*_{tag}.pt"), key=lambda p: p.stat().st_mtime)
    if not cks:
        return None
    ck = cks[-1]
    out = home / "Claude outputs" / f"mask_{ck.stem}"
    if (out / "summary.md").is_file() or any(out.glob("*.safetensors")) or (out / "mask_verdict.json").is_file():
        return ck
    card = ck.with_suffix(".json")
    try:
        c = json.loads(card.read_text(encoding="utf-8"))
        if len(c.get("history") or []) >= int((c.get("training") or {}).get("epochs") or 10 ** 9):
            return ck                                 # every epoch in the card's history (the last one improved)
    except (OSError, ValueError, TypeError):
        pass
    return None


def newest_training_set(mask_root: str) -> Optional[str]:
    sets = []
    for d in glob.glob(str(Path(mask_root) / "masks_training_set_v*")):
        m = re.search(r"_v(\d+)$", d)
        if m and (Path(d) / "masks").is_dir():
            sets.append((int(m.group(1)), d))
    return max(sets)[1] if sets else None


def previous_training_set(data: str) -> Optional[str]:
    m = re.search(r"^(.*_v)(\d+)$", str(data))
    if not m:
        return None
    prev = f"{m.group(1)}{int(m.group(2)) - 1}"
    return prev if (Path(prev) / "masks").is_dir() else None


# ------------------------------------------------------------------ helpers
def newest(folder: Path, pattern: str, marker: str, since: str = ""):
    best = None
    for d in glob.glob(str(folder / pattern)):
        p = Path(d)
        f = p / marker
        if not f.is_file():
            continue
        t = datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")
        if since and t < since:
            continue
        if best is None or f.stat().st_mtime > (best / marker).stat().st_mtime:
            best = p
    return best


def kill_tree(pid: int) -> None:
    try:
        from mppp.batch import kill_tree as kt
        kt(pid)
    except Exception:                                                                 # noqa: BLE001
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def git_head(tree: Path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(tree), "log", "--oneline", "-1"], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or r.stderr.strip()
    except Exception as e:                                                            # noqa: BLE001
        return f"? ({e})"


def code_version(r: Run) -> str:
    """``<version> <file>`` of the mppp the children will import (their environment)."""
    if r.a.dry_run:
        return f"{REQUIRED_VERSION}.x (dry run)"
    chk = subprocess.run([r.python, "-c", "import mppp; print(mppp.__version__, mppp.__file__)"], capture_output=True,
                         text=True, env=r.env, cwd=str(r.code) if r.code.is_dir() else None)
    if chk.stdout.strip():
        return chk.stdout.strip()
    err = chk.stderr.strip().splitlines()
    return err[-1] if err else "?"


def logon_task_register(r: Run) -> None:
    """A RunOnce entry of this user that starts freeze.bat again at the next logon (a reboot the run did not see
    coming).  Windows removes the entry when it fires; the run sets it at every start and removes it when it ends."""
    if not WINDOWS or r.a.no_logon_task or r.a.dry_run:
        return
    bat = r.code / "scripts" / "windows" / "freeze.bat"
    if not bat.is_file():
        bat = REPO / "scripts" / "windows" / "freeze.bat"
    if not bat.is_file():
        r.log(f"logon task: no {bat} - not registered")
        return
    launcher = r.dir / "freeze_logon.bat"
    try:
        launcher.write_text("@echo off\r\nset \"MPPP_NOPAUSE=1\"\r\n" + f"set \"MPPP_FREEZE_CODE={r.code}\"\r\n"
                            + f"call \"{bat}\" --root \"{r.root}\"\r\n", encoding="ascii", newline="")
        res = subprocess.run(["reg", "add", RUNONCE_KEY, "/v", LOGON_TASK, "/t", "REG_SZ", "/d", f'"{launcher}"', "/f"],
                             capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            r.log(f"logon task: RunOnce '{LOGON_TASK}' set - after a reboot the run continues at the next logon")
            r.status["logon_task"] = True
        else:
            r.log(f"logon task: not set (reg exit {res.returncode}) - after a reboot, double-click freeze.bat again")
    except Exception as e:                                                            # noqa: BLE001
        r.log(f"logon task: not registered ({type(e).__name__}: {e})")


def logon_task_delete(r: Run) -> None:
    if not WINDOWS or not r.status.get("logon_task"):
        return
    try:
        res = subprocess.run(["reg", "delete", RUNONCE_KEY, "/v", LOGON_TASK, "/f"], capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            r.status["logon_task"] = False
    except Exception:                                                                 # noqa: BLE001
        pass


def report_md(r: Run) -> str:
    st = r.status["steps"]
    mask = r.status.get("mask") or {}
    lines = [f"# Freeze run report ({now()})", "", f"Root `{r.root}`, code `{r.code}` ({r.status.get('code', '?')}), "
             f"models `{r.cmods}`, mask model: {mask.get('choice', '-')}.", "", "## Steps", "", "| step | state | note |",
             "|---|---|---|"]
    for s in STEPS:
        e = st.get(s) or {}
        lines.append(f"| {s} | {e.get('state', '-')} | {str(e.get('note', '')).replace('|', '/')[:400]} |")
    lines += ["", "## Mastcam-Z joints", ""] + [f"- {x}" for x in r.zjoint_lines(since=(st.get("zjoints") or {}).get("started", ""))]
    try:
        s_ = json.loads((r.dir / "state.json").read_text(encoding="utf-8"))
        lines += ["", "## Navcam joint and Hazcam fit (checks, not promoted)", "", f"- {(s_.get('ecam_joint') or {}).get('note')}"]
    except (OSError, ValueError):
        pass
    dk = r.dir / "freeze_deck.json"
    if dk.is_file():
        lines += ["", "## Deck in the adjustment (Z34 deck-pan blocks)", "",
                  "| block | deck pan | targets | rms px | baselines mm (start -> deck) |", "|---|---|---|---|---|"]
        try:
            rows = json.loads(dk.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rows = {}
        for k, v in rows.items():
            if "error" in v:
                lines.append(f"| {k} | | | | {v['error']} |")
                continue
            b0, b1 = v.get("baselines_mm_start") or {}, v.get("baselines_mm") or {}
            lines.append(f"| {k} | {v['deck_pan']['sequence']} sol {v['deck_pan']['sol']} | {v.get('observations')} | "
                         f"{v.get('deck_rms_px_start') or 0:.2f} -> {v.get('deck_rms_px') or 0:.2f} | "
                         + ", ".join(f"{n} {b0.get(n, float('nan')):.2f} -> {b1[n]:.2f}" for n in b1) + " |")
    lines += ["", "## Comparison joints (not promoted)", ""]
    for name, _ in ZTESTS:
        c = newest(r.root / "camera_analysis", f"{name}_freeze_*", "check.json")
        if c is None:
            lines.append(f"- {name}: not run")
            continue
        try:
            chk = json.loads((c / "check.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lines.append(f"- {name}: check.json unreadable")
            continue
        res = {k["check"]: k.get("note") for k in chk.get("checks", [])}
        lines.append(f"- {name}: {'passed' if chk.get('ok') else 'failed: ' + ', '.join(k['check'] for k in chk['checks'] if not k['ok'])}; "
                     f"residuals {res.get('Mastcam-Z residuals', '-')}, {res.get('Mastcam-Z residual rms', '-')} ({c})")
    lines += ["", "## Next", "", "- Compare the Z34 and Z110 aspect-held joints with the promoted (aspect-free) ones:"
              " adopt square pixels when the median residual rises by less than ~0.005 px.",
              f"- Make `{r.cmods}` the models in use (`mppp cmods init`, or scripts\\windows\\mppp_cmods.txt).", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ status, update, main
def describe(r: Run) -> str:
    """One line on the run: running, stopped by the user, done, failed, or not running without having ended."""
    st = r.status if r.resumed else {}
    if not st:
        return f"freeze on {r.root}: not started"
    state = st.get("state", "?")
    hb = st.get("heartbeat")
    try:
        age = (datetime.datetime.now() - datetime.datetime.fromisoformat(hb)).total_seconds()
    except (TypeError, ValueError):
        age = None
    cur = next((s for s in STEPS if (st["steps"].get(s) or {}).get("state") == "running"), None)
    if state == "running":
        alive = age is not None and age < STALE_S and st.get("pid") and pid_alive(int(st["pid"]))
        if alive:
            return (f"freeze on {r.root}: running - step {cur or '?'} since {(st['steps'].get(cur) or {}).get('started', '?')}, "
                    f"heartbeat {age:.0f} s ago (pid {st['pid']})")
        return (f"freeze on {r.root}: NOT running, and it did not end on its own (last heartbeat {hb}, step {cur or '?'}; "
                "the window was closed, the computer rebooted or it crashed) - start freeze.bat to continue")
    if state == "stopped":
        return f"freeze on {r.root}: stopped by the user at {st.get('ended', hb)} ({st.get('stopped_by', 'Ctrl-C or stop_mppp.bat')}) - start freeze.bat to continue"
    if state == "done":
        return f"freeze on {r.root}: done at {st.get('ended', hb)} - {r.dir / REPORT_FILE}"
    return f"freeze on {r.root}: {state} at {st.get('ended', hb)} - start freeze.bat to continue"


def show_status(r: Run) -> int:
    print(describe(r))
    st = r.status if r.resumed else {"steps": {}}
    print(f"  code {r.code}: {git_head(r.code)}" + (f"  (run: {st['code']})" if st.get("code") else ""))
    m = st.get("mask") or {}
    if m:
        print(f"  mask model: {m.get('choice')}")
    for s in STEPS:
        e = st["steps"].get(s) or {}
        extra = {k: v for k, v in e.items() if k in ("note", "started", "ended")}
        print(f"  {s:9s} {str(e.get('state', '-'))[:40]:40s} {json.dumps(extra, default=str)[:160] if extra else ''}")
    try:
        from mppp.runner import read_status
        cur = read_status(r.dir / "pipeline_status.json")
        print(f"  pipeline: {(cur or {}).get('state', 'not started')}" + (f", stage {cur.get('stage')}, alive {cur.get('alive')}" if cur else ""))
    except Exception as e:                                                            # noqa: BLE001
        print(f"  pipeline: unknown ({e})")
    return 0


def update_code(a) -> int:
    """``git pull --ff-only`` of the run's checkout from ``--from-tree`` main.  Nothing is started.  Refused while a
    run is alive on the root (code must not change under it) unless ``--force-update``."""
    r = Run(a, read_only=True)
    pid = r.alive_before()
    if pid and not a.force_update:
        print(f"not updated: {describe(r)}\n(--force-update to update anyway; a running step keeps the modules it loaded)")
        return 2
    code = Path(a.code)
    if not (code / ".git").exists():
        print(f"{code} is not a git checkout: freeze.bat clones it from {a.from_tree}")
        return 1
    print(f"before: {code}: {git_head(code)}")
    rc = subprocess.run(["git", "-C", str(code), "pull", "--ff-only", str(a.from_tree), "main"]).returncode
    print(f"git pull --ff-only {a.from_tree} main: exit {rc}")
    print(f"after:  {code}: {git_head(code)}")
    print(f"        {a.from_tree}: {git_head(Path(a.from_tree))}")
    return 0 if rc == 0 else 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_FREEZE_ROOT", r"D:\rover_blocks"))
    ap.add_argument("--code", default=os.environ.get("MPPP_FREEZE_CODE", str(REPO)), help="the run's checkout")
    ap.add_argument("--python", default=None, help="the Python of the steps (default: this one)")
    ap.add_argument("--cmods", default=os.environ.get("MPPP_FREEZE_CMODS", r"D:\rover_blocks\cmods_freeze2"),
                    help="the run's camera models (MPPP_CMODS of every step)")
    ap.add_argument("--cmods-from", default=r"D:\scapes\cmods", help="the models in use the folder is seeded from")
    ap.add_argument("--candidate", default=None, help="a folder of models promoted into --cmods when it holds no freeze rig")
    ap.add_argument("--candidate-note", default="freeze (0.96): the candidate models promoted at the start of the run")
    ap.add_argument("--mask-model", default="keep", metavar="keep|auto|FILE|NAME",
                    help="the mask model of the blocks (see above; default keep)")
    ap.add_argument("--mask-tag", default=None, help="auto: the training's --tag (default: the newest training)")
    ap.add_argument("--mask-home", default=os.environ.get("MPPP_HOME_MAIN", r"D:\code\MPPP"),
                    help="auto: the tree train_masks.bat writes checkpoints\\ and 'Claude outputs\\' into")
    ap.add_argument("--mask-root", default=r"D:\masks", help="auto: where the training sets are")
    ap.add_argument("--mask-data", default=None, help="auto: the training set of the comparison (default: the newest)")
    ap.add_argument("--mask-seen", default=None, help="auto: the older set the blocks' model trained on (default: the one before)")
    ap.add_argument("--mask-margin", type=float, default=0.01, help="auto: the exclusion-IoU margin the new model must win by")
    ap.add_argument("--mask-max-wait-h", type=float, default=14.0)
    ap.add_argument("--mask-quiet-min", type=float, default=60.0)
    ap.add_argument("--ab-python", default=os.environ.get("MPPP_TRAIN_PYTHON") or sys.executable,
                    help="auto: a Python with a CUDA torch for the comparison")
    ap.add_argument("--zcam-waive-aspect", type=int, nargs="*", default=list(WAIVE_ASPECT), metavar="ZOOM")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("MPPP_JOBS", "2") or 2))
    ap.add_argument("--now", action="store_true", help="start once the pipeline lock is free")
    ap.add_argument("--max-wait-h", type=float, default=24 * 7)
    ap.add_argument("--redo", nargs="*", choices=STEPS, default=[])
    ap.add_argument("--only", choices=STEPS, default=None)
    ap.add_argument("--report-copy", default=None, help="a folder the report is copied to")
    ap.add_argument("--no-logon-task", action="store_true")
    ap.add_argument("--any-version", action="store_true", help=f"run on an MPPP that is not {REQUIRED_VERSION}.x")
    ap.add_argument("--dry-run", action="store_true", help="log the commands, run nothing")
    ap.add_argument("--status", action="store_true", help="where the run is; nothing started")
    ap.add_argument("--update-code", action="store_true", help="git pull --ff-only --from-tree main into --code; nothing started")
    ap.add_argument("--from-tree", default=r"D:\code\MPPP", help="--update-code: the tree pulled from")
    ap.add_argument("--force-update", action="store_true")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.status:
        return show_status(Run(a, read_only=True))
    if a.update_code:
        return update_code(a)
    r = Run(a)
    pid = r.alive_before()
    if pid:
        r.log(f"not started: a freeze is already running on {r.root} (pid {pid}) - one window only")
        return 2
    r.status["state"] = "running"
    r.status.pop("ended", None)
    r.status.pop("stopped_by", None)
    r.log(f"freeze {'continued' if r.resumed and r.status['steps'] else 'started'} (pid {os.getpid()}, root {r.root}, "
          f"code {r.code} [{git_head(r.code)}], models {r.cmods}, mask {a.mask_model})")
    if r.resumed:
        done = [s for s in STEPS if r.done(s)]
        r.log(f"steps done before: {', '.join(done) if done else 'none'}" + (f"; redo: {', '.join(sorted(r.redo))}" if r.redo else ""))
    ver = code_version(r)
    r.status["code"] = f"{ver} [{git_head(r.code)}]"
    r.log(f"MPPP for the run: {ver}")
    if not ver.startswith(REQUIRED_VERSION) and not a.any_version:
        r.log(f"ERROR: the run needs MPPP {REQUIRED_VERSION}.x in {r.code} (freeze.bat --update-code; --any-version to run anyway)")
        r.status["state"] = "failed"
        r.save()
        return 1
    keep_awake(r)
    logon_task_register(r)
    free_gb(r, "start")
    try:
        for s in STEPS:
            r.guarded(s, getattr(r, f"s_{s}"))
        failed = [s for s in STEPS if str((r.status["steps"].get(s) or {}).get("state", "")).startswith("failed")]
        r.status.update(state="done", ended=now(), failed=failed)
        r.save()
        logon_task_delete(r)
        r.log("freeze done: " + str(r.dir / REPORT_FILE) + (f" - steps that failed: {', '.join(failed)}" if failed else ""))
        return 1 if failed else 0
    except ChainStopped as e:
        r.status.update(state="stopped", ended=now(), stopped_by=f"stop_mppp.bat during {e}")
        r.save()
        logon_task_delete(r)
        r.log("stopped by the user (stop_mppp.bat); start freeze.bat to continue")
        return 130
    except KeyboardInterrupt:
        r.status.update(state="stopped", ended=now(), stopped_by="Ctrl-C")
        r.save()
        logon_task_delete(r)
        r.log("stopped by the user (Ctrl-C); start freeze.bat to continue")
        return 130
    except Exception:                                                                 # noqa: BLE001
        r.status.update(state="failed", ended=now())
        r.save()
        logon_task_delete(r)
        r.log("the run itself failed:\n" + traceback.format_exc() + "start freeze.bat to continue")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
