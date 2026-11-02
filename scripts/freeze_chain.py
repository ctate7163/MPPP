"""
The freeze run, unattended, after the morning's work (2 Nov 2026, Christian: "I will be away from the computer when
these finish"):

1. wait for the mask training (``train_masks.bat``) to finish - its release candidate ``.safetensors`` in
   ``Claude outputs\\mask_<checkpoint>\\``, or its log quiet for ``--quiet-min`` minutes, at most ``--max-wait-h``;
2. the mask A/B of v3, v4 and v4.1 on the held-out frames neither older model trained on
   (``compare_mask_models.py --seen-split``) -> ``Claude outputs\\mask_ab_v3_v4_v41\\``;
3. the point-thinning study S2 on rockytop_example (all points against a cap of 1500 per image) and its verdict:
   the cap goes into the freeze run when the frame centres differ by < 1 cm (p95), the attitudes by < 0.1 mdeg (p95)
   and the per-bin focal lengths by < 0.05 % (max) from the all-points solution (Christian, 28 Oct);
4. the freeze run: ``run_pipeline.py --reset`` (every stage again: the state of 28 Oct says the ecam stages are done),
   with ``--set ba_points_cap=1500`` if S2 passed;
5. then, if it exists, ``<root>\\paper_studies\\run_paper_studies.py`` (the analyses for the paper, written during the
   freeze run outside the repository).

Each step that fails is logged and the chain goes on.  Log: ``<root>\\mppp_pipeline\\freeze_chain_log.txt``; status:
``freeze_chain_status.json`` next to it (for reading from afar).

    scripts\\windows\\freeze_chain.bat                 everything
    scripts\\windows\\freeze_chain.bat --skip-s2       no S2 (no cap)
    scripts\\windows\\freeze_chain.bat --no-freeze     steps 1-3 only
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CKPT_DIR = REPO / "checkpoints"
OUTPUTS = REPO / "Claude outputs"
V4 = CKPT_DIR / "convnext_tiny_s4_seg_20261030_v8.pt"
V41 = CKPT_DIR / "convnext_tiny_s4_seg_20261102_v8_v4.1.pt"
S2_SITE = "rockytop_example"
S2_LIMITS = {"centre_cm_p95": 1.0, "attitude_mdeg_p95": 0.1, "focal_pct_max": 0.05}


class Chain:
    def __init__(self, a):
        self.a = a
        self.root = Path(a.root)
        d = self.root / "mppp_pipeline"
        d.mkdir(parents=True, exist_ok=True)
        self.logf = d / "freeze_chain_log.txt"
        self.statf = d / "freeze_chain_status.json"
        self.status = {"started": now(), "steps": {}}

    def log(self, msg: str) -> None:
        line = f"{now()}  {msg}"
        print(line, flush=True)
        with open(self.logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def step(self, name: str, **kw) -> None:
        self.status["steps"].setdefault(name, {}).update(kw, updated=now())
        self.statf.write_text(json.dumps(self.status, indent=1, default=str), encoding="utf-8")

    def run(self, name: str, cmd, timeout_h: float) -> int:
        cmd = [str(c) for c in cmd]
        self.log(f"{name}: {' '.join(cmd)}")
        self.step(name, state="running", command=cmd)
        t = time.time()
        try:
            rc = subprocess.run(cmd, cwd=str(REPO), timeout=timeout_h * 3600).returncode
        except subprocess.TimeoutExpired:
            rc = 124
        except Exception as e:                                                # noqa: BLE001
            self.log(f"{name}: could not start ({type(e).__name__}: {e})")
            rc = 127
        self.log(f"{name}: exit {rc} after {(time.time() - t) / 3600:.2f} h")
        self.step(name, state="done" if rc == 0 else f"exit {rc}", hours=round((time.time() - t) / 3600, 2))
        return rc


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def wait_for_training(c: Chain) -> None:
    stem = V41.stem
    done = OUTPUTS / f"mask_{stem}" / f"{stem}.safetensors"
    log = CKPT_DIR / f"{stem}_debug" / "log.csv"
    t0 = time.time()
    c.step("wait_training", state="waiting", until=str(done))
    while True:
        if done.is_file():
            c.log(f"wait_training: {done.name} written - the training is done")
            c.step("wait_training", state="done")
            return
        age = (time.time() - log.stat().st_mtime) / 60 if log.is_file() else None
        if age is not None and age > c.a.quiet_min and (time.time() - t0) > 600:
            c.log(f"wait_training: the training log quiet for {age:.0f} min - going on")
            c.step("wait_training", state=f"log quiet {age:.0f} min")
            return
        if time.time() - t0 > c.a.max_wait_h * 3600:
            c.log(f"wait_training: still not done after {c.a.max_wait_h} h - going on")
            c.step("wait_training", state="timeout")
            return
        time.sleep(120)


def s2_verdict(c: Chain) -> bool:
    from mppp.waypoints.sites import load_site_table, work_folder
    from tie_point_studies import compare_solutions
    t = load_site_table()
    work = Path(work_folder(str(c.root), S2_SITE, True, t["sites"][S2_SITE]["sols"]))
    f = work / "studies" / "S2.json"
    try:
        res = json.loads(f.read_text(encoding="utf-8"))
        v = res["variants"]
        cmp = compare_solutions(v["all"], v["cap1500"])
    except Exception as e:                                                    # noqa: BLE001
        c.log(f"S2 verdict: no comparison ({type(e).__name__}: {e}) - no cap")
        c.step("s2", verdict="no comparison")
        return False
    got = {"centre_cm_p95": (cmp.get("centre_cm") or {}).get("p95"),
           "attitude_mdeg_p95": (cmp.get("attitude_mdeg") or {}).get("p95"),
           "focal_pct_max": (cmp.get("focal_pct") or {}).get("max")}
    ok = all(got[k] is not None and got[k] < lim for k, lim in S2_LIMITS.items())
    c.log(f"S2 verdict: {json.dumps(got)} against {json.dumps(S2_LIMITS)} -> {'cap 1500' if ok else 'no cap'}")
    c.step("s2", verdict="cap 1500" if ok else "no cap", comparison=got)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES") or "D:\\scapes")
    ap.add_argument("--ab-python", default=sys.executable, help="a Python with a CUDA torch (the mask A/B)")
    ap.add_argument("--max-wait-h", type=float, default=6.0)
    ap.add_argument("--quiet-min", type=float, default=45.0)
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--skip-s2", action="store_true")
    ap.add_argument("--no-freeze", action="store_true")
    a = ap.parse_args(argv)
    c = Chain(a)
    c.log(f"freeze chain started (root {c.root})")
    wait_for_training(c)
    if not a.skip_ab:
        if V41.is_file():
            c.run("mask_ab", [a.ab_python, HERE / "compare_mask_models.py", "--data", "D:\\masks\\masks_training_set_v8",
                              "--split-ckpt", V4, "--models", "mppp_mask_v3", V4, V41,
                              "--seen", "D:\\masks\\masks_training_set_v7",
                              "--seen-split", "mppp_mask_v3", "D:\\masks\\masks_training_set_v7",
                              "--out", OUTPUTS / "mask_ab_v3_v4_v41"], timeout_h=2.0)
        else:
            c.log(f"mask_ab: {V41} not found - skipped")
    cap = False
    if not a.skip_s2:
        c.run("s2", [sys.executable, HERE / "tie_point_studies.py", "S2", "--site", S2_SITE, "--variants", "all",
                     "cap1500", "--rerun", "--root", c.root], timeout_h=4.0)
        cap = s2_verdict(c)
    if a.no_freeze:
        c.log("--no-freeze: done")
        return 0
    cmd = [sys.executable, REPO / "studies" / "calibration" / "run_pipeline.py", "--root", c.root, "--reset"]
    if cap:
        cmd += ["--set", "ba_points_cap=1500"]
    c.run("freeze", cmd, timeout_h=24 * 7)
    extra = c.root / "paper_studies" / "run_paper_studies.py"
    if extra.is_file():
        c.run("paper_studies", [sys.executable, extra, "--root", c.root], timeout_h=72)
    else:
        c.log(f"paper_studies: {extra} not there - nothing more")
    c.log("freeze chain done")
    c.step("chain", state="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
