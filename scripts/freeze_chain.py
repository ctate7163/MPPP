"""
The freeze run, unattended (2 Nov 2026, Christian: "I will be away from the computer when these finish"; 14:52:
"include this v8_v4.2 training and its tests in the freeze. Train this before running the final blocks and use it if
it is better than v3. The blocks should be saved to D:\\rover_blocks"):

1. the new root (``--root``, default ``D:\\rover_blocks``): the found list and the focus-from-below list copied from the
   old root (``--old-root``, default ``D:\\scapes``) when the new one has none; its ``dem\\`` checked;
2. a wait for mask v4.2's training, started by hand (``train_masks.bat --tag v8_v4.2 --zoom-pairs``; 14:54: "don't add
   the mask training to the freeze, just delay the start so that it will be done by the time it runs the mask
   comparison"): until its release candidate ``Claude outputs\\mask_<checkpoint>\\<checkpoint>.safetensors`` is
   written, its log is quiet for ``--quiet-min`` minutes, or ``--max-wait-h`` hours have gone;
3. the mask A/B of v3, v4.1 and v4.2 on the held-out frames (v4.2's split, v4.1's the same set) and, for v3, the
   frames it did not train on (``--seen-split``) -> ``Claude outputs\\mask_ab_v3_v41_v42\\``; v4.2 is used for the
   freeze run when its exclusion IoU on those frames is higher than v3's, else v3 (the default) stays;
4. the freeze run: ``run_pipeline.py --root <root> --reset`` (+ ``--set mask_checkpoint=...`` when v4.2 won; no
   point cap - ``--cap`` adds ``ba_points_cap=1500``);
5. then, if it exists, ``<root>\\paper_studies\\run_paper_studies.py``;
6. last, the point-thinning study S2 on rockytop_example at ``--s2-root`` (default the old root, where its block is;
   14:57: "run that at the end of the freeze run after everything else important has completed") and its verdict
   (would a cap of 1500 do: frame centres < 1 cm p95, attitudes < 0.1 mdeg p95, per-bin focal lengths < 0.05 % max
   from the all-points solution) - for the paper and the next run; ``--skip-s2`` leaves it out.

Each step that fails is logged and the chain goes on.  Log: ``<root>\\mppp_pipeline\\freeze_chain_log.txt``; status:
``freeze_chain_status.json`` next to it (for reading from afar).

    scripts\\windows\\freeze_chain.bat                 everything
    scripts\\windows\\freeze_chain.bat --no-wait       no wait for the v4.2 training (the A/B with what there is)
    scripts\\windows\\freeze_chain.bat --no-freeze     steps 1-4 only
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
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CKPT_DIR = REPO / "checkpoints"
OUTPUTS = REPO / "Claude outputs"
V4 = CKPT_DIR / "convnext_tiny_s4_seg_20261030_v8.pt"
V41 = CKPT_DIR / "convnext_tiny_s4_seg_20261102_v8_v4.1.pt"
V42_TAG = "v8_v4.2"
MASK_DATA, MASK_SEEN = "D:\\masks\\masks_training_set_v8", "D:\\masks\\masks_training_set_v7"
ROOT_FILES = ("zcam_low_backlash_found.csv", "zcam_focus_from_below.csv")
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


def prepare_root(c: Chain) -> None:
    """The new root: the lists that live next to the blocks copied from the old root when missing; dem\\ checked."""
    import shutil
    old = Path(c.a.old_root)
    for name in ROOT_FILES:
        src, dst = old / name, c.root / name
        if dst.is_file():
            c.log(f"root: {dst} there")
        elif src.is_file():
            shutil.copy2(src, dst)
            c.log(f"root: {name} copied from {old}")
        else:
            c.log(f"root: no {name} in {old} or {c.root}")
    dem = c.root / "dem"
    c.log(f"root: {'DEMs in ' + str(dem) if dem.is_dir() else 'no dem folder in ' + str(c.root) + ' (maps without a DEM)'}")
    c.step("root", root=str(c.root), dem=dem.is_dir())


def v42_release() -> Optional[Path]:
    """The newest mask v4.2 release candidate (``Claude outputs\\mask_<checkpoint>\\<checkpoint>.safetensors``)."""
    cands = sorted(OUTPUTS.glob(f"mask_*_{V42_TAG}/*_{V42_TAG}.safetensors"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def v42_checkpoint() -> Optional[Path]:
    cands = sorted(CKPT_DIR.glob(f"*_{V42_TAG}.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def wait_for_v42(c: Chain) -> None:
    """Wait for the mask v4.2 training started by hand: its release candidate written = done."""
    t0 = time.time()
    c.step("wait_v42", state="waiting")
    while True:
        rel = v42_release()
        if rel is not None:
            c.log(f"wait_v42: {rel.name} written - the training is done")
            c.step("wait_v42", state="done", release=str(rel))
            return
        logs = sorted(CKPT_DIR.glob(f"*_{V42_TAG}_debug/log.csv"), key=lambda p: p.stat().st_mtime)
        age = (time.time() - logs[-1].stat().st_mtime) / 60 if logs else None
        if age is not None and age > c.a.quiet_min and (time.time() - t0) > 600:
            c.log(f"wait_v42: the training log quiet for {age:.0f} min - going on")
            c.step("wait_v42", state=f"log quiet {age:.0f} min")
            return
        if time.time() - t0 > c.a.max_wait_h * 3600:
            c.log(f"wait_v42: not done after {c.a.max_wait_h} h ({'no log yet' if logs is None or not logs else 'log ' + str(logs[-1])}) - going on")
            c.step("wait_v42", state="timeout")
            return
        time.sleep(120)


def mask_choice(c: Chain, ab_dir: Path, v42: Path) -> Optional[Path]:
    """v4.2's release candidate when its exclusion IoU on the frames no model trained on is higher than v3's."""
    import csv
    f = ab_dir / "mask_ab.csv"
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
    except OSError as e:
        c.log(f"mask choice: no {f} ({e}) - v3 stays")
        c.step("mask_choice", choice="v3", reason="no A/B")
        return None
    pick = {}
    for subset in ("not_seen", "all"):
        for r in rows:
            if r.get("subset") == subset and r.get("camera") == "all":
                pick.setdefault(subset, {})[r["model"]] = float(r["exclusion_IoU"])
    fair = pick.get("not_seen") or pick.get("all") or {}
    v3 = fair.get("mppp_mask_v3")
    v42_iou = next((v for k, v in fair.items() if k.endswith(V42_TAG)), None)
    zrec = {r["model"]: r["hardware_recall"] for r in rows if r.get("subset") == "all" and r.get("camera") == "Mastcam-Z"}
    c.log(f"mask choice: exclusion IoU on the frames no model trained on {json.dumps(fair)}; Mastcam-Z recall on all "
          f"held-out frames {json.dumps(zrec)}")
    if v3 is not None and v42_iou is not None and v42_iou > v3:
        c.log(f"mask choice: v4.2 ({v42_iou:.4f} > v3 {v3:.4f}) - the freeze run uses {v42.name}")
        c.step("mask_choice", choice="v4.2", v3=v3, v42=v42_iou, model=str(v42))
        return v42
    c.log(f"mask choice: v3 stays ({v3} vs v4.2 {v42_iou})")
    c.step("mask_choice", choice="v3", v3=v3, v42=v42_iou)
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
    except Exception as e:                                                    # noqa: BLE001
        c.log(f"S2 verdict: no comparison ({type(e).__name__}: {e})")
        c.step("s2", verdict="no comparison")
        return False
    got = {"centre_cm_p95": (cmp.get("centre_cm") or {}).get("p95"),
           "attitude_mdeg_p95": (cmp.get("attitude_mdeg") or {}).get("p95"),
           "focal_pct_max": (cmp.get("focal_pct") or {}).get("max")}
    ok = all(got[k] is not None and got[k] < lim for k, lim in S2_LIMITS.items())
    c.log(f"S2 verdict: {json.dumps(got)} against {json.dumps(S2_LIMITS)} -> {'a cap of 1500 would do' if ok else 'a cap of 1500 changes the solution'}")
    c.step("s2", verdict="cap 1500" if ok else "no cap", comparison=got)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:\\rover_blocks", help="the root of the freeze run's blocks")
    ap.add_argument("--old-root", default="D:\\scapes", help="where the found lists and the S2 study are")
    ap.add_argument("--s2-root", default=None, help="the S2 study's root (default --old-root)")
    ap.add_argument("--ab-python", default=sys.executable, help="a Python with a CUDA torch (the mask A/B)")
    ap.add_argument("--no-wait", action="store_true", help="no wait for the v4.2 training")
    ap.add_argument("--max-wait-h", type=float, default=14.0)
    ap.add_argument("--quiet-min", type=float, default=45.0)
    ap.add_argument("--skip-ab", action="store_true")
    ap.add_argument("--skip-s2", action="store_true", help="no S2 study at the end")
    ap.add_argument("--cap", action="store_true", help="the freeze run with ba_points_cap=1500")
    ap.add_argument("--no-freeze", action="store_true")
    a = ap.parse_args(argv)
    c = Chain(a)
    c.log(f"freeze chain started (root {c.root}, old root {a.old_root})")
    prepare_root(c)
    if not a.no_wait:
        wait_for_v42(c)
    mask = None
    v42_ck, v42_rel = v42_checkpoint(), v42_release()
    if not a.skip_ab and v42_ck is not None and v42_rel is not None:
        ab = OUTPUTS / "mask_ab_v3_v41_v42"
        c.run("mask_ab", [a.ab_python, HERE / "compare_mask_models.py", "--data", MASK_DATA,
                          "--split-ckpt", v42_ck, "--also-held-out", V41, "--models", "mppp_mask_v3", V41, v42_ck,
                          "--seen", MASK_SEEN, "--seen-split", "mppp_mask_v3", MASK_SEEN, "--out", ab], timeout_h=2.0)
        mask = mask_choice(c, ab, v42_rel)
    elif not a.skip_ab:
        c.log("mask_ab: no v4.2 checkpoint and release candidate - v3 stays")
        c.step("mask_choice", choice="v3", reason="no v4.2")
    s2_root = Path(a.s2_root or a.old_root)
    if not a.no_freeze:
        cmd = [sys.executable, REPO / "studies" / "calibration" / "run_pipeline.py", "--root", c.root, "--reset"]
        if a.cap:
            cmd += ["--set", "ba_points_cap=1500"]
        if mask is not None:
            cmd += ["--set", f"mask_checkpoint={mask}"]
        c.run("freeze", cmd, timeout_h=24 * 7)
        extra = c.root / "paper_studies" / "run_paper_studies.py"
        if extra.is_file():
            c.run("paper_studies", [sys.executable, extra, "--root", c.root], timeout_h=72)
        else:
            c.log(f"paper_studies: {extra} not there - nothing more")
    else:
        c.log("--no-freeze: no freeze run")
    if not a.skip_s2:
        c.run("s2", [sys.executable, HERE / "tie_point_studies.py", "S2", "--site", S2_SITE, "--variants", "all",
                     "cap1500", "--root", s2_root], timeout_h=8.0)
        s2_verdict(c, s2_root)
    c.log("freeze chain done")
    c.step("chain", state="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
