"""
The whole v0p74 calibration and alignment run, unattended, in one command (several days):

    python scripts\\run_pipeline.py                  (run_pipeline.bat)
    python scripts\\run_pipeline.py --status         what is done, what is running
    python scripts\\run_pipeline.py --dry-run        the plan, nothing run

Stages (each logged in ``<root>\\mppp_pipeline\\<stage>.log``, the pipeline in ``pipeline_log.txt``, the state in
``state.json``; started again, the pipeline continues after the last finished stage):

 1. ``wait``             - waits while another run_sites / process_sites batch runs on ROOT (at most --wait-h hours)
 2. ``prepare``          - prepare_next_run.py: the v0p72 consensus in the v0p73 form must be in use (promoted if
                           not) and the form OK - otherwise the pipeline stops here
 3. ``navcam_consensus`` - the Navcam consensus in the v0p73 form (fx, fy per eye and NR cx against temperature, the
                           rig yaw sol rate, early-mission offsets, the rig translation fitted on the large blocks and
                           applied, 2 rounds of thermal profiles) over the Navcam-only blocks on disk that pass
                           screening, checked and promoted if every check passes; then the rig residuals per block
 4. ``zcam_start``       - the current Navcam + Mastcam-Z blocks (problematic as they are): their low-backlash groups
                           into the found list, a start focus model (fx, fy lines, regular-state ratio) promoted if
                           its checks pass
 5. ``zcam_runs``        - run_sites.py --group zcam34_consensus --zcam --set ZCAM_ZOOMS=[34] (every site: the camera
                           models are in the run key), failed sites tried once more
 6. ``navcam_runs``      - run_sites.py --group navcam_consensus (``--no-navcam-runs`` skips it), failed ones once more
 7. ``zcam_calibrate``   - grade the v0p74 N+Z blocks, the best --n (2-3) give the Mastcam-Z 34 mm calibration
                           (lines per eye and focus state for fx, fy, cx, cy, k1, k2; lens-temperature terms where
                           significant; rig rotation and baseline) - checked, promoted, report
 8. ``zcam_final_runs``  - only if the calibration was promoted: every N+Z block again from it
 9. ``pointclouds``      - a PLY per finished block
10. ``report``           - ``<root>\\mppp_pipeline\\REPORT.md``

A stage that fails is logged and the pipeline goes on where that makes sense (only ``prepare`` stops it). Exit code 0
when every stage ran (whatever it found), 1 when the pipeline stopped.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
STAGES = ("wait", "prepare", "navcam_consensus", "zcam_start", "zcam_runs", "navcam_runs", "zcam_calibrate",
          "zcam_final_runs", "pointclouds", "report")
ZCAM_GROUP, NAVCAM_GROUP = "zcam34_consensus", "navcam_consensus"
ZCAM_SET = ["--set", "ZCAM_ZOOMS=[34]"]
EARLY_START = ("0.157", "0.243", "0.139", "-0.454")          # mean of the v0p72_2 and 20261003 consensus' early offsets
EARLY_ROUNDS = "3"            # a round is ~35 min; the cx direction is flat (the 20261003 run needed 3 from a poor start)


class Pipeline:
    def __init__(self, a):
        from mppp.runner import Log
        self.a = a
        self.root = Path(a.root)
        self.dir = self.root / "mppp_pipeline"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.dir / ("dry_run_log.txt" if a.dry_run else "pipeline_log.txt"))
        self.state_file = self.dir / "state.json"
        self.state = json.loads(self.state_file.read_text(encoding="utf-8")) if self.state_file.is_file() else {}

    # ---------------------------------------------------------------- bookkeeping
    def save(self):
        if self.a.dry_run:                        # a dry run leaves no state behind (a real run would skip "done" stages)
            return
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.state_file)

    def mark(self, stage, status, **kw):
        e = self.state.setdefault(stage, {})
        e.update(status=status, **kw)
        e.setdefault("started", datetime.datetime.now().isoformat(timespec="seconds"))
        if status in ("done", "failed", "skipped"):
            e["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
        self.save()

    def run(self, stage, cmd, timeout_h=None):
        """A command with its output in <stage>.log (appended); returns its exit code."""
        lf = self.dir / f"{stage}.log"
        self.log(f"  {stage}: " + " ".join(str(c) for c in cmd))
        if self.a.dry_run:
            return 0
        with open(lf, "a", encoding="utf-8") as fh:
            fh.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(map(str, cmd))}\n")
            fh.flush()
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            try:
                rc = p.wait(timeout=timeout_h * 3600 if timeout_h else None)
            except subprocess.TimeoutExpired:                 # the whole tree (a hung notebook kernel included)
                from mppp.runner import kill_tree
                kill_tree(p.pid)
                p.wait(timeout=60)
                fh.write(f"\nTIMEOUT after {timeout_h} h: stopped\n")
                rc = 124
        self.log(f"  {stage}: exit {rc} (log {lf})")
        return rc

    # ---------------------------------------------------------------- stages
    def s_wait(self):
        from mppp.runner import BATCH_FILE, read_status
        t0 = time.time()
        while True:
            cur = read_status(self.root / BATCH_FILE)
            if not (cur and cur.get("alive")):
                return "done", "no batch running"
            if time.time() - t0 > self.a.wait_h * 3600:
                return "failed", f"a batch ({cur.get('kind')}, pid {cur.get('pid')}) still runs after {self.a.wait_h} h"
            self.log(f"  wait: {cur.get('kind')} running (site {cur.get('site')}); checking again in 10 min")
            if self.a.dry_run:
                return "done", "dry run"
            time.sleep(600)

    def s_prepare(self):
        from mppp.sfm.cmods_form import check_form
        # cameras already in the frozen form (e.g. a consensus this pipeline promoted, then restarted with --reset):
        # only checked - an older candidate is never promoted over them
        keep = check_form()["ok"]
        cmd = [PY, ROOT / "scripts" / "prepare_next_run.py", "--root", self.root, "--groups", NAVCAM_GROUP, ZCAM_GROUP] \
            + (["--keep"] if keep else [])
        rc = self.run("prepare", cmd)
        if keep:
            return "done", ("cameras in use: the v0p72 consensus in the v0p73 form" if rc == 0 else
                            "cameras in use are in the v0p73 form (not the v0p72 candidate: kept)")
        if rc == 0:
            return "done", "the v0p72 consensus in the v0p73 form promoted and checked"
        return "failed", f"prepare_next_run.py exit {rc}: the camera models are not right - stopping"

    def navcam_blocks(self):
        """{label: WORK folder} of the Navcam-only blocks on disk for the consensus: finished (run_done), one folder
        per site (the site's own WORK folder when it has one), screening not FAIL."""
        from mppp.sfm.health import screen_blocks
        from mppp.sfm.sites import discover_scapes, load_site_table, parse_work_folder, work_folder
        table = load_site_table()
        found = discover_scapes(self.root, include_zcam=False, verbose=False)
        out = {}
        for label, w in found.items():
            site, zc = parse_work_folder(w)
            if zc or not (w / "colmap" / "run_done.json").is_file():
                continue
            if site in table["sites"]:
                own = work_folder(self.root, site, False, table["sites"][site]["sols"])
                if own != w and (own / "colmap" / "run_done.json").is_file():
                    continue                                   # a stale folder of a site that has its own
            out[label] = w
        scr = screen_blocks({k: str(v) for k, v in out.items()}, verbose=False)
        bad = sorted(k for k, r in scr.items() if r.get("verdict") == "fail")
        return {k: v for k, v in out.items() if k not in bad}, scr, bad

    def s_navcam_consensus(self):
        from mppp import VERSION_TAG
        blocks, scr, bad = self.navcam_blocks()
        if len(blocks) < 8:
            return "failed", f"only {len(blocks)} Navcam blocks pass screening (left out: {', '.join(bad)})"
        out = self.root / "camera_analysis" / f"navcal_{VERSION_TAG}_{datetime.date.today():%Y%m%d}_pipeline"
        n = 1
        base = out
        while out.exists() and (out / "navcam_joint" / "consensus.json").is_file():
            n += 1
            out = base.with_name(f"{base.name}_{n}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "scapes.json").write_text(json.dumps({k: str(v) for k, v in blocks.items()}, indent=1), encoding="utf-8")
        (out / "block_screening.json").write_text(json.dumps(scr, indent=1, default=str), encoding="utf-8")
        self.log(f"  navcam_consensus: {len(blocks)} blocks -> {out}" + (f" (screening FAIL left out: {', '.join(bad)})" if bad else ""))
        self.state.setdefault("navcam_consensus", {})["study"] = str(out)
        self.save()
        rc = self.run("navcam_consensus", [PY, ROOT / "scripts" / "navcam_calibration_study.py", "consensus", out,
                                           out / "scapes.json", "--points", "8000", "--translation-min-span", "30",
                                           "--apply-translation", "--refit-thermal", "--thermal-rounds", "2",
                                           "--form", "v0p73", "--early-mission-sol", "380", "--early-rounds", EARLY_ROUNDS,
                                           "--early-start", *EARLY_START, "--promote-if-ok",
                                           "--promote-note", f"v0p74 pipeline consensus of {len(blocks)} blocks (v0p73 form)"],
                      timeout_h=self.a.stage_timeout_h)
        chk = out / "promotion_check.json"
        rep = json.loads(chk.read_text(encoding="utf-8")) if chk.is_file() else {}
        promoted = bool(rep.get("ok")) and rc == 0
        # the rig residuals with the candidate (or the cameras in use) - best effort
        cams = out / "navcam_joint" if (out / "navcam_joint" / "M2020_N_rig.json").is_file() else None
        code = ("import json,sys; sys.path.insert(0,r'%s'); import matplotlib; matplotlib.use('Agg');"
                "from mppp.sfm import navcal_consensus as N;"
                "b=json.load(open(r'%s'));"
                "r=N.rig_residuals(b, cameras_dir=%s, points=8000);"
                "json.dump(r, open(r'%s','w'), indent=1, default=float);"
                "N.plot_rig_residuals(r, r'%s')") % (ROOT / "src", out / "scapes.json",
                                                    f"r'{cams}'" if cams else "None", out / "rig_residuals.json",
                                                    out / "navcam_rig_residuals.png")
        self.run("navcam_consensus", [PY, "-c", code], timeout_h=self.a.stage_timeout_h)
        if rc != 0 and not rep:
            return "failed", f"the consensus failed (exit {rc}); the cameras in use stay"
        return "done", ("promoted" if promoted else "not promoted (a check failed; the cameras in use stay): "
                        + ", ".join(c["check"] for c in rep.get("checks", []) if not c.get("ok")))

    def s_zcam_start(self):
        rc = self.run("zcam_start", [PY, ROOT / "scripts" / "zcam_calibration.py", "start-update", "--root", self.root,
                                     "--groups", ZCAM_GROUP, "--promote"], timeout_h=6)
        return ("done", {0: "start model promoted, found list updated", 1: "start model not promoted (a check failed); "
                         "found list updated"}.get(rc, f"error (exit {rc}); the focus model in use stays"))

    def _sites(self, stage, group, zcam):
        cmd = [PY, ROOT / "scripts" / "run_sites.py", "--root", self.root, "--group", group]
        if zcam:
            cmd += ["--zcam", *ZCAM_SET]
        rc = self.run(stage, cmd, timeout_h=self.a.sites_timeout_h)
        for _ in range(3):
            if rc == 3 and not self.a.dry_run:                 # another batch started meanwhile: wait for it
                st, note = self.s_wait()
                if st != "done":
                    return "failed", note
                rc = self.run(stage, cmd, timeout_h=self.a.sites_timeout_h)
            else:
                break
        if rc == 4:
            return "failed", "run_sites refused: the camera models are not in the v0p73 form"
        if rc == 1 and not self.a.dry_run:                     # a failed site once more (finished ones are skipped)
            self.log(f"  {stage}: some sites failed - once more")
            rc = self.run(stage, cmd, timeout_h=self.a.sites_timeout_h)
        return ("done", "every site finished") if rc == 0 else ("done", f"finished with failed sites (exit {rc}; see run_sites_log.txt)")

    def s_zcam_runs(self):
        return self._sites("zcam_runs", ZCAM_GROUP, True)

    def s_navcam_runs(self):
        if self.a.no_navcam_runs:
            return "skipped", "--no-navcam-runs"
        return self._sites("navcam_runs", NAVCAM_GROUP, False)

    def s_zcam_calibrate(self):
        rc = self.run("zcam_calibrate", [PY, ROOT / "scripts" / "zcam_calibration.py", "calibrate", "--root", self.root,
                                         "--groups", ZCAM_GROUP, "--n", str(self.a.n), "--promote"], timeout_h=6)
        self.state.setdefault("zcam_calibrate", {})["promoted"] = rc == 0 and not self.a.dry_run
        self.save()
        return "done", {0: "calibration promoted", 1: "calibration not promoted (too few passing blocks or a check "
                        "failed; see camera_analysis/zcam34_calibration_*)"}.get(rc, f"error (exit {rc})")

    def s_zcam_final_runs(self):
        if not (self.state.get("zcam_calibrate") or {}).get("promoted"):
            return "skipped", "no new calibration promoted"
        return self._sites("zcam_final_runs", ZCAM_GROUP, True)

    def s_pointclouds(self):
        rc = self.run("pointclouds", [PY, ROOT / "scripts" / "zcam_calibration.py", "pointclouds", "--root", self.root,
                                      "--groups", ZCAM_GROUP, NAVCAM_GROUP], timeout_h=6)
        return ("done", "point clouds written") if rc == 0 else ("done", f"exit {rc}")

    def s_report(self):
        from mppp.sfm.cmods_form import check_form, describe
        lines = [f"# MPPP {__import__('mppp').__version__} pipeline report ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
                 "## Stages", "", "| stage | status | note | started | ended |", "|---|---|---|---|---|"]
        for s in STAGES:
            e = self.state.get(s) or {}
            lines.append(f"| {s} | {e.get('status', '-')} | {e.get('note', '')} | {e.get('started', '')} | {e.get('ended', '')} |")
        lines += ["", "## Camera models in use", ""] + [f"    {ln}" for ln in describe()]
        f = check_form()
        lines += ["", f"form v0p73: {'OK' if f['ok'] else 'NOT OK: ' + '; '.join(f['errors'])}", ""]
        st = (self.state.get("navcam_consensus") or {}).get("study")
        if st:
            lines += ["## Navcam consensus", "", f"study: `{st}`"]
            for name in ("promotion_check.json", "rig_residuals.json"):
                p = Path(st) / name
                if p.is_file():
                    d = json.loads(p.read_text(encoding="utf-8"))
                    if name == "promotion_check.json":
                        lines += [f"- {'ok' if c['ok'] else 'FAIL'} {c['check']}: {c['value']} ({c.get('note', '')})"
                                  for c in d.get("checks", [])]
                    else:
                        for ang, ft in (d.get("fits") or {}).items():
                            vt, vs = ft.get("vs_T") or {}, ft.get("vs_sol") or {}
                            if vt.get("slope") is not None:
                                lines.append(f"- rig {ang} residual: vs block temperature {vt['slope']:+.3f} +- {vt['se']:.3f} "
                                             f"mdeg/degC (z {vt['z']:+.1f}); vs sol {vs['slope']:+.2f} +- {vs['se']:.2f} "
                                             f"mdeg/1000 sol (z {vs['z']:+.1f})")
            lines.append("")
        # every block's last run
        lines += ["## Blocks", "", "| block | mppp | verdict | registered | rms px | camera models |", "|---|---|---|---|---|---|"]
        for w in sorted(self.root.glob("mars2020_sol_*_colmap*")):
            rd = w / "colmap" / "run_done.json"
            if rd.is_file():
                d = json.loads(rd.read_text(encoding="utf-8"))
                lines.append(f"| {w.name} | {d.get('mppp')} | {d.get('verdict')} | {d.get('registered_images')}/{d.get('images')} | "
                             f"{(d.get('residual_rms_native_px') or float('nan')):.3f} | "
                             f"{(d.get('camera_models') or {}).get('fingerprint', '-')} |")
        cal = sorted((self.root / "camera_analysis").glob("zcam34_calibration_*"))
        if cal:
            lines += ["", "## Mastcam-Z 34 mm calibration", "", f"latest: `{cal[-1]}` (report.md there)"]
            rp = cal[-1] / "report.md"
            if rp.is_file():
                lines += ["", rp.read_text(encoding="utf-8")]
        idx = self.root / "pointclouds" / "index.json"
        if idx.is_file():
            pcs = json.loads(idx.read_text(encoding="utf-8"))
            lines += ["", "## Point clouds", ""] + [f"- {p['block']}: {p['points']} points, `{p['ply']}` ({p.get('verdict')})"
                                                    for p in pcs]
        (self.dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return "done", str(self.dir / "REPORT.md")

    # ---------------------------------------------------------------- driver
    def go(self) -> int:
        from mppp.runner import RunStatus
        lock = self.dir / "pipeline_status.json"
        from mppp.runner import read_status
        cur = read_status(lock)
        if cur and cur.get("alive") and not self.a.dry_run:
            self.log(f"not started: the pipeline is already running (pid {cur.get('pid')})")
            return 1
        if not self.a.dry_run:
            from mppp.runner import disable_quickedit
            disable_quickedit()                       # a click in the console window must not freeze days of work
        status = None if self.a.dry_run else RunStatus(lock, kind="pipeline", argv=sys.argv).start()
        self.log(f"MPPP pipeline {__import__('mppp').__version__} on {self.root}" + (" (dry run)" if self.a.dry_run else ""))
        order = list(STAGES)
        if self.a.start:
            for s in order[:order.index(self.a.start)]:
                if (self.state.get(s) or {}).get("status") != "done":
                    self.mark(s, "skipped", note=f"--from {self.a.start}")
        rc = 0
        for s in order:
            if (self.state.get(s) or {}).get("status") in ("done", "skipped") and s not in (self.a.redo or []) \
                    and s != "report":
                self.log(f"{s}: {self.state[s]['status']} before - not run again")
                continue
            self.log(f"{s}: start")
            if status:
                status.update(state="running", stage=s)
            self.mark(s, "running", started=datetime.datetime.now().isoformat(timespec="seconds"), note="")
            try:
                st, note = getattr(self, "s_" + s)()
            except Exception as e:                                         # noqa: BLE001
                import traceback
                self.log(traceback.format_exc())
                st, note = "failed", f"{type(e).__name__}: {e}"
            self.mark(s, st, note=note)
            self.log(f"{s}: {st} - {note}")
            if st == "failed" and s in ("prepare",):
                self.log("pipeline stopped: fix the above and start it again (it continues from here)")
                rc = 1
                break
        if status:
            status.stop("finished" if rc == 0 else "failed")
        self.log("pipeline " + ("finished" if rc == 0 else "stopped") + f"; report: {self.dir / 'REPORT.md'}")
        return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="D:/scapes/colmap")
    ap.add_argument("--n", type=int, default=3, help="the number of graded N+Z blocks for the calibration")
    ap.add_argument("--no-navcam-runs", action="store_true", help="skip the Navcam-only reruns")
    ap.add_argument("--wait-h", type=float, default=36.0, help="how long to wait for a running batch")
    ap.add_argument("--stage-timeout-h", type=float, default=30.0, help="the consensus stage's time limit")
    ap.add_argument("--sites-timeout-h", type=float, default=72.0,
                    help="time limit of one run_sites call (a hung alignment is stopped with its kernel)")
    ap.add_argument("--from", dest="start", choices=STAGES, help="start at this stage (the ones before: skipped)")
    ap.add_argument("--redo", nargs="*", choices=STAGES, help="run these stages again even if done")
    ap.add_argument("--reset", action="store_true", help="forget the state (every stage runs again)")
    ap.add_argument("--status", action="store_true", help="show the state and exit")
    ap.add_argument("--dry-run", action="store_true", help="show the commands, run nothing")
    a = ap.parse_args(argv)
    p = Pipeline(a)
    if a.status:
        from mppp.runner import read_status
        cur = read_status(p.dir / "pipeline_status.json")
        print(f"pipeline: {(cur or {}).get('state', 'not started')}" + (f", stage {cur.get('stage')}" if cur else ""))
        for s in STAGES:
            e = p.state.get(s) or {}
            print(f"  {s:18s} {e.get('status', '-'):8s} {e.get('note', '')}")
        return 0
    if a.reset and not a.dry_run:
        p.state = {}
        p.save()
    return p.go()


if __name__ == "__main__":
    sys.exit(main())
