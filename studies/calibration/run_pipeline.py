"""
The whole calibration and alignment run, unattended, in one command (several days).  v0p80: the camera-model recipe
of 14 Oct 2026 - Navcam first and iterated (blocks -> joint -> promote -> blocks -> joint), v0p82: then the waypoint
check and the Navcam baseline (promoted if significant, confirmed on the blocks, the joint refitted), then Mastcam-Z
with the promoted Navcam:

    python studies\\calibration\\run_pipeline.py                  (run_pipeline.bat)
    python studies\\calibration\\run_pipeline.py --status         what is done, what is running
    python studies\\calibration\\run_pipeline.py --dry-run        the plan, nothing run

Stages (each logged in ``<root>\\mppp_pipeline\\<stage>.log``, the pipeline in ``pipeline_log.txt``, the state in
``state.json``; started again, the pipeline continues after the last finished stage):

 1. ``wait``             - waits while another run_sites / process_sites batch runs on ROOT (at most --wait-h hours)
 2. ``prepare``          - prepare_next_run.py: the v0p72 consensus in the v0p73 form must be in use (promoted if
                           not) and the form OK - otherwise the pipeline stops here
 3. ``navcam_consensus`` - pass 1 of the Navcam consensus in the v0p80 form (the v0p73 NL-reference form - fx, fy per
                           eye and NR cx against temperature, the rig yaw sol rate, early-mission offsets, the rig
                           translation held (v0p82), 2 rounds of thermal profiles - with k4 free and k1 against temperature from the
                           blocks' temperature bins) over the Navcam-only blocks of the navcam_consensus group that pass
                           screening, leave-one-block-out, checked and promoted if every check passes; rig residuals
 4. ``navcam_runs``      - run_sites.py --group navcam_consensus with the promoted models and the v0p80 block recipe
                           (round 1 held, principal point held, fx fy k1 per temperature bin, rig pitch/roll per
                           epoch); ``--no-navcam-runs`` skips it; failed ones once more
 5. ``navcam_consensus_2`` - pass 2 of the consensus on those blocks; the change from pass 1 says whether the iteration
                           has converged (below 20 ppm in f, 0.05 px pp, 0.02 px corner distortion, 1 mdeg rig);
                           promoted if every check passes
 6. ``waypoint_check``   - v0p82: the waypoint localisation check over the Navcam blocks (navcam_consensus and
                           navcam_baseline groups): each block's stations against their waypoints after a robust
                           similarity; flagged (> 1 m, or > 0.5 m and > 3x the block's scatter) ->
                           camera_analysis/waypoint_flags_<date>
 7. ``baseline_study``   - v0p82: the Navcam baseline from the navcam_baseline sites with > 4 waypoints over > 30 m:
                           cameras held, rig translation free, flagged waypoints without prior, priors at 0.5 m;
                           combined (between-block term), with the block scale against the waypoints now / free /
                           predicted and an east-north isotropy test; promoted if > 3 sigma, the blocks agree, the scale
                           agreement improves and the scales are isotropic - and only when pass 2 converged (or with
                           ``--baseline-study``); the joint holds the translation in every pass (no --apply-translation)
 8. ``baseline_confirm`` - v0p82: after a promotion the same blocks with the new baseline held; if the scale agreement is
                           not better, the previous rig is promoted back
 9. ``navcam_consensus_3`` - v0p82: with a confirmed new baseline, the joint once more (the baseline held)
10. ``zcam_start``       - v0p82: the found list cleaned (clean-found), then the current Navcam + Mastcam-Z blocks: their
                           low-backlash groups into the found list, a start focus model written as a candidate - the
                           promoted v0p75 start model stays (``--zcam-start-promote`` promotes the candidate)
11. ``zcam_runs``        - run_sites.py --group zcam34_consensus --zcam --set ZCAM_ZOOMS=[34] (every site: the camera
                           models are in the run key), failed sites tried once more
12. ``zcam_calibrate``   - grade the N+Z blocks, every passing one gives the Mastcam-Z 34 mm calibration (lines per eye
                           and focus state; rig rotation) - checked, promoted, report
13. ``zcam_final_runs``  - only if the calibration was promoted: every N+Z block again from it
14. ``pointclouds``      - a PLY per finished block
15. ``report``           - ``<root>\\mppp_pipeline\\REPORT.md``

v0p82 alignments (every run): images whose mask keeps less than 1/3 of the image are left out
(MIN_INCLUDED_FRACTION), and the block is moved onto its position priors after each adjustment (PRIOR_GAUGE_RECENTRE).

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

HERE = Path(__file__).resolve().parent          # the studies/calibration folder (its modules import as siblings)
ROOT = HERE.parents[1]                          # the repository root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PY = sys.executable
STAGES = ("wait", "prepare", "navcam_consensus", "navcam_runs", "navcam_consensus_2", "waypoint_check",
          "baseline_study", "baseline_confirm", "navcam_consensus_3", "zcam_start", "zcam_runs", "zcam_calibrate",
          "zcam_final_runs", "pointclouds", "report")      # v0p80: Navcam first; v0p82: the baseline before Mastcam-Z
CONSENSUS_FORM = "v0p80"
ZCAM_GROUP, NAVCAM_GROUP = "zcam34_consensus", "navcam_consensus"     # --zcam-group overrides the first (v0p75)
BASELINE_GROUP = "navcam_baseline"                                     # v0p82
ZCAM_SET = ["--set", "ZCAM_ZOOMS=[34]"]
EARLY_START = ("0.197", "0.254", "0.324", "-0.572")          # v0p85: the 18 Oct pass-2 offsets (z 5.8-13.5)
EARLY_ROUNDS = "1"            # v0p85: one round around the pass-2 optimum (was 3 from a poorer start)
EARLY_ROUNDS_V0P82 = "3"            # a round is ~35 min; the cx direction is flat (the 20261003 run needed 3 from a poor start)


class Pipeline:
    def __init__(self, a):
        from mppp.runner import Log
        self.a = a
        self.zcam_group = getattr(self.a, "zcam_group", None) or ZCAM_GROUP    # v0p75: a smaller group for iterating
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
        from mppp.calib.models import check_form
        # cameras already in the frozen form (e.g. a consensus this pipeline promoted, then restarted with --reset):
        # only checked - an older candidate is never promoted over them
        keep = check_form()["ok"]
        cmd = [PY, HERE / "prepare_next_run.py", "--root", self.root, "--groups", NAVCAM_GROUP, self.zcam_group] \
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
        per site (the site's own WORK folder when it has one), screening not FAIL.  v0p75: the sites of the
        ``navcam_consensus`` group only (``--all-navcam-blocks`` takes every Navcam block on disk, as before)."""
        from mppp.sfm.health import screen_blocks
        from mppp.sfm.sites import discover_scapes, load_site_table, parse_work_folder, site_group, work_folder
        table = load_site_table()
        found = discover_scapes(self.root, include_zcam=False, verbose=False)
        members = None
        if not getattr(self.a, "all_navcam_blocks", False):
            try:
                members = set(site_group(NAVCAM_GROUP))
            except KeyError:                                   # a site table without the group: every block
                members = None
        out = {}
        skipped = []
        for label, w in found.items():
            site, zc = parse_work_folder(w)
            if zc or not (w / "colmap" / "run_done.json").is_file():
                continue
            if members is not None and site not in members:
                skipped.append(site)
                continue
            if site in table["sites"]:
                own = work_folder(self.root, site, False, table["sites"][site]["sols"])
                if own != w and (own / "colmap" / "run_done.json").is_file():
                    continue                                   # a stale folder of a site that has its own
            out[label] = w
        if skipped:
            self.log(f"  navcam_consensus: blocks on disk not in the {NAVCAM_GROUP} group, left out: {', '.join(sorted(set(skipped)))}")
        scr = screen_blocks({k: str(v) for k, v in out.items()}, verbose=False)
        # v0p74.5: a block matched with COLMAP guided matching (epipolar-consistent false matches) is a WARN for a
        # single alignment but must not shape the shared cameras: left out of the consensus like a FAIL
        bad = sorted(k for k, r in scr.items() if r.get("verdict") == "fail" or r.get("guided_matching"))
        return {k: v for k, v in out.items() if k not in bad}, scr, bad

    def s_navcam_consensus(self, stage="navcam_consensus", previous=None):
        from mppp import VERSION_TAG
        blocks, scr, bad = self.navcam_blocks()
        if len(blocks) < 8:
            return "failed", f"only {len(blocks)} Navcam blocks pass screening (left out: {', '.join(bad)})"
        tag = {"navcam_consensus": "pipeline", "navcam_consensus_2": "pipeline_pass2"}.get(stage, "pipeline_pass3")
        out = self.root / "camera_analysis" / f"navcal_{VERSION_TAG}_{datetime.date.today():%Y%m%d}_{tag}"
        n = 1
        base = out
        while out.exists() and (out / "navcam_joint" / "consensus.json").is_file():
            n += 1
            out = base.with_name(f"{base.name}_{n}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "scapes.json").write_text(json.dumps({k: str(v) for k, v in blocks.items()}, indent=1), encoding="utf-8")
        (out / "block_screening.json").write_text(json.dumps(scr, indent=1, default=str), encoding="utf-8")
        self.log(f"  {stage}: {len(blocks)} blocks -> {out}" + (f" (screening FAIL left out: {', '.join(bad)})" if bad else ""))
        self.state.setdefault(stage, {})["study"] = str(out)
        self.save()
        # v0p80: the recipe's joint - k4 free, k1(T) from the bins, leave-one-block-out as a promotion gate
        # v0p82: the joint holds the rig translation (the baseline stage carries it); the large blocks' translation
        # with the cameras held is still fitted and reported (--translation-min-span), not applied
        cmd = [PY, HERE / "navcam_calibration_study.py", "consensus", out,
               out / "scapes.json", "--points", "8000", "--translation-min-span", "30",
               "--refit-thermal",
               "--thermal-rounds", "2",
               "--form", CONSENSUS_FORM, "--k1-from-bins", "--leave-one-out",
               "--early-mission-sol", "380", "--early-rounds", EARLY_ROUNDS,
               "--early-start", *EARLY_START, "--promote-if-ok",
               "--promote-note", f"{VERSION_TAG} pipeline consensus {tag.replace('pipeline_', '') + ' ' if previous else ''}of "
                                 f"{len(blocks)} blocks ({CONSENSUS_FORM} form)"]
        if previous:
            cmd += ["--previous", previous]
        rc = self.run(stage, cmd, timeout_h=self.a.stage_timeout_h)
        chk = out / "promotion_check.json"
        rep = json.loads(chk.read_text(encoding="utf-8")) if chk.is_file() else {}
        promoted = bool(rep.get("ok")) and rc == 0
        # the rig residuals with the candidate (or the cameras in use) - best effort
        cams = out / "navcam_joint" if (out / "navcam_joint" / "M2020_N_rig.json").is_file() else None
        code = ("import json,sys; sys.path.insert(0,r'%s'); import matplotlib; matplotlib.use('Agg');"
                "import navcal_consensus as N;"
                "b=json.load(open(r'%s'));"
                "r=N.rig_residuals(b, cameras_dir=%s, points=8000);"
                "json.dump(r, open(r'%s','w'), indent=1, default=float);"
                "N.plot_rig_residuals(r, r'%s')") % (HERE, out / "scapes.json",
                                                    f"r'{cams}'" if cams else "None", out / "rig_residuals.json",
                                                    out / "navcam_rig_residuals.png")
        self.run(stage, [PY, "-c", code], timeout_h=self.a.stage_timeout_h)
        if rc != 0 and not rep:
            return "failed", f"the consensus failed (exit {rc}); the cameras in use stay"
        conv = ""
        pc = out / "pass_change.json"
        if pc.is_file():                                   # v0p80: pass 2 - has the iteration converged?
            ch = json.loads(pc.read_text(encoding="utf-8"))
            self.state.setdefault(stage, {})["converged"] = bool(ch.get("converged"))
            self.save()
            w = ch.get("worst") or {}
            conv = (f"; change from pass 1: f {w.get('f_ppm', 0):.1f} ppm, pp {w.get('pp_px', 0):.3f} px, corners "
                    f"{w.get('corner_px', 0):.3f} px, rig {w.get('rig_mdeg', 0):.2f} mdeg -> "
                    + ("converged" if ch.get("converged") else "NOT converged (another navcam_runs + consensus pass)"))
        return "done", ("promoted" if promoted else "not promoted (a check failed; the cameras in use stay): "
                        + ", ".join(c["check"] for c in rep.get("checks", []) if not c.get("ok"))) + conv

    def s_navcam_consensus_2(self):
        """v0p80: pass 2 of the joint over the blocks rerun with pass 1's models."""
        if (self.state.get("navcam_runs") or {}).get("status") != "done":
            return "skipped", "no navcam_runs this time (pass 2 needs the blocks rerun with pass 1's models)"
        st = (self.state.get("navcam_consensus") or {}).get("study")
        prev = str(Path(st) / "navcam_joint") if st and (Path(st) / "navcam_joint" / "M2020_N_rig.json").is_file() else None
        if prev is None:
            from mppp.paths import cmods_dir
            prev = str(cmods_dir())
        return self.s_navcam_consensus("navcam_consensus_2", previous=prev)

    def s_waypoint_check(self):
        """v0p82: the waypoint localisation check over the Navcam blocks (navcam_consensus and navcam_baseline groups,
        the production runs): flagged stations go into the baseline stage without their position prior."""
        out = self.root / "camera_analysis" / f"waypoint_flags_{datetime.date.today():%Y%m%d}"
        rc = self.run("waypoint_check", [PY, ROOT / "scripts" / "waypoint_check.py", "--root", self.root, "--groups",
                                         NAVCAM_GROUP, BASELINE_GROUP, "--out", out], timeout_h=2)
        js = out / "waypoint_flags.json"
        if rc != 0 or not js.is_file():
            return "done", f"no flags (exit {rc}); the baseline stage keeps every prior"
        self.state.setdefault("waypoint_check", {})["flags"] = str(js)
        self.save()
        fl = json.loads(js.read_text(encoding="utf-8")).get("flagged") or []
        return "done", f"{len(fl)} flagged stations: {', '.join(fl) or 'none'} ({out / 'waypoint_flags.md'})"

    def s_baseline_study(self):
        """v0p80/v0p82: the baseline from the large blocks (the navcam_baseline sites with > 4 waypoints over > 30 m),
        flagged waypoints without prior; promoted if significant and the block scale agreement improves - only once
        the Navcam consensus has converged (pass 2) or with --baseline-study; otherwise studied, not promoted."""
        if getattr(self.a, "no_baseline_study", False):
            return "skipped", "--no-baseline-study"
        conv = bool((self.state.get("navcam_consensus_2") or {}).get("converged"))
        promote = conv or getattr(self.a, "baseline_study", False)
        out = self.root / "camera_analysis" / f"navcam_baseline_{datetime.date.today():%Y%m%d}"
        self.state.setdefault("baseline_study", {})["study"] = str(out)
        self.save()
        cmd = [PY, HERE / "navcam_baseline_study.py", "all", "--root", self.root, "--out", out,
               "--write-rig", out / "candidate_rig"]
        fl = (self.state.get("waypoint_check") or {}).get("flags")
        if fl:
            cmd += ["--flags", fl]
        if promote:
            cmd.append("--promote")
        rc = self.run("baseline_study", cmd, timeout_h=self.a.sites_timeout_h)
        js = out / "baseline_study.json"
        if not js.is_file():
            return "done", f"no result (exit {rc})"
        rep = json.loads(js.read_text(encoding="utf-8"))
        verdict = rep["combined"].get("verdict")
        promoted = bool((rep.get("promoted") or {}).get("returncode") == 0)
        self.state["baseline_study"]["promoted"] = promoted
        self.save()
        return "done", (f"{verdict}; " + ("PROMOTED" if promoted else
                                          "not promoted" + ("" if promote else " (the consensus has not converged)")))

    def s_baseline_confirm(self):
        """v0p82: the study's blocks again with the promoted baseline held; reverted if the scale agreement with the
        waypoints is not better."""
        st = self.state.get("baseline_study") or {}
        if not st.get("promoted"):
            return "skipped", "no baseline promoted"
        cmd = [PY, HERE / "navcam_baseline_study.py", "confirm", "--root", self.root, "--out", st["study"]]
        fl = (self.state.get("waypoint_check") or {}).get("flags")
        if fl:
            cmd += ["--flags", fl]
        rc = self.run("baseline_confirm", cmd, timeout_h=self.a.sites_timeout_h)
        self.state["baseline_confirm"] = dict(self.state.get("baseline_confirm") or {}, confirmed=(rc == 0))
        self.save()
        return "done", {0: "confirmed: the block scale agreement improved", 1: "NOT better: the previous rig promoted "
                        "back"}.get(rc, f"exit {rc} (see baseline_confirm.log)")

    def s_navcam_consensus_3(self):
        """v0p82: the joint refitted once with the new baseline held (recipe section 4), before Mastcam-Z."""
        if not (self.state.get("baseline_confirm") or {}).get("confirmed"):
            return "skipped", "no new baseline in use"
        prev = None
        for stg in ("navcam_consensus_2", "navcam_consensus"):
            stp = (self.state.get(stg) or {}).get("study")
            if stp and (Path(stp) / "navcam_joint" / "M2020_N_rig.json").is_file():
                prev = str(Path(stp) / "navcam_joint")
                break
        if prev is None:
            from mppp.paths import cmods_dir
            prev = str(cmods_dir())
        return self.s_navcam_consensus("navcam_consensus_3", previous=prev)

    def s_zcam_start(self):
        """v0p82: the found list cleaned (one focus state per sequence), then the start update from the Mastcam-Z
        blocks on disk - a candidate and new low-backlash groups only: the promoted v0p75 start model stays unless
        ``--zcam-start-promote`` (the blocks on disk are the 5-6 Oct runs, made with the Navcam and focus models
        that are being replaced)."""
        self.run("zcam_start", [PY, HERE / "zcam_calibration.py", "clean-found", "--root", self.root,
                                "--groups", self.zcam_group], timeout_h=2)
        cmd = [PY, HERE / "zcam_calibration.py", "start-update", "--root", self.root, "--groups",
               self.zcam_group]
        promote = bool(getattr(self.a, "zcam_start_promote", False))
        if promote:
            cmd.append("--promote")
        rc = self.run("zcam_start", cmd, timeout_h=6)
        if not promote:
            return "done", (f"found list cleaned and updated; start candidate written (exit {rc}), not promoted: the "
                            f"v0p75 start model stays (--zcam-start-promote to promote)")
        return ("done", {0: "start model promoted, found list updated", 1: "start model not promoted (a check failed); "
                         "found list updated"}.get(rc, f"error (exit {rc}); the focus model in use stays"))

    def _sites(self, stage, group, zcam, variant=""):
        cmd = [PY, ROOT / "scripts" / "run_sites.py", "--root", self.root, "--group", group]
        if zcam:
            cmd += ["--zcam", *ZCAM_SET]
        if variant:                       # v0p74.9: results in <WORK>/colmap_<variant>, the first runs kept
            cmd += ["--variant", variant]
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
        return self._sites("zcam_runs", self.zcam_group, True)

    def s_navcam_runs(self):
        if self.a.no_navcam_runs:
            return "skipped", "--no-navcam-runs"
        return self._sites("navcam_runs", NAVCAM_GROUP, False)

    def s_zcam_calibrate(self):
        rc = self.run("zcam_calibrate", [PY, HERE / "zcam_calibration.py", "calibrate", "--root", self.root,
                                         "--groups", self.zcam_group, "--n", str(self.a.n), "--promote"], timeout_h=6)
        self.state.setdefault("zcam_calibrate", {})["promoted"] = rc == 0 and not self.a.dry_run
        self.save()
        return "done", {0: "calibration promoted", 1: "calibration not promoted (too few passing blocks or a check "
                        "failed; see camera_analysis/zcam34_calibration_*)"}.get(rc, f"error (exit {rc})")

    def s_zcam_final_runs(self):
        if not (self.state.get("zcam_calibrate") or {}).get("promoted"):
            return "skipped", "no new calibration promoted"
        # v0p74.9: the final runs go to <WORK>/colmap_<final_variant> (default "cal"; features and matches seeded from
        # the first run's): on 5-6 Oct they overwrote the first runs in place, and with the promoted model the right
        # Mastcam-Z eye lost every tie point in Chal Rocks, Enchanted Lake and Three Forks North
        return self._sites("zcam_final_runs", self.zcam_group, True, variant=self.a.final_variant)

    def s_pointclouds(self):
        rc = self.run("pointclouds", [PY, HERE / "zcam_calibration.py", "pointclouds", "--root", self.root,
                                      "--groups", self.zcam_group, NAVCAM_GROUP], timeout_h=6)
        return ("done", "point clouds written") if rc == 0 else ("done", f"exit {rc}")

    def s_report(self):
        from mppp.calib.models import check_form, describe
        lines = [f"# MPPP {__import__('mppp').__version__} pipeline report ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
                 "## Stages", "", "| stage | status | note | started | ended |", "|---|---|---|---|---|"]
        for s in STAGES:
            e = self.state.get(s) or {}
            lines.append(f"| {s} | {e.get('status', '-')} | {e.get('note', '')} | {e.get('started', '')} | {e.get('ended', '')} |")
        lines += ["", "## Camera models in use", ""] + [f"    {ln}" for ln in describe()]
        f = check_form()
        lines += ["", f"form v0p73: {'OK' if f['ok'] else 'NOT OK: ' + '; '.join(f['errors'])}", ""]
        for stage_name in ("navcam_consensus", "navcam_consensus_2", "navcam_consensus_3"):       # v0p80: every pass
            st = (self.state.get(stage_name) or {}).get("study")
            if st:
                lines += [f"## Navcam consensus (pass {({'navcam_consensus': 1, 'navcam_consensus_2': 2}).get(stage_name, 3)})", "",
                          f"study: `{st}`"]
                pc = Path(st) / "pass_change.json"
                if pc.is_file():
                    ch = json.loads(pc.read_text(encoding="utf-8"))
                    lines.append(f"- change from pass 1: {ch.get('worst')} -> {'converged' if ch.get('converged') else 'NOT converged'}")
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
        # v0p82: waypoint flags, baseline study and its confirmation
        for key, files in (("waypoint_check", ("waypoint_flags.md",)), ("baseline_study", ("baseline_study.md",
                                                                                            "baseline_confirm.md"))):
            e = self.state.get(key) or {}
            d = Path(e["flags"]).parent if e.get("flags") else (Path(e["study"]) if e.get("study") else None)
            for f in files:
                if d is not None and (d / f).is_file():
                    lines += [(d / f).read_text(encoding="utf-8"), ""]
        # every block's last run
        lines += ["## Blocks", "", "| block | mppp | verdict | registered | rms px | camera models |", "|---|---|---|---|---|---|"]
        for w in sorted(p for p in self.root.glob("mars2020_sol_*") if p.is_dir()):      # v0p85: no _colmap
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
            if getattr(self.a, "stop_before", None) and s == self.a.stop_before:     # v0p85
                self.log(f"stopped before {s} (--stop-before); run again without it to go on from here")
                break
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
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"))
    ap.add_argument("--n", type=int, default=0,
                    help="the number of graded N+Z blocks for the calibration (v0p74.9: 0 = every passing block)")
    ap.add_argument("--final-variant", default="cal",
                    help="v0p74.9: zcam_final_runs write to <WORK>/colmap_<this> ('' = in place, over the first runs)")
    ap.add_argument("--no-navcam-runs", action="store_true", help="skip the Navcam-only reruns (and so pass 2)")
    ap.add_argument("--baseline-study", action="store_true",
                    help="v0p80/v0p82: promote a significant baseline even if pass 2 of the Navcam consensus has not "
                         "converged (it is always studied)")
    ap.add_argument("--no-baseline-study", action="store_true", help="v0p82: skip the baseline stages")
    ap.add_argument("--zcam-start-promote", action="store_true",
                    help="v0p82: let zcam_start promote its start model (default: candidate only, the v0p75 start "
                         "model in use stays)")
    ap.add_argument("--zcam-group", default=None,
                    help="v0p75: the sites.json group of the Mastcam-Z stages (default zcam34_consensus; e.g. zcam34_dev "
                         "for a quick set while the form is being settled)")
    ap.add_argument("--all-navcam-blocks", action="store_true",
                    help="v0p75: the Navcam consensus takes every finished Navcam block on disk (default: the sites of "
                         "the navcam_consensus group only)")
    ap.add_argument("--wait-h", type=float, default=36.0, help="how long to wait for a running batch")
    ap.add_argument("--stage-timeout-h", type=float, default=30.0, help="the consensus stage's time limit")
    ap.add_argument("--sites-timeout-h", type=float, default=72.0,
                    help="time limit of one run_sites call (a hung alignment is stopped with its kernel)")
    ap.add_argument("--from", dest="start", choices=STAGES, help="start at this stage (the ones before: skipped)")
    ap.add_argument("--redo", nargs="*", choices=STAGES, help="run these stages again even if done")
    ap.add_argument("--stop-before", choices=STAGES, default=None,
                    help="v0p85: stop before this stage (e.g. zcam_start: the Navcam part only); start again without it to go on")
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
