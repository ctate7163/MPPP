"""
The whole alignment, calibration and error-analysis run, unattended, in one command (0.93.9, Christian 23 Oct 2026:
run_sites plus run_sites_zcam, then an error analysis on the SfM points):

    python studies\\calibration\\run_pipeline.py                  (run_pipeline.bat)
    python studies\\calibration\\run_pipeline.py --status         what is done, what is running
    python studies\\calibration\\run_pipeline.py --dry-run        the plan, nothing run

Stages (each logged in ``<root>\\mppp_pipeline\\<stage>.log``, the pipeline in ``pipeline_log.txt``, the state in
``state.json``; started again, the pipeline continues after the last finished stage):

 1. ``wait``           - waits while another run_sites / process_sites batch runs on ROOT (at most --wait-h hours)
 2. ``ecam_runs``      - run_sites.py --group ecam_consensus (run_sites.bat): every engineering-camera block -
                         Navcam with the front and rear Hazcams - with the camera models in use; sites that failed
                         once more
 3. ``ecam_joint``     - the engineering cameras fitted over those blocks (0.94.4, Christian): the Navcam tied joint
                         (mppp calib navcam) and the Hazcam fit of both pairs, FL / FR / RL / RR and the F and R rigs
                         (mppp calib hazcam), each promoted when its checks pass (--no-promote: never), so the
                         Mastcam-Z blocks start from them
 4. ``zcam_runs``      - run_sites.py --zcam over the zcam_consensus sites and the other zcam34_joint sites
                         (run_sites_zcam.bat): the Navcam + Mastcam-Z + Hazcam blocks at every zoom of each site's
                         "zcam" list, with the Navcam and Hazcam models just promoted
 5. ``zcam_joints``    - the Mastcam-Z joint of each zoom over zcam34_joint + zcam_consensus (27 Oct; 48 / 63 were
                         over zcam_consensus only), each promoted when its checks pass, then the Navcam + Mastcam-Z check
                         (never promoted).  No second Hazcam fit: the Mastcam-Z blocks hold the same Hazcam frames
                         as the ecam blocks of their sols (--zcam-hazcam-fit fits them anyway)
 6. ``error_analysis`` - studies/analyses/error_analysis_batch.py over every finished block of both groups
                         (``<WORK>/colmap/error_input``): epsilon intra / cross station, by range and by angle, the
                         cross-station gate, the decorrelation rho(theta), the view graph, the registration against
                         the priors -> camera_analysis/error_analysis_<date>/ (json and report.md)
 7. ``rover_frame``    - studies/rover_frame/run_rover_frame.py (study 3A, 0.94.4): the rover body pose per Hazcam
                         sequence, the mast-head pointing residual against the RSM angles, the camera positions in
                         the body and head frames -> camera_analysis/rover_frame_<date>/ (report.md)
 8. ``lightglue_ab``   - scripts/lightglue_ab.py on Chal Rocks tiny and Seitah Overlook tiny (24 Oct, Christian): SIFT,
                         LightGlue, LightGlue on every octave and SIFT + LightGlue again with the camera models just
                         promoted (--rerun), no tiny joint (nothing promoted) -> <WORK>/lightglue_ab.md of each
 9. ``zcam_states``    - 0.94.4: the zcam_runs sites again as the variant ``states`` (every Mastcam-Z image its own
                         camera, f free, no focus lines: run_zcam_states.bat), then ``mppp zcam-states`` ->
                         camera_analysis/zcam_states_<date>/ (report.md, found_list_proposal.txt; the found list is
                         not changed - ``--apply`` does that)
10. ``report``         - ``<root>\\mppp_pipeline\\REPORT.md``

A stage that fails is logged and the pipeline goes on (the next stage works with what there is).  Exit code 0 when
every stage ran (whatever it found).  The pipeline before 0.93.9 (consensus passes, baseline study, the Z34-only
Mastcam-Z calibration from per-block lines) is in the git history; its state file is kept aside as
``state_before_<tag>.json``.
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 27 Oct: as mppp_env.bat - two OpenMP runtimes (torch, pycolmap) abort a plain "python" start

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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))        # _batch (batch lock, process control)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable
STAGES = ("wait", "ecam_runs", "ecam_joint", "zcam_runs", "zcam_joints", "error_analysis", "rover_frame", "lightglue_ab",
          "zcam_states", "report")
# 0.94.4 (Christian, 25 Oct): the states run - the Mastcam-Z sites again as the variant "states", one camera per image
# with its own focal length (run_zcam_states.bat), then mppp zcam-states: which frames are low, the found-list proposal
ZCAM_STATES_SETS = ("ZCAM_BIN_BY=image", "ZCAM_HOLD_F_IMAGES=0", "ZCAM_BIN_REFINE=focal_pp", "ZCAM_FOCUS_LINE=false")
# 0.94.5 (Christian, 27 Oct): focal_pp - the right eye's centre free per image too (its state shows there; focal held it)
# the stage set of the state file (0.94.4: ecam stages).  Not changed for the rover_frame stage (added 23 Oct during
# the weekend run): a state of this tag simply lacks it, so a rerun does rover_frame and the report only
STATE_TAG = "v0944"
STAGE_ALIASES = {"navcam_runs": "ecam_runs", "navcam_joint": "ecam_joint"}     # the names before 0.94.4
STOP_ON_FAIL = ()                                # every stage goes on to the next
ZCAM_GROUP, ECAM_GROUP, ZCAM34_JOINT = "zcam_consensus", "ecam_consensus", "zcam34_joint"
NAVCAM_GROUP = ECAM_GROUP                        # the name before 0.94.4
ZOOMS = (34, 48, 63)                # Christian, 26 Oct: 79 and 110 mm out until the wider zooms have good models
ALL_ZOOMS = (34, 48, 63, 79, 110)
LIGHTGLUE_AB_SITES = ("chal_rocks_tiny", "seitah_overlook_tiny")        # the lightglue_ab stage (24 Oct)


class Pipeline:
    def __init__(self, a):
        from mppp.runner import Log
        self.a = a
        self.zcam_group = getattr(self.a, "zcam_group", None) or ZCAM_GROUP
        self.root = Path(a.root)
        self.dir = self.root / "mppp_pipeline"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.dir / ("dry_run_log.txt" if a.dry_run else "pipeline_log.txt"))
        self.state_file = self.dir / "state.json"
        self.state = json.loads(self.state_file.read_text(encoding="utf-8")) if self.state_file.is_file() else {}
        if self.state and self.state.get("_stages") != STATE_TAG:
            # the state of the pipeline before 0.93.9 (other stages, some with the same names): kept aside, not used
            if not a.dry_run:
                self.state_file.replace(self.dir / f"state_before_{STATE_TAG}.json")
            self.state = {}
        self.state["_stages"] = STATE_TAG

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
                from _batch import kill_tree
                kill_tree(p.pid)
                p.wait(timeout=60)
                fh.write(f"\nTIMEOUT after {timeout_h} h: stopped\n")
                rc = 124
        self.log(f"  {stage}: exit {rc} (log {lf})")
        return rc

    # ---------------------------------------------------------------- stages
    def s_wait(self):
        from _batch import alive_batches                         # 0.95.0: one status file per batch
        t0 = time.time()
        while True:
            alive = alive_batches(self.root)
            if not alive:
                return "done", "no batch running"
            cur = alive[0]
            if time.time() - t0 > self.a.wait_h * 3600:
                return "failed", f"a batch ({cur.get('kind')}, pid {cur.get('pid')}) still runs after {self.a.wait_h} h"
            self.log(f"  wait: {len(alive)} batch(es) running - {cur.get('kind')} (site {cur.get('site')}); checking again in 10 min")
            if self.a.dry_run:
                return "done", "dry run"
            time.sleep(600)

    def _out(self, name: str, marker: str) -> Path:
        """``<root>/camera_analysis/<name>_<date>[_N]``: the first one without ``marker`` in it."""
        base = self.root / "camera_analysis" / f"{name}_{datetime.date.today():%Y%m%d}"
        out, n = base, 1
        while (out / marker).exists():
            n += 1
            out = base.with_name(f"{base.name}_{n}")
        return out

    def _mppp(self, stage, out: Path, args, timeout_h=None):
        """``python -m mppp --log <out>_log.txt <args>``; returns the exit code."""
        return self.run(stage, [PY, "-m", "mppp", "--log", f"{out}_log.txt", *map(str, args)], timeout_h=timeout_h)

    def _sites(self, stage, sites_args, zcam):
        cmd = [PY, ROOT / "scripts" / "run_sites.py", "--root", self.root, *sites_args]
        cmd += ["--jobs", str(max(1, int(getattr(self.a, "jobs", 2) or 1)))]   # 0.94.3: two at a time by default
        for x in getattr(self.a, "set", None) or []:     # 0.94.4: pipeline settings for every block run
            cmd += ["--set", x]
        if zcam and getattr(self.a, "zooms", None) and not getattr(self.a, "all_site_zooms", False):
            # 0.95.0 (Christian, 28 Oct): the Mastcam-Z runs keep the zooms being calibrated (--zooms, default
            # 34 48 63); the site's wider zooms wait for a pass of their own (--all-site-zooms: every zoom listed)
            cmd += ["--set", "zcam_zooms_only=" + ",".join(str(int(z)) for z in self.a.zooms)]
        if zcam:
            cmd.append("--zcam")
        rc = self.run(stage, cmd, timeout_h=self.a.sites_timeout_h)
        for _ in range(3):
            if rc == 3 and not self.a.dry_run:                 # another batch started meanwhile: wait for it
                st, note = self.s_wait()
                if st != "done":
                    return "failed", note
                rc = self.run(stage, cmd, timeout_h=self.a.sites_timeout_h)
            else:
                break
        if rc == 1 and not self.a.dry_run:                     # a failed site once more (finished ones are skipped)
            self.log(f"  {stage}: some sites failed - once more")
            rc = self.run(stage, cmd, timeout_h=self.a.sites_timeout_h)
        return ("done", "every site finished") if rc == 0 else ("done", f"finished with failed sites (exit {rc}; see run_sites_log.txt)")

    def s_ecam_runs(self):
        if self.a.no_ecam_runs:
            return "skipped", "--no-ecam-runs"
        return self._sites("ecam_runs", ["--group", ECAM_GROUP], False)

    def s_ecam_joint(self):
        """The Navcam joint, then the Hazcam fit, over the ecam_consensus blocks - both before the Mastcam-Z runs."""
        promote = [] if self.a.no_promote else ["--promote-if-ok"]
        out = self._out("navcal_tied", "navcam_joint/tied_joint.json")
        self.state.setdefault("ecam_joint", {})["study"] = str(out)
        self.save()
        args = ["calib", "navcam", "--root", self.root, "--group", ECAM_GROUP, "--out", out, "--knot", "250",
                "--points", "8000", *promote]
        rc = self._mppp("ecam_joint", out, args, timeout_h=self.a.stage_timeout_h)
        chk = out / "promotion_check.json"
        rep = json.loads(chk.read_text(encoding="utf-8")) if chk.is_file() else {}
        if rc != 0 and not rep:
            nav = f"Navcam joint failed (exit {rc}; {out}_log.txt), the Navcam models in use stay"
        elif rep.get("ok"):
            nav = "Navcam: checks passed" + ("" if self.a.no_promote else ", promoted") + f" ({out})"
        else:
            nav = ("Navcam: not promoted (a check failed): " + ", ".join(
                c.get("check", "?") for c in rep.get("checks", []) if not c.get("ok")) + f" ({out})")
        notes = [nav]
        if not self.a.no_hazcam_fit:
            hout = self._out("hazcam_fit_ecam", "hazcam_fit.json")
            self.state["ecam_joint"]["hazcam_study"] = str(hout)
            self.save()
            hrc = self._mppp("ecam_joint", hout, ["calib", "hazcam", "--group", ECAM_GROUP, "--root", self.root,
                                                  "--out", hout, *promote], timeout_h=6)
            notes.append(f"Hazcam fit: exit {hrc} ({hout}; promoted: the eyes and rigs whose checks passed)")
        failed = rc != 0 and not rep
        return ("failed" if failed else "done"), "; ".join(notes)

    def zcam_sites(self):
        """The sites of the Mastcam-Z stages: zcam_consensus, then the other zcam34_joint sites (the Z34 joint takes
        both groups, and its blocks must be aligned with the Navcam models in use)."""
        from mppp.waypoints.sites import load_site_table
        groups = load_site_table().get("groups", {})
        names = list(groups.get(self.zcam_group) or [])
        names += [n for n in groups.get(ZCAM34_JOINT) or [] if n not in names]
        return names

    def s_zcam_runs(self):
        names = self.zcam_sites()
        if not names:
            return "failed", f"no sites in {self.zcam_group} / {ZCAM34_JOINT}"
        only = list(getattr(self.a, "zcam_sites", None) or [])
        if only:
            # 0.94.4 (Christian, 26 Oct - results by the end of the day after a restart): align only these; the
            # joints take the other sites' finished blocks as they are (a block aligned with other Navcam models
            # fails the joint's check; one never finished is left out)
            unknown = [n for n in only if n not in names]
            if unknown:
                return "failed", f"--zcam-sites: not Mastcam-Z sites of this run: {', '.join(unknown)}"
            names = list(dict.fromkeys(only))           # Christian, 26 Oct: in the order given (the first first)
        return self._sites("zcam_runs", ["--sites", *names], True)

    def s_zcam_joints(self):
        notes = []
        promote = [] if self.a.no_promote else ["--promote-if-ok"]
        if self.a.zcam_hazcam_fit:                   # 0.94.4: off by default (the ecam_joint stage fits them)
            out = self._out("hazcam_fit_zcam", "hazcam_fit.json")
            rc = self._mppp("zcam_joints", out, ["calib", "hazcam", "--zcam-blocks", "--group", self.zcam_group,
                                                 "--root", self.root, "--out", out, *promote], timeout_h=6)
            notes.append(f"Hazcam fit exit {rc}")
        for z in self.a.zooms:
            # 27 Oct (Christian: Enchanted Lake with 63 mm): every zoom over both groups - a site without the
            # zoom in its "zcam" list is left out by the joint
            group = f"{ZCAM34_JOINT},{self.zcam_group}"
            out = self._out(f"zcam{z}_joint", "zcam_joint.json")
            rc = self._mppp("zcam_joints", out, ["calib", "zcam", "--zoom", z, "--group", group, "--root", self.root,
                                                 "--out", out, *promote], timeout_h=self.a.stage_timeout_h)
            ck = out / "check.json"
            ok = json.loads(ck.read_text(encoding="utf-8")).get("ok") if ck.is_file() else None
            notes.append(f"Z{z}: " + ("not run / failed" if rc != 0 and ok is None else
                                      "checks passed" if ok else "a check failed"))
        if not self.a.no_nz:
            out = self._out("nz34_joint", "nz_joint.json")
            rc = self._mppp("zcam_joints", out, ["calib", "nz", "--group", f"{ZCAM34_JOINT},{self.zcam_group}",
                                                 "--navcam-group", ECAM_GROUP, "--root", self.root, "--out", out],
                            timeout_h=self.a.stage_timeout_h)
            notes.append(f"N+Z check exit {rc}")
        return "done", "; ".join(notes)

    def error_blocks(self):
        """{name: WORK} of every finished block of the Navcam and the Mastcam-Z stages with an error_input folder."""
        from mppp.waypoints.sites import load_site_table, work_folder
        table = load_site_table()
        out = {}
        for names, zcam in ((list(table.get("groups", {}).get(NAVCAM_GROUP) or []), False), (self.zcam_sites(), True)):
            for n in names:
                site = table["sites"].get(n)
                if not site or (zcam and not site.get("zcam")):
                    continue
                w = work_folder(self.root, n, zcam, site["sols"])
                if (w / "colmap" / "run_done.json").is_file() and (w / "colmap" / "error_input").is_dir():
                    out[f"{n}_zcam" if zcam else n] = w
        return out

    def s_error_analysis(self):
        blocks = self.error_blocks()
        if not blocks and not self.a.dry_run:
            return "failed", "no finished block with an error_input folder"
        out = self._out("error_analysis", "error_analysis_batch.json")
        self.state.setdefault("error_analysis", {})["study"] = str(out)
        self.save()
        cmd = [PY, ROOT / "studies" / "analyses" / "error_analysis_batch.py", out,
               *[f"{k}={v}" for k, v in blocks.items()], "--min-track", "3", "--n-boot", str(self.a.n_boot)]
        rc = self.run("error_analysis", cmd, timeout_h=self.a.stage_timeout_h)
        js = out / "error_analysis_batch.json"
        if not js.is_file():
            return "failed", f"no result (exit {rc}; see error_analysis.log)"
        md = error_report(json.loads(js.read_text(encoding="utf-8")))
        (out / "report.md").write_text(md, encoding="utf-8")
        return "done", f"{len(blocks)} blocks -> {out / 'report.md'}"

    def s_rover_frame(self):
        """Study 3A over every finished block of both groups (no new adjustment)."""
        out = self._out("rover_frame", "report.md")
        self.state.setdefault("rover_frame", {})["study"] = str(out)
        self.save()
        cmd = [PY, ROOT / "studies" / "rover_frame" / "run_rover_frame.py", "--root", self.root, "--out", out,
               "--groups", ECAM_GROUP, "--zcam-groups", self.zcam_group]
        rc = self.run("rover_frame", cmd, timeout_h=6)
        if self.a.dry_run:
            return "done", "dry run"
        if not (out / "report.md").is_file():
            return "failed", f"no report (exit {rc}; see rover_frame.log)"
        return "done", str(out / "report.md")

    def s_lightglue_ab(self):
        """The LightGlue A/B test on the two tiny blocks with the models in use, after everything else (24 Oct):
        every variant again, no tiny joint - it must not promote anything at the end of a run that promoted."""
        notes, ok = [], True
        for site in LIGHTGLUE_AB_SITES:
            cmd = [PY, ROOT / "scripts" / "lightglue_ab.py", "--root", self.root, "--site", site, "--no-joint",
                   "--no-promote", "--rerun"]
            rc = self.run("lightglue_ab", cmd, timeout_h=getattr(self.a, "ab_timeout_h", 8.0))
            md = next(iter(self.root.glob(f"*_{site}_zcam/lightglue_ab.md")), None)
            notes.append(f"{site}: exit {rc}" + (f" ({md})" if md else ""))
            ok = ok and rc == 0
        return ("done" if ok or self.a.dry_run else "failed"), "; ".join(notes)

    def s_zcam_states(self):
        """The states run over the zcam_runs sites, then the per-image states and the found-list proposal."""
        if getattr(self.a, "no_zcam_states", False):
            return "skipped", "--no-zcam-states"
        names = self.zcam_sites()
        if not names:
            return "failed", f"no sites in {self.zcam_group} / {ZCAM34_JOINT}"
        sets = []
        for x in ZCAM_STATES_SETS:
            sets += ["--set", x]
        st, note = self._sites("zcam_states", ["--sites", *names, "--variant", "states", *sets], True)
        out = self._out("zcam_states", "report.md")
        self.state.setdefault("zcam_states", {})["study"] = str(out)
        self.save()
        rc = self._mppp("zcam_states", out, ["zcam-states", "--root", self.root, "--variant", "states", "--out", out],
                        timeout_h=2)
        if self.a.dry_run:
            return "done", "dry run"
        if not (out / "report.md").is_file():
            return "failed", f"{note}; no states report (exit {rc}; see zcam_states.log)"
        return "done", f"{note}; {out / 'report.md'} (proposal: found_list_proposal.txt, not applied)"

    def s_report(self):
        from mppp.calib.models import describe
        lines = [f"# MPPP {__import__('mppp').__version__} pipeline report ({datetime.datetime.now():%Y-%m-%d %H:%M})", "",
                 "## Stages", "", "| stage | status | note | started | ended |", "|---|---|---|---|---|"]
        for s in STAGES:
            e = self.state.get(s) or {}
            lines.append(f"| {s} | {e.get('status', '-')} | {e.get('note', '')} | {e.get('started', '')} | {e.get('ended', '')} |")
        lines += ["", "## Camera models in use", ""] + [f"    {ln}" for ln in describe()]
        lines += ["", "## Blocks", "", "| block | mppp | verdict | registered | rms px | camera models |",
                  "|---|---|---|---|---|---|"]
        for w in sorted(p for p in self.root.glob("mars2020_sol_*") if p.is_dir()):
            rd = w / "colmap" / "run_done.json"
            if rd.is_file():
                d = json.loads(rd.read_text(encoding="utf-8"))
                lines.append(f"| {w.name} | {d.get('mppp')} | {d.get('verdict')} | {d.get('registered_images')}/{d.get('images')} | "
                             f"{(d.get('residual_rms_native_px') or float('nan')):.3f} | "
                             f"{(d.get('camera_models') or {}).get('fingerprint', '-')} |")
        ca = self.root / "camera_analysis"
        today = f"{datetime.date.today():%Y%m%d}"
        for pat, title in (("navcal_tied_*", "Navcam joint"), ("hazcam_fit_ecam_*", "Hazcam fit (ecam blocks)"),
                           ("hazcam_fit_zcam_*", "Hazcam fit (Mastcam-Z blocks)"),
                           *[(f"zcam{z}_joint_*", f"Mastcam-Z {z} mm joint") for z in ALL_ZOOMS],
                           ("nz34_joint_*", "Navcam + Mastcam-Z check"), ("error_analysis_*", "Error analysis"),
                           ("rover_frame_*", "Rover frame (study 3A)")):
            found = sorted(p for p in ca.glob(pat) if p.is_dir()) if ca.is_dir() else []
            if found:
                rp = found[-1] / "report.md"
                lines += ["", f"## {title}", "", f"latest: `{found[-1]}`" + ("" if today in found[-1].name else " (not from today)")]
                if rp.is_file():
                    lines += ["", rp.read_text(encoding="utf-8")]
        for site in LIGHTGLUE_AB_SITES:                         # 24 Oct: the LightGlue A/B of the tiny blocks
            md = next(iter(self.root.glob(f"*_{site}_zcam/lightglue_ab.md")), None)
            if md is not None and md.is_file():
                lines += ["", f"## LightGlue A/B: {site}", "", md.read_text(encoding="utf-8")]
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
            if st == "failed" and s in STOP_ON_FAIL:
                self.log("pipeline stopped: fix the above and start it again (it continues from here)")
                rc = 1
                break
        if status:
            status.stop("finished" if rc == 0 else "failed")
        self.log("pipeline " + ("finished" if rc == 0 else "stopped") + f"; report: {self.dir / 'REPORT.md'}")
        return rc


def error_report(res) -> str:
    """report.md of an error_analysis_batch.json: the parameter row of every block, then the pooled gate."""
    rows = res.get("summary") or []
    lines = ["# Error analysis on the SfM points", "",
             "Per block (`mppp.error.alignment.parameters`): image-measurement precision epsilon (px) on tracks of "
             "one station and across stations, the cross-station gate and the decorrelation; sparse tie points are "
             "biased to well-matched terrain, so epsilon is a lower bound.", ""]
    if rows:
        keys = [k for k in rows[0] if not isinstance(rows[0][k], (dict, list))][:14]
        lines += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r in rows:
            lines.append("| " + " | ".join((f"{r.get(k):.4g}" if isinstance(r.get(k), float) else str(r.get(k, "")))
                                            for k in keys) + " |")
    pooled = [g for g in res.get("gate_fits") or [] if g.get("alignment") == "pooled"]
    if pooled:
        g = pooled[0]
        keys = [k for k in g if not isinstance(g[k], (dict, list))][:16]
        lines += ["", "## Pooled cross-station gate (Navcam-Navcam)", ""] + \
                 [f"- {k}: {g[k]:.4g}" if isinstance(g[k], float) else f"- {k}: {g[k]}" for k in keys]
    eps = res.get("eps") or []
    if eps:
        keys = [k for k in eps[0] if not isinstance(eps[0][k], (dict, list))][:10]
        lines += ["", "## Epsilon by instrument and track kind", "", "| " + " | ".join(keys) + " |",
                  "|" + "---|" * len(keys)]
        for r in eps:
            lines.append("| " + " | ".join((f"{r.get(k):.4g}" if isinstance(r.get(k), float) else str(r.get(k, "")))
                                            for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"))
    ap.add_argument("--no-ecam-runs", "--no-navcam-runs", dest="no_ecam_runs", action="store_true",
                    help="skip the engineering-camera (Navcam + Hazcam) block runs")
    ap.add_argument("--no-promote", action="store_true", help="the joints and the Hazcam fit promote nothing")
    ap.add_argument("--no-hazcam-fit", action="store_true", help="ecam_joint without the Hazcam fit")
    ap.add_argument("--zcam-hazcam-fit", action="store_true",
                    help="zcam_joints also fits the Hazcams over the Mastcam-Z blocks (off since 0.94.4)")
    ap.add_argument("--no-nz", action="store_true", help="zcam_joints without the Navcam + Mastcam-Z check")
    ap.add_argument("--no-zcam-states", action="store_true", help="skip the zcam_states stage (the states run)")
    ap.add_argument("--zooms", type=int, nargs="*", default=list(ZOOMS), help="the Mastcam-Z joints, and the zooms the site runs keep (default 34, 48, 63; 0.95.0)")
    ap.add_argument("--all-site-zooms", action="store_true", help="the site runs align every zoom of the site's list (79, 110 mm too), not only --zooms")
    ap.add_argument("--zcam-group", default=None, help="the Mastcam-Z sites (default zcam_consensus)")
    ap.add_argument("--zcam-sites", nargs="+", default=None, metavar="SITE",
                    help="zcam_runs aligns only these sites; the joints use the others' finished blocks as they are")
    ap.add_argument("--n-boot", type=int, default=50, help="error_analysis: bootstrap rounds of the gate fit")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("MPPP_JOBS", "2") or 2),
                    help="ecam_runs / zcam_runs: sites aligned at the same time (default 2, or MPPP_JOBS; 0.94.3)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="a pipeline setting for every block run (run_sites --set), e.g. MATCH_MODE=exhaustive")
    ap.add_argument("--wait-h", type=float, default=36.0, help="how long to wait for a running batch")
    ap.add_argument("--stage-timeout-h", type=float, default=30.0, help="the time limit of a joint or the analysis")
    ap.add_argument("--ab-timeout-h", type=float, default=8.0, help="lightglue_ab: the time limit per tiny block")
    ap.add_argument("--sites-timeout-h", type=float, default=72.0,
                    help="time limit of one run_sites call (a hung alignment is stopped with its process tree)")
    names = list(STAGES) + list(STAGE_ALIASES)       # the old names still work
    ap.add_argument("--from", dest="start", choices=names, help="start at this stage (the ones before: skipped)")
    ap.add_argument("--redo", nargs="*", choices=names, help="run these stages again even if done")
    ap.add_argument("--stop-before", choices=names, default=None, help="stop before this stage; start again to go on")
    ap.add_argument("--reset", action="store_true", help="forget the state (every stage runs again)")
    ap.add_argument("--status", action="store_true", help="show the state and exit")
    ap.add_argument("--dry-run", action="store_true", help="show the commands, run nothing")
    a = ap.parse_args(argv)
    a.start = STAGE_ALIASES.get(a.start, a.start)
    a.stop_before = STAGE_ALIASES.get(a.stop_before, a.stop_before)
    a.redo = [STAGE_ALIASES.get(x, x) for x in a.redo] if a.redo else a.redo
    p = Pipeline(a)
    if a.status:
        from mppp.runner import read_status
        cur = read_status(p.dir / "pipeline_status.json")
        print(f"pipeline: {(cur or {}).get('state', 'not started')}" + (f", stage {cur.get('stage')}" if cur else ""))
        for s in STAGES:
            e = p.state.get(s) or {}
            print(f"  {s:16s} {e.get('status', '-'):8s} {e.get('note', '')}")
        return 0
    if a.reset and not a.dry_run:
        p.state = {"_stages": STATE_TAG}
        p.save()
    return p.go()


if __name__ == "__main__":
    sys.exit(main())
