"""Tests of the unattended pipeline driver and the study scripts (studies/calibration/run_pipeline.py,
navcam_baseline_study.py, navcam_best_candidate.py, zr_report.py, the .bat files); moved from the versioned tests
of 0.74-0.85 when the scripts left scripts/."""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
CMODS = ROOT / "src" / "mppp" / "data" / "cmods"
RP = HERE / "run_pipeline.py"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pipeline_dry_run_and_state(tmp_path):
    r = subprocess.run([sys.executable, str(RP), "--root", str(tmp_path), "--dry-run"],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    for s in ("wait", "prepare", "navcam_consensus", "zcam_start", "zcam_runs", "navcam_runs", "zcam_calibrate",
              "zcam_final_runs", "pointclouds", "report"):
        assert f"{s}: start" in out, (s, out[-3000:])
    assert "run_sites.py" in out and "ZCAM_ZOOMS=[34]" in out
    assert "zcam_final_runs: skipped" in out                              # nothing promoted in a dry run
    assert not (tmp_path / "mppp_pipeline" / "state.json").exists()      # a dry run leaves no state
    assert (tmp_path / "mppp_pipeline" / "REPORT.md").is_file()
    r2 = subprocess.run([sys.executable, str(RP), "--root", str(tmp_path), "--status"],
                        capture_output=True, text=True, timeout=120)
    assert "navcam_consensus" in r2.stdout


def test_pipeline_stage_logic(tmp_path, monkeypatch):
    """The driver: prepare failing stops it; a rerun continues after the finished stages; --from skips."""
    rp = _script("run_pipeline")
    calls = []

    def fake(name, result):
        def f(self):
            calls.append(name)
            return result
        return f
    for s in rp.STAGES:
        monkeypatch.setattr(rp.Pipeline, "s_" + s, fake(s, ("done", "ok")))
    monkeypatch.setattr(rp.Pipeline, "s_prepare", fake("prepare", ("failed", "bad cameras")))
    a = SimpleNamespace(root=str(tmp_path), dry_run=False, start=None, redo=None, n=3, no_navcam_runs=False, wait_h=1,
                        stage_timeout_h=1)
    assert rp.Pipeline(a).go() == 1 and calls == ["wait", "prepare"]
    monkeypatch.setattr(rp.Pipeline, "s_prepare", fake("prepare", ("done", "ok")))
    calls.clear()
    assert rp.Pipeline(a).go() == 0
    assert calls[0] == "prepare" and calls[-1] == "report" and "wait" not in calls      # wait was done before
    calls.clear()
    assert rp.Pipeline(a).go() == 0 and calls == ["report"]                               # all done: only the report
    a.redo = ["zcam_calibrate"]
    calls.clear()
    assert rp.Pipeline(a).go() == 0 and calls == ["zcam_calibrate", "report"]


def test_pipeline_prepare_keeps_newer_cameras(tmp_path, monkeypatch):
    rp = _script("run_pipeline")
    seen = {}
    monkeypatch.setattr(rp.Pipeline, "run", lambda self, stage, cmd, timeout_h=None: seen.setdefault("cmd", cmd) and 1)
    a = SimpleNamespace(root=str(tmp_path), dry_run=False, start=None, redo=None, n=3, no_navcam_runs=False, wait_h=1,
                        stage_timeout_h=1)
    st, note = rp.Pipeline(a).s_prepare()
    assert "--keep" in seen["cmd"] and st == "done"                       # the cameras in use are in the form


def test_bats():
    """0.91: every .bat lives in scripts\windows (two levels below the repo), the study ones included."""
    WIN = HERE.parents[1] / "scripts" / "windows"
    for b, needs in (("run_pipeline.bat", b"studies\\calibration\\run_pipeline.py"),
                     ("run_v0p91.bat", b"--stop-before zcam_start"),
                     ("navcam_tied_joint.bat", b"calib navcam"),
                     ("make_figures.bat", b"paper_figures.py"), ("m20_dem_env.bat", b"M20_ORTHOS"),
                     ("make_maps.bat", b"m20_dem_env.bat"), ("run_z34.bat", b"--group zcam34_consensus --zcam")):
        t = (WIN / b).read_bytes()
        assert b"\r\n" in t and needs in t
        assert b'"%~dp0..\\..\\.."' not in t, b                        # the repo is two levels up, not three
    assert not (HERE / "windows").exists()
    bat = (WIN / "run_v0p91.bat").read_text()
    assert "--reset --from navcam_runs --no-baseline-study" in bat
    assert not (WIN / "zr_diagnose_z34.bat").exists()                     # removed 20 Oct


def test_pipeline_dry_run_leaves_no_state(tmp_path):
    rp = _script("run_pipeline")
    root = tmp_path / "scapes"
    root.mkdir()
    assert rp.main(["--root", str(root), "--dry-run"]) == 0
    assert not (root / "mppp_pipeline" / "state.json").exists()
    assert (root / "mppp_pipeline" / "dry_run_log.txt").is_file()


def test_pipeline_run_timeout_stops_the_tree(tmp_path):
    rp = _script("run_pipeline")
    p = rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=False))
    t0 = time.time()
    rc = p.run("x", [sys.executable, "-c", "import time; time.sleep(60)"], timeout_h=2.0 / 3600)
    assert rc == 124 and time.time() - t0 < 30
    assert "TIMEOUT" in (tmp_path / "mppp_pipeline" / "x.log").read_text()


def test_pipeline_consensus_leaves_out_guided_matched_blocks(tmp_path, monkeypatch):
    rp = _script("run_pipeline")
    from mppp.sfm import health as H
    from mppp.waypoints import sites as S
    found = {"A": tmp_path / "mars2020_sol_0100_a_colmap", "B": tmp_path / "mars2020_sol_0200_b_colmap"}
    for w in found.values():
        (w / "colmap").mkdir(parents=True)
        (w / "colmap" / "run_done.json").write_text("{}")
    monkeypatch.setattr(S, "discover_scapes", lambda root, include_zcam=False, verbose=False: found)
    monkeypatch.setattr(S, "load_site_table", lambda path=None: {"sites": {}, "groups": {}})
    monkeypatch.setattr(S, "parse_work_folder", lambda w: (w.name.split("_")[3], False))
    monkeypatch.setattr(H, "screen_blocks", lambda d, verbose=False: {
        "A": {"verdict": "pass", "guided_matching": False}, "B": {"verdict": "warn", "guided_matching": True}})
    p = rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=True))
    blocks, scr, bad = p.navcam_blocks()
    assert list(blocks) == ["A"] and bad == ["B"]


def test_pipeline_consensus_takes_the_group_only(tmp_path, monkeypatch):
    rp = _script("run_pipeline")
    from mppp.sfm import health as H
    from mppp.waypoints import sites as S
    found = {"A": tmp_path / "mars2020_sol_0100_a_colmap", "B": tmp_path / "mars2020_sol_0200_b_colmap",
             "Olifants": tmp_path / "mars2020_sol_1880_olifants_colmap"}
    for w in found.values():
        (w / "colmap").mkdir(parents=True)
        (w / "colmap" / "run_done.json").write_text("{}")
    monkeypatch.setattr(S, "discover_scapes", lambda root, include_zcam=False, verbose=False: found)
    monkeypatch.setattr(S, "load_site_table", lambda path=None: {"sites": {}, "groups": {"navcam_consensus": ["a", "b"]}})
    monkeypatch.setattr(S, "parse_work_folder", lambda w: (w.name.split("_")[3], False))
    monkeypatch.setattr(H, "screen_blocks", lambda d, verbose=False: {k: {"verdict": "pass", "guided_matching": False} for k in d})
    p = rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=True))
    blocks, scr, bad = p.navcam_blocks()
    assert sorted(blocks) == ["A", "B"] and not bad
    p = rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=True, all_navcam_blocks=True))
    blocks, _, _ = p.navcam_blocks()
    assert sorted(blocks) == ["A", "B", "Olifants"]


def test_pipeline_zcam_group_option(tmp_path):
    rp = _script("run_pipeline")
    assert rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=True)).zcam_group == "zcam34_consensus"
    assert rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=True, zcam_group="zcam34_dev")).zcam_group == "zcam34_dev"


def test_pipeline_order_and_options():
    rp = _script("run_pipeline")
    s = list(rp.STAGES)
    assert s.index("navcam_consensus") < s.index("navcam_runs") < s.index("navcam_consensus_2") < s.index("zcam_start")
    assert s.index("navcam_consensus_2") < s.index("waypoint_check") < s.index("baseline_study") \
        < s.index("baseline_confirm") < s.index("navcam_consensus_3") < s.index("zcam_start") < s.index("report")
    src = RP.read_text()
    assert '"--apply-translation", "--joint-translation"' not in src
    assert '"--final-variant", default="cal"' in src and 'variant=self.a.final_variant' in src
    assert '"--n", type=int, default=0' in src
    assert 'EARLY_ROUNDS = "1"' in src and '"0.197", "0.254", "0.324", "-0.572"' in src
    from mppp.sfm.matching import CALIBRATED_ABORT_SIGNS
    msg = "E20261006 09:27:36 homography_matrix.cc:167] Check failed: cam_rays1.size() >= 4 (3 vs. 4)"
    assert any(k in msg for k in CALIBRATED_ABORT_SIGNS)


def test_pipeline_dry_run_log_and_stop_before(tmp_path):
    rp = _script("run_pipeline")
    assert rp.main(["--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "mppp_pipeline" / "dry_run_log.txt").read_text()
    assert "waypoint_check.py" in log and "navcam_baseline_study.py all" in log
    out = subprocess.run([sys.executable, str(RP), "--root", str(tmp_path / "b"), "--dry-run", "--from", "navcam_runs",
                          "--stop-before", "zcam_start", "--no-baseline-study"],
                         capture_output=True, text=True, timeout=120)
    log = out.stdout + out.stderr
    assert "stopped before zcam_start" in log, log[-2000:]


# ------------------------------------------------------------------------------------------- the study scripts
def test_baseline_combine():
    m = _script("navcam_baseline_study")
    rows = [{"site": s, "baseline_m": 0.4244 + d, "baseline_sd_m": 0.0003, "T_median_degC": T, "sol_median": sol}
            for s, d, T, sol in (("a", 0.0008, -20, 600), ("b", 0.0009, -15, 800), ("c", 0.0007, -25, 1600),
                                 ("d", 0.0010, -10, 1800))]
    r = m.combine_rows(rows, 0.4244)
    assert r["adopt"] and r["difference_mm"] == pytest.approx(0.85, abs=0.05) and r["chi2_dof"] < 2
    r2 = m.combine_rows(rows, 0.4252)
    assert not r2["adopt"]
    assert m.combine_rows(rows[:1], 0.4244)["adopt"] is False


def test_baseline_rule_needs_better_scale_agreement_and_isotropy():
    m = _script("navcam_baseline_study")
    b_use = 0.42401
    rows = [{"site": f"s{i}", "baseline_m": b_use * (1 - 2.5e-3), "baseline_sd_m": 0.0002, "scale_held_ppm": 2500.0,
             "anisotropy_held": {"scale_E_ppm": 0.0, "se_E_ppm": 300.0, "scale_N_ppm": 0.0, "se_N_ppm": 300.0}}
            for i in range(5)]
    res = m.combine_rows(rows, b_use)
    assert res["adopt"] and res["scale_agreement"]["rms_predicted_ppm"] < 50
    rows2 = [dict(r, scale_held_ppm=0.0) for r in rows]               # already in scale: the agreement would get worse
    res2 = m.combine_rows(rows2, b_use)
    assert not res2["adopt"] and "would not improve" in res2["verdict"]
    rows3 = [dict(r, anisotropy_held={"scale_E_ppm": 5000.0, "se_E_ppm": 300.0, "scale_N_ppm": 0.0, "se_N_ppm": 300.0})
             for r in rows]                                           # east-west only: a map-projection error
    res3 = m.combine_rows(rows3, b_use)
    assert not res3["adopt"] and "map-projection" in res3["verdict"]


def test_baseline_sites_qualify_from_the_waypoint_fields():
    m = _script("navcam_baseline_study")
    keep, left = m.qualifying_sites()
    assert keep and all(s in ("van_zyl_overlook_huge", "chal_rocks_large", "seitah_north_large", "threeforks_large",
                              "belva_crater_large", "bell_island_large", "taylorfjellet_huge", "origny_large")
                        for s in keep)
    t = {"sites": {"a": {"n_waypoints": 5, "max_waypoint_separation_m": 31.0},
                   "b": {"n_waypoints": 4, "max_waypoint_separation_m": 90.0},
                   "c": {"n_waypoints": 9, "max_waypoint_separation_m": 30.0}}, "groups": {}}
    keep, left = m.qualifying_sites(["a", "b", "c"], t)
    assert keep == ["a"] and [x for x, _ in left] == ["b", "c"]


def test_best_candidate_moves_yaw_into_nr_cx(tmp_path, monkeypatch):
    import shutil
    nbc = _script("navcam_best_candidate")
    src = tmp_path / "cmods"
    src.mkdir()
    for f in ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json"):
        shutil.copy(CMODS / f, src / f)
    monkeypatch.setenv("MPPP_CMODS", str(src))
    rr = tmp_path / "rig_residuals.json"
    rr.write_text(json.dumps({"fits": {"yaw": {"vs_T": {"slope": -0.339, "se": 0.134}}}}))
    rep = nbc.build(rr, tmp_path / "out")
    fx = json.loads((src / "M2020_NR_fisheye_tangential.json").read_text())["params"][0]
    assert rep["added"] == pytest.approx(fx * np.radians(-0.339e-3))
    assert rep["form_ok"]
    nr = json.loads((tmp_path / "out" / "M2020_NR_fisheye_tangential.json").read_text())
    assert nr["thermal"]["cx_px_per_degC"] == pytest.approx(rep["NR_cx_slope_before"] + rep["added"])
    with pytest.raises(ValueError):                               # too weak a term is not moved
        rr.write_text(json.dumps({"fits": {"yaw": {"vs_T": {"slope": -0.1, "se": 0.134}}}}))
        nbc.build(rr, tmp_path / "out2")


def test_diagnose_visits_helper():
    from diagnose import _visit_of, _visits
    rows = [{"sol": s, "instrument": "NL"} for s in (91, 99, 101, 360, 371)]
    v = _visits(rows, {"navcam_rig_epochs_breaks": [300.0]})
    assert [x["sols"] for x in v] == [[91, 101], [360, 371]]
    assert _visit_of(365, v) == 1 and _visit_of(200, v) == 0


def test_zr_trace_and_report(tmp_path):
    """zr_report finds the step where an eye loses its frames, in the last run of the trace."""
    def head(step, zr):
        return {"head": {"step": step, "coverage": {"ZL034": {"frames": 8, "with_points": 8, "observations": 1},
                                                     "ZR034": {"frames": 8, "with_points": zr, "observations": 1}},
                         "points": 100}}

    def row(step, n, obs):
        return {"step": step, "name": n, "group": n[:2] + "034", "registered": True, "obs": obs, "keypoints": 50,
                "frame_id": 1, "rig_id": 1, "camera_id": 1, "cam": [1, 1, 1, 1, 0, 0], "C": [0, 0, 0],
                "q": [0, 0, 0, 1]}
    proj = tmp_path / "mars2020_sol_1408_south_arm_zcam" / "colmap_zrdiag"
    proj.mkdir(parents=True)
    lines = []
    for run in range(2):
        for step, zr in (("stage2_start", 0), ("r1_tri", 8), ("r2_ba", 8), ("r3_tri", 0 if run else 8)):
            lines.append(head(step, zr))
            lines += [row(step, f"ZL{i}", 100) for i in range(8)]
            lines += [row(step, f"ZR{i}", 100 if zr else 0) for i in range(8)]
    (proj / "zr_trace.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    out = tmp_path / "r.md"
    subprocess.run([sys.executable, str(HERE / "zr_report.py"), str(proj), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    t = out.read_text()
    assert "**ZR lost its frames at `r3_tri`** (after `r2_ba`: 8 -> 0)" in t and t.count("| r3_tri |") == 1
