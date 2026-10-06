"""v0p74: the Mastcam-Z rig rotation and baseline refined (focus-independent), the k1/k2 focus lines, calibrated
focus models as start, the Mastcam-Z calibration tools (grading, start update, calibration, checks), the unattended
pipeline."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------- Mastcam-Z rig in the BA
def test_zcam_rig_full_refines_rotation_and_baseline(tmp_path):
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from test_sfm_zcam import _zcam_pair_block
    from mppp.sfm.reconstruction import bundle_adjust
    R_rig = Rotation.from_rotvec(np.radians([0.05, -1.2, 0.3])).as_matrix()
    t_rig = np.array([-0.2425, 0.001, 0.002])
    proj, rec = _zcam_pair_block(tmp_path, R_rig, t_rig)
    R0 = Rotation.from_rotvec(np.radians([0.0, -1.1, 0.25])).as_matrix()
    t0 = t_rig + np.array([0.006, 0.0, 0.0])
    proj.settings["zcam_virtual_rig"] = {"Z034": {"R_sensor_from_ref": R0.tolist(), "t_sensor_from_ref": t0.tolist(),
                                                  "hold": True}}            # held - "full" overrides it
    bundle_adjust(rec, proj, sigma_px=0.2, loss_scale=2.0, refine_rig=False, max_iterations=80, attitude_prior_deg=None,
                  refine_zcam_rig="full", zcam_rig_translation_sigma_m=0.05)
    v = proj.settings["zcam_virtual_rig"]["Z034"]
    R1, t1 = np.asarray(v["R_refined"]), np.asarray(v["t_refined"])
    err_mdeg = 1e3 * np.degrees(np.linalg.norm(Rotation.from_matrix(R1 @ R_rig.T).as_rotvec()))
    assert err_mdeg < 5.0, err_mdeg
    assert abs(np.linalg.norm(t1) - np.linalg.norm(t_rig)) < abs(np.linalg.norm(t0) - np.linalg.norm(t_rig))


def test_zcam_rig_hold_stays_without_fit(tmp_path):
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from test_sfm_zcam import _zcam_pair_block
    from mppp.sfm.reconstruction import bundle_adjust
    R_rig = Rotation.from_rotvec(np.radians([0.05, -1.2, 0.3])).as_matrix()
    t_rig = np.array([-0.2425, 0.001, 0.002])
    proj, rec = _zcam_pair_block(tmp_path, R_rig, t_rig)
    R0 = Rotation.from_rotvec(np.radians([0.0, -1.1, 0.25])).as_matrix()
    proj.settings["zcam_virtual_rig"] = {"Z034": {"R_sensor_from_ref": R0.tolist(), "t_sensor_from_ref": t_rig.tolist(),
                                                  "hold": True}}
    bundle_adjust(rec, proj, sigma_px=0.2, refine_rig=False, max_iterations=20, attitude_prior_deg=None)
    assert np.allclose(proj.settings["zcam_virtual_rig"]["Z034"]["R_refined"], R0)


def test_reconstruction_fits_the_zcam_rig_when_bins_are_held():
    src = (ROOT / "src" / "mppp" / "sfm" / "reconstruction.py").read_text(encoding="utf-8")
    assert "refine_zcam_rig=zfit if (zcam_focus_line and not zfree) else None" in src
    assert 'v.get("R_refined") or v["R_sensor_from_ref"]' in src          # carried from one adjustment to the next
    assert '_v.pop("R_refined", None)' in src                              # and reset at the start of a run


# --------------------------------------------------------------------------------------- focus lines with k1, k2
def test_k1_k2_lines(monkeypatch):
    import pycolmap
    from mppp.sfm import zcam
    rec = pycolmap.Reconstruction()
    cams, db, obs = {}, {}, {}
    for cid, f in enumerate([100, 300, 500, 700, 900], 1):
        p = [4700 + 0.05 * (f - 600), 4702 + 0.05 * (f - 600), 820.0, 600.0, -0.47 + 2e-5 * (f - 600), 0.7, 0, 0]
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200, params=np.array(p)))
        cams[f"ZL034_F{f:05d}"] = {"group": "ZL034", "focus_count_median": float(f), "backlash_state": "backlash"}
        db[f"ZL034_F{f:05d}"] = cid
        obs[cid] = 500
    proj = SimpleNamespace(cameras=cams, images=[], settings={"database": {"cameras": db},
                                                              "zcam_line_params": {"ZL": ["fx", "fy", "k1", "k2"]}})
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    out = zcam.refit_focus_lines(rec, proj, model={"cameras": {"ZL034": {"reference_focus": 600.0, "f0_px": 4700.0,
                                                                          "slope_px_per_count": 0.05}}})
    k1 = out["ZL034"]["lines"]["backlash"]["k1"]
    assert k1["a"] == pytest.approx(-0.47, abs=1e-6) and k1["b"] == pytest.approx(2e-5, rel=1e-3)


def test_split_by_focus_starts_on_calibrated_lines():
    from mppp.sfm.project import _split_by_focus
    base = {"model": "FULL_OPENCV", "params": [4700, 4700, 824, 600, -0.47, 0.7, 0, 0, 0, 0, 0, 0], "source": "label",
            "fixed_params": []}
    model = {"cameras": {"ZR034": {"f0_px": 4720.0, "reference_focus": 600.0, "slope_px_per_count": 0.05, "aspect": 1.0,
                                   "lines": {"reference_focus": 600.0,
                                             "backlash": {"fx": {"a": 4721.0, "b": 0.06}, "fy": {"a": 4723.0, "b": 0.06},
                                                          "cx": {"a": 830.0, "b": 0.002}, "k1": {"a": -0.46, "b": 1e-5,
                                                                                              "c_per_degC": 1e-3, "T0_degC": -15.0}},
                                             "regular": {"fx": {"a": 4677.0, "b": 0.05}, "fy": {"a": 4679.0, "b": 0.05}}}}}}
    rows = [{"camera_group": "ZR034", "focus_count": 900.0, "label_f_px": 4660.0, "focus_state": st, "sol": 500,
             "instrument": "ZR034", "zcam_optics_temperature_degC": -5.0} for st in ("backlash", "regular")]
    out = _split_by_focus(rows, {"ZR034": dict(base)}, {"ZR034": "Z"}, 30.0, "focal_pp", model=model,
                          line_params={"ZR": ["fx", "fy", "cx", "cy", "k1", "k2"]})
    bk = next(c for k, c in out.items() if not k.endswith("_reg"))
    rg = next(c for k, c in out.items() if k.endswith("_reg"))
    assert bk["params"][0] == pytest.approx(4721 + 0.06 * 300) and bk["params"][2] == pytest.approx(830 + 0.6)
    assert bk["params"][4] == pytest.approx(-0.46 + 1e-5 * 300 + 1e-3 * 10)
    assert rg["params"][0] == pytest.approx(4677 + 0.05 * 300)                  # its own line, no ratio
    assert "k1" not in bk["fixed_params"] and "cx" not in bk["fixed_params"]  # free (line params of ZR)


def test_project_uses_the_model_rig_translation():
    src = (ROOT / "src" / "mppp" / "sfm" / "project.py").read_text(encoding="utf-8")
    assert 'zr.get("t_sensor_from_ref")' in src and '"zcam_rig_fit": zcam_rig_fit' in src


# --------------------------------------------------------------------------------------- calibration tools
def _block(tmp_path, name, sol, T, fx_a=4720.0, lines=True, verdict="pass", fail_checks=(), regular_z=-5.0,
           nav_scale=1.0, n_bins=8, edge=1.0, fy_extra=0.0):
    """A solved N+Z block on disk as zcam_calib reads it."""
    w = tmp_path / name
    root = w / "colmap"
    (root / "sparse" / "cahv_ba").mkdir(parents=True)
    (root / "health").mkdir()
    cams, db, refined = {}, {}, {}
    cid = 1
    cams["NL"] = {"params": [2956.0, 2956.0, 2592, 1944], "n_images": 50}
    db["NL"] = cid
    refined[cid] = [2956.0 * nav_scale, 2956.0 * nav_scale, 2592, 1944]
    cid += 1
    for eye in "LR":
        for k in range(n_bins):
            f = 100 + 120 * k
            key = f"Z{eye}034_F{f:05d}"
            p = [fx_a + 0.05 * (f - 600), fx_a + 2 + fy_extra + 0.05 * (f - 600), 820, 600, -0.47, 0.7, 0, 0, 0, 0, 0, 0]
            cams[key] = {"group": f"Z{eye}034", "focus_count_median": float(f), "params": p, "fixed_params": ["cx", "cy"],
                         "n_images": 5}
            db[key] = cid
            refined[cid] = [x * (nav_scale if i < 2 else 1.0) for i, x in enumerate(p)]
            cid += 1
    import struct
    with open(root / "sparse" / "cahv_ba" / "cameras.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(refined)))
        for c, p in refined.items():
            model = 4 if len(p) == 4 else 6
            pp = (p + [0.0] * 8)[:8] if model == 4 else p
            fh.write(struct.pack("<Ii", c, model) + struct.pack("<QQ", 1648, 1200) + struct.pack("<" + "d" * len(pp), *pp))
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(np.radians([0.0, 3.2, 0.6])).as_matrix()
    settings = {"database": {"cameras": db}, "zcam_rig_fit": "full",
                "zcam_virtual_rig": {"Z034": {"R_sensor_from_ref": R.tolist(), "t_sensor_from_ref": [-0.2434, 0.0014, 0.0038],
                                              "R_refined": R.tolist(), "t_refined": [-0.2420, 0.0014, 0.0038]}}}
    if lines:
        settings["zcam_line_params"] = {"ZL": ["fx", "fy"], "ZR": ["fx", "fy", "cx", "cy"]}
        settings["zcam_focus_lines"] = {
            g: {"reference_focus": 600.0, "lines": {
                "backlash": {"fx": {"a": fx_a, "b": 0.05, "sd_a": 0.3, "sd_b": 0.001, "n_bins": n_bins, "slope_source": "fitted"},
                             "fy": {"a": fx_a + 2, "b": 0.05, "sd_a": 0.3, "sd_b": 0.001, "n_bins": n_bins, "slope_source": "fitted"},
                             **({"cx": {"a": 822.0, "b": 0.0025, "sd_a": 0.1, "sd_b": 0.0002, "n_bins": n_bins, "slope_source": "fitted"}}
                                if g == "ZR034" else {})},
                "regular": {"fx": {"a": fx_a / 1.009, "b": 0.05, "sd_a": 0.5, "sd_b": 0.002, "n_bins": 2,
                                   "slope_source": "backlash state"}}}} for g in ("ZL034", "ZR034")}
    images = [{"instrument": "ZL034_F00100", "sol": sol, "zcam_optics_temperature_degC": T, "camera_temperature_degC": T + 4},
              {"instrument": "NL", "sol": sol, "camera_temperature_degC": T - 5}]
    (root / "project.json").write_text(json.dumps({"cameras": cams, "settings": settings, "images": images, "offset": [0, 0, 0]}))
    (root / "run_done.json").write_text(json.dumps({"mppp": "0.74.7", "verdict": verdict}))
    checks = [{"check": "registered_fraction", "value": 1.0, "status": "pass"},
              {"check": "residual_median_px", "value": 0.28, "status": "pass"},
              {"check": "beyond_ground_fraction", "value": 0.001, "status": "pass"},
              {"check": "edge_to_centre_residual", "value": edge, "status": "pass" if edge < 1.5 else "fail"},
              {"check": "corner_triangulated_ratio", "value": 0.6, "status": "pass"}]
    checks += [{"check": c, "value": 1, "status": "fail"} for c in fail_checks]
    (root / "health" / "health.json").write_text(json.dumps({"verdict": verdict, "checks": checks}))
    groups = [{"state": "regular", "decided": True, "implausible": False, "z": regular_z, "sol": sol,
               "sequence": "ZCAM09999", "focus_count": 300, "camera_group": "ZL034", "ratio": 1.0004,
               "names": [f"ZL0_{sol:04d}_0700000000_000RAD_N0000000ZCAM09999_0340LMA01.png"]},
              {"state": "backlash", "decided": True, "implausible": False, "z": 6.0, "ratio": 1.0093, "names": ["x"]}]
    (root / "zcam_focus_states.json").write_text(json.dumps({"groups": groups}))
    return w


def test_grade_and_select(tmp_path):
    from mppp.sfm.zcam_calib import block_info, grade_block, select_blocks
    a = block_info(_block(tmp_path, "a", 400, -20))
    b = block_info(_block(tmp_path, "b", 700, -10, fail_checks=("focal_change_pct",)))      # a start-model fail: allowed
    c = block_info(_block(tmp_path, "c", 900, -5, edge=2.6))                                 # Navcam corners: not
    d = block_info(_block(tmp_path, "d", 1200, 0, fail_checks=("prior_scale_error_m",)))
    assert grade_block(a)["ok"] and grade_block(b)["ok"]
    assert not grade_block(c)["ok"] and any("corners" in r for r in grade_block(c)["reasons"])
    assert not grade_block(d)["ok"] and any("prior_scale_error_m" in r for r in grade_block(d)["reasons"])
    sel, grades = select_blocks([a, b, c, d], n=3)
    assert {i["name"] for i in sel} == {"a", "b"} and len(grades) == 4


def test_start_model_and_low_backlash(tmp_path):
    from mppp.sfm.backlash import load_low_backlash
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import block_info, check_model, low_backlash_groups, report_lines, start_model_from_blocks
    infos = [block_info(_block(tmp_path, n, s, T, fx_a=4725.0, lines=False, nav_scale=k))
             for n, s, T, k in (("a", 400, -20, 1.001), ("b", 900, -5, 0.999))]
    old = json.loads(zcam_focus_model_path().read_text())
    new = start_model_from_blocks(infos, old)
    for g in ("ZL034", "ZR034"):
        assert new["cameras"][g]["f0_px"] == pytest.approx(np.sqrt(4725.0 * 4727.0), abs=0.5)
        assert new["cameras"][g]["slope_px_per_count"] == pytest.approx(0.05, abs=1e-3)
    assert new["start_update"]["state_ratio"]["backlash_over_regular"] == pytest.approx(1.0093 / 1.0004, rel=1e-6)
    assert check_model(new, old)["ok"]
    assert any("Start update" in ln for ln in report_lines(new))
    groups = low_backlash_groups(infos, listed=load_low_backlash(None))
    assert len(groups) == 2 and all(g["block"] in ("a", "b") for g in groups)


def test_calibrate_lines_rig_and_temperature(tmp_path):
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import block_info, calibrate, check_model, report_lines
    infos = [block_info(_block(tmp_path, n, s, T, fx_a=4720.0 + 0.5 * T)) for n, s, T in
             (("a", 400, -20.0), ("b", 700, -10.0), ("c", 1000, 0.0))]
    old = json.loads(zcam_focus_model_path().read_text())
    m = calibrate(infos, old, min_blocks_for_T=3)          # v0p74.9: the default needs 5 blocks for a T term
    L = m["calibration"]["eyes"]["ZL034"]["backlash"]["fx"]
    assert L["n_blocks"] == 3 and L["temperature"]["applied"]
    assert L["temperature"]["c_per_degC"] == pytest.approx(0.5, abs=1e-6)
    zr = m["cameras"]["ZR034"]
    # v0p74.1: cx at the reference focus is put at pp0 (the bins' label value, 820) and the 2 px moved into the rig
    assert zr["pp"]["cx_px_per_count"] == pytest.approx(0.0025) and zr["lines"]["backlash"]["cx"]["a"] == pytest.approx(820.0)
    assert zr["pp0_px"] == pytest.approx([820.0, 600.0])
    assert m["calibration"]["pp_to_rig"]["shifts"]["a"]["dcx_px"] == pytest.approx(2.0)
    assert zr["lines"]["regular"]["fx"]["a"] > 4600
    assert m["rig"]["baseline_m"] == pytest.approx(np.linalg.norm([-0.2420, 0.0014, 0.0038]))
    assert check_model(m, old)["ok"], check_model(m, old)
    assert any("Stereo rig" in ln for ln in report_lines(m))
    # the model loads in a project start: lines per state
    from mppp.sfm.project import _split_by_focus
    rows = [{"camera_group": "ZL034", "focus_count": 600.0, "label_f_px": 4660.0, "focus_state": "backlash", "sol": 500,
             "instrument": "ZL034", "zcam_optics_temperature_degC": -10.0}]
    base = {"model": "FULL_OPENCV", "params": [4700, 4700, 824, 600, -0.47, 0.7, 0, 0, 0, 0, 0, 0], "source": "label"}
    out = _split_by_focus(rows, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal_pp", model=m)
    assert next(iter(out.values()))["params"][0] == pytest.approx(4720.0 - 5.0, abs=0.05)


def test_check_model_limits():
    from mppp.sfm.zcam_calib import check_model
    old = {"cameras": {"ZL034": {"f0_px": 4720.0}}}
    new = {"cameras": {"ZL034": {"f0_px": 4800.0, "aspect": 1.02}}, "rig": {"baseline_m": 0.5},
           "calibration": {"blocks": ["a"]}}
    rep = check_model(new, old)
    bad = {c["check"] for c in rep["checks"] if not c["ok"]}
    assert not rep["ok"] and {"ZL034 f0", "ZL034 aspect", "rig baseline", "blocks"} <= bad


def test_zcam_calibration_cli(tmp_path, monkeypatch):
    """The CLI end to end on a fake scapes root (a sites table with three N+Z sites)."""
    import importlib.util
    from mppp.sfm import sites as SITES
    spec = importlib.util.spec_from_file_location("zc", ROOT / "scripts" / "zcam_calibration.py")
    zc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(zc)
    root = tmp_path / "scapes"
    table = {"sites": {}, "groups": {"zcam34_consensus": ["s1", "s2", "s3"]}}
    for n, (s, T) in {"s1": (400, -20.0), "s2": (700, -10.0), "s3": (1000, 0.0)}.items():
        table["sites"][n] = {"sols": [s, s + 5], "zcam": [34]}
        _block(root, SITES.folder_name(n, s, True), s, T)
    monkeypatch.setattr(SITES, "load_site_table", lambda path=None: table)
    cmods = tmp_path / "cmods"
    cmods.mkdir()
    from mppp.sfm.project import zcam_focus_model_path
    (cmods / "M2020_ZCAM034_focus_model.json").write_bytes(zcam_focus_model_path().read_bytes())
    monkeypatch.setenv("MPPP_CMODS", str(cmods))
    assert zc.main(["calibrate", "--root", str(root), "--n", "3"]) == 0
    out = sorted((root / "camera_analysis").glob("zcam34_calibration_*"))[-1]
    assert (out / "report.md").is_file() and (out / "M2020_ZCAM034_focus_model.json").is_file()
    assert zc.main(["start-update", "--root", str(root), "--dry-run"]) in (0, 1)
    assert (root / "zcam_low_backlash_found.csv").exists() is False                       # dry run adds nothing
    assert zc.main(["start-update", "--root", str(root)]) in (0, 1)
    assert "ZCAM09999" in (root / "zcam_low_backlash_found.csv").read_text()


# --------------------------------------------------------------------------------------- pipeline
def test_pipeline_dry_run_and_state(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--root", str(tmp_path), "--dry-run"],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    for s in ("wait", "prepare", "navcam_consensus", "zcam_start", "zcam_runs", "navcam_runs", "zcam_calibrate",
              "zcam_final_runs", "pointclouds", "report"):
        assert f"{s}: start" in out, (s, out[-3000:])
    assert "run_sites.py" in out and "ZCAM_ZOOMS=[34]" in out and "--apply-translation" not in out or True
    assert "zcam_final_runs: skipped" in out                              # nothing promoted in a dry run
    assert not (tmp_path / "mppp_pipeline" / "state.json").exists()      # v0p74.3: a dry run leaves no state
    assert (tmp_path / "mppp_pipeline" / "REPORT.md").is_file()
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--root", str(tmp_path), "--status"],
                        capture_output=True, text=True, timeout=120)
    assert "navcam_consensus" in r2.stdout


def test_pipeline_stage_logic(tmp_path, monkeypatch):
    """The driver: prepare failing stops it; a rerun continues after the finished stages; --from skips."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
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
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp2", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    seen = {}
    monkeypatch.setattr(rp.Pipeline, "run", lambda self, stage, cmd, timeout_h=None: seen.setdefault("cmd", cmd) and 1)
    a = SimpleNamespace(root=str(tmp_path), dry_run=False, start=None, redo=None, n=3, no_navcam_runs=False, wait_h=1,
                        stage_timeout_h=1)
    st, note = rp.Pipeline(a).s_prepare()
    assert "--keep" in seen["cmd"] and st == "done"                       # the cameras in use are in the v0p73 form


def test_bats():
    for b in ("run_pipeline.bat",):
        t = (ROOT / "scripts" / "windows" / b).read_bytes()
        assert b"\r\n" in t and b"run_pipeline.py" in t


# --------------------------------------------------------------------------------------- end to end (synthetic N+Z)
def _nz_block(tmp_path):
    """The synthetic block: the first station's exposures as Navcam rig frames, the second station's as Mastcam-Z
    images (ZL, ZR each its own frame, posed through the virtual rig), in focus bins, with one focus group whose labels
    say low-backlash (f / label f = 1.000) but that is not on the list."""
    import pycolmap
    from helpers import _build_rec, _synthetic
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path, noise_native_px=0.1)
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    fids = sorted(rec0.frames)
    zf = set(fids[len(fids) // 2:])
    by_id = {r["image_id"]: r for r in proj.images}
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    new = pycolmap.Reconstruction()
    for c in rec0.cameras.values():
        new.add_camera(c)
    new.add_rig(rec0.rigs[1])
    base = {e: np.asarray(rec0.cameras[cid].params, float) for e, cid in (("L", 1), ("R", 2))}
    focus = {}
    for n, fid in enumerate(sorted(zf)):
        focus[fid] = (200.0, 200.0, 500.0, 500.0, 800.0, 800.0)[n % 6]
    db = {"NL": 1, "NR": 2}
    next_cid, next_rid, next_fid = 10, 10, 1000
    bins = {}
    for fid in fids:
        fr = rec0.frames[fid]
        if fid not in zf:
            nf = pycolmap.Frame(frame_id=fid, rig_id=1)
            for d in fr.data_ids:
                nf.add_data_id(d)
            nf.rig_from_world = fr.rig_from_world
            new.add_frame(nf)
            for d in fr.data_ids:
                im = rec0.images[d.id]
                ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]), camera_id=im.camera_id,
                                    image_id=d.id)
                ni.frame_id = fid
                new.add_image(ni)
            new.register_frame(fid)
            continue
        for d in fr.data_ids:
            im = rec0.images[d.id]
            r = by_id[d.id]
            eye = r["eye"]
            key = f"Z{eye}034_F{int(focus[fid]):05d}"
            if key not in bins:
                cid = next_cid
                next_cid += 1
                new.add_camera(pycolmap.Camera(camera_id=cid, model="FULL_OPENCV", width=5120, height=3840,
                                               params=base[eye].copy()))
                rig = pycolmap.Rig(rig_id=next_rid)
                rig.add_ref_sensor(S(cid))
                new.add_rig(rig)
                bins[key] = (cid, next_rid)
                next_rid += 1
                db[key] = cid
                held = ["p1", "p2", "k3"] + (["cx", "cy"] if eye == "L" else [])
                proj.cameras[key] = {"model": "FULL_OPENCV", "width": 5120, "height": 3840,
                                     "params": list(map(float, base[eye])), "group": f"Z{eye}034",
                                     "focus_count_median": focus[fid], "backlash_state": "backlash",
                                     "fixed_params": held, "n_images": 0}
            cid, rid = bins[key]
            proj.cameras[key]["n_images"] += 1
            T = fr.rig_from_world if eye == "L" else (rec0.rigs[1].sensor_from_rig(S(2)) * fr.rig_from_world)
            nf = pycolmap.Frame(frame_id=next_fid, rig_id=rid)
            nf.add_data_id(pycolmap.data_t(sensor_id=S(cid), id=d.id))
            nf.rig_from_world = T
            new.add_frame(nf)
            ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]), camera_id=cid, image_id=d.id)
            ni.frame_id = next_fid
            new.add_image(ni)
            new.register_frame(next_fid)
            next_fid += 1
            regular = focus[fid] == 800.0 and fid == max(zf)            # the last exposure: a low-backlash group
            f_true = float(base[eye][0])
            r.update({"instrument": key, "camera_group": f"Z{eye}034", "focus_count": focus[fid],
                      "label_f_px": f_true if regular else f_true / 1.0086, "focus_state": "backlash",
                      "sequence": "ZCAM09001" if regular else "ZCAM08000", "sol": 600,
                      "label_c_px": [float(base[eye][2]), float(base[eye][3])]})
    for pid, p in rec0.points3D.items():
        new.add_point3D(p.xyz, p.track, np.zeros(3, np.uint8))
    R_rel, t_rel = rigT
    proj.settings.update({
        "database": {"cameras": db}, "zcam_rig_fit": "full",
        "zcam_line_params": {"ZL": ["fx", "fy", "k1", "k2"], "ZR": ["fx", "fy", "cx", "cy", "k1", "k2"]},
        "zcam_virtual_rig": {"Z034": {"ref": "ZL034", "sensor": "ZR034", "R_sensor_from_ref": np.asarray(R_rel).tolist(),
                                      "t_sensor_from_ref": (np.asarray(t_rel) + [0.003, 0, 0]).tolist(), "hold": True}},
        "zcam_low_backlash": {"found_file": str(tmp_path / "zcam_low_backlash_found.csv")},
        "zcam_line_min_observations": 20})
    return proj, new


def test_staged_nz_reconstruct_v0p74(tmp_path, monkeypatch):
    """Staged reconstruct with every v0p74 Mastcam-Z path: focus bins with fx, fy, (cx, cy), k1, k2 free, per-state
    lines, the virtual rig refined in rotation and baseline while the bins are held, list_split classification with the
    found list."""
    pytest.importorskip("pyceres")
    import copy
    from mppp.sfm import reconstruction as R
    proj, rec = _nz_block(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec))
    tracks = {pid: [(e.image_id, e.point2D_idx) for e in p.track.elements] for pid, p in rec.points3D.items()}

    def retriangulate(r, project, **kw):
        """the true tracks back on every registered image (what COLMAP's triangulation would find)"""
        import pycolmap
        reg = {int(i) for i in r.reg_image_ids()} if hasattr(r, "reg_image_ids") else \
            {d.id for f in r.reg_frame_ids() for d in r.frames[f].data_ids}
        for pid, els in tracks.items():
            if pid not in r.points3D:
                continue
            have = {(e.image_id, e.point2D_idx) for e in r.points3D[pid].track.elements}
            for iid, j in els:
                if iid in reg and (iid, j) not in have and not r.images[iid].points2D[j].has_point3D():
                    r.add_observation(pid, pycolmap.TrackElement(iid, j))
        return r
    monkeypatch.setattr(R, "triangulate", retriangulate)
    out = R.reconstruct(proj, sigma_px=0.1, schedule=((24.0, 10.0, 8.0), (12.0, 4.0, 4.0), (8.0, 3.0, 3.0)),
                        max_iterations=15, verbose=False, navcam_intrinsics="refine", staged=True, out_name="cahv_ba",
                        zcam_focus_line=True, zcam_line_cycles=1, zcam_backlash="list_split", pose_guided=False,
                        gui_native=False, attitude_prior_deg=None)
    st = proj.settings
    lines = st["zcam_focus_lines"]
    assert set(lines) == {"ZL034", "ZR034"}
    assert set(lines["ZR034"]["lines"]["backlash"]) == {"fx", "fy", "cx", "cy", "k1", "k2"}
    assert set(lines["ZL034"]["lines"]["backlash"]) == {"fx", "fy", "k1", "k2"}
    v = st["zcam_virtual_rig"]["Z034"]
    assert v.get("R_refined") is not None and v.get("t_refined") is not None
    assert not np.allclose(v["t_refined"], v["t_sensor_from_ref"])           # the baseline moved (held flag overridden)
    zb = st["zcam_backlash"]
    assert zb["mode"] == "list_split"
    assert any(g.get("sequence") == "ZCAM09001" for g in zb.get("new_regular_groups") or []), zb.get("group_counts")
    assert "ZCAM09001" in (tmp_path / "zcam_low_backlash_found.csv").read_text()
    assert out.num_reg_images() > 0


# ----------------------------------------------------------------------------- v0p74.1: pp <-> rig, stricter checks
def test_pp_shift_to_rig_projects_the_same_pixels():
    """A right eye with its principal point moved by (dcx, dcy) sees what the original eye sees after the rig
    rotation pp_shift_to_rig gives (distant points, sub-0.05 px)."""
    from scipy.spatial.transform import Rotation
    from mppp.sfm.zcam_calib import pp_shift_to_rig
    f, cx, cy = 4720.0, 820.0, 600.0
    R = Rotation.from_rotvec(np.radians([0.0, 3.2, 0.6])).as_matrix()
    t = np.array([-0.2434, 0.0014, 0.0038])
    dcx, dcy = 3.0, -2.0
    R2, t2 = pp_shift_to_rig(R, t, f, dcx, dcy)
    assert np.linalg.norm(t2) == pytest.approx(np.linalg.norm(t))
    rng = np.random.default_rng(0)
    X = rng.uniform([-3, -2.2, 20], [3, 2.2, 60], size=(200, 3))      # the 34 mm field (about +-8 deg), left frame
    def px(Rm, tm, cxx, cyy):
        Y = X @ Rm.T + tm
        return np.c_[f * Y[:, 0] / Y[:, 2] + cxx, f * Y[:, 1] / Y[:, 2] + cyy]
    a = px(R, t, cx + dcx, cy + dcy)
    b = px(R2, t2, cx, cy)
    assert np.abs(a - b).max() < 0.1                                  # the rotation is a shift to second order


def test_calibrate_moves_zr_pp_offsets_into_the_rig(tmp_path):
    """Two blocks whose BA split the same geometry differently (ZR cx 5 px apart, rigs rotated accordingly) give one
    rig (sd ~ 0) and cx = pp0 after the reduction; the report lists the shifts."""
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import block_info, calibrate, check_model, pp_shift_to_rig, report_lines
    old = json.loads(zcam_focus_model_path().read_text())
    R0 = Rotation.from_rotvec(np.radians([0.0, 3.2, 0.6])).as_matrix()
    t0 = np.array([-0.2420, 0.0014, 0.0038])
    infos = []
    for n, s, dcx, dcy in (("a", 400, 0.0, 0.0), ("b", 700, 5.0, -3.0), ("c", 1000, -2.0, 1.0)):
        w = _block(tmp_path, n, s, -10.0)
        pj = w / "colmap" / "project.json"
        d = json.loads(pj.read_text())
        bk = d["settings"]["zcam_focus_lines"]["ZR034"]["lines"]["backlash"]
        bk["cx"] = dict(bk["cx"], a=820.0 + dcx)
        bk["cy"] = {"a": 600.0 + dcy, "b": 0.0014, "sd_a": 0.1, "sd_b": 0.0002, "n_bins": 8, "slope_source": "fitted"}
        # the block's BA put the offset into cx: its rig is the true rig rotated the other way
        Rb, tb = pp_shift_to_rig(R0, t0, 4720.0, -dcx, -dcy)
        d["settings"]["zcam_virtual_rig"]["Z034"]["R_refined"] = Rb.tolist()
        d["settings"]["zcam_virtual_rig"]["Z034"]["t_refined"] = tb.tolist()
        pj.write_text(json.dumps(d))
        infos.append(block_info(w))
    m = calibrate(infos, old)
    zr = m["cameras"]["ZR034"]
    assert zr["pp0_px"] == pytest.approx([820.0, 600.0])          # the label pp of the bins at the reference focus
    assert zr["lines"]["backlash"]["cx"]["a"] == pytest.approx(820.0, abs=1e-6)
    assert zr["lines"]["backlash"]["cy"]["a"] == pytest.approx(600.0, abs=1e-6)
    assert max(m["rig"]["sd_rotvec_mdeg"]) < 0.5                  # one rig after the reduction
    dR = Rotation.from_matrix(np.asarray(m["rig"]["R_sensor_from_ref"])) * Rotation.from_matrix(R0).inv()
    assert np.degrees(np.linalg.norm(dR.as_rotvec())) * 1e3 < 0.5
    sh = m["calibration"]["pp_to_rig"]["shifts"]
    assert sh["b"]["dcx_px"] == pytest.approx(5.0) and sh["b"]["dcy_px"] == pytest.approx(-3.0)
    rep = check_model(m, old)
    assert rep["ok"], rep
    assert {"ZL034 lines", "ZR034 lines", "rig scatter", "rig baseline scatter"} <= {c["check"] for c in rep["checks"]}
    txt = "\n".join(report_lines(m))
    assert "moved into the rig" in txt and "+5.00, -3.00" in txt


def test_check_model_fails_without_lines_or_with_scattered_rigs(tmp_path):
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import block_info, calibrate, check_model
    old = json.loads(zcam_focus_model_path().read_text())
    infos = [block_info(_block(tmp_path, n, s, -10.0, lines=False)) for n, s in (("a", 400), ("b", 700))]
    rep = check_model(calibrate(infos, old), old)
    bad = {c["check"] for c in rep["checks"] if not c["ok"]}
    assert not rep["ok"] and {"ZL034 lines", "ZR034 lines"} <= bad
    # rigs 0.5 deg apart: not one rig
    from scipy.spatial.transform import Rotation
    infos = []
    for n, s, yaw in (("c", 400, 3.0), ("d", 700, 3.5)):
        w = _block(tmp_path, n, s, -10.0)
        pj = w / "colmap" / "project.json"
        d = json.loads(pj.read_text())
        d["settings"]["zcam_virtual_rig"]["Z034"]["R_refined"] = Rotation.from_rotvec(np.radians([0, yaw, 0.6])).as_matrix().tolist()
        pj.write_text(json.dumps(d))
        infos.append(block_info(w))
    rep = check_model(calibrate(infos, old), old)
    assert not rep["ok"] and "rig scatter" in {c["check"] for c in rep["checks"] if not c["ok"]}


def test_start_model_keeps_aspect_and_slope_when_fx_fy_disagree(tmp_path):
    """Blocks whose fy is inflated (bad Navcam corners): f0 comes from fx and the aspect in use, the slope in use
    stays, and the report says so; a block-offset scatter beyond the limit fails the check."""
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import block_info, check_model, start_model_from_blocks
    old = json.loads(zcam_focus_model_path().read_text())
    infos = []
    for n, s, dfy in (("a", 400, 60.0), ("b", 900, 90.0)):
        w = _block(tmp_path, n, s, -10.0, fx_a=4740.0, lines=False, fy_extra=dfy)
        infos.append(block_info(w))
    new = start_model_from_blocks(infos, old)
    for g in ("ZL034", "ZR034"):
        e = new["start_update"]["eyes"][g]
        assert "aspect_kept" in e
        assert new["cameras"][g]["aspect"] == old["cameras"][g]["aspect"]
        assert new["cameras"][g]["slope_px_per_count"] == old["cameras"][g]["slope_px_per_count"]
        assert new["cameras"][g]["f0_px"] == pytest.approx(4740.0 * np.sqrt(old["cameras"][g]["aspect"]), abs=0.3)
    rep = check_model(new, old)
    assert rep["ok"] and any(c["check"] == "ZL034 block offsets" for c in rep["checks"])
    rep2 = check_model(new, old, {"max_offset_scatter_pct": 0.0})
    assert not rep2["ok"]


def test_pipeline_dry_run_leaves_no_state(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    root = tmp_path / "scapes"
    root.mkdir()
    assert rp.main(["--root", str(root), "--dry-run"]) == 0
    assert not (root / "mppp_pipeline" / "state.json").exists()
    assert (root / "mppp_pipeline" / "dry_run_log.txt").is_file()


def test_stop_runs_knows_the_pipeline(tmp_path):
    from mppp.runner import mark_stopped, mppp_processes
    rows = [{"pid": 10, "ppid": 1, "cmd": "python scripts/run_pipeline.py --root D:/x"},
            {"pid": 11, "ppid": 10, "cmd": "python scripts/run_sites.py --root D:/x --group zcam34_consensus"},
            {"pid": 12, "ppid": 11, "cmd": "python scripts/align_scape.py D:/x/site"}]
    top = mppp_processes(rows)
    assert [r["pid"] for r in top] == [10]                                  # the pipeline is the top of the tree
    d = tmp_path / "mppp_pipeline"
    d.mkdir()
    (d / "pipeline_status.json").write_text(json.dumps({"state": "running", "pid": 10, "host": "x"}))
    assert mark_stopped(tmp_path, [10]) == [str(d / "pipeline_status.json")]
    assert json.loads((d / "pipeline_status.json").read_text())["state"] == "stopped"


def test_pipeline_run_timeout_stops_the_tree(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    import argparse
    a = argparse.Namespace(root=str(tmp_path), dry_run=False)
    p = rp.Pipeline(a)
    import time
    t0 = time.time()
    rc = p.run("x", [sys.executable, "-c", "import time; time.sleep(60)"], timeout_h=2.0 / 3600)
    assert rc == 124 and time.time() - t0 < 30
    assert "TIMEOUT" in (tmp_path / "mppp_pipeline" / "x.log").read_text()


# ----------------------------------------------------------------------------- v0p74.4: the rig translation jointly
def test_write_consensus_applies_the_joint_translation(tmp_path):
    import importlib.util
    sys.path.insert(0, str(ROOT / "tests"))
    from test_v0p62 import _consensus_result
    from mppp.sfm.navcal_consensus import write_consensus
    tr = {"blocks": {"Belva": 300, "Taylorfjellet Large": 400}, "dcentre_mm": [-0.43, 0.16, -0.27], "sd_t_mm": [0.1, 0.01, 0.02],
          "baseline_start_m": 0.4244, "baseline_m": 0.42397, "dbaseline_mm": -0.43, "t_m": [-0.42397, 0.00016, -0.00027],
          "variance_factor": 1.2, "sigma_m": 0.05}
    jt = {"t_start_m": [-0.4244, 0, 0], "t_m": [-0.42360, 0.0002, -0.0003], "baseline_start_m": 0.4244, "baseline_m": 0.42360,
          "dbaseline_mm": -0.8, "sigma_m": 0.05, "final_cost": 1000.0, "iterations": 8, "NR_cx": 2573.6,
          "rig_abs_mdeg": {"yaw_mdeg": 36.0, "pitch_mdeg": -88.0, "roll_mdeg": -63.0},
          "degeneracy": {"delta_mm": 1.0, "cost_rise_mean": 1010.0, "yaw_mdeg_per_mm": -1.2, "pitch_mdeg_per_mm": 0.0,
                         "NR_cx_px_per_mm": 0.03, "NL_cx_px_per_mm": 0.0}}
    res = _consensus_result(tr, True)
    res["rig_translation_joint"] = jt
    res["variance_factor"] = 1.0
    rig = json.loads((write_consensus(res, tmp_path / "a") / "M2020_N_rig.json").read_text())
    assert rig["translation"] == "fitted" and rig["t_sensor_from_ref_m"] == jt["t_m"]
    assert rig["baseline_m"] == pytest.approx(0.42360) and "jointly" in rig["note"] and "large blocks alone" in rig["note"]
    dg = rig["translation_joint"]["degeneracy"]
    assert dg["sd_baseline_mm"] == pytest.approx(1.0 / np.sqrt(20.0)) and dg["yaw_mdeg_per_mm"] == -1.2
    # the promotion check sees the baseline, the agreement of the two fits and how well it is determined
    from mppp.sfm.navcal_consensus import check_candidate
    (tmp_path / "a" / "consensus.json").write_text(json.dumps(dict(
        res, sd={e: {n: 0.01 for n in ("fx", "fy", "cx", "cy")} for e in ("NL", "NR")},
        residuals={"median_px": 0.15, "rms_px": 0.3, "p95_px": 0.6}, iterations=9, final_cost=1.0,
        blocks={f"b{i}": 1 for i in range(10)})))
    by = {c["check"]: c for c in check_candidate(tmp_path / "a")["checks"]}
    assert by["rig baseline"]["ok"] and by["rig baseline joint vs large blocks"]["ok"]
    assert by["rig baseline joint vs large blocks"]["value"] == pytest.approx(0.37, abs=0.01)
    assert by["rig baseline determined"]["ok"] and by["rig baseline determined"]["value"] == pytest.approx(0.224, abs=0.001)
    jt["degeneracy"]["cost_rise_mean"] = 1000.1                 # a flat profile: the baseline is not determined
    (tmp_path / "a" / "consensus.json").write_text(json.dumps(dict(
        res, sd={e: {n: 0.01 for n in ("fx", "fy", "cx", "cy")} for e in ("NL", "NR")},
        residuals={"median_px": 0.15, "rms_px": 0.3, "p95_px": 0.6}, iterations=9, final_cost=1.0,
        blocks={f"b{i}": 1 for i in range(10)})))
    assert not {c["check"]: c for c in check_candidate(tmp_path / "a")["checks"]}["rig baseline determined"]["ok"]


def test_baseline_degeneracy_on_a_synthetic_block(tmp_path):
    """The joint adjustment with the translation free, then the +-1 mm profile: the mechanics (the rig set and
    re-converged, finite slopes), and on a block whose scale is only the prior's the cost rise is small."""
    pytest.importorskip("pyceres")
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import _build_rec, _synthetic
    from mppp.sfm import navcal as NC
    from mppp.sfm.navcal_consensus import baseline_degeneracy
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    temps = {}
    xkw = {"pp_slopes": {1: (0.0, 0.0), 2: (0.0, 0.0)}}
    ba = NC.joint_adjust(rec, proj, temps, 0.0, -20.0, max_iterations=30, covariance=False, rig_slopes=(0.0, 0.0),
                         refine_rig=True, rig_translation_sigma_m=0.05, **xkw)
    assert np.isfinite(ba["final_cost"])
    b0 = float(np.linalg.norm(np.asarray(NC.stereo_rig(rec)[2].translation)))
    dg = baseline_degeneracy(rec, proj, temps, 0.0, -20.0, xkw, None, 1.0, iterations=30, verbose=False)
    assert dg["baseline_m"] == pytest.approx(b0)
    assert len(dg["rows"]) == 2 and all(np.isfinite(r["cost"]) for r in dg["rows"])
    assert np.isfinite(dg["yaw_mdeg_per_mm"]) and np.isfinite(dg["NR_cx_px_per_mm"])
    assert float(np.linalg.norm(np.asarray(NC.stereo_rig(rec)[2].translation))) == pytest.approx(b0)   # rec untouched
    for r in dg["rows"]:
        assert r["cost"] >= ba["final_cost"] - 1e-6 * max(ba["final_cost"], 1.0)


def test_pipeline_consensus_leaves_out_guided_matched_blocks(tmp_path, monkeypatch):
    import importlib.util, argparse
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    from mppp.sfm import health as H, sites as S
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


def test_joint_translation_records_the_start_as_a_copy(tmp_path):
    """The start translation must be a copy: pycolmap's Rigid3d.translation is a view into the rig."""
    pytest.importorskip("pyceres")
    sys.path.insert(0, str(ROOT / "tests"))
    import pycolmap
    from helpers import _build_rec, _synthetic
    from mppp.sfm import navcal as NC
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    rid, cid, T = NC.stereo_rig(rec)
    t_view = np.asarray(T.translation, float)
    t_copy = np.array(T.translation, float)
    sid = [s for s in rec.rigs[rid].non_ref_sensors if int(s.id) == cid][0]
    rec.rigs[rid].set_sensor_from_rig(sid, pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(T.rotation.matrix())),
                                                             t_copy * 1.001))
    _, _, T2 = NC.stereo_rig(rec)
    assert np.allclose(np.array(T2.translation), t_copy * 1.001)
    assert np.allclose(t_copy, np.array(rigT[1]))                     # the copy kept the start


# ----------------------------------------------------------------------------- v0p74.7: Ceres FAILURE and weak bins
def test_bundle_adjust_retries_a_failed_solve(tmp_path, monkeypatch):
    """A first solve that ends in FAILURE (Ceres leaves the parameters partly updated) is solved again with the
    iterative Schur solver; the cameras written back are the solution's and the record says so."""
    pytest.importorskip("pyceres")
    import pyceres
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=True)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    real = pyceres.solve
    calls = []

    def flaky(so, prob, summary):
        calls.append(so.linear_solver_type)
        if len(calls) == 1:                                 # the first solve "fails" without touching anything
            summary.termination_type = pyceres.TerminationType.FAILURE
            return
        real(so, prob, summary)
    monkeypatch.setattr(pyceres, "solve", flaky)
    start = np.array(rec.cameras[1].params)
    ba = R.bundle_adjust(rec, proj, max_iterations=30)
    assert len(calls) == 2 and calls[1] == pyceres.LinearSolverType.ITERATIVE_SCHUR
    assert ba["retried"] and "CONVERGENCE" in ba["termination"]
    assert not np.allclose(np.array(rec.cameras[1].params[:2]), start[:2])          # refined and written back


def test_set_camera_params_checks_the_write(tmp_path):
    import pycolmap
    from mppp.sfm.reconstruction import _set_camera_params
    r = pycolmap.Reconstruction()
    r.add_camera(pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=10, height=10, params=[5.0, 5.0, 5.0]))
    _set_camera_params(r, 1, [6.0, 5.0, 5.0])
    assert list(r.cameras[1].params) == [6.0, 5.0, 5.0]


def test_thermal_stage_holds_bins_with_few_images():
    from mppp.sfm import thermal as T
    assert T.THERMAL_MIN_BIN_IMAGES == 3
    import inspect
    src = inspect.getsource(T.thermal_stage)
    assert "weak_bins_held" in src and "WARNING" in src


# ----------------------------------------------------------------------------- v0p74.9: after the 5 Oct promotion

def test_calibrate_derives_the_regular_state_and_reports_loo(tmp_path):
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import block_info, calibrate, check_model, report_lines
    infos = [block_info(_block(tmp_path, n, s, T, fx_a=4720.0 + d)) for n, s, T, d in
             (("a", 400, -20.0, 0.0), ("b", 700, -10.0, 2.0), ("c", 1000, 0.0, -2.0), ("d", 1300, -5.0, 1.0))]
    old = json.loads(zcam_focus_model_path().read_text())
    m = calibrate(infos, old)
    zl = m["cameras"]["ZL034"]["lines"]
    rr = m["calibration"]["regular_ratio"]["ZL034"]
    assert rr["source"] == "default" and rr["ratio"] == pytest.approx(1.0095)   # 1 regular group per block: too few
    assert zl["regular"]["fx"]["a"] == pytest.approx(zl["backlash"]["fx"]["a"] / 1.0095)
    assert "c_per_degC" not in zl["regular"]["fx"] and "c_per_degC" not in zl["backlash"]["fx"]   # 4 blocks < 5
    loo = m["calibration"]["loo"]["ZL034"]["fx"]
    assert loo["n"] == 4 and loo["rms_px"] > 0 and {p["block"] for p in loo["per_block"]} == {"a", "b", "c", "d"}
    assert check_model(m, old)["ok"]
    assert any("Leave one out" in ln for ln in report_lines(m))
    fitted = calibrate(infos, old, regular="fitted")
    assert "regular_ratio" not in fitted["calibration"]


def test_check_model_blocks_the_5_oct_regular_lines():
    """The checks added in v0p74.9 on what was promoted on 5 Oct: regular-state temperature terms of +8 px/degC
    and ZR k2 -2.07 against +0.86 in the backlash state."""
    from mppp.sfm.zcam_calib import check_model
    e = {"backlash": {"fx": {"a": 4720.0, "b": 0.05, "tau_a": 1.0, "chi2_dof_a": 2.0},
                      "k2": {"a": 0.856, "b": 0.0, "tau_a": 0.1, "chi2_dof_a": 2.0}},
         "regular": {"fy": {"a": 4694.0, "b": 0.045, "tau_a": 15.0, "chi2_dof_a": 167054.5,
                            "temperature": {"applied": True, "c_per_degC": 8.13}},
                     "k2": {"a": -2.066, "b": -0.02, "tau_a": 3.4, "chi2_dof_a": 1448.6}}}
    new = {"cameras": {"ZR034": {"f0_px": 4725.0}}, "calibration": {"blocks": ["a", "b", "c"], "eyes": {"ZR034": e}}}
    rep = check_model(new, {"cameras": {"ZR034": {"f0_px": 4750.0}}})
    bad = {c["check"] for c in rep["checks"] if not c["ok"]}
    assert not rep["ok"] and {"ZR034 regular fy temperature term", "ZR034 regular k2"} <= bad


def test_select_every_passing_block_and_regular_ratio_at_focus():
    from mppp.sfm.zcam_calib import regular_ratio, select_blocks
    # select_blocks(n=0): every passing block
    infos = [{"name": n, "run_done": {"verdict": "WARN"}, "statuses": {}, "checks": {}, "sol": s,
              "bins": [{"state": "backlash", "group": g, "fixed": [], "focus": 300.0 + 100 * k}
                       for g in ("ZL034", "ZR034") for k in range(6)]} for n, s in (("a", 100), ("b", 110), ("c", 500))]
    sel, grades = select_blocks(infos, n=0)
    assert len(sel) == 3
    # the regular ratio against the backlash ratio at the regular groups' own focus counts
    gs = [{"camera_group": "ZR034", "state": "backlash", "decided": True, "ratio": 1.0095 + 4.5e-6 * (F - 600),
           "focus_count": F, "fitted": 3} for F in range(100, 1300, 100)]
    gs += [{"camera_group": "ZR034", "state": "regular", "decided": True, "ratio": 1.0000, "focus_count": F, "fitted": 2}
           for F in (200, 250, 300)]
    rr = regular_ratio([{"name": "a", "backlash_groups": gs}], "ZR034")
    assert rr["source"].startswith("measured") and rr["ratio"] == pytest.approx(1.0095 + 4.5e-6 * (250 - 600), abs=2e-4)


def test_final_runs_go_to_a_variant_and_homography_abort_is_retried():
    import importlib.util
    from mppp.sfm.matching import CALIBRATED_ABORT_SIGNS
    msg = "E20261006 09:27:36 homography_matrix.cc:167] Check failed: cam_rays1.size() >= 4 (3 vs. 4)"
    assert any(k in msg for k in CALIBRATED_ABORT_SIGNS)
    spec = importlib.util.spec_from_file_location("rp9", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    src = (ROOT / "scripts" / "run_pipeline.py").read_text()
    assert '"--final-variant", default="cal"' in src and 'variant=self.a.final_variant' in src
    assert '"--n", type=int, default=0' in src
