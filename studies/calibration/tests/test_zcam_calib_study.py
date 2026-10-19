"""Tests of the Mastcam-Z calibration study (studies/calibration/zcam_calib.py, zcam_calibration.py); moved from the
versioned tests of 0.74-0.75 when the module left mppp.sfm (the fit itself is mppp.calib.fit.fit_zcam)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]


def _block_lines(fx_b, fy_b, cx_a, cy_a, T, sol, R=None, t=None):
    from scipy.spatial.transform import Rotation
    R0 = Rotation.from_rotvec(np.radians([1.136, 3.923, -0.607])).as_matrix() if R is None else R
    t0 = [-0.2435, 0.0012, 0.0058] if t is None else t
    ln = {"fx": {"a": 4722.0, "b": fx_b, "sd_a": 0.5, "sd_b": 0.002, "n_bins": 8, "slope_source": "fitted"},
          "fy": {"a": 4734.0, "b": fy_b, "sd_a": 0.6, "sd_b": 0.003, "n_bins": 8, "slope_source": "fitted"},
          "cx": {"a": cx_a, "b": 0.0004, "sd_a": 0.1, "sd_b": 0.0002, "n_bins": 8, "slope_source": "fitted"},
          "cy": {"a": cy_a, "b": 0.0016, "sd_a": 0.1, "sd_b": 0.0002, "n_bins": 8, "slope_source": "fitted"},
          "k1": {"a": -0.468, "b": 2e-5, "sd_a": 0.005, "sd_b": 1e-5, "n_bins": 8, "slope_source": "fitted"},
          "k2": {"a": 0.87, "b": -3e-4, "sd_a": 0.05, "sd_b": 2e-4, "n_bins": 8, "slope_source": "fitted"}}
    zl = {k: v for k, v in ln.items() if k in ("fx", "fy", "k1", "k2")}
    return {"name": f"b{sol}", "navcam_scale": 1.0, "T_optics_degC": T, "sol": sol,
            "lines": {"ZR034": {"reference_focus": 600.0, "lines": {"backlash": ln}},
                      "ZL034": {"reference_focus": 600.0, "lines": {"backlash": zl}}},
            "rig": {"Z034": {"R_sensor_from_ref": R0.tolist(), "t_sensor_from_ref": t0, "t_sensor_from_ref_cahv": t0,
                             "R_refined": (Rotation.from_rotvec(np.radians([0.0, 0.010 * (T + 20), 0.0])).as_matrix() @ R0).tolist(),
                             "t_refined": (np.asarray(t0) + [0.002, 0, 0]).tolist(), "rotation_source": "model"}},
            "backlash_groups": [], "backlash_fits": None, "backlash_nav_scale": None, "backlash_ratios": None}


def _old_v0p74():
    """The shipped focus model as the v0p74 tests knew it: no rig, no pp0 (v0p75 ships both)."""
    from mppp.sfm.project import zcam_focus_model_path
    old = json.loads(zcam_focus_model_path().read_text())
    old.pop("rig", None)
    for g in old.get("cameras", {}).values():
        g.pop("pp0_px", None)
    return old


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
    from zcam_calib import block_info, grade_block, select_blocks
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
    from zcam_calib import block_info, check_model, low_backlash_groups, report_lines, start_model_from_blocks
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


def test_calibrate_lines_rig_and_no_temperature_term(tmp_path):
    """The frozen form: no temperature term (blocks at different temperatures average into one line), ZR cx
    pinned at pp0 (the bins' label value) with the blocks' offsets moved into the rig, the regular state derived."""
    from mppp.sfm.project import zcam_focus_model_path
    from zcam_calib import block_info, calibrate, check_model, report_lines
    infos = [block_info(_block(tmp_path, n, s, T, fx_a=4720.0 + 0.5 * T)) for n, s, T in
             (("a", 400, -20.0), ("b", 700, -10.0), ("c", 1000, 0.0))]
    old = _old_v0p74()
    m = calibrate(infos, old, rig_translation="refined")
    L = m["calibration"]["eyes"]["ZL034"]["backlash"]["fx"]
    assert L["n_blocks"] == 3 and "temperature" not in L
    assert m["cameras"]["ZL034"]["lines"]["backlash"]["fx"]["a"] == pytest.approx(4715.0, abs=2.5)   # the mean over T
    assert "c_per_degC" not in m["cameras"]["ZL034"]["lines"]["backlash"]["fx"]
    zr = m["cameras"]["ZR034"]
    assert zr["pp"]["cx_px_per_count"] == 0.0 and zr["lines"]["backlash"]["cx"] == {"a": 820.0, "b": 0.0, "form": "pinned"}
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
    from zcam_calib import check_model
    old = {"cameras": {"ZL034": {"f0_px": 4720.0}}}
    new = {"cameras": {"ZL034": {"f0_px": 4800.0, "aspect": 1.02}}, "rig": {"baseline_m": 0.5},
           "calibration": {"blocks": ["a"]}}
    rep = check_model(new, old)
    bad = {c["check"] for c in rep["checks"] if not c["ok"]}
    assert not rep["ok"] and {"ZL034 f0", "ZL034 aspect", "rig baseline", "blocks"} <= bad


def test_zcam_calibration_cli(tmp_path, monkeypatch):
    """The CLI end to end on a fake scapes root (a sites table with three N+Z sites)."""
    import importlib.util
    from mppp.waypoints import sites as SITES
    spec = importlib.util.spec_from_file_location("zc", HERE / "zcam_calibration.py")
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
    old = json.loads(zcam_focus_model_path().read_text())
    old.pop("rig", None)                                  # v0p75 ships a rig; the fake blocks' rigs are elsewhere
    (cmods / "M2020_ZCAM034_focus_model.json").write_text(json.dumps(old))
    monkeypatch.setenv("MPPP_CMODS", str(cmods))
    assert zc.main(["calibrate", "--root", str(root), "--n", "3"]) == 0
    out = sorted((root / "camera_analysis").glob("zcam34_calibration_*"))[-1]
    assert (out / "report.md").is_file() and (out / "M2020_ZCAM034_focus_model.json").is_file()
    assert zc.main(["start-update", "--root", str(root), "--dry-run"]) in (0, 1)
    assert (root / "zcam_low_backlash_found.csv").exists() is False                       # dry run adds nothing
    assert zc.main(["start-update", "--root", str(root)]) in (0, 1)
    assert "ZCAM09999" in (root / "zcam_low_backlash_found.csv").read_text()


def test_pp_shift_to_rig_projects_the_same_pixels():
    """A right eye with its principal point moved by (dcx, dcy) sees what the original eye sees after the rig
    rotation pp_shift_to_rig gives (distant points, sub-0.05 px)."""
    from scipy.spatial.transform import Rotation
    from zcam_calib import pp_shift_to_rig
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
    from zcam_calib import block_info, calibrate, check_model, pp_shift_to_rig, report_lines
    old = _old_v0p74()
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
    m = calibrate(infos, old, rig_translation="refined")
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
    from zcam_calib import block_info, calibrate, check_model
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
    from zcam_calib import block_info, check_model, start_model_from_blocks
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


def test_calibrate_derives_the_regular_state_and_reports_loo(tmp_path):
    from mppp.sfm.project import zcam_focus_model_path
    from zcam_calib import block_info, calibrate, check_model, report_lines
    infos = [block_info(_block(tmp_path, n, s, T, fx_a=4720.0 + d)) for n, s, T, d in
             (("a", 400, -20.0, 0.0), ("b", 700, -10.0, 2.0), ("c", 1000, 0.0, -2.0), ("d", 1300, -5.0, 1.0))]
    old = json.loads(zcam_focus_model_path().read_text())
    old.pop("rig", None)                                  # the synthetic blocks' rigs are nowhere near the shipped one
    m = calibrate(infos, old)
    zl = m["cameras"]["ZL034"]["lines"]
    rr = m["calibration"]["regular_ratio"]["ZL034"]
    assert rr["source"] == "default" and rr["ratio"] == pytest.approx(1.0095)   # 1 regular group per block: too few
    assert zl["regular"]["fx"]["a"] == pytest.approx(zl["backlash"]["fx"]["a"] / 1.0095)
    assert "c_per_degC" not in zl["regular"]["fx"] and "c_per_degC" not in zl["backlash"]["fx"]   # no temperature term
    loo = m["calibration"]["loo"]["ZL034"]["fx"]
    assert loo["n"] == 4 and loo["rms_px"] > 0 and {p["block"] for p in loo["per_block"]} == {"a", "b", "c", "d"}
    assert check_model(m, old)["ok"]
    assert any("Leave one out" in ln for ln in report_lines(m))


def test_check_model_blocks_the_5_oct_regular_lines():
    """What was promoted on 5 Oct (regular-state temperature terms of +8 px/degC, ZR k2 -2.07 against +0.86 in
    the backlash state) fails the form: a temperature term and a regular state that is not derived."""
    from mppp.calib import load_zcam
    from zcam_calib import check_model
    old = load_zcam(34)
    new = json.loads(json.dumps(old))
    zr = new["cameras"]["ZR034"]["lines"]["regular"]
    zr["fy"] = {"a": 4694.0, "b": 0.045, "form": "common", "c_per_degC": 8.13, "T0_degC": -15.0}
    zr["k2"] = {"a": -2.066, "b": -0.02, "form": "constant"}
    rep = check_model(new, old)
    bad = {c["check"]: c["note"] for c in rep["checks"] if not c["ok"]}
    assert not rep["ok"] and "form" in bad
    assert "temperature" in bad["form"] or "c_per_degC" in bad["form"]


def test_select_every_passing_block_and_regular_ratio_at_focus():
    from zcam_calib import regular_ratio, select_blocks
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


def test_calibrate_on_the_form_with_rig_residuals():
    from zcam_calib import calibrate
    infos = [_block_lines(0.062, 0.066, 743.15, 663.96, T, s)          # cx, cy at pp0: the v0p75 runs pin them
             for T, s in ((-20.0, 400), (-10.0, 700), (0.0, 1000), (-15.0, 1300))]
    model = {"cameras": {"ZR034": {"reference_focus": 600.0, "f0_px": 4728.0, "aspect": 1.0025, "slope_px_per_count": 0.064,
                                   "pp0_px": [743.15, 663.96], "distortion": {"names": ["k1", "k2"], "params": [-0.468, 0.875]}},
                         "ZL034": {"reference_focus": 600.0, "f0_px": 4721.0, "aspect": 1.0022, "slope_px_per_count": 0.05,
                                   "pp0_px": [876.7, 559.1], "distortion": {"names": ["k1", "k2"], "params": [-0.46, 0.635]}}}}
    m = calibrate(infos, model)
    zr = m["cameras"]["ZR034"]["lines"]["backlash"]
    assert zr["fx"]["b"] == zr["fy"]["b"] and 0.062 < zr["fx"]["b"] < 0.066            # one slope per eye
    assert zr["cx"] == {"a": 743.15, "b": 0.0, "form": "pinned"} and zr["cy"]["a"] == 663.96
    assert zr["cy"]["b"] == pytest.approx(0.0016)
    assert zr["k1"]["b"] == 0.0 and zr["k2"]["b"] == 0.0 and zr["k1"]["a"] == pytest.approx(-0.468)
    assert m["cameras"]["ZR034"]["pp"]["cx_px_per_count"] == 0.0 and m["cameras"]["ZL034"]["pp0_px"] == [876.7, 559.1]
    assert m["cameras"]["ZR034"]["lines"]["regular"]["fx"]["b"] == pytest.approx(zr["fx"]["b"] / m["calibration"]["regular_ratio"]["ZR034"]["ratio"])
    rig = m["rig"]
    assert rig["translation_source"].startswith("CAHV") and rig["baseline_m"] == pytest.approx(np.linalg.norm([-0.2435, 0.0012, 0.0058]))
    res = rig["residuals"]
    assert len(res["per_block"]) == 4 and all(abs(r["pitch_mdeg"]) < 1e-6 for r in res["per_block"])
    assert res["fits"]["yaw"]["vs_T"]["slope"] == pytest.approx(10.0, rel=0.05)         # 0.010 deg/degC put in
    assert abs(res["fits"]["yaw"]["vs_T"]["z"]) > 3 and res["fits"]["roll"]["vs_sol"]["slope"] == pytest.approx(0.0, abs=1e-6)
    # the free v0p74 form keeps separate slopes and the refined translation
    m2 = calibrate(infos, model, min_blocks_for_T=99, form="v0p74", rig_translation="refined")
    zr2 = m2["cameras"]["ZR034"]["lines"]["backlash"]
    assert zr2["fx"]["b"] != zr2["fy"]["b"] and zr2["cx"]["b"] == pytest.approx(0.0004)
    assert m2["rig"]["baseline_m"] == pytest.approx(np.linalg.norm([-0.2415, 0.0012, 0.0058]))
