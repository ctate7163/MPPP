"""mppp.calib: the frozen forms, the promoted models and their constraint to an image."""
import copy
import json

import numpy as np
import pytest
from conftest import config_selection_off

from mppp import calib
from mppp.calib import forms, models as M


def _nl():
    return calib.load_camera("N", "L")


def test_camera_at_reference_temperature_is_the_file():
    m = _nl()
    t = calib.thermal_terms(m["thermal"])
    name, p, applied = calib.camera_at(m, t["T0"], sol=1000.0)
    assert name == "THIN_PRISM_FISHEYE" and len(p) == 12
    assert np.allclose(p, m["params"]) and "early_mission" not in applied
    assert applied["thermal"]["sx"] == 1.0 and applied["thermal"]["sy"] == 1.0
    name, p0, applied = calib.camera_at(m)                     # no temperature, no sol: the file as it is
    assert p0 == [float(v) for v in m["params"]] and applied == {}


def test_camera_at_scales_f_and_moves_pp_and_k1_with_temperature():
    m = _nl()
    t = calib.thermal_terms(m["thermal"])
    T = t["T0"] + 10.0
    _, p, applied = calib.camera_at(m, T, sol=1000.0)
    q = m["params"]
    assert p[0] == pytest.approx(q[0] * (1 + 1e-5 * t["fx_ppm"]))
    assert p[1] == pytest.approx(q[1] * (1 + 1e-5 * t["fy_ppm"]))
    assert p[2] == pytest.approx(q[2] + 10 * t["cx_px"]) and p[3] == pytest.approx(q[3] + 10 * t["cy_px"])
    assert p[4] == pytest.approx(q[4] + 10 * t["k1"]) and p[5:] == q[5:]
    assert applied["thermal"]["T_median_degC"] == T
    m2 = copy.deepcopy(m)
    m2["thermal"]["cx_px_per_degC"] = 0.05
    _, p2, _ = calib.camera_at(m2, T, sol=1000.0)
    assert p2[2] == pytest.approx(q[2] + 0.5)


def test_camera_at_adds_the_early_mission_offsets_before_the_sol():
    m = _nl()
    em = m["early_mission"]
    t = calib.thermal_terms(m["thermal"])
    _, late, a_late = calib.camera_at(m, t["T0"], sol=em["before_sol"])
    _, early, a_early = calib.camera_at(m, t["T0"], sol=em["before_sol"] - 1)
    assert "early_mission" not in a_late and a_early["early_mission"]["share"] == 1.0
    for i, k in enumerate(("fx", "fy", "cx", "cy")):
        d = em[f"d{k}_px"] if k in em["applied_terms"] else 0.0
        assert early[i] - late[i] == pytest.approx(d)
    _, half, a_half = calib.camera_at(m, t["T0"], early=0.5)
    assert half[2] - late[2] == pytest.approx(0.5 * em["dcx_px"]) and a_half["early_mission"]["share"] == 0.5
    assert calib.early_share(m, [100, 200, 900, 1000]) == 0.5 and calib.early_share({}, [1, 2]) == 0.0


def test_rig_at_applies_the_drift():
    from scipy.spatial.transform import Rotation
    rig = calib.load_rig("N")
    dr = rig["drift"]
    sol = float(dr["sol0"]) + 500.0
    R, t, applied = calib.rig_at(rig, sol)
    rv = np.degrees(Rotation.from_matrix(R @ np.asarray(rig["R_sensor_from_ref"]).T).as_rotvec()) * 1e3
    off = calib.drift_offset_mdeg(dr, sol)
    assert np.allclose(rv, [off["pitch_mdeg"], off["yaw_mdeg"], off["roll_mdeg"]], atol=1e-6)
    assert off["pitch_mdeg"] == pytest.approx(500 * dr["pitch_mdeg_per_sol"]
                                              + (dr["pitch_early_mdeg_per_sol"] - dr["pitch_mdeg_per_sol"])
                                              * (min(sol, dr["knot_sol"]) - dr["knot_ref"]))
    assert applied["drift"]["sol_median"] == sol
    assert t is not None and np.allclose(t, rig["t_sensor_from_ref_m"])     # the fitted translation
    R0, t0, a0 = calib.rig_at(rig, None)
    assert np.allclose(R0, rig["R_sensor_from_ref"]) and a0 == {}
    R1, t1, _ = calib.rig_at(dict(rig, translation="cahv"), sol)
    assert t1 is None
    # the hinge: a plain drift (rates only) still works
    off = calib.drift_offset_mdeg({"sol0": 1000.0, "pitch_mdeg_per_sol": 0.005}, 1200.0)
    assert off == {"pitch_mdeg": pytest.approx(1.0), "yaw_mdeg": 0.0, "roll_mdeg": 0.0}


def test_zcam_camera_at_both_states_and_the_regular_ratio():
    zm = calib.load_zcam(34)
    for g in ("ZL034", "ZR034"):
        c = zm["cameras"][g]
        ref = c["lines"]["reference_focus"]
        for focus in (ref, ref + 400.0):
            name, pb, ib = calib.zcam_camera_at(zm, g, focus, "backlash")
            name2, pr, ir = calib.zcam_camera_at(zm, g, focus, "regular")
            assert name == name2 == "FULL_OPENCV" and len(pb) == 12
            bl, rl = c["lines"]["backlash"], c["lines"]["regular"]
            assert pb[0] == pytest.approx(bl["fx"]["a"] + bl["fx"]["b"] * (focus - ref))
            assert pr[0] == pytest.approx(rl["fx"]["a"] + rl["fx"]["b"] * (focus - ref))
            if rl["fx"].get("derived"):                                      # regular derived from backlash
                assert pb[0] / pr[0] == pytest.approx(c["regular_ratio"], rel=1e-9)
            else:                                                              # regular measured on its own
                assert pb[0] / pr[0] == pytest.approx(c["regular_ratio"], rel=2e-3)
            assert pb[4] == pytest.approx(bl["k1"]["a"]) and pb[5] == pytest.approx(bl["k2"]["a"])
            assert pb[6:] == [0.0] * 6 and pr[6:] == [0.0] * 6              # p1 = p2 = k3 = 0 held
            if g == "ZL034":                                                   # pp0 held (the reference eye)
                assert pb[2:4] == pytest.approx(c["pp0_px"])
            else:
                assert pb[2] == pytest.approx(c["pp0_px"][0])
                assert pb[3] == pytest.approx(c["pp0_px"][1] + bl["cy"]["b"] * (focus - ref))
    # a state without lines follows the other state through regular_ratio
    z2 = copy.deepcopy(zm)
    z2["cameras"]["ZL034"]["lines"].pop("regular")
    _, p, info = calib.zcam_camera_at(z2, "ZL034", 600.0, "regular")
    assert p[0] == pytest.approx(calib.zcam_camera_at(zm, "ZL034", 600.0, "backlash")[1][0] / zm["cameras"]["ZL034"]["regular_ratio"])
    with pytest.raises(ValueError):
        calib.zcam_camera_at(zm, "ZL034", 600.0, "other")
    with pytest.raises(KeyError):
        calib.zcam_camera_at(zm, "ZL048", 600.0)
    assert calib.focus_state(False) == "backlash" == forms.ZCAM_DEFAULT_STATE and calib.focus_state(True) == "regular"


def test_hazcam_start_model_and_loader_warning(caplog):
    import logging
    for fam in ("F", "R"):
        for eye in ("L", "R"):
            h = calib.hazcam_start_model(fam, eye)
            assert h["model"] == "THIN_PRISM_FISHEYE" and (h["width"], h["height"]) == (5120, 3840)
            assert h["params"][:6] == [2130.0, 2130.0, 2560.0, 1920.0, -0.25, 0.08] and h["params"][6:] == [0.0] * 6
            assert h["source"] == "start model (not calibrated)" and forms.validate(h) == []
            assert calib.thermal_terms(h["thermal"]) == {"fx_ppm": 0.0, "fy_ppm": 0.0, "cx_px": 0.0, "cy_px": 0.0,
                                                         "k1": 0.0, "T0": -20.0}
    with caplog.at_level(logging.WARNING, logger="mppp.calib"):
        m = calib.load_camera("F", "L")
    assert m["file"] is None and "start model" in caplog.text
    name, p, applied = calib.camera_at(m, -30.0, sol=10)
    assert p == m["params"] and applied["thermal"]["sx"] == 1.0           # no thermal, no early term
    with pytest.raises(ValueError):
        calib.hazcam_start_model("N", "L")


def test_validate_rejects_a_held_term_that_is_not_zero():
    m = _nl()
    assert forms.validate(m) == [] and forms.validate_camera(m, forms.NAVCAM) == []
    bad = copy.deepcopy(m)
    bad["params"][10] = 1e-4
    assert any("sx1" in p for p in forms.validate(bad))
    bad = copy.deepcopy(m)
    bad.pop("thermal")
    assert any("thermal" in p for p in forms.validate(bad))
    rig = calib.load_rig("N")
    assert forms.validate(rig) == []
    assert any("temperature" in p for p in forms.validate(dict(rig, thermal={"yaw_mdeg_per_degC": -1.0})))
    z = calib.load_zcam(34)
    assert forms.validate(z) == []
    bad = copy.deepcopy(z)
    bad["cameras"]["ZL034"]["lines"]["backlash"]["p1"] = {"a": 0.001, "b": 0.0}
    assert any("p1" in p for p in forms.validate(bad))
    bad = copy.deepcopy(z)
    bad["cameras"]["ZR034"]["distortion"] = {"names": ["k1", "k2", "k3"], "params": [-0.4, 0.8, 0.01]}
    assert any("k3" in p for p in forms.validate(bad))
    bad = copy.deepcopy(z)
    bad["cameras"]["ZR034"]["thermal"]["f_px_per_degC"] = 0.5
    assert any("temperature" in p for p in forms.validate(bad))
    assert forms.form_for("N") is forms.NAVCAM and forms.form_for("R") is forms.HAZCAM and forms.form_for("Z") is forms.ZCAM
    with pytest.raises(ValueError):
        forms.form_for("S")


def test_check_form_describe_and_fingerprint(tmp_path, monkeypatch):
    rep = calib.check_form()
    assert rep["ok"] and not rep["errors"] and set(M.NAVCAM_FILES) <= set(rep["files"])
    assert any("start model" in w for w in rep["warnings"])                # no Hazcam files yet
    lines = calib.describe()
    assert len(lines) >= 4 and "M2020_N_rig.json" in "".join(lines)
    fp = calib.models_fingerprint()
    assert fp and len(fp) == 16
    # a folder with a wrong file: not in the form, another fingerprint
    for n in M.NAVCAM_FILES + ("M2020_ZCAM034_focus_model.json",):
        (tmp_path / n).write_bytes((M._folder(None) / n).read_bytes())
    d = json.loads((tmp_path / "M2020_NL_fisheye_tangential.json").read_text())
    d["params"][11] = 0.01
    (tmp_path / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(d))
    rep = calib.check_form(tmp_path)
    assert not rep["ok"] and any("sy1" in e for e in rep["errors"])
    assert calib.models_fingerprint(tmp_path) != fp
    monkeypatch.setenv("MPPP_CMODS", str(tmp_path))
    assert calib.models_fingerprint() == calib.models_fingerprint(tmp_path)
    assert calib.models_fingerprint(tmp_path / "empty") is None


def test_label_fallback_for_a_zoom_without_a_file():
    from conftest import ZL0
    if not ZL0.is_file():
        pytest.skip("example IMGs not present")
    import mppp
    from mppp.image import MPPPImage
    cfg = config_selection_off({"masking": {"infer_mask": False}})
    im = MPPPImage(ZL0, cfg, mppp.load_waypoints())
    meta = im.meta
    name, p, src = calib.label_camera(meta)
    K = np.asarray(meta["intrinsics_label"]["K"])
    assert name == "FULL_OPENCV" and "label" in src
    assert p[0] == pytest.approx(K[0, 0]) and p[2] == pytest.approx(K[0, 2] + 0.5)
    assert p[4] == pytest.approx(meta["intrinsics_label"]["dist_opencv"]["k1"]) and p[6:] == [0.0] * 6
    # the image's COLMAP prior camera: the 34 mm model (the test image is at 34 mm, on the low-backlash list)
    c = meta["colmap_camera"]
    assert c["model"] == "FULL_OPENCV" and "regular state" in c["source"] and c["width"] == meta["intrinsics"]["width"]
    zm = calib.load_zcam(34)
    focus = float(meta["focus_position_count"])
    want = calib.zcam_camera_at(zm, "ZL034", focus, "regular")[1]
    assert np.allclose(c["params"], want)                                    # full resolution, no extra padding
    # a zoom without a file (48 mm here): the label start
    meta2 = copy.deepcopy(meta)
    meta2["filename"]["zoom_mm"] = 48
    meta2["filename"]["camera_group"] = "ZL048"
    name2, p2, src2 = calib.label_camera(meta2)
    assert name2 == "FULL_OPENCV" and p2[:4] == pytest.approx(p[:4])
    assert calib.load_zcam(48) is None


def test_navcam_label_fallback_is_the_cahvore_fisheye():
    from conftest import NLF
    if not NLF.is_file():
        pytest.skip("example IMGs not present")
    import mppp
    from mppp.image import MPPPImage
    cfg = config_selection_off({"masking": {"infer_mask": False}})
    im = MPPPImage(NLF, cfg, mppp.load_waypoints())
    meta = im.meta
    name, p, src = calib.label_camera(meta)
    R = meta["camera_model_label"]["MODEL_COMPONENT_6"]
    K = np.asarray(meta["intrinsics_label"]["K"])
    assert name == "THIN_PRISM_FISHEYE" and "CAHVORE" in src
    assert p[0] == pytest.approx(K[0, 0] * (1 + R[0])) and p[4] == pytest.approx(R[1] / (1 + R[0]))
    assert p[10] == p[11] == 0.0
    # the prior camera of the processed image: the promoted NL model at the label temperature, scaled to the tile
    c = meta["colmap_camera"]
    nl = calib.load_camera("N", "L")
    want = calib.camera_at(nl, meta["camera_temperature_degC"], meta["sol"])[1]
    s = meta["filename"]["downsample_scale"]
    assert c["model"] == "THIN_PRISM_FISHEYE" and np.allclose(c["params"], calib.scaled("THIN_PRISM_FISHEYE", want, s))
    assert (c["width"], c["height"]) == (meta["intrinsics"]["width"], meta["intrinsics"]["height"])


def test_scaled_and_sol_epochs():
    p = calib.scaled("THIN_PRISM_FISHEYE", [2000.0, 2000.0, 100.0, 50.0, 0.1, 0, 0, 0, 0, 0, 0, 0], 0.5, 3.0, 4.0)
    assert p[:4] == [1000.0, 1000.0, 53.0, 29.0] and p[4] == 0.1
    assert calib.sol_epochs([60, 70, 360, 365], gap=50) == [(60, 70), (360, 365)]
    assert calib.sol_epochs([290, 310], gap=50, breaks=(300,)) == [(290, 290), (310, 310)]
    assert calib.sol_epochs([100, 110, 120, 130, 140], gap=50, max_span=25) == [(100, 120), (130, 140)]


def test_promote_replaces_the_file_and_logs_it(tmp_path, monkeypatch):
    src = M._folder(None)
    dst = tmp_path / "cmods"
    monkeypatch.setenv("MPPP_CMODS", str(dst))
    res = calib.promote([src], note="first")
    assert set(res["files"]) >= set(M.NAVCAM_FILES) and (dst / "CHANGES.md").read_text().count("first") == 1
    assert not (dst / "history").exists()
    new = tmp_path / "new"
    new.mkdir()
    d = json.loads((src / "M2020_NL_fisheye_tangential.json").read_text())
    d["params"][0] += 1.0
    (new / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(d))
    res = calib.promote([new], note="better")
    assert json.loads((dst / "M2020_NL_fisheye_tangential.json").read_text())["params"][0] == d["params"][0]
    assert list(res["files"]) == ["M2020_NL_fisheye_tangential.json"]
    d["params"][10] = 0.01                                                   # not in the form: refused, nothing copied
    (new / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(d))
    before = (dst / "M2020_NL_fisheye_tangential.json").read_bytes()
    with pytest.raises(ValueError):
        calib.promote([new], note="broken")
    assert (dst / "M2020_NL_fisheye_tangential.json").read_bytes() == before
    with pytest.raises(ValueError):
        M.collect([tmp_path / "x.json"])
    z = tmp_path / "M2020_ZCAM034_focus_model_candidate.json"
    z.write_bytes((src / "M2020_ZCAM034_focus_model.json").read_bytes())
    res = calib.promote([z])
    assert "M2020_ZCAM034_focus_model.json" in res["files"]


def test_shipped_navcam_models_carry_the_pass2_consensus():
    """In use is the 20 Oct pass-2 consensus of 16 blocks (NR cx(T) -0.0414, NL cx(T) +0.0134 px/degC), in the frozen
    form; it passes the form check."""
    nr = M._read(M._folder(None) / M.CAMERA_FILE[("N", "R")])
    nl = M._read(M._folder(None) / M.CAMERA_FILE[("N", "L")])
    assert nr["thermal"]["cx_px_per_degC"] == pytest.approx(-0.0414, abs=2e-4)
    assert nl["thermal"]["cx_px_per_degC"] == pytest.approx(0.0134, abs=2e-4)
    assert nl["form"] == "navcam-v1" and "ppm_per_degC" not in nl["thermal"]
    assert calib.check_form()["ok"]
