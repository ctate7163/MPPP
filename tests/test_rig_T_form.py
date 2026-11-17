"""0.99.11 (Christian, 17 Nov 2026): the rig_T Navcam form - the stereo temperature change on the rig (pitch, yaw, roll
mdeg/degC), the principal points constant in temperature - and the Mastcam-Z bar test.  The rig at a sol and
temperature, the form checks, the temperature bins' rigs, the tied joint's temperature rates, the deck joint's
candidate, the focus bins of a zoom without a model (Seitah Overlook sol 121) and the bar test's fits."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

MD = 180e3 / np.pi


def _rig(thermal=None):
    R = Rotation.from_rotvec(np.radians(1e-3 * np.array([-100.0, 44.0, -46.0]))).as_matrix()
    r = {"R_sensor_from_ref": R.tolist(), "t_sensor_from_ref_m": [-0.4235, 0.0, 0.0], "translation": "deck",
         "drift": {"sol0": 0.0, "pitch_mdeg_per_sol": 0.006, "yaw_mdeg_per_sol": 0.004, "roll_mdeg_per_sol": -0.006}}
    if thermal:
        r["thermal"] = dict(thermal)
    return r


def test_rig_at_turns_the_rotation_by_the_temperature_rates():
    from rpp.calib.constrain import rig_at, rig_thermal_terms, thermal_offset_mdeg
    th = {"T0_degC": -20.0, "pitch_mdeg_per_degC": 0.04, "yaw_mdeg_per_degC": -1.2, "roll_mdeg_per_degC": 0.25}
    r0, rT = _rig(), _rig(th)
    assert rig_thermal_terms(r0) is None and rig_thermal_terms(rT)["yaw"] == -1.2
    for sol in (0.0, 1000.0):
        R0, t0, a0 = rig_at(r0, sol)
        Ra, _, aa = rig_at(rT, sol, -20.0)                   # at T0: the file's rotation (with the drift)
        assert np.allclose(R0, Ra, atol=1e-12) and "thermal" in aa and "thermal" not in a0
        Rb, tb, ab = rig_at(rT, sol, -10.0)                  # 10 degC warmer: turned by the rates x 10
        d = np.degrees(Rotation.from_matrix(Rb @ R0.T).as_rotvec()) * 1e3
        assert np.allclose(d, [0.4, -12.0, 2.5], atol=0.01)
        assert ab["thermal"]["offset_mdeg"]["yaw"] == pytest.approx(-12.0)
        assert np.allclose(tb, t0)                           # the translation stays (one baseline)
    Rn, _, an = rig_at(rT, 500.0)                            # no temperature: the rotation at T0
    assert np.allclose(Rn, rig_at(r0, 500.0)[0]) and "thermal" not in an
    off = thermal_offset_mdeg(rT, -15.0, T_ref_degC=-25.0)   # a bin against its block's rig
    assert off["roll_mdeg"] == pytest.approx(2.5) and off["yaw_mdeg"] == pytest.approx(-12.0)


def _cmods(tmp_path):
    from rpp.paths import data_dir
    d = tmp_path / "cmods"
    shutil.copytree(data_dir() / "cmods", d)
    return d


def test_form_checks_refuse_both_terms_and_accept_either(tmp_path):
    from rpp.calib import forms, models
    d = _cmods(tmp_path)
    cams = {e: json.loads((d / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    rig = json.loads((d / "M2020_N_rig.json").read_text())
    base_form = forms.navcam_form(cams, rig)
    assert base_form in ("pp_T", "none")
    assert not forms.validate_rig(rig)
    rig["thermal"] = {"T0_degC": -20.0, "pitch_mdeg_per_degC": 0.0, "yaw_mdeg_per_degC": -1.2, "roll_mdeg_per_degC": 0.25}
    assert not forms.validate_rig(rig)                       # 0.99.11: the rig_T form's rates are valid
    (d / "M2020_N_rig.json").write_text(json.dumps(rig))
    chk = models.check_form(d)
    if base_form == "pp_T":                                  # slopes and rates: the change counted twice
        assert not chk["ok"] and any("counted twice" in e for e in chk["errors"]) and chk["navcam_form"] == "both"
    for e in ("NL", "NR"):
        cams[e]["thermal"]["cx_px_per_degC"] = cams[e]["thermal"]["cy_px_per_degC"] = 0.0
        (d / f"M2020_{e}_fisheye_tangential.json").write_text(json.dumps(cams[e]))
    chk = models.check_form(d)
    assert chk["ok"] and chk["navcam_form"] == "rig_T", chk["errors"]
    bad = dict(rig, thermal={"yaw_mdeg_per_degC": -1.2})
    assert any("T0_degC" in p for p in forms.validate_rig(bad))
    assert any("thermal" in line for line in models.describe(d))


def test_temperature_bins_turn_the_navcam_rig(tmp_path):
    pytest.importorskip("pyceres")
    from rpp.sfm.reconstruction import bundle_adjust
    from rpp.sfm.thermal import image_temperatures, split_by_temperature
    from helpers import _build_rec, _synthetic
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    for r in proj.images:
        r["camera_temperature_degC"] = -32.0 if r["station"].endswith("0000") else -14.0
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=5)
    rates = {"pitch_mdeg_per_degC": 0.1, "yaw_mdeg_per_degC": -1.0, "roll_mdeg_per_degC": 0.3}
    model = {"rig": {"T0_degC": -20.0, **rates}}
    rec2, p2, rows = split_by_temperature(rec, proj, image_temperatures(proj), bin_deg=10, min_images=4,
                                          thermal_model=model)
    turned = p2.settings["navcam_bin_rigs_turned"]["rigs"]
    assert turned and {round(t["T_bin_degC"]) for t in turned} == {-32, -14}
    old = next(r for r in rec.rigs.values() if list(r.non_ref_sensors))
    R_old = np.asarray(old.sensor_from_rig(next(iter(old.non_ref_sensors))).rotation.matrix())
    seen = set()
    for rg in rec2.rigs.values():
        sids = list(rg.non_ref_sensors)
        if not sids or rg.rig_id in rec.rigs:
            continue
        R = np.asarray(rg.sensor_from_rig(sids[0]).rotation.matrix())
        d = np.degrees(Rotation.from_matrix(R @ R_old.T).as_rotvec()) * 1e3
        T_ = -32.0 if d[1] > 0 else -14.0                    # yaw -1 mdeg/degC: the cold bin turned by +12, the warm by -6
        want = np.array([0.1, -1.0, 0.3]) * (T_ + 20.0)
        assert np.allclose(d, want, atol=1e-6)
        seen.add(T_)
    assert seen == {-32.0, -14.0}
    # without a rig term the bins copy the rig as before
    rec3, p3, _ = split_by_temperature(rec, proj, image_temperatures(proj), bin_deg=10, min_images=4)
    assert "navcam_bin_rigs_turned" not in p3.settings


def test_thermal_model_for_project_carries_the_rig_rates():
    from types import SimpleNamespace
    from rpp.sfm.thermal import thermal_model_for_project
    th = {"T_degC": -17.5, "T0_degC": -20.0, "rates_mdeg_per_degC": {"pitch": 0.0, "yaw": -1.2, "roll": 0.25}}
    p = SimpleNamespace(settings={"navcam_cameras": {"NL": {}, "rig": {"thermal": th}}, "navcam_rig_thermal": th})
    m = thermal_model_for_project(p)
    assert m["rig"]["T0_degC"] == -17.5 and m["rig"]["yaw_mdeg_per_degC"] == -1.2 and m["rig"]["roll_mdeg_per_degC"] == 0.25
    assert thermal_model_for_project(SimpleNamespace(settings={"navcam_rig_thermal": False})) is None


def _tied_setup(tmp_path, K_true=None):
    from helpers import _build_rec, _synthetic
    from rpp_studies.calibration import tied as T
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    temps = {}
    for r in proj.images:
        r["sol"] = 120 if r["station"] == "S001D0000" else 900
        temps[r["name"]] = -31.0 if int(r["sclk_key"][:10]) % 2 else -9.0
    if K_true is not None:                                   # the truth: NR seen through a rig turned by K (T - T0)
        name_T = temps
        for im in rec.images.values():
            if int(im.camera_id) != 2:
                continue
            d = np.radians(1e-3 * np.asarray(K_true) * (name_T[im.name] + 20.0))
            Rd = Rotation.from_rotvec(d).as_matrix()
            cw = im.cam_from_world()
            R, t = np.asarray(cw.rotation.matrix()), np.asarray(cw.translation)
            cam = rec.cameras[im.camera_id]
            for p2 in im.points2D:
                if not p2.has_point3D():
                    continue
                Xc = R @ np.asarray(rec.points3D[p2.point3D_id].xyz) + t
                a, b = cam.img_from_cam(np.array([Xc]))[0], cam.img_from_cam(np.array([Rd @ Xc]))[0]
                p2.xy = np.asarray(p2.xy) + (b - a)
    idx = {"scape_of_image": {int(r["image_id"]): "blk" for r in proj.images},
           "images": {"blk": [int(r["image_id"]) for r in proj.images]}, "temps": temps}
    st = T.start_values()
    for eye, cid in (("NL", 1), ("NR", 2)):
        st["G"][eye] = np.asarray(rec.cameras[cid].params, float)[:10].copy()
        st["S"][eye] = np.array([1e-6 * st["G"][eye][0] * 40.0, 1e-6 * st["G"][eye][1] * 45.0, 0.01, 0.0, 0.0])
        st["E"][eye] = np.zeros(4)
    st["Q0"] = T.q_xyz(rigT[0])
    st["D"] = np.zeros(3)
    st["R"], st["t"] = rigT
    return proj, rec, idx, st, T


def test_apply_form_holds_the_slopes_or_the_rates(tmp_path):
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    sr, hr = T.apply_form(st, "rig_T")
    assert set(hr) == set(T.PP_SLOPE_TERMS) and sr["form"] == "rig_T"
    for e in ("NL", "NR"):
        assert sr["S"][e][2] == 0.0 and sr["S"][e][3] == 0.0 and st["S"][e][2] == 0.01     # a copy
    want = T.pp_stereo_rates(st)                            # from a pp_T start: the slopes' stereo equivalent
    assert np.allclose(sr["K"], want) and "principal-point slopes" in sr["K_start_source"]
    sp, hp = T.apply_form(st, "pp_T")
    assert hp == ("K",) and not np.any(sp["K"]) and sp["S"]["NR"][2] == 0.01
    with pytest.raises(ValueError):
        T.apply_form(st, "pp")
    q = T.rig_q_at(st["Q0"], np.zeros(3), np.zeros(3), 500.0, 250.0, K=np.array([0.0, -1.0, 0.0]), T=-10.0, T0=-20.0)
    d = np.degrees(Rotation.from_matrix(T.R_of_q(q) @ T.R_of_q(st["Q0"]).T).as_rotvec()) * 1e3
    assert np.allclose(d, [0.0, -10.0, 0.0], atol=0.02)          # the law is linear in q_xyz (as the drift)


def test_tied_joint_recovers_the_rig_temperature_rates(tmp_path):
    pytest.importorskip("pyceres")
    K_true = np.array([0.3, -1.2, 0.25])
    proj, rec, idx, st, T = _tied_setup(tmp_path, K_true=K_true)
    st0 = copy.deepcopy(st)
    st, hold = T.apply_form(st, "rig_T")
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    for rid, u in units["rigs"].items():                     # every rig unit starts at the law at its own temperature
        sid = next(iter(rec2.rigs[rid].non_ref_sensors))
        R = np.asarray(rec2.rigs[rid].sensor_from_rig(sid).rotation.matrix())
        want = T.R_of_q(T.rig_q_at(st["Q0"], st["D"], st["H"], u["sol"], st["knot"], K=st["K"], T=u["T"], T0=st["T0_rig"]))
        assert np.allclose(R, want, atol=1e-9)
    ba, g, stats = T.adjust_tied(rec2, proj2, idx, st, units, hold=tuple(T.HOLD_DEFAULT) + tuple(hold), iterations=40,
                                 covariance=True, passes=1, verbose=False)
    held = (ba.get("extra") or {}).get("held") or {}
    assert "K" not in held and held.get("S_NL") and 2 in held["S_NL"] and 3 in held["S_NR"]
    assert np.allclose(g["K"], K_true, atol=0.05), g["K"]
    assert g["S_NL"][2] == 0.0 and g["S_NR"][3] == 0.0
    res = T.globals_result(st, g, ba, units)
    assert res["navcam_form"] == "rig_T"
    th = res["rig"]["thermal_mdeg_per_degC"]
    assert th["yaw"] == pytest.approx(-1.2, abs=0.05) and th["roll"] == pytest.approx(0.25, abs=0.05)
    sd = res["rig"]["sd_thermal"]
    assert sd and all(np.isfinite(sd[a]) and sd[a] > 0 for a in ("pitch", "yaw", "roll"))
    # the law files: the rig's thermal rates, the cameras' slopes 0 - a valid rig_T model set
    from rpp.calib import forms, models
    res.update(blocks=["blk"], early_test={"held": list(T.EARLY_ALL)})
    out = T.write_law_files(res, tmp_path / "law")
    rj = json.loads((out / "M2020_N_rig.json").read_text())
    assert rj["thermal"]["yaw_mdeg_per_degC"] == pytest.approx(th["yaw"]) and rj["thermal"]["T0_degC"] == -20.0
    cams = {e: json.loads((out / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    assert forms.navcam_form(cams, rj) == "rig_T"
    assert not forms.validate_rig(rj)
    # the pp_T form on the same data cannot carry the roll: a higher cost
    st_p, hold_p = T.apply_form(st0, "pp_T")
    rec3, proj3, units3 = T.tied_reconstruction(rec, proj, idx, st_p, bin_deg=10.0)
    ba3, g3, _ = T.adjust_tied(rec3, proj3, idx, st_p, units3, hold=tuple(T.HOLD_DEFAULT) + tuple(hold_p), iterations=40,
                               covariance=False, passes=1, verbose=False)
    assert not np.any(g3["K"]) and float(ba3["final_cost"]) > 1.2 * float(ba["final_cost"])


def test_epoch_rows_take_the_laws_temperature_part_out():
    from rpp_studies.calibration import tied as T
    res = {"rig": {"thermal_mdeg_per_degC": {"pitch": 0.0, "yaw": -1.0, "roll": 0.5}, "T0_degC": -20.0},
           "diagnostics": {"rigs": [{"scape": "a", "epoch": 0, "sol": 100, "T": -10.0, "frames": 20, "pitch_mdeg": 1.0,
                                     "yaw_mdeg": 2.0, "roll_mdeg": 3.0}]}}
    r = T._epoch_rows(res)[0]
    assert r["yaw_mdeg"] == pytest.approx(12.0) and r["roll_mdeg"] == pytest.approx(-2.0) and r["pitch_mdeg"] == 1.0


def _all_rig_T(a=(0.36, 7.8, 2.4), K=(0.03, -1.15, 0.24)):
    ang = {}
    for i, k in enumerate(("pitch", "yaw")):
        ang[k] = {"knee250": {"knee_sol": 250.0, "a_mdeg": a[i], "rate_before": 20.0 if k == "pitch" else -38.0, "se_before": 2.0,
                              "rate_after": 6.0 if k == "pitch" else 4.0, "se_after": 0.2, "sd_mdeg": 1.0, "n": 160,
                              "per_degC": K[i], "se_per_degC": 0.05}, "n": 160}
    ang["roll"] = {"linear": {"a_mdeg": a[2], "rate": -6.1, "se": 0.7, "sd_mdeg": 5.0, "per_degC": K[2], "se_per_degC": 0.1}, "n": 160}
    return {"stations": 160, "sol_range": [15, 2006], "source": "jr_ALL.json", "form": "rig_T", "T0_degC": -20.0,
            "angles": ang, "baseline": {"mean_mm": 423.5, "se_mean_um": 13.0, "b0_mm": 423.45, "um_per_100sol": 8.0,
                                        "se_um_per_100sol": 2.0, "n": 158}}


def test_deck_rig_candidate_in_the_rig_T_form(tmp_path):
    from rpp_studies.calibration import deck_rig as DR
    from rpp.calib import forms
    d = _cmods(tmp_path)
    base = json.loads((d / "M2020_N_rig.json").read_text())
    nav = copy.deepcopy(base)
    L = DR.law(_all_rig_T())
    assert L["form"] == "rig_T" and L["yaw"]["per_degC"] == -1.15 and L["roll"]["per_degC"] == 0.24
    new = DR.build(_all_rig_T(), nav, base, base, label="ALL")
    th = new["thermal"]
    assert th["yaw_mdeg_per_degC"] == -1.15 and th["roll_mdeg_per_degC"] == 0.24 and th["T0_degC"] == -20.0
    assert "rig_T" in new["note"] and not forms.validate_rig(new)
    cams = DR.rig_T_cameras(d)
    assert all(c["thermal"]["cx_px_per_degC"] == 0.0 and c["thermal"]["cy_px_per_degC"] == 0.0 for c in cams.values())
    assert forms.navcam_form(cams, new) == "rig_T"
    old_cams = {e: json.loads((d / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    chk = DR.check(new, base, cams_new=cams, cams_current=old_cams)
    assert any("stereo temperature change at -30 degC" in line for line in chk["lines"])
    assert "thermal_change_mdeg" in chk
    # the command: the rig and the two cameras into --out
    for name, obj in (("all_rig.json", _all_rig_T()), ("nav.json", nav), ("base.json", base), ("cur.json", base)):
        (tmp_path / name).write_text(json.dumps(obj))
    rc = DR.main(["--all-rig", str(tmp_path / "all_rig.json"), "--navjoint", str(tmp_path / "nav.json"),
                  "--base", str(tmp_path / "base.json"), "--current", str(tmp_path / "cur.json"),
                  "--cameras-from", str(d), "--out", str(tmp_path / "cand"), "--promote", "--force"])
    assert rc == 0
    for f in ("M2020_N_rig.json", "M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json"):
        assert (tmp_path / "cand" / f).is_file()
    nc = {e: json.loads((tmp_path / "cand" / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    assert forms.navcam_form(nc, json.loads((tmp_path / "cand" / "M2020_N_rig.json").read_text())) == "rig_T"


def test_zoom_without_a_model_starts_high_above_the_label_and_refines_f():
    """Seitah Overlook sol 121 (16 Nov): the 26 and 100 mm frames held at the label f slid along the boresight."""
    from rpp.sfm.project import ZCAM_NO_MODEL_HIGH_RATIO, _split_by_focus
    base = {"model": "FULL_OPENCV", "params": [3550.0, 3550.0, 824, 600, -0.02, 0.0, 0, 0, 0, 0, 0, 0],
            "source": "median label CAHVOR", "fixed_params": []}
    rows = [{"camera_group": "ZL026", "focus_count": 1000 + 5 * i, "label_f_px": 3550.0} for i in range(5)]
    out = _split_by_focus(rows, {"ZL026": base}, {"ZL026": "Z"}, 30.0, "focal_pp", min_zoom_images=6)
    for c in out.values():
        assert c["params"][0] == pytest.approx(3550.0 * ZCAM_NO_MODEL_HIGH_RATIO)
        assert "fx" not in c["fixed_params"] and "fy" not in c["fixed_params"]
        assert {"cx", "cy", "k1", "k2"} <= set(c["fixed_params"]) and c.get("held_except_f")
    rows_r = [dict(r, camera_group="ZL100", focus_state="regular") for r in rows]
    out_r = _split_by_focus(rows_r, {"ZL100": base}, {"ZL100": "Z"}, 30.0, "focal_pp", min_zoom_images=6)
    assert all(c["params"][0] == pytest.approx(3550.0) for c in out_r.values())      # the low state at the label
    rows_s = [dict(r, camera_group="ZL110") for r in rows]                              # 110 mm: one state
    out_s = _split_by_focus(rows_s, {"ZL110": base}, {"ZL110": "Z"}, 30.0, "focal_pp")
    assert all(c["params"][0] == pytest.approx(3550.0) for c in out_s.values())


def test_bar_test_units_and_fits():
    from rpp_studies.calibration.ztied import _rig_T_units, rig_T_test
    rows = []
    for k in range(24):                                     # one block, 24 pairs over 12 degC of lens temperature
        T_ = -28.0 + 0.5 * k
        for e in ("ZL", "ZR"):
            rows.append({"camera_group": f"{e}034@blk", "sclk_key": f"s{k}", "sol": 400, "zcam_optics_temperature_degC": T_})
    units = _rig_T_units(rows, [(400, 400)], 4.0, 6, "Z034", "blk")
    assert len(units) >= 2 and all(u["pairs"] >= 6 for u in units) and units[0]["rig_key"] == "Z034@blk"
    assert {r["zcam_rig_epoch"] for r in rows} == {u["index"] for u in units}
    # synthetic rigs: yaw -0.7 mdeg/degC within each block, roll +0.15, a block offset each, pitch 0
    rng = np.random.default_rng(1)
    us = []
    for b, off in (("a", 3.0), ("b", -2.0), ("c", 5.0), ("d", 0.0)):
        for T_ in (-27.0, -22.0, -17.0, -12.0):
            us.append({"block": b, "sol_median": 100.0 + 400 * len(us) / 4, "T_median_degC": T_, "pairs": 10,
                       "yaw_mdeg": off - 0.7 * T_ + rng.normal(0, 0.05), "roll_mdeg": 0.15 * T_ + rng.normal(0, 0.05),
                       "pitch_mdeg": rng.normal(0, 0.05)})
    t = rig_T_test(us, navcam_rates={"pitch": 0.0, "yaw": -1.2, "roll": 0.25})
    w = t["angles"]["yaw"]["within_mdeg_per_degC"]
    assert w["slope"] == pytest.approx(-0.7, abs=0.02) and w["blocks"] == 4
    assert t["angles"]["roll"]["within_mdeg_per_degC"]["slope"] == pytest.approx(0.15, abs=0.02)
    assert t["angles"]["yaw"]["ratio_to_navcam_within"] == pytest.approx(0.7 / 1.2, abs=0.03)
    from rpp_studies.calibration.ztied import rig_T_lines
    lines = rig_T_lines(dict(t, bin_deg=4.0))
    assert "within the blocks" in lines[1] and "Navcam rig" in lines[2]


def test_promotion_refuses_the_two_forms_at_once(tmp_path):
    from rpp.calib import forms, models
    from rpp_studies.calibration import deck_rig as DR
    d = _cmods(tmp_path)
    cams = {e: json.loads((d / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    if forms.navcam_form(cams, json.loads((d / "M2020_N_rig.json").read_text())) != "pp_T":
        pytest.skip("the package cameras carry no principal-point slopes")
    cand = tmp_path / "cand"
    cand.mkdir()
    rig = json.loads((d / "M2020_N_rig.json").read_text())
    rig["thermal"] = {"T0_degC": -20.0, "pitch_mdeg_per_degC": 0.0, "yaw_mdeg_per_degC": -1.2, "roll_mdeg_per_degC": 0.25}
    (cand / "M2020_N_rig.json").write_text(json.dumps(rig))
    with pytest.raises(ValueError, match="counted twice"):
        models.promote([cand], note="rig alone", folder=d)
    for e, c in DR.rig_T_cameras(d).items():
        (cand / f"M2020_{e}_fisheye_tangential.json").write_text(json.dumps(c))
    models.promote([cand], note="the rig_T set", folder=d)
    chk = models.check_form(d)
    assert chk["ok"] and chk["navcam_form"] == "rig_T"


def test_deck_rig_carried_over_to_other_cameras():
    from rpp_studies.calibration import deck_rig as DR
    d = _cmods(tmp_path_factory_dir())
    tgt = {e: json.loads((d / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    sta = copy.deepcopy(tgt)
    sta["NR"]["params"][2] = float(sta["NR"]["params"][2]) - 0.09         # the stations' NR cx 0.09 px left
    c = DR.camera_rig_correction(sta, tgt)
    assert c[1] == pytest.approx(-0.09 / float(tgt["NR"]["params"][0]) * MD, rel=1e-9) and c[0] == 0.0 and c[2] == 0.0
    base = json.loads((d / "M2020_N_rig.json").read_text())
    r0 = DR.build(_all_rig_T(), base, base, base)
    r1 = DR.build(_all_rig_T(), base, base, base, camera_correction_mdeg=c)
    dd = np.degrees(Rotation.from_matrix(np.asarray(r1["R_sensor_from_ref"]) @ np.asarray(r0["R_sensor_from_ref"]).T).as_rotvec()) * 1e3
    assert np.allclose(dd, c, atol=1e-3) and r1["camera_correction_mdeg"]["yaw"] == pytest.approx(c[1])


def tmp_path_factory_dir():
    import tempfile
    return Path(tempfile.mkdtemp(prefix="rigT_"))


def test_station_cameras_from_a_block_project(tmp_path):
    from rpp_studies.calibration import deck_rig as DR
    p = {"cameras": {"NL": {"params": [2950.0 * (1 + 35e-6 * 10), 2950.0, 2591.8, 1944.5]},
                     "NR": {"params": [2943.0, 2943.0, 2573.0, 1950.6]}},
         "settings": {"navcam_cameras": {
             "NL": {"thermal": {"T_median_degC": -10.0, "T0_degC": -20.0, "sx": 1 + 35e-6 * 10, "sy": 1.0, "cx_px_per_degC": 0.035,
                                "cy_px_per_degC": 0.015}},
             "NR": {"thermal": {"T_median_degC": -10.0, "T0_degC": -20.0, "cx_px_per_degC": -0.02, "cy_px_per_degC": 0.014},
                    "early_mission": {"applied_px": {"cy": 0.1}}}}}}
    f = tmp_path / "project.json"
    f.write_text(json.dumps(p))
    c = DR.station_cameras_from_project(f)
    assert c["NL"]["params"][0] == pytest.approx(2950.0) and c["NL"]["params"][2] == pytest.approx(2591.45)
    assert c["NR"]["params"][2] == pytest.approx(2573.2) and c["NR"]["params"][3] == pytest.approx(1950.36)
