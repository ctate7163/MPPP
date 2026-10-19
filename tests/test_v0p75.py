"""v0p75: sites with several visits (sol ranges), Navcam rig epochs, the Mastcam-Z line form (one f slope per eye,
pinned principal points, constant distortion), the rig consensus with residual tests, the bin-camera fallback of the
focus-state classification, the pipeline's consensus group."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ------------------------------------------------------------------------------------------------- sites, selection
def test_sol_ranges_span_and_membership():
    from mppp.waypoints import sites as S
    assert S.sol_ranges([1, 2]) == [(1, 2)]
    assert S.sol_ranges([[361, 378], [91, 101]]) == [(361, 378), (91, 101)]
    assert S.sol_span([[361, 378], [91, 101]]) == (361, 378)          # the main visit names the folder; last of all
    assert S.sol_span([[91, 101], [361, 378]]) == (91, 378)
    assert S.in_sol_ranges(95, [[361, 378], [91, 101]]) and not S.in_sol_ranges(200, [[361, 378], [91, 101]])
    for bad in ([1], [1, 2, 3], [[1, 2], 3], "x", [[1, "a"]]):
        with pytest.raises(ValueError):
            S.sol_ranges(bad)
    assert S.check_sites({"a": [[5, 9], [20, 10]]}) and not S.check_sites({"a": [[5, 9], [10, 20]]})


def test_site_table_with_visits(tmp_path):
    from mppp.waypoints import sites as S
    f = tmp_path / "sites.json"
    f.write_text(json.dumps({"sites": {
        "sid": {"sols": [[361, 378], [91, 101]], "zcam": [34], "settings": {}},
        "one": {"sols": [700, 712], "zcam": [], "settings": {}}},
        "groups": {"navcam_consensus": ["sid", "one"]}}))
    assert S.load_sites(f) == {"sid": (361, 378), "one": (700, 712)}
    assert S.load_site_ranges(f) == {"sid": [(361, 378), (91, 101)], "one": [(700, 712)]}
    assert S.work_folder(tmp_path, "sid", True, [[361, 378], [91, 101]]).name == "mars2020_sol_0361_sid_zcam"
    err, warn = S.validate_site_table(f)
    assert not err and not warn
    f.write_text(json.dumps({"sites": {"sid": {"sols": [[361, 378], [370, 400]], "zcam": [34]}}, "groups": {}}))
    err, _ = S.validate_site_table(f)
    assert err and "overlap" in err[0]
    f.write_text(json.dumps({"sites": {"sid": {"sols": [[361, 378], [91]], "zcam": [34]}}, "groups": {}}))
    err, _ = S.validate_site_table(f)
    assert err and "list of such ranges" in err[0]


def test_scan_accepts_a_folder_of_the_second_visit(tmp_path):
    from mppp.waypoints import sites as S
    w = tmp_path / "mars2020_sol_0361_sid_colmap"
    (w / "colmap" / "sparse" / "m").mkdir(parents=True)
    (w / "colmap" / "sparse" / "m" / "images.bin").write_bytes(b"")
    ims = [{"sol": 95, "instrument": "NL", "station": "S003D0100"}] * 12
    (w / "colmap" / "project.json").write_text(json.dumps({"images": ims, "settings": {"reconstruction": {"path": "sparse/m"}}}))
    rows = S.scan_scapes(tmp_path, {"sid": [[361, 378], [91, 101]]})
    assert rows[0]["ok"] and rows[0]["site_sol_ranges"] == [[361, 378], [91, 101]] and rows[0]["site_sols"] == [361, 378]
    rows = S.scan_scapes(tmp_path, {"sid": [361, 378]})
    assert not rows[0]["ok"] and "no image in the site's sols 361-378" in rows[0]["reason"]


def test_find_imgs_over_several_visits(tmp_path):
    from mppp.select import find_imgs, in_sol_range, sol_ranges
    assert sol_ranges((1, 2)) == [(1, 2)] and sol_ranges([[1, 2], [5, 6]]) == [(1, 2), (5, 6)]
    assert in_sol_range(5, [[1, 2], [5, 6]]) and not in_sol_range(3, [[1, 2], [5, 6]]) and in_sol_range(3, (1, 4))
    names = {s: f"NLF_{s:04d}_0700000000_000RAD_N{s:03d}0100NCAM00100_0A0095J01.IMG" for s in (95, 365, 700)}
    for n in names.values():
        (tmp_path / n).write_bytes(b"")
    got = find_imgs(tmp_path, ["NLF"], sol_range=[[361, 378], [91, 101]])
    assert [p.name for p in got] == [names[95], names[365]]
    assert [p.name for p in find_imgs(tmp_path, ["NLF"], sol_range=(361, 378))] == [names[365]]


def test_run_key_carries_the_visits(tmp_path, monkeypatch):
    from mppp import runner
    from mppp.waypoints import sites as S
    table = {"sites": {"sid": {"sols": [[361, 378], [91, 101]], "zcam": [34], "settings": {}},
                       "one": {"sols": [700, 712], "zcam": [34], "settings": {}}}, "groups": {}}
    monkeypatch.setattr(S, "load_site_table", lambda path=None: table)
    w = tmp_path / "mars2020_sol_0361_sid_colmap"
    w.mkdir()
    st = runner.work_settings(w, "sid")
    assert st["SITE_SOL_RANGES"] == [[361, 378], [91, 101]]
    assert "SITE_SOL_RANGES" not in runner.work_settings(tmp_path / "mars2020_sol_0700_one_colmap", "one")


# ------------------------------------------------------------------------------------------------- Navcam rig epochs
def _rig_entry():
    return {"ref": "NL", "sensor": "NR", "n_pairs": 10, "R_sensor_from_ref": np.eye(3).tolist(),
            "t_sensor_from_ref": [-0.424, 0.0, 0.0], "baseline_m": 0.424, "rotation_source": "CAHV"}


def test_navcam_rig_epochs_split_the_visits(tmp_path):
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import navcam_rig_epochs
    images = [{"instrument": "NL", "sol": s, "camera_temperature_degC": -20.0} for s in (91, 95, 101, 360, 365, 378)] + \
             [{"instrument": "NR", "sol": s, "camera_temperature_degC": -21.0} for s in (91, 95, 101, 360, 365, 378)] + \
             [{"instrument": "ZL034", "sol": 365}]
    rig = {"N": _rig_entry()}
    cams = {"NL": {"model": "FULL_OPENCV", "params": [1.0] * 12, "source": "consensus"},
            "NR": {"model": "FULL_OPENCV", "params": [2.0] * 12, "source": "consensus"}}
    # a rig file with a pitch drift: each epoch starts at its own sol
    rf = tmp_path / "M2020_N_rig.json"
    rf.write_text(json.dumps({"R_sensor_from_ref": np.eye(3).tolist(), "t_sensor_from_ref_m": [-0.424, 0, 0],
                              "drift": {"sol0": 932.0, "pitch_mdeg_per_sol": 0.0055, "yaw_mdeg_per_sol": 0.0,
                                        "roll_mdeg_per_sol": 0.0}, "convention": "NL reference"}))
    info = navcam_rig_epochs(images, rig, 100.0, {"rig": {"file": str(rf)}}, cameras=cams)
    assert set(rig) == {"N", "N#1"} and [e["sols"] for e in info["epochs"]] == [[91, 101], [360, 378]]
    assert info["epochs"][1] == {**info["epochs"][1], "rig_key": "N#1", "cameras": ["NL#1", "NR#1"], "images": 6}
    assert rig["N#1"]["ref"] == "NL#1" and rig["N#1"]["sensor"] == "NR#1" and rig["N"]["ref"] == "NL"
    assert set(cams) == {"NL", "NR", "NL#1", "NR#1"} and cams["NR#1"]["params"] == [2.0] * 12
    assert cams["NL#1"]["group"] == "NL" and cams["NL#1"]["epoch"] == 1 and "rig epoch 1" in cams["NL#1"]["source"]
    assert {r["instrument"] for r in images if r["sol"] < 200 and r.get("rig_epoch") == 0} == {"NL", "NR"}
    assert {r["instrument"] for r in images if r["sol"] > 300 and r.get("rig_epoch") == 1} == {"NL#1", "NR#1"}
    assert "rig_epoch" not in images[-1] and images[-1]["instrument"] == "ZL034"       # Mastcam-Z untouched
    p0 = Rotation.from_matrix(np.asarray(rig["N"]["R_sensor_from_ref"])).as_rotvec(degrees=True)[0] * 1e3
    p1 = Rotation.from_matrix(np.asarray(rig["N#1"]["R_sensor_from_ref"])).as_rotvec(degrees=True)[0] * 1e3
    assert p0 == pytest.approx(0.0055 * (95 - 932), abs=1e-6) and p1 == pytest.approx(0.0055 * (365 - 932), abs=1e-6)
    assert "rig epoch 1 (sols 360-378" in rig["N#1"]["rotation_source"] and rig["N"]["epoch"]["index"] == 0
    # one visit (or visits closer than the gap): nothing changes
    rig2 = {"N": _rig_entry()}
    one = [{"instrument": "NL", "sol": s} for s in (1408, 1412, 1359)]
    assert navcam_rig_epochs(one, rig2, 100.0, cameras={}) is None and set(rig2) == {"N"}
    assert all(r["instrument"] == "NL" and r["rig_epoch"] == 0 for r in one)


def test_eye_key():
    from mppp.sfm.thermal import eye_key
    assert eye_key("NL#1") == "NL" and eye_key("NR#2_T-20..-10") == "NR" and eye_key("NL_T-20..-10") == "NL"
    assert eye_key("ZL034_F00600") == "ZL034_F00600" and eye_key("NL") == "NL"


def test_split_rig_by_epoch_gap_is_a_parameter():
    from mppp.sfm.navcal import sol_epochs
    assert sol_epochs([91, 95, 101, 360, 365], 100) == [(91, 101), (360, 365)]
    assert sol_epochs([1359, 1360, 1408, 1412], 100) == [(1359, 1412)]
    assert sol_epochs([1359, 1360, 1408, 1412], 30) == [(1359, 1360), (1408, 1412)]


# ------------------------------------------------------------------------------------------------- Mastcam-Z line form
def _bins(monkeypatch, pp0=(820.0, 600.0), cx_bins=820.0, form=None):
    import pycolmap
    from mppp.sfm import zcam
    rec = pycolmap.Reconstruction()
    cams, db, obs = {}, {}, {}
    rng = np.random.default_rng(1)
    cid = 1
    for eye in ("L", "R"):
        g = f"Z{eye}034"
        for st, foci in (("backlash", [100, 300, 500, 700, 900, 1100]), ("regular", [200, 600, 800])):
            for f in foci:
                k = 1.0 / 1.0095 if st == "regular" else 1.0
                p = [4700.0 * k + 0.050 * (f - 600) + rng.normal(0, 0.3),          # fx slope 0.050
                     4710.0 * k + 0.060 * (f - 600) + rng.normal(0, 0.3),          # fy slope 0.060: the common one is between
                     cx_bins + 0.0005 * (f - 600) + rng.normal(0, 0.05),
                     600.0 + 0.0017 * (f - 600) + rng.normal(0, 0.05),
                     -0.47 + 3e-5 * (f - 600), 0.7, 0, 0]
                rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200, params=np.array(p, float)))
                key = f"{g}_F{f:05d}" + ("_reg" if st == "regular" else "")
                cams[key] = {"group": g, "focus_count_median": float(f), "backlash_state": st}
                db[key] = cid
                obs[cid] = 1000
                cid += 1
    settings = {"database": {"cameras": db},
                "zcam_line_params": {"ZL": ["fx", "fy", "k1", "k2"], "ZR": ["fx", "fy", "cx", "cy", "k1", "k2"]}}
    if form is not None:
        settings["zcam_line_form"] = form
    proj = SimpleNamespace(cameras=cams, images=[], settings=settings)
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    model = {"cameras": {g: {"reference_focus": 600.0, "f0_px": 4705.0, "label_f0_px": 4659.0, "slope_px_per_count": 0.05,
                             "aspect": 1.002, "pp0_px": list(pp0) if g == "ZR034" else [876.7, 559.1]}
                         for g in ("ZL034", "ZR034")}}
    return rec, proj, model, db


def test_line_form_lookup():
    from mppp.sfm.zcam import LINE_FORM_V0P75, line_form_for
    p = SimpleNamespace(settings={"zcam_line_params": {"ZR": ["fx", "fy", "cx", "cy", "k1", "k2"]}})
    assert line_form_for(p, "ZR034") == LINE_FORM_V0P75["ZR"]
    p.settings["zcam_line_form"] = "v0p74"
    assert set(line_form_for(p, "ZR034").values()) == {"free"}
    p.settings["zcam_line_form"] = {"ZR": {"cy": "constant"}}
    assert line_form_for(p, "ZR034")["cy"] == "constant" and line_form_for(p, "ZR034")["fx"] == "free"
    p.settings["zcam_line_form"] = {"ZR": {"cy": "odd"}}
    with pytest.raises(ValueError, match="unknown forms"):
        line_form_for(p, "ZR034")


def test_refit_lines_on_the_v0p75_form(monkeypatch):
    from mppp.sfm.zcam import refit_focus_lines
    rec, proj, model, db = _bins(monkeypatch)
    out = refit_focus_lines(rec, proj, model=model)
    zr, zl = out["ZR034"]["lines"], out["ZL034"]["lines"]
    assert out["ZR034"]["form"]["cx"] == "pinned" and out["ZR034"]["form"]["fx"] == "common"
    # one slope for fx and fy per eye, between the two true slopes; the regular state follows it
    for ln in (zr, zl):
        assert ln["backlash"]["fx"]["b"] == ln["backlash"]["fy"]["b"]
        assert 0.050 < ln["backlash"]["fx"]["b"] < 0.060 and ln["backlash"]["fx"]["slope_source"].startswith("common slope")
        assert ln["regular"]["fx"]["b"] == ln["backlash"]["fx"]["b"] and ln["regular"]["fx"]["a"] == pytest.approx(4700 / 1.0095, abs=0.6)
        assert ln["backlash"]["k1"]["b"] == 0.0 and ln["backlash"]["k1"]["a"] == pytest.approx(-0.47, abs=1e-6)   # constant
        assert ln["backlash"]["k2"]["b"] == 0.0
    assert zr["backlash"]["cx"] == {**zr["backlash"]["cx"], "a": 820.0, "b": 0.0}         # pinned, no slope
    assert zr["backlash"]["cx"]["a_source"].startswith("pinned")
    assert zr["backlash"]["cy"]["a"] == 600.0 and zr["backlash"]["cy"]["b"] == pytest.approx(0.0017, abs=2e-4)   # pinned level, slope
    assert "cx" not in zl["backlash"]                                                     # ZL: the reference pp
    p = rec.cameras[db["ZR034_F00300"]].params                                           # every bin set on the lines
    assert p[2] == pytest.approx(820.0) and p[3] == pytest.approx(600.0 + zr["backlash"]["cy"]["b"] * (300 - 600))
    assert p[4] == pytest.approx(zr["backlash"]["k1"]["a"]) and p[0] == pytest.approx(zr["backlash"]["fx"]["a"] + zr["backlash"]["fx"]["b"] * (300 - 600))


def test_refit_lines_free_form_and_foreign_pp0(monkeypatch):
    from mppp.sfm.zcam import PIN_MAX_OFFSET_PX, refit_focus_lines
    rec, proj, model, _ = _bins(monkeypatch, form="v0p74")
    out = refit_focus_lines(rec, proj, model=model)
    zr = out["ZR034"]["lines"]["backlash"]
    assert zr["fx"]["b"] != zr["fy"]["b"] and zr["fx"]["b"] == pytest.approx(0.050, abs=0.003) and zr["fy"]["b"] == pytest.approx(0.060, abs=0.003)
    assert zr["cx"]["b"] == pytest.approx(0.0005, abs=2e-4) and zr["k1"]["b"] == pytest.approx(3e-5, rel=0.05)
    # a model whose pp0 does not belong to these bins (a synthetic block against the shipped model): not pinned
    rec, proj, model, _ = _bins(monkeypatch, pp0=(820.0 + PIN_MAX_OFFSET_PX + 50, 600.0))
    out = refit_focus_lines(rec, proj, model=model)
    zr = out["ZR034"]["lines"]["backlash"]
    assert zr["cx"]["a"] == pytest.approx(820.0, abs=0.5) and zr["cx"]["b"] == 0.0 and "not pinned" in zr["cx"]["a_source"]


def test_split_by_focus_holds_pinned_params():
    from mppp.sfm.project import _split_by_focus
    base = {"model": "FULL_OPENCV", "params": [4700, 4700, 743.2, 664.0, -0.47, 0.7, 0, 0, 0, 0, 0, 0], "source": "label",
            "fixed_params": [], "pp_at_focus": 600.0}
    model = {"cameras": {"ZR034": {"f0_px": 4720.0, "reference_focus": 600.0, "slope_px_per_count": 0.064, "aspect": 1.0025,
                                   "pp0_px": [743.2, 664.0], "pp": {"cx_px_per_count": 0.0, "cy_px_per_count": 0.0017},
                                   "lines": {"reference_focus": 600.0,
                                             "backlash": {"fx": {"a": 4722.9, "b": 0.064}, "fy": {"a": 4734.7, "b": 0.064},
                                                          "cx": {"a": 743.2, "b": 0.0}, "cy": {"a": 664.0, "b": 0.0017},
                                                          "k1": {"a": -0.468, "b": 0.0}, "k2": {"a": 0.875, "b": 0.0}}}}}}
    rows = [{"camera_group": "ZR034", "focus_count": 900.0, "label_f_px": 4660.0, "focus_state": "backlash", "sol": 500,
             "instrument": "ZR034"} for _ in range(3)]
    lp = {"ZR": ["fx", "fy", "cx", "cy", "k1", "k2"]}
    out = _split_by_focus(rows, {"ZR034": dict(base)}, {"ZR034": "Z"}, 30.0, "focal_pp", model=model, line_params=lp)
    c = next(iter(out.values()))
    assert "cx" in c["fixed_params"] and "cy" not in c["fixed_params"] and "k1" not in c["fixed_params"]   # cx pinned
    assert c["params"][2] == pytest.approx(743.2) and c["params"][3] == pytest.approx(664.0 + 0.0017 * 300)
    assert c["params"][0] == pytest.approx(4722.9 + 0.064 * 300)
    out = _split_by_focus(rows, {"ZR034": dict(base)}, {"ZR034": "Z"}, 30.0, "focal_pp", model=model, line_params=lp,
                          line_form="v0p74")
    assert "cx" not in next(iter(out.values()))["fixed_params"]                        # the free v0p74 bins


# ------------------------------------------------------------------------------------------------- calibration
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


def test_calibrate_on_the_form_with_rig_residuals():
    from mppp.sfm.zcam_calib import calibrate
    infos = [_block_lines(0.062, 0.066, 743.15, 663.96, T, s)          # cx, cy at pp0: the v0p75 runs pin them
             for T, s in ((-20.0, 400), (-10.0, 700), (0.0, 1000), (-15.0, 1300))]
    model = {"cameras": {"ZR034": {"reference_focus": 600.0, "f0_px": 4728.0, "aspect": 1.0025, "slope_px_per_count": 0.064,
                                   "pp0_px": [743.15, 663.96], "distortion": {"names": ["k1", "k2"], "params": [-0.468, 0.875]}},
                         "ZL034": {"reference_focus": 600.0, "f0_px": 4721.0, "aspect": 1.0022, "slope_px_per_count": 0.05,
                                   "pp0_px": [876.7, 559.1], "distortion": {"names": ["k1", "k2"], "params": [-0.46, 0.635]}}}}
    m = calibrate(infos, model, min_blocks_for_T=99)
    zr = m["cameras"]["ZR034"]["lines"]["backlash"]
    assert zr["fx"]["b"] == zr["fy"]["b"] and 0.062 < zr["fx"]["b"] < 0.066            # one slope per eye
    assert zr["cx"] == {"a": 743.15, "b": 0.0} and zr["cy"]["a"] == 663.96 and zr["cy"]["b"] == pytest.approx(0.0016)
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


# ------------------------------------------------------------------------------------------------- backlash fallback
def test_bin_fallback_classifies_groups_without_image_fits():
    from mppp.sfm.backlash import BIN_FALLBACK_SD, classify_groups
    ims = []
    for sol, seq, foc, cam in ((1624, "ZCAM09705", 792.0, "ZR034_F00792"), (1618, "ZCAM09697", 1002.0, "ZR034_F01002")):
        ims += [{"name": f"ZR0_{sol}_{k}_{seq}.png", "instrument": cam, "camera_group": "ZR034", "eye": "R", "sol": sol,
                 "sequence": seq, "focus_count": foc, "label_f_px": 4650.0} for k in range(4)]
    proj = SimpleNamespace(images=ims, settings={})
    fits = {ims[4]["name"]: {"ratio_label": 1.0100, "sd_ratio": 0.0002, "observations": 2000}}   # one image of the second
    fb = {"ZR034_F00792": {"f_px": 4650.0 * 1.0005, "observations": 15000}}      # the first: its bin sits at the label
    rows = classify_groups(proj, fits, model=None, nav_scale=1.0, known=False, sequence_rule=False, measured_ratio=False,
                           bin_fallback=fb)
    by = {r["sequence"]: r for r in rows}
    assert by["ZCAM09705"]["fit_source"] == "bin camera" and by["ZCAM09705"]["sd"] == BIN_FALLBACK_SD
    assert by["ZCAM09705"]["state"] == "regular" and by["ZCAM09705"]["fitted"] == 0
    assert by["ZCAM09697"]["state"] == "backlash" and "fit_source" not in by["ZCAM09697"]
    rows = classify_groups(proj, fits, model=None, nav_scale=1.0, known=False, sequence_rule=False, measured_ratio=False)
    assert {r["sequence"]: r["state"] for r in rows}["ZCAM09705"] == "undecided"           # without the fallback
    # a bin shared with another sequence at the same focus count does not stand for one group
    ims.append(dict(ims[0], name="other.png", sequence="ZCAM09800"))
    rows = classify_groups(proj, fits, model=None, nav_scale=1.0, known=False, sequence_rule=False, measured_ratio=False,
                           bin_fallback=fb)
    assert {r["sequence"]: r["state"] for r in rows}["ZCAM09705"] == "undecided"


def test_append_found_survives_a_group_without_a_fit(tmp_path):
    from mppp.sfm.backlash import append_found
    f = tmp_path / "found.csv"
    n = append_found(f, [{"names": ["a.png"], "sol": 1, "sequence": "ZCAM0", "focus_count": 5.0, "ratio": None, "z": None,
                          "sequence_z": -4.0, "sequence_state": "regular", "own_state": "undecided"}], site="s")
    txt = f.read_text()
    assert n == 1 and "by its sequence" in txt and "z -4.0" in txt


# ------------------------------------------------------------------------------------------------- pipeline
def test_pipeline_consensus_takes_the_group_only(tmp_path, monkeypatch):
    import importlib.util, argparse
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    from mppp.sfm import health as H, sites as S
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


def test_shipped_sites_and_model_v0p75():
    from mppp.waypoints import sites as S
    from mppp.sfm.project import zcam_focus_model
    t = S.load_site_table()
    r = S.load_site_ranges()
    assert r["chal_rocks"] == [(361, 378), (91, 101)] and r["threeforks_large"][0] == (652, 693)
    assert r["enchanted_lake"] == [(556, 590), (421, 423)] and S.SITES["chal_rocks"] == (361, 378)
    g = t["groups"]
    assert "olifants" not in g["navcam_consensus"] and "cheyava_falls" in g["navcam_consensus"]
    assert "threeforks_large" in g["navcam_consensus"] and "threeforks_north" not in g["navcam_consensus"]
    assert "zcam34_consensus_v0p82" not in g and "threeforks_north" in g["zcam34_consensus"]   # 19 Oct pm
    assert "threeforks_large" not in g["zcam34_consensus"]
    m = zcam_focus_model()
    assert m.get("form") == "v0p75" and m["rig"]["translation_source"].startswith("CAHV")
    for eye in ("ZL034", "ZR034"):
        c = m["cameras"][eye]
        bk = c["lines"]["backlash"]
        assert bk["fx"]["b"] == bk["fy"]["b"] and bk["k1"]["b"] == 0.0 and len(c["pp0_px"]) == 2
        assert c["lines"]["regular"]["fx"]["a"] < bk["fx"]["a"]
    assert m["cameras"]["ZR034"]["lines"]["backlash"]["cx"]["b"] == 0.0
    assert 0.001 < m["cameras"]["ZR034"]["lines"]["backlash"]["cy"]["b"] < 0.003


def test_database_puts_epoch_frames_on_their_own_rig(tmp_path):
    """Navcam images of a later rig epoch (cameras NL#1, NR#1, rig N#1) get frames on that epoch's rig."""
    import pycolmap
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import _synthetic
    from mppp.sfm.database import build_database
    from mppp.sfm.reconstruction import initial_reconstruction
    proj, truth, *_ = _synthetic(tmp_path)
    nav = [r for r in proj.images if r["instrument"] in ("NL", "NR")]
    assert len(nav) >= 4
    last = nav[-2:]                                                # the last stereo pair is a later visit
    for r in last:
        r["instrument"], r["rig_epoch"] = r["instrument"] + "#1", 1
    for eye in ("NL", "NR"):
        proj.cameras[f"{eye}#1"] = dict(proj.cameras[eye], group=eye, epoch=1)
    proj.rig["N#1"] = dict(proj.rig["N"], ref="NL#1", sensor="NR#1", epoch={"index": 1})
    fdb = pycolmap.Database.open(str(proj.features_db))
    cam = fdb.write_camera(pycolmap.Camera(model="PINHOLE", width=100, height=100, params=[100, 100, 50, 50]))
    rng = np.random.default_rng(0)
    for r in proj.images:
        iid = fdb.write_image(pycolmap.Image(name=r["name"], camera_id=cam))
        fdb.write_keypoints(iid, rng.uniform(0, 100, (20, 2)).astype(np.float32))
        fdb.write_descriptors(iid, pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT,
                                                             rng.integers(0, 255, (20, 128)).astype(np.uint8)))
    fdb.close()
    summary = build_database(proj)
    assert set(summary["rig_ids"]) >= {"N", "N#1"} and summary["rig_ids"]["N"] != summary["rig_ids"]["N#1"]
    assert {"NL#1", "NR#1"} <= set(summary["cameras"])
    rec = initial_reconstruction(proj)
    by_name = {im.name: im for im in rec.images.values()}
    rid_last = {rec.frames[by_name[r["name"]].frame_id].rig_id for r in last}
    rid_first = {rec.frames[by_name[r["name"]].frame_id].rig_id for r in nav[:-2]}
    assert rid_last == {summary["rig_ids"]["N#1"]} and rid_first == {summary["rig_ids"]["N"]}
    assert rec.rigs[summary["rig_ids"]["N#1"]].num_sensors() == 2
    assert {by_name[r["name"]].camera_id for r in last} == {summary["cameras"]["NL#1"], summary["cameras"]["NR#1"]}
    assert rec.num_reg_images() == len(proj.images)
    # the consensus sees the epoch cameras as their eyes
    from mppp.sfm.thermal import eye_key
    assert {eye_key(k) for k in summary["cameras"] if k.startswith("N")} == {"NL", "NR"}


def test_reconstruct_with_two_rig_epochs_and_the_consensus_merge(tmp_path, monkeypatch):
    """A Navcam block whose second station is a later visit: two rigs (N, N#1 with cameras NL#1, NR#1) through
    stage 1 and the thermal stage, both refined to the true rig; the consensus merge maps the epoch cameras to
    their eyes and keeps every image."""
    pytest.importorskip("pyceres")
    import copy
    import pycolmap
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import _build_rec, _synthetic
    from mppp.sfm import navcal as NC
    from mppp.sfm import reconstruction as R
    from mppp.sfm.thermal import image_temperatures
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    for r in proj.images:
        later = r["station"].endswith("0100")
        r["camera_temperature_degC"] = -32.0 if not later else -14.0
        r["sol"] = 95 if not later else 365
        if later:
            r["instrument"], r["rig_epoch"] = r["instrument"] + "#1", 1
        else:
            r["rig_epoch"] = 0
    for eye in ("NL", "NR"):
        proj.cameras[f"{eye}#1"] = dict(proj.cameras[eye], group=eye, epoch=1)
    proj.rig["N#1"] = dict(proj.rig["N"], ref="NL#1", sensor="NR#1", epoch={"index": 1, "sols": [365, 365]})
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2, "NL#1": 3, "NR#1": 4}, "rig_ids": {"N": 1, "N#1": 2}}
    proj.settings["navcam_rig_epochs"] = {"gap_sols": 100.0, "epochs": [{"index": 0, "rig_key": "N"}, {"index": 1, "rig_key": "N#1"}]}
    # the synthetic reconstruction with the later frames on a second rig and the camera copies
    for r in proj.images:                                    # _build_rec keys cameras by NL / NR
        r["instrument"] = r["instrument"].split("#")[0]
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    sens = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)        # noqa: E731
    for cid in (1, 2):
        c = rec0.cameras[cid]
        rec0.add_camera(pycolmap.Camera(camera_id=cid + 2, model=c.model, width=c.width, height=c.height, params=np.array(c.params)))
    rig2 = pycolmap.Rig(rig_id=2)
    rig2.add_ref_sensor(sens(3))
    rig2.add_sensor(sens(4), rec0.rigs[1].sensor_from_rig(sens(2)))
    rec0.add_rig(rig2)
    later_names = {r["name"] for r in proj.images if r["rig_epoch"] == 1}
    new = pycolmap.Reconstruction()
    for c in rec0.cameras.values():
        new.add_camera(c)
    for rg in rec0.rigs.values():
        new.add_rig(rg)
    for fid, fr in rec0.frames.items():
        later = any(rec0.images[d.id].name in later_names for d in fr.data_ids)
        nf = pycolmap.Frame(frame_id=fid, rig_id=2 if later else 1)
        for d in fr.data_ids:
            nf.add_data_id(pycolmap.data_t(sensor_id=sens(d.sensor_id.id + (2 if later else 0)), id=d.id))
        nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
    for iid, im in rec0.images.items():
        later = im.name in later_names
        ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]),
                            camera_id=im.camera_id + (2 if later else 0), image_id=iid)
        ni.frame_id = im.frame_id
        new.add_image(ni)
    for fid in rec0.reg_frame_ids():
        new.register_frame(fid)
    for pid, p in rec0.points3D.items():
        new.add_point3D(p.xyz, p.track, np.zeros(3, np.uint8))
    for r in proj.images:
        if r["rig_epoch"] == 1:
            r["instrument"] += "#1"
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=8, verbose=False,
                        navcam_intrinsics="refine", refine_rig="rotation", temperatures=image_temperatures(proj),
                        thermal_bins_deg=10, thermal_min_images=4, out_name="cahv_ba")
    th = proj.settings["thermal"]
    eyes = {r["eye"] for r in th["rows"]}
    assert {"NL", "NR", "NL#1", "NR#1"} <= eyes                           # bins per epoch camera
    single = pycolmap.Reconstruction(str(proj.root / "sparse" / "cahv_ba_single"))
    assert len(single.cameras) == 4 and {rid for rid, rg in single.rigs.items() if rg.num_sensors() == 2} == {1, 2}
    R_rel = np.asarray(rigT[0])
    for rid in (1, 2):                                                    # both rigs refined to the true rotation
        rg = single.rigs[rid]
        sid = next(iter(rg.non_ref_sensors))
        d = Rotation.from_matrix(np.asarray(rg.sensor_from_rig(sid).rotation.matrix()) @ R_rel.T).as_rotvec()
        assert np.degrees(np.linalg.norm(d)) * 1e3 < 20.0
    # the consensus merge: every image, on the eyes' cameras
    proj.settings["database"]["cameras"] = {"NL": 1, "NR": 2, "NL#1": 3, "NR#1": 4}
    temps = {r["name"]: {"T": r["camera_temperature_degC"]} for r in proj.images}
    sc = NC.Scape("blk", proj.root, proj, single, temps)
    jcams = {k: pycolmap.Camera(model=single.cameras[i].model, width=5120, height=3840, params=np.array(single.cameras[i].params))
             for k, i in (("NL", 1), ("NR", 2))}
    merged, mproj, idx = NC.merge_scapes([sc], jcams, (R_rel, np.asarray(rigT[1])), points_per_scape=200, seed=0)
    assert merged.num_reg_images() == len(proj.images) and {im.camera_id for im in merged.images.values()} == {1, 2}


def test_pipeline_zcam_group_option(tmp_path):
    import importlib.util, argparse
    spec = importlib.util.spec_from_file_location("rp", ROOT / "scripts" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    assert rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=True)).zcam_group == "zcam34_consensus"
    assert rp.Pipeline(argparse.Namespace(root=str(tmp_path), dry_run=True, zcam_group="zcam34_dev")).zcam_group == "zcam34_dev"
    from mppp.waypoints.sites import site_group
    dev = site_group("zcam34_dev")
    from mppp.waypoints.sites import load_site_table, site_zooms
    t = load_site_table()                     # v0p82: Christian's dev group may use a smaller site of a consensus block
    assert len(dev) == 6 and all(34 in site_zooms(t["sites"][s]) for s in dev)


def test_run_jobs_runs_only_mppp_scripts(tmp_path):
    import subprocess
    jobs = tmp_path / "_jobs"
    jobs.mkdir()
    (jobs / "01_check.job").write_text("# a comment\ncheck_sites.py\n")
    (jobs / "02_bad.job").write_text("python -c print(1)\n")
    (jobs / "03_bad.job").write_text("/tmp/evil.py --root x\n")
    (jobs / "04_done.job").write_text("check_sites.py\n")
    (jobs / "04_done.done").write_text("{}")                                     # already run: skipped
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_jobs.py"), "--root", str(tmp_path), "--once"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "01_check.job: exit 0" in r.stdout and "04_done" not in r.stdout
    d = json.loads((jobs / "01_check.done").read_text())
    assert d["exit_code"] == 0 and d["running"] is False and "OK:" in (jobs / "01_check.log").read_text()
    assert json.loads((jobs / "02_bad.done").read_text())["exit_code"] == 2
    assert "not run" in (jobs / "03_bad.log").read_text()
    assert (jobs / "watcher.json").is_file()
