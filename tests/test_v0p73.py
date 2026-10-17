"""v0p73: the frozen camera form (NL reference: fx, fy per eye vs temperature, NR cx vs temperature, one rig yaw with a
sol rate, k4 = 0), its checks before a run, the Mastcam-Z fx, fy, cx, cy lines per focus state, the low-backlash found
list, the Mastcam-Z lens temperature, the camera models in the run key and the matching retry."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CMODS = ROOT / "src" / "mppp" / "data" / "cmods"
OLD = CMODS / "history" / "v0p60_consensus"
FILES = ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json")


def _copy(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(src / f, dst / f)
    return dst


# ------------------------------------------------------------------------------------------------- form and convert
def test_cameras_in_use_follow_the_form():
    from mppp.sfm.cmods_form import check_form, describe
    rep = check_form(CMODS)
    assert rep["ok"], rep["errors"]
    nl = json.loads((CMODS / FILES[0]).read_text())
    nr = json.loads((CMODS / FILES[1]).read_text())
    assert nl["thermal"]["cx_px_per_degC"] == 0.0 and nr["thermal"]["cx_px_per_degC"] < -0.05
    assert nl["convention"] == nr["convention"] == "NL reference"
    assert {"fx_ppm_per_degC", "fy_ppm_per_degC"} <= set(nl["thermal"])
    assert any("NL reference" in ln for ln in describe(CMODS))


def test_v0p60_models_fail_the_form_and_convert(tmp_path):
    from mppp.sfm.cmods_form import check_form, convert_to_nl_reference
    rep = check_form(OLD)
    assert not rep["ok"] and any("NL principal point" in e for e in rep["errors"])
    out = convert_to_nl_reference(OLD, tmp_path / "conv")
    nl0 = json.loads((OLD / FILES[0]).read_text())
    nr0 = json.loads((OLD / FILES[1]).read_text())
    nl = json.loads((tmp_path / "conv" / FILES[0]).read_text())
    nr = json.loads((tmp_path / "conv" / FILES[1]).read_text())
    a = nl0["thermal"]["cx_px_per_degC"]
    ratio = nr0["params"][0] / nl0["params"][0]
    assert nl["thermal"]["cx_px_per_degC"] == 0.0
    assert nr["thermal"]["cx_px_per_degC"] == pytest.approx(-a * ratio)
    assert nl["params"] == nl0["params"] and nr["params"] == nr0["params"]
    assert check_form(tmp_path / "conv")["ok"]
    assert out["moved"]["cx_px_per_degC"]["NL_was"] == pytest.approx(a)


def test_k4_not_zero_fails_the_form(tmp_path):
    from mppp.sfm.cmods_form import check_form
    d = _copy(CMODS, tmp_path / "k4")
    j = json.loads((d / FILES[0]).read_text())
    j["params"][j["param_names"].index("k4")] = -0.0059
    (d / FILES[0]).write_text(json.dumps(j))
    rep = check_form(d)
    assert not rep["ok"] and any("k4" in e for e in rep["errors"])


def test_held_distortion_with_k4_refuses():
    from mppp.sfm.project import navcam_distortion_terms
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"]
    cam = {"model": "THIN_PRISM_FISHEYE", "params": [2956, 2956, 2592, 1944, .045, -.01, 0, 0, .0024, -.0059, 0, 0],
           "source": "hand promotion"}
    with pytest.raises(ValueError, match="k4"):
        navcam_distortion_terms(dict(cam), "hold", k4="zero")
    c = navcam_distortion_terms(dict(cam), "refine", k4="zero")          # refined: k4 = 0 and k3 refits
    assert c["params"][names.index("k4")] == 0.0
    c = navcam_distortion_terms(dict(cam, params=cam["params"][:9] + [0.0, 0, 0]), "hold", k4="zero")
    assert "k4" in c["fixed_params"]


def test_thermal_scale_per_axis():
    from mppp.sfm.cmods_form import thermal_scale, thermal_terms
    th = {"ppm_per_degC": 37.0, "fx_ppm_per_degC": 40.0, "fy_ppm_per_degC": 30.0, "T0_degC": -20.0}
    sx, sy = thermal_scale(th, 0.0)
    assert sx == pytest.approx(1 + 40e-6 * 20) and sy == pytest.approx(1 + 30e-6 * 20)
    assert thermal_terms({"ppm_per_degC": 37.0, "T0_degC": -20})["fy_ppm"] == 37.0      # older files: shared value


def test_promotion_check_is_convention_aware(tmp_path):
    """The v0p72 candidate converted to NL reference is the same stereo model: the stereo slope check passes."""
    from mppp.sfm.cmods_form import convert_to_nl_reference
    from mppp.sfm.navcal_consensus import check_candidate
    cand = tmp_path / "cand"
    convert_to_nl_reference(OLD, cand)
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    (cand / "consensus.json").write_text(json.dumps({
        "blocks": {f"b{i}": 1 for i in range(10)}, "iterations": 9, "final_cost": 1.0,
        "sd": {e: {n: 0.01 for n in names} for e in ("NL", "NR")},
        "residuals": {"median_px": 0.15, "rms_px": 0.3, "p95_px": 0.6},
        "thermal": {"yaw_rate_mdeg_per_1000sol": 1.2, "sd": {"yaw_rate": 0.5}}}))
    rep = check_candidate(cand, current_dir=OLD)
    by = {c["check"]: c for c in rep["checks"]}
    assert by["stereo pp thermal slope change"]["ok"] and by["stereo pp thermal slope change"]["value"] < 1e-6
    assert by["form"]["ok"]
    assert by["rig yaw rate"]["value"] == pytest.approx(1.2)
    # the unconverted v0p60 models are not in the form: refused
    old = _copy(OLD, tmp_path / "oldcand")
    shutil.copy(cand / "consensus.json", old / "consensus.json")
    assert not {c["check"]: c for c in check_candidate(old, current_dir=OLD)["checks"]}["form"]["ok"]


def test_prepare_next_run_same_parameters(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pnr", ROOT / "scripts" / "prepare_next_run.py")
    pnr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pnr)
    from mppp.sfm.cmods_form import convert_to_nl_reference
    convert_to_nl_reference(OLD, tmp_path / "a")
    assert pnr.same_parameters(tmp_path / "a", OLD)["same"]          # numbers equal, convention differs
    assert not pnr.same_parameters(CMODS, OLD)["same"]


# --------------------------------------------------------------------------------------------- consensus pieces
def test_keypoint_scales_per_axis():
    from mppp.sfm.navcal import keypoint_scales
    proj = SimpleNamespace(images=[{"name": "a", "image_id": 1, "instrument": "NL"},
                                   {"name": "b", "image_id": 2, "instrument": "NR"}])
    ks = keypoint_scales(proj, {"a": 0.0, "b": -40.0}, 37.0, -20.0, axis_ppm={"NL": (40.0, 30.0), "NR": (10.0, 20.0)})
    assert ks[1] == pytest.approx((1 + 40e-6 * 20, 1 + 30e-6 * 20))
    assert ks[2] == pytest.approx((1 - 10e-6 * 20, 1 - 20e-6 * 20))
    assert keypoint_scales(proj, {"a": 0.0}, 37.0, -20.0)[1] == pytest.approx(1 + 37e-6 * 20)


def test_thermal_values_and_yaw_rate():
    from mppp.sfm.navcal_consensus import (V0P73_TERMS, _axis_ppm, _drift_with_yaw, _joint_kw, _set_term,
                                           _thermal_values)
    v = _thermal_values({"ppm_per_degC": 37.0, "pp_slopes": {"NL": [0, 0], "NR": [-0.05, 0]},
                         "axis_ppm": {"NL": [38, 36], "NR": [37, 35]}, "yaw_rate_mdeg_per_1000sol": 1.5})
    assert v["NR_cx"] == -0.05 and v["NL_fy"] == 36 and v["yaw_rate"] == 1.5
    assert _set_term(v, "f", 40.0)["NR_fy"] == 40.0
    assert _axis_ppm(v) == {"NL": (38.0, 36.0), "NR": (37.0, 35.0)}
    d = _drift_with_yaw({"sol0": 900, "pitch_mdeg_per_sol": 0.005, "yaw_mdeg_per_sol": 0.0,
                         "yaw_early_mdeg_per_sol": 0.0, "knot_sol": 300}, 2.0)
    assert d["yaw_mdeg_per_sol"] == pytest.approx(0.002) and d["yaw_early_mdeg_per_sol"] == pytest.approx(0.002)
    kw = _joint_kw(v, {"sol0": 900}, axis=True)
    assert kw["axis_ppm"]["NL"] == (38.0, 36.0) and kw["drift"]["yaw_mdeg_per_sol"] == pytest.approx(0.0015)
    assert "axis_ppm" not in _joint_kw(v, {"sol0": 900}, axis=False)
    assert "NR_cx" in V0P73_TERMS and "NL_cx" not in V0P73_TERMS


def test_consensus_refuses_nl_cx_in_v0p73(tmp_path):
    from mppp.sfm.navcal_consensus import fit_consensus
    with pytest.raises(ValueError, match="NL reference"):
        fit_consensus({}, thermal_terms=("NL_cx",), form="v0p73", start_dir=CMODS)
    with pytest.raises(ValueError, match="form"):
        fit_consensus({}, thermal_terms=("f",), form="v0p99", start_dir=CMODS)


def test_rig_residual_ols():
    from mppp.sfm.navcal_consensus import _ols
    x = np.array([-30, -20, -10, 0, 10.0])
    r = _ols(x, 0.5 * x + 2 + np.array([0.01, -0.01, 0.0, 0.01, -0.01]))
    assert r["slope"] == pytest.approx(0.5, abs=1e-3) and r["se"] < 0.01
    assert _ols(np.array([1.0, 1.0]), np.array([1.0, 2.0]))["slope"] is None


# --------------------------------------------------------------------------------------------- Mastcam-Z lines
def _zcam_setup(monkeypatch, with_regular=True):
    import pycolmap
    from mppp.sfm import zcam
    rec = pycolmap.Reconstruction()
    cams, db, obs = {}, {}, {}
    rng = np.random.default_rng(0)
    cid = 1
    truth = {"fx": (4700.0, 0.05), "fy": (4702.0, 0.05), "cx": (820.0, 0.002), "cy": (600.0, -0.001)}
    for eye in ("L", "R"):
        g = f"Z{eye}034"
        for st, foci in (("backlash", [100, 300, 500, 700, 900]), ("regular", [200, 600, 800] if with_regular else [])):
            for f in foci:
                k = 1.0 / 1.009 if st == "regular" else 1.0
                p = [truth["fx"][0] * k + truth["fx"][1] * (f - 600) + rng.normal(0, 0.3),
                     truth["fy"][0] * k + truth["fy"][1] * (f - 600) + rng.normal(0, 0.3),
                     truth["cx"][0] + truth["cx"][1] * (f - 600) + (3.0 if eye == "R" else 0.0) + rng.normal(0, 0.05),
                     truth["cy"][0] + truth["cy"][1] * (f - 600) + rng.normal(0, 0.05), 0, 0, 0, 0]
                rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200,
                                               params=np.array(p, float)))
                key = f"{g}_F{f:05d}" + ("_reg" if st == "regular" else "")
                cams[key] = {"group": g, "focus_count_median": float(f), "backlash_state": st}
                db[key] = cid
                obs[cid] = 1000
                cid += 1
    proj = SimpleNamespace(cameras=cams, images=[], settings={
        "database": {"cameras": db}, "zcam_line_params": {"ZL": ["fx", "fy"], "ZR": ["fx", "fy", "cx", "cy"]},
        "zcam_line_form": "v0p74"})                                           # the free v0p73 lines (v0p75: a form)
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    model = {"cameras": {g: {"reference_focus": 600.0, "f0_px": 4701.0, "label_f0_px": 4659.0,
                             "slope_px_per_count": 0.05, "aspect": 1.0} for g in ("ZL034", "ZR034")}}
    return rec, proj, model, db


def test_param_lines_per_state(monkeypatch):
    from mppp.sfm.zcam import focus_line_lines, refit_focus_lines
    rec, proj, model, db = _zcam_setup(monkeypatch)
    out = refit_focus_lines(rec, proj, model=model)
    zr = out["ZR034"]["lines"]
    assert set(zr) == {"backlash", "regular"}
    assert zr["backlash"]["fx"]["a"] == pytest.approx(4700.0, abs=0.5)
    assert zr["backlash"]["fx"]["b"] == pytest.approx(0.05, abs=0.003)
    assert zr["backlash"]["cx"]["a"] == pytest.approx(823.0, abs=0.1)
    assert zr["regular"]["fx"]["a"] == pytest.approx(4700.0 / 1.009, abs=0.6)
    assert set(out["ZL034"]["lines"]["backlash"]) == {"fx", "fy"}            # ZL: the reference principal point
    p = rec.cameras[db["ZR034_F00300"]].params                                # every bin set on its state's line
    ln = zr["backlash"]
    assert p[0] == pytest.approx(ln["fx"]["a"] + ln["fx"]["b"] * (300 - 600))
    assert p[2] == pytest.approx(ln["cx"]["a"] + ln["cx"]["b"] * (300 - 600))
    assert any("regular" in s for s in focus_line_lines(out))


def test_param_lines_regular_without_bins_follows_backlash(monkeypatch):
    from mppp.sfm.zcam import refit_focus_lines
    rec, proj, model, db = _zcam_setup(monkeypatch, with_regular=False)
    out = refit_focus_lines(rec, proj, model=model)
    assert "skipped" in out["ZL034"]["lines"]["regular"]


def test_focal_pp_bins_free_zr_principal_point():
    from mppp.sfm.project import ZCAM_BIN_REFINE_SETTINGS, _split_by_focus
    assert "focal_pp" in ZCAM_BIN_REFINE_SETTINGS
    base = {"model": "OPENCV", "params": [4700, 4700, 824, 600, 0, 0, 0, 0], "source": "label", "fixed_params": []}
    cams = {"ZL034": dict(base), "ZR034": dict(base)}
    rows = [{"camera_group": g, "focus_count": f, "label_f_px": 4660.0, "focus_state": "backlash", "sol": 500,
             "instrument": g} for g in ("ZL034", "ZR034") for f in (100, 105, 400)]
    out = _split_by_focus(rows, cams, {"ZL034": "Z", "ZR034": "Z"}, 30.0, "focal_pp")
    zr = [c for k, c in out.items() if k.startswith("ZR")]
    zl = [c for k, c in out.items() if k.startswith("ZL")]
    assert all("cx" not in c["fixed_params"] and "k1" in c["fixed_params"] for c in zr)
    assert all("cx" in c["fixed_params"] for c in zl)


# --------------------------------------------------------------------------------------------- low-backlash found list
def test_found_list_append_and_load(tmp_path):
    from mppp.sfm.backlash import append_found, is_low_backlash, load_low_backlash, low_backlash_fingerprint
    f = tmp_path / "zcam_low_backlash_found.csv"
    g = {"names": ["ZL0_0500_0711000000_111RAD_N0261154ZCAM07999_0340LMA01.png"], "sol": 500, "sequence": "ZCAM07999",
         "focus_count": 400, "ratio": 1.0004, "z": -9.0}
    fp0 = low_backlash_fingerprint(None, f)
    assert append_found(f, [g], site="test") == 1
    assert append_found(f, [g], site="test") == 0                            # not twice
    lb = load_low_backlash(None, found=f)
    assert lb["found"] == 1 and lb["found_file"] == str(f)
    row = {"stem": "ZL0_0500_0711000000_111RAD_N0261154ZCAM07999_0340LMA01", "camera_group": "ZL034", "sol": 500,
           "sequence": "ZCAM07999"}
    assert is_low_backlash(row, lb)
    assert not is_low_backlash(row, load_low_backlash(None))
    assert low_backlash_fingerprint(None, f) != fp0


def test_backlash_mode_list_split_accepted():
    from mppp.sfm.backlash import backlash_stage
    with pytest.raises(ValueError, match="list_split"):
        backlash_stage(None, None, mode="bogus")


# --------------------------------------------------------------------------------------------- Mastcam-Z temperatures
def test_zcam_optics_temperature():
    from mppp.image import zcam_optics_temperature
    t = {"DEA": 34.7, "HEAD_FPA": -9.6, "HEAD_HTR_1": -13.8, "HEAD_HTR_2": -13.6}
    assert zcam_optics_temperature(t) == pytest.approx(-13.7)
    assert zcam_optics_temperature({"HEAD_FPA": -9.6}) is None


def test_label_instrument_temperatures():
    from mppp.sfm.thermal import label_instrument_temperatures
    f = ROOT / "tests" / "data" / "m20" / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"
    t = label_instrument_temperatures(f)
    assert set(t) >= {"DEA", "HEAD_FPA", "HEAD_HTR_1", "HEAD_HTR_2"}
    assert t["HEAD_HTR_1"] == pytest.approx(-13.8094)


def _sol(name, T_opt, a_fx, sd=0.5, state="backlash"):
    lines = {"ZL034": {"reference_focus": 600.0, "lines": {state: {
        "fx": {"a": a_fx, "b": 0.05, "sd_a": sd, "sd_b": 0.001, "n_bins": 5, "slope_source": "fitted", "rms": 0.4}}}}}
    cam = SimpleNamespace(family="N", n_obs=100, params=np.array([2960.0, 2960.0]), initial=np.array([2956.0, 2956.0]))
    return SimpleNamespace(project={"settings": {"zcam_focus_lines": lines}}, cameras={"NL": cam}, manifest={},
                           images={"z1.png": {"instrument": "ZL034_F00600", "sol": 500, "camera_temperature_degC": -9.0,
                                              "zcam_optics_temperature_degC": T_opt},
                                   "n1.png": {"instrument": "NL", "sol": 500, "camera_temperature_degC": -20.0}})


def test_zcam_block_lines_and_consensus():
    from mppp.sfm.zcam_lines import block_lines, consensus_lines, lines_consensus
    sols = {f"B{i}": _sol(f"B{i}", T, 4700.0 + 0.2 * T) for i, T in enumerate((-30.0, -20.0, -10.0, 0.0))}
    rows = block_lines(sols)
    assert len(rows) == 4 and rows[0]["navcam_scale"] == pytest.approx(2960 / 2956)
    assert rows[0]["T_optics_degC"] == -30.0 and rows[0]["T_navcam_degC"] == -20.0
    cons = lines_consensus(rows)
    e = cons[0]
    assert e["n"] == 4 and e["a_vs_T"]["slope_per_degC"] == pytest.approx(0.2 / (2960 / 2956), rel=1e-3)
    assert consensus_lines(cons)
    one = lines_consensus(rows, blocks=["B0", "B1"])
    assert one[0]["n"] == 2


# --------------------------------------------------------------------------------------------- run key, matching
def test_run_key_holds_the_camera_models(monkeypatch):
    from mppp import runner
    from mppp.sfm import cmods_form
    assert runner.ALIGN_RULES >= 73                     # v0p82: 82
    k1 = runner.run_key({"A": 1}, "pds", "")
    monkeypatch.setattr(cmods_form, "models_fingerprint", lambda folder=None: "other")
    assert runner.run_key({"A": 1}, "pds", "") != k1
    assert runner.process_key({"A": 1}) == runner.process_key({"A": 1})


def test_set_prior_focal(tmp_path):
    import pycolmap
    from mppp.sfm.matching import CALIBRATED_ABORT_SIGNS, set_prior_focal
    db_path = tmp_path / "database.db"
    db = pycolmap.Database.open(str(db_path))
    ids = {}
    for k, model, w, h in (("NL", "SIMPLE_PINHOLE", 5120, 3840), ("ZL034_F00300", "SIMPLE_PINHOLE", 1648, 1200)):
        c = pycolmap.Camera(model=model, width=w, height=h, params=np.array([1000.0, w / 2, h / 2]))
        c.has_prior_focal_length = True
        ids[k] = db.write_camera(c)
    db.close()
    proj = SimpleNamespace(database=db_path, settings={"database": {"cameras": ids}})
    assert set_prior_focal(proj, ("Z",), False) == 1
    db = pycolmap.Database.open(str(db_path))
    flags = {int(c.camera_id): bool(c.has_prior_focal_length) for c in db.read_all_cameras()}
    db.close()
    assert flags[ids["NL"]] and not flags[ids["ZL034_F00300"]]
    assert set_prior_focal(proj, ("Z",), True) == 1
    assert "kMinNumSamples" in CALIBRATED_ABORT_SIGNS


def test_scripts_and_bat_exist():
    assert (ROOT / "scripts" / "prepare_next_run.py").is_file()
    b = (ROOT / "scripts" / "windows" / "prepare_next_run.bat").read_bytes()
    assert b"\r\n" in b and b"prepare_next_run.py" in b


def test_notebook_settings():
    nb3 = (ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8")
    assert 'ZCAM_BIN_REFINE   = \\"focal_pp\\"' in nb3 and 'ZCAM_BACKLASH     = \\"list_split\\"' in nb3
    assert "ZCAM_LINE_PARAMS" in nb3 and "ZCAM_LOW_BACKLASH_FOUND" in nb3 and "check_form" in nb3
    nb4 = (ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8")
    assert 'CONSENSUS_FORM = \\"v0p73\\"' in nb4 and "rig_residuals" in nb4 and "zcam_lines" in nb4


def test_profile_recovers_per_axis_and_nr_cx_slopes(tmp_path):
    """The real joint adjustment: NR keypoints made with fx +30 ppm/degC and cx -0.05 px/degC (fy 0, NL none) -
    the v0p73 profiles find them."""
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm import navcal_consensus as NCC
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path, noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    T0 = -20.0
    temps = {}
    for r in proj.images:
        k = int(r["sclk_key"][:10])
        temps[r["name"]] = -45.0 + 4.0 * k               # one temperature per exposure, -41 .. +3 degC
    true = {"NR_fx": 30.0, "NR_cx": -0.05}
    for r in proj.images:
        if r["instrument"] != "NR":
            continue
        im = rec.images[r["image_id"]]
        p = np.asarray(rec.cameras[im.camera_id].params)
        c = p[2:4]
        dT = temps[r["name"]] - T0
        s = np.array([1 + 1e-6 * true["NR_fx"] * dT, 1.0])
        for j in range(im.num_points2D()):
            u = np.asarray(im.points2D[j].xy)
            im.points2D[j].xy = c + s * (u + np.array([true["NR_cx"], 0.0]) * dT - c)
    vals = {k: 0.0 for k in NCC.THERMAL_TERMS}
    out = {}
    for term in ("NR_cx", "NR_fx", "NR_cx", "NR_fx"):              # two rounds, as THERMAL_ROUNDS = 2
        pr = NCC.profile_thermal(rec, proj, temps, vals, T0, term, drift=None, iterations=30, verbose=False, axis=True)
        out[term] = pr
        vals = NCC._set_term(vals, term, pr["best"])
    print({k: (v["best"], v["sd"]) for k, v in out.items()})
    assert out["NR_cx"]["best"] == pytest.approx(true["NR_cx"], abs=max(4 * out["NR_cx"]["sd"], 0.002))
    assert out["NR_fx"]["best"] == pytest.approx(true["NR_fx"], abs=max(4 * out["NR_fx"]["sd"], 1.5))


def test_rig_residuals_find_a_yaw_offset(tmp_path, monkeypatch):
    """rig_residuals on a synthetic block: the model's rig 2 mdeg off in yaw comes back as a -2 mdeg residual."""
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from helpers import _build_rec, _synthetic
    from mppp.sfm import navcal as NC
    from mppp.sfm import navcal_consensus as NCC
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    d = _copy(CMODS, tmp_path / "cams")
    R_model = Rotation.from_rotvec([0.0, np.radians(2e-3), 0.0]).as_matrix() @ rigT[0]
    rj = json.loads((d / "M2020_N_rig.json").read_text())
    rj["R_sensor_from_ref"] = R_model.tolist()
    rj["drift"] = None
    (d / "M2020_N_rig.json").write_text(json.dumps(rj))
    for e in ("NL", "NR"):                                   # no thermal or early terms in this synthetic block
        j = json.loads((d / f"M2020_{e}_fisheye_tangential.json").read_text())
        j["thermal"] = {"ppm_per_degC": 0.0, "fx_ppm_per_degC": 0.0, "fy_ppm_per_degC": 0.0, "T0_degC": -20.0,
                        "cx_px_per_degC": 0.0, "cy_px_per_degC": 0.0}
        j.pop("early_mission", None)
        (d / f"M2020_{e}_fisheye_tangential.json").write_text(json.dumps(j))
    rec.rigs[1].sensor_from_rig(pycolmap_sensor(2))                 # the block starts at the true rig
    monkeypatch.setattr(NC, "load_scape", lambda name, folder: name)
    monkeypatch.setattr(NC, "merge_scapes", lambda scs, cams, rig, points_per_scape=0, seed=0:
                        (rec, proj, {"temps": {}}))
    res = NCC.rig_residuals({"Synthetic": "x"}, cameras_dir=d, verbose=False, iterations=30)
    row = res["blocks"][0]
    assert "error" not in row, row
    assert row["yaw_mdeg"] == pytest.approx(-2.0, abs=0.3)
    assert abs(row["pitch_mdeg"]) < 0.3 and abs(row["roll_mdeg"]) < 0.3


def pycolmap_sensor(cid):
    import pycolmap
    return pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid)
