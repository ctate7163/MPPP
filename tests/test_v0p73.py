"""v0p73: the Mastcam-Z fx, fy, cx, cy lines per focus state, the low-backlash found list, the Mastcam-Z lens
temperature, the camera models in the run key and the matching retry (the camera form and its checks live in
mppp.calib since 0.90, tested in test_calib.py)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------- form and convert
# --------------------------------------------------------------------------------------------- consensus pieces
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
        "zcam_line_form": {"ZL": {"fx": "free", "fy": "free"},
                           "ZR": {"fx": "free", "fy": "free", "cx": "free", "cy": "free"}}})   # free lines (a study)
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
    from mppp.labels import instrument_temperatures, read_pds
    f = ROOT / "tests" / "data" / "m20" / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"
    t = instrument_temperatures(read_pds(f, load_image=False)[0])
    assert set(t) >= {"DEA", "HEAD_FPA", "HEAD_HTR_1", "HEAD_HTR_2"}
    assert t["HEAD_HTR_1"] == pytest.approx(-13.8094)


# --------------------------------------------------------------------------------------------- run key, matching
def test_run_key_holds_the_camera_models(monkeypatch):
    import mppp.calib
    import mppp.calib.models
    from mppp import runner
    assert runner.ALIGN_RULES >= 73                     # v0p82: 82
    k1 = runner.run_key({"A": 1}, "pds", "")
    for mod in (mppp.calib, mppp.calib.models):
        monkeypatch.setattr(mod, "models_fingerprint", lambda folder=None: "other")
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
    assert (ROOT / "studies" / "calibration" / "prepare_next_run.py").is_file()      # 0.90: a calibration study
    assert not (ROOT / "scripts" / "prepare_next_run.py").exists()
    assert not (ROOT / "scripts" / "windows" / "prepare_next_run.bat").exists()


def test_notebook_settings():
    nb3 = (ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8")
    assert 'ZCAM_BIN_REFINE   = \\"focal_pp\\"' in nb3 and 'ZCAM_BACKLASH     = \\"list_split\\"' in nb3
    assert "ZCAM_LINE_PARAMS" in nb3 and "ZCAM_LOW_BACKLASH_FOUND" in nb3 and "check_form" in nb3


def pycolmap_sensor(cid):
    import pycolmap
    return pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid)
