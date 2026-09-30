"""v0p42: Mastcam-Z focus model with temperature and sol terms, principal point against focus, HEAD_FPA."""
import copy
import types

import numpy as np
import pytest


def _model():
    from mppp.sfm.project import zcam_focus_model
    return zcam_focus_model()


def test_shipped_focus_model_layout():
    m = _model()
    for g in ("ZL034", "ZR034"):
        c = m["cameras"][g]
        assert c["reference_focus"] == 600.0
        assert 0.05 < c["slope_px_per_count"] < 0.07                 # refit (was 0.046 / 0.048)
        assert 1.006 < c["f0_px"] / c["label_f0_px"] < 1.012          # backlash state, ~0.9 % above the label
        assert c["thermal"]["sensor"] == "HEAD_FPA" and c["thermal"]["f_px_per_degC"] == 0.0
        assert c["trend"]["sol_range"] == [461, 991] and c["trend"]["sol0"] == 700.0
    assert m["cameras"]["ZL034"]["pp"]["cx_px_per_count"] == 0.0
    assert 0.001 < m["cameras"]["ZR034"]["pp"]["cx_px_per_count"] < 0.004
    assert 0.001 < m["cameras"]["ZR034"]["pp"]["cy_px_per_count"] < 0.004


def test_model_focal_terms_and_sol_clamp():
    from mppp.sfm.project import zcam_model_focal
    g = _model()["cameras"]["ZR034"]
    f, t = zcam_model_focal(g, 600.0, T=-15.0, sol=700.0)
    assert f == pytest.approx(g["f0_px"]) and t["thermal"] == 0.0 and t["trend"] == 0.0
    f1, _ = zcam_model_focal(g, 1100.0)
    assert f1 - g["f0_px"] == pytest.approx(500 * g["slope_px_per_count"])
    # the trend does not extrapolate beyond the fitted sols
    a, _ = zcam_model_focal(g, 600.0, sol=991)
    b, _ = zcam_model_focal(g, 600.0, sol=1700)
    c, _ = zcam_model_focal(g, 600.0, sol=100)
    d, _ = zcam_model_focal(g, 600.0, sol=461)
    assert a == pytest.approx(b) and c == pytest.approx(d) and b > c
    # a thermal slope is used when the model has one and the temperature is known
    g2 = copy.deepcopy(g)
    g2["thermal"]["f_px_per_degC"] = 0.5
    e, t2 = zcam_model_focal(g2, 600.0, T=-5.0, sol=700.0)
    assert t2["thermal"] == pytest.approx(5.0) and e == pytest.approx(g["f0_px"] + 5.0)
    assert zcam_model_focal(g2, 600.0, T=None, sol=700.0)[1]["thermal"] == 0.0


def test_pp_shift_about_the_median_focus():
    from mppp.sfm.project import zcam_model_pp_shift
    m = _model()["cameras"]
    assert zcam_model_pp_shift(m["ZL034"], 1200.0, 600.0) == (0.0, 0.0)
    dx, dy = zcam_model_pp_shift(m["ZR034"], 1200.0, 600.0)
    assert dx == pytest.approx(600 * m["ZR034"]["pp"]["cx_px_per_count"]) and dy > 0
    assert zcam_model_pp_shift(m["ZR034"], None, 600.0) == (0.0, 0.0)
    assert zcam_model_pp_shift(None, 1200.0, 600.0) == (0.0, 0.0)


def test_split_by_focus_uses_temperature_sol_and_pp():
    from mppp.sfm.project import _split_by_focus, zcam_model_focal
    m = copy.deepcopy(_model())
    m["cameras"]["ZR034"]["thermal"]["f_px_per_degC"] = 0.3            # exercise the thermal path
    base = {"model": "FULL_OPENCV", "params": [4680.0, 4680.0, 824.0, 600.0, 0, 0, 0, 0, 0, 0, 0, 0],
            "source": "median label CAHVOR", "fixed_params": []}
    rows = []
    for foc, T, sol in ((300, -20.0, 500), (300, -18.0, 500), (900, -10.0, 1500), (900, -12.0, 1500), (600, -15.0, 700)):
        rows.append({"camera_group": "ZR034", "focus_count": foc, "label_f_px": 4690.0,
                     "camera_temperature_degC": T, "sol": sol})
    out = _split_by_focus(rows, {"ZR034": base}, {"ZR034": "Z"}, 30.0, "focal", model=m, hold_f_images=0)
    by = {c["focus_count_median"]: c for c in out.values()}
    g = m["cameras"]["ZR034"]
    for foc, T, sol in ((300, -19.0, 500), (900, -11.0, 1500)):
        c = by[foc]
        f = np.sqrt(c["params"][0] * c["params"][1])
        assert f == pytest.approx(zcam_model_focal(g, foc, T, sol)[0])
        assert c["temperature_median_degC"] == pytest.approx(T) and c["sol_median"] == sol
        # pp moves with focus about the eye's median focus (600 here)
        assert c["params"][2] - 824.0 == pytest.approx(g["pp"]["cx_px_per_count"] * (foc - 600))
        assert c["params"][3] - 600.0 == pytest.approx(g["pp"]["cy_px_per_count"] * (foc - 600))
    assert by[600]["params"][2] == pytest.approx(824.0)
    assert "thermal" in by[900]["source"] and "trend" in by[900]["source"]
    assert by[300]["params"][2] < 824.0 < by[900]["params"][2]


def test_fit_focus_model_recovers_temperature_trend_and_navcam_scale():
    from mppp.sfm.calibration import fit_focus_model
    rng = np.random.default_rng(3)
    rows = []
    for k in range(60):
        foc = rng.uniform(0, 1200)
        T = rng.uniform(-28, -8)
        sol = rng.choice([480, 690, 970])
        scale = {480: 1.0003, 690: 1.0035, 970: 1.0}[sol]
        f = (4720 + 0.06 * (foc - 600) + 0.4 * (T + 15) + 0.02 * (sol - 700)) * scale + rng.normal(0, 0.3)
        rows.append({"scape": f"s{sol}", "group": "ZR034", "refined": True, "observations": 5000, "focus": foc,
                     "state": "backlash", "f_refined_px": f, "fx_refined_px": f / 1.0005, "fy_refined_px": f * 1.0005,
                     "temperature_degC": T, "sol": float(sol), "navcam_scale": scale, "f_label_median_px": None})
    r = fit_focus_model(rows, "ZR034", min_observations=1000, per_scape_offset=False, thermal=True, trend=True,
                        navcam_normalise=True, reference_focus=600.0)
    assert r["f0_px"] == pytest.approx(4720, abs=0.3)
    assert r["slope_px_per_count"] == pytest.approx(0.06, abs=0.001)
    assert r["thermal"]["f_px_per_degC"] == pytest.approx(0.4, abs=0.03)
    assert r["trend"]["f_px_per_sol"] == pytest.approx(0.02, abs=0.002)
    assert r["slope_sd_px_per_count"] < 0.001 and r["aspect"] == pytest.approx(1.001, abs=1e-4)
    # without the Navcam normalisation the Three Forks-like scale leaks into the fit
    r2 = fit_focus_model(rows, "ZR034", min_observations=1000, per_scape_offset=False, thermal=True, trend=True,
                         reference_focus=600.0)
    assert r2["rms_px"] > 3 * r["rms_px"]
    # the old call still works (no terms)
    r3 = fit_focus_model(rows, "ZR034", min_observations=1000)
    assert "thermal" not in r3 and "trend" not in r3


def test_boresight_fit_and_model_json_round_trip():
    from mppp.sfm.calibration import fit_zcam_boresight, fit_focus_model, focus_model_json
    from mppp.sfm.project import zcam_model_focal, zcam_model_pp_shift
    rng = np.random.default_rng(5)
    rows = []
    for k in range(200):
        fl = rng.uniform(0, 1250)
        sc = ["a", "b"][k % 2]
        rows.append({"scape": sc, "focus_left": fl, "focus_right": fl + 40, "temperature_degC": -15.0,
                     "eqx_px": 193 + (0.5 if sc == "b" else 0) + 0.0025 * (fl + 20 - 600) + rng.normal(0, 0.5),
                     "eqy_px": 11 + 0.0018 * (fl + 20 - 600) + rng.normal(0, 0.5),
                     "roll_mdeg": -630 + 0.02 * (fl + 20 - 600) + rng.normal(0, 5)})
    rows[0]["eqx_px"] += 50                                                  # an outlier
    b = fit_zcam_boresight(rows)
    assert b["eqx_px"]["slope_per_count"] == pytest.approx(0.0025, abs=3e-4)
    assert b["eqy_px"]["slope_per_count"] == pytest.approx(0.0018, abs=3e-4)
    assert b["roll_mdeg"]["slope_per_count"] == pytest.approx(0.02, abs=4e-3)
    assert b["eqx_px"]["n_downweighted"] >= 1 and b["eqx_px"]["n"] == 200
    frows = [{"scape": "s", "group": g, "refined": True, "observations": 3000, "focus": x, "state": "backlash",
              "f_refined_px": 4720 + 0.06 * (x - 600), "fx_refined_px": 4720 + 0.06 * (x - 600),
              "fy_refined_px": 4720 + 0.06 * (x - 600), "f_label_median_px": 4680 + 0.05 * (x - 600)}
             for g in ("ZL034", "ZR034") for x in (0, 300, 600, 900, 1200)]
    fits = {g: fit_focus_model(frows, g, min_observations=1000, reference_focus=600.0) for g in ("ZL034", "ZR034")}
    js = focus_model_json(fits, b)
    gR, gL = js["cameras"]["ZR034"], js["cameras"]["ZL034"]
    assert zcam_model_focal(gR, 900.0)[0] == pytest.approx(4738.0, abs=1e-6)
    assert zcam_model_pp_shift(gL, 900.0, 600.0) == (0.0, 0.0)
    assert zcam_model_pp_shift(gR, 900.0, 600.0)[0] == pytest.approx(300 * b["eqx_px"]["slope_per_count"])
    assert gR["label_f0_px"] == pytest.approx(4680.0)


def test_zcam_camera_temperature_is_head_fpa(monkeypatch):
    from mppp import image as I
    ns = types.SimpleNamespace(camera_model_label=types.SimpleNamespace(meta={"interpolation": "ZOOM"}),
                               fn=types.SimpleNamespace(stem="ZL0_0684_0727673848_348RAD_N0321174ZCAM07114_0340LMA01"),
                               label={"INSTRUMENT_STATE_PARMS": {
                                   "INSTRUMENT_TEMPERATURE_NAME": ["DEA", "HEAD_FPA", "HEAD_HTR_1", "HEAD_HTR_2"],
                                   "INSTRUMENT_TEMPERATURE": [30.76, -13.445, -14.77, -14.62]}})
    assert I.MPPPImage.camera_temperature_degC.fget(ns) == pytest.approx(-13.445)
    ns.label["INSTRUMENT_STATE_PARMS"]["INSTRUMENT_TEMPERATURE_NAME"] = ["DEA"]
    assert I.MPPPImage.camera_temperature_degC.fget(ns) is None


def test_zcam_label_temperature(tmp_path):
    from mppp.sfm.thermal import zcam_label_temperature
    lab = ("PDS_VERSION_ID = PDS3\r\nRECORD_TYPE = FIXED_LENGTH\r\nRECORD_BYTES = 100\r\nLABEL_RECORDS = 1\r\n"
           "GROUP = INSTRUMENT_STATE_PARMS\r\n"
           "  INSTRUMENT_TEMPERATURE = (28.15 <degC>, -15.04 <degC>, -15.41 <degC>, -16.23 <degC>)\r\n"
           "  INSTRUMENT_TEMPERATURE_NAME = (\"DEA\", \"HEAD_FPA\", \"HEAD_HTR_1\", \"HEAD_HTR_2\")\r\n"
           "END_GROUP = INSTRUMENT_STATE_PARMS\r\nEND\r\n")
    p = tmp_path / "ZL0_0461_0707874602_394RAD_N0260630ZCAM07114_0340LMA01.IMG"
    p.write_bytes(lab.encode("ascii"))
    try:
        v = zcam_label_temperature(p)
    except Exception as e:                                   # noqa: BLE001  (a reader that needs a full product)
        pytest.skip(f"minimal label not readable: {e}")
    assert v == pytest.approx(-15.04)


def test_focus_model_fingerprint(tmp_path):
    import hashlib
    import json
    from mppp.paths import data_dir
    from mppp.sfm.project import ZCAM_FOCUS_MODEL, zcam_focus_model_fingerprint
    shipped = data_dir() / "m20_cmods" / ZCAM_FOCUS_MODEL
    assert zcam_focus_model_fingerprint() == hashlib.sha256(shipped.read_bytes()).hexdigest()
    alt = tmp_path / "m.json"
    m = json.loads(shipped.read_text())
    m["cameras"]["ZL034"]["f0_px"] += 1
    alt.write_text(json.dumps(m))
    assert zcam_focus_model_fingerprint(alt) != zcam_focus_model_fingerprint()
