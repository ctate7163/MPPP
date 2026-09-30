"""Camera-model comparison and consensus across blocks (mppp.sfm.calibration). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
import numpy as np
import pytest
from pathlib import Path
from mppp.sfm import navcal as NC  # noqa: E402


def _rational():
    from mppp.paths import data_dir
    return json.loads((data_dir() / "cmods" / "M2020_NL_rational.json").read_text())


def _solution(tmp_path, yaw_deg=0.0, fx=2951.0):
    """A minimal notebook-03 export: one Navcam stereo pair."""
    from scipy.spatial.transform import Rotation
    root = tmp_path / "x_colmap" / "colmap"
    (root / "error_input" / "native").mkdir(parents=True)
    p0 = [2951.0, 2951.0, 2591.0, 1943.0, -0.27, 0.094, 0, 0, -0.0177, 0, 0, 0]
    p1 = list(p0); p1[0] = fx
    RL = Rotation.from_euler("x", -100, degrees=True).as_matrix()
    Rrel = Rotation.from_rotvec([0, np.radians(yaw_deg), 0]).as_matrix()
    CL, CR = np.array([0.0, 0, 2]), np.array([0.4244, 0, 2])
    RR = Rrel @ RL
    imgs = [{"name": "NLF_a.png", "stem": "NLF_a", "instrument": "NL", "eye": "L", "sclk_key": "1", "station": "S1D1",
             "sol": 1, "downsample_scale": 1.0, "prior_C": CL.tolist(), "prior_R_w2c": RL.tolist(), "image_id": 1},
            {"name": "NRF_a.png", "stem": "NRF_a", "instrument": "NR", "eye": "R", "sclk_key": "1", "station": "S1D1",
             "sol": 1, "downsample_scale": 1.0, "prior_C": CR.tolist(), "prior_R_w2c": RL.tolist(), "image_id": 2}]
    cams = {k: {"model": "FULL_OPENCV", "width": 5120, "height": 3840, "params": p0} for k in ("NL", "NR")}
    (root / "project.json").write_text(json.dumps({"images": imgs, "cameras": cams, "rig": {}, "offset": [0, 0, 0],
                                                   "settings": {}}))
    (root / "error_input" / "summary.json").write_text(json.dumps({
        "cameras_initial": cams, "cameras_refined": {k: {"model": "FULL_OPENCV", "params": p1} for k in cams},
        "rig_initial": {"N": {"R_sensor_from_ref": np.eye(3).tolist(), "t_sensor_from_ref": [-0.4244, 0, 0]}},
        "rig_refined": {"1": {"R": Rrel.tolist(), "t": [-0.4244, 0, 0]}}}))
    (root / "error_input" / "poses.csv").write_text("name,observations\nNLF_a.png,5000\nNRF_a.png,4000\n")
    lines = []
    for iid, R, C in ((1, RL, CL), (2, RR, CR)):
        x, y, z, w = Rotation.from_matrix(R).as_quat()
        t = -R @ C
        lines.append(f"{iid} {iid} {w} {x} {y} {z} {t[0]} {t[1]} {t[2]} 1 CAMERA {iid} {iid}")
    (root / "error_input" / "native" / "frames.txt").write_text("# frames\n" + "\n".join(lines) + "\n")
    return root


def test_stereo_effect_sign_and_size():
    from scipy.spatial.transform import Rotation
    from mppp.cmod import PixelCamera
    from mppp.sfm.calibration import stereo_effect
    f, B = 1000.0, 0.4
    cam = lambda ff: PixelCamera.colmap("PINHOLE", [ff, ff, 640, 480], 1280, 960)       # noqa: E731
    rig = (np.eye(3), np.array([-B, 0, 0]))
    same = stereo_effect(cam(f), cam(f), rig, cam(f), cam(f), rig, ranges_m=(10,))[0]
    assert abs(same["disparity_bias_px"]) < 1e-9 and abs(same["range_error_centre_pct"]) < 1e-9
    # both focal lengths 1 % short: the pair predicts 1 % less disparity -> ranges 1 % short
    short = stereo_effect(cam(0.99 * f), cam(0.99 * f), rig, cam(f), cam(f), rig, ranges_m=(10,))[0]
    assert short["range_error_centre_pct"] == pytest.approx(-1.0, abs=0.1)
    # a toe-in of the right camera by delta shifts disparity by ~ f delta everywhere
    delta = np.radians(0.05)
    toe = (Rotation.from_rotvec([0, delta, 0]).as_matrix(), rig[1])
    e = stereo_effect(cam(f), cam(f), toe, cam(f), cam(f), rig, ranges_m=(10,))[0]
    assert abs(e["centre_disparity_px"]) == pytest.approx(f * delta, rel=0.05)


def test_load_solution_stereo_pairs_and_consensus(tmp_path):
    from mppp.sfm import calibration as CAL
    root = _solution(tmp_path, yaw_deg=0.02, fx=2955.0)
    s = CAL.load_solution(root.parent, "x")
    assert set(s.cameras) == {"NL", "NR"} and s.cameras["NL"].refined and s.cameras["NL"].n_obs == 5000
    assert s.navcam_distortion == "polynomial"
    pr = CAL.stereo_pairs(s, "N")
    assert len(pr) == 1 and pr[0]["dyaw_mdeg"] == pytest.approx(20.0, abs=0.01)
    assert pr[0]["baseline_m"] == pytest.approx(0.4244, abs=1e-9)
    c = CAL.consensus_camera({"x": s}, "NL", min_observations=100)
    assert c.params[0] == pytest.approx(2955.0)
    rows = CAL.camera_table({"x": s}, "N")
    assert rows[0]["dfx"] == pytest.approx(4.0)


def test_focus_fit_recovers_slope_and_offsets():
    from mppp.sfm.calibration import fit_focus_model
    rows = []
    for scape, off in (("a", -3.0), ("b", 3.0)):
        for fc in np.linspace(-1500, 1200, 12):
            rows.append({"scape": scape, "group": "ZL034", "focus": float(fc), "refined": True, "observations": 5000,
                         "f_refined_px": 4700 + 0.046 * fc + off, "f_label_median_px": 4660 + 0.05 * fc})
    f = fit_focus_model(rows, "ZL034", min_observations=100)
    assert f["slope_px_per_count"] == pytest.approx(0.046, abs=1e-6)
    assert f["scape_offsets_px"]["b"] - f["scape_offsets_px"]["a"] == pytest.approx(6.0, abs=1e-6)
    assert f["label"]["slope_px_per_count"] == pytest.approx(0.05, abs=1e-6)


def test_undistort_pinhole_is_identity_and_mask_screen():
    from mppp.sfm.calibration import Camera, screen_mask, undistort
    rng = np.random.default_rng(0)
    img = rng.integers(1, 255, (60, 80, 3)).astype(np.uint8)
    p = np.array([100.0, 100, 40, 30, 0, 0, 0, 0, 0, 0, 0, 0])
    cam = Camera("t", "NL", "FULL_OPENCV", 80, 60, p, p)
    u = undistort(img, cam, fit="same")
    assert np.abs(u["image"][2:-2, 2:-2].astype(int) - img[2:-2, 2:-2]).max() <= 1
    mask = np.full((60, 80), 255, np.uint8); mask[:10] = 0
    img[:5] = 0                                              # no-data rows stay black
    out = screen_mask(img, mask, 0.3)
    assert np.all(out[:5] == 0) and np.allclose(out[5:10], np.round(0.7 * img[5:10] + 0.3 * 255), atol=1)
    assert np.array_equal(out[10:], img[10:])


def test_review_fixes_linearity_domain_type3_default_and_empty_consensus(tmp_path):
    from mppp.cmod import CameraModel, fit_to_colmap
    from mppp.sfm import calibration as CAL
    A = np.array([0.0, 0, 1])
    cm = CameraModel(np.zeros(3), A, [1000.0, 0, 500], [0, 1000.0, 400], A, [0, 0.01, 0], [0, 0, 0], 3, -1.5)
    th = np.radians(80)
    assert np.all(np.isnan(cm.project(np.array([[np.sin(th), 0, np.cos(th)]]))))     # |L| theta > pi/2
    nl = _rational()
    m3, r3 = fit_to_colmap(nl["model"], nl["params"], 5120, 3840, "CAHVORE", 3, step=160)
    assert m3.mtype == 3 and 0.2 < m3.linearity < 0.9
    s = CAL.load_solution(_solution(tmp_path), "x")
    assert CAL.consensus_camera({"x": s}, "NL", "rational", min_observations=1) is None


def test_radial_limit_full_frame_pick_and_focus_cutoff(tmp_path):
    from mppp.sfm import calibration as CAL
    poly = CAL.reference_camera("NL", "polynomial")
    lim = CAL.radial_limit(poly)
    assert 50 < np.degrees(np.arctan(lim)) < 60                     # the three-term polynomial folds back
    assert CAL.radial_limit(CAL.reference_camera("NL", "rational")) == float("inf")
    s = CAL.load_solution(_solution(tmp_path), "x")
    s.manifest = {"NLF_a": {"padding": {"left": 0, "right": 0, "top": 0, "bottom": 0}}}
    assert CAL.is_full_frame(s, "NLF_a.png") and CAL.is_full_frame(s, "NRF_a.png") is None
    assert CAL.pick_example(s, "NL") == "NLF_a.png"
    rows = [{"scape": "a", "group": "ZL034", "focus": float(fc), "refined": True, "observations": 5000,
             "f_refined_px": 4700 + 0.046 * fc + (300 if fc < -2000 else 0)} for fc in np.linspace(-3500, 1200, 20)]
    assert CAL.fit_focus_model(rows, "ZL034", 100)["slope_px_per_count"] == pytest.approx(0.046, abs=1e-9)
    assert CAL.fit_focus_model(rows, "ZL034", 100, min_focus=None)["rms_px"] > 10


def test_write_navcam_consensus_round_trip(tmp_path):
    from mppp.sfm import calibration as CAL
    from mppp.sfm.project import NAVCAM_RATIONAL_PATTERN, NAVCAM_RIG_FILE, camera_from_colmap_json
    cams = {g: CAL.reference_camera(g, "rational") for g in ("NL", "NR")}
    R = np.eye(3); t = np.array([-0.4244, 0.0, 0.0])
    written = CAL.write_navcam_consensus(cams, (R, t), tmp_path / "consensus", repeatability={"eps_ref_px": 0.25})
    assert set(written) == {"NL", "NR", "rig"}
    for g in ("NL", "NR"):
        c = camera_from_colmap_json(tmp_path / "consensus" / NAVCAM_RATIONAL_PATTERN.format(instrument=g))
        assert c["model"] == "FULL_OPENCV" and np.allclose(c["params"], cams[g].params) and c["free_params"] == ["k4"]
        d = json.loads(written[g].read_text())
        assert d["verification"]["repeatability"]["eps_ref_px"] == 0.25 and d["verification"]["per_scape"] == []
    rig = json.loads((tmp_path / "consensus" / NAVCAM_RIG_FILE).read_text())
    assert rig["ref"] == "NL" and np.allclose(rig["R_sensor_from_ref"], R) and abs(rig["baseline_m"] - 0.4244) < 1e-9


def test_focal_temperature_fit_and_label_temperature():
    from mppp.sfm.calibration import focal_temperature_fit, _label_temperature
    from mppp.cmod import CameraModel
    rows = [{"camera": c, "temp_median_degC": t, "fx": 2950.0 + 0.02 * t + (0.5 if c == "NL" else 0.0)}
            for c in ("NL", "NR") for t in (-40.0, -20.0, 0.0, 10.0)]
    fit = focal_temperature_fit(rows)
    assert abs(fit["NL"]["px_per_degC"] - 0.02) < 1e-9 and abs(fit["NR"]["fx_at_0C"] - 2950.0) < 1e-6
    assert abs(fit["NR"]["ppm_per_degC"] - 0.02 / 2950.0 * 1e6) < 1e-6
    gcm = {"MODEL_TYPE": "CAHVORE", "MODEL_COMPONENT_1": [0, 0, 0], "MODEL_COMPONENT_2": [0, 0, 1],
           "MODEL_COMPONENT_3": [2950, 0, 2560], "MODEL_COMPONENT_4": [0, 2950, 1920],
           "MODEL_COMPONENT_5": [0, 0, 1], "MODEL_COMPONENT_6": [0, 0.05, -0.017], "MODEL_COMPONENT_7": [0, 0, 0],
           "MODEL_COMPONENT_8": 2.0, "MODEL_COMPONENT_9": 0.0,
           "INTERPOLATION_METHOD": "TEMPERATURE", "INTERPOLATION_VALUE": -18.0172}
    assert _label_temperature(CameraModel.from_label(gcm, 5120, 3840)) == -18.0172


def _fake_solutions(ppm=60.0, T0=-20.0):
    from mppp.sfm.calibration import Camera, Solution, reference_camera
    ship = reference_camera("NL", "rational")
    sols = {}
    for name, temps in (("A", (-40.0, -25.0)), ("B", (-15.0,)), ("C", (-30.0, -5.0))):
        cams, pcams = {}, {}
        for T in temps:
            key = f"NL_T{int(T):+04d}"
            p = np.array(ship.params, float)
            p[:2] *= 1 + 1e-6 * ppm * (T - T0)
            cams[key] = Camera(key, "NL", ship.model, 5120, 3840, p, np.array(ship.params, float) * 0.999, None, 20, 50000)
            pcams[key] = {"group": "NL", "temperature_median_degC": T, "thermal_bin": True}
        sols[name] = Solution(name, Path("."), {"cameras": pcams, "settings": {}}, {}, cams, {})
    return sols, ship


def test_consensus_with_thermal_model_is_the_camera_at_T0():
    from mppp.sfm.calibration import consensus_camera, reference_differences, thermal_model, thermal_scale
    sols, ship = _fake_solutions()
    th = {"NL": {"ppm_per_degC": 60.0, "T0_degC": -20.0}}
    c = consensus_camera(sols, "NL", "rational", 1000, thermal=th)
    assert np.allclose(c.params, ship.params, rtol=0, atol=1e-6) and c.thermal == th["NL"] and not c.excluded
    plain = consensus_camera(sols, "NL", "rational", 1000)
    assert abs(plain.params[0] - ship.params[0]) > 1e-3                      # without the model: the mean temperature
    rows = reference_differences(sols, "NL", c, min_observations=1000, thermal=th)
    assert len(rows) == 5 and max(r["rms_px"] for r in rows) < 1e-3
    assert all(abs(r["thermal_scale"] - thermal_scale(th, "NL", r["T_degC"], to_T0=False)) < 1e-12 for r in rows)
    # model from a fit: T0 is the observation-weighted mean temperature of the cameras
    m = thermal_model(sols, {"NL": {"ppm_per_degC": 55.0, "ppm_sd": 5.0}}, None, "auto")
    assert m["NL"]["source"] == "within" and abs(m["NL"]["T0_degC"] - np.mean([-40, -25, -15, -30, -5])) < 1e-9
    assert thermal_model(sols, None, None, "fixed", 70.0)["NL"]["ppm_per_degC"] == 70.0
    assert thermal_model(sols, None, None, "auto") is None


def test_merge_temperature_bins_and_bin_rows(tmp_path):
    from mppp.sfm.calibration import merge_temperature_bins, thermal_bin_rows
    rows = [{"scape": "A", "camera": "NL_T-040-030", "eye": "NL", "images": 10, "n_temp": 10, "observations": 1000,
             "temperature_bin": True, "temp_median_degC": -35.0, "temp_min_degC": -38.0, "temp_max_degC": -31.0,
             "fx": 2954.0, "fy": 2954.0},
            {"scape": "A", "camera": "NL_T-030-020", "eye": "NL", "images": 30, "n_temp": 30, "observations": 3000,
             "temperature_bin": True, "temp_median_degC": -25.0, "temp_min_degC": -29.0, "temp_max_degC": -21.0,
             "fx": 2956.0, "fy": 2956.0},
            {"scape": "B", "camera": "NL", "eye": "NL", "images": 5, "n_temp": 5, "observations": 500,
             "temperature_bin": False, "temp_median_degC": -10.0, "temp_min_degC": -12.0, "temp_max_degC": -8.0,
             "fx": 2957.0, "fy": 2957.0}]
    m = {r["scape"]: r for r in merge_temperature_bins(rows)}
    assert len(m) == 2 and abs(m["A"]["fx"] - 2955.5) < 1e-9 and abs(m["A"]["temp_median_degC"] + 27.5) < 1e-9
    assert m["A"]["temp_min_degC"] == -38.0 and m["A"]["temp_max_degC"] == -21.0 and m["A"]["images"] == 40
    from mppp.sfm.calibration import Solution
    th = {"held": False, "rows": [{"eye": "NL", "camera": "NL_T-030-020", "T_median_degC": -25.0, "fx": 1.0, "fy": 1.0,
                                   "observations": 10}]}
    sols = {"Rockytop": Solution("Rockytop", tmp_path, {"settings": {"thermal": th}}, {}, {}, {}),
            "Held": Solution("Held", tmp_path, {"settings": {"thermal": dict(th, held=True)}}, {}, {}, {})}
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"scapes": {"rockytop": {"rows": [{"eye": "NL"}]},
                                          "olifants": {"rows": [{"eye": "NR", "camera": "NR_T-020-010"}]}}}))
    rr = thermal_bin_rows(sols, exp)
    assert [(r["scape"], r["source"]) for r in rr] == [("Rockytop", "thermal stage"), ("olifants", "experiment")]


pycolmap = pytest.importorskip("pycolmap")


def _cam(model="FULL_OPENCV", cid=1):
    p = [2956.0, 2955.5, 2591.0, 1944.0, 0.306, -0.026, 1.5e-4, 1.8e-4, 0.002, 0.594, 0.0, 0.0]
    return pycolmap.Camera(camera_id=cid, model=model, width=5120, height=3840, params=np.array(p))


def test_compare_and_fisheye_conversion():
    from mppp.sfm import navcal as NC
    c = _cam()
    d = NC.compare(c, c)
    assert d["rms_px"] < 1e-6
    q = np.array(c.params)
    q[0] *= 1 + 30e-6 * 10
    q[1] *= 1 + 30e-6 * 10
    d2 = NC.compare(c, NC.pycolmap_camera("FULL_OPENCV", q))
    assert 0.1 < d2["rms_px"] < 2.0
    fe, rms = NC.to_fisheye_tangential(c)
    assert fe.model.name == "THIN_PRISM_FISHEYE" and rms < 1.0
    assert NC.compare(c, fe)["rms_px"] < 1.0


def test_calibration_camera_names_the_fisheye_lens():
    from mppp.sfm.calibration import Camera
    c = Camera("NL", "NL", "THIN_PRISM_FISHEYE", 5120, 3840, np.zeros(12), np.zeros(12))
    assert c.distortion == "fisheye_tangential"
    assert np.all(np.isfinite(Camera("NL", "NL", "THIN_PRISM_FISHEYE", 5120, 3840,
                                     np.array([2956., 2956., 2591., 1944., .045, -.01, 0, 0, .003, -.006, 0, 0]),
                                     np.zeros(12)).pixel_camera().rays(np.array([[100.0, 100.0]]))))


def test_fit_camera_model_round_trip():
    cam = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840,
                          params=[2956.3, 2955.9, 2591.1, 1943.5, 0.3054, -0.02596, 1.6e-4, 1.9e-4, 0.002, 0.5934, 0, 0])
    f, r1 = NC.fit_camera_model(cam, "THIN_PRISM_FISHEYE")
    back, r2 = NC.fit_camera_model(f, "FULL_OPENCV")
    assert f.model.name == "THIN_PRISM_FISHEYE" and back.model.name == "FULL_OPENCV"
    assert r1 < 0.5 and r2 < 0.5
    assert np.allclose(back.params[2:4], cam.params[2:4], atol=0.2)


def test_scale_offsets_recover_an_injected_shift(tmp_path):
    pytest.importorskip("pyceres")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import _synthetic_block
    from mppp.sfm.reconstruction import bundle_adjust
    proj, rec, noise = _synthetic_block(tmp_path)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=20)
    # the frames of one station at half resolution, with their keypoints shifted by (+1.0, -0.5) full-res px
    st = sorted({r["station"] for r in proj.images})[0]
    half = {int(r["image_id"]) for r in proj.images if r["station"] == st}
    for r in proj.images:
        r["downsample_scale"] = 0.5 if int(r["image_id"]) in half else 1.0
    for iid in half:
        for p in rec.images[iid].points2D:          # in place (the list holds references)
            p.xy = np.asarray(p.xy) + np.array([1.0, -0.5])
    iid = next(iter(half))
    assert rec.images[iid].points2D[0].xy[0] != 0.0
    sc = NC.Scape("synthetic", tmp_path, proj, rec, {})
    rec2, proj2, rows = NC.split_by_scale(rec, proj, min_images=2)
    assert {r["scale"] for r in rows} == {0.5, 1.0} and rec2.num_reg_images() == rec.num_reg_images()
    out = NC.scale_offsets(sc, min_images=2, max_iterations=30, verbose=False)
    got = [r for r in out["rows"] if "dcx_px" in r]
    assert got
    for r in got:
        sign = 1.0 if r["scale"] == 0.5 else -1.0          # offset of the half-resolution camera from the full one
        assert r["dcx_px"] * sign == pytest.approx(1.0, abs=0.15)
        assert r["dcy_px"] * sign == pytest.approx(-0.5, abs=0.15)


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
