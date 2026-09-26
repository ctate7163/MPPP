"""v0p21: JPL camera models (mppp.cmod) and the camera-model comparison across scapes (mppp.sfm.calibration)."""
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data" / "m20"
NLF = DATA / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ01.IMG"
ZL0 = DATA / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"


def _label_model(path):
    from mppp.cmod import CameraModel
    from mppp.labels import label_get, read_pds
    L, _ = read_pds(path, load_image=False)
    w, h = int(label_get(L, "IMAGE.LINE_SAMPLES")), int(label_get(L, "IMAGE.LINES"))
    fs = float(label_get(L, "IMAGE.FIRST_LINE_SAMPLE", default=1))
    fl = float(label_get(L, "IMAGE.FIRST_LINE", default=1))
    return CameraModel.from_label(label_get(L, "GEOMETRIC_CAMERA_MODEL"), w, h), (fs - 1, fl - 1)


def _rational():
    from mppp.paths import data_dir
    return json.loads((data_dir() / "m20_cmods" / "M2020_NL_rational.json").read_text())


def test_navcam_label_is_cahvore_fisheye_and_matches_the_rational_model():
    from mppp.cmod import CameraModel, compare_to_colmap
    cm, (dx, dy) = _label_model(NLF)
    assert cm.kind == "CAHVORE" and cm.mtype == 2 and cm.linearity == 0.0
    full = cm.rescaled(1.0, dx, dy, 5120, 3840)                 # a full-resolution sub-frame
    nl = _rational()
    d = compare_to_colmap(full, nl["model"], nl["params"], 5120, 3840, step=160)
    assert d["rms_px"] < 4 and d["corner_rms_px"] < 7 and d["rotation_deg"] < 0.2
    # read as a perspective CAHVOR, the same numbers are hundreds of pixels off
    wrong = CameraModel(full.C, full.A, full.H, full.V, full.O, full.R)
    assert compare_to_colmap(wrong, nl["model"], nl["params"], 5120, 3840, step=160)["rms_px"] > 100


def test_decompose_matches_camera_py_and_rescale_is_exact():
    from mppp.camera import CAHVOR
    from mppp.labels import label_get, read_pds
    for path in (NLF, ZL0):
        cm, _ = _label_model(path)
        L, _ = read_pds(path, load_image=False)
        _, R_old = CAHVOR.from_label(label_get(L, "GEOMETRIC_CAMERA_MODEL")).decompose(cm.width, cm.height)
        R, _ = cm.decompose()
        assert np.abs(R - R_old).max() < 1e-5
    cm, _ = _label_model(ZL0)
    X = np.array([[1.0, 2.0, -1.0], [0.3, -2.0, -1.5]]) + cm.C + 5 * cm.A
    half = cm.rescaled(2.0, 10.0, 4.0)                          # a x2 "downsampled" product at (10, 4)
    np.testing.assert_allclose(half.project(X), (cm.project(X) + [10.5, 4.5]) / 2.0 - 0.5, atol=1e-9)


def test_cahvor_is_brown_k1_k2_and_cahvore_fits_navcam():
    from mppp.cmod import fit_to_colmap
    p = [4700.0, 4710.0, 830.0, 590.0, -0.47, 0.65, 0, 0, 0, 0, 0, 0]
    cm, rep = fit_to_colmap("FULL_OPENCV", p, 1648, 1200, "CAHVOR", step=64)
    assert rep["rms_px"] < 1e-6 and np.allclose(cm.R[1:], [-0.47, 0.65], atol=1e-8)
    nl = _rational()
    _, r2 = fit_to_colmap(nl["model"], nl["params"], 5120, 3840, "CAHVORE", 2, step=128)
    _, r3 = fit_to_colmap(nl["model"], nl["params"], 5120, 3840, "CAHVORE", 3, True, step=128)
    _, r1 = fit_to_colmap(nl["model"], nl["params"], 5120, 3840, "CAHVOR", step=128)
    assert r3["rms_px"] < r2["rms_px"] < 0.6 < 5 < r1["rms_px"]


def test_manifest_keeps_the_full_label_model(cfg_nomask):
    from mppp.cmod import PixelCamera, compare_cameras
    from mppp.image import MPPPImage
    from mppp.sfm.calibration import label_model_from_meta
    for path, tol in ((ZL0, 0.6), (NLF, 5.0)):
        m = json.loads(json.dumps(MPPPImage(path, cfg_nomask, None).meta, default=float))
        assert m["camera_model_label"]["MODEL_TYPE"] in ("CAHVOR", "CAHVORE")
        exact = label_model_from_meta(m)
        m.pop("camera_model_label")
        approx = label_model_from_meta(m)                       # older manifests: O along A
        d = compare_cameras(PixelCamera.cahv(exact, as_is=True), PixelCamera.cahv(approx, as_is=True), 96, False)
        assert d["max_px"] < tol
        _, li = exact.decompose()
        W, H = (5120, 3840) if path == NLF else (1648, 1200)
        assert (exact.width, exact.height) == (W, H) and 0.4 * W < li["hc"] < 0.6 * W


def test_compare_cameras_removes_rotation():
    from scipy.spatial.transform import Rotation
    from mppp.cmod import PixelCamera, compare_cameras
    p = np.array([1000.0, 1000, 640, 480, -0.2, 0.05, 0, 0, 0, 0, 0, 0])
    a = PixelCamera.colmap("FULL_OPENCV", p, 1280, 960)
    assert compare_cameras(a, a)["max_px"] < 1e-6
    q = p.copy(); q[2] += 5.0                                     # a principal-point shift is mostly a rotation
    d = compare_cameras(a, PixelCamera.colmap("FULL_OPENCV", q, 1280, 960))
    assert 0.2 < d["rotation_deg"] < 0.4 and d["rms_px"] < 0.5 < 5.0 == pytest.approx(
        compare_cameras(a, PixelCamera.colmap("FULL_OPENCV", q, 1280, 960), fit_rotation=False)["rms_px"], abs=1e-6)


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
