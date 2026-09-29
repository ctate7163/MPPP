"""v0p35: Navcam calibration study helpers (mppp.sfm.navcal), rig temperature model, keypoint thermal scaling."""
import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")


def _cam(model="FULL_OPENCV", cid=1):
    p = [2956.0, 2955.5, 2591.0, 1944.0, 0.306, -0.026, 1.5e-4, 1.8e-4, 0.002, 0.594, 0.0, 0.0]
    return pycolmap.Camera(camera_id=cid, model=model, width=5120, height=3840, params=np.array(p))


def test_rig_rotation_at_keeps_centre_and_is_identity_at_zero():
    from mppp.sfm.thermal import rig_rotation_at
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec([0.0014, 0.0011, 0.0002]).as_matrix()
    t = np.array([-0.42436, -0.00016, 0.00007])
    R0, t0 = rig_rotation_at(R, t, 0.0, -1.0, 0.3)
    assert np.allclose(R0, R) and np.allclose(t0, t)
    R2, t2 = rig_rotation_at(R, t, 10.0, -1.0, 0.3)
    assert np.allclose(-R2.T @ t2, -R.T @ t)                              # the right camera's centre does not move
    rv = Rotation.from_matrix(R2 @ R.T).as_rotvec()
    assert np.allclose(np.degrees(rv) * 1e3, [3.0, -10.0, 0.0], atol=1e-6)  # pitch x, yaw y (mdeg)


def test_rig_keypoint_map_moves_only_right_images():
    from mppp.sfm.navcal import rig_keypoint_map

    class P:
        images = [{"image_id": 1, "name": "L.png"}, {"image_id": 2, "name": "R.png"}]
    temps = {"L.png": 0.0, "R.png": 0.0}
    left, right = _cam(cid=1), _cam(cid=2)
    kps = np.array([[2591.0, 1944.0], [800.0, 600.0]])
    f = rig_keypoint_map(P, temps, -1.0, 0.0, T0=-10.0)                  # dT = +10 degC -> yaw -10 mdeg
    assert np.allclose(f(1, kps, left), kps)
    moved = f(2, kps, right)
    # a yaw of -10 mdeg moves the principal point by about f * 1.745e-4 = 0.52 px in x, nothing in y
    d = moved[0] - kps[0]
    assert abs(abs(d[0]) - 2956.0 * np.radians(0.01)) < 0.02 and abs(d[1]) < 0.01
    g = rig_keypoint_map(P, temps, 1.0, 0.0, T0=-10.0)
    assert np.allclose(g(2, moved, right), kps, atol=1e-6)               # the opposite slope undoes it


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


def test_wls_tau_and_slope():
    from mppp.sfm.navcal import _wls
    rng = np.random.default_rng(1)
    x = np.linspace(-40, -10, 12)
    y = 2.0 - 0.9 * x + rng.normal(0, 0.2, x.size)
    f = _wls(y, np.c_[np.ones_like(x), x], np.full(x.size, 0.04))
    assert abs(f["coef"][1] + 0.9) < 0.05 and f["tau"] < 0.2
    y2 = y + rng.normal(0, 3.0, x.size)                                  # extra between-block scatter
    f2 = _wls(y2, np.c_[np.ones_like(x), x], np.full(x.size, 0.04))
    assert f2["tau"] > 1.0 and f2["se"][1] > f["se"][1]


def test_stereo_offset_follows_yaw():
    from mppp.sfm.navcal import stereo_offset
    from scipy.spatial.transform import Rotation
    L, R = _cam(cid=1), _cam(cid=2)
    T0 = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([-0.424, 0, 0]))
    T1 = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec([0, np.radians(0.01), 0]).as_matrix()),
                          np.array([-0.424, 0, 0]))
    d0, d1 = stereo_offset(L, R, T0), stereo_offset(L, R, T1)
    assert abs(d0["disparity_inf_px"]) < 1e-6
    assert abs(abs(d1["disparity_inf_px"] - d0["disparity_inf_px"]) - 2956 * np.radians(0.01)) < 0.02


def test_write_joint_cameras_roundtrip(tmp_path):
    from mppp.sfm.navcal import write_joint_cameras
    from mppp.sfm.project import camera_from_colmap_json
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    cam = {"model": "FULL_OPENCV", "width": 5120, "height": 3840, "params": dict(zip(names, _cam().params)),
           "free_params": list(names[:10]), "sd": {n: 0.1 for n in names[:10]}, "covariance": np.eye(10).tolist()}
    joint = {"lens": "rational", "NL": cam, "NR": cam, "T0_degC": -18.0, "ppm_per_degC": 40.0, "sd_ppm_per_degC": 2.0,
             "final": {"observations": 1000}, "blocks": {"A": 10, "B": 12}, "rig_R": np.eye(3).tolist(),
             "rig_t": [-0.424, 0, 0], "rig": {"sd_yaw_mdeg": 0.5},
             "rig_thermal": {"yaw_mdeg_per_degC": -1.0, "pitch_mdeg_per_degC": 0.2, "yaw_profile": {"sd": 0.05}}}
    w = write_joint_cameras(joint, tmp_path, loo={"A": {"camera_difference": {"NL": {"rms_px": 0.2}}}})
    d = json.loads(w["NL"].read_text())
    assert d["thermal"]["ppm_per_degC"] == 40.0 and d["verification"]["per_scape"][0]["scape"] == "A"
    c = camera_from_colmap_json(w["NL"], ("b1", "b2"))
    assert np.allclose(c["params"][:4], _cam().params[:4])
    r = json.loads(w["rig"].read_text())
    assert r["thermal"]["yaw_mdeg_per_degC"] == -1.0 and r["thermal"]["T0_degC"] == -18.0


def test_bundle_adjust_keypoint_scale_equals_scaled_camera():
    """Keypoints divided by s about the principal point fit a camera whose fx, fy are s times smaller."""
    from mppp.sfm.navcal import keypoint_scales

    class P:
        images = [{"image_id": 1, "name": "a"}, {"image_id": 2, "name": "b"}]
    ks = keypoint_scales(P, {"a": -10.0, "b": -30.0}, 50.0, -20.0)
    assert ks[1] == pytest.approx(1 + 50e-6 * 10) and ks[2] == pytest.approx(1 - 50e-6 * 10)
    c = _cam()
    X = np.array([[0.3, -0.2, 1.0], [-0.5, 0.4, 1.0]])
    s = ks[1]
    q = np.array(c.params)
    q[:2] *= s
    hot = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=q)
    x_hot = np.array(hot.img_from_cam(X))
    c0 = np.array(c.params[2:4])
    assert np.allclose(c0 + (x_hot - c0) / s, np.array(c.img_from_cam(X)), atol=1e-9)


def test_notebooks_v0p35():
    nbformat = pytest.importorskip("nbformat")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "notebooks"
    nb3 = nbformat.read(str(root / "03_colmap_alignment.ipynb"), as_version=4)
    full3 = "\n".join(c.source for c in nb3.cells)
    assert "0.35." in nb3.cells[0].source and 'NAVCAM_RIG_REFINE = "auto"' in full3 and "navcam_joint" in full3
    assert '"auto": "auto"}.get(NAVCAM_RIG_REFINE' in full3
    nb4 = nbformat.read(str(root / "04_camera_models.ipynb"), as_version=4)
    full4 = "\n".join(c.source for c in nb4.cells)
    for k in ("navcam_calibration_study.py", "NCR.main(", "write_joint_cameras", "2d  Joint calibration", "2e  Frozen"):
        assert k in full4, k
    for name in ("01_process_images", "05_error_analysis"):
        nb = nbformat.read(str(root / f"{name}.ipynb"), as_version=4)
        assert "0.35." in nb.cells[0].source


def test_reconstruct_rig_auto_and_thermal_stage_rig_slopes():
    import inspect
    from mppp.sfm import reconstruction as R, thermal as TH
    src = inspect.getsource(R.reconstruct)
    assert 'refine_rig == "auto"' in src and "rig_slopes_for_project(project)" in src
    assert "rig_slopes" in inspect.signature(TH.thermal_stage).parameters
    assert "rig_slopes" in inspect.signature(TH.split_by_temperature).parameters
    assert TH.rig_slopes_for_project(type("P", (), {"settings": {}})()) is None
    p = type("P", (), {"settings": {"navcam_cameras": {"rig": {"thermal": {"yaw_mdeg_per_degC": -1.0, "T0_degC": -19.1}}}}})()
    assert TH.rig_slopes_for_project(p)["yaw_mdeg_per_degC"] == -1.0


def test_rig_drift_regression():
    from mppp.sfm.navcal import rig_drift
    rng = np.random.default_rng(3)
    studies = []
    for i in range(12):
        T, sol = -25 + 2 * i + rng.normal(0, 3), 100 + 150 * i
        ang = {"pitch": 0.005 * (sol - 1000) + rng.normal(0, 0.3), "yaw": -1.1 * (T + 18) + rng.normal(0, 0.5),
               "roll": -0.006 * (sol - 1000) + rng.normal(0, 0.3)}
        studies.append({"network": {"T_median_degC": T, "sol_median": sol},
                        "rotation_pp": {"rigs": [{**{f"{k}_mdeg": v for k, v in ang.items()},
                                                  **{f"sd_{k}_mdeg": 0.2 for k in ang}}]}})
    d = rig_drift(studies)
    assert abs(d["pitch_mdeg_per_sol"] - 0.005) < 0.001 and abs(d["roll_mdeg_per_sol"] + 0.006) < 0.001
    assert abs(d["yaw_T_mdeg_per_degC"] + 1.1) < 0.2 and d["blocks"] == 12


def test_joint_camera_files_carry_rig_drift(tmp_path):
    from mppp.sfm.navcal import write_joint_cameras
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    cam = {"model": "FULL_OPENCV", "width": 5120, "height": 3840, "params": dict(zip(names, _cam().params))}
    joint = {"NL": cam, "NR": cam, "T0_degC": -19.1, "ppm_per_degC": 38.6, "final": {"observations": 1}, "blocks": {},
             "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0], "rig_drift": {"sol0": 1084.0, "pitch_mdeg_per_sol": 0.005}}
    w = write_joint_cameras(joint, tmp_path)
    r = json.loads(w["rig"].read_text())
    assert r["drift"]["pitch_mdeg_per_sol"] == 0.005 and "thermal" not in r


def test_start_rig_rotation_temperature_and_drift():
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import start_rig_rotation
    R0 = Rotation.from_rotvec([0.0014, 0.0011, 0.0002]).as_matrix()
    sh = {"R_sensor_from_ref": R0.tolist(), "thermal": {"yaw_mdeg_per_degC": -1.0, "pitch_mdeg_per_degC": 0.0, "T0_degC": -19.0},
          "drift": {"sol0": 1000.0, "pitch_mdeg_per_sol": 0.005, "yaw_mdeg_per_sol": 0.0, "roll_mdeg_per_sol": -0.006}}
    R, applied = start_rig_rotation(sh, -9.0, 1400.0)
    rv = np.degrees(Rotation.from_matrix(R @ R0.T).as_rotvec()) * 1e3
    assert np.allclose(rv, [2.0, -10.0, -2.4], atol=0.01)                # pitch +0.005x400, yaw -1x10, roll -0.006x400
    assert applied["thermal"]["T_median_degC"] == -9.0 and applied["drift"]["sol_median"] == 1400.0
    R1, a1 = start_rig_rotation({"R_sensor_from_ref": R0.tolist()}, -9.0, 1400.0)
    assert np.allclose(R1, R0) and a1 == {}


def test_project_camera_fallback_for_fisheye_tangential():
    from mppp.colmap import project_camera, unproject_camera
    p = np.array([2956.3, 2956.4, 2591.8, 1944.2, 0.0451, -0.0100, 2.5e-4, 3.6e-4, 0.0030, -0.0062, 0.0, 0.0])
    X = np.array([[0.3, -0.2, 1.0], [-0.6, 0.45, 1.0], [0.0, 0.0, 1.0]])
    cam = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=p)
    uv = project_camera("THIN_PRISM_FISHEYE", p, X)
    assert np.allclose(uv, np.array(cam.img_from_cam(X)), atol=1e-9)
    xy = unproject_camera("THIN_PRISM_FISHEYE", p, uv)
    assert np.allclose(xy, X[:, :2], atol=1e-7)
    assert np.all(np.isnan(project_camera("THIN_PRISM_FISHEYE", p, np.array([[0.1, 0.1, -1.0]]))))


def test_calibration_camera_names_the_fisheye_lens():
    from mppp.sfm.calibration import Camera
    c = Camera("NL", "NL", "THIN_PRISM_FISHEYE", 5120, 3840, np.zeros(12), np.zeros(12))
    assert c.distortion == "fisheye_tangential"
    assert np.all(np.isfinite(Camera("NL", "NL", "THIN_PRISM_FISHEYE", 5120, 3840,
                                     np.array([2956., 2956., 2591., 1944., .045, -.01, 0, 0, .003, -.006, 0, 0]),
                                     np.zeros(12)).pixel_camera().rays(np.array([[100.0, 100.0]]))))


def test_project_create_with_fisheye_tangential_start_cameras(tmp_path):
    from conftest import NLF, needs_data
    if not NLF.is_file():
        pytest.skip("example IMGs not present")
    import mppp
    from mppp.sfm.navcal import write_joint_cameras
    from mppp.sfm.project import SfmProject
    out = tmp_path / "proc"
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": False}})
    man = mppp.process_images([NLF], out, cfg, mppp.load_waypoints(), progress=False)
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
    q = [2956.3, 2956.4, 2591.8, 1944.2, 0.0451, -0.0100, 2.5e-4, 3.6e-4, 0.0030, -0.0062, 0.0, 0.0]
    cam = {"model": "THIN_PRISM_FISHEYE", "width": 5120, "height": 3840, "params": dict(zip(names, q))}
    joint = {"NL": cam, "NR": cam, "T0_degC": -19.1, "ppm_per_degC": 38.6, "final": {"observations": 1}, "blocks": {},
             "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0]}
    write_joint_cameras(joint, tmp_path / "joint")
    with pytest.raises(ValueError, match="needs navcam_cameras"):
        SfmProject.create(man["images"], out, tmp_path / "p0", link=False, navcam_distortion="fisheye_tangential")
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, navcam_distortion="fisheye_tangential",
                             navcam_cameras=tmp_path / "joint")
    assert proj.cameras["NL"]["model"] == "THIN_PRISM_FISHEYE"
    assert abs(proj.cameras["NL"]["params"][4] - 0.0451) < 1e-9
