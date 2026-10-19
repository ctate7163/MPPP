"""The Navcam joint machinery and the consensus candidate (mppp.calib.joint, mppp.calib.fit)."""
import json
import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path


pycolmap = pytest.importorskip("pycolmap")


def _cam(model="FULL_OPENCV", cid=1):
    p = [2956.0, 2955.5, 2591.0, 1944.0, 0.306, -0.026, 1.5e-4, 1.8e-4, 0.002, 0.594, 0.0, 0.0]
    return pycolmap.Camera(camera_id=cid, model=model, width=5120, height=3840, params=np.array(p))


def test_stereo_offset_follows_yaw():
    from mppp.calib.joint import stereo_offset
    from scipy.spatial.transform import Rotation
    L, R = _cam(cid=1), _cam(cid=2)
    T0 = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([-0.424, 0, 0]))
    T1 = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec([0, np.radians(0.01), 0]).as_matrix()),
                          np.array([-0.424, 0, 0]))
    d0, d1 = stereo_offset(L, R, T0), stereo_offset(L, R, T1)
    assert abs(d0["disparity_inf_px"]) < 1e-6
    assert abs(abs(d1["disparity_inf_px"] - d0["disparity_inf_px"]) - 2956 * np.radians(0.01)) < 0.02


def test_bundle_adjust_keypoint_scale_equals_scaled_camera():
    """Keypoints divided by s about the principal point fit a camera whose fx, fy are s times smaller."""
    from mppp.calib.joint import keypoint_scales

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


def test_sol_epochs():
    from mppp.calib import sol_epochs
    assert sol_epochs([91, 95, 101, 360, 371, 365]) == [(91, 101), (360, 371)]
    assert sol_epochs([652, 660, 674]) == [(652, 674)]
    assert sol_epochs([]) == []
    assert sol_epochs([1, 20, 45], gap=30) == [(1, 45)]


def test_thermal_keypoint_map_moves_the_principal_point():
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.calib.joint import thermal_keypoint_map
    proj = SimpleNamespace(images=[{"image_id": 1, "name": "a", "sol": 1100}, {"image_id": 2, "name": "b", "sol": 1000}])
    temps = {"a": -10.0, "b": -30.0}
    f = thermal_keypoint_map(proj, temps, -20.0, pp_slopes={1: (0.05, -0.02)}, base_of={7: 1})
    cam = SimpleNamespace(camera_id=7)
    kp = np.array([[100.0, 200.0]])
    out = f(1, kp, cam)                     # dT = +10 -> keypoints move by -(0.5, -0.2)
    assert np.allclose(out, [[100.0 - 0.5, 200.0 + 0.2]])
    assert np.allclose(f(2, kp, cam), [[100.0 + 0.5, 200.0 - 0.2]])
    assert np.allclose(f(1, kp, SimpleNamespace(camera_id=3)), kp)        # another camera: unchanged


def test_write_navcam_candidate(tmp_path):
    """The Navcam candidate folder from a fit_navcam result (the files in use as the template)."""
    from mppp.paths import cmods_dir
    from mppp.calib.fit import write_navcam
    from mppp.sfm.project import PARAM_NAMES, camera_from_colmap_json
    names = PARAM_NAMES["THIN_PRISM_FISHEYE"]
    cur = {e: json.loads((cmods_dir() / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    res = {e: {n: float(v) + (0.5 if n == "cx" else 0.0) for n, v in zip(names, cur[e]["params"])} for e in ("NL", "NR")}
    res.update({"blocks": {"A": 10, "B": 12}, "rig_abs_mdeg": {"yaw_mdeg": 35.0, "pitch_mdeg": -88.0, "roll_mdeg": -63.0},
                "rig_R": np.eye(3).tolist(), "residuals": {"median_px": 0.18, "rms_px": 0.38, "p95_px": 0.8},
                "thermal": {"ppm_per_degC": 41.0, "T0_degC": -20.0,
                            "pp_slopes": {"NL": [0.0, 0.0], "NR": [-0.0525, 0.0]},
                            "axis_ppm": {"NL": [40.7, 45.2], "NR": [37.6, 43.5]}, "yaw_rate_mdeg_per_1000sol": 2.56,
                            "k1_slopes": {"NL": -7.2e-6, "NR": 0.0}, "k1_source": "test", "terms": ["NR_cx"],
                            "sd": {"NR_cx": 0.0003, "yaw_rate": 0.1}},
                "drift": {"model": "linear+hinge", "sol0": 900.0, "yaw_mdeg_per_sol": 0.00256}, "sd": {"NL": {"fx": 0.01}},
                "early_mission": {"before_sol": 380.0, "offsets_px": {"fx": 0.2, "fy": 0.25, "cx": 0.32, "cy": -0.57},
                                  "sd_px": {"fx": 0.01, "fy": 0.04, "cx": 0.05, "cy": 0.05}, "z": {"fx": 13.0, "fy": 6.0, "cx": 7.0, "cy": -11.0},
                                  "applied_terms": ["fx", "fy", "cx", "cy"], "early_images": 905}})
    d = write_navcam(res, tmp_path / "navcam_joint", note="test")
    nl = camera_from_colmap_json(d / "M2020_NL_fisheye_tangential.json")
    assert abs(nl["params"][2] - (cur["NL"]["params"][2] + 0.5)) < 1e-9 and nl["params"][10] == nl["params"][11] == 0.0
    j = json.loads((d / "M2020_NL_fisheye_tangential.json").read_text())
    assert j["fixed_params"] == [] and j["sd"]["fx"] == 0.01 and "2 blocks" in j["source"] and j["form"] == "navcam-v1"
    th = j["thermal"]
    assert th["fx_ppm_per_degC"] == 40.7 and th["k1_per_degC"] == -7.2e-6 and "ppm_per_degC" not in th
    assert json.loads((d / "M2020_NR_fisheye_tangential.json").read_text())["thermal"]["cx_px_per_degC"] == -0.0525
    assert j["early_mission"]["dcx_px"] == 0.32 and j["early_mission"]["applied_terms"] == ["fx", "fy", "cx", "cy"]
    rig = json.loads((d / "M2020_N_rig.json").read_text())
    assert rig["yaw_constant"] and "thermal" not in rig and rig["drift"]["sol0"] == 900.0 and rig["form"] == "navcam-v1"
    assert rig["translation"] == "fitted"                                   # carried over from the rig in use
    assert (d / "consensus.json").is_file()
    from mppp.calib import check_form
    assert check_form(d)["ok"]
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "promote_cmods.py"), str(d), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
