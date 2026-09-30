"""v0p41: principal-point temperature and sol terms of the Navcam consensus; exposure selection rules."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_thermal_keypoint_map_moves_the_principal_point():
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm.navcal import thermal_keypoint_map
    proj = SimpleNamespace(images=[{"image_id": 1, "name": "a", "sol": 1100}, {"image_id": 2, "name": "b", "sol": 1000}])
    temps = {"a": -10.0, "b": -30.0}
    f = thermal_keypoint_map(proj, temps, -20.0, pp_slopes={1: (0.05, -0.02)}, pp_sol_slopes={1: (-0.001, 0.0)}, sol0=1000,
                             base_of={7: 1})
    cam = SimpleNamespace(camera_id=7)
    kp = np.array([[100.0, 200.0]])
    out = f(1, kp, cam)                     # dT = +10, dsol = +100 -> keypoints move by -(0.5, -0.2) - (-0.1, 0)
    assert np.allclose(out, [[100.0 - 0.5 + 0.1, 200.0 + 0.2]])
    assert np.allclose(f(2, kp, cam), [[100.0 + 0.5, 200.0 - 0.2]])
    assert np.allclose(f(1, kp, SimpleNamespace(camera_id=3)), kp)        # another camera: unchanged


def test_joint_cameras_carry_pp_thermal_and_trend(tmp_path):
    from mppp.sfm.navcal import write_joint_cameras
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
    p = [2956.0, 2956.0, 2591.0, 1944.0, 0.04, -0.01, 2e-4, 3e-4, 0.004, -0.006, 0.0, 0.0]
    cam = {"model": "THIN_PRISM_FISHEYE", "width": 5120, "height": 3840, "params": dict(zip(names, p))}
    joint = {"lens": "fisheye_t", "NL": dict(cam), "NR": dict(cam), "T0_degC": -19.0, "ppm_per_degC": 38.0,
             "final": {"observations": 10}, "blocks": {"A": 1}, "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0],
             "rig_thermal": {"yaw_mdeg_per_degC": 0.0, "pitch_mdeg_per_degC": 0.0},
             "pp_thermal": {"NL": [0.05, 0.0], "NR": [0.0, 0.0]},
             "pp_trend": {"sol0": 974.0, "NL": [-4e-4, 0.0], "NR": [-4.4e-4, 0.0]}}
    w = write_joint_cameras(joint, tmp_path, T_ref=-20.0)
    L = json.loads(w["NL"].read_text()); R = json.loads(w["NR"].read_text()); rig = json.loads(w["rig"].read_text())
    assert L["thermal"]["cx_px_per_degC"] == 0.05 and R["thermal"]["cx_px_per_degC"] == 0.0
    assert abs(L["params"][2] - (2591.0 - 0.05)) < 1e-9 and R["params"][2] == 2591.0      # re-referenced by -1 degC
    assert L["trend"]["sol0"] == 974.0 and L["trend"]["cx_px_per_sol"] == -4e-4
    assert "thermal" not in rig and "principal points" in rig["note"]


def test_exposure_rules():
    from mppp.image import ExposureImage, MPPPImage
    from mppp.process import _exposure_rule
    cfg = {"selection": {"max_exposure_ms": 40.0, "max_centre_tint": 1.2, "exposure_filter_families": ["N"]}}
    assert _exposure_rule(cfg, {"source_product": "NLF_x.IMG", "exposure_duration_ms": 54.06}).startswith("exposure 54.1")
    assert _exposure_rule(cfg, {"source_product": "NLF_x.IMG", "exposure_duration_ms": 18.3}) is None
    assert _exposure_rule(cfg, {"source_product": "ZLF_x.IMG", "exposure_duration_ms": 80.0}) is None
    assert _exposure_rule(cfg, {"source_product": "NRF_x.IMG", "exposure_duration_ms": 10.0, "centre_tint": 1.7})
    im = MPPPImage.__new__(MPPPImage)
    im.fn = SimpleNamespace(stem="NLF_0658_0725353794_270RAD_N0320274NCAM08111_0A0095J01")
    im.config = cfg
    im.label = {"INSTRUMENT_STATE_PARMS": {"EXPOSURE_DURATION": 54.061}}
    with pytest.raises(ExposureImage):
        im._check_exposure()
    # the blue disk: the centre bluer than the edge
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
    rad = np.ones((h, w, 3), np.float32)
    rad[..., 2] = np.where(r < 0.5, 1.8, 1.0)
    with pytest.raises(ExposureImage):
        im._check_centre_tint(rad, np.ones((h, w), np.uint8))
    rad[..., 2] = 1.0
    im._check_centre_tint(rad, np.ones((h, w), np.uint8))
    assert abs(im.centre_tint - 1.0) < 1e-6
