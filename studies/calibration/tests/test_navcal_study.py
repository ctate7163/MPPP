"""Tests of the Navcam calibration studies (studies/calibration/navcal*.py)."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
TESTS = HERE.parents[1] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
import navcal as NC  # noqa: E402

pycolmap = pytest.importorskip("pycolmap")


def _cam(model="FULL_OPENCV", cid=1):
    p = [2956.0, 2955.5, 2591.0, 1944.0, 0.306, -0.026, 1.5e-4, 1.8e-4, 0.002, 0.594, 0.0, 0.0]
    return pycolmap.Camera(camera_id=cid, model=model, width=5120, height=3840, params=np.array(p))


def test_wls_tau_and_slope():
    from navcal import _wls
    rng = np.random.default_rng(1)
    x = np.linspace(-40, -10, 12)
    y = 2.0 - 0.9 * x + rng.normal(0, 0.2, x.size)
    f = _wls(y, np.c_[np.ones_like(x), x], np.full(x.size, 0.04))
    assert abs(f["coef"][1] + 0.9) < 0.05 and f["tau"] < 0.2
    y2 = y + rng.normal(0, 3.0, x.size)                                  # extra between-block scatter
    f2 = _wls(y2, np.c_[np.ones_like(x), x], np.full(x.size, 0.04))
    assert f2["tau"] > 1.0 and f2["se"][1] > f["se"][1]


def test_write_joint_cameras_roundtrip(tmp_path):
    from navcal import write_joint_cameras
    from mppp.sfm.project import camera_from_colmap_json
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    cam = {"model": "FULL_OPENCV", "width": 5120, "height": 3840, "params": dict(zip(names, _cam().params)),
           "free_params": list(names[:10]), "sd": {n: 0.1 for n in names[:10]}, "covariance": np.eye(10).tolist()}
    joint = {"lens": "rational", "NL": cam, "NR": cam, "T0_degC": -18.0, "ppm_per_degC": 40.0, "sd_ppm_per_degC": 2.0,
             "final": {"observations": 1000}, "blocks": {"A": 10, "B": 12}, "rig_R": np.eye(3).tolist(),
             "rig_t": [-0.424, 0, 0], "rig": {"sd_yaw_mdeg": 0.5},
             "rig_thermal": {"yaw_mdeg_per_degC": -1.0, "pitch_mdeg_per_degC": 0.2, "yaw_profile": {"sd": 0.05}}}
    w = write_joint_cameras(joint, tmp_path, loo={"A": {"camera_difference": {"NL": {"rms_px": 0.2}}}}, T_ref=None)
    d = json.loads(w["NL"].read_text())
    assert d["thermal"]["ppm_per_degC"] == 40.0 and d["verification"]["per_scape"][0]["scape"] == "A"
    c = camera_from_colmap_json(w["NL"], ("b1", "b2"))
    assert np.allclose(c["params"][:4], _cam().params[:4])
    r = json.loads(w["rig"].read_text())
    assert r["thermal"]["yaw_mdeg_per_degC"] == -1.0 and r["thermal"]["T0_degC"] == -18.0


def test_rig_drift_regression():
    from navcal import rig_drift
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
    from navcal import write_joint_cameras
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    cam = {"model": "FULL_OPENCV", "width": 5120, "height": 3840, "params": dict(zip(names, _cam().params))}
    joint = {"NL": cam, "NR": cam, "T0_degC": -19.1, "ppm_per_degC": 38.6, "final": {"observations": 1}, "blocks": {},
             "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0], "rig_drift": {"sol0": 1084.0, "pitch_mdeg_per_sol": 0.005}}
    w = write_joint_cameras(joint, tmp_path)
    r = json.loads(w["rig"].read_text())
    assert r["drift"]["pitch_mdeg_per_sol"] == 0.005 and "thermal" not in r


def _study(name, sol, T, pitch, yaw, roll, R_ref=None, sd=0.05):
    R_ref = np.eye(3) if R_ref is None else R_ref
    R = Rotation.from_rotvec(np.radians(np.array([pitch, yaw, roll]) * 1e-3)).as_matrix() @ R_ref
    rel = NC.rig_angles(R, R_ref)
    ab = {k.replace("_mdeg", "_abs_mdeg"): v for k, v in NC.rig_angles(R).items()}
    row = {**rel, **ab, "sd_pitch_mdeg": sd, "sd_yaw_mdeg": sd, "sd_roll_mdeg": sd}
    return {"scape": name, "R_reference": R_ref.tolist(),
            "network": {"sol_median": float(sol), "T_median_degC": float(T)},
            "rotation_pp": {"rigs": [dict(row)]}, "rotation": {"rigs": [dict(row)]}}


def _synthetic(rate=0.005, early=0.025, K=300.0, kT=-1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, sol in enumerate([60, 180, 240, 360, 480, 700, 800, 900, 1000, 1100, 1200, 1330, 1400, 1460, 1630, 1780, 1970]):
        T = -18 + 4 * math.sin(i)
        p = 1.0 + rate * sol + (early - rate) * min(sol, K) + rng.normal(0, 0.05)
        y = -5.0 + kT * (T + 18) + rng.normal(0, 0.05)
        r = 20.0 - 0.005 * sol + rng.normal(0, 0.05)
        out.append(_study(f"b{i}", sol, T, p, y, r))
    return out


def test_rig_drift_model_recovers_rate_and_hinge():
    st = _synthetic()
    m = NC.rig_drift_model(st, knot_sol=300.0, no_drift=("yaw",))
    assert m["pitch_fit_mdeg_per_sol"] == pytest.approx(0.005, abs=3e-4)
    assert m["pitch_early_mdeg_per_sol"] == pytest.approx(0.025, abs=2e-3)
    assert m["yaw_mdeg_per_sol"] == 0.0 and m["yaw_T_mdeg_per_degC"] == pytest.approx(-1.0, abs=0.05)
    assert m["roll_mdeg_per_sol"] == pytest.approx(-0.005, abs=3e-4)
    # the drift averages to ~0 over the reference blocks (the joint rig applies at their mean)
    offs = [NC.drift_offset_mdeg(m, s["network"]["sol_median"])["pitch_mdeg"] for s in st]
    assert abs(np.mean(offs)) < 0.05


def test_common_reference_removes_label_step():
    st = _synthetic(early=0.005)
    step = Rotation.from_rotvec(np.radians([0.23e-3, 0.0, 0.58e-3])).as_matrix()     # the early label rig
    early = []
    for s in st[:3]:
        r = s["rotation_pp"]["rigs"][0]
        early.append(_study(s["scape"], s["network"]["sol_median"], s["network"]["T_median_degC"],
                            r["pitch_mdeg"], r["yaw_mdeg"], r["roll_mdeg"]))
        # the same physical rig measured against a different label reference
        R = Rotation.from_rotvec(np.radians(np.array([r["pitch_mdeg"], r["yaw_mdeg"], r["roll_mdeg"]]) * 1e-3)).as_matrix()
        e = early[-1]
        e["R_reference"] = step.tolist()
        rel = NC.rig_angles(R, step)
        e["rotation_pp"]["rigs"][0].update(rel)
    mixed = early + st[3:]
    before = [s["rotation_pp"]["rigs"][0]["roll_mdeg"] for s in mixed[:3]]
    NC.common_reference(mixed)
    after = [s["rotation_pp"]["rigs"][0]["roll_mdeg"] for s in mixed[:3]]
    truth = [s["rotation_pp"]["rigs"][0]["roll_mdeg"] for s in st[:3]]
    assert np.allclose(after, truth, atol=1e-6) and not np.allclose(before, truth, atol=0.1)
    assert mixed[0]["reference_offset_mdeg"]["roll_mdeg"] == pytest.approx(0.58, abs=1e-3)


def test_drift_robustness_holdout_and_jackknife():
    st = _synthetic(early=0.005)
    rob = NC.drift_robustness(st, holdout=["b0", "b5"])
    assert rob["pitch_jackknife"]["sign_stable"] and rob["roll_jackknife"]["sign_stable"]
    for p in rob["holdout"]["predictions"]:
        assert abs(p["pitch"]["z"]) < 4


def test_split_rig_by_epoch_gives_each_epoch_its_rig(tmp_path):
    import sys
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import _synthetic_block
    from navcal import split_rig_by_epoch
    from mppp.sfm.reconstruction import bundle_adjust
    proj, rec, noise = _synthetic_block(tmp_path)
    for r in proj.images:
        r["sol"] = 95 if r["station"].endswith("0000") else 365
    new, proj2, rows = split_rig_by_epoch(rec, proj, {r["name"]: {"T": r["camera_temperature_degC"]} for r in proj.images})
    assert [(r["sol_min"], r["sol_max"]) for r in rows] == [(95, 95), (365, 365)]
    assert rows[0]["T_median_degC"] == -32.0 and rows[1]["T_median_degC"] == -14.0
    assert sum(r["frames"] for r in rows) == len(rec.frames)
    assert rows[0]["rig_id"] != rows[1]["rig_id"] and len(new.rigs) > len(rec.rigs)
    assert set(new.cameras) == set(rec.cameras)                       # the cameras stay shared
    assert new.num_reg_images() == rec.num_reg_images() and len(new.points3D) == len(rec.points3D)
    for r in rows:                                                     # each epoch's frames use its rig
        fr = [f for f in new.frames.values() if f.rig_id == r["rig_id"]]
        assert len(fr) == r["frames"]
    a, b = (new.rigs[r["rig_id"]] for r in rows)
    sa = next(iter(a.non_ref_sensors))
    assert np.allclose(a.sensor_from_rig(sa).matrix(), b.sensor_from_rig(sa).matrix())
    ba = bundle_adjust(new, proj2, sigma_px=noise, loss_scale=10.0, max_iterations=10, refine_rig="rotation")
    assert np.isfinite(ba["final_cost"])


def test_expand_epochs_places_each_epoch_at_its_sol():
    from navcal import expand_epochs
    base = {"scape": "Sid", "network": {"sol_median": 360.0, "T_median_degC": -15.7},
            "rotation_pp": {"rigs": [{"pitch_mdeg": 0.9}]}}
    ep = [{"epoch": 0, "sol_min": 91, "sol_max": 101, "sol_median": 98.0, "T_median_degC": -22.2, "frames": 28,
           "pitch_mdeg": -3.8},
          {"epoch": 1, "sol_min": 360, "sol_max": 371, "sol_median": 361.0, "T_median_degC": -8.6, "frames": 21,
           "pitch_mdeg": 1.2}]
    one = {"scape": "Van Zyl", "network": {"sol_median": 60.0, "T_median_degC": -20.0},
           "rotation_pp": {"rigs": [{"pitch_mdeg": -5.0}]}, "epochs": {"skipped": "one sol epoch"}}
    out = expand_epochs([dict(base, epochs={"rigs": ep}), one])
    assert [s["scape"] for s in out] == ["Sid (sols 91-101)", "Sid (sols 360-371)", "Van Zyl"]
    assert out[0]["network"]["sol_median"] == 98.0 and out[1]["network"]["T_median_degC"] == -8.6
    assert out[0]["rotation_pp"]["rigs"][0]["pitch_mdeg"] == -3.8 and out[0]["parent"] == "Sid"
    assert out[2] is one
    few = [dict(r, frames=3) if r["epoch"] == 0 else r for r in ep]      # too few frames in one epoch: not split
    assert [s["scape"] for s in expand_epochs([dict(base, epochs={"rigs": few})])] == ["Sid"]


def test_write_joint_cameras_at_reference_temperature(tmp_path):
    from navcal import NAVCAM_T_REF_DEGC, write_joint_cameras
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    p = [2956.0, 2956.0, 2591.0, 1944.0, 0.3, -0.02, 0.0, 0.0, 0.002, 0.59, 0.0, 0.0]
    cam = {"model": "FULL_OPENCV", "width": 5120, "height": 3840, "params": dict(zip(names, p)),
           "free_params": list(names[:10]), "sd": {n: 0.1 for n in names[:10]}, "covariance": np.eye(10).tolist()}
    joint = {"lens": "rational", "NL": cam, "NR": cam, "T0_degC": -18.0, "ppm_per_degC": 40.0,
             "final": {"observations": 1000}, "blocks": {"A": 10}, "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0],
             "rig_thermal": {"yaw_mdeg_per_degC": -1.0, "pitch_mdeg_per_degC": 0.0}}
    assert NAVCAM_T_REF_DEGC == -20.0
    w = write_joint_cameras(joint, tmp_path)
    d = json.loads(w["NL"].read_text())
    assert d["thermal"]["T0_degC"] == -20.0 and abs(d["params"][0] - 2956.0 * (1 - 80e-6)) < 1e-6
    assert d["params"][2] == 2591.0 and joint["T0_degC"] == -18.0            # the input is not changed
    r = json.loads(w["rig"].read_text())
    assert r["thermal"]["T0_degC"] == -20.0
    from scipy.spatial.transform import Rotation
    yaw = np.degrees(Rotation.from_matrix(np.array(r["R_sensor_from_ref"])).as_rotvec()[1]) * 1e3
    assert abs(yaw - 2.0) < 1e-6                                             # -1 mdeg/degC x (-2 degC)


def test_joint_cameras_carry_pp_thermal_and_trend(tmp_path):
    from navcal import write_joint_cameras
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


def test_study_start_state_defaults_to_the_cameras_in_use():
    """v0p60: navcam_calibration_study.start_state without --start-cameras starts from src/mppp/data/cmods (the
    projects of MPPP >= 0.50 start from the fisheye consensus; the old search for a rational start found none)."""
    import importlib.util
    import json
    from pathlib import Path
    pytest.importorskip("pycolmap")
    from mppp.paths import cmods_dir
    spec = importlib.util.spec_from_file_location("ncs", HERE / "navcam_calibration_study.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cams, rig, fits = m.start_state({}, None, "fisheye_t")
    want = json.loads((cmods_dir() / "M2020_NL_fisheye_tangential.json").read_text())["params"]
    assert list(map(float, cams["NL"].params)) == list(map(float, want)) and rig[0].shape == (3, 3)
