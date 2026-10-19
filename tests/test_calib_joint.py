"""The Navcam joint machinery and the consensus candidate (mppp.calib.joint, mppp.calib.fit)."""
import json
import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path
import shutil


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


# the rig translation in the joint adjustment (v0p74.4)


def test_baseline_degeneracy_on_a_synthetic_block(tmp_path):
    """The joint adjustment with the translation free, then the +-1 mm profile: the mechanics (the rig set and
    re-converged, finite slopes), and on a block whose scale is only the prior's the cost rise is small."""
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.calib import joint as NC
    from mppp.calib.fit import baseline_degeneracy
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    temps = {}
    xkw = {"pp_slopes": {1: (0.0, 0.0), 2: (0.0, 0.0)}}
    ba = NC.joint_adjust(rec, proj, temps, 0.0, -20.0, max_iterations=30, covariance=False,
                         refine_rig=True, rig_translation_sigma_m=0.05, **xkw)
    assert np.isfinite(ba["final_cost"])
    b0 = float(np.linalg.norm(np.asarray(NC.stereo_rig(rec)[2].translation)))
    dg = baseline_degeneracy(rec, proj, temps, 0.0, -20.0, xkw, None, 1.0, iterations=30, verbose=False)
    assert dg["baseline_m"] == pytest.approx(b0)
    assert len(dg["rows"]) == 2 and all(np.isfinite(r["cost"]) for r in dg["rows"])
    assert np.isfinite(dg["yaw_mdeg_per_mm"]) and np.isfinite(dg["NR_cx_px_per_mm"])
    assert float(np.linalg.norm(np.asarray(NC.stereo_rig(rec)[2].translation))) == pytest.approx(b0)   # rec untouched
    for r in dg["rows"]:
        assert r["cost"] >= ba["final_cost"] - 1e-6 * max(ba["final_cost"], 1.0)


def test_joint_translation_records_the_start_as_a_copy(tmp_path):
    """The start translation must be a copy: pycolmap's Rigid3d.translation is a view into the rig."""
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.calib import joint as NC
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    rid, cid, T = NC.stereo_rig(rec)
    t_view = np.asarray(T.translation, float)
    t_copy = np.array(T.translation, float)
    sid = [s for s in rec.rigs[rid].non_ref_sensors if int(s.id) == cid][0]
    rec.rigs[rid].set_sensor_from_rig(sid, pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(T.rotation.matrix())),
                                                             t_copy * 1.001))
    _, _, T2 = NC.stereo_rig(rec)
    assert np.allclose(np.array(T2.translation), t_copy * 1.001)
    assert np.allclose(t_copy, np.array(rigT[1]))                     # the copy kept the start


def test_sol_epochs_cut_at_the_pitch_knot():
    from mppp.calib.constrain import sol_epochs, RIG_EPOCH_BREAK_SOLS
    from mppp.sfm.project import NAVCAM_RIG_EPOCH_GAP_SOLS, NAVCAM_RIG_EPOCH_BREAKS_SOLS
    assert NAVCAM_RIG_EPOCH_GAP_SOLS == 50.0 and tuple(NAVCAM_RIG_EPOCH_BREAKS_SOLS) == (300.0,) == RIG_EPOCH_BREAK_SOLS
    assert sol_epochs([280, 290, 305, 310], 50) == [(280, 310)]
    assert sol_epochs([280, 290, 305, 310], 50, (300,)) == [(280, 290), (305, 310)]
    assert sol_epochs([280, 290, 300, 310], 50, (300,)) == [(280, 290), (300, 310)]          # sol 300 is after
    assert sol_epochs([178, 185, 190, 341], 50, (300,)) == [(178, 190), (341, 341)]           # Rochette
    assert sol_epochs([91, 95, 101, 361, 378], 50, (300,)) == [(91, 101), (361, 378)]         # Chal Rocks Sid
    assert sol_epochs([1359, 1360, 1408, 1412], 50, (300,)) == [(1359, 1412)]                 # South Arm: one


# k1(T), the consensus change between passes, k1 slopes from the blocks (v0p80)


def test_k1_keypoint_map():
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.calib.joint import k1_keypoint_map

    class P:
        images = [{"image_id": 1, "name": "a", "instrument": "NL"}, {"image_id": 2, "name": "b", "instrument": "NR"}]
    params = [2956.0, 2956.0, 2591.0, 1944.0, 0.0428, -0.0012, 2e-4, 3e-4, -0.0098, 0.0, 0.0, 0.0]
    c = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=params)
    k = np.array([[2591.0, 1944.0], [5000.0, 3800.0]])
    f = k1_keypoint_map(P, {"a": 10.0, "b": -20.0}, -20.0, {"NL": -3.8e-6, "NR": 0.0})
    d = f(1, k, c) - k
    assert np.allclose(d[0], 0, atol=1e-6)                       # the principal point does not move
    assert 0.1 < np.hypot(*d[1]) < 1.0                            # ~0.3 px at the corner for 30 degC
    assert np.allclose(f(2, k, c), k)                             # no slope for NR / at T0


def _write_form(d: Path, k4=0.0012, k1s=-3e-6):
    """A copy of the shipped Navcam models with k4 and the k1 slope set."""
    from mppp.calib.models import _folder
    d.mkdir(parents=True, exist_ok=True)
    for f in ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json"):
        shutil.copy(_folder(None) / f, d / f)
    for eye in ("NL", "NR"):
        f = d / f"M2020_{eye}_fisheye_tangential.json"
        j = json.loads(f.read_text())
        j["params"][j["param_names"].index("k4")] = k4
        j["thermal"]["k1_per_degC"] = k1s
        f.write_text(json.dumps(j))


def test_consensus_change(tmp_path):
    from mppp.calib.fit import PASS_LIMITS, consensus_change
    _write_form(tmp_path / "a", k4=0.0)
    _write_form(tmp_path / "b", k4=0.0)
    ch = consensus_change(tmp_path / "a", tmp_path / "b")
    assert ch["converged"] and ch["worst"]["f_ppm"] == 0
    f = tmp_path / "b" / "M2020_NR_fisheye_tangential.json"
    j = json.loads(f.read_text())
    j["params"][0] *= 1 + 50e-6
    j["params"][2] += 0.02
    f.write_text(json.dumps(j))
    ch = consensus_change(tmp_path / "a", tmp_path / "b")
    assert not ch["converged"] and ch["worst"]["f_ppm"] == pytest.approx(50, abs=0.1)
    assert ch["NR"]["dcx_px"] == pytest.approx(0.02) and PASS_LIMITS["pp_px"] == 0.05


def test_k1_slopes_from_blocks(tmp_path):
    from mppp.calib.fit import k1_slopes_from_blocks
    rng = np.random.default_rng(0)
    cfg = {}
    for b, off in (("A", 0.0), ("B", 2e-4), ("C", -1e-4)):
        rows = []
        for eye, s in (("NL", -4e-6), ("NR", 0.0)):
            for T in np.linspace(-45, -5, 8):
                rows.append({"camera": f"{eye}_T", "eye": eye, "T_median_degC": float(T), "free": ["fx", "fy", "k1"],
                             "k1": 0.043 + off + s * T + rng.normal(0, 2e-6), "observations": 5000})
        d = tmp_path / b / "colmap"
        d.mkdir(parents=True)
        (d / "project.json").write_text(json.dumps({"settings": {"thermal": {"rows": rows}}, "images": []}))
        cfg[b] = str(tmp_path / b)
    r = k1_slopes_from_blocks(cfg, verbose=False)
    assert r["fits"]["NL"]["slope"] == pytest.approx(-4e-6, abs=3e-7)
    assert r["apply"]["NL"] == pytest.approx(r["fits"]["NL"]["slope"]) and r["apply"]["NR"] == 0.0


def test_joint_kw_and_thermal_values_k1():
    from mppp.calib.fit import _joint_kw, _thermal_values
    v = _thermal_values({"ppm_per_degC": 38.0, "k1_slopes": {"NL": -4e-6, "NR": 1e-6}})
    assert v["NL_k1"] == -4e-6
    kw = _joint_kw(v, None)
    assert kw["k1_slopes"] == {"NL": -4e-6, "NR": 1e-6}
    assert "k1_slopes" not in _joint_kw(_thermal_values({"ppm_per_degC": 38.0}), None)


def test_fit_navcam_accepts_the_v0p80_gates():
    import inspect
    from mppp.calib.fit import LOO_LIMITS, fit_navcam
    p = inspect.signature(fit_navcam).parameters
    assert {"k1_slopes", "leave_one_out", "loo_iterations"} <= set(p)
    assert LOO_LIMITS == {"centre_px": 0.05, "corner_px": 0.10}


def test_rig_epochs_at_most_25_sols():
    from mppp.calib.constrain import sol_epochs, RIG_EPOCH_MAX_SPAN_SOLS
    from mppp.sfm.project import NAVCAM_RIG_EPOCH_MAX_SPAN_SOLS
    assert RIG_EPOCH_MAX_SPAN_SOLS == NAVCAM_RIG_EPOCH_MAX_SPAN_SOLS == 25.0
    assert sol_epochs([652, 660, 680, 693], 50, (300,), 25) == [(652, 660), (680, 693)]      # split at the middle sol
    eps = sol_epochs(list(range(20, 100)) + [361, 378], 50, (300,), 25)
    assert eps == [(20, 39), (40, 59), (60, 79), (80, 99), (361, 378)] and all(b - a <= 25 for a, b in eps)
    assert sol_epochs([652, 693], 50, (300,), None) == [(652, 693)]
