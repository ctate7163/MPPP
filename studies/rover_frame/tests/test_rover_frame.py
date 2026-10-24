"""Study 3A on a synthetic rover: known body errors, a known droop law and lever arm, recovered."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from rover_frame import analysis, geometry as G  # noqa: E402

MDEG = analysis.MDEG


def _q(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return [w, x, y, z]


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# the label mounts: camera axes (x right, y down, z forward) in the rover frame (x forward, y right, z down)
R_FWD = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])          # camera looking forward
R_FL = _Ry(0.0) @ (np.array([[1, 0, 0], [0, np.cos(-0.6), -np.sin(-0.6)], [0, np.sin(-0.6), np.cos(-0.6)]]) @ R_FWD)
C_FL = np.array([0.9, -0.12, -0.8])
R_RL = np.array([[-1.0, 0, 0], [0, 1, 0], [0, 0, -1]]) @ R_FL  # looking back (any fixed mount)
C_RL = np.array([-0.6, 0.45, -0.75])
R_NL_HEAD = R_FWD                                              # NL in the head frame
C_NL_HEAD = np.array([0.05, -0.21, -0.1])
O_HEAD = np.array([0.80, 0.56, -1.92])
DROOP = 3.0 / MDEG                                             # 3 mdeg x cos(el) about the head y axis
LEVER = np.array([0.004, -0.002, 0.003])                       # m, NL's centre in the head frame vs the label's


def head_R(az, el):
    return _Rz(az) @ _Ry(-el)                                  # head -> rover


def scene(rng):
    images, states = [], {}
    k = 0
    for block in ("b1", "b2"):
        for st in range(3):
            station = f"S001D{st:04d}"
            heading = rng.uniform(0, 2 * np.pi)
            for pose in (10, 11):                              # two rover poses at the station (the arm moved)
                tilt = rng.normal(0, 0.02, 3)
                R_tel = G.exp(tilt) @ _Rz(heading)              # rnav <- ... (any rotation will do)
                R_tel = R_tel @ G.P_NED_ENU                     # world (ENU) -> rnav
                w_true = rng.normal(0, 0.4 / 57.3, 3)          # the body attitude the telemetry missed
                R_true = G.exp(w_true) @ R_tel
                O_tel = np.array([10.0 * st, 3.0 * st, 0.0])
                O_true = O_tel + rng.normal(0, 0.3, 3)
                t0 = 1e6 * k
                arm = 0 if pose == 10 else 5

                def add(stem, fam, R_c_rnav_true, C_rnav_true, R_c_rnav_lab, C_rnav_lab, head=None, az=0, el=0, dt=0):
                    nonlocal k
                    k += 1
                    R_c_w = R_c_rnav_true @ R_true
                    C_w = O_true + R_true.T @ C_rnav_true
                    images.append({"block": block, "stem": stem, "instrument": stem[:2], "station": station, "sol": 1,
                                   "sequence": f"{fam}{pose}", "R_c_w": R_c_w, "C_w": C_w,
                                   "prior_R_c_w": R_c_rnav_lab @ R_tel, "prior_C": O_tel + R_tel.T @ C_rnav_lab,
                                   "observations": 100})
                    s = {"rmc": {"SITE": 1, "DRIVE": st, "POSE": pose, "ARM": arm, "RSM": k},
                         "rover": {"q_wxyz": _q(G.P_NED_ENU @ R_tel.T), "offset_m": [0, 0, 0]},
                         "R_cam_rnav": R_c_rnav_lab.tolist(), "C_rnav": C_rnav_lab.tolist(), "sclk": t0 + dt,
                         "rsm": {"az_encoder": az, "el_encoder": el, "az_resolver": az, "el_resolver": el},
                         "navcam_plate_T": {"L": -20.0, "R": -20.0}, "chassis": {"left_bogie": 0.001 * pose},
                         "arm": {"j1": 0.1 * arm}}
                    if head is not None:
                        s["head"] = {"q_wxyz": _q(head), "offset_m": O_HEAD.tolist()}
                    states[stem.upper()] = s

                add(f"FLF_{k:06d}", "F", R_FL, C_FL, R_FL, C_FL, dt=0)
                add(f"RLF_{k:06d}", "R", R_RL, C_RL, R_RL, C_RL, dt=10)
                for j in range(8):
                    az, el = rng.uniform(0, 2 * np.pi), rng.uniform(-0.6, 0.5)
                    H = head_R(az, el)
                    R_lab = R_NL_HEAD @ H.T                     # camera <- rnav through the label kinematics
                    C_lab = O_HEAD + H @ C_NL_HEAD
                    e = np.array([0.0, DROOP * np.cos(el), 0.0])
                    R_true_c = R_NL_HEAD @ G.exp(e) @ H.T       # R_c_head = R_label exp(e)
                    C_true = C_lab + H @ LEVER
                    add(f"NLF_{k:06d}", "N", R_true_c, C_true, R_lab, C_lab, head=H, az=az, el=el, dt=20 + j)
    return images, states, rng


def test_synthetic_rover_frame():
    rng = np.random.default_rng(3)
    images, states, _ = scene(rng)
    res = analysis.run(images, states)
    assert res["n_used"] == len(images) and not res["left_out"]
    # every Hazcam pose: the body residual it recovers is the one injected (the left frame defines it exactly)
    assert len(res["groups"]) == 2 * 3 * 2 * 2
    # the Navcam relative to the front body: the head residual is the droop law, the position the lever arm
    nl = [r for r in res["rows"] if r["group"] == "NL"]
    assert all(r["body"] == "front" for r in nl)
    for r in nl:
        el = r["el_encoder"]
        assert r["e_head_y_mdeg"] == pytest.approx(3.0 * np.cos(el), abs=1e-6)
        assert abs(r["e_head_x_mdeg"]) < 1e-6 and abs(r["e_head_z_mdeg"]) < 1e-6
        assert [r[f"d_head_{a}_mm"] for a in "xyz"] == pytest.approx(list(LEVER * 1000), abs=1e-6)
    m = res["mounts"]["NL"]
    assert m["frame"] == "head" and m["position_mm"]["x"]["nmad"] < 1e-6
    fit = res["fits"]["NL"]["pooled"]
    assert fit["axes"]["y"]["coef_mdeg"]["cos_el"] == pytest.approx(3.0, abs=1e-3)
    assert fit["axes"]["y"]["coef_mdeg"]["const"] == pytest.approx(0.0, abs=1e-3)
    assert fit["axes"]["y"]["rms_after_mdeg"] < 1e-6 and fit["axes"]["y"]["rms_before_mdeg"] > 0.1
    # rear vs front at the same pose: the rear Hazcam's mount relative to the front body is the label's
    rl = [r for r in res["rows"] if r["group"] == "RL"]
    assert all(r["body"] == "front" for r in rl)
    assert max(abs(r[f"e_rnav_{a}_mdeg"]) for r in rl for a in "xyz") < 1e-6
    assert max(abs(r[f"d_rnav_{a}_mm"]) for r in rl for a in "xyz") < 1e-6
    # body changes: two poses per station and pair; the adjusted change minus the telemetry change
    ch = res["changes"]
    assert len(ch) == 2 * 3 * 2 and all(c["arm_moved"] and c["pose_changed"] for c in ch)
    assert all(c["missed_mdeg"] > 0 for c in ch)


def test_no_hazcam_pose_means_telemetry():
    rng = np.random.default_rng(5)
    images, states, _ = scene(rng)
    keep = [im for im in images if not im["stem"].startswith(("FLF", "RLF"))]
    res = analysis.run(keep, states)
    assert {r["body"] for r in res["rows"]} == {"telemetry"} and not res["groups"] and not res["mounts"]
    assert res["mounts_telemetry"]["NL"]["images"] == len(keep)


def test_prior_mismatch_left_out():
    rng = np.random.default_rng(7)
    images, states, _ = scene(rng)
    images[0]["prior_R_c_w"] = G.exp([0.01, 0, 0]) @ images[0]["prior_R_c_w"]     # 0.57 deg off
    res = analysis.run(images, states)
    assert res["n_used"] == len(images) - 1 and sum(res["left_out"].values()) == 1


def test_label_state_reader():
    """The keyword names of a real label (the sample products, when present)."""
    import glob
    fs = sorted(glob.glob("/mnt/user-data/uploads/mission_data/zcam/NLF_*.IMG"))
    if not fs:
        pytest.skip("no sample label")
    from rover_frame.labels import read_state
    s = read_state(fs[0])
    assert s["rmc"]["SITE"] > 0 and len(s["rover"]["q_wxyz"]) == 4 and len(s["head"]["q_wxyz"]) == 4
    assert s["rsm"]["az_encoder"] is not None and s["navcam_plate_T"]["L"] is not None
    assert np.asarray(s["R_cam_rnav"]).shape == (3, 3) and "left_bogie" in s["chassis"]


def test_report_and_figures(tmp_path):
    rng = np.random.default_rng(3)
    images, states, _ = scene(rng)
    res = analysis.run(images, states)
    from rover_frame import report
    done = report.write_all(res, tmp_path, {"b1": "x", "b2": "y"})
    md = (tmp_path / "report.md").read_text()
    assert "| NL | head |" in md and "Pointing law" in md and "arm moved" in md
    assert {Path(f).name for f in done["figures"]} >= {"body_attitude.png", "pointing_navcam.png", "body_changes.png",
                                                       "position_scatter.png"}
    assert (tmp_path / "images.csv").stat().st_size > 1000


def test_other_rover_pose_is_carried_over():
    """A mast frame whose rover pose has no Hazcam frame gets the station's measured body error, carried over to
    its own telemetry attitude (here exact: the injected error is the one of the nearest Hazcam pose)."""
    rng = np.random.default_rng(3)
    images, states, _ = scene(rng)
    for im in images:
        s = states[im["stem"].upper()]
        if im["stem"].startswith("NLF") and s["rmc"]["POSE"] == 11:
            s["rmc"] = dict(s["rmc"], POSE=12)
    res = analysis.run(images, states)
    moved = [r for r in res["rows"] if r["group"] == "NL" and r["POSE"] == 12]
    assert moved and all(r["body"] == "front_propagated" for r in moved)
    for r in moved:
        assert r["e_head_y_mdeg"] == pytest.approx(3.0 * np.cos(r["el_encoder"]), abs=1e-6)
    assert res["mounts"]["NL"]["images"] < res["mounts_propagated"]["NL"]["images"]
    assert res["fits"]["NL"]["measured"]["n"] < res["fits"]["NL"]["pooled"]["n"]
