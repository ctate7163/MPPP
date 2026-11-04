"""The rover deck fiducial check (mppp.calib.fiducials, 0.95.5): detection, the pair solutions, the summary."""
import json
from pathlib import Path

import numpy as np
import pytest

from mppp.calib import fiducials as F
from mppp.paths import data_dir

pytest.importorskip("cv2")
pytest.importorskip("scipy")


def _navcam(eye="L"):
    m = json.loads((data_dir() / "cmods" / f"M2020_N{eye}_fisheye_tangential.json").read_text())
    return F.Projector("THIN_PRISM_FISHEYE", m["params"])


def _rig():
    r = json.loads((data_dir() / "cmods" / "M2020_N_rig.json").read_text())
    return np.asarray(r["R_sensor_from_ref"], float), np.asarray(r["t_sensor_from_ref_m"], float)


def _look_at(C, target, up=(0, 0, -1)):
    """x_cam = R (X - C): z toward the target, y down in the image (rover z is down)."""
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float) * -1.0, z)        # image x to the right of the view
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def _scene():
    P = F.survey_nav_m()
    C = np.array([0.55, 0.60, -1.90])                    # the NL centre seen on sol 203 (rover frame, m)
    R, t = _look_at(C, np.mean([P[k] for k in F.RIGID], 0))
    return P, R, t


def test_survey_nav():
    P = F.survey_nav_m(["TDF2", "AFT1"])
    assert np.allclose(P["TDF2"], [0.6103254, -0.0005257, -1.1339633])
    assert np.allclose(P["AFT1"] * 1000, [-850.4664, 503.9503, -1426.4209])
    assert "AFT2" in F.NOT_USED and set(F.BAR) == {"LDIFF", "RDIFF"}
    assert np.isclose(np.linalg.norm(F.survey_nav_m()["LDIFF"] - F.survey_nav_m()["RDIFF"]), 0.9099, atol=1e-3)


def test_pixel_conventions():
    xy = np.array([10.0, 20.0])
    assert np.allclose(F.full_to_native(F.native_to_full(xy, 0.5), 0.5), xy)
    assert np.allclose(F.native_to_full([0.0, 0.0], 0.25), [2.0, 2.0])       # centre of the first native pixel


def _render(cx, cy, r=18.0, shape=(120, 120), aspect=0.6, angle=0.4, ss=6):
    """A quadrant target (light/dark quarters inside a dark ring) on a light deck, foreshortened, supersampled."""
    h, w = shape
    ys, xs = np.mgrid[0:h * ss, 0:w * ss]
    u = (xs + 0.5) / ss - 0.5 - cx
    v = (ys + 0.5) / ss - 0.5 - cy
    ca, sa = np.cos(angle), np.sin(angle)
    a, b = ca * u + sa * v, (-sa * u + ca * v) / aspect
    rr = np.hypot(a, b)
    img = np.full(u.shape, 200.0)
    img[rr < 1.3 * r] = 40.0
    q = (a * b > 0)
    img[(rr < r) & q] = 230.0
    img[(rr < r) & ~q] = 25.0
    return img.reshape(h, ss, w, ss).mean((1, 3)).astype(np.float32)


@pytest.mark.parametrize("cx,cy", [(60.3, 58.7), (61.75, 60.2)])
def test_detect_target_subpixel(cx, cy):
    g = _render(cx, cy)
    d = F.detect_target(g, cx + 4, cy - 3, scale=1.0)
    assert d is not None
    assert abs(d["x"] - cx) < 0.1 and abs(d["y"] - cy) < 0.1
    assert d["cost"] < 0.2


def test_detect_target_rejects_flat_and_far():
    assert F.detect_target(np.full((120, 120), 255, np.float32), 60, 60) is None       # saturated deck
    g = _render(60.3, 58.7)
    assert F.detect_target(g, 60.3 + 40, 58.7, search_full_px=6) is None               # target outside the window


def test_similarity_recovers_scale():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 3))
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R *= np.sign(np.linalg.det(R))
    B = 1.0007 * (R @ A.T).T + [1, 2, 3]
    s = F.similarity(A, B)
    assert abs(s["s"] - 1.0007) < 1e-9 and s["rms"] < 1e-9


def _observe(proj, R, t, P, names, rng, sd):
    out = {}
    for k in names:
        uv = proj((R @ P[k] + t)[None])[0]
        if np.all(np.isfinite(uv)) and 0 < uv[0] < 5120 and 0 < uv[1] < 3840:
            out[k] = uv + rng.normal(0, sd, 2)
    return out


def test_scene_sees_the_deck():
    P, R, t = _scene()
    obs = _observe(_navcam("L"), R, t, P, F.RIGID, np.random.default_rng(0), 0.0)
    assert len(obs) >= 5


def test_solve_pair_measures_the_baseline_and_rig():
    rng = np.random.default_rng(3)
    P, R, t = _scene()
    pl, pr = _navcam("L"), _navcam("R")
    R_rel, t_rel = _rig()
    k_true = -800e-6                                          # the deck's baseline is 800 ppm shorter than the block's
    w_true = np.radians(np.array([0.0, 20e-3, -10e-3]))       # and its rig turned by (0, +20, -10) mdeg
    R_rel_true = F._rot(w_true) @ R_rel
    RR, tR = R_rel_true @ R, R_rel_true @ t + t_rel * (1 + k_true)
    oL = _observe(pl, R, t, P, F.RIGID, rng, 0.05)
    oR = _observe(pr, RR, tR, P, F.RIGID, rng, 0.05)
    # start 2 deg and 5 cm off, as a label would be (more than the labels are)
    R0 = F._rot(np.radians([1.0, -1.0, 1.0])) @ R
    t0 = t + [0.03, -0.02, 0.04]
    sc = F.solve_pair(pl, pr, oL, oR, R0, t0, R_rel=R_rel_true, t_rel=t_rel, mode="scale", points=P)
    assert abs(sc["scale_ppm"] - 1e6 * k_true) < max(150.0, 4 * sc["scale_sd_ppm"])
    assert sc["rms"] < 0.15
    fr = F.solve_pair(pl, pr, oL, oR, R0, t0, R_rel=R_rel, t_rel=t_rel, mode="free", points=P)
    assert np.allclose(fr["dR_mdeg"], np.degrees(w_true) * 1e3, atol=8.0)
    rig = F.solve_pair(pl, pr, oL, oR, R0, t0, R_rel=R_rel, t_rel=t_rel, mode="rig", points=P)
    assert rig["rms"] > 3 * fr["rms"]                         # the wrong rig shows in the residual
    one = F.solve_pair(pl, None, oL, {}, R0, t0, mode="rig", points=P)
    assert one["rms"] < 0.15 and np.allclose(one["R"], R, atol=2e-4)
    assert F.solve_pair(pl, None, dict(list(oL.items())[:2]), {}, R0, t0, points=P) is None


def test_solve_pair_drops_a_false_lock():
    rng = np.random.default_rng(5)
    P, R, t = _scene()
    pl = _navcam("L")
    oL = _observe(pl, R, t, P, F.RIGID, rng, 0.05)
    bad = sorted(oL)[0]
    oL[bad] = oL[bad] + [15.0, -20.0]
    r = F.solve_pair(pl, None, oL, {}, R, t, mode="rig", points=P)
    assert r["dropped"] == [f"L:{bad}"] and r["rms"] < 0.15


def test_check_pair_end_to_end():
    rng = np.random.default_rng(7)
    P, R, t = _scene()
    pl, pr = _navcam("L"), _navcam("R")
    R_rel, t_rel = _rig()
    k_true = 600e-6
    RR, tR = R_rel @ R, R_rel @ t + t_rel * (1 + k_true)
    names = list(F.RIGID) + list(F.BAR)
    det = {"L": _observe(pl, R, t, P, names, rng, 0.05), "R": _observe(pr, RR, tR, P, names, rng, 0.05)}
    # the block: a world frame where the rover sits at (10, 20, 1) m, heading 30 deg, level; the block's rig has
    # the nominal baseline (600 ppm shorter than the truth)
    hd = np.radians(30.0)
    f = np.array([np.sin(hd), np.cos(hd), 0.0])
    rt = np.array([np.cos(hd), -np.sin(hd), 0.0])
    R_wr = np.stack([f, rt, [0, 0, -1.0]], axis=1)
    o = np.array([10.0, 20.0, 1.0])
    RL_w, tL_w = R @ R_wr.T, t - R @ R_wr.T @ o               # x_cam = R (R_wr^T (X_w - o)) + t
    RR_w, tR_w = R_rel @ RL_w, R_rel @ tL_w + t_rel
    cfw = {"L": (RL_w, tL_w), "R": (RR_w, tR_w)}
    label = {"L": (F._rot(np.radians([0.3, -0.2, 0.1])) @ R, t + [0.004, -0.003, 0.002]), "R": (RR, tR)}
    res = F.check_pair(det, {"L": pl, "R": pr}, label, cfw)
    assert res["n_L"] >= 5 and res["n_R"] >= 4
    assert abs(res["scale_ppm"] - 600) < max(150, 4 * res["scale_sd_ppm"])
    assert abs(res["tri_scale_ppm"] + 600) < 300                # the block's shorter baseline shrinks the deck
    assert res["tri_shape_rms_mm"] < 0.3
    if "bar_ppm" in res:
        assert abs(res["bar_ppm"] + 600) < 600
    h, p, r = res["rover_in_block_hpr_deg"]
    assert abs(h - 30.0) < 0.05 and abs(p) < 0.05 and abs(r) < 0.05
    assert np.allclose(res["rover_in_block_origin_m"], o, atol=0.01)
    assert np.allclose(np.array(res["nl_in_rover_C_mm"]) / 1e3, -R.T @ t, atol=0.003)


def test_rover_attitude_conventions():
    level_north = np.stack([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]], axis=1)
    assert np.allclose(F.rover_attitude_deg(level_north), [0, 0, 0], atol=1e-9)
    nose_up = F._rot([np.radians(5.0), 0, 0]) @ level_north     # about east (the rover's right axis): nose up
    h, p, r = F.rover_attitude_deg(nose_up)
    assert abs(p - 5.0) < 1e-6 and abs(r) < 1e-6
    right_down = F._rot([0, np.radians(4.0), 0]) @ level_north   # about north (forward): right side down
    h, p, r = F.rover_attitude_deg(right_down)
    assert abs(r - 4.0) < 1e-6 and abs(p) < 1e-6


def test_summarize_and_station_spread():
    pairs = []
    for i in range(8):
        pairs.append({"block": "b1", "station": "S001D0001", "sol": 10 + i, "T_L": -20 + 2 * i, "rig_rms": 0.5,
                      "scale_ppm": -500 + 23 * (2 * i), "scale_sd_ppm": 100,
                      "rover_in_block_hpr_deg": [359.99 + 0.01 * (i % 2), 1, 2],
                      "rover_in_block_origin_m": [1, 2, 3 + 0.001 * i]})
    pairs.append({"block": "b1", "station": "S001D0002", "sol": 30, "rig_rms": None})
    s = F.summarize(pairs)
    assert s["solved"] == 8 and s["pairs"] == 9
    assert abs(s["scale_vs_T"]["ppm_per_degC"] - 23.0) < 1e-6
    st = s["stations"]["b1:S001D0001"]
    assert st["hpr_sd_mdeg"][0] < 10                              # heading wraps at 360
    assert s["blocks"]["b1"]["solved"] == 8


def test_write_csv(tmp_path):
    rows = [{"block": "b", "pair": "1", "scale_ppm": 1.23456789, "nl_in_rover_C_mm": [1.0, 2.0, 3.0], "extra": 5}]
    f = F.write_csv(tmp_path / "p.csv", rows, F.PAIR_COLUMNS)
    text = f.read_text()
    assert "1.234568" in text and "[1.0, 2.0, 3.0]" in text and "extra" not in text


def test_script_finds_blocks(tmp_path):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "fiducial_check.py"
    spec = importlib.util.spec_from_file_location("fiducial_check", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for b in ("mars2020_sol_0201_seitah_huge", "mars2020_sol_0413_threeforks_large"):
        d = tmp_path / b / "colmap"
        (d / "sparse" / "cahv_ba").mkdir(parents=True)
        (d / "sparse" / "cahv_ba" / "images.bin").write_bytes(b"")
        (d / "project.json").write_text("{}")
    (tmp_path / "mars2020_sol_0770_belva_crater_large").mkdir()                  # not solved
    assert [p.name for p in mod.find_blocks(tmp_path, "cahv_ba", [])] == ["mars2020_sol_0201_seitah_huge",
                                                                          "mars2020_sol_0413_threeforks_large"]
    assert [p.name for p in mod.find_blocks(tmp_path, "cahv_ba", ["threeforks_large"])] == ["mars2020_sol_0413_threeforks_large"]
    s = F.summarize([{"block": "b", "station": "s", "sol": 1, "T_L": -10, "rig_rms": 0.4, "scale_ppm": 10.0}])
    import argparse
    rep = mod.report(s, [], tmp_path, argparse.Namespace(model="cahv_ba"))
    assert "Rover deck fiducial check" in rep.read_text()


def test_fit_point_from_rays_and_tdf4():
    rng = np.random.default_rng(11)
    P = F.survey_nav_m(list(F.RIGID) + list(F.UNSURVEYED))
    assert "TDF4" in P and "TDF4" not in F.RIGID
    X = P["TDF4"] + [0.0012, -0.0008, 0.0005]                  # the truth, 1-2 mm from the position in use
    proj = _navcam("L")
    rays = []
    for i, C in enumerate([[0.55, 0.60, -1.90], [0.45, 0.20, -1.95], [0.60, -0.15, -1.92], [0.35, 0.45, -1.88]]):
        R, t = _look_at(np.array(C), P["TDF2"])
        uv = proj((R @ X + t)[None])[0] + rng.normal(0, 0.1, 2)
        rays.append({"proj": proj, "R": R, "t": t, "uv": uv, "image": f"im{i}"})
    R, t = _look_at(np.array([0.5, 0.0, -1.9]), P["TDF2"])
    rays.append({"proj": proj, "R": R, "t": t, "uv": proj((R @ X + t)[None])[0] + [12.0, 0.0], "image": "bad"})
    f = F.fit_point_from_rays(rays, P["TDF4"])
    assert f["dropped"] == ["bad"] and f["n"] == 4
    assert np.allclose(np.array(f["X_nav_mm"]) / 1e3, X, atol=4e-4)
    assert len(f["se_jackknife_mm"]) == 3 and f["rms_px"] < 0.3


def test_annotate_writes_a_figure(tmp_path):
    img = np.full((960, 1280, 3), 120, np.uint8)
    pred = {"TDF1": np.array([600.0, 500.0]), "TDF4": np.array([300.0, 400.0])}
    det = {"TDF1": F.native_to_full([601.2, 499.4], 0.25)}
    out = F.annotate(img, {"stem": "NLF_0001_test", "sol": 1}, 0.25, pred, det, {"TDF1": det["TDF1"] + 0.3}, "L",
                     tmp_path / "a.png")
    assert out.is_file() and out.stat().st_size > 10000
