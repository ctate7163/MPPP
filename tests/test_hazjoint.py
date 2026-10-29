"""The Hazcam joint (0.94.4, Christian 24 Oct): the slope fitted on the observations, FL's slope for all four eyes,
the early-mission offsets kept only where significant, the temperature bins."""
import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from mppp.calib import hazjoint as HJ  # noqa: E402
from mppp.calib import models as M  # noqa: E402


def _base(eye):
    return np.asarray(M.load_camera(eye[0], eye[1])["params"], float)


def _rot(rng, s=0.3):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rng.normal(0, s, 3)).as_matrix()


def synthetic(rng, slope_ppm=30.0, early_px=0.0, n_blocks=6, per_block=6, n_pts=200, noise=0.2):
    """Hazcam images of known law: points 2-12 m in front, each image at its own temperature."""
    law = {e: _base(e) for e in HJ.EYES}
    images = []
    for b in range(n_blocks):
        sol = 100.0 + 150.0 * b
        for k in range(per_block):
            for eye in HJ.EYES:
                if eye[0] == "R" and k % 2:
                    continue                       # fewer rear images
                T = (rng.uniform(-35, 5) if eye[0] == "F" else rng.uniform(-6, 11))
                p = law[eye].copy()
                p[0] *= 1 + 1e-6 * slope_ppm * (T - HJ.T0)
                p[1] *= 1 + 1e-6 * slope_ppm * (T - HJ.T0)
                if sol < HJ.EARLY_SOL:
                    p[0] += early_px
                    p[1] += early_px
                R, t = _rot(rng), rng.normal(0, 0.5, 3)
                dirs = np.c_[rng.uniform(-1.2, 1.2, (n_pts, 2)), np.ones(n_pts)]
                Xc = dirs * rng.uniform(2, 12, (n_pts, 1))
                X = (Xc - t) @ R                                   # world = R^T (Xc - t)
                cam = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=p)
                scale = 1.0 if eye[0] == "F" else 0.25
                xy = np.asarray(cam.img_from_cam(Xc), float) + rng.normal(0, noise / scale, (n_pts, 2))
                ok = (xy[:, 0] > 0) & (xy[:, 0] < 5120) & (xy[:, 1] > 0) & (xy[:, 1] < 3840)
                Rn = _rot(rng, 0.002) @ R                           # the start pose a little off
                images.append({"block": f"b{b}", "eye": eye, "name": f"{eye}_{b}_{k}", "T": T, "sol": sol,
                               "scale": scale, "R": Rn, "t": t + rng.normal(0, 0.003, 3), "X": X[ok], "xy": xy[ok],
                               "model": cam.model, "width": 5120, "height": 3840})
    start = {e: law[e] * np.r_[1.0005, 0.9995, np.ones(10)] for e in HJ.EYES}   # f0 start 500 ppm off
    return {"images": images, "start": start, "blocks": [f"b{b}" for b in range(n_blocks)]}, law


def test_joint_recovers_the_slope_and_drops_an_insignificant_early_offset():
    rng = np.random.default_rng(1)
    data, law = synthetic(rng, slope_ppm=30.0, early_px=0.0, per_block=10)
    j = HJ.run_joint(data, slope_mode="FL", slope_ppm=0.0, verbose=False)
    L, cov = j["law"], j["stats"]["cov"]
    s = 1e6 * L.x[L.ix["s"]]
    sd = 1e6 * np.sqrt(cov[L.ix["s"], L.ix["s"]])
    assert abs(s - 30.0) < max(4 * sd, 1.0) and sd < 5.0
    for e in HJ.EYES:
        assert L.x[L.ix[f"Fx_{e}"]] == pytest.approx(law[e][0], rel=2e-4)
    # no early offset in the data: none kept (|z| < 3), each set to 0 and held
    assert all(L.x[L.ix[n]] == 0.0 for n in j["early_held"])
    assert j["early_eyes"] and set(j["early_held"]) == {n for n in L.names if n.startswith(("Ex_", "Ey_"))}


def test_a_real_early_offset_is_kept():
    rng = np.random.default_rng(2)
    data, law = synthetic(rng, slope_ppm=30.0, early_px=3.0, per_block=10)
    j = HJ.run_joint(data, slope_mode="FL", verbose=False)
    L = j["law"]
    assert L.x[L.ix["Ex_FL"]] == pytest.approx(3.0, abs=0.3) and "Ex_FL" not in j["early_held"]


def test_rear_eyes_take_fls_slope_without_pulling_on_it():
    """The rear eyes get FL's slope; their data do not move it (here the rear images follow -100 ppm/degC)."""
    rng = np.random.default_rng(3)
    data, _ = synthetic(rng, slope_ppm=30.0)
    for im in data["images"]:                     # make the rear eyes follow another law
        if im["eye"][0] == "R":
            p = _base(im["eye"])
            f = 1 + 1e-6 * -100.0 * (im["T"] - HJ.T0)
            p[0] *= f
            p[1] *= f
            Xc = im["X"] @ im["R"].T + im["t"]
            cam = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=p)
            im["xy"] = np.asarray(cam.img_from_cam(Xc), float)
    fl = HJ.run_joint(data, slope_mode="FL", verbose=False)
    al = HJ.run_joint(data, slope_mode="all", verbose=False)
    pe = HJ.run_joint(data, slope_mode="per_eye", verbose=False)
    s_fl = 1e6 * fl["law"].x[fl["law"].ix["s"]]
    s_all = 1e6 * al["law"].x[al["law"].ix["s"]]
    assert abs(s_fl - 30.0) < 3.0 and s_all < s_fl - 1.0       # shared, the rear pull it down; FL only, not
    assert 1e6 * pe["law"].x[pe["law"].ix["s_RL"]] == pytest.approx(-100.0, abs=15.0)


def test_temperature_bins_wider_and_thinner_than_the_navcam():
    ims = [{"block": "b", "eye": "RL", "T": T} for T in (-30.0, -29.0, -27.0, 0.0, 1.0, 2.0, 2.5)]
    g = HJ.temperature_bins(ims, bin_deg=5.0, min_images=3, max_deg=15.0)[("b", "RL")]
    assert sorted(len(x) for x in g) == [3, 4]
    assert all(max(ims[i]["T"] for i in x) - min(ims[i]["T"] for i in x) <= 15.0 for x in g)
    one = HJ.temperature_bins([{"block": "b", "eye": "FL", "T": 0.0}], min_images=3)[("b", "FL")]
    assert one == [[0]]


def test_bin_rows_about_the_law():
    rng = np.random.default_rng(4)
    data, law = synthetic(rng, slope_ppm=30.0, n_blocks=3, per_block=4)
    j = HJ.run_joint(data, slope_mode="FL", verbose=False)
    rows = HJ.bin_rows(j["images"], j["law"], HJ.temperature_bins(j["images"]))
    assert rows and all(abs(r["ppm"]) < 6 * r["sd_ppm"] + 50 for r in rows)


def test_block_fit_keeps_early_offsets_only_where_significant():
    from mppp.calib import hazfit as H
    rng = np.random.default_rng(5)
    base = _base("RL")

    def rows(early_px):
        out = []
        for b in range(12):
            T = rng.uniform(-6, 11)
            e = 1.0 if b < 5 else 0.0
            f = base[0] * (1 + 30e-6 * (T - H.T0_DEGC)) + early_px * e + rng.normal(0, 2.0)
            p = base.copy()
            p[0], p[1] = f, f * base[1] / base[0]
            out.append({"block": f"b{b}", "params": p.tolist(), "start": base.tolist(), "T": T, "sol": 100 if e else 900,
                        "observations": 5000, "weak": False, "early_share": e})
        return out

    quiet = H.fit_eye(rows(0.0), "RL")
    assert not quiet["early_mission"]["fitted"] and quiet["early_mission"]["dfx_px"] == 0.0
    assert quiet["early_mission"].get("tested") and abs(quiet["early_mission"]["z"]["fx"]) < 3
    loud = H.fit_eye(rows(12.0), "RL")
    assert loud["early_mission"]["fitted"] and loud["early_mission"]["dfx_px"] == pytest.approx(12.0, abs=3.0)
    assert H.camera_file(quiet, "x")["early_mission"]["applied_terms"] == []


def test_joint_result_is_written_as_candidate_files(tmp_path):
    from mppp.calib import hazfit as H
    rng = np.random.default_rng(6)
    data, law = synthetic(rng, slope_ppm=30.0, per_block=10)
    j = HJ.run_joint(data, slope_mode="FL", verbose=False)
    rows = HJ.bin_rows(j["images"], j["law"], HJ.temperature_bins(j["images"]))
    res = {"when": "2026-10-24", "method": "joint", "blocks": data["blocks"], "T0_degC": HJ.T0, "early_mission_sol": 380.0,
           "eyes": {e: HJ.eye_result(j, e, rows, data["start"][e], {e: {"ppm_per_degC": 30.0, "sd": 1.0}})
                    for e in j["eyes"]}, "limits": {}, "rigs": {}}
    assert {e["ppm_per_degC"] for e in res["eyes"].values()} == {res["eyes"]["FL"]["ppm_per_degC"]}   # FL's slope
    w = H.write_hazcam(res, tmp_path)
    cf = json.loads((tmp_path / "M2020_RL.json").read_text())
    assert cf["thermal"]["slope_source"] == "FL" and "joint" in cf["source"]
    assert cf["thermal"]["fx_ppm_per_degC"] == pytest.approx(res["eyes"]["FL"]["ppm_per_degC"])
    assert any("own slope" in x for x in H.describe(res)) and w["check"]["checks"]


def test_front_mode_fits_fl_and_fr_apart_and_gives_the_rear_their_mean():
    """0.95.1 (Christian, 29 Oct: "The FL and FR should fit separately, and we can assume that both RL and RR are the
    average of FL and FR").  FL follows +20, FR +36 ppm/degC, the rear eyes -100 (they must not pull): FL and FR get
    their own slopes, RL and RR the mean (+28) with its sd; the default mode."""
    assert HJ.SLOPE_MODE == "front"
    rng = np.random.default_rng(4)
    data, _ = synthetic(rng, slope_ppm=20.0, per_block=10)
    for im in data["images"]:
        sl = {"FR": 36.0, "RL": -100.0, "RR": -100.0}.get(im["eye"])
        if sl is None:
            continue
        p = _base(im["eye"])
        f = 1 + 1e-6 * sl * (im["T"] - HJ.T0)
        p[0] *= f
        p[1] *= f
        Xc = im["X"] @ im["R"].T + im["t"]
        cam = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=p)
        im["xy"] = np.asarray(cam.img_from_cam(Xc), float)
    j = HJ.run_joint(data, verbose=False)
    L, cov = j["law"], j["stats"]["cov"]
    assert "s" not in L.ix and {"s_FL", "s_FR"} <= set(L.ix) and "s_RL" not in L.ix
    s_fl, s_fr = 1e6 * L.slope("FL"), 1e6 * L.slope("FR")
    assert s_fl == pytest.approx(20.0, abs=3.0) and s_fr == pytest.approx(36.0, abs=3.0)
    assert 1e6 * L.slope("RL") == pytest.approx(0.5 * (s_fl + s_fr), abs=1e-6) and L.slope("RR") == L.slope("RL")
    assert 0 < L.slope_sd("RL", cov) < max(L.slope_sd("FL", cov), L.slope_sd("FR", cov))
    rows = []
    r = HJ.eye_result(j, "RL", rows, data["start"]["RL"])
    assert r["slope_source"] == "front mean" and "mean of FL's and FR's" in r["slope_note"] and r["sd_ppm_per_degC"] > 0
    f = HJ.eye_result(j, "FR", rows, data["start"]["FR"])
    assert f["slope_source"] == "own" and f["ppm_per_degC"] == pytest.approx(s_fr)
