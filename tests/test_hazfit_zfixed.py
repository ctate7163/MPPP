"""0.93.1: the Hazcam fit from block results (mppp.calib.hazfit), the fixed Mastcam-Z distortion terms per zoom, the
Z joint's fixed globals, and the new Windows scripts."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------- fixed Mastcam-Z terms
def test_zcam_fixed_terms_per_zoom():
    from mppp.calib import forms
    assert forms.zcam_fixed_terms("ZL034") == {"k2": 0.5} and forms.zcam_fixed_terms("ZR034") == {"k2": 0.6}
    assert forms.zcam_fixed_terms("ZL048") == {} and forms.zcam_fixed_terms("ZR048_F02312_reg") == {}
    for z in (63, 79, 110):
        for e in ("ZL", "ZR"):
            assert forms.zcam_fixed_terms(f"{e}{z:03d}_F01234") == {"k2": 0.0, "p1": 0.0, "p2": 0.0}
    assert forms.zcam_fixed_terms("NL") == {} and forms.zcam_fixed_terms("FL") == {} and forms.zcam_fixed_terms("Zx") == {}


def test_apply_zcam_fixed_terms_sets_and_holds():
    from mppp.calib import forms
    names = list(forms.FULL_OPENCV_PARAMS)
    cam = {"model": "FULL_OPENCV", "params": [6000, 6000, 800, 600, -0.4, 0.3, 1e-3, -2e-3, 0, 0, 0, 0],
           "fixed_params": ["k3"], "free_params": ["fx", "fy", "k1", "k2", "p1", "p2"]}
    out = forms.apply_zcam_fixed_terms(dict(cam), "ZR110_F02000")
    assert out["params"][names.index("k2")] == 0.0 and out["params"][names.index("p1")] == 0.0
    assert set(out["fixed_params"]) == {"k2", "p1", "p2", "k3"} and out["free_params"] == ["fx", "fy", "k1"]
    z34 = forms.apply_zcam_fixed_terms(dict(cam), "ZL034_F00500")
    assert z34["params"][names.index("k2")] == 0.5 and z34["params"][names.index("p1")] == 1e-3   # p1, p2 free at 34
    assert "p1" in z34["free_params"] and "k2" not in z34["free_params"]
    assert forms.apply_zcam_fixed_terms(dict(cam), "ZL048")["params"] == cam["params"]          # nothing at 48


def test_zcam_camera_at_keeps_the_promoted_model():
    """The per-image prior keeps the promoted model's own k1, k2 (they belong together); the fixed values are for the
    block BA and the joint."""
    from mppp import calib
    zm = calib.load_zcam(34)
    if zm is None:
        pytest.skip("no Z34 focus model")
    c = zm["cameras"]["ZL034"]
    _, p, _ = calib.zcam_camera_at(zm, "ZL034", c["lines"]["reference_focus"], "backlash")
    assert p[5] == pytest.approx(c["lines"]["backlash"]["k2"]["a"])


def test_zjoint_fixed_globals():
    from mppp.calib import ztied
    G = ztied.GLOBAL_TERMS
    sg = {"eyes": {"ZL034": {"G": np.zeros(len(G))}, "ZR034": {"G": np.zeros(len(G))}}}
    fx = ztied.fixed_globals(sg)
    assert fx == {"ZL034": {"k2": 0.5}, "ZR034": {"k2": 0.6}}
    assert sg["eyes"]["ZL034"]["G"][G.index("k2")] == 0.5 and sg["eyes"]["ZR034"]["G"][G.index("k2")] == 0.6
    sg = {"eyes": {"ZL063": {"G": np.ones(len(G))}}}
    assert ztied.fixed_globals(sg) == {"ZL063": {"k2": 0.0, "p1": 0.0, "p2": 0.0}}
    assert sg["eyes"]["ZL063"]["G"][G.index("p2")] == 0.0 and sg["eyes"]["ZL063"]["G"][G.index("k1")] == 1.0
    sg = {"eyes": {"ZL048": {"G": np.ones(len(G))}}}
    assert ztied.fixed_globals(sg) == {}


# ------------------------------------------------------------------------------------------------ Hazcam fit
def _rows(eye, f0=2150.0, ppm=40.0, early_dfx=1.5, k1_per=2e-5, n=6, seed=0):
    from mppp.calib.hazfit import T0_DEGC
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        T = -30.0 + 6.0 * i
        sol = 100 + 150 * i
        early = 1.0 if sol < 380 else 0.0
        fx = f0 * (1 + ppm * 1e-6 * (T - T0_DEGC)) + early_dfx * early + rng.normal(0, 0.02)
        fy = f0 * 1.0005 * (1 + ppm * 1e-6 * (T - T0_DEGC)) + early_dfx * early + rng.normal(0, 0.02)
        start = [f0, f0 * 1.0005, 2560.0, 1920.0, 0.015, -0.008, 1e-4, -1e-4, 1e-3, -1e-4, 0.0, 0.0]
        p = list(start)
        p[0], p[1], p[4], p[5] = fx, fy, 0.015 + k1_per * (T - T0_DEGC), -0.0081
        rows.append({"block": f"b{i}", "params": p, "start": start, "T": T, "sol": sol, "observations": 5000,
                     "weak": False, "early_share": early})
    rows.append({"block": "weak", "params": [9999.0] * 12, "start": rows[0]["start"], "T": 0.0, "sol": 500,
                 "observations": 100, "weak": True, "early_share": 0.0})
    return rows


def test_hazcam_fit_eye_recovers_the_law():
    from mppp.calib import hazfit
    e = hazfit.fit_eye(_rows("FL"), "FL")
    assert e["blocks"] == [f"b{i}" for i in range(6)] and e["weak_blocks"] == ["weak"]
    assert e["ppm_per_degC"] == pytest.approx(40.0, abs=1.0) and e["slope_fitted"]
    assert e["early_mission"]["fitted"] and e["early_mission"]["dfx_px"] == pytest.approx(1.5, abs=0.1)
    assert e["k1_per_degC"] == pytest.approx(2e-5, rel=0.02) and e["params"][5] == pytest.approx(-0.0081)
    assert e["params"][0] == pytest.approx(2150.0, abs=0.1) and e["params"][2:4] == [2560.0, 1920.0]
    few = hazfit.fit_eye(_rows("RL", n=2), "RL")
    assert "skipped" in few and "params" not in few


def test_hazcam_fit_rig_drift_and_median():
    from scipy.spatial.transform import Rotation
    from mppp.calib import hazfit
    R0 = Rotation.from_euler("xyz", [0.5, -1.0, 0.2], degrees=True)
    rows = []
    for i, sol in enumerate((100, 400, 700, 1000)):
        off = Rotation.from_rotvec(np.radians(1e-3 * np.array([0.0, 2.0 * sol / 100, 0.0])))   # yaw 0.02 mdeg/sol
        rows.append({"block": f"b{i}", "R": (off * R0).as_matrix().tolist(), "R_start": R0.as_matrix().tolist(),
                     "t": [-0.249, 0.0, 0.0], "sol": sol, "refined": True})
    r = hazfit.fit_rig(rows, "F")
    assert r["baseline_m"] == pytest.approx(0.249) and r["drift"]["blocks"] == 4
    assert r["drift"]["yaw_mdeg_per_sol"] == pytest.approx(0.02, abs=1e-4)
    assert r["drift"]["pitch_mdeg_per_sol"] == pytest.approx(0.0, abs=1e-4)
    few = hazfit.fit_rig(rows[:2], "F")
    assert few["drift"] is None and "no drift" in few["note"]


def test_hazcam_write_check_and_camera_at(tmp_path):
    from mppp import calib
    from mppp.calib import hazfit
    res = {"when": "2026-10-22", "blocks": [f"b{i}" for i in range(6)], "T0_degC": hazfit.T0_DEGC,
           "eyes": {"FL": hazfit.fit_eye(_rows("FL"), "FL"), "FR": hazfit.fit_eye(_rows("FR", seed=1), "FR"),
                    "RL": hazfit.fit_eye(_rows("RL", n=2), "RL")}}
    w = hazfit.write_hazcam(res, tmp_path / "fit")
    chk = w["check"]
    assert chk["eyes"]["FL"] and chk["eyes"]["FR"] and not chk["eyes"]["RL"] and not chk["ok"]
    names = sorted(Path(f).name for f in w["files"])
    assert names == ["M2020_FL.json", "M2020_FR.json"]
    for f in ("hazcam_fit.json", "check.json", "report.md"):
        assert (tmp_path / "fit" / f).is_file()
    m = json.loads((tmp_path / "fit" / "candidate" / "M2020_FL.json").read_text())
    _, p20, _ = calib.camera_at(m, -20.0, 1000)
    _, p0, _ = calib.camera_at(m, 0.0, 1000)
    _, pe, _ = calib.camera_at(m, -20.0, 100)
    assert p0[0] / p20[0] - 1 == pytest.approx(20 * 40e-6, rel=0.05)          # the thermal law
    assert pe[0] - p20[0] == pytest.approx(1.5, abs=0.1)                       # the early-mission offset
    assert p0[4] - p20[4] == pytest.approx(20 * 2e-5, rel=0.05)                # k1 per degC


# ------------------------------------------------------------------------------------------ scripts and groups
def test_new_windows_scripts_exist_crlf():
    for name in ("zcam_joints.bat", "run_hazcam.bat", "hazcam_calibration.bat"):
        b = (ROOT / "scripts" / "windows" / name).read_bytes()
        assert b.count(b"\r\n") == b.count(b"\n") > 5, name
    z = (ROOT / "scripts" / "windows" / "zcam_joints.bat").read_text()
    assert "calib zcam" in z and "--promote-if-ok" in z and "--navcam-group navcam_consensus" in z
    h = (ROOT / "scripts" / "windows" / "run_hazcam.bat").read_text()
    assert "hazcam_consensus" in h and "NAVCAM_INTRINSICS=hold" in h and "hazcam_calibration.bat" in h
    assert "zcam_joints.bat" in (ROOT / "scripts" / "windows" / "run_sites_zcam.bat").read_text()


def test_groups_and_block_sites():
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))
    g = t["groups"]
    assert g["zcam_consensus"] == ["chal_rocks", "marble_mountain"]
    assert g["hazcam_consensus"] == ["chal_rocks_large", "threeforks_north", "belva_crater", "bell_island", "marble_mountain"]
    assert all(s in t["sites"] for s in g["hazcam_consensus"])
    ob = t["official_blocks"]
    assert ob["3"]["site"] == "chal_rocks" and ob["18"]["site"] == "belva_crater" and ob["9"]["site"] == "threeforks_north"
    assert not [b["site"] for b in ob.values() if b["site"].endswith(("_large", "_huge"))]
    assert g["blocks_all"] == [ob[n]["site"] for n in sorted(ob, key=int)]


def test_cli_calib_hazcam_and_navcam_group():
    from mppp.cli import build_parser
    a = build_parser().parse_args(["calib", "hazcam", "--variant", "hazcam", "--promote-if-ok"])
    assert a.which == "hazcam" and a.variant == "hazcam"
    a = build_parser().parse_args(["calib", "nz", "--group", "zcam34_joint,zcam_consensus",
                                   "--navcam-group", "navcam_consensus"])
    assert a.navcam_group == "navcam_consensus" and a.group == "zcam34_joint,zcam_consensus"


def test_shipped_exclusions_22_oct():
    from mppp.pipeline import shipped_exclusions
    ex = shipped_exclusions()
    assert {"seq:ZCAM05060", "seq:ZCAM05062", "seq:ZCAM03017", "*_0076_0673685057_000*"} <= set(ex)


def test_reset_cameras_to_start_and_backlash_hold():
    """0.93.1: a weak Hazcam eye is put back to its start camera and held in every later adjustment, the Mastcam-Z
    backlash split included (in a Navcam + Mastcam-Z + Hazcam block the eyes had drifted 1.2 %)."""
    import inspect
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm import backlash, reconstruction as R

    class P:
        settings = {"database": {"cameras": {"FL": 1, "FR": 2}}}
        cameras = {"FL": {"params": [2147.6, 2146.8, 2560.0, 1920.0] + [0.0] * 8},
                   "FR": {"params": [2145.4, 2144.3, 2560.0, 1920.0] + [0.0] * 8}}
    rec = pycolmap.Reconstruction()
    for cid, k in ((1, "FL"), (2, "FR")):
        p = np.array(P.cameras[k]["params"], float)
        p[0] *= 1.012
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=p))
    assert R.reset_cameras_to_start(rec, P, {"FL"}) == ["FL"]
    assert rec.cameras[1].params[0] == pytest.approx(2147.6) and rec.cameras[2].params[0] == pytest.approx(2145.4 * 1.012)
    assert "hold_cameras" in inspect.signature(backlash.backlash_stage).parameters
