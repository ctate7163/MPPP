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
    for name in ("zcam_joints.bat", "promote_models.bat"):         # 0.94.4: run_hazcam / hazcam_calibration removed
        b = (ROOT / "scripts" / "windows" / name).read_bytes()
        assert b.count(b"\r\n") == b.count(b"\n") > 5, name
    z = (ROOT / "scripts" / "windows" / "zcam_joints.bat").read_text()
    assert "calib zcam" in z and "--promote-if-ok" in z and "--navcam-group ecam_consensus" in z
    assert "zcam_joints.bat" in (ROOT / "scripts" / "windows" / "run_sites_zcam.bat").read_text()


def test_groups_and_block_sites():
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))
    g = t["groups"]
    # 24 Oct (Christian): Chal Rocks instead of Chal Rocks Large, without Pico Turquino - a faster run
    assert g["zcam_consensus"] == ["seitah_overlook", "chal_rocks", "threeforks_north", "belva_crater", "south_arm",
                                   "marble_mountain", "taylorfjellet"]
    assert "chal_rocks_large" not in g["zcam34_joint"] and "chal_rocks" in g["zcam34_joint"]
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
                                   "--navcam-group", "ecam_consensus"])
    assert a.navcam_group == "ecam_consensus" and a.group == "zcam34_joint,zcam_consensus"


def test_shipped_exclusions_22_oct():
    from mppp.pipeline import shipped_exclusions
    ex = shipped_exclusions()
    assert {"seq:ZCAM05060", "seq:ZCAM05062", "seq:ZCAM03017", "*_0076_0673685057_000*", "seq:ZCAM05114",
            "seq:ZCAM05069", "seq:ZCAM05070", "seq:ZCAM05071"} <= set(ex)


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


def test_sites_duplicate_keys_are_errors(tmp_path):
    from mppp.waypoints.sites import validate_site_table
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))
    assert "delta_front_huge" in t["sites"]
    assert not validate_site_table(ROOT / "src" / "mppp" / "data" / "sites.json")[0]
    f = tmp_path / "sites.json"
    f.write_text('{"sites": {"a": {"sols": [[1, 2]]}, "a": {"sols": [[3, 4]]}}, "groups": {}}', encoding="utf-8")
    err, _ = validate_site_table(f)
    assert any("'a' appears more than once" in e for e in err)


def test_excluded_station_is_not_selected_nor_an_anchor(tmp_path):
    """0.93.1 (Christian): rockytop without S026D0282 (sol 460, mid-drive) - its images are not selected and the two
    sol 456 waypoints within 10 m of it (S026D0218, S026D0160) are not added through it."""
    from mppp import load_waypoints
    from mppp.pipeline import load_settings
    from mppp.select import find_imgs_near
    from mppp.waypoints import blocks as B
    for sol, drive in ((460, 282), (461, 630), (456, 218)):
        for eye in ("NLF", "NRF"):
            (tmp_path / f"{eye}_{sol:04d}_0700000000_000RAD_N026{drive:04d}NCAM00200_0A0195J01.IMG").write_bytes(b"")
    wp = load_waypoints(refresh="never")
    _, rep = find_imgs_near(tmp_path, ["NLF", "NRF"], [[460, 530]], wp, radius_m=10.0)
    assert "S026D0218" in {a["station"] for a in rep["stations_added"]}           # through S026D0282
    paths, rep = find_imgs_near(tmp_path, ["NLF", "NRF"], [[460, 530]], wp, radius_m=10.0, exclude_stations=["S026D0282"])
    assert rep["stations_added"] == [] and rep["stations_in_range"] == ["S026D0630"]
    assert rep["stations_excluded"] == ["S026D0282"] and all("N0260630" in p.name for p in paths)
    assert load_settings(site="rockytop").exclude == ["S026D0282"]
    st = B.site_waypoint_stats({"sols": [[460, 530]], "settings": {"EXCLUDE": ["S026D0282"]}},
                               B.waypoint_points(ROOT / "src" / "mppp" / "data" / "M20_waypoints.json"))
    assert st["n_nearby"] == 0 and st["max_waypoint_separation_m"] < 30


def test_rochette_excludes_s006d2594():
    from mppp.pipeline import load_settings
    s = load_settings(site="rochette")
    assert s.exclude == ["S006D2594"] and s.nearby_from_first_sol


def test_block_station_images_on_other_sols_are_selected(tmp_path):
    """0.93.3 (Christian): Chal Rocks' sol 102 Mastcam-Z sequence ZCAM08062 at S004D0218 - the block's own sol 99
    station, the rover not yet moved - comes in although sol 102 is outside the block's sols (91-101)."""
    from mppp import load_waypoints
    from mppp.select import find_imgs_near
    for sol, drive, code, seq in ((99, 218, "NLF", "NCAM00599"), (102, 218, "ZL0", "ZCAM08062"),
                                  (102, 372, "ZL0", "ZCAM08070"), (102, 218, "NLF", "NCAM00502")):
        (tmp_path / f"{code}_{sol:04d}_0700000000_000RAD_N004{drive:04d}{seq}_034085A03.IMG").write_bytes(b"")
    wp = load_waypoints(refresh="never")
    nav, rep = find_imgs_near(tmp_path, ["NLF"], [[91, 101]], wp, radius_m=10.0)
    assert {p.name[:8] for p in nav} == {"NLF_0099", "NLF_0102"}
    assert rep["own_station_other_sols"] == [{"station": "S004D0218", "images": 1, "sols": [102]}]
    z, zr = find_imgs_near(tmp_path, ["ZL0"], [[91, 101]], wp, radius_m=10.0, anchor_stations=rep["stations_in_range"])
    assert [p.name for p in z] == ["ZL0_0102_0700000000_000RAD_N0040218ZCAM08062_034085A03.IMG"]   # not S004D0372
    _, r2 = find_imgs_near(tmp_path, ["NLF"], [[91, 101]], wp, radius_m=10.0, min_sol=103)
    assert r2["own_station_other_sols"] == []                                                  # min_sol holds


def test_zcam00015_kept_and_origny_navcam_only():
    """22 Oct (Christian): ZCAM00015 is kept (not caught by seq:ZCAM00*); Origny has no Mastcam-Z 34 mm 360."""
    from mppp.pipeline import shipped_exclusions
    assert "!seq:ZCAM00015" in shipped_exclusions()
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))
    assert t["sites"]["origny"]["zcam"] == [] and t["sites"]["origny_large"]["zcam"] == []
    assert t["official_blocks"]["35"]["mastcam_z_360"] == []


def test_corner_check_skips_uncalibrated_zooms_and_rejudge():
    """0.93.3: the corner check leaves out the Mastcam-Z zooms without a calibrated focus model (Seitah Overlook,
    22 Oct: ZR063 the only FAIL); a saved FAIL report is judged again under the current rules at the joint gate."""
    from mppp.sfm import health as H
    from mppp import calib
    edges = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    good = {"keypoints": [100] * 9, "triangulated_fraction": [0.5] * 9}
    bad = {"keypoints": [100] * 9, "triangulated_fraction": [0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.02, 0.02, 0.02]}
    cov = {"edges": edges, "cameras": {"NL": good, "ZR063": bad, "ZL034": good}}
    uncal = H.zcam_uncalibrated_groups(cov["cameras"])
    assert ("ZR063" in uncal) == (calib.load_zcam(63) is None) and "ZL034" not in uncal and "NL" not in uncal
    assert H._corner_ratio(cov, skip_groups=set())[0] == "ZR063"
    assert H._corner_ratio(cov, skip_groups={"ZR063"})[0] in ("NL", "ZL034")
    report = {"verdict": "fail", "coverage_by_radius": cov, "checks": [
        {"check": "corner_triangulated_ratio", "value": 0.04, "status": "fail"},
        {"check": "excluded_fraction", "value": 0.25, "status": "fail"},
        {"check": "rig_rotation_change_deg", "value": 0.27, "status": "fail"}]}
    v, changed = H.rejudge(report)
    if "ZR063" in uncal:
        assert v in ("pass", "warn") and len(changed) == 3
    assert H.rejudge({"checks": [{"check": "excluded_fraction", "value": 0.35, "status": "fail"}]})[0] == "fail"


def test_0936_defaults_and_hazcam_over_zcam_blocks(tmp_path):
    """0.93.6 (Christian, 23 Oct): a frame is high unless it is on the low-state list - the blocks split nothing off
    by their focal fits and the found list is not read; calib hazcam --zcam-blocks takes the _zcam folders;
    zcam_joints.bat fits the Hazcams first."""
    from mppp import cli
    from mppp.pipeline import Settings
    s = Settings()
    assert s.zcam_backlash == "list" and s.zcam_low_backlash_found == "auto"       # 0.94.6: on again
    ap = cli.build_parser() if hasattr(cli, "build_parser") else None
    if ap is not None:
        a = ap.parse_args(["calib", "hazcam", "--zcam-blocks", "--group", "zcam_consensus", "--root", str(tmp_path)])
        assert a.zcam_blocks
    bat = (Path(__file__).resolve().parents[1] / "scripts" / "windows" / "zcam_joints.bat").read_bytes()
    assert b"calib hazcam --zcam-blocks --group zcam_consensus" in bat and b"\r\n" in bat


def test_hazcam_rig_change_judged_at_its_own_limit():
    """0.93.6: a Hazcam rig's change from its (label) start fails at 0.5 deg, the Navcam / Mastcam-Z rigs at 0.3 -
    Marble Mountain's rear rig 0.304 deg (23 Oct) no longer fails the block, also in a report from before."""
    from mppp.sfm.health import DEFAULT_THRESHOLDS, rejudge
    assert DEFAULT_THRESHOLDS["hazcam_rig_rotation_change_deg"][:2] == (0.06, 0.5)
    rep = {"checks": [{"check": "rig_rotation_change_deg", "value": 0.304, "status": "fail",
                       "note": "R: baseline 0.93353 -> 0.93353 m"}]}
    assert rejudge(rep)[0] == "warn"
    rep["checks"][0]["note"] = "N: baseline 0.42 -> 0.42 m"
    assert rejudge(rep)[0] == "fail"


def test_mask_inference_off_by_sequence():
    """0.93.8 (Christian, 23 Oct): no_mask_inference_at takes sequence ids as well as stations."""
    from mppp.config import mask_inference_skipped, parse_sequences, parse_stations, sequence_of
    spec = ["S032D1184", "NCAM00699", "seq:ZCAM00034"]
    assert parse_stations(spec) == {(32, 1184)} and parse_sequences(spec) == {"NCAM00699", "ZCAM00034"}
    assert sequence_of("FLF_0681_0727400287_565RAD_N0320908FHAZ08111_0A0095J01") == "FHAZ08111"
    assert mask_inference_skipped(spec, 32, 1184)
    assert mask_inference_skipped(spec, 4, 100, sequence="NCAM00699")
    assert mask_inference_skipped(spec, 4, 100, name="NLF_0093_0675031997_228RAS_N0040100NCAM00699_01_295J01")
    assert not mask_inference_skipped(spec, 4, 100, name="NLF_0093_0675031997_228RAS_N0040100NCAM00700_01_295J01")
    from mppp.waypoints.sites import validate_site_table
    err, _ = validate_site_table()
    assert not err, err


def test_zcam01023_01025_kept_and_lightglue_prior_pairs():
    """0.93.8: ZCAM01025 kept despite seq:ZCAM01* (0.94.4: ZCAM01023 excluded again, and FHAZ00202/215/216);
    LightGlue on a big block gets the prior pairs."""
    from mppp.pipeline import shipped_exclusions
    from mppp.sfm.workdir import apply_exclusions
    metas = [{"stem": f"ZL0_0682_0727400000_000RAD_N0320908{q}_034085A01", "sequence": q, "sol": 682, "site": 32,
              "drive": 908} for q in ("ZCAM01023", "ZCAM01025", "ZCAM01024", "FHAZ00202", "FHAZ00215", "FHAZ00216",
                                     "FHAZ00207")]
    kept, dropped = apply_exclusions(metas, shipped_exclusions())
    assert sorted(m["sequence"] for m in kept) == ["FHAZ00207", "ZCAM01025"]
    import inspect
    from mppp import pipeline
    assert '("lightglue", "sift+lightglue") and big' in inspect.getsource(pipeline)


def test_promote_waits_for_a_held_file(tmp_path, monkeypatch):
    """0.93.8: a model file held by another process for a moment is copied once it is free."""
    from mppp.calib import models as M
    src = tmp_path / "a.json"
    src.write_text('{"x": 1}')
    dst = tmp_path / "out" / "b.json"
    dst.parent.mkdir()
    calls = {"n": 0}
    real = M.os.replace

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "being used by another process")
        return real(a, b)
    monkeypatch.setattr(M.os, "replace", flaky)
    M._copy_retry(src, dst, tries=5, wait_s=0.01)
    assert dst.read_text() == '{"x": 1}' and calls["n"] == 3 and not (dst.parent / "b.json.promote_tmp").exists()
    bat = (Path(__file__).resolve().parents[1] / "scripts" / "windows" / "zcam_joints.bat").read_bytes()
    assert b"zcam_joints.lock" in bat and b'rmdir "%LOCK%"' in bat


def test_error_analysis_batch_and_pipeline_report(tmp_path, monkeypatch):
    """0.93.9: the pipeline's error_analysis stage - the batch over blocks (a failing block does not stop it) and
    its report.md; the new stage list."""
    import copy
    import importlib.util
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "tests"))
    from helpers import _synthetic_model
    from mppp.error import alignment as A
    model, X = _synthetic_model(noise_px=0.2)
    e = tmp_path / "work" / "colmap" / "error_input"
    (e / "native").mkdir(parents=True)
    (e / "stations.csv").write_text("name,station,instrument\n" + "".join(
        f"img{k}.png,{'A' if k <= 3 else 'B'},NL\n" for k in range(1, 7)))
    monkeypatch.setattr(A, "read_colmap", lambda path: copy.deepcopy(model))
    spec = importlib.util.spec_from_file_location("eab", ROOT / "studies" / "analyses" / "error_analysis_batch.py")
    eab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eab)
    out = tmp_path / "ea"
    eab.main([str(out), f"good={tmp_path / 'work'}", f"missing={tmp_path / 'nothing'}", "--n-points", "300",
              "--n-boot", "0"])
    res = json.loads((out / "error_analysis_batch.json").read_text())
    assert "good" in res["alignments"] or "good" in (res.get("failed") or {})
    spec = importlib.util.spec_from_file_location("rp", ROOT / "studies" / "calibration" / "run_pipeline.py")
    rp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp)
    md = rp.error_report(res)
    assert md.startswith("# Error analysis on the SfM points")
    assert rp.STAGES == ("wait", "ecam_runs", "ecam_joint", "zcam_runs", "zcam_joints", "error_analysis", "rover_frame",
                         "lightglue_ab", "zcam_states", "report")


def test_lightglue_ab_report_and_chal_rocks_tiny(tmp_path):
    """0.93.9: Chal Rocks tiny (sols 91-98) and the A/B report from two finished alignments."""
    import importlib.util
    from mppp.waypoints.sites import load_site_table
    t = load_site_table()["sites"]
    assert t["chal_rocks_tiny"]["sols"] == [[91, 98]] and t["chal_rocks_tiny"]["zcam"] == [z for z in t["chal_rocks"]["zcam"] if z != 110]
    for s in ("chal_rocks", "chal_rocks_large", "chal_rocks_tiny"):
        assert t[s]["no_mask_inference_at"][-3:] == ["ZCAM00034", "ZCAM08060", "RHAZ0*"]
    assert t["bell_island"]["no_mask_inference_at"] == ["RHAZ0*"] == t["bell_island_large"]["no_mask_inference_at"]
    from mppp.config import mask_inference_skipped
    assert mask_inference_skipped(["RHAZ0*"], 1, 1, name="RLF_0093_0675031997_228RAS_N0040100RHAZ00102_0A0095J01")
    assert not mask_inference_skipped(["RHAZ0*"], 1, 1, name="RLF_0093_0675031997_228RAS_N0040100RHAZ10102_0A0095J01")
    assert mask_inference_skipped(["RHAZ0"], 1, 1, sequence="RHAZ01234")
    spec = importlib.util.spec_from_file_location("lab", ROOT / "scripts" / "lightglue_ab.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    work = tmp_path / "w"
    for v, m in (("sift", "sift"), ("lightglue", "lightglue")):
        c = work / f"colmap_{v}"
        (c / "health").mkdir(parents=True)
        (c / "run_done.json").write_text(json.dumps({"images": 300, "registered_images": 290 + (v == "lightglue") * 5,
                                                      "residual_rms_native_px": 0.31}))
        (c / "project.json").write_text(json.dumps({"settings": {"matching": {"matcher": m, "matched_pairs": 1000,
                                                                              "verified_pairs": 900, "inlier_matches": 5}}}))
        (c / "health" / "health.json").write_text(json.dumps({"verdict": "warn", "checks": [
            {"check": "cross_station_fraction", "value": 0.02 if v == "sift" else 0.04, "status": "pass", "note": ""}]}))
    log = work / "runs" / "x" / "log.txt"
    log.parent.mkdir(parents=True)
    log.write_text("2026-10-23T10:00:00      | database: {}\n2026-10-23T10:12:00      | matching: {}\n")
    assert lab.matching_seconds(log) == 720
    res = {"when": "now", "gpu": {"cuda": True}, "variants": {v: lab.summary(work, v, {"log": str(log), "seconds": 3600})
                                                             for v in ("sift", "lightglue")}}
    md = lab.report(work, res)
    assert "| matcher that ran | sift | lightglue |" in md and "cross_station_fraction" in md and "| 12 |" in md


def test_check_zcam_rig_scatter_of_too_few_blocks():
    """0.93.9 (Z48 of 23 Oct): a rig scatter that could not be measured (one block per angle: None) is no crash."""
    from mppp.calib.fit import check_zcam
    ln = {"fx": {"a": 6700.0, "b": 0.0}, "fy": {"a": 6702.0, "b": 0.0}}
    new = {"cameras": {"ZL048": {"f0_px": 6701.0, "aspect": 1.0003, "lines": {"backlash": ln}}},
           "rig": {"baseline_m": 0.2436, "sd_rotvec_mdeg": [None, None, None], "blocks": ["a"]}}
    c = {x["check"]: x for x in check_zcam(new, {"cameras": {}})["checks"]}
    assert c["rig scatter"]["ok"] and "not measured" in c["rig scatter"]["note"]


def test_hazcam_athermal_fallback():
    """0.93.10 (Christian): an eye whose temperature law fails its thermal check gets the athermal law (fx, fy, k1
    constant) when that passes every check; else the temperature law stays, with the attempt recorded."""
    from mppp.calib import hazfit as H
    rng = np.random.default_rng(1)
    start = [2147.0, 2147.0, 2560.0, 1920.0, 0.01, -0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def rows(slope_ppm, noise_ppm):
        out = []
        for k, T in enumerate((-23.0, -21.5, -20.0, -18.5, -17.0)):        # a 6 degC span
            f = 2141.0 * (1 + 1e-6 * slope_ppm * (T + 20) + 1e-6 * noise_ppm * rng.normal())
            p = list(start); p[0] = p[1] = f
            out.append({"block": f"b{k}", "T": T, "sol": 600 + k, "params": p, "start": start, "observations": 5000,
                        "weak": False, "early_share": 0.0})
        return out
    e = H.fit_eye_with_fallback(rows(150.0, 30.0), "FL")               # a slope beyond 100 ppm/degC, small scatter
    assert e["athermal"] and e["ppm_per_degC"] == 0.0 and e["k1_per_degC"] == 0.0 and "athermal" in e["fallback"]
    assert all(o for _, o, _ in H.eye_checks(e, "FL"))
    assert "athermal fallback" in H.camera_file(e, "x")["source"] and H.camera_file(e, "x")["thermal"]["fx_ppm_per_degC"] == 0.0
    e2 = H.fit_eye_with_fallback(rows(400.0, 30.0), "RL")              # constant f would scatter beyond 600 ppm
    assert not e2.get("athermal") and e2["athermal_tried"]["failed"]
    e3 = H.fit_eye_with_fallback(rows(40.0, 30.0), "FR")               # a good temperature law stays
    assert not e3.get("athermal") and "athermal_tried" not in e3


def test_0940_exclusions_before_processing_and_exposure_mask_off():
    """0.94.0: the shipped exclusion list applies to the selected products; no_mask_inference_at by exposure."""
    from mppp.pipeline import PROCESS_RULES, exclude_products, product_meta, shipped_exclusions
    assert PROCESS_RULES == 6
    ps = ["/x/ZL0_0091_0675031997_228RAD_N0040100ZCAM03017_034085A03.IMG",
          "/x/ZL0_0091_0675031997_228RAD_N0040100ZCAM00034_034085A03.IMG",
          "/x/NLF_0093_0675176765_000RAS_N0040100NCAM00699_01_295J01.IMG"]
    assert product_meta(ps[0])["filename"] == {"stem": Path(ps[0]).stem, "sol": 91, "site": 4, "drive": 100,
                                               "sequence": "ZCAM03017"}
    kept, ex = exclude_products(ps, shipped_exclusions())
    assert [Path(k).name for k in kept] == [Path(ps[1]).name, Path(ps[2]).name] and len(ex) == 1
    from mppp.config import mask_inference_skipped, parse_stations
    spec = ["0093_0675176765_000", "S032D1184"]
    assert parse_stations(spec) == {(32, 1184)}
    assert mask_inference_skipped(spec, 4, 100, name=Path(ps[2]).stem)
    assert not mask_inference_skipped(spec, 4, 100, name="NLF_0093_0675176999_000RAS_N0040100NCAM00699_01_295J01")
    from mppp.waypoints.sites import load_site_table
    t = load_site_table()["sites"]
    for s in ("chal_rocks", "chal_rocks_large", "chal_rocks_tiny"):
        assert t[s]["no_mask_inference_at"] == ["0093_0675176765_000", "0093_0675177115_000", "0360_0698912834_738", "ZCAM00034", "ZCAM08060",
                                                "RHAZ0*"]


def test_0940_models_in_use_and_first_calibration_ratio():
    """0.94.0: Z63 and FL (athermal) in use; a 110 mm model needs no low-state lines; the start of a new zoom has the
    default high / low ratio."""
    from mppp import calib
    from mppp.calib import forms
    z63 = calib.load_zcam(63)
    assert z63 is not None and all(abs(gm["regular_ratio"] - 1.0095) < 1e-9 for gm in z63["cameras"].values())
    fl = json.loads((ROOT / "src" / "mppp" / "data" / "cmods" / "M2020_FL.json").read_text())
    assert fl["thermal"]["fx_ppm_per_degC"] == 0.0 and fl["thermal"]["k1_per_degC"] == 0.0 and "athermal" in fl["source"]
    m = {"cameras": {"ZL110": {"lines": {"backlash": {"fx": {"a": 1.0}, "fy": {"a": 1.0}}}, "pp0_px": [1, 2]}}}
    assert not forms.validate_zcam(m)
    m["cameras"]["ZL063"] = {"lines": {"backlash": {"fx": {"a": 1.0}, "fy": {"a": 1.0}}}, "pp0_px": [1, 2]}
    assert any("ZL063: no regular line" in x for x in forms.validate_zcam(m))
    from mppp.pipeline import load_settings
    assert load_settings(None, "chal_rocks_tiny").exclude == ["S004D0218", "seq:ZCAM08390"]


def test_0941_tiny_sites_and_joint_commands(tmp_path, monkeypatch):
    """0.94.1: Seitah Overlook tiny; the tiny joint runs the Hazcam fit and a joint per zoom over the finished
    colmap_lightglue blocks, promoting only new models with as few blocks as there are."""
    import importlib.util
    from types import SimpleNamespace
    from mppp.waypoints.sites import load_site_table, work_folder
    t = load_site_table()["sites"]
    assert t["seitah_overlook_tiny"]["sols"] == [[116, 121]]
    spec = importlib.util.spec_from_file_location("lab", ROOT / "scripts" / "lightglue_ab.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    works = {}
    for s in ("chal_rocks_tiny", "seitah_overlook_tiny"):
        w = work_folder(tmp_path, s, True, t[s]["sols"])
        (w / "colmap_lightglue").mkdir(parents=True)
        (w / "colmap_lightglue" / "run_done.json").write_text("{}")
        works[s] = w
    calls = []
    monkeypatch.setattr(lab.subprocess, "call", lambda cmd, cwd=None: calls.append(cmd) or 0)
    (tmp_path / "camera_analysis").mkdir()
    out = lab.tiny_joint(SimpleNamespace(root=str(tmp_path), promote=True), works)
    assert out["exit"] == {"hazcam": 0, "z34": 0, "z48": 0, "z63": 0, "z79": 0, "z110": 0}
    haz = calls[0]
    assert "hazcam" in haz and "--variant" in haz and "--promote-new-only" in haz and "--min-blocks" in haz
    z110 = [c for c in calls if "110" in c][0]
    assert sum(1 for x in z110 if str(x).startswith("seitah_overlook_tiny=")) == 1 and not any(
        str(x).startswith("chal_rocks_tiny=") for x in z110)       # 25 Oct: chal rocks without 110 mm again (faster)
    from mppp.cli import build_parser
    a = build_parser().parse_args(["calib", "zcam", "--zoom", "48", "--min-blocks", "1", "--promote-new-only"])
    assert a.min_blocks == 1 and a.promote_new_only


def test_lightglue_ab_four_variants_and_the_joint_variant(tmp_path):
    """0.94.4: the A/B runs four variants (their settings), reports them side by side with the Navcam-Mastcam-Z
    tie points, and the tiny joint takes the variant with the most of them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lab", ROOT / "scripts" / "lightglue_ab.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    assert list(lab.VARIANTS)[:4] == ["sift", "lightglue", "lightglue_octaves", "sift_lightglue"]
    assert lab.DEFAULT_VARIANTS == ("sift", "lightglue", "lightglue_octaves", "sift_lightglue")
    assert "POSE_GUIDED=false" in lab.VARIANTS["sift_pg_off"] and "ISOLATED_POINTS_FACTOR=0" in lab.VARIANTS["sift_iso_off"]
    assert "MATCH_MODE=sift+lightglue" in lab.VARIANTS["sift_lightglue"]
    assert "LIGHTGLUE_KEYPOINTS=octaves" in lab.VARIANTS["lightglue_octaves"]
    nz = {"sift": 13512, "lightglue": 3786, "lightglue_octaves": 9000, "sift_lightglue": 15000}
    work = tmp_path / "w"
    for v, n in nz.items():
        c = work / f"colmap_{v}"
        (c / "health").mkdir(parents=True)
        (c / "run_done.json").write_text(json.dumps({"images": 110, "registered_images": 106, "mppp": "0.94.4"}))
        (c / "project.json").write_text(json.dumps({"settings": {
            "matching": {"matcher": v, "lightglue": {"added_inliers": 99} if v == "sift_lightglue" else None},
            "pose_guided": {"before": {"navcam_zcam_points": n - 1000}, "after": {"navcam_zcam_points": n},
                            "new_matches": 5, "filtered": 6}}}))
        (c / "health" / "health.json").write_text(json.dumps({"verdict": "warn", "checks": [
            {"check": "points", "value": 1000.0, "status": "info", "note": f"{n} observations"},
            {"check": "focal_change_pct", "value": 0.1, "status": "pass", "note": "FL"},
            {"check": "focal_change_pct", "value": 0.2, "status": "pass", "note": "FR"}]}))
    res = {"when": "now", "gpu": {}, "variants": {v: lab.summary(work, v) for v in nz}}
    md = lab.report(work, res)
    assert "| | SIFT | LightGlue | LightGlue, every octave | SIFT + LightGlue |" in md
    assert "| Navcam-Mastcam-Z tie points | 13512 | 3786 | 9000 | 15000 |" in md
    assert "| LightGlue inliers added to SIFT's | - | - | - | 99 |" in md
    assert "| points | 1000 (info) | 1000 (info) | 1000 (info) | 1000 (info) |" in md      # lined up by check
    assert "| focal_change_pct FL |" in md and "| focal_change_pct FR |" in md
    assert lab.best_variant({"a": work}) == "sift_lightglue"
    assert lab.best_variant({"a": work}, ["sift", "lightglue"]) == "sift"


def test_hazcam_fit_leaves_out_a_block_off_the_law_and_judges_many_blocks_by_precision():
    """0.94.4 (24 Oct): Belva Crater's rear eyes 0.7 % low on 1.5k observations (23 Oct) are left out; an eye of
    many blocks passes its scatter check on scatter / sqrt(blocks)."""
    from mppp.calib import hazfit
    rng = np.random.default_rng(4)
    rows = []
    for i in range(12):
        f = 2155.0 * (1 + rng.normal(0, 1200e-6))
        rows.append({"block": f"b{i}", "params": [f, f + 13, 2560.0, 1920.0, 0.014, -0.008, 0, 0, 0, 0, 0, 0],
                     "start": [2148.0, 2148.0, 2560.0, 1920.0, 0.0145, -0.0083, 0, 0, 0, 0, 0, 0], "T": -20.0 + i * 0.1,
                     "sol": 500 + 10 * i, "observations": 5000, "weak": False, "early_share": 0.0, "images": 10})
    rows.append(dict(rows[0], block="belva", params=[2137.4, 2150.3] + rows[0]["params"][2:], observations=1500))
    e = hazfit.fit_eye(rows, "RL")
    assert e["outlier_blocks"] == ["belva"] and "belva" not in e["blocks"]
    assert e["params"][0] == pytest.approx(2155.0, abs=3.0)
    checks = dict((n, ok) for n, ok, _ in hazfit.eye_checks(e, "RL"))
    assert e["scatter_ppm"] > 600 and checks["RL block scatter"]                 # 12 blocks: ~1200 / sqrt(12)
    few = hazfit.fit_eye(rows[:5], "RL")
    assert not dict((n, ok) for n, ok, _ in hazfit.eye_checks(few, "RL"))["RL block scatter"] or few["scatter_ppm"] <= 600
