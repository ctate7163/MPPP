"""COLMAP project set-up, start cameras, focus bins and models (mppp.sfm.project)."""
import numpy as np
import pytest
from mppp.paths import data_dir  # noqa: E402
from pathlib import Path
import json
from mppp import calib  # noqa: E402
import copy


pycolmap = pytest.importorskip("pycolmap")


DATA_DIR = data_dir()


@pytest.fixture(scope="module")
def processed_pair(tmp_path_factory):
    import mppp
    from conftest import NLF, ZL0, needs_data
    if not (NLF.is_file() and ZL0.is_file()):
        pytest.skip("example IMGs not present")
    out = tmp_path_factory.mktemp("proc")
    cfg = mppp.load_config({"masking": {"infer_mask": False},
                            "export": {"formats": ["PNG8"], "write_colmap": False}})
    man = mppp.process_images([NLF, ZL0], out, cfg, mppp.load_waypoints(), progress=False)
    assert man["n_processed"] == 2
    return man, out


def test_project_with_navcam_and_mastcamz34(processed_pair, tmp_path):
    from mppp.sfm.project import SfmProject, camera_key
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None,
                             navcam_distortion_fit="refine")          # the default holds the distortion
    assert set(proj.cameras) == {"NL", "ZL034"}
    z = proj.cameras["ZL034"]
    assert (z["model"], z["width"], z["height"]) == ("FULL_OPENCV", 1648, 1200) and "label" in z["source"]
    meta = [m for m in man["images"] if m["filename"]["family"] == "Z"][0]
    K = np.asarray(meta["intrinsics"]["K"])
    assert np.isclose(z["params"][0], K[0, 0])
    # ZL's principal point is the focus model's pp0 (the CAHVOR value at the reference focus, one for every block),
    # not the label's (corner origin: K[0, 2] + 0.5)
    from mppp.sfm.project import zcam_focus_model
    pp0 = (zcam_focus_model().get("cameras", {}).get("ZL034") or {}).get("pp0_px")
    assert pp0 and np.isclose(z["params"][2], pp0[0]) and np.isclose(z["params"][3], pp0[1])
    assert abs(z["params"][2] - (K[0, 2] + 0.5)) < 10
    # the distortion is the focus model's, not the label's
    assert np.isclose(z["params"][4], zcam_focus_model()["cameras"]["ZL034"]["distortion"]["params"][0])
    assert z["params"][6] == z["params"][7] == 0.0
    assert {r["instrument"] for r in proj.images} == {"NL", "ZL034"} and proj.rig == {}
    assert camera_key({"family": "Z", "camera_group": "ZR034", "instrument": "ZR"}) == "ZR034"
    # the Navcam eye: the promoted model at the block's (one image's) temperature and the early-mission share
    nl = proj.cameras["NL"]
    nmeta = [m for m in man["images"] if m["filename"]["family"] == "N"][0]
    want = calib.camera_at(calib.load_camera("N", "L"), nmeta["camera_temperature_degC"], nmeta["sol"])[1]
    assert nl["model"] == "THIN_PRISM_FISHEYE" and np.allclose(nl["params"], want)
    info = proj.settings["navcam_cameras"]["NL"]
    assert info["thermal"]["T_median_degC"] == nmeta["camera_temperature_degC"] and "verification" in info
    assert proj.settings["navcam_cameras"]["models_fingerprint"] == calib.models_fingerprint()
    with pytest.raises(ValueError):
        SfmProject.create(man["images"], out, tmp_path / "r", zcam_intrinsics="guess")
    with pytest.raises(ValueError):
        SfmProject.create(man["images"], out, tmp_path / "r", zcam_intrinsics="xml")


def test_rig_family_for_mastcamz_pairs_with_shared_clock():
    from mppp.sfm.project import _rig_from_pairs
    R = np.eye(3).tolist()
    imgs = [{"instrument": k, "eye": k[1], "sclk_key": "1", "prior_R_w2c": R, "prior_C": c}
            for k, c in (("ZL034", [0, 0, 0]), ("ZR034", [0.242, 0, 0]), ("NL", [0, 0, 0]))]
    rig = _rig_from_pairs(imgs)
    assert list(rig) == ["Z034"] and (rig["Z034"]["ref"], rig["Z034"]["sensor"]) == ("ZL034", "ZR034")
    assert abs(rig["Z034"]["baseline_m"] - 0.242) < 1e-9


def test_project_copy_refreshed_when_the_processed_image_changes(tmp_path):
    import os
    from mppp.sfm.project import _link_or_copy
    src, dst = tmp_path / "src.png", tmp_path / "dst.png"
    src.write_bytes(b"v1")
    _link_or_copy(src, dst, link=False)                                 # a copy (e.g. another drive)
    assert dst.read_bytes() == b"v1"
    _link_or_copy(src, dst, link=False)                                 # unchanged: kept
    src.write_bytes(b"version 2")
    os.utime(src, ns=(src.stat().st_atime_ns, dst.stat().st_mtime_ns + 10**9))
    _link_or_copy(src, dst, link=False)
    assert dst.read_bytes() == b"version 2"
    l = tmp_path / "link.png"
    _link_or_copy(src, l, link=True)
    src.write_bytes(b"version 3")                                       # a hard link follows in place
    _link_or_copy(src, l, link=True)
    assert l.read_bytes() == b"version 3"


def test_focus_bins_greedy_and_tags():
    from mppp.sfm.project import _focus_tag, focus_bins
    counts = [100, 110, 129, 131, 200, None, 205, 250]
    assert focus_bins(counts, 30) == [0, 0, 0, 1, 2, 4, 2, 3]            # no grid line splits 129 / 131 from 100?
    assert focus_bins([5, 6, 7], 30) == [0, 0, 0] and focus_bins([], 30) == []
    assert (_focus_tag(2312.4), _focus_tag(-150), _focus_tag(None)) == ("F02312", "Fm00150", "Fna")


def test_project_bins_mastcamz_by_focus(processed_pair, tmp_path):
    from mppp.sfm.project import ZCAM_BIN_HELD, SfmProject
    man, out = processed_pair
    zmeta = [m for m in man["images"] if m["filename"]["family"] == "Z"][0]
    fc = float(zmeta["focus_position_count"])
    key = f"ZL034_F{int(round(fc)):05d}"
    # the test image (ZCAM07114, 34 mm) is on the shipped low-backlash list: a regular-state bin, f held
    reg = SfmProject.create(man["images"], out, tmp_path / "g", link=False, zcam_intrinsics="label")
    assert set(reg.cameras) == {"NL", key + "_reg"} and reg.settings["zcam_low_backlash"]["images"] == 1
    assert {"fx", "fy"} <= set(reg.cameras[key + "_reg"]["fixed_params"])
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_intrinsics="label",
                             zcam_low_backlash=False)
    assert set(proj.cameras) == {"NL", key} and proj.settings["zcam_low_backlash_sha256"] is None
    z = proj.cameras[key]
    assert z["group"] == "ZL034" and z["focus_count_median"] == fc and z["n_images"] == 1
    assert z["fixed_params"] == list(ZCAM_BIN_HELD) and "focus bin" in z["source"]
    r = [r for r in proj.images if r["camera_group"] == "ZL034"][0]
    assert r["instrument"] == key and r["focus_count"] == fc and abs(r["label_f_px"] - z["params"][0]) < 1e-9
    assert proj.settings["zcam_focus_bin"] == 30.0 and proj.settings["zcam_bin_refine"] == "focal"
    pa = SfmProject.create(man["images"], out, tmp_path / "q", link=False, zcam_bin_refine="all",
                           zcam_intrinsics="label", zcam_low_backlash=False)
    assert pa.cameras[key]["fixed_params"] == []
    # default: f from the focus model; a one-image bin holds it
    pm = SfmProject.create(man["images"], out, tmp_path / "m", link=False, zcam_low_backlash=False)
    zm = pm.cameras[key]
    assert "focus model" in zm["source"] and {"fx", "fy"} <= set(zm["fixed_params"])
    assert zm["params"][0] > z["params"][0]                           # the label f is ~1 % short
    with pytest.raises(ValueError):
        SfmProject.create(man["images"], out, tmp_path / "r", zcam_bin_refine="some")


def test_station_labels_start_with_the_sol():
    from mppp.sfm.project import SfmProject, station_labels
    imgs = [{"station": "S032D1184", "sol": 686}, {"station": "S032D1184", "sol": 686},
            {"station": "S032D1174", "sol": 684}, {"station": "S032D1174", "sol": 685}, {"station": "S001D0000"}]
    lab = station_labels(imgs)
    assert lab == {"S032D1184": "Sol0686 S032D1184", "S032D1174": "Sol0684-0685 S032D1174", "S001D0000": "S001D0000"}
    assert sorted(lab.values())[1] == "Sol0684-0685 S032D1174"
    proj = SfmProject(Path("."), imgs, {}, {}, [0, 0, 0], {})
    assert proj.station_label("S032D1184") == "Sol0686 S032D1184" and proj.station_label("X") == "X"


def test_prior_rotation_correction_keeps_the_ray_at_the_camera_principal_point():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import prior_rotation_correction
    rng = np.random.default_rng(4)
    R = Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix()
    f, cl, cc = 4690.0, np.array([770.0, 650.0]), np.array([824.0, 600.0])
    M = prior_rotation_correction(cl, f, cc)
    Rn = M @ R
    ray = lambda Rw2c, c, u: Rw2c.T @ np.r_[(u - c) / f, 1.0]                     # noqa: E731
    u = cc
    a, b = ray(R, cl, u), ray(Rn, cc, u)
    assert np.allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), atol=1e-12)    # same ray at the camera pp
    ang = np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)))
    assert abs(ang - np.degrees(np.arctan(np.linalg.norm(cc - cl) / f))) < 1e-9     # ~0.89 deg here
    u = cc + [300.0, -200.0]                                                        # elsewhere: second order only
    a, b = ray(R, cl, u), ray(Rn, cc, u)
    err_px = f * np.linalg.norm(a / a @ (R.T[:, 2]) - b / b @ (R.T[:, 2]))
    assert err_px < 2.0
    assert np.allclose(prior_rotation_correction(cl, f, cl), np.eye(3))


def test_project_scope_is_checked_first(processed_pair, tmp_path):  # noqa: F811
    from mppp.sfm.project import SfmProject
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None,
                             navcam_distortion_fit="refine")
    nl = proj.cameras["NL"]
    assert nl["free_params"] == [] and proj.settings["navcam_distortion"] == "fisheye_tangential"
    assert abs(nl["params"][6]) > 1e-5                                      # p1/p2 not zeroed by default
    assert set(nl["fixed_params"]) == {"sx1", "sy1", "cx", "cy"}            # the held zeros and the principal point
    # out of scope: rejected before any file is linked into the project
    bad = json.loads(json.dumps(man["images"], default=str))
    bad[0]["filename"]["camera_code"] = "FLF"
    with pytest.raises(ValueError, match="outside MPPP's scope"):
        SfmProject.create(bad, out, tmp_path / "r", link=False, zcam_focus_bin=None)
    assert not any((tmp_path / "r" / "images").glob("*"))
    z = json.loads(json.dumps(man["images"], default=str))
    for m in z:
        if m["filename"]["family"] == "Z":
            m["filename"]["zoom_mm"] = 100                    # 34, 48, 63, 79, 110 are in scope; 100 is not
    with pytest.raises(ValueError, match="outside MPPP's scope"):
        SfmProject.create(z, out, tmp_path / "s", link=False)
    und = json.loads(json.dumps(man["images"], default=str))
    und[0]["undistorted"] = True
    with pytest.raises(ValueError, match="undistort"):
        SfmProject.create(und, out, tmp_path / "t", link=False)
    for kw in ({"navcam_distortion": "rational"}, {"navcam_distortion": "polynomial"}, {"navcam_k4": "zero"},
               {"navcam_rig_yaw": "zero"}):
        with pytest.raises(ValueError):
            SfmProject.create(man["images"], out, tmp_path / "u", **kw)


def test_refresh_images_updates_copies(processed_pair, tmp_path):  # noqa: F811
    import os
    from mppp.sfm.project import SfmProject
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None,
                             navcam_distortion_fit="refine")
    m = man["images"][0]
    src = out / m["outputs"]["PNG8"]
    dst = proj.images_dir / (Path(m["source_product"]).stem + ".png")
    data = dst.read_bytes()
    src.write_bytes(data + b"\0")                                           # processed again
    os.utime(src, ns=(src.stat().st_atime_ns, dst.stat().st_mtime_ns + 10 ** 9))
    assert proj.refresh_images(man["images"], link=False) == 2
    assert dst.read_bytes() == data + b"\0"
    src.write_bytes(data)


def test_focus_model_start_and_hold():
    from mppp.sfm.project import ZCAM_HOLD_F_IMAGES, _split_by_focus, zcam_focus_model
    m = zcam_focus_model()
    g = m["cameras"]["ZL034"]
    assert ZCAM_HOLD_F_IMAGES == 2 and g["focus_range"][0] <= -400 and g["focus_range"][1] == 1300
    base = {"model": "FULL_OPENCV", "params": [4680.0, 4680.0, 824, 600, -0.02, 0.0, 0, 0, 0, 0, 0, 0],
            "source": "median label CAHVOR", "fixed_params": []}
    rows = []
    for foc, n, lf in ((600, 5, 4680.0), (1000, 1, 4700.0), (-2600, 3, 4600.0)):
        rows += [{"camera_group": "ZL034", "focus_count": foc, "label_f_px": lf} for _ in range(n)]
    out = _split_by_focus(rows, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal", model=m, hold_f_images=2)
    by = {c["focus_count_median"]: c for c in out.values()}
    f600 = np.sqrt(by[600]["params"][0] * by[600]["params"][1])
    assert abs(f600 - (g["f0_px"] + g["slope_px_per_count"] * (600 - g["reference_focus"]))) < 0.01
    assert abs(by[600]["params"][1] / by[600]["params"][0] - g["aspect"]) < 1e-9
    assert "fx" not in by[600]["fixed_params"] and {"fx", "fy"} <= set(by[1000]["fixed_params"])
    # outside the fitted range: the bin's label f scaled by the refined/label ratio, not an extrapolation - unless
    # the model carries calibrated lines, which every bin starts on (they continue through zero)
    f_out = np.sqrt(by[-2600]["params"][0] * by[-2600]["params"][1])
    ln = (g.get("lines") or {}).get("backlash")
    if ln:
        ref = g["lines"].get("reference_focus", g["reference_focus"])
        want = np.sqrt((ln["fx"]["a"] + ln["fx"]["b"] * (-2600 - ref)) * (ln["fy"]["a"] + ln["fy"]["b"] * (-2600 - ref)))
    else:
        want = 4600.0 * g["f0_px"] / g["label_f0_px"]
    assert abs(f_out - want) < 0.01 and "outside" in by[-2600]["source"]
    assert "fx" not in by[-2600]["fixed_params"]
    lab = _split_by_focus(rows, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal")      # without a model: label f
    assert {round(c["params"][0]) for c in lab.values()} == {4680, 4700, 4600}


def test_shipped_rig_and_cameras_agree():
    from mppp.paths import data_dir
    from mppp.sfm.project import NAVCAM_RIG, NAVCAM_RIG_FILE
    from scipy.spatial.transform import Rotation
    d = data_dir() / "cmods"
    rig = json.loads((d / NAVCAM_RIG_FILE).read_text())
    assert NAVCAM_RIG == "consensus"
    R = np.asarray(rig["R_sensor_from_ref"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))
    assert 0.05 < ang < 0.5                                   # the two Navcams are toed by ~0.1-0.2 deg
    for e in "LR":
        c = json.loads((d / f"M2020_N{e}_fisheye_tangential.json").read_text())
        assert c["model"] == "THIN_PRISM_FISHEYE" and 2900 < c["params"][0] < 3000 and c["form"] == "navcam-v1"
    assert not (d / "history").exists() and not list(d.glob("*.xml")) and not list(d.glob("*rational*"))


def test_project_create_signature_has_no_removed_options():
    from mppp.sfm import project as P
    import inspect
    sig = inspect.signature(P.SfmProject.create)
    assert "navcam_cameras" in sig.parameters and "xml_dir" not in sig.parameters
    src = inspect.getsource(P)
    assert "XML_PATTERN" not in src and "camera_from_metashape_xml" not in src and "rig_without_yaw" not in src


def test_start_rig_from_the_promoted_rig():
    from scipy.spatial.transform import Rotation
    from mppp.sfm.start_cameras import navcam_start_rig
    rig = calib.load_rig("N")
    R0 = np.asarray(rig["R_sensor_from_ref"], float)
    images = [{"instrument": "NL", "sol": 1400, "camera_temperature_degC": -9.0},
              {"instrument": "NR", "sol": 1400, "camera_temperature_degC": -9.5}]
    ent = {"R_sensor_from_ref": np.eye(3).tolist(), "t_sensor_from_ref": [-0.4244, 0, 0], "baseline_m": 0.4244}
    info = navcam_start_rig(ent, images)
    rv = np.degrees(Rotation.from_matrix(np.asarray(ent["R_sensor_from_ref"]) @ R0.T).as_rotvec()) * 1e3
    off = calib.drift_offset_mdeg(rig["drift"], 1400.0)
    assert np.allclose(rv, [off["pitch_mdeg"], off["yaw_mdeg"], off["roll_mdeg"]], atol=1e-6)
    assert info["drift"]["sol_median"] == 1400.0 and info["T_median_degC"] == -9.25
    assert ent["t_sensor_from_ref"] == rig["t_sensor_from_ref_m"] and "fitted" in ent["rotation_source"]
    assert ent["R_sensor_from_ref_cahv"] == np.eye(3).tolist()
    assert navcam_start_rig(dict(ent), images, "F") is None                 # no Hazcam rig file: CAHV stays


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


def test_project_create_with_start_cameras_from_a_folder(tmp_path):
    from conftest import NLF
    if not NLF.is_file():
        pytest.skip("example IMGs not present")
    import mppp
    from mppp.sfm.project import SfmProject
    out = tmp_path / "proc"
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": False}})
    man = mppp.process_images([NLF], out, cfg, mppp.load_waypoints(), progress=False)
    cand = tmp_path / "joint"
    cand.mkdir()
    src = calib.models._folder(None)
    for n in calib.NAVCAM_FILES:
        (cand / n).write_bytes((src / n).read_bytes())
    d = json.loads((cand / "M2020_NL_fisheye_tangential.json").read_text())
    d["params"][4] = 0.0451
    d["thermal"] = {"T0_degC": -20.0, "fx_ppm_per_degC": 0.0, "fy_ppm_per_degC": 0.0, "cx_px_per_degC": 0.0,
                    "cy_px_per_degC": 0.0, "k1_per_degC": 0.0}
    d.pop("early_mission", None)
    (cand / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(d))
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, navcam_cameras=cand)
    assert proj.cameras["NL"]["model"] == "THIN_PRISM_FISHEYE"
    assert abs(proj.cameras["NL"]["params"][4] - 0.0451) < 1e-9
    assert proj.settings["navcam_cameras"]["dir"] == str(cand) and proj.settings["navcam_cameras"]["NL"]["file"].startswith(str(cand))
    assert proj.settings["navcam_cameras"]["fingerprint"] == calib.navcam_fingerprint(cand)


def test_rig_at_applies_the_hinge():
    rig = calib.load_rig("N")
    m = dict(rig["drift"])
    shipped = {"R_sensor_from_ref": np.eye(3).tolist(), "drift": m}
    from mppp.calib.joint import rig_angles
    for sol in (63.0, 1084.0, 1970.0):
        R, _, applied = calib.rig_at(shipped, sol)
        ang = rig_angles(R)
        exp = calib.drift_offset_mdeg(m, sol)
        for a in ("pitch", "yaw", "roll"):
            assert ang[f"{a}_mdeg"] == pytest.approx(exp[f"{a}_mdeg"], abs=1e-6)
        assert isinstance(applied["drift"]["offset_mdeg"]["pitch"], float)
    # a rig file with rates only still works
    R, _, _ = calib.rig_at({"R_sensor_from_ref": np.eye(3).tolist(),
                            "drift": {"sol0": 1000.0, "pitch_mdeg_per_sol": 0.005}}, 1200.0)
    assert rig_angles(R)["pitch_mdeg"] == pytest.approx(1.0, abs=1e-6)


def _model():
    from mppp.sfm.project import zcam_focus_model
    return zcam_focus_model()


def test_shipped_focus_model_layout():
    m = _model()
    for g in ("ZL034", "ZR034"):
        c = m["cameras"][g]
        assert c["reference_focus"] == 600.0
        assert 0.045 < c["slope_px_per_count"] < 0.07
        # f0 from the bins' fx over 10 blocks, ~1.4 % above the label
        assert 1.006 < c["f0_px"] / c["label_f0_px"] < 1.016
        assert c["thermal"]["sensor"] == "HEAD_FPA" and c["thermal"]["f_px_per_degC"] == 0.0
        assert "trend" not in c and "trend_v0p42" not in c                    # no sol trend in the form
        assert c["lines"]["backlash"]["fx"]["form"] == "common" and c["regular_ratio"] > 1.0
    assert m["cameras"]["ZL034"]["pp"]["cx_px_per_count"] == 0.0
    # ZR cx constant with focus (not significant between blocks), cy keeps its small slope
    assert m["cameras"]["ZR034"]["pp"]["cx_px_per_count"] == 0.0
    assert 0.001 < m["cameras"]["ZR034"]["pp"]["cy_px_per_count"] < 0.004
    assert m.get("form") == "zcam-v1" and m["cameras"]["ZL034"]["pp0_px"] and m["cameras"]["ZR034"]["pp0_px"]


def test_model_focal_terms_and_sol_clamp():
    from mppp.sfm.project import zcam_model_focal
    g = _model()["cameras"]["ZR034"]
    f, t = zcam_model_focal(g, 600.0, T=-15.0, sol=700.0)
    assert f == pytest.approx(g["f0_px"]) and t["thermal"] == 0.0 and t["trend"] == 0.0
    f1, _ = zcam_model_focal(g, 1100.0)
    assert f1 - g["f0_px"] == pytest.approx(500 * g["slope_px_per_count"])
    # a sol trend (none in the form; a model that carried one) does not extrapolate beyond the fitted sols
    g = dict(g, trend={"sol0": 700.0, "f_px_per_sol": 0.025, "sol_range": [461, 991]})
    a, _ = zcam_model_focal(g, 600.0, sol=991)
    b, _ = zcam_model_focal(g, 600.0, sol=1700)
    c, _ = zcam_model_focal(g, 600.0, sol=100)
    d, _ = zcam_model_focal(g, 600.0, sol=461)
    assert a == pytest.approx(b) and c == pytest.approx(d) and b > c
    # a thermal slope is used when the model has one and the temperature is known
    g2 = copy.deepcopy(g)
    g2["thermal"]["f_px_per_degC"] = 0.5
    e, t2 = zcam_model_focal(g2, 600.0, T=-5.0, sol=700.0)
    assert t2["thermal"] == pytest.approx(5.0) and e == pytest.approx(g["f0_px"] + 5.0)
    assert zcam_model_focal(g2, 600.0, T=None, sol=700.0)[1]["thermal"] == 0.0


def test_pp_shift_about_the_median_focus():
    from mppp.sfm.project import zcam_model_pp_shift
    m = _model()["cameras"]
    assert zcam_model_pp_shift(m["ZL034"], 1200.0, 600.0) == (0.0, 0.0)
    dx, dy = zcam_model_pp_shift(m["ZR034"], 1200.0, 600.0)
    assert dx == pytest.approx(600 * m["ZR034"]["pp"]["cx_px_per_count"]) and dy > 0
    assert zcam_model_pp_shift(m["ZR034"], None, 600.0) == (0.0, 0.0)
    assert zcam_model_pp_shift(None, 1200.0, 600.0) == (0.0, 0.0)


def test_split_by_focus_uses_temperature_sol_and_pp():
    from mppp.sfm.project import _split_by_focus, zcam_model_focal
    m = copy.deepcopy(_model())
    m["cameras"]["ZR034"]["thermal"]["f_px_per_degC"] = 0.3            # exercise the thermal path
    m["cameras"]["ZR034"]["trend"] = {"sol0": 700.0, "f_px_per_sol": 0.025, "sol_range": [461, 991]}   # a sol trend
    m["cameras"]["ZR034"].pop("lines", None)                            # the lines would set f and pp instead
    m["cameras"]["ZR034"]["pp"]["cx_px_per_count"] = 0.0025                # a pp slope to exercise (the form's is 0)
    base = {"model": "FULL_OPENCV", "params": [4680.0, 4680.0, 824.0, 600.0, 0, 0, 0, 0, 0, 0, 0, 0],
            "source": "median label CAHVOR", "fixed_params": []}
    rows = []
    for foc, T, sol in ((300, -20.0, 500), (300, -18.0, 500), (900, -10.0, 1500), (900, -12.0, 1500), (600, -15.0, 700)):
        rows.append({"camera_group": "ZR034", "focus_count": foc, "label_f_px": 4690.0,
                     "camera_temperature_degC": T, "sol": sol})
    out = _split_by_focus(rows, {"ZR034": base}, {"ZR034": "Z"}, 30.0, "focal", model=m, hold_f_images=0)
    by = {c["focus_count_median"]: c for c in out.values()}
    g = m["cameras"]["ZR034"]
    for foc, T, sol in ((300, -19.0, 500), (900, -11.0, 1500)):
        c = by[foc]
        f = np.sqrt(c["params"][0] * c["params"][1])
        assert f == pytest.approx(zcam_model_focal(g, foc, T, sol)[0])
        assert c["temperature_median_degC"] == pytest.approx(T) and c["sol_median"] == sol
        # pp moves with focus about the eye's median focus (600 here)
        assert c["params"][2] - 824.0 == pytest.approx(g["pp"]["cx_px_per_count"] * (foc - 600))
        assert c["params"][3] - 600.0 == pytest.approx(g["pp"]["cy_px_per_count"] * (foc - 600))
    assert by[600]["params"][2] == pytest.approx(824.0)
    assert "thermal" in by[900]["source"] and "trend" in by[900]["source"]
    assert by[300]["params"][2] < 824.0 < by[900]["params"][2]


def test_focus_model_fingerprint(tmp_path):
    import hashlib
    import json
    from mppp.paths import data_dir
    from mppp.sfm.project import ZCAM_FOCUS_MODEL, zcam_focus_model_fingerprint
    shipped = data_dir() / "cmods" / ZCAM_FOCUS_MODEL
    h = hashlib.sha256()                                   # over every zoom's file (name + content)
    h.update(shipped.name.encode())
    h.update(shipped.read_bytes())
    assert zcam_focus_model_fingerprint() == h.hexdigest()
    alt = tmp_path / "m.json"
    m = json.loads(shipped.read_text())
    m["cameras"]["ZL034"]["f0_px"] += 1
    alt.write_text(json.dumps(m))
    assert zcam_focus_model_fingerprint(alt) != zcam_focus_model_fingerprint()


def test_navcam_consensus_shipped_with_mppp():
    from mppp.sfm.project import NAVCAM_CONSENSUS_DIR, NAVCAM_FISHEYE_PATTERN, NAVCAM_RIG_FILE, camera_from_colmap_json
    for eye in ("NL", "NR"):
        f = NAVCAM_CONSENSUS_DIR / NAVCAM_FISHEYE_PATTERN.format(instrument=eye)
        cam = camera_from_colmap_json(f)
        assert cam["model"] == "THIN_PRISM_FISHEYE" and len(cam["params"]) == 12
        th = json.loads(f.read_text())["thermal"]
        # the 18 Oct 2026 pass-2 consensus (23 blocks): fx, fy per axis, ~38-41 and ~43-45 ppm/degC
        assert 37.0 < th["fx_ppm_per_degC"] < 41.5 and 43.0 < th["fy_ppm_per_degC"] < 46.0 and th["T0_degC"] == -20
        assert "ppm_per_degC" not in th                                          # the per-axis terms are the model
    assert (NAVCAM_CONSENSUS_DIR / NAVCAM_RIG_FILE).is_file()
    assert calib.navcam_fingerprint(NAVCAM_CONSENSUS_DIR)
    # the parameters of the 18 Oct 2026 pass-2 consensus of 23 blocks (the rig's fitted translation flag kept)
    nl = json.loads((NAVCAM_CONSENSUS_DIR / "M2020_NL_fisheye_tangential.json").read_text())
    rig = json.loads((NAVCAM_CONSENSUS_DIR / NAVCAM_RIG_FILE).read_text())
    assert nl["params"][0] == pytest.approx(2956.1655944938043) and nl["params"][9] == pytest.approx(-0.006617090516290966)
    assert rig["translation"] == "fitted" and rig["baseline_m"] == pytest.approx(0.4240102397823094)
    assert rig["drift"]["yaw_mdeg_per_sol"] == pytest.approx(0.0025605370970143183)


def test_navcam_distortion_hold_and_held_zeros():
    """The Navcam distortion is one set per eye for every sol and temperature (held in every block); the form's
    sx1, sy1 are fixed at 0 in either case; p1 has no setting of its own (TANGENTIAL)."""
    from mppp.paths import cmods_dir
    from mppp.sfm.project import camera_from_colmap_json, navcam_distortion_terms
    cam = camera_from_colmap_json(cmods_dir() / "M2020_NL_fisheye_tangential.json")
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
    held = navcam_distortion_terms(cam)                                    # defaults "hold"
    assert set(held["fixed_params"]) == set(names[4:]) and not held["free_params"]
    assert held["params"] == cam["params"] and held["distortion_fit"] == "hold"
    ref = navcam_distortion_terms(cam, "refine")
    assert ref["fixed_params"] == ["sx1", "sy1"] and ref["params"] == cam["params"]
    pp = navcam_distortion_terms(cam, "hold", pp="hold")
    assert {"cx", "cy"} <= set(pp["fixed_params"]) and "principal point held" in pp["source"]
    with pytest.raises(TypeError):
        navcam_distortion_terms(cam, "refine", k4="zero")
    with pytest.raises(ValueError):
        navcam_distortion_terms(cam, "free")
    with pytest.raises(ValueError):
        from mppp.sfm.project import SfmProject
        SfmProject.create([], "x", "y", navcam_distortion_fit="free")


def test_zero_terms_apply_to_the_fisheye_model(tmp_path):
    """TANGENTIAL = "zero" zeroes p1, p2 of a THIN_PRISM_FISHEYE start camera."""
    from mppp.paths import cmods_dir
    from mppp.sfm.project import camera_from_colmap_json
    cam = camera_from_colmap_json(cmods_dir() / "M2020_NL_fisheye_tangential.json", ("p1", "p2"))
    assert cam["params"][6] == cam["params"][7] == 0.0


def test_bundle_adjust_holds_the_navcam_distortion(tmp_path):
    """With the distortion held only f and the principal point change in the adjustment."""
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm.project import navcam_distortion_terms
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    for k in ("NL", "NR"):
        proj.cameras[k] = navcam_distortion_terms(proj.cameras[k], "hold")
    rec = _build_rec(proj, truth, P, cams, rigT, noise, rng)[0]
    before = {cid: list(c.params) for cid, c in rec.cameras.items()}
    bundle_adjust(rec, proj, refine_rig=False, verbose=False)
    for cid, c in rec.cameras.items():
        assert np.allclose(c.params[4:], before[cid][4:])                  # distortion unchanged


def test_zcam_focus_models_per_zoom(tmp_path):
    """One focus-model file per zoom, merged; a model with distortion and pp0 sets the eye camera; the
    fingerprint covers every zoom's file."""
    from mppp.sfm.project import zcam_focus_model, zcam_focus_model_files, zcam_focus_model_fingerprint, _split_by_focus
    from mppp.sfm.start_cameras import zcam_eye_camera
    fp0 = zcam_focus_model_fingerprint()
    m48 = {"cameras": {"ZL048": {"f0_px": 6600.0, "reference_focus": 500.0, "slope_px_per_count": 0.1, "aspect": 1.0,
                                 "focus_range": [0, 1000],
                                 "distortion": {"names": ["k1", "k2", "p1", "p2", "k3"], "params": [-0.1, 0.2, 0, 0, 0]},
                                 "pp0_px": [820.0, 610.0], "pp": {"cx_px_per_count": 0.01, "cy_px_per_count": 0.0}}},
           "rig": {"R_sensor_from_ref": np.eye(3).tolist(), "note": "test"}}
    (tmp_path / "M2020_ZCAM048_focus_model.json").write_text(json.dumps(m48))
    files = zcam_focus_model_files(tmp_path)
    assert set(files) == {34, 48} and files[48].parent == tmp_path
    m = zcam_focus_model(tmp_path)
    assert {"ZL034", "ZR034", "ZL048"} <= set(m["cameras"]) and m["zooms"][48]["rig"]["note"] == "test"
    assert m["state"] == "backlash"                          # the 34 mm file's top level
    assert zcam_focus_model_fingerprint(None, tmp_path) != fp0
    cam = zcam_eye_camera("ZL048", [("FULL_OPENCV", np.array([6500.0, 6500.0, 800, 600, -0.05, 0.1, 0.001, 0, 0, 0, 0, 0]))],
                          (1648, 1200), ("p1", "p2"), m["cameras"]["ZL048"])
    base = dict(cam)
    assert cam["params"][4:6] == [-0.1, 0.2] and cam["params"][2:4] == [820.0, 610.0] and cam["pp_at_focus"] == 500.0
    rows = [{"camera_group": "ZL048", "focus_count": 700, "label_f_px": 6500.0} for _ in range(4)]
    out = _split_by_focus(rows, {"ZL048": cam}, {"ZL048": "Z"}, 30.0, "model", model=m)
    c = next(iter(out.values()))
    assert abs(c["params"][0] - (6600.0 + 0.1 * 200)) < 1e-6 and {"fx", "fy"} <= set(c["fixed_params"])
    assert abs(c["params"][2] - (820.0 + 0.01 * 200)) < 1e-9                # pp linear in focus about pp0's focus
    rows63 = [{"camera_group": "ZL063", "focus_count": 700, "label_f_px": 6500.0} for _ in range(4)]
    lab = _split_by_focus(rows63, {"ZL063": dict(base)}, {"ZL063": "Z"}, 30.0, "focal", model=m)   # no 63 mm model
    assert abs(next(iter(lab.values()))["params"][0] - 6500.0) < 1e-6
