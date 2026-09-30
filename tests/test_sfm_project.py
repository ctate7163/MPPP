"""COLMAP project set-up, start cameras, focus bins and models (mppp.sfm.project). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import numpy as np
import pytest
from mppp.paths import data_dir  # noqa: E402
from pathlib import Path
import json
import math  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from mppp.sfm import navcal as NC  # noqa: E402
from mppp.sfm.project import start_rig_rotation  # noqa: E402
import copy


pycolmap = pytest.importorskip("pycolmap")


DATA_DIR = data_dir()


def _metashape_project(xyz, c):
    """Metashape frame model, corner pixel origin."""
    x, y = xyz[0] / xyz[2], xyz[1] / xyz[2]
    r2 = x * x + y * y
    rad = 1 + c["k1"] * r2 + c["k2"] * r2 ** 2 + c["k3"] * r2 ** 3
    xp, yp = x * rad, y * rad
    return np.array([c["width"] * 0.5 + c["cx"] + xp * c["f"], c["height"] * 0.5 + c["cy"] + yp * c["f"]])


def test_camera_from_metashape_xml_matches_metashape_projection():
    from mppp.sfm.project import camera_from_metashape_xml, read_metashape_calibration
    xml = DATA_DIR / "cmods/M2020_NL0_frame.xml"
    cam_d = camera_from_metashape_xml(xml, zero_terms=("p1", "p2", "b1", "b2"))    # v0p20 default keeps p1, p2
    assert cam_d["model"] == "FULL_OPENCV" and (cam_d["width"], cam_d["height"]) == (5120, 3840)
    p = cam_d["params"]
    assert p[0] == p[1] and p[6] == p[7] == 0.0 and p[9:] == [0.0, 0.0, 0.0]        # b1, p1, p2 zeroed; k4-k6 = 0
    c = read_metashape_calibration(xml)
    cam = pycolmap.Camera(model=cam_d["model"], width=5120, height=3840, params=p)
    for X in ([0.3, -0.2, 1.0], [-0.6, 0.4, 1.0], [0.0, 0.0, 1.0]):
        X = np.array(X)
        assert np.allclose(cam.img_from_cam(X), _metashape_project(X, c), atol=1e-6)


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
                             navcam_distortion_fit="refine")          # v0p50: the default holds the distortion   # <= 0.14.3
    assert set(proj.cameras) == {"NL", "ZL034"}
    z = proj.cameras["ZL034"]
    assert (z["model"], z["width"], z["height"]) == ("FULL_OPENCV", 1648, 1200) and "label" in z["source"]
    meta = [m for m in man["images"] if m["filename"]["family"] == "Z"][0]
    K = np.asarray(meta["intrinsics"]["K"])
    assert np.isclose(z["params"][0], K[0, 0]) and np.isclose(z["params"][2], K[0, 2] + 0.5)      # corner origin
    assert np.isclose(z["params"][4], meta["intrinsics"]["dist_opencv"]["k1"])
    assert z["params"][6] == z["params"][7] == 0.0
    assert {r["instrument"] for r in proj.images} == {"NL", "ZL034"} and proj.rig == {}
    assert camera_key({"family": "Z", "camera_group": "ZR034", "instrument": "ZR"}) == "ZR034"
    px = SfmProject.create(man["images"], out, tmp_path / "q", link=False, zcam_intrinsics="xml", zcam_focus_bin=None)
    assert "ZL034_frame.xml" in px.cameras["ZL034"]["source"] and px.cameras["ZL034"]["params"][0] == 4720
    with pytest.raises(ValueError):
        SfmProject.create(man["images"], out, tmp_path / "r", zcam_intrinsics="guess")


def test_rig_family_for_mastcamz_pairs_with_shared_clock():
    from mppp.sfm.project import _rig_from_pairs
    R = np.eye(3).tolist()
    imgs = [{"instrument": k, "eye": k[1], "sclk_key": "1", "prior_R_w2c": R, "prior_C": c}
            for k, c in (("ZL034", [0, 0, 0]), ("ZR034", [0.242, 0, 0]), ("NL", [0, 0, 0]))]
    rig = _rig_from_pairs(imgs)
    assert list(rig) == ["Z034"] and (rig["Z034"]["ref"], rig["Z034"]["sensor"]) == ("ZL034", "ZR034")
    assert abs(rig["Z034"]["baseline_m"] - 0.242) < 1e-9


def test_xml_tangential_terms_kept_when_not_zeroed():
    from mppp.paths import data_dir
    from mppp.sfm.project import camera_from_metashape_xml, read_metashape_calibration
    xml = data_dir() / "cmods/M2020_NL0_frame.xml"
    c = read_metashape_calibration(xml)
    p = camera_from_metashape_xml(xml, zero_terms=("b1", "b2"))["params"]
    assert p[6] == pytest.approx(c["p2"]) and p[7] == pytest.approx(c["p1"])     # OpenCV p1 = Metashape P2
    assert abs(p[6]) > 1e-4 and p[0] == p[1]                                      # b1 still zeroed


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
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_intrinsics="label")
    zmeta = [m for m in man["images"] if m["filename"]["family"] == "Z"][0]
    fc = float(zmeta["focus_position_count"])
    key = f"ZL034_F{int(round(fc)):05d}"
    assert set(proj.cameras) == {"NL", key}
    z = proj.cameras[key]
    assert z["group"] == "ZL034" and z["focus_count_median"] == fc and z["n_images"] == 1
    assert z["fixed_params"] == list(ZCAM_BIN_HELD) and "focus bin" in z["source"]
    r = [r for r in proj.images if r["camera_group"] == "ZL034"][0]
    assert r["instrument"] == key and r["focus_count"] == fc and abs(r["label_f_px"] - z["params"][0]) < 1e-9
    assert proj.settings["zcam_focus_bin"] == 30.0 and proj.settings["zcam_bin_refine"] == "focal"
    pa = SfmProject.create(man["images"], out, tmp_path / "q", link=False, zcam_bin_refine="all",
                           zcam_intrinsics="label")
    assert pa.cameras[key]["fixed_params"] == []
    # v0p22 default: f from the focus model; a one-image bin holds it
    pm = SfmProject.create(man["images"], out, tmp_path / "m", link=False)
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


def test_project_default_is_rational_and_scope_is_checked_first(processed_pair, tmp_path):  # noqa: F811
    from mppp.sfm.project import SfmProject
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None,
                             navcam_distortion_fit="refine")          # v0p50: the default holds the distortion
    nl = proj.cameras["NL"]
    assert nl["free_params"] == ["k4"] and "rational" in nl["source"] and proj.settings["navcam_distortion"] == "rational"
    assert abs(nl["params"][6]) > 1e-5                                      # p1/p2 not zeroed by default any more
    poly = SfmProject.create(man["images"], out, tmp_path / "q", link=False, zcam_focus_bin=None,
                             navcam_distortion="polynomial")
    assert poly.cameras["NL"]["params"][9] == 0.0 and not poly.cameras["NL"].get("free_params")
    # out of scope: rejected before any file is linked into the project
    bad = json.loads(json.dumps(man["images"], default=str))
    bad[0]["filename"]["camera_code"] = "FLF"
    with pytest.raises(ValueError, match="outside MPPP's scope"):
        SfmProject.create(bad, out, tmp_path / "r", link=False, zcam_focus_bin=None)
    assert not any((tmp_path / "r" / "images").glob("*"))
    z = json.loads(json.dumps(man["images"], default=str))
    for m in z:
        if m["filename"]["family"] == "Z":
            m["filename"]["zoom_mm"] = 110
    with pytest.raises(ValueError, match="outside MPPP's scope"):
        SfmProject.create(z, out, tmp_path / "s", link=False)
    und = json.loads(json.dumps(man["images"], default=str))
    und[0]["undistorted"] = True
    with pytest.raises(ValueError, match="undistort"):
        SfmProject.create(und, out, tmp_path / "t", link=False)
    with pytest.raises(ValueError):
        SfmProject.create(man["images"], out, tmp_path / "u", navcam_distortion="fisheye")


def test_refresh_images_updates_copies(processed_pair, tmp_path):  # noqa: F811
    import os
    from mppp.sfm.project import SfmProject
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None,
                             navcam_distortion_fit="refine")          # v0p50: the default holds the distortion
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
    assert ZCAM_HOLD_F_IMAGES == 2 and g["focus_range"][0] <= -400 and g["focus_range"][1] == 1300   # v0p42 range
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
    # outside the fitted range: the bin's label f scaled by the refined/label ratio, not an extrapolation
    f_out = np.sqrt(by[-2600]["params"][0] * by[-2600]["params"][1])
    assert abs(f_out - 4600.0 * g["f0_px"] / g["label_f0_px"]) < 0.01 and "outside" in by[-2600]["source"]
    assert "fx" not in by[-2600]["fixed_params"]
    lab = _split_by_focus(rows, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal")      # without a model: label f
    assert {round(c["params"][0]) for c in lab.values()} == {4680, 4700, 4600}


def test_shipped_rig_and_rational_cameras_agree():
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
        c = json.loads((d / f"M2020_N{e}_rational.json").read_text())
        p = c["params"] if "params" in c else c["camera"]["params"]
        assert 2900 < p[0] < 3000 and 0.2 < p[9] < 0.7


def test_project_create_accepts_navcam_cameras(tmp_path, monkeypatch):
    from mppp.sfm import project as P
    import inspect
    sig = inspect.signature(P.SfmProject.create)
    assert "navcam_cameras" in sig.parameters
    src = inspect.getsource(P.SfmProject.create)
    assert "NAVCAM_RATIONAL_PATTERN" in src and "NAVCAM_FISHEYE_PATTERN" in src and '"navcam_cameras"' in src


def test_shipped_cameras_are_the_five_site_consensus():
    from mppp.paths import data_dir
    for g in ("NL", "NR"):
        d = json.loads((data_dir() / "cmods" / f"M2020_{g}_rational.json").read_text())
        assert "five" in d["source"] and len(d["verification"]["per_scape"]) == 5
        assert all(r["rms_px"] < 0.5 for r in d["verification"]["per_scape"])
    rig = json.loads((data_dir() / "cmods" / "history" / "v0p22_consensus" / "M2020_N_rig.json").read_text())   # v0p50
    assert len(rig["per_solution_rotvec_rad"]) == 5 and abs(rig["baseline_m"] - 0.4244) < 1e-3


def test_start_rig_rotation_temperature_and_drift():
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import start_rig_rotation
    R0 = Rotation.from_rotvec([0.0014, 0.0011, 0.0002]).as_matrix()
    sh = {"R_sensor_from_ref": R0.tolist(), "thermal": {"yaw_mdeg_per_degC": -1.0, "pitch_mdeg_per_degC": 0.0, "T0_degC": -19.0},
          "drift": {"sol0": 1000.0, "pitch_mdeg_per_sol": 0.005, "yaw_mdeg_per_sol": 0.0, "roll_mdeg_per_sol": -0.006}}
    R, applied = start_rig_rotation(sh, -9.0, 1400.0)
    rv = np.degrees(Rotation.from_matrix(R @ R0.T).as_rotvec()) * 1e3
    assert np.allclose(rv, [2.0, -10.0, -2.4], atol=0.01)                # pitch +0.005x400, yaw -1x10, roll -0.006x400
    assert applied["thermal"]["T_median_degC"] == -9.0 and applied["drift"]["sol_median"] == 1400.0
    R1, a1 = start_rig_rotation({"R_sensor_from_ref": R0.tolist()}, -9.0, 1400.0)
    assert np.allclose(R1, R0) and a1 == {}


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


def test_project_create_with_fisheye_tangential_start_cameras(tmp_path):
    from conftest import NLF, needs_data
    if not NLF.is_file():
        pytest.skip("example IMGs not present")
    import mppp
    from mppp.sfm.navcal import write_joint_cameras
    from mppp.sfm.project import SfmProject
    out = tmp_path / "proc"
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": False}})
    man = mppp.process_images([NLF], out, cfg, mppp.load_waypoints(), progress=False)
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
    q = [2956.3, 2956.4, 2591.8, 1944.2, 0.0451, -0.0100, 2.5e-4, 3.6e-4, 0.0030, -0.0062, 0.0, 0.0]
    cam = {"model": "THIN_PRISM_FISHEYE", "width": 5120, "height": 3840, "params": dict(zip(names, q))}
    joint = {"NL": cam, "NR": cam, "T0_degC": -19.1, "ppm_per_degC": 38.6, "final": {"observations": 1}, "blocks": {},
             "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0]}
    write_joint_cameras(joint, tmp_path / "joint")
    with pytest.raises(ValueError, match="needs navcam_cameras"):
        SfmProject.create(man["images"], out, tmp_path / "p0", link=False, navcam_distortion="fisheye_tangential")
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, navcam_distortion="fisheye_tangential",
                             navcam_cameras=tmp_path / "joint")
    assert proj.cameras["NL"]["model"] == "THIN_PRISM_FISHEYE"
    assert abs(proj.cameras["NL"]["params"][4] - 0.0451) < 1e-9


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


def test_start_rig_rotation_applies_the_hinge():
    st = _synthetic()
    m = NC.rig_drift_model(st, knot_sol=300.0)
    shipped = {"R_sensor_from_ref": np.eye(3).tolist(), "drift": m}
    for sol in (63.0, 1084.0, 1970.0):
        R, applied = start_rig_rotation(shipped, None, sol)
        ang = NC.rig_angles(R)
        exp = NC.drift_offset_mdeg(m, sol)
        for a in ("pitch", "yaw", "roll"):
            assert ang[f"{a}_mdeg"] == pytest.approx(exp[f"{a}_mdeg"], abs=1e-6)
        assert isinstance(applied["drift"]["offset_mdeg"]["pitch"], float)
    # a v0p35 rig file (rates only) still works
    R, _ = start_rig_rotation({"R_sensor_from_ref": np.eye(3).tolist(),
                               "drift": {"sol0": 1000.0, "pitch_mdeg_per_sol": 0.005}}, None, 1200.0)
    assert NC.rig_angles(R)["pitch_mdeg"] == pytest.approx(1.0, abs=1e-6)


def _model():
    from mppp.sfm.project import zcam_focus_model
    return zcam_focus_model()


def test_shipped_focus_model_layout():
    m = _model()
    for g in ("ZL034", "ZR034"):
        c = m["cameras"][g]
        assert c["reference_focus"] == 600.0
        assert 0.05 < c["slope_px_per_count"] < 0.07                 # refit (was 0.046 / 0.048)
        assert 1.006 < c["f0_px"] / c["label_f0_px"] < 1.012          # backlash state, ~0.9 % above the label
        assert c["thermal"]["sensor"] == "HEAD_FPA" and c["thermal"]["f_px_per_degC"] == 0.0
        assert c["trend"]["sol_range"] == [461, 991] and c["trend"]["sol0"] == 700.0
    assert m["cameras"]["ZL034"]["pp"]["cx_px_per_count"] == 0.0
    assert 0.001 < m["cameras"]["ZR034"]["pp"]["cx_px_per_count"] < 0.004
    assert 0.001 < m["cameras"]["ZR034"]["pp"]["cy_px_per_count"] < 0.004


def test_model_focal_terms_and_sol_clamp():
    from mppp.sfm.project import zcam_model_focal
    g = _model()["cameras"]["ZR034"]
    f, t = zcam_model_focal(g, 600.0, T=-15.0, sol=700.0)
    assert f == pytest.approx(g["f0_px"]) and t["thermal"] == 0.0 and t["trend"] == 0.0
    f1, _ = zcam_model_focal(g, 1100.0)
    assert f1 - g["f0_px"] == pytest.approx(500 * g["slope_px_per_count"])
    # the trend does not extrapolate beyond the fitted sols
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
    assert zcam_focus_model_fingerprint() == hashlib.sha256(shipped.read_bytes()).hexdigest()
    alt = tmp_path / "m.json"
    m = json.loads(shipped.read_text())
    m["cameras"]["ZL034"]["f0_px"] += 1
    alt.write_text(json.dumps(m))
    assert zcam_focus_model_fingerprint(alt) != zcam_focus_model_fingerprint()


def test_navcam_consensus_shipped_with_mppp():
    import json
    from mppp.sfm.project import (NAVCAM_PACKAGE_CONSENSUS_DIR as NAVCAM_CONSENSUS_DIR, NAVCAM_FISHEYE_PATTERN, NAVCAM_RIG_FILE,
                                  camera_from_colmap_json, navcam_cameras_fingerprint)
    for eye in ("NL", "NR"):
        f = NAVCAM_CONSENSUS_DIR / NAVCAM_FISHEYE_PATTERN.format(instrument=eye)
        cam = camera_from_colmap_json(f)
        assert cam["model"] == "THIN_PRISM_FISHEYE" and len(cam["params"]) == 12
        th = json.loads(f.read_text())["thermal"]
        assert abs(th["ppm_per_degC"] - 38.1) < 0.5 and th["T0_degC"] == -20
    assert (NAVCAM_CONSENSUS_DIR / NAVCAM_RIG_FILE).is_file()
    assert navcam_cameras_fingerprint(NAVCAM_CONSENSUS_DIR)
    # the v0p41 joint (camera_analysis/navcal_v0p41/navcam_joint), byte for byte
    import hashlib
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in NAVCAM_CONSENSUS_DIR.glob("*.json")
           if "fisheye" in p.name or "rig" in p.name}
    assert sha == {"M2020_NL_fisheye_tangential.json": "15389f18ab97", "M2020_NR_fisheye_tangential.json": "8c114d80b388",
                   "M2020_N_rig.json": "949b9c26aca7"}


def test_navcam_distortion_hold_and_zero_terms():
    """v0p50: the Navcam distortion is one set per eye for every sol and temperature (held in every block);
    NAVCAM_K4 / NAVCAM_P1 = "zero" set that term to 0 and hold it."""
    from mppp.paths import cmods_dir
    from mppp.sfm.project import camera_from_colmap_json, navcam_distortion_terms
    cam = camera_from_colmap_json(cmods_dir() / "M2020_NL_fisheye_tangential.json")
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
    held = navcam_distortion_terms(cam)                                    # default "hold"
    assert set(held["fixed_params"]) == set(names[4:]) and not held["free_params"]
    assert held["params"] == cam["params"] and held["distortion_fit"] == "hold"
    ref = navcam_distortion_terms(cam, "refine")
    assert ref["fixed_params"] == [] and ref["params"] == cam["params"]
    z = navcam_distortion_terms(cam, "refine", k4="zero", p1="zero")
    assert z["params"][9] == 0.0 and z["params"][6] == 0.0 and set(z["fixed_params"]) == {"k4", "p1"}
    assert cam["params"][9] != 0.0 and z["zeroed_terms"] == ["k4", "p1"]
    with pytest.raises(ValueError):
        from mppp.sfm.project import SfmProject
        SfmProject.create([], "x", "y", navcam_distortion_fit="free")


def test_zero_terms_apply_to_the_fisheye_model(tmp_path):
    """v0p50: TANGENTIAL = "zero" zeroes p1, p2 of a THIN_PRISM_FISHEYE start camera too (was OPENCV only)."""
    from mppp.paths import cmods_dir
    from mppp.sfm.project import camera_from_colmap_json
    cam = camera_from_colmap_json(cmods_dir() / "M2020_NL_fisheye_tangential.json", ("p1", "p2"))
    assert cam["params"][6] == cam["params"][7] == 0.0


def test_bundle_adjust_holds_the_navcam_distortion(tmp_path):
    """v0p50: with the distortion held only f and the principal point change in the adjustment."""
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


def test_rig_without_yaw_keeps_pitch_and_roll():
    from scipy.spatial.transform import Rotation
    from mppp.paths import cmods_dir
    from mppp.sfm.project import rig_without_yaw, rig_yaw_mdeg
    R = np.asarray(json.loads((cmods_dir() / "M2020_N_rig.json").read_text())["R_sensor_from_ref"], float)
    assert abs(rig_yaw_mdeg(R)) > 1.0
    R0 = rig_without_yaw(R)
    rv, rv0 = Rotation.from_matrix(R).as_rotvec(), Rotation.from_matrix(R0).as_rotvec()
    assert abs(rv0[1]) < 1e-15 and np.allclose(rv0[[0, 2]], rv[[0, 2]])
    assert abs(Rotation.from_matrix(R0).as_quat()[1]) < 1e-15


def test_thermal_rig_slopes_drop_the_yaw_when_it_is_held():
    from types import SimpleNamespace
    from mppp.sfm.thermal import rig_slopes_for_project
    th = {"yaw_mdeg_per_degC": 0.4, "pitch_mdeg_per_degC": 0.1, "T0_degC": -20}
    p = SimpleNamespace(settings={"navcam_cameras": {"rig": {"thermal": th}}})
    assert rig_slopes_for_project(p)["yaw_mdeg_per_degC"] == 0.4
    p.settings["navcam_rig_yaw"] = "zero"
    s = rig_slopes_for_project(p)
    assert s["yaw_mdeg_per_degC"] == 0.0 and s["pitch_mdeg_per_degC"] == 0.1
