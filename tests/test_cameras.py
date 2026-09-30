"""File names, PDS labels, CAHV/CAHVOR(E) camera models, the Metashape and COLMAP camera conventions (mppp.filenames, labels, cmod, camera, colmap). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from pathlib import Path


def _synthetic_cahv(fx=1200.0, fy=1190.0, cx=640.3, cy=479.1, seed=0):
    rng = np.random.default_rng(seed)
    Rm = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()      # rows: cam axes in frame
    C = rng.normal(size=3)
    A = Rm[2]
    H = fx * Rm[0] + cx * A
    V = fy * Rm[1] + cy * A
    return C, A, H, V, Rm


def test_parse_zcam():
    from mppp import parse_filename
    fn = parse_filename("ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG")
    assert (fn.instrument, fn.family, fn.eye, fn.filter) == ("ZL", "Z", "L", "0")
    assert (fn.sol, fn.site, fn.drive) == (709, 33, 2864)
    assert fn.sclk == pytest.approx(729888971.069)
    assert fn.zoom_mm == 34 and fn.camera_group == "ZL034"
    assert fn.downsample_scale == 1.0 and fn.version == 1 and fn.product_type == "RAD"
    assert fn.stereo_partner_stem == "ZR0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01"


def test_parse_navcam_and_downsample():
    from mppp import parse_filename
    fn = parse_filename("NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ01.IMG")
    assert fn.camera_code == "NLF" and fn.camera_group == "NL0" and fn.zoom_mm is None
    half = parse_filename("NRF_0709_0729883381_848RAD_N0332864SAPP00601_0A01LLJ02.IMG")
    assert half.downsample_scale == 0.5 and half.camera_group == "NR1" and half.version == 2


def test_parse_rejects_garbage():
    from mppp import parse_filename
    with pytest.raises(ValueError):
        parse_filename("not_a_pds_name.IMG")


def test_cahv_decomposition_round_trip():
    from mppp.camera import CAHVOR
    C, A, H, V, Rm = _synthetic_cahv()
    cam = CAHVOR(C, A, H, V)
    intr, R = cam.decompose(1280, 960)
    assert np.allclose(R, Rm, atol=1e-10)
    assert intr.K[0, 0] == pytest.approx(1200) and intr.K[1, 1] == pytest.approx(1190)
    assert abs(intr.K[0, 1]) < 1e-9
    assert (intr.cx, intr.cy) == (pytest.approx(640.3), pytest.approx(479.1))
    # pinhole projection == CAHV projection for random points in front of the camera
    X = C + (Rm.T @ (np.random.default_rng(1).uniform([-1, -1, 2], [1, 1, 9], (50, 3))).T).T
    xc = (R @ (X - C).T).T
    uv = (intr.K @ (xc / xc[:, 2:3]).T).T[:, :2]
    assert np.allclose(uv, cam.project_cahv(X), atol=1e-8)


def test_padding_shifts_principal_point_only():
    from mppp.camera import CAHVOR
    intr, _ = CAHVOR(*_synthetic_cahv()[:4]).decompose(1280, 960)
    p = intr.shifted(100, 50, 1480, 1060)
    assert (p.cx - intr.cx, p.cy - intr.cy) == (pytest.approx(100), pytest.approx(50))
    assert p.fx == intr.fx and (p.width, p.height) == (1480, 1060)
    assert np.allclose(p.K_corner_origin[:2, 2], p.K[:2, 2] + 0.5)


def test_metashape_xml_conventions(tmp_path):
    from mppp.camera import intrinsics_from_metashape_xml
    x = tmp_path / "c.xml"
    x.write_text("<calibration><projection>frame</projection><width>2560</width><height>1920</height>"
                 "<f>1475</f><cx>17</cx><cy>11</cy><b1>0.5</b1><k1>-0.27</k1><k3>-0.02</k3>"
                 "<p1>1e-4</p1><p2>2e-4</p2></calibration>")
    i = intrinsics_from_metashape_xml(x, scale=2.0)
    assert (i.width, i.height) == (5120, 3840)
    assert i.fy == pytest.approx(2950) and i.fx == pytest.approx(2951)
    assert i.cx == pytest.approx(2560 + 34 - 0.5) and i.cy == pytest.approx(1920 + 22 - 0.5)
    assert i.K_corner_origin[0, 2] == pytest.approx(2560 + 34)       # Metashape/COLMAP convention restored
    assert (i.dist["p1"], i.dist["p2"]) == (2e-4, 1e-4)              # Metashape P1/P2 swapped vs OpenCV
    assert i.dist["k1"] == -0.27 and i.dist["k3"] == -0.02           # distortion is scale free


def test_pose_level_camera_looking_north():
    from mppp.camera import pose_from_label
    # rover-nav == site (identity quaternion); camera looks north (+x NED), x right = east, y down
    R_cam = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    pose = pose_from_label(R_cam, [1.0, 2.0, -3.0], [1, 0, 0, 0], [10.0, 20.0, -5.0], "t", "t")
    assert np.allclose(pose.C, [22.0, 11.0, 8.0])                    # E, N, U
    az, el = pose.boresight_az_el_deg
    assert az == pytest.approx(0, abs=1e-9) and el == pytest.approx(0, abs=1e-9)
    assert np.allclose(pose.R_w2c @ [0, 1, 0], [0, 0, 1])            # world north -> camera +z
    assert np.allclose(pose.R_w2c @ [1, 0, 0], [1, 0, 0])            # world east  -> camera +x
    assert np.allclose(pose.R_w2c @ [0, 0, 1], [0, -1, 0])           # world up    -> camera -y
    assert np.linalg.det(pose.R_w2c) == pytest.approx(1)


def test_ypr_matches_legacy_formula():
    from mppp.camera import pose_from_label
    rng = np.random.default_rng(3)
    for _ in range(20):
        rot_cam = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
        q = Rotation.from_rotvec(rng.normal(size=3))
        q_wxyz = np.roll(q.as_quat(), 1)
        pose = pose_from_label(rot_cam, rng.normal(size=3), q_wxyz, rng.normal(size=3), "t", "t")
        # --- verbatim legacy (image.py: cmod_for_landing_frame + find_ypr_from_R_ref)
        R_cam_site = q.apply(rot_cam)
        R_ref = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) @ R_cam_site
        Q = Rotation.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        ypr = (Q.inv() * Rotation.from_matrix(R_ref)).inv().as_euler("ZYX", degrees=True)
        if ypr[0] < 0:
            ypr[0] += 360
        assert np.allclose(pose.metashape_ypr_deg(), ypr, atol=1e-9)


def test_colmap_text_model_round_trip(tmp_path):
    from mppp.colmap import write_text_model
    R = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    C = np.array([105.0, 203.0, 7.0])
    intr = {"width": 100, "height": 80, "K": [[500, 0, 49.5], [0, 501, 39.5], [0, 0, 1]],
            "dist_opencv": {"k1": -0.1, "k2": 0.01, "k3": -0.002, "k4": 0.0, "p1": 1e-4, "p2": 2e-4}}
    metas = [{"camera_group": "NL0", "intrinsics": intr, "source_product": f"NLF_{i}.IMG",
              "filename": {"camera_code": c}, "pose": {"R_world_to_cam": R.tolist(), "C_enu_m": C.tolist()}}
             for i, c in enumerate(["NLF", "NRF"])]
    metas[1]["camera_group"] = "NR0"
    s = write_text_model(metas, tmp_path, offset=np.array([100.0, 200.0, 0.0]))
    cams = [l for l in (tmp_path / "cameras.txt").read_text().splitlines() if not l.startswith("#")]
    assert cams[0].split()[:4] == ["1", "FULL_OPENCV", "100", "80"]
    assert float(cams[0].split()[6]) == 50.0                      # cx + 0.5 (corner origin)
    assert float(cams[0].split()[12]) == -0.002                   # k3 kept
    img = [l for l in (tmp_path / "images.txt").read_text().splitlines() if l and not l.startswith("#")][0].split()
    qw, qx, qy, qz, tx, ty, tz = map(float, img[1:8])
    R_back = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    assert np.allclose(R_back, R, atol=1e-9)
    assert np.allclose(-R_back.T @ [tx, ty, tz], C - [100, 200, 0], atol=1e-6)
    assert img[9] == "NLF_0.png" and s["rigs"] == 1
    assert json.loads((tmp_path / "rig_config.json").read_text())[0]["cameras"][0] == {"image_prefix": "NLF", "ref_sensor": True}


def test_one_projection_for_everyone():
    import pycolmap
    from mppp.colmap import project_camera, scale_camera_params
    from mppp.error.colmap import project_camera as pe
    assert pe is project_camera
    p = [2950, 2951, 2560, 1920, -0.27, 0.1, 0, 0, -0.02, 0, 0, 0]
    x = np.array([[0.3, -0.2, 1.0], [0.01, 0.02, 2.0]])
    cam = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=p)
    assert np.allclose(project_camera("FULL_OPENCV", p, x), cam.img_from_cam(x), atol=1e-6)
    assert project_camera("FULL_OPENCV", p, x[0]).shape == (2,)
    q = scale_camera_params("FULL_OPENCV", p, 0.25)
    assert np.allclose(q[:4], np.array(p[:4]) / 4) and np.allclose(q[4:], p[4:])


def test_unproject_inverts_project():
    from mppp.colmap import project_camera, unproject_camera
    p = [2951, 2951, 2594, 1942, -0.27, 0.10, 1.7e-4, 1.7e-4, -0.02, 0, 0, 0]
    xy = np.random.default_rng(1).uniform(-0.8, 0.8, (500, 2))
    uv = project_camera("FULL_OPENCV", p, np.c_[xy, np.ones(len(xy))])
    assert np.abs(unproject_camera("FULL_OPENCV", p, uv) - xy).max() < 1e-8
    assert np.allclose(unproject_camera("SIMPLE_RADIAL", [1000, 500, 400, 0.0], [[600, 400]]), [[0.1, 0.0]])


def _rational(eye="L", zero=()):
    from mppp.paths import data_dir
    from mppp.sfm.project import camera_from_colmap_json
    return camera_from_colmap_json(data_dir() / f"cmods/M2020_N{eye}_rational.json", zero)


def test_rational_navcam_camera_is_invertible_over_the_whole_frame():
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.colmap import project_camera, unproject_camera
    g = np.stack(np.meshgrid(np.linspace(0.5, 5119.5, 41), np.linspace(0.5, 3839.5, 31)), -1).reshape(-1, 2)
    for eye in "LR":
        c = _rational(eye)
        assert c["model"] == "FULL_OPENCV" and (c["width"], c["height"]) == (5120, 3840)
        assert c["free_params"] == ["k4"] and c["params"][9] > 0.3 and c["distortion"] == "rational"
        assert abs(c["params"][6]) > 1e-5                                   # fitted p1/p2 kept by default
        cam = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=c["params"])
        xy = np.asarray(cam.cam_from_img(g))
        assert np.isfinite(xy).all()                                        # COLMAP can undistort every pixel
        corner_deg = np.degrees(np.arctan(np.hypot(*xy[0])))
        assert 55 < corner_deg < 65
        mine = unproject_camera("FULL_OPENCV", c["params"], g)
        assert np.abs(mine - xy).max() < 1e-8
        back = project_camera("FULL_OPENCV", c["params"], np.c_[mine, np.ones(len(mine))])
        assert np.abs(back - g).max() < 1e-8
    # the Metashape three-term polynomial cannot be inverted in the corners
    from mppp.paths import data_dir
    from mppp.sfm.project import camera_from_metashape_xml
    poly = camera_from_metashape_xml(data_dir() / "cmods/M2020_NL0_frame.xml")
    xy = np.asarray(pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=poly["params"]).cam_from_img(g))
    assert 0 < np.isnan(xy[:, 0]).sum() < 0.05 * len(g)
    assert _rational("L", ("p1", "p2"))["params"][6] == 0.0              # TANGENTIAL = "zero" still possible


DATA = Path(__file__).parent / "data" / "m20"


NLF = DATA / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ01.IMG"


ZL0 = DATA / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"


def _label_model(path):
    from mppp.cmod import CameraModel
    from mppp.labels import label_get, read_pds
    L, _ = read_pds(path, load_image=False)
    w, h = int(label_get(L, "IMAGE.LINE_SAMPLES")), int(label_get(L, "IMAGE.LINES"))
    fs = float(label_get(L, "IMAGE.FIRST_LINE_SAMPLE", default=1))
    fl = float(label_get(L, "IMAGE.FIRST_LINE", default=1))
    return CameraModel.from_label(label_get(L, "GEOMETRIC_CAMERA_MODEL"), w, h), (fs - 1, fl - 1)


def _rational_v0p21():
    from mppp.paths import data_dir
    return json.loads((data_dir() / "cmods" / "M2020_NL_rational.json").read_text())


def test_navcam_label_is_cahvore_fisheye_and_matches_the_rational_model():
    from mppp.cmod import CameraModel, compare_to_colmap
    cm, (dx, dy) = _label_model(NLF)
    assert cm.kind == "CAHVORE" and cm.mtype == 2 and cm.linearity == 0.0
    full = cm.rescaled(1.0, dx, dy, 5120, 3840)                 # a full-resolution sub-frame
    nl = _rational_v0p21()
    d = compare_to_colmap(full, nl["model"], nl["params"], 5120, 3840, step=160)
    assert d["rms_px"] < 4 and d["corner_rms_px"] < 7 and d["rotation_deg"] < 0.2
    # read as a perspective CAHVOR, the same numbers are hundreds of pixels off
    wrong = CameraModel(full.C, full.A, full.H, full.V, full.O, full.R)
    assert compare_to_colmap(wrong, nl["model"], nl["params"], 5120, 3840, step=160)["rms_px"] > 100


def test_decompose_matches_camera_py_and_rescale_is_exact():
    from mppp.camera import CAHVOR
    from mppp.labels import label_get, read_pds
    for path in (NLF, ZL0):
        cm, _ = _label_model(path)
        L, _ = read_pds(path, load_image=False)
        _, R_old = CAHVOR.from_label(label_get(L, "GEOMETRIC_CAMERA_MODEL")).decompose(cm.width, cm.height)
        R, _ = cm.decompose()
        assert np.abs(R - R_old).max() < 1e-5
    cm, _ = _label_model(ZL0)
    X = np.array([[1.0, 2.0, -1.0], [0.3, -2.0, -1.5]]) + cm.C + 5 * cm.A
    half = cm.rescaled(2.0, 10.0, 4.0)                          # a x2 "downsampled" product at (10, 4)
    np.testing.assert_allclose(half.project(X), (cm.project(X) + [10.5, 4.5]) / 2.0 - 0.5, atol=1e-9)


def test_cahvor_is_brown_k1_k2_and_cahvore_fits_navcam():
    from mppp.cmod import fit_to_colmap
    p = [4700.0, 4710.0, 830.0, 590.0, -0.47, 0.65, 0, 0, 0, 0, 0, 0]
    cm, rep = fit_to_colmap("FULL_OPENCV", p, 1648, 1200, "CAHVOR", step=64)
    assert rep["rms_px"] < 1e-6 and np.allclose(cm.R[1:], [-0.47, 0.65], atol=1e-8)
    nl = _rational_v0p21()
    _, r2 = fit_to_colmap(nl["model"], nl["params"], 5120, 3840, "CAHVORE", 2, step=128)
    _, r3 = fit_to_colmap(nl["model"], nl["params"], 5120, 3840, "CAHVORE", 3, True, step=128)
    _, r1 = fit_to_colmap(nl["model"], nl["params"], 5120, 3840, "CAHVOR", step=128)
    assert r3["rms_px"] < r2["rms_px"] < 0.6 < 5 < r1["rms_px"]


def test_compare_cameras_removes_rotation():
    from scipy.spatial.transform import Rotation
    from mppp.cmod import PixelCamera, compare_cameras
    p = np.array([1000.0, 1000, 640, 480, -0.2, 0.05, 0, 0, 0, 0, 0, 0])
    a = PixelCamera.colmap("FULL_OPENCV", p, 1280, 960)
    assert compare_cameras(a, a)["max_px"] < 1e-6
    q = p.copy(); q[2] += 5.0                                     # a principal-point shift is mostly a rotation
    d = compare_cameras(a, PixelCamera.colmap("FULL_OPENCV", q, 1280, 960))
    assert 0.2 < d["rotation_deg"] < 0.4 and d["rms_px"] < 0.5 < 5.0 == pytest.approx(
        compare_cameras(a, PixelCamera.colmap("FULL_OPENCV", q, 1280, 960), fit_rotation=False)["rms_px"], abs=1e-6)
