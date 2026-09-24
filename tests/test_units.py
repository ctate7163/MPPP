"""Unit tests that need no PDS data (synthetic round trips with known ground truth)."""
import json
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import synthetic_waypoints


# ------------------------------------------------------------------ filenames
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


def test_dedupe_keeps_highest_version_and_never_deletes(tmp_path):
    from mppp.select import dedupe_by_version, find_imgs
    a = tmp_path / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ01.IMG"
    b = tmp_path / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ02.IMG"
    c = tmp_path / "ZL0_0710_0729888971_069RAD_N0332864ZCAM07114_1100LMA01.IMG"
    for p in (a, b, c):
        p.write_bytes(b"x")
    kept, dropped = dedupe_by_version([a, b, c])
    assert kept == [b, c] and dropped == [a]
    assert a.exists()                                            # archive untouched
    assert find_imgs(tmp_path, ["ZL0"], sequ_id="_110") == [c]
    assert find_imgs(tmp_path, ["ZL0"], sequ_id="_034") == []
    assert find_imgs(tmp_path, ["NLF", "ZL0"], sol_range=(709, 709)) == [a, b]


# --------------------------------------------------------------------- config
def test_config_defaults_and_legacy_migration(tmp_path):
    from mppp import load_config
    cfg = load_config()
    assert cfg["export"]["formats"] == ["PNG16"]
    legacy = {"output_dir": "/", "cameras": ["N"], "export": {"format": "PNG8A", "colmap_model": "OPENCV"},
              "masking": {"checkpoint_trained": "x.pt", "checkpoint_template": "", "max_length": 1024},
              "resize": {"standard_width": 2560}, "camera_model": {"coordinate_frame": "NED"}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                           # legacy keys must migrate silently
        cfg = load_config(legacy)
    assert cfg["export"]["formats"] == ["PNG8"] and cfg["export"]["store_mask_in_alpha"] is True
    assert cfg["masking"]["checkpoint"] == "x.pt"
    with pytest.warns(UserWarning, match="unknown key"):
        load_config({"color": {"typo_key": 1}})
    with pytest.raises(ValueError):
        load_config({"export": {"formats": ["JPEG"]}})


def test_config_snapshot_has_version(tmp_path):
    from mppp import load_config, save_config_snapshot, VERSION_TAG
    p = save_config_snapshot(load_config(), tmp_path)
    snap = json.loads(p.read_text())
    assert p.name == f"mppp_config_{VERSION_TAG}.json"
    assert snap["mppp_version_tag"] == VERSION_TAG and "color" in snap["config"]


# --------------------------------------------------------------------- camera
def _synthetic_cahv(fx=1200.0, fy=1190.0, cx=640.3, cy=479.1, seed=0):
    rng = np.random.default_rng(seed)
    Rm = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()      # rows: cam axes in frame
    C = rng.normal(size=3)
    A = Rm[2]
    H = fx * Rm[0] + cx * A
    V = fy * Rm[1] + cy * A
    return C, A, H, V, Rm


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


# ----------------------------------------------------------------- radiometry
def test_tau_table_and_zenith_scale(tmp_path):
    from mppp.radiometry import interpolate_table, zenith_scale
    t = tmp_path / "tau.csv"
    t.write_text("L_s,tau\n0,0.7\n25,0.4\n80,0.4\n360,0.7\n")
    assert interpolate_table(t, 12.5) == pytest.approx(0.55)
    assert interpolate_table(t, 50) == pytest.approx(0.4)
    assert interpolate_table(t, 372.5) == pytest.approx(0.55)        # periodic in L_s
    s, mu = zenith_scale(90.0, 0.3, 0.3, 0.2)
    assert (s, mu) == (pytest.approx(1.0), pytest.approx(1.0))
    s, mu = zenith_scale(30.0, 0.9, 0.3, 0.2)
    assert s == pytest.approx(0.5 * np.exp(-0.6 / 6 / 0.5))
    assert zenith_scale(2.0, 0.3, 0.3, 0.2)[1] == 0.2                # floor near the horizon


def test_quantise_reserves_zero_for_invalid():
    from mppp import default_config
    from mppp.radiometry import quantise
    rad = np.zeros((2, 2, 3))
    rad[0, 0] = 0.05
    rad[0, 1] = 1e-9          # valid but darker than one count
    rad[1, 0] = 10.0          # saturates
    valid = np.array([[True, True], [True, False]])
    i16, i8 = quantise(rad, valid, default_config()["color"])
    assert i16.dtype == np.uint16 and i8.dtype == np.uint8
    assert i16[0, 0, 0] == 10000 and i8[0, 0, 0] == round(10000 / 64)
    assert i16[0, 1, 0] == 1 and i8[0, 1, 0] == 1                    # valid never 0
    assert i16[1, 0, 0] == 65535 and i8[1, 0, 0] == 255
    assert not i16[1, 1].any() and not i8[1, 1].any()


# ------------------------------------------------------------------ waypoints
def test_waypoint_lookup_and_radius():
    from mppp.waypoints import waypoint_for_site_drive, waypoints_within_radius, offset_lonlat
    wp = synthetic_waypoints()
    assert waypoint_for_site_drive(wp, 33, 1000)["properties"]["sol"] == 705
    assert waypoint_for_site_drive(wp, 33, 900)["properties"]["drive"] == 1000     # nearest
    assert waypoint_for_site_drive(wp, 33, 900, exact=True) is None
    assert waypoint_for_site_drive(wp, 99, 0) is None
    assert len(waypoints_within_radius(wp, 13, 10.0)) == 2
    assert len(waypoints_within_radius(wp, 20, 10.0)) == 2          # rover parked since sol 14
    lon, lat = offset_lonlat(77.0, 18.0, d_east=0.0, d_north=3396190.0 * np.pi / 180)
    assert (lon, lat) == (pytest.approx(77.0), pytest.approx(19.0))


# -------------------------------------------------------------------- writers
def _meta():
    return {"mppp_version": "v0p1", "source_product": "X.IMG", "camera_group": "NL0",
            "start_time_utc": "2023-02-17T05:32:48.193", "zoom_mm": None, "undistorted": False,
            "LMST": "a", "LTST": "b",
            "intrinsics": {"K": [[1000, 0, 50], [0, 1000, 40], [0, 0, 1]], "pixel_origin": "centre",
                           "dist_opencv": {"k1": -0.1}},
            "geo": {"lon_east_deg": 77.45089, "lat_deg": 18.44463, "elev_geoid_m": -2567.25}}


def test_png16_rgba_is_lossless_and_carries_metadata(tmp_path):
    import cv2
    import piexif
    from mppp.writers import save_png, read_png_chunks
    rng = np.random.default_rng(0)
    img = rng.integers(0, 65536, (40, 60, 4), dtype=np.uint16)
    p = save_png(img, tmp_path / "a", _meta())
    back = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    assert back.dtype == np.uint16 and np.array_equal(back, img)
    ch = read_png_chunks(p)
    assert list(ch)[0] == "IHDR" and "eXIf" in ch and len(ch["iTXt"]) > 5
    xmp = [c for c in ch["iTXt"] if c.startswith(b"XML:com.adobe.xmp")][0]
    assert b"mppp:payload" in xmp and b"X.IMG" in xmp
    gps = piexif.load(ch["eXIf"][0])["GPS"]
    d, m, s = [n / q for n, q in gps[piexif.GPSIFD.GPSLatitude]]
    assert d + m / 60 + s / 3600 == pytest.approx(18.44463, abs=1e-7)
    d, m, s = [n / q for n, q in gps[piexif.GPSIFD.GPSLongitude]]
    assert d + m / 60 + s / 3600 == pytest.approx(77.45089, abs=1e-7)
    assert gps[piexif.GPSIFD.GPSLongitudeRef] == b"E" and gps[piexif.GPSIFD.GPSAltitudeRef] == 1
    n, q = gps[piexif.GPSIFD.GPSAltitude]
    assert n / q == pytest.approx(2567.25)
    from PIL import Image                                         # an independent decoder accepts the file
    with Image.open(p) as im:
        im.verify()


def test_png8_and_tiff16(tmp_path):
    import tifffile
    from mppp.writers import save_png, save_tiff16
    rng = np.random.default_rng(1)
    i8 = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    from PIL import Image
    assert np.array_equal(np.asarray(Image.open(save_png(i8, tmp_path / "b", _meta()))), i8)
    i16 = rng.integers(0, 65536, (20, 30, 4), dtype=np.uint16)
    p = save_tiff16(i16, tmp_path / "c", _meta())
    assert np.array_equal(tifffile.imread(p), i16)
    with pytest.raises(ValueError):
        save_png(i16.astype(np.float32), tmp_path / "d")


def test_references_offset(tmp_path):
    from mppp.writers import reference_offset, save_references
    refs = [["a", 1234.5, -987.6, 44.0, 10, 20, 30], ["b", 1236.5, -985.6, 46.0, 11, 21, 31]]
    off = reference_offset(refs)
    assert np.allclose(off, [1230, -990, 40])
    rows = save_references(refs, tmp_path / "r.txt", off).read_text().splitlines()
    assert rows[0].split("\t")[0] == "filename" and rows[1].split("\t")[:2] == ["a.png", "4.500000"]


# --------------------------------------------------------------------- colmap
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
