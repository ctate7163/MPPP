"""Tests on the two real M2020 PDS products in data/m20 (skipped when absent)."""
import json

import cv2
import numpy as np
import pytest

from conftest import CKPT, NLF, ZL0, needs_ckpt, needs_data

pytestmark = needs_data


def _great_circle_deg(az1, el1, az2, el2):
    def v(az, el):
        az, el = np.radians(az), np.radians(el)
        return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    return np.degrees(np.arccos(np.clip(v(az1, el1) @ v(az2, el2), -1, 1)))


@pytest.fixture(scope="module")
def nlf(cfg_nomask, waypoints):
    from mppp import MPPPImage
    return MPPPImage(NLF, cfg_nomask, waypoints)


@pytest.fixture(scope="module")
def zl0(cfg_nomask, waypoints):
    from mppp import MPPPImage
    return MPPPImage(ZL0, cfg_nomask, waypoints)


def test_label_reading():
    from mppp.labels import read_pds, label_float, label_get
    label, img = read_pds(NLF)
    assert img.shape == (944, 1264, 3)                               # band-sequential -> HWC
    assert label_get(label, "ROVER_MOTION_COUNTER")[:2] == [33, 2864]
    assert label_float(label, "SITE_DERIVED_GEOMETRY_PARMS.SOLAR_ELEVATION") == pytest.approx(78.0691)
    assert label_float(label, "derived_image_parms.radiance_scaling_factor") == pytest.approx(5e-6)


def test_navcam_tile_is_padded_to_full_frame(nlf):
    assert (nlf.width, nlf.height) == (5120, 3840) and nlf.native_size == (1264, 944)
    assert nlf.padding == {"left": 1928, "right": 1928, "top": 1448, "bottom": 1448}
    assert nlf.image_int16.shape == (3840, 5120, 3) and nlf.mask.shape == (3840, 5120)
    assert not nlf.image_int16[:1448].any() and not nlf.mask_valid[:1448].any()
    inside = nlf.mask_valid[1448:1448 + 944, 1928:1928 + 1264]
    assert (inside > 0).mean() > 0.99
    # label principal point moves with the padding
    lab = nlf.intrinsics_label
    assert lab.cx == pytest.approx(666.709, abs=1e-2)
    # XML calibration (2560x1920) rescaled x2 to the full frame
    assert "M2020_NL1_frame.xml" in nlf.intrinsics.source
    assert nlf.intrinsics.fy == pytest.approx(2 * 1475.4568151785811)
    assert nlf.intrinsics.dist["k3"] == pytest.approx(-0.022056754778257665)
    # and the XML principal point agrees with the padded label model to a few pixels
    assert abs(nlf.intrinsics.cx - (lab.cx + 1928)) < 10 and abs(nlf.intrinsics.cy - (lab.cy + 1448)) < 10
    assert abs(nlf.intrinsics.fx / lab.fx - 1) < 0.01


def test_zcam_is_demosaiced_and_keeps_label_model(zl0):
    assert (zl0.width, zl0.height) == (1648, 1200) and not any(zl0.padding.values())
    assert any("demosaiced" in m for m in zl0.log)
    assert zl0.intrinsics.source.startswith("PDS label CAHVOR")
    assert zl0.intrinsics.fx == pytest.approx(4664.06, abs=0.1)     # 34 mm / 7.4 um ~ 4600 px
    assert zl0.fn.zoom_mm == 34 and zl0.meta["focus_position_count"] == 270
    rgb = zl0.image_int16[zl0.mask_valid > 0].astype(float).mean(axis=0)
    assert rgb.min() > 1000 and rgb.max() / rgb.min() < 2.0         # three plausible colour channels
    assert zl0.cahvor.o_a_angle_deg() < 0.5                          # O ~= A assumption holds


def test_radiometry_matches_hand_calculation(zl0):
    from mppp.labels import read_pds
    _, dn = read_pds(ZL0)
    assert zl0.tau_estimated == pytest.approx(0.4)                   # L_s 25.4 lies on the 25..80 plateau
    mu = np.sin(np.radians(58.1934))
    assert zl0.scale_zenith == pytest.approx(mu * np.exp(-(0.4 - 0.3) / 6 / mu))
    # a green Bayer site keeps its own sample through Malvar demosaicing: check one pixel end to end
    y, x = 600, 801                                                  # RGGB: (even row, odd col) = G
    assert dn[y, x] > 0
    rad = dn[y, x] * 1.921518333e-06 + 0.0007140575326
    expect = rad / zl0.scale_zenith * 1.3 * 2e5
    assert zl0.image_int16[y, x, 1] == pytest.approx(expect, abs=1.0)
    assert zl0.image_int8[y, x, 1] == pytest.approx(expect / 64, abs=1.0)


def test_boresight_agrees_with_label_pointing(nlf, zl0):
    """Independent check of CAHV -> rover-nav -> site -> ENU: the label reports mast pointing."""
    for im in (nlf, zl0):
        g = im.label["SITE_DERIVED_GEOMETRY_PARMS"]
        az, el = im.pose.boresight_az_el_deg
        assert _great_circle_deg(az, el, g["INSTRUMENT_AZIMUTH"] % 360, g["INSTRUMENT_ELEVATION"]) < 1.5
        assert np.linalg.det(im.pose.R_w2c) == pytest.approx(1.0)


def test_camera_height_and_site_frame_position(cfg_nomask):
    from mppp import MPPPImage
    im = MPPPImage(ZL0, cfg_nomask, None)                            # no waypoints: own site frame
    assert im.pose.frame == "site33_enu" and im.geo["lat_deg"] is None
    n, e, d = 254.136, -335.623, -42.4607
    assert np.allclose(im.pose.C[:2], [e, n], atol=1.5)              # camera within the rover footprint
    assert 1.8 < im.pose.C[2] - (-d) < 2.3                           # Mastcam-Z ~2 m above the nav origin


def test_waypoint_position_and_gps(zl0, waypoints):
    assert zl0.pose.frame == "site3_enu"
    assert zl0.geo["lat_deg"] is not None and 18.3 < zl0.geo["lat_deg"] < 18.6
    assert 77.2 < zl0.geo["lon_east_deg"] < 77.6
    if "_mppp_source" not in waypoints:                              # synthetic fixture: exact expectation
        d_e, d_n = 4350000.0 - 4354494.0 - 335.623, 1096000.0 - 1093299.0 + 254.136
        assert np.allclose(zl0.pose.C[:2], [d_e, d_n], atol=1.5)
        assert zl0.pose.C[2] == pytest.approx(69.9 + 42.4607 + 2.0, abs=0.4)


def test_stereo_free_projection_consistency(zl0):
    """A point on the optical axis projects to the principal point under K, R, C."""
    az, el = np.radians(zl0.pose.boresight_az_el_deg)
    X = zl0.pose.C + 10 * np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    xc = zl0.pose.R_w2c @ (X - zl0.pose.C)
    uv = zl0.intrinsics.K @ (xc / xc[2])
    assert np.allclose(uv[:2], [zl0.intrinsics.cx, zl0.intrinsics.cy], atol=1e-6)


def test_undistort_zeroes_distortion_and_keeps_size(waypoints):
    from mppp import MPPPImage, load_config
    cfg = load_config({"masking": {"infer_mask": False}, "resize": {"undistort": True}})
    im = MPPPImage(ZL0, cfg, waypoints)
    assert im.undistorted and not im.intrinsics.is_distorted()
    assert im.image_int16.shape == (1200, 1648, 3) and im.intrinsics.skew == 0
    assert set(np.unique(im.mask_valid)) <= {0, 255}
    assert not im.image_int16[im.mask_valid == 0].any()


@needs_ckpt
def test_mask_inference(waypoints):
    from mppp import MPPPImage, load_config
    from mppp.mask import get_model, read_card
    cfg = load_config()
    cfg["masking"]["checkpoint"] = str(CKPT)            # the v0p10 default if present, else the 2025 model
    im = MPPPImage(ZL0, cfg, waypoints)
    assert im.mask_card["backbone"] == "convnext_tiny" and im.mask_card["threshold"] == read_card(CKPT)["threshold"]
    assert set(np.unique(im.mask)) <= {0, 255}
    assert not im.mask[im.mask_valid == 0].any()                    # invalid pixels never included
    assert im.mask_probability.shape == (1200, 1648) and 0 <= im.mask_probability.min() <= im.mask_probability.max() <= 1
    # this frame looks down at terrain (elevation -43 deg): nearly all valid pixels are terrain
    assert (im.mask > 0).sum() / (im.mask_valid > 0).sum() > 0.9
    assert get_model(CKPT, "cpu")[0] is get_model(CKPT, "cpu")[0]   # model loaded once


def test_process_images_end_to_end(tmp_path, waypoints):
    from mppp import process_images, VERSION_TAG
    from mppp.writers import read_png_chunks
    cfg = {"masking": {"infer_mask": False},
           "export": {"formats": ["PNG16", "PNG8"], "write_mask_files": True}}
    man = process_images([NLF, ZL0, tmp_path / "missing_0000_0000000000_000RAD_N0000000XXXX00000_0000LLJ01.IMG"],
                         tmp_path, cfg, waypoints, progress=False)
    assert man["n_processed"] == 2 and len(man["failed"]) == 1 and man["world_frames"] == ["site3_enu"]
    for stem in (NLF.stem, ZL0.stem):                               # PDS names preserved
        for d in ("images_png16", "images_png8", "masks"):
            assert (tmp_path / d / f"{stem}.png").is_file()
    png = tmp_path / "images_png16" / f"{ZL0.stem}.png"
    arr = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    assert arr.dtype == np.uint16 and arr.shape == (1200, 1648, 4)
    assert set(np.unique(arr[..., 3])) <= {0, 65535}
    assert "eXIf" in read_png_chunks(png)
    snap = json.loads((tmp_path / f"mppp_config_{VERSION_TAG}.json").read_text())
    assert snap["mppp_version_tag"] == VERSION_TAG and snap["config"]["color"]["scale_rad_to_int16"] == 2e5
    assert (tmp_path / f"mppp_manifest_{VERSION_TAG}.json").is_file()
    refs = (tmp_path / "references.txt").read_text().splitlines()
    assert len(refs) == 3 and refs[1].startswith(NLF.stem + ".png")
    cams = (tmp_path / "colmap/sparse_prior/cameras.txt").read_text()
    assert "FULL_OPENCV 5120 3840" in cams and "OPENCV 1648 1200" in cams
    assert man["colmap"]["cameras"].keys() == {"NL0", "ZL034"}


def test_two_positioning_routes_agree(cfg_nomask):
    """
    Independent check on real waypoints: exact (site 33, drive 2864) waypoint vs
    (site 33, drive 0) waypoint + label ORIGIN_OFFSET_VECTOR.  They differ only by
    localisation updates between the two (5.6 m E, 0.4 m N, 0.7 m U here).
    """
    from mppp import MPPPImage, load_waypoints
    from mppp.waypoints import waypoint_for_site_drive
    from mppp.waypoints import snapshot_path
    wp = load_waypoints(snapshot_path())
    exact = MPPPImage(ZL0, cfg_nomask, wp)
    assert exact.pose.position_source.endswith("(exact)")
    trimmed = {"features": [f for f in wp["features"]
                            if (f["properties"]["site"], f["properties"]["drive"]) != (33, 2864)]}
    via_offset = MPPPImage(ZL0, cfg_nomask, trimmed)
    assert "drive 0 + label" in via_offset.pose.position_source
    d = exact.pose.C - via_offset.pose.C
    assert np.hypot(d[0], d[1]) < 10.0 and abs(d[2]) < 2.0
    assert exact.geo["lat_deg"] == pytest.approx(18.46167105) and exact.geo["lon_east_deg"] == pytest.approx(77.40293429)
