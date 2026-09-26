"""v0p20: rational Navcam cameras, corner coverage, scope, and the review fixes."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_v0p13 import processed_pair  # noqa: F401  (module fixture: NLF + ZL0 processed once)


def _rational(eye="L", zero=()):
    from mppp.paths import data_dir
    from mppp.sfm.project import camera_from_colmap_json
    return camera_from_colmap_json(data_dir() / f"m20_cmods/M2020_N{eye}_rational.json", zero)


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
    poly = camera_from_metashape_xml(data_dir() / "m20_cmods/M2020_NL0_frame.xml")
    xy = np.asarray(pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=poly["params"]).cam_from_img(g))
    assert 0 < np.isnan(xy[:, 0]).sum() < 0.05 * len(g)
    assert _rational("L", ("p1", "p2"))["params"][6] == 0.0              # TANGENTIAL = "zero" still possible


def test_bundle_adjustment_refines_k4_of_the_rational_camera(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    for with_map in (True, False):
        proj, truth, P, _, rigT, noise, rng = _synthetic(tmp_path / str(with_map))
        cams = {f"N{e}": _rational(e) for e in "LR"}
        proj.cameras = cams
        if with_map:
            proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
        rec, true_params = _build_rec(proj, truth, P, cams, rigT, noise, rng)
        for cid in (1, 2):
            q = np.array(rec.cameras[cid].params)
            q[9] += 0.02                                                    # start k4 off
            rec.cameras[cid].params = q
        if not with_map:                                                    # free_params would be ignored
            with pytest.raises(RuntimeError, match="build_database"):
                bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=5)
            continue
        bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=200)
        for cid in (1, 2):
            p, t = np.asarray(rec.cameras[cid].params), true_params[cid]
            assert abs(p[9] - t[9]) < 0.005 and abs(p[0] - t[0]) < 1.0      # k4 and f recovered
            assert p[10] == p[11] == 0.0                                    # k5, k6 held


def test_project_default_is_rational_and_scope_is_checked_first(processed_pair, tmp_path):  # noqa: F811
    from mppp.sfm.project import SfmProject
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None)
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
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None)
    m = man["images"][0]
    src = out / m["outputs"]["PNG8"]
    dst = proj.images_dir / (Path(m["source_product"]).stem + ".png")
    data = dst.read_bytes()
    src.write_bytes(data + b"\0")                                           # processed again
    os.utime(src, ns=(src.stat().st_atime_ns, dst.stat().st_mtime_ns + 10 ** 9))
    assert proj.refresh_images(man["images"], link=False) == 2
    assert dst.read_bytes() == data + b"\0"
    src.write_bytes(data)


def test_coverage_by_radius_and_corner_ratio():
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm.health import _corner_ratio, coverage_by_radius
    from mppp.sfm.project import SfmProject
    rng = np.random.default_rng(0)
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(camera_id=1, model="PINHOLE", width=1000, height=800, params=[800, 800, 500, 400])
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    fr = pycolmap.Frame(frame_id=1, rig_id=1)
    fr.rig_from_world = pycolmap.Rigid3d()
    fr.add_data_id(pycolmap.data_t(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), 1))
    rec.add_frame(fr)
    xy = rng.uniform([0, 0], [1000, 800], (4000, 2))
    im = pycolmap.Image(name="a.png", keypoints=xy, camera_id=1, image_id=1)
    im.frame_id = 1
    rec.add_image(im)
    rec.register_frame(1)
    r = np.hypot(xy[:, 0] - 500, xy[:, 1] - 400) / np.hypot(500, 400)
    for k in np.nonzero(r < 0.85)[0]:                                       # corners never triangulated
        X = np.r_[(xy[k] - [500, 400]) / 800 * 10, 10.0]
        tr = pycolmap.Track()
        tr.add_element(1, int(k))
        pid = rec.add_point3D(X, tr)
        rec.images[1].set_point3D_for_point2D(int(k), pid)
    proj = SfmProject(Path("."), [{"name": "a.png", "image_id": 1, "camera_group": "NL", "downsample_scale": 1.0}],
                      {}, {}, [0, 0, 0], {})
    cov = coverage_by_radius(rec, proj)
    f = cov["cameras"]["NL"]["triangulated_fraction"]
    assert f[0] == pytest.approx(1.0) and f[-1] == 0.0 and f[-2] == 0.0
    assert cov["cameras"]["NL"]["residual_median_native_px"][0] == pytest.approx(0.0, abs=1e-6)
    grp, ratio, f_in, f_out = _corner_ratio(cov)
    assert grp == "NL" and ratio == 0.0 and f_in == pytest.approx(1.0)


def test_health_thresholds_v0p20():
    from mppp.sfm.health import DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS["station_shift_median_m"][:2] == (2.0, 6.0)
    assert DEFAULT_THRESHOLDS["within_station_shift_spread_m"][:2] == (0.1, 0.4)
    assert DEFAULT_THRESHOLDS["corner_triangulated_ratio"] == (0.5, 0.25, "below")


def test_camera_change_reaches_the_corners():
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm.health import _camera_change
    c0 = _rational("L")
    p1 = np.array(c0["params"])
    p1[9] += 0.01                                                           # k4 acts mostly in the corners
    ch = _camera_change(c0, pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=p1))
    cam0 = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=c0["params"])
    cam1 = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=p1)
    ray = np.c_[np.asarray(cam0.cam_from_img(np.array([[0.5, 0.5]]))), [[1.0]]]
    d_corner = float(np.linalg.norm(np.asarray(cam1.img_from_cam(ray)) - [0.5, 0.5]))
    assert ch["ray_displacement_max_px"] == pytest.approx(d_corner, rel=1e-6)   # the frame corner is sampled
    assert ch["ray_displacement_max_px"] > 2 * ch["ray_displacement_rms_px"] > 0
    assert ch["k4_initial"] == pytest.approx(c0["params"][9]) and ch["k4_refined"] == pytest.approx(p1[9])


def test_reuse_existing_notices_a_new_waypoint_table(tmp_path):
    import mppp
    from mppp.process import reusable_images
    cfg = mppp.load_config({"masking": {"infer_mask": False}})
    (tmp_path / "images_png8").mkdir()
    (tmp_path / "images_png8" / "a.png").write_bytes(b"x")
    meta = {"source_product": "a.IMG", "site": 1, "drive": 2, "outputs": {"PNG8": "images_png8/a.png"},
            "mask": {"inferred": False}}
    (tmp_path / f"mppp_manifest_{mppp.VERSION_TAG}.json").write_text(json.dumps({"images": [meta]}))
    (tmp_path / f"mppp_config_{mppp.VERSION_TAG}.json").write_text(
        json.dumps({"config": cfg, "waypoints": {"sha256": "old"}}, default=str))
    have, rep = reusable_images(tmp_path, cfg, {"_mppp_source": {"sha256": "old"}})
    assert list(have) == ["a"]
    have, rep = reusable_images(tmp_path, cfg, {"_mppp_source": {"sha256": "new"}})
    assert have == {} and rep["reason"] == "waypoint table changed"


def test_feature_extraction_default_is_native_resolution():
    import inspect
    from mppp.sfm.database import DEFAULT_MAX_IMAGE_SIZE, extract_features
    assert DEFAULT_MAX_IMAGE_SIZE == 5120
    assert inspect.signature(extract_features).parameters["max_image_size"].default == 5120


def test_gate_curve_cv_zero_limit_has_no_warnings():
    import warnings
    from mppp.error.alignment import gate_curve
    th = np.linspace(0, 30, 31)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = gate_curve(th, 1.0, 5.0, 0.0)
        b = gate_curve(th, 1.0, 5.0, 2e-3)
    assert np.allclose(a, np.exp(-th / 5.0)) and np.allclose(a, b, rtol=1e-3)


def test_attitude_prior_holds_the_block_orientation(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    rot = {}
    for sigma in (None, 0.01):
        proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path / str(sigma))
        proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
        rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, np.random.default_rng(1), perturb=False)
        # priors: every attitude rotated by 0.5 deg about the line through the two stations (a gauge direction)
        axis = np.array([4.0, 1.0, 0.0]) / np.linalg.norm([4.0, 1.0, 0.0])
        dR = Rotation.from_rotvec(np.radians(0.5) * axis).as_matrix()
        for r in proj.images:
            r["prior_R_w2c"] = (np.asarray(r["prior_R_w2c"]) @ dR.T).tolist()
        out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=100, attitude_prior_deg=sigma)
        assert out["attitude_priors"] == (0 if sigma is None else len(rec.frames))
        ang = []
        for im in rec.images.values():
            R1 = np.asarray(im.cam_from_world().rotation.matrix())
            R0, _ = truth[im.name]
            ang.append(np.degrees(np.linalg.norm(Rotation.from_matrix(R0.T @ R1).as_rotvec())))
        rot[sigma] = float(np.median(ang))
    assert rot[None] < 0.05                  # no attitude prior: the block stays where the tie points put it
    assert 0.4 < rot[0.01] < 0.6             # a strong attitude prior pulls it onto the rotated priors
