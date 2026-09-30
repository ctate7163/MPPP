"""Alignment health (mppp.sfm.health). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
from pathlib import Path
import numpy as np
import pytest


def test_alignment_health_on_synthetic_block(tmp_path):
    import pycolmap
    from helpers import _build_rec, _synthetic
    from mppp.sfm.health import assess_alignment, health_table, write_health
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=200)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rep = assess_alignment(proj, rec)
    by = {c["check"]: c for c in rep["checks"]}
    assert rep["verdict"] in ("pass", "warn")
    assert by["registered_fraction"]["value"] == 1.0 and by["registered_fraction"]["status"] == "pass"
    assert by["residual_median_px"]["value"] < 0.6 and by["residual_median_px"]["status"] == "pass"
    assert by["ray_displacement_max_px"]["value"] < 3 and by["rig_rotation_change_deg"]["value"] < 0.02
    assert by["station_shift_median_m"]["value"] < 0.1
    assert sum(rep["track_length_histogram"].values()) == len(rec.points3D)
    assert set(rep["cameras"]) == {"NL", "NR"} and "N" in rep["rig"]
    files = write_health(rep, tmp_path / "health", rec, proj)
    assert Path(files["png"]).is_file() and "alignment health" in Path(files["md"]).read_text()
    assert "residual_median_px" in health_table(rep)
    # a broken rig is caught
    from scipy.spatial.transform import Rotation
    sid = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=2)
    T = rec.rigs[1].sensor_from_rig(sid)
    bad = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec(np.radians([0.3, 0, 0])).as_matrix()
                                               @ T.rotation.matrix()), T.translation)
    rec.rigs[1].set_sensor_from_rig(sid, bad)
    by = {c["check"]: c for c in assess_alignment(proj, rec)["checks"]}
    assert by["rig_rotation_change_deg"]["status"] == "fail"
    # thresholds can be overridden
    rep2 = assess_alignment(proj, rec, thresholds={"rig_rotation_change_deg": (1.0, 2.0, "above")})
    assert {c["check"]: c for c in rep2["checks"]}["rig_rotation_change_deg"]["status"] == "pass"


def test_weak_image_causes():
    from mppp.sfm.health import classify_weak_image as c
    assert c(100, 0, 0, 0) == "few_keypoints"
    assert c(5000, None, None, 0) == "no_database"
    assert c(5000, 3, 10, 0) == "unmatched"
    assert c(5000, 5, 400, 2) == "stereo_only_far"
    assert c(5000, 900, 400, 10) == "lost_in_triangulation"
    assert c(5000, 6600, 0, 9, inliers_other_station=0) == "same_station_only"        # a left-only mast pan
    assert c(5000, 6600, 0, 9, inliers_other_station=500) == "lost_in_triangulation"


def test_health_lists_weak_images_with_advice(tmp_path):
    from helpers import _build_rec, _synthetic
    from mppp.sfm.health import DEFAULT_THRESHOLDS, assess_alignment, health_table
    assert DEFAULT_THRESHOLDS["rig_rotation_change_deg"][:2] == (0.06, 0.2)            # doubled (v0p14.2)
    assert DEFAULT_THRESHOLDS["outlier_image_fraction"][:2] == (0.04, 0.20)
    assert DEFAULT_THRESHOLDS["ray_displacement_max_px"][:2] == (20.0, 60.0)
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    victim = 3
    for pid in [pid for pid, pt in rec.points3D.items() if any(e.image_id == victim for e in pt.track.elements)]:
        if rec.points3D[pid].track.length() <= 2:
            rec.delete_point3D(pid)
        else:
            idx = [e.point2D_idx for e in rec.points3D[pid].track.elements if e.image_id == victim][0]
            rec.delete_observation(victim, idx)
    rep = assess_alignment(proj, rec)
    weak = rep["weak_images"]
    assert [w["name"] for w in weak] == [proj.images[victim - 1]["name"]] and weak[0]["observations"] == 0
    assert weak[0]["cause"] == "no_database"                                           # no database.db here
    txt = health_table(rep)
    assert "images with few observations (1)" in txt and proj.images[victim - 1]["name"] in txt


def _rational(eye="L", zero=()):
    from mppp.paths import data_dir
    from mppp.sfm.project import camera_from_colmap_json
    return camera_from_colmap_json(data_dir() / f"cmods/M2020_N{eye}_rational.json", zero)


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


def test_write_health_adds_station_map(tmp_path):
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm.health import assess_alignment, write_health
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    files = write_health(assess_alignment(proj, rec), tmp_path / "health", rec, proj)
    assert Path(files["station_map"]).is_file() and Path(files["png"]).is_file()
