"""Alignment health (mppp.sfm.health). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
from pathlib import Path
import numpy as np
import pytest
import copy
import json
from types import SimpleNamespace


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
    bad = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec(np.radians([0.5, 0, 0])).as_matrix()
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
    assert DEFAULT_THRESHOLDS["rig_rotation_change_deg"][:2] == (0.06, 0.3)            # 22 Oct 2026 (0.2 before)
    assert DEFAULT_THRESHOLDS["excluded_fraction"][:2] == (0.05, 0.33)          # 24 Oct (Christian)
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
    """A FULL_OPENCV Navcam camera with a rational radial term (the synthetic test camera's f and principal point
    with the k1-k4 of an earlier rational fit, invertible to the frame corners)."""
    from helpers import synthetic_navcam_cameras
    c = synthetic_navcam_cameras()[f"N{eye}"]
    c["params"][4:10] = [0.305, -0.026, 0.0, 0.0, 0.002, 0.593]
    return c


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


# beyond-ground and aspect checks, block screening (v0p65)


def _mock_rec(points, centres):
    ims = {i: SimpleNamespace(has_pose=True, projection_center=(lambda c=c: np.asarray(c, float)))
           for i, c in centres.items()}
    pts = {k + 1: SimpleNamespace(xyz=np.asarray(X, float),
                                  track=SimpleNamespace(elements=[SimpleNamespace(image_id=i) for i in t]))
           for k, (X, t) in enumerate(points)}
    return SimpleNamespace(images=ims, points3D=pts)


def test_beyond_ground_flags_points_far_below_the_horizon():
    from mppp.sfm.health import beyond_ground
    C = {1: [0, 0, 1.9], 2: [0.4, 0, 1.9]}
    good = [([5, y, 0.0], (1, 2)) for y in np.linspace(-2, 2, 17)]          # on flat ground 5 m away
    far = [([40, 0, -1.9 * 40 / 5 + 1.9], (1, 2))] * 3                       # along a ray that meets ground at 5 m
    sky = [([30, 0, 6.0], (1, 2))]                                           # above the horizon: not considered
    r = beyond_ground(_mock_rec(good + far + sky, C))
    assert r["points"] == 21 and r["beyond"] == 3 and r["below_horizon"] == 20
    assert r["fraction"] == pytest.approx(3 / 21) and r["fraction_of_below_horizon"] == pytest.approx(3 / 20)
    # a point that one camera sees close (within 3x its ground range) is not flagged
    mixed = [([40, 0, -1.9 * 40 / 5 + 1.9], (1, 3))]
    C3 = {**C, 3: [38, 0, -1.9 * 40 / 5 + 1.9 + 1.9]}
    assert beyond_ground(_mock_rec(mixed, C3))["beyond"] == 0
    assert beyond_ground(_mock_rec([], C))["fraction"] is None


def test_health_thresholds_v0p65():
    from mppp.sfm.health import DEFAULT_THRESHOLDS, _status
    assert DEFAULT_THRESHOLDS["beyond_ground_fraction"] == (0.05, 0.15, "above")
    assert DEFAULT_THRESHOLDS["navcam_aspect_change_pct"] == (0.06, 0.15, "above")
    thr = DEFAULT_THRESHOLDS
    # the nine good blocks (0-4 %) pass, Three Forks South (24 %) fails; Van Zyl's NR bin (0.21 %) fails the aspect
    assert _status("beyond_ground_fraction", 0.039, thr) == "pass"
    assert _status("beyond_ground_fraction", 0.24, thr) == "fail"
    assert _status("navcam_aspect_change_pct", 0.052, thr) == "pass"
    assert _status("navcam_aspect_change_pct", 0.21, thr) == "fail"


def test_navcam_aspect_changes():
    from mppp.sfm.health import navcam_aspect_changes
    p = [1500.0, 1500.0, 512, 512, 0, 0, 0, 0, 0, 0, 0, 0]
    proj = SimpleNamespace(settings={"database": {"cameras": {"NL": 1, "NR_T-030-020": 2, "ZL034": 3}}},
                           cameras={"NL": {"model": "THIN_PRISM_FISHEYE", "params": p},
                                    "NR_T-030-020": {"model": "THIN_PRISM_FISHEYE", "params": p, "group": "NR"},
                                    "ZL034": {"model": "FULL_OPENCV", "params": p}})
    cams = {1: SimpleNamespace(params=np.array([1500.0, 1500.0 * 1.001])),
            2: SimpleNamespace(params=np.array([1501.0, 1501.0])),
            3: SimpleNamespace(params=np.array([1500.0, 1600.0]))}
    out = navcam_aspect_changes(proj, SimpleNamespace(cameras=cams))
    assert set(out) == {"NL", "NR_T-030-020"}
    assert out["NL"] == pytest.approx(0.1) and out["NR_T-030-020"] == pytest.approx(0.0, abs=1e-9)


def test_screen_block(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from helpers import _block_without_navcam_zcam_matches
    from mppp.sfm import reconstruction as R
    from mppp.sfm.health import screen_block, screen_blocks
    proj, new, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                  navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False, pose_guided=False)
    proj.save()
    r = screen_block(proj.root)
    assert r["verdict"] in ("pass", "warn", "fail") and r["beyond_ground_fraction"] is not None
    assert r["navcam_aspect_change_pct"] is not None and abs(r["navcam_aspect_change_pct"]) < 0.06
    (proj.root / "database_matches.json").write_text(json.dumps({"matching": {"guided_matching": True}}))
    r2 = screen_block(proj.root)
    assert r2["verdict"] != "pass" and any("guided" in s for s in r2["reasons"])
    out = screen_blocks({"a": proj.root, "missing": tmp_path / "none"}, verbose=False)
    assert out["missing"]["verdict"] == "fail"


def test_screen_block_cache(tmp_path, monkeypatch):
    from mppp.sfm import health as H
    root = tmp_path / "w" / "colmap"
    sp = root / "sparse" / "cahv_ba"
    sp.mkdir(parents=True)
    (sp / "points3D.bin").write_bytes(b"x")
    (sp / "cameras.bin").write_bytes(b"y")
    calls = []

    class FakeRec:
        def __init__(self, p):
            calls.append(p)
    import pycolmap
    monkeypatch.setattr(pycolmap, "Reconstruction", FakeRec)
    monkeypatch.setattr(H.SfmProject, "load", staticmethod(lambda r: None))
    monkeypatch.setattr(H, "beyond_ground", lambda rec, sample=None: {"fraction": 0.01})
    monkeypatch.setattr(H, "navcam_aspect_changes", lambda p, r: {"NL": 0.01})
    a = H.screen_block(tmp_path / "w")
    b = H.screen_block(tmp_path / "w")
    assert a["verdict"] == "pass" and b.get("cached") and len(calls) == 1
    (sp / "points3D.bin").write_bytes(b"xx")                  # the alignment changed: screened again
    H.screen_block(tmp_path / "w")
    assert len(calls) == 2


def test_restation_uses_the_label_priors(tmp_path):
    """0.94.4: a block run with moved Hazcam priors (Van Zyl Overlook Huge, 23 Oct: within_station_shift_spread_m
    FAIL 1.06 m at a station whose Navcam moved 2.1 m) is judged again from the label priors."""
    import csv
    import json
    from mppp.sfm.health import rejudge, restation
    root = tmp_path / "colmap"
    (root / "error_input").mkdir(parents=True)
    shift = np.array([2.1, 0.0, 0.0])
    imgs, rows = [], []
    for k in range(6):                                   # 4 Navcam images, 2 Hazcam images at one station
        name = f"{'NLF' if k < 4 else 'FLF'}_{k}.png"
        label = np.array([float(k), 0.0, 0.0])
        r = {"name": name, "prior_C": (label + (shift if k >= 4 else 0)).tolist()}
        if k >= 4:
            r["prior_C_label"] = label.tolist()
        imgs.append(r)
        E, N, U = label + shift + 0.01 * k
        d = np.array([E, N, U]) - np.asarray(r["prior_C"])          # the old table: from the moved prior
        rows.append({"name": name, "station": "S003D0834", "E_m": E, "N_m": N, "U_m": U, "dE_m": d[0], "dN_m": d[1],
                     "dU_m": d[2], "observations": 500})
    (root / "project.json").write_text(json.dumps({"images": imgs}))
    with (root / "error_input" / "poses.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = {"checks": [{"section": "poses", "check": "within_station_shift_spread_m", "value": 1.056, "status": "fail"},
                         {"section": "poses", "check": "station_shift_median_m", "value": 2.1, "status": "warn"}],
              "stations": {"S003D0834": {"label": "Sol0031 S003D0834", "position_prior": True}}}
    changed = restation(report, root)
    assert any("within_station_shift_spread_m" in c for c in changed)
    spread = next(c for c in report["checks"] if c["check"] == "within_station_shift_spread_m")["value"]
    assert spread < 0.05
    assert rejudge(report)[0] in ("pass", "warn")
    # a block without moved priors is left alone
    (root / "project.json").write_text(json.dumps({"images": [{"name": "NLF_0.png", "prior_C": [0, 0, 0]}]}))
    assert restation({"checks": []}, root) == []


def test_hazcam_eye_ratio_does_not_fail_a_block():
    """0.94.4: Knob Mountain Large failed on the rear Hazcams' eye ratio (RL 0.184 / RR 0.422 px = 2.29); a Hazcam
    pair has its own check, and saved reports are judged again with it."""
    from mppp.sfm.health import DEFAULT_THRESHOLDS, rejudge
    assert DEFAULT_THRESHOLDS["hazcam_residual_eye_ratio"][1] > DEFAULT_THRESHOLDS["residual_eye_ratio"][1]
    rep = {"checks": [{"check": "residual_eye_ratio", "value": 2.29, "status": "fail",
                       "note": "median residual, worse eye / better eye: RL: 0.184, RR: 0.422"},
                      {"check": "residual_eye_ratio", "value": 1.05, "status": "pass",
                       "note": "median residual, worse eye / better eye: NL: 0.150, NR: 0.158"}]}
    v, changed = rejudge(rep)
    assert v == "warn" and any("hazcam_residual_eye_ratio" in c for c in changed)
    rep["checks"][1].update(value=2.5, status="fail")              # a Navcam pair still fails
    assert rejudge(rep)[0] == "fail"


def test_label_pointing_checks_do_not_keep_a_block_out_of_the_joints():
    """0.94.4: Witch Hazel Hill (one site's heading 3.3 deg off its label) and Tuxedo Park (block rotation 1.62 deg)
    are judged WARN for the calibration joints; elsewhere the FAIL stands."""
    from mppp.sfm.health import LABEL_POINTING_CHECKS, rejudge
    rep = {"checks": [{"check": "attitude_residual_p95_deg", "value": 3.363, "status": "fail"},
                      {"check": "block_rotation_deg", "value": 1.62, "status": "fail"},
                      {"check": "residual_median_px", "value": 0.12, "status": "pass"}]}
    assert rejudge(rep)[0] == "fail"
    assert rejudge(rep, warn_only=LABEL_POINTING_CHECKS)[0] == "warn"
    rep["checks"].append({"check": "residual_median_px", "value": 1.5, "status": "fail"})
    assert rejudge(rep, warn_only=LABEL_POINTING_CHECKS)[0] == "fail"


def test_a_zoom_joint_judges_the_block_on_its_zoom(tmp_path):
    """0.94.4 (Chal Rocks Mastcam-Z, 24 Oct): 30 % of the frames excluded, nearly all at 79 mm - a FAIL that kept the
    block out of the Z34 joint.  The joint of one zoom counts that zoom's frames (and the Navcam, Hazcams) only."""
    import json as _json
    from mppp.sfm.health import LABEL_POINTING_CHECKS, rejudge, restrict_to_zoom
    nav = [f"NLF_0091_0000000{i:03d}_000RAD_N0040136NCAM00500_0A0195J01.png" for i in range(60)]
    z34 = [f"ZL0_0091_0000001{i:03d}_000RAD_N0040136ZCAM08060_034085A03.png" for i in range(30)]
    z79 = [f"ZL0_0091_0000002{i:03d}_000RAD_N0040136ZCAM08060_079085A03.png" for i in range(50)]
    ex = [{"images": [n], "kind": "outlier", "observations": 5} for n in z79 + z34[:1]]
    (tmp_path / "project.json").write_text(_json.dumps({"images": [{"name": n} for n in nav + z34 + z79],
                                                         "settings": {"reconstruction": {"excluded": ex}}}))
    rep = {"checks": [{"check": "excluded_fraction", "value": 51 / 140, "status": "fail"}]}
    assert rejudge(dict(rep, checks=[dict(rep["checks"][0])]), warn_only=LABEL_POINTING_CHECKS)[0] == "fail"
    notes = restrict_to_zoom(rep, tmp_path, 34)
    assert rep["checks"][0]["value"] == 1 / 90 and notes
    assert rejudge(rep, warn_only=LABEL_POINTING_CHECKS)[0] == "pass"


def test_a_zoom_joint_sets_aside_other_zooms_camera_checks(tmp_path):
    """0.94.4 (Christian, 27 Oct): Taylorfjellet failed on its one 48 mm pair (ZR048_F02070 principal point +90 px)
    and was left out of the Z34 and Z63 joints.  Another zoom's camera-model checks no longer fail the block."""
    from mppp.sfm.health import LABEL_POINTING_CHECKS, rejudge, restrict_to_zoom

    def rep():
        return {"checks": [
            {"section": "camera model", "check": "principal_point_change_px", "value": 90.2, "status": "fail", "warn": 20.0, "fail": 60.0, "note": "ZR048_F02070"},
            {"section": "camera model", "check": "focal_change_pct", "value": 0.1, "status": "pass", "warn": 0.5, "fail": 2.0, "note": "ZL034_F00600"},
            {"section": "camera model", "check": "focal_change_pct", "value": 0.2, "status": "pass", "warn": 0.5, "fail": 2.0, "note": "ZR063_F02500_reg"},
            {"section": "camera model", "check": "principal_point_change_px", "value": 0.5, "status": "pass", "warn": 20.0, "fail": 60.0, "note": "NL_T-020-017.5"}]}
    r = rep()
    assert rejudge(rep(), warn_only=LABEL_POINTING_CHECKS)[0] == "fail"
    notes = restrict_to_zoom(r, tmp_path, 63)                     # no project.json: the camera checks still go
    assert [c["note"] for c in r["checks"]] == ["ZR063_F02500_reg", "NL_T-020-017.5"] and "ZR048_F02070 fail" in notes[0]
    assert rejudge(r, warn_only=LABEL_POINTING_CHECKS)[0] == "pass"
    r = rep()
    restrict_to_zoom(r, tmp_path, 48)                             # the 48 mm joint keeps its own failure
    assert rejudge(r, warn_only=LABEL_POINTING_CHECKS)[0] == "fail"
