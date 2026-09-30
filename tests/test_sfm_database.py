"""Features, database, matching, pairs, GPU steps (mppp.sfm.database, matching, pairs, gpu). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import numpy as np
import pytest
from mppp.paths import data_dir  # noqa: E402
import json
from pathlib import Path


pycolmap = pytest.importorskip("pycolmap")


DATA_DIR = data_dir()


def _synthetic(tmp_path, noise_native_px=0.3, seed=0):
    """Two stations, a mast pan of 6 stereo exposures each (half and quarter resolution), terrain points."""
    from mppp.sfm.project import SfmProject, camera_from_metashape_xml
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    cams = {f"N{e}": camera_from_metashape_xml(DATA_DIR / f"cmods/M2020_N{e}0_frame.xml") for e in "LR"}
    R_rel = Rotation.from_rotvec(np.radians([-0.09, 0.015, -0.08])).as_matrix()
    t_rel = np.array([-0.4244, 0.0, 0.0])
    images, truth = [], {}
    stations = {"S001D0000": np.array([0.0, 0.0, 1.9]), "S001D0100": np.array([4.0, 1.0, 1.9])}
    k = 0
    for sname, C0 in stations.items():
        for az in np.linspace(0, 150, 6):
            k += 1
            # camera: z forward along azimuth, pitched 35 deg down; x right, y down
            a = np.radians(az)
            fwd = np.array([np.sin(a), np.cos(a), 0.0]) * np.cos(np.radians(35)) + np.array([0, 0, -np.sin(np.radians(35))])
            right = np.array([np.cos(a), -np.sin(a), 0.0])
            down = np.cross(fwd, right)
            RL = np.stack([right, down, fwd])
            CL = C0 + rng.normal(0, 0.02, 3)
            RR = R_rel @ RL
            CR = CL - RR.T @ t_rel
            s = 0.5 if k % 3 else 0.25
            for eye, R, C in (("L", RL, CL), ("R", RR, CR)):
                name = f"N{eye}F_{k:04d}.png"
                images.append({"name": name, "stem": name[:-4], "instrument": f"N{eye}", "eye": eye,
                               "sclk_key": f"{k:010d}_000", "sol": 1, "site": 1, "drive": int(sname[-4:]),
                               "station": sname, "sequence": "NCAM00000", "downsample_scale": s,
                               "native_size": [int(5120 * s), int(3840 * s)], "prior_C": C.tolist(),
                               "prior_R_w2c": R.tolist(), "has_mask": False})
                truth[name] = (R, C)
    rig = {"N": {"ref": "NL", "sensor": "NR", "n_pairs": k, "R_sensor_from_ref": R_rel.tolist(),
                 "t_sensor_from_ref": t_rel.tolist(), "baseline_m": 0.4244}}
    proj = SfmProject(tmp_path, images, cams, rig, [0, 0, 0], {"prior_sigma_m": [0.05, 0.05, 0.05]})
    proj.root.mkdir(parents=True, exist_ok=True)
    # terrain points around both stations
    P = np.column_stack([rng.uniform(-12, 16, 4000), rng.uniform(-10, 14, 4000), rng.normal(0, 0.3, 4000)])
    return proj, truth, P, cams, (R_rel, t_rel), noise_native_px, rng


def test_keypoint_scaling_is_exact_for_binned_padded_frames():
    from mppp.sfm.database import scale_keypoints
    kp = np.array([[0.5, 0.5, 2, 0, 0, 2], [1279.5, 959.5, 1, 0, 0, 1]], np.float32)   # quarter-res pixel centres
    full = scale_keypoints(kp, 4.0)
    assert np.allclose(full[:, :2], [[2.0, 2.0], [5118.0, 3838.0]])                  # centre of the 4x4 bin
    assert np.allclose(full[0, 2:], [8, 0, 0, 8])
    kp4 = np.array([[10.0, 20.0, 3.0, 0.7]], np.float32)
    assert np.allclose(scale_keypoints(kp4, 2.0), [[20, 40, 6, 0.7]])


def test_prior_pairs_see_overlap_and_reject_opposite_views(tmp_path):
    from mppp.sfm.pairs import footprints, _visible_fraction
    proj, truth, *_ = _synthetic(tmp_path)
    for r in proj.images:
        R, C = truth[r["name"]]
        r["prior_R_w2c"], r["prior_C"] = R.tolist(), C.tolist()
    import cv2
    (tmp_path / "images").mkdir(exist_ok=True)
    for r in proj.images:
        w, h = r["native_size"]
        cv2.imwrite(str(tmp_path / "images" / r["name"]), np.full((h, w), 100, np.uint8))
    fps = footprints(proj, grid=(16, 12))
    a = fps["NLF_0001.png"]
    assert _visible_fraction(a, fps["NRF_0001.png"], (16, 12)) > 0.8          # stereo partner
    assert _visible_fraction(a, fps["NLF_0006.png"], (16, 12)) < 0.05         # 150 deg away


def test_gpu_steps_run_in_another_python(tmp_path):
    """v0p11: extraction and matching can run as a subprocess in a second environment (a CUDA pycolmap on
    Windows, whose ceres types are incompatible with pip pyceres); settings come back through project.json."""
    import shutil, sys
    import cv2
    from mppp.sfm import SfmProject, check_gpu_python, extract_features, match
    rng = np.random.default_rng(3)
    big = cv2.GaussianBlur((rng.random((700, 900)) * 255).astype(np.uint8), (0, 0), 2)
    big = cv2.equalizeHist(big)
    (tmp_path / "images").mkdir()
    names = []
    for k, x0 in enumerate((0, 120, 240)):
        n = f"im{k}.png"
        cv2.imwrite(str(tmp_path / "images" / n), cv2.cvtColor(big[50:650, x0:x0 + 600], cv2.COLOR_GRAY2BGR))
        names.append(n)
    proj = SfmProject(tmp_path, [{"name": n, "has_mask": False} for n in names], {}, {}, [0, 0, 0], {"keep": 1})
    from mppp.sfm import check_ba_environment
    assert check_ba_environment()["pyceres"]                                  # pip pair: BA works here
    with pytest.warns(UserWarning, match="own python"):
        info = check_gpu_python(sys.executable, require_cuda=False)
    assert info["pycolmap"] == info["pycolmap_here"]
    extract_features(proj, max_num_features=2000, use_gpu=False, python=sys.executable)
    assert proj.features_db.is_file() and proj.settings["features"]["max_num_features"] == 2000
    assert proj.settings["keep"] == 1                                           # parent's settings survive
    shutil.copy(proj.features_db, proj.database)                               # matchable as it is
    res = match(proj, mode="exhaustive", use_gpu=False, max_error_px=4.0, python=sys.executable)
    assert res["verified_pairs"] >= 2 and proj.settings["matching"]["verified_pairs"] == res["verified_pairs"]
    if not pycolmap.has_cuda:
        with pytest.raises(RuntimeError, match="no CUDA"):
            check_gpu_python(sys.executable, require_cuda=True)
    with pytest.raises(FileNotFoundError):
        check_gpu_python(tmp_path / "nope" / "python.exe")


def test_build_database_overwrites_by_default():
    import inspect
    from mppp.sfm.database import build_database
    assert inspect.signature(build_database).parameters["overwrite"].default is True


def test_features_reused_only_with_same_settings(tmp_path):
    from mppp.sfm.database import features_up_to_date, _features_record, image_fingerprints
    from mppp.sfm.project import SfmProject
    proj = SfmProject(tmp_path, [{"name": "a.png"}, {"name": "b.png"}], {}, {}, [0, 0, 0], {})
    assert not features_up_to_date(proj)                                # nothing yet
    proj.features_db.write_bytes(b"")
    assert not features_up_to_date(proj)                                # no record (e.g. extracted by 0.14.2)
    rec = {"max_num_features": 8192, "max_image_size": 5120, "domain_size_pooling": False, "images": ["a.png", "b.png"]}
    _features_record(proj).write_text(json.dumps(rec))
    assert not features_up_to_date(proj, max_num_features=8192)         # no file record (before 0.14.7)
    rec["files"] = image_fingerprints(proj)
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj, max_num_features=8192)
    assert not features_up_to_date(proj, max_num_features=16384)        # new setting -> extract again
    rec["max_num_features"] = 16384
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj)                                    # default is 16384
    proj.images.append({"name": "c.png"})
    assert not features_up_to_date(proj)                                # an image without features


def test_features_extracted_again_when_an_image_or_mask_changes(tmp_path):
    import os
    from mppp.sfm.database import features_up_to_date, _features_record, image_fingerprints
    from mppp.sfm.project import SfmProject
    proj = SfmProject(tmp_path, [{"name": "a.png"}], {}, {}, [0, 0, 0], {})
    proj.images_dir.mkdir(parents=True, exist_ok=True)
    proj.masks_dir.mkdir(parents=True, exist_ok=True)
    (proj.images_dir / "a.png").write_bytes(b"img")
    (proj.masks_dir / "a.png.png").write_bytes(b"mask")
    proj.features_db.write_bytes(b"")
    rec = {"max_num_features": 16384, "max_image_size": 5120, "domain_size_pooling": False, "images": ["a.png"],
           "files": image_fingerprints(proj)}
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj)
    m = proj.masks_dir / "a.png.png"
    m.write_bytes(b"new mask")                                          # the image was processed again
    os.utime(m, ns=(m.stat().st_atime_ns, m.stat().st_mtime_ns + 10**9))
    assert not features_up_to_date(proj)


def test_colmap_gui_project_files(tmp_path):
    from mppp.sfm.database import write_gui_project
    from mppp.sfm.project import SfmProject
    (tmp_path / "masks").mkdir()
    proj = SfmProject(tmp_path, [{"name": "a.png", "has_mask": True}], {}, {}, [0, 0, 0], {})
    out = write_gui_project(proj, model="cahv_ba")
    ini = Path(out["ini"]).read_text().splitlines()
    assert ini[0].startswith("# MPPP") and f"database_path={tmp_path.resolve() / 'database.db'}" in ini
    assert f"image_path={tmp_path.resolve() / 'images'}" in ini and "[ImageReader]" in ini
    bat = Path(out["bat"]).read_bytes()
    assert b"\r\n" in bat and b"--import_path \"%MODEL%\"" in bat and b"sparse\\cahv_ba" in bat
    assert b"%COLMAP_BAT%" in bat


def test_feature_extraction_default_is_native_resolution():
    import inspect
    from mppp.sfm.database import DEFAULT_MAX_IMAGE_SIZE, extract_features
    assert DEFAULT_MAX_IMAGE_SIZE == 5120
    assert inspect.signature(extract_features).parameters["max_image_size"].default == 5120


def test_affine_shape_is_part_of_the_feature_record():
    from mppp.sfm.database import _feature_settings
    assert "estimate_affine_shape" not in _feature_settings(16380, 5120, False)       # older records stay valid
    assert _feature_settings(16380, 5120, True, True)["estimate_affine_shape"] is True


def test_launcher_searches_colmap_bat(tmp_path):
    from mppp.sfm.project import SfmProject
    from mppp.sfm.database import write_gui_project, COLMAP_BAT_CANDIDATES
    proj = SfmProject(tmp_path, [], {}, {}, [0, 0, 0], {})
    out = write_gui_project(proj, model="cahv_ba", colmap_bat=r"D:\tools\colmap-x64-windows-nocuda\COLMAP.bat")
    bat = Path(out["bat"]).read_bytes().decode()
    assert proj.settings["colmap_bat"].endswith("COLMAP.bat")
    assert 'if exist "D:\\tools\\colmap-x64-windows-nocuda\\COLMAP.bat"' in bat
    assert all(c.replace("%LOCALAPPDATA%", "%LOCALAPPDATA%") in bat for c in COLMAP_BAT_CANDIDATES)
    assert "--import_path" in bat and "--database_path" in bat and "$PATH:I" in bat and bat.count("\r\n") > 10
    out2 = write_gui_project(proj, model="cahv_ba")                  # remembered in the settings
    assert 'colmap-x64-windows-nocuda' in Path(out2["bat"]).read_text()


def test_matching_block_size_and_fisheye_param_names():
    import inspect
    from mppp.sfm.matching import match
    from mppp.sfm.reconstruction import _PARAM_NAMES, _FIXED_EXTRA
    assert inspect.signature(match).parameters["block_size"].default == 100
    assert _PARAM_NAMES["THIN_PRISM_FISHEYE"].index("sx1") == 10 and _FIXED_EXTRA["THIN_PRISM_FISHEYE"] == [6, 7, 10, 11]


def test_right_only_exposure_gets_a_single_eye_rig(tmp_path):
    """v0p30: a right image whose left partner is missing is posed in a one-sensor rig, not dropped."""
    import pycolmap
    from helpers import _synthetic
    from mppp.sfm.database import build_database
    from mppp.sfm.reconstruction import initial_reconstruction
    proj, truth, *_ = _synthetic(tmp_path)
    lone = proj.images[3]["name"]
    assert lone.startswith("NR")
    proj.images = [r for r in proj.images if r["name"] != proj.images[2]["name"]]    # drop its left partner
    fdb = pycolmap.Database.open(str(proj.features_db))
    cam = fdb.write_camera(pycolmap.Camera(model="PINHOLE", width=100, height=100, params=[100, 100, 50, 50]))
    rng = np.random.default_rng(0)
    for r in proj.images:
        iid = fdb.write_image(pycolmap.Image(name=r["name"], camera_id=cam))
        fdb.write_keypoints(iid, rng.uniform(0, 100, (20, 2)).astype(np.float32))
        fdb.write_descriptors(iid, pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT,
                                                             rng.integers(0, 255, (20, 128)).astype(np.uint8)))
    fdb.close()
    summary = build_database(proj)
    assert summary["images_without_frame"] == [] and summary["single_eye_images"] == [lone]
    assert summary["rigs"] == 2 and summary["images"] == len(proj.images)
    rec = initial_reconstruction(proj)
    by_name = {im.name: im for im in rec.images.values()}
    im = by_name[lone]
    assert im.has_pose and rec.rigs[rec.frames[im.frame_id].rig_id].num_sensors() == 1
    R, C = truth[lone]
    T = im.cam_from_world()
    assert np.allclose(-T.rotation.matrix().T @ T.translation, next(r for r in proj.images if r["name"] == lone)["prior_C"])
    assert rec.num_reg_images() == len(proj.images)


class _Stub:
    def __init__(self, root, db, order, params):
        self.features_db = root / "features.db"
        self.database = db
        self.images = [{"name": n, "instrument": "C"} for n in order]
        self.cameras = {"C": {"model": "SIMPLE_RADIAL", "width": 400, "height": 300, "params": params}}


def test_match_reuse(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    from PIL import Image
    from scipy.ndimage import affine_transform, gaussian_filter
    from mppp.sfm import database as D
    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(3)
    base = gaussian_filter(rng.random((500, 650)), 1.5)
    base = (255 * (base - base.min()) / np.ptp(base)).astype(np.uint8)
    names = []
    for k in range(4):
        im = affine_transform(base.astype(float), np.eye(2), offset=(15 * k, 20 * k), output_shape=(300, 400), order=1)
        names.append(f"i{k}.png")
        Image.fromarray(im.astype(np.uint8)).save(tmp_path / "images" / names[-1])
    fdb = tmp_path / "features.db"
    pycolmap.extract_features(str(fdb), str(tmp_path / "images"), camera_mode=pycolmap.CameraMode.PER_IMAGE)
    (tmp_path / "features.json").write_text('{"k": 1}')

    def build(db, order, params):
        f = pycolmap.Database.open(str(fdb))
        byn = {im.name: im.image_id for im in f.read_all_images()}
        d = pycolmap.Database.open(str(db))
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=400, height=300, params=params)
        cam.has_prior_focal_length = True
        cid = d.write_camera(cam)
        sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid)
        rig = pycolmap.Rig()
        rig.add_ref_sensor(sensor)
        rid = d.write_rig(rig)
        for n in order:
            iid = d.write_image(pycolmap.Image(name=n, camera_id=cid))
            d.write_keypoints(iid, f.read_keypoints(byn[n]))
            d.write_descriptors(iid, f.read_descriptors(byn[n]))
            fr = pycolmap.Frame()
            fr.rig_id = rid
            fr.add_data_id(pycolmap.data_t(sensor_id=sensor, id=iid))
            d.write_frame(fr)
        d.close()
        f.close()
        return _Stub(tmp_path, db, order, params)

    cam = [350.0, 200.0, 150.0, 0.0]
    p1 = build(tmp_path / "a.db", names, cam)
    pycolmap.match_exhaustive(str(p1.database))
    settings = {"mode": "exhaustive", "max_ratio": 0.8}
    D.write_matches_record(p1, settings)
    rec = json.loads(D.matches_record_path(p1.database).read_text())
    assert rec["features_key"] and set(rec["cameras"]) == set(names)

    def pairs(db):
        import sqlite3
        con = sqlite3.connect(str(db))
        nm = {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
        out = {}
        for pid, rows, cols, data in con.execute("SELECT pair_id, rows, cols, data FROM matches"):
            a, b = D._pair(pid)
            arr = np.frombuffer(data, np.uint32).reshape(rows, cols) if rows else np.zeros((0, 2), np.uint32)
            if nm[a] > nm[b]:
                arr, key = arr[:, ::-1], (nm[b], nm[a])
            else:
                key = (nm[a], nm[b])
            out[key] = sorted(map(tuple, arr.tolist()))
        con.close()
        return out

    ref = pairs(p1.database)
    # one image left out, order reversed: raw matches carried over (columns swapped), geometries verified again
    order = [n for n in names[::-1] if n != "i1.png"]
    p2 = build(tmp_path / "b.db", order, cam)
    r = D.reuse_matches(p2, [p1.database], matching=settings)
    assert r["matches"] == 3 and r["geometries"] == 0
    assert all(pairs(p2.database)[k] == ref[k] for k in pairs(p2.database))
    # same order and camera: geometries kept too
    p3 = build(tmp_path / "c.db", [n for n in names if n != "i1.png"], cam)
    r = D.reuse_matches(p3, [p1.database], matching=settings)
    assert r["matches"] == 3 and r["geometries"] == 3
    # another camera: raw matches only; other matching settings or features: nothing
    p4 = build(tmp_path / "d.db", names, [360.0, 200.0, 150.0, 0.0])
    assert D.reuse_matches(p4, [p1.database], matching=settings)["geometries"] == 0
    p5 = build(tmp_path / "e.db", names, cam)
    assert D.reuse_matches(p5, [p1.database], matching={"mode": "exhaustive", "max_ratio": 0.9})["matches"] == 0
    (tmp_path / "features.json").write_text('{"k": 2}')
    p6 = build(tmp_path / "f.db", names, cam)
    assert D.reuse_matches(p6, [p1.database], matching=settings)["sources"][0]["skipped"] == "other features"


def test_match_settings_defaults():
    from mppp.sfm.matching import match_settings
    s = match_settings("exhaustive", max_ratio=0.9, max_distance=1.0, cross_check=True, guided_matching=False,
                       max_error_px=6.0)
    assert s["min_num_inliers"] == 15 and s["max_num_matches"] == 32768 and s["max_error_px"] == 6.0
    assert match_settings("exhaustive", max_ratio=0.9) != match_settings("exhaustive", max_ratio=0.8)
