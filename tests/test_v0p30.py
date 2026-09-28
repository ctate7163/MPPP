"""v0p30: Navcam brightness, LMST window and saturation rules, worker pool, station map, track-mean point colours,
COLMAP launcher, linear solver / block size options, block-rotation health check, five-site consensus cameras."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import NLF, ZL0, needs_data, synthetic_waypoints

ROOT = Path(__file__).resolve().parents[1]


def test_defaults():
    import mppp
    cfg = mppp.load_config({})
    assert cfg["color"]["brightness_by_family"] == {"N": 0.9}
    # the tests run with the selection rules off (conftest); the shipped defaults are these
    import mppp.config as c
    src = (ROOT / "src" / "mppp" / "config.py").read_text()
    assert '"lmst_window_h": [9.0, 17.0]' in src and '"max_saturated_fraction": 0.05' in src
    assert '"max_boresight_elevation_deg": 45.0' in src
    import inspect
    from mppp.process import process_images
    assert inspect.signature(process_images).parameters["workers"].default == 4


@needs_data
def test_brightness_lmst_and_saturation_rules(tmp_path):
    import mppp
    from mppp.image import MPPPImage, LmstOutOfWindow, SaturatedImage, lmst_hours
    from mppp.process import process_images
    wp = synthetic_waypoints()
    assert lmst_hours("Sol-01451M13:00:25.313") == pytest.approx(13.007, abs=1e-3) and lmst_hours(None) is None
    base = {"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": None}}
    im = MPPPImage(NLF, mppp.load_config(base), wp)
    assert im.brightness == 0.9 and im.meta["brightness"] == 0.9 and im.meta["saturated_fraction"] > 0.99
    assert MPPPImage(ZL0, mppp.load_config(base), wp).brightness == 1.0
    dark = mppp.load_config(dict(base, color={"brightness_by_family": {"N": 0.5}}))
    a = MPPPImage(ZL0, mppp.load_config(base), wp).image_int8.astype(int)
    b = MPPPImage(ZL0, mppp.load_config(dict(base, color={"brightness_by_family": {"Z": 0.5}})), wp).image_int8.astype(int)
    v = a > 10
    assert np.median(b[v] / a[v]) == pytest.approx(0.5, abs=0.05)          # brightness scales the product
    with pytest.raises(SaturatedImage):
        MPPPImage(NLF, mppp.load_config(dict(base, selection={"max_boresight_elevation_deg": None, "max_saturated_fraction": 0.05})), wp)
    with pytest.raises(LmstOutOfWindow):
        MPPPImage(ZL0, mppp.load_config(dict(base, selection={"lmst_window_h": [9.0, 12.0]})), wp)   # taken at 14.5 h
    MPPPImage(ZL0, mppp.load_config(dict(base, selection={"lmst_window_h": [9.0, 17.0]})), wp)
    man = process_images([NLF, ZL0], tmp_path / "out", mppp.load_config(dict(base, selection={
        "max_boresight_elevation_deg": None, "max_saturated_fraction": 0.05, "lmst_window_h": [9.0, 17.0]})), wp,
        progress=False, workers=1)
    assert man["n_processed"] == 1 and len(man["skipped"]) == 1 and "saturated" in man["skipped"][0]["reason"]


@needs_data
def test_worker_pool_matches_sequential(tmp_path):
    import mppp
    from mppp.process import process_images
    wp = synthetic_waypoints()
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"]}})
    seq = process_images([ZL0, NLF], tmp_path / "seq", cfg, wp, progress=False, workers=1)
    par = process_images([ZL0, NLF], tmp_path / "par", cfg, wp, progress=False, workers=2)
    assert seq["n_processed"] == par["n_processed"] == 2 and not par["failed"]
    assert [m["source_product"] for m in seq["images"]] == [m["source_product"] for m in par["images"]]   # selection order
    for a, b in zip(seq["images"], par["images"]):
        assert a["pose"]["C_enu_m"] == b["pose"]["C_enu_m"] and a["saturated_fraction"] == b["saturated_fraction"]


def test_bundle_adjust_solver_choice_and_block_rotation(tmp_path):
    pytest.importorskip("pyceres")
    import pycolmap
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.health import assess_alignment
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=10)
    assert out["linear_solver"] == "dense_schur" and out["num_threads"] >= 1 and out["seconds"] > 0
    out2 = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=3, linear_solver="sparse_schur")
    assert out2["linear_solver"] == "sparse_schur"
    with pytest.raises(KeyError):
        bundle_adjust(rec, proj, sigma_px=noise, max_iterations=1, linear_solver="magic")
    rep = assess_alignment(proj, rec)
    c = next(x for x in rep["checks"] if x["check"] == "block_rotation_deg")
    assert c["value"] < 0.3 and c["status"] == "pass" and len(rep["world_frame"]["block_rotation_about_E_N_U_deg"]) == 3
    assert rep["world_frame"]["offset_enu_m"] == [0, 0, 0]
    assert any(x["check"] == "attitude_residual_p95_deg" and x["status"] == "pass" for x in rep["checks"])
    # turn the whole solution by +0.8 deg about Up: reported as +0.8 about U (sign: prior -> refined)
    from scipy.spatial.transform import Rotation
    W = Rotation.from_rotvec(np.radians([0, 0, 0.8])).as_matrix()
    base = np.array(rep["world_frame"]["block_rotation_about_E_N_U_deg"])
    for fr in rec.frames.values():
        T = fr.rig_from_world
        R, C = T.rotation.matrix(), -T.rotation.matrix().T @ T.translation
        R2 = R @ W.T                                      # camera axes in the world: W @ R^T
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R2), -R2 @ (W @ C))
    rep2 = assess_alignment(proj, rec)
    d = np.array(rep2["world_frame"]["block_rotation_about_E_N_U_deg"]) - base
    assert abs(d[2] - 0.8) < 0.05 and abs(d[0]) < 0.05 and abs(d[1]) < 0.05
    c2 = next(x for x in rep2["checks"] if x["check"] == "block_rotation_deg")
    assert c2["status"] == "warn"


def test_write_health_adds_station_map(tmp_path):
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.health import assess_alignment, write_health
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    files = write_health(assess_alignment(proj, rec), tmp_path / "health", rec, proj)
    assert Path(files["station_map"]).is_file() and Path(files["png"]).is_file()


def test_point_colors_from_tracks(tmp_path):
    pytest.importorskip("pycolmap")
    import cv2
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.export import native_reconstruction, point_colors_from_tracks
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    nat = native_reconstruction(rec, proj, observed_only=False)
    # images: left eye red, right eye blue -> a point seen by both is purple; a black left image contributes nothing
    proj.images_dir.mkdir(exist_ok=True)
    for im in nat.images.values():
        cam = nat.cameras[im.camera_id]
        col = (0, 0, 255) if im.name.startswith("NL") else (255, 0, 0)          # BGR
        cv2.imwrite(str(proj.images_dir / im.name), np.full((cam.height, cam.width, 3), col, np.uint8))
    rep = point_colors_from_tracks(nat, proj.images_dir)
    assert rep["coloured_from_tracks"] == rep["points"] and rep["images_missing"] == 0
    cols = np.array([p.color for p in nat.points3D.values()])
    both = [p for p in nat.points3D.values() if {nat.images[e.image_id].name[:2] for e in p.track.elements} == {"NL", "NR"}]
    assert both and all(abs(int(p.color[0]) - int(p.color[2])) < 130 and p.color[1] == 0 for p in both)
    # a missing image: the point keeps the colour of the images that are there
    (proj.images_dir / next(im.name for im in nat.images.values() if im.name.startswith("NL"))).unlink()
    rep2 = point_colors_from_tracks(nat, proj.images_dir)
    assert rep2["images_missing"] == 1


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


def test_shipped_cameras_are_the_five_site_consensus():
    from mppp.paths import data_dir
    for g in ("NL", "NR"):
        d = json.loads((data_dir() / "m20_cmods" / f"M2020_{g}_rational.json").read_text())
        assert "five" in d["source"] and len(d["verification"]["per_scape"]) == 5
        assert all(r["rms_px"] < 0.5 for r in d["verification"]["per_scape"])
    rig = json.loads((data_dir() / "m20_cmods" / "M2020_N_rig.json").read_text())
    assert len(rig["per_solution_rotvec_rad"]) == 5 and abs(rig["baseline_m"] - 0.4244) < 1e-3


def test_notebook_03_v0p30_settings():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)
    assert ns["KEEP_ONLY_REMAINING"] is False and ns["ADD_NEARBY_WAYPOINTS"] == 10 and ns["SITE"] == "threeforks"
    assert ns["LMST_WINDOW_H"] == (9.0, 17.0) and ns["MAX_SATURATED_FRACTION"] == 0.05 and ns["NAVCAM_BRIGHTNESS"] == 0.9
    assert ns["SKY_ELEVATION_DEG"] == 10.0 and ns["WORKERS"] == 4 and ns["COLMAP_BAT"].endswith("COLMAP.bat")
    assert ns["MAX_NUM_FEATURES"] == 16000 and ns["MATCH"]["max_distance"] == 1.0 and len(ns["SCHEDULE"]) == 3
    assert ns["ATTITUDE_PRIOR_DEG"] == 5.0 and ns["SITES"]["sid"] == (360, 378) and ns["SITES"]["south_arm"] == (1408, 1412)
    full = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for k in ("workers=WORKERS", "block_size=MATCH_BLOCK_SIZE", "linear_solver=LINEAR_SOLVER", 'proj.settings["colmap_bat"]',
              "lmst_window_h", "max_saturated_fraction", "brightness_by_family", 'files.get("station_map")'):
        assert k in full, k
    assert "plt.show()" not in full.split("plot_camera_shifts(proj, rows=")[1][:300]


def test_right_only_exposure_gets_a_single_eye_rig(tmp_path):
    """v0p30: a right image whose left partner is missing is posed in a one-sensor rig, not dropped."""
    import pycolmap
    from test_sfm import _synthetic
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


def test_focal_temperature_fit_and_label_temperature():
    from mppp.sfm.calibration import focal_temperature_fit, _label_temperature
    from mppp.cmod import CameraModel
    rows = [{"camera": c, "temp_median_degC": t, "fx": 2950.0 + 0.02 * t + (0.5 if c == "NL" else 0.0)}
            for c in ("NL", "NR") for t in (-40.0, -20.0, 0.0, 10.0)]
    fit = focal_temperature_fit(rows)
    assert abs(fit["NL"]["px_per_degC"] - 0.02) < 1e-9 and abs(fit["NR"]["fx_at_0C"] - 2950.0) < 1e-6
    assert abs(fit["NR"]["ppm_per_degC"] - 0.02 / 2950.0 * 1e6) < 1e-6
    gcm = {"MODEL_TYPE": "CAHVORE", "MODEL_COMPONENT_1": [0, 0, 0], "MODEL_COMPONENT_2": [0, 0, 1],
           "MODEL_COMPONENT_3": [2950, 0, 2560], "MODEL_COMPONENT_4": [0, 2950, 1920],
           "MODEL_COMPONENT_5": [0, 0, 1], "MODEL_COMPONENT_6": [0, 0.05, -0.017], "MODEL_COMPONENT_7": [0, 0, 0],
           "MODEL_COMPONENT_8": 2.0, "MODEL_COMPONENT_9": 0.0,
           "INTERPOLATION_METHOD": "TEMPERATURE", "INTERPOLATION_VALUE": -18.0172}
    assert _label_temperature(CameraModel.from_label(gcm, 5120, 3840)) == -18.0172


def test_notebook_03_results_cells_start_with_the_site_banner():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    code_cells = [c.source for c in nb.cells if c.cell_type == "code"]
    k = next(i for i, s in enumerate(code_cells) if "def banner()" in s)
    assert "COLMAP_DIR = WORK / \"colmap\"" in code_cells[k]
    assert len(code_cells) - k - 1 >= 8 and all(s.startswith("banner()\n") for s in code_cells[k + 1:])
