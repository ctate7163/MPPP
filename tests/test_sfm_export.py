"""Export for the error analysis (mppp.sfm.export). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import pytest
import numpy as np


def test_camera_shift_plot(tmp_path):
    pytest.importorskip("pyceres")
    import matplotlib
    matplotlib.use("Agg")
    from helpers import _build_rec, _synthetic
    from mppp.sfm.export import _nice, plot_camera_shifts, pose_residual_table
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    for r in proj.images:                                          # priors = truth, start perturbed
        r["prior_R_w2c"], r["prior_C"] = truth[r["name"]][0].tolist(), truth[r["name"]][1].tolist()
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=50)
    rows = pose_residual_table(rec, proj)
    fig = plot_camera_shifts(proj, rows=rows, out_png=tmp_path / "shifts.png")
    assert (tmp_path / "shifts.png").stat().st_size > 20000 and len(fig.axes) >= 2
    assert (_nice(37), _nice(0.23), _nice(1)) == (20, 0.2, 1)
    import csv
    with open(tmp_path / "p.csv", "w", newline="") as f:              # also from poses.csv (strings)
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot_camera_shifts(proj, rows=list(csv.DictReader(open(tmp_path / "p.csv"))), exaggeration=100)


def test_native_model_is_a_valid_colmap_model(tmp_path):
    pytest.importorskip("pyceres")
    import numpy as np
    import pycolmap
    from helpers import _build_rec, _synthetic
    from mppp.colmap import read_colmap
    from mppp.sfm.export import native_reconstruction, write_native_text_model
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    info = write_native_text_model(rec, proj, tmp_path / "native")
    back = pycolmap.Reconstruction(str(tmp_path / "native"))       # <= 0.14.4: 'Check failed: point2D.point3D_id'
    assert len(back.points3D) == info["points"] == len(rec.points3D) and len(back.images) == len(rec.images)
    assert {(c.width, c.height) for c in back.cameras.values()} == {(2560, 1920), (1280, 960)}
    res = []
    for pt in list(back.points3D.values())[:300]:
        for el in pt.track.elements:
            im = back.images[el.image_id]
            uv = back.cameras[im.camera_id].img_from_cam(im.cam_from_world() * pt.xyz)
            res.append(np.linalg.norm(uv - im.points2D[el.point2D_idx].xy))
    assert np.median(res) < 3 * noise                                  # native keypoints and native cameras agree
    assert len(read_colmap(str(tmp_path / "native")).points) == info["points"]      # the text reader reads it
    full = native_reconstruction(rec, proj, observed_only=False)
    assert sum(len(im.points2D) for im in full.images.values()) == sum(len(im.points2D) for im in rec.images.values())


def test_point_colors_from_tracks(tmp_path):
    pytest.importorskip("pycolmap")
    import cv2
    from helpers import _build_rec, _synthetic
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


def _short_stop(tmp_path, offset=(1.5, -1.0, 0.0)):
    """The synthetic block with one stereo frame of the second station relabelled as a two-image station whose
    waypoint prior is ``offset`` metres off."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import _synthetic_block
    proj, rec, noise = _synthetic_block(tmp_path)
    fid = next(rec.images[int(r["image_id"])].frame_id for r in proj.images if r["station"] == "S001D0100")
    ims = {rec.images[d.id].name for d in rec.frames[fid].data_ids}
    for r in proj.images:
        if r["name"] in ims:
            r["station"] = "S001D0150"
            r["prior_C"] = (np.asarray(r["prior_C"], float) + np.asarray(offset)).tolist()
    return proj, rec, noise, fid


def test_station_map_has_the_two_overview_panels_only(tmp_path):
    pytest.importorskip("pyceres")
    import matplotlib
    matplotlib.use("Agg")
    from mppp.sfm.export import plot_camera_shifts
    proj, rec, noise, fid = _short_stop(tmp_path)
    proj.settings["reconstruction"] = {"unlocalized_stations": ["S001D0150"]}
    fig = plot_camera_shifts(proj, rec=rec, out_png=tmp_path / "station_map.png")
    plotted = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    assert len(plotted) == 2 and (tmp_path / "station_map.png").is_file()
    assert any("(no prior)" in t.get_text() for t in fig.axes[0].texts)
    fig2 = plot_camera_shifts(proj, rec=rec, station_panels=True)
    assert len([a for a in fig2.axes if a.get_label() != "<colorbar>"]) == 2 + 3


def test_epsilon_from_residuals_splits_intra_and_cross_station_points():
    """eps = rms / sqrt(2) over the observations of points seen from one station (intra) and from several (cross);
    non-finite residuals (behind the camera) are left out; the keys are those of summary.json's "eps"."""
    from mppp.sfm.export import epsilon_from_residuals
    names = {1: "a.png", 2: "b.png", 3: "c.png"}
    stations = {"a.png": "S1", "b.png": "S1", "c.png": "S2"}
    res = {"residual_native_px": np.array([1.0, 1.0, 2.0, 2.0, 2.0, np.inf]),
           "image_id": np.array([1, 2, 1, 2, 3, 3]),
           "point3D_id": np.array([10, 10, 11, 11, 11, 12]), "point2D_idx": np.zeros(6, int)}
    e = epsilon_from_residuals(res, names, stations)
    assert e["n_intra_obs"] == 2 and e["n_cross_obs"] == 3
    assert e["eps_intra_px"] == pytest.approx(1 / np.sqrt(2)) and e["eps_cross_px"] == pytest.approx(2 / np.sqrt(2))
    assert e["resid_rms_cross_px"] == pytest.approx(2.0) and e["eps_ratio"] == pytest.approx(2.0)
    assert set(e) == {"eps_intra_px", "eps_cross_px", "n_intra_obs", "n_cross_obs", "resid_rms_intra_px",
                      "resid_rms_cross_px", "eps_ratio"}
    empty = {k: v[:0] for k, v in res.items()}
    assert np.isnan(epsilon_from_residuals(empty, names, stations)["eps_intra_px"])


def test_export_for_error_computes_eps_from_its_own_residuals(tmp_path):
    """summary.json's "eps" comes from the residuals the export holds (v0p90), not from re-reading the text model;
    it equals the per-observation calibration of the error model on that model."""
    pytest.importorskip("pyceres")
    import json
    from helpers import _build_rec, _synthetic
    from mppp.colmap import read_colmap
    from mppp.error.colmap import calibrate_eps_from_residuals
    from mppp.sfm.export import export_for_error
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    summary = export_for_error(proj, rec, tmp_path / "ei")
    eps = json.loads((tmp_path / "ei" / "summary.json").read_text())["eps"]
    assert eps["n_intra_obs"] > 0 and eps["n_cross_obs"] > 0 and 0.5 * noise < eps["eps_intra_px"] < 2 * noise
    m = read_colmap(str(tmp_path / "ei" / "native"))
    m.assign_stations({r["name"]: r["station"] for r in proj.images})
    ref = calibrate_eps_from_residuals(m)
    for k in ("eps_intra_px", "eps_cross_px", "n_intra_obs", "n_cross_obs"):
        assert summary["eps"][k] == pytest.approx(ref[k], rel=1e-9)


def test_stations_csv_records_the_sun_and_the_block_frame(tmp_path):
    """0.95.0 (Christian, 27 Oct): stations.csv carries the Sun-Mars distance, the label sun vector in ENU and the
    same vector in the block frame; block_frame.json the block's rotation from the label ENU frame."""
    pytest.importorskip("pyceres")
    import csv
    import json
    from helpers import _build_rec, _synthetic
    from mppp.sfm.export import export_for_error
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    for k, r in enumerate(proj.images):                                   # the label values (no manifest sun facts)
        r["solar_azimuth_deg"], r["solar_elevation_deg"] = 90.0, 30.0
        if k == 0:
            r["solar_elevation_deg"] = None                               # one without: empty columns
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    export_for_error(proj, rec, tmp_path / "ei")
    bf = json.loads((tmp_path / "ei" / "block_frame.json").read_text())
    assert bf["images"] > 0 and bf["angle_deg"] < 0.5 and len(bf["R_block_from_enu"]) == 3     # the truth: no tilt
    rows = list(csv.DictReader((tmp_path / "ei" / "stations.csv").open(encoding="utf-8")))
    assert {"d_mars_au", "d_mars_source", "sun_e", "sun_n", "sun_u", "sun_bx", "sun_by", "sun_bz"} <= set(rows[0])
    r1 = rows[1]
    assert float(r1["sun_e"]) == pytest.approx(np.cos(np.radians(30)), abs=1e-9) and float(r1["sun_u"]) == pytest.approx(0.5)
    assert abs(float(r1["sun_bx"]) - float(r1["sun_e"])) < 0.01 and abs(float(r1["sun_bz"]) - 0.5) < 0.01
    assert rows[0]["sun_e"] == "" and rows[0]["sun_bx"] == ""
    assert rows[1]["d_mars_source"] in ("ls_ellipse", "") and (rows[1]["d_mars_au"] == "" or 1.3 < float(rows[1]["d_mars_au"]) < 1.7)


def test_export_level_light_leaves_out_the_native_model_and_residuals(tmp_path):
    """0.95.0 (Christian, 28 Oct): a calibration run's export keeps the summary, the tables and the block frame
    and leaves out the native-pixel text model and the residuals (the joints read sparse/<model>, project.json and
    the health report); the setting is not in the run key, and run_pipeline's site runs are light unless --export."""
    pytest.importorskip("pyceres")
    import json
    from helpers import _build_rec, _synthetic
    from mppp.sfm.export import export_for_error
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    s = export_for_error(proj, rec, tmp_path / "ei", level="light")
    ei = tmp_path / "ei"
    assert s["native_model"] == {"skipped": "export level light"} and not (ei / "native").exists()
    assert not (ei / "residuals.npz").exists()
    assert all((ei / f).is_file() for f in ("summary.json", "stations.csv", "poses.csv", "block_frame.json"))
    assert json.loads((ei / "summary.json").read_text())["eps"]["n_intra_obs"] > 0      # eps from the residuals held
    from mppp.pipeline import NOT_IN_KEY, Settings
    assert Settings().export_level == "full" and "export_level" in NOT_IN_KEY
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "studies" / "calibration" / "run_pipeline.py").read_text(encoding="utf-8")
    assert '"--export"' in src and "export_level=light" in src


def test_dense_workspace_preloaded_from_the_native_model(tmp_path, monkeypatch):
    """0.95.0 (Christian, 28 Oct): prepare_dense undistorts the images with the native-pixel model into
    <project>/dense (pinhole cameras, COLMAP's stereo configs), undistorts the rover masks the same way for the
    fusion (binary again), records it; dense_reconstruct.bat next to open_in_colmap.bat runs the steps with a CUDA
    COLMAP; write_gui_project writes the batch file; mppp dense --prepare does the workspace."""
    pytest.importorskip("pyceres")
    import cv2
    import json
    import pycolmap
    from helpers import _build_rec, _synthetic, _write_database
    from mppp.sfm.database import write_gui_project
    from mppp.sfm.dense import DENSE_MAX_IMAGE_SIZE, prepare_dense, write_dense_bat
    from mppp.sfm.export import write_gui_native
    assert DENSE_MAX_IMAGE_SIZE == 2000
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    proj.images_dir.mkdir(parents=True, exist_ok=True)
    proj.masks_dir.mkdir(parents=True, exist_ok=True)
    masked = []
    for k, r in enumerate(proj.images):
        w, h = r["native_size"]
        img = rng.integers(60, 200, (h, w), dtype=np.uint8)
        cv2.imwrite(str(proj.images_dir / r["name"]), img)
        if k < 4:                                                        # four images with a mask: the lower half out
            m = np.full((h, w), 255, np.uint8)
            m[h // 2:, :] = 0
            cv2.imwrite(str(proj.masks_dir / (r["name"] + ".png")), m)
            r["has_mask"] = True
            masked.append(r["name"])
    _write_database(rec, proj.database)
    proj.settings["reconstruction"] = {"path": "sparse/cahv_ba"}
    (proj.root / "sparse" / "cahv_ba").mkdir(parents=True, exist_ok=True)
    rec.write(str(proj.root / "sparse" / "cahv_ba"))
    proj.save()
    write_gui_native(proj, rec, colors=False)
    res = prepare_dense(proj, max_image_size=400)
    ws = proj.root / "dense"
    assert res["workspace"] == str(ws) and res["images"] == rec.num_reg_images() and res["max_image_size"] == 400
    und = pycolmap.Reconstruction(str(ws / "sparse"))
    assert all(c.model.name == "PINHOLE" and max(c.width, c.height) <= 400 for c in und.cameras.values())
    assert (ws / "stereo" / "patch-match.cfg").is_file() and (ws / "stereo" / "fusion.cfg").is_file()
    assert len(list((ws / "images").iterdir())) == rec.num_reg_images()
    assert res["masks"]["images"] == 4 and all((ws / "masks" / (n + ".png")).is_file() for n in masked)
    m = cv2.imread(str(ws / "masks" / (masked[0] + ".png")), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(m).tolist()) <= {0, 255} and m.shape[1] <= 400 and m[-1].max() == 0 and m[0].max() == 255
    assert not (ws / "_masks_in").exists() and not (ws / "_masks_out").exists()
    assert json.loads((ws / "mppp_dense.json").read_text())["images"] == res["images"]
    assert prepare_dense(proj, max_image_size=400)["seconds"] == res["seconds"]       # there already: not redone
    bat = write_dense_bat(proj)
    b = bat.read_bytes()
    assert bat.name == "dense_reconstruct.bat" and b.count(b"\n") == b.count(b"\r\n")
    t = b.decode()
    assert "image_undistorter" in t and "patch_match_stereo" in t and "stereo_fusion" in t and "poisson_mesher" in t
    assert "--StereoFusion.mask_path" in t and "MPPP_DENSE_STEP" in t and "MPPP_DENSE_MAX_SIZE=2000" in t
    assert t.index("colmap-x64-windows-cuda") < t.index("colmap-x64-windows-nocuda")      # a CUDA COLMAP first
    out = write_gui_project(proj, model="cahv_ba")
    assert out["bat_dense"] == str(bat)
    from mppp.cli import main
    assert main(["dense", str(proj.root), "--prepare", "--max-image-size", "300", "--overwrite"]) == 0
    assert json.loads((ws / "mppp_dense.json").read_text())["max_image_size"] == 300


def test_dense_maps_as_images_and_the_bat_prepares_the_masks(tmp_path):
    """0.95.1 (Christian, 29 Oct): COLMAP's depth / normal maps (text header w&h&c&, float32 column-major) as PNG or
    JPEG in dense/stereo/maps_png; dense_reconstruct.bat prepares the workspace with MPPP's Python (the masks) before
    falling back to COLMAP's undistorter, and does not rewrite itself."""
    import numpy as np
    from PIL import Image
    from mppp.sfm.dense import export_dense_maps, read_colmap_array

    def write(path, a):
        a = np.asarray(a, np.float32)
        h, w = a.shape[:2]
        c = 1 if a.ndim == 2 else a.shape[2]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(f"{w}&{h}&{c}&".encode())
            f.write(np.transpose(a.reshape(h, w, c), (1, 0, 2)).astype(np.float32).tobytes(order="F"))
    ws = tmp_path / "dense"
    d = np.tile(np.linspace(2.0, 10.0, 40, dtype=np.float32), (30, 1))
    d[:5, :5] = 0.0
    write(ws / "stereo" / "depth_maps" / "NLF_0001.png.geometric.bin", d)
    nm = np.zeros((30, 40, 3), np.float32); nm[..., 2] = -1.0
    write(ws / "stereo" / "normal_maps" / "NLF_0001.png.geometric.bin", nm)
    assert np.allclose(read_colmap_array(ws / "stereo" / "depth_maps" / "NLF_0001.png.geometric.bin"), d)
    r = export_dense_maps(ws)
    assert r["depth_maps"] == 1 and r["normal_maps"] == 1
    files = sorted(p.name for p in (ws / "stereo" / "maps_png").iterdir())
    assert files[0].startswith("NLF_0001.depth_") and files[0].endswith("m.png") and files[1] == "NLF_0001.normal.png"
    im = np.asarray(Image.open(ws / "stereo" / "maps_png" / files[0]))
    assert im.shape == (30, 40, 3) and (im[:5, :5] == 0).all() and im[10:, 6:].any()
    assert export_dense_maps(ws, fmt="jpg")["depth_maps"] == 1
    from mppp.sfm.dense import write_dense_bat

    class P:
        root = tmp_path
        settings: dict = {}
    t = write_dense_bat(P()).read_text()
    assert "mppp dense \"%~dp0.\" --prepare --no-bat" in t and "mppp_env.bat" in t and ":prepared" in t
    assert t.index("--no-bat") < t.index("image_undistorter")
