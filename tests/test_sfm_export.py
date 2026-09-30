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
    from mppp.error.colmap import read_colmap
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
    assert len(read_colmap(str(tmp_path / "native")).points) == info["points"]      # mppp.error still reads it
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
