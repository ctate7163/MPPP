"""v0p22: outlier frames, convergence statistics, triangulation options, rig translation prior,
best start cameras (focus model, consensus rig), >= 3-image tracks in the error analysis, batch runner."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _solved(tmp_path, seed=0):
    """The synthetic two-station rig block of test_sfm at its true poses, with a database camera map."""
    pytest.importorskip("pycolmap")
    from test_sfm import _build_rec, _synthetic
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path, seed=seed)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, np.random.default_rng(seed + 1), perturb=False)
    return proj, rec, truth


def test_find_outlier_frames_flags_a_frame_that_left_its_station(tmp_path):
    from mppp.sfm.reconstruction import OUTLIER_DEFAULTS, exclude_frames, find_outlier_frames
    proj, rec, _ = _solved(tmp_path)
    assert find_outlier_frames(rec, proj) == []                       # a consistent block: nothing flagged
    # frame 3's priors are 1.5 m off (its solved pose did not follow them: a frame that "did not align")
    bad = next(fid for fid in rec.frames if fid == 3)
    names = {rec.images[d.id].name for d in rec.frames[bad].data_ids}
    for r in proj.images:
        if r["name"] in names:
            r["prior_C"] = (np.asarray(r["prior_C"]) + [1.5, 0.0, 0.0]).tolist()
    found = find_outlier_frames(rec, proj)
    assert [f["frame_id"] for f in found] == [bad]
    assert any("shift" in s for s in found[0]["reasons"]) and found[0]["family"] == "N"
    assert set(found[0]["images"]) == names
    # a tilt common to a whole station (a rover attitude error) is not an outlier; one frame's is
    from scipy.spatial.transform import Rotation
    for r in proj.images:
        if r["name"] in names:
            r["prior_C"] = (np.asarray(r["prior_C"]) - [1.5, 0.0, 0.0]).tolist()
    dR = Rotation.from_rotvec(np.radians([0.0, 0.0, 2.0])).as_matrix()
    for r in proj.images:
        if r["station"] == "S001D0000":
            r["prior_R_w2c"] = (np.asarray(r["prior_R_w2c"]) @ dR).tolist()
    assert find_outlier_frames(rec, proj) == []
    for r in proj.images:
        if r["name"] in names:
            r["prior_R_w2c"] = (np.asarray(r["prior_R_w2c"]) @ dR).tolist()
    found = find_outlier_frames(rec, proj)
    assert [f["frame_id"] for f in found] == [bad] and "attitude" in found[0]["reasons"][0]
    # thresholds can be relaxed; min_observations flags weakly tied frames
    assert find_outlier_frames(rec, proj, min_attitude_deg=5.0) == []
    many = find_outlier_frames(rec, proj, min_attitude_deg=5.0, min_observations=10 ** 6)
    assert len(many) == len(rec.frames)
    assert set(OUTLIER_DEFAULTS) >= {"residual_factor", "min_observations", "shift_mad_factor"}
    # exclusion deregisters the frame and its images
    n_img = rec.num_reg_images()
    assert exclude_frames(rec, [bad]) == 1 and exclude_frames(rec, [bad]) == 0
    assert rec.num_reg_images() == n_img - 2 and bad not in set(rec.reg_frame_ids())


def test_find_outlier_frames_flags_large_residuals(tmp_path):
    from scipy.spatial.transform import Rotation
    import pycolmap
    from mppp.sfm.reconstruction import find_outlier_frames
    proj, rec, _ = _solved(tmp_path)
    fr = rec.frames[5]
    T = fr.rig_from_world
    dR = Rotation.from_rotvec(np.radians([0.3, 0.0, 0.0])).as_matrix()
    fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(dR @ T.rotation.matrix()), dR @ np.asarray(T.translation))
    found = find_outlier_frames(rec, proj)
    assert 5 in [f["frame_id"] for f in found]
    f5 = next(f for f in found if f["frame_id"] == 5)
    assert any("median residual" in s for s in f5["reasons"]) and f5["median_residual_px"] > 5


def test_convergence_statistics_counts_ray_angles(tmp_path):
    from mppp.sfm.reconstruction import CONVERGENCE_BINS_DEG, convergence_statistics
    proj, rec, _ = _solved(tmp_path)
    cs = convergence_statistics(rec, proj)
    n2 = sum(1 for p in rec.points3D.values() if p.track.length() >= 2)
    assert sum(cs["points"]) == n2 and cs["bins_deg"] == list(CONVERGENCE_BINS_DEG)
    assert all(a <= b for a, b in zip(cs["cross_station_points"], cs["points"]))
    assert cs["points_over_10deg"] == sum(c for c, lo in zip(cs["points"], cs["bins_deg"]) if lo >= 10)
    assert cs["observations_over_10deg"] >= 2 * cs["points_over_10deg"]
    # the check by hand on one cross-station point
    C = {i: np.asarray(im.projection_center()) for i, im in rec.images.items()}
    pid, pt = next((k, p) for k, p in rec.points3D.items()
                   if len({rec.images[e.image_id].name.split("_")[1] for e in p.track.elements}) > 4)
    d = np.array([pt.xyz - C[e.image_id] for e in pt.track.elements])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert np.degrees(np.arccos(np.clip(d @ d.T, -1, 1).min())) > 0
    s = convergence_statistics(rec, proj, max_points=500)
    assert s["sampled"] and abs(sum(s["points"]) - n2) < 0.05 * n2       # sample scaled to all points


def test_triangulate_passes_the_track_options(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm import reconstruction as R
    seen = {}

    def fake(rec, db, images, out, clear_points, options, refine_intrinsics):
        seen["t"] = options.triangulation
        return rec
    monkeypatch.setattr(pycolmap, "triangulate_points", fake)
    proj, rec, _ = _solved(tmp_path)
    R.triangulate(rec, proj, max_reproj_px=6.0, min_angle_deg=0.25, max_transitivity=3, create_max_angle_error_deg=4.0,
                  continue_max_angle_error_deg=3.0, complete_max_transitivity=8)
    t = seen["t"]
    assert (t.max_transitivity, t.complete_max_transitivity) == (3, 8)
    assert (t.create_max_angle_error, t.continue_max_angle_error) == (4.0, 3.0)
    assert t.merge_max_reproj_error == 6.0 and t.min_angle == 0.25 and not t.ignore_two_view_tracks


def test_rig_translation_prior_keeps_the_baseline(tmp_path):
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig=True, max_iterations=100,
                        rig_translation_sigma_m=0.002)
    assert out["rig_translation_priors"] == 1
    assert list(out["rig"]) == ["1:2"]
    rig = out["rig"]["1:2"]
    assert abs(rig["baseline_m"] - 0.4244) < 0.002 and abs(rig["baseline_start_m"] - 0.4244) < 1e-6
    assert np.linalg.norm(rig["centre_change_m"]) < 0.003
    out2 = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=20)
    assert out2["rig_translation_priors"] == 0 and abs(out2["rig"]["1:2"]["baseline_m"] - rig["baseline_m"]) < 1e-9


def test_focus_model_start_and_hold():
    from mppp.sfm.project import ZCAM_HOLD_F_IMAGES, _split_by_focus, zcam_focus_model
    m = zcam_focus_model()
    g = m["cameras"]["ZL034"]
    assert ZCAM_HOLD_F_IMAGES == 2 and g["focus_range"] == [-2000, 1300]
    base = {"model": "FULL_OPENCV", "params": [4680.0, 4680.0, 824, 600, -0.02, 0.0, 0, 0, 0, 0, 0, 0],
            "source": "median label CAHVOR", "fixed_params": []}
    rows = []
    for foc, n, lf in ((600, 5, 4680.0), (1000, 1, 4700.0), (-2600, 3, 4600.0)):
        rows += [{"camera_group": "ZL034", "focus_count": foc, "label_f_px": lf} for _ in range(n)]
    out = _split_by_focus(rows, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal", model=m, hold_f_images=2)
    by = {c["focus_count_median"]: c for c in out.values()}
    f600 = np.sqrt(by[600]["params"][0] * by[600]["params"][1])
    assert abs(f600 - (g["f0_px"] + g["slope_px_per_count"] * (600 - g["reference_focus"]))) < 0.01
    assert abs(by[600]["params"][1] / by[600]["params"][0] - g["aspect"]) < 1e-9
    assert "fx" not in by[600]["fixed_params"] and {"fx", "fy"} <= set(by[1000]["fixed_params"])
    # outside the fitted range: the bin's label f scaled by the refined/label ratio, not an extrapolation
    f_out = np.sqrt(by[-2600]["params"][0] * by[-2600]["params"][1])
    assert abs(f_out - 4600.0 * g["f0_px"] / g["label_f0_px"]) < 0.01 and "outside" in by[-2600]["source"]
    assert "fx" not in by[-2600]["fixed_params"]
    lab = _split_by_focus(rows, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal")      # without a model: label f
    assert {round(c["params"][0]) for c in lab.values()} == {4680, 4700, 4600}


def test_shipped_rig_and_rational_cameras_agree():
    from mppp.paths import data_dir
    from mppp.sfm.project import NAVCAM_RIG, NAVCAM_RIG_FILE
    from scipy.spatial.transform import Rotation
    d = data_dir() / "m20_cmods"
    rig = json.loads((d / NAVCAM_RIG_FILE).read_text())
    assert NAVCAM_RIG == "consensus"
    R = np.asarray(rig["R_sensor_from_ref"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))
    assert 0.05 < ang < 0.5                                   # the two Navcams are toed by ~0.1-0.2 deg
    for e in "LR":
        c = json.loads((d / f"M2020_N{e}_rational.json").read_text())
        p = c["params"] if "params" in c else c["camera"]["params"]
        assert 2900 < p[0] < 3000 and 0.2 < p[9] < 0.7


def test_error_analysis_keeps_tracks_of_three_or_more(tmp_path, monkeypatch):
    from test_v0p15 import _synthetic_model
    from mppp.error import alignment as A
    model, X = _synthetic_model(noise_px=0.2)
    for pid in list(model.points)[:80]:                      # 80 points seen by two images only
        p = model.points[pid]
        p.image_ids, p.point2D_idxs = p.image_ids[:2], p.point2D_idxs[:2]
    e = tmp_path / "work" / "colmap" / "error_input"
    (e / "native").mkdir(parents=True)
    (e / "stations.csv").write_text("name,station,instrument\n" + "".join(
        f"img{k}.png,{'A' if k <= 3 else 'B'},NL\n" for k in range(1, 7)))
    import copy
    monkeypatch.setattr(A, "read_colmap", lambda path: copy.deepcopy(model))
    a2 = A.load_alignment(tmp_path / "work", min_track_length=2)
    a3 = A.load_alignment(tmp_path / "work", min_track_length=3)
    assert len(a2.model.points) == 200 and len(a3.model.points) == 120
    assert a3.points_before_track_filter == 200 and a3.min_track_length == 3
    rows = A.eps_table(a3)
    top = rows[0]
    assert {"n_points", "eps_dof_px"} <= set(top) and top["n_points"] == 120
    # 120 points x 6 observations: eps_dof = eps sqrt(2N / (2N - 3P)) with N = 720, P = 120
    assert abs(top["eps_dof_px"] / top["eps_px"] - np.sqrt(1440 / (1440 - 360))) < 1e-9


def test_run_scapes_injects_after_the_parameters_cell():
    nbformat = pytest.importorskip("nbformat")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_scapes
    for name in ("03_colmap_alignment", "05_error_analysis", "04_camera_models"):
        nb = nbformat.read(str(ROOT / "notebooks" / f"{name}.ipynb"), as_version=4)
        i = next(i for i, c in enumerate(nb.cells) if "parameters" in c.metadata.get("tags", []))
        run_scapes.inject(nb, {"SITE": "'belva'", "X": "1"})
        assert nb.cells[i + 1].metadata["tags"] == ["injected-parameters"]
        assert "SITE = 'belva'\nX = 1" in nb.cells[i + 1].source
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("a = 1")])
    with pytest.raises(ValueError):
        run_scapes.inject(nb, {"a": "2"})


def test_notebook_03_settings_use_the_latest_methods():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)   # the cell is plain assignments
    assert ns["NAVCAM_DISTORTION"] == "rational" and ns["NAVCAM_RIG"] == "consensus"
    assert ns["ZCAM_INTRINSICS"] == "focus_model" and ns["EXCLUDE_OUTLIERS"] is True
    assert ns["SITES"]["threeforks_south"] == (652, 683)
    assert {"taylorfjellet", "rockytop", "belva_crater", "butler_landing"} <= set(ns["SITES"])
    full = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for k in ("exclude_outliers=EXCLUDE_OUTLIERS", "triangulation_options=TRIANGULATION", "navcam_rig=NAVCAM_RIG",
              "convergence_statistics", "match(", "**MATCH"):
        assert k in full, k


def test_affine_shape_is_part_of_the_feature_record():
    from mppp.sfm.database import _feature_settings
    assert "estimate_affine_shape" not in _feature_settings(16380, 5120, False)       # older records stay valid
    assert _feature_settings(16380, 5120, True, True)["estimate_affine_shape"] is True
