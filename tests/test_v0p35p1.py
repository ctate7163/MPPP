"""v0p35.1: notebook 03 camera print-out after the thermal stage."""
import re
from types import SimpleNamespace

import pytest

pycolmap = pytest.importorskip("pycolmap")

from mppp.sfm.reconstruction import camera_changes, print_camera_changes  # noqa: E402


def _rec():
    rec = pycolmap.Reconstruction()
    p = [2956.0, 2956.0, 2591.0, 1944.0, 0.3, -0.02, 0.0, 0.0, 0.002, 0.59, 0.0, 0.0]
    for cid in (1, 2, 3, 4):
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="THIN_PRISM_FISHEYE", width=5120, height=3840,
                                       params=[x * (1 + 0.001 * cid) for x in p]))
    return rec, p


def test_eye_camera_without_images_after_thermal_split(capsys):
    rec, p = _rec()
    proj = SimpleNamespace(
        cameras={"NL": {"params": p}, "NR": {"params": p},
                 "NL_T10": {"params": p, "group": "NL", "thermal_bin": True},
                 "ZL_f1": {"params": p, "group": "ZL"}},
        # every Navcam image moved to a bin camera: the old notebook cell raised IndexError here
        images=[{"name": "a", "instrument": "NL_T10", "base_instrument": "NL", "camera_id": 1}],
        settings={"database": {"cameras": {"NL": 1, "NR": 2, "NL_T10": 3, "ZL_f1": 4}}})
    rows = camera_changes(rec, proj)
    assert [r["camera"] for r in rows] == ["NL", "NR", "NL_T10"]          # Mastcam-Z focus bin left out
    assert rows[0]["names"][-2:] == ["sx1", "sy1"] and rows[0]["images"] == 0
    assert rows[2]["thermal_bin"] and rows[2]["refined"][0] == pytest.approx(p[0] * 1.003)
    print_camera_changes(rec, proj)
    out = capsys.readouterr().out
    assert "no images after the thermal split" in out and "[temperature bin]" in out


def test_camera_missing_from_reconstruction():
    rec, p = _rec()
    proj = SimpleNamespace(cameras={"NL": {"params": p}, "NX": {"params": p}}, images=[],
                           settings={"database": {"cameras": {"NL": 1}}})
    rows = camera_changes(rec, proj)
    assert rows[1] == {"camera": "NX", "missing": True}


# ------------------------------------------------------------------ v0p35.1 rig drift
import math  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from mppp.sfm import navcal as NC  # noqa: E402
from mppp.sfm.project import start_rig_rotation  # noqa: E402


def _study(name, sol, T, pitch, yaw, roll, R_ref=None, sd=0.05):
    R_ref = np.eye(3) if R_ref is None else R_ref
    R = Rotation.from_rotvec(np.radians(np.array([pitch, yaw, roll]) * 1e-3)).as_matrix() @ R_ref
    rel = NC.rig_angles(R, R_ref)
    ab = {k.replace("_mdeg", "_abs_mdeg"): v for k, v in NC.rig_angles(R).items()}
    row = {**rel, **ab, "sd_pitch_mdeg": sd, "sd_yaw_mdeg": sd, "sd_roll_mdeg": sd}
    return {"scape": name, "R_reference": R_ref.tolist(),
            "network": {"sol_median": float(sol), "T_median_degC": float(T)},
            "rotation_pp": {"rigs": [dict(row)]}, "rotation": {"rigs": [dict(row)]}}


def _synthetic(rate=0.005, early=0.025, K=300.0, kT=-1.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, sol in enumerate([60, 180, 240, 360, 480, 700, 800, 900, 1000, 1100, 1200, 1330, 1400, 1460, 1630, 1780, 1970]):
        T = -18 + 4 * math.sin(i)
        p = 1.0 + rate * sol + (early - rate) * min(sol, K) + rng.normal(0, 0.05)
        y = -5.0 + kT * (T + 18) + rng.normal(0, 0.05)
        r = 20.0 - 0.005 * sol + rng.normal(0, 0.05)
        out.append(_study(f"b{i}", sol, T, p, y, r))
    return out


def test_rig_drift_model_recovers_rate_and_hinge():
    st = _synthetic()
    m = NC.rig_drift_model(st, knot_sol=300.0, no_drift=("yaw",))
    assert m["pitch_fit_mdeg_per_sol"] == pytest.approx(0.005, abs=3e-4)
    assert m["pitch_early_mdeg_per_sol"] == pytest.approx(0.025, abs=2e-3)
    assert m["yaw_mdeg_per_sol"] == 0.0 and m["yaw_T_mdeg_per_degC"] == pytest.approx(-1.0, abs=0.05)
    assert m["roll_mdeg_per_sol"] == pytest.approx(-0.005, abs=3e-4)
    # the drift averages to ~0 over the reference blocks (the joint rig applies at their mean)
    offs = [NC.drift_offset_mdeg(m, s["network"]["sol_median"])["pitch_mdeg"] for s in st]
    assert abs(np.mean(offs)) < 0.05


def test_start_rig_rotation_applies_the_hinge():
    st = _synthetic()
    m = NC.rig_drift_model(st, knot_sol=300.0)
    shipped = {"R_sensor_from_ref": np.eye(3).tolist(), "drift": m}
    for sol in (63.0, 1084.0, 1970.0):
        R, applied = start_rig_rotation(shipped, None, sol)
        ang = NC.rig_angles(R)
        exp = NC.drift_offset_mdeg(m, sol)
        for a in ("pitch", "yaw", "roll"):
            assert ang[f"{a}_mdeg"] == pytest.approx(exp[f"{a}_mdeg"], abs=1e-6)
        assert isinstance(applied["drift"]["offset_mdeg"]["pitch"], float)
    # a v0p35 rig file (rates only) still works
    R, _ = start_rig_rotation({"R_sensor_from_ref": np.eye(3).tolist(),
                               "drift": {"sol0": 1000.0, "pitch_mdeg_per_sol": 0.005}}, None, 1200.0)
    assert NC.rig_angles(R)["pitch_mdeg"] == pytest.approx(1.0, abs=1e-6)


def test_common_reference_removes_label_step():
    st = _synthetic(early=0.005)
    step = Rotation.from_rotvec(np.radians([0.23e-3, 0.0, 0.58e-3])).as_matrix()     # the early label rig
    early = []
    for s in st[:3]:
        r = s["rotation_pp"]["rigs"][0]
        early.append(_study(s["scape"], s["network"]["sol_median"], s["network"]["T_median_degC"],
                            r["pitch_mdeg"], r["yaw_mdeg"], r["roll_mdeg"]))
        # the same physical rig measured against a different label reference
        R = Rotation.from_rotvec(np.radians(np.array([r["pitch_mdeg"], r["yaw_mdeg"], r["roll_mdeg"]]) * 1e-3)).as_matrix()
        e = early[-1]
        e["R_reference"] = step.tolist()
        rel = NC.rig_angles(R, step)
        e["rotation_pp"]["rigs"][0].update(rel)
    mixed = early + st[3:]
    before = [s["rotation_pp"]["rigs"][0]["roll_mdeg"] for s in mixed[:3]]
    NC.common_reference(mixed)
    after = [s["rotation_pp"]["rigs"][0]["roll_mdeg"] for s in mixed[:3]]
    truth = [s["rotation_pp"]["rigs"][0]["roll_mdeg"] for s in st[:3]]
    assert np.allclose(after, truth, atol=1e-6) and not np.allclose(before, truth, atol=0.1)
    assert mixed[0]["reference_offset_mdeg"]["roll_mdeg"] == pytest.approx(0.58, abs=1e-3)


def test_drift_robustness_holdout_and_jackknife():
    st = _synthetic(early=0.005)
    rob = NC.drift_robustness(st, holdout=["b0", "b5"])
    assert rob["pitch_jackknife"]["sign_stable"] and rob["roll_jackknife"]["sign_stable"]
    for p in rob["holdout"]["predictions"]:
        assert abs(p["pitch"]["z"]) < 4


def test_fit_camera_model_round_trip():
    cam = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840,
                          params=[2956.3, 2955.9, 2591.1, 1943.5, 0.3054, -0.02596, 1.6e-4, 1.9e-4, 0.002, 0.5934, 0, 0])
    f, r1 = NC.fit_camera_model(cam, "THIN_PRISM_FISHEYE")
    back, r2 = NC.fit_camera_model(f, "FULL_OPENCV")
    assert f.model.name == "THIN_PRISM_FISHEYE" and back.model.name == "FULL_OPENCV"
    assert r1 < 0.5 and r2 < 0.5
    assert np.allclose(back.params[2:4], cam.params[2:4], atol=0.2)


def test_notebooks_v0p35p1():
    nbformat = pytest.importorskip("nbformat")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "notebooks"
    nb3 = nbformat.read(str(root / "03_colmap_alignment.ipynb"), as_version=4)
    full3 = "\n".join(c.source for c in nb3.cells)
    assert ("0.35." in nb3.cells[0].source or "0.4" in nb3.cells[0].source) and "print_camera_changes(rec, proj)" in full3
    assert '[r["camera_id"] for r in proj.images if r["instrument"] == k' not in full3
    for site in ("van_zyl", "seitah_north", "whale_mountain", "origny", "(1880, 1889)"):
        assert site in full3, site
    assert ("navcal_v0p35p1" in full3 or "navcal_v0p4" in full3) and "holds a block of sols" in full3
    assert re.search(r"LOCALIZE_MIN_IMAGES\s*= [34]\b", full3) and "localize_min_images=LOCALIZE_MIN_IMAGES" in full3
    nb4 = nbformat.read(str(root / "04_camera_models.ipynb"), as_version=4)
    full4 = "\n".join(c.source for c in nb4.cells)
    for k in ("2f  Rig drift with more blocks", "rig_drift_model(", "drift_robustness(", "drift_figure("):
        assert k in full4, k
    # 2f (which defines DRIFT) comes before 2e (which writes it)
    heads = [c.source.splitlines()[0] for c in nb4.cells if c.cell_type == "markdown" and c.source.startswith("### 2")]
    assert heads.index(next(h for h in heads if h.startswith("### 2f"))) < heads.index(next(h for h in heads if h.startswith("### 2e")))


def test_scale_offsets_recover_an_injected_shift(tmp_path):
    pytest.importorskip("pyceres")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_v0p31 import _synthetic_block
    from mppp.sfm.reconstruction import bundle_adjust
    proj, rec, noise = _synthetic_block(tmp_path)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=20)
    # the frames of one station at half resolution, with their keypoints shifted by (+1.0, -0.5) full-res px
    st = sorted({r["station"] for r in proj.images})[0]
    half = {int(r["image_id"]) for r in proj.images if r["station"] == st}
    for r in proj.images:
        r["downsample_scale"] = 0.5 if int(r["image_id"]) in half else 1.0
    for iid in half:
        for p in rec.images[iid].points2D:          # in place (the list holds references)
            p.xy = np.asarray(p.xy) + np.array([1.0, -0.5])
    iid = next(iter(half))
    assert rec.images[iid].points2D[0].xy[0] != 0.0
    sc = NC.Scape("synthetic", tmp_path, proj, rec, {})
    rec2, proj2, rows = NC.split_by_scale(rec, proj, min_images=2)
    assert {r["scale"] for r in rows} == {0.5, 1.0} and rec2.num_reg_images() == rec.num_reg_images()
    out = NC.scale_offsets(sc, min_images=2, max_iterations=30, verbose=False)
    got = [r for r in out["rows"] if "dcx_px" in r]
    assert got
    for r in got:
        sign = 1.0 if r["scale"] == 0.5 else -1.0          # offset of the half-resolution camera from the full one
        assert r["dcx_px"] * sign == pytest.approx(1.0, abs=0.15)
        assert r["dcy_px"] * sign == pytest.approx(-0.5, abs=0.15)


def test_notebook04_has_resolution_section():
    nbformat = pytest.importorskip("nbformat")
    from pathlib import Path
    nb4 = nbformat.read(str(Path(__file__).resolve().parents[1] / "notebooks" / "04_camera_models.ipynb"), as_version=4)
    full4 = "\n".join(c.source for c in nb4.cells)
    for k in ("2g  Pixel offsets between the Navcam resolutions", "label_scale_consistency(", "scale_summary("):
        assert k in full4, k


# ------------------------------------------------------------------ v0p35.2 LOCALIZE_MIN_IMAGES, station map
def _short_stop(tmp_path, offset=(1.5, -1.0, 0.0)):
    """The synthetic block with one stereo frame of the second station relabelled as a two-image station whose
    waypoint prior is ``offset`` metres off."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_v0p31 import _synthetic_block
    proj, rec, noise = _synthetic_block(tmp_path)
    fid = next(rec.images[int(r["image_id"])].frame_id for r in proj.images if r["station"] == "S001D0100")
    ims = {rec.images[d.id].name for d in rec.frames[fid].data_ids}
    for r in proj.images:
        if r["name"] in ims:
            r["station"] = "S001D0150"
            r["prior_C"] = (np.asarray(r["prior_C"], float) + np.asarray(offset)).tolist()
    return proj, rec, noise, fid


def _centre(rec, fid):
    fr = rec.frames[fid]
    return np.asarray(fr.rig_from_world.inverse().translation)


def test_unlocalized_stations_and_dropped_priors(tmp_path):
    pytest.importorskip("pyceres")
    from mppp.sfm.reconstruction import bundle_adjust, unlocalized_stations
    proj, rec, noise, fid = _short_stop(tmp_path)
    truth = _centre(rec, fid)
    u = unlocalized_stations(proj, 4, rec=rec)
    assert list(u) == ["S001D0150"] and u["S001D0150"]["images"] == 2 and u["S001D0150"]["prior"] == "dropped"
    assert u["S001D0150"]["cross_points"] > 20
    assert all(v["prior"].startswith("kept: every") for v in unlocalized_stations(proj, 50).values())
    import copy
    r0, p0 = copy.deepcopy(rec), copy.deepcopy(proj)
    ba0 = bundle_adjust(r0, p0, sigma_px=noise, loss_scale=10.0, max_iterations=30)
    r1, p1 = copy.deepcopy(rec), copy.deepcopy(proj)
    p1.settings["localize_min_images"] = 4
    ba1 = bundle_adjust(r1, p1, sigma_px=noise, loss_scale=10.0, max_iterations=30)
    assert ba1["frames_without_position_prior"] == 1 and ba1["priors"] == ba0["priors"] - 1
    # with its prior the 1.8 m error drags the short stop and the whole block (~12-14 cm); without it the block
    # moves only by the noise (~4-5 cm)
    e0 = np.linalg.norm(_centre(r0, fid) - truth)
    e1 = np.linalg.norm(_centre(r1, fid) - truth)
    assert e1 < 0.6 * e0
    other = [f for f in rec.frames if f != fid]
    m0 = np.mean([np.linalg.norm(_centre(r0, f) - _centre(rec, f)) for f in other])
    m1 = np.mean([np.linalg.norm(_centre(r1, f) - _centre(rec, f)) for f in other])
    assert m1 < 0.6 * m0


def test_register_stations_only_moves_the_short_stop(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    import pycolmap
    from mppp.sfm import reconstruction as RC
    proj, rec, noise, fid = _short_stop(tmp_path)
    truth = _centre(rec, fid)
    # matches from the tracks; then the short stop's observations leave the tracks (as after triangulating with
    # its wrong prior pose), so that its tie points are only correspondences to the other station's points
    pairs = {}
    for pt in rec.points3D.values():
        els = [(el.image_id, el.point2D_idx) for el in pt.track.elements]
        for a in range(len(els)):
            for b in range(a + 1, len(els)):
                (i1, k1), (i2, k2) = sorted((els[a], els[b]))
                if i1 != i2:
                    pairs.setdefault((i1, i2), []).append((k1, k2))
    monkeypatch.setattr(RC, "_verified_matches", lambda project: [(i1, i2, np.array(m)) for (i1, i2), m in pairs.items()])
    mine = {d.id for d in rec.frames[fid].data_ids}
    for pid in list(rec.points3D):
        for el in list(rec.points3D[pid].track.elements):
            if el.image_id in mine and pid in rec.points3D:
                rec.delete_observation(el.image_id, el.point2D_idx)
    register_stations = RC.register_stations
    others = {f: _centre(rec, f) for f in rec.frames if f != fid}
    # the frame starts at its (wrong) prior
    fr = rec.frames[fid]
    fr.rig_from_world = fr.rig_from_world * pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), -np.array([1.5, -1.0, 0.0]))
    assert np.linalg.norm(_centre(rec, fid) - truth) > 1.0
    rep = register_stations(rec, proj, only=["S001D0150"], max_error_px=40.0, min_inliers=10, verbose=False)
    assert rep["stations"]["S001D0150"]["registered"]
    assert np.linalg.norm(_centre(rec, fid) - truth) < 0.2          # from 1.8 m: within reach of triangulation
    assert all(np.allclose(_centre(rec, f), c) for f, c in others.items())


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


def test_reconstruct_with_localize_min_images(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    import copy
    from mppp.sfm import reconstruction as R
    from mppp.sfm import health as H
    proj, rec0, noise, fid = _short_stop(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec0))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)          # no database: keep the points
    calls = []
    monkeypatch.setattr(R, "register_stations", lambda rec, project, **kw: calls.append(kw) or
                        {"stations": {s: {"registered": True} for s in kw.get("only", [])}, "only": kw.get("only")})
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        localize_min_images=4, gui_native=False)
    rs = proj.settings["reconstruction"]
    assert calls and calls[0]["only"] == ["S001D0150"]
    assert rs["localize_min_images"] == 4 and rs["unlocalized_stations"] == ["S001D0150"]
    assert proj.settings["localize_min_images"] == 4 and rs["unlocalized"]["S001D0150"]["prior"] == "dropped"
    rep = H.assess_alignment(proj, rec)
    names = {c["check"]: c for c in rep["checks"]}
    assert "unlocalized_station_shift_max_m" in names and names["unlocalized_station_shift_max_m"]["value"] > 1.0
    assert rep["stations"]["S001D0150"]["position_prior"] is False
    assert all("S001D0150" not in s["stations"] for s in rep["prior_similarity"])
    # 0 keeps every prior and moves no station
    calls.clear()
    R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False, gui_native=False)
    assert not calls and proj.settings["reconstruction"]["unlocalized_stations"] == []
