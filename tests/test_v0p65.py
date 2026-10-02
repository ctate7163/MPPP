"""v0p65: the Navcam pixel aspect held in the bundle adjustment, the beyond-ground and aspect health checks (and the
block screening that uses them), the local depth window of pose-guided matching (with its undo guard), and the
alignment rules version in the run key."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------- beyond the ground
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


# --------------------------------------------------------------------------------------------- aspect held in the BA
def test_aspect_prior_cost():
    pyceres = pytest.importorskip("pyceres")
    from mppp.sfm.reconstruction import AspectPrior
    x = np.array([100.0, 101.0, 5.0, 5.0])
    prob = pyceres.Problem()
    c = AspectPrior(1.0, 1e-4, 4)
    prob.add_residual_block(c, None, [x])
    so = pyceres.SolverOptions()
    s = pyceres.SolverSummary()
    pyceres.solve(so, prob, s)
    assert x[1] / x[0] == pytest.approx(1.0, abs=1e-6) and x[2] == 5.0


def test_bundle_adjust_holds_the_navcam_aspect(tmp_path):
    pytest.importorskip("pyceres")
    from helpers import _synthetic, _build_rec
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    if "database" not in proj.settings:
        proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    key_of = {int(v): k for k, v in proj.settings["database"]["cameras"].items()}
    # the start camera (the "consensus") with a pixel aspect 0.3 % off the truth: free, the data pulls fy / fx to the
    # truth; held, it stays at the start
    for cid, k in key_of.items():
        p = list(proj.cameras[k]["params"])
        p[1] = p[0] * 1.003
        proj.cameras[k]["params"] = p
        rec0.cameras[cid].params = np.asarray(p, float)
    a0 = {cid: rec0.cameras[cid].params[1] / rec0.cameras[cid].params[0] for cid in key_of}
    out = {}
    for mode in ("free", "hold"):
        rec = copy.deepcopy(rec0)
        ba = R.bundle_adjust(rec, proj, sigma_px=0.3, max_iterations=30, navcam_aspect=mode)
        out[mode] = {cid: rec.cameras[cid].params[1] / rec.cameras[cid].params[0] / a0[cid] - 1 for cid in key_of}
        assert ba["navcam_aspect"] == mode and ba["aspect_priors"] == (len(key_of) if mode == "hold" else 0)
    for cid in key_of:
        assert abs(out["hold"][cid]) < 2e-4 < abs(out["free"][cid])
    # the project setting decides when the argument is not given (as in the thermal stage)
    proj.settings["navcam_aspect"] = "hold"
    assert R.bundle_adjust(copy.deepcopy(rec0), proj, sigma_px=0.3, max_iterations=2)["aspect_priors"] == len(key_of)


# ------------------------------------------------------------------------------------- pose-guided local depth window
def test_local_depths_bracket_the_true_depth(tmp_path):
    from helpers import _zcam_rig_rec
    from mppp.sfm.guided import _pose, local_depths
    proj, rec, zf = _zcam_rig_rec(tmp_path)
    iid = sorted(rec.images)[-1]
    im = rec.images[iid]
    _, C = _pose(im)
    tri = [(p.xy, np.linalg.norm(rec.points3D[p.point3D_id].xyz - C)) for p in im.points2D if p.has_point3D()]
    xy = np.array([t[0] for t in tri]) + 2.0                                 # next to the triangulated keypoints
    d = np.array([t[1] for t in tri])
    lo, hi = local_depths(rec, iid, xy)
    ok = np.isfinite(lo)
    assert ok.mean() > 0.9 and np.mean((lo[ok] <= d[ok]) & (d[ok] <= hi[ok])) > 0.95
    # far from every triangulated keypoint: no window
    lo2, _ = local_depths(rec, iid, np.array([[-1e5, -1e5]]))
    assert np.isnan(lo2[0])


def test_guided_defaults_v0p65_and_options():
    from mppp.sfm.guided import GUIDED_DEFAULTS
    assert GUIDED_DEFAULTS["depth"] == "local" and GUIDED_DEFAULTS["local_factor"] == 1.4
    assert GUIDED_DEFAULTS["local_fallback"] == "skip"


def test_pose_guided_local_depth_matches(tmp_path):
    pytest.importorskip("pyceres")
    from test_v0p64 import _block_without_navcam_zcam_matches
    from mppp.sfm import reconstruction as R
    from mppp.sfm.guided import pose_guided_matching
    proj, rec, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    truth = {(e.image_id, e.point2D_idx): pid for pid, pt in rec.points3D.items() for e in pt.track.elements}
    rec = R.triangulate(rec, proj, max_reproj_px=8.0)
    _, rep_l = pose_guided_matching(rec, proj, proj.database, out_database=tmp_path / "l.db", verbose=False)
    _, rep_g = pose_guided_matching(rec, proj, proj.database, out_database=tmp_path / "g.db", verbose=False,
                                    depth="global")
    assert rep_l["options"]["depth"] == "local" and rep_l["new_matches_by_family"].get("NZ", 0) > 1000
    # the local window keeps only true matches (the synthetic Mastcam-Z frames have few triangulated keypoints, so
    # fewer keypoints get a window here than in a real stage-2 block)
    import pycolmap
    db = pycolmap.Database.open(str(tmp_path / "l.db"))
    good = bad = 0
    for a in sorted(fam):
        for b in sorted(fam):
            if a < b and fam[a] != fam[b] and db.exists_two_view_geometry(a, b):
                for i, j in np.asarray(db.read_two_view_geometry(a, b).inlier_matches, int):
                    t = truth.get((a, int(i)))
                    ok = t is not None and t == truth.get((b, int(j)))
                    good += int(ok)
                    bad += int(not ok)
    db.close()
    assert good > 1000 and bad <= 0.01 * good
    assert rep_g["new_matches"] >= 0.8 * rep_l["new_matches"]
    with pytest.raises(ValueError):
        pose_guided_matching(rec, proj, proj.database, out_database=tmp_path / "x.db", verbose=False, depth="x")


def test_staged_reconstruct_records_aspect_and_guard(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from test_v0p64 import _block_without_navcam_zcam_matches
    from mppp.sfm import reconstruction as R
    proj, new, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0), (8.0, 3.0, 4.0)), max_iterations=5,
                  verbose=False, navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False)
    assert proj.settings["navcam_aspect"] == "hold" and proj.settings["reconstruction"]["navcam_aspect"] == "hold"
    pg = proj.settings["pose_guided"]
    assert "beyond_ground_before" in pg and "beyond_ground_after" in pg and "reverted" not in pg
    # the guard: a rise above the limit undoes the stage
    monkeypatch.setattr(R, "POSE_GUIDED_MAX_BEYOND_GROUND_RISE", -1.0)
    rec = R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False)
    assert "reverted" in proj.settings["pose_guided"] and rec.num_points3D() > 0
    with pytest.raises(ValueError):
        R.reconstruct(proj, verbose=False, navcam_aspect="x")


def test_screen_block(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from test_v0p64 import _block_without_navcam_zcam_matches
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


# ------------------------------------------------------------------------------------------------------- run key
def test_run_key_includes_the_alignment_rules(monkeypatch):
    from mppp import runner
    s = {"A": 1}
    k1 = runner.run_key(s, "pds", "")
    p1 = runner.process_key(s)
    monkeypatch.setattr(runner, "ALIGN_RULES", runner.ALIGN_RULES + 1)
    assert runner.run_key(s, "pds", "") != k1                 # a new rules version reruns the alignments
    assert runner.process_key(s) == p1                        # ... but not the processing
    assert runner.ALIGN_RULES - 1 >= 65


def test_notebook_defaults_v0p65():
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'NAVCAM_ASPECT     = "hold"' in src and "navcam_aspect=NAVCAM_ASPECT" in src
    nb4 = json.loads((ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8"))
    src4 = "".join("".join(c["source"]) for c in nb4["cells"] if c["cell_type"] == "code")
    assert "STUDY_NAVCAM_ONLY = True" in src4 and 'STUDY_SCREEN = "fail"' in src4 and "screen_blocks(blocks)" in src4
    import inspect
    from mppp.sfm import navcal
    assert inspect.getsource(navcal).count('navcam_aspect="free"') == 3     # the calibration studies keep it free
