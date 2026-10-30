"""Pair selection and matching (mppp.sfm.matching, pairs, guided): defaults, pose-guided Navcam - Mastcam-Z matching."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
def test_feature_and_matching_defaults():
    import inspect
    from mppp.sfm import database as D
    from mppp.sfm.matching import match
    assert D.DEFAULT_MAX_NUM_FEATURES == 32768 and D.DEFAULT_NUM_OCTAVES == 4 and D.DEFAULT_FIRST_OCTAVE == -1
    assert inspect.signature(match).parameters["guided_matching"].default is False   # v0p64.1: off again
    assert D._feature_settings(32768, 5120, False) == {"max_num_features": 32768, "max_image_size": 5120,
                                                        "domain_size_pooling": False}     # old records stay valid
    assert D._feature_settings(32768, 5120, False, num_octaves=5)["num_octaves"] == 5
    from mppp.pipeline import Settings
    s = Settings()
    # 0.95.0 (Christian, 27 Oct): twice the features on the Navcam frames; the rest as before
    # 0.95.2 (Christian, 30 Oct): 16384 at 63 and 79 mm, 8192 at 110 mm
    assert s.max_num_features == {"N": 65536, "Z063": 16384, "Z079": 16384, "Z110": 8192, "*": 32768}
    assert "num_octaves" in s.sift and s.match["guided_matching"] is False
    assert D.feature_count_for(s.max_num_features, "N") == 65536 and D.feature_count_for(s.max_num_features, "Z") == 32768
    assert D.feature_count_for(s.max_num_features, "Z", 34) == 32768 and D.feature_count_for(s.max_num_features, "Z", 48) == 32768
    assert D.feature_count_for(s.max_num_features, "Z", 63) == 16384 and D.feature_count_for(s.max_num_features, "Z", 110) == 8192
    assert D.feature_count_for(s.max_num_features, "N", 63) == 65536 and D.feature_count_for({"z63": 100, "*": 5}, "Z", 63) == 100
    assert D.feature_count_for(s.max_num_features, "F") == 32768 and D.feature_count_for(32768, "N") == 32768
    assert D._feature_settings(s.max_num_features, 5120, False)["max_num_features"] == {
        "*": 32768, "N": 65536, "Z063": 16384, "Z079": 16384, "Z110": 8192}
    assert D.image_zoom({"camera_group": "ZR079_F03300"}) == 79
    assert D.image_zoom({"name": "ZL0_1603_0809247900_114RAD_N0784716ZCAM09673_0630LMA01.png"}) == 63
    assert D.image_zoom({"name": "NLF_1606_0809515303_604RAD_N0790000NCAM00709_0A0095J02.png"}) is None
    # a block without those zooms keeps its features record and its run key (every Navcam block)
    class _P:
        images = [{"name": "NLF_x.png", "instrument": "NL"}, {"name": "ZL0_1603_x_N0784716ZCAM09673_0340LMA01.png",
                                                              "instrument": "ZL034", "camera_group": "ZL034_F01134"}]
    assert D.features_for_project(s.max_num_features, _P()) == {"N": 65536, "*": 32768}
    _P.images = _P.images + [{"name": "ZR0_x_ZCAM09673_0630LMA01.png", "instrument": "ZR063"}]
    assert D.features_for_project(s.max_num_features, _P()) == {"N": 65536, "Z063": 16384, "*": 32768}
    assert s.key_dict()["max_num_features"] == {"N": 65536, "*": 32768}            # include_zcam off
    from dataclasses import replace
    assert replace(s, include_zcam=True).key_dict()["max_num_features"]["Z110"] == 8192
    assert D._feature_settings({"*": 32768}, 5120, False)["max_num_features"] == 32768       # one count: as before


# pose-guided Navcam - Mastcam-Z matching (mppp.sfm.guided, v0p64)


def test_pose_guided_matching_finds_the_navcam_zcam_matches(tmp_path):
    pytest.importorskip("pyceres")
    import pycolmap
    from helpers import _block_without_navcam_zcam_matches
    from mppp.sfm import reconstruction as R
    from mppp.sfm.guided import _family_ties, pose_guided_matching
    proj, rec, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    truth = {(e.image_id, e.point2D_idx): pid for pid, pt in rec.points3D.items() for e in pt.track.elements}
    rec = R.triangulate(rec, proj, max_reproj_px=8.0)
    assert _family_ties(rec, proj)["navcam_zcam_points"] == 0
    db2, rep = pose_guided_matching(rec, proj, proj.database, verbose=False, depth="global")   # v0p65: local default
    assert db2 != proj.database and rep["new_matches_by_family"].get("NZ", 0) > 1000
    db = pycolmap.Database.open(str(db2))
    good = bad = 0
    ids = sorted(fam)
    for a in ids:
        for b in ids:
            if a < b and fam[a] != fam[b] and db.exists_two_view_geometry(a, b):
                for i, j in np.asarray(db.read_two_view_geometry(a, b).inlier_matches, int):
                    t = truth.get((a, int(i)))
                    good += int(t is not None and t == truth.get((b, int(j))))
                    bad += int(not (t is not None and t == truth.get((b, int(j)))))
    db.close()
    assert good > 1000 and bad <= 0.01 * good                                  # 1 % wrong at most
    out = R.triangulate(rec, proj, max_reproj_px=8.0, database=db2)
    assert _family_ties(out, proj)["navcam_zcam_points"] > 500


def test_staged_reconstruct_runs_pose_guided_matching(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from helpers import _block_without_navcam_zcam_matches
    from mppp.sfm import reconstruction as R
    proj, new, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    kw = dict(sigma_px=0.3, schedule=((24.0, 10.0, 8.0), (8.0, 3.0, 4.0)), max_iterations=5, verbose=False,
              navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False)
    # 0.95.0 (Christian, 28 Oct): the default scope "weak" matches only the Mastcam-Z frames short of anchor ties
    # (every one here: no Navcam - Mastcam-Z matches at all), in the exclusion round, before the exclusion
    rec = R.reconstruct(proj, **kw)
    pg = proj.settings["pose_guided_weak"]
    assert pg["new_matches"] > 0 and pg["after"]["navcam_zcam_points"] > pg["before"]["navcam_zcam_points"]
    n_z = sum(1 for f in fam.values() if f == "Z")
    assert pg["round"] == 2 and pg["weak_images_before"] == pg["images"] == n_z and pg["weak_images_after"] <= n_z
    assert pg["no_points_before"] == 0 and "pose_guided" not in proj.settings
    assert (proj.root / "database_guided_weak.db").is_file() and not (proj.root / "database_guided.db").exists()
    assert set(zf) <= set(rec.reg_frame_ids())
    # "final": the 0.94 pass over every frame after the rounds, and no weak pass
    (proj.root / "database_guided_weak.db").unlink()
    rec = R.reconstruct(proj, pose_guided={"scope": "final"}, **kw)
    pg = proj.settings["pose_guided"]
    assert pg["new_matches"] > 0 and pg["after"]["navcam_zcam_points"] > pg["before"]["navcam_zcam_points"]
    assert pg["images"] is None and "pose_guided_weak" not in proj.settings
    assert (proj.root / "database_guided.db").is_file() and not (proj.root / "database_guided_weak.db").exists()
    # "all": both
    (proj.root / "database_guided.db").unlink()
    R.reconstruct(proj, pose_guided={"scope": "all"}, **kw)
    assert proj.settings["pose_guided_weak"]["new_matches"] > 0 and "pose_guided" in proj.settings
    assert (proj.root / "database_guided.db").is_file() and (proj.root / "database_guided_weak.db").is_file()
    with pytest.raises(ValueError, match="scope"):
        R.reconstruct(proj, pose_guided={"scope": "some"}, **kw)
    # off: no guided database
    (proj.root / "database_guided.db").unlink()
    (proj.root / "database_guided_weak.db").unlink()
    R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                  navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False, pose_guided=False)
    assert not (proj.root / "database_guided.db").exists() and not (proj.root / "database_guided_weak.db").exists()


def test_weak_pass_reaches_a_frame_with_no_tie_points_before_its_exclusion(tmp_path, monkeypatch):
    """0.95.0 (Overlook Mountain, 28 Oct: 101 frames "no tie points" lost in round 2 before the guided step): a
    Mastcam-Z frame no match reached is posed from its label, gets its pairs from the points of the frames around
    it (select_pairs the other way round, partner_depth_range) and its matches before the exclusion, and stays."""
    pytest.importorskip("pyceres")
    from helpers import _write_database, _zcam_rig_rec
    from mppp.sfm import reconstruction as R
    from mppp.sfm.guided import select_pairs, weak_zcam_images
    proj, rec0, zf = _zcam_rig_rec(tmp_path)
    fam = {r["image_id"]: r["instrument"][0] for r in proj.images}
    lone = sorted(i for i, f in fam.items() if f == "Z")[0]              # one Mastcam-Z frame with no matches at all
    lone_frame = rec0.images[lone].frame_id
    lone_ims = {int(d.id) for d in rec0.frames[lone_frame].data_ids}
    _write_database(rec0, proj.database, descriptors=True, keep_pair=lambda a, b: not ({a, b} & lone_ims))
    proj.images_dir.mkdir(parents=True, exist_ok=True)
    # the start of stage 2 as the pipeline sees it: the lone frame registered at its label with no points
    bare = copy.deepcopy(rec0)
    for p in list(bare.points3D):
        pt = bare.points3D[p]
        if any(int(e.image_id) in lone_ims for e in pt.track.elements):
            bare.delete_point3D(p)
    assert bare.images[lone].num_points3D == 0
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(bare))
    weak = weak_zcam_images(bare, proj, 50)
    assert lone in weak and weak[lone] == 0
    pairs = select_pairs(bare, proj, ("Z",), 8, 5, 50.0, only=[lone])      # the narrow synthetic frame sees few
    assert pairs and all(a == lone for a, b, n in pairs) and all(n >= 5 for a, b, n in pairs)
    assert not select_pairs(bare, proj, ("Z",), 8, 5, 50.0, only=[])
    kw = dict(sigma_px=0.3, schedule=((24.0, 10.0, 8.0), (8.0, 3.0, 4.0)), max_iterations=5, verbose=False,
              navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False, exclude_outliers=True)
    rec = R.reconstruct(proj, pose_guided={"scope": "final", "min_points": 5}, **kw)
    assert lone_frame in {e["frame_id"] for e in proj.settings["reconstruction"]["excluded"]}   # 0.94: dropped first
    rec = R.reconstruct(proj, pose_guided={"min_points": 5}, **kw)
    pg = proj.settings["pose_guided_weak"]
    assert pg["no_points_before"] >= 1 and pg["no_points_after"] < pg["no_points_before"]
    assert lone_frame not in {e["frame_id"] for e in proj.settings["reconstruction"]["excluded"]}
    assert lone_frame in set(rec.reg_frame_ids()) and rec.images[lone].num_points3D > 0


# the local depth window (v0p65)


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
    assert GUIDED_DEFAULTS["workers"] == 1                                    # 0.94.4: one thread unless asked
    assert GUIDED_DEFAULTS["local_fallback"] == "skip"


def test_pose_guided_local_depth_matches(tmp_path):
    pytest.importorskip("pyceres")
    from helpers import _block_without_navcam_zcam_matches
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


def test_set_prior_focal(tmp_path):
    import pycolmap
    from mppp.sfm.matching import CALIBRATED_ABORT_SIGNS, set_prior_focal
    db_path = tmp_path / "database.db"
    db = pycolmap.Database.open(str(db_path))
    ids = {}
    for k, model, w, h in (("NL", "SIMPLE_PINHOLE", 5120, 3840), ("ZL034_F00300", "SIMPLE_PINHOLE", 1648, 1200)):
        c = pycolmap.Camera(model=model, width=w, height=h, params=np.array([1000.0, w / 2, h / 2]))
        c.has_prior_focal_length = True
        ids[k] = db.write_camera(c)
    db.close()
    proj = SimpleNamespace(database=db_path, settings={"database": {"cameras": ids}})
    assert set_prior_focal(proj, ("Z",), False) == 1
    db = pycolmap.Database.open(str(db_path))
    flags = {int(c.camera_id): bool(c.has_prior_focal_length) for c in db.read_all_cameras()}
    db.close()
    assert flags[ids["NL"]] and not flags[ids["ZL034_F00300"]]
    assert set_prior_focal(proj, ("Z",), True) == 1
    assert "kMinNumSamples" in CALIBRATED_ABORT_SIGNS
    msg = "E20261006 09:27:36 homography_matrix.cc:167] Check failed: cam_rays1.size() >= 4 (3 vs. 4)"
    assert any(k in msg for k in CALIBRATED_ABORT_SIGNS)                                   # v0p74.9


def test_mastcam_z_pairs_verified_uncalibrated_from_the_start(tmp_path, monkeypatch):
    """0.95.0 (Christian, 28 Oct): with uncalibrated_families=("Z",) the Mastcam-Z database cameras have no prior
    focal length while the matcher runs (a fundamental-matrix verification, no calibrated pass to abort and lose),
    get it back after, and the matches record says so (a rematch when the setting changes); the pipeline passes
    it for a Mastcam-Z run only, and the Navcam run keys stay what they were."""
    import pycolmap
    from mppp.sfm import matching as M
    db_path = tmp_path / "database.db"
    db = pycolmap.Database.open(str(db_path))
    ids = {}
    for k, w, h in (("NL", 5120, 3840), ("ZL034_F00300", 1648, 1200), ("ZR034_F00300", 1648, 1200)):
        c = pycolmap.Camera(model="SIMPLE_PINHOLE", width=w, height=h, params=np.array([1000.0, w / 2, h / 2]))
        c.has_prior_focal_length = True
        ids[k] = db.write_camera(c)
    db.close()
    proj = SimpleNamespace(database=db_path, settings={"database": {"cameras": ids}}, save=lambda: None)
    seen = {}

    def fake(project, mode, *a):
        d = pycolmap.Database.open(str(project.database))
        seen.update({int(c.camera_id): bool(c.has_prior_focal_length) for c in d.read_all_cameras()})
        d.close()
        seen["fams"] = a[-1]
        return {"mode": mode}
    monkeypatch.setattr(M, "_match", fake)
    out = M.match(proj, "exhaustive", uncalibrated_families=("Z",))
    assert not seen[ids["ZL034_F00300"]] and not seen[ids["ZR034_F00300"]] and seen[ids["NL"]]
    assert seen["fams"] == ("Z",) and out["uncalibrated"] == ["Z"] and out["uncalibrated_cameras"] == 2
    d = pycolmap.Database.open(str(db_path))
    assert all(c.has_prior_focal_length for c in d.read_all_cameras())      # back after the matching
    d.close()
    assert proj.settings["matching"]["uncalibrated"] == ["Z"]
    seen.clear()
    assert "uncalibrated" not in M.match(proj, "exhaustive") and seen["fams"] == ()
    # the record: only when any (the records of every run before 0.95.0 stay valid)
    from mppp.sfm.database import MATCH_KEYS
    assert "uncalibrated" in MATCH_KEYS and "uncalibrated" not in M.match_settings("exhaustive")
    assert M.match_settings("exhaustive", uncalibrated_families=("Z",))["uncalibrated"] == ["Z"]
    assert M.match_settings("exhaustive", uncalibrated=["Z"])["uncalibrated"] == ["Z"]
    from mppp.pipeline import Settings
    s = Settings()
    assert s.zcam_uncalibrated_matching is True
    assert "zcam_uncalibrated_matching" not in s.key_dict()                  # a Navcam run: the keys unchanged
    assert Settings(include_zcam=True).key_dict()["zcam_uncalibrated_matching"] is True


def test_pose_guided_threads_give_the_same_matches(tmp_path):
    """0.94.2: the pairs matched in threads write the same database as one thread."""
    pytest.importorskip("pyceres")
    import pycolmap
    from helpers import _block_without_navcam_zcam_matches
    from mppp.sfm import reconstruction as R
    from mppp.sfm.guided import pose_guided_matching
    proj, rec, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    rec = R.triangulate(rec, proj, max_reproj_px=8.0)
    _, r1 = pose_guided_matching(rec, proj, proj.database, out_database=tmp_path / "w1.db", verbose=False, workers=1)
    _, r4 = pose_guided_matching(rec, proj, proj.database, out_database=tmp_path / "w4.db", verbose=False, workers=4)
    assert r1["workers"] == 1 and r4["workers"] == 4
    assert r1["new_matches"] == r4["new_matches"] > 0 and r1["new_matches_by_family"] == r4["new_matches_by_family"]

    def inliers(p):
        db = pycolmap.Database.open(str(p))
        try:
            out = {}
            for a in rec.reg_image_ids():
                for b in rec.reg_image_ids():
                    if a < b and db.exists_two_view_geometry(int(a), int(b)):
                        out[(a, b)] = np.asarray(db.read_two_view_geometry(int(a), int(b)).inlier_matches).tolist()
            return out
        finally:
            db.close()
    assert inliers(tmp_path / "w1.db") == inliers(tmp_path / "w4.db")


def test_pose_guided_writes_elsewhere_when_its_database_is_held(tmp_path, monkeypatch):
    """0.94.4 (23 Oct 16:20, WinError 32): database_guided.db held by another process - a fresh name instead."""
    pytest.importorskip("pyceres")
    from helpers import _block_without_navcam_zcam_matches
    from mppp.sfm import reconstruction as R
    from mppp.sfm import guided as G
    proj, rec, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    rec = R.triangulate(rec, proj, max_reproj_px=8.0)
    held = proj.root / "database_guided.db"
    held.write_bytes(b"old")
    real = Path.unlink

    def unlink(self, *a, **k):
        if self == held:
            raise PermissionError(32, "in use by another process")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", unlink)
    out, rep = G.pose_guided_matching(rec, proj, proj.database, verbose=False)
    assert out != held and out.name.startswith("database_guided_") and out.is_file() and rep["new_matches"] > 0
