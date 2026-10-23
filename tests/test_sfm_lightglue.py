"""SIFT + LightGlue matching (mppp.sfm.lightglue): keypoint geometry, keypoint selection, and the end-to-end match
and verification on the two test products."""
import json
import shutil

import numpy as np
import pytest

from conftest import NLF, ZL0, needs_data


def test_keypoint_scale_and_orientation():
    from mppp.sfm.lightglue import keypoint_scale_ori, select_keypoints
    s, o = 3.0, 0.7
    kp = np.array([[10, 20, s * np.cos(o), -s * np.sin(o), s * np.sin(o), s * np.cos(o)],
                   [30, 40, 1.0, 0.0, 0.0, 1.0]])
    scale, ori = keypoint_scale_ori(kp)
    assert np.allclose(scale, [3.0, 1.0]) and np.allclose(ori, [0.7, 0.0])
    scale4, ori4 = keypoint_scale_ori(np.array([[1, 2, 5.0, 0.3]]))
    assert scale4[0] == 5.0 and ori4[0] == 0.3
    kp = np.array([[0, 0, 1, 0, 0, 1], [0, 0, 4, 0, 0, 4], [0, 0, 2, 0, 0, 2]], float)
    assert list(select_keypoints(kp, 2)) == [1, 2]             # the two largest scales, in index order
    assert list(select_keypoints(kp, 0)) == [0, 1, 2] and list(select_keypoints(kp, 10)) == [0, 1, 2]


def test_descriptors_are_unit_vectors():
    from mppp.sfm.lightglue import _descriptors_float
    d = (np.random.default_rng(0).random((5, 128)) * 255).astype(np.uint8)
    f = _descriptors_float(d)
    assert f.dtype == np.float32 and np.allclose(np.linalg.norm(f, axis=1), 1.0)


@needs_data
def test_lightglue_matches_and_verifies(tmp_path, monkeypatch):
    pytest.importorskip("kornia")
    for k in ("MPPP_PDS", "MPPP_SCAPES", "MPPP_GPU_PYTHON", "COLMAP_BAT"):
        monkeypatch.delenv(k, raising=False)
    from mppp.pipeline import Settings, align, process_site
    pds = tmp_path / "pds" / "sol0461"
    pds.mkdir(parents=True)
    for f in (NLF, ZL0):
        shutil.copy(f, pds / f.name)
    s = Settings(pds_dir=pds.parent, scapes_root=tmp_path / "scapes", sites_extra={"rockytop_test": (461, 464)},
                 add_nearby_waypoints=0, include_zcam=True, zcam_zooms=[34], sequence_prefixes=("NCAM", "SAPP"),
                 navcam_min_frame_fraction=None, sky_elevation_deg=None, lmst_window_h=None,
                 max_saturated_fraction=None, max_exposure_ms=None, max_centre_tint=None, infer_mask=False,
                 workers=1, max_num_features=2048, match_mode="lightglue")
    s.match.update(max_keypoints=512, device="cpu")
    processed = process_site("rockytop_test", s)
    work = processed.parent
    rep = align(work, s, stop_after="database")
    assert "database" in rep
    # the two test products do not overlap (a Sun tile and a Mastcam-Z frame): give the Navcam image the Mastcam-Z
    # image's keypoints and descriptors, so that the pair must match
    import pycolmap
    from mppp.sfm import match
    from mppp.sfm.project import SfmProject
    db_path = work / "colmap" / "database.db"
    db = pycolmap.Database.open(str(db_path))
    try:
        ims = {im.name: im.image_id for im in db.read_all_images()}
        src = ims[ZL0.name.replace(".IMG", ".png")] if ZL0.name.replace(".IMG", ".png") in ims else max(ims.values())
        dst = min(ims.values()) if src != min(ims.values()) else max(ims.values())
        db.update_keypoints(dst, db.read_keypoints(src))
    finally:
        db.close()
    import sqlite3
    con = sqlite3.connect(db_path)
    con.execute("UPDATE descriptors SET rows = (SELECT rows FROM descriptors WHERE image_id = ?), "
                "cols = (SELECT cols FROM descriptors WHERE image_id = ?), "
                "data = (SELECT data FROM descriptors WHERE image_id = ?) WHERE image_id = ?", (src, src, src, dst))
    con.commit()
    con.close()
    proj = SfmProject.load(work / "colmap")
    m = match(proj, mode="lightglue", max_keypoints=512, filter_threshold=0.1, device="cpu", max_error_px=6.0,
              min_num_inliers=15)
    assert m["mode"] == "lightglue" and m["max_keypoints"] == 512 and m["lightglue"]["device"] == "cpu"
    assert m["lightglue"]["pairs"] == 1 and (work / "colmap" / "pairs_lightglue.txt").is_file()
    assert m["lightglue"]["matches"] >= 400                      # identical features: (nearly) every keypoint matches
    assert m["matched_pairs"] == 1 and m["verified_pairs"] == 1 and m["inlier_matches"] >= 15
    db = pycolmap.Database.open(str(db_path))
    try:
        raw = db.read_matches(src, dst)
        assert raw.shape[1] == 2 and len(raw) == m["lightglue"]["matches"]
        db.clear_two_view_geometries()
        db.clear_matches()
    finally:
        db.close()
    # 0.94.4: SIFT and LightGlue together, for real (COLMAP's SIFT matcher in this python, LightGlue on the CPU,
    # keypoints of every octave): the SIFT inliers stand, LightGlue's are added where they use other keypoints
    m2 = match(proj, mode="sift+lightglue", max_keypoints=512, filter_threshold=0.1, device="cpu", max_error_px=6.0,
               min_num_inliers=15, keypoint_selection="octaves", use_gpu=False)
    assert m2["mode"] == "sift+lightglue" and m2["matcher"] == "sift+lightglue"
    assert m2["sift"]["inlier_matches"] >= 15 and m2["lightglue"]["added_inliers"] >= 0
    assert m2["inlier_matches"] == m2["sift"]["inlier_matches"] + m2["lightglue"]["added_inliers"]
    assert m2["keypoint_selection"] == "octaves" and not (work / "colmap" / "database_lightglue_add.db").exists()
    rec = json.loads((work / "colmap" / "database_matches.json").read_text())
    assert rec["matching"]["mode"] == "sift+lightglue" and rec["matching"]["keypoint_selection"] == "octaves"


def test_keypoints_of_every_octave():
    """0.94.4: "octaves" keeps a share of every octave of scale (the fine ones too), "largest" the coarse ones."""
    from mppp.sfm.lightglue import select_keypoints
    rng = np.random.default_rng(0)
    # most keypoints fine (as SIFT finds them): 20000 at scale ~1-2, 6000 at 2-4, 2000 at 4-8, 500 at 8-16, 50 above
    scales = np.concatenate([rng.uniform(1, 2, 20000), rng.uniform(2, 4, 6000), rng.uniform(4, 8, 2000),
                             rng.uniform(8, 16, 500), rng.uniform(16, 64, 50)])
    kp = np.c_[rng.uniform(0, 5000, (len(scales), 2)), scales, np.zeros(len(scales))]
    big = select_keypoints(kp, 8192, "largest")
    oct_ = select_keypoints(kp, 8192, "octaves")
    assert len(big) == len(oct_) == 8192 and np.all(np.diff(oct_) > 0)
    assert scales[big].min() >= 2.0                                        # "largest" drops every fine keypoint
    o = np.floor(np.log2(scales[oct_])).astype(int)
    counts = {b: int((o == b).sum()) for b in np.unique(o)}
    assert counts[0] >= 1800 and counts[1] >= 1800                        # the fine octaves are kept
    assert counts[3] == 500 and sum(v for b, v in counts.items() if b >= 4) == 50     # small octaves keep all
    assert np.array_equal(oct_, select_keypoints(kp, 8192, "octaves"))   # deterministic
    assert len(select_keypoints(kp[:100], 8192, "octaves")) == 100
    with pytest.raises(ValueError):
        select_keypoints(kp, 10, "x")


def _db_with_images(path, n=3):
    import pycolmap
    db = pycolmap.Database.open(str(path))
    cid = db.write_camera(pycolmap.Camera(model="SIMPLE_PINHOLE", width=100, height=100, params=[100.0, 50, 50]))
    ids = [db.write_image(pycolmap.Image(name=f"i{k}.png", camera_id=cid)) for k in range(n)]
    for i in ids:
        db.write_keypoints(i, np.random.default_rng(i).uniform(0, 100, (50, 2)).astype(np.float32))
    db.close()
    return ids


def test_merge_inliers_adds_lightglue_to_sift(tmp_path):
    """0.94.4: LightGlue's inliers join the SIFT inliers of a pair; a keypoint SIFT uses keeps its SIFT match;
    a pair SIFT did not verify gets LightGlue's geometry."""
    import pycolmap
    from mppp.sfm.matching import merge_inliers
    dst, src = tmp_path / "sift.db", tmp_path / "lg.db"
    ids = _db_with_images(dst)
    import shutil
    shutil.copyfile(dst, src)

    def tvg(m):
        g = pycolmap.TwoViewGeometry()
        g.config = 2
        g.inlier_matches = np.asarray(m, np.uint32)
        return g
    d = pycolmap.Database.open(str(dst))
    d.write_two_view_geometry(ids[0], ids[1], tvg([[0, 0], [1, 1], [2, 2]]))
    d.close()
    s = pycolmap.Database.open(str(src))
    s.write_two_view_geometry(ids[0], ids[1], tvg([[0, 5], [3, 3], [4, 4], [6, 2]]))   # 2 conflicts, 2 new
    s.write_two_view_geometry(ids[1], ids[2], tvg([[7, 7], [8, 8]]))                   # a pair SIFT lacks
    s.close()
    st = merge_inliers(src, dst)
    assert st == {"added_inliers": 4, "pairs_added_to": 2, "new_pairs": 1, "conflicts": 2}
    d = pycolmap.Database.open(str(dst))
    try:
        m01 = np.asarray(d.read_two_view_geometry(ids[0], ids[1]).inlier_matches).tolist()
        m12 = np.asarray(d.read_two_view_geometry(ids[1], ids[2]).inlier_matches).tolist()
    finally:
        d.close()
    assert m01 == [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]] and m12 == [[7, 7], [8, 8]]


def test_sift_plus_lightglue_mode(monkeypatch, tmp_path):
    """0.94.4: "sift+lightglue" runs the SIFT matcher, then LightGlue in a copy, merged; the record says so; without
    LightGlue the SIFT matches stand alone."""
    from types import SimpleNamespace
    from mppp.sfm import matching as M
    calls = []
    real_match = M.match

    def fake_match(project, mode="exhaustive", **kw):
        if mode == "sift+lightglue":
            return real_match(project, mode=mode, **kw)
        calls.append(("sift", mode, kw.get("pairs")))
        return {"matcher": "sift", "verified_pairs": 3, "inlier_matches": 30}
    monkeypatch.setattr(M, "match", fake_match)
    monkeypatch.setattr(M, "add_lightglue", lambda *a: calls.append(("lg", a[-1])) or
                        {"added_inliers": 7, "pairs_added_to": 2, "new_pairs": 1, "conflicts": 0})
    finished = {}
    monkeypatch.setattr(M, "_finish", lambda project, mode, *a, extra=None: finished.update(mode=mode, **extra) or finished)
    out = real_match(SimpleNamespace(), mode="sift+lightglue", keypoint_selection="octaves", python="py")
    assert calls == [("sift", "exhaustive", None), ("lg", "octaves")]
    assert out["mode"] == "sift+lightglue" and out["matcher"] == "sift+lightglue"
    assert out["lightglue"]["added_inliers"] == 7 and out["sift"]["inlier_matches"] == 30
    assert out["keypoint_selection"] == "octaves"
    calls.clear()
    finished.clear()
    monkeypatch.setattr(M, "add_lightglue", lambda *a: None)
    out = real_match(SimpleNamespace(), mode="sift+lightglue", pairs=[("a", "b", 1)])
    assert calls == [("sift", "prior_pairs", [("a", "b", 1)])] and out["matcher"] == "sift"
    assert "keypoint_selection" not in out
    with pytest.raises(ValueError):
        real_match(SimpleNamespace(), mode="nope")


def test_keypoint_selection_keeps_old_run_keys_and_records():
    """0.94.4: the default ("largest") is in neither the run key nor the matches record, so finished runs and
    their matches stay valid; "octaves" is in both."""
    from mppp.pipeline import Settings, run_key
    from mppp.sfm.matching import match_settings
    s = Settings()
    assert s.lightglue_keypoints == "largest" and "lightglue_keypoints" not in s.key_dict()
    o = s.replace(lightglue_keypoints="octaves")
    assert o.key_dict()["lightglue_keypoints"] == "octaves" and run_key(o) != run_key(s)
    assert "keypoint_selection" not in match_settings("lightglue", **s.match)
    assert match_settings("lightglue", keypoint_selection="octaves", **s.match)["keypoint_selection"] == "octaves"


def test_lightglue_budget_is_three_hours():
    """0.94.4 (Christian): a block's LightGlue matching may take up to 3 h (projected) before the SIFT fallback."""
    import inspect
    from mppp.sfm.matching import match
    assert inspect.signature(match).parameters["max_hours"].default == 3.0
