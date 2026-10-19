"""SIFT + LightGlue matching (mppp.sfm.lightglue): keypoint geometry, keypoint selection, and the end-to-end match
and verification on the two test products."""
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
    pds = tmp_path / "pds" / "sol0709"
    pds.mkdir(parents=True)
    for f in (NLF, ZL0):
        shutil.copy(f, pds / f.name)
    s = Settings(pds_dir=pds.parent, scapes_root=tmp_path / "scapes", sites_extra={"sol709": (709, 709)},
                 add_nearby_waypoints=0, include_zcam=True, zcam_zooms=[34], sequence_prefixes=("NCAM", "SAPP"),
                 navcam_min_frame_fraction=None, sky_elevation_deg=None, lmst_window_h=None,
                 max_saturated_fraction=None, max_exposure_ms=None, max_centre_tint=None, infer_mask=False,
                 workers=1, max_num_features=2048, match_mode="lightglue")
    s.match.update(max_keypoints=512, device="cpu")
    processed = process_site("sol709", s)
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
    finally:
        db.close()
