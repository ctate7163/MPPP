"""v0p64: pose-guided matching after stage 2, the thermal refit in the consensus form, six SIFT octaves with
Mastcam-Z, and tags pushed separately in github_push.bat."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _block_without_navcam_zcam_matches(tmp_path):
    from helpers import _write_database, _zcam_rig_rec
    proj, rec, zf = _zcam_rig_rec(tmp_path)
    fam = {r["image_id"]: r["instrument"][0] for r in proj.images}
    _write_database(rec, proj.database, descriptors=True, keep_pair=lambda a, b: fam[a] == fam[b])
    proj.images_dir.mkdir(parents=True, exist_ok=True)
    return proj, rec, zf, fam


def test_pose_guided_matching_finds_the_navcam_zcam_matches(tmp_path):
    pytest.importorskip("pyceres")
    import pycolmap
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
    from mppp.sfm import reconstruction as R
    proj, new, zf, fam = _block_without_navcam_zcam_matches(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    rec = R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0), (8.0, 3.0, 4.0)), max_iterations=5,
                        verbose=False, navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False)
    pg = proj.settings["pose_guided"]
    assert pg["new_matches"] > 0 and pg["after"]["navcam_zcam_points"] > pg["before"]["navcam_zcam_points"]
    assert (proj.root / "database_guided.db").is_file() and set(zf) <= set(rec.reg_frame_ids())
    # off: no guided database
    (proj.root / "database_guided.db").unlink()
    R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                  navcam_intrinsics="refine", staged=True, out_name="cahv_ba", zcam_focus_line=False, pose_guided=False)
    assert not (proj.root / "database_guided.db").exists()


def test_pipeline_and_script_defaults_v0p64(tmp_path):
    from mppp.pipeline import Settings, resolve_site
    s = Settings(scapes_root=tmp_path)
    assert s.sift["num_octaves"] is None and s.pose_guided is True
    assert resolve_site("rockytop", s)[1].sift["num_octaves"] == 4          # 6 with Mastcam-Z, else COLMAP's 4
    assert resolve_site("rockytop", s.replace(include_zcam=True))[1].sift["num_octaves"] == 6
