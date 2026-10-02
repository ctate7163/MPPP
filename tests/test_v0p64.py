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
    db2, rep = pose_guided_matching(rec, proj, proj.database, verbose=False)
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


def test_profile_thermal_finds_the_minimum(monkeypatch):
    from mppp.sfm import navcal as NC
    from mppp.sfm import navcal_consensus as NCC
    calls = []

    def fake_joint(r, proj, temps, ppm, T0, max_iterations=50, rig_slopes=None, pp_slopes=None, **kw):
        x = pp_slopes[1][0]
        calls.append(x)
        return {"final_cost": 1000.0 + 5e5 * (x - 0.031) ** 2 + 0.01 * (ppm - 38.0) ** 2, "observations": 10000,
                "iterations": 3}
    monkeypatch.setattr(NC, "joint_adjust", fake_joint)
    v = {"f": 38.1, "NL_cx": 0.0517, "NL_cy": 0.0, "NR_cx": 0.0, "NR_cy": 0.0}
    out = NCC.profile_thermal(object(), None, {}, v, -20.0, "NL_cx", verbose=False)
    assert abs(out["best"] - 0.031) < 1e-6 and out["sd"] > 0 and not out["at_edge"] and len(calls) == 5
    far = NCC.profile_thermal(object(), None, {}, dict(v, NL_cx=0.2), -20.0, "NL_cx", verbose=False)
    assert abs(far["best"] - 0.031) < 1e-6 and len(far["rows"]) > 5          # the grid was extended
    with pytest.raises(KeyError):
        NCC.profile_thermal(object(), None, {}, v, -20.0, "fy", verbose=False)


def test_write_consensus_with_refitted_thermal_terms(tmp_path):
    from test_v0p62 import _consensus_result
    from mppp.sfm.navcal_consensus import write_consensus
    res = _consensus_result()
    res["thermal"] = {"ppm_per_degC": 36.5, "T0_degC": -20.0, "cx_px_per_degC_NL": 0.045,
                      "pp_slopes": {"NL": [0.045, 0.01], "NR": [0.002, -0.004]}, "refit": True,
                      "terms": ["f", "NL_cx", "NR_cy"], "sd": {"f": 0.4, "NL_cx": 0.003, "NR_cy": 0.002}, "profiles": {}}
    d = write_consensus(res, tmp_path / "nj")
    L = json.loads((d / "M2020_NL_fisheye_tangential.json").read_text())["thermal"]
    R = json.loads((d / "M2020_NR_fisheye_tangential.json").read_text())["thermal"]
    assert L["ppm_per_degC"] == 36.5 and L["cx_px_per_degC"] == 0.045 and L["cy_px_per_degC"] == 0.01
    assert L["sd_ppm_per_degC"] == 0.4 and R["cy_px_per_degC"] == -0.004 and R["sd_cy_px_per_degC"] == 0.002
    assert "refitted" in json.loads((d / "M2020_NL_fisheye_tangential.json").read_text())["source"]


def test_notebook_and_script_defaults_v0p64():
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "num_octaves=None" in src and "num_octaves=6 if INCLUDE_ZCAM else 4" in src
    assert "POSE_GUIDED" in src and "pose_guided=POSE_GUIDED" in src
    nb4 = json.loads((ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8"))
    src4 = "".join("".join(c["source"]) for c in nb4["cells"] if c["cell_type"] == "code")
    assert "REFIT_THERMAL = False" in src4 and "--refit-thermal" in src4
    g = (ROOT / "scripts" / "windows" / "github_push.bat").read_bytes()
    assert b"\n" not in g.replace(b"\r\n", b"") and b"git push origin --tags" in g and b"--tags ||" not in g.split(b":tags")[0]
