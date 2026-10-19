"""v0p62: stage 2 after the Navcam temperature bins of stage 1 (database copy with the bin rigs, no rig-id
collision with the set-aside Mastcam-Z rigs), and the Navcam difference maps against the consensus and the label
CAHVORE (E = 0)."""
import copy
from pathlib import Path

import numpy as np
import pytest


def _binned_stage1(tmp_path):
    from helpers import _write_database, _zcam_rig_rec
    from mppp.sfm import reconstruction as R
    from mppp.sfm.thermal import image_temperatures, thermal_stage
    proj, rec, zf = _zcam_rig_rec(tmp_path)
    _write_database(rec, proj.database)
    proj.images_dir.mkdir(parents=True, exist_ok=True)
    for r in proj.images:
        r["camera_temperature_degC"] = -32.0 if r["image_id"] % 4 < 2 else -14.0
    init = copy.deepcopy(rec)
    for f in zf:
        rec.deregister_frame(f)
    rec = R.triangulate(rec, proj, max_reproj_px=8.0)            # stage 1: COLMAP drops the Mastcam-Z rig
    assert 2 not in rec.rigs
    rec, rep = thermal_stage(rec, proj, image_temperatures(proj), bin_deg=10, fine_range=None, min_images=2, verbose=False)
    return proj, rec, init, zf, rep


def test_thermal_bin_rigs_avoid_the_set_aside_zcam_rig(tmp_path):
    pytest.importorskip("pyceres")
    from mppp.sfm import reconstruction as R
    proj, rec, init, zf, rep = _binned_stage1(tmp_path)
    assert 2 not in rec.rigs and min(r for r in rec.rigs if r != 1) > 2          # the bin rigs start above rig 2
    assert R.restore_frames(rec, init, zf) == len(zf)                           # 0.61.0: "rig.HasSensor" failed here
    assert all(rec.frames[f].rig_id == 2 for f in zf)


def test_stage2_triangulates_against_a_synced_database(tmp_path):
    pytest.importorskip("pyceres")
    from mppp.sfm import reconstruction as R
    proj, rec, init, zf, rep = _binned_stage1(tmp_path)
    R.restore_frames(rec, init, zf)
    for f in zf:
        rec.frames[f].rig_from_world = init.frames[f].rig_from_world
        rec.register_frame(f)
    with pytest.raises(ValueError, match="RigId"):                              # the error of 1 Oct (chal_rocks_sid_large)
        R.triangulate(copy.deepcopy(rec), proj, max_reproj_px=8.0)
    db2 = R.sync_database(rec, proj)
    assert db2.is_file() and db2 != proj.database
    out = R.triangulate(rec, proj, max_reproj_px=8.0, database=db2)
    assert len(out.points3D) > 1000 and all(out.frames[f].has_pose for f in zf)


def test_staged_reconstruct_with_thermal_after_stage1_and_a_real_database(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from helpers import _write_database, _zcam_rig_rec
    from mppp.sfm import reconstruction as R
    from mppp.sfm.thermal import image_temperatures
    proj, new, zf = _zcam_rig_rec(tmp_path)
    _write_database(new, proj.database)
    proj.images_dir.mkdir(parents=True, exist_ok=True)
    for r in proj.images:
        r["camera_temperature_degC"] = -32.0 if r["image_id"] % 4 < 2 else -14.0
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    rec = R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0), (8.0, 3.0, 4.0)), max_iterations=5,
                        verbose=False, navcam_intrinsics="refine", staged=True, out_name="cahv_ba",
                        temperatures=image_temperatures(proj), thermal_bins_deg=10, thermal_min_images=2,
                        thermal_after_stage1=True, zcam_focus_line=False)
    assert proj.settings["thermal"]["when"] == "end of stage 1 (Navcam only)"
    assert set(zf) <= set(rec.reg_frame_ids()) and {3, 4} <= set(rec.cameras)
    assert (proj.root / "database_thermal.db").is_file()


# ----------------------------------------------------------------------------- difference maps
def _consensus_solutions(offset_px=0.0):
    from mppp.sfm.calibration import Camera, Solution, camera_at_temperature, reference_camera
    ref = reference_camera("NL", "consensus")
    sols = {}
    for name, T in (("A", -35.0), ("B", -10.0), ("C", -22.0)):
        c = camera_at_temperature(ref, T)
        p = np.array(c.params, float)
        if name == "C":
            p[4] += offset_px                                    # a site whose lens differs (k1)
        cams = {"NL_T": Camera("NL_T", "NL", ref.model, 5120, 3840, p, np.array(ref.params) * 0.999, None, 20, 50000)}
        sols[name] = Solution(name, Path("."), {"cameras": {"NL_T": {"group": "NL", "temperature_median_degC": T}},
                                                "settings": {}}, {}, cams, {})
    return sols, ref


# ----------------------------------------------------------------------------- rig translation on the large blocks
