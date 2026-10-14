"""v0p75.1: Navcam temperature bins with fx, fy and k1 free and finer bins over -25..-5 degC; rig epochs every
50 sols and at the pitch knot (sol 300), with the epoch copies' intrinsics, distortion and baseline tied to the eye's."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_bin_edges_fine_in_the_middle():
    from mppp.sfm.thermal import bin_edges, THERMAL_FINE_RANGE_DEGC, THERMAL_FINE_BIN_DEG
    fr, fd = THERMAL_FINE_RANGE_DEGC, THERMAL_FINE_BIN_DEG
    assert (fr, fd) == ((-25.0, -5.0), 2.5)
    assert bin_edges(-17.3, 5.0, fr, fd) == (-17.5, -15.0)
    assert bin_edges(-25.0, 5.0, fr, fd) == (-25.0, -22.5)
    assert bin_edges(-5.01, 5.0, fr, fd) == (-7.5, -5.0)
    assert bin_edges(-27.0, 5.0, fr, fd) == (-30.0, -25.0) and bin_edges(-47.0, 5.0, fr, fd) == (-50.0, -45.0)
    assert bin_edges(-5.0, 5.0, fr, fd) == (-5.0, 0.0) and bin_edges(-3.0, 5.0, fr, fd) == (-5.0, 0.0)
    assert bin_edges(-17.3, 5.0) == (-20.0, -15.0)                       # no fine range: as before
    # an odd range: the coarse bins are cut at its ends
    assert bin_edges(-23.0, 5.0, (-22.0, -8.0), 2.0) == (-25.0, -22.0) and bin_edges(-7.0, 5.0, (-22.0, -8.0), 2.0) == (-8.0, -5.0)


def test_temperature_bins_fine_then_merged():
    from mppp.sfm.thermal import temperature_bins
    T = {1: -18.0, 2: -17.0, 3: -16.0, 4: -14.0, 5: -13.0, 6: -40.0}
    n = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
    b = temperature_bins(T, n, 5.0, 4, (-25.0, -5.0), 2.5)
    assert b[2] == b[3] == (-17.5, -15.0) and b[4] == b[5] == (-15.0, -12.5)     # full fine bins stay 2.5 degC
    assert b[1] == b[6] == (-40.0, -17.5)        # the two small ones merged (nearest occupied neighbour, as before)
    u = temperature_bins(T, n, 5.0, 4)                                   # uniform 5 degC bins as before
    assert u[1] == u[2] == u[3] and u[4] == u[5] == (-15.0, -10.0) and u[6] == u[1]


def test_bin_tags_keep_half_degrees():
    from mppp.sfm.thermal import _tag, eye_key
    assert _tag(-20.0, -15.0) == "T-020-015"
    assert _tag(-22.5, -20.0) == "T-022.5-020" and _tag(-25.0, -22.5) == "T-025-022.5"
    assert eye_key("NL_" + _tag(-22.5, -20.0)) == "NL" and eye_key("NR#1_" + _tag(-22.5, -20.0)) == "NR"


def test_thermal_stage_takes_k1_but_no_other_distortion():
    from mppp.sfm.thermal import thermal_stage, THERMAL_FREE
    assert THERMAL_FREE == ("fx", "fy", "k1")
    with pytest.raises(ValueError, match="distortion"):
        thermal_stage(None, None, {}, free=("fx", "k2"))


def test_sol_epochs_cut_at_the_pitch_knot():
    from mppp.sfm.navcal import sol_epochs, RIG_EPOCH_BREAK_SOLS
    from mppp.sfm.project import NAVCAM_RIG_EPOCH_GAP_SOLS, NAVCAM_RIG_EPOCH_BREAKS_SOLS
    assert NAVCAM_RIG_EPOCH_GAP_SOLS == 50.0 and tuple(NAVCAM_RIG_EPOCH_BREAKS_SOLS) == (300.0,) == RIG_EPOCH_BREAK_SOLS
    assert sol_epochs([280, 290, 305, 310], 50) == [(280, 310)]
    assert sol_epochs([280, 290, 305, 310], 50, (300,)) == [(280, 290), (305, 310)]
    assert sol_epochs([280, 290, 300, 310], 50, (300,)) == [(280, 290), (300, 310)]          # sol 300 is after
    assert sol_epochs([178, 185, 190, 341], 50, (300,)) == [(178, 190), (341, 341)]           # Rochette
    assert sol_epochs([91, 95, 101, 361, 378], 50, (300,)) == [(91, 101), (361, 378)]         # Chal Rocks Sid
    assert sol_epochs([1359, 1360, 1408, 1412], 50, (300,)) == [(1359, 1412)]                 # South Arm: one


def test_navcam_rig_epochs_with_the_knot(tmp_path):
    from mppp.sfm.project import navcam_rig_epochs
    images = [{"instrument": e, "sol": s, "camera_temperature_degC": -20.0} for s in (280, 290, 305, 310) for e in ("NL", "NR")]
    rig = {"N": {"ref": "NL", "sensor": "NR", "R_sensor_from_ref": np.eye(3).tolist(), "t_sensor_from_ref": [-0.424, 0, 0]}}
    cams = {"NL": {"params": [1.0] * 12}, "NR": {"params": [2.0] * 12}}
    info = navcam_rig_epochs(images, rig, 50.0, cameras=cams, breaks=(300.0,))
    assert [e["sols"] for e in info["epochs"]] == [[280, 290], [305, 310]] and info["breaks_sols"] == [300.0]
    assert set(rig) == {"N", "N#1"} and "tied to NL" in cams["NL#1"]["source"]
    assert navcam_rig_epochs([dict(r, instrument=r["instrument"].split("#")[0]) for r in images],
                             {"N": dict(rig["N"])}, 50.0, cameras={}) is None              # no break: one epoch


def test_epoch_tie_key_and_tie():
    from mppp.sfm.reconstruction import epoch_tie_key, _tie_epoch_cameras
    assert epoch_tie_key("NL#1") == "NL" and epoch_tie_key("NR#2_T-022.5-020") == "NR_T-022.5-020"
    assert epoch_tie_key("NL_T-020-015") == "NL_T-020-015" and epoch_tie_key("ZL034#1") == "ZL034#1"
    cams = {1: "NL", 2: "NR", 3: "NL#1", 4: "NR#1", 5: "NL_T-020-015", 6: "NL#1_T-020-015", 7: "NL#1_T-010-005"}
    proj = SimpleNamespace(settings={"database": {"cameras": {v: k for k, v in cams.items()}}})
    work = SimpleNamespace(cameras={k: SimpleNamespace(model="THIN_PRISM_FISHEYE") for k in cams})
    blocks = {k: np.full(12, float(k)) for k in cams}
    tied = _tie_epoch_cameras(blocks, work, proj)
    assert tied == {"NL": ["NL#1"], "NR": ["NR#1"], "NL_T-020-015": ["NL#1_T-020-015"]}
    assert blocks[3] is blocks[1] and blocks[4] is blocks[2] and blocks[6] is blocks[5] and blocks[7][0] == 7.0
    proj.settings["navcam_rig_epochs_tie"] = False
    blocks2 = {k: np.full(12, float(k)) for k in cams}
    assert _tie_epoch_cameras(blocks2, work, proj) == {} and blocks2[3] is not blocks2[1]


def _two_epoch_block(tmp_path, monkeypatch, **kw):
    """A Navcam block whose second station is a later visit: two rigs (N, N#1 with cameras NL#1, NR#1) through
    stage 1 and the thermal stage, both refined to the true rig; the consensus merge maps the epoch cameras to
    their eyes and keeps every image."""
    pytest.importorskip("pyceres")
    import copy
    import pycolmap
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(ROOT / "tests"))
    from helpers import _build_rec, _synthetic
    from mppp.sfm import navcal as NC
    from mppp.sfm import reconstruction as R
    from mppp.sfm.thermal import image_temperatures
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    for r in proj.images:
        later = r["station"].endswith("0100")
        r["camera_temperature_degC"] = -32.0 if not later else -14.0
        r["sol"] = 95 if not later else 365
        if later:
            r["instrument"], r["rig_epoch"] = r["instrument"] + "#1", 1
        else:
            r["rig_epoch"] = 0
    for eye in ("NL", "NR"):
        proj.cameras[f"{eye}#1"] = dict(proj.cameras[eye], group=eye, epoch=1)
    proj.rig["N#1"] = dict(proj.rig["N"], ref="NL#1", sensor="NR#1", epoch={"index": 1, "sols": [365, 365]})
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2, "NL#1": 3, "NR#1": 4}, "rig_ids": {"N": 1, "N#1": 2}}
    proj.settings["navcam_rig_epochs"] = {"gap_sols": 100.0, "epochs": [{"index": 0, "rig_key": "N"}, {"index": 1, "rig_key": "N#1"}]}
    # the synthetic reconstruction with the later frames on a second rig and the camera copies
    for r in proj.images:                                    # _build_rec keys cameras by NL / NR
        r["instrument"] = r["instrument"].split("#")[0]
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    sens = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)        # noqa: E731
    for cid in (1, 2):
        c = rec0.cameras[cid]
        rec0.add_camera(pycolmap.Camera(camera_id=cid + 2, model=c.model, width=c.width, height=c.height, params=np.array(c.params)))
    rig2 = pycolmap.Rig(rig_id=2)
    rig2.add_ref_sensor(sens(3))
    rig2.add_sensor(sens(4), rec0.rigs[1].sensor_from_rig(sens(2)))
    rec0.add_rig(rig2)
    later_names = {r["name"] for r in proj.images if r["rig_epoch"] == 1}
    new = pycolmap.Reconstruction()
    for c in rec0.cameras.values():
        new.add_camera(c)
    for rg in rec0.rigs.values():
        new.add_rig(rg)
    for fid, fr in rec0.frames.items():
        later = any(rec0.images[d.id].name in later_names for d in fr.data_ids)
        nf = pycolmap.Frame(frame_id=fid, rig_id=2 if later else 1)
        for d in fr.data_ids:
            nf.add_data_id(pycolmap.data_t(sensor_id=sens(d.sensor_id.id + (2 if later else 0)), id=d.id))
        nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
    for iid, im in rec0.images.items():
        later = im.name in later_names
        ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]),
                            camera_id=im.camera_id + (2 if later else 0), image_id=iid)
        ni.frame_id = im.frame_id
        new.add_image(ni)
    for fid in rec0.reg_frame_ids():
        new.register_frame(fid)
    for pid, p in rec0.points3D.items():
        new.add_point3D(p.xyz, p.track, np.zeros(3, np.uint8))
    for r in proj.images:
        if r["rig_epoch"] == 1:
            r["instrument"] += "#1"
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=8, verbose=False,
                        navcam_intrinsics="refine", refine_rig="rotation", temperatures=image_temperatures(proj),
                        thermal_bins_deg=10, thermal_min_images=4, out_name="cahv_ba", **kw)
    single = pycolmap.Reconstruction(str(proj.root / "sparse" / "cahv_ba_single"))
    return proj, rec, single, rigT


def test_two_epochs_share_the_cameras_and_refine_their_rigs(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    proj, rec, single, rigT = _two_epoch_block(tmp_path, monkeypatch)
    # the epoch copies NL#1, NR#1 come out with exactly the eye's parameters; each rig its own rotation
    assert np.array_equal(np.asarray(single.cameras[1].params), np.asarray(single.cameras[3].params))
    assert np.array_equal(np.asarray(single.cameras[2].params), np.asarray(single.cameras[4].params))
    R_rel = np.asarray(rigT[0])
    for rid in (1, 2):
        rg = single.rigs[rid]
        sid = next(iter(rg.non_ref_sensors))
        d = Rotation.from_matrix(np.asarray(rg.sensor_from_rig(sid).rotation.matrix()) @ R_rel.T).as_rotvec()
        assert np.degrees(np.linalg.norm(d)) * 1e3 < 20.0
    t1 = np.asarray(single.rigs[1].sensor_from_rig(next(iter(single.rigs[1].non_ref_sensors))).translation)
    t2 = np.asarray(single.rigs[2].sensor_from_rig(next(iter(single.rigs[2].non_ref_sensors))).translation)
    assert np.allclose(t1, t2)                                            # one baseline for both visits
    # the thermal stage: fx, fy, k1 free per bin, the finer bins reported
    th = proj.settings["thermal"]
    assert th["free"] == ["fx", "fy", "k1"] and th["fine_range_degC"] == [-25.0, -5.0] and th["fine_bin_deg"] == 2.5
    assert all("k1" in r and "k1" in r["start"] for r in th["rows"])
    assert any("k1" in r["free"] for r in th["rows"]) or th["k1_held_bins"]


def test_untied_epochs_keep_their_own_cameras(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    def setup(project):
        project.settings["navcam_rig_epochs_tie"] = False
    import mppp.sfm.reconstruction as R
    orig = R._tie_epoch_cameras
    monkeypatch.setattr(R, "_tie_epoch_cameras", lambda b, w, p: (setup(p), orig(b, w, p))[1])
    proj, rec, single, rigT = _two_epoch_block(tmp_path, monkeypatch)
    assert not np.array_equal(np.asarray(single.cameras[1].params), np.asarray(single.cameras[3].params))
