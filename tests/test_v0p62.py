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
    rec, rep = thermal_stage(rec, proj, image_temperatures(proj), bin_deg=10, min_images=2, verbose=False)
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


def test_consensus_reference_and_temperature():
    from mppp.sfm.calibration import camera_at_temperature, reference_camera
    ref = reference_camera("NL", "consensus")
    assert ref.model == "THIN_PRISM_FISHEYE" and ref.thermal["T0_degC"] == -20.0
    c = camera_at_temperature(ref, 0.0)
    assert c.params[0] == pytest.approx(ref.params[0] * (1 + 20e-6 * ref.thermal["ppm_per_degC"]))
    assert c.params[2] - ref.params[2] == pytest.approx(20 * ref.thermal["cx_px_per_degC"])
    assert camera_at_temperature(ref, None) is ref


def test_consensus_differences_show_only_the_site_effect():
    from mppp.sfm.calibration import consensus_differences, difference_maps_figure
    sols, ref = _consensus_solutions(offset_px=0.002)
    rows = {r["scape"]: r for r in consensus_differences(sols, "NL", ref, min_observations=1000)}
    assert rows["A"]["rms_px"] < 1e-3 and rows["B"]["rms_px"] < 1e-3            # the thermal terms explain them
    assert rows["C"]["corner_rms_px"] > 1.0 and rows["C"]["T_degC"] == -22.0
    fig = difference_maps_figure(list(rows.values()), "test")
    assert len(fig.axes) == 4                                                    # 3 panels and the colour bar


def test_label_differences_drop_e(tmp_path):
    from mppp.cmod import fit_to_colmap
    from mppp.sfm.calibration import label_camera, label_differences
    sols, ref = _consensus_solutions()
    cm, fit = fit_to_colmap(ref.model, ref.params, 5120, 3840, kind="CAHVORE", mtype=2, step=160)
    cm.E = np.array([0.002, 0.004, -0.003])                                      # an entrance pupil that moves (m)
    for s in sols.values():
        s.images["NLF_1.png"] = {"name": "NLF_1.png", "instrument": "NL_T", "stem": "NLF_1"}
        s.manifest["NLF_1"] = {"filename": {"family": "N", "downsample_scale": 1.0},
                               "camera_model_label": cm.to_label_dict(12)}
    s = sols["A"]
    assert np.all(label_camera(s, "NL_T").E == 0) and np.any(label_camera(s, "NL_T", drop_e=False).E != 0)
    rows = label_differences(sols, "NL", min_observations=1000)
    assert len(rows) == 3 and rows[0]["reference"] == "label CAHVORE (E = 0)"
    assert rows[0]["e_effect_1m_px"] > 0.1                                       # E matters at 1 m ...
    assert rows[0]["centre_rms_px"] < 1.0                                        # ... the E = 0 label fits the centre


# ----------------------------------------------------------------------------- rig translation on the large blocks
def _consensus_result(translation=None, applied=False):
    import json
    from mppp.paths import cmods_dir
    from mppp.sfm.project import PARAM_NAMES
    names = PARAM_NAMES["THIN_PRISM_FISHEYE"]
    cur = {e: json.loads((cmods_dir() / f"M2020_{e}_fisheye_tangential.json").read_text()) for e in ("NL", "NR")}
    res = {e: dict(zip(names, map(float, cur[e]["params"]))) for e in ("NL", "NR")}
    res.update({"blocks": {"A": 10, "B": 12}, "rig_abs_mdeg": {"yaw_mdeg": 35.0, "pitch_mdeg": -88.0, "roll_mdeg": -63.0},
                "rig_R": np.eye(3).tolist(), "rig_t": [-0.4244, 0.0, 0.0],
                "residuals": {"median_px": 0.18, "rms_px": 0.38, "p95_px": 0.8},
                "thermal": {"ppm_per_degC": 38.1, "T0_degC": -20.0, "cx_px_per_degC_NL": 0.0517}, "k4_zero": True})
    if translation:
        res["rig_translation"] = translation
        res["rig_translation_applied"] = applied
    return res


def test_write_consensus_records_or_applies_the_large_block_translation(tmp_path):
    import json
    from mppp.sfm.navcal_consensus import write_consensus
    tr = {"blocks": {"Belva": 300, "Taylorfjellet Large": 400}, "dcentre_mm": [-0.43, 0.16, -0.27], "sd_t_mm": [0.1, 0.01, 0.02],
          "baseline_start_m": 0.4244, "baseline_m": 0.42397, "dbaseline_mm": -0.43, "t_m": [-0.42397, 0.00016, -0.00027],
          "variance_factor": 1.2, "sigma_m": 0.05}
    rig = json.loads((write_consensus(_consensus_result(tr, False), tmp_path / "a") / "M2020_N_rig.json").read_text())
    assert rig["translation_fit"]["dcentre_mm"] == tr["dcentre_mm"] and "translation" not in rig
    assert "recorded but not applied" in rig["note"]
    rig = json.loads((write_consensus(_consensus_result(tr, True), tmp_path / "b") / "M2020_N_rig.json").read_text())
    assert rig["translation"] == "fitted" and rig["t_sensor_from_ref_m"] == tr["t_m"] and "2 large blocks" in rig["note"]
    rig = json.loads((write_consensus(_consensus_result(), tmp_path / "c") / "M2020_N_rig.json").read_text())
    assert "translation_fit" not in rig and rig["note"].endswith("Translation from CAHV in each project.")


def test_large_blocks_by_station_span(monkeypatch):
    from mppp.sfm import navcal_consensus as NCC
    spans = {"a": 12.0, "b": 36.3, "c": 704.0}
    monkeypatch.setattr(NCC, "block_span_m", lambda root: spans[str(root)])
    assert NCC.large_blocks({k: k for k in spans}, 30.0) == {"b": ["b", 36.3], "c": ["c", 704.0]}


def test_study_cli_has_the_translation_options():
    import subprocess, sys
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "navcam_calibration_study.py"), "--help"],
                       capture_output=True, text=True)
    assert "--translation-min-span" in r.stdout and "--apply-translation" in r.stdout
