"""Navcam temperature model and thermal stage (mppp.sfm.thermal). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import copy
import json
import pytest
import numpy as np
from pathlib import Path


def _synthetic_block(tmp_path):
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    for i, r in enumerate(proj.images):
        r["camera_temperature_degC"] = -32.0 if r["station"].endswith("0000") else -14.0
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    return proj, rec, noise


def test_temperature_bins_merge_small_bins():
    from mppp.sfm.thermal import temperature_bins, _tag
    T = {1: -35.0, 2: -34.0, 3: -25.0, 4: -24.0, 5: -23.0, 6: -22.0, 7: -12.0}
    n = {f: 2 for f in T}
    b = temperature_bins(T, n, bin_deg=10, min_images=4)
    assert b[1] == b[2] == (-40.0, -30.0)                     # 4 images: kept
    assert b[3] == b[6] and b[7] == (-30.0, -10.0)            # the lone -20..-10 frame joins its neighbour
    assert b[3] == (-30.0, -10.0)
    assert _tag(-30, -20) == "T-030-020"
    assert temperature_bins({1: 5.0}, {1: 2}) == {1: (0.0, 10.0)}


def test_thermal_stage_splits_relabels_and_strips(tmp_path):
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.thermal import image_temperatures, thermal_stage, strip_thermal_bins
    proj, rec, noise = _synthetic_block(tmp_path)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=20)
    temps = image_temperatures(proj)
    assert len(temps) == len(proj.images) and all(v["source"] == "manifest" for v in temps.values())
    n_before = sum(len(im.points2D) for im in rec.images.values())
    rec2, rep = thermal_stage(rec, proj, temps, bin_deg=10, fine_range=None, min_images=4, max_iterations=20, verbose=False)
    assert rep["bins"] == 2 and len(rep["rows"]) == 4 and not rep["held"]
    keys = {r["camera"] for r in rep["rows"]}
    assert keys == {"NL_T-040-030", "NL_T-020-010", "NR_T-040-030", "NR_T-020-010"}
    assert all(proj.cameras[k]["thermal_bin"] and proj.cameras[k]["group"] in ("NL", "NR") for k in keys)
    assert all(r["instrument"] in keys and r["base_instrument"] in ("NL", "NR") for r in proj.images)
    assert rec2.num_reg_images() == rec.num_reg_images() == len(proj.images)
    assert sum(len(im.points2D) for im in rec2.images.values()) == n_before
    assert rep["after"]["rms_px"] <= rep["before"]["rms_px"] * 1.05
    assert strip_thermal_bins(proj) == 4
    assert set(proj.cameras) == {"NL", "NR"} and all(r["instrument"] in ("NL", "NR") for r in proj.images)
    assert "base_instrument" not in proj.images[0] and "thermal" not in proj.settings


def test_thermal_stage_held_bins_follow_the_model(tmp_path):
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.thermal import image_temperatures, thermal_stage
    proj, rec, noise = _synthetic_block(tmp_path)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=10)
    f0 = {k: float(rec.cameras[v].params[0]) for k, v in proj.settings["database"]["cameras"].items()}
    model = {"NL": {"fx_ppm_per_degC": 60.0, "fy_ppm_per_degC": 60.0, "T0_degC": -20.0},
             "NR": {"fx_ppm_per_degC": 60.0, "fy_ppm_per_degC": 60.0, "T0_degC": -20.0}}
    rec2, rep = thermal_stage(rec, proj, image_temperatures(proj), bin_deg=10, fine_range=None, min_images=4, hold=True,
                              thermal_model=model, max_iterations=5, verbose=False)
    assert rep["held"]
    for r in rep["rows"]:
        want = f0[r["eye"]] * (1 + 60e-6 * (r["T_median_degC"] + 20.0))
        assert abs(r["fx"] - want) < 1e-6 and abs(r["start_fx"] - want) < 1e-6


def test_reconstruct_ends_with_the_thermal_stage_and_a_rerun_strips_it(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    import copy
    import pycolmap
    from helpers import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    from mppp.sfm.thermal import image_temperatures
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    for r in proj.images:
        r["camera_temperature_degC"] = -32.0 if r["station"].endswith("0000") else -14.0
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec0))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)
    kw = dict(sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False, navcam_intrinsics="refine",
              temperatures=image_temperatures(proj), thermal_bins_deg=10, thermal_min_images=4, out_name="cahv_ba")
    rec = R.reconstruct(proj, **kw)
    th = proj.settings["thermal"]
    assert th["bins"] == 2 and len(th["rows"]) == 4 and th["single_camera_model"] == "sparse/cahv_ba_single"
    assert (proj.root / "sparse" / "cahv_ba_single").is_dir() and len(rec.cameras) == 6
    assert len(pycolmap.Reconstruction(str(proj.root / "sparse" / "cahv_ba_single")).cameras) == 2
    assert all(r["instrument"].startswith(("NL_T", "NR_T")) for r in proj.images)
    rec2 = R.reconstruct(proj, **dict(kw, thermal_bins_deg=None))        # a rerun without bins starts from one camera per eye
    assert "thermal" not in proj.settings and set(proj.cameras) == {"NL", "NR"} and len(rec2.cameras) == 2


def test_health_and_export_on_a_binned_block(tmp_path):
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.thermal import image_temperatures, thermal_stage
    from mppp.sfm.health import assess_alignment
    from mppp.sfm.export import export_for_error, native_reconstruction
    proj, rec, noise = _synthetic_block(tmp_path)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=10)
    rec2, rep = thermal_stage(rec, proj, image_temperatures(proj), bin_deg=10, fine_range=None, min_images=4, max_iterations=5, verbose=False)
    h = assess_alignment(proj, rec2)
    rig = [c for c in h["checks"] if "rig" in c["check"]]
    assert rig and all(c["status"] != "fail" for c in rig)
    native_reconstruction(rec2, proj)
    s = export_for_error(proj, rec2, tmp_path / "error_input")
    summ = json.loads((tmp_path / "error_input" / "summary.json").read_text())
    assert {r["camera"] for r in rep["rows"]} <= set(summ["cameras_refined"])


def test_zcam_label_temperature(tmp_path):
    from mppp.labels import camera_temperature, read_pds
    lab = ("PDS_VERSION_ID = PDS3\r\nRECORD_TYPE = FIXED_LENGTH\r\nRECORD_BYTES = 100\r\nLABEL_RECORDS = 1\r\n"
           "GROUP = INSTRUMENT_STATE_PARMS\r\n"
           "  INSTRUMENT_TEMPERATURE = (28.15 <degC>, -15.04 <degC>, -15.41 <degC>, -16.23 <degC>)\r\n"
           "  INSTRUMENT_TEMPERATURE_NAME = (\"DEA\", \"HEAD_FPA\", \"HEAD_HTR_1\", \"HEAD_HTR_2\")\r\n"
           "END_GROUP = INSTRUMENT_STATE_PARMS\r\nEND\r\n")
    p = tmp_path / "ZL0_0461_0707874602_394RAD_N0260630ZCAM07114_0340LMA01.IMG"
    p.write_bytes(lab.encode("ascii"))
    try:
        L, _ = read_pds(p, load_image=False)
    except Exception as e:                                   # noqa: BLE001  (a reader that needs a full product)
        pytest.skip(f"minimal label not readable: {e}")
    assert camera_temperature(L, "Z", "L") == pytest.approx(-15.04)


def test_thermal_stage_has_no_rig_temperature_term():
    import inspect
    from mppp.sfm import thermal as TH
    from mppp.sfm.reconstruction import reconstruct
    src = inspect.getsource(reconstruct)
    assert 'refine_rig == "auto"' in src and "rig_slopes" not in src
    assert "rig_slopes" not in inspect.signature(TH.thermal_stage).parameters
    assert "rig_slopes" not in inspect.signature(TH.split_by_temperature).parameters
    assert not hasattr(TH, "rig_slopes_for_project") and not hasattr(TH, "thermal_adjust")


def test_thermal_model_for_project_is_the_start_cameras_model():
    from types import SimpleNamespace
    from mppp.sfm.thermal import thermal_model_for_project
    th = {"T_median_degC": -12.5, "T0_degC": -20.0, "fx_ppm_per_degC": 40.0, "fy_ppm_per_degC": 45.0,
          "cx_px_per_degC": 0.0, "cy_px_per_degC": 0.0, "k1_per_degC": -7e-6}
    p = SimpleNamespace(settings={"navcam_cameras": {"NL": {"thermal": th}, "NR": {}}})
    m = thermal_model_for_project(p)
    assert set(m) == {"NL"} and m["NL"]["T0_degC"] == -12.5 and m["NL"]["fx_ppm_per_degC"] == 40.0
    assert m["NL"]["k1_per_degC"] == -7e-6
    assert thermal_model_for_project(SimpleNamespace(settings={})) is None


ROOT = Path(__file__).resolve().parents[1]


def test_thermal_defaults_and_notebook():
    import inspect
    from mppp.sfm import thermal as T
    from mppp.sfm.reconstruction import reconstruct
    assert T.THERMAL_BIN_DEG == 5.0 and T.THERMAL_MIN_IMAGES == 5
    sig = inspect.signature(T.thermal_stage).parameters
    assert sig["bin_deg"].default == 5.0 and sig["min_images"].default == 5
    assert inspect.signature(reconstruct).parameters["thermal_min_images"].default == 5
    from mppp.pipeline import Settings
    s = Settings()
    assert s.thermal_bins_deg == 5 and s.thermal_min_images == 5 and s.thermal_free == ("fx", "fy", "k1")


def test_temperature_bins_merge_as_documented():
    from mppp.sfm.thermal import temperature_bins
    # 5 degC bins: -20..-15 (6 images), -15..-10 (2 images), -5..0 (8 images)
    T = {1: -19.0, 2: -18.0, 3: -17.0, 4: -12.0, 5: -3.0, 6: -2.0, 7: -1.5, 8: -1.0}
    n = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}
    b = temperature_bins(T, n, 5.0, 5)
    assert b[4] == (-20.0, -10.0) and b[1] == (-20.0, -10.0)       # the small bin joins its nearer neighbour
    assert b[5] == (-5.0, 0.0)
    # across an empty bin: -25..-20 (2 images) joins -15..-10 (the next occupied one)
    b = temperature_bins({1: -22.0, 2: -12.0, 3: -11.0, 4: -13.0}, {1: 2, 2: 2, 3: 2, 4: 2}, 5.0, 5)
    assert set(b.values()) == {(-25.0, -10.0)}
    # a tie goes to the colder neighbour
    b = temperature_bins({1: -17.0, 2: -12.0, 3: -7.0}, {1: 6, 2: 2, 3: 6}, 5.0, 5)
    assert b[2] == (-20.0, -10.0) and b[3] == (-10.0, -5.0)


def test_thermal_bin_lines_show_cx_cy_and_held_marks():
    from mppp.sfm.thermal import thermal_bin_lines
    rows = [{"camera": "NL_T-020-015", "images": 12, "T_median_degC": -17.2, "T_min_degC": -19.0, "T_max_degC": -15.1,
             "fx": 2956.5, "fy": 2956.4, "cx": 2561.0, "cy": 1920.5,
             "start": {"fx": 2956.0, "fy": 2956.0, "cx": 2561.0, "cy": 1920.5}}]
    lines = thermal_bin_lines(rows, ("fx", "fy"))
    head, row = lines[0], lines[1]
    assert "cx*" in head and "cy*" in head and "fx*" not in head
    assert "2561.00" in row and "1920.50" in row and "+0.50" in row
    assert "held at the start (cx, cy)" in lines[-1]
    lines = thermal_bin_lines(rows, ("fx", "fy", "cx", "cy"))
    assert "*" not in lines[0] and not lines[-1].lstrip().startswith("*")
    lines = thermal_bin_lines(rows, ("fx", "fy"), hold=True)
    assert "fx*" in lines[0]


def test_split_by_temperature_records_the_start(tmp_path):
    pytest.importorskip("pycolmap")
    src = (ROOT / "src" / "mppp" / "sfm" / "thermal.py").read_text(encoding="utf-8")
    assert '"start": {"fx": float(p[0]), "fy": float(p[1]), "cx": float(p[2]), "cy": float(p[3]),' in src   # v0p75.1: + k1


def test_thermal_stage_refuses_distortion_terms():
    """v0p50: a temperature bin fits at most fx, fy, cx, cy - never the Navcam distortion; v0p75.1: except k1."""
    from mppp.sfm.thermal import thermal_stage
    with pytest.raises(ValueError, match="distortion"):
        thermal_stage(None, None, {}, free=("fx", "k2"))


# stage 2 after the temperature bins of stage 1 (v0p62)


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


def test_thermal_stage_holds_bins_with_few_images():
    from mppp.sfm import thermal as T
    assert T.THERMAL_MIN_BIN_IMAGES == 3


def test_eye_key():
    from mppp.sfm.thermal import eye_key
    assert eye_key("NL#1") == "NL" and eye_key("NR#2_T-20..-10") == "NR" and eye_key("NL_T-20..-10") == "NL"
    assert eye_key("ZL034_F00600") == "ZL034_F00600" and eye_key("NL") == "NL"


# fine bins over -25..-5 degC, k1 free (v0p75.1)


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


def test_bin_rig_gets_its_own_camera_for_an_eye_without_images(tmp_path):
    """v0p86 (Marble Mountain, 19 Oct): an eye without images in a temperature bin still gets a (held) bin camera on
    that bin's rig - a camera may sit on one rig only in the COLMAP database ("SQLite error: constraint failed" in
    sync_database), so the bin rig must not carry the eye's original camera."""
    from mppp.sfm.thermal import image_temperatures, split_by_temperature
    proj, rec, noise = _synthetic_block(tmp_path)
    cold = {r["name"] for r in proj.images if r["station"].endswith("0000")}
    temps = {k: v for k, v in image_temperatures(proj).items() if not (k.startswith("NR") and k in cold)}
    r1, p1, rows = split_by_temperature(rec, proj, temps, bin_deg=10, min_images=1)
    seen = {}
    for rid, rig in r1.rigs.items():
        for s in [rig.ref_sensor_id] + list(rig.non_ref_sensors):
            assert s.id not in seen, f"camera {s.id} on rigs {seen[s.id]} and {rid}"
            seen[s.id] = rid
    held = [k for k, c in p1.cameras.items() if c.get("n_images") == 0 and c.get("thermal_bin")]
    assert held and all(set(p1.cameras[k]["fixed_params"]) >= {"fx", "fy", "cx", "cy"} for k in held)
    assert all(k not in {r["camera"] for r in rows} for k in held)
