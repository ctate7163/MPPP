"""Navcam temperature model and thermal stage (mppp.sfm.thermal). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
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
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    ns = {}
    params = next("".join(c["source"]) for c in nb["cells"] if "parameters" in c.get("metadata", {}).get("tags", []))
    assert "THERMAL_BINS_DEG  = 5 " in params and "THERMAL_MIN_IMAGES = 5 " in params
    assert 'THERMAL_FREE = ("fx", "fy", "k1")' in params and "thermal_free=THERMAL_FREE" in src


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
