"""v0p31: Navcam temperature bins (thermal stage), temperature-corrected consensus, 1 px outlier floor,
notebook order (04 camera models, 05 error analysis), notebook 03 defaults."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


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


def test_interpolate_temperatures_in_sclk():
    from mppp.sfm.thermal import interpolate_temperatures
    mk = lambda sol, sclk: f"NLF_{sol:04d}_{sclk:010d}_000ECM_N0000000NCAM00000_01_095J01"       # noqa: E731
    samples = {mk(100, 1000000): {"NL": -30.0, "NR": -31.0}, mk(100, 1001000): {"NL": -20.0, "NR": -21.0}}
    out = interpolate_temperatures(samples, [mk(100, 1000000), mk(100, 1000500), mk(100, 1002000), mk(100, 1009000),
                                             mk(101, 1100000)])
    assert out[mk(100, 1000000)]["source"] == "label" and out[mk(100, 1000000)]["NL"] == -30.0
    assert out[mk(100, 1000500)]["source"] == "interpolated" and abs(out[mk(100, 1000500)]["NR"] + 26.0) < 1e-9
    assert out[mk(100, 1002000)]["source"] == "nearest" and out[mk(100, 1002000)]["NL"] == -20.0
    assert mk(100, 1009000) not in out and mk(101, 1100000) not in out          # beyond the gap / sol without samples


def test_outlier_floor_is_one_pixel():
    from mppp.sfm.reconstruction import OUTLIER_DEFAULTS
    assert OUTLIER_DEFAULTS["min_residual_px"] == 1.0


def _fake_solutions(ppm=60.0, T0=-20.0):
    from mppp.sfm.calibration import Camera, Solution, reference_camera
    ship = reference_camera("NL", "rational")
    sols = {}
    for name, temps in (("A", (-40.0, -25.0)), ("B", (-15.0,)), ("C", (-30.0, -5.0))):
        cams, pcams = {}, {}
        for T in temps:
            key = f"NL_T{int(T):+04d}"
            p = np.array(ship.params, float)
            p[:2] *= 1 + 1e-6 * ppm * (T - T0)
            cams[key] = Camera(key, "NL", ship.model, 5120, 3840, p, np.array(ship.params, float) * 0.999, None, 20, 50000)
            pcams[key] = {"group": "NL", "temperature_median_degC": T, "thermal_bin": True}
        sols[name] = Solution(name, Path("."), {"cameras": pcams, "settings": {}}, {}, cams, {})
    return sols, ship


def test_consensus_with_thermal_model_is_the_camera_at_T0():
    from mppp.sfm.calibration import consensus_camera, reference_differences, thermal_model, thermal_scale
    sols, ship = _fake_solutions()
    th = {"NL": {"ppm_per_degC": 60.0, "T0_degC": -20.0}}
    c = consensus_camera(sols, "NL", "rational", 1000, thermal=th)
    assert np.allclose(c.params, ship.params, rtol=0, atol=1e-6) and c.thermal == th["NL"] and not c.excluded
    plain = consensus_camera(sols, "NL", "rational", 1000)
    assert abs(plain.params[0] - ship.params[0]) > 1e-3                      # without the model: the mean temperature
    rows = reference_differences(sols, "NL", c, min_observations=1000, thermal=th)
    assert len(rows) == 5 and max(r["rms_px"] for r in rows) < 1e-3
    assert all(abs(r["thermal_scale"] - thermal_scale(th, "NL", r["T_degC"], to_T0=False)) < 1e-12 for r in rows)
    # model from a fit: T0 is the observation-weighted mean temperature of the cameras
    m = thermal_model(sols, {"NL": {"ppm_per_degC": 55.0, "ppm_sd": 5.0}}, None, "auto")
    assert m["NL"]["source"] == "within" and abs(m["NL"]["T0_degC"] - np.mean([-40, -25, -15, -30, -5])) < 1e-9
    assert thermal_model(sols, None, None, "fixed", 70.0)["NL"]["ppm_per_degC"] == 70.0
    assert thermal_model(sols, None, None, "auto") is None


def test_fit_focal_temperature_uses_within_scape_changes_only():
    from mppp.sfm.thermal import fit_focal_temperature
    rows = []
    for sc, off, temps in (("A", 0.0, (-40, -30, -20)), ("B", 3.0, (-25, -10)), ("C", -2.0, (-35, -15))):
        for T in temps:
            rows.append({"scape": sc, "eye": "NL", "T_median_degC": T, "fx": 2955.0 + off + 0.08 * T, "observations": 1e5})
    f = fit_focal_temperature(rows, "fx")["NL"]
    assert abs(f["px_per_degC"] - 0.08) < 1e-9 and f["scapes"] == 3 and f["residual_rms_px"] < 1e-9
    assert abs(f["offsets"]["B"] - f["offsets"]["A"] - 3.0) < 1e-9


def test_merge_temperature_bins_and_bin_rows(tmp_path):
    from mppp.sfm.calibration import merge_temperature_bins, thermal_bin_rows
    rows = [{"scape": "A", "camera": "NL_T-040-030", "eye": "NL", "images": 10, "n_temp": 10, "observations": 1000,
             "temperature_bin": True, "temp_median_degC": -35.0, "temp_min_degC": -38.0, "temp_max_degC": -31.0,
             "fx": 2954.0, "fy": 2954.0},
            {"scape": "A", "camera": "NL_T-030-020", "eye": "NL", "images": 30, "n_temp": 30, "observations": 3000,
             "temperature_bin": True, "temp_median_degC": -25.0, "temp_min_degC": -29.0, "temp_max_degC": -21.0,
             "fx": 2956.0, "fy": 2956.0},
            {"scape": "B", "camera": "NL", "eye": "NL", "images": 5, "n_temp": 5, "observations": 500,
             "temperature_bin": False, "temp_median_degC": -10.0, "temp_min_degC": -12.0, "temp_max_degC": -8.0,
             "fx": 2957.0, "fy": 2957.0}]
    m = {r["scape"]: r for r in merge_temperature_bins(rows)}
    assert len(m) == 2 and abs(m["A"]["fx"] - 2955.5) < 1e-9 and abs(m["A"]["temp_median_degC"] + 27.5) < 1e-9
    assert m["A"]["temp_min_degC"] == -38.0 and m["A"]["temp_max_degC"] == -21.0 and m["A"]["images"] == 40
    from mppp.sfm.calibration import Solution
    th = {"held": False, "rows": [{"eye": "NL", "camera": "NL_T-030-020", "T_median_degC": -25.0, "fx": 1.0, "fy": 1.0,
                                   "observations": 10}]}
    sols = {"Rockytop": Solution("Rockytop", tmp_path, {"settings": {"thermal": th}}, {}, {}, {}),
            "Held": Solution("Held", tmp_path, {"settings": {"thermal": dict(th, held=True)}}, {}, {}, {})}
    exp = tmp_path / "exp.json"
    exp.write_text(json.dumps({"scapes": {"rockytop": {"rows": [{"eye": "NL"}]},
                                          "olifants": {"rows": [{"eye": "NR", "camera": "NR_T-020-010"}]}}}))
    rr = thermal_bin_rows(sols, exp)
    assert [(r["scape"], r["source"]) for r in rr] == [("Rockytop", "thermal stage"), ("olifants", "experiment")]


def _synthetic_block(tmp_path):
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    for i, r in enumerate(proj.images):
        r["camera_temperature_degC"] = -32.0 if r["station"].endswith("0000") else -14.0
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    return proj, rec, noise


def test_thermal_stage_splits_relabels_and_strips(tmp_path):
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.thermal import image_temperatures, thermal_stage, strip_thermal_bins
    proj, rec, noise = _synthetic_block(tmp_path)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=20)
    temps = image_temperatures(proj)
    assert len(temps) == len(proj.images) and all(v["source"] == "manifest" for v in temps.values())
    n_before = sum(len(im.points2D) for im in rec.images.values())
    rec2, rep = thermal_stage(rec, proj, temps, bin_deg=10, min_images=4, max_iterations=20, verbose=False)
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
    model = {"NL": {"ppm_per_degC": 60.0, "T0_degC": -20.0}, "NR": {"ppm_per_degC": 60.0, "T0_degC": -20.0}}
    rec2, rep = thermal_stage(rec, proj, image_temperatures(proj), bin_deg=10, min_images=4, hold=True,
                              thermal_model=model, max_iterations=5, verbose=False)
    assert rep["held"]
    for r in rep["rows"]:
        want = f0[r["eye"]] * (1 + 60e-6 * (r["T_median_degC"] + 20.0))
        assert abs(r["fx"] - want) < 1e-6 and abs(r["start_fx"] - want) < 1e-6


def test_notebook_order_and_defaults():
    nbformat = pytest.importorskip("nbformat")
    nbdir = ROOT / "notebooks"
    assert (nbdir / "04_camera_models.ipynb").is_file() and (nbdir / "05_error_analysis.ipynb").is_file()
    assert not (nbdir / "04_error_analysis.ipynb").exists() and not (nbdir / "05_camera_models.ipynb").exists()
    nb = nbformat.read(str(nbdir / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)
    assert ns["THERMAL_BINS_DEG"] == 10 and ns["THERMAL_MIN_IMAGES"] == 8
    assert ns["SITES"]["rochette"] == (179, 190) and ns["SITES"]["rio_chiquito"] == (1333, 1337) and ns["SITES"]["threeforks_south"] == (652, 683)
    assert ns["SITES"]["sid"] == (361, 378) and ns["SITES"]["airey_hill"] == (960, 991) and len(ns["SITES"]) == 18
    full = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for k in ("thermal_bins_deg=THERMAL_BINS_DEG", "thermal_model=thermal_model_for_project(proj)",
              "image_temperatures(proj", "navcam_cameras_fingerprint"):
        assert k in full, k
    cam = nbformat.read(str(nbdir / "04_camera_models.ipynb"), as_version=4)
    full = "\n".join(c.source for c in cam.cells if c.cell_type == "code")
    for k in ("thermal=THERMAL", "CAL.thermal_model(", "CAL.thermal_bin_rows(", "VMAX", "fig.colorbar(im, cax=cax"):
        assert k in full, k
    assert cam.cells[0].source.startswith("# MPPP — 04 Camera models")
    err = nbformat.read(str(nbdir / "05_error_analysis.ipynb"), as_version=4)
    assert err.cells[0].source.startswith("# MPPP — 05")


def test_reconstruct_ends_with_the_thermal_stage_and_a_rerun_strips_it(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    import copy
    import pycolmap
    from test_sfm import _build_rec, _synthetic
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
    rec2, rep = thermal_stage(rec, proj, image_temperatures(proj), bin_deg=10, min_images=4, max_iterations=5, verbose=False)
    h = assess_alignment(proj, rec2)
    rig = [c for c in h["checks"] if "rig" in c["check"]]
    assert rig and all(c["status"] != "fail" for c in rig)
    native_reconstruction(rec2, proj)
    s = export_for_error(proj, rec2, tmp_path / "error_input")
    summ = json.loads((tmp_path / "error_input" / "summary.json").read_text())
    assert {r["camera"] for r in rep["rows"]} <= set(summ["cameras_refined"])
