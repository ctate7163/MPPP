"""MPPP v0p80: the camera-model recipe (block pp hold, round-1 hold, k4 with k3, k1(T), single NR cx(T) carrier,
leave-one-out and pass-change gates, baseline study), sites v4 block records, mppp.maps, the block diagnosis."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
CMODS = ROOT / "src" / "mppp" / "data" / "cmods"


def _fisheye_cam():
    return {"model": "THIN_PRISM_FISHEYE", "width": 5120, "height": 3840,
            "params": [2956.0, 2956.0, 2591.0, 1944.0, 0.0428, -0.0012, 2e-4, 3e-4, -0.0098, 0.0, 0.0, 0.0]}


def test_navcam_pp_hold_adds_cx_cy_to_fixed():
    from mppp.sfm.project import NAVCAM_PP, navcam_distortion_terms
    assert NAVCAM_PP == "hold"
    c = navcam_distortion_terms(_fisheye_cam(), "hold", k4="consensus", pp="hold")
    assert {"cx", "cy"} <= set(c["fixed_params"]) and "fx" not in c["fixed_params"] and "fy" not in c["fixed_params"]
    assert c["pp"] == "hold" and "principal point held" in c["source"]
    r = navcam_distortion_terms(_fisheye_cam(), "hold", k4="consensus")          # the function default: refine
    assert "cx" not in r["fixed_params"]
    with pytest.raises(ValueError):
        navcam_distortion_terms(_fisheye_cam(), "hold", pp="maybe")


def test_k4_consensus_keeps_a_fitted_k4():
    from mppp.sfm.project import NAVCAM_K4, navcam_distortion_terms
    assert NAVCAM_K4 == "consensus"
    cam = _fisheye_cam()
    cam["params"][9] = 0.0021
    c = navcam_distortion_terms(cam, "hold", k4="consensus", pp="hold")
    assert c["params"][9] == pytest.approx(0.0021) and "k4" in c["fixed_params"] and "k3" in c["fixed_params"]


def test_notebook_v0p80_settings():
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for s in ('NOTEBOOK_FOR_MPPP = "v0p80"', 'NAVCAM_K4         = "consensus"', 'NAVCAM_PP         = "hold"',
              "ROUND1_HOLD       = True", '"full": True', "round1_hold=bool(ROUND1_HOLD)", "navcam_pp=NAVCAM_PP",
              '"navcam_pp": NAVCAM_PP', "rig_translation_sigma_m=RIG_TRANSLATION_SIGMA_M if RIG_REFINE is True"):
        assert s in src, s


def test_reconstruct_signature_round1_hold():
    import inspect
    from mppp.sfm.reconstruction import ROUND1_HOLD, reconstruct
    assert ROUND1_HOLD is True
    assert inspect.signature(reconstruct).parameters["round1_hold"].default is True


def test_k1_keypoint_map():
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm.navcal import k1_keypoint_map

    class P:
        images = [{"image_id": 1, "name": "a", "instrument": "NL"}, {"image_id": 2, "name": "b", "instrument": "NR"}]
    c = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=_fisheye_cam()["params"])
    k = np.array([[2591.0, 1944.0], [5000.0, 3800.0]])
    f = k1_keypoint_map(P, {"a": 10.0, "b": -20.0}, -20.0, {"NL": -3.8e-6, "NR": 0.0})
    d = f(1, k, c) - k
    assert np.allclose(d[0], 0, atol=1e-6)                       # the principal point does not move
    assert 0.1 < np.hypot(*d[1]) < 1.0                            # ~0.3 px at the corner for 30 degC
    assert np.allclose(f(2, k, c), k)                             # no slope for NR / at T0


def test_translation_tie_cost():
    pytest.importorskip("pyceres")
    from mppp.sfm.reconstruction import _translation_tie_class
    c = _translation_tie_class()(1e-3)
    a = np.array([0, 0, 0, 1, 0.42, 0.001, 0.002])
    b = np.array([0, 0, 0, 1, 0.41, 0.001, 0.0])
    res = np.zeros(3)
    ja, jb = np.zeros(21), np.zeros(21)
    assert c.Evaluate([a, b], res, [ja, jb])
    assert np.allclose(res, [10.0, 0.0, 2.0])
    assert ja.reshape(3, 7)[0, 4] == pytest.approx(1e3) and jb.reshape(3, 7)[2, 6] == pytest.approx(-1e3)


def _write_form(d: Path, form="v0p80", k4=0.0012, k1s=-3e-6):
    d.mkdir(parents=True, exist_ok=True)
    for f in ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json"):
        shutil.copy(CMODS / f, d / f)
    for eye in ("NL", "NR"):
        f = d / f"M2020_{eye}_fisheye_tangential.json"
        j = json.loads(f.read_text())
        j["params"][j["param_names"].index("k4")] = k4
        j["form"] = form
        j["thermal"]["k1_per_degC"] = k1s
        f.write_text(json.dumps(j))


def test_check_form_v0p80(tmp_path):
    from mppp.sfm.cmods_form import check_form, describe, thermal_terms
    _write_form(tmp_path / "a")
    r = check_form(tmp_path / "a")
    assert r["ok"] and r["form"] == "v0p80", r["errors"]
    _write_form(tmp_path / "b", form="v0p73")
    assert not check_form(tmp_path / "b")["ok"]                   # k4 != 0 needs the v0p80 form
    nl = json.loads((tmp_path / "a" / "M2020_NL_fisheye_tangential.json").read_text())
    assert thermal_terms(nl["thermal"])["k1"] == pytest.approx(-3e-6)
    assert any("k1" in ln for ln in describe(tmp_path / "a"))
    assert check_form()["ok"]                                     # the shipped models still pass


def test_consensus_change(tmp_path):
    from mppp.sfm.navcal_consensus import PASS_LIMITS, consensus_change
    _write_form(tmp_path / "a", k4=0.0)
    _write_form(tmp_path / "b", k4=0.0)
    ch = consensus_change(tmp_path / "a", tmp_path / "b")
    assert ch["converged"] and ch["worst"]["f_ppm"] == 0
    f = tmp_path / "b" / "M2020_NR_fisheye_tangential.json"
    j = json.loads(f.read_text())
    j["params"][0] *= 1 + 50e-6
    j["params"][2] += 0.02
    f.write_text(json.dumps(j))
    ch = consensus_change(tmp_path / "a", tmp_path / "b")
    assert not ch["converged"] and ch["worst"]["f_ppm"] == pytest.approx(50, abs=0.1)
    assert ch["NR"]["dcx_px"] == pytest.approx(0.02) and PASS_LIMITS["pp_px"] == 0.05


def test_k1_slopes_from_blocks(tmp_path):
    from mppp.sfm.navcal_consensus import k1_slopes_from_blocks
    rng = np.random.default_rng(0)
    cfg = {}
    for b, off in (("A", 0.0), ("B", 2e-4), ("C", -1e-4)):
        rows = []
        for eye, s in (("NL", -4e-6), ("NR", 0.0)):
            for T in np.linspace(-45, -5, 8):
                rows.append({"camera": f"{eye}_T", "eye": eye, "T_median_degC": float(T), "free": ["fx", "fy", "k1"],
                             "k1": 0.043 + off + s * T + rng.normal(0, 2e-6), "observations": 5000})
        d = tmp_path / b / "colmap"
        d.mkdir(parents=True)
        (d / "project.json").write_text(json.dumps({"settings": {"thermal": {"rows": rows}}, "images": []}))
        cfg[b] = str(tmp_path / b)
    r = k1_slopes_from_blocks(cfg, verbose=False)
    assert r["fits"]["NL"]["slope"] == pytest.approx(-4e-6, abs=3e-7)
    assert r["apply"]["NL"] == pytest.approx(r["fits"]["NL"]["slope"]) and r["apply"]["NR"] == 0.0


def test_joint_kw_and_thermal_values_k1():
    from mppp.sfm.navcal_consensus import _joint_kw, _thermal_values
    v = _thermal_values({"ppm_per_degC": 38.0, "k1_slopes": {"NL": -4e-6, "NR": 1e-6}})
    assert v["NL_k1"] == -4e-6
    kw = _joint_kw(v, None, True)
    assert kw["k1_slopes"] == {"NL": -4e-6, "NR": 1e-6}
    assert "k1_slopes" not in _joint_kw(_thermal_values({"ppm_per_degC": 38.0}), None, True)


def test_fit_consensus_accepts_v0p80():
    import inspect
    from mppp.sfm.navcal_consensus import LOO_LIMITS, fit_consensus
    p = inspect.signature(fit_consensus).parameters
    assert {"k1_slopes", "leave_one_out", "loo_iterations"} <= set(p)
    assert LOO_LIMITS == {"centre_px": 0.05, "corner_px": 0.10}


def test_baseline_combine():
    from navcam_baseline_study import combine_rows
    rows = [{"site": s, "baseline_m": 0.4244 + d, "baseline_sd_m": 0.0003, "T_median_degC": T, "sol_median": sol}
            for s, d, T, sol in (("a", 0.0008, -20, 600), ("b", 0.0009, -15, 800), ("c", 0.0007, -25, 1600),
                                 ("d", 0.0010, -10, 1800))]
    r = combine_rows(rows, 0.4244)
    assert r["adopt"] and r["difference_mm"] == pytest.approx(0.85, abs=0.05) and r["chi2_dof"] < 2
    r2 = combine_rows(rows, 0.4252)
    assert not r2["adopt"]
    assert combine_rows(rows[:1], 0.4244)["adopt"] is False


def test_pipeline_order_navcam_first():
    import run_pipeline
    s = run_pipeline.STAGES
    assert s.index("navcam_consensus") < s.index("navcam_runs") < s.index("navcam_consensus_2") < s.index("zcam_start")
    assert s.index("zcam_final_runs") < s.index("baseline_study") < s.index("report")
    assert run_pipeline.CONSENSUS_FORM == "v0p80"


def test_best_candidate_moves_yaw_into_nr_cx(tmp_path, monkeypatch):
    import navcam_best_candidate as nbc
    src = tmp_path / "cmods"
    src.mkdir()
    for f in ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json"):
        shutil.copy(CMODS / f, src / f)
    monkeypatch.setenv("MPPP_CMODS", str(src))
    rr = tmp_path / "rig_residuals.json"
    rr.write_text(json.dumps({"fits": {"yaw": {"vs_T": {"slope": -0.339, "se": 0.134}}}}))
    rep = nbc.build(rr, tmp_path / "out")
    fx = json.loads((src / "M2020_NR_fisheye_tangential.json").read_text())["params"][0]
    assert rep["added"] == pytest.approx(fx * np.radians(-0.339e-3))
    assert rep["form_ok"]
    nr = json.loads((tmp_path / "out" / "M2020_NR_fisheye_tangential.json").read_text())
    assert nr["thermal"]["cx_px_per_degC"] == pytest.approx(rep["NR_cx_slope_before"] + rep["added"])
    with pytest.raises(ValueError):                               # too weak a term is not moved
        rr.write_text(json.dumps({"fits": {"yaw": {"vs_T": {"slope": -0.1, "se": 0.134}}}}))
        nbc.build(rr, tmp_path / "out2")


def test_sites_block_record_and_groups():
    from mppp.sfm.sites import block_sites, check_block_record, load_site_table, site_block, site_group, validate_site_table
    assert check_block_record({"number": 3, "half_size_m": 30, "centre": {"lon": 77.4, "lat": 18.4}}) == []
    assert check_block_record({"number": 0}) and check_block_record([1])
    t = load_site_table()
    assert t["version"] == 4
    b = block_sites(t)
    assert len(b) == 34 and b[0] == "blk01_oeb_landing_site" and b[-1] == "blk34_marble_mountain"
    assert site_block("blk03_chal_rocks", t)["legacy_site"] == "chal_rocks_large"
    assert len(site_group("blocks_all")) == 34
    assert "chal_rocks_large" in site_group("navcam_baseline")
    err, _ = validate_site_table()
    assert not err


def test_maps_blocks_from_snapshot():
    from mppp import maps
    res = maps.build_from_waypoints(ROOT / "src" / "mppp" / "data" / "M20_waypoints.json")
    b = {x["number"]: x for x in res["blocks"]}
    assert len(b) == 34 and not any(x.get("missing") for x in b.values())
    # N× counts of the waypoint-map work (15 Oct) that the packaged snapshot (to sol 1962) can reproduce
    expected = {1: 10, 2: 9, 3: 9, 4: 5, 5: 2, 6: 6, 7: 5, 8: 10, 10: 6, 11: 6, 12: 7, 13: 7, 14: 2, 15: 8, 16: 3,
                17: 6, 18: 2, 19: 8, 20: 5, 21: 3, 22: 8, 23: 3, 24: 5, 25: 2, 26: 4, 27: 7, 28: 7, 29: 9, 30: 8,
                31: 6, 32: 4}
    got = {k: b[k]["navcam_360s"] for k in expected}
    assert got == expected
    assert b[9]["navcam_360s"] in (7, 8)                          # Three Forks North: 8 in the 15 Oct list
    assert 413 not in b[9]["end_of_drive_sols"]
    assert b[17]["note"].startswith("square shifted -20")
    assert len(res["clusters"]) == 22
    assert set(res["groups"]) >= {"blocks_all", "blocks_crater_rim_west"}
    assert res["sites"]["blk28_bell_island"]["sols"] == [1451, 1478]


def test_maps_visits_and_merge(tmp_path):
    from mppp.maps import visits_of
    from mppp.maps.export import compare_blocks, merge_into_sites_file
    assert visits_of([91, 99, 100, 360, 361, 371]) == [[360, 371], [91, 100]]     # 11 sols before 9
    assert visits_of([5, 6, 100, 120]) == [[100, 120], [5, 6]]           # longest first
    f = tmp_path / "sites.json"
    shutil.copy(ROOT / "src" / "mppp" / "data" / "sites.json", f)
    before = f.read_text().splitlines()
    merge_into_sites_file(f, {"blk99_test": {"sols": [1, 2], "label": "T", "zcam": [], "block": {"number": 99}}},
                          {"blocks_test": ["blk99_test"]})
    d = json.loads(f.read_text())
    assert "blk99_test" in d["sites"] and d["groups"]["blocks_test"] == ["blk99_test"]
    after = f.read_text().splitlines()
    assert sum(1 for x in before if x not in after) <= 2                # only the comma lines change
    assert compare_blocks({"a": {"sols": [1, 2]}}, {"a": {"sols": [1, 3]}})[0].startswith("a: sols")


def test_diagnose_visits_helper():
    from mppp.sfm.diagnose import _visit_of, _visits
    rows = [{"sol": s, "instrument": "NL"} for s in (91, 99, 101, 360, 371)]
    v = _visits(rows, {"navcam_rig_epochs_breaks": [300.0]})
    assert [x["sols"] for x in v] == [[91, 101], [360, 371]]
    assert _visit_of(365, v) == 1 and _visit_of(200, v) == 0


def test_diagnose_on_an_upload_if_present(tmp_path):
    p = Path("/mnt/user-data/uploads/colmap/mars2020_sol_0361_chal_rocks_sid_colmap/colmap")
    if not (p / "error_input" / "native").is_dir():
        pytest.skip("no Chal Rocks Sid run here")
    from mppp.sfm.diagnose import write_report
    r = write_report(p, out_dir=tmp_path)
    res = r["result"]["residuals"]["by_class"]
    assert res["across visits"]["eps_px"] > res["within station"]["eps_px"]
    assert (tmp_path / "diagnose.png").is_file()
