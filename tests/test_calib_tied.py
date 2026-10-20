"""The one-adjustment Navcam joint (mppp.calib.tied) on a synthetic block: units, ties, result, files."""
import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")


def _tied_setup(tmp_path):
    from helpers import _build_rec, _synthetic
    from mppp.calib import tied as T
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path / "blk", noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    # sols: the first station early in the mission (before the knot), the second late; temperatures in two bins
    temps = {}
    for r in proj.images:
        r["sol"] = 120 if r["station"] == "S001D0000" else 900
        temps[r["name"]] = -31.0 if int(r["sclk_key"][:10]) % 2 else -9.0
    idx = {"scape_of_image": {int(r["image_id"]): "blk" for r in proj.images},
           "images": {"blk": [int(r["image_id"]) for r in proj.images]}, "temps": temps}
    st = T.start_values()
    # the synthetic cameras are FULL_OPENCV (fx fy cx cy k1 k2 p1 p2 k3 k4 ...): the ties read the same indices
    for eye, cid in (("NL", 1), ("NR", 2)):
        st["G"][eye] = np.asarray(rec.cameras[cid].params, float)[:10].copy()
        st["S"][eye] = np.array([1e-6 * st["G"][eye][0] * 40.0, 1e-6 * st["G"][eye][1] * 45.0, 0.01, 0.0, 0.0])
        st["E"][eye] = np.zeros(4)
    st["Q0"] = T.q_xyz(rigT[0]) - 0.5 * T._MDEG * (st["D"] * 0 + st["H"] * 0) / 1000.0
    st["R"], st["t"] = rigT
    return proj, rec, idx, st, T


def test_tied_units_and_law(tmp_path):
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    eyes = {(u["eye"], u["early"]) for u in units["cameras"].values()}
    assert ("NL", True) in eyes and ("NR", False) in eyes              # both eras present
    assert len(units["cameras"]) == 8                                   # 2 eyes x 2 bins x 2 eras
    assert len(units["rigs"]) == 4                                      # 2 epochs x 2 bins (one era each)
    for u in units["rigs"].values():
        assert u["frames"] >= 1 and u["stations"] == 1
    # every unit starts at the law
    for cid, u in units["cameras"].items():
        p = np.asarray(rec2.cameras[cid].params)
        assert np.allclose(p[:10], T.camera_params(st, u["eye"], u["T"], u["early"])[:10])
    rows = T.unit_rows(rec2, proj2, idx, units, st, T.make_ties(st, units)[1])
    assert all(abs(r["dfx_ppm"]) < 1e-6 for r in rows["cameras"])
    assert all(abs(r["dpitch_mdeg"]) < 1e-6 for r in rows["rigs"])


def test_tied_adjust_and_files(tmp_path):
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba, g, stats = T.adjust_tied(rec2, proj2, idx, st, units, iterations=20, covariance=True, passes=2, verbose=False)
    assert np.isfinite(ba["final_cost"]) and stats["median_px"] < 2.0
    held = (ba.get("extra") or {}).get("held") or {}
    assert held.get("Q0") == [1] and held.get("H") == [0, 1, 2]         # yaw held; two sols only: no hinge
    # the bin cameras follow the law of the global terms (the ties are effectively exact)
    rows = T.unit_rows(rec2, proj2, idx, units, st, g)
    assert max(abs(r["dfx_ppm"]) for r in rows["cameras"]) < 5.0
    assert max(abs(r["dcx_px"]) for r in rows["cameras"]) < 0.02
    assert max(abs(r["dpitch_mdeg"]) for r in rows["rigs"]) < 0.05
    res = T.globals_result(st, g, ba, units)
    res.update({"blocks": ["blk"], "residuals": stats, "final_cost": ba["final_cost"], "iterations": ba["iterations"],
                "seconds": 0.0, "units": rows})
    assert np.isfinite(res["NL"]["fx_ppm_per_degC"]) and res["rig"]["rate_mdeg_per_1000sol"]["yaw"] == pytest.approx(float(st["D"][1]), abs=50)
    # one normal matrix: the global terms have finite sd (the 0.83 prototype's came out infinite)
    assert res["covariance_error"] is None
    assert 0 < res["NL"]["sd"]["fx_ppm_per_degC"] < 5 and 0 < res["NR"]["early_mission"]["sd_px"]["cx"] < 0.1
    assert 0 < res["rig"]["sd_rate"]["yaw"] < 5 and np.isnan(res["rig"]["sd_sol0_mdeg"]["yaw"])
    assert T.describe_result(res)
    out = T.write_tied(res, tmp_path / "cand", note="test")
    from mppp.calib import forms
    nl = json.loads((out / "M2020_NL_fisheye_tangential.json").read_text())
    rig = json.loads((out / "M2020_N_rig.json").read_text())
    assert nl["early_mission"]["per_eye"] and "dfx_px" in nl["early_mission"]
    assert nl["early_mission"]["applied_terms"] == ["fx", "fy", "cx", "cy"]     # the projects apply the offsets
    from mppp.calib.constrain import camera_at
    _, p_early, applied = camera_at(nl, -20.0, sol=100)
    _, p_late, _ = camera_at(nl, -20.0, sol=1000)
    assert p_early[2] - p_late[2] == pytest.approx(nl["early_mission"]["dcx_px"]) and "early_mission" in applied
    assert rig["drift"]["sol0"] == 0.0 and rig["drift"]["knot_ref"] == 0.0 and "roll_mdeg_per_sol" in rig["drift"]
    assert not forms.validate_rig(rig)
    assert (out / "tied_joint.json").exists() and (out / "consensus.json").exists()
    from mppp.calib.fit import check_navcam
    rep = check_navcam(out)
    names = [c["check"] for c in rep["checks"]]
    assert "block agreement" in names and "leave-one-block-out" not in names
    # the diagnostic pass: per-block units, globals held, no ties
    diag = T.diagnostic_pass(rec, proj, idx, st, g, bin_deg=10.0, iterations=10, verbose=False)
    assert diag["cameras"] and all(c["scape"] == "blk" for c in diag["cameras"])
    assert diag["rigs"] and all("dpitch_mdeg" in r for r in diag["rigs"])


def test_tied_figure_inputs_feed_the_refined_figures(tmp_path):
    pytest.importorskip("pyceres")
    pytest.importorskip("matplotlib")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba, g, stats = T.adjust_tied(rec2, proj2, idx, st, units, iterations=10, covariance=False, passes=1, verbose=False)
    res = T.globals_result(st, g, ba, units)
    res.update({"blocks": ["blk"], "residuals": stats, "final_cost": ba["final_cost"], "iterations": ba["iterations"],
                "seconds": 0.0, "units": T.unit_rows(rec2, proj2, idx, units, st, g)})
    res["diagnostics"] = T.diagnostic_pass(rec, proj, idx, st, g, bin_deg=10.0, iterations=5, verbose=False)
    fi = T.figure_inputs(res, min_images=1)
    assert fi["bins"] and fi["rig_residuals"]["blocks"] and "pp_slopes" in fi["consensus"]["thermal"]
    out = T.write_tied(res, tmp_path / "cand")
    assert (out.parent / "rig_residuals.json").is_file() and (out / "thermal_bins.json").is_file()
    from mppp.calib import constrain as C, load_camera
    rig = json.loads((out / "M2020_N_rig.json").read_text())
    inuse = {e: load_camera("N", e[1]) for e in ("NL", "NR")}
    cons = json.loads((out / "consensus.json").read_text())
    C.plot_rig_trends(fi["rig_residuals"], cons, rig["drift"], tmp_path / "rig.png", cx_inuse_px_per_degC=-0.05)
    C.plot_intrinsics_vs_temperature(cons, inuse, fi["bins"], fi["rig_residuals"], tmp_path / "intr.png")
    assert (tmp_path / "rig.png").stat().st_size > 1000 and (tmp_path / "intr.png").stat().st_size > 1000


def test_sd_fallback_from_the_bin_cameras(tmp_path):
    """When the problem's covariance of the global blocks is unusable (20 Oct: infinite on 22 blocks), the sd of
    the camera globals is propagated from the bin cameras' own covariances."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba, g, stats = T.adjust_tied(rec2, proj2, idx, st, units, iterations=15, covariance=True, passes=1, verbose=False)
    cov = ba["covariance"]
    assert any(k.startswith("camera:") for k in cov["blocks"])
    for k in [k for k in cov["blocks"] if k.startswith("extra:")]:
        cov["blocks"][k] = np.full_like(cov["blocks"][k], np.inf)
    res = T.globals_result(st, g, ba, units)
    assert res.get("sd_source", {}).get("NL") and np.isfinite(res["NL"]["sd"]["fx_ppm_per_degC"]) and res["NL"]["sd"]["fx_ppm_per_degC"] > 0
    assert np.isfinite(res["NR"]["early_mission"]["sd_px"]["cx"])


def test_pointing_pass_rigs_at_the_trend(tmp_path):
    """The principal points per block (pointing pass): one camera per block, eye and era with fx, fy, cx, cy free,
    the rig epochs held at the trend, a 1 px prior on NL's principal point; finite sd, the stereo rows."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba, g, stats = T.adjust_tied(rec2, proj2, idx, st, units, iterations=10, covariance=False, passes=1, verbose=False)
    stl = dict(st, G={"NL": g["G_NL"], "NR": g["G_NR"]}, S={"NL": g["S_NL"], "NR": g["S_NR"]},
               E={"NL": g["E_NL"], "NR": g["E_NR"]}, Q0=g["Q0"], D=g["D"], H=g["H"])
    pt = T.pointing_pass(rec, proj, idx, stl, g, iterations=10, verbose=False)
    assert len(pt["cameras"]) == 4                                   # 2 eyes x 2 eras, one block
    assert all(c["sd_cx_px"] is not None and 0 < c["sd_cx_px"] < 1.0 for c in pt["cameras"])
    assert len(pt["stereo"]) == 2 and all(abs(s["dcx_stereo_px"]) < 0.5 for s in pt["stereo"])


def test_isotropic_focal_slope(tmp_path):
    """One ppm/degC per eye by default (fy's slope tied to fx's, 20 Oct); ``iso_f=False`` frees fy's; the k1(T)
    significance line (z, corner displacement)."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba, g, stats = T.adjust_tied(rec2, proj2, idx, st, units, iterations=15, covariance=True, passes=1, verbose=False)
    assert 1 in ((ba.get("extra") or {}).get("held") or {}).get("S_NL", [])
    res = T.globals_result(st, g, ba, units)
    res["units"] = T.unit_rows(rec2, proj2, idx, units, st, g)
    for eye in ("NL", "NR"):
        assert res[eye]["fy_ppm_per_degC"] == pytest.approx(res[eye]["fx_ppm_per_degC"], abs=1e-9)
        assert res[eye]["sd"]["fy_ppm_per_degC"] == pytest.approx(res[eye]["sd"]["fx_ppm_per_degC"], rel=1e-6)
    assert res["isotropic_f_slope"]
    ks = T.k1_significance(res)
    assert set(ks) == {"NL", "NR"} and all(np.isfinite(v["corner_px"]) and v["span_degC"] > 0 for v in ks.values())
    assert any("isotropic" in line for line in T.describe_result(res))
    # the bins still follow the law with fy on fx's slope
    assert max(abs(r["dfy_ppm"]) for r in res["units"]["cameras"]) < 5.0
    rec3, proj3, units3 = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba3, g3, _ = T.adjust_tied(rec3, proj3, idx, st, units3, iterations=15, covariance=False, passes=1, verbose=False,
                               iso_f=False)
    res3 = T.globals_result(st, g3, ba3, units3)
    assert not res3["isotropic_f_slope"]
    assert 1 not in ((ba3.get("extra") or {}).get("held") or {}).get("S_NL", [])
