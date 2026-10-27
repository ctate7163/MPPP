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
    # 24 Oct: only the median residual gates against the models in use (another block set); rms and p95 report
    res_checks = {c["check"]: c for c in rep["checks"] if c["check"].startswith("residual ")}
    assert set(res_checks) == {"residual median_px", "residual rms_px", "residual p95_px"}
    for k in ("residual rms_px", "residual p95_px"):
        assert res_checks[k]["ok"] and (res_checks[k]["note"].endswith("(report only: another block set)")
                                        or res_checks[k]["note"] == "no reference")
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


def test_diagnostic_bins_merge_thin_cold_bins():
    """22 Oct (Christian): the cold morning images (1-3 per 2 degC bin of a block) were all below the figure cut;
    the diagnostic pass merges a block's adjacent thin bins until each eye has 8 images, at most 10 degC wide."""
    from mppp.calib.tied import merge_bins
    frames, counts = {}, {}
    k = 0
    for b, n in ((-14, 2), (-13, 1), (-12, 3), (-11, 2), (-10, 9), (-9, 1), (-2, 10), (0, 3)):
        for _ in range(n):
            frames[k] = {"scape": "a", "early": False, "bin": b}; counts[k] = {"NL": 1, "NR": 1}; k += 1
    frames[k] = {"scape": "b", "early": False, "bin": -14}; counts[k] = {"NL": 1}
    merge_bins(frames, counts, 8, 10.0, 2.0)
    lab = [f["bin"] for f in frames.values() if f["scape"] == "a"]
    assert set(lab) == {-14, -10, -2}                    # -14..-11 (8 images), -10..-9, then -2 and 0 together
    assert lab.count(-14) == 8 and lab.count(-10) == 10 and lab.count(-2) == 13
    assert frames[k]["bin"] == -14                       # other blocks are untouched


def test_joint_refines_the_rig_roll_the_blocks_hold(tmp_path):
    """0.94.4: the blocks hold the Navcam rig roll and their settings came through merge_scapes, so the joint held
    every unit's roll and never refitted the global roll (20-24 Oct: identical to 7 digits, sd 6e-5 mdeg)."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    proj.settings = dict(proj.settings, navcam_rig_roll="hold", navcam_rig_yaw="hold")
    st["D"] = st["D"] + np.array([0.0, 0.0, 3.0])                     # a roll rate the data do not have
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    assert proj2.settings["navcam_rig_roll"] == "refine" and proj2.settings["navcam_rig_yaw"] == "refine"
    ba, g, stats = T.adjust_tied(rec2, proj2, idx, st, units, iterations=30, covariance=True, passes=2, verbose=False)
    res = T.globals_result(st, g, ba, units)
    roll, sd = res["rig"]["rate_mdeg_per_1000sol"]["roll"], res["rig"]["sd_rate"]["roll"]
    assert abs(roll - float(st["D"][2])) > 1.0                         # moved off the start
    assert 0.01 < sd < 5.0                                            # a real sd, not the 6e-5 of a held term
    assert res["pass_costs"] and len(res["pass_costs"]) >= 2


def test_early_offsets_kept_only_where_significant():
    from mppp.calib import tied as T
    em = lambda off, sd: {"offsets_px": dict(zip(T.EARLY_TERMS, off)), "sd_px": dict(zip(T.EARLY_TERMS, sd))}  # noqa
    res = {"NL": {"early_mission": em([-0.015, -0.009, 0.207, 0.137], [0.04, 0.05, 0.05, 0.04])},
           "NR": {"early_mission": em([0.029, 0.022, 0.293, 0.160], [0.04, 0.05, 0.05, 0.04])}}
    held, judged = T.early_terms_to_hold(res)
    assert judged and set(held) == {"E_NL_fx", "E_NL_fy", "E_NR_fx", "E_NR_fy"}
    res["sd_source"] = {"NL": "propagated from the bin cameras' covariances"}
    assert T.early_terms_to_hold(res) == ([], False)                  # not judged on the fallback sd


def test_covariance_check_needs_the_joints_own_sd(tmp_path):
    from mppp.calib import tied as T
    res = {"rig": {"sd_sol0_mdeg": {"pitch": float("inf"), "yaw": float("nan"), "roll": 0.1},
                   "sd_rate": {"pitch": float("inf"), "yaw": float("inf"), "roll": 0.08}, "sd_pitch_early": 0.4},
           "held": {"Q0": [1], "H": [1, 2]}}
    sd = T.rig_sd_free(res)
    assert "Q0_yaw" not in sd and sd["D_yaw"] == float("inf") and sd["Q0_roll"] == 0.1


def test_keypoint_scale_follows_fx_slope_with_iso(tmp_path):
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    g = T.make_ties(st, units)[1]
    g["S_NL"][1] = 0.0                                              # a stale fy slope
    temps = {k: v + 0.7 for k, v in idx["temps"].items()}            # each image a little off its bin's median
    ks, _ = T.keypoint_terms(units, proj2, temps, g, iso_f=True)
    ks0, _ = T.keypoint_terms(units, proj2, temps, g, iso_f=False)
    nl = [int(r["image_id"]) for r in proj2.images if units["cameras"][int(r["camera_id"])]["eye"] == "NL"]
    moved = [i for i in nl if i in ks and abs(ks[i][0] - 1.0) > 0]
    assert moved and all(ks[i][1] == pytest.approx(ks[i][0], rel=1e-12) for i in moved)
    assert any(ks0[i][1] == 1.0 for i in moved)


def test_joint_points_by_stations_within_the_angle_limits(tmp_path):
    from mppp.calib import joint as J
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    ids = J._subsample_points(rec, 200, seed=1)
    ang, nst = J.point_geometry(rec, ids)
    assert len(ids) == 200 and ang.min() >= J.MIN_TRI_ANGLE_DEG and ang.max() <= J.MAX_TRI_ANGLE_DEG
    alla, alln = J.point_geometry(rec, list(rec.points3D))
    assert nst.mean() >= alln[(alla >= 0.25) & (alla <= 60)].mean() - 1e-9    # multi-station points preferred


def test_exact_tie_covariance_agrees_with_colmap_and_reports_correlations(tmp_path, monkeypatch):
    """0.94.4 (Christian, 25 Oct: the 24/25 Oct joints' global sd came out infinite): the covariance of the global
    terms with the ties eliminated exactly (mppp.calib.tiecov) agrees with COLMAP's estimator where that works (one
    small block, a hinge, four sols), does not depend on the tie sigmas, and the log says when the sd are unusable."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    for r in proj.images:                                     # four sols, two before the knot: the hinge is free
        k = int(r["sclk_key"][:10])
        r["sol"] = (100 if k % 4 < 2 else 200) if r["station"] == "S001D0000" else (900 if k % 4 < 2 else 1000)

    def run(exact, q_sigma=1e-8):
        monkeypatch.setattr(T, "EXACT_TIE_COVARIANCE", exact)
        monkeypatch.setattr(T, "EXACT_TIE_SOLVE", False)       # COLMAP's estimator needs the stiff ties' own optimum
        monkeypatch.setitem(T.TIE_SIGMA, "q", q_sigma)
        monkeypatch.setitem(T.TIE_SIGMA, "k", 1e-7)
        rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
        ba, g, _ = T.adjust_tied(rec2, proj2, idx, st, units, iterations=20, covariance=True, passes=1, verbose=False)
        return T.globals_result(st, g, ba, units), ba

    r0, _ = run(False)
    r1, ba1 = run(True)
    assert r0["covariance_method"] == "colmap" and r1["covariance_method"].startswith("exact ties")
    assert ba1["covariance"]["extra_exact"]["error"] is None and ba1["covariance"]["extra_exact"]["dims"]["tied_columns"] > 0
    for a in ("pitch", "roll"):
        assert r1["rig"]["sd_rate"][a] == pytest.approx(r0["rig"]["sd_rate"][a], rel=0.03)
        assert r1["rig"]["sd_sol0_mdeg"][a] == pytest.approx(r0["rig"]["sd_sol0_mdeg"][a], rel=0.03)
    assert r1["rig"]["sd_pitch_early"] == pytest.approx(r0["rig"]["sd_pitch_early"], rel=0.03)
    assert r1["NL"]["sd"]["fx_ppm_per_degC"] == pytest.approx(r0["NL"]["sd"]["fx_ppm_per_degC"], rel=0.03)
    assert not r1.get("sd_source")
    r2, _ = run(True, q_sigma=1e-10)                          # stiffer ties: the exact covariance does not move
    assert r2["rig"]["sd_rate"]["pitch"] == pytest.approx(r1["rig"]["sd_rate"]["pitch"], rel=1e-2)
    assert isinstance(r1["global_correlations"], list)
    assert T.covariance_line(dict(r1, residuals={})).endswith("; ok")
    bad = dict(r1, rig=dict(r1["rig"], sd_rate={"pitch": float("inf"), "yaw": 0.1, "roll": 0.1}))
    assert "NOT USABLE" in T.covariance_line(bad) and "D_pitch" in T.covariance_line(bad)


def test_exact_tie_solve_does_not_depend_on_the_tie_sigma(tmp_path, monkeypatch):
    """0.94.4 (25 Oct): with stiff tie residuals Levenberg-Marquardt stops in a flat valley (the synthetic pitch rate
    0.24 at 1e-6, 0.19 at 1e-10 at the same cost); the exact refinement (mppp.calib.tiesolve: the ties as
    constraints) reaches the same globals whatever the sigma, never raises the cost, and leaves the ties in place."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    for r in proj.images:
        k = int(r["sclk_key"][:10])
        r["sol"] = (100 if k % 4 < 2 else 200) if r["station"] == "S001D0000" else (900 if k % 4 < 2 else 1000)

    def run(q_sigma):
        monkeypatch.setitem(T.TIE_SIGMA, "q", q_sigma)
        rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
        ba, g, _ = T.adjust_tied(rec2, proj2, idx, st, units, iterations=200, covariance=True, passes=1, verbose=False,
                                 exact_solve=True)
        return T.globals_result(st, g, ba, units), g, ba

    r1, g1, ba1 = run(1e-6)
    r2, g2, ba2 = run(1e-10)
    for ba in (ba1, ba2):
        ex = ba["extra"]["exact_refine"]
        assert ex["error"] is None and 0.0 <= ex["cost_end"] <= ex["cost_start"]
    sd = r1["rig"]["sd_rate"]["pitch"]
    assert abs(g1["D"][0] - g2["D"][0]) < 0.02 * sd
    assert np.allclose(g1["S_NL"], g2["S_NL"], rtol=1e-3, atol=1e-7)
    assert r1["covariance_method"].startswith("exact ties") and r1["exact_refine"]["iterations"] >= 1


def test_law_check_deviations_and_held_residuals(tmp_path):
    """0.94.4 (Christian, 25 Oct: how good the global solutions are at predicting the cameras): the law check holds
    every unit at the law (residuals per block), frees them (deviations from the law in sd units, chi^2 / dof, trends)
    and compares any law with the same free units."""
    pytest.importorskip("pyceres")
    import copy
    from mppp.calib import lawcheck as L
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    free = L.units_free(rec, proj, idx, st, iterations=30, verbose=False)
    held = L.law_held(rec, proj, idx, st, iterations=30)
    assert free["cameras"] and free["rigs"] and free["camera_cov_dims"] == [3]
    blk = next(iter(held["blocks"]))
    assert held["blocks"][blk]["rms_px"] >= free["blocks"][blk]["rms_px"] - 1e-6   # holding never fits better
    d = L.deviations(free, st, min_images=1)
    assert {"NL_fx", "NR_fx", "rig_pitch"} <= set(d["chi2"])
    c = d["cameras"][0]
    assert c["sd_fx"] > 0 and c["z_fx"] == pytest.approx(c["d_fx"] / c["sd_fx"])
    st2 = copy.deepcopy(st)                                    # a law 100 ppm longer in fx: the deviations 100 ppm less
    st2["G"]["NL"] = np.array(st2["G"]["NL"], float)
    st2["G"]["NL"][0] *= 1 + 100e-6
    d2 = L.deviations(free, st2, min_images=1)
    a = next(r for r in d["cameras"] if r["eye"] == "NL")
    b = next(r for r in d2["cameras"] if r["eye"] == "NL" and r["key"] == a["key"])
    assert b["d_fx"] == pytest.approx(a["d_fx"] - 100.0, abs=0.1)
    res = {"laws": {"x": st["folder"]}, "held": {"x": held}, "deviations": {"x": d}, "free": {"x": free}}
    lines = L.describe(res, "x")
    assert lines[0].startswith("law x") and any("chi2/dof" in s for s in lines)
    out = L.write_validation(res, tmp_path / "lc")
    assert (out / "report.md").exists() and (out / "law_validation.json").exists()


def test_a_held_slope_keeps_the_joints_own_sd(tmp_path):
    """25 Oct (the hold study): a held thermal slope has no sd by design; the other terms keep the joint's own
    covariance (before, any held slope sent every sd to the bin-camera fallback and skipped the early-mission test)."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba, g, _ = T.adjust_tied(rec2, proj2, idx, st, units, hold=tuple(T.HOLD_DEFAULT) + ("S_NL_cx",), iterations=20,
                             covariance=True, passes=1, verbose=False)
    res = T.globals_result(st, g, ba, units)
    assert not res.get("sd_source")
    assert np.isfinite(res["NL"]["sd"]["fx_ppm_per_degC"]) and np.isnan(res["NL"]["sd"]["cx_px_per_degC"])


def test_calib_lawcheck_command(tmp_path, monkeypatch):
    """27 Oct: the 25 Oct law check as a command - ``mppp calib lawcheck --law LABEL=DIR``."""
    from mppp import cli
    from mppp.calib import lawcheck as L
    seen = {}
    monkeypatch.setattr(cli, "_navcam_blocks", lambda a: {"a": tmp_path / "a", "b": tmp_path / "b"})

    def fake_validate(blocks, laws, points=4000, **kw):
        seen.update(blocks=blocks, laws=laws, points=points)
        return {"held": {}}
    monkeypatch.setattr(L, "validate_laws", fake_validate)
    monkeypatch.setattr(L, "write_validation", lambda res, out: seen.setdefault("out", out))
    a = cli.build_parser().parse_args(["calib", "lawcheck", "--root", str(tmp_path), "--out", str(tmp_path / "o"),
                                       "--law", "in_use=", "--law", "cand=" + str(tmp_path / "c"), "--points", "500"])
    assert cli.cmd_calib(a) == 0
    assert seen["laws"] == {"in_use": None, "cand": str(tmp_path / "c")} and seen["points"] == 500
    assert set(seen["blocks"]) == {"a", "b"} and str(seen["out"]) == str(tmp_path / "o")
    a = cli.build_parser().parse_args(["calib", "lawcheck", "--root", str(tmp_path), "--out", str(tmp_path / "o")])
    seen.clear()
    assert cli.cmd_calib(a) == 0 and seen["laws"] == {"in_use": None}


def test_nl_cy_held_and_one_k1_slope(tmp_path):
    """0.95 (Christian, 27 Oct): NL cy held (level and early offset; the shared per-block part is head pitch) and NR's
    k1 thermal slope tied to NL's (one design); ``tie_k1=False`` frees NR's again."""
    pytest.importorskip("pyceres")
    proj, rec, idx, st, T = _tied_setup(tmp_path)
    assert "G_NL_cy" in T.HOLD_DEFAULT and "E_NL_cy" in T.HOLD_DEFAULT and T.TIE_K1_SLOPE
    rec2, proj2, units = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba, g, _ = T.adjust_tied(rec2, proj2, idx, st, units, iterations=15, covariance=True, passes=1, verbose=False)
    held = (ba.get("extra") or {}).get("held") or {}
    assert T.TIED_PARAMS.index("cy") in held.get("G_NL", []) and T.SLOPE_TERMS.index("k1") in held.get("S_NR", [])
    assert g["G_NL"][T.TIED_PARAMS.index("cy")] == pytest.approx(st["G"]["NL"][T.TIED_PARAMS.index("cy")])
    assert g["S_NR"][4] == g["S_NL"][4]
    res = T.globals_result(st, g, ba, units)
    assert res["NR"]["k1_per_degC"] == res["NL"]["k1_per_degC"] and res.get("k1_slope_tied")
    assert res["NR"]["sd"]["k1_per_degC"] == res["NL"]["sd"]["k1_per_degC"]
    rec3, proj3, units3 = T.tied_reconstruction(rec, proj, idx, st, bin_deg=10.0)
    ba3, g3, _ = T.adjust_tied(rec3, proj3, idx, st, units3, iterations=10, covariance=False, passes=1, verbose=False,
                               tie_k1=False)
    assert 4 not in ((ba3.get("extra") or {}).get("held") or {}).get("S_NR", [])
