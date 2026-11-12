"""0.97.0: mppp_studies.validation - the pair rotation recovered from synthetic matches, the ledger, the site rules."""
import json

import numpy as np
import pytest


def test_relative_rotation_recovers_pitch_and_roll():
    from scipy.spatial.transform import Rotation
    from mppp_studies.validation import zpairs as Z
    rng = np.random.default_rng(3)
    R0 = Rotation.from_rotvec([0.02, 0.068, -0.01]).as_matrix()
    t = np.array([-0.243, 0.001, 0.003])
    w = np.radians([0.012, 0.0, -0.020])                       # 12 mdeg pitch, -20 mdeg roll
    X = np.c_[rng.uniform(-3, 3, 800), rng.uniform(-2, 2, 800), rng.uniform(5, 60, 800)]
    xL = X[:, :2] / X[:, 2:]
    Xr = X @ (Rotation.from_rotvec(w).as_matrix() @ R0).T + t
    xR = Xr[:, :2] / Xr[:, 2:] + rng.normal(0, 0.2 / 4730, (800, 2))
    f = Z.relative_rotation(xL, xR, R0, t, 4730.0)
    assert abs(f["w_mdeg"][0] - 12.0) < 0.5 and abs(f["w_mdeg"][2] + 20.0) < 3.0 and f["inliers"] > 700
    units = [{"block": f"b{i}", "zoom": 34, "sol": s, "pitch_mdeg": 0.005 * s + e, "focus": 900.0, "T": -20.0, "f_px": 4730.0}
             for i, (s, e) in enumerate(zip(range(100, 2000, 200), rng.normal(0, 0.5, 10)))]
    tr = Z.trend(units)
    assert abs(tr["value"][1] - 5.0) < 1.0 and tr["n"] == 10


def test_ledger_table_and_scores():
    from mppp_studies.validation import ledger as L
    e = L.load()
    assert len({q["id"] for q in e}) == len(e) >= 10
    md = L.table_md(e)
    assert "## Independent validation" in md and "| claim |" in md
    assert L.z_score({"value": 8.2, "se": 2.2, "expected": 5.4}) == pytest.approx(1.2727, abs=1e-3)
    assert L.z_score({"value": 1.0, "se": None, "expected": 0.0}) is None


def test_workspace_station_warns_and_sites_of_0p97():
    from mppp.sfm.health import DEFAULT_THRESHOLDS, _status, is_workspace_station, shift_spread_checks
    from mppp.waypoints.sites import load_site_table
    assert is_workspace_station("Sol1635-1640 S079D0732") and not is_workspace_station("Sol1606 S078D4826")
    st = {"a": {"label": "Sol1635-1640 S079D0732", "shift_spread_m": 0.65}, "b": {"label": "Sol1606 S078D4826", "shift_spread_m": 0.02}}
    sp = shift_spread_checks(st)
    assert sp["workspace"][1] == 0.65 and sp["station"][1] == 0.02
    assert _status("workspace_station_shift_spread_m", 0.65, DEFAULT_THRESHOLDS) == "warn"
    assert _status("within_station_shift_spread_m", 0.65, DEFAULT_THRESHOLDS) == "fail"
    t = load_site_table()
    s = t["sites"]
    assert set(s["rockytop"]["zcam_include"]) == {"110"} and s["rose_river_falls"]["zcam"] == [34, 63, 110]
    assert "zcam_joint_exclude" not in s["wildcat_ridge"] and s["skinner_ridge"]["sols"] == [[460, 499]]
    for z in ("chal_rocks_large", "seitah_overlook"):                    # 26 and 100 mm
        assert s[z]["zcam"] == [26, 34, 48, 63, 79, 100, 110]
    assert s["seitah_overlook"]["settings"]["EXCLUDE"] == ["*ZCAM08122_034*"]
    for g in ("zcam_consensus", "zcam34_consensus", "zcam79_consensus", "zcam110_consensus"):
        assert "rockytop" not in t["groups"][g] and "wildcat_ridge" in t["groups"][g] and "skinner_ridge" in t["groups"][g]
    assert "rockytop_skinner" not in t["groups"]["zcam_consensus"]
    assert s["overlook_mountain"]["sols"] == [[1145, 1157]] and s["overlook_mountain"]["zcam"] == [48, 63, 79, 110]
    assert s["hippo_pools"]["zcam"] == [34, 48, 63, 110]
    for n in ("rose_river_falls", "knob_mountain_large", "taylorfjellet", "overlook_mountain", "hippo_pools"):
        assert n in t["groups"]["zcam_consensus"]


def test_zooms_26_and_100_and_the_zoom_ladder(tmp_path):
    """0.97.0 (Christian, 12 Nov 2026): 26 and 100 mm are zooms MPPP aligns (no joint, no single state); the official
    figure of the Seitah zoom ladder - states from the sweep direction, frames missing from the study allowed."""
    from mppp.calib import forms
    from mppp.pipeline import ZCAM_ZOOMS_ALL
    from mppp.waypoints import sites as WS
    from mppp_studies.calibration import plots as C
    assert forms.ZCAM_ZOOMS == ZCAM_ZOOMS_ALL == WS.ZCAM_ZOOMS == (26, 34, 48, 63, 79, 100, 110)
    assert forms.ZCAM_SINGLE_STATE_ZOOMS == (110,) and forms.zcam_fixed_terms("ZL100") == forms.zcam_fixed_terms("ZL079")
    assert not forms.is_single_state("ZL026") and forms.ZCAM_ZOOMS_WITH_JOINT == (34, 48, 63, 79, 110)
    lines = ["block,group,sol,sequence,clock,focus,ratio,sd,observations"]
    t = 1000
    for k, zs in enumerate(([34, 48, 63, 79], [79, 63, 48, 34], [34, 48, 63], [79, 63, 48, 34])):   # the third sweep lost its 79
        t += 90
        for z in zs:
            t += 30
            up = k % 2 == 0
            for eye in ("ZL", "ZR"):
                lines.append(f"mars2020_sol_0113_seitah_overlook_zcam,{eye}{z:03d},121,ZCAM08114,{t},1254,"
                             f"{1.0 - (0.006 if up else 0.0)},1e-5,{500 + 100 * k}")
    f = tmp_path / "images.csv"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = C.zoom_ladder_rows(f)
    assert len(rows) == 30 and {r["state"] for r in rows if r["clock"] > 1000 + 90 * 4 + 30 * 11} == {"high"}
    g = C.plot_zoom_ladder(rows, tmp_path / "ladder.png", dpi=60)
    assert (tmp_path / "ladder.png").is_file() and abs(g["ZL079"]["gap_pct"] - 0.6) < 1e-6 and g["ZR034"]["n_low"] == 2


def test_deck_frame_figure(tmp_path):
    """0.97.0: the one-frame fiducial figure draws from a result of deck_frame_models, and the resolver angles are
    read by name from the label."""
    import numpy as np
    from mppp_studies.calibration import fiducials as FD
    lab = {"RSM_ARTICULATION_STATE": {"ARTICULATION_DEVICE_ANGLE": [0.66, 0.81, 0, 0, 0, 0, 0.67, 0.82],
                                      "ARTICULATION_DEVICE_ANGLE_NAME": ["AZIMUTH FINAL-RESOLVER", "ELEVATION FINAL-RESOLVER",
                                                                         "AZIMUTH INITIAL-RESOLVER", "ELEVATION INITIAL-RESOLVER",
                                                                         "AZIMUTH REQUESTED", "ELEVATION REQUESTED",
                                                                         "AZIMUTH FINAL-ENCODER", "ELEVATION FINAL-ENCODER"]}}
    assert FD.resolver_angles(lab) == (0.66, 0.81)
    rng = np.random.default_rng(0)
    T = {k: {"measured": [100.0 + 40 * i, 80.0 + 20 * i], "label": [103.0 + 40 * i, 78.0 + 20 * i],
             "constrained": [100.5 + 40 * i, 80.4 + 20 * i], "rigid": k in FD.RIGID,
             "kind": "bar" if k in FD.BAR else "unsurveyed" if k in FD.UNSURVEYED else "rigid"}
         for i, k in enumerate(list(FD.RIGID) + list(FD.BAR) + list(FD.UNSURVEYED))}
    res = {"stem": "NRF_TEST", "eye": "R", "sol": 1642, "T": -8.6, "scale": 1.0, "az": 0.66, "el": 0.81, "targets": T,
           "rms_px": {"label": 3.6, "constrained": 0.64}, "scatter_px": {"label": 0.0, "constrained": 0.0},
           "image": rng.integers(0, 255, (384, 512, 3)).astype(np.uint8)}
    out = FD.plot_deck_frame(res, tmp_path / "deck.png", dpi=60)
    assert out.is_file() and out.stat().st_size > 10000


def test_deck_baseline_figure(tmp_path):
    """0.97.0: the Navcam baseline figure from a deck_joint work folder (station rigs + report), and from the shipped
    mast-chain file alone when there is no such folder."""
    import json
    import numpy as np
    from pathlib import Path
    import mppp_studies
    from mppp_studies.calibration.plots import deck_baseline_rows, fit_deck_baseline
    from mppp_studies.paper.paper_figures import SHIPPED_MAST_CHAIN, deck_baseline_figure
    chain = Path(mppp_studies.__file__).parent / "calibration" / "data" / SHIPPED_MAST_CHAIN
    rng = np.random.default_rng(1)
    nb = {}
    for i in range(40):
        T, sol = float(rng.uniform(-40, -8)), float(rng.uniform(15, 2000))
        b = 423.53 - 0.0066 * (T + 20) + 0.03 * (sol - 1000) / 1000 + float(rng.normal(0, 0.1))
        nb[f"S{i:03d}D0000/rig0"] = {"T": T, "sols": [sol], "baseline_mm": b, "deck_obs": 12, "t_mm": [0, 0, 0]}
    nb["S999D0000/rig0"] = {"T": -20.0, "sols": [500], "baseline_mm": 425.0, "deck_obs": 2}     # too few deck observations
    work = tmp_path / "eod"
    work.mkdir()
    (work / "jr_EOD.json").write_text(json.dumps({"nav_baselines": nb}))
    (work / "jr_EOD_st0.json").write_text(json.dumps({"nav_baselines": {}}))
    rows = deck_baseline_rows({"nav_baselines": nb})
    assert len(rows) == 40
    f = fit_deck_baseline(rows)
    assert abs(f["coef"][0] - 423.53) < 0.06 and f["n"] >= 38
    assert deck_baseline_figure(work, chain, tmp_path / "a.png", 60).stat().st_size > 10000
    assert deck_baseline_figure(tmp_path / "none", chain, tmp_path / "b.png", 60).stat().st_size > 10000
