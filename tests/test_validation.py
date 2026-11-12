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
    assert s["wildcat_ridge"]["zcam_joint_exclude"] == [63] and s["skinner_ridge"]["sols"] == [[460, 499]]
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
