"""0.99.3 (Christian, 16 Nov 2026: "Promote the latest deck joint parameters"): the freeze-3 Navcam rig candidate from a
deck joint's all_rig.json (rpp_studies.calibration.deck_rig)."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("scipy")


def _all_rig(a=(0.36, 7.8, 2.4), pitch=(19.2, 6.43), yaw=(-38.9, 4.78), roll=-6.08, mean=423.536, n=165):
    k = lambda a0, rb, ra: {"knee_sol": 250.0, "a_mdeg": a0, "rate_before": rb, "se_before": 2.0, "rate_after": ra,   # noqa: E731
                            "se_after": 0.3, "sd_mdeg": 1.0, "n": n}
    return {"source": "jr_ALL.json", "stations": n, "sol_range": [15.0, 2006.0],
            "angles": {"pitch": {"n": n, "knee250": k(a[0], *pitch), "linear": {"a_mdeg": 2.0, "rate": 7.0, "se": 0.2, "sd_mdeg": 1.1}},
                       "yaw": {"n": n, "knee250": k(a[1], *yaw), "linear": {"a_mdeg": 0.5, "rate": 2.0, "se": 0.7, "sd_mdeg": 3.9}},
                       "roll": {"n": n, "knee250": k(0.6, 4.0, -6.8), "linear": {"a_mdeg": a[2], "rate": roll, "se": 0.9, "sd_mdeg": 5.4}}},
            "baseline": {"mean_mm": mean, "se_mean_um": 13.0, "b0_mm": 423.5, "um_per_100sol": 6.0, "se_um_per_100sol": 2.0, "n": n}}


def test_candidate_from_the_deck_joint_and_its_check(tmp_path):
    from scipy.spatial.transform import Rotation
    from rpp_studies.calibration import deck_rig as D
    R0 = Rotation.from_rotvec(np.radians([-0.1003, 0.04044, -0.05646])).as_matrix()
    nav = {"ref": "NL", "sensor": "NR", "R_sensor_from_ref": R0.tolist(), "sd_mdeg": {"pitch": 0.4, "yaw": 0.6, "roll": 0.08}}
    base = {"ref": "NL", "sensor": "NR", "R_sensor_from_ref": R0.tolist(), "t_sensor_from_ref_m": [-0.4235265, -0.000262, 0.000331],
            "baseline_m": 0.4235267, "translation": "deck", "family": "N", "form": "navcam-v1", "convention": "NL reference",
            "yaw_constant": True, "drift": {"model": "linear+hinge"}, "source": "freeze-2 rig"}
    old = D.build(_all_rig(n=101), nav, base, label="ALL 101")
    new = D.build(_all_rig(), nav, base, old, label="ALL")
    a = np.array([0.36, 7.8, 2.4])
    assert np.allclose(new["R_sensor_from_ref"], Rotation.from_rotvec(np.radians(1e-3 * a)).as_matrix() @ R0, atol=1e-15)
    t = np.array(new["t_sensor_from_ref_m"])
    assert abs(np.linalg.norm(t) - 0.423536) < 1e-12 and new["baseline_m"] == pytest.approx(0.423536)
    assert np.allclose(t / np.linalg.norm(t), np.array(base["t_sensor_from_ref_m"]) / np.linalg.norm(base["t_sensor_from_ref_m"]))
    d = new["drift"]
    assert d["pitch_mdeg_per_sol"] == pytest.approx(0.00643) and d["pitch_early_mdeg_per_sol"] == pytest.approx(0.0192)
    assert d["yaw_early_mdeg_per_sol"] == pytest.approx(-0.0389) and d["roll_mdeg_per_sol"] == pytest.approx(-0.00608)
    assert d["stations"] == 165 and d["hinge_angles"] == ["pitch", "yaw"] and new["translation"] == "deck"
    assert new["previous"]["replaced_source"] == old["source"] and "165-station" in new["note"]
    assert D.check(new, old)["ok"]                                       # the same law, more stations
    moved = D.build(_all_rig(yaw=(-38.9, 9.0), mean=423.60), nav, base, old)
    c = D.check(moved, old)
    assert not c["ok"] and any("MOVED" in x for x in c["lines"]) and c["baseline_change_um"] == pytest.approx(64.0, abs=0.5)
    hand_made = {k: v for k, v in old.items() if k != "deck_law"}         # 15 Nov: the candidate made before deck_law
    c = D.check(new, hand_made)
    assert c["ok"] and any(x.startswith("rates: the replaced candidate carries no deck law") for x in c["lines"])
    assert sum(" rate " in x for x in c["lines"]) == 5
    paths = {}
    for name, obj in (("all_rig", _all_rig()), ("nav", nav), ("base", base), ("cur", old)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(obj), encoding="utf-8")
    out = tmp_path / "cand"
    args = ["--all-rig", str(paths["all_rig"]), "--navjoint", str(paths["nav"]), "--base", str(paths["base"]),
            "--current", str(paths["cur"]), "--out", str(out)]
    assert D.main(args) == 0 and (out / "M2020_N_rig_new.json").is_file() and not (out / "M2020_N_rig.json").exists()
    (out / "M2020_N_rig.json").write_text(json.dumps(old), encoding="utf-8")
    assert D.main(args + ["--promote"]) == 0
    assert json.loads((out / "M2020_N_rig.json").read_text())["drift"]["stations"] == 165
    assert len(list(out.glob("M2020_N_rig_replaced_*.json"))) == 1


def test_knee_profile_finds_the_knee_of_synthetic_rigs(tmp_path):
    """0.99.5: rpp_studies.calibration.knee - a hinge at sol 350 in the deck rigs and at 200 in the block epochs is found
    in each, the knee at 250 is worse where it is wrong, and roll stays a line."""
    from rpp_studies.calibration import knee as K
    rng = np.random.default_rng(3)
    s = np.r_[rng.uniform(15, 450, 80), rng.uniform(450, 2000, 80)]
    T = rng.uniform(-30, -5, s.size)
    hinge = lambda s, k, b, a: (a * s + (b - a) * np.minimum(s - k, 0)) / 1e3          # noqa: E731
    rows = [{"tag": f"S{i:03d}", "sol": float(x), "T": float(t), "pitch_mdeg": float(hinge(x, 350, 20, 6) + rng.normal(0, 0.5)),
             "yaw_mdeg": float(hinge(x, 300, -35, 4) + rng.normal(0, 1.0)), "roll_mdeg": float(-6e-3 * x + rng.normal(0, 1.0))}
            for i, (x, t) in enumerate(zip(s, T))]
    sb = rng.uniform(15, 2000, 100)
    blocks = {"blocks": [{"sol_median": float(x), "T_median_degC": -20.0, "pitch_mdeg": float(hinge(x, 200, 25, 5.5) + rng.normal(0, 0.5)),
                          "yaw_mdeg": float(rng.normal(0, 1.0)), "roll_mdeg": float(rng.normal(0, 1.0))} for x in sb]}
    drift = {"knot_sol": 250.0, "pitch_mdeg_per_sol": 0.0, "pitch_early_mdeg_per_sol": 0.0, "yaw_mdeg_per_sol": 0.0,
             "roll_mdeg_per_sol": 0.0}
    res = K.analyze({"rows": rows, "angles": {}}, {"residuals": blocks, "drift": drift})
    assert abs(res["pitch"]["deck"]["best"] - 350) <= 30 and res["pitch"]["deck"]["law_knee_chi2"] > 4
    assert abs(res["pitch"]["blocks"]["best"] - 200) <= 40 and abs(res["pitch"]["deck"]["rate_after"] - 6) < 0.5
    assert res["roll"]["deck"]["dbic_vs_line"] > 0                        # flat-then-linear not preferred over a line
    assert res["roll_late"]["deck"]["dbic_vs_line"] + np.log(160) > 0 and "late knee" in K.text(res)   # 0.99.7: no late knee
    assert K.text(res).count("knee 250 worse by") == 6
