"""
Tables and figure of the v0p35 Navcam calibration study (``scripts/navcam_calibration_study.py``; notebook 04
section 2d).  :func:`main` writes ``rig_study.json`` (per block: network, rigs of every mode), ``rig_tests.json``
(between- and within-block tests of yaw, pitch, roll), ``joint_<lens>.json`` and ``loo_<lens>.json`` (copies
without the working state), ``lens_comparison.json`` and ``navcam_calibration.png``, and prints the tables.
"""
import json
from pathlib import Path

import numpy as np

from . import navcal as NC



COLORS = {"rational": "#2a78d6", "fisheye_t": "#eb6834", "within": "#1baf7a"}


def load_rig(rig_dir, scapes_json=None, samples_json=None):
    """The rig studies; with the scape folders and the label samples, the left-right temperature difference is
    filled in where the project records only each image's own eye (MPPP 0.30/0.31 manifests)."""
    studies = [json.loads(f.read_text()) for f in sorted(Path(rig_dir).glob("rig_*.json"))]
    if scapes_json and samples_json:
        from mppp.sfm.project import SfmProject
        from mppp.sfm.thermal import interpolate_temperatures
        scapes = json.loads(Path(scapes_json).read_text())
        samples = json.loads(Path(samples_json).read_text())
        for s in studies:
            if s["network"].get("dT_LR_median_degC") is not None or s["scape"] not in scapes:
                continue
            p = SfmProject.load(Path(scapes[s["scape"]]))
            it = interpolate_temperatures(samples, [r["stem"] for r in p.images if str(r["instrument"]).startswith("N")])
            d = [v["NL"] - v["NR"] for v in it.values()]
            if d:
                s["network"]["dT_LR_median_degC"] = float(np.median(d))
                s["network"]["n_dT_LR"] = len(d)
                s["network"]["dT_LR_source"] = "label samples"
    return studies


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not k.startswith("cov") and k not in ("profile_states", "pred_state", "own_state", "full_state")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def rig_table(studies):
    rows = []
    for s in studies:
        n = s["network"]
        r = {"scape": s["scape"], "stations": n["stations"], "span_m": n["span_m"], "images": n["images"],
             "observations": n["observations"], "points_used": (s.get("points_used") or {}).get("kept", n["points"]),
             "T_median_degC": n["T_median_degC"], "dT_LR_degC": n["dT_LR_median_degC"], "sol": n["sol_median"],
             "cross_station_fraction": n["cross_station_fraction"]}
        for m in ("rotation", "rotation_pp", "full"):
            g = ((s.get(m) or {}).get("rigs") or [{}])[0]
            for k in ("yaw", "pitch", "roll"):
                r[f"{m}_{k}_mdeg"] = g.get(f"{k}_mdeg")
                r[f"{m}_{k}_sd_mdeg"] = g.get(f"sd_{k}_mdeg")
            r[f"{m}_disparity_inf_px"] = g.get("disparity_inf_px")
            r[f"{m}_vparallax_inf_px"] = g.get("vparallax_inf_px")
            if m == "full":
                r["full_baseline_m"] = g.get("baseline_m")
                r["full_baseline_sd_mm"] = g.get("sd_baseline_mm")
                r["full_baseline_direction_sd_mdeg"] = g.get("sd_baseline_direction_mdeg")
        bins = [g for g in ((s.get("bins") or {}).get("rigs") or []) if g.get("frames")]
        r["bins"] = [{"T_degC": g.get("T_median_degC"), "frames": g.get("frames"), "yaw_mdeg": g["yaw_mdeg"],
                      "yaw_sd_mdeg": g.get("sd_yaw_mdeg"), "pitch_mdeg": g["pitch_mdeg"], "pitch_sd_mdeg": g.get("sd_pitch_mdeg"),
                      "disparity_inf_px": g.get("disparity_inf_px")} for g in bins]
        rows.append(r)
    return rows


def main(rig_dir, joint_dir, out_dir, scapes_json=None, samples_json=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    studies = load_rig(rig_dir, scapes_json, samples_json)
    table = rig_table(studies)
    (out / "rig_study.json").write_text(json.dumps(table, indent=1))
    tests = {}
    for mode in ("rotation_pp", "rotation"):
        for ang in ("yaw", "pitch", "roll"):
            tests[f"{mode}:{ang}"] = NC.rig_tests(studies, ang, mode=mode)
    (out / "rig_tests.json").write_text(json.dumps(tests, indent=1, default=float))
    print("\n== rig (common principal points): between blocks")
    for ang in ("yaw", "pitch", "roll"):
        t = tests[f"rotation_pp:{ang}"]
        w = t.get("within") or {}
        print(f"{ang:5s} mean {t['mean_mdeg']:+7.2f}  scatter sd {t['scatter_sd_mdeg']:.2f}  median formal sd {t['median_sd_mdeg']:.2f}  "
              f"tau {t['tau_mdeg']:.2f}  Q {t['Q']:.0f}/{t['dof']}  | within: {w.get('slope_mdeg_per_degC', float('nan')):+.3f} "
              f"+- {w.get('se', float('nan')):.3f} mdeg/degC (p {w.get('p', float('nan')):.1e}, {w.get('bins')} bins / {w.get('blocks')} blocks, "
              f"tau {w.get('tau_mdeg', float('nan')):.2f})")
        for c, v in t["covariates"].items():
            print(f"      vs {v['label']:38s} slope {v['slope']:+9.4f} +- {v['se']:.4f}  p {v['p']:.3f}  rho {v['spearman']:+.2f} "
                  f"(p {v['spearman_p']:.3f})  tau {v['tau_mdeg']:.2f}")
    res = {"rig_tests": tests}
    for lens in ("rational", "fisheye_t"):
        f = Path(joint_dir) / f"joint_{lens}.json"
        if f.exists():
            j = json.loads(f.read_text())
            (out / f"joint_{lens}.json").write_text(json.dumps(strip(j), indent=1))
            res[f"joint_{lens}"] = j
        f = Path(joint_dir) / f"loo_{lens}.json"
        if f.exists():
            lo = json.loads(f.read_text())
            (out / f"loo_{lens}.json").write_text(json.dumps(strip(lo), indent=1))
            res[f"loo_{lens}"] = lo
    comp = lens_comparison(res)
    for lens in ("rational", "fisheye_t"):
        if res.get(f"loo_{lens}"):
            comp.setdefault(lens, {})["rig_prediction"] = rig_prediction(res[f"loo_{lens}"], studies)
    (out / "lens_comparison.json").write_text(json.dumps(comp, indent=1))
    figure(table, tests, res, out / "navcam_calibration.png")
    return res, comp


def lens_comparison(res):
    comp = {}
    for lens in ("rational", "fisheye_t"):
        lo = res.get(f"loo_{lens}")
        if not lo:
            continue
        rows = []
        for s, r in lo.items():
            rows.append({"scape": s, "cost_cameras_pct": r["cost_increase_cameras_pct"], "cost_all_pct": r["cost_increase_pct"],
                         "cost_no_rigT_pct": r.get("cost_increase_no_rig_thermal_pct"),
                         "held_cost": r["held_cameras"]["cost"], "own_cost": r["own"]["cost"],
                         "held_median_px": r["held_cameras"]["median_px"], "held_corner_px": r["held_cameras"]["corner_median_px"],
                         "own_median_px": r["own"]["median_px"], "own_corner_px": r["own"]["corner_median_px"],
                         "pred_rms_NL": r["camera_difference"]["NL"]["rms_px"], "pred_rms_NR": r["camera_difference"]["NR"]["rms_px"],
                         "pred_corner_NL": r["camera_difference"]["NL"]["corner_rms_px"],
                         "pred_corner_NR": r["camera_difference"]["NR"]["corner_rms_px"],
                         "dfx_NL": r["camera_difference"]["NL"]["dfx_px"], "dfx_NR": r["camera_difference"]["NR"]["dfx_px"],
                         "disparity_inf_diff_px": r["disparity_inf_diff_px"], "vparallax_inf_diff_px": r["vparallax_inf_diff_px"]})
        comp[lens] = {"rows": rows,
                      "summary": {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k != "scape"}}
        comp[lens]["summary"].update({f"median_{k}": float(np.median([r[k] for r in rows if r[k] is not None]))
                                      for k in ("pred_rms_NL", "pred_rms_NR", "cost_cameras_pct")})
        comp[lens]["summary"]["rms_dfx_px"] = float(np.sqrt(np.mean([r["dfx_NL"] ** 2 + r["dfx_NR"] ** 2 for r in rows]) / 2))
        comp[lens]["summary"]["rms_disparity_inf_px"] = float(np.sqrt(np.mean([r["disparity_inf_diff_px"] ** 2 for r in rows])))
    if "rational" in comp and "fisheye_t" in comp:
        a = {r["scape"]: r for r in comp["rational"]["rows"]}
        b = {r["scape"]: r for r in comp["fisheye_t"]["rows"]}
        common = sorted(set(a) & set(b))
        d_cost = [100 * (b[s]["held_cost"] - a[s]["held_cost"]) / a[s]["held_cost"] for s in common]
        d_pred = [(b[s]["pred_rms_NL"] + b[s]["pred_rms_NR"] - a[s]["pred_rms_NL"] - a[s]["pred_rms_NR"]) / 2 for s in common]
        from scipy import stats
        comp["paired"] = {"blocks": len(common),
                          "held_out_cost_fisheye_minus_rational_pct": {"mean": float(np.mean(d_cost)), "median": float(np.median(d_cost)),
                                                                        "fisheye_better": int(sum(x < 0 for x in d_cost)),
                                                                        "wilcoxon_p": float(stats.wilcoxon(d_cost).pvalue) if len(common) > 5 else None},
                          "prediction_rms_fisheye_minus_rational_px": {"mean": float(np.mean(d_pred)), "median": float(np.median(d_pred)),
                                                                        "fisheye_better": int(sum(x < 0 for x in d_pred)),
                                                                        "wilcoxon_p": float(stats.wilcoxon(d_pred).pvalue) if len(common) > 5 else None},
                          "per_block": [{"scape": s, "d_cost_pct": c, "d_pred_px": p} for s, c, p in zip(common, d_cost, d_pred)]}
    return comp


def figure(table, tests, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
    # (a, b) rig yaw / pitch against camera temperature: blocks (common principal points) and per-bin rigs
    for k, (ang, a_) in enumerate((("yaw", ax[0, 0]),)):
        T = [r["T_median_degC"] for r in table]
        y = [r[f"rotation_pp_{ang}_mdeg"] for r in table]
        e = [r[f"rotation_pp_{ang}_sd_mdeg"] for r in table]
        a_.errorbar(T, y, yerr=e, fmt="o", ms=6, color=COLORS["rational"], ecolor=COLORS["rational"], elinewidth=1,
                    capsize=0, label="block (one rig, common principal points)", zorder=3)
        for r in table:
            b = sorted([g for g in r["bins"] if g["T_degC"] is not None], key=lambda g: g["T_degC"])
            if len(b) >= 2:
                # the per-bin rigs hold the block's own principal points: shift them onto the block's common-pp value
                off = r[f"rotation_pp_{ang}_mdeg"] - np.average([g[f"{ang}_mdeg"] for g in b], weights=[g["frames"] for g in b])
                a_.plot([g["T_degC"] for g in b], [g[f"{ang}_mdeg"] + off for g in b], "-", color=COLORS["within"], lw=1.2,
                        alpha=0.9, zorder=2)
        w = tests[f"rotation_pp:{ang}"].get("within") or {}
        a_.plot([], [], "-", color=COLORS["within"], label=f"per-bin rigs of a block (slope {w.get('slope_mdeg_per_degC', float('nan')):+.2f} "
                                                              f"± {w.get('se', float('nan')):.2f} mdeg/°C)")
        a_.set_xlabel("median camera temperature (°C)")
        a_.set_ylabel(f"rig {ang} relative to the label rig (mdeg)")
        a_.set_title(f"{'a' if k == 0 else 'b'}  Right-camera {ang} in the rig", loc="left", fontsize=10)
        a_.legend(frameon=False, fontsize=8, loc="best")
    # (b) rig pitch and roll against sol (the slow drift)
    a_ = ax[0, 1]
    sol = np.array([r["sol"] for r in table])
    for ang, col in (("pitch", COLORS["rational"]), ("roll", COLORS["fisheye_t"])):
        y = np.array([r[f"rotation_pp_{ang}_mdeg"] for r in table])
        e = np.array([r[f"rotation_pp_{ang}_sd_mdeg"] for r in table])
        a_.errorbar(sol, y - np.average(y, weights=1 / e ** 2), yerr=e, fmt="o", ms=6, color=col, elinewidth=1,
                    capsize=0, label=f"{ang} (minus its mean)", zorder=3)
        c = np.polyfit(sol, y - np.average(y, weights=1 / e ** 2), 1)
        xs = np.array([sol.min(), sol.max()])
        a_.plot(xs, np.polyval(c, xs), "-", color=col, lw=1.2, label=f"{ang}: {1e3 * c[0]:+.1f} mdeg per 1000 sols")
    a_.set_xlabel("median sol of the block")
    a_.set_ylabel("rig angle relative to its mean (mdeg)")
    a_.set_title("b  Right-camera pitch and roll in the rig over the mission", loc="left", fontsize=10)
    a_.legend(frameon=False, fontsize=8, loc="best")
    # (c) LOO camera prediction error per block
    a_ = ax[1, 0]
    lo = {lens: res.get(f"loo_{lens}") for lens in ("rational", "fisheye_t")}
    names = sorted(set().union(*[set(v) for v in lo.values() if v]), key=lambda n: [r["scape"] for r in table].index(n)
                   if n in [r["scape"] for r in table] else 99) if any(lo.values()) else []
    x = np.arange(len(names))
    for i, lens in enumerate(("rational", "fisheye_t")):
        if not lo[lens]:
            continue
        v = [np.mean([lo[lens][n]["camera_difference"][g]["rms_px"] for g in ("NL", "NR")]) if n in lo[lens] else np.nan for n in names]
        a_.bar(x + (i - 0.5) * 0.38, v, width=0.36, color=COLORS[lens], label={"rational": "rational (k1–k4, p1, p2)",
                                                                              "fisheye_t": "fisheye + tangential"}[lens])
    a_.set_xticks(x)
    a_.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    a_.set_ylabel("held-out camera vs its own calibration, rms (px)")
    a_.set_title("c  Leave one block out: camera prediction error", loc="left", fontsize=10)
    a_.legend(frameon=False, fontsize=8)
    # (d) LOO held-out cost increase: cameras only, cameras + rig, cameras + rig without rig(T)
    a_ = ax[1, 1]
    if lo["rational"]:
        L = lo["rational"]
        v1 = [L[n]["cost_increase_cameras_pct"] if n in L else np.nan for n in names]
        v2 = [L[n]["cost_increase_pct"] if n in L else np.nan for n in names]
        v3 = [L[n].get("cost_increase_no_rig_thermal_pct", np.nan) if n in L else np.nan for n in names]
        a_.bar(x - 0.27, v1, width=0.26, color=COLORS["rational"], label="cameras held (rig free)")
        a_.bar(x, v2, width=0.26, color=COLORS["within"], label="cameras + rig held, rig(T)")
        a_.bar(x + 0.27, v3, width=0.26, color=COLORS["fisheye_t"], label="cameras + rig held, one rig")
        a_.set_xticks(x)
        a_.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        a_.set_ylabel("cost increase over the block's own calibration (%)")
        a_.set_title("d  Leave one block out: held-out fit (rational)", loc="left", fontsize=10)
        a_.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)




def rig_prediction(loo, studies, f_px: float = 2952.0):
    """
    How well the joint rig predicts a held-out block's own rig (leave one block out), as the disparity (x) and
    vertical parallax (y) at infinity through the left principal point - the combination of rig and principal
    points that stereo sees - and the roll; before and after adding the rig's drift with sol (:func:`navcal.rig_drift`
    fitted without the held-out block).  d_inf changes by -f dyaw, v_inf by +f dpitch.
    """
    by = {s["scape"]: s for s in studies}
    rows = []
    for name, r in loo.items():
        if name not in by:
            continue
        others = [s for s in studies if s["scape"] != name]
        dr = NC.rig_drift(others)
        ds = by[name]["network"]["sol_median"] - dr["sol0"]
        k = np.radians(1e-3) * f_px
        d_inf, v_inf = r["disparity_inf_diff_px"], r["vparallax_inf_diff_px"]
        roll = r["rig_diff"]["roll_mdeg"]
        # the drift turns the predicted rig by rate x ds: d_inf of the prediction -f yaw, v_inf +f pitch
        d2 = d_inf - k * dr["yaw_mdeg_per_sol"] * ds
        v2 = v_inf + k * dr["pitch_mdeg_per_sol"] * ds
        roll2 = roll + dr["roll_mdeg_per_sol"] * ds
        rows.append({"scape": name, "sol": by[name]["network"]["sol_median"], "d_inf_px": d_inf, "v_inf_px": v_inf,
                     "roll_mdeg": roll, "d_inf_drift_px": d2, "v_inf_drift_px": v2, "roll_drift_mdeg": roll2})
    rms = lambda k: float(np.sqrt(np.mean([r[k] ** 2 for r in rows]))) if rows else float("nan")     # noqa: E731
    return {"rows": rows, "rms": {k: rms(k) for k in ("d_inf_px", "v_inf_px", "roll_mdeg", "d_inf_drift_px",
                                                        "v_inf_drift_px", "roll_drift_mdeg")}}
