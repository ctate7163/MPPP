"""
mppp_studies.calibration.plots - the calibration figures (0.96.0: moved out of ``mppp.calib.constrain``, which keeps
the constraint of the promoted models to an image or block): the rig trends, the Hazcam EFL against temperature,
the intrinsics against temperature, the Mastcam-Z EFL against focus and the Z rig trends.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from mppp.calib.constrain import (EYE_COLORS, FIG_DPI, IFOV_MDEG, NAVCAM_T0_DEGC, RIG_TREND_YLIMS, STEREO_CX_YLIM,  # noqa: F401
                                  _out_of_axis_note, _plt, drift_offset_mdeg, drift_offset_rel, panel_letters, rig_at,
                                  thermal_scale, thermal_terms, zcam_lines_at)

PathLike = Union[str, Path]


def hinge_fit(sol: Sequence[float], y: Sequence[float], knot: float, extra: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """0.97.2 (Christian, 12 Nov 2026: "a knee at 250 sols for yaw too and possibly roll"): least squares
    ``y = a + late (sol) + (early - late) min(sol, knot) [+ c extra]`` - the rig law's form with sol0 = 0, so ``a`` is
    the value at sol 0, ``late`` the rate after the knot and ``early`` the rate before it.  {"a", "late", "se_late",
    "early", "se_early", "hinge" (early - late), "se_hinge", "z_hinge", "c", "se_c", "sd", "n", "n_early"}; None with
    fewer than two points on either side of the knot or too few points for the terms."""
    x, y = np.asarray(sol, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(extra) if extra is not None else True)
    x, y = x[ok], y[ok]
    cols = [np.ones_like(x), x, np.minimum(x, knot)] + ([np.asarray(extra, float)[ok]] if extra is not None else [])
    k = len(cols)
    if (x < knot).sum() < 2 or (x >= knot).sum() < 2 or len(x) <= k:
        return None
    A = np.c_[tuple(cols)]
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - A @ c
    s2 = float(r @ r / (len(y) - k))
    cov = s2 * np.linalg.pinv(A.T @ A)
    se_h = float(np.sqrt(cov[2, 2]))
    return {"a": float(c[0]), "late": float(c[1]), "se_late": float(np.sqrt(cov[1, 1])), "early": float(c[1] + c[2]),
            "se_early": float(np.sqrt(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2])), "hinge": float(c[2]), "se_hinge": se_h,
            "z_hinge": float(c[2] / se_h) if se_h > 0 else float("nan"), "c": float(c[3]) if extra is not None else 0.0,
            "se_c": float(np.sqrt(cov[3, 3])) if extra is not None else float("nan"), "sd": float(np.sqrt(s2)),
            "n": int(len(y)), "n_early": int((x < knot).sum()), "knot": float(knot)}


def plot_rig_trends(rig_residuals: Dict[str, Any], consensus: Dict[str, Any], drift: Dict[str, Any], path: PathLike,
                    cx_inuse_px_per_degC: Optional[float] = None, label: str = "", sol_ref: float = 0.0,
                    T0: float = NAVCAM_T0_DEGC, model_label: str = "drift model",
                    ylims: Optional[Dict[str, Tuple[float, float]]] = None, dpi: int = FIG_DPI,
                    cx_ylim: Optional[Tuple[float, float]] = STEREO_CX_YLIM, point_labels: bool = True,
                    knee_fit: Optional[Sequence[str]] = ("yaw", "roll"), knee_sol: float = 250.0) -> Path:
    """
    Rig trends 2 x 2 (fig_rig.py, 12.5 x 8.2 in, dpi 200): pitch (UL), roll (UR) and yaw with its temperature
    slope removed to T0 (LL) of every block rig epoch against its median sol ("sol epoch"), each relative to the joint rig at
    ``sol_ref`` (= 0), with the ``drift`` model line, dashed lines at the knot, the residual-trend statistics in the
    titles and a secondary IFOV axis; LR: the stereo pointing NR cx − NL cx against epoch temperature with the
    model slope (dashed), the slope in use for the block runs (dotted) and the slope with the yaw residual folded
    in (solid, ± se band).  ``rig_residuals`` is the study's ``rig_residuals.json`` (rig rotation free, cameras
    held), ``consensus`` its ``consensus.json``.  ``ylims``: per angle (default :data:`RIG_TREND_YLIMS`: pitch
    -10 to +30, yaw and roll -20 to +20 mdeg), ``cx_ylim`` the stereo panel's primary axis (default
    :data:`STEREO_CX_YLIM`, -20 to -16 px; None = from the data); points outside are counted on the panel.
    Panels lettered A-D; ``point_labels=False`` leaves out the block name at every point.
    ``knee_fit`` (0.97.2): the angles whose drift model has no knee get a hinge fitted to the plotted epochs
    (:func:`hinge_fit`, knot at the model's or ``knee_sol``), drawn dashed with its rates in the title - a fit of this
    figure only, not the law the block runs use.  The fits are returned in ``plot_rig_trends.last_knee_fits``.
    """
    plt = _plt()
    rr, cons = rig_residuals, consensus
    ylims = dict(RIG_TREND_YLIMS if ylims is None else ylims)
    B = rr["blocks"]; names = [b["block"] for b in B]
    sol = np.array([b["sol_median"] for b in B], float); T = np.array([b["T_median_degC"] for b in B], float)
    ppl = cons["thermal"]["pp_slopes"]
    CX_T = ppl["NR"][0] - ppl["NL"][0]
    CX0 = cons["NR"]["cx"] - cons["NL"]["cx"]

    def annotate(ax, x, y, nm):
        if not point_labels:
            return
        for xx, yy, n in zip(x, y, nm):
            ax.annotate(n.replace(" [sols ", " [")[:22], (xx, yy), fontsize=5.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ss = np.linspace(0, 2050, 400)
    panels = {"pitch": axs[0][0], "roll": axs[0][1], "yaw": axs[1][0]}     # roll upper right (Christian, 22 Oct)
    yaw_T = rr["fits"]["yaw"]["vs_T"]
    knee_fits: Dict[str, Dict[str, float]] = {}
    plot_rig_trends.last_knee_fits = knee_fits
    for ang, ax in panels.items():
        tot = np.array([b[f"{ang}_mdeg"] for b in B]) + drift_offset_rel(drift, ang, sol, sol_ref)
        if ang == "yaw":
            tot = tot - yaw_T["slope"] * (T - T0)
            lab = f"block rig epoch, {yaw_T['slope']:+.3f} ± {yaw_T['se']:.3f} mdeg/°C removed (to {T0:.0f} °C)"
        else:
            lab = "block rig epoch (rig rotation free, cameras held)"
        ax.plot(sol, tot, "o", color="C0", label=lab)
        ax.plot(ss, drift_offset_rel(drift, ang, ss, sol_ref), "-", color="C3", label=model_label)
        hf = None
        if knee_fit and ang in knee_fit and drift.get(f"{ang}_early_mdeg_per_sol") is None:
            kn = float(drift.get("knot_sol") or knee_sol)
            hf = hinge_fit(sol, tot, kn)
            if hf:
                knee_fits[ang] = hf
                ax.plot(ss, hf["a"] + hf["late"] * ss + hf["hinge"] * np.minimum(ss, kn), "--", color="C1", lw=1.4,
                        label=f"knee fit to these epochs: {1e3 * hf['late']:+.2f} after, {1e3 * hf['early']:+.1f} ± "
                              f"{1e3 * hf['se_early']:.1f} before sol {kn:.0f} (knee z {hf['z_hinge']:+.1f}, sd {hf['sd']:.2f} mdeg)")
                if not drift.get("knot_sol"):
                    ax.axvline(kn, color="0.4", ls="--", lw=0.8)
        annotate(ax, sol, tot, names)
        r = tot - drift_offset_rel(drift, ang, sol, sol_ref)
        rate = 1000 * (drift.get(f"{ang}_mdeg_per_sol") or 0.0)
        f = rr["fits"][ang]["vs_sol"]
        extra = (f"; {1000 * drift[f'{ang}_early_mdeg_per_sol']:+.1f} before sol {drift['knot_sol']:.0f}"
                 if drift.get(f"{ang}_early_mdeg_per_sol") is not None else "")     # 4 Nov: any angle with a knee
        ax.set_title(f"{ang}: drift {rate:+.2f} mdeg/1000 sol{extra}; residual trend {f['slope']:+.2f} ± {f['se']:.2f} (z {f['z']:+.1f})\n"
                     f"epoch sd about the model {r.std(ddof=1):.2f} mdeg = {r.std(ddof=1) / IFOV_MDEG:.3f} px", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.8)
        if drift.get("knot_sol"):
            ax.axvline(drift["knot_sol"], color="0.4", ls="--", lw=0.8)
        ax.set_xlabel("sol epoch", fontsize=8)
        ax.set_ylabel(f"{ang} relative to the joint rig at sol {sol_ref:.0f} (mdeg)", fontsize=8)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    for ang, a in panels.items():
        if ang in ylims:
            lo, hi = ylims[ang]
            y = np.concatenate([ln.get_ydata() for ln in a.get_lines() if ln.get_marker() == "o"] or [np.array([])])
            _out_of_axis_note(a, y, lo, hi)
        a.set_xlim(0, 2050)
        a.secondary_yaxis("right", functions=(lambda m: m / IFOV_MDEG, lambda p: p * IFOV_MDEG)).set_ylabel("Navcam IFOV equivalent (px)", fontsize=8)
    ax = axs[1][1]
    yaw_px = np.array([b["yaw_mdeg"] for b in B]) / IFOV_MDEG
    cx_blk = CX0 + CX_T * (T - T0) + yaw_px
    corr = CX_T + yaw_T["slope"] / IFOV_MDEG
    tt = np.linspace(np.nanmin(T) - 2, np.nanmax(T) + 2, 2)
    ax.plot(T, cx_blk, "o", color="C3", label="epoch stereo pointing as NR cx − NL cx (model + rig-yaw residual)")
    ax.plot(tt, CX0 + CX_T * (tt - T0), "--", color="0.4", label=f"{model_label}: {CX_T:+.4f} px/°C (NR {ppl['NR'][0]:+.4f}, NL {ppl['NL'][0]:+.4f})")
    if cx_inuse_px_per_degC is not None:
        ax.plot(tt, CX0 + cx_inuse_px_per_degC * (tt - T0), ":", color="0.6", label=f"in use for these blocks: {cx_inuse_px_per_degC:+.4f} px/°C")
    ax.fill_between(tt, CX0 + (corr - yaw_T["se"] / IFOV_MDEG) * (tt - T0), CX0 + (corr + yaw_T["se"] / IFOV_MDEG) * (tt - T0), color="C3", alpha=0.12, lw=0)
    ax.plot(tt, CX0 + corr * (tt - T0), "-", color="C3", label=f"with the yaw residual folded in: {corr:+.4f} ± {yaw_T['se'] / IFOV_MDEG:.4f} px/°C")
    annotate(ax, T, cx_blk, names)
    ax.axvline(T0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("epoch median Navcam temperature (°C)", fontsize=8); ax.set_ylabel("NR cx − NL cx (px)", fontsize=8)
    ax.set_title(f"stereo cx against temperature: yaw residual vs T {yaw_T['slope']:+.2f} ± {yaw_T['se']:.2f} mdeg/°C (z {yaw_T['z']:+.1f})\n"
                 f"(scatter about the line {np.std(cx_blk - (CX0 + corr * (T - T0)), ddof=2):.3f} px)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    if cx_ylim is not None:
        _out_of_axis_note(ax, cx_blk, *cx_ylim)
    ax.secondary_yaxis("right", functions=(lambda p: (p - CX0) * IFOV_MDEG, lambda m: CX0 + m / IFOV_MDEG)).set_ylabel(f"yaw equivalent about cx at {T0:.0f} °C (mdeg)", fontsize=8)
    panel_letters(axs)
    nb = len({b.get("site") or b["block"] for b in B})
    fig.suptitle(f"Navcam rig trends{(' ' + label) if label else ''} ({nb} blocks, {len(B)} rig epochs; drift referred to sol {sol_ref:.0f}; "
                 f"right axes in Navcam IFOV, 0.33 mrad = {IFOV_MDEG} mdeg)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


HAZCAM_IFOV_MDEG = float(np.degrees(1.0 / 2142.0) * 1e3)    # one Hazcam pixel (f about 2142 px): 26.7 mdeg


def plot_hazcam_rig_trends(rig: Dict[str, Any], path: PathLike, label: str = "", dpi: int = FIG_DPI,
                           point_labels: bool = True) -> Optional[Path]:
    """0.94.5 (Christian, 27 Oct: "make similar figures for the front and rear hazcams"): a Hazcam pair's rig trends,
    1 x 3 like the Navcam's (pitch, roll, yaw): every block's refined rig rotation (``rig["per_block"]`` from
    :func:`mppp_studies.calibration.hazfit.fit_rig`) relative to the fitted rig at sol 0 against the block's sol, with the drift
    line, the scatter about it in the titles and a secondary axis in Hazcam IFOV.  None without per-block rows."""
    rows = [r for r in (rig.get("per_block") or []) if r.get("sol") is not None]
    if not rows:
        return None
    plt = _plt()
    fam = rig.get("family", "?")
    sol = np.array([r["sol"] for r in rows], float)
    d = rig.get("drift") or {}
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.6), squeeze=False)
    ss = np.linspace(0, max(2050.0, float(sol.max()) + 50), 200)
    for ax, ang in zip(axs[0], ("pitch", "roll", "yaw")):
        y = np.array([r[f"{ang}_mdeg"] for r in rows], float)
        rate = float(d.get(f"{ang}_mdeg_per_sol") or 0.0)
        ax.plot(sol, y, "o", color="C0", label="block rig (rotation free, cameras held)")
        if d:
            ax.plot(ss, rate * ss, "-", color="C3", label=f"drift {1000 * rate:+.2f} mdeg/1000 sol")
        if point_labels:
            for xx, yy, r in zip(sol, y, rows):
                ax.annotate(str(r["block"])[:22], (xx, yy), fontsize=5.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
        res = y - rate * sol
        sd = float(np.std(res, ddof=1)) if len(res) > 1 else float("nan")
        ax.set_title(f"{ang}: {len(rows)} blocks; scatter about the {'drift' if d else 'median'} {sd:.1f} mdeg "
                     f"= {sd / HAZCAM_IFOV_MDEG:.2f} px", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlim(0, ss[-1]); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("sol", fontsize=8)
        ax.set_ylabel(f"{ang} relative to the rig at sol 0 (mdeg)", fontsize=8)
        ax.secondary_yaxis("right", functions=(lambda m: m / HAZCAM_IFOV_MDEG, lambda q: q * HAZCAM_IFOV_MDEG)).set_ylabel(
            "Hazcam IFOV equivalent (px)", fontsize=8)
    panel_letters(axs)
    name = {"F": "front", "R": "rear"}.get(fam, fam)
    fig.suptitle(f"{name} Hazcam rig trends{(' ' + label) if label else ''} ({len(rows)} blocks; right axes in Hazcam "
                 f"IFOV, {HAZCAM_IFOV_MDEG:.1f} mdeg)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def plot_hazcam_efl_vs_temperature(res: Dict[str, Any], path: PathLike, label: str = "",
                                   dpi: int = FIG_DPI) -> Optional[Path]:
    """0.95.2 (Christian, 30 Oct: "make_figures for the fhaz and rhaz"): the Hazcam eyes' focal length against
    temperature, 2 x 2 (FL FR / RL RR) on one y range: every block bin's focal scale (the bins of
    :func:`mppp_studies.calibration.hazjoint.bin_rows`, ``eyes[eye]["rows"]``: fx relative to the eye's fx at T0, ppm, with its
    sd; open before the early-mission sol) and the fitted law (its slope through T0; "own" for FL and FR, the
    front mean for RL and RR); efl in mm on the right axis.  ``res``: ``hazcam_fit.json``.  None without rows."""
    eyes = [e for e in ("FL", "FR", "RL", "RR") if (res.get("eyes") or {}).get(e) and not res["eyes"][e].get("skipped")
            and any(r.get("T") is not None and r.get("fx") for r in (res["eyes"][e].get("rows") or []))]
    if not eyes:
        return None
    plt = _plt()
    T0 = float(res.get("T0_degC", -20.0))
    early_sol = float(res.get("early_mission_sol") or 0.0)
    fig, axs = plt.subplots(2, 2, figsize=(13.0, 9.0), squeeze=False, sharex=True, sharey=True)
    ecol = {"FL": "#1f5fbf", "FR": "#d4572a", "RL": "#1f5fbf", "RR": "#d4572a"}
    Ts = [r["T"] for e in eyes for r in res["eyes"][e]["rows"] if r.get("T") is not None]
    T = np.linspace(min(Ts + [T0]) - 3, max(Ts + [T0]) + 3, 100)
    for k, eye in enumerate(("FL", "FR", "RL", "RR")):
        ax = axs[k // 2][k % 2]
        if eye not in eyes:
            ax.set_title(f"{eye}: not fitted", fontsize=9)
            ax.axis("off")
            continue
        e = res["eyes"][eye]
        f0 = float(e["params"][0])
        slope = float(e.get("ppm_per_degC") or 0.0)
        sd = e.get("sd_ppm_per_degC")
        rows = [r for r in e["rows"] if r.get("T") is not None and r.get("fx")]
        y = np.array([1e6 * (float(r["fx"]) / f0 - 1) for r in rows])
        x = np.array([float(r["T"]) for r in rows])
        ys = np.array([float(r.get("sd_ppm") or 0.0) for r in rows])
        early = np.array([float(r.get("sol") or 1e9) < early_sol for r in rows])
        src = {"own": "own slope", "front mean": "mean of FL and FR", "FL": "FL's slope"}.get(str(e.get("slope_source")),
                                                                                            str(e.get("slope_source") or ""))
        ax.plot(T, slope * (T - T0), "-", color=ecol[eye], lw=2,
                label=f"law {slope:+.1f}" + (f" +- {sd:.1f}" if sd else "") + f" ppm/degC ({src})", zorder=3)
        for m, filled, lab in ((~early, True, "block bins"), (early, False, f"block bins before sol {early_sol:g}")):
            if m.any():
                ax.errorbar(x[m], y[m], yerr=ys[m], fmt="o", ms=5, color=ecol[eye], ecolor="0.6", elinewidth=0.8,
                            mfc=ecol[eye] if filled else "none", zorder=4, label=lab)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(T0, color="0.7", lw=0.8, ls=":")
        sc = e.get("scatter_ppm")
        ax.set_title(f"{eye}: f {f0:.2f} px at {T0:g} degC; {len(rows)} bins, {len(set(r['block'] for r in rows))} "
                     f"blocks" + (f"; scatter about the law {sc:.0f} ppm" if sc is not None and np.isfinite(sc) else ""),
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        if k % 2 == 0:
            ax.set_ylabel(f"fx relative to fx at {T0:g} degC (ppm)", fontsize=8)
        if k // 2 == 1:
            ax.set_xlabel("camera temperature (degC)", fontsize=8)
        if k % 2 == 1:
            ax.secondary_yaxis("right", functions=(lambda q, f0=f0: q * f0 * 1e-6, lambda d, f0=f0: d / (f0 * 1e-6))
                               ).set_ylabel("fx change (px)", fontsize=8)
    panel_letters(axs)
    fig.suptitle(f"Hazcam focal length against temperature{(' ' + label) if label else ''}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def plot_intrinsics_vs_temperature(consensus: Dict[str, Any], inuse: Dict[str, Dict[str, Any]], bins: Sequence[Dict[str, Any]],
                                   rig_residuals: Dict[str, Any], path: PathLike, early_sol: float = 380.0,
                                   label: str = "", T0: float = NAVCAM_T0_DEGC, model_label: str = "new model",
                                   pp_units: Optional[Sequence[Dict[str, Any]]] = None, dpi: int = FIG_DPI) -> Path:
    """
    Intrinsics against temperature 2 x 2 (fig_intr.py, 12.5 x 8.6 in, sharex): columns NL (cool) / NR (warm), rows
    focal length and principal point as the change from T0; left/right share y range and ticks.  Points: the
    per-temperature-bin cameras of the block runs (``bins`` rows {eye, block, T, n, fx, fy}, size ~ sqrt images,
    blocks before ``early_sol`` hollow), solid lines the ``consensus`` model, dotted grey the model ``inuse``
    ({"NL": model, "NR": model}, what the block runs started from); the stereo cx of the rig epochs on the NR
    principal-point panel.  Secondary axes in ppm of f and pointing mdeg; T0 dashed; grey band = bin temperatures.
    """
    plt = _plt()
    cons, rr = consensus, rig_residuals
    sols: Dict[str, List[float]] = {}
    for b in rr["blocks"]:
        sols.setdefault(b.get("site") or b["block"], []).append(b["sol_median"])
    early_blk = {k for k, v in sols.items() if min(v) < early_sol}
    ax_ppm = cons["thermal"]["axis_ppm"]; pp = cons["thermal"]["pp_slopes"]
    yawT = rr["fits"]["yaw"]["vs_T"]
    CX2 = pp["NR"][0] - pp["NL"][0]; CX_ALL = CX2 + yawT["slope"] / IFOV_MDEG; CX_ALL_SE = yawT["se"] / IFOV_MDEG
    B = rr["blocks"]; Tblk = np.array([b["T_median_degC"] for b in B]); yaw = np.array([b["yaw_mdeg"] for b in B])
    tt = np.linspace(-60, 0, 121)
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    for j, eye in enumerate(("NL", "NR")):
        col = EYE_COLORS[eye]; f2 = cons[eye]; iu = inuse[eye]; th_iu = thermal_terms(iu.get("thermal")); p_iu = iu["params"]
        rows = [r for r in bins if r["eye"] == eye]
        ax = axs[0][j]
        for k, (p, ls, mk, alpha) in enumerate((("fx", "-", "o", 0.8), ("fy", "--", "s", 0.35))):
            ppm2 = ax_ppm[eye][k]; ppm_iu = th_iu[f"{p}_ppm"]
            ax.plot(tt, f2[p] * (1 + 1e-6 * ppm2 * (tt - T0)), ls, color=col, lw=1.4,
                    label=f"{p} {model_label}: {f2[p]:.2f} px at {T0:.0f} °C, {ppm2:+.1f} ppm/°C ({f2[p] * 1e-6 * ppm2:+.4f} px/°C)")
            # 3 Nov 2026 (Christian: "Is the starting model the black dashed line for both fx and fy?"): the model in
            # use for fx (dotted) and fy (dash-dot), both grey
            ax.plot(tt, p_iu[k] * (1 + 1e-6 * ppm_iu * (tt - T0)), linestyle=":" if k == 0 else (0, (5, 2, 1, 2)),
                    color="0.4", lw=1.1, label=f"{p} in use for the block runs: {p_iu[k]:.2f} px, {ppm_iu:+.1f} ppm/°C")
            for early, mfc in ((False, None), (True, "white")):
                rs = [r for r in rows if (r["block"] in early_blk) == early]
                if not rs:
                    continue
                Tb = np.array([r["T"] for r in rs]); nb = np.array([r["n"] for r in rs])
                v = np.array([r[p] for r in rs])                           # the bin's focal length itself (px)
                ax.scatter(Tb, v, s=6 + 1.5 * np.sqrt(nb * 10), marker=mk, facecolors=mfc or col, edgecolors=col, alpha=alpha, zorder=3,
                           label=(f"{p} per temperature bin of the block runs (size ~ images)" if not early else
                                  f"{p}, blocks before sol {early_sol:.0f} (early-mission offset not removed)")
                           if p == "fx" or not early else None)
        ax.axhline(f2["fx"], color="0.6", lw=0.6)
        ax.set_title(f"{eye}: focal length", fontsize=10)
        ax.set_ylabel("focal length (full-frame px)", fontsize=8)
        ax.secondary_yaxis("right", functions=(lambda f, f0=f2["fx"]: f - f0, lambda d, f0=f2["fx"]: d + f0)).set_ylabel(
            f"Δf from {T0:.0f} °C (px; {model_label} fx)", fontsize=8)
        ax.legend(fontsize=6.5, loc="lower right")
        ax = axs[1][j]
        cx_iu = th_iu["cx_px"]
        if eye == "NL":
            ax.plot(tt, cx_iu * (tt - T0), ":", color="0.4", lw=1.2, label=f"cx in use for the block runs: {cx_iu:+.4f} px/°C")
            ax.plot(tt, pp["NL"][0] * (tt - T0), "-", color=col, lw=1.3, label=f"cx {model_label}: {pp['NL'][0]:+.4f} px/°C")
            ax.plot(tt, pp["NL"][1] * (tt - T0), "-.", color=col, lw=0.8, label=f"cy {model_label}: {pp['NL'][1]:+.4f} px/°C")
        else:
            ax.plot(tt, cx_iu * (tt - T0), ":", color="0.4", lw=1.2, label=f"cx in use for the block runs: {cx_iu:+.4f} px/°C")
            ax.plot(tt, pp["NR"][0] * (tt - T0), "--", color=col, lw=1.2, label=f"cx {model_label}: {pp['NR'][0]:+.4f} px/°C")
            ax.fill_between(tt, (CX_ALL - CX_ALL_SE) * (tt - T0), (CX_ALL + CX_ALL_SE) * (tt - T0), color=col, alpha=0.12, lw=0)
            ax.plot(tt, CX_ALL * (tt - T0), "-", color=col, lw=1.6, label=f"stereo cx (NR − NL) with the yaw residual folded in: {CX_ALL:+.4f} ± {CX_ALL_SE:.4f} px/°C")
            ax.plot(tt, pp["NR"][1] * (tt - T0), "-.", color=col, lw=0.8, label=f"cy {model_label}: {pp['NR'][1]:+.4f} px/°C")
            y = CX2 * (Tblk - T0) + yaw / IFOV_MDEG
            ax.plot(Tblk, y, "o", color=col, ms=5, mec="white", mew=0.5, zorder=3, label="rig epoch stereo pointing as NR cx − NL cx (model + yaw residual)")
        pu = [r for r in (pp_units or []) if r["eye"] == eye]
        if pu:                               # 0.91: the pointing pass - one per block and era, rigs at the trend
            sl = pp[eye]
            for comp, mk, k in (("cx", "D", 0), ("cy", "v", 1)):
                Tu = np.array([r["T"] for r in pu]); vu = np.array([r[f"d{comp}"] for r in pu]) + sl[k] * (Tu - T0)
                eu = np.array([np.nan if r.get(f"sd_{comp}") is None else r[f"sd_{comp}"] for r in pu])
                ax.errorbar(Tu, vu, yerr=np.where(np.isfinite(eu), eu, 0), fmt=mk, ms=4, color=col, mfc=col if comp == "cx" else "white",
                            ecolor=col, elinewidth=0.6, alpha=0.75, zorder=3,
                            label=f"{comp} per block (pointing pass: rigs at the trend{', 1 px prior' if eye == 'NL' else ''}; ± sd)")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"{eye}: principal point", fontsize=10)
        ax.set_ylabel(f"Δ principal point from {T0:.0f} °C (full-frame px)", fontsize=8)
        ax.set_xlabel("Navcam camera temperature (°C)", fontsize=9)
        ax.secondary_yaxis("right", functions=(lambda d: d * IFOV_MDEG, lambda m: m / IFOV_MDEG)).set_ylabel(f"pointing equivalent (mdeg; 1 px = {IFOV_MDEG} mdeg)", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right" if eye == "NR" else "upper left")
    Tb_all = np.array([r["T"] for r in bins]) if bins else np.array([T0])
    # the focal row shows each eye's own f (px): the two panels share the span and tick spacing about their own
    # f(T0); the principal-point row shares its limits and ticks outright
    f0 = {eye: cons[eye]["fx"] for eye in ("NL", "NR")}
    half = max(max(abs(a.get_ylim()[0] - f0[e]), abs(a.get_ylim()[1] - f0[e])) for a, e in zip(axs[0], ("NL", "NR")))
    step = float(np.diff(axs[0][0].get_yticks()[:2])[0]) if len(axs[0][0].get_yticks()) > 1 else 2.0
    for a, e in zip(axs[0], ("NL", "NR")):
        a.set_ylim(f0[e] - half, f0[e] + half)
        a.set_yticks(np.arange(np.ceil((f0[e] - half) / step) * step, f0[e] + half, step))
    lo = min(axs[1][0].get_ylim()[0], axs[1][1].get_ylim()[0]); hi = max(axs[1][0].get_ylim()[1], axs[1][1].get_ylim()[1])
    for a in axs[1]:
        a.set_ylim(lo, hi)
    axs[1][1].set_yticks(axs[1][0].get_yticks()); axs[1][1].set_ylim(lo, hi)
    for a in axs.flat:
        a.axvline(T0, color="0.4", ls="--", lw=0.8)
        a.axvspan(Tb_all.min(), Tb_all.max(), color="0.85", alpha=0.35, lw=0, zorder=0); a.grid(alpha=0.3); a.set_xlim(-60, 0)
    panel_letters(axs)
    nb = len(sols)
    fig.suptitle(f"Navcam intrinsic temperature dependence{(' ' + label) if label else ''} ({nb} blocks; T0 = {T0:.0f} °C; grey band: temperatures of the bins)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


ZCAM_PIXEL_MM = 0.0074            # Mastcam-Z detector pixel pitch (mm): efl = f [px] x pitch


def plot_zcam_efl_vs_focus(res: Dict[str, Any], path: PathLike, dpi: int = FIG_DPI, label: str = "") -> Optional[Path]:
    """0.95.1 (Christian, 29 Oct: "the Z34 efl versus focus curves"): a Mastcam-Z joint's focal length against focus
    count, left eye (cool) and right eye (warm) side by side on the same y range.  Top: the joint's law (high
    backlash state solid, low state dashed) and every bin's own fx from the diagnostic pass (``diagnostics.bins``:
    a dot per bin, all blocks alike (Christian, 29 Oct), open for the low state, x for a thin bin); efl in mm on
    the right axis (x 7.4 um).  Bottom: the bins about the law in ppm.  A joint written before 0.95.1 has no bins:
    the blocks' own lines (``diagnostics.blocks``) are drawn instead.  The reference focus dashed.  ``res``: the
    ``zcam_joint.json`` content.  None without eyes."""
    eyes = sorted((res.get("eyes") or {}).keys())
    if not eyes:
        return None
    plt = _plt()
    diag = res.get("diagnostics") or {}
    bins = diag.get("bins") or []
    blocks = sorted({b["block"] for b in bins} or {r["block"] for r in diag.get("blocks") or []})
    cm = plt.get_cmap("tab20")
    bcol = {b: cm(i % 20) for i, b in enumerate(blocks)}
    ecol = {e: ("#1f5fbf" if e.startswith("ZL") else "#d4572a") for e in eyes}       # left cool, right warm
    fig, axs = plt.subplots(2, len(eyes), figsize=(7.2 * len(eyes), 8.4), squeeze=False,
                            gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    focus_all = [b["focus"] for b in bins] or [0.0]
    lo_F = min(focus_all) if bins else None
    hi_F = max(focus_all) if bins else None
    tops, bots = [], []
    for j, eye in enumerate(eyes):
        e = res["eyes"][eye]
        t = e.get("terms") or {}
        ref = float(e.get("reference_focus") or 600.0)
        F0, F1 = (lo_F - 50, hi_F + 50) if bins else (ref - 500, ref + 500)
        F = np.linspace(F0, F1, 200)
        a_h, b = float(t.get("a_fx_backlash", np.nan)), float(t.get("b", 0.0))
        law_h = a_h + b * (F - ref)
        ax, axr = axs[0][j], axs[1][j]
        ax.plot(F, law_h, color=ecol[eye], lw=2.2, label="joint law, high state", zorder=3)
        a_l = t.get("a_fx_regular")
        held = set(e.get("held") or [])
        if a_l is not None and "a_fx_regular" not in held and np.isfinite(a_l):
            ratio = a_h / float(a_l)
            ax.plot(F, float(a_l) + b / ratio * (F - ref), color=ecol[eye], lw=1.6, ls="--", label="joint law, low state",
                    zorder=3)
        else:
            ratio = None
        eb = [x for x in bins if x["eye"] == eye]
        if eb:                                              # 0.95.1 (Christian, 29 Oct): every bin a dot, one style
            for st in ("backlash", "regular"):
                pts = [x for x in eb if x["state"] == st]
                if not pts:
                    continue
                Fp = np.array([x["focus"] for x in pts]); fx = np.array([x["fx"] for x in pts])
                thin = np.array([bool(x.get("thin")) for x in pts])
                law = (a_h + b * (Fp - ref)) if st == "backlash" else (
                    (float(a_l) + b / ratio * (Fp - ref)) if ratio else np.full_like(Fp, np.nan))
                face = ecol[eye] if st == "backlash" else "none"
                lab = (f"bins, {'high' if st == 'backlash' else 'low'} state ({int((~thin).sum())})")
                ax.scatter(Fp[~thin], fx[~thin], s=14, facecolors=face, edgecolors=ecol[eye], lw=0.7, alpha=0.75,
                           label=lab, zorder=4)
                if thin.any():
                    ax.scatter(Fp[thin], fx[thin], s=18, marker="x", color="0.45", lw=0.8, zorder=4,
                               label=f"thin bins ({int(thin.sum())})")
                r_ = 1e6 * (fx - law) / law
                axr.scatter(Fp[~thin], r_[~thin], s=12, facecolors=face, edgecolors=ecol[eye], lw=0.7, alpha=0.75,
                            zorder=4)
                bots += list(r_[~thin][np.isfinite(r_[~thin])])
                tops += list(fx[~thin])
        else:                                                # a joint before 0.95.1: the blocks' own lines
            for r in diag.get("blocks") or []:
                ln = (((r.get("lines") or {}).get(eye) or {}).get("backlash") or {}).get("fx")
                if not ln:
                    continue
                y = ln["a"] + ln["b"] * (F - ref)
                ax.plot(F, y, color="0.55", lw=0.9, alpha=0.8, zorder=2,
                        label=("each block's own line (no bins in this joint)" if not tops else None))
                axr.plot(F, 1e6 * (y - law_h) / law_h, color="0.55", lw=0.9, alpha=0.8)
                tops += [y.min(), y.max()]
                bots += list(1e6 * (y - law_h) / law_h)
        tops += [law_h.min(), law_h.max()]
        for a_ in (ax, axr):
            a_.axvline(ref, color="0.4", ls="--", lw=0.8)
            a_.grid(alpha=0.3)
        sd_b = (e.get("sd") or {}).get("b")
        ax.set_title(f"{eye}: fx {a_h:.2f} px at focus {ref:.0f} (high state), slope {b:.5f}"
                     + (f" +- {sd_b:.5f}" if sd_b else "") + " px/count", loc="left", fontsize=10)
        ax.set_ylabel("focal length fx (px)")
        axr.axhline(0.0, color=ecol[eye], lw=1.2)
        axr.set_ylabel("bin - law (ppm)")
        axr.set_xlabel("focus motor count")
        sec = ax.secondary_yaxis("right", functions=(lambda v: v * ZCAM_PIXEL_MM, lambda v: v / ZCAM_PIXEL_MM))
        sec.set_ylabel("efl (mm)")
    if tops:                                                  # the same y range and ticks for both eyes
        pad = 0.04 * (max(tops) - min(tops) or 1.0)
        for ax in axs[0]:
            ax.set_ylim(min(tops) - pad, max(tops) + pad)
    if bots:
        q = np.percentile(np.abs(bots), 99) if len(bots) > 5 else max(abs(x) for x in bots)
        lim = max(200.0, 1.15 * float(q))
        for ax in axs[1]:
            ax.set_ylim(-lim, lim)
    for ax in axs[0]:
        h, l_ = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l_, fontsize=8, loc="upper left", framealpha=0.85)
    panel_letters(axs)
    zoom = int(res.get("zoom", 34))
    src = "bins: the diagnostic pass (law held, fx free)" if bins else "lines: each block's own (diagnostic pass)"
    fig.suptitle(f"Mastcam-Z {zoom} mm focal length against focus{(' ' + label) if label else ''} - "
                 f"{len(res.get('blocks') or blocks)} blocks; {src}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


ZCAM_T0_DEGC = -20.0             # reference lens temperature of the Mastcam-Z rig figure (as the Navcam's T0)


def zcam_rig_trend_fit(rows: Sequence[Dict[str, Any]], T0: float = ZCAM_T0_DEGC,
                       knee_sol: Optional[float] = None) -> Dict[str, Any]:
    """Least-squares trends of the blocks' Mastcam-Z rig residuals (mdeg): pitch and roll linear in sol, yaw linear
    in sol and in the lens temperature T (as the Navcam's yaw is shown with its temperature slope removed).  Per
    angle {"a" (at sol 0, T0), "per_sol", "se_per_sol", "per_degC", "se_per_degC", "sd", "n"}."""
    out = {}
    for ang in ("pitch", "roll", "yaw"):
        pts = [(float(r["sol"]), float(r.get("T_optics_degC", T0) if r.get("T_optics_degC") is not None else T0),
                float(r[f"{ang}_mdeg"])) for r in rows if r.get("sol") is not None and r.get(f"{ang}_mdeg") is not None
               and np.isfinite(r[f"{ang}_mdeg"])]
        f = {"a": 0.0, "per_sol": 0.0, "se_per_sol": float("nan"), "per_degC": 0.0, "se_per_degC": float("nan"),
             "sd": float("nan"), "n": len(pts)}
        withT = ang == "yaw" and len(pts) >= 5 and np.ptp([p[1] for p in pts]) > 0
        k = 3 if withT else 2
        if len(pts) > k and np.ptp([p[0] for p in pts]) > 0:
            x, T, y = (np.array(v, float) for v in zip(*pts))
            A = np.c_[np.ones_like(x), x] if not withT else np.c_[np.ones_like(x), x, T - T0]
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            r_ = y - A @ c
            cov = float(r_ @ r_ / (len(y) - k)) * np.linalg.inv(A.T @ A)
            se = np.sqrt(np.diag(cov))
            f.update(a=float(c[0]), per_sol=float(c[1]), se_per_sol=float(se[1]), sd=float(np.sqrt(r_ @ r_ / (len(y) - k))))
            if withT:
                f.update(per_degC=float(c[2]), se_per_degC=float(se[2]))
        if knee_sol and len(pts) > k + 1:               # 0.97.2: the Navcam law's knee (rate before / after knee_sol)
            x, T, y = (np.array(v, float) for v in zip(*pts))
            h = hinge_fit(x, y, float(knee_sol), (T - T0) if withT else None)
            if h:
                f.update(a=h["a"], per_sol=h["late"], se_per_sol=h["se_late"], sd=h["sd"], early_per_sol=h["early"],
                         se_early_per_sol=h["se_early"], hinge_z=h["z_hinge"], knee_sol=float(knee_sol), n_early=h["n_early"])
                if withT:
                    f.update(per_degC=h["c"], se_per_degC=h["se_c"])
        out[ang] = f
    return out


def plot_zcam_rig_trends(model: Dict[str, Any], path: PathLike, dpi: int = FIG_DPI, label: str = "",
                         point_labels: bool = True, T0: float = ZCAM_T0_DEGC,
                         ylims: Optional[Dict[str, Tuple[float, float]]] = None, knee_sol: Optional[float] = 250.0,
                         navcam_drift: Any = "promoted") -> Optional[Path]:
    """Mastcam-Z rig trends in the format of the Navcam's :func:`plot_rig_trends` (Christian, 12 Nov 2026: "make the
    zcam 34 rig trend figure identical in format to the ncam trends figure"): 2 x 2, 12.5 x 8.2 in; pitch (A), roll
    (B) and yaw with its temperature slope removed to ``T0`` (C) of every block's own rig against the block's sol,
    relative to the fitted trend at sol 0, with the trend line, the same y limits as the Navcam figure and a right
    axis in Mastcam-Z pixels; D: the stereo pointing as ZR cx - ZL cx (px, relative to its value at ``T0``) against
    the block lens temperature, the sol trend removed, with the slope and its +- se band and a right axis in mdeg.
    The Mastcam-Z joint has no drift model, so the lines are least-squares fits to these points
    (:func:`zcam_rig_trend_fit`); the dashed vertical lines are the Navcam knee (``knee_sol``) and ``T0``.
    ``navcam_drift`` (Christian, 12 Nov: "add the latest promoted ncam trend law too"): the Navcam rig's drift law
    drawn in each sol panel (dashed green, relative to sol 0 like the Navcam figure's model line) - a ``drift`` dict,
    "promoted" (default: the promoted Navcam rig file's, :func:`mppp.calib.models.load_rig`), or None for none.
    ``model``: a joint's ``M2020_ZCAM<zoom>_focus_model.json`` (``rig.residuals.per_block``).  None without rig rows."""
    rig = model.get("rig") or (model.get("calibration") or {}).get("rig") or {}
    rows = [r for r in ((rig.get("residuals") or {}).get("per_block")) or [] if r.get("sol") is not None]
    if not rows:
        return None
    plt = _plt()
    cams = model.get("cameras") or {}
    f0 = [float(c["f0_px"]) for c in cams.values() if isinstance(c, dict) and c.get("f0_px")]
    zoom = next((int(k[2:5]) for k in cams if str(k)[2:5].isdigit()), 34)
    ifov = float(np.degrees(1.0 / (np.mean(f0) if f0 else 4735.0)) * 1e3)             # one pixel, mdeg
    ylims = dict(RIG_TREND_YLIMS if ylims is None else ylims)
    fit = zcam_rig_trend_fit(rows, T0, knee_sol)
    names = [str(r["block"]).replace("mars2020_", "") for r in rows]
    sol = np.array([r["sol"] for r in rows], float)
    T = np.array([r.get("T_optics_degC") if r.get("T_optics_degC") is not None else np.nan for r in rows], float)
    Tz = np.where(np.isfinite(T), T, T0)

    def annotate(ax, x, y):
        for xx, yy, n in (zip(x, y, names) if point_labels else []):
            ax.annotate(n[:22], (xx, yy), fontsize=5.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    if navcam_drift == "promoted":
        try:
            from mppp.calib.models import load_rig
            navcam_drift = (load_rig("N") or {}).get("drift")
        except Exception:                                    # noqa: BLE001 - the figure without the Navcam law
            navcam_drift = None
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ss = np.linspace(0, 2050, 400)
    panels = {"pitch": axs[0][0], "roll": axs[0][1], "yaw": axs[1][0]}
    for ang, ax in panels.items():
        f = fit[ang]
        y = np.array([r.get(f"{ang}_mdeg", np.nan) for r in rows], float) - f["a"]
        if ang == "yaw":
            y = y - f["per_degC"] * (Tz - T0)
            lab = f"block rig, {f['per_degC']:+.3f} ± {f['se_per_degC']:.3f} mdeg/°C removed (to {T0:.0f} °C)"
        else:
            lab = "block rig (rig rotation free, cameras held)"
        ax.plot(sol, y, "o", color="C0", label=lab)
        knee = f.get("early_per_sol") is not None
        line = f["per_sol"] * ss + ((f["early_per_sol"] - f["per_sol"]) * np.minimum(ss, f["knee_sol"]) if knee else 0.0)
        ax.plot(ss, line, "-", color="C3", label=(f"trend with a knee at sol {f['knee_sol']:.0f} (fit to these blocks, "
                                                  f"{f['n_early']} before it)" if knee else "linear trend (fit to these blocks)"))
        if navcam_drift:
            nr = 1000 * (navcam_drift.get(f"{ang}_mdeg_per_sol") or 0.0)
            ne = navcam_drift.get(f"{ang}_early_mdeg_per_sol")
            ax.plot(ss, drift_offset_rel(navcam_drift, ang, ss, 0.0), "--", color="#2a9d4a", lw=1.3,
                    label=f"Navcam rig law (promoted): {nr:+.2f} mdeg/1000 sol"
                          + (f", {1000 * ne:+.1f} before sol {navcam_drift.get('knot_sol', 0):.0f}" if ne is not None else ""))
        annotate(ax, sol, y)
        z = f["per_sol"] / f["se_per_sol"] if f["se_per_sol"] and np.isfinite(f["se_per_sol"]) else float("nan")
        early = (f"; {1000 * f['early_per_sol']:+.1f} ± {1000 * f['se_early_per_sol']:.1f} before sol {f['knee_sol']:.0f} "
                 f"(knee z {f['hinge_z']:+.1f})" if knee else "")
        ax.set_title(f"{ang}: trend {1000 * f['per_sol']:+.2f} ± {1000 * f['se_per_sol']:.2f} mdeg/1000 sol (z {z:+.1f}){early}\n"
                     f"block sd about the trend {f['sd']:.2f} mdeg = {f['sd'] / ifov:.3f} px", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.8)
        if knee_sol:
            ax.axvline(knee_sol, color="0.4", ls="--", lw=0.8)
        ax.set_xlabel("sol of the block", fontsize=8)
        ax.set_ylabel(f"{ang} relative to the trend at sol 0 (mdeg)", fontsize=8)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
        if ang in ylims:
            _out_of_axis_note(ax, y, *ylims[ang], what="block(s)")
        ax.set_xlim(0, 2050)
        ax.secondary_yaxis("right", functions=(lambda m: m / ifov, lambda p: p * ifov)).set_ylabel(
            "Mastcam-Z IFOV equivalent (px)", fontsize=8)
    ax = axs[1][1]
    fy = fit["yaw"]
    ok = np.isfinite(T)
    yaw_sol = fy["per_sol"] * sol + ((fy["early_per_sol"] - fy["per_sol"]) * np.minimum(sol, fy["knee_sol"])
                                     if fy.get("early_per_sol") is not None else 0.0)
    cx = (np.array([r.get("yaw_mdeg", np.nan) for r in rows], float) - fy["a"] - yaw_sol) / ifov
    sl, se = fy["per_degC"] / ifov, fy["se_per_degC"] / ifov
    if ok.any():
        tt = np.linspace(np.nanmin(T) - 2, np.nanmax(T) + 2, 2)
        ax.plot(T[ok], cx[ok], "o", color="C3", label="block stereo pointing as ZR cx − ZL cx (rig-yaw residual, sol trend removed)")
        if np.isfinite(se):
            ax.fill_between(tt, (sl - se) * (tt - T0), (sl + se) * (tt - T0), color="C3", alpha=0.12, lw=0)
        ax.plot(tt, sl * (tt - T0), "-", color="C3", label=f"yaw against temperature: {sl:+.4f} ± {se:.4f} px/°C")
        annotate(ax, T[ok], cx[ok])
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(T0, color="0.4", ls="--", lw=0.8)
    zT = fy["per_degC"] / fy["se_per_degC"] if fy["se_per_degC"] and np.isfinite(fy["se_per_degC"]) else float("nan")
    ax.set_xlabel("block median lens temperature (°C)", fontsize=8)
    ax.set_ylabel(f"ZR cx − ZL cx relative to {T0:.0f} °C (px)", fontsize=8)
    ax.set_title(f"stereo cx against temperature: yaw vs T {fy['per_degC']:+.2f} ± {fy['se_per_degC']:.2f} mdeg/°C (z {zT:+.1f})\n"
                 f"(scatter about the line {fy['sd'] / ifov:.3f} px)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    ax.secondary_yaxis("right", functions=(lambda p: p * ifov, lambda m: m / ifov)).set_ylabel(
        f"yaw equivalent about cx at {T0:.0f} °C (mdeg)", fontsize=8)
    panel_letters(axs)
    fig.suptitle(f"Mastcam-Z {zoom} mm rig trends{(' ' + label) if label else ''} ({len(rows)} blocks, one rig each; trends "
                 f"referred to sol 0; right axes in Mastcam-Z IFOV, {ifov:.1f} mdeg)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


# ------------------------------------------------------------------------------------------------ the zoom ladder
ZOOM_SWEEP_GAP_S = 60.0          # the zoom steps of one sweep are 25-40 s apart; the pointing changes take longer
ZOOM_LADDER = {"block": "seitah_overlook", "sol": 121, "sequence": "ZCAM08114"}     # five pointings, the zoom swept
#                                                                                     up (low state) and down (high)


def zoom_ladder_rows(images_csv: PathLike, block: str = ZOOM_LADDER["block"], sol: int = ZOOM_LADDER["sol"],
                     sequence: str = ZOOM_LADDER["sequence"]) -> List[Dict[str, Any]]:
    """The frames of a zoom ladder from a states study's ``images.csv`` (per-image focal fits, f / label f): one row
    per frame with its zoom, eye, clock, ratio, observations and ``state`` - "low" in a sweep that steps the zoom UP,
    "high" in one that steps it DOWN (the zoom-history rule).  The frames are grouped into sweeps by time (:data:`ZOOM_SWEEP_GAP_S`): a frame can be
    missing from the study (too few ties) and 110 mm is not in it, so neighbouring zooms alone would misread a
    sweep that turned at 110 mm."""
    import csv
    rows = []
    with open(images_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if block in r["block"] and r["sequence"] == sequence and int(float(r["sol"])) == int(sol) and r.get("ratio"):
                g = r["group"]
                rows.append({"eye": g[:2], "zoom": int(g[2:5]), "clock": float(r["clock"]), "ratio": float(r["ratio"]),
                             "observations": int(float(r.get("observations") or 0)), "focus": float(r.get("focus") or "nan")})
    for eye in ("ZL", "ZR"):
        seq = sorted((r for r in rows if r["eye"] == eye), key=lambda r: r["clock"])
        sweeps: List[List[Dict[str, Any]]] = []
        for r in seq:                                  # a sweep: frames less than ZOOM_SWEEP_GAP_S apart
            if sweeps and r["clock"] - sweeps[-1][-1]["clock"] <= ZOOM_SWEEP_GAP_S:
                sweeps[-1].append(r)
            else:
                sweeps.append([r])
        last = None
        for sw in sweeps:
            d = np.sign(sw[-1]["zoom"] - sw[0]["zoom"]) if len(sw) > 1 else (0 if last is None else -last)
            st = "low" if d >= 0 else "high"            # a lone first frame: taken as a step up
            for r in sw:
                r["state"] = st
            last = d if d else 1
    return rows


def zoom_ladder_gaps(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per eye and zoom (``"ZL034"``): the high and low levels of f / label f weighted by observations, and the gap
    ``g = (f_high - f_low) / f_high`` in percent; None where a state has no frame."""
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted({(r["eye"], r["zoom"]) for r in rows}):
        lev = {}
        for st in ("high", "low"):
            rr = [r for r in rows if (r["eye"], r["zoom"]) == key and r["state"] == st]
            w = np.array([max(1, r["observations"]) for r in rr], float)
            lev[st] = float(np.sum(w * np.array([r["ratio"] for r in rr])) / w.sum()) if rr else None
            lev["n_" + st] = len(rr)
        lev["gap_pct"] = None if None in (lev["high"], lev["low"]) else 100.0 * (lev["high"] - lev["low"]) / lev["high"]
        out[f"{key[0]}{key[1]:03d}"] = lev
    return out


def plot_zoom_ladder(rows: Sequence[Dict[str, Any]], out_png: PathLike, dpi: int = 200,
                     title: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """0.97.0 (Christian, 12 Nov 2026: "in the official figures, add one that compares sol 121 low and high zooms"):
    one panel per zoom, f relative to that eye's high-state level (%) against the minutes into the sequence; the
    left eye cool, the right warm, high-state frames as up triangles and low-state as down triangles sized by their
    tie-point observations, the low level of each eye dotted with its gap g; the panels share the y axis.  Returns
    :func:`zoom_ladder_gaps`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gaps = zoom_ladder_gaps(rows)
    zooms = sorted({r["zoom"] for r in rows})
    t0 = min(r["clock"] for r in rows)
    col = {"ZL": "#1f5fbf", "ZR": "#c8502a"}
    fig, axes = plt.subplots(1, len(zooms), figsize=(4.3 * len(zooms), 4.2), sharey=True, squeeze=False)
    for ax, z in zip(axes[0], zooms):
        ax.axhline(0.0, color="0.35", ls="--", lw=1)
        for eye in ("ZL", "ZR"):
            g = gaps.get(f"{eye}{z:03d}") or {}
            rr = sorted((r for r in rows if r["eye"] == eye and r["zoom"] == z), key=lambda r: r["clock"])
            ref = g.get("high")
            if ref is None and g.get("low") is not None:           # no high frame: placed by the other eye's gap
                other = gaps.get(f"{'ZR' if eye == 'ZL' else 'ZL'}{z:03d}") or {}
                if other.get("gap_pct") is not None:
                    ref = g["low"] / (1.0 - other["gap_pct"] / 100.0)
            if ref is None or not rr:
                continue
            t = [(r["clock"] - t0) / 60.0 for r in rr]
            y = [100.0 * (r["ratio"] / ref - 1.0) for r in rr]
            ax.plot(t, y, color=col[eye], lw=0.8, alpha=0.5)
            for r, ti, yi in zip(rr, t, y):
                ax.scatter([ti], [yi], marker="^" if r["state"] == "high" else "v", color=col[eye], zorder=3,
                           s=float(np.clip(12.0 * np.sqrt(max(1, r["observations"]) / 50.0), 18, 170)))
            if g.get("low") is not None:
                lo = 100.0 * (g["low"] / ref - 1.0)
                ax.axhline(lo, color=col[eye], ls=":", lw=1.2)
                lab = (f"{eye}: g = {g['gap_pct']:.2f} %" if g.get("gap_pct") is not None
                       else f"{eye}: no high-state frame (placed by the other eye's gap)")
                ax.text(0.02 if eye == "ZL" else 0.98, lo, lab, color=col[eye], fontsize=7.5, va="bottom" if eye == "ZR" else "top",
                        ha="left" if eye == "ZL" else "right", transform=ax.get_yaxis_transform())
        ax.set_title(f"{z} mm")
        ax.set_xlabel("minutes into the sequence")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("f relative to the eye's high-state level (%)")
    h = [plt.Line2D([], [], marker="^", ls="", color="0.3", label="high state (zoom stepped down)"),
         plt.Line2D([], [], marker="v", ls="", color="0.3", label="low state (zoom stepped up)"),
         plt.Line2D([], [], color=col["ZL"], label="ZL"), plt.Line2D([], [], color=col["ZR"], label="ZR"),
         plt.Line2D([], [], color="0.35", ls="--", label="high level"), plt.Line2D([], [], color="0.35", ls=":", label="low level (per eye)")]
    fig.legend(handles=h, fontsize=8, loc="lower center", ncol=6, frameon=False)
    fig.suptitle(title or "Mastcam-Z zoom ladder: high and low backlash states at each zoom (symbol size: tie-point observations)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return gaps


# ------------------------------------------------------------------------------------------------ deck_joint baseline
NAVCAM_LABEL_BASELINE_MM = 424.36      # |C_R - C_L| of the label CAHVORE pair
DECK_MIN_OBS = 5                       # station rigs with fewer deck observations are left out (eod_report.py)
DECK_T0, DECK_SOL0 = -20.0, 1000.0     # the deck_joint's reference temperature (C) and sol
NAVCAM_KNEE_SOL = 250.0


def deck_baseline_rows(jr: Dict[str, Any], min_deck: int = DECK_MIN_OBS) -> List[Dict[str, Any]]:
    """The station rigs of a deck_joint result (``jr_<run>.json`` of the end-of-drive run: ``nav_baselines``) as rows
    {"station", "T", "sol", "baseline_mm", "deck_obs"}: one per Navcam rig with a temperature and at least
    ``min_deck`` deck observations."""
    rows = []
    for k, v in (jr.get("nav_baselines") or {}).items():
        if v.get("T") is None or v.get("baseline_mm") is None or v.get("deck_obs", 0) < min_deck:
            continue
        sol = v.get("sol_median")
        if sol is None and v.get("sols"):
            sol = float(np.median(v["sols"]))
        if sol is None:
            continue
        rows.append({"station": str(k).split("/")[0], "T": float(v["T"]), "sol": float(sol),
                     "baseline_mm": float(v["baseline_mm"]), "deck_obs": int(v.get("deck_obs", 0))})
    return sorted(rows, key=lambda r: r["sol"])


def fit_deck_baseline(rows: Sequence[Dict[str, Any]], clip: float = 3.5) -> Optional[Dict[str, Any]]:
    """b = a + b_T (T - T0) + b_s (sol - sol0) / 1000 over the station rigs, with one pass of rejection beyond
    ``clip`` x 1.4826 MAD: {"coef": [a mm, b_T mm/C, b_s mm/1000 sol], "se", "rms_mm", "n", "rejected"}."""
    if len(rows) < 5:
        return None
    T = np.array([r["T"] for r in rows]); S = np.array([r["sol"] for r in rows]); y = np.array([r["baseline_mm"] for r in rows])
    A = np.c_[np.ones(len(y)), T - DECK_T0, (S - DECK_SOL0) / 1000.0]
    keep = np.ones(len(y), bool)
    for _ in range(2):
        c = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        r = y - A @ c
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) <= clip * max(mad, 1e-9)
        if np.array_equal(new, keep):
            break
        keep = new
    n = int(keep.sum())
    s2 = float((r[keep] ** 2).sum() / max(n - 3, 1))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A[keep].T @ A[keep])))
    return {"coef": c.tolist(), "se": se.tolist(), "rms_mm": float(np.sqrt((r[keep] ** 2).mean())), "n": n,
            "rejected": [rows[i]["station"] for i in np.flatnonzero(~keep)]}


def plot_deck_baseline(rows: Sequence[Dict[str, Any]], path: PathLike, fit: Optional[Dict[str, Any]] = None,
                       variants: Optional[Dict[str, float]] = None, label: str = "", dpi: int = FIG_DPI,
                       in_use_mm: Optional[float] = None) -> Optional[Path]:
    """0.97.0 (Christian, 12 Nov 2026: "a navcam baseline figure coming from the deck 360 alignments with the
    fiducials"): the Navcam stereo baseline B = |C_NR - C_NL| of every station rig of the deck_joint (one end-of-drive
    360 each, scale from the surveyed deck targets), A against the camera temperature T with the sol term removed,
    B against sol with the temperature term removed; the panels share the y axis.  The line is
    B = a + b_T (T + 20) + b_s (sol - 1000) / 1000 (``fit``: {"coef", "se" or "jk_se"}; fitted here when None) with its
    1-sigma band on a; the label CAHVORE baseline and the datum ``variants`` ({name: mm at -20 C}) are horizontal
    references, the sol-250 knee a dashed vertical line.  The right axis is ppm from a.  None without rows and fit."""
    rows = list(rows or [])
    own = fit_deck_baseline(rows) if rows else None
    fit = fit or own
    if fit is None:
        return None
    plt = _plt()
    a, bT, bs = (float(v) for v in fit["coef"][:3])
    se = fit.get("jk_se") or fit.get("se") or [0, 0, 0]
    rej = set(fit.get("rejected") or [])
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True, squeeze=False)
    T = np.array([r["T"] for r in rows]); S = np.array([r["sol"] for r in rows]); y = np.array([r["baseline_mm"] for r in rows])
    out_ = np.array([r["station"] in rej for r in rows], bool)
    panels = ((T, y - bs * (S - DECK_SOL0) / 1000.0, "Navcam camera temperature T (°C)", (-50.0, 0.0), DECK_T0, bT, 1.0,
               f"b_T = {1e3 * bT:+.1f} ± {1e3 * se[1]:.1f} µm/°C = {1e6 * bT / a:+.1f} ± {1e6 * se[1] / a:.1f} ppm/°C"),
              (S, y - bT * (T - DECK_T0), "sol", (0.0, 2100.0), DECK_SOL0, bs, 1000.0,
               f"b_s = {1e3 * bs:+.0f} ± {1e3 * se[2]:.0f} µm / 1000 sol = {1e6 * bs / a:+.0f} ± {1e6 * se[2] / a:.0f} ppm"))
    for ax, (x, yy, xl, xr, x0, slope, per, txt) in zip(axs[0], panels):
        if len(x):
            xr = (min(xr[0], float(x.min()) - 2), max(xr[1], float(x.max()) + 2))
            ax.plot(x[~out_], yy[~out_], "o", color="#1f5fbf", ms=5.5, mec="white", mew=0.5,
                    label=f"station rig ({int((~out_).sum())}), other term removed")
            if out_.any():
                ax.plot(x[out_], yy[out_], "x", color="0.45", ms=6, label=f"left out of the fit ({int(out_.sum())})")
        xx = np.linspace(xr[0], xr[1], 50)
        ax.plot(xx, a + slope * (xx - x0) / per, "-", color="k", lw=1.2, label=f"fit: {txt}")
        ax.fill_between(xx, a + slope * (xx - x0) / per - se[0], a + slope * (xx - x0) / per + se[0], color="k", alpha=0.12, lw=0)
        ax.axhline(NAVCAM_LABEL_BASELINE_MM, color="#c8502a", lw=1.0, ls="--", label=f"label CAHVORE {NAVCAM_LABEL_BASELINE_MM:.2f} mm")
        if in_use_mm:
            ax.axhline(in_use_mm, color="#2a9d4a", lw=1.0, ls="-.", label=f"rig in use {in_use_mm:.2f} mm")
        for i, (nm, v) in enumerate((variants or {}).items()):
            ax.axhline(float(v), color="0.4", lw=0.8, ls=":", label=f"{nm} {float(v):.2f} mm")
        ax.axvline(x0, color="0.5", lw=0.8, ls="--")
        ax.set_xlim(*xr); ax.set_xlabel(xl, fontsize=9); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right")
    axs[0][1].axvline(NAVCAM_KNEE_SOL, color="0.5", lw=0.8, ls="--")
    axs[0][1].annotate("sol 250", (NAVCAM_KNEE_SOL, 0.02), xycoords=("data", "axes fraction"), fontsize=7, color="0.35",
                       xytext=(3, 0), textcoords="offset points")
    lo = min([a - 0.45] + ([float(y[~out_].min()) - 0.05] if (~out_).any() else []))
    hi = max([NAVCAM_LABEL_BASELINE_MM + 0.15] + ([float(y[~out_].max()) + 0.05] if (~out_).any() else []))
    axs[0][0].set_ylim(lo, hi)
    axs[0][0].set_ylabel("Navcam baseline B = |C_NR − C_NL| (mm)", fontsize=9)
    axs[0][1].secondary_yaxis("right", functions=(lambda m: 1e6 * (m / a - 1), lambda q: a * (1 + q / 1e6))).set_ylabel(
        f"ppm from a = {a:.3f} mm", fontsize=9)
    panel_letters(axs)
    fig.suptitle(f"Navcam stereo baseline from the deck fiducials{(' ' + label) if label else ''}\n"
                 f"B = a + b_T (T + 20) + b_s (sol − 1000)/1000,  a = {a:.3f} ± {se[0]:.3f} mm "
                 f"({1e6 * (a / NAVCAM_LABEL_BASELINE_MM - 1):+.0f} ppm from the label)"
                 + ("" if rows else "; no station rows found: the fit alone"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)
