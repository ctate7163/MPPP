"""
Mastcam-Z focus breathing: refined focal length versus focus motor count, and the focus lines of a block.

With ``SfmProject.create(zcam_focus_bin=30)`` every Mastcam-Z eye and zoom (``ZL034``, ``ZR034``) is split into
cameras by focus motor count and backlash state, and each bin's focal length is refined on its own.
:func:`focus_breathing_table` collects, per bin, the focus counts, the number of images and observations, and the
initial and refined focal lengths; :func:`fit_focus_slopes` fits f = f0 + a (focus - ref) per group;
:func:`plot_focus_breathing` draws both.  :func:`refit_focus_lines` (stage 2 of the reconstruction) puts every
focus parameter of each eye on one line in focus count per backlash state, in the form of
:data:`mppp.calib.ZCAM_LINE_FORM`, and sets the bin cameras back on their lines.  All focal lengths are in
full-frame pixels (1648 x 1200).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .. import calib
from ..calib import forms
from ..calib.fit import huber_fit, huber_line
from .project import SfmProject

PathLike = Union[str, Path]


def _observations_per_camera(rec) -> Dict[int, int]:
    n: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            cid = rec.images[el.image_id].camera_id
            n[cid] = n.get(cid, 0) + 1
    return n


def focus_breathing_table(project: SfmProject, rec) -> List[Dict[str, Any]]:
    """One row per Mastcam-Z focus-bin camera (empty if the project has none)."""
    ids = project.settings.get("database", {}).get("cameras", {})
    n_obs = _observations_per_camera(rec)
    rows = []
    for key, cam in sorted(project.cameras.items()):
        if not cam.get("group") or cam.get("focus_count_median") is None:
            continue
        cid = ids.get(key)
        p0 = np.asarray(cam["params"], float)
        p1 = np.asarray(rec.cameras[int(cid)].params, float) if cid is not None and int(cid) in rec.cameras else None
        imgs = [r for r in project.images if r["instrument"] == key]
        obs = int(n_obs.get(int(cid), 0)) if cid is not None else 0
        rows.append({"camera": key, "group": cam["group"], "focus_count_median": cam["focus_count_median"],
                     "focus_count_min": cam["focus_count_range"][0], "focus_count_max": cam["focus_count_range"][1],
                     "images": len(imgs), "observations": obs,
                     "f_initial_px": float(0.5 * (p0[0] + p0[1])),
                     "f_refined_px": float(0.5 * (p1[0] + p1[1])) if p1 is not None else None,
                     "fx_refined_px": float(p1[0]) if p1 is not None else None,
                     "fy_refined_px": float(p1[1]) if p1 is not None else None,
                     "refined": bool(p1 is not None and obs > 0),
                     "state": calib.zcam_state(key, cam),          # 0.93.5: one rule; 110 mm "single"
                     "held_params": ",".join(cam.get("fixed_params") or [])})
    return rows


def _wfit(x: np.ndarray, y: np.ndarray, w: np.ndarray, ref: float) -> Optional[Dict[str, float]]:
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return None
    X = np.c_[np.ones(ok.sum()), x[ok] - ref]
    W = w[ok] / w[ok].sum()
    A = X.T @ (X * W[:, None])
    beta = np.linalg.solve(A, X.T @ (W * y[ok]))
    res = y[ok] - X @ beta
    return {"f0_px": float(beta[0]), "slope_px_per_count": float(beta[1]),
            "slope_pct_per_1000_counts": float(100 * 1000 * beta[1] / beta[0]),
            "rms_px": float(np.sqrt(np.sum(W * res ** 2))), "n": int(ok.sum())}


def fit_focus_slopes(project: SfmProject, table: List[Dict[str, Any]], min_observations: int = 100) -> Dict[str, Any]:
    """Per group: linear fits of the refined bin focal lengths (weights = observations, bins with at
    least ``min_observations``) and of the per-image label focal lengths, about the group's median count."""
    out: Dict[str, Any] = {}
    for g in sorted({r["group"] for r in table}):
        rows = [r for r in table if r["group"] == g]
        imgs = [r for r in project.images if r.get("camera_group") == g and r.get("focus_count") is not None
                and r.get("label_f_px") is not None]
        ref = float(np.median([r["focus_count"] for r in imgs])) if imgs else float(
            np.median([r["focus_count_median"] for r in rows]))
        use = [r for r in rows if r["refined"] and r["observations"] >= min_observations
               and r.get("state", "backlash") != "regular"]         # the line is the backlash state's
        refined = _wfit(np.array([r["focus_count_median"] for r in use], float),
                        np.array([r["f_refined_px"] for r in use], float),
                        np.array([r["observations"] for r in use], float), ref) if use else None
        label = _wfit(np.array([r["focus_count"] for r in imgs], float), np.array([r["label_f_px"] for r in imgs], float),
                      np.ones(len(imgs)), ref) if imgs else None
        out[g] = {"reference_count": ref, "bins": len(rows), "bins_fitted": len(use), "refined": refined, "label": label}
    return out


def plot_focus_breathing(project: SfmProject, table: List[Dict[str, Any]], fits: Dict[str, Any],
                         out_png: PathLike, min_observations: int = 100) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    groups = sorted({r["group"] for r in table})
    fig, axes = plt.subplots(1, max(1, len(groups)), figsize=(6.2 * max(1, len(groups)), 4.8), squeeze=False)
    for ax, g in zip(axes[0], groups):
        imgs = [r for r in project.images if r.get("camera_group") == g and r.get("focus_count") is not None
                and r.get("label_f_px") is not None]
        if imgs:
            ax.plot([r["focus_count"] for r in imgs], [r["label_f_px"] for r in imgs], ".", color="0.6", ms=4,
                    label="label f, per image")
        rows = [r for r in table if r["group"] == g]
        x = np.array([r["focus_count_median"] for r in rows], float)
        ax.plot(x, [r["f_initial_px"] for r in rows], "o", mfc="none", mec="C0", ms=7, label="bin start (label median)")
        reg = [r for r in rows if r.get("state") == "regular" and r["f_refined_px"] is not None]
        rows = [r for r in rows if r not in reg]
        good = [r for r in rows if r["refined"] and r["observations"] >= min_observations]
        weak = [r for r in rows if r not in good and r["f_refined_px"] is not None]
        if reg:
            ax.plot([r["focus_count_median"] for r in reg], [r["f_refined_px"] for r in reg], "s", mfc="none",
                    mec="C2", ms=7, label="regular focus state (own camera)")
        if good:
            ax.scatter([r["focus_count_median"] for r in good], [r["f_refined_px"] for r in good],
                       s=[4 + 0.6 * np.sqrt(r["observations"]) for r in good], color="C3", zorder=3,
                       label="refined (size ~ observations)")
            for r in good:
                ax.plot([r["focus_count_min"], r["focus_count_max"]], [r["f_refined_px"]] * 2, "-", color="C3", lw=1)
        if weak:
            ax.plot([r["focus_count_median"] for r in weak], [r["f_refined_px"] for r in weak], "x", color="C3",
                    ms=5, label=f"refined, < {min_observations} observations")
        fit = fits.get(g, {})
        xs = np.linspace(np.nanmin(x) - 20, np.nanmax(x) + 20, 50) if x.size else np.array([])
        txt = []
        for kind, style, col in (("refined", "-", "C3"), ("label", "--", "0.4")):
            f = fit.get(kind)
            if f and xs.size:
                ax.plot(xs, f["f0_px"] + f["slope_px_per_count"] * (xs - fit["reference_count"]), style, color=col, lw=1)
                txt.append(f"{kind}: {f['slope_px_per_count']:+.4f} px/count "
                           f"({f['slope_pct_per_1000_counts']:+.2f} %/1000), rms {f['rms_px']:.2f} px, n={f['n']}")
        if txt:
            ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(fc="white", ec="0.8", alpha=0.9))
        ax.set_title(f"{g}: focal length vs focus ({len(rows)} bins)")
        ax.set_xlabel("focus motor count")
        ax.set_ylabel("focal length [full-frame px]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")
    if not groups:
        axes[0][0].text(0.5, 0.5, "no Mastcam-Z focus-bin cameras", ha="center", va="center")
        axes[0][0].set_axis_off()
    elif len(groups) > 1:
        from ..calib.constrain import panel_letters
        panel_letters(axes)                 # 27 Oct (Christian): "A. title", left-justified, as the maps
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def write_focus_breathing(project: SfmProject, rec, out_dir: Optional[PathLike] = None,
                          min_observations: int = 100) -> Dict[str, Any]:
    """Table (csv), fits (json) and plot (png) in ``<project>/health/`` (default).  Returns the fits and paths."""
    out = Path(out_dir) if out_dir else project.root / "health"
    out.mkdir(parents=True, exist_ok=True)
    table = focus_breathing_table(project, rec)
    fits = fit_focus_slopes(project, table, min_observations)
    if table:
        with (out / "zcam_focus_breathing.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    (out / "zcam_focus_breathing.json").write_text(json.dumps({"fits": fits, "bins": table}, indent=1),
                                                   encoding="utf-8")
    png = plot_focus_breathing(project, table, fits, out / "zcam_focus_breathing.png", min_observations)
    return {"fits": fits, "table": table, "png": str(png), "csv": str(out / "zcam_focus_breathing.csv"),
            "json": str(out / "zcam_focus_breathing.json")}


# ------------------------------------------------------------------ the focus lines of a block
def refit_focus_lines(rec, project: SfmProject, model: Optional[Dict[str, Any]] = None, min_observations: int = 200,
                      min_span: float = 300.0, huber_k: float = 1.345, outlier_sigma: float = 4.0,
                      apply: bool = True) -> Dict[str, Any]:
    """Every focus parameter of each Mastcam-Z eye and zoom on one line in focus count per backlash state
    (:func:`refit_param_lines`); ``model`` defaults to the project's focus model."""
    if model is None:
        try:
            from .project import zcam_focus_model
            f = (project.settings.get("zcam_focus_model") or {}).get("file")
            model = zcam_focus_model(model_file=f)
        except Exception:                                                  # noqa: BLE001
            model = {"cameras": {}}
    min_observations = int(project.settings.get("zcam_line_min_observations") or min_observations)
    return refit_param_lines(rec, project, model, min_observations=min_observations, min_span=min_span,
                             huber_k=huber_k, outlier_sigma=outlier_sigma, apply=apply)


# ------------------------------------------------------------------ fx, fy, cx, cy, k1, k2 lines per focus state
PARAM_INDEX = {"fx": 0, "fy": 1, "cx": 2, "cy": 3, "k1": 4, "k2": 5, "p1": 6, "p2": 7}     # FULL_OPENCV (p1, p2: zcam-v2)
LINE_PARAMS = forms.ZCAM_LINE_PARAMS                 # the parameters each eye puts on lines (ZL's principal point: the reference)
LINE_STATES = forms.ZCAM_STATES
LINE_FORM = forms.ZCAM_LINE_FORM                     # the form of each line (see mppp.calib.forms)
LINE_FORMS = forms.ZCAM_LINE_FORMS
PIN_MAX_OFFSET_PX = 300.0    # a pp0 further than this from the bins' principal points is not pinned (wrong model)


def line_form_for(project: SfmProject, group: str) -> Dict[str, str]:
    """{param: form} of a Mastcam-Z group (``zcam_line_form`` in the settings as {eye: {param: form}} for a study;
    default :data:`mppp.calib.ZCAM_LINE_FORM`), for the parameters of :func:`line_params_for`."""
    lf = project.settings.get("zcam_line_form")
    params = line_params_for(project, group)
    eye = str(group)[:2]
    if isinstance(lf, dict):
        base = lf.get(group) or lf.get(eye) or lf.get(eye[1:2]) or {}
        out = {pn: str(base.get(pn, "free")) for pn in params}
    else:
        out = forms.zcam_line_form(eye, params)
    bad = {pn: f for pn, f in out.items() if f not in LINE_FORMS}
    if bad:
        raise ValueError(f"zcam_line_form of {group}: unknown forms {bad}; use one of {LINE_FORMS}")
    return out


def line_params_for(project: SfmProject, group: str) -> Tuple[str, ...]:
    """The focus parameters of a Mastcam-Z group (``ZR034``) that stage 2 puts on lines (``zcam_line_params``: keys
    ``ZR034``, ``ZR`` or ``R``; default :data:`mppp.calib.ZCAM_LINE_PARAMS`)."""
    lp = project.settings.get("zcam_line_params") or {}
    eye = str(group)[:2]
    v = lp.get(group) or lp.get(eye) or lp.get(eye[1:2])
    return tuple(v) if v else LINE_PARAMS.get(eye, ("fx", "fy"))


def refit_param_lines(rec, project: SfmProject, model: Optional[Dict[str, Any]] = None, min_observations: int = 200,
                      min_span: float = 300.0, huber_k: float = 1.345, outlier_sigma: float = 4.0,
                      apply: bool = True, min_observations_regular: int = 50) -> Dict[str, Any]:
    """
    Every focus parameter of each Mastcam-Z eye and zoom a line in focus count, **one line per focus (backlash)
    state**: p = a + b (focus - reference) for p in the group's ``zcam_line_params`` (default fx, fy, k1, k2 on ZL;
    plus cx, cy on ZR, whose principal point the narrow field measures relative to ZL's through the rig).  Each line
    is fitted (Huber, weights = observations) to the refined focus-bin cameras of that state with at least
    ``min_observations`` (``min_observations_regular`` for the regular state); with fewer than 3 bins or a focus span
    below ``min_span`` counts only a is fitted, b being the backlash state's line (regular state) or the focus
    model's (fx, fy: its f slope; cx, cy on ZR: its principal-point slope; else 0).  A regular state without a
    usable bin follows the backlash line (fx, fy divided by the model's regular ratio).  ``apply``: every bin camera
    is then set on its state's lines in ``rec`` (the caller holds them).  Returns {group: {"reference_focus",
    "params", "lines": {state: {param: {"a", "b", "sd_a", "sd_b", "n_bins", "rms", "slope_source", "outliers"}}},
    "bins": [...], plus the f summary (``f0_px``, ``slope_px_per_count``, ...)}}.
    Each parameter's line has a *form* (:func:`line_form_for`, :data:`mppp.calib.ZCAM_LINE_FORM`): fx and fy share
    one slope per eye, k1 and k2 are constant with focus (one value per block), ZR's cx is pinned at the model's pp0
    with no slope and ZR's cy is pinned at pp0 with a fitted slope.
    """
    db = (project.settings.get("database") or {}).get("cameras") or {}
    obs = _observations_per_camera(rec)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for key, cid in db.items():
        c = project.cameras.get(key) or {}
        if not str(key).startswith("Z") or int(cid) not in rec.cameras or c.get("group") is None:
            continue
        p = np.array(rec.cameras[int(cid)].params, float)
        groups.setdefault(c["group"], []).append({
            "camera": key, "camera_id": int(cid), "focus": c.get("focus_count_median"), "p": p,
            "observations": int(obs.get(int(cid), 0)),
            # the lines of a state: a single-state (110 mm) bin is on the zoom's one line, stored as the backlash one
            "state": "regular" if calib.zcam_state(key, c) == "regular" else "backlash",
            "focus_state": calib.zcam_state(key, c)})
    out: Dict[str, Any] = {}
    for g, rows in sorted(groups.items()):
        gm = (model.get("cameras") or {}).get(g) or {}
        params = line_params_for(project, g)
        form = line_form_for(project, g)
        foc = [r["focus"] for r in rows if r["focus"] is not None]
        ref = float(gm.get("reference_focus", np.median(foc) if foc else 0.0))
        ratio = float(gm.get("regular_ratio") or ((float(gm["f0_px"]) / float(gm["label_f0_px"]))
                                                   if gm.get("label_f0_px") and gm.get("f0_px") else 1.0))
        aspect = float(gm.get("aspect", 1.0))
        f_slope = gm.get("slope_px_per_count")
        if f_slope is None:
            lab = [(r.get("focus_count"), r.get("label_f_px")) for r in project.images
                   if r.get("camera_group") == g and r.get("focus_count") is not None and r.get("label_f_px")]
            f_slope = float(np.polyfit([a for a, _ in lab], [b for _, b in lab], 1)[0]) if len(lab) >= 3 else 0.0
        pp = gm.get("pp") or {}
        model_slope = {"fx": float(f_slope) / np.sqrt(aspect), "fy": float(f_slope) * np.sqrt(aspect),
                       "cx": float(pp.get("cx_px_per_count") or 0.0) if g[:2] == "ZR" else 0.0,
                       "cy": float(pp.get("cy_px_per_count") or 0.0) if g[:2] == "ZR" else 0.0, "k1": 0.0, "k2": 0.0,
                       "p1": 0.0, "p2": 0.0}
        for _st, _ln in ((gm.get("lines") or {}).items() if isinstance(gm.get("lines"), dict) else []):
            if _st == "backlash" and isinstance(_ln, dict):          # a calibrated model's slopes
                for _pn, _e in _ln.items():
                    if isinstance(_e, dict) and _e.get("b") is not None and _pn in model_slope:
                        model_slope[_pn] = float(_e["b"])
        res: Dict[str, Any] = {"reference_focus": ref, "params": list(params), "form": dict(form), "lines": {},
                               "bins": [], "outliers": []}
        for st in LINE_STATES:
            mo = min_observations if st == "backlash" else min_observations_regular
            use = [r for r in rows if r["state"] == st and r["focus"] is not None and r["observations"] >= mo]
            lines_st: Dict[str, Any] = {}
            if not use:
                res["lines"][st] = {"skipped": f"no {st}-state bin with >= {mo} observations"}
                continue
            F = np.array([r["focus"] for r in use], float)
            w = np.array([r["observations"] for r in use], float)
            fit_slope = len(use) >= 3 and float(np.ptp(F)) >= min_span
            # the common slope of the eye's "common" parameters (fx, fy): one Huber fit of both against focus
            # with their own offsets; the slope is this state's, the regular state follows the backlash state's
            common = [pn for pn in params if form.get(pn) == "common"]
            common_fit = None
            if len(common) >= 2 and fit_slope:
                Xc = np.zeros((len(use) * len(common), len(common) + 1))
                yc = np.zeros(len(use) * len(common))
                wc = np.tile(w, len(common))
                for j, pn in enumerate(common):
                    Xc[j * len(use):(j + 1) * len(use), j] = 1.0
                    Xc[j * len(use):(j + 1) * len(use), -1] = F - ref
                    yc[j * len(use):(j + 1) * len(use)] = [r["p"][PARAM_INDEX[pn]] for r in use]
                c, se_c, _, _ = huber_fit(Xc, yc, wc, huber_k, floor=0.05)
                common_fit = {"b": float(c[-1]), "sd_b": float(se_c[-1]), "params": list(common)}
            for pn in params:
                k = PARAM_INDEX[pn]
                y = np.array([r["p"][k] for r in use], float)
                fm = form.get(pn, "free")
                bl = (res["lines"].get("backlash") or {}).get(pn) if st == "regular" else None
                slope0 = bl["b"] if (bl and "b" in bl) else model_slope.get(pn, 0.0)
                src0 = "backlash state" if (bl and "b" in bl) else ("focus model" if gm else "labels")
                fit_b, held_b, src_b = fit_slope, None, None
                if fm == "constant" or fm == "pinned":
                    fit_b, held_b, src_b = False, 0.0, "constant with focus (form)"
                elif fm == "common":
                    if st == "regular" and bl and "b" in bl:
                        fit_b, held_b, src_b = False, float(bl["b"]), "backlash state (common slope)"
                    elif common_fit is not None:
                        fit_b, held_b, src_b = False, common_fit["b"], f"common slope of {', '.join(common_fit['params'])}"
                elif fm == "pinned_slope" and not fit_slope:
                    fit_b, held_b, src_b = False, slope0, src0
                pin_a = None
                pin_note = None
                if fm in ("pinned", "pinned_slope") and pn in ("cx", "cy"):
                    pp0 = gm.get("pp0_px") if gm else None
                    if pp0 and len(pp0) == 2:
                        pin_a = float(pp0[0 if pn == "cx" else 1])
                        far = abs(float(np.median(y)) - pin_a)
                        if far > PIN_MAX_OFFSET_PX:        # a model that does not belong to these bins: not pinned
                            pin_note = (f"not pinned: the model's pp0 {pin_a:.1f} is {far:.0f} px from the bins' median "
                                        f"(> {PIN_MAX_OFFSET_PX:g}); the level is fitted")
                            pin_a = None
                            if fm == "pinned":
                                fit_b, held_b, src_b = False, 0.0, "constant with focus (form; pp0 not pinned)"
                if pin_a is not None and held_b is not None:        # a and b both set: nothing to fit
                    a, b = pin_a, float(held_b)
                    resid = y - (a + b * (F - ref))
                    sig = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 0.05)
                    sda, sdb = float("nan"), None
                elif pin_a is not None:                             # a pinned, b fitted through the pinned point
                    Xp = (F - ref)[:, None]
                    c, se, sig, resid = huber_fit(Xp, y - pin_a, w, huber_k, floor=0.05)
                    a, b, sda, sdb = pin_a, float(c[0]), float("nan"), float(se[0])
                else:
                    a, b, sig, resid, (sda, sdb) = huber_line(F, y, w, ref, fit_b, held_b if held_b is not None else slope0,
                                                              huber_k, floor=0.05 if pn in ("fx", "fy", "cx", "cy") else 1e-5)
                    if fm == "common" and common_fit is not None and st == "backlash":
                        sdb = common_fit["sd_b"]
                lim = max(outlier_sigma * sig, 3.0 if pn in ("fx", "fy") else (1.0 if pn in ("cx", "cy") else 0.0))
                ent = {"a": a, "b": b, "sd_a": sda, "sd_b": sdb, "n_bins": len(use), "form": fm,
                       "focus_range": [float(F.min()), float(F.max())],
                       "slope_source": src_b if src_b else ("fitted" if fit_b else src0),
                       "rms": float(np.sqrt(np.average(resid ** 2, weights=w))), "robust_sd": sig, "outlier_limit": lim,
                       "outliers": []}
                if pin_a is not None:
                    ent["a_source"] = "pinned at the model's pp0 (form)"
                elif pin_note:
                    ent["a_source"] = pin_note
                for rr, d in zip(use, resid):
                    row = {"camera": rr["camera"], "state": st, "param": pn, "focus": rr["focus"],
                           "value": float(rr["p"][k]), "residual": float(d), "observations": rr["observations"]}
                    res["bins"].append(row)
                    if abs(d) > lim:
                        ent["outliers"].append(row)
                        res["outliers"].append(row)
                lines_st[pn] = ent
            res["lines"][st] = lines_st
        bk = res["lines"].get("backlash") or {}
        if all(p in bk for p in ("fx", "fy")):            # the f summary of refit_focus_lines (printing, reports)
            res.update({"f0_px": float(np.sqrt(bk["fx"]["a"] * bk["fy"]["a"])),
                        "slope_px_per_count": 0.5 * (bk["fx"]["b"] + bk["fy"]["b"]),
                        "slope_source": bk["fx"]["slope_source"], "n_bins": bk["fx"]["n_bins"],
                        "rms_px": float(np.hypot(bk["fx"]["rms"], bk["fy"]["rms"]) / np.sqrt(2)),
                        "model_f0_px": gm.get("f0_px"), "model_slope_px_per_count": gm.get("slope_px_per_count")})
        else:
            res["skipped"] = "no backlash-state line (too few observations)"
        if apply:
            for rr in rows:
                fc = rr["focus"] if rr["focus"] is not None else ref
                ln = res["lines"].get(rr["state"]) or {}
                own = bool(ln) and "skipped" not in ln
                if not own:
                    ln = res["lines"].get("backlash") or {}
                if not ln or "skipped" in ln:
                    continue
                p = np.array(rec.cameras[rr["camera_id"]].params, float)
                for pn in params:
                    if pn not in ln:
                        continue
                    v = ln[pn]["a"] + ln[pn]["b"] * (fc - ref)
                    if rr["state"] == "regular" and not own and pn in ("fx", "fy"):
                        v /= ratio                           # the regular state sits at the label
                    if rr["state"] == "regular" and not own and pn in ("cx", "cy"):
                        # 0.93.5: on the backlash lines, a regular bin keeps the regular state's principal point
                        off = gm.get("regular_pp_offset_px") or [0.0, 0.0]
                        v += float(off[0 if pn == "cx" else 1])
                    if pn == "k2":                           # |k2| <= K2_LIMIT
                        from .reconstruction import K2_LIMIT
                        v = float(np.clip(v, -K2_LIMIT, K2_LIMIT))
                    p[PARAM_INDEX[pn]] = v
                rec.cameras[rr["camera_id"]].params = p
        out[g] = res
    return out


def focus_line_lines(lines: Dict[str, Any]) -> List[str]:
    """Printable summary of :func:`refit_focus_lines` (every parameter line per state)."""
    out = []
    for g, r in lines.items():
        if r.get("lines"):
            out.append(f"      {g} (about focus {r['reference_focus']:.0f}):")
            for st, ln in r["lines"].items():
                if "skipped" in ln:
                    out.append(f"        {st}: {ln['skipped']}")
                    continue
                out.append(f"        {st}: " + "; ".join(
                    f"{pn} {e['a']:.2f} {e['b']:+.5f}/count [{e['slope_source']}, {e['n_bins']} bins, rms {e['rms']:.2f}]"
                    + (f" {len(e['outliers'])} outlier bins" if e["outliers"] else "") for pn, e in ln.items()))
            continue
        if r.get("skipped"):
            out.append(f"      {g}: {r['skipped']}")
            continue
        m = r.get("model_f0_px")
        out.append(f"      {g}: f = {r['f0_px']:.1f} + {r['slope_px_per_count']:+.4f} (focus - {r['reference_focus']:.0f}) px "
                   f"[{r['slope_source']}], {r['n_bins']} bins, rms {r['rms_px']:.1f} px"
                   + (f"; model f0 {m:.1f} ({r['f0_px'] - m:+.1f})" if m else "")
                   + ("; outlier bins: " + ", ".join(f"{o['camera']} ({o['residual_px']:+.1f} px)" for o in r["outliers"])
                      if r["outliers"] else ""))
    return out
