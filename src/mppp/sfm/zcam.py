"""
Mastcam-Z focus breathing (v0p14.4): refined focal length versus focus motor count.

With ``SfmProject.create(zcam_focus_bin=30)`` every Mastcam-Z eye and zoom
(``ZL034``, ``ZR034``) is split into cameras by focus motor count, and each
bin's focal length is refined on its own.  :func:`focus_breathing_table`
collects, per bin, the focus counts, the number of images and observations,
and the initial (median label) and refined focal lengths;
:func:`fit_focus_slopes` fits f = f0 + a (focus - ref) per group, to the
refined bins (weighted by observations) and to the per-image label values;
:func:`plot_focus_breathing` draws both.  All focal lengths are in
full-frame pixels (1648 x 1200).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

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
        use = [r for r in rows if r["refined"] and r["observations"] >= min_observations]
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
        good = [r for r in rows if r["refined"] and r["observations"] >= min_observations]
        weak = [r for r in rows if r not in good and r["f_refined_px"] is not None]
        if good:
            ax.scatter([r["focus_count_median"] for r in good], [r["f_refined_px"] for r in good],
                       s=[12 + 3 * np.sqrt(r["observations"]) for r in good], color="C3", zorder=3,
                       label="refined (size ~ observations)")
            for r in good:
                ax.plot([r["focus_count_min"], r["focus_count_max"]], [r["f_refined_px"]] * 2, "-", color="C3", lw=1)
        if weak:
            ax.plot([r["focus_count_median"] for r in weak], [r["f_refined_px"] for r in weak], "x", color="C3",
                    ms=7, label=f"refined, < {min_observations} observations")
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
