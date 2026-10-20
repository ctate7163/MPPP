"""
Fitting the frozen forms to solved blocks: one streamlined fit per form.

* :func:`fit_navcam` - the Navcam consensus: one joint adjustment of many solved blocks with one camera per eye and
  one rig (:mod:`mppp.calib.joint`), the thermal terms profiled, the early-mission offsets fitted, the rig
  translation fitted jointly, each block left out once; :func:`write_navcam` writes the candidate folder,
  :func:`check_navcam` gates it and :func:`promote_navcam` promotes it.
* :func:`fit_zcam` - the Mastcam-Z calibration of one zoom from solved Navcam + Mastcam-Z blocks: the blocks'
  focus lines averaged per eye and backlash state with their between-block scatter, the regular state derived
  from the backlash state and the measured ratio, the stereo rig averaged; :func:`check_zcam` gates it.
* the robust helpers every fit (and the pipeline stages) share: :func:`huber_fit`, :func:`weighted_mean`,
  :func:`meta_line`, :func:`ols`, :func:`weighted_median`.
"""
from __future__ import annotations

import copy
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import forms, joint as J, models as M
from .constrain import rig_at, thermal_terms

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]


# ====================================================================================== robust helpers
def huber_fit(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, k: float = 1.345, floor: float = 0.05,
              iterations: int = 30) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Weighted Huber IRLS least squares y = X c.  Returns (coefficients, standard errors from the residual scatter,
    robust sd of the residuals (1.4826 MAD, at least ``floor``), residuals)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    wt = np.ones(len(y))
    c = np.zeros(X.shape[1])
    sig = float(floor)
    for _ in range(int(iterations)):
        sw = np.sqrt(w * wt)
        c = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = y - X @ c
        sig = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), float(floor))
        u = np.abs(r) / (k * sig)
        wt = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    resid = y - X @ c
    W = w * wt / max(float(np.mean(w * wt)), 1e-12)
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(np.sum(W * resid ** 2) / dof)
    try:
        C = np.linalg.inv(X.T @ (X * W[:, None])) * s2
        se = np.sqrt(np.maximum(np.diag(C), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return c, se, sig, resid


def huber_line(F: np.ndarray, y: np.ndarray, w: np.ndarray, ref: float, fit_slope: bool, slope0: float = 0.0,
               k: float = 1.345, floor: float = 0.05) -> Tuple[float, float, float, np.ndarray, Tuple[float, Optional[float]]]:
    """(a, b, robust sd, residuals, (sd_a, sd_b)) of y = a + b (F - ref) by :func:`huber_fit`; ``fit_slope``
    False: b = ``slope0`` and only a is fitted."""
    F, y = np.asarray(F, float), np.asarray(y, float)
    X = np.c_[np.ones(len(F)), F - ref] if fit_slope else np.ones((len(F), 1))
    yy = y if fit_slope else y - float(slope0) * (F - ref)
    c, se, sig, _ = huber_fit(X, yy, w, k, floor)
    a, b = float(c[0]), (float(c[1]) if fit_slope else float(slope0))
    return a, b, sig, y - (a + b * (F - ref)), (float(se[0]), float(se[1]) if fit_slope else None)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def weighted_mean(y: np.ndarray, sd: np.ndarray) -> Dict[str, float]:
    """The inverse-variance weighted mean of ``y`` with the between-value scatter tau (DerSimonian-Laird):
    {"value", "se", "tau", "chi2_dof", "n"}."""
    return meta_line(y, sd)


def meta_line(y: np.ndarray, sd: np.ndarray, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Weighted mean (or line in ``x``) of ``y`` with the between-value scatter tau (DerSimonian-Laird) and
    chi^2 / dof; a line adds "slope", "slope_se", "slope_z"."""
    y = np.asarray(y, float)
    sd = np.asarray(sd, float)
    good = np.isfinite(sd) & (sd > 0)
    sd = np.where(good, sd, np.nanmedian(sd[good]) if np.any(good) else 1.0)
    w = 1.0 / sd ** 2
    X = np.ones((len(y), 1)) if x is None else np.c_[np.ones(len(y)), np.asarray(x, float)]
    p = X.shape[1]
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    r = y - X @ beta
    Q = float(np.sum(w * r ** 2))
    dof = max(len(y) - p, 1)
    c = float(np.sum(w) - (np.sum(w ** 2) / np.sum(w) if p == 1 else 0.0))
    tau2 = max(0.0, (Q - dof) / c) if (p == 1 and c > 0) else max(0.0, (Q / dof - 1.0) * float(np.mean(sd ** 2)))
    w2 = 1.0 / (sd ** 2 + tau2)
    beta2 = np.linalg.lstsq(X * np.sqrt(w2)[:, None], y * np.sqrt(w2), rcond=None)[0]
    try:
        cov = np.linalg.inv(X.T @ (X * w2[:, None]))
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    out = {"n": int(len(y)), "chi2_dof": Q / dof, "tau": float(np.sqrt(tau2)), "value": float(beta2[0]),
           "se": float(np.sqrt(max(cov[0, 0], 0.0)))}
    if p == 2:
        out.update({"slope": float(beta2[1]), "slope_se": float(np.sqrt(max(cov[1, 1], 0.0)))})
        out["slope_z"] = out["slope"] / out["slope_se"] if out["slope_se"] > 0 else None
    return out


def ols(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """y = c + s (x - mean x): slope, its se (residual scatter), z, the scatter (rms) and n (rows with finite x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return {"n": int(len(x)), "slope": None, "se": None, "z": None}
    xm = float(np.mean(x))
    X = np.c_[np.ones(len(x)), x - xm]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = float(np.sum(r ** 2) / max(len(x) - 2, 1))
    cov = np.linalg.inv(X.T @ X) * s2
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {"n": int(len(x)), "mean_x": xm, "intercept": float(b[0]), "slope": float(b[1]), "se": se,
            "z": float(b[1] / se) if se > 0 else None, "rms": float(np.sqrt(np.mean(r ** 2)))}


# ====================================================================================== Navcam consensus
RAD_BINS = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
THERMAL_TERMS = ("NL_cx", "NR_cx", "NL_cy", "NR_cy", "NL_fx", "NL_fy", "NR_fx", "NR_fy", "yaw_rate")
THERMAL_STEPS = {"NL_cx": 0.03, "NR_cx": 0.03, "NL_cy": 0.03, "NR_cy": 0.03,      # profile grid steps
                 "NL_fx": 10.0, "NL_fy": 10.0, "NR_fx": 10.0, "NR_fy": 10.0, "yaw_rate": 3.0}
AXIS_TERMS = ("NL_fx", "NL_fy", "NR_fx", "NR_fy")
# the terms profiled by default: the principal points against temperature (NL and NR cx), fx and fy of each eye,
# the rig yaw rate in sol; NR_cx and NR_fx trade a little, so NR_cx comes early
DEFAULT_TERMS = ("NL_cx", "NR_cx", "NL_fx", "NL_fy", "NR_fx", "NR_fy", "yaw_rate")
UNITS = {"NL_fx": "ppm/degC", "NL_fy": "ppm/degC", "NR_fx": "ppm/degC", "NR_fy": "ppm/degC",
         "NL_cx": "px/degC", "NR_cx": "px/degC", "NL_cy": "px/degC", "NR_cy": "px/degC", "yaw_rate": "mdeg/1000 sol"}
EARLY_MISSION_SOL = 380.0                   # the Navcam principal points sat ~2.5-3 px lower in cx before this sol
EARLY_TERMS = ("fx", "fy", "cx", "cy")
EARLY_STEP_PX = 0.5                         # design step of the response surface (px)
EARLY_APPLY_Z = 3.0                         # terms at least this many sd from 0 are applied by the projects
LOO_LIMITS = {"centre_px": 0.05, "corner_px": 0.10}              # leave-one-block-out, per eye (rotation removed)
PASS_LIMITS = {"f_ppm": 20.0, "pp_px": 0.05, "corner_px": 0.02, "rig_mdeg": 1.0}   # converged between passes
K1_APPLY_Z = 3.0                    # a k1 temperature slope from the bins is applied when |slope / se| >= this
PROMOTE_LIMITS = {"min_blocks": 8, "residual_factor": 1.15, "max_camera_change_px": 0.5, "max_rig_change_mdeg": 10.0,
                  "max_ppm_change": 5.0, "max_pp_slope_change": 0.02,
                  "baseline_m": (0.420, 0.428), "max_baseline_disagreement_mm": 1.5, "max_baseline_sd_mm": 1.0}
FORM_ID = "navcam-v1"


def radial_stats(rec, proj, ks, kmap) -> Dict[str, Any]:
    """Residuals (native px) of every observation, overall and by image radius (1 = the frame corner)."""
    by = {int(r["image_id"]): float(r.get("downsample_scale", 1.0)) for r in proj.images if "image_id" in r}
    res, rad, cache = [], [], {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            iid = el.image_id
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            if iid not in cache:
                k = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
                if iid in ks:
                    c0 = np.asarray(cam.params[2:4])
                    k = c0 + (k - c0) / np.asarray(ks[iid], float)
                if kmap is not None and len(k):
                    k = kmap(iid, k, cam)
                cache[iid] = k
            p = im.project_point(pt.xyz)
            if p is None:
                continue
            xy = cache[iid][el.point2D_idx]
            res.append(np.linalg.norm(np.asarray(p) - xy) * by.get(iid, 1.0))
            rad.append(np.linalg.norm(xy - np.array([cam.width, cam.height]) / 2) / (np.hypot(cam.width, cam.height) / 2))
    res, rad = np.asarray(res), np.asarray(rad)
    bins = []
    for a, b in zip(RAD_BINS[:-1], RAD_BINS[1:]):
        m = (rad >= a) & (rad < b)
        bins.append({"r": [a, b], "n": int(m.sum()), "median_px": float(np.median(res[m])) if m.any() else None})
    return {"observations": int(res.size), "median_px": float(np.median(res)), "rms_px": float(np.sqrt(np.mean(res ** 2))),
            "p95_px": float(np.percentile(res, 95)), "radial": bins}


def _thermal_values(th: Dict[str, Any]) -> Dict[str, float]:
    """{"NL_cx": px/degC, "NL_fx": ppm/degC, ..., "yaw_rate", "NL_k1", "NR_k1"} of a consensus result's ``thermal``."""
    pp = th.get("pp_slopes") or {}
    ax = th.get("axis_ppm") or {}
    f = float(th.get("ppm_per_degC") or 0.0)
    return {"NL_cx": float(pp.get("NL", [0.0, 0.0])[0]), "NL_cy": float(pp.get("NL", [0.0, 0.0])[1]),
            "NR_cx": float(pp.get("NR", [0.0, 0.0])[0]), "NR_cy": float(pp.get("NR", [0.0, 0.0])[1]),
            "NL_fx": float(ax.get("NL", [f, f])[0]), "NL_fy": float(ax.get("NL", [f, f])[1]),
            "NR_fx": float(ax.get("NR", [f, f])[0]), "NR_fy": float(ax.get("NR", [f, f])[1]),
            "yaw_rate": float(th.get("yaw_rate_mdeg_per_1000sol") or 0.0),
            "NL_k1": float((th.get("k1_slopes") or {}).get("NL") or 0.0),
            "NR_k1": float((th.get("k1_slopes") or {}).get("NR") or 0.0)}


def _pp_slopes(v: Dict[str, float]) -> Dict[int, Tuple[float, float]]:
    return {1: (float(v["NL_cx"]), float(v["NL_cy"])), 2: (float(v["NR_cx"]), float(v["NR_cy"]))}


def _axis_ppm(v: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    return {"NL": (float(v["NL_fx"]), float(v["NL_fy"])), "NR": (float(v["NR_fx"]), float(v["NR_fy"]))}


def _mean_ppm(v: Dict[str, float]) -> float:
    return float(np.mean([v[k] for k in AXIS_TERMS]))


def _drift_with_yaw(drift: Optional[Dict[str, Any]], yaw_rate_per_1000: float) -> Optional[Dict[str, Any]]:
    """The drift with one yaw rate (mdeg per 1000 sol, before and after the knot alike)."""
    if not drift:
        return drift
    d = dict(drift)
    r = 1e-3 * float(yaw_rate_per_1000)
    d["yaw_mdeg_per_sol"] = r
    d["yaw_fit_mdeg_per_sol"] = r
    if "yaw_early_mdeg_per_sol" in d or d.get("knot_sol"):
        d["yaw_early_mdeg_per_sol"] = r
    return d


def _joint_kw(v: Dict[str, float], drift: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The keyword arguments of :func:`mppp.calib.joint.joint_adjust` for thermal values ``v``."""
    kw: Dict[str, Any] = {"pp_slopes": _pp_slopes(v), "axis_ppm": _axis_ppm(v)}
    k1 = {e: float(v.get(f"{e}_k1") or 0.0) for e in ("NL", "NR")}
    if any(k1.values()):
        kw["k1_slopes"] = k1
    d = _drift_with_yaw(drift, v.get("yaw_rate", 0.0))
    if d:
        kw["drift"] = d
    return kw


def profile_thermal(rec, proj, temps, values: Dict[str, float], T0: float, term: str, drift=None,
                    iterations: int = 50, npoints: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """The joint cost over a grid of one thermal term, the others at ``values``, each adjustment started from the
    same converged state (``rec`` is not changed).  The best value is the minimum of a parabola through the costs,
    its sd sqrt(variance factor / curvature).  A minimum outside the grid extends it by two steps (twice at most)."""
    step = THERMAL_STEPS[term]
    c0 = float(values[term])
    grid = [c0 + step * (k - (npoints - 1) / 2) for k in range(npoints)]
    rows: list = []
    done = set()
    t_all = time.time()
    fit: Dict[str, Any] = {}
    lo = hi = c0
    for _ in range(3):
        for g in grid:
            if round(g, 9) in done:
                continue
            done.add(round(g, 9))
            v = dict(values, **{term: float(g)})
            r = copy.deepcopy(rec)
            t = time.time()
            ba = J.joint_adjust(r, proj, temps, _mean_ppm(v), T0, max_iterations=iterations, **_joint_kw(v, drift))
            red = max(2 * ba["observations"] - 1, 1)
            rows.append({"ppm_per_degC": float(g), "cost": float(ba["final_cost"]), "iterations": ba["iterations"],
                         "variance_factor": float(2 * ba["final_cost"] / red)})
            if verbose:
                log.info(f"  {term} {g:+.4f}: cost {ba['final_cost']:.2f} it {ba['iterations']} ({time.time() - t:.0f} s)  "
                      f"{J.progress(len(rows), max(npoints, len(rows)), t_all, 'profile ' + term)}")
        fit = J.fit_profile(sorted(rows, key=lambda r: r["ppm_per_degC"]))
        lo, hi = min(r["ppm_per_degC"] for r in rows), max(r["ppm_per_degC"] for r in rows)
        b = fit["best_ppm_per_degC"]
        if lo <= b <= hi:
            break
        grid = [hi + step, hi + 2 * step] if b > hi else [lo - step, lo - 2 * step]
    out = {"term": term, "unit": UNITS.get(term), "best": float(fit["best_ppm_per_degC"]),
           "sd": float(fit["sd_ppm_per_degC"]), "start": c0, "curvature": fit["curvature"],
           "variance_factor": fit["variance_factor"],
           "rows": [{"value": r["ppm_per_degC"], "cost": r["cost"], "iterations": r["iterations"]} for r in fit["rows"]],
           "at_edge": not (lo <= fit["best_ppm_per_degC"] <= hi)}
    out["z"] = (out["best"] / out["sd"]) if out["sd"] and np.isfinite(out["sd"]) and out["sd"] > 0 else None
    if verbose:
        log.info(f"{term}: {out['best']:+.4f} +- {out['sd']:.4f} {UNITS.get(term, '')} (start {c0:+.4f})"
              + ("  [outside the grid]" if out["at_edge"] else ""))
    return out


# ------------------------------------------------------------------------------------------- early-mission offsets
def _quadratic_design(n: int, h: float):
    """Centre, +-h on each axis and +h on each pair of axes: (1 + 2n + n(n-1)/2) points, a full quadratic."""
    pts = [np.zeros(n)]
    for i in range(n):
        for s in (1.0, -1.0):
            e = np.zeros(n)
            e[i] = s * h
            pts.append(e)
    for i in range(n):
        for j in range(i + 1, n):
            e = np.zeros(n)
            e[i] = e[j] = h
            pts.append(e)
    return pts


def _fit_quadratic(D: np.ndarray, c: np.ndarray):
    """cost = a + g.d + 1/2 d^T H d through the design points ``D`` (rows); returns (a, g, H, rms of the fit)."""
    n = D.shape[1]
    cols = [np.ones(len(D))] + [D[:, i] for i in range(n)] + [0.5 * D[:, i] ** 2 for i in range(n)] \
        + [D[:, i] * D[:, j] for i in range(n) for j in range(i + 1, n)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    a, g = coef[0], coef[1:1 + n]
    H = np.diag(coef[1 + n:1 + 2 * n])
    k = 1 + 2 * n
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = H[j, i] = coef[k]
            k += 1
    return float(a), g, H, float(np.sqrt(np.mean((A @ coef - c) ** 2)))


def read_cameras_bin(path: PathLike) -> Dict[int, np.ndarray]:
    """{camera id: params} of a COLMAP ``cameras.bin`` (no image or point data read)."""
    import struct
    npar = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 4, 12: 5}
    out = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(n):
            cid, model = struct.unpack("<Ii", fh.read(8))
            fh.read(16)
            k = npar[model]
            out[int(cid)] = np.array(struct.unpack("<" + "d" * k, fh.read(8 * k)))
    return out


def early_start_from_blocks(scapes_cfg: Dict[str, PathLike], before_sol: float = EARLY_MISSION_SOL,
                            model: str = "cahv_ba") -> Dict[str, Any]:
    """A start for :func:`fit_early_offsets` from the block alignments: each block's refined eye cameras minus
    their start (fx, fy, cx, cy); the mean over the blocks all before ``before_sol`` minus that over the blocks all
    after it."""
    from ..sfm.project import SfmProject
    rows: Dict[str, Any] = {}
    for name, w in scapes_cfg.items():
        root = Path(w) / "colmap" if (Path(w) / "colmap").is_dir() else Path(w)
        try:
            proj = SfmProject.load(root)
            cams = read_cameras_bin(root / "sparse" / model / "cameras.bin")
        except Exception:                                   # noqa: BLE001 - a block without them is left out
            continue
        sols = [float(r["sol"]) for r in proj.images if r.get("sol") is not None]
        if not sols:
            continue
        db = proj.settings.get("database", {}).get("cameras", {})
        d = [cams[int(db[e])][:4] - np.asarray(proj.cameras[e]["params"][:4], float)
             for e in ("NL", "NR") if e in db and int(db[e]) in cams and e in proj.cameras]
        if not d:
            continue
        rows[name] = {"sol_median": float(np.median(sols)), "early": bool(max(sols) < before_sol),
                      "late": bool(min(sols) >= before_sol), "d": np.mean(d, axis=0).tolist()}
    e = [v["d"] for v in rows.values() if v["early"]]
    l_ = [v["d"] for v in rows.values() if v["late"]]
    start = (np.mean(e, axis=0) - (np.mean(l_, axis=0) if l_ else 0.0)).tolist() if e else [0.0] * 4
    return {"start": [float(x) for x in start], "blocks": rows, "early_blocks": len(e), "late_blocks": len(l_)}


def fit_early_offsets(rec, proj, temps, ppm: float, T0: float, xkw: Dict[str, Any], before_sol: float = EARLY_MISSION_SOL,
                      start=(0.0, 0.0, 0.0, 0.0), step: float = EARLY_STEP_PX, rounds: int = 4, iterations: int = 50,
                      verbose: bool = True) -> Dict[str, Any]:
    """fx, fy, cx, cy offsets (px, common to both eyes) of the Navcam images taken before ``before_sol``.  Each
    evaluation is one joint adjustment from the same state (``rec`` is not changed) with the images' keypoints
    mapped by :func:`mppp.calib.joint.early_keypoint_map`.  A full quadratic response surface of the cost over the
    four offsets (15 adjustments) gives the minimum and the covariance vf H^-1; ``rounds`` re-centres the design on
    the minimum.  z = offset / sd per term; the likelihood ratio 2 (cost(0) - cost(best)) / vf against chi2(4)."""
    from scipy import stats
    n_early = len(J.early_keypoint_map(proj, before_sol, np.zeros(4)).early_images)
    out: Dict[str, Any] = {"before_sol": float(before_sol), "terms": list(EARLY_TERMS), "early_images": n_early,
                           "step_px": float(step), "rounds": []}
    if not n_early:
        out["note"] = f"no images before sol {before_sol:g}: nothing to fit"
        return out
    cache: Dict[tuple, Dict[str, float]] = {}
    t_all = time.time()
    n_max = max(1, int(rounds)) * 16 + 2

    def cost_at(x):
        key = tuple(np.round(x, 6))
        if key not in cache:
            r = copy.deepcopy(rec)
            t = time.time()
            ba = J.joint_adjust(r, proj, temps, ppm, T0, max_iterations=iterations,
                                extra_map=J.early_keypoint_map(proj, before_sol, x), **xkw)
            red = max(2 * ba["observations"] - 1, 1)
            cache[key] = {"cost": float(ba["final_cost"]), "vf": float(2 * ba["final_cost"] / red),
                          "iterations": ba["iterations"]}
            if verbose:
                log.info("  early offsets " + " ".join(f"{k} {v:+.3f}" for k, v in zip(EARLY_TERMS, x))
                      + f": cost {ba['final_cost']:.2f} it {ba['iterations']} ({time.time() - t:.0f} s)  "
                      + J.progress(len(cache), n_max, t_all, "adjustment (at most)"))
        return cache[key]

    # Newton steps on the response surface within a trust region of ``max_move`` px per term; a step is kept only
    # if it lowers the cost, else the region is halved.  The covariance comes from the last design.
    best = np.asarray(start, float)
    cov, vf, max_move, converged = None, None, 4.0 * step, False
    for rd in range(max(1, int(rounds))):
        D = np.array(_quadratic_design(4, step))
        rows = [cost_at(best + d) for d in D]
        c = np.array([r["cost"] for r in rows])
        vf = float(np.median([r["vf"] for r in rows]))
        a, g, H, fit_rms = _fit_quadratic(D, c)
        ev = np.linalg.eigvalsh(H)
        ok = bool(np.all(ev > 0))
        if ok:
            cov = vf * np.linalg.inv(H)
            dx = -np.linalg.solve(H, g)
        else:
            dx = -g / max(float(np.max(np.abs(np.diag(H)))), 1e-9)
        big = float(np.max(np.abs(dx)))
        if big > max_move:
            dx = dx * (max_move / big)
        rec_rd = {"centre": best.tolist(), "costs": c.tolist(), "gradient": g.tolist(), "hessian": H.tolist(),
                  "eigenvalues": ev.tolist(), "fit_rms": fit_rms, "step": dx.tolist(), "positive_definite": ok}
        out["rounds"].append(rec_rd)
        if verbose:
            log.info(f"  early offsets, round {rd + 1}/{rounds}: centre " + " ".join(f"{k} {v:+.3f}" for k, v in zip(EARLY_TERMS, best))
                  + "; step " + " ".join(f"{v:+.3f}" for v in dx) + (" (positive definite)" if ok else " (not convex)"))
        if ok and np.all(np.abs(dx) < 0.5 * step):
            best = best + dx
            converged = True
            break
        trial = best + dx
        if cost_at(trial)["cost"] < c[0]:
            best = trial
        else:
            max_move *= 0.5
            rec_rd["rejected"] = True
            for d, cc in zip(D, c):
                if cc < cost_at(best)["cost"]:
                    best = best + d
            if max_move < 0.25 * step:
                break
    out["converged"] = converged
    c_best = cost_at(best)["cost"]
    c_zero = cost_at(np.zeros(4))["cost"]
    out.update({"offsets_px": dict(zip(EARLY_TERMS, map(float, best))), "cost_at_zero": c_zero, "cost_at_best": c_best,
                "variance_factor": vf})
    if cov is not None:
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        z = np.where(sd > 0, best / np.where(sd > 0, sd, 1.0), np.nan)
        corr = cov / np.outer(np.where(sd > 0, sd, 1.0), np.where(sd > 0, sd, 1.0))
        lr = max(0.0, 2.0 * (c_zero - c_best) / vf)
        out.update({"sd_px": dict(zip(EARLY_TERMS, map(float, sd))), "z": dict(zip(EARLY_TERMS, map(float, z))),
                    "p_value": dict(zip(EARLY_TERMS, (float(2 * stats.norm.sf(abs(v))) for v in z))),
                    "correlation": corr.tolist(), "likelihood_ratio": lr,
                    "p_value_all": float(stats.chi2.sf(lr, 4)),
                    "applied_terms": [k for k, v in zip(EARLY_TERMS, z) if abs(v) >= EARLY_APPLY_Z]})
    if verbose and cov is not None:
        log.info("early-mission offsets (before sol %g, %d images): " % (before_sol, n_early)
              + ", ".join(f"{k} {out['offsets_px'][k]:+.3f} +- {out['sd_px'][k]:.3f} px (z {out['z'][k]:+.1f})"
                          for k in EARLY_TERMS)
              + f"; all four: LR {out['likelihood_ratio']:.1f}, p {out['p_value_all']:.2g}")
    return out


# --------------------------------------------------------------------------------------------- joint-side tools
def leave_one_block_out(rec, proj, idx: Dict[str, Any], temps, ppm: float, T0: float, xkw: Dict[str, Any],
                        extra_map=None, iterations: int = 40, verbose: bool = True) -> Dict[str, Any]:
    """The joint without each block in turn (:func:`mppp.calib.joint.subset`), started from the converged joint
    with the thermal terms held, and how far each eye's camera moves (:func:`mppp.calib.joint.compare`, the best
    rotation removed): centre / corner rms px, the cx changes, the rig yaw / pitch change and the stereo pointing
    change NR cx - NL cx x fR/fL + fR yaw (px).  Returns {"blocks", "max", "worst", "limits", "ok", "errors"}."""
    full = J.shared_state(rec)
    cams_full = {k: J.pycolmap_camera(full[k]["model"], full[k]["params"]) for k in ("NL", "NR")}
    R_full = np.asarray(full["rig_R"], float)
    rows: Dict[str, Any] = {}
    t_all = time.time()
    names = list(idx["images"])
    for kb, name in enumerate(names, 1):
        keep = [n for n in names if n != name]
        r = J.subset(rec, idx, keep)
        t = time.time()
        try:
            ba = J.joint_adjust(r, proj, temps, ppm, T0, max_iterations=iterations, extra_map=extra_map, **xkw)
        except Exception as e:                                         # noqa: BLE001
            rows[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        st = J.shared_state(r)
        row: Dict[str, Any] = {"iterations": ba["iterations"], "seconds": time.time() - t}
        for k in ("NL", "NR"):
            d = J.compare(cams_full[k], J.pycolmap_camera(st[k]["model"], st[k]["params"]))
            row[k] = {"centre_px": d["centre_rms_px"], "corner_px": d["corner_rms_px"], "rms_px": d["rms_px"],
                      "dcx_px": float(st[k]["params"][2] - full[k]["params"][2]),
                      "dcy_px": float(st[k]["params"][3] - full[k]["params"][3])}
        ang = J.rig_angles(np.asarray(st["rig_R"], float), R_full)
        ratio = float(full["NR"]["params"][0]) / float(full["NL"]["params"][0])
        fR = float(full["NR"]["params"][0])
        row["rig_yaw_mdeg"], row["rig_pitch_mdeg"] = float(ang["yaw_mdeg"]), float(ang["pitch_mdeg"])
        row["stereo_px"] = float(row["NR"]["dcx_px"] - ratio * row["NL"]["dcx_px"]
                                 + fR * np.radians(1e-3 * row["rig_yaw_mdeg"]))
        rows[name] = row
        if verbose:
            log.info(f"  leave out {name}: NL centre {row['NL']['centre_px']:.3f} corner {row['NL']['corner_px']:.3f}, NR "
                  f"centre {row['NR']['centre_px']:.3f} corner {row['NR']['corner_px']:.3f} px, stereo "
                  f"{row['stereo_px']:+.3f} px  {J.progress(kb, len(names), t_all, 'leave-one-out')}")
    ok_rows = [v for v in rows.values() if "error" not in v]
    mx = {"centre_px": max((max(v["NL"]["centre_px"], v["NR"]["centre_px"]) for v in ok_rows), default=None),
          "corner_px": max((max(v["NL"]["corner_px"], v["NR"]["corner_px"]) for v in ok_rows), default=None),
          "stereo_px": max((abs(v["stereo_px"]) for v in ok_rows), default=None)}
    worst = {k: max(rows, key=lambda n: (max(rows[n]["NL"][k], rows[n]["NR"][k]) if "error" not in rows[n] else -1))
             for k in ("centre_px", "corner_px")} if ok_rows else {}
    ok = bool(ok_rows) and mx["centre_px"] <= LOO_LIMITS["centre_px"] and mx["corner_px"] <= LOO_LIMITS["corner_px"]
    return {"blocks": rows, "max": mx, "worst": worst, "limits": dict(LOO_LIMITS), "ok": ok,
            "errors": sorted(n for n, v in rows.items() if "error" in v)}


def k1_slopes_from_blocks(scapes_cfg: Dict[str, PathLike], min_rows: int = 6, z_apply: float = K1_APPLY_Z,
                          verbose: bool = True) -> Dict[str, Any]:
    """k1 against camera temperature per Navcam eye from the temperature bins of finished blocks
    (``colmap/project.json`` ``settings.thermal.rows`` with k1 refined): k1 = a_block,eye + s_eye T, weighted by
    the bin's observations, one offset per block and eye so that only the within-block change counts.  ``apply``
    {"NL", "NR"}: the slope where |z| >= ``z_apply``, else 0."""
    from ..sfm.thermal import eye_key
    data: Dict[str, list] = {"NL": [], "NR": []}
    used = {}
    for name, folder in scapes_cfg.items():
        f = Path(folder)
        pj = f / "colmap" / "project.json" if (f / "colmap" / "project.json").is_file() else f / "project.json"
        if not pj.is_file():
            continue
        try:
            th = (json.loads(pj.read_text(encoding="utf-8")).get("settings") or {}).get("thermal") or {}
        except ValueError:
            continue
        n = 0
        for r in th.get("rows") or []:
            if "k1" not in r or "k1" not in (r.get("free") or []) or r.get("T_median_degC") is None:
                continue
            eye = eye_key(r.get("eye") or r.get("camera", ""))[:2]
            if eye not in data:
                continue
            data[eye].append((name, float(r["T_median_degC"]), float(r["k1"]), float(r.get("observations") or r.get("images") or 1)))
            n += 1
        if n:
            used[name] = n
    out: Dict[str, Any] = {"blocks": used, "z_apply": float(z_apply), "fits": {}, "apply": {"NL": 0.0, "NR": 0.0}}
    for eye, rows in data.items():
        blocks = sorted({r[0] for r in rows})
        if len(rows) < min_rows or len(rows) <= len(blocks) + 1:
            out["fits"][eye] = {"n": len(rows), "slope": None, "note": "too few bins with k1"}
            continue
        y = np.array([r[2] for r in rows])
        X = np.zeros((len(rows), len(blocks) + 1))
        for i, r in enumerate(rows):
            X[i, blocks.index(r[0])] = 1.0
            X[i, -1] = r[1]
        w = np.array([r[3] for r in rows])
        w = w / w.mean()
        A = X.T @ (X * w[:, None])
        b = X.T @ (w * y)
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            out["fits"][eye] = {"n": len(rows), "slope": None, "note": "singular (one temperature per block?)"}
            continue
        res = y - X @ x
        dof = max(len(rows) - X.shape[1], 1)
        s2 = float(np.sum(w * res ** 2) / dof)
        se = float(np.sqrt(max(np.linalg.inv(A)[-1, -1] * s2, 0.0)))
        z = float(x[-1] / se) if se > 0 else None
        out["fits"][eye] = {"n": len(rows), "blocks": len(blocks), "slope": float(x[-1]), "se": se, "z": z,
                            "rms": float(np.sqrt(np.mean(res ** 2)))}
        if z is not None and abs(z) >= z_apply:
            out["apply"][eye] = float(x[-1])
    if verbose:
        log.info("k1 against temperature from the blocks' bins: " + "; ".join(
            f"{e} {v['slope']:+.2e} +- {v['se']:.1e} /degC (z {v['z']:+.1f}, {v['n']} bins)" if v.get("slope") is not None
            else f"{e} -" for e, v in out["fits"].items()) + f" -> applied NL {out['apply']['NL']:+.2e}, NR "
            f"{out['apply']['NR']:+.2e}")
    return out


def consensus_change(a_dir: PathLike, b_dir: PathLike) -> Dict[str, Any]:
    """How much the Navcam models changed between two folders: fx, fy per eye in ppm, cx, cy in px, the
    distortion's corner change in px (rotation removed), the rig rotation in mdeg, and whether every one is below
    :data:`PASS_LIMITS` (``converged``)."""
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    out: Dict[str, Any] = {"a": str(a_dir), "b": str(b_dir), "limits": dict(PASS_LIMITS)}
    worst = {"f_ppm": 0.0, "pp_px": 0.0, "corner_px": 0.0}
    for eye in ("L", "R"):
        A = M._read(M.camera_file("N", eye, a_dir))
        B = M._read(M.camera_file("N", eye, b_dir))
        pa, pb = np.asarray(A["params"], float), np.asarray(B["params"], float)
        e = {"dfx_ppm": 1e6 * (pb[0] / pa[0] - 1), "dfy_ppm": 1e6 * (pb[1] / pa[1] - 1),
             "dcx_px": float(pb[2] - pa[2]), "dcy_px": float(pb[3] - pa[3])}
        pm = pb.copy()
        pm[:4] = pa[:4]
        d = J.compare(J.pycolmap_camera(A["model"], pa), J.pycolmap_camera(B["model"], pm))
        e["distortion_corner_px"] = float(d["corner_rms_px"])
        out[f"N{eye}"] = e
        worst["f_ppm"] = max(worst["f_ppm"], abs(e["dfx_ppm"]), abs(e["dfy_ppm"]))
        worst["pp_px"] = max(worst["pp_px"], abs(e["dcx_px"]), abs(e["dcy_px"]))
        worst["corner_px"] = max(worst["corner_px"], e["distortion_corner_px"])
    ra = M._read(M.rig_file("N", a_dir))
    rb = M._read(M.rig_file("N", b_dir))
    worst["rig_mdeg"] = J.rotation_mdeg(ra["R_sensor_from_ref"], rb["R_sensor_from_ref"])
    out["worst"] = worst
    out["converged"] = all(worst[k] <= PASS_LIMITS[k] for k in PASS_LIMITS)
    return out


def block_span_m(root: PathLike) -> float:
    """The largest horizontal distance between the Navcam station centres of a block (its waypoint priors)."""
    from ..sfm.project import SfmProject
    from ..sfm.reconstruction import navcam_network
    root = Path(root)
    if (root / "colmap").is_dir():
        root = root / "colmap"
    return float(navcam_network(SfmProject.load(root))["span_m"])


def large_blocks(scapes_cfg: Dict[str, PathLike], min_span_m: float = 30.0) -> Dict[str, Any]:
    """The blocks of ``scapes_cfg`` whose stations span at least ``min_span_m`` (``{name: [folder, span]}``)."""
    out = {}
    for n, f in scapes_cfg.items():
        try:
            s = block_span_m(f)
        except Exception:                                                  # noqa: BLE001
            continue
        if s >= min_span_m:
            out[n] = [str(f), s]
    return out


def baseline_degeneracy(rec, proj, temps, ppm: float, T0: float, xkw: Dict[str, Any], extra_map, delta_mm: float = 1.0,
                        iterations: int = 100, verbose: bool = True) -> Dict[str, Any]:
    """How the rest of the model compensates a baseline change: from the converged joint (``rec`` is not changed)
    the rig translation is held at its length +- ``delta_mm`` and everything else re-converged.  Returns the yaw,
    pitch and NR / NL cx change per mm and the mean cost rise (with the variance factor: the baseline's sd)."""
    import pycolmap
    rid, cid, T = J.stereo_rig(rec)
    t_joint = np.array(T.translation, float)
    R_joint = np.array(T.rotation.matrix())
    b0 = float(np.linalg.norm(t_joint))
    ang0 = J.rig_angles(R_joint)
    base = {"cost": None, "yaw_mdeg": ang0["yaw_mdeg"], "pitch_mdeg": ang0["pitch_mdeg"],
            "NR_cx": float(rec.cameras[2].params[2]), "NL_cx": float(rec.cameras[1].params[2])}
    rows = []
    for sgn in (+1.0, -1.0):
        r = copy.deepcopy(rec)
        t_new = t_joint * (b0 + sgn * delta_mm * 1e-3) / b0
        sid = [s for s in r.rigs[rid].non_ref_sensors if int(s.id) == cid][0]
        r.rigs[rid].set_sensor_from_rig(sid, pycolmap.Rigid3d(pycolmap.Rotation3d(R_joint), t_new))
        t = time.time()
        ba = J.joint_adjust(r, proj, temps, ppm, T0, max_iterations=iterations, refine_rig="rotation",
                            extra_map=extra_map, **xkw)
        _, _, T1 = J.stereo_rig(r)
        ang = J.rig_angles(np.asarray(T1.rotation.matrix()))
        rows.append({"dbaseline_mm": sgn * delta_mm, "cost": float(ba["final_cost"]), "iterations": ba["iterations"],
                     "yaw_mdeg": ang["yaw_mdeg"], "pitch_mdeg": ang["pitch_mdeg"],
                     "NR_cx": float(r.cameras[2].params[2]), "NL_cx": float(r.cameras[1].params[2]),
                     "seconds": time.time() - t})
        if verbose:
            log.info(f"  baseline {sgn * delta_mm:+.1f} mm: cost {ba['final_cost']:.1f}, yaw {ang['yaw_mdeg']:+.2f} mdeg, "
                  f"NR cx {rows[-1]['NR_cx']:.3f} px ({rows[-1]['seconds']:.0f} s)")
    plus, minus = rows
    return {"delta_mm": delta_mm, "baseline_m": b0, "rows": rows, "at_joint": base,
            "yaw_mdeg_per_mm": (plus["yaw_mdeg"] - minus["yaw_mdeg"]) / (2 * delta_mm),
            "pitch_mdeg_per_mm": (plus["pitch_mdeg"] - minus["pitch_mdeg"]) / (2 * delta_mm),
            "NR_cx_px_per_mm": (plus["NR_cx"] - minus["NR_cx"]) / (2 * delta_mm),
            "NL_cx_px_per_mm": (plus["NL_cx"] - minus["NL_cx"]) / (2 * delta_mm),
            "cost_rise_mean": 0.5 * (plus["cost"] + minus["cost"]),
            "note": "yaw / cx per mm of baseline with everything else re-converged; cost_rise_mean - the joint cost "
                    "is the rise for +-delta: sd_baseline_mm ~ delta / sqrt(2 rise / variance factor)"}


def fit_rig_translation(scapes_cfg: Dict[str, PathLike], res: Dict[str, Any], points: int = 8000,
                        sigma_m: float = 0.05, iterations: int = 100, verbose: bool = True) -> Dict[str, Any]:
    """The Navcam rig translation from the large blocks (their waypoint priors fix the scale independently of the
    baseline): merged as in :func:`fit_navcam` with the cameras of ``res`` held, the rig yaw held, pitch, roll and
    the translation refined (a weak prior of ``sigma_m`` per axis on the right centre)."""
    import pycolmap
    from ..sfm.project import navcam_distortion_terms
    cams = {k: pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840,
                               params=np.array([res[k][n] for n in forms.NAVCAM_PARAMS], float))
            for k in ("NL", "NR")}
    R0 = np.asarray(res["rig_R"], float)
    t0 = np.asarray(res["rig_t"], float)
    th = res["thermal"]
    t = time.time()
    rec, proj, idx = J.merge_scapes((J.load_scape(n, scapes_cfg[n]) for n in scapes_cfg), cams, (R0, t0),
                                    points_per_scape=points, seed=0)
    temps = idx["temps"]
    if verbose:
        log.info(f"rig translation: merged {len(idx['images'])} large blocks, {rec.num_reg_images()} images "
              f"({time.time() - t:.0f} s)")
    proj.settings["navcam_rig_yaw"] = "hold"
    for cid, key in ((1, "NL"), (2, "NR")):
        c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
        proj.cameras[key] = navcam_distortion_terms(c, "refine")
    tv = _thermal_values(th)
    xkw: Dict[str, Any] = {"pp_slopes": _pp_slopes(tv), "axis_ppm": _axis_ppm(tv), "rig_translation_sigma_m": float(sigma_m)}
    if res.get("drift"):
        xkw["drift"] = res["drift"]
    if (th.get("k1_slopes") or {}) and any((th.get("k1_slopes") or {}).values()):
        xkw["k1_slopes"] = dict(th["k1_slopes"])
    t = time.time()
    ba = J.joint_adjust(rec, proj, temps, _mean_ppm(tv), float(th["T0_degC"]), max_iterations=iterations,
                        covariance=True, refine_rig=True, hold_cameras=("NL", "NR"), **xkw)
    _, _, T = J.stereo_rig(rec)
    R1, t1 = np.asarray(T.rotation.matrix()), np.asarray(T.translation)
    C0, C1 = -R0.T @ t0, -R1.T @ t1
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor", 1.0))
    rig_cov = [np.asarray(v) for k, v in (cov.get("blocks") or {}).items() if k.startswith("rig:")]
    sd_t = (np.sqrt(np.clip(np.diag(rig_cov[0])[-3:], 0, None) * vf) * 1e3).tolist() if rig_cov else None
    out = {"blocks": {k: len(v) for k, v in idx["images"].items()}, "sigma_m": float(sigma_m),
           "final_cost": float(ba["final_cost"]), "iterations": ba["iterations"], "variance_factor": vf,
           "t_start_m": t0.tolist(), "t_m": t1.tolist(), "R": R1.tolist(),
           "dcentre_mm": (1e3 * (C1 - C0)).tolist(), "sd_t_mm": sd_t,
           "baseline_start_m": float(np.linalg.norm(t0)), "baseline_m": float(np.linalg.norm(t1)),
           "dbaseline_mm": 1e3 * float(np.linalg.norm(t1) - np.linalg.norm(t0)),
           "rig_abs_mdeg": J.rig_angles(R1), "seconds": time.time() - t}
    if verbose:
        log.info(f"rig translation ({len(out['blocks'])} large blocks): right centre change [mm] "
              f"{np.round(out['dcentre_mm'], 3).tolist()} (x along the baseline), sd [mm] "
              f"{None if sd_t is None else np.round(sd_t, 3).tolist()}, baseline {out['baseline_m']:.5f} m "
              f"({out['dbaseline_mm']:+.2f} mm)  ({out['seconds']:.0f} s)")
    return out


# ---------------------------------------------------------------------------------------------------- the fit
def fit_navcam(scapes_cfg: Dict[str, PathLike], points: int = 8000, iterations: int = 100,
               start_dir: Optional[PathLike] = None, verbose: bool = True,
               translation_scapes: Optional[Dict[str, PathLike]] = None, translation_sigma_m: float = 0.05,
               apply_translation: bool = False, refit_thermal: bool = True,
               thermal_terms_profiled: Optional[Sequence[str]] = None, early_mission_sol: Optional[float] = EARLY_MISSION_SOL,
               early_rounds: int = 2, early_start: Optional[Tuple[float, float, float, float]] = None,
               thermal_rounds: int = 2, joint_translation: bool = True,
               degeneracy_mm: float = 1.0, k1_slopes: Optional[Dict[str, float]] = None,
               leave_one_out: bool = True, loo_iterations: int = 40, early_hold: bool = False) -> Dict[str, Any]:
    """
    The Navcam consensus in the frozen form: the blocks ``{name: WORK folder}`` in one joint adjustment with one
    camera per eye (fisheye + tangential, k3 and k4 fitted as a pair, sx1 = sy1 = 0) at T0 and one rig (one yaw
    for every temperature, with a rate in sol; pitch and roll with the start rig's drift), started from the cameras
    and rig of ``start_dir`` (default: the models in use).

    * ``refit_thermal``: the thermal terms (``thermal_terms_profiled``, default :data:`DEFAULT_TERMS`: NL and NR cx
      px/degC, fx and fy of each eye ppm/degC, the yaw rate mdeg/1000 sol) profiled one after the other
      (:func:`profile_thermal`), ``thermal_rounds`` times, then the joint adjusted once more at the best values;
    * ``k1_slopes`` {"NL", "NR"} per degC (:func:`k1_slopes_from_blocks`; default: the start cameras') is applied
      per image, not profiled;
    * ``early_mission_sol``: fx, fy, cx, cy offsets of the images before that sol (:func:`fit_early_offsets`;
      ``early_hold``: the start's offsets applied, not refitted);
    * ``joint_translation``: the final adjustment frees the rig translation (prior ``translation_sigma_m`` per axis)
      together with the cameras and the rotation, with a degeneracy profile (:func:`baseline_degeneracy`);
      ``translation_scapes`` (the large blocks, :func:`large_blocks`): the cross-check with the cameras held;
    * ``leave_one_out``: each block left out once (:func:`leave_one_block_out`), the promotion gate.

    Returns the shared cameras, the rig, the thermal terms with their sd and profiles, the early-mission offsets,
    residuals by radius and the per-block image counts - what :func:`write_navcam` writes.
    """
    import pycolmap
    from ..sfm.project import navcam_distortion_terms
    d = Path(start_dir) if start_dir else M._folder(None)
    nl, nr = M.load_camera("N", "L", d), M.load_camera("N", "R", d)
    rj = M.load_rig("N", d)
    if rj is None:
        raise FileNotFoundError(f"no Navcam rig file in {d}")
    cams = {k: pycolmap.Camera(model=j["model"], width=5120, height=3840, params=np.asarray(j["params"], float))
            for k, j in (("NL", nl), ("NR", nr))}
    rig = (np.asarray(rj["R_sensor_from_ref"], float), np.asarray(rj["t_sensor_from_ref_m"], float))
    tl, tr = thermal_terms(nl.get("thermal")), thermal_terms(nr.get("thermal"))
    t0 = tl["T0"]
    tvals = {"NL_cx": tl["cx_px"], "NL_cy": tl["cy_px"], "NR_cx": tr["cx_px"], "NR_cy": tr["cy_px"],
             "NL_fx": tl["fx_ppm"], "NL_fy": tl["fy_ppm"], "NR_fx": tr["fx_ppm"], "NR_fy": tr["fy_ppm"],
             "NL_k1": tl["k1"], "NR_k1": tr["k1"]}
    k1_source = "start cameras" if (tl["k1"] or tr["k1"]) else "none (0)"
    if k1_slopes is not None:
        for eye in ("NL", "NR"):
            tvals[f"{eye}_k1"] = float(k1_slopes.get(eye) or 0.0)
        k1_source = "temperature bins of the blocks (k1_slopes_from_blocks)"
    drift = copy.deepcopy(rj.get("drift"))
    tvals["yaw_rate"] = 1000.0 * float((drift or {}).get("yaw_mdeg_per_sol") or 0.0)
    terms = tuple(thermal_terms_profiled) if thermal_terms_profiled is not None else DEFAULT_TERMS
    bad = [k for k in terms if k not in THERMAL_TERMS]
    if bad:
        raise ValueError(f"thermal terms {bad}: use {THERMAL_TERMS}")
    t = time.time()
    rec, proj, idx = J.merge_scapes((J.load_scape(n, scapes_cfg[n]) for n in scapes_cfg), cams, rig,
                                    points_per_scape=points, seed=0)
    temps = idx["temps"]
    if verbose:
        log.info(f"merged {len(idx['images'])} blocks, {rec.num_reg_images()} images, {rec.num_points3D()} points "
              f"({time.time() - t:.0f} s)")
        if idx.get("duplicates"):
            log.info("  images also in an earlier block (kept, counted twice): "
                  + ", ".join(f"{k} {v}" for k, v in idx["duplicates"].items()))
        n_adj = 1 + (len(terms) * 5 * max(1, int(thermal_rounds)) + 1 if refit_thermal else 0) \
            + (((max(1, int(early_rounds)) * 16 + 3) if not early_hold else 1) if early_mission_sol else 0)
        log.info(f"{datetime.datetime.now():%H:%M} plan: about {n_adj} joint adjustments (start"
              + (f", thermal profiles {', '.join(terms)}" if refit_thermal else "")
              + ((", early-mission offsets held" if early_hold else f", early-mission offsets up to {early_rounds} rounds of 16")
                 if early_mission_sol else "")
              + ", final with covariance" + (", rig translation" if translation_scapes else "") + ")")
        log.info(f"{FORM_ID}: fx, fy per eye against temperature, NL and NR cx against temperature, one rig yaw with a sol "
              f"rate (start {tvals['yaw_rate']:+.2f} mdeg/1000 sol); k4 free; k1 per degC NL {tvals['NL_k1']:+.2e}, NR "
              f"{tvals['NR_k1']:+.2e} ({k1_source})")
    proj.settings["navcam_rig_yaw"] = "refine"            # one yaw for all: the drift carries the sol rate
    names = forms.NAVCAM_PARAMS
    for cid, key in ((1, "NL"), (2, "NR")):
        c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
        c = navcam_distortion_terms(c, "refine")
        proj.cameras[key] = c
        rec.cameras[cid].params = np.asarray(c["params"], float)
    xkw: Dict[str, Any] = _joint_kw(tvals, drift)
    ppm = _mean_ppm(tvals)
    t = time.time()
    ba = J.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=not refit_thermal, **xkw)
    profiles = {}
    if refit_thermal:
        if verbose:
            log.info(f"{datetime.datetime.now():%H:%M} consensus at the start values: cost {ba['final_cost']:.1f} "
                  f"({time.time() - t:.0f} s); thermal profiles: {', '.join(terms)}")
        history = []
        for rnd in range(max(1, int(thermal_rounds))):
            if verbose and thermal_rounds > 1:
                log.info(f"{datetime.datetime.now():%H:%M} thermal profiles, round {rnd + 1} of {thermal_rounds}")
            for term in terms:
                pr = profile_thermal(rec, proj, temps, tvals, t0, term, drift=drift, verbose=verbose)
                pr["round"] = rnd + 1
                profiles[term] = pr
                tvals = dict(tvals, **{term: float(pr["best"])})
            history.append({k: tvals[k] for k in terms})
        if len(history) > 1 and verbose:
            log.info("change in the last round: " + ", ".join(f"{k} {history[-1][k] - history[-2][k]:+.4f}"
                                                          for k in terms))
        ppm = _mean_ppm(tvals)
        xkw = _joint_kw(tvals, drift)
        t = time.time()
        ba = J.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True, **xkw)
    early = None
    if early_mission_sol:
        em0 = nl.get("early_mission") or {}
        if early_start is not None:
            x0 = [float(v) for v in early_start]
        elif em0:
            x0 = [float(em0.get(f"d{k}_px", 0.0)) for k in EARLY_TERMS]
        else:
            x0 = early_start_from_blocks(scapes_cfg, float(early_mission_sol))["start"]
        if verbose:
            log.info(f"{datetime.datetime.now():%H:%M} early-mission offsets before sol {early_mission_sol:g} (start "
                  + " ".join(f"{k} {v:+.3f}" for k, v in zip(EARLY_TERMS, x0)) + "):")
        if early_hold:
            n_e = len(J.early_keypoint_map(proj, float(early_mission_sol), np.zeros(4)).early_images)
            early = {"before_sol": float(early_mission_sol), "terms": list(EARLY_TERMS), "early_images": n_e,
                     "held": True, "converged": True, "offsets_px": dict(zip(EARLY_TERMS, map(float, x0))),
                     "sd_px": em0.get("sd_px"), "z": em0.get("z"), "p_value": em0.get("p_value"),
                     "p_value_all": em0.get("p_value_all"),
                     "applied_terms": em0.get("applied_terms") or list(EARLY_TERMS), "rounds": []}
        else:
            early = fit_early_offsets(rec, proj, temps, ppm, t0, xkw, float(early_mission_sol), start=x0,
                                      rounds=early_rounds, verbose=verbose)
        if early.get("offsets_px"):
            xb = [early["offsets_px"][k] for k in EARLY_TERMS]
            ba = J.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True,
                                extra_map=J.early_keypoint_map(proj, float(early_mission_sol), xb), **xkw)
    _emap = (J.early_keypoint_map(proj, float(early_mission_sol), [early["offsets_px"][k] for k in EARLY_TERMS])
             if early and early.get("offsets_px") else None)
    joint_tr = None
    if joint_translation:
        t = time.time()
        _, _, T_before = J.stereo_rig(rec)
        t_before = np.array(T_before.translation, float)
        ba = J.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True, refine_rig=True,
                            rig_translation_sigma_m=float(translation_sigma_m), extra_map=_emap, **xkw)
        _, _, T_after = J.stereo_rig(rec)
        joint_tr = {"t_start_m": t_before.tolist(), "t_m": np.asarray(T_after.translation, float).tolist(),
                    "baseline_start_m": float(np.linalg.norm(t_before)),
                    "baseline_m": float(np.linalg.norm(np.asarray(T_after.translation, float))),
                    "sigma_m": float(translation_sigma_m), "final_cost": float(ba["final_cost"]),
                    "iterations": ba["iterations"], "seconds": time.time() - t,
                    "rig_abs_mdeg": J.rig_angles(np.asarray(T_after.rotation.matrix())),
                    "NR_cx": float(rec.cameras[2].params[2]), "NL_cx": float(rec.cameras[1].params[2])}
        joint_tr["dbaseline_mm"] = 1e3 * (joint_tr["baseline_m"] - joint_tr["baseline_start_m"])
        if verbose:
            log.info(f"{datetime.datetime.now():%H:%M} joint rig: translation free with the cameras and the rotation: baseline "
                  f"{joint_tr['baseline_m']:.5f} m ({joint_tr['dbaseline_mm']:+.2f} mm), yaw "
                  f"{joint_tr['rig_abs_mdeg']['yaw_mdeg']:+.2f} mdeg, NR cx {joint_tr['NR_cx']:.3f} px, cost "
                  f"{ba['final_cost']:.1f} ({joint_tr['seconds']:.0f} s)")
        if degeneracy_mm:
            joint_tr["degeneracy"] = baseline_degeneracy(rec, proj, temps, ppm, t0, xkw, _emap, float(degeneracy_mm),
                                                         iterations=iterations, verbose=verbose)
    loo = None
    if leave_one_out:
        loo = leave_one_block_out(rec, proj, idx, temps, ppm, t0, xkw, extra_map=_emap, iterations=loo_iterations,
                                  verbose=verbose)
    ks = J.keypoint_scales(proj, temps, ppm, t0, axis_ppm=xkw.get("axis_ppm"))
    kmap = J.thermal_keypoint_map(proj, temps, t0, xkw["pp_slopes"], xkw.get("drift"))
    if xkw.get("k1_slopes"):
        _k1 = J.k1_keypoint_map(proj, temps, t0, xkw["k1_slopes"])
        _kb = kmap
        kmap = lambda iid, k, cam: _k1(iid, _kb(iid, k, cam), cam)          # noqa: E731
    if _emap is not None:
        _km = kmap
        kmap = lambda iid, k, cam: _emap(iid, _km(iid, k, cam), cam)        # noqa: E731
    st = radial_stats(rec, proj, ks, kmap)
    _, _, T = J.stereo_rig(rec)
    cov = ba.get("covariance") or {}
    sd = {}
    vf = float(cov.get("variance_factor", 1.0))
    blocks = cov.get("blocks") or {}
    for (cid, key) in ((1, "NL"), (2, "NR")):
        c = blocks.get(f"camera:{key}", blocks.get(f"camera:{cid}"))
        if c is not None:
            free = [n for n in names if n not in (proj.cameras[key].get("fixed_params") or [])]
            if len(free) != len(np.diag(np.asarray(c))):            # the thin-prism terms are held by the BA
                free = [n for n in free if n not in ("sx1", "sy1")]
            dg = np.sqrt(np.clip(np.diag(np.asarray(c)), 0, None) * vf)
            if len(dg) == len(free):
                sd[key] = dict(zip(free, map(float, dg)))
            else:
                sd[key] = {"diag": [float(x) for x in dg], "free_guess": free}
    out = {"final_cost": float(ba["final_cost"]), "iterations": ba["iterations"], "variance_factor": vf,
           "observations": ba.get("observations"), "brief": ba["brief"], "seconds": time.time() - t,
           "NL": dict(zip(names, map(float, rec.cameras[1].params))),
           "NR": dict(zip(names, map(float, rec.cameras[2].params))), "sd": sd,
           "covariance_blocks": sorted(blocks), "covariance_error": cov.get("error"),
           "covariance_damping": cov.get("damping"),
           "rig_abs_mdeg": J.rig_angles(np.asarray(T.rotation.matrix())),
           "rig_R": np.asarray(T.rotation.matrix()).tolist(), "rig_t": np.asarray(T.translation).tolist(),
           "stereo": J.stereo_offset(rec.cameras[1], rec.cameras[2], T), "residuals": st,
           "blocks": {k: len(v) for k, v in idx["images"].items()}, "start_dir": str(d),
           "thermal": {"ppm_per_degC": ppm, "T0_degC": t0,
                       "pp_slopes": {"NL": [tvals["NL_cx"], tvals["NL_cy"]], "NR": [tvals["NR_cx"], tvals["NR_cy"]]},
                       "axis_ppm": {k: list(v) for k, v in _axis_ppm(tvals).items()},
                       "yaw_rate_mdeg_per_1000sol": tvals["yaw_rate"],
                       "k1_slopes": {"NL": tvals["NL_k1"], "NR": tvals["NR_k1"]}, "k1_source": k1_source,
                       "refit": bool(refit_thermal), "terms": list(terms) if refit_thermal else [],
                       "rounds": int(thermal_rounds) if refit_thermal else 0,
                       "sd": {k: profiles[k]["sd"] for k in profiles},
                       "z": {k: profiles[k].get("z") for k in profiles}, "profiles": profiles},
           "drift": xkw.get("drift"), "form": FORM_ID, "convention": "NL reference",
           "early_mission": early, "leave_one_out": loo}
    if joint_tr is not None:
        out["rig_translation_joint"] = joint_tr
    if translation_scapes:
        tr = fit_rig_translation(translation_scapes, out, points=points, sigma_m=translation_sigma_m,
                                 iterations=iterations, verbose=verbose)
        out["rig_translation"] = tr
        if joint_tr is not None and verbose:
            log.info(f"rig translation: joint {joint_tr['baseline_m']:.5f} m vs large blocks alone (cameras held) "
                  f"{tr['baseline_m']:.5f} m: {1e3 * (joint_tr['baseline_m'] - tr['baseline_m']):+.2f} mm")
    if translation_scapes or joint_tr is not None:
        out["rig_translation_applied"] = bool(apply_translation)
        out["rig_translation_source"] = "joint" if joint_tr is not None else "large blocks (cameras held)"
        if apply_translation and joint_tr is None:
            out["rig_t"] = list(tr["t_m"])
    if verbose:
        log.info(f"consensus: cost {out['final_cost']:.1f}, it {out['iterations']}, median {st['median_px']:.4f} / rms "
              f"{st['rms_px']:.4f} px, corners {st['radial'][-1]['median_px'] or float('nan'):.3f} px, rig yaw "
              f"{out['rig_abs_mdeg']['yaw_mdeg']:+.2f} mdeg  ({out['seconds']:.0f} s)")
    return out


def write_navcam(res: Dict[str, Any], out_dir: PathLike, note: str = "",
                 template_dir: Optional[PathLike] = None) -> Path:
    """The candidate folder of a :func:`fit_navcam` result: the camera and rig files in use (``template_dir``,
    default the models in use) with the new parameters, thermal terms, early-mission offsets, rig rotation and
    notes; sd from the joint's covariance where it has them."""
    from scipy.spatial.transform import Rotation
    tdir = M._folder(template_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = sorted(res["blocks"])
    stamp = datetime.date.today().isoformat()
    a = res["rig_abs_mdeg"]
    rt = res["thermal"]
    tv = _thermal_values(rt)
    for eye in ("NL", "NR"):
        c = M._read(M.camera_file("N", eye[1], tdir))
        c["params"] = [float(res[eye][n]) for n in c["param_names"]]
        c["fixed_params"] = []
        k1txt = f", k1 {tv[eye + '_k1']:+.2e}/degC" if tv.get(eye + "_k1") else ""
        c["source"] = (f"MPPP Navcam consensus (mppp.calib.fit.fit_navcam, {stamp}): one {eye} camera shared by "
                       f"{len(blocks)} blocks ({', '.join(blocks)}); fisheye + tangential at {rt['T0_degC']:g} degC, "
                       f"fx {tv[eye + '_fx']:+.1f} / fy {tv[eye + '_fy']:+.1f} ppm/degC, cx {tv[eye + '_cx']:+.4f} "
                       f"px/degC{k1txt}, k4 {float(res[eye]['k4']):+.5f} (fitted with k3), one rig yaw "
                       f"{a['yaw_mdeg']:+.2f} mdeg (rate {tv['yaw_rate']:+.2f} mdeg/1000 sol). {note}".strip())
        sdt = rt.get("sd") or {}
        th0 = {"T0_degC": float(rt["T0_degC"]),
               "fx_ppm_per_degC": tv[f"{eye}_fx"], "fy_ppm_per_degC": tv[f"{eye}_fy"],
               "cx_px_per_degC": tv[f"{eye}_cx"], "cy_px_per_degC": tv[f"{eye}_cy"],
               "k1_per_degC": float(tv.get(f"{eye}_k1") or 0.0), "k1_source": rt.get("k1_source"),
               "note": f"refitted with the consensus ({stamp}; profiles of {', '.join(rt.get('terms') or []) or 'none'})"}
        for k, kk in ((f"{eye}_cx", "sd_cx_px_per_degC"), (f"{eye}_cy", "sd_cy_px_per_degC"),
                      (f"{eye}_fx", "sd_fx_ppm_per_degC"), (f"{eye}_fy", "sd_fy_ppm_per_degC")):
            if k in sdt:
                th0[kk] = float(sdt[k])
        c["thermal"] = th0
        c["convention"] = "NL reference"
        c["form"] = FORM_ID
        em = res.get("early_mission")
        if em and em.get("offsets_px"):
            c["early_mission"] = {"before_sol": em["before_sol"],
                                  **{f"d{k}_px": float(em["offsets_px"][k]) for k in EARLY_TERMS},
                                  "sd_px": em.get("sd_px"), "z": em.get("z"), "p_value": em.get("p_value"),
                                  "p_value_all": em.get("p_value_all"), "applied_terms": em.get("applied_terms", []),
                                  "early_images": em.get("early_images"),
                                  "note": f"({stamp}) a camera with these offsets for the images before sol "
                                          f"{em['before_sol']:g}; projects apply the terms in applied_terms "
                                          f"(|z| >= {EARLY_APPLY_Z:g}), weighted by the block's share of early images"}
        elif "early_mission" in c and em is not None:
            c.pop("early_mission", None)
        sde = (res.get("sd") or {}).get(eye) or {}
        if sde and "diag" not in sde:
            c["sd"] = {k: v for k, v in {**(c.get("sd") or {}), **sde}.items()}
            c.pop("covariance", None)
            c["sd_source"] = "this joint (formal x variance factor)"
        else:
            c["sd_source"] = "copied from the cameras in use (not refitted)"
        lo = res.get("leave_one_out")
        c["verification"] = {"note": ("leave-one-block-out: max centre {:.3f} px, corners {:.3f} px over {} blocks ({})".format(
                                 lo["max"]["centre_px"], lo["max"]["corner_px"], len(lo["blocks"]), "ok" if lo["ok"] else "NOT ok")
                                 if lo and lo.get("max", {}).get("centre_px") is not None
                                 else "not verified (no leave-one-block-out run)"),
                             "leave_one_out": ({k: lo[k] for k in ("max", "worst", "limits", "ok")} if lo else None),
                             "residuals": {k: res["residuals"][k] for k in ("median_px", "rms_px", "p95_px")}}
        M.write_model(out / M.CAMERA_FILE[("N", eye[1])], c)
    r = M._read(M.rig_file("N", tdir))
    R = np.asarray(res["rig_R"], float)
    r["R_sensor_from_ref"] = R.tolist()
    r["rotvec_rad"] = Rotation.from_matrix(R).as_rotvec().tolist()
    if res.get("drift"):
        r["drift"] = res["drift"]
    r["yaw_constant"] = True
    r.pop("thermal", None)
    r["convention"] = "NL reference"
    r["form"] = FORM_ID
    yr = tv["yaw_rate"]
    sdy = (rt.get("sd") or {}).get("yaw_rate")
    r["yaw_model"] = (f"one yaw for every temperature (the stereo temperature change is the principal points'); yaw "
                      f"rate {yr:+.2f}" + (f" +- {sdy:.2f}" if sdy else "") + " mdeg per 1000 sol")
    tr = res.get("rig_translation")
    jt = res.get("rig_translation_joint")
    applied = bool((tr or jt) and res.get("rig_translation_applied"))
    carried = not applied and r.get("translation") == "fitted"     # a fit only recorded keeps the one in use
    if tr:
        r["translation_fit"] = {k: tr[k] for k in ("blocks", "dcentre_mm", "sd_t_mm", "baseline_start_m", "baseline_m",
                                                   "dbaseline_mm", "t_m", "variance_factor", "sigma_m")}
    if jt:
        r["translation_joint"] = {k: jt.get(k) for k in ("t_start_m", "t_m", "baseline_start_m", "baseline_m",
                                                         "dbaseline_mm", "sigma_m", "final_cost", "NR_cx", "rig_abs_mdeg")}
        dg = jt.get("degeneracy") or {}
        if dg:
            rise = dg["cost_rise_mean"] - float(jt["final_cost"])
            vf = float(res.get("variance_factor") or 1.0)
            r["translation_joint"]["degeneracy"] = {
                "delta_mm": dg["delta_mm"], "yaw_mdeg_per_mm": dg["yaw_mdeg_per_mm"],
                "pitch_mdeg_per_mm": dg["pitch_mdeg_per_mm"], "NR_cx_px_per_mm": dg["NR_cx_px_per_mm"],
                "NL_cx_px_per_mm": dg["NL_cx_px_per_mm"], "cost_rise": rise,
                "sd_baseline_mm": (dg["delta_mm"] / np.sqrt(2.0 * rise / vf)) if rise > 0 else None}
    if applied:
        src = jt if jt else tr
        r["t_sensor_from_ref_m"] = [float(x) for x in src["t_m"]]
        r["baseline_m"] = float(src["baseline_m"])
        r["translation"] = "fitted"
        r["translation_source"] = "joint with the cameras and the rotation" if jt else "large blocks, cameras held"
    elif not carried:
        r.pop("translation", None)
    r["note"] = (f"x_NR = R x_NL + t. Navcam consensus {stamp} ({len(blocks)} blocks); yaw {a['yaw_mdeg']:+.2f}, "
                 f"pitch {a['pitch_mdeg']:+.2f}, roll {a['roll_mdeg']:+.2f} mdeg. One yaw for every block: no "
                 "temperature term (the principal points carry it), the yaw follows its sol rate; pitch and roll "
                 "follow the mission drift. "
                 + ((f"Translation fitted jointly with the cameras and the rotation over every block: baseline "
                     f"{jt['baseline_m']:.5f} m" + (f" (large blocks alone, cameras held: {tr['baseline_m']:.5f} m)" if tr else "")
                     + "; projects use it (translation = fitted).") if (applied and jt) else
                    f"Translation fitted on {len(tr['blocks'])} large blocks ({', '.join(sorted(tr['blocks']))}): "
                    f"baseline {tr['baseline_m']:.5f} m; projects use it (translation = fitted)." if applied else
                    (f"Translation fitted earlier, carried over from the cameras in use (baseline "
                     f"{float(r.get('baseline_m', 0)):.5f} m; translation = fitted)"
                     + (" - the new fit, translation_fit, is recorded but not applied." if tr else ".")
                     if carried else
                     "Translation from CAHV in each project"
                     + (" (the large-block fit, translation_fit, is recorded but not applied)." if tr else "."))))
    M.write_model(out / M.RIG_FILE["N"], r)
    (out / "consensus.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return out


def check_navcam(candidate_dir: PathLike, current_dir: Optional[PathLike] = None,
                 limits: Optional[Dict[str, float]] = None, screening: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Is a Navcam candidate folder (:func:`write_navcam`) fit to replace the models in use (``current_dir``)?
    Checks (``limits``, :data:`PROMOTE_LIMITS`): the files load and follow the form; at least ``min_blocks``
    blocks; the adjustment converged with finite sd; residuals at most ``residual_factor`` x those of the models in
    use; each eye within ``max_camera_change_px`` of the camera in use (rotation removed), the rig within
    ``max_rig_change_mdeg``; the thermal slopes within their limits of the ones in use; the early-mission fit
    converged on a convex surface; the leave-one-block-out within :data:`LOO_LIMITS`; no block that failed
    ``screening``; the rig baseline in range and determined.  Returns {"ok", "checks": [...], "early_mission",
    "candidate", "current"}.
    """
    lim = dict(PROMOTE_LIMITS, **(limits or {}))
    cand, cur = Path(candidate_dir), M._folder(current_dir)
    checks = []

    def add(name, value, limit, ok, note=""):
        checks.append({"check": name, "value": value, "limit": limit, "ok": bool(ok), "note": note})

    files = list(M.NAVCAM_FILES) + ["consensus.json"]
    missing = [f for f in files if not (cand / f).is_file()]
    add("files", len(files) - len(missing), len(files), not missing, ", ".join(missing) or "all present")
    if missing:
        return {"ok": False, "checks": checks, "candidate": str(cand), "current": str(cur)}
    res = M._read(cand / "consensus.json")
    new = {e: M._read(M.camera_file("N", e[1], cand)) for e in ("NL", "NR")}
    old = {e: M._read(M.camera_file("N", e[1], cur)) for e in ("NL", "NR")}
    rn, ro = M._read(M.rig_file("N", cand)), M._read(M.rig_file("N", cur))
    nb = len(res.get("blocks") or {})
    add("blocks", nb, lim["min_blocks"], nb >= lim["min_blocks"])
    sd = res.get("sd") or {}
    finite = all(isinstance(sd.get(e), dict) and "diag" not in sd[e] and all(np.isfinite(list(sd[e].values())))
                 for e in ("NL", "NR"))
    _damp = res.get("covariance_damping")
    add("covariance", "finite" if finite else "missing", "finite sd of both eyes", finite,
        str(res.get("covariance_error") or "") + (f"damped {float(_damp):g} (sd of near-null directions slightly low)"
                                                  if _damp and float(_damp) > 1e-8 else ""))
    it = int(res.get("iterations") or 0)
    add("converged", it, "< 100 iterations", 0 < it < 100 and np.isfinite(res.get("final_cost", np.nan)))
    ref = (old["NL"].get("verification") or {}).get("residuals") or {}
    for k in ("median_px", "rms_px", "p95_px"):
        v = float(res["residuals"][k])
        r = ref.get(k)
        add(f"residual {k}", round(v, 4), None if r is None else round(lim["residual_factor"] * float(r), 4),
            r is None or v <= lim["residual_factor"] * float(r), f"cameras in use {r:.4f}" if r else "no reference")
    for e in ("NL", "NR"):
        d = J.compare(J.pycolmap_camera(old[e]["model"], old[e]["params"]), J.pycolmap_camera(new[e]["model"], new[e]["params"]))
        add(f"{e} change vs in use", round(float(d["rms_px"]), 3), lim["max_camera_change_px"],
            d["rms_px"] <= lim["max_camera_change_px"], f"corners {d.get('corner_rms_px', float('nan')):.3f} px (rotation removed)")
    # the files' rotations are at their own sol0 (932 for the profile joint's, 0 for the tied joint's): compare the
    # rigs at one sol in the middle of the mission
    Ro, _, _ = rig_at(ro, sol=1000.0)
    Rn, _, _ = rig_at(rn, sol=1000.0)
    dr = J.rotation_mdeg(Ro, Rn)
    add("rig rotation change", round(dr, 2), lim["max_rig_change_mdeg"], dr <= lim["max_rig_change_mdeg"], "mdeg at sol 1000")
    tt = {(e, w): thermal_terms(m[e].get("thermal")) for e in ("NL", "NR") for w, m in (("new", new), ("old", old))}
    dppm = max(abs(tt[(e, "new")][a] - tt[(e, "old")][a]) for e in ("NL", "NR") for a in ("fx_ppm", "fy_ppm"))
    tied = res.get("method") == "tied"
    # the tied joint estimates the slopes over all blocks in one adjustment; the files in use carry the 0.86 binned
    # estimate, so the change is reported, not gated (20 Oct: 10.9 ppm/degC, the joint's sd 0.5-1.3)
    add("thermal f slope change", round(dppm, 2), lim["max_ppm_change"], tied or dppm <= lim["max_ppm_change"],
        ("report only (tied joint); " if tied else "") + "ppm/degC, largest of fx, fy per eye: " + ", ".join(f"{e} {tt[(e, 'new')]['fx_ppm']:+.1f}/{tt[(e, 'new')]['fy_ppm']:+.1f}"
                                                            for e in ("NL", "NR")))

    def stereo(m, k):           # the stereo temperature change: NR's principal-point slope minus NL's (x fR/fL)
        ratio = float(m["NR"]["params"][0]) / float(m["NL"]["params"][0])
        return thermal_terms(m["NR"].get("thermal"))[k] - ratio * thermal_terms(m["NL"].get("thermal"))[k]
    dpp = max(abs(stereo(new, k) - stereo(old, k)) for k in ("cx_px", "cy_px"))
    add("stereo pp thermal slope change", round(dpp, 4), lim["max_pp_slope_change"], dpp <= lim["max_pp_slope_change"],
        f"px/degC, NR - NL x fR/fL: cx {stereo(new, 'cx_px'):+.4f} (in use {stereo(old, 'cx_px'):+.4f}), "
        f"cy {stereo(new, 'cy_px'):+.4f} (in use {stereo(old, 'cy_px'):+.4f})")
    fm = M.check_form(cand)
    add("form", FORM_ID if fm["ok"] else "not in the form", "THIN_PRISM_FISHEYE, sx1 = sy1 = 0, NL reference, no rig "
        "temperature term", fm["ok"], "; ".join(fm["errors"] + fm["warnings"]) or "ok")
    lo = res.get("leave_one_out")
    ag = res.get("block_agreement")
    if res.get("method") == "tied" and (cand / "tied_joint.json").is_file():
        try:                     # re-judged with the limits of today (the candidate may predate them)
            from .tied import block_agreement
            ag = block_agreement(M._read(cand / "tied_joint.json")) or ag
        except Exception as e:                                        # noqa: BLE001
            log.warning("block agreement from tied_joint.json: %s", e)
    if res.get("method") == "tied":
        # the tied joint (0.91): the blocks' agreement with the one solution in the diagnostic pass is the gate
        if ag:
            add("block agreement", f"cameras {ag['cameras_off']}/{ag['cameras']}, rigs {ag['rigs_off']}/{ag['rigs']} off",
                f"<= {ag['limits']['max_share']:.0%} off (fx {ag['limits']['f_ppm']:g} ppm, fy "
                f"{ag['limits'].get('fy_ppm', ag['limits']['f_ppm']):g} ppm, rig {ag['limits']['rig_mdeg']:g} mdeg)",
                bool(ag.get("ok")), f"cameras {ag['share_cameras_off']:.0%}, rigs {ag['share_rigs_off']:.0%} off; "
                f"worst camera {ag['worst_camera_ppm']:.0f} ppm, worst rig {ag['worst_rig_mdeg']:.1f} mdeg"
                + (f"; blocks off: {', '.join(ag['blocks_off'])}" if ag.get("blocks_off") else ""))
        else:
            add("block agreement", "not run", "required", False, "run the tied joint with diagnostics")
    elif lo and (lo.get("max") or {}).get("centre_px") is not None:
        mx = lo["max"]
        add("leave-one-block-out", f"centre {mx['centre_px']:.3f} / corners {mx['corner_px']:.3f} px",
            f"{LOO_LIMITS['centre_px']} / {LOO_LIMITS['corner_px']} px", bool(lo.get("ok")),
            f"worst: centre {lo.get('worst', {}).get('centre_px')}, corners {lo.get('worst', {}).get('corner_px')}; "
            f"stereo pointing up to {mx.get('stereo_px') or 0.0:.3f} px" + (f"; failed: {', '.join(lo['errors'])}"
                                                                              if lo.get("errors") else ""))
    else:
        add("leave-one-block-out", "not run", "required", False, "run the consensus with leave_one_out")
    yr = (res.get("thermal") or {}).get("yaw_rate_mdeg_per_1000sol")
    if yr is not None:
        sdy = ((res.get("thermal") or {}).get("sd") or {}).get("yaw_rate")
        add("rig yaw rate", round(float(yr), 3), "measured", True,
            "mdeg per 1000 sol" + (f", sd {sdy:.3f} (z {float(yr) / sdy:+.1f})" if sdy else ""))
    em = res.get("early_mission")
    if em and em.get("held"):
        add("early-mission fit", "held", "converged, convex, finite sd", True,
            "held at the start's offsets: " + ", ".join(f"{k} {v:+.3f}" for k, v in em["offsets_px"].items()))
    elif em:
        rounds = em.get("rounds") or []
        convex = bool(rounds) and bool(rounds[-1].get("positive_definite"))
        sds = list((em.get("sd_px") or {}).values())
        ok = bool(em.get("converged")) and convex and sds and all(np.isfinite(sds))
        add("early-mission fit", "converged" if em.get("converged") else "not converged", "converged, convex, finite sd",
            ok, "applied terms: " + (", ".join(em.get("applied_terms") or []) or "none")
            + "; " + ", ".join(f"{k} {em['offsets_px'][k]:+.3f} +- {em['sd_px'][k]:.3f} (z {em['z'][k]:+.1f})"
                               for k in (em.get("sd_px") or {})))
    if screening:
        bad = sorted(k for k, r in screening.items() if r.get("verdict") == "fail" and k in (res.get("blocks") or {}))
        add("screening", len(bad), 0, not bad, ", ".join(bad) or "no failed block in the joint")
    jt, tr = res.get("rig_translation_joint"), res.get("rig_translation")
    if res.get("rig_translation_applied") and (jt or tr):
        src = jt or tr
        b = float(src["baseline_m"])
        add("rig baseline", round(b, 5), lim["baseline_m"], lim["baseline_m"][0] <= b <= lim["baseline_m"][1],
            f"{'joint' if jt else 'large blocks'}: {src.get('dbaseline_mm', 0.0):+.2f} mm from the start")
        if jt and tr:
            d = 1e3 * abs(float(jt["baseline_m"]) - float(tr["baseline_m"]))
            add("rig baseline joint vs large blocks", round(d, 2), lim["max_baseline_disagreement_mm"],
                d <= lim["max_baseline_disagreement_mm"], "mm")
        dg = (jt or {}).get("degeneracy")
        if dg:
            rise = dg["cost_rise_mean"] - float(jt["final_cost"])
            vf = float(res.get("variance_factor") or 1.0)
            sdb = (dg["delta_mm"] / np.sqrt(2.0 * rise / vf)) if rise > 0 else float("inf")
            add("rig baseline determined", round(sdb, 3), lim["max_baseline_sd_mm"], sdb <= lim["max_baseline_sd_mm"],
                f"sd mm from the +-{dg['delta_mm']:g} mm profile; per mm: yaw {dg['yaw_mdeg_per_mm']:+.2f} mdeg, "
                f"NR cx {dg['NR_cx_px_per_mm']:+.3f} px, pitch {dg['pitch_mdeg_per_mm']:+.2f} mdeg")
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "early_mission": em, "candidate": str(cand),
            "current": str(cur)}


def check_lines(rep: Dict[str, Any]) -> List[str]:
    """Printable lines of a :func:`check_navcam` report."""
    out = [f"candidate {rep['candidate']} against the cameras in use {rep['current']}:"]
    for c in rep["checks"]:
        lim = "" if c["limit"] is None else f"  (limit {c['limit']})"
        out.append(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']:28s} {c['value']}{lim}  {c['note']}")
    out.append("=> " + ("everything checks out" if rep["ok"] else "NOT ready to promote"))
    return out


def promote_navcam(candidate_dir: PathLike, note: str, folder: Optional[PathLike] = None,
                   check: bool = True) -> Dict[str, Any]:
    """Promote a Navcam candidate folder (:func:`check_navcam` first unless ``check`` is False) with
    :func:`mppp.calib.models.promote`."""
    if check:
        rep = check_navcam(candidate_dir, folder)
        if not rep["ok"]:
            raise ValueError("candidate not ready: " + "; ".join(c["check"] for c in rep["checks"] if not c["ok"]))
    cand = Path(candidate_dir)
    return M.promote([cand / n for n in M.NAVCAM_FILES], note=note, folder=folder)


# ====================================================================================== Mastcam-Z calibration
ZCAM_MODEL_LIMITS = {"max_f0_change_pct": 1.5, "max_aspect_dev": 0.01, "max_k1": 1.5, "baseline_m": (0.20, 0.29),
                     "max_rig_change_deg": 1.0, "max_rig_scatter_mdeg": 150.0, "max_baseline_scatter_mm": 3.0,
                     "max_tau_f_px": 10.0, "min_blocks": 3,
                     "max_p": forms.ZCAM_MAX_P, "max_tau_p": 1e-3,     # zcam-v2: |p1|, |p2| and their block scatter
                     "max_offset_scatter_pct": 0.5,     # a start update: the blocks' f offsets about the model's f0
                     "min_blocks_start": 2}
REGULAR_RATIO_DEFAULT = 1.0095          # backlash / regular focal length when the blocks do not measure it


def block_info(work: PathLike, model: str = "cahv_ba") -> Dict[str, Any]:
    """Everything the Mastcam-Z fit needs of one solved Navcam + Mastcam-Z block (no points are read): its
    focus-bin cameras (refined and start), the Navcam focal scale, the lens temperature, the focus lines and the
    rig of the block's settings, its backlash classification.  Raises FileNotFoundError without a solution."""
    def _json(p: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    w = Path(work)
    root = w / "colmap" if (w / "colmap" / "project.json").is_file() else w
    proj = _json(root / "project.json")
    cb = root / "sparse" / model / "cameras.bin"
    if proj is None or not cb.is_file():
        raise FileNotFoundError(f"no solved project in {root}")
    st = proj.get("settings") or {}
    db = (st.get("database") or {}).get("cameras") or {}
    refined = read_cameras_bin(cb)
    bins, nav = [], []
    for key, cam in (proj.get("cameras") or {}).items():
        cid = db.get(key)
        p = refined.get(int(cid)) if cid is not None else None
        if p is None:
            continue
        if str(key).startswith("N") and cam.get("params"):
            p0 = np.asarray(cam["params"], float)
            nav.append((float(np.sqrt(p[0] * p[1]) / np.sqrt(p0[0] * p0[1])), int(cam.get("n_images") or 1)))
        if str(key).startswith("Z") and cam.get("group"):
            bins.append({"camera": key, "group": cam["group"], "focus": cam.get("focus_count_median"),
                         "state": "regular" if (cam.get("backlash_state") == "regular" or key.endswith("_reg")) else "backlash",
                         "params": [float(x) for x in p], "start": [float(x) for x in cam["params"]],
                         "fixed": list(cam.get("fixed_params") or []), "n_images": int(cam.get("n_images") or 1)})
    ns = (sum(r * w_ for r, w_ in nav) / sum(w_ for _, w_ in nav)) if nav else None
    imgs = proj.get("images") or []
    To = [float(r["zcam_optics_temperature_degC"]) for r in imgs
          if str(r.get("instrument", ""))[:1] == "Z" and r.get("zcam_optics_temperature_degC") is not None]
    Tf = [float(r["camera_temperature_degC"]) for r in imgs
          if str(r.get("instrument", ""))[:1] == "Z" and r.get("camera_temperature_degC") is not None]
    sols = [float(r["sol"]) for r in imgs if str(r.get("instrument", ""))[:1] == "Z" and r.get("sol") is not None]
    health = _json(root / "health" / "health.json") or {}
    checks = {c.get("check"): c for c in health.get("checks") or [] if isinstance(c, dict)}
    statuses: Dict[str, List[str]] = {}
    for c in health.get("checks") or []:
        if isinstance(c, dict):
            statuses.setdefault(c.get("check"), []).append(c.get("status"))
    states = _json(root / "zcam_focus_states.json") or {}
    return {"work": str(w), "root": str(root), "name": w.name, "run_done": _json(root / "run_done.json"),
            "health_verdict": health.get("verdict"), "checks": checks, "statuses": statuses, "bins": bins,
            "navcam_scale": ns, "T_optics_degC": float(np.mean(To)) if To else None,
            "T_fpa_degC": float(np.mean(Tf)) if Tf else None, "sol": float(np.median(sols)) if sols else None,
            "lines": st.get("zcam_focus_lines") or {}, "rig": st.get("zcam_virtual_rig") or {},
            "rig_fit": st.get("zcam_rig_fit"), "backlash_groups": states.get("groups") or [],
            "backlash_fits": states.get("fits_before") or {}, "backlash_nav_scale": states.get("nav_scale"),
            "backlash_ratios": states.get("backlash_ratios") or {},
            "line_params": st.get("zcam_line_params"), "mppp": (_json(root / "run_done.json") or {}).get("mppp")}


def pp_shift_to_rig(R_sensor_from_ref: np.ndarray, t_sensor_from_ref: np.ndarray, f_px: float,
                    dcx_px: float, dcy_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """The rig rotation that replaces a principal-point offset of the right (sensor) eye: a camera whose principal
    point is moved by (dcx, dcy) px sees the same pixels as the camera with the original principal point rotated
    about its own centre by (wx, wy) = (-dcy, dcx) / f.  Returns (R', t') with R' = R_d R, t' = R_d t."""
    from scipy.spatial.transform import Rotation
    Rd = Rotation.from_rotvec([-dcy_px / f_px, dcx_px / f_px, 0.0]).as_matrix()
    return Rd @ np.asarray(R_sensor_from_ref, float), Rd @ np.asarray(t_sensor_from_ref, float)


def _label_pp_at(info: Dict[str, Any], group: str, ref: float) -> Optional[Tuple[float, float]]:
    """The eye's start (label) principal point at the reference focus: a line through the bins' start values."""
    rows = [(b["focus"], b["start"][2], b["start"][3]) for b in info["bins"]
            if b["group"] == group and b["focus"] is not None and len(b["start"]) >= 4]
    if not rows:
        return None
    F = np.array([r[0] for r in rows], float) - ref
    out = []
    for k in (1, 2):
        y = np.array([r[k] for r in rows], float)
        if len(rows) >= 3 and np.ptp(F) > 0:
            out.append(float(np.polyval(np.polyfit(F, y, 1), 0.0)))
        else:
            out.append(float(np.median(y)))
    return out[0], out[1]


def regular_ratio(infos: Sequence[Dict[str, Any]], group: str, min_regular_groups: int = 3) -> Dict[str, Any]:
    """The ratio of the backlash to the regular focal length of an eye, measured by the blocks' focus-state
    classification (``zcam_focus_states.json``): the median over the blocks with at least ``min_regular_groups``
    regular and 3 backlash groups of (backlash ratio at the group's focus / regular ratio); else
    :data:`REGULAR_RATIO_DEFAULT`."""
    from ..sfm.backlash import backlash_ratio_at, measured_backlash_lines
    per = []
    for i in infos:
        gs = [g for g in (i.get("backlash_groups") or []) if g.get("camera_group") == group and g.get("ratio") is not None
              and not g.get("implausible")]
        b = [g for g in gs if g.get("state") == "backlash" and g.get("decided")]
        r = [g for g in gs if g.get("state") == "regular" and g.get("decided")]
        if len(r) >= min_regular_groups and len(b) >= 3:
            rows = [dict(g, ratio_fit=g.get("ratio_fit", g["ratio"]), fitted_unlisted=g.get("fitted_unlisted", g.get("fitted")))
                    for g in b]
            ln = measured_backlash_lines(rows, float(i.get("backlash_nav_scale") or 1.0)).get(group)
            q = [backlash_ratio_at(ln, g.get("focus_count")) / float(g["ratio"]) for g in r] if ln else \
                [float(np.median([g["ratio"] for g in b])) / float(g["ratio"]) for g in r]
            per.append({"block": i["name"], "ratio": float(np.median(q)), "regular": len(r), "backlash": len(b),
                        "line": (ln or {}).get("source")})
    if per:
        return {"ratio": float(np.median([p["ratio"] for p in per])), "n_blocks": len(per), "per_block": per,
                "source": "measured (backlash / regular groups)"}
    return {"ratio": REGULAR_RATIO_DEFAULT, "n_blocks": 0, "per_block": [], "source": "default"}


def zcam_leave_one_out(infos: Sequence[Dict[str, Any]], params: Sequence[str] = ("fx", "fy")) -> Dict[str, Any]:
    """Each block left out in turn: its backlash-state offset a of each eye and parameter against the weighted mean
    of the other blocks.  Returns {group: {param: {"rms_px", "max_px", "n", "per_block"}}}."""
    out: Dict[str, Any] = {}
    groups = sorted({g for i in infos for g in (i.get("lines") or {})})
    for g in groups:
        for pn in params:
            rows = []
            for i in infos:
                e = ((((i.get("lines") or {}).get(g) or {}).get("lines") or {}).get("backlash") or {}).get(pn)
                if isinstance(e, dict) and "a" in e:
                    k = (i.get("navcam_scale") or 1.0)
                    rows.append((i["name"], float(e["a"]) / k, float(e.get("sd_a") or np.nan) / k))
            if len(rows) < 3:
                continue
            per = []
            for j, (name, a, _) in enumerate(rows):
                others = [r for k_, r in enumerate(rows) if k_ != j]
                m = weighted_mean(np.array([r[1] for r in others]), np.array([r[2] for r in others], float))["value"]
                per.append({"block": name, "a": a, "predicted": m, "residual_px": a - m})
            res = np.array([p["residual_px"] for p in per])
            out.setdefault(g, {})[pn] = {"rms_px": float(np.sqrt(np.mean(res ** 2))), "max_px": float(np.max(np.abs(res))),
                                         "n": len(per), "per_block": per}
    return out


def _baseline_vs_cahv(rigs: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The blocks' refined Mastcam-Z baselines against the CAHV baselines held in the runs (mm): mean and sd of the
    difference, and its trends with sol and lens temperature (Christian, 20 Oct)."""
    d = [(float(np.linalg.norm(r["t"]) * 1e3) - r["baseline_cahv_mm"], r.get("sol"), r.get("T_optics_degC"))
         for r in rigs if r.get("baseline_cahv_mm") is not None]
    if not d:
        return None
    y = np.array([x[0] for x in d]); S = np.array([np.nan if x[1] is None else x[1] for x in d], float)
    T = np.array([np.nan if x[2] is None else x[2] for x in d], float)
    return {"n": len(d), "mean_mm": float(y.mean()), "sd_mm": float(y.std(ddof=1)) if len(y) > 1 else None,
            "vs_sol_per_1000": ols(S / 1000.0, y), "vs_T": ols(T, y)}


def zcam_without_tangential(model: Dict[str, Any]) -> Dict[str, Any]:
    """The same focus model with p1, p2 dropped (zcam-v1's form) - for the with / without comparison."""
    out = copy.deepcopy(model)
    for g, gm in (out.get("cameras") or {}).items():
        dist = gm.get("distortion") or {}
        names = list(dist.get("names") or [])
        keep = [i for i, n in enumerate(names) if n not in ("p1", "p2")]
        if len(keep) != len(names):
            gm["distortion"] = {"names": [names[i] for i in keep], "params": [dist["params"][i] for i in keep],
                                "note": dist.get("note", "") + " (p1, p2 dropped for the comparison)"}
        for st, ln in (gm.get("lines") or {}).items():
            if isinstance(ln, dict):
                for pn in ("p1", "p2"):
                    ln.pop(pn, None)
    out["form"] = "zcam-v1"
    return out


def zcam_tangential_comparison(model: Dict[str, Any], focus: Optional[float] = None) -> Dict[str, Any]:
    """With and without p1, p2 (Christian, 20 Oct): per eye the fitted p1, p2 with their sd and block scatter and the
    image displacement between the two models over the frame (:func:`mppp.calib.joint.compare`: rms and corners,
    rotation removed) at the reference focus, backlash state."""
    from . import constrain as C
    out: Dict[str, Any] = {"eyes": {}}
    without = zcam_without_tangential(model)
    for g, gm in (model.get("cameras") or {}).items():
        dist = gm.get("distortion") or {}
        names = list(dist.get("names") or [])
        if "p1" not in names:
            out["eyes"][g] = {"p1": 0.0, "p2": 0.0, "note": "no p1, p2 in this calibration"}
            continue
        f = float(focus if focus is not None else gm.get("reference_focus", 600.0))
        _, pa, _ = C.zcam_camera_at(model, g, f)
        _, pb, _ = C.zcam_camera_at(without, g, f)
        w, h = forms.FULL_FRAME["Z"]
        d = J.compare(J.pycolmap_camera("FULL_OPENCV", pa, w, h), J.pycolmap_camera("FULL_OPENCV", pb, w, h))
        e: Dict[str, Any] = {"focus": f, "displacement_px": d}
        for pn in ("p1", "p2"):
            i = names.index(pn)
            e[pn] = float(dist["params"][i])
            e[f"sd_{pn}"] = (dist.get("sd") or [None] * len(names))[i]
            e[f"tau_{pn}"] = (dist.get("tau") or [None] * len(names))[i]
        out["eyes"][g] = e
    return out


def fit_zcam(infos: Sequence[Dict[str, Any]], model: Dict[str, Any], rig_translation: str = "cahv") -> Dict[str, Any]:
    """
    The Mastcam-Z calibration of one zoom from solved blocks (:func:`block_info`): per eye and parameter the
    backlash-state line offset a (at the reference focus) and slope b averaged over the blocks (weights 1 / sd^2
    with the between-block scatter tau; fx, fy divided by the block's Navcam scale), put on the form (fx and fy
    share one slope per eye, k1, k2 constant, ZR cx pinned at pp0, ZR cy at pp0 with its slope); the regular state
    derived (fx, fy = the backlash lines divided by the measured ratio, :func:`regular_ratio`); no temperature term.
    Before anything is averaged each block's ZR principal-point offset at the reference focus is moved into its rig
    rotation (:func:`pp_shift_to_rig`: the two are degenerate in a block's adjustment).  The rig: the mean rotation
    of the blocks' refined rigs, the translation the CAHV baseline (``rig_translation="cahv"``) or the refined mean.
    Returns the new focus model (``model`` with ``lines``, ``pp``, ``pp0_px``, ``distortion``, ``regular_ratio``,
    ``aspect``, ``rig`` updated) with ``calibration`` (every number, the blocks, the scatter, the leave-one-out).
    """
    from scipy.spatial.transform import Rotation
    out = copy.deepcopy(model)
    out.pop("file", None)
    out.pop("zoom_mm", None)
    cal: Dict[str, Any] = {"blocks": [i["name"] for i in infos], "when": datetime.date.today().isoformat(), "eyes": {}}
    shifts: Dict[str, Dict[str, Any]] = {}
    rig_adj: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    zoom_key = None
    for g, gm in (out.get("cameras") or {}).items():
        zoom_key = f"Z{str(g)[2:5]}"
        if not str(g).startswith("ZR"):
            if not gm.get("pp0_px"):                  # the form needs every eye's pp0 (ZL: the label pp at the reference)
                ref_l = float(gm.get("reference_focus", 600.0))
                lab = [p for p in (_label_pp_at(i, g, ref_l) for i in infos) if p is not None]
                if lab:
                    gm["pp0_px"] = [float(np.median([p[0] for p in lab])), float(np.median([p[1] for p in lab]))]
            continue
        ref0 = None
        for i in infos:
            ref0 = ((i["lines"].get(g) or {}).get("reference_focus")) or ref0
        ref0 = float(ref0 if ref0 is not None else gm.get("reference_focus", 600.0))
        pp0 = gm.get("pp0_px")
        if not pp0:
            lab = [p for p in (_label_pp_at(i, g, ref0) for i in infos) if p is not None]
            pp0 = [float(np.median([p[0] for p in lab])), float(np.median([p[1] for p in lab]))] if lab else None
        if not pp0:
            continue
        gm["pp0_px"] = [float(pp0[0]), float(pp0[1])]
        for i in infos:
            bk = (((i["lines"].get(g) or {}).get("lines") or {}).get("backlash") or {})
            has = {pn: isinstance(bk.get(pn), dict) and "a" in bk[pn] for pn in ("cx", "cy")}
            if not any(has.values()):
                continue
            d = (float(bk["cx"]["a"]) - float(pp0[0]) if has["cx"] else 0.0,
                 float(bk["cy"]["a"]) - float(pp0[1]) if has["cy"] else 0.0)
            f = float(bk["fx"]["a"]) if isinstance(bk.get("fx"), dict) and "a" in bk["fx"] else float(gm.get("f0_px") or 4720.0)
            shifts[i["name"]] = {"group": g, "dcx_px": d[0], "dcy_px": d[1], "f_px": f}
            v = (i.get("rig") or {}).get(zoom_key) or {}
            if v.get("R_refined") is not None and v.get("t_refined") is not None:
                rig_adj[i["name"]] = pp_shift_to_rig(np.asarray(v["R_refined"], float), np.asarray(v["t_refined"], float),
                                                     f, d[0], d[1])
    cal["pp_to_rig"] = {"pp0_px": {g: gm.get("pp0_px") for g, gm in (out.get("cameras") or {}).items() if str(g).startswith("ZR")},
                        "shifts": shifts,
                        "note": "ZR cx, cy at the reference focus put at pp0 in every block; the difference moved into "
                                "that block's rig rotation before the rigs were averaged"}
    for g, gm in (out.get("cameras") or {}).items():
        form = forms.zcam_line_form(g)
        pars: Dict[str, Any] = {}
        ref = None
        names = sorted({pn for i in infos for pn in (((i["lines"].get(g) or {}).get("lines") or {}).get("backlash") or {})
                        if pn in forms.FULL_OPENCV_PARAMS})
        for pn in names:
            rows = []
            for i in infos:
                L = (i["lines"].get(g) or {})
                e = ((L.get("lines") or {}).get("backlash") or {}).get(pn)
                if not isinstance(e, dict) or "a" not in e:
                    continue
                ref = L.get("reference_focus", ref)
                k = (i["navcam_scale"] or 1.0) if pn in ("fx", "fy") else 1.0
                sh = shifts.get(i["name"]) if pn in ("cx", "cy") and str(g).startswith("ZR") else None
                a_val = e["a"] / k - (sh[f"d{pn}_px"] if sh else 0.0)
                rows.append({"block": i["name"], "a": a_val, "b": e["b"] / k,
                             "sd_a": (e.get("sd_a") or np.nan) / k, "sd_b": (e.get("sd_b") or np.nan) / k,
                             "T": i.get("T_optics_degC"), "n_bins": e.get("n_bins"), "slope_source": e.get("slope_source")})
            if not rows:
                continue
            A = weighted_mean(np.array([r["a"] for r in rows]), np.array([r["sd_a"] for r in rows], float))
            fitted = [r for r in rows if r["slope_source"] == "fitted"] or rows
            B = weighted_mean(np.array([r["b"] for r in fitted]), np.array([r["sd_b"] for r in fitted], float))
            ent = {"a": A["value"], "b": B["value"], "sd_a": A["se"], "sd_b": B["se"], "tau_a": A["tau"],
                   "tau_b": B["tau"], "chi2_dof_a": A["chi2_dof"], "n_blocks": A["n"], "blocks": rows}
            Ts = np.array([np.nan if r["T"] is None else r["T"] for r in rows], float)
            ok = np.isfinite(Ts)
            if ok.sum() >= 3 and np.ptp(Ts[ok]) > 0:        # the trend with the lens temperature: reported, not applied
                T0 = float(np.mean(Ts[ok]))
                y = np.array([r["a"] for r in rows])[ok]
                sdv = np.array([r["sd_a"] for r in rows], float)[ok]
                try:
                    mt = meta_line(y, sdv, Ts[ok] - T0)
                    ent["temperature"] = {"c_per_degC": float(mt["slope"]), "sd": float(mt["slope_se"]),
                                          "z": mt["slope_z"], "T0_degC": T0, "sensor": "HEAD_HTR (optics)",
                                          "applied": False}
                except (np.linalg.LinAlgError, ValueError):
                    pass
            pars[pn] = ent
        if not pars:
            cal["eyes"][g] = {"skipped": "no block with focus lines for this eye"}
            continue
        common = [pn for pn in pars if form.get(pn) == "common"]
        if len(common) >= 2:
            bs = np.array([pars[pn]["b"] for pn in common])
            ss = np.array([pars[pn]["sd_b"] or np.nan for pn in common], float)
            ss = np.where(np.isfinite(ss) & (ss > 0), ss, np.nanmax(ss) if np.any(np.isfinite(ss)) else 1.0)
            wb = 1.0 / ss ** 2
            b_c, sd_c = float(np.sum(wb * bs) / np.sum(wb)), float(1.0 / np.sqrt(np.sum(wb)))
            for pn in common:
                pars[pn].update({"b": b_c, "sd_b": sd_c, "form": "common", "common_slope_of": list(common)})
        pp0 = gm.get("pp0_px")
        for pn, e in pars.items():
            f_ = form.get(pn, "free")
            if f_ in ("constant", "pinned"):
                e.update({"b": 0.0, "sd_b": 0.0, "form": f_})
            if f_ in ("pinned", "pinned_slope") and pn in ("cx", "cy") and pp0:
                e.update({"a": float(pp0[0 if pn == "cx" else 1]), "sd_a": 0.0, "form": f_,
                          "a_source": "pinned at pp0 (form)"})
        ref = float(ref if ref is not None else gm.get("reference_focus", 600.0))
        rr = regular_ratio(infos, g)
        reg = {}
        for pn, e in pars.items():
            k = rr["ratio"] if pn in ("fx", "fy") else 1.0
            reg[pn] = {"a": e["a"] / k, "b": e["b"] / k, "sd_a": e["sd_a"] / k, "sd_b": e["sd_b"] / k,
                       "tau_a": e.get("tau_a", 0.0), "chi2_dof_a": e.get("chi2_dof_a", float("nan")),
                       "n_blocks": e.get("n_blocks"), "derived": f"backlash / {k:.5f}" if k != 1.0 else "backlash"}
        lines = {"backlash": pars, "regular": reg, "reference_focus": ref}
        cal.setdefault("regular_ratio", {})[g] = rr
        gm["regular_ratio"] = float(rr["ratio"])
        bk = pars
        if "fx" in bk and "fy" in bk:
            gm["f0_px"] = float(np.sqrt(bk["fx"]["a"] * bk["fy"]["a"]))
            gm["slope_px_per_count"] = 0.5 * (bk["fx"]["b"] + bk["fy"]["b"])
            gm["aspect"] = bk["fy"]["a"] / bk["fx"]["a"]
            gm["reference_focus"] = ref
        if "cx" in bk or "cy" in bk:
            pp = dict(gm.get("pp") or {}, note="ZR principal point line (the R-L boresight; ZL the reference)")
            pp0l = list(gm.get("pp0_px") or [None, None])
            for k_, pn in ((0, "cx"), (1, "cy")):
                if pn in bk:
                    pp[f"{pn}_px_per_count"] = bk[pn]["b"]
                    pp0l[k_] = bk[pn]["a"]
            gm["pp"] = pp
            if all(v is not None for v in pp0l) and not gm.get("pp0_px"):
                gm["pp0_px"] = pp0l
        if "k1" in bk and "k2" in bk:
            dn = [n for n in ("k1", "k2", "p1", "p2") if n in bk]
            gm["distortion"] = {"names": dn, "params": [bk[n]["a"] for n in dn],
                                "sd": [bk[n].get("sd_a") for n in dn], "tau": [bk[n].get("tau_a") for n in dn],
                                "note": "k1, k2" + (", p1, p2" if "p1" in dn else "") + " constant with focus (k3 = 0 held"
                                + ("; p1 = p2 = 0 held" if "p1" not in dn else "") + ")"}
        gm["lines"] = {st: {pn: {k: v for k, v in e.items() if k in ("a", "b", "form", "derived")}
                            for pn, e in prs.items()} for st, prs in lines.items() if isinstance(prs, dict)}
        gm["lines"]["reference_focus"] = ref
        gm.pop("trend", None)
        gm.pop("trend_v0p42", None)
        gm["thermal"] = {"T0_degC": -15.0, "sensor": "HEAD_FPA", "f_px_per_degC": 0.0, "note": "no temperature term"}
        cal["eyes"][g] = lines
    rigs = []
    for i in infos:
        for fam, v in (i.get("rig") or {}).items():
            if str(fam) == zoom_key and v.get("R_refined") is not None and v.get("t_refined") is not None:
                Ra, ta = rig_adj.get(i["name"], (np.asarray(v["R_refined"], float), np.asarray(v["t_refined"], float)))
                tcahv = v.get("t_sensor_from_ref_cahv") or v.get("t_sensor_from_ref")
                rigs.append({"block": i["name"], "rotvec": Rotation.from_matrix(Ra).as_rotvec(), "t": ta,
                             "baseline_cahv_mm": None if tcahv is None else float(np.linalg.norm(np.asarray(tcahv, float)) * 1e3),
                             "T_optics_degC": i.get("T_optics_degC"), "sol": i.get("sol"),
                             "pp_shift_px": [shifts[i["name"]]["dcx_px"], shifts[i["name"]]["dcy_px"]] if i["name"] in shifts else None,
                             "R0": np.asarray(v["R_sensor_from_ref"], float) if v.get("R_sensor_from_ref") is not None else None})
    if rigs:
        rv = np.array([r["rotvec"] for r in rigs])
        tt = np.array([r["t"] for r in rigs])
        R = Rotation.from_rotvec(rv.mean(axis=0)).as_matrix()
        t = tt.mean(axis=0)
        t_source = "mean of the blocks' refined translations"
        if rig_translation == "cahv":
            tc = []
            for i in infos:
                v = (i.get("rig") or {}).get(zoom_key) or {}
                tv_ = v.get("t_sensor_from_ref_cahv") or v.get("t_sensor_from_ref")
                if tv_ is not None:
                    tc.append(np.asarray(tv_, float))
            if tc:
                t = np.mean(np.array(tc), axis=0)
                t_source = f"CAHV median of the pairs, mean of {len(tc)} blocks (held in the runs)"
        res_rows = []
        for r in rigs:
            if r["R0"] is None:
                continue
            ang = Rotation.from_matrix(Rotation.from_rotvec(r["rotvec"]).as_matrix() @ r["R0"].T).as_rotvec()
            res_rows.append({"block": r["block"], "T_optics_degC": r["T_optics_degC"], "sol": r["sol"],
                             "yaw_mdeg": float(np.degrees(ang[1]) * 1e3), "pitch_mdeg": float(np.degrees(ang[0]) * 1e3),
                             "roll_mdeg": float(np.degrees(ang[2]) * 1e3)})
        residuals: Dict[str, Any] = {"per_block": res_rows, "fits": {}}
        for ang_name in ("yaw", "pitch", "roll"):
            y = np.array([rr_[f"{ang_name}_mdeg"] for rr_ in res_rows], float)
            T = np.array([np.nan if rr_["T_optics_degC"] is None else rr_["T_optics_degC"] for rr_ in res_rows], float)
            S = np.array([np.nan if rr_["sol"] is None else rr_["sol"] for rr_ in res_rows], float)
            residuals["fits"][ang_name] = {"mean_mdeg": float(np.mean(y)) if len(y) else None,
                                           "sd_mdeg": float(np.std(y, ddof=1)) if len(y) > 1 else None,
                                           "vs_T": ols(T, y), "vs_sol": ols(S / 1000.0, y)}
        sd_mdeg = (np.degrees(rv.std(axis=0, ddof=1)) * 1e3).tolist() if len(rigs) > 1 else None
        out["rig"] = {"R_sensor_from_ref": R.tolist(), "t_sensor_from_ref": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
                      "translation_source": t_source, "residuals": residuals,
                      "rotvec_mdeg": (np.degrees(rv.mean(axis=0)) * 1e3).tolist(), "sd_rotvec_mdeg": sd_mdeg,
                      "sd_baseline_mm": float(np.std(np.linalg.norm(tt, axis=1), ddof=1) * 1e3) if len(rigs) > 1 else None,
                      "blocks": [r["block"] for r in rigs],
                      "per_block": [{"block": r["block"], "rotvec_mdeg": (np.degrees(r["rotvec"]) * 1e3).tolist(),
                                     "baseline_mm": float(np.linalg.norm(r["t"]) * 1e3), "T_optics_degC": r["T_optics_degC"],
                                     "sol": r["sol"], "pp_shift_px": r["pp_shift_px"],
                                     "baseline_cahv_mm": r.get("baseline_cahv_mm"),
                                     "dbaseline_mm": (None if r.get("baseline_cahv_mm") is None
                                                      else float(np.linalg.norm(r["t"]) * 1e3 - r["baseline_cahv_mm"]))} for r in rigs],
                      "baseline_vs_cahv": _baseline_vs_cahv(rigs),
                      "note": f"mean of {len(rigs)} blocks' refined Mastcam-Z rigs (right from left; the same for every "
                              f"focus - the focus motion is the principal points'; ZR cx, cy at the reference focus = "
                              f"pp0, each block's offset moved into its rig rotation first)"}
        cal["rig"] = {k: v for k, v in out["rig"].items() if k != "R_sensor_from_ref"}
    cal["loo"] = zcam_leave_one_out(infos)
    out["calibration"] = cal
    out["state"] = "calibrated"
    out["form"] = forms.ZCAM_FORM_ID
    out["source"] = f"MPPP Mastcam-Z calibration (mppp.calib.fit.fit_zcam) from {len(infos)} blocks: {', '.join(cal['blocks'])}"
    return out


def check_zcam(new: Dict[str, Any], old: Dict[str, Any], limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sanity limits of a new focus model against the one in use (:data:`ZCAM_MODEL_LIMITS`): the form, at least
    ``min_blocks`` blocks (``min_blocks_start`` for a start update, ``new["start_update"]`` from the study's
    ``start_model_from_blocks``), every eye's fx, fy lines fitted (a calibration) or its block offsets within
    ``max_offset_scatter_pct`` (a start update), f0 within ``max_f0_change_pct``, aspect within ``max_aspect_dev``
    of 1, |k1| below ``max_k1``, the between-block scatter of f, the rig baseline in range and within
    ``max_rig_change_deg`` of the old rig.  Returns {"ok", "checks"}."""
    lim = dict(ZCAM_MODEL_LIMITS, **(limits or {}))
    checks = []

    def add(name, ok, note):
        checks.append({"check": name, "ok": bool(ok), "note": note})
    bad = forms.validate_zcam(new)
    add("form", not bad, "; ".join(bad) or forms.ZCAM_FORM_ID)
    cal, su = new.get("calibration"), new.get("start_update")
    nb = len(((cal or su or {}).get("blocks")) or [])
    add("blocks", nb >= (lim["min_blocks"] if cal is not None else lim["min_blocks_start"]), f"{nb} blocks")
    for g, c in (new.get("cameras") or {}).items():
        o = (old.get("cameras") or {}).get(g) or {}
        e = ((cal or {}).get("eyes") or {}).get(g) or {}
        bk = e.get("backlash") if isinstance(e, dict) else None
        if cal is not None:                     # a calibration must have fitted every eye
            has = bool(bk) and all(isinstance(bk.get(pn), dict) for pn in ("fx", "fy"))
            add(f"{g} lines", has, "fx, fy lines fitted" if has else str(e.get("skipped") or "no backlash fx, fy lines"))
        if su is not None:
            e2 = (su.get("eyes") or {}).get(g) or {}
            if e2.get("skipped"):
                add(f"{g} start", False, str(e2["skipped"]))
            elif e2.get("scatter_offsets_pct") is not None:
                add(f"{g} block offsets", e2["scatter_offsets_pct"] <= lim["max_offset_scatter_pct"],
                    f"sd {e2['scatter_offsets_pct']:.2f} % of f0")
        if c.get("f0_px") and o.get("f0_px"):
            d = 100.0 * abs(c["f0_px"] / o["f0_px"] - 1.0)
            add(f"{g} f0", d <= lim["max_f0_change_pct"], f"{o['f0_px']:.1f} -> {c['f0_px']:.1f} px ({d:.2f} %)")
        if c.get("aspect"):
            add(f"{g} aspect", abs(c["aspect"] - 1.0) <= lim["max_aspect_dev"], f"{c['aspect']:.5f}")
        dist = c.get("distortion") or {}
        if "k1" in (dist.get("names") or []):
            k1 = dist["params"][dist["names"].index("k1")]
            add(f"{g} k1", abs(k1) <= lim["max_k1"], f"{k1:+.4f}")
        for pn in ("p1", "p2"):                 # zcam-v2: the tangential terms, small and agreed between the blocks
            if pn in (dist.get("names") or []):
                i_ = dist["names"].index(pn)
                v = float(dist["params"][i_])
                tau = (dist.get("tau") or [None] * len(dist["names"]))[i_]
                ok_ = abs(v) <= lim["max_p"] and (tau is None or not np.isfinite(tau) or float(tau) <= lim["max_tau_p"])
                add(f"{g} {pn}", ok_, f"{v:+.2e}" + (f", between-block sd {float(tau):.1e}" if tau is not None and np.isfinite(tau) else ""))
        for pn in ("fx", "fy"):
            x = (bk or {}).get(pn) or {}
            if isinstance(x, dict) and np.isfinite(x.get("chi2_dof_a", np.nan)):
                add(f"{g} backlash {pn} block scatter", float(x.get("tau_a") or 0.0) <= lim["max_tau_f_px"],
                    f"between-block sd {float(x.get('tau_a') or 0.0):.1f} px (chi2/dof {x['chi2_dof_a']:.1f})")
    rig = new.get("rig")
    if rig:
        b = float(rig.get("baseline_m") or 0.0)
        add("rig baseline", lim["baseline_m"][0] <= b <= lim["baseline_m"][1], f"{b:.4f} m")
        if rig.get("sd_rotvec_mdeg"):
            s = max(float(x) for x in rig["sd_rotvec_mdeg"])
            add("rig scatter", s <= lim["max_rig_scatter_mdeg"],
                f"rotation sd {', '.join(f'{float(x):.1f}' for x in rig['sd_rotvec_mdeg'])} mdeg over {len(rig.get('blocks') or [])} blocks")
        if rig.get("sd_baseline_mm") is not None:
            add("rig baseline scatter", float(rig["sd_baseline_mm"]) <= lim["max_baseline_scatter_mm"],
                f"sd {float(rig['sd_baseline_mm']):.2f} mm")
        orig = old.get("rig") or {}
        if orig.get("R_sensor_from_ref") and rig.get("R_sensor_from_ref"):
            ang = 1e-3 * J.rotation_mdeg(orig["R_sensor_from_ref"], rig["R_sensor_from_ref"])
            add("rig change", ang <= lim["max_rig_change_deg"], f"{ang:.3f} deg from the rig in use")
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# ------------------------------------------------------------------------------------------- block results vs the joint
def read_rigs_bin(path: PathLike) -> Dict[int, Dict[str, Any]]:
    """{rig id: {"ref_camera", "sensors": {camera id: {"R", "t"}}}} of a COLMAP 4 ``rigs.bin`` (the sensor pose is
    x_sensor = R x_ref + t)."""
    import struct
    from scipy.spatial.transform import Rotation
    b = Path(path).read_bytes()
    o = 0
    (n,) = struct.unpack_from("<Q", b, o)
    o += 8
    out: Dict[int, Dict[str, Any]] = {}
    for _ in range(n):
        rid, ns = struct.unpack_from("<II", b, o)
        o += 8
        _t, ref = struct.unpack_from("<iI", b, o)
        o += 8
        rig: Dict[str, Any] = {"ref_camera": int(ref), "sensors": {}}
        for _ in range(ns - 1):
            _t, cid = struct.unpack_from("<iI", b, o)
            o += 8
            (hp,) = struct.unpack_from("<B", b, o)
            o += 1
            if hp:
                q = struct.unpack_from("<4d", b, o)
                o += 32
                t = struct.unpack_from("<3d", b, o)
                o += 24
                rig["sensors"][int(cid)] = {"R": Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(),
                                            "t": np.array(t)}
            else:
                rig["sensors"][int(cid)] = None
        out[int(rid)] = rig
    return out


def block_navcam_results(folder: PathLike, model: str = "cahv_ba") -> Dict[str, Any]:
    """
    What one aligned block measured of the Navcam (0.90, for the parameter plots and the block table): per camera
    key (eye, rig-epoch copy ``NL#1`` and temperature bin ``NL_T-020-017.5``) the refined parameters and the start,
    the images' mean temperature and sols; per rig epoch the right-from-left rotation.  From ``colmap/project.json``,
    ``colmap/sparse/<model>/cameras.bin`` and ``rigs.bin``.  ``sol_median`` and ``T_median`` are over the Navcam images.
    """
    from ..sfm.thermal import eye_key
    pdir = Path(folder)
    pdir = pdir / "colmap" if (pdir / "colmap" / "project.json").is_file() else pdir
    P = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
    cams = read_cameras_bin(pdir / "sparse" / model / "cameras.bin")
    rigs = read_rigs_bin(pdir / "sparse" / model / "rigs.bin") if (pdir / "sparse" / model / "rigs.bin").is_file() else {}
    db = (P.get("settings") or {}).get("database") or {}
    id_of_key = {str(k): int(v) for k, v in (db.get("cameras") or {}).items()}
    key_of_id = {v: k for k, v in id_of_key.items()}
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    nav = []
    for r in P.get("images") or []:
        k = str(r.get("instrument") or "")
        if not k.startswith("N"):
            continue
        nav.append(r)
        by_key.setdefault(k, []).append(r)
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    T_all = [t for t in (_f(r.get("camera_temperature_degC")) for r in nav) if t is not None]
    sols_all = [int(r["sol"]) for r in nav if r.get("sol") is not None]
    import re
    name = pdir.parent.name
    out: Dict[str, Any] = {"block": name, "site": re.sub(r"^mars2020_sol_\d+_", "", name).replace("_zcam", ""),
                           "project": str(pdir), "model": model,
                           "sol_median": float(np.median(sols_all)) if sols_all else None,
                           "T_median": float(np.median(T_all)) if T_all else None,
                           "navcam_images": len(nav), "cameras": {}, "rigs": {}}
    for key, rows in by_key.items():
        cid = id_of_key.get(key)
        if cid is None or cid not in cams:
            continue
        T = [t for t in (_f(r.get("camera_temperature_degC")) for r in rows) if t is not None]
        sols = [int(r["sol"]) for r in rows if r.get("sol") is not None]
        start = ((P.get("cameras") or {}).get(key) or {}).get("params")
        out["cameras"][key] = {"eye": eye_key(key), "camera_id": int(cid), "images": len(rows),
                               "T_mean": float(np.mean(T)) if T else None, "T_range": [min(T), max(T)] if T else None,
                               "sols": [min(sols), max(sols)] if sols else None,
                               "sol_median": float(np.median(sols)) if sols else None,
                               "params": [float(v) for v in cams[cid]],
                               "start": [float(v) for v in start] if start else None,
                               "bin": "_T" in key}
    for rkey, rid in (db.get("rig_ids") or {}).items():
        rig = rigs.get(int(rid))
        if not rig:
            continue
        ref_key = key_of_id.get(rig["ref_camera"])
        for cid, s in rig["sensors"].items():
            if s is None:
                continue
            sk = key_of_id.get(cid)
            epoch = int(str(rkey).split("#", 1)[1]) if "#" in str(rkey) else 0     # "N" -> epoch 0, "N#1" -> 1
            rows = [r for r in nav if int(r.get("rig_epoch") or 0) == epoch]
            sols = [int(r["sol"]) for r in rows if r.get("sol") is not None]
            T = [t for t in (_f(r.get("camera_temperature_degC")) for r in rows) if t is not None]
            out["rigs"][str(rkey)] = {"rig_id": int(rid), "ref": ref_key, "sensor": sk, "R": s["R"].tolist(),
                                      "t": s["t"].tolist(), "baseline_m": float(np.linalg.norm(s["t"])),
                                      "images": len(rows), "sol_median": float(np.median(sols)) if sols else None,
                                      "sols": [min(sols), max(sols)] if sols else None,
                                      "T_mean": float(np.mean(T)) if T else None}
    return out


def navcam_block_table(results: Sequence[Dict[str, Any]], nl: Dict[str, Any], nr: Dict[str, Any],
                       rig: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Each block's Navcam parameters against the joint (promoted) model evaluated at the block's temperature and sol
    (:func:`mppp.calib.constrain.camera_at`, :func:`rig_at`): one row per block with, per eye, the image-weighted
    mean over its temperature bins of fx, fy (ppm), cx, cy (px) and k1 (x 1e3) refined minus joint, and the rig
    yaw / pitch / roll (mdeg) of the block's epochs minus the drift model.  ``rows[i]["agreement"]`` is the largest
    |fx, fy| in ppm and the largest rig angle in mdeg - the two numbers that say whether the block agrees.
    """
    from .constrain import camera_at, rig_at
    from .joint import rig_angles
    models = {"NL": nl, "NR": nr}
    rows = []
    for b in results:
        row: Dict[str, Any] = {"block": b.get("site") or b["block"], "sol": b.get("sol_median"), "T": b.get("T_median"),
                               "images": b.get("navcam_images"), "bins": 0}
        for eye in ("NL", "NR"):
            cams = [c for k, c in b["cameras"].items() if c["eye"] == eye and c["bin"] and c.get("T_mean") is not None]
            if not cams:                                       # no temperature bins: the eye camera itself
                cams = [c for k, c in b["cameras"].items() if c["eye"] == eye and c.get("T_mean") is not None and "#" not in k]
            d = {"fx_ppm": [], "fy_ppm": [], "cx_px": [], "cy_px": [], "k1_1e3": [], "w": []}
            for c in cams:
                _m, pj, _a = camera_at(models[eye], T_degC=c["T_mean"], sol=c.get("sol_median"))
                p = c["params"]
                d["fx_ppm"].append(1e6 * (p[0] / pj[0] - 1))
                d["fy_ppm"].append(1e6 * (p[1] / pj[1] - 1))
                d["cx_px"].append(p[2] - pj[2])
                d["cy_px"].append(p[3] - pj[3])
                d["k1_1e3"].append(1e3 * (p[4] - pj[4]))
                d["w"].append(c["images"])
            w = np.array(d.pop("w"), float)
            row["bins"] += len(cams)
            for k, v in d.items():
                row[f"{eye}_{k}"] = float(np.average(v, weights=w)) if len(v) else None
        if rig and b.get("rigs"):
            angs = []
            for rk, r in b["rigs"].items():
                Rj, _t, _a = rig_at(rig, sol=r.get("sol_median"))
                a = rig_angles(np.asarray(r["R"]), Rj)
                angs.append((r["images"], a))
            w = np.array([x[0] for x in angs], float)
            for a in ("yaw", "pitch", "roll"):
                row[f"rig_{a}_mdeg"] = float(np.average([x[1][f"{a}_mdeg"] for x in angs], weights=w))
            row["rig_epochs"] = len(angs)
        f = [abs(row.get(k) or 0.0) for k in ("NL_fx_ppm", "NL_fy_ppm", "NR_fx_ppm", "NR_fy_ppm")]
        g = [abs(row.get(k) or 0.0) for k in ("rig_yaw_mdeg", "rig_pitch_mdeg", "rig_roll_mdeg")]
        row["agreement"] = {"focal_ppm": max(f) if f else None, "rig_mdeg": max(g) if g else None}
        rows.append(row)
    return sorted(rows, key=lambda r: (r["sol"] is None, r["sol"] or 0))


def block_table_markdown(rows: Sequence[Dict[str, Any]], focal_limit_ppm: float = 100.0,
                         rig_limit_mdeg: float = 5.0) -> str:
    """The block table (:func:`navcam_block_table`) as Markdown: refined minus joint per block; a block beyond the
    limits is marked."""
    L = ["| block | sol | T °C | images | bins | NL fx fy ppm | NL cx cy px | NL k1 ×1e3 | NR fx fy ppm | NR cx cy px | "
         "NR k1 ×1e3 | rig yaw pitch roll mdeg | agrees |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def _p(r, a, b, fmt):
        x, y = r.get(a), r.get(b)
        return "-" if x is None else f"{x:{fmt}} {y:{fmt}}"
    for r in rows:
        ag = r["agreement"]
        ok = (ag["focal_ppm"] is None or ag["focal_ppm"] <= focal_limit_ppm) and \
             (ag["rig_mdeg"] is None or ag["rig_mdeg"] <= rig_limit_mdeg)
        rigs = ("-" if r.get("rig_yaw_mdeg") is None else
                f"{r['rig_yaw_mdeg']:+.1f} {r['rig_pitch_mdeg']:+.1f} {r['rig_roll_mdeg']:+.1f}")
        L.append(f"| {r['block']} | {r['sol']:.0f} | {r['T']:+.1f} | {r['images']} | {r['bins']} | "
                 f"{_p(r, 'NL_fx_ppm', 'NL_fy_ppm', '+.0f')} | {_p(r, 'NL_cx_px', 'NL_cy_px', '+.2f')} | "
                 f"{'-' if r.get('NL_k1_1e3') is None else f'{r['NL_k1_1e3']:+.3f}'} | "
                 f"{_p(r, 'NR_fx_ppm', 'NR_fy_ppm', '+.0f')} | {_p(r, 'NR_cx_px', 'NR_cy_px', '+.2f')} | "
                 f"{'-' if r.get('NR_k1_1e3') is None else f'{r['NR_k1_1e3']:+.3f}'} | {rigs} | "
                 f"{'yes' if ok else '**no**'} |")
    L.append("")
    L.append(f"Refined minus the joint model at each bin's temperature and the block's sol; agrees: |fx, fy| ≤ "
             f"{focal_limit_ppm:g} ppm and rig angles ≤ {rig_limit_mdeg:g} mdeg.")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------- rig drift (0.90)
def wls_tau(y: np.ndarray, X: np.ndarray, var: np.ndarray) -> Dict[str, Any]:
    """Weighted least squares with an additive between-block variance tau^2 (DerSimonian-Laird generalised to a
    regression; the study's ``navcal._wls``).  Returns coef, se, p, tau, Q, cov, n_used."""
    from scipy import stats
    y, X, var = np.asarray(y, float), np.asarray(X, float), np.asarray(var, float)
    ok = np.isfinite(y) & np.isfinite(var) & np.all(np.isfinite(X), axis=1)
    if ok.sum() <= X.shape[1]:
        nan = [float("nan")] * X.shape[1]
        return {"coef": nan, "se": nan, "p": nan, "tau": float("nan"), "Q": float("nan"),
                "cov": np.full((X.shape[1], X.shape[1]), np.nan), "n_used": int(ok.sum())}
    y, X, var = y[ok], X[ok], var[ok]
    w = 1.0 / var
    W = np.diag(w)
    XtWX = X.T @ W @ X
    b = np.linalg.solve(XtWX, X.T @ W @ y)
    r = y - X @ b
    Q = float(np.sum(w * r ** 2))
    k, p = X.shape
    P = W - W @ X @ np.linalg.solve(XtWX, X.T @ W)
    tau2 = max(0.0, (Q - (k - p)) / float(np.trace(P)))
    w2 = 1.0 / (var + tau2)
    A = X.T @ np.diag(w2) @ X
    cov = np.linalg.inv(A)
    b2 = cov @ (X.T @ (w2 * y))
    se = np.sqrt(np.diag(cov))
    z = b2 / se
    pv = 2 * stats.norm.sf(np.abs(z))
    r2 = y - X @ b2
    return {"coef": b2.tolist(), "se": se.tolist(), "p": pv.tolist(), "tau": float(np.sqrt(tau2)), "Q": Q,
            "Q2": float(np.sum(w2 * r2 ** 2)), "cov": cov, "n_used": int(ok.sum()), "dof": int(k - p)}


def rig_rows(blocks: Sequence[Dict[str, Any]], sd_mdeg_at_100_images: float = 1.0) -> List[Dict[str, Any]]:
    """One row per rig epoch of the blocks (:func:`block_navcam_results`): absolute pitch, yaw, roll (mdeg, the
    rotation vector of right-from-left in the sensor frame, the rig file's convention), sol, T, images, and a
    nominal sd that scales with 1/sqrt(images) (the block adjustments do not give one)."""
    from .joint import rig_angles
    rows = []
    for b in blocks:
        for key, r in (b.get("rigs") or {}).items():
            if r.get("sol_median") is None or r.get("T_mean") is None:
                continue
            a = rig_angles(np.asarray(r["R"]))
            sd = sd_mdeg_at_100_images * np.sqrt(100.0 / max(r["images"], 1))
            rows.append({"block": b.get("site") or b["block"], "epoch": key, "sol": float(r["sol_median"]),
                         "T": float(r["T_mean"]), "images": int(r["images"]), "sols": r.get("sols"),
                         "pitch_mdeg": a["pitch_mdeg"], "yaw_mdeg": a["yaw_mdeg"], "roll_mdeg": a["roll_mdeg"],
                         "sd_pitch_mdeg": sd, "sd_yaw_mdeg": sd, "sd_roll_mdeg": sd})
    return rows


def fit_rig_drift(rows: Sequence[Dict[str, Any]], knot_sol: Union[float, str, None] = "free",
                  knot_angles: Sequence[str] = ("pitch",), no_drift: Sequence[str] = (),
                  held: Sequence[str] = (), knot_grid: Sequence[float] = tuple(range(100, 1001, 10)),
                  reference_sols: Optional[Sequence[float]] = None, sol0: Optional[float] = 0.0) -> Dict[str, Any]:
    """
    The rig drift model in the rig file's form (``drift``: ``rate (sol - sol0) + (early_rate - rate) (min(sol,
    knot_sol) - knot_ref)`` per angle, :func:`mppp.calib.constrain.drift_offset_mdeg`) from rig-epoch rows
    (:func:`rig_rows`, or the study's per-block rigs): per angle a between-block regression on camera temperature
    and sol with weights 1 / (sd^2 + tau^2); ``knot_angles`` get a hinge whose knot is ``knot_sol`` or, with
    ``"free"`` (0.90), the sol in ``knot_grid`` that minimises the weighted residual sum of squares (profile fit;
    ``knot_profile`` in the result); ``no_drift`` angles keep their fit for information but get rate 0 (yaw is
    held in the block runs); ``held`` angles get rate 0 and no hinge.  ``sol0`` and ``knot_ref`` are means over
    ``reference_sols`` (default: the rows) so the drift averages to zero over the blocks the joint rig came from.
    """
    rows = [r for r in rows if all(np.isfinite(r.get(k, np.nan)) for k in ("sol", "T", "pitch_mdeg", "yaw_mdeg", "roll_mdeg"))]
    T = np.array([r["T"] for r in rows], float)
    sol = np.array([r["sol"] for r in rows], float)
    ref_sol = np.array(reference_sols if reference_sols is not None else sol, float)
    out: Dict[str, Any] = {"model": "linear", "sol0": float(ref_sol.mean()) if sol0 is None else float(sol0), "blocks": len(rows),
                           "blocks_names": [f"{r['block']}{'' if r.get('epoch') in (None, 'N') else ' ' + str(r['epoch'])}" for r in rows],
                           "sol_range": [float(sol.min()), float(sol.max())], "T_mean_degC": float(T.mean()),
                           "no_drift": list(no_drift), "held": list(held)}

    def _fit(ang: str, knot: Optional[float]) -> Dict[str, Any]:
        y = np.array([r[f"{ang}_mdeg"] for r in rows], float)
        v = np.maximum(np.array([r.get(f"sd_{ang}_mdeg", np.nan) for r in rows], float) ** 2, 1e-6)
        cols = [np.ones(len(rows)), T - T.mean(), sol - sol.mean()]
        if knot is not None:
            cols.append(np.minimum(sol, knot))
        return wls_tau(y, np.c_[tuple(cols)], v)

    knot: Optional[float] = None
    if knot_sol == "free" and knot_angles:
        prof = []
        for kk in knot_grid:
            if not (sol.min() + 30 < kk < sol.max() - 30):
                continue
            q = sum(_fit(a, float(kk))["Q2"] for a in knot_angles if a not in held)
            prof.append((float(kk), float(q)))
        if prof:
            knot = min(prof, key=lambda t: t[1])[0]
            qs = np.array([q for _, q in prof])
            # the 1-sigma range of the knot: where the profile is within 1 of its minimum (chi^2 with weights 1/(var+tau^2))
            inside = [k for (k, q) in prof if q <= qs.min() + 1.0]
            out["knot_profile"] = prof
            out["knot_sol_range"] = [min(inside), max(inside)]
    elif knot_sol not in (None, "free", False):
        knot = float(knot_sol)
    if knot is not None:
        out["model"] = "linear+hinge"
        out["knot_sol"] = knot
        out["knot_ref"] = float(np.mean(np.minimum(ref_sol, knot))) if sol0 is None else 0.0
    for ang in ("pitch", "yaw", "roll"):
        h = knot is not None and ang in knot_angles and ang not in held
        f = _fit(ang, knot if h else None)
        rate, se = float(f["coef"][2]), float(f["se"][2])
        out[f"{ang}_fit_mdeg_per_sol"] = rate
        out[f"{ang}_se"] = se
        out[f"{ang}_p"] = float(f["p"][2])
        out[f"{ang}_T_mdeg_per_degC"] = float(f["coef"][1])
        out[f"{ang}_T_se"] = float(f["se"][1])
        out[f"{ang}_tau_mdeg"] = float(f["tau"])
        out[f"{ang}_Q2"] = float(f["Q2"])
        out[f"{ang}_level_mdeg"] = float(f["coef"][0])
        out[f"{ang}_mdeg_per_sol"] = 0.0 if (ang in no_drift or ang in held) else rate
        if h:
            c = np.asarray(f["cov"])
            out[f"{ang}_early_mdeg_per_sol"] = rate + float(f["coef"][3])
            out[f"{ang}_early_se"] = float(np.sqrt(c[2, 2] + c[3, 3] + 2 * c[2, 3]))
        # the fitted angle at sol 0 (mean temperature): intercept minus the regressors' values at sol 0
        at0 = float(f["coef"][0]) + rate * (0.0 - sol.mean())
        if h:
            at0 -= float(f["coef"][3]) * min(0.0, knot)
        out[f"{ang}_at_sol0_mdeg"] = at0
    return out


def rig_with_drift(rig: Dict[str, Any], drift: Dict[str, Any], note: str = "",
                   keep_from_file: Sequence[str] = ("yaw",)) -> Dict[str, Any]:
    """A rig file (:func:`mppp.calib.models.load_rig`) with the drift model of :func:`fit_rig_drift` and its rotation
    moved to the model's reference sol (``drift["sol0"]``; the fitted ``<angle>_at_sol0_mdeg`` when sol0 = 0, else the
    rotation at sol0 of the previous model), so that :func:`mppp.calib.constrain.rig_at` reproduces the fit."""
    from scipy.spatial.transform import Rotation
    from .constrain import rig_at
    out = copy.deepcopy(rig)
    if float(drift.get("sol0", 0.0)) == 0.0 and all(f"{a}_at_sol0_mdeg" in drift for a in ("pitch", "yaw", "roll")):
        rv = np.radians(1e-3 * np.array([drift["pitch_at_sol0_mdeg"], drift["yaw_at_sol0_mdeg"], drift["roll_at_sol0_mdeg"]]))
        # an angle the runs hold to the model (yaw; ``keep_from_file``) keeps the file's own value at sol 0
        R0, _t, _ = rig_at(rig, sol=0.0)
        rv_old = Rotation.from_matrix(R0).as_rotvec()
        for i, a in enumerate(("pitch", "yaw", "roll")):
            if a in keep_from_file:
                rv[i] = rv_old[i]
        R = Rotation.from_rotvec(rv).as_matrix()
    else:
        R, _t, _ = rig_at(rig, sol=float(drift.get("sol0", 0.0)))
    out["R_sensor_from_ref"] = R.tolist()
    out["rotvec_rad"] = Rotation.from_matrix(R).as_rotvec().tolist()
    out["drift"] = {k: v for k, v in drift.items() if k != "knot_profile"}
    out["drift"]["note"] = (note or "rig drift refitted from the block rig epochs (mppp.calib.fit.fit_rig_drift)") + \
        f"; sol0 = {drift.get('sol0', 0.0):g}: the rotation is the rig at that sol"
    return out


def thermal_bin_rows(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The per-temperature-bin cameras of the blocks (:func:`block_navcam_results`) as the rows the intrinsics
    figure takes: {eye, block, T, n, fx, fy, cx, cy, k1} (full-frame px)."""
    from ..waypoints.sites import site_label
    rows = []
    for b in blocks:
        for key, c in b["cameras"].items():
            if not c["bin"] or c.get("T_mean") is None:
                continue
            p = c["params"]
            rows.append({"eye": c["eye"], "block": site_label(b.get("site") or b["block"]), "key": key,
                         "T": c["T_mean"], "n": c["images"], "fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3], "k1": p[4]})
    return rows
