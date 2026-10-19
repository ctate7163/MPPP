"""
The Navcam consensus in the v0p52/v0p53 form (MPPP v0p60): one joint adjustment of many blocks with

  - fisheye + tangential cameras at T0 (-20 degC), k4 = 0 (held), the other distortion terms shared and refined,
  - f scaled per image by the thermal slope (ppm/degC) and the NL principal point by its slope (px/degC), both
    taken from the cameras in use (not refitted: measured in v0p40/41),
  - one rig: a constant yaw for every block (refined; no temperature or drift term), pitch and roll with the mission
    drift of the rig in use (its yaw rate set to 0), translation from CAHV (v0p62: or fitted on the large blocks,
    :func:`fit_rig_translation`),

started from the cameras and rig in use (``src/mppp/data/cmods``, ``MPPP_CMODS``).  :func:`fit_consensus` runs it
(``scripts/navcam_calibration_study.py consensus``, notebook 04 §2d) and :func:`write_consensus` writes the
candidate folder (``M2020_NL/NR_fisheye_tangential.json``, ``M2020_N_rig.json``) for ``scripts/promote_cmods.py``.
"""
from __future__ import annotations

import copy
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
RAD_BINS = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)


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


THERMAL_TERMS = ("f", "NL_cx", "NR_cx", "NL_cy", "NR_cy", "NL_fx", "NL_fy", "NR_fx", "NR_fy", "yaw_rate")
THERMAL_STEPS = {"f": 10.0, "NL_cx": 0.03, "NR_cx": 0.03, "NL_cy": 0.03, "NR_cy": 0.03,      # profile grid steps
                 "NL_fx": 10.0, "NL_fy": 10.0, "NR_fx": 10.0, "NR_fy": 10.0, "yaw_rate": 3.0}
AXIS_TERMS = ("NL_fx", "NL_fy", "NR_fx", "NR_fy")
# v0p73 (Christian, 3 Oct 2026): the NL-reference form - fx and fy of each eye against temperature, NR cx against
# temperature (the stereo yaw change), one rig yaw for every temperature with a rate in sol (mdeg / 1000 sol)
V0P73_TERMS = ("NR_cx", "NL_fx", "NL_fy", "NR_fx", "NR_fy", "yaw_rate")   # NR_cx first: it and NR_fx trade a little
# v0p83 (Christian, 18 Oct 2026): an athermal rig - the stereo temperature change on both eyes' principal points (NL
# and NR cx against temperature), never a rig temperature term (that would need a rig per temperature bin).  The
# consensus in these forms profiles NL_cx too (pipeline pass 3 on; pass 2 of the 17 Oct run was NR cx only).
NL_PP_THERMAL_FORMS = ("v0p80",)
THERMAL_ROUNDS = 2          # v0p73: profile the terms this many times (one term at a time: the second round settles
                            # what the first left to the terms after it - synthetic: NR_fx 31.3 -> 30.0 for a true 30)
UNITS = {"f": "ppm/degC", "NL_fx": "ppm/degC", "NL_fy": "ppm/degC", "NR_fx": "ppm/degC", "NR_fy": "ppm/degC",
         "NL_cx": "px/degC", "NR_cx": "px/degC", "NL_cy": "px/degC", "NR_cy": "px/degC", "yaw_rate": "mdeg/1000 sol"}


def _thermal_values(th: Dict[str, Any]) -> Dict[str, float]:
    """{"f": ppm/degC, "NL_cx": px/degC, ...} of a consensus result's ``thermal`` entry (v0p73: also the per-eye,
    per-axis ppm and the yaw rate)."""
    pp = th.get("pp_slopes") or {}
    ax = th.get("axis_ppm") or {}
    f = float(th["ppm_per_degC"])
    return {"f": f,
            "NL_cx": float(pp.get("NL", [th.get("cx_px_per_degC_NL", 0.0), 0.0])[0]),
            "NL_cy": float(pp.get("NL", [0.0, 0.0])[1]),
            "NR_cx": float(pp.get("NR", [0.0, 0.0])[0]), "NR_cy": float(pp.get("NR", [0.0, 0.0])[1]),
            "NL_fx": float(ax.get("NL", [f, f])[0]), "NL_fy": float(ax.get("NL", [f, f])[1]),
            "NR_fx": float(ax.get("NR", [f, f])[0]), "NR_fy": float(ax.get("NR", [f, f])[1]),
            "yaw_rate": float(th.get("yaw_rate_mdeg_per_1000sol") or 0.0),
            # v0p80: k1 against temperature per eye (from the blocks' temperature bins; not profiled in the joint)
            "NL_k1": float((th.get("k1_slopes") or {}).get("NL") or 0.0),
            "NR_k1": float((th.get("k1_slopes") or {}).get("NR") or 0.0)}


def _pp_slopes(v: Dict[str, float]) -> Dict[int, Tuple[float, float]]:
    return {1: (float(v["NL_cx"]), float(v["NL_cy"])), 2: (float(v["NR_cx"]), float(v["NR_cy"]))}


def _axis_ppm(v: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """v0p73: {"NL": (fx, fy), "NR": (fx, fy)} ppm/degC of the thermal values."""
    return {"NL": (float(v["NL_fx"]), float(v["NL_fy"])), "NR": (float(v["NR_fx"]), float(v["NR_fy"]))}


def _set_term(v: Dict[str, float], term: str, value: float) -> Dict[str, float]:
    """The thermal values with ``term`` set (``"f"`` sets the four per-axis values too)."""
    out = dict(v, **{term: float(value)})
    if term == "f":
        out.update({k: float(value) for k in AXIS_TERMS})
    return out


def _drift_with_yaw(drift: Optional[Dict[str, Any]], yaw_rate_per_1000: float) -> Optional[Dict[str, Any]]:
    """v0p73: the drift with one yaw rate (mdeg per 1000 sol, before and after the knot alike); None stays None
    unless a rate is given (then a plain drift about sol 0... is not meaningful: the drift's sol0 is needed)."""
    if not drift:
        return drift
    d = dict(drift)
    r = 1e-3 * float(yaw_rate_per_1000)
    d["yaw_mdeg_per_sol"] = r
    d["yaw_fit_mdeg_per_sol"] = r
    if "yaw_early_mdeg_per_sol" in d or d.get("knot_sol"):
        d["yaw_early_mdeg_per_sol"] = r
    return d


def _joint_kw(v: Dict[str, float], drift: Optional[Dict[str, Any]], axis: bool) -> Dict[str, Any]:
    """The keyword arguments of :func:`navcal.joint_adjust` for thermal values ``v`` (v0p73: ``axis`` - per-eye,
    per-axis ppm and the yaw rate in the drift)."""
    kw: Dict[str, Any] = {"pp_slopes": _pp_slopes(v)}
    if axis:
        kw["axis_ppm"] = _axis_ppm(v)
    k1 = {e: float(v.get(f"{e}_k1") or 0.0) for e in ("NL", "NR")}
    if any(k1.values()):                                  # v0p80
        kw["k1_slopes"] = k1
    d = _drift_with_yaw(drift, v.get("yaw_rate", 0.0)) if axis else drift
    if d:
        kw["drift"] = d
    return kw


def profile_thermal(rec, proj, temps, values: Dict[str, float], T0: float, term: str, drift=None,
                    iterations: int = 50, npoints: int = 5, verbose: bool = True, axis: bool = False) -> Dict[str, Any]:
    """
    v0p64: the joint cost over a grid of one thermal term (``"f"`` ppm/degC, ``"NL_cx"`` ... px/degC; v0p73 with
    ``axis``: ``"NL_fx"`` ... ppm/degC of one eye and axis, ``"yaw_rate"`` mdeg per 1000 sol of the rig yaw), the
    others at ``values``, each adjustment started from the same converged state (``rec`` is not changed).  The best
    value is the minimum of a parabola through the costs, its sd sqrt(variance factor / curvature)
    (:func:`navcal.fit_profile`).  A minimum outside the grid extends it by two steps (twice at most).
    """
    import copy as _copy
    from . import navcal as NC
    step = THERMAL_STEPS[term]
    c0 = float(values[term])
    grid = [c0 + step * (k - (npoints - 1) / 2) for k in range(npoints)]
    rows: list = []
    done = set()
    t_all = time.time()
    for _ in range(3):
        for g in grid:
            if round(g, 9) in done:
                continue
            done.add(round(g, 9))
            v = _set_term(values, term, g)
            r = _copy.deepcopy(rec)
            t = time.time()
            kw = _joint_kw(v, drift, axis)
            ba = NC.joint_adjust(r, proj, temps, v["f"], T0, max_iterations=iterations, rig_slopes=(0.0, 0.0), **kw)
            red = max(2 * ba["observations"] - 1, 1)
            rows.append({"ppm_per_degC": float(g), "cost": float(ba["final_cost"]), "iterations": ba["iterations"],
                         "variance_factor": float(2 * ba["final_cost"] / red)})
            if verbose:
                print(f"  {term} {g:+.4f}: cost {ba['final_cost']:.2f} it {ba['iterations']} ({time.time() - t:.0f} s)  "
                      f"{NC.progress(len(rows), max(npoints, len(rows)), t_all, 'profile ' + term)}", flush=True)
        fit = NC.fit_profile(sorted(rows, key=lambda r: r["ppm_per_degC"]))
        lo, hi = min(r["ppm_per_degC"] for r in rows), max(r["ppm_per_degC"] for r in rows)
        b = fit["best_ppm_per_degC"]
        if lo <= b <= hi:
            break
        grid = [hi + step, hi + 2 * step] if b > hi else [lo - step, lo - 2 * step]
    out = {"term": term, "unit": UNITS.get(term), "best": float(fit["best_ppm_per_degC"]),
           "sd": float(fit["sd_ppm_per_degC"]),
           "start": c0, "curvature": fit["curvature"], "variance_factor": fit["variance_factor"],
           "rows": [{"value": r["ppm_per_degC"], "cost": r["cost"], "iterations": r["iterations"]} for r in fit["rows"]],
           "at_edge": not (lo <= fit["best_ppm_per_degC"] <= hi)}
    out["z"] = (out["best"] / out["sd"]) if out["sd"] and np.isfinite(out["sd"]) and out["sd"] > 0 else None
    if verbose:
        print(f"{term}: {out['best']:+.4f} +- {out['sd']:.4f} {UNITS.get(term, '')} (start {c0:+.4f})"
              + ("  [outside the grid]" if out["at_edge"] else ""), flush=True)
    return out


# ------------------------------------------------------------------------------------- early-mission offsets (v0p70)
EARLY_MISSION_SOL = 380.0                   # v0p70: the Navcam principal points sat ~2.5-3 px lower in cx before this
EARLY_TERMS = ("fx", "fy", "cx", "cy")
EARLY_STEP_PX = 0.5                         # design step of the response surface (px)
EARLY_APPLY_Z = 3.0                         # terms at least this many sd from 0 are applied by the projects


def early_keypoint_map(proj, before_sol: float, offsets) -> Any:
    """
    v0p70: the keypoint map that makes the images taken before ``before_sol`` see a camera with fx + dfx, fy + dfy,
    cx + dcx, cy + dcy (``offsets``, px, the same for both eyes) while they keep the shared camera: a keypoint u of
    such an image becomes c + f (u - c - dc) / (f + df) per axis - exact for any COLMAP model, whose focal lengths
    scale and principal point shifts the distorted normalised coordinates.
    """
    o = np.asarray(offsets, float)
    early = {int(r["image_id"]) for r in proj.images
             if r.get("sol") is not None and float(r["sol"]) < float(before_sol) and "image_id" in r}

    def fmap(iid, kps, cam):
        if int(iid) not in early or not np.any(o):
            return kps
        p = np.asarray(cam.params, float)
        f, c = p[0:2], p[2:4]
        return c + f * (kps - c - o[2:4]) / (f + o[0:2])
    fmap.early_images = early                      # type: ignore[attr-defined]
    fmap.offsets = o                               # type: ignore[attr-defined]
    return fmap


def _quadratic_design(n: int, h: float):
    """Centre, +-h on each axis and +h on each pair of axes: (1 + 2n + n(n-1)/2) points, enough for a full quadratic."""
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


def _read_cameras_bin(path: PathLike) -> Dict[int, np.ndarray]:
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
    """
    v0p70: a start for :func:`fit_early_offsets` from the block alignments: each block's refined eye cameras (NL, NR)
    minus their start (the consensus at the block's temperature), fx, fy, cx, cy; the mean over the eyes of the
    blocks whose images are all before ``before_sol`` minus that of the blocks all after it (blocks spanning the sol
    are left out).  Returns {"start": [dfx, dfy, dcx, dcy], "blocks": {name: {"sol_median", "early", "d": [...]}}}.
    """
    from .project import SfmProject
    rows: Dict[str, Any] = {}
    for name, w in scapes_cfg.items():
        root = Path(w) / "colmap" if (Path(w) / "colmap").is_dir() else Path(w)
        try:
            proj = SfmProject.load(root)
            cams = _read_cameras_bin(root / "sparse" / model / "cameras.bin")
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
        early = max(sols) < before_sol
        late = min(sols) >= before_sol
        rows[name] = {"sol_median": float(np.median(sols)), "early": bool(early), "late": bool(late),
                      "d": np.mean(d, axis=0).tolist()}
    e = [v["d"] for v in rows.values() if v["early"]]
    l_ = [v["d"] for v in rows.values() if v["late"]]
    start = (np.mean(e, axis=0) - (np.mean(l_, axis=0) if l_ else 0.0)).tolist() if e else [0.0] * 4
    return {"start": [float(x) for x in start], "blocks": rows, "early_blocks": len(e), "late_blocks": len(l_)}


def fit_early_offsets(rec, proj, temps, ppm: float, T0: float, xkw: Dict[str, Any], before_sol: float = EARLY_MISSION_SOL,
                      start=(0.0, 0.0, 0.0, 0.0), step: float = EARLY_STEP_PX, rounds: int = 4, iterations: int = 50,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    v0p70: fx, fy, cx, cy offsets (px, common to both eyes) of the Navcam images taken before ``before_sol``, in the
    consensus form.  Each evaluation is one joint adjustment (shared cameras, rig, poses and points re-converged, from
    the same state; ``rec`` is not changed) with the images' keypoints mapped by :func:`early_keypoint_map`.  A full
    quadratic response surface of the cost over the four offsets (15 adjustments: centre, +-``step`` per term, +step on
    each pair) gives the minimum and the covariance vf H^-1 (cost = chi2 / 2; vf the variance factor); ``rounds``
    re-centres the design on the minimum.  Significance: z = offset / sd per term, and the likelihood-ratio statistic
    2 (cost(0) - cost(best)) / vf against chi2 with 4 degrees of freedom for the four together.
    """
    import copy as _copy
    from scipy import stats
    from . import navcal as NC
    n_early = len(early_keypoint_map(proj, before_sol, np.zeros(4)).early_images)
    out: Dict[str, Any] = {"before_sol": float(before_sol), "terms": list(EARLY_TERMS), "early_images": n_early,
                           "step_px": float(step), "rounds": []}
    if not n_early:
        out["note"] = f"no images before sol {before_sol:g}: nothing to fit"
        return out
    cache: Dict[tuple, Dict[str, float]] = {}
    t_all = time.time()
    n_max = max(1, int(rounds)) * 16 + 2                  # 15 design points + 1 trial per round, + best and zero

    def cost_at(x):
        key = tuple(np.round(x, 6))
        if key not in cache:
            r = _copy.deepcopy(rec)
            t = time.time()
            ba = NC.joint_adjust(r, proj, temps, ppm, T0, max_iterations=iterations, rig_slopes=(0.0, 0.0),
                                 extra_map=early_keypoint_map(proj, before_sol, x), **xkw)
            red = max(2 * ba["observations"] - 1, 1)
            cache[key] = {"cost": float(ba["final_cost"]), "vf": float(2 * ba["final_cost"] / red),
                          "iterations": ba["iterations"]}
            if verbose:
                print("  early offsets " + " ".join(f"{k} {v:+.3f}" for k, v in zip(EARLY_TERMS, x))
                      + f": cost {ba['final_cost']:.2f} it {ba['iterations']} ({time.time() - t:.0f} s)  "
                      + NC.progress(len(cache), n_max, t_all, "adjustment (at most)"), flush=True)
        return cache[key]

    # Newton steps on the response surface, limited to ``max_move`` px per term (a trust region: far from the minimum
    # the adjustments re-converge along different paths and the surface is only roughly quadratic); a step is kept
    # only if it lowers the cost, else the region is halved.  The covariance comes from the last design (at the end).
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
        else:                                             # not convex here: a gradient step instead
            dx = -g / max(float(np.max(np.abs(np.diag(H)))), 1e-9)
        big = float(np.max(np.abs(dx)))
        if big > max_move:
            dx = dx * (max_move / big)
        rec_rd = {"centre": best.tolist(), "costs": c.tolist(), "gradient": g.tolist(), "hessian": H.tolist(),
                  "eigenvalues": ev.tolist(), "fit_rms": fit_rms, "step": dx.tolist(), "positive_definite": ok}
        out["rounds"].append(rec_rd)
        if verbose:
            print(f"  early offsets, round {rd + 1}/{rounds}: centre " + " ".join(f"{k} {v:+.3f}" for k, v in zip(EARLY_TERMS, best))
                  + "; step " + " ".join(f"{v:+.3f}" for v in dx) + (" (positive definite)" if ok else " (not convex)"),
                  flush=True)
        if ok and np.all(np.abs(dx) < 0.5 * step):        # the minimum lies well inside this design: done
            best = best + dx
            converged = True
            break
        trial = best + dx
        if cost_at(trial)["cost"] < c[0]:
            best = trial
        else:
            max_move *= 0.5
            rec_rd["rejected"] = True
            for d, cc in zip(D, c):                       # fall back to the best design point
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
        print("early-mission offsets (before sol %g, %d images): " % (before_sol, n_early)
              + ", ".join(f"{k} {out['offsets_px'][k]:+.3f} +- {out['sd_px'][k]:.3f} px (z {out['z'][k]:+.1f})"
                          for k in EARLY_TERMS)
              + f"; all four: LR {out['likelihood_ratio']:.1f}, p {out['p_value_all']:.2g}", flush=True)
    return out


def fit_consensus(scapes_cfg: Dict[str, PathLike], points: int = 8000, iterations: int = 100,
                  k4_zero: bool = True, start_dir: Optional[PathLike] = None, verbose: bool = True,
                  translation_scapes: Optional[Dict[str, PathLike]] = None, translation_sigma_m: float = 0.05,
                  apply_translation: bool = False, refit_thermal: bool = False,
                  thermal_terms: Optional[Tuple[str, ...]] = None, early_mission_sol: Optional[float] = None,
                  early_rounds: int = 2, early_start: Optional[Tuple[float, float, float, float]] = None,
                  form: str = "v0p73", thermal_rounds: int = 1, joint_translation: bool = False,
                  degeneracy_mm: float = 1.0, k1_slopes: Optional[Dict[str, float]] = None,
                  leave_one_out: bool = False, loo_iterations: int = 40, nl_pp_thermal: Optional[bool] = None,
                  early_hold: bool = False) -> Dict[str, Any]:
    """The joint adjustment (see the module docstring) of the blocks ``{name: WORK folder}``; returns the shared
    cameras, the rig, residuals by radius and the per-block image counts.  v0p62: ``translation_scapes`` (the large
    blocks, :func:`large_blocks`): then :func:`fit_rig_translation` on them with the new cameras held
    (``res["rig_translation"]``); ``apply_translation`` makes it the rig's translation (``rig_t``), which
    :func:`write_consensus` then writes with ``"translation": "fitted"`` for the projects to use.  v0p64:
    ``refit_thermal`` profiles the ``thermal_terms`` (``"f"`` ppm/degC, ``"NL_cx"``, ``"NR_cx"``, ``"NL_cy"``,
    ``"NR_cy"`` px/degC, about T0) one after the other in this same form (:func:`profile_thermal`), then adjusts once
    more at the best values; the result's ``thermal`` holds them with their sd and the profiles.  v0p70:
    ``early_mission_sol`` (e.g. 380): fx, fy, cx, cy offsets of the images before that sol (:func:`fit_early_offsets`,
    started from the start cameras' ``early_mission`` values), then the final adjustment with them; the result's
    ``early_mission`` holds the offsets, sd, z, p-values and the terms to apply (|z| >= ``EARLY_APPLY_Z``).  v0p74.4:
    ``joint_translation``: the final adjustment frees the rig translation (prior ``translation_sigma_m`` per axis on
    the right centre) TOGETHER with the cameras (NR cx included) and the rig rotation (yaw included), over every
    block, so the baseline is not estimated conditionally on them; the result's cameras and rig are that joint
    solution, and ``rig_translation_joint`` records it with a degeneracy profile (:func:`baseline_degeneracy`: the
    baseline held at +-``degeneracy_mm``, everything else re-converged - how much yaw and NR cx move per mm and
    what a mm costs).  The large-block fit (``translation_scapes``) is still made as a cross-check.
    v0p73 (``form="v0p73"``, the default): the NL-reference form (:mod:`mppp.sfm.cmods_form`) - fx and fy of each eye
    their own ppm/degC, the stereo temperature change on NR cx (a start with it on NL cx is converted), NL's principal
    point fixed in temperature, one rig yaw with a rate in sol (``"yaw_rate"``, mdeg per 1000 sol, in the drift);
    ``thermal_terms`` default :data:`V0P73_TERMS`.  ``form="v0p72"``: the earlier form (one ppm, NL cx, no yaw rate).

    v0p80 (``form="v0p80"``, the camera-model recipe of 14 Oct 2026): the v0p73 form with **k4 free** (k3 and k4 fitted
    as a pair; blocks then hold both at the consensus, NAVCAM_K4 = "consensus") and **k1 against temperature** per eye
    (``k1_slopes`` {"NL": per degC, "NR": per degC}, measured in the blocks' temperature bins -
    :func:`k1_slopes_from_blocks` - and applied per image through :func:`mppp.sfm.navcal.k1_keypoint_map`; default:
    the start cameras' ``thermal.k1_per_degC``).  NR cx(T) stays the one carrier of the stereo temperature change (NL
    cx slope 0, no rig temperature term).  ``leave_one_out``: after the final adjustment, the joint again without each
    block (``loo_iterations``, the thermal terms held), and how far each eye moves (:func:`navcal.compare`, rotation
    removed; centre / corner rms px) and the stereo terms (NR cx, rig yaw) - ``res["leave_one_out"]``, gated by
    :func:`check_candidate` (0.05 px centre, 0.1 px corners).

    v0p83 (Christian, 18 Oct 2026: the stereo temperature change on the principal points of both eyes, never a rig
    temperature term - a rig per temperature bin would be needed): ``nl_pp_thermal`` lets NL's principal point move
    with temperature too (``NL_cx`` / ``NL_cy`` profiled; the start's NL slope is kept, not moved to NR; the cameras
    are written with ``thermal.pp_thermal = "both"``, which :func:`mppp.sfm.cmods_form.check_form` accepts).  Default
    (None): on in the forms of :data:`NL_PP_THERMAL_FORMS` (v0p80) or when the start cameras carry
    ``pp_thermal = "both"``; the default terms then start with ``NL_cx``.  ``early_hold``: apply the start's early-mission offsets (or ``early_start``) without refitting them -
    a side study on the same blocks without the ~19 adjustments of a response-surface round."""
    import pycolmap
    from . import navcal as NC
    from .project import PARAM_NAMES, navcam_distortion_terms
    from ..paths import cmods_dir
    d = Path(start_dir) if start_dir else cmods_dir()
    nl = json.loads((d / "M2020_NL_fisheye_tangential.json").read_text(encoding="utf-8"))
    nr = json.loads((d / "M2020_NR_fisheye_tangential.json").read_text(encoding="utf-8"))
    rj = json.loads((d / "M2020_N_rig.json").read_text(encoding="utf-8"))
    cams = {k: pycolmap.Camera(model=j["model"], width=5120, height=3840, params=np.asarray(j["params"], float))
            for k, j in (("NL", nl), ("NR", nr))}
    rig = (np.asarray(rj["R_sensor_from_ref"], float), np.asarray(rj["t_sensor_from_ref_m"], float))
    if form not in ("v0p80", "v0p73", "v0p72"):
        raise ValueError(f"form {form!r}: 'v0p80' (recipe: k4 free, k1(T)), 'v0p73' (NL reference) or 'v0p72'")
    axis = form in ("v0p73", "v0p80")
    if form == "v0p80":
        k4_zero = False                                         # v0p80: k3 and k4 as a pair
    th = nl.get("thermal") or {}
    ppm, t0, pp_nl = float(th["ppm_per_degC"]), float(th["T0_degC"]), float(th.get("cx_px_per_degC") or 0.0)
    thr = nr.get("thermal") or {}
    tvals = {"f": ppm, "NL_cx": pp_nl, "NL_cy": float(th.get("cy_px_per_degC") or 0.0),
             "NR_cx": float(thr.get("cx_px_per_degC") or 0.0), "NR_cy": float(thr.get("cy_px_per_degC") or 0.0)}
    for eye, j in (("NL", th), ("NR", thr)):                        # v0p73: per eye and axis (else the shared ppm)
        tvals[f"{eye}_fx"] = float(j.get("fx_ppm_per_degC", j.get("ppm_per_degC", ppm)))
        tvals[f"{eye}_fy"] = float(j.get("fy_ppm_per_degC", j.get("ppm_per_degC", ppm)))
        # v0p80: k1 against temperature (the start cameras', or the blocks' bins through ``k1_slopes``)
        tvals[f"{eye}_k1"] = float(j.get("k1_per_degC") or 0.0) if form == "v0p80" else 0.0
    k1_source = "start cameras" if any(tvals[f"{e}_k1"] for e in ("NL", "NR")) else "none (0)"
    if k1_slopes is not None and form == "v0p80":
        for eye in ("NL", "NR"):
            tvals[f"{eye}_k1"] = float(k1_slopes.get(eye) or 0.0)
        k1_source = "temperature bins of the blocks (k1_slopes_from_blocks)"
    drift = copy.deepcopy(rj.get("drift"))
    tvals["yaw_rate"] = (1000.0 * float((drift or {}).get("yaw_mdeg_per_sol") or 0.0)
                         if (axis and rj.get("convention") == "NL reference") else 0.0)
    if nl_pp_thermal is None:                    # v0p83: the form (NL_PP_THERMAL_FORMS) or the start cameras say so
        nl_pp_thermal = axis and (form in NL_PP_THERMAL_FORMS or str(th.get("pp_thermal") or "") == "both")
    nl_pp_thermal = bool(nl_pp_thermal) and axis
    converted = None
    if axis and not nl_pp_thermal and (tvals["NL_cx"] or tvals["NL_cy"]):
        # v0p73: the start has the stereo temperature change on NL's principal point - move it to NR's
        ratio = float(nr["params"][0]) / float(nl["params"][0])
        converted = {"NL_cx": tvals["NL_cx"], "NL_cy": tvals["NL_cy"], "ratio": ratio}
        tvals["NR_cx"] -= tvals["NL_cx"] * ratio
        tvals["NR_cy"] -= tvals["NL_cy"] * ratio
        tvals["NL_cx"] = tvals["NL_cy"] = 0.0
    if thermal_terms is None:
        thermal_terms = (("NL_cx",) + V0P73_TERMS if nl_pp_thermal else V0P73_TERMS) if axis else ("f", "NL_cx")
    thermal_terms = tuple(thermal_terms)
    bad = [k for k in thermal_terms if k not in THERMAL_TERMS]
    if bad:
        raise ValueError(f"thermal_terms {bad}: use {THERMAL_TERMS}")
    if axis and not nl_pp_thermal and any(k in thermal_terms for k in ("NL_cx", "NL_cy")):
        raise ValueError("the v0p73 form keeps NL's principal point fixed in temperature (NL reference): fit NR_cx / "
                         "NR_cy, or pass nl_pp_thermal=True (v0p83: both eyes' principal points)")
    if not axis and any(k in thermal_terms for k in AXIS_TERMS + ("yaw_rate",)):
        raise ValueError(f"{AXIS_TERMS} and yaw_rate need form='v0p73'")
    if drift and not axis:
        for k in ("yaw_mdeg_per_sol", "yaw_early_mdeg_per_sol", "yaw_fit_mdeg_per_sol"):
            if k in drift:
                drift[k] = 0.0
    t = time.time()
    rec, proj, idx = NC.merge_scapes((NC.load_scape(n, scapes_cfg[n]) for n in scapes_cfg), cams, rig,
                                     points_per_scape=points, seed=0)
    temps = idx["temps"]
    if verbose:
        print(f"merged {len(idx['images'])} blocks, {rec.num_reg_images()} images, {rec.num_points3D()} points "
              f"({time.time() - t:.0f} s)", flush=True)
        if idx.get("duplicates"):
            print("  images also in an earlier block (kept, counted twice): "
                  + ", ".join(f"{k} {v}" for k, v in idx["duplicates"].items()), flush=True)
        # v0p71: the plan of the run, in joint adjustments (each about 1-3 min for ~15 blocks)
        n_adj = 1 + (len(thermal_terms) * 5 * max(1, int(thermal_rounds)) + 1 if refit_thermal else 0) \
            + (((max(1, int(early_rounds)) * 16 + 3) if not early_hold else 1) if early_mission_sol else 0)
        print(f"{datetime.datetime.now():%H:%M} plan: about {n_adj} joint adjustments (start"
              + (f", thermal profiles {', '.join(thermal_terms)}" if refit_thermal else "")
              + ((", early-mission offsets held" if early_hold else f", early-mission offsets up to {early_rounds} rounds of 16")
                 if early_mission_sol else "")
              + ", final with covariance" + (", rig translation" if translation_scapes else "") + ")", flush=True)
        if converted:
            print(f"v0p73 NL-reference form: the start's NL cx slope {converted['NL_cx']:+.4f} px/degC moved to NR "
                  f"(NR cx {tvals['NR_cx']:+.4f} px/degC)", flush=True)
        if axis:
            print(f"{form} form: fx, fy per eye against temperature, "
                  + ("NL and NR cx against temperature (v0p83)" if nl_pp_thermal else
                     "NR cx against temperature, NL principal point fixed")
                  + f", one rig yaw with a sol rate (start {tvals['yaw_rate']:+.2f} mdeg/1000 sol)"
                  + (f"; k4 free; k1 per degC NL {tvals['NL_k1']:+.2e}, NR {tvals['NR_k1']:+.2e} ({k1_source})"
                     if form == "v0p80" else ""), flush=True)
    proj.settings["navcam_rig_yaw"] = "refine"            # one yaw for all: no drift / temperature term below
    names = PARAM_NAMES["THIN_PRISM_FISHEYE"]
    for cid, key in ((1, "NL"), (2, "NR")):
        c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
        c = navcam_distortion_terms(c, "refine", k4="zero" if k4_zero else "consensus")
        proj.cameras[key] = c
        rec.cameras[cid].params = np.asarray(c["params"], float)
    xkw: Dict[str, Any] = _joint_kw(tvals, drift, axis)
    t = time.time()
    ba = NC.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=not refit_thermal,
                         rig_slopes=(0.0, 0.0), **xkw)
    profiles = {}
    if refit_thermal:
        # v0p64: the thermal terms in the consensus form, one profile after the other from the converged state
        if verbose:
            print(f"{datetime.datetime.now():%H:%M} consensus at the start values: cost {ba['final_cost']:.1f} ({time.time() - t:.0f} s); "
                  f"thermal profiles: {', '.join(thermal_terms)}", flush=True)
        history = []
        for rnd in range(max(1, int(thermal_rounds))):     # v0p73: the terms trade a little - profile them again
            if verbose and thermal_rounds > 1:
                print(f"{datetime.datetime.now():%H:%M} thermal profiles, round {rnd + 1} of {thermal_rounds}", flush=True)
            for term in thermal_terms:
                pr = profile_thermal(rec, proj, temps, tvals, t0, term, drift=drift, verbose=verbose, axis=axis)
                pr["round"] = rnd + 1
                profiles[term] = pr
                tvals = _set_term(tvals, term, pr["best"])
            history.append({k: tvals[k] for k in thermal_terms})
        if len(history) > 1 and verbose:
            print("change in the last round: " + ", ".join(f"{k} {history[-1][k] - history[-2][k]:+.4f}"
                                                          for k in thermal_terms), flush=True)
        ppm = tvals["f"]
        xkw = _joint_kw(tvals, drift, axis)
        t = time.time()
        ba = NC.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True,
                             rig_slopes=(0.0, 0.0), **xkw)
    early = None
    if early_mission_sol:
        # v0p70: the early-mission offsets in the consensus form, then the consensus once more with them
        em0 = nl.get("early_mission") or {}
        if early_start is not None:                 # v0p72: e.g. the offsets of a previous run
            x0 = [float(v) for v in early_start]
        elif em0:
            x0 = [float(em0.get(f"d{k}_px", 0.0)) for k in EARLY_TERMS]
        else:                                       # the block alignments' own (per-block) estimate as the start
            x0 = early_start_from_blocks(scapes_cfg, float(early_mission_sol))["start"]
        if verbose:
            print(f"{datetime.datetime.now():%H:%M} early-mission offsets before sol {early_mission_sol:g} (start "
                  + " ".join(f"{k} {v:+.3f}" for k, v in zip(EARLY_TERMS, x0)) + "):", flush=True)
        if early_hold:                              # v0p83: held at the start's offsets, not refitted
            n_e = len(early_keypoint_map(proj, float(early_mission_sol), np.zeros(4)).early_images)
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
            t = time.time()
            ba = NC.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True,
                                 rig_slopes=(0.0, 0.0), extra_map=early_keypoint_map(proj, float(early_mission_sol), xb),
                                 **xkw)
    joint_tr = None
    if joint_translation:                          # v0p74.4: translation, cameras and rotation together
        _emap = (early_keypoint_map(proj, float(early_mission_sol), [early["offsets_px"][k] for k in EARLY_TERMS])
                 if early and early.get("offsets_px") else None)
        t = time.time()
        _, _, T_before = NC.stereo_rig(rec)
        t_before = np.array(T_before.translation, float)        # a copy: .translation is a view into the rig
        ba = NC.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True,
                             rig_slopes=(0.0, 0.0), refine_rig=True, rig_translation_sigma_m=float(translation_sigma_m),
                             extra_map=_emap, **xkw)
        _, _, T_after = NC.stereo_rig(rec)
        joint_tr = {"t_start_m": t_before.tolist(), "t_m": np.asarray(T_after.translation, float).tolist(),
                    "baseline_start_m": float(np.linalg.norm(t_before)),
                    "baseline_m": float(np.linalg.norm(np.asarray(T_after.translation, float))),
                    "sigma_m": float(translation_sigma_m), "final_cost": float(ba["final_cost"]),
                    "iterations": ba["iterations"], "seconds": time.time() - t,
                    "rig_abs_mdeg": NC.rig_angles(np.asarray(T_after.rotation.matrix())),
                    "NR_cx": float(rec.cameras[2].params[2]), "NL_cx": float(rec.cameras[1].params[2])}
        joint_tr["dbaseline_mm"] = 1e3 * (joint_tr["baseline_m"] - joint_tr["baseline_start_m"])
        if verbose:
            print(f"{datetime.datetime.now():%H:%M} joint rig: translation free with the cameras and the rotation: baseline "
                  f"{joint_tr['baseline_m']:.5f} m ({joint_tr['dbaseline_mm']:+.2f} mm), yaw "
                  f"{joint_tr['rig_abs_mdeg']['yaw_mdeg']:+.2f} mdeg, NR cx {joint_tr['NR_cx']:.3f} px, cost "
                  f"{ba['final_cost']:.1f} ({joint_tr['seconds']:.0f} s)", flush=True)
        if degeneracy_mm:
            joint_tr["degeneracy"] = baseline_degeneracy(rec, proj, temps, ppm, t0, xkw, _emap, float(degeneracy_mm),
                                                         iterations=iterations, verbose=verbose)
    loo = None
    if leave_one_out:                              # v0p80: each block left out once, the thermal terms held
        _emap = (early_keypoint_map(proj, float(early_mission_sol), [early["offsets_px"][k] for k in EARLY_TERMS])
                 if early and early.get("offsets_px") else None)
        loo = leave_one_block_out(rec, proj, idx, temps, ppm, t0, xkw, extra_map=_emap, iterations=loo_iterations,
                                  verbose=verbose)
    if axis:                                       # v0p73: the mean of the four per-axis slopes, for older readers
        ppm = float(np.mean([tvals[k] for k in AXIS_TERMS]))
    ks = NC.keypoint_scales(proj, temps, ppm, t0, axis_ppm=xkw.get("axis_ppm"))
    kmap = NC.thermal_keypoint_map(proj, temps, t0, (0.0, 0.0), xkw["pp_slopes"], xkw.get("drift"))
    if xkw.get("k1_slopes"):                      # v0p80: k1(T) as the joint applied it
        _k1 = NC.k1_keypoint_map(proj, temps, t0, xkw["k1_slopes"])
        _kb = kmap
        kmap = lambda iid, k, cam: _k1(iid, _kb(iid, k, cam), cam)          # noqa: E731
    if early and early.get("offsets_px"):
        _em = early_keypoint_map(proj, float(early_mission_sol), [early["offsets_px"][k] for k in EARLY_TERMS])
        _km = kmap
        kmap = lambda iid, k, cam: _em(iid, _km(iid, k, cam), cam)          # noqa: E731
    st = radial_stats(rec, proj, ks, kmap)
    _, _, T = NC.stereo_rig(rec)
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
            else:                                       # the tangent space: report it by position
                sd[key] = {"diag": [float(x) for x in dg], "free_guess": free}
    out = {"final_cost": float(ba["final_cost"]), "iterations": ba["iterations"], "variance_factor": vf,
           "observations": ba.get("observations"), "brief": ba["brief"], "seconds": time.time() - t,
           "NL": dict(zip(names, map(float, rec.cameras[1].params))),
           "NR": dict(zip(names, map(float, rec.cameras[2].params))), "sd": sd,
           "covariance_blocks": sorted(blocks), "covariance_error": cov.get("error"),
           "covariance_damping": cov.get("damping"),
           "rig_abs_mdeg": NC.rig_angles(np.asarray(T.rotation.matrix())),
           "rig_R": np.asarray(T.rotation.matrix()).tolist(), "rig_t": np.asarray(T.translation).tolist(),
           "stereo": NC.stereo_offset(rec.cameras[1], rec.cameras[2], T), "residuals": st,
           "blocks": {k: len(v) for k, v in idx["images"].items()}, "start_dir": str(d),
           "thermal": {"ppm_per_degC": ppm, "T0_degC": t0, "cx_px_per_degC_NL": tvals["NL_cx"],
                       "pp_slopes": {"NL": [tvals["NL_cx"], tvals["NL_cy"]], "NR": [tvals["NR_cx"], tvals["NR_cy"]]},
                       "axis_ppm": ({k: list(v) for k, v in _axis_ppm(tvals).items()} if axis else None),
                       "yaw_rate_mdeg_per_1000sol": tvals["yaw_rate"] if axis else None,
                       "k1_slopes": ({"NL": tvals["NL_k1"], "NR": tvals["NR_k1"]} if form == "v0p80" else None),
                       "k1_source": k1_source if form == "v0p80" else None,
                       "pp_thermal": "both" if nl_pp_thermal else ("NR" if axis else None),
                       "refit": bool(refit_thermal), "terms": list(thermal_terms) if refit_thermal else [],
                       "rounds": int(thermal_rounds) if refit_thermal else 0,
                       "sd": {k: profiles[k]["sd"] for k in profiles},
                       "z": {k: profiles[k].get("z") for k in profiles}, "converted_start": converted,
                       "profiles": profiles}, "drift": xkw.get("drift") if axis else drift,
           "form": form, "convention": "NL reference" if axis else None,
           "k4_zero": bool(k4_zero), "early_mission": early, "leave_one_out": loo}
    if joint_tr is not None:
        out["rig_translation_joint"] = joint_tr
    if translation_scapes:
        tr = fit_rig_translation(translation_scapes, out, points=points, sigma_m=translation_sigma_m,
                                 iterations=iterations, verbose=verbose)
        out["rig_translation"] = tr
        if joint_tr is not None and verbose:
            print(f"rig translation: joint {joint_tr['baseline_m']:.5f} m vs large blocks alone (cameras held) "
                  f"{tr['baseline_m']:.5f} m: {1e3 * (joint_tr['baseline_m'] - tr['baseline_m']):+.2f} mm", flush=True)
    if translation_scapes or joint_tr is not None:
        out["rig_translation_applied"] = bool(apply_translation)
        out["rig_translation_source"] = "joint" if joint_tr is not None else "large blocks (cameras held)"
        if apply_translation and joint_tr is None:
            out["rig_t"] = list(tr["t_m"])        # the joint solution is already in rig_t / rig_R / the cameras
    if verbose:
        print(f"consensus: cost {out['final_cost']:.1f}, it {out['iterations']}, median {st['median_px']:.4f} / rms "
              f"{st['rms_px']:.4f} px, corners {st['radial'][-1]['median_px'] or float('nan'):.3f} px, rig yaw "
              f"{out['rig_abs_mdeg']['yaw_mdeg']:+.2f} mdeg  ({out['seconds']:.0f} s)", flush=True)
    return out


# ------------------------------------------------------------------------- v0p80: the recipe's joint-side tools
LOO_LIMITS = {"centre_px": 0.05, "corner_px": 0.10}              # v0p80: leave-one-block-out, per eye (rotation removed)
PASS_LIMITS = {"f_ppm": 20.0, "pp_px": 0.05, "corner_px": 0.02, "rig_mdeg": 1.0}   # v0p80: converged between passes
K1_APPLY_Z = 3.0                    # v0p80: a k1 temperature slope from the bins is applied when |slope / se| >= this


def leave_one_block_out(rec, proj, idx: Dict[str, Any], temps, ppm: float, T0: float, xkw: Dict[str, Any],
                        extra_map=None, iterations: int = 40, verbose: bool = True) -> Dict[str, Any]:
    """
    v0p80: the joint without each block in turn (:func:`navcal.subset`), started from the converged joint with the
    thermal terms (``xkw``) held, and how far each eye's camera moves: :func:`navcal.compare` (the best rotation
    removed - a pointing change is absorbed by the frame poses) centre / corner rms px, the NR cx and NL cx change,
    the rig yaw / pitch change (mdeg) and the stereo pointing change NR cx - NL cx x fR/fL + fR yaw (px).  Returns
    {"blocks": {name: {...}}, "max": {...}, "limits": LOO_LIMITS, "ok"}.
    """
    from . import navcal as NC
    full = NC.shared_state(rec)
    cams_full = {k: NC.pycolmap_camera(full[k]["model"], full[k]["params"]) for k in ("NL", "NR")}
    R_full = np.asarray(full["rig_R"], float)
    rows: Dict[str, Any] = {}
    t_all = time.time()
    names = list(idx["images"])
    for kb, name in enumerate(names, 1):
        keep = [n for n in names if n != name]
        r = NC.subset(rec, idx, keep)
        t = time.time()
        try:
            ba = NC.joint_adjust(r, proj, temps, ppm, T0, max_iterations=iterations, rig_slopes=(0.0, 0.0),
                                 extra_map=extra_map, **xkw)
        except Exception as e:                                         # noqa: BLE001
            rows[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        st = NC.shared_state(r)
        row: Dict[str, Any] = {"iterations": ba["iterations"], "seconds": time.time() - t}
        for k in ("NL", "NR"):
            d = NC.compare(cams_full[k], NC.pycolmap_camera(st[k]["model"], st[k]["params"]))
            row[k] = {"centre_px": d["centre_rms_px"], "corner_px": d["corner_rms_px"], "rms_px": d["rms_px"],
                      "dcx_px": float(st[k]["params"][2] - full[k]["params"][2]),
                      "dcy_px": float(st[k]["params"][3] - full[k]["params"][3])}
        ang = NC.rig_angles(np.asarray(st["rig_R"], float), R_full)
        ratio = float(full["NR"]["params"][0]) / float(full["NL"]["params"][0])
        fR = float(full["NR"]["params"][0])
        row["rig_yaw_mdeg"], row["rig_pitch_mdeg"] = float(ang["yaw_mdeg"]), float(ang["pitch_mdeg"])
        row["stereo_px"] = float(row["NR"]["dcx_px"] - ratio * row["NL"]["dcx_px"]
                                 + fR * np.radians(1e-3 * row["rig_yaw_mdeg"]))
        rows[name] = row
        if verbose:
            print(f"  leave out {name}: NL centre {row['NL']['centre_px']:.3f} corner {row['NL']['corner_px']:.3f}, NR "
                  f"centre {row['NR']['centre_px']:.3f} corner {row['NR']['corner_px']:.3f} px, stereo "
                  f"{row['stereo_px']:+.3f} px  {NC.progress(kb, len(names), t_all, 'leave-one-out')}", flush=True)
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
    """
    v0p80: k1 against camera temperature per Navcam eye from the temperature bins of finished blocks
    (``colmap/project.json`` ``settings.thermal.rows`` with k1 refined, MPPP >= 0.75.1): k1 = a_block,eye + s_eye T,
    weighted by the bin's observations (relative; se from the scatter of the residuals), one offset per block and eye
    so that only the within-block change counts.  ``apply`` {"NL", "NR"}: the slope where |z| >= ``z_apply``, else 0.
    """
    from .thermal import eye_key
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
        print("k1 against temperature from the blocks' bins: " + "; ".join(
            f"{e} {v['slope']:+.2e} +- {v['se']:.1e} /degC (z {v['z']:+.1f}, {v['n']} bins)" if v.get("slope") is not None
            else f"{e} -" for e, v in out["fits"].items()) + f" -> applied NL {out['apply']['NL']:+.2e}, NR "
            f"{out['apply']['NR']:+.2e}", flush=True)
    return out


def consensus_change(a_dir: PathLike, b_dir: PathLike) -> Dict[str, Any]:
    """
    v0p80: how much the Navcam models changed between two consensus passes (folders with the camera and rig files):
    fx, fy per eye in ppm (at T0), cx, cy per eye in px, the distortion's corner change in px (:func:`navcal.compare`
    with fx, fy, cx, cy of ``b`` set to ``a``'s, rotation removed), the rig rotation in mdeg, and whether every one is
    below :data:`PASS_LIMITS` (``converged``: stop iterating blocks -> joint).
    """
    from . import navcal as NC
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    out: Dict[str, Any] = {"a": str(a_dir), "b": str(b_dir), "limits": dict(PASS_LIMITS)}
    worst = {"f_ppm": 0.0, "pp_px": 0.0, "corner_px": 0.0}
    for eye in ("NL", "NR"):
        A = json.loads((a_dir / f"M2020_{eye}_fisheye_tangential.json").read_text(encoding="utf-8"))
        B = json.loads((b_dir / f"M2020_{eye}_fisheye_tangential.json").read_text(encoding="utf-8"))
        pa, pb = np.asarray(A["params"], float), np.asarray(B["params"], float)
        e = {"dfx_ppm": 1e6 * (pb[0] / pa[0] - 1), "dfy_ppm": 1e6 * (pb[1] / pa[1] - 1),
             "dcx_px": float(pb[2] - pa[2]), "dcy_px": float(pb[3] - pa[3])}
        pm = pb.copy()
        pm[:4] = pa[:4]
        d = NC.compare(NC.pycolmap_camera(A["model"], pa), NC.pycolmap_camera(B["model"], pm))
        e["distortion_corner_px"] = float(d["corner_rms_px"])
        out[eye] = e
        worst["f_ppm"] = max(worst["f_ppm"], abs(e["dfx_ppm"]), abs(e["dfy_ppm"]))
        worst["pp_px"] = max(worst["pp_px"], abs(e["dcx_px"]), abs(e["dcy_px"]))
        worst["corner_px"] = max(worst["corner_px"], e["distortion_corner_px"])
    ra = json.loads((a_dir / "M2020_N_rig.json").read_text(encoding="utf-8"))
    rb = json.loads((b_dir / "M2020_N_rig.json").read_text(encoding="utf-8"))
    worst["rig_mdeg"] = _rotation_mdeg(ra["R_sensor_from_ref"], rb["R_sensor_from_ref"])
    out["worst"] = worst
    out["converged"] = all(worst[k] <= PASS_LIMITS[k] for k in PASS_LIMITS)
    return out


def block_span_m(root: PathLike) -> float:
    """v0p62: the largest horizontal distance between the Navcam station centres of a block (its waypoint priors)."""
    from .project import SfmProject
    from .reconstruction import navcam_network
    root = Path(root)
    if (root / "colmap").is_dir():
        root = root / "colmap"
    return float(navcam_network(SfmProject.load(root))["span_m"])


def large_blocks(scapes_cfg: Dict[str, PathLike], min_span_m: float = 30.0) -> Dict[str, Any]:
    """v0p62: the blocks of ``scapes_cfg`` whose stations span at least ``min_span_m`` (``{name: [folder, span]}``)."""
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
    """
    v0p74.4: how the rest of the model compensates a baseline change.  From the converged joint solution (``rec``
    is not changed) the rig translation is held at its length +- ``delta_mm`` (direction kept) and everything else
    - cameras, rig rotation, poses, points - re-converged.  Returns the cost rise per mm^2 (the curvature: with the
    variance factor, the baseline's sd), and the yaw, pitch and NR / NL cx change per mm (the degenerate
    direction).  Yaw and NR cx moving by their own sd or more per mm means a baseline error hides in them.
    """
    import copy as _copy
    from . import navcal as NC
    import pycolmap
    rid, cid, T = NC.stereo_rig(rec)
    t_joint = np.array(T.translation, float)
    R_joint = np.array(T.rotation.matrix())
    b0 = float(np.linalg.norm(t_joint))
    ang0 = NC.rig_angles(R_joint)
    base = {"cost": None, "yaw_mdeg": ang0["yaw_mdeg"], "pitch_mdeg": ang0["pitch_mdeg"],
            "NR_cx": float(rec.cameras[2].params[2]), "NL_cx": float(rec.cameras[1].params[2])}
    rows = []
    for sgn in (+1.0, -1.0):
        r = _copy.deepcopy(rec)
        t_new = t_joint * (b0 + sgn * delta_mm * 1e-3) / b0
        sid = [s for s in r.rigs[rid].non_ref_sensors if int(s.id) == cid][0]
        r.rigs[rid].set_sensor_from_rig(sid, pycolmap.Rigid3d(pycolmap.Rotation3d(R_joint), t_new))
        t = time.time()
        ba = NC.joint_adjust(r, proj, temps, ppm, T0, max_iterations=iterations, rig_slopes=(0.0, 0.0),
                             refine_rig="rotation", extra_map=extra_map, **xkw)
        _, _, T1 = NC.stereo_rig(r)
        ang = NC.rig_angles(np.asarray(T1.rotation.matrix()))
        rows.append({"dbaseline_mm": sgn * delta_mm, "cost": float(ba["final_cost"]), "iterations": ba["iterations"],
                     "yaw_mdeg": ang["yaw_mdeg"], "pitch_mdeg": ang["pitch_mdeg"],
                     "NR_cx": float(r.cameras[2].params[2]), "NL_cx": float(r.cameras[1].params[2]),
                     "seconds": time.time() - t})
        if verbose:
            print(f"  baseline {sgn * delta_mm:+.1f} mm: cost {ba['final_cost']:.1f}, yaw {ang['yaw_mdeg']:+.2f} mdeg, "
                  f"NR cx {rows[-1]['NR_cx']:.3f} px ({rows[-1]['seconds']:.0f} s)", flush=True)
    plus, minus = rows
    # the joint cost at the minimum: the mean of the two is an upper bound; the curvature from the two points about
    # the minimum assumed at 0 is (c+ + c-) / 2 - c0 ... without c0 use the symmetric estimate from the pair
    out = {"delta_mm": delta_mm, "baseline_m": b0, "rows": rows, "at_joint": base,
           "yaw_mdeg_per_mm": (plus["yaw_mdeg"] - minus["yaw_mdeg"]) / (2 * delta_mm),
           "pitch_mdeg_per_mm": (plus["pitch_mdeg"] - minus["pitch_mdeg"]) / (2 * delta_mm),
           "NR_cx_px_per_mm": (plus["NR_cx"] - minus["NR_cx"]) / (2 * delta_mm),
           "NL_cx_px_per_mm": (plus["NL_cx"] - minus["NL_cx"]) / (2 * delta_mm),
           "cost_rise_mean": 0.5 * (plus["cost"] + minus["cost"]),
           "note": "yaw / cx per mm of baseline with everything else re-converged; cost_rise_mean - the joint cost "
                   "(res['final_cost']) is the rise for +-delta: sd_baseline_mm ~ delta / sqrt(2 rise / variance factor)"}
    return out


def fit_rig_translation(scapes_cfg: Dict[str, PathLike], res: Dict[str, Any], points: int = 8000,
                        sigma_m: float = 0.05, iterations: int = 100, verbose: bool = True) -> Dict[str, Any]:
    """
    v0p62: the Navcam rig translation (the right camera centre in the left camera frame) from the large blocks.
    Over tens to hundreds of metres the waypoint priors fix the block scale independently of the 0.42 m baseline, so
    the baseline length and direction can be separated from the waypoint scale; on small blocks they cannot.  The
    blocks of ``scapes_cfg`` are merged as in :func:`fit_consensus` with the cameras of ``res`` (the consensus) held,
    the rig yaw held, pitch, roll and the translation refined (a weak prior of ``sigma_m`` per axis on the right
    centre, i.e. free at the mm level).  Returns the start and fitted translation, the change of the right centre
    and the baseline in mm with formal sd (x variance factor; the true error is probably 2-10 times larger), and
    the blocks used.
    """
    import pycolmap
    from . import navcal as NC
    from .project import navcam_distortion_terms
    cams = {k: pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840,
                               params=np.array([res[k][n] for n in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2",
                                                                    "k3", "k4", "sx1", "sy1")], float))
            for k in ("NL", "NR")}
    R0 = np.asarray(res["rig_R"], float)
    t0 = np.asarray(res["rig_t"], float)
    th = res["thermal"]
    t = time.time()
    rec, proj, idx = NC.merge_scapes((NC.load_scape(n, scapes_cfg[n]) for n in scapes_cfg), cams, (R0, t0),
                                     points_per_scape=points, seed=0)
    temps = idx["temps"]
    if verbose:
        print(f"rig translation: merged {len(idx['images'])} large blocks, {rec.num_reg_images()} images "
              f"({time.time() - t:.0f} s)", flush=True)
    proj.settings["navcam_rig_yaw"] = "hold"
    for cid, key in ((1, "NL"), (2, "NR")):
        c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
        proj.cameras[key] = navcam_distortion_terms(c, "refine", k4="zero" if res.get("k4_zero", True) else "consensus")
    xkw: Dict[str, Any] = {"pp_slopes": _pp_slopes(_thermal_values(th)), "rig_translation_sigma_m": float(sigma_m)}
    if res.get("drift"):
        xkw["drift"] = res["drift"]
    if (th.get("k1_slopes") or {}) and any((th.get("k1_slopes") or {}).values()):     # v0p80
        xkw["k1_slopes"] = dict(th["k1_slopes"])
    t = time.time()
    ba = NC.joint_adjust(rec, proj, temps, float(th["ppm_per_degC"]), float(th["T0_degC"]), max_iterations=iterations,
                         covariance=True, refine_rig=True, rig_slopes=(0.0, 0.0), hold_cameras=("NL", "NR"), **xkw)
    _, _, T = NC.stereo_rig(rec)
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
           "rig_abs_mdeg": NC.rig_angles(R1), "seconds": time.time() - t}
    if verbose:
        print(f"rig translation ({len(out['blocks'])} large blocks): right centre change [mm] "
              f"{np.round(out['dcentre_mm'], 3).tolist()} (x along the baseline), sd [mm] "
              f"{None if sd_t is None else np.round(sd_t, 3).tolist()}, baseline {out['baseline_m']:.5f} m "
              f"({out['dbaseline_mm']:+.2f} mm)  ({out['seconds']:.0f} s)", flush=True)
    return out


def write_consensus(res: Dict[str, Any], out_dir: PathLike, note: str = "",
                    template_dir: Optional[PathLike] = None) -> Path:
    """The candidate folder of a :func:`fit_consensus` result: the camera and rig files in use (``template_dir``,
    default ``cmods_dir()``) with the new parameters, rig rotation and notes; sd from the joint's covariance where
    it has them (else the template's)."""
    from scipy.spatial.transform import Rotation
    from ..paths import cmods_dir
    tdir = Path(template_dir) if template_dir else cmods_dir()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = sorted(res["blocks"])
    stamp = datetime.date.today().isoformat()
    a = res["rig_abs_mdeg"]
    for eye in ("NL", "NR"):
        c = json.loads((tdir / f"M2020_{eye}_fisheye_tangential.json").read_text(encoding="utf-8"))
        c["params"] = [float(res[eye][n]) for n in c["param_names"]]
        c["fixed_params"] = ["k4"] if res.get("k4_zero", True) else []
        rt = res["thermal"]
        form = res.get("form") or "v0p72"
        v73 = form in ("v0p73", "v0p80")                     # v0p80: the v0p73 form + k4 free + k1(T)
        if v73:
            tv = _thermal_values(rt)
            k4txt = "k4 = 0 (held)" if res.get("k4_zero", True) else f"k4 {float(res[eye]['k4']):+.5f} (fitted with k3)"
            k1txt = (f", k1 {tv[eye + '_k1']:+.2e}/degC" if tv.get(eye + "_k1") else "")
            c["source"] = (f"MPPP Navcam consensus (mppp.sfm.navcal_consensus, {stamp}, {form} NL-reference form): one "
                           f"{eye} camera shared by {len(blocks)} blocks ({', '.join(blocks)}); fisheye + tangential at "
                           f"{rt['T0_degC']:g} degC, fx {tv[eye + '_fx']:+.1f} / fy {tv[eye + '_fy']:+.1f} ppm/degC, "
                           f"cx {tv[eye + '_cx']:+.4f} px/degC{k1txt}, {k4txt}, one rig yaw {a['yaw_mdeg']:+.2f} mdeg "
                           f"(rate {tv['yaw_rate']:+.2f} mdeg/1000 sol). {note}".strip())
        else:
            c["source"] = (f"MPPP Navcam consensus (mppp.sfm.navcal_consensus, {stamp}): one {eye} camera shared by "
                           f"{len(blocks)} blocks ({', '.join(blocks)}); fisheye + tangential at "
                           f"{rt['T0_degC']:g} degC, f {rt['ppm_per_degC']:+.1f} ppm/degC, NL cx "
                           f"{rt['cx_px_per_degC_NL']:+.4f} px/degC "
                           f"({'refitted' if rt.get('refit') else 'held'}), k4 = 0 (held), one rig yaw "
                           f"{a['yaw_mdeg']:+.2f} mdeg. {note}".strip())
        if rt.get("refit") or v73:                         # v0p64: the refitted thermal terms of this consensus
            tv = _thermal_values(rt)
            sdt = rt.get("sd") or {}
            th0 = dict(c.get("thermal") or {})
            th0.update({"ppm_per_degC": tv["f"], "T0_degC": float(rt["T0_degC"]),
                        "cx_px_per_degC": tv[f"{eye}_cx"], "cy_px_per_degC": tv[f"{eye}_cy"],
                        "note": f"v0p64: refitted with the consensus ({stamp}; profiles of {', '.join(rt.get('terms') or [])})"})
            for k, kk in (("f", "sd_ppm_per_degC"), (f"{eye}_cx", "sd_cx_px_per_degC"), (f"{eye}_cy", "sd_cy_px_per_degC"),
                          (f"{eye}_fx", "sd_fx_ppm_per_degC"), (f"{eye}_fy", "sd_fy_ppm_per_degC")):
                if k in sdt:
                    th0[kk] = float(sdt[k])
            if v73:                                         # v0p73: fx and fy each their own slope
                th0.update({"fx_ppm_per_degC": tv[f"{eye}_fx"], "fy_ppm_per_degC": tv[f"{eye}_fy"],
                            "ppm_per_degC": 0.5 * (tv[f"{eye}_fx"] + tv[f"{eye}_fy"]), "form": form,
                            "note": f"{form}: NL-reference form, refitted with the consensus ({stamp}; profiles of "
                                    f"{', '.join(rt.get('terms') or []) or 'none'})"})
                th0.pop("cx_px_per_degC_NL", None)
                if rt.get("pp_thermal") == "both":          # v0p83: both principal points move with temperature
                    th0["pp_thermal"] = "both"
                else:
                    th0.pop("pp_thermal", None)
                if form == "v0p80":                         # v0p80: k1 against temperature (from the blocks' bins)
                    th0["k1_per_degC"] = float(tv.get(f"{eye}_k1") or 0.0)
                    th0["k1_source"] = rt.get("k1_source")
            c["thermal"] = th0
        if v73:
            c["convention"] = "NL reference"
            c["form"] = form
        em = res.get("early_mission")
        if em and em.get("offsets_px"):                    # v0p70: the early-mission offsets (both eyes)
            c["early_mission"] = {"before_sol": em["before_sol"],
                                  **{f"d{k}_px": float(em["offsets_px"][k]) for k in EARLY_TERMS},
                                  "sd_px": em.get("sd_px"), "z": em.get("z"), "p_value": em.get("p_value"),
                                  "p_value_all": em.get("p_value_all"), "applied_terms": em.get("applied_terms", []),
                                  "early_images": em.get("early_images"),
                                  "note": f"v0p70 ({stamp}): a camera with these offsets for the images before sol "
                                          f"{em['before_sol']:g}; projects apply the terms in applied_terms "
                                          f"(|z| >= {EARLY_APPLY_Z:g}), weighted by the block's share of early images"}
        elif "early_mission" in c and em is not None:
            c.pop("early_mission", None)
        sde = (res.get("sd") or {}).get(eye) or {}
        if sde and "diag" not in sde:
            c["sd"] = {k: v for k, v in {**(c.get("sd") or {}), **sde}.items() if k not in c["fixed_params"]}
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
        (out / f"M2020_{eye}_fisheye_tangential.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    r = json.loads((tdir / "M2020_N_rig.json").read_text(encoding="utf-8"))
    R = np.asarray(res["rig_R"], float)
    r["R_sensor_from_ref"] = R.tolist()
    r["rotvec_rad"] = Rotation.from_matrix(R).as_rotvec().tolist()
    if res.get("drift"):
        r["drift"] = res["drift"]
    r["yaw_constant"] = True
    r.pop("thermal", None)
    v73 = res.get("form") in ("v0p73", "v0p80")
    if v73:                                               # v0p73: NL reference, a yaw rate in sol
        r["convention"] = "NL reference"
        r["form"] = res.get("form")
        yr = (res.get("thermal") or {}).get("yaw_rate_mdeg_per_1000sol") or 0.0
        sdy = ((res.get("thermal") or {}).get("sd") or {}).get("yaw_rate")
        r["yaw_model"] = (f"one yaw for every temperature (the stereo temperature change is NR cx's); yaw rate "
                          f"{yr:+.2f}" + (f" +- {sdy:.2f}" if sdy else "") + " mdeg per 1000 sol")
    tr = res.get("rig_translation")
    jt = res.get("rig_translation_joint")                        # v0p74.4: the joint fit, when made, is the one applied
    applied = bool((tr or jt) and res.get("rig_translation_applied"))
    # v0p73: no new fit - keep the fitted one in use; v0p85: also when a new fit is only recorded (the v0p82 pipeline
    # never applies one - the baseline stage carries it): the 18 Oct pass-2 candidate had lost "translation": "fitted",
    # which would have sent every project back to its own CAHV baseline
    carried = not applied and r.get("translation") == "fitted"
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
                 + ("temperature term (NR cx carries it), the yaw follows its sol rate; " if v73 else
                    "temperature term, the drift's yaw rate 0; ")
                 + "pitch and roll follow the mission drift. "
                 + ((f"Translation fitted jointly with the cameras and the rotation over every block: baseline "
                     f"{jt['baseline_m']:.5f} m" + (f" (large blocks alone, cameras held: {tr['baseline_m']:.5f} m)" if tr else "")
                     + "; projects use it (translation = fitted).") if (applied and jt) else
                    f"Translation fitted on {len(tr['blocks'])} large blocks ({', '.join(sorted(tr['blocks']))}): "
                    f"baseline {tr['baseline_m']:.5f} m; projects use it (translation = fitted)." if applied else
                    (f"Translation fitted earlier, carried over from the cameras in use (baseline "
                     f"{float(r.get('baseline_m', 0)):.5f} m; translation = fitted)"
                     + (" - the new large-block fit, translation_fit, is recorded but not applied." if tr else ".")
                     if carried else
                     "Translation from CAHV in each project"
                     + (" (the large-block fit, translation_fit, is recorded but not applied)." if tr else "."))))
    (out / "M2020_N_rig.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (out / "consensus.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return out


# ----------------------------------------------------------------------------------------- check and promote (v0p72)
PROMOTE_LIMITS = {"min_blocks": 8, "residual_factor": 1.15, "max_camera_change_px": 0.5, "max_rig_change_mdeg": 10.0,
                  "max_ppm_change": 5.0, "max_pp_slope_change": 0.02,
                  # v0p74.4: the rig translation (label baseline 0.42436 m)
                  "baseline_m": (0.420, 0.428), "max_baseline_disagreement_mm": 1.5, "max_baseline_sd_mm": 1.0}


def _rotation_mdeg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(np.asarray(Rb) @ np.asarray(Ra).T).as_rotvec())) * 1e3)


def check_candidate(candidate_dir: PathLike, current_dir: Optional[PathLike] = None,
                    limits: Optional[Dict[str, float]] = None, screening: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    v0p72: is a consensus candidate (``STUDY_DIR/navcam_joint``) fit to replace the cameras in use (``current_dir``,
    default ``cmods_dir()``)?  Checks (``limits``, :data:`PROMOTE_LIMITS`): the files load; at least ``min_blocks``
    blocks; the adjustment converged with finite sd; residual median / rms / p95 at most ``residual_factor`` x those of
    the cameras in use (different block sets, hence the margin); each eye within ``max_camera_change_px`` (rms over
    the frame, rotation removed) of the camera in use, the rig rotation within ``max_rig_change_mdeg``; the thermal
    focal slope within ``max_ppm_change`` ppm/degC and the principal-point slopes within ``max_pp_slope_change``
    px/degC of the ones in use; the early-mission fit (when made) converged on a convex surface with finite sd; no
    block that failed screening (``screening``: the study's ``block_screening.json``).  Returns {"ok", "checks":
    [{"check", "value", "limit", "ok", "note"}], "early_mission", "candidate", "current"}.
    """
    from . import navcal as NC
    from ..paths import cmods_dir
    lim = dict(PROMOTE_LIMITS, **(limits or {}))
    cand, cur = Path(candidate_dir), Path(current_dir) if current_dir else cmods_dir()
    checks = []

    def add(name, value, limit, ok, note=""):
        checks.append({"check": name, "value": value, "limit": limit, "ok": bool(ok), "note": note})

    files = ["M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json", "consensus.json"]
    missing = [f for f in files if not (cand / f).is_file()]
    add("files", len(files) - len(missing), len(files), not missing, ", ".join(missing) or "all present")
    if missing:
        return {"ok": False, "checks": checks, "candidate": str(cand), "current": str(cur)}
    res = json.loads((cand / "consensus.json").read_text(encoding="utf-8"))
    new = {e: json.loads((cand / f"M2020_{e}_fisheye_tangential.json").read_text(encoding="utf-8")) for e in ("NL", "NR")}
    old = {e: json.loads((cur / f"M2020_{e}_fisheye_tangential.json").read_text(encoding="utf-8")) for e in ("NL", "NR")}
    rn = json.loads((cand / "M2020_N_rig.json").read_text(encoding="utf-8"))
    ro = json.loads((cur / "M2020_N_rig.json").read_text(encoding="utf-8"))
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
        a = NC.pycolmap_camera(old[e]["model"], old[e]["params"])
        b = NC.pycolmap_camera(new[e]["model"], new[e]["params"])
        d = NC.compare(a, b)
        add(f"{e} change vs in use", round(float(d["rms_px"]), 3), lim["max_camera_change_px"],
            d["rms_px"] <= lim["max_camera_change_px"], f"corners {d.get('corner_rms_px', float('nan')):.3f} px (rotation removed)")
    dr = _rotation_mdeg(ro["R_sensor_from_ref"], rn["R_sensor_from_ref"])
    add("rig rotation change", round(dr, 2), lim["max_rig_change_mdeg"], dr <= lim["max_rig_change_mdeg"], "mdeg")
    from .cmods_form import check_form, thermal_terms
    # v0p73: fx and fy of each eye their own slope - the largest change of the four
    tt = {(e, w): thermal_terms(m[e].get("thermal")) for e in ("NL", "NR") for w, m in (("new", new), ("old", old))}
    dppm = max(abs(tt[(e, "new")][a] - tt[(e, "old")][a]) for e in ("NL", "NR") for a in ("fx_ppm", "fy_ppm"))
    add("thermal f slope change", round(dppm, 2), lim["max_ppm_change"], dppm <= lim["max_ppm_change"],
        "ppm/degC, largest of fx, fy per eye: " + ", ".join(f"{e} {tt[(e, 'new')]['fx_ppm']:+.1f}/{tt[(e, 'new')]['fy_ppm']:+.1f}"
                                                            for e in ("NL", "NR")))
    # v0p73: the stereo temperature change is what the data measure - NR's principal-point slope minus NL's (x fR/fL);
    # which eye carries it is a convention (NL reference since v0p73), so a candidate that moved it is not a change
    def stereo(m, k):
        ratio = float(m["NR"]["params"][0]) / float(m["NL"]["params"][0])
        return thermal_terms(m["NR"].get("thermal"))[k] - ratio * thermal_terms(m["NL"].get("thermal"))[k]
    dpp = max(abs(stereo(new, k) - stereo(old, k)) for k in ("cx_px", "cy_px"))
    add("stereo pp thermal slope change", round(dpp, 4), lim["max_pp_slope_change"], dpp <= lim["max_pp_slope_change"],
        f"px/degC, NR - NL x fR/fL: cx {stereo(new, 'cx_px'):+.4f} (in use {stereo(old, 'cx_px'):+.4f}), "
        f"cy {stereo(new, 'cy_px'):+.4f} (in use {stereo(old, 'cy_px'):+.4f})")
    fm = check_form(cand)                                     # v0p73: the frozen form (k4 = 0, NL reference, ...)
    add("form", fm.get("form", "v0p73") if fm["ok"] else "not in the form",
        "NL reference, no rig temperature term; k4 = 0 (v0p73) or fitted with k3 (v0p80)", fm["ok"],
        "; ".join(fm["errors"] + fm["warnings"]) or "ok")
    lo = res.get("leave_one_out")                             # v0p80: the recipe's promotion gate
    if lo and (lo.get("max") or {}).get("centre_px") is not None:
        mx = lo["max"]
        add("leave-one-block-out", f"centre {mx['centre_px']:.3f} / corners {mx['corner_px']:.3f} px",
            f"{LOO_LIMITS['centre_px']} / {LOO_LIMITS['corner_px']} px", bool(lo.get("ok")),
            f"worst: centre {lo.get('worst', {}).get('centre_px')}, corners {lo.get('worst', {}).get('corner_px')}; "
            f"stereo pointing up to {mx.get('stereo_px') or 0.0:.3f} px" + (f"; failed: {', '.join(lo['errors'])}"
                                                                              if lo.get("errors") else ""))
    elif res.get("form") == "v0p80":
        add("leave-one-block-out", "not run", "required for v0p80", False, "run the consensus with --leave-one-out")
    yr = (res.get("thermal") or {}).get("yaw_rate_mdeg_per_1000sol")
    if yr is not None:
        sdy = ((res.get("thermal") or {}).get("sd") or {}).get("yaw_rate")
        add("rig yaw rate", round(float(yr), 3), "measured", True,
            "mdeg per 1000 sol" + (f", sd {sdy:.3f} (z {float(yr) / sdy:+.1f})" if sdy else ""))
    em = res.get("early_mission")
    if em and em.get("held"):                       # v0p83: held at the start's offsets (a side study)
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
        if jt and tr:                                       # the two estimates must agree
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


def check_lines(rep: Dict[str, Any]) -> list:
    out = [f"candidate {rep['candidate']} against the cameras in use {rep['current']}:"]
    for c in rep["checks"]:
        lim = "" if c["limit"] is None else f"  (limit {c['limit']})"
        out.append(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']:28s} {c['value']}{lim}  {c['note']}")
    out.append("=> " + ("everything checks out" if rep["ok"] else "NOT ready to promote"))
    return out


def promote_candidate(candidate_dir: PathLike, note: str, repo_root: PathLike, commit: bool = True,
                      verbose: bool = True) -> Dict[str, Any]:
    """
    v0p72: ``scripts/promote_cmods.py candidate --note note`` (the files into ``src/mppp/data/cmods``, the old ones to
    ``history/``), then - with ``commit`` and a git working copy - a local commit of ``src/mppp/data/cmods``, and
    ``_transfer/promoted.json`` (what was promoted, when, the commit) for Claude to take into the next delivery.
    """
    import subprocess
    import sys
    root = Path(repo_root)
    cmd = [sys.executable, str(root / "scripts" / "promote_cmods.py"), str(candidate_dir), "--note", note]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out: Dict[str, Any] = {"command": cmd, "returncode": p.returncode, "output": p.stdout + p.stderr}
    if verbose:
        print(p.stdout + p.stderr, flush=True)
    if p.returncode:
        return out
    if commit and (root / ".git").exists():
        a = subprocess.run(["git", "-C", str(root), "add", "src/mppp/data/cmods"], capture_output=True, text=True)
        c = subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", f"cmods: Navcam consensus promoted ({note})",
                            "--", "src/mppp/data/cmods"], capture_output=True, text=True)
        h = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        out["commit"] = h.stdout.strip() if c.returncode == 0 else None
        out["git"] = (a.stdout + a.stderr + c.stdout + c.stderr).strip()
        if verbose:
            print(("committed " + out["commit"]) if out["commit"] else ("not committed: " + out["git"]), flush=True)
    tr = root / "_transfer"
    tr.mkdir(exist_ok=True)
    (tr / "promoted.json").write_text(json.dumps({"candidate": str(candidate_dir), "note": note,
                                                  "when": datetime.datetime.now().isoformat(timespec="seconds"),
                                                  "commit": out.get("commit")}, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------- rig residuals (v0p73)
def _ols(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """y = c + s (x - mean x): slope, its se (residual scatter), the scatter (rms) and n."""
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


def rig_residuals(scapes_cfg: Dict[str, PathLike], cameras_dir: Optional[PathLike] = None, points: int = 8000,
                  iterations: int = 60, verbose: bool = True, epoch_gap: Optional[float] = 50.0,
                  epoch_breaks: Optional[Sequence[float]] = (300.0,)) -> Dict[str, Any]:
    """
    v0p73: does the rig follow the model?  Each block on its own, with the Navcam cameras of ``cameras_dir`` (default:
    in use) held and their whole model applied per image - fx, fy per axis and NR cx against temperature, the rig
    drift with its yaw rate, the early-mission offsets - and only the rig rotation free.  The block's rig minus the
    model's rig (its constant rotation; the temperature and sol terms are in the keypoints) is the residual: yaw,
    pitch, roll mdeg.  In the v0p73 form the yaw residual should not follow the block temperature (NR cx carries the
    stereo temperature change) and none should follow the sol (the drift rates).  Returns {"blocks": [{"block",
    "T_median_degC", "sol_median", "images", "yaw_mdeg", "pitch_mdeg", "roll_mdeg"}], "fits": {angle: {"vs_T": ...,
    "vs_sol": ...}}, "cameras": dir}: slopes mdeg/degC and mdeg per 1000 sol with se and z.
    v0p75: ``epoch_gap`` (sols; None = off): a block whose Navcam visits are more than that many sols apart gets one
    rig per visit (:func:`mppp.sfm.navcal.split_rig_by_epoch`) and one row per visit (``block`` "name [sols a-b]",
    ``site`` the block, ``epoch``) - the early visits of Chal Rocks Sid and Three Forks then test the drift on their own.
    v0p75.1: ``epoch_gap`` 50 sols (was 100) and ``epoch_breaks`` (sol 300, the pitch knot): images on both sides of
    the knot are separate rows even within one visit (Rochette 178-190 + 341).
    """
    import pycolmap
    from . import navcal as NC
    from .cmods_form import thermal_terms
    from .project import navcam_distortion_terms
    from ..paths import cmods_dir
    d = Path(cameras_dir) if cameras_dir else cmods_dir()
    nl = json.loads((d / "M2020_NL_fisheye_tangential.json").read_text(encoding="utf-8"))
    nr = json.loads((d / "M2020_NR_fisheye_tangential.json").read_text(encoding="utf-8"))
    rj = json.loads((d / "M2020_N_rig.json").read_text(encoding="utf-8"))
    R_model = np.asarray(rj["R_sensor_from_ref"], float)
    rig = (R_model, np.asarray(rj["t_sensor_from_ref_m"], float))
    tl, tr = thermal_terms(nl.get("thermal")), thermal_terms(nr.get("thermal"))
    t0 = tl["T0"]
    axis_ppm = {"NL": (tl["fx_ppm"], tl["fy_ppm"]), "NR": (tr["fx_ppm"], tr["fy_ppm"])}
    pp = {1: (tl["cx_px"], tl["cy_px"]), 2: (tr["cx_px"], tr["cy_px"])}
    drift = rj.get("drift")
    em = nl.get("early_mission") or {}
    rows = []
    t_all = time.time()
    for kb, (name, folder) in enumerate(scapes_cfg.items(), 1):
        t = time.time()
        try:
            sc = NC.load_scape(name, folder)
            cams = {k: pycolmap.Camera(model=j["model"], width=5120, height=3840, params=np.asarray(j["params"], float))
                    for k, j in (("NL", nl), ("NR", nr))}
            rec, proj, idx = NC.merge_scapes([sc], cams, rig, points_per_scape=points, seed=0)
        except Exception as e:                                                # noqa: BLE001
            rows.append({"block": name, "error": f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"  {name}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        temps = idx["temps"]
        for cid, key in ((1, "NL"), (2, "NR")):
            c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
            proj.cameras[key] = navcam_distortion_terms(c, "refine", k4="consensus")
        proj.settings["navcam_rig_yaw"] = "refine"
        kw: Dict[str, Any] = {"pp_slopes": pp, "axis_ppm": axis_ppm}
        if drift:
            kw["drift"] = drift
        k1s = {"NL": float((nl.get("thermal") or {}).get("k1_per_degC") or 0.0),
               "NR": float((nr.get("thermal") or {}).get("k1_per_degC") or 0.0)}
        if any(k1s.values()):                                                    # v0p80
            kw["k1_slopes"] = k1s
        applied = [k for k in (em.get("applied_terms") or []) if k in EARLY_TERMS]
        if applied:
            kw["extra_map"] = early_keypoint_map(proj, float(em["before_sol"]),
                                                 [float(em.get(f"d{k}_px", 0.0)) if k in applied else 0.0 for k in EARLY_TERMS])
        # v0p75: a block whose Navcam visits are more than ``epoch_gap`` sols apart (Chal Rocks Sid: sols 91-101 and
        # 360-378) gets one rig per visit, refined on its own, and one residual row per visit
        epochs = []
        if epoch_gap:
            rec_e, proj_e, eps = NC.split_rig_by_epoch(rec, proj, temps, gap=float(epoch_gap),
                                                       breaks=tuple(epoch_breaks or ()))     # v0p75.1: + sol 300
            if len(eps) > 1:
                rec, proj, epochs = rec_e, proj_e, eps
        ba = NC.joint_adjust(rec, proj, temps, 0.5 * (tl["fx_ppm"] + tl["fy_ppm"]), t0, max_iterations=iterations,
                             refine_rig="rotation", hold_cameras=("NL", "NR"), rig_slopes=(0.0, 0.0), **kw)
        units = []
        if epochs:
            for e in epochs:
                if e.get("rig_id") is None:
                    continue
                rg = rec.rigs[int(e["rig_id"])]
                sid = next(iter(rg.non_ref_sensors), None)
                if sid is None:
                    continue
                units.append((f"{name} [sols {e['sol_min']}-{e['sol_max']}]", rg.sensor_from_rig(sid),
                              e["T_median_degC"], e["sol_median"], e["images"],
                              {"epoch": e["epoch"], "sols": [e["sol_min"], e["sol_max"]], "frames": e["frames"]}))
        if not units:
            _, _, Tr = NC.stereo_rig(rec)
            Ts = [v for n, v in temps.items() if n in {im.name for im in rec.images.values()}]
            sols = [float(r["sol"]) for r in proj.images if r.get("sol") is not None]
            units.append((name, Tr, float(np.median(Ts)) if Ts else None, float(np.median(sols)) if sols else None,
                          int(rec.num_reg_images()), None))
        for label, Tr, Tm, sm, n_im, ep in units:
            a = NC.rig_angles(np.asarray(Tr.rotation.matrix()), R_ref=R_model)
            row = {"block": label, "site": name, "T_median_degC": Tm, "sol_median": sm, "images": int(n_im),
                   "yaw_mdeg": a["yaw_mdeg"], "pitch_mdeg": a["pitch_mdeg"], "roll_mdeg": a["roll_mdeg"],
                   "cost": float(ba["final_cost"]), "iterations": ba["iterations"]}
            if ep:
                row["epoch"] = ep
            rows.append(row)
            if verbose:
                print(f"  {label}: T {(row['T_median_degC'] if row['T_median_degC'] is not None else float('nan')):+.1f} degC, "
                      f"sol {(row['sol_median'] or float('nan')):.0f}: residual yaw "
                      f"{row['yaw_mdeg']:+.2f}, pitch {row['pitch_mdeg']:+.2f}, roll {row['roll_mdeg']:+.2f} mdeg "
                      f"({time.time() - t:.0f} s)  {NC.progress(kb, len(scapes_cfg), t_all, 'block')}", flush=True)
    ok = [r for r in rows if "error" not in r]
    fits = {}
    for ang in ("yaw", "pitch", "roll"):
        y = np.array([r[f"{ang}_mdeg"] for r in ok], float)
        T = np.array([np.nan if r["T_median_degC"] is None else r["T_median_degC"] for r in ok], float)
        S = np.array([np.nan if r["sol_median"] is None else r["sol_median"] for r in ok], float)
        vt, vs = _ols(T, y), _ols(S / 1000.0, y)
        fits[ang] = {"mean_mdeg": float(np.mean(y)) if len(y) else None, "sd_mdeg": float(np.std(y)) if len(y) > 1 else None,
                     "vs_T": vt, "vs_sol": vs}
        if verbose and vt.get("slope") is not None:
            print(f"{ang}: residual mean {fits[ang]['mean_mdeg']:+.2f} (sd {fits[ang]['sd_mdeg']:.2f}) mdeg; vs block "
                  f"temperature {vt['slope']:+.3f} +- {vt['se']:.3f} mdeg/degC (z {vt['z']:+.1f}); vs sol "
                  f"{vs['slope']:+.2f} +- {vs['se']:.2f} mdeg/1000 sol (z {vs['z']:+.1f})", flush=True)
    return {"blocks": rows, "fits": fits, "cameras": str(d),
            "model": {"axis_ppm": axis_ppm, "pp_slopes": {"NL": list(pp[1]), "NR": list(pp[2])},
                      "yaw_rate_mdeg_per_1000sol": 1000.0 * float((drift or {}).get("yaw_mdeg_per_sol") or 0.0)}}


def plot_rig_residuals(res: Dict[str, Any], out: Optional[PathLike] = None, figsize=(12, 7)):
    """Rig residuals (yaw, pitch, roll) against the block temperature (top) and sol (bottom), with the fitted lines."""
    import matplotlib.pyplot as plt
    ok = [r for r in res["blocks"] if "error" not in r]
    fig, axs = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    for j, ang in enumerate(("yaw", "pitch", "roll")):
        y = np.array([r[f"{ang}_mdeg"] for r in ok], float)
        for i, (k, lab, scale) in enumerate((("T_median_degC", "block Navcam temperature (degC)", 1.0),
                                              ("sol_median", "sol", 1.0))):
            ax = axs[i][j]
            x = np.array([np.nan if r[k] is None else r[k] for r in ok], float)
            ax.plot(x, y, "o", color="C0")
            for xx, yy, r in zip(x, y, ok):
                ax.annotate(r["block"][:12], (xx, yy), fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
            f = res["fits"][ang]["vs_T" if i == 0 else "vs_sol"]
            if f.get("slope") is not None:
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 2)
                xm = f["mean_x"] * (1000.0 if i == 1 else 1.0)
                sl = f["slope"] / (1000.0 if i == 1 else 1.0)
                ax.plot(xs, f["intercept"] + sl * (xs - xm), "-", color="C3",
                        label=(f"{f['slope']:+.3f} +- {f['se']:.3f} mdeg/degC" if i == 0 else
                               f"{f['slope']:+.2f} +- {f['se']:.2f} mdeg/1000 sol") + f" (z {f['z']:+.1f})")
                ax.legend(fontsize=7)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xlabel(lab, fontsize=8)
            ax.set_ylabel(f"{ang} residual (mdeg)", fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("Navcam rig residuals per block (cameras and rig model held, rig rotation free)", fontsize=10)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
    return fig
