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
from typing import Any, Dict, Optional, Tuple, Union

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
                    k = c0 + (k - c0) / ks[iid]
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


THERMAL_TERMS = ("f", "NL_cx", "NR_cx", "NL_cy", "NR_cy")
THERMAL_STEPS = {"f": 10.0, "NL_cx": 0.03, "NR_cx": 0.03, "NL_cy": 0.03, "NR_cy": 0.03}    # profile grid steps


def _thermal_values(th: Dict[str, Any]) -> Dict[str, float]:
    """{"f": ppm/degC, "NL_cx": px/degC, ...} of a consensus result's ``thermal`` entry."""
    pp = th.get("pp_slopes") or {}
    return {"f": float(th["ppm_per_degC"]),
            "NL_cx": float(pp.get("NL", [th.get("cx_px_per_degC_NL", 0.0), 0.0])[0]),
            "NL_cy": float(pp.get("NL", [0.0, 0.0])[1]),
            "NR_cx": float(pp.get("NR", [0.0, 0.0])[0]), "NR_cy": float(pp.get("NR", [0.0, 0.0])[1])}


def _pp_slopes(v: Dict[str, float]) -> Dict[int, Tuple[float, float]]:
    return {1: (float(v["NL_cx"]), float(v["NL_cy"])), 2: (float(v["NR_cx"]), float(v["NR_cy"]))}


def profile_thermal(rec, proj, temps, values: Dict[str, float], T0: float, term: str, drift=None,
                    iterations: int = 50, npoints: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """
    v0p64: the joint cost over a grid of one thermal term (``"f"`` ppm/degC, ``"NL_cx"`` ... px/degC), the others at
    ``values``, each adjustment started from the same converged state (``rec`` is not changed).  The best value is
    the minimum of a parabola through the costs, its sd sqrt(variance factor / curvature) (:func:`navcal.fit_profile`).
    A minimum outside the grid extends it by two steps (twice at most).
    """
    import copy as _copy
    from . import navcal as NC
    step = THERMAL_STEPS[term]
    c0 = float(values[term])
    grid = [c0 + step * (k - (npoints - 1) / 2) for k in range(npoints)]
    rows: list = []
    done = set()
    for _ in range(3):
        for g in grid:
            if round(g, 9) in done:
                continue
            done.add(round(g, 9))
            v = dict(values, **{term: float(g)})
            r = _copy.deepcopy(rec)
            t = time.time()
            kw: Dict[str, Any] = {"pp_slopes": _pp_slopes(v)}
            if drift:
                kw["drift"] = drift
            ba = NC.joint_adjust(r, proj, temps, v["f"], T0, max_iterations=iterations, rig_slopes=(0.0, 0.0), **kw)
            red = max(2 * ba["observations"] - 1, 1)
            rows.append({"ppm_per_degC": float(g), "cost": float(ba["final_cost"]), "iterations": ba["iterations"],
                         "variance_factor": float(2 * ba["final_cost"] / red)})
            if verbose:
                print(f"  {term} {g:+.4f}: cost {ba['final_cost']:.2f} it {ba['iterations']} ({time.time() - t:.0f} s)",
                      flush=True)
        fit = NC.fit_profile(sorted(rows, key=lambda r: r["ppm_per_degC"]))
        lo, hi = min(r["ppm_per_degC"] for r in rows), max(r["ppm_per_degC"] for r in rows)
        b = fit["best_ppm_per_degC"]
        if lo <= b <= hi:
            break
        grid = [hi + step, hi + 2 * step] if b > hi else [lo - step, lo - 2 * step]
    out = {"term": term, "best": float(fit["best_ppm_per_degC"]), "sd": float(fit["sd_ppm_per_degC"]),
           "start": c0, "curvature": fit["curvature"], "variance_factor": fit["variance_factor"],
           "rows": [{"value": r["ppm_per_degC"], "cost": r["cost"], "iterations": r["iterations"]} for r in fit["rows"]],
           "at_edge": not (lo <= fit["best_ppm_per_degC"] <= hi)}
    if verbose:
        print(f"{term}: {out['best']:+.4f} +- {out['sd']:.4f} (start {c0:+.4f})" + ("  [outside the grid]" if out["at_edge"] else ""),
              flush=True)
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
                      + f": cost {ba['final_cost']:.2f} it {ba['iterations']} ({time.time() - t:.0f} s)", flush=True)
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
                  thermal_terms: Tuple[str, ...] = ("f", "NL_cx"), early_mission_sol: Optional[float] = None,
                  early_rounds: int = 2) -> Dict[str, Any]:
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
    ``early_mission`` holds the offsets, sd, z, p-values and the terms to apply (|z| >= ``EARLY_APPLY_Z``)."""
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
    th = nl.get("thermal") or {}
    ppm, t0, pp_nl = float(th["ppm_per_degC"]), float(th["T0_degC"]), float(th.get("cx_px_per_degC") or 0.0)
    thr = nr.get("thermal") or {}
    tvals = {"f": ppm, "NL_cx": pp_nl, "NL_cy": float(th.get("cy_px_per_degC") or 0.0),
             "NR_cx": float(thr.get("cx_px_per_degC") or 0.0), "NR_cy": float(thr.get("cy_px_per_degC") or 0.0)}
    bad = [k for k in thermal_terms if k not in THERMAL_TERMS]
    if bad:
        raise ValueError(f"thermal_terms {bad}: use {THERMAL_TERMS}")
    drift = copy.deepcopy(rj.get("drift"))
    if drift:
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
    proj.settings["navcam_rig_yaw"] = "refine"            # one yaw for all: no drift / temperature term below
    names = PARAM_NAMES["THIN_PRISM_FISHEYE"]
    for cid, key in ((1, "NL"), (2, "NR")):
        c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
        c = navcam_distortion_terms(c, "refine", k4="zero" if k4_zero else "consensus")
        proj.cameras[key] = c
        rec.cameras[cid].params = np.asarray(c["params"], float)
    xkw: Dict[str, Any] = {"pp_slopes": _pp_slopes(tvals)}
    if drift:
        xkw["drift"] = drift
    t = time.time()
    ba = NC.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=not refit_thermal,
                         rig_slopes=(0.0, 0.0), **xkw)
    profiles = {}
    if refit_thermal:
        # v0p64: the thermal terms in the consensus form, one profile after the other from the converged state
        if verbose:
            print(f"consensus at the start values: cost {ba['final_cost']:.1f} ({time.time() - t:.0f} s); "
                  f"thermal profiles: {', '.join(thermal_terms)}", flush=True)
        for term in thermal_terms:
            pr = profile_thermal(rec, proj, temps, tvals, t0, term, drift=drift, verbose=verbose)
            profiles[term] = pr
            tvals[term] = pr["best"]
        ppm = tvals["f"]
        xkw["pp_slopes"] = _pp_slopes(tvals)
        t = time.time()
        ba = NC.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True,
                             rig_slopes=(0.0, 0.0), **xkw)
    early = None
    if early_mission_sol:
        # v0p70: the early-mission offsets in the consensus form, then the consensus once more with them
        em0 = nl.get("early_mission") or {}
        if em0:
            x0 = [float(em0.get(f"d{k}_px", 0.0)) for k in EARLY_TERMS]
        else:                                       # the block alignments' own (per-block) estimate as the start
            x0 = early_start_from_blocks(scapes_cfg, float(early_mission_sol))["start"]
        if verbose:
            print(f"early-mission offsets before sol {early_mission_sol:g} (start {x0}):", flush=True)
        early = fit_early_offsets(rec, proj, temps, ppm, t0, xkw, float(early_mission_sol), start=x0,
                                  rounds=early_rounds, verbose=verbose)
        if early.get("offsets_px"):
            xb = [early["offsets_px"][k] for k in EARLY_TERMS]
            t = time.time()
            ba = NC.joint_adjust(rec, proj, temps, ppm, t0, max_iterations=iterations, covariance=True,
                                 rig_slopes=(0.0, 0.0), extra_map=early_keypoint_map(proj, float(early_mission_sol), xb),
                                 **xkw)
    ks = NC.keypoint_scales(proj, temps, ppm, t0)
    kmap = NC.thermal_keypoint_map(proj, temps, t0, (0.0, 0.0), xkw["pp_slopes"], drift)
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
           "rig_abs_mdeg": NC.rig_angles(np.asarray(T.rotation.matrix())),
           "rig_R": np.asarray(T.rotation.matrix()).tolist(), "rig_t": np.asarray(T.translation).tolist(),
           "stereo": NC.stereo_offset(rec.cameras[1], rec.cameras[2], T), "residuals": st,
           "blocks": {k: len(v) for k, v in idx["images"].items()}, "start_dir": str(d),
           "thermal": {"ppm_per_degC": ppm, "T0_degC": t0, "cx_px_per_degC_NL": tvals["NL_cx"],
                       "pp_slopes": {"NL": [tvals["NL_cx"], tvals["NL_cy"]], "NR": [tvals["NR_cx"], tvals["NR_cy"]]},
                       "refit": bool(refit_thermal), "terms": list(thermal_terms) if refit_thermal else [],
                       "sd": {k: profiles[k]["sd"] for k in profiles},
                       "profiles": profiles}, "drift": drift,
           "k4_zero": bool(k4_zero), "early_mission": early}
    if translation_scapes:
        tr = fit_rig_translation(translation_scapes, out, points=points, sigma_m=translation_sigma_m,
                                 iterations=iterations, verbose=verbose)
        out["rig_translation"] = tr
        out["rig_translation_applied"] = bool(apply_translation)
        if apply_translation:
            out["rig_t"] = list(tr["t_m"])
    if verbose:
        print(f"consensus: cost {out['final_cost']:.1f}, it {out['iterations']}, median {st['median_px']:.4f} / rms "
              f"{st['rms_px']:.4f} px, corners {st['radial'][-1]['median_px'] or float('nan'):.3f} px, rig yaw "
              f"{out['rig_abs_mdeg']['yaw_mdeg']:+.2f} mdeg  ({out['seconds']:.0f} s)", flush=True)
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
        c["source"] = (f"MPPP Navcam consensus (mppp.sfm.navcal_consensus, {stamp}): one {eye} camera shared by "
                       f"{len(blocks)} blocks ({', '.join(blocks)}); fisheye + tangential at "
                       f"{res['thermal']['T0_degC']:g} degC, f {res['thermal']['ppm_per_degC']:+.1f} ppm/degC, NL cx "
                       f"{res['thermal']['cx_px_per_degC_NL']:+.4f} px/degC "
                       f"({'refitted' if res['thermal'].get('refit') else 'held'}), k4 = 0 (held), one rig yaw "
                       f"{a['yaw_mdeg']:+.2f} mdeg. {note}".strip())
        rt = res["thermal"]
        if rt.get("refit"):                                # v0p64: the refitted thermal terms of this consensus
            tv = _thermal_values(rt)
            sdt = rt.get("sd") or {}
            th0 = dict(c.get("thermal") or {})
            th0.update({"ppm_per_degC": tv["f"], "T0_degC": float(rt["T0_degC"]),
                        "cx_px_per_degC": tv[f"{eye}_cx"], "cy_px_per_degC": tv[f"{eye}_cy"],
                        "note": f"v0p64: refitted with the consensus ({stamp}; profiles of {', '.join(rt.get('terms') or [])})"})
            for k, kk in (("f", "sd_ppm_per_degC"), (f"{eye}_cx", "sd_cx_px_per_degC"), (f"{eye}_cy", "sd_cy_px_per_degC")):
                if k in sdt:
                    th0[kk] = float(sdt[k])
            c["thermal"] = th0
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
        c["verification"] = {"note": "not verified (no leave-one-block-out run)",
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
    tr = res.get("rig_translation")
    applied = bool(tr and res.get("rig_translation_applied"))
    if tr:
        r["translation_fit"] = {k: tr[k] for k in ("blocks", "dcentre_mm", "sd_t_mm", "baseline_start_m", "baseline_m",
                                                   "dbaseline_mm", "t_m", "variance_factor", "sigma_m")}
    if applied:
        r["t_sensor_from_ref_m"] = [float(x) for x in tr["t_m"]]
        r["baseline_m"] = float(tr["baseline_m"])
        r["translation"] = "fitted"
    else:
        r.pop("translation", None)
    r["note"] = (f"x_NR = R x_NL + t. Navcam consensus {stamp} ({len(blocks)} blocks); yaw {a['yaw_mdeg']:+.2f}, "
                 f"pitch {a['pitch_mdeg']:+.2f}, roll {a['roll_mdeg']:+.2f} mdeg. One yaw for every block: no "
                 f"temperature term, the drift's yaw rate 0; pitch and roll follow the mission drift. "
                 + (f"Translation fitted on {len(tr['blocks'])} large blocks ({', '.join(sorted(tr['blocks']))}): "
                    f"baseline {tr['baseline_m']:.5f} m; projects use it (translation = fitted)." if applied else
                    "Translation from CAHV in each project"
                    + (f" (the large-block fit, translation_fit, is recorded but not applied)." if tr else ".")))
    (out / "M2020_N_rig.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (out / "consensus.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return out
