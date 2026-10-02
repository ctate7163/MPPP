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


def fit_consensus(scapes_cfg: Dict[str, PathLike], points: int = 8000, iterations: int = 100,
                  k4_zero: bool = True, start_dir: Optional[PathLike] = None, verbose: bool = True,
                  translation_scapes: Optional[Dict[str, PathLike]] = None, translation_sigma_m: float = 0.05,
                  apply_translation: bool = False, refit_thermal: bool = False,
                  thermal_terms: Tuple[str, ...] = ("f", "NL_cx")) -> Dict[str, Any]:
    """The joint adjustment (see the module docstring) of the blocks ``{name: WORK folder}``; returns the shared
    cameras, the rig, residuals by radius and the per-block image counts.  v0p62: ``translation_scapes`` (the large
    blocks, :func:`large_blocks`): then :func:`fit_rig_translation` on them with the new cameras held
    (``res["rig_translation"]``); ``apply_translation`` makes it the rig's translation (``rig_t``), which
    :func:`write_consensus` then writes with ``"translation": "fitted"`` for the projects to use.  v0p64:
    ``refit_thermal`` profiles the ``thermal_terms`` (``"f"`` ppm/degC, ``"NL_cx"``, ``"NR_cx"``, ``"NL_cy"``,
    ``"NR_cy"`` px/degC, about T0) one after the other in this same form (:func:`profile_thermal`), then adjusts once
    more at the best values; the result's ``thermal`` holds them with their sd and the profiles."""
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
    ks = NC.keypoint_scales(proj, temps, ppm, t0)
    kmap = NC.thermal_keypoint_map(proj, temps, t0, (0.0, 0.0), xkw["pp_slopes"], drift)
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
           "k4_zero": bool(k4_zero)}
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
