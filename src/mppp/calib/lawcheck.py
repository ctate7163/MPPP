"""
How well a Navcam law predicts the cameras on Mars (0.94.4; Christian, 25 Oct 2026: "load the colmap files with the
global camera, rig and other kernels so that we know how good those global solutions are ... This a key place
where we empirically say how good our camera models are at predicting the real behavior of cameras on Mars").

A law is a cameras folder (the models in use, or a joint's candidate): the cameras at T0 with their thermal and
early-mission terms, and the stereo rig with its drift.  Over a set of finished blocks:

* **law held** - one camera per (block, eye, merged temperature bin, era) and one stereo rig per (block, epoch, bin,
  era), each SET from the law at the unit's own temperature, era and sol (the images' own temperatures through the
  keypoint map), and HELD; only the poses and points re-solved.  Per block: the residuals the law leaves.
* **units free** (per law) - the same units refined freely (fx, fy, k1 and the rig rotation; the principal points and the
  rest of the distortion at the law, as in the joint's diagnostic pass), with the covariance of every unit.  Per
  block: the residuals; per unit: its deviation from the law in units of its own sd (z).
* **pointing** - the joint's pointing pass (:func:`mppp.calib.tied.pointing_pass`): one camera per (block, eye, era)
  with the principal point free, every rig at the law; per unit the principal point's deviation and its sd.

The law predicts the cameras where the deviations are standard normal (chi^2 / dof near 1) with no trend in
temperature or sol; chi^2 / dof >> 1, or a trend, is variation the law does not model.  The residual cost of holding
the law (law held minus units free, per block) says whether that variation matters for the 3-D.  The free and
pointing passes hold the terms they do not free (principal points or k2-k4, the rig) at the law under test, so they
are run per law.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from . import joint as J
from . import tied as T

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]
MIN_IMAGES = 8                    # units with fewer images / frames are listed, not used in the statistics
FREE_ORDER = ("fx", "fy", "k1")   # the free pass's camera tangent (the SubsetManifold keeps the free ones in order)


def law_globals(st: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {"G_NL": st["G"]["NL"], "G_NR": st["G"]["NR"], "S_NL": st["S"]["NL"], "S_NR": st["S"]["NR"],
            "E_NL": st["E"]["NL"], "E_NR": st["E"]["NR"], "Q0": st["Q0"], "D": st["D"], "H": st["H"]}


def _block_stats(rec, proj, idx, ks, kmap) -> Dict[str, Dict[str, float]]:
    out = {}
    reg = {int(i) for i in rec.reg_image_ids()}
    for s, ims in idx["images"].items():
        ids = [int(i) for i in ims if int(i) in reg]
        if ids:
            out[s] = J.residual_stats(rec, proj, keypoint_scale=ks, images=ids, keypoint_map=kmap)
    return out


def _unit_sols(proj) -> Dict[int, float]:
    sols: Dict[int, List[int]] = {}
    for r in proj.images:
        if r.get("sol") is not None:
            sols.setdefault(int(r["camera_id"]), []).append(int(r["sol"]))
    return {cid: float(np.median(v)) for cid, v in sols.items()}


def load_blocks(scapes_cfg: Dict[str, PathLike], st: Dict[str, Any], points: int):
    import pycolmap
    cams = {k: pycolmap.Camera(model=st["cameras"][k]["model"], width=5120, height=3840,
                               params=np.asarray(st["cameras"][k]["params"], float)) for k in ("NL", "NR")}
    return J.merge_scapes((J.load_scape(n, scapes_cfg[n]) for n in scapes_cfg), cams, (st["R"], st["t"]),
                          points_per_scape=points, seed=0)


def law_held(rec0, proj0, idx, st: Dict[str, Any], iterations: int = 60) -> Dict[str, Any]:
    """Every unit set from the law and held; poses and points re-solved.  Per-block residuals."""
    from ..sfm.reconstruction import bundle_adjust
    t = time.time()
    g = law_globals(st)
    rec, proj, units = T.tied_reconstruction(rec0, proj0, idx, st, per_block=True,
                                             merge_min_images=T.DIAG_BIN_MIN_IMAGES)
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], g)
    ba = bundle_adjust(rec, proj, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=False, refine_principal_point=False,
                       refine_tangential=False, refine_rig=False, max_iterations=iterations, attitude_prior_deg=5.0,
                       linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap)
    return {"law": st["folder"], "blocks": _block_stats(rec, proj, idx, ks, kmap),
            "all": J.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap), "brief": ba["brief"],
            "final_cost": float(ba["final_cost"]), "seconds": time.time() - t}


def units_free(rec0, proj0, idx, st: Dict[str, Any], iterations: int = 60, verbose: bool = True) -> Dict[str, Any]:
    """The units refined freely from the law, with their covariance: the absolute values and sd of every unit (so
    they can be compared with any law, :func:`deviations`)."""
    t = time.time()
    g = law_globals(st)
    rec, proj, units = T.tied_reconstruction(rec0, proj0, idx, st, per_block=True,
                                             merge_min_images=T.DIAG_BIN_MIN_IMAGES)
    ba, _g, _ = T.adjust_tied(rec, proj, idx, st, units, hold=(), tie_rigs=False, tie_cameras=False,
                              iterations=iterations, covariance=True, passes=1, verbose=verbose, refine_pp=False)
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], g)
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor", 1.0) or 1.0)
    blocks = cov.get("blocks") or {}
    sols = _unit_sols(proj)
    cams, rigs, dims = [], [], set()
    for cid, u in units["cameras"].items():
        if cid not in rec.cameras:
            continue
        p = np.asarray(rec.cameras[cid].params, float)
        c = blocks.get(f"camera:{u['key']}")
        sd = {}
        if c is not None:
            c = np.asarray(c, float)
            dims.add(c.shape[0])
            d = np.sqrt(np.clip(np.diag(c) * vf, 0, None))
            order = FREE_ORDER if c.shape[0] == len(FREE_ORDER) else (
                T.TIED_PARAMS if c.shape[0] >= len(T.TIED_PARAMS) else ())
            sd = {k: float(d[i]) for i, k in enumerate(order) if i < len(d)}
        cams.append(dict(u, camera_id=int(cid), params=list(map(float, p)), sd=sd, sol=sols.get(int(cid))))
    for rid, u in units["rigs"].items():
        if rid not in rec.rigs:
            continue
        rg = rec.rigs[rid]
        sid = next(iter(rg.non_ref_sensors), None)
        if sid is None:
            continue
        R = np.asarray(rg.sensor_from_rig(sid).rotation.matrix())
        c = blocks.get(f"rig:{rid}:{u['nr_camera']}")
        sd = {}
        if c is not None:
            d = np.sqrt(np.clip(np.diag(np.asarray(c, float)) * vf, 0, None))
            # the quaternion's tangent is half the rotation vector (Ceres' EigenQuaternionManifold)
            sd = {a: float(2.0 * np.degrees(d[i]) * 1e3) for i, a in enumerate(T.ANGLES) if i < len(d)}
        rigs.append(dict(u, rig_id=int(rid), R=R.tolist(), sd_mdeg=sd))
    out = {"cameras": cams, "rigs": rigs, "variance_factor": vf, "camera_cov_dims": sorted(dims),
           "covariance_error": cov.get("error"), "blocks": _block_stats(rec, proj, idx, ks, kmap),
           "all": J.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap), "brief": ba["brief"],
           "final_cost": float(ba["final_cost"]), "seconds": time.time() - t}
    if verbose:
        log.info(f"units free: {len(cams)} cameras, {len(rigs)} rigs, camera covariance dims {sorted(dims)}, "
                 f"vf {vf:.3g}; {ba['brief'][:120]}  ({out['seconds']:.0f} s)")
    return out


def _wfit(x, y, sd):
    """Weighted straight line y = a + b x (weights 1 / sd^2), the slope's se scaled by max(1, chi^2/dof)."""
    x, y, sd = (np.asarray(v, float) for v in (x, y, sd))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sd) & (sd > 0)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0:
        return None
    x, y, w = x[ok], y[ok], 1.0 / sd[ok] ** 2
    X = np.column_stack([np.ones_like(x), x])
    N = (X.T * w) @ X
    beta = np.linalg.solve(N, (X.T * w) @ y)
    chi = float(np.sum(w * (y - X @ beta) ** 2) / (len(x) - 2))
    se = float(np.sqrt(np.linalg.inv(N)[1, 1] * max(chi, 1.0)))
    return {"slope": float(beta[1]), "se": se, "z": float(beta[1] / se) if se > 0 else None, "n": int(len(x)),
            "chi2_dof_after": chi}


def _chi2(z: Sequence[Optional[float]], dev: Sequence[float]) -> Optional[Dict[str, float]]:
    z = np.asarray([v for v in z if v is not None and np.isfinite(v)], float)
    if not z.size:
        return None
    return {"n": int(z.size), "chi2_dof": float(np.mean(z ** 2)), "robust": float((np.median(np.abs(z)) / 0.6745) ** 2),
            "mean_z": float(np.mean(z)), "median_abs_dev": float(np.median(np.abs(np.asarray(dev, float))))}


def deviations(free: Dict[str, Any], st: Dict[str, Any], pointing: Optional[Dict[str, Any]] = None,
               min_images: int = MIN_IMAGES) -> Dict[str, Any]:
    """The free units against the law ``st``: per unit the deviation (fx, fy ppm; k1; rig pitch, yaw, roll mdeg)
    and z; per parameter chi^2 / dof; the trends of the deviations with temperature and sol (weighted, z >= 3
    flagged); and, with the pointing pass, the same for the principal points (cx, cy px) and the stereo pointing."""
    cams, rigs = [], []
    for c in free["cameras"]:
        p = np.asarray(c["params"], float)
        law = T.camera_params(st, c["eye"], c["T"], c["early"])
        r = {k: c.get(k) for k in ("key", "scape", "eye", "bin", "early", "T", "images", "sol")}
        for k, i, scale in (("fx", 0, 1e6 / law[0]), ("fy", 1, 1e6 / law[1]), ("k1", 4, 1.0)):
            d = float((p[i] - law[i]) * scale)
            s_ = c["sd"].get(k)
            r[f"d_{k}"] = d
            r[f"sd_{k}"] = float(s_ * scale) if s_ else None
            r[f"z_{k}"] = d / (s_ * scale) if s_ else None
        cams.append(r)
    for u in free["rigs"]:
        R_law = T.R_of_q(T.rig_q_at(st["Q0"], st["D"], st["H"], u["sol"], st["knot"]))
        a = J.rig_angles(np.asarray(u["R"], float), R_ref=R_law)
        r = {k: u.get(k) for k in ("scape", "epoch", "bin", "early", "sol", "T", "frames")}
        for ang in T.ANGLES:
            d = float(a[f"{ang}_mdeg"])
            s_ = (u.get("sd_mdeg") or {}).get(ang)
            r[f"d_{ang}"], r[f"sd_{ang}"], r[f"z_{ang}"] = d, s_, (d / s_ if s_ else None)
        rigs.append(r)
    chi2: Dict[str, Any] = {}
    trends: Dict[str, Any] = {}
    for eye in ("NL", "NR"):
        cs = [r for r in cams if r["eye"] == eye and r["images"] >= min_images]
        for k, unit in (("fx", "ppm"), ("fy", "ppm"), ("k1", "1")):
            chi2[f"{eye}_{k}"] = _chi2([r[f"z_{k}"] for r in cs], [r[f"d_{k}"] for r in cs])
            for var, vu in (("T", "degC"), ("sol", "1000 sol")):
                x = [r["T"] if var == "T" else (r["sol"] / 1000.0 if r["sol"] is not None else np.nan) for r in cs]
                f = _wfit(x, [r[f"d_{k}"] for r in cs], [r[f"sd_{k}"] or np.nan for r in cs])
                if f:
                    trends[f"{eye}_{k}_vs_{var}"] = dict(f, units=f"{unit} per {vu}")
    rs = [r for r in rigs if r["frames"] >= min_images]
    for ang in T.ANGLES:
        chi2[f"rig_{ang}"] = _chi2([r[f"z_{ang}"] for r in rs], [r[f"d_{ang}"] for r in rs])
        for var, vu in (("T", "degC"), ("sol", "1000 sol")):
            x = [r["T"] if var == "T" else r["sol"] / 1000.0 for r in rs]
            f = _wfit(x, [r[f"d_{ang}"] for r in rs], [r[f"sd_{ang}"] or np.nan for r in rs])
            if f:
                trends[f"rig_{ang}_vs_{var}"] = dict(f, units=f"mdeg per {vu}")
    out = {"law": st["folder"], "cameras": cams, "rigs": rigs, "chi2": {k: v for k, v in chi2.items() if v},
           "trends": trends}
    if len(rs) >= 3:
        out["rig_sol_T_r"] = float(np.corrcoef([r["sol"] for r in rs], [r["T"] for r in rs])[0, 1])
    cs = [r for r in cams if r["images"] >= min_images and r["sol"] is not None]
    if len(cs) >= 3:
        out["camera_sol_T_r"] = float(np.corrcoef([r["sol"] for r in cs], [r["T"] for r in cs])[0, 1])
    if pointing:
        pp = []
        for c in pointing["cameras"]:
            law = T.camera_params(st, c["eye"], c["T"], c["early"])
            p = np.asarray(c["params"], float)
            r = {k: c.get(k) for k in ("key", "scape", "eye", "early", "T", "images")}
            for k, i in (("cx", 2), ("cy", 3)):
                d, s_ = float(p[i] - law[i]), c.get(f"sd_{k}_px")
                r[f"d_{k}"], r[f"sd_{k}"], r[f"z_{k}"] = d, s_, (d / s_ if s_ else None)
            pp.append(r)
        out["pointing"] = pp
        for eye in ("NL", "NR"):
            ps = [r for r in pp if r["eye"] == eye and r["images"] >= min_images]
            for k in ("cx", "cy"):
                v = _chi2([r[f"z_{k}"] for r in ps], [r[f"d_{k}"] for r in ps])
                if v:
                    out["chi2"][f"{eye}_{k}"] = v
                f = _wfit([r["T"] for r in ps], [r[f"d_{k}"] for r in ps], [r[f"sd_{k}"] or np.nan for r in ps])
                if f:
                    out["trends"][f"{eye}_{k}_vs_T"] = dict(f, units="px per degC")
        # the stereo pointing: NR's principal point about the law against NL's (both in px)
        st_rows = []
        for (s, e) in sorted({(r["scape"], r["early"]) for r in pp}):
            L = next((r for r in pp if r["scape"] == s and r["early"] == e and r["eye"] == "NL"), None)
            R = next((r for r in pp if r["scape"] == s and r["early"] == e and r["eye"] == "NR"), None)
            if not (L and R and L.get("sd_cx") and R.get("sd_cx")):
                continue
            row = {"block": s, "early": e, "T": 0.5 * (L["T"] + R["T"]), "images": L["images"] + R["images"]}
            for k in ("cx", "cy"):
                d = R[f"d_{k}"] - L[f"d_{k}"]
                sd = float(np.hypot(R[f"sd_{k}"], L[f"sd_{k}"])) if R.get(f"sd_{k}") and L.get(f"sd_{k}") else None
                row.update({f"d_{k}": d, f"sd_{k}": sd, f"z_{k}": d / sd if sd else None})
            st_rows.append(row)
        out["stereo"] = st_rows
        for k in ("cx", "cy"):
            v = _chi2([r[f"z_{k}"] for r in st_rows], [r[f"d_{k}"] for r in st_rows])
            if v:
                out["chi2"][f"stereo_{k}"] = v
    return out


def validate_laws(scapes_cfg: Dict[str, PathLike], laws: Dict[str, Optional[PathLike]], points: int = 4000,
                  iterations: int = 60, pointing: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Each law of ``laws`` {label: cameras folder} against the blocks: per law the law-held pass, the free pass and
    the pointing pass.  (25 Oct: one free pass from the first law compared with the others was biased - the free
    units hold the principal point and k2-k4 at the law they start from, so k1 and the rig pitch absorbed the other
    laws' differences in those: NR k1 chi^2 / dof 31 against 4.)"""
    labels = list(laws)
    st = {k: T.start_values(v) for k, v in laws.items()}
    rec0, proj0, idx = load_blocks(scapes_cfg, st[labels[0]], points)
    out: Dict[str, Any] = {"blocks": list(scapes_cfg), "points_per_block": points,
                           "laws": {k: str(st[k]["folder"]) for k in labels},
                           "images": {s: len(v) for s, v in idx["images"].items()},
                           "free": {}, "pointing_pass": {}, "held": {}, "deviations": {}}
    for k in labels:
        free = units_free(rec0, proj0, idx, st[k], iterations=iterations, verbose=verbose)
        point = (T.pointing_pass(rec0, proj0, idx, st[k], law_globals(st[k]), iterations=iterations, verbose=verbose)
                 if pointing else None)
        out["free"][k] = {kk: free[kk] for kk in ("blocks", "all", "brief", "final_cost", "seconds", "variance_factor",
                                                  "camera_cov_dims", "covariance_error")}
        if point:
            out["pointing_pass"][k] = {kk: point.get(kk) for kk in ("residuals", "brief", "covariance_error", "pp_prior_px")}
        out["held"][k] = law_held(rec0, proj0, idx, st[k], iterations=iterations)
        out["deviations"][k] = deviations(free, st[k], point)
        if verbose:
            for line in describe(out, k):
                log.info(line)
    return out


def describe(res: Dict[str, Any], label: str) -> List[str]:
    h, d = res["held"][label], res["deviations"][label]
    f = res["free"][label] if label in res["free"] else res["free"]
    out = [f"law {label} ({res['laws'][label]}): held {h['all']['median_px']:.4f} / {h['all']['rms_px']:.4f} px "
           f"(median / rms), units free {f['all']['median_px']:.4f} / {f['all']['rms_px']:.4f} px"]
    for s in sorted(h["blocks"]):
        a, b = h["blocks"][s], f["blocks"].get(s)
        if b:
            out.append(f"  {s:28s} held {a['median_px']:.4f} / {a['rms_px']:.4f}  free {b['median_px']:.4f} / "
                       f"{b['rms_px']:.4f}  cost of the law {a['median_px'] - b['median_px']:+.4f} / "
                       f"{a['rms_px'] - b['rms_px']:+.4f} px; corners {a['corner_median_px'] - b['corner_median_px']:+.4f}")
    for k, c in sorted(d["chi2"].items()):
        out.append(f"  {k:9s} chi2/dof {c['chi2_dof']:9.2f} (robust {c['robust']:8.2f}), mean z {c['mean_z']:+6.2f}, "
                   f"{c['n']} units; median |dev| {c['median_abs_dev']:.4g}")
    for k, t in sorted(d["trends"].items()):
        if t.get("z") is not None and abs(t["z"]) >= 3:
            out.append(f"  trend {k}: {t['slope']:+.4g} +- {t['se']:.2g} {t['units']} (z {t['z']:+.1f})")
    for k in ("rig_sol_T_r", "camera_sol_T_r"):
        if d.get(k) is not None:
            out.append(f"  {k}: {d[k]:+.2f}")
    return out


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def write_validation(res: Dict[str, Any], out: PathLike) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "law_validation.json").write_text(json.dumps(res, indent=1, default=_default), encoding="utf-8")
    lines: List[str] = []
    for k in res["held"]:
        lines += describe(res, k) + [""]
    (out / "report.md").write_text("# Navcam laws against the blocks\n\n```\n" + "\n".join(lines) + "```\n",
                                   encoding="utf-8")
    return out
