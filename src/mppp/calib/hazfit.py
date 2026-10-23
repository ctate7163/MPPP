"""
The Hazcam law from the finished Navcam blocks (0.93, Christian 22 Oct 2026): the front (FL, FR) and rear (RL, RR)
Hazcam cameras in the Navcam form - focal lengths linear in the camera temperature (one ppm/degC per eye, as the
Navcam's iso slope), k1 linear in temperature, focal-length offsets before the early-mission sol - and each pair's
stereo rig with a rotation drift linear in sol.

The blocks carry the Hazcams since 0.93: each block refines fx, fy, k1, k2 of every eye with enough tie points
(``hazcam_intrinsics = "radial"``; a weak eye stays at its label fit) and, with ``hazcam_rig_refine = "rotation"``,
the rotation of each pair's rig.  This fit gathers, per eye and block, the refined camera at the block's median
camera temperature and sol, and fits the law to them by weighted least squares (weights: the eye's tie-point
observations in the block); cx, cy, p1, p2, k3, k4 are the median of the blocks' start cameras (held in the blocks).
The rig rotations of the blocks give the drift (rate per axis about sol 0).  The result is a set of candidate files
in the frozen form (``M2020_FL.json`` ... ``M2020_F_rig.json``, ``M2020_R_rig.json``), checked
(:func:`check_hazcam`) and promoted on request (``mppp calib hazcam --promote-if-ok``).

    res = fit_hazcam(blocks)             # {name: WORK} of finished Navcam blocks with Hazcams
    w = write_hazcam(res, out_dir)       # the candidate files, hazcam_fit.json, report.md, check.json
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import forms

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]

T0_DEGC = -20.0                    # the reference temperature of the Navcam form
EARLY_MISSION_SOL = 380.0          # as the Navcam law's early_mission.before_sol
MIN_BLOCKS = 3                     # blocks per eye for a law (refined, not held as weak)
MIN_T_SPAN_DEGC = 5.0              # temperature span for a slope (else 0)
MIN_EARLY_BLOCKS = 2               # blocks on each side of the early-mission sol for an offset (else 0)
MIN_RIG_BLOCKS = 3                 # blocks with a refined rig for a drift (else the median rotation, no drift)
HAZCAM_LIMITS = {"max_ppm_per_degC": 100.0, "max_f0_change_pct": 1.0, "max_aspect_dev": 0.02,
                 "max_scatter_ppm": 600.0, "max_rig_change_deg": 0.5, "min_blocks": MIN_BLOCKS}
EYES = ("FL", "FR", "RL", "RR")


def _median_T(rows: List[Dict[str, Any]]) -> Optional[float]:
    v = [float(r["camera_temperature_degC"]) for r in rows if r.get("camera_temperature_degC") is not None]
    return float(np.median(v)) if v else None


def block_rows(name: str, work: PathLike, model: str = "cahv_ba") -> Dict[str, Any]:
    """The Hazcam results of one finished block: per eye {"T", "sol", "params" (refined), "start", "observations",
    "weak", "images"}; per family the rig {"R" (refined), "R_start", "t", "sol", "refined"}."""
    import pycolmap
    root = Path(work)
    if (root / "colmap" / "project.json").is_file():
        root = root / "colmap"
    pj = json.loads((root / "project.json").read_text(encoding="utf-8"))
    st = pj.get("settings") or {}
    db = (st.get("database") or {}).get("cameras") or {}
    recst = st.get("reconstruction") or {}
    cov = (recst.get("hazcam") or {}).get("eyes") or {}
    weak = set()
    for e in recst.get("log") or []:
        if isinstance(e, dict) and e.get("hazcam_held_weak"):
            weak |= set(e["hazcam_held_weak"])
    hz = st.get("hazcam") or {}
    out: Dict[str, Any] = {"block": name, "eyes": {}, "rigs": {}, "intrinsics": hz.get("intrinsics"),
                           "rig_refine": hz.get("rig_refine")}
    if not hz:
        return out
    rec = pycolmap.Reconstruction(str(root / "sparse" / model))
    reg = {rec.images[i].name for i in rec.reg_image_ids()}
    for eye in EYES:
        cam = (pj.get("cameras") or {}).get(eye)
        cid = db.get(eye)
        if cam is None or cid is None or int(cid) not in rec.cameras:
            continue
        rows = [r for r in pj.get("images") or [] if str(r.get("instrument")) == eye and r.get("name") in reg]
        if not rows:
            continue
        sols = [float(r["sol"]) for r in rows if r.get("sol") is not None]
        out["eyes"][eye] = {"params": [float(x) for x in rec.cameras[int(cid)].params], "start": list(map(float, cam["params"])),
                            "T": _median_T(rows), "sol": float(np.median(sols)) if sols else None, "images": len(rows),
                            "observations": int((cov.get(eye) or {}).get("observations") or 0),
                            "weak": eye in weak or hz.get("intrinsics") == "hold",
                            "early_share": float(np.mean([s < EARLY_MISSION_SOL for s in sols])) if sols else 0.0}
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))      # noqa: E731
    for fam in ("F", "R"):
        r0 = (pj.get("rig") or {}).get(fam)
        cid = db.get(f"{fam}R")
        if not r0 or cid is None:
            continue
        for rig in rec.rigs.values():
            if any(s.id == int(cid) for s in rig.non_ref_sensors):
                T = rig.sensor_from_rig(S(cid))
                e = out["eyes"].get(f"{fam}L") or out["eyes"].get(f"{fam}R") or {}
                out["rigs"][fam] = {"R": np.asarray(T.rotation.matrix()).tolist(), "R_start": r0["R_sensor_from_ref"],
                                    "t": list(map(float, r0["t_sensor_from_ref"])), "sol": e.get("sol"),
                                    "refined": hz.get("rig_refine") == "rotation" and fam not in set(hz.get("rig_hold") or ())}
                break
    return out


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares: (beta, covariance scaled by the residual variance, residuals)."""
    W = np.sqrt(np.asarray(w, float))
    A, b = X * W[:, None], y * W
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(np.sum(w * r ** 2) / max(np.sum(w), 1e-12) * len(y) / dof) if len(y) > X.shape[1] else float("nan")
    try:
        cov = np.linalg.inv(A.T @ A) * (s2 if np.isfinite(s2) else 1.0) * max(np.sum(w), 1e-12) / len(y)
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    return beta, cov, r


def fit_eye(rows: List[Dict[str, Any]], eye: str, athermal: bool = False, min_blocks: Optional[int] = None) -> Dict[str, Any]:
    """The law of one eye from its block rows (:func:`block_rows`; weak eyes left out).  ``athermal`` (0.93.10): no
    temperature terms - fx, fy and k1 constant (the fallback when the temperature slope fails its check)."""
    use = [r for r in rows if not r["weak"] and r["T"] is not None and r["observations"] > 0]
    starts = np.array([r["start"] for r in rows], float) if rows else np.zeros((0, 12))
    out: Dict[str, Any] = {"eye": eye, "blocks": [r["block"] for r in use], "weak_blocks": [r["block"] for r in rows if r["weak"]]}
    need = int(min_blocks or MIN_BLOCKS)                  # 0.94.1: --min-blocks (the tiny joints)
    if len(use) < need:
        out["skipped"] = f"{len(use)} blocks with this eye refined (>= {need} needed)"
        return out
    P = np.array([r["params"] for r in use], float)
    T = np.array([r["T"] for r in use], float) - T0_DEGC
    w = np.array([max(r["observations"], 1) for r in use], float)
    early = np.array([r["early_share"] for r in use], float)
    n_early, n_late = int(np.sum(early > 0.5)), int(np.sum(early <= 0.5))
    fit_slope = float(np.ptp(T)) >= MIN_T_SPAN_DEGC and not athermal
    fit_early = n_early >= MIN_EARLY_BLOCKS and n_late >= MIN_EARLY_BLOCKS
    # fx, fy: separate intercepts, one slope g (px/degC) shared by both, an early offset per axis
    nrow = len(use)
    cols = ["ax", "ay"] + (["g"] if fit_slope else []) + (["ex", "ey"] if fit_early else [])
    X = np.zeros((2 * nrow, len(cols)))
    X[:nrow, 0] = 1.0
    X[nrow:, 1] = 1.0
    if fit_slope:
        X[:, cols.index("g")] = np.r_[T, T]
    if fit_early:
        X[:nrow, cols.index("ex")] = early
        X[nrow:, cols.index("ey")] = early
    y = np.r_[P[:, 0], P[:, 1]]
    beta, cov, r = _wls(X, y, np.r_[w, w])
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    v = dict(zip(cols, beta))
    s = dict(zip(cols, sd))
    fbar = 0.5 * (v["ax"] + v["ay"])
    ppm = 1e6 * v.get("g", 0.0) / fbar
    # k1 linear in temperature, k2 the weighted mean
    Xk = np.c_[np.ones(nrow), T] if fit_slope else np.ones((nrow, 1))
    bk, ck, _ = _wls(Xk, P[:, 4], w)
    k2 = float(np.average(P[:, 5], weights=w))
    held = np.median(starts, axis=0)
    params = [float(v["ax"]), float(v["ay"]), float(held[2]), float(held[3]), float(bk[0]), k2,
              float(held[6]), float(held[7]), float(held[8]), float(held[9]), 0.0, 0.0]
    scatter_ppm = float(1e6 * np.sqrt(np.average(r ** 2, weights=np.r_[w, w])) / fbar)
    out.update({"params": params, "ppm_per_degC": float(ppm),
                "sd_ppm_per_degC": float(1e6 * s.get("g", np.nan) / fbar) if fit_slope else None,
                "k1_per_degC": float(bk[1]) if fit_slope else 0.0,
                "sd_k1_per_degC": float(np.sqrt(max(ck[1, 1], 0))) if fit_slope else None,
                "early_mission": {"dfx_px": float(v.get("ex", 0.0)), "dfy_px": float(v.get("ey", 0.0)),
                                  "fitted": fit_early, "blocks_early": n_early, "blocks_late": n_late},
                "sd": {"fx": float(s["ax"]), "fy": float(s["ay"])}, "scatter_ppm": scatter_ppm,
                "slope_fitted": fit_slope, "T_span_degC": float(np.ptp(T)), "athermal": bool(athermal),
                "label_start": [float(x) for x in held[:6]],
                "rows": [{"block": r_["block"], "T": r_["T"], "sol": r_["sol"], "fx": r_["params"][0], "fy": r_["params"][1],
                          "k1": r_["params"][4], "observations": r_["observations"]} for r_ in use]})
    return out


def fit_rig(rows: List[Dict[str, Any]], fam: str) -> Dict[str, Any]:
    """The rig of a Hazcam pair from the blocks' rig rotations: the rotation at sol 0 and a linear drift (pitch,
    yaw, roll about x, y, z of the right camera, mdeg per sol) - or, with too few refined rigs, the median CAHV rig
    without a drift."""
    from scipy.spatial.transform import Rotation
    refined = [r for r in rows if r["refined"] and r["sol"] is not None]
    allr = [r for r in rows]
    out: Dict[str, Any] = {"family": fam, "blocks": [r["block"] for r in refined]}
    if not allr:
        out["skipped"] = "no block with this Hazcam pair"
        return out
    R0 = Rotation.from_rotvec(np.median([Rotation.from_matrix(np.asarray(r["R_start"])).as_rotvec() for r in allr], axis=0))
    t = np.median(np.asarray([r["t"] for r in allr], float), axis=0)
    out.update({"t": t.tolist(), "baseline_m": float(np.linalg.norm(t))})
    if len(refined) < MIN_RIG_BLOCKS:
        Rm = R0.as_matrix() if not refined else Rotation.from_rotvec(np.median(
            [Rotation.from_matrix(np.asarray(r["R"])).as_rotvec() for r in refined], axis=0)).as_matrix()
        out.update({"R": Rm.tolist(), "drift": None, "note": f"{len(refined)} refined rigs: no drift "
                    f"(>= {MIN_RIG_BLOCKS} needed); the {'refined' if refined else 'CAHV'} median rotation"})
        return out
    off = np.array([(Rotation.from_matrix(np.asarray(r["R"])) * R0.inv()).as_rotvec() for r in refined]) * 1e3 * 180 / np.pi
    sols = np.array([r["sol"] for r in refined], float)
    X = np.c_[np.ones(len(sols)), sols]
    coef = np.linalg.lstsq(X, off, rcond=None)[0]            # (2, 3): intercept, rate per sol (mdeg)
    res = off - X @ coef
    R_sol0 = Rotation.from_rotvec(np.radians(1e-3 * coef[0])) * R0
    out.update({"R": R_sol0.as_matrix().tolist(), "change_from_cahv_mdeg": float(np.linalg.norm(coef[0])),
                "drift": {"model": "linear", "sol0": 0.0, "pitch_mdeg_per_sol": float(coef[1, 0]),
                          "yaw_mdeg_per_sol": float(coef[1, 1]), "roll_mdeg_per_sol": float(coef[1, 2]),
                          "blocks": len(refined)},
                "scatter_mdeg": [float(x) for x in np.sqrt(np.mean(res ** 2, axis=0))]})
    return out


def fit_hazcam(blocks: Dict[str, PathLike], model: str = "cahv_ba", limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The Hazcam law of every eye and pair from the finished blocks {name: WORK}."""
    rows = []
    for n, w in blocks.items():
        try:
            rows.append(block_rows(n, w, model))
        except (OSError, ValueError) as e:
            log.info(f"hazcam fit: {n} left out ({type(e).__name__}: {e})")
    per_eye = {e: [dict(r["eyes"][e], block=r["block"]) for r in rows if e in r["eyes"]] for e in EYES}
    per_fam = {f: [dict(r["rigs"][f], block=r["block"]) for r in rows if f in r["rigs"]] for f in ("F", "R")}
    return {"when": datetime.date.today().isoformat(), "blocks": sorted(r["block"] for r in rows if r["eyes"]),
            "T0_degC": T0_DEGC, "early_mission_sol": EARLY_MISSION_SOL,
            "eyes": {e: fit_eye_with_fallback(rs, e, limits) for e, rs in per_eye.items() if rs}, "limits": limits or {},
            "rigs": {f: fit_rig(rs, f) for f, rs in per_fam.items() if rs}}


def fit_eye_with_fallback(rows: List[Dict[str, Any]], eye: str,
                          limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    mb = (limits or {}).get("min_blocks")
    """0.93.10 (Christian, 23 Oct): the eye's law with its temperature terms; when that law fails its thermal check
    (FL of 23 Oct: +151 ppm/degC), the athermal law (fx, fy, k1 constant) instead if it passes every check - else the
    temperature law, with the athermal attempt recorded (``athermal_tried``)."""
    e = fit_eye(rows, eye, min_blocks=mb)
    if e.get("skipped") or not e.get("slope_fitted"):
        return e
    checks = eye_checks(e, eye, limits)
    if dict((n, o) for n, o, _ in checks).get(f"{eye} thermal", True):
        return e
    a = fit_eye(rows, eye, athermal=True, min_blocks=mb)
    ca = eye_checks(a, eye, limits)
    th = f"{e['ppm_per_degC']:+.1f} ppm/degC"
    if all(o for _, o, _ in ca):
        a["fallback"] = (f"athermal: the temperature law failed its thermal check ({th}); fx, fy, k1 constant, "
                         f"block scatter {a['scatter_ppm']:.0f} ppm")
        a["thermal_fit"] = {k: e.get(k) for k in ("ppm_per_degC", "sd_ppm_per_degC", "k1_per_degC", "scatter_ppm")}
        return a
    e["athermal_tried"] = {"scatter_ppm": a.get("scatter_ppm"),
                           "failed": [n for n, o, _ in ca if not o]}
    return e


def camera_file(eye_res: Dict[str, Any], stamp: str) -> Dict[str, Any]:
    """An eye's law in the Navcam form (the ``M2020_FL.json`` layout)."""
    e = eye_res
    fam, eye = e["eye"][0], e["eye"][1]
    return {"model": "THIN_PRISM_FISHEYE", "width": 5120, "height": 3840, "params": e["params"],
            "param_names": list(forms.NAVCAM_PARAMS),
            "distortion": "fisheye (theta polynomial k1-k4) + tangential p1, p2; sx1 = sy1 = 0",
            "free_params": [], "fixed_params": [], "pixel_origin": "corner of the first pixel (COLMAP)",
            "form": "navcam-v1", "convention": f"{fam}L reference", "family": fam, "eye": eye,
            "thermal": {"T0_degC": T0_DEGC, "fx_ppm_per_degC": e["ppm_per_degC"], "fy_ppm_per_degC": e["ppm_per_degC"],
                        "cx_px_per_degC": 0.0, "cy_px_per_degC": 0.0, "k1_per_degC": e["k1_per_degC"],
                        "sd_ppm_per_degC": e.get("sd_ppm_per_degC"), "sd_k1_per_degC": e.get("sd_k1_per_degC"),
                        "note": "one ppm/degC for fx and fy; the principal point held (the blocks hold it)"},
            "early_mission": {"before_sol": EARLY_MISSION_SOL, "per_eye": True,
                              "dfx_px": e["early_mission"]["dfx_px"], "dfy_px": e["early_mission"]["dfy_px"],
                              "dcx_px": 0.0, "dcy_px": 0.0,
                              "applied_terms": ["fx", "fy"] if e["early_mission"]["fitted"] else [],
                              "note": "Hazcam fit: focal-length offsets of the images before before_sol"},
            "sd": e.get("sd"),
            "source": (f"MPPP Hazcam fit (mppp.calib.hazfit, {stamp}) over {len(e['blocks'])} blocks "
                       f"({', '.join(e['blocks'])}): "
                       + ("athermal fallback - fx, fy, k1 constant (the temperature law failed its thermal check)"
                          if e.get("athermal") else "fx, fy, k1 linear in temperature")
                       + "; k2 the blocks' mean; cx, cy, p1, p2, k3, k4 the label fits' median"),
            "fit": {"method": "hazcam fit from the blocks", "blocks": e["blocks"], "scatter_ppm": e["scatter_ppm"],
                    "T_span_degC": e["T_span_degC"], "rows": e["rows"]}}


def rig_file(rig_res: Dict[str, Any], stamp: str) -> Dict[str, Any]:
    fam = rig_res["family"]
    d = {"ref": f"{fam}L", "sensor": f"{fam}R", "R_sensor_from_ref": rig_res["R"], "t_sensor_from_ref_m": rig_res["t"],
         "baseline_m": rig_res["baseline_m"], "translation": "cahv", "family": fam, "form": "navcam-v1",
         "convention": f"{fam}L reference",
         "source": f"MPPP Hazcam fit (mppp.calib.hazfit, {stamp}): {rig_res.get('note') or 'rotation at sol 0 and a linear drift'}"}
    if rig_res.get("drift"):
        d["drift"] = rig_res["drift"]
    return d


def eye_checks(e: Dict[str, Any], eye: str, limits: Optional[Dict[str, Any]] = None) -> List[Tuple[str, bool, str]]:
    """The checks of one eye's law: [(name, ok, note)]."""
    lim = dict(HAZCAM_LIMITS, **(limits or {}))
    c: List[Tuple[str, bool, str]] = []
    if e.get("skipped"):
        return [(f"{eye} blocks", False, e["skipped"])]
    c.append((f"{eye} blocks", len(e["blocks"]) >= lim["min_blocks"], f"{len(e['blocks'])} blocks"))
    if e.get("athermal"):
        c.append((f"{eye} thermal", True, "athermal (fallback): " + str(e.get("fallback") or "no temperature terms")))
    else:
        c.append((f"{eye} thermal", abs(e["ppm_per_degC"]) <= lim["max_ppm_per_degC"],
                  f"{e['ppm_per_degC']:+.1f} ppm/degC" + (f" +- {e['sd_ppm_per_degC']:.1f}" if e.get("sd_ppm_per_degC") else "")
                  + ("" if e["slope_fitted"] else " (not fitted: temperature span under 5 degC)")
                  + (f"; the athermal law failed too ({', '.join(e['athermal_tried']['failed'])}; scatter "
                     f"{e['athermal_tried']['scatter_ppm']:.0f} ppm)" if e.get("athermal_tried") else "")))
    f0 = 0.5 * (e["params"][0] + e["params"][1])
    fl = 0.5 * (e["label_start"][0] + e["label_start"][1])
    d = 100 * abs(f0 / fl - 1.0)
    c.append((f"{eye} f0", d <= lim["max_f0_change_pct"], f"{fl:.1f} -> {f0:.1f} px ({d:.2f} % from the label fits)"))
    a = e["params"][1] / e["params"][0]
    c.append((f"{eye} aspect", abs(a - 1.0) <= lim["max_aspect_dev"], f"{a:.5f}"))
    c.append((f"{eye} block scatter", e["scatter_ppm"] <= lim["max_scatter_ppm"],
              f"{e['scatter_ppm']:.0f} ppm about the law"))
    return c


def check_hazcam(res: Dict[str, Any], limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sanity limits per eye and pair (:data:`HAZCAM_LIMITS`); an eye or pair passes or fails on its own:
    {"ok" (all), "eyes": {eye: ok}, "rigs": {fam: ok}, "checks": [...]}."""
    lim = dict(HAZCAM_LIMITS, **(limits or {}))
    checks: List[Dict[str, Any]] = []
    ok_e: Dict[str, bool] = {}
    for eye, e in (res.get("eyes") or {}).items():
        c = eye_checks(e, eye, lim)
        ok_e[eye] = all(x[1] for x in c)
        checks += [{"check": n, "ok": bool(o), "note": t} for n, o, t in c]
    ok_r: Dict[str, bool] = {}
    for fam, r in (res.get("rigs") or {}).items():
        if r.get("skipped"):
            ok_r[fam] = False
            checks.append({"check": f"{fam} rig", "ok": False, "note": r["skipped"]})
            continue
        ch = 1e-3 * float(r.get("change_from_cahv_mdeg") or 0.0)
        ok_r[fam] = ch <= lim["max_rig_change_deg"]
        checks.append({"check": f"{fam} rig", "ok": ok_r[fam],
                       "note": f"baseline {r['baseline_m']:.4f} m; " + (r.get("note") or
                               f"{ch:.3f} deg from the CAHV rig at sol 0, drift over {r['drift']['blocks']} blocks")})
    return {"ok": all(ok_e.values()) and all(ok_r.values()) and bool(ok_e), "eyes": ok_e, "rigs": ok_r,
            "checks": checks}


def write_hazcam(res: Dict[str, Any], out_dir: PathLike) -> Dict[str, Any]:
    """The candidate files of the eyes and pairs that pass (:func:`check_hazcam`) in ``out_dir``, with
    ``hazcam_fit.json``, ``check.json`` and ``report.md``.  Returns {"folder", "files" (passing), "check"}."""
    from .models import CAMERA_FILE, RIG_FILE
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = res.get("when") or datetime.date.today().isoformat()
    chk = check_hazcam(res, res.get("limits") or None)
    cand = out / "candidate"
    cand.mkdir(exist_ok=True)
    files = []
    for eye, e in (res.get("eyes") or {}).items():
        if e.get("skipped"):
            continue
        name = CAMERA_FILE[(eye[0], eye[1])]
        (out / name).write_text(json.dumps(camera_file(e, stamp), indent=1), encoding="utf-8")
        if chk["eyes"].get(eye):
            (cand / name).write_text(json.dumps(camera_file(e, stamp), indent=1), encoding="utf-8")
            files.append(str(cand / name))
    for fam, r in (res.get("rigs") or {}).items():
        if r.get("skipped"):
            continue
        name = RIG_FILE[fam]
        (out / name).write_text(json.dumps(rig_file(r, stamp), indent=1), encoding="utf-8")
        # 0.93.8: a rig that passes its own checks is a candidate whatever its eyes' laws do (23 Oct: the rear rig,
        # 0.267 deg from the CAHV rig in all blocks, was held back by the rear eyes' thermal fits) - the blocks hold
        # the principal points at their start, so the rig rotation does not depend on the eyes' focal laws
        if chk["rigs"].get(fam):
            (cand / name).write_text(json.dumps(rig_file(r, stamp), indent=1), encoding="utf-8")
            files.append(str(cand / name))
    (out / "hazcam_fit.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (out / "check.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
    lines = [f"# Hazcam fit ({stamp})", "", f"Blocks: {', '.join(res.get('blocks') or [])}.", ""]
    lines += [f"- {x}" for x in describe(res)]
    lines += ["", "## Checks (each eye and pair on its own; the passing ones are in candidate/)", ""]
    lines += [f"- {'ok' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}" for c in chk["checks"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"folder": str(out), "candidate": str(cand), "files": files, "check": chk}


def describe(res: Dict[str, Any]) -> List[str]:
    out = []
    for eye, e in (res.get("eyes") or {}).items():
        if e.get("skipped"):
            out.append(f"{eye}: skipped - {e['skipped']}" + (f" (weak in {', '.join(e['weak_blocks'])})" if e.get("weak_blocks") else ""))
            continue
        p = e["params"]
        out.append(f"{eye}: fx {p[0]:.2f}, fy {p[1]:.2f} px at {T0_DEGC:g} degC, {e['ppm_per_degC']:+.1f}"
                   + (f" +- {e['sd_ppm_per_degC']:.1f}" if e.get("sd_ppm_per_degC") else "") + " ppm/degC, k1 "
                   f"{p[4]:+.5f} ({e['k1_per_degC']:+.2e}/degC), k2 {p[5]:+.5f}; early offsets "
                   f"{e['early_mission']['dfx_px']:+.2f}, {e['early_mission']['dfy_px']:+.2f} px"
                   + ("" if e["early_mission"]["fitted"] else " (not fitted)")
                   + f"; {len(e['blocks'])} blocks over {e['T_span_degC']:.1f} degC, scatter {e['scatter_ppm']:.0f} ppm"
                   + (f" - {e['fallback']}" if e.get("fallback") else ""))
    for fam, r in (res.get("rigs") or {}).items():
        if r.get("skipped"):
            out.append(f"{fam} rig: skipped - {r['skipped']}")
        elif r.get("drift"):
            d = r["drift"]
            out.append(f"{fam} rig: baseline {r['baseline_m']:.4f} m; drift pitch {1e3 * d['pitch_mdeg_per_sol']:+.2f}, yaw "
                       f"{1e3 * d['yaw_mdeg_per_sol']:+.2f}, roll {1e3 * d['roll_mdeg_per_sol']:+.2f} mdeg/1000 sol over "
                       f"{d['blocks']} blocks; scatter {', '.join(f'{x:.1f}' for x in r['scatter_mdeg'])} mdeg")
        else:
            out.append(f"{fam} rig: baseline {r['baseline_m']:.4f} m; {r.get('note')}")
    return out


__all__ = ["fit_hazcam", "write_hazcam", "check_hazcam", "block_rows", "fit_eye", "fit_rig", "HAZCAM_LIMITS"]
