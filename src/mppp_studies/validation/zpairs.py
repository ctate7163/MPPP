"""
mppp_studies.validation.zpairs - test 3: the Mastcam-Z rig rotation against sol from Mastcam-Z pairs alone.

The Mastcam-Z joints measure each block's ZL-ZR rig with the Navcam law held (12 Nov 2026: pitch +5.4 +- 1.5 mdeg /
1000 sols at 34 mm, +5.3 +- 1.7 at 110 mm, the Navcam's own rate).  This test uses no Navcam, no bundle adjustment
and no tie to other frames: one simultaneous ZL / ZR exposure, SIFT matches between the two eyes, and the rotation of
ZR against ZL that makes the matches satisfy the epipolar constraint - the baseline direction held at the rig file's
(hardware), the three small angles ``w = (w_x, w_y, w_z)`` free:

    x_R^T [t]_x  dR(w) R0  x_L = 0        x: undistorted rays, R0 and t: the rig file

``w_x`` (about ZR's x axis, the baseline: "pitch", the vertical alignment of the eyes) is seen directly as vertical
parallax and is well determined; ``w_z`` (roll) from its change across the frame; ``w_y`` (yaw, the vergence) only
moves points along their epipolar lines and is NOT determined by a pair (reported, not used).

The intrinsics are deliberately plain: each image's focal length from its block camera (the focus line), ONE
principal point per eye and zoom (the focus model's ``pp0_px``) and the model's distortion.  So ``w_x`` is

    w_x = rig pitch + (dcy_R - dcy_L) / f

and a principal-point change in one eye cannot be told from a rig pitch in one pair.  Across zooms it can: the same
camera looks through the same detector at 34 and 110 mm, so a detector / axis shift gives the same PIXELS at both
zooms (3.1 times the angle at 34 mm), a rotation of the camera gives the same ANGLE.  :func:`trend` fits

    w_x = a + r (sol - 1000) / 1000 + c (focus - F0) / 1000  [+ knee]

per zoom, and :func:`compare_zooms` the two readings.

    python -m mppp_studies.validation.zpairs --root D:\\rover_blocks --pds D:\\data\\m2020\\zcam --zooms 34 110
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.validation")
PathLike = Union[str, Path]
FULL_FRAME = (1648, 1200)
MIN_INLIERS = 150
KNEE_SOL = 250.0                  # the Navcam rig law's knee
EL_RANGE_DEG = (-30.0, 3.0)       # frames looking at terrain (not the sky, not the deck)


# ------------------------------------------------------------------------------------------------ selection
def block_pairs(project_json: PathLike, zooms: Sequence[int], per_block: int = 4, seed: int = 7) -> List[Dict[str, Any]]:
    """Simultaneous full-frame ZL / ZR exposures of one block (its ``project.json``), ``per_block`` per zoom, spread
    over sols and sequences, boresight elevation within :data:`EL_RANGE_DEG`."""
    import random
    rnd = random.Random(seed)
    pj = Path(project_json)
    P = json.loads(pj.read_text(encoding="utf-8"))
    block = pj.parent.parent.name
    cams = P.get("cameras") or {}
    by: Dict[Tuple[int, str, str], Dict[str, Dict[str, Any]]] = collections.defaultdict(dict)
    for r in P["images"]:
        if r["name"][:2] not in ("ZL", "ZR"):
            continue
        m = re.search(r"(\d{3})", str(r.get("camera_group", "")))
        if not m or int(m.group(1)) not in [int(z) for z in zooms]:
            continue
        by[(int(m.group(1)), r["sclk_key"], r["sequence"])][r["eye"]] = r
    groups: Dict[int, Dict[Tuple[int, str], List[Any]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for (zoom, sclk, seq), d in by.items():
        if len(d) != 2 or any(tuple(d[e].get("native_size") or ()) != FULL_FRAME for e in "LR"):
            continue
        L, R = d["L"], d["R"]
        v = np.asarray(L["prior_R_w2c"], float).T @ np.array([0.0, 0.0, 1.0])
        el = math.degrees(math.asin(max(-1.0, min(1.0, float(v[2])))))
        if not EL_RANGE_DEG[0] <= el <= EL_RANGE_DEG[1]:
            continue
        groups[zoom][(int(L["sol"]), seq)].append((el, L, R))
    out = []
    for zoom, g in groups.items():
        keys = sorted(g)
        for k in keys:
            rnd.shuffle(g[k])
        i, n = 0, 0
        while n < per_block and any(g.values()):
            k = keys[i % len(keys)]
            i += 1
            if not g[k]:
                continue
            el, L, R = g[k].pop()
            n += 1
            out.append({"block": block, "zoom": zoom, "sol": int(L["sol"]), "sequence": L["sequence"], "sclk": L["sclk_key"],
                        "el_deg": round(el, 1), "L": L["name"], "R": R["name"],
                        "focus_L": L.get("focus_count"), "focus_R": R.get("focus_count"),
                        "T_optics_degC": L.get("zcam_optics_temperature_degC"),
                        "state_L": L.get("focus_state"), "state_R": R.get("focus_state"),
                        "cam_L": (cams.get(L["instrument"]) or {}).get("params"),
                        "cam_R": (cams.get(R["instrument"]) or {}).get("params")})
    return out


# ------------------------------------------------------------------------------------------------ images, matches
def read_gray(path: PathLike) -> np.ndarray:
    """A PDS ``.IMG`` (or any image file) as an 8-bit grey image, stretched between its 0.5 and 99.5 percentiles."""
    p = Path(path)
    if p.suffix.upper() == ".IMG":
        from mppp.labels import ensure_hwc, read_pds
        _, a = read_pds(p)
        a = ensure_hwc(np.asarray(a)).astype(np.float32)
        g = a.mean(axis=2) if a.ndim == 3 else a
    else:
        import cv2
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ok = np.isfinite(g) & (g > 0)
    if not ok.any():
        return np.zeros(g.shape, np.uint8)
    lo, hi = np.percentile(g[ok], [0.5, 99.5])
    return np.clip((g - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)


def match_pair(gL: np.ndarray, gR: np.ndarray, max_features: int = 12000, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """SIFT matches ZL -> ZR (ratio test, mutual), (n, 2) pixel arrays (pixel centres at integers + 0.5 as COLMAP)."""
    import cv2
    sift = cv2.SIFT_create(nfeatures=int(max_features), contrastThreshold=0.02)
    kL, dL = sift.detectAndCompute(gL, None)
    kR, dR = sift.detectAndCompute(gR, None)
    if dL is None or dR is None or len(kL) < 20 or len(kR) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = {}
    for m, n in (p for p in bf.knnMatch(dL, dR, k=2) if len(p) == 2):
        if m.distance < ratio * n.distance:
            fwd[m.queryIdx] = m.trainIdx
    back = {}
    for m, n in (p for p in bf.knnMatch(dR, dL, k=2) if len(p) == 2):
        if m.distance < ratio * n.distance:
            back[m.queryIdx] = m.trainIdx
    pairs = [(q, t) for q, t in fwd.items() if back.get(t) == q]
    a = np.array([kL[q].pt for q, _ in pairs], float).reshape(-1, 2) + 0.5
    b = np.array([kR[t].pt for _, t in pairs], float).reshape(-1, 2) + 0.5
    return a, b


def rays(xy: np.ndarray, params: Sequence[float], pp: Optional[Sequence[float]] = None) -> np.ndarray:
    """Pixels -> undistorted normalised coordinates (n, 2) through a FULL_OPENCV camera (fx, fy, cx, cy, k1, k2,
    p1, p2, k3 ...); ``pp``: the principal point used instead of the camera's."""
    import cv2
    p = [float(v) for v in params]
    cx, cy = (float(pp[0]), float(pp[1])) if pp is not None else (p[2], p[3])
    K = np.array([[p[0], 0, cx], [0, p[1], cy], [0, 0, 1.0]])
    dist = np.array((p[4:12] + [0.0] * 8)[:8], float)
    return cv2.undistortPoints(np.asarray(xy, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)


def _skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0.0]])


def _rot(w):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(w, float)).as_matrix()


def sampson(xL: np.ndarray, xR: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Signed Sampson distances (normalised units) of x_R^T [t]_x R x_L = 0."""
    E = _skew(t / np.linalg.norm(t)) @ R
    a = np.c_[xL, np.ones(len(xL))]
    b = np.c_[xR, np.ones(len(xR))]
    Ea = a @ E.T
    Etb = b @ E
    num = np.sum(b * Ea, axis=1)
    den = np.sqrt(Ea[:, 0] ** 2 + Ea[:, 1] ** 2 + Etb[:, 0] ** 2 + Etb[:, 1] ** 2)
    return num / np.maximum(den, 1e-12)


def relative_rotation(xL: np.ndarray, xR: np.ndarray, R0: np.ndarray, t: np.ndarray, f_px: float,
                      gate_px: float = 3.0, free: Sequence[bool] = (True, False, True)) -> Dict[str, Any]:
    """The small rotation ``w`` (rad, in ZR's frame, applied after ``R0``) that fits the matches with the baseline
    direction ``t`` held: soft-L1 on the Sampson distance in pixels (``f_px``), two passes (a wide gate, then
    ``gate_px``).  ``free``: which of (w_x, w_y, w_z) are fitted - the yaw w_y is not determined by a pair and is held
    at 0 by default.  Returns {"w_mdeg" (3,), "sd_mdeg" (3,), "inliers", "rms_px", "n"}."""
    from scipy.optimize import least_squares
    idx = [i for i, f in enumerate(free) if f]

    def full(v):
        w = np.zeros(3)
        w[idx] = v
        return w

    def res(v, sel):
        return f_px * sampson(xL[sel], xR[sel], _rot(full(v)) @ R0, t)

    sel = np.ones(len(xL), bool)
    v = np.zeros(len(idx))
    out: Dict[str, Any] = {"n": int(len(xL))}
    for gate in (30.0, 8.0, gate_px):
        if sel.sum() < 20:
            return dict(out, inliers=int(sel.sum()), error="too few matches")
        r = least_squares(res, v, args=(sel,), loss="soft_l1", f_scale=1.0)
        v = r.x
        d = np.abs(f_px * sampson(xL, xR, _rot(full(v)) @ R0, t))
        sel = d < gate
    r = least_squares(res, v, args=(sel,), loss="linear")
    v = r.x
    rr = res(v, sel)
    dof = max(1, len(rr) - len(idx))
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * float(rr @ rr) / dof
        sd = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        sd = np.full(len(idx), np.nan)
    w, s = full(v), np.full(3, np.nan)
    s[idx] = sd
    out.update(w_mdeg=(np.degrees(w) * 1e3).tolist(), sd_mdeg=(np.degrees(s) * 1e3).tolist(), inliers=int(sel.sum()),
               rms_px=float(np.sqrt(np.mean(rr ** 2))), inlier_mask=sel)
    return out


def measure_pair(pair: Dict[str, Any], model: Dict[str, Any], pds: PathLike, images: Optional[PathLike] = None) -> Dict[str, Any]:
    """One pair -> its row: the matches, ``w`` and what describes the frame.  ``model``: the zoom's focus model (the
    rig and each eye's ``pp0_px``); ``pds``: the Mastcam-Z archive (``<sol 5 digits>/<stem>.IMG``)."""
    zoom = int(pair["zoom"])
    row = {k: pair.get(k) for k in ("block", "zoom", "sol", "sequence", "sclk", "el_deg", "focus_L", "focus_R",
                                    "T_optics_degC", "state_L", "state_R", "L", "R")}
    files = {}
    for e in "LR":
        stem = Path(pair[e]).stem
        cands = [Path(pds) / f"{int(pair['sol']):05d}" / f"{stem}.IMG", Path(pds) / f"{stem}.IMG"]
        if images:
            cands.append(Path(images) / pair[e])
        files[e] = next((c for c in cands if c.is_file()), None)
        if files[e] is None:
            return dict(row, error=f"{stem}: product not found")
    if not pair.get("cam_L") or not pair.get("cam_R"):
        return dict(row, error="no block camera")
    gL, gR = read_gray(files["L"]), read_gray(files["R"])
    if gL.shape[::-1] != FULL_FRAME or gR.shape[::-1] != FULL_FRAME:
        return dict(row, error=f"not full frames: {gL.shape}, {gR.shape}")
    a, b = match_pair(gL, gR)
    row["matches"] = int(len(a))
    if len(a) < MIN_INLIERS:
        return dict(row, error="too few matches")
    cams = model["cameras"]
    ppL = cams[f"ZL{zoom:03d}"]["pp0_px"]
    ppR = cams[f"ZR{zoom:03d}"]["pp0_px"]
    xL, xR = rays(a, pair["cam_L"], ppL), rays(b, pair["cam_R"], ppR)
    R0 = np.asarray(model["rig"]["R_sensor_from_ref"], float)
    t = np.asarray(model["rig"]["t_sensor_from_ref"], float)
    f = 0.5 * (float(pair["cam_L"][0]) + float(pair["cam_R"][0]))
    fit = relative_rotation(xL, xR, R0, t, f)
    if fit.get("error") or fit["inliers"] < MIN_INLIERS:
        return dict(row, error=fit.get("error") or "too few inliers", inliers=fit.get("inliers"))
    sel = fit.pop("inlier_mask")
    d = (a[sel, 0] - b[sel, 0])
    row.update(inliers=fit["inliers"], rms_px=round(fit["rms_px"], 3), f_px=round(f, 1),
               pitch_mdeg=fit["w_mdeg"][0], roll_mdeg=fit["w_mdeg"][2],
               pitch_sd_mdeg=fit["sd_mdeg"][0], roll_sd_mdeg=fit["sd_mdeg"][2],
               pitch_px=fit["w_mdeg"][0] * 1e-3 * math.pi / 180.0 * f,
               disparity_median_px=float(np.median(d)), disparity_iqr_px=float(np.subtract(*np.percentile(d, [75, 25]))),
               dy_median_px=float(np.median(b[sel, 1] - a[sel, 1])))
    return row


# ------------------------------------------------------------------------------------------------ trends
def _wls(A: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    Aw, yw = A * w[:, None], y * w
    x, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    r = yw - Aw @ x
    dof = max(1, len(y) - A.shape[1])
    cov = np.linalg.pinv(Aw.T @ Aw) * float(r @ r) / dof
    return x, np.sqrt(np.diag(cov)), float(np.sqrt(r @ r / dof))


def block_means(rows: Sequence[Dict[str, Any]], key: str = "pitch_mdeg") -> List[Dict[str, Any]]:
    """One value per block, zoom and sol group (visits 100 sols apart): the median of its pairs - the unit of the
    trend fit (pairs of one sequence share their systematic errors)."""
    g: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        if r.get(key) is None or r.get("error"):
            continue
        g[(r["block"], int(r["zoom"]), int(r["sol"]) // 100)].append(r)
    out = []
    for (block, zoom, _), v in sorted(g.items()):
        x = np.array([q[key] for q in v], float)
        out.append({"block": block, "zoom": zoom, "sol": float(np.median([q["sol"] for q in v])), "n": len(v),
                    key: float(np.median(x)), "spread": float(np.std(x)) if len(x) > 1 else None,
                    "focus": float(np.median([q["focus_L"] for q in v if q.get("focus_L") is not None] or [np.nan])),
                    "T": float(np.median([q["T_optics_degC"] for q in v if q.get("T_optics_degC") is not None] or [np.nan])),
                    "f_px": float(np.median([q["f_px"] for q in v]))})
    return out


def trend(units: Sequence[Dict[str, Any]], key: str = "pitch_mdeg", focus: bool = False, knee: bool = False) -> Dict[str, Any]:
    """``key = a + r (sol - 1000)/1000 [+ c (focus - median)/1000] [+ k min(sol - 250, 0)/1000]`` over the block
    units; se from the fit's own scatter, and a jackknife over the units."""
    u = [q for q in units if q.get(key) is not None and np.isfinite(q[key])]
    if len(u) < 4:
        return {"n": len(u), "error": "fewer than 4 units"}
    sol = np.array([q["sol"] for q in u], float)
    y = np.array([q[key] for q in u], float)
    cols, names = [np.ones(len(u)), (sol - 1000.0) / 1000.0], ["a", "rate_per_1000_sols"]
    if focus:
        fo = np.array([q["focus"] for q in u], float)
        fo = np.where(np.isfinite(fo), fo, np.nanmedian(fo))
        cols.append((fo - np.median(fo)) / 1000.0)
        names.append("per_1000_focus_counts")
    if knee:
        cols.append(np.minimum(sol - KNEE_SOL, 0.0) / 1000.0)
        names.append("extra_rate_before_knee")
    A = np.stack(cols, axis=1)
    x, sd, rms = _wls(A, y)
    jk = []
    for i in range(len(u)):
        m = np.ones(len(u), bool)
        m[i] = False
        jk.append(_wls(A[m], y[m])[0])
    jk = np.array(jk)
    sd_jk = np.sqrt((len(u) - 1) / len(u) * np.sum((jk - jk.mean(0)) ** 2, axis=0))
    return {"n": len(u), "terms": names, "value": x.tolist(), "se": sd.tolist(), "se_jackknife": sd_jk.tolist(),
            "rms": rms, "sol_range": [float(sol.min()), float(sol.max())]}


def compare_zooms(units: Sequence[Dict[str, Any]], zooms: Sequence[int] = (34, 110)) -> Dict[str, Any]:
    """The rotation and the principal-point readings of the drift: the rate per zoom in mdeg and in pixels.  A camera
    rotation gives one rate in mdeg at every zoom; a detector / axis shift one rate in pixels."""
    out: Dict[str, Any] = {"zooms": {}}
    for z in zooms:
        u = [dict(q) for q in units if int(q["zoom"]) == int(z)]
        for q in u:
            q["pitch_px"] = q["pitch_mdeg"] * 1e-3 * math.pi / 180.0 * q["f_px"]
        t1, t2 = trend(u, "pitch_mdeg"), trend(u, "pitch_px")
        if t1.get("error"):
            continue
        out["zooms"][int(z)] = {"n": t1["n"], "rate_mdeg": t1["value"][1], "se_mdeg": t1["se_jackknife"][1],
                                "rate_px": t2["value"][1], "se_px": t2["se_jackknife"][1],
                                "f_px": float(np.median([q["f_px"] for q in u]))}
    zs = sorted(out["zooms"])
    if len(zs) >= 2:
        a, b = out["zooms"][zs[0]], out["zooms"][zs[-1]]
        out["rotation_reading"] = {"difference_mdeg": b["rate_mdeg"] - a["rate_mdeg"],
                                   "se": math.hypot(a["se_mdeg"], b["se_mdeg"])}
        out["principal_point_reading"] = {"difference_px": b["rate_px"] - a["rate_px"],
                                          "se": math.hypot(a["se_px"], b["se_px"])}
        for k in ("rotation_reading", "principal_point_reading"):
            d = out[k]
            d["z"] = abs(next(v for kk, v in d.items() if kk.startswith("difference"))) / max(d["se"], 1e-12)
    return out


def write_csv(path: PathLike, rows: Sequence[Dict[str, Any]]) -> Path:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return Path(path)


def figure(rows, units, fits, out_png: PathLike, joint: Optional[Dict[int, Tuple[float, float]]] = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zooms = sorted({int(u["zoom"]) for u in units})
    fig, axes = plt.subplots(1, len(zooms), figsize=(6.2 * len(zooms), 4.4), squeeze=False)
    for ax, z in zip(axes[0], zooms):
        rr = [r for r in rows if not r.get("error") and int(r["zoom"]) == z]
        uu = [u for u in units if int(u["zoom"]) == z]
        ax.scatter([r["sol"] for r in rr], [r["pitch_mdeg"] for r in rr], s=10, c="0.7", label="pairs")
        ax.scatter([u["sol"] for u in uu], [u["pitch_mdeg"] for u in uu], s=38, c="#1f5fbf", zorder=3, label="block medians")
        f = fits.get(z)
        if f and not f.get("error"):
            s = np.linspace(0, 2050, 50)
            ax.plot(s, f["value"][0] + f["value"][1] * (s - 1000) / 1000, c="#c8502a",
                    label=f"pairs alone: {f['value'][1]:+.1f} +- {f['se_jackknife'][1]:.1f} mdeg / 1000 sols")
            if joint and z in joint:
                ax.plot(s, f["value"][0] + joint[z][0] * (s - 1000) / 1000, c="k", ls="--", lw=1,
                        label=f"joint (Navcam held): {joint[z][0]:+.1f} +- {joint[z][1]:.1f}")
        ax.axvline(KNEE_SOL, c="0.5", ls=":", lw=1)
        ax.set_title(f"Mastcam-Z {z} mm: ZR against ZL, rotation about the baseline")
        ax.set_xlabel("sol")
        ax.set_ylabel("w_x from pairs alone (mdeg, about the rig file's rotation)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return Path(out_png)


def run(pairs: Sequence[Dict[str, Any]], models: Dict[int, Dict[str, Any]], pds: PathLike, out: PathLike,
        joint: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(pairs):
        if int(p["zoom"]) not in models:
            continue
        r = measure_pair(p, models[int(p["zoom"])], pds)
        rows.append(r)
        log.info("%3d/%d %s Z%s sol %s: %s", i + 1, len(pairs), p["block"][14:34], p["zoom"], p["sol"],
                 r.get("error") or f"pitch {r['pitch_mdeg']:+.1f} +- {r['pitch_sd_mdeg']:.1f} mdeg, {r['inliers']} inliers, "
                                    f"rms {r['rms_px']:.2f} px")
    write_csv(out / "zpairs.csv", rows)
    units = block_means(rows)
    res: Dict[str, Any] = {"pairs": len(rows), "measured": sum(1 for r in rows if not r.get("error")), "units": units,
                           "trend": {}, "trend_focus": {}, "trend_knee": {}, "roll": {}}
    for z in sorted({int(u["zoom"]) for u in units}):
        uz = [u for u in units if int(u["zoom"]) == z]
        res["trend"][z] = trend(uz)
        res["trend_focus"][z] = trend(uz, focus=True)
        res["trend_knee"][z] = trend(uz, knee=True)
        res["roll"][z] = trend(block_means([r for r in rows if not r.get("error") and int(r["zoom"]) == z], "roll_mdeg"),
                               "roll_mdeg")
    res["zooms"] = compare_zooms(units, sorted(res["trend"]))
    res["joint"] = joint
    (out / "zpairs.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    try:
        figure(rows, units, res["trend"], out / "zpairs_pitch_vs_sol.png", joint)
    except Exception as e:                                   # noqa: BLE001 - the numbers without the figure
        log.warning("figure: %s", e)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the blocks' root (the ..._zcam folders)")
    ap.add_argument("--pds", required=True, help="the Mastcam-Z archive (<sol 5 digits>/<stem>.IMG)")
    ap.add_argument("--cmods", default=None, help="the folder of the focus models (default MPPP_CMODS / the models in use)")
    ap.add_argument("--zooms", nargs="+", type=int, default=[34, 110])
    ap.add_argument("--per-block", type=int, default=6)
    ap.add_argument("--pairs", default=None, help="a pairs.json made before (default: selected from the blocks)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import datetime
    import os
    root = Path(a.root)
    cm = Path(a.cmods or os.environ.get("MPPP_CMODS") or root / "cmods")
    models = {}
    for z in a.zooms:
        f = cm / f"M2020_ZCAM{int(z):03d}_focus_model.json"
        if f.is_file():
            models[int(z)] = json.loads(f.read_text(encoding="utf-8"))
    if a.pairs:
        pairs = json.loads(Path(a.pairs).read_text(encoding="utf-8"))
    else:
        pairs = []
        for pj in sorted(root.glob("*_zcam/colmap/project.json")):
            pairs += block_pairs(pj, a.zooms, a.per_block)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"validation_zpairs_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.json").write_text(json.dumps(pairs, indent=0, default=float), encoding="utf-8")
    res = run(pairs, models, a.pds, out)
    for z, t in res["trend"].items():
        if not t.get("error"):
            print(f"Z{z}: pitch {t['value'][1]:+.2f} +- {t['se_jackknife'][1]:.2f} mdeg / 1000 sols over {t['n']} block units "
                  f"(rms {t['rms']:.1f} mdeg)")
    print(out / "zpairs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
