"""
Error-model parameters measured from MPPP COLMAP alignments (v0p14.7).

An alignment made by notebook 03 writes ``<WORK>/colmap/error_input/``
(:func:`mppp.sfm.export_for_error`): the native-pixel COLMAP model, per-image
station table, pose-vs-prior table, per-observation residuals and a summary.
This module turns one or more of those folders into the numbers the error
model (:mod:`mppp.error.core`) runs on, so that they can be compared with the
values it assumes (``ModelConfig`` defaults, ``cases.CASE_PARAMS``):

========================  ===================================================
``eps_table``             image-measurement precision eps = rms residual / sqrt(2),
                          per instrument, same-station vs cross-station tracks,
                          and against range
``pair_survival``         every geometrically possible image pair of a sample
                          of points: convergence angle, same/cross station,
                          |dLMST|, matched or not
``fit_gate``              the cross-station gate A (1 + theta/theta_c)^-k exp(-dL/L0),
                          theta_c = theta_bar/CV^2, k = 1/CV^2, fitted to those
                          pairs relative to the same-station rate (binomial
                          maximum likelihood, optional bootstrap over points)
``decorrelation``         rho(theta) of pairwise triangulation residuals
                          (``colmap.measure_theta_c``) on a sample of tracks
``view_graph``            measured station graph (shared tracks) as a ViewGraph
``registration``          BA pose minus telemetry prior, per station
``parameters``            all of the above as one row
========================  ===================================================

Caveats (see docs/error/mppp_error_README_v0p15.md, "Calibrating from real
data"): sparse tracks are biased towards well-matched terrain, so eps is a
lower bound and the survival curves are optimistic; "possible" pairs ignore
occlusion and the MPPP masks, which lowers the same-station and cross-station
rates alike (the gate is their ratio).
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..colmap import project_camera
from .colmap import ColmapModel, measure_theta_c, measured_view_graph, read_colmap

PathLike = Union[str, Path]
MARS_DAY_H = 24.0                     # LMST hours per sol

__all__ = ["Alignment", "load_alignment", "find_error_input", "eps_table", "eps_by_range", "pair_survival",
           "survival_curve", "gate_curve", "gate_expected", "fit_gate", "combine_pairs", "decorrelation", "view_graph", "registration",
           "parameters", "model_defaults"]


# ------------------------------------------------------------------ loading
@dataclass
class Alignment:
    label: str
    root: Path                              # the error_input folder
    model: ColmapModel
    images: Dict[str, Dict[str, Any]]       # image name -> stations.csv row (+ lmst_h, family)
    poses: List[Dict[str, Any]]
    summary: Dict[str, Any]
    residuals: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def stations(self) -> List[str]:
        return sorted({r["station"] for r in self.images.values()})

    def station_label(self, st: str) -> str:
        for r in self.images.values():
            if r["station"] == st and r.get("station_label"):
                return r["station_label"]
        return st


def find_error_input(path: PathLike) -> Path:
    """``path`` = an error_input folder, the COLMAP project folder, or the WORK folder above it."""
    p = Path(path)
    for cand in (p, p / "error_input", p / "colmap" / "error_input"):
        if (cand / "stations.csv").is_file() and (cand / "native").is_dir():
            return cand
    raise FileNotFoundError(f"no error_input (stations.csv + native/) at {p}, {p / 'error_input'} or "
                            f"{p / 'colmap' / 'error_input'}: run notebook 03 to the export step first")


def _lmst_hours(s: Any) -> Optional[float]:
    m = re.search(r"M(\d+):(\d+):(\d+(?:\.\d*)?)", str(s or ""))
    return None if not m else int(m.group(1)) + int(m.group(2)) / 60 + float(m.group(3)) / 3600


def _family(instrument: str) -> str:
    """NL/NR -> 'Navcam', ZL034/ZR0xx -> 'Mastcam-Z', anything else as is."""
    s = str(instrument).upper()
    return "Navcam" if s[:1] == "N" else "Mastcam-Z" if s[:1] == "Z" else s


def load_alignment(path: PathLike, label: Optional[str] = None) -> Alignment:
    root = find_error_input(path)
    model = read_colmap(str(root / "native"))
    with (root / "stations.csv").open(newline="", encoding="utf-8") as f:
        images = {r["name"]: dict(r) for r in csv.DictReader(f)}
    for r in images.values():
        r["lmst_h"] = _lmst_hours(r.get("lmst"))
        r["family"] = _family(r.get("instrument", ""))
        r.setdefault("station_label", r["station"])
    model.assign_stations({n: r["station"] for n, r in images.items()})
    poses = []
    if (root / "poses.csv").is_file():
        with (root / "poses.csv").open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for k, v in r.items():
                    try:
                        r[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                poses.append(r)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8")) if (root / "summary.json").is_file() else {}
    res = {}
    if (root / "residuals.npz").is_file():
        z = np.load(root / "residuals.npz")
        res = {k: z[k] for k in z.files}
    if label is None:                                    # <WORK>/colmap/error_input -> WORK name
        label = root.parent.parent.name if root.parent.name == "colmap" else root.parent.name
    return Alignment(label, root, model, images, poses, summary, res)


# ---------------------------------------------------------------------- eps
def _observations(al: Alignment) -> Dict[str, np.ndarray]:
    """Per observation: residual [native px], instrument family, instrument, station, cross-station track, range."""
    m = al.model
    track_st = {pid: {m.images[i].station for i in p.image_ids if int(i) in m.images} for pid, p in m.points.items()}
    r = al.residuals
    if r and "residual_native_px" in r and "point3D_id" in r:
        res, iid, pid = r["residual_native_px"], r["image_id"].astype(int), r["point3D_id"].astype(int)
        if "point2D_idx" in r and np.mean([int(q) in m.points for q in pid[:2000]]) < 0.5:
            # 0.14.5+: the native model numbers its points afresh; look them up by keypoint index
            p2d = r["point2D_idx"].astype(int)
            pid = np.array([int(m.images[int(i)].point3D_ids[k]) if int(i) in m.images and
                            k < m.images[int(i)].point3D_ids.size else -1 for i, k in zip(iid, p2d)])
    else:                                                # no residuals.npz: recompute from the model
        from .colmap import observation_residuals
        o = observation_residuals(m)
        res, iid = o["residual_px"], o["image_id"].astype(int)
        pid = None                                       # observation_residuals has no point ids
    if pid is not None:
        ok = np.array([int(i) in m.images and int(p) in m.points for i, p in zip(iid, pid)], bool)
        res, iid, pid = res[ok], iid[ok], pid[ok]
    names = np.array([m.images[int(i)].name for i in iid])
    inst = np.array([al.images.get(n, {}).get("instrument", "?") for n in names])
    fam = np.array([_family(x) for x in inst])
    st = np.array([m.images[int(i)].station for i in iid])
    if pid is not None:
        cross = np.array([len(track_st[int(p)]) > 1 for p in pid], bool)
        centres = {i: m.images[i].center for i in m.images}
        rng = np.array([np.linalg.norm(m.points[int(p)].xyz - centres[int(i)]) for p, i in zip(pid, iid)])
    else:
        cross = np.zeros(res.size, bool)
        rng = np.full(res.size, np.nan)
    return {"residual_px": np.asarray(res, float), "family": fam, "instrument": inst, "station": st,
            "cross": cross, "range_m": rng}


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def eps_table(al: Alignment) -> List[Dict[str, Any]]:
    """
    eps = rms(residual)/sqrt(2) (one image axis, the error model's convention),
    for all observations and per instrument family / instrument, split by
    whether the track stays at one station ("intra") or spans stations ("cross").
    Residuals are post-fit, so these are lower bounds.
    """
    o = al.__dict__.setdefault("_obs", _observations(al))
    rows = []
    groups = [("all", np.ones(o["residual_px"].size, bool))]
    groups += [(f, o["family"] == f) for f in sorted(set(o["family"]))]
    if len(set(o["instrument"])) > 1:
        groups += [(i, o["instrument"] == i) for i in sorted(set(o["instrument"]))]
    for name, g in groups:
        for kind, sel in (("all", g), ("intra", g & ~o["cross"]), ("cross", g & o["cross"])):
            r = o["residual_px"][sel]
            rows.append({"alignment": al.label, "group": name, "tracks": kind, "n_obs": int(r.size),
                         "rms_px": _rms(r), "eps_px": _rms(r) / np.sqrt(2.0),
                         "median_px": float(np.median(r)) if r.size else float("nan")})
    return rows


def eps_by_range(al: Alignment, edges_m: Sequence[float] = (0, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100),
                 family: Optional[str] = None) -> Dict[str, np.ndarray]:
    """eps [px] against observation range (point to camera centre), intra and cross tracks."""
    o = al.__dict__.setdefault("_obs", _observations(al))
    e = np.asarray(edges_m, float)
    sel0 = np.ones(o["residual_px"].size, bool) if family is None else o["family"] == family
    out = {"range_m": 0.5 * (e[:-1] + e[1:])}
    for kind, sel in (("intra", sel0 & ~o["cross"]), ("cross", sel0 & o["cross"])):
        k = np.digitize(o["range_m"][sel], e) - 1
        r = o["residual_px"][sel]
        out[f"eps_{kind}_px"] = np.array([_rms(r[k == j]) / np.sqrt(2) if np.any(k == j) else np.nan
                                          for j in range(e.size - 1)])
        out[f"n_{kind}"] = np.array([int(np.sum(k == j)) for j in range(e.size - 1)])
    return out


# ------------------------------------------------------------ pair survival
def pair_survival(al: Alignment, n_points: int = 5000, seed: int = 0, margin_px: float = 8.0,
                  max_theta_deg: float = 60.0, conditional: bool = True) -> Dict[str, np.ndarray]:
    """
    For ``n_points`` randomly chosen reconstructed points, every image pair in
    which the point is geometrically visible (in front of both cameras and
    inside both frames by ``margin_px``), and whether the track contains both
    images.  Per pair: ``point`` (index in the sample), ``theta_deg``
    (convergence angle at the point), ``cross`` (different stations),
    ``dlmst_h`` (|LMST difference|, wrapped), ``families`` ("Navcam-Navcam",
    ...), ``weight`` (number of possible trials) and ``observed``.

    ``conditional`` (default): a trial is "the point is matched in image a;
    is it also matched in image b?", counted both ways - so ``weight`` is the
    number of the two images whose track contains the point (pairs where
    neither does are dropped) and the rate is P(b | a).  This keeps terrain
    that was never matchable (occluded, masked rover pixels, no texture) out
    of the denominator.  ``conditional=False``: every geometrically possible
    pair is one trial (``match_survival``'s definition), which counts those
    too and gives much lower rates.

    Points sit where matching worked at least once, so the rates are upper
    bounds.
    """
    m = al.model
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(m.points))
    if n_points and pids.size > n_points:
        pids = rng.choice(pids, n_points, replace=False)
    X = np.array([m.points[int(p)].xyz for p in pids])                 # (P,3)
    iids = sorted(m.images)
    n_img, n_pt = len(iids), len(pids)
    col = {int(i): k for k, i in enumerate(iids)}
    vis = np.zeros((n_img, n_pt), bool)
    obs = np.zeros((n_img, n_pt), bool)
    dirs = np.zeros((n_img, n_pt, 3))
    for a, p in enumerate(pids):
        for i in m.points[int(p)].image_ids:
            if int(i) in col:
                obs[col[int(i)], a] = True
    for k, i in enumerate(iids):
        im = m.images[i]
        cam = m.cameras[im.camera_id]
        xc = X @ im.R.T + im.tvec
        front = xc[:, 2] > 1e-6
        z = np.where(front, xc[:, 2], 1.0)
        # pinhole position inside a generous frame, so strong distortion cannot fold far points back in
        one_f = cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL")
        f = float(cam.params[0])
        cx, cy = (float(cam.params[1]), float(cam.params[2])) if one_f else (float(cam.params[2]), float(cam.params[3]))
        up, vp = f * xc[:, 0] / z + cx, f * xc[:, 1] / z + cy
        near = front & (up > -0.5 * cam.width) & (up < 1.5 * cam.width) & (vp > -0.5 * cam.height) & \
            (vp < 1.5 * cam.height)
        uv = project_camera(cam.model, cam.params, np.where(near[:, None], xc, [[0, 0, 1.0]]))
        inside = near & (uv[:, 0] >= margin_px) & (uv[:, 0] < cam.width - margin_px) & \
            (uv[:, 1] >= margin_px) & (uv[:, 1] < cam.height - margin_px)
        vis[k] = inside | obs[k]                        # an observation is possible by definition
        d = X - im.center
        dirs[k] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    rows = [al.images.get(m.images[i].name, {}) for i in iids]
    st = [m.images[i].station for i in iids]
    lm = [r.get("lmst_h") for r in rows]
    fam = [r.get("family", "?") for r in rows]
    out: Dict[str, List[np.ndarray]] = {k: [] for k in ("point", "theta_deg", "cross", "dlmst_h", "families",
                                                        "observed", "weight", "image_a", "image_b")}
    cos_min = np.cos(np.radians(max_theta_deg))
    for a in range(n_img):
        va = vis[a]
        if not va.any():
            continue
        for b in range(a + 1, n_img):
            both = va & vis[b]
            if not both.any():
                continue
            idx = np.nonzero(both)[0]
            if conditional:
                idx = idx[obs[a, idx] | obs[b, idx]]
                if not idx.size:
                    continue
            c = np.einsum("ij,ij->i", dirs[a, idx], dirs[b, idx])
            keep = c >= cos_min
            idx, c = idx[keep], c[keep]
            if not idx.size:
                continue
            n = idx.size
            dl = np.nan if lm[a] is None or lm[b] is None else abs(lm[a] - lm[b]) % MARS_DAY_H
            dl = min(dl, MARS_DAY_H - dl) if np.isfinite(dl) else dl
            out["point"].append(idx.astype(np.int32))
            out["theta_deg"].append(np.degrees(np.arccos(np.clip(c, -1, 1))).astype(np.float32))
            out["cross"].append(np.full(n, st[a] != st[b]))
            out["dlmst_h"].append(np.full(n, dl, np.float32))
            out["families"].append(np.full(n, "-".join(sorted((fam[a], fam[b])))))
            both_obs = obs[a, idx] & obs[b, idx]
            out["observed"].append(both_obs)
            out["weight"].append((obs[a, idx].astype(np.int8) + obs[b, idx]) if conditional
                                 else np.ones(n, np.int8))
            out["image_a"].append(np.full(n, a, np.int32))
            out["image_b"].append(np.full(n, b, np.int32))
    res = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}
    res["n_points"] = n_pt
    res["label"] = al.label
    res["conditional"] = bool(conditional)
    return res


def _select(pairs: Dict[str, Any], families: Optional[str] = None, cross: Optional[bool] = None,
            dl_max_h: Optional[float] = None) -> np.ndarray:
    sel = np.ones(pairs["observed"].size, bool)
    if families:
        sel &= pairs["families"] == families
    if cross is not None:
        sel &= pairs["cross"] == cross
    if dl_max_h is not None:
        sel &= np.nan_to_num(pairs["dlmst_h"], nan=0.0) <= dl_max_h
    return sel


def survival_curve(pairs: Dict[str, Any], edges_deg: Sequence[float] = tuple(np.arange(0, 31, 1.0)),
                   families: Optional[str] = None, cross: Optional[bool] = True,
                   dl_max_h: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Matched / possible pairs per convergence-angle bin, with a binomial (Wilson) 68 % interval."""
    e = np.asarray(edges_deg, float)
    sel = _select(pairs, families, cross, dl_max_h)
    k = np.digitize(pairs["theta_deg"][sel], e) - 1
    ok = (k >= 0) & (k < e.size - 1)
    w = pairs["weight"][sel][ok].astype(float) if "weight" in pairs else np.ones(int(ok.sum()))
    n = np.bincount(k[ok], weights=w, minlength=e.size - 1)
    s = np.bincount(k[ok], weights=w * pairs["observed"][sel][ok], minlength=e.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s / n
        z = 1.0
        den = 1 + z * z / n
        mid = (p + z * z / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"theta_deg": 0.5 * (e[:-1] + e[1:]), "n_possible": n, "n_observed": s, "rate": p,
            "lo": mid - half, "hi": mid + half}


def _half_angle(sc: Dict[str, np.ndarray], min_trials: float = 50.0) -> float:
    """Angle where the survival curve first falls below half its value in the first two bins (interpolated)."""
    ok = np.isfinite(sc["rate"]) & (sc["n_possible"] >= min_trials)
    th, r = sc["theta_deg"][ok], sc["rate"][ok]
    if r.size < 3:
        return float("nan")
    half = 0.5 * np.mean(r[:2])
    below = np.nonzero(r < half)[0]
    if not below.size:
        return float("nan")                         # does not halve inside the range
    j = int(below[0])
    if j == 0:
        return float(th[0])
    return float(th[j - 1] + (half - r[j - 1]) / (r[j] - r[j - 1]) * (th[j] - th[j - 1]))


def gate_curve(theta_deg: np.ndarray, A: float, theta_bar_deg: float, cv: float) -> np.ndarray:
    """A (1 + theta/theta_c)^-k with theta_c = theta_bar/CV^2, k = 1/CV^2 (core.gate_powerlaw)."""
    cv2 = max(cv, 1e-6) ** 2
    return A * (1.0 + np.asarray(theta_deg, float) / (theta_bar_deg / cv2)) ** (-1.0 / cv2)


def fit_gate(pairs: Dict[str, Any], families: Optional[str] = None, L0_h: float = 2.3, fit_L0: bool = False,
             theta_max_deg: float = 30.0, n_boot: int = 0, seed: int = 0) -> Dict[str, Any]:
    """
    Maximum-likelihood fit of the cross-station gate to the pair data:

        P(matched | cross, theta, dL) = s_intra * A (1 + theta/theta_c)^-k * exp(-dL/L0)

    s_intra = the same-station match rate (pooled; A is relative to it, as in
    ``ModelConfig.gate_A``).  ``L0_h`` is held (default 2.3 h, the archive
    value) unless ``fit_L0`` - which needs cross-station pairs over a range of
    |dLMST|.  ``n_boot`` > 0 adds a bootstrap over the sampled points (16-84 %).
    """
    from scipy.optimize import minimize

    def binned(weights):
        pw = pairs["weight"].astype(float) if "weight" in pairs else np.ones(pairs["observed"].size)
        sel_i = _select(pairs, families, cross=False)
        wi = weights[pairs["point"][sel_i]] * pw[sel_i]
        s_intra = np.sum(wi * pairs["observed"][sel_i]) / max(np.sum(wi), 1e-12)
        sel = _select(pairs, families, cross=True) & (pairs["theta_deg"] <= theta_max_deg)
        th = np.round(pairs["theta_deg"][sel] * 2) / 2                   # 0.5 deg cells
        dl = np.round(np.nan_to_num(pairs["dlmst_h"][sel], nan=0.0) * 4) / 4
        w = weights[pairs["point"][sel]] * pw[sel]
        key, inv = np.unique(np.stack([th, dl], 1), axis=0, return_inverse=True)
        inv = inv.ravel()
        n = np.bincount(inv, weights=w)
        s = np.bincount(inv, weights=w * pairs["observed"][sel])
        return float(s_intra), key[:, 0], key[:, 1], n, s

    def solve(weights):
        s_intra, th, dl, n, s = binned(weights)
        if s_intra <= 0 or n.sum() == 0:
            return None

        def nll(q):
            A, thb, cv = np.exp(q[0]), np.exp(q[1]), np.exp(q[2])
            L0 = np.exp(q[3]) if fit_L0 else L0_h
            p = s_intra * gate_curve(th, A, thb, cv) * np.exp(-dl / L0)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -float(np.sum(s * np.log(p) + (n - s) * np.log(1 - p)))
        q0 = [np.log(0.4), np.log(4.3), np.log(0.36)] + ([np.log(L0_h)] if fit_L0 else [])
        best = None
        for start in (q0, [np.log(0.2), np.log(2.0), np.log(0.6)] + q0[3:], [np.log(0.8), np.log(8.0), np.log(0.25)] + q0[3:]):
            r = minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        A, thb, cv = np.exp(best.x[:3])
        return {"A": float(A), "A_absolute": float(A * s_intra), "theta_bar_deg": float(thb), "cv": float(cv),
                "L0_h": float(np.exp(best.x[3])) if fit_L0 else float(L0_h), "s_intra": s_intra,
                "nll": float(best.fun), "n_cross_pairs": float(n.sum()), "matched_cross_pairs": float(s.sum())}

    n_pt = int(pairs["n_points"])
    out = solve(np.ones(n_pt))
    if out is None:
        return {"families": families or "all", "error": "no same-station matches or no cross-station pairs"}
    # the power-law parameters are only determined if they are not at a limit of the form
    out["constrained"] = bool(out["theta_bar_deg"] < theta_max_deg and out["cv"] > 0.02)
    # shape-free numbers: the half-survival angle and the goodness of fit per 1-deg bin
    edges = np.arange(0.0, theta_max_deg + 1e-9, 1.0)
    sc = survival_curve(pairs, edges, families, cross=True)
    pred = gate_expected(pairs, out["s_intra"], out["A"], out["theta_bar_deg"], out["cv"], out["L0_h"], edges, families)
    ok = (sc["n_possible"] > 0) & np.isfinite(pred) & (pred > 0) & (pred < 1)
    chi2 = np.sum((sc["n_observed"][ok] - sc["n_possible"][ok] * pred[ok]) ** 2 /
                  (sc["n_possible"][ok] * pred[ok] * (1 - pred[ok])))
    out["chi2_dof"] = float(chi2 / max(int(ok.sum()) - (4 if fit_L0 else 3), 1))
    out["theta_half_deg"] = _half_angle(sc)
    out.update(families=families or "all", theta_max_deg=theta_max_deg, fit_L0=fit_L0,
               n_intra_pairs=int(np.sum(_select(pairs, families, cross=False))),
               dlmst_h_range=[float(np.nanmin(pairs["dlmst_h"][_select(pairs, families, True)]))
                              if np.any(_select(pairs, families, True)) else float("nan"),
                              float(np.nanmax(pairs["dlmst_h"][_select(pairs, families, True)]))
                              if np.any(_select(pairs, families, True)) else float("nan")])
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(int(n_boot)):
            w = np.bincount(rng.integers(0, n_pt, n_pt), minlength=n_pt).astype(float)
            b = solve(w)
            if b:
                boots.append([b["A"], b["theta_bar_deg"], b["cv"], b["L0_h"]])
        if boots:
            q = np.percentile(np.array(boots), [16, 84], axis=0)
            for j, k in enumerate(("A", "theta_bar_deg", "cv", "L0_h")):
                out[f"{k}_16_84"] = [float(q[0, j]), float(q[1, j])]
    return out


def gate_expected(pairs: Dict[str, Any], s_intra: float, A: float, theta_bar_deg: float, cv: float, L0_h: float,
                  edges_deg: Sequence[float] = tuple(np.arange(0, 31, 1.0)), families: Optional[str] = None,
                  dl_max_h: Optional[float] = None) -> np.ndarray:
    """
    Cross-station match rate per angle bin predicted by a gate, averaged over
    the actual |dLMST| of the pairs in each bin - comparable with
    :func:`survival_curve` of the same pairs.
    """
    e = np.asarray(edges_deg, float)
    sel = _select(pairs, families, True, dl_max_h)
    th = pairs["theta_deg"][sel]
    dl = np.nan_to_num(pairs["dlmst_h"][sel], nan=0.0)
    w = pairs["weight"][sel].astype(float) if "weight" in pairs else np.ones(th.size)
    pr = s_intra * gate_curve(th, A, theta_bar_deg, cv) * np.exp(-dl / max(L0_h, 1e-9))
    k = np.digitize(th, e) - 1
    ok = (k >= 0) & (k < e.size - 1)
    n = np.bincount(k[ok], weights=w[ok], minlength=e.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.bincount(k[ok], weights=(w * pr)[ok], minlength=e.size - 1) / n


def combine_pairs(pair_sets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool :func:`pair_survival` results of several alignments (point indices kept distinct)."""
    keys = ("point", "theta_deg", "cross", "dlmst_h", "families", "observed", "weight", "image_a", "image_b")
    out: Dict[str, Any] = {k: [] for k in keys}
    off = 0
    for p in pair_sets:
        for k in keys:
            v = p[k] if k in p else np.ones(p["observed"].size, np.int8)
            out[k].append(v + off if k == "point" else v)
        off += int(p["n_points"])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update(n_points=off, label="+".join(str(p.get("label")) for p in pair_sets),
               conditional=all(p.get("conditional", False) for p in pair_sets))
    return res


# ------------------------------------------------------ decorrelation, graph
def decorrelation(al: Alignment, n_points: int = 4000, seed: int = 0, theta_max_deg: float = 5.0,
                  n_bins: int = 10, min_track: int = 3) -> Dict[str, Any]:
    """
    rho(theta) of pairwise triangulation residuals (``colmap.measure_theta_c``)
    on ``n_points`` tracks of >= ``min_track`` images, over 0-``theta_max_deg``
    (the model uses theta_c ~0.4 deg, so fine bins at small angles matter).
    """
    import copy
    m = al.model
    ids = np.array([p for p, pt in m.points.items() if pt.track_length >= min_track])
    rng = np.random.default_rng(seed)
    if ids.size > n_points:
        ids = rng.choice(ids, n_points, replace=False)
    sub = copy.copy(m)
    sub.points = {int(i): m.points[int(i)] for i in ids}
    out = measure_theta_c(sub, n_bins=n_bins, theta_max_deg=theta_max_deg, min_track=min_track)
    out["n_tracks"] = int(ids.size)
    return out


def view_graph(al: Alignment):
    """Measured station graph: W = tracks shared by two stations (a ``viewgraph.ViewGraph``)."""
    from .viewgraph import ViewGraph
    W, names, th = measured_view_graph(al.model)
    mx = W.max() if W.size and W.max() > 0 else 1.0
    return ViewGraph(W=W, names=names, shared_fraction=W / mx, mean_theta_deg=th, mean_w=np.ones_like(W))


def registration(al: Alignment) -> List[Dict[str, Any]]:
    """Per station: images, median |BA - prior| position [m] and attitude [deg], and its E/N/U components."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in al.poses:
        if isinstance(r.get("observations"), float) and r["observations"] < 30:
            continue                                            # held / weakly observed images
        by.setdefault(r["station"], []).append(r)
    rows = []
    for st, rs in sorted(by.items()):
        g = lambda k: np.array([x[k] for x in rs if isinstance(x.get(k), float)])     # noqa: E731
        rows.append({"alignment": al.label, "station": st, "station_label": al.station_label(st),
                     "images": len(rs), "dC_median_m": float(np.median(g("dC_m"))),
                     "dE_median_m": float(np.median(g("dE_m"))), "dN_median_m": float(np.median(g("dN_m"))),
                     "dU_median_m": float(np.median(g("dU_m"))),
                     "dAttitude_median_deg": float(np.median(g("dAttitude_deg"))) if g("dAttitude_deg").size
                     else float("nan")})
    return rows


# ------------------------------------------------------------------ summary
def model_defaults() -> Dict[str, Any]:
    """The values the error model assumes (ModelConfig defaults and the archive 'measured' case)."""
    from .cases import CASE_PARAMS
    from .core import ModelConfig
    c = ModelConfig()
    A, thb, cv, L0, _ = CASE_PARAMS["measured"]
    out = {"gate_A": A, "theta_bar_deg": thb, "gate_cv": cv, "L0_h": L0,
           "theta_c_deg": c.theta_c_deg, "rho_0": c.rho_0, "rho_inf": c.rho_inf,
           "correlation_kernel": c.correlation_kernel, "eps_intra_px (archive, pooled)": 0.169,
           "reg_sigma_m (telemetry registration)": 0.3}
    out["cases"] = {k: {"A": v[0], "theta_bar_deg": v[1], "cv": v[2], "L0_h": v[3], "note": v[4]}
                    for k, v in CASE_PARAMS.items()}
    return out


def parameters(al: Alignment, gate: Optional[Dict[str, Any]] = None,
               rho: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One row of measured error-model inputs for an alignment (plus its size)."""
    eps = {(r["group"], r["tracks"]): r for r in eps_table(al)}
    tr = al.summary.get("tracks", {})
    reg = registration(al)
    row = {"alignment": al.label, "images": len(al.model.images), "stations": len(al.stations),
           "points": len(al.model.points), "cross_station_fraction": tr.get("cross_station_fraction"),
           "eps_intra_px": eps[("all", "intra")]["eps_px"], "eps_cross_px": eps[("all", "cross")]["eps_px"],
           "registration_dC_median_m": float(np.median([r["dC_median_m"] for r in reg])) if reg else float("nan")}
    for fam in ("Navcam", "Mastcam-Z"):
        if (fam, "intra") in eps:
            row[f"eps_intra_px[{fam}]"] = eps[(fam, "intra")]["eps_px"]
            row[f"eps_cross_px[{fam}]"] = eps[(fam, "cross")]["eps_px"]
    if gate and "A" in gate:
        row.update(gate_A=gate["A"], theta_bar_deg=gate["theta_bar_deg"], gate_cv=gate["cv"], L0_h=gate["L0_h"],
                   gate_families=gate.get("families"), gate_constrained=gate.get("constrained"),
                   same_station_rate=gate.get("s_intra"))
    if rho is not None:
        ok = np.isfinite(rho["rho"]) & (rho["n"] > 50)
        row["rho_first_bin"] = float(rho["rho"][ok][0]) if ok.any() else float("nan")
        row["rho_mean_0_5deg"] = float(np.nanmean(rho["rho"][ok])) if ok.any() else float("nan")
    return row
