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
``fit_gate``              the cross-station gate A (1 + theta/theta_c)^-k exp(-dL/tau),
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
from typing import Tuple, Any, Dict, List, Optional, Sequence, Union

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
    min_track_length: int = 2                   # v0p22: tie points with fewer images were left out
    points_before_track_filter: int = 0

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


def load_alignment(path: PathLike, label: Optional[str] = None, min_track_length: int = 2) -> Alignment:
    """
    One alignment's ``error_input``.  ``min_track_length`` (v0p22): keep only
    tie points observed in at least this many images (3 drops the two-view
    points, about half of all points and mostly single stereo pairs); every
    analysis then works on the kept points and their observations only.
    """
    root = find_error_input(path)
    model = read_colmap(str(root / "native"))
    n_all = len(model.points)
    if int(min_track_length) > 2:
        model.points = {k: p for k, p in model.points.items() if p.track_length >= int(min_track_length)}
    with (root / "stations.csv").open(newline="", encoding="utf-8") as f:
        images = {r["name"]: dict(r) for r in csv.DictReader(f)}
    for r in images.values():
        r["lmst_h"] = _lmst_hours(r.get("lmst"))
        r["family"] = _family(r.get("instrument", ""))
        r.setdefault("station_label", r["station"])
    _attach_solar_geometry(root, images)
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
    return Alignment(label, root, model, images, poses, summary, res, int(min_track_length), n_all)


def _attach_solar_geometry(root: Path, images: Dict[str, Dict[str, Any]]) -> None:
    """
    Per image: solar azimuth and elevation [deg, site frame, from the PDS label]
    as floats, from ``stations.csv`` (v0p22.2 exports) or else from the newest
    MPPP manifest beside the work folder (``<work>/processed/mppp_manifest_*.json``).
    Then the sun unit vector (east, north, up) and the shadow-tip vector of a
    unit post (site frame; length cot(elevation), pointing away from the sun).
    """
    need = [n for n, r in images.items() if not str(r.get("solar_azimuth_deg") or "").strip()]
    if need:
        work = root.parent.parent if root.parent.name == "colmap" else root.parent
        mans = sorted((work / "processed").glob("mppp_manifest_*.json")) if (work / "processed").is_dir() else []
        if mans:
            try:
                by_stem = {m["filename"]["stem"]: m for m in json.loads(mans[-1].read_text(encoding="utf-8"))["images"]}
                for n in need:
                    m = by_stem.get(Path(n).stem)
                    if m:
                        images[n]["solar_azimuth_deg"] = m.get("solar_azimuth_deg")
                        images[n]["solar_elevation_deg"] = m.get("solar_elevation_deg")
            except (OSError, ValueError, KeyError):
                pass
    for r in images.values():
        try:
            az, el = float(r.get("solar_azimuth_deg")), float(r.get("solar_elevation_deg"))
        except (TypeError, ValueError):
            r["sun_vector"] = None
            r["shadow_tip"] = None
            continue
        r["solar_azimuth_deg"], r["solar_elevation_deg"] = az, el
        a, e = np.radians(az), np.radians(el)
        r["sun_vector"] = np.array([np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)])
        L = 1.0 / np.tan(max(e, np.radians(2.0)))
        r["shadow_tip"] = np.array([-L * np.sin(a), -L * np.cos(a)])


def sun_angle_deg(r1: Dict[str, Any], r2: Dict[str, Any]) -> float:
    """Angle between the sun vectors of two images [deg]; nan if either is unknown."""
    a, b = r1.get("sun_vector"), r2.get("sun_vector")
    if a is None or b is None:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


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
            pid = _native_point_ids(m, iid, r["point2D_idx"].astype(int))
    else:                                                # no residuals.npz: recompute from the model
        from .colmap import observation_residuals
        o = observation_residuals(m)
        res, iid, pid = o["residual_px"], o["image_id"].astype(int), o["point3D_id"].astype(int)
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
            "cross": cross, "range_m": rng, "point": np.asarray(pid if pid is not None else np.full(res.size, -1)),
            "image": np.asarray(iid, int)}


def _native_point_ids(m, iid: np.ndarray, p2d: np.ndarray) -> np.ndarray:
    """
    3-D point id of each residual row, looked up in the native model by keypoint index.
    ``residuals.npz`` holds the keypoint index of the reconstruction, but the native
    model written by ``export_for_error`` keeps only the observed keypoints of each
    image, in their original order.  Every observation has a residual row, so the
    native index is the rank of the keypoint index among that image's rows.  Before
    v0p20.1 this was looked up directly: about a third of the observations fell off
    the end of the shortened lists and the rest were attributed to the wrong points.
    A model with full keypoint lists (``observed_only=False``) is looked up directly.
    """
    pid = np.full(p2d.size, -1, np.int64)
    order = np.argsort(iid, kind="stable")
    bounds = np.flatnonzero(np.diff(iid[order])) + 1
    for rows in np.split(order, bounds):
        if rows.size == 0 or int(iid[rows[0]]) not in m.images:
            continue
        ids = np.asarray(m.images[int(iid[rows[0]])].point3D_ids)
        u, rank = np.unique(p2d[rows], return_inverse=True)
        idx = rank if u.size == ids.size else p2d[rows]       # observed-only list, or the full one
        ok = (idx >= 0) & (idx < ids.size)
        pid[rows[ok]] = ids[idx[ok]]
    return pid


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def eps_table(al: Alignment) -> List[Dict[str, Any]]:
    """
    eps = rms(residual)/sqrt(2) (one image axis, the error model's convention),
    for all observations and per instrument family / instrument, split by
    whether the track stays at one station ("intra") or spans stations ("cross").
    Residuals are post-fit, so these are lower bounds.  ``eps_dof_px`` (v0p22)
    corrects for the 3 coordinates each point absorbs: eps sqrt(2N / (2N - 3P))
    for N observations of P points (the camera parameters, a few per image,
    are neglected); for two-view points the factor is 2.  For a subset (one
    family, one instrument, intra or cross tracks) a point shared with other
    observations absorbs only its share: the 3 P term becomes 3 sum over the
    subset's observations of 1 / (the point's track length) (v0p22.1; before,
    every point touched by the subset counted fully, which overcorrected
    subsets such as one eye of a stereo pair).
    """
    o = al.__dict__.setdefault("_obs", _observations(al))
    rows = []
    have_pts = o["point"].size > 0 and o["point"][0] >= 0
    if have_pts:
        _, inv, cnt = np.unique(o["point"], return_inverse=True, return_counts=True)
        share = 1.0 / cnt[inv]                              # each observation's share of its point's 3 DoF
    groups = [("all", np.ones(o["residual_px"].size, bool))]
    groups += [(f, o["family"] == f) for f in sorted(set(o["family"]))]
    if len(set(o["instrument"])) > 1:
        groups += [(i, o["instrument"] == i) for i in sorted(set(o["instrument"]))]
    for name, g in groups:
        for kind, sel in (("all", g), ("intra", g & ~o["cross"]), ("cross", g & o["cross"])):
            r = o["residual_px"][sel]
            n_pt = int(np.unique(o["point"][sel]).size) if r.size and have_pts else 0
            dof = 2.0 * r.size - 3.0 * (float(share[sel].sum()) if have_pts else 0.0)
            f = np.sqrt(2.0 * r.size / dof) if n_pt and dof > 0 else float("nan")
            rows.append({"alignment": al.label, "group": name, "tracks": kind, "n_obs": int(r.size),
                         "n_points": n_pt, "rms_px": _rms(r), "eps_px": _rms(r) / np.sqrt(2.0),
                         "eps_dof_px": _rms(r) / np.sqrt(2.0) * f, "median_px": float(np.median(r)) if r.size else float("nan")})
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
def eps_by_angle(al: Alignment, edges_deg: Sequence[float] = (0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 90),
                 min_obs: int = 200) -> List[Dict[str, Any]]:
    """
    eps against convergence angle (v0p22.2): for every observation the largest
    ray angle of its point (``theta_max``) and the angle to the nearest other
    ray of the same point (``theta_nn``), same-station and cross-station tracks
    apart.  Rows: kind, angle, bin, n, eps_px, median_px.  A rise with angle
    is precision degrading with convergence; the drop of completeness (the
    gate) is a separate quantity.
    """
    o = al.__dict__.setdefault("_obs", _observations(al))
    m = al.model
    if o["point"].size == 0 or o["point"][0] < 0:
        return []
    iid = o["image"]
    C = {i: im.center for i, im in m.images.items()}
    pid = o["point"]
    th_max = np.full(pid.size, np.nan)
    th_nn = np.full(pid.size, np.nan)
    order = np.argsort(pid, kind="stable")
    for rows in np.split(order, np.flatnonzero(np.diff(pid[order])) + 1):
        X = m.points[int(pid[rows[0]])].xyz
        d = np.array([X - C[int(i)] for i in iid[rows]])
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
        np.fill_diagonal(ang, np.nan)
        if rows.size > 1:
            th_max[rows] = np.nanmax(ang)
            th_nn[rows] = np.nanmin(ang, axis=1)
    e = np.asarray(edges_deg, float)
    out = []
    for kind, sel in (("intra", ~o["cross"]), ("cross", o["cross"])):
        for aname, th in (("theta_max", th_max), ("theta_nn", th_nn)):
            k = np.digitize(th[sel], e) - 1
            rr = o["residual_px"][sel]
            for b in range(e.size - 1):
                s_ = k == b
                if s_.sum() >= min_obs:
                    out.append({"alignment": al.label, "tracks": kind, "angle": aname, "theta_lo": float(e[b]),
                                "theta_hi": float(e[b + 1]), "n": int(s_.sum()),
                                "eps_px": float(np.sqrt(np.mean(rr[s_] ** 2)) / np.sqrt(2.0)),
                                "median_px": float(np.median(rr[s_]))})
    return out


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
    out: Dict[str, List[np.ndarray]] = {k: [] for k in ("point", "theta_deg", "cross", "dlmst_h", "dsun_deg",
                                                        "dshadow", "families", "observed", "weight", "image_a",
                                                        "image_b")}
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
            out["dsun_deg"].append(np.full(n, sun_angle_deg(rows[a], rows[b]), np.float32))
            ta, tb = rows[a].get("shadow_tip"), rows[b].get("shadow_tip")
            out["dshadow"].append(np.full(n, float(np.linalg.norm(ta - tb)) if ta is not None and tb is not None
                                          else np.nan, np.float32))
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
    th = np.asarray(theta_deg, float)
    if cv < 1e-3:                                   # the CV -> 0 limit: A exp(-theta/theta_bar), without overflow
        return A * np.exp(-th / max(theta_bar_deg, 1e-12))
    cv2 = min(cv, 1e3) ** 2
    return A * (1.0 + th / (max(theta_bar_deg, 1e-12) / cv2)) ** (-1.0 / cv2)


ILLUMINATION = {"dlmst": ("dlmst_h", "tau_h", 2.3), "sunangle": ("dsun_deg", "sun0_deg", 40.0),
                "shadow": ("dshadow", "s0", 2.0), "none": (None, None, None)}
ANGLE_FORMS = ("power", "exp", "stretch", "logistic")


Q_LIMIT = 9.0      # |log-parameter| bound in the gate fits: e^9 ~ 8100 (deg, h) - beyond it the form is at its limit


def _angle_form(form: str, th: np.ndarray, q: Sequence[float]) -> np.ndarray:
    """Cross-station survival vs convergence angle, without amplitude; q are log-parameters (bounded by Q_LIMIT)."""
    q = np.clip(np.asarray(q, float), -Q_LIMIT, Q_LIMIT)
    if form == "power":                                 # (1 + theta/theta_c)^-k, theta_c = theta_bar/CV^2, k = 1/CV^2
        return gate_curve(th, 1.0, float(np.exp(q[0])), float(np.exp(q[1])))
    if form == "exp":
        return np.exp(-th / np.exp(q[0]))
    if form == "stretch":
        return np.exp(-(th / np.exp(q[0])) ** np.exp(q[1]))
    if form == "logistic":
        return 1.0 / (1.0 + np.exp((th - np.exp(q[0])) / np.exp(q[1])))
    raise ValueError(form)


_FORM_NPAR = {"power": 2, "exp": 1, "stretch": 2, "logistic": 2}
_FORM_NAMES = {"power": ("theta_bar_deg", "cv"), "exp": ("theta_bar_deg",), "stretch": ("theta_bar_deg", "beta"),
               "logistic": ("theta_0_deg", "width_deg")}


def _binned_cross(pairs, families, theta_max_deg, illumination, weights):
    """Same-station rate and the cross-station trials binned in (theta 0.5 deg, covariate cell)."""
    col, _, _ = ILLUMINATION[illumination]
    pw = pairs["weight"].astype(float) if "weight" in pairs else np.ones(pairs["observed"].size)
    sel_i = _select(pairs, families, cross=False)
    wi = weights[pairs["point"][sel_i]] * pw[sel_i]
    s_intra = np.sum(wi * pairs["observed"][sel_i]) / max(np.sum(wi), 1e-12)
    sel = _select(pairs, families, cross=True) & (pairs["theta_deg"] <= theta_max_deg)
    if col is not None:
        sel &= np.isfinite(pairs[col])
    th = np.round(pairs["theta_deg"][sel] * 2) / 2
    if col is None:
        cov = np.zeros(th.size)
    elif col == "dlmst_h":
        cov = np.round(pairs[col][sel] * 4) / 4
    elif col == "dsun_deg":
        cov = np.round(pairs[col][sel])
    else:
        cov = np.round(pairs[col][sel] * 20) / 20
    w = weights[pairs["point"][sel]] * pw[sel]
    key, inv = np.unique(np.stack([th, cov], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv, weights=w)
    s = np.bincount(inv, weights=w * pairs["observed"][sel])
    return float(s_intra), key[:, 0], key[:, 1], n, s


def fit_gate(pairs: Dict[str, Any], families: Optional[str] = None, tau_h: float = 2.3, fit_tau: bool = True,
             theta_max_deg: float = 30.0, n_boot: int = 0, seed: int = 0, form: str = "power",
             illumination: str = "dlmst") -> Dict[str, Any]:
    """
    Maximum-likelihood fit of the cross-station gate to the pair data:

        P(matched | cross, theta, x) = s_intra * A * g(theta) * exp(-x / x0)

    with g the angle form (``form``: "power" (1 + theta/theta_c)^-k with
    theta_c = theta_bar/CV^2 and k = 1/CV^2, the default; "exp"; "stretch"
    exp(-(theta/theta_bar)^beta); "logistic" 1/(1 + exp((theta-theta_0)/width)))
    and x the illumination covariate (``illumination``: "dlmst" |dLMST| in
    hours with e-fold ``tau_h``, the default; "sunangle" the angle between
    the sun vectors [deg], e-fold ``sun0_deg``; "shadow" the shadow-tip
    distance of a unit post, e-fold ``s0``; "none").  s_intra is the
    same-station match rate (A is relative to it, as in ``ModelConfig.gate_A``).
    Every parameter is fitted from the data (v0p22.2); ``fit_tau=False``
    holds the illumination e-fold at ``tau_h``.  ``n_boot`` > 0 adds a
    bootstrap over the sampled points (16-84 %).
    """
    from scipy.optimize import minimize
    if form not in ANGLE_FORMS or illumination not in ILLUMINATION:
        raise ValueError(f"form {form!r} / illumination {illumination!r}")
    col, xname, x_default = ILLUMINATION[illumination]
    na = _FORM_NPAR[form]
    fit_x = bool(fit_tau) and col is not None
    x_hold = tau_h if illumination == "dlmst" else x_default

    def solve(weights):
        s_intra, th, cov, n, s = _binned_cross(pairs, families, theta_max_deg, illumination, weights)
        if s_intra <= 0 or n.sum() == 0:
            return None

        def nll(q):
            q = np.clip(np.asarray(q, float), -Q_LIMIT, Q_LIMIT)
            p = s_intra * np.exp(q[0]) * _angle_form(form, th, q[1:1 + na])
            if col is not None:
                p = p * np.exp(-cov / (np.exp(q[1 + na]) if fit_x else x_hold))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -float(np.sum(s * np.log(p) + (n - s) * np.log(1 - p)))
        starts = []
        for a0 in (0.4, 1.5):
            for t0 in (2.0, 5.0, 12.0):
                q0 = [np.log(a0), np.log(t0)]
                if na == 2:
                    q0.append(np.log(0.5 if form != "logistic" else 3.0))
                if fit_x:
                    q0.append(np.log(x_hold))
                starts.append(q0)
        best = None
        for q0 in starts:
            r = minimize(nll, q0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 8000})
            if best is None or r.fun < best.fun:
                best = r
        q = np.clip(best.x, -Q_LIMIT, Q_LIMIT)
        at_limit = bool(np.any(np.abs(q) > Q_LIMIT - 0.05))
        out = {"A": float(np.exp(q[0])), "A_absolute": float(np.exp(q[0]) * s_intra), "s_intra": s_intra,
               "nll": float(best.fun), "n_cross_pairs": float(n.sum()), "matched_cross_pairs": float(s.sum()),
               "n_params": 1 + na + int(fit_x), "at_limit": at_limit}
        for name_, val in zip(_FORM_NAMES[form], np.exp(q[1:1 + na])):
            out[name_] = float(val)
        if col is not None:
            out[xname] = float(np.exp(q[1 + na])) if fit_x else float(x_hold)
        return out

    n_pt = int(pairs["n_points"])
    out = solve(np.ones(n_pt))
    if out is None:
        return {"families": families or "all", "form": form, "illumination": illumination,
                "error": "no same-station matches or no cross-station pairs"}
    out["aic"] = 2 * out["n_params"] + 2 * out["nll"]
    out["form"], out["illumination"] = form, illumination
    # constrained: no parameter at a bound of the fit and the angle scale inside the fitted range
    scale = out.get("theta_bar_deg", out.get("theta_0_deg", 0.0))
    out["constrained"] = bool(not out.get("at_limit", False) and scale < theta_max_deg and out.get("cv", 1.0) > 0.02)
    edges = np.arange(0.0, theta_max_deg + 1e-9, 1.0)
    sc = survival_curve(pairs, edges, families, cross=True)
    pred = gate_expected(pairs, out, edges, families)
    ok = (sc["n_possible"] > 0) & np.isfinite(pred) & (pred > 0) & (pred < 1)
    chi2 = np.sum((sc["n_observed"][ok] - sc["n_possible"][ok] * pred[ok]) ** 2 /
                  (sc["n_possible"][ok] * pred[ok] * (1 - pred[ok])))
    out["chi2_dof"] = float(chi2 / max(int(ok.sum()) - out["n_params"], 1))
    out["theta_half_deg"] = _half_angle(sc)
    cr = _select(pairs, families, True)
    out.update(families=families or "all", form=form, illumination=illumination, theta_max_deg=theta_max_deg,
               fit_tau=fit_x, n_intra_pairs=int(np.sum(_select(pairs, families, cross=False))),
               dlmst_h_range=[float(np.nanmin(pairs["dlmst_h"][cr])) if cr.any() else float("nan"),
                              float(np.nanmax(pairs["dlmst_h"][cr])) if cr.any() else float("nan")])
    if "tau_h" not in out:
        out["tau_h"] = float("nan")
    if n_boot:
        rng = np.random.default_rng(seed)
        keys = ["A"] + list(_FORM_NAMES[form]) + ([xname] if col is not None else [])
        boots = []
        for _ in range(int(n_boot)):
            w = np.bincount(rng.integers(0, n_pt, n_pt), minlength=n_pt).astype(float)
            b = solve(w)
            if b:
                boots.append([b[k] for k in keys])
        if boots:
            qq = np.percentile(np.array(boots), [16, 84], axis=0)
            for j, k in enumerate(keys):
                out[f"{k}_16_84"] = [float(qq[0, j]), float(qq[1, j])]
    return out


def compare_gate_forms(pairs: Dict[str, Any], families: Optional[str] = None, theta_max_deg: float = 30.0,
                       forms: Sequence[str] = ANGLE_FORMS, illuminations: Sequence[str] = ("none", "dlmst", "sunangle"),
                       ) -> List[Dict[str, Any]]:
    """Every angle form x illumination covariate fitted to the same binned data; rows sorted by AIC (dAIC from the best)."""
    rows = []
    for f in forms:
        for il in illuminations:
            g = fit_gate(pairs, families, theta_max_deg=theta_max_deg, form=f, illumination=il, fit_tau=True)
            if "error" in g:
                continue
            rows.append({k: g[k] for k in ("form", "illumination", "aic", "nll", "n_params", "chi2_dof", "A") if k in g}
                        | {k: g[k] for k in ("theta_bar_deg", "cv", "beta", "theta_0_deg", "width_deg", "tau_h",
                                             "sun0_deg", "s0") if k in g and np.isfinite(g[k])})
    if rows:
        best = min(r["aic"] for r in rows)
        for r in rows:
            r["dAIC"] = r["aic"] - best
        rows.sort(key=lambda r: r["aic"])
    return rows


def gate_expected(pairs: Dict[str, Any], fit: Dict[str, Any], edges_deg: Sequence[float] = tuple(np.arange(0, 31, 1.0)),
                  families: Optional[str] = None, dl_max_h: Optional[float] = None) -> np.ndarray:
    """
    Cross-station match rate per angle bin predicted by a fitted gate
    (:func:`fit_gate` result), averaged over the actual covariate values of
    the pairs in each bin - comparable with :func:`survival_curve`.
    """
    e = np.asarray(edges_deg, float)
    sel = _select(pairs, families, True, dl_max_h)
    th = pairs["theta_deg"][sel]
    form, il = fit.get("form", "power"), fit.get("illumination", "dlmst")
    q = [np.log(fit[k]) for k in _FORM_NAMES[form]]
    pr = fit["s_intra"] * fit["A"] * _angle_form(form, th, q)
    col, xname, _ = ILLUMINATION[il]
    if col is not None and np.isfinite(fit.get(xname, np.nan)):
        pr = pr * np.exp(-np.nan_to_num(pairs[col][sel], nan=0.0) / max(fit[xname], 1e-9))
    w = pairs["weight"][sel].astype(float) if "weight" in pairs else np.ones(th.size)
    k = np.digitize(th, e) - 1
    ok = (k >= 0) & (k < e.size - 1)
    n = np.bincount(k[ok], weights=w[ok], minlength=e.size - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.bincount(k[ok], weights=(w * pr)[ok], minlength=e.size - 1) / n


def combine_pairs(pair_sets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool :func:`pair_survival` results of several alignments (point indices kept distinct)."""
    keys = ("point", "theta_deg", "cross", "dlmst_h", "dsun_deg", "dshadow", "families", "observed", "weight",
            "image_a", "image_b")
    out: Dict[str, Any] = {k: [] for k in keys}
    off = 0
    for p in pair_sets:
        for k in keys:
            v = p[k] if k in p else (np.full(p["observed"].size, np.nan, np.float32) if k in ("dsun_deg", "dshadow")
                                     else np.ones(p["observed"].size, np.int8))
            out[k].append(v + off if k == "point" else v)
        off += int(p["n_points"])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update(n_points=off, label="+".join(str(p.get("label")) for p in pair_sets),
               conditional=all(p.get("conditional", False) for p in pair_sets))
    return res


# ------------------------------------------------------ decorrelation, graph
RHO_EDGES_DEG = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0)


def decorrelation(al: Alignment, n_points: int = 6000, seed: int = 0, edges_deg: Sequence[float] = RHO_EDGES_DEG,
                  min_track: int = 4, max_track: int = 12, n_boot: int = 200) -> Dict[str, Any]:
    """
    rho(theta) of pair-triangulation residuals (v0p22.2 estimator).  For each
    track of ``min_track``-``max_track`` images, every image pair is
    triangulated on its own (observed rays); residuals are taken from the mean
    of the pair solutions and normalised by their rms; the correlation of two
    residuals is binned by the angle between the two pairs' bisectors.  Only
    pairs that share NO image are used - two pairs sharing an image are
    correlated by that image's observation error whatever the angle
    (``rho_shared_image`` shows this) - and the constraint bias of residuals
    about their own mean, -1/(M-1) for M pair solutions, is removed.
    Standard errors come from a bootstrap over tracks.  ``fit_rho`` fits
    rho_inf + rho_0 exp(-theta^2 / 2 theta_c^2) (and an exponential) to it.
    """
    from .colmap import _triangulate, observed_rays
    m = al.model
    e = np.asarray(edges_deg, float)
    nb = e.size - 1
    rng = np.random.default_rng(seed)
    ids = np.array([p for p, pt in m.points.items() if min_track <= pt.track_length <= max_track])
    if ids.size > n_points:
        ids = rng.choice(ids, n_points, replace=False)
    tracks = []
    for p in ids:
        pt = m.points[int(p)]
        iids, C, u = observed_rays(m, pt)
        n = len(iids)
        if n < min_track:
            continue
        pairs, bis, res = [], [], []
        for a in range(n):
            for b in range(a + 1, n):
                x = _triangulate(C[a], u[a], C[b], u[b])
                if x is None:
                    continue
                bb = u[a] + u[b]
                pairs.append((a, b))
                bis.append(bb / max(np.linalg.norm(bb), 1e-12))
                res.append(x - pt.xyz)
        if len(res) < 3:
            continue
        res = np.array(res) - np.mean(res, axis=0)
        sc = np.sqrt(np.mean(np.sum(res ** 2, axis=1)))
        if not np.isfinite(sc) or sc <= 0:
            continue
        rn = res / sc
        M = len(res)
        null = -1.0 / (M - 1)
        bis = np.array(bis)
        disj: List[Tuple[int, float]] = []
        shared: List[Tuple[int, float]] = []
        for i in range(M):
            for j in range(i + 1, M):
                th = np.degrees(np.arccos(np.clip(float(bis[i] @ bis[j]), -1, 1)))
                k = int(np.searchsorted(e, th) - 1)
                if not (0 <= k < nb):
                    continue
                v = float(rn[i] @ rn[j]) / 3.0 - null
                (shared if set(pairs[i]) & set(pairs[j]) else disj).append((k, v))
        tracks.append((disj, shared))

    def curve(ts, which):
        sums, cnt = np.zeros(nb), np.zeros(nb)
        for t in ts:
            for k, v in t[which]:
                sums[k] += v
                cnt[k] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return sums / cnt, cnt
    rho, n = curve(tracks, 0)
    rho_s, n_s = curve(tracks, 1)
    se = np.full(nb, np.nan)
    if n_boot and tracks:
        boots = [curve([tracks[i] for i in rng.integers(0, len(tracks), len(tracks))], 0)[0] for _ in range(int(n_boot))]
        se = np.nanstd(np.array(boots), axis=0)
    centres = 0.5 * (e[:-1] + e[1:])
    return {"theta_deg": centres, "edges_deg": e, "rho": rho, "se": se, "n": n, "rho_shared_image": rho_s,
            "n_shared": n_s, "n_tracks": len(tracks), "min_track": min_track}


def fit_rho(dc: Dict[str, Any]) -> Dict[str, Any]:
    """Gaussian and exponential fits of rho_inf + rho_0 f(theta/theta_c) to a :func:`decorrelation` curve (weighted)."""
    from scipy.optimize import curve_fit
    c = np.array(dc["theta_deg"], float)
    c[0] = 0.5 * float(dc["edges_deg"][1])
    r, se = np.asarray(dc["rho"], float), np.asarray(dc["se"], float)
    ok = np.isfinite(r) & np.isfinite(se) & (se > 0)
    out: Dict[str, Any] = {}
    forms = {"gaussian": lambda t, r0, tc, ri: ri + r0 * np.exp(-t ** 2 / (2 * tc ** 2)),
             "exponential": lambda t, r0, tc, ri: ri + r0 * np.exp(-t / tc)}
    for name, f in forms.items():
        try:
            p, cov = curve_fit(f, c[ok], r[ok], sigma=se[ok], p0=[0.15, 0.1, 0.01],
                               bounds=([0, 0.01, -0.2], [1, 20, 0.5]), absolute_sigma=True)
            chi2 = float(np.sum(((r[ok] - f(c[ok], *p)) / se[ok]) ** 2))
            out[name] = {"rho_0": float(p[0]), "theta_c_deg": float(p[1]), "rho_inf": float(p[2]),
                         "rho_0_se": float(np.sqrt(cov[0, 0])), "theta_c_se": float(np.sqrt(cov[1, 1])),
                         "rho_inf_se": float(np.sqrt(cov[2, 2])), "chi2_dof": chi2 / max(int(ok.sum()) - 3, 1)}
        except (RuntimeError, ValueError) as ex:
            out[name] = {"error": str(ex)}
    first = int(np.argmax(ok)) if ok.any() else 0
    out["first_bin"] = {"theta_lo": float(dc["edges_deg"][0]), "theta_hi": float(dc["edges_deg"][1]),
                        "rho": float(r[first]) if ok.any() else float("nan"), "se": float(se[first]) if ok.any() else float("nan"),
                        "consistent_with_zero": bool(ok.any() and abs(r[first]) < 3 * se[first])}
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
    A, thb, cv, tau, _ = CASE_PARAMS["measured"]
    out = {"gate_A": A, "theta_bar_deg": thb, "gate_cv": cv, "tau_h": tau,
           "theta_c_deg": c.theta_c_deg, "rho_0": c.rho_0, "rho_inf": c.rho_inf,
           "correlation_kernel": c.correlation_kernel, "eps_intra_px (archive, pooled)": 0.169,
           "reg_sigma_m (telemetry registration)": 0.3}
    out["cases"] = {k: {"A": v[0], "theta_bar_deg": v[1], "cv": v[2], "tau_h": v[3], "note": v[4]}
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
        row.update(gate_A=gate["A"], theta_bar_deg=gate.get("theta_bar_deg"), gate_cv=gate.get("cv"),
                   tau_h=gate.get("tau_h"), gate_form=gate.get("form"), gate_illumination=gate.get("illumination"),
                   gate_families=gate.get("families"), gate_constrained=gate.get("constrained"),
                   gate_chi2_dof=gate.get("chi2_dof"), same_station_rate=gate.get("s_intra"))
    if rho is not None:
        ok = np.isfinite(rho["rho"]) & (rho["n"] > 50)
        row["rho_first_bin"] = float(rho["rho"][ok][0]) if ok.any() else float("nan")
        row["rho_first_bin_se"] = float(rho["se"][ok][0]) if ok.any() and "se" in rho else float("nan")
        fr = fit_rho(rho).get("gaussian", {})
        row["rho_0"] = fr.get("rho_0"); row["rho_theta_c_deg"] = fr.get("theta_c_deg"); row["rho_inf"] = fr.get("rho_inf")
    return row
