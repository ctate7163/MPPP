"""
mppp.error.colmap -- back the error-model parameters out of a COLMAP reconstruction.

This is the bridge from predicted precision to measured precision.  Everything
the error model needs to be calibrated is already sitting in a COLMAP sparse
reconstruction; it just has to be extracted.

    eps          <- reprojection residual RMS, per image
    eps_cross    <- residual RMS on cross-station tracks, vs. intra-station
    theta_max    <- convergence angle at which track survival collapses
    theta_c      <- correlation of pairwise residuals vs. angular separation
    view graph   <- shared-track counts between stations
    validation   <- predicted sigma_n vs. COLMAP's own point covariance

The model is the plain ``ColmapModel`` read by :func:`mppp.colmap.read_colmap`
(TEXT format only, ``cameras.txt`` / ``images.txt`` / ``points3D.txt``; export
with ``colmap model_converter --output_type TXT``); no pycolmap needed.  The
MPPP pipeline writes such a model in native pixels to
``<project>/error_input/native/`` (:func:`mppp.sfm.export.export_for_error`),
and :mod:`mppp.error.alignment` reads it from there.

WHAT A COARSE SPARSE MODEL IS AND IS NOT ENOUGH FOR
---------------------------------------------------
Enough:  eps calibration, the view graph, track-length statistics, convergence
         angle distributions, theta_max, theta_c, and validating predicted
         RELATIVE precision.  All of these live in the sparse tie points and
         their residuals.  You do not need dense MVS for any of it.

Not enough:  absolute accuracy against ground truth (needs external control or
         a reference DTM), surface completeness, or anything about the terrain
         between tie points.  Sparse tracks are biased towards well-textured,
         well-matched terrain -- exactly the easy cases -- so eps measured this
         way is an OPTIMISTIC estimate and should be labelled as such.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from ..colmap import (ColmapCamera, ColmapImage, ColmapModel, ColmapPoint, project_camera,
                      read_colmap, unproject_camera)

__all__ = ["ColmapModel", "ColmapCamera", "ColmapImage", "ColmapPoint", "read_colmap",
           "measured_view_graph", "calibrate_eps", "calibrate_eps_from_residuals",
           "measure_theta_c", "match_survival", "observation_residuals", "observed_rays"]


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def calibrate_eps_from_residuals(model: ColmapModel) -> Dict[str, float]:
    """
    eps_intra / eps_cross from TRUE per-observation reprojection residuals.

    Preferred over calibrate_eps(), which uses COLMAP's track-level scalar.

    An intra-station track needs a point seen by two or more frames of the SAME
    station, so it requires overlap between adjacent mosaic frames.  If your
    mosaic has little frame-to-frame overlap you will get few intra tracks and
    eps_intra will be poorly determined -- check n_intra_obs before trusting it.
    """
    r = observation_residuals(model)
    if r["residual_px"].size == 0:
        return {"eps_intra_px": float("nan"), "eps_cross_px": float("nan")}
    a = r["residual_px"][~r["is_cross"]]
    b = r["residual_px"][r["is_cross"]]

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")
    out = {"eps_intra_px": rms(a) / np.sqrt(2.0),
           "eps_cross_px": rms(b) / np.sqrt(2.0),
           "n_intra_obs": int(a.size), "n_cross_obs": int(b.size),
           "resid_rms_intra_px": rms(a), "resid_rms_cross_px": rms(b)}
    out["eps_ratio"] = (out["eps_cross_px"] / out["eps_intra_px"]
                        if out["eps_intra_px"] > 0 else float("nan"))
    return out


def calibrate_eps(model: ColmapModel,
                  min_track: int = 2) -> Dict[str, float]:
    """
    Estimate eps_intra and eps_cross from COLMAP reprojection errors.

    A track observed only within one station calibrates eps_intra.  A track
    spanning two or more stations calibrates eps_cross.  The RATIO is what the
    error model needs, and it is far more robust than either absolute value.

    NOTE: COLMAP's per-point `error` is the mean reprojection residual of the
    track after bundle adjustment, so it is a POST-FIT residual and therefore
    an underestimate of true measurement precision by roughly sqrt(r/n), where
    r is redundancy.  It is also biased optimistic because badly matched points
    were filtered out.  Treat the result as a lower bound and label it as such.
    """
    intra, cross, allpt = [], [], []
    tl_intra, tl_cross = [], []
    for p in model.points.values():
        if p.track_length < min_track:
            continue
        stations = {model.images[i].station for i in p.image_ids
                    if i in model.images}
        allpt.append(p.error)
        if len(stations) <= 1:
            intra.append(p.error); tl_intra.append(p.track_length)
        else:
            cross.append(p.error); tl_cross.append(p.track_length)

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")

    out = {
        "eps_intra_px": rms(intra),
        "eps_cross_px": rms(cross),
        "eps_all_px": rms(allpt),
        "n_intra_tracks": len(intra),
        "n_cross_tracks": len(cross),
        "cross_track_fraction": (len(cross) / max(len(allpt), 1)),
        "mean_track_len_intra": float(np.mean(tl_intra)) if tl_intra else float("nan"),
        "mean_track_len_cross": float(np.mean(tl_cross)) if tl_cross else float("nan"),
    }
    out["eps_ratio"] = (out["eps_cross_px"] / out["eps_intra_px"]
                        if out["eps_intra_px"] > 0 else float("nan"))
    return out


def _project(model: ColmapModel, image_id: int, X: np.ndarray):
    """Project a world point into an image. Returns (u, v) or None if behind."""
    im = model.images[image_id]
    cam = model.cameras[im.camera_id]
    x = im.R @ X + im.tvec
    if x[2] <= 1e-6:
        return None
    return project_camera(cam.model, cam.params, x)



def observation_residuals(model: ColmapModel) -> Dict[str, np.ndarray]:
    """
    TRUE per-observation reprojection residuals.

    COLMAP's `point.error` is a single scalar per TRACK, so any statistic binned
    against a per-observation quantity (like convergence angle) comes out flat.
    Recomputing the residual for every observation from the poses, camera model
    and keypoints gives a real per-observation quantity.

    Returns arrays over all observations: residual magnitude [px], the image and
    station of the observation, range, and whether the parent track spans more
    than one station.
    """
    res, img, stn, rng_, cross, tlen, pids = [], [], [], [], [], [], []
    for pt in model.points.values():
        sts = {model.images[i].station for i in pt.image_ids if i in model.images}
        is_cross = len(sts) > 1
        for i in pt.image_ids:
            im = model.images.get(int(i))
            if im is None:
                continue
            k = np.where(im.point3D_ids == pt.point3D_id)[0]
            if k.size == 0:
                continue
            uv = _project(model, int(i), pt.xyz)
            if uv is None:
                continue
            res.append(float(np.linalg.norm(im.xys[k[0]] - uv)))
            img.append(int(i)); stn.append(im.station)
            rng_.append(float(np.linalg.norm(pt.xyz - im.center)))
            cross.append(is_cross); tlen.append(pt.track_length); pids.append(int(pt.point3D_id))
    return {"residual_px": np.array(res), "image_id": np.array(img), "point3D_id": np.array(pids, np.int64),
            "station": np.array(stn), "range_m": np.array(rng_),
            "is_cross": np.array(cross, dtype=bool),
            "track_length": np.array(tlen)}


def match_survival(model: ColmapModel, n_bins: int = 15,
                   theta_max_deg: float = 90.0,
                   margin_px: float = 0.0) -> Dict[str, np.ndarray]:
    """
    MEASURED theta_max: the fraction of GEOMETRICALLY POSSIBLE image pairs that
    were actually matched, as a function of convergence angle.

    For every 3D point, an image pair is "possible" if the point projects inside
    both frames (and in front of both cameras), and "observed" if the track
    actually contains both.  The ratio is a matching-success curve; the angle at
    which it falls to half is theta_max for whatever matcher built this model.

    Without the geometric denominator you are just measuring how much of the
    scene happens to be at small convergence angle, which tells you nothing
    about the matcher.  This is the single most valuable number to extract,
    because theta_max is otherwise a literature guess.

    Caveat: the denominator only counts points that were reconstructed at all,
    so terrain where matching failed COMPLETELY is invisible.  The curve is
    therefore optimistic; treat it as an upper bound on matcher tolerance.
    """
    edges = np.linspace(0.0, np.radians(theta_max_deg), n_bins + 1)
    obs = np.zeros(n_bins)
    poss = np.zeros(n_bins)
    ids = list(model.images)

    for pt in model.points.values():
        seen = set(int(i) for i in pt.image_ids)
        cand = []
        for i in ids:
            im = model.images[i]
            cam = model.cameras[im.camera_id]
            uv = _project(model, i, pt.xyz)
            if uv is None:
                continue
            if not (margin_px <= uv[0] < cam.width - margin_px
                    and margin_px <= uv[1] < cam.height - margin_px):
                continue
            u = pt.xyz - im.center
            cand.append((i, u / max(np.linalg.norm(u), 1e-12)))
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                th = np.arccos(np.clip(float(cand[a][1] @ cand[b][1]), -1.0, 1.0))
                k = int(np.searchsorted(edges, th) - 1)
                if not (0 <= k < n_bins):
                    continue
                poss[k] += 1
                if cand[a][0] in seen and cand[b][0] in seen:
                    obs[k] += 1

    centres = np.degrees(0.5 * (edges[:-1] + edges[1:]))
    with np.errstate(invalid="ignore", divide="ignore"):
        surv = np.where(poss > 0, obs / np.maximum(poss, 1), np.nan)
    # half-survival crossing, linearly interpolated
    theta_half = np.nan
    ok = np.isfinite(surv) & (poss > 20)
    if ok.sum() > 2:
        c, v = centres[ok], surv[ok]
        below = np.where(v < 0.5 * v[0])[0]
        if below.size:
            j = below[0]
            if j > 0:
                t = (0.5 * v[0] - v[j - 1]) / (v[j] - v[j - 1])
                theta_half = float(c[j - 1] + t * (c[j] - c[j - 1]))
            else:
                theta_half = float(c[0])
    return {"theta_deg": centres, "n_observed": obs, "n_possible": poss,
            "survival": surv, "theta_half_deg": theta_half}


def observed_rays(model: ColmapModel, pt: "ColmapPoint") -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    (image ids, camera centres (N,3), unit ray directions (N,3)) of a track,
    from the OBSERVED keypoints (undistorted through the camera model), not
    from the fitted point.  Works with complete keypoint lists (MPPP >= 0.14.5,
    COLMAP) and with lists that hold only the observed keypoints (earlier MPPP
    exports, whose track indices do not match them): the keypoint is taken at
    the track index if it belongs to this point, else found by point id.
    """
    ids, C, U = [], [], []
    for i, k in zip(pt.image_ids, pt.point2D_idxs):
        im = model.images.get(int(i))
        if im is None:
            continue
        k = int(k)
        if not (0 <= k < im.point3D_ids.size and int(im.point3D_ids[k]) == pt.point3D_id):
            w = np.nonzero(im.point3D_ids == pt.point3D_id)[0]
            if not w.size:
                continue
            k = int(w[0])
        cam = model.cameras[im.camera_id]
        xy = unproject_camera(cam.model, cam.params, im.xys[k])
        d = im.R.T @ np.array([xy[0], xy[1], 1.0])
        ids.append(int(i))
        C.append(im.center)
        U.append(d / np.linalg.norm(d))
    return ids, np.array(C).reshape(-1, 3), np.array(U).reshape(-1, 3)


def measure_theta_c(model: ColmapModel, n_bins: int = 14,
                    theta_max_deg: float = 70.0,
                    min_track: int = 3) -> Dict[str, np.ndarray]:
    """
    Empirical decorrelation angle theta_c.

    For each track observed in >= 3 images, triangulate from each image PAIR
    independently, then correlate the resulting position residuals (relative to
    the full-track solution) against the angular separation between the two
    pairs' bisectors.  Fit

        rho(theta) = rho_inf + (1 - rho_inf) exp(-theta^2 / 2 theta_c^2)

    Returns the binned correlation curve; fit it yourself so you can see the
    scatter rather than trusting a single number.

    the pair triangulations use the OBSERVED keypoint rays
    (:func:`observed_rays`).  Before, the rays were the directions from each
    camera to the fitted point itself, so every pair triangulated exactly to
    that point, the residuals were rounding noise and rho was meaningless.

    This measurement does not exist in the literature for Mars surface imagery.
    It is self-contained, needs only a sparse model, and is the main thing
    standing between the correlation model and being defensible.
    """
    edges = np.linspace(0.0, np.radians(theta_max_deg), n_bins + 1)
    prod = [[] for _ in range(n_bins)]
    var = []

    for p in model.points.values():
        if p.track_length < min_track:
            continue
        ids, C, u = observed_rays(model, p)
        if len(ids) < min_track:
            continue

        # per-pair triangulation residual relative to the full-track point
        bis, res = [], []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                x = _triangulate(C[a], u[a], C[b], u[b])
                if x is None:
                    continue
                m = u[a] + u[b]
                bis.append(m / max(np.linalg.norm(m), 1e-12))
                res.append(x - p.xyz)
        if len(res) < 2:
            continue
        res = np.array(res)
        bis = np.array(bis)
        s = np.sqrt(np.mean(np.sum(res ** 2, axis=1)))
        if not np.isfinite(s) or s <= 0:
            continue
        rn = res / s
        var.append(s)
        for a in range(len(res)):
            for b in range(a + 1, len(res)):
                th = np.arccos(np.clip(float(bis[a] @ bis[b]), -1.0, 1.0))
                k = int(np.searchsorted(edges, th) - 1)
                if 0 <= k < n_bins:
                    prod[k].append(float(rn[a] @ rn[b]) / 3.0)

    centres = np.degrees(0.5 * (edges[:-1] + edges[1:]))
    rho = np.array([np.mean(v) if v else np.nan for v in prod])
    n = np.array([len(v) for v in prod])
    return {"theta_deg": centres, "rho": rho, "n": n,
            "residual_scale_m": float(np.median(var)) if var else float("nan")}


def _triangulate(c1, u1, c2, u2):
    """Midpoint of the common perpendicular between two rays."""
    w0 = c1 - c2
    a, b, c = u1 @ u1, u1 @ u2, u2 @ u2
    d, e = u1 @ w0, u2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:
        return None
    s = (b * e - c * d) / den
    t = (a * e - b * d) / den
    return 0.5 * ((c1 + s * u1) + (c2 + t * u2))


def measured_view_graph(model: ColmapModel) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Observed station-station view graph from shared tracks.

    Returns (W, names, mean_theta_deg) where W[i,j] is the number of 3D points
    observed by both stations.  Feed W into mppp.error.viewgraph.ViewGraph to reuse
    the spectral machinery, and compare against the PREDICTED graph -- the
    discrepancy is the calibration signal.
    """
    names = model.station_names
    idx = {n: k for k, n in enumerate(names)}
    n = len(names)
    W = np.zeros((n, n))
    th_sum = np.zeros((n, n))
    th_cnt = np.zeros((n, n))

    for p in model.points.values():
        by_st: Dict[str, List[int]] = defaultdict(list)
        for i in p.image_ids:
            im = model.images.get(int(i))
            if im is not None and im.station in idx:
                by_st[im.station].append(int(i))
        sts = list(by_st)
        for a in range(len(sts)):
            for b in range(a + 1, len(sts)):
                ia, ib = idx[sts[a]], idx[sts[b]]
                W[ia, ib] += 1; W[ib, ia] += 1
                ca = model.images[by_st[sts[a]][0]].center
                cb = model.images[by_st[sts[b]][0]].center
                ua = p.xyz - ca; ua /= max(np.linalg.norm(ua), 1e-12)
                ub = p.xyz - cb; ub /= max(np.linalg.norm(ub), 1e-12)
                t = np.degrees(np.arccos(np.clip(float(ua @ ub), -1, 1)))
                th_sum[ia, ib] += t; th_sum[ib, ia] += t
                th_cnt[ia, ib] += 1; th_cnt[ib, ia] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_th = np.where(th_cnt > 0, th_sum / np.maximum(th_cnt, 1), np.nan)
    return W, names, mean_th
