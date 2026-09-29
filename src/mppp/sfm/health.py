"""
Alignment health (v0p13): is the COLMAP alignment good enough to analyse?

Run it after :func:`mppp.sfm.reconstruct` (and before any error analysis on the
result).  Each check has a value, provisional warn/fail thresholds and a
status; the report's ``verdict`` is the worst status.

Checks
------
registration      images registered (frames excluded as outliers counted separately); frames held at their prior because they have < 30 observations
tie points        points, observations, track-length population (incl. two-view points),
                  observations per image (5th percentile), cross-station fraction, tied station blocks
reprojection      native-pixel residuals: median, RMS, 95th percentile; per camera and resolution;
                  left/right balance; images whose median residual is far above the rest;
                  residual growth towards the image edge (distortion-model misfit);
                  tie-point coverage in rings of image radius, and the corners vs the centre (v0p20)
camera model      per camera: change in f, principal point and k1-k4, p1, p2 from the calibration, and the
                  resulting image displacement of the same ray (max / RMS over the frame)
stereo rig        rotation of the right camera relative to the CAHV rig; baseline (held);
                  spread of the CAHV pairs the rig was built from
poses             refined minus prior camera centre and attitude: per station median, and
                  the spread within a station (frames of one station should move together);
                  similarity (scale, rotation) between refined and prior station positions

The thresholds (``DEFAULT_THRESHOLDS``) are provisional values set a priori,
not validated limits; pass your own to tighten or relax them.  On the Belva
Navcam test (sols 748-815, prior-pair matching) they flag 55 frames held at
their prior (11.5 %), a 0.13 deg rotation of the refined right camera relative
to the CAHV rig, residuals growing towards the image edge, and 6 of 13
stations outside the main tied block.

    report = assess_alignment(project, rec)
    print(health_table(report))
    write_health(report, project.root / "health")        # health.json, health.md, health.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .export import _umeyama, pose_residual_table, station_components
from .project import SfmProject, station_labels
from .reconstruction import native_residuals, track_statistics

PathLike = Union[str, Path]

# name: (warn, fail, direction) - "above": bad when value > threshold; "below": bad when value < threshold
DEFAULT_THRESHOLDS: Dict[str, tuple] = {
    "registered_fraction": (0.98, 0.90, "below"),
    "excluded_fraction": (0.05, 0.20, "above"),
    "unconstrained_fraction": (0.25, 0.60, "above"),
    "held_frame_fraction": (0.02, 0.10, "above"),
    "obs_per_image_p05": (200, 50, "below"),
    "cross_station_fraction": (0.02, 0.005, "below"),
    "tied_block_fraction": (0.8, 0.5, "below"),
    "residual_median_px": (0.5, 1.0, "above"),
    "residual_p95_px": (2.0, 4.0, "above"),
    "residual_eye_ratio": (1.3, 2.0, "above"),
    "outlier_image_fraction": (0.04, 0.20, "above"),       # doubled in v0p14.2 (user request)
    "edge_to_centre_residual": (1.5, 2.5, "above"),
    "focal_change_pct": (0.5, 2.0, "above"),
    "principal_point_change_px": (20.0, 60.0, "above"),
    "ray_displacement_max_px": (20.0, 60.0, "above"),      # doubled in v0p14.2
    "rig_rotation_change_deg": (0.06, 0.2, "above"),       # doubled in v0p14.2
    "rig_cahv_spread_deg": (0.02, 0.1, "above"),
    "station_shift_median_m": (2.0, 6.0, "above"),          # doubled in v0p20 (user request)
    # v0p30: attitude_change_p95_deg (block rotation + per-frame scatter) is reported without a verdict; the
    # two parts are judged separately: block_rotation_deg and attitude_residual_p95_deg (label pointing
    # knowledge is ~0.1-0.35 deg per frame; the p95 over nine sites was 0.23-0.69 deg)
    "attitude_residual_p95_deg": (0.75, 2.0, "above"),
    "within_station_shift_spread_m": (0.1, 0.4, "above"),   # doubled in v0p20 (user request)
    "corner_triangulated_ratio": (0.5, 0.25, "below"),      # v0p20
    # v0p30: the scale error of the station layout is judged as the displacement it makes at the stations
    # (|s - 1| x rms station distance from the centroid x sqrt(n)): 2 % over an 8 m block is 0.2 m, within the
    # waypoint accuracy; prior_scale_error_pct is reported without a verdict
    "prior_scale_error_m": (0.5, 1.5, "above"),
    "block_rotation_deg": (0.3, 1.0, "above"),              # v0p30: the whole block turned away from the ENU frame
}
_RANK = {"pass": 0, "info": 0, "warn": 1, "fail": 2}


def _status(name: str, value: Optional[float], thr: Dict[str, tuple]) -> str:
    if value is None or not np.isfinite(value) or name not in thr:
        return "info"
    warn, fail, direction = thr[name]
    if direction == "above":
        return "fail" if value > fail else "warn" if value > warn else "pass"
    return "fail" if value < fail else "warn" if value < warn else "pass"


def _q(a: np.ndarray, q: float) -> Optional[float]:
    return float(np.percentile(a, q)) if a.size else None


def _camera_change(cam0: Dict[str, Any], cam1) -> Dict[str, Any]:
    """Intrinsic change, and the image displacement of the same ray over the whole frame."""
    import pycolmap
    p0, p1 = np.asarray(cam0["params"], float), np.asarray(cam1.params, float)
    w, h = int(cam0["width"]), int(cam0["height"])
    c0 = pycolmap.Camera(model=cam0["model"], width=w, height=h, params=p0)
    one = cam0["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL")
    f0, f1 = (p0[0], p1[0]) if one else (p0[0], p1[0])
    cx0, cy0 = (p0[1], p0[2]) if one else (p0[2], p0[3])
    cx1, cy1 = (p1[1], p1[2]) if one else (p1[2], p1[3])
    # v0p20: rays through a pixel grid over the whole frame (corners included), undistorted with the initial
    # camera; pixels it cannot invert (the Navcam polynomial beyond ~0.88 of the corner radius) are left out
    gx, gy = np.meshgrid(np.linspace(0.5, w - 0.5, 33), np.linspace(0.5, h - 0.5, 25))
    xy0 = np.asarray(c0.cam_from_img(np.c_[gx.ravel(), gy.ravel()]), float)
    rays = np.c_[xy0, np.ones(len(xy0))]
    fin = np.all(np.isfinite(rays), axis=1)
    rays = np.where(fin[:, None], rays, [[0.0, 0.0, 1.0]])
    u0, u1 = np.asarray(c0.img_from_cam(rays), float), np.asarray(cam1.img_from_cam(rays), float)
    u0[~fin] = np.nan
    ok = np.all(np.isfinite(u0), axis=1) & np.all(np.isfinite(u1), axis=1) & \
        (u0[:, 0] >= 0) & (u0[:, 0] <= w) & (u0[:, 1] >= 0) & (u0[:, 1] <= h)
    d = np.linalg.norm(u1[ok] - u0[ok], axis=1)
    out = {"model": cam0["model"], "f_initial": float(f0), "f_refined": float(f1),
           "focal_change_pct": float(100 * abs(f1 - f0) / f0),
           "principal_point_change_px": float(np.hypot(cx1 - cx0, cy1 - cy0)),
           "ray_displacement_max_px": float(d.max()) if d.size else None,
           "ray_displacement_rms_px": float(np.sqrt(np.mean(d ** 2))) if d.size else None,
           "units": "full-resolution pixels"}
    if cam0["model"] in ("OPENCV", "FULL_OPENCV"):
        for k, i in (("k1", 4), ("k2", 5), ("p1", 6), ("p2", 7), ("k3", 8), ("k4", 9)):
            if i < len(p0):
                out[f"{k}_initial"], out[f"{k}_refined"] = float(p0[i]), float(p1[i])
    return out


def _rotation_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


# ------------------------------------------------ coverage across the image
RADIUS_EDGES = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)


def coverage_by_radius(rec, project: SfmProject, edges: Sequence[float] = RADIUS_EDGES) -> Dict[str, Any]:
    """
    Per camera (``NL``, ``NR``, ``ZL034`` ...): keypoints, the fraction of them
    that became tie points, and the median residual, in rings of radius from
    the principal point (1 = the farthest frame corner).  A distortion model
    that cannot be inverted towards the corners shows up as keypoints there
    that are never triangulated (v0p20: the three-term Navcam polynomial lost
    everything beyond ~0.88).
    """
    by_iid = {r["image_id"]: r for r in project.images if "image_id" in r}
    e = np.asarray(edges, float)
    acc: Dict[str, Dict[str, np.ndarray]] = {}
    res_acc: Dict[str, List[List[float]]] = {}
    for iid in rec.reg_image_ids():
        im = rec.images[int(iid)]
        meta = by_iid.get(int(iid), {})
        grp = meta.get("camera_group") or meta.get("instrument", "?")
        cam = rec.cameras[im.camera_id]
        p = np.asarray(cam.params, float)
        one = cam.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL")
        cx, cy = (p[1], p[2]) if one else (p[2], p[3])
        rmax = np.hypot(max(cx, cam.width - cx), max(cy, cam.height - cy))
        pts = im.points2D
        if not len(pts):
            continue
        xy = np.array([q.xy for q in pts], float)
        tri = np.array([q.has_point3D() for q in pts], bool)
        k = np.clip(np.digitize(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) / rmax, e) - 1, 0, e.size - 2)
        a = acc.setdefault(grp, {"keypoints": np.zeros(e.size - 1), "triangulated": np.zeros(e.size - 1)})
        a["keypoints"] += np.bincount(k, minlength=e.size - 1)
        a["triangulated"] += np.bincount(k, weights=tri.astype(float), minlength=e.size - 1)
        rl = res_acc.setdefault(grp, [[] for _ in range(e.size - 1)])
        j = np.nonzero(tri)[0]
        if j.size:
            X = np.array([rec.points3D[pts[int(q)].point3D_id].xyz for q in j], float)
            T = im.cam_from_world()
            xc = X @ np.asarray(T.rotation.matrix()).T + np.asarray(T.translation)
            uv = np.asarray(cam.img_from_cam(xc), float)
            d = np.linalg.norm(uv - xy[j], axis=1) * float(meta.get("downsample_scale", 1.0))   # native px
            okd = np.isfinite(d)
            for b in range(e.size - 1):
                sel = okd & (k[j] == b)
                if sel.any():
                    rl[b].extend(d[sel].tolist())
    out: Dict[str, Any] = {"edges": list(map(float, e)), "cameras": {}}
    for grp, a in sorted(acc.items()):
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = a["triangulated"] / a["keypoints"]
        out["cameras"][grp] = {"keypoints": a["keypoints"].astype(int).tolist(),
                               "triangulated_fraction": [None if not np.isfinite(v) else float(v) for v in frac],
                               "residual_median_native_px": [float(np.median(v)) if v else None for v in res_acc[grp]]}
    return out


def _corner_ratio(cov: Dict[str, Any], inner: float = 0.6, outer: float = 0.85, min_keypoints: int = 200):
    """(worst camera, ratio): triangulated fraction beyond ``outer`` over that inside ``inner``."""
    e = np.asarray(cov["edges"])
    lo, hi = e[:-1], e[1:]
    worst = None
    for grp, c in cov["cameras"].items():
        kp, tr = np.asarray(c["keypoints"], float), np.asarray([v or 0.0 for v in c["triangulated_fraction"]])
        i_in, i_out = hi <= inner + 1e-9, lo >= outer - 1e-9
        if kp[i_out].sum() < min_keypoints or kp[i_in].sum() == 0:
            continue
        f_in = float(np.sum(tr[i_in] * kp[i_in]) / kp[i_in].sum())
        f_out = float(np.sum(tr[i_out] * kp[i_out]) / kp[i_out].sum())
        ratio = f_out / f_in if f_in > 0 else 0.0
        if worst is None or ratio < worst[1]:
            worst = (grp, ratio, f_in, f_out)
    return worst


# ------------------------------------------------------- weak-image diagnosis
WEAK_CAUSES = {
    "few_keypoints": "few SIFT keypoints: the image is mostly masked (rover, sky) or featureless",
    "unmatched": "keypoints but no verified matches with any other image",
    "stereo_only_far": "matched only with its stereo partner; those points are too far for the minimum "
                       "triangulation angle (distant scene)",
    "same_station_only": "no stereo partner, matched only with images of its own station: a mast pan has "
                         "almost no baseline, so these matches cannot be triangulated",
    "lost_in_triangulation": "verified matches exist, but few survive triangulation and the residual filter",
    "no_database": "database not available for the diagnosis",
}
WEAK_ADVICE = {
    "few_keypoints": "nothing to recover: drop the image (delete it from images_png8, KEEP_ONLY_REMAINING=True) "
                     "or accept it held at its prior",
    "unmatched": "with MATCH_MODE='prior_pairs' use 'exhaustive'; otherwise the image overlaps nothing - drop it",
    "stereo_only_far": "lower reconstruct(min_tri_angle_deg=...): the default 0.25 keeps stereo-only points to "
                       "~97 m, 0.1 to ~240 m",
    "same_station_only": "needs matches to another station (MATCH_MODE='exhaustive' if 'prior_pairs' was used) or "
                         "its right-eye partner in the selection; otherwise it stays at its prior pose (harmless for "
                         "the others) - or drop it",
    "lost_in_triangulation": "raise the first-round triangulation threshold, e.g. schedule=((32, 10, 8), (12, 2, 4), "
                             "(8, 2, 2)), or add a round",
    "no_database": "",
}


def classify_weak_image(keypoints: int, inliers_other: Optional[int], inliers_partner: Optional[int],
                        observations: int, min_keypoints: int = 300, min_inliers: int = 30,
                        inliers_other_station: Optional[int] = None) -> str:
    """Most likely reason an image ended up with few observations (see ``WEAK_CAUSES``).
    ``inliers_other`` counts matches with every image except the stereo partner;
    ``inliers_other_station`` those with images of other stations (None = unknown)."""
    if keypoints < min_keypoints:
        return "few_keypoints"
    if inliers_other is None:
        return "no_database"
    if inliers_other + (inliers_partner or 0) < min_inliers:
        return "unmatched"
    if inliers_other < min_inliers and (inliers_partner or 0) >= min_inliers:
        return "stereo_only_far"
    if (inliers_partner or 0) < min_inliers and inliers_other_station is not None \
            and inliers_other_station < min_inliers:
        return "same_station_only"
    return "lost_in_triangulation"


def diagnose_weak_images(project: SfmProject, rec, min_observations: int = 30) -> List[Dict[str, Any]]:
    """
    For every registered image with fewer than ``min_observations`` tie-point
    observations: keypoints, verified inlier matches with its stereo partner
    and with all other images (from ``database.db``), observations, and the
    likely cause (``classify_weak_image``) with advice.
    """
    by_iid = {r["image_id"]: r for r in project.images if "image_id" in r}
    labels = station_labels(project.images)
    n_obs: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            n_obs[el.image_id] = n_obs.get(el.image_id, 0) + 1
    weak = [int(i) for i in rec.reg_image_ids() if n_obs.get(int(i), 0) < min_observations]
    if not weak:
        return []
    frame_of = {iid: rec.images[iid].frame_id for iid in weak}
    partner: Dict[int, Optional[int]] = {}
    for iid in weak:
        fr = rec.frames[frame_of[iid]]
        others = [d.id for d in fr.data_ids if d.id != iid]
        partner[iid] = int(others[0]) if others else None
    inl_other = {iid: 0 for iid in weak}
    inl_cross = {iid: 0 for iid in weak}
    inl_partner = {iid: 0 for iid in weak}
    n_pairs = {iid: 0 for iid in weak}
    have_db = project.database.is_file()
    if have_db:
        from .reconstruction import _verified_matches
        wset = set(weak)
        for i1, i2, m in _verified_matches(project):
            for a, b in ((i1, i2), (i2, i1)):
                if a in wset:
                    if b == partner[a]:
                        inl_partner[a] += len(m)
                    else:
                        inl_other[a] += len(m)
                        n_pairs[a] += 1
                        if by_iid.get(b, {}).get("station") != by_iid.get(a, {}).get("station"):
                            inl_cross[a] += len(m)
    out = []
    for iid in weak:
        kp = int(rec.images[iid].num_points2D())
        cause = classify_weak_image(kp, inl_other[iid] if have_db else None, inl_partner[iid] if have_db else None,
                                    n_obs.get(iid, 0), inliers_other_station=inl_cross[iid] if have_db else None)
        st_id = by_iid.get(iid, {}).get("station")
        out.append({"name": by_iid.get(iid, {}).get("name", str(iid)), "station": st_id,
                    "station_label": labels.get(st_id, st_id),
                    "keypoints": kp, "observations": n_obs.get(iid, 0),
                    "inliers_with_stereo_partner": inl_partner[iid] if have_db else None,
                    "inliers_with_other_images": inl_other[iid] if have_db else None,
                    "inliers_with_other_stations": inl_cross[iid] if have_db else None,
                    "matched_images": n_pairs[iid] if have_db else None, "cause": cause})
    return sorted(out, key=lambda d: (d["cause"], d["name"]))


def assess_alignment(project: SfmProject, rec, thresholds: Optional[Dict[str, tuple]] = None,
                     min_frame_observations: int = 30, outlier_factor: float = 3.0) -> Dict[str, Any]:
    """Health report of a refined reconstruction (see module docstring)."""
    thr = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    checks: List[Dict[str, Any]] = []

    def check(section: str, name: str, value, note: str = "", unit: str = "") -> None:
        v = None if value is None else float(value)
        checks.append({"section": section, "check": name, "value": v, "unit": unit,
                       "status": _status(name, v, thr), "warn": thr.get(name, (None,))[0],
                       "fail": thr.get(name, (None, None))[1], "note": note})

    by_iid = {r["image_id"]: r for r in project.images if "image_id" in r}
    reg = set(int(i) for i in rec.reg_image_ids())
    rows = pose_residual_table(rec, project)
    n_obs = {r["name"]: r["observations"] for r in rows}

    # ---- registration
    # frames excluded as outliers (v0p22) are a decision, not a registration failure: counted separately
    ex_recs = (project.settings.get("reconstruction", {}) or {}).get("excluded", []) or []
    unc = {n for e in ex_recs if e.get("kind", "outlier" if e.get("observations", 1) else "unconstrained") == "unconstrained"
           for n in e.get("images", [])}
    excl = {n for e in ex_recs for n in e.get("images", [])} - unc
    n_cand = len(project.images) - len(excl) - len(unc)
    check("registration", "registered_fraction", len(reg) / max(1, n_cand),
          f"{len(reg)} of {n_cand} images" + (f" ({len(excl)} excluded as outliers, {len(unc)} without tie points)"
                                              if excl or unc else ""))
    check("registration", "excluded_fraction", len(excl) / max(1, len(project.images)),
          f"{len(excl)} images in frames that did not align with the majority (exclude_outliers)")
    check("registration", "unconstrained_fraction", len(unc) / max(1, len(project.images)),
          f"{len(unc)} images with no tie points at all (defocused, sky, calibration target...): left out")
    held = [r["name"] for r in rows if r["observations"] < min_frame_observations]
    check("registration", "held_frame_fraction", len(held) / max(1, len(rows)),
          f"{len(held)} images with < {min_frame_observations} observations stay at their prior pose")

    # ---- tie points
    ts = track_statistics(rec, project)
    lens = np.array([pt.track.length() for pt in rec.points3D.values()], int)
    hist = {"2": int((lens == 2).sum()), "3": int((lens == 3).sum()), "4": int((lens == 4).sum()),
            "5-9": int(((lens >= 5) & (lens <= 9)).sum()), "10+": int((lens >= 10).sum())}
    check("tie points", "points", ts["points"], f"{ts['observations']} observations")
    check("tie points", "two_view_fraction", ts["two_view_fraction"], f"track lengths {hist}")
    check("tie points", "mean_track_length", ts["mean_track_length"])
    per_image = np.array([n_obs.get(by_iid[i]["name"], 0) for i in reg if i in by_iid], float)
    check("tie points", "obs_per_image_p05", _q(per_image, 5),
          f"median {_q(per_image, 50):.0f}" if per_image.size else "")
    check("tie points", "cross_station_fraction", ts["cross_station_fraction"],
          f"{ts['cross_station_points']} points seen from more than one station")
    comps = station_components(rec, project)
    n_st = sum(len(c) for c in comps)
    check("tie points", "tied_block_fraction", (len(comps[0]) / n_st) if comps else None,
          f"largest tied block {len(comps[0]) if comps else 0} of {n_st} stations; "
          f"{sum(len(c) == 1 for c in comps)} stations with no cross-station ties")

    # ---- reprojection
    res = native_residuals(rec, project)
    r = res["residual_native_px"]
    ok = np.isfinite(r)
    r, iid_o, idx_o = r[ok], res["image_id"][ok], res["point2D_idx"][ok]
    check("reprojection", "residual_median_px", _q(r, 50), f"RMS {np.sqrt(np.mean(r ** 2)):.3f}" if r.size else "",
          "native px")
    check("reprojection", "residual_p95_px", _q(r, 95), f"max {r.max():.2f}" if r.size else "", "native px")
    groups: Dict[str, List[float]] = {}
    eye_med: Dict[str, float] = {}
    img_med: Dict[int, float] = {}
    edge: List[np.ndarray] = []
    if r.size:
        order = np.argsort(iid_o, kind="stable")
        for sel in np.split(order, np.flatnonzero(np.diff(iid_o[order])) + 1):
            iid = int(iid_o[sel[0]])
            meta = by_iid.get(iid, {})
            grp = meta.get("camera_group") or meta.get("instrument", "?")          # ZL034, not its focus bins
            key = f"{grp} x{meta.get('downsample_scale', '?')}"
            groups.setdefault(key, []).extend(r[sel].tolist())
            img_med[iid] = float(np.median(r[sel]))
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            p = np.asarray(cam.params, float)
            one = cam.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL")
            cx, cy = (p[1], p[2]) if one else (p[2], p[3])
            xy = np.array([im.points2D[int(k)].xy for k in idx_o[sel]], float)
            # v0p20: radius relative to the farthest frame corner (as coverage_by_radius), so an off-centre
            # principal point (Mastcam-Z) no longer pushes the outermost keypoints past the last bin
            rad = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) / np.hypot(max(cx, cam.width - cx), max(cy, cam.height - cy))
            edge.append(np.c_[rad, r[sel]])
        for key, v in groups.items():
            eye_med[key] = float(np.median(v))
    by_cam: Dict[str, List[float]] = {}
    for key, v in groups.items():
        by_cam.setdefault(key.split(" ")[0], []).extend(v)
    for cam_l in sorted(k for k in by_cam if len(k) > 1 and k[1] == "L"):
        cam_r = cam_l[0] + "R" + cam_l[2:]
        if cam_r in by_cam and by_cam[cam_l] and by_cam[cam_r]:
            a, b = float(np.median(by_cam[cam_l])), float(np.median(by_cam[cam_r]))
            check("reprojection", "residual_eye_ratio", max(a, b) / max(min(a, b), 1e-9),
                  f"median residual, worse eye / better eye: {cam_l}: {a:.3f}, {cam_r}: {b:.3f}")
    med_all = float(np.median(r)) if r.size else 0.0
    outl = sorted(((m, iid) for iid, m in img_med.items() if m > outlier_factor * med_all), reverse=True)
    check("reprojection", "outlier_image_fraction", len(outl) / max(1, len(img_med)),
          f"{len(outl)} images with median residual > {outlier_factor:g} x the overall median"
          + (": " + ", ".join(f"{by_iid[i]['name']} ({m:.2f})" for m, i in outl[:5]) if outl else ""))
    radial = {}
    if edge:
        E = np.vstack(edge)
        bins = [0.0, 0.25, 0.5, 0.75, 1.01]
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (E[:, 0] >= lo) & (E[:, 0] < hi)
            radial[f"{lo:.2f}-{min(hi, 1.0):.2f}"] = float(np.median(E[sel, 1])) if sel.any() else None
        cen, out_ = radial.get("0.00-0.25"), radial.get("0.75-1.00")
        check("reprojection", "edge_to_centre_residual", (out_ / cen) if cen and out_ else None,
              "median residual in the outer vs inner quarter of the radius: "
              + ", ".join(f"{k}: {v:.3f}" for k, v in radial.items() if v is not None))

    cov = coverage_by_radius(rec, project)
    wc = _corner_ratio(cov)
    if wc:
        check("reprojection", "corner_triangulated_ratio", wc[1],
              f"{wc[0]}: {100 * wc[3]:.0f} % of keypoints beyond 0.85 of the corner radius become tie points, "
              f"{100 * wc[2]:.0f} % inside 0.6 (distortion model invertible to the corners?)")

    # ---- camera model
    cams_after = {}
    db_cams = project.settings.get("database", {}).get("cameras", {})
    for instr, c0 in sorted(project.cameras.items()):
        cid = db_cams.get(instr)
        if cid is None or int(cid) not in rec.cameras:
            continue
        ch = _camera_change(c0, rec.cameras[int(cid)])
        cams_after[instr] = ch
        for k in ("focal_change_pct", "principal_point_change_px", "ray_displacement_max_px"):
            check("camera model", k, ch[k], instr, "%" if k.endswith("pct") else "full-res px")

    # ---- stereo rig
    rig_out = {}
    for fam, r0 in sorted(project.rig.items()):
        ref_id, sen_id = db_cams.get(r0["ref"]), db_cams.get(r0["sensor"])
        if ref_id is None or sen_id is None:
            continue
        # v0p31: with Navcam temperature bins the rig is copied per bin (ref NL_T..., sensor NR_T...), all held at
        # the same value; any of them stands for the family
        grp = lambda k: str((project.cameras.get(k) or {}).get("group") or k)                  # noqa: E731
        refs = {int(v) for k, v in db_cams.items() if grp(k) == r0["ref"]} | {int(ref_id)}
        sens = {int(v) for k, v in db_cams.items() if grp(k) == r0["sensor"]} | {int(sen_id)}
        for rid, rig in sorted(rec.rigs.items(), key=lambda kv: kv[1].ref_sensor_id.id != int(ref_id)):
            if rig.ref_sensor_id.id not in refs or fam in rig_out:
                continue
            for sid in rig.non_ref_sensors:
                if sid.id not in sens:
                    continue
                T = rig.sensor_from_rig(sid)
                R1, t1 = T.rotation.matrix(), np.asarray(T.translation)
                R0, t0 = np.asarray(r0["R_sensor_from_ref"]), np.asarray(r0["t_sensor_from_ref"])
                rig_out[fam] = {"rotation_change_deg": _rotation_deg(R1 @ R0.T),
                                "baseline_initial_m": float(np.linalg.norm(t0)),
                                "baseline_refined_m": float(np.linalg.norm(t1)),
                                "cahv_pairs": r0.get("n_pairs"), "cahv_rot_spread_deg": r0.get("rot_spread_deg"),
                                "cahv_t_spread_m": r0.get("t_spread_m")}
                check("stereo rig", "rig_rotation_change_deg", rig_out[fam]["rotation_change_deg"],
                      f"{fam}: baseline {rig_out[fam]['baseline_initial_m']:.5f} -> "
                      f"{rig_out[fam]['baseline_refined_m']:.5f} m", "deg")
                check("stereo rig", "rig_cahv_spread_deg", r0.get("rot_spread_deg"),
                      f"{fam}: largest deviation of the {r0.get('n_pairs')} CAHV pairs from their median", "deg")

    # ---- poses vs priors
    good = [x for x in rows if x["observations"] >= min_frame_observations]
    by_st: Dict[str, List[Dict[str, Any]]] = {}
    for x in good:
        by_st.setdefault(x["station"], []).append(x)
    labels = station_labels(project.images)
    st_tab = {}
    for st, v in sorted(by_st.items(), key=lambda kv: labels.get(kv[0], kv[0])):
        d = np.array([[x["dE_m"], x["dN_m"], x["dU_m"]] for x in v])
        st_tab[st] = {"label": labels.get(st, st), "images": len(v), "shift_median_m": float(np.median(np.linalg.norm(d, axis=1))),
                      "shift_spread_m": float(np.sqrt(np.mean(np.sum((d - np.median(d, 0)) ** 2, axis=1)))),
                      "attitude_median_deg": float(np.median([x["dAttitude_deg"] for x in v]))}
    if st_tab:
        worst = max(st_tab.items(), key=lambda kv: kv[1]["shift_median_m"])
        check("poses", "station_shift_median_m", worst[1]["shift_median_m"],
              f"largest station median |refined - prior| ({labels.get(worst[0], worst[0])})", "m")
        spread = max(st_tab.items(), key=lambda kv: kv[1]["shift_spread_m"])
        check("poses", "within_station_shift_spread_m", spread[1]["shift_spread_m"],
              f"largest RMS spread of the shifts within one station ({labels.get(spread[0], spread[0])})", "m")
    att = np.array([x["dAttitude_deg"] for x in good])
    check("poses", "attitude_change_p95_deg", _q(att, 95), f"median {_q(att, 50):.3f} deg" if att.size else "", "deg")
    # v0p30: is the block still in the East-North-Up frame?  The common part of every frame's attitude change
    # (refined relative to the label attitude, expressed in the world frame) is a rotation of the whole block;
    # the priors allow it only within attitude_prior_deg.  Reported about E, N and U so a tilt is told from a
    # turn in azimuth (about U).
    from scipy.spatial.transform import Rotation as _Rot
    rv = []
    for x in good:
        r = project.image(x["name"])
        iid = r.get("image_id")
        if iid is None or iid not in rec.images or not rec.images[iid].has_pose:
            continue
        R1 = rec.images[iid].cam_from_world().rotation.matrix()
        R0 = np.asarray(r["prior_R_w2c"], float)
        # camera axes in the world are R^T, so refined = W @ prior with W = R1^T R0: the world-frame rotation
        # taking the prior attitude to the refined one (v0p30.0 used R0^T R1, the same angle with the sign reversed)
        rv.append(_Rot.from_matrix(R1.T @ R0).as_rotvec())
    if rv:
        rv = np.degrees(np.array(rv))
        mean_rv = np.median(rv, axis=0)
        resid = np.linalg.norm(rv - mean_rv, axis=1)
        report_extra = {"block_rotation_deg": float(np.linalg.norm(mean_rv)),
                        "block_rotation_about_E_N_U_deg": mean_rv.tolist(),
                        "attitude_residual_median_deg": float(np.median(resid)),
                        "attitude_residual_p95_deg": float(np.percentile(resid, 95))}
        check("poses", "block_rotation_deg", float(np.linalg.norm(mean_rv)),
              f"median attitude change of the block in the world frame: about E {mean_rv[0]:+.3f}, N {mean_rv[1]:+.3f}, "
              f"U (azimuth) {mean_rv[2]:+.3f} deg; the frame stays ENU with the offset in project.offset", "deg")
        check("poses", "attitude_residual_p95_deg", float(np.percentile(resid, 95)),
              f"per-frame attitude change about the block rotation (label pointing knowledge); median "
              f"{np.median(resid):.3f} deg", "deg")
    else:
        report_extra = {}
    sims = []
    for comp in comps:
        cen = {s: by_st[s] for s in comp if s in by_st}
        if len(cen) < 3:
            continue
        P0 = np.array([np.mean([project.image(x["name"])["prior_C"] for x in v], 0) for v in cen.values()])
        P1 = np.array([np.mean([[x["E_m"], x["N_m"], x["U_m"]] for x in v], 0) for v in cen.values()])
        s_, R_, t_ = _umeyama(P0, P1)
        extent = float(np.sqrt(np.sum((P0 - P0.mean(0)) ** 2)))      # sqrt(sum |station - centroid|^2)
        rot = _rotation_deg(R_)
        sims.append({"stations": list(cen), "scale": float(s_), "rotation_deg": rot, "extent_m": extent,
                     "scale_error_m": abs(s_ - 1) * extent, "rotation_error_m": float(np.radians(rot)) * extent})
        check("poses", "prior_scale_error_pct", 100 * abs(s_ - 1), f"block of {len(cen)} stations", "%")
        check("poses", "prior_scale_error_m", abs(s_ - 1) * extent,
              f"{100 * abs(s_ - 1):.2f} % over {len(cen)} stations (sqrt sum r^2 {extent:.1f} m); the station layout "
              f"of the waypoints is turned by {rot:.2f} deg ({np.radians(rot) * extent:.2f} m) against the "
              f"solution, whose attitude follows the labels (block_rotation_deg)", "m")

    weak = diagnose_weak_images(project, rec, min_frame_observations)
    verdict = max((c["status"] for c in checks), key=lambda s: _RANK[s], default="pass")
    import datetime as _dt
    from .. import __version__
    return {"verdict": verdict, "project": str(project.root), "images": len(project.images),
            "created": _dt.datetime.now().isoformat(timespec="seconds"), "mppp_version": __version__,
            "checks": checks, "thresholds": {k: list(v) for k, v in thr.items()},
            "track_length_histogram": hist, "residual_by_camera_resolution": eye_med,
            "residual_by_radius": radial, "coverage_by_radius": cov, "worst_images": [{"name": by_iid[i]["name"], "median_px": m}
                                                           for m, i in outl[:20]],
            "cameras": cams_after, "rig": rig_out, "stations": st_tab, "prior_similarity": sims,
            "held_images": held, "weak_images": weak, "station_components": comps,
            "station_labels": labels, "world_frame": {"frame": project.settings.get("world_frame"),
                                                       "offset_enu_m": list(project.offset), **report_extra},
            "note": "thresholds are provisional values set a priori, not validated limits"}


def health_table(report: Dict[str, Any]) -> str:
    """Plain-text table: status, section, check, value, thresholds, note."""
    sym = {"pass": "ok  ", "info": "    ", "warn": "WARN", "fail": "FAIL"}
    out = [f"alignment health: {report['verdict'].upper()}   ({report['note']})"]
    if report.get("project"):
        out.append(f"project {report['project']} ({report.get('images')} images), assessed {report.get('created')} "
                   f"with MPPP {report.get('mppp_version')}")
    out.append("")
    for c in report["checks"]:
        v = "-" if c["value"] is None else (f"{c['value']:.4g}")
        t = "" if c["warn"] is None else f"[warn {c['warn']:g} / fail {c['fail']:g}]"
        out.append(f"{sym[c['status']]}  {c['section']:<13} {c['check']:<31} {v:>10} {c['unit']:<11} {t:<26} {c['note']}")
    cov = report.get("coverage_by_radius") or {}
    if cov.get("cameras"):
        e = cov["edges"]
        out += ["", "tie-point coverage by image radius (1 = farthest frame corner): % of keypoints triangulated "
                    "/ median residual [native px]"]
        out.append(f"  {'camera':<14}" + "".join(f"{e[i]:.2f}-{e[i + 1]:.2f}".rjust(13) for i in range(len(e) - 1)))
        for grp, c in cov["cameras"].items():
            cells = []
            for f, r, n in zip(c["triangulated_fraction"], c["residual_median_native_px"], c["keypoints"]):
                cells.append("-".rjust(13) if not n or f is None else
                             (f"{100 * f:3.0f}% / " + ("  -  " if r is None else f"{r:.2f}")).rjust(13))
            out.append(f"  {grp:<14}" + "".join(cells))
    weak = report.get("weak_images") or []
    if weak:
        out += ["", f"images with few observations ({len(weak)}), by likely cause:"]
        causes: Dict[str, List[Dict[str, Any]]] = {}
        for w in weak:
            causes.setdefault(w["cause"], []).append(w)
        for cause, ws in causes.items():
            out.append(f"  {cause} ({len(ws)}): {WEAK_CAUSES[cause]}")
            if WEAK_ADVICE.get(cause):
                out.append(f"    -> {WEAK_ADVICE[cause]}")
            for w in ws:
                m = "" if w["inliers_with_other_images"] is None else \
                    (f", inliers: stereo partner {w['inliers_with_stereo_partner']}, "
                     f"{w['matched_images']} other images {w['inliers_with_other_images']} "
                     f"(other stations {w['inliers_with_other_stations']})")
                out.append(f"      {w['name']}  ({w.get('station_label') or w['station']}; {w['keypoints']} keypoints, "
                           f"{w['observations']} observations{m})")
    return "\n".join(out)


def write_health(report: Dict[str, Any], out_dir: PathLike, rec=None, project: Optional[SfmProject] = None) -> Dict[str, str]:
    """``health.json``, ``health.md`` (the table) and, with ``rec`` and ``project``, ``health.png``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "health.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (out / "health.md").write_text("```\n" + health_table(report) + "\n```\n", encoding="utf-8")
    files = {"json": str(out / "health.json"), "md": str(out / "health.md")}
    if rec is not None and project is not None:
        try:
            files["png"] = str(plot_health(report, rec, project, out / "health.png"))
            try:                                                   # v0p30: top-down station map with the SfM shifts
                from .export import plot_camera_shifts
                import matplotlib.pyplot as plt
                fig = plot_camera_shifts(project, rec=rec, out_png=out / "station_map.png")
                plt.close(fig)
                files["station_map"] = str(out / "station_map.png")
            except Exception as e:                                 # noqa: BLE001
                files["station_map_error"] = f"{type(e).__name__}: {e}"
        except ImportError:
            pass
    return files


def plot_health(report: Dict[str, Any], rec, project: SfmProject, path: PathLike) -> Path:
    """Four panels: residual histogram, track lengths, residual vs radius, station shifts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = native_residuals(rec, project)["residual_native_px"]
    res = res[np.isfinite(res)]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(res, bins=np.linspace(0, max(2.0, float(np.percentile(res, 99))) if res.size else 2, 81))
    ax[0, 0].set_xlabel("residual [native px]")
    ax[0, 0].set_title(f"reprojection: median {np.median(res):.3f} px" if res.size else "reprojection")
    h = report["track_length_histogram"]
    ax[0, 1].bar(list(h), list(h.values()))
    ax[0, 1].set_xlabel("track length (images per tie point)")
    ax[0, 1].set_title("tie-point population")
    rr = report["residual_by_radius"]
    ax[1, 0].plot(range(len(rr)), [v if v is not None else np.nan for v in rr.values()], "o-")
    ax[1, 0].set_xticks(range(len(rr)))
    ax[1, 0].set_xticklabels(list(rr), fontsize=8)
    ax[1, 0].set_xlabel("radius / half-diagonal")
    ax[1, 0].set_ylabel("median residual [native px]")
    ax[1, 0].set_ylim(bottom=0)
    ax[1, 0].set_title("residual vs image radius (distortion fit)")
    st = report["stations"]
    ax[1, 1].bar(range(len(st)), [v["shift_median_m"] for v in st.values()], label="median shift")
    ax[1, 1].errorbar(range(len(st)), [v["shift_median_m"] for v in st.values()],
                      yerr=[v["shift_spread_m"] for v in st.values()], fmt="none", ecolor="k", label="spread")
    ax[1, 1].set_xticks(range(len(st)))
    ax[1, 1].set_xticklabels([v.get("label", k) for k, v in st.items()], rotation=90, fontsize=7)
    ax[1, 1].set_ylabel("|refined - prior| [m]")
    ax[1, 1].set_ylim(bottom=0)
    ax[1, 1].set_title("camera centre vs prior, by station")
    fig.suptitle(f"alignment health: {report['verdict'].upper()}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)
