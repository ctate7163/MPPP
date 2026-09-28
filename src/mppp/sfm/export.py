"""
Exports for error analysis.

``export_for_error(project, rec)`` writes ``<project>/error_input/``:

* ``native/`` — COLMAP TEXT model in each image's NATIVE pixels: one camera per
  (instrument, resolution) with f, cx, cy scaled by the downsample factor (the
  distortion coefficients are resolution-free), keypoints scaled back.  This is
  what ``mppp.error.read_colmap`` reads, and what matches the image files.
* ``stations.csv`` — image -> station (site/drive), sol, SCLK, LMST, solar
  elevation, instrument, downsample scale; ``mppp.error`` ``assign_stations``
  takes the name -> station mapping.
* ``poses.csv`` — CAHV/waypoint prior vs refined camera centre (ENU, metres)
  and attitude difference (degrees), per image.
* ``residuals.npz`` — per-observation residual in native pixels.
* ``summary.json`` — cameras and rig before/after, track statistics, and the
  ``mppp.error`` epsilon estimates.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..colmap import scale_camera_params, write_cameras_txt, write_images_txt
from .project import SfmProject, station_labels
from .reconstruction import native_residuals, track_statistics

PathLike = Union[str, Path]


def native_reconstruction(rec, project: SfmProject, observed_only: bool = True):
    """
    ``rec`` in each image's NATIVE pixels, as a complete COLMAP 4 model (v0p14.5):
    one camera per (camera, downsample scale) with f and c scaled, a trivial rig
    and one frame per image, keypoints scaled back, the 3-D points with their
    mean native residual as ``error``.  ``observed_only``: keep only keypoints
    that observe a 3-D point (track indices renumbered); otherwise all
    keypoints, in database order.
    """
    import pycolmap
    scale = {r["name"]: float(r["downsample_scale"]) for r in project.images}
    sensor = lambda cid: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid)       # noqa: E731
    nat = pycolmap.Reconstruction()
    cam_ids: Dict[tuple, int] = {}
    idx_map: Dict[int, Dict[int, int]] = {}
    reg = sorted(int(i) for i in rec.reg_image_ids())
    for iid in reg:
        im = rec.images[iid]
        s = scale[im.name]
        key = (int(im.camera_id), s)
        if key not in cam_ids:
            cam = rec.cameras[im.camera_id]
            cid = len(cam_ids) + 1
            nat.add_camera(pycolmap.Camera(camera_id=cid, model=cam.model.name, width=int(round(cam.width * s)),
                                           height=int(round(cam.height * s)),
                                           params=scale_camera_params(cam.model.name, cam.params, s)))
            rig = pycolmap.Rig(rig_id=cid)
            rig.add_ref_sensor(sensor(cid))
            nat.add_rig(rig)
            cam_ids[key] = cid
        cid = cam_ids[key]
        p2 = im.points2D
        keep = [k for k in range(len(p2)) if p2[k].has_point3D()] if observed_only else list(range(len(p2)))
        idx_map[iid] = {k: n for n, k in enumerate(keep)}
        kp = np.array([p2[k].xy for k in keep], float).reshape(-1, 2) * s
        fr = pycolmap.Frame(frame_id=iid, rig_id=cid)
        fr.add_data_id(pycolmap.data_t(sensor_id=sensor(cid), id=iid))
        fr.rig_from_world = im.cam_from_world()
        nat.add_frame(fr)
        new = pycolmap.Image(name=im.name, keypoints=kp, camera_id=cid, image_id=iid)
        new.frame_id = iid
        nat.add_image(new)
    for fid in list(nat.frames):
        nat.register_frame(fid)
    res = native_residuals(rec, project)
    err: Dict[int, List[float]] = {}
    for r, pid in zip(res["residual_native_px"], res["point3D_id"]):
        err.setdefault(int(pid), []).append(float(r))
    for pid, pt in rec.points3D.items():
        tr = pycolmap.Track()
        for el in pt.track.elements:
            m = idx_map.get(int(el.image_id))
            if m is not None and int(el.point2D_idx) in m:
                tr.add_element(int(el.image_id), m[int(el.point2D_idx)])
        if tr.length() < 2:
            continue
        new_pid = nat.add_point3D(np.asarray(pt.xyz, float), tr, np.asarray(pt.color, np.uint8))
        e = err.get(int(pid))
        if e:
            nat.points3D[new_pid].error = float(np.mean(e))
    return nat


def write_native_text_model(rec, project: SfmProject, out_dir: PathLike) -> Dict[str, Any]:
    """:func:`native_reconstruction` (observed keypoints only) as a COLMAP TEXT model in ``out_dir``.
    Up to v0p14.4 the track indices in points3D.txt did not match the shortened keypoint lists in
    images.txt, which made COLMAP (and the GUI) stop with 'Check failed: point2D.point3D_id == point3D_id'."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nat = native_reconstruction(rec, project, observed_only=True)
    nat.write_text(str(out))
    return {"cameras": len(nat.cameras), "images": len(nat.images), "points": len(nat.points3D)}


def point_colors_from_tracks(nat, images_dir: PathLike, invalid_black: bool = True) -> Dict[str, Any]:
    """
    Colour every 3-D point of the native-pixel model ``nat`` with the mean of
    the image pixels of all its observations (v0p30).  COLMAP's own
    ``extract_colors_for_all_images`` leaves a point black when the image it
    happens to take the colour from cannot be read or the keypoint falls on an
    invalid (black, masked) pixel; here every observation contributes, black
    pixels (``invalid_black``: RGB all zero, MPPP's invalid flag) are left out,
    and a point with no valid sample gets mid-grey.  Each image is read once.
    Returns counts.
    """
    import cv2
    images_dir = Path(images_dir)
    pids = np.array(sorted(nat.points3D), dtype=np.int64)
    index = {int(p): i for i, p in enumerate(pids)}
    acc = np.zeros((len(pids), 3), np.float64)
    cnt = np.zeros(len(pids), np.int64)
    n_img = n_missing = 0
    for iid, im in nat.images.items():
        rows = [(index[int(q.point3D_id)], k) for k, q in enumerate(im.points2D) if q.has_point3D() and int(q.point3D_id) in index]
        if not rows:
            continue
        f = images_dir / im.name
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f.is_file() else None
        if img is None:
            n_missing += 1
            continue
        n_img += 1
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.shape[2] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = (img.astype(np.float64) / (65535.0 if img.dtype == np.uint16 else img.max() or 1) * 255).astype(np.uint8)
        rgb = img[..., ::-1]                                       # cv2 loads BGR
        h, w = rgb.shape[:2]
        xy = np.array([im.points2D[k].xy for _, k in rows], float)
        x = np.clip(np.round(xy[:, 0] - 0.5).astype(int), 0, w - 1)  # COLMAP corner origin -> pixel index
        y = np.clip(np.round(xy[:, 1] - 0.5).astype(int), 0, h - 1)
        px = rgb[y, x].astype(np.float64)
        ok = np.ones(len(rows), bool) if not invalid_black else px.sum(axis=1) > 0
        idx = np.array([i for i, _ in rows])
        np.add.at(acc, idx[ok], px[ok])
        np.add.at(cnt, idx[ok], 1)
    mean = np.full((len(pids), 3), 128.0)
    has = cnt > 0
    mean[has] = acc[has] / cnt[has, None]
    for i, pid in enumerate(pids):
        nat.points3D[int(pid)].color = np.round(mean[i]).astype(np.uint8)
    return {"points": int(len(pids)), "coloured_from_tracks": int(has.sum()), "images_read": n_img,
            "images_missing": n_missing}


def write_gui_native(project: SfmProject, rec, out_dir: Optional[PathLike] = None, colors: bool = True) -> Dict[str, Any]:
    """
    A copy of the project for LOOKING at it in the COLMAP GUI (v0p14.5), in
    ``<project>/gui_native/``: the refined model and a database in each image's
    native pixels (one camera per camera and resolution), so keypoints, tie
    points and matches line up with the image files.  ``sparse/`` is the model
    (all keypoints, point colours taken from the images), ``database.db`` holds
    cameras, rigs, frames, images, native keypoints and the verified matches
    (no descriptors: not for matching).  Opened by ``open_in_colmap.bat``.
    Not for processing: MPPP's bundle adjustment works on the full-resolution
    project.
    """
    import pycolmap
    from .database import scale_keypoints
    out = Path(out_dir) if out_dir else project.root / "gui_native"
    (out / "sparse").mkdir(parents=True, exist_ok=True)
    nat = native_reconstruction(rec, project, observed_only=False)
    color_report = None
    if colors:
        try:
            color_report = point_colors_from_tracks(nat, project.images_dir)      # v0p30: mean over the track
        except Exception as e:                               # noqa: BLE001  (colours are cosmetic)
            color_report = {"error": f"{type(e).__name__}: {e}"}
    nat.write(str(out / "sparse"))
    dbp = out / "database.db"
    if dbp.exists():
        dbp.unlink()
    src = pycolmap.Database.open(str(project.database))
    dst = pycolmap.Database.open(str(dbp))
    scale = {r["name"]: float(r["downsample_scale"]) for r in project.images}
    for cid, cam in sorted(nat.cameras.items()):
        dst.write_camera(cam, use_camera_id=True)
    for rid, rig in sorted(nat.rigs.items()):
        dst.write_rig(rig, use_rig_id=True)
    cam_of = {im.name: (im.image_id, im.camera_id) for im in nat.images.values()}
    n_img = 0
    for im in src.read_all_images():
        if im.name not in cam_of:                            # not registered: left out of the view
            continue
        iid, cid = cam_of[im.name]
        dst.write_image(pycolmap.Image(name=im.name, camera_id=cid, image_id=iid), use_image_id=True)
        dst.write_keypoints(iid, scale_keypoints(src.read_keypoints(im.image_id), scale[im.name]))
        n_img += 1
    for fid, fr in sorted(nat.frames.items()):
        dst.write_frame(fr, use_frame_id=True)
    ids = {v[0] for v in cam_of.values()}
    n_pairs = 0
    pair_ids, geoms = src.read_two_view_geometries()
    for pid, g in zip(pair_ids, geoms):
        i1, i2 = pycolmap.pair_id_to_image_pair(pid)
        if int(i1) in ids and int(i2) in ids:
            dst.write_two_view_geometry(int(i1), int(i2), g)
            n_pairs += 1
    src.close()
    dst.close()
    return {"dir": str(out), "images": n_img, "verified_pairs": n_pairs, "points": len(nat.points3D), "colors": color_report,
            "cameras": len(nat.cameras)}


def station_components(rec, project: SfmProject, min_shared_points: int = 50) -> List[List[str]]:
    """Groups of stations tied by >= ``min_shared_points`` shared 3-D points (connected components)."""
    station = {r["image_id"]: r["station"] for r in project.images if "image_id" in r}
    shared: Dict[tuple, int] = {}
    for pt in rec.points3D.values():
        sts = sorted({station[el.image_id] for el in pt.track.elements})
        for a in range(len(sts)):
            for b in range(a + 1, len(sts)):
                shared[(sts[a], sts[b])] = shared.get((sts[a], sts[b]), 0) + 1
    parent = {s: s for s in set(station.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), n in shared.items():
        if n >= min_shared_points:
            parent[find(a)] = find(b)
    comps: Dict[str, List[str]] = {}
    for s in parent:
        comps.setdefault(find(s), []).append(s)
    return sorted((sorted(v) for v in comps.values()), key=lambda c: (-len(c), c[0]))


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity dst ~ s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / A.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def pose_residual_table(rec, project: SfmProject) -> List[Dict[str, Any]]:
    from scipy.spatial.transform import Rotation
    labels = station_labels(project.images)
    rows = []
    n_obs: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            n_obs[el.image_id] = n_obs.get(el.image_id, 0) + 1
    for r in project.images:
        iid = r.get("image_id")
        if iid is None or iid not in rec.images or not rec.images[iid].has_pose:
            continue
        T = rec.images[iid].cam_from_world()
        R = T.rotation.matrix()
        C = -R.T @ np.asarray(T.translation)
        C0 = np.asarray(r["prior_C"], float)
        R0 = np.asarray(r["prior_R_w2c"], float)
        dang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R @ R0.T).as_rotvec())))
        d = C - C0
        rows.append({"name": r["name"], "station": r["station"], "station_label": labels.get(r["station"], r["station"]),
                     "sol": r["sol"], "instrument": r["instrument"],
                     "downsample_scale": r["downsample_scale"], "dE_m": d[0], "dN_m": d[1], "dU_m": d[2],
                     "dC_m": float(np.linalg.norm(d)), "dAttitude_deg": dang,
                     "E_m": C[0], "N_m": C[1], "U_m": C[2], "observations": n_obs.get(iid, 0)})
    return rows


def _nice(x: float) -> float:
    """1, 2 or 5 times a power of ten, nearest to ``x`` from below."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(x))
    return float(max(m for m in (1, 2, 5) if m * e <= x * 1.0000001) * e)


def plot_camera_shifts(project: SfmProject, rec=None, rows: Optional[List[Dict[str, Any]]] = None,
                       out_png: Optional[PathLike] = None, exaggeration: Optional[float] = None,
                       min_observations: int = 30, ncols: int = 4):
    """
    Top-down view of how far each camera moved from its reference (its CAHV +
    waypoint prior) in the refinement (v0p14.4).  Stations are tens of metres
    apart while the cameras of one station lie within a metre, so:

    * top left - overview: every station's median shift, prior -> refined, as an
      arrow exaggerated ``k_overview`` times, coloured by its median vertical
      shift dU, labelled ``Sol0686 S032D1184``;
    * top right - every camera's total shift |dC| per station (dots coloured by
      the attitude change);
    * below - one panel per station in local coordinates (metres from the
      station's median camera centre): one arrow per image from its prior to its
      refined centre, horizontal shift exaggerated ``exaggeration`` times (shared
      by all panels; default: the median arrow is ~30 % of the median station
      extent, rounded to 1/2/5), coloured by dU (shared scale).

    Navcam = circles, Mastcam-Z = triangles; images with fewer than
    ``min_observations`` observations (held at their prior) are grey crosses.
    ``rows``: from :func:`pose_residual_table` or ``poses.csv`` (computed from
    ``rec`` if omitted).  Returns the figure (saved to ``out_png`` if given).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    if rows is None:
        if rec is None:
            raise ValueError("give rec or rows")
        rows = pose_residual_table(rec, project)
    labels = station_labels(project.images)
    R = []
    for r in rows:
        r = dict(r)
        for k in ("E_m", "N_m", "U_m", "dE_m", "dN_m", "dU_m", "dC_m", "dAttitude_deg"):
            r[k] = float(r[k])
        r["observations"] = int(float(r["observations"]))
        r["label"] = r.get("station_label") or labels.get(r["station"], r["station"])
        R.append(r)
    stations = sorted({r["label"] for r in R})
    by_st = {s: [r for r in R if r["label"] == s] for s in stations}
    good = [r for r in R if r["observations"] >= min_observations]
    fam_marker = lambda r: "^" if r["instrument"][:1] == "Z" else ("o" if r["instrument"][:1] == "N" else "s")   # noqa: E731
    cmap = plt.get_cmap("coolwarm")
    du = np.array([r["dU_m"] for r in good])
    lim = max(float(np.percentile(np.abs(du), 98)) if du.size else 0.0, 1e-3)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)

    # station summaries and local coordinates
    summ = {}
    for s, v in by_st.items():
        g = [r for r in v if r["observations"] >= min_observations]
        cE, cN = float(np.median([r["E_m"] for r in v])), float(np.median([r["N_m"] for r in v]))
        summ[s] = {"E": cE, "N": cN, "n": len(g),
                   "dE": float(np.median([r["dE_m"] for r in g])) if g else 0.0,
                   "dN": float(np.median([r["dN_m"] for r in g])) if g else 0.0,
                   "dU": float(np.median([r["dU_m"] for r in g])) if g else 0.0,
                   "dH": float(np.median([np.hypot(r["dE_m"], r["dN_m"]) for r in g])) if g else 0.0,
                   "extent": max(float(np.ptp([r["E_m"] for r in v])), float(np.ptp([r["N_m"] for r in v])), 0.5)}
    dh_all = np.array([np.hypot(r["dE_m"], r["dN_m"]) for r in good])
    med_dh = float(np.median(dh_all)) if dh_all.size and np.median(dh_all) > 0 else 0.01
    if exaggeration is None:
        exaggeration = max(1.0, _nice(0.3 * float(np.median([v["extent"] for v in summ.values()] or [1.0])) / med_dh))
    k = float(exaggeration)
    E_all = np.array([v["E"] for v in summ.values()]); N_all = np.array([v["N"] for v in summ.values()])
    extent_all = max(float(np.ptp(E_all)) if E_all.size else 0.0, float(np.ptp(N_all)) if N_all.size else 0.0, 5.0)
    k_ov = _nice(0.08 * extent_all / max(float(np.median([v["dH"] for v in summ.values()] or [med_dh])), 1e-3))

    n_rows = int(np.ceil(len(stations) / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, 5.2 + 3.9 * n_rows))
    gs = fig.add_gridspec(1 + n_rows, ncols, height_ratios=[1.45] + [1.0] * n_rows)
    ax0 = fig.add_subplot(gs[0, : ncols // 2])
    ax1 = fig.add_subplot(gs[0, ncols // 2:])

    # overview
    for r in R:
        ax0.plot(r["E_m"], r["N_m"], fam_marker(r), color="0.75", ms=3, zorder=1)
    for s, v in summ.items():
        ax0.annotate("", (v["E"], v["N"]), (v["E"] - k_ov * v["dE"], v["N"] - k_ov * v["dN"]),
                     arrowprops=dict(arrowstyle="->", lw=1.8, color=cmap(norm(v["dU"]))), zorder=2)
        ax0.text(v["E"], v["N"], "  " + s, fontsize=6.5, va="center", zorder=3)
    bar = _nice(0.15 * extent_all / k_ov)
    x0, y0 = float(E_all.min()), float(N_all.min()) - 0.06 * extent_all
    ax0.annotate("", (x0 + k_ov * bar, y0), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.6, color="k"))
    ax0.text(x0, y0 - 0.035 * extent_all, f"{bar * 100:g} cm (x{k_ov:g})", fontsize=8, va="top")
    xo = np.r_[E_all, E_all - k_ov * np.array([v["dE"] for v in summ.values()]), x0, x0 + k_ov * bar]
    yo = np.r_[N_all, N_all - k_ov * np.array([v["dN"] for v in summ.values()]), y0 - 0.05 * extent_all]
    half = 0.55 * max(np.ptp(xo), np.ptp(yo)) + 0.05 * extent_all
    ax0.set_xlim(0.5 * (xo.min() + xo.max()) - half, 0.5 * (xo.min() + xo.max()) + half)
    ax0.set_ylim(0.5 * (yo.min() + yo.max()) - half, 0.5 * (yo.min() + yo.max()) + half)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("E [m]"); ax0.set_ylabel("N [m]"); ax0.grid(alpha=0.3)
    ax0.set_title(f"station median shift, prior -> refined (x{k_ov:g}); colour = dU")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax0, shrink=0.85, label="dU, refined - prior [m]")

    # per-camera total shift by station, coloured by attitude change
    att = np.array([r["dAttitude_deg"] for r in good])
    vmax = max(float(np.percentile(att, 98)) if att.size else 0.1, 1e-3)
    rng = np.random.default_rng(0)
    sc = None
    for i, s in enumerate(stations):
        g = [r for r in by_st[s] if r["observations"] >= min_observations]
        h = [r for r in by_st[s] if r["observations"] < min_observations]
        for fam in ("o", "^", "s"):
            gg = [r for r in g if fam_marker(r) == fam]
            if gg:
                sc = ax1.scatter(i + rng.uniform(-0.18, 0.18, len(gg)), [100 * r["dC_m"] for r in gg],
                                 c=[r["dAttitude_deg"] for r in gg], cmap="viridis", vmin=0, vmax=vmax, marker=fam,
                                 s=18, edgecolors="k", linewidths=0.25)
        if h:
            ax1.plot(i + rng.uniform(-0.18, 0.18, len(h)), [100 * r["dC_m"] for r in h], "x", color="0.55", ms=5)
    ax1.set_xticks(range(len(stations)))
    ax1.set_xticklabels(stations, rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("|refined - prior| camera centre [cm]")
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_title(f"per camera: median {100 * float(np.median([r['dC_m'] for r in good])) if good else 0:.1f} cm, "
                  f"attitude median {float(np.median(att)) if att.size else 0:.3f} deg "
                  f"(95 %: {float(np.percentile(att, 95)) if att.size else 0:.3f})")
    if sc is not None:
        fig.colorbar(sc, ax=ax1, shrink=0.85, label="attitude change [deg]")

    # one panel per station, local coordinates, shared exaggeration and colours
    for i, s in enumerate(stations):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        v, c = by_st[s], summ[s]
        xs, ys = [], []
        for r in v:
            x, y = r["E_m"] - c["E"], r["N_m"] - c["N"]
            xs += [x, x - k * r["dE_m"]]; ys += [y, y - k * r["dN_m"]]
            if r["observations"] < min_observations:
                ax.plot(x, y, "x", color="0.55", ms=6)
                continue
            ax.annotate("", (x, y), (x - k * r["dE_m"], y - k * r["dN_m"]),
                        arrowprops=dict(arrowstyle="->", lw=1.1, color=cmap(norm(r["dU_m"]))))
            ax.plot(x, y, fam_marker(r), color=cmap(norm(r["dU_m"])), mec="k", mew=0.3, ms=5)
        half = 0.55 * max(np.ptp(xs) if xs else 1.0, np.ptp(ys) if ys else 1.0, 0.3)
        mx, my = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))) if xs else (0.0, 0.0)
        ax.set_xlim(mx - half, mx + half); ax.set_ylim(my - half, my + half)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{s}\nmedian |dH| {100 * c['dH']:.1f} cm, dU {100 * c['dU']:+.1f} cm, {c['n']} cameras",
                     fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("N - station [m]", fontsize=8)
        ax.set_xlabel("E - station [m]", fontsize=8)
    fig.suptitle(f"camera shifts relative to the CAHV + waypoint references; station panels: horizontal x{k:g}, "
                 f"colour = dU (+/-{lim * 100:.1f} cm); o Navcam, ^ Mastcam-Z, x held (< {min_observations} obs.)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
    return fig


def export_for_error(project: SfmProject, rec, out_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    out = Path(out_dir) if out_dir else project.root / "error_input"
    out.mkdir(parents=True, exist_ok=True)
    native = write_native_text_model(rec, project, out / "native")

    with (out / "stations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        labels = station_labels(project.images)
        w.writerow(["name", "station", "site", "drive", "sol", "sclk_key", "lmst", "solar_elevation_deg", "solar_azimuth_deg",
                    "instrument", "downsample_scale", "sequence", "registered", "station_label"])
        reg = {rec.images[i].name for i in rec.reg_image_ids()}
        for r in project.images:
            w.writerow([r["name"], r["station"], r["site"], r["drive"], r["sol"], r["sclk_key"], r.get("lmst"),
                        r.get("solar_elevation_deg"), r.get("solar_azimuth_deg"), r["instrument"], r["downsample_scale"], r["sequence"],
                        r["name"] in reg, labels.get(r["station"], r["station"])])
    rows = pose_residual_table(rec, project)
    if rows:
        with (out / "poses.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    res = native_residuals(rec, project)
    np.savez_compressed(out / "residuals.npz", **res)

    eps = {}
    try:
        from ..error.colmap import read_colmap, calibrate_eps_from_residuals
        model = read_colmap(str(out / "native"))
        model.assign_stations({r["name"]: r["station"] for r in project.images})
        eps = calibrate_eps_from_residuals(model)
    except Exception as e:                                       # noqa: BLE001
        eps = {"error": f"{type(e).__name__}: {e}"}

    cams_after = {}
    for cid, cam in rec.cameras.items():
        instr = next((k for k, v in project.settings.get("database", {}).get("cameras", {}).items() if v == cid), cid)
        cams_after[str(instr)] = {"model": cam.model.name, "params": np.asarray(cam.params).tolist()}
    rigs_after = {}
    for rid, rig in rec.rigs.items():
        for sid in rig.non_ref_sensors:
            T = rig.sensor_from_rig(sid)
            rigs_after[str(rid)] = {"sensor_camera_id": sid.id, "R": T.rotation.matrix().tolist(),
                                    "t": np.asarray(T.translation).tolist(),
                                    "baseline_m": float(np.linalg.norm(T.translation))}
    comps = station_components(rec, project)
    weak = [r["name"] for r in rows if r["observations"] < 30]
    sims = []
    for comp in comps:
        sel = [r for r in rows if r["station"] in comp and r["observations"] >= 30]
        cen = {}
        for r in sel:
            cen.setdefault(r["station"], []).append(r)
        if len(cen) < 3:
            continue
        P0 = np.array([np.mean([np.asarray(project.image(x["name"])["prior_C"]) for x in v], 0) for v in cen.values()])
        P1 = np.array([np.mean([[x["E_m"], x["N_m"], x["U_m"]] for x in v], 0) for v in cen.values()])
        s_, R_, t_ = _umeyama(P0, P1)
        resid = P1 - (s_ * P0 @ R_.T + t_)
        sims.append({"stations": list(cen), "scale_refined_over_prior": float(s_),
                     "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R_) - 1) / 2, -1, 1)))),
                     "station_residual_after_similarity_m": dict(zip(cen, np.linalg.norm(resid, axis=1).round(4).tolist()))})
    dC = np.array([r["dC_m"] for r in rows if r["observations"] >= 30]) if rows else np.zeros(0)
    by_station: Dict[str, List[float]] = {}
    for r in rows:
        if r["observations"] >= 30:
            by_station.setdefault(r["station"], []).append(r["dC_m"])
    summary = {"native_model": native, "tracks": track_statistics(rec, project),
               "residual_rms_native_px": float(np.sqrt(np.mean(res["residual_native_px"] ** 2)))
               if res["residual_native_px"].size else None,
               "residual_median_native_px": float(np.median(res["residual_native_px"]))
               if res["residual_native_px"].size else None,
               "registered_images": len(rows), "project_images": len(project.images),
               "prior_offset_median_m": float(np.median(dC)) if dC.size else None,
               "prior_offset_by_station_median_m": {k: float(np.median(v)) for k, v in sorted(by_station.items())},
               "station_components": [[labels.get(x, x) for x in c] for c in comps],
               "prior_similarity_by_component": sims,
               "weak_images_lt30_obs": weak,
               "cameras_initial": project.cameras, "cameras_refined": cams_after,
               "rig_initial": project.rig, "rig_refined": rigs_after, "eps": eps,
               "world": {"frame": project.settings.get("world_frame"), "offset_enu_m": project.offset}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary
