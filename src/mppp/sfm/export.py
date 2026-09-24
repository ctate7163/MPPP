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
from .project import SfmProject
from .reconstruction import native_residuals, track_statistics

PathLike = Union[str, Path]


def write_native_text_model(rec, project: SfmProject, out_dir: PathLike) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scale = {r["name"]: float(r["downsample_scale"]) for r in project.images}
    cams: Dict[tuple, int] = {}
    cam_rows, img_cam = [], {}
    for iid in sorted(rec.reg_image_ids()):
        im = rec.images[iid]
        s = scale[im.name]
        key = (im.camera_id, s)
        if key not in cams:
            cam = rec.cameras[im.camera_id]
            cams[key] = len(cams) + 1
            cam_rows.append((cams[key], cam.model.name, int(round(cam.width * s)), int(round(cam.height * s)),
                             scale_camera_params(cam.model.name, cam.params, s)))
        img_cam[iid] = cams[key]
    write_cameras_txt(out / "cameras.txt", cam_rows, header="native resolution per image")

    img_rows = []
    for iid in sorted(rec.reg_image_ids()):
        im = rec.images[iid]
        T = im.cam_from_world()
        q = T.rotation.quat                       # x, y, z, w
        s = scale[im.name]
        obs = [(*(np.asarray(p2.xy) * s), p2.point3D_id) for p2 in im.points2D if p2.has_point3D()]
        img_rows.append((iid, (q[3], q[0], q[1], q[2]), np.asarray(T.translation), img_cam[iid], im.name, obs))
    write_images_txt(out / "images.txt", img_rows)

    res = native_residuals(rec, project)
    err: Dict[int, List[float]] = {}
    for r, pid in zip(res["residual_native_px"], res["point3D_id"]):
        err.setdefault(int(pid), []).append(r)
    lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR(native px), TRACK[] as (IMAGE_ID, POINT2D_IDX)"]
    for pid, pt in rec.points3D.items():
        tr = " ".join(f"{el.image_id} {el.point2D_idx}" for el in pt.track.elements)
        c = pt.color
        lines.append(f"{pid} {pt.xyz[0]:.9g} {pt.xyz[1]:.9g} {pt.xyz[2]:.9g} {int(c[0])} {int(c[1])} {int(c[2])} "
                     f"{np.mean(err.get(pid, [0.0])):.6g} {tr}")
    (out / "points3D.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"cameras": len(cams), "images": len(img_cam), "points": len(rec.points3D)}


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
        rows.append({"name": r["name"], "station": r["station"], "sol": r["sol"], "instrument": r["instrument"],
                     "downsample_scale": r["downsample_scale"], "dE_m": d[0], "dN_m": d[1], "dU_m": d[2],
                     "dC_m": float(np.linalg.norm(d)), "dAttitude_deg": dang,
                     "E_m": C[0], "N_m": C[1], "U_m": C[2], "observations": n_obs.get(iid, 0)})
    return rows


def export_for_error(project: SfmProject, rec, out_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    out = Path(out_dir) if out_dir else project.root / "error_input"
    out.mkdir(parents=True, exist_ok=True)
    native = write_native_text_model(rec, project, out / "native")

    with (out / "stations.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "station", "site", "drive", "sol", "sclk_key", "lmst", "solar_elevation_deg",
                    "instrument", "downsample_scale", "sequence", "registered"])
        reg = {rec.images[i].name for i in rec.reg_image_ids()}
        for r in project.images:
            w.writerow([r["name"], r["station"], r["site"], r["drive"], r["sol"], r["sclk_key"], r.get("lmst"),
                        r.get("solar_elevation_deg"), r["instrument"], r["downsample_scale"], r["sequence"],
                        r["name"] in reg])
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
               "station_components": comps, "prior_similarity_by_component": sims,
               "weak_images_lt30_obs": weak,
               "cameras_initial": project.cameras, "cameras_refined": cams_after,
               "rig_initial": project.rig, "rig_refined": rigs_after, "eps": eps,
               "world": {"frame": project.settings.get("world_frame"), "offset_enu_m": project.offset}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary
