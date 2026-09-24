"""
Alignment health (v0p13): is the COLMAP alignment good enough to analyse?

Run it after :func:`mppp.sfm.reconstruct` (and before any error analysis on the
result).  Each check has a value, provisional warn/fail thresholds and a
status; the report's ``verdict`` is the worst status.

Checks
------
registration      images registered; frames held at their prior because they have < 30 observations
tie points        points, observations, track-length population (incl. two-view points),
                  observations per image (5th percentile), cross-station fraction, tied station blocks
reprojection      native-pixel residuals: median, RMS, 95th percentile; per camera and resolution;
                  left/right balance; images whose median residual is far above the rest;
                  residual growth towards the image edge (distortion-model misfit)
camera model      per camera: change in f, principal point and k1-k3 from the calibration, and the
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
from .project import SfmProject
from .reconstruction import native_residuals, track_statistics

PathLike = Union[str, Path]

# name: (warn, fail, direction) - "above": bad when value > threshold; "below": bad when value < threshold
DEFAULT_THRESHOLDS: Dict[str, tuple] = {
    "registered_fraction": (0.98, 0.90, "below"),
    "held_frame_fraction": (0.02, 0.10, "above"),
    "obs_per_image_p05": (200, 50, "below"),
    "cross_station_fraction": (0.02, 0.005, "below"),
    "tied_block_fraction": (0.8, 0.5, "below"),
    "residual_median_px": (0.5, 1.0, "above"),
    "residual_p95_px": (2.0, 4.0, "above"),
    "residual_eye_ratio": (1.3, 2.0, "above"),
    "outlier_image_fraction": (0.02, 0.10, "above"),
    "edge_to_centre_residual": (1.5, 2.5, "above"),
    "focal_change_pct": (0.5, 2.0, "above"),
    "principal_point_change_px": (20.0, 60.0, "above"),
    "ray_displacement_max_px": (10.0, 30.0, "above"),
    "rig_rotation_change_deg": (0.03, 0.1, "above"),
    "rig_cahv_spread_deg": (0.02, 0.1, "above"),
    "station_shift_median_m": (1.0, 3.0, "above"),
    "attitude_change_p95_deg": (0.5, 2.0, "above"),
    "within_station_shift_spread_m": (0.05, 0.2, "above"),
    "prior_scale_error_pct": (1.0, 3.0, "above"),
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
    """Intrinsic change, and the image displacement of the same ray, over the calibration's pinhole footprint."""
    import pycolmap
    p0, p1 = np.asarray(cam0["params"], float), np.asarray(cam1.params, float)
    w, h = int(cam0["width"]), int(cam0["height"])
    c0 = pycolmap.Camera(model=cam0["model"], width=w, height=h, params=p0)
    one = cam0["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL")
    f0, f1 = (p0[0], p1[0]) if one else (p0[0], p1[0])
    cx0, cy0 = (p0[1], p0[2]) if one else (p0[2], p0[3])
    cx1, cy1 = (p1[1], p1[2]) if one else (p1[2], p1[3])
    gx, gy = np.meshgrid(np.linspace(0, w, 33), np.linspace(0, h, 25))
    rays = np.c_[(gx.ravel() - cx0) / f0, (gy.ravel() - cy0) / f0, np.ones(gx.size)]
    u0, u1 = c0.img_from_cam(rays), cam1.img_from_cam(rays)
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
        for k, i in (("k1", 4), ("k2", 5), ("k3", 8)):
            if i < len(p0):
                out[f"{k}_initial"], out[f"{k}_refined"] = float(p0[i]), float(p1[i])
    return out


def _rotation_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


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
    check("registration", "registered_fraction", len(reg) / max(1, len(project.images)),
          f"{len(reg)} of {len(project.images)} images")
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
            key = f"{meta.get('instrument', '?')} x{meta.get('downsample_scale', '?')}"
            groups.setdefault(key, []).extend(r[sel].tolist())
            img_med[iid] = float(np.median(r[sel]))
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            p = np.asarray(cam.params, float)
            one = cam.model.name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL")
            cx, cy = (p[1], p[2]) if one else (p[2], p[3])
            xy = np.array([im.points2D[int(k)].xy for k in idx_o[sel]], float)
            rad = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) / np.hypot(cam.width / 2, cam.height / 2)
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
        for rid, rig in rec.rigs.items():
            if rig.ref_sensor_id.id != int(ref_id):
                continue
            for sid in rig.non_ref_sensors:
                if sid.id != int(sen_id):
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
    st_tab = {}
    for st, v in sorted(by_st.items()):
        d = np.array([[x["dE_m"], x["dN_m"], x["dU_m"]] for x in v])
        st_tab[st] = {"images": len(v), "shift_median_m": float(np.median(np.linalg.norm(d, axis=1))),
                      "shift_spread_m": float(np.sqrt(np.mean(np.sum((d - np.median(d, 0)) ** 2, axis=1)))),
                      "attitude_median_deg": float(np.median([x["dAttitude_deg"] for x in v]))}
    if st_tab:
        worst = max(st_tab.items(), key=lambda kv: kv[1]["shift_median_m"])
        check("poses", "station_shift_median_m", worst[1]["shift_median_m"],
              f"largest station median |refined - prior| ({worst[0]})", "m")
        spread = max(st_tab.items(), key=lambda kv: kv[1]["shift_spread_m"])
        check("poses", "within_station_shift_spread_m", spread[1]["shift_spread_m"],
              f"largest RMS spread of the shifts within one station ({spread[0]})", "m")
    att = np.array([x["dAttitude_deg"] for x in good])
    check("poses", "attitude_change_p95_deg", _q(att, 95), f"median {_q(att, 50):.3f} deg" if att.size else "", "deg")
    sims = []
    for comp in comps:
        cen = {s: by_st[s] for s in comp if s in by_st}
        if len(cen) < 3:
            continue
        P0 = np.array([np.mean([project.image(x["name"])["prior_C"] for x in v], 0) for v in cen.values()])
        P1 = np.array([np.mean([[x["E_m"], x["N_m"], x["U_m"]] for x in v], 0) for v in cen.values()])
        s_, R_, t_ = _umeyama(P0, P1)
        sims.append({"stations": list(cen), "scale": float(s_), "rotation_deg": _rotation_deg(R_)})
        check("poses", "prior_scale_error_pct", 100 * abs(s_ - 1), f"block of {len(cen)} stations; rotation "
              f"{_rotation_deg(R_):.2f} deg (orientation is held only by the position priors)", "%")

    verdict = max((c["status"] for c in checks), key=lambda s: _RANK[s], default="pass")
    return {"verdict": verdict, "checks": checks, "thresholds": {k: list(v) for k, v in thr.items()},
            "track_length_histogram": hist, "residual_by_camera_resolution": eye_med,
            "residual_by_radius": radial, "worst_images": [{"name": by_iid[i]["name"], "median_px": m}
                                                           for m, i in outl[:20]],
            "cameras": cams_after, "rig": rig_out, "stations": st_tab, "prior_similarity": sims,
            "held_images": held, "station_components": comps,
            "note": "thresholds are provisional values set a priori, not validated limits"}


def health_table(report: Dict[str, Any]) -> str:
    """Plain-text table: status, section, check, value, thresholds, note."""
    sym = {"pass": "ok  ", "info": "    ", "warn": "WARN", "fail": "FAIL"}
    out = [f"alignment health: {report['verdict'].upper()}   ({report['note']})", ""]
    for c in report["checks"]:
        v = "-" if c["value"] is None else (f"{c['value']:.4g}")
        t = "" if c["warn"] is None else f"[warn {c['warn']:g} / fail {c['fail']:g}]"
        out.append(f"{sym[c['status']]}  {c['section']:<13} {c['check']:<31} {v:>10} {c['unit']:<11} {t:<26} {c['note']}")
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
    ax[1, 1].set_xticklabels(list(st), rotation=90, fontsize=7)
    ax[1, 1].set_ylabel("|refined - prior| [m]")
    ax[1, 1].set_ylim(bottom=0)
    ax[1, 1].set_title("camera centre vs prior, by station")
    fig.suptitle(f"alignment health: {report['verdict'].upper()}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)
