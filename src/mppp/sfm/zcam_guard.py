"""
v0p85 (19 Oct 2026): a guard against a Mastcam-Z eye dropping out of a block in stage 2.

In the v0p82 run (18-19 Oct) the right eye (ZR034) aligned in the first rounds of stage 2 and then lost every tie
point at one re-triangulation: at round 3 (right after its cameras were set on the focus lines) in Chal Rocks, South
Arm and Marble Mountain, and at the pose-guided step in Rochette - the block's points halved each time and the ZR
frames were excluded "with no tie points" (626 of 1096 Mastcam-Z frames aligned, 17 ZR outside Rio Chiquito).

:func:`group_coverage` counts, per Mastcam-Z group (eye and zoom, ``ZR034``), the frames with at least ``min_obs``
observations; :func:`collapsed` names the groups that lost more than ``max_loss`` of them.  Stage 2 uses it after
each triangulation: a collapse right after the focus lines were applied undoes that group's line values (the cameras
go back to their refined values) and triangulates again; a collapse after the pose-guided matches undoes that step.
The counts of every round go to the alignment log (``zcam_coverage``), so the next run shows where an eye is lost.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

MIN_OBS = 20            # a frame "has tie points" with at least this many observations (min_frame_observations)
MAX_LOSS = 0.5          # a group that loses more than this share of its frames with tie points has collapsed
MIN_FRAMES = 4          # groups with fewer frames with tie points are not judged


def group_of(project) -> Dict[int, str]:
    """{image_id: Mastcam-Z group} (``camera_group``, e.g. ``ZR034``) for the Mastcam-Z images of the project."""
    out = {}
    for r in project.images:
        if "image_id" in r and str(r.get("instrument", "")).startswith("Z"):
            g = r.get("camera_group") or str(r.get("instrument", ""))[:5]
            out[int(r["image_id"])] = str(g)[:5]
    return out


def group_coverage(rec, project, min_obs: int = MIN_OBS) -> Dict[str, Dict[str, int]]:
    """{group: {"frames": registered frames, "with_points": frames with >= min_obs observations, "observations"}}."""
    out: Dict[str, Dict[str, int]] = {}
    for iid, g in group_of(project).items():
        if iid not in rec.images:
            continue
        im = rec.images[iid]
        try:
            if not im.has_pose:
                continue
        except Exception:                                    # noqa: BLE001 - older pycolmap: registered images only
            pass
        n = int(im.num_points3D)
        e = out.setdefault(g, {"frames": 0, "with_points": 0, "observations": 0})
        e["frames"] += 1
        e["with_points"] += int(n >= min_obs)
        e["observations"] += n
    return out


def collapsed(before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]], max_loss: float = MAX_LOSS,
              min_frames: int = MIN_FRAMES) -> List[str]:
    """Groups whose frames with tie points fell by more than ``max_loss`` (only groups with >= ``min_frames``)."""
    out = []
    for g, b in (before or {}).items():
        nb = int(b.get("with_points", 0))
        if nb < min_frames:
            continue
        na = int((after.get(g) or {}).get("with_points", 0))
        if na < (1.0 - max_loss) * nb:
            out.append(g)
    return sorted(out)


def camera_ids_of(project, groups: Iterable[str]) -> List[int]:
    """The COLMAP camera ids of the Mastcam-Z bin cameras of ``groups``."""
    gs = set(groups)
    db = (project.settings.get("database") or {}).get("cameras") or {}
    out = []
    for key, cid in db.items():
        k = str(key)
        if k.startswith("Z") and k[:5] in gs:
            out.append(int(cid))
    return sorted(set(out))


def snapshot(rec, project) -> Dict[int, Any]:
    """The parameters of every Mastcam-Z camera in ``rec`` (to undo a line application)."""
    import numpy as np
    db = (project.settings.get("database") or {}).get("cameras") or {}
    return {int(cid): np.array(rec.cameras[int(cid)].params, float).copy()
            for key, cid in db.items() if str(key).startswith("Z") and int(cid) in rec.cameras}


def restore(rec, snap: Dict[int, Any], camera_ids: Iterable[int]) -> int:
    n = 0
    for cid in camera_ids:
        if cid in snap and cid in rec.cameras:
            rec.cameras[cid].params = snap[cid]
            n += 1
    return n


def brief(cov: Dict[str, Dict[str, int]]) -> str:
    return ", ".join(f"{g} {c['with_points']}/{c['frames']}" for g, c in sorted(cov.items())) or "-"


# ------------------------------------------------------------------ what changed when a group collapsed
def frame_poses(rec, project) -> Dict[int, Any]:
    """{image_id: (R cam_from_world 3x3, centre)} of the registered Mastcam-Z images."""
    import numpy as np
    out = {}
    for iid in group_of(project):
        if iid not in rec.images:
            continue
        im = rec.images[iid]
        try:
            if not im.has_pose:
                continue
            T = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
            R = np.asarray(T.rotation.matrix(), float)
            out[iid] = (R, -R.T @ np.asarray(T.translation, float))
        except Exception:                                    # noqa: BLE001
            continue
    return out


def _db_pairs(database, image_ids, min_inliers: int = 15) -> Dict[int, int]:
    """{image_id: two-view geometries with >= min_inliers inliers} for ``image_ids`` (sqlite, read only)."""
    import sqlite3
    want = set(int(i) for i in image_ids)
    out = {i: 0 for i in want}
    if not database:
        return out
    con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        for pid, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
            if rows is None or int(rows) < min_inliers:
                continue
            i2 = int(pid) % 2147483647
            i1 = (int(pid) - i2) // 2147483647
            for i in (i1, i2):
                if i in want:
                    out[i] += 1
    finally:
        con.close()
    return out


def diagnose(rec, project, groups: Iterable[str], poses0: Dict[int, Any], cams0: Dict[int, Any], reg0: set,
             database=None) -> Dict[str, Any]:
    """What changed for the frames of ``groups`` between the snapshot (after the adjustment) and now (after the
    triangulation that lost them): registration, poses, cameras, keypoints and the database's pairs."""
    import numpy as np
    gs = set(groups)
    gof = group_of(project)
    ids = [i for i, g in gof.items() if g[:5] in gs]
    poses1 = frame_poses(rec, project)
    reg1 = set(int(f) for f in rec.reg_frame_ids()) if hasattr(rec, "reg_frame_ids") else set()
    rot, cen, n_kp0, n_obs = [], [], 0, 0
    for i in ids:
        if i in rec.images:
            n_kp0 += int(rec.images[i].num_points2D() == 0) if callable(getattr(rec.images[i], "num_points2D", None)) else 0
            n_obs += int(rec.images[i].num_points3D)
        if i in poses0 and i in poses1:
            R0, C0 = poses0[i]
            R1, C1 = poses1[i]
            c = np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1)
            rot.append(float(np.degrees(np.arccos(c))))
            cen.append(float(np.linalg.norm(C1 - C0)))
    dcam = {}
    for cid in camera_ids_of(project, gs):
        if cid in cams0 and cid in rec.cameras:
            d = np.abs(np.asarray(rec.cameras[cid].params, float)[:6] - np.asarray(cams0[cid], float)[:6])
            dcam[cid] = d.tolist()
    dmax = np.max(np.array(list(dcam.values())), axis=0).tolist() if dcam else []
    frames0 = {rec.images[i].frame_id for i in ids if i in rec.images}
    pairs = {}
    try:
        pairs = _db_pairs(database, ids)
    except Exception as e:                                   # noqa: BLE001
        pairs = {"error": f"{type(e).__name__}: {e}"}
    n_pairs = [v for v in pairs.values() if isinstance(v, int)] if isinstance(pairs, dict) else []
    return {"groups": sorted(gs), "images": len(ids), "observations_now": n_obs,
            "images_without_keypoints": n_kp0,
            "frames_registered_before": len(frames0 & reg0) if reg0 else None,
            "frames_registered_now": len(frames0 & reg1) if reg1 else None,
            "pose_change_deg_median": float(np.median(rot)) if rot else None,
            "pose_change_deg_max": float(np.max(rot)) if rot else None,
            "centre_change_m_median": float(np.median(cen)) if cen else None,
            "centre_change_m_max": float(np.max(cen)) if cen else None,
            "images_with_both_poses": len(rot),
            "camera_change_max_fx_fy_cx_cy_k1_k2": dmax,
            "database": str(database) if database else None,
            "db_pairs_per_image_median": float(np.median(n_pairs)) if n_pairs else None,
            "db_images_without_pairs": int(sum(1 for v in n_pairs if v == 0)) if n_pairs else None}


def diagnosis_line(d: Dict[str, Any]) -> str:
    f = lambda v, fmt: ("-" if v is None else format(v, fmt))       # noqa: E731
    cm = d.get("camera_change_max_fx_fy_cx_cy_k1_k2") or []
    return (f"{', '.join(d['groups'])}: {d['images']} images, {d['observations_now']} observations now; registered "
            f"{d.get('frames_registered_before')} -> {d.get('frames_registered_now')}; without keypoints "
            f"{d['images_without_keypoints']}; pose change median {f(d['pose_change_deg_median'], '.4f')} deg "
            f"(max {f(d['pose_change_deg_max'], '.3f')}), centre {f(d['centre_change_m_median'], '.4f')} m "
            f"(max {f(d['centre_change_m_max'], '.3f')}) over {d['images_with_both_poses']} images; camera change max "
            + (" ".join(f"{v:.3g}" for v in cm) if cm else "-")
            + f"; database pairs per image median {f(d['db_pairs_per_image_median'], '.0f')}, images without pairs "
              f"{d.get('db_images_without_pairs')}")


def restore_poses(rec, project, groups: Iterable[str], poses0: Dict[int, Any]) -> int:
    """Put the frames of ``groups``' images back at their snapshot poses (single-image frames only)."""
    import numpy as np
    import pycolmap
    gs = set(groups)
    n = 0
    for iid, g in group_of(project).items():
        if g[:5] not in gs or iid not in poses0 or iid not in rec.images:
            continue
        im = rec.images[iid]
        fr = rec.frames[im.frame_id]
        if len(list(fr.data_ids)) != 1:
            continue
        R, C = poses0[iid]
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(R, float)), -np.asarray(R, float) @ C)
        n += 1
    return n
