r"""
An MSL (Curiosity) Navcam block over several sites and both computer sides (0.99.5).

Christian, 16 Nov 2026: "Let's use a dense block at Yellowknife sol 124 to 302.  align the A and B navcams with a
opencv perspective lens with k1,k2,p1,p2,k3 distortion.  initialize the cameras from the cahvore parameters and let
them refine.  Download the files into the D:/data folder for reuse." and "we will run mask v3 on the yellowknife bay
images.  If they are good enough, I'll clean them and add them to the mask training set for the next run."

The preset ``yellowknife`` (the default) is that block: sols 124-302, every site, both sides.  Curiosity ran on the
A-side computer to sol 200 and on the B side after it (the A side failed on sol 200 at John Klein, where the rover
stayed for months), so the bay holds the A-side Navcams (NLA, NRA) and the B-side ones (NLB, NRB) looking at the
same ground: four cameras, each its own COLMAP camera.

    python -m rpp_studies.msl.block waypoints              the MSL waypoint table into D:\data\msl (rpp.waypoints)
    python -m rpp_studies.msl.block fetch --list           what the archive has for the sols (nothing downloaded)
    python -m rpp_studies.msl.block fetch                  RAD + MXY of every full-frame Navcam -> D:\data\msl\navcam
    python -m rpp_studies.msl.block prepare                images, masks (mask CNN + MXY), cameras, pairs, manifest
    python -m rpp_studies.msl.block align                  COLMAP, the label frame, the cameras refined
    python -m rpp_studies.msl.block all                    the four in turn

Steps
-----
* **positions** - every frame's label pose is in its own site frame.  The sites are joined by the label's
  ``SITE_COORD_SYSTEM_PARMS`` (each site's origin in the previous site's frame, the rover's own estimate) into the
  frame of the site with the most frames (the "block frame", east-north-up).  Where a site of the chain has no frame
  of its own, the waypoint table bridges it when it has site and drive numbers; the report compares the chain with
  the waypoint table wherever both exist.
* **cameras** - one per camera (NLA, NRA, NLB, NRB): COLMAP's ``FULL_OPENCV`` with k4 = k5 = k6 = 0, i.e. the OpenCV
  perspective lens with k1, k2, p1, p2, k3, fitted to each frame's label CAHVOR (:func:`rpp.calib.labelfit.
  opencv5_from_cahvore`; about 0.02 px rms on the Navcam) and the median taken.  ``--camera fisheye`` keeps the
  first block's THIN_PRISM_FISHEYE.
* **masks** - the mask CNN (``--mask-model``, default the released ``rpp_mask_v3``) on every frame, and MIPL's
  rover mask (MXY) where the archive has one.  COLMAP's mask leaves out what either calls rover (``--mask both``;
  ``cnn`` or ``mxy`` for one).  For the training set the CNN's masks are written beside the images in the layout of
  a mask training set (``mask_candidates\images\<stem>.png`` RGBA, ``masks\<stem>.png`` 255 = terrain), with the MXY
  mask, an overlay and ``review.csv`` (frames where the CNN and MXY disagree most first) to clean them by.
* **pairs** - an image pair is matched when the two cameras are within ``--pair-distance-m`` and their views meet on
  the ground: the point where each optical axis meets the ground 1.9 m below the camera (at most 40 m out) lies
  inside the other image, or the middle of the two such points lies inside both.  Both eyes of a stereo pair always.  At most ``--pairs-per-image`` per image (nearest
  first).  ``pairs.txt``.
* **align** - SIFT with the masks; the pairs matched; the incremental mapper with the label cameras held; the model
  put into the block frame by a similarity onto the label camera centers (RANSAC, ``--max-error-m``); then the
  bundle adjustment with the cameras free (focal lengths, principal points, k1, k2, p1, p2, k3; k4-k6 held at zero),
  twice, with the observations above 4 px dropped in between.  ``report.txt``: frames registered per camera,
  reprojection error, each camera's refined parameters against its CAHVOR start, the stereo baselines against the
  labels', the label-center residuals per site.

Data go to ``D:\data\msl`` (``navcam\SOLnnnnn``, ``MSL_waypoints.json``) and stay there for later blocks; the block
to ``D:\rover_blocks\msl_sol_<first sol, 4 digits>_<block name>`` (0.99.12, Christian, 17 Nov 2026: "put the msl
blocks in the rover_blocks folder. Use the naming convention D:/rover_blocks/msl_sol_0{sol start}_{block name}", as the
Mars 2020 blocks' ``mars2020_sol_<sol>_<site>``; the Mars 2020 code reads only ``mars2020_sol_*`` folders), e.g.
``msl_sol_0124_yellowknife_bay``; ``--sols 162 296`` makes ``msl_sol_0162_yellowknife_bay``.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import first_block as F

log = logging.getLogger("rpp.msl")

DEFAULT_DATA = r"D:\data\msl"
DEFAULT_ROOT = r"D:\rover_blocks"            # 0.99.12: with the Mars 2020 blocks (RPP_SCAPES); was D:\rover_blocks_msl
FOLDER_PREFIX = "msl_sol_"
NAVCAM_HEIGHT_M = 1.9                  # the Navcam above the ground (label C at about -1.9 m in the rover frame)
LOOK_MAX_M = 40.0
PRESETS: Dict[str, Dict[str, Any]] = {
    "yellowknife": {"sols": (124, 302), "side": "AB", "families": "N", "sites": None, "block": "yellowknife_bay",
                    "radius_m": 250.0,
                    "note": "Yellowknife Bay, sols 124-302: A-side Navcams to sol 199, B-side from sol 200"},
    "site87": {"sols": (3047, 3056), "side": "B", "families": "N", "sites": [87], "block": "mont_mercou",
               "radius_m": 50.0, "note": "the first block (Mont Mercou approach), B side"},
}


def blocks_root() -> Path:
    """The folder of the MSL blocks: the Mars 2020 blocks' root (``RPP_SCAPES``, else ``D:\\rover_blocks``)."""
    from rpp import env as rpp_env
    return Path(rpp_env.get("RPP_SCAPES") or DEFAULT_ROOT)


def block_folder_name(first_sol: int, block: str) -> str:
    """``msl_sol_<first sol, 4 digits>_<block>`` (0.99.12): ``block_folder_name(124, "yellowknife_bay")`` is
    ``msl_sol_0124_yellowknife_bay``."""
    return f"{FOLDER_PREFIX}{int(first_sol):04d}_{str(block).strip().lower().replace(' ', '_')}"


# ------------------------------------------------------------------------------------------------- waypoints
def waypoints(data: Path, refresh: Any = "auto") -> Dict[str, Any]:
    """The MSL waypoint table (``rpp.waypoints``, mission "msl"), kept as ``<data>\\MSL_waypoints.json``."""
    from rpp.waypoints.table import load_waypoints
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    return load_waypoints(path=data / "MSL_waypoints.json", refresh=refresh, mission="msl")


def waypoint_summary(wp: Dict[str, Any], sols: Tuple[int, int]) -> Dict[str, Any]:
    from rpp.waypoints.table import local_en
    rows = [r for r in local_en(wp) if r["sol"] is not None and sols[0] <= r["sol"] <= sols[1]]
    if not rows:
        return {"waypoints": 0}
    E, N = np.array([r["E"] for r in rows]), np.array([r["N"] for r in rows])
    sites = sorted({r["site"] for r in rows if r["site"] is not None})
    return {"waypoints": len(rows), "sites": sites, "has_site_drive": bool(sites),
            "extent_m": [round(float(E.max() - E.min()), 1), round(float(N.max() - N.min()), 1)],
            "first": {k: rows[0][k] for k in ("sol", "site", "drive", "lon", "lat")},
            "last": {k: rows[-1][k] for k in ("sol", "site", "drive", "lon", "lat")}}


# ------------------------------------------------------------------------------------------------- positions
def _enu(ned: Sequence[float]) -> np.ndarray:
    n, e, d = (float(x) for x in ned)
    return np.array([e, n, -d])


def site_origins(infos: List[Dict[str, Any]], anchor: Optional[int] = None,
                 wp: Optional[Dict[str, Any]] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """{site: its origin in the block frame (ENU, m)} from the label links (site -> parent), and a report.  The block
    frame is the frame of ``anchor`` (default: the site with the most frames).  Links are taken as the median over
    the frames that carry them (they are one number per site).  A site that cannot be chained to the anchor is
    placed by the waypoint table when it has site / drive numbers, else left out (reported)."""
    links: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    count: Dict[int, int] = {}
    for r in infos:
        count[r["site"]] = count.get(r["site"], 0) + 1
        lk = r.get("site_link")
        if lk and int(lk["site"]) == int(r["site"]):
            q = lk.get("quaternion") or [1, 0, 0, 0]
            if abs(abs(float(q[0])) - 1.0) > 1e-6:
                log.warning("site %d: its frame is rotated against site %d (q %s) - not handled", lk["site"], lk["parent"], q)
            links.setdefault(int(lk["site"]), []).append((int(lk["parent"]), _enu(lk["offset_ned_m"])))
    link = {}
    for s, v in links.items():
        parents = {p for p, _ in v}
        p = max(parents, key=lambda x: sum(1 for pp, _ in v if pp == x))
        link[s] = (p, np.median(np.array([o for pp, o in v if pp == p]), axis=0))
    if not count:
        return {}, {"anchor": None}
    anchor = int(anchor) if anchor is not None else max(count, key=lambda s: (count[s], -s))
    origin: Dict[int, np.ndarray] = {anchor: np.zeros(3)}
    # walk down (site -> parent) and up (child -> site) from the anchor over consecutive links
    changed = True
    while changed:
        changed = False
        for s, (p, off) in link.items():
            if p in origin and s not in origin:
                origin[s] = origin[p] + off
                changed = True
            elif s in origin and p not in origin:
                origin[p] = origin[s] - off
                changed = True
    rep: Dict[str, Any] = {"anchor": anchor, "frames_per_site": {str(k): v for k, v in sorted(count.items())},
                           "chained": sorted(int(s) for s in origin), "by_waypoints": [], "left_out": []}
    missing = sorted(s for s in count if s not in origin)
    if missing and wp is not None:
        by_wp = _origins_from_waypoints(infos, wp)
        common = [s for s in origin if s in by_wp]
        if common:
            shift = np.median(np.array([origin[s] - by_wp[s] for s in common]), axis=0)
            for s in missing:
                if s in by_wp:
                    origin[s] = by_wp[s] + shift
                    rep["by_waypoints"].append(int(s))
    rep["left_out"] = sorted(int(s) for s in count if s not in origin)
    if wp is not None:
        rep["chain_vs_waypoints_m"] = _chain_check(origin, infos, wp)
    rep["origins_enu_m"] = {str(k): [round(float(x), 3) for x in v] for k, v in sorted(origin.items())}
    return origin, rep


def _origins_from_waypoints(infos: List[Dict[str, Any]], wp: Dict[str, Any]) -> Dict[int, np.ndarray]:
    """Each site's origin in the waypoint table's local frame (E, N; height 0): the waypoint of a (site, drive)
    minus the rover's position in its site frame at that drive (the frames' label rover offsets, median)."""
    from rpp.waypoints.table import local_en
    rows = {(r["site"], r["drive"]): r for r in local_en(wp) if r["site"] is not None and r["drive"] is not None}
    rover: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for r in infos:
        if r.get("rover_enu_m") is not None:
            rover.setdefault((r["site"], r["drive"]), []).append(np.asarray(r["rover_enu_m"], float))
    est: Dict[int, List[np.ndarray]] = {}
    for sd, v in rover.items():
        w = rows.get(sd)
        if w is None:
            continue
        est.setdefault(sd[0], []).append(np.array([w["E"], w["N"], 0.0]) - np.r_[np.median(np.array(v), axis=0)[:2], 0.0])
    return {s: np.median(np.array(v), axis=0) for s, v in est.items()}


def _chain_check(origin: Dict[int, np.ndarray], infos: List[Dict[str, Any]], wp: Dict[str, Any]) -> Dict[str, Any]:
    by_wp = _origins_from_waypoints(infos, wp)
    common = [s for s in origin if s in by_wp]
    if len(common) < 2:
        return {"sites": len(common)}
    d = np.array([(origin[s] - by_wp[s])[:2] for s in common])
    d = d - np.median(d, axis=0)
    e = np.linalg.norm(d, axis=1)
    return {"sites": len(common), "median": round(float(np.median(e)), 2), "max": round(float(e.max()), 2),
            "worst": int(common[int(np.argmax(e))])}


def place(infos: List[Dict[str, Any]], origin: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
    """Frames with ``C_block_m`` (camera center in the block frame); frames of sites without an origin are dropped."""
    out = []
    for r in infos:
        o = origin.get(int(r["site"]))
        if o is None:
            continue
        r = dict(r)
        r["C_block_m"] = [float(x) for x in np.asarray(r["C_enu_m"], float) + o]
        out.append(r)
    return out


def select(frames: List[Dict[str, Any]], radius_m: Optional[float], anchor_site: Optional[int] = None
           ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The frames within ``radius_m`` (horizontal) of the middle of the block: the median camera position of the
    anchor site's frames (``anchor_site``; else of all frames).  None: all."""
    if not frames:
        return [], {"frames": 0}
    C = np.array([r["C_block_m"] for r in frames])
    mid = [i for i, r in enumerate(frames) if anchor_site is not None and int(r["site"]) == int(anchor_site)]
    c = np.median(C[mid] if mid else C, axis=0)
    d = np.linalg.norm((C - c)[:, :2], axis=1)
    keep = np.ones(len(frames), bool) if not radius_m else d <= float(radius_m)
    if not keep.any():
        raise ValueError(f"no frame within {radius_m} m of the block's middle {np.round(c, 1).tolist()}")
    kept = [r for r, k in zip(frames, keep) if k]
    st = {(r["site"], r["drive"]) for r in kept}
    return kept, {"frames": len(kept), "dropped": int((~keep).sum()), "center_m": [round(float(x), 2) for x in c],
                  "radius_m": radius_m, "stations": len(st),
                  "by_camera": {g: sum(1 for r in kept if r["group"] == g) for g in sorted({r["group"] for r in kept})},
                  "sols": [min(r["sol"] for r in kept), max(r["sol"] for r in kept)]}


# ------------------------------------------------------------------------------------------------- cameras
def group_camera(infos: List[Dict[str, Any]], kind: str = "opencv5", step: float = 16.0) -> Dict[str, Any]:
    """The COLMAP camera of a camera group (NLA, NRB ...) from its frames' label models: ``opencv5`` (FULL_OPENCV with
    k4-k6 = 0) or ``fisheye`` (THIN_PRISM_FISHEYE, as the first block).  Median over the distinct label models."""
    if kind == "fisheye":
        return F.group_camera(infos, step=step)
    from rpp.calib.labelfit import opencv5_from_cahvore
    from rpp.cmod import CameraModel
    ps, rms, seen = [], [], set()
    for r in infos:
        key = json.dumps(r["gcm"], sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        cm = CameraModel.from_label(r["gcm"], r["width"], r["height"])
        mc, _ = cm.camera_frame()
        p, rep = opencv5_from_cahvore(mc, r["width"], r["height"], step=step)
        ps.append(p)
        rms.append(rep["rms_px"])
    P = np.median(np.asarray(ps, float), axis=0)
    spread = np.asarray(ps, float).std(axis=0) if len(ps) > 1 else np.zeros(12)
    return {"model": "FULL_OPENCV", "params": [float(x) for x in P], "params_names": ["fx", "fy", "cx", "cy", "k1", "k2",
            "p1", "p2", "k3", "k4", "k5", "k6"], "held_zero": ["k4", "k5", "k6"], "width": infos[0]["width"],
            "height": infos[0]["height"], "fit_rms_px_max": float(max(rms)), "label_models": len(ps),
            "label_spread": [float(x) for x in spread], "frames": len(infos),
            "source": "label CAHVOR fit (rpp.calib.labelfit.opencv5_from_cahvore), median"}


# ------------------------------------------------------------------------------------------------- pairs
def _axis_and_look(r: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(r["R_w2c"], float)
    C = np.asarray(r["C_block_m"], float)
    a = R[2]                                   # the optical axis in the block frame
    t = LOOK_MAX_M
    if a[2] < -1e-3:
        t = min(t, NAVCAM_HEIGHT_M / -a[2])
    return a, C + t * a


def _sees(r: Dict[str, Any], cam: Dict[str, Any], X: np.ndarray, margin: float = 0.1) -> bool:
    from rpp.calib.labelfit import project_full_opencv, project_thin_prism_fisheye
    R = np.asarray(r["R_w2c"], float)
    Xc = R @ (np.asarray(X, float) - np.asarray(r["C_block_m"], float))
    if Xc[2] <= 0:
        return False
    proj = project_full_opencv if cam["model"] == "FULL_OPENCV" else project_thin_prism_fisheye
    u = proj(cam["params"], Xc[None])[0]
    w, h = cam["width"], cam["height"]
    return bool(np.all(np.isfinite(u)) and -margin * w <= u[0] <= (1 + margin) * w and -margin * h <= u[1] <= (1 + margin) * h)


def image_pairs(frames: List[Dict[str, Any]], cameras: Dict[str, Dict[str, Any]], max_distance_m: float = 60.0,
                per_image: int = 40) -> List[Tuple[str, str]]:
    """The pairs to match (names); see the module's text."""
    n = len(frames)
    C = np.array([r["C_block_m"] for r in frames])
    looks = [_axis_and_look(r) for r in frames]
    L = np.array([lk[1] for lk in looks])
    cand: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n)}
    for i in range(n):
        d = np.linalg.norm(C - C[i], axis=1)
        for j in np.nonzero((d <= max_distance_m) & (np.arange(n) > i))[0]:
            j = int(j)
            stereo = frames[i]["pair_key"] == frames[j]["pair_key"] and frames[i]["eye"] != frames[j]["eye"]
            ci, cj = cameras[frames[i]["group"]], cameras[frames[j]["group"]]
            M = 0.5 * (L[i] + L[j])
            if not stereo and not (_sees(frames[i], ci, L[j]) or _sees(frames[j], cj, L[i])
                                   or (_sees(frames[i], ci, M) and _sees(frames[j], cj, M))):
                continue
            score = -1.0 if stereo else float(np.linalg.norm(L[i] - L[j]))
            cand[i].append((score, j))
            cand[j].append((score, i))
    keep = set()
    for i, v in cand.items():
        for _, j in sorted(v)[:per_image]:
            keep.add((min(i, j), max(i, j)))
    return [(frames[i]["image"], frames[j]["image"]) for i, j in sorted(keep)]


# ------------------------------------------------------------------------------------------------- prepare
def scan(data: Path, sols: Sequence[int], side: Optional[str], families: str, sites: Optional[Sequence[int]] = None,
         cache: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Path]]:
    """Every full-frame RAD of the sols under ``data`` with its label information (cached by file in ``cache``)."""
    rad, mxy = F.find_products(data, None, side, families, sols=sols)
    old = {}
    if cache is not None and Path(cache).is_file():
        try:
            old = json.loads(Path(cache).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
    infos, new = [], {}
    for p in rad:
        f = F.parse_name(p.name)
        if sites and f["site"] not in set(sites):
            continue
        key = f"{p}|{p.stat().st_size}|{int(p.stat().st_mtime)}"
        r = old.get(key)
        if r is None:
            try:
                r = F.product_info(p)
            except Exception as e:                                     # noqa: BLE001
                log.warning("%s: left out (%s)", p.name, e)
                continue
            r["path"] = str(p)
            r["rover_enu_m"] = _rover_enu(p)
        new[key] = r
        infos.append(r)
    if cache is not None:
        Path(cache).write_text(json.dumps(new, default=float), encoding="utf-8")
    return infos, mxy


def _rover_enu(img: Path) -> Optional[List[float]]:
    from rpp.labels import read_pds
    try:
        L, _ = read_pds(img, load_image=False)
        rk = next(k for k in L if str(k).upper() in ("ROVER_COORD_SYSTEM_PARMS", "ROVER_COORDINATE_SYSTEM_PARMS"))
        return [float(x) for x in _enu(L[rk]["ORIGIN_OFFSET_VECTOR"])]
    except Exception:                                                  # noqa: BLE001
        return None


def write_masks(work: Path, frames: List[Dict[str, Any]], mxy: Dict[str, Path], mask_mode: str = "both",
                model: Optional[str] = "rpp_mask_v3", candidates: bool = True) -> Dict[str, Any]:
    """Images, COLMAP masks and the training candidates (see the module's text)."""
    import cv2
    from rpp.labels import read_pds
    for d in ("images", "masks"):
        (work / d).mkdir(parents=True, exist_ok=True)
    cand = work / "mask_candidates"
    if candidates and mask_mode != "mxy":
        for d in ("images", "masks", "mxy", "overlays"):
            (cand / d).mkdir(parents=True, exist_ok=True)
    infer = None
    if mask_mode in ("cnn", "both") and model:
        from rpp.mask.infer import infer_mask
        from rpp.sfm.gpu import gpu_lock
        infer = infer_mask
    rows = []
    t0 = time.time()
    for k, r in enumerate(frames):
        name = r["stem"] + ".png"
        r["image"] = name
        _, a = read_pds(r["path"])
        img8 = F.stretch8(np.squeeze(a))
        cv2.imwrite(str(work / "images" / name), img8)
        rover = np.zeros(img8.shape, bool)
        row: Dict[str, Any] = {"stem": r["stem"], "camera": r["group"], "sol": r["sol"], "site": r["site"], "drive": r["drive"]}
        m = mxy.get(r["stem"])
        mx = None
        if m is not None and mask_mode in ("mxy", "both"):
            _, mk = read_pds(m)
            mx = np.squeeze(mk) == F.MASK_ROVER
            rover |= mx
            row["mxy_rover_pct"] = round(100 * float(mx.mean()), 2)
        if infer is not None:
            with gpu_lock("msl mask", enabled=_cuda()):
                cm, _, _ = infer(np.dstack([img8] * 3).astype(np.float32), model)
            cx = cm == 0
            rover |= cx
            row["cnn_excluded_pct"] = round(100 * float(cx.mean()), 2)
            if mx is not None:
                u = float((cx | mx).sum())
                row["iou"] = round(float((cx & mx).sum()) / u, 3) if u > 0 else 1.0
                row["cnn_not_mxy_pct"] = round(100 * float((cx & ~mx).mean()), 2)
                row["mxy_not_cnn_pct"] = round(100 * float((mx & ~cx).mean()), 2)
            if candidates:
                rgba = np.dstack([img8, img8, img8, np.full(img8.shape, 255, np.uint8)])
                cv2.imwrite(str(cand / "images" / name), rgba)
                cv2.imwrite(str(cand / "masks" / name), np.where(cx, 0, 255).astype(np.uint8))
                if mx is not None:
                    cv2.imwrite(str(cand / "mxy" / name), np.where(mx, 0, 255).astype(np.uint8))
                ov = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
                red = ov.copy()
                red[cx] = (0, 0, 255)
                ov = cv2.addWeighted(ov, 0.65, red, 0.35, 0)
                if mx is not None:
                    cs, _ = cv2.findContours(mx.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(ov, cs, -1, (0, 255, 0), 1)
                cv2.imwrite(str(cand / "overlays" / (r["stem"] + ".jpg")), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(work / "masks" / (name + ".png")), np.where(rover, 0, 255).astype(np.uint8))
        r["mask_rover_pct"] = round(100 * float(rover.mean()), 2)
        r["mask"] = mask_mode if (infer is not None or mx is not None) else None
        rows.append(row)
        if k % 200 == 199:
            log.info("masks: %d of %d (%.0f s)", k + 1, len(frames), time.time() - t0)
    if candidates and infer is not None and rows:
        def order(x):
            return (-(1 - x["iou"]) if "iou" in x else 0.0, -x.get("cnn_excluded_pct", 0.0))
        fields = ["stem", "camera", "sol", "site", "drive", "cnn_excluded_pct", "mxy_rover_pct", "iou", "cnn_not_mxy_pct",
                  "mxy_not_cnn_pct"]
        with open(cand / "review.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for x in sorted(rows, key=order):
                w.writerow(x)
        (cand / "README.txt").write_text(
            "Mask candidates for the training set (rpp_studies.msl.block, 0.99.5).\n"
            f"model: {model}; images: MSL Navcam RAD, percentile stretch {F.STRETCH} to 8 bit, one band in R, G and B "
            "(the network's input at inference), alpha 255.\n"
            "masks/: the CNN's mask, 255 = terrain, 0 = rover - to clean (e.g. in Metashape, as the M2020 set).\n"
            "mxy/: MIPL's rover mask (geometric: the pixel's stereo XYZ inside the rover volume; blocky, no shadows).\n"
            "overlays/: red = CNN rover, green = MXY outline.  review.csv: the frames where the two disagree most "
            "first.\n", encoding="utf-8")
    ious = [x["iou"] for x in rows if "iou" in x]
    return {"frames": len(rows), "cnn": infer is not None, "with_mxy": sum(1 for x in rows if "mxy_rover_pct" in x),
            "iou_median": float(np.median(ious)) if ious else None, "seconds": round(time.time() - t0, 1)}


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                                  # noqa: BLE001
        return False


def prepare(work: Path, data: Path, preset: Dict[str, Any], camera: str = "opencv5", mask_mode: str = "both",
            mask_model: Optional[str] = "rpp_mask_v3", radius_m: Optional[float] = None, pair_distance_m: float = 60.0,
            pairs_per_image: int = 40, wp: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    sols = list(range(int(preset["sols"][0]), int(preset["sols"][1]) + 1))
    infos, mxy = scan(Path(data), sols, preset["side"], preset["families"], preset.get("sites"), cache=work / "labels_cache.json")
    if not infos:
        raise ValueError(f"no full-frame Navcam RAD of sols {sols[0]}-{sols[-1]} under {data} - run the fetch first")
    origin, site_rep = site_origins(infos, wp=wp)
    frames = place(infos, origin)
    frames, sel = select(frames, radius_m if radius_m is not None else preset.get("radius_m"), site_rep.get("anchor"))
    if limit:
        frames = sorted(frames, key=lambda r: (r["sol"], r["sclk"]))[:int(limit)]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in frames:
        groups.setdefault(r["group"], []).append(r)
    cameras = {g: group_camera(v, camera) for g, v in sorted(groups.items())}
    mrep = write_masks(work, frames, mxy, mask_mode, mask_model)
    pairs = image_pairs(frames, cameras, pair_distance_m, pairs_per_image)
    (work / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
    keep = ("stem", "image", "group", "family", "eye", "side", "sol", "lmst_h", "sclk", "site", "drive", "sequence",
            "pair_key", "C_enu_m", "C_block_m", "R_w2c", "ccd_degC", "radiance_scale", "mask", "mask_rover_pct", "width",
            "height", "path")
    man = {"mission": "MSL", "preset": preset, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "data": str(data), "camera_kind": camera, "mask_mode": mask_mode, "mask_model": mask_model, "sites": site_rep,
           "selection": sel, "masks": mrep, "pairs": len(pairs), "cameras": cameras,
           "images": [{k: r.get(k) for k in keep} for r in frames]}
    (work / "cameras.json").write_text(json.dumps(cameras, indent=1), encoding="utf-8")
    (work / "manifest.json").write_text(json.dumps(man, indent=1, default=float), encoding="utf-8")
    log.info("prepare: %d frames (%s), %d stations, sites %s (anchor %s), %d pairs; cameras %s", len(frames),
             sel.get("by_camera"), sel.get("stations"), site_rep.get("chained"), site_rep.get("anchor"), len(pairs),
             ", ".join(f"{g} ({c['frames']} frames, fit {c['fit_rms_px_max']:.3f} px)" for g, c in cameras.items()))
    return man


# ------------------------------------------------------------------------------------------------- align
def refine_cameras(rec, man: Dict[str, Any], rounds: int = 2, max_error_px: float = 4.0) -> List[Dict[str, Any]]:
    """Bundle adjustment with every camera free (fx fy cx cy k1 k2 p1 p2 k3) and k4-k6 held at zero (a subset
    manifold on the camera block); two images' poses held for the gauge.  Between rounds the observations above
    ``max_error_px`` are dropped."""
    import pycolmap
    import pyceres
    out = []
    for k in range(int(rounds)):
        opts = pycolmap.BundleAdjustmentOptions()
        opts.refine_focal_length = opts.refine_principal_point = opts.refine_extra_params = True
        cfg = pycolmap.BundleAdjustmentConfig()
        ids = sorted(rec.reg_image_ids())
        for i in ids:
            cfg.add_image(i)
        far = max(ids[1:], key=lambda j: np.linalg.norm(rec.image(j).projection_center() - rec.image(ids[0]).projection_center()))
        cfg.set_constant_rig_from_world_pose(rec.image(ids[0]).frame_id)
        cfg.set_constant_rig_from_world_pose(rec.image(far).frame_id)
        ba = pycolmap.create_default_ceres_bundle_adjuster(opts, cfg, rec)
        held = []
        for cid in list(rec.cameras):
            cam = rec.camera(cid)                      # the camera in the reconstruction (its params are the block)
            if cam.model_name == "FULL_OPENCV":
                pa = cam.params
                if ba.problem.has_parameter_block(pa):
                    m = pyceres.SubsetManifold(12, [9, 10, 11])
                    ba.problem.set_manifold(pa, m)
                    held.append((pa, m))           # the problem does not own the manifold: kept alive to the solve
        summ = ba.solve()
        del held
        n_drop = 0
        if k < rounds - 1:
            n_drop = int(pycolmap.ObservationManager(rec).filter_all_points3D(float(max_error_px), 1.5))
        out.append({"round": k + 1, "reprojection_px": float(rec.compute_mean_reprojection_error()), "dropped": n_drop,
                    "cost": [float(getattr(summ, "initial_cost", float("nan"))), float(getattr(summ, "final_cost", float("nan")))]})
    return out


def align(work: Path, max_features: int = 8192, threads: int = -1, max_error_m: float = 2.0, refine: bool = True,
          rounds: int = 2) -> Dict[str, Any]:
    import pycolmap
    work = Path(work)
    man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    db = work / "database.db"
    if db.exists():
        db.unlink()
    t0 = time.time()
    for g, cam in man["cameras"].items():
        names = [r["image"] for r in man["images"] if r["group"] == g]
        ro = pycolmap.ImageReaderOptions(camera_model=cam["model"], camera_params=",".join(f"{x:.10g}" for x in cam["params"]),
                                         mask_path=str(work / "masks"))
        eo = pycolmap.FeatureExtractionOptions()
        eo.sift.max_num_features = int(max_features)
        eo.num_threads = int(threads)
        pycolmap.extract_features(db, work / "images", image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=eo)
    log.info("features: %.1f min", (time.time() - t0) / 60)
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(work / "pairs.txt")
    pycolmap.match_image_pairs(db, pairing_options=po)
    log.info("matched %d pairs: %.1f min", man.get("pairs", 0), (time.time() - t0) / 60)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False   # the labels'
    opts.num_threads = int(threads)
    out = work / "sparse"
    out.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db, work / "images", out, options=opts)
    if not recs:
        rep = {"ok": False, "message": "COLMAP registered no model", "frames": len(man["images"])}
        (work / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        return rep
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    others = sorted((int(r.num_reg_images()) for r in recs.values()), reverse=True)[1:6]
    return finish(rec, man, work, len(recs), others, max_error_m, refine, rounds, t0)


def finish(rec, man: Dict[str, Any], work: Path, models: int = 1, others: Sequence[int] = (), max_error_m: float = 2.0,
           refine: bool = True, rounds: int = 2, t0: Optional[float] = None) -> Dict[str, Any]:
    """The mapped model into the block frame, the cameras refined, the report and exports (``align``'s second half)."""
    work = Path(work)
    t0 = time.time() if t0 is None else t0
    out = work / "sparse"
    out.mkdir(exist_ok=True)
    lab = {r["image"]: np.asarray(r["C_block_m"], float) for r in man["images"]}
    rep: Dict[str, Any] = {"ok": True, "frames": len(man["images"]), "registered": int(rec.num_reg_images()),
                           "models": int(models), "other_models": list(others),
                           "reprojection_px_mapper": float(rec.compute_mean_reprojection_error())}
    _to_block(rec, lab, max_error_m, rep, "similarity")
    start = {cid: np.array(c.params, float) for cid, c in rec.cameras.items()}
    if refine:
        rep["refine"] = refine_cameras(rec, man, rounds=rounds)
        _to_block(rec, lab, max_error_m, rep, "similarity_after_refine")
    rep["points"] = int(rec.num_points3D())
    rep["reprojection_px"] = float(rec.compute_mean_reprojection_error())
    rep["minutes"] = round((time.time() - t0) / 60.0, 1)
    gname = {}
    for im in rec.images.values():
        gname.setdefault(im.camera_id, next((r["group"] for r in man["images"] if r["image"] == im.name), "?"))
    rep["cameras"] = {}
    for cid, c in rec.cameras.items():
        g = gname.get(cid, str(cid))
        p0, p1 = start[cid], np.array(c.params, float)
        rep["cameras"][g] = {"registered": sum(1 for im in rec.images.values() if im.camera_id == cid),
                             "start": [float(x) for x in p0], "refined": [float(x) for x in p1],
                             "change": [float(x) for x in p1 - p0]}
    centers = {im.name: np.asarray(im.projection_center(), float) for im in rec.images.values()}
    sb = F.stereo_baselines([dict(r, C_enu_m=r["C_block_m"]) for r in man["images"]], centers)
    if sb:
        rat = np.array([s["ratio"] for s in sb if s["ratio"]])
        rep["stereo_baseline"] = {"pairs": len(sb), "model_m_median": float(np.median([s["model_m"] for s in sb])),
                                  "label_m_median": float(np.median([s["label_m"] for s in sb])),
                                  "ratio_median": float(np.median(rat)), "ratio_sd": float(np.std(rat))}
    per_site: Dict[int, List[float]] = {}
    for r in man["images"]:
        if r["image"] in centers:
            per_site.setdefault(int(r["site"]), []).append(float(np.linalg.norm(centers[r["image"]] - lab[r["image"]])))
    rep["label_residual_by_site_m"] = {str(s): {"frames": len(v), "median": round(float(np.median(v)), 3),
                                                "max": round(float(np.max(v)), 3)} for s, v in sorted(per_site.items())}
    (out / "block_frame").mkdir(exist_ok=True)
    rec.write(out / "block_frame")
    rec.export_PLY(str(work / "points_block_frame.ply"))
    (work / "cameras_refined.json").write_text(json.dumps(rep["cameras"], indent=1), encoding="utf-8")
    (work / "report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    (work / "report.txt").write_text(report_text(man, rep), encoding="utf-8")
    log.info("\n%s", report_text(man, rep))
    return rep


def _to_block(rec, lab: Dict[str, np.ndarray], max_error_m: float, rep: Dict[str, Any], key: str) -> None:
    import pycolmap
    names = [im.name for im in rec.images.values() if im.name in lab]
    ro = pycolmap.RANSACOptions()
    ro.max_error = float(max_error_m)
    sim = pycolmap.align_reconstruction_to_locations(rec, names, np.array([lab[n] for n in names]), 3, ro)
    if sim is None:
        rep[key] = None
        rep["message"] = "the model could not be put into the block frame (too few frames with agreeing label positions)"
        return
    rec.transform(sim)
    res = [float(np.linalg.norm(np.asarray(rec.images[i].projection_center()) - lab[rec.images[i].name]))
           for i in rec.images if rec.images[i].name in lab]
    rep[key] = {"scale": float(sim.scale), "median_m": float(np.median(res)), "p95_m": float(np.percentile(res, 95)),
                "max_m": float(np.max(res))}


def report_text(man: Dict[str, Any], rep: Dict[str, Any]) -> str:
    pre = man.get("preset") or {}
    sel = man.get("selection") or {}
    sr = man.get("sites") or {}
    lines = [f"MSL block {pre.get('name')}: {pre.get('note', '')}",
             f"frames {rep.get('frames')} ({sel.get('by_camera')}), stations {sel.get('stations')}, sols {sel.get('sols')}; "
             f"sites {sr.get('chained')} chained to site {sr.get('anchor')}"
             + (f", {sr.get('by_waypoints')} by the waypoints" if sr.get("by_waypoints") else "")
             + (f", left out {sr.get('left_out')}" if sr.get("left_out") else "")]
    if sr.get("chain_vs_waypoints_m", {}).get("sites", 0) >= 2:
        c = sr["chain_vs_waypoints_m"]
        lines.append(f"label site chain vs waypoint table: {c['sites']} sites, median {c['median']} m, max {c['max']} m "
                     f"(site {c['worst']})")
    m = man.get("masks") or {}
    lines.append(f"masks: {man.get('mask_mode')} (model {man.get('mask_model')}), {m.get('with_mxy')} frames with MXY, "
                 f"CNN-MXY IoU median {m.get('iou_median')}")
    lines.append(f"registered {rep.get('registered')} of {rep.get('frames')} in the largest model ({rep.get('models')} models; "
                 f"next {rep.get('other_models')}), points {rep.get('points')}, reprojection {rep.get('reprojection_px', float('nan')):.3f} px")
    for k in ("similarity", "similarity_after_refine"):
        s = rep.get(k)
        if s:
            lines.append(f"{k}: scale {s['scale']:.4f}, label centers off by median {s['median_m']:.3f} m, 95 % {s['p95_m']:.3f}, "
                         f"max {s['max_m']:.3f}")
    for r in rep.get("refine") or []:
        lines.append(f"refine round {r['round']}: reprojection {r['reprojection_px']:.3f} px, dropped {r['dropped']}")
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    for g, c in sorted((rep.get("cameras") or {}).items()):
        ch = ", ".join(f"{n} {c['refined'][i]:.6g} ({c['change'][i]:+.3g})" for i, n in enumerate(names) if i < len(c["refined"]))
        lines.append(f"  {g}: {c['registered']} frames; {ch}")
    if rep.get("stereo_baseline"):
        s = rep["stereo_baseline"]
        lines.append(f"stereo baseline: {s['pairs']} pairs, model {1000 * s['model_m_median']:.1f} mm vs label "
                     f"{1000 * s['label_m_median']:.1f} mm (ratio {s['ratio_median']:.4f} +- {s['ratio_sd']:.4f})")
    for s, v in (rep.get("label_residual_by_site_m") or {}).items():
        lines.append(f"  site {s}: {v['frames']} frames, label centers off by median {v['median']} m, max {v['max']} m")
    if rep.get("message"):
        lines.append(rep["message"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("waypoints", "fetch", "prepare", "align", "all", "where"),
                    help="where: print the block's folder and stop")
    ap.add_argument("--preset", default="yellowknife", choices=sorted(PRESETS))
    ap.add_argument("--data", default=DEFAULT_DATA, help=f"the MSL archive on disk (default {DEFAULT_DATA})")
    ap.add_argument("--work", default=None, help=f"the block's folder (default {DEFAULT_ROOT}\\msl_sol_<first sol>_<block>)")
    ap.add_argument("--block", default=None, help="the block's name in its folder (default the preset's: yellowknife_bay)")
    ap.add_argument("--sols", nargs=2, type=int, default=None, help="FIRST LAST (default: the preset's)")
    ap.add_argument("--side", default=None, help="A, B or AB (default: the preset's)")
    ap.add_argument("--list", action="store_true", help="fetch: list what would come, download nothing")
    ap.add_argument("--camera", default="opencv5", choices=("opencv5", "fisheye"))
    ap.add_argument("--mask", default="both", choices=("both", "cnn", "mxy"), help="COLMAP's mask (default both)")
    ap.add_argument("--mask-model", default="rpp_mask_v3", help="the mask CNN (registry name or file)")
    ap.add_argument("--radius-m", type=float, default=None, help="prepare: frames within this of the block's middle")
    ap.add_argument("--pair-distance-m", type=float, default=60.0)
    ap.add_argument("--pairs-per-image", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="prepare: the first N frames only (a quick try)")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--max-error-m", type=float, default=2.0, help="align: RANSAC limit of the similarity to the labels")
    ap.add_argument("--no-refine", action="store_true", help="align: keep the label cameras")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    preset = dict(PRESETS[a.preset])
    if a.sols:
        preset["sols"] = tuple(a.sols)
    if a.side:
        preset["side"] = a.side.upper()
    data = Path(a.data)
    preset["name"] = block_folder_name(preset["sols"][0], a.block or preset["block"])
    work = Path(a.work or blocks_root() / preset["name"])
    if a.step == "where":                                  # 0.99.12: the batch file asks for the folder
        print(work)
        return 0
    work.mkdir(parents=True, exist_ok=True)
    print(f"block folder: {work}")
    wp = None
    if a.step in ("waypoints", "prepare", "all"):
        try:
            wp = waypoints(data)
            ws = waypoint_summary(wp, preset["sols"])
            print(f"waypoints: {wp['_rpp_source']['n_features']} in {wp['_rpp_source']['path']}; sols {preset['sols']}: {ws}")
        except Exception as e:                                         # noqa: BLE001
            print(f"waypoints: not available ({type(e).__name__}: {e}) - the label site chain alone places the sites")
            wp = None
        if a.step == "waypoints":
            return 0 if wp is not None else 1
    rc = 0
    if a.step in ("fetch", "all"):
        sols = list(range(preset["sols"][0], preset["sols"][1] + 1))
        rep = F.fetch(data, sols, None, preset["side"], None, a.list, preset["families"])
        (work / f"fetch{'_list' if a.list else ''}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        found = sum(1 for v in rep["sols"].values() if v.get("found"))
        n = sum(v.get("products", 0) for v in rep["sols"].values())
        print(f"fetch: {found} of {len(rep['sols'])} sols found, {n} products (about {rep.get('est_mb', 0) / 1000:.1f} GB), "
              f"{rep['downloaded']} files downloaded ({rep['bytes'] / 1e9:.2f} GB), {rep['kept']} already there, "
              f"{len(rep['failed'])} failed")
        if not found:
            print(f"no sol folder answered - the addresses tried are in {work}")
            return 1
        if a.list:
            return 0
    if a.step in ("prepare", "all"):
        man = prepare(work, data, preset, a.camera, a.mask, a.mask_model, a.radius_m, a.pair_distance_m,
                      a.pairs_per_image, wp, a.limit)
        print(f"prepare: {len(man['images'])} frames, {man['pairs']} pairs in {work}")
    if a.step in ("align", "all"):
        rep = align(work, a.max_features, max_error_m=a.max_error_m, refine=not a.no_refine)
        rc = 0 if rep.get("ok") else 1
        print((work / "report.txt").read_text(encoding="utf-8") if (work / "report.txt").is_file() else rep)
    return rc


if __name__ == "__main__":
    sys.exit(main())
