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
* **masks** - the mask CNN (``--mask-model``, default the released ``rpp_mask_v3``) on every frame; the mask is the
  image's alpha channel (gray + alpha PNG) and ``masks\\`` for COLMAP (0.99.16; ``--mask both`` / ``mxy`` also fetch
  and use MIPL's rover mask, MXY).  For the training set the CNN's masks are written beside the images in the layout of
  a mask training set (``mask_candidates\images\<stem>.png`` RGBA, ``masks\<stem>.png`` 255 = terrain), with the MXY
  mask, an overlay and ``review.csv`` (frames where the CNN and MXY disagree most first) to clean them by.
* **pairs** - an image pair is matched when the two cameras are within ``--pair-distance-m`` and their views meet on
  the ground: the point where each optical axis meets the ground 1.9 m below the camera (at most 40 m out) lies
  inside the other image, or the middle of the two such points lies inside both.  Both eyes of a stereo pair always.  At most ``--pairs-per-image`` per image (nearest
  first).  ``pairs.txt``.
* **align** - SIFT with the masks; the pairs matched; the incremental mapper with the start cameras held; the model
  put into the block frame by a similarity onto the label camera centers (RANSAC, ``--max-error-m``); then, 0.99.16
  (Christian, 17 Nov 2026: "hold the principal point on the first stage. Solve A pair rig and B pair rig separately"),
  the bundle adjustment in stages, each twice with the observations above 4 px dropped in between:
  1. *free images* - focal lengths and k1, k2, p1, p2, k3 free; the principal points held (with a free attitude per
     image a principal point is not observable: on 17 Nov NLA's went 15 px); k4-k6 held at zero;
  2. *stereo rigs* - each side's pair (NLA + NRA, NLB + NRB, and the front Hazcams' FLA + FRA, FLB + FRB) becomes one
     COLMAP rig: one right-from-left pose per side, solved with the cameras, principal points held.  These are the
     block's cameras;
  3. *check* - on a copy, the principal points freed too: how far they move and what that gains.
  4. *temperature* (0.99.17) - cameras and rigs in 10 degC bins: fx, fy per camera bin, a right-from-left pose per
     rig bin; lines through the bins give the focal length (ppm/degC) and the rig rotation (mdeg/degC) about -20 degC.
  The model is then scaled so that the Navcam rigs' baselines are the labels' (``--scale labels`` keeps the label
  positions' scale): the baseline is hardware, the label station separations of a small block are not good to 5 %.
  ``report.txt``: frames registered per camera, each stage's reprojection error, each camera's parameters against
  its start, its residuals by radius, each rig against the labels' pair, the scale, the label-center residuals per
  site, and the Navcam verdict (:func:`verdict`).  ``cmods_candidate\\MSL_*.json`` are the model files.
* **significance** (0.99.17) - are p1, p2, k3 needed?  See :func:`significance`.
* **finish** - the stages again from the mapper's model (``sparse\\0``), without matching.
* **promote** - the Navcam model files into the camera models in use (``RPP_CMODS``), as the Mars 2020 models go
  (:func:`rpp.calib.models.promote`: checked, copied, a line in ``CHANGES.md``), when the verdict is ok
  (``--promote-if-ok`` on align does it at once; ``--force`` past a failed verdict).  The next ``prepare`` starts from
  promoted cameras (``--label-cameras``: from the labels).

Front Hazcams (0.99.16, "please include front hazcam images in the msl block"): family F of the preset; fetched from
the Hazcam volume (MSLHAZ_1XXX) into ``D:\\data\\msl\\hazcam\\SOLnnnnn``; 124 degrees wide, so their COLMAP camera is the
fisheye one (THIN_PRISM_FISHEYE fitted to the label CAHVORE), never the perspective one.

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
CAMERA_HEIGHT_M = {"N": NAVCAM_HEIGHT_M, "F": 0.68, "R": 0.78}     # 0.99.16: the Hazcams sit low on the body
LOOK_MAX_M = 40.0
# 0.99.16 (Christian, 17 Nov 2026: "hold the principal point on the first stage. Solve A pair rig and B pair rig
# separately. If the camera models look good, then promote the MSL navcams the same way we do for Mars 2020"):
# the parameters held in the bundle adjustments, by COLMAP model (indices into params)
HELD_ZERO = {"FULL_OPENCV": (9, 10, 11), "THIN_PRISM_FISHEYE": (10, 11)}       # k4-k6; sx1, sy1 (as the Mars 2020 form)
PRINCIPAL_POINT = (2, 3)
ALPHA_MASKED = 128                     # the mask in the image's alpha channel, as the Mars 2020 blocks (0.5)
MODEL_FORM = "msl-brown-v1"
# 0.99.17 (Christian, 17 Nov 2026: "the msl navcams do have a strong temperature dependence. So model the focal length
# and rig rotation as a function of temperature"): the thermal stage's bins and the laws' reference temperature
T_BIN_DEGC = 10.0
T0_DEGC = -20.0
MIN_BIN_FRAMES = 8
RADIAL_BINS = ((0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.01))
PRESETS: Dict[str, Dict[str, Any]] = {
    "yellowknife": {"sols": (124, 302), "side": "AB", "families": "NF", "sites": None, "block": "yellowknife_bay",
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
            if not np.any(np.abs(np.asarray(q, float)) > 1e-12):         # 0.99.18: an all-zero quaternion is "not given"
                if not np.any(np.abs(np.asarray(lk["offset_ned_m"], float)) > 1e-9):   # (early products, sols 124-161):
                    continue                                             # with a zero offset there is no link in the label
                q = [1, 0, 0, 0]
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
    if kind == "fisheye" or infos[0].get("family", "N") != "N":          # 0.99.16: the Hazcams (124 degrees) always
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


def promoted_camera(group: str, cam: Dict[str, Any]) -> Dict[str, Any]:
    """``cam`` (the label fit) with the promoted model's parameters when the camera models in use hold ``MSL_<group>.json``
    of the same COLMAP model and size (0.99.16: a promoted camera is the start of the next block)."""
    try:
        from rpp.calib import models as M
        f = M._folder(None) / f"MSL_{group}.json"
        if not f.is_file():
            return cam
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:                                                  # noqa: BLE001
        return cam
    if d.get("model") != cam["model"] or (d.get("width"), d.get("height")) != (cam["width"], cam["height"]):
        return cam
    return dict(cam, label_params=cam["params"], params=[float(x) for x in d["params"]],
                source=f"promoted {f.name} ({d.get('source', '')})"[:200])


# ------------------------------------------------------------------------------------------------- pairs
def _axis_and_look(r: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    R = np.asarray(r["R_w2c"], float)
    C = np.asarray(r["C_block_m"], float)
    a = R[2]                                   # the optical axis in the block frame
    t = LOOK_MAX_M
    if a[2] < -1e-3:
        t = min(t, CAMERA_HEIGHT_M.get(r.get("family", "N"), NAVCAM_HEIGHT_M) / -a[2])
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


def write_image(path: Path, img8: np.ndarray, rover: Optional[np.ndarray] = None) -> None:
    """The block's image: gray, or - with a mask - four channels (the gray value three times and the mask as alpha:
    255 terrain, ``ALPHA_MASKED`` where masked).  0.99.18: OpenCV writes no two-channel (gray + alpha) PNG - 0.99.16's
    attempt stopped the prepare step at its first image."""
    import cv2
    if rover is None:
        ok = cv2.imwrite(str(path), img8)
    else:
        a = np.where(rover, ALPHA_MASKED, 255).astype(np.uint8)
        ok = cv2.imwrite(str(path), np.dstack([img8, img8, img8, a]))
    if not ok:
        raise OSError(f"could not write {path}")


def write_masks(work: Path, frames: List[Dict[str, Any]], mxy: Dict[str, Path], mask_mode: str = "cnn",
                model: Optional[str] = "rpp_mask_v3", candidates: bool = True) -> Dict[str, Any]:
    """Images, COLMAP masks and the training candidates (see the module's text).  0.99.16 (Christian, 17 Nov 2026: "for
    msl images, use the current mask cnn model and save them as the alpha channel. no need to download the official
    hardware masks"): the default is the CNN alone, and each image is a PNG whose alpha channel is the mask
    (255 terrain, ``ALPHA_MASKED`` where masked, as the Mars 2020 blocks); ``masks\\`` stays for COLMAP."""
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
        write_image(work / "images" / name, img8, rover if (infer is not None or mx is not None) else None)
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
            f"model: {model}; images: MSL Navcam RAD, percentile stretch {F.STRETCH} to 8 bit x {F.BRIGHTNESS:g}, one band in R, G and B "
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


def prepare(work: Path, data: Path, preset: Dict[str, Any], camera: str = "opencv5", mask_mode: str = "cnn",
            mask_model: Optional[str] = "rpp_mask_v3", radius_m: Optional[float] = None, pair_distance_m: float = 60.0,
            pairs_per_image: int = 40, wp: Optional[Dict[str, Any]] = None, limit: Optional[int] = None,
            label_cameras: bool = False) -> Dict[str, Any]:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    sols = list(range(int(preset["sols"][0]), int(preset["sols"][1]) + 1))
    infos, mxy = scan(Path(data), sols, preset["side"], preset["families"], preset.get("sites"), cache=work / "labels_cache.json")
    if not infos:
        raise ValueError(f"no full-frame RAD of sols {sols[0]}-{sols[-1]} under {data} - run the fetch first")
    origin, site_rep = site_origins(infos, wp=wp)
    frames = place(infos, origin)
    frames, sel = select(frames, radius_m if radius_m is not None else preset.get("radius_m"), site_rep.get("anchor"))
    if limit:
        frames = sorted(frames, key=lambda r: (r["sol"], r["sclk"]))[:int(limit)]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in frames:
        groups.setdefault(r["group"], []).append(r)
    cameras = {g: group_camera(v, camera) for g, v in sorted(groups.items())}
    if not label_cameras:
        cameras = {g: promoted_camera(g, c) for g, c in cameras.items()}
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
    F.write_open_in_colmap(work)                               # 0.99.15: there from the start (it opens the model once made)
    log.info("prepare: %d frames (%s), %d stations, sites %s (anchor %s), %d pairs; cameras %s", len(frames),
             sel.get("by_camera"), sel.get("stations"), site_rep.get("chained"), site_rep.get("anchor"), len(pairs),
             ", ".join(f"{g} ({c['frames']} frames, fit {c['fit_rms_px_max']:.3f} px)" for g, c in cameras.items()))
    return man


# ------------------------------------------------------------------------------------------------- align
def refine_cameras(rec, man: Dict[str, Any], rounds: int = 2, max_error_px: float = 4.0, hold_pp: bool = False,
                   rigs: bool = False, only: Optional[Sequence[int]] = None,
                   extra_const: Optional[Dict[int, Sequence[int]]] = None, skip_images: Sequence[int] = ()) -> List[Dict[str, Any]]:
    """Bundle adjustment with every camera's focal lengths and distortion free and the terms of ``HELD_ZERO`` held (a
    subset manifold on the camera block); ``hold_pp``: the principal points held too (0.99.16).  ``rigs``: the stereo
    rigs' sensor-from-rig poses free (one per rig, :func:`add_rigs`).  Two frames' poses are held for the gauge.
    Between rounds the observations above ``max_error_px`` are dropped.  0.99.17: ``only`` - the parameter indices left
    free in every camera (the thermal stage: fx, fy); ``extra_const`` {camera id: indices} held besides (the
    significance study); ``skip_images`` are left out of the problem (its jackknife)."""
    import pycolmap
    import pyceres
    out = []
    skip = set(int(i) for i in skip_images)
    for k in range(int(rounds)):
        opts = pycolmap.BundleAdjustmentOptions()
        opts.refine_focal_length = opts.refine_principal_point = opts.refine_extra_params = True
        opts.refine_sensor_from_rig = bool(rigs)
        cfg = pycolmap.BundleAdjustmentConfig()
        ids = [i for i in sorted(rec.reg_image_ids()) if i not in skip]
        for i in ids:
            cfg.add_image(i)
        far = max(ids[1:], key=lambda j: np.linalg.norm(rec.image(j).projection_center() - rec.image(ids[0]).projection_center()))
        cfg.set_constant_rig_from_world_pose(rec.image(ids[0]).frame_id)
        cfg.set_constant_rig_from_world_pose(rec.image(far).frame_id)
        ba = pycolmap.create_default_ceres_bundle_adjuster(opts, cfg, rec)
        held = []
        for cid in list(rec.cameras):
            cam = rec.camera(cid)                      # the camera in the reconstruction (its params are the block)
            const = list(HELD_ZERO.get(cam.model_name, ())) + (list(PRINCIPAL_POINT) if hold_pp else [])
            const += list((extra_const or {}).get(int(cid), ()))
            pa = cam.params
            if only is not None:
                const = [i for i in range(len(pa)) if i not in set(only)]
            const = sorted(set(const))
            if const and ba.problem.has_parameter_block(pa):
                m = pyceres.SubsetManifold(len(pa), const)
                ba.problem.set_manifold(pa, m)
                held.append((pa, m))               # the problem does not own the manifold: kept alive to the solve
        summ = ba.solve()
        del held
        rec.update_point_3d_errors()                           # 0.99.17: the stored errors are the mapper's until refreshed -
        n_drop = 0                                             # compute_mean_reprojection_error() reads them
        if k < rounds - 1:
            n_drop = int(pycolmap.ObservationManager(rec).filter_all_points3D(float(max_error_px), 1.5))
        out.append({"round": k + 1, "reprojection_px": float(rec.compute_mean_reprojection_error()), "dropped": n_drop,
                    "hold_pp": bool(hold_pp), "rigs": bool(rigs),
                    "cost": [float(getattr(summ, "initial_cost", float("nan"))), float(getattr(summ, "final_cost", float("nan")))]})
    return out


# ------------------------------------------------------------------------------------------------- rigs (0.99.16)
def rig_sides(man: Dict[str, Any]) -> List[Tuple[str, str]]:
    """The stereo rigs of the block: (family, side) with both eyes among the cameras - ("N", "A"), ("N", "B"),
    ("F", "B") ...  Each is solved on its own (Christian: "Solve A pair rig and B pair rig separately")."""
    g = set(man["cameras"])
    return sorted({(x[0], x[2]) for x in g if x[0] + "L" + x[2] in g and x[0] + "R" + x[2] in g})


def add_rigs(rec, man: Dict[str, Any], db_path: Optional[Path]) -> Dict[str, Any]:
    """Each side's stereo pair as a COLMAP rig (left eye the reference): the two images of a pair (the same name but
    for the eye) become one frame, and the right-from-left pose - one per rig - starts from the model's own pairs
    (COLMAP's ``apply_rig_config``, which also writes the rigs and frames into the database).  Returns
    {"<family><side>": {"rig_id", "pairs", "singles"}}."""
    import pycolmap
    sides = rig_sides(man)
    if not sides:
        return {}
    cfg = [pycolmap.RigConfig(cameras=[pycolmap.RigConfigCamera(ref_sensor=True, image_prefix=f"{f}L{sd}_"),
                                        pycolmap.RigConfigCamera(ref_sensor=False, image_prefix=f"{f}R{sd}_")])
           for f, sd in sides]
    tmp = None
    if db_path is None or not Path(db_path).is_file():           # a model without its database (tests, a re-finish)
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="rpp_msl_rig_")) / "rig.db"
        db = pycolmap.Database.open(str(tmp))
        for c in rec.cameras.values():
            db.write_camera(c, use_camera_id=True)
        for r in rec.rigs.values():
            db.write_rig(r, use_rig_id=True)
        for iid, im in rec.images.items():
            db.write_image(pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=iid), use_image_id=True)
        for fid, fr in rec.frames.items():
            nf = pycolmap.Frame(frame_id=fid, rig_id=fr.rig_id)
            for d in fr.data_ids:
                nf.add_data_id(d)
            db.write_frame(nf, use_frame_id=True)
    else:
        db = pycolmap.Database.open(str(db_path))
    try:
        pycolmap.apply_rig_config(cfg, db, rec)
    finally:
        db.close()
    out: Dict[str, Any] = {}
    cam_group = _camera_groups(rec, man)
    for rid, rig in rec.rigs.items():
        g = cam_group.get(int(rig.ref_sensor_id.id), "")
        if not list(rig.non_ref_sensors):
            continue
        fr = [f for f in rec.frames.values() if f.rig_id == rid and f.has_pose()]
        out[g[0] + g[2]] = {"rig_id": int(rid), "pairs": sum(1 for f in fr if f.num_data_ids() == 2),
                            "singles": sum(1 for f in fr if f.num_data_ids() == 1)}
    return out


def _camera_groups(rec, man: Dict[str, Any]) -> Dict[int, str]:
    grp = {r["image"]: r["group"] for r in man["images"]}
    out: Dict[int, str] = {}
    for im in rec.images.values():
        out.setdefault(int(im.camera_id), grp.get(im.name, "?"))
    return out


def label_rigs(man: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The labels' right-from-left pose of each rig: {"NA": {"pairs", "baseline_m", "R_sensor_from_ref",
    "t_sensor_from_ref_m"}} - medians over the pairs (the label CAHVOR pair)."""
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in man["images"]:
        pairs.setdefault(r["pair_key"], {})[r["eye"]] = r
    acc: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for d in pairs.values():
        if "L" in d and "R" in d:
            Rl, Rr = np.asarray(d["L"]["R_w2c"], float), np.asarray(d["R"]["R_w2c"], float)
            t = -(Rr @ (np.asarray(d["R"]["C_block_m"], float) - np.asarray(d["L"]["C_block_m"], float)))
            acc.setdefault(d["L"]["family"] + d["L"]["side"], []).append((Rr @ Rl.T, t))
    out = {}
    for k, v in acc.items():
        t = np.median([x[1] for x in v], axis=0)
        U, _, Vt = np.linalg.svd(np.mean([x[0] for x in v], axis=0))
        out[k] = {"pairs": len(v), "baseline_m": float(np.linalg.norm(t)), "t_sensor_from_ref_m": [float(x) for x in t],
                  "R_sensor_from_ref": (U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt).tolist()}
    return out


def _angles_mdeg(R: np.ndarray) -> List[float]:
    """Small rotation as (pitch about x, yaw about y, roll about z) of the camera frame, mdeg."""
    R = np.asarray(R, float)
    return [float(1e3 * np.degrees(x)) for x in (0.5 * (R[2, 1] - R[1, 2]), 0.5 * (R[0, 2] - R[2, 0]), 0.5 * (R[1, 0] - R[0, 1]))]


def pair_scatter(rec, man: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Before the rigs: each pair's own right-from-left rotation against the labels' (how constant the pair is when
    every image is free), per rig: {"NA": {"pairs", "sd_mdeg": [pitch, yaw, roll], "baseline_sd_ppm"}}."""
    lab = label_rigs(man)
    by = {im.name: im for im in rec.images.values()}
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in man["images"]:
        if r["image"] in by:
            pairs.setdefault(r["pair_key"], {})[r["eye"]] = r
    acc: Dict[str, List[Tuple[List[float], float]]] = {}
    for d in pairs.values():
        if "L" in d and "R" in d:
            k = d["L"]["family"] + d["L"]["side"]
            L, R = by[d["L"]["image"]], by[d["R"]["image"]]
            rel = R.cam_from_world().rotation.matrix() @ L.cam_from_world().rotation.matrix().T
            b = float(np.linalg.norm(np.asarray(L.projection_center()) - np.asarray(R.projection_center())))
            acc.setdefault(k, []).append((_angles_mdeg(rel @ np.asarray(lab[k]["R_sensor_from_ref"]).T), b))
    out = {}
    for k, v in acc.items():
        a, b = np.array([x[0] for x in v]), np.array([x[1] for x in v])
        out[k] = {"pairs": len(v), "sd_mdeg": [round(float(x), 1) for x in a.std(axis=0)],
                  "baseline_sd_ppm": round(float(1e6 * b.std() / b.mean()), 0)}
    return out


def solved_rigs(rec, man: Dict[str, Any], rigs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The solved right-from-left pose of each rig with its change from the labels': rotation (pitch, yaw, roll, mdeg),
    baseline (model, label, ratio)."""
    lab = label_rigs(man)
    out = {}
    for k, info in rigs.items():
        rig = rec.rigs[info["rig_id"]]
        T = rig.sensor_from_rig(list(rig.non_ref_sensors)[0])
        R, t = T.rotation.matrix(), np.asarray(T.translation, float)
        lb = lab.get(k)
        out[k] = dict(info, R_sensor_from_ref=R.tolist(), t_sensor_from_ref_m=[float(x) for x in t],
                      baseline_m=float(np.linalg.norm(t)))
        if lb:
            out[k].update(label_baseline_m=lb["baseline_m"], baseline_ratio=float(np.linalg.norm(t) / lb["baseline_m"]),
                          rotation_vs_label_mdeg=[round(x, 1) for x in _angles_mdeg(R @ np.asarray(lb["R_sensor_from_ref"]).T)],
                          label_R_sensor_from_ref=lb["R_sensor_from_ref"], label_t_sensor_from_ref_m=lb["t_sensor_from_ref_m"])
    return out


def scale_to_baseline(rec, rigs: Dict[str, Dict[str, Any]], family: str = "N") -> Optional[Dict[str, Any]]:
    """The model scaled (about the middle of its cameras) so that the ``family`` rigs' baselines are the labels' - the
    one length of the block that is hardware (0.99.16; on 17 Nov the label station separations, 4 m across with
    5-30 cm errors, had set a scale 5 % off the baseline).  Returns {"factor", "by_rig"} and updates ``rigs``."""
    import pycolmap
    use = {k: v for k, v in rigs.items() if k[0] == family and v.get("baseline_ratio")}
    if not use:
        return None
    w = np.array([v["pairs"] for v in use.values()], float)
    k = float(1.0 / (np.sum(w * np.array([v["baseline_ratio"] for v in use.values()])) / w.sum()))
    c = np.mean([np.asarray(im.projection_center(), float) for im in rec.images.values()], axis=0)
    rec.transform(pycolmap.Sim3d(k, pycolmap.Rotation3d(np.eye(3)), (1.0 - k) * c))
    by = {kk: round(v["baseline_ratio"], 5) for kk, v in rigs.items() if v.get("baseline_ratio")}
    for v in rigs.values():
        v["t_sensor_from_ref_m"] = [float(x) * k for x in v["t_sensor_from_ref_m"]]
        v["baseline_m"] = float(v["baseline_m"] * k)
        if v.get("baseline_ratio"):
            v["baseline_ratio"] = float(v["baseline_ratio"] * k)
    return {"factor": k, "family": family, "ratio_before_by_rig": by}


# ------------------------------------------------------------------------------------------------- temperature (0.99.17)
def temperature_bins(man: Dict[str, Any], names: Sequence[str], width: float = T_BIN_DEGC,
                     min_frames: int = MIN_BIN_FRAMES) -> Tuple[Dict[str, int], Dict[str, float]]:
    """({image: bin}, {image: temperature}) for the registered images ``names``.  A pair takes one temperature (the mean
    of its two CCDs) so that both eyes fall into one bin; the bin is floor(T / width); within a family and side, a bin
    with fewer than ``min_frames`` frames joins the nearest one that has them."""
    names = set(names)
    by_pair: Dict[str, List[Dict[str, Any]]] = {}
    for r in man["images"]:
        if r["image"] in names and r.get("ccd_degC") is not None:
            by_pair.setdefault(r["pair_key"], []).append(r)
    T: Dict[str, float] = {}
    side: Dict[str, str] = {}
    for v in by_pair.values():
        t = float(np.mean([r["ccd_degC"] for r in v]))
        for r in v:
            T[r["image"]] = t
            side[r["image"]] = r["family"] + r["side"]
    bins = {n: int(np.floor(t / width)) for n, t in T.items()}
    for sd in set(side.values()):
        mine = [n for n in bins if side[n] == sd]
        count: Dict[int, int] = {}
        for n in mine:
            count[bins[n]] = count.get(bins[n], 0) + 1
        good = sorted(b for b, c in count.items() if c >= min_frames) or sorted(count)[:1]
        for n in mine:
            if bins[n] not in good:
                bins[n] = min(good, key=lambda b: (abs(b - bins[n]), b))
    return bins, T


def _line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Dict[str, Any]:
    """Weighted line y = a + b x: {"a", "b", "sd_b", "rms", "n"} (sd from the scatter about the line; None under 3 points)."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    if len(x) < 2 or np.ptp(x) < 1e-9:
        return {"a": float(np.average(y, weights=w)) if len(x) else None, "b": None, "sd_b": None, "rms": None, "n": int(len(x))}
    A = np.c_[np.ones(len(x)), x] * np.sqrt(w)[:, None]
    c, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    r = y - (c[0] + c[1] * x)
    out = {"a": float(c[0]), "b": float(c[1]), "sd_b": None, "rms": float(np.sqrt(np.average(r ** 2, weights=w))), "n": int(len(x))}
    if len(x) >= 3:
        s2 = float(np.sum(w * r ** 2) / (len(x) - 2))
        out["sd_b"] = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]))
    return out


def thermal_stage(rec, man: Dict[str, Any], rounds: int = 1, out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Focal length and rig rotation against temperature, from the rig model ``rec`` (not changed; the binned model is
    written to ``out_dir``).

    Every camera is split into temperature bins (:func:`temperature_bins`): a bin's camera keeps the block's
    distortion and principal point and solves its own fx, fy; each side's rig is split the same way, a bin's rig
    solving its own right-from-left pose.  (COLMAP has no parameter that depends on temperature; a camera and a rig
    per bin is how the Mars 2020 thermal stage does it.)  Lines through the bins, weighted by their frames, give
    * per camera: f(T) = f0 (1 + ppm 1e-6 (T - T0)), from the mean of fx and fy;
    * per rig: pitch, yaw, roll (mdeg) of the bin's rotation from the block's, a + rate (T - T0).
    Returns {"reprojection_px", "T0_degC", "bin_degC", "cameras": {group: {"f_ppm_per_degC", "sd", "f0_scale", "bins"}},
    "rigs": {"NA": {"rate_mdeg_per_degC", "sd", "offset_mdeg", "rms_mdeg", "bins"}}}."""
    import copy
    import tempfile
    import pycolmap
    groups = _camera_groups(rec, man)
    base = {g: np.array(rec.cameras[cid].params, float) for cid, g in groups.items()}
    base_R = {}
    for rig in rec.rigs.values():
        nr = list(rig.non_ref_sensors)
        if nr:
            g = groups.get(int(rig.ref_sensor_id.id), "")
            base_R[g[0] + g[2]] = rig.sensor_from_rig(nr[0]).rotation.matrix()
    names = {im.image_id: im.name for im in rec.images.values()}
    bins, T = temperature_bins(man, list(names.values()))
    # a new reconstruction with a camera per (camera, bin): the images' cameras cannot be changed in place (their frames
    # and rigs name the old ones), so cameras, images (with their observations) and points (with their ids and tracks)
    # are copied over with trivial rigs, and the rigs are then configured per bin
    src = rec
    rec = pycolmap.Reconstruction()
    new_cam: Dict[Tuple[str, int], int] = {}
    cam_of: Dict[int, int] = {}
    nxt = 1
    for iid, name in sorted(names.items()):
        g = groups[int(src.image(iid).camera_id)]
        key = (g, bins.get(name, 0))
        if key not in new_cam:
            c = copy.deepcopy(src.cameras[src.image(iid).camera_id])
            c.camera_id = nxt
            rec.add_camera_with_trivial_rig(c)
            new_cam[key] = nxt
            nxt += 1
        cam_of[iid] = new_cam[key]
    for pid, pt in src.points3D.items():                       # the tracks follow once the images are in
        q = copy.deepcopy(pt)
        q.track = pycolmap.Track()
        rec.add_point3D_with_id(pid, q)
    for iid, name in sorted(names.items()):
        old = src.image(iid)
        g, b = groups[int(old.camera_id)], bins.get(name, 0)
        im = pycolmap.Image(image_id=iid, camera_id=cam_of[iid],          # the pair's two names differ by the eye alone:
                            name=f"{g}@{b:+04d}_" + (name[len(g) + 1:] if name.startswith(g + "_") else name))   # the prefix replaces it
        im.points2D = old.points2D
        rec.add_image_with_trivial_frame(im, old.cam_from_world())
    for pid, pt in src.points3D.items():
        rec.point3D(pid).track = copy.deepcopy(pt.track)
    sides = sorted({(g[0], g[2], b) for g, b in new_cam if (g[0] + "L" + g[2], b) in new_cam and (g[0] + "R" + g[2], b) in new_cam})
    cfg = [pycolmap.RigConfig(cameras=[pycolmap.RigConfigCamera(ref_sensor=True, image_prefix=f"{f}L{sd}@{b:+04d}_"),
                                        pycolmap.RigConfigCamera(ref_sensor=False, image_prefix=f"{f}R{sd}@{b:+04d}_")])
           for f, sd, b in sides]
    db = pycolmap.Database.open(str(Path(tempfile.mkdtemp(prefix="rpp_msl_thermal_")) / "thermal.db"))
    try:
        for c in rec.cameras.values():
            db.write_camera(c, use_camera_id=True)
        for r in rec.rigs.values():
            db.write_rig(r, use_rig_id=True)
        for iid, im in rec.images.items():
            db.write_image(pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=iid), use_image_id=True)
        for fid, fr in rec.frames.items():
            nf = pycolmap.Frame(frame_id=fid, rig_id=fr.rig_id)
            for d in fr.data_ids:
                nf.add_data_id(d)
            db.write_frame(nf, use_frame_id=True)
        if cfg:
            pycolmap.apply_rig_config(cfg, db, rec)
    finally:
        db.close()
    for iid, name in names.items():
        rec.image(iid).name = name
    runs = refine_cameras(rec, man, rounds=rounds, rigs=True, only=(0, 1))
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        rec.write(str(out_dir))
    Tm: Dict[Tuple[str, int], List[float]] = {}
    grp = {r["image"]: r["group"] for r in man["images"]}
    for iid, name in names.items():
        if name in bins:
            Tm.setdefault((grp[name], bins[name]), []).append(T[name])
    out: Dict[str, Any] = {"reprojection_px": float(rec.compute_mean_reprojection_error()), "rounds": runs,
                           "T0_degC": T0_DEGC, "bin_degC": T_BIN_DEGC, "cameras": {}, "rigs": {}}
    for g in sorted({k[0] for k in new_cam}):
        rows = []
        for (gg, b), cid in sorted(new_cam.items()):
            if gg != g:
                continue
            pp = np.array(rec.cameras[cid].params, float)
            rows.append({"bin": b, "T_degC": round(float(np.mean(Tm[(g, b)])), 2), "frames": len(Tm[(g, b)]),
                         "fx": float(pp[0]), "fy": float(pp[1]),
                         "f_ppm": float(1e6 * (0.5 * (pp[0] / base[g][0] + pp[1] / base[g][1]) - 1.0))})
        ln = _line([r["T_degC"] - T0_DEGC for r in rows], [r["f_ppm"] for r in rows], [r["frames"] for r in rows])
        out["cameras"][g] = {"f_ppm_per_degC": ln["b"], "sd": ln["sd_b"], "f0_ppm": ln["a"], "rms_ppm": ln["rms"], "bins": rows,
                             "T_range_degC": [min(r["T_degC"] for r in rows), max(r["T_degC"] for r in rows)]}
    cam_bin = {cid: k for k, cid in new_cam.items()}
    for rid, rig in rec.rigs.items():
        nr = list(rig.non_ref_sensors)
        k = cam_bin.get(int(rig.ref_sensor_id.id))
        if not nr or k is None:
            continue
        g, b = k
        sd = g[0] + g[2]
        Tr = rig.sensor_from_rig(nr[0])
        n_pairs = sum(1 for f in rec.frames.values() if f.rig_id == rid and f.has_pose() and f.num_data_ids() == 2)
        ang = _angles_mdeg(Tr.rotation.matrix() @ base_R[sd].T) if sd in base_R else [float("nan")] * 3
        out["rigs"].setdefault(sd, {"bins": []})["bins"].append(
            {"bin": b, "T_degC": round(float(np.mean(Tm[(g, b)])), 2), "pairs": n_pairs, "pitch_yaw_roll_mdeg": [round(x, 2) for x in ang],
             "baseline_m": float(np.linalg.norm(Tr.translation))})
    for sd, v in out["rigs"].items():
        rows = sorted((r for r in v["bins"] if r["pairs"] >= 2), key=lambda r: r["bin"])
        v["bins"] = rows
        x, w = [r["T_degC"] - T0_DEGC for r in rows], [r["pairs"] for r in rows]
        ls = [_line(x, [r["pitch_yaw_roll_mdeg"][i] for r in rows], w) for i in range(3)]
        v.update(rate_mdeg_per_degC=[ln["b"] for ln in ls], sd=[ln["sd_b"] for ln in ls], offset_mdeg=[ln["a"] for ln in ls],
                 rms_mdeg=[ln["rms"] for ln in ls],
                 T_range_degC=[min(r["T_degC"] for r in rows), max(r["T_degC"] for r in rows)] if rows else None)
        bl = _line(x, [1e6 * (r["baseline_m"] / np.average([q["baseline_m"] for q in rows], weights=w) - 1) for r in rows], w) if rows else {}
        v["baseline_ppm_per_degC"], v["baseline_sd"] = bl.get("b"), bl.get("sd_b")
    return out


def _rot(pitch_yaw_roll_mdeg: Sequence[float]) -> np.ndarray:
    """The rotation of small angles (pitch about x, yaw about y, roll about z; mdeg) - the inverse of ``_angles_mdeg``."""
    import cv2
    R, _ = cv2.Rodrigues(np.radians(np.asarray(pitch_yaw_roll_mdeg, float) / 1e3))
    return R


# ------------------------------------------------------------------------------------------------- residuals, verdict
def camera_residuals(rec, man: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per camera: observations, reprojection rms and median (px), and the rms by radius (``RADIAL_BINS``, radius over
    the half diagonal from the image's middle) - a distortion model that fits leaves it flat."""
    groups = _camera_groups(rec, man)
    acc: Dict[str, List[np.ndarray]] = {}
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        half = 0.5 * float(np.hypot(cam.width, cam.height))
        rows = []
        for p2 in im.points2D:
            if not p2.has_point3D():
                continue
            pr = cam.img_from_cam(T * rec.points3D[p2.point3D_id].xyz)
            if pr is None:
                continue
            rows.append((float(np.linalg.norm(pr - p2.xy)), float(np.hypot(p2.xy[0] - 0.5 * cam.width, p2.xy[1] - 0.5 * cam.height) / half)))
        if rows:
            acc.setdefault(groups.get(int(im.camera_id), "?"), []).append(np.array(rows))
    out = {}
    for g, v in acc.items():
        a = np.vstack(v)
        e, r = a[:, 0], a[:, 1]
        bins = []
        for lo, hi in RADIAL_BINS:
            m = (r >= lo) & (r < hi)
            bins.append({"r": [lo, min(hi, 1.0)], "share": round(float(m.mean()), 3),
                         "rms_px": round(float(np.sqrt(np.mean(e[m] ** 2))), 3) if m.sum() >= 50 else None})
        rms = float(np.sqrt(np.mean(e ** 2)))
        have = [b["rms_px"] for b in bins if b["rms_px"] is not None]
        out[g] = {"observations": int(len(e)), "rms_px": round(rms, 3), "median_px": round(float(np.median(e)), 3),
                  "by_radius": bins, "radial_spread": round((max(have) - min(have)) / rms, 3) if have else None}
    return out


LIMITS = {"registered_share": 0.75, "registered_frames": 100, "median_px": 0.6, "radial_spread": 0.15, "rig_pairs": 50,
          "rig_cost_rise": 0.10, "pp_gain": 0.03, "baseline_agreement": 0.005, "thermal_bins": 3}


def verdict(rep: Dict[str, Any], man: Dict[str, Any], family: str = "N") -> Dict[str, Any]:
    """Do the ``family`` cameras look good enough to promote?  {"ok", "checks": [{"what", "value", "limit", "ok"}]}.
    The limits (``LIMITS``): each camera registered in 75 % of its frames and in 100 at least; median residual at most
    0.6 px; the residual flat with radius (spread of the radial bins' rms at most 15 % of the rms); each rig from 50
    pairs at least; the rigs cost at most 10 % in reprojection error over free images; freeing the principal points
    gains at most 3 %; the rigs' baseline ratios agree within 0.5 %."""
    checks: List[Dict[str, Any]] = []

    def add(what, value, limit, ok, side=None):
        checks.append({"what": what, "value": value, "limit": limit, "ok": bool(ok), "side": side})
    frames = {g: c["frames"] for g, c in man["cameras"].items()}
    cams = {g: c for g, c in (rep.get("cameras") or {}).items() if g[0] == family}
    if not cams:
        add(f"{family} cameras in the model", 0, ">= 1", False)
    for g, c in sorted(cams.items()):
        sh = c["registered"] / max(frames.get(g, 1), 1)
        add(f"{g} registered share", round(sh, 3), f">= {LIMITS['registered_share']}", sh >= LIMITS["registered_share"], g[2])
        add(f"{g} registered frames", c["registered"], f">= {LIMITS['registered_frames']}", c["registered"] >= LIMITS["registered_frames"], g[2])
        tc = ((rep.get("thermal") or {}).get("cameras") or {}).get(g)
        if rep.get("thermal"):                                 # 0.99.17: a temperature law needs three bins for its error
            nb = len((tc or {}).get("bins") or [])
            add(f"{g} temperature bins", nb, f">= {LIMITS['thermal_bins']}", nb >= LIMITS["thermal_bins"], g[2])
        r = (rep.get("residuals") or {}).get(g) or {}
        add(f"{g} median residual px", r.get("median_px"), f"<= {LIMITS['median_px']}",
            r.get("median_px") is not None and r["median_px"] <= LIMITS["median_px"], g[2])
        add(f"{g} residual spread with radius", r.get("radial_spread"), f"<= {LIMITS['radial_spread']}",
            r.get("radial_spread") is not None and r["radial_spread"] <= LIMITS["radial_spread"], g[2])
    rigs = {k: v for k, v in (rep.get("rigs") or {}).items() if k[0] == family}
    sides = {g[2] for g in cams}
    for sd in sorted(sides):
        v = rigs.get(family + sd)
        add(f"{family}{sd} rig pairs", v["pairs"] if v else 0, f">= {LIMITS['rig_pairs']}", bool(v) and v["pairs"] >= LIMITS["rig_pairs"], sd)
    st = rep.get("stages") or {}
    if st.get("free") and st.get("rigs"):
        last = "thermal" if st.get("thermal") else "rigs"      # 0.99.17: the rigs with their temperature bins, when solved
        rise = st[last]["reprojection_px"] / st["free"]["reprojection_px"] - 1.0
        add("reprojection rise with the rigs" + (" and temperature bins" if last == "thermal" else ""), round(rise, 3),
            f"<= {LIMITS['rig_cost_rise']}", rise <= LIMITS["rig_cost_rise"])
    else:
        add("rig stage", None, "run", False)
    if st.get("pp_check") and st.get("rigs") and not st["pp_check"].get("diverged"):
        gain = 1.0 - st["pp_check"]["reprojection_px"] / st["rigs"]["reprojection_px"]
        add("reprojection gain from free principal points", round(gain, 3), f"<= {LIMITS['pp_gain']}", gain <= LIMITS["pp_gain"])
    ratios = [v["baseline_ratio"] for v in rigs.values() if v.get("baseline_ratio")]
    if len(ratios) >= 2:
        d = max(ratios) / min(ratios) - 1.0
        add("rig baseline ratios agree", round(d, 4), f"<= {LIMITS['baseline_agreement']}", d <= LIMITS["baseline_agreement"])
    by_side = {sd: all(c["ok"] for c in checks if c["side"] in (None, sd)) for sd in sorted(sides)}
    return {"ok": all(c["ok"] for c in checks), "family": family, "sides": by_side, "checks": checks}


# ------------------------------------------------------------------------------------------------- significance (0.99.17)
# Christian, 17 Nov 2026: "could the ncam p1, p2, or k3 be held or set to zero? do a significance study after the next
# alignment."  (name, parameter indices, where their held values come from)
SIG_VARIANTS = (("p1 = p2 = 0", (6, 7), "zero"), ("k3 = 0", (8,), "zero"), ("p1 = p2 = k3 = 0", (6, 7, 8), "zero"),
                ("p1, p2 at the label's", (6, 7), "label"), ("k3 at the label's", (8,), "label"),
                ("p1, p2, k3 at the label's", (6, 7, 8), "label"))
SIG_LIMITS = {"image_px": 0.15, "rms_rise": 0.01}


def _opencv_px(params: Sequence[float], xy: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = [float(x) for x in params[:9]]
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return np.c_[fx * (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx, fy * (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy]


def camera_difference(pa: Sequence[float], pb: Sequence[float], width: int, height: int, n: int = 33) -> Dict[str, float]:
    """How far two FULL_OPENCV cameras image the same rays apart over the frame (px): as they are, and after the best
    common scale and shift (what a focal length and a pointing change take up - the part a block can tell apart)."""
    g = np.stack(np.meshgrid(np.linspace(0, width - 1, n), np.linspace(0, height - 1, n)), -1).reshape(-1, 2)
    xy = np.c_[(g[:, 0] - pa[2]) / pa[0], (g[:, 1] - pa[3]) / pa[1]]
    for _ in range(30):                                          # the rays of camera a's pixels
        d = _opencv_px(pa, xy) - g
        xy = xy - np.c_[d[:, 0] / pa[0], d[:, 1] / pa[1]]
    q = _opencv_px(pb, xy)
    raw = np.linalg.norm(q - g, axis=1)
    res = []
    for k, c in ((0, 0.5 * width), (1, 0.5 * height)):
        A = np.c_[g[:, k] - c, np.ones(len(g))]
        sol, *_ = np.linalg.lstsq(A, q[:, k], rcond=None)
        res.append(q[:, k] - A @ sol)
    fit = np.hypot(*res)
    return {"raw_rms": float(np.sqrt(np.mean(raw ** 2))), "raw_max": float(raw.max()),
            "rms": float(np.sqrt(np.mean(fit ** 2))), "max": float(fit.max())}


def remove_focal_temperature(rec, man: Dict[str, Any], slopes: Dict[str, Optional[float]]) -> int:
    """The observations of ``rec`` moved to the reference temperature: each image point scaled about its camera's
    principal point by 1 / (1 + ppm 1e-6 (T - T0)), with the camera's focal slope ``slopes[group]`` (ppm/degC).  With
    this a single camera per group fits images of every temperature.  Returns the number of images changed."""
    groups = _camera_groups(rec, man)
    T = {r["image"]: r.get("ccd_degC") for r in man["images"]}
    n = 0
    for im in rec.images.values():
        a, t = slopes.get(groups.get(int(im.camera_id), "")), T.get(im.name)
        if not a or t is None:
            continue
        k = 1.0 / (1.0 + 1e-6 * a * (float(t) - T0_DEGC))
        cam = rec.cameras[im.camera_id]
        c = np.array([cam.params[2], cam.params[3]], float)
        for p2 in im.points2D:                                 # the list's elements are the image's own points
            p2.xy = c + k * (np.asarray(p2.xy, float) - c)
        n += 1
    return n


def significance(work: Path, folds: int = 4, family: str = "N") -> Dict[str, Any]:
    """Are p1, p2 and k3 of the ``family`` cameras needed?  From the block's free-image model (``sparse\\free``: one
    camera per group, every image on its own, principal points held), with the focal lengths' temperature slopes of
    the last alignment taken out of the observations:
    * "full" - focal lengths, k1, k2, p1, p2, k3 refined;
    * each of ``SIG_VARIANTS`` - the named terms held at zero or at the label's values, the rest refined;
    * a jackknife of "full" over ``folds`` groups of sols (each left out in turn): the terms' standard deviations.
    Per camera and term: value, jackknife sd and z; per camera and variant: the reprojection rms against "full" and how
    far the variant's camera images the same rays from the full one (px, after a common scale and shift).  A variant
    "can be used" when that is at most ``SIG_LIMITS["image_px"]`` at the worst point of the frame and the rms rises by
    at most ``SIG_LIMITS["rms_rise"]``.  Writes ``significance.json`` / ``significance.txt``."""
    import copy
    import pycolmap
    work = Path(work)
    man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    rep = json.loads((work / "report.json").read_text(encoding="utf-8")) if (work / "report.json").is_file() else {}
    src = work / "sparse" / "free"
    if not (src / "images.bin").is_file():
        raise FileNotFoundError(f"no {src} - run the align (or finish) step of 0.99.17 first")
    base = pycolmap.Reconstruction(str(src))
    slopes = {g: c.get("f_ppm_per_degC") for g, c in ((rep.get("thermal") or {}).get("cameras") or {}).items()}
    n_T = remove_focal_temperature(base, man, slopes)
    groups = _camera_groups(base, man)
    nav = {cid: g for cid, g in groups.items() if g[0] == family and base.cameras[cid].model_name == "FULL_OPENCV"}
    t0 = time.time()

    def solve(r, const=None, skip=()):
        runs = refine_cameras(r, man, rounds=1, hold_pp=True, extra_const=const, skip_images=skip)
        return {"reprojection_px": float(r.compute_mean_reprojection_error()), "cost": runs[-1]["cost"],
                "cameras": {g: [float(x) for x in r.cameras[cid].params] for cid, g in nav.items()}}
    full_rec = copy.deepcopy(base)
    out: Dict[str, Any] = {"block": work.name, "family": family, "focal_slopes_removed_ppm_per_degC": slopes, "images_moved_to_T0": n_T,
                           "limits": SIG_LIMITS, "full": solve(full_rec), "variants": {}, "terms": {}}
    out["full"]["residuals"] = {g: v for g, v in camera_residuals(full_rec, man).items() if g[0] == family}
    log.info("significance: full %.3f px (%.1f min)", out["full"]["reprojection_px"], (time.time() - t0) / 60)
    for name, idx, source in SIG_VARIANTS:
        r = copy.deepcopy(base)
        for cid, g in nav.items():
            lab = man["cameras"][g].get("label_params", man["cameras"][g]["params"])
            pa = np.array(r.cameras[cid].params, float)
            for i in idx:
                pa[i] = 0.0 if source == "zero" else float(lab[i])
            r.camera(cid).params = pa
        v = solve(r, {cid: idx for cid in nav})
        res = camera_residuals(r, man)
        v["by_camera"] = {}
        for cid, g in nav.items():
            d = camera_difference(out["full"]["cameras"][g], v["cameras"][g], base.cameras[cid].width, base.cameras[cid].height)
            rise = res[g]["rms_px"] / out["full"]["residuals"][g]["rms_px"] - 1.0
            v["by_camera"][g] = {"rms_px": res[g]["rms_px"], "rms_rise": round(rise, 4), "image_rms_px": round(d["rms"], 3),
                                 "image_max_px": round(d["max"], 3),
                                 "ok": bool(d["max"] <= SIG_LIMITS["image_px"] and rise <= SIG_LIMITS["rms_rise"])}
        v["ok"] = all(c["ok"] for c in v["by_camera"].values())
        out["variants"][name] = v
        del r
        log.info("significance: %s %.3f px, %s (%.1f min)", name, v["reprojection_px"], "ok" if v["ok"] else "not ok", (time.time() - t0) / 60)
    sols = sorted({int(r["sol"]) for r in man["images"]})
    sol_of = {r["image"]: int(r["sol"]) for r in man["images"]}
    K = max(2, min(int(folds), len(sols)))
    jk = []
    for k in range(K):
        drop = set(sols[k::K])
        r = copy.deepcopy(full_rec)
        jk.append(solve(r, skip=[iid for iid, im in r.images.items() if sol_of.get(im.name) in drop])["cameras"])
        del r
    for g in sorted(nav.values()):
        P = np.array([j[g] for j in jk])
        sd = np.sqrt((K - 1) / K * np.sum((P - P.mean(axis=0)) ** 2, axis=0))
        full = np.array(out["full"]["cameras"][g])
        lab = np.array(man["cameras"][g].get("label_params", man["cameras"][g]["params"]), float)
        out["terms"][g] = {n: {"value": float(full[i]), "sd": float(sd[i]), "z": float(full[i] / sd[i]) if sd[i] > 0 else None,
                               "label": float(lab[i]), "z_from_label": float((full[i] - lab[i]) / sd[i]) if sd[i] > 0 else None}
                           for n, i in (("k1", 4), ("k2", 5), ("p1", 6), ("p2", 7), ("k3", 8))}
    out["jackknife_folds"] = K
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (work / "significance.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (work / "significance.txt").write_text(significance_text(out), encoding="utf-8")
    return out


def significance_text(out: Dict[str, Any]) -> str:
    L = [f"Significance of p1, p2, k3 - {out['block']}, {out['family']} cameras ({out['minutes']} min)",
         f"free-image model, principal points held, focal slopes removed (ppm/degC): {out['focal_slopes_removed_ppm_per_degC']}",
         f"full model: reprojection {out['full']['reprojection_px']:.3f} px", "",
         f"terms (value +- jackknife sd over {out['jackknife_folds']} groups of sols; z = value / sd; label value; z of the change from the label):"]
    for g, t in out["terms"].items():
        L.append(f"  {g}: " + "; ".join(f"{n} {v['value']:+.3g} +- {v['sd']:.2g} (z {v['z']:+.1f}; label {v['label']:+.3g}, z {v['z_from_label']:+.1f})"
                                       if v["z"] is not None else f"{n} {v['value']:+.3g}" for n, v in t.items()))
    L += ["", f"variants (per camera: rms rise over full; image change against the full camera after a common scale and shift, rms / max px;"
              f" usable when max <= {out['limits']['image_px']} px and rise <= {100 * out['limits']['rms_rise']:g} %):"]
    for name, v in out["variants"].items():
        L.append(f"  {name}: {v['reprojection_px']:.3f} px - {'USABLE' if v['ok'] else 'not usable'}")
        for g, c in v["by_camera"].items():
            L.append(f"      {g}: rise {100 * c['rms_rise']:+.2f} %, image {c['image_rms_px']:.3f} / {c['image_max_px']:.3f} px{'' if c['ok'] else '  <-'}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------- model files, promotion
def model_files(work: Path, rep: Dict[str, Any], man: Dict[str, Any], family: str = "N") -> Dict[str, Path]:
    """The block's cameras and rigs as model files in ``<work>\\cmods_candidate``: ``MSL_<group>.json`` (COLMAP model,
    size, parameters, what was held, the label start, the fit) and ``MSL_<family><side>_rig.json`` (right from left:
    the solved rotation; the translation in the solved direction with the labels' length).  These are what
    :func:`promote` copies into the camera models in use."""
    from rpp import __version__
    out_dir = Path(work) / "cmods_candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    temps: Dict[str, List[float]] = {}
    sols: Dict[str, List[int]] = {}
    for r in man["images"]:
        if r.get("ccd_degC") is not None:
            temps.setdefault(r["group"], []).append(float(r["ccd_degC"]))
        sols.setdefault(r["group"], []).append(int(r["sol"]))
    out: Dict[str, Path] = {}
    names = {"FULL_OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
             "THIN_PRISM_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"]}
    for g, c in sorted((rep.get("cameras") or {}).items()):
        if g[0] != family:
            continue
        cam = man["cameras"][g]
        nm = names.get(cam["model"], [])
        tc = ((rep.get("thermal") or {}).get("cameras") or {}).get(g) or {}
        if tc.get("f0_ppm") is not None:                       # the parameters at T0: the block's focal lengths moved there
            c = dict(c, refined=[c["refined"][0] * (1 + 1e-6 * tc["f0_ppm"]), c["refined"][1] * (1 + 1e-6 * tc["f0_ppm"])] + list(c["refined"][2:]))
        held = [nm[i] for i in sorted(set(HELD_ZERO.get(cam["model"], ())) | set(PRINCIPAL_POINT))] if nm else []
        t = temps.get(g) or []
        d = {"mission": "MSL", "model": cam["model"], "width": cam["width"], "height": cam["height"],
             "params": [float(x) for x in c["refined"]], "param_names": nm,
             "distortion": "perspective with Brown-Conrady distortion (radial k1, k2, k3; tangential p1, p2); k4 = k5 = k6 = 0"
             if cam["model"] == "FULL_OPENCV" else "fisheye (theta polynomial k1-k4) + tangential p1, p2; sx1 = sy1 = 0",
             "held_params": held, "pixel_origin": "corner of the first pixel (COLMAP)", "form": MODEL_FORM,
             "family": g[0], "eye": g[1], "side": g[2],
             "temperature": {"ccd_degC_min": round(min(t), 1), "ccd_degC_median": round(float(np.median(t)), 1),
                             "ccd_degC_max": round(max(t), 1), "note": "no temperature terms yet: one model over the block's range"} if t else None,
             "thermal": _camera_thermal(rep, g),
             "label_params": [float(x) for x in cam.get("label_params", cam["params"])],
             "start_params": [float(x) for x in c["start"]],
             "source": f"RPP {__version__} MSL block {man['preset'].get('name')} ({stamp}), sols {min(sols[g])}-{max(sols[g])}: "
                       "focal lengths and distortion from the block with the side's stereo rig; principal point held at its start",
             "fit": {"block": man["preset"].get("name"), "registered": c["registered"], "frames": cam["frames"],
                     "residuals": (rep.get("residuals") or {}).get(g), "reprojection_px": rep.get("reprojection_px")}}
        out[f"MSL_{g}.json"] = out_dir / f"MSL_{g}.json"
        out[f"MSL_{g}.json"].write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    for k, v in sorted((rep.get("rigs") or {}).items()):
        if k[0] != family or not v.get("label_baseline_m"):
            continue
        t = np.asarray(v["t_sensor_from_ref_m"], float)
        t = t / np.linalg.norm(t) * float(v["label_baseline_m"])
        tr = ((rep.get("thermal") or {}).get("rigs") or {}).get(k) or {}
        R0 = np.asarray(v["R_sensor_from_ref"], float)
        th_rig = None
        if tr.get("rate_mdeg_per_degC") and all(x is not None for x in tr["rate_mdeg_per_degC"]):
            R0 = _rot(tr["offset_mdeg"]) @ R0                  # the rotation at T0
            th_rig = {"T0_degC": T0_DEGC, "pitch_mdeg_per_degC": tr["rate_mdeg_per_degC"][0], "yaw_mdeg_per_degC": tr["rate_mdeg_per_degC"][1],
                      "roll_mdeg_per_degC": tr["rate_mdeg_per_degC"][2], "sd_mdeg_per_degC": tr["sd"], "rms_mdeg": tr["rms_mdeg"],
                      "T_range_degC": tr["T_range_degC"], "bins": len(tr["bins"]),
                      "note": "R(T) = rot(rate (T - T0)) R_sensor_from_ref; pitch about x, yaw about y, roll about z of the right camera"}
        d = {"mission": "MSL", "ref": f"{k[0]}L{k[1]}", "sensor": f"{k[0]}R{k[1]}", "R_sensor_from_ref": R0.tolist(), "thermal": th_rig,
             "t_sensor_from_ref_m": [float(x) for x in t], "baseline_m": float(v["label_baseline_m"]),
             "translation": "the block's direction, the label CAHVOR pair's length", "family": k[0], "side": k[1],
             "form": MODEL_FORM, "convention": f"{k[0]}L{k[1]} reference",
             "rotation_vs_label_mdeg": v.get("rotation_vs_label_mdeg"), "pairs": v["pairs"],
             "free_pair_sd_mdeg": ((rep.get("pair_scatter") or {}).get(k) or {}).get("sd_mdeg"),
             "source": f"RPP {__version__} MSL block {man['preset'].get('name')} ({stamp}): one right-from-left pose for "
                       f"the {v['pairs']} pairs of side {k[1]}"}
        out[f"MSL_{k}_rig.json"] = out_dir / f"MSL_{k}_rig.json"
        out[f"MSL_{k}_rig.json"].write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    return out


def _camera_thermal(rep: Dict[str, Any], g: str) -> Optional[Dict[str, Any]]:
    tc = ((rep.get("thermal") or {}).get("cameras") or {}).get(g)
    if not tc or tc.get("f_ppm_per_degC") is None:
        return None
    return {"T0_degC": T0_DEGC, "f_ppm_per_degC": tc["f_ppm_per_degC"], "sd_ppm_per_degC": tc["sd"], "rms_ppm": tc["rms_ppm"],
            "T_range_degC": tc["T_range_degC"], "bins": len(tc["bins"]),
            "note": "fx(T) = fx (1 + f_ppm_per_degC 1e-6 (T - T0)), fy alike; params are at T0; distortion and principal point without temperature terms"}


def promote(work: Path, force: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """The block's Navcam model files into the camera models in use, as the Mars 2020 models go
    (:func:`rpp.calib.models.promote`: checked, copied over the folder's files, a line with source and SHA-256 in
    ``CHANGES.md``) - when the block's verdict is ok (``force``: anyway)."""
    from rpp.calib import models as M
    work = Path(work)
    rep = json.loads((work / "report.json").read_text(encoding="utf-8"))
    v = rep.get("verdict") or {"ok": False, "checks": []}
    cand = work / "cmods_candidate"
    files = sorted(cand.glob("MSL_*.json")) if cand.is_dir() else []
    if not files:
        return {"promoted": False, "reason": f"no model files in {cand} (run the align step)"}
    bad = [f"{c['what']} {c['value']} (limit {c['limit']})" for c in v.get("checks", []) if not c["ok"]]
    good = [sd for sd, ok in (v.get("sides") or {}).items() if ok]
    if force or v.get("ok"):
        use, how = files, ("verdict ok" if v.get("ok") else "FORCED past the verdict")
    elif good:                                                 # 0.99.17: a side that passes goes in without the other
        use = [f for f in files if f.stem.split("_")[1][-1] in good]
        how = f"side {', '.join(good)} only (not passed: {'; '.join(bad)})"
    else:
        return {"promoted": False, "reason": "the cameras do not pass: " + "; ".join(bad)}
    note = f"MSL block {work.name}: {how}, reprojection {rep.get('reprojection_px', float('nan')):.3f} px"
    res = M.promote(use, note=note[:400], dry_run=dry_run)
    return {"promoted": not dry_run, "dry_run": dry_run, "sides": good if not (force or v.get("ok")) else sorted(v.get("sides") or {}), **res}


def align(work: Path, max_features: int = 8192, threads: int = -1, max_error_m: float = 2.0, refine: bool = True,
          rounds: int = 2, rigs: bool = True, pp_check: bool = True, scale: str = "baseline",
          thermal: bool = True) -> Dict[str, Any]:
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
    return finish(rec, man, work, len(recs), others, max_error_m, refine, rounds, t0, rigs, pp_check, scale, thermal)


def finish(rec, man: Dict[str, Any], work: Path, models: int = 1, others: Sequence[int] = (), max_error_m: float = 2.0,
           refine: bool = True, rounds: int = 2, t0: Optional[float] = None, rigs: bool = True, pp_check: bool = True,
           scale: str = "baseline", thermal: bool = True) -> Dict[str, Any]:
    """The mapped model into the block frame, the cameras refined, the report and exports (``align``'s second half).

    0.99.16, in stages: (1) "free" - every image on its own, focal lengths and distortion free, the principal points
    held; (2) "rigs" - each side's stereo pair one rig (:func:`add_rigs`), its right-from-left pose solved with the
    cameras, the principal points still held: these are the block's cameras; (3) "pp_check" - on a copy, the
    principal points freed too: how far they go and what it gains (a check, not a result).  ``scale``: "baseline"
    scales the model to the Navcam rigs' label baseline after the similarity onto the label centers; "labels" leaves
    the similarity's scale."""
    import copy
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
    gname = _camera_groups(rec, man)
    stages: Dict[str, Any] = {}
    rig_info: Dict[str, Any] = {}

    def snap(key: str, r, runs) -> None:
        stages[key] = {"reprojection_px": float(r.compute_mean_reprojection_error()), "rounds": runs,
                       "cameras": {gname.get(int(cid), str(cid)): [float(x) for x in c.params] for cid, c in r.cameras.items()}}
    if refine:
        runs = refine_cameras(rec, man, rounds=rounds, hold_pp=True)
        snap("free", rec, runs)
        (out / "free").mkdir(exist_ok=True)
        rec.write(out / "free")                                # 0.99.17: the significance study starts here
        rep["pair_scatter"] = pair_scatter(rec, man)
        if rigs and rig_sides(man):
            rig_info = add_rigs(rec, man, work / "database.db")
            runs = refine_cameras(rec, man, rounds=rounds, hold_pp=True, rigs=True)
            snap("rigs", rec, runs)
            if pp_check:
                chk = copy.deepcopy(rec)                       # 0.99.17: the rig poses held here - with them free too the
                runs = refine_cameras(chk, man, rounds=1, hold_pp=False, rigs=False)   # problem has no minimum (it ran away)
                snap("pp_check", chk, runs)
                e = stages["pp_check"]["reprojection_px"]
                stages["pp_check"]["diverged"] = bool(not np.isfinite(e) or e > 10 * stages["rigs"]["reprojection_px"])
                stages["pp_check"]["principal_point_moved_px"] = {
                    g: [round(v[2] - stages["rigs"]["cameras"][g][2], 2), round(v[3] - stages["rigs"]["cameras"][g][3], 2)]
                    for g, v in stages["pp_check"]["cameras"].items()}
                del chk
            if thermal:                                        # 0.99.17: f(T) per camera, rig rotation (T) per side
                rep["thermal"] = thermal_stage(rec, man, out_dir=out / "thermal")
                stages["thermal"] = {"reprojection_px": rep["thermal"]["reprojection_px"], "rounds": rep["thermal"].pop("rounds")}
        rep["refine"] = [dict(r, stage=k) for k in ("free", "rigs") if k in stages for r in stages[k]["rounds"]]
        _to_block(rec, lab, max_error_m, rep, "similarity_after_refine")
    rep["stages"] = stages
    if rig_info:
        rep["rigs"] = solved_rigs(rec, man, rig_info)
        if scale == "baseline":
            rep["scale"] = scale_to_baseline(rec, rep["rigs"])
            if rep["scale"]:
                res = [float(np.linalg.norm(np.asarray(im.projection_center()) - lab[im.name])) for im in rec.images.values() if im.name in lab]
                rep["scale"]["label_centers_m"] = {"median": float(np.median(res)), "p95": float(np.percentile(res, 95)), "max": float(np.max(res))}
    rep["points"] = int(rec.num_points3D())
    rep["reprojection_px"] = float(rec.compute_mean_reprojection_error())
    rep["cameras"] = {}
    for cid, c in rec.cameras.items():
        g = gname.get(int(cid), str(cid))
        p0, p1 = start[cid], np.array(c.params, float)
        rep["cameras"][g] = {"registered": sum(1 for im in rec.images.values() if im.camera_id == cid), "model": c.model_name,
                             "start": [float(x) for x in p0], "refined": [float(x) for x in p1],
                             "change": [float(x) for x in p1 - p0]}
    rep["residuals"] = camera_residuals(rec, man)
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
    rep["verdict"] = verdict(rep, man)
    rep["minutes"] = round((time.time() - t0) / 60.0, 1)
    (out / "block_frame").mkdir(exist_ok=True)
    rec.write(out / "block_frame")
    F.write_open_in_colmap(work)                               # 0.99.15: always there
    rec.export_PLY(str(work / "points_block_frame.ply"))
    (work / "cameras_refined.json").write_text(json.dumps(rep["cameras"], indent=1), encoding="utf-8")
    (work / "report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    rep["model_files"] = {k: str(v) for k, v in model_files(work, rep, man).items()}
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
        lines.append(f"refine {r.get('stage', '')} round {r['round']}: reprojection {r['reprojection_px']:.3f} px, dropped {r['dropped']}")
    st = rep.get("stages") or {}
    if st:
        lines.append("stages (principal points held in 1 and 2): " + "; ".join(
            f"{n} {st[k]['reprojection_px']:.3f} px" for k, n in (("free", "1 free images"), ("rigs", "2 stereo rigs"),
                                                                  ("pp_check", "check, principal points free"),
                                                                  ("thermal", "3 temperature bins"))
            if k in st and not st[k].get("diverged")))
        mv = (st.get("pp_check") or {}).get("principal_point_moved_px")
        if (st.get("pp_check") or {}).get("diverged"):
            lines.append("  check: with free principal points the solve ran away (no minimum)")
        elif mv:
            lines.append("  check: freed principal points move by (cx, cy px) " + ", ".join(f"{g} {v[0]:+.1f} {v[1]:+.1f}" for g, v in sorted(mv.items())))
    for g, c in sorted((rep.get("cameras") or {}).items()):
        names = (["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"] if c.get("model", "FULL_OPENCV") == "FULL_OPENCV"
                 else ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4"])
        ch = ", ".join(f"{n} {c['refined'][i]:.6g} ({c['change'][i]:+.3g})" for i, n in enumerate(names) if i < len(c["refined"]))
        lines.append(f"  {g}: {c['registered']} frames; {ch}")
        r = (rep.get("residuals") or {}).get(g)
        if r:
            lines.append(f"      residual rms {r['rms_px']} px, median {r['median_px']} px; by radius "
                         + ", ".join(f"{b['r'][0]:.1f}-{b['r'][1]:.1f}: {b['rms_px']}" for b in r["by_radius"]))
    for k, v in sorted((rep.get("rigs") or {}).items()):
        ps = (rep.get("pair_scatter") or {}).get(k) or {}
        lines.append(f"rig {k} ({k[0]}R{k[1]} from {k[0]}L{k[1]}): {v['pairs']} pairs, {v['singles']} single frames; baseline "
                     f"{1000 * v['baseline_m']:.1f} mm (label {1000 * v.get('label_baseline_m', float('nan')):.1f}, ratio "
                     f"{v.get('baseline_ratio', float('nan')):.4f}); rotation from the label's (pitch, yaw, roll) "
                     f"{v.get('rotation_vs_label_mdeg')} mdeg; free pairs scattered {ps.get('sd_mdeg')} mdeg")
    th = rep.get("thermal")
    if th:
        def pm(v, sd, f="{:+.2f}"):
            return "n/a" if v is None else f.format(v) + (" +- " + f.format(sd).lstrip("+") if sd is not None else "")
        lines.append(f"temperature (bins of {th['bin_degC']:g} degC, laws about T0 = {th['T0_degC']:g} degC):")
        for g, c in sorted(th["cameras"].items()):
            lines.append(f"  {g}: focal length {pm(c['f_ppm_per_degC'], c['sd'], '{:+.1f}')} ppm/degC over {c['T_range_degC'][0]:.0f} to "
                         f"{c['T_range_degC'][1]:.0f} degC ({len(c['bins'])} bins: " + ", ".join(f"{r['T_degC']:.0f}: {r['f_ppm']:+.0f}" for r in c["bins"]) + " ppm)")
        for k, v in sorted(th["rigs"].items()):
            if v.get("rate_mdeg_per_degC"):
                lines.append(f"  rig {k}: pitch {pm(v['rate_mdeg_per_degC'][0], v['sd'][0])}, yaw {pm(v['rate_mdeg_per_degC'][1], v['sd'][1])}, "
                             f"roll {pm(v['rate_mdeg_per_degC'][2], v['sd'][2])} mdeg/degC ({len(v['bins'])} bins, scatter about the lines "
                             + ", ".join("n/a" if x is None else f"{x:.1f}" for x in v["rms_mdeg"]) + " mdeg)")
    sc = rep.get("scale")
    if sc:
        lc = sc.get("label_centers_m") or {}
        lines.append(f"scale: the model times {sc['factor']:.4f} to give the {sc['family']} rigs the label baseline (ratios before: "
                     f"{sc['ratio_before_by_rig']}); label centers then off by median {lc.get('median', float('nan')):.3f} m, "
                     f"max {lc.get('max', float('nan')):.3f} m")
    v = rep.get("verdict")
    if v:
        some = [sd for sd, ok in (v.get("sides") or {}).items() if ok]
        lines.append("Navcam verdict: " + ("OK to promote" if v["ok"] else f"side {', '.join(some)} OK to promote, the rest not" if some else "NOT promoted"))
        for c in v["checks"]:
            if not c["ok"]:
                lines.append(f"  fails: {c['what']} = {c['value']} (limit {c['limit']})")
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
    ap.add_argument("step", choices=("waypoints", "fetch", "prepare", "align", "all", "where", "gui", "finish", "promote",
                                     "significance"),
                    help="where: print the block's folder and stop; gui: write open_in_colmap.bat into it and stop; "
                         "finish (0.99.16): the stages after the mapper again, from sparse\\0 (no matching); promote: "
                         "the block's Navcam models into the camera models in use, when its verdict is ok")
    ap.add_argument("--preset", default="yellowknife", choices=sorted(PRESETS))
    ap.add_argument("--data", default=DEFAULT_DATA, help=f"the MSL archive on disk (default {DEFAULT_DATA})")
    ap.add_argument("--work", default=None, help=f"the block's folder (default {DEFAULT_ROOT}\\msl_sol_<first sol>_<block>)")
    ap.add_argument("--block", default=None, help="the block's name in its folder (default the preset's: yellowknife_bay)")
    ap.add_argument("--sols", nargs=2, type=int, default=None, help="FIRST LAST (default: the preset's)")
    ap.add_argument("--side", default=None, help="A, B or AB (default: the preset's)")
    ap.add_argument("--list", action="store_true", help="fetch: list what would come, download nothing")
    ap.add_argument("--camera", default="opencv5", choices=("opencv5", "fisheye"))
    ap.add_argument("--mask", default="cnn", choices=("both", "cnn", "mxy"),
                    help="the mask (default cnn, 0.99.16: the mask CNN alone, kept in the images' alpha channel; both / mxy "
                         "also fetch and use MIPL's rover masks)")
    ap.add_argument("--families", default=None, help="N Navcams, F front Hazcams, R rear Hazcams (default the preset's: NF)")
    ap.add_argument("--label-cameras", action="store_true", help="prepare: start from the label cameras even when "
                    "promoted MSL models are in the camera models in use")
    ap.add_argument("--no-rigs", action="store_true", help="align: every image on its own (no stereo rigs)")
    ap.add_argument("--no-pp-check", action="store_true", help="align: skip the check with free principal points")
    ap.add_argument("--no-thermal", action="store_true", help="align: skip the temperature stage (f(T), rig rotation (T))")
    ap.add_argument("--folds", type=int, default=4, help="significance: jackknife folds (by sol)")
    ap.add_argument("--scale", default="baseline", choices=("baseline", "labels"),
                    help="align: the model's scale from the Navcam rigs' label baseline (default) or the label positions")
    ap.add_argument("--promote-if-ok", action="store_true", help="align / finish: promote the Navcam models when the verdict is ok")
    ap.add_argument("--force", action="store_true", help="promote: past a failed verdict")
    ap.add_argument("--dry-run", action="store_true", help="promote: check and show, copy nothing")
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
    if a.families:
        preset["families"] = a.families.upper()
    data = Path(a.data)
    preset["name"] = block_folder_name(preset["sols"][0], a.block or preset["block"])
    work = Path(a.work or blocks_root() / preset["name"])
    if a.step == "where":                                  # 0.99.12: the batch file asks for the folder
        print(work)
        return 0
    if a.step == "gui":                                    # 0.99.15: the COLMAP launcher for a block made before
        if not Path(work).is_dir():
            print(f"ERROR: no block folder {work}", file=sys.stderr)
            return 1
        print(f"written: {F.write_open_in_colmap(work)}")
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
        rep = F.fetch(data, sols, None, preset["side"], None, a.list, preset["families"], mxy=a.mask != "cnn")
        (work / f"fetch{'_list' if a.list else ''}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        found = sum(1 for v in rep["sols"].values() if v.get("found"))
        n = sum(v.get("products", 0) for v in rep["sols"].values())
        print(f"fetch: {found} of {len(rep['sols'])} sols found, {n} products (about {rep.get('est_mb', 0) / 1000:.1f} GB), "
              f"{rep['downloaded']} files downloaded ({rep['bytes'] / 1e9:.2f} GB), {rep['kept']} already there, "
              f"{len(rep['failed'])} failed")
        hz = [v for k, v in rep["sols"].items() if str(k).endswith("hazcam")]
        if hz:
            print(f"  Hazcam volume: {sum(1 for v in hz if v.get('found'))} of {len(hz)} sols found, "
                  f"{sum(v.get('products', 0) for v in hz)} products")
            if not any(v.get("found") for v in hz):
                print(f"  WARNING: no Hazcam folder answered - the block will hold Navcams only (addresses tried in {work})")
        if not found:
            print(f"no sol folder answered - the addresses tried are in {work}")
            return 1
        if a.list:
            return 0
    if a.step in ("prepare", "all"):
        man = prepare(work, data, preset, a.camera, a.mask, a.mask_model, a.radius_m, a.pair_distance_m,
                      a.pairs_per_image, wp, a.limit, a.label_cameras)
        print(f"prepare: {len(man['images'])} frames, {man['pairs']} pairs in {work}")
    if a.step in ("align", "all"):
        rep = align(work, a.max_features, max_error_m=a.max_error_m, refine=not a.no_refine, rigs=not a.no_rigs,
                    pp_check=not a.no_pp_check, scale=a.scale, thermal=not a.no_thermal)
        rc = 0 if rep.get("ok") else 1
        print((work / "report.txt").read_text(encoding="utf-8") if (work / "report.txt").is_file() else rep)
    if a.step == "finish":
        import pycolmap
        man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        rep = finish(pycolmap.Reconstruction(str(work / "sparse" / "0")), man, work, refine=not a.no_refine,
                     max_error_m=a.max_error_m, rigs=not a.no_rigs, pp_check=not a.no_pp_check, scale=a.scale,
                     thermal=not a.no_thermal)
        rc = 0 if rep.get("ok") else 1
        print((work / "report.txt").read_text(encoding="utf-8"))
    if a.step == "significance":                           # 0.99.17: are p1, p2, k3 needed?
        significance(work, folds=a.folds)
        print((work / "significance.txt").read_text(encoding="utf-8"))
        return 0
    if a.step == "promote" or (a.promote_if_ok and a.step in ("align", "all", "finish") and rc == 0):
        res = promote(work, force=a.force, dry_run=a.dry_run)
        if res.get("promoted") or res.get("dry_run"):
            print(f"{'would promote' if res.get('dry_run') else 'promoted'} into {res['folder']}: " + ", ".join(sorted(res["files"])))
        else:
            print(f"not promoted: {res['reason']}")
            rc = rc or (1 if a.step == "promote" else 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
