"""
mppp_error.waypoints -- build Stations from an M2020 traverse GeoJSON.

Rewrite of the v1 `mars20_waypoints.py` with these changes:

  * TRUE ELEVATIONS ARE PRESERVED.  v1's make_report did
    `stations_xyz[:,2] = 1.9`, discarding relative station elevations.  On any
    slope that removes real network geometry -- and elevation diversity between
    stations is one of the few things that breaks the collinear-drive
    degeneracy.  Camera height above local ground is now a separate parameter
    added to the terrain elevation.
  * PATH LENGTH is accumulated along the traverse (in drive order), not
    straight-line from the anchor, so PoseModel('telemetry') sees a realistic
    VO drift distance.
  * site_drive_for_sol returns (site, drive, matched_sol).  v1's caller
    unpacked the third value as `anchor_method`, which was a naming bug.
  * Station selection radius is decoupled from the grid extent.

Azimuth convention: degrees clockwise from North (compass), matching the
rotation used to place the camera offset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from collections import defaultdict
from typing import Union, Optional, List, Dict, Tuple, Sequence

from .core import Station, Instrument, OcclusionMask, MASTCAM_Z_34, DEFAULT_ROVER_MASK

__all__ = ["load_featurecollection", "site_drive_for_sol", "build_stations"]


def load_featurecollection(src: Union[str, dict, None] = None) -> dict:
    """GeoJSON FeatureCollection from a path, a dict, or None = the waypoints MPPP uses
    (:func:`mppp.load_waypoints`: the user-cache copy, else the packaged snapshot; v0p13)."""
    if src is None:
        from ..waypoints import load_waypoints
        src = load_waypoints()
    fc = json.load(open(src)) if isinstance(src, (str, Path)) else src
    if fc.get("type") != "FeatureCollection":
        raise ValueError("input must be a GeoJSON FeatureCollection")
    if not fc.get("features"):
        raise ValueError("FeatureCollection has no features")
    return fc


def _rot_z_cw(az_rad: float) -> np.ndarray:
    """Rotation about +Z by a CLOCKWISE angle (compass convention)."""
    c, s = np.cos(az_rad), np.sin(az_rad)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _az_rad(props: dict) -> float:
    if props.get("yaw_rad") is not None:
        return float(props["yaw_rad"])
    if props.get("yaw") is not None:
        return np.deg2rad(float(props["yaw"]))
    raise KeyError("waypoint properties need 'yaw_rad' or 'yaw'")


def _ints(p: dict) -> Optional[Tuple[int, int, int]]:
    try:
        return int(p["sol"]), int(p["site"]), int(p["drive"])
    except (TypeError, ValueError, KeyError):
        return None


def site_drive_for_sol(fc: dict, sol: int, fallback: str = "nearest",
                       tie_preference: str = "lower") -> Tuple[int, int, int]:
    """Return (site, drive, matched_sol). sol=-1 selects the latest sol present."""
    by_sol: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for feat in fc["features"]:
        rec = _ints(feat.get("properties") or {})
        if rec:
            by_sol[rec[0]].append((rec[1], rec[2]))
    if not by_sol:
        raise ValueError("no valid (sol, site, drive) entries")

    sols = sorted(by_sol)
    if sol == -1:
        sol = sols[-1]
    if sol in by_sol:
        site, drive = max(by_sol[sol])
        return site, drive, sol

    if fallback == "none":
        raise ValueError(f"sol {sol} not found and fallback='none'")
    if fallback == "previous":
        cands = [s for s in sols if s < sol]
    elif fallback == "next":
        cands = [s for s in sols if s > sol]
    elif fallback == "nearest":
        d = min(abs(s - sol) for s in sols)
        cands = [s for s in sols if abs(s - sol) == d]
        cands = [min(cands) if tie_preference == "lower" else max(cands)]
    else:
        raise ValueError("fallback must be none|nearest|previous|next")
    if not cands:
        raise ValueError(f"no fallback sol available from {sol}")
    matched = max(cands) if fallback == "previous" else min(cands)
    site, drive = max(by_sol[matched])
    return site, drive, matched


def _last_per_sol_features(feats: Sequence[dict]) -> List[dict]:
    """Keep only the highest (site, drive) waypoint for each sol."""
    best: Dict[int, Tuple[Tuple[int, int], dict]] = {}
    others = []
    for f in feats:
        rec = _ints(f.get("properties") or {})
        if rec is None:
            others.append(f)
            continue
        sol, site, drive = rec
        key = (site, drive)
        if sol not in best or key > best[sol][0]:
            best[sol] = (key, f)
    return [v[1] for v in best.values()] + others


def _last_per_sol(props_by_rmc: dict, rmcs):
    """
    Keep only the LAST rover position of each sol.  A sol with a mid-drive
    localization and an end-of-drive localization has two waypoints but one
    imaging station (the 360 is taken where the rover stops); the mid-drive
    entry is dropped.  'Last' = highest drive count within the sol.
    """
    best = {}
    for r in rmcs:
        pr = props_by_rmc[r]
        sol = pr["sol"]; drive = int(str(r).split("_")[1])
        if sol not in best or drive > best[sol][0]:
            best[sol] = (drive, r)
    keep = {v[1] for v in best.values()}
    return [r for r in rmcs if r in keep]


def stations_from_rmcs(
    fc: Union[str, dict],
    rmcs: Sequence[str],
    anchor_rmc: Optional[str] = None,
    instrument: Instrument = MASTCAM_Z_34,
    mask: Optional[OcclusionMask] = DEFAULT_ROVER_MASK,
    camera_height_m: float = 1.9,
    last_per_sol: bool = True,
    elev_field: str = "elev_geoid",
    use_odometry_path: bool = True,
) -> Tuple[List[Station], dict]:
    """
    Build Station objects for an EXPLICIT, hand-picked list of RMC waypoints
    ("{site}_{drive}" strings), rather than a radius-based selection.

    Use this when you want specific real waypoints -- e.g. a small, non-
    collinear subset near a landing site -- rather than everything within a
    radius (which, for a real drive path, often pulls in near-duplicate
    revisits alongside the ones you actually want; `build_stations`'s radius
    selection can't distinguish them).

    anchor_rmc defaults to rmcs[0]. The frame is centred on the anchor's
    GROUND position (see the frame-convention note in `build_stations`).

    use_odometry_path=True sets each station's path_m from the ACTUAL rover
    odometry (`dist_total_m`, cumulative driven distance from mission start),
    relative to the anchor -- real telemetry, not a straight-line
    reconstruction between waypoints. This is what PoseModel('telemetry')
    should be driven from when real data is available. Falls back to
    straight-line-from-anchor if `dist_total_m` is absent.
    """
    fc = load_featurecollection(fc)
    by_rmc = {}
    for f in fc["features"]:
        p = f.get("properties") or {}
        if "RMC" in p:
            by_rmc[p["RMC"]] = p
        else:
            rec = _ints(p)
            if rec:
                by_rmc[f"{rec[1]}_{rec[2]}"] = p

    missing = [r for r in rmcs if r not in by_rmc]
    if not missing and last_per_sol:
        rmcs = _last_per_sol(by_rmc, list(rmcs))
    if missing:
        raise ValueError(f"RMC(s) not found in waypoint file: {missing}")

    anchor_rmc = anchor_rmc or rmcs[0]
    ap = by_rmc[anchor_rmc]
    ax, ay = float(ap["easting"]), float(ap["northing"])
    az_anchor = float(ap[elev_field])
    anchor_ground = np.array([ax, ay, az_anchor])
    anchor_odo = float(ap.get("dist_total_m", 0.0))

    stations: List[Station] = []
    for rmc in rmcs:
        p = by_rmc[rmc]
        a = _az_rad(p)
        z = float(p[elev_field])
        pos = np.array([float(p["easting"]), float(p["northing"]), z])
        pos[2] += camera_height_m
        rel = pos - anchor_ground

        if use_odometry_path and "dist_total_m" in p:
            path = abs(float(p["dist_total_m"]) - anchor_odo)
        else:
            path = float(np.linalg.norm(rel[:2]))

        stations.append(Station(
            xyz=rel, az_deg=float(np.degrees(a) % 360.0),
            name=f"Sol {p.get('sol','?')} | {rmc}",
            instrument=instrument, mask=mask, path_m=path,
            is_anchor=(rmc == anchor_rmc),
        ))

    anchor_site, anchor_drive = (int(x) for x in anchor_rmc.split("_", 1))
    info = dict(anchor_name=anchor_rmc, anchor_rmc=anchor_rmc,
               anchor_sol=int(ap.get("sol", -1)),
               anchor_site=anchor_site, anchor_drive=anchor_drive,
               anchor_ground_xyz_abs=anchor_ground.tolist(),
               camera_height_m=camera_height_m, n_stations=len(stations),
               rmcs=list(rmcs))
    return stations, info


def build_stations(
    fc: Union[str, dict],
    anchor_site: int = -1,
    anchor_drive: int = -1,
    select_radius_m: float = 50.0,
    instrument: Instrument = MASTCAM_Z_34,
    mask: Optional[OcclusionMask] = DEFAULT_ROVER_MASK,
    cam_offset_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    camera_height_m: float = 1.9,
    elev_field: str = "elev_geoid",
    last_per_sol: bool = True,
    flatten_elevation: bool = False,
) -> Tuple[List[Station], dict]:
    """
    Build Station objects from a traverse, centred on the anchor.

    cam_offset_xyz  : mast offset in the ROVER body frame (forward, left, up),
                      rotated into the world frame by the rover heading.
    camera_height_m : additional height of the camera above the waypoint
                      elevation, applied on top of cam_offset_xyz[2].
    flatten_elevation : if True, reproduce v1's behaviour of forcing all
                      stations to a common height.  Provided only for
                      back-comparison; leave False for real work.

    Returns (stations, info) where info carries anchor metadata.
    """
    fc = load_featurecollection(fc)
    feats = fc["features"]

    # ---- resolve the anchor ------------------------------------------------
    by_site: Dict[int, List[Tuple[int, dict]]] = defaultdict(list)
    for f in feats:
        rec = _ints(f.get("properties") or {})
        if rec:
            by_site[rec[1]].append((rec[2], f))
    if not by_site:
        raise ValueError("no waypoints with site/drive")
    site = max(by_site) if anchor_site == -1 else anchor_site
    if site not in by_site:
        raise ValueError(f"site {anchor_site} not present")
    recs = sorted(by_site[site], key=lambda t: t[0])
    drives = [d for d, _ in recs]
    drive = drives[-1] if anchor_drive == -1 else \
        min(drives, key=lambda d: abs(d - anchor_drive))
    anchor_feat = next(f for d, f in recs if d == drive)

    ap = anchor_feat["properties"]
    ax, ay = float(ap["easting"]), float(ap["northing"])
    az_anchor = float(ap[elev_field])

    off = np.asarray(cam_offset_xyz, dtype=float)

    # FRAME CONVENTION: the output frame is centred on the anchor's GROUND
    # position, not its camera position.  The evaluation grid therefore sits at
    # z = 0 (local ground) and cameras sit at +camera_height_m above it.
    # Centring on the camera instead puts the cameras in the ground plane, every
    # emission angle goes to 90 deg, and nothing is visible anywhere.
    anchor_ground = np.array([ax, ay, az_anchor])

    def station_xyz(p: dict) -> Tuple[np.ndarray, float]:
        a = _az_rad(p)
        world = _rot_z_cw(a) @ off
        z = float(p[elev_field])
        pos = np.array([float(p["easting"]), float(p["northing"]), z]) + world
        pos[2] += camera_height_m
        return pos, a

    # ---- select and order --------------------------------------------------
    cand = _last_per_sol_features(feats) if last_per_sol else list(feats)
    keep = []
    for f in cand:
        p = f.get("properties") or {}
        try:
            d_xy = np.hypot(float(p["easting"]) - ax, float(p["northing"]) - ay)
        except (TypeError, ValueError, KeyError):
            continue
        if d_xy <= select_radius_m:
            keep.append(f)
    if not keep:
        raise ValueError("no waypoints within select_radius_m of the anchor")

    def sortkey(f):
        r = _ints(f["properties"]) or (0, 0, 0)
        return r
    keep.sort(key=sortkey)

    # ---- build -------------------------------------------------------------
    stations: List[Station] = []
    prev = None
    cum = 0.0
    anchor_key = _ints(ap)
    for f in keep:
        p = f["properties"]
        pos, a = station_xyz(p)
        if prev is not None:
            cum += float(np.linalg.norm(pos[:2] - prev[:2]))
        prev = pos
        rel = pos - anchor_ground
        if flatten_elevation:
            rel[2] = camera_height_m
        rec = _ints(p)
        sol = rec[0] if rec else None
        name = (f"Sol {sol} | site {rec[1]} | drive {rec[2]}" if rec
                else p.get("RMC", "unknown"))
        stations.append(Station(
            xyz=rel, az_deg=float(np.degrees(a) % 360.0), name=name,
            instrument=instrument, mask=mask, path_m=cum,
            is_anchor=(rec == anchor_key),
        ))

    if not any(s.is_anchor for s in stations):
        # anchor fell outside the per-sol filter; re-reference path lengths to
        # the closest station so telemetry pose is still meaningful
        i = int(np.argmin([np.linalg.norm(s.xyz) for s in stations]))
        stations[i].is_anchor = True
    a_i = next(i for i, s in enumerate(stations) if s.is_anchor)
    base = stations[a_i].path_m or 0.0
    for s in stations:
        s.path_m = abs((s.path_m or 0.0) - base)

    info = dict(anchor_name=ap.get("RMC", f"{site}_{drive}"),
                anchor_site=site, anchor_drive=drive,
                anchor_ground_xyz_abs=anchor_ground.tolist(),
                camera_height_m=camera_height_m,
                n_stations=len(stations))
    return stations, info
