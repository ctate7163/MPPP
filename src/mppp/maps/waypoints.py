"""
Waypoint rows for the maps (MPPP v0p80, :mod:`mppp.maps`).

:func:`waypoint_rows` turns the M2020 waypoint GeoJSON (``mppp.load_waypoints()``, or any file such as
``M20_waypoints_sol2009_all.json``) into flat rows: sol, site, drive, easting / northing (m, the map's projected
frame), lon / lat, the localisation kind and whether the stop has a Navcam 360 panorama.  :func:`lonlat_fit` is the
exact linear relation between (easting, northing) and (lon, lat) of a file, for points that have only one of them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def load(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """The waypoint GeoJSON: ``path``, else the user-cache / packaged copy (:func:`mppp.waypoints.load_waypoints`)."""
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    from ..waypoints import load_waypoints
    return load_waypoints()


def _kind(note: Any) -> str:
    n = str(note or "").lower()
    if "end-of-drive" in n or "first localization" in n:
        return "end_of_drive"
    if "mid-drive" in n:
        return "mid_drive"
    if "interpolated" in n:
        return "interpolated"
    if "site increment" in n:
        return "site_increment"
    return n or "unknown"


def waypoint_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per waypoint, in file order (sol order): {"sol", "site", "drive", "rmc", "E", "N", "lon", "lat", "kind",
    "pano" (a Navcam 360 panorama at the stop), "end_of_drive" (kind end_of_drive)}.  Waypoints without easting /
    northing are left out."""
    out = []
    for f in data.get("features") or []:
        p = f.get("properties", f)
        try:
            E, N = float(p["easting"]), float(p["northing"])
            sol = int(p["sol"])
        except (KeyError, TypeError, ValueError):
            continue
        ims = p.get("images") or []
        pano = any(bool(i.get("isPanoramic")) or "pano" in str(i.get("name", "")).lower() for i in ims if isinstance(i, dict))
        kind = _kind(p.get("Note"))
        out.append({"sol": sol, "site": int(p.get("site") or 0), "drive": int(p.get("drive") or 0),
                    "rmc": p.get("RMC") or f"{p.get('site')}_{p.get('drive')}", "E": E, "N": N,
                    "lon": float(p["lon"]) if p.get("lon") is not None else None,
                    "lat": float(p["lat"]) if p.get("lat") is not None else None,
                    "kind": kind, "pano": bool(pano), "end_of_drive": kind == "end_of_drive"})
    return out


def lonlat_fit(rows: Sequence[Dict[str, Any]]):
    """(to_lonlat, to_en): the least-squares affine maps between (E, N) and (lon, lat) of the rows that have both
    (exact to the file's precision over the crater: the map frame is an equirectangular projection)."""
    r = [x for x in rows if x.get("lon") is not None and x.get("lat") is not None]
    if len(r) < 3:
        raise ValueError("fewer than 3 waypoints with both easting/northing and lon/lat")
    EN = np.array([[x["E"], x["N"], 1.0] for x in r])
    LL = np.array([[x["lon"], x["lat"], 1.0] for x in r])
    A, *_ = np.linalg.lstsq(EN, LL[:, :2], rcond=None)
    B, *_ = np.linalg.lstsq(LL, EN[:, :2], rcond=None)

    def to_lonlat(E, N) -> Tuple[float, float]:
        v = np.array([E, N, 1.0]) @ A
        return float(v[0]), float(v[1])

    def to_en(lon, lat) -> Tuple[float, float]:
        v = np.array([lon, lat, 1.0]) @ B
        return float(v[0]), float(v[1])
    return to_lonlat, to_en
