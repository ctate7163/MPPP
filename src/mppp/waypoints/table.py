"""
Mars 2020 rover waypoints (localised site/drive positions).

Source: the M2020 mission "waypoints" GeoJSON.  A snapshot ships with the
package (``mppp/data/M20_waypoints.json``) so that processing is reproducible
and works offline; ``load_waypoints(refresh=True)`` downloads the current file
into the user cache (``mppp.paths.cache_dir()``), which is then used by
default.  The SHA-256 of the file used is recorded in every run manifest.

The error model's ``waypoints`` module builds stations from the same file
(one loader for both).
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..paths import cache_dir, data_dir

PathLike = Union[str, Path]

WAYPOINTS_URL = "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json"
WAYPOINTS_FILE = "M20_waypoints.json"
MARS_RADIUS_M = 3396190.0          # IAU 2000 Mars sphere used by the M2020 maps


def snapshot_path() -> Path:
    """The waypoint snapshot shipped with the package (frozen; the error-model self-tests rely on it)."""
    return data_dir() / WAYPOINTS_FILE


def cached_path() -> Path:
    return cache_dir() / WAYPOINTS_FILE


def load_waypoints(path: Optional[PathLike] = None, refresh: bool = False,
                   url: str = WAYPOINTS_URL, timeout: float = 30.0) -> Dict[str, Any]:
    """
    The waypoint GeoJSON as a dict (``_mppp_source`` records path, SHA-256, count).

    * ``path`` given: that file (``refresh`` downloads into it first).
    * default: the user-cache copy if one exists, else the packaged snapshot.
    * ``refresh=True``: download the current file from ``url`` into the cache
      (or ``path``); if the download fails, a warning is issued and the
      existing copy is used.
    """
    target = Path(path) if path is not None else cached_path()
    if refresh:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                raw = r.read()
            json.loads(raw.decode("utf-8"))                       # validate before caching
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        except Exception as e:                                     # noqa: BLE001
            warnings.warn(f"could not download the waypoints from {url} ({type(e).__name__}: {e}); "
                          f"using the existing copy")
    if not target.is_file():
        if path is not None:
            raise FileNotFoundError(f"waypoint file not found: {target}")
        target = snapshot_path()
    raw = target.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not data.get("features"):
        raise ValueError(f"No 'features' in waypoint file {target}")
    data["_mppp_source"] = {"path": str(target), "sha256": hashlib.sha256(raw).hexdigest(),
                            "n_features": len(data["features"]),
                            "packaged_snapshot": target.resolve() == snapshot_path().resolve()}
    return data


def _props(feature: Dict[str, Any]) -> Dict[str, Any]:
    return feature.get("properties", feature)


def _int(props: Dict[str, Any], key: str) -> Optional[int]:
    try:
        return int(props.get(key))
    except (TypeError, ValueError):
        return None


def _en(props: Dict[str, Any]) -> Optional[tuple]:
    try:
        return float(props["easting"]), float(props["northing"])
    except (KeyError, TypeError, ValueError):
        return None


def waypoint_for_site_drive(data: Dict[str, Any], site: int, drive: int,
                            exact: bool = False) -> Optional[Dict[str, Any]]:
    """
    Waypoint *feature* for (site, drive).

    1. exact (site, drive): the LAST such feature;
    2. if ``exact`` is False: the feature of that site with the nearest drive
       (ties: larger drive, then later entry);
    3. ``None`` if nothing qualifies.  Always returns the full feature
       (the legacy function returned the feature in case 1 but its
       'properties' in case 2).
    """
    feats = [(i, f) for i, f in enumerate(data.get("features", []))
             if _int(_props(f), "site") == site]
    if not feats:
        return None
    hits = [f for _, f in feats if _int(_props(f), "drive") == drive]
    if hits:
        return hits[-1]
    if exact:
        return None
    cands = [(abs(drive - _int(_props(f), "drive")), -_int(_props(f), "drive"), -i, f)
             for i, f in feats if _int(_props(f), "drive") is not None]
    if not cands:
        return feats[-1][1]
    cands.sort(key=lambda c: c[:3])
    return cands[0][3]


def waypoints_within_radius(data: Dict[str, Any], anchor_sol: int, radius: float) -> List[Dict[str, Any]]:
    """
    All waypoint features within ``radius`` metres (planar E-N) of the anchor:
    the last waypoint of ``anchor_sol``; ``-1`` = last waypoint of the highest sol.
    Unlike the legacy function, an anchor sol with no waypoint raises an error
    instead of silently using the latest waypoint.
    """
    feats = data.get("features", [])
    if not feats:
        raise ValueError("No waypoints.")
    sols = [_int(_props(f), "sol") for f in feats]
    if anchor_sol == -1:
        anchor_sol = max(s for s in sols if s is not None)
    idx = [i for i, s in enumerate(sols) if s == anchor_sol]
    if not idx:
        earlier = [s for s in sols if s is not None and s < anchor_sol]
        if not earlier:
            raise ValueError(f"No waypoint at or before sol {anchor_sol}.")
        prev = max(earlier)                               # rover was parked since then
        idx = [i for i, s in enumerate(sols) if s == prev]
    anchor = _en(_props(feats[idx[-1]]))
    if anchor is None:
        raise ValueError("Anchor waypoint lacks easting/northing.")
    out = []
    for f in feats:
        en = _en(_props(f))
        if en is not None and math.hypot(en[0] - anchor[0], en[1] - anchor[1]) <= float(radius):
            out.append(f)
    return out


def stations_near(data: Dict[str, Any], stations: Iterable[Tuple[int, int]], radius_m: float = 5.0) -> List[Dict[str, Any]]:
    """
    Waypoint stations within ``radius_m`` metres (planar E-N) of any of ``stations``
    (site, drive pairs; each placed at its waypoint, or the nearest drive of its
    site when the exact one is not in the table).  Returns one row per (site,
    drive) of the table, anchors included: ``site``, ``drive``, ``sol``,
    ``distance_m`` (to the nearest anchor), ``nearest`` (that anchor) and
    ``anchor`` (True for the given stations).  Used to add the images of later
    or earlier visits to the same spot.
    """
    anchors = []
    for sd in {(int(a), int(b)) for a, b in stations}:
        f = waypoint_for_site_drive(data, sd[0], sd[1])
        en = _en(_props(f)) if f is not None else None
        if en is not None:
            anchors.append((sd, en))
    if not anchors:
        return []
    given = {sd for sd, _ in anchors}
    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for f in data.get("features", []):
        p = _props(f)
        site, drive, en = _int(p, "site"), _int(p, "drive"), _en(p)
        if site is None or drive is None or en is None:
            continue
        d, near = min((math.hypot(en[0] - a[0], en[1] - a[1]), sd) for sd, a in anchors)
        if d <= float(radius_m) and ((site, drive) not in best or d < best[(site, drive)]["distance_m"]):
            best[(site, drive)] = {"site": site, "drive": drive, "sol": _int(p, "sol"), "distance_m": round(d, 2),
                                   "nearest": list(near), "anchor": (site, drive) in given}
    for sd, _ in anchors:                                   # anchors without an exact table entry
        best.setdefault(sd, {"site": sd[0], "drive": sd[1], "sol": None, "distance_m": 0.0, "nearest": list(sd),
                             "anchor": True})
    return sorted(best.values(), key=lambda r: (r["site"], r["drive"]))


def lonlat_of(feature: Dict[str, Any]) -> Optional[tuple]:
    """(lon_east_deg, lat_deg) from the feature — properties first, then GeoJSON geometry."""
    p = _props(feature)
    for klon, klat in (("lon", "lat"), ("longitude", "latitude")):
        try:
            return float(p[klon]), float(p[klat])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        c = feature["geometry"]["coordinates"]
        return float(c[0]), float(c[1])
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def offset_lonlat(lon_deg: float, lat_deg: float, d_east: float, d_north: float) -> tuple:
    """Local-tangent-plane offset on the Mars sphere (valid for offsets << R)."""
    lat = lat_deg + math.degrees(d_north / MARS_RADIUS_M)
    lon = lon_deg + math.degrees(d_east / (MARS_RADIUS_M * math.cos(math.radians(lat_deg))))
    return lon, lat


# --- flat rows and the map-frame <-> lon/lat relation (used by the site/block tools) -----------------

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
