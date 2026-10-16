"""
Terrain backgrounds for the maps (MPPP v0p80, :mod:`mppp.maps`): a hillshade from a DEM GeoTIFF in the waypoint map
frame (the HiRISE DTMs of the Mars 2020 / MSR TRN; listed in ``M20_DEMS`` or given by path).  Needs rasterio; without
it, or without a DEM, the maps are drawn on a plain background.  The DEM must be in the same projected frame as the
waypoints' easting / northing (the TRN products are; a check of the corners against the waypoints is the user's).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def dem_candidates() -> List[Path]:
    """DEM files from ``M20_DEMS`` (paths separated by ``os.pathsep``) that exist; MSR TRN first by name."""
    v = os.environ.get("M20_DEMS") or ""
    out = [Path(p) for p in v.split(os.pathsep) if p and Path(p).is_file()]
    return sorted(out, key=lambda p: (0 if "msr" in p.name.lower() else 1, p.name))


def hillshade(z: np.ndarray, dx: float, dy: float, azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
              exaggeration: float = 1.0) -> np.ndarray:
    gy, gx = np.gradient(z * exaggeration, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth_deg), np.radians(altitude_deg)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def hillshade_extent(dem: PathLike, bounds: Sequence[float]) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, ...]]]:
    """(hillshade image, (xmin, xmax, ymin, ymax)) of the DEM cropped to ``bounds`` (E0, E1, N0, N1); (None, None)
    without rasterio or overlap."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        return None, None
    with rasterio.open(str(dem)) as src:
        e0, e1, n0, n1 = bounds
        try:
            w = from_bounds(e0, n0, e1, n1, transform=src.transform).round_offsets().round_lengths()
            z = src.read(1, window=w, masked=True).astype(float).filled(np.nan)
        except Exception:                                          # noqa: BLE001
            return None, None
        if z.size == 0:
            return None, None
        t = src.window_transform(w)
        z = np.where(np.isfinite(z), z, np.nanmedian(z))
        img = hillshade(z, abs(t.a), abs(t.e))
        ext = (t.c, t.c + t.a * z.shape[1], t.f + t.e * z.shape[0], t.f)
        return img, ext
