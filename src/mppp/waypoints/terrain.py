"""
Terrain for the site maps (0.90): quantized-mesh tiles (Cesium ``.terrain``, as served for the Perseverance traverse
viewers) read into a longitude-latitude height grid and shaded.  The package ships no tiles; pass a folder of
``<z>_<x>_<y>.terrain`` files (``MPPP_TERRAIN``), such as the HiRISE-DTM-based Jezero tiles of ``studies/m20_sites``.
A 1 m HiRISE DTM (GeoTIFF) can be sampled the same way through :func:`terrain_grid` with ``dtm=``.
"""
from __future__ import annotations

import glob
import gzip
import os
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

MARS_RADIUS_M = 3389500.0


def read_qmesh(path: os.PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, h) of a quantized-mesh tile: u, v in 0..1 across the tile, h in metres."""
    b = Path(path).read_bytes()
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    hdr = struct.unpack("<3d2f4d3d", b[:88])
    minh, maxh = hdr[3], hdr[4]
    n = struct.unpack("<I", b[88:92])[0]
    arr = np.frombuffer(b[92:92 + 6 * n], dtype="<u2").reshape(3, n).astype(np.int32)
    zz = lambda a: (a >> 1) ^ (-(a & 1))
    u, v, h = (np.cumsum(zz(arr[i])) for i in range(3))
    return u / 32767.0, v / 32767.0, minh + h / 32767.0 * (maxh - minh)


def terrain_points(folder: os.PathLike, box: Sequence[float], levels: Optional[Sequence[int]] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All tile vertices (lon, lat, h) of ``folder`` within ``box`` = (lon0, lon1, lat0, lat1), de-duplicated;
    ``levels`` (zoom) default to every level present."""
    U, V, H = [], [], []
    for fn in glob.glob(str(Path(folder) / "*.terrain")):
        try:
            z, x, y = map(int, Path(fn).name[:-8].split("_"))
        except ValueError:
            continue
        if levels is not None and z not in levels:
            continue
        deg = 180.0 / 2 ** z
        lon0, lat0 = -180 + x * deg, -90 + y * deg
        if lon0 + deg < box[0] - deg or lon0 > box[1] + deg or lat0 + deg < box[2] - deg or lat0 > box[3] + deg:
            continue
        u, v, h = read_qmesh(fn)
        lo, la = lon0 + u * deg, lat0 + v * deg
        m = (lo > box[0] - deg) & (lo < box[1] + deg) & (la > box[2] - deg) & (la < box[3] + deg)
        U.append(lo[m]); V.append(la[m]); H.append(h[m])
    if not U:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    U, V, H = np.concatenate(U), np.concatenate(V), np.concatenate(H)
    k = np.round(U * 1e5).astype(np.int64) * 10 ** 8 + np.round(V * 1e5).astype(np.int64)
    _, i = np.unique(k, return_index=True)
    return U[i], V[i], H[i]


def terrain_grid(folder: os.PathLike, box: Sequence[float], step_deg: float, levels: Optional[Sequence[int]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height grid (lon axis, lat axis, Z[lat, lon]) over ``box`` at ``step_deg`` from the tiles (linear
    interpolation; NaN where there is no tile)."""
    from scipy.interpolate import griddata
    U, V, H = terrain_points(folder, box, levels)
    gx = np.arange(box[0], box[1], step_deg)
    gy = np.arange(box[2], box[3], step_deg)
    if len(U) < 3:
        return gx, gy, np.full((len(gy), len(gx)), np.nan)
    GX, GY = np.meshgrid(gx, gy)
    return gx, gy, griddata((U, V), H, (GX, GY), method="linear")


def shaded_relief(gx: np.ndarray, gy: np.ndarray, Z: np.ndarray, vert_exag: float = 3.0,
                  cmap: str = "gist_earth", clip_pct: Tuple[float, float] = (1.0, 99.0)) -> Dict[str, Any]:
    """RGB hillshade of a height grid (muted tint, light from the north-west) for ``imshow``: {"rgb", "extent",
    "norm", "cmap"}."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    dx = np.deg2rad(gx[1] - gx[0]) * MARS_RADIUS_M * np.cos(np.deg2rad(gy.mean()))
    dy = np.deg2rad(gy[1] - gy[0]) * MARS_RADIUS_M
    fill = np.nanmean(Z) if np.isfinite(Z).any() else 0.0
    Zf = np.where(np.isnan(Z), fill, Z)
    hs = LightSource(azdeg=315, altdeg=40).hillshade(Zf, vert_exag=vert_exag, dx=dx, dy=dy)
    cm = plt.get_cmap(cmap)
    norm = plt.Normalize(np.nanpercentile(Z, clip_pct[0]) if np.isfinite(Z).any() else 0,
                         np.nanpercentile(Z, clip_pct[1]) if np.isfinite(Z).any() else 1)
    rgb = cm(norm(Zf))[..., :3]
    rgb = 0.4 * rgb + 0.6 * 0.88
    out = rgb * (0.35 + 0.65 * hs[..., None])
    out = np.where(np.isnan(Z)[..., None], 0.93, out)
    return {"rgb": out, "extent": [gx[0], gx[-1], gy[0], gy[-1]], "norm": norm, "cmap": cm}
