"""The station as a textured 3D surface: the terrain within 10 m and the rover (0.99.26).

Christian, 18 Nov 2026: "I do want a simplified medium poly mesh of the 10m radius area to be textured. The texture
should be 8kx8k and projected onto the terrain largely from the perspective of the viewer so that the near terrain
gets higher res than the outer area" ... "the texture projection center might be better a little above the pivot
point without the rover. Add this as a fourth mode in rpv as 3D terrain and rover. It should include the 3D rover.
The 3D rover surfaces should be textured if they are visible anywhere from 1m of the pivot and viewed by the ncams."
Answers: the center 0.5 m above the pivot; the terrain-only stereo panorama as the backdrop beyond 10 m; head motion
held within 1 m of the pivot.

**The terrain** (the rover-masked run of a station: ``rpp ods ... --station-mesh``):

* the mesh of the station folder (``mesh.ply``, without the rover model) within ``RADIUS_M`` of the mast axis,
  simplified to about ``TARGET_TRIANGLES`` triangles (quadric decimation, no smoothing);
* a ``TEXTURE_PX`` square texture.  A texel stands for a direction from the projection center C, ``CENTER_ABOVE_M``
  above the pivot: the texture's center is straight down, the distance from the center grows evenly with the angle
  from straight down (an azimuthal equidistant map; north up, east right), the disc's rim is the angle of the farthest
  vertex.  So every texel covers the same angle seen from C, and the near terrain gets the most texels per square
  meter.  A vertex's texture coordinates are its direction from C, so the disc has no seam;
* the triangles C does not see (behind a rock, an overhang) or sees almost edge on are given cells of their own in the
  texture's corners (outside the disc: a shelf packing at one texel size, the largest that fits), each colored from
  the frames that see it;
* every texel is colored by the panorama's blend (:func:`rpp.ibr.ods.shade_rays`): a disc texel along its ray from C,
  a cell texel along the surface normal onto its point.  What no frame colored is filled from the texels around it.

**The rover** (the rover-shown run: the Perseverance model of ``mesh.ply``, ``scene.rover_triangles``):

* a triangle is textured when it can be seen from somewhere within ``VIEW_RADIUS_M`` of the pivot (from the pivot and
  from ``VIEWPOINTS`` points on spheres of 0.5 and 1 m about it, those inside the model left out) - the space the
  viewer may move in - and the Navcam frames (``ROVER_FAMILIES``) color it; it gets a cell of the ``ROVER_TEXTURE_PX``
  texture, colored as the terrain's cells are.  In a Mastcam-Z run (0.99.27) the Mastcam-Z frames color it first and
  the Navcam frames where they do not (:func:`rover_families`); the terrain of such a run is the same: the Mastcam-Z
  frames are its regular sources, the Navcam and front Hazcam frames fill in as the Mastcam-Z frames' weight goes out;
* the other triangles are the rover gray of the panoramas, shaded a little by their slope.

The two runs write ``<prefix>_station_mesh_terrain.npz`` / ``.jpg`` / ``.json`` and ``<prefix>_rover_station_mesh_rover...``
beside their panoramas; :func:`assemble` puts them into one binary glTF, ``<prefix>_station_mesh.glb`` in the terrain
folder: two meshes in meters about the pivot (x east, y up, z south: the viewer's frame), each with its unlit texture.
The viewer shows it as the fourth mode, "3D terrain and rover" (:mod:`rpp.ibr.web`), over the terrain-only
panorama.
"""
from __future__ import annotations

import json
import logging
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
log = logging.getLogger("rpp.ibr")

RADIUS_M = 10.0                 # the terrain within this horizontal distance of the mast axis
CENTER_ABOVE_M = 0.5            # the terrain texture's projection center above the pivot
TARGET_TRIANGLES = 300_000      # the simplified terrain ("medium poly")
TEXTURE_PX = 8192               # the terrain texture's side
ROVER_TEXTURE_PX = 4096         # the rover texture's side
VIEW_RADIUS_M = 1.0             # the rover is textured where it can be seen from within this distance of the pivot
VIEWPOINTS = 32                 # directions of the viewpoints on each sphere (0.5 m and VIEW_RADIUS_M)
ROVER_FAMILIES = "N"            # the frames that color the rover ("viewed by the ncams")
GRAZING_COS = 0.12              # a terrain triangle C sees at more than about 83 deg from its normal gets a cell
GUTTER_PX = 2                   # texels around every cell and the disc, colored as if the surface went on
MAX_CELL_PX = 384               # a cell's longest side; a larger triangle is given coarser texels
TERRAIN_OFFSET_M = 0.05         # a cell texel's ray starts this far off the simplified surface (it may lie below
ROVER_OFFSET_M = 0.005          # the measured one); the rover model is exact
INSIDE_HITS = 5                 # a viewpoint whose rays meet the rover model in this many of six directions is inside it
JPEG_QUALITY = 90
GRAY_STRIP = (4, 64)            # the rover gray's cell: width, height (the shade by slope down the strip)
SUFFIX = "station_mesh.glb"
PART = {"terrain": "station_mesh_terrain", "rover": "station_mesh_rover", "scape": "station_scape_terrain"}
# 0.99.32 (Christian, 18 Nov 2026: "expanding the mesh to the horizon with a low poly mesh ... continuous from below the
# rover to infinity ... One square uv mapping could go from the point below the prime pivot point, and extend outwards
# to the horizon which lies on a circle about 90% of the max size of circle that will fit in the square. Beyond that
# horizon circle, the uv ... goes to the sky"): the fifth mode's surface - see :func:`horizon_shell`
SCAPE = True                    # the masked run also makes the fifth mode's terrain (a second 8k texture)
SCAPE_SUFFIX = "station_scape.glb"
HORIZON_FRAC = 0.90             # the skyline's circle: this share of the texture's inscribed circle
SKYLINE_MARGIN_DEG = 1.5        # the full-density zone ends this far above the highest mesh seen from C ...
SKYLINE_MIN_DEG = 10.0          # ... and at least this far above the horizontal: the scene's far surface (beyond the
#                                 mesh) does not tell far hills from the sky, and hills seldom stand higher than this
SHELL_FROM_M = 7.0              # the shell starts this far from the mast axis (the near mesh fades out over FADE_M)
FADE_M = (8.0, 10.0)            # the near terrain's opacity: 1 at the first, 0 at the second (horizontal m from the axis)
SHELL_AZ_STEP_DEG = 0.5
SHELL_STEP_DEG = 0.25           # the shell's rings below the skyline
SHELL_SKY_RINGS = 8             # ... and above it, to the zenith
DOME_M = 3000.0                 # the sky's distance (and the farthest a terrain vertex is put)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def page_frame(X: np.ndarray, pivot, east, up, north) -> np.ndarray:
    """Block coordinates -> the viewer's frame: meters about the pivot, x east, y up, z south."""
    d = np.asarray(X, float) - np.asarray(pivot, float)
    return np.stack([d @ _unit(east), d @ _unit(up), -(d @ _unit(north))], axis=1).astype(np.float32)


# ------------------------------------------------------------------------------------------- geometry
def terrain_mesh(scene, pivot, up, radius_m: float = RADIUS_M, target: int = TARGET_TRIANGLES) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(V, F, record) of the scene's mesh within ``radius_m`` of the mast axis, without the rover model, simplified."""
    from .mesh import decimate_and_smooth, horizontal_distance, keep_triangles
    V, F = np.asarray(scene.mesh_V, float), np.asarray(scene.mesh_F, np.int64)
    keep = np.ones(len(F), bool)
    rt = getattr(scene, "rover_triangles", None)
    if rt is not None:
        keep[rt[0]:rt[1]] = False
    near = horizontal_distance(V, pivot, up) <= float(radius_m)
    keep &= near[F].all(axis=1)
    if not keep.any():
        raise ValueError(f"station mesh: no terrain within {radius_m:g} m of the mast axis")
    V2, F2 = keep_triangles(V, F, keep)
    n0 = len(F2)
    V2, F2, rep = decimate_and_smooth(V2, F2, target_triangles=int(target), taubin_iterations=0)
    return np.asarray(V2, float), np.asarray(F2, np.int64), {"triangles_within_radius": int(n0), "triangles": int(len(F2)),
                                                            "radius_m": float(radius_m)}


def _raycaster(V: np.ndarray, F: np.ndarray):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    return sc


def _cast(rc, o: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import open3d as o3d
    rays = np.concatenate([np.broadcast_to(np.asarray(o, np.float32), np.shape(d)), np.asarray(d, np.float32)], axis=1)
    ans = rc.cast_rays(o3d.core.Tensor(np.ascontiguousarray(rays)))
    return ans["t_hit"].numpy().astype(np.float64), ans["primitive_ids"].numpy().astype(np.int64)


def triangle_normals(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(unit normals, areas) of the triangles (the winding's normal)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    return n / np.maximum(ln, 1e-15)[:, None], 0.5 * ln


def test_points(V: np.ndarray, F: np.ndarray, shrink: float = 0.25) -> np.ndarray:
    """(m, 4, 3): the centroid and the three corners moved ``shrink`` of the way to it, per triangle."""
    tri = V[F]
    cen = tri.mean(axis=1)
    return np.concatenate([cen[:, None], tri + shrink * (cen[:, None] - tri)], axis=1)


def seen_from(rc, pts: np.ndarray, viewpoint: np.ndarray, tol_rel: float = 2e-3, tol_m: float = 2e-3) -> np.ndarray:
    """True where nothing of the mesh in ``rc`` lies between ``viewpoint`` and each point (k, 3)."""
    v = np.asarray(pts, float) - viewpoint
    r = np.linalg.norm(v, axis=1)
    t, _ = _cast(rc, viewpoint, v / np.maximum(r, 1e-12)[:, None])
    return ~(t < r * (1.0 - tol_rel) - tol_m)


# ------------------------------------------------------------------------------------------- the disc
def _theta_of_rho(rho: np.ndarray, rpx: float, theta_max: float, knee) -> np.ndarray:
    """The angle from straight down of a texel ``rho`` px from the map's center.  ``knee`` (0.99.32) = (angle, share):
    the angles up to it fill the share of the radius evenly, the rest - the sky, to the zenith - the remainder."""
    if not knee:
        return rho / rpx * theta_max
    tk, fr = float(knee[0]), float(knee[1])
    x = rho / rpx
    return np.where(x <= fr, x / fr * tk, tk + (np.clip(x, fr, 1.0) - fr) / (1.0 - fr) * (np.pi - tk))


def _rho_of_theta(th: np.ndarray, rpx: float, theta_max: float, knee) -> np.ndarray:
    if not knee:
        return th / theta_max * rpx
    tk, fr = float(knee[0]), float(knee[1])
    return rpx * np.where(th <= tk, th / tk * fr, fr + (th - tk) / (np.pi - tk) * (1.0 - fr))


def disc_dirs(rows: np.ndarray, cols: np.ndarray, size: int, rpx: float, theta_max: float, up, north, east, knee=None) -> Tuple[np.ndarray, np.ndarray]:
    """(unit directions (n, 3), texel radius (n,) px) of the texels (rows, cols) of the azimuthal map (see the module)."""
    c = size / 2.0
    dx = np.asarray(cols, float) + 0.5 - c
    dy = c - (np.asarray(rows, float) + 0.5)
    rho = np.hypot(dx, dy)
    th = _theta_of_rho(rho, rpx, theta_max, knee)
    ph = np.arctan2(dx, dy)
    s = np.sin(th)
    d = (np.cos(th)[:, None] * -_unit(up) + (s * np.sin(ph))[:, None] * _unit(east) + (s * np.cos(ph))[:, None] * _unit(north))
    return d, rho


def disc_uv(X: np.ndarray, C, size: int, rpx: float, theta_max: float, up, north, east, knee=None) -> np.ndarray:
    """Texture coordinates (0..1, v down from the top) of the points (n, 3) on the azimuthal map about C."""
    v = np.asarray(X, float) - np.asarray(C, float)
    r = np.linalg.norm(v, axis=1)
    v = v / np.maximum(r, 1e-12)[:, None]
    th = np.arccos(np.clip(v @ -_unit(up), -1.0, 1.0))
    ph = np.arctan2(v @ _unit(east), v @ _unit(north))
    rho = _rho_of_theta(th, rpx, theta_max, knee)
    c = size / 2.0
    return np.stack([(c + rho * np.sin(ph)) / size, (c - rho * np.cos(ph)) / size], axis=1)


# ------------------------------------------------------------------------------------------- the cells
def triangle_layout(V: np.ndarray, F: np.ndarray, normals: np.ndarray) -> Dict[str, np.ndarray]:
    """Each triangle laid flat in its own plane: "e1", "e2" (unit, in the plane; e1 along the first edge, e2 = n x e1),
    "a" (the first corner), "p" (m, 3, 2) the corners in (e1, e2) meters, "lo", "hi" (m, 2) their box."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1 = b - a
    e1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-15)[:, None]
    e2 = np.cross(normals, e1)
    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=1), 1e-15)[:, None]
    p = np.stack([np.zeros((len(F), 2)), np.stack([((b - a) * e1).sum(1), ((b - a) * e2).sum(1)], 1),
                  np.stack([((c - a) * e1).sum(1), ((c - a) * e2).sum(1)], 1)], axis=1)
    return {"e1": e1, "e2": e2, "a": a, "p": p, "lo": p.min(axis=1), "hi": p.max(axis=1)}


def cell_sizes(lay: Dict[str, np.ndarray], texel_m: float, gutter: int = GUTTER_PX, max_px: int = MAX_CELL_PX) -> Tuple[np.ndarray, np.ndarray]:
    """((m, 2) int cell width and height in texels, (m,) the texel size of each cell, m)."""
    ext = lay["hi"] - lay["lo"]
    s = np.full(len(ext), float(texel_m))
    big = ext.max(axis=1) / s > (max_px - 2 * gutter)
    s[big] = ext[big].max(axis=1) / (max_px - 2 * gutter)
    wh = np.ceil(ext / s[:, None]).astype(np.int64) + 1 + 2 * gutter
    return wh, s


def free_intervals(y0: int, y1: int, size: int, disc: Optional[Tuple[float, float]]) -> List[Tuple[int, int]]:
    """The column intervals [a, b) of the rows [y0, y1) that are clear of the disc (center, radius in px)."""
    if disc is None:
        return [(0, size)]
    c, R = disc
    lo, hi = float(y0), float(y1)
    dmin = 0.0 if lo <= c <= hi else min(abs(lo - c), abs(hi - c))
    if dmin >= R:
        return [(0, size)]
    half = float(np.sqrt(R * R - dmin * dmin))
    return [iv for iv in ((0, int(np.floor(c - half))), (int(np.ceil(c + half)), size)) if iv[1] > iv[0]]


def shelf_pack(wh: np.ndarray, size: int, disc: Optional[Tuple[float, float]] = None,
               reserved: Sequence[Tuple[int, int, int, int]] = ()) -> Optional[np.ndarray]:
    """(m, 2) top-left corners of the cells ``wh`` in a ``size`` square, shelves from the top, the tallest first, clear
    of the disc and of the ``reserved`` rectangles (x, y, w, h); None when they do not fit."""
    n = len(wh)
    pos = np.full((n, 2), -1, np.int64)
    order = np.argsort(-wh[:, 1], kind="stable")
    i, y = 0, 0
    while i < n:
        h = int(wh[order[i], 1])
        if y + h > size:
            return None
        ivs = free_intervals(y, y + h, size, disc)
        cut = []
        for a, b in ivs:                                        # the reserved rectangles split the intervals further
            parts = [(a, b)]
            for (rx, ry, rw, rh) in reserved:
                if ry < y + h and y < ry + rh:
                    nxt = []
                    for p0, p1 in parts:
                        if rx >= p1 or rx + rw <= p0:
                            nxt.append((p0, p1))
                        else:
                            nxt += [q for q in ((p0, rx), (rx + rw, p1)) if q[1] > q[0]]
                    parts = nxt
            cut += parts
        for a, b in cut:
            x = a
            while i < n and x + int(wh[order[i], 0]) <= b:
                pos[order[i]] = (x, y)
                x += int(wh[order[i], 0])
                i += 1
        y += h
    return pos


def layout_cells(lay: Dict[str, np.ndarray], areas: np.ndarray, size: int, disc: Optional[Tuple[float, float]] = None,
                 reserved: Sequence[Tuple[int, int, int, int]] = (), fill: float = 0.8) -> Dict[str, Any]:
    """The cells of the triangles of ``lay`` at the finest texel size that fits: {"pos", "wh", "s" (texel m per
    cell), "texel_m", "A" (m, 3, 3: texel center (x + 0.5, y + 0.5, 1) -> the point in the block), "uv" (m, 3, 2)}."""
    m = len(areas)
    if m == 0:
        return {"pos": np.zeros((0, 2), np.int64), "wh": np.zeros((0, 2), np.int64), "s": np.zeros(0), "texel_m": None,
                "A": np.zeros((0, 3, 3)), "uv": np.zeros((0, 3, 2))}
    free = size * size - (np.pi * disc[1] ** 2 if disc else 0.0) - sum(r[2] * r[3] for r in reserved)
    ext = lay["hi"] - lay["lo"]
    t = float(np.sqrt(max(2.2 * float((ext[:, 0] * ext[:, 1]).sum()), 1e-12) / max(fill * free, 1.0)))
    for _ in range(60):
        wh, s = cell_sizes(lay, t)
        pos = shelf_pack(wh, size, disc, reserved)
        if pos is not None:
            break
        t *= 1.12
    else:
        raise RuntimeError("station mesh: the cells do not fit the texture")
    g = GUTTER_PX
    ox = pos[:, 0] + g - lay["lo"][:, 0] / s                  # texel x of the plane's origin (corner a)
    oy = pos[:, 1] + g - lay["lo"][:, 1] / s
    A = np.zeros((m, 3, 3))
    A[:, :, 0] = lay["e1"] * s[:, None]
    A[:, :, 1] = lay["e2"] * s[:, None]
    A[:, :, 2] = lay["a"] - lay["e1"] * (ox * s)[:, None] - lay["e2"] * (oy * s)[:, None]
    px = ox[:, None] + lay["p"][:, :, 0] / s[:, None]
    py = oy[:, None] + lay["p"][:, :, 1] / s[:, None]
    uv = np.stack([px / size, py / size], axis=2)
    return {"pos": pos, "wh": wh, "s": s, "texel_m": float(t), "A": A, "uv": uv}


def band_cells(cells: Dict[str, Any], normals: np.ndarray, r0: int, r1: int) -> Dict[str, np.ndarray]:
    """The cells that meet the rows [r0, r1): {"rect" (k, 4) x, y, w, h, "A", "n"}."""
    if not len(cells["pos"]):
        return {"rect": np.zeros((0, 4), np.int64), "A": np.zeros((0, 3, 3)), "n": np.zeros((0, 3))}
    y, h = cells["pos"][:, 1], cells["wh"][:, 1]
    k = np.flatnonzero((y < r1) & (y + h > r0))
    return {"rect": np.concatenate([cells["pos"][k], cells["wh"][k]], axis=1), "A": cells["A"][k], "n": normals[k]}


# ------------------------------------------------------------------------------------------- the texels
def texture_rows(scene, sources: Sequence[Any], args: Dict[str, Any], r0: int, r1: int) -> Dict[str, Any]:
    """The rows [r0, r1) of a station mesh texture (the band of a worker, :mod:`rpp.ibr.parallel`): the fields of
    :data:`rpp.ibr.ods.BAND_FIELDS`, (r1 - r0, size) each."""
    from .ods import BAND_FIELDS, shade_rays
    S = int(args["size"])
    nrow = r1 - r0
    flat, O, D = [], [], []
    dsc = args.get("disc")
    if dsc:
        rr, cc = np.mgrid[r0:r1, 0:S]
        d, rho = disc_dirs(rr.ravel(), cc.ravel(), S, dsc["rpx"], dsc["theta_max"], dsc["up"], dsc["north"], dsc["east"], knee=dsc.get("knee"))
        inside = rho <= dsc["rpx"] + GUTTER_PX
        flat.append(np.flatnonzero(inside))
        D.append(d[inside])
        O.append(np.broadcast_to(np.asarray(dsc["C"], float), (int(inside.sum()), 3)))
    bc = args.get("cells") or {}
    off = float(args.get("offset_m", TERRAIN_OFFSET_M))
    for (x, y, w, h), A, n in zip(bc.get("rect", ()), bc.get("A", ()), bc.get("n", ())):
        ya, yb = max(int(y), r0), min(int(y + h), r1)
        if yb <= ya:
            continue
        yy, xx = np.mgrid[ya:yb, int(x):int(x + w)]
        q = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5, np.ones(xx.size)], axis=1)
        X = q @ np.asarray(A).T
        flat.append((yy.ravel() - r0) * S + xx.ravel())
        O.append(X + off * np.asarray(n))
        D.append(np.broadcast_to(-np.asarray(n, float), X.shape))
    linear = bool(args.get("linear"))
    band: Dict[str, Any] = {"rgb": np.zeros((nrow * S, 3), np.float32 if linear else np.uint8),
                            "source": np.full(nrow * S, -1, np.int16), "label": np.zeros(nrow * S, np.uint8),
                            "range": np.zeros(nrow * S, np.float32), "hidden": np.zeros(nrow * S, bool),
                            "ground": np.zeros(nrow * S, bool), "rover": np.zeros(nrow * S, bool),
                            "rej_map": np.zeros(nrow * S, bool), "prov": np.zeros((nrow * S, 3), np.uint8),
                            "rejected": [0, 0], "neff": None}
    if flat:
        idx = np.concatenate(flat)
        o, d = np.concatenate(O).astype(float), np.concatenate(D).astype(float)
        fam = args.get("families")
        srcs = [s for s in sources if s.name[:1] in fam] if fam else list(sources)
        if len(idx) and srcs:
            res = shade_rays(scene, srcs, o, d, up=args.get("up", (0, 0, 1)), linear=linear)
            for k in BAND_FIELDS:
                band[k][idx] = res[k]
            band["rejected"] = res["rejected"]
    for k in BAND_FIELDS:
        band[k] = band[k].reshape((nrow, S) + band[k].shape[1:])
    return band


def render_texture(scene, sources: Sequence[Any], args: Dict[str, Any], pool=None, rows_per_chunk: int = 128) -> Dict[str, np.ndarray]:
    """{"rgb" (S, S, 3), "source" (S, S) int16} of a texture (``args`` as :func:`texture_rows` takes them, without
    "cells": ``args["all_cells"]`` and ``args["normals"]``)."""
    S = int(args["size"])
    spans = [(r0, min(S, r0 + rows_per_chunk)) for r0 in range(0, S, rows_per_chunk)]
    base = {k: v for k, v in args.items() if k not in ("all_cells", "normals", "want_sky")}
    per = [dict(base, cells=band_cells(args["all_cells"], args["normals"], r0, r1)) for r0, r1 in spans]
    rgb = np.zeros((S, S, 3), np.float32 if args.get("linear") else np.uint8)
    src = np.full((S, S), -1, np.int16)
    sky = np.zeros((S, S), bool) if args.get("want_sky") else None      # 0.99.32: the far surface's texels

    def here(todo):
        for r0, r1 in todo:
            yield r0, r1, texture_rows(scene, sources, per[spans.index((r0, r1))], r0, r1)

    bands = pool.run("texture", per, spans, S, S, bool(args.get("linear")), here, "texture") if pool is not None else here(spans)
    for r0, r1, band in bands:
        rgb[r0:r1] = band["rgb"]
        src[r0:r1] = band["source"]
        if sky is not None:                                      # 0.99.32: the sky's texels (not the far terrain's)
            from .raycast import LABEL_FAR                       # beyond the mesh and the plane: the sky and the far hills
            sky[r0:r1] = band["label"] == LABEL_FAR
    band = bands = None
    return {"rgb": rgb, "source": src, "sky": sky}


def push_pull(rgb: np.ndarray, have: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The texels of ``region`` that are not ``have`` filled from the ones that are, coarse to fine (a pull-push
    pyramid); ``rgb`` float (H, W, 3) changed in place.  Returns the texels filled."""
    import cv2
    todo = region & ~have
    if not todo.any() or not have.any():
        return np.zeros(have.shape, bool)
    w = have.astype(np.float32)
    a = rgb.astype(np.float32) * w[..., None]
    pyr = [(a, w)]
    while min(pyr[-1][1].shape) > 8:
        a_, w_ = pyr[-1]
        size = (max(1, w_.shape[1] // 2), max(1, w_.shape[0] // 2))
        pyr.append((cv2.resize(a_, size, interpolation=cv2.INTER_AREA), cv2.resize(w_, size, interpolation=cv2.INTER_AREA)))
    a_, w_ = pyr[-1]
    mean = (rgb[have].mean(axis=0) if have.any() else np.zeros(3)).astype(np.float32)
    cur = np.where(w_[..., None] > 1e-6, a_ / np.maximum(w_, 1e-6)[..., None], mean)   # a coarsest cell without data: the mean
    for a_, w_ in reversed(pyr[:-1]):
        up_ = cv2.resize(cur, (w_.shape[1], w_.shape[0]), interpolation=cv2.INTER_LINEAR)
        k = np.clip(w_, 0, 1)[..., None]
        cur = np.where(w_[..., None] > 1e-6, a_ / np.maximum(w_, 1e-6)[..., None], 0) * k + up_ * (1 - k)
    rgb[todo] = cur[todo]
    return todo


def fill_cells(rgb: np.ndarray, have: np.ndarray, cells: Dict[str, Any], default: Sequence[float]) -> int:
    """A cell's texels no frame colored take the mean of its colored ones (``default`` where it has none)."""
    n = 0
    for (x, y), (w, h) in zip(cells["pos"], cells["wh"]):
        sl = (slice(int(y), int(y + h)), slice(int(x), int(x + w)))
        hv = have[sl]
        if hv.all():
            continue
        col = rgb[sl][hv].mean(axis=0) if hv.any() else np.asarray(default, np.float32)
        blk = rgb[sl]
        blk[~hv] = col
        n += int((~hv).sum())
    return n


# ------------------------------------------------------------------------------------------- to the horizon (0.99.32)
def horizon_shell(scene, C, pivot, up, north, east, from_m: float = SHELL_FROM_M, az_step_deg: float = SHELL_AZ_STEP_DEG,
                  step_deg: float = SHELL_STEP_DEG, sky_rings: int = SHELL_SKY_RINGS, dome_m: float = DOME_M,
                  margin_deg: float = SKYLINE_MARGIN_DEG) -> Dict[str, Any]:
    """The surface seen from C beyond the near terrain, as one sheet: on rings of equal angle from straight down, a
    vertex per ``az_step_deg`` where its ray from C meets the scene (the mesh, the ground plane, the far surface -
    the sky sphere or the far field, which also carries the sky) or, where it meets nothing, at ``dome_m``.  The
    rings run from the angle of ``from_m`` on the ground up to the map's knee (the highest ray that meets the mesh
    or the plane plus ``margin_deg``, at least ``SKYLINE_MIN_DEG`` above the horizontal) every ``step_deg``, then
    ``sky_rings`` more to the zenith.  Seen from C the sheet has no fold; its parallax for a head off C is the true one out to the far
    field and none on the sky.  Returns {"V" (n, 3), "F" (m, 3) - the outer rings first, so a viewer that draws them
    in order without a depth buffer draws the near ones over the far ones -, "knee" rad, "theta0" rad, "rings", "az"}."""
    from .raycast import LABEL_FAR, LABEL_NONE, SKY_RANGE_M
    C, up, north, east = np.asarray(C, float), _unit(up), _unit(north), _unit(east)

    def cast(th, ph):
        s = np.sin(th)
        d = np.cos(th)[:, None] * -up + (s * np.sin(ph))[:, None] * east + (s * np.cos(ph))[:, None] * north
        h = scene.cast(np.broadcast_to(C, d.shape).copy(), d)
        t = np.asarray(h["t"], float)
        open_ = (h["label"] == LABEL_NONE) | ~np.isfinite(t) | (t >= dome_m)       # nothing there: the dome
        return d, np.where(open_, dome_m, t), open_ | (h["label"] == LABEL_FAR)   # ..., beyond the mesh and the plane
    _, t_down, _ = cast(np.zeros(1), np.zeros(1))
    h_c = float(t_down[0]) if t_down[0] < dome_m else 2.5
    th0 = max(np.radians(20.0), float(np.arctan2(from_m, h_c)) - np.radians(3.0))
    ph_c = np.radians(np.arange(0.0, 360.0, 1.0))
    th_c = np.arange(th0, np.pi - 1e-6, np.radians(0.25))
    TT, PP = np.meshgrid(th_c, ph_c, indexing="ij")
    _, _, sky_c = cast(TT.ravel(), PP.ravel())
    ground = ~sky_c.reshape(TT.shape)
    top = float(th_c[np.flatnonzero(ground.any(axis=1)).max()]) if ground.any() else np.pi / 2
    knee = float(np.clip(max(top + np.radians(margin_deg), np.pi / 2 + np.radians(SKYLINE_MIN_DEG)), th0 + np.radians(5.0),
                         np.pi - np.radians(20.0)))
    ths = np.r_[np.arange(th0, knee - 1e-9, np.radians(step_deg)), knee, np.linspace(knee, np.pi, int(sky_rings) + 1)[1:]]
    phs = np.radians(np.arange(0.0, 360.0, az_step_deg))
    nr, na = len(ths), len(phs)
    TT, PP = np.meshgrid(ths, phs, indexing="ij")
    d, t, sky_f = cast(TT.ravel(), PP.ravel())
    low = (t >= dome_m) & (TT.ravel() < np.pi / 2 - 1e-3)        # a ray below the horizontal that meets nothing (a gap
    t[low] = np.minimum(dome_m, h_c / np.cos(TT.ravel()[low]))   # in the scene): on the level ground under C, not the sky
    V = C + d * t[:, None]
    hd = np.linalg.norm((V - np.asarray(pivot, float)) - ((V - np.asarray(pivot, float)) @ up)[:, None] * up, axis=1).reshape(nr, na)
    F = []
    j = np.arange(na)
    j1 = (j + 1) % na
    for i in range(nr - 2, -1, -1):                              # the outer rings first
        a_, b_, c_, d_ = i * na + j, i * na + j1, (i + 1) * na + j1, (i + 1) * na + j
        far = (hd[i, j] >= from_m) | (hd[i, j1] >= from_m) | (hd[i + 1, j] >= from_m) | (hd[i + 1, j1] >= from_m)
        F.append(np.stack([a_[far], b_[far], c_[far]], 1))
        F.append(np.stack([a_[far], c_[far], d_[far]], 1))
    F = np.vstack(F).astype(np.int64)
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    remap = np.cumsum(used) - 1
    return {"V": V[used], "F": remap[F], "knee": knee, "theta0": float(th0), "rings": int(nr), "az": int(na)}


# ------------------------------------------------------------------------------------------- the two parts
def terrain_part(scene, sources, station, pool=None, size: int = TEXTURE_PX, radius_m: float = RADIUS_M,
                 center_above_m: float = CENTER_ABOVE_M, target: int = TARGET_TRIANGLES, linear: bool = False,
                 rows_per_chunk: int = 128, full: bool = False) -> Dict[str, Any]:
    """The terrain mesh and its texture (see the module): {"V" block (n, 3), "F", "uv" (n, 2), "texture" (S, S, 3)
    float or uint8 as the sources, "record"}.

    ``full`` (0.99.32, the fifth mode): the texture's disc is the whole view from C - the angles up to the skyline on
    ``HORIZON_FRAC`` of its radius, the sky to the zenith on the rest - and the part also holds "shell", the surface
    from ``SHELL_FROM_M`` to the far field and the sky (:func:`horizon_shell`: {"V", "F", "uv"}), on the same texture."""
    t0 = time.time()
    P, up, north, east = (np.asarray(x, float) for x in (station.pivot, station.up, station.north, station.east))
    up = _unit(up)
    V, F, rec = terrain_mesh(scene, P, up, radius_m, target)
    C = P + float(center_above_m) * up
    nrm, area = triangle_normals(V, F)
    rc = _raycaster(V, F)
    pts = test_points(V, F)
    seen = np.ones(len(F), bool)
    for j in range(pts.shape[1]):
        seen &= seen_from(rc, pts[:, j], C)
    view = pts[:, 0] - C
    view /= np.maximum(np.linalg.norm(view, axis=1), 1e-12)[:, None]
    grazing = np.abs((nrm * view).sum(1)) < GRAZING_COS
    vis = seen & ~grazing
    used = np.unique(F[vis])
    th = np.arccos(np.clip(((V[used] - C) / np.linalg.norm(V[used] - C, axis=1)[:, None]) @ -up, -1, 1))
    theta_max = float(min(np.pi * 0.995, th.max() + np.radians(0.5)))
    shell = knee = None
    if full:
        shell = horizon_shell(scene, C, P, up, north, east)
        knee = (max(shell["knee"], theta_max), HORIZON_FRAC)
        theta_max = float(np.pi)
    rpx = size / 2.0 - GUTTER_PX - 2
    disc = (size / 2.0, rpx + GUTTER_PX + 3)
    if float((nrm * area[:, None]).sum(0) @ up) < 0:            # the winding's normals point down: turned up (rays
        nrm = -nrm                                              # start above the surface)
    hid = np.flatnonzero(~vis)
    nh = nrm[hid].copy()
    lay = triangle_layout(V, F[hid], nh)
    cells = layout_cells(lay, area[hid], size, disc=disc)
    args = {"size": size, "linear": linear, "up": tuple(up), "offset_m": TERRAIN_OFFSET_M,
            "disc": {"C": tuple(C), "rpx": rpx, "theta_max": theta_max, "up": tuple(up), "north": tuple(north), "east": tuple(east),
                     "knee": knee},
            "all_cells": cells, "normals": nh, "want_sky": bool(full)}
    tex = render_texture(scene, sources, args, pool, rows_per_chunk)
    rgb, have = np.asarray(tex.pop("rgb"), np.float32), tex.pop("source") >= 0
    sky = tex.pop("sky", None)
    ax = (np.arange(size, dtype=np.float32) + 0.5 - size / 2.0) ** 2
    in_disc = (ax[None, :] + ax[:, None]) <= np.float32((rpx + GUTTER_PX) ** 2)
    del ax
    seen_disc = in_disc & have
    if full and sky is not None and (sky & have & in_disc).any():
        # the sky's unseen texels from the seen sky, the ground's from the ground: no terrain color runs up the sky
        ground = in_disc & ~sky
        filled = push_pull(rgb, have & ground, ground)
        filled |= push_pull(rgb, have & sky & in_disc, sky & in_disc)
        del ground
    else:
        filled = push_pull(rgb, have & in_disc, in_disc)
    sky = None
    med = np.median(rgb[seen_disc][::97], axis=0) if seen_disc.any() else np.full(3, 128.0)
    cell_filled = fill_cells(rgb, have, cells, med)
    # the mesh: the disc's triangles share their vertices; a cell's triangle has its own three
    uv_v = disc_uv(V, C, size, rpx, theta_max, up, north, east, knee=knee)
    Fv = F[vis]
    nv = len(V)
    Vo = np.concatenate([V, V[F[hid]].reshape(-1, 3)])
    UVo = np.concatenate([uv_v, cells["uv"].reshape(-1, 2)])
    Fo = np.concatenate([Fv, nv + np.arange(3 * len(hid)).reshape(-1, 3)])
    keep = np.zeros(len(Vo), bool)
    keep[Fo.ravel()] = True
    remap = np.cumsum(keep) - 1
    n_disc_texels = int(in_disc.sum())
    rec.update(texture_px=size, center_above_pivot_m=float(center_above_m), theta_max_deg=round(float(np.degrees(theta_max)), 2),
               triangles_from_center=int(vis.sum()), triangles_in_cells=int(len(hid)),
               hidden_from_center=int((~seen).sum()), edge_on_from_center=int((seen & grazing).sum()),
               cell_texel_mm=None if cells["texel_m"] is None else round(1000 * cells["texel_m"], 2),
               disc_texels_colored=round(float(seen_disc.sum() / max(n_disc_texels, 1)), 4),
               disc_texels_filled=int(filled.sum()), cell_texels_filled=int(cell_filled),
               mrad_per_texel=round(1000 * (knee[0] / (knee[1] * rpx) if knee else theta_max / rpx), 4))
    out_shell = None
    if shell is not None:
        out_shell = {"V": shell["V"], "F": shell["F"], "uv": disc_uv(shell["V"], C, size, rpx, theta_max, up, north, east, knee=knee)}
        rec.update(full=True, skyline_deg=round(float(np.degrees(knee[0])) - 90.0, 2), horizon_frac=HORIZON_FRAC,
                   texels_per_deg=round(float(knee[1] * rpx / np.degrees(knee[0])), 1),
                   texels_per_deg_around_horizon=round(float(2 * np.pi * knee[1] * rpx * (np.pi / 2 / knee[0]) / 360.0), 1),
                   shell={"triangles": int(len(shell["F"])), "from_m": SHELL_FROM_M, "rings": shell["rings"], "az": shell["az"],
                          "dome_m": DOME_M}, fade_m=list(FADE_M))
    rec["seconds"] = round(time.time() - t0, 1)
    log.info("station mesh: terrain %d triangles (%d from the center, %d in cells at %s mm), texture %d px, %.1f %% of the "
             "disc colored by the frames (%.0f s)", len(Fo), int(vis.sum()), len(hid), rec["cell_texel_mm"], size,
             100 * rec["disc_texels_colored"], rec["seconds"])
    return {"V": Vo[keep], "F": remap[Fo], "uv": UVo[keep], "texture": rgb, "record": rec, "shell": out_shell}


def viewpoints(pivot, radius_m: float = VIEW_RADIUS_M, n: int = VIEWPOINTS) -> np.ndarray:
    """The pivot and ``n`` points on spheres of half ``radius_m`` and ``radius_m`` about it (a Fibonacci sphere)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    ph = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z * z)
    d = np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=1)
    P = np.asarray(pivot, float)
    return np.concatenate([P[None], P + 0.5 * radius_m * d, P + radius_m * d])


def rover_families(sources) -> str:
    """The frames that color the rover: the Navcams, and the run's own cameras when they are others (0.99.27,
    Christian, 18 Nov 2026: "For the z34 runs, the texture should come from the Z34 and the ncam with preferences to
    the z34" - in a Mastcam-Z run its frames are the regular sources and the Navcam frames the fill-only ones, which
    come in only as the Mastcam-Z frames' weight goes out)."""
    own = "".join(sorted({s.name[:1] for s in sources if not getattr(s, "fill_only", False) and s.name}))
    return ROVER_FAMILIES + "".join(f for f in own if f not in ROVER_FAMILIES and f != "F" and f != "R")


def rover_part(scene, sources, station, pool=None, size: int = ROVER_TEXTURE_PX, view_radius_m: float = VIEW_RADIUS_M,
               families: Optional[str] = None, linear: bool = False, rows_per_chunk: int = 128) -> Dict[str, Any]:
    """The rover model and its texture (see the module): {"V", "F", "uv", "texture", "record"}.  ``families``: the
    frames that color it (default :func:`rover_families`)."""
    t0 = time.time()
    if families is None:
        families = rover_families(sources)
    rt = getattr(scene, "rover_triangles", None)
    if rt is None:
        raise ValueError("station mesh: the scene holds no rover model (rpp mesh --rover)")
    up = _unit(station.up)
    Fall = np.asarray(scene.mesh_F, np.int64)[rt[0]:rt[1]]
    used = np.unique(Fall)
    remap = -np.ones(len(scene.mesh_V), np.int64)
    remap[used] = np.arange(len(used))
    V, F = np.asarray(scene.mesh_V, float)[used], remap[Fall]
    nrm, area = triangle_normals(V, F)
    rc_all = _raycaster(np.asarray(scene.mesh_V, float), np.asarray(scene.mesh_F, np.int64))   # terrain and rover
    rc_rov = _raycaster(V, F)
    vps = viewpoints(station.pivot, view_radius_m)
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    good = []
    for vp in vps:                                              # a viewpoint inside the model sees its inside
        t, _ = _cast(rc_rov, vp, axes)
        if int(np.isfinite(t).sum()) < INSIDE_HITS:
            good.append(vp)
    good = np.asarray(good)
    pts = test_points(V, F)
    seen = np.zeros(len(F), bool)
    side = np.zeros(len(F))
    for vp in good:
        for j in range(pts.shape[1]):
            s = seen_from(rc_all, pts[:, j], vp)
            side[s] += np.sign(((vp - pts[s, j]) * nrm[s]).sum(1))
            seen |= s
    k = np.flatnonzero(seen)
    nk = nrm[k] * np.where(side[k] < 0, -1.0, 1.0)[:, None]    # the side that is seen
    gw, gh = GRAY_STRIP
    lay = triangle_layout(V, F[k], nk)
    cells = layout_cells(lay, area[k], size, reserved=[(0, 0, gw, gh)])
    args = {"size": size, "linear": linear, "up": tuple(up), "offset_m": ROVER_OFFSET_M, "families": families,
            "all_cells": cells, "normals": nk}
    tex = render_texture(scene, sources, args, pool, rows_per_chunk)
    rgb, have = np.asarray(tex.pop("rgb"), np.float32), tex.pop("source") >= 0
    from .ods import ROVER_GRAY_REL
    from .rover_surface import GRAY_SHADE
    colored = np.zeros(have.shape, bool)
    for (x, y), (w, h) in zip(cells["pos"], cells["wh"]):
        colored[int(y):int(y + h), int(x):int(x + w)] = True
    base = rgb[colored & have]
    g0 = ROVER_GRAY_REL * float(np.mean(base)) if len(base) else ROVER_GRAY_REL * 128.0
    gray = np.full(3, g0, np.float32)
    cell_filled = fill_cells(rgb, have, cells, gray)
    shade = 1.0 - GRAY_SHADE * (1.0 - (np.arange(gh) + 0.5) / gh)   # row j: |n . up| = (j + 0.5) / gh
    rgb[0:gh, 0:gw] = (gray[None, None, :] * shade[:, None, None])
    un = np.flatnonzero(~seen)
    ny = np.clip(np.abs(nrm[un] @ up), 0, 1)
    uv_un = np.stack([np.full(len(un), gw / 2.0 / size), np.clip(ny * gh, 0.5, gh - 0.5) / size], axis=1)
    Vo = np.concatenate([V[F[k]].reshape(-1, 3), V[F[un]].reshape(-1, 3)])
    UVo = np.concatenate([cells["uv"].reshape(-1, 2), np.repeat(uv_un, 3, axis=0)])
    Fo = np.arange(len(Vo)).reshape(-1, 3)
    rec = {"triangles": int(len(F)), "textured_triangles": int(len(k)),
           "textured_area_share": round(float(area[k].sum() / max(area.sum(), 1e-12)), 4),
           "viewpoints": int(len(good)), "viewpoints_inside_the_model": int(len(vps) - len(good)),
           "view_radius_m": float(view_radius_m), "families": families, "texture_px": size,
           "texel_mm": None if cells["texel_m"] is None else round(1000 * cells["texel_m"], 2),
           "texels_colored": round(float((colored & have).sum() / max(colored.sum(), 1)), 4),
           "cell_texels_filled": int(cell_filled), "gray_rgb": [round(float(x), 1) for x in gray],
           "seconds": round(time.time() - t0, 1)}
    log.info("station mesh: rover %d of %d triangles textured (%.0f %% of the area; %d viewpoints within %.1f m), texel "
             "%s mm, %.0f %% of the texels colored by the frames (%.0f s)", len(k), len(F), 100 * rec["textured_area_share"],
             len(good), view_radius_m, rec["texel_mm"], 100 * rec["texels_colored"], rec["seconds"])
    return {"V": Vo, "F": Fo, "uv": UVo, "texture": rgb, "record": rec}


def write_part(out_dir: PathLike, prefix: str, kind: str, part: Dict[str, Any], station, to8) -> Dict[str, Any]:
    """``<prefix>_<PART[kind]>.npz`` (V in the viewer's frame, F, uv), ``.jpg`` (the texture, 8-bit by ``to8``), ``.json``."""
    import cv2
    out = Path(out_dir)
    stem = f"{prefix + '_' if prefix else ''}{PART[kind]}"
    tex8 = to8(part["texture"])
    ok, enc = cv2.imencode(".jpg", np.ascontiguousarray(tex8[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("station mesh: the texture could not be encoded")
    (out / (stem + ".jpg")).write_bytes(enc.tobytes())
    Vp = page_frame(part["V"], station.pivot, station.east, station.up, station.north)
    extra = {}
    sh = part.get("shell")
    if sh is not None:                                           # 0.99.32: the surface to the horizon, on the same texture
        extra = {"shell_V": page_frame(sh["V"], station.pivot, station.east, station.up, station.north),
                 "shell_F": np.asarray(sh["F"], np.uint32), "shell_uv": np.asarray(sh["uv"], np.float32)}
    np.savez_compressed(out / (stem + ".npz"), V=Vp, F=np.asarray(part["F"], np.uint32), uv=np.asarray(part["uv"], np.float32), **extra)
    rec = dict(part["record"], kind=kind, files=[stem + ".npz", stem + ".jpg"], vertices=int(len(Vp)), mesh_triangles=int(len(part["F"])))
    (out / (stem + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ------------------------------------------------------------------------------------------- the file
def glb_bytes(parts: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, bytes]], extras: Optional[Dict[str, Any]] = None) -> bytes:
    """A binary glTF 2.0 file of textured meshes ``(name, V, F, uv, jpeg)``: unlit, double-sided materials.  A part
    whose ``jpeg`` is a string uses the texture of the earlier part of that name."""
    chunks, views = [], []

    def add(b: bytes, target: Optional[int] = None) -> int:
        off = sum(len(p) for p in chunks)
        chunks.append(b + b"\x00" * (-len(b) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(b)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1
    acc, meshes, mats, texs, imgs, nodes = [], [], [], [], [], []
    by_name: Dict[str, int] = {}
    for name, V, F, uv, jpg in parts:
        V = np.ascontiguousarray(V, "<f4")
        uv = np.ascontiguousarray(uv, "<f4")
        F = np.ascontiguousarray(F, "<u4").reshape(-1)
        a_v = len(acc)
        acc.append({"bufferView": add(V.tobytes(), 34962), "componentType": 5126, "count": int(len(V)), "type": "VEC3",
                    "min": [float(x) for x in V.min(axis=0)], "max": [float(x) for x in V.max(axis=0)]})
        acc.append({"bufferView": add(uv.tobytes(), 34962), "componentType": 5126, "count": int(len(uv)), "type": "VEC2"})
        acc.append({"bufferView": add(F.tobytes(), 34963), "componentType": 5125, "count": int(len(F)), "type": "SCALAR"})
        if isinstance(jpg, str):                                 # 0.99.32: the texture of the part of that name
            ti = by_name[jpg]
        else:
            imgs.append({"bufferView": add(bytes(jpg)), "mimeType": "image/jpeg", "name": name})
            texs.append({"source": len(imgs) - 1, "sampler": 0})
            ti = by_name[name] = len(texs) - 1
        mats.append({"name": name, "doubleSided": True, "extensions": {"KHR_materials_unlit": {}},
                     "pbrMetallicRoughness": {"baseColorTexture": {"index": ti}, "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": a_v, "TEXCOORD_0": a_v + 1}, "indices": a_v + 2,
                                                     "material": len(mats) - 1, "mode": 4}]})
        nodes.append({"mesh": len(meshes) - 1, "name": name})
    doc = {"asset": {"version": "2.0", "generator": "rpp.ibr.station_mesh"}, "extensionsUsed": ["KHR_materials_unlit"],
           "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "accessors": acc,
           "materials": mats, "textures": texs, "images": imgs,
           "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
           "bufferViews": views, "buffers": [{"byteLength": sum(len(p) for p in chunks)}]}
    if extras:
        doc["asset"]["extras"] = extras
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    binary = b"".join(chunks)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


def _part_files(folder: Path, kind: str) -> Optional[Tuple[Path, Path, Path]]:
    for npz in sorted(Path(folder).glob(f"*{PART[kind]}.npz")):
        jpg, js = npz.with_suffix(".jpg"), npz.with_suffix(".json")
        if jpg.is_file():
            return npz, jpg, js
    return None


def assemble_scape(folder: PathLike, rover_folder: Optional[PathLike] = None, out: Optional[PathLike] = None) -> Optional[Dict[str, Any]]:
    """0.99.32: ``<prefix>_station_scape.glb`` in ``folder`` - the fifth mode's file: the meshes "horizon" (the shell:
    first, the viewer draws it first), "terrain" (the near terrain; both on the one terrain texture) and "rover" (the
    rover part of ``rover_folder``, default ``folder/with_rover``, with its own).  None without a scape part."""
    d = Path(folder)
    tp = _part_files(d, "scape")
    if tp is None:
        return None
    rp = _part_files(Path(rover_folder) if rover_folder else d / "with_rover", "rover")
    recs: Dict[str, Any] = {}
    with np.load(tp[0]) as z:
        if "shell_V" not in z:
            return None
        parts = [("terrain", z["V"], z["F"], z["uv"], tp[1].read_bytes()), ("horizon", z["shell_V"], z["shell_F"], z["shell_uv"], "terrain")]
    try:
        recs["terrain"] = json.loads(tp[2].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recs["terrain"] = {}
    if rp is not None:
        with np.load(rp[0]) as z:
            parts.append(("rover", z["V"], z["F"], z["uv"], rp[1].read_bytes()))
        try:
            recs["rover"] = json.loads(rp[2].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recs["rover"] = {}
    stem = tp[0].name[:-len(PART["scape"] + ".npz")]
    glb = Path(out) if out else d / (stem + SCAPE_SUFFIX)
    t = recs["terrain"]
    extras = {"station_scape": {"frame": "meters about the pivot: x east, y up, z south", "parts": [p[0] for p in parts],
                                "fade_m": t.get("fade_m") or list(FADE_M), "dome_m": (t.get("shell") or {}).get("dome_m", DOME_M),
                                "view_radius_m": (recs.get("rover") or {}).get("view_radius_m", VIEW_RADIUS_M)}}
    glb.write_bytes(glb_bytes(parts, extras))
    rec = {"file": glb.name, "bytes": glb.stat().st_size, "parts": recs, "triangles": int(sum(len(p[2]) for p in parts)),
           "fade_m": extras["station_scape"]["fade_m"], "dome_m": extras["station_scape"]["dome_m"]}
    glb.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log.info("station mesh: %s - %s, %d triangles, %.1f MB", glb.name, " + ".join(p[0] for p in parts), rec["triangles"], rec["bytes"] / 1e6)
    return rec


def assemble(folder: PathLike, rover_folder: Optional[PathLike] = None, out: Optional[PathLike] = None) -> Optional[Dict[str, Any]]:
    """``<prefix>_station_mesh.glb`` in ``folder`` (the station's terrain-only folder) from its terrain part and the
    rover part of ``rover_folder`` (default ``folder/with_rover``; the rover is left out when it has none).  None
    without a terrain part."""
    d = Path(folder)
    tp = _part_files(d, "terrain")
    if tp is None:
        log.info("station mesh: no terrain part in %s (rpp ods --station-mesh)", d)
        return None
    rp = _part_files(Path(rover_folder) if rover_folder else d / "with_rover", "rover")
    parts, recs = [], {}
    for kind, files in (("terrain", tp), ("rover", rp)):
        if files is None:
            continue
        with np.load(files[0]) as z:
            parts.append((kind, z["V"], z["F"], z["uv"], files[1].read_bytes()))
        try:
            recs[kind] = json.loads(files[2].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recs[kind] = {}
    stem = tp[0].name[:-len(PART["terrain"] + ".npz")]
    glb = Path(out) if out else d / (stem + SUFFIX)
    extras = {"station_mesh": {"frame": "meters about the pivot: x east, y up, z south", "parts": list(recs),
                               "radius_m": recs["terrain"].get("radius_m"), "fade_m": list(FADE_M), "view_radius_m": (recs.get("rover") or {}).get("view_radius_m", VIEW_RADIUS_M)}}
    glb.write_bytes(glb_bytes(parts, extras))
    rec = {"file": glb.name, "bytes": glb.stat().st_size, "parts": recs,
           "triangles": int(sum(len(p[2]) for p in parts))}
    glb.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log.info("station mesh: %s - %s, %d triangles, %.1f MB", glb.name, " + ".join(recs), rec["triangles"], rec["bytes"] / 1e6)
    return rec


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="the station mesh file (.glb) of a station's terrain and rover parts")
    ap.add_argument("folder", help="the station's terrain-only folder (its with_rover folder holds the rover part)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    one, two = assemble(a.folder), assemble_scape(a.folder)
    return 0 if (one or two) else 1


if __name__ == "__main__":
    raise SystemExit(main())
