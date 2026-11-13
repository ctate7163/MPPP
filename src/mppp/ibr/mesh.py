"""
mppp.ibr.mesh - one clean surface for a station: the fused points meshed, the mesh cleaned, a far-field sphere.

Plan of 31 Oct 2026 (step 2).  The dense fusion (``mppp dense``, ``<project>/dense/fused.ply``) gives points; a
mesh from them is one surface every image's depth map is then cast from (:mod:`.raycast`), so the depth maps agree
with each other by construction.  The surface comes from COLMAP's Delaunay mesher when the COLMAP executable is at
hand (it carves free space with the visibility rays and is honest about holes), else from Open3D's Poisson
reconstruction with its low-density skin trimmed (the fallback is named in the report).

Cleanup, in order, each step counted in ``mesh_report.json``:

1. connected components holding less than ``component_fraction`` of the triangles are dropped (floaters);
2. rubber sheets: triangles both long-edged (longest edge > max(0.10 m, 4 x 3 x range x 0.68 mrad)) and grazing
   as seen from the pivot (|n . v| < ``grazing``) are dropped - the skins stretched across occlusion edges;
3. support: triangles whose centroid is farther than tau(range) + half their longest edge from the nearest fused
   point are dropped (surface hallucinated over gaps);
4. radius: terrain farther than ``radius_m`` (30 m; 20 m on 10 Nov morning) from the mast's axis - the vertical through the pivot,
   horizontal distance - is dropped (Christian, 10 Nov 2026: "cut off at 20 m from the axis of the mast rotation", then 30 m);
   the ground plane takes over beyond it;
5. quadric decimation to ``target_triangles`` and five Taubin smoothing iterations (no shrinkage).

Beyond the mesh (10 Nov 2026): **the ground plane** out to ``plane_radius_m`` (100 m) from the axis, then **the sky
at infinity** (:mod:`.raycast`).  The plane (:func:`fit_ground_plane`) is fitted to the terrain the station sees,
the fused points and the block's tie points 3-100 m from the axis: the median point of every 10 deg x range-ring
bin, then a Huber line fit z = a x + b y + c over the bins, so the near field's density does not outvote the far
rings and an outlier bin does not tilt it.  It follows the regional slope (a horizontal plane on a 10 deg slope is
17 m off at 100 m); ``horizontal=True``, or a tilt over 20 deg, or fewer than 8 bins, gives the horizontal plane at
the median height instead.  Its misfit per ring is in ``ground_plane.json``; the tie points on it are checked again
in the depth maps' report.  The mesh's holes within 30 m show the plane too.  Beyond 100 m the 64 mm disparity of an
8K equirect is under a pixel (about 0.7 px at 100 m), so the sky at infinity shows no stereo error to speak of.
Only the fused points within ``radius_m`` + 5 m are meshed; the farther ones feed the plane fit.

tau(z) = max(0.05 m, 0.02 z, 3 sigma_z) with sigma_z the single-pair Navcam estimate at half resolution
(0.5 px x 0.68 mrad x z^2 / 0.424 m) - the depth tolerance of the support check, the coherence check and the
visibility test of the ODS render.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.ibr.mesh")
PathLike = Union[str, Path]

NAVCAM_HALF_IFOV_RAD = 0.68e-3       # Navcam at half resolution (8K ODS sampling)
NAVCAM_BASELINE_M = 0.424
DEFAULT_RADIUS_M = 75.0             # the mesh: horizontal distance from the mast axis (10 Nov: 20 m, then 30; 12 Nov, 0.97.0: 50 m - Christian: "make sure it extends all the way to 50. The mesh can be extra simplified at that distances")
#                                     12 Nov, later: 75 m - "the terrain should go out further to maybe 75 m. the sky sphere can be at
#                                     infinity, and a plane extending out to 200 m should blend between the terrain and sky")
DEFAULT_PLANE_RADIUS_M = 200.0      # the ground plane out to here (horizontal); then the sky at infinity
DEFAULT_FAR_RADIUS_M = 100.0        # the 0.96.0 prototype's finite far sphere (mppp mesh --far-radius)
PLANE_RINGS_M = (3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 65.0, 100.0, 140.0, 200.0)
PLANE_SECTOR_DEG = 10.0
PLANE_MAX_TILT_DEG = 20.0
PLANE_MIN_BINS = 8
PLANE_MIN_POINTS_PER_BIN = 5
DEFAULT_TARGET_TRIANGLES = 6_000_000
# 10 Nov (Christian: "The near and mid terrain needs a little more detail and higher polygon count"): Poisson per zone of
# horizontal distance from the mast axis - each zone meshed from the points within its outer edge + 1 m, so its octree
# box is small and depth 10 is ~1 cm at 0-4 m, ~2.5 cm at 4-12 m, ~4 cm at 12-20 m, ~6 cm at 20-30 m; the zones overlap by ZONE_OVERLAP_M
# (no gap for a ray to fall through)
MESH_ZONES = ((0.0, 4.0, 10), (4.0, 12.0, 10), (12.0, 20.0, 10), (20.0, 30.0, 10), (30.0, None, 9))   # (inner m, outer m or None = the mesh radius, depth); 0.97.0: 30-50 m at depth 9 (~20 cm), "extra simplified"
ZONE_OVERLAP_M = 0.3
GRAZING = 0.10
COMPONENT_FRACTION = 0.01


def sigma_z(z) -> np.ndarray:
    """The single-pair Navcam depth error at half resolution (metres) at range ``z``."""
    z = np.asarray(z, float)
    return 0.5 * NAVCAM_HALF_IFOV_RAD * z ** 2 / NAVCAM_BASELINE_M


def tau_m(z) -> np.ndarray:
    """tau(z) = max(0.05 m, 0.02 z, 3 sigma_z): the depth tolerance at range ``z``."""
    z = np.asarray(z, float)
    return np.maximum(0.05, np.maximum(0.02 * z, 3.0 * sigma_z(z)))


# ----------------------------------------------------------------------------- geometry as arrays
def triangle_geometry(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centroids (m, 3), unit normals (m, 3), longest edge (m,)) of the triangles ``F`` over the vertices ``V``."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cen = (a + b + c) / 3.0
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln > 0, ln, 1.0)[:, None]
    e = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)], axis=1)
    return cen, n, e.max(axis=1)


def connected_components(F: np.ndarray, n_vertices: int) -> np.ndarray:
    """A component label per triangle (triangles sharing a vertex are connected): union-find over the vertices."""
    parent = np.arange(n_vertices)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in F:
        r0 = find(tri[0])
        for v in tri[1:]:
            r = find(v)
            if r != r0:
                parent[r] = r0
    roots = np.array([find(v) for v in F[:, 0]])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def keep_triangles(V: np.ndarray, F: np.ndarray, keep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``F[keep]`` with the unused vertices dropped and the indices renumbered."""
    F2 = F[keep]
    used = np.unique(F2)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F2]


def horizontal_distance(X: np.ndarray, pivot: Sequence[float], up: Sequence[float] = (0, 0, 1)) -> np.ndarray:
    """Distance of the points from the vertical axis (``up``) through ``pivot``: the mast's azimuth axis to within
    the rover's tilt (a 5 deg tilt moves the 20 m boundary by 0.2 m for ground 2 m below the head)."""
    u = np.asarray(up, float) / np.linalg.norm(up)
    d = np.asarray(X, float).reshape(-1, 3) - np.asarray(pivot, float)
    return np.linalg.norm(d - (d @ u)[:, None] * u, axis=1)


def _horizontal_axes(up: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(up, float) / np.linalg.norm(up)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ u) * u
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1), u


def fit_ground_plane(points: np.ndarray, pivot: Sequence[float], up: Sequence[float] = (0, 0, 1),
                     radius_m: float = DEFAULT_PLANE_RADIUS_M, rings: Sequence[float] = PLANE_RINGS_M,
                     sector_deg: float = PLANE_SECTOR_DEG, horizontal: bool = False,
                     max_tilt_deg: float = PLANE_MAX_TILT_DEG) -> Dict[str, Any]:
    """The ground plane of a station (see the module doc): ``normal`` (unit, upward), ``offset`` (n . X = offset),
    ``centre`` (the pivot), ``up``, ``radius_m`` (its horizontal reach), the tilt and its downhill azimuth, the
    misfit per ring, and how it was made."""
    P = np.asarray(pivot, float)
    e1, e2, u = _horizontal_axes(up)
    X = np.asarray(points, float).reshape(-1, 3)
    X = X[np.all(np.isfinite(X), axis=1)]
    d = X - P
    x, y, z = d @ e1, d @ e2, d @ u
    r = np.hypot(x, y)
    rings = np.asarray([v for v in rings if v <= radius_m] + ([radius_m] if rings[-1] < radius_m else []), float)
    sel = (r >= rings[0]) & (r <= rings[-1])
    rep: Dict[str, Any] = {"centre": P.tolist(), "up": u.tolist(), "radius_m": float(radius_m), "points": int(sel.sum())}
    bins = []
    if sel.any():
        ring = np.clip(np.searchsorted(rings, r[sel], side="right") - 1, 0, len(rings) - 2)
        sec = np.floor((np.degrees(np.arctan2(y[sel], x[sel])) % 360.0) / sector_deg).astype(int)
        key = ring * 1000 + sec
        order = np.argsort(key, kind="stable")
        ks, starts = np.unique(key[order], return_index=True)
        ends = np.r_[starts[1:], len(order)]
        xs, ys, zs = x[sel][order], y[sel][order], z[sel][order]
        for k, a, b in zip(ks, starts, ends):
            if b - a < PLANE_MIN_POINTS_PER_BIN:
                continue
            bins.append((int(k // 1000), float(np.median(xs[a:b])), float(np.median(ys[a:b])), float(np.median(zs[a:b])), int(b - a)))
    rep["bins"] = len(bins)
    B = np.array([b[1:4] for b in bins]) if bins else np.zeros((0, 3))
    az_cov = len({int((np.degrees(np.arctan2(b[2], b[1])) % 360) // 30) for b in bins}) * 30
    rep["azimuth_coverage_deg"] = int(az_cov)
    a_, b_, c_ = 0.0, 0.0, float(np.median(B[:, 2])) if len(B) else (float(np.median(z[sel])) if sel.any() else -2.0)
    method = "fit"
    if horizontal:
        method = "horizontal (asked)"
    elif len(B) < PLANE_MIN_BINS or az_cov < 90:
        method = f"horizontal ({len(B)} bins over {az_cov} deg of azimuth: too few to fit a tilt)"
    else:
        A = np.c_[B[:, 0], B[:, 1], np.ones(len(B))]
        w = np.ones(len(B))
        coef = np.array([0.0, 0.0, c_])
        for _ in range(20):                                    # Huber IRLS, scale 0.3 m
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(A * sw[:, None], B[:, 2] * sw, rcond=None)[0]
            res = B[:, 2] - A @ coef
            k = 1.345 * max(0.3, 1.4826 * float(np.median(np.abs(res - np.median(res)))))
            w = np.where(np.abs(res) <= k, 1.0, k / np.maximum(np.abs(res), 1e-12))
        tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
        if tilt > max_tilt_deg:
            method = f"horizontal (the fit tilted {tilt:.1f} deg > {max_tilt_deg:g})"
        else:
            a_, b_, c_ = (float(v) for v in coef)
    n = u - a_ * e1 - b_ * e2
    n /= np.linalg.norm(n)
    point = P + c_ * u
    grad = a_ * e1 + b_ * e2                                  # uphill, horizontal
    rep.update({"method": method, "normal": n.tolist(), "offset": float(n @ point), "point": point.tolist(),
                "height_below_pivot_m": float(-c_), "tilt_deg": float(np.degrees(np.arctan(np.hypot(a_, b_)))),
                "downhill_direction": (-grad / np.linalg.norm(grad)).tolist() if np.hypot(a_, b_) > 0 else None})
    if len(B):
        res = B[:, 2] - (a_ * B[:, 0] + b_ * B[:, 1] + c_)
        rr = np.array([b[0] for b in bins])
        rep["rms_m"] = float(np.sqrt(np.mean(res ** 2)))
        rep["by_ring"] = [{"ring_m": [float(rings[i]), float(rings[i + 1])], "bins": int((rr == i).sum()),
                           "median_dz_m": float(np.median(res[rr == i])), "p90_abs_dz_m": float(np.quantile(np.abs(res[rr == i]), 0.9))}
                          for i in range(len(rings) - 1) if (rr == i).any()]
    return rep


def clean_arrays(V: np.ndarray, F: np.ndarray, pivot: Sequence[float], points: Optional[np.ndarray] = None,
                 radius_m: float = DEFAULT_RADIUS_M, component_fraction: float = COMPONENT_FRACTION,
                 grazing: float = GRAZING, up: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Cleanup steps 1-4 on arrays: components, rubber sheets, support (when ``points`` are given), radius (the
    horizontal distance from the axis along ``up`` through the pivot; the 3-D range when ``up`` is None).
    Returns (V, F, report) with the triangles in and out per step."""
    P = np.asarray(pivot, float)
    rep: Dict[str, Any] = {"triangles_in": int(len(F)), "steps": []}
    # 1 components
    if len(F) and component_fraction > 0:
        lab = connected_components(F, len(V))
        counts = np.bincount(lab)
        keep = counts[lab] >= component_fraction * len(F)
        rep["steps"].append({"step": "components", "removed": int((~keep).sum()), "components": int(len(counts)),
                             "kept_components": int((counts >= component_fraction * len(F)).sum())})
        V, F = keep_triangles(V, F, keep)
    # 2 rubber sheets
    if len(F):
        cen, n, longest = triangle_geometry(V, F)
        v = cen - P
        r = np.linalg.norm(v, axis=1)
        v = v / np.where(r > 0, r, 1.0)[:, None]
        long_edge = longest > np.maximum(0.10, 4.0 * 3.0 * r * NAVCAM_HALF_IFOV_RAD)
        graze = np.abs(np.sum(n * v, axis=1)) < grazing
        keep = ~(long_edge & graze)
        rep["steps"].append({"step": "rubber_sheets", "removed": int((~keep).sum()), "long_edged": int(long_edge.sum()),
                             "grazing": int(graze.sum())})
        V, F = keep_triangles(V, F, keep)
    # 3 support
    if len(F) and points is not None and len(points):
        from scipy.spatial import cKDTree
        cen, _, longest = triangle_geometry(V, F)
        d, _ = cKDTree(np.asarray(points, float)).query(cen, k=1)
        r = np.linalg.norm(cen - P, axis=1)
        d_eff = np.maximum(0.0, d - 0.5 * longest)          # a triangle's own size: its centroid is never on a point
        keep = d_eff <= tau_m(r)
        rep["steps"].append({"step": "support", "removed": int((~keep).sum()), "median_distance_m": float(np.median(d))})
        V, F = keep_triangles(V, F, keep)
    # 4 radius
    if len(F) and radius_m:
        cen, _, _ = triangle_geometry(V, F)
        dist = horizontal_distance(cen, P, up) if up is not None else np.linalg.norm(cen - P, axis=1)
        keep = dist <= radius_m
        rep["steps"].append({"step": "radius", "removed": int((~keep).sum()), "radius_m": float(radius_m),
                             "measured": "from the mast axis (horizontal)" if up is not None else "from the pivot (3-D)"})
        V, F = keep_triangles(V, F, keep)
    rep["triangles_out"] = int(len(F))
    rep["vertices_out"] = int(len(V))
    return V, F, rep


def coverage_by_range(V: np.ndarray, F: np.ndarray, pivot: Sequence[float], rings=(0, 2, 5, 10, 20, 40, 60, 100)) -> list:
    """Triangles and area per range ring from the pivot (the report's coverage table)."""
    if not len(F):
        return []
    cen, _, _ = triangle_geometry(V, F)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    r = np.linalg.norm(cen - np.asarray(pivot, float), axis=1)
    out = []
    for lo, hi in zip(rings[:-1], rings[1:]):
        m = (r >= lo) & (r < hi)
        out.append({"range_m": [lo, hi], "triangles": int(m.sum()), "area_m2": float(area[m].sum())})
    return out


def sphere_arrays(pivot: Sequence[float], radius: float = DEFAULT_FAR_RADIUS_M, step_deg: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """A UV sphere about ``pivot`` with ~``step_deg`` facets, triangles facing inwards (seen from the pivot)."""
    P = np.asarray(pivot, float)
    n_az = max(8, int(round(360.0 / step_deg)))
    n_el = max(4, int(round(180.0 / step_deg)))
    az = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    el = np.linspace(-np.pi / 2, np.pi / 2, n_el + 1)
    V = [P + np.array([0.0, 0.0, -radius])]
    for e in el[1:-1]:
        ce, se = np.cos(e), np.sin(e)
        V.extend(P + radius * np.stack([ce * np.cos(az), ce * np.sin(az), np.full_like(az, se)], axis=1))
    V.append(P + np.array([0.0, 0.0, radius]))
    V = np.asarray(V)
    F = []
    rows = n_el - 1
    base = lambda k: 1 + k * n_az            # noqa: E731
    for j in range(n_az):
        F.append([0, base(0) + (j + 1) % n_az, base(0) + j])
    for k in range(rows - 1):
        for j in range(n_az):
            a, b = base(k) + j, base(k) + (j + 1) % n_az
            c, d = base(k + 1) + j, base(k + 1) + (j + 1) % n_az
            F.append([a, b, d])
            F.append([a, d, c])
    top = len(V) - 1
    for j in range(n_az):
        F.append([top, base(rows - 1) + j, base(rows - 1) + (j + 1) % n_az])
    return V, np.asarray(F, dtype=np.int64)


# ----------------------------------------------------------------------------- Open3D in and out
def _o3d():
    try:
        import open3d as o3d
    except ImportError as e:                                                        # pragma: no cover
        raise ImportError('mppp.ibr needs open3d (0.20 or later for Python 3.13): python -m pip install "open3d>=0.20"') from e
    return o3d


def to_o3d(V: np.ndarray, F: np.ndarray):
    o3d = _o3d()
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(V, float))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(F, np.int32))
    return m


def from_o3d(m) -> Tuple[np.ndarray, np.ndarray]:
    return np.asarray(m.vertices, float), np.asarray(m.triangles, np.int64)


def read_points(ply: PathLike) -> np.ndarray:
    """The points of a PLY (``fused.ply``) as (n, 3)."""
    return np.asarray(_o3d().io.read_point_cloud(str(ply)).points, float)


def read_points_normals(ply: PathLike) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(points (n, 3), normals (n, 3) or None) of a PLY: COLMAP's fused cloud carries normals oriented to its cameras."""
    pc = _o3d().io.read_point_cloud(str(ply))
    P = np.asarray(pc.points, float)
    return P, (np.asarray(pc.normals, float) if pc.has_normals() else None)


def read_mesh(ply: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    return from_o3d(_o3d().io.read_triangle_mesh(str(ply)))


def write_mesh(ply: PathLike, V: np.ndarray, F: np.ndarray) -> Path:
    o3d = _o3d()
    Path(ply).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(ply), to_o3d(V, F), write_ascii=False, compressed=False, print_progress=False)
    return Path(ply)


def poisson_arrays(points: np.ndarray, normals: Optional[np.ndarray] = None, depth: int = 10,
                   density_quantile: float = 0.02, viewpoint: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Open3D Poisson over the points, the lowest ``density_quantile`` of the vertices (the skin Poisson stretches
    over gaps) trimmed.  Normals: the given ones (the fused cloud's, oriented to the cameras by COLMAP); else
    estimated and turned towards ``viewpoint`` (the mast head) - 10 Nov: before, a cloud of more than 200,000
    points got unoriented normals, which Poisson turns into lumps and holes."""
    o3d = _o3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    nsrc = "given"
    if normals is not None and len(normals) == len(points) and np.isfinite(normals).all() and np.abs(normals).sum() > 0:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, float))
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        if viewpoint is not None:
            pcd.orient_normals_towards_camera_location(np.asarray(viewpoint, float))
            nsrc = "estimated, towards the mast head"
        elif len(points) < 200_000:
            pcd.orient_normals_consistent_tangent_plane(10)
            nsrc = "estimated, tangent-plane orientation"
        else:
            nsrc = "estimated, unoriented"
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    dens = np.asarray(dens)
    if len(dens) and density_quantile > 0:
        thr = np.quantile(dens, density_quantile)
        mesh.remove_vertices_by_mask(dens < thr)
    V, F = from_o3d(mesh)
    return V, F, {"method": "poisson (open3d, fallback)", "depth": int(depth), "density_quantile": density_quantile,
                  "normals": nsrc}


def zoned_poisson(points: np.ndarray, normals: Optional[np.ndarray], pivot: Sequence[float], up: Sequence[float],
                  radius_m: float, zones=MESH_ZONES, overlap_m: float = ZONE_OVERLAP_M,
                  clean: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Poisson per zone of horizontal distance from the axis (see MESH_ZONES), each part cleaned (``clean`` =
    clean_arrays' keywords) and clipped to its zone +- ``overlap_m``, the parts joined.  Returns (V, F, surface
    report, cleanup report)."""
    P = np.asarray(pivot, float)
    h = horizontal_distance(points, P, up)
    Vs, Fs, parts, cleans = [], [], [], []
    base = 0
    for lo, hi, depth in zones:
        outer = float(radius_m) if hi is None else min(float(hi), float(radius_m))
        if lo >= outer:
            continue
        sel = h <= outer + 1.0
        if lo > 0:
            sel &= h >= max(0.0, lo - 1.0)
        if sel.sum() < 500:
            parts.append({"zone_m": [lo, outer], "points": int(sel.sum()), "triangles": 0})
            continue
        V, F, how = poisson_arrays(points[sel], None if normals is None else normals[sel], depth=depth, viewpoint=P)
        last = hi is None or float(hi) >= float(radius_m)          # the outermost zone ends at the mesh radius itself
        V, F, crep = clean_arrays(V, F, P, points=points[sel], up=up, radius_m=outer if last else outer + overlap_m,
                                  **(clean or {}))
        if len(F):
            cen, _, _ = triangle_geometry(V, F)
            keep = horizontal_distance(cen, P, up) >= lo - (overlap_m if lo > 0 else 0.0)
            V, F = keep_triangles(V, F, keep)
        parts.append({"zone_m": [lo, outer], "points": int(sel.sum()), "depth": int(depth), "triangles": int(len(F)),
                      "normals": how.get("normals")})
        cleans.append({"zone_m": [lo, outer], **crep})
        Vs.append(V)
        Fs.append(F + base)
        base += len(V)
    V = np.vstack(Vs) if Vs else np.zeros((0, 3))
    F = np.vstack(Fs) if Fs else np.zeros((0, 3), np.int64)
    how = {"method": "poisson per zone (open3d)", "zones": parts, "overlap_m": overlap_m}
    crep = {"triangles_in": int(sum(c["triangles_in"] for c in cleans)), "steps": [s_ for c in cleans for s_ in
            [{**st, "zone_m": c["zone_m"]} for st in c["steps"]]], "triangles_out": int(len(F)), "vertices_out": int(len(V))}
    return V, F, how, crep


def delaunay_mesh(dense_dir: PathLike, out_ply: PathLike, colmap_exe: str = "colmap") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """COLMAP's Delaunay mesher over a dense workspace (``fused.ply`` + ``.vis``) through the executable."""
    cmd = [colmap_exe, "delaunay_mesher", "--input_type", "dense", "--input_path", str(dense_dir), "--output_path", str(out_ply)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not Path(out_ply).is_file():
        raise RuntimeError(f"colmap delaunay_mesher failed (exit {r.returncode}): {(r.stderr or r.stdout)[-400:]}")
    V, F = read_mesh(out_ply)
    return V, F, {"method": "delaunay (colmap)", "seconds": round(time.time() - t0, 1)}


def decimate_and_smooth(V: np.ndarray, F: np.ndarray, target_triangles: int = DEFAULT_TARGET_TRIANGLES,
                        taubin_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Quadric decimation to ``target_triangles`` (when the mesh is larger) and Taubin smoothing."""
    rep = {"triangles_before": int(len(F))}
    if not len(F):
        return V, F, rep
    m = to_o3d(V, F)
    if target_triangles and len(F) > target_triangles:
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(target_triangles))
    if taubin_iterations:
        m = m.filter_smooth_taubin(number_of_iterations=int(taubin_iterations))
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    V, F = from_o3d(m)
    rep.update(triangles_after=int(len(F)), taubin_iterations=int(taubin_iterations))
    return V, F, rep


# ----------------------------------------------------------------------------- the station's surface
def points_by_ring(points: np.ndarray, pivot: Sequence[float], up: Sequence[float] = (0, 0, 1),
                   rings=(0, 2, 5, 10, 15, 20, 30, 45, 65, 100)) -> list:
    """Fused points per ring of horizontal distance from the axis: how far the dense cloud reaches."""
    h = horizontal_distance(points, pivot, up) if len(points) else np.zeros(0)
    return [{"ring_m": [lo, hi], "points": int(((h >= lo) & (h < hi)).sum())} for lo, hi in zip(rings[:-1], rings[1:])]


ROVER_MODEL_PARTS = ("body", "arm", "arm_base", "suspension", "wheels", "antenna")     # of mppp.ibr.rover_model, with ``deck``
FLOAT_CLEARANCE_M = 1.0      # beyond FLOAT_BEYOND_M the surface is at least this far below the cameras
FLOAT_BEYOND_M = 3.0         # (horizontal distance from the mast axis; inside it the rover reaches up to the head)


def height_above_plane(X: np.ndarray, plane: Dict[str, Any], up: Sequence[float] = (0, 0, 1)) -> np.ndarray:
    """The vertical height of points above the ground plane (m)."""
    n = np.asarray(plane["normal"], float)
    u = np.asarray(up, float) / np.linalg.norm(up)
    return (np.asarray(X, float) @ n - float(plane["offset"])) / float(n @ u)


FLOAT_GROUND_CELL_M = 2.0     # 0.97.0: the local ground under a point - a grid of this cell size ...
FLOAT_GROUND_PERCENTILE = 20.0   # ... holding this percentile of its points' heights above the plane, median-smoothed
FLOAT_GROUND_MIN_POINTS = 5
FLOAT_GROUND_MAX_RISE = 0.30  # the local ground is taken no higher than this x the distance above the plane (17 deg)
FLOAT_SUPPORT_CELL_M = 0.5    # 0.97.0: a column of this size whose points climb from the ground ...
FLOAT_SUPPORT_GAP_M = 0.3     # ... in steps no larger than this is a boulder, not a floater
FLOAT_RANGE_FRACTION = 0.05   # far terrain: the allowed height above the local ground grows to this x the distance


def local_ground(points: np.ndarray, pivot: Sequence[float], up: Sequence[float], plane: Dict[str, Any],
                 cell_m: float = FLOAT_GROUND_CELL_M) -> Dict[str, Any]:
    """The ground under the terrain cloud as heights above the plane on a horizontal grid about the pivot: per cell
    the ``FLOAT_GROUND_PERCENTILE`` of its points (>= ``FLOAT_GROUND_MIN_POINTS``), then a 3 x 3 median over the
    filled cells, capped at ``FLOAT_GROUND_MAX_RISE`` x the distance.  {"cell_m", "origin", "e", "n", "grid"
    (NaN: no ground known there)} for :func:`ground_height`."""
    P = np.asarray(pivot, float)
    u = np.asarray(up, float) / np.linalg.norm(up)
    e = np.cross([0.0, 1.0, 0.0], u); e = e / np.linalg.norm(e)
    n = np.cross(u, e)
    X = np.asarray(points, float)
    d = X - P
    x, y = d @ e, d @ n
    h = height_above_plane(X, plane, up)
    R = max(float(np.hypot(x, y).max()) if len(X) else 1.0, cell_m) + cell_m
    m = int(np.ceil(2 * R / cell_m))
    i = np.clip(((x + R) / cell_m).astype(int), 0, m - 1); j = np.clip(((y + R) / cell_m).astype(int), 0, m - 1)
    grid = np.full((m, m), np.nan)
    if len(X):
        key = i * m + j
        order = np.argsort(key, kind="stable")
        ks, hs = key[order], h[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], len(ks)]
        for a, b in zip(starts, ends):
            if b - a >= FLOAT_GROUND_MIN_POINTS:
                grid[ks[a] // m, ks[a] % m] = np.percentile(hs[a:b], FLOAT_GROUND_PERCENTILE)
        pad = np.pad(grid, 1, constant_values=np.nan)
        st = np.stack([pad[1 + di:1 + di + m, 1 + dj:1 + dj + m] for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sm = np.nanmedian(st, axis=0)
        grid = np.where(np.isfinite(grid), sm, np.nan)
        cx = (np.arange(m) + 0.5) * cell_m - R
        rr = np.hypot(cx[:, None], cx[None, :])
        grid = np.minimum(grid, FLOAT_GROUND_MAX_RISE * rr)
    return {"cell_m": float(cell_m), "half_m": float(R), "e": e, "n": n, "grid": grid}


def ground_height(X: np.ndarray, pivot: Sequence[float], ground: Dict[str, Any]) -> np.ndarray:
    """The local ground (height above the plane, m) under points ``X``; 0 (the plane) where the grid has none in
    the point's cell or its eight neighbours."""
    d = np.asarray(X, float) - np.asarray(pivot, float)
    c, R, g = ground["cell_m"], ground["half_m"], ground["grid"]
    m = g.shape[0]
    i = np.clip(((d @ ground["e"] + R) / c).astype(int), 0, m - 1); j = np.clip(((d @ ground["n"] + R) / c).astype(int), 0, m - 1)
    out = g[i, j]
    miss = ~np.isfinite(out)
    if miss.any():
        pad = np.pad(g, 1, constant_values=np.nan)
        st = np.stack([pad[1 + di + i[miss], 1 + dj + j[miss]] for di in (-1, 0, 1) for dj in (-1, 0, 1)])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nb = np.nanmedian(st, axis=0)
        out = out.copy(); out[miss] = np.where(np.isfinite(nb), nb, 0.0)
    return out


def floating(X: np.ndarray, pivot: Sequence[float], up: Sequence[float], plane: Dict[str, Any],
             clearance_m: float = FLOAT_CLEARANCE_M, beyond_m: float = FLOAT_BEYOND_M,
             ground: Optional[Dict[str, Any]] = None, supported: bool = True) -> np.ndarray:
    """Points that cannot be surface seen from the mast (Christian, 10 Nov: "floating polygons should be removed ...
    everything above the cameras and every thing 1 meter below the camera beyond 3 meters"): above the cameras
    anywhere, or beyond ``beyond_m`` of the mast axis and higher than ``clearance_m`` below the cameras.

    Heights are above the ground.  Without ``ground`` that is the fitted plane (0.96).  On terrain that is not a
    plane the rule then removes the terrain itself: at Rockytop (plane rms 1.1 m, the camera 1.5 m above it) it took
    42 % of the fused points at 15-20 m and 71 % at 20-30 m, whole sectors of them, so the mesh ended at about 15 m
    there.  0.97.0: with ``ground`` (:func:`local_ground` of the terrain cloud) the height is above the local
    ground, and beyond ``beyond_m`` the allowed height is max(camera height - ``clearance_m``,
    ``FLOAT_RANGE_FRACTION`` x distance) - a 2 m rise in the ground 40 m away is terrain, 2 m above the ground at
    5 m is not."""
    X = np.asarray(X, float)
    hc = float(height_above_plane(np.asarray(pivot, float)[None], plane, up)[0])
    h = height_above_plane(X, plane, up)
    r = horizontal_distance(X, pivot, up)
    if ground is None:
        return (h > hc) | ((r > beyond_m) & (h > hc - clearance_m))
    hl = h - ground_height(X, pivot, ground)
    allow = np.maximum(hc - clearance_m, FLOAT_RANGE_FRACTION * r)
    fl = ((r <= beyond_m) & (h > hc)) | ((r > beyond_m) & (hl > allow))
    far = r > beyond_m
    if supported and (fl & far).any():
        # 0.97.0 (Christian, 12 Nov 2026, Rockytop: a 1.5 m boulder 8 m away, whole in the stereo depth map, had "a gap in
        # its center"): a boulder is not floating - its surface goes down to the ground.  In columns of
        # FLOAT_SUPPORT_CELL_M the points are followed upwards from the lowest one; while that one is within the allowed
        # height and no step is larger than FLOAT_SUPPORT_GAP_M the points are supported and stay.
        P0 = np.asarray(pivot, float)
        u = np.asarray(up, float) / np.linalg.norm(up)
        e_ = np.cross([0.0, 1.0, 0.0], u); e_ = e_ / np.linalg.norm(e_)
        n_ = np.cross(u, e_)
        idx = np.flatnonzero(far)
        d = X[idx] - P0
        ci = np.floor((d @ e_) / FLOAT_SUPPORT_CELL_M).astype(np.int64)
        cj = np.floor((d @ n_) / FLOAT_SUPPORT_CELL_M).astype(np.int64)
        key = (ci - ci.min()) * (int(cj.max() - cj.min()) + 1) + (cj - cj.min())
        order = np.lexsort((hl[idx], key))
        ks, hs, al = key[order], hl[idx][order], allow[idx][order]
        start = np.r_[True, ks[1:] != ks[:-1]]
        brk = np.r_[False, (hs[1:] - hs[:-1]) > FLOAT_SUPPORT_GAP_M] & ~start
        first = np.flatnonzero(start)
        seg = np.cumsum(start) - 1
        cs = np.cumsum(brk)
        ok = ((cs - cs[first][seg]) == 0) & (hs[first][seg] <= al[first][seg])
        keep = np.zeros(len(X), bool)
        keep[idx[order[ok]]] = True
        fl &= ~keep
    return fl


ROVER_FOOTPRINT_X_M = (-1.75, 1.60)   # RMECH x of the ground the rover stands over (rear of the RTG to the stowed turret)
ROVER_FOOTPRINT_Y_M = (-1.50, 1.50)   # RMECH y: outside the wheels (track about 2.7 m)
FILL_AZ_STEP_DEG = 2.0                # the fill's polar grid about the mast axis: azimuth step ...
FILL_RING0_M = 0.4                    # ... and rings from here, each FILL_RING_RATIO x the last, to the mesh radius
FILL_RING_RATIO = 1.06
# 0.97.0 (Christian, 12 Nov 2026, Van Zyl: "the mesh preview still has some holes near the horizon ... so that they
# don't appear as holes in the stereo view"): far from the mast the Poisson mesh is ragged - gaps smaller than a grid
# cell, which the hole rule below does not see.  From FILL_UNDERLAY_FROM_M on every cell of the grid gets its two
# triangles, FILL_UNDERLAY_DROP_M below the cell heights: a ray through a gap ends on it, at the terrain's range.
# Past the mesh radius the grid goes on to FILL_APRON_TO_M (the ground plane's reach), its height falling to the plane's:
# mesh and plane meet without a step.
FILL_UNDERLAY_FROM_M = 20.0
FILL_UNDERLAY_DROP_M = 0.05
FILL_APRON_TO_M = 200.0
# 0.97.13: nearer than FILL_UNDERLAY_FROM_M every cell gets a second sheet of the interpolated surface,
# FILL_NEAR_DROP_M below the cell heights (deep enough not to cover a hollow beside a rock; a ray through a gap of the
# mesh ends on it, not on the regional ground plane, which no longer reaches within raycast.PLANE_FROM_M of the axis).
FILL_NEAR_DROP_M = 0.20
FILL_MIN_COVER = 0.6              # a cell counts as covered when its triangles' area is this share of the cell's
FILL_ROVER_REF_RADIUS_M = 5.0    # rover_keep_m: the ground's height from the triangles within this of the axis, off the footprint


def under_rover(X: np.ndarray, pose: Dict[str, Any]) -> np.ndarray:
    """Points over the rover's footprint on the ground (``pose``: :func:`mppp.ibr.rover_deck.rover_pose`)."""
    from .rover_deck import block_to_mech
    M = block_to_mech(np.asarray(X, float), pose)
    return ((M[:, 0] >= ROVER_FOOTPRINT_X_M[0]) & (M[:, 0] <= ROVER_FOOTPRINT_X_M[1])
            & (M[:, 1] >= ROVER_FOOTPRINT_Y_M[0]) & (M[:, 1] <= ROVER_FOOTPRINT_Y_M[1]))


def fill_terrain(V: np.ndarray, F: np.ndarray, pivot: Sequence[float], up: Sequence[float], plane: Dict[str, Any],
                 radius_m: float, rover: Optional[Dict[str, Any]] = None,
                 underlay: bool = True, rover_keep_m: Optional[float] = None,
                 apron_to_m: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    0.97.0 (Christian, 12 Nov 2026: "interpolate terrain under the rover so that it is flattish terrain without
    holes. The far field should interpolate over hole in the terrain. This is especially important behind the
    rover"): the terrain mesh made whole out to ``radius_m``.

    A polar grid about the mast axis (``FILL_AZ_STEP_DEG`` x rings growing by ``FILL_RING_RATIO``).  A cell is
    *covered* when a triangle's centroid lies in it; its height is the median of those centroids above the plane.
    With ``rover`` (the rover pose) the triangles over the rover's footprint are removed first, so that ground is
    interpolated, not meshed from what the masks left.  The heights of the grid's nodes are interpolated linearly
    over the covered cells (nearest beyond their hull) - flat between the edges of a hole - and every cell that is
    not covered, and its neighbours, get two triangles.  Returns (V, F, {"cells", "covered", "filled", "triangles",
    "under_rover_triangles_removed", "filled_by_ring"}).
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    P = np.asarray(pivot, float)
    u = np.asarray(up, float) / np.linalg.norm(up)
    e = np.cross([0.0, 1.0, 0.0], u); e = e / np.linalg.norm(e)
    n = np.cross(u, e)
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    rep: Dict[str, Any] = {"under_rover_triangles_removed": 0}
    if rover is not None and len(F):
        cen, _, _ = triangle_geometry(V, F)
        drop = under_rover(cen, rover)
        if rover_keep_m and drop.any():
            # 0.97.0: with the neighbours' frames in the cloud the ground under the rover is measured - what lies
            # within rover_keep_m of the ground's height around the footprint stays; what is higher (deck, belly,
            # mask leaks) or lower goes
            h = height_above_plane(cen, plane, up)
            dP = cen - P
            near = (~drop) & (np.hypot(dP @ e, dP @ n) < FILL_ROVER_REF_RADIUS_M)
            if near.sum() >= 50:
                ref = float(np.median(h[near]))
                keep_under = drop & (np.abs(h - ref) <= float(rover_keep_m))
                rep["under_rover_triangles_kept"] = int(keep_under.sum())
                rep["under_rover_ground_ref_m"] = ref
                drop &= ~keep_under
        rep["under_rover_triangles_removed"] = int(drop.sum())
        V, F = keep_triangles(V, F, ~drop)
    edges = [0.0, FILL_RING0_M]
    while edges[-1] < radius_m:
        edges.append(min(edges[-1] * FILL_RING_RATIO, float(radius_m)))
    apron_m = max(float(radius_m), float(apron_to_m if apron_to_m is not None else FILL_APRON_TO_M)) if underlay else float(radius_m)
    while edges[-1] < apron_m:                                  # the apron: past the mesh, down to the plane
        edges.append(min(edges[-1] * FILL_RING_RATIO, apron_m))
    edges = np.array(edges)
    na, nr = int(round(360.0 / FILL_AZ_STEP_DEG)), len(edges) - 1
    hits = np.zeros((nr, na), int)
    hsum: Dict[Tuple[int, int], list] = {}
    if len(F):
        cen, _, _ = triangle_geometry(V, F)
        d = cen - P
        x, y = d @ e, d @ n
        r = np.hypot(x, y)
        ok = r < radius_m
        ir = np.clip(np.searchsorted(edges, r[ok], side="right") - 1, 0, nr - 1)
        ia = (np.floor(np.degrees(np.arctan2(y[ok], x[ok])) % 360.0 / FILL_AZ_STEP_DEG).astype(int)) % na
        h = height_above_plane(cen[ok], plane, up)
        key = ir * na + ia
        order = np.argsort(key, kind="stable")
        ks, hs = key[order], h[order]
        st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        en = np.r_[st[1:], len(ks)]
        med = np.full((nr, na), np.nan)
        for a_, b_ in zip(st, en):
            med[ks[a_] // na, ks[a_] % na] = np.median(hs[a_:b_])
            hits[ks[a_] // na, ks[a_] % na] = b_ - a_
    else:
        med = np.full((nr, na), np.nan)
    covered = hits > 0
    if len(F):                                                   # 0.97.0: a cell is covered by area, not by one triangle -
        a3, b3, c3 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]          # at a hole's rim (the rover's footprint) a few triangles
        ar = 0.5 * np.linalg.norm(np.cross(b3 - a3, c3 - a3), axis=1)[ok]      # in a cell left gaps to the plane
        asum = np.bincount(key, weights=ar, minlength=nr * na).reshape(nr, na)
        cell_area = (0.5 * (edges[1:] ** 2 - edges[:-1] ** 2) * np.radians(FILL_AZ_STEP_DEG))[:, None]
        covered &= asum >= FILL_MIN_COVER * cell_area
    rc = 0.5 * (edges[:-1] + edges[1:])
    ac = np.radians((np.arange(na) + 0.5) * FILL_AZ_STEP_DEG)
    cx, cy = rc[:, None] * np.cos(ac)[None, :], rc[:, None] * np.sin(ac)[None, :]
    an = np.radians(np.arange(na) * FILL_AZ_STEP_DEG)
    nx, ny = edges[:, None] * np.cos(an)[None, :], edges[:, None] * np.sin(an)[None, :]      # nodes (nr + 1, na)
    if covered.sum() >= 3:
        pts = np.c_[cx[covered], cy[covered]]
        lin = LinearNDInterpolator(pts, med[covered])
        hn = lin(nx, ny)
        miss = ~np.isfinite(hn)
        if miss.any():
            hn[miss] = NearestNDInterpolator(pts, med[covered])(nx[miss], ny[miss])
    else:
        hn = np.zeros_like(nx)                                   # no terrain at all: the plane
    hn[0, :] = np.mean(hn[0, :])                                 # the axis is one point
    if underlay:
        rn = np.broadcast_to(edges[:, None], hn.shape)
        far = np.clip((rn - FILL_UNDERLAY_FROM_M) / max(0.1 * FILL_UNDERLAY_FROM_M, 1e-6), 0.0, 1.0)
        hn = hn - FILL_UNDERLAY_DROP_M * far                     # just under the terrain, from the underlay's start
        if apron_m > radius_m:                                   # the apron's height: the rim's, falling to the plane
            t = np.clip((rn - radius_m) / (apron_m - radius_m), 0.0, 1.0)
            hn = hn * (0.5 + 0.5 * np.cos(np.pi * t))
    nu = np.asarray(plane["normal"], float)
    Xn = P[None, None, :] + nx[..., None] * e + ny[..., None] * n
    # the node on the vertical through (x, y) at height hn above the plane
    h0 = height_above_plane(Xn.reshape(-1, 3), plane, up).reshape(nx.shape)
    Xn = Xn + (hn - h0)[..., None] * u
    Vf = Xn.reshape(-1, 3)
    idx = lambda i, j: i * na + (j % na)                         # noqa: E731
    Ff = []
    hole = ~covered
    need = hole.copy()                                           # and the covered cells around a hole: a mesh ends
    need |= np.roll(hole, 1, axis=1) | np.roll(hole, -1, axis=1)  # inside its cell, and a ray must not pass between
    need[1:] |= hole[:-1]
    need[:-1] |= hole[1:]
    if underlay:
        need |= (edges[:-1] >= FILL_UNDERLAY_FROM_M)[:, None]
    ii, jj = np.nonzero(need)
    for i, j in zip(ii, jj):
        a_, b_, c_, d_ = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
        if i == 0:
            Ff.append((a_, c_, d_))                              # the innermost ring: triangles from the axis
        else:
            Ff.append((a_, b_, c_)); Ff.append((a_, c_, d_))
    Ff = np.asarray(Ff, np.int64).reshape(-1, 3)
    near_tris = 0
    if underlay:                                                 # 0.97.13: a second sheet under every near cell - see
        inner = edges[:-1] < FILL_UNDERLAY_FROM_M                # FILL_NEAR_DROP_M; it rises to the first sheet at the
        ii, jj = np.nonzero(np.broadcast_to(inner[:, None], need.shape))    # ring where the far underlay starts: no slit
        Fn = []
        off = len(Vf)
        for i, j in zip(ii, jj):
            a_, b_, c_, d_ = (off + q for q in (idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)))
            if i == 0:
                Fn.append((a_, c_, d_))
            else:
                Fn.append((a_, b_, c_)); Fn.append((a_, c_, d_))
        if Fn:
            r_end = float(edges[int(inner.sum())])
            rn_ = np.broadcast_to(edges[:, None], nx.shape)
            taper = np.clip((r_end - rn_) / max(0.25 * r_end, 1e-6), 0.0, 1.0)
            Vf = np.vstack([Vf, (Xn - (FILL_NEAR_DROP_M * taper)[..., None] * u).reshape(-1, 3)])
            Ff = np.vstack([Ff, np.asarray(Fn, np.int64)])
            near_tris = len(Fn)
    rings = (0, 3, 10, 20, 30, 50, 1e9)
    inside = rc < radius_m                                       # the report: the mesh's own reach, not the apron
    by_ring = [{"ring_m": [lo, min(hi, float(radius_m))], "cells": int((inside & (rc >= lo) & (rc < hi)).sum() * na),
                "filled": int(hole[inside & (rc >= lo) & (rc < hi)].sum())} for lo, hi in zip(rings[:-1], rings[1:])
               if lo < radius_m]
    rep.update(cells=int(inside.sum() * na), covered=int(covered.sum()), filled=int(hole[inside].sum()), triangles=int(len(Ff)),
               underlay=({"from_m": FILL_UNDERLAY_FROM_M, "drop_m": FILL_UNDERLAY_DROP_M, "apron_to_m": apron_m,
                          "near_drop_m": FILL_NEAR_DROP_M, "near_triangles": int(near_tris)}
                         if underlay else None),
               filled_by_ring=by_ring, grid={"az_step_deg": FILL_AZ_STEP_DEG, "rings": int(nr)})
    if len(Ff):
        F = np.vstack([F, Ff + len(V)])
        V = np.vstack([V, Vf])
    return V, F, rep


def build_station_mesh(points: np.ndarray, pivot: Sequence[float], out_dir: PathLike, normals: Optional[np.ndarray] = None,
                       dense_dir: Optional[PathLike] = None, colmap_exe: Optional[str] = None,
                       radius_m: float = DEFAULT_RADIUS_M, far_radius_m: Optional[float] = None,
                       target_triangles: int = DEFAULT_TARGET_TRIANGLES, poisson_depth: int = 10,
                       station: Optional[Dict[str, Any]] = None, up: Sequence[float] = (0, 0, 1),
                       plane_radius_m: float = DEFAULT_PLANE_RADIUS_M, ground_points: Optional[np.ndarray] = None,
                       horizontal_plane: bool = False, plane_points: Optional[np.ndarray] = None,
                       deck: Optional[Dict[str, Any]] = None, remove_floating: bool = True,
                       rover_model_parts: Optional[Sequence[str]] = ROVER_MODEL_PARTS, terrain_only: bool = False,
                       fill: bool = True, rover_pose: Optional[Dict[str, Any]] = None,
                       rover_keep_m: Optional[float] = None) -> Dict[str, Any]:
    """The whole of step 2 for one station: the surface (Delaunay through ``colmap_exe`` over ``dense_dir`` when
    given, else Poisson over the ``points`` within ``radius_m`` + 5 m of the axis), the cleanup (the mesh to
    ``radius_m`` from the axis), the decimation, the ground plane (:func:`fit_ground_plane` over ``points`` and
    ``ground_points``, the block's tie points) out to ``plane_radius_m``, then the sky at infinity - or a finite far
    sphere when ``far_radius_m`` is given (the 0.96.0 prototype).  ``mesh.ply``, ``ground_plane.json`` (and
    ``far_sphere.ply``) and ``mesh_report.json`` in ``out_dir``.  ``plane_points``: the terrain cloud for the plane
    when ``points`` hold more than terrain (``mppp mesh --rover``: the rover-masked cloud, not the rover's deck).
    ``deck``: the rover pose of :func:`mppp.ibr.rover_deck.rover_pose` (``--rover``): nothing below the deck top
    inside its footprint - the points there dropped, the vertices lifted onto it, and the deck floor added.
    ``remove_floating``: :func:`floating` points and triangles removed.  ``rover_model_parts`` (with ``deck``): the
    parts of the Perseverance model (:mod:`mppp.ibr.rover_model`) added to the mesh, the fused points on them
    dropped; None or empty: no model (the deck floors alone).  0.97.0: ``terrain_only`` - ``points`` are the
    rover-masked cloud (terrain alone): nothing of the rover is meshed from points, the deck floor is not needed, and
    with ``deck`` the Perseverance model is added to the finished terrain (the unmasked images then project onto
    it).  ``fill`` (:func:`fill_terrain`): the terrain interpolated under the rover (``rover_pose``, default
    ``deck``) and over every hole out to ``radius_m``.  Returns the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    points = np.asarray(points, float)
    deck_rep = None
    use_model = False
    if deck is not None and terrain_only:                       # 0.97.0: the terrain alone; the model comes after
        from . import rover_model
        use_model = bool(rover_model_parts) and rover_model.available()
        deck_rep = {"terrain_only": True}
    elif deck is not None:                                      # 10 Nov: a floor at the rover deck
        from .rover_deck import below_deck
        from . import rover_model
        use_model = bool(rover_model_parts) and rover_model.available()
        drop = below_deck(points, deck)
        deck_rep = {"points_below_dropped": int(drop.sum())}
        if use_model:                                           # the rover model stands for the points on it
            near = rover_model.near_model(points, deck, parts=rover_model_parts)
            deck_rep["points_on_model_dropped"] = int((near & ~drop).sum())
            drop |= near
            mast = rover_model.near_model(points, deck, near_m=rover_model.MAST_NEAR_M, parts=("mast",))
            deck_rep["points_on_mast_dropped"] = int((mast & ~drop).sum())     # the mast and its booms: not in the mesh
            drop |= mast
        if drop.any():
            points = points[~drop]
            if normals is not None:
                normals = np.asarray(normals)[~drop]
    all_points = points
    base = all_points if plane_points is None else np.asarray(plane_points, float)
    gp_in = base if ground_points is None or not len(ground_points) else np.vstack([base, np.asarray(ground_points, float)])
    plane = fit_ground_plane(gp_in, pivot, up, radius_m=plane_radius_m, horizontal=horizontal_plane)
    plane["sources"] = {"fused_points": int(len(base)), "tie_points": int(0 if ground_points is None else len(ground_points))}
    h = horizontal_distance(points, pivot, up)
    near = h <= radius_m + 5.0
    float_rep = None
    ground = None
    if remove_floating:                                         # 10 Nov: nothing floating above the terrain
        ground = local_ground(base[horizontal_distance(base, pivot, up) <= radius_m + 5.0], pivot, up, plane)   # 0.97.0
        fl = floating(points, pivot, up, plane, ground=ground)
        float_rep = {"above_camera_m": 0.0, "clearance_below_camera_m": FLOAT_CLEARANCE_M, "beyond_m": FLOAT_BEYOND_M,
                     "heights_from": f"the local ground ({FLOAT_GROUND_CELL_M:g} m grid)", "range_fraction": FLOAT_RANGE_FRACTION,
                     "camera_height_above_plane_m": float(height_above_plane(np.asarray(pivot, float)[None], plane, up)[0]),
                     "points_dropped": int((fl & near).sum())}
        near &= ~fl
    points = points[near]
    if normals is not None:
        normals = np.asarray(normals)[near]
    if dense_dir is not None and colmap_exe:
        try:
            V, F, how = delaunay_mesh(dense_dir, out / "mesh_raw.ply", colmap_exe)
        except Exception as e:                                                      # noqa: BLE001
            log.warning("mesh: Delaunay failed (%s) - Poisson instead", e)
            V, F, how = poisson_arrays(points, normals, depth=poisson_depth, viewpoint=pivot)
            how["delaunay_error"] = str(e)[:300]
        V, F, clean = clean_arrays(V, F, pivot, points=points, radius_m=radius_m, up=up)
    else:
        zones = tuple((lo, hi, int(poisson_depth)) for lo, hi, _ in MESH_ZONES)
        V, F, how, clean = zoned_poisson(points, normals, pivot, up, radius_m, zones=zones)
    V, F, dec = decimate_and_smooth(V, F, target_triangles=target_triangles)
    if remove_floating and len(F):
        bad = floating(V, pivot, up, plane, ground=ground)
        dropf = bad[np.asarray(F)].any(axis=1)
        F = np.asarray(F)[~dropf]
        float_rep["triangles_dropped"] = int(dropf.sum())
        log.info("mesh: floating surface - %d points and %d triangles removed (above the cameras, or less than %g m below "
                 "them beyond %g m; heights from the ground plane)", float_rep["points_dropped"], int(dropf.sum()),
                 FLOAT_CLEARANCE_M, FLOAT_BEYOND_M)
    rover_tris = None
    fill_rep = None
    if fill:                                                    # 0.97.0: no holes, flattish ground under the rover
        V, F, fill_rep = fill_terrain(V, F, pivot, up, plane, radius_m, rover=rover_pose if rover_pose is not None else deck,
                                      rover_keep_m=rover_keep_m, apron_to_m=plane_radius_m)
        log.info("mesh: terrain fill - %d of %d cells had no surface (%d triangles); %d triangles under the rover replaced",
                 fill_rep["filled"], fill_rep["cells"], fill_rep["triangles"], fill_rep["under_rover_triangles_removed"])
    if deck is not None and terrain_only:
        if use_model:
            Vm, Fm = rover_model.model_in_block(deck, parts=rover_model_parts)
            rover_tris = [int(len(F)), int(len(F) + len(Fm))]       # the model's triangles: last in mesh.ply
            F = np.vstack([np.asarray(F, np.int64), np.asarray(Fm, np.int64) + len(V)])
            V = np.vstack([V, Vm])
            deck_rep.update(model_triangles=int(len(Fm)), model_parts=list(rover_model_parts))
            log.info("mesh: the rover model on the terrain - %d triangles (%s)", len(Fm), ", ".join(rover_model_parts))
        else:
            from .rover_deck import REAR_X_M, REAR_Y_M, deck_floor_arrays
            for kw in ({}, {"x_m": REAR_X_M, "y_m": REAR_Y_M}):
                Vd, Fd = deck_floor_arrays(deck, **kw)
                F = np.vstack([np.asarray(F, np.int64), np.asarray(Fd, np.int64) + len(V)])
                V = np.vstack([V, Vd])
            deck_rep["model"] = "not available: the deck floors alone"
    elif deck is not None:
        from .rover_deck import clamp_to_deck, deck_floor_arrays
        V, F, moved, cut = clamp_to_deck(V, deck, F)
        deck_rep["triangles_below_dropped"] = cut
        Vd, Fd = deck_floor_arrays(deck)
        F = np.vstack([np.asarray(F, np.int64), np.asarray(Fd, np.int64) + len(V)])
        V = np.vstack([V, Vd])
        if use_model:                                           # the Perseverance model, placed by the rover pose
            Vm, Fm = rover_model.model_in_block(deck, parts=rover_model_parts)
            F = np.vstack([F, np.asarray(Fm, np.int64) + len(V)])
            V = np.vstack([V, Vm])
            deck_rep.update(model_triangles=int(len(Fm)), model_parts=list(rover_model_parts))
            log.info("mesh: the rover model - %d points on it dropped, %d triangles added (%s)",
                     deck_rep["points_on_model_dropped"], len(Fm), ", ".join(rover_model_parts))
        else:
            from .rover_deck import REAR_X_M, REAR_Y_M          # the second floor behind the deck (holes in the back)
            Vr, Fr = deck_floor_arrays(deck, x_m=REAR_X_M, y_m=REAR_Y_M)
            F = np.vstack([F, np.asarray(Fr, np.int64) + len(V)])
            V = np.vstack([V, Vr])
            deck_rep["rear_floor_triangles"] = int(len(Fr))
        deck_rep.update(vertices_lifted=moved, floor_triangles=int(len(Fd)))
        log.info("mesh: the rover deck floor - %d points below it dropped, %d vertices lifted onto it, %d floor triangles",
                 deck_rep["points_below_dropped"], moved, len(Fd))
    write_mesh(out / "mesh.ply", V, F)
    (out / "ground_plane.json").write_text(json.dumps(plane, indent=1, default=float), encoding="utf-8")
    files = {"mesh": str(out / "mesh.ply"), "ground_plane": str(out / "ground_plane.json")}
    sphere = None
    if far_radius_m:
        Vs, Fs = sphere_arrays(pivot, far_radius_m)
        write_mesh(out / "far_sphere.ply", Vs, Fs)
        files["far_sphere"] = str(out / "far_sphere.ply")
        sphere = {"radius_m": far_radius_m, "triangles": int(len(Fs))}
    elif (out / "far_sphere.ply").is_file():
        (out / "far_sphere.ply").unlink()                       # a stale prototype sphere would hide the sky
    rep = {"station": station, "pivot_block_m": [float(v) for v in np.asarray(pivot, float)], "surface": how,
           "points": int(len(all_points)), "points_meshed": int(len(points)), "points_by_ring": points_by_ring(all_points, pivot, up),
           "cleanup": clean, "decimation": dec,
           "scene": {"mesh_radius_m": float(radius_m), "mesh_radius_from": "the mast axis (horizontal)",
                     "plane": {k: plane[k] for k in ("method", "radius_m", "tilt_deg", "height_below_pivot_m", "rms_m", "bins")
                               if k in plane},
                     "beyond": "a far sphere" if sphere else "the sky at infinity"},
           "far_sphere": sphere, "rover_deck": deck_rep, "rover_triangles": rover_tris, "fill": fill_rep, "floating": float_rep, "coverage_by_range": coverage_by_range(V, F, pivot), "tau_m": {str(z): float(tau_m(z))
           for z in (2, 5, 10, 15, 20, 30, 50, 85)}, "files": files, "seconds": round(time.time() - t0, 1)}
    (out / "mesh_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    log.info("mesh: %d -> %d triangles (%s) within %g m of the axis; ground plane %s (tilt %.1f deg, %.2f m rms over %d "
             "bins) to %g m; then %s -> %s", clean["triangles_in"], len(F), how["method"], radius_m, plane["method"],
             plane["tilt_deg"], plane.get("rms_m", float("nan")), plane["bins"], plane_radius_m,
             rep["scene"]["beyond"], out)
    return rep


def load_scene(out_dir: PathLike):
    """The :class:`mppp.ibr.raycast.Scene` of a station folder: the mesh, the ground plane and the sky (or the far
    sphere of a 0.96.0 prototype folder without ``ground_plane.json``)."""
    from .raycast import Scene
    out = Path(out_dir)
    if not (out / "mesh.ply").is_file():
        raise FileNotFoundError(f"{out / 'mesh.ply'}: run mppp mesh first")
    mesh = read_mesh(out / "mesh.ply")
    sphere = read_mesh(out / "far_sphere.ply") if (out / "far_sphere.ply").is_file() else None
    gp = out / "ground_plane.json"
    if gp.is_file():
        plane = json.loads(gp.read_text(encoding="utf-8"))
        sc = Scene(mesh, sphere, plane=plane, sky=sphere is None)
    else:
        sc = Scene(mesh, sphere)
    mr = out / "mesh_report.json"                           # 0.97.0: which triangles are the rover model
    if mr.is_file():
        try:
            mrep = json.loads(mr.read_text(encoding="utf-8"))
            rt = mrep.get("rover_triangles")
            sc.mesh_is_terrain = mrep.get("rover") == "masked out"      # 0.97.15: a mesh of the rover-masked cloud
            if rt and int(rt[1]) > int(rt[0]) and int(rt[1]) <= len(mesh[1]):
                sc.rover_triangles = (int(rt[0]), int(rt[1]))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return sc


PREVIEW_HALF_WIDTH_M = 75.0          # the views from above of mesh_preview.png: +/- this, east and north


def mesh_preview(V: np.ndarray, F: np.ndarray, pivot: Sequence[float], out_png: PathLike, sphere: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 radius_m: float = DEFAULT_RADIUS_M, scene=None, up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0)) -> Path:
    """``mesh_preview.png``: two views from above - the vertices coloured by height (left) and by range from the pivot
    (right), both linear, east to the right and north up, +/- ``PREVIEW_HALF_WIDTH_M`` - and under them the range
    panorama from the pivot (0.97.8, Christian 13 Nov 2026: "two subplots in the top looking down ... height linear
    on the left and range on the right.  The x and y limits should be +/- 50 meters")."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .raycast import Scene, panorama_range
    P = np.asarray(pivot, float)
    L = PREVIEW_HALF_WIDTH_M
    fig = plt.figure(figsize=(11, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax_h, ax_r, ax_p = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    axes = [None, ax_p]
    if len(V):
        d = V - P
        e, n_, h = d @ np.asarray(east, float), d @ np.asarray(north, float), d @ np.asarray(up, float)
        box = (np.abs(e) <= L) & (np.abs(n_) <= L)
        idx = np.flatnonzero(box)
        if len(idx) > 300_000:
            idx = np.random.default_rng(0).choice(idx, size=300_000, replace=False)
        r = np.linalg.norm(d[idx], axis=1)
        ring = (np.hypot(e[idx], n_[idx]) > 3.0) & (np.hypot(e[idx], n_[idx]) < 10.0)
        h0 = float(np.median(h[idx][ring])) if ring.sum() > 50 else float(np.median(h[idx])) if len(idx) else 0.0
        hh = h[idx] - h0                                     # height above the ground 3 to 10 m from the mast
        lim = float(max(0.5, np.percentile(np.abs(hh), 98))) if len(idx) else 1.0
        order = np.argsort(hh)                               # the higher points drawn last
        sc = ax_h.scatter(e[idx][order], n_[idx][order], c=hh[order], s=0.3, cmap="turbo", vmin=-lim, vmax=lim, rasterized=True)
        plt.colorbar(sc, ax=ax_h, label="height above the ground near the rover (m)", shrink=0.85)
        sc = ax_r.scatter(e[idx], n_[idx], c=r, s=0.3, cmap="turbo", vmin=0, vmax=float(np.hypot(L, L)), rasterized=True)
        plt.colorbar(sc, ax=ax_r, label="range from the pivot (m)", shrink=0.85)
    for ax, title in ((ax_h, "height"), (ax_r, "range")):
        ax.plot(0, 0, "k+", ms=12)
        ax.set_aspect("equal")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_xlabel("east of the pivot (m)")
        ax.set_ylabel("north of the pivot (m)")
        ax.set_title(f"from above: {title}" + (f" ({len(F)} triangles)" if title == "height" else ""))
    if scene is None:
        scene = Scene((V, F), sphere)
    pan = panorama_range(scene, P, width=1440, up=up, north=north, east=east)
    show = np.log10(np.clip(pan["range"], 0.5, 200))
    show = np.ma.masked_where(~np.isfinite(pan["range"]) | (pan["range"] <= 0), show)
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("white")
    im = axes[1].imshow(show, cmap=cmap, aspect="auto", vmin=np.log10(0.5), vmax=np.log10(200))
    from .raycast import LABEL_PLANE
    if (pan["label"] == LABEL_PLANE).any():
        axes[1].contour((pan["label"] == LABEL_PLANE).astype(float), levels=[0.5], colors="k", linewidths=0.6)
    axes[1].set_title("range from the pivot (log10 m), equirect: mesh, ground plane (outlined), sky at infinity white; black = no data")
    axes[1].set_xticks([0, 360, 720, 1080, 1439])
    axes[1].set_xticklabels(["N", "E", "S", "W", "N"])
    axes[1].set_yticks([])
    plt.colorbar(im, ax=axes[1], label="log10 range (m)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return Path(out_png)
