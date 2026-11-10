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
4. radius: terrain farther than ``radius_m`` (20 m) from the mast's axis - the vertical through the pivot,
   horizontal distance - is dropped (Christian, 10 Nov 2026: "cut off at 20 m from the axis of the mast rotation");
   the ground plane takes over beyond it;
5. quadric decimation to ``target_triangles`` and five Taubin smoothing iterations (no shrinkage).

Beyond the mesh (10 Nov 2026): **the ground plane** out to ``plane_radius_m`` (100 m) from the axis, then **the sky
at infinity** (:mod:`.raycast`).  The plane (:func:`fit_ground_plane`) is fitted to the terrain the station sees,
the fused points and the block's tie points 3-100 m from the axis: the median point of every 10 deg x range-ring
bin, then a Huber line fit z = a x + b y + c over the bins, so the near field's density does not outvote the far
rings and an outlier bin does not tilt it.  It follows the regional slope (a horizontal plane on a 10 deg slope is
17 m off at 100 m); ``horizontal=True``, or a tilt over 20 deg, or fewer than 8 bins, gives the horizontal plane at
the median height instead.  Its misfit per ring is in ``ground_plane.json``; the tie points on it are checked again
in the depth maps' report.  The mesh's holes within 20 m show the plane too.  Beyond 100 m the 64 mm disparity of an
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
DEFAULT_RADIUS_M = 20.0             # the mesh: horizontal distance from the mast axis (10 Nov; was 60 m of range)
DEFAULT_PLANE_RADIUS_M = 100.0      # the ground plane out to here (horizontal); then the sky at infinity
DEFAULT_FAR_RADIUS_M = 100.0        # the 0.96.0 prototype's finite far sphere (mppp mesh --far-radius)
PLANE_RINGS_M = (3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 65.0, 100.0)
PLANE_SECTOR_DEG = 10.0
PLANE_MAX_TILT_DEG = 20.0
PLANE_MIN_BINS = 8
PLANE_MIN_POINTS_PER_BIN = 5
DEFAULT_TARGET_TRIANGLES = 3_000_000
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


def read_mesh(ply: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    return from_o3d(_o3d().io.read_triangle_mesh(str(ply)))


def write_mesh(ply: PathLike, V: np.ndarray, F: np.ndarray) -> Path:
    o3d = _o3d()
    Path(ply).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(ply), to_o3d(V, F), write_ascii=False, compressed=False, print_progress=False)
    return Path(ply)


def poisson_arrays(points: np.ndarray, normals: Optional[np.ndarray] = None, depth: int = 10,
                   density_quantile: float = 0.02) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Open3D Poisson over the points (normals estimated when missing), the lowest ``density_quantile`` of the
    vertices (the skin Poisson stretches over gaps) trimmed."""
    o3d = _o3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    if normals is not None and len(normals) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, float))
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(10) if len(points) < 200_000 else None
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    dens = np.asarray(dens)
    if len(dens) and density_quantile > 0:
        thr = np.quantile(dens, density_quantile)
        mesh.remove_vertices_by_mask(dens < thr)
    V, F = from_o3d(mesh)
    return V, F, {"method": "poisson (open3d, fallback)", "depth": int(depth), "density_quantile": density_quantile}


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


def build_station_mesh(points: np.ndarray, pivot: Sequence[float], out_dir: PathLike, normals: Optional[np.ndarray] = None,
                       dense_dir: Optional[PathLike] = None, colmap_exe: Optional[str] = None,
                       radius_m: float = DEFAULT_RADIUS_M, far_radius_m: Optional[float] = None,
                       target_triangles: int = DEFAULT_TARGET_TRIANGLES, poisson_depth: int = 10,
                       station: Optional[Dict[str, Any]] = None, up: Sequence[float] = (0, 0, 1),
                       plane_radius_m: float = DEFAULT_PLANE_RADIUS_M, ground_points: Optional[np.ndarray] = None,
                       horizontal_plane: bool = False, plane_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """The whole of step 2 for one station: the surface (Delaunay through ``colmap_exe`` over ``dense_dir`` when
    given, else Poisson over the ``points`` within ``radius_m`` + 5 m of the axis), the cleanup (the mesh to
    ``radius_m`` from the axis), the decimation, the ground plane (:func:`fit_ground_plane` over ``points`` and
    ``ground_points``, the block's tie points) out to ``plane_radius_m``, then the sky at infinity - or a finite far
    sphere when ``far_radius_m`` is given (the 0.96.0 prototype).  ``mesh.ply``, ``ground_plane.json`` (and
    ``far_sphere.ply``) and ``mesh_report.json`` in ``out_dir``.  ``plane_points``: the terrain cloud for the plane
    when ``points`` hold more than terrain (``mppp mesh --rover``: the rover-masked cloud, not the rover's deck).
    Returns the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    points = np.asarray(points, float)
    all_points = points
    h = horizontal_distance(points, pivot, up)
    near = h <= radius_m + 5.0
    points = points[near]
    if normals is not None:
        normals = np.asarray(normals)[near]
    if dense_dir is not None and colmap_exe:
        try:
            V, F, how = delaunay_mesh(dense_dir, out / "mesh_raw.ply", colmap_exe)
        except Exception as e:                                                      # noqa: BLE001
            log.warning("mesh: Delaunay failed (%s) - Poisson instead", e)
            V, F, how = poisson_arrays(points, normals, depth=poisson_depth)
            how["delaunay_error"] = str(e)[:300]
    else:
        V, F, how = poisson_arrays(points, normals, depth=poisson_depth)
    V, F, clean = clean_arrays(V, F, pivot, points=points, radius_m=radius_m, up=up)
    V, F, dec = decimate_and_smooth(V, F, target_triangles=target_triangles)
    write_mesh(out / "mesh.ply", V, F)
    base = all_points if plane_points is None else np.asarray(plane_points, float)
    gp_in = base if ground_points is None or not len(ground_points) else np.vstack([base, np.asarray(ground_points, float)])
    plane = fit_ground_plane(gp_in, pivot, up, radius_m=plane_radius_m, horizontal=horizontal_plane)
    plane["sources"] = {"fused_points": int(len(base)), "tie_points": int(0 if ground_points is None else len(ground_points))}
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
           "far_sphere": sphere, "coverage_by_range": coverage_by_range(V, F, pivot), "tau_m": {str(z): float(tau_m(z))
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
        return Scene(mesh, sphere, plane=plane, sky=sphere is None)
    return Scene(mesh, sphere)


def mesh_preview(V: np.ndarray, F: np.ndarray, pivot: Sequence[float], out_png: PathLike, sphere: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 radius_m: float = DEFAULT_RADIUS_M, scene=None, up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0)) -> Path:
    """``mesh_preview.png``: a top-down view of the vertices coloured by range, and the range panorama from the pivot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .raycast import Scene, panorama_range
    P = np.asarray(pivot, float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    if len(V):
        r = np.linalg.norm(V - P, axis=1)
        idx = np.random.default_rng(0).choice(len(V), size=min(len(V), 200_000), replace=False)
        sc = ax.scatter(V[idx, 0] - P[0], V[idx, 1] - P[1], c=r[idx], s=0.3, cmap="turbo", vmin=0, vmax=radius_m)
        plt.colorbar(sc, ax=ax, label="range from the pivot (m)")
    ax.plot(0, 0, "k+", ms=12)
    ax.set_aspect("equal")
    ax.set_xlabel("x - pivot (m)")
    ax.set_ylabel("y - pivot (m)")
    ax.set_title(f"mesh: {len(F)} triangles, top-down")
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
