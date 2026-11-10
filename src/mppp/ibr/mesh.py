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
4. radius: terrain beyond ``radius_m`` of the pivot is dropped (single-pair depth error is metres there; the far
   field takes over);
5. quadric decimation to ``target_triangles`` and five Taubin smoothing iterations (no shrinkage).

Small holes are not filled in this prototype: a hole is a black "no data" pixel downstream, which is the honest
answer.  The **far-field sphere** (``far_radius_m``, default 100 m, ~1 degree facets) is a separate geometry: ray
hits on it are labelled "far".  Beyond ~85 m the 64 mm stereo disparity of an 8K equirect is under a pixel, so the
sphere shows no stereo error by construction.

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
DEFAULT_RADIUS_M = 60.0
DEFAULT_FAR_RADIUS_M = 100.0
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


def clean_arrays(V: np.ndarray, F: np.ndarray, pivot: Sequence[float], points: Optional[np.ndarray] = None,
                 radius_m: float = DEFAULT_RADIUS_M, component_fraction: float = COMPONENT_FRACTION,
                 grazing: float = GRAZING) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Cleanup steps 1-4 on arrays: components, rubber sheets, support (when ``points`` are given), radius.
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
        keep = np.linalg.norm(cen - P, axis=1) <= radius_m
        rep["steps"].append({"step": "radius", "removed": int((~keep).sum()), "radius_m": float(radius_m)})
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
def build_station_mesh(points: np.ndarray, pivot: Sequence[float], out_dir: PathLike, normals: Optional[np.ndarray] = None,
                       dense_dir: Optional[PathLike] = None, colmap_exe: Optional[str] = None,
                       radius_m: float = DEFAULT_RADIUS_M, far_radius_m: float = DEFAULT_FAR_RADIUS_M,
                       target_triangles: int = DEFAULT_TARGET_TRIANGLES, poisson_depth: int = 10,
                       station: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole of step 2 for one station: the surface (Delaunay through ``colmap_exe`` over ``dense_dir`` when
    given, else Poisson over ``points``), the cleanup, the decimation, the far sphere; ``mesh.ply``,
    ``far_sphere.ply`` and ``mesh_report.json`` in ``out_dir``.  Returns the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if dense_dir is not None and colmap_exe:
        try:
            V, F, how = delaunay_mesh(dense_dir, out / "mesh_raw.ply", colmap_exe)
        except Exception as e:                                                      # noqa: BLE001
            log.warning("mesh: Delaunay failed (%s) - Poisson instead", e)
            V, F, how = poisson_arrays(points, normals, depth=poisson_depth)
            how["delaunay_error"] = str(e)[:300]
    else:
        V, F, how = poisson_arrays(points, normals, depth=poisson_depth)
    V, F, clean = clean_arrays(V, F, pivot, points=points, radius_m=radius_m)
    V, F, dec = decimate_and_smooth(V, F, target_triangles=target_triangles)
    Vs, Fs = sphere_arrays(pivot, far_radius_m)
    write_mesh(out / "mesh.ply", V, F)
    write_mesh(out / "far_sphere.ply", Vs, Fs)
    rep = {"station": station, "pivot_block_m": [float(v) for v in np.asarray(pivot, float)], "surface": how,
           "points": int(len(points)), "cleanup": clean, "decimation": dec, "far_sphere": {"radius_m": far_radius_m,
           "triangles": int(len(Fs))}, "coverage_by_range": coverage_by_range(V, F, pivot), "tau_m": {str(z): float(tau_m(z))
           for z in (2, 5, 10, 20, 30, 50, 85)}, "files": {"mesh": str(out / "mesh.ply"), "far_sphere": str(out / "far_sphere.ply")},
           "seconds": round(time.time() - t0, 1)}
    (out / "mesh_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    log.info("mesh: %d -> %d triangles (%s), %d on the far sphere -> %s", clean["triangles_in"], len(F), how["method"],
             len(Fs), out)
    return rep


def mesh_preview(V: np.ndarray, F: np.ndarray, pivot: Sequence[float], out_png: PathLike, sphere: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 radius_m: float = DEFAULT_RADIUS_M) -> Path:
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
    scene = Scene((V, F), sphere)
    pan = panorama_range(scene, P, width=1440)
    im = axes[1].imshow(np.log10(np.clip(pan["range"], 0.5, 200)), cmap="turbo", aspect="auto", vmin=np.log10(0.5), vmax=np.log10(200))
    axes[1].set_title("range from the pivot (log10 m), equirect: mesh and far sphere; black = no data")
    axes[1].set_xticks([0, 360, 720, 1080, 1439])
    axes[1].set_xticklabels(["N", "E", "S", "W", "N"])
    axes[1].set_yticks([])
    plt.colorbar(im, ax=axes[1], label="log10 range (m)")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return Path(out_png)
