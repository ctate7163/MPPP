"""
mppp.ibr.cli - ``mppp mesh``, ``mppp depthmaps``, ``mppp ods`` on one finished block.

    mppp mesh WORK --station 3_1234 [--points dense/fused.ply] [--colmap colmap.exe] [--mesh-radius 60] [--far-radius 100]
    mppp depthmaps WORK --station 3_1234 [--cameras N] [--scale 0.5] [--photometry]
    mppp ods WORK --station 3_1234 [--width 8192] [--ipd 0.064] [--cameras N]

``WORK`` is the block folder (or its ``colmap`` project folder).  The station's products go to
``<project>/ods/<station>/``: ``station.json``, ``mesh.ply``, ``far_sphere.ply``, ``mesh_report.json``,
``mesh_preview.png``, ``maps/``, ``maps_png/``, ``maps_report.{json,md}``, ``photometry/``, ``photometry_maps/``,
``photometry_png/``, ``photometry_report.json``, ``ods_left.png``, ``ods_right.png``, ``provenance.png``, ``ods.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("mppp.ibr")


def _project(work: str):
    from ..sfm.project import SfmProject
    w = Path(work)
    d = w if (w / "project.json").is_file() else w / "colmap"
    if not (d / "project.json").is_file():
        raise FileNotFoundError(f"no project at {w} (project.json): align the block first")
    return SfmProject.load(d)


def _model(project):
    import pycolmap
    from ..sfm.dense import native_model_dir
    rec = pycolmap.Reconstruction()
    rec.read(str(native_model_dir(project)))
    return rec


def _station(project, rec, a):
    from .station import select_station
    st = select_station(project, rec, a.station, families=getattr(a, "cameras", "N") or "N")
    out = Path(project.root) / "ods" / st.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "station.json").write_text(json.dumps(st.to_json(), indent=1, default=float), encoding="utf-8")
    fit = st.pivot_fit
    log.info("station %s: %d frames, pivot %s (rms %.1f mm%s)", st.label, len(st.images),
             np.round(st.pivot, 3).tolist(), 1e3 * float(fit.get("rms_m", 0)), f"; {fit['fallback']}" if fit.get("fallback") else "")
    return st, out


def _scene(out: Path):
    from .mesh import read_mesh
    from .raycast import Scene
    if not (out / "mesh.ply").is_file():
        raise FileNotFoundError(f"{out / 'mesh.ply'}: run mppp mesh first")
    sphere = read_mesh(out / "far_sphere.ply") if (out / "far_sphere.ply").is_file() else None
    return Scene(read_mesh(out / "mesh.ply"), sphere)


def add_parsers(sub) -> None:
    p = sub.add_parser("mesh", help="0.96: a clean mesh and a far-field sphere for one station of a finished block (mppp.ibr.mesh)")
    p.add_argument("work")
    p.add_argument("--station", required=True, help="S003D1234 / 3_1234 / sol:91")
    p.add_argument("--points", default=None, help="the fused dense cloud (default <project>/dense*/fused.ply, the newest)")
    p.add_argument("--colmap", default=None, help="the COLMAP executable: Delaunay over --dense-dir instead of Poisson")
    p.add_argument("--dense-dir", default=None, help="the dense workspace of the Delaunay mesher (default: --points' folder)")
    p.add_argument("--mesh-radius", type=float, default=60.0)
    p.add_argument("--far-radius", type=float, default=100.0)
    p.add_argument("--target-triangles", type=int, default=3_000_000)
    p.add_argument("--poisson-depth", type=int, default=10)
    p.add_argument("--cameras", default="N", help="the station's frames for the pivot (default N)")
    p.add_argument("--no-preview", action="store_true")
    p = sub.add_parser("depthmaps", help="0.96: per-image depth / range / normal / label maps cast from the station's mesh (mppp.ibr.raycast)")
    p.add_argument("work")
    p.add_argument("--station", required=True)
    p.add_argument("--cameras", default="N", help="camera family letters, e.g. N or NZ")
    p.add_argument("--scale", type=float, default=0.5, help="Navcam maps at this fraction of the native size")
    p.add_argument("--zcam-scale", type=float, default=1.0)
    p.add_argument("--images", nargs="*", default=None, help="only these image names")
    p.add_argument("--photometry", action="store_true", help="then the photometric cube (incidence, emission, phase, shadow)")
    p.add_argument("--no-shadow", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-validate", action="store_true")
    p = sub.add_parser("ods", help="0.96: the stereoscopic panorama of the station - two equirect eye images (mppp.ibr.ods)")
    p.add_argument("work")
    p.add_argument("--station", required=True)
    p.add_argument("--width", type=int, default=8192)
    p.add_argument("--ipd", type=float, default=0.064, help="eye separation in metres")
    p.add_argument("--cameras", default="N")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--zcam-scale", type=float, default=1.0)
    p.add_argument("--images", nargs="*", default=None)
    p.add_argument("--rows-per-chunk", type=int, default=128)
    p.add_argument("--eyes", nargs="+", default=["left", "right"], choices=["left", "right", "centre"])


def _points_file(project, given: Optional[str], label: Optional[str] = None) -> Path:
    """``--points``, else the station's own dense workspace (``dense_<station>*/fused.ply``, ``mppp dense --station``),
    else the newest ``dense*/fused.ply``."""
    if given:
        return Path(given)
    own = sorted(Path(project.root).glob(f"dense_{label}*/fused.ply"), key=lambda p: p.stat().st_mtime) if label else []
    if own:
        return own[-1]
    cands = sorted(Path(project.root).glob("dense*/fused.ply"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"{project.root}: no dense*/fused.ply - run mppp dense (prepare, stereo, fusion) first, or give --points")
    return cands[-1]


def cmd_mesh(a: argparse.Namespace) -> int:
    from .mesh import build_station_mesh, mesh_preview, read_mesh, read_points
    project = _project(a.work)
    rec = _model(project)
    st, out = _station(project, rec, a)
    pts_file = _points_file(project, a.points, st.label)
    pts = read_points(pts_file)
    log.info("mesh: %d points from %s", len(pts), pts_file)
    rep = build_station_mesh(pts, st.pivot, out, dense_dir=a.dense_dir or pts_file.parent, colmap_exe=a.colmap,
                             radius_m=a.mesh_radius, far_radius_m=a.far_radius, target_triangles=a.target_triangles,
                             poisson_depth=a.poisson_depth, station=st.to_json())
    rep["points_file"] = str(pts_file)
    (out / "mesh_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    if not a.no_preview:
        V, F = read_mesh(out / "mesh.ply")
        mesh_preview(V, F, st.pivot, out / "mesh_preview.png", sphere=read_mesh(out / "far_sphere.ply"), radius_m=a.mesh_radius)
    print(f"{st.label}: {rep['cleanup']['triangles_in']} -> {rep['decimation'].get('triangles_after', rep['cleanup']['triangles_out'])} "
          f"triangles ({rep['surface']['method']}), far sphere {a.far_radius:g} m -> {out}")
    return 0


def cmd_depthmaps(a: argparse.Namespace) -> int:
    from .raycast import cast_station
    project = _project(a.work)
    rec = _model(project)
    st, out = _station(project, rec, a)
    scene = _scene(out)
    rep = cast_station(project, rec, st, scene, out, families=a.cameras, scale=a.scale, zcam_scale=a.zcam_scale,
                       images=a.images, previews=not a.no_preview, validate=not a.no_validate)
    print(f"{st.label}: {rep['n_images']} images -> {out / 'maps'}; {out / 'maps_report.md'}")
    if a.photometry:
        from .photometry import photometric_cube
        prep = photometric_cube(project, rec, st, scene, out / "maps", out, previews=not a.no_preview, with_shadow=not a.no_shadow)
        print(f"photometric cube: {prep['n_images']} images -> {out / 'photometry'}")
    return 0


def cmd_ods(a: argparse.Namespace) -> int:
    from .ods import render_station
    project = _project(a.work)
    rec = _model(project)
    st, out = _station(project, rec, a)
    scene = _scene(out)
    rep = render_station(project, rec, st, scene, out, width=a.width, ipd_m=a.ipd, families=a.cameras, scale=a.scale,
                         zcam_scale=a.zcam_scale, images=a.images, rows_per_chunk=a.rows_per_chunk, eyes=a.eyes)
    for eye, e in rep["eyes"].items():
        c = e["coverage"]
        print(f"{eye}: {e['file']}  coloured {100 * c['coloured']:.1f}%, mesh {100 * c['mesh']:.1f}%, far {100 * c['far']:.1f}%, "
              f"no surface {100 * c['no_surface']:.1f}%, {e['sources_used']} sources")
    return 0


COMMANDS = {"mesh": cmd_mesh, "depthmaps": cmd_depthmaps, "ods": cmd_ods}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mppp.ibr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_parsers(sub)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return COMMANDS[a.cmd](a)
