"""
mppp.ibr.cli - ``mppp mesh``, ``mppp depthmaps``, ``mppp ods`` on one finished block.

    mppp mesh WORK --station 3_1234 [--points dense/fused.ply] [--colmap colmap.exe] [--mesh-radius 30] [--plane-radius 100]
    mppp depthmaps WORK --station 3_1234 [--cameras N] [--scale 0.5] [--photometry]
    mppp ods WORK --station 3_1234 [--width 8192] [--ipd 0.064] [--cameras N]

``WORK`` is the block folder (or its ``colmap`` project folder).  The station's products go to
``<project>/ods/<station>/``: ``station.json``, ``mesh.ply``, ``ground_plane.json``, ``mesh_report.json``,
``mesh_preview.png``, ``maps/``, ``maps_png/``, ``maps_report.{json,md}``, ``photometry/``, ``photometry_maps/``,
``photometry_png/``, ``photometry_report.json``, ``<station>_ods_left.png``, ``<station>_ods_right.png``,
``<station>_ods_stereo_square.png/.jpg`` (``<station>_rover_...`` with ``--rover``), ``provenance.png``, ``ods.json``
(``rover_deck.json`` with ``mesh --rover``).
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
    _rover(a)
    st = select_station(project, rec, a.station, families=getattr(a, "cameras", "N") or "N")
    if getattr(a, "out", None):                      # 10 Nov: e.g. D:\rover_blocks\single_sites\S003D2046[\with_rover]
        out = Path(a.out)
    else:
        out = Path(project.root) / "ods" / (st.label + ("_rover" if getattr(a, "rover", False) else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "station.json").write_text(json.dumps(st.to_json(), indent=1, default=float), encoding="utf-8")
    fit = st.pivot_fit
    log.info("station %s: %d frames, pivot %s (rms %.1f mm%s)", st.label, len(st.images),
             np.round(st.pivot, 3).tolist(), 1e3 * float(fit.get("rms_m", 0)), f"; {fit['fallback']}" if fit.get("fallback") else "")
    return st, out


def _scene(out: Path):
    from .mesh import load_scene
    return load_scene(out)


def _frames(project, st, a) -> Optional[list]:
    """The frames for depthmaps / ods: ``--images``; else the frames the station's dense workspace was made from
    (``dense_<station>*/mppp_dense.json``: 10 Nov, the Van Zyl run - the panorama took all 16 frames of the station,
    sols 52, 54 and 64, and showed the arm where it was on the other sols); else every frame of the station.  Then
    ``--sols``."""
    names = list(a.images) if getattr(a, "images", None) else None
    if names is None:
        recs = sorted(Path(project.root).glob(f"dense_{st.label}*/mppp_dense.json"), key=lambda f: f.stat().st_mtime)
        if recs:
            sub = json.loads(recs[-1].read_text(encoding="utf-8")).get("subset") or []
            have = set(st.images)
            names = [n for n in sub if n in have] or None
            if names:
                log.info("frames: the %d of the dense workspace %s (--images to choose others)", len(names), recs[-1].parent.name)
    excl = getattr(a, "exclude", None)
    if excl:
        keys = {_pair_key(e) for e in excl}
        base = names if names is not None else list(st.images)
        names = [n for n in base if _pair_key(n) not in keys]
        log.info("frames: %d left out (--exclude, both eyes of a pair): %s", len(base) - len(names),
                 ", ".join(sorted(set(base) - set(names))) or "none of them were in")
    sols = getattr(a, "sols", None)
    if sols:
        base = names if names is not None else list(st.images)
        names = [n for n in base if n[4:8].isdigit() and int(n[4:8]) in set(int(x) for x in sols)]
        log.info("frames: %d on sols %s", len(names), sols)
    return names


def _pair_key(name: str) -> str:
    """A frame's name without the eye and the extension: NLF_... and NRF_... share it (a stereo pair)."""
    n = Path(str(name).strip()).name
    if n.lower().endswith(".png"):
        n = n[:-4]
    return (n[0] + n[2:]).upper() if len(n) > 2 and n[1].upper() in "LR" else n.upper()


def _tie_points(rec, min_track: int = 3) -> np.ndarray:
    """The block's tie points seen by ``min_track`` images or more (for the ground plane beyond the mesh)."""
    pts = [p.xyz for p in rec.points3D.values() if p.track.length() >= min_track]
    return np.asarray(pts, float).reshape(-1, 3)


def _rover_args(p) -> None:
    p.add_argument("--rover", action="store_true", help="the rover body and deck shown - the default since 0.96.1 "
                   "(kept for old command lines)")
    p.add_argument("--mask-rover", action="store_true", help="the rover masked out: the rover-masked cloud and the rover "
                   "masks, products in ods/<station>/ (default since 0.96.1, Christian 10 Nov: the rover and its deck "
                   "shown - the cloud fused without the masks, mppp dense --fusion --keep-rover - products in "
                   "ods/<station>_rover/)")


def _rover(a) -> bool:
    """0.96.1: the rover shown unless ``--mask-rover``."""
    a.rover = not getattr(a, "mask_rover", False)
    return a.rover


def add_parsers(sub) -> None:
    from .single_sites import add_parser as _single_sites_parser
    _single_sites_parser(sub)
    p = sub.add_parser("mesh", help="0.96: the surface of one station of a finished block: a clean mesh to 30 m from the mast "
                       "axis, the ground plane to 100 m, the sky at infinity (mppp.ibr.mesh)")
    p.add_argument("work")
    p.add_argument("--station", required=True, help="S003D1234 / 3_1234 / sol:91")
    p.add_argument("--points", default=None, help="the fused dense cloud (default <project>/dense*/fused.ply, the newest)")
    p.add_argument("--colmap", default=None, help="the COLMAP executable: Delaunay over --dense-dir instead of Poisson")
    p.add_argument("--dense-dir", default=None, help="the dense workspace of the Delaunay mesher (default: --points' folder)")
    p.add_argument("--mesh-radius", type=float, default=30.0, help="the mesh to this horizontal distance from the mast "
                   "axis (m; default 30)")
    p.add_argument("--plane-radius", type=float, default=100.0, help="the ground plane beyond the mesh, to this horizontal "
                   "distance (m; default 100); the sky at infinity beyond")
    p.add_argument("--horizontal-plane", action="store_true", help="a horizontal ground plane at the median height "
                   "(default: fitted to the terrain's regional slope, up to 20 deg)")
    p.add_argument("--no-tie-points", action="store_true", help="fit the plane to the fused points only (default: and the "
                   "block's tie points to 100 m, which reach farther)")
    p.add_argument("--far-radius", type=float, default=None, help="a finite far sphere at this radius instead of the sky "
                   "at infinity (the 0.96.0 prototype)")
    p.add_argument("--target-triangles", type=int, default=6_000_000, help="decimate to at most this many (default 6 M)")
    p.add_argument("--poisson-depth", type=int, default=10)
    p.add_argument("--cameras", default="N", help="the station's frames for the pivot (default N)")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--out", default=None, help="0.96: the folder for this station's products (default <project>/ods/<station>[_rover]), e.g. "
                   "D:\\rover_blocks\\single_sites\\S003D2046; use the same --out for mesh, depthmaps and ods")
    _rover_args(p)
    p.add_argument("--keep-floating", action="store_true", help="keep surface floating above the terrain (default, "
                   "0.96.1: removed - anything above the cameras, and beyond 3 m of the mast axis anything less than "
                   "1 m below them; heights above the ground plane)")
    p.add_argument("--no-rover-model", action="store_true", help="rover shown: no Perseverance model in the mesh "
                   "(default, 0.96.1: NASA's model, simplified, the arm stowed, placed by the rover pose - the fused "
                   "points within 4 cm of it dropped; mppp.ibr.rover_model)")
    p.add_argument("--rover-model-parts", nargs="+", default=None, choices=["body", "arm", "arm_base", "suspension", "wheels"],
                   help="only these parts of the model (default all; e.g. without suspension wheels on rough ground)")
    p.add_argument("--no-deck-floor", action="store_true", help="--rover: no floor at the rover deck (default: nothing "
                   "below the deck top inside its footprint, RMECH x -1.10..+0.95 m, y +-0.60 m, placed by the frames' "
                   "label CAHVORE poses; the floor added to the mesh; rover_deck.json)")
    p = sub.add_parser("depthmaps", help="0.96: per-image depth / range / normal / label maps cast from the station's mesh (mppp.ibr.raycast)")
    p.add_argument("work")
    p.add_argument("--station", required=True)
    p.add_argument("--cameras", default="N", help="camera family letters, e.g. N or NZ")
    p.add_argument("--scale", type=float, default=1.0, help="Navcam maps at this fraction of the frame's own size "
                   "(default 1: its own pixel grid; 0.5 before 10 Nov)")
    p.add_argument("--zcam-scale", type=float, default=1.0)
    p.add_argument("--images", nargs="*", default=None, help="only these image names (default: the frames of the "
                   "station's dense workspace, else all its frames)")
    p.add_argument("--sols", nargs="+", type=int, default=None, help="only the frames of these sols")
    p.add_argument("--photometry", action="store_true", help="then the photometric cube (incidence, emission, phase, shadow)")
    p.add_argument("--no-shadow", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--out", default=None, help="0.96: the folder for this station's products (default <project>/ods/<station>[_rover]), e.g. "
                   "D:\\rover_blocks\\single_sites\\S003D2046; use the same --out for mesh, depthmaps and ods")
    _rover_args(p)
    p = sub.add_parser("ods", help="0.96: the stereoscopic panorama of the station - two equirect eye images (mppp.ibr.ods)")
    p.add_argument("work")
    p.add_argument("--station", required=True)
    p.add_argument("--width", type=int, default=8192)
    p.add_argument("--ipd", type=float, default=0.064, help="eye separation in metres")
    p.add_argument("--cameras", default="N")
    p.add_argument("--scale", type=float, default=1.0, help="the Navcam frames sampled at this fraction of their own "
                   "size (default 1, Christian 10 Nov: the half-resolution frames are 0.68 mrad a pixel, the 8K "
                   "panorama 0.77 - at 0.5 they were 1.36, coarser than the panorama)")
    p.add_argument("--zcam-scale", type=float, default=1.0)
    p.add_argument("--images", nargs="*", default=None, help="only these image names (default: the frames of the "
                   "station's dense workspace, else all its frames)")
    p.add_argument("--sols", nargs="+", type=int, default=None, help="only the frames of these sols")
    p.add_argument("--exclude", nargs="+", default=None, help="leave these frames out (stems or names; a stereo pair "
                   "goes together: NLF_... also drops its NRF_ partner)")
    p.add_argument("--bits", type=int, default=8, choices=[8, 16], help="16 (0.96.1): the frames made again from their "
                   "PDS products as 16-bit linear images (<project>/images16/, mppp.ibr.linear16), the panorama blended "
                   "in them: <station>_ods_left_16.png, _right_16.png, _stereo_square_16.png (uint16 linear, for colour "
                   "work) next to the 8-bit ones; saturated pixels neutral. Falls back to 8 bits where a product is "
                   "not found")
    p.add_argument("--pds", default=None, help="--bits 16: the PDS archive (default MPPP_PDS, else the one the block's "
                   "manifest names)")
    p.add_argument("--tone", default="auto", choices=["auto", "soft", "clip"], help="--bits 16: the 8-bit panorama "
                   "from the linear one - auto (default: the black and white points set from the panorama, 0.5 and "
                   "99.7 %% of what the frames coloured, the same for both eyes, the brightest rolled off), soft "
                   "(unchanged to 190, the highlights rolled off) or clip (as the 8-bit pipeline: cut at 255)")
    p.add_argument("--exposure16", type=float, default=0.5, help="--bits 16: the 16-bit files at this factor of the "
                   "frames' own 16-bit scale (default 0.5: darker, with headroom so nothing saturates)")
    p.add_argument("--no-ground-fill", action="store_true", help="leave black what no frame coloured below the horizon "
                   "(default: filled as plain terrain from the terrain around it - under the rover, behind the arm)")
    p.add_argument("--prefix", default=None, help="the panoramas' file-name prefix (default the station, S003D2046_ / "
                   "S003D2046_rover_: S003D2046_ods_left.png ...; '' for the bare ods_left.png)")
    p.add_argument("--rows-per-chunk", type=int, default=128)
    p.add_argument("--eyes", nargs="+", default=["left", "right"], choices=["left", "right", "centre"])
    p.add_argument("--no-gains", action="store_true", help="no gain compensation between the source frames")
    p.add_argument("--stack-top", default="left", choices=["left", "right"], help="the eye on top in "
                   "<station>_ods_stereo_square.png / .jpg (default left: the left eye above the right; right before 10 Nov)")
    p.add_argument("--keep-head-hardware", action="store_true", help="--rover: keep the hardware fixed to the mast head "
                   "(the wind-sensor booms; default: masked, found as what the rover mask covers at the same pixels in the "
                   "frames of one eye and elevation)")
    p.add_argument("--no-sky-fill", action="store_true", help="leave the sky above the frames black (default: the seen "
                   "sky carried up to the zenith, the frames' tops blended away)")
    p.add_argument("--azimuth0", type=float, default=0.0, help="the azimuth (deg, from north through east) at the left "
                   "edge of the panorama: 0 = north in the first column (default)")
    p.add_argument("--north-center", action="store_true", help="north in the middle of the panorama (--azimuth0 -180), "
                   "for viewers that face the image centre")
    p.add_argument("--stack-only", action="store_true", help="only stack the eye images already in the folder into "
                   "ods_stereo_square.png / .jpg (no rendering)")
    p.add_argument("--out", default=None, help="0.96: the folder for this station's products (default <project>/ods/<station>[_rover]), e.g. "
                   "D:\\rover_blocks\\single_sites\\S003D2046; use the same --out for mesh, depthmaps and ods")
    _rover_args(p)


def _points_file(project, given: Optional[str], label: Optional[str] = None, rover: bool = False) -> Path:
    """``--points``, else the station's own dense workspace (``dense_<station>*/fused.ply``, ``mppp dense --station``;
    ``fused_rover.ply`` with ``rover``), else the newest ``dense*/`` cloud of that name."""
    if given:
        return Path(given)
    from ..sfm.dense import FUSED_ROVER_PLY
    fname = FUSED_ROVER_PLY if rover else "fused.ply"
    own = sorted(Path(project.root).glob(f"dense_{label}*/{fname}"), key=lambda p: p.stat().st_mtime) if label else []
    if own:
        return own[-1]
    cands = sorted(Path(project.root).glob(f"dense*/{fname}"), key=lambda p: p.stat().st_mtime)
    if not cands:
        hint = " --keep-rover" if rover else ""
        more = " (or --mask-rover for the rover-masked products)" if rover else ""
        raise FileNotFoundError(f"{project.root}: no dense*/{fname} - run mppp dense --station ... --fusion{hint} first, or give "
                                f"--points{more}")
    return cands[-1]


def cmd_mesh(a: argparse.Namespace) -> int:
    from .mesh import ROVER_MODEL_PARTS, build_station_mesh, mesh_preview, read_mesh, read_points, read_points_normals
    project = _project(a.work)
    rec = _model(project)
    st, out = _station(project, rec, a)
    pts_file = _points_file(project, a.points, st.label, rover=a.rover)
    pts, nrm = read_points_normals(pts_file)
    log.info("mesh: %d points from %s (%s)", len(pts), pts_file, "with normals" if nrm is not None else "no normals")
    terrain = None
    if a.rover:                                   # the plane from the terrain alone, not the deck
        try:
            terrain = read_points(_points_file(project, None, st.label))
        except FileNotFoundError:
            log.warning("mesh --rover: no rover-masked fused.ply; the plane is fitted to the cloud with the rover in it")
    deck = None
    if a.rover and not getattr(a, "no_deck_floor", False):     # 10 Nov: nothing below the rover deck
        from .rover_deck import rover_pose, write_deck
        try:
            deck = rover_pose(project, rec, st.images)
            d = write_deck(out, deck, st.up)
            log.info("rover deck: %d frames agree to %.2f deg / %.1f mm; rover tilt %.1f deg; floor %.2f x %.2f m -> %s",
                     d["frames"], d["rot_spread_deg"], d["pos_spread_mm"], d.get("rover_tilt_deg", float("nan")),
                     d["size_m"][0], d["size_m"][1], out / "rover_deck.json")
        except Exception as e:                                   # noqa: BLE001 - the mesh without the floor
            log.warning("mesh --rover: no rover pose for the deck floor (%s); meshing without it", e)
            deck = None
    ties = None if a.no_tie_points else _tie_points(rec)
    rep = build_station_mesh(pts, st.pivot, out, normals=nrm, dense_dir=a.dense_dir or pts_file.parent, colmap_exe=a.colmap,
                             radius_m=a.mesh_radius, far_radius_m=a.far_radius, target_triangles=a.target_triangles,
                             poisson_depth=a.poisson_depth, station=st.to_json(), up=st.up, plane_radius_m=a.plane_radius,
                             ground_points=ties, horizontal_plane=a.horizontal_plane, plane_points=terrain,
                             deck=deck, remove_floating=not a.keep_floating,
                             rover_model_parts=None if a.no_rover_model else (a.rover_model_parts or ROVER_MODEL_PARTS))
    rep["rover"] = "shown" if a.rover else "masked out"
    rep["points_file"] = str(pts_file)
    (out / "mesh_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    if not a.no_preview:
        V, F = read_mesh(out / "mesh.ply")
        mesh_preview(V, F, st.pivot, out / "mesh_preview.png", radius_m=a.mesh_radius, scene=_scene(out),
                     up=st.up, north=st.north, east=st.east)
    pl = rep["scene"]["plane"]
    reach = ", ".join(f"{r['ring_m'][0]:g}-{r['ring_m'][1]:g} m {r['points']}" for r in rep["points_by_ring"] if r["points"])
    print(f"{st.label}: {rep['cleanup']['triangles_in']} -> {rep['decimation'].get('triangles_after', rep['cleanup']['triangles_out'])} "
          f"triangles ({rep['surface']['method']}) within {a.mesh_radius:g} m of the mast axis; ground plane {pl['method']}, "
          f"tilt {pl['tilt_deg']:.1f} deg, {pl['height_below_pivot_m']:.2f} m below the pivot, rms {pl.get('rms_m', float('nan')):.2f} m "
          f"over {pl['bins']} bins, to {a.plane_radius:g} m; then {rep['scene']['beyond']} -> {out}")
    print(f"  fused points by distance from the axis: {reach}")
    return 0


def cmd_depthmaps(a: argparse.Namespace) -> int:
    from .raycast import cast_station
    project = _project(a.work)
    rec = _model(project)
    st, out = _station(project, rec, a)
    scene = _scene(out)
    rep = cast_station(project, rec, st, scene, out, families=a.cameras, scale=a.scale, zcam_scale=a.zcam_scale,
                       images=_frames(project, st, a), previews=not a.no_preview, validate=not a.no_validate, use_masks=not a.rover)
    print(f"{st.label}: {rep['n_images']} images -> {out / 'maps'}; {out / 'maps_report.md'}")
    if a.photometry:
        from .photometry import photometric_cube
        prep = photometric_cube(project, rec, st, scene, out / "maps", out, previews=not a.no_preview, with_shadow=not a.no_shadow)
        print(f"photometric cube: {prep['n_images']} images -> {out / 'photometry'}")
    return 0


def cmd_ods(a: argparse.Namespace) -> int:
    from .ods import render_station, stack_eyes
    if a.stack_only:
        out = Path(a.out) if a.out else None
        sq = a.prefix
        if out is None:
            project = _project(a.work)
            rec = _model(project)
            st, out = _station(project, rec, a)
            if sq is None:
                from .ods import ods_prefix
                sq = ods_prefix(st.label, rover=a.rover)
        r = stack_eyes(out, top=a.stack_top, square_prefix=sq)
        print(f"{r['jpg']}  ({r['size'][0]} x {r['size'][1]}, {r['top']} eye on top; also .png)")
        return 0
    project = _project(a.work)
    rec = _model(project)
    st, out = _station(project, rec, a)
    scene = _scene(out)
    rep = render_station(project, rec, st, scene, out, width=a.width, ipd_m=a.ipd, families=a.cameras, scale=a.scale,
                         zcam_scale=a.zcam_scale, images=_frames(project, st, a), rows_per_chunk=a.rows_per_chunk, eyes=a.eyes,
                         use_masks=not a.rover, gains=not a.no_gains, stack_top=a.stack_top,
                         sky_fill=not a.no_sky_fill, mask_head_hardware=not a.keep_head_hardware,
                         azimuth0_deg=-180.0 if a.north_center else a.azimuth0, prefix=a.prefix,
                         bits=a.bits, pds=a.pds, tone=a.tone, exposure16=a.exposure16,
                         ground_fill=not a.no_ground_fill)
    for eye, e in rep["eyes"].items():
        c = e["coverage"]
        print(f"{eye}: {e['file']}  coloured {100 * c['coloured']:.1f}%, mesh {100 * c['mesh']:.1f}%, "
              f"plane {100 * c.get('plane', 0):.1f}%, far {100 * c['far']:.1f}%, "
              f"no surface {100 * c['no_surface']:.1f}%, {e['sources_used']} sources")
    if rep.get("bits") == 16:
        print(f"16-bit linear: {rep['stacked_16bit']['png'] if rep.get('stacked_16bit') else rep['linear16']['dir']}; "
              f"the 8-bit view with the '{rep['linear16']['view_tone']}' tone curve")
    elif rep.get("linear16"):
        print(f"--bits 16 NOT possible: {len(rep['linear16']['missing'])} frames without a 16-bit image (see the log) - rendered in 8 bits")
    if rep.get("stacked"):
        print(f"stacked: {rep['stacked']['jpg']} ({rep['stacked']['size'][0]} x {rep['stacked']['size'][1]}, "
              f"{rep['stacked']['top']} eye on top; also .png)")
    if rep.get("gains"):
        g = list(rep["gains"]["gains"].values())
        print(f"gains: {min(g):.3f} - {max(g):.3f} over {rep['gains']['pairs']} source pairs")
    return 0


def cmd_single_sites(a: argparse.Namespace) -> int:
    from .single_sites import cmd_single_sites as run
    return run(a)


COMMANDS = {"mesh": cmd_mesh, "depthmaps": cmd_depthmaps, "ods": cmd_ods, "single-sites": cmd_single_sites}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mppp.ibr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_parsers(sub)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return COMMANDS[a.cmd](a)
