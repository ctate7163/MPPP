"""
mppp.ibr.cli - ``mppp mesh``, ``mppp depthmaps``, ``mppp ods`` on one finished block.

    mppp mesh WORK --station 3_1234 [--points dense/fused.ply] [--colmap colmap.exe] [--mesh-radius 75] [--plane-radius 200]
    mppp depthmaps WORK --station 3_1234 [--cameras N] [--scale 0.5] [--photometry]
    mppp ods WORK --station 3_1234 [--width 8192] [--ipd 0.064] [--cameras N]

``WORK`` is the block folder (or its ``colmap`` project folder).  The station's products go to
``<project>/ods/<station>/``: ``station.json``, ``mesh.ply``, ``ground_plane.json``, ``mesh_report.json``,
``mesh_preview.png``, ``maps/``, ``maps_report.{json,md}``, ``photometry/``, ``photometry_maps/``,
``photometry_jpg/`` (0.97.0: per image _rad, _depth_linear, _depth_log, _normal, _incidence, _emission, _phase,
_shadow .jpg), ``photometry_report.json``, ``<station>_ods_left.png``, ``<station>_ods_right.png``,
``<station>_360_3DTD.png/.jpg`` (``<station>_rover_...`` with ``--rover``), ``provenance.png``, ``ods.json``
(``rover_deck.json`` with ``mesh --rover``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
    st = select_station(project, rec, a.station, families=getattr(a, "cameras", "N") or "N",
                        pivot_height=getattr(a, "pivot_height", None) or "zcam")
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
    p.add_argument("--pivot-height", default="zcam", choices=["zcam", "fit"], help="0.97.0: the pivot's height on the "
                   "azimuth axis - zcam (default: the Mastcam-Z centres' height with the head level, whatever "
                   "cameras are rendered, so Navcam and Mastcam-Z products share one viewpoint) or fit (the frames' "
                   "own fit, as before); use the same for mesh, depthmaps and ods")
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
    p.add_argument("--mesh-radius", type=float, default=75.0, help="the mesh to this horizontal distance from the mast "
                   "axis (m; default 75)")
    p.add_argument("--keep-under-rover", type=float, default=None, metavar="M",
                   help="0.97.0: under the rover's footprint keep the terrain within M metres of the ground's height "
                        "around it (for a cloud that holds nearby stations' frames, mppp dense --nearby-m; default: all "
                        "of it is replaced by the interpolated ground)")
    p.add_argument("--terrain-from", default=None, metavar="DIR",
                   help="0.97.0: the folder of this station's other variant (rover shown / masked) - its terrain mesh is used "
                        "as it is, with or without the rover model (no second surface run)")
    p.add_argument("--plane-radius", type=float, default=200.0, help="the ground plane beyond the mesh, to this horizontal "
                   "distance (m; default 200); the sky sphere beyond (mppp ods --sky-radius)")
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
                   "(default, 0.96.1: NASA's model, the arm stowed, placed by the rover pose - the fused "
                   "points within 4 cm of it dropped; mppp.ibr.rover_model)")
    p.add_argument("--rover-model-parts", nargs="+", default=None, choices=["body", "arm", "arm_base", "suspension", "wheels", "antenna"],
                   help="only these parts of the model (default all; e.g. without suspension wheels on rough ground)")
    p.add_argument("--rover-cloud", action="store_true", help="--rover the 0.96 way: the mesh from the cloud fused with "
                   "the rover in it (fused_rover.ply), with the deck floor (default since 0.97.0: the terrain of the "
                   "rover-masked cloud, then the Perseverance model on it)")
    p.add_argument("--no-fill", action="store_true", help="0.97.0: leave the terrain's holes to the ground plane "
                   "(default: the terrain interpolated under the rover and over every hole out to --mesh-radius)")
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
    p.add_argument("--add-cameras", default="", metavar="LETTERS",
                   help="0.97.8: also the station's frames of these camera families taken on the same sols (F: the front "
                        "Hazcam pair, R: the rear pair) - their maps and photometric pictures beside the others")
    p.add_argument("--no-shadow", action="store_true")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--out", default=None, help="0.96: the folder for this station's products (default <project>/ods/<station>[_rover]), e.g. "
                   "D:\\rover_blocks\\single_sites\\S003D2046; use the same --out for mesh, depthmaps and ods")
    _rover_args(p)
    p = sub.add_parser("ods", help="0.96: the stereoscopic panorama of the station - two equirect eye images (mppp.ibr.ods)")
    p.add_argument("work")
    p.add_argument("--station", required=True)
    p.add_argument("--width", type=int, default=8192, help="per eye (default 8192; 16384 for a 16K panorama - the "
                   "default from 0.97.4 to 0.97.9)")
    p.add_argument("--ipd", type=float, default=0.064, help="eye separation in metres")
    p.add_argument("--sky-radius", type=float, default=None, metavar="M",
                   help="0.97.0: the sky on a sphere of this radius about the mast axis and the ground plane out to it "
                        "(default 200; the eye separation fades from 75 m to zero at 200 m)")
    p.add_argument("--pole-merge", type=float, nargs=2, default=None, metavar=("START", "END"),
                   help="0.97.0: above the horizon, the elevation (deg) where the eye separation starts to fall and where "
                        "it is zero (default 30 80); 90 90: none")
    p.add_argument("--pole-merge-down", type=float, nargs=2, default=None, metavar=("START", "END"),
                   help="0.97.1: the same below the horizon (default 60 88 since 0.97.5, with the near-surface limit "
                        "of 2.5 m); 90 90: none")
    p.add_argument("--cameras", default="N")
    p.add_argument("--fill-cameras", default="", metavar="LETTERS",
                   help="0.97.7: the station's frames of these camera families as fill-only sources - they colour what "
                        "the panorama's own frames do not see (F: the front Hazcam pair, for the terrain under the arm; "
                        "R: the rear pair); those of the same sols; default none")
    p.add_argument("--site-name", default=None, help="0.97.7: the site's name in the mark at the nadir (e.g. Rockytop)")
    p.add_argument("--no-nadir-mark", action="store_true", help="0.97.7: no disc with the credits at the nadir "
                   "(default: within 20 deg of straight down, light blue, two rows of black text ahead of the rover - Mars 2020 Perseverance Rover, the sol and site, made by "
                   "C. Tate, the image credit)")
    p.add_argument("--nadir-mark-deg", type=float, default=None, help="the disc's angular radius (default 20)")
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
                   "in them: <station>_ods_left_16.png, _right_16.png, _360_3DTD_16.png (uint16 linear, for colour "
                   "work) next to the 8-bit ones; saturated pixels neutral. Falls back to 8 bits where a product is "
                   "not found")
    p.add_argument("--pds", default=None, help="--bits 16: the PDS archive (default MPPP_PDS, else the one the block's "
                   "manifest names)")
    p.add_argument("--tone", default="auto", choices=["auto", "soft", "clip"], help="--bits 16: the 8-bit panorama "
                   "from the linear one - auto (default: the exposure set on the ground - its median to 100 of 255 - the "
                   "same for both eyes, the highlights rolled off and faded to white instead of clipped), soft "
                   "(unchanged to 190, the highlights rolled off) or clip (as the 8-bit pipeline: cut at 255)")
    p.add_argument("--exposure16", default="auto", type=lambda v: v if v == "auto" else float(v),
                   help="--bits 16: the 16-bit files at this factor of the frames' own 16-bit scale; auto (default, "
                        "0.97.0): the largest gain between 0.5 and 4 that keeps the 99.99 %% point of the brightest "
                        "channel at 95 %% of the range - one gain per station, kept in the levels file")
    p.add_argument("--keep-mast", action="store_true", help="rover shown: keep the mast and its wind-sensor booms in "
                   "the frames (default: masked - the rover model's mast projected into every frame)")
    p.add_argument("--no-ground-fill", action="store_true", help="leave black what no frame coloured below the horizon "
                   "(default: filled as plain terrain from the terrain around it - under the rover, behind the arm)")
    p.add_argument("--prefix", default=None, help="the panoramas' file-name prefix (default the station, S003D2046_ / "
                   "S003D2046_rover_: S003D2046_ods_left.png ...; '' for the bare ods_left.png)")
    p.add_argument("--rows-per-chunk", type=int, default=128)
    p.add_argument("--eyes", nargs="+", default=["left", "right"], choices=["left", "right", "centre"])
    p.add_argument("--no-gains", action="store_true", help="no gain compensation between the source frames")
    p.add_argument("--stack-top", default="left", choices=["left", "right"], help="the eye on top in "
                   "<station>_360_3DTD.png / .jpg (default left: the left eye above the right; right before 10 Nov)")
    p.add_argument("--keep-head-hardware", action="store_true", help="--rover: keep the hardware fixed to the mast head "
                   "(the wind-sensor booms; default: masked, found as what the rover mask covers at the same pixels in the "
                   "frames of one eye and elevation)")
    p.add_argument("--no-sky-fill", action="store_true", help="leave the sky above the frames black (default: the seen "
                   "sky carried up to the zenith, the frames' tops blended away)")
    p.add_argument("--azimuth0", type=float, default=0.0, help="the azimuth (deg, from north through east) at the left "
                   "edge of the panorama: 0 = north in the first column (default)")
    p.add_argument("--north-center", action="store_true", help="north in the middle of the panorama (--azimuth0 -180), "
                   "for viewers that face the image centre")
    p.add_argument("--tone-levels", default=None, metavar="FILE", help="0.97.0, --tone auto: the black point and "
                   "the ground level of the 8-bit view kept in FILE (json) - read when it is there, else found on this "
                   "panorama and written; give the rover-shown and the rover-masked run the same FILE and their "
                   "16-to-8-bit scaling is identical")
    p.add_argument("--no-depth", action="store_true", help="0.97.0: no depth products for the centre panorama "
                   "(default with --eyes centre: <prefix>_ods_centre_depth.npz, _depth_16.png, _depth_linear.jpg, "
                   "_depth_log.jpg - the range from the pivot along each pixel's ray)")
    p.add_argument("--retone", action="store_true", help="only make the 8-bit panoramas and the square again from the "
                   "16-bit ones in the folder (--tone auto; no rendering)")
    p.add_argument("--stack-only", action="store_true", help="only stack the eye images already in the folder into "
                   "360_3DTD.png / .jpg (no rendering)")
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


def _mesh_on_terrain(src: Path, out: Path, deck, st, a, parts) -> Optional[Dict[str, Any]]:
    """The rover-shown mesh from the terrain mesh in ``src`` (the same station's rover-masked ``mppp mesh``) with the
    Perseverance model on it; None when ``src`` does not hold that terrain (then the mesh is built as usual)."""
    import shutil
    from . import rover_model
    from .mesh import read_mesh, write_mesh
    need = [src / n for n in ("mesh.ply", "ground_plane.json", "mesh_report.json")]
    if not all(f.is_file() for f in need) or src.resolve() == Path(out).resolve():
        return None
    try:
        rep = json.loads(need[2].read_text(encoding="utf-8"))
    except ValueError:
        return None
    same = ((rep.get("station") or {}).get("label") == st.label
            and abs(float((rep.get("scene") or {}).get("mesh_radius_m", -1)) - float(a.mesh_radius)) < 1e-6
            and bool((rep.get("fill") or {})) == (not getattr(a, "no_fill", False)))
    if not same:
        log.info("mesh --terrain-from %s: not this station's rover-masked terrain with these settings - built again", src)
        return None
    V, F = read_mesh(need[0])
    rt = rep.get("rover_triangles")                 # the other variant's rover model: the last triangles of its mesh
    if rt and int(rt[1]) == len(F) and 0 <= int(rt[0]) < int(rt[1]):
        F = F[:int(rt[0])]
        used = np.zeros(len(V), bool)
        used[F.ravel()] = True
        V, F = V[used], (np.cumsum(used) - 1)[F]
    elif rt:
        return None
    tris = None
    deck_rep: Dict[str, Any] = {"terrain_only": True}
    if deck is not None and parts and rover_model.available():
        Vm, Fm = rover_model.model_in_block(deck, parts=parts)
        tris = [int(len(F)), int(len(F) + len(Fm))]
        F = np.vstack([F, Fm + len(V)])
        V = np.vstack([V, Vm])
        deck_rep.update(model_triangles=int(len(Fm)), model_parts=list(parts))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_mesh(out / "mesh.ply", V, F)
    shutil.copy2(need[1], out / "ground_plane.json")
    rep.update(rover_deck=deck_rep if deck is not None else None, rover_triangles=tris, terrain_from=str(src),
               files={"mesh": str(out / "mesh.ply"), "ground_plane": str(out / "ground_plane.json")})
    log.info("mesh: the terrain of %s (%d triangles) with the rover model (%d triangles) - no second surface run",
             src, tris[0] if tris else len(F), (tris[1] - tris[0]) if tris else 0)
    return rep


def _pole_merge(a):
    """The panorama's pole merging from --pole-merge / --pole-merge-down ("default": the module's)."""
    if a.pole_merge is None and getattr(a, "pole_merge_down", None) is None:
        return "default"
    from .ods import POLE_MERGE_DEG, POLE_MERGE_DOWN_DEG
    pair = lambda v, d: d if v is None else (None if v[0] >= 90 else tuple(v))      # noqa: E731
    return {"up": pair(a.pole_merge, POLE_MERGE_DEG), "down": pair(getattr(a, "pole_merge_down", None), POLE_MERGE_DOWN_DEG)}


def cmd_mesh(a: argparse.Namespace) -> int:
    from .mesh import ROVER_MODEL_PARTS, build_station_mesh, mesh_preview, read_mesh, read_points, read_points_normals
    project = _project(a.work)
    rec = _model(project)
    st, out = _station(project, rec, a)
    # 0.97.0 (Christian, 12 Nov 2026: "only rendering the terrain without the rover. then add the rover model in"):
    # the rover-shown mesh is the terrain of the rover-masked cloud with the Perseverance model set on it;
    # --rover-cloud: the 0.96 way, the mesh from the cloud fused with the rover in it
    old_way = bool(a.rover and getattr(a, "rover_cloud", False))
    pts_file = _points_file(project, a.points, st.label, rover=old_way)
    terrain = None
    if old_way:                                   # the plane from the terrain alone, not the deck
        try:
            terrain = read_points(_points_file(project, None, st.label))
        except FileNotFoundError:
            log.warning("mesh --rover: no rover-masked fused.ply; the plane is fitted to the cloud with the rover in it")
    deck, pose = None, None
    from .rover_deck import rover_pose, write_deck
    try:
        pose = rover_pose(project, rec, st.images)   # also for the masked variant: the ground under the rover
    except Exception as e:                           # noqa: BLE001 - the mesh without the rover's footprint
        log.warning("mesh: no rover pose (%s); the terrain under the rover is not replaced%s", e,
                    ", no rover model" if a.rover else "")
    if a.rover and pose is not None and not getattr(a, "no_deck_floor", False):
        deck = pose
        d = write_deck(out, deck, st.up)
        log.info("rover deck: %d frames agree to %.2f deg / %.1f mm; rover tilt %.1f deg; floor %.2f x %.2f m -> %s",
                 d["frames"], d["rot_spread_deg"], d["pos_spread_mm"], d.get("rover_tilt_deg", float("nan")),
                 d["size_m"][0], d["size_m"][1], out / "rover_deck.json")
    rep = None
    if not old_way and getattr(a, "terrain_from", None) and (deck is not None or not a.rover):
        # 0.97.0 (Christian, 12 Nov 2026: "Build the terrain mesh once"): the rover-masked variant's mesh is this
        # variant's terrain - read it and set the model on it, no second Poisson run
        rep = _mesh_on_terrain(Path(a.terrain_from), out, deck if a.rover else None, st, a,
                               None if a.no_rover_model else (a.rover_model_parts or ROVER_MODEL_PARTS))
    if rep is None:                               # 0.97.2: the cloud and the tie points are read only when a surface is built
        pts, nrm = read_points_normals(pts_file)
        log.info("mesh: %d points from %s (%s)", len(pts), pts_file, "with normals" if nrm is not None else "no normals")
        ties = None if a.no_tie_points else _tie_points(rec)
        rep = build_station_mesh(pts, st.pivot, out, normals=nrm, dense_dir=a.dense_dir or pts_file.parent, colmap_exe=a.colmap,
                                 radius_m=a.mesh_radius, far_radius_m=a.far_radius, target_triangles=a.target_triangles,
                                 poisson_depth=a.poisson_depth, station=st.to_json(), up=st.up, plane_radius_m=a.plane_radius,
                                 ground_points=ties, horizontal_plane=a.horizontal_plane, plane_points=terrain,
                                 deck=deck, remove_floating=not a.keep_floating, terrain_only=not old_way,
                                 fill=not getattr(a, "no_fill", False), rover_pose=pose,
                                     rover_keep_m=getattr(a, "keep_under_rover", None),
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
    names = _frames(project, st, a)
    if getattr(a, "add_cameras", ""):                # 0.97.8: e.g. the front Hazcam pair
        from .ods import fill_frames
        base = list(names) if names else [n for n in st.images if n[:1].upper() in set((a.cameras or "N").upper())]
        more = fill_frames(project, rec, st, a.add_cameras, base)
        log.info("depthmaps: %d more frame(s) of the camera families %r: %s", len(more), a.add_cameras, ", ".join(more) or "none at this station")
        names = base + more
    rep = cast_station(project, rec, st, scene, out, families=a.cameras, scale=a.scale, zcam_scale=a.zcam_scale,
                       images=names, previews=not a.no_preview, validate=not a.no_validate, use_masks=not a.rover)
    print(f"{st.label}: {rep['n_images']} images -> {out / 'maps'}; {out / 'maps_report.md'}")
    if a.photometry:
        from .photometry import photometric_cube
        pose = None
        if not a.no_shadow:                          # 0.97.7: the mast, its booms and the camera head cast shadows too
            from .rover_deck import rover_pose
            try:
                pose = rover_pose(project, rec, st.images)
            except Exception as e:                   # noqa: BLE001 - the shadows of the mesh alone
                log.warning("depthmaps: no rover pose (%s); the shadows without the mast and the camera head", e)
        prep = photometric_cube(project, rec, st, scene, out / "maps", out, previews=not a.no_preview, with_shadow=not a.no_shadow,
                                rover_pose=pose)
        print(f"photometric cube: {prep['n_images']} images -> {out / 'photometry'}")
    return 0


def cmd_ods(a: argparse.Namespace) -> int:
    from .ods import render_station, stack_eyes
    if a.retone:                                 # the 8-bit panoramas again from the 16-bit ones, no rendering
        from .linear16 import retone
        from .ods import find_prefix
        out = Path(a.out) if a.out else None
        if out is None:
            project = _project(a.work)
            st, out = _station(project, _model(project), a)
        r = retone(out, a.prefix if a.prefix is not None else find_prefix(out), top=a.stack_top,
                   levels_file=a.tone_levels)
        print(f"{r['square']}  (levels {r['levels_16bit_units'][0]:.0f} / ground {r['levels_16bit_units'][1]:.0f} of the 16-bit scale, "
              f"{r['top']} eye on top)")
        return 0
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
    rep = render_station(project, rec, st, scene, out, width=a.width, ipd_m=a.ipd, sky_radius_m=("default" if a.sky_radius is None else a.sky_radius), pole_merge_deg=_pole_merge(a), families=a.cameras, scale=a.scale,
                         zcam_scale=a.zcam_scale, images=_frames(project, st, a), rows_per_chunk=a.rows_per_chunk, eyes=a.eyes,
                         use_masks=not a.rover, gains=not a.no_gains, stack_top=a.stack_top,
                         sky_fill=not a.no_sky_fill, mask_head_hardware=not a.keep_head_hardware,
                         azimuth0_deg=-180.0 if a.north_center else a.azimuth0, prefix=a.prefix,
                         bits=a.bits, pds=a.pds, tone=a.tone, exposure16=a.exposure16,
                         ground_fill=not a.no_ground_fill, mask_mast=not a.keep_mast,
                         tone_levels_file=a.tone_levels, centre_depth=not a.no_depth,
                         fill_families=a.fill_cameras, nadir_mark=not a.no_nadir_mark, site_name=a.site_name,
                         nadir_mark_deg=a.nadir_mark_deg)
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
