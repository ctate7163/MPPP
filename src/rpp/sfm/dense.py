"""
Dense reconstruction (0.95.0, Christian, 28 Oct 2026): the COLMAP dense workspace of a finished block, preloaded
from the alignment so that the GUI's Reconstruction > Dense reconstruction dialog - or ``dense_reconstruct.bat``
next to ``open_in_colmap.bat`` - runs the stereo and the fusion straight away.

* :func:`prepare_dense` - ``<project>/dense/``: the images undistorted to pinhole cameras from the NATIVE-pixel
  model ``gui_native/sparse`` (the image files are at their native resolution; the full-resolution model's cameras
  would not fit them), ``sparse/`` (the pinhole model), ``stereo/patch-match.cfg`` and ``fusion.cfg`` (COLMAP's
  ``image_undistorter``, through pycolmap), and ``masks/`` - the rover masks undistorted the same way, for the
  fusion (``StereoFusionOptions.mask_path``: a pixel at 0 is not fused; the patch-match stereo itself takes no
  masks).  ``mppp_dense.json`` records what was done.
* :func:`write_dense_bat` - ``dense_reconstruct.bat``: double-click on the PC: the workspace (if not there yet),
  ``patch_match_stereo`` (needs a CUDA COLMAP; the batch file looks for one first), ``stereo_fusion`` with the
  masks into ``dense/fused.ply``, and a Poisson mesh on request.  ``set RPP_DENSE_STEP=prepare|stereo|fusion|mesh``
  runs one step, ``RPP_DENSE_MAX_SIZE`` (default 2000 px on the long side) the working resolution.
* :func:`run_dense` - the same steps from Python (a CUDA pycolmap for the stereo: the ``gpu_python`` environment).

The masks and the undistortion are per image: a block of 1000 images at 2000 px is about 1-2 GB and a few minutes
to prepare; the stereo takes hours on the GPU.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List, Sequence

import numpy as np

from .project import SfmProject
from .. import ondisk

log = logging.getLogger("rpp.sfm")
PathLike = Union[str, Path]
DENSE_DIR = "dense"
DENSE_MAX_IMAGE_SIZE = 2000        # the long side of the undistorted (working) images; the GUI's default too
# 10 Nov 2026 (Rockytop S026D0630, the first real station run): with COLMAP's defaults the fused cloud of a Navcam 360
# stopped at 7.7 m (99th percentile; not one point beyond 10 m) and the mesh covered 2-10 m only.  Patch-match keeps a
# pixel only if its triangulation angle is >= 3 deg (filter_min_triangulation_angle) and two source views agree
# (filter_min_num_consistent); the fusion wants 5 pixels (min_num_pixels).  Navcam's 0.424 m baseline makes 3 deg at
# 0.424 / tan(3 deg) = 8.1 m, and the far terrain of one station is seen by one stereo pair.  For a station (or an
# --images subset): 0.4 deg (0.424 m / tan 0.4 deg = 61 m, the mesh radius), one consistent source, two pixels; the
# mesh's range-dependent support check (tau(z)) and rubber-sheet removal clean what the looser filter lets through.
# 0.97.0 (12 Nov, Christian: "the terrain should go out further to maybe 75 m"): 0.3 deg - 0.424 m / tan 0.3 deg = 81 m.
STATION_STEREO_OPTIONS = {"min_triangulation_angle": 0.3, "filter_min_triangulation_angle": 0.3,
                          "filter_min_num_consistent": 1}
STATION_FUSION_OPTIONS = {"min_num_pixels": 2}
# 1.0 (Christian, 20 Nov 2026: "look into some way to accelerate the depth map generation and improve the fusion. The
# depth fusion takes a lot less time than making the depth maps, so we can increase the processing on that to ensure
# the mesh is higher quality").
#
# THE DEPTH MAPS, faster.  Patch-match's cost grows with the source images each reference image is matched against
# (COLMAP's image_undistorter writes __auto__, 20 into stereo/patch-match.cfg) and with the pixels of the matching
# window it samples (window_step 1: all 11 x 11).  A new workspace now takes PATCH_MATCH_SOURCES source images, the
# ones that share the most tie points with it, and samples every other pixel of the window (STATION_STEREO_FAST).
# Estimate, not measured: about 1.7 x from the sources and 2-3 x from the window, so a 303-frame block in well under
# half its time; the first new block's mppp_dense_run.json ("stereo": "seconds") tells.  A workspace whose maps were
# made with other options keeps its own options when it resumes (workspace_stereo_record), so no map is made again.
PATCH_MATCH_SOURCES = 12
COLMAP_SOURCES = 20                 # what a workspace made by COLMAP's undistorter has (and every workspace before 1.0)
STATION_STEREO_FAST = {"window_step": 2}
# THE FUSION, two passes.  The loose fusion (STATION_FUSION_OPTIONS: two pixels) keeps the far terrain that one stereo
# pair sees, and with it the noise of the near terrain.  A strict fusion (STATION_FUSION_STRICT: four agreeing pixels,
# 1 px, 8 deg) keeps only what several views agree on.  fused.ply is the strict cloud plus the loose points farther than
# MERGE_PX working pixels (at their distance from the nearest camera) from any strict point: the strict surface where
# it exists, the loose one only where it does not (merge_fusions).  The two clouds stay as fused_loose.ply and
# fused_strict.ply.
STATION_FUSION_STRICT = {"min_num_pixels": 4, "max_reproj_error": 1.0, "max_normal_error": 8.0}
MERGE_PX = 2.0
FUSED_LOOSE_PLY, FUSED_STRICT_PLY = "fused_loose.ply", "fused_strict.ply"
# 10 Nov 2026 (Christian: "try this with and without the rover body and deck shown"): the fusion without the rover
# masks goes to its own cloud, so the rover-removed products stay as they are
FUSED_ROVER_PLY = "fused_rover.ply"


def native_model_dir(project: SfmProject) -> Path:
    """``gui_native/sparse`` - written at the end of a run (``gui_native``), or now from the saved model."""
    nat = Path(project.root) / "gui_native" / "sparse"
    if not (nat / "cameras.bin").is_file():
        import pycolmap
        from .export import write_gui_native
        rs = project.settings.get("reconstruction") or {}
        model = Path(project.root) / (rs.get("path") or "sparse/cahv_ba")
        if not (model / "cameras.bin").is_file():
            raise FileNotFoundError(f"no model at {model}: align the block first")
        log.info("dense: gui_native missing - writing the native-pixel model from %s", model)
        write_gui_native(project, pycolmap.Reconstruction(str(model)), colors=False)
    return nat


def subset_images(project: SfmProject, station: Optional[str] = None, families: str = "N",
                  sequences: Optional[Sequence[str]] = None, names: Optional[Sequence[str]] = None,
                  scale: Optional[float] = None) -> List[str]:
    """The image names of a dense subset (0.96.0, Christian 9 Nov: "test the 3d meshing, reprojection and stereoscopic
    pipeline on one med res navcam sequence at sol 461"): the frames of ``station`` (S026D0630, 26_630, sol:461)
    whose camera family is in ``families``, of ``sequences`` (NCAM03461 ...) if given, at the downsample ``scale``
    (0.5 = the half-resolution frames) if given; or the ``names`` themselves."""
    if names:
        have = {r["name"] for r in project.images}
        missing = [n for n in names if n not in have]
        if missing:
            raise ValueError(f"not in the project: {missing[:3]}")
        return list(names)
    rows = list(project.images)
    if station:
        from ..ibr.station import parse_station
        site, drive, sol = parse_station(station)
        rows = [r for r in rows if (sol is None or int(r.get("sol") or -1) == sol)
                and (site is None or (int(r.get("site") or -1) == site and int(r.get("drive") or -1) == drive))]
    if families:
        rows = [r for r in rows if str(r.get("camera_group") or r.get("instrument") or "?")[:1] in set(families)]
    if sequences:
        rows = [r for r in rows if r.get("sequence") in set(sequences)]
    if scale is not None:
        rows = [r for r in rows if abs(float(r.get("downsample_scale") or 1.0) - float(scale)) < 1e-6]
    return sorted(r["name"] for r in rows)


def nearby_station_images(project: SfmProject, station: str, radius_m: float, families: str = "N",
                          scale: Optional[float] = None) -> Dict[str, Any]:
    """0.97.0 (Christian, 12 Nov 2026: "constrain the terrain from the masked images of several nearby stations"):
    the frames of the block's other stations whose position (the mean of their frames' prior centers) is within
    ``radius_m`` of ``station``'s - they see the ground this station's rover stands on, the rover being elsewhere.
    Returns {"names": [...], "stations": {label: {"distance_m", "images"}}}."""
    from ..ibr.station import parse_station
    site, drive, sol = parse_station(station)
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in project.images:
        if families and str(r.get("camera_group") or r.get("instrument") or "?")[:1] not in set(families):
            continue
        if r.get("site") is None or r.get("drive") is None or r.get("prior_C") is None:
            continue
        groups.setdefault((int(r["site"]), int(r["drive"])), []).append(r)
    if site is None:                                            # sol:NNNN - the station of that sol's frames
        own = [k for k, rows in groups.items() if any(int(x.get("sol") or -1) == sol for x in rows)]
        key = own[-1] if own else None
    else:
        key = (site, drive)
    out: Dict[str, Any] = {"names": [], "stations": {}}
    if key not in groups:
        return out
    c0 = np.mean([r["prior_C"] for r in groups[key]], axis=0)
    for k, rows in sorted(groups.items()):
        if k == key:
            continue
        d = float(np.linalg.norm(np.mean([r["prior_C"] for r in rows], axis=0) - c0))
        if d > float(radius_m):
            continue
        if scale is not None:
            rows = [r for r in rows if abs(float(r.get("downsample_scale") or 1.0) - float(scale)) < 1e-6]
        if rows:
            out["stations"][f"S{k[0]:03d}D{k[1]:04d}"] = {"distance_m": round(d, 2), "images": len(rows)}
            out["names"] += [r["name"] for r in rows]
    out["names"] = sorted(set(out["names"]))
    return out


def subset_label(station: Optional[str] = None, sequences: Optional[Sequence[str]] = None,
                 names: Optional[Sequence[str]] = None) -> str:
    """``dense_<station>[_<sequence>...]`` (or ``dense_subset_<n>``): the workspace of a subset."""
    if station:
        from ..ibr.station import parse_station
        site, drive, sol = parse_station(station)
        lab = f"S{site:03d}D{drive:04d}" if site is not None else f"sol{sol:04d}"
        return "_".join([f"{DENSE_DIR}_{lab}", *(sequences or [])])
    return f"{DENSE_DIR}_subset_{len(names or [])}"


def subset_model(model_dir: PathLike, names: Sequence[str], out_dir: PathLike):
    """The native model with only the frames of ``names`` registered (the others' observations dropped, points
    seen by fewer than two of the kept frames gone): ``out_dir`` (cameras, frames, images, points3D)."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(model_dir))
    keep = set(names)
    for fid, fr in list(rec.frames.items()):
        mine = [rec.images[d.id].name for d in fr.data_ids
                if d.sensor_id.type == pycolmap.SensorType.CAMERA and d.id in rec.images]
        if fr.has_pose and not any(n in keep for n in mine):
            rec.deregister_frame(fid)
    found = {im.name for im in rec.images.values() if im.has_pose}
    lost = sorted(keep - found)
    if lost:
        log.warning("dense subset: %d of the frames asked for are not registered in the model: %s", len(lost), lost[:3])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rec.write(str(out_dir))
    return rec


def prepare_dense(project: SfmProject, out_dir: Optional[PathLike] = None,
                  max_image_size: int = DENSE_MAX_IMAGE_SIZE, masks: bool = True, overwrite: bool = False,
                  num_threads: int = -1, images: Optional[Sequence[str]] = None, full_fov: bool = False) -> Dict[str, Any]:
    """The dense workspace ``out_dir`` (default ``<project>/dense``); returns its record (also ``mppp_dense.json``).
    ``images`` (0.96.0): only these frames (:func:`subset_images`) - a sub-model of them is undistorted, so the
    patch-match stereo and the fusion see nothing else (its own workspace, :func:`subset_label`).

    ``full_fov`` (10 Nov 2026, Christian: "the undistorted images throw away the edges of the fov leading to less
    deck coverage"): COLMAP's undistortion crops the pinhole image to the rectangle of valid pixels, which drops the
    fisheye's corners - 10 % of a Navcam frame (corner half-angle 60.9 -> 53.5 deg) and 11 % of the rover's pixels at
    Van Zyl S003D2046 (up to 17 % in a frame).  ``full_fov`` keeps the whole field of view (``blank_pixels`` 1; the
    corners outside the fisheye are black and unused) at a working size scaled up so that the center keeps the
    default's focal length (2000 -> ~2550 px for the half-resolution frames: ~1.8x the patch-match time)."""
    import numpy as np
    import pycolmap
    out = Path(out_dir) if out_dir else Path(project.root) / DENSE_DIR
    rec_file = out / "mppp_dense.json"
    if (out / "sparse" / "cameras.bin").is_file() and rec_file.is_file() and not overwrite:
        prev = json.loads(rec_file.read_text(encoding="utf-8"))
        if (int(prev.get("max_image_size", -1)) == int(max_image_size) and sorted(prev.get("subset") or []) == sorted(images or [])
                and bool(prev.get("full_fov", False)) == bool(full_fov)):
            log.info("dense: workspace %s is there (max_image_size %d%s); overwrite=True to redo it", out, max_image_size,
                     ", full field of view" if full_fov else "")
            return prev
        for sub in ("images", "sparse", "masks"):                 # other undistortion: the old images go first
            if (out / sub).exists():
                shutil.rmtree(out / sub)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = native_model_dir(project)
    if images:
        missing = [n for n in images if not (Path(project.images_dir) / n).is_file()]
        if missing:
            raise FileNotFoundError(f"dense subset: {len(missing)} image files missing in {project.images_dir}: {missing[:2]}")
        model = subset_model(model, images, out / "_model_subset") and (out / "_model_subset")
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = int(max_image_size)
    size_used = int(max_image_size)
    if full_fov:
        cams = list(pycolmap.Reconstruction(str(model)).cameras.values())
        ratio = 1.0
        if cams:
            cam = max(cams, key=lambda c: c.width)
            f_def = float(pycolmap.undistort_camera(opts, cam).params[0])
            o_full = pycolmap.UndistortCameraOptions()
            o_full.max_image_size = int(max_image_size)
            o_full.blank_pixels = 1.0
            f_full = float(pycolmap.undistort_camera(o_full, cam).params[0])
            ratio = max(1.0, f_def / max(f_full, 1e-9))
        size_used = int(np.ceil(max_image_size * ratio))
        opts.blank_pixels = 1.0
        opts.max_image_size = size_used
    from .reconstruction import default_threads
    nt = int(num_threads) if int(num_threads) > 0 else default_threads()
    pycolmap.undistort_images(str(out), str(model), str(project.images_dir), image_names=list(images or []),
                              output_type="COLMAP", undistort_options=opts, num_threads=nt)
    und = pycolmap.Reconstruction(str(out / "sparse"))
    res: Dict[str, Any] = {"workspace": str(out), "source_model": str(model), "images": und.num_images(),
                           "cameras": und.num_cameras(), "max_image_size": int(max_image_size), "masks": None,
           "full_fov": bool(full_fov), "max_image_size_used": int(size_used),
                           "seconds": None, "subset": sorted(images) if images else None}
    if masks and project.masks_dir.is_dir():
        names = [r["name"] for r in project.images if r.get("has_mask") and (project.masks_dir / (r["name"] + ".png")).is_file()]
        in_model = {im.name for im in und.images.values()}
        names = [n for n in names if n in in_model]
        if names:
            # the masks undistorted with the same cameras: a folder in which each mask carries its image's name,
            # undistorted into a workspace of its own, then moved to masks/<image name>.png (what the fusion reads),
            # thresholded back to 0 / 255 (the resampling makes gray edges)
            tmp_in, tmp_out = out / "_masks_in", out / "_masks_out"
            for d in (tmp_in, tmp_out):
                if d.exists():
                    shutil.rmtree(d)
            tmp_in.mkdir()
            from ..mask_levels import TIGHT_DIR, tight_mask_file
            tight = 0
            for n in names:
                # 0.99.9: the tight mask (the probability at the card's "tight" threshold) where the block has one;
                # masks/ is the feature extraction's, with its generous margin
                src = tight_mask_file(project.root, n, project.masks_dir)
                tight += int(src.parent.name == TIGHT_DIR)
                dst = tmp_in / n
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
            pycolmap.undistort_images(str(tmp_out), str(model), str(tmp_in), image_names=names, output_type="COLMAP",
                                      undistort_options=opts, num_threads=nt)
            mdir = out / "masks"
            mdir.mkdir(exist_ok=True)
            import cv2
            n_ok = 0
            for n in names:
                f = tmp_out / "images" / n
                if not f.is_file():
                    continue
                m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                m = np.where(m >= 128, 255, 0).astype(np.uint8)
                target = mdir / (n + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), m)
                n_ok += 1
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
            res["masks"] = {"dir": str(mdir), "images": n_ok, "tight": tight}
    res["seconds"] = round(time.time() - t0, 1)
    rec_file.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("dense: workspace %s - %d images undistorted at <= %d px%s in %.0f s", out, res["images"], max_image_size,
             f", {res['masks']['images']} masks" if res.get("masks") else "", res["seconds"])
    return res


STEREO_RECORD = ondisk.STEREO_FILE      # mppp_stereo.json (rpp.ondisk); stereo/<this>: the options the depth maps in the workspace were made with


def stereo_outputs_ready(ws: PathLike, record: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    """Make the workspace's patch-match outputs match ``record`` (the options of the run about to start).

    COLMAP's patch-match skips every image whose depth and normal maps already exist (10 Nov 2026 on the PC: a second
    ``--stereo`` with the new station options took 0.3 s and left the 8 m cut-off in place).  So when the maps were
    made with other options - or with unrecorded ones while this run sets some, or ``force`` - they are deleted first
    (depth_maps, normal_maps, consistency_graphs); maps made with the same options are kept, and COLMAP resumes the
    images still missing.  Returns {"cleared": n files, "why": ...}."""
    st = Path(ws) / "stereo"
    rec_file = st / STEREO_RECORD
    old = json.loads(rec_file.read_text(encoding="utf-8")) if rec_file.is_file() else None
    maps = [f for d in ("depth_maps", "normal_maps", "consistency_graphs") if (st / d).is_dir() for f in (st / d).glob("*.bin")]
    why = None
    if maps:
        if force:
            why = "asked (--restereo)"
        elif old is None and record.get("options"):
            why = "maps from a run with unrecorded options (before 0.96's record) and this run sets options"
        elif old is not None and old != record:
            why = "maps made with other options: " + ", ".join(
                f"{k} {old.get(k)!r} -> {record.get(k)!r}" for k in sorted(set(old) | set(record)) if old.get(k) != record.get(k))
    if why:
        for f in maps:
            f.unlink()
        log.info("dense: %d old patch-match maps deleted (%s)", len(maps), why)
    st.mkdir(parents=True, exist_ok=True)
    rec_file.write_text(json.dumps(record, indent=1, sort_keys=True, default=str), encoding="utf-8")
    return {"cleared": len(maps) if why else 0, "why": why, "kept": 0 if why else len(maps)}


def workspace_stereo_record(ws: PathLike) -> Optional[Dict[str, Any]]:
    """1.0: the record of the depth maps a workspace already holds (``stereo/mppp_stereo.json``, with at least one map),
    else None - a run that resumes takes these options, so that its maps are kept."""
    st = Path(ws) / "stereo"
    f = st / STEREO_RECORD
    if not f.is_file() or not any((st / "depth_maps").glob("*.bin")):
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def set_source_images(ws: PathLike, n: int) -> int:
    """1.0: ``stereo/patch-match.cfg`` with ``__auto__, n`` for every image (the n images that share the most tie points
    with it); the images changed.  A list of named source images is left as it is."""
    f = Path(ws) / "stereo" / "patch-match.cfg"
    if not f.is_file():
        return 0
    lines = f.read_text(encoding="utf-8").splitlines()
    k = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("__auto__"):
            new = f"__auto__, {int(n)}"
            if ln.strip() != new:
                lines[i] = new
                k += 1
    if k:
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return k


def stereo_record(ws: PathLike, stereo_options: Optional[Dict[str, Any]], geom_consistency: bool, max_image_size: int,
                  sources: Optional[int] = None):
    """(PatchMatchOptions, the fields set, the record of ``stereo/mppp_stereo.json``) of a stereo run on ``ws``.
    ``sources`` (1.0): the source images per image (patch-match.cfg); the record names it only when it is not
    COLMAP's 20, so the records of the maps made before 1.0 stay what they were."""
    import pycolmap
    ws = Path(ws)
    wrec = ws / "mppp_dense.json"
    if wrec.is_file():
        max_image_size = int(json.loads(wrec.read_text(encoding="utf-8")).get("max_image_size_used") or max_image_size)
    po = pycolmap.PatchMatchOptions()
    po.geom_consistency = bool(geom_consistency)
    po.max_image_size = int(max_image_size)
    so = set_options(po, stereo_options)
    fov = json.loads(wrec.read_text(encoding="utf-8")).get("full_fov", False) if wrec.is_file() else False
    rec = {"options": {k: float(v) if isinstance(v, float) else v for k, v in so.items()},
           "geom_consistency": bool(geom_consistency), "max_image_size": int(max_image_size), "full_fov": bool(fov)}
    if sources is not None and int(sources) != COLMAP_SOURCES:
        rec["sources"] = int(sources)
    return po, so, rec


def stereo_cached(ws: PathLike, stereo_options: Optional[Dict[str, Any]] = None, geom_consistency: bool = True,
                  max_image_size: int = 0, sources: Optional[int] = None) -> bool:
    """0.97.0 (Christian, 12 Nov 2026: "Don't wait for the GPU when the stereo maps are cached"): True when every
    image of the workspace has its depth and normal map from a run with these very options - a stereo step would
    do nothing, so it needs neither the GPU nor the wait for it."""
    ws = Path(ws)
    st = ws / "stereo"
    rec_file = st / STEREO_RECORD
    imgs = [f for f in (ws / "images").rglob("*") if f.is_file()] if (ws / "images").is_dir() else []
    if not imgs or not rec_file.is_file():
        return False
    try:
        _, _, rec = stereo_record(ws, stereo_options, geom_consistency, max_image_size, sources)
        old = json.loads(rec_file.read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001 - then the stereo step decides
        return False
    if json.loads(json.dumps(rec, sort_keys=True, default=str)) != old:
        return False
    kind = "geometric" if geom_consistency else "photometric"
    for f in imgs:
        rel = f.relative_to(ws / "images").as_posix()
        if not all((st / d / f"{rel}.{kind}.bin").is_file() for d in ("depth_maps", "normal_maps")):
            return False
    return True


# 0.99.20 (Christian, 17 Nov 2026, Rockytop: "the sky had many floating artifacts behind it" - "Keep sky out of the dense
# cloud"): THE SKY MASK.  The patch-match stereo gives the sky depths too (a smooth sky matches anywhere), and the fusion
# keeps those that agree by chance: points floating behind the skyline.  The sky of an undistorted frame is found from
# the picture and the camera's pose: smooth (the gradient's size, averaged, under SKY_TEXTURE), above SKY_MIN_EL_DEG
# (the frame's pose gives each pixel's elevation), as bright as the sky seen higher up (the smooth pixels above
# SKY_SEED_EL_DEG set the sky's brightness; within SKY_BRIGHT_TOL of it - a smooth sand slope under the horizon is
# darker), and joined to those higher pixels.  The fusion's mask of a frame is its rover mask without its sky.
SKY_MASK = True
SKY_WORK_PX = 640                 # the picture's long side for the test
SKY_TEXTURE = 0.9                 # 8-bit levels per pixel (the sky of the Rockytop frames: under 0.3; terrain: 2-5)
SKY_MIN_EL_DEG = -3.0
SKY_SEED_EL_DEG = 4.0
SKY_BRIGHT_TOL = 30.0             # 8-bit levels around the seeds' median (or 4 x their spread, when that is more)
SKY_GROW_PX = 2                   # the sky grown by this many pixels of the frame (the silhouette's mixed pixels)
SKY_MIN_SHARE = 0.002             # less sky than this share of the frame: none
SKY_MASK_DIR = "masks_fusion"


def pixel_elevation_deg(shape: Tuple[int, int], pinhole: Sequence[float], R_cam_from_world: np.ndarray, scale: float = 1.0,
                        up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """The elevation (deg above the horizon of ``up``) of every pixel's ray of an undistorted (PINHOLE fx fy cx cy)
    frame reduced by ``scale``."""
    h, w = shape
    fx, fy, cx, cy = [float(v) for v in pinhole[:4]]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.stack([((xx + 0.5) / scale - cx) / fx, ((yy + 0.5) / scale - cy) / fy, np.ones((h, w))], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip((d @ np.asarray(R_cam_from_world, float)) @ np.asarray(up, float), -1.0, 1.0)))


def sky_mask(image: np.ndarray, elevation_deg: Optional[np.ndarray] = None, pinhole: Optional[Sequence[float]] = None,
             R_cam_from_world: Optional[np.ndarray] = None) -> np.ndarray:
    """True where ``image`` (gray or BGR, 8 bit; black = no picture) shows the sky - see THE SKY MASK.  The pixels'
    elevations come from ``elevation_deg`` (at the reduced size) or from the frame's ``pinhole`` and rotation."""
    import cv2
    from scipy import ndimage
    g = image if image.ndim == 2 else image[..., :3].max(axis=2)
    H, W = g.shape
    sc = min(1.0, SKY_WORK_PX / float(max(H, W)))
    q = cv2.resize(g, (max(1, int(round(W * sc))), max(1, int(round(H * sc)))), interpolation=cv2.INTER_AREA).astype(np.float32)
    valid = ndimage.binary_erosion(q > 0, iterations=2)
    b = cv2.GaussianBlur(q, (0, 0), 1.0)
    tex = cv2.GaussianBlur(np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)) / 8.0, (0, 0), 2.0)
    if elevation_deg is None:
        elevation_deg = pixel_elevation_deg(q.shape, pinhole, R_cam_from_world, scale=sc)
    el = np.asarray(elevation_deg, float)
    smooth = valid & (tex < SKY_TEXTURE)
    seed = smooth & (el > SKY_SEED_EL_DEG)
    none = np.zeros((H, W), bool)
    if seed.sum() < max(20, SKY_MIN_SHARE * q.size):
        return none
    med = float(np.median(b[seed]))
    tol = max(SKY_BRIGHT_TOL, 4.0 * 1.4826 * float(np.median(np.abs(b[seed] - med))))
    cand = smooth & (el > SKY_MIN_EL_DEG) & (np.abs(b - med) <= tol)
    lab, _ = ndimage.label(cand, structure=np.ones((3, 3), bool))
    ids = np.unique(lab[seed & cand])
    sky = np.isin(lab, ids[ids > 0])
    sky = ndimage.binary_fill_holes(sky) & valid                   # a speck in the sky is sky
    rim = (q > 0) & ~valid                                         # the picture's rim, left out of the test above:
    sky |= rim & ndimage.binary_dilation(sky, iterations=4)        # sky where the sky reaches it
    if sky.sum() < SKY_MIN_SHARE * q.size:
        return none
    full = cv2.resize(sky.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    if SKY_GROW_PX:
        full = cv2.dilate(full.astype(np.uint8), np.ones((2 * SKY_GROW_PX + 1,) * 2, np.uint8)) > 0
    return full & (g > 0)


def write_sky_masks(ws: PathLike, force: bool = False) -> Dict[str, Any]:
    """``<ws>/masks_fusion/<image>.png`` for every frame of a dense workspace: the frame's rover mask (``masks/``, when
    it has one) with its sky taken out (0 = not fused).  Made once per workspace (``sky_masks.json``); ``force``
    again.  Returns {"dir", "images", "with_sky", "sky_share_median", "seconds"}."""
    import cv2
    import pycolmap
    ws = Path(ws)
    out = ws / SKY_MASK_DIR
    marker = ws / "sky_masks.json"
    key = {"texture": SKY_TEXTURE, "min_el_deg": SKY_MIN_EL_DEG, "seed_el_deg": SKY_SEED_EL_DEG, "bright_tol": SKY_BRIGHT_TOL,
           "grow_px": SKY_GROW_PX}
    if marker.is_file() and out.is_dir() and not force:
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("settings") == key and old.get("images") == len(list((ws / "images").glob("*"))):
            return dict(old, cached=True)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rec = pycolmap.Reconstruction(str(ws / "sparse"))
    shares, n = [], 0
    for im in rec.images.values():
        f = ws / "images" / im.name
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED) if f.is_file() else None
        if img is None:
            continue
        cam = rec.cameras[im.camera_id]
        sky = sky_mask(img, pinhole=cam.params, R_cam_from_world=im.cam_from_world().rotation.matrix())
        rm = ws / "masks" / (im.name + ".png")
        m = cv2.imread(str(rm), cv2.IMREAD_GRAYSCALE) if rm.is_file() else None
        if m is None or m.shape != sky.shape:
            m = np.full(sky.shape, 255, np.uint8)
        m = np.where(sky, 0, m).astype(np.uint8)
        (out / (im.name + ".png")).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / (im.name + ".png")), m)
        shares.append(float(sky.mean()))
        n += 1
    rep = {"dir": str(out), "images": n, "with_sky": int(sum(1 for x in shares if x > 0)),
           "sky_share_median": float(np.median([x for x in shares if x > 0])) if any(x > 0 for x in shares) else 0.0,
           "settings": key, "seconds": round(time.time() - t0, 1)}
    marker.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log.info("dense: sky masks - %d of %d frames show sky (median %.0f %% of the frame) -> %s", rep["with_sky"], n,
             100 * rep["sky_share_median"], out)
    return rep


def set_options(opts: Any, values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Set the fields ``values`` on a pycolmap options object; returns the fields as set (a typo raises)."""
    done: Dict[str, Any] = {}
    for k, v in (values or {}).items():
        if not hasattr(opts, k):
            raise ValueError(f"{type(opts).__name__} has no field {k!r}")
        setattr(opts, k, type(getattr(opts, k))(v))
        done[k] = getattr(opts, k)
    return done


def run_dense(project: SfmProject, workspace: Optional[PathLike] = None, steps=("prepare", "stereo", "fusion"),
              max_image_size: int = DENSE_MAX_IMAGE_SIZE, geom_consistency: bool = True,
              gpu_index: str = "-1", images: Optional[Sequence[str]] = None,
              stereo_options: Optional[Dict[str, Any]] = None,
              fusion_options: Optional[Dict[str, Any]] = None, keep_rover: bool = False,
              restereo: bool = False, full_fov: bool = False, sky: Optional[bool] = None,
              sources: Optional[int] = None, fusion_strict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """0.99.20: ``sky`` (default :data:`SKY_MASK`) - the fusion leaves the frames' sky out (:func:`write_sky_masks`).

    The dense steps from Python: ``prepare`` (:func:`prepare_dense`), ``stereo`` (patch-match stereo; needs a CUDA
    pycolmap), ``fusion`` (``dense/fused.ply``, the masks applied), ``mesh`` (``dense/meshed-poisson.ply``).
    ``stereo_options`` / ``fusion_options``: ``PatchMatchOptions`` / ``StereoFusionOptions`` fields to set (e.g.
    :data:`STATION_STEREO_OPTIONS`); COLMAP's defaults otherwise.  ``keep_rover``: the fusion without the rover masks,
    into ``fused_rover.ply`` (the rover body and deck kept; the stereo is the same - it takes no masks).
    ``sources`` (1.0): the source images per image (:func:`set_source_images`; None: as the workspace has them).
    ``fusion_strict`` (1.0): a second fusion with these options into ``fused_strict.ply`` (not with ``keep_rover``);
    :func:`merge_fusions` then makes ``fused.ply`` of the two."""
    import pycolmap
    ws = Path(workspace) if workspace else Path(project.root) / DENSE_DIR
    out: Dict[str, Any] = {"workspace": str(ws), "steps": {}}
    if "prepare" in steps or not (ws / "sparse" / "cameras.bin").is_file():
        out["steps"]["prepare"] = prepare_dense(project, ws, max_image_size=max_image_size, images=images, full_fov=full_fov)
    wrec = ws / "mppp_dense.json"                               # the size the workspace's images were made at
    if wrec.is_file():
        max_image_size = int(json.loads(wrec.read_text(encoding="utf-8")).get("max_image_size_used") or max_image_size)
    if "stereo" in steps:
        if not getattr(pycolmap, "has_cuda", False):
            raise RuntimeError("patch-match stereo needs a CUDA pycolmap (the gpu_python environment) or "
                               "dense_reconstruct.bat with a CUDA COLMAP")
        t0 = time.time()
        po, so, srec = stereo_record(ws, stereo_options, geom_consistency, max_image_size, sources)
        po.gpu_index = str(gpu_index)
        state = stereo_outputs_ready(ws, srec, force=restereo)
        if sources is not None:                                  # 1.0: after the old maps went, if they did
            set_source_images(ws, int(sources))
        pycolmap.patch_match_stereo(str(ws), "COLMAP", options=po)
        out["steps"]["stereo"] = {"seconds": round(time.time() - t0, 1), "geom_consistency": bool(geom_consistency),
                                  "options": so, "old_maps": state}
    if "fusion" in steps:
        t0 = time.time()
        fo = pycolmap.StereoFusionOptions()
        used_masks = (ws / "masks").is_dir() and not keep_rover
        if used_masks:
            fo.mask_path = str(ws / "masks")
        sky_rep = None
        if sky is None:                                          # RPP_SKY_MASK=0 turns it off
            from .. import env as rpp_env
            sky = SKY_MASK and str(rpp_env.get("RPP_SKY_MASK") or "1").strip().lower() not in ("0", "false", "no")
        if sky and not keep_rover:                               # 0.99.20: the rover masks without the sky
            sky_rep = write_sky_masks(ws)
            fo.mask_path = sky_rep["dir"]
            used_masks = True
        ply = ws / (FUSED_ROVER_PLY if keep_rover else "fused.ply")
        fo.max_image_size = int(max_image_size)
        fset = set_options(fo, fusion_options)
        # 9 Nov (the sandbox run on Rockytop): pycolmap 4.2's stereo_fusion writes a model folder by default
        # (output_type "bin") - the fused points never reached fused.ply and the call raised after the fusion
        fused = pycolmap.stereo_fusion(str(ply), str(ws), "COLMAP", input_type="geometric" if geom_consistency else "photometric",
                                       options=fo, output_type="ply")
        out["steps"]["fusion"] = {"seconds": round(time.time() - t0, 1), "points": fused.num_points3D(),
                                  "ply": str(ply), "masks": str(fo.mask_path) if used_masks else None, "sky": sky_rep, "options": fset, "rover_kept": bool(keep_rover)}
        if fusion_strict and not keep_rover:                     # 1.0: the second, strict pass with the same masks
            t1 = time.time()
            fs = pycolmap.StereoFusionOptions()
            fs.mask_path = fo.mask_path
            fs.max_image_size = int(max_image_size)
            sset = set_options(fs, {**(fusion_options or {}), **fusion_strict})
            strict = pycolmap.stereo_fusion(str(ws / FUSED_STRICT_PLY), str(ws), "COLMAP",
                                            input_type="geometric" if geom_consistency else "photometric", options=fs, output_type="ply")
            out["steps"]["fusion_strict"] = {"seconds": round(time.time() - t1, 1), "points": strict.num_points3D(),
                                             "ply": str(ws / FUSED_STRICT_PLY), "options": sset}
    if "mesh" in steps:
        t0 = time.time()
        pycolmap.poisson_meshing(str(ws / "fused.ply"), str(ws / "meshed-poisson.ply"))
        out["steps"]["mesh"] = {"seconds": round(time.time() - t0, 1), "ply": str(ws / "meshed-poisson.ply")}
    (ws / "mppp_dense_run.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------- the two fusions (1.0)
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
              "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}


def read_ply_vertices(path: PathLike) -> Tuple[bytes, np.ndarray]:
    """(the header, the vertex records) of a binary little-endian PLY of vertices only (COLMAP's fused cloud)."""
    with open(path, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            ln = fh.readline()
            if not ln:
                raise ValueError(f"{path}: no end_header")
            head += ln
        txt = head.decode("ascii", "replace").splitlines()
        if "format binary_little_endian 1.0" not in txt:
            raise ValueError(f"{path}: not a binary little-endian PLY")
        n, fields, inside = 0, [], False
        for ln in txt:
            w = ln.split()
            if w[:2] == ["element", "vertex"]:
                n, inside = int(w[2]), True
            elif w[:1] == ["element"]:
                inside = False
            elif inside and w[:1] == ["property"] and len(w) == 3:
                fields.append((w[2], _PLY_TYPES[w[1]]))
        data = np.fromfile(fh, dtype=np.dtype(fields), count=n)
    return head, data


def write_ply_vertices(path: PathLike, head: bytes, data: np.ndarray) -> None:
    import re
    head = re.sub(rb"element vertex \d+", b"element vertex %d" % len(data), head, count=1)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(head)
        data.tofile(fh)
    os.replace(tmp, path)


def read_vis(path: PathLike) -> List[np.ndarray]:
    """COLMAP's ``fused.ply.vis``: per point, the images that see it."""
    raw = np.fromfile(path, dtype=np.uint32)
    n = int(raw[0]) | (int(raw[1]) << 32)
    out, i = [], 2
    for _ in range(n):
        k = int(raw[i])
        out.append(raw[i + 1:i + 1 + k])
        i += 1 + k
    return out


def write_vis(path: PathLike, vis: Sequence[np.ndarray]) -> None:
    parts = [np.array([len(vis) & 0xFFFFFFFF, len(vis) >> 32], np.uint32)]
    for v in vis:
        parts.append(np.array([len(v)], np.uint32))
        parts.append(np.asarray(v, np.uint32))
    np.concatenate(parts).tofile(path)


def merge_fusions(ws: PathLike, merge_px: float = MERGE_PX) -> Dict[str, Any]:
    """1.0: ``fused.ply`` (the loose fusion) and ``fused_strict.ply`` into one cloud: every strict point, and the loose
    points farther than ``merge_px`` working pixels from any strict point - a pixel at the point's distance from the
    nearest camera of the workspace (its median focal length).  The loose cloud is kept as ``fused_loose.ply``; the
    ``.vis`` files go with their points.  Needs scipy (the main environment)."""
    from scipy.spatial import cKDTree
    ws = Path(ws)
    t0 = time.time()
    loose_f, strict_f = ws / "fused.ply", ws / FUSED_STRICT_PLY
    if not strict_f.is_file():
        raise FileNotFoundError(f"{strict_f}: no strict fusion to merge")
    if (ws / FUSED_LOOSE_PLY).is_file() and loose_f.stat().st_mtime <= (ws / FUSED_LOOSE_PLY).stat().st_mtime:
        loose_f = ws / FUSED_LOOSE_PLY                            # fused.ply is an earlier merge: from the loose cloud again
    head, L = read_ply_vertices(loose_f)
    _, S = read_ply_vertices(strict_f)
    if L.dtype != S.dtype:
        raise ValueError("the two fusions' PLY records differ")
    xyz = lambda a: np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)   # noqa: E731
    PL, PS = xyz(L), xyz(S)
    import pycolmap
    rec = pycolmap.Reconstruction(str(ws / "sparse"))
    C = np.array([im.projection_center() for im in rec.images.values()], float)
    fpx = float(np.median([c.params[0] for c in rec.cameras.values()]))
    d_cam, _ = cKDTree(C).query(PL, k=1, workers=-1)
    r = merge_px * np.maximum(d_cam, 0.1) / max(fpx, 1.0)
    d_s, _ = cKDTree(PS).query(PL, k=1, workers=-1, distance_upper_bound=float(r.max()) if len(r) else 0.0) if len(PS) else (np.full(len(PL), np.inf), None)
    keep = d_s > r
    out = np.concatenate([S, L[keep]])
    if loose_f.name == "fused.ply":                              # the first merge: the loose cloud kept by its own name
        os.replace(loose_f, ws / FUSED_LOOSE_PLY)
        if (ws / "fused.ply.vis").is_file():
            os.replace(ws / "fused.ply.vis", ws / (FUSED_LOOSE_PLY + ".vis"))
    write_ply_vertices(ws / "fused.ply", head, out)
    vis_rep = None
    lv, sv = ws / (FUSED_LOOSE_PLY + ".vis"), ws / (FUSED_STRICT_PLY + ".vis")
    if lv.is_file() and sv.is_file():
        vl, vs = read_vis(lv), read_vis(sv)
        write_vis(ws / "fused.ply.vis", list(vs) + [vl[i] for i in np.flatnonzero(keep)])
        vis_rep = "fused.ply.vis"
    rep = {"strict_points": int(len(S)), "loose_points": int(len(L)), "loose_kept": int(keep.sum()),
           "points": int(len(out)), "merge_px": float(merge_px), "focal_px": round(fpx, 1), "vis": vis_rep,
           "seconds": round(time.time() - t0, 1)}
    (ws / "rpp_fusion_merge.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    log.info("dense: fused.ply = %d strict + %d of %d loose points (%.0f s)", len(S), int(keep.sum()), len(L), rep["seconds"])
    return rep


def read_colmap_array(path: PathLike):
    """A COLMAP dense map (``stereo/depth_maps/<image>.{photometric,geometric}.bin``, ``normal_maps``): the text header
    ``width&height&channels&`` then float32 in column-major order; returns (height, width) or (height, width, 3)."""
    import numpy as np
    raw = Path(path).read_bytes()
    k, n = 0, 0
    while n < 3:
        if raw[k:k + 1] == b"&":
            n += 1
        k += 1
    w, h, c = (int(x) for x in raw[:k].decode("ascii").strip("&").split("&"))
    a = np.frombuffer(raw[k:], dtype=np.float32, count=w * h * c).reshape((w, h, c), order="F")
    return np.transpose(a, (1, 0, 2)).squeeze()


MAPS_MAX_DEPTH_M = 100.0           # 0.95.1 (Christian, 29 Oct): depth beyond this is no depth in the map images


def _map_mask(ws: Path, image_name: str, shape) -> Optional[Any]:
    """The rover mask of an image in the workspace (``masks/<image>.png``, 0 = rover: not fused) at ``shape``
    (nearest neighbor when the map's resolution differs); True where the terrain is.  None without a mask."""
    import numpy as np
    from PIL import Image
    f = ws / "masks" / (image_name + ".png")
    if not f.is_file():
        return None
    m = Image.open(f).convert("L")
    if m.size != (shape[1], shape[0]):
        m = m.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(m) > 0


def export_dense_maps(workspace: PathLike, kind: str = "geometric", out_dir: Optional[PathLike] = None,
                      fmt: str = "png", lo_hi: tuple = (2.0, 98.0), max_depth_m: float = MAPS_MAX_DEPTH_M,
                      masks: bool = True) -> Dict[str, Any]:
    """0.95.1 (Christian, 29 Oct: "how can I view those in jpeg or png?"): the depth and normal maps of a dense
    workspace as images in ``<workspace>/stereo/maps_png/`` - a depth map in a color scale over its own 2-98th
    percentile (near blue, far red) with the range in the file name, a normal map as RGB = (n + 1) / 2 in the camera
    frame (x right, y down, z forward).  No value ("NaN": transparent in a PNG, black in a JPEG) where there is no
    depth, where the depth is beyond ``max_depth_m`` (100 m) and inside the rover mask (``masks/``, when ``masks``);
    the normal map gets the same cut.  ``kind``: "geometric" (after the consistency check, what the fusion uses) or
    "photometric"."""
    import numpy as np
    from PIL import Image
    ws = Path(workspace)
    out = Path(out_dir) if out_dir else ws / "stereo" / "maps_png"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib import colormaps
        cmap = colormaps["turbo"]
    except Exception:                                      # noqa: BLE001
        cmap = None
    jpeg = fmt.lower() in ("jpg", "jpeg")

    def save(rgb, ok, path):
        if jpeg:
            rgb = rgb.copy()
            rgb[~ok] = 0
            Image.fromarray(rgb).save(path, quality=92)
        else:
            Image.fromarray(np.dstack([rgb, np.where(ok, 255, 0).astype(np.uint8)]), "RGBA").save(path)

    n = {"depth": 0, "normal": 0, "masked": 0}
    for f in sorted((ws / "stereo" / "depth_maps").glob(f"*.{kind}.bin")):
        name = f.name[: -len(f".{kind}.bin")]                     # the image's name (NLF_....png)
        stem = name.rsplit(".", 1)[0] if name.lower().endswith((".png", ".jpg", ".tif")) else name
        d = read_colmap_array(f)
        ok = (d > 0) & (d <= float(max_depth_m))
        m = _map_mask(ws, name, d.shape) if masks else None
        if m is not None:
            ok &= m
            n["masked"] += 1
        if ok.any():
            d0, d1 = np.percentile(d[ok], lo_hi)
            t = np.clip((d - d0) / max(d1 - d0, 1e-9), 0, 1)
            rgb = (cmap(t)[..., :3] * 255).astype(np.uint8) if cmap else np.repeat((t * 255).astype(np.uint8)[..., None], 3, 2)
            save(rgb, ok, out / f"{stem}.depth_{d0:.2f}-{d1:.2f}m.{fmt}")
            n["depth"] += 1
        fn = ws / "stereo" / "normal_maps" / f.name
        if fn.is_file():
            a = read_colmap_array(fn)
            okn = ok & (np.linalg.norm(a, axis=-1) > 0) if a.shape[:2] == ok.shape else (np.linalg.norm(a, axis=-1) > 0)
            rgb = ((np.clip(a, -1, 1) + 1) * 127.5).astype(np.uint8)
            save(rgb, okn, out / f"{stem}.normal.{fmt}")
            n["normal"] += 1
    return {"folder": str(out), "depth_maps": n["depth"], "normal_maps": n["normal"], "with_masks": n["masked"],
            "kind": kind, "max_depth_m": float(max_depth_m)}


def write_dense_bat(project: SfmProject, colmap_bat: Optional[str] = None, max_image_size: int = DENSE_MAX_IMAGE_SIZE) -> Path:
    """``<project>/dense_reconstruct.bat`` (next to ``open_in_colmap.bat``): the dense reconstruction with the COLMAP
    executable - the workspace from ``gui_native/sparse`` and ``images/`` unless it is there, the stereo, the fusion
    with the masks, a Poisson mesh with ``RPP_DENSE_MESH=1``.  A CUDA COLMAP is looked for first (the stereo
    needs one); the masks need ``rpp dense --prepare`` (pycolmap) or are left out."""
    from .. import __version__
    from .database import COLMAP_BAT_CANDIDATES
    root = Path(project.root).resolve()
    env_bat = Path(__file__).resolve().parents[3] / "scripts" / "windows" / "rpp_env.bat"   # 0.95.1: RPP's Python
    cands = ([str(colmap_bat or project.settings.get("colmap_bat") or "")] if (colmap_bat or project.settings.get("colmap_bat")) else [])
    cuda_first = [c for c in COLMAP_BAT_CANDIDATES if "cuda" in c.lower() and "nocuda" not in c.lower()]
    rest = [c for c in COLMAP_BAT_CANDIDATES if c not in cuda_first]
    lines = ["@echo off", "setlocal EnableExtensions",
             f"rem RPP {__version__}: dense reconstruction of this block with COLMAP (4.x) - the workspace dense\\ from the",
             "rem native-pixel model gui_native\\sparse and images\\, patch-match stereo (CUDA), fusion with the rover masks.",
             "rem Double-click.  set RPP_DENSE_STEP=prepare|stereo|fusion|mesh for one step; RPP_DENSE_MAX_SIZE (long side,",
             f"rem default {int(max_image_size)} px); RPP_DENSE_MESH=1 for a Poisson mesh at the end.  The GUI's Reconstruction >",
             "rem Dense reconstruction dialog takes the same dense\\ folder as its workspace (open_in_colmap.bat first).",
             'set "WS=%~dp0dense"',
             f'if not defined RPP_DENSE_MAX_SIZE set "RPP_DENSE_MAX_SIZE={int(max_image_size)}"',
             'if not defined RPP_DENSE_STEP set "RPP_DENSE_STEP=all"',
             'set "CB="',
             'if defined COLMAP_BAT if exist "%COLMAP_BAT%" set "CB=%COLMAP_BAT%"']
    for c in cands + cuda_first:
        lines.append(f'if not defined CB if exist "{c}" set "CB={c}"')
    for c in rest:
        lines.append(f'if not defined CB if exist "{c}" set "CB={c}"')
    lines += ['if not defined CB for %%I in (COLMAP.bat) do if not "%%~$PATH:I"=="" set "CB=%%~$PATH:I"',
              'if not defined CB (echo COLMAP.bat not found: setx COLMAP_BAT ^<path^>\\COLMAP.bat  ^(a CUDA build for the stereo^) & pause & exit /b 1)',
              'echo %CB%',
              'echo "%CB%" | find /i "nocuda" >nul && echo WARNING: a COLMAP without CUDA - patch_match_stereo will not run; the workspace can still be prepared.',
              'if not exist "%~dp0gui_native\\sparse\\cameras.bin" (echo No gui_native\\sparse yet: run  rpp dense --prepare  ^(it writes it^) or open_in_colmap.bat after a run with the full export. & pause & exit /b 1)',
              'if /i "%RPP_DENSE_STEP%"=="stereo" goto :stereo',
              'if /i "%RPP_DENSE_STEP%"=="fusion" goto :fusion',
              'if /i "%RPP_DENSE_STEP%"=="mesh" goto :mesh',
              ':prepare',
              'if exist "%WS%\\sparse\\cameras.bin" goto :prepared',
              # 0.95.1 (Christian, 29 Oct: "does dense_reconstruct.bat also undistort the masks?"): RPP's Python
              # prepares the workspace with the rover masks (rpp dense --prepare --no-bat: this file is not rewritten
              # while it runs); without it, COLMAP's undistorter and no masks
              f'if exist "{env_bat}" call "{env_bat}" >nul 2>nul',
              'python -m rpp version >nul 2>nul && (',
              '  echo %DATE% %TIME%  undistorting images\\ and the rover masks with gui_native\\sparse into %WS% ^(max %RPP_DENSE_MAX_SIZE% px^)',
              '  python -m rpp dense "%~dp0." --prepare --no-bat --max-image-size %RPP_DENSE_MAX_SIZE%',
              '  if not errorlevel 1 goto :prepared',
              '  echo rpp dense --prepare failed - COLMAP\'s undistorter instead ^(no masks^)',
              ')',
              '(',
              '  echo %DATE% %TIME%  undistorting images\\ with gui_native\\sparse into %WS% ^(max %RPP_DENSE_MAX_SIZE% px^)',
              '  call "%CB%" image_undistorter --image_path "%~dp0images" --input_path "%~dp0gui_native\\sparse" --output_path "%WS%" --output_type COLMAP --max_image_size %RPP_DENSE_MAX_SIZE%',
              '  if errorlevel 1 (echo image_undistorter failed & pause & exit /b 1)',
              '  echo NOTE: the rover masks are not undistorted by this step - run  rpp dense --prepare  for a workspace with masks ^(the fusion then leaves the rover out^).',
              ')',
              ':prepared',
              'if exist "%WS%\\masks" (echo rover masks: %WS%\\masks ^(the fusion leaves the rover out^)) else (echo no masks in %WS%: the fusion keeps the rover)',
              'if /i "%RPP_DENSE_STEP%"=="prepare" goto :done',
              ':stereo',
              'echo %DATE% %TIME%  patch-match stereo ^(geometric consistency^) - hours on the GPU',
              'call "%CB%" patch_match_stereo --workspace_path "%WS%" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size %RPP_DENSE_MAX_SIZE%',
              'if errorlevel 1 (echo patch_match_stereo failed ^(a CUDA COLMAP is needed^) & pause & exit /b 1)',
              'if /i "%RPP_DENSE_STEP%"=="stereo" goto :done',
              ':fusion',
              'set "MASKS="',
              'if exist "%WS%\\masks" set MASKS=--StereoFusion.mask_path "%WS%\\masks"',
              'echo %DATE% %TIME%  stereo fusion -^> %WS%\\fused.ply %MASKS%',
              'call "%CB%" stereo_fusion --workspace_path "%WS%" --workspace_format COLMAP --input_type geometric --output_path "%WS%\\fused.ply" --StereoFusion.max_image_size %RPP_DENSE_MAX_SIZE% %MASKS%',
              'if errorlevel 1 (echo stereo_fusion failed & pause & exit /b 1)',
              'if /i "%RPP_DENSE_STEP%"=="fusion" goto :done',
              'if not defined RPP_DENSE_MESH goto :done',
              ':mesh',
              'echo %DATE% %TIME%  Poisson mesh -^> %WS%\\meshed-poisson.ply',
              'call "%CB%" poisson_mesher --input_path "%WS%\\fused.ply" --output_path "%WS%\\meshed-poisson.ply"',
              'if errorlevel 1 (echo poisson_mesher failed & pause & exit /b 1)',
              ':done',
              'echo %DATE% %TIME%  done: %WS%  ^(fused.ply opens in CloudCompare / MeshLab; File ^> Import model in the COLMAP GUI shows dense\\sparse^)',
              'pause']
    path = root / "dense_reconstruct.bat"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path
