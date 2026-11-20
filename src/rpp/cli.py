"""
rpp: Mars rover PDS images to aligned, true-scale 3D.

  First steps
    rpp doctor                       what is installed and set, and what is missing
    rpp example fetch                download a small Mars 2020 example (Rockytop, 36 PDS products)
    rpp run rockytop_example --zcam  process and align it

  The pipeline
    rpp pds fetch --sols A B         download the PDS products of a sol range into RPP_PDS
    rpp sites [list|check|folder]    the blocks of rover stations RPP knows (rpp/data/sites.json)
    rpp process SITE|WORK            PDS products to 16-bit images, masks, cameras and poses
    rpp align WORK                   the COLMAP project: features, matching, bundle adjustment, health report
    rpp run SITE|WORK                process, then align
    rpp cmods [list|status|...]      the camera models in use
    rpp mask [list|fetch|install]    the rover mask model

  Exports
    rpp gui                          the files that open a finished block in COLMAP or Metashape
    rpp metashape WORK               an Agisoft Metashape job from an aligned block
    rpp nerfstudio WORK              a Nerfstudio data set from an aligned block

  Station products and the viewer (experimental; the "stations" extra)
    rpp dense WORK --station S       the dense cloud around one rover station
    rpp mesh / depthmaps / ods       its mesh, per-image depth maps and stereo panorama
    rpp single-sites [STATION ...]   all of the above for the stations of a list, and the viewer's files

  Research studies (a source checkout: src/rpp_studies)
    rpp calib ...                    fit and check the camera models from aligned blocks
    rpp zcam-states ...              the Mastcam-Z focus-backlash states
    rpp maps ...                     the site-map figures

"rpp COMMAND --help" lists a command's options.  -v / -q set the logging level and --log FILE copies the log to a
file.  process, align and run also write <WORK>/runs/<time>/log.txt and <WORK>/mppp_status.json.  Exit codes: 0 ok,
1 failed, 2 bad arguments, 3 a run is already going on.  Documentation: https://ctate7163.github.io/rpp/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from . import env as rpp_env

log = logging.getLogger("rpp.cli")

_SET_HELP = "a pipeline setting (rpp.pipeline.Settings field), VALUE as JSON or Python; repeatable"


def _common(ap: argparse.ArgumentParser, root: bool = True) -> None:
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help=_SET_HELP)
    ap.add_argument("--settings", default=None, metavar="FILE", help="a JSON file of settings {NAME: value}")
    ap.add_argument("--sites-file", default=None, metavar="FILE", help="site definitions (default rpp/data/sites.json)")
    if root:
        ap.add_argument("--root", default=None, metavar="DIR", help="the scapes root (default: RPP_SCAPES)")
        ap.add_argument("--pds", default=None, metavar="DIR", help="the PDS archive (default: RPP_PDS)")
        ap.add_argument("--zcam", action="store_true", help="the Navcam + Mastcam-Z block of the site (..._zcam)")
    ap.add_argument("--hazcam", default=None, choices=("front", "rear", "both", "none"),
                    help="the Hazcam frames of the block's stations: front, rear, both (the default since 0.93.1) or none")
    ap.add_argument("--force", action="store_true", help="start even if the status file says a run is alive")
    ap.add_argument("--no-echo", action="store_true", help="log to the file only")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rpp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    v = ap.add_mutually_exclusive_group()
    v.add_argument("-v", "--verbose", action="store_true", help="debug messages")
    v.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    ap.add_argument("--log", default=None, metavar="FILE", help="also write the log to this file")
    from . import __version__
    ap.add_argument("--version", action="version", version=f"rpp {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("process", help="select the PDS products of a site and process them into <WORK>/processed")
    p.add_argument("target", metavar="SITE|WORK", help="a site of the site definitions, or its WORK folder")
    _common(p)
    p.add_argument("--dry-run", action="store_true", help="select only: list what would be processed")

    p = sub.add_parser("align", help="align the images processed in a WORK folder")
    p.add_argument("work", metavar="WORK", help="the WORK folder (the one holding processed/)")
    p.add_argument("--variant", default="", help="results in <WORK>/colmap_<variant> instead of colmap/")
    _common(p, root=False)
    p.add_argument("--gui", action="store_true", help="write the COLMAP GUI files (colmap_gui.ini, open_in_colmap.bat)")
    p.add_argument("--stop-after", default=None, metavar="STEP",
                   help="stop after this step: manifest, project, features, database, match, reconstruct, health")
    p.add_argument("--status", action="store_true", help="print the folder's run status and exit")

    p = sub.add_parser("run", help="process, then align a site")
    p.add_argument("target", metavar="SITE|WORK", help="a site of the site definitions, or its WORK folder")
    p.add_argument("--variant", default="", help="results in <WORK>/colmap_<variant> instead of colmap/")
    _common(p)
    p.add_argument("--gui", action="store_true", help="write the COLMAP GUI files")
    p.add_argument("--stop-after", default=None, metavar="STEP", help="stop the alignment after this step")
    p.add_argument("--metashape", action="store_true", help="then the Metashape job (rpp metashape)")
    p.add_argument("--nerfstudio", action="store_true", help="then the Nerfstudio export (rpp nerfstudio)")

    p = sub.add_parser("sites", help="the site definitions")
    p.add_argument("what", nargs="?", default="list", choices=("list", "check", "folder"))
    p.add_argument("site", nargs="?", default=None, help="folder: the site name")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--root", default=None, metavar="DIR", help="folder: the scapes root (default: RPP_SCAPES)")
    p.add_argument("--zcam", action="store_true", help="folder: the Navcam + Mastcam-Z block")

    p = sub.add_parser("gui", help="write the viewer files (open_in_colmap.bat, open_in_metashape.bat ...) into finished projects")
    p.add_argument("works", nargs="*", help="WORK folders (default: every WORK folder under --root with a model)")
    p.add_argument("--root", default=None, metavar="DIR", help="the scapes root (default: RPP_SCAPES)")
    p.add_argument("--variant", default="", help="the project colmap_<variant> instead of colmap")

    p = sub.add_parser("cmods", help="the camera models in use")
    p.add_argument("what", nargs="?", default="list", choices=("list", "promote", "init", "ship", "status"))
    p.add_argument("paths", nargs="*", help="promote: folders or files with the new models; init: the folder to seed "
                                           "(outside git, e.g. D:\\scapes\\cmods; RPP_CMODS)")
    p.add_argument("--note", default="", help="promote: why these are the better models (goes into CHANGES.md)")
    p.add_argument("--dry-run", action="store_true", help="promote: check and show, copy nothing")

    p = sub.add_parser("mask", help="the terrain mask model")
    p.add_argument("what", nargs="?", default="list", choices=("list", "fetch", "install", "levels"))
    p.add_argument("arg", nargs="?", default=None, help="fetch: the model name; install: a .pt/.safetensors file; "
                   "levels (0.99.9): a COLMAP project folder (with masks_prob/)")
    p.add_argument("--name", default=None, help="install: the registry name")
    p.add_argument("--force", action="store_true", help="fetch: download again")
    p.add_argument("--sfm", type=float, default=None, metavar="T",
                   help="levels: masks/ (the feature extraction) at P(terrain) >= T (a margin; the next model's card: 0.9)")
    p.add_argument("--tight", type=float, default=None, metavar="T",
                   help="levels: masks_tight/ (dense fusion, mesh, panoramas) at P(terrain) >= T (the card: 0.5)")
    p.add_argument("--min-island-frac", type=float, default=None,
                   help="levels: islands under this share of the frame removed (default: the project's record, else 0)")

    p = sub.add_parser("example", help="the Mars 2020 example data (Rockytop, sols 461-464): fetch or build")
    p.add_argument("what", choices=("fetch", "build"), help="fetch: download it (from the PDS; the release zip if "
                   "that fails); build: zip it from a local PDS archive (for the release)")
    p.add_argument("--dest", default=None, metavar="DIR", help="fetch: where to put it (default ~/rpp_example)")
    p.add_argument("--source", default="pds", choices=("pds", "release"),
                   help="fetch: 0.99.5 - the PDS Imaging Node (default) or the GitHub release zip")
    p.add_argument("--force", action="store_true", help="fetch: download again")
    p.add_argument("--pds", default=None, metavar="DIR", help="build: the PDS archive (default RPP_PDS)")
    p.add_argument("--zcam", default=None, metavar="DIR", help="build: the Mastcam-Z folder (default <PDS>/../zcam)")
    p.add_argument("--out", default=".", metavar="DIR", help="build: where to write the zip")

    p = sub.add_parser("waypoints", help="the rover waypoint table in use (cached download or packaged snapshot)")
    p.add_argument("--refresh", action="store_true", help="download the current table now (every sync should)")
    p.add_argument("--mission", default="m2020", choices=("m2020", "msl"), help="m2020 (default) or msl (Curiosity)")
    p.add_argument("--path", default=None, help="the table file to use / download into (default: the cache)")

    from .pds import add_parser as _pds_parser
    _pds_parser(sub)                          # 0.99.5: rpp pds fetch
    _study_parsers(sub)                       # 0.96.0: calib, zcam-states (rpp_studies.calibration), maps (rpp_studies.sites)
    from .ibr.cli import add_parsers as _ibr_parsers
    _ibr_parsers(sub)                         # 0.96.0: mesh, depthmaps, ods (rpp.ibr)
    p = sub.add_parser("dense", help="the COLMAP dense reconstruction of a finished block - the workspace "
                       "<project>/dense (images undistorted from the native-pixel model, the rover masks for the "
                       "fusion), patch-match stereo, fusion, a mesh; dense_reconstruct.bat next to open_in_colmap.bat "
                       "does the same with the COLMAP executable")
    p.add_argument("work", help="the block's WORK folder (its colmap/ project) or the project folder itself")
    p.add_argument("--variant", default="", help="the project colmap_<variant> (default colmap)")
    p.add_argument("--prepare", action="store_true", help="only the workspace (undistorted images, masks, the configs)")
    p.add_argument("--stereo", action="store_true", help="patch-match stereo (a CUDA pycolmap: --python or GPU_PYTHON)")
    p.add_argument("--fusion", action="store_true", help="the fusion into dense/fused.ply (the masks applied)")
    p.add_argument("--mesh", action="store_true", help="a Poisson mesh of the fused cloud")
    p.add_argument("--max-image-size", type=int, default=None, help="the long side of the working images (default 2000)")
    p.add_argument("--python", default=os.environ.get("GPU_PYTHON"), help="the Python with the CUDA pycolmap for the "
                   "stereo (default GPU_PYTHON, else the alignment's GPU python: RPP_GPU_PYTHON, RPP_GPU_PY, "
                   "gpu_python.txt, the rpp_gpu conda env; "
                   "without one the stereo runs here if this pycolmap has CUDA)")
    p.add_argument("--overwrite", action="store_true", help="undistort again even if the workspace is there")
    p.add_argument("--no-bat", action="store_true", help="do not (re)write dense_reconstruct.bat (0.95.1: the batch file "
                   "calls --prepare itself)")
    p.add_argument("--maps", nargs="?", const="png", default=None, choices=("png", "jpg"),
                   help="the depth and normal maps of dense/stereo as images in dense/stereo/maps_png (png or jpg)")
    p.add_argument("--maps-kind", default="geometric", choices=("geometric", "photometric"),
                   help="--maps: the geometric maps (after the consistency check, default) or the photometric ones")
    p.add_argument("--max-depth", type=float, default=None, help="--maps: depth beyond this (m) is left out (default 100)")
    p.add_argument("--station", default=None, help="only the frames of this station (S026D0630, 26_630, sol:461): "
                   "a sub-model of them, its own workspace <project>/dense_<station>[_<sequence>]")
    p.add_argument("--cameras", default="N", help="--station: the camera families (default N)")
    p.add_argument("--sequence", nargs="+", default=None, help="--station: only these sequences (NCAM03461 ...)")
    p.add_argument("--nearby-m", type=float, default=None, metavar="M",
                   help="0.97.0, with --station: also the frames of the block's other stations within M meters (they "
                        "see the ground under this station's rover); its own workspace dense_<station>..._near<M>")
    p.add_argument("--whole-block", action="store_true",
                   help="the terrain from the whole alignment - every frame of the block of the --cameras "
                        "families (all stations, all sizes) in the workspace dense_block_<cameras>, which the block's "
                        "stations share; with --station, that station's own frames (--scale, --sequence, --images) are "
                        "recorded in dense_<station>_block/mppp_dense.json for the panorama")
    p.add_argument("--block-cameras", default=None, metavar="FAMILIES",
                   help="with --whole-block, the block's cloud from the frames of these camera families "
                        "(NFR: Navcam, front and rear Hazcam - every engineering camera, every size; default: --cameras); "
                        "the workspace is dense_block_<families>; --cameras still chooses the station's own frames")
    p.add_argument("--block-images", nargs="+", default=None, metavar="NAME",
                   help="with --whole-block, the block's cloud from these frames only (a terrain set of the "
                        "single-sites config), in the workspace dense_block_<--block-name>")
    p.add_argument("--block-name", default=None, help="the name of the --block-images set (its workspace)")
    p.add_argument("--scale", type=float, default=None, help="--station: only frames at this downsample (0.5: half-res)")
    p.add_argument("--images", nargs="+", default=None, help="only these image names (instead of --station)")
    p.add_argument("--add-images", nargs="+", default=None, metavar="NAME",
                   help="these frames of the block as well, whatever --station / --scale / --images select "
                        "(context frames from another place or sol: they add to the terrain)")
    p.add_argument("--min-angle", type=float, default=None, help="the patch-match triangulation angle (deg) a pixel "
                   "needs; default 0.4 with --station / --images (Navcam 0.424 m baseline: 61 m), COLMAP's 3 (8 m) "
                   "for a whole block")
    p.add_argument("--full-fov", action="store_true", help="undistort to the whole field of view (the fisheye's "
                   "corners kept: 10 %% of a Navcam frame, 11 %% of the rover's pixels at Van Zyl), the working size "
                   "scaled so the center keeps its resolution (~1.8x the stereo time); the default crops to the "
                   "rectangle of valid pixels")
    p.add_argument("--restereo", action="store_true", help="delete the workspace's patch-match maps before --stereo "
                   "(they are deleted anyway when they were made with other options; COLMAP skips every image whose "
                   "maps exist)")
    p.add_argument("--keep-rover", action="store_true", help="the fusion without the rover masks, into "
                   "fused_rover.ply (the rover body and deck kept, for rpp mesh / depthmaps / ods --rover); the "
                   "stereo is the same (it takes no masks), so --fusion --keep-rover alone after a run is enough")
    p.add_argument("--min-views", type=int, default=None, help="the images a fused point needs (fusion "
                   "min_num_pixels; patch-match then wants max(1, N-1) consistent sources); default 2 with --station / "
                   "--images, COLMAP's 5 for a whole block")
    p = sub.add_parser("metashape", help="a finished block as an Agisoft Metashape project - cameras (calibration "
                       "and poses fixed), COLMAP's tie points and the masks, the photos linked from images/: writes "
                       "<project>/metashape/ (job, Bundler tie points, the build script, make_metashape_project.bat); "
                       "Metashape Professional builds <block>.psx from it")
    p.add_argument("work", help="the block's WORK folder (its colmap/ project) or the project folder itself")
    p.add_argument("--variant", default="", help="the project colmap_<variant> (default colmap)")
    p.add_argument("--model", default=None, help="a native-pixel COLMAP model (default <project>/gui_native/sparse, "
                   "written from the block's model if missing)")
    p.add_argument("--out", default=None, help="the folder (default <project>/metashape)")
    p.add_argument("--max-points", type=int, default=None, help="at most this many tie points (a random subset; "
                   "default all)")
    p.add_argument("--metashape", default=rpp_env.get("RPP_METASHAPE"),
                   help="metashape.exe (default RPP_METASHAPE, else the standard Metashape Pro install)")
    p.add_argument("--run", action="store_true", help="build the project now (runs make_metashape_project.bat)")
    p.add_argument("--open", action="store_true", help="convert and build when missing or older than the "
                   "model, then open the .psx in Metashape (what open_in_metashape.bat runs)")
    p = sub.add_parser("nerfstudio", help="a finished block for Nerfstudio (splatfacto): every frame "
                       "undistorted through COLMAP's model into one pinhole size, the masks likewise, transforms.json "
                       "and the tie points as sparse_pc.ply, in <project>/nerfstudio")
    p.add_argument("work", help="the block's WORK folder (its colmap/ project) or the project folder itself")
    p.add_argument("--variant", default="", help="the project colmap_<variant> (default colmap)")
    p.add_argument("--model", default=None, help="a native-pixel COLMAP model (default <project>/gui_native/sparse)")
    p.add_argument("--out", default=None, help="the folder (default <project>/nerfstudio)")
    p.add_argument("--size", default="2048x1536", help="the frame size WxH (default 2048x1536)")
    p.add_argument("--zcam-clip", default="22,16", help="Mastcam-Z columns left out: left,right (default 22,16)")
    p.add_argument("--families", default=None, help="only these camera families, e.g. N or N,F,R,Z (default all)")
    p.add_argument("--jpg", action="store_true", help="JPEG images (quality 95) instead of PNG")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--overwrite", action="store_true", help="write every frame again (default: keep those there)")
    sub.add_parser("doctor", help="what is set, what is missing: environment variables, dependencies, models, waypoints")
    sub.add_parser("version", help="the RPP version")
    return ap


# ------------------------------------------------------------------------------------------------------ helpers
def setup_logging(a: argparse.Namespace) -> None:
    level = logging.DEBUG if getattr(a, "verbose", False) else logging.WARNING if getattr(a, "quiet", False) else logging.INFO
    root = logging.getLogger("rpp")
    root.setLevel(level)
    if not any(getattr(h, "_rpp_cli", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        h._rpp_cli = True                                                # type: ignore[attr-defined]
        root.addHandler(h)
    if getattr(a, "log", None):
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)      # 0.99.3: a first joint on a root has no camera_analysis yet
        f = logging.FileHandler(a.log, encoding="utf-8")
        f.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(f)


def _study_parsers(sub) -> None:
    """0.96.0: the study commands - ``calib``, ``zcam-states`` (:mod:`rpp_studies.calibration.cli`) and ``maps``
    (:mod:`rpp_studies.sites.cli`) - are defined by the studies and only forwarded here; without the studies
    installed the commands say so."""
    for mod, names in (("rpp_studies.calibration.cli", ("calib", "zcam-states")), ("rpp_studies.sites.cli", ("maps",))):
        try:
            __import__(mod, fromlist=["add_parsers"]).add_parsers(sub)
        except ImportError:                                            # the studies come with a source checkout
            for n in names:
                q = sub.add_parser(n, help="a research study: needs a source checkout (src/rpp_studies)")
                q.add_argument("_rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)   # so the command reaches its message
                q.set_defaults(_missing_study=mod)


def _study_command(name: str):
    """The command function of a study command, imported when the command runs."""
    mod = {"calib": "rpp_studies.calibration.cli", "zcam-states": "rpp_studies.calibration.cli",
           "maps": "rpp_studies.sites.cli"}[name]

    def run(a):
        if getattr(a, "_missing_study", None):
            print(f"ERROR: {name} needs {a._missing_study}: the studies come with a source checkout "
                  f"(git clone https://github.com/ctate7163/rpp, then pip install -e .)", file=sys.stderr)
            return 2
        return __import__(mod, fromlist=["COMMANDS"]).COMMANDS[name](a)
    return run


def _settings(a: argparse.Namespace, target: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
    """The settings of a ``process`` / ``align`` / ``run`` call: the site's, the WORK folder's, ``--settings``,
    ``--set``, then the options."""
    from .pipeline import Settings, load_settings
    from .runner import parse_set
    from .waypoints.sites import parse_work_folder
    sets = parse_set(a.set)
    base = Settings()
    if getattr(a, "root", None):
        base = base.replace(scapes_root=Path(a.root))
    if getattr(a, "pds", None):
        base = base.replace(pds_dir=Path(a.pds))
    if getattr(a, "zcam", False):
        base = base.replace(include_zcam=True)
    if getattr(a, "hazcam", None):
        sets = dict(sets, hazcams=a.hazcam)
    site, work = None, None
    if target is not None:
        if any(c in target for c in "/\\") or Path(target).is_dir():
            work = Path(target).expanduser().resolve()
            site = parse_work_folder(work)[0] or work.name
        else:
            site = target
    if work is None:
        from .example import example_defaults
        dflt = example_defaults(site, base.pds_dir, base.scapes_root)
        if dflt:
            print("example: " + ", ".join(f"{k} {v}" for k, v in dflt.items()), file=sys.stderr)
            base = base.replace(**dflt)
    s = load_settings(work, site, a.settings, sets, a.sites_file, base=base)
    return s.replace(**(extra or {}))


def _run(fn, work: Path, s, a: argparse.Namespace, stage: str, key: str, **kw: Any) -> int:
    """A logged run: the log file echoes to the console, so the console handler steps aside meanwhile."""
    from .runner import RunAlive, run_logged
    root = logging.getLogger("rpp")
    console = [h for h in root.handlers if getattr(h, "_rpp_cli", False)]
    for h in console:
        root.removeHandler(h)
    try:
        st = run_logged(fn, work, s, stage=stage, key=key, variant=s.variant, echo=not a.no_echo, force=a.force,
                        **kw)
    except RunAlive as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    finally:
        for h in console:
            root.addHandler(h)
    if st.get("state") != "finished":
        print(f"FAILED: {st.get('last_error')} (log {st.get('log')})", file=sys.stderr)
        return 1
    r = st.get("result") or {}
    if isinstance(r, dict) and r:
        print(json.dumps(r, indent=1, default=str))
    return 0


# --------------------------------------------------------------------------------------------------- subcommands
def cmd_process(a: argparse.Namespace) -> int:
    from .pipeline import apply_exclusions, process_key, process_site, product_summary, resolve_site, select_products
    s = _settings(a, a.target)
    st, s = resolve_site(a.target, s)
    if a.dry_run:
        paths, sel = select_products(st, s)
        n0 = len(paths)
        paths = apply_exclusions(paths, s, sel)        # 2 Nov 2026: as processing does (EXCLUDE, zcam_include, the list)
        print(f"{st.name} (sols {st.sols_text}) -> {st.work / 'processed'}: {len(paths)} products would be processed"
              + (f" ({n0 - len(paths)} of the {n0} selected left out by the exclusions)" if n0 != len(paths) else ""))
        for p in paths[:20]:
            print("  ", p.name)
        if len(paths) > 20:
            print(f"   ... {len(paths) - 20} more")
        for line in product_summary(paths):
            print(line)
        print("(the selection rules - sky, LMST window, saturation, exposure - leave out more when processing)")
        return 0
    return _run(process_site, st.work, s, a, "process", process_key(s), info={"site": st.name})


def cmd_align(a: argparse.Namespace) -> int:
    from .pipeline import STEPS, align, resolve_site, run_key
    work = Path(a.work).expanduser().resolve()
    if a.status:
        return _show_status(work)
    if not (work / "processed").is_dir():
        print(f"ERROR: {work} has no processed/ folder: process the images first (rpp process)", file=sys.stderr)
        return 2
    if a.stop_after and a.stop_after not in STEPS:
        print(f"ERROR: --stop-after {a.stop_after!r}: use one of {', '.join(STEPS)}", file=sys.stderr)
        return 2
    s = _settings(a, str(work), {"variant": a.variant, "source": "processed", **({"gui_files": True} if a.gui else {})})
    st, s = resolve_site(work, s)
    return _run(align, work, s, a, "align", run_key(s), stop_after=a.stop_after, info={"site": st.name})


def cmd_run(a: argparse.Namespace) -> int:
    from .pipeline import STEPS, resolve_site, run_key, run_site
    if a.stop_after and a.stop_after not in STEPS:
        print(f"ERROR: --stop-after {a.stop_after!r}: use one of {', '.join(STEPS)}", file=sys.stderr)
        return 2
    s = _settings(a, a.target, {"variant": a.variant, "source": "pds", **({"gui_files": True} if a.gui else {})})
    st, s = resolve_site(a.target, s)
    rc = _run(run_site, st.work, s, a, "run", run_key(s), stop_after=a.stop_after, info={"site": st.name})
    if rc in (0, 1) and (getattr(a, "metashape", False) or getattr(a, "nerfstudio", False)):
        d = _project_dir(str(st.work), a.variant)          # 0.95.2: the exports of the finished block
        if d is None:
            print(f"no project at {st.work}: no exports", file=sys.stderr)
            return rc or 1
        if a.metashape:
            cmd_metashape(argparse.Namespace(work=str(d), variant="", out=None, model=None, max_points=None,
                                             metashape=rpp_env.get("RPP_METASHAPE"),
                                             run=bool(rpp_env.get("RPP_METASHAPE_RUN"))))
        if a.nerfstudio:
            cmd_nerfstudio(argparse.Namespace(work=str(d), variant="", out=None, model=None, size="2048x1536",
                                              zcam_clip="22,16", families=None, jpg=False, threads=8,
                                              overwrite=True))     # a new alignment: every frame resampled again
    return rc


def _show_status(work: Path) -> int:
    from .runner import STATUS_FILE, read_status
    st = read_status(work / STATUS_FILE)
    if st is None:
        print(f"{work}: no run recorded ({STATUS_FILE} missing)")
        return 1
    print(json.dumps({k: st.get(k) for k in ("state", "stage", "variant", "started", "step_started", "ended",
                                              "heartbeat", "age_s", "alive", "last_error", "result", "log")},
                     indent=1, default=str))
    return 0


def cmd_sites(a: argparse.Namespace) -> int:
    from .waypoints.sites import SITES_FILE, load_site_table, site_zooms, sol_ranges, validate_site_table, work_folder
    f = Path(a.sites_file) if a.sites_file else SITES_FILE
    if a.what == "check":
        from .pipeline import setting_names
        err, warn = validate_site_table(f, set(setting_names()) | set(setting_names(upper=True)))
        for e in err:
            print(f"  ERROR    {e}")
        for w in warn:
            print(f"  warning  {w}")
        print(f"{f}: {len(err)} error(s), {len(warn)} warning(s)")
        return 1 if err else 0
    table = load_site_table(f)
    if a.what == "folder":
        if not a.site:
            print("ERROR: rpp sites folder SITE", file=sys.stderr)
            return 2
        if a.site not in table["sites"]:
            print(f"ERROR: site {a.site!r} is not in {f}", file=sys.stderr)
            return 2
        root = a.root or rpp_env.get("RPP_SCAPES") or "."
        print(work_folder(root, a.site, a.zcam, table["sites"][a.site]["sols"]))
        return 0
    print(f"site definitions: {f} ({len(table['sites'])} sites)")
    for name, v in table["sites"].items():
        rs = sol_ranges(v["sols"])
        sols = ", ".join(f"{x}-{y}" for x, y in rs)
        zooms = ("zcam " + ",".join(map(str, site_zooms(v)))) if site_zooms(v) else ""
        extra = f"  settings {v['settings']}" if v.get("settings") else ""
        print(f"  {name:24s} sols {sols:24s} {zooms:14s}{extra}")
    groups = table.get("groups") or {}
    if groups:
        print("groups: " + ", ".join(f"{g} ({len(m)})" for g, m in groups.items()))
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    """``gui``: ``open_in_colmap.bat``, ``open_in_metashape.bat`` (0.98.18; ``open_in_colmap_fullres.bat`` removed)
    and ``colmap_gui.ini`` for finished projects.  The project files are not rewritten."""
    from .sfm.database import write_gui_project
    from .sfm.project import SfmProject
    from .sfm.workdir import project_dir
    works = [Path(w) for w in a.works]
    if not works:
        root = Path(a.root or rpp_env.get("RPP_SCAPES") or "")
        if not str(root) or not root.is_dir():
            print("ERROR: give WORK folders or --root (or RPP_SCAPES)", file=sys.stderr)
            return 1
        works = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("mars2020_sol_"))
    n = 0
    for w in works:
        pd = project_dir(w, a.variant)
        if not (pd / "project.json").is_file():
            continue
        proj = SfmProject.load(pd)
        model = "cahv_ba" if (pd / "sparse" / "cahv_ba").is_dir() else None
        out = write_gui_project(proj, model=model)        # 0.98.18: project.json left as it is
        n += 1
        print(f"  {w.name}: {Path(out.get('bat', pd)).name if isinstance(out, dict) else 'written'}")
    print(f"{n} project(s) with COLMAP GUI files")
    return 0


def cmd_cmods(a: argparse.Namespace) -> int:
    from .calib import models as M
    if a.what == "status":                                   # 0.95.0: the folder in use against the package copy
        st = M.status()
        print(f"camera models in use: {st['folder']}" + (" (RPP_CMODS)" if st["override"] else " (the package copy)"))
        print(f"  fingerprint {st['fingerprint']}; package copy {st['package']} fingerprint {st['package_fingerprint']}"
              + (" - the same" if st["same"] else ""))
        if st["missing"]:
            print(f"  MISSING from the folder: {', '.join(st['missing'])} (rpp cmods init {st['folder']} seeds them)")
        if st["differs"]:
            print(f"  differ from the package copy: {', '.join(st['differs'])} (rpp cmods ship copies them in for a release)")
        if st["extra"]:
            print(f"  not in the package copy: {', '.join(st['extra'])}")
        return 0 if not st["missing"] else 1
    if a.what == "init":
        if not a.paths:
            print("ERROR: rpp cmods init DIR - the folder to seed with the package's camera models", file=sys.stderr)
            return 1
        res = M.init_folder(a.paths[0])
        print(f"{res['folder']}: {len(res['copied'])} file(s) copied from the package copy, {len(res['kept'])} kept")
        try:                                                 # the batch files read it (rpp_env.bat)
            from .paths import source_root
            env = source_root() / "scripts" / "windows" / "rpp_cmods.txt"
            env.write_text(str(Path(res["folder"]).resolve()) + "\n", encoding="utf-8")
            print(f"written {env} - the batch files set RPP_CMODS from it; for a shell: setx RPP_CMODS {res['folder']}")
        except Exception as e:                               # noqa: BLE001 - not a checkout
            print(f"(no scripts/windows/rpp_cmods.txt written: {e}); set RPP_CMODS={res['folder']} in the environment")
        return 0
    if a.what == "ship":
        res = M.ship(note=a.note)
        if res.get("note"):
            print(res["note"])
            return 0
        print(f"{len(res['changed'])} file(s) changed in {res['to']} from {res['from']}: {', '.join(res['changed']) or '-'}; "
              f"commit src/rpp/data/cmods (the release's shipped models)")
        return 0
    if a.what == "list" or not a.paths:
        print("\n".join(M.describe()))
        form = M.check_form()
        for w in form["warnings"]:
            print(f"  warning: {w}")
        for e in form["errors"]:
            print(f"  ERROR: {e}")
        return 0 if form["ok"] else 1
    try:
        todo = M.collect(a.paths)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not todo:
        print(f"ERROR: no camera model files found in {a.paths}", file=sys.stderr)
        return 1
    for name, f in todo.items():
        try:
            print(f"  {name:40s} <- {f}  ({M.check_file(name, f)})")
        except Exception as e:                                         # noqa: BLE001
            print(f"ERROR: {f} does not load as {name}: {e}; nothing copied", file=sys.stderr)
            return 1
    if a.dry_run:
        return 0
    res = M.promote(a.paths, note=a.note)
    from .paths import package_cmods_dir
    inside = Path(res["folder"]).resolve() == package_cmods_dir().resolve()
    print(f"{len(res['files'])} file(s) now in {res['folder']} ({res['stamp']}); "
          + ("commit src/rpp/data/cmods" if inside else "outside git (RPP_CMODS); rpp cmods ship copies them into the package for a release"))
    return 0


def _pds_command(a: argparse.Namespace) -> int:
    from .pds import cmd_pds
    return cmd_pds(a)


def cmd_example(a: argparse.Namespace) -> int:
    """``example fetch`` / ``example build``: the Mars 2020 example data set (:mod:`rpp.example`)."""
    from . import example as E
    if a.what == "fetch":
        d = E.fetch_example(a.dest, force=a.force, source=getattr(a, "source", "pds"))
        print(f"example data in {d}\nnext: rpp run rockytop_example --zcam"
              + ("" if Path(d) == E.default_dest() else f" --pds {d}"))
        return 0
    pds = a.pds or rpp_env.get("RPP_PDS")
    if not pds:
        print("ERROR: --pds (or RPP_PDS) is required", file=sys.stderr)
        return 2
    zc = a.zcam or str(Path(pds).parent / "zcam")
    man = E.build_example(pds, zc, a.out)
    print(f"{man['zip']}: {man['zip_bytes'] / 1e6:.0f} MB ({man['counts']}), SHA-256 {man['zip_sha256']}\n"
          f"  upload it and {Path(man['zip']).stem}.sha256 as assets of the GitHub release '{E.EXAMPLE_TAG}'")
    return 0


def cmd_mask(a: argparse.Namespace) -> int:
    from .mask import hub
    if a.what == "fetch":
        print(hub.fetch_model(a.arg, a.force))
        return 0
    if a.what == "install":
        if not a.arg:
            print("ERROR: rpp mask install FILE [--name NAME]", file=sys.stderr)
            return 2
        print(hub.install_model(a.arg, a.name))
        return 0
    if a.what == "levels":
        return _mask_levels(a)
    reg = hub.load_registry()
    for n, ent in reg["models"].items():
        p = hub.model_path(n)
        print(f"{n}{' (default)' if n == reg['default'] else ''}: {ent['file']}  sha256 {str(ent.get('sha256', '?'))[:16]}  "
              f"{'cached' if p.is_file() else 'not cached'} ({p})")
    return 0


def _mask_levels(a: argparse.Namespace) -> int:
    """0.99.9: a step's threshold changed from the stored probability (masks_prob/), without the network."""
    from .mask_levels import PROB_DIR, write_step_masks
    root = Path(a.arg or ".")
    pd = root / PROB_DIR
    if not pd.is_dir():
        print(f"ERROR: no {PROB_DIR}/ in {root}: the block was processed without probability maps (a mask model "
              f"before 0.99.9, or export.write_mask_probability off)", file=sys.stderr)
        return 2
    thr = {k: v for k, v in (("sfm", a.sfm), ("tight", a.tight)) if v is not None}
    if not thr:
        print("ERROR: rpp mask levels PROJECT --sfm T and/or --tight T", file=sys.stderr)
        return 2
    rec = pd / "step_masks.json"
    old = json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else {}
    isl = a.min_island_frac if a.min_island_frac is not None else float(old.get("min_island_frac") or 0.0)
    names = sorted(f.name[:-4] for f in pd.glob("*.png"))
    rep = write_step_masks(root, names, thr, isl, which=tuple(thr))
    if old.get("thresholds"):                                  # keep the step that was not asked about
        merged = dict(old["thresholds"], **thr)
        rec.write_text(json.dumps({"thresholds": merged, "min_island_frac": isl, "written": rep["written"]}, indent=1),
                       encoding="utf-8")
    print(f"{len(names)} probability maps: written {rep['written']}, unchanged {rep['kept']}"
          + (f", missing {len(rep['missing'])}" if rep["missing"] else ""))
    if rep["written"].get("sfm"):
        print("masks/ changed: it takes effect when the block's features are extracted again (rpp align)")
    return 0


def cmd_waypoints(a: argparse.Namespace) -> int:
    from .waypoints import load_waypoints
    from .waypoints.table import cached_path, mission_url, waypoints_age_days
    mission = getattr(a, "mission", "m2020") or "m2020"
    path = getattr(a, "path", None)
    w = load_waypoints(path=path, refresh=bool(a.refresh), mission=mission)
    src = w["_rpp_source"]
    last = max(w["features"], key=lambda f: (int((f.get("properties", f)).get("sol") or 0)))
    pl = last.get("properties", last)
    age = waypoints_age_days(src["path"])
    print(f"{src['n_features']} waypoints from {src['path']} ({'packaged snapshot' if src['packaged_snapshot'] else 'downloaded'}, "
          f"file time {src['file_time']}{'' if age is None else f', {age:.1f} days old'})")
    print(f"last waypoint: sol {pl.get('sol')} site {pl.get('site')} drive {pl.get('drive')}")
    if src["packaged_snapshot"]:
        print(f"no downloaded copy at {cached_path(mission)}: rpp waypoints --refresh downloads {mission_url(mission)}")
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    """What is set and what is missing: the environment, the dependencies, the camera models, the waypoint table."""
    import importlib
    from . import __version__
    from .calib import check_form, models_fingerprint
    from .paths import cmods_dir, source_root
    from .waypoints.table import waypoints_age_days
    lines = [f"rpp {__version__}", f"python {sys.version.split()[0]} ({sys.executable})",
             f"source checkout: {source_root() or 'no (installed package)'}" + _git_head(source_root())]
    try:
        import rpp_studies
        lines.append(f"studies          {Path(rpp_studies.__file__).parent}")
    except Exception as e:                                                # noqa: BLE001
        lines.append(f"studies          not installed ({type(e).__name__}) - not needed for the pipeline; the calibration, "
                     "error and mask-training studies come with a source checkout (src/rpp_studies)")
    missing = 0
    no_pipeline: List[str] = []
    for var, what in (("RPP_SCAPES", "the scapes root (WORK folders)"), ("RPP_PDS", "the PDS archive"),
                      ("RPP_GPU_PYTHON", "the GPU python for SIFT (optional)"), ("COLMAP_BAT", "COLMAP (optional, GUI)")):
        v = rpp_env.get(var) if var.startswith(rpp_env.PREFIX) else os.environ.get(var)
        ok = v and Path(v).exists()
        if not v:
            txt = "not set"
            if var in ("RPP_SCAPES", "RPP_PDS"):
                missing += 1
        else:
            txt = f"{v}" + ("" if ok else "  (MISSING on disk)")
            if var.startswith(rpp_env.PREFIX) and rpp_env.which(var) != var:
                txt += f"  (from {rpp_env.which(var)}, the name before 0.99)"
        lines.append(f"{var:<16} {txt}   - {what}")
    old = sorted(rpp_env.legacy_only())
    if old:                                   # 0.99.0: MPPP_* still works; say which ones a PC still sets
        lines.append(f"old names        {', '.join(old)} - read as RPP_* (set the RPP_* names to silence this)")
    for mod, what in (("pycolmap", "the COLMAP pipeline"), ("pyceres", "the weighted bundle adjustment"),
                      ("torch", "LightGlue / mask inference"), ("kornia", "SIFT + LightGlue matching"),
                      ("matplotlib", "figures"), ("rasterio", "1 m DTM map backgrounds (optional)"),
                      ("adjustText", "map labels (optional)")):
        try:
            m = importlib.import_module(mod)
            lines.append(f"{mod:<16} {getattr(m, '__version__', 'ok')}   - {what}")
        except Exception as e:                                        # noqa: BLE001
            lines.append(f"{mod:<16} MISSING ({type(e).__name__})   - {what}")
            if "optional" not in what and mod != "matplotlib":
                no_pipeline.append(mod)
    fm = check_form()
    lines.append(f"camera models    {cmods_dir()} fingerprint {models_fingerprint()} form {'ok' if fm['ok'] else 'NOT OK: ' + '; '.join(fm['errors'])}")
    age = waypoints_age_days()
    lines.append(f"waypoints        {'packaged snapshot (rpp waypoints --refresh)' if age is None else f'downloaded copy {age:.1f} days old'}")
    lines += _live_runs(rpp_env.get("RPP_SCAPES"))
    # 1.0: what to do next, for someone who has just installed RPP
    if no_pipeline:
        lines.append(f"next             install what rpp run needs ({', '.join(no_pipeline)} missing): "
                     'pip install "rover-photogrammetry[pipeline]"')
    elif missing:
        lines.append("next             try the example (it needs no settings): rpp example fetch, then "
                     "rpp run rockytop_example --zcam; for other sites set RPP_PDS and RPP_SCAPES")
    print("\n".join(lines))
    return 1 if missing else 0


def _git_head(root: Optional[Path]) -> str:
    """`` @ <short hash> <subject>`` of a source checkout, or nothing (0.96.0: which code a tree holds, at a glance)."""
    if root is None or not (root / ".git").exists():
        return ""
    try:
        r = subprocess.run(["git", "-C", str(root), "log", "--oneline", "-1"], capture_output=True, text=True, timeout=20)
        return f" @ {r.stdout.strip()}" if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:                                                     # noqa: BLE001
        return ""


def _live_runs(scapes: Optional[str]) -> List[str]:
    """0.96.0: what runs on the scapes root right now - the pipeline lock and the freeze drivers' status files
    (pid alive and heartbeat fresh), so a sync or a second start does not walk into a run."""
    if not scapes:
        return []
    from .runner import read_status
    d = Path(scapes) / "mppp_pipeline"
    out = []
    for name, label in (("pipeline_status.json", "pipeline"), ("freeze_status.json", "freeze (0.96)"),
                        ("freeze2_status.json", "freeze-2"), ("freeze_chain_status.json", "freeze chain")):
        st = read_status(d / name)
        if not st:
            continue
        cur = next((k for k, v in (st.get("steps") or {}).items() if isinstance(v, dict) and v.get("state") == "running"), None)
        where = st.get("stage") or cur or "-"
        out.append(f"{label:<16} {'RUNNING' if st.get('alive') else st.get('state', '?')} (pid {st.get('pid')}, "
                   f"heartbeat {st.get('heartbeat')}, {where})")
    return out or [f"runs on {scapes}: none"]


def _project_dir(work: str, variant: str = "") -> Optional[Path]:
    w = Path(work)
    d = w if (w / "project.json").is_file() else w / ("colmap" + (f"_{variant}" if variant else ""))
    return d if (d / "project.json").is_file() else None


def cmd_nerfstudio(a: argparse.Namespace) -> int:
    """``nerfstudio``: :mod:`rpp.sfm.nerfstudio` on one block."""
    from .sfm.nerfstudio import write_nerfstudio
    from .sfm.project import SfmProject
    d = _project_dir(a.work, a.variant)
    if d is None:
        print(f"nerfstudio: no project at {a.work}", file=sys.stderr)
        return 2
    W, H = (int(v) for v in str(a.size).lower().split("x"))
    clip = tuple(int(v) for v in str(a.zcam_clip).split(","))
    res = write_nerfstudio(SfmProject.load(d), out_dir=a.out, model_dir=a.model, size=(W, H), zcam_clip=clip,
                           fmt="jpg" if a.jpg else "png", threads=a.threads, overwrite=a.overwrite,
                           families=a.families.split(",") if a.families else None)
    print(f"{res['frames']} frames at {W}x{H} ({res['cameras']} cameras), {res['points']} tie points -> {res['folder']}"
          + (f"; {len(res['errors'])} not written: {res['errors'][:3]}" if res["errors"] else ""))
    print(f"ns-train splatfacto --data \"{res['folder']}\" nerfstudio-data")
    return 0 if res["frames"] else 1


def cmd_metashape(a: argparse.Namespace) -> int:
    """``metashape``: :mod:`rpp.sfm.metashape` on one block."""
    from .sfm.metashape import write_metashape_job
    from .sfm.project import SfmProject
    w = Path(a.work)
    proj_dir = w if (w / "project.json").is_file() else w / ("colmap" + (f"_{a.variant}" if a.variant else ""))
    if not (proj_dir / "project.json").is_file():
        print(f"metashape: no project at {proj_dir}", file=sys.stderr)
        return 2
    if getattr(a, "open", False):
        from .sfm.metashape import open_in_metashape
        try:
            done = open_in_metashape(proj_dir, metashape_exe=a.metashape, max_points=a.max_points)
        except (OSError, RuntimeError) as e:
            print(f"metashape: {e}", file=sys.stderr)
            return 1
        print(("converted, " if done["converted"] else "") + ("built, " if done["built"] else "")
              + f"opening {done['psx']}")
        return 0
    res = write_metashape_job(SfmProject.load(proj_dir), out_dir=a.out, model_dir=a.model, max_points=a.max_points,
                              metashape_exe=a.metashape)
    tp = res["tie_points"]
    print(f"{res['images']} images, {res['sensors']} sensors (conversion error at most "
          f"{res['conversion_max_error_px']:.1e} px" + (f"; fitted, not exact: {', '.join(res['inexact_sensors'])}"
                                                        if res["inexact_sensors"] else "") + f"), {res['masks']} masks, "
          f"{tp['points']} tie points with {tp['observations']} observations -> {res['folder']}")
    print(f"Metashape project: {res['psx']} (made by {res['bat']})")
    if a.run:
        import subprocess
        return subprocess.call(["cmd", "/c", res["bat"]]) if os.name == "nt" else 0
    return 0


FUSION_RECORD = "rpp_fusion.json"
# 1.0: a fused cloud is made again when the fusion's own code changes (sfm/dense.py), not when any of the station steps'
# code does - the 0.99.52 note: "The fusion is redone on every code change, because the record carries code_key ...
# If that is too much, limit the fusion's key to sfm/dense.py".  The records of 0.99.51 to 0.99.53 hold their station
# code key (rpp.ibr.single_sites.code_key) instead (0.99.53's too); their sfm/dense.py is the one whose hash is DENSE_CODE_0P99, so
# their clouds count as this code's while dense.py is unchanged.
FUSION_CODES_0P99 = ("5aa9d5f18a9a6fdc", "74faa498abf96418", "0c917caf3b0a008e",   # 0.99.51, 0.99.52 (in progress),
                     "5f46dcfc18415f4b")                                          # 0.99.52, 0.99.53
DENSE_CODE_0P99 = "4e328c4415a18d58"


def fusion_code() -> str:
    """A hash of the fusion's code, ``sfm/dense.py``."""
    import hashlib
    f = Path(__file__).resolve().parent / "sfm" / "dense.py"
    return hashlib.sha1(b"dense.py" + f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def _fusion_record(ws, so, fo, size, a, write: bool = False):
    """0.99.47: what a fused cloud was made from - the stereo record's text, the fusion options, rover kept or not, the
    sky mask setting, and the cloud's size and time once made.  ``write``: written next to the cloud after a fusion;
    otherwise returned without the cloud's own fields, for :func:`_fusion_cached`."""
    import hashlib
    from . import env as rpp_env
    from .sfm.dense import FUSED_ROVER_PLY, STEREO_RECORD
    ws = Path(ws)
    keep = bool(getattr(a, "keep_rover", False))
    try:
        stereo = hashlib.sha1((ws / "stereo" / STEREO_RECORD).read_bytes()).hexdigest()
    except OSError:
        stereo = None
    try:                                                         # 0.99.51: a fusion of other code is made again;
        code = fusion_code()                                     # 1.0: of other fusion code (sfm/dense.py)
    except OSError:
        code = None
    rec = {"stereo_record": stereo, "code": code, "fusion_options": {k: fo[k] for k in sorted(fo)}, "keep_rover": keep,
           "sky": str(rpp_env.get("RPP_SKY_MASK") or "1").strip().lower(), "max_image_size": int(size or 0)}
    ply = ws / (FUSED_ROVER_PLY if keep else "fused.ply")
    if write and ply.is_file() and stereo:
        st = ply.stat()
        (ws / (FUSION_RECORD if not keep else "rpp_fusion_rover.json")).write_text(json.dumps(
            dict(rec, ply=ply.name, ply_bytes=st.st_size, ply_mtime=int(st.st_mtime)), indent=1, default=str), encoding="utf-8")
    return rec, ply


def _fusion_cached(ws, so, fo, size, a) -> bool:
    """True when the workspace's fused cloud was made by a fusion of these stereo maps with these options, and is the
    very file that fusion wrote (size and time).  A cloud without the record (before 0.99.47) is fused once again."""
    rec, ply = _fusion_record(ws, so, fo, size, a)
    f = Path(ws) / (FUSION_RECORD if not rec["keep_rover"] else "rpp_fusion_rover.json")
    try:
        st = ply.stat()
    except OSError:
        return False
    if not f.is_file():                                          # 0.99.51: no record (a cloud of 0.99.50 or before,
        return False                                             # or of other code): fused again
    try:
        old = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    want = json.loads(json.dumps(rec, default=str))
    same = {k: old.get(k) for k in rec if k != "code"} == {k: v for k, v in want.items() if k != "code"}
    code_ok = old.get("code") == want.get("code") or (old.get("code") in FUSION_CODES_0P99 and want.get("code") == DENSE_CODE_0P99)
    same = same and code_ok
    return bool(rec["stereo_record"]) and same and old.get("ply_bytes") == st.st_size and old.get("ply_mtime") == int(st.st_mtime)


def cmd_dense(a: argparse.Namespace) -> int:
    """``dense``: :mod:`rpp.sfm.dense` on one block."""
    from .sfm.dense import DENSE_MAX_IMAGE_SIZE, prepare_dense, run_dense, write_dense_bat
    from .sfm.project import SfmProject
    w = Path(a.work)
    proj_dir = w if (w / "project.json").is_file() else w / ("colmap" + (f"_{a.variant}" if a.variant else ""))
    if not (proj_dir / "project.json").is_file():
        print(f"dense: no project at {proj_dir}", file=sys.stderr)
        return 2
    proj = SfmProject.load(proj_dir)
    size = int(a.max_image_size or DENSE_MAX_IMAGE_SIZE)
    from .sfm.dense import DENSE_DIR, subset_images, subset_label
    subset, ws = None, proj_dir / DENSE_DIR
    stub = None
    if getattr(a, "whole_block", False):                         # 0.98.14: one cloud for the block, shared by its stations
        fam = set((getattr(a, "block_cameras", None) or a.cameras or "N").upper())        # 0.99.43: --block-cameras
        subset = sorted(r["name"] for r in proj.images if str(r.get("camera_group") or r.get("instrument") or "?")[:1] in fam)
        if not subset:
            print(f"dense --whole-block: no frames of the families {''.join(sorted(fam))!r} in {proj_dir}", file=sys.stderr)
            return 2
        ws = proj_dir / f"{DENSE_DIR}_block_{''.join(sorted(fam))}"
        pick = [n if str(n).lower().endswith(".png") else f"{n}.png" for n in (getattr(a, "block_images", None) or [])]
        if pick:                                                 # 0.98.19: a named set of the block's frames
            known = set(subset)
            gone = [n for n in pick if n not in known]
            if gone:
                print(f"dense --block-images: {len(gone)} not among the block's {a.cameras} frames, left out: {gone[:3]}",
                      file=sys.stderr)
            subset = sorted(set(pick) & known)
            if not subset:
                return 2
            ws = proj_dir / f"{DENSE_DIR}_block_{a.block_name or 'set'}"
        if getattr(a, "station", None) or getattr(a, "images", None):
            own = subset_images(proj, station=a.station, families=a.cameras, sequences=a.sequence, names=a.images, scale=a.scale)
            if not own:
                print(f"dense: no frames for --station {a.station} --cameras {a.cameras} --sequence {a.sequence}", file=sys.stderr)
                return 2
            stub = (proj_dir / (subset_label(a.station, a.sequence, a.images) + "_block"), own)
        print(f"dense: the whole block - {len(subset)} frames of {''.join(sorted(fam))} -> {ws}"
              + (f"; {len(stub[1])} of them are the station's own" if stub else ""))
    elif getattr(a, "station", None) or getattr(a, "images", None):
        subset = subset_images(proj, station=a.station, families=a.cameras, sequences=a.sequence, names=a.images,
                               scale=a.scale)
        if not subset:
            print(f"dense: no frames for --station {a.station} --cameras {a.cameras} --sequence {a.sequence}", file=sys.stderr)
            return 2
        ws = proj_dir / subset_label(a.station, a.sequence, a.images)
        if getattr(a, "nearby_m", None) and a.station:          # 0.97.0: the neighbors' frames, for the ground here
            from .sfm.dense import nearby_station_images
            nb = nearby_station_images(proj, a.station, a.nearby_m, families=a.cameras, scale=a.scale)
            subset = sorted(set(subset) | set(nb["names"]))
            ws = ws.with_name(ws.name + f"_near{a.nearby_m:g}")
            print(f"dense: {len(nb['names'])} frames of {len(nb['stations'])} stations within {a.nearby_m:g} m added: "
                  + (", ".join(f"{k} ({v['distance_m']:g} m, {v['images']})" for k, v in nb["stations"].items()) or "none"))
        extra = [n if str(n).lower().endswith(".png") else f"{n}.png" for n in (getattr(a, "add_images", None) or [])]
        if extra:                                                # 0.97.19: context frames
            known = {r["name"] for r in proj.images}
            gone = [n for n in extra if n not in known]
            if gone:
                print(f"dense: --add-images not in the project: {gone[:3]}", file=sys.stderr)
                return 2
            subset = sorted(set(subset) | set(extra))
            print(f"dense: {len(extra)} context frame(s) added (--add-images)")
        print(f"dense subset: {len(subset)} frames -> {ws}")
    if getattr(a, "maps", None):                             # 0.95.1: the maps as images, nothing else
        from .sfm.dense import export_dense_maps
        from .sfm.dense import MAPS_MAX_DEPTH_M
        r = export_dense_maps(ws, kind=a.maps_kind, fmt=a.maps,
                              max_depth_m=a.max_depth if a.max_depth else MAPS_MAX_DEPTH_M)
        print(f"{r['depth_maps']} depth and {r['normal_maps']} normal maps ({r['kind']}; {r['with_masks']} with rover "
              f"masks; beyond {r['max_depth_m']:g} m left out) -> {r['folder']}")
        return 0 if (r["depth_maps"] or r["normal_maps"]) else 1
    steps = [s for s, on in (("stereo", a.stereo), ("fusion", a.fusion), ("mesh", a.mesh)) if on]
    if stub is not None:                                         # the station's own frames, for rpp depthmaps / ods
        stub[0].mkdir(parents=True, exist_ok=True)
        (stub[0] / "mppp_dense.json").write_text(json.dumps(
            {"subset": stub[1], "terrain": "block", "block_workspace": ws.name, "points": str(ws / "fused.ply"),
             "note": "the station's own frames; the cloud is the block's (rpp dense --whole-block)"}, indent=1), encoding="utf-8")
    if a.prepare or not steps:
        res = prepare_dense(proj, ws, max_image_size=size, overwrite=a.overwrite, images=subset,
                            full_fov=bool(getattr(a, "full_fov", False)))
        print(f"workspace {res['workspace']}: {res['images']} images" + (f", {res['masks']['images']} masks" if res.get("masks") else ""))
        if not getattr(a, "no_bat", False) and subset is None:
            print(f"batch file: {write_dense_bat(proj, max_image_size=size)}")
        if not steps:
            return 0
    import pycolmap
    from .pipeline import _default_gpu_python
    from .sfm.gpu import run_step
    # the CUDA python the alignment itself uses (pipeline._default_gpu_python: RPP_GPU_PYTHON, RPP_GPU_PY,
    # %LOCALAPPDATA%\mppp\gpu_python.txt, the rpp_gpu conda env) when neither --python nor GPU_PYTHON names one.
    # 10 Nov on the PC: the 9 Nov fix looked at gpu_python() alone and missed RPP_GPU_PYTHON / the rpp_gpu env
    from .sfm.dense import STATION_FUSION_OPTIONS, STATION_STEREO_OPTIONS
    so = dict(STATION_STEREO_OPTIONS) if subset is not None else {}     # 10 Nov: COLMAP's 3 deg stopped Rockytop at 8 m
    fo = dict(STATION_FUSION_OPTIONS) if subset is not None else {}
    if getattr(a, "min_angle", None) is not None:
        so.update(min_triangulation_angle=min(float(a.min_angle), 1.0), filter_min_triangulation_angle=float(a.min_angle))
    if getattr(a, "min_views", None) is not None:
        so["filter_min_num_consistent"] = max(1, int(a.min_views) - 1)
        fo["min_num_pixels"] = int(a.min_views)
    if so or fo:
        print(f"dense options: stereo {so or 'COLMAP defaults'}; fusion {fo or 'COLMAP defaults'}")
    py = a.python or (None if getattr(pycolmap, "has_cuda", False) else _default_gpu_python())
    need_gpu = True
    if "stereo" in steps and not getattr(a, "restereo", False):           # 0.97.0: nothing to compute -> no GPU, no wait
        from .sfm.dense import stereo_cached
        if stereo_cached(ws, so, max_image_size=size):
            steps = [s for s in steps if s != "stereo"]
            need_gpu = False
            print("dense: the stereo maps of this workspace are there, made with these options - stereo skipped, "
                  "no wait for the GPU")
            # 0.99.47 (Christian, 19 Nov 2026: "Will it need to redo that if I stop it?"): the fusion again only when
            # its maps or options changed - a new fused.ply (its time) made the station's mesh, maps and panoramas
            # again after every restart, and the block's second station (a Z34 version) fused the same cloud again
            if "fusion" in steps and _fusion_cached(ws, so, fo, size, a):
                steps = [s for s in steps if s != "fusion"]
                print("dense: the fused cloud of these stereo maps and options is there - fusion skipped")
            if not steps:
                return 0
    if py and ("stereo" in steps or not getattr(pycolmap, "has_cuda", False)):
        print(f"dense: {', '.join(steps)} in {py}")
        res = run_step(py, "dense", proj, lock=need_gpu, steps=steps, max_image_size=size, workspace=str(ws), images=subset,
                       stereo_options=so, fusion_options=fo, keep_rover=bool(getattr(a, "keep_rover", False)),
                       restereo=bool(getattr(a, "restereo", False)), full_fov=bool(getattr(a, "full_fov", False)))
    else:
        res = run_dense(proj, workspace=ws, steps=steps, max_image_size=size, images=subset,
                        stereo_options=so, fusion_options=fo, keep_rover=bool(getattr(a, "keep_rover", False)),
                       restereo=bool(getattr(a, "restereo", False)), full_fov=bool(getattr(a, "full_fov", False)))
    for k, v in (res.get("steps") or {}).items():
        print(f"  {k}: {v}")
    if "fusion" in (res.get("steps") or {}):
        _fusion_record(ws, so, fo, size, a, write=True)
    return 0


def cmd_version(a: argparse.Namespace) -> int:
    from . import __file__ as where, __version__
    print(f"rpp {__version__} ({Path(where).parent})")
    return 0


def _ibr_command(name: str):
    def run(a: argparse.Namespace) -> int:
        from .ibr.cli import COMMANDS as IBR
        return IBR[name](a)
    run.__name__ = f"cmd_{name}"
    return run


COMMANDS = {"gui": cmd_gui, "process": cmd_process, "align": cmd_align, "run": cmd_run, "sites": cmd_sites, "cmods": cmd_cmods,
            "mask": cmd_mask, "example": cmd_example, "waypoints": cmd_waypoints, "pds": _pds_command, "maps": _study_command("maps"), "calib": _study_command("calib"), "doctor": cmd_doctor,
            "zcam-states": _study_command("zcam-states"), "dense": cmd_dense, "metashape": cmd_metashape, "nerfstudio": cmd_nerfstudio,
            "mesh": _ibr_command("mesh"), "depthmaps": _ibr_command("depthmaps"), "ods": _ibr_command("ods"),
            "single-sites": _ibr_command("single-sites"), "hf-check": _ibr_command("hf-check"),
            "version": cmd_version}



def main(argv: Optional[List[str]] = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 27 Oct: as rpp_env.bat (torch and pycolmap both bring OpenMP)
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 2
    setup_logging(a)
    try:
        return int(COMMANDS[a.cmd](a))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
