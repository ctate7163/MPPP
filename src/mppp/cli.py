"""
The ``mppp`` command: the pipeline (:mod:`mppp.pipeline`) and its bookkeeping from the shell.

::

    mppp process SITE|WORK [--root DIR] [--pds DIR] [--zcam] [--hazcam front|rear|both] [--set NAME=VALUE ...]
                 [--settings FILE] [--dry-run]
    mppp align WORK [--variant NAME] [--set ...] [--settings FILE] [--gui] [--stop-after STEP] [--force] [--status]
    mppp run SITE|WORK [...]                  process, then align
    mppp sites [list | check | folder SITE]   the site definitions (mppp/data/sites.json)
    mppp cmods [list | status | promote PATH ... --note TEXT | init DIR | ship]   the camera models in use (MPPP_CMODS)
    mppp mask [list | fetch [NAME] | install FILE]      the terrain mask model
    mppp example [fetch [--dest DIR] | build --pds DIR --zcam DIR --out DIR]   the Mars 2020 example data (Rockytop)
    mppp maps [all | zooms | figures 2 3 ...] [--out DIR] [--dpi N] [--background dtm|ortho] [--tag T] [--dems F;F] [--orthos F;F] [--waypoints refresh|auto|study]
    mppp calib navcam [--root DIR] [--group ecam_consensus | --blocks NAME=WORK ...] [--out DIR] [--promote-if-ok]
    mppp mesh WORK --station S003D1234 [--points PLY] [--colmap EXE] [--mesh-radius 60] [--far-radius 100]   0.96: one clean surface
    mppp depthmaps WORK --station ... [--cameras N] [--scale 0.5] [--photometry]   0.96: per-image depth / normal maps, the photometric cube
    mppp ods WORK --station ... [--width 8192] [--ipd 0.064]                       0.96: the stereoscopic panorama, two eye images
    mppp doctor                               what is set, what is missing (environment, dependencies, models)
    mppp version

``-v`` / ``-q`` set the logging level, ``--log FILE`` tees the log to a file.  ``process``, ``align`` and ``run``
write ``<WORK>/runs/<time>/log.txt`` and ``<WORK>/mppp_status.json`` as well (:func:`mppp.runner.run_logged`).
Every subcommand returns an exit code (0 ok, 1 failed, 2 bad arguments, 3 a run is already going on).
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

log = logging.getLogger("mppp.cli")

_SET_HELP = "a pipeline setting (mppp.pipeline.Settings field), VALUE as JSON or Python; repeatable"


def _common(ap: argparse.ArgumentParser, root: bool = True) -> None:
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help=_SET_HELP)
    ap.add_argument("--settings", default=None, metavar="FILE", help="a JSON file of settings {NAME: value}")
    ap.add_argument("--sites-file", default=None, metavar="FILE", help="site definitions (default mppp/data/sites.json)")
    if root:
        ap.add_argument("--root", default=None, metavar="DIR", help="the scapes root (default: MPPP_SCAPES)")
        ap.add_argument("--pds", default=None, metavar="DIR", help="the PDS archive (default: MPPP_PDS)")
        ap.add_argument("--zcam", action="store_true", help="the Navcam + Mastcam-Z block of the site (..._zcam)")
    ap.add_argument("--hazcam", default=None, choices=("front", "rear", "both", "none"),
                    help="the Hazcam frames of the block's stations: front, rear, both (the default since 0.93.1) or none")
    ap.add_argument("--force", action="store_true", help="start even if the status file says a run is alive")
    ap.add_argument("--no-echo", action="store_true", help="log to the file only")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mppp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    v = ap.add_mutually_exclusive_group()
    v.add_argument("-v", "--verbose", action="store_true", help="debug messages")
    v.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    ap.add_argument("--log", default=None, metavar="FILE", help="also write the log to this file")
    from . import __version__
    ap.add_argument("--version", action="version", version=f"mppp {__version__}")
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
    p.add_argument("--metashape", action="store_true", help="0.95.2: then the Metashape job (mppp metashape)")
    p.add_argument("--nerfstudio", action="store_true", help="0.95.2: then the Nerfstudio export (mppp nerfstudio)")

    p = sub.add_parser("sites", help="the site definitions")
    p.add_argument("what", nargs="?", default="list", choices=("list", "check", "folder"))
    p.add_argument("site", nargs="?", default=None, help="folder: the site name")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--root", default=None, metavar="DIR", help="folder: the scapes root (default: MPPP_SCAPES)")
    p.add_argument("--zcam", action="store_true", help="folder: the Navcam + Mastcam-Z block")

    p = sub.add_parser("gui", help="write the COLMAP GUI files (open_in_colmap.bat ...) into finished projects")
    p.add_argument("works", nargs="*", help="WORK folders (default: every WORK folder under --root with a model)")
    p.add_argument("--root", default=None, metavar="DIR", help="the scapes root (default: MPPP_SCAPES)")
    p.add_argument("--variant", default="", help="the project colmap_<variant> instead of colmap")

    p = sub.add_parser("cmods", help="the camera models in use")
    p.add_argument("what", nargs="?", default="list", choices=("list", "promote", "init", "ship", "status"))
    p.add_argument("paths", nargs="*", help="promote: folders or files with the new models; init: the folder to seed "
                                           "(outside git, e.g. D:\\scapes\\cmods; MPPP_CMODS)")
    p.add_argument("--note", default="", help="promote: why these are the better models (goes into CHANGES.md)")
    p.add_argument("--dry-run", action="store_true", help="promote: check and show, copy nothing")

    p = sub.add_parser("mask", help="the terrain mask model")
    p.add_argument("what", nargs="?", default="list", choices=("list", "fetch", "install"))
    p.add_argument("arg", nargs="?", default=None, help="fetch: the model name; install: a .pt/.safetensors file")
    p.add_argument("--name", default=None, help="install: the registry name")
    p.add_argument("--force", action="store_true", help="fetch: download again")

    p = sub.add_parser("example", help="the Mars 2020 example data (Rockytop, sols 461-464): fetch or build")
    p.add_argument("what", choices=("fetch", "build"), help="fetch: download the release asset and unpack it; "
                   "build: zip it from a local PDS archive (for the release)")
    p.add_argument("--dest", default=None, metavar="DIR", help="fetch: where to unpack (default ~/mppp_example)")
    p.add_argument("--force", action="store_true", help="fetch: download again")
    p.add_argument("--pds", default=None, metavar="DIR", help="build: the PDS archive (default MPPP_PDS)")
    p.add_argument("--zcam", default=None, metavar="DIR", help="build: the Mastcam-Z folder (default <PDS>/../zcam)")
    p.add_argument("--out", default=".", metavar="DIR", help="build: where to write the zip")

    p = sub.add_parser("waypoints", help="the Mars 2020 waypoint table in use (cached download or packaged snapshot)")
    p.add_argument("--refresh", action="store_true", help="download the current table now (every sync should)")

    _study_parsers(sub)                       # 0.96.0: calib, zcam-states (mppp_studies.calibration), maps (mppp_studies.sites)
    from .ibr.cli import add_parsers as _ibr_parsers
    _ibr_parsers(sub)                         # 0.96.0: mesh, depthmaps, ods (mppp.ibr)
    p = sub.add_parser("dense", help="0.95.0: the COLMAP dense reconstruction of a finished block - the workspace "
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
                   "stereo (default GPU_PYTHON; without it the stereo runs here if this pycolmap has CUDA)")
    p.add_argument("--overwrite", action="store_true", help="undistort again even if the workspace is there")
    p.add_argument("--no-bat", action="store_true", help="do not (re)write dense_reconstruct.bat (0.95.1: the batch file "
                   "calls --prepare itself)")
    p.add_argument("--maps", nargs="?", const="png", default=None, choices=("png", "jpg"),
                   help="0.95.1: the depth and normal maps of dense/stereo as images in dense/stereo/maps_png (png or jpg)")
    p.add_argument("--maps-kind", default="geometric", choices=("geometric", "photometric"),
                   help="--maps: the geometric maps (after the consistency check, default) or the photometric ones")
    p.add_argument("--max-depth", type=float, default=None, help="--maps: depth beyond this (m) is left out (default 100)")
    p.add_argument("--station", default=None, help="0.96: only the frames of this station (S026D0630, 26_630, sol:461): "
                   "a sub-model of them, its own workspace <project>/dense_<station>[_<sequence>]")
    p.add_argument("--cameras", default="N", help="--station: the camera families (default N)")
    p.add_argument("--sequence", nargs="+", default=None, help="--station: only these sequences (NCAM03461 ...)")
    p.add_argument("--scale", type=float, default=None, help="--station: only frames at this downsample (0.5: half-res)")
    p.add_argument("--images", nargs="+", default=None, help="0.96: only these image names (instead of --station)")
    p = sub.add_parser("metashape", help="0.95.2: a finished block as an Agisoft Metashape project - cameras (calibration "
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
    p.add_argument("--metashape", default=os.environ.get("MPPP_METASHAPE"),
                   help="metashape.exe (default MPPP_METASHAPE, else the standard Metashape Pro install)")
    p.add_argument("--run", action="store_true", help="build the project now (runs make_metashape_project.bat)")
    p = sub.add_parser("nerfstudio", help="0.95.2: a finished block for Nerfstudio (splatfacto): every frame "
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
    sub.add_parser("version", help="the MPPP version")
    return ap


# ------------------------------------------------------------------------------------------------------ helpers
def setup_logging(a: argparse.Namespace) -> None:
    level = logging.DEBUG if getattr(a, "verbose", False) else logging.WARNING if getattr(a, "quiet", False) else logging.INFO
    root = logging.getLogger("mppp")
    root.setLevel(level)
    if not any(getattr(h, "_mppp_cli", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        h._mppp_cli = True                                                # type: ignore[attr-defined]
        root.addHandler(h)
    if getattr(a, "log", None):
        f = logging.FileHandler(a.log, encoding="utf-8")
        f.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root.addHandler(f)


def _study_parsers(sub) -> None:
    """0.96.0: the study commands - ``calib``, ``zcam-states`` (:mod:`mppp_studies.calibration.cli`) and ``maps``
    (:mod:`mppp_studies.sites.cli`) - are defined by the studies and only forwarded here; without the studies
    installed the commands say so."""
    for mod, names in (("mppp_studies.calibration.cli", ("calib", "zcam-states")), ("mppp_studies.sites.cli", ("maps",))):
        try:
            __import__(mod, fromlist=["add_parsers"]).add_parsers(sub)
        except ImportError as e:                                       # pragma: no cover - the studies are optional
            for n in names:
                q = sub.add_parser(n, help=f"(the study package {mod.split('.')[0]} is not installed: {e})")
                q.set_defaults(_missing_study=mod)


def _study_command(name: str):
    """The command function of a study command, imported when the command runs."""
    mod = {"calib": "mppp_studies.calibration.cli", "zcam-states": "mppp_studies.calibration.cli",
           "maps": "mppp_studies.sites.cli"}[name]

    def run(a):
        if getattr(a, "_missing_study", None):
            print(f"ERROR: {name} needs {a._missing_study} (pip install -e . installs the studies with the package)", file=sys.stderr)
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
    s = load_settings(work, site, a.settings, sets, a.sites_file, base=base)
    return s.replace(**(extra or {}))


def _run(fn, work: Path, s, a: argparse.Namespace, stage: str, key: str, **kw: Any) -> int:
    """A logged run: the log file echoes to the console, so the console handler steps aside meanwhile."""
    from .runner import RunAlive, run_logged
    root = logging.getLogger("mppp")
    console = [h for h in root.handlers if getattr(h, "_mppp_cli", False)]
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
        print(f"ERROR: {work} has no processed/ folder: process the images first (mppp process)", file=sys.stderr)
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
                                             metashape=os.environ.get("MPPP_METASHAPE"),
                                             run=bool(os.environ.get("MPPP_METASHAPE_RUN"))))
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
            print("ERROR: mppp sites folder SITE", file=sys.stderr)
            return 2
        if a.site not in table["sites"]:
            print(f"ERROR: site {a.site!r} is not in {f}", file=sys.stderr)
            return 2
        root = a.root or os.environ.get("MPPP_SCAPES") or "."
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
    """``gui``: ``open_in_colmap.bat``, ``open_in_colmap_fullres.bat`` and ``colmap_gui.ini`` for projects aligned
    without ``--gui`` (batch runs before 0.92.2 did not write them)."""
    from .sfm.database import write_gui_project
    from .sfm.project import SfmProject
    from .sfm.workdir import project_dir
    works = [Path(w) for w in a.works]
    if not works:
        root = Path(a.root or os.environ.get("MPPP_SCAPES") or "")
        if not str(root) or not root.is_dir():
            print("ERROR: give WORK folders or --root (or MPPP_SCAPES)", file=sys.stderr)
            return 1
        works = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("mars2020_sol_"))
    n = 0
    for w in works:
        pd = project_dir(w, a.variant)
        if not (pd / "project.json").is_file():
            continue
        proj = SfmProject.load(pd)
        model = "cahv_ba" if (pd / "sparse" / "cahv_ba").is_dir() else None
        out = write_gui_project(proj, model=model)
        proj.save()
        n += 1
        print(f"  {w.name}: {Path(out.get('bat', pd)).name if isinstance(out, dict) else 'written'}")
    print(f"{n} project(s) with COLMAP GUI files")
    return 0


def cmd_cmods(a: argparse.Namespace) -> int:
    from .calib import models as M
    if a.what == "status":                                   # 0.95.0: the folder in use against the package copy
        st = M.status()
        print(f"camera models in use: {st['folder']}" + (" (MPPP_CMODS)" if st["override"] else " (the package copy)"))
        print(f"  fingerprint {st['fingerprint']}; package copy {st['package']} fingerprint {st['package_fingerprint']}"
              + (" - the same" if st["same"] else ""))
        if st["missing"]:
            print(f"  MISSING from the folder: {', '.join(st['missing'])} (mppp cmods init {st['folder']} seeds them)")
        if st["differs"]:
            print(f"  differ from the package copy: {', '.join(st['differs'])} (mppp cmods ship copies them in for a release)")
        if st["extra"]:
            print(f"  not in the package copy: {', '.join(st['extra'])}")
        return 0 if not st["missing"] else 1
    if a.what == "init":
        if not a.paths:
            print("ERROR: mppp cmods init DIR - the folder to seed with the package's camera models", file=sys.stderr)
            return 1
        res = M.init_folder(a.paths[0])
        print(f"{res['folder']}: {len(res['copied'])} file(s) copied from the package copy, {len(res['kept'])} kept")
        try:                                                 # the batch files read it (mppp_env.bat)
            from .paths import source_root
            env = source_root() / "scripts" / "windows" / "mppp_cmods.txt"
            env.write_text(str(Path(res["folder"]).resolve()) + "\n", encoding="utf-8")
            print(f"written {env} - the batch files set MPPP_CMODS from it; for a shell: setx MPPP_CMODS {res['folder']}")
        except Exception as e:                               # noqa: BLE001 - not a checkout
            print(f"(no scripts/windows/mppp_cmods.txt written: {e}); set MPPP_CMODS={res['folder']} in the environment")
        return 0
    if a.what == "ship":
        res = M.ship(note=a.note)
        if res.get("note"):
            print(res["note"])
            return 0
        print(f"{len(res['changed'])} file(s) changed in {res['to']} from {res['from']}: {', '.join(res['changed']) or '-'}; "
              f"commit src/mppp/data/cmods (the release's shipped models)")
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
          + ("commit src/mppp/data/cmods" if inside else "outside git (MPPP_CMODS); mppp cmods ship copies them into the package for a release"))
    return 0


def cmd_example(a: argparse.Namespace) -> int:
    """``example fetch`` / ``example build``: the Mars 2020 example data set (:mod:`mppp.example`)."""
    from . import example as E
    if a.what == "fetch":
        d = E.fetch_example(a.dest, force=a.force)
        print(f"example data in {d}\n  set MPPP_PDS={d} (it holds datadrive/ and zcam/)")
        return 0
    pds = a.pds or os.environ.get("MPPP_PDS")
    if not pds:
        print("ERROR: --pds (or MPPP_PDS) is required", file=sys.stderr)
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
            print("ERROR: mppp mask install FILE [--name NAME]", file=sys.stderr)
            return 2
        print(hub.install_model(a.arg, a.name))
        return 0
    reg = hub.load_registry()
    for n, ent in reg["models"].items():
        p = hub.model_path(n)
        print(f"{n}{' (default)' if n == reg['default'] else ''}: {ent['file']}  sha256 {str(ent.get('sha256', '?'))[:16]}  "
              f"{'cached' if p.is_file() else 'not cached'} ({p})")
    return 0


def cmd_waypoints(a: argparse.Namespace) -> int:
    from .waypoints import load_waypoints
    from .waypoints.table import WAYPOINTS_URL, cached_path, waypoints_age_days
    w = load_waypoints(refresh=bool(a.refresh))
    src = w["_mppp_source"]
    last = max(w["features"], key=lambda f: (int((f.get("properties", f)).get("sol") or 0)))
    pl = last.get("properties", last)
    age = waypoints_age_days()
    print(f"{src['n_features']} waypoints from {src['path']} ({'packaged snapshot' if src['packaged_snapshot'] else 'downloaded'}, "
          f"file time {src['file_time']}{'' if age is None else f', {age:.1f} days old'})")
    print(f"last waypoint: sol {pl.get('sol')} site {pl.get('site')} drive {pl.get('drive')}")
    if src["packaged_snapshot"]:
        print(f"no downloaded copy at {cached_path()}: mppp waypoints --refresh downloads {WAYPOINTS_URL}")
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    """What is set and what is missing: the environment, the dependencies, the camera models, the waypoint table."""
    import importlib
    from . import __version__
    from .calib import check_form, models_fingerprint
    from .paths import cmods_dir, source_root
    from .waypoints.table import waypoints_age_days
    lines = [f"mppp {__version__}", f"python {sys.version.split()[0]} ({sys.executable})",
             f"source checkout: {source_root() or 'no (installed package)'}" + _git_head(source_root())]
    try:
        import mppp_studies
        lines.append(f"studies          {Path(mppp_studies.__file__).parent}")
    except Exception as e:                                                # noqa: BLE001
        lines.append(f"studies          not importable ({type(e).__name__}): the calibration, error and mask-training "
                     "studies need the source checkout (src/mppp_studies)")
    missing = 0
    for var, what in (("MPPP_SCAPES", "the scapes root (WORK folders)"), ("MPPP_PDS", "the PDS archive"),
                      ("MPPP_GPU_PYTHON", "the GPU python for SIFT (optional)"), ("COLMAP_BAT", "COLMAP (optional, GUI)")):
        v = os.environ.get(var)
        ok = v and Path(v).exists()
        if not v:
            txt = "not set"
            if var in ("MPPP_SCAPES", "MPPP_PDS"):
                missing += 1
        else:
            txt = f"{v}" + ("" if ok else "  (MISSING on disk)")
        lines.append(f"{var:<16} {txt}   - {what}")
    for mod, what in (("pycolmap", "the COLMAP pipeline"), ("pyceres", "the weighted bundle adjustment"),
                      ("torch", "LightGlue / mask inference"), ("kornia", "SIFT + LightGlue matching"),
                      ("matplotlib", "figures"), ("rasterio", "1 m DTM map backgrounds (optional)"),
                      ("adjustText", "map labels (optional)")):
        try:
            m = importlib.import_module(mod)
            lines.append(f"{mod:<16} {getattr(m, '__version__', 'ok')}   - {what}")
        except Exception as e:                                        # noqa: BLE001
            lines.append(f"{mod:<16} MISSING ({type(e).__name__})   - {what}")
    fm = check_form()
    lines.append(f"camera models    {cmods_dir()} fingerprint {models_fingerprint()} form {'ok' if fm['ok'] else 'NOT OK: ' + '; '.join(fm['errors'])}")
    age = waypoints_age_days()
    lines.append(f"waypoints        {'packaged snapshot (mppp waypoints --refresh)' if age is None else f'downloaded copy {age:.1f} days old'}")
    lines += _live_runs(os.environ.get("MPPP_SCAPES"))
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
    """``nerfstudio``: :mod:`mppp.sfm.nerfstudio` on one block."""
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
    """``metashape``: :mod:`mppp.sfm.metashape` on one block."""
    from .sfm.metashape import write_metashape_job
    from .sfm.project import SfmProject
    w = Path(a.work)
    proj_dir = w if (w / "project.json").is_file() else w / ("colmap" + (f"_{a.variant}" if a.variant else ""))
    if not (proj_dir / "project.json").is_file():
        print(f"metashape: no project at {proj_dir}", file=sys.stderr)
        return 2
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


def cmd_dense(a: argparse.Namespace) -> int:
    """``dense``: :mod:`mppp.sfm.dense` on one block."""
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
    if getattr(a, "station", None) or getattr(a, "images", None):
        subset = subset_images(proj, station=a.station, families=a.cameras, sequences=a.sequence, names=a.images,
                               scale=a.scale)
        if not subset:
            print(f"dense: no frames for --station {a.station} --cameras {a.cameras} --sequence {a.sequence}", file=sys.stderr)
            return 2
        ws = proj_dir / subset_label(a.station, a.sequence, a.images)
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
    if a.prepare or not steps:
        res = prepare_dense(proj, ws, max_image_size=size, overwrite=a.overwrite, images=subset)
        print(f"workspace {res['workspace']}: {res['images']} images" + (f", {res['masks']['images']} masks" if res.get("masks") else ""))
        if not getattr(a, "no_bat", False) and subset is None:
            print(f"batch file: {write_dense_bat(proj, max_image_size=size)}")
        if not steps:
            return 0
    import pycolmap
    if a.python and ("stereo" in steps or not getattr(pycolmap, "has_cuda", False)):
        from .sfm.gpu import run_step
        res = run_step(a.python, "dense", proj, steps=steps, max_image_size=size, workspace=str(ws), images=subset)
    else:
        res = run_dense(proj, workspace=ws, steps=steps, max_image_size=size, images=subset)
    for k, v in (res.get("steps") or {}).items():
        print(f"  {k}: {v}")
    return 0


def cmd_version(a: argparse.Namespace) -> int:
    from . import __file__ as where, __version__
    print(f"mppp {__version__} ({Path(where).parent})")
    return 0


def _ibr_command(name: str):
    def run(a: argparse.Namespace) -> int:
        from .ibr.cli import COMMANDS as IBR
        return IBR[name](a)
    run.__name__ = f"cmd_{name}"
    return run


COMMANDS = {"gui": cmd_gui, "process": cmd_process, "align": cmd_align, "run": cmd_run, "sites": cmd_sites, "cmods": cmd_cmods,
            "mask": cmd_mask, "example": cmd_example, "waypoints": cmd_waypoints, "maps": _study_command("maps"), "calib": _study_command("calib"), "doctor": cmd_doctor,
            "zcam-states": _study_command("zcam-states"), "dense": cmd_dense, "metashape": cmd_metashape, "nerfstudio": cmd_nerfstudio,
            "mesh": _ibr_command("mesh"), "depthmaps": _ibr_command("depthmaps"), "ods": _ibr_command("ods"),
            "version": cmd_version}



def main(argv: Optional[List[str]] = None) -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 27 Oct: as mppp_env.bat (torch and pycolmap both bring OpenMP)
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
