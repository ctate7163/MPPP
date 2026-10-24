"""
The ``mppp`` command: the pipeline (:mod:`mppp.pipeline`) and its bookkeeping from the shell.

::

    mppp process SITE|WORK [--root DIR] [--pds DIR] [--zcam] [--hazcam front|rear|both] [--set NAME=VALUE ...]
                 [--settings FILE] [--dry-run]
    mppp align WORK [--variant NAME] [--set ...] [--settings FILE] [--gui] [--stop-after STEP] [--force] [--status]
    mppp run SITE|WORK [...]                  process, then align
    mppp sites [list | check | folder SITE]   the site definitions (mppp/data/sites.json)
    mppp cmods [list | promote PATH ... --note TEXT]   the camera models in use
    mppp mask [list | fetch [NAME] | install FILE]      the terrain mask model
    mppp example [fetch [--dest DIR] | build --pds DIR --zcam DIR --out DIR]   the Mars 2020 example data (Rockytop)
    mppp maps [all | zooms | figures 2 3 ...] [--out DIR] [--dpi N] [--background dtm|ortho] [--tag T] [--dems F;F] [--orthos F;F]
    mppp calib navcam [--root DIR] [--group ecam_consensus | --blocks NAME=WORK ...] [--out DIR] [--promote-if-ok]
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
    p.add_argument("what", nargs="?", default="list", choices=("list", "promote"))
    p.add_argument("paths", nargs="*", help="promote: folders or files with the new models")
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

    p = sub.add_parser("maps", help="the M20 site-map figures (studies/m20_sites) from the official blocks of sites.json")
    p.add_argument("what", nargs="*", default=["all"], help="all (default) | zooms | figures [2 3 4 5 6]")
    p.add_argument("--out", default=None, metavar="DIR", help="output folder (default <repo>/Claude outputs/maps_<yyyymmdd>)")
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--background", choices=["dtm", "ortho", "white"], default=None)
    p.add_argument("--tag", default=None, help="appended to every figure name")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--dems", default=None, help="1 m DTM GeoTIFFs, os.pathsep-separated (M20_DEMS)")
    p.add_argument("--orthos", default=None, help="orthomosaic GeoTIFFs (M20_ORTHOS)")
    p = sub.add_parser("calib", help="the camera calibrations (0.91: the Navcam tied joint, the Mastcam-Z joint)")
    p.add_argument("which", choices=["navcam", "zcam", "nz", "hazcam"],
                   help="navcam: the one-adjustment joint over the finished Navcam blocks; zcam: the Mastcam-Z joint "
                        "over the finished Navcam + Mastcam-Z blocks, the Navcam held at the models in use; nz: the "
                        "combined joint (both laws free) compared with the Navcam-only law - a check, never promoted; "
                        "hazcam: the Hazcam law (Navcam form) from the finished blocks with Hazcams (0.93)")
    p.add_argument("--variant", default="", help="the blocks' project colmap_<variant> (default colmap; the Hazcam-only "
                   "alignments: hazcam)")
    p.add_argument("--root", default=None, metavar="DIR", help="the scapes root (default: MPPP_SCAPES)")
    p.add_argument("--group", default=None, help="site group(s) of the blocks, comma-separated (default ecam_consensus / "
                   "zcam<zoom>_joint, else zcam<zoom>_consensus); zcam / nz take the sites whose \"zcam\" list has the zoom")
    p.add_argument("--zoom", type=int, default=34, help="zcam: the Mastcam-Z zoom (default 34)")
    p.add_argument("--zcam-blocks", action="store_true",
                   help="hazcam: the sites' Navcam + Mastcam-Z blocks (..._zcam folders, every zoom) - 0.93.6, "
                        "zcam_joints.bat fits the Hazcams of the run_sites_zcam blocks")
    p.add_argument("--navcam-group", default=None,
                   help="nz: also the finished Navcam-only blocks of this group (e.g. ecam_consensus), "
                        "3000 points each, so the combined Navcam law is held by every Navcam block")
    p.add_argument("--blocks", nargs="*", default=None, metavar="NAME=WORK", help="blocks given explicitly instead")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--out", default=None, metavar="DIR", help="the study folder (default <root>/camera_analysis/navcal_tied_<date>)")
    p.add_argument("--points", type=int, default=None, help="Navcam points per block (default 8000)")
    p.add_argument("--zpoints", type=int, default=None, help="zcam: Mastcam-Z-only points per block (default 8000; "
                   "every Navcam-Mastcam-Z point with a third look is kept besides)")
    p.add_argument("--hold", nargs="*", default=None, help="global terms held (default Q0_yaw)")
    p.add_argument("--knot", type=float, default=None, help="the pitch hinge sol (default the rig file's)")
    p.add_argument("--aniso-f", action="store_true",
                   help="separate fx and fy thermal slopes (default: one ppm/degC per eye, fy tied to fx)")
    p.add_argument("--no-diagnostics", action="store_true", help="skip the per-block diagnostic pass (no gate then)")
    p.add_argument("--include-failed", action="store_true", help="also blocks whose alignment health is FAIL "
                   "(left out by default)")
    p.add_argument("--no-k-tests", action="store_true", help="zcam: skip the k1/k2 likelihood-ratio tests (2 more "
                   "adjustments)")
    p.add_argument("--promote-if-ok", action="store_true", help="promote the candidate when every check passes")
    p.add_argument("--no-promote", action="store_true", help="never promote (overrides --promote-if-ok)")
    p.add_argument("--note", default="", help="promotion note")
    p.add_argument("--min-blocks", type=int, default=None,
                   help="zcam / hazcam: the blocks a candidate needs (0.94.1, the tiny joints: 1 or 2)")
    p.add_argument("--promote-new-only", action="store_true",
                   help="zcam / hazcam: promote only models of which none is in use (0.94.1: a tiny joint never "
                        "replaces a model of more blocks)")
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


def _work_of(target: str, s) -> Path:
    from .pipeline import resolve_site
    return resolve_site(target, s)[0].work


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
    from .pipeline import process_key, process_site, resolve_site, select_products
    s = _settings(a, a.target)
    st, s = resolve_site(a.target, s)
    if a.dry_run:
        paths, sel = select_products(st, s)
        print(f"{st.name} (sols {st.sols_text}) -> {st.work / 'processed'}: {len(paths)} products would be processed")
        for p in paths[:20]:
            print("  ", p.name)
        if len(paths) > 20:
            print(f"   ... {len(paths) - 20} more")
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
    return _run(run_site, st.work, s, a, "run", run_key(s), stop_after=a.stop_after, info={"site": st.name})


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
    print(f"{len(res['files'])} file(s) now in {res['folder']} ({res['stamp']}); commit src/mppp/data/cmods")
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


def cmd_maps(a: argparse.Namespace) -> int:
    """The six site-map figures: the study scripts of ``studies/m20_sites/m20_sites`` (``mppp_map.py``), one process
    per step, with the settings passed through the ``M20_*`` environment (the figures' module-level code reads it)."""
    import datetime
    import subprocess
    from .paths import source_root
    root = source_root()
    study = root / "studies" / "m20_sites" / "m20_sites" if root else None
    if study is None or not (study / "mppp_map.py").is_file():
        print("ERROR: the site-map study (studies/m20_sites/m20_sites) is only in a source checkout", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["M20_SITES_FILE"] = str(a.sites_file or (root / "src" / "mppp" / "data" / "sites.json"))
    for key, val in (("M20_DPI", a.dpi), ("M20_BACKGROUND", a.background), ("M20_TAG", a.tag), ("M20_DEMS", a.dems),
                     ("M20_ORTHOS", a.orthos)):
        if val is not None:
            env[key] = str(val)
    if a.tag is None and a.background:
        env["M20_TAG"] = a.background
    out = Path(a.out) if a.out else root / "Claude outputs" / f"maps_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    what = list(a.what) or ["all"]
    cmd, rest = what[0], what[1:]
    if cmd not in ("all", "zooms", "figures", "sites"):
        print(f"ERROR: maps {cmd}: use all, zooms, figures [2 3 4 5 6] or sites", file=sys.stderr)
        return 2
    if cmd in ("zooms", "figures"):                       # the block list first (from sites.json)
        subprocess.run([sys.executable, "mppp_map.py", "sites"], cwd=str(study), env=env, check=True)
    args = [sys.executable, "mppp_map.py", cmd] + ([str(out)] if cmd != "sites" else []) + rest
    r = subprocess.run(args, cwd=str(study), env=env)
    if r.returncode == 0:
        print(f"maps: {out}")
    return int(r.returncode != 0)


def _navcam_blocks(a: argparse.Namespace) -> Dict[str, Path]:
    """{name: WORK} of the finished Navcam-only blocks of the group under the scapes root (run_done.json there)."""
    from .pipeline import run_done
    from .waypoints.sites import load_site_table, work_folder
    if a.blocks:
        out = {}
        for b in a.blocks:
            name, _, work = b.partition("=")
            if not work:
                raise SystemExit(f"--blocks takes NAME=WORK, got {b!r}")
            out[name] = Path(work)
        return out
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or "")
    if not str(root):
        raise SystemExit("--root (or MPPP_SCAPES) is required")
    table = load_site_table(a.sites_file)
    zcam = getattr(a, "which", "navcam") in ("zcam", "nz")
    zfolders = zcam or bool(getattr(a, "zcam_blocks", False))     # 0.93.6: calib hazcam --zcam-blocks
    a.group = a.group or (f"zcam{int(getattr(a, 'zoom', 34))}_joint" if zcam and f"zcam{int(getattr(a, 'zoom', 34))}_joint"
                          in table.get("groups", {}) else f"zcam{int(getattr(a, 'zoom', 34))}_consensus" if zcam
                          else "hazcam_consensus" if getattr(a, "which", "") == "hazcam" else "ecam_consensus")
    names: List[str] = []
    for grp in str(a.group).split(","):                 # 0.93: several groups, comma-separated
        g_names = table.get("groups", {}).get(grp.strip())
        if not g_names:
            raise SystemExit(f"no group {grp.strip()!r} in the site definitions")
        names += list(g_names)
    out, skipped = {}, []
    zoom = int(getattr(a, "zoom", 34) or 34)
    for n in dict.fromkeys(names):
        site = table["sites"].get(n)
        if not site:
            skipped.append(f"{n} (not a site)")
            continue
        if zcam and site.get("zcam") and zoom not in [int(z) for z in site["zcam"]]:
            skipped.append(f"{n} (no {zoom} mm in its \"zcam\" list)")
            continue
        if zfolders and not zcam and not site.get("zcam"):
            skipped.append(f"{n} (no Mastcam-Z block)")
            continue
        work = work_folder(root, n, zfolders, site["sols"])
        done = run_done(work, getattr(a, "variant", "") or "")
        if done is None:
            skipped.append(f"{n} (not finished)")
            continue
        if str(done.get("verdict")) == "fail" and not getattr(a, "include_failed", False):
            # 0.93.3: judged again under the current thresholds and rules (a block run before a rule changed)
            v2, changed = _rejudged(work, getattr(a, "variant", "") or "")
            if v2 == "fail" or v2 is None:
                skipped.append(f"{n} (alignment health FAIL; --include-failed takes it)")    # 22 Oct
                continue
            log.info("calib: %s was FAIL under the rules of its run, %s under the current ones (%s)", n, v2.upper(),
                     "; ".join(changed) or "-")
        out[n] = work
    if skipped:
        log.info("calib %s: left out: %s", "zcam" if zcam else "navcam", ", ".join(skipped))
    return out


def _rejudged(work, variant: str = ""):
    """(verdict, changed checks) of a block's saved health report under the current rules; (None, []) without one."""
    from .sfm.health import rejudge, restation
    from .sfm.workdir import project_dir
    try:
        h = json.loads((project_dir(work, variant) / "health" / "health.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    fixed = restation(h, project_dir(work, variant))     # 0.94.4: shifts from the label priors (moved Hazcam priors)
    v, changed = rejudge(h)
    return v, fixed + changed


def cmd_calib(a: argparse.Namespace) -> int:
    """``calib navcam``: the one-adjustment Navcam joint (:mod:`mppp.calib.tied`) over the finished blocks, the
    candidate folder, the promotion check and, with ``--promote-if-ok``, the promotion."""
    import datetime
    from .calib import tied as TJ
    from .calib.fit import check_lines, check_navcam, promote_navcam
    if a.which == "zcam":
        return _calib_zcam(a)
    if a.which == "nz":
        return _calib_nz(a)
    if a.which == "hazcam":
        return _calib_hazcam(a)
    blocks = _navcam_blocks(a)
    if len(blocks) < 2:
        print(f"ERROR: {len(blocks)} finished block(s); the joint needs several", file=sys.stderr)
        return 1
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or next(iter(blocks.values())).parent)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"navcal_tied_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("calib navcam: %d blocks -> %s", len(blocks), out)
    res = TJ.fit_tied(blocks, points=a.points or TJ.POINTS_PER_BLOCK, hold=tuple(a.hold) if a.hold is not None else TJ.HOLD_DEFAULT,
                      knot_sol=a.knot, diagnostics=not a.no_diagnostics, iso_f=not a.aniso_f)
    cand = TJ.write_tied(res, out / "navcam_joint", note=a.note)
    rep = check_navcam(cand)
    (out / "promotion_check.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("\n".join(check_lines(rep)))
    if a.promote_if_ok and not a.no_promote and rep["ok"]:
        pr = promote_navcam(cand, a.note or f"tied joint of {len(blocks)} blocks", check=False)
        print(f"promoted into {pr['folder']} ({pr['stamp']})")
    elif a.promote_if_ok:
        print("not promoted: a check failed (the models in use stay)")
    return 0


def _calib_zcam(a: argparse.Namespace) -> int:
    """``calib zcam``: the Mastcam-Z joint (:mod:`mppp.calib.ztied`) over the finished Navcam + Mastcam-Z blocks, the
    Navcam held at the models in use; the candidate focus model, its check and, with ``--promote-if-ok``, the
    promotion."""
    import datetime
    from .calib import models as M
    from .calib import ztied as Z
    blocks = _navcam_blocks(a)
    if not blocks:
        print(f"no finished Mastcam-Z {a.zoom} mm block: no joint", file=sys.stderr)
        return 1
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or next(iter(blocks.values())).parent)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"zcam{a.zoom}_joint_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("calib zcam: %d blocks -> %s", len(blocks), out)
    res = Z.fit_zjoint(blocks, zoom=a.zoom, nav_points=a.points or Z.NAV_POINTS_PER_BLOCK,
                       z_points=a.zpoints or Z.Z_POINTS_PER_BLOCK, diagnostics=not a.no_diagnostics,
                       k_tests=not a.no_k_tests)
    from .calib import models_fingerprint
    from .pipeline import run_done
    res["models_fingerprint"] = models_fingerprint()
    res["block_runs"] = {n: run_done(w, "") for n, w in blocks.items()}
    # 0.93: the joint holds the Navcam: a block must have been aligned with the Navcam models in use (a block aligned
    # with an earlier Mastcam-Z model is fine - the joint refits the Mastcam-Z law from its bins)
    from .calib import navcam_fingerprint
    res["navcam_fingerprint"] = navcam_fingerprint(None)
    res["block_navcam_fingerprints"] = {}
    for n, w in blocks.items():
        try:
            pjf = Path(w) / "colmap" / "project.json"
            if not pjf.is_file():                         # 0.94.1: a block given as its project folder (colmap_<variant>)
                pjf = Path(w) / "project.json"
            pj = json.loads(pjf.read_text(encoding="utf-8"))
            res["block_navcam_fingerprints"][n] = ((pj.get("settings") or {}).get("navcam_cameras") or {}).get("fingerprint")
        except (OSError, ValueError):
            res["block_navcam_fingerprints"][n] = None
    lim = {"min_blocks": a.min_blocks, "min_blocks_start": a.min_blocks} if getattr(a, "min_blocks", None) else None
    w = Z.write_zjoint(res, out, limits=lim)
    for c in w["check"]["checks"]:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}")
    print(f"=> {'every check passes' if w['check']['ok'] else 'a check failed'}; {out / 'report.md'}")
    promote = a.promote_if_ok and not a.no_promote
    if promote and getattr(a, "promote_new_only", False) and (M.zcam_files() or {}).get(int(a.zoom)):
        print(f"not promoted: a {a.zoom} mm model is in use (--promote-new-only)")
        promote = False
    if promote and w["check"]["ok"]:
        pr = M.promote([Path(w["model"])], note=a.note or f"Mastcam-Z {a.zoom} mm joint of {len(blocks)} blocks")
        print(f"promoted into {pr['folder']} ({pr['stamp']})")
    elif promote:
        print("not promoted: a check failed (the model in use stays)")
    return 0


def _calib_hazcam(a: argparse.Namespace) -> int:
    """``calib hazcam``: the Hazcam law (:mod:`mppp.calib.hazfit`) from the finished blocks with Hazcams; the
    candidate files of the eyes and pairs that pass their checks, promoted with ``--promote-if-ok``."""
    import datetime
    from .calib import hazfit as H
    from .calib import models as M
    from .sfm.workdir import project_dir
    blocks = _navcam_blocks(a)
    if not blocks:
        print("no finished block: no Hazcam fit", file=sys.stderr)
        return 1
    blocks = {n: project_dir(w, a.variant or None) for n, w in blocks.items()}
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or ".")
    out = Path(a.out) if a.out else root / "camera_analysis" / f"hazcam_fit_{datetime.date.today():%Y%m%d}"
    log.info("calib hazcam: %d blocks -> %s", len(blocks), out)
    res = H.fit_hazcam(blocks, limits={"min_blocks": a.min_blocks} if getattr(a, "min_blocks", None) else None)
    w = H.write_hazcam(res, out)
    for line in H.describe(res):
        print("  " + line)
    for c in w["check"]["checks"]:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}")
    promote = a.promote_if_ok and not a.no_promote
    if promote and getattr(a, "promote_new_only", False):        # 0.94.1: only models of which none is in use
        from .paths import cmods_dir
        kept = [f for f in w["files"] if not (cmods_dir() / Path(f).name).is_file()]
        if len(kept) < len(w["files"]):
            print("in use, not replaced (--promote-new-only): " + ", ".join(Path(f).name for f in w["files"] if f not in kept))
        w["files"] = kept
    if promote and w["files"]:
        try:
            pr = M.promote([Path(f) for f in w["files"]], note=a.note or f"Hazcam fit of {len(res['blocks'])} blocks "
                           f"({', '.join(res['blocks'])})")
        except OSError as e:                          # 0.93.8: say so instead of a traceback; the candidate stays
            print(f"NOT promoted: {e} - promote later with: mppp cmods promote {w['candidate']}", file=sys.stderr)
            return 1
        print(f"promoted {', '.join(Path(f).name for f in w['files'])} into {pr['folder']} ({pr['stamp']})")
    elif promote:
        print("not promoted: no eye or pair passed its checks")
    print(f"=> {out / 'report.md'}")
    return 0


def _calib_nz(a: argparse.Namespace) -> int:
    """``calib nz``: the combined Navcam + Mastcam-Z joint (:mod:`mppp.calib.nzjoint`) over the finished Navcam +
    Mastcam-Z blocks - both laws free - and its comparison with the Navcam-only law.  Writes a report; promotes
    nothing.  Exit 0 whether or not they agree (the report says which)."""
    import datetime
    from .calib import nzjoint as NZ
    from .calib import ztied as Z
    blocks = _navcam_blocks(a)
    if len(blocks) < 2:
        print(f"ERROR: {len(blocks)} finished Mastcam-Z block(s); the joint needs several", file=sys.stderr)
        return 1
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or next(iter(blocks.values())).parent)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"nz{a.zoom}_joint_{datetime.date.today():%Y%m%d}"
    log.info("calib nz: %d blocks -> %s", len(blocks), out)
    extra = {}
    if getattr(a, "navcam_group", None):
        import copy as _copy
        b = _copy.copy(a)
        b.which, b.group, b.blocks = "navcam", a.navcam_group, None
        extra = _navcam_blocks(b)
    res = NZ.fit_nzjoint(blocks, zoom=a.zoom, nav_points=a.points or Z.NAV_POINTS_PER_BLOCK,
                         z_points=a.zpoints or Z.Z_POINTS_PER_BLOCK, navcam_blocks=extra)
    w = NZ.write_nzjoint(res, out)
    for line in NZ.describe(res):
        print("  " + line)
    print(f"=> {'agrees with' if w['agree'] else 'DIFFERS from'} the Navcam-only law; {out / 'report.md'} (nothing promoted)")
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    """What is set and what is missing: the environment, the dependencies, the camera models, the waypoint table."""
    import importlib
    from . import __version__
    from .calib import check_form, models_fingerprint
    from .paths import cmods_dir, source_root
    from .waypoints.table import waypoints_age_days
    lines = [f"mppp {__version__}", f"python {sys.version.split()[0]} ({sys.executable})",
             f"source checkout: {source_root() or 'no (installed package)'}"]
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
    print("\n".join(lines))
    return 1 if missing else 0


def cmd_version(a: argparse.Namespace) -> int:
    from . import __file__ as where, __version__
    print(f"mppp {__version__} ({Path(where).parent})")
    return 0


COMMANDS = {"gui": cmd_gui, "process": cmd_process, "align": cmd_align, "run": cmd_run, "sites": cmd_sites, "cmods": cmd_cmods,
            "mask": cmd_mask, "example": cmd_example, "waypoints": cmd_waypoints, "maps": cmd_maps, "calib": cmd_calib, "doctor": cmd_doctor,
            "version": cmd_version}


def main(argv: Optional[List[str]] = None) -> int:
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
