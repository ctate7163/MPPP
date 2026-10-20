"""
The ``mppp`` command: the pipeline (:mod:`mppp.pipeline`) and its bookkeeping from the shell.

::

    mppp process SITE|WORK [--root DIR] [--pds DIR] [--zcam] [--set NAME=VALUE ...] [--settings FILE] [--dry-run]
    mppp align WORK [--variant NAME] [--set ...] [--settings FILE] [--gui] [--stop-after STEP] [--force] [--status]
    mppp run SITE|WORK [...]                  process, then align
    mppp sites [list | check | folder SITE]   the site definitions (mppp/data/sites.json)
    mppp cmods [list | promote PATH ... --note TEXT]   the camera models in use
    mppp mask [list | fetch [NAME] | install FILE]      the terrain mask model
    mppp maps [all | zooms | figures 2 3 ...] [--out DIR] [--dpi N] [--background dtm|ortho] [--tag T] [--dems F;F] [--orthos F;F]
    mppp calib navcam [--root DIR] [--group navcam_consensus | --blocks NAME=WORK ...] [--out DIR] [--promote-if-ok]
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
    ap.add_argument("--force", action="store_true", help="start even if the status file says a run is alive")
    ap.add_argument("--no-echo", action="store_true", help="log to the file only")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mppp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    v = ap.add_mutually_exclusive_group()
    v.add_argument("-v", "--verbose", action="store_true", help="debug messages")
    v.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    ap.add_argument("--log", default=None, metavar="FILE", help="also write the log to this file")
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

    p = sub.add_parser("waypoints", help="the Mars 2020 waypoint table in use (cached download or packaged snapshot)")
    p.add_argument("--refresh", action="store_true", help="download the current table now (every sync should)")

    p = sub.add_parser("maps", help="the M20 site-map figures (studies/m20_sites) from the official blocks of sites.json")
    p.add_argument("what", nargs="*", default=["all"], help="all (default) | zooms | figures [2 3 4 5 6]")
    p.add_argument("--out", default=None, metavar="DIR", help="output folder (default <repo>/Claude outputs/maps_<yyyymmdd>)")
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--background", choices=["dtm", "ortho"], default=None)
    p.add_argument("--tag", default=None, help="appended to every figure name")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--dems", default=None, help="1 m DTM GeoTIFFs, os.pathsep-separated (M20_DEMS)")
    p.add_argument("--orthos", default=None, help="orthomosaic GeoTIFFs (M20_ORTHOS)")
    p = sub.add_parser("calib", help="the camera calibrations (0.91: the Navcam tied joint)")
    p.add_argument("which", choices=["navcam"], help="navcam: the one-adjustment joint over the finished Navcam blocks")
    p.add_argument("--root", default=None, metavar="DIR", help="the scapes root (default: MPPP_SCAPES)")
    p.add_argument("--group", default="navcam_consensus", help="site group of the blocks (default navcam_consensus)")
    p.add_argument("--blocks", nargs="*", default=None, metavar="NAME=WORK", help="blocks given explicitly instead")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--out", default=None, metavar="DIR", help="the study folder (default <root>/camera_analysis/navcal_tied_<date>)")
    p.add_argument("--points", type=int, default=None, help="points per block (default 3000)")
    p.add_argument("--hold", nargs="*", default=None, help="global terms held (default Q0_yaw)")
    p.add_argument("--knot", type=float, default=None, help="the pitch hinge sol (default the rig file's)")
    p.add_argument("--aniso-f", action="store_true",
                   help="separate fx and fy thermal slopes (default: one ppm/degC per eye, fy tied to fx)")
    p.add_argument("--no-diagnostics", action="store_true", help="skip the per-block diagnostic pass (no gate then)")
    p.add_argument("--promote-if-ok", action="store_true", help="promote the candidate when every check passes")
    p.add_argument("--note", default="", help="promotion note")
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
    s = _settings(a, str(work), {"variant": a.variant, "source": "processed", "gui_files": bool(a.gui)})
    st, s = resolve_site(work, s)
    return _run(align, work, s, a, "align", run_key(s), stop_after=a.stop_after, info={"site": st.name})


def cmd_run(a: argparse.Namespace) -> int:
    from .pipeline import STEPS, resolve_site, run_key, run_site
    if a.stop_after and a.stop_after not in STEPS:
        print(f"ERROR: --stop-after {a.stop_after!r}: use one of {', '.join(STEPS)}", file=sys.stderr)
        return 2
    s = _settings(a, a.target, {"variant": a.variant, "source": "pds", "gui_files": bool(a.gui)})
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
    names = table.get("groups", {}).get(a.group)
    if not names:
        raise SystemExit(f"no group {a.group!r} in the site definitions")
    out, skipped = {}, []
    for n in dict.fromkeys(names):
        site = table["sites"].get(n)
        if not site:
            skipped.append(f"{n} (not a site)")
            continue
        work = work_folder(root, n, False, site["sols"])
        if run_done(work, "") is None:
            skipped.append(f"{n} (not finished)")
            continue
        out[n] = work
    if skipped:
        log.info("calib navcam: left out: %s", ", ".join(skipped))
    return out


def cmd_calib(a: argparse.Namespace) -> int:
    """``calib navcam``: the one-adjustment Navcam joint (:mod:`mppp.calib.tied`) over the finished blocks, the
    candidate folder, the promotion check and, with ``--promote-if-ok``, the promotion."""
    import datetime
    from .calib import tied as TJ
    from .calib.fit import check_lines, check_navcam, promote_navcam
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
    if a.promote_if_ok and rep["ok"]:
        pr = promote_navcam(cand, a.note or f"tied joint of {len(blocks)} blocks", check=False)
        print(f"promoted into {pr['folder']} ({pr['stamp']})")
    elif a.promote_if_ok:
        print("not promoted: a check failed (the models in use stay)")
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


COMMANDS = {"process": cmd_process, "align": cmd_align, "run": cmd_run, "sites": cmd_sites, "cmods": cmd_cmods,
            "mask": cmd_mask, "waypoints": cmd_waypoints, "maps": cmd_maps, "calib": cmd_calib, "doctor": cmd_doctor,
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
