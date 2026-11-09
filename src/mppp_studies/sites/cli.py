"""
mppp_studies.sites.cli - the ``maps`` command: the M20 site-map figures of ``m20_sites`` from the official blocks of
``sites.json`` (0.96.0: moved out of ``mppp.cli``, which forwards it here).

    python -m mppp maps all --background ortho ...        or        python -m mppp_studies.sites maps all ...
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

STUDY_DIR = Path(__file__).resolve().parent / "m20_sites"      # the figure scripts (module-level code, cwd = here)


def add_parsers(sub) -> None:
    """The ``maps`` sub-parser on an argparse subparsers object."""
    p = sub.add_parser("maps", help="the M20 site-map figures (mppp_studies.sites) from the official blocks of sites.json")
    p.add_argument("what", nargs="*", default=["all"], help="all (default) | zooms | figures [2 3 4 5 6]")
    p.add_argument("--out", default=None, metavar="DIR", help="output folder (default <repo>/Claude outputs/maps_<yyyymmdd>)")
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--background", choices=["dtm", "ortho", "white"], default=None)
    p.add_argument("--tag", default=None, help="appended to every figure name")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--dems", default=None, help="1 m DTM GeoTIFFs, os.pathsep-separated (M20_DEMS)")
    p.add_argument("--orthos", default=None, help="orthomosaic GeoTIFFs (M20_ORTHOS)")
    p.add_argument("--waypoints", choices=["refresh", "auto", "study"], default="auto",
                   help="0.96: the waypoint table the figures draw - refresh: download the current table and copy it into "
                        "the study folder; auto (default): the same when the cached copy is missing or older than a day, "
                        "else the cached copy; study: the folder's own copy as it is")


def sync_study_waypoints(study: Path, refresh: Union[bool, str] = "auto") -> str:
    """0.96: bring the site-map study's ``M20_waypoints.json`` (the copy every study script reads) to the current
    waypoint table - ``load_waypoints(refresh=...)`` downloads it into the user cache (a failed download keeps the
    cached copy, with a warning) and the file is copied into ``study`` when it differs.  Returns a one-line report
    (last sol, count, whether the study copy changed)."""
    import shutil
    from mppp.waypoints import load_waypoints
    from mppp.waypoints.table import snapshot_path
    w = load_waypoints(refresh=refresh)
    src = Path(w["_mppp_source"]["path"])
    last = max(int((f.get("properties", f)).get("sol") or 0) for f in w["features"])
    dst = study / "M20_waypoints.json"
    same = dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest() == w["_mppp_source"]["sha256"]
    if not same:
        shutil.copyfile(src, dst)
    where = "packaged snapshot" if src.resolve() == snapshot_path().resolve() else "downloaded table"
    return (f"{w['_mppp_source']['n_features']} waypoints to sol {last} ({where}, {w['_mppp_source']['file_time']}); "
            f"{'study copy updated' if not same else 'study copy already current'}")


def cmd_maps(a: argparse.Namespace) -> int:
    """The six site-map figures: the study scripts of ``mppp_studies/sites/m20_sites`` (``mppp_map.py``), one process
    per step, with the settings passed through the ``M20_*`` environment (the figures' module-level code reads it)."""
    import datetime
    import subprocess
    from mppp.paths import source_root
    root = source_root()
    study = STUDY_DIR
    if not (study / "mppp_map.py").is_file():
        print(f"ERROR: the site-map study scripts are not at {study}", file=sys.stderr)
        return 2
    if getattr(a, "waypoints", "auto") != "study":          # 0.96: the figures draw the current waypoint table
        print(f"maps: waypoints {sync_study_waypoints(study, refresh=True if a.waypoints == 'refresh' else 'auto')}")
    env = dict(os.environ)
    from mppp.paths import data_file
    env["M20_SITES_FILE"] = str(a.sites_file or data_file("sites.json"))
    for key, val in (("M20_DPI", a.dpi), ("M20_BACKGROUND", a.background), ("M20_TAG", a.tag), ("M20_DEMS", a.dems),
                     ("M20_ORTHOS", a.orthos)):
        if val is not None:
            env[key] = str(val)
    if a.tag is None and a.background:
        env["M20_TAG"] = a.background
    out = Path(a.out) if a.out else (root or Path.cwd()) / "Claude outputs" / f"maps_{datetime.date.today():%Y%m%d}"
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


COMMANDS = {"maps": cmd_maps}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mppp_studies.sites", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    add_parsers(sub)
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 2
    return int(COMMANDS[a.cmd](a))


if __name__ == "__main__":
    sys.exit(main())
