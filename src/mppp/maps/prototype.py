"""
The waypoint-map pipeline of 4-16 Oct 2026 (``m20_sites/``, the prototype for mppp.map) run from the package
(MPPP v0p80, :mod:`mppp.maps`).

``m20_sites/`` is that work as delivered (``m20_site_figures_with_dem.zip``, 16 Oct): ``mppp_map.py`` and its step
scripts (``cluster.py``, ``analyze.py``, ``sitesdef.py``, ``zooms2.py``, ``mapplot3.py``, ``export_sites.py``), the
support modules (``refdata.py`` - Mastcam-Z numbered 360s #1-#56, sample cores; ``dem.py``; ``terrain_tiles.py``,
``qmesh.py``), the inputs (``M20_waypoints.json`` sols 13-2009, the legacy ``sites.json`` the blocks were built from,
``table_rows.json``, ``clusters*.json``, the ~70 m ``terrain/`` tiles) and its outputs (``sites_zoom.json``,
``sites_blocks.json``), with ``docs/mppp_map.md``.  The scripts work in their own folder and keep module-level code,
so :func:`run` copies the folder to a work directory and runs ``mppp_map.py`` there, one process per step (each map
figure needs one to a few GB at 300-450 dpi).

    from mppp.maps import prototype
    prototype.run("all", "D:/scapes/maps_20261017")                       # blocks, zooms, figures 2-6, export
    prototype.run("figures", out, figures=("5", "6"), dpi=450)
    prototype.run("all", out, waypoints="M20_waypoints_newer.json", recluster=True)   # new waypoints

``sites_blocks.json`` of the run goes into ``src/mppp/data/sites.json`` with ``scripts/mppp_maps.py merge``.  Figures
need matplotlib, scipy, pillow and adjustText (``pip install adjustText``); rasterio and pyproj only for 1 m HiRISE
DTM backgrounds (``M20_DEMS`` or ``<workdir>/dem/``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]
HERE = Path(__file__).resolve().parent / "m20_sites"
COMMANDS = ("all", "clusters", "blocks", "zooms", "figures", "export")
OUTPUTS = ("sites_zoom.json", "sites_blocks.json", "clusters.json", "clusters_big.json")


def source_dir() -> Path:
    """The packaged prototype folder (read-only use: :func:`run` works on a copy)."""
    return HERE


def packaged_sites_blocks() -> Path:
    """The 34 blocks as delivered (``m20_sites/sites_blocks.json``, waypoints to sol 2009)."""
    return HERE / "sites_blocks.json"


def prepare_workdir(workdir: PathLike, waypoints: Optional[PathLike] = None,
                    legacy_sites: Optional[PathLike] = None, refresh: bool = False) -> Path:
    """Copy the prototype into ``workdir`` (only missing files unless ``refresh``); ``waypoints`` / ``legacy_sites``
    replace its ``M20_waypoints.json`` / ``sites.json``."""
    w = Path(workdir)
    w.mkdir(parents=True, exist_ok=True)
    for src in HERE.rglob("*"):
        if "__pycache__" in src.parts or src.is_dir():
            continue
        dst = w / src.relative_to(HERE)
        if refresh or not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if waypoints:
        shutil.copy2(waypoints, w / "M20_waypoints.json")
    if legacy_sites:
        shutil.copy2(legacy_sites, w / "sites.json")
    return w


def run(cmd: str, out: PathLike, figures: Sequence[str] = (), workdir: Optional[PathLike] = None,
        waypoints: Optional[PathLike] = None, legacy_sites: Optional[PathLike] = None, recluster: bool = False,
        dpi: Optional[int] = None, refresh: bool = False, python: Optional[str] = None) -> Path:
    """
    Run a step of the prototype (``mppp_map.py``): ``cmd`` in :data:`COMMANDS`; figures and ``sites_blocks.json`` go to
    ``out``; the work folder (default ``<out>/_m20_sites``) keeps the intermediate files.  ``recluster`` runs
    ``clusters`` first (needed after a new waypoints file).  ``dpi`` sets ``M20_DPI``.  Returns ``out``.
    """
    if cmd not in COMMANDS:
        raise ValueError(f"cmd {cmd!r}: one of {COMMANDS}")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    w = prepare_workdir(workdir or out / "_m20_sites", waypoints, legacy_sites, refresh)
    env = dict(os.environ)
    if dpi:
        env["M20_DPI"] = str(int(dpi))
    py = python or sys.executable

    def step(*args):
        subprocess.run([py, "mppp_map.py", *args], cwd=str(w), env=env, check=True)
    if recluster and cmd != "clusters":
        step("clusters")
    if cmd == "figures":
        step("figures", str(out), *figures)
    elif cmd in ("zooms", "export", "all"):
        step(cmd, str(out))
    else:
        step(cmd)
    for name in OUTPUTS:                                      # the block list next to the figures
        if (w / name).is_file() and not (out / name).is_file():
            shutil.copy2(w / name, out / name)
    return out
