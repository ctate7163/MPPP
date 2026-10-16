"""
mppp.maps (MPPP v0p80): the blocks of rover stations - where Perseverance parked several times close together - from
the waypoints, into the site definitions, and their maps.

    from mppp.maps import build_from_waypoints
    res = build_from_waypoints("M20_waypoints_sol2009_all.json")        # or None: the cached / packaged waypoints
    res["blocks"]            # 34 blocks: number, name, region, centre, +-30 m square, visits, N× / Z× counts
    res["sites"], res["groups"]    # sites.json v4 entries blk01_... and the groups blocks_all, blocks_<region>

Modules: :mod:`.waypoints` (rows, lon/lat fit), :mod:`.blocks` (clusters, the 34 block definitions),
:mod:`.export` (sites.json entries, merge, the waypoint-map work's sites_blocks.json), :mod:`.figures` (traverse map,
zoom grid), :mod:`.terrain` (DEM hillshade).  Command line: ``scripts/mppp_maps.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from .blocks import BLOCK_DEFS, HALF_SIZE_M, REGIONS, block_lines, build_blocks, clusters, visits_of  # noqa: F401
from .export import blocks_to_sites, compare_blocks, load_blocks_file, merge_sites, write_sites  # noqa: F401
from .waypoints import load, lonlat_fit, waypoint_rows  # noqa: F401
from . import blocks, export, figures, terrain, waypoints  # noqa: F401,E402

PathLike = Union[str, Path]


def load_zcam360(path: Optional[PathLike]) -> Optional[list]:
    """The numbered Mastcam-Z 360s from a CSV with columns number, sol[, name] (None without a file)."""
    if not path:
        return None
    import csv
    with open(path, encoding="utf-8-sig") as f:
        return [{"number": int(r["number"]), "sol": int(r["sol"]), "name": r.get("name")} for r in csv.DictReader(f)
                if r.get("sol")]


def build_from_waypoints(waypoints: Optional[PathLike] = None, zcam360: Optional[PathLike] = None,
                         site_table: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Waypoints -> rows, clusters, blocks, sites.json entries and groups (see the module docstring)."""
    data = load(waypoints)
    rows = waypoint_rows(data)
    blocks = build_blocks(rows, site_table, load_zcam360(zcam360))
    sites, groups = blocks_to_sites(blocks, site_table)
    return {"rows": rows, "clusters": clusters(rows), "blocks": blocks, "sites": sites, "groups": groups,
            "sol_range": [min(r["sol"] for r in rows), max(r["sol"] for r in rows)],
            "waypoints": str(waypoints) if waypoints else "mppp.load_waypoints()"}
