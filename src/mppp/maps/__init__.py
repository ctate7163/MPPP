"""
mppp.maps (MPPP v0p80): the blocks of rover stations - where Perseverance ended its drive repeatedly in a small area -
from the waypoints to the MPPP site definitions and their maps.

- :mod:`.prototype`: the waypoint-map pipeline as delivered on 16 Oct 2026 (``m20_sites/``; block rules, the 34
  blocks, Mastcam-Z 360 and sample reference lists, figures 1-6 from Mars to the +-30 m blocks, the sites export) -
  ``prototype.run("all", out)``; its documentation is ``m20_sites/docs/mppp_map.md``.
- :mod:`.export`: the blocks into ``src/mppp/data/sites.json`` (``merge_into_sites_file``), comparisons.
- :mod:`.waypoints`: flat waypoint rows and the lon/lat <-> easting/northing fit, for other code.

    from mppp import maps
    sites, groups = maps.load_blocks()            # the 34 blocks as delivered (sites.json v4 entries and groups)
    maps.prototype.run("figures", "out", figures=("5",))

Command line: ``scripts/mppp_maps.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import export, official, prototype, waypoints  # noqa: F401
from .export import compare_blocks, load_blocks_file, merge_into_sites_file, merge_sites, write_sites  # noqa: F401
from .waypoints import load, lonlat_fit, waypoint_rows  # noqa: F401

PathLike = Union[str, Path]


def load_blocks(path: Optional[PathLike] = None) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """(sites, groups) of a ``sites_blocks.json`` (default: the packaged one of 16 Oct, waypoints to sol 2009)."""
    return load_blocks_file(path or prototype.packaged_sites_blocks())
