"""
The waypoint-map study of 16 Oct 2026 (the blocks of rover stations and their figures), kept outside the package.

``prototype.py`` runs the delivered pipeline in ``m20_sites/`` (needs matplotlib, adjustText, rasterio, pyproj);
``export.py`` merges a blocks file into ``mppp/data/sites.json``; ``mppp_maps.py`` is the command line.  The official
block table that the package uses lives in ``mppp.waypoints.blocks``.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import export, prototype  # noqa: F401
from .export import compare_blocks, load_blocks_file, merge_into_sites_file, merge_sites, write_sites  # noqa: F401

PathLike = Union[str, Path]


def load_blocks(path: Optional[PathLike] = None) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """(sites, groups) of a ``sites_blocks.json`` (default: the one of 16 Oct, waypoints to sol 2009)."""
    return load_blocks_file(path or prototype.packaged_sites_blocks())
