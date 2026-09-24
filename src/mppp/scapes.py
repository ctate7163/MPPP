"""
Scapes: named, informal image selections used so far (from ``workspace.ipynb``).

A scape is only a convenience for picking PDS products; nothing else in MPPP
depends on it.  Each selection line is tagged with the workspace "group":

* group 1 — Navcam at a specific downsample (e.g. ``_0A01``)
* group 2 — full engineering-camera and Mastcam-Z sets for the site
* group 3 — targeted Mastcam-Z mosaics by sequence (Z110 / Z063 at Rockytop)

``select_scape`` takes groups 1 and 2 and drops 110 mm Mastcam-Z by default;
group 3 is kept in the definitions for provenance only.

Definitions transcribed from ``workspace.ipynb`` (cells 3, 5-8).  Where a
workspace cell assigned ``camera_codes`` twice, only the last assignment took
effect; that is what is transcribed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .filenames import parse_filename
from .select import dedupe_by_version, find_imgs

PathLike = Union[str, Path]

SCAPES: Dict[str, Dict[str, Any]] = {
    "rockytop": {
        "sols": (460, 535),
        "selections": [
            {"group": 1, "cameras": ["NLF", "NRF"], "sequ_id": "_0A01"},
            {"group": 2, "cameras": ["FLF", "FRF"], "sequ_id": None},
            {"group": 2, "cameras": ["NLF", "NRF"], "sequ_id": None},
            {"group": 2, "cameras": ["ZL0", "ZR0"], "sequ_id": "_034"},
            # group 3: Mastcam-Z mosaics by sequence (not used by default)
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08541", "note": "sol 518 Z063 mosaic of Wildcat"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08529", "note": "sol 507 Z110 Rockytop"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08528", "note": "sol 507 Z110 Rockytop"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08508", "note": "sol 484 Z110 Bettys Rock"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08498", "note": "sol 477 Z110 Bettys Rock"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08494", "note": "sol 471 Z110 mosaic"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08490", "note": "sol 470 Z110 Backon Strip"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08487", "note": "sol 467 Z110 upper Rockytop"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08488", "note": "sol 467 Z110 mosaic"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08486", "note": "sol 466 Z110 upper Rockytop, Bettys Rock"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08489", "note": "sol 466 Z110 lower Rockytop, Bettys Rock"},
            {"group": 3, "cameras": ["ZL0", "ZR0"], "sequ_id": "08482", "note": "sol 461 Z110 Backon Strip, Bettys Rock"},
        ],
    },
    "landing": {
        "sols": (9, 48),
        "selections": [
            {"group": 2, "cameras": ["NLF", "NRF"], "sequ_id": None},
            {"group": 2, "cameras": ["ZL0", "ZR0"], "sequ_id": "_034"},
        ],
    },
    "belva": {
        "sols": (770, 835),
        "selections": [
            {"group": 2, "cameras": ["NLF", "NRF"], "sequ_id": None},
            {"group": 2, "cameras": ["ZL0", "ZR0"], "sequ_id": None},
        ],
    },
    "bunsen": {
        "sols": (1055, 1095),
        "selections": [
            {"group": 2, "cameras": ["NLF", "NRF"], "sequ_id": None},
            {"group": 2, "cameras": ["ZL0", "ZR0"], "sequ_id": None},
        ],
    },
    "hellandfjellet": {
        "sols": (1601, 1645),
        "selections": [
            {"group": 2, "cameras": ["NLF", "NRF"], "sequ_id": None},
            {"group": 2, "cameras": ["ZL0", "ZR0"], "sequ_id": None},
        ],
    },
}


def select_scape(name: str, input_dir: PathLike, groups: Sequence[int] = (1, 2),
                 exclude_zoom_mm: Sequence[int] = (110,)) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Products of scape ``name`` under ``input_dir``: union of the selections in
    ``groups``, minus Mastcam-Z frames at the zooms in ``exclude_zoom_mm``,
    de-duplicated to the highest product version.  Returns ``(paths, report)``;
    the report records the definition and counts, for the run provenance.
    """
    key = name.lower()
    if key not in SCAPES:
        raise KeyError(f"unknown scape {name!r}; known: {sorted(SCAPES)}")
    d = SCAPES[key]
    picked: Dict[Path, None] = {}
    lines = []
    for sel in d["selections"]:
        if sel["group"] not in groups:
            continue
        found = find_imgs(input_dir, sel["cameras"], sol_range=d["sols"], sequ_id=sel["sequ_id"])
        lines.append({**sel, "n_found": len(found)})
        picked.update(dict.fromkeys(found))
    paths = list(picked)
    dropped_zoom = [p for p in paths if parse_filename(p).zoom_mm in set(exclude_zoom_mm)]
    paths = [p for p in paths if p not in set(dropped_zoom)]
    paths, superseded = dedupe_by_version(paths)
    by_group: Dict[str, int] = {}
    for p in paths:
        g = parse_filename(p).camera_group
        by_group[g] = by_group.get(g, 0) + 1
    report = {"scape": key, "sols": list(d["sols"]), "groups": list(groups),
              "exclude_zoom_mm": list(exclude_zoom_mm), "selections": lines,
              "n_selected": len(paths), "n_dropped_zoom": len(dropped_zoom),
              "n_superseded_versions": len(superseded), "by_camera_group": dict(sorted(by_group.items())),
              "input_dir": str(input_dir)}
    return sorted(paths), report
