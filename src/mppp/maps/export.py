"""
Blocks of rover stations into the MPPP site definitions (MPPP v0p80, :mod:`mppp.maps`).

:func:`blocks_to_sites` turns blocks (:func:`mppp.maps.blocks.build_blocks`) into ``sites.json`` (version 4) entries,
keys ``blk01_oeb_landing_site`` ... ``blk34_marble_mountain``: ``"sols"`` = the visits (longest first: it names the WORK
folder), ``"zcam"`` and ``"no_mask_inference_at"`` copied from the matching legacy site, and the ``"block"`` record
(number, region, kind, legacy_site, centre, half_size_m, end-of-drive sols, Navcam 360 count, numbered Mastcam-Z 360s).
Groups: ``blocks_all`` and one per region (``blocks_crater_floor_seitah`` ...).

:func:`merge_sites` adds such entries (or the ``sites_blocks.json`` the waypoint-map work wrote, :func:`load_blocks_file`)
to a site table without touching the other sites; :func:`write_sites` writes it in the one-line-per-site layout of
``src/mppp/data/sites.json`` so that diffs stay readable.  Run ``scripts/check_sites.py`` after a merge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]


def blocks_to_sites(blocks: Sequence[Dict[str, Any]], legacy: Optional[Dict[str, Any]] = None
                    ) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """(sites, groups) for the blocks; ``legacy``: the site table the zcam lists and mask stations come from."""
    from .blocks import REGIONS
    if legacy is None:
        from ..sfm.sites import load_site_table
        legacy = load_site_table()
    sites: Dict[str, Any] = {}
    groups: Dict[str, List[str]] = {"blocks_all": []}
    for b in blocks:
        if b.get("missing") or not b.get("visits"):
            continue
        old = legacy["sites"].get(b.get("legacy_site") or "") or {}
        vis = [list(map(int, v)) for v in b["visits"]]
        rec = {"sols": vis[0] if len(vis) == 1 else vis, "label": b["name"],
               "zcam": list(old.get("zcam") or []),
               "note": f"block {b['number']} ({b['kind']}"
                       + (f", from {b['legacy_site']}" if b.get("legacy_site") else "") + ")"
                       + (f"; {b['note']}" if b.get("note") else ""),
               "no_mask_inference_at": list(old.get("no_mask_inference_at") or []), "settings": {},
               "block": {"number": int(b["number"]), "region": b["region"], "kind": b["kind"],
                         "legacy_site": b.get("legacy_site"),
                         "centre": {k: round(float(v), 7 if k in ("lon", "lat") else 3) for k, v in b["centre"].items()},
                         "half_size_m": float(b["half_size_m"]), "end_of_drive_sols": list(b["end_of_drive_sols"]),
                         "navcam_360s": int(b["navcam_360s"]), "zcam_360s": list(b.get("zcam_360s") or [])}}
        sites[b["key"]] = rec
        groups["blocks_all"].append(b["key"])
        groups.setdefault(f"blocks_{b['region']}", []).append(b["key"])
    order = ["blocks_all"] + [f"blocks_{k}" for k, _ in REGIONS if f"blocks_{k}" in groups]
    return sites, {k: groups[k] for k in order}


def load_blocks_file(path: PathLike) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """(sites, groups) of a ``sites_blocks.json`` (the waypoint-map work's export, or :func:`write_sites` output)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(d.get("sites") or {}), dict(d.get("groups") or {})


def merge_sites(table: Dict[str, Any], sites: Dict[str, Any], groups: Dict[str, List[str]],
                replace: bool = True) -> Dict[str, Any]:
    """A copy of ``table`` with ``sites`` and ``groups`` added (``replace``: an existing key is overwritten, else kept);
    the file version set to 4."""
    from ..sfm.sites import SITES_FILE_VERSION
    out = json.loads(json.dumps(table))
    out["version"] = max(int(out.get("version") or 0), SITES_FILE_VERSION)
    for k, v in sites.items():
        if replace or k not in out["sites"]:
            out["sites"][k] = v
    out.setdefault("groups", {})
    for k, v in groups.items():
        if replace or k not in out["groups"]:
            out["groups"][k] = list(v)
    return out


def write_sites(table: Dict[str, Any], path: PathLike) -> Path:
    """The site table in the layout of ``src/mppp/data/sites.json``: one line per site and per group."""
    p = Path(path)
    lines = ["{"]
    head = [k for k in table if k not in ("sites", "groups")]
    for k in head:
        lines.append(f" {json.dumps(k)}: {json.dumps(table[k], ensure_ascii=False)},")
    lines.append(' "sites": {')
    items = list(table["sites"].items())
    w = max((len(json.dumps(k)) for k, _ in items), default=10) + 1
    for i, (k, v) in enumerate(items):
        lines.append(f"  {(json.dumps(k) + ':').ljust(w)} {json.dumps(v, ensure_ascii=False)}" + ("," if i + 1 < len(items) else ""))
    lines.append(" },")
    lines.append(' "groups": {')
    g = list((table.get("groups") or {}).items())
    for i, (k, v) in enumerate(g):
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)}" + ("," if i + 1 < len(g) else ""))
    lines.append(" }")
    lines.append("}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.loads(p.read_text(encoding="utf-8"))                  # it must read back
    return p


def merge_into_sites_file(path: PathLike, sites: Dict[str, Any], groups: Dict[str, List[str]]) -> Path:
    """Add (or replace) the block sites and groups in a ``sites.json`` written in the one-line-per-site layout, keeping
    every other line as it is (so that the diff shows only the blocks); falls back to :func:`write_sites` for a file in
    another layout.  The version is set to 4."""
    import re
    from ..sfm.sites import SITES_FILE_VERSION
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    table = json.loads(text)
    lines = text.splitlines()
    try:
        i_end_sites = next(i for i, l in enumerate(lines) if l.strip() == "},"
                           and any(x.strip().startswith('"groups"') for x in lines[i + 1:i + 2]))
        i_groups = i_end_sites + 1
        i_end_groups = next(i for i in range(len(lines) - 1, i_groups, -1) if lines[i].strip() == "}")
    except StopIteration:
        return write_sites(merge_sites(table, sites, groups), p)
    keep_sites = [l for l in lines[:i_end_sites] if not any(l.strip().startswith(json.dumps(k) + ":") for k in sites)]
    body = [l for l in keep_sites]
    if body and not body[-1].rstrip().endswith((",", "{")):
        body[-1] = body[-1].rstrip() + ","
    w = max(len(json.dumps(k)) for k in sites) + 1 if sites else 10
    new_lines = [f"  {(json.dumps(k) + ':').ljust(w)} {json.dumps(v, ensure_ascii=False)}" for k, v in sites.items()]
    new_lines = [l + "," for l in new_lines[:-1]] + new_lines[-1:]
    glines = [l for l in lines[i_groups + 1:i_end_groups - 1]
              if not any(l.strip().startswith(json.dumps(k) + ":") for k in groups)]
    glines = [l.rstrip().rstrip(",") for l in glines] + [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in groups.items()]
    glines = [l + "," for l in glines[:-1]] + glines[-1:]
    out = body + new_lines + [lines[i_end_sites], lines[i_groups]] + glines + lines[i_end_groups - 1:]
    txt = "\n".join(out) + "\n"
    txt = re.sub(r'"version": \d+', f'"version": {SITES_FILE_VERSION}', txt, count=1)
    json.loads(txt)                                             # must parse before it is written
    p.write_text(txt, encoding="utf-8")
    return p


def compare_blocks(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Lines: where two block site sets (e.g. :func:`blocks_to_sites` here and the waypoint-map work's
    ``sites_blocks.json``) differ - keys, visits, Navcam 360 counts, centres (m)."""
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            out.append(f"{k}: only in {'the first' if k in a else 'the second'}")
            continue
        x, y = a[k], b[k]
        if x.get("sols") != y.get("sols"):
            out.append(f"{k}: sols {x.get('sols')} vs {y.get('sols')}")
        bx, by = x.get("block") or {}, y.get("block") or {}
        if bx.get("navcam_360s") != by.get("navcam_360s"):
            out.append(f"{k}: N× {bx.get('navcam_360s')} vs {by.get('navcam_360s')}")
        cx, cy = bx.get("centre") or {}, by.get("centre") or {}
        if all(q in cx and q in cy for q in ("easting_m", "northing_m")):
            d = ((cx["easting_m"] - cy["easting_m"]) ** 2 + (cx["northing_m"] - cy["northing_m"]) ** 2) ** 0.5
            if d > 1.0:
                out.append(f"{k}: centres {d:.1f} m apart")
    return out or ["the same keys, visits, counts and centres (within 1 m)"]
