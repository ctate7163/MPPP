"""
The blocks of rover stations in the MPPP site definitions (MPPP v0p80, :mod:`mppp.maps`).

The blocks themselves come from the waypoint-map pipeline (:mod:`mppp.maps.prototype`, ``m20_sites/export_sites.py``:
``sites_blocks.json``, sites.json version 4, keys ``blk01_oeb_landing_site`` ... ``blk34_marble_mountain``, groups
``blocks_all`` and ``blocks_<region>``).  Here: :func:`load_blocks_file` reads such a file, :func:`merge_into_sites_file`
puts its sites and groups into ``src/mppp/data/sites.json`` keeping every other line as it is (blocks of an earlier
merge, ``blk..`` keys, are replaced as a set), :func:`merge_sites` / :func:`write_sites` do the same on a table in
memory, and :func:`compare_blocks` lists how two block sets differ.  Run ``scripts/check_sites.py`` after a merge.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]


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


def merge_into_sites_file(path: PathLike, sites: Dict[str, Any], groups: Dict[str, List[str]],
                          replace_blocks: bool = True) -> Path:
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
    def _old_block(l: str) -> bool:                           # a site of an earlier block merge (blkNN_...)
        return replace_blocks and l.strip().startswith('"blk') and l.strip()[4:6].isdigit()
    keep_sites = [l for l in lines[:i_end_sites] if not any(l.strip().startswith(json.dumps(k) + ":") for k in sites)
                  and not _old_block(l)]
    body = [l for l in keep_sites]
    if body and not body[-1].rstrip().endswith((",", "{")):
        body[-1] = body[-1].rstrip() + ","
    w = max(len(json.dumps(k)) for k in sites) + 1 if sites else 10
    new_lines = [f"  {(json.dumps(k) + ':').ljust(w)} {json.dumps(v, ensure_ascii=False)}" for k, v in sites.items()]
    new_lines = [l + "," for l in new_lines[:-1]] + new_lines[-1:]
    glines = [l for l in lines[i_groups + 1:i_end_groups - 1]
              if not any(l.strip().startswith(json.dumps(k) + ":") for k in groups)
              and not (replace_blocks and l.strip().startswith('"blocks_'))]
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
        nx = bx.get("n_navcam_360", bx.get("navcam_360s"))
        ny = by.get("n_navcam_360", by.get("navcam_360s"))
        if nx != ny:
            out.append(f"{k}: N× {nx} vs {ny}")
        cx, cy = bx.get("centre") or {}, by.get("centre") or {}
        ex = (cx.get("easting", cx.get("easting_m")), cx.get("northing", cx.get("northing_m")))
        ey = (cy.get("easting", cy.get("easting_m")), cy.get("northing", cy.get("northing_m")))
        if None not in ex + ey:
            d = ((ex[0] - ey[0]) ** 2 + (ex[1] - ey[1]) ** 2) ** 0.5
            if d > 1.0:
                out.append(f"{k}: centres {d:.1f} m apart")
    return out or ["the same keys, visits, counts and centres (within 1 m)"]
