"""
The official numbered blocks of rover stations in ``src/mppp/data/sites.json`` (MPPP v0p81, sites file version 5).

Version 4 carried the 34 blocks of :mod:`mppp.maps` as extra sites ``blk01_...`` - ``blk34_...`` that
duplicated the named sites.  Version 5 (17 Oct 2026, Christian's layout):

- no ``blk..`` sites: the blocks are a top-level table ``"official_blocks"``, ``{"1": {...}, ..., "36": {...}}``, one
  record per block (name, the site that runs it, region, kind, centre, half size, visits, end-of-drive sols, Navcam and
  Mastcam-Z 360 counts, waypoint statistics);
- every site has ``"official_blocks"``: the numbers of the blocks it includes or is part of (``[]`` for none) - when
  at least half of block N's end-of-drive stations (inside its +-30 m square, during its visits) fall in the site's
  sol ranges (the site includes N), or at least half of the site's own end-of-drive stations are stations of block N
  (the site is a smaller part of N, e.g. chal_rocks of block 3) (:data:`INCLUDE_FRACTION`);
- every site has ``"notes"`` (was ``"note"``);
- every ``"sols"`` is a list of ranges, ``[[first, last]]`` for one visit;
- every site and every block records ``"max_waypoint_separation_m"`` - the largest horizontal distance between two
  waypoints (any kind: end-of-drive, mid-drive, interpolated) on its sols - with ``"n_waypoints"`` and
  ``"rover_sites"`` (the rover localisation sites among them: each site is one localisation, so offsets of the
  waypoints are shared within a site and average out only over several sites).

:func:`restructure` turns a version-4 table into version 5; :func:`update_waypoint_stats` recomputes the waypoint
fields after a new waypoints file (``scripts/mppp_maps.py waypoints``).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
INCLUDE_FRACTION = 0.5
MATCH_JACCARD = 0.5
SITES_FILE_VERSION_OFFICIAL = 5


def _ranges(sols: Any) -> List[List[int]]:
    """``[a, b]`` or ``[[a, b], ...]`` -> ``[[a, b], ...]``."""
    if not sols:
        return []
    if isinstance(sols[0], (list, tuple)):
        return [[int(a), int(b)] for a, b in sols]
    return [[int(sols[0]), int(sols[1])]]


def _in(s: int, rs: Sequence[Sequence[int]]) -> bool:
    return any(a <= s <= b for a, b in rs)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --- waypoints -------------------------------------------------------------------------------------------------------

def waypoint_points(waypoints: Union[PathLike, Dict[str, Any]]) -> np.ndarray:
    """Rows (sol, rover site, easting, northing, end of drive 1/0) of a waypoint GeoJSON (file or dict)."""
    d = waypoints if isinstance(waypoints, dict) else json.loads(Path(waypoints).read_text(encoding="utf-8"))
    rows = []
    for f in d.get("features") or []:
        p = f.get("properties") or {}
        try:
            rows.append((int(p["sol"]), int(p.get("site", -1)), float(p["easting"]), float(p["northing"]),
                         1.0 if p.get("final") == "y" else 0.0))
        except (KeyError, TypeError, ValueError):
            continue
    return np.array(rows, float).reshape(-1, 5)


def waypoint_stats(ranges: Sequence[Sequence[int]], W: np.ndarray) -> Dict[str, Any]:
    """``max_waypoint_separation_m`` (largest horizontal distance between two waypoints on the sols of ``ranges``,
    m, 0.1 m), ``n_waypoints`` and ``rover_sites`` (sorted) of the waypoint rows ``W`` (:func:`waypoint_points`)."""
    m = np.zeros(len(W), bool)
    for a, b in ranges:
        m |= (W[:, 0] >= a) & (W[:, 0] <= b)
    P = W[m]
    sep = 0.0
    if len(P) > 1:
        xy = P[:, 2:4]
        sep = float(np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)).max())
    return {"max_waypoint_separation_m": round(sep, 1), "n_waypoints": int(len(P)),
            "rover_sites": sorted({int(s) for s in P[:, 1]})}


# --- official blocks -------------------------------------------------------------------------------------------------

def official_table(blocks_sites: Dict[str, Any], W: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """The ``"official_blocks"`` table from the block sites of a ``sites_blocks.json`` (``blkNN_...`` entries with a
    ``"block"`` record), with the waypoint statistics of each block's visits.  The ``"site"`` field is filled by
    :func:`restructure`."""
    out = {}
    for k, v in sorted(blocks_sites.items(), key=lambda kv: kv[1]["block"]["number"]):
        b = v["block"]
        visits = sorted(_ranges(v["sols"]), key=lambda r: (-(r[1] - r[0]), r[0]))       # longest first
        rec = {"name": b.get("name") or v.get("label"), "site": None, "region": b.get("region", ""),
               "kind": b.get("kind"), "source_key": b.get("source_key"),
               "visits": visits, "end_of_drive_sols": list(b.get("end_of_drive_sols") or []),
               "n_navcam_360": b.get("n_navcam_360"), "n_partial_navcam": b.get("n_partial_navcam"),
               "mastcam_z_360": list(b.get("mastcam_z_360") or []),
               "centre": b.get("centre"), "half_size_m": b.get("half_size_m"),
               "zcam": list(v.get("zcam") or []), "notes": v.get("notes", v.get("note", ""))}
        rec.update(waypoint_stats(visits, W))
        out[str(int(b["number"]))] = rec
    return out


def included_blocks(ranges: Sequence[Sequence[int]], official: Dict[str, Dict[str, Any]],
                    W: Optional[np.ndarray] = None, fraction: float = INCLUDE_FRACTION) -> List[int]:
    """The numbers of the official blocks a site with sol ``ranges`` includes or is part of (module docstring);
    ``W`` (:func:`waypoint_points`) gives the site's own end-of-drive stations for the second test."""
    own = [] if W is None else [int(r[0]) for r in W if r[4] > 0 and _in(int(r[0]), ranges)]
    out = []
    for n, b in official.items():
        eod = [s for s in b["end_of_drive_sols"] if _in(s, b["visits"])] or list(b["end_of_drive_sols"])
        if not eod:
            continue
        covers = sum(_in(s, ranges) for s in eod) >= fraction * len(eod)
        part = bool(own) and sum(s in set(eod) for s in own) >= fraction * len(own)
        if covers or part:
            out.append(int(n))
    return sorted(out)


def _sol_set(rs):
    return {s for a, b in rs for s in range(a, b + 1)}


def restructure(table: Dict[str, Any], blocks: Dict[str, Any], waypoints: Union[PathLike, Dict[str, Any]],
                renames: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    A version-4 site table -> version 5 (module docstring).  ``blocks``: the block sites of a ``sites_blocks.json``
    (:func:`mppp.maps.load_blocks`); ``renames``: old site name -> new, applied to the groups.  Returns (table, report
    lines).  A block that no site includes on its own gets a new site (its visits, key = the block name as a slug).
    """
    renames = dict(renames or {})
    W = waypoint_points(waypoints)
    rep: List[str] = []
    official = official_table(blocks, W)
    sites: Dict[str, Any] = {}
    for k, v in table["sites"].items():
        if re.match(r"blk\d\d_", k):
            continue
        v = dict(v)
        v["sols"] = _ranges(v["sols"])
        bad = [r for r in v["sols"] if r[1] < r[0]]
        for r in bad:
            rep.append(f"site {k}: sols {r} reversed -> {[r[1], r[0]]}")
            r[0], r[1] = r[1], r[0]
        notes = v.pop("note", None)
        v["notes"] = v.get("notes") or notes or ""
        v.pop("block", None)
        sites[k] = v
    for k, v in sites.items():
        v["official_blocks"] = included_blocks(v["sols"], official, W)
    # the site that runs each block: one that lists this block only and either includes it or matches its sols
    # closely (Jaccard >= MATCH_JACCARD); else a new site with the block's visits
    for n, b in official.items():
        bs = _sol_set(b["visits"])
        eod = [s for s in b["end_of_drive_sols"] if _in(s, b["visits"])] or list(b["end_of_drive_sols"])

        def jac(k):
            ss = _sol_set(sites[k]["sols"])
            return len(bs & ss) / max(1, len(bs | ss))

        def covers(k):
            return bool(eod) and sum(_in(s, sites[k]["sols"]) for s in eod) >= INCLUDE_FRACTION * len(eod)
        cand = [k for k, v in sites.items() if (v["official_blocks"] == [int(n)] and (covers(k) or jac(k) >= MATCH_JACCARD))
                or sorted(v["sols"]) == sorted(b["visits"])]             # (or a site made from this block)
        if cand:
            b["site"] = max(cand, key=lambda k: (jac(k), -len(k)))
            continue
        key = slug(b["name"])
        while key in sites:
            key += "_block"
        sites[key] = {"sols": [list(r) for r in b["visits"]], "label": b["name"], "zcam": list(b["zcam"]),
                      "notes": b["notes"] or f"official block {n}", "no_mask_inference_at": [],
                      "settings": {}, "official_blocks": included_blocks(b["visits"], official, W)}
        b["site"] = key
        rep.append(f"block {n} {b['name']}: no site included it on its own -> new site {key!r} {b['visits']}")
    for k, v in sites.items():
        if not v["notes"]:
            ob = v["official_blocks"]
            v["notes"] = ("official block " + ", ".join(f"{n} {official[str(n)]['name']}" for n in ob)) if ob \
                else "not an official block"
        v.update(waypoint_stats(v["sols"], W))
    for b in official.values():
        b.pop("zcam", None)
    # field order: sols, label, official_blocks, notes, the waypoint fields, then the rest as they were
    first = ("sols", "label", "official_blocks", "notes", "max_waypoint_separation_m", "n_waypoints", "rover_sites")
    sites = {k: {**{f: v[f] for f in first if f in v}, **{f: x for f, x in v.items() if f not in first}}
             for k, v in order_by_first_sol(sites).items()}
    groups = {}
    for g, members in (table.get("groups") or {}).items():
        if g.startswith("blocks_"):
            continue
        new = [renames.get(m, m) for m in members]
        for a, b2 in zip(members, new):
            if a != b2:
                rep.append(f"group {g}: {a} -> {b2}")
        groups[g] = list(dict.fromkeys(new))
    groups["blocks_all"] = [official[n]["site"] for n in sorted(official, key=int)]
    for n in sorted(official, key=int):
        groups.setdefault("blocks_" + (official[n]["region"] or "other"), []).append(official[n]["site"])
    out = {k: x for k, x in table.items() if k not in ("sites", "groups", "official_blocks", "version")}
    out["version"] = SITES_FILE_VERSION_OFFICIAL
    out["sites"] = sites
    out["official_blocks"] = official
    out["groups"] = groups
    return out, rep


def order_by_first_sol(sites: Dict[str, Any]) -> Dict[str, Any]:
    """The sites in the order of their first visit (the earliest sol of any range; Christian, 17 Oct), the file
    order kept among sites that start on the same sol."""
    pos = {k: i for i, k in enumerate(sites)}
    return dict(sorted(sites.items(), key=lambda kv: (min(a for a, _ in _ranges(kv[1]["sols"])), pos[kv[0]])))


def update_waypoint_stats(table: Dict[str, Any], waypoints: Union[PathLike, Dict[str, Any]]) -> List[str]:
    """Recompute the waypoint fields of every site and official block in place (after a new waypoints file);
    returns the lines that changed."""
    W = waypoint_points(waypoints)
    out = []
    for kind, items in (("site", table["sites"].items()), ("block", (table.get("official_blocks") or {}).items())):
        for k, v in items:
            new = waypoint_stats(_ranges(v["sols"] if kind == "site" else v["visits"]), W)
            old = {f: v.get(f) for f in new}
            if old != new:
                out.append(f"{kind} {k}: {old['max_waypoint_separation_m']} -> {new['max_waypoint_separation_m']} m, "
                           f"{old['n_waypoints']} -> {new['n_waypoints']} waypoints")
                v.update(new)
    return out


def write_table(table: Dict[str, Any], path: PathLike) -> Path:
    """A version-5 table in the one-line-per-entry layout of ``src/mppp/data/sites.json``."""
    p = Path(path)
    L = ["{"]
    for k in table:
        if k not in ("sites", "official_blocks", "groups"):
            L.append(f" {json.dumps(k)}: {json.dumps(table[k], ensure_ascii=False)},")
    for sec in ("sites", "official_blocks", "groups"):
        items = list((table.get(sec) or {}).items())
        L.append(f" {json.dumps(sec)}: {{")
        w = max((len(json.dumps(k)) for k, _ in items), default=8) + 1
        for i, (k, v) in enumerate(items):
            if sec == "sites" and isinstance(v, dict) and "notes" in v:     # v0p85 (Christian): "notes" last
                v = {**{kk: vv for kk, vv in v.items() if kk != "notes"}, "notes": v["notes"]}
            L.append(f"  {(json.dumps(k) + ':').ljust(w)} {json.dumps(v, ensure_ascii=False)}"
                     + ("," if i + 1 < len(items) else ""))
        L.append(" }" + ("," if sec != "groups" else ""))
    L.append("}")
    txt = "\n".join(L) + "\n"
    json.loads(txt)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:   # LF on Windows too (no line-ending-only diffs)
        fh.write(txt)
    return p


def site_official_blocks(site: str, table: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The official block records a site includes (with their ``"number"``)."""
    from .sites import load_site_table
    t = table if table is not None else load_site_table()
    ob = t.get("official_blocks") or {}
    return [{"number": int(n), **ob[str(n)]} for n in (t["sites"][site].get("official_blocks") or []) if str(n) in ob]


def block_runs(table: Optional[Dict[str, Any]] = None) -> List[Tuple[int, str]]:
    """(number, site) of every official block, by number."""
    from .sites import load_site_table
    t = table if table is not None else load_site_table()
    return [(int(n), b.get("site")) for n, b in sorted((t.get("official_blocks") or {}).items(), key=lambda x: int(x[0]))]
