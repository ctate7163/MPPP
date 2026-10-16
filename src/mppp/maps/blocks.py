"""
Blocks of rover stations (MPPP v0p80, :mod:`mppp.maps`) - Christian's term for the places where the rover parked
several times close together, the sites of the multi-station SfM experiment.

Rules (waypoint-map work of 4-15 Oct 2026, project doc ``m20-site-figures-2026-10-14-31-sites-dem-ready.md``):

- **clusters**: end-of-drive stops with a Navcam 360 chained at <= ``LINK_M`` (10 m, single linkage), at least
  ``MIN_STOPS`` (4) of them; the landing stops of sols 13-14 belong to the OEB cluster; the cluster of sols 1756-1761
  is not a block;
- **blocks**: the clusters plus sites of ``sites.json`` and two sol windows (:data:`BLOCK_DEFS`, 34 blocks numbered in
  traverse order within six regions); each block is a square of +-``HALF_SIZE_M`` (30 m) about the mean position of
  its defining stops (Tuxedo Park shifted 20 m west so that the sol 905 and 907 stops fall inside);
- per block: the visits (sols of every waypoint inside the square, split where the rover was away more than
  ``VISIT_GAP_SOLS``; longest visit first - it names the WORK folder), the end-of-drive Navcam 360s inside the square
  (``N×``) and, given a list of the numbered Mastcam-Z 360s, those taken while the rover was inside (``Z×``).

The definitions are written so that a newer waypoint file re-derives the blocks (Hippo Pools becomes a cluster once
sols 2006 and 2009 are in it; until then it is the sol window 1947-2009).  The numbering and names are fixed here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LINK_M = 10.0
MIN_STOPS = 4
HALF_SIZE_M = 30.0
VISIT_GAP_SOLS = 30
REGIONS = (("crater_floor_seitah", "Crater Floor and Séítah"), ("delta_front", "Delta Front"),
           ("upper_fan_belva", "Upper Fan and Belva Crater"), ("margin_unit_neretva_vallis", "Margin Unit and Neretva Vallis"),
           ("crater_rim", "Crater rim"), ("crater_rim_west", "Crater rim west"))

# number, slug, name, region, how it is defined:
#   ("cluster", sol)    - the cluster of end-of-drive 360s with a stop on this sol
#   ("site", name)      - a sites.json site: the densest +-30 m square of the end-of-drive stops of its main visit
#   ("window", (a, b))  - the densest +-30 m square of the end-of-drive stops of these sols
# options: "shift_m" (dE, dN) of the square; "drop_sols" left out of the visits; "fallback" window if no cluster
BLOCK_DEFS: Tuple[Dict[str, Any], ...] = (
    dict(n=1, slug="oeb_landing_site", name="OEB Landing Site", region="crater_floor_seitah", by=("cluster", 33),
         extra_sols=(13, 14)),
    dict(n=2, slug="van_zyl_overlook", name="Van Zyl Overlook", region="crater_floor_seitah", by=("cluster", 49)),
    dict(n=3, slug="chal_rocks", name="Chal Rocks", region="crater_floor_seitah", by=("site", "chal_rocks_large")),
    dict(n=4, slug="artuby_ridge", name="Artuby Ridge", region="crater_floor_seitah", by=("cluster", 173)),
    dict(n=5, slug="rochette", name="Rochette", region="crater_floor_seitah", by=("site", "rochette")),
    dict(n=6, slug="issole", name="Issole", region="crater_floor_seitah", by=("cluster", 201)),
    dict(n=7, slug="brac", name="Brac", region="crater_floor_seitah", by=("cluster", 238)),
    dict(n=8, slug="three_forks_south", name="Three Forks South", region="delta_front", by=("cluster", 658)),
    dict(n=9, slug="three_forks_north", name="Three Forks North", region="delta_front", by=("site", "threeforks_north"),
         drop_sols=(413,)),
    dict(n=10, slug="enchanted_lake", name="Enchanted Lake", region="delta_front", by=("cluster", 421)),
    dict(n=11, slug="delta_front_ascent", name="Delta front ascent", region="delta_front", by=("cluster", 437)),
    dict(n=12, slug="rockytop", name="Rockytop", region="delta_front", by=("cluster", 461)),
    dict(n=13, slug="knob_mountain", name="Knob Mountain", region="delta_front", by=("cluster", 536)),
    dict(n=14, slug="tenby_berea", name="Tenby/Berea", region="upper_fan_belva", by=("site", "tenby_berea")),
    dict(n=15, slug="belva_crater", name="Belva Crater", region="upper_fan_belva", by=("cluster", 784)),
    dict(n=16, slug="wild_basin_lake", name="Wild Basin Lake", region="upper_fan_belva", by=("window", (838, 853))),
    dict(n=17, slug="tuxedo_park", name="Tuxedo Park", region="upper_fan_belva", by=("cluster", 896),
         shift_m=(-20.0, 0.0)),
    dict(n=18, slug="airey_hill", name="Airey Hill", region="upper_fan_belva", by=("site", "airey_hill")),
    dict(n=19, slug="bunsen_peak", name="Bunsen Peak", region="margin_unit_neretva_vallis", by=("cluster", 1066)),
    dict(n=20, slug="overlook_mountain", name="Overlook Mountain", region="margin_unit_neretva_vallis",
         by=("site", "overlook_mountain")),
    dict(n=21, slug="bright_angel", name="Bright Angel", region="margin_unit_neretva_vallis", by=("site", "bright_angle")),
    dict(n=22, slug="cheyava_falls", name="Cheyava Falls", region="margin_unit_neretva_vallis", by=("cluster", 1185)),
    dict(n=23, slug="plateau_point", name="Plateau Point", region="margin_unit_neretva_vallis", by=("window", (1252, 1263))),
    dict(n=24, slug="pico_turquino", name="Pico Turquino", region="margin_unit_neretva_vallis", by=("cluster", 1307)),
    dict(n=25, slug="rio_chiquito", name="Rio Chiquito", region="margin_unit_neretva_vallis", by=("site", "rio_chiquito")),
    dict(n=26, slug="south_arm", name="South Arm", region="crater_rim", by=("cluster", 1359)),
    dict(n=27, slug="witch_hazel_hill", name="Witch Hazel Hill", region="crater_rim", by=("cluster", 1363)),
    dict(n=28, slug="bell_island", name="Bell Island", region="crater_rim", by=("cluster", 1451)),
    dict(n=29, slug="taylorfjellet", name="Taylorfjellet", region="crater_rim", by=("cluster", 1606)),
    dict(n=30, slug="origny", name="Origny", region="crater_rim_west", by=("cluster", 1781)),
    dict(n=31, slug="vingeanne", name="Vingeanne", region="crater_rim_west", by=("cluster", 1817)),
    dict(n=32, slug="groloy", name="Groloy", region="crater_rim_west", by=("site", "groloy")),
    dict(n=33, slug="hippo_pools", name="Hippo Pools", region="crater_rim_west", by=("cluster", 1947),
         fallback=(1947, 2009)),
    dict(n=34, slug="marble_mountain", name="Marble Mountain", region="crater_rim_west", by=("site", "marble_mountain")),
)
EXCLUDED_CLUSTERS = ((1756, 1761),)          # not a block (Christian, 15 Oct)


def clusters(rows: Sequence[Dict[str, Any]], link_m: float = LINK_M, min_stops: int = MIN_STOPS,
             landing_sols: Sequence[int] = (13, 14)) -> List[Dict[str, Any]]:
    """Single-linkage clusters of the end-of-drive stops with a Navcam 360 (links <= ``link_m``) of at least
    ``min_stops``, in sol order: [{"sols": [...], "indices": [...], "E", "N" (mean)}].  The landing stops
    (``landing_sols``) join the cluster of the first stop after them within 2 ``link_m``."""
    idx = [i for i, r in enumerate(rows) if r["end_of_drive"] and r["pano"]]
    X = np.array([[rows[i]["E"], rows[i]["N"]] for i in idx], float)
    n = len(idx)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        d = np.hypot(*(X[i + 1:] - X[i]).T) if i + 1 < n else np.zeros(0)
        for j in np.nonzero(d <= link_m)[0]:
            parent[find(i)] = find(i + 1 + int(j))
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) < min_stops:
            continue
        ii = [idx[k] for k in g]
        out.append({"indices": sorted(ii), "sols": sorted(rows[k]["sol"] for k in ii)})
    out.sort(key=lambda c: c["sols"][0])
    land = [i for i, r in enumerate(rows) if r["sol"] in set(landing_sols)]
    if land and out:
        L = np.array([[rows[i]["E"], rows[i]["N"]] for i in land])
        for c in out:
            C = np.array([[rows[i]["E"], rows[i]["N"]] for i in c["indices"]])
            if np.min(np.hypot(*(L[:, None] - C[None]).reshape(-1, 2).T)) <= 2 * link_m:
                c["indices"] = sorted(set(c["indices"]) | set(land))
                c["sols"] = sorted(rows[k]["sol"] for k in c["indices"])
                break
    for c in out:
        c["E"] = float(np.mean([rows[i]["E"] for i in c["indices"]]))
        c["N"] = float(np.mean([rows[i]["N"] for i in c["indices"]]))
    return out


def densest(idx: Sequence[int], rows: Sequence[Dict[str, Any]], half_size_m: float = HALF_SIZE_M) -> List[int]:
    """The stops of ``idx`` inside the +-``half_size_m`` square about the stop that has the most of them inside its
    square (ties: the earliest) - a site's sol ranges can include stops the rover made on its way in or out."""
    if len(idx) <= 1:
        return list(idx)
    X = np.array([[rows[i]["E"], rows[i]["N"]] for i in idx])
    inside = (np.abs(X[:, None, 0] - X[None, :, 0]) <= half_size_m) & (np.abs(X[:, None, 1] - X[None, :, 1]) <= half_size_m)
    k = int(np.argmax(inside.sum(1)))
    return [i for i, m in zip(idx, inside[k]) if m]


def _in_square(r, E, N, h) -> bool:
    return abs(r["E"] - E) <= h and abs(r["N"] - N) <= h


def visits_of(sols: Sequence[int], gap: int = VISIT_GAP_SOLS) -> List[List[int]]:
    """Runs of sols split where the next is more than ``gap`` later; longest run (in sols) first, then by first sol."""
    s = sorted(set(int(x) for x in sols))
    if not s:
        return []
    runs, a, prev = [], s[0], s[0]
    for x in s[1:]:
        if x - prev > gap:
            runs.append([a, prev])
            a = x
        prev = x
    runs.append([a, prev])
    return sorted(runs, key=lambda r: (-(r[1] - r[0]), r[0]))


def build_blocks(rows: Sequence[Dict[str, Any]], site_table: Optional[Dict[str, Any]] = None,
                 zcam360: Optional[Sequence[Dict[str, Any]]] = None, half_size_m: float = HALF_SIZE_M,
                 defs: Sequence[Dict[str, Any]] = BLOCK_DEFS) -> List[Dict[str, Any]]:
    """
    The blocks of :data:`BLOCK_DEFS` from waypoint rows (:func:`mppp.maps.waypoints.waypoint_rows`).  ``site_table``:
    ``sites.json`` (default the package's) for the blocks defined by a site; ``zcam360``: [{"number", "sol", "name"}]
    the numbered Mastcam-Z 360s (``Z×`` counts; none without).  Returns one dict per block (number order) with
    "number", "slug", "key" (``blkNN_<slug>``), "name", "region", "kind", "legacy_site", "centre" {"easting_m",
    "northing_m", "lon", "lat"}, "half_size_m", "end_of_drive_sols", "navcam_360s", "zcam_360s", "visits",
    "defining_sols", "note".
    """
    from .waypoints import lonlat_fit
    if site_table is None:
        from ..sfm.sites import load_site_table
        site_table = load_site_table()
    to_ll, _ = lonlat_fit(rows)
    cl = clusters(rows)
    out = []
    for d in defs:
        kind, arg = d["by"]
        note = []
        idx: List[int] = []
        if kind == "cluster":
            hit = [c for c in cl if int(arg) in c["sols"]
                   and not any(a <= c["sols"][0] and c["sols"][-1] <= b for a, b in EXCLUDED_CLUSTERS)]
            if hit:
                idx = hit[0]["indices"]
            elif d.get("fallback"):
                a, b = d["fallback"]
                idx = densest([i for i, r in enumerate(rows) if r["end_of_drive"] and a <= r["sol"] <= b], rows, half_size_m)
                note.append(f"no cluster of >= {MIN_STOPS} in the waypoints: the end-of-drive stops of sols {a}-{b}")
                kind = "window"
            else:
                note.append(f"no cluster with a stop on sol {arg}")
        elif kind == "window":
            a, b = arg
            idx = densest([i for i, r in enumerate(rows) if r["end_of_drive"] and a <= r["sol"] <= b], rows, half_size_m)
        elif kind == "site":
            from ..sfm.sites import in_sol_ranges
            v = site_table["sites"].get(arg)
            if v is None:
                note.append(f"site {arg!r} not in the site table")
            else:
                from ..sfm.sites import sol_ranges
                main = [sol_ranges(v["sols"])[0]]                # the main visit places the square
                cand = [i for i, r in enumerate(rows) if r["end_of_drive"] and in_sol_ranges(r["sol"], main)]
                if not cand:
                    cand = [i for i, r in enumerate(rows) if in_sol_ranges(r["sol"], v["sols"])]
                idx = densest(cand, rows, half_size_m)
        extra = [i for i, r in enumerate(rows) if r["sol"] in set(d.get("extra_sols") or ())]
        idx = sorted(set(idx) | set(extra))
        if not idx:
            out.append({"number": d["n"], "slug": d["slug"], "key": f"blk{d['n']:02d}_{d['slug']}", "name": d["name"],
                        "region": d["region"], "kind": kind, "missing": True,
                        "note": "; ".join(note + ["no waypoint defines it"])})
            continue
        E = float(np.mean([rows[i]["E"] for i in idx]))
        N = float(np.mean([rows[i]["N"] for i in idx]))
        sh = d.get("shift_m")
        if sh:
            E, N = E + float(sh[0]), N + float(sh[1])
            note.append(f"square shifted {sh[0]:+g} m east, {sh[1]:+g} m north")
        inside = [r for r in rows if _in_square(r, E, N, half_size_m)]
        drop = set(d.get("drop_sols") or ())
        sols = [r["sol"] for r in inside if r["sol"] not in drop]
        if drop:
            note.append(f"sols {sorted(drop)} left out of the visits")
        eod = sorted({r["sol"] for r in inside if r["end_of_drive"] and r["sol"] not in drop})
        # a Navcam 360 at a stop: end of drive, or the landing's site increment (sol 13)
        n360 = sum(1 for r in inside if r["pano"] and r["kind"] in ("end_of_drive", "site_increment") and r["sol"] not in drop)
        vis = visits_of(sols)
        z = []
        if zcam360:
            z = [dict(x) for x in zcam360 if any(a <= int(x["sol"]) <= b for a, b in vis)]
        lon, lat = to_ll(E, N)
        legacy = arg if d["by"][0] == "site" else _legacy_match(site_table, vis)
        out.append({"number": d["n"], "slug": d["slug"], "key": f"blk{d['n']:02d}_{d['slug']}", "name": d["name"],
                    "region": d["region"], "kind": kind, "legacy_site": legacy,
                    "centre": {"easting_m": E, "northing_m": N, "lon": lon, "lat": lat},
                    "half_size_m": float(half_size_m), "end_of_drive_sols": eod, "navcam_360s": int(n360),
                    "zcam_360s": [{"number": x.get("number"), "sol": int(x["sol"]), "name": x.get("name")} for x in z],
                    "visits": vis, "defining_sols": sorted({rows[i]["sol"] for i in idx}), "note": "; ".join(note)})
    return out


def _legacy_match(site_table: Dict[str, Any], visits: Sequence[Sequence[int]]) -> Optional[str]:
    """The sites.json site whose sol ranges overlap the block's visits most (None if none does)."""
    from ..sfm.sites import sol_ranges
    best, score = None, 0
    for k, v in site_table["sites"].items():
        try:
            rs = sol_ranges(v["sols"])
        except Exception:                                          # noqa: BLE001
            continue
        s = sum(max(0, min(b, d) - max(a, c) + 1) for a, b in rs for c, d in visits)
        if s > score:
            best, score = k, s
    return best


def block_lines(blocks: Sequence[Dict[str, Any]]) -> List[str]:
    """Printable list: number, name, N×, Z×, visits, region."""
    out = []
    for b in blocks:
        if b.get("missing"):
            out.append(f"{b['number']:2d} {b['name']:22s} MISSING ({b.get('note')})")
            continue
        z = len(b.get("zcam_360s") or [])
        out.append(f"{b['number']:2d} {b['name']:22s} N×{b['navcam_360s']:<2d}" + (f" Z×{z}" if z else "    ")
                   + f"  sols {', '.join(f'{a}-{c}' for a, c in b['visits'])}  ({b['kind']}"
                   + (f", {b['legacy_site']}" if b.get("legacy_site") else "") + ")" + (f"  {b['note']}" if b.get("note") else ""))
    return out
