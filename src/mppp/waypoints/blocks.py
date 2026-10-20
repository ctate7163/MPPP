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


# ------------------------------------------------------------------------------------------- plots (0.90)
def site_waypoints(sites: Dict[str, Any], waypoints: Union[PathLike, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per site: its end-of-drive waypoints (lon, lat, sol) within its sol ranges and their centre.  ``sites`` is
    the ``sites`` entry of the site table ({name: {"sols": ...}})."""
    from .sites import sol_ranges
    from .table import lonlat_of
    d = waypoints if isinstance(waypoints, dict) else json.loads(Path(waypoints).read_text(encoding="utf-8"))
    feats = []
    for f in d.get("features") or []:
        p = f.get("properties") or {}
        ll = lonlat_of(f)
        if ll is None or p.get("sol") is None:
            continue
        feats.append((int(p["sol"]), ll[0], ll[1], str(p.get("final", "")).lower() == "y"))
    out = {}
    for name, v in sites.items():
        try:
            rs = sol_ranges(v.get("sols", v) if isinstance(v, dict) else v)
        except (ValueError, TypeError):
            continue
        pts = [(s, lo, la) for s, lo, la, eod in feats if any(a <= s <= b for a, b in rs)]
        if not pts:
            continue
        A = np.array(pts, float)
        out[name] = {"sols": A[:, 0].astype(int).tolist(), "lon": A[:, 1].tolist(), "lat": A[:, 2].tolist(),
                     "centre": [float(np.median(A[:, 1])), float(np.median(A[:, 2]))], "first_sol": int(A[:, 0].min())}
    return out


def plot_site_map(blocks: Optional[Dict[str, Any]], waypoints: Union[PathLike, Dict[str, Any]], path: PathLike,
                  terrain_dir: Optional[PathLike] = None, panos: Sequence[Sequence[Any]] = (),
                  samples: Sequence[Sequence[Any]] = (), highlight: Sequence[str] = (), dpi: int = 300,
                  box_crater: Sequence[float] = (77.1, 78.1, 17.9, 18.8),
                  box_traverse: Optional[Sequence[float]] = None, title_c: Optional[str] = None) -> Path:
    """
    The traverse map of the site study (figure 2 of 16 Oct 2026, ``studies/m20_sites``): **A** Jezero crater with the
    traverse and the box of B, **B** the traverse with the blocks of rover stations as orange ±30 m squares numbered
    by first sol (white labels, spread with adjustText when installed), end-of-drive stops, the Mastcam-Z numbered
    360s (triangles, ``panos`` rows ``(number, [sols], name, zooms)``) and sample cores (diamonds, ``samples`` rows
    ``(..., sol)``) on the tinted hillshade of the terrain tiles in ``terrain_dir`` with 100 m contours and a scale
    bar; **C** the block table (number, name, N× end-of-drive Navcam 360s, Z× numbered Mastcam-Z 360s, visits).
    ``blocks`` is the site table's ``official_blocks`` (default: the shipped table); ``highlight`` names the sites
    whose labels get the warm colour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from .sites import load_site_table
    from .table import lonlat_of
    from .terrain import terrain_grid
    try:
        from adjustText import adjust_text
    except ImportError:                                            # pragma: no cover - optional
        adjust_text = None
    C1, C2, C3, C4, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
    FS = 1.15
    plt.rcParams.update({"font.size": 10 * FS})
    CMAP = plt.get_cmap("gist_earth")
    if blocks is None:
        blocks = load_site_table()["official_blocks"]
    d = waypoints if isinstance(waypoints, dict) else json.loads(Path(waypoints).read_text(encoding="utf-8"))
    W = sorted(((int(p.get("sol")), *lonlat_of(f), str(p.get("final", "")).lower())
                for f in d.get("features") or [] for p in [f.get("properties") or {}]
                if lonlat_of(f) is not None and p.get("sol") is not None), key=lambda t: t[0])
    S = np.array([w[0] for w in W]); lon = np.array([w[1] for w in W]); lat = np.array([w[2] for w in W])
    F = np.array([w[3] for w in W]); isE = F == "y"
    COS = np.cos(np.deg2rad(lat.mean()))
    M_LON, M_LAT = np.deg2rad(1) * MARS_RADIUS_M * COS, np.deg2rad(1) * MARS_RADIUS_M
    if box_traverse is None:
        box_traverse = (lon.min() - 0.006, lon.max() + 0.012, lat.min() - 0.006, lat.max() + 0.006)
    BOX_A, BOX_B = tuple(box_crater), tuple(box_traverse)
    rows = sorted(((int(k) if str(k).isdigit() else k, v) for k, v in blocks.items()),
                  key=lambda kv: min(min(x) for x in (kv[1].get("visits") or [[0]])))
    hl = set(highlight)

    def posll(sol):
        w = np.where(S <= sol)[0]
        i = w[-1] if len(w) else 0
        return lon[i], lat[i]
    ZP = np.array([posll(r[1][0]) for r in panos]).reshape(-1, 2)
    SP = np.array([posll(r[-1]) for r in samples]).reshape(-1, 2)

    def draw_terrain(ax, box, step, ci=100):
        if not (terrain_dir and Path(terrain_dir).is_dir()):
            ax.set_facecolor("#efede8")
            return None
        from matplotlib.colors import LightSource
        gx, gy, Z = terrain_grid(terrain_dir, box, step)
        if not np.isfinite(Z).any():
            ax.set_facecolor("#efede8")
            return None
        norm = plt.Normalize(np.nanpercentile(Z, 1), np.nanpercentile(Z, 99))
        dx = np.deg2rad(gx[1] - gx[0]) * MARS_RADIUS_M * COS; dy = np.deg2rad(gy[1] - gy[0]) * MARS_RADIUS_M
        Zf = np.where(np.isnan(Z), np.nanmean(Z), Z)
        hs = LightSource(azdeg=315, altdeg=40).hillshade(Zf[::-1], vert_exag=3, dx=dx, dy=dy)[::-1]
        rgb = CMAP(norm(Zf))[..., :3]; rgb = 0.55 * rgb + 0.45 * 0.86
        out = np.clip(rgb * (0.35 + 0.65 * hs)[..., None], 0, 1); out = np.where(np.isnan(Z)[..., None], 0.93, out)
        ax.imshow(out, extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower", interpolation="bilinear", zorder=0)
        lev = np.arange(np.floor(np.nanmin(Zf) / ci) * ci, np.nanmax(Zf) + ci, ci)
        ax.contour(gx, gy, Zf, levels=lev, colors="k", linewidths=0.3, alpha=0.25, zorder=1)
        return norm

    def scalebar(ax, box, metres, label):
        dl = metres / M_LON; y0 = box[2] + 0.06 * (box[3] - box[2]); x0 = box[0] + 0.05 * (box[1] - box[0])
        ax.plot([x0, x0 + dl], [y0, y0], color=INK, lw=3, zorder=9, solid_capstyle="butt")
        ax.text(x0 + dl / 2, y0 + 0.025 * (box[3] - box[2]), label, ha="center", fontsize=8 * FS, zorder=9,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))

    Wd, Ht = 18.0, 7.4
    rect = lambda x, y, w, h: [x / Wd, y / Ht, w / Wd, h / Ht]                       # noqa: E731
    rA = (BOX_A[1] - BOX_A[0]) * COS / (BOX_A[3] - BOX_A[2]); rB = (BOX_B[1] - BOX_B[0]) * COS / (BOX_B[3] - BOX_B[2])
    left, gap, cbar, top = 0.85, 0.95, 1.25, Ht - 0.45
    hh = (Wd - left - gap - cbar) / (rA + rB)
    a = (left, top - hh, rA * hh, hh); b = (left + rA * hh + gap, top - hh, rB * hh, hh)
    fig = plt.figure(figsize=(Wd, Ht), facecolor="#fcfcfb")
    # ---- A: the crater
    ax = fig.add_axes(rect(*a))
    draw_terrain(ax, (BOX_A[0] - 0.02, BOX_A[1] + 0.02, BOX_A[2] - 0.02, BOX_A[3] + 0.02), 0.003)
    ax.plot(lon, lat, color="white", lw=3.2); ax.plot(lon, lat, color=C1, lw=1.6)
    ax.add_patch(plt.Rectangle((BOX_B[0], BOX_B[2]), BOX_B[1] - BOX_B[0], BOX_B[3] - BOX_B[2], fill=False, ec=INK, lw=1.2, zorder=5))
    ax.text(BOX_B[0] + 0.01, BOX_B[3] + 0.012, "B", fontsize=11 * FS, fontweight="bold", color=INK, zorder=6)
    ax.plot(lon[0], lat[0], marker="*", ms=13, color=C4, mec=INK, mew=0.6, zorder=7)
    ax.annotate("Landing (OEB)", (lon[0], lat[0]), xytext=(8, -14), textcoords="offset points", fontsize=8 * FS, color=INK)
    ax.set_xlim(BOX_A[0], BOX_A[1]); ax.set_ylim(BOX_A[2], BOX_A[3]); ax.set_aspect(1 / COS)
    ax.set_title(f"A. Jezero crater and the Perseverance traverse (sols {S.min()}–{S.max()})", loc="left", fontsize=10 * FS, color=INK)
    ax.set_xlabel("Longitude [°E]"); ax.set_ylabel("Latitude [°N]")
    dl = 10000 / M_LON; x0, y0 = BOX_A[0] + 0.05, BOX_A[2] + 0.04
    ax.plot([x0, x0 + dl], [y0, y0], color=INK, lw=3); ax.text(x0 + dl / 2, y0 + 0.015, "10 km", ha="center", fontsize=8 * FS)
    # ---- B: the traverse with the blocks
    ax = fig.add_axes(rect(*b))
    nB = draw_terrain(ax, BOX_B, 0.00005)
    ax.plot(lon, lat, color="white", lw=3.0, zorder=2); ax.plot(lon, lat, color=C1, lw=1.4, zorder=3)
    ax.scatter(lon[isE], lat[isE], s=10, color="white", ec=C1, lw=0.7, zorder=4)
    m = F == "m"; ax.scatter(lon[m], lat[m], s=8, marker="s", color="white", ec=C1, lw=0.6, zorder=4)
    if len(ZP):
        ax.scatter(ZP[:, 0], ZP[:, 1], marker="^", s=46, color=C4, ec=INK, lw=0.5, zorder=7)
    if len(SP):
        ax.scatter(SP[:, 0], SP[:, 1], marker="D", s=30, color=C3, ec=INK, lw=0.5, zorder=7)
    txt = []
    half = 30.0
    for k, o in rows:
        c = o.get("centre") or {}
        lo, la = float(c.get("lon")), float(c.get("lat"))
        if not (BOX_B[0] <= lo <= BOX_B[1] and BOX_B[2] <= la <= BOX_B[3]):
            continue
        hw, hh2 = float(o.get("half_size_m", half)) / M_LON, float(o.get("half_size_m", half)) / M_LAT
        ax.add_patch(plt.Rectangle((lo - hw, la - hh2), 2 * hw, 2 * hh2, fill=False, ec=C2, lw=1.6, zorder=6))
        ec = C2
        txt.append(ax.text(lo, la, str(k), fontsize=8.5 * FS, fontweight="bold",
                           color=(C2 if o.get("site") in hl else INK), zorder=8,
                           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ec, lw=0.8, alpha=0.92)))
    ax.set_xlim(BOX_B[0], BOX_B[1]); ax.set_ylim(BOX_B[2], BOX_B[3]); ax.set_aspect(1 / COS)
    if adjust_text is not None and txt:
        adjust_text(txt, ax=ax, expand=(1.35, 1.6), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6), max_move=(60, 60))
    h = [plt.Line2D([], [], color=C1, lw=1.4, label="Traverse"),
         plt.Line2D([], [], marker="^", color=C4, mec=INK, ls="none", ms=7, label="Mastcam-Z numbered 360"),
         plt.Line2D([], [], marker="D", color=C3, mec=INK, ls="none", ms=6, label="Sample core"),
         plt.Line2D([], [], marker="s", mfc="none", mec=C2, mew=2, ls="none", ms=11, label="Block of rover stations")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.6, 0.01), fontsize=7.6 * FS, framealpha=0.92, facecolor="white", edgecolor="#d6d5d0")
    bg = "coarse HiRISE-DTM-based terrain, ~70 m grid" if nB is not None else "no terrain tiles"
    ax.set_title("B. Blocks of rover stations, numbered by first sol. Background: " + bg, loc="left", fontsize=9.5 * FS, color=INK)
    ax.set_xlabel("Longitude [°E]")
    scalebar(ax, BOX_B, 1000, "1 km")
    if nB is not None:
        cax = fig.add_axes(rect(b[0] + b[2] + 0.12, b[1], 0.2, b[3]))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nB, cmap=CMAP), cax=cax); cb.set_label("Elevation [m] (tint)", fontsize=8 * FS)
        cb.ax.tick_params(labelsize=7 * FS)
    # ---- C: the table
    ax = fig.add_axes(rect(0.85, 0.1, Wd - 1.2, a[1] - 1.05)); ax.axis("off")
    NW = max(len(str(o.get("name", ""))) for _, o in rows)
    lines = []
    for k, o in rows:
        runs = sorted((min(v), max(v)) for v in (o.get("visits") or []))
        lab = ", ".join(f"{x}" if x == y else f"{x}–{y}" for x, y in runs) + ("+" if runs and runs[-1][1] >= S.max() else "")
        nz = len(o.get("mastcam_z_360") or [])
        cnt = f"N×{int(o.get('n_navcam_360') or 0):>2}" + (f", Z×{nz}" if nz else "")
        lines.append(f"{k:>2}  {str(o.get('name', ''))[:34]:<{NW}} {cnt:<10} sols {lab}")
    ncol, per = 3, int(np.ceil(len(lines) / 3))
    for k in range(ncol):
        ax.text(k / ncol, 1.0, "\n".join(lines[per * k:per * (k + 1)]), va="top", ha="left", family="DejaVu Sans Mono",
                fontsize=7.2 * FS, color=INK, transform=ax.transAxes, linespacing=1.25)
    ax.set_title(title_c or ("C. Blocks of rover stations: number, name, N× = end-of-drive Navcam 360s and Z× = numbered "
                             "Mastcam-Z 360s within ±30 m of the block centre, and the sols the rover ended inside that square"),
                 loc="left", fontsize=10 * FS, color=INK)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return Path(path)


MARS_RADIUS_M = 3389500.0


def manifest_lmst_temperature(manifest: PathLike, instruments: Sequence[str] = ("NL", "NR")) -> Dict[str, np.ndarray]:
    """(lmst_h, camera temperature, sol) of the images of a processing manifest (``mppp_manifest_<version>.json``)
    whose camera group starts with one of ``instruments``."""
    m = json.loads(Path(manifest).read_text(encoding="utf-8"))
    ims = m.get("images") or []
    ims = ims if isinstance(ims, list) else list(ims.values())
    rows = []
    for r in ims:
        g = str(r.get("camera_group") or "")
        if not any(g.startswith(i) for i in instruments):
            continue
        try:
            rows.append((float(r["lmst_h"]), float(r["camera_temperature_degC"]), int(r["sol"])))
        except (KeyError, TypeError, ValueError):
            continue
    A = np.array(rows, float).reshape(-1, 3)
    return {"lmst_h": A[:, 0], "T": A[:, 1], "sol": A[:, 2]}


def plot_lmst_histograms(data: Dict[str, Dict[str, np.ndarray]], path: PathLike, bin_h: float = 0.5,
                         ncols: int = 4, title: Optional[str] = None) -> Path:
    """
    One panel per site: the histogram of local mean solar time (LMST) of its Navcam images (bars) with the camera
    temperature of every image against LMST on a secondary (right, warm) axis.  ``data`` = {site label:
    :func:`manifest_lmst_temperature`}.  Temperature axes share one range so the sites compare.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(data)
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows + 0.5), squeeze=False, sharex=True)
    allT = np.concatenate([d["T"] for d in data.values() if len(d["T"])]) if n else np.zeros(1)
    Tlim = (np.floor(allT.min() / 5) * 5 - 2, np.ceil(allT.max() / 5) * 5 + 2) if len(allT) else (-40, 0)
    edges = np.arange(6, 19 + bin_h, bin_h)
    for k, name in enumerate(names):
        ax = axs[k // ncols, k % ncols]
        d = data[name]
        ax.hist(d["lmst_h"], bins=edges, color="#2a78d6", alpha=0.75)
        ax.set_title(f"{name}  ({len(d['lmst_h'])} images, sols {int(d['sol'].min()) if len(d['sol']) else '-'}"
                     f"–{int(d['sol'].max()) if len(d['sol']) else '-'})", fontsize=8)
        ax.axvline(12.0, color="k", ls="--", lw=0.7, alpha=0.5)
        ax2 = ax.twinx()
        ax2.scatter(d["lmst_h"], d["T"], s=5, color="#eb6834", alpha=0.6, edgecolor="none")
        ax2.set_ylim(*Tlim)
        ax2.tick_params(axis="y", colors="#eb6834", labelsize=7)
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
        if k % ncols == ncols - 1 or k == n - 1:
            ax2.set_ylabel("camera T  °C", color="#eb6834", fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel("images", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(n, nrows * ncols):
        axs[k // ncols, k % ncols].set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("LMST  h", fontsize=8)
    ax.set_xlim(6, 19)
    fig.suptitle(title or "Local mean solar time of the Navcam images per site, with camera temperature", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
