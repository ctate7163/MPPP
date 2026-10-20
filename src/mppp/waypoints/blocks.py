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


def plot_site_map(sites: Dict[str, Any], waypoints: Union[PathLike, Dict[str, Any]], path: PathLike,
                  terrain_dir: Optional[PathLike] = None, highlight: Sequence[str] = (),
                  margin_deg: float = 0.004, step_deg: float = 0.0002, title: Optional[str] = None,
                  labels: bool = True) -> Path:
    """
    Map of the sites over the rover traverse: every waypoint as a dot joined in sol order (the traverse), each site's
    waypoints as a square with its label, on a shaded-relief background from quantized-mesh tiles in
    ``terrain_dir`` (:mod:`mppp.waypoints.terrain`; without tiles a plain background).  ``highlight`` names the
    sites drawn in the warm colour (e.g. the group being run).  Longitude east, latitude north in degrees.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .table import lonlat_of
    d = waypoints if isinstance(waypoints, dict) else json.loads(Path(waypoints).read_text(encoding="utf-8"))
    trav = sorted((int((f.get("properties") or {}).get("sol", -1)), *lonlat_of(f)) for f in d.get("features") or []
                  if lonlat_of(f) is not None and (f.get("properties") or {}).get("sol") is not None)
    T = np.array(trav, float)
    sw = site_waypoints(sites, d)
    lon0, lon1 = T[:, 1].min() - margin_deg, T[:, 1].max() + margin_deg
    lat0, lat1 = T[:, 2].min() - margin_deg, T[:, 2].max() + margin_deg
    aspect = 1.0 / np.cos(np.deg2rad(T[:, 2].mean()))
    fig = plt.figure(figsize=(14, 11 * (lat1 - lat0) * aspect / (lon1 - lon0) + 0.8))
    ax = fig.add_axes([0.05, 0.1, 0.72, 0.82])                   # the map; the numbered list to its right
    lst = fig.add_axes([0.78, 0.1, 0.21, 0.82]); lst.axis("off")
    if terrain_dir and Path(terrain_dir).is_dir():
        from .terrain import shaded_relief, terrain_grid
        gx, gy, Z = terrain_grid(terrain_dir, (lon0, lon1, lat0, lat1), step_deg)
        if np.isfinite(Z).any():
            sr = shaded_relief(gx, gy, Z)
            ax.imshow(sr["rgb"], extent=sr["extent"], origin="lower", interpolation="bilinear", aspect=aspect)
            ax.contour(gx, gy, np.where(np.isnan(Z), np.nanmean(Z), Z), levels=np.arange(-2700, -1000, 20),
                            colors="k", linewidths=0.25, alpha=0.2)
    else:
        ax.set_facecolor("#efede8")
    ax.plot(T[:, 1], T[:, 2], color="#555555", lw=0.8, alpha=0.8, label="traverse (waypoints in sol order)")
    ax.scatter(T[:, 1], T[:, 2], s=3, color="#333333", alpha=0.6, edgecolor="none")
    hl = set(highlight)
    from .sites import site_label
    import matplotlib.patheffects as pe
    listed = []
    for k, (name, s) in enumerate(sorted(sw.items(), key=lambda kv: kv[1]["first_sol"]), 1):
        c = "#eb6834" if name in hl else "#2a78d6"
        lo, la = np.array(s["lon"]), np.array(s["lat"])
        half = 30.0 / (MARS_RADIUS_M * np.deg2rad(1.0))         # a +-30 m square in degrees of latitude
        cx, cy = s["centre"]
        ax.add_patch(plt.Rectangle((cx - half * aspect, cy - half), 2 * half * aspect, 2 * half, fill=False,
                                   edgecolor=c, lw=1.2))
        ax.scatter(lo, la, s=14, color=c, edgecolor="white", linewidth=0.4, zorder=5)
        if labels:
            ax.annotate(str(k), (cx + half * aspect, cy + half), fontsize=6.5, color=c, ha="left", va="bottom",
                        xytext=(1, 1), textcoords="offset points", fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        listed.append((k, name, s["first_sol"], c))
    if labels and listed:
        ncol = 1 if len(listed) <= 30 else 2
        per = int(np.ceil(len(listed) / ncol))
        for j in range(ncol):
            for i, (k, name, fs, c) in enumerate(listed[j * per:(j + 1) * per]):
                lst.text(0.02 + 0.5 * j, 0.98 - i * (0.96 / per), f"{k:>2}  {site_label(name)} ({fs})", fontsize=6.5,
                         color=c, transform=lst.transAxes, va="top", family="DejaVu Sans Mono" if False else None)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(aspect)
    ax.set_xlabel("longitude east  °")
    ax.set_ylabel("latitude  °")
    ax.set_title(title or f"Sites (blocks of rover stations) along the Perseverance traverse: {len(sw)} sites, "
                          f"{len(T)} waypoints to sol {int(T[:, 0].max())}")
    if hl:
        ax.scatter([], [], color="#eb6834", label="sites of the run"); ax.scatter([], [], color="#2a78d6", label="other sites")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.savefig(path, dpi=200)
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
