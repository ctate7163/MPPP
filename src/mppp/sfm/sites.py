"""
The Navcam sites and their notebook 03 alignments (v0p40).

``SITES`` is the default site list of notebook 03 (name -> sol range of the block), read from the stable site
definitions file ``mppp/data/sites.json`` (v0p43; :func:`load_site_table`).  :func:`discover_scapes` walks ``SCAPES_ROOT`` (``D:/scapes/colmap``) for the WORK folders notebook 03
made (v0p53: ``mars2020_sol_<first sol 0000>_<site>_colmap``, and ``..._colmap_zcam`` with Mastcam-Z) and returns the ones that hold a finished
alignment, labelled and in sol order, for notebooks 04 (camera models) and 05 (error analysis).  :func:`scan_scapes`
returns every folder with the reason it was left out.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

SITES_FILE = Path(__file__).resolve().parents[1] / "data" / "sites.json"     # v0p43: the stable site definitions


def sol_ranges(sols: Any) -> List[Tuple[int, int]]:
    """
    v0p75: the sol ranges of a site's ``"sols"`` entry as a list of (first, last): ``[a, b]`` is one range,
    ``[[a, b], [c, d], ...]`` several - the visits of a site the rover left and came back to (Chal Rocks Sid: sols
    361-378 and 91-101).  **The first range is the main visit**: it names the WORK folder (``mars2020_sol_<its first
    sol>_...``) and is the block's sol range for everything that wants one number; the others are the other visits,
    in any order.  Raises ``ValueError`` for anything else.
    """
    if isinstance(sols, (list, tuple)) and len(sols) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                                   for x in sols):
        return [(int(sols[0]), int(sols[1]))]
    if isinstance(sols, (list, tuple)) and sols and all(
            isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                                 for x in r) for r in sols):
        return [(int(r[0]), int(r[1])) for r in sols]
    raise ValueError(f"'sols' must be [first, last] or [[first, last], [first, last], ...], not {sols!r}")


def sol_span(sols: Any) -> Tuple[int, int]:
    """v0p75: (first sol of the main visit, last sol of all visits) of a ``"sols"`` entry - what the older code
    (``SITES[name]``, the folder name, ``work_folder``) wants as one range.  The first sol is the **main visit's**
    (the first range listed), so a site that gained an earlier visit keeps its folder name."""
    rs = sol_ranges(sols)
    return int(rs[0][0]), max(int(b) for _, b in rs)


def in_sol_ranges(sol: int, sols: Any) -> bool:
    """v0p75: is ``sol`` in any range of a ``"sols"`` entry (or a list of (first, last))?"""
    return any(int(a) <= int(sol) <= int(b) for a, b in sol_ranges(sols))


def site_sol_ranges(table: Dict[str, Any], site: str) -> List[Tuple[int, int]]:
    """v0p75: the sol ranges of ``site`` in a site table (:func:`load_site_table`), main visit first."""
    v = table["sites"][site]
    return sol_ranges(v.get("sols") if isinstance(v, dict) else v)


def load_site_table(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """v0p43: the site definitions file (``mppp/data/sites.json`` by default): ``{"sites": {name: {"sols": [a, b],
    "label", "zcam", "note", "no_mask_inference_at", "settings"}}, "groups": {name: [site, ...]}}``.  ``zcam``
    (v0p53, a list of Mastcam-Z zooms in mm, 34 / 48 / 63 / 79 / 110; before: ``"zcam34": true``): the site also has a Navcam +
    Mastcam-Z block with those zooms (``mars2020_sol_<sol>_<site>_colmap_zcam``).  v0p75: ``"sols"`` may be a list of
    ranges, ``[[361, 378], [91, 101]]`` (:func:`sol_ranges`; the first names the folder)."""
    f = Path(path) if path else SITES_FILE
    d = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(d.get("sites"), dict):
        raise ValueError(f"{f}: no 'sites' object")
    for k, v in d["sites"].items():
        sols = v.get("sols") if isinstance(v, dict) else v
        try:
            sol_ranges(sols)
        except ValueError as e:
            raise ValueError(f"{f}: site {k!r}: {e}") from None
    return d


def load_sites(path: Optional[PathLike] = None) -> Dict[str, Tuple[int, int]]:
    """v0p43: name -> (first sol, last sol) from the site definitions file (notebook 03 ``SITES``).  v0p75: for a site
    with several visits this is :func:`sol_span` (first sol of the main visit, last sol of all); the visits themselves
    are :func:`load_site_ranges`."""
    d = load_site_table(path)
    out = {}
    for k, v in d["sites"].items():
        sols = v.get("sols") if isinstance(v, dict) else v
        out[k] = sol_span(sols)
    return out


def load_site_ranges(path: Optional[PathLike] = None) -> Dict[str, List[Tuple[int, int]]]:
    """v0p75: name -> [(first, last), ...] (main visit first) from the site definitions file."""
    d = load_site_table(path)
    return {k: sol_ranges(v.get("sols") if isinstance(v, dict) else v) for k, v in d["sites"].items()}


ZCAM_ZOOMS = (34, 48, 63, 79, 110) # v0p53: the Mastcam-Z zooms (mm) MPPP aligns


def site_zooms(v: Any) -> List[int]:
    """v0p53: the Mastcam-Z zooms of one site definition: ``"zcam": [34, 48]``; the older ``"zcam34": true``
    (``"zcam48"``, ``"zcam63"``) is still read."""
    if not isinstance(v, dict):
        return []
    z = v.get("zcam")
    if isinstance(z, list):
        return sorted({int(x) for x in z})
    return [zz for zz in ZCAM_ZOOMS if v.get(f"zcam{zz}") is True]


def zcam_sites(table: Optional[Dict[str, Any]] = None, path: Optional[PathLike] = None,
               zoom: Optional[int] = None) -> List[str]:
    """v0p53: the sites with a Navcam + Mastcam-Z block (any zoom, or ``zoom``), in file order."""
    t = table if table is not None else load_site_table(path)
    return [k for k, v in t["sites"].items() if site_zooms(v) and (zoom is None or int(zoom) in site_zooms(v))]


def zcam34_sites(table: Optional[Dict[str, Any]] = None, path: Optional[PathLike] = None) -> List[str]:
    """v0p50: the sites with Mastcam-Z 34 mm (v0p53: :func:`zcam_sites` with ``zoom=34``)."""
    return zcam_sites(table, path, 34)


def site_group(name: str, path: Optional[PathLike] = None) -> List[str]:
    """v0p43: the sites of a group in the site definitions file (e.g. ``"navcam_consensus"``); v0p53: a site listed
    twice is used once."""
    groups = load_site_table(path).get("groups", {})
    if name not in groups:
        raise KeyError(f"no site group {name!r}; groups: {sorted(groups)}")
    return list(dict.fromkeys(groups[name]))


def validate_site_table(path: Optional[PathLike] = None,
                        known_settings: Optional[Iterable[str]] = None) -> Tuple[List[str], List[str]]:
    """v0p43.3: (errors, warnings) of a site definitions file - what ``scripts/check_sites.py`` prints.  Errors stop
    the runs (bad JSON, missing or reversed sol ranges, unknown sites in groups, unreadable stations); warnings do not
    (overlapping sol ranges, settings that are not notebook 03 settings, odd site names)."""
    import re
    from ..config import parse_stations
    f = Path(path) if path else SITES_FILE
    err: List[str] = []
    warn: List[str] = []
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"{f} does not exist"], []
    except json.JSONDecodeError as e:
        lines = f.read_text(encoding="utf-8").splitlines()
        show = "".join(f"\n      line {i}: {lines[i - 1].strip()[:140]}" for i in (e.lineno - 1, e.lineno)
                       if 0 < i <= len(lines))
        hint = (" (a comma missing at the end of the line before, or one too many after the last site?)"
                if "delimiter" in e.msg or "double quotes" in e.msg else "")
        return [f"not valid JSON at line {e.lineno}, column {e.colno}: {e.msg}{hint}{show}"], []
    sites = d.get("sites")
    if not isinstance(sites, dict) or not sites:
        return ["no 'sites' object"], []
    ver = d.get("version")
    if ver is not None and (not isinstance(ver, int) or ver > SITES_FILE_VERSION):
        warn.append(f"file version {ver!r}: this MPPP reads site files up to version {SITES_FILE_VERSION}")
    ranges = {}
    for name, v in sites.items():
        if not re.fullmatch(r"[a-z0-9_]+", name):
            warn.append(f"site {name!r}: use lower case letters, digits and _ (it becomes a folder name)")
        if name.endswith(("_colmap", "_zcam34", "_zcam")):
            err.append(f"site {name!r}: the name must not end in _colmap / _zcam (the folder suffixes)")
        if not isinstance(v, dict):
            err.append(f"site {name!r}: must be an object like {{\"sols\": [700, 712]}}")
            continue
        sols = v.get("sols")
        try:
            rs = sol_ranges(sols)
        except ValueError:
            err.append(f"site {name!r}: 'sols' must be [first, last] (two whole numbers) or a list of such ranges "
                       f"[[first, last], ...], not {sols!r}")
            continue
        if any(not isinstance(x, int) for r in (sols if isinstance(sols[0], list) else [sols]) for x in r):
            err.append(f"site {name!r}: sols {sols} must be whole numbers")
            continue
        bad = [r for r in rs if r[1] < r[0]]
        if bad:
            err.append(f"site {name!r}: sols {bad[0]} end before they start")
        for i, r1 in enumerate(rs):                       # v0p75: the visits of one site must not overlap
            for r2 in rs[i + 1:]:
                if r1[0] <= r2[1] and r2[0] <= r1[1]:
                    err.append(f"site {name!r}: its sol ranges {list(r1)} and {list(r2)} overlap")
        ranges[name] = rs
        if "zcam" in v:
            z = v["zcam"]
            if not (isinstance(z, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in z)):
                err.append(f"site {name!r}: 'zcam' must be a list of zooms like [34, 48] (or []), not {z!r}")
            elif set(z) - set(ZCAM_ZOOMS):
                err.append(f"site {name!r}: 'zcam' {z}: MPPP aligns the Mastcam-Z zooms {list(ZCAM_ZOOMS)} only")
        for old in ("zcam34", "zcam48", "zcam63", "zcam79", "zcam110"):
            if old in v:
                if not isinstance(v[old], bool):
                    err.append(f"site {name!r}: '{old}' must be true or false, not {v[old]!r} (v0p53: use 'zcam': [34, ...])")
                else:
                    warn.append(f"site {name!r}: '{old}' is read but is replaced by 'zcam': [34, 48, 63] (v0p53)")
        if "z34" in v:
            err.append(f"site {name!r}: 'z34' is now 'zcam': [34] (v0p53)")
        unknown = set(v) - {"sols", "label", "zcam", "zcam34", "zcam48", "zcam63", "zcam79", "zcam110", "z34", "note",
                            "no_mask_inference_at", "settings", "block"}
        if "block" in v:                                   # v0p80: the block-of-stations record (mppp.maps)
            err.extend(f"site {name!r}: {e}" for e in check_block_record(v["block"]))
        if unknown:
            warn.append(f"site {name!r}: unknown fields {sorted(unknown)} are ignored")
        try:
            parse_stations(v.get("no_mask_inference_at") or [])
        except ValueError as e:
            err.append(f"site {name!r}: no_mask_inference_at: {e}")
        st = v.get("settings") or {}
        if not isinstance(st, dict):
            err.append(f"site {name!r}: 'settings' must be an object like {{\"ATTITUDE_PRIOR_DEG\": 1.0}}")
        elif known_settings is not None:
            bad = sorted(k for k in st if k not in set(known_settings))
            if bad:
                warn.append(f"site {name!r}: settings {bad} are not notebook 03 settings (typo?) - they would be set "
                            f"but not used")
    groups = d.get("groups") or {}
    if not isinstance(groups, dict):
        err.append("'groups' must be an object of name: [site, ...]")
    else:
        for g, members in groups.items():
            if not isinstance(members, list):
                err.append(f"group {g!r}: must be a list of site names")
                continue
            missing = [m for m in members if m not in sites]
            if missing:
                err.append(f"group {g!r}: {missing} are not sites")
            dup = sorted({m for m in members if members.count(m) > 1})
            if dup:
                warn.append(f"group {g!r}: {dup} listed more than once (used once)")
            import re as _re
            mz = _re.fullmatch(r"zcam(\d+)_consensus", str(g))
            if mz:
                no_z = [m for m in members if m in sites and int(mz.group(1)) not in site_zooms(sites[m])]
                if no_z:
                    warn.append(f"group {g!r}: {no_z} have no {mz.group(1)} in their 'zcam' list")
    names = sorted(ranges, key=lambda k: ranges[k][0][0])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sorted(ranges[a]) == sorted(ranges[b]):     # v0p50: e.g. rockytop / rockytop_skinner
                warn.append(f"sites {a!r} and {b!r} have the same sols {ranges[a]}: they select the same images "
                            f"(a site is defined by its sol ranges only)")
                continue
            # v0p75: every pair of visits: inside one another is a larger / smaller block, a partial overlap a warning
            partial = []
            for (a0, a1) in ranges[a]:
                for (b0, b1) in ranges[b]:
                    if b0 > a1 or a0 > b1:
                        continue
                    if (a0 <= b0 and b1 <= a1) or (b0 <= a0 and a1 <= b1):
                        continue
                    partial.append(([a0, a1], [b0, b1]))
            for ra, rb in partial[:1]:
                warn.append(f"sites {a!r} {ra} and {b!r} {rb} partly overlap (fine if meant)")
    return err, warn


SITES_FILE_VERSION = 4          # v0p75/v0p80: "sols" as a list of visits (longest / main first), the "block" record


def check_block_record(b: Any) -> List[str]:
    """v0p80: problems of a site's ``"block"`` record (the block of rover stations it was made from, written by
    :mod:`mppp.maps`): an object with ``number`` (int), ``half_size_m`` (> 0), ``centre`` ({"lon", "lat"} and/or
    {"easting_m", "northing_m"}), and optionally ``region``, ``kind``, ``legacy_site``, ``end_of_drive_sols``,
    ``navcam_360s``, ``zcam_360s``.  Unknown fields are allowed (later versions may add some)."""
    if not isinstance(b, dict):
        return [f"'block' must be an object, not {type(b).__name__}"]
    out = []
    if "number" in b and not (isinstance(b["number"], int) and not isinstance(b["number"], bool) and b["number"] > 0):
        out.append(f"block number {b['number']!r} must be a positive whole number")
    hs = b.get("half_size_m")
    if hs is not None and not (isinstance(hs, (int, float)) and not isinstance(hs, bool) and hs > 0):
        out.append(f"block half_size_m {hs!r} must be a positive number")
    c = b.get("centre")
    if c is not None:
        if not isinstance(c, dict):
            out.append("block centre must be an object like {\"lon\": 77.4, \"lat\": 18.4}")
        else:
            for k in ("lon", "lat", "easting", "northing", "easting_m", "northing_m"):
                if k in c and not isinstance(c[k], (int, float)):
                    out.append(f"block centre {k} {c[k]!r} must be a number")
    return out


def site_block(site: str, table: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """v0p80: the ``"block"`` record of a site (None for a site without one)."""
    t = table if table is not None else load_site_table()
    v = t["sites"].get(site)
    return v.get("block") if isinstance(v, dict) else None


def block_sites(table: Optional[Dict[str, Any]] = None) -> List[str]:
    """v0p80: the sites that are blocks of rover stations (with a ``"block"`` record), by block number."""
    t = table if table is not None else load_site_table()
    rows = [(int((v.get("block") or {}).get("number") or 10 ** 6), k) for k, v in t["sites"].items()
            if isinstance(v, dict) and isinstance(v.get("block"), dict)]
    return [k for _, k in sorted(rows)]


try:
    SITES: Dict[str, Tuple[int, int]] = load_sites()   # sol range of the Navcam (+ Mastcam-Z) block (notebook 03)
except Exception as _e:                                 # noqa: BLE001 - a broken edit must not break "import mppp"
    import warnings as _w
    _w.warn(f"MPPP: the site definitions {SITES_FILE} cannot be read ({type(_e).__name__}: {_e}); "
            f"run scripts/check_sites.py to find the problem")
    SITES = {}

_WORDS = {"threeforks": "Three Forks", "seitah": "Seitah"}
FOLDER_PREFIX = "mars2020_sol_"              # v0p53: mars2020_sol_<first sol, 4 digits>_<site>_colmap[_zcam]
ZCAM_SUFFIX = "_colmap_zcam"                 # v0p53 (Navcam is in every block; any Mastcam-Z zoom)
NAV_SUFFIX = "_colmap"
_OLD_ZCAM_SUFFIXES = ("_colmap_zcam34", "_colmap_nav_zcam34")   # before v0p53: read by parse_work_folder only


def site_label(name: str) -> str:
    """``"threeforks_south"`` -> ``"Three Forks South"``, ``"van_zyl"`` -> ``"Van Zyl"``."""
    return " ".join(_WORDS.get(w, w.capitalize()) for w in str(name).split("_") if w)


def check_sites(sites: Dict[str, Any]) -> List[str]:
    """Problems of a site list: a sol range whose end is before its start (it selects nothing).  v0p75: a site may
    hold several ranges."""
    out = []
    for k, v in sites.items():
        try:
            rs = sol_ranges(v)
        except ValueError as e:
            out.append(f"{k}: {e}")
            continue
        for a, b in rs:
            if int(b) < int(a):
                out.append(f"{k}: sol range ({a}, {b}) ends before it starts - it selects no images")
    return out


def folder_name(site: str, first_sol: int, zcam: bool = False) -> str:
    """v0p53: ``mars2020_sol_1451_bell_island_colmap`` (``..._colmap_zcam`` with Mastcam-Z)."""
    return f"{FOLDER_PREFIX}{int(first_sol):04d}_{site}" + (ZCAM_SUFFIX if zcam else NAV_SUFFIX)


def work_folder(root: PathLike, site: str, zcam: bool = False,
                sols: Optional[Sequence[int]] = None) -> Path:
    """Notebook 03's WORK folder of a site (v0p53): ``<root>/mars2020_sol_<first sol>_<site>_colmap`` or
    ``..._colmap_zcam``.  ``sols``: the site's sol range (default: from the site definitions file)."""
    if sols is None:
        table = SITES if site in SITES else load_sites()
        if site not in table:
            raise KeyError(f"site {site!r} is not in the site definitions ({SITES_FILE}); give sols=(first, last)")
        sols = table[site]
    return Path(root) / folder_name(site, sol_span(sols)[0], zcam)      # v0p75: the main visit's first sol


def parse_work_folder(folder: PathLike) -> Tuple[Optional[str], bool]:
    """v0p43: ``(site, with Mastcam-Z)`` from a WORK folder name; ``(None, False)`` for another name.  v0p53:
    ``mars2020_sol_<sol>_<site>_colmap[_zcam]``; the names before v0p53 (``<site>_colmap``, ``<site>_colmap_zcam34``)
    are still read here, for a WORK_DIR given by hand."""
    site, z = _site_of_folder(Path(folder))
    if site is not None:
        return site, z
    n = Path(folder).name
    for suf in _OLD_ZCAM_SUFFIXES:
        if n.endswith(suf):
            return n[: -len(suf)], True
    if n.endswith(NAV_SUFFIX):
        return n[: -len(NAV_SUFFIX)], False
    return None, False


_FOLDER_RE = None


def _site_of_folder(folder: Path) -> Tuple[Optional[str], bool]:
    """v0p53 names only (``discover_scapes`` / ``scan_scapes`` skip the folders of earlier versions)."""
    import re
    global _FOLDER_RE
    if _FOLDER_RE is None:
        _FOLDER_RE = re.compile(r"mars2020_sol_(\d{4,})_(.+?)_colmap(_zcam)?")
    m = _FOLDER_RE.fullmatch(folder.name)
    if not m:
        return None, False
    return m.group(2), bool(m.group(3))


def _in_range(s: int, r: Any) -> bool:
    return in_sol_ranges(s, r)                              # v0p75: r may hold several visits


def scan_scapes(root: PathLike, sites: Optional[Dict[str, Sequence[int]]] = None,
                require_error_input: bool = False, include_zcam: bool = True,
                exclude: Iterable[str] = (), min_images: int = 10) -> List[Dict[str, Any]]:
    """
    Every notebook 03 WORK folder below ``root`` with what it holds and whether it can be analysed (``ok``, else
    ``reason``).  A folder is used when its ``colmap/project.json`` records a finished alignment (``settings.
    reconstruction.path``, whose ``images.bin`` exists; a run in progress rewrites project.json first and is left
    out), it has at least ``min_images`` images, ``colmap/error_input/summary.json`` exists if
    ``require_error_input``, and its images fall in the site's ``sites`` sol range (a folder holding another site's
    block - a renamed site - is left out; sites not in ``sites`` are not checked).  ``exclude``: folder names, site
    names or labels to leave out.  Rows in sol order.
    """
    sites = SITES if sites is None else sites
    root = Path(root)
    excl = {str(e).lower() for e in exclude}
    rows: List[Dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        site, zcam = _site_of_folder(folder)
        if site is None:
            continue                                           # camera_analysis and other folders
        label = site_label(site) + (" N+Z" if zcam else "")
        row: Dict[str, Any] = {"label": label, "site": site, "zcam": zcam, "zcam34": zcam, "folder": str(folder), "ok": False,
                               "reason": None, "in_sites": site in sites,
                               "site_sols": list(sol_span(sites[site])) if site in sites else None,
                               "site_sol_ranges": [list(r) for r in sol_ranges(sites[site])] if site in sites else None}
        rows.append(row)
        if zcam and not include_zcam:
            row["reason"] = "Mastcam-Z block (include_zcam=False)"
            continue
        if {folder.name.lower(), site.lower(), label.lower()} & excl:
            row["reason"] = "excluded"
            continue
        col = folder / "colmap"
        pj = col / "project.json"
        if not pj.is_file():
            row["reason"] = "no colmap/project.json (not aligned)"
            continue
        try:
            p = json.loads(pj.read_text(encoding="utf-8"))
        except Exception as e:                                  # noqa: BLE001
            row["reason"] = f"project.json unreadable ({type(e).__name__})"
            continue
        ims = p.get("images") or []
        sols = [int(r["sol"]) for r in ims if "sol" in r]
        fam = [str(r.get("instrument", ""))[:1] for r in ims]
        row.update({"images": len(ims), "navcam_images": fam.count("N"), "zcam_images": fam.count("Z"),
                    "stations": len({r.get("station") for r in ims}),
                    "sol_min": min(sols) if sols else None, "sol_max": max(sols) if sols else None,
                    "sol_median": float(statistics.median(sols)) if sols else None})
        h = col / "health" / "health.json"
        if h.is_file():
            try:
                hj = json.loads(h.read_text(encoding="utf-8"))
                row.update({"health": hj.get("verdict"), "mppp_version": hj.get("mppp_version")})
            except Exception:                                   # noqa: BLE001
                pass
        rs = (p.get("settings") or {}).get("reconstruction") or {}
        model = col / str(rs.get("path", "")).replace("\\", "/") if rs.get("path") else None
        if model is None or not (model / "images.bin").is_file():
            row["reason"] = ("no finished alignment (project.json has no reconstruction: a run in progress?)"
                             if model is None else f"model colmap/{str(rs.get('path')).replace(chr(92), '/')} missing")
            continue
        row["model"] = str(model)
        row["error_input"] = (col / "error_input" / "summary.json").is_file()
        if require_error_input and not row["error_input"]:
            row["reason"] = "no colmap/error_input/summary.json"
            continue
        if len(ims) < min_images:
            row["reason"] = f"{len(ims)} images (< {min_images})"
            continue
        if site in sites and sols and sol_span(sites[site])[1] >= sol_span(sites[site])[0]:
            if not any(_in_range(s, sites[site]) for s in sols):
                other = [k for k, r in sites.items() if k != site and sol_span(r)[1] >= sol_span(r)[0]
                         and _in_range(int(statistics.median(sols)), r)]
                sp = sol_span(sites[site])
                row["reason"] = (f"no image in the site's sols {sp[0]}-{sp[1]}: the folder holds sols "
                                 f"{min(sols)}-{max(sols)}" + (f" ({', '.join(other)}?)" if other else ""))
                continue
        row["ok"] = True
    rows.sort(key=lambda r: (r.get("sol_median") is None, r.get("sol_median") or 0, r["label"]))
    return rows


def discover_scapes(root: PathLike, sites: Optional[Dict[str, Sequence[int]]] = None,
                    require_error_input: bool = False, include_zcam: bool = True, exclude: Iterable[str] = (),
                    min_images: int = 10, verbose: bool = True) -> Dict[str, Path]:
    """
    {label: WORK folder} of every usable alignment below ``root`` (see :func:`scan_scapes`), in sol order;
    ``verbose`` prints the table of all folders with the reasons for the ones left out.
    """
    rows = scan_scapes(root, sites, require_error_input, include_zcam, exclude, min_images)
    # v0p71: two sites with the same label (e.g. bell_island and bell_island_large, both "Bell Island") stay apart:
    # the later one is labelled with its site name (before, one silently replaced the other)
    seen: Dict[str, int] = {}
    for r in rows:
        if r["ok"]:
            seen[r["label"]] = seen.get(r["label"], 0) + 1
    used = set()
    for r in rows:
        if r["ok"] and seen[r["label"]] > 1:
            if r["label"] in used:
                r["label"] = f"{r['label']} ({r['site']}{', N+Z' if r.get('zcam') else ''})"
            used.add(r["label"].split(" (")[0])
    if verbose:
        print_scan(rows, sites)
    return {r["label"]: Path(r["folder"]) for r in rows if r["ok"]}


def print_scan(rows: List[Dict[str, Any]], sites: Optional[Dict[str, Sequence[int]]] = None) -> None:
    for w in check_sites(SITES if sites is None else sites):
        print("SITES:", w)
    used = [r for r in rows if r["ok"]]
    print(f"{len(used)} of {len(rows)} site folders used:")
    for r in rows:
        sol = f"{r['sol_min']}-{r['sol_max']}" if r.get("sol_min") is not None else "-"
        info = (f"{r.get('images', 0):4d} images, {r.get('stations', 0):3d} stations, sols {sol:>9s}, "
                f"health {r.get('health') or '-'}" if r.get("images") is not None else "")
        print(f"  {'+' if r['ok'] else '-'} {r['label']:24s} {info}" + ("" if r["ok"] else f"  [{r['reason']}]"))
