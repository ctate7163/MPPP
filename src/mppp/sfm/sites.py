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


def load_site_table(path: Optional[PathLike] = None) -> Dict[str, Any]:
    """v0p43: the site definitions file (``mppp/data/sites.json`` by default): ``{"sites": {name: {"sols": [a, b],
    "label", "zcam", "note", "no_mask_inference_at", "settings"}}, "groups": {name: [site, ...]}}``.  ``zcam``
    (v0p53, a list of Mastcam-Z zooms in mm, 34 / 48 / 63 / 79 / 110; before: ``"zcam34": true``): the site also has a Navcam +
    Mastcam-Z block with those zooms (``mars2020_sol_<sol>_<site>_colmap_zcam``)."""
    f = Path(path) if path else SITES_FILE
    d = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(d.get("sites"), dict):
        raise ValueError(f"{f}: no 'sites' object")
    for k, v in d["sites"].items():
        sols = v.get("sols") if isinstance(v, dict) else v
        if not (isinstance(sols, (list, tuple)) and len(sols) == 2):
            raise ValueError(f"{f}: site {k!r} needs 'sols': [first, last]")
    return d


def load_sites(path: Optional[PathLike] = None) -> Dict[str, Tuple[int, int]]:
    """v0p43: name -> (first sol, last sol) from the site definitions file (notebook 03 ``SITES``)."""
    d = load_site_table(path)
    out = {}
    for k, v in d["sites"].items():
        sols = v.get("sols") if isinstance(v, dict) else v
        out[k] = (int(sols[0]), int(sols[1]))
    return out


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
        if not (isinstance(sols, list) and len(sols) == 2 and all(isinstance(x, int) for x in sols)):
            err.append(f"site {name!r}: 'sols' must be [first, last] (two whole numbers), not {sols!r}")
            continue
        if sols[1] < sols[0]:
            err.append(f"site {name!r}: sols {sols} end before they start")
        ranges[name] = sols
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
                            "no_mask_inference_at", "settings"}
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
    names = sorted(ranges, key=lambda k: ranges[k][0])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if ranges[b][0] > ranges[a][1]:
                break
            (a0, a1), (b0, b1) = ranges[a], ranges[b]
            if (a0, a1) == (b0, b1):                       # v0p50: e.g. rockytop / rockytop_skinner
                warn.append(f"sites {a!r} and {b!r} have the same sols {ranges[a]}: they select the same images "
                            f"(a site is defined by its sol range only)")
                continue
            if (a0 <= b0 and b1 <= a1) or (b0 <= a0 and a1 <= b1):
                continue                                   # one inside the other: a larger / smaller block
            warn.append(f"sites {a!r} {ranges[a]} and {b!r} {ranges[b]} partly overlap (fine if meant)")
    return err, warn


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


def check_sites(sites: Dict[str, Sequence[int]]) -> List[str]:
    """Problems of a site list: a sol range whose end is before its start (it selects nothing)."""
    out = []
    for k, (a, b) in sites.items():
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
    return Path(root) / folder_name(site, int(sols[0]), zcam)


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


def _in_range(s: int, r: Sequence[int]) -> bool:
    return int(r[0]) <= s <= int(r[1])


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
                               "site_sols": list(sites[site]) if site in sites else None}
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
        if site in sites and sols and int(sites[site][1]) >= int(sites[site][0]):
            if not any(_in_range(s, sites[site]) for s in sols):
                other = [k for k, r in sites.items() if k != site and int(r[1]) >= int(r[0])
                         and _in_range(int(statistics.median(sols)), r)]
                row["reason"] = (f"no image in the site's sols {sites[site][0]}-{sites[site][1]}: the folder holds sols "
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
