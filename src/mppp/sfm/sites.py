"""
The Navcam sites and their notebook 03 alignments (v0p40).

``SITES`` is the default site list of notebook 03 (name -> sol range of the block); notebook 03 keeps its own
editable copy.  :func:`discover_scapes` walks ``SCAPES_ROOT`` (``D:/scapes/colmap``) for the WORK folders notebook 03
made (``<site>_colmap``, and ``<site>_colmap_nav_zcam34`` with Mastcam-Z) and returns the ones that hold a finished
alignment, labelled and in sol order, for notebooks 04 (camera models) and 05 (error analysis).  :func:`scan_scapes`
returns every folder with the reason it was left out.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

SITES: Dict[str, Tuple[int, int]] = {         # sol range of the Navcam (+ Mastcam-Z) block (notebook 03)
    "butler_landing":       (   1,   14),
    "van_zyl":              (  49,   71),
    "rochette":             ( 178,  190),
    "seitah_north":         ( 238,  279),
    "sid":                  ( 361,  378),
    "rose_river_falls":     ( 448,  450),
    "rockytop":             ( 461,  530),
    "enchanted_lake":       ( 556,  590),
    "whale_mountain":       ( 606,  610),
    "threeforks_south":     ( 652,  674),
    "threeforks_north":     ( 680,  692),
    "threeforks":           ( 652,  692),
    "knob_mountain":        ( 698,  706),
    "berea":                ( 732,  738),
    "belva_crater":         ( 784,  815),
    "tuxedo_park":          ( 897,  908),
    "airey_hill":           ( 960,  991),
    "bunsen_peak":          (1066, 1095),
    "overlook_mountain":    (1150, 1155),
    "pearce_canyon":        (1183, 1218),
    "pico_turquino":        (1307, 1310),
    "rio_chiquito":         (1333, 1337),
    "south_arm":            (1408, 1412),
    "bell_island":          (1451, 1467),
    "taylorfjellet_large":  (1601, 1645),
    "taylorfjellet":        (1606, 1625),
    "origny_large":         (1765, 1813),
    "origny":               (1781, 1813),
    "olifants":             (1880, 1889),
    "groloy":               (1922, 1934),
    "marble_mountain":      (1965, 1979),
    "hippo_pools":          (1947, 1955),
}

_WORDS = {"threeforks": "Three Forks", "seitah": "Seitah"}
ZCAM_SUFFIX = "_colmap_nav_zcam34"
NAV_SUFFIX = "_colmap"


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


def _site_of_folder(folder: Path) -> Tuple[Optional[str], bool]:
    n = folder.name
    if n.endswith(ZCAM_SUFFIX):
        return n[: -len(ZCAM_SUFFIX)], True
    if n.endswith(NAV_SUFFIX):
        return n[: -len(NAV_SUFFIX)], False
    return None, False


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
        label = site_label(site) + (" N+Z34" if zcam else "")
        row: Dict[str, Any] = {"label": label, "site": site, "zcam34": zcam, "folder": str(folder), "ok": False,
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
