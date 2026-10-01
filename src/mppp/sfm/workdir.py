"""
A site's WORK folder (v0p43): what notebook 03 needs to align the images already processed there, and to rerun
an alignment cheaply.

A WORK folder (v0p53: ``<SCAPES_ROOT>/mars2020_sol_<sol>_<site>_colmap`` or ``..._colmap_zcam``) holds ``processed/`` (images,
masks, the manifest ``mppp_manifest_v*.json``) and one COLMAP project per variant: ``colmap/`` for the default
run and ``colmap_<variant>/`` for experiments (other SfM settings or camera models) that should not overwrite it.

- :func:`load_manifest`: the newest manifest of ``processed/`` (notebook 03 ``SOURCE = "processed"``: no PDS
  search, no image processing).
- ``exclude_images.txt`` in the WORK folder (:func:`read_exclusions`, :func:`apply_exclusions`): images left
  out of the alignment without deleting anything.  One entry per line, ``#`` starts a comment:

  - ``S032D1184``: every image of a station (site 32, drive 1184);
  - ``sol:658`` or ``sol:654-693``: a sol or a sol range;
  - ``seq:NCAM08111``: a sequence;
  - anything else: a file-name pattern (``*`` and ``?``), matched against the product name without extension,
    e.g. ``NLF_0658_0725353794_270RAD_N0320274NCAM08111_0A0095J01`` or ``ZR0_0690_*``.

- :func:`seed_variant`: a new variant starts from the default project's features (the images are hard links of
  the same processed files, so the features stay valid) and reuses its matches
  (:func:`mppp.sfm.database.reuse_matches`).
"""
from __future__ import annotations

import fnmatch
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

EXCLUDE_FILE = "exclude_images.txt"
SETTINGS_FILE = "mppp_settings.json"
_STATION = re.compile(r"^S(\d{1,3})D(\d{1,4})$", re.I)


def project_dir(work: PathLike, variant: Optional[str] = None) -> Path:
    """``<work>/colmap`` or ``<work>/colmap_<variant>``."""
    v = (variant or "").strip()
    if v and not re.fullmatch(r"[A-Za-z0-9._-]+", v):
        raise ValueError(f"variant {variant!r}: use letters, digits, '.', '_' or '-'")
    return Path(work) / ("colmap" if not v else f"colmap_{v}")


def load_manifest(processed_dir: PathLike) -> Tuple[Dict[str, Any], Path]:
    """The newest ``mppp_manifest_v*.json`` of ``processed_dir`` and its path."""
    d = Path(processed_dir)
    files = sorted(d.glob("mppp_manifest_v*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no mppp_manifest_v*.json in {d}: process the images first "
                                f"(notebook 03 with SOURCE = 'pds', or notebook 01)")
    return json.loads(files[-1].read_text(encoding="utf-8")), files[-1]


def _fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    fn = meta.get("filename") or {}
    stem = fn.get("stem") or Path(str(meta.get("source_product", ""))).stem
    return {"stem": stem, "sol": fn.get("sol", meta.get("sol")), "site": fn.get("site", meta.get("site")),
            "drive": fn.get("drive", meta.get("drive")), "sequence": fn.get("sequence", meta.get("sequence"))}


def read_exclusions(path: PathLike) -> List[str]:
    """The entries of an exclusion file (see the module docstring); [] if it does not exist."""
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def exclusion_matches(meta: Dict[str, Any], entry: str) -> bool:
    """Whether one exclusion entry selects this manifest image."""
    f = _fields(meta)
    e = entry.strip()
    m = _STATION.match(e)
    if m:
        return f["site"] is not None and f["drive"] is not None and (int(f["site"]), int(f["drive"])) == (
            int(m.group(1)), int(m.group(2)))
    low = e.lower()
    if low.startswith("sol:"):
        a, _, b = low[4:].partition("-")
        lo, hi = int(a), int(b or a)
        return f["sol"] is not None and lo <= int(f["sol"]) <= hi
    if low.startswith("seq:"):
        return str(f["sequence"] or "").upper() == e[4:].strip().upper()
    stem = f["stem"].upper()
    pat = re.sub(r"\.(img|png|png\.png)$", "", e, flags=re.I).upper()
    return fnmatch.fnmatchcase(stem, pat)


def apply_exclusions(metas: Sequence[Dict[str, Any]], entries: Sequence[str]
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """``(kept, removed)``; ``removed`` lists ``{"image", "entry"}`` (the first entry that selected it).  Entries
    that select nothing are reported as ``{"image": None, "entry"}`` so a typo is visible."""
    kept, removed, used = [], [], set()
    for m in metas:
        hit = next((e for e in entries if exclusion_matches(m, e)), None)
        if hit is None:
            kept.append(m)
        else:
            removed.append({"image": _fields(m)["stem"], "entry": hit})
            used.add(hit)
    removed += [{"image": None, "entry": e} for e in entries if e not in used]
    return kept, removed


def keep_existing(metas: Sequence[Dict[str, Any]], processed_dir: PathLike, fmt: str = "PNG8"
                  ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The manifest images whose ``fmt`` output is still in ``processed_dir`` (deleted images are left out, as
    ``KEEP_ONLY_REMAINING`` does for a PDS run)."""
    kept, missing = [], []
    for m in metas:
        rel = (m.get("outputs") or {}).get(fmt)
        if rel and (Path(processed_dir) / rel).is_file():
            kept.append(m)
        else:
            missing.append(_fields(m)["stem"])
    return kept, missing


def sol_range(metas: Sequence[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    sols = [int(_fields(m)["sol"]) for m in metas if _fields(m)["sol"] is not None]
    return (min(sols), max(sols)) if sols else None


def seed_variant(project_root: PathLike, base_root: PathLike) -> Dict[str, Any]:
    """Start a variant project from the base project's features: copy ``features.db`` and ``features.json`` when
    the variant has none.  Returns what was done and the base database to reuse matches from."""
    pr, br = Path(project_root), Path(base_root)
    out: Dict[str, Any] = {"base": str(br), "copied": [], "reuse_matches_from": []}
    if pr.resolve() == br.resolve():
        return out
    pr.mkdir(parents=True, exist_ok=True)
    if not (pr / "features.db").exists() and (br / "features.db").is_file() and (br / "features.json").is_file():
        for n in ("features.db", "features.json"):
            shutil.copy2(br / n, pr / n)
            out["copied"].append(n)
    if (br / "database.db").is_file() and (br / "database_matches.json").is_file():
        out["reuse_matches_from"].append(str(br / "database.db"))
    return out


def load_settings(path: PathLike) -> Dict[str, Any]:
    """A notebook 03 settings file: ``{"NAME": value, ...}`` (JSON values; keys starting with ``_`` are
    comments)."""
    p = Path(path)
    if not p.is_file():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{p}: a settings file holds one JSON object {{\"NAME\": value}}")
    return {k: v for k, v in d.items() if not str(k).startswith("_")}
