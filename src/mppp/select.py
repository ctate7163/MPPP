"""
Selecting PDS ``.IMG`` products from a local archive.

Selection returns lists of paths.  How they are bundled ("scapes") is the
user's business; nothing here requires a scape.  Nothing here ever deletes
archive files (the legacy ``dedupe_by_version_index(delete=True)`` did).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .filenames import M2020Filename, parse_filename

PathLike = Union[str, Path]


def iter_imgs(input_dir: PathLike) -> Iterable[Tuple[Path, M2020Filename]]:
    """All parsable ``*.IMG`` below ``input_dir`` (recursive, case-insensitive)."""
    seen = set()
    for fp in Path(input_dir).rglob("*"):
        if fp.suffix.lower() != ".img" or not fp.is_file() or fp in seen:
            continue
        seen.add(fp)
        try:
            yield fp, parse_filename(fp)
        except ValueError:
            continue


def find_imgs(input_dir: PathLike, camera_codes: Sequence[str],
              sol_range: Optional[Tuple[int, int]] = None,
              site_drive: Optional[Tuple[int, int]] = None,
              sequ_id: Optional[str] = None,
              product_type: Optional[str] = "RAD",
              include_thumbnails: bool = False) -> List[Path]:
    """
    ``camera_codes``: prefixes of the file name ('NLF', 'ZL0', 'Z', ...).
    ``sequ_id``: substring looked for after the site/drive field, i.e. in the
    sequence id + camera-specific field ('ZCAM08529', '_034', '_0A01'...).
    """
    codes = tuple(c.upper() for c in camera_codes)
    out: List[Path] = []
    for fp, fn in iter_imgs(input_dir):
        name = fn.stem.upper()
        if not name.startswith(codes):
            continue
        if product_type and fn.product_type != product_type.upper():
            continue
        if fn.thumbnail and not include_thumbnails:
            continue
        if sol_range and not (sol_range[0] <= fn.sol <= sol_range[1]):
            continue
        if site_drive and (fn.site, fn.drive) != tuple(site_drive):
            continue
        if sequ_id and sequ_id.upper() not in name[35:]:
            continue
        out.append(fp)
    return sorted(out)


def find_imgs_in_sol_range(sol_first: int, sol_last: int, camera_codes: Sequence[str],
                           input_dir: PathLike, sequ_id: Optional[str] = None, **kw) -> List[Path]:
    """Legacy-compatible signature."""
    return find_imgs(input_dir, camera_codes, sol_range=(sol_first, sol_last), sequ_id=sequ_id, **kw)


def find_imgs_for_waypoint(waypoint: Dict[str, Any], camera_codes: Sequence[str],
                           input_dir: PathLike, sequ_id: Optional[str] = None, **kw) -> List[Path]:
    """Legacy-compatible signature."""
    p = waypoint.get("properties", waypoint)
    return find_imgs(input_dir, camera_codes, site_drive=(int(p["site"]), int(p["drive"])),
                     sequ_id=sequ_id, **kw)


def dedupe_by_version(paths: Iterable[PathLike]) -> Tuple[List[Path], List[Path]]:
    """
    Keep, for products identical up to the two-digit version field, only the
    highest version.  Returns ``(kept, superseded)`` in input order.
    Files are never deleted.
    """
    paths = [Path(p) for p in paths]
    best: Dict[str, Tuple[int, Path]] = {}
    for p in paths:
        try:
            fn = parse_filename(p)
            key, ver = fn.stem[:52].upper(), fn.version
        except ValueError:
            key, ver = f"__unparsed__{p.name}", -1
        if key not in best or (ver, p.name) > (best[key][0], best[key][1].name):
            best[key] = (ver, p)
    keep = {v[1] for v in best.values()}
    kept = [p for p in paths if p in keep]
    return list(dict.fromkeys(kept)), [p for p in paths if p not in keep]
