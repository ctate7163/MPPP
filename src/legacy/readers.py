# import urllib, json

import math
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import re
import csv
from bisect import bisect_left
import xml.etree.ElementTree as ET



def waypoints_within_radius(anchor_sol: int, radius: float, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find all waypoints within 'radius' (same units as 'easting'/'northing', typically meters)
    of the anchor waypoint defined by 'anchor_sol'.

    Anchor selection policy:
    - If anchor_sol == -1, choose the LAST waypoint with the highest sol number.
    - Else if waypoints exist with properties['sol'] == anchor_sol, choose the LAST such waypoint.
    - Otherwise, choose the LAST waypoint overall (i.e., latest in the dataset order).
    - If the chosen anchor is missing easting/northing, raise a ValueError.

    Returns:
        A list of waypoint feature dicts (as in data['features']) whose planar distance
        from the anchor (in E-N space) is <= radius. The anchor itself is included
        if it has valid coordinates (distance == 0).

    Notes:
        - Distance is computed in 2D using (easting, northing) only.
        - Waypoints missing valid easting/northing are skipped.
        - If data['features'] is empty or no waypoint has valid coordinates, a ValueError is raised.
    """
    features = data.get("features", [])
    if not features:
        raise ValueError("No waypoints found: data['features'] is empty.")

    def get_sol(feat: Dict[str, Any]) -> Optional[int]:
        p = feat.get("properties", {})
        try:
            return int(p.get("sol"))
        except Exception:
            return None

    def get_en(feat: Dict[str, Any]) -> Optional[tuple]:
        p = feat.get("properties", {})
        try:
            e = float(p["easting"])
            n = float(p["northing"])
            return e, n
        except Exception:
            return None

    # Anchor selection
    if anchor_sol == -1:
        # Find the maximum sol among features
        sols = [(i, get_sol(f)) for i, f in enumerate(features) if get_sol(f) is not None]
        if not sols:
            raise ValueError("No valid sol values in waypoints.")
        max_sol = max(s for _, s in sols)
        sol_indices = [i for i, s in sols if s == max_sol]
        anchor_idx = sol_indices[-1]  # last waypoint with the highest sol
    else:
        # Candidates for the anchor sol (choose the last one on that sol)
        sol_indices = [i for i, f in enumerate(features) if get_sol(f) == anchor_sol]
        if sol_indices:
            anchor_idx = sol_indices[-1]
        else:
            # Choose the latest waypoint overall that has valid coordinates
            valid_indices = [i for i, f in enumerate(features) if get_en(f) is not None]
            if not valid_indices:
                raise ValueError("No waypoints with valid easting/northing present.")
            anchor_idx = valid_indices[-1]

    anchor_feat = features[anchor_idx]
    anchor_en = get_en(anchor_feat)
    if anchor_en is None:
        raise ValueError("Anchor waypoint lacks valid easting/northing.")

    ae, an = anchor_en
    r = float(radius)

    in_radius: List[Dict[str, Any]] = []
    for f in features:
        en = get_en(f)
        if en is None:
            continue
        e, n = en
        if math.hypot(e - ae, n - an) <= r:
            in_radius.append(f)

    return in_radius


def find_imgs_in_sol_range(
    sol_first: int,
    sol_last: int,
    camera_codes: List[str],
    input_dir: str | Path,
    sequ_id: Optional[str] = None,
) -> List[Path]:
    """
    Find all .IMG files under 'input_dir' whose sol is in [sol_first, sol_last],
    camera code matches one of camera_codes, and sequ_id (if provided) appears
    after the RAD_Nsssdddd site+drive token.

    Args:
        input_dir: directory to search recursively.
        sol_first, sol_last: inclusive sol range.
        camera_codes: list of valid camera code prefixes (0–10 chars).
        sequ_id: optional string to match after RAD_Nsssdddd.

    Returns:
        List of Path objects to matching .IMG files.
    """
    input_dir = Path(input_dir)
    cam_codes = [c.upper() for c in camera_codes]

    # Pattern extracts both sol and site/drive.
    # Example filename snippet: ..._S0123_RAD_N05301234...
    sol_pat = re.compile(r"[Ss](?P<sol>\d{3,5})")
    site_drive_pat = re.compile(r"(RAD_N\d{3}\d{4})", re.IGNORECASE)

    results: List[Path] = []

    for fp in input_dir.rglob("*.img"):
        if not fp.is_file():
            continue

        name = fp.name.upper()

        # Camera prefix check
        if not any(name.startswith(code) for code in cam_codes):
            continue

        # Sol extraction
        sol_val = int(fp.name.upper()[4:8])
        if not (sol_first <= sol_val <= sol_last):
            continue

        # site+drive extraction (so we know where to check sequ_id)
        msite = site_drive_pat.search(name)
        if not msite:
            continue

        # sequ_id check
        if sequ_id:
            pos_after = msite.end()
            if sequ_id.upper() not in name[pos_after:]:
                continue

        results.append(fp)

    return results




def find_imgs_for_waypoint(
    waypoint: Dict[str, Any],
    camera_codes: List[str],
    input_dir: str | Path,
    sequ_id: Optional[str] = None,
) -> List[Path]:
    """
    Return all .IMG files under 'input_dir' (recursively) that match:
      - The given waypoint's (site, drive), where the filename encodes them as:
            ... RAD_N sss dddd ...
        with exactly 3 digits for site (sss) and 4 for drive (dddd), zero-padded.
      - The camera code prefix (0–10 chars) is one of the provided camera_codes.
      - If sequ_id is given, it must appear somewhere *after* the site+drive pattern.

    Example match in filename:
        'NLF_...RAD_N05310123...SEQ123.IMG'
         site=053 -> 53, drive=0123 -> 123, sequ_id="SEQ123"

    Args:
        waypoint: dict with "properties" containing "site" and "drive" (ints).
        camera_codes: list of string prefixes (0–10 chars).
        input_dir: directory to search (recursively).
        sequ_id: optional string; must be present in filename after site/drive.

    Returns:
        List[Path]: all matching IMG file paths.
    """
    props = waypoint.get("properties", {})
    try:
        target_site = int(props["site"])
        target_drive = int(props["drive"])
    except Exception as e:
        raise ValueError("Waypoint missing valid 'site'/'drive' in properties.") from e

    input_dir = Path(input_dir)
    cam_codes = [c.upper() for c in camera_codes]

    # Pattern: RAD_Nsssdddd (sss=site, dddd=drive)
    pat = re.compile(r"(RAD_N(?P<site>\d{3})(?P<drive>\d{4}))", re.IGNORECASE)

    results: List[Path] = []
    for fp in input_dir.rglob("*.img"):
        if not fp.is_file():
            continue

        name = fp.name.upper()

        # Check prefix
        if not any(name.startswith(code) for code in cam_codes):
            continue

        m = pat.search(name)
        if not m:
            continue

        site_val = int(m.group("site"))
        drive_val = int(m.group("drive"))

        if site_val != target_site or drive_val != target_drive:
            continue

        # If sequ_id is provided, check it's present *after* the match position
        if sequ_id:
            pos_after = m.end()  # end index of RAD_Nsssdddd
            if sequ_id.upper() not in name[pos_after:]:
                continue

        results.append(fp)

    return results

def find_waypoint_for_site_drive(
    data: Dict[str, Any],
    site: int,
    drive: int,
) -> Dict[str, Any]:
    """
    Return the waypoint feature for a given (site, drive) using M20 waypoints data.

    Selection rules:
      1) If (site, drive) exists, return the LAST such feature (handles duplicates).
      2) Otherwise, among all features with the requested 'site', return the one
         with the nearest 'drive' (min |drive - target|). Tie-breaker: prefer the
         larger drive; if still tied, prefer the later (last) entry in the dataset.
      3) If the site is not present at all, raise ValueError.

    Args:
        data: Parsed JSON dict as loaded from M20_waypoints.json.
        site: Target site index (int).
        drive: Target drive index (int).

    Returns:
        The selected waypoint feature (a dict from data['features']).

    Raises:
        ValueError if 'features' missing/empty, or if the site does not exist.
    """
    features: List[Dict[str, Any]] = data.get("features", [])
    if not features:
        raise ValueError("Waypoints data has no 'features'.")

    def get_int(p: Dict[str, Any], key: str) -> Optional[int]:
        try:
            return int(p.get(key))
        except Exception:
            return None

    # Collect features for this site
    site_feats: List[Tuple[int, Dict[str, Any]]] = []  # (index, feature)
    for idx, feat in enumerate(features):
        props = feat.get("properties", {})
        s = get_int(props, "site")
        if s is None:
            continue
        if s == site:
            site_feats.append((idx, feat))

    if not site_feats:
        raise ValueError(f"Requested site {site} not found in waypoint database.")

    # First, look for exact (site, drive); choose the LAST one
    exact_indices = []
    for idx, feat in site_feats:
        d = get_int(feat.get("properties", {}), "drive")
        if d is not None and d == drive:
            exact_indices.append(idx)

    if exact_indices:
        chosen_idx = exact_indices[-1]
        return features[chosen_idx]

    # Otherwise, choose the nearest drive within the site
    # Build list: (abs_diff, -drive_for_tiebreak, original_index)
    candidates: List[Tuple[int, int, int]] = []
    for idx, feat in site_feats:
        d = get_int(feat.get("properties", {}), "drive")
        if d is None:
            continue
        diff = abs(drive - d)
        # Tie-break: prefer larger drive (use -d), then prefer later entry (larger idx)
        candidates.append((diff, -d, idx))

    if not candidates:
        # Site exists but no usable drive values — fall back to last entry for that site
        return site_feats[-1][1]

    candidates.sort()
    _, _, chosen_idx = candidates[0]

    # Among any identical (diff, -drive) ties, pick LAST in file order
    # (Because .sort() is stable, we can scan from end to enforce "last" when equal.)
    best_key = candidates[0][:2]
    for diff, negd, idx in reversed(candidates):
        if (diff, negd) == best_key:
            chosen_idx = idx
            break

    return features[chosen_idx]['properties']


# def find_waypoint_for_site_drive(
#     data: Dict[str, Any],
#     site: int,
#     drive: int,
# ) -> Dict[str, Any]:
#     """
#     Return the waypoint feature for a given (site, drive) using M20 waypoints data.

#     Selection rules:
#       1) If (site, drive) exists, return the LAST such feature (handles duplicates).
#       2) Otherwise, return the waypoint from the same site whose 'drive' is the
#          largest value STRICTLY LESS than the requested drive (i.e., the previous drive).
#          If multiple features share that drive, return the LAST one (latest entry).
#       3) If the site is not present at all, raise ValueError.
#       4) If the site exists but there is no earlier drive (< requested), raise ValueError.

#     Args:
#         data: Parsed JSON dict as loaded from M20_waypoints.json.
#         site: Target site index (int).
#         drive: Target drive index (int).

#     Returns:
#         The selected waypoint feature (a dict from data['features']).

#     Raises:
#         ValueError if 'features' missing/empty, site not found, or no previous drive exists.
#     """
#     features: List[Dict[str, Any]] = data.get("features", [])
#     if not features:
#         raise ValueError("Waypoints data has no 'features'.")

#     def get_int(props: Dict[str, Any], key: str) -> Optional[int]:
#         try:
#             return int(props.get(key))
#         except Exception:
#             return None

#     # Collect features for this site, keeping original indices to resolve "last occurrence".
#     site_feats: List[Tuple[int, Dict[str, Any]]] = []
#     for idx, feat in enumerate(features):
#         props = feat.get("properties", {}) or {}
#         s = get_int(props, "site")
#         if s is None:
#             continue
#         if s == site:
#             site_feats.append((idx, feat))

#     if not site_feats:
#         raise ValueError(f"Requested site {site} not found in waypoint database.")

#     # 1) Exact (site, drive): return LAST occurrence.
#     exact_indices: List[int] = []
#     for idx, feat in site_feats:
#         d = get_int(feat.get("properties", {}) or {}, "drive")
#         if d is not None and d == drive:
#             exact_indices.append(idx)
#     if exact_indices:
#         return features[exact_indices[-1]]

#     # 2) No exact match: return previous drive within the site.
#     #    Find the maximum d such that d < drive; if ties (multiple features with same d), take last.
#     prev_drive: Optional[int] = None
#     for _, feat in site_feats:
#         d = get_int(feat.get("properties", {}) or {}, "drive")
#         if d is None:
#             continue
#         if d < drive and (prev_drive is None or d > prev_drive):
#             prev_drive = d

#     if prev_drive is None:
#         raise ValueError(
#             f"Site {site} exists but no previous drive (< {drive}) found."
#         )

#     # Among features with prev_drive, return LAST occurrence in file order.
#     last_idx_for_prev = None
#     for idx, feat in site_feats:
#         d = get_int(feat.get("properties", {}) or {}, "drive")
#         if d == prev_drive:
#             last_idx_for_prev = idx  # overwritten to end up with the last

#     # last_idx_for_prev must be set because prev_drive came from the dataset
#     return features[last_idx_for_prev]



def clean_path(path: Union[str, Path]) -> str:
    """
    Convert a pathlib.Path (or strings) into plain string path
    without the WindowsPath(...) wrapper.
    
    Uses POSIX style (forward slashes) for portability.
    """
    return Path(path).as_posix()

def clean_paths(paths: List[Union[str, Path]]) -> List[str]:
    """
    Convert a list of pathlib.Path (or strings) into plain string paths
    without the WindowsPath(...) wrapper.
    
    Uses POSIX style (forward slashes) for portability.
    """
    return [Path(p).as_posix() for p in paths]

from pathlib import Path
from typing import Iterable, List, Tuple, Union, Dict

def dedupe_by_version_index(
    paths: Iterable[Union[str, Path]],
    delete: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """
    Deduplicate IMG files by the 'image version index' defined as the last digit
    immediately before the '.IMG' (case-insensitive) extension.

    Files that are identical up to that digit are considered duplicates.
    For each such group, keep only the file with the highest version digit (0–9).
    Optionally delete the lower-version files from disk.

    Args:
        paths: Iterable of paths (str or Path).
        delete: If True, delete redundant files from disk.

    Returns:
        kept, removed:
            kept    - list of Paths that remain after deduplication
            removed - list of redundant Paths that were excluded (and deleted if delete=True)
    """
    def normalize(p: Union[str, Path]) -> Path:
        return Path(p)

    def parse_key_and_version(p: Path) -> Tuple[Tuple[Path, str], int]:
        """
        Returns:
            key: (parent_dir, base_prefix) where base_prefix is the filename up to (but not including)
                 the last digit before '.IMG' / '.img'.
            version: int digit [0..9] for that last char before the extension.

        Raises:
            ValueError if the pattern is not matched.
        """
        name = p.name
        lower = name.lower()
        if not lower.endswith(".img"):
            raise ValueError(f"Not an IMG file: {p}")
        # Position of '.' before extension
        dot_idx = name.rfind(".")
        if dot_idx <= 0:
            raise ValueError(f"Invalid IMG filename: {p}")

        # Character immediately before the dot must be a digit
        if dot_idx - 1 < 0 or not name[dot_idx - 1].isdigit():
            raise ValueError(f"No version digit before .IMG in: {p}")

        version_char = name[dot_idx - 1]
        version = int(version_char)

        # Key: everything up to but not including that digit (basename prefix)
        base_prefix = name[: dot_idx - 1]  # exclude the version digit
        key = (p.parent, base_prefix)
        return key, version

    # Build groups
    groups: Dict[Tuple[Path, str], List[Path]] = {}
    keyed_versions: Dict[Path, Tuple[Tuple[Path, str], int]] = {}

    for raw in paths:
        p = normalize(raw)
        try:
            key, ver = parse_key_and_version(p)
        except ValueError:
            # Not matching the rule; keep it as-is by placing it in its own unique group
            key = (p.parent, f"__NO_VER__::{p.name}")
            ver = -1  # single-member group; won't compete with others
        keyed_versions[p] = (key, ver)
        groups.setdefault(key, []).append(p)

    kept: List[Path] = []
    removed: List[Path] = []

    for key, plist in groups.items():
        if len(plist) == 1:
            kept.append(plist[0])
            continue

        # Choose the file with the highest version; tie-breaker: lexicographic name
        # to keep determinism when multiple files have same version (shouldn't happen, but safe).
        best = max(plist, key=lambda p: (keyed_versions[p][1], p.name))
        kept.append(best)
        for p in plist:
            if p is not best:
                removed.append(p)

    if delete:
        for p in removed:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                # Intentionally silent: we don't block dedup result if deletion fails
                pass

    # Preserve original ordering of input for 'kept' where possible
    input_order = {Path(p): i for i, p in enumerate(map(normalize, paths))}
    kept.sort(key=lambda p: input_order.get(p, 10**9))
    removed.sort(key=lambda p: input_order.get(p, 10**9))

    return kept, removed


def find_file_from_parent(directory_name: Union[str, Path], filename: Union[str, Path], start: Optional[Union[str, Path]] = None) -> Path:
    """
    Return the absolute Path to `../params/<filename>` relative to `start`
    (defaults to the current working directory).

    Args:
        filename: The file name (or relative path inside params) to locate.
        start:    Base directory to resolve from. Defaults to Path.cwd().

    Returns:
        Path to the file.

    Raises:
        FileNotFoundError if the parent directory, 'params' directory,
        or the target file does not exist.
    """
    base = Path(start) if start is not None else Path.cwd()
    parent = base.parent
    filename_path = parent / directory_name / filename

    return clean_path( filename_path )


def interpolate_csv_xy(csv_path: Union[str, Path], x_query: float) -> float:
    """
    Linearly interpolate y for a given x_query from a CSV file.
    - Ignores the first row (assumed header).
    - Uses the first two columns as x and y.
    - Accepts arbitrary row order; sorts by x.
    - If x_query is outside [min(x), max(x)], performs linear extrapolation
      using the nearest two points.
    - If x_query exactly matches an existing x, returns its y.

    Raises:
        ValueError if fewer than 2 valid (x,y) rows are found.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No such CSV file: {csv_path}")

    pairs: List[Tuple[float, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        # skip header
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except (TypeError, ValueError):
                continue
            pairs.append((x, y))

    if len(pairs) < 2:
        raise ValueError("Need at least two valid (x, y) rows to interpolate.")

    # Sort by x and collapse duplicate x by keeping the last occurrence
    pairs.sort(key=lambda p: p[0])
    xs: List[float] = []
    ys: List[float] = []
    for x, y in pairs:
        if xs and x == xs[-1]:
            ys[-1] = y
        else:
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        raise ValueError("After deduplication, fewer than two unique x values remain.")

    # Exact hit
    i = bisect_left(xs, x_query)
    if i < len(xs) and xs[i] == x_query:
        return ys[i]

    # Left extrapolation
    if i == 0:
        x0, y0 = xs[0], ys[0]
        x1, y1 = xs[1], ys[1]
        return y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)

    # Right extrapolation
    if i == len(xs):
        x0, y0 = xs[-2], ys[-2]
        x1, y1 = xs[-1], ys[-1]
        return y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)

    # Interpolation between xs[i-1] and xs[i]
    x0, y0 = xs[i - 1], ys[i - 1]
    x1, y1 = xs[i], ys[i]
    # Guard against accidental identical x (shouldn't happen after dedup)
    if x1 == x0:
        return (y0 + y1) / 2.0
    return y0 + (y1 - y0) * (x_query - x0) / (x1 - x0)


def read_xml(filename: str | Path) -> Dict[str, Any]:
    """
    Read a calibration XML file and return its contents as a dict.
    Converts numeric values to float when possible.
    
    Args:
        filename: Path to the XML file.
    
    Returns:
        A dictionary with tag names as keys and text values (floats if possible).
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"XML file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    result: Dict[str, Any] = {}
    for child in root:
        text = (child.text or "").strip()
        if text == "":
            result[child.tag] = None
            continue
        # try converting to float
        try:
            val = float(text)
            result[child.tag] = val
        except ValueError:
            result[child.tag] = text
    return result