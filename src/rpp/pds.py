"""
Mars 2020 products straight from the PDS Imaging Node, into the local archive layout RPP reads (0.99.5).

Christian, 16 Nov 2026: "does RPP now have a way to download data directly from the PDS? ... the example data should
be downloaded directly from the pds if possible.  let's make that as easy as possible for someone testing out the
software for the first time."

    rpp pds fetch --sols 461 464                      Navcam RAD of the sols -> RPP_PDS (or --dest)
    rpp pds fetch --sols 464 --cameras zcam --match ZCAM03381    Mastcam-Z frames of one sequence
    rpp pds fetch --sols 461 464 --list               what the PDS has, nothing downloaded
    rpp example fetch                                 the example set (Rockytop, sols 461-464), from the PDS

The archive layout (``RPP_PDS``): ``datadrive/<sol5>/ids/rdr/<ncam|fcam|rcam>/`` and ``zcam/`` (flat), as the PDS has
them under ``<collection>/data/sol/<sol5>/ids/rdr/<camera>/``.  Each product comes with its PDS4 label (``.xml``).
A file already on disk is not downloaded again; a download goes to ``<name>.part`` and is renamed when complete.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

log = logging.getLogger("rpp.pds")

PathLike = Union[str, Path]
BASE = "https://planetarydata.jpl.nasa.gov/img/data/mars2020"
COLLECTIONS = {"ncam": "mars2020_navcam_ops_calibrated", "fcam": "mars2020_hazcam_ops_calibrated",
               "rcam": "mars2020_hazcam_ops_calibrated", "zcam": "mars2020_mastcamz_ops_calibrated"}
CAMERA_OF_LETTER = {"N": "ncam", "F": "fcam", "R": "rcam", "Z": "zcam"}
NAME = re.compile(r"^([NFRZ])[A-Z0-9]{2}_(\d{4})_\d{10}_\d{3}([A-Z]{3})_", re.I)
UA = {"User-Agent": "RPP (Rover Photogrammetry Pipeline; github.com/ctate7163/rpp)"}


def camera_of(name: str) -> str:
    """``NLF_0461_...`` -> ``ncam``; ``FLF`` -> ``fcam``; ``RLF`` -> ``rcam``; ``ZL0`` -> ``zcam``."""
    m = NAME.match(Path(name).name)
    if not m:
        raise ValueError(f"not a Mars 2020 engineering-camera or Mastcam-Z product name: {name}")
    return CAMERA_OF_LETTER[m.group(1).upper()]


def sol_of(name: str) -> int:
    m = NAME.match(Path(name).name)
    if not m:
        raise ValueError(f"not a Mars 2020 product name: {name}")
    return int(m.group(2))


def folder_url(camera: str, sol: int, base: str = BASE) -> str:
    return f"{base}/{COLLECTIONS[camera]}/data/sol/{int(sol):05d}/ids/rdr/{camera}/"


def local_folder(root: PathLike, camera: str, sol: int) -> Path:
    """Where the archive layout keeps a product: ``<root>/zcam`` for Mastcam-Z, else
    ``<root>/datadrive/<sol5>/ids/rdr/<camera>``."""
    root = Path(root)
    return root / "zcam" if camera == "zcam" else root / "datadrive" / f"{int(sol):05d}" / "ids" / "rdr" / camera


def _get(url: str, timeout: float = 300.0) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:    # noqa: S310
        return r.read()


def listing(url: str, get: Callable[[str], bytes] = _get) -> List[str]:
    """The file names a PDS folder page links to (relative or absolute links; folders left out)."""
    text = get(url).decode("utf-8", "replace")
    names = (h.rstrip().rsplit("/", 1)[-1] for h in re.findall(r'href\s*=\s*["\']([^"\'?#]+)["\']', text, re.I))
    return sorted({n for n in names if n and "." in n})


def _download(url: str, dst: Path, get: Callable[[str], bytes], tries: int = 3) -> int:
    for k in range(tries):
        try:
            b = get(url)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(b)
            tmp.replace(dst)
            return len(b)
        except Exception:                                              # noqa: BLE001 - the last try re-raises
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))
    return 0


def fetch_products(names: Iterable[str], root: PathLike, get: Callable[[str], bytes] = _get, base: str = BASE,
                   labels: bool = True, stop_on_failure: bool = False, tries: int = 3) -> Dict[str, Any]:
    """Download the named products (``.IMG``; with their ``.xml`` labels) into ``root`` in the archive layout.
    ``stop_on_failure``: stop at the first product that cannot be had (the example set needs them all).
    Returns {"downloaded", "kept", "bytes", "failed": [(name, error)], "files": [paths]}."""
    rep: Dict[str, Any] = {"downloaded": 0, "kept": 0, "bytes": 0, "failed": [], "files": []}
    for n in names:
        if stop_on_failure and rep["failed"]:
            break
        cam, sol = camera_of(n), sol_of(n)
        folder = local_folder(root, cam, sol)
        folder.mkdir(parents=True, exist_ok=True)
        stem = Path(n).name.rsplit(".", 1)[0]
        for ext in ((".IMG", ".xml") if labels else (".IMG",)):
            dst = folder / (stem + ext)
            if dst.is_file() and dst.stat().st_size > 0:
                rep["kept"] += 1
                rep["files"].append(str(dst))
                continue
            try:
                rep["bytes"] += _download(folder_url(cam, sol, base) + stem + ext, dst, get, tries=tries)
                rep["downloaded"] += 1
                rep["files"].append(str(dst))
            except Exception as e:                                     # noqa: BLE001
                if ext == ".IMG":
                    rep["failed"].append((stem + ext, f"{type(e).__name__}: {e}"))
                    break
                log.info("%s: no label (%s)", stem, e)                 # the product is usable without it
    return rep


def fetch_sols(root: PathLike, sols: Sequence[int], cameras: Sequence[str] = ("ncam",), product: str = "RAD",
               match: Optional[str] = None, list_only: bool = False, get: Callable[[str], bytes] = _get,
               base: str = BASE) -> Dict[str, Any]:
    """Every ``product`` (RAD) ``.IMG`` of the sols and cameras (names containing ``match``, when given), into
    ``root``.  ``list_only``: the names only.  Returns {"sols": {sol: {camera: n}}, "names": [...], and
    fetch_products' counts}."""
    rep: Dict[str, Any] = {"sols": {}, "names": [], "missing_folders": []}
    for sol in sols:
        for cam in cameras:
            url = folder_url(cam, sol, base)
            try:
                names = listing(url, get)
            except Exception as e:                                     # noqa: BLE001
                rep["missing_folders"].append(f"{url} ({type(e).__name__})")
                continue
            want = [n for n in names if n.upper().endswith(".IMG") and f"{product.upper()}_" in n.upper()
                    and (not match or match.upper() in n.upper())]
            rep["sols"].setdefault(int(sol), {})[cam] = len(want)
            rep["names"] += want
    if not list_only and rep["names"]:
        rep.update(fetch_products(rep["names"], root, get, base))
    return rep


def default_root() -> Optional[Path]:
    from . import env as rpp_env
    v = rpp_env.get("RPP_PDS")
    return Path(v) if v else None


def add_parser(sub) -> None:
    p = sub.add_parser("pds", help="0.99.5: Mars 2020 products from the PDS Imaging Node into the local archive")
    p.add_argument("what", choices=("fetch",))
    p.add_argument("--sols", nargs="+", type=int, required=True, help="FIRST LAST, or a list of sols")
    p.add_argument("--cameras", nargs="+", default=["ncam"], choices=sorted(COLLECTIONS),
                   help="ncam (default), fcam, rcam, zcam")
    p.add_argument("--match", default=None, help="only names containing this (e.g. NCAM03 or ZCAM03381)")
    p.add_argument("--product", default="RAD", help="the product type in the name (default RAD)")
    p.add_argument("--dest", default=None, help="the archive root (default RPP_PDS)")
    p.add_argument("--list", action="store_true", help="what the PDS has; nothing downloaded")


def cmd_pds(a) -> int:
    sols = [int(x) for x in a.sols]
    if len(sols) == 2 and sols[1] > sols[0] + 1:
        sols = list(range(sols[0], sols[1] + 1))
    root = Path(a.dest) if a.dest else default_root()
    if root is None and not a.list:
        print("ERROR: --dest (or RPP_PDS) is required")
        return 2
    rep = fetch_sols(root or Path("."), sols, a.cameras, a.product, a.match, a.list)
    for sol, by in sorted(rep["sols"].items()):
        print(f"sol {sol}: " + ", ".join(f"{c} {n}" for c, n in by.items()))
    for m in rep["missing_folders"]:
        print(f"not on the PDS (or not reachable): {m}")
    if not a.list:
        print(f"{rep.get('downloaded', 0)} files downloaded ({rep.get('bytes', 0) / 1e6:.0f} MB), {rep.get('kept', 0)} already "
              f"there, {len(rep.get('failed', []))} failed, into {root}")
    return 0 if rep["names"] and not rep.get("failed") else 1
