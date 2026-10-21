"""
The Mars 2020 example data set (Christian, 20 Oct 2026): Rockytop, sols 461-464 - the main Navcam images (the
Navcam 360 and its companions: sequences NCAM02xxx, NCAM03xxx, NCAM13xxx at the two end-of-drive waypoints), one
front Hazcam pair each on sols 461 and 464 and the eight Mastcam-Z 34 mm frames of ZCAM03381 (sol 464, ZL0 / ZR0 ...0340LMA01), zipped in the layout of the
local PDS archive (``datadrive/<sol>/ids/rdr/<ncam|fcam>/`` and ``zcam/``) and published as a GitHub Release asset.

    mppp example build --pds D:\\data\\m2020\\datadrive --zcam D:\\data\\m2020\\zcam --out D:\\code\\MPPP\\_transfer
    mppp example fetch [--dest DIR]       # download the release asset, check it, unpack; then set MPPP_PDS

The zip carries ``example_manifest.json`` (every file with its size and SHA-256).
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]

EXAMPLE_NAME = "mppp_example_m2020_rockytop"
EXAMPLE_TAG = "example-data-v1"
EXAMPLE_URL = f"https://github.com/ctate7163/MPPP/releases/download/{EXAMPLE_TAG}/{EXAMPLE_NAME}.zip"
EXAMPLE_SOLS = (461, 464)
ZCAM_SEQUENCE = "ZCAM03381"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# the files of the set (Christian, 20 Oct 2026): one image per eye and exposure (the largest product of it)
_N461 = "N0260630"
_N464 = "N0260756"
EXAMPLE_FILES = [
    # sol 461, waypoint 26/630: the Navcam 360 (NCAM03461, 3 exposures), its companions (NCAM02461 2, NCAM13461 3)
    f"NLF_0461_0707870987_044RAD_{_N461}NCAM03461_0A1195J01.IMG", f"NRF_0461_0707870987_044RAD_{_N461}NCAM03461_0A0195J01.IMG",
    f"NLF_0461_0707871166_243RAD_{_N461}NCAM03461_0A0195J01.IMG", f"NRF_0461_0707871166_243RAD_{_N461}NCAM03461_0A0195J01.IMG",
    f"NLF_0461_0707871340_883RAD_{_N461}NCAM03461_0A0195J01.IMG", f"NRF_0461_0707871340_883RAD_{_N461}NCAM03461_0A0195J01.IMG",
    f"NLF_0461_0707874229_177RAD_{_N461}NCAM02461_0A0195J02.IMG", f"NRF_0461_0707874229_177RAD_{_N461}NCAM02461_0A0195J01.IMG",
    f"NLF_0461_0707874458_288RAD_{_N461}NCAM02461_0A0195J01.IMG", f"NRF_0461_0707874458_288RAD_{_N461}NCAM02461_0A0195J01.IMG",
    f"NLF_0461_0707875039_745RAD_{_N461}NCAM13461_0A0195J01.IMG", f"NRF_0461_0707875039_745RAD_{_N461}NCAM13461_0A0195J01.IMG",
    f"NLF_0461_0707875150_237RAD_{_N461}NCAM13461_0A0195J01.IMG", f"NRF_0461_0707875150_237RAD_{_N461}NCAM13461_0A0195J01.IMG",
    f"NLF_0461_0707875261_552RAD_{_N461}NCAM13461_0A0195J01.IMG", f"NRF_0461_0707875261_552RAD_{_N461}NCAM13461_0A0195J01.IMG",
    # sol 461: one front Hazcam pair (the end-of-drive FHAZ02008)
    f"FLF_0461_0707870635_633RAD_{_N461}FHAZ02008_0A0085J02.IMG", f"FRF_0461_0707870635_633RAD_{_N461}FHAZ02008_0A0085J02.IMG",
    # sol 464, waypoint 26/756: NCAM03464 3, NCAM02464 2, NCAM13464 3
    f"NLF_0464_0708139366_801RAD_{_N464}NCAM03464_0A0195J01.IMG", f"NRF_0464_0708139366_801RAD_{_N464}NCAM03464_0A0195J01.IMG",
    f"NLF_0464_0708139537_820RAD_{_N464}NCAM03464_0A0195J01.IMG", f"NRF_0464_0708139537_820RAD_{_N464}NCAM03464_0A0195J01.IMG",
    f"NLF_0464_0708139709_055RAD_{_N464}NCAM03464_0A0195J01.IMG", f"NRF_0464_0708139709_055RAD_{_N464}NCAM03464_0A0195J01.IMG",
    f"NLF_0464_0708139859_356RAD_{_N464}NCAM02464_0A0195J01.IMG", f"NRF_0464_0708139859_356RAD_{_N464}NCAM02464_0A0195J01.IMG",
    f"NLF_0464_0708140116_668RAD_{_N464}NCAM02464_0A0195J01.IMG", f"NRF_0464_0708140116_668RAD_{_N464}NCAM02464_0A0195J01.IMG",
    f"NLF_0464_0708143598_121RAD_{_N464}NCAM13464_0A0195J01.IMG", f"NRF_0464_0708143598_121RAD_{_N464}NCAM13464_0A0195J01.IMG",
    f"NLF_0464_0708143725_291RAD_{_N464}NCAM13464_0A0195J01.IMG", f"NRF_0464_0708143725_291RAD_{_N464}NCAM13464_0A0195J01.IMG",
    f"NLF_0464_0708143819_551RAD_{_N464}NCAM13464_0A0195J01.IMG", f"NRF_0464_0708143819_551RAD_{_N464}NCAM13464_0A0195J01.IMG",
    # sol 464: one front Hazcam pair (FHAZ02008)
    f"FLF_0464_0708138950_977RAD_{_N464}FHAZ02008_0A00LLJ01.IMG", f"FRF_0464_0708138950_977RAD_{_N464}FHAZ02008_0A00LLJ01.IMG",
    # sol 464: Mastcam-Z 34 mm ZCAM03381, four stereo pairs (taken at 26/630 before the drive)
    f"ZL0_0464_0708130229_159RAD_{_N461}ZCAM03381_0340LMA01.IMG", f"ZR0_0464_0708130229_159RAD_{_N461}ZCAM03381_0340LMA01.IMG",
    f"ZL0_0464_0708130412_159RAD_{_N461}ZCAM03381_0340LMA01.IMG", f"ZR0_0464_0708130412_159RAD_{_N461}ZCAM03381_0340LMA01.IMG",
    f"ZL0_0464_0708130423_159RAD_{_N461}ZCAM03381_0340LMA01.IMG", f"ZR0_0464_0708130423_159RAD_{_N461}ZCAM03381_0340LMA01.IMG",
    f"ZL0_0464_0708130503_144RAD_{_N461}ZCAM03381_0340LMA01.IMG", f"ZR0_0464_0708130503_144RAD_{_N461}ZCAM03381_0340LMA01.IMG",
]


def _kind(name: str) -> str:
    return {"N": "navcam", "F": "hazcam", "Z": "zcam"}[name[0]]


def select_files(pds: PathLike, zcam: Optional[PathLike] = None, files: Sequence[str] = tuple(EXAMPLE_FILES)) -> Dict[str, Any]:
    """The files of the set (:data:`EXAMPLE_FILES`) in a local archive: {"navcam", "hazcam", "zcam": [Path],
    "missing": [name]}.  Navcam and Hazcam under ``pds/<sol>/ids/rdr/ncam|fcam/``; Mastcam-Z in ``zcam`` (flat, or
    a sol tree)."""
    pds = Path(pds)
    out: Dict[str, Any] = {"navcam": [], "hazcam": [], "zcam": [], "missing": []}
    for n in files:
        k = _kind(n)
        if k == "zcam":
            p = None
            if zcam:
                zr = Path(zcam)
                p = zr / n if (zr / n).is_file() else next(iter(zr.glob(f"*/**/{n}")), None)
        else:
            sol = int(n.split("_")[1])
            p = pds / f"{sol:05d}" / "ids" / "rdr" / ("ncam" if k == "navcam" else "fcam") / n
        if p is not None and Path(p).is_file():
            out[k].append(Path(p))
        else:
            out["missing"].append(n)
    return out


def _siblings(p: Path) -> List[Path]:
    """The product and its detached labels, if any (``.xml``, ``.LBL``)."""
    out = [p]
    for ext in (".xml", ".XML", ".lbl", ".LBL"):
        q = p.with_suffix(ext)
        if q.is_file():
            out.append(q)
    return out


def build_example(pds: PathLike, zcam: Optional[PathLike], out_dir: PathLike, name: str = EXAMPLE_NAME,
                  sols: Sequence[int] = EXAMPLE_SOLS) -> Dict[str, Any]:
    """Zip the example set into ``out_dir/<name>.zip`` in the archive layout, with ``example_manifest.json``.
    Returns the manifest (and the zip's path, size and SHA-256 - the number ``fetch`` checks)."""
    sel = select_files(pds, zcam)
    if sel["missing"]:
        raise FileNotFoundError(f"example set incomplete - not found under {pds} / {zcam}: " + ", ".join(sel["missing"]))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zp = out / f"{name}.zip"
    files = []
    with zipfile.ZipFile(zp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for kind in ("navcam", "hazcam", "zcam"):
            for p in sel[kind]:
                for q in _siblings(p):
                    if kind == "zcam":
                        arc = f"zcam/{q.name}"
                    else:
                        sol = int(q.name.split("_")[1])
                        arc = f"datadrive/{sol:05d}/ids/rdr/{'ncam' if kind == 'navcam' else 'fcam'}/{q.name}"
                    zf.write(q, arc)
                    files.append({"path": arc, "kind": kind, "bytes": q.stat().st_size, "sha256": _sha256(q)})
        man = {"name": name, "sols": list(sols), "waypoints": ["26/630", "26/756"], "zcam_sequence": ZCAM_SEQUENCE,
               "counts": {k: len(sel[k]) for k in ("navcam", "hazcam", "zcam")},
               "bytes": sum(f["bytes"] for f in files), "files": files,
               "use": "unzip, then MPPP_PDS=<folder> (it holds datadrive/ and zcam/)"}
        zf.writestr("example_manifest.json", json.dumps(man, indent=1))
    man.update({"zip": str(zp), "zip_bytes": zp.stat().st_size, "zip_sha256": _sha256(zp)})
    (out / f"{name}.sha256").write_text(f"{man['zip_sha256']}  {zp.name}\n", encoding="utf-8")
    return man


def fetch_example(dest: Optional[PathLike] = None, url: str = EXAMPLE_URL, sha256: Optional[str] = None,
                  force: bool = False) -> Path:
    """Download the example zip (the GitHub Release asset), check its SHA-256 when given (``sha256``, or the
    ``.sha256`` file next to the asset), unpack it into ``dest`` (default ``~/mppp_example``).  Returns ``dest``."""
    import urllib.request
    dest = Path(dest or Path.home() / "mppp_example").expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    zp = dest / url.rsplit("/", 1)[-1]
    if force or not zp.is_file():
        tmp = zp.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:          # noqa: S310 (a fixed https URL)
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        tmp.replace(zp)
    if sha256 is None:
        try:
            with urllib.request.urlopen(url.rsplit(".", 1)[0] + ".sha256") as r:   # noqa: S310
                sha256 = r.read().decode().split()[0]
        except Exception:                                                      # noqa: BLE001
            sha256 = None
    if sha256 and _sha256(zp) != sha256:
        raise ValueError(f"{zp}: SHA-256 does not match the release ({sha256[:12]}...); download again with --force")
    with zipfile.ZipFile(zp) as zf:
        zf.extractall(dest)
    return dest
