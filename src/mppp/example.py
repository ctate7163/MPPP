"""
The Mars 2020 example data set (Christian, 20 Oct 2026): Rockytop, sols 461-464 - the main Navcam images (the
Navcam 360 and its companions: sequences NCAM02xxx, NCAM03xxx, NCAM13xxx), the front Hazcams at the same waypoints
and the eight Mastcam-Z 34 mm frames of ZCAM03381 (sol 464, ZL0 / ZR0 ...0340LMA01), zipped in the layout of the
local PDS archive (``datadrive/<sol>/ids/rdr/<ncam|fcam>/`` and ``zcam/``) and published as a GitHub Release asset.

    mppp example build --pds D:\\data\\m2020\\datadrive --zcam D:\\data\\m2020\\zcam --out D:\\code\\MPPP\\_transfer
    mppp example fetch [--dest DIR]       # download the release asset, check it, unpack; then set MPPP_PDS

The zip carries ``example_manifest.json`` (every file with its size and SHA-256).
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]

EXAMPLE_NAME = "mppp_example_m2020_rockytop"
EXAMPLE_TAG = "example-data-v1"
EXAMPLE_URL = f"https://github.com/ctate7163/MPPP/releases/download/{EXAMPLE_TAG}/{EXAMPLE_NAME}.zip"
EXAMPLE_SOLS = (461, 464)
NAVCAM_SEQUENCES = r"NCAM(0[23]|13)\d{3}"          # the Navcam 360 and its companions (not NCAM00xxx drive images)
ZCAM_SEQUENCE = "ZCAM03381"
ZCAM_SOL = 464
ZCAM_PATTERN = r"^Z[LR]0_0464_\d{10}_\d{3}RAD_N\d{7}ZCAM03381_0340LMA\d{2}\.IMG$"

_NAME = re.compile(r"^(?P<inst>[A-Z]{2}[A-Z0-9])(?P<sfx>[A-Z_])_?(?P<sol>\d{4})_(?P<sclk>\d{10})_(?P<ms>\d{3})"
                   r"(?P<prod>[A-Z]{3})_N(?P<site>\d{3})(?P<drive>\d{4})(?P<seq>[A-Z_]{4}\d{5})_(?P<rest>\w+?)"
                   r"(?P<ver>J\d\d|A\d\d)?\.IMG$", re.I)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest(paths: Sequence[Path]) -> List[Path]:
    """One file per product (the same name without its version suffix J01/J02): the highest version."""
    best: Dict[str, Path] = {}
    for p in paths:
        stem = re.sub(r"(J|A)\d\d\.IMG$", "", p.name, flags=re.I)
        if stem not in best or p.name > best[stem].name:
            best[stem] = p
    return sorted(best.values())


def select_files(pds: PathLike, zcam: Optional[PathLike] = None, sols: Sequence[int] = EXAMPLE_SOLS) -> Dict[str, Any]:
    """The products of the example set in a local archive: {"navcam", "hazcam", "zcam": [Path], "waypoints":
    ["site drive"]}.  Navcam: NLF/NRF of the 360 sequences in ``sols``; front Hazcam: FLF/FRF at the Navcam's
    waypoints; Mastcam-Z: ZCAM03381 of sol 464 (``zcam``: a flat folder or a sol tree)."""
    pds = Path(pds)
    nav: List[Path] = []
    haz: List[Path] = []
    for sol in range(int(sols[0]), int(sols[1]) + 1):
        d = pds / f"{sol:05d}" / "ids" / "rdr"
        nav += [p for p in (d / "ncam").glob("N[LR]F_*.IMG") if re.search(NAVCAM_SEQUENCES, p.name)]
        haz += list((d / "fcam").glob("F[LR]F_*.IMG"))
    nav = _latest(nav)
    wps = sorted({m.group("site") + m.group("drive") for m in (_NAME.match(p.name) for p in nav) if m})
    haz = _latest([p for p in haz if (m := _NAME.match(p.name)) and (m.group("site") + m.group("drive")) in wps])
    z: List[Path] = []
    if zcam:
        zr = Path(zcam)
        cands = list(zr.glob("Z*_0464_*.IMG")) + list(zr.glob(f"{ZCAM_SOL:05d}/**/Z*.IMG"))
        z = _latest([p for p in cands if re.match(ZCAM_PATTERN, p.name, re.I)])
    return {"navcam": nav, "hazcam": haz, "zcam": z, "waypoints": wps}


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
    sel = select_files(pds, zcam, sols)
    if not sel["navcam"] or not sel["zcam"]:
        raise FileNotFoundError(f"example set incomplete: {len(sel['navcam'])} Navcam, {len(sel['zcam'])} Mastcam-Z "
                                f"files under {pds} / {zcam}")
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
        man = {"name": name, "sols": list(sols), "waypoints": sel["waypoints"], "zcam_sequence": ZCAM_SEQUENCE,
               "counts": {k: len(sel[k]) for k in ("navcam", "hazcam", "zcam")},
               "bytes": sum(f["bytes"] for f in files), "files": files,
               "use": "unzip, then MPPP_PDS=<folder>/datadrive and the Mastcam-Z frames in <folder>/zcam"}
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
