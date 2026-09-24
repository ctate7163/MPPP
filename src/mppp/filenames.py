"""
Mars 2020 PDS image file-name parsing.

PDS file names are never altered by MPPP (provenance); output files keep the
PDS stem and only change the extension.

Fixed-width layout (M2020 Camera SIS), e.g.::

    ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG
    0         1         2         3         4         5
    012345678901234567890123456789012345678901234567890123

    [0:2]   instrument   NL NR FL FR RL RR ZL ZR SC ...
    [2]     colour/filter (F = RGB, M = mono/VCE, 0-7 = Mastcam-Z filter)
    [4:8]   sol
    [9:19]  SCLK seconds        [20:23] SCLK milliseconds
    [23:26] product type (RAD, IOF, EDR, FDR ...)
    [26]    geometry  (_ raw, L linearised)      [27] thumbnail flag
    [28:31] site      [31:35] drive
    [35:44] sequence id
    [44:48] camera specific: '_034' = Mastcam-Z focal length in mm
    [48]    downsample code: 0 full, 1 half, 2 quarter, 3 eighth
    [49:51] compression   [51] producer   [52:54] version
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union

CAMERA_FAMILIES = {
    "N": "Navcam", "F": "Front Hazcam", "R": "Rear Hazcam",
    "Z": "Mastcam-Z", "S": "SuperCam RMI", "C": "Cachecam",
}
_DOWNSAMPLE = {"0": 1.0, "1": 0.5, "2": 0.25, "3": 0.125}


@dataclass(frozen=True)
class M2020Filename:
    name: str
    stem: str
    instrument: str          # e.g. 'NL'
    family: str              # e.g. 'N'
    eye: Optional[str]       # 'L' | 'R' | None
    filter: str              # 3rd character
    camera_code: str         # first three characters, e.g. 'NLF'
    sol: int
    sclk: float              # seconds
    sclk_key: str            # '0729888971_069' — shared by the two eyes of a stereo pair
    product_type: str
    geometry: str
    thumbnail: bool
    site: Optional[int]
    drive: Optional[int]
    sequence: str
    camera_specific: str
    zoom_mm: Optional[int]   # Mastcam-Z only
    downsample_code: str
    downsample_scale: float
    compression: str
    producer: str
    version: int
    camera_group: str        # intrinsics group: camera + zoom (Z) or + resolution (ECAM)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def stereo_partner_stem(self) -> Optional[str]:
        """Stem of the other eye (differs only in the 2nd character)."""
        if self.eye is None:
            return None
        other = "R" if self.eye == "L" else "L"
        return self.stem[0] + other + self.stem[2:]


def _int_or_none(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:        # alphanumeric overflow encoding of site/drive
        return None


def parse_filename(path: Union[str, Path]) -> M2020Filename:
    p = Path(path)
    stem = p.stem
    if len(stem) < 54:
        raise ValueError(f"Not an M2020 PDS image name (stem shorter than 54 characters): {p.name}")
    s = stem.upper()
    family = s[0]
    eye = s[1] if s[1] in ("L", "R") else None
    try:
        sol = int(s[4:8])
        sclk = float(s[9:19]) + float(s[20:23]) / 1000.0
        version = int(s[52:54])
    except ValueError as e:
        raise ValueError(f"Cannot parse sol / SCLK / version from: {p.name}") from e

    ds_code = s[48]
    zoom = _int_or_none(s[45:48]) if family == "Z" else None
    if family == "Z" and zoom is not None:
        group = f"{s[:2]}{zoom:03d}"              # one group per eye and zoom, e.g. ZL034
    else:
        group = f"{s[:2]}{ds_code}"               # one group per eye and resolution, e.g. NL0

    return M2020Filename(
        name=p.name, stem=stem, instrument=s[:2], family=family, eye=eye, filter=s[2],
        camera_code=s[:3], sol=sol, sclk=sclk, sclk_key=s[9:23],
        product_type=s[23:26], geometry=s[26], thumbnail=(s[27] == "T"),
        site=_int_or_none(s[28:31]), drive=_int_or_none(s[31:35]),
        sequence=s[35:44], camera_specific=s[44:48], zoom_mm=zoom,
        downsample_code=ds_code, downsample_scale=_DOWNSAMPLE.get(ds_code, 1.0),
        compression=s[49:51], producer=s[51], version=version, camera_group=group,
    )
