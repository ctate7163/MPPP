"""
Image writers.  Pixel values are never modified here.

* PNG16 / PNG8: RGB or RGBA (alpha = reconstruction mask), written with OpenCV
  (Pillow cannot write 16-bit RGBA).  Metadata is then inserted as standard
  PNG chunks: ``iTXt`` key/values, an XMP packet, and an ``eXIf`` chunk
  carrying camera tags and the Mars position in EXIF-GPS form.
* TIFF16: tifffile, deflate, with XMP (tag 700), ImageDescription (270) and
  the same EXIF/GPS IFDs.

EXIF GPS on Mars: latitude = planetocentric, longitude = east-positive,
altitude = elevation above the areoid (GPSAltitudeRef 1 when negative),
GPSMapDatum = "Mars 2000 sphere (IAU), planetocentric".  Software that assumes
WGS-84 must be told the coordinate system (Metashape: set the chunk CRS).
"""
from __future__ import annotations

import csv
import json
import struct
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
XMP_NS = "https://github.com/msss/mppp/ns/0.1/"          # identifier only; not resolvable
PNG_SIG = b"\x89PNG\r\n\x1a\n"


# ----------------------------------------------------------------- metadata
def _rational(x: float, max_den: int = 1_000_000):
    f = Fraction(abs(float(x))).limit_denominator(max_den)
    return (f.numerator, f.denominator)


def _dms(deg: float):
    deg = abs(float(deg))
    d = int(deg)
    m = int((deg - d) * 60)
    s = (deg - d - m / 60.0) * 3600.0
    return (_rational(d, 1), _rational(m, 1), _rational(s, 10_000_000))


def build_exif(meta: Dict[str, Any]) -> Optional[bytes]:
    """EXIF blob (TIFF structure, without the 'Exif\\0\\0' prefix the PNG eXIf chunk forbids)."""
    try:
        import piexif
    except ImportError:
        return None
    intr = meta.get("intrinsics", {})
    zeroth = {
        piexif.ImageIFD.Make: "NASA/JPL Mars 2020 Perseverance",
        piexif.ImageIFD.Model: str(meta.get("camera_group", "")),
        piexif.ImageIFD.Software: f"MPPP {meta.get('mppp_version', '')}",
        piexif.ImageIFD.ImageDescription: str(meta.get("source_product", "")),
    }
    exif = {}
    if meta.get("start_time_utc"):
        t = str(meta["start_time_utc"]).replace("-", ":").replace("T", " ")[:19]
        exif[piexif.ExifIFD.DateTimeOriginal] = t
    if meta.get("zoom_mm"):
        exif[piexif.ExifIFD.FocalLength] = _rational(meta["zoom_mm"], 10)
    gps = {}
    geo = meta.get("geo") or {}
    if geo.get("lat_deg") is not None and geo.get("lon_east_deg") is not None:
        lat, lon = float(geo["lat_deg"]), float(geo["lon_east_deg"])
        lon = (lon + 180.0) % 360.0 - 180.0
        gps = {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
            piexif.GPSIFD.GPSLatitude: _dms(lat),
            piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
            piexif.GPSIFD.GPSLongitude: _dms(lon),
            piexif.GPSIFD.GPSMapDatum: "Mars 2000 sphere (IAU), planetocentric",
        }
        if geo.get("elev_geoid_m") is not None:
            alt = float(geo["elev_geoid_m"])
            gps[piexif.GPSIFD.GPSAltitudeRef] = 1 if alt < 0 else 0
            gps[piexif.GPSIFD.GPSAltitude] = _rational(alt, 1000)
    blob = piexif.dump({"0th": zeroth, "Exif": exif, "GPS": gps})
    return blob[6:] if blob.startswith(b"Exif\x00\x00") else blob


def build_xmp(meta: Dict[str, Any]) -> str:
    from xml.sax.saxutils import escape
    payload = escape(json.dumps(meta, separators=(",", ":"), default=str))
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:mppp="{XMP_NS}">'
            f'<mppp:payload>{payload}</mppp:payload>'
            '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


def simple_tags(meta: Dict[str, Any], bits: int, channels: int) -> Dict[str, str]:
    intr = meta.get("intrinsics", {})
    K = intr.get("K")
    tags = {
        "Software": f"MPPP {meta.get('mppp_version', '')}",
        "Source": str(meta.get("source_product", "")),
        "CameraGroup": str(meta.get("camera_group", "")),
        "BitDepthPerChannel": str(bits), "Channels": str(channels),
        "ColorSpace": "linear_rgb" if bits == 16 else "see mppp config (gamma)",
        "AlphaSemantics": "reconstruction mask (max = include)" if channels == 4 else "none",
        "Undistorted": str(bool(meta.get("undistorted"))).lower(),
        "LMST": str(meta.get("LMST")), "LTST": str(meta.get("LTST")),
    }
    if K:
        tags["FocalPixels"] = f"{K[0][0]:.8g},{K[1][1]:.8g}"
        tags["PrincipalPointPx"] = f"{K[0][2]:.8g},{K[1][2]:.8g}"
        tags["PixelOrigin"] = str(intr.get("pixel_origin"))
        tags["DistortionOpenCV"] = json.dumps(intr.get("dist_opencv"), separators=(",", ":"))
    return tags


# ---------------------------------------------------------------------- PNG
def _chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def _itxt(key: str, text: str, compress: bool = False) -> bytes:
    body = text.encode("utf-8")
    flag = b"\x01\x00" if compress else b"\x00\x00"
    if compress:
        body = zlib.compress(body)
    return _chunk(b"iTXt", key.encode("latin-1")[:79] + b"\x00" + flag + b"\x00" + b"\x00" + body)


def insert_png_chunks(png: bytes, chunks: Iterable[bytes]) -> bytes:
    """Insert ancillary chunks right after IHDR (valid position for eXIf and iTXt)."""
    if png[:8] != PNG_SIG:
        raise ValueError("not a PNG")
    ihdr_end = 8 + 12 + struct.unpack(">I", png[8:12])[0]
    return png[:ihdr_end] + b"".join(chunks) + png[ihdr_end:]


def read_png_chunks(path: PathLike) -> Dict[str, List[bytes]]:
    data = Path(path).read_bytes()
    pos, out = 8, {}
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8].decode("latin-1")
        out.setdefault(ctype, []).append(data[pos + 8:pos + 8 + n])
        pos += 12 + n
    return out


def save_png(image: np.ndarray, path: PathLike, meta: Optional[Dict[str, Any]] = None,
             embed_gps: bool = True, compression: int = 4) -> Path:
    if image.dtype not in (np.uint8, np.uint16) or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"PNG writer needs uint8/uint16 HxWx3|4, got {image.dtype} {image.shape}")
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    code = cv2.COLOR_RGBA2BGRA if image.shape[2] == 4 else cv2.COLOR_RGB2BGR
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image, code), [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise IOError(f"PNG encoding failed for {path}")
    png = buf.tobytes()
    if meta is not None:
        bits = image.dtype.itemsize * 8
        chunks = [_itxt(k, v) for k, v in simple_tags(meta, bits, image.shape[2]).items()]
        chunks.append(_itxt("XML:com.adobe.xmp", build_xmp(meta)))
        exif = build_exif(meta if embed_gps else {**meta, "geo": {}})
        if exif:
            chunks.insert(0, _chunk(b"eXIf", exif))
        png = insert_png_chunks(png, chunks)
    path.write_bytes(png)
    return path


# --------------------------------------------------------------------- TIFF
def save_tiff16(image: np.ndarray, path: PathLike, meta: Optional[Dict[str, Any]] = None) -> Path:
    import tifffile
    if image.dtype != np.uint16:
        raise ValueError("TIFF16 writer needs uint16")
    path = Path(path).with_suffix(".tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    extratags = []
    if meta is not None:
        xmp = build_xmp(meta).encode("utf-8")
        extratags = [(700, "B", len(xmp), xmp, True),
                     (270, "s", 0, json.dumps(simple_tags(meta, 16, image.shape[2]), separators=(",", ":")), True)]
    tifffile.imwrite(path, image, photometric="rgb", metadata=None, extratags=extratags,
                     compression="deflate",
                     extrasamples=["unassalpha"] if image.shape[2] == 4 else None)
    return path


# ----------------------------------------------------------------- run files
def save_mask(mask: np.ndarray, path: PathLike) -> Path:
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255):
        raise IOError(path)
    return path


def reference_offset(refs: List[list]) -> np.ndarray:
    """floor(mean/10)*10 — keeps coordinates small for single-precision software."""
    xyz = np.array([r[1:4] for r in refs], dtype=np.float64)
    return np.floor(xyz.mean(axis=0) / 10.0) * 10.0


def save_references(refs: List[list], path: PathLike, offset: Optional[np.ndarray] = None,
                    ext: str = ".png") -> Path:
    """Tab-separated: filename, X(E), Y(N), Z(U), yaw, pitch, roll — Metashape reference import."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    off = np.zeros(3) if offset is None else np.asarray(offset, float)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["filename", "X", "Y", "Z", "Yaw", "Pitch", "Roll"])
        for r in refs:
            xyz = np.asarray(r[1:4], float) - off
            w.writerow([str(r[0]) + ext, *(f"{v:.6f}" for v in xyz), *(f"{v:.6f}" for v in r[4:7])])
    return path
