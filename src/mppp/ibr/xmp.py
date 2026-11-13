"""
XMP for the top-down stereo 360 products (``*_360_3DTD*.jpg`` / ``.png``), written into the files without re-encoding
them (0.97.4; Christian, 13 Nov 2026: "inject XMP stereoscopic metadata ... so that common viewers know").

What is written, and what it is worth: there is no single standard for a stereo still.  The packet carries
- ``GPano`` (Google Photo Sphere, the one most 360 viewers read): equirectangular, the whole sphere, the heading of the
  image centre;
- ``GSpherical`` (the spherical video fields, read by some players for stills too): ``StereoMode = top-bottom``;
- ``mppp``: eye order (left on top), the eye's size, the station.
A viewer that reads GPano alone will treat the square as one mono sphere; the file name's ``_360_3DTD`` is what the
Quest's own viewers key on.  JPEG: one APP1 segment after the JFIF header.  PNG: an ``iTXt`` chunk with the keyword
``XML:com.adobe.xmp`` after ``IHDR``.  An XMP packet already in the file is replaced.
"""
from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]
JPEG_XMP = b"http://ns.adobe.com/xap/1.0/\x00"
PNG_SIG = b"\x89PNG\r\n\x1a\n"
PNG_XMP_KEY = b"XML:com.adobe.xmp"
STACK_GLOBS = ("*_360_3DTD*.jpg", "*_360_3DTD*.png")
_BLOCK = 1 << 24


def stereo_xmp(width: int, height: int, software: str, heading_centre_deg: float = 180.0,
               extra: Optional[Dict[str, Any]] = None) -> bytes:
    """The packet for a ``width`` x ``height`` image holding two equirectangular eyes, the left on top."""
    from xml.sax.saxutils import escape
    f = {"GPano:UsePanoramaViewer": "True", "GPano:ProjectionType": "equirectangular",
         "GPano:StitchingSoftware": software, "GPano:PoseHeadingDegrees": f"{heading_centre_deg % 360.0:.2f}",
         "GPano:CroppedAreaImageWidthPixels": width, "GPano:CroppedAreaImageHeightPixels": height,
         "GPano:FullPanoWidthPixels": width, "GPano:FullPanoHeightPixels": height,
         "GPano:CroppedAreaLeftPixels": 0, "GPano:CroppedAreaTopPixels": 0,
         "GSpherical:Spherical": "true", "GSpherical:Stitched": "true", "GSpherical:StitchingSoftware": software,
         "GSpherical:ProjectionType": "equirectangular", "GSpherical:StereoMode": "top-bottom",
         "mppp:StereoLayout": "top-bottom", "mppp:TopEye": "left", "mppp:EyeWidthPixels": width,
         "mppp:EyeHeightPixels": height // 2, "mppp:AzimuthOfColumn0Deg": f"{(heading_centre_deg - 180.0) % 360.0:.2f}"}
    for k, v in (extra or {}).items():
        f[f"mppp:{k}"] = v
    body = "\n".join(f"   <{k}>{escape(str(v))}</{k}>" for k, v in f.items())
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="MPPP">\n'
            ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            '  <rdf:Description rdf:about=""\n'
            '    xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"\n'
            '    xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/"\n'
            '    xmlns:mppp="https://github.com/ctate7163/MPPP/ns/1.0/">\n'
            f'{body}\n'
            '  </rdf:Description>\n </rdf:RDF>\n</x:xmpmeta>\n<?xpacket end="w"?>').encode("utf-8")


def _replace(path: Path, head: bytes, rest_from: int) -> None:
    """``path`` rewritten as ``head`` + its bytes from ``rest_from`` (through a temporary file beside it)."""
    tmp = path.with_name(path.name + ".xmp_tmp")
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(head)
        src.seek(rest_from)
        while True:
            b = src.read(_BLOCK)
            if not b:
                break
            dst.write(b)
    os.replace(tmp, path)


def _jpeg_header(fh) -> Tuple[List[bytes], int]:
    """The marker segments before the first that is not APPn / COM (each whole, with its marker), and the offset of
    what follows."""
    if fh.read(2) != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    segs, pos = [], 2
    while True:
        m = fh.read(4)
        if len(m) < 4 or m[0] != 0xFF or not (0xE0 <= m[1] <= 0xEF or m[1] == 0xFE):
            return segs, pos
        n = struct.unpack(">H", m[2:4])[0]
        segs.append(m + fh.read(n - 2))
        pos += 2 + n


def inject_jpeg(path: PathLike, xmp: bytes) -> None:
    path = Path(path)
    if len(xmp) + len(JPEG_XMP) + 2 > 65535:
        raise ValueError("the XMP packet does not fit one JPEG segment")
    with open(path, "rb") as fh:
        segs, pos = _jpeg_header(fh)
    keep = [s for s in segs if not (s[1] == 0xE1 and s[4:4 + len(JPEG_XMP)] == JPEG_XMP)]
    seg = b"\xff\xe1" + struct.pack(">H", len(xmp) + len(JPEG_XMP) + 2) + JPEG_XMP + xmp
    first = [s for s in keep if s[1] == 0xE0][:1]                        # the JFIF header stays first
    _replace(path, b"\xff\xd8" + b"".join(first) + seg + b"".join(s for s in keep if s not in first), pos)


def _png_chunks_before_idat(fh) -> Tuple[List[Tuple[bytes, bytes]], int]:
    if fh.read(8) != PNG_SIG:
        raise ValueError("not a PNG file")
    chunks, pos = [], 8
    while True:
        h = fh.read(8)
        if len(h) < 8:
            return chunks, pos
        n, typ = struct.unpack(">I", h[:4])[0], h[4:8]
        if typ == b"IDAT":
            return chunks, pos
        chunks.append((typ, fh.read(n)))
        fh.read(4)
        pos += 12 + n


def _png_chunk(typ: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)


def inject_png(path: PathLike, xmp: bytes) -> None:
    path = Path(path)
    with open(path, "rb") as fh:
        chunks, pos = _png_chunks_before_idat(fh)
    if not chunks or chunks[0][0] != b"IHDR":
        raise ValueError("no IHDR chunk")
    keep = [(t, d) for t, d in chunks if not (t == b"iTXt" and d.startswith(PNG_XMP_KEY + b"\x00"))]
    itxt = PNG_XMP_KEY + b"\x00\x00\x00\x00\x00" + xmp                  # keyword, not compressed, no language, no translation
    head = PNG_SIG + _png_chunk(*keep[0]) + _png_chunk(b"iTXt", itxt) + b"".join(_png_chunk(t, d) for t, d in keep[1:])
    _replace(path, head, pos)


def read_xmp(path: PathLike) -> Optional[bytes]:
    """The XMP packet of a JPEG or PNG file, or None."""
    path = Path(path)
    with open(path, "rb") as fh:
        if path.suffix.lower() in (".jpg", ".jpeg"):
            for s in _jpeg_header(fh)[0]:
                if s[1] == 0xE1 and s[4:4 + len(JPEG_XMP)] == JPEG_XMP:
                    return s[4 + len(JPEG_XMP):]
            return None
        for t, d in _png_chunks_before_idat(fh)[0]:
            if t == b"iTXt" and d.startswith(PNG_XMP_KEY + b"\x00"):
                return d[len(PNG_XMP_KEY) + 5:]
    return None


def image_size(path: PathLike) -> Tuple[int, int]:
    """(width, height) from the file's header, without decoding it."""
    path = Path(path)
    with open(path, "rb") as fh:
        if path.suffix.lower() == ".png":
            b = fh.read(24)
            return struct.unpack(">II", b[16:24])
        fh.read(2)
        while True:
            m = fh.read(4)
            if len(m) < 4:
                raise ValueError(f"{path}: no JPEG frame header")
            n = struct.unpack(">H", m[2:4])[0]
            if 0xC0 <= m[1] <= 0xCF and m[1] not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", fh.read(5)[1:5])
                return w, h
            fh.seek(n - 2, 1)


def tag_stereo(path: PathLike, software: Optional[str] = None, heading_centre_deg: float = 180.0,
               extra: Optional[Dict[str, Any]] = None) -> bool:
    """Write the stereo packet into one stacked product; False for a file that is not a square pair."""
    from .. import __version__
    path = Path(path)
    w, h = image_size(path)
    if h != w:                                                           # two w x w/2 eyes make a square
        return False
    xmp = stereo_xmp(w, h, software or f"MPPP {__version__}", heading_centre_deg, extra)
    (inject_png if path.suffix.lower() == ".png" else inject_jpeg)(path, xmp)
    return True


def tag_folder(folder: PathLike, heading_centre_deg: float = 180.0, extra: Optional[Dict[str, Any]] = None) -> List[str]:
    """Every ``*_360_3DTD*`` JPEG and PNG of ``folder`` tagged; the names done."""
    done = []
    for g in STACK_GLOBS:
        for f in sorted(Path(folder).glob(g)):
            try:
                if tag_stereo(f, heading_centre_deg=heading_centre_deg, extra=extra):
                    done.append(f.name)
            except (OSError, ValueError, struct.error):
                continue
    return done


def fields_xmp(fields: Dict[str, Any]) -> bytes:
    """A packet of ``mppp:`` fields only (0.97.7: the value scale of a per-image picture, :mod:`mppp.ibr.scales`)."""
    from xml.sax.saxutils import escape
    body = "\n".join(f"   <mppp:{k}>{escape(str(v))}</mppp:{k}>" for k, v in fields.items())
    return ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="MPPP">\n'
            ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
            '  <rdf:Description rdf:about=""\n'
            '    xmlns:mppp="https://github.com/ctate7163/MPPP/ns/1.0/">\n'
            f'{body}\n'
            '  </rdf:Description>\n </rdf:RDF>\n</x:xmpmeta>\n<?xpacket end="w"?>').encode("utf-8")


def tag_fields(path: PathLike, fields: Dict[str, Any]) -> None:
    """Write ``mppp:`` fields into a JPEG or PNG (no re-encoding)."""
    path = Path(path)
    (inject_png if path.suffix.lower() == ".png" else inject_jpeg)(path, fields_xmp(fields))


def read_fields(path: PathLike) -> Dict[str, str]:
    """The ``mppp:`` fields of an image's XMP packet ({} without one)."""
    import re
    from xml.sax.saxutils import unescape
    x = read_xmp(path)
    if not x:
        return {}
    return {k: unescape(v) for k, v in re.findall(r"<mppp:(\w+)>(.*?)</mppp:\1>", x.decode("utf-8", "replace"), flags=re.S)}
