"""
The rover as a textured surface for the viewer (0.98.2, experimental).

Christian, 14 Nov 2026: "Prepare to render a rover that is textured from the navcam images. You may need to
retopologize the rover surface to get a projection optimized for the limited angles and resolutions from which the
ncam sees the rover. rendering a retextured rover is an experiment, so keep the ods with the rover".

The Navcam sees the rover from one place only: the mast head, turning about the station's pivot.  So the surface
is built from that place.  The panorama from the pivot (``mppp ods --eyes left right centre``: ``_ods_centre.png``)
already holds the Navcam frames projected onto the rover model, and ``_ods_centre_depth.npz`` its range and which
pixels are the rover.  This module turns the two into a mesh:

- one vertex every ``step`` pixels of the panorama where the rover is, at its range along the pixel's ray;
- two triangles per grid cell whose four corners are rover and whose ranges agree within ``EDGE_RATIO`` (a larger
  jump is an edge of the rover: no surface is stretched across it);
- the texture is the panorama's rows that hold the rover, and a vertex's texture coordinate is its own pixel.

Every part of the texture is therefore at the resolution and from the direction the cameras had, and what the
cameras never saw (the rover's sides and underside) is not in the mesh.  The file is a binary glTF (``.glb``) in
metres about the pivot: x east, y up, z south (the viewer page's frame; right-handed).

``python -m mppp.ibr.rover_surface FOLDER`` (the ``with_rover`` folder of a station run).
"""
from __future__ import annotations

import json
import logging
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
log = logging.getLogger(__name__)

SUFFIX = "rover_surface.glb"
DEFAULT_STEP = 4          # panorama pixels between vertices (8192 wide: 2.7 mrad, 3 mm at 1 m)
EDGE_RATIO = 1.08         # a cell whose farthest corner is more than this x its nearest is an edge, left open
MAX_RANGE_M = 6.0         # the rover is nearer than this from the pivot
TEXTURE_QUALITY = 90


def directions(cols: np.ndarray, rows: np.ndarray, width: int, height: int, azimuth0_deg: float = 0.0) -> np.ndarray:
    """Unit directions (..., 3) in the page's frame (x east, y up, z south) of panorama pixels (``cols``, ``rows``
    broadcast): the same rays as :func:`mppp.ibr.ods.ods_rays` from the pivot."""
    az = (np.asarray(cols, float) + 0.5) / width * 2 * np.pi + np.radians(float(azimuth0_deg))
    el = np.pi / 2 - (np.asarray(rows, float) + 0.5) / height * np.pi
    ce = np.cos(el)
    return np.stack(np.broadcast_arrays(ce * np.sin(az), np.sin(el), -ce * np.cos(az)), axis=-1)


def build(rng: np.ndarray, rover: np.ndarray, azimuth0_deg: float = 0.0, step: int = DEFAULT_STEP,
          edge_ratio: float = EDGE_RATIO, max_range_m: float = MAX_RANGE_M,
          exclude_below_deg: Optional[float] = None) -> Dict[str, Any]:
    """The grid mesh of the rover pixels.  ``rng`` (H, W) metres from the pivot, ``rover`` (H, W) bool.
    ``exclude_below_deg``: pixels within this angle of straight down are left out (the nadir mark is painted
    there).  Returns {"vertices" (n, 3) float32, "uv" (n, 2) float32 in the cropped texture, "triangles" (m, 3)
    uint32, "rows" (r0, r1) of the panorama the texture holds}; no vertices when the rover is not in the picture."""
    rng = np.asarray(rng, np.float32)
    H, W = rng.shape
    step = max(1, int(step))
    ok = np.asarray(rover, bool) & np.isfinite(rng) & (rng > 0) & (rng < max_range_m)
    if exclude_below_deg:
        cut = int(np.floor(H * (1.0 - float(exclude_below_deg) / 180.0)))
        ok[cut:] = False
    empty = {"vertices": np.zeros((0, 3), np.float32), "uv": np.zeros((0, 2), np.float32),
             "triangles": np.zeros((0, 3), np.uint32), "rows": (0, 0)}
    if not ok.any():
        return empty
    rr = np.flatnonzero(ok.any(axis=1))
    r0, r1 = int(rr[0]), int(rr[-1]) + 1
    rows = np.arange(r0, r1, step)
    cols = np.arange(0, W, step)                      # the last column joins the first again (360 degrees)
    g_ok = ok[np.ix_(rows, cols)]
    g_r = rng[np.ix_(rows, cols)]
    nr, nc = g_ok.shape
    nxt = np.roll(np.arange(nc), -1)
    c00, c01 = g_ok[:-1], g_ok[:-1][:, nxt]
    c10, c11 = g_ok[1:], g_ok[1:][:, nxt]
    q = np.stack([g_r[:-1], g_r[:-1][:, nxt], g_r[1:], g_r[1:][:, nxt]])
    cell = c00 & c01 & c10 & c11
    with np.errstate(invalid="ignore", divide="ignore"):
        cell &= (q.max(axis=0) <= edge_ratio * q.min(axis=0))
    if not cell.any():
        return empty
    idx = np.arange(nr * nc).reshape(nr, nc)
    i00, i01, i10, i11 = idx[:-1], idx[:-1][:, nxt], idx[1:], idx[1:][:, nxt]
    tri = np.concatenate([np.stack([i00[cell], i10[cell], i01[cell]], 1), np.stack([i01[cell], i10[cell], i11[cell]], 1)])
    used = np.zeros(nr * nc, bool)
    used[tri.ravel()] = True
    new = np.cumsum(used) - 1
    R, C = np.meshgrid(rows, cols, indexing="ij")
    d = directions(C, R, W, H, azimuth0_deg).reshape(-1, 3)
    V = (d * g_r.reshape(-1, 1))[used].astype(np.float32)
    # the wrap cells use column 0's vertices with u near 0: their texture would run backwards across the picture, so
    # the seam's cells are given their own copies of those vertices with u = 1 + ...
    u = (C.reshape(-1) + 0.5) / W
    v = (R.reshape(-1) + 0.5 - r0) / float(r1 - r0)
    UV = np.stack([u, v], 1)[used].astype(np.float32)
    tri = new[tri].astype(np.uint32)
    seam = (np.abs(UV[tri][:, :, 0].max(axis=1) - UV[tri][:, :, 0].min(axis=1)) > 0.5)
    if seam.any():
        st = tri[seam]
        low = UV[st][:, :, 0] < 0.5
        ids = np.unique(st[low])
        copy = {int(i): len(V) + k for k, i in enumerate(ids)}
        V = np.concatenate([V, V[ids]])
        UV = np.concatenate([UV, UV[ids] + np.array([1.0, 0.0], np.float32)])
        st2 = st.copy()
        st2[low] = np.array([copy[int(i)] for i in st[low]], np.uint32)
        tri[seam] = st2
    return {"vertices": V, "uv": UV, "triangles": tri, "rows": (r0, r1)}


def glb_bytes(vertices: np.ndarray, uv: np.ndarray, triangles: np.ndarray, jpeg: bytes, extras: Optional[Dict[str, Any]] = None) -> bytes:
    """A binary glTF 2.0 file: one mesh with positions, texture coordinates and an unlit, double-sided material
    whose colour is the JPEG (the texture repeats in u, for the seam's vertices at u > 1)."""
    V = np.ascontiguousarray(vertices, "<f4")
    T = np.ascontiguousarray(uv, "<f4")
    F = np.ascontiguousarray(triangles, "<u4").reshape(-1)
    parts, views = [], []

    def add(b: bytes, target: Optional[int] = None) -> int:
        off = sum(len(p) for p in parts)
        parts.append(b + b"\x00" * (-len(b) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(b)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    bv_v, bv_t, bv_f, bv_i = add(V.tobytes(), 34962), add(T.tobytes(), 34962), add(F.tobytes(), 34963), add(bytes(jpeg))
    doc = {
        "asset": {"version": "2.0", "generator": "mppp.ibr.rover_surface"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "rover_surface"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0, "mode": 4}]}],
        "accessors": [
            {"bufferView": bv_v, "componentType": 5126, "count": int(len(V)), "type": "VEC3",
             "min": [float(x) for x in V.min(axis=0)], "max": [float(x) for x in V.max(axis=0)]},
            {"bufferView": bv_t, "componentType": 5126, "count": int(len(T)), "type": "VEC2"},
            {"bufferView": bv_f, "componentType": 5125, "count": int(len(F)), "type": "SCALAR"}],
        "materials": [{"doubleSided": True, "extensions": {"KHR_materials_unlit": {}},
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "textures": [{"source": 0, "sampler": 0}],
        "samplers": [{"wrapS": 10497, "wrapT": 33071, "magFilter": 9729, "minFilter": 9987}],
        "images": [{"bufferView": bv_i, "mimeType": "image/jpeg"}],
        "bufferViews": views, "buffers": [{"byteLength": sum(len(p) for p in parts)}],
    }
    if extras:
        doc["asset"]["extras"] = extras
    js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    js += b" " * (-len(js) % 4)
    binary = b"".join(parts)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)


def read_glb(data: bytes) -> Tuple[Dict[str, Any], bytes]:
    """(the JSON document, the binary chunk) of a .glb written by :func:`glb_bytes`."""
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError("not a binary glTF 2.0 file")
    n, kind = struct.unpack_from("<I4s", data, 12)
    doc = json.loads(data[20:20 + n].decode("utf-8"))
    m, kind2 = struct.unpack_from("<I4s", data, 20 + n)
    return doc, data[28 + n:28 + n + m]


def from_folder(folder: PathLike, step: int = DEFAULT_STEP, out: Optional[PathLike] = None,
                texture_width: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The rover surface of one ``with_rover`` folder: reads ``*_ods_centre_depth.npz`` (with its ``rover`` array,
    written from 0.98.2 on when the centre panorama is rendered) and ``*_ods_centre.png``, writes
    ``<prefix>_rover_surface.glb`` and ``<prefix>_rover_surface.json`` beside them.  None, with the reason logged,
    when the folder has no centre panorama or no rover flag."""
    import cv2
    t0 = time.time()
    d = Path(folder)
    npz = next(iter(sorted(d.glob("*ods_centre_depth.npz"))), None)
    if npz is None:
        log.warning("rover surface: no *_ods_centre_depth.npz in %s (mppp ods --eyes left right centre)", d)
        return None
    stem = npz.name[:-len("ods_centre_depth.npz")]
    img_f = d / (stem + "ods_centre.png")
    with np.load(npz) as z:
        if "rover" not in z.files:
            log.warning("rover surface: %s has no rover array (render the centre panorama with 0.98.2 or later)", npz.name)
            return None
        rng, rover = z["range"], z["rover"].astype(bool)
    if not img_f.is_file():
        log.warning("rover surface: %s is missing", img_f.name)
        return None
    img = cv2.imread(str(img_f), cv2.IMREAD_COLOR)
    if img is None or img.shape[:2] != rng.shape:
        log.warning("rover surface: %s does not match the depth's size %s", img_f.name, rng.shape)
        return None
    az0, below = 0.0, None
    for j in sorted(d.glob("*ods.json")):
        try:
            rec = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        az0 = float(rec.get("azimuth0_deg") or 0.0)
        below = (rec.get("nadir_mark") or {}).get("radius_deg")
        break
    m = build(rng, rover, azimuth0_deg=az0, step=step, exclude_below_deg=float(below) + 0.5 if below else None)
    if not len(m["triangles"]):
        log.warning("rover surface: no rover surface in %s", npz.name)
        return None
    r0, r1 = m["rows"]
    tex = img[r0:r1]
    if texture_width and tex.shape[1] > int(texture_width):
        k = int(texture_width) / tex.shape[1]
        tex = cv2.resize(tex, (int(texture_width), max(1, int(round(tex.shape[0] * k)))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", tex, [cv2.IMWRITE_JPEG_QUALITY, TEXTURE_QUALITY])
    if not ok:
        raise RuntimeError("the texture could not be encoded")
    rec = {"experimental": True, "vertices": int(len(m["vertices"])), "triangles": int(len(m["triangles"])),
           "step_px": int(step), "edge_ratio": EDGE_RATIO, "panorama_rows": [r0, r1], "texture_px": [int(tex.shape[1]), int(tex.shape[0])],
           "azimuth0_deg": az0, "frame": "metres about the pivot: x east, y up, z south",
           "rover_pixels_share_meshed": float(len(m["triangles"]) / 2 * step * step / max(int(rover.sum()), 1)),
           "from": [npz.name, img_f.name]}
    glb = Path(out) if out else d / (stem + SUFFIX)
    glb.write_bytes(glb_bytes(m["vertices"], m["uv"], m["triangles"], enc.tobytes(), extras=rec))
    rec.update(file=glb.name, bytes=glb.stat().st_size, seconds=round(time.time() - t0, 1))
    glb.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log.info("rover surface: %s - %d triangles, texture %d x %d, %.1f MB (%.0f s)", glb.name, rec["triangles"],
             tex.shape[1], tex.shape[0], rec["bytes"] / 1e6, rec["seconds"])
    return rec


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m mppp.ibr.rover_surface", description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="the with_rover folder of a station run (holding *_ods_centre.png and *_ods_centre_depth.npz)")
    p.add_argument("--step", type=int, default=DEFAULT_STEP, help=f"panorama pixels between vertices (default {DEFAULT_STEP})")
    p.add_argument("--texture-width", type=int, default=None, metavar="PX", help="reduce the texture to this width")
    p.add_argument("--out", default=None, help="the .glb file (default <prefix>_rover_surface.glb in the folder)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rec = from_folder(a.folder, step=a.step, out=a.out, texture_width=a.texture_width)
    return 0 if rec else 1


if __name__ == "__main__":
    raise SystemExit(main())
