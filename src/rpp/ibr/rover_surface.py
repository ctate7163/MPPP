"""
The rover as a 3D model colored by the Navcam images, for the viewer.

0.98.14 (Christian, 15 Nov 2026: "The 3d model of the rover still has many occluded surfaces. The resolution of it
looks good, but the edges look pixelated. the rover should not have a hole visible from any where within the 0.15 m
radius"): the surface is the Perseverance model itself - the triangles ``rpp mesh --rover`` put on the terrain
(``mesh.ply``, the last ones: ``mesh_report.json`` "rover_triangles") - not a grid cut out of the panorama's depth.
So every surface of the rover is there from every viewpoint, and its edges are the model's own.

The color is the panorama from the pivot (``rpp ods --eyes left right center``: ``_ods_centre.png``), which holds
the Navcam frames projected onto that same model.  The viewer colors a point X of the model (meters about the pivot)
with the panorama's pixel in the direction of X, when X is the surface the pivot sees in that direction: its distance
|X| is within ``BIAS_M + BIAS_REL * |X|`` of the panorama's range there (``_ods_centre_depth.npz``).  A point hidden
from the pivot - behind another part, a far side, an underside - was seen by no camera of the station and is the
rover gray of the panorama (:data:`rpp.ibr.ods.ROVER_GRAY_REL`), shaded a little by its slope.  The test is made per
pixel of the display, on four neighboring depth values, so the border between colored and gray is smooth.

The file is a binary glTF (``.glb``): one mesh in meters about the pivot (x east, y up, z south: the viewer page's
frame; right-handed), an unlit gray material, and two images - the panorama's rows that hold the rover (JPEG) and
their range (PNG: red x 256 + green = range x ``DEPTH_UNITS_PER_M``; 0 = not the rover).  ``asset.extras.projection``
tells the viewer how the two are laid out.  (0.98.2-0.98.13: a grid mesh of the panorama's depth with a second layer
behind its edges; what the pivot did not see was missing, and the grid's edges were steps.)

``python -m rpp.ibr.rover_surface FOLDER`` (the ``with_rover`` folder of a station run).
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
DEPTH_STEP = 2                # the range image keeps every n-th pixel of the panorama (8192 wide: 1.5 mrad, 3 mm at 2 m)
DEPTH_UNITS_PER_M = 8000.0    # the range image: 16 bits, 0.125 mm steps, to 8.19 m
MAX_RANGE_M = 8.0             # the rover is nearer than this from the pivot
BIAS_M = 0.012                # a point is "seen from the pivot" when its distance is within BIAS_M + BIAS_REL x distance
BIAS_REL = 0.012              # of the range image's value (a surface seen at a grazing angle changes range by this much
#                               from one range pixel to the next; checked on the model: tests/test_ibr.py)
MARGIN_ROWS = 4               # panorama rows kept above and below the rover
TEXTURE_QUALITY = 90
GRAY_SHADE = 0.25             # the gray of unseen surfaces: 1 - this on a vertical face, 1 on a level one


def directions(cols: np.ndarray, rows: np.ndarray, width: int, height: int, azimuth0_deg: float = 0.0) -> np.ndarray:
    """Unit directions (..., 3) in the page's frame (x east, y up, z south) of panorama pixels (``cols``, ``rows``
    broadcast): the same rays as :func:`rpp.ibr.ods.ods_rays` from the pivot."""
    az = (np.asarray(cols, float) + 0.5) / width * 2 * np.pi + np.radians(float(azimuth0_deg))
    el = np.pi / 2 - (np.asarray(rows, float) + 0.5) / height * np.pi
    ce = np.cos(el)
    return np.stack(np.broadcast_arrays(ce * np.sin(az), np.sin(el), -ce * np.cos(az)), axis=-1)


def pixel_of(points: np.ndarray, width: int, height: int, azimuth0_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The inverse of :func:`directions`, as the viewer's shader does it: (column, row, distance) of points (n, 3) in
    the page's frame; a pixel's center is at (column + 0.5, row + 0.5)."""
    P = np.asarray(points, float)
    r = np.linalg.norm(P, axis=1)
    az = np.arctan2(P[:, 0], -P[:, 2]) - np.radians(float(azimuth0_deg))
    el = np.arcsin(np.clip(P[:, 1] / np.maximum(r, 1e-12), -1, 1))
    return (az / (2 * np.pi) % 1.0) * width, (0.5 - el / np.pi) * height, r


def model_in_page(folder: PathLike, station: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(vertices (n, 3) float32 in the page's frame about the pivot, triangles (m, 3) uint32) of the rover model in a
    ``with_rover`` folder's ``mesh.ply``; None when the mesh holds no model (``mesh_report.json`` "rover_triangles")."""
    from .mesh import read_mesh
    d = Path(folder)
    try:
        rt = json.loads((d / "mesh_report.json").read_text(encoding="utf-8")).get("rover_triangles")
    except (OSError, ValueError):
        return None
    if not rt or not (d / "mesh.ply").is_file():
        return None
    V, F = read_mesh(d / "mesh.ply")
    if not (0 <= int(rt[0]) < int(rt[1]) <= len(F)):
        return None
    F = np.asarray(F[int(rt[0]):int(rt[1])], np.int64)
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    X = np.asarray(V, float)[used] - np.asarray(station["pivot_block_m"], float)
    e, u, n = (np.asarray(station[k], float) for k in ("east_block", "up_block", "north_block"))
    return np.stack([X @ e, X @ u, -(X @ n)], axis=1).astype(np.float32), (np.cumsum(used) - 1)[F].astype(np.uint32)


def encode_range(rng: np.ndarray) -> np.ndarray:
    """(h, w) meters -> (h, w, 3) uint8 BGR for a PNG: red x 256 + green = range x DEPTH_UNITS_PER_M (0: none)."""
    q = np.clip(np.rint(np.nan_to_num(np.asarray(rng, np.float64), nan=0.0, posinf=0.0) * DEPTH_UNITS_PER_M), 0, 65535).astype(np.uint16)
    return np.stack([np.zeros(q.shape, np.uint8), (q & 255).astype(np.uint8), (q >> 8).astype(np.uint8)], axis=2)


def decode_range(bgr: np.ndarray) -> np.ndarray:
    return (bgr[..., 2].astype(np.float32) * 256.0 + bgr[..., 1]) / DEPTH_UNITS_PER_M


def seen_from_pivot(points: np.ndarray, depth: np.ndarray, proj: Dict[str, Any]) -> np.ndarray:
    """What the viewer's shader computes, here for tests and reports: for points (n, 3) in the page's frame, the share
    (0..1) of the four range pixels around their direction that say "this is the surface the pivot sees".  ``depth``:
    the range image (meters; :func:`decode_range`), ``proj``: the file's ``asset.extras.projection``."""
    c, r, dist = pixel_of(points, proj["width"], proj["height"], proj["azimuth0_deg"])
    h, w = depth.shape
    x = (c - 0.5 - proj["depth_offset_px"]) / proj["depth_step"]            # the range image's pixel index, continuous
    y = (r - 0.5 - proj["rows"][0] - proj["depth_offset_px"]) / proj["depth_step"]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    out = np.zeros(len(dist))
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            yy = y0 + dy
            ok = (yy >= 0) & (yy < h)
            z = np.where(ok, depth[np.clip(yy, 0, h - 1), (x0 + dx) % w], 0.0)
            out += wy * wx * ((z > 0) & (dist <= z * (1.0 + proj["bias_rel"]) + proj["bias_m"]))
    return out


def glb_bytes(vertices: np.ndarray, triangles: np.ndarray, jpeg: bytes, depth_png: bytes, gray_rgb,
              extras: Optional[Dict[str, Any]] = None) -> bytes:
    """A binary glTF 2.0 file: one mesh (positions, 32-bit indices) with an unlit, double-sided material of the color
    ``grey_rgb`` (0-255, sRGB) - what a general glTF viewer shows - and the two images of the projection (0: the
    panorama's rows as JPEG, 1: their range as PNG), which this module's viewer page uses (``extras``: asset.extras)."""
    parts, views = [], []

    def add(b: bytes, target: Optional[int] = None) -> int:
        off = sum(len(p) for p in parts)
        parts.append(b + b"\x00" * (-len(b) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(b)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    V = np.ascontiguousarray(vertices, "<f4")
    F = np.ascontiguousarray(triangles, "<u4").reshape(-1)
    bv_v, bv_f, bv_c, bv_d = add(V.tobytes(), 34962), add(F.tobytes(), 34963), add(bytes(jpeg)), add(bytes(depth_png))
    lin = [float(((c / 255.0 + 0.055) / 1.055) ** 2.4 if c / 255.0 > 0.04045 else c / 255.0 / 12.92) for c in gray_rgb]
    doc = {
        "asset": {"version": "2.0", "generator": "rpp.ibr.rover_surface"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "rover"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0, "mode": 4}]}],
        "accessors": [{"bufferView": bv_v, "componentType": 5126, "count": int(len(V)), "type": "VEC3",
                       "min": [float(x) for x in V.min(axis=0)], "max": [float(x) for x in V.max(axis=0)]},
                      {"bufferView": bv_f, "componentType": 5125, "count": int(len(F)), "type": "SCALAR"}],
        "materials": [{"name": "rover gray", "doubleSided": True, "extensions": {"KHR_materials_unlit": {}},
                       "pbrMetallicRoughness": {"baseColorFactor": lin + [1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
        "images": [{"bufferView": bv_c, "mimeType": "image/jpeg", "name": "panorama rows"},
                   {"bufferView": bv_d, "mimeType": "image/png", "name": "range"}],
        "bufferViews": views, "buffers": [{"byteLength": sum(len(p) for p in parts)}],
    }
    from .xmp import author_fields, software                    # 0.99.53: the author and the RPP version
    doc["asset"]["generator"] = f"{software()} (rpp.ibr.rover_surface)"
    doc["asset"]["extras"] = dict(extras or {}, **author_fields())
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


def gray_unseen(img: np.ndarray, unseen: np.ndarray, rover: Optional[np.ndarray] = None) -> int:
    """0.98.12 (Christian, 15 Nov 2026: "The 3d rover models should not interpolate colors across the rover especially at
    the bottom pole. Make in images mars rover gray"): the rover pixels no frame colored take, in place, one flat
    neutral gray (:data:`rpp.ibr.ods.ROVER_GRAY_REL` x the ground's mean level below the horizon), as the panorama
    paints them from 0.98.12 on - nothing is carried in from neighboring pixels.  (0.98.4-0.98.11 filled them with the
    nearest colored pixel, smoothed: colors smeared across the rover, worst at the bottom pole.)  Returns their count."""
    from .ods import ROVER_GRAY_REL
    u = np.asarray(unseen, bool)
    n = int(u.sum())
    if not n:
        return 0
    H = img.shape[0]
    low = np.zeros(u.shape, bool)
    low[H // 2:] = True
    don = low & ~u & (~np.asarray(rover, bool) if rover is not None else True)
    lev = float(np.median(img[don].astype(np.float32))) if don.sum() >= 50 else 110.0
    img[u] = np.clip(ROVER_GRAY_REL * lev, 0, 255)
    return n


def rover_gray(img: np.ndarray, unseen: Optional[np.ndarray], rover: np.ndarray) -> Tuple[int, int, int]:
    """The gray (R, G, B; 0-255) of rover surfaces no camera saw: the panorama's own where it painted some
    (:func:`gray_unseen`), else ROVER_GRAY_REL x the ground's median level below the horizon."""
    from .ods import ROVER_GRAY_REL
    if unseen is not None and np.asarray(unseen, bool).any():
        v = float(np.median(img[np.asarray(unseen, bool)].astype(np.float32)))
    else:
        low = np.zeros(rover.shape, bool)
        low[rover.shape[0] // 2:] = True
        don = low & ~np.asarray(rover, bool)
        v = ROVER_GRAY_REL * (float(np.median(img[don].astype(np.float32))) if don.sum() >= 50 else 110.0)
    g = int(np.clip(round(v), 0, 255))
    return g, g, g


def from_folder(folder: PathLike, out: Optional[PathLike] = None, texture_width: Optional[int] = None,
                depth_step: int = DEPTH_STEP) -> Optional[Dict[str, Any]]:
    """The rover of one ``with_rover`` folder: reads ``*_ods_centre_depth.npz`` (with its ``rover`` array),
    ``*_ods_centre.png``, the panorama's record ``*ods.json`` (the station's frame) and ``mesh.ply`` with
    ``mesh_report.json`` (the model's triangles); writes ``<prefix>_rover_surface.glb`` and ``.json`` beside them.
    None, with the reason logged, when one of them is missing."""
    import cv2
    t0 = time.time()
    d = Path(folder)
    npz = next(iter(sorted(d.glob("*ods_centre_depth.npz"))), None)
    if npz is None:
        log.warning("rover surface: no *_ods_centre_depth.npz in %s (rpp ods --eyes left right center)", d)
        return None
    stem = npz.name[:-len("ods_centre_depth.npz")]
    img_f = d / (stem + "ods_centre.png")
    with np.load(npz) as z:
        if "rover" not in z.files:
            log.warning("rover surface: %s has no rover array (render the center panorama with 0.98.2 or later)", npz.name)
            return None
        rng, rover = z["range"], z["rover"].astype(bool)
        unseen = z["unseen"].astype(bool) if "unseen" in z.files else None
    if not img_f.is_file():
        log.warning("rover surface: %s is missing", img_f.name)
        return None
    img = cv2.imread(str(img_f), cv2.IMREAD_COLOR)
    if img is None or img.shape[:2] != rng.shape:
        log.warning("rover surface: %s does not match the depth's size %s", img_f.name, rng.shape)
        return None
    H, W = rng.shape
    az0, station = 0.0, None
    for j in sorted(d.glob("*ods.json")):
        try:
            rec = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        az0 = float(rec.get("azimuth_of_column_0_deg") or rec.get("azimuth0_deg") or 0.0)
        station = rec.get("station")
        break
    if not station or not station.get("pivot_block_m"):
        log.warning("rover surface: no panorama record (*ods.json) with the station's frame in %s", d)
        return None
    model = model_in_page(d, station)
    if model is None:
        log.warning("rover surface: %s has no mesh.ply with the rover model (mesh_report.json: rover_triangles)", d)
        return None
    V, F = model
    ok = rover & np.isfinite(rng) & (rng > 0) & (rng < MAX_RANGE_M)
    if not ok.any():
        log.warning("rover surface: the rover is not in %s", npz.name)
        return None
    gray_px = gray_unseen(img, unseen, rover) if unseen is not None else 0
    gray = rover_gray(img, unseen, rover)
    rr = np.flatnonzero(ok.any(axis=1))
    step = max(1, int(depth_step))
    r0 = max(0, (int(rr[0]) - MARGIN_ROWS) // step * step)
    r1 = min(H, -(-(int(rr[-1]) + 1 + MARGIN_ROWS) // step) * step)
    tex = img[r0:r1]
    if texture_width and tex.shape[1] > int(texture_width):
        k = int(texture_width) / tex.shape[1]
        tex = cv2.resize(tex, (int(texture_width), max(1, int(round(tex.shape[0] * k)))), interpolation=cv2.INTER_AREA)
    okc, enc = cv2.imencode(".jpg", tex, [cv2.IMWRITE_JPEG_QUALITY, TEXTURE_QUALITY])
    depth = np.where(ok, rng, 0.0)[r0:r1][step // 2::step, step // 2::step]
    okd, encd = cv2.imencode(".png", encode_range(depth), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not (okc and okd):
        raise RuntimeError("the textures could not be encoded")
    proj = {"width": int(W), "height": int(H), "rows": [int(r0), int(r1)], "azimuth0_deg": float(az0),
            "depth_step": step, "depth_px": [int(depth.shape[1]), int(depth.shape[0])], "depth_units_per_m": DEPTH_UNITS_PER_M,
            "depth_offset_px": step // 2, "bias_m": BIAS_M, "bias_rel": BIAS_REL, "grey_rgb": list(gray), "grey_shade": GRAY_SHADE,
            "note": "a point X (m about the pivot; x east, y up, z south) takes the color of image 0 at its direction - "
                    "column = fract((atan2(x, -z) - azimuth0) / 2 pi) x width, row = (0.5 - asin(y / |X|) / pi) x height, rows[0] "
                    "being the image's first row - when |X| <= range x (1 + bias_rel) + bias_m, the range (image 1: "
                    "(red x 256 + green) / depth_units_per_m meters, every depth_step-th pixel from depth_offset_px) "
                    "weighted over the four pixels around the direction; else grey_rgb"}
    # how much of the model the pivot sees: the centers of its triangles through the same test, weighted by area
    a, b, c = V[F[:, 0]].astype(float), V[F[:, 1]].astype(float), V[F[:, 2]].astype(float)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    seen = seen_from_pivot((a + b + c) / 3.0, depth.astype(np.float32), proj)
    rec = {"model": "the Perseverance model of mesh.ply (rpp.ibr.rover_model), colored from the pivot's panorama",
           "vertices": int(len(V)), "triangles": int(len(F)), "area_m2": round(float(area.sum()), 2),
           "area_seen_from_pivot": round(float((area * seen).sum() / max(area.sum(), 1e-9)), 3),
           "texture_px": [int(tex.shape[1]), int(tex.shape[0])], "projection": proj,
           "frame": "meters about the pivot: x east, y up, z south", "unseen_pixels_grey": int(gray_px),
           "rover_pixels": int(rover.sum()), "from": [npz.name, img_f.name, "mesh.ply"]}
    glb = Path(out) if out else d / (stem + SUFFIX)
    glb.write_bytes(glb_bytes(V, F, enc.tobytes(), encd.tobytes(), gray, extras={"projection": proj}))
    rec.update(file=glb.name, bytes=glb.stat().st_size, seconds=round(time.time() - t0, 1))
    glb.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log.info("rover surface: %s - the model's %d triangles (%.0f %% of its area seen from the pivot), texture %d x %d, "
             "range %d x %d, %.1f MB (%.0f s)", glb.name, rec["triangles"], 100 * rec["area_seen_from_pivot"], tex.shape[1],
             tex.shape[0], depth.shape[1], depth.shape[0], rec["bytes"] / 1e6, rec["seconds"])
    return rec


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m rpp.ibr.rover_surface", description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="the with_rover folder of a station run (*_ods_centre.png, *_ods_centre_depth.npz, mesh.ply)")
    p.add_argument("--depth-step", type=int, default=DEPTH_STEP, help=f"the range image keeps every n-th panorama pixel (default {DEPTH_STEP})")
    p.add_argument("--texture-width", type=int, default=None, metavar="PX", help="reduce the color image to this width")
    p.add_argument("--out", default=None, help="the .glb file (default <prefix>_rover_surface.glb in the folder)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return 0 if from_folder(a.folder, out=a.out, texture_width=a.texture_width, depth_step=a.depth_step) else 1


if __name__ == "__main__":
    raise SystemExit(main())
