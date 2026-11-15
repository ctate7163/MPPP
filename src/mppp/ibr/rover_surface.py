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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
log = logging.getLogger(__name__)

SUFFIX = "rover_surface.glb"
DEFAULT_STEP = 2          # panorama pixels between vertices.  4 until 0.98.3 (Christian, 14 Nov 2026: "it's too pixelated.
#                           Double the voxel resolution if there is headroom"); 8192 wide: 1.5 mrad, 1.5 mm at 1 m
MAX_TRIANGLES = 2_000_000  # ... the headroom: with more than this in both layers the step grows by one and it is built again
HEAD_RADIUS_M = 0.15      # the layer behind covers what a head this far from the pivot can see past an edge
BEHIND_RATIO = 1.15       # an edge between two parts of the rover: neighbouring vertices whose ranges differ by more
BEHIND_EDGE_RATIO = 1.35  # the layer behind is cut where its own ranges jump by more than this
BEHIND_MAX_DEG = 12.0     # ... and reaches at most this far under the nearer part
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


def _grid_triangles(ok: np.ndarray, r: np.ndarray, ratio: float, need: Optional[np.ndarray] = None) -> np.ndarray:
    """Triangles (m, 3) of grid-vertex indices over an (nr, nc) grid that wraps in its columns.  A triangle is made
    when its three corners are ``ok`` and their ranges agree within ``ratio``; each cell is cut along the diagonal
    that keeps more of it (so a cell with one corner across an edge still gives one triangle).  ``need``: only triangles
    with a corner where it is True."""
    nr, nc = ok.shape
    nxt = np.roll(np.arange(nc), -1)
    idx = np.arange(nr * nc).reshape(nr, nc)
    a, b, c, d = idx[:-1], idx[:-1][:, nxt], idx[1:], idx[1:][:, nxt]
    okf, rf = ok.reshape(-1), r.reshape(-1)

    def good(i, j, k):
        q = np.stack([rf[i], rf[j], rf[k]])
        with np.errstate(invalid="ignore", divide="ignore"):
            return okf[i] & okf[j] & okf[k] & (q.max(axis=0) <= ratio * q.min(axis=0))

    t1, t2 = good(a, c, b), good(b, c, d)          # cut b-c
    t3, t4 = good(a, c, d), good(a, d, b)          # cut a-d
    first = (t1.astype(np.int8) + t2) >= (t3.astype(np.int8) + t4)
    if need is not None:
        nf = need.reshape(-1)
        t1, t2, t3, t4 = t1 & (nf[a] | nf[c] | nf[b]), t2 & (nf[b] | nf[c] | nf[d]), t3 & (nf[a] | nf[c] | nf[d]), t4 & (nf[a] | nf[d] | nf[b])
    out = []
    for m, (i, j, k) in ((t1 & first, (a, c, b)), (t2 & first, (b, c, d)), (t3 & ~first, (a, c, d)), (t4 & ~first, (a, d, b))):
        out.append(np.stack([i[m], j[m], k[m]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


MAX_COARSEN = 4           # towards the nadir the columns are thinned by 2, 4, ... up to 2 ** this


def _bands(rows: np.ndarray, H: int, nc: int) -> list:
    """An equirectangular grid has ever more columns per angle towards the nadir.  The grid rows are cut into bands
    [(i0, i1, m)]: rows i0..i1 (i1 is shared with the next band) use every m-th column, m = 2 ** k with
    2 ** k <= 1 / cos(elevation), so that a cell stays about as wide as it is tall."""
    el = np.radians(np.abs(90.0 - (rows + 0.5) / H * 180.0))
    with np.errstate(divide="ignore"):
        k = np.clip(np.floor(np.log2(1.0 / np.maximum(np.cos(el), 1e-6))), 0, MAX_COARSEN).astype(int)
    k = np.maximum.accumulate(k) if rows[0] >= H // 2 else k       # below the horizon it only grows
    while (nc % (2 ** int(k.max()))) and k.max() > 0:
        k = np.minimum(k, k.max() - 1)
    out, i0 = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or k[i] != k[i0]:
            out.append((i0, min(i, len(rows) - 1), int(2 ** k[i0])))
            i0 = i
    return out


def _banded(ok: np.ndarray, r: np.ndarray, pos: np.ndarray, ratio: float, bands: list, need: Optional[np.ndarray] = None) -> np.ndarray:
    """The triangles of every band, as indices of the full grid's vertices.  Where a band meets a coarser one, its
    last row's in-between vertices are moved (in ``pos``, in place) onto the line between their neighbours - the
    coarser band's edge - so the two bands meet without a crack."""
    nr, nc = ok.shape
    P = pos.reshape(nr, nc, 3)
    tris = []
    for b, (i0, i1, m) in enumerate(bands):
        if i1 <= i0:
            continue
        sub_ok, sub_r = ok[i0:i1 + 1, ::m], r[i0:i1 + 1, ::m]
        nd = None
        if need is not None:
            nd = need[i0:i1 + 1].reshape(i1 - i0 + 1, nc // m, m).any(axis=2)
        t = _grid_triangles(sub_ok, sub_r, ratio, nd)
        if len(t):
            w = nc // m
            tris.append((i0 + t // w) * nc + (t % w) * m)
        if b + 1 < len(bands) and bands[b + 1][2] > m:               # the next band is coarser: close the joint on row i1
            M = bands[b + 1][2]
            for c0 in range(m, M, m):                                # the columns between the coarser band's
                mid = np.arange(c0, nc, M)
                lo, hi = mid - c0, (mid - c0 + M) % nc
                f = c0 / M
                both = ok[i1, lo] & ok[i1, hi] & ok[i1, mid]
                with np.errstate(invalid="ignore", divide="ignore"):
                    both &= np.maximum(r[i1, lo], r[i1, hi]) <= ratio * np.minimum(r[i1, lo], r[i1, hi])
                P[i1, mid[both]] = (1 - f) * P[i1, lo[both]] + f * P[i1, hi[both]]
    return np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)


def _pack(tri: np.ndarray, pos: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Only the vertices the triangles use; the cells across the panorama's seam get their own copies of the low-u
    vertices at u + 1 (the texture repeats), so their texture does not run backwards across the picture."""
    used = np.zeros(len(pos), bool)
    used[tri.ravel()] = True
    new = np.cumsum(used) - 1
    V, UV, tri = pos[used].astype(np.float32), uv[used].astype(np.float32), new[tri].astype(np.uint32)
    if len(tri):
        u3 = UV[tri][:, :, 0]
        seam = (u3.max(axis=1) - u3.min(axis=1)) > 0.5
        if seam.any():
            st = tri[seam]
            low = UV[st][:, :, 0] < 0.5
            ids = np.unique(st[low])
            copy = np.full(len(V), -1, np.int64)
            copy[ids] = len(V) + np.arange(len(ids))
            V = np.concatenate([V, V[ids]])
            UV = np.concatenate([UV, UV[ids] + np.array([1.0, 0.0], np.float32)])
            st2 = st.copy()
            st2[low] = copy[st[low]].astype(np.uint32)
            tri[seam] = st2
    return V, UV, tri


def build(rng: np.ndarray, rover: np.ndarray, azimuth0_deg: float = 0.0, step: int = DEFAULT_STEP,
          edge_ratio: float = EDGE_RATIO, max_range_m: float = MAX_RANGE_M,
          exclude_below_deg: Optional[float] = None, head_radius_m: float = HEAD_RADIUS_M) -> Dict[str, Any]:
    """The grid mesh of the rover pixels.  ``rng`` (H, W) metres from the pivot, ``rover`` (H, W) bool.
    ``exclude_below_deg``: pixels within this angle of straight down are left out (the nadir mark is painted
    there).  Returns {"vertices" (n, 3) float32, "uv" (n, 2) float32 in the cropped texture, "triangles" (m, 3)
    uint32, "rows" (r0, r1) of the panorama the texture holds, "grid": (rows, cols) of the vertices' pixels,
    "behind": the second layer or None}; no vertices when the rover is not in the picture.

    0.98.4, the second layer (``head_radius_m`` > 0; Christian, 14 Nov 2026: "make sure that the rover does not have
    any occlusions no matter where the viewer moves the head within a .15 m radius"): from the pivot a nearer part
    hides what is behind it, and a head that moves sees past its edge.  Behind every such edge the farther surface is
    continued under the nearer part: a vertex of the nearer part gets the range of the nearest vertex on the far side
    of an edge, as far in as a head ``head_radius_m`` from the pivot can see past it (the angle
    head_radius_m x (1 / near - 1 / far)).  "behind": {"vertices", "uv" (in the grid: u = column / columns,
    v = row / rows), "triangles", "source" (nr, nc, 2) the grid vertex whose colour each vertex takes}."""
    rng = np.asarray(rng, np.float32)
    H, W = rng.shape
    step = max(1, int(step))
    ok = np.asarray(rover, bool) & np.isfinite(rng) & (rng > 0) & (rng < max_range_m)
    if exclude_below_deg:
        cut = int(np.floor(H * (1.0 - float(exclude_below_deg) / 180.0)))
        ok[cut:] = False
    empty = {"vertices": np.zeros((0, 3), np.float32), "uv": np.zeros((0, 2), np.float32),
             "triangles": np.zeros((0, 3), np.uint32), "rows": (0, 0), "behind": None, "grid": None}
    if not ok.any():
        return empty
    rr = np.flatnonzero(ok.any(axis=1))
    r0, r1 = int(rr[0]), int(rr[-1]) + 1
    rows = np.arange(r0, r1, step)
    cols = np.arange(0, W, step)                      # the last column joins the first again (360 degrees)
    g_ok = ok[np.ix_(rows, cols)]
    g_r = np.where(g_ok, rng[np.ix_(rows, cols)], np.nan).astype(np.float32)
    nr, nc = g_ok.shape
    bands = _bands(rows, H, nc)
    R, C = np.meshgrid(rows, cols, indexing="ij")
    d = directions(C, R, W, H, azimuth0_deg).reshape(-1, 3)
    pos = d * np.nan_to_num(g_r).reshape(-1, 1)
    tri = _banded(g_ok, g_r, pos, edge_ratio, bands)
    if not len(tri):
        return empty
    uv = np.stack([(C.reshape(-1) + 0.5) / W, (R.reshape(-1) + 0.5 - r0) / float(r1 - r0)], 1)
    V, UV, T = _pack(tri, pos, uv)
    out = {"vertices": V, "uv": UV, "triangles": T, "rows": (r0, r1), "grid": (rows, cols), "behind": None}
    if not head_radius_m or head_radius_m <= 0:
        return out
    # ---- the layer behind
    from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
    pad = min(nc, int(np.ceil(BEHIND_MAX_DEG / 360.0 * nc)) + 2)
    wrap = lambda x: np.concatenate([x[:, -pad:], x, x[:, :pad]], axis=1)
    rw, okw = wrap(np.nan_to_num(g_r, nan=0.0)), wrap(g_ok)
    lo = minimum_filter(np.where(okw, rw, np.inf), size=3, mode="nearest")
    hi = maximum_filter(np.where(okw, rw, 0.0), size=3, mode="nearest")
    seed = okw & (rw > BEHIND_RATIO * lo)                         # on the far side of an edge
    near = okw & (hi > BEHIND_RATIO * rw)
    if not seed.any() or not near.any():
        return out
    dist, (ir, ic) = distance_transform_edt(~seed, return_distances=True, return_indices=True)
    far = rw[ir, ic]
    el = np.radians(np.abs(90.0 - (rows + 0.5) / H * 180.0))
    px = (2 * np.pi / W) * step * np.maximum(np.cos(el), 0.25)[:, None]      # the smallest angle a grid step spans there
    with np.errstate(divide="ignore", invalid="ignore"):
        reach = np.minimum(head_radius_m * (1.0 / np.maximum(rw, 0.05) - 1.0 / np.maximum(far, 0.05)), np.radians(BEHIND_MAX_DEG)) / px
    G = okw & ~seed & (far > BEHIND_RATIO * rw) & (dist <= reach + 1.5)
    if not G.any():
        return out
    sl = slice(pad, pad + nc)
    Gc, far_c = G[:, sl], far[:, sl]
    src = np.stack([ir[:, sl], (ic[:, sl] - pad) % nc], axis=2)              # whose colour: the far vertex's
    b_r = np.where(Gc, far_c, g_r).astype(np.float32)
    b_ok = Gc | (g_ok & maximum_filter(wrap(Gc.astype(np.uint8)), size=3, mode="nearest")[:, sl].astype(bool))
    posb = d * np.nan_to_num(b_r).reshape(-1, 1)
    tb = _banded(b_ok, b_r, posb, BEHIND_EDGE_RATIO, bands, need=Gc)
    if len(tb):
        uvb = np.stack([(np.tile(np.arange(nc), nr) + 0.5) / nc, (np.repeat(np.arange(nr), nc) + 0.5) / nr], 1)
        Vb, UVb, Tb = _pack(tb, posb, uvb)
        own = np.stack(np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij"), axis=2)
        out["behind"] = {"vertices": Vb, "uv": UVb, "triangles": Tb, "source": np.where(Gc[..., None], src, own),
                         "filled_vertices": int(Gc.sum())}
    return out


def glb_bytes(vertices: np.ndarray, uv: np.ndarray, triangles: np.ndarray, jpeg: bytes, extras: Optional[Dict[str, Any]] = None,
              more: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, bytes]] = ()) -> bytes:
    """A binary glTF 2.0 file: one mesh whose first primitive has these positions, texture coordinates and an unlit,
    double-sided material coloured by the JPEG (the texture repeats in u, for the seam's vertices at u > 1).
    ``more``: further (vertices, uv, triangles, jpeg) layers, each a primitive with its own texture (0.98.4: the
    layer behind)."""
    parts, views, acc, prims, mats, texs, imgs = [], [], [], [], [], [], []

    def add(b: bytes, target: Optional[int] = None) -> int:
        off = sum(len(p) for p in parts)
        parts.append(b + b"\x00" * (-len(b) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(b)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    for n, (Vn, Tn, Fn, jn) in enumerate([(vertices, uv, triangles, jpeg)] + list(more)):
        V = np.ascontiguousarray(Vn, "<f4")
        T = np.ascontiguousarray(Tn, "<f4")
        F = np.ascontiguousarray(Fn, "<u4").reshape(-1)
        bv_v, bv_t, bv_f, bv_i = add(V.tobytes(), 34962), add(T.tobytes(), 34962), add(F.tobytes(), 34963), add(bytes(jn))
        a0 = len(acc)
        acc += [{"bufferView": bv_v, "componentType": 5126, "count": int(len(V)), "type": "VEC3",
                 "min": [float(x) for x in V.min(axis=0)], "max": [float(x) for x in V.max(axis=0)]},
                {"bufferView": bv_t, "componentType": 5126, "count": int(len(T)), "type": "VEC2"},
                {"bufferView": bv_f, "componentType": 5125, "count": int(len(F)), "type": "SCALAR"}]
        prims.append({"attributes": {"POSITION": a0, "TEXCOORD_0": a0 + 1}, "indices": a0 + 2, "material": n, "mode": 4})
        mats.append({"name": "front" if n == 0 else "behind", "doubleSided": True, "extensions": {"KHR_materials_unlit": {}},
                     "pbrMetallicRoughness": {"baseColorTexture": {"index": n}, "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        texs.append({"source": n, "sampler": 0})
        imgs.append({"bufferView": bv_i, "mimeType": "image/jpeg"})
    doc = {
        "asset": {"version": "2.0", "generator": "mppp.ibr.rover_surface"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "rover_surface"}],
        "meshes": [{"primitives": prims}], "accessors": acc, "materials": mats, "textures": texs,
        "samplers": [{"wrapS": 10497, "wrapT": 33071, "magFilter": 9729, "minFilter": 9987}],
        "images": imgs, "bufferViews": views, "buffers": [{"byteLength": sum(len(p) for p in parts)}],
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


def heal_unseen(img: np.ndarray, unseen: np.ndarray) -> int:
    """The rover pixels no frame coloured (the panorama paints them one flat brown) take, in place, the colour of the
    nearest coloured pixel, smoothed: a blind spot then reads as more of the surface around it.  Returns their count."""
    import cv2
    from scipy.ndimage import distance_transform_edt
    u = np.asarray(unseen, bool)
    n = int(u.sum())
    if not n or n == u.size:
        return 0
    rr = np.flatnonzero(u.any(axis=1))
    a, b = max(0, int(rr[0]) - 64), min(u.shape[0], int(rr[-1]) + 65)
    m = u[a:b]
    ir, ic = distance_transform_edt(m, return_distances=False, return_indices=True)
    near = img[a:b][ir, ic]
    soft = cv2.GaussianBlur(near, (0, 0), 6.0)
    img[a:b][m] = soft[m]
    return n


def from_folder(folder: PathLike, step: int = DEFAULT_STEP, out: Optional[PathLike] = None,
                texture_width: Optional[int] = None, max_triangles: Optional[int] = MAX_TRIANGLES,
                head_radius_m: float = HEAD_RADIUS_M) -> Optional[Dict[str, Any]]:
    """The rover surface of one ``with_rover`` folder: reads ``*_ods_centre_depth.npz`` (with its ``rover`` array,
    written from 0.98.2 on when the centre panorama is rendered) and ``*_ods_centre.png``, writes
    ``<prefix>_rover_surface.glb`` and ``<prefix>_rover_surface.json`` beside them.  None, with the reason logged,
    when the folder has no centre panorama or no rover flag.  ``max_triangles``: with more in the two layers the
    step grows by one and the surface is built again (None: no limit)."""
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
        unseen = z["unseen"].astype(bool) if "unseen" in z.files else None
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
        nm = rec.get("nadir_mark") or {}
        if nm.get("in_centre_panorama", True):                   # before 0.98.4 the mark is painted there: left out
            below = nm.get("radius_deg")
        break
    healed = heal_unseen(img, unseen) if unseen is not None else 0
    step = max(1, int(step))
    while True:
        m = build(rng, rover, azimuth0_deg=az0, step=step, exclude_below_deg=float(below) + 0.5 if below else None,
                  head_radius_m=head_radius_m)
        nb = len(m["behind"]["triangles"]) if m.get("behind") else 0
        if not max_triangles or len(m["triangles"]) + nb <= max_triangles or step >= 16:
            break
        log.info("rover surface: %d triangles at a step of %d px is over %d: one px coarser", len(m["triangles"]) + nb, step, max_triangles)
        step += 1
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
    more, brec = [], None
    if m.get("behind"):
        b = m["behind"]
        rows, cols = m["grid"]
        grid_img = img[np.ix_(rows, cols)]                       # the panorama at the grid's vertices
        tb = grid_img[b["source"][..., 0], b["source"][..., 1]]  # ... each filled vertex with its far neighbour's colour
        tb = cv2.GaussianBlur(np.ascontiguousarray(tb), (0, 0), 1.5)
        okb, encb = cv2.imencode(".jpg", tb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if okb:
            more.append((b["vertices"], b["uv"], b["triangles"], encb.tobytes()))
            brec = {"triangles": int(len(b["triangles"])), "vertices": int(len(b["vertices"])), "texture_px": [int(tb.shape[1]), int(tb.shape[0])],
                    "head_radius_m": float(head_radius_m), "edge_ratio": BEHIND_RATIO,
                    "note": "the farther surface continued under the nearer parts' edges, coloured from its own side of the edge: "
                            "what a head within head_radius_m of the pivot sees past them"}
    rec = {"experimental": True, "vertices": int(len(m["vertices"])), "triangles": int(len(m["triangles"])),
           "step_px": int(step), "edge_ratio": EDGE_RATIO, "panorama_rows": [r0, r1], "texture_px": [int(tex.shape[1]), int(tex.shape[0])],
           "azimuth0_deg": az0, "frame": "metres about the pivot: x east, y up, z south",
           "behind": brec, "unseen_pixels_healed": int(healed), "nadir_excluded_deg": (float(below) + 0.5) if below else None,
           "rover_pixels": int(rover.sum()), "from": [npz.name, img_f.name]}
    glb = Path(out) if out else d / (stem + SUFFIX)
    glb.write_bytes(glb_bytes(m["vertices"], m["uv"], m["triangles"], enc.tobytes(), extras=rec, more=more))
    rec.update(file=glb.name, bytes=glb.stat().st_size, seconds=round(time.time() - t0, 1))
    glb.with_suffix(".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log.info("rover surface: %s - %d triangles (+ %d behind) at %d px, texture %d x %d, %.1f MB (%.0f s)", glb.name, rec["triangles"],
             brec["triangles"] if brec else 0, step, tex.shape[1], tex.shape[0], rec["bytes"] / 1e6, rec["seconds"])
    return rec


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m mppp.ibr.rover_surface", description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="the with_rover folder of a station run (holding *_ods_centre.png and *_ods_centre_depth.npz)")
    p.add_argument("--step", type=int, default=DEFAULT_STEP, help=f"panorama pixels between vertices (default {DEFAULT_STEP})")
    p.add_argument("--max-triangles", type=int, default=MAX_TRIANGLES,
                   help=f"both layers together; over it the step grows by one (default {MAX_TRIANGLES}; 0: no limit)")
    p.add_argument("--head-radius", type=float, default=HEAD_RADIUS_M, metavar="M",
                   help=f"the layer behind covers a head this far from the pivot (default {HEAD_RADIUS_M}; 0: no layer behind)")
    p.add_argument("--texture-width", type=int, default=None, metavar="PX", help="reduce the texture to this width")
    p.add_argument("--out", default=None, help="the .glb file (default <prefix>_rover_surface.glb in the folder)")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rec = from_folder(a.folder, step=a.step, out=a.out, texture_width=a.texture_width,
                      max_triangles=a.max_triangles or None, head_radius_m=a.head_radius)
    return 0 if rec else 1


if __name__ == "__main__":
    raise SystemExit(main())
