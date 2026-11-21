"""
The JOSS paper's figures 1-4 (1.0, Christian, 20 Nov 2026), one entry per output file in the standard figure set
(``make_figures.bat`` -> :mod:`rpp_studies.paper.paper_figures` -> :func:`make_all`).  The specifications are the
project notes "For the v1p0 agent: JOSS figures 1-4" (20 Nov 2026) and the draft scripts beside them.

=================================================  ========================================================
output                                             inputs
=================================================  ========================================================
joss_figure1_pipeline_diagram.png / .svg           none
joss_figure2_navcam_undistorted_sol0009.png        the Butler Landing block (freeze 3)
joss_figure2b_navcam_undistorted_sol0052.png       the Van Zyl Overlook block
joss_figure3_block_butler_landing.png              Butler Landing block (freeze 3), its block cloud, the Sol0009 mesh
joss_figure3b_block_van_zyl_overlook.png           Van Zyl Overlook block, the S003D2046 cloud, the Sol0052 mesh
joss_figure3c_block_rockytop.png                   Rockytop block (freeze 3), tie points; panel D if a mesh exists
joss_figure4_station_sol0009.png                   the Sol0009 station (freeze 3)
joss_figure4b_station_sol0461.png                  the Sol0461 station (freeze 3, else the 8K test of 19 Nov)
joss_figure4c_station_sol0052.png                  the Sol0052 station - skipped until it has a full 3D model
=================================================  ========================================================

Every entry skips with one line saying why when an input is missing.  Each figure also writes ``<name>.json`` with
the numbers a caption needs (counts from the block, the files used).

Common rules: 400 dpi; panel titles 10 pt as "A. ..."; American spelling; colors the pipeline's own 8-bit conversion
(the processed 8-bit images; from 16-bit data: value / 64, no gamma); the camera symbols of :data:`CAMERA_STYLE`.
The oblique views and renders are ray cast (Open3D's Embree ray casting: every pixel sees the nearest triangle, as a
rasterizer would draw it); point clouds are drawn as points with a depth buffer.

    python -m rpp_studies.paper.joss_figures --out DIR [--root D:\\rover_blocks] [--freeze-root D:\\rover_blocks_f3]
                                             [--only figure3c ...]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, os.PathLike]
DPI = 400
TITLE_PT = 10
CAMERA_STYLE = {"N": ("o", "#2a78d6", "Navcam"), "Z": ("^", "#eb6834", "Mastcam-Z"),
                "F": ("s", "#1baf7a", "front Hazcam"), "R": ("s", "#eda100", "rear Hazcam")}
SUN = np.array([-0.45, -0.35, 0.82]) / np.linalg.norm([-0.45, -0.35, 0.82])
MODEL_DIR = Path("colmap") / "sparse" / "cahv_ba"


class Skip(Exception):
    """An input of a figure is not there: the entry is skipped with this message."""


# =========================================================================================== reading a block
# COLMAP camera model id -> number of parameters (SIMPLE_PINHOLE ... RAD_TAN_THIN_PRISM_FISHEYE)
_NPARAMS: Dict[int, int] = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 16}


def _nparams(model_id: int) -> int:
    if model_id not in _NPARAMS:
        raise ValueError(f"cameras.bin: camera model id {model_id} is not known")
    return _NPARAMS[model_id]


def read_cameras_bin(model: PathLike) -> Dict[int, Tuple[int, int, int, np.ndarray]]:
    """camera id -> (model id, width, height, params) of a COLMAP ``cameras.bin``."""
    raw = (Path(model) / "cameras.bin").read_bytes()
    n, = struct.unpack_from("<Q", raw, 0)
    o, out = 8, {}
    for _ in range(n):
        cid, mid, w, h = struct.unpack_from("<iiQQ", raw, o)
        o += 24
        k = _nparams(mid)
        out[cid] = (mid, int(w), int(h), np.array(struct.unpack_from("<" + "d" * k, raw, o)))
        o += 8 * k
    return out


def read_images_bin(model: PathLike, want: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """image name -> {camera_id, R (cam from world), t, C (center)} from ``images.bin`` record by record (a large
    block's file is not loaded whole); ``want``: only these names."""
    from scipy.spatial.transform import Rotation as Rot
    out = {}
    with open(Path(model) / "images.bin", "rb") as fh:
        n, = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            head = fh.read(64)
            qw, qx, qy, qz, tx, ty, tz = struct.unpack_from("<7d", head, 4)
            cid, = struct.unpack_from("<i", head, 60)
            nm = bytearray()
            while True:
                c = fh.read(1)
                if c in (b"\0", b""):
                    break
                nm += c
            k, = struct.unpack("<Q", fh.read(8))
            fh.seek(24 * k, 1)
            name = nm.decode("utf-8", "replace")
            if want is not None and name not in want:
                continue
            R = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            out[name] = {"camera_id": cid, "R": R, "t": t, "C": -R.T @ t}
    return out


def read_points3d_bin(model: PathLike) -> Dict[str, np.ndarray]:
    """xyz (n, 3), rgb (n, 3) uint8, error (n,), track length (n,) of ``points3D.bin``."""
    raw = (Path(model) / "points3D.bin").read_bytes()
    n, = struct.unpack_from("<Q", raw, 0)
    P = np.empty((n, 3)); C = np.empty((n, 3), np.uint8); E = np.empty(n); L = np.empty(n, np.int64)   # noqa: E702
    o = 8
    unpack = struct.Struct("<Q3d3BdQ").unpack_from
    for i in range(n):
        _, x, y, z, r, g, b, e, k = unpack(raw, o)
        P[i] = (x, y, z); C[i] = (r, g, b); E[i] = e; L[i] = k                                        # noqa: E702
        o += 51 + 8 * k
    return {"xyz": P, "rgb": C, "error": E, "track": L}


def read_cloud(ply: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """(xyz, rgb uint8) of a fused cloud (COLMAP's binary PLY)."""
    from rpp.sfm.dense import read_ply_vertices
    _, d = read_ply_vertices(ply)
    X = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    if "red" in d.dtype.names:
        RGB = np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.uint8)
    else:
        RGB = np.full((len(X), 3), 160, np.uint8)
    return X, RGB


def find_block(roots: Sequence[PathLike], pattern: str) -> Optional[Path]:
    """The first ``<root>/<pattern>`` (glob) with an aligned model, in the order of ``roots``."""
    for r in roots:
        if not r:
            continue
        for d in sorted(Path(r).glob(pattern)):
            if (d / MODEL_DIR / "images.bin").is_file() and (d / "colmap" / "project.json").is_file():
                return d
    return None


def _family(group: str) -> str:
    g = str(group or "?").upper()
    return g[:1]


def block_cameras(block: Path) -> Dict[str, Any]:
    """The registered images of a block: names, centers, camera groups (NL, ZL034 ...), stops."""
    rows = {r["name"]: r for r in json.loads((block / "colmap" / "project.json").read_text(encoding="utf-8"))["images"]}
    ims = read_images_bin(block / MODEL_DIR)
    names = sorted(n for n in ims if n in rows)
    from rpp.ibr.station import _group_of
    groups = [_group_of(n, rows[n]) for n in names]
    return {"names": names, "C": np.array([ims[n]["C"] for n in names]).reshape(-1, 3), "groups": groups,
            "family": np.array([_family(g) for g in groups]), "stop": np.array([str(rows[n].get("station") or "") for n in names]),
            "poses": ims, "rows": rows}


def health_numbers(block: Path) -> Dict[str, Any]:
    f = block / "colmap" / "health" / "health.json"
    if not f.is_file():
        return {}
    try:
        h = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: h.get(k) for k in ("status", "verdict", "images", "points", "stations") if k in h}


# =========================================================================================== drawing helpers
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def tone8(a: np.ndarray) -> np.ndarray:
    """The pipeline's 8-bit conversion of 16-bit data: value / 64, clipped, no gamma (``color.scale_int8_to_int16``)."""
    return np.clip(np.asarray(a, np.float32) / 64.0, 0, 255)


def fill_holes(img: np.ndarray, mask: np.ndarray, iterations: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Pixels next to covered ones get their neighbors' mean, ``iterations`` times (closes pin holes)."""
    from scipy import ndimage as ndi
    m = mask.copy()
    out = img.astype(np.float32).copy()
    for _ in range(iterations):
        w = ndi.uniform_filter(m.astype(np.float32), 3)
        if out.ndim == 3:
            new = np.dstack([ndi.uniform_filter(np.where(m, out[..., j], 0), 3) for j in range(out.shape[2])]) / np.maximum(w, 1e-6)[..., None]
        else:
            new = ndi.uniform_filter(np.where(m, out, 0), 3) / np.maximum(w, 1e-6)
        g = (~m) & (w > 0.12)
        out[g] = new[g]
        m = m | g
    return out, m


class View:
    """A pinhole view: ``eye``, unit forward / right / up, focal length ``f`` (px) on a ``W`` x ``H`` picture."""

    def __init__(self, eye, fwd, world_up, f: float, W: int, H: int):
        self.eye = np.asarray(eye, float)
        fw = np.asarray(fwd, float)
        self.fwd = fw / np.linalg.norm(fw)
        rt = np.cross(self.fwd, world_up)
        self.right = rt / np.linalg.norm(rt)
        self.up = np.cross(self.right, self.fwd)
        self.f, self.W, self.H = float(f), int(W), int(H)

    def scaled(self, k: int) -> "View":
        """The same view with ``k`` x the pixels (supersampling)."""
        v = View.__new__(View)
        v.eye, v.fwd, v.right, v.up = self.eye, self.fwd, self.right, self.up
        v.f, v.W, v.H = self.f * k, self.W * k, self.H * k
        return v

    def project(self, P: np.ndarray):
        q = np.asarray(P, float) - self.eye
        zc = q @ self.fwd
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.W / 2 + self.f * (q @ self.right) / zc, self.H / 2 - self.f * (q @ self.up) / zc, zc

    def rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """(rays (H, W, 6) for Open3D, the cosine of each ray to the forward axis (H, W))."""
        u = (np.arange(self.W) + 0.5 - self.W / 2) / self.f
        v = (np.arange(self.H) + 0.5 - self.H / 2) / self.f
        D = self.fwd[None, None, :] + u[None, :, None] * self.right[None, None, :] - v[:, None, None] * self.up[None, None, :]
        D /= np.linalg.norm(D, axis=2, keepdims=True)
        R = np.concatenate([np.broadcast_to(self.eye, D.shape), D], axis=2).astype(np.float32)
        return R, (D @ self.fwd).astype(np.float32)


def raycast(view: View, meshes: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Ray cast the triangle ``meshes`` [(V, F), ...] from ``view``: per pixel the depth along the forward axis (inf: no
    hit), the mesh index (-1), the triangle, its barycentric (u, v) and its normal turned toward the eye."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    ids = []
    for V, F in meshes:
        ids.append(sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32))))
    R, cosf = view.rays()
    ans = sc.cast_rays(o3d.core.Tensor(R))
    t = ans["t_hit"].numpy()
    gid = ans["geometry_ids"].numpy().astype(np.int64)
    hit = np.isfinite(t)
    which = np.full(t.shape, -1, np.int64)
    for k, g in enumerate(ids):
        which[hit & (gid == g)] = k
    N = ans["primitive_normals"].numpy()
    D = R[..., 3:]
    flip = (N * D).sum(2) > 0
    N[flip] *= -1
    return {"depth": np.where(hit, t * cosf, np.inf), "mesh": which, "triangle": ans["primitive_ids"].numpy().astype(np.int64),
            "uv": ans["primitive_uvs"].numpy(), "normal": N}


def splat(view: View, P: np.ndarray, RGB: np.ndarray, size: int = 1, fill: int = 3):
    """Points drawn nearest-last with a depth buffer: (picture uint8 (H, W, 3), depth (H, W), covered)."""
    W, H = view.W, view.H
    u, v, zc = view.project(P)
    ok = (zc > 1) & (u >= size) & (u < W - size - 1) & (v >= size) & (v < H - size - 1)
    u, v, zc, rg = u[ok].astype(int), v[ok].astype(int), zc[ok], RGB[ok]
    o = np.argsort(-zc)
    u, v, zc, rg = u[o], v[o], zc[o], rg[o]
    img = np.full((H, W, 3), 255, np.uint8)
    cov = np.zeros((H, W), bool)
    dep = np.full((H, W), np.inf, np.float32)
    for du in range(-size, size + 1):
        for dv in range(-size, size + 1):
            img[v + dv, u + du] = rg
            cov[v + dv, u + du] = True
            dep[v + dv, u + du] = zc
    f, m = fill_holes(img, cov, fill)
    f[~m] = 255
    return np.clip(f, 0, 255).astype(np.uint8), dep, cov


def gray_shade(N: np.ndarray, level: float = 0.72) -> np.ndarray:
    return (0.30 + 0.70 * np.clip(N @ SUN, 0, 1)) * level


# =========================================================================================== figure 1
def figure1_pipeline(out_png: Path, dpi: int = DPI) -> Dict[str, Any]:
    """The pipeline diagram, v5 of 20 Nov 2026: inputs by origin, the three RPP stages, RPP outputs, users."""
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12.9, 9.1))
    ax.set_xlim(0, 137)
    ax.set_ylim(3, 100)
    ax.axis("off")
    INK, EDGE = "#1f2933", "#52606d"
    GREY, SAND, BLUE, GREEN, WHITE = "#eef0f2", "#f6ead7", "#dbe9f6", "#e3f1e4", "#ffffff"

    def box(x, y, w, h, title, lines, fc, dashed=False, fs=7.8, z=3, lw=1.1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=EDGE, lw=lw,
                                    ls="--" if dashed else "-", zorder=z))
        top = y + h / 2 + (len(lines) * 2.7 + 3.2) / 2
        ax.text(x + w / 2, top - 1.6, title, ha="center", va="center", fontsize=9.4, fontweight="bold", color=INK, zorder=z + 1)
        for i, s in enumerate(lines):
            ax.text(x + w / 2, top - 5.1 - 2.7 * i, s, ha="center", va="center", fontsize=fs, color=INK, zorder=z + 1)

    def arrow(x0, y0, x1, y1, dashed=False, z=2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=EDGE, shrinkA=0,
                                     shrinkB=0, ls="--" if dashed else "-", zorder=z))

    def line(xs, ys, z=1):
        ax.plot(xs, ys, color=EDGE, lw=1.2, zorder=z, solid_capstyle="round")

    def head(x, y, s):
        ax.text(x, y, s, ha="left", va="center", fontsize=8.6, color=EDGE, style="italic", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.5))
    C1, C2, C3, W, LW = 42, 73, 104, 27, 34
    head(1, 97.5, "From outside")
    box(1, 84, LW, 11, "PDS archive", ["image products and labels,", "radiance calibrated for science"], GREY)
    box(1, 72, LW, 10, "Rover localization", ["waypoints and attitude", "for every stop"], GREY)
    box(1, 60, LW, 10, "Atmospheric conditions", ["local weather and solar", "brightness measurements"], GREY)
    head(1, 55, "Solved with RPP")
    box(1, 39, LW, 13, "Camera models", ["focal length, intrinsics and", "distortion varied with temperature,",
                                         "focus, zoom and sol trend"], SAND)
    box(1, 23.5, LW, 12, "Rig and mast models", ["stereo rig extrinsics, mast head,", "mast-to-rover transform"], SAND)
    box(1, 8, LW, 12, "Masking neural network", ["trained weights to remove", "rover hardware and shadows"], SAND)
    Y, H = 67, 27
    head(C1, 97.5, "Rover photogrammetry pipeline (RPP)")
    box(C1, Y, W, H, "1. Preprocessing", ["read PDS products", "radiometric and color conversion", "rover and shadow masks",
                                          "camera model for every image"], BLUE)
    box(C2, Y, W, H, "2. SfM and MVS", ["feature matching (COLMAP)", "bundle adjustment with pose", "priors and rigs (RPP, Ceres)",
                                        "alignment health report", "dense stereo and fusion (COLMAP)"], BLUE)
    box(C3, Y, W, H, "3. Image-based rendering", ["station terrain mesh", "depth and normal maps", "stereo 360 panoramas",
                                                  "full 3D terrain models"], BLUE)
    arrow(C1 + W + 0.4, Y + H / 2, C2 - 0.4, Y + H / 2)
    arrow(C2 + W + 0.4, Y + H / 2, C3 - 0.4, Y + H / 2)
    PY, PH = 46, 14
    head(C1, 63.3, "RPP outputs")
    box(C1, PY, W, PH, "Processed images", ["16-bit PNG with masks", "and metadata"], GREEN)
    box(C2, PY, W, PH, "Aligned blocks", ["COLMAP, Metashape and", "Nerfstudio projects;", "point clouds and meshes"], GREEN)
    box(C3, PY, W, PH, "3D assets", ["panoramas (PNG, JPEG),", "meshes (glTF), depth maps,", "JSON records"], GREEN)
    for x in (C1, C2, C3):
        arrow(x + W / 2, Y - 0.4, x + W / 2, PY + PH + 0.4)
    UY, UH = 29.5, 10.5
    head(C1, 42.4, "Who uses them")
    box(C2, UY, W - 7, UH, "3D GUIs", ["Blender, COLMAP,", "Metashape, Nerfstudio"], WHITE)
    box(C3, UY, W - 7, UH, "RPV", ["Rover Photogrammetry", "Viewer: browser or VR"], WHITE)
    arrow(C2 + (W - 7) / 2, PY - 0.4, C2 + (W - 7) / 2, UY + UH + 0.4)
    arrow(C3 + (W - 7) / 2, PY - 0.4, C3 + (W - 7) / 2, UY + UH + 0.4)
    box(C1, 8, C3 + W - C1, 12, "Your application", ["The outputs are plain files in documented formats: any viewer, "
                                                     "analysis tool or game engine can read them.", "RPV is one example."],
        WHITE, dashed=True, lw=2.2)
    for x in (C1 + W / 2, C2 + W - 3.5, C3 + W - 3.5):
        arrow(x, PY - 0.4, x, 20.4, dashed=True)
    E = 1 + LW + 0.4
    arrow(E, 89.5, C1 - 0.4, 89.5)
    arrow(E, 77, C1 - 0.4, 77)
    line([E, 37.5], [65, 65])
    line([37.5, 37.5], [65, 73])
    arrow(37.5, 73, C1 - 0.4, 73)
    for y in (45.5, 29.5, 14):
        line([E, 39.5], [y, y])
    line([39.5, 39.5], [14, 69.5])
    arrow(39.5, 69.5, C1 - 0.4, 69.5)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"files": [out_png.name, out_png.with_suffix(".svg").name]}


# =========================================================================================== figure 2
FIGURE2 = {
    "sol0009": {"block": "mars2020_sol_0001_butler_landing*", "freeze": True, "scale": 2,
                "names": ("NLF_0009_0667755636_926RAD_N0030000NCAM05000_0M00LLJ03",
                          "NRF_0009_0667755636_926RAD_N0030000NCAM05000_0M00LLJ06")},
    "sol0052": {"block": "mars2020_sol_0015_van_zyl_overlook*", "freeze": False, "scale": 1,
                "names": ("NLF_0052_0671568531_150RAD_N0032046NCAM02052_0A0295J01",
                          "NRF_0052_0671568531_150RAD_N0032046NCAM02052_0A0295J01")},
}
FIG2_FOV_DEG = 95.0


def _processed(block: Path, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(the processed picture as 0-1 floats, its mask: True = kept) - the 8-bit file the alignment used, else 16-bit / 64."""
    import cv2
    c = block / "colmap"
    a8 = c / "images" / f"{name}.png"
    if a8.is_file():
        img = cv2.imread(str(a8), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    else:
        a16 = next((p for p in (c / "images16" / f"{name}.png", block / "processed" / f"{name}.png") if p.is_file()), None)
        if a16 is None:
            raise Skip(f"no processed image {name}.png in {c / 'images'}")
        img = tone8(cv2.imread(str(a16), cv2.IMREAD_UNCHANGED)[..., ::-1]) / 255.0
    mf = c / "masks" / f"{name}.png.png"
    m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) > 0 if mf.is_file() else None
    if m is not None and m.shape != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return img, m


def figure2_navcam(block: Path, names: Sequence[str], out_png: Path, scale: int = 1, dpi: int = DPI) -> Dict[str, Any]:
    """A Navcam pair as processed (masked area half grayed, no-data corners black) and undistorted to one 95-degree
    pinhole camera (masked area and no-data white), through the block's refined camera of each eye."""
    import cv2
    import pycolmap
    plt = _plt()
    cams = read_cameras_bin(block / MODEL_DIR)
    ims = read_images_bin(block / MODEL_DIR, {f"{n}.png" for n in names})
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    lab = "ABCD"
    info = {"block": str(block), "images": list(names), "fov_deg": FIG2_FOV_DEG, "mask": "colmap/masks (the alignment's)"}
    for j, name in enumerate(names):
        if f"{name}.png" not in ims:
            raise Skip(f"{name} is not in the model {block / MODEL_DIR}")
        img, m = _processed(block, name)
        H, W = img.shape[:2]
        mid, w0, h0, p = cams[ims[f"{name}.png"]["camera_id"]]
        p = p.copy()
        p[:4] *= W / float(w0)                                   # a reduced-resolution product
        cam = pycolmap.Camera(model=pycolmap.CameraModelId(mid), width=W, height=H, params=p)
        keep = m if m is not None else np.ones((H, W), bool)
        nodata = img.sum(2) == 0
        S = int(scale)
        t = cv2.resize(img, None, fx=1 / S, fy=1 / S, interpolation=cv2.INTER_AREA) if S > 1 else img.copy()
        ks = cv2.resize(keep.astype(np.uint8), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        nd = cv2.resize(nodata.astype(np.uint8), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        t[~ks] = 0.5 * t[~ks] + 0.5 * 0.62
        t[nd] = 0.0
        eye = "left" if name[1] == "L" else "right"
        ax[0, j].imshow(np.clip(t, 0, 1))
        ax[0, j].set_title(f"{lab[j]}. Navcam {eye}, processed; mask in gray", fontsize=TITLE_PT, loc="left")
        fo = (W / 2) / np.tan(np.radians(FIG2_FOV_DEG / 2)) / S
        Wo, Ho = W // S, H // S
        xs, ys = np.meshgrid((np.arange(Wo) - Wo / 2 + 0.5) / fo, (np.arange(Ho) - Ho / 2 + 0.5) / fo)
        uv = cam.img_from_cam(np.c_[xs.ravel(), ys.ravel(), np.ones(xs.size)])
        mx = (uv[:, 0] - 0.5).reshape(Ho, Wo).astype(np.float32)
        my = (uv[:, 1] - 0.5).reshape(Ho, Wo).astype(np.float32)
        u = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0)
        mo = cv2.remap(keep.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0) > 0.5
        inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1) & np.isfinite(mx)
        u[~mo | ~inside] = 1.0
        ax[1, j].imshow(np.clip(u, 0, 1))
        ax[1, j].set_title(f"{lab[2 + j]}. Navcam {eye}, undistorted; mask blanked", fontsize=TITLE_PT, loc="left")
        info.setdefault("cameras", []).append({"image": name, "camera_id": ims[f"{name}.png"]["camera_id"],
                                               "model": pycolmap.CameraModelId(mid).name, "size": [W, H],
                                               "mask_kept_pct": round(100 * float(keep.mean()), 1)})
    for a_ in ax.ravel():
        a_.set_xticks([])
        a_.set_yticks([])
    fig.subplots_adjust(wspace=0.04, hspace=0.12)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return info


# =========================================================================================== figure 3
FIGURE3 = {
    "butler_landing": dict(block="mars2020_sol_0001_butler_landing*", freeze=True, cloud="colmap/dense_block_FN/fused.ply",
                           mesh=("single_sites/Sol0009", "single_sites/Sol0009_S003D0000"), clean=False, half=10.0, res=0.02, minc=4, fill=3,
                           spike=0.04, exag=3.0, c0=(0, 0.0), eye=(0.0, -17.0, 4.6), tgt=(0.0, 2.0, -1.6), focal=2450.0,
                           bar=4, what="Dense point cloud", splat=1, exclude=("S003D0028",)),
    "van_zyl_overlook": dict(block="mars2020_sol_0015_van_zyl_overlook*", freeze=False, cloud="colmap/dense_S003D2046/fused.ply",
                             mesh=("single_sites/Sol0052", "single_sites/Sol0052_S003D2046"), clean=True, half=15.0, res=0.03, minc=3, fill=3,
                             spike=0.08, exag=2.0, c0=(0, 0.0), center_station=("single_sites/Sol0052", "single_sites/Sol0052_S003D2046"),
                             center=(29.925, 6.905), cameras_within_m=16.0, cloud_within_m=60.0, eye=(0.0, -24.0, 7.5),
                             tgt=(0.0, 4.0, -1.0), focal=3000.0, bar=5, what="Dense point cloud", splat=1, exclude=(), ztop=1.0),
    "rockytop": dict(block="mars2020_sol_0460_rockytop*", freeze=True, cloud=None, mesh=("single_sites/Sol0461", "single_sites/Sol0461_S026D0630"),
                     clean=True, half=24.0, res=0.06, minc=2, fill=5, spike=0.15, exag=1.5, c0=(0, 2.0), eye=(0.0, -60.0, 22.0),
                     tgt=(0.0, 6.0, 0.8), focal=4000.0, bar=10, what="Tie points", splat=2, exclude=()),
}
PITCH_UP_DEG = 3.0
RMAX_M = 50.0
FIG3_W, FIG3_H = 4000, 1430
ROVER_MIN_NAVCAM = 6           # a stop gets a rover model when it has this many Navcam frames (a panorama, not a pair)
_REF_RNAV = {"P": (0.820, 0.553, -1.978), "FL": (1.088, -0.043, -0.769), "FR": (1.075, 0.205, -0.776),
             "RL": (-1.051, 0.438, -0.772), "RR": (-1.060, -0.496, -0.765)}   # Navcam pivot, Hazcam centers (RNAV m)


class _Im:                                                       # what rover_deck.rover_pose reads of a model image
    def __init__(self, name, R, C):
        self.name, self._R, self._C, self.has_pose = name, R, C, True

    def cam_from_world(self):
        R = self._R

        class _T:
            class rotation:                                      # noqa: N801
                @staticmethod
                def matrix():
                    return R
        return _T()

    def projection_center(self):
        return self._C


def rover_poses(block: Path, cams: Dict[str, Any], exclude: Sequence[str] = ()) -> List[Tuple[str, Dict[str, Any]]]:
    """(stop, the rover's pose in the block) of every stop with ``ROVER_MIN_NAVCAM`` Navcam frames: from the frames'
    label camera models and solved poses (:func:`rpp.ibr.rover_deck.rover_pose`), else a rigid fit of the Navcam pivot
    and the Hazcam centers in RNAV to the aligned centers."""
    from rpp.ibr.rover_deck import rover_pose
    from rpp.sfm.project import SfmProject
    proj = None
    try:
        proj = SfmProject.load(block / "colmap")
    except Exception:                                            # noqa: BLE001
        pass
    out = []
    fam2 = np.array([g[:2] for g in cams["groups"]])
    for stop in sorted(set(cams["stop"])):
        if not stop or any(x in stop for x in exclude):
            continue
        q = cams["stop"] == stop
        if (q & (cams["family"] == "N")).sum() < ROVER_MIN_NAVCAM:
            continue
        names = [cams["names"][i] for i in np.flatnonzero(q)]
        pose = None
        if proj is not None:
            try:
                rec = type("R", (), {"images": {i: _Im(n, cams["poses"][n]["R"], cams["poses"][n]["C"]) for i, n in enumerate(names)}})
                pose = rover_pose(proj, rec, names)
                if not pose.get("n"):
                    pose = None
            except Exception:                                    # noqa: BLE001 - the rigid fit below
                pose = None
        if pose is None:
            A, B = [], []
            qn = q & np.isin(fam2, ["NL", "NR"])
            A.append(_REF_RNAV["P"])
            B.append(cams["C"][qn].mean(0))
            for k in ("FL", "FR", "RL", "RR"):
                qk = q & (fam2 == k)
                if qk.any():
                    A.append(_REF_RNAV[k])
                    B.append(np.median(cams["C"][qk], 0))
            if len(A) < 3:
                continue
            A, B = np.array(A, float), np.array(B, float)
            a0, b0 = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((B - b0).T @ (A - a0))
            Rr = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            pose = {"R_block_from_rnav": Rr, "t_block_from_rnav": b0 - Rr @ a0, "n": len(A), "method": "rigid fit"}
        out.append((stop, pose))
    return out


def _tinted_relief(Hs, mh, zmin, zmax, dx, exag):
    """The maps figures' tinted hillshade (mapplot3.draw_terrain): gist_earth, light from 315 / 40."""
    plt = _plt()
    from matplotlib.colors import LightSource
    hs = LightSource(azdeg=315, altdeg=40).hillshade(Hs, vert_exag=exag, dx=dx, dy=dx)
    rgb = plt.get_cmap("gist_earth")(plt.Normalize(zmin, zmax)(Hs))[..., :3]
    out = np.clip((0.55 * rgb + 0.45 * 0.86) * (0.35 + 0.65 * hs)[..., None], 0, 1)
    out[~mh] = 1
    return out


def _station_mesh(folder: Path, c0: np.ndarray, rmax: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """A station's terrain mesh (``mesh.ply`` without the rover model's triangles) within ``rmax`` of ``c0``."""
    from rpp.ibr.mesh import read_mesh
    f = folder / "mesh.ply"
    if not f.is_file():
        return None
    V, F = read_mesh(f)
    try:
        rep = json.loads((folder / "mesh_report.json").read_text(encoding="utf-8"))
        rt = rep.get("rover_triangles")
        if rt and int(rt[1]) > int(rt[0]):
            keep = np.ones(len(F), bool)
            keep[int(rt[0]):int(rt[1])] = False
            F = F[keep]
    except (OSError, ValueError, TypeError):
        pass
    T = V[F]
    F = F[(np.hypot(T[..., 0] - c0[0], T[..., 1] - c0[1]) <= rmax).all(1)]
    return V, F


def figure3_block(key: str, roots: Dict[str, Optional[Path]], out_png: Path, dpi: int = DPI) -> Dict[str, Any]:
    """The aligned block: A the cloud from above, B shaded relief, C an oblique view looking north with rover models
    (arm stowed) and the camera centers, D the terrain mesh from the same view when the block has one."""
    from scipy import ndimage as ndi
    from rpp.ibr.rover_deck import mech_to_block
    from rpp.ibr import rover_model
    plt = _plt()
    cfg = FIGURE3[key]
    order = [roots.get("freeze"), roots.get("root")] if cfg["freeze"] else [roots.get("root"), roots.get("freeze")]
    block = find_block(order, cfg["block"])
    if block is None:
        raise Skip(f"no aligned block {cfg['block']} under {', '.join(str(r) for r in order if r)}")
    base = block.parent
    cams = block_cameras(block)
    C, fam = cams["C"], cams["family"]
    center = None
    if cfg.get("center_station"):
        sj = next((base / c / "station.json" for c in cfg["center_station"] if (base / c / "station.json").is_file()),
                  base / cfg["center_station"][0] / "station.json")
        try:
            center = np.array(json.loads(sj.read_text(encoding="utf-8"))["pivot_block_m"][:2], float)
        except (OSError, ValueError, KeyError):
            center = np.array(cfg["center"], float)
    if cfg.get("cameras_within_m") and center is not None:
        keepc = np.hypot(C[:, 0] - center[0], C[:, 1] - center[1]) < cfg["cameras_within_m"]
        for k in ("C", "family", "stop"):
            cams[k] = cams[k][keepc]
        cams["names"] = [n for n, k_ in zip(cams["names"], keepc) if k_]
        cams["groups"] = [g for g, k_ in zip(cams["groups"], keepc) if k_]
        C, fam = cams["C"], cams["family"]
    if not len(C):
        raise Skip(f"{block.name}: no registered cameras")
    c0 = np.r_[(center if center is not None else np.median(C[:, :2], 0)) + np.array(cfg["c0"]), np.median(C[:, 2])]
    info: Dict[str, Any] = {"block": str(block), "images": int(len(C)), "stops": int(len(set(cams["stop"]))),
                            "health": health_numbers(block)}
    if cfg["cloud"]:
        ply = block / cfg["cloud"]
        if not ply.is_file():
            raise Skip(f"no block cloud {ply}")
        X, RGB = read_cloud(ply)
        err = np.zeros(len(X))
        info["cloud"] = {"file": str(ply), "points": int(len(X))}
        if cfg.get("cloud_within_m"):
            k = np.hypot(X[:, 0] - c0[0], X[:, 1] - c0[1]) < cfg["cloud_within_m"]
            X, RGB, err = X[k], RGB[k], err[k]
    else:
        tp = read_points3d_bin(block / MODEL_DIR)
        X, RGB, err = tp["xyz"], tp["rgb"], tp["error"]
        info["tie_points"] = {"points": int(len(X)), "median_error_px": round(float(np.median(err)), 3),
                              "zero_color": int((RGB.sum(1) == 0).sum())}
    if cfg["clean"]:                                             # quality, range, isolated points (for display)
        k = (err < 1.5) & (np.hypot(X[:, 0] - c0[0], X[:, 1] - c0[1]) < 70)
        X, RGB = X[k], RGB[k]
        Vx = np.floor((X - X.min(0)) / 0.3).astype(np.int64) + 1
        cnt = np.zeros(tuple(Vx.max(0) + 2), np.uint16)
        np.add.at(cnt, (Vx[:, 0], Vx[:, 1], Vx[:, 2]), 1)
        nb = ndi.uniform_filter(cnt.astype(np.float32), 3) * 27
        k = nb[Vx[:, 0], Vx[:, 1], Vx[:, 2]] >= 5.5
        X, RGB = X[k], RGB[k]
    zref = np.median(C[fam == "N", 2]) - 2.0 if (fam == "N").any() else np.median(C[:, 2]) - 2.0
    if cfg.get("ztop") is not None:
        k = X[:, 2] < zref + cfg["ztop"]
        X, RGB = X[k], RGB[k]
    info["shown_points"] = int(len(X))
    HALF, RES = cfg["half"], cfg["res"]
    n = int(round(2 * HALF / RES)) // 2 * 2                      # even, for the 2 x 2 height cells
    ix = ((X[:, 0] - c0[0] + HALF) / RES).astype(int)
    iy = ((X[:, 1] - c0[1] + HALF) / RES).astype(int)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    if not cfg["clean"]:
        ok &= X[:, 2] < c0[2] + 0.3
    ix, iy, z, rgb = ix[ok], iy[ok], X[ok, 2], RGB[ok]
    o = np.argsort(z)
    ix, iy, z, rgb = ix[o], iy[o], z[o], rgb[o]
    hc = rgb.sum(1) > 0                                          # zero-color points (the tie-point color bug) left out
    lin0 = (iy * n + ix)[hc]
    cc = np.bincount(lin0, minlength=n * n)
    ortho = np.dstack([np.bincount(lin0, weights=rgb[hc, j], minlength=n * n) / np.maximum(cc, 1) for j in range(3)]).reshape(n, n, 3)
    of, m2 = fill_holes(ortho, (cc > 0).reshape(n, n), 3)
    of = np.dstack([ndi.median_filter(of[..., j], 3) for j in range(3)])
    of[~m2] = 255
    n2 = n // 2
    lin = (iy // 2) * n2 + (ix // 2)
    o2 = np.lexsort((z, lin))
    l2, z2 = lin[o2], z[o2]
    st = np.r_[0, np.flatnonzero(np.diff(l2)) + 1] if len(l2) else np.zeros(0, int)
    en = np.r_[st[1:], len(l2)] if len(l2) else np.zeros(0, int)
    Hm = np.zeros(n2 * n2, np.float32)
    ct = np.zeros(n2 * n2, int)
    Hm[l2[st]] = z2[(st + en - 1) // 2]
    ct[l2[st]] = en - st
    Hf, mh = fill_holes(Hm.reshape(n2, n2), (ct >= cfg["minc"]).reshape(n2, n2), cfg["fill"])
    med = ndi.median_filter(Hf, size=5)
    Hf = np.where(np.abs(Hf - med) > cfg["spike"], med, Hf)
    Hs = ndi.gaussian_filter(np.where(mh, Hf, 0), 1.2) / np.maximum(ndi.gaussian_filter(mh.astype(np.float32), 1.2), 1e-6) - zref
    zmin, zmax = np.percentile(Hs[mh], [2, 98]) if mh.any() else (0, 1)
    shade = _tinted_relief(Hs, mh, zmin, zmax, RES * 2, cfg["exag"])
    # ---- the oblique view: pitched so the middle of its top edge meets the ground at RMAX_M, then up PITCH_UP_DEG
    W, H, f = FIG3_W, FIG3_H, cfg["focal"]
    eye = c0 + np.array(cfg["eye"])

    def view(tz):
        tg = c0 + np.array([cfg["tgt"][0], cfg["tgt"][1], tz])
        return View(eye, tg - eye, (0, 0, 1), f, W, H)

    def top_inside(v):
        ray = v.fwd + v.up * (H / 2) / f
        if ray[2] >= -1e-6:
            return False
        hit = eye + ray * ((zref - eye[2]) / ray[2])
        return np.hypot(hit[0] - c0[0], hit[1] - c0[1]) <= RMAX_M - 1.0
    tz = cfg["tgt"][2]
    while not top_inside(view(tz)) and tz > -200:
        tz -= 0.25
    p0 = -np.degrees(np.arcsin(view(tz).fwd[2]))
    while -np.degrees(np.arcsin(view(tz).fwd[2])) > p0 - PITCH_UP_DEG:
        tz += 0.05
    V_ = view(tz)
    info["pitch_deg"] = round(float(-np.degrees(np.arcsin(V_.fwd[2]))), 1)
    poses = rover_poses(block, cams, cfg["exclude"]) if rover_model.available() else []
    rovers = []
    if poses:
        RV, RF, _ = rover_model.load()
        for stop, pose in poses:
            rovers.append((mech_to_block(RV, pose), RF))
    info["rovers"] = [s for s, _ in poses]
    k3 = (RGB.sum(1) > 0) & (np.hypot(X[:, 0] - c0[0], X[:, 1] - c0[1]) <= RMAX_M)

    def with_rovers(img, dep):
        if not rovers:
            return img
        r = raycast(V_, rovers)
        show = (r["mesh"] >= 0) & (r["depth"] < dep + 0.5)
        g = np.clip(gray_shade(r["normal"]) * 255, 0, 255)
        out = img.copy()
        out[show] = np.dstack([g, g, np.clip(g * 1.01, 0, 255)]).astype(np.uint8)[show]
        return out
    tv, td, _ = splat(V_, X[k3], RGB[k3], size=cfg["splat"], fill=4)
    panel_c = with_rovers(tv, td)
    panel_d = None
    # the station's folder by its 1.0 name (Sol0009), else by its 0.99 name (Sol0009_S003D0000)
    mesh_dir = next((base / m for m in cfg.get("mesh") or () if (base / m / "mesh.ply").is_file()),
                    base / cfg["mesh"][0] if cfg.get("mesh") else None)
    if mesh_dir is not None:
        mesh = _station_mesh(mesh_dir, c0, RMAX_M)
        if mesh is not None and len(mesh[1]):
            r = raycast(V_, [mesh] + rovers)
            img = np.full((H, W, 3), 255, np.uint8)
            terr = r["mesh"] == 0
            sh = 0.30 + 0.70 * np.clip(r["normal"] @ SUN, 0, 1)
            img[terr] = np.clip(np.outer(sh[terr], [176, 150, 132.0]) * 1.15, 0, 255).astype(np.uint8)
            rov = r["mesh"] > 0
            g = np.clip(gray_shade(r["normal"][rov]) * 255, 0, 255)
            img[rov] = np.stack([g, g, np.clip(g * 1.01, 0, 255)], 1).astype(np.uint8)
            panel_d = img
            info["mesh"] = {"file": str(mesh_dir / "mesh.ply"), "triangles_within_50m": int(len(mesh[1]))}
        else:
            info["mesh"] = f"no mesh at {mesh_dir} - panel D left out"
    # ---- the figure
    fig = plt.figure(figsize=(11, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.78], hspace=0.12, wspace=0.08)
    ext = [-HALF, HALF, -HALF, HALF]
    a = fig.add_subplot(gs[0, 0])
    a.imshow(np.clip(of, 0, 255).astype(np.uint8), origin="lower", extent=ext)
    b = fig.add_subplot(gs[0, 1])
    b.imshow(shade, origin="lower", extent=ext)
    for ax_ in (a, b):
        for kk, (mk, col, lab) in CAMERA_STYLE.items():
            q = fam == kk
            if q.any():
                ax_.scatter(C[q, 0] - c0[0], C[q, 1] - c0[1], s=12, marker=mk, facecolor=col, edgecolor="white", lw=0.35,
                            label=lab, zorder=3)
        ax_.set_xlim(-HALF, HALF)
        ax_.set_ylim(-HALF, HALF)
        bar = cfg["bar"]
        x0 = 0.83 * HALF - bar
        ax_.plot([x0, x0 + bar], [-0.9 * HALF, -0.9 * HALF], color="k", lw=2.5)
        ax_.text(x0 + bar / 2, -0.86 * HALF, f"{bar} m", ha="center", fontsize=8)
        ax_.annotate("N", xy=(0.875 * HALF, 0.925 * HALF), xytext=(0.875 * HALF, 0.73 * HALF), ha="center", fontsize=9,
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=1.3))
        ax_.set_xticks([])
        ax_.set_yticks([])
    a.legend(loc="lower left", fontsize=7.5, framealpha=0.9, title="camera centers", title_fontsize=7.5)
    a.set_title(f"A. {cfg['what']} from above", fontsize=TITLE_PT, loc="left")
    b.set_title("B. Shaded relief", fontsize=TITLE_PT, loc="left")
    fig.canvas.draw()
    pa, pb = a.get_position(), b.get_position()
    wf = pb.x1 - pa.x0
    fw_, fh_ = fig.get_size_inches()
    hf = wf * fw_ / (W / H) / fh_
    from matplotlib.lines import Line2D

    def cam_marks(ax_):
        uc, vc, zc = V_.project(C)
        for kk, (mk, col, _) in CAMERA_STYLE.items():
            q = (fam == kk) & (zc > 0)
            ax_.scatter(uc[q], vc[q], s=7, marker=mk, facecolor=col, edgecolor="white", lw=0.25, zorder=3)
        ax_.set_xlim(0, W)
        ax_.set_ylim(H, 0)
        ax_.set_xticks([])
        ax_.set_yticks([])
    c = fig.add_axes([pa.x0, pa.y0 - 0.055 - hf, wf, hf])
    c.imshow(panel_c, aspect="auto")
    cam_marks(c)
    c.set_title("C. Oblique view, looking north, with rover models", fontsize=TITLE_PT, loc="left")
    c.legend(handles=[Line2D([], [], ls="", marker=m_, markerfacecolor=c_, markeredgecolor="white", markeredgewidth=0.4,
                             markersize=5, label=l_) for k_, (m_, c_, l_) in CAMERA_STYLE.items() if (fam == k_).any()],
             loc="upper right", fontsize=7.5, framealpha=0.9, title="camera centers", title_fontsize=7.5)
    if panel_d is not None:
        pc = c.get_position()
        d_ = fig.add_axes([pa.x0, pc.y0 - 0.055 - hf, wf, hf])
        d_.imshow(panel_d, aspect="auto")
        cam_marks(d_)
        d_.set_title("D. Terrain mesh, same view", fontsize=TITLE_PT, loc="left")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return info


# =========================================================================================== figure 4
FIGURE4 = {"sol0009": (("Sol0009", "Sol0009_S003D0000"), ("freeze:single_sites",)),
           "sol0461": (("Sol0461", "Sol0461_S026D0630"), ("freeze:single_sites", "root:single_sites_8k")),
           "sol0052": (("Sol0052", "Sol0052_S003D2046"), ("freeze:single_sites", "root:single_sites"))}   # 1.0 name, 0.99 name
FIG4_PX = 1600
FIG4_FOV_DEG = 90.0


def read_scape_glb(path: PathLike) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """name -> (V, F, uv, texture RGB uint8) of a full 3D station model written by
    :func:`rpp.ibr.station_mesh.glb_bytes` (one primitive per mesh, a JPEG texture each, shared by name)."""
    import cv2
    from rpp.ibr.rover_surface import read_glb
    doc, binary = read_glb(Path(path).read_bytes())
    views = doc["bufferViews"]

    def view_bytes(i):
        v = views[i]
        return binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]

    def accessor(i):
        a = doc["accessors"][i]
        dt = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[a["componentType"]]
        k = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        return np.frombuffer(view_bytes(a["bufferView"]), dtype=dt, count=a["count"] * k).reshape(a["count"], k)
    tex_cache: Dict[int, np.ndarray] = {}
    out = {}
    for m in doc["meshes"]:
        p = m["primitives"][0]
        V = accessor(p["attributes"]["POSITION"]).astype(np.float64)
        uv = accessor(p["attributes"]["TEXCOORD_0"]).astype(np.float64)
        F = accessor(p["indices"]).reshape(-1, 3).astype(np.int64)
        ti = doc["materials"][p["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        if ti not in tex_cache:
            src = doc["images"][doc["textures"][ti]["source"]]
            tex_cache[ti] = cv2.imdecode(np.frombuffer(view_bytes(src["bufferView"]), np.uint8), cv2.IMREAD_COLOR)[..., ::-1].copy()
        out[m.get("name") or f"mesh{len(out)}"] = (V, F, uv, tex_cache[ti])
    return out


def sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``img`` (h, w, c) at the pixel coordinates ``x``, ``y`` (0 = the first pixel's center), edges repeated."""
    h, w = img.shape[:2]
    x = np.clip(np.asarray(x, np.float64), 0, w - 1)
    y = np.clip(np.asarray(y, np.float64), 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    a = img.astype(np.float32)
    out = (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x1] * fx * (1 - fy) + a[y1, x0] * (1 - fx) * fy + a[y1, x1] * fx * fy)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def render_scape(parts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], view: View,
                 supersample: int = 2) -> np.ndarray:
    """A textured, unlit render of the model's parts (ray cast, ``supersample`` x supersampled), white where nothing."""
    import cv2
    big = view.scaled(supersample)
    names = list(parts)
    r = raycast(big, [(parts[k][0], parts[k][1]) for k in names])
    img = np.full((big.H, big.W, 3), 255, np.uint8)
    for k, name in enumerate(names):
        V, F, uv, tex = parts[name]
        m = r["mesh"] == k
        if not m.any():
            continue
        tri = r["triangle"][m]
        bu, bv = r["uv"][m][:, 0], r["uv"][m][:, 1]
        t = uv[F[tri]]
        st = (1 - bu - bv)[:, None] * t[:, 0] + bu[:, None] * t[:, 1] + bv[:, None] * t[:, 2]
        th, tw = tex.shape[:2]
        img[m] = sample_bilinear(tex, st[:, 0] * tw - 0.5, st[:, 1] * th - 0.5)   # glTF: uv (0, 0) is the top left
    return cv2.resize(img, (view.W, view.H), interpolation=cv2.INTER_AREA)


def _station_folder(labels: Sequence[str], places: Sequence[str], roots: Dict[str, Optional[Path]]) -> Optional[Path]:
    """The first station folder with a full 3D model: the places in order, each with the 1.0 name, then the 0.99 one."""
    for p in places:
        which, sub = p.split(":", 1)
        r = roots.get(which)
        for label in labels:
            if r and sorted((Path(r) / sub / label).glob("*_station_scape.glb")):
                return Path(r) / sub / label
    return None


def figure4_station(key: str, roots: Dict[str, Optional[Path]], out_png: Path, dpi: int = DPI) -> Dict[str, Any]:
    """A station's 3D assets: A and B the stereo 360 panorama's left eye (the top third cut) without and with the rover,
    C the full 3D model's radial texture, D a render of the full 3D model from a head position."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    plt = _plt()
    labels, places = FIGURE4[key]
    label = labels[0]
    d = _station_folder(labels, places, roots)
    if d is None:
        raise Skip(f"no full 3D model (*_station_scape.glb) of {label} under "
                   + ", ".join(f"{roots.get(p.split(':')[0])}\\{p.split(':')[1]}" for p in places))
    glb = next(iter(sorted(d.glob("*_station_scape.glb"))))
    pre = glb.name[:-len("_station_scape.glb")]
    pano_f = d / f"{pre}_360_3DTD.jpg"
    rover_f = next(iter(sorted((d / "with_rover").glob("*_360_3DTD.jpg"))), None) if (d / "with_rover").is_dir() else None
    tex_f = d / f"{pre}_station_scape_terrain.jpg"
    for f_, what in ((pano_f, "terrain-only panorama"), (rover_f, "with-rover panorama"), (tex_f, "radial texture")):
        if f_ is None or not Path(f_).is_file():
            raise Skip(f"{label}: no {what} in {d}")
    pano = np.asarray(Image.open(pano_f).convert("RGB"))
    rov = np.asarray(Image.open(rover_f).convert("RGB"))
    tex = np.asarray(Image.open(tex_f).convert("RGB"))
    parts = read_scape_glb(glb)
    order = [k for k in ("horizon", "terrain", "rover") if k in parts] + [k for k in parts if k not in ("horizon", "terrain", "rover")]
    parts = {k: parts[k] for k in order}
    # the head position (frame: meters about the pivot, x east, y up, z south): 0.4 m from the pivot, looking away
    # from the rover body, a little to the left and down (20 Nov)
    rc = parts["rover"][0].mean(0) if "rover" in parts else np.array([0.0, 0.0, 1.0])
    hd = np.array([rc[0], 0.0, rc[2]])
    hd = hd / np.linalg.norm(hd) if np.linalg.norm(hd) > 1e-6 else np.array([0.0, 0.0, 1.0])
    side = np.cross(hd, [0, 1, 0])
    eye = np.array([0, 0.10, 0]) + 0.30 * side + 0.25 * hd
    f0 = -hd + 0.45 * side + np.array([0, -0.42, 0])
    f0 /= np.linalg.norm(f0)
    r0 = np.cross(f0, [0, 1, 0])
    r0 /= np.linalg.norm(r0)
    fv = f0 - 0.18 * r0 + np.array([0, -0.12, 0])
    f = (FIG4_PX / 2) / np.tan(np.radians(FIG4_FOV_DEG / 2))
    img = render_scape(parts, View(eye, fv, (0, 1, 0), f, FIG4_PX, FIG4_PX))
    hh = pano.shape[0] // 2
    FW, TT, G = 8.5, 0.36, 0.06
    ph = FW / 3
    sq = (FW - 0.15) / 2
    FH = 2 * ph + sq + 3 * TT + 2 * G
    fig = plt.figure(figsize=(FW, FH))
    y = FH
    TP = "Omnidirectional stereo 360-degree panorama in cylindrical projection; left eye; "
    for im, lab in ((pano[hh // 3:hh], "A. " + TP + "terrain only"), (rov[rov.shape[0] // 2 // 3:rov.shape[0] // 2], "B. " + TP + "terrain with rover")):
        ax = fig.add_axes([0, (y - TT - ph) / FH, 1, ph / FH])
        ax.imshow(im, aspect="auto")
        ax.axis("off")
        ax.set_title(lab, fontsize=11, loc="left", pad=3)
        y -= TT + ph + G
    for k, (im, lab) in enumerate(((tex, "C. Full-3D texture projection"), (img, "D. Full-3D model rendered"))):
        ax = fig.add_axes([k * (sq + 0.15) / FW, (y - TT - sq) / FH, sq / FW, sq / FH])
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(lab, fontsize=11, loc="left", pad=3)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"station": d.name, "folder": str(d), "files": [pano_f.name, Path(rover_f).name, tex_f.name, glb.name],
            "render": {"eye_m": [round(float(v), 3) for v in eye], "fov_deg": FIG4_FOV_DEG}}


# =========================================================================================== the list
def entries(roots: Dict[str, Optional[Path]]) -> List[Tuple[str, str, Callable[[Path], Dict[str, Any]]]]:
    """(id, output file, maker(out_png)) of every figure, in the order of the paper."""
    def fig2(key):
        def make(out):
            c = FIGURE2[key]
            order = [roots.get("freeze"), roots.get("root")] if c["freeze"] else [roots.get("root"), roots.get("freeze")]
            b = find_block(order, c["block"])
            if b is None:
                raise Skip(f"no aligned block {c['block']} under {', '.join(str(r) for r in order if r)}")
            return figure2_navcam(b, c["names"], out, scale=c["scale"])
        return make
    return [
        ("figure1", "joss_figure1_pipeline_diagram.png", figure1_pipeline),
        ("figure2", "joss_figure2_navcam_undistorted_sol0009.png", fig2("sol0009")),
        ("figure2b", "joss_figure2b_navcam_undistorted_sol0052.png", fig2("sol0052")),
        ("figure3", "joss_figure3_block_butler_landing.png", lambda out: figure3_block("butler_landing", roots, out)),
        ("figure3b", "joss_figure3b_block_van_zyl_overlook.png", lambda out: figure3_block("van_zyl_overlook", roots, out)),
        ("figure3c", "joss_figure3c_block_rockytop.png", lambda out: figure3_block("rockytop", roots, out)),
        ("figure4", "joss_figure4_station_sol0009.png", lambda out: figure4_station("sol0009", roots, out)),
        ("figure4b", "joss_figure4b_station_sol0461.png", lambda out: figure4_station("sol0461", roots, out)),
        ("figure4c", "joss_figure4c_station_sol0052.png", lambda out: figure4_station("sol0052", roots, out)),
    ]


def make_all(out: PathLike, root: Optional[PathLike] = None, freeze_root: Optional[PathLike] = None,
             only: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """Every figure of :func:`entries` (or ``only`` these ids) into ``out``: id -> "made" or the reason it was skipped."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roots = {"root": Path(root) if root else None, "freeze": Path(freeze_root) if freeze_root else None}
    done = {}
    for fid, name, make in entries(roots):
        if only and fid not in only:
            continue
        try:
            info = make(out / name)
            (out / (Path(name).stem + ".json")).write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
            done[fid] = "made"
            print(f"JOSS {fid}: {name}")
        except Skip as e:
            done[fid] = f"skipped: {e}"
            print(f"JOSS {fid}: skipped - {e}")
        except Exception as e:                                   # noqa: BLE001 - the other figures still made
            done[fid] = f"failed: {type(e).__name__}: {e}"
            print(f"JOSS {fid}: failed ({type(e).__name__}: {e})")
    return done


def default_freeze_root() -> Optional[str]:
    from rpp import env as rpp_env
    for r in (rpp_env.get("RPP_FREEZE_ROOT"), rpp_env.get("RPP_RPV_ROOT"), r"D:\rover_blocks_f3"):
        if r and Path(r).is_dir():
            return r
    return None


def main(argv=None) -> int:
    from rpp import env as rpp_env
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=rpp_env.get("RPP_SCAPES"), help="the blocks' root (default RPP_SCAPES)")
    ap.add_argument("--freeze-root", default=None, help="the freeze-3 root (default RPP_FREEZE_ROOT, RPP_RPV_ROOT or "
                                                        "D:\\rover_blocks_f3)")
    ap.add_argument("--only", nargs="+", default=None, help="figure ids: " + " ".join(e[0] for e in entries({})))
    a = ap.parse_args(argv)
    res = make_all(a.out, a.root, a.freeze_root or default_freeze_root(), a.only)
    return 0 if not any(v.startswith("failed") for v in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
