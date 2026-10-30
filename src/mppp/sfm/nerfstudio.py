"""
A finished block for Nerfstudio (splatfacto) (0.95.2, Christian, 30 Oct 2026: "undistorted in a perspective frame,
with images and masks all of one size and an accurate transformations.json"; "2048 x 1536 because the mipmap will
downsample it further. Zcam frames can be clipped 22 columns on the left and 16 on the right").

:func:`write_nerfstudio` writes ``<project>/nerfstudio/``:

* ``images/<stem>.png`` - every registered frame resampled through COLMAP's exact camera model (the native-pixel
  model ``gui_native/sparse``, so the cameras fit the image files) into a PINHOLE frame of one size (default
  2048 x 1536).  Each frame keeps its own focal length and principal point: the centre resolution of the camera
  scaled by the frame's width (Navcam and Hazcam: 2048 / width; full-resolution frames are reduced 2.5 x, the
  quarter-resolution ones enlarged 1.6 x) - the pinhole frame then spans about 82 deg of a Navcam and 100 deg of
  a Hazcam (the fisheye edges beyond are left out); Mastcam-Z: the 1648-column frame without its first 22 and last
  16 columns, scaled by 1536 / 1200 (square pixels; the 1610 columns are 2061 px at that scale, so 6-7 px are
  trimmed on each side).
* ``masks/<stem>.png`` - the rover masks resampled the same way (nearest neighbour), 0 where the frame has no
  source pixel (outside the fisheye's pinhole window, the clipped Mastcam-Z columns): Nerfstudio ignores pixels
  at 0.
* ``transforms.json`` - Nerfstudio's format: ``camera_model`` OPENCV with no distortion, per frame ``fl_x``,
  ``fl_y``, ``cx``, ``cy``, ``w``, ``h`` and ``transform_matrix`` (camera to world, OpenGL camera axes: x right,
  y up, z back - COLMAP's with y and z flipped) in MPPP's world frame of the block (x east, y north, z up, metres;
  the block offset in ``mppp``); ``ply_file_path`` the tie points (``sparse_pc.ply``) for splatfacto's start.

    ns-train splatfacto --data <project>/nerfstudio nerfstudio-data

Nerfstudio's dataparser orients, centres and scales the scene itself (``--orientation-method``,
``--center-method``, ``--auto-scale-poses``); the files are left in metres.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .project import SfmProject

log = logging.getLogger("mppp.sfm")
PathLike = Union[str, Path]
NERFSTUDIO_DIR = "nerfstudio"
NS_SIZE = (2048, 1536)                 # one size for every frame (Christian, 30 Oct)
ZCAM_WIDTH = 1648                      # a full Mastcam-Z frame
ZCAM_CLIP = (22, 16)                   # columns left out of a Mastcam-Z frame: left, right (Christian, 30 Oct)


def _fxfycxcy(model: str, p: Sequence[float]) -> Tuple[float, float, float, float]:
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def target_camera(model: str, params: Sequence[float], width: int, height: int, family: str,
                  size: Tuple[int, int] = NS_SIZE, zcam_clip: Tuple[int, int] = ZCAM_CLIP) -> Dict[str, Any]:
    """The pinhole camera of one source camera in the common frame: {"fl_x", "fl_y", "cx", "cy", "w", "h", "scale",
    "window": [x0, y0, x1, y1] - the source pixels used}.  ``family``: N, F, R (centre resolution, the frame's
    principal point kept) or Z (the clipped window fitted by height)."""
    W, H = int(size[0]), int(size[1])
    fx, fy, cx, cy = _fxfycxcy(model, params)
    w, h = int(width), int(height)
    if str(family).upper().startswith("Z"):
        k = w / ZCAM_WIDTH
        x0, x1 = zcam_clip[0] * k, w - zcam_clip[1] * k
        s = H / h
        xc, yc = 0.5 * (x0 + x1), 0.5 * h
    else:
        x0, x1 = 0.0, float(w)
        s = W / w
        xc, yc = 0.5 * w, 0.5 * h
    return {"fl_x": fx * s, "fl_y": fy * s, "cx": 0.5 * W + (cx - xc) * s, "cy": 0.5 * H + (cy - yc) * s,
            "w": W, "h": H, "scale": s, "window": [float(x0), 0.0, float(x1), float(h)]}


def remap_tables(cam, tgt: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every target pixel (centre at +0.5, COLMAP's convention) its source position for ``cv2.remap`` (pixel
    centres at integers) and whether it falls in the source window."""
    W, H = tgt["w"], tgt["h"]
    u, v = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5, np.arange(H, dtype=np.float64) + 0.5)
    rays = np.stack([(u - tgt["cx"]) / tgt["fl_x"], (v - tgt["cy"]) / tgt["fl_y"], np.ones_like(u)], -1).reshape(-1, 3)
    uv = np.asarray(cam.img_from_cam(rays), np.float64).reshape(H, W, 2)
    x0, y0, x1, y1 = tgt["window"]
    ok = np.all(np.isfinite(uv), -1) & (uv[..., 0] >= x0) & (uv[..., 0] < x1) & (uv[..., 1] >= y0) & (uv[..., 1] < y1)
    mx = np.where(ok, uv[..., 0] - 0.5, -1).astype(np.float32)
    my = np.where(ok, uv[..., 1] - 0.5, -1).astype(np.float32)
    return mx, my, ok


def _family(name: str, row: Dict[str, Any]) -> str:
    g = str(row.get("camera_group") or row.get("instrument") or name)
    return g[:1].upper()


def write_ply(path: PathLike, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """A binary PLY of points with colours (Nerfstudio's ``ply_file_path``)."""
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, np.uint8).reshape(-1, 3)
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii")
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["r"], rec["g"], rec["b"] = rgb.T
    with open(path, "wb") as f:
        f.write(head)
        f.write(rec.tobytes())


def write_nerfstudio(project: SfmProject, out_dir: Optional[PathLike] = None, model_dir: Optional[PathLike] = None,
                     size: Tuple[int, int] = NS_SIZE, zcam_clip: Tuple[int, int] = ZCAM_CLIP, fmt: str = "png",
                     families: Optional[Sequence[str]] = None, threads: int = 8,
                     overwrite: bool = False) -> Dict[str, Any]:
    """``<project>/nerfstudio/`` (see the module).  ``families``: only these (N, F, R, Z); ``fmt``: png or jpg for
    the images (the masks are PNG).  Frames already written are kept unless ``overwrite``."""
    import cv2
    import pycolmap
    from .dense import native_model_dir
    root = Path(project.root)
    out = Path(out_dir) if out_dir else root / NERFSTUDIO_DIR
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    mdir = Path(model_dir) if model_dir else native_model_dir(project)
    rec = pycolmap.Reconstruction(str(mdir))
    rows = {r["name"]: r for r in project.images}
    fams = {f.upper()[:1] for f in families} if families else None
    order = sorted((int(i) for i in rec.reg_image_ids()), key=lambda i: rec.images[i].name)
    order = [i for i in order if fams is None or _family(rec.images[i].name, rows.get(rec.images[i].name, {})) in fams]
    targets: Dict[Tuple[int, str], Dict[str, Any]] = {}
    tables: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for iid in order:
        im = rec.images[iid]
        fam = _family(im.name, rows.get(im.name, {}))
        key = (int(im.camera_id), "Z" if fam == "Z" else "E")
        if key not in targets:
            cam = rec.cameras[im.camera_id]
            targets[key] = target_camera(cam.model.name, cam.params, cam.width, cam.height, fam, size, zcam_clip)
    ext = ".jpg" if str(fmt).lower() in ("jpg", "jpeg") else ".png"

    def table(key):
        if key not in tables:
            tables[key] = remap_tables(rec.cameras[key[0]], targets[key])
        return tables[key]

    for key in targets:                                    # the tables first (pycolmap in this thread)
        table(key)

    def one(iid):
        im = rec.images[iid]
        stem = Path(im.name).stem
        fam = _family(im.name, rows.get(im.name, {}))
        key = (int(im.camera_id), "Z" if fam == "Z" else "E")
        ip, mp = out / "images" / (stem + ext), out / "masks" / (stem + ".png")
        if overwrite or not (ip.is_file() and mp.is_file()):
            mx, my, ok = tables[key]
            src = cv2.imread(str(project.images_dir / im.name), cv2.IMREAD_UNCHANGED)
            if src is None:
                return iid, f"unreadable {im.name}"
            img = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            m = ok.copy()
            mf = project.masks_dir / (im.name + ".png")
            if mf.is_file():
                ms = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if ms is not None:
                    m &= cv2.remap(ms, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
            img[~m] = 0
            cv2.imwrite(str(ip), img, [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3])
            cv2.imwrite(str(mp), (m * 255).astype(np.uint8))
        return iid, None

    with ThreadPoolExecutor(max(1, int(threads))) as ex:
        results = list(ex.map(one, order))
    errors = [e for _, e in results if e]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    frames = []
    for iid, err in results:
        if err:
            continue
        im = rec.images[iid]
        stem = Path(im.name).stem
        fam = _family(im.name, rows.get(im.name, {}))
        t = targets[(int(im.camera_id), "Z" if fam == "Z" else "E")]
        wc = im.cam_from_world().inverse()
        T = np.eye(4)
        T[:3, :3] = np.asarray(wc.rotation.matrix(), float)
        T[:3, 3] = np.asarray(wc.translation, float)
        frames.append({"file_path": f"images/{stem}{ext}", "mask_path": f"masks/{stem}.png",
                       "transform_matrix": (T @ flip).tolist(),
                       "fl_x": t["fl_x"], "fl_y": t["fl_y"], "cx": t["cx"], "cy": t["cy"], "w": t["w"], "h": t["h"],
                       "colmap_image_id": int(iid), "camera": rows.get(im.name, {}).get("camera_group") or fam})
    pids = sorted(int(p) for p in rec.points3D)
    xyz = np.array([rec.points3D[p].xyz for p in pids], float).reshape(-1, 3)
    rgb = np.array([rec.points3D[p].color for p in pids], np.uint8).reshape(-1, 3)
    write_ply(out / "sparse_pc.ply", xyz, rgb)
    from .metashape import _block_name, _version
    meta = {"camera_model": "OPENCV", "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "w": int(size[0]), "h": int(size[1]),
            "ply_file_path": "sparse_pc.ply", "frames": frames,
            "mppp": {"version": _version(), "block": _block_name(project), "model": str(mdir),
                     "world_frame": project.settings.get("world_frame"),
                     "offset_m": [float(v) for v in (project.offset or [0, 0, 0])],
                     "axes": "world x east, y north, z up (metres, the block offset removed); cameras OpenGL "
                             "(x right, y up, z back)",
                     "size": list(size), "zcam_clip_columns": list(zcam_clip),
                     "cameras": {f"{k[0]}{k[1]}": v for k, v in targets.items()}}}
    (out / "transforms.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"folder": str(out), "frames": len(frames), "errors": errors, "points": len(pids),
            "size": list(size), "cameras": len(targets)}


__all__ = ["write_nerfstudio", "target_camera", "remap_tables", "NS_SIZE", "ZCAM_CLIP", "NERFSTUDIO_DIR"]
