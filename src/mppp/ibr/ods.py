"""
mppp.ibr.ods - the omni-directional stereo (ODS) panorama of a station: two equirect images, left and right eye.

Prototype (Christian, 9 Nov 2026: "0.96.0 should prototype the stereoscopic pipeline.  The output should just be
two images.  Don't build the webviewer.").  Plan of 31 Oct, step 4.

Geometry: the viewer's eyes sit on a circle of diameter IPD (64 mm) about the station's pivot, in the local
horizontal plane.  The pixel (u, v) of a W x H equirect looks along azimuth theta (from north through east) and
elevation phi; the left eye's ray starts at ``P - (IPD/2) right(theta)`` and the right eye's at ``P + (IPD/2)
right(theta)`` with ``right = h x up`` and ``h`` the horizontal view direction - the standard ODS construction.
Both rays are cast at the station's surface (mesh within 20 m of the mast axis, the ground plane to 100 m, the sky at
infinity; :mod:`.raycast`): the hit point X and its label.

Colour: X is projected into every image of the station (the Navcam frames of that waypoint; ``--cameras`` adds
Mastcam-Z) through the exact camera model; an image is a candidate when X falls inside it, in front of it, is not
masked, and the segment camera -> X meets no surface before X (ray cast; tolerance tau(z)).  Each candidate is
scored by the angle between its ray to X and the eye's ray (smaller = less parallax error) and a border feather;
the two best are blended by their scores.  Holes - no surface, or no image sees the point - are black and
recorded in the provenance map (``provenance.png``: the source image index; 0 = no data).

Resolution: 8192 x 4096 is 0.77 mrad/px, about the Navcam's half-resolution sampling; the processed images are
read at ``scale`` (0.5 for the Navcam).  Works in row chunks so the memory stays bounded.  ``ods.json`` records
the pivot, the images, the eye separation and the coverage.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .mesh import tau_m
from .raycast import LABEL_FAR, LABEL_MESH, LABEL_NONE, LABEL_PLANE, Scene

log = logging.getLogger("mppp.ibr.ods")
PathLike = Union[str, Path]

DEFAULT_WIDTH = 8192
DEFAULT_IPD_M = 0.064
FEATHER_PX = 40.0                # a source's weight falls to zero this close to its image border
MAX_SOURCES_BLENDED = 2


def ods_rays(width: int, height: int, rows: Tuple[int, int], pivot: Sequence[float], ipd_m: float, eye: str,
             up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """Origins and unit directions (n, 3) of the equirect pixels in ``rows`` = (r0, r1) for ``eye`` ("left" /
    "right" / "centre").  Column 0 looks north and the azimuth grows eastward (clockwise seen from above), so the
    panorama reads N, E, S, W from left to right; row 0 is the zenith."""
    P = np.asarray(pivot, float)
    e, n, u = (np.asarray(v, float) for v in (east, north, up))
    r0, r1 = rows
    az = (np.arange(width) + 0.5) / width * 2 * np.pi
    el = np.pi / 2 - (np.arange(r0, r1) + 0.5) / height * np.pi
    A, E = np.meshgrid(az, el)
    h = np.sin(A)[..., None] * e + np.cos(A)[..., None] * n                 # horizontal view direction
    d = np.cos(E)[..., None] * h + np.sin(E)[..., None] * u
    sign = {"left": -1.0, "right": 1.0, "centre": 0.0, "center": 0.0}[eye]
    right = np.cross(h, u)                                                 # points to the viewer's right
    o = P + sign * (ipd_m / 2.0) * right
    return o.reshape(-1, 3), d.reshape(-1, 3)


class SourceImage:
    """One candidate image: its camera, pose, native mask and the colour image at ``scale`` (loaded on demand)."""

    def __init__(self, name: str, camera, R: np.ndarray, C: np.ndarray, image_file: Path, mask_file: Optional[Path], scale: float):
        self.name, self.camera, self.R, self.C = name, camera, np.asarray(R, float), np.asarray(C, float)
        self.image_file, self.mask_file, self.scale = Path(image_file), mask_file, float(scale)
        self._rgb: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self.W = max(1, int(round(camera.width * scale)))
        self.H = max(1, int(round(camera.height * scale)))

    def project(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pixels at scale (n, 2), depth z (n,), inside (n,)) of the world points ``X``."""
        Xc = X @ self.R.T + (-self.R @ self.C)            # R X + t with t = -R C
        z = Xc[:, 2]
        front = z > 1e-3
        uv = np.full((len(X), 2), np.nan)
        if front.any():
            uv[front] = np.asarray(self.camera.img_from_cam(Xc[front]), float)
        uv *= self.scale
        inside = front & np.all(np.isfinite(uv), axis=1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] < self.H)
        return uv, z, inside

    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            import cv2
            img = cv2.imread(str(self.image_file), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(self.image_file)
            if img.shape[1] != self.W or img.shape[0] != self.H:
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            self._rgb = img[..., ::-1].copy()
        return self._rgb

    def mask(self) -> Optional[np.ndarray]:
        if self.mask_file is None or not Path(self.mask_file).is_file():
            return None
        if self._mask is None:
            import cv2
            m = cv2.imread(str(self.mask_file), cv2.IMREAD_GRAYSCALE)
            if m is not None and (m.shape[1] != self.W or m.shape[0] != self.H):
                m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            self._mask = None if m is None else (m > 0)
        return self._mask

    def release(self) -> None:
        self._rgb = None

    def sample(self, uv: np.ndarray) -> np.ndarray:
        """Bilinear colour (n, 3) float at the scaled pixel coordinates ``uv``."""
        from scipy.ndimage import map_coordinates
        img = self.rgb()
        coords = [uv[:, 1] - 0.5, uv[:, 0] - 0.5]
        return np.stack([map_coordinates(img[..., c].astype(np.float32), coords, order=1, mode="nearest") for c in range(3)], axis=1)


def station_sources(project, rec, station, families: str = "N", scale: float = 0.5, zcam_scale: float = 1.0,
                    images: Optional[Sequence[str]] = None, use_masks: bool = True) -> List[SourceImage]:
    """The station's frames as sources; ``use_masks=False`` (``mppp ods --rover``) lets the rover's own pixels in."""
    by_name = {im.name: im for im in rec.images.values()}
    names = list(images) if images else [n for n in station.images if n[:1].upper() in set(families.upper())]
    out = []
    for name in names:
        im = by_name.get(name)
        if im is None or not im.has_pose:
            continue
        cw = im.cam_from_world()
        s = zcam_scale if name[:1].upper() == "Z" else scale
        mf = Path(project.root) / "masks" / (name + ".png")
        out.append(SourceImage(name, rec.cameras[im.camera_id], np.asarray(cw.rotation.matrix(), float),
                               np.asarray(im.projection_center(), float), Path(project.root) / "images" / name,
                               mf if (use_masks and mf.is_file()) else None, s))
    return out


def render_eye(scene: Scene, sources: Sequence[SourceImage], pivot: Sequence[float], eye: str, width: int, height: int,
               ipd_m: float, rows_per_chunk: int = 128, up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0),
               feather_px: float = FEATHER_PX, progress=None) -> Dict[str, np.ndarray]:
    """One eye's equirect: {"rgb" (H, W, 3) uint8, "source" (H, W) int16 (-1 none; index into sources), "label"
    (H, W) uint8 surface label, "range" (H, W) float32}."""
    P = np.asarray(pivot, float)
    rgb = np.zeros((height, width, 3), np.uint8)
    source = np.full((height, width), -1, np.int16)
    label = np.zeros((height, width), np.uint8)
    rng = np.zeros((height, width), np.float32)
    t0 = time.time()
    for r0 in range(0, height, rows_per_chunk):
        r1 = min(height, r0 + rows_per_chunk)
        o, d = ods_rays(width, height, (r0, r1), P, ipd_m, eye, up, north, east)
        hit = scene.cast(o, d)
        lab = hit["label"]
        ok = lab != LABEL_NONE
        X = o + hit["t"][:, None] * d
        X[~ok] = P
        n = len(d)
        best_s = np.zeros((n, MAX_SOURCES_BLENDED), np.float32)           # scores (higher better)
        best_i = np.full((n, MAX_SOURCES_BLENDED), -1, np.int32)
        best_uv = np.zeros((n, MAX_SOURCES_BLENDED, 2), np.float32)
        for k, src in enumerate(sources):
            uv, z, inside = src.project(X)
            cand = inside & ok
            if not cand.any():
                continue
            m = src.mask()
            if m is not None:
                ij = cand.nonzero()[0]
                px = np.clip(uv[ij].astype(int), 0, [src.W - 1, src.H - 1])
                cand[ij[~m[px[:, 1], px[:, 0]]]] = False
            if not cand.any():
                continue
            ij = cand.nonzero()[0]
            # visibility: nothing between the camera and X (within tau(z) of X)
            # (tau of at most 100 m: a point on the sky is still hidden by a rock in front of the camera)
            occ = scene.occluded(np.broadcast_to(src.C, (len(ij), 3)), X[ij], tau_m(np.minimum(z[ij], 100.0)))
            ij = ij[~occ]
            if not len(ij):
                continue
            v_src = X[ij] - src.C
            v_src /= np.linalg.norm(v_src, axis=1)[:, None]
            cosang = np.clip(np.sum(v_src * d[ij], axis=1), -1, 1)
            ang = np.arccos(cosang)                                        # parallax between the eye and the source
            border = np.minimum.reduce([uv[ij, 0], uv[ij, 1], src.W - uv[ij, 0], src.H - uv[ij, 1]])
            feather = np.clip(border / feather_px, 0.0, 1.0)
            score = (feather / (ang + 0.01)).astype(np.float32)
            # keep the two best per pixel
            better0 = score > best_s[ij, 0]
            better1 = ~better0 & (score > best_s[ij, 1])
            i0 = ij[better0]
            best_s[i0, 1], best_i[i0, 1], best_uv[i0, 1] = best_s[i0, 0], best_i[i0, 0], best_uv[i0, 0]
            best_s[i0, 0], best_i[i0, 0], best_uv[i0, 0] = score[better0], k, uv[i0]
            i1 = ij[better1]
            best_s[i1, 1], best_i[i1, 1], best_uv[i1, 1] = score[better1], k, uv[i1]
        # colour: blend the best two by score
        col = np.zeros((n, 3), np.float32)
        wsum = np.zeros(n, np.float32)
        for slot in range(MAX_SOURCES_BLENDED):
            for k in np.unique(best_i[:, slot]):
                if k < 0:
                    continue
                sel = np.flatnonzero(best_i[:, slot] == k)
                c = sources[k].sample(best_uv[sel, slot])
                w = best_s[sel, slot]
                col[sel] += c * w[:, None]
                wsum[sel] += w
        has = wsum > 0
        col[has] /= wsum[has][:, None]
        rgb[r0:r1] = np.clip(col, 0, 255).astype(np.uint8).reshape(r1 - r0, width, 3)
        src_idx = np.where(has, best_i[:, 0], -1).astype(np.int16)
        source[r0:r1] = src_idx.reshape(r1 - r0, width)
        label[r0:r1] = lab.reshape(r1 - r0, width)
        rng[r0:r1] = np.where(ok, hit["t"], 0).reshape(r1 - r0, width)
        if progress:
            progress(eye, r1, height, time.time() - t0)
    return {"rgb": rgb, "source": source, "label": label, "range": rng}


def provenance_png(source: np.ndarray, out_png: PathLike, n_sources: int) -> Path:
    """The source index per pixel as colours (golden-angle hues), black = no data."""
    import cv2
    img = np.zeros(source.shape + (3,), np.uint8)
    for k in range(max(0, n_sources)):
        m = source == k
        if not m.any():
            continue
        hue = int((k * 137.508) % 180)
        hsv = np.array([[[hue, 200, 230]]], np.uint8)
        img[m] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return Path(out_png)


def render_station(project, rec, station, scene: Scene, out_dir: PathLike, width: int = DEFAULT_WIDTH, ipd_m: float = DEFAULT_IPD_M,
                   families: str = "N", scale: float = 0.5, zcam_scale: float = 1.0, images: Optional[Sequence[str]] = None,
                   rows_per_chunk: int = 128, eyes: Sequence[str] = ("left", "right"), use_masks: bool = True) -> Dict[str, Any]:
    """The two eye images of a station -> ``out_dir/ods_left.png``, ``ods_right.png``, ``provenance.png`` (left
    eye's sources), ``ods.json``.  Returns the record."""
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    height = width // 2
    sources = station_sources(project, rec, station, families=families, scale=scale, zcam_scale=zcam_scale, images=images,
                              use_masks=use_masks)
    if not sources:
        raise ValueError(f"station {station.label}: no source images of families {families!r} with a pose")
    axes = dict(up=station.up, north=station.north, east=station.east)
    t0 = time.time()

    def progress(eye, r, h, dt):
        log.info("ods %s: %d / %d rows (%.0f s)", eye, r, h, dt)

    rec_out: Dict[str, Any] = {"station": station.to_json(), "width": width, "height": height, "ipd_m": ipd_m,
                               "sources": [s.name for s in sources], "scale": scale, "zcam_scale": zcam_scale,
                               "rover": "shown" if not use_masks else "masked out", "eyes": {}}
    for eye in eyes:
        res = render_eye(scene, sources, station.pivot, eye, width, height, ipd_m, rows_per_chunk=rows_per_chunk,
                         progress=progress, **axes)
        f = out / f"ods_{eye}.png"
        cv2.imwrite(str(f), res["rgb"][..., ::-1])
        if eye == eyes[0]:
            provenance_png(res["source"], out / "provenance.png", len(sources))
        lab = res["label"]
        n = lab.size
        rec_out["eyes"][eye] = {"file": str(f), "coverage": {"coloured": float((res["source"] >= 0).sum() / n),
                                "mesh": float((lab == LABEL_MESH).sum() / n), "plane": float((lab == LABEL_PLANE).sum() / n),
                                "far": float((lab == LABEL_FAR).sum() / n),
                                "no_surface": float((lab == 0).sum() / n)},
                                "sources_used": int(len(np.unique(res["source"][res["source"] >= 0])))}
        for s in sources:
            s.release()
    rec_out["seconds"] = round(time.time() - t0, 1)
    (out / "ods.json").write_text(json.dumps(rec_out, indent=1, default=float), encoding="utf-8")
    log.info("ods: %s -> %s (%.0f s)", station.label, out, rec_out["seconds"])
    return rec_out
