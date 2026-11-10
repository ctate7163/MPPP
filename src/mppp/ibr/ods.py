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
masked, and the segment camera -> X meets no surface before X (ray cast; tolerance tau(z)).  Holes - no surface, or
no image sees the point - are black and recorded in the provenance map (``provenance.png``: the source image index).

Blending (10 Nov 2026, Christian: "the mosaic should be blended together to avoid jagged borders"):

- **feathers from every edge**: a candidate's weight rises smoothly (smoothstep) with its distance from the nearest
  edge of what it may contribute - the image border *and* the rover mask's edge (a distance transform per source) -
  so no seam follows a mask outline or a frame border;
- **two bands**: the low band (each source blurred, sigma ``LOW_SIGMA_FRAC`` of its short side) is blended over a
  wide feather (``FEATHER_LOW_FRAC`` of the short side), the detail (image - low band) over a narrow one, so
  brightness and shading change across a wide strip while the texture stays sharp and is not ghosted;
- **all of the best four** candidates per pixel, weighted by the feather and a soft parallax term
  1 / (1 + angle / 0.02 rad) (the eye's ray against the source's ray to X);
- **gain compensation**: one gain per source, from the low-band brightness of the pixels that two sources share
  (a first pass at 1024 px; medians of the log ratios per pair, least squares with a weak pull to 1, clipped to
  0.6-1.6), so frames of different exposures or sols meet without a step.  ``ods.json`` records the gains;
- **no black cracks** (10 Nov, the Van Zyl S003D2046 run): a ray through a gap in the mesh ends on the plane or
  the terrain behind, which the mesh then hides from every camera - a black triangle on the deck, a sliver along a
  mesh edge.  A pixel no source sees unhidden takes the best hidden one (by the same weight); cracks still left
  (at most ``HOLE_FILL_MAX_PX`` pixels and ``HOLE_FILL_MAX_WIDTH_PX`` wide) are filled from their edges
  (``cv2.inpaint``) and marked in the provenance map (grey); large unseen areas (straight down, above the
  frames' tops) stay black.

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
FEATHER_PX = 40.0                # (0.96.0 prototype) a source's weight fell to zero this close to its image border
MAX_SOURCES_BLENDED = 4
FEATHER_LOW_FRAC = 0.15          # the low band's feather: this fraction of the source's short side
FEATHER_HIGH_FRAC = 0.03         # the detail band's feather
LOW_SIGMA_FRAC = 0.006           # the low band: Gaussian sigma, this fraction of the short side (~6 px at 960)
ANGLE_SCALE_RAD = 0.02           # parallax term 1 / (1 + angle / this)
GAIN_RANGE = (0.6, 1.6)
HOLE_FILL_MAX_PX = 4000          # holes (a surface, no source) up to this many pixels are filled from their edges
HOLE_FILL_MAX_WIDTH_PX = 24      # ... and only if thin: no pixel farther than this from the hole's edge
GAIN_PASS_WIDTH = 1024


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
        self._low: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self.gain = 1.0
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
        self._low = None
        self._dist = None

    @property
    def short_side(self) -> int:
        return min(self.W, self.H)

    def low(self) -> np.ndarray:
        """The low band: the image blurred (sigma LOW_SIGMA_FRAC of the short side), float32."""
        if self._low is None:
            import cv2
            sig = max(0.8, LOW_SIGMA_FRAC * self.short_side)
            self._low = cv2.GaussianBlur(self.rgb().astype(np.float32), (0, 0), sig)
        return self._low

    def edge_distance(self) -> np.ndarray:
        """Distance (pixels at scale) of every pixel from the nearest edge of what this source may give: the image
        border and the rover mask's edge."""
        if self._dist is None:
            import cv2
            ok = np.ones((self.H, self.W), np.uint8)
            m = self.mask()
            if m is not None:
                ok &= m.astype(np.uint8)
            pad = np.zeros((self.H + 2, self.W + 2), np.uint8)
            pad[1:-1, 1:-1] = ok
            self._dist = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[1:-1, 1:-1].astype(np.float32)
        return self._dist

    def _bilinear(self, img: np.ndarray, uv: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        coords = [uv[:, 1] - 0.5, uv[:, 0] - 0.5]
        if img.ndim == 2:
            return map_coordinates(img, coords, order=1, mode="nearest")
        return np.stack([map_coordinates(img[..., c], coords, order=1, mode="nearest") for c in range(img.shape[2])], axis=1)

    def sample(self, uv: np.ndarray) -> np.ndarray:
        """Bilinear colour (n, 3) float at the scaled pixel coordinates ``uv``."""
        return self._bilinear(self.rgb().astype(np.float32), uv)

    def sample_bands(self, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(low band, full image) colours (n, 3) at ``uv``."""
        return self._bilinear(self.low(), uv), self.sample(uv)

    def sample_edge_distance(self, uv: np.ndarray) -> np.ndarray:
        return self._bilinear(self.edge_distance(), uv)


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


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def candidates(scene: Scene, sources: Sequence[SourceImage], o: np.ndarray, d: np.ndarray, k_max: int = MAX_SOURCES_BLENDED):
    """The surface hit of the rays (o, d) and, per ray, the best ``k_max`` sources that see it: {"label", "t",
    "k" (n, K) source index or -1, "uv" (n, K, 2), "dist" (n, K) edge distance in px, "short" (n, K) the source's
    short side, "ang" (n, K) the parallax term}.  Ranked by the wide feather x the parallax term."""
    hit = scene.cast(o, d)
    lab = hit["label"]
    ok = lab != LABEL_NONE
    X = o + hit["t"][:, None] * d
    X[~ok] = np.asarray(o, float).reshape(-1, 3)[0] if np.ndim(o) == 1 else X[~ok]
    n = len(d)
    K = int(k_max)
    best_w = np.zeros((n, K), np.float32)
    best = {"k": np.full((n, K), -1, np.int32), "uv": np.zeros((n, K, 2), np.float32), "dist": np.zeros((n, K), np.float32),
            "short": np.ones((n, K), np.float32), "ang": np.zeros((n, K), np.float32)}
    fb_w = np.zeros(n, np.float32)
    fb_k = np.full(n, -1, np.int32)
    fb_uv = np.zeros((n, 2), np.float32)
    fb_dist = np.zeros(n, np.float32)
    fb_short = np.ones(n, np.float32)
    for k, src in enumerate(sources):
        uv, z, inside = src.project(X)
        cand = inside & ok
        if not cand.any():
            continue
        ij = cand.nonzero()[0]
        dist = src.sample_edge_distance(uv[ij])
        keep = dist > 0.5                                    # inside the image and off the rover mask
        ij, dist = ij[keep], dist[keep]
        if not len(ij):
            continue
        # visibility: nothing between the camera and X (within tau(z) of X; tau of at most 100 m: a point on the sky
        # is still hidden by a rock in front of the camera)
        occ = scene.occluded(np.broadcast_to(src.C, (len(ij), 3)), X[ij], tau_m(np.minimum(z[ij], 100.0)))
        if occ.any():                                        # the best hidden view, for pixels no source sees unhidden
            io, do = ij[occ], dist[occ]
            wo = _smoothstep(do / (FEATHER_LOW_FRAC * src.short_side)).astype(np.float32) + 1e-6
            b = wo > fb_w[io]
            io, wo, do = io[b], wo[b], do[b]
            fb_w[io], fb_k[io], fb_uv[io], fb_dist[io], fb_short[io] = wo, k, uv[io], do, src.short_side
        ij, dist = ij[~occ], dist[~occ]
        if not len(ij):
            continue
        v_src = X[ij] - src.C
        v_src /= np.linalg.norm(v_src, axis=1)[:, None]
        ang = np.arccos(np.clip(np.sum(v_src * d[ij], axis=1), -1, 1))
        aterm = 1.0 / (1.0 + ang / ANGLE_SCALE_RAD)
        w = (_smoothstep(dist / (FEATHER_LOW_FRAC * src.short_side)) * aterm).astype(np.float32) + 1e-6
        slot = np.argmin(best_w[ij], axis=1)                 # streaming top-K: replace the weakest kept
        take = w > best_w[ij, slot]
        ij, slot, w = ij[take], slot[take], w[take]
        best_w[ij, slot] = w
        best["k"][ij, slot] = k
        best["uv"][ij, slot] = uv[ij]
        best["dist"][ij, slot] = dist[take]
        best["short"][ij, slot] = src.short_side
        best["ang"][ij, slot] = aterm[take]
    none = best["k"][:, 0] < 0
    for slot in range(1, K):
        none &= best["k"][:, slot] < 0
    use = none & (fb_k >= 0)                                 # nobody sees it unhidden: the best hidden view
    best["k"][use, 0], best["uv"][use, 0], best["dist"][use, 0] = fb_k[use], fb_uv[use], fb_dist[use]
    best["short"][use, 0], best["ang"][use, 0] = fb_short[use], 1.0
    best.update(label=lab, t=hit["t"], ok=ok, hidden_fallback=use)
    return best


def estimate_gains(scene: Scene, sources: Sequence[SourceImage], pivot: Sequence[float], width: int = GAIN_PASS_WIDTH,
                   up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0), rows_per_chunk: int = 128) -> Dict[str, Any]:
    """One gain per source from the pixels two sources share (low-band brightness; the centre eye at ``width``):
    a_k - a_l = median(log L_l - log L_k) per pair, least squares with a weak pull to a = 0, g = exp(a) clipped to
    GAIN_RANGE.  Sets ``src.gain``; returns {"gains": {name: g}, "pairs": n, "samples": n}."""
    height = width // 2
    sums: Dict[Tuple[int, int], list] = {}
    for r0 in range(0, height, rows_per_chunk):
        r1 = min(height, r0 + rows_per_chunk)
        o, d = ods_rays(width, height, (r0, r1), pivot, 0.0, "centre", up, north, east)
        c = candidates(scene, sources, o, d)
        K = c["k"].shape[1]
        lum = np.full(c["k"].shape, np.nan, np.float32)
        for k in np.unique(c["k"][c["k"] >= 0]):
            sel = c["k"] == k
            low = sources[k].low()
            L = sources[k]._bilinear(low.mean(axis=2), c["uv"][sel])
            inner = c["dist"][sel] > 0.5 * FEATHER_LOW_FRAC * c["short"][sel]        # well inside: no edge effects
            lum[sel] = np.where(inner & (L > 8.0), np.log(np.maximum(L, 1e-3)), np.nan)
        for a in range(K):
            for b in range(a + 1, K):
                both = np.isfinite(lum[:, a]) & np.isfinite(lum[:, b])
                if not both.any():
                    continue
                ka, kb = c["k"][both, a], c["k"][both, b]
                diff = lum[both, b] - lum[both, a]
                swap = ka > kb                                   # key (low index, high index), diff = log L_high - log L_low
                lo_k, hi_k = np.where(swap, kb, ka), np.where(swap, ka, kb)
                diff = np.where(swap, -diff, diff)
                key = lo_k.astype(np.int64) * 100_000 + hi_k
                order = np.argsort(key, kind="stable")
                uk, st_ = np.unique(key[order], return_index=True)
                for kk, a0, a1 in zip(uk, st_, np.r_[st_[1:], len(order)]):
                    sums.setdefault((int(kk // 100_000), int(kk % 100_000)), []).extend(diff[order[a0:a1]].tolist())
    n_src = len(sources)
    pairs = [(k, l, float(np.median(v)), len(v)) for (k, l), v in sums.items() if len(v) >= 50]
    a = np.zeros(n_src)
    if pairs:
        A = np.zeros((len(pairs) + n_src, n_src))
        y = np.zeros(len(pairs) + n_src)
        wts = np.zeros(len(pairs) + n_src)
        for i, (k, l, m, cnt) in enumerate(pairs):
            A[i, k], A[i, l], y[i], wts[i] = 1.0, -1.0, m, np.sqrt(cnt)
        lam = 0.05 * float(np.median([np.sqrt(p[3]) for p in pairs]))
        for k in range(n_src):
            A[len(pairs) + k, k], wts[len(pairs) + k] = 1.0, lam
        a = np.linalg.lstsq(A * wts[:, None], y * wts, rcond=None)[0]
    g = np.clip(np.exp(a), *GAIN_RANGE)
    for s_, gk in zip(sources, g):
        s_.gain = float(gk)
    return {"gains": {s_.name: round(float(gk), 4) for s_, gk in zip(sources, g)}, "pairs": len(pairs),
            "samples": int(sum(p[3] for p in pairs))}


def render_eye(scene: Scene, sources: Sequence[SourceImage], pivot: Sequence[float], eye: str, width: int, height: int,
               ipd_m: float, rows_per_chunk: int = 128, up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0),
               feather_px: float = FEATHER_PX, progress=None) -> Dict[str, np.ndarray]:
    """One eye's equirect: {"rgb" (H, W, 3) uint8, "source" (H, W) int16 (-1 none; the strongest source), "label"
    (H, W) uint8 surface label, "range" (H, W) float32}.  The two-band blend of the module doc, with each source's
    ``gain`` (:func:`estimate_gains`; 1 when not estimated)."""
    P = np.asarray(pivot, float)
    rgb = np.zeros((height, width, 3), np.uint8)
    source = np.full((height, width), -1, np.int16)
    label = np.zeros((height, width), np.uint8)
    rng = np.zeros((height, width), np.float32)
    hidden = np.zeros((height, width), bool)
    t0 = time.time()
    for r0 in range(0, height, rows_per_chunk):
        r1 = min(height, r0 + rows_per_chunk)
        o, d = ods_rays(width, height, (r0, r1), P, ipd_m, eye, up, north, east)
        c = candidates(scene, sources, o, d)
        hidden[r0:r1] = c["hidden_fallback"].reshape(r1 - r0, width)
        n = len(d)
        low_acc = np.zeros((n, 3), np.float32)
        hi_acc = np.zeros((n, 3), np.float32)
        w_low_sum = np.zeros(n, np.float32)
        w_hi_sum = np.zeros(n, np.float32)
        strongest = np.full(n, -1, np.int32)
        strongest_w = np.zeros(n, np.float32)
        K = c["k"].shape[1]
        for slot in range(K):
            ks = c["k"][:, slot]
            for k in np.unique(ks[ks >= 0]):
                sel = np.flatnonzero(ks == k)
                src = sources[k]
                lo, full = src.sample_bands(c["uv"][sel, slot])
                dist = c["dist"][sel, slot]
                aterm = c["ang"][sel, slot]
                w_lo = _smoothstep(dist / (FEATHER_LOW_FRAC * src.short_side)) * aterm
                w_hi = _smoothstep(dist / (FEATHER_HIGH_FRAC * src.short_side)) * aterm ** 3
                g = src.gain
                low_acc[sel] += (g * lo) * w_lo[:, None]
                hi_acc[sel] += (g * (full - lo)) * w_hi[:, None]
                w_low_sum[sel] += w_lo
                w_hi_sum[sel] += w_hi
                better = w_lo > strongest_w[sel]
                strongest[sel[better]] = k
                strongest_w[sel[better]] = w_lo[better]
        has = w_low_sum > 1e-6
        col = np.zeros((n, 3), np.float32)
        col[has] = low_acc[has] / w_low_sum[has][:, None]
        hh = has & (w_hi_sum > 1e-6)
        col[hh] += hi_acc[hh] / w_hi_sum[hh][:, None]
        rgb[r0:r1] = np.clip(col, 0, 255).astype(np.uint8).reshape(r1 - r0, width, 3)
        source[r0:r1] = np.where(has, strongest, -1).astype(np.int16).reshape(r1 - r0, width)
        label[r0:r1] = c["label"].reshape(r1 - r0, width)
        rng[r0:r1] = np.where(c["ok"], c["t"], 0).reshape(r1 - r0, width)
        if progress:
            progress(eye, r1, height, time.time() - t0)
    filled = fill_cracks(rgb, source, label)
    return {"rgb": rgb, "source": source, "label": label, "range": rng, "hidden_fallback": hidden, "filled": filled}


def fill_cracks(rgb: np.ndarray, source: np.ndarray, label: np.ndarray, max_px: int = HOLE_FILL_MAX_PX,
                max_width_px: float = HOLE_FILL_MAX_WIDTH_PX) -> np.ndarray:
    """Fill (in place) the small, thin holes - a surface but no source - from their edges; ``source`` = -2 there.
    Returns the filled mask."""
    import cv2
    hole = ((source < 0) & (label != LABEL_NONE)).astype(np.uint8)
    filled = np.zeros(hole.shape, bool)
    if not hole.any():
        return filled
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
    dist = cv2.distanceTransform(hole, cv2.DIST_L2, 3)
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_px
    if small.any():
        widest = np.zeros(n, np.float32)
        np.maximum.at(widest, lab.ravel(), dist.ravel())
        small &= widest <= max_width_px
        small[0] = False
    filled = small[lab]
    if filled.any():
        ys, xs = np.nonzero(filled)
        y0, y1, x0, x1 = max(0, ys.min() - 8), min(rgb.shape[0], ys.max() + 9), max(0, xs.min() - 8), min(rgb.shape[1], xs.max() + 9)
        sub = cv2.inpaint(np.ascontiguousarray(rgb[y0:y1, x0:x1]), filled[y0:y1, x0:x1].astype(np.uint8), 5, cv2.INPAINT_TELEA)
        rgb[y0:y1, x0:x1][filled[y0:y1, x0:x1]] = sub[filled[y0:y1, x0:x1]]
        source[filled] = -2
    return filled


def provenance_png(source: np.ndarray, out_png: PathLike, n_sources: int) -> Path:
    """The source index per pixel as colours (golden-angle hues), black = no data."""
    import cv2
    img = np.zeros(source.shape + (3,), np.uint8)
    img[source == -2] = (128, 128, 128)                       # cracks filled from their edges
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
                   rows_per_chunk: int = 128, eyes: Sequence[str] = ("left", "right"), use_masks: bool = True,
                   gains: bool = True) -> Dict[str, Any]:
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
    if gains:
        g = estimate_gains(scene, sources, station.pivot, width=min(GAIN_PASS_WIDTH, width), rows_per_chunk=rows_per_chunk, **axes)
        rec_out["gains"] = g
        log.info("ods: gains over %d source pairs: %s", g["pairs"], ", ".join(f"{v:.3f}" for v in g["gains"].values()))
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
                                "no_surface": float((lab == 0).sum() / n),
                                "hidden_view_used": float(res["hidden_fallback"].sum() / n),
                                "cracks_filled": float(res["filled"].sum() / n)},
                                "sources_used": int(len(np.unique(res["source"][res["source"] >= 0])))}
        for s in sources:
            s.release()
    rec_out["seconds"] = round(time.time() - t0, 1)
    (out / "ods.json").write_text(json.dumps(rec_out, indent=1, default=float), encoding="utf-8")
    log.info("ods: %s -> %s (%.0f s)", station.label, out, rec_out["seconds"])
    return rec_out
