"""
mppp.ibr.ods - the omni-directional stereo (ODS) panorama of a station: two equirect images, left and right eye.

Prototype (Christian, 9 Nov 2026: "0.96.0 should prototype the stereoscopic pipeline.  The output should just be
two images.  Don't build the webviewer.").  Plan of 31 Oct, step 4.

Geometry: the viewer's eyes sit on a circle of diameter IPD (64 mm) about the station's pivot, in the local
horizontal plane.  The pixel (u, v) of a W x H equirect looks along azimuth theta (from north through east) and
elevation phi; the left eye's ray starts at ``P - (IPD/2) right(theta)`` and the right eye's at ``P + (IPD/2)
right(theta)`` with ``right = h x up`` and ``h`` the horizontal view direction - the standard ODS construction.
Both rays are cast at the station's surface (mesh within 30 m of the mast axis, the ground plane to 100 m, the sky at
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
  (``cv2.inpaint``) and marked in the provenance map (grey); large unseen areas below the horizon (straight down)
  stay black;
- **the sky above the frames** (10 Nov, Christian: "extrapolate and blur the light blue sky upwards so that there
  isn't a shape boundary there"): per column the colour of the band just below the highest sky pixel a frame saw,
  that boundary and its colour smoothed around the panorama (no sawtooth from the frames' tops), the colour carried
  up and blended into the panorama's mean sky colour at the zenith, cross-faded with the seen sky over a band about
  the smoothed boundary; light blue in the provenance map; ``--no-sky-fill``.

Resolution: 8192 x 4096 is 0.77 mrad/px; a half-resolution Navcam frame (2560 px) is 0.68 mrad/px, a full-resolution
one 0.34.  The processed images are read at ``scale`` of their own size - 1 by default (10 Nov; 0.5 before, which
made the half-resolution frames 1.36 mrad/px, coarser than the panorama); 0.5 suits full-resolution frames.  Works in row chunks so the memory stays bounded.  ``ods.json`` records
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
# 0.97.0 (Christian, 12 Nov 2026, Quest 3: "when I tilt my head down ... I go cross-eyed"): pole merging.  An ODS
# pair is right only at the horizon: towards a pole each column keeps its own eye offset, so in a view looking down
# the columns to the side carry their disparity along the wrong direction for the viewer's eyes (vertical
# misalignment), and near surfaces (the deck 1 m below) carry far too much of it.  The eye separation is full to
# POLE_MERGE_DEG[0] of elevation (up or down), falls as a cosine and is zero from POLE_MERGE_DEG[1]: one image
# for both eyes at the poles.  None: no merging (the panoramas before 12 Nov).
POLE_MERGE_DEG: Optional[Tuple[float, float]] = (30.0, 80.0)


def pole_merge(el_rad, merge_deg=POLE_MERGE_DEG) -> np.ndarray:
    """The share of the eye separation at elevation ``el_rad`` (1 at the horizon, 0 at the poles)."""
    a = np.abs(np.degrees(np.asarray(el_rad, float)))
    if not merge_deg:
        return np.ones_like(a)
    lo, hi = float(merge_deg[0]), float(merge_deg[1])
    return np.cos(0.5 * np.pi * np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
FEATHER_PX = 40.0                # (0.96.0 prototype) a source's weight fell to zero this close to its image border
MAX_SOURCES_BLENDED = 4
EDGE_MARGIN_PX = 3.0             # a source gives nothing within this many pixels of its border, padding or mask edge
                                 # (10 Nov: the frames' outermost rows are darker - a faint line where the sky fill starts)
FEATHER_LOW_FRAC = 0.15          # the low band's feather: this fraction of the source's short side
FEATHER_HIGH_FRAC = 0.03         # the detail band's feather
LOW_SIGMA_FRAC = 0.006           # the low band: Gaussian sigma, this fraction of the short side (~6 px at 960)
ANGLE_SCALE_RAD = 0.02           # parallax term 1 / (1 + angle / this)
GAIN_RANGE = (0.6, 1.6)
HOLE_FILL_MAX_PX = 4000          # holes (a surface, no source) up to this many pixels are filled from their edges
HOLE_FILL_MAX_WIDTH_PX = 24      # ... and only if thin: no pixel farther than this from the hole's edge
PAD_MAX_VALUE = 1                # a source pixel this dark in every channel, in a region touching the border, is padding
PAD_MIN_FRAC = 0.002             # ... when the region is at least this fraction of the image
SKY_MIN_VALUE = 12               # the sky fill samples no pixel darker than this (no black or near-black areas)
SKY_ENVELOPE_FRAC = 0.10         # the boundary: the lowest of the frames' tops within this fraction of the width
SKY_SKIP_FRAC = 0.004            # the sky colour sampled this far (of the height) below each column's highest seen pixel
SKY_BAND_FRAC = 0.03             #   over a band this tall, from its brightest SKY_BRIGHT_SHARE (sky, not the distant
SKY_BRIGHT_SHARE = 0.4           #   hills the band can reach - both are "far"), median
SKY_SMOOTH_FRAC = 0.02           # the boundary and the colour smoothed around the panorama (Gaussian, of the width)
SKY_EDGE_ROWS_FRAC = 0.001       # the first rows below each column's highest seen sky pixel are replaced by the fill
SKY_FEATHER_FRAC = 0.02          # the fill cross-fades with the seen sky over this band (of the height) about the boundary
GAIN_PASS_WIDTH = 1024


def ods_rays(width: int, height: int, rows: Tuple[int, int], pivot: Sequence[float], ipd_m: float, eye: str,
             up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0), azimuth0_deg: float = 0.0, pole_merge_deg="default") -> Tuple[np.ndarray, np.ndarray]:
    """Origins and unit directions (n, 3) of the equirect pixels in ``rows`` = (r0, r1) for ``eye`` ("left" /
    "right" / "centre").  The left edge of column 0 looks along azimuth ``azimuth0_deg`` (0 = north, the default;
    -180 puts north in the middle) and the azimuth grows eastward (clockwise seen from above), so by default the
    panorama reads N, E, S, W from left to right; row 0 is the zenith."""
    P = np.asarray(pivot, float)
    e, n, u = (np.asarray(v, float) for v in (east, north, up))
    r0, r1 = rows
    az = (np.arange(width) + 0.5) / width * 2 * np.pi + np.radians(float(azimuth0_deg))
    el = np.pi / 2 - (np.arange(r0, r1) + 0.5) / height * np.pi
    A, E = np.meshgrid(az, el)
    h = np.sin(A)[..., None] * e + np.cos(A)[..., None] * n                 # horizontal view direction
    d = np.cos(E)[..., None] * h + np.sin(E)[..., None] * u
    sign = {"left": -1.0, "right": 1.0, "centre": 0.0, "center": 0.0}[eye]
    right = np.cross(h, u)                                                 # points to the viewer's right
    share = pole_merge(E, POLE_MERGE_DEG if isinstance(pole_merge_deg, str) else pole_merge_deg)
    o = P + sign * (ipd_m / 2.0) * share[..., None] * right
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
        self.extra_mask: Optional[np.ndarray] = None      # True = may not be used (the mast-head hardware, rover shown)
        self.rover_mask_file: Optional[Path] = None       # 0.97.0, rover shown: the rover mask kept for the surface check
        self._rover: Optional[np.ndarray] = None
        self.linear_file: Optional[Path] = None           # 0.96.1: the 16-bit linear image (mppp.ibr.linear16)
        self.sat_file: Optional[Path] = None              # ... and its saturation mask
        self._sat: Optional[np.ndarray] = None
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
        """The colour image at scale: uint8, or - from the 16-bit linear image - float32 in the 8-bit image's units
        (the 16-bit value / 64: up to 1024, not clipped at 255), saturated pixels made neutral."""
        if self._rgb is None and self.linear_file is not None:
            import cv2
            img = cv2.imread(str(self.linear_file), cv2.IMREAD_UNCHANGED)
            if img is None or img.dtype != np.uint16:
                raise FileNotFoundError(f"{self.linear_file}: not a 16-bit image")
            img = img[..., :3][..., ::-1].astype(np.float32) / 64.0
            sat = cv2.imread(str(self.sat_file), cv2.IMREAD_GRAYSCALE) if self.sat_file and Path(self.sat_file).is_file() else None
            if sat is not None and sat.any():
                m = sat > 0
                img[m] = img[m].max(axis=1, keepdims=True)          # white at its brightness, not the white balance's tint
            if img.shape[1] != self.W or img.shape[0] != self.H:
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
                if sat is not None:
                    sat = cv2.resize(sat, (self.W, self.H), interpolation=cv2.INTER_AREA)
            self._sat = None if sat is None or not sat.any() else cv2.GaussianBlur(sat.astype(np.float32) / 255.0, (0, 0), 1.5)
            self._rgb = np.ascontiguousarray(img)
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

    def sample_rover(self, uv: np.ndarray) -> Optional[np.ndarray]:
        """True where the pixel at ``uv`` is inside the frame's rover mask (rover hardware, and whatever else the mask
        covers: its shadow); None without ``rover_mask_file``."""
        if self.rover_mask_file is None or not Path(self.rover_mask_file).is_file():
            return None
        if self._rover is None:
            import cv2
            m = cv2.imread(str(self.rover_mask_file), cv2.IMREAD_GRAYSCALE)
            if m is None:
                self.rover_mask_file = None
                return None
            if m.shape[1] != self.W or m.shape[0] != self.H:
                m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            self._rover = ~(m > 0)                            # the mask file: > 0 is terrain
        uv = np.asarray(uv, float)
        i = np.clip(np.rint(uv[:, 0]).astype(int), 0, self.W - 1)
        j = np.clip(np.rint(uv[:, 1]).astype(int), 0, self.H - 1)
        return self._rover[j, i]

    def sample_saturation(self, uv: np.ndarray) -> Optional[np.ndarray]:
        """The saturated share (0-1) at ``uv``; None for a frame without saturation (or an 8-bit source)."""
        self.rgb()
        return None if self._sat is None else self._bilinear(self._sat, uv)

    def release(self) -> None:
        self._rgb = None
        self._sat = None
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

    def padding(self) -> np.ndarray:
        """The black padding of the processed frame (MPPP pads the subframes to the frame size): pixels at most
        PAD_MAX_VALUE in every channel in a connected region that touches the image border and covers at least
        PAD_MIN_FRAC of the image (10 Nov: the padding was blended in as image content - the fade to black at the
        frames' tops)."""
        import cv2
        z = (self.rgb().max(axis=2) <= PAD_MAX_VALUE).astype(np.uint8)
        pad = np.zeros((self.H, self.W), bool)
        if not z.any():
            return pad
        n, lab, stats, _ = cv2.connectedComponentsWithStats(z, connectivity=4)
        border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        for k in border:
            if k and stats[k, cv2.CC_STAT_AREA] >= PAD_MIN_FRAC * self.H * self.W:
                pad |= lab == k
        return pad

    def edge_distance(self) -> np.ndarray:
        """Distance (pixels at scale) of every pixel from the nearest edge of what this source may give: the image
        border, the frame's black padding, the rover mask's edge (and, with the rover shown, the hardware fixed to
        the mast head: ``extra_mask``)."""
        if self._dist is None:
            import cv2
            ok = (~self.padding()).astype(np.uint8)
            m = self.mask()
            if m is not None:
                ok &= m.astype(np.uint8)
            if self.extra_mask is not None:
                em = self.extra_mask
                if em.shape != ok.shape:
                    em = cv2.resize(em.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
                ok &= (~em).astype(np.uint8)
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


HEAD_HARDWARE_FRACTION = 0.75    # rover shown: a pixel the rover mask covers in this share of a tier's frames is mast-head
HEAD_TIER_DEG = 4.0              # hardware (it turns with the cameras); the frames grouped by eye and elevation (deg)
HEAD_MIN_FRAMES = 3


def head_hardware_masks(sources: Sequence["SourceImage"], masks: Sequence[Optional[np.ndarray]], up: Sequence[float],
                        fraction: float = HEAD_HARDWARE_FRACTION, tier_deg: float = HEAD_TIER_DEG,
                        min_frames: int = HEAD_MIN_FRAMES) -> List[Optional[np.ndarray]]:
    """The hardware fixed to the mast head (the wind-sensor booms below the cameras; Christian, 10 Nov: "what can we
    do about the light colored boom on the side of the mast?").  It turns with the cameras, so it sits at the same
    pixels of every frame of one eye at one elevation, while the deck moves across the frames as the mast turns: per
    group (eye, elevation tier) of at least ``min_frames`` frames, the pixels the rover mask covers in at least
    ``fraction`` of them.  Returns, per source, True = mast-head hardware (or None)."""
    u = np.asarray(up, float) / np.linalg.norm(up)
    groups: Dict[Tuple[str, int, Tuple[int, int]], List[int]] = {}
    for i, (src, m) in enumerate(zip(sources, masks)):
        if m is None:
            continue
        z = src.R[2]                                          # the optical axis in the block frame
        el = float(np.degrees(np.arcsin(np.clip(z @ u, -1, 1))))
        groups.setdefault((src.name[:2], int(round(el / tier_deg)), m.shape), []).append(i)
    out: List[Optional[np.ndarray]] = [None] * len(sources)
    for (_eye, _tier, _shape), ids in groups.items():
        if len(ids) < min_frames:
            continue
        rover_share = np.mean([~masks[i] for i in ids], axis=0)          # masks: True = terrain
        fixed = rover_share >= fraction
        if fixed.any():
            for i in ids:
                out[i] = fixed & ~masks[i]                    # only where this frame's own mask says rover
    return out


def station_sources(project, rec, station, families: str = "N", scale: float = 1.0, zcam_scale: float = 1.0,
                    images: Optional[Sequence[str]] = None, use_masks: bool = True,
                    mask_head_hardware: bool = True, linear16: Optional[Dict[str, Any]] = None,
                    mast_pose: Optional[Dict[str, Any]] = None) -> List[SourceImage]:
    """The station's frames as sources; ``use_masks=False`` (``mppp ods --rover``) lets the rover's own pixels in,
    except the hardware fixed to the mast head (:func:`head_hardware_masks`) unless ``mask_head_hardware`` is off."""
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
        if not use_masks and mf.is_file():                    # rover shown: the mask still says which pixels are rover
            out[-1].rover_mask_file = mf
        if linear16 is not None:
            out[-1].linear_file, out[-1].sat_file = linear16[name]
    if not use_masks and mask_head_hardware and out:
        import cv2
        ms = []
        for src in out:
            mf = Path(project.root) / "masks" / (src.name + ".png")
            m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) if mf.is_file() else None
            if m is not None and (m.shape[1] != src.W or m.shape[0] != src.H):
                m = cv2.resize(m, (src.W, src.H), interpolation=cv2.INTER_NEAREST)
            ms.append(None if m is None else m > 0)
        hw = head_hardware_masks(out, ms, station.up)
        for src, h in zip(out, hw):
            src.extra_mask = h
        n = sum(1 for h in hw if h is not None and h.any())
        log.info("ods: mast-head hardware masked in %d of %d frames (rover shown)", n, len(out))
    if not use_masks and mast_pose is not None and out:      # 0.96.1: the mast and its wind-sensor booms, from the model
        from .rover_model import available, mast_masks
        if available():
            mm = mast_masks(out, mast_pose)
            for src, m in zip(out, mm):
                if m is not None:
                    src.extra_mask = m if src.extra_mask is None else (src.extra_mask | m)
                    src.mast_masked = True
            log.info("ods: the mast and its booms masked in %d of %d frames (the rover model at the rover pose)",
                     sum(1 for m in mm if m is not None), len(out))
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


SURFACE_CHECK = True              # 0.97.0, rover shown: see candidates()
BROWN_FILL_8BIT = (150.0, 105.0, 70.0)   # RGB, 8-bit units: the fill where the check leaves a surface without a source and
#                                          the panorama has no terrain to take the colour from (else: the median terrain colour)
BROWN_FILL_DARKEN = 0.8                  # the fill a little darker than the terrain, so it reads as fill


def candidates(scene: Scene, sources: Sequence[SourceImage], o: np.ndarray, d: np.ndarray, k_max: int = MAX_SOURCES_BLENDED,
               surface_check: Optional[bool] = None):
    """The surface hit of the rays (o, d) and, per ray, the best ``k_max`` sources that see it: {"label", "t",
    "k" (n, K) source index or -1, "uv" (n, K, 2), "dist" (n, K) edge distance in px, "short" (n, K) the source's
    short side, "ang" (n, K) the parallax term}.  Ranked by the wide feather x the parallax term.

    ``surface_check`` (0.97.0; Christian, 12 Nov 2026: "nothing from the rover mask lands on the terrain and nothing
    from the terrain mask lands on the rover"; default :data:`SURFACE_CHECK`): with the rover model in the mesh
    (``scene.rover_triangles``) and a source that kept its rover mask (``rover_mask_file``: the rover-shown render),
    a source pixel inside the rover mask may colour the rover model only, and one outside it anything but the model.
    ``"rejected"``: [rover pixels kept off the terrain, terrain pixels kept off the rover] (samples)."""
    if surface_check is None:
        surface_check = SURFACE_CHECK
    hit = scene.cast(o, d)
    on_rover = scene.on_rover(hit["tri"]) if (surface_check and hasattr(scene, "on_rover") and "tri" in hit) else None
    rejected = [0, 0]
    rej_px = np.zeros(len(d), bool)                          # rays with a source sample the check turned away
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
        keep = dist > EDGE_MARGIN_PX                          # inside the image, off the padding and the rover mask
        ij, dist = ij[keep], dist[keep]
        if on_rover is not None and len(ij):                  # 0.97.0: rover pixels on the rover, terrain on terrain
            rp = src.sample_rover(uv[ij])
            if rp is not None:
                surf = on_rover[ij]
                rejected[0] += int((rp & ~surf).sum())
                rejected[1] += int((~rp & surf).sum())
                same = rp == surf
                rej_px[ij[~same]] = True
                ij, dist = ij[same], dist[same]
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
    best.update(label=lab, t=hit["t"], ok=ok, hidden_fallback=use, rejected=rejected, rejected_px=rej_px,
                on_rover=on_rover if on_rover is not None else np.zeros(n, bool))
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
               feather_px: float = FEATHER_PX, progress=None, sky_fill: bool = True,
               azimuth0_deg: float = 0.0, linear: bool = False, ground_fill: bool = True, pole_merge_deg="default") -> Dict[str, np.ndarray]:
    """One eye's equirect: {"rgb" (H, W, 3) uint8 - or, ``linear``, float32 in 8-bit units up to 1024 (x 64 = the
    16-bit linear value) -, "source" (H, W) int16 (-1 none; the strongest source), "label"
    (H, W) uint8 surface label, "range" (H, W) float32}.  The two-band blend of the module doc, with each source's
    ``gain`` (:func:`estimate_gains`; 1 when not estimated)."""
    P = np.asarray(pivot, float)
    rgb = np.zeros((height, width, 3), np.float32 if linear else np.uint8)
    top = 4.0 * LINEAR_TOP if linear else 255.0             # float: not clipped here; the 16-bit file has its own scale
    source = np.full((height, width), -1, np.int16)
    label = np.zeros((height, width), np.uint8)
    rng = np.zeros((height, width), np.float32)
    hidden = np.zeros((height, width), bool)
    ground = np.zeros((height, width), bool)                 # terrain at the ground plane's level (the ground fill's donors)
    rej = [0, 0]                                             # the surface check's rejected samples
    rover_px = np.zeros((height, width), bool)               # the rover model under the pixel
    rej_map = np.zeros((height, width), bool)                # a source sample was turned away by the surface check
    u_ = np.asarray(up, float) / np.linalg.norm(up)
    pl = getattr(scene, "plane", None)
    t0 = time.time()
    for r0 in range(0, height, rows_per_chunk):
        r1 = min(height, r0 + rows_per_chunk)
        o, d = ods_rays(width, height, (r0, r1), P, ipd_m, eye, up, north, east, azimuth0_deg, pole_merge_deg=pole_merge_deg)
        c = candidates(scene, sources, o, d)
        rej[0] += c["rejected"][0]; rej[1] += c["rejected"][1]
        rover_px[r0:r1] = c["on_rover"].reshape(r1 - r0, width)
        rej_map[r0:r1] = c["rejected_px"].reshape(r1 - r0, width)
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
                sat = src.sample_saturation(c["uv"][sel, slot])
                if sat is not None:                              # a saturated pixel: another frame, where there is one
                    keep = (1.0 - SAT_DOWNWEIGHT * np.clip(sat, 0.0, 1.0)).astype(np.float32)
                    w_lo, w_hi = w_lo * keep, w_hi * keep
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
        rgb[r0:r1] = np.clip(col, 0, top).astype(rgb.dtype).reshape(r1 - r0, width, 3)
        if pl is not None:
            tt = np.where(c["ok"] & np.isfinite(c["t"]), c["t"], 0.0)
            X = o + tt[:, None] * d
            hgt = (X @ pl["normal"] - pl["offset"]) / float(pl["normal"] @ u_)
            g = has & ~c["on_rover"] & ((c["label"] == LABEL_PLANE) | ((c["label"] == LABEL_MESH) & (np.abs(hgt) < GROUND_BAND_M)))
            ground[r0:r1] = g.reshape(r1 - r0, width)
        source[r0:r1] = np.where(has, strongest, -1).astype(np.int16).reshape(r1 - r0, width)
        label[r0:r1] = c["label"].reshape(r1 - r0, width)
        rng[r0:r1] = np.where(c["ok"], c["t"], 0).reshape(r1 - r0, width)
        if progress:
            progress(eye, r1, height, time.time() - t0)
    brown = np.zeros(label.shape, bool)
    checked = getattr(scene, "rover_triangles", None) is not None

    def _brown(sel: np.ndarray) -> None:                     # 0.97.0 (Christian, 12 Nov 2026: "where the disagreement is
        if not sel.any():                                    # too much ... leave that a brown fill color")
            return
        don = ground & (source >= 0)
        col = (np.median(rgb[don].reshape(-1, 3).astype(np.float32), axis=0) if don.sum() >= 50
               else np.asarray(BROWN_FILL_8BIT, np.float32))
        rgb[sel] = np.clip(BROWN_FILL_DARKEN * col, 0, top).astype(rgb.dtype)
        source[sel] = -5
        brown[sel] = True
    if checked:                                              # the rover model where the check left no source: flat brown,
        _brown((source == -1) & rover_px & rej_map)          # before the crack and ground fills would paint over it
    filled = fill_cracks(rgb, source, label)
    if checked:
        _brown((source == -1) & rover_px)                    # the rest of the model nobody coloured
    gfill = fill_ground(rgb, source, ground) if ground_fill else np.zeros(label.shape, bool)
    sky = fill_sky(rgb, source, label) if sky_fill else np.zeros(label.shape, bool)
    if checked:                                              # and any surface still without a source
        _brown((source == -1) & ((label == LABEL_MESH) | (label == LABEL_PLANE)))
    return {"brown_filled": brown, "rover": rover_px, "rgb": rgb, "source": source, "label": label, "range": rng, "hidden_fallback": hidden, "filled": filled,
            "sky_filled": sky, "ground_filled": gfill, "surface_check_rejected": rej}


GROUND_BAND_M = 0.25             # mesh this close to the ground plane counts as terrain for the ground fill
GROUND_PAD_FRAC = 0.25           # the panorama wrapped by this share of its width for the fill (no seam at north)


def fill_ground(rgb: np.ndarray, source: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Fill (in place) what no frame coloured below the horizon - under the rover, behind the arm, through holes in
    the rover - as plain terrain (Christian, 10 Nov: "the holes in the rover should be filled at ground level. Same
    with the area occluded by the arm. It shouldn't be zero, but continuous with a planar terrain"): the colours of
    the terrain pixels (``donors``: coloured, on the ground plane or mesh at its level - not the rover) carried
    inwards by a pull-push pyramid, so the fill is continuous with the terrain at its rim and smooth far from it.
    ``source`` = -4 there.  Returns the mask of the pixels filled."""
    import cv2
    H, W = source.shape
    rows = np.arange(H)
    below = ((90.0 - (rows + 0.5) / H * 180.0) <= 0.0)[:, None]
    holes = below & (source == -1)
    if not holes.any() or donors.sum() < 50:
        return np.zeros((H, W), bool)
    pad = max(1, int(GROUND_PAD_FRAC * W))
    wrap = lambda a: np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)      # noqa: E731
    wgt = wrap((donors & below).astype(np.float32))
    img = wrap(rgb.astype(np.float32)) * wgt[..., None]
    pyr = [(img, wgt)]
    while min(pyr[-1][1].shape) > 2:                          # pull: weighted averages, coarser and coarser
        a, w = pyr[-1]
        size = (max(1, (a.shape[1] + 1) // 2), max(1, (a.shape[0] + 1) // 2))
        pyr.append((cv2.resize(a, size, interpolation=cv2.INTER_AREA), cv2.resize(w, size, interpolation=cv2.INTER_AREA)))
    a, w = pyr[-1]
    cur = a / np.maximum(w, 1e-6)[..., None]
    cur[w <= 1e-6] = (a.reshape(-1, 3).sum(axis=0) / max(float(w.sum()), 1e-6))
    for a, w in reversed(pyr[:-1]):                           # push: where a level knows little, the coarser one
        up_ = cv2.resize(cur, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        conf = np.clip(w, 0.0, 1.0)[..., None]
        cur = a + (1.0 - conf) * up_
    fill = cur[:, pad:pad + W]
    top = 255.0 if rgb.dtype == np.uint8 else LINEAR_TOP
    rgb[holes] = np.clip(fill[holes], 0, top).astype(rgb.dtype)
    source[holes] = -4
    return holes


def fill_sky(rgb: np.ndarray, source: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Fill (in place) the sky no frame saw, above the horizon (see the module doc).  The boundary is the lower
    envelope of the frames' tops (the lowest top within SKY_ENVELOPE_FRAC of the width), smoothed; the colour the
    median of the seen sky SKY_SKIP_FRAC below it (past the frames' darker top edges), smoothed around the panorama;
    carried up into the panorama's median sky colour at the zenith; the seen sky above the boundary and within
    SKY_FEATHER_FRAC of it cross-faded into the fill.  ``source`` = -3 where nothing was seen.  Returns the mask of
    the pixels changed (empty when no sky was seen at all)."""
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    H, W = label.shape
    rows = np.arange(H)
    above = (90.0 - (rows + 0.5) / H * 180.0) > 0.0
    sky = ((label == LABEL_FAR) | (source == -1)) & above[:, None]     # 10 Nov: an uncoloured surface up there too
    bright = rgb.max(axis=2) >= SKY_MIN_VALUE
    seen = sky & (source != -1) & bright                     # no black or near-black pixel is carried up
    changed = np.zeros((H, W), bool)
    has = seen.any(axis=0)
    if has.sum() < max(8, W // 100):
        return changed
    idx = np.arange(W)
    good = np.flatnonzero(has)
    r_top = np.where(has, np.argmax(seen, axis=0), 0).astype(float)
    r_top[~has] = np.interp(idx[~has], good, r_top[good], period=W)
    env = maximum_filter1d(r_top, size=max(3, int(SKY_ENVELOPE_FRAC * W)), mode="wrap")     # the notches' bottoms
    sig = max(2.0, SKY_SMOOTH_FRAC * W)
    r_s = gaussian_filter1d(env, sig, mode="wrap")
    skip, band = max(1, int(SKY_SKIP_FRAC * H)), max(2, int(SKY_BAND_FRAC * H))
    horizon = int(np.searchsorted(-(90.0 - (rows + 0.5) / H * 180.0), 0.0))          # the first row below the horizon
    col = np.full((W, 3), np.nan)
    for c in np.flatnonzero(has):
        r0 = int(r_top[c]) + skip
        sl = slice(min(H - 1, r0), max(min(H - 1, r0) + 1, min(horizon, r0 + band)))
        m = seen[sl, c]
        if m.sum() >= min(3, band):
            px = rgb[sl, c][m].astype(float)
            lum = px.mean(axis=1)
            keep = lum >= np.quantile(lum, 1.0 - SKY_BRIGHT_SHARE)
            col[c] = np.median(px[keep], axis=0)
    ok = np.isfinite(col[:, 0])
    if ok.sum() < max(8, W // 100):
        return changed
    g = np.flatnonzero(ok)
    for k in range(3):
        col[~ok, k] = np.interp(idx[~ok], g, col[g, k], period=W)
    zenith = np.median(col[ok], axis=0)
    # the horizontal blur widens upwards as 1 / cos(elevation) - the same width on the sky at every elevation, so the
    # column streaks do not converge at the pole (Christian, 10 Nov) - from levels sig * 2^k, interpolated per row
    levels = []
    sg = sig
    while True:
        levels.append(np.stack([gaussian_filter1d(col[:, k], min(sg, 4.0 * W), mode="wrap") for k in range(3)], axis=1))
        if sg >= W:
            break
        sg *= 2.0
    levels.append(np.broadcast_to(zenith, (W, 3)).astype(float))      # the last level: uniform
    levels = np.stack(levels)                                          # (L, W, 3)
    el_rows = 90.0 - (rows + 0.5) / H * 180.0
    el_b = 90.0 - (r_s.mean() + 0.5) / H * 180.0                        # the boundary's elevation (mean)
    want = 1.0 / np.maximum(np.cos(np.radians(el_rows)), 1e-3) * np.cos(np.radians(min(el_b, 89.0)))
    lk = np.clip(np.log2(np.maximum(want, 1.0)), 0.0, len(levels) - 1.0)  # level per row (0 at and below the boundary)
    k0 = np.floor(lk).astype(int)
    k1 = np.minimum(k0 + 1, len(levels) - 1)
    fr = (lk - k0)[:, None]
    R = rows[:, None].astype(float)
    t = np.clip((r_s[None, :] - R) / np.maximum(r_s[None, :], 1.0), 0.0, 1.0)          # 0 at the boundary, 1 at the zenith
    sm = t * t * (3 - 2 * t)
    feather = max(2.0, SKY_FEATHER_FRAC * H)
    w = np.clip((r_s[None, :] + feather - R) / (2.0 * feather), 0.0, 1.0)              # the cross-fade about the boundary
    w = w * w * (3 - 2 * w)
    hole = sky & ((source == -1) | ~bright)
    w = np.where(hole & (R <= r_s[None, :] + 3 * feather), 1.0, w)      # dark leftovers near the boundary: the fill
    edge = max(2, int(round(SKY_EDGE_ROWS_FRAC * H)))                    # the frames' top rows: the fill (Christian:
    top_c = np.where(has, np.argmax(seen, axis=0), -1)                   # "a faint dark line where the extrapolation
    w = np.where((top_c[None, :] >= 0) & (R >= top_c[None, :]) & (R < top_c[None, :] + edge), 1.0, w)   # starts")
    w = np.where(sky, w, 0.0)
    changed = w > 1e-3
    ys, xs = np.nonzero(changed)
    c_row = (1 - fr[ys]) * levels[k0[ys], xs] + fr[ys] * levels[k1[ys], xs]
    fill = (1 - sm[ys, xs])[:, None] * c_row + sm[ys, xs][:, None] * zenith[None, :]
    ww = w[ys, xs][:, None]
    top = 255.0 if rgb.dtype == np.uint8 else LINEAR_TOP
    rgb[ys, xs] = np.clip((1 - ww) * rgb[ys, xs].astype(np.float32) + ww * fill, 0, top).astype(rgb.dtype)
    source[hole] = -3
    return changed


def fill_cracks(rgb: np.ndarray, source: np.ndarray, label: np.ndarray, max_px: int = HOLE_FILL_MAX_PX,
                max_width_px: float = HOLE_FILL_MAX_WIDTH_PX) -> np.ndarray:
    """Fill (in place) the small, thin holes - a surface but no source - from their edges; ``source`` = -2 there.
    Returns the filled mask."""
    import cv2
    hole = ((source == -1) & (label != LABEL_NONE)).astype(np.uint8)
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
        fm = filled[y0:y1, x0:x1].astype(np.uint8)
        if rgb.dtype == np.uint8:
            sub = cv2.inpaint(np.ascontiguousarray(rgb[y0:y1, x0:x1]), fm, 5, cv2.INPAINT_TELEA)
        else:                                                # float: one channel at a time
            sub = np.dstack([cv2.inpaint(np.ascontiguousarray(rgb[y0:y1, x0:x1, k]), fm, 5, cv2.INPAINT_TELEA) for k in range(3)])
        rgb[y0:y1, x0:x1][filled[y0:y1, x0:x1]] = sub[filled[y0:y1, x0:x1]]
        source[filled] = -2
    return filled


def provenance_png(source: np.ndarray, out_png: PathLike, n_sources: int) -> Path:
    """The source index per pixel as colours (golden-angle hues), black = no data."""
    import cv2
    img = np.zeros(source.shape + (3,), np.uint8)
    img[source == -2] = (128, 128, 128)                       # cracks filled from their edges
    img[source == -3] = (230, 200, 160)                       # the sky filled above the frames (BGR: light blue)
    img[source == -4] = (90, 120, 150)                        # the ground filled as plain terrain (brown)
    img[source == -5] = (40, 60, 110)                         # 0.97.0: the brown fill where the surface check left no source
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


LINEAR_TOP = 65535.0 / 64.0      # the 16-bit ceiling in 8-bit units
SAT_DOWNWEIGHT = 0.98            # a saturated source pixel keeps this much less weight in the blend
STACK_NAME = "360_3DTD"              # .png and .jpg: the left eye on top of the right (Christian, 10 Nov: "put the left view on
#                                      top"); 0.97.0 (Christian, 12 Nov 2026: "remove '_ods_stereo_square' and add '_360_3DTD' to every top
#                                      down 360 image product so that meta knows how to project it"): <prefix>_360_3DTD.jpg, was
#                                      <prefix>_ods_stereo_square.jpg
OLD_STACK_NAME = "ods_stereo_square"
OLD_NAMES = ("ods_left.png", "ods_right.png", "ods_stereo_square.png", "ods_stereo_square.jpg")


def ods_prefix(station_label: str, rover: bool = False) -> str:
    """The file-name prefix of a station's panoramas (Christian, 10 Nov: "name the ods with a site and drive
    prefix"): ``S003D2046`` / ``S003D2046_rover``."""
    return f"{station_label}_rover" if rover else str(station_label)


def ods_file(out_dir: PathLike, kind: str, prefix: str = "", ext: str = ".png") -> Path:
    """``<prefix>_ods_<kind><ext>`` (kind left / right / centre; the top-down square: :func:`stack_file`); ``ods_<kind><ext>`` without a prefix."""
    return Path(out_dir) / f"{(prefix + '_') if prefix else ''}ods_{kind}{ext}"


def stack_file(out_dir: PathLike, prefix: str = "", ext: str = ".png", suffix: str = "") -> Path:
    """The top-down stereo 360 (left eye above the right): ``<prefix>_360_3DTD<suffix><ext>`` (``suffix`` "_16": the
    16-bit linear one); ``360_3DTD...`` without a prefix."""
    return Path(out_dir) / f"{(prefix + '_') if prefix else ''}{STACK_NAME}{suffix}{ext}"


def drop_old_stacks(out_dir: PathLike, prefix: str = "") -> List[str]:
    """Removes the same station's squares under the old name (``<prefix>_ods_stereo_square[_16].png/.jpg``), so a
    folder made again holds one top-down image per kind.  Returns the names removed."""
    gone = []
    for suffix, ext in (("", ".png"), ("", ".jpg"), ("_16", ".png")):
        f = Path(out_dir) / f"{(prefix + '_') if prefix else ''}{OLD_STACK_NAME}{suffix}{ext}"
        if f.is_file():
            f.unlink()
            gone.append(f.name)
    return gone


def find_prefix(out_dir: PathLike) -> str:
    """The prefix of the newest ``*ods_left.png`` in ``out_dir`` ('' for the unprefixed name of before 10 Nov)."""
    c = sorted(Path(out_dir).glob("*ods_left.png"), key=lambda f: f.stat().st_mtime)
    if not c:
        return ""
    n = c[-1].name[:-len("ods_left.png")]
    return n[:-1] if n.endswith("_") else n


def stack_eyes(out_dir: PathLike, top: str = "left", jpeg_quality: int = 95, prefix: Optional[str] = None,
               square_prefix: Optional[str] = None, suffix: str = "") -> Dict[str, Any]:
    """``<prefix>_360_3DTD.png`` / ``.jpg`` (0.97.0; ``_ods_stereo_square`` before): the two eye images stacked into one square (W x W) - ``top``
    eye above (Christian, 10 Nov: "make a version with the ods_right.png stacked on top of the ods_left so that it is
    a square that I can upload").  ``prefix`` None: that of the eye images in ``out_dir``; ``square_prefix`` (default
    ``prefix``): the square's, e.g. ``S003D2046`` for eye images still named without one.  Returns {"png", "jpg",
    "size", "top"}."""
    import cv2
    out = Path(out_dir)
    if prefix is None:
        prefix = find_prefix(out)
    other = "left" if top == "right" else "right"     # 10 Nov: left on top by default (was right)
    if suffix:                                           # "_16": the 16-bit linear eye images, a 16-bit PNG only
        a = cv2.imread(str(ods_file(out, top + suffix, prefix)), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(ods_file(out, other + suffix, prefix)), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or a.shape != b.shape:
            raise FileNotFoundError(f"{out}: the two {suffix} eye images are needed, of one size")
        sp = prefix if square_prefix is None else square_prefix
        png = stack_file(out, sp, ".png", suffix)
        sq = np.vstack([a, b])
        cv2.imwrite(str(png), sq)
        drop_old_stacks(out, sp)
        return {"png": str(png), "size": [int(sq.shape[1]), int(sq.shape[0])], "top": top, "bits": 16}
    a, b = cv2.imread(str(ods_file(out, top, prefix))), cv2.imread(str(ods_file(out, other, prefix)))
    if a is None or b is None:
        raise FileNotFoundError(f"{out}: {ods_file(out, 'left', prefix).name} and {ods_file(out, 'right', prefix).name} "
                                f"are both needed")
    if a.shape != b.shape:
        raise ValueError(f"{out}: the eye images differ in size ({a.shape} vs {b.shape})")
    sq = np.vstack([a, b])
    sp = prefix if square_prefix is None else square_prefix
    png, jpg = stack_file(out, sp), stack_file(out, sp, ".jpg")
    cv2.imwrite(str(png), sq)
    cv2.imwrite(str(jpg), sq, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    drop_old_stacks(out, sp)
    return {"png": str(png), "jpg": str(jpg), "size": [int(sq.shape[1]), int(sq.shape[0])], "top": top}


DEPTH16_UNITS_PER_M = 500.0      # <prefix>_ods_centre_depth_16.png: uint16 = range (m) x this (2 mm steps, to 131 m)


def write_centre_depth(out_dir: PathLike, prefix: str, rng: np.ndarray, label: np.ndarray) -> Dict[str, Any]:
    """0.97.0 (Christian, 12 Nov 2026: "also make a version projected from the center rotation with a depth map"): the
    depth of the centre panorama - the range from the pivot along each pixel's ray, on the panorama's own grid -
    as ``<prefix>_ods_centre_depth.npz`` (float32 metres ``range``, uint8 ``label``; 0 = no surface, the sky inf),
    ``_depth_16.png`` (uint16 = metres x DEPTH16_UNITS_PER_M; 0 = no surface, 65535 = the sky or beyond 131 m) and the
    pictures ``_depth_linear.jpg`` / ``_depth_log.jpg`` (:func:`mppp.ibr.raycast.depth_rgb`)."""
    import cv2
    from .raycast import JPEG_QUALITY, SKY_RANGE_M, depth_rgb
    out = Path(out_dir)
    r = np.asarray(rng, np.float32).copy()
    lab = np.asarray(label, np.uint8)
    sky = (lab == LABEL_FAR) | (r >= 0.5 * SKY_RANGE_M)
    r[sky] = np.inf
    r[lab == 0] = 0.0
    files = {"npz": ods_file(out, "centre_depth", prefix, ".npz"), "png16": ods_file(out, "centre_depth_16", prefix),
             "linear": ods_file(out, "centre_depth_linear", prefix, ".jpg"), "log": ods_file(out, "centre_depth_log", prefix, ".jpg")}
    np.savez_compressed(files["npz"], range=r, label=lab)
    d16 = np.where(np.isfinite(r), np.clip(np.rint(r * DEPTH16_UNITS_PER_M), 0, 65535), 65535).astype(np.uint16)
    cv2.imwrite(str(files["png16"]), d16)
    for mode in ("linear", "log"):
        cv2.imwrite(str(files[mode]), depth_rgb(np.where(np.isfinite(r), r, 0), lab, mode), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    ok = np.isfinite(r) & (r > 0)
    return {"files": {k: str(v) for k, v in files.items()}, "units_per_m_16bit": DEPTH16_UNITS_PER_M,
            "encoding": "range from the pivot along the pixel's ray (m); npz: 0 none, inf sky; 16-bit png: 0 none, 65535 sky",
            "median_m": float(np.median(r[ok])) if ok.any() else None, "surface_share": float(ok.mean())}


def render_station(project, rec, station, scene: Scene, out_dir: PathLike, width: int = DEFAULT_WIDTH, ipd_m: float = DEFAULT_IPD_M,
                   families: str = "N", scale: float = 1.0, zcam_scale: float = 1.0, images: Optional[Sequence[str]] = None,
                   rows_per_chunk: int = 128, eyes: Sequence[str] = ("left", "right"), use_masks: bool = True,
                   gains: bool = True, stack_top: str = "left", sky_fill: bool = True,
                   mask_head_hardware: bool = True, azimuth0_deg: float = 0.0, prefix: Optional[str] = None,
                   bits: int = 8, pds: Optional[PathLike] = None, tone: str = "auto", exposure16: Union[float, str] = "auto",
                   ground_fill: bool = True, mask_mast: bool = True,
                   tone_levels_file: Optional[PathLike] = None, centre_depth: bool = True, pole_merge_deg="default") -> Dict[str, Any]:
    """The two eye images of a station -> ``out_dir/<prefix>_ods_left.png``, ``<prefix>_ods_right.png``,
    ``<prefix>_360_3DTD.png/.jpg``, ``provenance.png`` (left eye's sources), ``ods.json``.  ``prefix`` None:
    :func:`ods_prefix` (``S003D2046`` / ``S003D2046_rover``); '' for the bare names.  Returns the record."""
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = ods_prefix(station.label, rover=not use_masks)
    height = width // 2
    lin, lin_rep = None, None
    if int(bits) == 16:                              # 0.96.1: the frames as 16-bit linear images (mppp.ibr.linear16)
        from .linear16 import ensure_linear16, files as _lin_files
        by = {im.name for im in rec.images.values() if im.has_pose}
        want = [n for n in (list(images) if images else [n for n in station.images if n[:1].upper() in set(families.upper())])
                if n in by]
        lin_rep = ensure_linear16(project, want, pds=pds)
        if lin_rep["missing"]:
            for n, why in list(lin_rep["missing"].items())[:5]:
                log.warning("ods --bits 16: %s: %s", n, why)
            log.warning("ods --bits 16: %d of %d frames have no 16-bit image - THE PANORAMA IS RENDERED IN 8 BITS",
                        len(lin_rep["missing"]), len(want))
        else:
            lin = {n: _lin_files(project, n) for n in want}
            log.info("ods: 16-bit linear sources (%d frames, %d made now) from %s", len(want), lin_rep["made"], lin_rep["dir"])
    linear = lin is not None
    mast_pose = None
    if not use_masks and mask_mast:                  # the rover in the block, for the mast's mask
        try:
            from .rover_deck import rover_pose
            mast_pose = rover_pose(project, rec, station.images)
        except Exception as e:                       # noqa: BLE001 - the panorama without that mask
            log.warning("ods: no rover pose for the mast mask (%s)", e)
    sources = station_sources(project, rec, station, families=families, scale=scale, zcam_scale=zcam_scale, images=images,
                              use_masks=use_masks, mask_head_hardware=mask_head_hardware, linear16=lin,
                              mast_pose=mast_pose)
    if not sources:
        raise ValueError(f"station {station.label}: no source images of families {families!r} with a pose")
    axes = dict(up=station.up, north=station.north, east=station.east)
    t0 = time.time()

    def progress(eye, r, h, dt):
        log.info("ods %s: %d / %d rows (%.0f s)", eye, r, h, dt)

    rec_out: Dict[str, Any] = {"station": station.to_json(), "prefix": prefix, "width": width, "height": height, "ipd_m": ipd_m,
                               "pole_merge_deg": (list(POLE_MERGE_DEG) if POLE_MERGE_DEG else None) if isinstance(pole_merge_deg, str) else (list(pole_merge_deg) if pole_merge_deg else None),
                               "sources": [s.name for s in sources], "scale": scale, "zcam_scale": zcam_scale,
                               "rover": "shown" if not use_masks else "masked out",
                               "azimuth_of_column_0_deg": float(azimuth0_deg) % 360.0,
                               "head_hardware_masked": [s.name for s in sources if s.extra_mask is not None and s.extra_mask.any()],
                               "mast_masked": [s.name for s in sources if getattr(s, "mast_masked", False)],
                               "eyes": {}}
    if gains:
        g = estimate_gains(scene, sources, station.pivot, width=min(GAIN_PASS_WIDTH, width), rows_per_chunk=rows_per_chunk, **axes)
        rec_out["gains"] = g
        log.info("ods: gains over %d source pairs: %s", g["pairs"], ", ".join(f"{v:.3f}" for v in g["gains"].values()))
    levels, levels_from, depth_rec = None, None, None
    for eye in eyes:
        res = render_eye(scene, sources, station.pivot, eye, width, height, ipd_m, pole_merge_deg=pole_merge_deg, rows_per_chunk=rows_per_chunk,
                         progress=progress, sky_fill=sky_fill, azimuth0_deg=azimuth0_deg, linear=linear,
                         ground_fill=ground_fill, **axes)
        f = ods_file(out, eye, prefix)
        if linear:                                   # the 16-bit linear panorama, and the 8-bit one for viewing
            f16 = ods_file(out, eye + "_16", prefix)
            if isinstance(exposure16, str):          # 0.97.0 "auto": one gain per station, shared by its variants
                from .linear16 import auto_exposure16, shared_value
                exposure16, _src = shared_value(tone_levels_file, "exposure16",
                                                lambda: auto_exposure16(res["rgb"], res["source"] >= 0))
                exposure16 = float(exposure16)
                log.info("ods: 16-bit exposure %.3f of the frames' scale (%s)", exposure16, _src)
            k16 = 64.0 * float(exposure16)
            v16 = np.rint(res["rgb"][..., ::-1] * k16)
            clipped16 = float((v16.max(axis=2) > 65535).mean())
            cv2.imwrite(str(f16), np.clip(v16, 0, 65535).astype(np.uint16))
            del v16
            from .linear16 import auto_levels, tone_levels, tone_soft
            if tone == "auto":                       # the black and white points from the first eye, for both
                if levels is None:                   # 0.97.0: or from the file the other variant wrote
                    from .linear16 import shared_levels
                    levels, levels_from = shared_levels(tone_levels_file,
                                                        lambda: auto_levels(res["rgb"], res["source"] >= 0))
                    log.info("ods: 8-bit view levels %.1f / %.1f (%s%s)", levels[0], levels[1], levels_from,
                             f": {tone_levels_file}" if tone_levels_file else "")
                view = tone_levels(res["rgb"], *levels)
            else:
                view = tone_soft(res["rgb"]) if tone == "soft" else np.clip(res["rgb"], 0, 255)
            over = float((res["rgb"].max(axis=2) > 255.0).mean())
            cv2.imwrite(str(f), np.rint(view[..., ::-1]).astype(np.uint8))
        else:
            cv2.imwrite(str(f), res["rgb"][..., ::-1])
        if eye in ("centre", "center") and centre_depth:     # 0.97.0: the panorama from the pivot, with its depth
            depth_rec = write_centre_depth(out, prefix, res["range"], res["label"])
        if eye == eyes[0]:
            provenance_png(res["source"], out / "provenance.png", len(sources))
        lab = res["label"]
        n = lab.size
        rec_out["eyes"][eye] = {"file": str(f), "coverage": {"coloured": float((res["source"] >= 0).sum() / n),
                                "mesh": float((lab == LABEL_MESH).sum() / n), "plane": float((lab == LABEL_PLANE).sum() / n),
                                "far": float((lab == LABEL_FAR).sum() / n),
                                "no_surface": float((lab == 0).sum() / n),
                                "hidden_view_used": float(res["hidden_fallback"].sum() / n),
                                "brown_filled": float(res["brown_filled"].sum() / n),
                                "brown_filled_on_rover": float((res["brown_filled"] & res["rover"]).sum() / n),
                                "surface_check_rejected_samples": {"rover_pixels_off_terrain": int(res["surface_check_rejected"][0]),
                                                                   "terrain_pixels_off_rover": int(res["surface_check_rejected"][1])},
                                "cracks_filled": float(res["filled"].sum() / n),
                                "sky_filled": float(res["sky_filled"].sum() / n),
                                "ground_filled": float(res["ground_filled"].sum() / n)},
                                "sources_used": int(len(np.unique(res["source"][res["source"] >= 0])))}
        if linear:
            rec_out["eyes"][eye].update(file_16bit=str(f16), above_8bit_range=over, clipped_in_16bit=clipped16)
        for s in sources:
            s.release()
    if "left" in eyes and "right" in eyes:
        rec_out["stacked"] = stack_eyes(out, top=stack_top, prefix=prefix)
        if linear:
            rec_out["stacked_16bit"] = stack_eyes(out, top=stack_top, prefix=prefix, suffix="_16")
            rec_out.setdefault("linear16", {})["exposure16"] = float(exposure16)
            (out / "ods.json").write_text(json.dumps(rec_out, indent=1, default=str), encoding="utf-8")   # write_enhanced reads the exposure
            from .linear16 import write_enhanced
            rec_out["enhanced"] = write_enhanced(out, prefix, top=stack_top, levels_file=tone_levels_file)
    if prefix:                                       # the unprefixed files of before 10 Nov would be stale
        for n in OLD_NAMES:
            if (out / n).is_file():
                (out / n).unlink()
                log.info("ods: removed the stale %s (now %s_%s)", n, prefix, n)
    rec_out["prefix"] = prefix
    if depth_rec:
        rec_out["centre_depth"] = depth_rec
    rec_out["view_levels_from"] = levels_from
    rec_out["tone_levels_file"] = str(tone_levels_file) if tone_levels_file else None
    rec_out["bits"] = 16 if linear else 8
    if int(bits) == 16:
        fr = [v for v in (lin_rep.get("frames") or {}).values() if v]
        rec_out["linear16"] = {
            "rendered": linear, "missing": lin_rep["missing"], "dir": lin_rep["dir"],
            "encoding": "uint16 linear, no gamma: value = radiance x white balance x brightness / zenith scale x "
                        "scale_rad_to_int16 x the frame's gain (gains in this file); value / 64 = the 8-bit panorama "
                        "before its tone curve; 0 = no data",
            "white_balance": fr[0]["white_balance"] if fr else None, "brightness": fr[0]["brightness"] if fr else None,
            "scale_rad_to_int16": fr[0]["scale_rad_to_int16"] if fr else None,
            "view_tone": tone if linear else None,
            "view_levels_8bit_units": None if levels is None else [float(v) for v in levels],
            "exposure16": None if isinstance(exposure16, str) else float(exposure16),
            "value_16bit": "the linear value x 64 x exposure16 (auto: the gain that puts the 99.99 % point of the brightest channel at 95 % of the range, 0.5-4)",
            "frames_saturated_share_max": max([f["saturated_share"] for f in fr], default=None),
            "frames_clipped_in_8bit_share_max": max([f["clipped_in_8bit_share"] for f in fr], default=None)}
    rec_out["seconds"] = round(time.time() - t0, 1)
    (out / "ods.json").write_text(json.dumps(rec_out, indent=1, default=float), encoding="utf-8")
    log.info("ods: %s -> %s (%.0f s)", station.label, out, rec_out["seconds"])
    return rec_out
