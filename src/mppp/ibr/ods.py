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

DEFAULT_WIDTH = 8192              # 0.97.10 (Christian, 13 Nov 2026: "let's make 8k ods standard for now"); 16K: --width 16384
#                                   (0.97.4 to 0.97.9: 16384)
DEFAULT_IPD_M = 0.064
# 0.97.0 (Christian, 12 Nov 2026, Quest 3: "when I tilt my head down ... I go cross-eyed"): pole merging.  An ODS
# pair is right only at the horizon: towards a pole each column keeps its own eye offset, so in a view looking down
# the columns to the side carry their disparity along the wrong direction for the viewer's eyes (vertical
# misalignment), and near surfaces (the deck 1 m below) carry far too much of it.  The eye separation is full to
# POLE_MERGE_DEG[0] of elevation (up or down), falls as a cosine and is zero from POLE_MERGE_DEG[1]: one image
# for both eyes at the poles.  None: no merging (the panoramas before 12 Nov).
POLE_MERGE_DEG: Optional[Tuple[float, float]] = (0.0, 90.0)       # 0.97.13: cos(elevation); (30, 80) from 0.97.0
# 0.97.1 (Christian, 12 Nov, Quest 3: "the pole merging helps a lot, but not quite enough. maybe the cosine fall off
# should be from the horizon to the -90 el. The -30 area still feels a little strained"): below the horizon the fall
# starts at the horizon and ends at the nadir - the share is cos(elevation): 0.87 at -30 deg (1 before), 0.71 at -45
# (0.89), 0.5 at -60 (0.59).  Above the horizon POLE_MERGE_DEG as before (sky; nothing near).
# 0.97.5 (Christian, 13 Nov, Quest 3: "when I look down the terrain looks like it dips down ... any way of making the
# terrain in front of the rover look flat while reducing eye strain while looking at the rover deck?"): the cosine
# law is what made the dip - with a share s of the eye separation a point at distance d is seen at d / s, so the
# ground h below the eyes is seen h / cos(elevation) below them: 1.15 h at -30 deg, 1.41 h at -45, 2 h at -60.
# Now the separation is full to -60 deg and merges only towards the nadir, and what strains the eyes is limited
# where it arises - on near surfaces: NEAR_MERGE_M.
# 0.97.13 (Christian, 13 Nov, Quest 3, after 0.97.12: "The rover depth looks too far away relative to the terrain.
# Let's return everything to true disparity and implement a cosine pole merging"): the range-dependent shares are off
# (NEAR_MERGE_M and the range merge: they showed the deck 1.2 m away as if at 2.5 m, next to terrain at its true
# range), and the one share left is cos(elevation), above and below the horizon.  What that does: everything along
# one line of sight is seen 1 / cos(elevation) times farther - the rover and the terrain behind it alike, so their
# depths stay in proportion - and flat ground h below the eyes is seen h / cos(elevation) below them (1.15 h at
# -30 deg, 1.41 h at -45, 2 h at -60: the dip of 0.97.1).  (60, 88) from 0.97.5 to 0.97.12.
POLE_MERGE_DOWN_DEG: Optional[Tuple[float, float]] = (0.0, 90.0)
# The eye separation is full for surfaces from NEAR_MERGE_M[1] on and falls in proportion to the range below that
# (never under the share NEAR_MERGE_MIN): a surface nearer than 2.5 m is shown as if at 2.5 m, so its disparity
# stops at 64 mm / 2.5 m = 1.5 deg (the deck 1 m below the cameras had 3.7 deg).  Terrain at 2.5 m and beyond - with
# the cameras 2 m up, everything more than 1.5 m out on flat ground - keeps its true depth: flat ground looks flat.
NEAR_MERGE_M: Optional[float] = None       # 0.97.13: off (2.5 from 0.97.5 to 0.97.12)
NEAR_MERGE_MIN = 0.2
# 0.97.0 (Christian, 12 Nov 2026: "Fade the eye separation with range past the mesh edge, so the last terrain and the
# sky do not jump in depth"): the eye separation is full to RANGE_MERGE_M[0] (the mesh's reach), falls as a cosine
# and is zero from the sky sphere's radius; the range is that of the ray from the pivot, so both eyes fade alike.
SKY_RADIUS_M: Optional[float] = 300.0       # 0.97.13 (Christian: "Maybe move the 200 m sphere to 300 m"); 200: 13 Nov (Christian: "the sphere at 200 instead of infinity actually helped a
                                            # lot"); None on 12 Nov ("the sky sphere can be at infinity")
RANGE_MERGE_FROM_M = 75.0                   # the eye separation full to the mesh's reach ...
RANGE_MERGE_TO_M = 200.0                    # ... and zero from the ground plane's rim on: the plane blends terrain and sky
RANGE_MERGE = False                         # 0.97.13: off - true disparity at every range (at 300 m it is 0.3 px of 8K)


def range_merge(range_m, to_m: Optional[float] = RANGE_MERGE_TO_M, from_m: float = RANGE_MERGE_FROM_M) -> np.ndarray:
    """The share of the eye separation at ``range_m`` from the pivot (1 near, 0 from ``to_m`` on)."""
    r = np.asarray(range_m, float)
    if not to_m or to_m <= from_m:
        return np.ones_like(r)
    return np.cos(0.5 * np.pi * np.clip((r - from_m) / (to_m - from_m), 0.0, 1.0))


def near_merge(range_m, full_m: Optional[float] = NEAR_MERGE_M, least: float = NEAR_MERGE_MIN) -> np.ndarray:
    """The share of the eye separation for a surface ``range_m`` from the pivot: range / ``full_m`` below ``full_m``
    (not under ``least``), 1 beyond and where the ray hit nothing.  None or 0: 1 everywhere."""
    r = np.asarray(range_m, float)
    if not full_m:
        return np.ones_like(r)
    return np.where(np.isfinite(r) & (r > 0), np.clip(r / float(full_m), least, 1.0), 1.0)


def pole_merge(el_rad, merge_deg=POLE_MERGE_DEG, down_deg="same") -> np.ndarray:
    """The share of the eye separation at elevation ``el_rad`` (1 at the horizon, 0 at the poles).  ``merge_deg``:
    (start, end) above the horizon; ``down_deg``: below it ("same": as above).  None: no merging on that side."""
    e = np.degrees(np.asarray(el_rad, float))
    a = np.abs(e)

    def side(md):
        if not md:
            return np.ones_like(a)
        lo, hi = float(md[0]), float(md[1])
        return np.cos(0.5 * np.pi * np.clip((a - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    up_share = side(merge_deg)
    return up_share if isinstance(down_deg, str) else np.where(e < 0, side(down_deg), up_share)


FEATHER_PX = 40.0                # (0.96.0 prototype) a source's weight fell to zero this close to its image border
MAX_SOURCES_BLENDED = 4
EDGE_MARGIN_PX = 3.0             # a source gives nothing within this many pixels of its border, padding or mask edge
EDGE_MARGIN_FRAC = 0.005         # ... or within this share of its short side, whichever is more.  0.97.17 (Christian, 13
#                                  Nov 2026: "there are still some blemishes in the sky"): a Navcam frame's outermost rows
#                                  are darker than those just inside - measured on NLF_0052_0671564560 (2560 x 1920),
#                                  against rows 10 to 16: 4.7 % at the first row, 1.6 % at the fourth, under 0.5 % from
#                                  the seventh.  With 3 pixels the rows 4 to 6 were used, and where a frame is the only
#                                  one that sees a direction (the top of the coverage, in the sky) they are a darker
#                                  line.  Now 9.6 pixels for that frame.
                                 # (10 Nov: the frames' outermost rows are darker - a faint line where the sky fill starts)
FEATHER_LOW_FRAC = 0.15          # the low band's feather: this fraction of the source's short side
FEATHER_HIGH_FRAC = 0.03         # the detail band's feather
LOW_SIGMA_FRAC = 0.006           # the low band: Gaussian sigma, this fraction of the short side (~6 px at 960)
ANGLE_SCALE_RAD = 0.02           # parallax term 1 / (1 + angle / this)
GAIN_RANGE = (0.6, 1.6)
HOLE_FILL_MAX_PX = 4000          # holes (a surface, no source) up to this many pixels are filled from their edges
HOLE_FILL_MAX_WIDTH_PX = 24      # ... and only if thin: no pixel farther than this from the hole's edge
HOLE_FILL_REF_WIDTH = 8192       # both for a panorama of this width (0.97.7: scaled for a wider one - at 16384 the
#                                  same limits in pixels were half the angle, and cracks filled at 8K stayed open)
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
             up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0), azimuth0_deg: float = 0.0, pole_merge_deg="default", share=None) -> Tuple[np.ndarray, np.ndarray]:
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
    if isinstance(pole_merge_deg, str):                                    # the defaults: up and down apart
        share_ = pole_merge(E, POLE_MERGE_DEG, POLE_MERGE_DOWN_DEG)
    elif isinstance(pole_merge_deg, dict):                                 # {"up": (a, b) | None, "down": (a, b) | None}
        share_ = pole_merge(E, pole_merge_deg.get("up", POLE_MERGE_DEG), pole_merge_deg.get("down", POLE_MERGE_DOWN_DEG))
    else:                                                                  # one pair (or None) for both sides
        share_ = pole_merge(E, pole_merge_deg)
    if share is not None:                                                  # a further share per ray (the range fade)
        share_ = share_ * np.asarray(share, float).reshape(share_.shape)
    o = P + sign * (ipd_m / 2.0) * share_[..., None] * right
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
        self.fill_only = False                            # 0.97.7: colours only what no regular source sees (a Hazcam frame)
        self.context = False                              # 0.97.15: a frame from another place or sol (see station_sources)
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
            img = self.rgb().astype(np.float32)
            pad = self.padding()
            if pad.any():                                        # 0.97.17: the blur over the frame's own pixels only -
                ok = (~pad).astype(np.float32)                   # next to black padding the low band was pulled dark
                den = cv2.GaussianBlur(ok, (0, 0), sig)          # and the detail band (image - low band) had a
                low = cv2.GaussianBlur(img * ok[..., None], (0, 0), sig) / np.maximum(den, 1e-3)[..., None]   # bright rim
                self._low = np.where(pad[..., None], img, low).astype(np.float32)
            else:
                self._low = cv2.GaussianBlur(img, (0, 0), sig)
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


FILL_WITHIN_S = 7200.0            # a fill frame is taken within this of one of the panorama's own frames


def _sclk(name: str) -> Optional[float]:
    """The spacecraft clock (s) in a frame's name (``NLF_0461_0707871340_...``)."""
    t = Path(str(name)).name[9:19]
    return float(t) if t.isdigit() else None


def fill_frames(project, rec, station, families: str, names: Sequence[str]) -> List[str]:
    """The station's registered frames of the camera ``families`` (letters; "F": the front Hazcams) for filling in
    what the panorama's own frames ``names`` do not see - those taken within FILL_WITHIN_S of one of ``names``."""
    from .station import station_images
    if not families:
        return []
    try:
        _, rows = station_images(project.images, station.label, families)
    except ValueError:
        return []
    by_name = {im.name: im for im in rec.images.values()}
    have = [r["name"] for r in rows if r["name"] in by_name and by_name[r["name"]].has_pose and r["name"] not in set(names)]
    # 0.97.11 (Christian, 13 Nov 2026: "only include them if they are taken within 2 hours of the primary ncam"):
    # by the spacecraft clock in the names; before, any frame of the same sol - or of any sol when none was
    t0 = [t for t in (_sclk(n) for n in names) if t is not None]
    out = []
    for n in have:
        t = _sclk(n)
        if t is not None and t0 and min(abs(t - x) for x in t0) <= FILL_WITHIN_S:
            out.append(n)
    if have and not out:
        log.info("fill frames: %d of the camera families %r at this station, none within %.0f h of the panorama's frames",
                 len(have), families, FILL_WITHIN_S / 3600.0)
    return out


def station_sources(project, rec, station, families: str = "N", scale: float = 1.0, zcam_scale: float = 1.0,
                    images: Optional[Sequence[str]] = None, use_masks: bool = True,
                    mask_head_hardware: bool = True, linear16: Optional[Dict[str, Any]] = None,
                    mast_pose: Optional[Dict[str, Any]] = None, fill_images: Sequence[str] = (),
                    context_images: Sequence[str] = ()) -> List[SourceImage]:
    """The station's frames as sources; ``use_masks=False`` (``mppp ods --rover``) lets the rover's own pixels in,
    except the hardware fixed to the mast head (:func:`head_hardware_masks`) unless ``mask_head_hardware`` is off.

    ``context_images`` (0.97.15; Christian, 13 Nov 2026: "include NLF_1790_... from a near location for context.
    Since it is from a different location, it should be masked in both the with and without rover parts"): frames of
    the block taken from another place, or at this place on another sol.  They are fill-only sources (they colour
    what the station's own frames do not see), their rover mask is always applied - the rover in them is somewhere
    else, or in another state - and they never colour the rover model."""
    by_name = {im.name: im for im in rec.images.values()}
    names = list(images) if images else [n for n in station.images if n[:1].upper() in set(families.upper())]
    ctx = list(dict.fromkeys(context_images or ()))           # a context frame at this station is not one of its own
    names = [n for n in names if n not in set(ctx)]
    fill_names = [n for n in fill_images if n not in set(names) and n not in set(ctx)] + ctx      # 0.97.7: fill-only sources
    out = []
    for name in names + fill_names:
        im = by_name.get(name)
        if im is None or not im.has_pose:
            continue
        cw = im.cam_from_world()
        s = zcam_scale if name[:1].upper() == "Z" else scale
        mf = Path(project.root) / "masks" / (name + ".png")
        out.append(SourceImage(name, rec.cameras[im.camera_id], np.asarray(cw.rotation.matrix(), float),
                               np.asarray(im.projection_center(), float), Path(project.root) / "images" / name,
                               mf if ((use_masks or name in ctx) and mf.is_file()) else None, s))
        if not use_masks and mf.is_file() and name not in ctx:      # rover shown: the mask still says which pixels are rover
            out[-1].rover_mask_file = mf
        out[-1].fill_only = name in fill_names
        out[-1].context = name in ctx
        if name in ctx and not mf.is_file():
            log.warning("ods: the context frame %s has no rover mask - its rover is not masked", name)
        if linear16 is not None:
            out[-1].linear_file, out[-1].sat_file = linear16[name]
    if not use_masks and mask_head_hardware and out:
        import cv2
        ms = []
        own = [s_ for s_ in out if not s_.context]            # the context frames are masked already
        for src in own:
            mf = Path(project.root) / "masks" / (src.name + ".png")
            m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) if mf.is_file() else None
            if m is not None and (m.shape[1] != src.W or m.shape[0] != src.H):
                m = cv2.resize(m, (src.W, src.H), interpolation=cv2.INTER_NEAREST)
            ms.append(None if m is None else m > 0)
        hw = head_hardware_masks(own, ms, station.up)
        for src, h in zip(own, hw):
            src.extra_mask = h
        n = sum(1 for h in hw if h is not None and h.any())
        log.info("ods: mast-head hardware masked in %d of %d frames (rover shown)", n, len(own))
    if not use_masks and mast_pose is not None and out:      # 0.96.1: the mast and its wind-sensor booms, from the model
        from .rover_model import available, mast_masks
        if available():
            own = [s_ for s_ in out if not s_.context]
            mm = mast_masks(own, mast_pose)
            for src, m in zip(own, mm):
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


# 0.97.8 (Christian, 13 Nov 2026, Van Zyl with the rover: "the 3d model rover looks just as bad"): on a surface this
# near, a frame is weighted by how little parallax it has against the panorama's ray, far more steeply than on the
# terrain, and its feather is the narrow one.  A frame whose camera lies on the panorama's ray shows exactly what the
# panorama should show there, whatever the model's shape; the error on the picture is (camera's distance from the
# ray) x (the model's depth error) / range^2 - at 1.2 m, 20 cm off the ray and 5 cm of model error are 23 pixels of a
# 16K panorama.  The Navcam frames are 4 to 21 cm off the ray depending on where in the frame the surface is; the
# wide feather chose the frame's middle, which is the worst place.  None: the same weights everywhere, as before.
# 0.97.13 (Christian, 13 Nov, after 0.97.12: "The image blending could be smoother. The provenance should be a
# smoother function when multiple images see the same surfaces. the mid and far distances terrain looks well
# blended"): 0.97.8 gave near surfaces the narrow feather and the 4th power in both bands, switched on at 3 m: one
# frame took nearly everything, with a hard change of frame where two were equally far from the ray, and a line on
# the ground at 3 m.  Now the feathers are the terrain's everywhere (wide for the colour, narrow for the detail), and
# only the parallax term's power rises on near surfaces, gradually: from the terrain's 1 (colour) and 3 (detail) at
# NEAR_BLEND_M[1] to NEAR_LOW_POWER and NEAR_PARALLAX_POWER at NEAR_BLEND_M[0] and nearer.  The colour of a near
# surface is again a blend of the frames that see it; its detail still comes mostly from the frame nearest the ray.
NEAR_BLEND_M: Optional[Tuple[float, float]] = (2.0, 4.0)
TERRAIN_DETAIL_POWER = 2.0        # 0.98.0 (Christian, 14 Nov 2026: "When blending the images, make sure to preserve the noise
#                                   canceling advantage of multiple frames seeing the same terrain"): the detail band's
#                                   power on the parallax term for terrain.  3.0 until 0.97.24: with it the frame nearest
#                                   the eye's ray took most of the weight inside about 20 m (two frames 0.4 m apart seen
#                                   from 5 m: weights 1 : 0.11, 1.2 frames' worth; with 2.0: 1 : 0.23, 1.45), so the
#                                   noise of one frame came through.  The cost of a lower power is a softer detail band
#                                   where the mesh is off.  ods.json records the frames' worth per pixel (effective_frames).
NEAR_LOW_POWER = 2.0
NEAR_PARALLAX_POWER = 4.0


def near_share(range_m) -> np.ndarray:
    """How near a surface at ``range_m`` is for the blend: 1 up to NEAR_BLEND_M[0], 0 from NEAR_BLEND_M[1] on."""
    r = np.asarray(range_m, float)
    if not NEAR_BLEND_M:
        return np.zeros_like(r, dtype=np.float32)
    a, b = float(NEAR_BLEND_M[0]), float(NEAR_BLEND_M[1])
    return (1.0 - _smoothstep((np.where(np.isfinite(r), r, b) - a) / max(b - a, 1e-9))).astype(np.float32)


FILL_RANK = 1e-3                  # a fill-only source's rank among the candidates of a pixel (x its weight)


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
    reg_max = np.zeros(n, np.float32)                        # 0.97.7: the strongest regular source's feather weight
    near = np.where(ok, near_share(hit["t"]), 0.0).astype(np.float32)      # 0.97.8 / 0.97.13: see NEAR_BLEND_M
    for k, src in enumerate(sources):
        uv, z, inside = src.project(X)
        cand = inside & ok
        if not cand.any():
            continue
        ij = cand.nonzero()[0]
        dist = src.sample_edge_distance(uv[ij])
        keep = dist > max(EDGE_MARGIN_PX, EDGE_MARGIN_FRAC * src.short_side)   # inside the image, off the padding and the mask
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
        if on_rover is not None and getattr(src, "context", False) and len(ij):
            keep = ~on_rover[ij]                             # 0.97.15: a context frame never colours the rover model
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
        feather = _smoothstep(dist / (FEATHER_LOW_FRAC * src.short_side)).astype(np.float32)
        w = (feather * aterm ** (1.0 + near[ij] * (NEAR_LOW_POWER - 1.0))).astype(np.float32) + 1e-6   # near: steeper
        if getattr(src, "fill_only", False):
            w = w * FILL_RANK                                # ranked after every regular source
        else:
            reg_max[ij] = np.maximum(reg_max[ij], feather)
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
    # 0.97.7 (Christian, 13 Nov 2026: "add the fhaz cam image pair taken with the navcams to cover the terrain under
    # the arm"): a fill-only source colours what the regular ones do not see, and comes in as their feather goes out
    fill = np.array([bool(getattr(s_, "fill_only", False)) for s_ in sources] + [False])
    fw = np.ones((n, K), np.float32)
    if fill.any():
        isf = fill[best["k"]]                                # (k = -1 reads the trailing False)
        fw = np.where(isf, ((1.0 - reg_max) ** 2)[:, None], 1.0).astype(np.float32)
    best["fill_weight"] = fw
    best["near"] = near
    best.update(label=lab, t=hit["t"], ok=ok, hidden_fallback=use, rejected=rejected, rejected_px=rej_px,
                on_rover=on_rover if on_rover is not None else np.zeros(n, bool))
    return best


# 0.97.13 (Christian, 13 Nov 2026: "we might need sky sphere at both 100 m, 200m and infinity and blend between them
# favoring the sharpest one" ... "move the 200 m sphere to 300 m. and pay attention to the blending between the 100,
# 300, and inf m spheres"): THE FAR FIELD.  What lies beyond the mesh and the ground plane - hills from 100 m to
# kilometres, the sky - was painted on one sphere.  The frames' cameras are up to 0.4 m apart, so where two frames
# overlap, a surface drawn at the wrong range is drawn twice: 0.4 m / 100 m against 0.4 m / infinity is 4 mrad, 5
# pixels of an 8K panorama.  Before the eyes are rendered, the far directions are tried on each of FAR_RADII_M (the
# sky sphere's radius is one of them; None = infinity): the frames that would be blended there are sampled, and the
# cost of a radius is the spread of their detail bands (weighted as in the blend; of two frames: half the squared
# difference), summed over a window of FAR_AGG_DEG.  The sharpest radius is the one with the least cost.  The
# surface is one smooth field of 1 / radius per direction, not three shells with seams: a soft minimum over the
# radii; where the costs do not tell the radii apart (one frame only, or sky without texture: FAR_DECISIVE) it
# is the decided directions around (FAR_SPREAD_DEG), or the sky sphere's 1 / radius when there are none; smoothed
# over FAR_SMOOTH_DEG, and equal to the sky sphere's at the ground
# plane's rim, released over FAR_PIN_DEG above it - so plane and far field meet without a step in depth.
FAR_FIELD = True
FAR_RADII_M: Tuple[Optional[float], ...] = (100.0, None)     # with the sky sphere's radius: 100 m, 300 m, infinity
FAR_EL_DEG = (-6.0, 30.0)        # the elevations tried (above: the sky sphere's radius held from the last row)
FAR_WIDTH = 4096                 # on a panorama at most this wide (and half the output's width)
FAR_AGG_DEG = 1.0                # the costs are summed over +/- this
FAR_SMOOTH_DEG = 1.5             # the field's smoothing (Gaussian sigma)
FAR_SPREAD_DEG = 8.0             # undecided directions take the decided ones within about this (Gaussian sigma) ...
FAR_SPREAD_MIN = 0.02            # ... in full when the decided share around them is at least this
FAR_NOISE = 1.0                  # the detail band's noise (8-bit units squared): costs this close are the same
FAR_DECISIVE = 0.3               # (worst - best) / worst cost from which the choice counts in full
FAR_PIN_FROM_DEG = -0.5          # the sky sphere's radius up to this elevation (the plane's rim is at about -0.4 deg) ...
FAR_PIN_DEG = 2.5                # ... and the field's own value from this much above it


def estimate_far_field(scene: Scene, sources: Sequence[SourceImage], pivot: Sequence[float], width: int,
                       up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0), rows_per_chunk: int = 128,
                       radii: Optional[Sequence[Optional[float]]] = None) -> Optional[Dict[str, Any]]:
    """The far field of the station (see FAR_FIELD): sets ``scene.far_field`` (read by :meth:`Scene.cast`) and
    returns its record {"radii_m", "width", "rows", "share_nearest": {radius: share of the decided directions},
    "decided_share", "median_radius_m", ...}; None (and no field) without a sky sphere or a far direction that
    two frames see."""
    import cv2
    prior_r = getattr(scene, "sky_radius_m", None)
    if not prior_r or getattr(scene, "plane", None) is None:
        return None
    t0 = time.time()
    if radii is None:
        radii = sorted({float(prior_r)} | {float(r) for r in FAR_RADII_M if r}) + ([None] if None in FAR_RADII_M else [])
    w = int(min(FAR_WIDTH, max(256, width // 2)))
    hf = w // 2
    r_top = int(np.clip(np.floor((90.0 - FAR_EL_DEG[1]) / 180.0 * hf), 0, hf - 1))
    r_bot = int(np.clip(np.ceil((90.0 - FAR_EL_DEG[0]) / 180.0 * hf), r_top + 1, hf))
    h = r_bot - r_top
    cost = np.zeros((len(radii), h, w), np.float32)
    val = np.zeros((len(radii), h, w), np.float32)
    far = np.zeros((h, w), bool)
    keep_r, keep_f = scene.sky_radius_m, getattr(scene, "far_field", None)
    try:
        scene.far_field = None
        for i, R in enumerate(radii):
            scene.sky_radius_m = R
            for r0 in range(r_top, r_bot, rows_per_chunk):
                r1 = min(r_bot, r0 + rows_per_chunk)
                o, d = ods_rays(w, hf, (r0, r1), pivot, 0.0, "centre", up, north, east)
                sub = np.flatnonzero(scene.cast(o, d)["label"] == LABEL_FAR)
                if i == 0:
                    far[r0 - r_top:r1 - r_top].reshape(-1)[sub] = True
                if not len(sub):
                    continue
                c = candidates(scene, sources, o[sub], d[sub], surface_check=False)
                n = len(sub)
                s0, s1, s2, sw2 = (np.zeros(n, np.float64) for _ in range(4))
                for slot in range(c["k"].shape[1]):
                    ks = c["k"][:, slot]
                    for k in np.unique(ks[ks >= 0]):
                        sel = np.flatnonzero(ks == k)
                        src = sources[k]
                        lo, full = src.sample_bands(c["uv"][sel, slot])
                        hi = src.gain * (full - lo).mean(axis=1)
                        wt = (_smoothstep(c["dist"][sel, slot] / (FEATHER_HIGH_FRAC * src.short_side)) * c["ang"][sel, slot] ** 3
                              * c["fill_weight"][sel, slot]).astype(np.float64)
                        s0[sel] += wt; s1[sel] += wt * hi; s2[sel] += wt * hi * hi; sw2[sel] += wt * wt
                ok = s0 > 1e-9
                share = np.zeros(n)                                    # 1 - sum w^2 / (sum w)^2: 0 one frame, 1/2 two alike
                share[ok] = 1.0 - sw2[ok] / s0[ok] ** 2
                two = share > 0.02
                var = np.zeros(n)
                var[two] = np.maximum(s2[two] / s0[two] - (s1[two] / s0[two]) ** 2, 0.0) / share[two]
                cost[i, r0 - r_top:r1 - r_top].reshape(-1)[sub] = var
                val[i, r0 - r_top:r1 - r_top].reshape(-1)[sub] = np.clip(2.0 * share, 0.0, 1.0)
    finally:
        scene.sky_radius_m, scene.far_field = keep_r, keep_f
    for s_ in sources:
        s_.release()
    v = val.min(axis=0)                                                # seen by two frames at every radius
    if float(v.sum()) < 50:
        return None
    ka = max(1, int(round(FAR_AGG_DEG / 360.0 * w)))
    pad = 4 * max(ka, int(round(FAR_SPREAD_DEG / 360.0 * w))) + 2
    wrap = lambda x: np.concatenate([x[:, -pad:], x, x[:, :pad]], axis=1)                      # noqa: E731
    box = lambda x: cv2.blur(wrap(x), (2 * ka + 1, 2 * ka + 1), borderType=cv2.BORDER_REPLICATE)[:, pad:pad + w]   # noqa: E731
    den = box(v)
    agg = np.stack([box(cost[i] * v) / np.maximum(den, 1e-6) for i in range(len(radii))])
    cmin, cmax = agg.min(axis=0), agg.max(axis=0)
    conf = _smoothstep((cmax - cmin) / (cmax + FAR_NOISE) / FAR_DECISIVE) * np.clip(den / 0.2, 0.0, 1.0)
    inv_r = np.array([0.0 if not R else 1.0 / float(R) for R in radii], np.float32)
    p = np.exp(-(agg - cmin[None]) / (0.1 * cmin[None] + 0.1 * FAR_NOISE))
    est = (p * inv_r[:, None, None]).sum(axis=0) / p.sum(axis=0)
    prior = 1.0 / float(prior_r)
    # where the frames do not tell (the middle of a frame that nobody overlaps, plain sky): the decided directions
    # around, spread over FAR_SPREAD_DEG - not the sky sphere's radius in stripes between the overlaps
    gauss = lambda x, sg_: cv2.GaussianBlur(wrap(x.astype(np.float32)), (0, 0), sg_, borderType=cv2.BORDER_REPLICATE)[:, pad:pad + w]   # noqa: E731
    sw_ = max(1.0, FAR_SPREAD_DEG / 360.0 * w)
    cden = gauss(conf, sw_)
    spread = gauss(conf * est, sw_) / np.maximum(cden, 1e-6)
    cw = np.clip(cden / FAR_SPREAD_MIN, 0.0, 1.0)
    inv = conf * est + (1.0 - conf) * (cw * spread + (1.0 - cw) * prior)
    inv = gauss(inv, max(0.5, FAR_SMOOTH_DEG / 360.0 * w))
    el = 90.0 - (np.arange(r_top, r_bot) + 0.5) / hf * 180.0
    pin = _smoothstep((el - FAR_PIN_FROM_DEG) / FAR_PIN_DEG)[:, None]
    inv = (prior + pin * (inv - prior)).astype(np.float32)
    scene.far_field = {"inv": inv, "el_top_deg": 90.0 - r_top / hf * 180.0, "el_bottom_deg": 90.0 - r_bot / hf * 180.0,
                       "east": np.asarray(east, float), "north": np.asarray(north, float), "up": np.asarray(up, float),
                       "confidence": conf.astype(np.float32), "estimate": est.astype(np.float32)}
    dec = far & (conf > 0.5)
    best = agg.argmin(axis=0)
    name = lambda R: "infinity" if not R else f"{float(R):g} m"                                    # noqa: E731
    out = {"radii_m": [None if not R else float(R) for R in radii], "width": w, "rows": [r_top, r_bot],
           "elevation_deg": [float(FAR_EL_DEG[0]), float(FAR_EL_DEG[1])],
           "far_directions": int(far.sum()), "seen_by_two_frames_share": float((far & (v > 0.5)).sum() / max(1, far.sum())),
           "decided_share": float(dec.sum() / max(1, far.sum())),
           "share_nearest": {name(R): float(((best == i) & dec).sum() / max(1, dec.sum())) for i, R in enumerate(radii)},
           "median_radius_m_where_decided": (float(1.0 / max(np.median(inv[dec]), 1e-6)) if dec.any() else None),
           "mean_cost": {name(R): float(agg[i][dec].mean()) if dec.any() else None for i, R in enumerate(radii)},
           "seconds": round(time.time() - t0, 1),
           "rule": "per far direction the radius whose frames agree best in their detail band (soft minimum over the "
                   "radii; the sky sphere's radius where they do not tell: one frame, or no texture); one smooth field "
                   "of 1 / radius, the sky sphere's at the ground plane's rim"}
    return out


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
            sat = sources[k].sample_saturation(c["uv"][sel])       # 0.97.4: a clipped pixel is darker than the scene was,
            if sat is not None:                                    # so it says nothing about the frame's gain
                inner = inner & ~(sat > GAIN_MAX_SATURATED)
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


FILL_MATCH_MIN_SAMPLES = 300      # shared pixels a fill camera needs for its colour factors
FILL_MATCH_RANGE = (0.4, 2.5)     # the factors are clipped to this
FILL_MATCH_MAX_SCLK_S = 1800.0    # ... from regular frames taken within this many seconds of the fill frame
FILL_MATCH_POINTS = 200_000       # terrain points tried
FILL_MATCH_WITHIN_M = 12.0        # ... on the mesh within this distance of a fill camera


def _fill_key(src) -> str:
    """The colour-match group of a fill source: its camera (``FL``), or - a context frame, whose light is its own -
    the frame itself."""
    return Path(src.name).stem if getattr(src, "context", False) else Path(src.name).name[:2].upper()


def estimate_fill_colour(scene: Scene, sources: Sequence[SourceImage], pivot: Sequence[float] = (0, 0, 0), width: int = GAIN_PASS_WIDTH,
                         up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0), rows_per_chunk: int = 128, apply: bool = True,
                         n_points: int = FILL_MATCH_POINTS, within_m: float = FILL_MATCH_WITHIN_M) -> Dict[str, Any]:
    """Per fill camera (``FL``, ``FR``: the first two letters of the name) the factor per colour channel that makes
    its frames match the regular sources on the surface both see (0.97.7; Christian, 13 Nov 2026: "The front hazcams
    need a different processing white balance for its colors to be consistent with the Navcams.  Please try to
    estimate that without changing ncam.  Just make fhaz images taken at the same time of day projected onto the
    same terrain to be more navcams").

    0.97.8: the comparison is made on points of the terrain mesh itself (``n_points`` spread over the triangles
    within ``within_m`` of a fill camera, the rover model left out), not on the panorama's pixels - the first
    version found no shared pixel at Van Zyl, because what the mast sees of the ground in front of the rover is
    mostly not what the Hazcam sees well.  A point counts for a pair (fill frame, regular frame) when both see it
    unhidden, off their edges and masks, with neither clipped, and the frames were taken within
    FILL_MATCH_MAX_SCLK_S of each other; of several regular frames the one that sees it farthest from its edge.
    r_c = (gain x low band of the regular frame)_c / (low band of the fill frame)_c, c = R, G, B; the factor of a
    camera is the median of r_c.  The regular frames are not changed.  ``apply``: the fill sources' ``gain`` becomes
    that factor (three numbers).  Returns {"cameras": {"FL": {"factor_rgb", "white_balance_ratio_rgb" (the factor /
    its green: what to multiply the processing white balance by), "brightness" (the green factor), "samples",
    "scatter_rgb" (the robust sd of log r), "frames"}}, ...}.  A white balance only when the sources are the 16-bit
    linear frames (``mppp ods --bits 16``).  (``pivot``, ``width`` and the axes are not used any more.)"""
    from .mesh import tau_m
    fill = [bool(getattr(s_, "fill_only", False)) for s_ in sources]
    out: Dict[str, Any] = {"cameras": {}, "method": "median over terrain points both frames see of (gain x regular low band) / "
                                                    f"(fill low band) per channel; frames within {FILL_MATCH_MAX_SCLK_S:g} s"}
    V, F = getattr(scene, "mesh_V", None), getattr(scene, "mesh_F", None)
    if not any(fill) or all(fill) or V is None or F is None or not len(F):
        return out
    V, F = np.asarray(V, float), np.asarray(F)
    rtri = getattr(scene, "rover_triangles", None)
    keep = np.ones(len(F), bool)
    if rtri is not None:
        keep[int(rtri[0]):int(rtri[1])] = False
    cen = V[F].mean(axis=1)
    Cf = np.array([s_.C for s_, f_ in zip(sources, fill) if f_])
    near = np.zeros(len(F), bool)
    for c_ in Cf:
        near |= np.linalg.norm(cen - c_, axis=1) < within_m
    tri = np.flatnonzero(keep & near)
    if not len(tri):
        return out
    a_, b_, c_ = V[F[tri, 0]], V[F[tri, 1]], V[F[tri, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=1)
    rng_ = np.random.default_rng(0)
    pick = rng_.choice(len(tri), size=int(n_points), p=area / area.sum())
    u, v = rng_.random(len(pick)), rng_.random(len(pick))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    X = a_[pick] + u[:, None] * (b_[pick] - a_[pick]) + v[:, None] * (c_[pick] - a_[pick])
    out["points"] = int(len(X))

    def seen(src):                                   # (colour (n, 3) with NaN where not usable, edge share (n,))
        col = np.full((len(X), 3), np.nan, np.float32)
        share = np.zeros(len(X), np.float32)
        uv, z, inside = src.project(X)
        ij = np.flatnonzero(inside)
        if not len(ij):
            return col, share
        dist = src.sample_edge_distance(uv[ij])
        ok = dist > 0.5 * FEATHER_HIGH_FRAC * src.short_side
        ij, dist = ij[ok], dist[ok]
        if not len(ij):
            return col, share
        occ = scene.occluded(np.broadcast_to(src.C, (len(ij), 3)), X[ij], tau_m(np.minimum(z[ij], 100.0)))
        ij, dist = ij[~occ], dist[~occ]
        if not len(ij):
            return col, share
        L = src._bilinear(src.low(), uv[ij]).astype(np.float32)
        sat = src.sample_saturation(uv[ij])
        if sat is not None:
            L[sat > GAIN_MAX_SATURATED] = np.nan
        L[(L < 4.0).any(axis=1)] = np.nan            # too dark to form a ratio
        col[ij], share[ij] = L, dist / src.short_side
        return col, share

    reg = [(k, _sclk(s_.name), seen(s_)) for k, s_ in enumerate(sources) if not fill[k]]
    logs: Dict[str, list] = {}
    frames: Dict[str, set] = {}
    for kf, sf in enumerate(sources):
        if not fill[kf]:
            continue
        cf, _ = seen(sf)
        tf = _sclk(sf.name)
        best = np.full((len(X), 3), np.nan, np.float32)
        best_share = np.zeros(len(X), np.float32)
        for kr, tr, (cr, shr) in reg:
            if not getattr(sf, "context", False) and tf is not None and tr is not None and abs(tr - tf) > FILL_MATCH_MAX_SCLK_S:
                continue                             # (a context frame is matched to the panorama whenever it was taken)
            g = np.asarray(sources[kr].gain, np.float32)
            better = np.isfinite(cr).all(axis=1) & (shr > best_share)
            best[better], best_share[better] = cr[better] * g, shr[better]
        both = np.isfinite(best).all(axis=1) & np.isfinite(cf).all(axis=1)
        if both.any():
            cam = _fill_key(sf)
            logs.setdefault(cam, []).append(np.log(best[both] / cf[both]))
            frames.setdefault(cam, set()).add(sf.name)
        sf.release()
    for s_ in sources:
        if not getattr(s_, "fill_only", False):
            s_.release()
    for cam, parts in sorted(logs.items()):
        v_ = np.vstack(parts)
        rec: Dict[str, Any] = {"samples": int(len(v_)), "frames": sorted(frames[cam])}
        if len(v_) >= FILL_MATCH_MIN_SAMPLES:
            med = np.median(v_, axis=0)
            f = np.clip(np.exp(med), *FILL_MATCH_RANGE)
            rec.update(factor_rgb=[round(float(x), 4) for x in f], brightness=round(float(f[1]), 4),
                       white_balance_ratio_rgb=[round(float(x / f[1]), 4) for x in f],
                       scatter_rgb=[round(float(1.4826 * np.median(np.abs(v_[:, i] - med[i]))), 4) for i in range(3)])
            if apply:
                for s_ in sources:
                    if getattr(s_, "fill_only", False) and _fill_key(s_) == cam:
                        s_.gain = f.astype(np.float32)
        else:
            rec["note"] = f"fewer than {FILL_MATCH_MIN_SAMPLES} shared points: no factor, the frames as processed"
        out["cameras"][cam] = rec
    return out


def render_eye(scene: Scene, sources: Sequence[SourceImage], pivot: Sequence[float], eye: str, width: int, height: int,
               ipd_m: float, rows_per_chunk: int = 128, up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0),
               feather_px: float = FEATHER_PX, progress=None, sky_fill: bool = True,
               azimuth0_deg: float = 0.0, linear: bool = False, ground_fill: bool = True, pole_merge_deg="default",
               range_merge_m="default", near_merge_m="default") -> Dict[str, np.ndarray]:
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
    neff_vals: List[np.ndarray] = []                         # 0.98.0: the detail band's frames' worth on terrain pixels
    prov = np.zeros((height, width, 3), np.uint8)            # 0.97.13: the provenance picture's colours (BGR), blended
    u_ = np.asarray(up, float) / np.linalg.norm(up)
    pl = getattr(scene, "plane", None)
    t0 = time.time()
    for r0 in range(0, height, rows_per_chunk):
        r1 = min(height, r0 + rows_per_chunk)
        rshare = None
        if isinstance(near_merge_m, str):
            near_merge_m = NEAR_MERGE_M
        if isinstance(range_merge_m, str):
            range_merge_m = (RANGE_MERGE_FROM_M, RANGE_MERGE_TO_M) if RANGE_MERGE else None
        if (range_merge_m or near_merge_m) and eye in ("left", "right"):     # the range from the pivot's own ray
            oc, dc = ods_rays(width, height, (r0, r1), P, 0.0, "centre", up, north, east, azimuth0_deg)
            tc = scene.cast(oc, dc)["t"]
            rshare = np.ones(len(tc))
            if range_merge_m:
                rshare = rshare * range_merge(tc, range_merge_m[1], range_merge_m[0])
            if near_merge_m:
                rshare = rshare * near_merge(tc, near_merge_m)
        o, d = ods_rays(width, height, (r0, r1), P, ipd_m, eye, up, north, east, azimuth0_deg, pole_merge_deg=pole_merge_deg,
                        share=rshare)
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
        w_hi_sq = np.zeros(n, np.float32)                      # 0.98.0: for the frames' worth of the detail band
        strongest = np.full(n, -1, np.int32)
        strongest_w = np.zeros(n, np.float32)
        prov_lo = np.zeros((n, 3), np.float32)                 # 0.97.13: the frames' colours by their weights
        prov_hi = np.zeros((n, 3), np.float32)
        K = c["k"].shape[1]
        for slot in range(K):
            ks = c["k"][:, slot]
            for k in np.unique(ks[ks >= 0]):
                sel = np.flatnonzero(ks == k)
                src = sources[k]
                lo, full = src.sample_bands(c["uv"][sel, slot])
                dist = c["dist"][sel, slot]
                aterm = c["ang"][sel, slot]
                fwt = c["fill_weight"][sel, slot]
                nr = c["near"][sel]                              # 0.97.13: 0 (terrain) to 1 (near): the powers rise
                w_lo = _smoothstep(dist / (FEATHER_LOW_FRAC * src.short_side)) * aterm ** (1.0 + nr * (NEAR_LOW_POWER - 1.0)) * fwt
                w_hi = _smoothstep(dist / (FEATHER_HIGH_FRAC * src.short_side)) * aterm ** (TERRAIN_DETAIL_POWER + nr * (NEAR_PARALLAX_POWER - TERRAIN_DETAIL_POWER)) * fwt
                sat = src.sample_saturation(c["uv"][sel, slot])
                if sat is not None:                              # a saturated pixel: another frame, where there is one
                    keep = (1.0 - SAT_DOWNWEIGHT * np.clip(sat, 0.0, 1.0)).astype(np.float32)
                    w_lo, w_hi = w_lo * keep, w_hi * keep
                g = src.gain
                low_acc[sel] += (g * lo) * w_lo[:, None]
                hi_acc[sel] += (g * (full - lo)) * w_hi[:, None]
                w_low_sum[sel] += w_lo
                w_hi_sum[sel] += w_hi
                w_hi_sq[sel] += w_hi * w_hi
                pc = _source_colour(k)
                prov_lo[sel] += w_lo[:, None] * pc
                prov_hi[sel] += w_hi[:, None] * pc
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
            # 0.97.13: with the rover model in the mesh the model itself says what is rover - all other mesh is
            # terrain (so it is in a mesh of the rover-masked cloud), whatever its height above the regional plane (on a slope the ground near the rover is not
            # within GROUND_BAND_M of it: Taylorfjellet, 0.3 m)
            level = ((np.abs(hgt) < GROUND_BAND_M) | (getattr(scene, "rover_triangles", None) is not None)
                     | bool(getattr(scene, "mesh_is_terrain", False)))      # 0.97.15: or the mesh is terrain alone
            g = has & ~c["on_rover"] & ((c["label"] == LABEL_PLANE) | ((c["label"] == LABEL_MESH) & level))
            ground[r0:r1] = g.reshape(r1 - r0, width)
        source[r0:r1] = np.where(has, strongest, -1).astype(np.int16).reshape(r1 - r0, width)
        ne = np.where(w_hi_sq > 1e-12, w_hi_sum * w_hi_sum / np.maximum(w_hi_sq, 1e-12), 0.0)     # (sum w)^2 / sum w^2
        terr = hh & (c["label"] == LABEL_MESH) & ~c["on_rover"]
        if terr.any():
            neff_vals.append(ne[terr][::7].astype(np.float32))
        pm = 0.5 * prov_lo / np.maximum(w_low_sum, 1e-6)[:, None] + 0.5 * prov_hi / np.maximum(w_hi_sum, 1e-6)[:, None]
        pm[~hh] = (prov_lo / np.maximum(w_low_sum, 1e-6)[:, None])[~hh]
        prov[r0:r1] = np.clip(np.rint(pm), 0, 255).astype(np.uint8).reshape(r1 - r0, width, 3)
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
    kw = max(1.0, width / HOLE_FILL_REF_WIDTH)             # 0.97.7: the limits are angles - they grow with the panorama
    filled = fill_cracks(rgb, source, label, max_px=int(HOLE_FILL_MAX_PX * kw * kw), max_width_px=HOLE_FILL_MAX_WIDTH_PX * kw,
                         inpaint_px=int(round(5 * kw)))
    if checked:
        _brown((source == -1) & rover_px)                    # the rest of the model nobody coloured
    gfill = fill_ground(rgb, source, ground) if ground_fill else np.zeros(label.shape, bool)
    sky = fill_sky(rgb, source, label) if sky_fill else np.zeros(label.shape, bool)
    evened = even_sky(rgb, label) if sky_fill else {}
    if checked:                                              # and any surface still without a source
        _brown((source == -1) & ((label == LABEL_MESH) | (label == LABEL_PLANE)))
    return {"brown_filled": brown, "rover": rover_px, "rgb": rgb, "source": source, "label": label, "range": rng, "hidden_fallback": hidden, "filled": filled,
            "sky_filled": sky, "sky_evened": evened, "ground_filled": gfill, "surface_check_rejected": rej, "provenance": prov,
            "sky_weight": evened.pop("weight", None), "effective_frames": _effective_frames(neff_vals)}


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


GAIN_MAX_SATURATED = 0.02        # gain pairs leave out samples within reach of a saturated pixel (its blurred share)
SKY_EVEN_WIDTH = 1024            # the sky is evened on a panorama of this width ...
SKY_EVEN_AZ_FRAC = 0.30          # dark patches narrower than this share of the panorama's width,
SKY_EVEN_EL_DEG = 40.0           #   or shorter than this in elevation, are lifted to the sky around them
SKY_EVEN_SMOOTH_FRAC = 0.006     # the sky's brightness is compared after this blur (of the width)
SKY_EVEN_MAX_GAIN = 2.5          # no pixel is brightened by more than this
SKY_EVEN_MIN_GAIN = 1.02         # ... and patches less than 2 % dark are left alone
SKY_EVEN_EDGE = 0.07             # a relative brightness step per pixel (at SKY_EVEN_WIDTH) above this is a skyline
SKY_EVEN_MARGIN_DEG = 1.5        # the evening fades out over this much above the skyline
SKY_EVEN_SKYLINE_FRAC = 0.06     # ... taken as the highest within this share of the width


def even_sky(rgb: np.ndarray, label: np.ndarray, sky: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Lift (in place) the dark patches of the sky (0.97.4; Christian, 13 Nov 2026: "dark splotches in the sky for
    over saturated images looking into the sun").  A frame that clipped on the bright sky, or was matched darker than
    its neighbours, leaves a patch darker than the sky around it; the real sky has no such holes.  The smoothed sky
    brightness is closed morphologically (a maximum, then a minimum, over SKY_EVEN_AZ_FRAC of the width by
    SKY_EVEN_EL_DEG; wrapped in azimuth): valleys smaller than that window are filled to their rim, slopes and bright
    regions are unchanged.  Each sky pixel is multiplied by closed / smoothed (at least 1, at most SKY_EVEN_MAX_GAIN),
    channel by channel (a patch's tint is lifted with it); fine texture stays.  ``sky``: the mask (default: far, above the horizon).
    Returns {"pixels", "share_of_sky", "gain_p99", "gain_max"}."""
    import cv2
    from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_closing, minimum_filter1d
    H, W = label.shape
    el_s = lambda n: 90.0 - (np.arange(n) + 0.5) / n * 180.0
    el = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    m = ((label == LABEL_FAR) & (el > 0.0)[:, None]) if sky is None else np.asarray(sky, bool)
    out = {"pixels": 0, "share_of_sky": 0.0, "gain_p99": 1.0, "gain_max": 1.0}
    if m.sum() < 100:
        return out
    w = min(SKY_EVEN_WIDTH, W)
    h = max(2, int(round(H * w / W)))
    top = 255.0 if rgb.dtype == np.uint8 else LINEAR_TOP
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    ms = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    # the far hills are "far" too: the sky ends, column by column, at the highest sharp edge (a skyline; a dark
    # patch has soft edges) or surface pixel, and the evening fades out SKY_EVEN_MARGIN_DEG above it
    Ls = gaussian_filter(small.mean(axis=2), 1.0, mode="nearest")
    gy, gx = np.gradient(Ls)
    edge = (np.hypot(gx, gy) / np.maximum(Ls, 1.0) > SKY_EVEN_EDGE) | (ms < 0.5)
    edge[el_s(h) <= 0.0] = True
    skyline = np.where(edge.any(axis=0), np.argmax(edge, axis=0), h).astype(np.float32)
    k = max(3, int(round(SKY_EVEN_SKYLINE_FRAC * w)) | 1)                   # a hazy hill's edge is missed in places:
    skyline = minimum_filter1d(skyline, size=k, mode="wrap")               # the highest skyline nearby holds
    skyline = gaussian_filter(np.concatenate([skyline[-k:], skyline, skyline[:k]]), 0.25 * k, mode="nearest")[k:-k]
    margin = SKY_EVEN_MARGIN_DEG / 180.0 * h
    ramp = np.clip((skyline[None, :] - margin - np.arange(h)[:, None]) / max(1.0, margin), 0.0, 1.0).astype(np.float32)
    ms = ms * (ramp > 0)
    out["weight"] = (ramp * (ms > 0.5)).astype(np.float32)                 # 0.97.13: 1 = sky well above the skyline
    sg = max(1.0, SKY_EVEN_SMOOTH_FRAC * w)
    pad = int(4 * sg) + int(SKY_EVEN_AZ_FRAC * w) + 2
    wrap = lambda x: np.concatenate([x[:, -pad:], x, x[:, :pad]], axis=1)
    den = gaussian_filter(wrap(ms), sg, mode="nearest")
    inside = den > 0.05
    if not inside.any():
        return out
    # off the sky the brightness continues from the nearest sky pixel: the horizon band is then a slope to the edge of
    # the window, not a valley between the sky and nothing, and is left as it is
    near = distance_transform_edt(~inside, return_distances=False, return_indices=True)
    size = (max(3, int(round(SKY_EVEN_EL_DEG / 180.0 * h)) | 1), max(3, int(round(SKY_EVEN_AZ_FRAC * w)) | 1))
    gain = np.ones((h, w, 3), np.float32)
    for k in range(3):                                                     # each channel: the patch's tint goes with it
        A = gaussian_filter(wrap(small[..., k] * ms), sg, mode="nearest") / np.maximum(den, 1e-6)
        A = A[near[0], near[1]]
        C = gaussian_filter(grey_closing(A, size=size, mode="nearest"), sg, mode="nearest")
        g = np.where(inside & (A > 1.0), np.clip(C / np.maximum(A, 1e-6), 1.0, SKY_EVEN_MAX_GAIN), 1.0)[:, pad:pad + w]
        gain[..., k] = g
    gain = np.where(gain.max(axis=2, keepdims=True) < SKY_EVEN_MIN_GAIN, 1.0, gain).astype(np.float32)
    gain = cv2.GaussianBlur(gain, (0, 0), max(1.0, 0.5 * sg), borderType=cv2.BORDER_REPLICATE)
    gain = 1.0 + (gain - 1.0) * ramp[..., None]
    if float(gain.max()) <= 1.0 + 1e-4:
        return out
    f = H / h
    n_ch, g_max = 0, 1.0
    for r0 in range(0, H, 1024):                                           # in strips: no full-size float copy
        r1 = min(H, r0 + 1024)
        if not m[r0:r1].any():
            continue
        s0, s1 = max(0, int(r0 / f) - 2), min(h, int(np.ceil(r1 / f)) + 2)
        blk = cv2.resize(gain[s0:s1], (W, max(1, int(round((s1 - s0) * f)))), interpolation=cv2.INTER_CUBIC)
        off = min(max(0, int(round(r0 - s0 * f))), max(0, blk.shape[0] - (r1 - r0)))
        G = np.clip(blk[off:off + (r1 - r0)], 1.0, SKY_EVEN_MAX_GAIN)
        if G.shape[0] < r1 - r0:
            G = np.concatenate([G, np.repeat(G[-1:], r1 - r0 - G.shape[0], axis=0)])
        G[~m[r0:r1]] = 1.0
        rgb[r0:r1] = np.clip(rgb[r0:r1].astype(np.float32) * G, 0, top).astype(rgb.dtype)
        gm = G.max(axis=2)
        n_ch += int((gm > SKY_EVEN_MIN_GAIN).sum())
        g_max = max(g_max, float(gm.max()))
    gs = gain.max(axis=2)[ms > 0.5]
    out.update(pixels=n_ch, share_of_sky=float(n_ch / max(1, m.sum())),
               gain_p99=float(np.percentile(gs, 99)) if gs.size else 1.0, gain_max=g_max)
    return out


def fill_cracks(rgb: np.ndarray, source: np.ndarray, label: np.ndarray, max_px: int = HOLE_FILL_MAX_PX,
                max_width_px: float = HOLE_FILL_MAX_WIDTH_PX, inpaint_px: int = 5) -> np.ndarray:
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
            sub = cv2.inpaint(np.ascontiguousarray(rgb[y0:y1, x0:x1]), fm, int(inpaint_px), cv2.INPAINT_TELEA)
        else:                                                # float: one channel at a time
            sub = np.dstack([cv2.inpaint(np.ascontiguousarray(rgb[y0:y1, x0:x1, k]), fm, int(inpaint_px), cv2.INPAINT_TELEA) for k in range(3)])
        rgb[y0:y1, x0:x1][filled[y0:y1, x0:x1]] = sub[filled[y0:y1, x0:x1]]
        source[filled] = -2
    return filled


def _source_colour(k: int) -> np.ndarray:
    """The provenance colour (BGR, float32) of source ``k``: golden-angle hues."""
    import cv2
    hsv = np.array([[[int((k * 137.508) % 180), 200, 230]]], np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def provenance_png(source: np.ndarray, out_png: PathLike, n_sources: int, mix: Optional[np.ndarray] = None) -> Path:
    """Which frames coloured each pixel (golden-angle hues), black = no data.  ``mix`` (0.97.13; Christian, 13 Nov
    2026: "The provenance should be a smoother function when multiple images see the same surfaces"): the frames'
    colours mixed by their weights in the blend (the mean of the colour band's and the detail band's shares), (H,
    W, 3) BGR - a pixel between two hues is a blend of the two frames; without it the strongest frame's colour alone
    (the picture before 0.97.13).  The fills keep their flat colours: grey cracks, light blue sky fill, brown
    ground fill, dark brown where the surface check left no source."""
    import cv2
    img = np.zeros(source.shape + (3,), np.uint8)
    if mix is not None:
        img[source >= 0] = mix[source >= 0]
        n_sources = 0
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
SKY_WEIGHT_NAME = "ods_sky_weight.png"   # 0.97.13: 8-bit, small: 255 = sky well above the skyline (see even_sky)
SAT_DOWNWEIGHT = 0.98            # a saturated source pixel keeps this much less weight in the blend
STACK_NAME = "360_3DTD"              # .png and .jpg: the left eye on top of the right (Christian, 10 Nov: "put the left view on
#                                      top"); 0.97.0 (Christian, 12 Nov 2026: "remove '_ods_stereo_square' and add '_360_3DTD' to every top
#                                      down 360 image product so that meta knows how to project it"): <prefix>_360_3DTD.jpg, was
#                                      <prefix>_ods_stereo_square.jpg
OLD_STACK_NAME = "ods_stereo_square"
OLD_NAMES = ("ods_left.png", "ods_right.png", "ods_stereo_square.png", "ods_stereo_square.jpg")


def ods_prefix(station_label: str, rover: bool = False) -> str:
    """The file-name prefix of a station's panoramas (Christian, 10 Nov: "name the ods with a site and drive
    prefix"): ``Sol0052_S003D2046`` / ``Sol0052_S003D2046_rover``."""
    return f"{station_label}_rover" if rover else str(station_label)


def ods_file(out_dir: PathLike, kind: str, prefix: str = "", ext: str = ".png") -> Path:
    """``<prefix>_ods_<kind><ext>`` (kind left / right / centre; the top-down square: :func:`stack_file`); ``ods_<kind><ext>`` without a prefix."""
    return Path(out_dir) / f"{(prefix + '_') if prefix else ''}ods_{kind}{ext}"


def stack_file(out_dir: PathLike, prefix: str = "", ext: str = ".png", suffix: str = "") -> Path:
    """The top-down stereo 360 (left eye above the right): ``<prefix>_360_3DTD<suffix><ext>`` (``suffix`` "_16": the
    16-bit linear one); ``360_3DTD...`` without a prefix."""
    return Path(out_dir) / f"{(prefix + '_') if prefix else ''}{STACK_NAME}{suffix}{ext}"


GEOMETRY_VIEWS = ("range_log", "range_linear", "normal")


def geometry_panorama(scene: Scene, station, out_dir: PathLike, prefix: str = "", width: int = 8192, ipd_m: float = 0.064,
                      azimuth0_deg: float = 0.0, pole_merge_deg="default", rows_per_chunk: int = 64, top: str = "left") -> Dict[str, str]:
    """0.97.20 (Christian, 13 Nov 2026: "how hard would it be to render a ods of the range or normal map? that would
    help checking the model during debugging and dev"): the model alone, seen through the panorama's own rays -
    ``<prefix>_range_log_360_3DTD.jpg``, ``_range_linear_`` and ``_normal_`` (left eye on top, as the colour ones).
    Range: from each eye's ray origin, in the colours of the per-image depth pictures (:func:`raycast.depth_rgb`:
    turbo, the ground plane darker, the sky white, nothing black).  Normal: in the station's frame,
    [R, G, B] = 127.5 * (1 + [east, north, up]); the sky white, nothing black.  No source frame is used: a fault that
    shows here is in the mesh, the plane, the rover model or the far field, not in the blending."""
    import cv2
    from .raycast import JPEG_QUALITY, depth_rgb
    height = width // 2
    e, n, u = (np.asarray(v, float) / np.linalg.norm(v) for v in (station.east, station.north, station.up))
    enu = np.stack([e, n, u])
    order = ("left", "right") if top == "left" else ("right", "left")
    imgs = {k: np.zeros((2 * height, width, 3), np.uint8) for k in GEOMETRY_VIEWS}
    for i, eye in enumerate(order):
        for r0 in range(0, height, rows_per_chunk):
            r1 = min(height, r0 + rows_per_chunk)
            o, d = ods_rays(width, height, (r0, r1), station.pivot, ipd_m, eye, station.up, station.north, station.east,
                            azimuth0_deg, pole_merge_deg=pole_merge_deg)
            h = scene.cast(o, d)
            lab = h["label"].reshape(r1 - r0, width)
            rng = np.where(np.isfinite(h["t"]), h["t"], 0.0).reshape(r1 - r0, width)
            rows = slice(i * height + r0, i * height + r1)
            imgs["range_log"][rows] = depth_rgb(rng, lab, "log")
            imgs["range_linear"][rows] = depth_rgb(rng, lab, "linear")
            nm = (np.asarray(h["normal"], float) @ enu.T).reshape(r1 - r0, width, 3)
            nimg = np.clip((nm * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)[..., ::-1].copy()      # BGR for OpenCV
            nimg[(lab != LABEL_MESH) & (lab != LABEL_PLANE)] = 0
            nimg[lab == LABEL_FAR] = 255
            imgs["normal"][rows] = nimg
    outs = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for k, im in imgs.items():
        f = stack_file(out_dir, f"{prefix}_{k}" if prefix else k, ".jpg")
        cv2.imwrite(str(f), im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        outs[k] = str(f)
    return outs


def _effective_frames(vals: Sequence[np.ndarray]) -> Optional[Dict[str, Any]]:
    """The detail band's frames' worth on the terrain: (sum w)^2 / sum w^2 of the frames' weights per pixel - 1 when
    one frame carries a pixel, n when n frames share it equally; the noise of the blend is that of a frame divided
    by its square root."""
    if not vals:
        return None
    v = np.concatenate(vals)
    return {"median": float(np.median(v)), "mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "share_above_1p5": float((v > 1.5).mean()), "noise_factor_median": float(1.0 / np.sqrt(max(np.median(v), 1e-6))),
            "detail_power": TERRAIN_DETAIL_POWER,
            "meaning": "per terrain pixel of the mesh: (sum w)^2 / sum w^2 over the frames' detail-band weights; noise = one "
                       "frame's / sqrt(this)"}


def _radiance_record(frames: Sequence[Dict[str, Any]], exposure16: Optional[float]) -> Optional[Dict[str, Any]]:
    """0.98.0 (Christian, 14 Nov 2026: "The 16 bit images should have a universal scale to radiance"): how a value of
    the 16-bit panorama turns back into radiance.  A frame's 16-bit value is
        radiance x white balance x brightness / zenith scale x scale_rad_to_int16
    with the block's settings (the same at every station) and the frame's own zenith scale (the sun's height and the
    opacity, :func:`mppp.radiometry.zenith_scale`; one 360 is taken within minutes, so it hardly varies in a station).
    The panorama multiplies by exposure16 and by each frame's blending gain (median 1)."""
    if not frames or exposure16 is None:
        return None
    z = np.array([f["scale_zenith"] if f.get("scale_zenith") else 1.0 for f in frames], float)
    wb = np.asarray(frames[0]["white_balance"], float) * float(frames[0]["brightness"])
    k = float(frames[0]["scale_rad_to_int16"]) * float(exposure16)
    zm = float(np.median(z))
    return {"units": "the units of the PDS RAD product (W m-2 sr-1 um-1)",
            "value_per_unit_radiance_at_zenith_scale_1": [float(v) for v in wb * k],
            "zenith_scale_median": zm, "zenith_scale_min_max": [float(z.min()), float(z.max())],
            "radiance_rgb": "value / value_per_unit_radiance_at_zenith_scale_1 x zenith_scale_median",
            "value_per_unit_radiance": [float(v) for v in wb * k / zm],
            "same_at_every_station": "white balance, brightness, scale_rad_to_int16 and exposure16 (1 by default); the zenith scale "
                                     "is the station's own - with it left in, the values compare as radiance scaled to a common sun",
            "not_radiance": "the frames' blending gains (0.6-1.6, median 1; in this file), the saturated pixels made neutral, the fills"}


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
                   bits: int = 8, pds: Optional[PathLike] = None, tone: str = "pipeline", exposure16: Union[float, str] = 1.0,
                   ground_fill: bool = True, mask_mast: bool = True,
                   tone_levels_file: Optional[PathLike] = None, centre_depth: bool = True, pole_merge_deg="default", sky_radius_m="default",
                   fill_families: str = "", nadir_mark: bool = True, site_name: Optional[str] = None,
                   nadir_mark_deg: Optional[float] = None, far_field: Optional[bool] = None,
                   context_images: Sequence[str] = (), geometry: bool = True) -> Dict[str, Any]:
    """The two eye images of a station -> ``out_dir/<prefix>_ods_left.png``, ``<prefix>_ods_right.png``,
    ``<prefix>_360_3DTD.png/.jpg``, ``provenance.png`` (left eye's sources), ``ods.json``.  ``prefix`` None:
    :func:`ods_prefix` (``Sol0052_S003D2046`` / ``Sol0052_S003D2046_rover``); '' for the bare names.  Returns the record."""
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = ods_prefix(station.label, rover=not use_masks)
    height = width // 2
    by_ = {im.name for im in rec.images.values() if im.has_pose}
    own = [n for n in (list(images) if images else [n for n in station.images if n[:1].upper() in set(families.upper())]) if n in by_]
    ctx_names = [n for n in dict.fromkeys(context_images or ()) if n in by_]         # 0.97.15: see station_sources
    gone = [n for n in (context_images or ()) if n not in by_]
    if gone:
        log.warning("ods: %d context frame(s) not registered in the block, left out: %s", len(gone), ", ".join(gone))
    own = [n for n in own if n not in set(ctx_names)]
    fill_names = fill_frames(project, rec, station, fill_families, own) + ctx_names
    if ctx_names:
        log.info("ods: %d context frame(s), masked, fill only: %s", len(ctx_names), ", ".join(ctx_names))
    if fill_families:
        log.info("ods: %d fill-only frame(s) of the camera families %r: %s", len(fill_names), fill_families, ", ".join(fill_names) or "none at this station")
    lin, lin_rep = None, None
    if int(bits) == 16:                              # 0.96.1: the frames as 16-bit linear images (mppp.ibr.linear16)
        from .linear16 import ensure_linear16, files as _lin_files
        by = {im.name for im in rec.images.values() if im.has_pose}
        want = [n for n in (list(images) if images else [n for n in station.images if n[:1].upper() in set(families.upper())])
                if n in by and n not in set(ctx_names)]
        lin_rep = ensure_linear16(project, want + fill_names, pds=pds)
        lost = [n for n in fill_names if n in lin_rep["missing"]]
        if lost:                                     # a fill frame without a 16-bit image is left out, not the 16 bits
            log.warning("ods --bits 16: %d fill-only frame(s) left out (no 16-bit image): %s", len(lost), ", ".join(lost))
            fill_names = [n for n in fill_names if n not in lost]
            for n in lost:
                lin_rep["missing"].pop(n, None)
        want = want + fill_names
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
                              mast_pose=mast_pose, fill_images=fill_names, context_images=ctx_names)
    if not sources:
        raise ValueError(f"station {station.label}: no source images of families {families!r} with a pose")
    axes = dict(up=station.up, north=station.north, east=station.east)
    t0 = time.time()

    def progress(eye, r, h, dt):
        log.info("ods %s: %d / %d rows (%.0f s)", eye, r, h, dt)

    if hasattr(scene, "set_sky_sphere"):                 # 0.97.0: the sky on a sphere, the plane out to it
        scene.set_sky_sphere(SKY_RADIUS_M if isinstance(sky_radius_m, str) else sky_radius_m)
    rec_out: Dict[str, Any] = {"station": station.to_json(), "prefix": prefix, "width": width, "height": height, "ipd_m": ipd_m,
                               "sky_radius_m": getattr(scene, "sky_radius_m", None), "range_merge_m": ([RANGE_MERGE_FROM_M, RANGE_MERGE_TO_M] if RANGE_MERGE else None), "near_merge_m": NEAR_MERGE_M,
                               "pole_merge_deg": ({"up": POLE_MERGE_DEG, "down": POLE_MERGE_DOWN_DEG} if isinstance(pole_merge_deg, str)
                                                  else pole_merge_deg if isinstance(pole_merge_deg, dict)
                                                  else (list(pole_merge_deg) if pole_merge_deg else None)),
                               "sources": [s.name for s in sources], "scale": scale, "zcam_scale": zcam_scale,
                               "rover": "shown" if not use_masks else "masked out",
                               "azimuth_of_column_0_deg": float(azimuth0_deg) % 360.0,
                               "head_hardware_masked": [s.name for s in sources if s.extra_mask is not None and s.extra_mask.any()],
                               "mast_masked": [s.name for s in sources if getattr(s, "mast_masked", False)],
                               "fill_only_sources": [s.name for s in sources if s.fill_only],
                               "context_sources": [s.name for s in sources if s.context],
                               "near_blend": {"range_m": NEAR_BLEND_M, "parallax_power_colour": NEAR_LOW_POWER, "parallax_power_detail": NEAR_PARALLAX_POWER},
                               "eyes": {}}
    mark = None
    if nadir_mark:                                       # 0.97.7: the black disc with the credits at the nadir
        from . import nadir as _nadir
        lines = _nadir.mark_lines([s.name for s in sources if not s.context], site_name)   # the station's own sols
        fwd = 0.0
        try:                                             # the text ahead of the rover's front
            from .rover_deck import rover_pose as _rover_pose
            fwd = _nadir.forward_azimuth_deg(mast_pose or _rover_pose(project, rec, station.images), station.north, station.east)
        except Exception as e:                           # noqa: BLE001 - then it starts at north
            log.warning("ods: no rover pose for the nadir mark (%s); its text is centred on north", e)
        mark = _nadir.make_mark(width, height, lines, _nadir.NADIR_MARK_DEG if nadir_mark_deg is None else nadir_mark_deg,
                                azimuth0_deg, centre_azimuth_deg=fwd)
        # 0.97.13: the disc lies on the surface under the mast - the median depth below the pivot of what the rays at
        # the disc's rim meet (the deck with the rover shown, the ground without)
        rim = int(np.clip(mark["row0"], 0, height - 1))
        o_, d_ = ods_rays(width, height, (rim, rim + 1), station.pivot, 0.0, "centre", station.up, station.north, station.east, azimuth0_deg)
        h_ = scene.cast(o_, d_)
        surf = (h_["label"] == LABEL_MESH) | (h_["label"] == LABEL_PLANE)
        u_st = np.asarray(station.up, float) / np.linalg.norm(station.up)
        below = -(h_["t"][surf, None] * d_[surf]) @ u_st
        below = below[np.isfinite(below) & (below > 0.05)]
        mark["depth_m"] = float(np.median(below)) if len(below) >= 8 else 2.0
        rec_out["nadir_mark"] = {"depth_below_pivot_m": mark["depth_m"], "lines": lines, "text": mark["text"], "radius_deg": mark["radius_deg"], "first_row": mark["row0"],
                                 "text_centre_azimuth_deg": mark["centre_azimuth_deg"],
                                 "note": "a light blue disc with this text in black, ahead of the rover's front, not data; it lies on the "
                                         "horizontal plane depth_below_pivot_m below the pivot (each eye sees it with that plane's "
                                         "disparity; from the pivot it covers the rows from first_row down); in every colour "
                                         "panorama, not in the depth products"}
    if gains:
        g = estimate_gains(scene, sources, station.pivot, width=min(GAIN_PASS_WIDTH, width), rows_per_chunk=rows_per_chunk, **axes)
        rec_out["gains"] = g
        log.info("ods: gains over %d source pairs: %s", g["pairs"], ", ".join(f"{v:.3f}" for v in g["gains"].values()))
    if hasattr(scene, "far_field"):
        scene.far_field = None
    if (FAR_FIELD if far_field is None else far_field) and getattr(scene, "sky_radius_m", None):       # 0.97.13
        ff = estimate_far_field(scene, sources, station.pivot, width, rows_per_chunk=rows_per_chunk, **axes)
        rec_out["far_field"] = ff or {"note": "no far direction that two frames see: the sky sphere alone"}
        if ff:
            log.info("ods: far field in %.0f s: %.0f %% of the far directions decided (%s); median radius there %.0f m",
                     ff["seconds"], 100 * ff["decided_share"],
                     ", ".join(f"{k} {100 * x:.0f} %" for k, x in ff["share_nearest"].items()), ff["median_radius_m_where_decided"] or 0)
            np.savez_compressed(out / "far_field.npz", inverse_radius_per_m=scene.far_field["inv"],
                                el_top_deg=scene.far_field["el_top_deg"], el_bottom_deg=scene.far_field["el_bottom_deg"])
    if any(s.fill_only for s in sources):                # 0.97.7: the fill frames' colours matched to the regular ones
        fc = estimate_fill_colour(scene, sources, station.pivot, width=min(GAIN_PASS_WIDTH, width), rows_per_chunk=rows_per_chunk, **axes)
        fc["linear_sources"] = bool(linear)
        if not linear:
            fc["note"] = "8-bit sources: the factors match the pictures, they are not a white balance of the radiance"
        rec_out["fill_colour_match"] = fc
        if not fc["cameras"]:
            log.info("ods: the fill frames share no terrain point with a regular frame taken within %g s - not matched", FILL_MATCH_MAX_SCLK_S)
        for cam, r in fc["cameras"].items():
            if "factor_rgb" in r:
                log.info("ods: %s matched to the regular frames: x %s (R, G, B) over %d shared pixels; white balance x %s, "
                         "brightness x %.3f", cam, r["factor_rgb"], r["samples"], r["white_balance_ratio_rgb"], r["brightness"])
            else:
                log.info("ods: %s not matched (%d shared pixels)", cam, r["samples"])
    levels, levels_from, depth_rec = None, None, None
    mark0 = mark
    for eye in eyes:
        if mark0 is not None and eye in ("left", "right"):       # 0.97.13: the mark as this eye sees it on the ground
            m0 = max(0, mark0["row0"] - int(np.ceil(_nadir.GROUND_MARGIN_DEG / 180.0 * height)))
            al = np.zeros((height - m0, width), np.float32)
            ins = np.zeros((height - m0, width), bool)
            for q0 in range(m0, height, rows_per_chunk):
                q1 = min(height, q0 + rows_per_chunk)
                oe, de = ods_rays(width, height, (q0, q1), station.pivot, ipd_m, eye, station.up, station.north, station.east,
                                  azimuth0_deg, pole_merge_deg=pole_merge_deg)
                al[q0 - m0:q1 - m0], ins[q0 - m0:q1 - m0] = _nadir.on_ground(mark0, height, oe, de, (q0, q1), station.pivot, station.up,
                                                                           station.north, station.east, mark0["depth_m"], azimuth0_deg)
            mark = dict(mark0, row0=m0, alpha=al, inside=ins)
            del al, ins
        elif mark0 is not None:
            mark = mark0
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
            v16 = np.clip(v16, 0, 65535).astype(np.uint16)
            if mark is not None:
                mark["bg16"] = np.clip(np.asarray(_nadir.BACKGROUND_RGB)[::-1] * _nadir.BACKGROUND_LINEAR_FACTOR * k16, 0, 65535)
                _nadir.apply_mark(v16, mark, mark["bg16"])                  # light blue (BGR here)
            cv2.imwrite(str(f16), v16)
            from .linear16 import auto_levels, tone_levels, tone_soft, tone_pipeline
            lin8 = v16[..., ::-1].astype(np.float32) / k16           # 0.98.0: the 8-bit view is made from the 16-bit image
            del v16
            if tone == "pipeline":                   # 0.98.0 default: the scale of the block's own 8-bit frames
                view = tone_pipeline(lin8)
            elif tone == "auto":                       # the black and white points from the first eye, for both
                if levels is None:                   # 0.97.0: or from the file the other variant wrote
                    from .linear16 import shared_levels
                    levels, levels_from = shared_levels(tone_levels_file,
                                                        lambda: auto_levels(res["rgb"], res["source"] >= 0))
                    log.info("ods: 8-bit view levels %.1f / %.1f (%s%s)", levels[0], levels[1], levels_from,
                             f": {tone_levels_file}" if tone_levels_file else "")
                view = tone_levels(lin8, *levels)
            else:
                view = tone_soft(lin8) if tone == "soft" else np.clip(lin8, 0, 255)
            del lin8
            over = float((res["rgb"].max(axis=2) > 255.0).mean())
            view8 = np.rint(view[..., ::-1]).astype(np.uint8)
            if mark is not None:
                mark["bg8"] = np.asarray(_nadir.BACKGROUND_RGB)[::-1]
                _nadir.apply_mark(view8, mark, mark["bg8"])
            cv2.imwrite(str(f), view8)
            del view8
        else:
            view8 = np.ascontiguousarray(res["rgb"][..., ::-1])
            if mark is not None:
                mark["bg8"] = np.asarray(_nadir.BACKGROUND_RGB)[::-1]
                _nadir.apply_mark(view8, mark, mark["bg8"])
            cv2.imwrite(str(f), view8)
        if eye in ("centre", "center") and centre_depth:     # 0.97.0: the panorama from the pivot, with its depth
            depth_rec = write_centre_depth(out, prefix, res["range"], res["label"])
        if eye == eyes[0]:
            provenance_png(res["source"], out / "provenance.png", len(sources), mix=res.get("provenance"))
            sw = res.get("sky_weight")                   # 0.97.13: where the enhanced views may smooth the sky
            fsw = out / ((f"{prefix}_" if prefix else "") + SKY_WEIGHT_NAME)
            if sw is not None:
                cv2.imwrite(str(fsw), np.clip(np.rint(255.0 * sw), 0, 255).astype(np.uint8))
            elif fsw.is_file():
                fsw.unlink()
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
        rec_out["eyes"][eye]["effective_frames"] = res.get("effective_frames")      # 0.98.0
        for s in sources:
            s.release()
    if centre_depth and depth_rec is None:               # 0.97.11: the depth without the centre panorama's colours
        td = time.time()                                 # (Christian, 13 Nov 2026: "skip the center render because it
        rng_c = np.zeros((height, width), np.float32)    # takes 1/3 more time"): rays only, no frame is sampled
        lab_c = np.zeros((height, width), np.uint8)
        for r0 in range(0, height, rows_per_chunk):
            r1 = min(height, r0 + rows_per_chunk)
            oc, dc = ods_rays(width, height, (r0, r1), station.pivot, 0.0, "centre", station.up, station.north, station.east,
                              azimuth0_deg)
            hc = scene.cast(oc, dc)
            lab_c[r0:r1] = hc["label"].reshape(r1 - r0, width)
            rng_c[r0:r1] = np.where(hc["label"] != LABEL_NONE, hc["t"], 0).reshape(r1 - r0, width)
        depth_rec = write_centre_depth(out, prefix, rng_c, lab_c)
        depth_rec["seconds"] = round(time.time() - td, 1)
        depth_rec["from"] = "rays from the pivot alone (no centre panorama rendered)"
        log.info("ods: the depth from the pivot without the centre panorama (%.0f s)", depth_rec["seconds"])
        del rng_c, lab_c
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
    if mark is not None:
        try:                                             # 0.97.24: the mark seen from above, for a viewer's ground surface
            tex = out / f"{(prefix + '_') if prefix else ''}{_nadir.TEXTURE_SUFFIX}"
            _nadir.write_texture(mark0, tex)
            rec_out["nadir_mark"].update({"texture": tex.name,
                                          "radius_m": float(mark0["depth_m"] * np.tan(np.radians(mark0["radius_deg"]))),
                                          "compass": mark0.get("compass"),
                                          "texture_note": "the mark seen from above: north up, east to the right, the disc's rim at the "
                                                          "picture's sides; on the ground it is radius_m wide at depth_below_pivot_m "
                                                          "below the pivot (the viewer lays it there as a surface)"})
        except Exception as e:                           # noqa: BLE001 - the panorama does not depend on it
            log.warning("ods: the nadir mark's texture was not written (%s)", e)
        mark0.update({k_: mark[k_] for k_ in ("bg8", "bg16") if k_ in mark})
        rec_out["nadir_mark"]["background_bgr_8bit"] = [float(x) for x in mark.get("bg8", [])]
        if "bg16" in mark:
            rec_out["nadir_mark"]["background_bgr_16bit"] = [float(x) for x in mark["bg16"]]
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
            "value_16bit": "the linear value x 64 x exposure16.  exposure16 is 1 by default (0.98.0): every station on the frames' "
                           "own scale, so one value is one radiance everywhere; 'auto' (0.97) set a gain per station",
            "radiance": _radiance_record(fr, None if isinstance(exposure16, str) else float(exposure16)),
            "view_8bit": {"pipeline": "the 16-bit value / (64 x exposure16), as the block's 8-bit frames, value for value up to 235; "
                                      "each channel rolled off above it (255 at the 16-bit ceiling)",
                          "auto": "the ground's median to 100 of 255, highlights rolled off", "soft": "unchanged to 190, rolled off above",
                          "clip": "cut at 255"}.get(tone) if linear else None,
            "frames_saturated_share_max": max([f["saturated_share"] for f in fr], default=None),
            "frames_clipped_in_8bit_share_max": max([f["clipped_in_8bit_share"] for f in fr], default=None)}
    if geometry and set(eyes) >= {"left", "right"}:                         # 0.97.20: the model through the same rays
        tg = time.time()
        rec_out["geometry"] = {"files": geometry_panorama(scene, station, out, prefix or "", width, ipd_m, azimuth0_deg, pole_merge_deg,
                                                          rows_per_chunk, top=stack_top),
                               "range": "from each eye's ray origin; colours as the per-image depth pictures (photometry_jpg/scales.json: "
                                        "depth_log, depth_linear)",
                               "normal": "station frame: [R, G, B] = 127.5 * (1 + [east, north, up]); sky white, nothing black",
                               "note": "the model alone (mesh, plane, rover model, far field): no source frame is used"}
        rec_out["geometry"]["seconds"] = round(time.time() - tg, 1)
        log.info("ods: range and normal panoramas in %.0f s", rec_out["geometry"]["seconds"])
    rec_out["seconds"] = round(time.time() - t0, 1)
    try:                                                                    # 0.97.4: XMP in the stacked products
        from .xmp import tag_folder
        rec_out["xmp"] = tag_folder(out, heading_centre_deg=float(rec_out.get("azimuth_of_column_0_deg") or 0.0) + 180.0,
                                    extra={"Station": str(getattr(station, "label", "") or prefix)})
    except Exception as e:                                                  # noqa: BLE001 - the products are complete without it
        log.warning("ods: XMP not written (%s: %s)", type(e).__name__, e)
    (out / "ods.json").write_text(json.dumps(rec_out, indent=1, default=float), encoding="utf-8")
    log.info("ods: %s -> %s (%.0f s)", station.label, out, rec_out["seconds"])
    return rec_out
