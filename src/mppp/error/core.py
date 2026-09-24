"""
mppp_error.core -- Mars Photogrammetric Precision Prediction: physics engine.

First-principles 3D precision model for rover-based surface reconstruction.

DESIGN NOTES
============

Ray-based internal representation
---------------------------------
A single image measurement of an object point is an *angular* observation: it
constrains the two directions perpendicular to the ray and says nothing about
range.  Its contribution to the precision (inverse covariance) matrix is
therefore rank-2:

    Lambda_ray = (1 / sigma_perp^2) * (I - u u^T),     sigma_perp = eps * psi * r

where u is the unit view direction, r the range, psi the IFOV (rad/px), and eps
the image-coordinate measurement precision in pixels (per image, per axis --
this is the standard photogrammetric sigma_x').

A stereo station is *not* special-cased.  It is simply two rays whose origins
are separated by the baseline b.  Everything else (the cigar-shaped error
ellipsoid, the r^2/b range scaling) falls out of the sum.

Where the sqrt(2) comes from
----------------------------
Because eps is defined per-image-per-axis, summing two rays gives

    sigma_transverse = eps * psi * r / sqrt(2)        (two measurements average)
    sigma_range      = sqrt(2) * eps * psi * r * r/b  (disparity is a difference)

The familiar textbook form sigma_Z = Z^2 * sigma_disp / (c * b) has the sqrt(2)
hidden inside sigma_disp, which is the precision of the *measured disparity* --
a single number formed by correlating two patches.  If the two image
measurements are independent then sigma_disp = sqrt(2) * sigma_x', and the two
formulations agree exactly.  Both conventions appear in the literature; this
code uses sigma_x' (per-image) throughout because it is the quantity that
generalises to N rays and to mixed mono/stereo visibility.
`selftest.py::test_two_ray_analytic` verifies the correspondence numerically.

eps, rho and theta_c are independent
------------------------------------
  eps    -- magnitude of the random error on one image measurement.
  rho    -- correlation between the errors of two *different* observations.
  theta_c-- angular scale over which rho decays.

rho is not derivable from eps and theta_c.  eps sets how large errors are; rho
describes how much of that error is *shared* between two views (sub-pixel
interpolation bias, foreshortening bias, interior-orientation residual, surface
definition ambiguity).  They are different moments of the same underlying
physics and must be measured separately.

Angular gating and correlation are OFF by default
-------------------------------------------------
Under the current scaling argument theta_c (decorrelation) and theta_max
(matching failure) are within roughly a factor of two of each other, so
applying either without measured values would impose a strong, unvalidated
constraint.  Both are implemented and switchable but default to disabled.
The maps produced with defaults are pure network geometry.
"""

from __future__ import annotations

import warnings
import os
import re
import numpy as np
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple, Literal, List

__all__ = [
    "Surface", "FlatPlane", "GriddedSurface",
    "OcclusionMask", "MASTCAM_Z_34", "NAVCAM", "DEFAULT_ROVER_MASK",
    "Instrument", "Station", "PoseModel", "ModelConfig",
    "Grid", "PrecisionField", "solve_precision_field", "choose_anchor",
    "correlation_kernel",
    "skew", "tangent_basis",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# small linear-algebra helpers (all batched over leading axes)
# --------------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]_x, batched over leading axes. (...,3) -> (...,3,3)."""
    z = np.zeros(v.shape[:-1], dtype=float)
    return np.stack([
        np.stack([z, -v[..., 2], v[..., 1]], axis=-1),
        np.stack([v[..., 2], z, -v[..., 0]], axis=-1),
        np.stack([-v[..., 1], v[..., 0], z], axis=-1),
    ], axis=-2)


def outer(u: np.ndarray) -> np.ndarray:
    """u u^T batched. (...,3) -> (...,3,3)."""
    return np.einsum("...i,...j->...ij", u, u)


def tangent_basis(n: np.ndarray) -> np.ndarray:
    """
    Orthonormal 2x3 basis for the plane perpendicular to n.

    Returns T with shape (...,2,3) such that T @ Sigma @ T.T is the tangent-plane
    block of Sigma.  The in-plane orientation is arbitrary but continuous away
    from n = +/- x_hat.
    """
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    a = np.zeros_like(n)
    # pick the reference axis least aligned with n, per element
    use_x = np.abs(n[..., 0]) < 0.9
    a[..., 0] = np.where(use_x, 1.0, 0.0)
    a[..., 1] = np.where(use_x, 0.0, 1.0)
    t1 = a - n * np.sum(a * n, axis=-1, keepdims=True)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), _EPS)
    t2 = np.cross(n, t1)
    return np.stack([t1, t2], axis=-2)


def sym_inv3(M: np.ndarray) -> np.ndarray:
    """Batched inverse of symmetric 3x3, symmetrised on output."""
    Mi = np.linalg.inv(M)
    return 0.5 * (Mi + np.swapaxes(Mi, -1, -2))


# --------------------------------------------------------------------------
# surfaces
# --------------------------------------------------------------------------

class Surface:
    """Interface: given X, Y arrays return (Z, N) with N the unit up-normal."""

    def sample(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class FlatPlane(Surface):
    """Horizontal plane at z = z0. Normal is +Z everywhere."""
    z0: float = 0.0

    def sample(self, X, Y):
        Z = np.full(X.shape, float(self.z0))
        N = np.zeros(X.shape + (3,))
        N[..., 2] = 1.0
        return Z, N


@dataclass
class GriddedSurface(Surface):
    """
    Bilinear-sampled DTM on a regular grid.  Normals from the local gradient.

    x, y are 1-D coordinate vectors (ascending); Z has shape (len(y), len(x)).
    Points outside the DTM footprint fall back to the nearest edge value, which
    is adequate for a design tool but should not be trusted near the boundary.
    """
    x: np.ndarray
    y: np.ndarray
    Z: np.ndarray

    def sample(self, X, Y):
        xi = np.clip(np.interp(X, self.x, np.arange(self.x.size)), 0, self.x.size - 1)
        yi = np.clip(np.interp(Y, self.y, np.arange(self.y.size)), 0, self.y.size - 1)
        x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, self.x.size - 1)
        y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, self.y.size - 1)
        fx = xi - x0; fy = yi - y0
        Zi = (self.Z[y0, x0] * (1 - fx) * (1 - fy) + self.Z[y0, x1] * fx * (1 - fy)
              + self.Z[y1, x0] * (1 - fx) * fy + self.Z[y1, x1] * fx * fy)
        # normals from central differences of the source grid, then sampled
        dzdy, dzdx = np.gradient(self.Z, self.y, self.x)
        gx = (dzdx[y0, x0] * (1 - fx) * (1 - fy) + dzdx[y0, x1] * fx * (1 - fy)
              + dzdx[y1, x0] * (1 - fx) * fy + dzdx[y1, x1] * fx * fy)
        gy = (dzdy[y0, x0] * (1 - fx) * (1 - fy) + dzdy[y0, x1] * fx * (1 - fy)
              + dzdy[y1, x0] * (1 - fx) * fy + dzdy[y1, x1] * fx * fy)
        N = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
        N /= np.linalg.norm(N, axis=-1, keepdims=True)
        return Zi, N


# --------------------------------------------------------------------------
# occlusion
# --------------------------------------------------------------------------

@dataclass
class OcclusionMask:
    """
    Hardware self-occlusion profile: minimum visible elevation vs. azimuth,
    both in degrees, in the ROVER frame (azimuth measured clockwise from the
    rover's forward direction).

    A view direction is occluded when its elevation falls BELOW the critical
    elevation at its azimuth, i.e. the rover deck blocks steep downward looks.
    """
    az_deg: np.ndarray
    el_deg: np.ndarray
    name: str = "rover"

    def __post_init__(self):
        az = np.asarray(self.az_deg, dtype=float)
        el = np.asarray(self.el_deg, dtype=float)
        if az.size != el.size:
            raise ValueError("az_deg and el_deg must have equal length")
        # np.interp(period=...) requires no duplicated wrap point
        keep = np.ones(az.size, dtype=bool)
        if az.size > 1 and np.isclose(az[0] % 360.0, az[-1] % 360.0):
            keep[-1] = False
        order = np.argsort(az[keep] % 360.0)
        self.az_deg = (az[keep] % 360.0)[order]
        self.el_deg = el[keep][order]

    def critical_el(self, az_rover_deg: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(az_rover_deg) % 360.0,
                         self.az_deg, self.el_deg, period=360.0)

    def occluded(self, az_rover_deg, el_rover_deg) -> np.ndarray:
        return np.asarray(el_rover_deg) < self.critical_el(az_rover_deg)


#: Working profile carried over from the original error_map.py.
#: PROVENANCE UNVERIFIED -- confirm against rover CAD / ray-cast before publishing,
#: and note whether it is referenced to the mast rotation centre or to each camera
#: (the difference is a few degrees, which matters in the near field).
DEFAULT_ROVER_MASK = OcclusionMask(
    az_deg=[0, 100, 150, 160, 170, 185, 187, 205, 220, 240, 260, 310, 320, 360],
    el_deg=[-60, -60, -60, -30, -5, -5, -10, -10, -25, -30, -18, -18, -45, -60],
    name="M2020 working profile (unverified)",
)


# --------------------------------------------------------------------------
# instrument
# --------------------------------------------------------------------------

@dataclass
class Instrument:
    """
    An omnidirectional stereo imaging capability at a station.

    The station abstraction assumes a full 360-degree mosaic is acquired, so the
    stereo baseline is always horizontal and perpendicular to the line of sight,
    and every ray uses the mid-FOV IFOV.  This removes the need to model
    individual frames (5-100 stereo pairs per station).

    eye_offsets are fractions of the baseline along the baseline direction.
    (-0.5, +0.5) is a standard stereo pair; (0.0,) is a monocular station, which
    contributes only a rank-2 ray constraint (usable for cross-station
    triangulation, but with no intra-station range information).
    """
    name: str
    ifov_rad: float
    baseline_m: float
    eps_intra_px: float = 0.5
    eye_offsets: Tuple[float, ...] = (-0.5, 0.5)
    #: Optional per-eye occlusion masks. If None, the station mask is used for
    #: every eye. Length must match eye_offsets when supplied.
    eye_masks: Optional[List[Optional[OcclusionMask]]] = None
    #: How eps_intra_px is defined.
    #:   'per_image' -- eps is sigma_x', the precision of ONE image-coordinate
    #:                  measurement in ONE image along ONE axis.  This is the
    #:                  standard photogrammetric convention and the one used
    #:                  throughout this code.
    #:   'disparity' -- eps is sigma_d, the precision of the MEASURED DISPARITY
    #:                  from correlating two patches.  Most stereo-matching
    #:                  papers quote this.  If the two image measurements are
    #:                  independent then sigma_d = sqrt(2) sigma_x', so this
    #:                  option simply divides by sqrt(2) internally.
    #: The textbook sigma_Z = Z^2 sigma_d /(c b) hides that sqrt(2) inside
    #: sigma_d; the two conventions give identical answers once converted.
    precision_convention: str = "per_image"

    @property
    def sigma_x_px(self) -> float:
        """eps expressed as per-image, per-axis precision, whatever the convention."""
        if self.precision_convention == "disparity":
            return self.eps_intra_px / np.sqrt(2.0)
        if self.precision_convention != "per_image":
            raise ValueError("precision_convention must be 'per_image' or 'disparity'")
        return self.eps_intra_px

    def __post_init__(self):
        if self.eye_masks is not None and len(self.eye_masks) != len(self.eye_offsets):
            raise ValueError("eye_masks must match eye_offsets in length")


#: Mastcam-Z 34 mm.  Baseline 0.244 m from Hayes et al. 2021 pre-flight
#: calibration (24.3 +/- 0.1 cm).  IFOV is an interpolated working value between
#: the published 26 mm (283 urad) and 110 mm (67.4 urad) endpoints -- VERIFY.
MASTCAM_Z_34 = Instrument(
    name="Mastcam-Z 34 mm", ifov_rad=1.0 / 4720.0, baseline_m=0.244,
    eps_intra_px=0.169)
    # IFOV from the flight CAHVOR frame model (mppp/data/m20_cmods/ZL034_frame.xml,
    # ZR034 identical): f = 4720 px over 1648x1200, so 1/f = 2.119e-4 rad/px.
    # This REPLACES an earlier linear interpolation between the 26 mm and
    # 110 mm published endpoints (2.17e-4), which was flagged VERIFY.
MASTCAM_Z_110 = Instrument(
    name="Mastcam-Z 110 mm", ifov_rad=2.0 / (14852.0 + 14830.0), baseline_m=0.244,
    eps_intra_px=0.169)
    # ZL110 f = 14852 px, ZR110 f = 14830 px -> mean 1/f = 6.738e-5 rad/px.
    # Used for the far-target long-baseline pairs.
NAVCAM = Instrument(
    name="Navcam", ifov_rad=2.0 / (2950.913630357162 + 2943.646045723977),
    baseline_m=0.424, eps_intra_px=0.169)
    # eps_intra = 0.169 px: pooled over five Navcam sites, 1068 images, two SfM
    # packages (0.154-0.183 per site).  Replaces the 0.5 px working value.
    # NL0 f = 2950.91 px, NR0 f = 2943.65 px over 5120x3840 (full res)
    # -> mean 1/f = 3.393e-4 rad/px, from mppp/data/m20_cmods/. Replaces the
    # earlier 3.3e-4 working value flagged VERIFY.


# --------------------------------------------------------------------------
# station
# --------------------------------------------------------------------------

@dataclass
class Station:
    """
    One imaging station (waypoint).

    xyz     : (3,) position in the working frame [m], typically E,N,U centred on
              the anchor station.
    az_deg  : rover heading, degrees clockwise from North.
    path_m  : cumulative distance driven from the anchor along the traverse.
              Used by PoseModel in 'telemetry' mode.  If None it is filled in
              from inter-station straight-line distance, which underestimates
              true path length.
    """
    xyz: np.ndarray
    az_deg: float
    name: str = ""
    instrument: Instrument = field(default_factory=lambda: MASTCAM_Z_34)
    mask: Optional[OcclusionMask] = field(default_factory=lambda: DEFAULT_ROVER_MASK)
    path_m: Optional[float] = None
    #: Local mean solar time of the station's imaging [hours, 0-24].  Drives the
    #: illumination gate h(dL) = exp(-|L_i - L_j| / L0) between station pairs.
    #: None means "unknown -> assume co-temporal" (dL = 0, h = 1).
    lmst_h: Optional[float] = None
    is_anchor: bool = False

    def __post_init__(self):
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3)


# --------------------------------------------------------------------------
# pose covariance
# --------------------------------------------------------------------------

@dataclass
class PoseModel:
    """
    Relative-to-anchor pose uncertainty, propagated into object space as

        Sigma_pose,i = Sigma_t,i + [r_i]_x Sigma_w,i [r_i]_x^T

    The attitude term scales as r^2, so it dominates the far field absolutely.

    IMPORTANT MODELLING CHOICES
    ---------------------------
    * Errors are RELATIVE TO THE ANCHOR.  The anchor has zero pose covariance by
      definition (free-network / inner-constraint convention).  The common-mode
      part of pose error is a pure datum shift and does not degrade internal
      reconstruction quality, so it is correctly excluded here.
    * Interior orientation (principal distance, principal point, distortion
      residual) is NOT included.  It is common to every station, so modelling it
      per-station would let averaging remove it -- exactly wrong.  It belongs in
      rho_inf (the common-mode correlation floor) instead.
    * Pose errors are treated as independent between stations.  For telemetry
      mode this is a simplification: VO drift is cumulative and therefore
      strongly correlated between adjacent stations.
    * This is a two-stage approximation (pose fixed with known covariance, then
      propagated).  A rigorous treatment estimates pose and points jointly; the
      two-stage form is generally conservative.

    Modes
    -----
    'none'      : no pose error.  Pure network geometry.  Fast path -- skips all
                  per-station matrix inversions.
    'telemetry' : position sigma grows as vo_drift_frac * path_m; attitude fixed
                  at att_sigma_rad.
    'ba'        : post-bundle-adjustment values, constant per station.  In
                  reality Sigma_pose is an OUTPUT of the adjustment; use
                  representative numbers, or iterate (run with 'telemetry',
                  estimate BA precision from the resulting geometry, re-run).
    """
    #: 'none'       zero pose error -- the IDEAL ceiling.
    #: 'deadreckon' ~10% of PATH driven (IMU + wheel odometry).
    #: 'telemetry'  ~2-3% of PATH driven (visual odometry).  The FIXED case:
    #:              each station's product is registered to the common frame by
    #:              telemetry, so error follows the path, not the offset.
    #: 'sfm'        ~0.2% of straight-line BASELINE to the anchor.  The
    #:              SfM+MVS case: relative pose comes from cross-station ties.
    #: 'ba'         ~0.2% of PATH (legacy; 'sfm' is the correct scaling).
    #: 'registered' CONSTANT absolute registration error of a single-station
    #:              product placed in the common frame by telemetry.  MEASURED:
    #:              Metashape alignment vs telemetry references agree to
    #:              0.15-0.5 m at Belva (archive). Use for FIXED.  Path-based
    #:              'telemetry' is wrong for archive clusters spanning many sols
    #:              (dist_total_m accumulates whole excursions; Rockytop gave 454 m).
    mode: Literal["none", "telemetry", "ba", "sfm", "deadreckon", "registered"] = "none"
    reg_sigma_m: float = 0.3
    # telemetry.  Literature tiers for relative position error, as a fraction of
    # distance driven (1-sigma):
    #   DEAD RECKONING (IMU + wheel odometry)   ~10%   MER design goal was
    #        "at most 10% error"; JPL Mars Yard tests report wheel odometry "not
    #        better than 10% of distance traveled", worse on slopes.
    #   VISUAL ODOMETRY                         ~2-3%  MER VO reduced error to
    #        ~3% of range walked; JPL 25 m Mars Yard runs ended below 2.5%.
    #        M2020 AutoNav is quoted at ~2% per 100 m with reliable feature
    #        tracking; independent stereo-VO analysis of Perseverance found
    #        10-30 cm differences from telemetry on longer drives.
    #   BUNDLE ADJUSTMENT (ground-in-the-loop)  ~0.2% of distance traveled.
    # These are RELATIVE-TO-ANCHOR values, which is what the model needs.
    vo_drift_frac: float = 0.03          # fraction of distance driven (1-sigma)
    vo_min_sigma_m: float = 0.01         # floor, so adjacent stations are not exact
    att_sigma_rad: float = 2.0e-3        # rover attitude + mast pointing, in quadrature
    # bundle adjustment
    ba_drift_frac: float = 0.002         # ~0.2%: of PATH in 'ba', of BASELINE in 'sfm'
    tie_range_m: float = 20.0            # typical tie-point range, for sigma_att = sigma_pos / r_tie
    ba_pos_sigma_m: float = 0.02         # floor
    ba_att_sigma_rad: float = 2.0e-4
    deadreckon_drift_frac: float = 0.10  # ~10%

    def station_sigmas(self, st: Station) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sigma_t (3,), sigma_w (3,)) 1-sigma diagonals for one station."""
        if self.mode == "none" or st.is_anchor:
            return np.zeros(3), np.zeros(3)
        p = 0.0 if st.path_m is None else float(st.path_m)
        if self.mode == "registered":
            return np.full(3, self.reg_sigma_m), np.full(3, self.att_sigma_rad)
        if self.mode == "sfm":
            # SfM+MVS: relative pose comes from BUNDLE ADJUSTMENT on the
            # cross-station ties, so it scales with the STRAIGHT-LINE
            # separation from the anchor, NOT the path driven.  (Telemetry
            # scales with path: station 3_1266 is 18 m away but 167 m driven.)
            # ~0.2% of baseline is the ground-in-the-loop literature tier.
            anc = getattr(self, "_anchor_xyz", None)
            b = float(np.linalg.norm(np.asarray(st.xyz) - anc)) if anc is not None else p
            s_t = max(self.ba_drift_frac * b, self.ba_pos_sigma_m)
            # attitude follows from position over the typical tie range:
            # a station located to s_t by ties at r_tie has orientation to
            # s_t/r_tie.  One rule, no separate parameter.
            return np.full(3, s_t), np.full(3, max(s_t / self.tie_range_m,
                                                   self.ba_att_sigma_rad))
        if self.mode == "telemetry":
            return (np.full(3, max(self.vo_drift_frac * p, self.vo_min_sigma_m)),
                    np.full(3, self.att_sigma_rad))
        if self.mode == "deadreckon":
            return (np.full(3, max(self.deadreckon_drift_frac * p, self.vo_min_sigma_m)),
                    np.full(3, self.att_sigma_rad))
        if self.mode == "ba":
            return (np.full(3, max(self.ba_drift_frac * p, self.ba_pos_sigma_m)),
                    np.full(3, self.ba_att_sigma_rad))
        raise ValueError(f"unknown pose mode {self.mode!r}")

    def bind_anchor(self, stations) -> "PoseModel":
        """Record the anchor position; needed by mode='sfm' (baseline scaling)."""
        self._anchor_xyz = None
        for s_ in stations:
            if s_.is_anchor:
                self._anchor_xyz = np.asarray(s_.xyz, dtype=float)
                return self
        if len(stations):
            self._anchor_xyz = np.asarray(stations[0].xyz, dtype=float)
        return self

    def sigma_pose(self, st: Station, d: np.ndarray) -> Optional[np.ndarray]:
        """
        Object-space pose covariance at points offset d = p - X0 from the station.
        d has shape (...,3); returns (...,3,3), or None if identically zero.
        """
        s_t, s_w = self.station_sigmas(st)
        if not np.any(s_t) and not np.any(s_w):
            return None
        S = np.zeros(d.shape[:-1] + (3, 3))
        if np.any(s_t):
            S += np.diag(s_t ** 2)
        if np.any(s_w):
            K = skew(d)
            S += np.einsum("...ij,jk,...lk->...il", K, np.diag(s_w ** 2), K)
        return S


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Solver configuration.

    Defaults produce a PURE GEOMETRY map: emission-angle gating disabled, no
    angular matchability gate, no error correlation.  Every optional term is
    off unless explicitly enabled, so the baseline map has no unvalidated
    parameters in it.
    """
    # --- always active -----------------------------------------------------
    apply_occlusion: bool = True
    cos_e_min: float = 0.0
        # Emission-angle cutoff, cos(e) below which a view is discarded.
        # 0.0 disables it.  0.34 is the classical ~70-degree grazing cutoff.
    prior_sigma_m: float = 1.0e3
        # Tikhonov / "no information" prior.  Keeps every precision matrix
        # invertible, including at points seen by zero or one ray.  Cells whose
        # sigma approaches this value are flagged unconstrained.

    # --- cross-station matchability (OFF by default) -----------------------
    # A station can only contribute to a FUSED solution if it can be tied to the
    # rest of the network.  That tie is a cross-station image match, whose
    # precision eps_cross is worse than the within-station eps_intra because the
    # two views differ in angle, scale, foreshortening and illumination.
    #
    #     eps_cross(theta, tau) = eps_cross_px
    #                             * exp((theta/theta_max)^m)
    #                             * exp(ln^2(tau) / (2 s_tau^2))
    #     w_ij = (eps_intra / eps_cross)^2                clipped to <= 1
    #     w_i  = 1 - prod_j (1 - w_ij)     "is station i tied to anything?"
    #
    # w_ij is a PRECISION RATIO: if cross-station matching is 2x worse than
    # intra-station matching, that link carries 1/4 the weight.  Station i's
    # precision contribution is scaled by w_i.  The anchor always has w = 1
    # because it defines the datum.
    use_theta_gate: bool = False
    eps_cross_px: float = 0.5
        # cross-station matching precision at zero angular separation.
        #
        # NOTE: w_ij = (eps_intra/eps_cross)^2 is CLIPPED AT 1.  Setting
        # eps_cross BELOW eps_intra therefore changes nothing, and correctly so:
        # eps is a measurement precision in image space, and a cross-station
        # match cannot be more precise than the images themselves.  All the
        # geometry is already carried by the ray directions.
        # So eps_cross = eps_intra IS the perfect-matching case; eps_cross = 0
        # is not a distinct scenario.
    # ---- cross-station gate, MEASURED form (Navcam archive, five sites) ----
    #   w_ij = A * (1 + theta/theta_c)^(-k) * exp(-|dL|/L0),
    #   theta_c = theta_bar / CV^2,   k = 1 / CV^2
    # A plain-exponential gate is the CV -> 0 limit; heavier tails for CV > 0
    # come from a Gamma-distributed per-patch angular tolerance.  The old
    # Gaussian gate (exp[-2(theta/theta_max)^m]) is retained as gate_form=
    # "gaussian" for comparison; it was rejected at chi2/dof 38-855.
    gate_form: Literal["powerlaw", "gaussian"] = "powerlaw"
    gate_A: float = 0.4          # cross-station completeness at theta->0, dL->0, rel. to intra
    theta_bar_deg: float = 4.3   # mean per-patch angular tolerance
    gate_cv: float = 0.36        # spread of tolerance across the scene
    L0_h: float = 2.3            # YIELD e-fold in |dLMST| (cross-station stratum)
        # Two illumination scales, measured 2026-09-15, and they are NOT
        # interchangeable:
        #   L0     = 2.3 h  -- matched-pair COUNTS decay with this.  Use it for
        #                      completeness (holes / coverage).
        #   L_eff  = 1.46 h -- INFORMATION per possible pair, S/eps^2, decays
        #                      with this (pooled; LOSO 1.40-1.56; single
        #                      exponential at 0.061 dex, no sub-hour plateau --
        #                      the plateau reported earlier was an artefact of
        #                      describing completeness and precision separately).
        # Using L_eff for counts under-predicts them by the whole eps^2 factor.
        # An earlier analytic 1/L_eff = 1/L0 + 2/L_eps gave 1.09 h and was wrong
        # by 26 %: it used the same-station L0 (1.90 h) instead of the
        # cross-station one (2.32 h), and linearised (1+x)^-2 over x up to 1.5.
        # Separability S(theta,dL) = P(theta)h(dL) holds below ~8 deg
        # (L_eff 1.40-1.45 h); at 11 deg 1.21 h, at 20 deg 0.98 h.
    L_eff_h: float = 1.46        # information scale; use if weighting by S/eps^2
    L_eps_h: float = 5.1         # eps(dL) = eps0(1+dL/L_eps), object-space; 2nd order
    #: Matches within a cell CLUMP (a textured rock completeness ~20, sand completeness 0),
    #: so P(cell has no cross-station match) = exp(-mu/k) with clumping factor
    #: k = 9-40 at 1 m (measured).  The old hard threshold of 1 expected pair
    #: was ~10x too permissive.  The cross term is now weighted by the
    #: probability the cell actually has a tie, P = 1 - exp(-mu/k).
    clump_k: float = 20.0
    theta_max_deg: float = 20.0  # (gaussian form only)
    #: DISPLAY threshold for the coverage contour, in expected matched pairs.
    #: With the clumping model the natural choice is P_cross = 0.5, i.e.
    #: mu = k ln 2 ~ 14 for k = 20.  No longer used as a hard gate.
    min_expected_pairs: float = 14.0
        # Literature anchors: dense NCC/SGM ~10-20 deg; SIFT-class ~25-35 deg;
        # affine-covariant (ASIFT) ~45-60 deg; learned dense ~45-70 deg.
    theta_gate_exponent: float = 2.0
        # Shape exponent m in w = exp(-2 (theta/theta_max)^m).  FIXED at 2, not
        # fitted.  m is a SHAPE, so no other parameter can absorb it: at m=1
        # the plateau vanishes (a pair at 0.35*theta_max keeps weight 0.50
        # instead of 0.97), which contradicts eps_cross,0 = eps_intra, and the
        # information optimum moves from 0.71*theta_max to 1.0*theta_max --
        # i.e. theta_max would mean something 1.4x different.  m=2 and m=4 put
        # the optimum at the SAME place (2^-1/2 = 4^-1/4 = 0.71) and differ
        # only in shoulder softness, so 2 (a Gaussian shoulder) is the
        # conventional, safe choice.  Once the experiment measures the
        # survival curve S(theta), use it directly and m disappears.

    # --- transition-tilt gate (OFF by default) -----------------------------
    use_tau_gate: bool = False
    tau_max: float = 2.0
        # tau = max(cos e)/min(cos e), the ASIFT transition tilt (Morel & Yu
        # 2009).  For rover geometry this is more discriminating than theta
        # alone, because at grazing emission a small delta-theta produces a
        # large tilt change.

    # --- error correlation (OFF by default) --------------------------------
    use_correlation: bool = True
    correlation_kernel: Literal["cluster", "gaussian", "exponential", "cosine", "tilt"] = "exponential"
        # 'cluster'     THE SIMPLE N_eff MODEL (default).  Stations whose view
        #               directions at a cell fall within theta_c of each other
        #               are ONE look, at the precision of the best of them:
        #               the most precise member keeps its weight, the rest are
        #               dropped at that cell.  A bunch of k near-parallel
        #               stations therefore contributes exactly one ray;
        #               separated stations are untouched.  Applied to STATIONS, never to a station's
        #               own two eyes -- those are the inside of a single look
        #               and the classical range equation already treats their
        #               disparity noise as independent.  This is a covariance
        #               (redundancy) effect and is deliberately kept separate
        #               from eps_cross(theta), which is a variance (precision)
        #               effect; folding redundancy into eps is NOT equivalent
        #               (see test_cluster_neff_is_not_a_bandpass_eps).
        # 'gaussian'    exp(-th^2 / 2 th_c^2).  Conventional in geostatistics,
        #               but NEVER reaches zero, so with many stations tiny
        #               spurious correlations accumulate and suppress N_eff.
        # 'exponential' exp(-th / th_c).  Heavier tail still.
        # 'cosine'      max(cos(pi th / 2 th_c), 0)^2.  COMPACT SUPPORT: exactly
        #               zero beyond th_c, so distant views are treated as truly
        #               independent.  Default for that reason.
        # 'tilt'        based on transition tilt tau rather than angle, which is
        #               the quantity that actually governs affine matchability
        #               for grazing rover geometry.
        # The kernel SHAPE is a modelling assumption, not received wisdom from
        # the stereo literature.  It is swappable precisely because it is
        # unsettled; report results under at least two kernels.
    theta_min_deg: Optional[float] = None
        # Preferred name for the decorrelation angle theta_rho: the MINIMUM
        # convergence angle at which two stations count as separate looks.
        # With theta_max it brackets the useful baseline window:
        #     theta_min * r  <  b  <  theta_max * r
        # below it a second station is redundant, above it unmatchable.
        # None falls back to theta_c_deg (kept for backwards compatibility).
    theta_c_deg: float = 0.4   # measured decorrelation e-fold (0.14-0.81 deg per site)
        # decorrelation angle theta_rho.  2 deg is a PLACEHOLDER chosen so
        # that a cross-station pair is never treated as worth less than a
        # station's own intra-pair (0.42 m at 40 m = 0.6 deg, which the
        # classical range equation treats as independent); a 5 deg cut made
        # the spacing sweep flat-line at 1x below 5 m for exactly that reason.  Distinct from theta_max (matching
        # cutoff): shared sub-pixel bias decorrelates within a few degrees,
        # matching survives to tens of degrees, so theta_c << theta_max.
        # Unmeasured for Mars imagery; mppp_error.colmap.measure_theta_c estimates it.
        # Decorrelation angle.  Scaling argument theta_c ~ GSD / h_window gives
        # 12-23 deg for MCZ-34 at 25 m on rocky terrain -- UNMEASURED.
    rho_inf: float = 0.0       # floor for well-separated looks (~0 measured)
    rho_0: float = 0.22        # correlation at zero separation (repeat looks from one place)
        # Common-mode correlation floor: rho_inf = sigma_com^2 / sigma^2.
        # Puts a hard floor sigma_fused >= sqrt(rho_inf) * sigma_single.
        # Sweep {0, 0.05, 0.2}; 0 is the optimistic bound.

    # --- bookkeeping -------------------------------------------------------
    store_station_dirs: bool = False     # needed for correlation & gating
    on_no_visibility: Literal["raise", "warn", "ignore"] = "raise"
        # What to do when NO grid cell is visible from any station.  This is
        # almost always a frame error (cameras placed at or below the
        # evaluation surface), which otherwise fails silently as an all-NaN
        # map, so it raises by default.  Set to "ignore" for deliberately
        # fully-occluded test geometries.
    dtype: type = np.float64

    #: 'full'  -- all stations fuse freely (assumes perfect cross-station
    #:            correspondence).  Upper bound.
    #: 'gated' -- stations fuse with weight w_i from the eps_cross model.
    #: 'ops'   -- NO cross-station correspondence.  Each cell falls back to the
    #:            single best station, which is what operational fixed-baseline
    #:            stereo delivers.  (Note: precision addition is associative, so
    #:            "each station triangulates then average" is identical to "one
    #:            bundle".  The real difference is that without correspondence
    #:            you cannot combine stations at all.)
    #: 'cross'  -- CORRECT formulation (default for the gated cases).  Every
    #:            station's own intra-pair stereo enters at FULL weight (it
    #:            needs no cross-station match to exist); the gate weights ONLY
    #:            the cross-station coupling between station pairs.  The old
    #:            'gated' mode scaled a station's whole information by its tie
    #:            strength, which erased its own stereo and made the fusion
    #:            lose to the best single station on 99.8 % of cells.
    #: 'nearest' -- FIXED baseline: every cell takes the CLOSEST visible
    #:            station's own stereo, nothing else.
    link_mode: Literal["full", "gated", "ops", "cross", "nearest"] = "full"

    def any_pairwise(self) -> bool:
        return self.use_theta_gate or self.use_tau_gate or self.use_correlation

    @property
    def theta_min(self) -> float:
        """Decorrelation angle in degrees; theta_min_deg wins over theta_c_deg."""
        return self.theta_c_deg if self.theta_min_deg is None else self.theta_min_deg

    def needs_dirs(self) -> bool:
        return (self.any_pairwise() or self.store_station_dirs
                or self.link_mode in ("gated", "cross") or self.use_correlation)


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

@dataclass
class Grid:
    """Regular evaluation grid on a surface."""
    x: np.ndarray
    y: np.ndarray
    surface: Surface = field(default_factory=FlatPlane)

    @classmethod
    def square(cls, half_width_m: float, n: int, surface: Optional[Surface] = None) -> "Grid":
        v = np.linspace(-half_width_m, half_width_m, n)
        return cls(x=v, y=v.copy(), surface=surface or FlatPlane())

    @classmethod
    def covering(cls, stations: Sequence["Station"], margin_m: float = 15.0,
                 n: int = 201, surface: Optional[Surface] = None) -> "Grid":
        """Square grid sized to contain every station plus a margin."""
        P = np.array([s.xyz[:2] for s in stations])
        c = 0.5 * (P.max(axis=0) + P.min(axis=0))
        h = float(np.max(P.max(axis=0) - P.min(axis=0)) / 2 + margin_m)
        return cls(x=np.linspace(c[0]-h, c[0]+h, n),
                   y=np.linspace(c[1]-h, c[1]+h, n),
                   surface=surface or FlatPlane())

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.y.size, self.x.size)

    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = np.meshgrid(self.x, self.y)
        Z, N = self.surface.sample(X, Y)
        return np.stack([X, Y, Z], axis=-1), N

    def extent_edges(self) -> Tuple[float, float, float, float]:
        """imshow extent using cell EDGES, not centres."""
        dx = (self.x[-1] - self.x[0]) / max(self.x.size - 1, 1)
        dy = (self.y[-1] - self.y[0]) / max(self.y.size - 1, 1)
        return (self.x[0] - dx / 2, self.x[-1] + dx / 2,
                self.y[0] - dy / 2, self.y[-1] + dy / 2)


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------

@dataclass
class PrecisionField:
    """Fused covariance field and the per-station bookkeeping needed downstream.

    Extra attributes may be attached by analysis modules (e.g.
    `effective_baseline` from mppp_error.cases.lbs_field).
    """
    grid: Grid
    Sigma: np.ndarray            # (Ny,Nx,3,3) fused covariance [m^2]
    normals: np.ndarray          # (Ny,Nx,3)
    n_vis: np.ndarray            # (Ny,Nx) int, stations with >=1 visible ray
    n_rays: np.ndarray           # (Ny,Nx) int, total visible rays
    range_min: np.ndarray        # (Ny,Nx) range to nearest contributing station
    cos_e_ref: np.ndarray        # (Ny,Nx) cos(emission) at that nearest station
    sigma_n_single: np.ndarray   # (Nst,Ny,Nx) per-station-alone normal sigma
    station_vis: np.ndarray      # (Nst,Ny,Nx) bool
    station_dirs: Optional[np.ndarray]   # (Nst,Ny,Nx,3) or None
    link_weights: np.ndarray     # (Nst,Ny,Nx) cross-station tie weight w_i
    station_cos_e: np.ndarray    # (Nst,Ny,Nx)
    station_range: np.ndarray    # (Nst,Ny,Nx)
    stations: List[Station]
    config: ModelConfig
    pose: PoseModel
    unconstrained: np.ndarray
    completeness: Optional[np.ndarray] = None   # (Ny,Nx) expected matched cross-station pairs (the robotics-stereo term for the fraction of a scene that receives a valid match)
    p_cross: Optional[np.ndarray] = None       # (Ny,Nx) P(cell has >=1 cross tie) = 1-exp(-mu/k)
    # (Ny,Nx) bool

    @property
    def eps_intra_px(self) -> float:
        """Reference eps for normalisation; the min if stations differ."""
        vals = [s.instrument.eps_intra_px for s in self.stations]
        return float(min(vals)) if vals else float("nan")

    @property
    def ifov_ref(self) -> float:
        """
        Reference IFOV for G_n / GSD normalisation.

        With MIXED instruments (e.g. Z34 + Navcam in one network) there is no
        single pixel angle, so the FINEST is used: G_n then reads as "how many
        best-available-pixel footprints is the error", which is the meaningful
        benchmark currency for a mixed network.  Previously this returned NaN,
        which silently blanked the metric for exactly the mixed networks this
        experiment proposes.  `mixed_instruments` flags the case.
        """
        vals = [s.instrument.ifov_rad for s in self.stations]
        return float(min(vals)) if vals else float("nan")

    @property
    def mixed_instruments(self) -> bool:
        return len({s.instrument.ifov_rad for s in self.stations}) > 1


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------

def _station_ray_precision(st: Station, P: np.ndarray, N: np.ndarray,
                           cfg: ModelConfig):
    """
    Accumulate the rank-2 ray precisions for one station.

    Returns (Lambda (Ny,Nx,3,3), n_rays (Ny,Nx), vis_any (Ny,Nx),
             u_centre (Ny,Nx,3), r_centre (Ny,Nx), cos_e_centre (Ny,Nx)).
    """
    inst = st.instrument
    d0 = P - st.xyz
    r0 = np.linalg.norm(d0, axis=-1)
    u0 = d0 / np.maximum(r0, _EPS)[..., None]
    cos_e0 = -np.einsum("...i,...i->...", u0, N)

    # Baseline direction: horizontal, perpendicular to the horizontal LOS.
    # (The mast rotates to point at each target, so this holds for every cell.)
    hx, hy = d0[..., 0], d0[..., 1]
    hn = np.hypot(hx, hy)
    safe = hn > _EPS
    eb = np.zeros_like(d0)
    eb[..., 0] = np.where(safe, -hy / np.maximum(hn, _EPS), 1.0)
    eb[..., 1] = np.where(safe, hx / np.maximum(hn, _EPS), 0.0)

    Lam = np.zeros(P.shape[:-1] + (3, 3), dtype=cfg.dtype)
    n_rays = np.zeros(P.shape[:-1], dtype=np.int16)

    for k, frac in enumerate(inst.eye_offsets):
        C = st.xyz + frac * inst.baseline_m * eb
        d = P - C
        r = np.linalg.norm(d, axis=-1)
        u = d / np.maximum(r, _EPS)[..., None]
        cos_e = -np.einsum("...i,...i->...", u, N)

        vis = (r > _EPS) & (cos_e > max(cfg.cos_e_min, 0.0))

        if cfg.apply_occlusion:
            mask = st.mask
            if inst.eye_masks is not None and inst.eye_masks[k] is not None:
                mask = inst.eye_masks[k]
            if mask is not None:
                az_site = np.degrees(np.arctan2(u[..., 0], u[..., 1]))   # compass
                el_site = np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))
                az_rover = (az_site - st.az_deg) % 360.0
                vis &= ~mask.occluded(az_rover, el_site)

        sigma_perp = inst.sigma_x_px * inst.ifov_rad * r
        w = np.where(vis, 1.0 / np.maximum(sigma_perp, _EPS) ** 2, 0.0)

        I3 = np.eye(3)
        Lam += w[..., None, None] * (I3 - outer(u))
        n_rays += vis.astype(np.int16)

    return Lam, n_rays, n_rays > 0, u0, r0, cos_e0


def _fill_path_lengths(stations: Sequence[Station]) -> None:
    """Fill path_m from cumulative straight-line inter-station distance."""
    if all(s.path_m is not None for s in stations):
        return
    anchor_idx = next((i for i, s in enumerate(stations) if s.is_anchor), 0)
    cum = 0.0
    for i in range(anchor_idx, len(stations)):
        if i > anchor_idx:
            cum += float(np.linalg.norm(stations[i].xyz - stations[i - 1].xyz))
        if stations[i].path_m is None:
            stations[i].path_m = cum
    cum = 0.0
    for i in range(anchor_idx - 1, -1, -1):
        cum += float(np.linalg.norm(stations[i].xyz - stations[i + 1].xyz))
        if stations[i].path_m is None:
            stations[i].path_m = cum


def solve_precision_field(grid: Grid,
                          stations: Sequence[Station],
                          config: Optional[ModelConfig] = None,
                          pose: Optional[PoseModel] = None) -> PrecisionField:
    """
    Fuse per-station precision into a 3D covariance field over the grid.

    Fast path (pose.mode == 'none' and no correlation): all ray precisions are
    summed directly and inverted once per cell.  No per-station inversions.

    Pose path: each station's precision is inverted, Sigma_pose added, and
    re-inverted before summing.  Two extra batched 3x3 inversions per station.
    """
    cfg = config or ModelConfig()
    pm = pose or PoseModel()
    stations = list(stations)
    if not stations:
        raise ValueError("no stations supplied")
    if pm.mode == "telemetry":
        _fill_path_lengths(stations)

    P, N = grid.points()
    Ny, Nx = grid.shape
    Nst = len(stations)

    if pm.mode == "sfm" and pm._anchor_xyz is None:
        pm.bind_anchor(stations)

    prior = np.eye(3) / (cfg.prior_sigma_m ** 2)

    # ---- pass 1: per-station ray precision and geometry ------------------
    Lam_st = np.zeros((Nst, Ny, Nx, 3, 3), dtype=cfg.dtype)
    st_vis = np.zeros((Nst, Ny, Nx), dtype=bool)
    st_r = np.zeros((Nst, Ny, Nx))
    st_cos = np.zeros((Nst, Ny, Nx))
    n_rays_tot = np.zeros((Ny, Nx), dtype=np.int32)
    st_dirs = np.zeros((Nst, Ny, Nx, 3)) if cfg.needs_dirs() else None
    sig_n_single = np.full((Nst, Ny, Nx), np.inf)

    st_nray = np.zeros((Nst, Ny, Nx), dtype=np.int16)
    for i, st in enumerate(stations):
        Li, n_ray_i, vis_i, u_i, r_i, cos_i = _station_ray_precision(st, P, N, cfg)
        st_nray[i] = n_ray_i
        Lam_st[i] = Li
        st_vis[i] = vis_i
        st_r[i] = r_i
        st_cos[i] = cos_i
        n_rays_tot += n_ray_i.astype(np.int32)
        if st_dirs is not None:
            st_dirs[i] = u_i
        Sig_alone = sym_inv3(Li + prior)
        sn = np.sqrt(np.maximum(np.einsum("...i,...ij,...j->...", N, Sig_alone, N), 0.0))
        sig_n_single[i] = np.where(vis_i, sn, np.inf)

    # ---- cross-station coupling, and the YIELD field ---------------------
    # Precision and completeness are SEPARATE observables (archive, 2026-09): the gate
    # is a survival probability (a count of matches), not a precision penalty.
    # So the gate is NOT multiplied into Fisher information.  Instead:
    #   * intra-station stereo enters at full weight, always;
    #   * the cross-station coupling of a PAIR enters at full weight too --
    #     a matched tie is as precise as any other (epsilon is flat in theta);
    #   * the gate is accumulated separately as the expected number of matched
    #     cross-station pairs per cell, which is what decides whether the cell
    #     has a measurement at all.
    completeness = np.zeros((Ny, Nx))
    if st_dirs is not None and Nst > 1:
        for i in range(Nst):
            for j in range(i + 1, Nst):
                both = st_vis[i] & st_vis[j]
                if not both.any():
                    continue
                c = np.clip(np.einsum("...k,...k->...", st_dirs[i], st_dirs[j]), -1.0, 1.0)
                th = np.arccos(c)
                if cfg.gate_form == "powerlaw":
                    Sij = gate_powerlaw(th, cfg) * illumination_gate(
                        stations[i].lmst_h, stations[j].lmst_h, cfg.L0_h)
                else:
                    Sij = np.exp(-2.0 * (th / np.radians(cfg.theta_max_deg))
                                 ** cfg.theta_gate_exponent)
                # expected matched pairs: gate x number of possible image pairs
                npair = (len(stations[i].instrument.eye_offsets)
                         * len(stations[j].instrument.eye_offsets))
                completeness += np.where(both, Sij * npair, 0.0)

    p_cross = (1.0 - np.exp(-completeness / max(cfg.clump_k, 1e-9))) if Nst > 1 else np.zeros((Ny, Nx))

    # ---- cross-station link weights --------------------------------------
    if cfg.link_mode == "gated":
        w_st = _link_weights(st_dirs, st_cos, st_vis, stations, cfg, st_r=st_r, st_nray=st_nray, p_cross=p_cross)
    elif cfg.link_mode == "cross":
        # CORRECT cell model.  A point measured by its nearest station has that
        # station's own stereo (weight 1).  With probability p_cross the cell
        # holds a cross-station tie, in which case the point is ALSO seen by the
        # other stations and their rays add.  Summing every station's rays at
        # full weight is not "own stereo" -- it is full fusion, and assumes a
        # tie exists everywhere (that was the v0p11 mistake).
        #   E[Lambda] = Lambda_nearest + p_cross * sum_{j != nearest} Lambda_j
        # FIXED is the p_cross = 0 limit; BEST is the p_cross = 1 limit.
        usable = st_vis & (st_nray >= 2)
        r = np.where(usable, st_r, np.inf)
        idx = np.argmin(r, axis=0)
        w_st = np.zeros((Nst, Ny, Nx))
        for i in range(Nst):
            near_i = (idx == i) & usable[i]
            w_st[i] = np.where(near_i, 1.0, np.where(st_vis[i], p_cross, 0.0))
    elif cfg.link_mode == "nearest":
        # closest station that can actually deliver STEREO at the cell (both
        # eyes visible); a station seeing it with one eye is no baseline at all
        usable = st_vis & (st_nray >= 2)
        r = np.where(usable, st_r, np.inf)
        idx = np.argmin(r, axis=0)
        w_st = np.zeros((Nst, Ny, Nx))
        for i in range(Nst):
            w_st[i] = ((idx == i) & st_vis[i]).astype(float)
    elif cfg.link_mode == "ops":
        # No cross-station correspondence: each cell falls back to whichever
        # single station gives the best normal precision on its own.  This is
        # what operational fixed-baseline stereo delivers.
        w_st = np.zeros((Nst, Ny, Nx))
        best = np.argmin(np.where(np.isfinite(sig_n_single), sig_n_single, np.inf),
                         axis=0)
        any_vis = st_vis.any(axis=0)
        for i in range(Nst):
            w_st[i] = ((best == i) & any_vis).astype(float)
    else:
        w_st = np.ones((Nst, Ny, Nx))

    # ---- redundancy: the 'cluster' N_eff model --------------------------
    # Stations whose directions at a cell lie within theta_c of each other are
    # ONE look, at the precision of the BEST of them: the most precise member
    # keeps its full weight, the others are dropped.  (An even 1/k split is
    # wrong when members differ in precision -- it averages a Z34 with a
    # Navcam instead of keeping the Z34.)  Only stations that actually
    # contribute at the cell (visible, not gated or deselected) count.
    if cfg.use_correlation and cfg.correlation_kernel == "cluster" \
            and st_dirs is not None and Nst > 1:
        cth = np.cos(np.radians(cfg.theta_min))
        contrib = st_vis & (w_st > 1e-9)
        # per-cell information quality of each station's ray, incl. tie weight
        qual = np.zeros((Nst, Ny, Nx))
        for i, st in enumerate(stations):
            sp = st.instrument.sigma_x_px * st.instrument.ifov_rad * np.maximum(st_r[i], 1e-9)
            qual[i] = np.where(contrib[i], w_st[i] / sp ** 2, -1.0)
        # Greedy leader clustering, per cell: take stations in descending
        # quality; keep one unless it lies within theta_c of a station already
        # KEPT.  (Dropping a station because a *dropped* neighbour was better
        # lets a chain A-B-C-D collapse to one look although A and C are
        # independent -- caught by the spacing sweep flat-lining at 1.0x.)
        near = np.zeros((Nst, Nst, Ny, Nx), dtype=bool)
        for i in range(Nst):
            for j in range(Nst):
                if i != j:
                    c = np.einsum("...k,...k->...", st_dirs[i], st_dirs[j])
                    near[i, j] = (c >= cth) & contrib[i] & contrib[j]
        order = np.argsort(-qual, axis=0, kind="stable")      # best first
        kept = np.zeros((Nst, Ny, Nx), dtype=bool)
        for k in range(Nst):
            idx = order[k]                                     # (Ny,Nx)
            near_k = np.take_along_axis(near, idx[None, None], axis=0)[0]
            blocked = np.any(near_k & kept, axis=0)
            this_contrib = np.take_along_axis(contrib, idx[None], axis=0)[0]
            np.put_along_axis(kept, idx[None], (this_contrib & ~blocked)[None], axis=0)
        w_st = np.where(kept, w_st, 0.0)

    # ---- pass 2: pose, weighting, accumulation ---------------------------
    Lam_tot = np.zeros((Ny, Nx, 3, 3), dtype=cfg.dtype)
    n_vis = np.zeros((Ny, Nx), dtype=np.int16)
    r_min = np.full((Ny, Nx), np.inf)
    cos_e_ref = np.full((Ny, Nx), np.nan)

    for i, st in enumerate(stations):
        Li = Lam_st[i]
        vis_i = st_vis[i]
        if pm.mode != "none":
            S_pose = pm.sigma_pose(st, P - st.xyz)
            if S_pose is not None:
                Li = sym_inv3(sym_inv3(Li + prior) + S_pose) - prior
                Li = np.where(vis_i[..., None, None], Li, 0.0)
        Lam_tot += w_st[i][..., None, None] * Li
        n_vis += (vis_i & (w_st[i] > 1e-6)).astype(np.int16)
        upd = (st_r[i] < r_min) & vis_i
        r_min = np.where(upd, st_r[i], r_min)
        cos_e_ref = np.where(upd, st_cos[i], cos_e_ref)

    # ---- optional pairwise terms ----------------------------------------
    if cfg.use_correlation and cfg.correlation_kernel != "cluster" and st_dirs is not None:
        infl = _correlation_inflation(st_dirs, st_vis, cfg)
    else:
        infl = None

    Sigma = sym_inv3(Lam_tot + prior)
    if infl is not None:
        Sigma = Sigma * infl[..., None, None]

    unconstrained = (n_vis == 0) | (
        np.sqrt(np.maximum(np.trace(Sigma, axis1=-2, axis2=-1) / 3.0, 0.0))
        > 0.5 * cfg.prior_sigma_m)

    r_min = np.where(np.isfinite(r_min), r_min, np.nan)

    if not np.any(n_vis) and cfg.on_no_visibility != "ignore":
        z_st = np.array([s.xyz[2] for s in stations])
        z_gr = float(np.nanmedian(P[..., 2]))
        msg = (
            "no grid cell is visible from any station. Most often this is a "
            f"FRAME error: station z ranges {z_st.min():.2f}..{z_st.max():.2f} m "
            f"while the surface sits at z ~ {z_gr:.2f} m, so every emission "
            "angle is ~90 deg. Cameras must be ABOVE the evaluation surface. "
            "Other causes: cos_e_min set too high, or an occlusion mask that "
            "excludes the whole grid. Set ModelConfig(on_no_visibility='warn'"
            "|'ignore') if this is intentional.")
        if cfg.on_no_visibility == "raise":
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    return PrecisionField(
        grid=grid, Sigma=Sigma, normals=N, n_vis=n_vis.astype(int),
        n_rays=n_rays_tot.astype(int), range_min=r_min, cos_e_ref=cos_e_ref,
        sigma_n_single=sig_n_single, station_vis=st_vis, station_dirs=st_dirs,
        stations=stations, config=cfg, pose=pm, unconstrained=unconstrained,
        link_weights=w_st, completeness=completeness,
        p_cross=p_cross,
        station_cos_e=st_cos, station_range=st_r,
    )


def _correlation_inflation(u: np.ndarray, vis: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    """
    Variance inflation factor N / N_eff from the participation ratio.

        rho_ij  = rho_inf + (1 - rho_inf) * exp(-theta_ij^2 / (2 theta_c^2))
        N_eff   = N^2 / (1^T C 1)
        inflate = N / N_eff

    This is an approximation to full GLS (Sigma^-1 = J^T C^-1 J) that captures
    the saturation behaviour without any N x N inversion.  Spot-check against
    full GLS at a handful of cells before relying on it.

    NOTE: the exponential kernel is a modelling assumption imported from
    geodetic covariance practice.  The correlated-error MECHANISMS are well
    documented in the stereo literature (sub-pixel interpolation bias /
    pixel-locking, foreshortening bias, interior-orientation residual, surface
    definition ambiguity) but this parameterisation is not standard, and
    theta_c has not been measured for Mars surface imagery.
    """
    Nst = u.shape[0]
    tc = np.radians(cfg.theta_c_deg)
    Ny, Nx = u.shape[1], u.shape[2]
    num = np.zeros((Ny, Nx))
    Nvis = vis.sum(axis=0).astype(float)
    for i in range(Nst):
        for j in range(Nst):
            both = vis[i] & vis[j]
            if i == j:
                num += both
                continue
            c = np.clip(np.einsum("...k,...k->...", u[i], u[j]), -1.0, 1.0)
            rho = correlation_kernel(np.arccos(c), tc, cfg.rho_inf,
                                     cfg.correlation_kernel, rho_0=cfg.rho_0)
            num += np.where(both, rho, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        N_eff = np.where(num > 0, Nvis ** 2 / num, 0.0)
        infl = np.where(N_eff > 0, Nvis / N_eff, 1.0)
    return np.where(Nvis > 0, infl, 1.0)


def correlation_kernel(theta: np.ndarray, theta_c: float, rho_inf: float,
                       kind: str = "cosine",
                       rho_0: float = 1.0) -> np.ndarray:
    """
    rho(theta) = rho_inf + (rho_0 - rho_inf) * k(theta / theta_c)

    rho_0 is the correlation at ZERO separation.  Measured 0.22 on the archive
    (repeat frames from one place); it decays to ~0 within theta_rho ~ 0.4 deg,
    so rho_inf (the floor for well-separated looks) is ~0.  Setting rho_inf =
    0.22 instead -- the v0p9 mistake -- put a 22 % correlation on EVERY pair
    regardless of angle and made fused networks worse than a single station.

    rho is the CORRELATION between the errors of two observations separated by
    angle theta.  It is NOT derivable from eps and theta_c: eps is the variance
    of one measurement, rho is the shared fraction between two.  They are the
    diagonal and off-diagonal of one covariance function

        Cov[e(u_i), e(u_j)] = eps_i eps_j rho(theta_ij)

    and are independent inputs unless a generative model for e(u) is posited.
    """
    t = np.asarray(theta, dtype=float)
    if kind == "gaussian":
        k = np.exp(-t ** 2 / (2 * theta_c ** 2))
    elif kind == "exponential":
        k = np.exp(-t / theta_c)
    elif kind in ("cosine", "tilt"):
        k = np.cos(np.clip(np.pi * t / (2 * theta_c), 0.0, np.pi / 2)) ** 2
    else:
        raise ValueError(f"unknown correlation kernel {kind!r}")
    return rho_inf + (rho_0 - rho_inf) * k


def _link_weights(u: np.ndarray, cos_e: np.ndarray, vis: np.ndarray,
                  stations: Sequence[Station], cfg: ModelConfig,
                  st_r: Optional[np.ndarray] = None,
                  st_nray: Optional[np.ndarray] = None,
                  p_cross: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Per-station cross-station tie weight w_i, from the eps_cross model.

        w_ij = (eps_intra/eps_cross_px)^2
               * exp(-2 (theta_ij/theta_max)^m)          [angular gate]
               * exp(-ln^2(tau_ij) / (2 s_tau^2))        [transition-tilt gate]
        w_i  = 1 - prod_{j != i} (1 - w_ij)

    The transition tilt tau = max(cos e)/min(cos e) is the ASIFT quantity
    (Morel & Yu 2009).  For rover geometry it is more discriminating than theta
    alone, because at grazing emission a small change in view direction produces
    a large change in affine distortion.

    The anchor is forced to w = 1: it defines the datum and contributes whether
    or not anything ties to it.
    """
    Nst, Ny, Nx = vis.shape
    m = cfg.theta_gate_exponent
    th_max = np.radians(cfg.theta_max_deg)
    s_tau = max(np.log(max(cfg.tau_max, 1.0000001)) / 2.0, 1e-6)

    not_w = np.ones((Nst, Ny, Nx))
    for i in range(Nst):
        eps_i = stations[i].instrument.sigma_x_px
        eps_x = eps_i if cfg.eps_cross_px is None else cfg.eps_cross_px
        base = min((eps_i / max(eps_x, 1e-9)) ** 2, 1.0)
        for j in range(Nst):
            if i == j:
                continue
            w = np.full((Ny, Nx), base)
            if cfg.use_theta_gate:
                c = np.clip(np.einsum("...k,...k->...", u[i], u[j]), -1.0, 1.0)
                th = np.arccos(c)
                if cfg.gate_form == "powerlaw":
                    w = w * gate_powerlaw(th, cfg)
                    w = w * illumination_gate(stations[i].lmst_h, stations[j].lmst_h, cfg.L0_h)
                else:
                    w = w * np.exp(-2.0 * (th / th_max) ** m)
            if cfg.use_tau_gate:
                hi = np.maximum(cos_e[i], cos_e[j])
                lo = np.maximum(np.minimum(cos_e[i], cos_e[j]), 1e-9)
                w = w * np.exp(-np.log(hi / lo) ** 2 / (2 * s_tau ** 2))
            w = np.where(vis[i] & vis[j], np.clip(w, 0.0, 1.0), 0.0)
            not_w[i] *= (1.0 - w)
    w_i = 1.0 - not_w
    for i, st in enumerate(stations):
        if st.is_anchor:
            w_i[i] = 1.0
    return np.where(vis, w_i, 0.0)


def gate_powerlaw(theta_rad, cfg) -> np.ndarray:
    """
    Measured cross-station survival gate, A*(1+theta/theta_c)^(-k).

    theta_c = theta_bar/CV^2 and k = 1/CV^2, so the gate is parameterised by
    the two numbers that transfer between sites: the mean angular tolerance
    theta_bar and its scene-wide spread CV.  For CV->0 it tends to
    A*exp(-theta/theta_bar).  A is the completeness at zero angle relative to the
    intra-station rate; it multiplies the whole gate.
    """
    cv2 = max(cfg.gate_cv, 1e-6) ** 2
    th_c = np.radians(cfg.theta_bar_deg) / cv2
    k = 1.0 / cv2
    return cfg.gate_A * (1.0 + np.asarray(theta_rad) / th_c) ** (-k)


def illumination_gate(l_i, l_j, L0_h: float) -> float:
    """exp(-|dLMST|/L0) between two stations; 1 if either time is unknown."""
    if l_i is None or l_j is None or L0_h <= 0:
        return 1.0
    dL = abs(float(l_i) - float(l_j))
    dL = min(dL, 24.0 - dL)          # wrap
    return float(np.exp(-dL / L0_h))


def load_occlusion_profiles(path: Optional[str] = None) -> Dict[str, "OcclusionMask"]:
    """
    Per-camera, per-eye rover occlusion profiles from the delivered CSV
    (mppp/data/M2020_occlusion_profiles.csv).

    Columns: Az, then "<Cam> <Eye> Min El" for each of Zcam/Ncam x Left/Right.
    Each row gives, for that azimuth in the ROVER frame, the minimum elevation
    at which terrain is visible past rover hardware; below it the view is
    blocked. Hand-measured by CT from rover-frame az-el projected mosaics
    centred on the mast rotation axis, by reading where the rover meets the
    terrain.

    Returns e.g. {"zcam_left": OcclusionMask, ..., "ncam_right": ...}.

    NOTE: as delivered, the Zcam and Ncam columns are IDENTICAL -- one profile
    duplicated across cameras. That is expected for now (a single hand-tuned
    profile applied to both); the file format already supports per-camera
    profiles, so replacing the Ncam columns with their own measurements needs
    no code change.
    """
    import csv
    if path is None:
        from ..paths import data_file
        path = str(data_file("M2020_occlusion_profiles.csv"))
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"no rows in {path}")
    az = np.array([float(r["Az"]) for r in rows])
    out: Dict[str, OcclusionMask] = {}
    for col in rows[0]:
        if col == "Az":
            continue
        m = re.match(r"\s*(\w+)\s+(Left|Right)\s+Min\s+El\s*", col, re.I)
        if not m:
            continue
        key = f"{m.group(1).lower()}_{m.group(2).lower()}"
        el = np.array([float(r[col]) for r in rows])
        out[key] = OcclusionMask(az_deg=az.copy(), el_deg=el, name=key)
    return out


def eye_masks_for(camera: str, path: Optional[str] = None):
    """[left, right] OcclusionMasks for 'zcam' or 'ncam', for Instrument.eye_masks."""
    prof = load_occlusion_profiles(path)
    c = camera.lower()
    try:
        return [prof[f"{c}_left"], prof[f"{c}_right"]]
    except KeyError:
        raise KeyError(f"no profiles for camera {camera!r}; have {sorted(prof)}")


def choose_anchor(stations: Sequence[Station],
                  mode: str = "center") -> int:
    """
    Pick the datum station and set is_anchor accordingly.

    'first'  -- first in traverse order.  Natural when errors should be read as
                "how far has the network drifted from where we started".
    'center' -- closest to the geometric centre of the station set.  Minimises
                the maximum path length to any station, so telemetry pose error
                is smallest in the worst case.  Default.
    'site'   -- the station flagged is_anchor already, i.e. the SITE frame
                origin.  Use when tying to the mission site frame matters more
                than minimising internal error.

    Returns the index chosen.  Note the anchor carries ZERO pose covariance by
    the free-network convention: reported sigmas are relative to it.
    """
    if not stations:
        raise ValueError("no stations")
    if mode == "site":
        idx = next((i for i, s in enumerate(stations) if s.is_anchor), 0)
    elif mode == "first":
        idx = 0
    elif mode == "center":
        P = np.array([s.xyz[:2] for s in stations])
        idx = int(np.argmin(np.linalg.norm(P - P.mean(axis=0), axis=1)))
    else:
        raise ValueError("anchor mode must be first|center|site")
    for i, s in enumerate(stations):
        s.is_anchor = (i == idx)
    return idx
