"""
Frames of study 3A (rotations as ``R_a_b``: b coordinates -> a coordinates; centres as points in a frame):

- world ``w``: the block's frame (ENU of the landing site, minus the project offset; COLMAP's after the adjustment)
- rover nav frame ``rnav``: x forward, y right, z down; its attitude in the site frame (north-east-down) is the label's
  ``ROVER_COORDINATE_SYSTEM`` quaternion: ``R_rnav_w = R_site_rnav^T P`` with ``P`` = ENU -> NED
- RSM head frame ``head``: the label's ``RSM_COORDINATE_SYSTEM`` (head -> rnav quaternion, origin in rnav)
- camera ``c``: COLMAP's (x right, y down, z forward); the label's CAHV pose in rnav is ``R_c_rnav``, ``C_rnav``

A Hazcam is fixed to the body, so a Hazcam frame's adjusted pose gives the body pose:
``R_rnav_w = R_c_rnav^T R_c_w``, origin ``O_w = C_w - R_rnav_w^T C_rnav``.  A mast camera's attitude relative to
that body, in the head frame of its label, is a constant mount; what is left is the RSM pointing error plus flexure.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

P_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def quat_matrix(q_wxyz: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    w, x, y, z = (float(v) for v in q_wxyz)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def exp(w: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(w, float)).as_matrix()


def angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(rotvec(R))))


def mean_rotation(Rs: Iterable[np.ndarray]) -> np.ndarray:
    """The chordal L2 mean of rotations."""
    from scipy.spatial.transform import Rotation
    Rs = [np.asarray(R, float) for R in Rs]
    return Rotation.from_matrix(np.stack(Rs)).mean().as_matrix()


def telemetry_rnav_from_world(q_rnav2site: Sequence[float]) -> np.ndarray:
    """``R_rnav_w`` of the label attitude (world = ENU, site frames north-aligned)."""
    return quat_matrix(q_rnav2site).T @ P_NED_ENU


def body_from_camera(R_c_w: np.ndarray, C_w: np.ndarray, R_c_rnav: np.ndarray,
                     C_rnav: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(R_rnav_w, O_w) of the rover body from a body-fixed camera's pose and its pose in the rover frame."""
    R_rnav_w = np.asarray(R_c_rnav, float).T @ np.asarray(R_c_w, float)
    O_w = np.asarray(C_w, float) - R_rnav_w.T @ np.asarray(C_rnav, float)
    return R_rnav_w, O_w


def residual_rnav(R_rnav_w_est: np.ndarray, R_rnav_w_ref: np.ndarray) -> np.ndarray:
    """The body attitude residual as a rotation vector in the rover frame (rad): ``R_est = exp(w) R_ref``."""
    return rotvec(np.asarray(R_rnav_w_est, float) @ np.asarray(R_rnav_w_ref, float).T)


def camera_in_head(R_c_w: np.ndarray, R_rnav_w: np.ndarray, R_rnav_head: np.ndarray) -> np.ndarray:
    """``R_c_head`` of an adjusted camera, through a body attitude and the label's head frame."""
    return np.asarray(R_c_w, float) @ np.asarray(R_rnav_w, float).T @ np.asarray(R_rnav_head, float)


def head_residual(R_c_head_meas: np.ndarray, R_c_head_label: np.ndarray) -> np.ndarray:
    """The pointing residual in the head frame (rad): ``R_meas = R_label exp(e)`` (``e`` acts on head coordinates)."""
    return rotvec(np.asarray(R_c_head_label, float).T @ np.asarray(R_c_head_meas, float))


def point_in_frame(X_w: np.ndarray, R_f_w: np.ndarray, O_w: np.ndarray) -> np.ndarray:
    """A world point in the frame (R_f_w, origin O_w)."""
    return np.asarray(R_f_w, float) @ (np.asarray(X_w, float) - np.asarray(O_w, float))


def project_correction(prior_R_c_w: np.ndarray, R_c_rnav: np.ndarray, R_rnav_w_tel: np.ndarray) -> np.ndarray:
    """The rotation ``M`` with ``prior = M R_c_rnav R_rnav_w``: identity for the Navcam and Hazcams, the principal
    point correction for Mastcam-Z (``mppp.sfm.project.prior_rotation_correction``)."""
    return np.asarray(prior_R_c_w, float) @ (np.asarray(R_c_rnav, float) @ np.asarray(R_rnav_w_tel, float)).T


def fit_terms(az: np.ndarray, el: np.ndarray, T: Optional[np.ndarray], T0: float = -20.0) -> Tuple[np.ndarray, list]:
    """The pointing-law design matrix: a constant (the mount), cos / sin of the elevation (droop, a tilted
    elevation axis), cos / sin of the azimuth (azimuth bearing eccentricity, a tilted azimuth axis) and the
    temperature from ``T0``."""
    cols = [np.ones_like(az), np.cos(el), np.sin(el), np.cos(az), np.sin(az)]
    names = ["const", "cos_el", "sin_el", "cos_az", "sin_az"]
    if T is not None and np.isfinite(T).sum() >= 3 and np.nanstd(T) > 0.5:
        cols.append(np.where(np.isfinite(T), T - T0, 0.0))
        names.append("T")
    return np.stack(cols, axis=1), names
