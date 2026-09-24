"""
Camera models: CAHVOR label -> pinhole + Brown distortion, Metashape XML
calibrations, and the pose of the camera in the local-level world frame.

Conventions used throughout MPPP
--------------------------------
* Camera axes: x right (increasing sample), y down (increasing line), z forward
  (OpenCV / COLMAP convention).  This is what CAHV gives directly:
  H' -> x, V' -> y, A -> z.
* Pixel coordinates (INTERNAL): the centre of the first pixel is (0, 0) — the
  CAHVOR and OpenCV convention.  Metashape and COLMAP put the *corner* of the
  first pixel at (0, 0); exporters add 0.5 px (``K_corner_origin``).
* Distortion (INTERNAL): OpenCV ordering and meaning,
  ``x' = x(1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)``.
  Metashape's P1/P2 are the OpenCV p2/p1 (swapped); this is handled on import.
* World frame: local East-North-Up (ENU), metres.  PDS site and rover-nav
  frames are North-East-Down (NED); ``P_NED_ENU`` converts (it is its own inverse).
* Rotations are world->camera matrices (rows = camera axes in world coords).
"""
from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation

PathLike = Union[str, Path]

P_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
DIST_KEYS = ("k1", "k2", "k3", "k4", "p1", "p2")


def quat_wxyz_to_rotation(q_wxyz) -> Rotation:
    """PDS quaternions are scalar-first; SciPy is scalar-last."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


@dataclass
class Intrinsics:
    width: int
    height: int
    K: np.ndarray                                   # 3x3, pixel-centre-origin convention
    dist: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in DIST_KEYS})
    source: str = "unknown"

    # -- derived -----------------------------------------------------------
    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def skew(self) -> float: return float(self.K[0, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])

    @property
    def K_corner_origin(self) -> np.ndarray:
        """K for software whose pixel (0,0) is the image corner (COLMAP, Metashape)."""
        K = self.K.copy()
        K[0, 2] += 0.5
        K[1, 2] += 0.5
        return K

    def opencv_dist(self) -> np.ndarray:
        """[k1, k2, p1, p2, k3] for cv2.  Metashape's k4 (r^8) has no OpenCV equivalent."""
        d = self.dist
        if d.get("k4", 0.0):
            warnings.warn("k4 (r^8 term) is non-zero and is ignored by the OpenCV model.")
        return np.array([d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]], dtype=np.float64)

    def is_distorted(self) -> bool:
        return any(abs(self.dist.get(k, 0.0)) > 0 for k in DIST_KEYS)

    # -- image-geometry edits ------------------------------------------------
    def shifted(self, dx: float, dy: float, width: int, height: int) -> "Intrinsics":
        """Image padded/cropped: origin moves by (dx, dy) pixels; new size given."""
        K = self.K.copy()
        K[0, 2] += dx
        K[1, 2] += dy
        return Intrinsics(int(width), int(height), K, dict(self.dist), self.source)

    def to_dict(self) -> Dict[str, Any]:
        return {"width": self.width, "height": self.height, "K": self.K.tolist(),
                "dist_opencv": dict(self.dist), "pixel_origin": "centre_of_first_pixel",
                "source": self.source}


@dataclass
class CAHVOR:
    C: np.ndarray
    A: np.ndarray
    H: np.ndarray
    V: np.ndarray
    O: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    model_type: str = "CAHV"

    @classmethod
    def from_label(cls, gcm: Dict[str, Any]) -> "CAHVOR":
        def comp(i):
            v = gcm.get(f"MODEL_COMPONENT_{i}")
            return None if v is None else np.asarray(v, dtype=np.float64)
        return cls(comp(1), comp(2), comp(3), comp(4), comp(5), comp(6),
                   str(gcm.get("MODEL_TYPE", "CAHV")))

    def project_cahv(self, X: np.ndarray) -> np.ndarray:
        """Linear (CAHV) projection of 3-D points (N,3), rover-nav frame -> (sample, line)."""
        d = np.atleast_2d(X) - self.C
        z = d @ self.A
        return np.stack([d @ self.H / z, d @ self.V / z], axis=1)

    def decompose(self, width: int, height: int):
        """
        -> (Intrinsics, R_cam_from_frame).  Standard CAHV decomposition
        (Di & Li 2004):  hs=|AxH|, vs=|AxV|, hc=A.H, vc=A.V,
        H'=(H-hc A)/hs, V'=(V-vc A)/vs.
        """
        A = self.A / np.linalg.norm(self.A)
        hs, vs = np.linalg.norm(np.cross(A, self.H)), np.linalg.norm(np.cross(A, self.V))
        hc, vc = float(A @ self.H), float(A @ self.V)
        Hp, Vp = (self.H - hc * A) / hs, (self.V - vc * A) / vs
        # angle between the image axes (pi/2 for square, unskewed pixels)
        theta = np.arccos(np.clip(Hp @ Vp, -1.0, 1.0))
        K = np.array([[hs * np.sin(theta), hs * np.cos(theta), hc],
                      [0.0, vs, vc],
                      [0.0, 0.0, 1.0]])
        M = np.linalg.inv(K) @ np.vstack([self.H, self.V, self.A])
        U, _, Vt = np.linalg.svd(M)                          # nearest rotation
        Rm = U @ Vt
        if np.linalg.det(Rm) < 0:
            raise ValueError("CAHV model is left-handed; cannot build a rotation.")

        dist = {k: 0.0 for k in DIST_KEYS}
        if self.R is not None:
            # CAHVOR: delta = r0 + r1 tau + r2 tau^2 with tau = tan^2(angle from O).
            # With O ~= A this is the Brown radial series; r0 is a scale term that is
            # absorbed by the focal length during self-calibration and dropped here.
            dist["k1"], dist["k2"] = float(self.R[1]), float(self.R[2])
        intr = Intrinsics(int(width), int(height), K, dist,
                          source=f"PDS label {self.model_type} (R1,R2 -> k1,k2; O~=A assumed)")
        return intr, Rm

    def o_a_angle_deg(self) -> Optional[float]:
        if self.O is None:
            return None
        c = (self.O @ self.A) / np.linalg.norm(self.O) / np.linalg.norm(self.A)
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def read_metashape_xml(path: PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Calibration XML not found: {path}")
    out: Dict[str, Any] = {}
    for child in ET.parse(path).getroot():
        text = (child.text or "").strip()
        try:
            out[child.tag] = float(text)
        except ValueError:
            out[child.tag] = text or None
    if out.get("projection", "frame") != "frame":
        raise ValueError(f"Only 'frame' calibrations are supported ({path})")
    return out


def intrinsics_from_metashape_xml(path: PathLike, scale: float = 1.0) -> Intrinsics:
    """
    Metashape calibration -> internal convention.

    Metashape:  u = w/2 + cx + x' f + x' b1 + y' b2 ;  v = h/2 + cy + y' f   (corner origin)
    with tangential terms P1 (r^2+2x^2) + 2 P2 x y in x  ->  OpenCV p2 = P1, p1 = P2.
    ``scale`` rescales the calibration to another resolution of the same
    detector (f, cx, cy, b1, b2 scale; distortion coefficients do not).
    """
    c = read_metashape_xml(path)
    g = lambda k: float(c.get(k) or 0.0)
    w, h = g("width") * scale, g("height") * scale
    f, b1, b2 = g("f") * scale, g("b1") * scale, g("b2") * scale
    cx = w / 2 + g("cx") * scale - 0.5
    cy = h / 2 + g("cy") * scale - 0.5
    K = np.array([[f + b1, b2, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    dist = {"k1": g("k1"), "k2": g("k2"), "k3": g("k3"), "k4": g("k4"),
            "p1": g("p2"), "p2": g("p1")}
    return Intrinsics(int(round(w)), int(round(h)), K, dist,
                      source=f"Metashape XML {Path(path).name} (scale {scale:g})")


@dataclass
class Pose:
    """Camera pose in the ENU world frame."""
    R_w2c: np.ndarray                 # 3x3 world(ENU) -> camera
    C: np.ndarray                     # camera centre, ENU metres
    frame: str                        # e.g. 'site3_enu' (landing frame) or 'site33_enu'
    position_source: str

    @property
    def t(self) -> np.ndarray:
        return -self.R_w2c @ self.C

    @property
    def boresight_az_el_deg(self):
        """Azimuth (clockwise from north) and elevation (up) of the optical axis."""
        e, n, u = self.R_w2c[2]
        return float(np.degrees(np.arctan2(e, n)) % 360.0), float(np.degrees(np.arcsin(np.clip(u, -1, 1))))

    def metashape_ypr_deg(self) -> np.ndarray:
        """
        Yaw, pitch, roll as used for Metashape reference import.  Formula kept
        verbatim from the validated pre-package code (``find_ypr_from_R_ref``).
        """
        R_cam_ned = self.R_w2c @ P_NED_ENU                  # back to world = NED
        R_ref = P_NED_ENU @ R_cam_ned
        Q = Rotation.from_matrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        ypr = (Q.inv() * Rotation.from_matrix(R_ref)).inv().as_euler("ZYX", degrees=True)
        if ypr[0] < 0:
            ypr[0] += 360.0
        return ypr

    def to_dict(self) -> Dict[str, Any]:
        az, el = self.boresight_az_el_deg
        return {"frame": self.frame, "position_source": self.position_source,
                "C_enu_m": self.C.tolist(), "R_world_to_cam": self.R_w2c.tolist(),
                "boresight_azimuth_deg": az, "boresight_elevation_deg": el,
                "metashape_ypr_deg": self.metashape_ypr_deg().tolist()}


def pose_from_label(R_cam_rnav: np.ndarray, C_rnav: np.ndarray,
                    q_rnav2site_wxyz, rover_origin_ned: np.ndarray,
                    frame: str, position_source: str) -> Pose:
    """
    ``rover_origin_ned``: position of the rover-nav origin in the world frame
    expressed as (north, east, down) metres.
    """
    R_rnav2site = quat_wxyz_to_rotation(q_rnav2site_wxyz).as_matrix()
    R_cam_ned = R_cam_rnav @ R_rnav2site.T                   # rows: camera axes in site NED
    C_ned = R_rnav2site @ np.asarray(C_rnav, float) + np.asarray(rover_origin_ned, float)
    return Pose(R_w2c=R_cam_ned @ P_NED_ENU, C=P_NED_ENU @ C_ned,
                frame=frame, position_source=position_source)
