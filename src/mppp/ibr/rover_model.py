"""
mppp.ibr.rover_model - the Perseverance model in the station mesh (0.96.1, Christian 10 Nov 2026: "take this model
of perseverance's, simplify it, add it to the mesh, and project the images onto it. The arm is stowed").

``mppp/data/rover/perseverance_rmech.npz`` is NASA's glTF model "Mars Perseverance Rover, 3D Model" (credit
NASA/JPL-Caltech, https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/) with NASA's own triangles (197,539;
0.96.1-0.97.6: simplified to 59,000-71,000) in the rover mechanical frame RMECH (x forward, y right, z down, z 0 = the deck top; see
:mod:`mppp.ibr.rover_deck`), built by ``scripts/build_rover_model.py``.  What MPPP changed (docs/rover-model.md has
the numbers): the model is placed by seven deck fiducial discs (2.2 mm rms); the arm is turned to its stowed pose
from the arm joint angles of the image labels (:func:`arm_model_angles_deg`); the rear deck block, the high-gain
antenna (the part "antenna", one neutral pose) and the UHF antenna can are adjusted to the Navcam images; the mast
tube with its booms ("mast") and the camera head ("head", 0.97.7) are parts of their own and are not put into the
station mesh - the head turns with every image (:func:`head_in_mech`), and both cast shadows (:func:`shadow_casters`).  The model is one pose: a deployed
arm, the antenna's pointing of a sol and the suspension on rough ground are not followed.

With the rover shown, ``mppp mesh`` then
1. drops the fused points within ``NEAR_M`` of the model (the model stands for them) - what is farther stays and is
   meshed as before (the antenna, cables, anything the model lacks);
2. adds the model's triangles, placed in the block by the station's rover pose (:func:`mppp.ibr.rover_deck.rover_pose`),
   to ``mesh.ply``.
The images are then projected onto it like onto any other surface of the mesh.  The suspension and the wheels are in
the model's level-ground pose: on rough ground they are off by the rocker-bogie's articulation (``parts`` to leave
them out).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

FILE = Path(__file__).resolve().parents[1] / "data" / "rover" / "perseverance_rmech.npz"
NEAR_M = 0.04                     # fused points this close to the model are dropped
SAMPLE_SPACING_M = 0.012          # the model's surface sampled about this densely for the distance test
_cache: Dict[str, Any] = {}


def available() -> bool:
    return FILE.is_file()


def load(parts: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(vertices (n, 3) in RMECH metres, faces (m, 3), meta) of the model, or of the named ``parts`` of it (body,
    arm, arm_base, suspension, wheels, mast, antenna; ``meta["parts"]``)."""
    if "raw" not in _cache:
        d = np.load(FILE)
        _cache["raw"] = (d["vertices"].astype(float), d["faces"].astype(np.int64), d["part"], json.loads(str(d["meta"])))
    V, F, P, meta = _cache["raw"]
    if parts is not None:
        ids = [meta["parts"].index(p) for p in parts]
        F = F[np.isin(P, ids)]
    return V, F, meta


def surface_samples(parts: Optional[Sequence[str]] = None, spacing_m: float = SAMPLE_SPACING_M) -> np.ndarray:
    """Points on the model's surface about ``spacing_m`` apart (RMECH)."""
    key = ("samples", tuple(parts) if parts else None, spacing_m)
    if key not in _cache:
        V, F, _ = load(parts)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        n = int(max(1000, area.sum() / spacing_m ** 2))
        rng = np.random.default_rng(0)
        f = rng.choice(len(F), size=n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        _cache[key] = a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f])
    return _cache[key]


def near_model(points: np.ndarray, pose: Dict[str, Any], near_m: float = NEAR_M, parts: Optional[Sequence[str]] = None) -> np.ndarray:
    """The block-frame points within ``near_m`` of the model's surface."""
    from scipy.spatial import cKDTree
    from .rover_deck import block_to_mech
    M = block_to_mech(points, pose)
    S = surface_samples(parts)
    lo, hi = S.min(axis=0) - near_m, S.max(axis=0) + near_m
    box = np.all((M >= lo) & (M <= hi), axis=1)
    out = np.zeros(len(M), bool)
    if box.any():
        d, _ = cKDTree(S).query(M[box], distance_upper_bound=near_m + SAMPLE_SPACING_M)
        out[np.flatnonzero(box)] = d <= near_m + 0.5 * SAMPLE_SPACING_M
    return out


def model_in_block(pose: Dict[str, Any], parts: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices in the block frame, faces) of the model at the station's rover pose (only the vertices in use)."""
    from .rover_deck import mech_to_block
    V, F, _ = load(parts)
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return mech_to_block(V[used], pose), remap[F]


MAST_DILATE_FRAC = 0.012          # the mast mask grown by this share of the frame's width (pose and model error)
MAST_NEAR_M = 0.06                # fused points this close to the mast and its booms are dropped


def mast_masks(sources, pose: Dict[str, Any], dilate_frac: float = MAST_DILATE_FRAC):
    """Per source frame (:class:`mppp.ibr.ods.SourceImage`) the pixels of the mast and its wind-sensor booms (the
    model's part "mast"), True = hardware, or None where it is not in the frame (Christian, 10 Nov: "use this to
    predict the location of the wind booms and remove them from the 360").  The mast below the head is fixed to
    the rover - only the head turns - so the part is placed by the rover pose alone and its triangles are projected
    through each frame's own camera; at Van Zyl the white boom in the frames at azimuth 105 deg falls on the
    prediction.  The mask is grown by ``dilate_frac`` of the width."""
    import cv2
    from .rover_deck import mech_to_block
    V, F, _ = load(["mast"])
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    Vb, Fm = mech_to_block(V[used], pose), remap[F]
    out = []
    for src in sources:
        uv, z, _ = src.project(Vb)
        ok = np.isfinite(uv).all(axis=1) & (z > 0.02)
        big = 3.0 * max(src.W, src.H)
        ok &= (np.abs(uv) < big).all(axis=1)
        tri = Fm[ok[Fm].all(axis=1)]
        if not len(tri):
            out.append(None)
            continue
        p = uv[tri]                                              # (n, 3, 2)
        lo, hi = p.min(axis=1), p.max(axis=1)
        vis = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < src.W) & (lo[:, 1] < src.H) & ((hi - lo).max(axis=1) < 0.6 * max(src.W, src.H))
        if not vis.any():
            out.append(None)
            continue
        m = np.zeros((src.H, src.W), np.uint8)
        cv2.fillPoly(m, list(np.rint(p[vis]).astype(np.int32)), 255)
        if not m.any():
            out.append(None)
            continue
        r = max(3, int(round(dilate_frac * src.W)))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        out.append(m > 0)
    return out


# 0.97.0 (12 Nov 2026): NASA's model's arm joint angles from the labels' resolver angles (scripts/build_rover_model.py,
# where the signs and offsets are derived): model = sign * label + offset, degrees.
ARM_LABEL_SIGN = (1.0, -1.0, -1.0, -1.0, -1.0)
ARM_LABEL_OFFSET_DEG = (0.0, 0.0, -90.0, 90.0, 0.0)
ARM_LABEL_STOW_DEG = (90.09, -16.37, -161.66, 178.18, 278.52)
ARM_STOWED_WITHIN_DEG = 3.0


def arm_model_angles_deg(label_deg):
    """The model's five arm joint angles (deg, each in (-180, 180]) for the label's resolver angles (deg)."""
    import numpy as np
    a = np.asarray(ARM_LABEL_SIGN) * np.asarray(label_deg, float) + np.asarray(ARM_LABEL_OFFSET_DEG)
    return -((-a + 180.0) % 360.0 - 180.0)


def arm_is_stowed(label_deg, within_deg: float = ARM_STOWED_WITHIN_DEG) -> bool:
    """Whether every joint is within ``within_deg`` of the stow pose (the pose of the shipped model)."""
    import numpy as np
    d = (np.asarray(label_deg, float) - np.asarray(ARM_LABEL_STOW_DEG) + 180.0) % 360.0 - 180.0
    return bool(np.all(np.abs(d) <= within_deg))



# ---------------------------------------------------------------------------------------------------------------
# 0.97.7 (Christian, 13 Nov 2026: "find how the mast resolver angles relate to the mast's articulation in the 3d
# model"): the camera head (the part "head": the head, the Navcams, the Mastcam-Z, SuperCam's window and the
# microphone) turns about two axes; the mast tube and the wind-sensor booms (the part "mast") are fixed to the deck.
#
# The axes are those of the fitted mast chain (mppp_studies/calibration/data/deck_joint_mast_model_DJ32_20261107.json,
# 32 stations; the label's RSM resolver azimuth az and elevation el, ROVER_NAV_FRAME).  A point x of the head, in the
# head frame, is at
#
#     X_rnav = P + Rot(G) Rz(az) (h + Ry(-el) x)
#
# P: where the azimuth axis meets z = 0 (m); G: the axis' tilt and the azimuth zero (rotation vector, rad); h: the
# elevation axis from P in the azimuth-turned frame (m); Rz, Ry: right-handed rotations about z (down) and y.
# In the labels' convention az = 180 deg looks forward (+x), el = 90 deg is level, el = 0 looks straight down.
#
# NASA's model has the head at az = 180 deg, el = 90 deg (HEAD_MODEL_AZ_EL_DEG): looking forward, level.  Checked
# against the chain's Navcam centres at that pose: the model's two Navcam housings are centred 17 mm inside the
# chain's camera centres in y (the model's stereo base is short by about 34 mm, or the lenses are not centred in
# their housings), 1 mm in z, the lens front 14 mm ahead of the centres in x; the model's mast tube is centred 5 mm
# from the chain's azimuth axis.  So the head is good to about 2 cm - enough for its shadow, not for anything else.
MAST_P_XY_RNAV_M = (0.8011894511997478, 0.5568794458630825)
MAST_G_ROTVEC_RAD = (0.0015152142044422285, -0.0023839926438474776, -0.019291570441226168)
MAST_H_M = (-0.0003506674984743533, 0.0, -1.9177004015584642)
MAST_W_ROTVEC_RAD = (0.003913956929754023, 0.00266991321087316, -3.122261291130396)
NAVCAM_Q_ROTVEC_RAD = {"L": (-0.0571317210820025, -0.05495643684105014, 4.713989481400855),
                       "R": (0.017207252511186724, 0.01762287619051183, -1.5693821313696412)}
HEAD_MODEL_AZ_EL_DEG = (180.0, 90.0)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot(v) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(v, float)).as_matrix()


def head_transform_mech(az_deg: float, el_deg: float) -> np.ndarray:
    """The 4 x 4 rigid motion (RMECH) that takes the head from the model's pose to the label's RSM resolver angles
    ``az_deg``, ``el_deg`` (180, 90: forward and level - the identity)."""
    from .rover_deck import MECH_IN_RNAV_M
    P, G, h = np.r_[MAST_P_XY_RNAV_M, 0.0], _rot(MAST_G_ROTVEC_RAD), np.asarray(MAST_H_M)

    def pose(az, el):                                    # head frame -> RNAV: X = A x + b
        A = G @ _rz(az) @ _ry(-el)
        return A, P + G @ _rz(az) @ h
    A1, b1 = pose(np.radians(az_deg), np.radians(el_deg))
    A0, b0 = pose(*np.radians(HEAD_MODEL_AZ_EL_DEG))
    T = np.eye(4)
    T[:3, :3] = A1 @ A0.T
    T[:3, 3] = b1 - T[:3, :3] @ b0 + (T[:3, :3] - np.eye(3)) @ MECH_IN_RNAV_M      # RNAV -> RMECH: the same axes
    return T


def head_in_mech(az_deg: float, el_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices in RMECH, faces) of the camera head at the label's RSM resolver angles."""
    V, F, _ = load(["head"])
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    T = head_transform_mech(az_deg, el_deg)
    return V[used] @ T[:3, :3].T + T[:3, 3], remap[F]


def head_angles_deg(R_cam_from_rnav: np.ndarray, eye: Optional[str] = None) -> Tuple[float, float]:
    """The RSM (azimuth, elevation) in the labels' convention (deg) of a frame from its camera rotation in
    ROVER_NAV_FRAME - what the resolvers read, to the 9 mdeg of the chain's fit.  ``eye`` "L" / "R": a Navcam frame,
    through the chain (R_cam_from_rnav = Rot(Q_eye) Ry(el)^T Rz(az)^T Rot(W)); None: any camera of the head, from its
    optical axis alone (a Navcam's or Mastcam-Z's axis is within a few degrees of the head's)."""
    R = np.asarray(R_cam_from_rnav, float)
    if eye in NAVCAM_Q_ROTVEC_RAD:
        M = (_rot(NAVCAM_Q_ROTVEC_RAD[eye]).T @ R @ _rot(MAST_W_ROTVEC_RAD).T).T        # Rz(az) Ry(el)
        return float(np.degrees(np.arctan2(-M[0, 1], M[1, 1])) % 360.0), float(np.degrees(np.arctan2(-M[2, 0], M[2, 2])))
    a = R.T @ np.array([0.0, 0.0, 1.0])                  # the optical axis in RNAV (x forward, y right, z down)
    return (float((np.degrees(np.arctan2(a[1], a[0])) + 180.0) % 360.0),
            float(90.0 + np.degrees(np.arcsin(np.clip(-a[2], -1, 1)))))


SHADOW_PARTS = ("body", "arm", "arm_base", "suspension", "wheels", "antenna", "mast")


def shadow_casters(pose: Dict[str, Any], R_cam_from_block: Optional[np.ndarray] = None, eye: Optional[str] = None,
                   static_parts: Optional[Sequence[str]] = ("mast",), head: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """What of the rover casts a shadow at the time of one frame and is not in the station mesh (Christian, 13 Nov
    2026: "when rendering the shadows, make sure to include the rover mast and booms"): ``static_parts`` (the mast
    and its booms; with a terrain-only mesh the whole rover, :data:`SHADOW_PARTS`) and the camera head where the
    frame's own camera had it (``R_cam_from_block``, ``eye`` as :func:`head_angles_deg`).  (vertices in the block
    frame, faces, {"az_deg", "el_deg"})."""
    from .rover_deck import mech_to_block
    Vs, Fs, off, info = [], [], 0, {}
    if static_parts:
        V, F = model_in_block(pose, parts=tuple(static_parts))
        Vs.append(V), Fs.append(F)
        off += len(V)
    if head and R_cam_from_block is not None and "head" in load()[2]["parts"]:
        R_cr = np.asarray(R_cam_from_block, float) @ np.asarray(pose["R_block_from_rnav"], float)
        az, el = head_angles_deg(R_cr, eye)
        V, F = head_in_mech(az, el)
        Vs.append(mech_to_block(V, pose)), Fs.append(F + off)
        info = {"az_deg": az, "el_deg": el}
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), info
    return np.vstack(Vs), np.vstack(Fs), info
