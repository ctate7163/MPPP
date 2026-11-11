"""
mppp.ibr.rover_deck - the rover's top deck as a floor for the with-rover station products (0.96, Christian 10 Nov 2026:
"put a minimum height at the rover deck. Things can be above it, but not below ... It doesn't change, so it can be
hardwired for the with rover parts").

The deck is fixed in the rover frame, so it is hardwired here in the rover mechanical frame RMECH (aligned with the
rover navigation frame RNAV: +x forward, +y right, +z down; RMECH origin at RNAV (+90.02, 0, -1133.38) mm, its z
origin the top of the deck; RIMFAX CDR SIS 3.3.2, Table 7 - the frame of the deck fiducial survey).

**The footprint** (10 Nov, from the Van Zyl S003D2046 rover-shown cloud placed in RMECH by the label CAHVORE poses,
checked against the surveyed fiducials and the label Hazcam positions):

* top: RMECH z = 0 (the surveyed TDF1-3 sit at z = -0.4 to -0.6 mm; the fused deck points at a median 20 mm above);
* front edge x = +0.95 m: the deck stops where the body's front face drops (the front Hazcams at x = +1.0 m, 0.40 m
  below the deck top); the stowed arm and turret in front of it (x 1.0-1.35 m, at and below deck height) stay;
* rear x = -1.10 m: the rear Hazcams at x = -1.14 m, y = +-0.47 m, 0.40 m below the deck top; behind and above the
  rear of the deck the RTG and the aft structure rise above it (AFT1 0.29 m above) - above the floor, so kept;
* sides y = +-0.60 m: the deck points reach y = -0.65 / +0.59 m; the rockers and wheels outside (y >= 0.85 m) stay.

Inside the footprint nothing seen from the mast can be below the deck top, so the with-rover mesh is

1. cut: fused points inside the footprint and more than ``DECK_TOLERANCE_M`` below the top are dropped before meshing;
2. clamped: mesh vertices inside the footprint up to 10 cm below the top are lifted onto it (Poisson's sagging
   bubbles under the holes of the deck); faces with a vertex deeper than that are dropped;
3. floored: a grid of triangles at the deck top over the footprint is added to the mesh, so a ray through a hole in
   the reconstructed deck lands on the deck (coloured from the deck's own pixels) instead of the ground 1.1 m below.

**The rover in the block** (:func:`rover_pose`): every registered Navcam frame of the station has its label CAHVORE in
RNAV (the manifest's ``camera_model_label``) and its solved pose in the block, so each gives ``T_block_from_rnav``;
the station's estimate is the rotation mean and the median translation (Van Zyl: the 12 frames agree to 0.07-0.12 deg
and 1-5 mm).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("mppp.ibr.rover_deck")

MECH_IN_RNAV_M = np.array([0.09002, 0.0, -1.13338])     # the RMECH origin in RNAV (m)
DECK_TOP_MECH_Z_M = 0.0                                  # RMECH z of the top of the deck (+z down)
DECK_X_M = (-1.10, 0.95)                                 # RMECH x: rear, front edge
DECK_Y_M = (-0.60, 0.60)                                 # RMECH y: left, right
DECK_TOLERANCE_M = 0.03                                  # fused points this far below the top are dropped
DECK_GRID_M = 0.05                                       # the floor's triangle grid
DECK_JSON = "rover_deck.json"


def deck_corners_mech() -> np.ndarray:
    """The four footprint corners at the deck top in RMECH (m), counter-clockwise from rear-left."""
    (x0, x1), (y0, y1), z = DECK_X_M, DECK_Y_M, DECK_TOP_MECH_Z_M
    return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)


def label_pose_rnav(cm: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(R_cam_from_rnav, C_rnav) of a label CAHVOR(E) in ROVER_NAV_FRAME."""
    from ..cmod import CameraModel
    m = CameraModel.from_label(cm, width=cm.get("width"), height=cm.get("height"))
    _, R = m.camera_frame()
    return np.asarray(R, float), np.asarray(m.C, float)


def pose_from_pairs(R_cam_from_block: Sequence[np.ndarray], C_block: Sequence[np.ndarray],
                    R_cam_from_rnav: Sequence[np.ndarray], C_rnav: Sequence[np.ndarray]) -> Dict[str, Any]:
    """``T_block_from_rnav`` from frames with a pose in both frames: R the rotation mean of R_cb^T R_cr, t the median
    of C_b - R C_r.  Returns {"R_block_from_rnav", "t_block_from_rnav", "n", "rot_spread_deg", "pos_spread_mm"}."""
    from scipy.spatial.transform import Rotation as Rot
    Rs = np.array([np.asarray(a, float).T @ np.asarray(b, float) for a, b in zip(R_cam_from_block, R_cam_from_rnav)])
    if not len(Rs):
        raise ValueError("no frames with both poses")
    R = Rot.from_matrix(Rs).mean().as_matrix()
    ts = np.array([np.asarray(cb, float) - R @ np.asarray(cr, float) for cb, cr in zip(C_block, C_rnav)])
    t = np.median(ts, axis=0)
    dev = [float(np.degrees(np.linalg.norm(Rot.from_matrix(R.T @ Ri).as_rotvec()))) for Ri in Rs]
    return {"R_block_from_rnav": R, "t_block_from_rnav": t, "n": int(len(Rs)),
            "rot_spread_deg": float(np.max(dev)), "rot_rms_deg": float(np.sqrt(np.mean(np.square(dev)))),
            "pos_spread_mm": float(1e3 * np.max(np.linalg.norm(ts - t, axis=1)))}


def _manifest_models(project) -> Dict[str, Dict[str, Any]]:
    from ..sfm.workdir import load_manifest
    cands = [Path(project.settings.get("processed_dir", "")), Path(project.root).parent / "processed"]
    for d in cands:
        if d.is_dir() and list(d.glob("mppp_manifest_v*.json")):
            man, _ = load_manifest(d)
            out = {}
            for im in man.get("images", []):
                stem = (im.get("filename") or {}).get("stem") or Path(str(im.get("source_product", ""))).stem
                if im.get("camera_model_label"):
                    out[stem] = im["camera_model_label"]
            return out
    raise FileNotFoundError(f"no mppp_manifest_v*.json in {[str(c) for c in cands]}")


def rover_pose(project, rec, names: Iterable[str]) -> Dict[str, Any]:
    """The rover (RNAV) in the block at a station, from its registered frames' label CAHVOR(E) and solved poses."""
    models = _manifest_models(project)
    by_name = {im.name: im for im in rec.images.values()}
    Rb, Cb, Rr, Cr = [], [], [], []
    for n in names:
        im = by_name.get(n)
        cm = models.get(Path(n).stem)
        if im is None or cm is None or not im.has_pose:
            continue
        R_cr, C_r = label_pose_rnav(cm)
        cfw = im.cam_from_world()
        Rb.append(np.asarray(cfw.rotation.matrix(), float))
        Cb.append(np.asarray(im.projection_center(), float))
        Rr.append(R_cr)
        Cr.append(C_r)
    return pose_from_pairs(Rb, Cb, Rr, Cr)


def mech_to_block(X_mech: np.ndarray, pose: Dict[str, Any]) -> np.ndarray:
    R, t = np.asarray(pose["R_block_from_rnav"], float), np.asarray(pose["t_block_from_rnav"], float)
    return (np.asarray(X_mech, float) + MECH_IN_RNAV_M) @ R.T + t


def block_to_mech(X: np.ndarray, pose: Dict[str, Any]) -> np.ndarray:
    R, t = np.asarray(pose["R_block_from_rnav"], float), np.asarray(pose["t_block_from_rnav"], float)
    return (np.asarray(X, float) - t) @ R - MECH_IN_RNAV_M


def in_footprint(M: np.ndarray) -> np.ndarray:
    return (M[:, 0] >= DECK_X_M[0]) & (M[:, 0] <= DECK_X_M[1]) & (M[:, 1] >= DECK_Y_M[0]) & (M[:, 1] <= DECK_Y_M[1])


def below_deck(points: np.ndarray, pose: Dict[str, Any], tolerance_m: float = DECK_TOLERANCE_M) -> np.ndarray:
    """Points inside the footprint more than ``tolerance_m`` below the deck top (RMECH +z down)."""
    M = block_to_mech(points, pose)
    return in_footprint(M) & (M[:, 2] > DECK_TOP_MECH_Z_M + tolerance_m)


DECK_MAX_LIFT_M = 0.10                                   # a vertex deeper than this is not a sagging deck: its faces go


def clamp_to_deck(V: np.ndarray, pose: Dict[str, Any], F: Optional[np.ndarray] = None,
                  max_lift_m: float = DECK_MAX_LIFT_M):
    """Vertices inside the footprint below the deck top lifted onto it (along the rover's z).  With ``F``: only those
    within ``max_lift_m`` of the top are lifted, and the faces with a deeper vertex are dropped (10 Nov, Van Zyl:
    half of the 2,000 vertices below the top were 0.15-1.5 m down - ground and wheel surface under the footprint;
    lifted, the 219 faces that also reach outside it became sheets from the deck's edge down to the ground).
    Returns (V, n lifted), or (V, F, n lifted, n faces dropped) with ``F``."""
    M = block_to_mech(V, pose)
    ins = in_footprint(M)
    below = ins & (M[:, 2] > DECK_TOP_MECH_Z_M)
    deep = below & (M[:, 2] > DECK_TOP_MECH_Z_M + max_lift_m) if F is not None else np.zeros(len(M), bool)
    sel = below & ~deep
    V = np.array(V, float, copy=True)
    if sel.any():
        M[sel, 2] = DECK_TOP_MECH_Z_M
        V[sel] = mech_to_block(M[sel], pose)
    if F is None:
        return V, int(sel.sum())
    F = np.asarray(F)
    drop = deep[F].any(axis=1)
    return V, F[~drop], int(sel.sum()), int(drop.sum())


REAR_X_M = (-1.55, DECK_X_M[0])                          # behind the deck, under the RTG and the aft structure:
REAR_Y_M = (-0.45, 0.45)                                 # a second floor at the deck top's level (10 Nov; a rough
                                                         # stand-in until the rover model is in the mesh)


def deck_floor_arrays(pose: Dict[str, Any], step_m: float = DECK_GRID_M, x_m: Sequence[float] = DECK_X_M,
                      y_m: Sequence[float] = DECK_Y_M) -> Tuple[np.ndarray, np.ndarray]:
    """A floor: a grid of triangles at the deck top over ``x_m`` x ``y_m`` (default the deck's footprint), in the
    block frame, facing up."""
    xs = np.linspace(x_m[0], x_m[1], max(2, int(round((x_m[1] - x_m[0]) / step_m)) + 1))
    ys = np.linspace(y_m[0], y_m[1], max(2, int(round((y_m[1] - y_m[0]) / step_m)) + 1))
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    M = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, DECK_TOP_MECH_Z_M)], axis=1)
    nx, ny = len(xs), len(ys)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    # +z is down in RMECH: (a, d, c) / (a, c, b) wind so the normal points up (-z_mech)
    F = np.concatenate([np.stack([a, d, c], 1), np.stack([a, c, b], 1)]).astype(np.int64)
    return mech_to_block(M, pose), F


def deck_record(pose: Dict[str, Any], up: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """``rover_deck.json``: the pose, the footprint in both frames and the rover's attitude."""
    R = np.asarray(pose["R_block_from_rnav"], float)
    rec = {"frame": "RMECH (= RNAV axes, +x forward, +y right, +z down; origin RNAV (90.02, 0, -1133.38) mm; z 0 = deck top)",
           "deck_top_mech_z_m": DECK_TOP_MECH_Z_M, "x_m": list(DECK_X_M), "y_m": list(DECK_Y_M),
           "size_m": [DECK_X_M[1] - DECK_X_M[0], DECK_Y_M[1] - DECK_Y_M[0]],
           "corners_block_m": mech_to_block(deck_corners_mech(), pose).tolist(),
           "R_block_from_rnav": R.tolist(), "t_block_from_rnav": np.asarray(pose["t_block_from_rnav"], float).tolist(),
           "frames": pose.get("n"), "rot_spread_deg": pose.get("rot_spread_deg"), "pos_spread_mm": pose.get("pos_spread_mm")}
    if up is not None:
        u = np.asarray(up, float)
        u = u / np.linalg.norm(u)
        down = R[:, 2]
        rec["rover_tilt_deg"] = float(np.degrees(np.arccos(np.clip(-down @ u, -1, 1))))
    return rec


def write_deck(out_dir, pose: Dict[str, Any], up=None) -> Dict[str, Any]:
    rec = deck_record(pose, up)
    Path(out_dir, DECK_JSON).write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    return rec
