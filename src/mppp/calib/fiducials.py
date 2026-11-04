"""
Rover deck fiducials: a metric check of the solved Navcam blocks (0.95.5, Christian 3 Nov 2026).

The rover top deck carries black-and-white quadrant targets whose positions were surveyed before launch in the
rover mechanical frame (RMECH, mm; point group "A::ROVER TOP DECK TARGETS").  RMECH is aligned with the rover
navigation frame RNAV (+x forward, +y right, +z down) with its origin at RNAV (+90.02, 0, -1133.38) mm (RIMFAX CDR
SIS section 3.3.2, Table 7), and the label CAHVORE models are in RNAV, so the targets project straight through them.

The six rigid targets (AFT1, RPFA1, RPFA2, TDF1-3) are one known object in the rover frame.  For every Navcam
stereo pair that sees them, with the block's own refined cameras and the block's NL -> NR relative pose:

* the NL pose in the rover frame is solved from the targets (one 6-DOF pose; NR placed by the block's rig);
* the reprojection RMS tests the block's intrinsics and rig at metric scale (``rig`` fit);
* freeing a scale of the rig translation measures the baseline against the survey (``scale_ppm``), freeing the
  translation vector and then the rotation gives the rig the deck sees (``dt_mm``, ``dR_mdeg``);
* triangulating the targets with the block's model and fitting a similarity to the survey gives the shape (mm)
  and scale (ppm) of the deck as the block sees it;
* LDIFF and RDIFF ride the differential bar, so only their 3-D separation (fixed) is compared;
* the NL pose in the rover frame together with its pose in the block gives the rover pose in the block at that
  moment (``rover_in_block``), the link the rover-frame estimator needs.

AFT2 is behind the hexagonal plate from the mast and is not used.  Pixel coordinates in the outputs are
full-resolution COLMAP pixels (corner origin); detection works on the native (product) image, centre origin.
Reads only; nothing in the blocks changes.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.calib.fiducials")
PathLike = Union[str, Path]

# ------------------------------------------------------------------------------------------------ the survey
SURVEY_MECH_MM: Dict[str, Tuple[float, float, float]] = {
    "AFT1": (-940.4864, 503.9503, -293.0409),
    "AFT2": (-822.5762, -464.8931, -132.1765),
    "LDIFF": (-142.8247, -454.8615, -53.5541),
    "RDIFF": (-141.6256, 455.0397, -53.8406),
    "RPFA1": (-319.8301, 455.9888, -79.8331),
    "RPFA2": (-461.5137, 505.1874, -170.5306),
    "TDF1": (595.1369, -231.5517, -0.4608),
    "TDF2": (520.3054, -0.5257, -0.5833),
    "TDF3": (378.6375, -381.1053, -0.399),
}
MECH_TO_NAV_MM = np.array([90.02, 0.0, -1133.38])      # RMECH origin in RNAV (RIMFAX CDR SIS 3.3.2, Table 7)
RIGID: Tuple[str, ...] = ("AFT1", "RPFA1", "RPFA2", "TDF1", "TDF2", "TDF3")
BAR: Tuple[str, ...] = ("LDIFF", "RDIFF")                # on the differential bar: only their separation is fixed
NOT_USED: Tuple[str, ...] = ("AFT2",)                    # behind the hexagonal plate from the mast
DECK_CTE_PPM_PER_DEGC = 23.0                             # aluminium; the survey is taken as a room-temperature one

# ------------------------------------------------------------------------------------------------ detection
SYM_RADIUS_FULL_PX = 14.0        # half-size of the point-symmetry window (full-resolution px; >= MIN_SYM_RADIUS)
MIN_SYM_RADIUS = 5               # native px
SEARCH_FULL_PX = 12.0            # first pass: +- around the label prediction (labels are 3-6 px off)
REFINE_SEARCH_FULL_PX = 6.0      # second pass: +- around the block solution's prediction
MAX_COST = 0.30                  # 1 - correlation of the window with its 180-degree rotation
MIN_SD = 8.0                     # DN: a flat (saturated) patch is symmetric too
MAX_CORNER_PX = 0.75             # native px: cv2.cornerSubPix used when it stays this close to the symmetry centre
EDGE_MARGIN = 20                 # native px: predictions closer to the frame edge are skipped
BAR_SEARCH_FULL_PX = 40.0        # LDIFF / RDIFF: the differential bar turns them 10-30 px from the label prediction
BAR_MAX_COST = 0.15              # and a wide search needs a stricter symmetry
REJECT_FULL_PX = 3.0             # a target further than this from the fitted solution is dropped
MIN_PAIR_POINTS = 3              # rigid targets over both eyes for a rig-held solution
MIN_EYE_POINTS = 4               # rigid targets in one eye for its own resection


def survey_nav_m(names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """Survey positions in RNAV, metres."""
    names = list(names) if names is not None else list(SURVEY_MECH_MM)
    return {k: (np.asarray(SURVEY_MECH_MM[k], float) + MECH_TO_NAV_MM) / 1000.0 for k in names}


def native_to_full(xy, scale: float) -> np.ndarray:
    """Native (product) pixel, centre origin -> full-resolution COLMAP pixel, corner origin."""
    return (np.asarray(xy, float) + 0.5) / float(scale)


def full_to_native(xy, scale: float) -> np.ndarray:
    return np.asarray(xy, float) * float(scale) - 0.5


def to_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)   # BGR (cv2.imread)
    return a.astype(np.float32)


def _sym_cost(g: np.ndarray, cx: int, cy: int, R: int) -> Tuple[float, float]:
    p = g[cy - R:cy + R + 1, cx - R:cx + R + 1]
    if p.shape != (2 * R + 1, 2 * R + 1):
        return 9.0, 0.0
    sd = float(p.std())
    if sd < MIN_SD:
        return 9.0, sd
    q = p[::-1, ::-1]
    pz, qz = p - p.mean(), q - q.mean()
    c = float((pz * qz).sum() / np.sqrt((pz * pz).sum() * (qz * qz).sum() + 1e-9))
    return 1.0 - c, sd


def detect_target(gray: np.ndarray, x: float, y: float, scale: float = 1.0,
                  search_full_px: float = SEARCH_FULL_PX, max_cost: float = MAX_COST) -> Optional[Dict[str, float]]:
    """
    The centre of a quadrant target near native pixel (x, y) (centre origin): the window position whose content
    best matches its own 180-degree rotation (point symmetry), to sub-pixel by phase correlation of the window with
    its rotation, then ``cv2.cornerSubPix`` (the centre is a saddle) where that stays within ``MAX_CORNER_PX``
    (``method`` "corner", else "symmetry").  None when no symmetric, contrasted window is found or the best one is
    on the search border.
    """
    import cv2
    g = gray if gray.dtype == np.float32 else gray.astype(np.float32)
    s = float(scale)
    R = max(MIN_SYM_RADIUS, int(round(SYM_RADIUS_FULL_PX * s)))
    S = int(round(search_full_px * s)) + 2
    x0, y0 = int(round(x)), int(round(y))
    C = np.full((2 * S + 1, 2 * S + 1), 9.0)
    SD = np.zeros_like(C)
    for j, dy in enumerate(range(-S, S + 1)):
        for i, dx in enumerate(range(-S, S + 1)):
            C[j, i], SD[j, i] = _sym_cost(g, x0 + dx, y0 + dy, R)
    j, i = np.unravel_index(int(np.argmin(C)), C.shape)
    cost = float(C[j, i])
    if cost > max_cost or not (0 < i < 2 * S and 0 < j < 2 * S):
        return None

    xi, yi = x0 + i - S, y0 + j - S
    # sub-pixel: the shift between the window and its 180-degree rotation is twice the offset of the centre
    p = g[yi - R:yi + R + 1, xi - R:xi + R + 1].astype(np.float64)
    win = cv2.createHanningWindow((p.shape[1], p.shape[0]), cv2.CV_64F)
    (sx, sy), _ = cv2.phaseCorrelate(np.ascontiguousarray(p[::-1, ::-1]), p, win)
    if abs(sx) > 2.0 or abs(sy) > 2.0:                  # no clean shift: the parabola through the cost instead
        def para(a, b, c):
            d = a - 2 * b + c
            return 0.0 if d <= 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))
        sx, sy = 2 * para(C[j, i - 1], C[j, i], C[j, i + 1]), 2 * para(C[j - 1, i], C[j, i], C[j + 1, i])
    xs, ys = xi + 0.5 * sx, yi + 0.5 * sy
    # the centre is a saddle: cv2.cornerSubPix is the most precise estimate when it stays at the symmetry centre
    # (strongly foreshortened targets can pull it onto the ring; then the symmetry centre is kept)
    w = max(3, R // 2)
    cs = cv2.cornerSubPix(g, np.array([[[xs, ys]]], np.float32), (w, w), (-1, -1),
                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001))[0, 0]
    dcs = float(np.hypot(cs[0] - xs, cs[1] - ys))
    method = "corner" if dcs <= MAX_CORNER_PX else "symmetry"
    if method == "corner":
        xs, ys = float(cs[0]), float(cs[1])
    return {"x": float(xs), "y": float(ys), "cost": cost, "sd": float(SD[j, i]), "corner_dev": dcs, "method": method}


# ------------------------------------------------------------------------------------------------ cameras
class Projector:
    """A block camera: camera-frame points -> full-resolution pixels (corner origin)."""

    def __init__(self, model: str, params: Sequence[float], width: int = 5120, height: int = 3840):
        self.model, self.params = str(model), [float(v) for v in params]
        self.width, self.height = int(width), int(height)
        self._pc = None
        if self.model != "THIN_PRISM_FISHEYE":
            import pycolmap
            self._pc = pycolmap.Camera(model=self.model, width=self.width, height=self.height, params=self.params)

    @classmethod
    def from_pycolmap(cls, cam) -> "Projector":
        name = getattr(cam.model, "name", str(cam.model)).split(".")[-1]
        return cls(name, list(cam.params), int(cam.width), int(cam.height))

    def __call__(self, Xc: np.ndarray) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        if self._pc is None:
            from .labelfit import project_thin_prism_fisheye
            return project_thin_prism_fisheye(self.params, Xc)
        out = np.asarray(self._pc.img_from_cam(Xc), float).reshape(-1, 2)
        out[~(Xc[:, 2] > 0)] = np.nan
        return out


def label_pose(cm: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(R_cam_from_rnav, t) of a label CAHVOR(E) in ROVER_NAV_FRAME (x_cam = R X + t)."""
    from ..cmod import CameraModel
    m = CameraModel.from_label(cm, width=cm.get("width"), height=cm.get("height"))
    _, R = m.camera_frame()
    C = np.asarray(m.C, float)
    return R, -R @ C


def label_predict(cm: Dict[str, Any], names: Sequence[str], margin: int = EDGE_MARGIN) -> Dict[str, np.ndarray]:
    """Native pixels (centre origin) of the targets through a label CAHVOR(E); only those inside the frame."""
    from ..cmod import CameraModel
    w, h = cm.get("width"), cm.get("height")
    m = CameraModel.from_label(cm, width=w, height=h)
    P = survey_nav_m(names)
    out = {}
    for k, X in P.items():
        uv = m.project(X[None])[0]
        inside = (w is None or margin <= uv[0] < w - margin) and (h is None or margin <= uv[1] < h - margin)
        if np.all(np.isfinite(uv)) and inside:
            out[k] = uv
    return out


# ------------------------------------------------------------------------------------------------ geometry
def _rot(v) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(v, float)).as_matrix()


def _rotvec(R) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def similarity(A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
    """Umeyama: B ~ s R A + t.  -> {"s", "R", "t", "rms" (in B's units), "residuals" (N, 3)}."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / ((a * a).sum() / len(A)))
    t = mb - s * R @ ma
    r = B - (s * (R @ A.T).T + t)
    return {"s": s, "R": R, "t": t, "rms": float(np.sqrt((r * r).sum(1).mean())), "residuals": r}


def _lsq(fun, x0, robust: bool = True):
    from scipy.optimize import least_squares
    return least_squares(fun, np.asarray(x0, float), loss="soft_l1" if robust else "linear", f_scale=1.0, x_scale="jac")


def _sd(res) -> np.ndarray:
    """Parameter sd of a least_squares result, scaled by the residual variance (NaN without redundancy)."""
    J = np.asarray(res.jac, float)
    m, n = J.shape
    if m <= n:
        return np.full(n, np.nan)
    s2 = float((res.fun ** 2).sum() / (m - n))
    try:
        return np.sqrt(np.clip(np.diag(np.linalg.inv(J.T @ J)) * s2, 0, None))
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)


MODES = ("rig", "scale", "translation", "free")
_N_EXTRA = {"rig": 0, "scale": 1, "translation": 3, "free": 6}


def solve_pair(projL: Optional[Projector], projR: Optional[Projector], obsL: Dict[str, np.ndarray],
               obsR: Dict[str, np.ndarray], R0: np.ndarray, t0: np.ndarray,
               R_rel: Optional[np.ndarray] = None, t_rel: Optional[np.ndarray] = None, mode: str = "rig",
               points: Optional[Dict[str, np.ndarray]] = None, reject_px: float = REJECT_FULL_PX) -> Optional[Dict[str, Any]]:
    """
    The NL pose in the rover frame (x_cam = R X + t) from the targets seen by NL (``obsL``) and NR (``obsR``),
    full-resolution pixels, with NR placed by the block's rig x_NR = R_rel x_NL + t_rel.  ``mode``:
    ``rig`` (rig held), ``scale`` (t_rel scaled by 1 + k), ``translation`` (t_rel + d), ``free`` (R_rel turned by
    w, t_rel + d).  A robust first fit, targets beyond ``reject_px`` dropped, then a plain refit.  None when there
    are fewer observations than unknowns + 1.  For one eye alone pass it as NL (R_rel unused).
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    P = points or survey_nav_m()
    obs = [("L", k, np.asarray(v, float)) for k, v in obsL.items() if k in P]
    if projR is not None and R_rel is not None:
        obs += [("R", k, np.asarray(v, float)) for k, v in obsR.items() if k in P]
    n_par = 6 + (_N_EXTRA[mode] if any(e == "R" for e, _, _ in obs) else 0)
    if mode != "rig" and not any(e == "R" for e, _, _ in obs):
        return None

    def poses(x):
        R = _rot(x[:3]) @ R0
        t = np.asarray(t0, float) + x[3:6]
        out = {"L": (R, t)}
        if R_rel is not None:
            Rr, tr = np.asarray(R_rel, float), np.asarray(t_rel, float)
            if mode == "scale":
                tr = tr * (1.0 + x[6])
            elif mode == "translation":
                tr = tr + x[6:9]
            elif mode == "free":
                Rr = _rot(x[6:9]) @ Rr
                tr = tr + x[9:12]
            out["R"] = (Rr @ R, Rr @ t + tr)
        return out

    def fun(x, use):
        ps = poses(x)
        r = []
        for e, k, uv in use:
            R, t = ps[e]
            pr = (projL if e == "L" else projR)((R @ P[k] + t)[None])[0]
            r.append(pr - uv if np.all(np.isfinite(pr)) else np.array([50.0, 50.0]))
        return np.concatenate(r) if r else np.zeros(0)

    use = list(obs)
    if 2 * len(use) < n_par + 1:
        return None
    x0 = np.zeros(n_par)
    res = _lsq(lambda x: fun(x, use), x0, robust=True)
    rr = res.fun.reshape(-1, 2)
    keep = [o for o, d in zip(use, np.hypot(rr[:, 0], rr[:, 1])) if d <= reject_px]
    dropped = [f"{e}:{k}" for (e, k, _), d in zip(use, np.hypot(rr[:, 0], rr[:, 1])) if d > reject_px]
    if 2 * len(keep) < n_par + 1:
        return None
    res = _lsq(lambda x: fun(x, keep), res.x, robust=False)
    rr = res.fun.reshape(-1, 2)
    ps = poses(res.x)
    sd = _sd(res)
    out: Dict[str, Any] = {"mode": mode, "x": res.x, "sd": sd, "n_obs": len(keep), "dropped": dropped,
                           "rms": float(np.sqrt((rr ** 2).sum(1).mean())),
                           "residuals": {f"{e}:{k}": rr[i] for i, (e, k, _) in enumerate(keep)},
                           "R": ps["L"][0], "t": ps["L"][1]}
    if "R" in ps:
        out["R_R"], out["t_R"] = ps["R"]
    if mode == "scale":
        out["scale_ppm"], out["scale_sd_ppm"] = 1e6 * float(res.x[6]), 1e6 * float(sd[6])
    if mode in ("translation", "free"):
        j = 6 if mode == "translation" else 9
        out["dt_mm"], out["dt_sd_mm"] = 1e3 * res.x[j:j + 3], 1e3 * sd[j:j + 3]
    if mode == "free":
        out["dR_mdeg"], out["dR_sd_mdeg"] = np.degrees(res.x[6:9]) * 1e3, np.degrees(sd[6:9]) * 1e3
    return out


def triangulate(projL: Projector, projR: Projector, R_rel: np.ndarray, t_rel: np.ndarray,
                uvL: np.ndarray, uvR: np.ndarray, X0: np.ndarray) -> Tuple[np.ndarray, float]:
    """A point in the NL camera frame from its two full-resolution observations; -> (X, rms px)."""
    from scipy.optimize import least_squares
    Rr, tr = np.asarray(R_rel, float), np.asarray(t_rel, float)

    def f(X):
        a = projL(X[None])[0] - uvL
        b = projR((Rr @ X + tr)[None])[0] - uvR
        r = np.r_[a, b]
        return np.where(np.isfinite(r), r, 50.0)

    res = least_squares(f, np.asarray(X0, float), x_scale="jac")
    return res.x, float(np.sqrt((res.fun ** 2).sum() / 2))


# ------------------------------------------------------------------------------------------------ one image pair
def rover_attitude_deg(R_world_from_rover: np.ndarray) -> List[float]:
    """Heading (deg clockwise from north), pitch (nose up +) and roll (right side down +) of the rover frame
    (x forward, y right, z down) in an east-north-up world (the block frame)."""
    R = np.asarray(R_world_from_rover, float)
    f, r, d = R[:, 0], R[:, 1], R[:, 2]
    heading = float(np.degrees(np.arctan2(f[0], f[1]))) % 360.0
    pitch = float(np.degrees(np.arcsin(np.clip(f[2], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(-r[2], -d[2])))
    return [heading, pitch, roll]


def check_pair(det: Dict[str, Dict[str, np.ndarray]], proj: Dict[str, Projector], label: Dict[str, Tuple[np.ndarray, np.ndarray]],
               cam_from_world: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """
    The metrics of one Navcam image pair (or one image).  ``det[eye]``: target -> full-resolution pixel
    (eye "L" / "R"); ``proj[eye]``: the block's camera; ``label[eye]``: the label (R, t) in RNAV (start values);
    ``cam_from_world[eye]``: the block pose (R, t).  Missing eyes are simply absent.
    """
    P = survey_nav_m()
    out: Dict[str, Any] = {"n_L": len([k for k in det.get("L", {}) if k in RIGID]),
                           "n_R": len([k for k in det.get("R", {}) if k in RIGID]),
                           "targets_L": ",".join(sorted(det.get("L", {}))), "targets_R": ",".join(sorted(det.get("R", {})))}
    rigidL = {k: v for k, v in det.get("L", {}).items() if k in RIGID}
    rigidR = {k: v for k, v in det.get("R", {}).items() if k in RIGID}
    # one-eye resections (intrinsics at metric scale, pose free)
    for e, ob in (("L", rigidL), ("R", rigidR)):
        if e in proj and e in label and len(ob) >= MIN_EYE_POINTS:
            r = solve_pair(proj[e], None, ob, {}, *label[e], mode="rig", points=P)
            if r:
                out[f"resect_rms_{e}"] = r["rms"]
                out[f"resect_n_{e}"] = r["n_obs"]
    pair = all(e in proj and e in cam_from_world for e in ("L", "R"))
    if not pair:
        return out
    (RL, tL), (RR, tR) = cam_from_world["L"], cam_from_world["R"]
    R_rel = RR @ RL.T
    t_rel = tR - R_rel @ tL
    out["baseline_block_mm"] = 1e3 * float(np.linalg.norm(t_rel))
    if len(rigidL) + len(rigidR) < MIN_PAIR_POINTS or "L" not in label:
        return out
    fits = {}
    for mode in MODES:
        fits[mode] = solve_pair(proj["L"], proj["R"], rigidL, rigidR, *label["L"], R_rel=R_rel, t_rel=t_rel, mode=mode, points=P)
    rig = fits["rig"]
    if rig is None:
        return out
    out.update({"rig_rms": rig["rms"], "rig_n_obs": rig["n_obs"], "rig_dropped": ";".join(rig["dropped"]),
                "_residuals": rig["residuals"], "_fit": rig})
    if fits["scale"]:
        out.update({"scale_ppm": fits["scale"]["scale_ppm"], "scale_sd_ppm": fits["scale"]["scale_sd_ppm"],
                    "scale_rms": fits["scale"]["rms"],
                    "baseline_fid_mm": out["baseline_block_mm"] * (1 + 1e-6 * fits["scale"]["scale_ppm"])})
    if fits["translation"]:
        d = fits["translation"]["dt_mm"]
        out.update({"dt_x_mm": d[0], "dt_y_mm": d[1], "dt_z_mm": d[2], "translation_rms": fits["translation"]["rms"]})
    if fits["free"]:
        w = fits["free"]["dR_mdeg"]
        ws = fits["free"]["dR_sd_mdeg"]
        out.update({"dR_pitch_mdeg": w[0], "dR_yaw_mdeg": w[1], "dR_roll_mdeg": w[2], "dR_pitch_sd": ws[0],
                    "dR_yaw_sd": ws[1], "dR_roll_sd": ws[2], "free_rms": fits["free"]["rms"]})
    # NL in the rover frame and the rover in the block
    R, t = rig["R"], rig["t"]
    C_rover = -R.T @ t
    out["nl_in_rover_C_mm"] = (1e3 * C_rover).tolist()
    out["nl_in_rover_rotvec"] = _rotvec(R).tolist()
    R_wr = RL.T @ R                       # rover -> world (block): X_w = RL^T (R X_r + t - tL)
    o_w = RL.T @ (t - tL)
    out["rover_in_block_origin_m"] = o_w.tolist()
    out["rover_in_block_hpr_deg"] = rover_attitude_deg(R_wr)
    out["rover_in_block_rotvec"] = _rotvec(R_wr).tolist()
    # triangulation with the block's model: deck shape and scale, the bar separation
    tri = {}
    for k in sorted(set(det.get("L", {})) & set(det.get("R", {}))):
        if k in NOT_USED:
            continue
        X, rms = triangulate(proj["L"], proj["R"], R_rel, t_rel, det["L"][k], det["R"][k], R @ P[k] + t)
        if rms <= REJECT_FULL_PX:
            tri[k] = X
    rig_tri = [k for k in tri if k in RIGID]
    out["tri_n"] = len(rig_tri)
    if len(rig_tri) >= 3:
        sim = similarity(np.array([P[k] for k in rig_tri]), np.array([tri[k] for k in rig_tri]))
        out["tri_scale_ppm"] = 1e6 * (sim["s"] - 1.0)
        out["tri_shape_rms_mm"] = 1e3 * sim["rms"]
    if all(k in tri for k in BAR):
        d = float(np.linalg.norm(tri["LDIFF"] - tri["RDIFF"]))
        d0 = float(np.linalg.norm(P["LDIFF"] - P["RDIFF"]))
        out["bar_mm"], out["bar_ppm"] = 1e3 * d, 1e6 * (d / d0 - 1.0)
    return out


# ------------------------------------------------------------------------------------------------ one block
PAIR_COLUMNS = ["block", "pair", "station", "sol", "T_L", "T_R", "scale_native", "n_L", "n_R", "targets_L", "targets_R",
                "label_rms_L", "label_rms_R", "resect_rms_L", "resect_rms_R", "rig_rms", "rig_n_obs", "rig_dropped",
                "baseline_block_mm", "baseline_fid_mm", "scale_ppm", "scale_sd_ppm", "dt_x_mm", "dt_y_mm", "dt_z_mm",
                "dR_pitch_mdeg", "dR_yaw_mdeg", "dR_roll_mdeg", "dR_pitch_sd", "dR_yaw_sd", "dR_roll_sd",
                "tri_n", "tri_scale_ppm", "tri_shape_rms_mm", "bar_mm", "bar_ppm",
                "nl_in_rover_C_mm", "nl_in_rover_rotvec", "rover_in_block_origin_m", "rover_in_block_hpr_deg",
                "rover_in_block_rotvec"]
DET_COLUMNS = ["block", "image", "eye", "sol", "target", "x_full", "y_full", "label_x_full", "label_y_full", "fit_x_full",
               "fit_y_full", "cost", "sd", "corner_dev", "method", "pass"]


def _manifest(work: Path) -> Dict[str, Dict[str, Any]]:
    cands = sorted((work / "processed").glob("mppp_manifest_v*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return {}
    d = json.loads(cands[-1].read_text(encoding="utf-8"))
    return {im["filename"]["stem"]: im for im in d.get("images", [])}


def _read_image(path: Path) -> Optional[np.ndarray]:
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if a is None else a


def check_block(work: PathLike, model: str = "cahv_ba", images_dir: Optional[PathLike] = None,
                out_dir: Optional[PathLike] = None, figures: int = 3,
                progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    The fiducial check of one solved block (``work``: the block folder, or its ``colmap`` folder).  Writes
    ``<out_dir>/<block>_pairs.csv`` and ``<block>_detections.csv`` (and annotated figures of up to ``figures``
    images with every rigid target found) when ``out_dir`` is given.  ``images_dir`` overrides where the
    images are read (default: the project's ``images`` folder).  -> {"block", "pairs" (rows), "detections"}.
    """
    import pycolmap
    w = Path(work)
    root = w / "colmap" if (w / "colmap" / "project.json").is_file() else w
    block_dir = root.parent if root.name == "colmap" else root
    name = block_dir.name
    say = progress or (lambda s: log.info(s))
    pj = json.loads((root / "project.json").read_text(encoding="utf-8"))
    rec = pycolmap.Reconstruction(str(root / "sparse" / model))
    by_name = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    man = _manifest(block_dir)
    idir = Path(images_dir) if images_dir else root / "images"
    rows = [r for r in pj.get("images") or [] if str(r.get("base_instrument") or r.get("instrument", ""))[:1] == "N"
            and r.get("name") in by_name]
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("sclk_key") or r["stem"]), {})[str(r.get("eye") or r["stem"][1])] = r
    names_all = list(RIGID) + list(BAR)
    pairs_out, det_out, figs = [], [], []
    for key, eyes in sorted(groups.items()):
        pred, det, proj, label, cfw, meta = {}, {}, {}, {}, {}, {}
        for e, r in eyes.items():
            m = man.get(r["stem"])
            cm = (m or {}).get("camera_model_label")
            if not cm:
                continue
            p = label_predict(cm, names_all)
            if len([k for k in p if k in RIGID]) < 2 and not all(k in p for k in BAR):
                continue
            ip = idir / r["name"]
            img = _read_image(ip) if ip.is_file() else None
            if img is None:
                continue
            s = float(r.get("downsample_scale") or 1.0)
            g = to_gray(img)
            found = {}
            for k, uv in p.items():
                d = (detect_target(g, uv[0], uv[1], s, search_full_px=BAR_SEARCH_FULL_PX, max_cost=BAR_MAX_COST)
                     if k in BAR else detect_target(g, uv[0], uv[1], s))
                if d:
                    found[k] = d
            im = by_name[r["name"]]
            T = im.cam_from_world()
            pred[e], det[e] = p, {k: native_to_full([d["x"], d["y"]], s) for k, d in found.items()}
            proj[e] = Projector.from_pycolmap(rec.cameras[im.camera_id])
            label[e] = label_pose(cm)
            cfw[e] = (np.asarray(T.rotation.matrix(), float), np.asarray(T.translation, float))
            meta[e] = {"row": r, "scale": s, "gray": g, "img": img, "found": found}
        if not meta:
            continue
        res = check_pair(det, proj, label, cfw)
        # second pass: targets missed in the first pass, searched around the block solution's prediction
        fit = res.get("_fit")
        if fit is not None:
            P = survey_nav_m(names_all)
            added = False
            for e in meta:
                R, t = (fit["R"], fit["t"]) if e == "L" else (fit.get("R_R"), fit.get("t_R"))
                if R is None:
                    continue
                for k in pred[e]:
                    if k in det[e] or k in BAR:
                        continue
                    uv = proj[e]((R @ P[k] + t)[None])[0]
                    d = detect_target(meta[e]["gray"], *full_to_native(uv, meta[e]["scale"]), meta[e]["scale"],
                                      search_full_px=REFINE_SEARCH_FULL_PX)
                    if d:
                        meta[e]["found"][k] = dict(d, refined=True)
                        det[e][k] = native_to_full([d["x"], d["y"]], meta[e]["scale"])
                        added = True
            if added:
                res = check_pair(det, proj, label, cfw)
        fit = res.pop("_fit", None)
        res.pop("_residuals", None)
        r0 = next(iter(meta.values()))["row"]
        row = {"block": name, "pair": key, "station": r0.get("station"), "sol": r0.get("sol"),
               "T_L": (meta.get("L") or {}).get("row", {}).get("camera_temperature_degC"),
               "T_R": (meta.get("R") or {}).get("row", {}).get("camera_temperature_degC"),
               "scale_native": r0.get("downsample_scale")}
        for e in meta:
            s = meta[e]["scale"]
            lab = [native_to_full(pred[e][k], s) - det[e][k] for k in det[e] if k in RIGID]
            if lab:
                row[f"label_rms_{e}"] = float(np.sqrt(np.mean(np.sum(np.square(lab), 1))))
        row.update(res)
        pairs_out.append(row)
        for e in meta:
            s = meta[e]["scale"]
            for k, d in meta[e]["found"].items():
                fx = fy = None
                if fit is not None:
                    R, t = (fit["R"], fit["t"]) if e == "L" else (fit.get("R_R"), fit.get("t_R"))
                    if R is not None:
                        fx, fy = proj[e]((R @ survey_nav_m([k])[k] + t)[None])[0]
                lx, ly = native_to_full(pred[e][k], s)
                xf, yf = det[e][k]
                det_out.append({"block": name, "image": meta[e]["row"]["stem"], "eye": e, "sol": meta[e]["row"].get("sol"),
                                "target": k, "x_full": xf, "y_full": yf, "label_x_full": lx, "label_y_full": ly,
                                "fit_x_full": fx, "fit_y_full": fy, "cost": d["cost"], "sd": d["sd"],
                                "corner_dev": d["corner_dev"], "method": d.get("method"),
                                "pass": 2 if d.get("refined") else 1})
            if out_dir and len(figs) < figures and all(k in meta[e]["found"] for k in RIGID):
                figs.append((e, meta[e], pred[e], det[e], fit))
        say(f"{name} {key}: L {res.get('n_L', 0)} R {res.get('n_R', 0)} targets, "
            f"rig rms {res.get('rig_rms', float('nan')):.2f} px, "
            f"scale {res.get('scale_ppm', float('nan')):+.0f} ppm")
        kept = {id(f[1]) for f in figs}
        for e in list(meta):
            meta[e].pop("gray", None)
            if id(meta[e]) not in kept:
                meta[e].pop("img", None)
    out = {"block": name, "pairs": pairs_out, "detections": det_out}
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        write_csv(o / f"{name}_pairs.csv", pairs_out, PAIR_COLUMNS)
        write_csv(o / f"{name}_detections.csv", det_out, DET_COLUMNS)
        for e, mm, pr, dt, fit in figs:
            try:
                annotate(mm["img"], mm["row"], mm["scale"], pr, dt, fit, e, o / f"{name}_{mm['row']['stem']}_fiducials.png")
            except Exception as exc:                    # noqa: BLE001 - a figure never stops the check
                log.warning("figure %s: %s", mm["row"]["stem"], exc)
        out["figures"] = len(figs)
    return out


def write_csv(path: PathLike, rows: List[Dict[str, Any]], columns: Sequence[str]) -> Path:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (json.dumps([round(float(x), 6) for x in v]) if isinstance(v, (list, tuple, np.ndarray))
                             else (round(v, 6) if isinstance(v, float) else v)) for k, v in r.items() if k in columns})
    return path


# ------------------------------------------------------------------------------------------------ figure
def annotate(img: np.ndarray, row: Dict[str, Any], scale: float, pred: Dict[str, np.ndarray], det: Dict[str, np.ndarray],
             fit: Optional[Dict[str, Any]], eye: str, out: PathLike, width: int = 2048) -> Path:
    """One image with the label prediction (red x) and the measurement (green o); insets of the six rigid targets
    with measured - label in full-resolution px (the block solution's residuals are in the pairs table)."""
    import cv2
    RED, GRN, YEL, WHT = (40, 40, 255), (60, 230, 60), (0, 220, 255), (255, 255, 255)
    h0, w0 = img.shape[:2]
    s = width / w0
    main = cv2.resize(img, (width, int(h0 * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    nat = {k: full_to_native(v, scale) for k, v in det.items()}

    def put(t_, pos, col, sc=0.8):
        cv2.putText(main, t_, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(main, t_, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)
    for k, uv in pred.items():
        x, y = int(uv[0] * s), int(uv[1] * s)
        cv2.drawMarker(main, (x, y), RED, cv2.MARKER_TILTED_CROSS, 22, 2)
        if k in nat:
            cv2.circle(main, (int(nat[k][0] * s), int(nat[k][1] * s)), 16, GRN, 2)
        put(k, (x + 20, y - 18) if x < width - 140 else (x - 110, y - 28), YEL)
    cv2.rectangle(main, (10, 10), (760, 120), (0, 0, 0), -1)
    cv2.drawMarker(main, (35, 40), RED, cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.putText(main, "predicted: survey through the label CAHVORE", (60, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHT, 1,
                cv2.LINE_AA)
    cv2.circle(main, (35, 85), 14, GRN, 2)
    cv2.putText(main, "measured: point-symmetry centre", (60, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHT, 1, cv2.LINE_AA)
    order = ["AFT1", "RPFA2", "RPFA1", "TDF3", "TDF2", "TDF1"]
    T = width // len(order)
    hh = max(12, int(round(50 * scale)))
    z = T / (2 * hh)
    tiles = []
    for k in order:
        c0 = nat.get(k, pred.get(k))
        if c0 is None:
            tiles.append(np.zeros((T, T, 3), np.uint8))
            continue
        cx, cy = int(round(c0[0])), int(round(c0[1]))
        pad = cv2.copyMakeBorder(img, hh, hh, hh, hh, cv2.BORDER_CONSTANT)
        c = cv2.resize(pad[cy:cy + 2 * hh, cx:cx + 2 * hh], (T, T), interpolation=cv2.INTER_CUBIC)
        if k in pred:
            p = ((pred[k][0] - cx + hh) * z, (pred[k][1] - cy + hh) * z)
            cv2.drawMarker(c, (int(p[0]), int(p[1])), RED, cv2.MARKER_TILTED_CROSS, 24, 2)
        txt = "not found"
        if k in nat:
            m = ((nat[k][0] - cx + hh) * z, (nat[k][1] - cy + hh) * z)
            cv2.drawMarker(c, (int(m[0]), int(m[1])), GRN, cv2.MARKER_CROSS, 24, 2)
            dl = det[k] - native_to_full(pred[k], scale) if k in pred else np.array([np.nan, np.nan])
            txt = f"meas-label ({dl[0]:+.1f}, {dl[1]:+.1f})"
        cv2.rectangle(c, (0, 0), (T, 30), (0, 0, 0), -1)
        cv2.rectangle(c, (0, T - 30), (T, T), (0, 0, 0), -1)
        cv2.putText(c, k, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YEL, 1, cv2.LINE_AA)
        cv2.putText(c, txt, (6, T - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHT, 1, cv2.LINE_AA)
        tiles.append(c)
    strip = np.hstack(tiles)
    strip = np.pad(strip, ((0, 0), (0, width - strip.shape[1]), (0, 0)))
    title = np.zeros((44, width, 3), np.uint8)
    res = "full resolution" if scale == 1 else f"1/{int(round(1 / scale))} resolution"
    cv2.putText(title, f"{row['stem'][:43]}  sol {row.get('sol')}, {res} - offsets in full-resolution px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHT, 1, cv2.LINE_AA)
    out = Path(out)
    cv2.imwrite(str(out), np.vstack([title, main, strip]))
    return out


# ------------------------------------------------------------------------------------------------ summary
def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def robust_mean(x: Sequence[float]) -> Dict[str, float]:
    """Median, mean, sd and standard error after a 3.5-MAD clip."""
    a = np.array([v for v in (_num(u) for u in x) if v is not None], float)
    if a.size == 0:
        return {"n": 0}
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    k = a[np.abs(a - med) <= 3.5 * mad] if mad > 0 else a
    return {"n": int(a.size), "n_kept": int(k.size), "median": med, "mean": float(k.mean()),
            "sd": float(k.std(ddof=1)) if k.size > 1 else float("nan"),
            "se": float(k.std(ddof=1) / np.sqrt(k.size)) if k.size > 1 else float("nan")}


SUMMARY_TERMS = ("label_rms_L", "label_rms_R", "resect_rms_L", "resect_rms_R", "rig_rms", "scale_ppm", "tri_scale_ppm",
                 "tri_shape_rms_mm", "dt_x_mm", "dt_y_mm", "dt_z_mm", "dR_pitch_mdeg", "dR_yaw_mdeg", "dR_roll_mdeg",
                 "bar_ppm")


def summarize(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Robust means of the pair metrics over all blocks and per block (pairs with a rig-held solution), the
    scale against the camera temperature, and the spread of the rover pose within each station."""
    use = [p for p in pairs if _num(p.get("rig_rms")) is not None]
    out: Dict[str, Any] = {"pairs": len(pairs), "solved": len(use),
                           "all": {t: robust_mean([p.get(t) for p in use]) for t in SUMMARY_TERMS}}
    out["blocks"] = {}
    for b in sorted({p["block"] for p in use}):
        bp = [p for p in use if p["block"] == b]
        out["blocks"][b] = {"solved": len(bp), **{t: robust_mean([p.get(t) for p in bp]).get("median") for t in
                                                  ("rig_rms", "scale_ppm", "tri_scale_ppm", "dR_roll_mdeg", "dR_yaw_mdeg")}}
    xs = [(_num(p.get("T_L")), _num(p.get("scale_ppm"))) for p in use]
    xs = [(a, b) for a, b in xs if a is not None and b is not None]
    if len(xs) >= 5:
        T, y = np.array(xs).T
        A = np.c_[np.ones_like(T), T - T.mean()]
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        cov = np.linalg.inv(A.T @ A) * float(r @ r) / max(len(y) - 2, 1)
        out["scale_vs_T"] = {"n": len(y), "T_mean": float(T.mean()), "ppm_at_T_mean": float(beta[0]),
                             "ppm_per_degC": float(beta[1]), "sd_ppm_per_degC": float(np.sqrt(cov[1, 1])),
                             "deck_cte_ppm_per_degC": DECK_CTE_PPM_PER_DEGC}
    st: Dict[str, Any] = {}
    for key in sorted({(p["block"], p.get("station")) for p in use}):
        sp = [p for p in use if (p["block"], p.get("station")) == key and p.get("rover_in_block_hpr_deg") is not None]
        if len(sp) >= 2:
            hpr = np.array([p["rover_in_block_hpr_deg"] for p in sp], float)
            hpr[:, 0] = (hpr[:, 0] - hpr[0, 0] + 180.0) % 360.0 - 180.0          # heading about the first pair
            org = np.array([p["rover_in_block_origin_m"] for p in sp], float)
            st[f"{key[0]}:{key[1]}"] = {"pairs": len(sp), "hpr_sd_mdeg": (1e3 * hpr.std(0, ddof=1)).tolist(),
                                        "origin_sd_mm": (1e3 * org.std(0, ddof=1)).tolist()}
    out["stations"] = st
    return out
