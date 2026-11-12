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
import re as _re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp_studies.calibration.fiducials")
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
# Rigid targets without a survey (Christian, 3 Nov: "There is also a seventh TDF4 that is good but was not surveyed";
# "we can fit the tdf4 location and use it in the future"), in the rover navigation frame RNAV (Christian, 4 Nov:
# "work consistently in rover frame, not rmech").  TDF4 is the 4 Nov fit: the rays of 24 Navcam images (sols 31-1980,
# 16 NL + 8 NR, >= 5 surveyed targets each, resection rms <= 1 px, the law at each image's temperature) from poses
# resected on the surveyed targets, intersected (:func:`fit_point_from_rays`): RNAV (335.75, 126.60, -1133.43) mm,
# leave-one-image-out se (0.08, 0.08, 0.16) mm, ray rms 0.86 px.  The 9 images with all seven targets give
# (335.84, 126.64, -1133.37) +- (0.07, 0.11, 0.24); 43 images with >= 4, 5 dropped, (335.85, 126.92, -1133.76) +-
# (0.09, 0.13, 0.22).  Each check refits it: per pair from both eyes (``tdf4_rover_mm``, :func:`fit_unsurveyed`) and
# from every resected image (``rays``, :func:`fit_point_from_rays`).  Not used in the fits (not in RIGID) until promoted.
# 4 Nov, all seven (Christian: "Resolve the tdf4 location"): the 567 archive images with >= 4 of the seven predicted
# in frame, 80 rays (48 NL, 32 NR, sols 33-2006, >= 5 surveyed, rms <= 1 px): (335.83, 126.70, -1133.66) +- (0.05,
# 0.05, 0.09) mm, ray rms 0.87 px, none dropped; sols <= 1135 and later agree to 0.05 mm (the target does not move).
# One eye alone lands 2-4 mm along the line of sight (nearly parallel rays).  With the deck's focal-scale term freed
# (below the law by ~0.18 % at the deck) it moves by (+0.23, +0.27, -0.19) mm: known to ~0.3 mm, model-dependent.
UNSURVEYED_NAV_MM: Dict[str, Tuple[float, float, float]] = {"TDF4": (335.83, 126.70, -1133.66)}
RAY_MIN_TARGETS = 5              # an image's ray to an unsurveyed target needs a resection on this many surveyed ones
RAY_MAX_RMS = 1.0                # and at most this rms (full-resolution px)
UNSURVEYED: Tuple[str, ...] = tuple(UNSURVEYED_NAV_MM)
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
UNSURVEYED_SEARCH_FULL_PX = 20.0  # TDF4: around its prior (a few mm at 1-2 m, plus the label pose's 3-6 px)
REJECT_FULL_PX = 3.0             # a target further than this from the fitted solution is dropped
MIN_PAIR_POINTS = 3              # rigid targets over both eyes for a rig-held solution
MIN_EYE_POINTS = 4               # rigid targets in one eye for its own resection
MIN_ROTATION_POINTS = 6          # rigid targets over both eyes (>= 3 in NR) for the rig rotation with the translation held


def survey_nav_m(names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """Survey positions in RNAV, metres (the unsurveyed targets at their fitted RNAV positions when named)."""
    names = list(names) if names is not None else list(SURVEY_MECH_MM)
    return {k: (np.asarray(UNSURVEYED_NAV_MM[k], float) if k in UNSURVEYED_NAV_MM
                else np.asarray(SURVEY_MECH_MM[k], float) + MECH_TO_NAV_MM) / 1000.0 for k in names}


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
            from mppp.calib.labelfit import project_thin_prism_fisheye
            return project_thin_prism_fisheye(self.params, Xc)
        out = np.asarray(self._pc.img_from_cam(Xc), float).reshape(-1, 2)
        out[~(Xc[:, 2] > 0)] = np.nan
        return out


def label_pose(cm: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(R_cam_from_rnav, t) of a label CAHVOR(E) in ROVER_NAV_FRAME (x_cam = R X + t)."""
    from mppp.cmod import CameraModel
    m = CameraModel.from_label(cm, width=cm.get("width"), height=cm.get("height"))
    _, R = m.camera_frame()
    C = np.asarray(m.C, float)
    return R, -R @ C


def label_predict(cm: Dict[str, Any], names: Sequence[str], margin: int = EDGE_MARGIN) -> Dict[str, np.ndarray]:
    """Native pixels (centre origin) of the targets through a label CAHVOR(E); only those inside the frame."""
    from mppp.cmod import CameraModel
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


MODES = ("rig", "scale", "translation", "rotation", "free")
_N_EXTRA = {"rig": 0, "scale": 1, "translation": 3, "rotation": 3, "free": 6}


def solve_pair(projL: Optional[Projector], projR: Optional[Projector], obsL: Dict[str, np.ndarray],
               obsR: Dict[str, np.ndarray], R0: np.ndarray, t0: np.ndarray,
               R_rel: Optional[np.ndarray] = None, t_rel: Optional[np.ndarray] = None, mode: str = "rig",
               points: Optional[Dict[str, np.ndarray]] = None, reject_px: float = REJECT_FULL_PX) -> Optional[Dict[str, Any]]:
    """
    The NL pose in the rover frame (x_cam = R X + t) from the targets seen by NL (``obsL``) and NR (``obsR``),
    full-resolution pixels, with NR placed by the block's rig x_NR = R_rel x_NL + t_rel.  ``mode``:
    ``rig`` (rig held), ``scale`` (t_rel scaled by 1 + k), ``translation`` (t_rel + d), ``rotation`` (R_rel turned
    by w, t_rel held: the rig rotation the deck sees, 4 Nov - with the targets 1-2 m away a free translation trades
    with the rotation), ``free`` (R_rel turned by w, t_rel + d).  A robust first fit, targets beyond ``reject_px`` dropped, then a plain refit.  None when there
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
            elif mode == "rotation":
                Rr = _rot(x[6:9]) @ Rr
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
    if mode in ("rotation", "free"):
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
               cam_from_world: Dict[str, Tuple[np.ndarray, np.ndarray]], rig_ref: Optional[np.ndarray] = None,
               extra_rigid: Sequence[str] = ()) -> Dict[str, Any]:
    """
    The metrics of one Navcam image pair (or one image).  ``det[eye]``: target -> full-resolution pixel
    (eye "L" / "R"); ``proj[eye]``: the block's camera; ``label[eye]``: the label (R, t) in RNAV (start values);
    ``cam_from_world[eye]``: the block pose (R, t).  Missing eyes are simply absent.  ``rig_ref``: a reference
    rig rotation (the rig file's ``R_sensor_from_ref``, before its drift): the block's and the deck's rig
    rotations are then also given about it (``rig_block_*_mdeg``, ``rig_deck_*_mdeg``: pitch, yaw, roll), the
    form of the rig trends against sol.
    """
    P = survey_nav_m(list(SURVEY_MECH_MM) + list(UNSURVEYED))
    rig_set = set(RIGID) | set(extra_rigid)       # 4 Nov: extra_rigid=("TDF4",) uses the fitted TDF4 as a rigid target
    out: Dict[str, Any] = {"n_L": len([k for k in det.get("L", {}) if k in rig_set]),
                           "n_R": len([k for k in det.get("R", {}) if k in rig_set]),
                           "targets_L": ",".join(sorted(det.get("L", {}))), "targets_R": ",".join(sorted(det.get("R", {})))}
    rigidL = {k: v for k, v in det.get("L", {}).items() if k in rig_set}
    rigidR = {k: v for k, v in det.get("R", {}).items() if k in rig_set}
    # one-eye resections (intrinsics at metric scale, pose free)
    for e, ob in (("L", rigidL), ("R", rigidR)):
        if e in proj and e in label and len(ob) >= MIN_EYE_POINTS:
            r = solve_pair(proj[e], None, ob, {}, *label[e], mode="rig", points=P)
            if r:
                out[f"resect_rms_{e}"] = r["rms"]
                out[f"resect_n_{e}"] = r["n_obs"]
                out[f"_pose_{e}"] = (r["R"], r["t"])
                # 4 Nov (Christian: "can this tell us if it is the NL or NR that is moving?"): this eye's pose from
                # the deck minus its label pose, in the label camera axes (x right, y down, z forward; rotation
                # pitch, yaw, roll).  Common to both eyes: the mast head / kinematics; different: the eye.
                R0, t0 = label[e]
                dC = 1e3 * (R0 @ ((-r["R"].T @ r["t"]) - (-R0.T @ t0)))
                out[f"minus_label_C_{e}_mm"] = dC.tolist()
                out[f"minus_label_rot_{e}_mdeg"] = (np.degrees(_rotvec(r["R"] @ R0.T)) * 1e3).tolist()
    pair = all(e in proj and e in cam_from_world for e in ("L", "R"))
    if not pair:
        return out
    if "_pose_L" in out and "_pose_R" in out and "L" in label and "R" in label:
        # both eyes resected independently on the deck: the baseline with no rig assumed, and the label's
        (RLd, tLd), (RRd, tRd) = out["_pose_L"], out["_pose_R"]
        (RL0, tL0), (RR0, tR0) = label["L"], label["R"]
        out["baseline_independent_mm"] = 1e3 * float(np.linalg.norm(-RRd.T @ tRd + RLd.T @ tLd))
        out["baseline_label_mm"] = 1e3 * float(np.linalg.norm(-RR0.T @ tR0 + RL0.T @ tL0))
    (RL, tL), (RR, tR) = cam_from_world["L"], cam_from_world["R"]
    R_rel = RR @ RL.T
    t_rel = tR - R_rel @ tL
    out["baseline_block_mm"] = 1e3 * float(np.linalg.norm(t_rel))
    if len(rigidL) + len(rigidR) < MIN_PAIR_POINTS or "L" not in label:
        return out
    fits = {}
    for mode in MODES:
        if (mode == "translation" and len(rigidR) < 3) or (mode == "free" and (len(rigidR) < MIN_EYE_POINTS or len(rigidL) < 3)) \
                or (mode == "rotation" and (len(rigidR) < 3 or len(rigidL) + len(rigidR) < MIN_ROTATION_POINTS)):
            fits[mode] = None              # the right eye's own pose needs its own targets
            continue
        fits[mode] = solve_pair(proj["L"], proj["R"], rigidL, rigidR, *label["L"], R_rel=R_rel, t_rel=t_rel, mode=mode,
                                points=P)
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
    Rb = np.asarray(R_rel, float) @ np.asarray(rig_ref, float).T if rig_ref is not None else None
    if Rb is not None:
        a = np.degrees(_rotvec(Rb)) * 1e3
        out.update({"rig_block_pitch_mdeg": a[0], "rig_block_yaw_mdeg": a[1], "rig_block_roll_mdeg": a[2]})
    if fits["free"]:
        w = fits["free"]["dR_mdeg"]
        ws = fits["free"]["dR_sd_mdeg"]
        out.update({"dR_pitch_mdeg": w[0], "dR_yaw_mdeg": w[1], "dR_roll_mdeg": w[2], "dR_pitch_sd": ws[0],
                    "dR_yaw_sd": ws[1], "dR_roll_sd": ws[2], "free_rms": fits["free"]["rms"]})
        if Rb is not None:
            a = np.degrees(_rotvec(_rot(fits["free"]["x"][6:9]) @ Rb)) * 1e3
            out.update({"rig_deck_pitch_mdeg": a[0], "rig_deck_yaw_mdeg": a[1], "rig_deck_roll_mdeg": a[2]})
    if fits["rotation"]:
        # 4 Nov (Christian: "Can you use the fiducial data to check the roll trend?"): the rig rotation with the
        # translation held - the deck pins the translation to ~0.05 mm, and a free one trades with the rotation
        w = fits["rotation"]["dR_mdeg"]
        ws = fits["rotation"]["dR_sd_mdeg"]
        out.update({"rot_pitch_mdeg": w[0], "rot_yaw_mdeg": w[1], "rot_roll_mdeg": w[2], "rot_pitch_sd": ws[0],
                    "rot_yaw_sd": ws[1], "rot_roll_sd": ws[2], "rot_rms": fits["rotation"]["rms"],
                    "rot_n_obs": fits["rotation"]["n_obs"]})
        if Rb is not None:
            a = np.degrees(_rotvec(_rot(fits["rotation"]["x"][6:9]) @ Rb)) * 1e3
            out.update({"rig_deckrot_pitch_mdeg": a[0], "rig_deckrot_yaw_mdeg": a[1], "rig_deckrot_roll_mdeg": a[2]})
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
    for k in UNSURVEYED:
        if k in tri:
            out[f"{k.lower()}_rover_mm"] = (1e3 * (R.T @ (tri[k] - t))).tolist()
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
                "rig_block_pitch_mdeg", "rig_block_yaw_mdeg", "rig_block_roll_mdeg", "rig_deck_pitch_mdeg",
                "rig_deck_yaw_mdeg", "rig_deck_roll_mdeg",
                "rot_pitch_mdeg", "rot_yaw_mdeg", "rot_roll_mdeg", "rot_pitch_sd", "rot_yaw_sd", "rot_roll_sd", "rot_rms",
                "rot_n_obs", "rig_deckrot_pitch_mdeg", "rig_deckrot_yaw_mdeg", "rig_deckrot_roll_mdeg",
                "tri_n", "tri_scale_ppm", "tri_shape_rms_mm", "bar_mm", "bar_ppm", "tdf4_rover_mm",
                "minus_label_C_L_mm", "minus_label_C_R_mm", "minus_label_rot_L_mdeg", "minus_label_rot_R_mdeg",
                "baseline_independent_mm", "baseline_label_mm",
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
                progress: Optional[Callable[[str], None]] = None, rec=None,
                rig_ref: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    The fiducial check of one solved block (``work``: the block folder, or its ``colmap`` folder).  Writes
    ``<out_dir>/<block>_pairs.csv`` and ``<block>_detections.csv`` (and annotated figures of up to ``figures``
    images with every rigid target found) when ``out_dir`` is given.  ``images_dir`` overrides where the
    images are read (default: the project's ``images`` folder).  ``rec``: the solved reconstruction when already
    read (anything with ``images``, ``cameras`` and ``reg_image_ids()`` like a ``pycolmap.Reconstruction``);
    ``rig_ref``: the reference rig rotation (default: the Navcam rig file in use, before its drift).
    -> {"block", "pairs" (rows), "detections"}.
    """
    w = Path(work)
    root = w / "colmap" if (w / "colmap" / "project.json").is_file() else w
    block_dir = root.parent if root.name == "colmap" else root
    name = block_dir.name
    say = progress or (lambda s: log.info(s))
    pj = json.loads((root / "project.json").read_text(encoding="utf-8"))
    if rec is None:
        import pycolmap
        rec = pycolmap.Reconstruction(str(root / "sparse" / model))
    if rig_ref is None:
        try:
            from mppp.calib.models import load_rig
            rr = load_rig("N")
            rig_ref = np.asarray(rr["R_sensor_from_ref"], float) if rr else None
        except Exception:                               # noqa: BLE001 - the absolute rig angles are optional
            rig_ref = None
    by_name = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    man = _manifest(block_dir)
    idir = Path(images_dir) if images_dir else root / "images"
    rows = [r for r in pj.get("images") or [] if str(r.get("base_instrument") or r.get("instrument", ""))[:1] == "N"
            and r.get("name") in by_name]
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("sclk_key") or r["stem"]), {})[str(r.get("eye") or r["stem"][1])] = r
    names_all = list(RIGID) + list(BAR) + list(UNSURVEYED)
    pairs_out, det_out, figs, rays = [], [], [], []
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
                if k in BAR:
                    d = detect_target(g, uv[0], uv[1], s, search_full_px=BAR_SEARCH_FULL_PX, max_cost=BAR_MAX_COST)
                elif k in UNSURVEYED:
                    d = detect_target(g, uv[0], uv[1], s, search_full_px=UNSURVEYED_SEARCH_FULL_PX)
                else:
                    d = detect_target(g, uv[0], uv[1], s)
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
        res = check_pair(det, proj, label, cfw, rig_ref)
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
                    if k in det[e] or k in BAR or k in UNSURVEYED:
                        continue
                    uv = proj[e]((R @ P[k] + t)[None])[0]
                    d = detect_target(meta[e]["gray"], *full_to_native(uv, meta[e]["scale"]), meta[e]["scale"],
                                      search_full_px=REFINE_SEARCH_FULL_PX)
                    if d:
                        meta[e]["found"][k] = dict(d, refined=True)
                        det[e][k] = native_to_full([d["x"], d["y"]], meta[e]["scale"])
                        added = True
            if added:
                res = check_pair(det, proj, label, cfw, rig_ref)
        fit = res.pop("_fit", None)
        for e in ("L", "R"):
            pose = res.pop(f"_pose_{e}", None)
            if pose is None or e not in meta or res.get(f"resect_n_{e}", 0) < RAY_MIN_TARGETS \
                    or res.get(f"resect_rms_{e}", 9.0) > RAY_MAX_RMS:
                continue
            for k in UNSURVEYED:
                if k in det.get(e, {}):
                    rays.append({"target": k, "proj": proj[e], "R": pose[0], "t": pose[1], "uv": det[e][k],
                                 "image": meta[e]["row"]["stem"], "block": name, "sol": meta[e]["row"].get("sol")})
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
                figs.append((e, meta[e], pred[e], det[e], fit, proj[e]))
        say(f"{name} {key}: L {res.get('n_L', 0)} R {res.get('n_R', 0)} targets, "
            f"rig rms {res.get('rig_rms', float('nan')):.2f} px, "
            f"scale {res.get('scale_ppm', float('nan')):+.0f} ppm")
        kept = {id(f[1]) for f in figs}
        for e in list(meta):
            meta[e].pop("gray", None)
            if id(meta[e]) not in kept:
                meta[e].pop("img", None)
    out = {"block": name, "pairs": pairs_out, "detections": det_out, "rays": rays}
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        write_csv(o / f"{name}_pairs.csv", pairs_out, PAIR_COLUMNS)
        write_csv(o / f"{name}_detections.csv", det_out, DET_COLUMNS)
        for e, mm, pr, dt, fit, pj_ in figs:
            try:
                fitted = {}
                if fit is not None:
                    R, t = (fit["R"], fit["t"]) if e == "L" else (fit.get("R_R"), fit.get("t_R"))
                    if R is not None:
                        P = survey_nav_m(list(RIGID) + list(UNSURVEYED))
                        fitted = {k: pj_((R @ X + t)[None])[0] for k, X in P.items()}
                annotate(mm["img"], mm["row"], mm["scale"], pr, dt, fitted, e,
                         o / f"{name}_{mm['row']['stem']}_fiducials.png")
            except Exception as exc:                    # noqa: BLE001 - a figure never stops the check
                log.warning("figure %s: %s", mm["row"]["stem"], exc)
        out["figures"] = len(figs)
    return out


# ------------------------------------------------------------------------------------------------ the PDS archive
# 4 Nov 2026 (Christian: "Can you use the fiducial data to check the roll trend? Can you get access to more deck
# images across the mission?"): every Navcam stereo pair of the local PDS archive (MPPP_PDS, sols 0-2023) whose
# label puts the deck targets in frame, measured with the cameras and the rig in use (the law at each image's
# temperature and sol) - no block needed.  The rig rotation the deck sees (translation held, ``rot_*``) against sol
# tests the rig drift of the Navcam joint independently of the tie points.
PDS_NAVCAM = _re.compile(r"^N([LR])F_(\d{4})_\d{10}_\d{3}RAD_.{27}\.IMG$", _re.I)
PDS_MIN_RIGID = 3                # rigid targets predicted in frame for a product to be kept by the scan
PDS_COLUMNS = ["block", "pair", "station", "sol", "T_L", "T_R", "scale_native", "stem_L", "stem_R", "rig_source"] + \
    [c for c in PAIR_COLUMNS if c not in ("block", "pair", "station", "sol", "T_L", "T_R", "scale_native")]


def stretch_gray(a: np.ndarray) -> np.ndarray:
    """A PDS image (H x W or H x W x bands, any number type) as float32 grey 0-255: the band mean stretched between
    its 0.5 and 99.5 percentiles over the valid (> 0) pixels - the 8-bit scale the detection thresholds assume."""
    g = np.asarray(a, np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    v = g[g > 0]
    if v.size < 100:
        return np.zeros_like(g)
    lo, hi = np.percentile(v, [0.5, 99.5])
    return np.clip((g - lo) * (255.0 / max(hi - lo, 1e-6)), 0.0, 255.0).astype(np.float32)


def pds_frame(path: PathLike, names: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """The label facts of one Navcam RAD product: ``stem``, ``eye`` ("L"/"R"), ``sol``, ``sclk_key``, ``scale``,
    ``version``, ``site``/``drive``, ``T`` (degC), ``cm`` (the label CAHVOR(E) in the manifest form) and ``pred``
    (target -> native px).  None for a sub-frame, a model not in ROVER_NAV_FRAME, or an unreadable label."""
    from mppp.cmod import CameraModel
    from mppp.filenames import parse_filename
    from mppp.labels import camera_temperature, label_get, read_pds
    p = Path(path)
    try:
        fn = parse_filename(p)
        L, _ = read_pds(p, load_image=False)
        gcm = label_get(L, "GEOMETRIC_CAMERA_MODEL")
        if gcm is None or str(gcm.get("REFERENCE_COORD_SYSTEM_NAME", "ROVER_NAV_FRAME")) != "ROVER_NAV_FRAME":
            return None
        w, h = int(label_get(L, "IMAGE.LINE_SAMPLES")), int(label_get(L, "IMAGE.LINES"))
        fs = label_get(L, "IMAGE.FIRST_LINE_SAMPLE", default=1)
        fl = label_get(L, "IMAGE.FIRST_LINE", default=1)
        fs, fl = (int(v[0] if isinstance(v, (list, tuple)) else v) for v in (fs, fl))
        s = float(fn.downsample_scale)
        if fs != 1 or fl != 1 or abs(w - 5120 * s) > 1 or abs(h - 3840 * s) > 1:
            return None                                   # sub-frames: the label model is the sub-frame's
        cm = dict(CameraModel.from_label(gcm, w, h).to_label_dict(precision=12), width=w, height=h)
        pred = label_predict(cm, names or (list(RIGID) + list(BAR) + list(UNSURVEYED)))
        return {"path": str(p), "stem": fn.stem, "eye": fn.eye, "sol": fn.sol, "sclk_key": fn.sclk_key, "scale": s,
                "version": fn.version, "site": fn.site, "drive": fn.drive, "T": camera_temperature(L, "N", fn.eye),
                "cm": cm, "pred": {k: [float(v[0]), float(v[1])] for k, v in pred.items()},
                "n_rigid": len([k for k in pred if k in RIGID])}
    except Exception as exc:                              # noqa: BLE001 - one bad product never stops the scan
        log.debug("%s: %s", p.name, exc)
        return None


def scan_pds(root: PathLike, sols: Optional[Tuple[int, int]] = None, cache: Optional[PathLike] = None,
             min_rigid: int = PDS_MIN_RIGID, progress: Optional[Callable[[str], None]] = None,
             workers: int = 1) -> List[Dict[str, Any]]:
    """Every Navcam RAD full frame below ``root`` whose label puts >= ``min_rigid`` rigid targets in frame
    (:func:`pds_frame`).  Labels only (about 0.05 s a product; ``workers`` processes).  ``cache``: a JSON file of
    earlier scans (keyed by path, size and time), read and rewritten, so a later scan reads only new products."""
    import os
    say = progress or (lambda s: log.info(s))
    old: Dict[str, Any] = {}
    cf = Path(cache) if cache else None
    if cf and cf.is_file():
        try:
            old = json.loads(cf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
    new: Dict[str, Any] = {}
    todo: List[Tuple[str, int, int]] = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            m = PDS_NAVCAM.match(f)
            if not m or (sols and not sols[0] <= int(m.group(2)) <= sols[1]):
                continue
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            c = old.get(p)
            if c and c.get("size") == st.st_size and c.get("mtime") == int(st.st_mtime):
                new[p] = c
            else:
                todo.append((p, st.st_size, int(st.st_mtime)))
    say(f"  {len(new) + len(todo)} Navcam products, {len(todo)} labels to read ({len(new)} from the cache)")

    def keep(fr):
        return fr if fr and fr["n_rigid"] >= min_rigid else None

    def save():
        if cf:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({**old, **new}), encoding="utf-8")

    paths = [q[0] for q in todo]
    if workers > 1 and len(todo) > 200:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            it = ex.map(pds_frame, paths, chunksize=64)
            for i, ((p, size, mt), fr) in enumerate(zip(todo, it), 1):
                new[p] = {"size": size, "mtime": mt, "frame": keep(fr)}
                if i % 5000 == 0:
                    say(f"  {i} / {len(todo)} labels read")
                    save()
    else:
        for i, (p, size, mt) in enumerate(todo, 1):
            new[p] = {"size": size, "mtime": mt, "frame": keep(pds_frame(p))}
            if i % 2000 == 0:
                say(f"  {i} / {len(todo)} labels read")
                save()
    save()
    found = [c["frame"] for c in new.values() if c.get("frame")]
    say(f"  {len(found)} Navcam full frames with >= {min_rigid} deck targets in frame")
    return found


def pds_pairs(frames: Sequence[Dict[str, Any]], min_each: int = 3,
              extra_rigid: Sequence[str] = ()) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Stereo pairs (same SCLK) of :func:`scan_pds` frames: per eye the finest product, then the latest version;
    both eyes with >= ``min_each`` rigid targets predicted."""
    by: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for f in frames:
        cur = by.setdefault(f["sclk_key"], {}).get(f["eye"])
        if cur is None or (f["scale"], f["version"]) > (cur["scale"], cur["version"]):
            by[f["sclk_key"]][f["eye"]] = f
    out = []
    for k in sorted(by):
        e = by[k]
        n = {x: e[x]["n_rigid"] + len([k for k in extra_rigid if k in e[x].get("pred", {})]) for x in e}
        if "L" in e and "R" in e and n["L"] >= min_each and n["R"] >= min_each:
            out.append((e["L"], e["R"]))
    return out


def _detect_all(g: np.ndarray, pred: Dict[str, Sequence[float]], s: float) -> Dict[str, Dict[str, float]]:
    found = {}
    for k, uv in pred.items():
        if k in BAR:
            d = detect_target(g, uv[0], uv[1], s, search_full_px=BAR_SEARCH_FULL_PX, max_cost=BAR_MAX_COST)
        elif k in UNSURVEYED:
            d = detect_target(g, uv[0], uv[1], s, search_full_px=UNSURVEYED_SEARCH_FULL_PX)
        else:
            d = detect_target(g, uv[0], uv[1], s)
        if d:
            found[k] = d
    return found


def check_pds_pair(fL: Dict[str, Any], fR: Dict[str, Any], cams: Optional[Dict[str, Dict[str, Any]]] = None,
                   rig: Optional[Dict[str, Any]] = None, rig_ref: Optional[np.ndarray] = None,
                   read: Optional[Callable[[str], np.ndarray]] = None,
                   extra_rigid: Sequence[str] = ()) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    One PDS stereo pair (:func:`pds_pairs`) measured on the deck with the cameras in use at each eye's temperature
    and sol (``cams``: {"L", "R"} model files, default the promoted ones) and the rig in use at the sol (``rig``,
    default the promoted rig file; its drift applied).  The rig's own rotation (``rig_block_*``, here the law) and
    the deck's (``rig_deckrot_*``, translation held; ``rig_deck_*``, translation free) are about ``rig_ref``
    (default the rig file's rotation before its drift).  ``read``: path -> image array (default the PDS reader).
    ``extra_rigid``: unsurveyed targets used as rigid at their fitted positions (("TDF4",); 4 Nov: simulated on the
    block sols' label geometry it adds about 1 % of the pairs and nothing to the roll trend's se).
    -> (row, detections, rays).
    """
    from mppp.calib.constrain import camera_at, rig_at
    from mppp.calib.models import load_camera, load_rig
    if cams is None:
        cams = {e: load_camera("N", e) for e in ("L", "R")}
    if rig is None:
        rig = load_rig("N")
    if rig_ref is None:
        rig_ref = np.asarray(rig["R_sensor_from_ref"], float)
    if read is None:
        from mppp.labels import read_pds
        read = lambda p: read_pds(p)[1]                   # noqa: E731
    sol = fL["sol"]
    R_law, t_law, _ = rig_at(rig, sol)
    if t_law is None:                                      # the labels' baseline when the rig keeps no translation
        (RL0, tL0), (RR0, tR0) = label_pose(fL["cm"]), label_pose(fR["cm"])
        t_law = tR0 - RR0 @ RL0.T @ tL0
    names_all = list(RIGID) + list(BAR) + list(UNSURVEYED)
    det, proj, label, meta, pred = {}, {}, {}, {}, {}
    for e, f in (("L", fL), ("R", fR)):
        g = stretch_gray(read(f["path"]))
        s = f["scale"]
        found = _detect_all(g, f["pred"], s)
        model, params, _ = camera_at(cams[e], f["T"], sol)
        proj[e] = Projector(model, params)
        label[e] = label_pose(f["cm"])
        det[e] = {k: native_to_full([d["x"], d["y"]], s) for k, d in found.items()}
        pred[e] = {k: np.asarray(v, float) for k, v in f["pred"].items()}
        meta[e] = {"gray": g, "scale": s, "found": found, "frame": f}
    cfw = {"L": (np.eye(3), np.zeros(3)), "R": (np.asarray(R_law, float), np.asarray(t_law, float))}
    res = check_pair(det, proj, label, cfw, rig_ref, extra_rigid)
    fit = res.get("_fit")
    if fit is not None:                                    # second pass around the fitted prediction
        P = survey_nav_m(names_all)
        added = False
        for e in meta:
            R, t = (fit["R"], fit["t"]) if e == "L" else (fit.get("R_R"), fit.get("t_R"))
            if R is None:
                continue
            for k in pred[e]:
                if k in det[e] or k in BAR or k in UNSURVEYED:
                    continue
                uv = proj[e]((R @ P[k] + t)[None])[0]
                d = detect_target(meta[e]["gray"], *full_to_native(uv, meta[e]["scale"]), meta[e]["scale"],
                                  search_full_px=REFINE_SEARCH_FULL_PX)
                if d:
                    meta[e]["found"][k] = dict(d, refined=True)
                    det[e][k] = native_to_full([d["x"], d["y"]], meta[e]["scale"])
                    added = True
        if added:
            res = check_pair(det, proj, label, cfw, rig_ref, extra_rigid)
    fit = res.pop("_fit", None)
    res.pop("_residuals", None)
    rays = []
    for e in ("L", "R"):
        pose = res.pop(f"_pose_{e}", None)
        if pose is None or res.get(f"resect_n_{e}", 0) < RAY_MIN_TARGETS or res.get(f"resect_rms_{e}", 9.0) > RAY_MAX_RMS:
            continue
        for k in UNSURVEYED:
            if k in det.get(e, {}):
                rays.append({"target": k, "proj": proj[e], "R": pose[0], "t": pose[1], "uv": det[e][k],
                             "image": meta[e]["frame"]["stem"], "block": "pds", "sol": sol})
    row = {"block": "pds", "pair": fL["sclk_key"], "station": f"S{fL.get('site')}D{fL.get('drive')}", "sol": sol,
           "T_L": fL["T"], "T_R": fR["T"], "scale_native": fL["scale"], "stem_L": fL["stem"], "stem_R": fR["stem"],
           "rig_source": "law"}
    for e in meta:
        s = meta[e]["scale"]
        lab = [native_to_full(pred[e][k], s) - det[e][k] for k in det[e] if k in RIGID and k in pred[e]]
        if lab:
            row[f"label_rms_{e}"] = float(np.sqrt(np.mean(np.sum(np.square(lab), 1))))
    row.update(res)
    dets = []
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
            dets.append({"block": "pds", "image": meta[e]["frame"]["stem"], "eye": e, "sol": sol, "target": k,
                         "x_full": xf, "y_full": yf, "label_x_full": lx, "label_y_full": ly, "fit_x_full": fx,
                         "fit_y_full": fy, "cost": d["cost"], "sd": d["sd"], "corner_dev": d["corner_dev"],
                         "method": d.get("method"), "pass": 2 if d.get("refined") else 1})
    return row, dets, rays


def rig_trend(pairs: Sequence[Dict[str, Any]], drift: Optional[Dict[str, Any]] = None, source: str = "deckrot",
              max_rms: float = 1.0, clip: float = 3.5, step_sol: Optional[float] = None) -> Dict[str, Any]:
    """
    The rig rotation the deck sees against sol, per angle (pitch, yaw, roll; ``rig_<source>_<angle>_mdeg`` about the
    reference rotation, pairs with ``rot_rms`` <= ``max_rms`` px): weighted fits of (a) a constant, (b) a line in sol,
    (c) the ``drift`` model's shape plus a constant (the law, its rate as is), and with ``step_sol`` (d) a constant
    with a step there.  Weights 1 / (sd^2 + tau^2), tau the scatter beyond the formal sd (method of moments);
    points whose residual from the line is beyond ``clip`` times its own sigma (at least; the robust scale of the
    normalised residuals when that is larger) are dropped.  Per fit: the coefficients and se (scaled by
    the reduced chi-square when it exceeds 1), chi-square, BIC; ``rate_minus_law`` (mdeg / 1000 sol) and its z.
    """
    from mppp.calib.constrain import drift_offset_mdeg
    out: Dict[str, Any] = {"source": source, "max_rms": max_rms}
    for ang in ("pitch", "yaw", "roll"):
        sdc, rmc = (f"rot_{ang}_sd", "rot_rms") if source == "deckrot" else (f"dR_{ang}_sd", "free_rms")
        rows = [(float(p["sol"]), _num(p.get(f"rig_{source}_{ang}_mdeg")), _num(p.get(sdc)),
                 _num(p.get("T_L")), p.get("pair"))
                for p in pairs if _num(p.get(f"rig_{source}_{ang}_mdeg")) is not None
                and (_num(p.get(rmc)) or 9.0) <= max_rms and _num(p.get(sdc))]
        if len(rows) < 4:
            out[ang] = {"n": len(rows)}
            continue
        sol = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        sd = np.array([r[2] for r in rows])
        law = np.array([drift_offset_mdeg(drift, s)[f"{ang}_mdeg"] for s in sol]) if drift else np.zeros_like(sol)

        def wfit(A, yy, w):
            W = np.sqrt(w)
            b, *_ = np.linalg.lstsq(A * W[:, None], yy * W, rcond=None)
            r = yy - A @ b
            chi2 = float(np.sum(w * r ** 2))
            dof = max(len(yy) - A.shape[1], 1)
            C = np.linalg.pinv((A * w[:, None]).T @ A) * max(chi2 / dof, 1.0)
            return b, np.sqrt(np.diag(C)), r, chi2

        A_line = np.c_[np.ones_like(sol), (sol - 1000.0) / 1000.0]
        tau = 0.0
        keep = np.ones(len(y), bool)
        for _ in range(2):                                 # tau, then one clip, then tau again
            w = 1.0 / (sd ** 2 + tau ** 2)
            b, se, r, chi2 = wfit(A_line[keep], y[keep], w[keep])
            rr = y - A_line @ b
            ex = np.mean(rr[keep] ** 2 - sd[keep] ** 2)
            tau = float(np.sqrt(max(ex, 0.0)))
            z = rr / np.sqrt(sd ** 2 + tau ** 2)          # each pair against its own sd
            mad = 1.4826 * np.median(np.abs(z[keep] - np.median(z[keep])))
            keep = np.abs(z) <= clip * max(mad, 1.0)
        w = 1.0 / (sd ** 2 + tau ** 2)
        k = keep
        n = int(k.sum())
        res: Dict[str, Any] = {"n": n, "dropped": [rows[i][4] for i in np.where(~k)[0]], "tau_mdeg": tau,
                               "median_sd_mdeg": float(np.median(sd[k])), "sol_range": [float(sol[k].min()), float(sol[k].max())]}
        fits = {"constant": np.ones((len(y), 1)), "line": A_line}
        if drift:
            fits["law"] = np.ones((len(y), 1))
        if step_sol is not None and (sol[k] < step_sol).sum() >= 2 and (sol[k] >= step_sol).sum() >= 2:
            fits["step"] = np.c_[np.ones_like(sol), (sol >= step_sol).astype(float)]
        for name, A in fits.items():
            yy = (y - law) if name == "law" else y
            b, se, r, chi2 = wfit(A[k], yy[k], w[k])
            res[name] = {"coef": b.tolist(), "se": se.tolist(), "chi2": chi2,
                         "bic": chi2 + A.shape[1] * np.log(n), "rms_mdeg": float(np.sqrt(np.mean(r ** 2)))}
        if drift:
            law_rate = 1e3 * float(drift.get(f"{ang}_mdeg_per_sol") or 0.0)
            ln = res["line"]
            res["law_rate_mdeg_per_1000sol"] = law_rate
            res["rate_minus_law"] = ln["coef"][1] - law_rate
            res["rate_minus_law_z"] = res["rate_minus_law"] / ln["se"][1] if ln["se"][1] > 0 else float("nan")
            res["offset_vs_law_mdeg"] = res["law"]["coef"][0]
        res["points"] = [{"pair": rows[i][4], "sol": float(sol[i]), "y": float(y[i]), "sd": float(sd[i]),
                          "law": float(law[i]), "T": rows[i][3], "kept": bool(k[i])} for i in range(len(y))]
        out[ang] = res
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
def _cross(img: np.ndarray, x: float, y: float, size: float, color, thick: int = 2, tilted: bool = False) -> None:
    """A cross centred at sub-pixel (x, y) in the drawing's own pixel-centre coordinates."""
    import cv2
    S = 16                                               # 4 fractional bits
    h = 0.5 * size
    segs = ([(-h, -h, h, h), (-h, h, h, -h)] if tilted else [(-h, 0, h, 0), (0, -h, 0, h)])
    for a, b, c, d in segs:
        cv2.line(img, (int(round((x + a) * S)), int(round((y + b) * S))), (int(round((x + c) * S)), int(round((y + d) * S))),
                 color, thick, cv2.LINE_AA, shift=4)


def annotate(img: np.ndarray, row: Dict[str, Any], scale: float, pred: Dict[str, np.ndarray], det: Dict[str, np.ndarray],
             fitted: Optional[Dict[str, np.ndarray]], eye: str, out: PathLike, width: int = 2048) -> Path:
    """
    One image with the label prediction (red x), the measurement (green o) and, in the insets, the block
    solution's reprojection (cyan +) of the six rigid targets with measured - fitted and measured - label in
    full-resolution px.  ``pred``: native pixels; ``det`` and ``fitted``: full-resolution pixels.  Marks are drawn
    at sub-pixel positions with the pixel-centre mapping of the resampled crops (a pixel centre u of the crop at
    zoom z sits at (u + 0.5) z - 0.5).
    """
    import cv2
    RED, GRN, YEL, CYN, WHT = (40, 40, 255), (60, 230, 60), (0, 220, 255), (255, 220, 0), (255, 255, 255)
    PNK = (230, 120, 255)
    h0, w0 = img.shape[:2]
    s = width / w0
    main = cv2.resize(img, (width, int(round(h0 * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    nat = {k: full_to_native(v, scale) for k, v in det.items()}
    fit_nat = {k: full_to_native(v, scale) for k, v in (fitted or {}).items()}
    to_main = lambda v: (float(v[0] + 0.5) * s - 0.5, float(v[1] + 0.5) * s - 0.5)          # noqa: E731

    def put(t_, pos, col, sc=1.15):
        cv2.putText(main, t_, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.putText(main, t_, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 3, cv2.LINE_AA)
    for k, uv in pred.items():
        x, y = to_main(uv)
        col = PNK if k in UNSURVEYED else YEL
        _cross(main, x, y, 24, RED, 2, tilted=True)
        if k in nat:
            mx, my = to_main(nat[k])
            cv2.circle(main, (int(round(mx * 16)), int(round(my * 16))), 18 * 16, GRN, 3, cv2.LINE_AA, shift=4)
        lab = f"{k} (not surveyed)" if k in UNSURVEYED else k
        put(lab, (int(x) + 24, int(y) - 22) if x < width - 32 * len(lab) else (int(x) - 26 * len(lab), int(y) - 30), col)
    cv2.rectangle(main, (10, 10), (1060, 196), (0, 0, 0), -1)
    _cross(main, 40, 48, 24, RED, 2, tilted=True)
    cv2.putText(main, "label: survey (TDF4: prior) through the label CAHVORE", (75, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                WHT, 2, cv2.LINE_AA)
    cv2.circle(main, (40, 104), 17, GRN, 3)
    cv2.putText(main, "measured: point-symmetry / saddle centre", (75, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHT, 2,
                cv2.LINE_AA)
    _cross(main, 40, 158, 24, CYN, 2)
    cv2.putText(main, "fit (insets): block cameras + rig, pose from the deck", (75, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                WHT, 2, cv2.LINE_AA)
    order = ["AFT1", "RPFA2", "RPFA1", "TDF3", "TDF2", "TDF1"] + list(UNSURVEYED)
    T = width // len(order)
    hh = max(10, int(round(48 * scale)))                # +- 48 full-resolution px around the target
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
        to_tile = lambda v: ((float(v[0]) - (cx - hh) + 0.5) * z - 0.5, (float(v[1]) - (cy - hh) + 0.5) * z - 0.5)  # noqa: E731
        lines = []
        if k in pred:
            _cross(c, *to_tile(pred[k]), 22, RED, 2, tilted=True)
        if k in fit_nat:
            _cross(c, *to_tile(fit_nat[k]), 22, CYN, 1)
        if k in nat:
            _cross(c, *to_tile(nat[k]), 22, GRN, 1)
            if k in fitted:
                d = det[k] - fitted[k]
                lines.append(f"{'-prior' if k in UNSURVEYED else '-fit'} {d[0]:+.2f} {d[1]:+.2f}")
            if k in pred:
                d = det[k] - native_to_full(pred[k], scale)
                lines.append(f"-label {d[0]:+.1f} {d[1]:+.1f}")
        else:
            lines.append("not found")
        cv2.rectangle(c, (0, 0), (T, 38), (0, 0, 0), -1)
        cv2.rectangle(c, (0, T - 30 * len(lines) - 10), (T, T), (0, 0, 0), -1)
        cv2.putText(c, k, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, PNK if k in UNSURVEYED else YEL, 2, cv2.LINE_AA)
        for i, ln in enumerate(lines):
            cv2.putText(c, ln, (8, T - 12 - 30 * (len(lines) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHT, 2, cv2.LINE_AA)
        tiles.append(c)
    strip = np.hstack(tiles)
    strip = np.pad(strip, ((0, 0), (0, width - strip.shape[1]), (0, 0)))
    title = np.zeros((56, width, 3), np.uint8)
    res = "full res" if scale == 1 else f"1/{int(round(1 / scale))} res"
    cv2.putText(title, f"{row['stem'][:35]}  sol {row.get('sol')}, {res}; insets +-48 px, offsets in full-res px",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHT, 2, cv2.LINE_AA)
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
    out["unsurveyed"] = fit_unsurveyed(use)
    # 4 Nov: each eye against its label pose, over every image resected on the deck (pairs or not)
    eyes = {}
    for e in ("L", "R"):
        C_ = np.array([p[f"minus_label_C_{e}_mm"] for p in pairs if p.get(f"minus_label_C_{e}_mm") is not None], float)
        W_ = np.array([p[f"minus_label_rot_{e}_mdeg"] for p in pairs if p.get(f"minus_label_rot_{e}_mdeg") is not None], float)
        if len(C_):
            eyes[e] = {"images": int(len(C_)),
                       "C_mm": [robust_mean(C_[:, i]) for i in range(3)],
                       "rot_mdeg": [robust_mean(W_[:, i]) for i in range(3)]}
    out["eyes_minus_label"] = eyes
    out["baseline_independent_minus_label_mm"] = robust_mean(
        [p["baseline_independent_mm"] - p["baseline_label_mm"] for p in pairs
         if p.get("baseline_independent_mm") is not None and p.get("baseline_label_mm") is not None])
    return out


def fit_point_from_rays(rays: List[Dict[str, Any]], X0: np.ndarray,
                        reject_px: Optional[float] = REJECT_FULL_PX) -> Dict[str, Any]:
    """
    One rover-frame point seen in many images whose poses in the rover frame are known (each from its own
    resection on the surveyed targets): ``rays`` [{"proj": Projector, "R", "t" (x_cam = R X + t), "uv" (full px),
    "image"}].  Least squares on the reprojection; -> {"X_nav_mm" (RNAV), "sd_formal_mm" (scaled by the
    residual variance), "se_jackknife_mm" (leave one image out: carries each image's own pose error),
    "rms_px", "residuals_px", "n", "dropped" (rays beyond ``reject_px`` after a first fit, then refitted)}.
    """
    from scipy.optimize import least_squares

    def fit(rs, x0, robust=False):
        def f(X):
            r = []
            for q in rs:
                pr = q["proj"]((q["R"] @ X + q["t"])[None])[0]
                r.append(pr - q["uv"] if np.all(np.isfinite(pr)) else np.array([50.0, 50.0]))
            return np.concatenate(r)
        return least_squares(f, np.asarray(x0, float), x_scale="jac", loss="soft_l1" if robust else "linear", f_scale=1.0)

    res = fit(rays, X0, robust=bool(reject_px))
    dropped = []
    if reject_px:
        d = np.hypot(res.fun[0::2], res.fun[1::2])
        dropped = [q.get("image", str(i)) for i, q in enumerate(rays) if d[i] > reject_px]
        if len(rays) - len(dropped) >= 2:
            rays = [q for i, q in enumerate(rays) if d[i] <= reject_px]
        else:
            dropped = []
        res = fit(rays, res.x)
    n = len(rays)
    out: Dict[str, Any] = {"n": n, "dropped": dropped, "X_nav_mm": (1e3 * res.x).tolist(),
                           "rms_px": float(np.sqrt((res.fun ** 2).sum() / max(n, 1))),
                           "residuals_px": {q.get("image", str(i)): res.fun[2 * i:2 * i + 2].tolist()
                                            for i, q in enumerate(rays)}}
    out["sd_formal_mm"] = (1e3 * _sd(res)).tolist()
    if n >= 3:
        loo = np.array([fit(rays[:i] + rays[i + 1:], res.x).x for i in range(n)])
        se = np.sqrt((n - 1) / n * ((loo - loo.mean(0)) ** 2).sum(0))
        out["se_jackknife_mm"] = (1e3 * se).tolist()
    return out


def fit_unsurveyed(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The rover-frame positions of the unsurveyed targets (TDF4) from the pairs that triangulated them: per axis a
    3.5-MAD-clipped mean in RNAV (mm) with its standard error, against the position in use."""
    out = {}
    for k in UNSURVEYED:
        col = f"{k.lower()}_rover_mm"
        X = np.array([p[col] for p in pairs if p.get(col) is not None], float).reshape(-1, 3)
        if len(X) == 0:
            continue
        med = np.median(X, 0)
        mad = np.median(np.abs(X - med), 0) * 1.4826
        keep = np.all((np.abs(X - med) <= 3.5 * np.where(mad > 0, mad, np.inf)), 1) if len(X) > 2 else np.ones(len(X), bool)
        Xk = X[keep]
        m = Xk.mean(0)
        se = Xk.std(0, ddof=1) / np.sqrt(len(Xk)) if len(Xk) > 1 else np.full(3, np.nan)
        prior = np.asarray(UNSURVEYED_NAV_MM[k], float)
        out[k] = {"n": int(len(X)), "n_kept": int(len(Xk)), "nav_mm": m.tolist(),
                  "se_mm": se.tolist(), "sd_mm": (Xk.std(0, ddof=1) if len(Xk) > 1 else np.full(3, np.nan)).tolist(),
                  "in_use_nav_mm": prior.tolist(), "minus_in_use_mm": (m - prior).tolist()}
    return out


# ------------------------------------------------------------------------------------------------ one deck frame
DECK_FRAME_DEFAULT = "NRF_1642_0812712250_112RAD_N0790870NCAM03642_0A0095J01"   # sol 1642, full resolution, -8.6 C:
#                                                     all six rigid targets, TDF4 and both differential-bar targets
DECK_INSET_HALF_PX = 14            # half-size of a target's inset (full-resolution px)
MODEL_COLOURS = {"measured": "#ffd400", "label": "#e6007e", "constrained": "#00c8a0"}


def resolver_angles(label) -> Tuple[float, float]:
    """The RSM final-resolver azimuth and elevation (rad) of a product label (the angles of the deck_joint chain)."""
    from mppp.labels import label_get
    st = label_get(label, "RSM_ARTICULATION_STATE")
    names = [str(n).upper() for n in st["ARTICULATION_DEVICE_ANGLE_NAME"]]
    ang = st["ARTICULATION_DEVICE_ANGLE"]
    return float(ang[names.index("AZIMUTH FINAL-RESOLVER")]), float(ang[names.index("ELEVATION FINAL-RESOLVER")])


def deck_frame_models(path: PathLike, chain: Union[PathLike, Dict[str, Any]], cmods: Optional[PathLike] = None,
                      ) -> Dict[str, Any]:
    """
    0.97.0 (Christian, 12 Nov 2026: "a fiducial figure for one high res full deck navcam ... the locations of the
    fiducial using the label and constrained camera models").  One full-frame Navcam RAD product, each deck target
    three ways (full-resolution px, corner origin):
      measured     :func:`detect_target` around the label prediction
      label        the product label's CAHVOR(E), intrinsics and pose
      constrained  the Navcam law at the frame's temperature and sol (:func:`mppp.calib.constrain.camera_at`) with
                   the pose of the deck_joint mast chain at the label's resolver angles - nothing fitted to this frame
    Returns {"stem", "eye", "sol", "T", "scale", "az", "el", "image" (uint8 RGB or grey), "targets": {name: {"measured",
    "label", "constrained", "rigid"}}, "rms_px": {"label", "constrained"} (rigid targets), "mean_px", "scatter_px" (rms about the mean offset)}.
    """
    from mppp.calib.constrain import camera_at
    from mppp.calib.models import load_camera
    from mppp.labels import read_pds
    from mppp_studies.calibration.zcam_mast_head import chain_pose
    fr = pds_frame(path)
    if fr is None:
        raise ValueError(f"{path}: not a full-frame Navcam product with a ROVER_NAV_FRAME model")
    m = chain if isinstance(chain, dict) else json.loads(Path(chain).read_text(encoding="utf-8"))
    L, img = read_pds(Path(path))
    az, el = resolver_angles(L)
    s = fr["scale"]
    name, par, _ = camera_at(load_camera("N", fr["eye"], cmods), fr["T"], fr["sol"])
    proj = Projector(name, par)
    R, C = chain_pose(m, fr["eye"], az, el, fr["T"])
    g = stretch_gray(to_gray(img))
    P = survey_nav_m(list(fr["pred"]))
    targets: Dict[str, Dict[str, Any]] = {}
    for k, xy in fr["pred"].items():
        search = BAR_SEARCH_FULL_PX if k in BAR else UNSURVEYED_SEARCH_FULL_PX if k in UNSURVEYED else SEARCH_FULL_PX
        d = detect_target(g, xy[0], xy[1], s, search, BAR_MAX_COST if k in BAR else MAX_COST)
        if d is None:
            continue
        targets[k] = {"measured": native_to_full([d["x"], d["y"]], s).tolist(), "label": native_to_full(xy, s).tolist(),
                      "constrained": proj((R @ (P[k] - C))[None])[0].tolist(), "rigid": k in RIGID,
                      "kind": "bar" if k in BAR else "unsurveyed" if k in UNSURVEYED else "rigid"}
    out = {"stem": fr["stem"], "eye": fr["eye"], "sol": fr["sol"], "T": fr["T"], "scale": s, "az": az, "el": el,
           "targets": targets, "rms_px": {}, "mean_px": {}, "chain": m.get("name"), "camera": name}
    rigid = [t for t in targets.values() if t["rigid"]]
    for key in ("label", "constrained"):
        d = np.array([np.subtract(t[key], t["measured"]) for t in rigid], float).reshape(-1, 2)
        out["rms_px"][key] = float(np.sqrt((d ** 2).sum(1).mean())) if len(d) else None
        out["mean_px"][key] = d.mean(0).tolist() if len(d) else None
        out.setdefault("scatter_px", {})[key] = float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean())) if len(d) else None
    a = np.asarray(img, np.float32)
    lo, hi = np.percentile(a[a > 0], [0.5, 99.7]) if np.any(a > 0) else (0.0, 1.0)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** (1 / 2.2)           # one stretch for all bands, display gamma
    out["image"] = (255.0 * (a[..., ::-1] if a.ndim == 3 else a)).astype(np.uint8)   # read_pds gives BGR
    return out


def plot_deck_frame(res: Dict[str, Any], out_png: PathLike, dpi: int = 200, inset_half_px: int = DECK_INSET_HALF_PX,
                    ) -> Path:
    """The frame of :func:`deck_frame_models` with every target marked, and one inset per target (+-``inset_half_px``
    full-resolution px about the measured centre): the measured centre (cross), the label model's prediction (circle)
    and the constrained model's (diamond), with each model's offset from the measured centre in px."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img, T, s = res["image"], res["targets"], res["scale"]
    names = sorted(T, key=lambda k: (T[k]["kind"] != "rigid", T[k]["kind"] == "bar", k))
    n = len(names)
    ncol = 3
    nrow = max(1, int(np.ceil(n / ncol)))
    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(nrow, 2 * ncol + ncol, wspace=0.10, hspace=0.34, left=0.005, right=0.995, top=0.88, bottom=0.02)
    ax = fig.add_subplot(gs[:, :2 * ncol])
    kw = {"cmap": "gray"} if img.ndim == 2 else {}
    ext = (0, img.shape[1] / s, img.shape[0] / s, 0)                     # full-resolution px, corner origin
    ax.imshow(img, extent=ext, interpolation="bilinear", **kw)
    MC = MODEL_COLOURS
    for k in names:
        x, y = T[k]["measured"]
        ax.plot([x], [y], "s", ms=13, mfc="none", mec=MC["measured"], mew=1.3)
        ax.annotate(k + ("" if T[k]["rigid"] else "*"), (x, y), xytext=(10, -10), textcoords="offset points",
                    color=MC["measured"], fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.55))
    ax.set_xticks([]); ax.set_yticks([])
    eye = {"L": "left", "R": "right"}.get(res["eye"], res["eye"])
    fig.suptitle(f"Deck fiducials in one {eye} Navcam frame: sol {res['sol']}, camera temperature T = {res['T']:.1f} °C, "
                 f"mast azimuth {np.degrees(res['az']):.1f}°, elevation {np.degrees(res['el']):.1f}° (resolver)\n"
                 f"{res['stem']}", fontsize=10.5, y=0.985)
    h = inset_half_px
    for i, k in enumerate(names):
        a = fig.add_subplot(gs[i // ncol, 2 * ncol + i % ncol])
        x, y = T[k]["measured"]
        a.imshow(img, extent=ext, interpolation="nearest", **kw)
        a.set_xlim(x - h, x + h); a.set_ylim(y + h, y - h)
        a.plot([x], [y], "+", ms=16, mew=1.4, color=MC["measured"])
        txt = []
        for key, mk in (("label", "o"), ("constrained", "D")):
            px, py = T[k][key]
            a.plot([x, px], [y, py], "-", lw=1.0, color=MC[key])
            a.plot([px], [py], mk, ms=8, mfc="none", mec=MC[key], mew=1.6)
            txt.append(f"{key} ({px - x:+.1f}, {py - y:+.1f})")
        a.set_title(f"{k}{'' if T[k]['rigid'] else '*'}   Δ(x, y) px\n" + "\n".join(txt), fontsize=7.5, pad=3, loc="left")
        a.set_xticks([]); a.set_yticks([])
    from matplotlib.lines import Line2D
    r = res["rms_px"]
    hnd = [Line2D([], [], marker="+", ls="", ms=11, mew=1.4, color=MC["measured"], label="measured centre"),
           Line2D([], [], marker="o", ls="", ms=8, mfc="none", mec=MC["label"], mew=1.6,
                  label=f"label CAHVORE (rms {r['label']:.1f} px, {res['scatter_px']['label']:.1f} px about its mean)"),
           Line2D([], [], marker="D", ls="", ms=8, mfc="none", mec=MC["constrained"], mew=1.6,
                  label=f"constrained: Navcam law + mast chain (rms {r['constrained']:.1f} px, "
                        f"{res['scatter_px']['constrained']:.1f} px about its mean)")]
    leg = ax.legend(handles=hnd, loc="lower left", fontsize=9, framealpha=0.75, facecolor="black", edgecolor="none")
    for t in leg.get_texts():
        t.set_color("white")
    ax.text(0.995, 0.988, f"insets ±{h} px, full resolution; rms over the six surveyed rigid targets; "
            "* not in the rms (TDF4 unsurveyed, LDIFF/RDIFF on the moving differential bar)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="white",
            bbox=dict(fc="black", ec="none", alpha=0.55, pad=1.5))
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
