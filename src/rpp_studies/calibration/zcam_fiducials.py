"""
Mastcam-Z anchored to the deck fiducials: the cameras' positions in the mast head, along the boresight too (0.99.34;
Christian, 19 Nov 2026: "Z34 needs to be anchor to the fiducials in order to fix its position in the mast frame.
Prepare that study.").

Why.  Inside a block the Mastcam-Z position along its boresight trades with its focal length: moving a camera
forward by d makes a point at range r look larger by d / r, which a focal length smaller by d / r cancels.  The
terrain sits at 2-20 m, so the joints cannot separate the two.  The 18 Nov N+Z34 joint moved the ZR center 3.7 mm
forward of the start rig (the Mastcam-Z-only joint: 0.8 mm) while the ZR focal length fell by about 1000 ppm - the
trade at about 3 m.  The deck targets are surveyed points 0.8-2.4 m away: there d / r is 2-5 times larger and
varies across the target field, so a frame of the deck measures the position and the focal length apart.

Model.  The deck_joint mast chain (``deck_joint_mast_model*.json``; the Navcam's chain, fitted to the deck targets
and the vicarious fiducials) carries each Mastcam-Z eye ``e`` as it carries the Navcam: per frame ``i`` with the
resolver angles (az_i, el_i) of its label,

    R_i = Rot(d_i) . Rot(Q_e) . Ry(el_i)^T . Rz(az_i)^T . Rot(W)                  camera <- RNAV
    C_i = [P_x, P_y, 0] + Rot(G) . Rz(az_i) . (h + Ry(-el_i) . c_e)               the center in RNAV
    x_ik = pi(K_e(focus_i) (1 + s_e), R_i (X_k - C_i))                            target k (survey, RNAV)

  c_e   the eye's center in the head frame (m; origin the elevation axis on the azimuth axis) - the result
  Q_e   the eye's rotation in the head (rotation vector)
  s_e   a focal-length scale at the deck focus against the zoom's focus law (``--focal free``; held 0 by default)
  d_i   a small per-frame pointing change (rotation vector) with a prior of ``POINTING_SD_MRAD`` - the chain
        points a Mastcam-Z frame to 0.7-1.6 mrad (8 Nov), which d_i takes up so it does not leak into c_e
  K_e   the eye's camera from the zoom's focus law at the frame's focus count (rpp.calib.constrain.zcam_camera_at)
  W, G, P, h   the chain's, held

The chain carries c_e in its position frame and Q_e in its orientation frame, which differ (Ry(-el) against
Ry(el)), so the results are read from the poses at the level head (az 0, the Navcam's level elevation; 0.99.35): each
center in the level mast frame (x forward, y right, z down), its component along the boresight, the baseline, and
the ZR center in the ZL camera frame, R_L (C_R - C_L), to set against the rig files and the joints' "right_centre_m"
(the 18 Nov joints: z 14.96 mm at the start, 15.73 Mastcam-Z only, 18.70 with the Navcam free); the sd propagate the
fit's covariance (the rms scaled in).
Run it twice, with the focal law held and with ``s_e`` free: if c_e along the boresight does not move between the
two (and s_e comes out small), the deck separates position and focal length as intended.

Frames.  Every Mastcam-Z RAD product of the zoom under ``--pds`` whose label model (ROVER_NAV_FRAME) puts at least
``MIN_TARGETS`` surveyed targets (and TDF4) in its frame - the deck pans - read from labels only (:func:`scan`),
then each target found by the point-symmetry detector of :mod:`rpp_studies.calibration.fiducials` around the label
prediction, with the window scaled to the zoom's pixel (``WINDOW_SCALE``).

    python -m rpp_studies.calibration.zcam_fiducials --pds D:\\data\\m2020\\zcam --out DIR [--zoom 34]
           [--chain deck_joint_mast_model.json] [--focal law|free|both] [--cache DIR\\scan.json] [--sols 1 2100]

Writes into DIR: ``zcam<zoom>_fiducial_frames.json`` (the scan), ``zcam<zoom>_fiducial_obs.csv`` (one row per
target found), ``zcam<zoom>_fiducial_anchor.json`` and ``.md`` (the fits).  Reads only.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import fiducials as FD

log = logging.getLogger("rpp_studies.calibration.zcam_fiducials")
PathLike = Union[str, Path]

FULL_FRAME = (1648, 1200)          # the Mastcam-Z full frame (COLMAP px, corner origin)
TARGETS: Tuple[str, ...] = FD.RIGID + FD.UNSURVEYED           # the bar targets move; AFT2 is hidden from the mast
MIN_TARGETS = 2                    # targets predicted in a frame for the scan to keep it
WINDOW_SCALE = {34: 1.55, 48: 2.2, 63: 2.9, 79: 3.6, 100: 4.5, 110: 5.0}   # the zoom's IFOV against the Navcam's
SIGMA_PX = 0.5                     # a target's pixel sd (full resolution)
POINTING_SD_MRAD = 1.5             # the per-frame pointing change's prior
REJECT_PX = 3.0                    # a target this far from the fit is dropped (one pass)
CHAIN_FILE = Path(__file__).resolve().parent / "data" / "deck_joint_mast_model_DJ32_20261107.json"
JOINT_RIGHT_CENTER_MM = {          # the 18 Nov 2026 joints' ZR center in the ZL frame (x right, y down, z boresight)
    "start rig (in use)": (242.593, 1.671, 14.964),
    "Mastcam-Z 34 mm joint (pre-freeze-3 c)": (242.64, 1.65, 15.73),
    "N+Z34 joint (pre-freeze-3 c)": (242.187, 1.759, 18.697),
}


# ------------------------------------------------------------------ the chain
def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _RV(v) -> np.ndarray:
    return FD._rot(np.asarray(v, float))


def load_chain(path: Optional[PathLike] = None) -> Dict[str, Any]:
    return json.loads(Path(path or CHAIN_FILE).read_text(encoding="utf-8"))


def pose(chain: Dict[str, Any], c: np.ndarray, q: np.ndarray, az: float, el: float,
         d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(R camera <- RNAV, C in RNAV) of a camera at head center ``c`` and rotation ``q`` (rotation vector) for the
    resolver angles ``az``, ``el`` (rad); ``d`` a pointing change in the camera frame (see the module)."""
    p = chain["params"]
    ka, ke = float(p.get("k_az", 0.0)), float(p.get("k_el", 0.0))
    A = _Rz(az * (1 + ka))
    E = _Ry(el * (1 + ke))
    R = _RV(q) @ E.T @ A.T @ _RV(p["W_rotvec_rad"])
    if d is not None:
        R = _RV(d) @ R
    C = np.r_[p["P_xy_m"], 0.0] + _RV(p["G_rotvec_rad"]) @ A @ (np.asarray(p["h_m"], float) + _Ry(-el * (1 + ke)) @ np.asarray(c, float))
    return R, C


def mount_from_pose(chain: Dict[str, Any], R: np.ndarray, C: np.ndarray, az: float, el: float) -> Tuple[np.ndarray, np.ndarray]:
    """The inverse of :func:`pose` without ``d``: (c, q) of one frame's pose (R, C) - a label pose, for the start."""
    from scipy.spatial.transform import Rotation
    p = chain["params"]
    ka, ke = float(p.get("k_az", 0.0)), float(p.get("k_el", 0.0))
    A, E = _Rz(az * (1 + ka)), _Ry(el * (1 + ke))
    c = _Ry(-el * (1 + ke)).T @ (A.T @ _RV(p["G_rotvec_rad"]).T @ (np.asarray(C, float) - np.r_[p["P_xy_m"], 0.0]) - np.asarray(p["h_m"], float))
    Q = np.asarray(R, float) @ _RV(p["W_rotvec_rad"]).T @ A @ E
    return c, Rotation.from_matrix(Q).as_rotvec()


# ------------------------------------------------------------------ the frames
ZCAM_RAD = re.compile(r"^Z[LR]\d_(\d{4})_\d{10}_\d{3}RAD_.{27}\.IMG$", re.I)


def z_frame(path: PathLike, zoom: int = 34, names: Sequence[str] = TARGETS, min_targets: int = MIN_TARGETS) -> Optional[Dict[str, Any]]:
    """The label facts of one Mastcam-Z RAD product of ``zoom``: stem, eye, sol, focus count, the resolver angles,
    the lens temperature, the sub-frame and scale, the label pose (RNAV) and the targets the label model puts in the
    frame (full-frame px, corner origin).  None for another zoom, a model not in ROVER_NAV_FRAME, too few targets or an
    unreadable label."""
    from rpp.cmod import CameraModel
    from rpp.filenames import parse_filename
    from rpp.labels import instrument_temperatures, label_get, read_pds, zcam_optics_temperature
    p = Path(path)
    try:
        fn = parse_filename(p)
        if fn.zoom_mm != int(zoom):
            return None
        L, _ = read_pds(p, load_image=False)
        gcm = label_get(L, "GEOMETRIC_CAMERA_MODEL")
        if gcm is None or str(gcm.get("REFERENCE_COORD_SYSTEM_NAME", "ROVER_NAV_FRAME")) != "ROVER_NAV_FRAME":
            return None
        w, h = int(label_get(L, "IMAGE.LINE_SAMPLES")), int(label_get(L, "IMAGE.LINES"))
        fs = label_get(L, "IMAGE.FIRST_LINE_SAMPLE", default=1)
        fl = label_get(L, "IMAGE.FIRST_LINE", default=1)
        fs, fl = (int(v[0] if isinstance(v, (list, tuple)) else v) for v in (fs, fl))
        s = float(fn.downsample_scale)
        cm = dict(CameraModel.from_label(gcm, w, h).to_label_dict(precision=12), width=w, height=h)
        pred = FD.label_predict(cm, names)
        if len(pred) < int(min_targets):
            return None
        R, t = FD.label_pose(cm)
        az, el = FD.resolver_angles(L)
        focus = label_get(L, "INSTRUMENT_STATE_PARMS.FOCUS_POSITION_COUNT")
        off = np.array([fs - 1, fl - 1], float)                  # the sub-frame's corner in the full frame (full px)
        return {"path": str(p), "stem": fn.stem, "eye": fn.eye, "group": f"Z{fn.eye}{int(zoom):03d}", "sol": fn.sol,
                "sclk_key": fn.sclk_key, "sequence": fn.sequence, "scale": s, "offset_full_px": off.tolist(),
                "focus": None if focus is None else float(focus), "az": az, "el": el,
                "T_lens": zcam_optics_temperature(instrument_temperatures(L)),
                "label_R": R.tolist(), "label_C": (-R.T @ t).tolist(),
                "pred_native": {k: [float(v[0]), float(v[1])] for k, v in pred.items()},
                "pred_full": {k: (FD.native_to_full(v, s) + off).tolist() for k, v in pred.items()}}
    except Exception as exc:                                     # noqa: BLE001 - one bad product never stops the scan
        log.debug("%s: %s", p.name, exc)
        return None


def scan(root: PathLike, zoom: int = 34, sols: Optional[Tuple[int, int]] = None, cache: Optional[PathLike] = None,
         min_targets: int = MIN_TARGETS) -> List[Dict[str, Any]]:
    """Every Mastcam-Z RAD product below ``root`` that :func:`z_frame` keeps (labels only); ``cache`` a JSON file of
    earlier scans keyed by path, size and time."""
    cf = Path(cache) if cache else None
    old: Dict[str, Any] = {}
    if cf and cf.is_file():
        try:
            old = json.loads(cf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
    new: Dict[str, Any] = {}
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            m = ZCAM_RAD.match(f)
            if not m or (sols and not sols[0] <= int(m.group(1)) <= sols[1]):
                continue
            q = os.path.join(dirpath, f)
            st = os.stat(q)
            c = old.get(q)
            if c and c.get("size") == st.st_size and c.get("mtime") == int(st.st_mtime) and c.get("zoom") == int(zoom):
                new[q] = c
                continue
            new[q] = {"size": st.st_size, "mtime": int(st.st_mtime), "zoom": int(zoom),
                      "frame": z_frame(q, zoom, min_targets=min_targets)}
    if cf:
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps({**old, **new}), encoding="utf-8")
    found = [c["frame"] for c in new.values() if c.get("frame")]
    log.info("zcam fiducials: %d products read, %d %d mm frames with >= %d deck targets in view", len(new), len(found),
             zoom, min_targets)
    return found


def measure(frames: Sequence[Dict[str, Any]], zoom: int = 34, search_full_px: float = 16.0) -> List[Dict[str, Any]]:
    """Each predicted target found in its frame (point symmetry, the window scaled to the zoom): rows {"stem", "eye",
    "group", "target", "x", "y" (full-frame px, corner origin), "cost", "method"}."""
    from rpp.labels import read_pds
    k = float(WINDOW_SCALE.get(int(zoom), 1.55))
    rows = []
    for fr in frames:
        try:
            _, img = read_pds(Path(fr["path"]))
        except Exception as exc:                                 # noqa: BLE001
            log.warning("%s: %s", fr["stem"], exc)
            continue
        g = FD.stretch_gray(FD.to_gray(np.asarray(img)))
        s, off = float(fr["scale"]), np.asarray(fr["offset_full_px"], float)
        for name, xy in fr["pred_native"].items():
            det = FD.detect_target(g, xy[0], xy[1], k * s, search_full_px * (1.3 if name in FD.UNSURVEYED else 1.0))
            if det is None:
                continue
            full = FD.native_to_full([det["x"], det["y"]], s) + off
            rows.append({"stem": fr["stem"], "eye": fr["eye"], "group": fr["group"], "target": name,
                         "x": float(full[0]), "y": float(full[1]), "cost": det["cost"], "method": det["method"]})
    return rows


# ------------------------------------------------------------------ the fit
def opencv_project(par: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """COLMAP FULL_OPENCV (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6): camera-frame points (n, 3) -> pixels
    (n, 2); NaN behind the camera.  Vectorized (the fit calls it for every target of a frame at once)."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = [float(v) for v in par[:12]]
    Z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = Xc[:, 0] / Z, Xc[:, 1] / Z
    r2 = x * x + y * y
    rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv = np.stack([fx * xd + cx, fy * yd + cy], axis=1)
    uv[~(Z > 0)] = np.nan
    return uv


def fit(frames: Sequence[Dict[str, Any]], obs: Sequence[Dict[str, Any]], chain: Dict[str, Any], zoom_model: Dict[str, Any],
        focal: str = "law", pointing_sd_mrad: float = POINTING_SD_MRAD, sigma_px: float = SIGMA_PX,
        reject_px: float = REJECT_PX, start: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None) -> Dict[str, Any]:
    """The mounts (see the module).  ``focal`` "law" holds s_e = 0, "free" fits it.  ``start``: {eye: (c, q)}, else the
    median of the frames' label poses through the chain.  Returns {"eyes": {eye: {"c_m", "c_sd_mm", "q_rotvec",
    "c_along_boresight_mm", "s_ppm", "s_sd_ppm", "frames", "targets", "rms_px"}}, "pair": {...}, "rms_px", ...}."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from rpp.calib.constrain import zcam_camera_at
    by = {f["stem"]: f for f in frames}
    obs = [o for o in obs if o["stem"] in by and by[o["stem"]].get("focus") is not None]
    stems = sorted({o["stem"] for o in obs})
    eyes = sorted({by[s]["eye"] for s in stems})
    if not eyes:
        raise ValueError("no observations to fit")
    P = FD.survey_nav_m(sorted({o["target"] for o in obs}))
    par = {}
    for s in stems:
        fr = by[s]
        name, pv, _ = zcam_camera_at(zoom_model, fr["group"], fr["focus"], T_lens=fr.get("T_lens"))
        if name != "FULL_OPENCV":
            raise ValueError(f"{fr['group']}: a {name} camera; the fit projects FULL_OPENCV")
        par[s] = np.asarray(pv, float)
    if start is None:
        start = {}
        for e in eyes:
            cs, qs = zip(*[mount_from_pose(chain, np.asarray(by[s]["label_R"]), np.asarray(by[s]["label_C"]),
                                           by[s]["az"], by[s]["el"]) for s in stems if by[s]["eye"] == e])
            start[e] = (np.median(np.asarray(cs), axis=0), np.median(np.asarray(qs), axis=0))
    free_s = focal == "free"
    ne = 7 if free_s else 6
    ix = {e: i * ne for i, e in enumerate(eyes)}
    sp = 1e-3 * float(pointing_sd_mrad) if pointing_sd_mrad else None
    jd = {s: len(eyes) * ne + 3 * i for i, s in enumerate(stems)} if sp else {}
    n_par = len(eyes) * ne + (3 * len(stems) if sp else 0)
    x0 = np.zeros(n_par)
    for e in eyes:
        x0[ix[e]:ix[e] + 3], x0[ix[e] + 3:ix[e] + 6] = start[e]
    # the observations grouped by frame (one pose per frame and residual evaluation)
    groups = []
    for s in stems:
        mine = [j for j, o in enumerate(obs) if o["stem"] == s]
        groups.append((s, np.array(mine), np.array([P[obs[j]["target"]] for j in mine]),
                       np.array([[obs[j]["x"], obs[j]["y"]] for j in mine], float)))
    p_ = chain["params"]
    ka, ke = float(p_.get("k_az", 0.0)), float(p_.get("k_el", 0.0))
    W, G = _RV(p_["W_rotvec_rad"]), _RV(p_["G_rotvec_rad"])
    P0, h = np.r_[p_["P_xy_m"], 0.0], np.asarray(p_["h_m"], float)
    pre = {s: (_Rz(by[s]["az"] * (1 + ka)), _Ry(by[s]["el"] * (1 + ke)), _Ry(-by[s]["el"] * (1 + ke))) for s in stems}

    def residuals(x, keep):
        r = np.zeros(2 * len(obs))
        for s, js, X, uv0 in groups:
            e = by[s]["eye"]
            c, q = x[ix[e]:ix[e] + 3], x[ix[e] + 3:ix[e] + 6]
            A, E, Em = pre[s]
            R = _RV(q) @ E.T @ A.T @ W
            if sp:
                R = _RV(x[jd[s]:jd[s] + 3]) @ R
            C = P0 + G @ A @ (h + Em @ c)
            pv = par[s].copy()
            if free_s:
                pv[:2] *= 1.0 + x[ix[e] + 6]
            uv = opencv_project(pv, (X - C) @ R.T)
            d = np.where(np.isfinite(uv), uv - uv0, 1e4) / sigma_px
            r[2 * js], r[2 * js + 1] = d[:, 0], d[:, 1]
        r = r.reshape(-1, 2)[keep].ravel()
        if sp:
            r = np.r_[r, x[len(eyes) * ne:] / sp]
        return r

    def sparsity(keep):
        rows = [j for j in range(len(obs)) if keep[j]]
        S = lil_matrix((2 * len(rows) + (3 * len(stems) if sp else 0), n_par), dtype=int)
        for k, j in enumerate(rows):
            s = obs[j]["stem"]
            e = by[s]["eye"]
            for rr in (2 * k, 2 * k + 1):
                S[rr, ix[e]:ix[e] + ne] = 1
                if sp:
                    S[rr, jd[s]:jd[s] + 3] = 1
        if sp:
            for i in range(3 * len(stems)):
                S[2 * len(rows) + i, len(eyes) * ne + i] = 1
        return S

    keep = np.ones(len(obs), bool)
    sol = least_squares(residuals, x0, args=(keep,), jac_sparsity=sparsity(keep), x_scale="jac", method="trf",
                        loss="soft_l1", f_scale=3.0, max_nfev=100)
    res = residuals(sol.x, keep)[:2 * len(obs)].reshape(-1, 2) * sigma_px       # one pass of rejection
    bad = np.flatnonzero(np.hypot(res[:, 0], res[:, 1]) > reject_px)
    keep[bad] = False
    sol = least_squares(residuals, sol.x, args=(keep,), jac_sparsity=sparsity(keep), x_scale="jac", method="trf",
                        loss="linear", max_nfev=100)
    x = sol.x
    r = residuals(x, keep)
    n_obs = 2 * int(keep.sum())
    res = r[:n_obs].reshape(-1, 2) * sigma_px
    J = sol.jac.toarray() if hasattr(sol.jac, "toarray") else np.asarray(sol.jac)
    dof = max(1, len(r) - len(x))
    s2 = float(np.sum(r ** 2) / dof)
    cov = np.linalg.pinv(J.T @ J) * s2
    out: Dict[str, Any] = {"focal": focal, "pointing_sd_mrad": pointing_sd_mrad, "frames": len(stems),
                           "observations": int(keep.sum()), "rejected": int(len(bad)),
                           "rms_px": float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))) if len(res) else None,
                           "variance_factor": s2, "eyes": {}}
    kept_eye = np.array([by[obs[j]["stem"]]["eye"] for j in range(len(obs)) if keep[j]])
    # 0.99.35: what the results are read in.  The chain carries a center in its position frame (through Ry(-el)) and
    # a rotation in its orientation frame (through Ry(el)): c and Q are not in one frame, so neither c . (Q^T z) nor
    # Rot(Q_L) (c_R - c_L) means anything (0.99.34 reported both: the ZR center came out at x -241 mm).  The results
    # are now the poses themselves at the level head (az 0, the Navcam's level elevation): each center in the level
    # mast frame (x forward along the level boresight, y right, z down; mm from the elevation axis on the azimuth
    # axis), its boresight there, and the ZR center in the ZL camera frame R_L (C_R - C_L); their sd by propagating the
    # fit's covariance (a numerical Jacobian).
    from .zcam_mast_head import Mast
    mast = Mast(chain)
    el0 = float(mast.el_level)

    def at_level(xx, e):
        c, q = xx[ix[e]:ix[e] + 3], xx[ix[e] + 3:ix[e] + 6]
        R, C = pose(chain, c, q, 0.0, el0)
        return R, C

    def mast_center(xx, e):
        _, C = at_level(xx, e)
        return 1e3 * mast.F @ (C - mast.O)

    def propagate(g):
        y0 = np.asarray(g(x), float)
        Jg = np.zeros((y0.size, len(x)))
        for k in range(len(x)):
            if not np.any(cov[k]):
                continue
            h = 1e-7 if k < len(eyes) * ne else 1e-6
            xp = x.copy()
            xp[k] += h
            Jg[:, k] = (np.asarray(g(xp), float) - y0) / h
        Cy = Jg @ cov @ Jg.T
        return y0, np.sqrt(np.clip(np.diag(Cy), 0, None)), Cy

    for e in eyes:
        i = ix[e]
        R0, _ = at_level(x, e)
        bore = mast.F @ R0[2]                                     # the optical axis in the level mast frame
        cm, cm_sd, Cm = propagate(lambda xx, e=e: mast_center(xx, e))
        al, al_sd, _ = propagate(lambda xx, e=e: [mast_center(xx, e) @ (mast.F @ at_level(xx, e)[0][2])])
        m = kept_eye == e
        var_s = max(cov[i + 6, i + 6], 0.0) if free_s else 0.0
        out["eyes"]["Z" + e] = {
            "c_m": x[i:i + 3].tolist(), "q_rotvec": x[i + 3:i + 6].tolist(),
            "c_mast_mm": cm.tolist(), "c_mast_sd_mm": cm_sd.tolist(), "boresight_mast": bore.tolist(),
            "c_along_boresight_mm": float(al[0]), "c_along_boresight_sd_mm": float(al_sd[0]),
            "start_c_m": np.asarray(start[e][0]).tolist(),
            "s_ppm": float(1e6 * x[i + 6]) if free_s else 0.0,
            "s_sd_ppm": float(1e6 * np.sqrt(var_s)) if free_s else None,
            "corr_s_along": None,
            "frames": len({by[s]["stem"] for s in stems if by[s]["eye"] == e}),
            "targets": int(m.sum()), "rms_px": float(np.sqrt(np.mean(np.sum(res[m] ** 2, axis=1)))) if m.any() else None}
        if free_s:
            _, _, Cas = propagate(lambda xx, e=e: [mast_center(xx, e) @ (mast.F @ at_level(xx, e)[0][2]), 1e6 * xx[ix[e] + 6]])
            out["eyes"]["Z" + e]["corr_s_along"] = float(Cas[0, 1] / np.sqrt(max(Cas[0, 0] * Cas[1, 1], 1e-30)))
    if {"L", "R"} <= set(eyes):
        def right_in_left(xx):
            RL, CL = at_level(xx, "L")
            _, CR = at_level(xx, "R")
            return 1e3 * RL @ (CR - CL)

        def baseline(xx):
            return [1e3 * np.linalg.norm(at_level(xx, "R")[1] - at_level(xx, "L")[1])]
        r, r_sd, _ = propagate(right_in_left)
        b, b_sd, _ = propagate(baseline)
        out["pair"] = {"baseline_mm": float(b[0]), "baseline_sd_mm": float(b_sd[0]),
                       "right_center_in_left_mm": r.tolist(), "right_center_in_left_sd_mm": r_sd.tolist(),
                       "joints_mm": {k: list(v) for k, v in JOINT_RIGHT_CENTER_MM.items()}}
    return out


def describe(fits: Dict[str, Dict[str, Any]], zoom: int = 34) -> List[str]:
    lines = [f"# Mastcam-Z {zoom} mm anchored to the deck fiducials", ""]
    for how, f in fits.items():
        lines += [f"## focal {how}: {f['frames']} frames, {f['observations']} targets ({f['rejected']} dropped), "
                  f"rms {f['rms_px']:.2f} px", ""]
        lines.append("| eye | center, level mast frame (mm; x fwd, y right, z down) | sd (mm) | along the boresight (mm) | focal scale (ppm) | frames | rms (px) |")
        lines.append("|---|---|---|---|---|---:|---:|")
        for e, v in f["eyes"].items():
            c = ", ".join(f"{a:.1f}" for a in v["c_mast_mm"])
            sd = ", ".join(f"{a:.2f}" for a in v["c_mast_sd_mm"])
            sc = "held" if v["s_sd_ppm"] is None else f"{v['s_ppm']:+.0f} +- {v['s_sd_ppm']:.0f} (corr with along {v['corr_s_along']:+.2f})"
            lines.append(f"| {e} | {c} | {sd} | {v['c_along_boresight_mm']:.2f} +- {v['c_along_boresight_sd_mm']:.2f} | {sc} | "
                         f"{v['frames']} | {v['rms_px']:.2f} |")
        pr = f.get("pair")
        if pr:
            r, rs = pr["right_center_in_left_mm"], pr["right_center_in_left_sd_mm"]
            lines += ["", f"baseline {pr['baseline_mm']:.2f} +- {pr['baseline_sd_mm']:.2f} mm; ZR center in the ZL frame (x right, "
                          f"y down, z boresight) ({r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}) +- ({rs[0]:.2f}, {rs[1]:.2f}, {rs[2]:.2f}) mm"]
            for k, v in pr["joints_mm"].items():
                lines.append(f"  - {k}: ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}) mm; the deck minus it: z {r[2] - v[2]:+.2f} mm "
                             f"({(r[2] - v[2]) / max(rs[2], 1e-9):+.1f} sd)")
        lines.append("")
    return lines


def run(pds: PathLike, out: PathLike, zoom: int = 34, chain: Optional[PathLike] = None, focal: str = "both",
        cache: Optional[PathLike] = None, sols: Optional[Tuple[int, int]] = None, cmods: Optional[PathLike] = None) -> Dict[str, Any]:
    from rpp.sfm.project import zcam_focus_model_path
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ch = load_chain(chain)
    zm = json.loads(zcam_focus_model_path(cmods, zoom=zoom).read_text(encoding="utf-8"))
    frames = scan(pds, zoom, sols, cache or out / f"zcam{zoom}_fiducial_scan_cache.json")
    (out / f"zcam{zoom}_fiducial_frames.json").write_text(json.dumps(frames, indent=1), encoding="utf-8")
    obs = measure(frames, zoom)
    with open(out / f"zcam{zoom}_fiducial_obs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "eye", "group", "target", "x", "y", "cost", "method"])
        w.writeheader()
        w.writerows(obs)
    fits = {}
    for how in (("law", "free") if focal == "both" else (focal,)):
        fits[how] = fit(frames, obs, ch, zm, focal=how)
    rec = {"zoom": zoom, "chain": ch.get("name"), "frames_scanned": len(frames), "observations": len(obs), "fits": fits}
    (out / f"zcam{zoom}_fiducial_anchor.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    (out / f"zcam{zoom}_fiducial_anchor.md").write_text("\n".join(describe(fits, zoom)) + "\n", encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pds", required=True, help="the Mastcam-Z PDS products (searched recursively)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=int, default=34)
    ap.add_argument("--chain", default=None, help=f"the deck_joint mast chain (default the shipped {CHAIN_FILE.name})")
    ap.add_argument("--focal", default="both", choices=("law", "free", "both"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--sols", nargs=2, type=int, default=None)
    ap.add_argument("--cmods", default=None, help="the camera models (default the ones in use)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    rec = run(a.pds, a.out, a.zoom, a.chain, a.focal, a.cache, tuple(a.sols) if a.sols else None, a.cmods)
    print((Path(a.out) / f"zcam{a.zoom}_fiducial_anchor.md").read_text(encoding="utf-8"))
    return 0 if rec["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
