"""
A promoted model constrained to one image or block: the Navcam / Hazcam camera at a temperature and sol
(:func:`camera_at`), the stereo rig at a sol (:func:`rig_at`), the Mastcam-Z camera at a focus count and backlash
state (:func:`zcam_camera_at`), a camera scaled to an image resolution (:func:`scaled`), and the start camera from a
PDS label where no promoted model exists (:func:`label_camera`).

Mastcam-Z backlash state: an image is in the ``"backlash"`` state (the higher focal length, about 1 % above the
label) unless the low-backlash observation list, the found list or its own focal fit puts it in the ``"regular"``
state (the lower focal length, the label's) - :func:`focus_state`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, "Path"]

from pathlib import Path

import numpy as np

from . import forms
from .forms import ZCAM_DEFAULT_STATE

NAVCAM_T0_DEGC = -20.0


# ------------------------------------------------------------------------------------------- Navcam / Hazcam
def thermal_terms(th: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """{"fx_ppm", "fy_ppm", "cx_px", "cy_px", "k1", "T0"} of a camera file's ``thermal`` entry (fx and fy each their
    own ppm/degC, else the shared ``ppm_per_degC``); zeros without one."""
    th = th or {}
    ppm = float(th.get("ppm_per_degC") or 0.0)
    return {"fx_ppm": float(th.get("fx_ppm_per_degC", ppm) or 0.0), "fy_ppm": float(th.get("fy_ppm_per_degC", ppm) or 0.0),
            "cx_px": float(th.get("cx_px_per_degC") or 0.0), "cy_px": float(th.get("cy_px_per_degC") or 0.0),
            "k1": float(th.get("k1_per_degC") or 0.0),
            "T0": float(th.get("T0_degC", NAVCAM_T0_DEGC))}


def thermal_scale(th: Optional[Dict[str, Any]], T: float) -> Tuple[float, float]:
    """(sx, sy): fx and fy at temperature ``T`` over their values at T0 (1 + ppm 1e-6 (T - T0) per axis)."""
    t = thermal_terms(th)
    d = float(T) - t["T0"]
    return 1.0 + 1e-6 * t["fx_ppm"] * d, 1.0 + 1e-6 * t["fy_ppm"] * d


def early_share(model: Dict[str, Any], sols: Sequence[float]) -> float:
    """The share of ``sols`` before the model's ``early_mission.before_sol`` (0 without the term)."""
    em = model.get("early_mission") or {}
    if not em or not len(sols):
        return 0.0
    return float(np.mean([float(s) < float(em["before_sol"]) for s in sols]))


def camera_at(model: Dict[str, Any], T_degC: Optional[float] = None, sol: Optional[float] = None,
              early: Optional[float] = None) -> Tuple[str, List[float], Dict[str, Any]]:
    """
    The full-frame camera of a Navcam / Hazcam model file at camera temperature ``T_degC`` and ``sol``:
    fx, fy scaled by their ppm/degC, cx, cy moved by their px/degC, k1 by its per degC (all about ``thermal.T0_degC``),
    and the early-mission offsets (``early_mission.d<fx|fy|cx|cy>_px`` of ``applied_terms``) added for an image
    before ``before_sol``.  ``early``: the share of a block's images before that sol (default 1 or 0 from ``sol``).
    Returns (COLMAP model name, params, applied) with ``applied`` = {"thermal": {...}, "early_mission": {...}}.
    """
    names = list(model.get("param_names") or forms.PARAM_NAMES[model["model"]])
    p = [float(v) for v in model["params"]]
    applied: Dict[str, Any] = {}
    th = model.get("thermal")
    if th and T_degC is not None:
        T = float(T_degC)
        sx, sy = thermal_scale(th, T)
        p[0] *= sx
        p[1] *= sy
        t = thermal_terms(th)
        dcx, dcy = t["cx_px"] * (T - t["T0"]), t["cy_px"] * (T - t["T0"])
        p[2] += dcx
        p[3] += dcy
        dk1 = t["k1"] * (T - t["T0"])
        if dk1 and "k1" in names:
            p[names.index("k1")] += dk1
        applied["thermal"] = {"T_degC": T, "T_median_degC": T, "scale": float(np.sqrt(sx * sy)), "sx": sx, "sy": sy,
                              "dcx_px": dcx, "dcy_px": dcy, "dk1": dk1, **th}
    em = model.get("early_mission") or {}
    share = early
    if share is None and sol is not None and em:
        share = 1.0 if float(sol) < float(em["before_sol"]) else 0.0
    if em and share:
        terms = [t for t in (em.get("applied_terms") or []) if t in forms.INTRINSIC_CORE]
        if terms:
            d = {}
            for t in terms:
                d[t] = float(share) * float(em.get(f"d{t}_px") or 0.0)
                p[forms.INTRINSIC_CORE.index(t)] += d[t]
            applied["early_mission"] = {"share": float(share), "applied_px": d, "before_sol": float(em["before_sol"])}
    return str(model["model"]), p, applied


def scaled(model_name: str, params: Sequence[float], s: float, dx: float = 0.0, dy: float = 0.0) -> List[float]:
    """The same camera at ``s`` x the resolution (focal length and principal point scale, the distortion does not),
    then its principal point moved by (dx, dy) pixels (extra padding)."""
    from ..colmap import scale_camera_params
    p = scale_camera_params(model_name, params, float(s))
    if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        p[1] += dx
        p[2] += dy
    else:
        p[2] += dx
        p[3] += dy
    return [float(v) for v in p]


# ----------------------------------------------------------------------------------------------------- rig
def drift_offset_mdeg(drift: Dict[str, Any], sol: float) -> Dict[str, float]:
    """The rig rotation offset (pitch, yaw, roll mdeg) of a drift model at ``sol``: rate x (sol - sol0), plus the
    hinge (early - rate) x (min(sol, knot_sol) - knot_ref) for an angle with an early rate before ``knot_sol``."""
    out = {}
    ds = float(sol) - float(drift["sol0"])
    for a in ("pitch", "yaw", "roll"):
        rate = float(drift.get(f"{a}_mdeg_per_sol") or 0.0)
        v = rate * ds
        if drift.get(f"{a}_early_mdeg_per_sol") is not None and drift.get("knot_sol"):
            v += (float(drift[f"{a}_early_mdeg_per_sol"]) - rate) * (min(float(sol), float(drift["knot_sol"]))
                                                                    - float(drift.get("knot_ref") or 0.0))
        out[f"{a}_mdeg"] = float(v)
    return out


def rig_at(rig: Dict[str, Any], sol: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    The stereo rig (x_sensor = R x_ref + t) of a rig file at ``sol``: the file's rotation turned by the drift offset
    at that sol (:func:`drift_offset_mdeg`; pitch, yaw, roll about x, y, z of the sensor camera).  Returns (R, t,
    applied): ``t`` is the fitted translation (``translation = "fitted"``) or None when the block keeps its CAHV
    baseline.
    """
    from scipy.spatial.transform import Rotation
    R = np.asarray(rig["R_sensor_from_ref"], float)
    applied: Dict[str, Any] = {}
    dr = rig.get("drift")
    if dr and sol is not None:
        off = drift_offset_mdeg(dr, float(sol))
        rv = np.radians(1e-3 * np.array([off["pitch_mdeg"], off["yaw_mdeg"], off["roll_mdeg"]]))
        R = Rotation.from_rotvec(rv).as_matrix() @ R
        applied["drift"] = {**dr, "sol_median": float(sol),
                            "offset_mdeg": {"pitch": off["pitch_mdeg"], "yaw": off["yaw_mdeg"], "roll": off["roll_mdeg"]}}
    t = None
    if rig.get("translation") == "fitted" and rig.get("t_sensor_from_ref_m") is not None:
        t = np.asarray(rig["t_sensor_from_ref_m"], float)
    return R, t, applied


# ------------------------------------------------------------------------------------------- rig epochs in sol
RIG_EPOCH_GAP_SOLS = 50.0        # Navcam visits of a block more than this many sols apart get their own rig rotation
RIG_EPOCH_BREAK_SOLS = (300.0,)  # and so do the images on each side of the pitch knot
RIG_EPOCH_MAX_SPAN_SOLS = 25.0   # an epoch spanning more sols is split at its middle sol


def sol_epochs(sols: Sequence[int], gap: float = RIG_EPOCH_GAP_SOLS, breaks: Sequence[float] = (),
               max_span: Optional[float] = None) -> List[Tuple[int, int]]:
    """Sol ranges of the clusters of ``sols`` separated by more than ``gap`` sols; a cluster is also cut between two
    consecutive sols a < b with a < break <= b for every sol in ``breaks`` (the pitch knot), and a cluster spanning
    more than ``max_span`` sols is split at its middle sol ((first + last) / 2; sols up to the middle in the first
    part), recursively."""
    u = sorted(set(int(x) for x in sols))
    if not u:
        return []
    brk = sorted(float(x) for x in (breaks or ()))
    out, start = [], u[0]
    for a, b in zip(u, u[1:]):
        if b - a > gap or any(a < k <= b for k in brk):
            out.append((start, a))
            start = b
    out.append((start, u[-1]))
    if max_span is not None and max_span > 0:
        def _split(lo: int, hi: int) -> List[Tuple[int, int]]:
            mem = [x for x in u if lo <= x <= hi]
            if hi - lo <= max_span or len(mem) < 2:
                return [(lo, hi)]
            mid = 0.5 * (lo + hi)
            left, right = [x for x in mem if x <= mid], [x for x in mem if x > mid]
            if not left or not right:
                return [(lo, hi)]
            return _split(left[0], left[-1]) + _split(right[0], right[-1])
        out = [piece for a_, b_ in out for piece in _split(a_, b_)]
    return out


# -------------------------------------------------------------------------------------------------- Mastcam-Z
def focus_state(listed: bool, group: Any = None) -> str:
    """The backlash state of a Mastcam-Z image: ``"single"`` at a zoom without states (110 mm, ``group`` its eye
    group or camera key; the lists never apply there), ``"regular"`` (the low-focal-length state) when the
    low-backlash list, the found list or its own focal fit says so, else the default ``"backlash"`` (the
    high-focal-length state)."""
    if group is not None and forms.is_single_state(group):
        return forms.ZCAM_SINGLE_STATE
    return "regular" if listed else ZCAM_DEFAULT_STATE


def zcam_line_value(entry: Dict[str, Any], focus: float, reference_focus: float,
                    T_lens: Optional[float] = None) -> float:
    """p = a + b (focus - reference_focus) [+ c_per_degC (T_lens - T0_degC)] of one focus line."""
    v = float(entry["a"]) + float(entry.get("b") or 0.0) * (float(focus) - float(reference_focus))
    if entry.get("c_per_degC") and T_lens is not None and entry.get("T0_degC") is not None:
        v += float(entry["c_per_degC"]) * (float(T_lens) - float(entry["T0_degC"]))
    return v


def zcam_lines_at(gm: Dict[str, Any], focus: float, state: str = ZCAM_DEFAULT_STATE,
                  T_lens: Optional[float] = None) -> Tuple[Dict[str, float], str]:
    """{param: value} of an eye's calibrated lines (``gm["lines"][state]``) at ``focus``; a state without lines
    follows the other state's, fx and fy divided by ``regular_ratio`` for the regular state.  Returns (values,
    source note)."""
    lmodel = gm.get("lines") or {}
    ref = float(lmodel.get("reference_focus", gm.get("reference_focus", focus)))
    if state == forms.ZCAM_SINGLE_STATE:                # 110 mm: its one state's lines (stored as the backlash lines)
        state = "backlash"
    own = lmodel.get(state) or {}
    other = "backlash" if state == "regular" else "regular"
    src = state
    if not own:
        own = lmodel.get(other) or {}
        src = f"{other} lines" + (" / regular_ratio" if state == "regular" else " x regular_ratio")
    out = {}
    for pn, e in own.items():
        if pn in forms.FULL_OPENCV_PARAMS and isinstance(e, dict) and "a" in e:
            out[pn] = zcam_line_value(e, focus, ref, T_lens)
    if src != state and gm.get("regular_ratio"):
        k = float(gm["regular_ratio"]) if state == "regular" else 1.0 / float(gm["regular_ratio"])
        for pn in ("fx", "fy"):
            if pn in out:
                out[pn] /= k
    return out, src


def zcam_camera_at(zoom_model: Dict[str, Any], group: str, focus: float, state: str = ZCAM_DEFAULT_STATE,
                   T_lens: Optional[float] = None) -> Tuple[str, List[float], Dict[str, Any]]:
    """
    The full-frame FULL_OPENCV camera of a Mastcam-Z eye (``group`` ``"ZL034"`` / ``"ZR034"``) at a focus count and
    backlash state from a zoom's model: fx, fy, k1, k2 (and cx, cy on ZR) on the state's lines, the principal point
    at ``pp0_px`` moved by the ``pp`` slopes about the reference focus where no line gives it, p1 = p2 = k3 = 0.
    Returns (model name, params, {"source", "lines"}).
    """
    cams = zoom_model.get("cameras") or {}
    gm = cams.get(group)
    if gm is None:
        raise KeyError(f"{group} is not in the zoom model ({sorted(cams)})")
    if forms.is_single_state(group) or state == forms.ZCAM_SINGLE_STATE:
        state = "backlash"                              # 0.93.5: 110 mm has one state, its lines are the backlash ones
    if state not in forms.ZCAM_STATES:
        raise ValueError(f"state {state!r}: use one of {forms.ZCAM_STATES}")
    ref = float(gm.get("reference_focus", 600.0))
    p = [0.0] * 12
    vals, src = zcam_lines_at(gm, focus, state, T_lens)
    if "fx" not in vals or "fy" not in vals:            # no lines: f0 + slope (focus - ref), the aspect of the file
        f = float(gm["f0_px"]) + float(gm.get("slope_px_per_count") or 0.0) * (float(focus) - ref)
        if state == "regular" and gm.get("regular_ratio"):
            f /= float(gm["regular_ratio"])
        a = float(gm.get("aspect", 1.0))
        vals.setdefault("fx", f / np.sqrt(a))
        vals.setdefault("fy", f * np.sqrt(a))
        src = f"f0 + slope ({state})"
    pp0 = gm.get("pp0_px") or [None, None]
    pp = gm.get("pp") or {}
    for i, pn in ((2, "cx"), (3, "cy")):
        if pn in vals:
            p[i] = vals[pn]
        elif pp0[i - 2] is not None:
            p[i] = float(pp0[i - 2]) + float(pp.get(f"{pn}_px_per_count") or 0.0) * (float(focus) - ref)
    p[0], p[1] = float(vals["fx"]), float(vals["fy"])
    dist = gm.get("distortion") or {}
    for n, v in zip(dist.get("names") or [], dist.get("params") or []):
        if n in forms.FULL_OPENCV_PARAMS[4:] and n not in forms.ZCAM.held_zero:
            p[forms.FULL_OPENCV_PARAMS.index(n)] = float(v)
    for pn in ("k1", "k2"):
        if pn in vals:
            p[forms.FULL_OPENCV_PARAMS.index(pn)] = vals[pn]
    for n in forms.ZCAM.held_zero:
        p[forms.FULL_OPENCV_PARAMS.index(n)] = 0.0
    # the zoom's fixed distortion terms (forms.ZCAM_FIXED_TERMS) are not forced here: the promoted model's own k1, k2
    # belong together (k1 and k2 correlate 0.98); the block BA holds the fixed values and the joint fits with them
    return forms.ZCAM.colmap_model, p, {"source": f"{group} at focus {float(focus):g} ({src})", "lines": sorted(vals)}


# -------------------------------------------------------------------------------------------- label fallback
FIT_STEP_FRACTION = 80.0 / 5120.0       # the label fit's grid step as a share of the frame (80 px at full frame)


def label_camera(meta: Dict[str, Any]) -> Tuple[str, List[float], str]:
    """
    The start camera of an image from its PDS label where no promoted model applies: a Mastcam-Z (or any
    CAHVOR) label gives FULL_OPENCV from the pinhole intrinsics with the label's k1, k2 and p1 = p2 = k3 = 0; a
    Navcam / Hazcam CAHVORE label gives the THIN_PRISM_FISHEYE fitted to it over the frame
    (:func:`mppp.calib.labelfit.fisheye_from_cahvore`; 0.93), or when that fails, for type 2, the closed form
    f (1 + r0), k1 = r1 / (1 + r0), k2 = r2 / (1 + r0), the other terms 0.  ``meta``: an ``MPPPImage.meta`` dict (``intrinsics_label`` /
    ``intrinsics``, ``camera_model_label``) - the result is at the image's padded resolution.  Returns (model
    name, params, source).
    """
    from ..colmap import colmap_camera_params
    intr = meta.get("intrinsics_label") or meta["intrinsics"]
    fam = str((meta.get("filename") or {}).get("family") or meta.get("source_product", "")[:1]).upper()
    cm = meta.get("camera_model_label") or {}
    if fam in ("N", "F", "R") and str(cm.get("MODEL_TYPE", "")).upper() == "CAHVORE":
        # 0.93: the least-squares fit over the frame (mppp.calib.labelfit; 0.13-0.2 px rms where the closed form
        # below misses by 1.4-2.5 px: the O tilt, and the rear Hazcams' entrance-pupil terms)
        try:
            from ..cmod import CameraModel
            from .labelfit import fisheye_from_cahvore
            K = np.asarray(intr["K"], float)
            m0 = CameraModel.from_label(cm)
            _, lin = m0.decompose()
            W, H = int(intr["width"]), int(intr["height"])
            mp = m0.rescaled(1.0, float(K[0, 2]) - lin["hc"], float(K[1, 2]) - lin["vc"], W, H)
            p, rep = fisheye_from_cahvore(mp, W, H, step=max(8.0, FIT_STEP_FRACTION * max(W, H)))
            return "THIN_PRISM_FISHEYE", p, f"label CAHVORE fit ({rep['rms_px']:.2f} px rms over the frame)"
        except Exception:                                   # noqa: BLE001 - the closed form below
            pass
        R = cm.get("MODEL_COMPONENT_6") or [0.0, 0.0, 0.0]
        mtype = int(round(float(cm.get("MODEL_COMPONENT_8", 2))))
        if mtype == 2:
            K = np.asarray(intr["K"], float)
            r0, r1, r2 = (float(x) for x in R[:3])
            g = 1.0 + r0
            p = [K[0, 0] * g, K[1, 1] * g, K[0, 2] + 0.5, K[1, 2] + 0.5, r1 / g, r2 / g, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return "THIN_PRISM_FISHEYE", p, "label CAHVORE fisheye (f (1 + r0), k1 = r1 / (1 + r0), k2 = r2 / (1 + r0))"
    model, p = colmap_camera_params(intr)
    full = [0.0] * 12
    full[:len(p)] = [float(v) for v in p]
    if model == "FULL_OPENCV":
        for n in forms.ZCAM.held_zero:
            full[forms.FULL_OPENCV_PARAMS.index(n)] = 0.0
        return model, full, "label CAHVOR (k1, k2; p1 = p2 = k3 = 0)"
    if model == "OPENCV":
        full[6] = full[7] = 0.0
        return "FULL_OPENCV", full, "label CAHVOR (k1, k2; p1 = p2 = k3 = 0)"
    return "PINHOLE", full[:4], "label CAHV (pinhole)"


# ------------------------------------------------------------------------------------------- plots (0.90)
# The refined formats of the calibration study, ported unchanged (project doc "plot formats", 20 Oct 2026):
# docs/history/results/figures_20261018/fig_rig.py (+ fig_rig_trends_20oct.py) and fig_intr.py.
IFOV_MDEG = 18.9          # one Navcam pixel, 0.33 mrad
FIG_DPI = 200             # the standard figures (make_figures.bat; Christian, 20 Oct)
RIG_TREND_YLIMS = {"pitch": (-10.0, 30.0), "yaw": (-20.0, 20.0), "roll": (-20.0, 20.0)}   # mdeg (Christian, 22 Oct)
STEREO_CX_YLIM = (-20.0, -16.0)        # px, the rig trends' stereo cx panel (D), primary axis (Christian, 22 Oct; top -16 since 27 Oct)
EYE_COLORS = {"NL": "C0", "NR": "C3"}        # left eye cool, right eye warm


def panel_letters(axs, fontsize: Optional[float] = None) -> None:
    """Letter the visible subplots A, B, C ... in reading order (row by row) as the maps do (Christian, 27 Oct): the
    letter starts the title - "A. pitch: ..." - and the title is left-justified.  Call after the titles are set;
    ``fontsize`` overrides the title's own size."""
    for k, ax in enumerate(a for a in np.ravel(axs) if a.axison):
        centre, left = ax.get_title(loc="center"), ax.get_title(loc="left")
        size = fontsize or (ax.title.get_fontsize() if centre else ax._left_title.get_fontsize())
        text = left or centre
        ax.set_title("", loc="center")
        ax.set_title(f"{chr(ord('A') + k)}. {text}" if text else f"{chr(ord('A') + k)}.", loc="left", fontsize=size)


def _out_of_axis_note(ax, y, lo, hi, what="epoch(s)") -> None:
    y = np.asarray(y, float)
    n_out = int(np.sum(np.isfinite(y) & ((y < lo) | (y > hi))))
    ax.set_ylim(lo, hi)
    if n_out:
        ax.text(0.01, 0.98, f"{n_out} {what} outside the axis", transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color="0.3")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def drift_offset_rel(drift: Dict[str, Any], angle: str, sols, sol_ref: float = 0.0) -> np.ndarray:
    """The drift offset of ``angle`` at ``sols`` relative to the model at ``sol_ref`` (:func:`drift_offset_mdeg`),
    so that every run's angles are referred to the rig at sol 0 whatever the fit's own reference."""
    s = np.atleast_1d(np.asarray(sols, float))
    v = np.array([drift_offset_mdeg(drift, x)[f"{angle}_mdeg"] for x in s])
    return v - drift_offset_mdeg(drift, float(sol_ref))[f"{angle}_mdeg"]


def plot_rig_trends(rig_residuals: Dict[str, Any], consensus: Dict[str, Any], drift: Dict[str, Any], path: PathLike,
                    cx_inuse_px_per_degC: Optional[float] = None, label: str = "", sol_ref: float = 0.0,
                    T0: float = NAVCAM_T0_DEGC, model_label: str = "drift model",
                    ylims: Optional[Dict[str, Tuple[float, float]]] = None, dpi: int = FIG_DPI,
                    cx_ylim: Optional[Tuple[float, float]] = STEREO_CX_YLIM, point_labels: bool = True) -> Path:
    """
    Rig trends 2 x 2 (fig_rig.py, 12.5 x 8.2 in, dpi 200): pitch (UL), roll (UR) and yaw with its temperature
    slope removed to T0 (LL) of every block rig epoch against its median sol ("sol epoch"), each relative to the joint rig at
    ``sol_ref`` (= 0), with the ``drift`` model line, dashed lines at the knot, the residual-trend statistics in the
    titles and a secondary IFOV axis; LR: the stereo pointing NR cx − NL cx against epoch temperature with the
    model slope (dashed), the slope in use for the block runs (dotted) and the slope with the yaw residual folded
    in (solid, ± se band).  ``rig_residuals`` is the study's ``rig_residuals.json`` (rig rotation free, cameras
    held), ``consensus`` its ``consensus.json``.  ``ylims``: per angle (default :data:`RIG_TREND_YLIMS`: pitch
    -10 to +30, yaw and roll -20 to +20 mdeg), ``cx_ylim`` the stereo panel's primary axis (default
    :data:`STEREO_CX_YLIM`, -20 to -16 px; None = from the data); points outside are counted on the panel.
    Panels lettered A-D; ``point_labels=False`` leaves out the block name at every point.
    """
    plt = _plt()
    rr, cons = rig_residuals, consensus
    ylims = dict(RIG_TREND_YLIMS if ylims is None else ylims)
    B = rr["blocks"]; names = [b["block"] for b in B]
    sol = np.array([b["sol_median"] for b in B], float); T = np.array([b["T_median_degC"] for b in B], float)
    ppl = cons["thermal"]["pp_slopes"]
    CX_T = ppl["NR"][0] - ppl["NL"][0]
    CX0 = cons["NR"]["cx"] - cons["NL"]["cx"]

    def annotate(ax, x, y, nm):
        if not point_labels:
            return
        for xx, yy, n in zip(x, y, nm):
            ax.annotate(n.replace(" [sols ", " [")[:22], (xx, yy), fontsize=5.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ss = np.linspace(0, 2050, 400)
    panels = {"pitch": axs[0][0], "roll": axs[0][1], "yaw": axs[1][0]}     # roll upper right (Christian, 22 Oct)
    yaw_T = rr["fits"]["yaw"]["vs_T"]
    for ang, ax in panels.items():
        tot = np.array([b[f"{ang}_mdeg"] for b in B]) + drift_offset_rel(drift, ang, sol, sol_ref)
        if ang == "yaw":
            tot = tot - yaw_T["slope"] * (T - T0)
            lab = f"block rig epoch, {yaw_T['slope']:+.3f} ± {yaw_T['se']:.3f} mdeg/°C removed (to {T0:.0f} °C)"
        else:
            lab = "block rig epoch (rig rotation free, cameras held)"
        ax.plot(sol, tot, "o", color="C0", label=lab)
        ax.plot(ss, drift_offset_rel(drift, ang, ss, sol_ref), "-", color="C3", label=model_label)
        annotate(ax, sol, tot, names)
        r = tot - drift_offset_rel(drift, ang, sol, sol_ref)
        rate = 1000 * (drift.get(f"{ang}_mdeg_per_sol") or 0.0)
        f = rr["fits"][ang]["vs_sol"]
        extra = (f"; {1000 * drift['pitch_early_mdeg_per_sol']:+.1f} before sol {drift['knot_sol']:.0f}"
                 if ang == "pitch" and drift.get("pitch_early_mdeg_per_sol") is not None else "")
        ax.set_title(f"{ang}: drift {rate:+.2f} mdeg/1000 sol{extra}; residual trend {f['slope']:+.2f} ± {f['se']:.2f} (z {f['z']:+.1f})\n"
                     f"epoch sd about the model {r.std(ddof=1):.2f} mdeg = {r.std(ddof=1) / IFOV_MDEG:.3f} px", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.8)
        if drift.get("knot_sol"):
            ax.axvline(drift["knot_sol"], color="0.4", ls="--", lw=0.8)
        ax.set_xlabel("sol epoch", fontsize=8)
        ax.set_ylabel(f"{ang} relative to the joint rig at sol {sol_ref:.0f} (mdeg)", fontsize=8)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    for ang, a in panels.items():
        if ang in ylims:
            lo, hi = ylims[ang]
            y = np.concatenate([ln.get_ydata() for ln in a.get_lines() if ln.get_marker() == "o"] or [np.array([])])
            _out_of_axis_note(a, y, lo, hi)
        a.set_xlim(0, 2050)
        a.secondary_yaxis("right", functions=(lambda m: m / IFOV_MDEG, lambda p: p * IFOV_MDEG)).set_ylabel("Navcam IFOV equivalent (px)", fontsize=8)
    ax = axs[1][1]
    yaw_px = np.array([b["yaw_mdeg"] for b in B]) / IFOV_MDEG
    cx_blk = CX0 + CX_T * (T - T0) + yaw_px
    corr = CX_T + yaw_T["slope"] / IFOV_MDEG
    tt = np.linspace(np.nanmin(T) - 2, np.nanmax(T) + 2, 2)
    ax.plot(T, cx_blk, "o", color="C3", label="epoch stereo pointing as NR cx − NL cx (model + rig-yaw residual)")
    ax.plot(tt, CX0 + CX_T * (tt - T0), "--", color="0.4", label=f"{model_label}: {CX_T:+.4f} px/°C (NR {ppl['NR'][0]:+.4f}, NL {ppl['NL'][0]:+.4f})")
    if cx_inuse_px_per_degC is not None:
        ax.plot(tt, CX0 + cx_inuse_px_per_degC * (tt - T0), ":", color="0.6", label=f"in use for these blocks: {cx_inuse_px_per_degC:+.4f} px/°C")
    ax.fill_between(tt, CX0 + (corr - yaw_T["se"] / IFOV_MDEG) * (tt - T0), CX0 + (corr + yaw_T["se"] / IFOV_MDEG) * (tt - T0), color="C3", alpha=0.12, lw=0)
    ax.plot(tt, CX0 + corr * (tt - T0), "-", color="C3", label=f"with the yaw residual folded in: {corr:+.4f} ± {yaw_T['se'] / IFOV_MDEG:.4f} px/°C")
    annotate(ax, T, cx_blk, names)
    ax.axvline(T0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("epoch median Navcam temperature (°C)", fontsize=8); ax.set_ylabel("NR cx − NL cx (px)", fontsize=8)
    ax.set_title(f"stereo cx against temperature: yaw residual vs T {yaw_T['slope']:+.2f} ± {yaw_T['se']:.2f} mdeg/°C (z {yaw_T['z']:+.1f})\n"
                 f"(scatter about the line {np.std(cx_blk - (CX0 + corr * (T - T0)), ddof=2):.3f} px)", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
    if cx_ylim is not None:
        _out_of_axis_note(ax, cx_blk, *cx_ylim)
    ax.secondary_yaxis("right", functions=(lambda p: (p - CX0) * IFOV_MDEG, lambda m: CX0 + m / IFOV_MDEG)).set_ylabel(f"yaw equivalent about cx at {T0:.0f} °C (mdeg)", fontsize=8)
    panel_letters(axs)
    nb = len({b.get("site") or b["block"] for b in B})
    fig.suptitle(f"Navcam rig trends{(' ' + label) if label else ''} ({nb} blocks, {len(B)} rig epochs; drift referred to sol {sol_ref:.0f}; "
                 f"right axes in Navcam IFOV, 0.33 mrad = {IFOV_MDEG} mdeg)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


HAZCAM_IFOV_MDEG = float(np.degrees(1.0 / 2142.0) * 1e3)    # one Hazcam pixel (f about 2142 px): 26.7 mdeg


def plot_hazcam_rig_trends(rig: Dict[str, Any], path: PathLike, label: str = "", dpi: int = FIG_DPI,
                           point_labels: bool = True) -> Optional[Path]:
    """0.94.5 (Christian, 27 Oct: "make similar figures for the front and rear hazcams"): a Hazcam pair's rig trends,
    1 x 3 like the Navcam's (pitch, roll, yaw): every block's refined rig rotation (``rig["per_block"]`` from
    :func:`mppp.calib.hazfit.fit_rig`) relative to the fitted rig at sol 0 against the block's sol, with the drift
    line, the scatter about it in the titles and a secondary axis in Hazcam IFOV.  None without per-block rows."""
    rows = [r for r in (rig.get("per_block") or []) if r.get("sol") is not None]
    if not rows:
        return None
    plt = _plt()
    fam = rig.get("family", "?")
    sol = np.array([r["sol"] for r in rows], float)
    d = rig.get("drift") or {}
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.6), squeeze=False)
    ss = np.linspace(0, max(2050.0, float(sol.max()) + 50), 200)
    for ax, ang in zip(axs[0], ("pitch", "roll", "yaw")):
        y = np.array([r[f"{ang}_mdeg"] for r in rows], float)
        rate = float(d.get(f"{ang}_mdeg_per_sol") or 0.0)
        ax.plot(sol, y, "o", color="C0", label="block rig (rotation free, cameras held)")
        if d:
            ax.plot(ss, rate * ss, "-", color="C3", label=f"drift {1000 * rate:+.2f} mdeg/1000 sol")
        if point_labels:
            for xx, yy, r in zip(sol, y, rows):
                ax.annotate(str(r["block"])[:22], (xx, yy), fontsize=5.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
        res = y - rate * sol
        sd = float(np.std(res, ddof=1)) if len(res) > 1 else float("nan")
        ax.set_title(f"{ang}: {len(rows)} blocks; scatter about the {'drift' if d else 'median'} {sd:.1f} mdeg "
                     f"= {sd / HAZCAM_IFOV_MDEG:.2f} px", fontsize=9)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xlim(0, ss[-1]); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
        ax.set_xlabel("sol", fontsize=8)
        ax.set_ylabel(f"{ang} relative to the rig at sol 0 (mdeg)", fontsize=8)
        ax.secondary_yaxis("right", functions=(lambda m: m / HAZCAM_IFOV_MDEG, lambda q: q * HAZCAM_IFOV_MDEG)).set_ylabel(
            "Hazcam IFOV equivalent (px)", fontsize=8)
    panel_letters(axs)
    name = {"F": "front", "R": "rear"}.get(fam, fam)
    fig.suptitle(f"{name} Hazcam rig trends{(' ' + label) if label else ''} ({len(rows)} blocks; right axes in Hazcam "
                 f"IFOV, {HAZCAM_IFOV_MDEG:.1f} mdeg)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def plot_intrinsics_vs_temperature(consensus: Dict[str, Any], inuse: Dict[str, Dict[str, Any]], bins: Sequence[Dict[str, Any]],
                                   rig_residuals: Dict[str, Any], path: PathLike, early_sol: float = 380.0,
                                   label: str = "", T0: float = NAVCAM_T0_DEGC, model_label: str = "new model",
                                   pp_units: Optional[Sequence[Dict[str, Any]]] = None, dpi: int = FIG_DPI) -> Path:
    """
    Intrinsics against temperature 2 x 2 (fig_intr.py, 12.5 x 8.6 in, sharex): columns NL (cool) / NR (warm), rows
    focal length and principal point as the change from T0; left/right share y range and ticks.  Points: the
    per-temperature-bin cameras of the block runs (``bins`` rows {eye, block, T, n, fx, fy}, size ~ sqrt images,
    blocks before ``early_sol`` hollow), solid lines the ``consensus`` model, dotted grey the model ``inuse``
    ({"NL": model, "NR": model}, what the block runs started from); the stereo cx of the rig epochs on the NR
    principal-point panel.  Secondary axes in ppm of f and pointing mdeg; T0 dashed; grey band = bin temperatures.
    """
    plt = _plt()
    cons, rr = consensus, rig_residuals
    sols: Dict[str, List[float]] = {}
    for b in rr["blocks"]:
        sols.setdefault(b.get("site") or b["block"], []).append(b["sol_median"])
    early_blk = {k for k, v in sols.items() if min(v) < early_sol}
    ax_ppm = cons["thermal"]["axis_ppm"]; pp = cons["thermal"]["pp_slopes"]
    yawT = rr["fits"]["yaw"]["vs_T"]
    CX2 = pp["NR"][0] - pp["NL"][0]; CX_ALL = CX2 + yawT["slope"] / IFOV_MDEG; CX_ALL_SE = yawT["se"] / IFOV_MDEG
    B = rr["blocks"]; Tblk = np.array([b["T_median_degC"] for b in B]); yaw = np.array([b["yaw_mdeg"] for b in B])
    tt = np.linspace(-60, 0, 121)
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    for j, eye in enumerate(("NL", "NR")):
        col = EYE_COLORS[eye]; f2 = cons[eye]; iu = inuse[eye]; th_iu = thermal_terms(iu.get("thermal")); p_iu = iu["params"]
        rows = [r for r in bins if r["eye"] == eye]
        ax = axs[0][j]
        for k, (p, ls, mk, alpha) in enumerate((("fx", "-", "o", 0.8), ("fy", "--", "s", 0.35))):
            ppm2 = ax_ppm[eye][k]; ppm_iu = th_iu[f"{p}_ppm"]
            ax.plot(tt, f2[p] * (1 + 1e-6 * ppm2 * (tt - T0)), ls, color=col, lw=1.4,
                    label=f"{p} {model_label}: {f2[p]:.2f} px at {T0:.0f} °C, {ppm2:+.1f} ppm/°C ({f2[p] * 1e-6 * ppm2:+.4f} px/°C)")
            if k == 0:
                ax.plot(tt, p_iu[k] * (1 + 1e-6 * ppm_iu * (tt - T0)), ":", color="0.4", lw=1.1,
                        label=f"fx in use for the block runs: {p_iu[k]:.2f} px, {ppm_iu:+.1f} ppm/°C")
            for early, mfc in ((False, None), (True, "white")):
                rs = [r for r in rows if (r["block"] in early_blk) == early]
                if not rs:
                    continue
                Tb = np.array([r["T"] for r in rs]); nb = np.array([r["n"] for r in rs])
                v = np.array([r[p] for r in rs])                           # the bin's focal length itself (px)
                ax.scatter(Tb, v, s=6 + 1.5 * np.sqrt(nb * 10), marker=mk, facecolors=mfc or col, edgecolors=col, alpha=alpha, zorder=3,
                           label=(f"{p} per temperature bin of the block runs (size ~ images)" if not early else
                                  f"{p}, blocks before sol {early_sol:.0f} (early-mission offset not removed)") if p == "fx" else None)
        ax.axhline(f2["fx"], color="0.6", lw=0.6)
        ax.set_title(f"{eye}: focal length", fontsize=10)
        ax.set_ylabel("focal length (full-frame px)", fontsize=8)
        ax.secondary_yaxis("right", functions=(lambda f, f0=f2["fx"]: f - f0, lambda d, f0=f2["fx"]: d + f0)).set_ylabel(
            f"Δf from {T0:.0f} °C (px; {model_label} fx)", fontsize=8)
        ax.legend(fontsize=6.5, loc="lower right")
        ax = axs[1][j]
        cx_iu = th_iu["cx_px"]
        if eye == "NL":
            ax.plot(tt, cx_iu * (tt - T0), ":", color="0.4", lw=1.2, label=f"cx in use for the block runs: {cx_iu:+.4f} px/°C")
            ax.plot(tt, pp["NL"][0] * (tt - T0), "-", color=col, lw=1.3, label=f"cx {model_label}: {pp['NL'][0]:+.4f} px/°C")
            ax.plot(tt, pp["NL"][1] * (tt - T0), "-.", color=col, lw=0.8, label=f"cy {model_label}: {pp['NL'][1]:+.4f} px/°C")
        else:
            ax.plot(tt, cx_iu * (tt - T0), ":", color="0.4", lw=1.2, label=f"cx in use for the block runs: {cx_iu:+.4f} px/°C")
            ax.plot(tt, pp["NR"][0] * (tt - T0), "--", color=col, lw=1.2, label=f"cx {model_label}: {pp['NR'][0]:+.4f} px/°C")
            ax.fill_between(tt, (CX_ALL - CX_ALL_SE) * (tt - T0), (CX_ALL + CX_ALL_SE) * (tt - T0), color=col, alpha=0.12, lw=0)
            ax.plot(tt, CX_ALL * (tt - T0), "-", color=col, lw=1.6, label=f"stereo cx (NR − NL) with the yaw residual folded in: {CX_ALL:+.4f} ± {CX_ALL_SE:.4f} px/°C")
            ax.plot(tt, pp["NR"][1] * (tt - T0), "-.", color=col, lw=0.8, label=f"cy {model_label}: {pp['NR'][1]:+.4f} px/°C")
            y = CX2 * (Tblk - T0) + yaw / IFOV_MDEG
            ax.plot(Tblk, y, "o", color=col, ms=5, mec="white", mew=0.5, zorder=3, label="rig epoch stereo pointing as NR cx − NL cx (model + yaw residual)")
        pu = [r for r in (pp_units or []) if r["eye"] == eye]
        if pu:                               # 0.91: the pointing pass - one per block and era, rigs at the trend
            sl = pp[eye]
            for comp, mk, k in (("cx", "D", 0), ("cy", "v", 1)):
                Tu = np.array([r["T"] for r in pu]); vu = np.array([r[f"d{comp}"] for r in pu]) + sl[k] * (Tu - T0)
                eu = np.array([np.nan if r.get(f"sd_{comp}") is None else r[f"sd_{comp}"] for r in pu])
                ax.errorbar(Tu, vu, yerr=np.where(np.isfinite(eu), eu, 0), fmt=mk, ms=4, color=col, mfc=col if comp == "cx" else "white",
                            ecolor=col, elinewidth=0.6, alpha=0.75, zorder=3,
                            label=f"{comp} per block (pointing pass: rigs at the trend{', 1 px prior' if eye == 'NL' else ''}; ± sd)")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(f"{eye}: principal point", fontsize=10)
        ax.set_ylabel(f"Δ principal point from {T0:.0f} °C (full-frame px)", fontsize=8)
        ax.set_xlabel("Navcam camera temperature (°C)", fontsize=9)
        ax.secondary_yaxis("right", functions=(lambda d: d * IFOV_MDEG, lambda m: m / IFOV_MDEG)).set_ylabel(f"pointing equivalent (mdeg; 1 px = {IFOV_MDEG} mdeg)", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right" if eye == "NR" else "upper left")
    Tb_all = np.array([r["T"] for r in bins]) if bins else np.array([T0])
    # the focal row shows each eye's own f (px): the two panels share the span and tick spacing about their own
    # f(T0); the principal-point row shares its limits and ticks outright
    f0 = {eye: cons[eye]["fx"] for eye in ("NL", "NR")}
    half = max(max(abs(a.get_ylim()[0] - f0[e]), abs(a.get_ylim()[1] - f0[e])) for a, e in zip(axs[0], ("NL", "NR")))
    step = float(np.diff(axs[0][0].get_yticks()[:2])[0]) if len(axs[0][0].get_yticks()) > 1 else 2.0
    for a, e in zip(axs[0], ("NL", "NR")):
        a.set_ylim(f0[e] - half, f0[e] + half)
        a.set_yticks(np.arange(np.ceil((f0[e] - half) / step) * step, f0[e] + half, step))
    lo = min(axs[1][0].get_ylim()[0], axs[1][1].get_ylim()[0]); hi = max(axs[1][0].get_ylim()[1], axs[1][1].get_ylim()[1])
    for a in axs[1]:
        a.set_ylim(lo, hi)
    axs[1][1].set_yticks(axs[1][0].get_yticks()); axs[1][1].set_ylim(lo, hi)
    for a in axs.flat:
        a.axvline(T0, color="0.4", ls="--", lw=0.8)
        a.axvspan(Tb_all.min(), Tb_all.max(), color="0.85", alpha=0.35, lw=0, zorder=0); a.grid(alpha=0.3); a.set_xlim(-60, 0)
    panel_letters(axs)
    nb = len(sols)
    fig.suptitle(f"Navcam intrinsic temperature dependence{(' ' + label) if label else ''} ({nb} blocks; T0 = {T0:.0f} °C; grey band: temperatures of the bins)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


ZCAM_PIXEL_MM = 0.0074            # Mastcam-Z detector pixel pitch (mm): efl = f [px] x pitch


def plot_zcam_efl_vs_focus(res: Dict[str, Any], path: PathLike, dpi: int = FIG_DPI, label: str = "") -> Optional[Path]:
    """0.95.1 (Christian, 29 Oct: "the Z34 efl versus focus curves"): a Mastcam-Z joint's focal length against focus
    count, left eye (cool) and right eye (warm) side by side on the same y range.  Top: the joint's law (high
    backlash state solid, low state dashed) and every bin's own fx from the diagnostic pass (``diagnostics.bins``:
    one colour per block, the marker area by observations, open for the low state, x for a thin bin); efl in mm on
    the right axis (x 7.4 um).  Bottom: the bins about the law in ppm.  A joint written before 0.95.1 has no bins:
    the blocks' own lines (``diagnostics.blocks``) are drawn instead.  The reference focus dashed.  ``res``: the
    ``zcam_joint.json`` content.  None without eyes."""
    eyes = sorted((res.get("eyes") or {}).keys())
    if not eyes:
        return None
    plt = _plt()
    diag = res.get("diagnostics") or {}
    bins = diag.get("bins") or []
    blocks = sorted({b["block"] for b in bins} or {r["block"] for r in diag.get("blocks") or []})
    cm = plt.get_cmap("tab20")
    bcol = {b: cm(i % 20) for i, b in enumerate(blocks)}
    ecol = {e: ("#1f5fbf" if e.startswith("ZL") else "#d4572a") for e in eyes}       # left cool, right warm
    fig, axs = plt.subplots(2, len(eyes), figsize=(7.2 * len(eyes), 8.4), squeeze=False,
                            gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    focus_all = [b["focus"] for b in bins] or [0.0]
    lo_F = min(focus_all) if bins else None
    hi_F = max(focus_all) if bins else None
    tops, bots = [], []
    for j, eye in enumerate(eyes):
        e = res["eyes"][eye]
        t = e.get("terms") or {}
        ref = float(e.get("reference_focus") or 600.0)
        F0, F1 = (lo_F - 50, hi_F + 50) if bins else (ref - 500, ref + 500)
        F = np.linspace(F0, F1, 200)
        a_h, b = float(t.get("a_fx_backlash", np.nan)), float(t.get("b", 0.0))
        law_h = a_h + b * (F - ref)
        ax, axr = axs[0][j], axs[1][j]
        ax.plot(F, law_h, color=ecol[eye], lw=2.2, label="joint law, high state", zorder=3)
        a_l = t.get("a_fx_regular")
        held = set(e.get("held") or [])
        if a_l is not None and "a_fx_regular" not in held and np.isfinite(a_l):
            ratio = a_h / float(a_l)
            ax.plot(F, float(a_l) + b / ratio * (F - ref), color=ecol[eye], lw=1.6, ls="--", label="joint law, low state",
                    zorder=3)
        else:
            ratio = None
        eb = [x for x in bins if x["eye"] == eye]
        if eb:
            for blk in blocks:
                xs = [x for x in eb if x["block"] == blk]
                if not xs:
                    continue
                for st in ("backlash", "regular"):
                    pts = [x for x in xs if x["state"] == st]
                    if not pts:
                        continue
                    Fp = np.array([x["focus"] for x in pts]); fx = np.array([x["fx"] for x in pts])
                    sz = 12 + 60 * np.sqrt(np.array([x["observations"] for x in pts]) / max(1, max(x["observations"] for x in eb)))
                    thin = np.array([x.get("thin", False) for x in pts])
                    law = (a_h + b * (Fp - ref)) if st == "backlash" else (
                        (float(a_l) + b / ratio * (Fp - ref)) if ratio else np.full_like(Fp, np.nan))
                    face = bcol[blk] if st == "backlash" else "none"
                    ax.scatter(Fp[~thin], fx[~thin], s=sz[~thin], facecolors=face, edgecolors=bcol[blk], lw=0.8,
                               alpha=0.85, label=(blk if st == "backlash" and j == len(eyes) - 1 else None), zorder=4)
                    if thin.any():
                        ax.scatter(Fp[thin], fx[thin], s=20, marker="x", color=bcol[blk], lw=0.8, zorder=4)
                    axr.scatter(Fp, 1e6 * (fx - law) / law, s=sz * 0.6, facecolors=face, edgecolors=bcol[blk], lw=0.7,
                                alpha=0.85, zorder=4)
                    bots += list(1e6 * (fx[~thin] - law[~thin]) / law[~thin])
                    tops += list(fx[~thin])
        else:                                                # a joint before 0.95.1: the blocks' own lines
            for r in diag.get("blocks") or []:
                ln = (((r.get("lines") or {}).get(eye) or {}).get("backlash") or {}).get("fx")
                if not ln:
                    continue
                y = ln["a"] + ln["b"] * (F - ref)
                ax.plot(F, y, color=bcol[r["block"]], lw=1.0, alpha=0.9,
                        label=(r["block"] if j == len(eyes) - 1 else None), zorder=2)
                axr.plot(F, 1e6 * (y - law_h) / law_h, color=bcol[r["block"]], lw=1.0, alpha=0.9)
                tops += [y.min(), y.max()]
                bots += list(1e6 * (y - law_h) / law_h)
        tops += [law_h.min(), law_h.max()]
        for a_ in (ax, axr):
            a_.axvline(ref, color="0.4", ls="--", lw=0.8)
            a_.grid(alpha=0.3)
        sd_b = (e.get("sd") or {}).get("b")
        ax.set_title(f"{eye}: fx {a_h:.2f} px at focus {ref:.0f} (high state), slope {b:.5f}"
                     + (f" +- {sd_b:.5f}" if sd_b else "") + " px/count", loc="left", fontsize=10)
        ax.set_ylabel("focal length fx (px)")
        axr.axhline(0.0, color=ecol[eye], lw=1.2)
        axr.set_ylabel("bin - law (ppm)")
        axr.set_xlabel("focus motor count")
        sec = ax.secondary_yaxis("right", functions=(lambda v: v * ZCAM_PIXEL_MM, lambda v: v / ZCAM_PIXEL_MM))
        sec.set_ylabel("efl (mm)")
    if tops:                                                  # the same y range and ticks for both eyes
        pad = 0.04 * (max(tops) - min(tops) or 1.0)
        for ax in axs[0]:
            ax.set_ylim(min(tops) - pad, max(tops) + pad)
    if bots:
        q = np.percentile(np.abs(bots), 99) if len(bots) > 5 else max(abs(x) for x in bots)
        lim = max(200.0, 1.15 * float(q))
        for ax in axs[1]:
            ax.set_ylim(-lim, lim)
    h, l_ = axs[0][-1].get_legend_handles_labels()
    if h:
        axs[0][-1].legend(h, l_, fontsize=7, loc="upper left", ncol=2, framealpha=0.85)
    h0, l0 = axs[0][0].get_legend_handles_labels()
    if h0:
        axs[0][0].legend(h0, l0, fontsize=8, loc="upper left", framealpha=0.85)
    panel_letters(axs)
    zoom = int(res.get("zoom", 34))
    src = "bins: the diagnostic pass (law held, fx free)" if bins else "lines: each block's own (diagnostic pass)"
    fig.suptitle(f"Mastcam-Z {zoom} mm focal length against focus{(' ' + label) if label else ''} - "
                 f"{len(res.get('blocks') or blocks)} blocks; {src}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def plot_zcam_rig_trends(model: Dict[str, Any], path: PathLike, dpi: int = FIG_DPI, label: str = "",
                         point_labels: bool = True) -> Optional[Path]:
    """Mastcam-Z rig trends 2 x 4 (Christian, 20 Oct): the blocks' rig rotation residuals about the mean (yaw, pitch,
    roll, mdeg) and their refined baseline minus the CAHV baseline (mm), against sol (top) and the lens temperature
    (bottom), each with its least-squares line and slope +- se in the title.  ``model``: a calibration's
    ``M2020_ZCAM034_focus_model.json`` (``rig.residuals.per_block``, ``rig.per_block``).  None without rig rows.
    Panels lettered A, B ...; ``point_labels=False`` leaves out the block name at every point."""
    rig = model.get("rig") or (model.get("calibration") or {}).get("rig") or {}
    res = ((rig.get("residuals") or {}).get("per_block")) or []
    pb = {r["block"]: r for r in rig.get("per_block") or []}
    if not res:
        return None
    plt = _plt()
    cols = [("yaw", "yaw_mdeg", "mdeg"), ("pitch", "pitch_mdeg", "mdeg"), ("roll", "roll_mdeg", "mdeg"),
            ("baseline − CAHV", "dbaseline_mm", "mm")]
    fig, axs = plt.subplots(2, 4, figsize=(15.5, 7.6), squeeze=False)
    for j, (name, key, unit) in enumerate(cols):
        rows = [dict(r, **{k: v for k, v in pb.get(r["block"], {}).items() if k not in r}) for r in res]
        for i, (xk, xl, scale) in enumerate((("sol", "sol of the block", 1.0), ("T_optics_degC", "block lens temperature (°C)", 1.0))):
            ax = axs[i][j]
            pts = [(float(r[xk]), float(r[key]), r["block"]) for r in rows
                   if r.get(xk) is not None and r.get(key) is not None and np.isfinite(r[key])]
            if not pts:
                ax.set_axis_off()
                continue
            x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
            ax.plot(x, y, "o", color="C0" if i == 0 else "C3", ms=6, mec="white", mew=0.5)
            for xx, yy, n in (pts if point_labels else []):
                ax.annotate(n.replace("mars2020_", "")[:18], (xx, yy), fontsize=6, alpha=0.6, xytext=(3, 3),
                            textcoords="offset points")
            title = f"{name}"
            if len(x) >= 3 and np.ptp(x) > 0:
                A = np.c_[np.ones_like(x), x - x.mean()]
                b, *_ = np.linalg.lstsq(A, y, rcond=None)
                r_ = y - A @ b
                se = float(np.sqrt(r_ @ r_ / (len(y) - 2) / np.sum((x - x.mean()) ** 2)))
                xs = np.linspace(x.min(), x.max(), 2)
                ax.plot(xs, b[0] + b[1] * (xs - x.mean()), "-", color="0.3", lw=1)
                per = 1000.0 if xk == "sol" else 1.0
                z = b[1] / se if se > 0 else float("nan")
                title += (f": {b[1] * per:+.2f} ± {se * per:.2f} {unit}/{'1000 sol' if xk == 'sol' else '°C'} (z {z:+.1f})"
                          f"\nsd about the line {np.std(r_, ddof=2):.2f} {unit}")
            ax.set_title(title, fontsize=8.5)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(f"{name} ({unit})", fontsize=8)
            ax.grid(alpha=0.3)
    bc = rig.get("baseline_vs_cahv") or {}
    sup = (f"Mastcam-Z 34 mm rig trends{(' ' + label) if label else ''}: {len(res)} blocks; rotation residuals about the "
           f"mean rig; baseline {rig.get('baseline_m', float('nan')) * 1e3:.1f} mm")
    if bc.get("mean_mm") is not None:
        sup += f" ({bc['mean_mm']:+.2f} mm from the CAHV baselines, sd {bc.get('sd_mm') or float('nan'):.2f} mm)"
    panel_letters(axs)
    fig.suptitle(sup, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)
