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
def focus_state(listed: bool) -> str:
    """The backlash state of a Mastcam-Z image: ``"regular"`` (the low-focal-length state) when the low-backlash
    list, the found list or its own focal fit says so, else the default ``"backlash"`` (the high-focal-length
    state)."""
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
    return forms.ZCAM.colmap_model, p, {"source": f"{group} at focus {float(focus):g} ({src})", "lines": sorted(vals)}


# -------------------------------------------------------------------------------------------- label fallback
def label_camera(meta: Dict[str, Any]) -> Tuple[str, List[float], str]:
    """
    The start camera of an image from its PDS label where no promoted model applies: a Mastcam-Z (or any
    CAHVOR) label gives FULL_OPENCV from the pinhole intrinsics with the label's k1, k2 and p1 = p2 = k3 = 0; a
    Navcam / Hazcam CAHVORE fisheye label (type 2) gives THIN_PRISM_FISHEYE exactly - f (1 + r0), k1 = r1 / (1 + r0),
    k2 = r2 / (1 + r0), the other terms 0.  ``meta``: an ``MPPPImage.meta`` dict (``intrinsics_label`` /
    ``intrinsics``, ``camera_model_label``) - the result is at the image's padded resolution.  Returns (model
    name, params, source).
    """
    from ..colmap import colmap_camera_params
    intr = meta.get("intrinsics_label") or meta["intrinsics"]
    fam = str((meta.get("filename") or {}).get("family") or meta.get("source_product", "")[:1]).upper()
    cm = meta.get("camera_model_label") or {}
    if fam in ("N", "F", "R") and str(cm.get("MODEL_TYPE", "")).upper() == "CAHVORE":
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
EYE_COLORS = {"NL": "C0", "NR": "C3"}        # left eye cool, right eye warm


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
                    T0: float = NAVCAM_T0_DEGC, model_label: str = "drift model") -> Path:
    """
    Rig trends 2 x 2 (fig_rig.py, 12.5 x 8.2 in, dpi 130): pitch (UL), yaw with its temperature slope removed
    to T0 (UR) and roll (LL) of every block rig epoch against its median sol, each relative to the joint rig at
    ``sol_ref`` (= 0), with the ``drift`` model line, dashed lines at the knot, the residual-trend statistics in the
    titles and a secondary IFOV axis; LR: the stereo pointing NR cx − NL cx against epoch temperature with the
    model slope (dashed), the slope in use for the block runs (dotted) and the slope with the yaw residual folded
    in (solid, ± se band).  ``rig_residuals`` is the study's ``rig_residuals.json`` (rig rotation free, cameras
    held), ``consensus`` its ``consensus.json``.
    """
    plt = _plt()
    rr, cons = rig_residuals, consensus
    B = rr["blocks"]; names = [b["block"] for b in B]
    sol = np.array([b["sol_median"] for b in B], float); T = np.array([b["T_median_degC"] for b in B], float)
    ppl = cons["thermal"]["pp_slopes"]
    CX_T = ppl["NR"][0] - ppl["NL"][0]
    CX0 = cons["NR"]["cx"] - cons["NL"]["cx"]

    def annotate(ax, x, y, nm):
        for xx, yy, n in zip(x, y, nm):
            ax.annotate(n.replace(" [sols ", " [")[:22], (xx, yy), fontsize=5.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
    ss = np.linspace(0, 2050, 400)
    panels = {"pitch": axs[0][0], "yaw": axs[0][1], "roll": axs[1][0]}
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
        ax.set_xlabel("median sol of the rig epoch", fontsize=8)
        ax.set_ylabel(f"{ang} relative to the joint rig at sol {sol_ref:.0f} (mdeg)", fontsize=8)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    lo = min(a.get_ylim()[0] for a in panels.values()); hi = max(a.get_ylim()[1] for a in panels.values())
    for a in panels.values():
        a.set_ylim(lo, hi); a.set_xlim(0, 2050)
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
    ax.secondary_yaxis("right", functions=(lambda p: (p - CX0) * IFOV_MDEG, lambda m: CX0 + m / IFOV_MDEG)).set_ylabel(f"yaw equivalent about cx at {T0:.0f} °C (mdeg)", fontsize=8)
    nb = len({b.get("site") or b["block"] for b in B})
    fig.suptitle(f"Navcam rig trends{(' ' + label) if label else ''} ({nb} blocks, {len(B)} rig epochs; drift referred to sol {sol_ref:.0f}; "
                 f"right axes in Navcam IFOV, 0.33 mrad = {IFOV_MDEG} mdeg)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_intrinsics_vs_temperature(consensus: Dict[str, Any], inuse: Dict[str, Dict[str, Any]], bins: Sequence[Dict[str, Any]],
                                   rig_residuals: Dict[str, Any], path: PathLike, early_sol: float = 380.0,
                                   label: str = "", T0: float = NAVCAM_T0_DEGC, model_label: str = "new model") -> Path:
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
            ax.plot(tt, f2[p] * 1e-6 * ppm2 * (tt - T0), ls, color=col, lw=1.4, label=f"{p} {model_label}: {ppm2:+.1f} ppm/°C ({f2[p] * 1e-6 * ppm2:+.4f} px/°C)")
            if k == 0:
                ax.plot(tt, p_iu[k] * 1e-6 * ppm_iu * (tt - T0), ":", color="0.4", lw=1.1, label=f"fx in use for the block runs: {ppm_iu:+.1f} ppm/°C")
            for early, mfc in ((False, None), (True, "white")):
                rs = [r for r in rows if (r["block"] in early_blk) == early]
                if not rs:
                    continue
                Tb = np.array([r["T"] for r in rs]); nb = np.array([r["n"] for r in rs])
                v = np.array([r[p] for r in rs]) - p_iu[k]
                ax.scatter(Tb, v, s=6 + 1.5 * np.sqrt(nb * 10), marker=mk, facecolors=mfc or col, edgecolors=col, alpha=alpha, zorder=3,
                           label=(f"{p} per temperature bin of the block runs (size ~ images)" if not early else
                                  f"{p}, blocks before sol {early_sol:.0f} (early-mission offset not removed)") if p == "fx" else None)
        ax.set_title(f"{eye}: focal length", fontsize=10)
        ax.set_ylabel(f"Δf from {T0:.0f} °C (full-frame px)", fontsize=8)
        ax.secondary_yaxis("right", functions=(lambda d, f=f2["fx"]: 1e6 * d / f, lambda q, f=f2["fx"]: q * f / 1e6)).set_ylabel("ppm of f", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")
        ax = axs[1][j]
        cx_iu = th_iu["cx_px"]
        if eye == "NL":
            ax.plot(tt, cx_iu * (tt - T0), ":", color="0.4", lw=1.2, label=f"cx in use for the block runs: {cx_iu:+.4f} px/°C")
            ax.plot(tt, pp["NL"][0] * (tt - T0), "-", color=col, lw=1.3, label=f"cx {model_label}: {pp['NL'][0]:+.4f} px/°C; cy: 0")
        else:
            ax.plot(tt, cx_iu * (tt - T0), ":", color="0.4", lw=1.2, label=f"cx in use for the block runs: {cx_iu:+.4f} px/°C")
            ax.plot(tt, pp["NR"][0] * (tt - T0), "--", color=col, lw=1.2, label=f"cx {model_label}: {pp['NR'][0]:+.4f} px/°C")
            ax.fill_between(tt, (CX_ALL - CX_ALL_SE) * (tt - T0), (CX_ALL + CX_ALL_SE) * (tt - T0), color=col, alpha=0.12, lw=0)
            ax.plot(tt, CX_ALL * (tt - T0), "-", color=col, lw=1.6, label=f"stereo cx (NR − NL) with the yaw residual folded in: {CX_ALL:+.4f} ± {CX_ALL_SE:.4f} px/°C")
            ax.plot(tt, 0 * tt, "-.", color=col, lw=0.8, label="cy: 0")
            y = CX2 * (Tblk - T0) + yaw / IFOV_MDEG
            ax.plot(Tblk, y, "o", color=col, ms=5, mec="white", mew=0.5, zorder=3, label="rig epoch stereo pointing as NR cx − NL cx (model + yaw residual)")
        ax.set_title(f"{eye}: principal point", fontsize=10)
        ax.set_ylabel(f"Δ principal point from {T0:.0f} °C (full-frame px)", fontsize=8)
        ax.set_xlabel("Navcam camera temperature (°C)", fontsize=9)
        ax.secondary_yaxis("right", functions=(lambda d: d * IFOV_MDEG, lambda m: m / IFOV_MDEG)).set_ylabel(f"pointing equivalent (mdeg; 1 px = {IFOV_MDEG} mdeg)", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right" if eye == "NR" else "upper left")
    Tb_all = np.array([r["T"] for r in bins]) if bins else np.array([T0])
    for r in range(2):
        lo = min(axs[r][0].get_ylim()[0], axs[r][1].get_ylim()[0]); hi = max(axs[r][0].get_ylim()[1], axs[r][1].get_ylim()[1])
        for a in axs[r]:
            a.set_ylim(lo, hi)
        axs[r][1].set_yticks(axs[r][0].get_yticks()); axs[r][1].set_ylim(lo, hi)
    for a in axs.flat:
        a.axvline(T0, color="0.4", ls="--", lw=0.8); a.axhline(0, color="0.6", lw=0.6)
        a.axvspan(Tb_all.min(), Tb_all.max(), color="0.85", alpha=0.35, lw=0, zorder=0); a.grid(alpha=0.3); a.set_xlim(-60, 0)
    nb = len(sols)
    fig.suptitle(f"Navcam intrinsic temperature dependence{(' ' + label) if label else ''} ({nb} blocks; T0 = {T0:.0f} °C; grey band: temperatures of the bins)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=140)
    plt.close(fig)
    return Path(path)
