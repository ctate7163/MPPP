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
PLOT_COLORS = {"NL": "#2a78d6", "NR": "#eb6834", "grey": "#8a8a8a"}     # left eye cool, right eye warm


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})
    return plt


def plot_camera_vs_temperature(nl: Dict[str, Any], nr: Dict[str, Any], path: PathLike,
                               blocks: Optional[Sequence[Dict[str, Any]]] = None,
                               T_range: Optional[Tuple[float, float]] = None, sol: Optional[float] = 1000.0,
                               title: Optional[str] = None) -> Path:
    """
    The Navcam camera models against camera temperature (``cmod vs temp``): fx, fy (ppm about their value at T0),
    cx, cy (px about T0) and k1 of the frozen form (:func:`camera_at`) for the left (cool) and right (warm) eye,
    with the per-block temperature-bin cameras of ``blocks`` (:func:`mppp.calib.fit.block_navcam_results`) as
    points sized by their images (a bin before the early-mission sol is shown without the early offsets, so that it
    compares with the line).  Rows share their y range between the eyes; T0 is the dashed line.  ``T_range``
    defaults to the bins' range (else -30..0 °C).
    """
    plt = _plt()
    T0 = thermal_terms(nl.get("thermal"))["T0"]
    if T_range is None:
        Tb = [c["T_mean"] for b in (blocks or []) for c in b["cameras"].values() if c["bin"] and c.get("T_mean") is not None]
        T_range = (min(-30.0, np.floor(min(Tb)) - 1), max(0.0, np.ceil(max(Tb)) + 1)) if Tb else (-30.0, 0.0)
    Ts = np.linspace(T_range[0], T_range[1], 121)
    rows = [("fx", "fx − fx(T0)  ppm"), ("fy", "fy − fy(T0)  ppm"), ("cx", "cx − cx(T0)  px"), ("cy", "cy − cy(T0)  px"),
            ("k1", "k1")]
    fig, axs = plt.subplots(len(rows), 2, figsize=(10, 11), sharex=True)
    for j, (eye, m) in enumerate((("NL", nl), ("NR", nr))):
        _n, p0, _ = camera_at(m, T_degC=T0, sol=sol)
        curves = np.array([camera_at(m, T_degC=T, sol=sol)[1][:5] for T in Ts])
        for i, (name, lab) in enumerate(rows):
            ax = axs[i, j]
            k = ["fx", "fy", "cx", "cy", "k1"].index(name)
            if name in ("fx", "fy"):
                y = 1e6 * (curves[:, k] / p0[k] - 1)
            elif name in ("cx", "cy"):
                y = curves[:, k] - p0[k]
            else:
                y = curves[:, k]
            ax.plot(Ts, y, color=PLOT_COLORS[eye], lw=1.8, label=f"{eye} joint model")
            if blocks:
                xs, ys, ss = [], [], []
                for b in blocks:
                    for key, c in b["cameras"].items():
                        if c["eye"] != eye or not c["bin"] or c.get("T_mean") is None:
                            continue
                        v = c["params"][k]
                        early = camera_at(m, T_degC=T0, sol=c.get("sol_median"))[1][k] - p0[k]   # early offsets out
                        v = v - early if name in ("cx", "cy") else v / (1 + early / p0[k]) if name in ("fx", "fy") else v
                        if name in ("fx", "fy"):
                            v = 1e6 * (v / p0[k] - 1)
                        elif name in ("cx", "cy"):
                            v = v - p0[k]
                        xs.append(c["T_mean"]); ys.append(v); ss.append(c["images"])
                if xs:
                    ax.scatter(xs, ys, s=8 + 0.6 * np.sqrt(ss) * 4, color=PLOT_COLORS[eye], alpha=0.35,
                               edgecolor="none", label="block temperature bins")
            ax.axvline(T0, color="k", ls="--", lw=0.8, alpha=0.6)
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{eye}  ({'left' if eye == 'NL' else 'right'} eye)")
                ax.legend(loc="upper left", fontsize=8, frameon=False)
    for i in range(len(rows)):                                   # the eyes share each row's y range
        lo = min(ax.get_ylim()[0] for ax in axs[i]); hi = max(ax.get_ylim()[1] for ax in axs[i])
        for ax in axs[i]:
            ax.set_ylim(lo, hi)
    for ax in axs[-1]:
        ax.set_xlabel("camera temperature  °C")
    fig.suptitle(title or f"Navcam camera model against temperature (T0 = {T0:g} °C)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_rig_vs_sol(rig: Dict[str, Any], path: PathLike, blocks: Optional[Sequence[Dict[str, Any]]] = None,
                    sol_range: Tuple[float, float] = (0.0, 2100.0), title: Optional[str] = None) -> Path:
    """
    The Navcam stereo rig against sol: pitch, yaw and roll of the right-from-left rotation relative to the rig
    file's rotation (its drift model, :func:`rig_at`), with the per-block rig epochs of ``blocks``
    (:func:`mppp.calib.fit.block_navcam_results`) as points; the pitch knot is the dashed line.
    """
    plt = _plt()
    from .joint import rig_angles
    R0 = np.asarray(rig["R_sensor_from_ref"], float)
    sols = np.linspace(sol_range[0], sol_range[1], 300)
    model = np.array([[rig_angles(rig_at(rig, sol=s)[0], R0)[f"{a}_mdeg"] for a in ("pitch", "yaw", "roll")] for s in sols])
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    knot = (rig.get("drift") or {}).get("knot_sol")
    for i, a in enumerate(("pitch", "yaw", "roll")):
        ax = axs[i]
        ax.plot(sols, model[:, i], color=PLOT_COLORS["NL"], lw=1.8, label="rig drift model")
        if blocks:
            xs, ys, ss = [], [], []
            for b in blocks:
                for key, r in (b.get("rigs") or {}).items():
                    if r.get("sol_median") is None:
                        continue
                    xs.append(r["sol_median"]); ys.append(rig_angles(np.asarray(r["R"]), R0)[f"{a}_mdeg"]); ss.append(r["images"])
            if xs:
                ax.scatter(xs, ys, s=8 + 2.4 * np.sqrt(ss), color=PLOT_COLORS["NR"], alpha=0.6, edgecolor="none",
                           label="block rig epochs")
        if knot:
            ax.axvline(float(knot), color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylabel(f"{a}  mdeg")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)
    axs[-1].set_xlabel("sol")
    fig.suptitle(title or "Navcam stereo rig rotation against sol (relative to the promoted rig)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)
