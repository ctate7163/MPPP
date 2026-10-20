"""
The frozen camera-model forms: which COLMAP model each camera family uses, which parameters are held at zero,
and what varies with what.  :func:`validate` says whether a model file follows its form.

Navcam and Hazcam: ``THIN_PRISM_FISHEYE`` at the full detector frame (5120 x 3840) with sx1 = sy1 = 0 held;
fx, fy, cx, cy and k1 are linear in the camera temperature; fx, fy, cx, cy carry early-mission offsets before a sol;
the stereo rig rotation drifts linearly in sol.  Mastcam-Z: ``FULL_OPENCV`` at 1648 x 1200 with p1 = p2 = k3 = 0
held; fx, fy are lines in focus count per eye and backlash state, cx, cy per the ``pp`` terms, k1, k2 constant; no
temperature term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

NAVCAM_PARAMS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
FULL_OPENCV_PARAMS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
# parameter names per COLMAP model (the table every module of the pipeline reads)
PARAM_NAMES: Dict[str, Tuple[str, ...]] = {
    "FULL_OPENCV": FULL_OPENCV_PARAMS,
    "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
    "PINHOLE": ("fx", "fy", "cx", "cy"),
    "THIN_PRISM_FISHEYE": NAVCAM_PARAMS,
    "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"),
}
INTRINSIC_CORE = ("fx", "fy", "cx", "cy")          # focal length and principal point; everything else is distortion
# full detector frame (width, height) at downsample scale 1, by camera family
FULL_FRAME = {"N": (5120, 3840), "F": (5120, 3840), "R": (5120, 3840), "Z": (1648, 1200)}
ZCAM_ZOOMS = (34, 48, 63, 79, 110)
# the two Mastcam-Z focus (backlash) states: "backlash" is the high-focal-length state every image is in unless the
# low-backlash list or its own focal fit puts it in the "regular" (low-focal-length, label-like) state
ZCAM_STATES = ("backlash", "regular")
ZCAM_DEFAULT_STATE = "backlash"
TOL = 1e-12


@dataclass(frozen=True)
class Form:
    """One frozen form.  ``held_zero``: parameters held at 0 in the model and in every adjustment;
    ``temperature``: the parameters that are linear in temperature; ``sol``: what moves with sol."""
    name: str
    families: Tuple[str, ...]
    colmap_model: str
    param_names: Tuple[str, ...]
    frame: Tuple[int, int]
    held_zero: Tuple[str, ...]
    temperature: Tuple[str, ...]
    sol: Tuple[str, ...]
    focus: Tuple[str, ...]
    description: str

    @property
    def distortion_names(self) -> Tuple[str, ...]:
        return tuple(n for n in self.param_names if n not in INTRINSIC_CORE)


NAVCAM = Form(
    name="navcam", families=("N",), colmap_model="THIN_PRISM_FISHEYE", param_names=NAVCAM_PARAMS,
    frame=FULL_FRAME["N"], held_zero=("sx1", "sy1"),
    temperature=("fx", "fy", "cx", "cy", "k1"), sol=("early_mission", "rig_drift"), focus=(),
    description=("THIN_PRISM_FISHEYE at 5120 x 3840, sx1 = sy1 = 0; fx, fy in ppm/degC, cx, cy in px/degC and k1 "
                 "per degC about T0; fx, fy, cx, cy offsets before early_mission.before_sol; NL is the rig reference "
                 "(it keeps the label CAHVORE pointing and position); the rig rotation drifts linearly in sol"))

HAZCAM = Form(
    name="hazcam", families=("F", "R"), colmap_model="THIN_PRISM_FISHEYE", param_names=NAVCAM_PARAMS,
    frame=FULL_FRAME["F"], held_zero=("sx1", "sy1"),
    temperature=("fx", "fy", "cx", "cy", "k1"), sol=("early_mission", "rig_drift"), focus=(),
    description="the Navcam form for the front (F) and rear (R) Hazcams; shipped as start models (not calibrated)")

ZCAM = Form(
    name="zcam", families=("Z",), colmap_model="FULL_OPENCV", param_names=FULL_OPENCV_PARAMS,
    frame=FULL_FRAME["Z"], held_zero=("k3", "k4", "k5", "k6"),
    temperature=(), sol=(), focus=("fx", "fy", "cx", "cy"),
    description=("FULL_OPENCV at 1648 x 1200 with k3 = 0 held (zcam-v2, 0.91: p1, p2 refined per eye, constant over "
                 "focus); one calibration per zoom: fx, fy lines in focus count per eye and backlash state (a + b "
                 "(focus - reference_focus)), the principal point per the pp terms, k1, k2, p1, p2 constant; no "
                 "temperature term"))

# the line form of each Mastcam-Z focus parameter in a block: what of p = a + b (focus - ref) is fitted
#   "common"        b shared by every "common" parameter of the eye (fx and fy: one df/dfocus per eye), a fitted each;
#   "constant"      b = 0, a fitted (k1, k2: one value per eye and block);
#   "pinned"        a = the model's pp0 (ZR cx: the level is the rig's), b = 0;
#   "pinned_slope"  a = the model's pp0, b fitted (ZR cy: the right-minus-left boresight moves a little with focus).
# ZL's cx, cy are not on lines (held at the model's pp0: the reference eye).
ZCAM_LINE_FORM = {"ZL": {"fx": "common", "fy": "common", "k1": "constant", "k2": "constant", "p1": "constant", "p2": "constant"},
                  "ZR": {"fx": "common", "fy": "common", "cx": "pinned", "cy": "pinned_slope",
                         "k1": "constant", "k2": "constant", "p1": "constant", "p2": "constant"}}
ZCAM_LINE_PARAMS = {"ZL": ("fx", "fy", "k1", "k2", "p1", "p2"), "ZR": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")}
ZCAM_FORM_ID = "zcam-v2"                 # 0.91: p1, p2 refined (zcam-v1 held them at 0; its files still load)
ZCAM_LINE_FORMS = ("free", "common", "constant", "pinned", "pinned_slope")
ZCAM_MAX_P = 5e-3                        # |p1|, |p2| beyond this is not a lens (Rochette 20 Oct: 7.5e-4, 4.2e-4)

FORMS = {"navcam": NAVCAM, "hazcam": HAZCAM, "zcam": ZCAM}


def form_for(family: str) -> Form:
    """The form of a camera family letter (N, F, R, Z)."""
    fam = str(family)[:1].upper()
    for f in FORMS.values():
        if fam in f.families:
            return f
    raise ValueError(f"no camera form for family {family!r} (N, F, R or Z)")


def _param(model: Dict[str, Any], name: str) -> Optional[float]:
    names = list(model.get("param_names") or PARAM_NAMES.get(str(model.get("model")), ()))
    p = model.get("params") or []
    if name in names and names.index(name) < len(p):
        return float(p[names.index(name)])
    return None


def validate_camera(model: Dict[str, Any], form: Form = NAVCAM) -> List[str]:
    """Problems of a Navcam / Hazcam camera file against ``form`` (empty: it follows the form)."""
    out: List[str] = []
    if model.get("model") != form.colmap_model:
        out.append(f"model {model.get('model')!r}, the form is {form.colmap_model}")
    if (int(model.get("width") or 0), int(model.get("height") or 0)) != form.frame:
        out.append(f"frame {model.get('width')}x{model.get('height')}, the form is {form.frame[0]}x{form.frame[1]}")
    names = tuple(model.get("param_names") or ())
    if names and names != form.param_names:
        out.append(f"param_names {names} differ from the form's {form.param_names}")
    if len(model.get("params") or []) != len(form.param_names):
        out.append(f"{len(model.get('params') or [])} params, the form has {len(form.param_names)}")
    for n in form.held_zero:
        v = _param(model, n)
        if v is None:
            out.append(f"no {n} parameter")
        elif abs(v) > TOL:
            out.append(f"{n} = {v:+.3g}, the form holds it at 0")
    if not isinstance(model.get("thermal"), dict):
        out.append("no thermal entry (fx, fy ppm/degC, cx, cy px/degC, k1 per degC about T0_degC)")
    else:
        th = model["thermal"]
        for k in ("T0_degC", "fx_ppm_per_degC", "fy_ppm_per_degC"):
            if k not in th:
                out.append(f"thermal.{k} missing")
    rig_th = model.get("rig_thermal")
    if rig_th:
        out.append("a rig temperature term: the form carries the stereo temperature change on the principal points")
    return out


def validate_rig(rig: Dict[str, Any]) -> List[str]:
    """Problems of a stereo rig file (``R_sensor_from_ref``, translation, ``drift`` rates in sol; no temperature
    term)."""
    out: List[str] = []
    R = rig.get("R_sensor_from_ref")
    if not (isinstance(R, list) and len(R) == 3 and all(isinstance(r, list) and len(r) == 3 for r in R)):
        out.append("R_sensor_from_ref is not a 3 x 3 matrix")
    if rig.get("t_sensor_from_ref_m") is None and rig.get("t_sensor_from_ref") is None:
        out.append("no translation (t_sensor_from_ref_m)")
    th = rig.get("thermal") or {}
    if th.get("yaw_mdeg_per_degC") or th.get("pitch_mdeg_per_degC"):
        out.append("the rig has a temperature term: the form carries the stereo temperature change on the principal "
                   "points")
    dr = rig.get("drift")
    if dr is not None and not isinstance(dr, dict):
        out.append("drift is not a dict of <angle>_mdeg_per_sol rates")
    if isinstance(dr, dict) and "sol0" not in dr:
        out.append("drift.sol0 missing")
    return out


def validate_zcam(model: Dict[str, Any]) -> List[str]:
    """Problems of a Mastcam-Z focus-model file (one zoom, ``cameras`` ZL<zoom>, ZR<zoom>)."""
    out: List[str] = []
    cams = model.get("cameras")
    if not isinstance(cams, dict) or not cams:
        return ["no cameras"]
    for g, c in cams.items():
        if not str(g).startswith("Z") or len(str(g)) != 5:
            out.append(f"{g}: not a Mastcam-Z eye and zoom (ZL034 ...)")
        dist = c.get("distortion") or {}
        names, params = list(dist.get("names") or []), list(dist.get("params") or [])
        for n in ("k3", "k4", "k5", "k6"):
            if n in names and abs(float(params[names.index(n)])) > TOL:
                out.append(f"{g}: distortion {n} = {params[names.index(n)]:+.3g}, the form holds it at 0")
        for n in ("p1", "p2"):
            if n in names and abs(float(params[names.index(n)])) > ZCAM_MAX_P:
                out.append(f"{g}: distortion {n} = {params[names.index(n)]:+.3g}, beyond {ZCAM_MAX_P:g}")
        lines = c.get("lines") or {}
        for st in ZCAM_STATES:
            ln = lines.get(st) or {}
            for n in ("k3",):
                e = ln.get(n)
                if isinstance(e, dict) and (abs(float(e.get("a") or 0.0)) > TOL or abs(float(e.get("b") or 0.0)) > TOL):
                    out.append(f"{g}: {st} line for {n} is not 0, the form holds it at 0")
            for n in ("p1", "p2", "k1", "k2"):
                e = ln.get(n)
                if isinstance(e, dict) and abs(float(e.get("b") or 0.0)) > TOL:
                    out.append(f"{g}: {st} line for {n} has a focus slope, the form keeps it constant")
            for n in ("fx", "fy"):
                if lines and n not in ln:
                    out.append(f"{g}: no {st} line for {n}")
        th = c.get("thermal") or {}
        if float(th.get("f_px_per_degC") or 0.0):
            out.append(f"{g}: thermal.f_px_per_degC = {th['f_px_per_degC']}, the form has no temperature term")
        for st in ZCAM_STATES:
            for n, e in (lines.get(st) or {}).items():
                if isinstance(e, dict) and e.get("c_per_degC"):
                    out.append(f"{g}: {st} {n} has a temperature term, the form has none")
        if c.get("trend"):
            out.append(f"{g}: a sol trend, the form has none")
        if "pp0_px" not in c:
            out.append(f"{g}: no pp0_px (the principal point at the reference focus)")
    return out


def validate(model: Dict[str, Any], kind: Optional[str] = None) -> List[str]:
    """Problems of a model file (``kind``: "camera", "rig" or "zcam"; guessed from the keys when None)."""
    if kind is None:
        kind = "zcam" if "cameras" in model else ("rig" if "R_sensor_from_ref" in model else "camera")
    if kind == "zcam":
        return validate_zcam(model)
    if kind == "rig":
        return validate_rig(model)
    return validate_camera(model, HAZCAM if str(model.get("family", "N")) in ("F", "R") else NAVCAM)


def zcam_line_form(eye: str, params: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """{param: form} of a Mastcam-Z eye (``"ZL"`` / ``"ZR"`` or a group ``"ZR034"``) for ``params`` (default the
    eye's :data:`ZCAM_LINE_PARAMS`)."""
    e = str(eye)[:2]
    base = ZCAM_LINE_FORM.get(e, {})
    ps = tuple(params) if params else ZCAM_LINE_PARAMS.get(e, ("fx", "fy"))
    return {pn: str(base.get(pn, "free")) for pn in ps}
