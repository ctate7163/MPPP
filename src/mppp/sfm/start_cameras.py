"""
The start cameras and rig of a block from the promoted models (:mod:`mppp.calib`): the Navcam eye camera at the
block's median temperature and sol, the stereo rig at the block's median sol, the Hazcam eye camera from the
images' labels (no Hazcam model is promoted yet: :func:`hazcam_start_camera`), and the Mastcam-Z eye camera from the
zoom's model (or the labels).  ``SfmProject.create`` calls these; the per-focus-bin cameras are made in
:mod:`mppp.sfm.project`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import calib
from ..calib import forms

log = logging.getLogger("mppp.sfm")
PathLike = Union[str, Path]
FULL_OPENCV_NAMES = forms.FULL_OPENCV_PARAMS


def navcam_start_camera(instrument: str, images: Sequence[Dict[str, Any]], folder: Optional[PathLike] = None,
                        zero_terms: Sequence[str] = ()) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    The start camera of a Navcam / Hazcam eye (``instrument`` NL, NR, FL ...) for a block: the promoted model
    (:func:`mppp.calib.load_camera`) constrained to the block's median camera temperature and the share of its
    images before the early-mission sol (:func:`mppp.calib.camera_at`), as a ``project.cameras`` dict.  Returns
    (camera, info) with ``info`` what ``settings["navcam_cameras"][instrument]`` records (the file, its
    verification, the thermal and early-mission terms as applied).
    """
    fam, eye = str(instrument)[0], str(instrument)[1]
    model = calib.load_camera(fam, eye, folder)
    rows = [r for r in images if r.get("instrument") == instrument]
    Ts = [float(r["camera_temperature_degC"]) for r in rows if r.get("camera_temperature_degC") is not None]
    sols = [float(r["sol"]) for r in rows if r.get("sol") is not None]
    T = float(np.median(Ts)) if Ts else None
    share = calib.early_share(model, sols) if sols else None
    name, params, applied = calib.camera_at(model, T, early=share)
    names = forms.PARAM_NAMES.get(name, FULL_OPENCV_NAMES)
    zeroed = []
    for n in ("p1", "p2"):
        if n in zero_terms and n in names:
            params[names.index(n)] = 0.0
            zeroed.append(n)
    src = Path(model["file"]).name if model.get("file") else model.get("source", "start model")
    if zeroed:
        src += f" ({', '.join(zeroed)} set to 0)"
    th = applied.get("thermal")
    if th:
        t = calib.thermal_terms(model.get("thermal"))
        src += (f"; fx x {th['sx']:.6f}, fy x {th['sy']:.6f} for the median camera temperature {th['T_degC']:.1f} degC "
                f"(consensus at {t['T0']:.1f} degC, {t['fx_ppm']:+.1f} / {t['fy_ppm']:+.1f} ppm/degC)"
                + (f"; cx, cy {th['dcx_px']:+.3f}, {th['dcy_px']:+.3f} px (principal point thermal)"
                   if th["dcx_px"] or th["dcy_px"] else "")
                + (f"; k1 {th['dk1']:+.2e} (k1 thermal)" if th["dk1"] else ""))
    em = applied.get("early_mission")
    if em:
        src += ("; early-mission offsets " + ", ".join(f"{k} {v:+.3f}" for k, v in em["applied_px"].items())
                + f" px ({100 * em['share']:.0f} % of the images before sol {em['before_sol']:g})")
    cam = {"model": name, "width": int(model["width"]), "height": int(model["height"]), "params": params,
           "free_params": [], "distortion": model.get("distortion"), "source": src}
    info: Dict[str, Any] = {"file": model.get("file"), "verification": model.get("verification"),
                            "form": model.get("form"), "calibrated": model.get("file") is not None}
    if th:
        info["thermal"] = {k: v for k, v in th.items() if k not in ("sx", "sy", "dcx_px", "dcy_px", "dk1", "T_degC")}
    if em:
        info["early_mission"] = em
    if not model.get("file"):
        log.warning("%s: no calibrated model, the %s start model is used (f = %g px, not calibrated)", instrument,
                    fam, params[0])
    return cam, info


# what a block refines of a Hazcam eye (0.93, setting ``hazcam_intrinsics``); every other parameter is held at the
# start camera (the label fit).  "hold": the camera is held entirely (a hold_cameras key in the adjustment).
HAZCAM_INTRINSICS = {"hold": (), "focal": ("fx", "fy"), "radial": ("fx", "fy", "k1", "k2"),
                     "all": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2")}
HAZCAM_INTRINSICS_DEFAULT = "radial"


def hazcam_start_camera(instrument: str, images: Sequence[Dict[str, Any]], metas: Sequence[Dict[str, Any]],
                        folder: Optional[PathLike] = None, intrinsics: str = HAZCAM_INTRINSICS_DEFAULT
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    The start camera of a Hazcam eye (``instrument`` FL, FR, RL, RR) for a block: a promoted model when there is one
    (the Navcam form, :func:`navcam_start_camera`), else the median of the fits to the block's label CAHVOREs
    (:func:`mppp.calib.labelfit.block_camera`; ``metas`` are the manifest entries of that eye).  ``intrinsics``
    (:data:`HAZCAM_INTRINSICS`) says which parameters the block refines: they are the camera's free parameters,
    every other one goes into ``fixed_params``.  Returns (camera, info) as :func:`navcam_start_camera`.
    """
    from ..calib import labelfit
    if intrinsics not in HAZCAM_INTRINSICS:
        raise ValueError(f"hazcam_intrinsics must be one of {tuple(HAZCAM_INTRINSICS)}")
    fam, eye = str(instrument)[0], str(instrument)[1]
    if fam not in ("F", "R"):
        raise ValueError(f"{instrument!r} is not a Hazcam eye (FL, FR, RL, RR)")
    if calib.models.has_camera_file(fam, eye, folder):
        cam, info = navcam_start_camera(instrument, images, folder)
    else:
        cam, info = labelfit.block_camera(instrument, metas)
    names = list(forms.PARAM_NAMES[cam["model"]])
    free = [n for n in HAZCAM_INTRINSICS[intrinsics] if n in names]
    params = list(map(float, cam["params"]))
    for n in forms.HAZCAM.held_zero:
        params[names.index(n)] = 0.0
    cam = dict(cam, params=params, fixed_params=[n for n in names if n not in free], free_params=[],
               hazcam_intrinsics=intrinsics,
               source=f"{cam.get('source', '')}; refined per block: {', '.join(free) if free else 'nothing (held)'}")
    info = dict(info, intrinsics=intrinsics, refined=free)
    return cam, info


def navcam_start_rig(rig: Dict[str, Any], images: Sequence[Dict[str, Any]], family: str = "N",
                     folder: Optional[PathLike] = None) -> Optional[Dict[str, Any]]:
    """
    Start the stereo rig of ``family`` (a ``_rig_from_pairs`` entry, changed in place) from the promoted rig file
    (:func:`mppp.calib.load_rig`): the rotation at the block's median sol (:func:`mppp.calib.rig_at`), the fitted
    translation when the file has one (else the block's CAHV baseline).  Returns the info for
    ``settings["navcam_cameras"]["rig"]`` (None when the family has no rig file: the CAHV rig stays).
    """
    shipped = calib.load_rig(family, folder)
    if shipped is None:
        return None
    rows = [r for r in images if str(r.get("instrument", "")).startswith(family)]
    sols = [int(r["sol"]) for r in rows if r.get("sol") is not None]
    Ts = [float(r["camera_temperature_degC"]) for r in rows if r.get("camera_temperature_degC") is not None]
    sol = float(np.median(sols)) if sols else None
    R, t, applied = calib.rig_at(shipped, sol)
    rig["R_sensor_from_ref_cahv"] = rig["R_sensor_from_ref"]
    rig["R_sensor_from_ref"] = R.tolist()
    info: Dict[str, Any] = {"file": shipped["file"], "T_median_degC": float(np.median(Ts)) if Ts else None, **applied}
    if t is not None:
        rig["t_sensor_from_ref_cahv"] = rig["t_sensor_from_ref"]
        rig["t_sensor_from_ref"] = [float(x) for x in t]
        rig["baseline_m"] = float(np.linalg.norm(t))
    rig["rotation_source"] = (f"{Path(shipped['file']).name} (refined consensus; translation "
                              f"{'fitted on the large blocks' if t is not None else 'from CAHV'}) from "
                              f"{Path(shipped['file']).parent}")
    return info


def zcam_eye_camera(group: str, label_params: List[tuple], size: Tuple[int, int], zero_terms: Sequence[str],
                    gm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    The start camera of a Mastcam-Z eye and zoom (``group`` ZL034 ...): FULL_OPENCV at the full frame from the median
    of the images' label CAHVOR intrinsics (``label_params``: (model, params) per image, full-frame pixels), with the
    zoom model's distortion (k1, k2) and absolute principal point ``pp0_px`` (at the reference focus) when the zoom
    has one (``gm``: the eye's entry of :func:`mppp.calib.load_zcam`).  p1, p2 and k3 follow ``zero_terms``; the
    form holds them at 0 in any case.
    """
    full = []
    for model, p in label_params:
        q = np.zeros(12)
        q[:4] = p[:4]
        if model in ("OPENCV", "FULL_OPENCV"):
            q[4:8] = p[4:8]
        if model == "FULL_OPENCV":
            q[8:] = p[8:12]
        full.append(q)
    P = np.array(full)
    med = np.median(P, axis=0)
    for n in ("p1", "p2", "k3"):
        if n in zero_terms:
            med[FULL_OPENCV_NAMES.index(n)] = 0.0
    cam = {"model": "FULL_OPENCV", "width": int(size[0]), "height": int(size[1]), "params": med.tolist(),
           "source": f"median of {len(P)} label CAHVOR models ({group}; {', '.join(zero_terms)} set to 0)",
           "label_f_spread_px": float(np.ptp(P[:, 0])) if len(P) else 0.0}
    if not gm:
        return cam
    notes = []
    dist = gm.get("distortion")
    if dist and dist.get("names") and dist.get("params"):
        for n, v in zip(dist["names"], dist["params"]):
            if n in FULL_OPENCV_NAMES[4:]:
                cam["params"][FULL_OPENCV_NAMES.index(n)] = 0.0 if n in zero_terms else float(v)
        notes.append("distortion of the focus model")
    if gm.get("pp0_px"):
        cam["params"][2], cam["params"][3] = map(float, gm["pp0_px"])
        cam["pp_at_focus"] = float(gm["reference_focus"])
        notes.append(f"cx, cy of the focus model at focus {float(gm['reference_focus']):.0f}")
    if notes:
        cam["source"] = f"{cam['source']}; " + "; ".join(notes)
        cam["model_distortion"] = bool(dist)
    return cam
