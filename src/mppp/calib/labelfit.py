"""
The engineering-camera (Navcam, Hazcam) start camera from a PDS label: the COLMAP ``THIN_PRISM_FISHEYE`` camera that
reproduces the label's CAHVORE model over the whole detector, by least squares.

Why a fit and not the closed form.  A CAHVORE of type 2 (fisheye, linearity 0) with E = 0 and O along A is exactly
the fisheye f (1 + r0), k1 = r1 / (1 + r0), k2 = r2 / (1 + r0) (:func:`mppp.calib.label_camera`).  The flight models
are not quite that: O is tilted from A by 0.05-0.25 deg, and the rear Hazcams carry entrance-pupil terms E of 3-9 mm
(the front Hazcams and the Navcams about 1e-7 m).  The closed form then misses by 1.4-2.5 px rms (front Hazcam, full
frame) and 6-7 px rms (rear); the fit below by 0.13-0.19 px (front) and 2-3.7 px over 2-20 m (rear, where a central
camera cannot follow the moving pupil; checked on the sol 461 labels, 22 Oct 2026).

The fit works in the label model's own camera frame (:meth:`mppp.cmod.CameraModel.camera_frame`: z = A, y from V,
the frame the pose prior is built in), so no rotation is absorbed and the prior attitude stays the label's.  For a
central model (max |E| below :data:`CENTRAL_E_M`) the points lie at one range; for a non-central one at the ranges
of :data:`NONCENTRAL_DEPTHS_M` (the Hazcam terrain), so the fit is the best central camera for that terrain.  The
parameters fitted are fx, fy, cx, cy, k1-k4, p1, p2; sx1 = sy1 = 0 (the form).

The Hazcams have no promoted model (``M2020_FL.json`` ...), so this is their start camera: per image in the
processing (:mod:`mppp.image`) and per eye and block in the COLMAP project (the median over the block's images,
:func:`block_camera`).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import forms

log = logging.getLogger("mppp.calib")

CENTRAL_E_M = 1e-4                       # a CAHVORE with every |E| below this is central (Navcam, front Hazcam)
CENTRAL_DEPTH_M = 10.0
NONCENTRAL_DEPTHS_M = (2.0, 5.0, 20.0)   # rear Hazcam (E 3-9 mm): the terrain ranges the central fit serves
FIT_STEP_PX = 80.0                       # grid of full-frame pixels the fit uses (64 x 48 = 3185 points)
FIT_PARAMS = ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2")


def project_thin_prism_fisheye(params: Sequence[float], X: np.ndarray) -> np.ndarray:
    """COLMAP's ``THIN_PRISM_FISHEYE`` projection of camera-frame points (N, 3) -> pixels (N, 2) (corner origin), in
    numpy (equal to pycolmap's to 1e-12 px; points behind the camera give NaN)."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = (float(v) for v in params)
    X = np.asarray(X, float).reshape(-1, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
        r = np.hypot(u, v)
        s = np.where(r > 1e-12, np.arctan(r) / np.where(r > 1e-12, r, 1.0), 1.0)
    u, v = u * s, v * s
    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2
    rad = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4
    du = u * rad + 2 * p1 * uv + p2 * (r2 + 2 * u2) + sx1 * r2
    dv = v * rad + 2 * p2 * uv + p1 * (r2 + 2 * v2) + sy1 * r2
    out = np.c_[fx * (u + du) + cx, fy * (v + dv) + cy]
    out[~(X[:, 2] > 0)] = np.nan
    return out


def _closed_form(cm) -> np.ndarray:
    """The type-2 closed form (E and the O tilt ignored) as the fit's start: fx fy cx cy k1 k2 k3 k4 p1 p2."""
    _, lin = cm.decompose()
    r = cm.R if cm.R is not None else np.zeros(3)
    g = 1.0 + float(r[0])
    return np.array([lin["hs"] * g, lin["vs"] * g, lin["hc"] + 0.5, lin["vc"] + 0.5, float(r[1]) / g, float(r[2]) / g,
                     0.0, 0.0, 0.0, 0.0])


def _colmap_order(x: np.ndarray) -> List[float]:
    """fx fy cx cy k1 k2 k3 k4 p1 p2 -> the THIN_PRISM_FISHEYE parameter order (sx1 = sy1 = 0)."""
    return [float(v) for v in (x[0], x[1], x[2], x[3], x[4], x[5], x[8], x[9], x[6], x[7], 0.0, 0.0)]


def is_central(cm) -> bool:
    return cm.E is None or float(np.max(np.abs(cm.E))) < CENTRAL_E_M


def fisheye_from_cahvore(cm, width: int = 5120, height: int = 3840, depths: Optional[Sequence[float]] = None,
                         step: float = FIT_STEP_PX) -> Tuple[List[float], Dict[str, Any]]:
    """
    The ``THIN_PRISM_FISHEYE`` parameters (12, COLMAP order, corner pixel origin) that reproduce the CAHV-family model
    ``cm`` (:class:`mppp.cmod.CameraModel` in full-frame pixels, centre origin; any frame) over a ``width`` x
    ``height`` frame.  The model is taken in its own camera frame; points along the rays of a pixel grid (every
    ``step`` px) at the ranges ``depths`` (default: :data:`CENTRAL_DEPTH_M` for a central model,
    :data:`NONCENTRAL_DEPTHS_M` otherwise) are projected by the model, and the fisheye is fitted to them.  Returns
    (params, report) with the closed-form start's and the fit's rms / p95 / max differences per range (px).
    """
    from scipy.optimize import least_squares
    from ..cmod import _rays_of, pixel_grid
    c, _ = cm.camera_frame()
    if depths is None:
        depths = (CENTRAL_DEPTH_M,) if is_central(c) else NONCENTRAL_DEPTHS_M
    uv = pixel_grid(int(width), int(height), float(step))
    with np.errstate(all="ignore"):
        d = _rays_of(c, uv - 0.5)
    ok = np.all(np.isfinite(d), axis=1)
    d = d[ok]
    P, U, which = [], [], []
    for k, D in enumerate(depths):
        X = d * float(D)
        with np.errstate(all="ignore"):
            u = c.project(X) + 0.5
        m = np.all(np.isfinite(u), axis=1)
        P.append(X[m])
        U.append(u[m])
        which.append(np.full(int(m.sum()), k))
    P, U, which = np.vstack(P), np.vstack(U), np.concatenate(which)
    if len(P) < 50:
        raise ValueError(f"the camera model images too few grid points ({len(P)}) to fit a fisheye")
    x0 = _closed_form(c)

    def res(x):
        r = project_thin_prism_fisheye(_colmap_order(x), P) - U
        return np.nan_to_num(r, nan=1e3).ravel()

    sol = least_squares(res, x0, x_scale="jac", method="lm", max_nfev=2000)

    def stats(x) -> List[Dict[str, Any]]:
        e = np.linalg.norm(res(x).reshape(-1, 2), axis=1)
        out = []
        for k, D in enumerate(depths):
            ek = e[which == k]
            out.append({"range_m": float(D), "rms_px": float(np.sqrt(np.mean(ek ** 2))),
                        "p95_px": float(np.percentile(ek, 95)), "max_px": float(np.max(ek)), "n": int(len(ek))})
        return out

    fit, start = stats(sol.x), stats(x0)
    rep = {"rms_px": float(np.sqrt(np.mean([s["rms_px"] ** 2 for s in fit]))),
           "max_px": max(s["max_px"] for s in fit), "by_range": fit,
           "closed_form_rms_px": float(np.sqrt(np.mean([s["rms_px"] ** 2 for s in start]))),
           "closed_form_max_px": max(s["max_px"] for s in start),
           "central": bool(is_central(c)), "depths_m": [float(x) for x in depths],
           "E_max_m": float(np.max(np.abs(c.E))) if c.E is not None else 0.0,
           "o_tilt_deg": float(np.degrees(np.arccos(np.clip(c.O @ c.A, -1, 1)))) if c.O is not None else 0.0,
           "frame": [int(width), int(height)], "success": bool(sol.success)}
    return _colmap_order(sol.x), rep


def full_frame_model(label_model: Dict[str, Any], downsample_scale: float, first_sample: float = 0.0,
                     first_line: float = 0.0, family: str = "N"):
    """The label model of a product (``MODEL_COMPONENT_*`` dict, product pixels, centre origin) in full-frame
    detector pixels: ``first_sample`` / ``first_line`` are the product pixels between the detector corner and the
    product's first pixel (the padding an MPPP image records), ``downsample_scale`` the product's scale."""
    from ..cmod import CameraModel
    W, H = forms.FULL_FRAME[str(family)[:1].upper()]
    cm = CameraModel.from_label(label_model)
    return cm.rescaled(float(downsample_scale), float(first_sample), float(first_line), W, H)


def _model_key(label_model: Dict[str, Any]) -> Tuple:
    return tuple(tuple(np.round(np.ravel(np.asarray(label_model.get(f"MODEL_COMPONENT_{i}", [np.nan]), float)), 12))
                 for i in range(1, 10)) + (str(label_model.get("MODEL_TYPE")),)


@functools.lru_cache(maxsize=512)
def _fit_cached(key: Tuple, scale: float, dx: float, dy: float, family: str) -> Tuple[Tuple[float, ...], str]:
    import json
    lm = {"MODEL_TYPE": key[-1]}
    for i, comp in enumerate(key[:-1], start=1):
        if not np.all(np.isnan(comp)):
            lm[f"MODEL_COMPONENT_{i}"] = list(comp) if len(comp) > 1 else float(comp[0])
    cm = full_frame_model(lm, scale, dx, dy, family)
    W, H = forms.FULL_FRAME[family]
    params, rep = fisheye_from_cahvore(cm, W, H)
    return tuple(params), json.dumps(rep)


def label_fisheye(label_model: Dict[str, Any], downsample_scale: float, first_sample: float = 0.0,
                  first_line: float = 0.0, family: str = "F") -> Tuple[List[float], Dict[str, Any]]:
    """The full-frame ``THIN_PRISM_FISHEYE`` start camera of one Navcam / Hazcam product from its label model (see
    :func:`full_frame_model`, :func:`fisheye_from_cahvore`); cached by model and placement (a block's images share a
    few models: the label is interpolated to temperature, or not at all for the rear Hazcams)."""
    import json
    fam = str(family)[:1].upper()
    if str(label_model.get("MODEL_TYPE", "")).upper() != "CAHVORE":
        raise ValueError(f"label model {label_model.get('MODEL_TYPE')!r}: the Navcam / Hazcam fit needs a CAHVORE")
    p, rep = _fit_cached(_model_key(label_model), float(downsample_scale), float(first_sample), float(first_line), fam)
    return list(p), json.loads(rep)


def meta_fisheye(meta: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
    """:func:`label_fisheye` of a manifest entry (``MPPPImage.meta``: ``camera_model_label``, ``filename``,
    ``padding``)."""
    fn = meta.get("filename") or {}
    pad = meta.get("padding") or {}
    return label_fisheye(meta["camera_model_label"], float(fn.get("downsample_scale") or 1.0),
                         float(pad.get("left") or 0), float(pad.get("top") or 0),
                         str(fn.get("family") or meta["source_product"][:1]))


def block_camera(instrument: str, metas: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    The start camera of a Hazcam (or Navcam) eye in a block from the labels: the median of the per-image label fits
    (:func:`meta_fisheye`) over the block's images of ``instrument`` (FL, FR, RL, RR; full frame).  Returns (camera,
    info) in the form :func:`mppp.sfm.start_cameras.navcam_start_camera` returns them.
    """
    fam = str(instrument)[:1].upper()
    W, H = forms.FULL_FRAME[fam]
    fits, reps = [], []
    for m in metas:
        try:
            p, rep = meta_fisheye(m)
        except (KeyError, ValueError, TypeError) as e:
            log.debug("%s: no label fit (%s)", m.get("source_product"), e)
            continue
        fits.append(p)
        reps.append(rep)
    if not fits:
        raise ValueError(f"{instrument}: no image with a CAHVORE label model")
    P = np.asarray(fits, float)
    med = np.median(P, axis=0)
    med[10:] = 0.0
    spread = {n: float(np.ptp(P[:, i])) for i, n in enumerate(forms.NAVCAM_PARAMS[:4])}
    rms = float(np.median([r["rms_px"] for r in reps]))
    src = (f"label CAHVORE fit (median of {len(fits)} images; fit rms {rms:.2f} px full frame"
           + ("" if reps[0]["central"] else f" over {', '.join(f'{d:g}' for d in reps[0]['depths_m'])} m, "
              f"non-central E up to {1e3 * max(r['E_max_m'] for r in reps):.1f} mm") + ")")
    cam = {"model": "THIN_PRISM_FISHEYE", "width": int(W), "height": int(H), "params": [float(v) for v in med],
           "free_params": [], "distortion": "fisheye (theta polynomial k1-k4) + tangential p1, p2; sx1 = sy1 = 0",
           "source": src}
    info = {"file": None, "calibrated": False, "form": "navcam-v1", "label_fit": {
        "images": len(fits), "fit_rms_px_median": rms, "fit_rms_px_max": float(max(r["rms_px"] for r in reps)),
        "closed_form_rms_px_median": float(np.median([r["closed_form_rms_px"] for r in reps])),
        "central": bool(reps[0]["central"]), "depths_m": reps[0]["depths_m"], "spread_px": spread}}
    return cam, info


__all__ = ["CENTRAL_E_M", "NONCENTRAL_DEPTHS_M", "fisheye_from_cahvore", "project_thin_prism_fisheye",
           "full_frame_model", "label_fisheye", "meta_fisheye", "block_camera", "is_central"]
