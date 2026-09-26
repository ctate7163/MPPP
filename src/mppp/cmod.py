"""
JPL camera models (CAHV, CAHVOR, CAHVORE) and their link to COLMAP cameras (v0p21).

The PDS labels carry the flight team's calibration as a CAHV-family model in
the rover navigation frame:

* Mastcam-Z: CAHVOR, interpolated by zoom motor count.
* Navcam: CAHVORE of type 2 ("fisheye", linearity 0), interpolated by
  temperature; its E terms are ~1e-8 m (a central camera in practice).

:class:`CameraModel` projects 3-D points exactly as the JPL ``cmod`` library
does (Gennery 2006), in pixels whose origin is the CENTRE of the first pixel.
COLMAP puts the origin at the pixel CORNER, hence the ±0.5 px in the
conversions below.

:func:`fit_to_colmap` fits a CAHVOR or CAHVORE model to a refined COLMAP camera
(rays through a grid of pixels): the bridge from a bundle adjustment back to
an updated flight-style calibration.  :func:`compare_to_colmap` measures how
far a label model and a COLMAP camera disagree, after removing the camera
rotation both can absorb.

CAHVORE projection, for P relative to C:  zeta = p.O,  lambda = p - zeta O.
theta solves  zeta sin(theta) - |lambda| cos(theta) - (theta - sin theta) E(theta) = 0
with E(theta) = e0 + e1 theta^2 + e2 theta^4 (entrance-pupil movement);
chi = tan(L theta)/L (L > 0), theta (L = 0, fisheye) or sin(L theta)/L (L < 0);
mu = r0 + r1 chi^2 + r2 chi^4;  p' = (|lambda|/chi) O + (1 + mu) lambda;
x = p'.H / p'.A,  y = p'.V / p'.A.  Type 1 has L = 1 (perspective; this is then
CAHVOR for E = 0), type 2 L = 0, type 3 L = the model's parameter P.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

CAHVORE_LINEARITY = {1: 1.0, 2: 0.0}          # type -> linearity (type 3: the label's parameter)


@dataclass
class CameraModel:
    """A CAHV / CAHVOR / CAHVORE model.  Pixels: centre of the first pixel = (0, 0)."""
    C: np.ndarray
    A: np.ndarray
    H: np.ndarray
    V: np.ndarray
    O: Optional[np.ndarray] = None
    R: Optional[np.ndarray] = None
    E: Optional[np.ndarray] = None
    mtype: Optional[int] = None                 # CAHVORE type: 1 perspective, 2 fisheye, 3 general
    mparm: float = 0.0                          # CAHVORE type-3 linearity
    width: Optional[int] = None
    height: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for k in ("C", "A", "H", "V", "O", "R", "E"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, np.asarray(v, float).reshape(3))
        # A, H and V may be scaled together without changing the projection: make |A| = 1 (labels
        # carry |A| = 1 to ~1e-6), so that A.H is exactly the principal point; O is a direction.
        n = float(np.linalg.norm(self.A))
        if n > 0 and abs(n - 1.0) > 0:
            self.A, self.H, self.V = self.A / n, self.H / n, self.V / n
        if self.O is not None:
            self.O = self.O / np.linalg.norm(self.O)

    # ------------------------------------------------------------ identity
    @property
    def kind(self) -> str:
        if self.E is not None:
            return "CAHVORE"
        return "CAHVOR" if self.R is not None else "CAHV"

    @property
    def linearity(self) -> float:
        if self.mtype in CAHVORE_LINEARITY:
            return CAHVORE_LINEARITY[self.mtype]
        return float(self.mparm)

    @classmethod
    def from_label(cls, gcm: Dict[str, Any], width: Optional[int] = None,
                   height: Optional[int] = None) -> "CameraModel":
        """From a PDS ``GEOMETRIC_CAMERA_MODEL`` group (MODEL_COMPONENT_1..9)."""
        def comp(i):
            v = gcm.get(f"MODEL_COMPONENT_{i}")
            return None if v is None else np.asarray(v, float)
        t = str(gcm.get("MODEL_TYPE", "CAHV")).upper()
        o = r = e = None
        mtype, mparm = None, 0.0
        if t in ("CAHVOR", "CAHVORE"):
            o, r = comp(5), comp(6)
        if t == "CAHVORE":
            e = comp(7)
            mtype = int(round(float(np.ravel(comp(8))[0]))) if comp(8) is not None else 1
            mparm = float(np.ravel(comp(9))[0]) if comp(9) is not None else 0.0
        return cls(comp(1), comp(2), comp(3), comp(4), o, r, e, mtype, mparm, width, height,
                   meta={"source": "PDS label", "model_type": t, "interpolation": gcm.get("INTERPOLATION_METHOD"),
                         "interpolation_value": gcm.get("INTERPOLATION_VALUE")})

    # ------------------------------------------------------------ projection
    def project(self, X: np.ndarray) -> np.ndarray:
        """3-D points (..., 3) in the model's frame -> (sample, line) (..., 2); NaN behind the camera."""
        X = np.asarray(X, float)
        p = X.reshape(-1, 3) - self.C
        if self.O is None or self.R is None:
            pp = p
        elif self.E is None:
            pp = self._cahvor(p)
        else:
            pp = self._cahvore(p)
        a = pp @ self.A
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([pp @ self.H / a, pp @ self.V / a], axis=1)
        uv[~(a > 0)] = np.nan
        return uv.reshape(X.shape[:-1] + (2,))

    def _cahvor(self, p: np.ndarray) -> np.ndarray:
        omega = p @ self.O
        lam = p - omega[:, None] * self.O
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.einsum("ij,ij->i", lam, lam) / omega ** 2
        mu = self.R[0] + self.R[1] * tau + self.R[2] * tau ** 2
        return p + mu[:, None] * lam

    def _cahvore(self, p: np.ndarray) -> np.ndarray:
        zeta = p @ self.O
        lamv = p - zeta[:, None] * self.O
        lam = np.linalg.norm(lamv, axis=1)
        e0, e1, e2 = self.E
        th = np.arctan2(lam, zeta)
        for _ in range(100):                                   # Newton on g(theta) = 0
            ct, st, t2 = np.cos(th), np.sin(th), th * th
            Et = e0 + t2 * (e1 + t2 * e2)
            dE = 2 * th * (e1 + 2 * t2 * e2)
            g = zeta * st - lam * ct - (th - st) * Et
            dg = zeta * ct + lam * st - (1 - ct) * Et - (th - st) * dE
            step = np.where(dg != 0, g / np.where(dg != 0, dg, 1.0), 0.0)
            th = th - step
            if np.nanmax(np.abs(step), initial=0.0) < 1e-13:
                break
        L = self.linearity
        with np.errstate(divide="ignore", invalid="ignore"):
            if L > 1e-15:
                chi = np.tan(L * th) / L
            elif L < -1e-15:
                chi = np.sin(L * th) / L
            else:
                chi = th
            mu = self.R[0] + self.R[1] * chi ** 2 + self.R[2] * chi ** 4
            zp = np.where(chi > 1e-12, lam / chi, zeta)
        pp = zp[:, None] * self.O + (1 + mu)[:, None] * lamv
        small = th < 1e-9
        pp[small] = p[small]
        bad = abs(L) * th > np.pi / 2                        # as JPL: outside the model's domain
        if np.any(bad):
            pp[bad] = np.nan
        return pp

    # ------------------------------------------------------------ geometry
    def decompose(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        -> (R_cam_from_frame, linear intrinsics), as :meth:`mppp.camera.CAHVOR.decompose`
        (Di & Li 2004): z = A, y = V' = (V - vc A)/vs, x = the part of H' = (H - hc A)/hs
        orthogonal to V'.  Intrinsics: hs = |A x H|, vs = |A x V| (focal lengths),
        hc = A.H, vc = A.V (principal point, pixel-centre origin), the H'/V' angle
        (90 deg for unskewed pixels).
        """
        A = self.A / np.linalg.norm(self.A)
        hc, vc = float(A @ self.H), float(A @ self.V)
        hs, vs = np.linalg.norm(np.cross(A, self.H)), np.linalg.norm(np.cross(A, self.V))
        Hp, Vp = (self.H - hc * A) / hs, (self.V - vc * A) / vs
        c = float(np.clip(Hp @ Vp, -1, 1))
        x = (Hp - c * Vp) / np.sqrt(1 - c * c)
        R = np.stack([x, Vp, A])
        return R, {"hs": float(hs), "vs": float(vs), "hc": hc, "vc": vc, "hv_angle_deg": float(np.degrees(np.arccos(c)))}

    def in_frame(self, R_new_from_old: np.ndarray, C_new: Optional[np.ndarray] = None) -> "CameraModel":
        """The same camera expressed in another frame: vectors rotated, centre moved to ``C_new``
        (default: rotated centre).  ``R_new_from_old`` maps old-frame vectors to the new frame."""
        R = np.asarray(R_new_from_old, float)
        rot = lambda v: None if v is None else R @ v                       # noqa: E731
        C = R @ self.C if C_new is None else np.asarray(C_new, float)
        return replace(self, C=C, A=rot(self.A), H=rot(self.H), V=rot(self.V), O=rot(self.O),
                       meta=dict(self.meta))

    def camera_frame(self) -> Tuple["CameraModel", np.ndarray]:
        """-> (this model in its own camera frame with C = 0 and A = z, R_cam_from_frame)."""
        R, _ = self.decompose()
        return self.in_frame(R, np.zeros(3)), R

    def rescaled(self, scale: float, dx: float = 0.0, dy: float = 0.0, width: Optional[int] = None,
                 height: Optional[int] = None) -> "CameraModel":
        """
        The model for full-frame pixels: a product downsampled by ``scale`` (0.5 for a
        half-resolution Navcam) whose first pixel is (dx, dy) product pixels from the
        detector corner.  Pixel-centre origin in and out:  x_full = (x + dx + 0.5)/scale - 0.5.
        ``width``/``height``: the full-frame size to record.
        """
        s = float(scale)
        H = (self.H + (dx + 0.5) * self.A) / s - 0.5 * self.A
        V = (self.V + (dy + 0.5) * self.A) / s - 0.5 * self.A
        return replace(self, H=H, V=V, width=width, height=height,
                       meta=dict(self.meta, rescaled={"scale": s, "dx": dx, "dy": dy}))

    def to_label_dict(self, precision: int = 7) -> Dict[str, Any]:
        """MODEL_COMPONENT_n values as a PDS label would carry them."""
        f = lambda v: [float(f"{x:.{precision}g}") for x in v]             # noqa: E731
        d: Dict[str, Any] = {"MODEL_TYPE": self.kind, "MODEL_COMPONENT_1": f(self.C), "MODEL_COMPONENT_2": f(self.A),
                             "MODEL_COMPONENT_3": f(self.H), "MODEL_COMPONENT_4": f(self.V)}
        if self.O is not None:
            d["MODEL_COMPONENT_5"], d["MODEL_COMPONENT_6"] = f(self.O), f(self.R)
        if self.E is not None:
            d["MODEL_COMPONENT_7"] = f(self.E)
            d["MODEL_COMPONENT_8"] = float(self.mtype or 1)
            d["MODEL_COMPONENT_9"] = float(self.mparm)
        return d

    def label_text(self, precision: int = 7) -> str:
        """The model as the lines of a PDS GEOMETRIC_CAMERA_MODEL group."""
        d = self.to_label_dict(precision)
        lines = [f"  MODEL_TYPE                      = {d['MODEL_TYPE']}"]
        for k, v in d.items():
            if k == "MODEL_TYPE":
                continue
            val = f"({','.join(f'{x:.{precision}g}' for x in v)})" if isinstance(v, list) else f"{v:g}"
            lines.append(f"  {k:<32s}= {val}")
        return "\n".join(lines)


# ---------------------------------------------------------------- COLMAP bridge
def pixel_grid(width: int, height: int, step: float = 64.0, margin: float = 0.5) -> np.ndarray:
    """Grid of COLMAP pixel coordinates (corner origin) covering the frame, corners included."""
    xs = np.unique(np.r_[np.arange(margin, width - margin, step), width - margin])
    ys = np.unique(np.r_[np.arange(margin, height - margin, step), height - margin])
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel()]


def colmap_rays(model: str, params: Sequence[float], uv: np.ndarray) -> np.ndarray:
    """Unit rays (N, 3) in the COLMAP camera frame through COLMAP pixels ``uv``; NaN where not invertible."""
    from .colmap import unproject_camera
    xy = unproject_camera(model, params, uv)
    d = np.c_[xy, np.ones(len(xy))]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _best_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising sum |R a_i - b_i|^2 (unit vectors, Kabsch)."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


class PixelCamera:
    """
    One interface for a COLMAP camera and a CAHV-family model, in its own camera
    frame (x right, y down, z forward) and COLMAP pixels (corner origin):
    ``project(Xc)`` and ``rays(uv)``.  Build with :meth:`colmap` or :meth:`cahv`.
    """

    def __init__(self, width: int, height: int, project, rays, name: str = ""):
        self.width, self.height, self._p, self._r, self.name = int(width), int(height), project, rays, name

    @classmethod
    def colmap(cls, model: str, params: Sequence[float], width: int, height: int, name: str = "") -> "PixelCamera":
        from .colmap import project_camera
        p = np.asarray(params, float)
        return cls(width, height, lambda X: project_camera(model, p, X), lambda uv: colmap_rays(model, p, uv),
                   name or model)

    @classmethod
    def cahv(cls, cm: CameraModel, width: Optional[int] = None, height: Optional[int] = None,
             name: str = "", as_is: bool = False) -> "PixelCamera":
        """``cm`` is taken in its own camera frame (see :meth:`CameraModel.camera_frame`), or as
        given (``as_is``: a model already in a camera frame with C = 0, e.g. one whose axes
        must match a pose taken from elsewhere)."""
        c = cm if as_is else cm.camera_frame()[0]
        w, h = width or cm.width, height or cm.height
        if w is None or h is None:
            raise ValueError("the CAHV model needs a width and height")
        return cls(w, h, lambda X: c.project(X) + 0.5, lambda uv: _rays_of(c, np.asarray(uv, float) - 0.5),
                   name or c.kind)

    def project(self, Xc: np.ndarray) -> np.ndarray:
        return self._p(np.asarray(Xc, float))

    def rays(self, uv: np.ndarray) -> np.ndarray:
        return self._r(np.asarray(uv, float))

    def valid(self, Xc: np.ndarray, uv: np.ndarray, margin: float = 0.02) -> np.ndarray:
        """True where ``uv`` = project(``Xc``) is a trustworthy image of ``Xc``: finite, inside
        the frame (plus ``margin``) and mapped back onto the same ray (no fold-over)."""
        uv = np.asarray(uv, float)
        d = np.asarray(Xc, float)
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        fin = np.all(np.isfinite(uv), axis=1)
        mx, my = margin * self.width, margin * self.height
        ok = fin & (uv[:, 0] > -mx) & (uv[:, 0] < self.width + mx) & (uv[:, 1] > -my) & (uv[:, 1] < self.height + my)
        back = np.full_like(d, np.nan)
        if ok.any():
            back[ok] = self.rays(uv[ok])
        return ok & (np.einsum("ij,ij->i", np.nan_to_num(back), d) > np.cos(1e-6))


def compare_cameras(a: PixelCamera, b: PixelCamera, step: float = 64.0, fit_rotation: bool = True,
                    rotation_radius: float = 0.85) -> Dict[str, Any]:
    """
    Where camera ``b`` puts the rays that camera ``a`` sees through a grid of its
    pixels: the difference (b minus a, pixels) over the frame.  ``fit_rotation``
    (default) first removes the best-fitting rotation between the two camera
    frames, since a pose absorbs it (a principal-point shift is largely a
    rotation); the rotation is reported.  Grid points where either camera is
    not invertible (a polynomial beyond its turning point) are dropped and
    counted in ``coverage``, as are rays that ``b`` puts outside its frame.
    """
    from scipy.spatial.transform import Rotation
    uv = pixel_grid(a.width, a.height, step)
    d = a.rays(uv)
    ok = np.all(np.isfinite(d), axis=1)
    R = np.eye(3)
    r = np.linalg.norm(uv - np.array([a.width, a.height]) / 2, axis=1) / np.hypot(a.width, a.height) * 2
    if fit_rotation:
        db = b.rays(uv)
        good = ok & np.all(np.isfinite(db), axis=1)
        inner = good & (r <= rotation_radius)            # corners, where lens models differ most, do not steer it
        R = _best_rotation(d[inner if inner.sum() >= 10 else good], db[inner if inner.sum() >= 10 else good])
    uvp = np.full_like(uv, np.nan)
    uvp[ok] = b.project(d[ok] @ R.T)
    # keep only pixels that b maps back onto the same ray (a polynomial folds back beyond its turning point)
    fin = np.all(np.isfinite(uvp), axis=1)
    back = np.full_like(d, np.nan)
    back[fin] = b.rays(uvp[fin])
    folded = ~(np.einsum("ij,ij->i", np.nan_to_num(back), np.nan_to_num(d @ R.T)) > np.cos(1e-6))
    uvp[folded] = np.nan
    # and pixels that land inside b's frame (with a 2 % margin): outside it a model is extrapolating
    mx, my = 0.02 * b.width, 0.02 * b.height
    out = ~((uvp[:, 0] > -mx) & (uvp[:, 0] < b.width + mx) & (uvp[:, 1] > -my) & (uvp[:, 1] < b.height + my))
    uvp[out] = np.nan
    diff = uvp - uv
    n = np.linalg.norm(diff, axis=1)
    m = np.isfinite(n)
    rms = lambda sel: float(np.sqrt(np.mean(n[sel & m] ** 2))) if np.any(sel & m) else float("nan")   # noqa: E731
    return {"uv": uv, "diff_px": diff, "norm_px": n, "radius_frac": r, "coverage": float(m.mean()),
            "rms_px": rms(np.ones_like(m)), "p95_px": float(np.nanpercentile(n, 95)) if m.any() else float("nan"),
            "max_px": float(np.nanmax(n)) if m.any() else float("nan"), "centre_rms_px": rms(r < 0.5),
            "edge_rms_px": rms((r >= 0.5) & (r <= 0.85)), "corner_rms_px": rms(r > 0.85),
            "rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))), "R_b_from_a": R}


def compare_to_colmap(cm: CameraModel, model: str, params: Sequence[float], width: int, height: int,
                      step: float = 64.0, fit_rotation: bool = True) -> Dict[str, Any]:
    """:func:`compare_cameras` of a COLMAP camera (``a``) and a CAHV-family model (``b``, its own camera frame)."""
    return compare_cameras(PixelCamera.colmap(model, params, width, height), PixelCamera.cahv(cm, width, height),
                           step, fit_rotation)


def _rays_of(cm: CameraModel, uv: np.ndarray, iterations: int = 30) -> np.ndarray:
    """Unit rays of a CAHV-family model through pixels (numerical inverse, Gauss-Newton)."""
    _, lin = cm.decompose()
    Rc, _ = cm.decompose()
    x = np.c_[(uv[:, 0] - lin["hc"]) / lin["hs"], (uv[:, 1] - lin["vc"]) / lin["vs"]]
    one = np.ones((len(x), 1))
    to3 = lambda q: (np.c_[q, one] @ Rc) + cm.C                     # noqa: E731  camera -> model frame
    h = 1e-7
    for _ in range(iterations):
        p0 = cm.project(to3(x))
        r = uv - p0
        jx = (cm.project(to3(x + [h, 0])) - p0) / h
        jy = (cm.project(to3(x + [0, h])) - p0) / h
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        dx = (r[:, 0] * jy[:, 1] - jy[:, 0] * r[:, 1]) / det
        dy = (jx[:, 0] * r[:, 1] - r[:, 0] * jx[:, 1]) / det
        x = x + np.c_[dx, dy]
        if np.nanmax(np.abs(np.c_[dx, dy]), initial=0) < 1e-13:
            break
    d = np.c_[x, one] @ Rc
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def fit_to_colmap(model: str, params: Sequence[float], width: int, height: int, kind: str = "CAHVOR",
                  mtype: int = 2, fit_linearity: Optional[bool] = None, step: float = 48.0,
                  start: Optional[CameraModel] = None) -> Tuple[CameraModel, Dict[str, Any]]:
    """
    CAHVOR / CAHVORE model, in the COLMAP camera frame (C = 0, A = z), that
    reproduces the COLMAP camera over the whole frame.  Free: hs, vs, hc, vc,
    the O tilt (2), r1, r2 (r0 = 0, which would only rescale the focal
    lengths), and for CAHVORE type 3 the linearity (``fit_linearity``).  E is 0:
    COLMAP cameras are central.  Returns the model and the fit residuals
    (pixel-centre origin, full frame).
    """
    from scipy.optimize import least_squares
    uv = pixel_grid(width, height, step)
    d = colmap_rays(model, params, uv)
    ok = np.all(np.isfinite(d), axis=1)
    uv, d = uv[ok] - 0.5, d[ok]
    p = np.asarray(params, float)
    kind = kind.upper()
    if kind not in ("CAHVOR", "CAHVORE"):
        raise ValueError("kind must be CAHVOR or CAHVORE")
    if fit_linearity is None:                               # type 3 means a fitted linearity
        fit_linearity = kind == "CAHVORE" and mtype == 3
    lin0 = 0.0 if mtype == 2 else 1.0
    if start is not None:
        s, _ = start.camera_frame()
        _, l0 = s.decompose()
        o = s.O / s.O[2]
        x0 = [l0["hs"], l0["vs"], l0["hc"], l0["vc"], o[0], o[1], s.R[1], s.R[2]]
        if kind == "CAHVORE" and fit_linearity:
            x0.append(s.linearity if s.E is not None else lin0)
    else:
        from .colmap import _ONE_FOCAL
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model in _ONE_FOCAL else (p[0], p[1], p[2], p[3])
        x0 = [fx, fy, cx - 0.5, cy - 0.5, 0.0, 0.0, 0.0, 0.0]
        if kind == "CAHVORE" and fit_linearity:
            x0.append(lin0)

    def build(x) -> CameraModel:
        A = np.array([0.0, 0, 1])
        H = np.array([x[0], 0, 0]) + x[2] * A
        V = np.array([0, x[1], 0]) + x[3] * A
        O = np.array([x[4], x[5], 1.0])
        O /= np.linalg.norm(O)
        R = np.array([0.0, x[6], x[7]])
        cm = CameraModel(np.zeros(3), A, H, V, O, R, width=width, height=height,
                         meta={"source": f"fit to COLMAP {model}"})
        if kind == "CAHVORE":
            cm.E = np.zeros(3)
            if fit_linearity:
                cm.mtype, cm.mparm = 3, float(x[8])
            else:
                cm.mtype = mtype
        return cm

    def res(x):
        r = build(x).project(d) - uv
        return np.nan_to_num(r, nan=1e3).ravel()

    sol = least_squares(res, np.array(x0, float), x_scale="jac", method="lm", max_nfev=4000)
    cm = build(sol.x)
    r = (cm.project(d) - uv)
    n = np.linalg.norm(r, axis=1)
    rad = np.linalg.norm(uv + 0.5 - np.array([width, height]) / 2, axis=1) / np.hypot(width, height) * 2
    rep = {"rms_px": float(np.sqrt(np.mean(n ** 2))), "max_px": float(np.max(n)),
           "p95_px": float(np.percentile(n, 95)),
           "corner_rms_px": float(np.sqrt(np.mean(n[rad > 0.85] ** 2))) if np.any(rad > 0.85) else float("nan"),
           "n_points": int(len(n)), "kind": cm.kind, "mtype": cm.mtype, "linearity": cm.linearity if cm.E is not None else None,
           "o_tilt_deg": float(np.degrees(np.arccos(np.clip(cm.O @ cm.A, -1, 1)))), "success": bool(sol.success)}
    return cm, rep


def model_in_world(cm_cam: CameraModel, R_cam_from_world: np.ndarray, C_world: np.ndarray) -> CameraModel:
    """A camera-frame model (C = 0) placed at ``C_world`` with attitude ``R_cam_from_world``."""
    R = np.asarray(R_cam_from_world, float)
    return cm_cam.in_frame(R.T, np.asarray(C_world, float))
