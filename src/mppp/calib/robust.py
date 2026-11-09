"""
mppp.calib.robust - the robust numerical helpers the pipeline stages and the calibration fits share (0.96.0: moved
out of the calibration study's ``fit`` module so that the core never imports the solvers): :func:`huber_fit`,
:func:`huber_line`, :func:`weighted_median`, :func:`weighted_mean`, :func:`meta_line`, :func:`ols`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def huber_fit(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, k: float = 1.345, floor: float = 0.05,
              iterations: int = 30) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Weighted Huber IRLS least squares y = X c.  Returns (coefficients, standard errors from the residual scatter,
    robust sd of the residuals (1.4826 MAD, at least ``floor``), residuals)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    wt = np.ones(len(y))
    c = np.zeros(X.shape[1])
    sig = float(floor)
    for _ in range(int(iterations)):
        sw = np.sqrt(w * wt)
        c = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = y - X @ c
        sig = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), float(floor))
        u = np.abs(r) / (k * sig)
        wt = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    resid = y - X @ c
    W = w * wt / max(float(np.mean(w * wt)), 1e-12)
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(np.sum(W * resid ** 2) / dof)
    try:
        C = np.linalg.inv(X.T @ (X * W[:, None])) * s2
        se = np.sqrt(np.maximum(np.diag(C), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return c, se, sig, resid


def huber_line(F: np.ndarray, y: np.ndarray, w: np.ndarray, ref: float, fit_slope: bool, slope0: float = 0.0,
               k: float = 1.345, floor: float = 0.05) -> Tuple[float, float, float, np.ndarray, Tuple[float, Optional[float]]]:
    """(a, b, robust sd, residuals, (sd_a, sd_b)) of y = a + b (F - ref) by :func:`huber_fit`; ``fit_slope``
    False: b = ``slope0`` and only a is fitted."""
    F, y = np.asarray(F, float), np.asarray(y, float)
    X = np.c_[np.ones(len(F)), F - ref] if fit_slope else np.ones((len(F), 1))
    yy = y if fit_slope else y - float(slope0) * (F - ref)
    c, se, sig, _ = huber_fit(X, yy, w, k, floor)
    a, b = float(c[0]), (float(c[1]) if fit_slope else float(slope0))
    return a, b, sig, y - (a + b * (F - ref)), (float(se[0]), float(se[1]) if fit_slope else None)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def weighted_mean(y: np.ndarray, sd: np.ndarray) -> Dict[str, float]:
    """The inverse-variance weighted mean of ``y`` with the between-value scatter tau (DerSimonian-Laird):
    {"value", "se", "tau", "chi2_dof", "n"}."""
    return meta_line(y, sd)


def meta_line(y: np.ndarray, sd: np.ndarray, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Weighted mean (or line in ``x``) of ``y`` with the between-value scatter tau (DerSimonian-Laird) and
    chi^2 / dof; a line adds "slope", "slope_se", "slope_z"."""
    y = np.asarray(y, float)
    sd = np.asarray(sd, float)
    good = np.isfinite(sd) & (sd > 0)
    sd = np.where(good, sd, np.nanmedian(sd[good]) if np.any(good) else 1.0)
    w = 1.0 / sd ** 2
    X = np.ones((len(y), 1)) if x is None else np.c_[np.ones(len(y)), np.asarray(x, float)]
    p = X.shape[1]
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    r = y - X @ beta
    Q = float(np.sum(w * r ** 2))
    dof = max(len(y) - p, 1)
    c = float(np.sum(w) - (np.sum(w ** 2) / np.sum(w) if p == 1 else 0.0))
    tau2 = max(0.0, (Q - dof) / c) if (p == 1 and c > 0) else max(0.0, (Q / dof - 1.0) * float(np.mean(sd ** 2)))
    w2 = 1.0 / (sd ** 2 + tau2)
    beta2 = np.linalg.lstsq(X * np.sqrt(w2)[:, None], y * np.sqrt(w2), rcond=None)[0]
    try:
        cov = np.linalg.inv(X.T @ (X * w2[:, None]))
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    out = {"n": int(len(y)), "chi2_dof": Q / dof, "tau": float(np.sqrt(tau2)), "value": float(beta2[0]),
           "se": float(np.sqrt(max(cov[0, 0], 0.0)))}
    if p == 2:
        out.update({"slope": float(beta2[1]), "slope_se": float(np.sqrt(max(cov[1, 1], 0.0)))})
        out["slope_z"] = out["slope"] / out["slope_se"] if out["slope_se"] > 0 else None
    return out


def ols(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """y = c + s (x - mean x): slope, its se (residual scatter), z, the scatter (rms) and n (rows with finite x, y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return {"n": int(len(x)), "slope": None, "se": None, "z": None}
    xm = float(np.mean(x))
    X = np.c_[np.ones(len(x)), x - xm]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = float(np.sum(r ** 2) / max(len(x) - 2, 1))
    cov = np.linalg.inv(X.T @ X) * s2
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {"n": int(len(x)), "mean_x": xm, "intercept": float(b[0]), "slope": float(b[1]), "se": se,
            "z": float(b[1] / se) if se > 0 else None, "rms": float(np.sqrt(np.mean(r ** 2)))}
