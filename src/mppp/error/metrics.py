"""
mppp_error.metrics -- derived scalars from a fused covariance field.

PRIMARY METRIC
--------------
sigma_n  : precision along the surface normal.  This IS the DEM vertical
           precision and is directly comparable to published numbers (HiRISE
           expected precision, rover DTM vertical precision).

G_n      : sigma_n / (eps_intra * psi * r_ref), dimensionless.

           "Vertical error in units of the best available pixel footprint."
           G_n = 1 means you are at the resolution limit; G_n = 20 means network
           geometry costs a factor of 20 over what the optics could deliver.
           Because eps_intra is the only dimensional scale on Sigma (eps_cross
           lives in the gate weights, not the covariance), G_n is independent of
           eps and therefore comparable across zoom settings, instruments,
           ranges and sites.  This is the currency for a benchmark.

INTERPRETATION GUIDE  (see metric_guide())
------------------------------------------
sigma_t1/t2 : lateral registration accuracy.  Matters for feature localisation
              and for tying to orbital data; NOT for DEM quality.
kappa       : anisotropy.  kappa >> 1 means one direction is unconstrained --
              look at worst_direction() to find out WHICH, because that tells
              you where to put the next station.
n_eff       : effective independent station count.  n_eff << n_vis means your
              stations are redundant and adding more nearby ones will not help.
gain        : sigma_n(best single station) / sigma_n(fused).  Pure network
              value, independent of eps.  The cleanest network-design figure of
              merit in the set.
sigma_slope : differential error between adjacent cells.  Predicts PERCEIVED
              quality -- smooth bias is invisible, local bumpiness is not.
              Often the more relevant map for outreach / VR products.
coverage_at : area fraction meeting a science threshold.  The single headline
              number for ranking candidate acquisitions.
gsd         : ground sample distance, psi*r/cos(e) downrange.  The resolution
              floor, independent of network geometry.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .core import PrecisionField, tangent_basis

__all__ = ["compute_metrics", "metric_guide", "worst_direction", "summarize"]


def _quad(v, M):
    return np.einsum("...i,...ij,...j->...", v, M, v)


def compute_metrics(field: PrecisionField,
                    rho_adjacent: float = 0.0,
                    coverage_threshold_m: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Compute all derived maps.  Unconstrained cells are set to NaN.

    rho_adjacent : correlation between neighbouring cells' errors, used for
                   sigma_slope.  0 gives the pessimistic (fully independent)
                   bumpiness; values near 1 mean errors are smooth and slope
                   error is small.  UNMEASURED -- treat as a swept parameter.
    """
    S = field.Sigma
    N = field.normals
    bad = field.unconstrained

    out: Dict[str, np.ndarray] = {}

    # --- normal (vertical) precision -- the primary metric -----------------
    var_n = np.maximum(_quad(N, S), 0.0)
    sigma_n = np.sqrt(var_n)
    out["sigma_n"] = np.where(bad, np.nan, sigma_n)

    # --- tangent-plane block ----------------------------------------------
    T = tangent_basis(N)                                   # (Ny,Nx,2,3)
    S_t = np.einsum("...ai,...ij,...bj->...ab", T, S, T)   # (Ny,Nx,2,2)
    ev_t = np.linalg.eigvalsh(S_t)                         # ascending
    out["sigma_t_min"] = np.where(bad, np.nan, np.sqrt(np.maximum(ev_t[..., 0], 0)))
    out["sigma_t_max"] = np.where(bad, np.nan, np.sqrt(np.maximum(ev_t[..., 1], 0)))

    # --- full 3D ellipsoid -------------------------------------------------
    ev = np.linalg.eigvalsh(S)
    s_min = np.sqrt(np.maximum(ev[..., 0], 0))
    s_max = np.sqrt(np.maximum(ev[..., 2], 0))
    out["sigma_min"] = np.where(bad, np.nan, s_min)
    out["sigma_max"] = np.where(bad, np.nan, s_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["kappa"] = np.where(bad | (s_min <= 0), np.nan, s_max / s_min)

    # --- resolution & normalisation ---------------------------------------
    psi = field.ifov_ref
    eps = field.eps_intra_px
    r_ref = field.range_min
    with np.errstate(divide="ignore", invalid="ignore"):
        footprint = eps * psi * r_ref
        out["G_n"] = np.where(bad | ~np.isfinite(footprint) | (footprint <= 0),
                              np.nan, sigma_n / footprint)
        out["gsd"] = np.where(np.isfinite(r_ref) & (field.cos_e_ref > 0),
                              psi * r_ref / np.maximum(field.cos_e_ref, 1e-6), np.nan)
    out["range_min"] = r_ref
    out["cos_e_ref"] = field.cos_e_ref

    # --- counts ------------------------------------------------------------
    out["n_vis"] = field.n_vis.astype(float)
    out["n_rays"] = field.n_rays.astype(float)

    # --- effective independent stations ------------------------------------
    out["n_eff"] = _n_eff(field)

    # --- improvement ------------------------------------------------------
    best_single = np.min(field.sigma_n_single, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(np.isfinite(best_single) & (sigma_n > 0),
                        best_single / sigma_n, np.nan)
    out["improvement"] = np.where(bad, np.nan, gain)
    out["sigma_n_best_single"] = np.where(np.isfinite(best_single), best_single, np.nan)
    out["gain"] = out["improvement"]      # backwards-compatible alias

    # --- differential (slope) error ---------------------------------------
    out["sigma_slope"] = _sigma_slope(field, out["sigma_n"], rho_adjacent)

    # --- coverage ----------------------------------------------------------
    if coverage_threshold_m is not None:
        ok = np.isfinite(out["sigma_n"]) & (out["sigma_n"] < coverage_threshold_m)
        out["coverage_mask"] = ok.astype(float)
        out["_coverage_fraction"] = np.array(
            float(ok.sum()) / float(np.isfinite(out["sigma_n"]).sum() or 1))

    out["unconstrained"] = bad.astype(float)
    return out


def _n_eff(field: PrecisionField) -> np.ndarray:
    """
    Effective independent station count via the participation ratio.

    With correlation disabled every visible station is independent by
    construction, so n_eff == n_vis.  The map is still worth producing when
    correlation is enabled: n_eff << n_vis localises the redundant geometry.
    """
    cfg = field.config
    vis = field.station_vis
    Nvis = vis.sum(axis=0).astype(float)
    if not cfg.use_correlation or field.station_dirs is None:
        return Nvis
    u = field.station_dirs
    tc = np.radians(cfg.theta_c_deg)
    num = np.zeros(Nvis.shape)
    for i in range(u.shape[0]):
        for j in range(u.shape[0]):
            both = vis[i] & vis[j]
            if i == j:
                num += both
                continue
            c = np.clip(np.einsum("...k,...k->...", u[i], u[j]), -1.0, 1.0)
            th = np.arccos(c)
            rho = cfg.rho_inf + (1.0 - cfg.rho_inf) * np.exp(-th ** 2 / (2 * tc ** 2))
            num += np.where(both, rho, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(num > 0, Nvis ** 2 / num, 0.0)


def _sigma_slope(field: PrecisionField, sigma_n: np.ndarray,
                 rho_adjacent: float) -> np.ndarray:
    """
    Slope (differential) error between adjacent cells:

        sigma_slope ~ sigma_n * sqrt(2 (1 - rho_adj)) / dx

    Uses the local mean of sigma_n over the neighbour pair.  Dimensionless
    (rise over run); multiply by 100 for percent slope.
    """
    g = field.grid
    dx = float(np.mean(np.diff(g.x))) if g.x.size > 1 else 1.0
    dy = float(np.mean(np.diff(g.y))) if g.y.size > 1 else 1.0
    k = np.sqrt(2.0 * max(1.0 - rho_adjacent, 0.0))
    sx = np.full_like(sigma_n, np.nan)
    sy = np.full_like(sigma_n, np.nan)
    sx[:, :-1] = 0.5 * (sigma_n[:, :-1] + sigma_n[:, 1:]) * k / abs(dx)
    sy[:-1, :] = 0.5 * (sigma_n[:-1, :] + sigma_n[1:, :]) * k / abs(dy)
    return np.sqrt(np.nan_to_num(sx, nan=0.0) ** 2 + np.nan_to_num(sy, nan=0.0) ** 2) \
        * np.where(np.isfinite(sigma_n), 1.0, np.nan)


def worst_direction(field: PrecisionField) -> np.ndarray:
    """
    Unit eigenvector of the largest covariance eigenvalue, per cell (Ny,Nx,3).

    This is the direction in which the reconstruction is least constrained.
    Where kappa is large, a new station placed so as to look ALONG this
    direction will improve the network most -- it is the actionable output of
    the anisotropy map.
    """
    w, V = np.linalg.eigh(field.Sigma)
    return V[..., :, -1]


def summarize(field: PrecisionField, m: Dict[str, np.ndarray]) -> str:
    """Compact text summary for logs and notebooks."""
    def stat(k, scale=1.0, unit=""):
        a = m.get(k)
        if a is None or a.ndim == 0:
            return f"  {k:<22s} n/a"
        v = a[np.isfinite(a)]
        if v.size == 0:
            return f"  {k:<22s} all NaN"
        return (f"  {k:<22s} med {np.median(v)*scale:9.4g}  "
                f"p90 {np.percentile(v,90)*scale:9.4g}  "
                f"max {v.max()*scale:9.4g} {unit}")

    lines = [
        f"stations           : {len(field.stations)}",
        f"grid               : {field.grid.shape[0]} x {field.grid.shape[1]}",
        f"pose mode          : {field.pose.mode}",
        f"occlusion          : {field.config.apply_occlusion}",
        f"theta gate         : {field.config.use_theta_gate} "
        f"({field.config.theta_max_deg} deg)",
        f"correlation        : {field.config.use_correlation} "
        f"(theta_c={field.config.theta_c_deg} deg, rho_inf={field.config.rho_inf})",
        f"eps_intra [px]     : {field.eps_intra_px}",
        f"IFOV [urad]        : {field.ifov_ref*1e6:.1f}",
        f"unconstrained cells: {int(field.unconstrained.sum())} "
        f"/ {field.unconstrained.size}",
        "",
        stat("sigma_n", 100.0, "cm"),
        stat("G_n"),
        stat("gsd", 100.0, "cm"),
        stat("kappa"),
        stat("n_vis"),
        stat("n_eff"),
        stat("improvement"),
        stat("sigma_slope", 100.0, "%"),
    ]
    if "_coverage_fraction" in m:
        lines.append(f"  coverage fraction      {float(m['_coverage_fraction']):.3f}")
    return "\n".join(lines)


def metric_guide() -> str:
    """Printable interpretation guide."""
    return __doc__
