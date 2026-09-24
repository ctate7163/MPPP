"""
mppp.compat -- drop-in replacements for the v1 error_map.py entry points.

Lets the existing notebook keep running while you migrate.  Both functions
carry the v1 signatures but are backed by the v2 engine, so the numbers WILL
change.  Expect, relative to v1:

  * sigma_range larger by sqrt(2) and sigma_transverse smaller by sqrt(2)
    (v1 used sigma_t = eps*psi*r and sigma_rho = eps*psi*r^2/b, so its
    anisotropy ratio was off by a factor of 2).
  * Different absolute scale if you were using baseline=0.42 with an IFOV that
    did not correspond to the same instrument.  v2 forces you to name the
    instrument.
  * Sensible values at grazing emission, because the bogus r/h obliquity
    factor is gone (it double-counted an effect the 3D geometry already
    produces).
  * Metres, not centimetres.  v1 multiplied by 100 inside the covariance while
    documenting the outputs as m^2.

Migration path: replace `fused_surface_error_field` with
`mppp.solve_precision_field` + `mppp.compute_metrics`, which give you sigma_n,
G_n, gain and n_eff instead of only 3D invariants.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Iterable, Tuple, Dict, Optional

from .core import (Grid, FlatPlane, Instrument, Station, ModelConfig, PoseModel,
                   OcclusionMask, DEFAULT_ROVER_MASK, solve_precision_field)
from .metrics import compute_metrics

__all__ = ["fused_surface_error_field", "make_report"]


def fused_surface_error_field(
    stations_xyz: Iterable[Iterable[float]],
    stations_az: Iterable[float],
    X_bounds: Tuple[float, float],
    Y_bounds: Tuple[float, float],
    grid_shape: Tuple[int, int],
    ifov: float,
    baseline: float,
    Z_plane: float = 0.0,
    cos_min: float = 0.34,
    apply_cos_scaling: bool = False,
    apply_rover_occlusion: bool = True,
    eps_intra_px: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    v1-compatible wrapper.  Returns the v1 keys (in METRES, not centimetres)
    plus the v2 additions.

    apply_cos_scaling is accepted and ignored.  The r/h obliquity factor it
    applied double-counts: build the 3D covariance correctly and 1/cos(e)
    emerges from the geometry.  Applying it again is wrong, and badly so at
    grazing emission, where it inflates the vertical error precisely where the
    vertical error is at its best.
    """
    if apply_cos_scaling:
        warnings.warn(
            "apply_cos_scaling is ignored in v2: the r/h obliquity factor "
            "double-counts an effect already present in the 3D geometry.",
            stacklevel=2)

    xyz = np.asarray(stations_xyz, dtype=float).reshape(-1, 3)
    az = np.asarray(stations_az, dtype=float).reshape(-1)
    inst = Instrument(name="v1", ifov_rad=float(ifov), baseline_m=float(baseline),
                      eps_intra_px=eps_intra_px)
    stations = [Station(xyz=xyz[i], az_deg=float(az[i]), instrument=inst,
                        mask=DEFAULT_ROVER_MASK if apply_rover_occlusion else None,
                        is_anchor=(i == 0))
                for i in range(len(xyz))]

    Ny, Nx = grid_shape
    grid = Grid(x=np.linspace(X_bounds[0], X_bounds[1], Nx),
                y=np.linspace(Y_bounds[0], Y_bounds[1], Ny),
                surface=FlatPlane(Z_plane))
    cfg = ModelConfig(apply_occlusion=apply_rover_occlusion,
                      cos_e_min=float(cos_min))
    field = solve_precision_field(grid, stations, cfg)
    m = compute_metrics(field)

    tr = np.trace(field.Sigma, axis1=-2, axis2=-1)
    det = np.linalg.det(field.Sigma)
    bad = field.unconstrained
    return dict(
        X=grid.x, Y=grid.y,
        Sigma_fused=field.Sigma,
        sigma_min=m["sigma_min"], sigma_max=m["sigma_max"],
        tra_Sigma=np.where(bad, np.nan, tr),
        det_Sigma=np.where(bad, np.nan, det),
        cond=m["kappa"],
        geom_mean=np.where(bad, np.nan, np.maximum(det, 0.0) ** (1 / 6)),
        rms_mean=np.where(bad, np.nan, np.sqrt(np.maximum(tr, 0.0) / 3.0)),
        visible_count=field.n_vis,
        # --- v2 additions ---
        sigma_n=m["sigma_n"], G_n=m["G_n"], gain=m["gain"],
        n_eff=m["n_eff"], gsd=m["gsd"], sigma_slope=m["sigma_slope"],
        _field=field, _metrics=m,
        meta=dict(Z_plane=Z_plane, ifov=ifov, baseline=baseline,
                  cos_min=cos_min, stations_xyz=xyz, stations_az=az,
                  X_bounds=X_bounds, Y_bounds=Y_bounds, units="metres"),
    )


def make_report(data: dict, anchor_sol: int = -1, title: str = "",
                radius_m: float = 25.0, vmax: Optional[float] = None,
                instrument: Optional[Instrument] = None,
                pose_mode: str = "none", camera_height_m: float = 1.9):
    """
    v1-compatible report entry point, rebuilt on the v2 stack.

    Differences from v1 that change results:
      * station elevations are preserved (v1 forced them all to 1.9 m)
      * the frame is centred on the anchor's GROUND position, so the evaluation
        plane is at z=0 and cameras sit above it
      * station selection radius is 2x the grid half-width, not 1x
    """
    import matplotlib.pyplot as plt
    from .waypoints import build_stations, site_drive_for_sol
    from .plotting import plot_panel
    from .metrics import summarize
    from .core import MASTCAM_Z_34

    inst = instrument or MASTCAM_Z_34
    site, drive, matched = site_drive_for_sol(data, anchor_sol)
    stations, info = build_stations(data, anchor_site=site, anchor_drive=drive,
                                    select_radius_m=2.0 * radius_m,
                                    instrument=inst,
                                    camera_height_m=camera_height_m)
    for s in stations:
        print(s.name)

    grid = Grid.square(np.ceil(radius_m), int(10 * radius_m) | 1)
    field = solve_precision_field(grid, stations, ModelConfig(),
                                  PoseModel(mode=pose_mode))
    m = compute_metrics(field)
    print(summarize(field, m))
    fig, _ = plot_panel(field, m, suptitle=title or f"Sol {matched}")
    plt.show()
    return dict(field=field, metrics=m, stations=stations, info=info,
                matched_sol=matched)
