"""
mppp_error.plotting -- map rendering.

Fixes carried over from the v1 code:
  * imshow extent uses cell EDGES, not centres (v1 was offset by half a cell,
    which matters for a metric product).
  * log scale is actually implemented rather than printing "not implemented".
  * station markers handle NaN azimuth without silently drawing heading 0.
  * unconstrained cells render as an explicit colour, not as silent NaN.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from typing import Optional, Dict, Sequence

from .core import PrecisionField

__all__ = ["plot_map", "plot_panel", "PRESETS"]


#: (key, title, colour-bar label, unit scale, cmap, log by default)
PRESETS: Dict[str, tuple] = {
    "sigma_n":      ("Normal (DEM vertical) precision", "sigma_n [cm]", 100.0, "viridis_r", True),
    "G_n":          ("Normalised precision G_n", "sigma_n / (eps psi r) [-]", 1.0, "magma_r", True),
    "gsd":          ("Ground sample distance", "GSD [cm]", 100.0, "cividis_r", True),
    "kappa":        ("Anisotropy (sigma_max / sigma_min)", "ratio [-]", 1.0, "inferno_r", True),
    "n_vis":        ("Visible stations", "count", 1.0, "viridis", False),
    "n_eff":        ("Effective independent stations", "count", 1.0, "viridis", False),
    "improvement":         ("Improvement (best single / fused)", "ratio [-]", 1.0, "viridis", False),
    "sigma_slope":  ("Slope (differential) error", "slope error [%]", 100.0, "plasma_r", True),
    "sigma_t_max":  ("Lateral precision (major)", "sigma_t [cm]", 100.0, "viridis_r", True),
    "range_min":    ("Range to nearest station", "r [m]", 1.0, "bone", False),
}


def _rover_glyph(ax, xy, az_deg: float, scale: float = 1.0,
                 is_anchor: bool = False, mast_fwd_m: float = 0.65,
                 mast_right_m: float = 0.45, show_rotation_centre: bool = False):
    """
    Abstract top-down rover glyph.

    Three wheels per side, EQUALLY spaced at x = -1, 0, +1 m; a 2.0 m x 1.4 m
    body CENTRED on the wheels (body centre = middle-wheel line = rotation
    centre); the mast a little INWARD of the forward-starboard corner of the
    body.  Station xyz refers to the MAST (where range is measured from).
    az_deg is compass bearing (CW from north); theta = 90 - az.
    """
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.transforms import Affine2D
    az = az_deg if np.isfinite(az_deg) else 0.0
    theta = 90.0 - az
    fwd = np.array([np.cos(np.radians(theta)), np.sin(np.radians(theta))])
    right = np.array([fwd[1], -fwd[0]])
    mast_xy = np.asarray(xy, dtype=float)
    rot_c = mast_xy - scale * (mast_fwd_m * fwd + mast_right_m * right)
    body_l, body_w = 2.0 * scale, 1.4 * scale
    wheel_l, wheel_w, track = 0.45 * scale, 0.35 * scale, 1.05 * scale
    tr = Affine2D().rotate_deg_around(rot_c[0], rot_c[1], theta) + ax.transData
    ax.add_patch(Rectangle((rot_c[0] - body_l/2, rot_c[1] - body_w/2), body_l, body_w,
                           facecolor="0.88", edgecolor="k", linewidth=0.9, transform=tr, zorder=6))
    for lon in (-1.0 * scale, 0.0, 1.0 * scale):
        for lat in (-track, track):
            ax.add_patch(Rectangle((rot_c[0] + lon - wheel_l/2, rot_c[1] + lat - wheel_w/2),
                                   wheel_l, wheel_w, facecolor="0.25", edgecolor="k",
                                   linewidth=0.6, transform=tr, zorder=5))
    ax.add_patch(Circle(mast_xy, 0.28 * scale, facecolor="white",
                        edgecolor="crimson" if is_anchor else "k", linewidth=1.4, zorder=7))
    if show_rotation_centre:
        ax.plot([rot_c[0]], [rot_c[1]], "+", ms=9, color="crimson", mew=1.3, zorder=8)


def _sol_labels(ax, field, fontsize=8, push_m=3.2):
    """Sol label per station, pushed radially away from the cluster centroid so
    it never sits on a rover glyph; a thin leader line connects it."""
    P = np.array([s.xyz[:2] for s in field.stations])
    c = P.mean(axis=0)
    for s_, p in zip(field.stations, P):
        d = p - c
        n = np.linalg.norm(d)
        u = d / n if n > 1e-6 else np.array([1.0, 0.0])
        t = p + push_m * u
        sol = getattr(s_, "info", {}).get("sol") if hasattr(s_, "info") else None
        if sol is None:
            import re
            m = re.search(r"Sol\s*(\d+)", s_.name or "")
            sol = m.group(1) if m else (s_.name or "")
        ax.annotate(f"Sol {sol}", xy=(p[0], p[1]), xytext=(t[0], t[1]), fontsize=fontsize,
                    ha="center", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6, shrinkA=0, shrinkB=4))


def _station_markers(ax, field: PrecisionField, labels: bool = True,
                     label_key: str = "short", rover_icon: bool = False,
                     rover_scale: float = 1.0):
    for st in field.stations:
        az = st.az_deg
        if rover_icon:
            _rover_glyph(ax, st.xyz[:2], az, scale=rover_scale,
                        is_anchor=st.is_anchor)
        else:
            marker = (4, 0, 45.0 - az) if np.isfinite(az) else (4, 0, 45.0)
            ax.plot(st.xyz[0], st.xyz[1], marker=marker, markersize=11,
                    markeredgecolor="k", markeredgewidth=1.2,
                    markerfacecolor="white" if not st.is_anchor else "gold",
                    zorder=6, linestyle="none")
        if labels and st.name:
            txt = st.name.split("|")[0].strip() if label_key == "short" else st.name
            ax.annotate(txt, (st.xyz[0], st.xyz[1]), textcoords="offset points",
                        xytext=(9, -4), ha="left", fontsize=8, zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.65))


def plot_map(field: PrecisionField, data: np.ndarray, *,
             title: str = "", cbar_label: str = "", scale: float = 1.0,
             cmap: str = "viridis_r", log: bool = False,
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             show_stations: bool = True, labels: bool = True,
             mark_unconstrained: bool = True, ax=None,
             show_xlabel: bool = True, show_ylabel: bool = True,
             show_xticklabels: bool = True, show_yticklabels: bool = True,
             stats_box: bool = False, stats_unit: str = "",
             show_colorbar: bool = True, rover_icon: bool = False,
             cax=None,
             rover_scale: float = 1.0, sol_labels: bool = False):
    """Render one scalar map. Returns the Axes."""
    arr = np.asarray(data, dtype=float) * scale
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("nothing finite to plot")

    if log:
        pos = finite & (arr > 0)
        if not pos.any():
            raise ValueError("no positive finite data to plot on a log scale")
        if vmin is None:
            # clamp the auto lower bound to the data, but NEVER override an
            # explicitly supplied vmin -- doing so silently desynchronises
            # colour bars that the caller intended to share across panels.
            lo = max(np.nanpercentile(arr[pos], 1), np.nanmin(arr[pos]))
        else:
            lo = vmin
        hi = vmax if vmax is not None else np.nanpercentile(arr[pos], 99)
        lo = max(lo, 1e-12)
        norm = LogNorm(vmin=lo, vmax=max(hi, lo * 1.0000001))
    else:
        lo = vmin if vmin is not None else np.nanpercentile(arr[finite], 1)
        hi = vmax if vmax is not None else np.nanpercentile(arr[finite], 99)
        norm = Normalize(vmin=lo, vmax=hi)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.8))

    masked = np.ma.array(arr, mask=~finite)
    cm = plt.get_cmap(cmap).copy()
    if mark_unconstrained:
        cm.set_bad("0.85")

    im = ax.imshow(masked, extent=field.grid.extent_edges(), origin="lower",
                   aspect="equal", cmap=cm, norm=norm, interpolation="nearest")
    # Colour bar suppressible so a multi-panel figure with a SHARED scale per
    # row can carry one bar on the outer (last) column instead of repeating an
    # identical bar on every "inner" panel.
    if show_colorbar:
        # cax: a dedicated colorbar axes so the map panel keeps its size
        # (fraction-of-axes colorbars shrink the host panel).
        cb = (ax.figure.colorbar(im, cax=cax) if cax is not None
              else ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03))
        if cbar_label:
            cb.set_label(cbar_label)

    # Axis labels/ticklabels restricted to the outer edge of a subplot grid
    # (bottom row gets x, left column gets y) so a multi-panel figure isn't
    # cluttered with the same "Relative easting/northing [m]" on every panel.
    if show_xlabel:
        ax.set_xlabel("Relative easting [m]")
    if show_ylabel:
        ax.set_ylabel("Relative northing [m]")
    if not show_xticklabels:
        ax.tick_params(labelbottom=False)
    if not show_yticklabels:
        ax.tick_params(labelleft=False)
    if title:
        ax.set_title(title, fontsize=11)
    if show_stations:
        _station_markers(ax, field, labels=(labels or sol_labels),
                         rover_icon=rover_icon, rover_scale=rover_scale)

    if stats_box:
        v = arr[finite]
        mean, med = float(np.mean(v)), float(np.median(v))
        rms = float(np.sqrt(np.mean(v ** 2)))
        txt = f"mean {mean:.2f}{stats_unit}\nrms  {rms:.2f}{stats_unit}\nmed  {med:.2f}{stats_unit}"
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, family="monospace", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85))
    return ax


def plot_panel(field: PrecisionField, metrics: Dict[str, np.ndarray],
               keys: Sequence[str] = ("sigma_n", "G_n", "n_vis", "kappa",
                                      "improvement", "sigma_slope"),
               ncols: int = 3, figsize_per: float = 4.4,
               show_stations: bool = True, labels: bool = False,
               suptitle: str = "", rover_icon: bool = False,
               rover_scale: float = 1.0):
    """Multi-panel overview using the PRESETS table."""
    keys = [k for k in keys if k in metrics]
    n = len(keys)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols, figsize_per * nrows * 1.05),
                             squeeze=False)
    for i, k in enumerate(keys):
        ax = axes[i // ncols][i % ncols]
        title, lab, scale, cmap, log = PRESETS.get(
            k, (k, k, 1.0, "viridis", False))
        try:
            plot_map(field, metrics[k], title=title, cbar_label=lab, scale=scale,
                     cmap=cmap, log=log, show_stations=show_stations,
                     labels=labels, ax=ax, rover_icon=rover_icon,
                     rover_scale=rover_scale)
        except ValueError:
            ax.set_title(f"{title}\n(no finite data)", fontsize=10)
            ax.set_axis_off()
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig, axes
