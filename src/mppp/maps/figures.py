"""
Figures of the blocks of rover stations (MPPP v0p80, :mod:`mppp.maps`).

- :func:`traverse_map`: the traverse (every waypoint, end-of-drive Navcam 360s as dots) with the blocks as orange
  squares and their numbers, one colour per region in the legend; optional DEM background (:mod:`mppp.maps.terrain`).
- :func:`zoom_grid`: one +-30 m panel per block (5 columns), the waypoints inside with their sols, orange borders,
  "Relative northing [m]" on the left column and "Relative easting [m]" under the bottom panel of each column, the
  counts in the title ("N×n, Z×m").

The publication figures of the waypoint-map work (global Mars / Isidis / Jezero / traverse / regional panels on
HiRISE+MOLA terrain, ``mapplot3.py``) are not reproduced here; these are the working figures for the block list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]
ORANGE = "#ff8c00"


def _counts(b) -> str:
    z = len(b.get("zcam_360s") or [])
    return f"N×{b['navcam_360s']}" + (f", Z×{z}" if z else "")


def traverse_map(rows: Sequence[Dict[str, Any]], blocks: Sequence[Dict[str, Any]], out_png: Optional[PathLike] = None,
                 dem: Optional[PathLike] = None, figsize=(11, 11), label_size: float = 8.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    E = np.array([r["E"] for r in rows])
    N = np.array([r["N"] for r in rows])
    E0, N0 = float(E.min()), float(N.min())
    fig, ax = plt.subplots(figsize=figsize)
    if dem:
        from .terrain import hillshade_extent
        img, ext = hillshade_extent(dem, (E.min() - 200, E.max() + 200, N.min() - 200, N.max() + 200))
        if img is not None:
            ax.imshow(img, extent=(ext[0] - E0, ext[1] - E0, ext[2] - N0, ext[3] - N0), cmap="gray", origin="upper", alpha=0.8)
    ax.plot(E - E0, N - N0, "-", color="0.35", lw=0.7, zorder=2)
    m = np.array([r["end_of_drive"] and r["pano"] for r in rows])
    ax.plot(E[m] - E0, N[m] - N0, "o", ms=1.8, color="0.15", zorder=3, label="end-of-drive Navcam 360")
    for b in blocks:
        if b.get("missing"):
            continue
        c, h = b["centre"], b["half_size_m"]
        ax.add_patch(Rectangle((c["easting_m"] - h - E0, c["northing_m"] - h - N0), 2 * h, 2 * h, fill=False,
                               ec=ORANGE, lw=1.4, zorder=4))
        ax.annotate(str(b["number"]), (c["easting_m"] + h - E0, c["northing_m"] + h - N0), fontsize=label_size,
                    color=ORANGE, fontweight="bold", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel(f"easting − {E0:.0f} m")
    ax.set_ylabel(f"northing − {N0:.0f} m")
    s0, s1 = min(r["sol"] for r in rows), max(r["sol"] for r in rows)
    ax.set_title(f"Perseverance traverse, sols {s0}–{s1}: {sum(not b.get('missing') for b in blocks)} blocks of rover "
                 f"stations (±{blocks[0]['half_size_m']:g} m squares)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200)
    return fig


def zoom_grid(rows: Sequence[Dict[str, Any]], blocks: Sequence[Dict[str, Any]], out_png: Optional[PathLike] = None,
              ncols: int = 5, panel_in: float = 2.6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bl = [b for b in blocks if not b.get("missing")]
    nrows = int(np.ceil(len(bl) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * panel_in, nrows * panel_in * 1.05), squeeze=False)
    for k, ax in enumerate(axs.flat):
        if k >= len(bl):
            ax.axis("off")
            continue
        b = bl[k]
        c, h = b["centre"], b["half_size_m"]
        inside = [r for r in rows if abs(r["E"] - c["easting_m"]) <= h and abs(r["N"] - c["northing_m"]) <= h]
        x = np.array([r["E"] - c["easting_m"] for r in inside])
        y = np.array([r["N"] - c["northing_m"] for r in inside])
        if len(x):
            ax.plot(x, y, "-", color="0.6", lw=0.6)
            eod = np.array([r["end_of_drive"] and r["pano"] for r in inside])
            ax.plot(x[eod], y[eod], "o", ms=3.2, color="C0")
            ax.plot(x[~eod], y[~eod], ".", ms=2.5, color="0.5")
            for r, xx, yy, e in zip(inside, x, y, eod):
                if e:
                    ax.annotate(str(r["sol"]), (xx, yy), fontsize=6.2, xytext=(2, 2), textcoords="offset points")
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_edgecolor(ORANGE)
            s.set_linewidth(1.6)
        ax.set_title(f"{b['number']} {b['name']}\n{_counts(b)}", fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        col = k % ncols
        if col == 0:
            ax.set_ylabel("Relative northing [m]", fontsize=7.5)
        last_in_col = k + ncols >= len(bl)
        if last_in_col:
            ax.set_xlabel("Relative easting [m]", fontsize=7.5)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200)
    return fig
