"""
[Research script, not part of the installed package (v0p13). Run from the repository:
 python studies/error/study_cross_station.py --help]

mppp.study_cross_station -- how much does cross-station correspondence buy?

THE QUESTION
============
Operational Mars stereo triangulates on a FIXED baseline: 42.4 cm for Navcam,
24.4 cm for Mastcam-Z.  Range error scales as r^2/b, so at 20 m a Navcam pair
is working with a parallactic angle of about 1.2 degrees.

SfM offers a completely different baseline: the station separation, typically
5-30 m.  That is 15-70x larger, and range error scales as 1/b, so the
theoretical improvement is enormous.

But the theoretical gain is only realised if cross-station matching actually
works.  Two views of the same outcrop from 15 m apart differ in view angle, in
scale, in foreshortening and in illumination.  If the matcher fails, the
stations cannot be tied together and you are back to fixed-baseline stereo.

So the question is not "is SfM better" -- it obviously is in principle -- but
"how much of the principle survives contact with Mars terrain", and that
reduces to two numbers: eps_cross and theta_max.

THE EXPERIMENT
==============
Three cases over identical Navcam station geometry:

  OPS         no cross-station correspondence.  Each cell falls back to its
              best single station.  The operational baseline.
  OPTIMISTIC  eps_cross = 0.6 px, theta_max = 45 deg.  A learned dense matcher
              (LoFTR/DKM/RoMa class) coping well with wide baselines.
  PESSIMISTIC eps_cross = 2.0 px, theta_max = 12 deg.  Dense NCC/SGM on
              low-texture, differently-illuminated regolith.

The spread between OPTIMISTIC and PESSIMISTIC brackets the answer.  Real data
then tells you which end you are at -- and the same measurement backs out
eps_cross and theta_max directly (see mppp_error.colmap).

Run:  python -m mppp_error.study_cross_station --out DIR
"""

from __future__ import annotations

import argparse
import os
import numpy as np

from mppp.error.core import (Grid, Station, Instrument, NAVCAM, ModelConfig, PoseModel,
                   solve_precision_field)
from mppp.error.metrics import compute_metrics
from mppp.error.viewgraph import build_view_graph, graph_report

__all__ = ["build_case_stations", "run_cases", "CASES"]


#: name -> (eps_cross_px, theta_max_deg, tau_max, description)
CASES = {
    "optimistic": (0.6, 45.0, 3.0,
                   "learned dense matcher, wide-baseline tolerant"),
    "pessimistic": (2.0, 12.0, 1.4,
                    "dense NCC/SGM, low texture, illumination change"),
}


def build_case_stations(eps_px: float = 0.5, n_stations: int = 5,
                        spacing_m: float = 8.0, seed: int = 3):
    """
    Five Navcam stations on a realistic drive: broadly linear with lateral
    wander, which is what a rover traverse actually looks like and is close to
    the collinear degeneracy that makes this question interesting.
    """
    rng = np.random.default_rng(seed)
    inst = Instrument(name=NAVCAM.name, ifov_rad=NAVCAM.ifov_rad,
                      baseline_m=NAVCAM.baseline_m, eps_intra_px=eps_px)
    sts, pos, path = [], np.array([0.0, 0.0]), 0.0
    heading = 35.0
    for k in range(n_stations):
        if k:
            heading += rng.normal(0, 25)
            step = spacing_m * rng.uniform(0.75, 1.25)
            pos = pos + step * np.array([np.sin(np.radians(heading)),
                                         np.cos(np.radians(heading))])
            path += step
        sts.append(Station(
            xyz=[pos[0], pos[1], 1.9 + rng.normal(0, 0.12)],
            az_deg=float((heading + rng.normal(0, 30)) % 360),
            name=f"S{k+1}", instrument=inst, path_m=path, is_anchor=(k == 0)))
    return sts


def run_cases(stations, grid, pose_mode: str = "ba"):
    """Solve OPS + both bracketing cases on identical geometry."""
    pm = PoseModel(mode=pose_mode)
    out = {}

    f = solve_precision_field(grid, stations,
                              ModelConfig(link_mode="ops",
                                          store_station_dirs=True), pm)
    out["ops"] = (f, compute_metrics(f))

    for name, (eps_c, th, tau, _) in CASES.items():
        cfg = ModelConfig(link_mode="gated", use_theta_gate=True,
                          use_tau_gate=True, eps_cross_px=eps_c,
                          theta_max_deg=th, tau_max=tau,
                          store_station_dirs=True)
        f = solve_precision_field(grid, stations, cfg, pm)
        out[name] = (f, compute_metrics(f))

    f = solve_precision_field(grid, stations,
                              ModelConfig(link_mode="full",
                                          store_station_dirs=True), pm)
    out["ideal"] = (f, compute_metrics(f))
    return out


# --------------------------------------------------------------------------

def _fmt(a, scale=1.0):
    v = a[np.isfinite(a)]
    return "n/a" if v.size == 0 else f"{np.median(v)*scale:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".")
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=18.0,
                    help="grid margin beyond the station footprint [m]")
    ap.add_argument("--grid", type=int, default=221)
    ap.add_argument("--stations", type=int, default=5)
    ap.add_argument("--spacing", type=float, default=8.0)
    ap.add_argument("--pose", default="ba", choices=["none", "ba", "telemetry"])
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    stations = build_case_stations(args.eps, args.stations, args.spacing)
    grid = Grid.covering(stations, margin_m=args.margin, n=args.grid)
    res = run_cases(stations, grid, args.pose)

    ops_s = res["ops"][1]["sigma_n"]
    print("=" * 78)
    print("HOW MUCH DOES CROSS-STATION CORRESPONDENCE BUY?")
    span = np.max(np.ptp(np.array([s.xyz[:2] for s in stations]), axis=0))
    print(f"Navcam (b={NAVCAM.baseline_m} m, psi={NAVCAM.ifov_rad*1e6:.0f} urad), "
          f"eps_intra={args.eps} px, {args.stations} stations, pose={args.pose}")
    print(f"station footprint {span:.1f} m across; "
          f"grid {grid.shape[0]}x{grid.shape[1]} over "
          f"{grid.x[0]:.0f}..{grid.x[-1]:.0f} m")
    print("=" * 78)
    print(f"{'case':<13} {'eps_x':>6} {'th_max':>7} {'sig_n[cm]':>10} "
          f"{'G_n':>7} {'x vs OPS':>9} {'kappa':>8} {'cover<2cm':>10}")
    print("-" * 78)
    rows = {}
    for key in ("ops", "pessimistic", "optimistic", "ideal"):
        f, m = res[key]
        s = m["sigma_n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            imp = np.where(np.isfinite(s) & (s > 0), ops_s / s, np.nan)
        cov = np.nanmean((s < 0.02).astype(float))
        ec, th = ("--", "--")
        if key in CASES:
            ec, th = f"{CASES[key][0]:.1f}", f"{CASES[key][1]:.0f}"
        print(f"{key:<13} {ec:>6} {th:>7} {_fmt(s,100):>10} {_fmt(m['G_n']):>7} "
              f"{_fmt(imp):>9} {_fmt(m['kappa']):>8} {cov:>10.3f}")
        rows[key] = (s, imp, m)
    print("-" * 78)

    med = {k: np.nanmedian(rows[k][1]) for k in ("pessimistic", "optimistic", "ideal")}
    print(f"\nBRACKET: cross-station correspondence improves median sigma_n by")
    print(f"         {med['pessimistic']:.1f}x (pessimistic) to "
          f"{med['optimistic']:.1f}x (optimistic);"
          f"  {med['ideal']:.1f}x if matching were perfect.")
    print(f"         The factor of {med['optimistic']/max(med['pessimistic'],1e-9):.1f} "
          "between them is what real data has to resolve.")

    # ---- view graph -------------------------------------------------------
    print("\n" + "=" * 78)
    for key in ("optimistic", "pessimistic"):
        print(f"PREDICTED VIEW GRAPH -- {key} ({CASES[key][3]})")
        print("=" * 78)
        f_case = res[key][0]
        print(graph_report(build_view_graph(f_case, f_case.config)))
        print()

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))
    from mppp.error.plotting import plot_map
    order = ["ops", "pessimistic", "optimistic"]
    titles = {"ops": "OPS: fixed-baseline stereo only\n(no cross-station match)",
              "pessimistic": "PESSIMISTIC SfM\neps_cross=2.0 px, theta_max=12 deg",
              "optimistic": "OPTIMISTIC SfM\neps_cross=0.6 px, theta_max=45 deg"}
    lo = min(np.nanpercentile(rows[k][0][np.isfinite(rows[k][0])] * 100, 1)
             for k in order)
    hi = np.nanpercentile(rows["ops"][0][np.isfinite(rows["ops"][0])] * 100, 99)
    for c, k in enumerate(order):
        plot_map(res[k][0], rows[k][0], title=titles[k],
                 cbar_label="sigma_n [cm]", scale=100.0, cmap="viridis_r",
                 log=True, vmin=lo, vmax=hi, ax=axes[0][c], labels=(c == 0))
    axes[1][0].axis("off")
    for c, k in enumerate(["pessimistic", "optimistic"]):
        plot_map(res[k][0], rows[k][1],
                 title=f"Improvement over OPS -- {k}", cbar_label="factor [x]",
                 cmap="magma", log=True, vmin=1.0,
                 vmax=np.nanpercentile(rows["optimistic"][1], 99),
                 ax=axes[1][c + 1], labels=False)
    fig.suptitle("How much does cross-station correspondence buy? "
                 f"Navcam, {args.stations} stations, pose={args.pose}",
                 fontsize=14)
    fig.tight_layout()
    p1 = os.path.join(args.out, "cross_station_comparison.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight")

    # ---- sensitivity surface ---------------------------------------------
    eps_grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    th_grid = np.array([8.0, 12.0, 18.0, 25.0, 35.0, 45.0, 60.0])
    Z = np.zeros((eps_grid.size, th_grid.size))
    g_small = Grid.covering(stations, margin_m=args.margin, n=81)
    ops_small = compute_metrics(solve_precision_field(
        g_small, stations, ModelConfig(link_mode="ops"),
        PoseModel(mode=args.pose)))["sigma_n"]
    for a, ec in enumerate(eps_grid):
        for b, th in enumerate(th_grid):
            m = compute_metrics(solve_precision_field(
                g_small, stations,
                ModelConfig(link_mode="gated", use_theta_gate=True,
                            use_tau_gate=True, eps_cross_px=ec,
                            theta_max_deg=th, tau_max=2.0,
                            store_station_dirs=True),
                PoseModel(mode=args.pose)))
            Z[a, b] = np.nanmedian(ops_small / m["sigma_n"])

    fig2, ax = plt.subplots(figsize=(8.2, 5.6))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="magma",
                   extent=(th_grid[0], th_grid[-1], eps_grid[0], eps_grid[-1]))
    cs = ax.contour(th_grid, eps_grid, Z, levels=[2, 5, 10, 20, 40],
                    colors="w", linewidths=0.9)
    ax.clabel(cs, fmt="%gx", fontsize=9)
    for k, (ec, th, _, _) in CASES.items():
        ax.plot(th, ec, "o", ms=11, mfc="none", mec="cyan", mew=2.2)
        ax.annotate(k, (th, ec), color="cyan", fontsize=10,
                    xytext=(8, 6), textcoords="offset points")
    fig2.colorbar(im, ax=ax, label="median improvement over OPS [x]")
    ax.set_xlabel("theta_max [deg]   (matcher wide-baseline tolerance)")
    ax.set_ylabel("eps_cross [px]   (cross-station matching precision)")
    ax.set_title("Realised SfM gain vs. the two unknown matching parameters\n"
                 "Navcam; the whole result lives on this surface")
    fig2.tight_layout()
    p2 = os.path.join(args.out, "cross_station_sensitivity.png")
    fig2.savefig(p2, dpi=130, bbox_inches="tight")

    print(f"wrote {p1}\nwrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
