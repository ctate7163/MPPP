"""
[Research script, not part of the installed package (v0p13). Run from the repository:
 python studies/error/study_navcam.py --help]

mppp_error.study_navcam -- the four regimes on a real Navcam station geometry.

    python -m mppp_error.study_navcam --out DIR [--spacing 3] [--instrument navcam|mcz34]

WHAT THIS ANSWERS
=================
Operational Mars stereo triangulates on a FIXED baseline (42.4 cm Navcam,
24.4 cm MCZ).  SfM+MVS instead uses the station separation, 1-40 m, which is
3-100x larger.  How much of that theoretical gain survives on Mars terrain?

The answer turns out to depend on station spacing far more than on the matcher,
and that has a direct experimental consequence -- see the spacing sweep.

KEY FINDING FROM THE SWEEP
--------------------------
Optimistic improvement over FIXED peaks near ~10 m station spacing and DECLINES beyond
it: wider stations triangulate better but overlap less and match worse.  There
is an optimum.

Meanwhile the BRACKET -- how much the answer depends on the unmeasured matching
parameters -- grows with spacing out to ~70 m (1.9x at 1.5 m, ~6.0x at 70 m),
then NARROWS SLIGHTLY by 100 m (~5.5x).  Both cases degrade toward FIXED as
spacing keeps growing (matching gets harder for everyone), so the ratio
between them eventually stops widening too.  Extending the sweep to 100 m
(from an earlier 40 m version) is what surfaced this -- the "monotonically"
claim in an earlier version of this docstring was true only over the range
then tested.  So:

  * for RECONSTRUCTION QUALITY, ~10 m spacing is close to optimal;
  * for CALIBRATING eps_cross and theta_max from real data, you want WIDELY
    separated stations, but not unboundedly so -- the bracket peaks around
    ~70 m for this geometry and instrument, not at the sweep's far end.

Those are different experiments and should not be run on the same stations.

A NOTE ON eps_cross = 0
-----------------------
w_ij = (eps_intra/eps_cross)^2 is clipped at 1, so eps_cross BELOW eps_intra
changes nothing.  That is correct: eps is a measurement precision in image
space, and a cross-station match cannot be more precise than the images
themselves.  eps_cross = eps_intra IS the perfect-matching case.
"""

from __future__ import annotations

import argparse
import os
import numpy as np

from mppp.error.core import (Grid, Station, Instrument, NAVCAM, MASTCAM_Z_34,
                   ModelConfig, PoseModel, choose_anchor)
from mppp.error.sitemap import load_site_image, crop_to_extent, trim_border_fraction
from mppp.error.cases import run_four_cases, case_table, CASE_PARAMS
from mppp.error.metrics import compute_metrics
from mppp.error.network import pose_covariance_from_network, network_report
from mppp.error.waypoints import stations_from_rmcs

#: Bundled real M2020 data (mppp/data/).  DEFAULT_RMCS is the landing-site
#: subset picked from mppp/data/M20_waypoints.json: RMC "3_0" is the site-3
#: frame origin itself (sol 13, drive 0, "Site increment, no motion" -- the
#: landing site frame, localized at sol 13 from imagery spanning sols 3-11,
#: matching the bundled Butler Landing panorama's date range), plus three
#: nearby early-mission waypoints chosen for the best available azimuthal
#: spread within about 20 m.
#:
#: NOTE ON REALISM: unlike the hand-placed synthetic pattern used by the
#: spacing sweep, these are REAL waypoints from an actual drive, and the real
#: azimuthal spread is poor -- all four fall within a 0-146 degree arc, none
#: on the far side. Early-mission drives near the landing site were checkout
#: traverses, not chosen for photogrammetric coverage. This is itself an
#: honest illustration of the paper's premise: real acquisitions are usually
#: far from the geometry a network-design tool would recommend.
_HERE = os.path.dirname(os.path.abspath(__file__))
from mppp.waypoints import snapshot_path as _snapshot  # noqa: E402
DEFAULT_WAYPOINTS = str(_snapshot())                  # the packaged, frozen waypoint snapshot
DEFAULT_SITE_IMAGE = os.path.join(_HERE, "data", "butler_landing_1_vertical_50m.jpg")   # not distributed
DEFAULT_RMCS = ["3_0", "3_110", "3_1266", "3_1398"]

#: Standard error-metric choices for the four-panel figure.
#: key -> (display label for suptitle/filename slug, colour-bar unit,
#:         display scale factor, colormap, log-scale, stats_box unit suffix)
#: All four are "lower is better", so the same ref/value improvement ratio
#: formulation applies uniformly across every entry here.
ERROR_METRICS = {
    "sigma_n":     ("vertical (normal) precision", "sigma_n [cm]", 100.0,
                    "viridis_r", True, " cm"),
    "kappa":       ("anisotropy (sigma_max / sigma_min)", "kappa [-]", 1.0,
                    "inferno_r", True, "x"),
    "sigma_slope": ("slope (differential) error", "slope error [%]", 100.0,
                    "plasma_r", True, "%"),
    "G_n":         ("normalised precision G_n", "G_n [-]", 1.0,
                    "magma_r", True, ""),
    "sigma_t_max": ("lateral (tangent-plane) precision", "sigma_t [cm]", 100.0,
                    "viridis_r", True, " cm"),
}

from mppp.error.viewgraph import build_view_graph, graph_report

__all__ = ["make_stations", "spacing_sweep"]

#: unit-scale station pattern; multiplied by `spacing`
PATTERN = [(0.0, 0.0), (1.0, 0.25), (0.55, 0.95), (-0.4, 0.7)]


def make_stations(spacing_m: float = 4.0, n: int = 4, eps_px: float = 0.5,
                  instrument: str = "navcam", jitter: float = 0.0,
                  seed: int = 0, anchor: str = "first") -> list:
    """Hand-placed test stations, `n` of them, scaled by `spacing_m`."""
    base = MASTCAM_Z_34 if instrument == "mcz34" else NAVCAM
    inst = Instrument(name=base.name, ifov_rad=base.ifov_rad,
                      baseline_m=base.baseline_m, eps_intra_px=eps_px)
    rng = np.random.default_rng(seed)
    sts, path = [], 0.0
    pat = PATTERN[:n]
    for k, (x, y) in enumerate(pat):
        p = np.array([x, y]) * spacing_m
        if jitter:
            p = p + rng.normal(0, jitter, 2)
        if k:
            path += float(np.linalg.norm(p - np.array(pat[k-1]) * spacing_m))
        sts.append(Station(xyz=[p[0], p[1], 1.9], az_deg=(35 + 95*k) % 360,
                           name=f"W{k+1}", instrument=inst, path_m=path))
    choose_anchor(sts, anchor)
    return sts


def spacing_sweep(spacings=None,
                  eps_px: float = 0.5, instrument: str = "navcam",
                  n_grid: int = 121, margin_m: float = 50.0,
                  half_width: float = 0.0):
    """
    Improvement and bracket vs. station spacing.  Returns a structured array.

    Default spacings START AT the instrument's own stereo baseline (not an
    arbitrary 1.5 m) and run to 100 m, so the sweep actually shows the point
    where "two independent stations b_inst apart" starts to add anything over
    the instrument's own fixed baseline -- see the marker in the figure -- and
    covers the range out to a full order of magnitude beyond the ~10 m
    reconstruction optimum, where the calibration bracket keeps widening.
    """
    if spacings is None:
        base = MASTCAM_Z_34 if instrument == "mcz34" else NAVCAM
        b = base.baseline_m
        spacings = tuple(sorted({round(b, 3), 0.75, 1.5, 3.0, 5.0, 10.0,
                                 20.0, 40.0, 70.0, 100.0}))
    rows = []
    for sp in spacings:
        sts = make_stations(sp, eps_px=eps_px, instrument=instrument)
        span = float(np.max(np.ptp(np.array([s.xyz[:2] for s in sts]), axis=0)))
        # the sweep auto-sizes regardless of half_width: a fixed window would
        # clip the wide-spacing cases and confound spacing with map extent
        g = Grid.covering(sts, margin_m=margin_m, n=n_grid)
        r = run_four_cases(sts, g, include_full=False)
        sf = np.nanmedian(r["fixed"][1]["sigma_n"])
        sp_ = np.nanmedian(r["pessimistic"][1]["sigma_n"])
        so = np.nanmedian(r["optimistic"][1]["sigma_n"])
        si = np.nanmedian(r["ideal"][1]["sigma_n"])
        rows.append((sp, span, sf, sp_, so, si, sf/sp_, sf/so, (sf/so)/(sf/sp_)))
    return np.array(rows, dtype=[
        ("spacing", "f8"), ("span", "f8"), ("fixed", "f8"), ("pessimistic", "f8"),
        ("optimistic", "f8"), ("ideal", "f8"), ("improvement_pess", "f8"),
        ("improvement_opt", "f8"), ("bracket", "f8")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".")
    ap.add_argument("--instrument", choices=["navcam", "mcz34"], default="navcam")
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--spacing", type=float, default=4.0)
    ap.add_argument("--stations", type=int, default=4)
    ap.add_argument("--margin", type=float, default=50.0,
                    help="used only when --half-width is 0 (auto-size to stations)")
    ap.add_argument("--half-width", type=float, default=25.0,
                    help="fixed map half-extent [m]; 0 = auto-size to the stations")
    ap.add_argument("--grid", type=int, default=201)
    ap.add_argument("--pose", default="network",
                    choices=["none", "network", "telemetry", "deadreckon"])
    ap.add_argument("--tie-correlation", type=float, default=0.0)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--anchor", default="first",
                    choices=["first", "center", "site"])
    ap.add_argument("--site-image", default=None,
                    help="path or URL to an overhead site mosaic to show in "
                         "place of the FIXED improvement panel (which is "
                         "otherwise blank, since FIXED has no improvement "
                         "over itself)")
    ap.add_argument("--site-image-width-m", type=float, default=50.0,
                    help="real-world width/height of the supplied image [m], "
                         "assumed centred on the anchor and north-up")
    ap.add_argument("--no-site-image", action="store_true",
                    help="disable the default bundled site image")
    ap.add_argument("--waypoints", default=None,
                    help="path to a real M2020 waypoint GeoJSON; defaults to "
                         "the packaged mppp/data/M20_waypoints.json")
    ap.add_argument("--rmcs", default=None,
                    help="comma-separated 'site_drive' RMC list to use as real "
                         "stations, e.g. '3_0,3_110,3_1266,3_1398'; defaults to "
                         "DEFAULT_RMCS (a landing-site subset). First RMC is "
                         "the anchor.")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the hand-placed synthetic pattern (make_stations) "
                         "instead of real waypoints for the main figure")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--rover-icon", action="store_true",
                    help="draw an abstract top-down rover glyph (chassis + "
                         "wheels + mast circle) at each station instead of "
                         "the plain rotated-square marker")
    ap.add_argument("--rover-scale", type=float, default=1.0,
                    help="size multiplier for --rover-icon (metres-ish at "
                         "scale=1.0; tune for map extent)")
    ap.add_argument("--site-image-trim-frac", type=float, default=0.25,
                    help="fraction trimmed off EACH side of the raw site "
                         "image before it is treated as the nominal "
                         "--site-image-width-m footprint; see "
                         "mppp_error.sitemap.trim_border_fraction")
    ap.add_argument("--metrics", default="sigma_n,kappa,sigma_slope,G_n",
                    help="comma-separated metric keys to render, one figure "
                         "each; see ERROR_METRICS for the available set")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mppp.error.plotting import plot_map
    os.makedirs(args.out, exist_ok=True)

    if args.synthetic:
        sts = make_stations(args.spacing, args.stations, args.eps, args.instrument,
                            anchor=args.anchor)
        geom_note = f"{args.stations} synthetic stations, {args.spacing:.1f} m spacing"
    else:
        wp_path = args.waypoints or DEFAULT_WAYPOINTS
        rmcs = args.rmcs.split(",") if args.rmcs else DEFAULT_RMCS
        base_inst = MASTCAM_Z_34 if args.instrument == "mcz34" else NAVCAM
        inst0 = Instrument(name=base_inst.name, ifov_rad=base_inst.ifov_rad,
                           baseline_m=base_inst.baseline_m, eps_intra_px=args.eps)
        sts, wp_info = stations_from_rmcs(wp_path, rmcs, instrument=inst0)
        geom_note = f"{len(sts)} REAL waypoints from {os.path.basename(wp_path)}"
    span = float(np.max(np.ptp(np.array([s.xyz[:2] for s in sts]), axis=0)))
    anchor = next(s.name for s in sts if s.is_anchor)
    grid = (Grid.square(args.half_width, args.grid) if args.half_width > 0
            else Grid.covering(sts, margin_m=args.margin, n=args.grid))
    inst = sts[0].instrument

    print("=" * 76)
    print(f"{geom_note} ({span:.1f} m footprint), anchor={anchor}")
    print(f"b={inst.baseline_m} m  psi={inst.ifov_rad*1e6:.0f} urad  "
          f"eps_intra={args.eps} px  pose={args.pose}")
    print(f"grid {grid.shape[0]}x{grid.shape[1]} spanning "
          f"{grid.x[0]:.0f}..{grid.x[-1]:.0f} m")
    if not args.synthetic:
        print("station RMCs and real path_m (from actual rover odometry, "
              "dist_total_m -- NOT straight-line distance):")
        for s in sts:
            print(f"    {s.name:<20s} xyz=({s.xyz[0]:6.2f},{s.xyz[1]:6.2f},"
                  f"{s.xyz[2]:5.2f})  path_m={s.path_m:7.2f}"
                  + ("  [anchor]" if s.is_anchor else ""))
    print("=" * 76)

    if not args.no_site_image and args.site_image is None \
            and not args.synthetic and os.path.exists(DEFAULT_SITE_IMAGE):
        args.site_image = DEFAULT_SITE_IMAGE

    # ---- pose: derived from the network, not assumed ---------------------
    if args.pose == "network":
        net = pose_covariance_from_network(sts, grid, n_tie=50,
                                           tie_correlation=args.tie_correlation)
        rel = net.relative_to(anchor)
        print("\nBA POSE DERIVED FROM THE NETWORK")
        print("(lower bound: grid cells treated as independent tie points)")
        print(network_report(rel))
        pm = rel.to_pose_model()
    else:
        pm = PoseModel(mode="none" if args.pose == "none" else args.pose)

    res = run_four_cases(sts, grid, pose=pm)
    print("\n" + case_table(res) + "\n")

    fb = getattr(res["pessimistic"][0], "fallback_taken", None)
    if fb is not None and fb.any():
        print(f"note: {100*fb.mean():.1f}% of cells fell back to fixed-baseline "
              "stereo (well measured, poorly registered)\n")

    print("PREDICTED VIEW GRAPH (optimistic)")
    f_o = res["optimistic"][0]
    print(graph_report(build_view_graph(f_o, f_o.config)))

    # ---- sol/site/drive of the primary (anchor) location -------------------
    if args.synthetic:
        sol_str, site_str, drive_str = "NA", "NA", "NA"
    else:
        sol_str = str(wp_info["anchor_sol"])
        site_str = str(wp_info["anchor_site"])
        drive_str = str(wp_info["anchor_drive"])

    # ---- pre-crop the site image once, shared by every rendered figure -----
    site_img_cropped = None
    if args.site_image:
        try:
            raw = load_site_image(args.site_image)
            trimmed = trim_border_fraction(raw, args.site_image_trim_frac)
            site_img_cropped = crop_to_extent(trimmed, args.site_image_width_m,
                                              grid.extent_edges())
        except Exception as e:                                    # noqa: BLE001
            print(f"note: site image unavailable ({e})")

    order = ["fixed", "pessimistic", "optimistic", "ideal"]
    from mppp.error.plotting import _station_markers

    def render_metric_figure(metric_key: str) -> str:
        """Build and save the four-panel figure for one error metric."""
        label, cbar_unit, scale, cmap, log, stats_unit = ERROR_METRICS[metric_key]
        ref = res["fixed"][1][metric_key]

        fin = [res[k][1][metric_key][np.isfinite(res[k][1][metric_key])]
              for k in order]
        fin = [v for v in fin if v.size]
        lo = min(np.percentile(v*scale, 1) for v in fin)
        hi = max(np.percentile(v*scale, 99) for v in fin)

        imps = {}
        for k in order:
            val = res[k][1][metric_key]
            with np.errstate(invalid="ignore", divide="ignore"):
                imps[k] = np.where(np.isfinite(val) & (val > 0), ref / val, np.nan)
        ilo = 1.0
        ihi_candidates = [np.nanpercentile(v, 99.5) for v in imps.values()
                          if np.isfinite(v).any()]
        ihi = max(ihi_candidates) if ihi_candidates else 2.0

        # Narrower than the original 21-inch-wide layout, and colour bars are
        # suppressed on every column except the last ("inner" bars removed) --
        # both free up width, which is where the narrowing mostly comes from.
        fig, ax = plt.subplots(2, 4, figsize=(18.0, 9.2))
        titles = {
            "fixed": f"FIXED\nfixed-baseline stereo only, eps_i={args.eps} px",
            "pessimistic": f"PESSIMISTIC SfM+MVS\neps_i={args.eps} px, "
                           f"eps_x=eps_i, "
                           f"th_max={CASE_PARAMS['pessimistic'][1]:.0f} deg",
            "optimistic": f"OPTIMISTIC SfM+MVS\neps_i={args.eps} px, "
                          f"eps_x=eps_i, "
                          f"th_max={CASE_PARAMS['optimistic'][1]:.0f} deg",
            "ideal": f"IDEAL\nbest of ALL pairs, combined, eps_i={args.eps} px",
        }
        for c, k in enumerate(order):
            last_col = (c == len(order) - 1)
            # Axis labels only on the outer edge of the grid: bottom row gets
            # an x-label, left column gets a y-label -- not every panel.
            plot_map(res[k][0], res[k][1][metric_key], title=titles[k],
                     cbar_label=f"{cbar_unit}  (shared scale)", scale=scale,
                     cmap=cmap, log=log, vmin=lo, vmax=hi,
                     ax=ax[0][c], labels=(c == 0),
                     show_xlabel=False, show_xticklabels=False,
                     show_ylabel=(c == 0), show_yticklabels=(c == 0),
                     stats_box=True, stats_unit=stats_unit,
                     show_colorbar=last_col, rover_icon=args.rover_icon,
                     rover_scale=args.rover_scale)
            if k == "fixed" and site_img_cropped is not None:
                # FIXED has no improvement over itself, so that panel is
                # otherwise blank -- put the overhead site mosaic there
                # instead, cropped (and pre-trimmed) to the same map extent.
                title_suffix = (" (padded, exceeds source coverage)"
                                if site_img_cropped.partial_coverage else "")
                ax[1][c].imshow(site_img_cropped, extent=grid.extent_edges(),
                               origin="upper")
                ax[1][c].set_title(f"site context (overhead mosaic){title_suffix}",
                                   fontsize=11)
                ax[1][c].set_ylabel("Relative northing [m]")   # left column
                ax[1][c].set_xlabel("Relative easting [m]")    # bottom row
                _station_markers(ax[1][c], res[k][0], labels=True,
                                 rover_icon=args.rover_icon,
                                 rover_scale=args.rover_scale)
            else:
                plot_map(res[k][0], imps[k],
                         title=f"improvement over FIXED -- {k.upper()}",
                         cbar_label="factor [x]  (shared scale)", cmap="magma",
                         log=True, vmin=ilo, vmax=ihi, ax=ax[1][c],
                         labels=True,   # sol numbers on the bottom row
                         show_xlabel=True, show_xticklabels=True,
                         show_ylabel=(c == 0), show_yticklabels=(c == 0),
                         stats_box=True, stats_unit="x",
                         show_colorbar=last_col, rover_icon=args.rover_icon,
                         rover_scale=args.rover_scale)

        fig.suptitle(
            f"Waypoints of Perseverance with estimated reconstructed error: "
            f"{label}\nPrimary location: Sol {sol_str} | Site {site_str} | "
            f"Drive {drive_str}", fontsize=14)
        fig.tight_layout()
        fname = f"Mars2020_recon_error_Sol_{sol_str}_{metric_key}.png"
        path = os.path.join(args.out, fname)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    metric_keys = [k.strip() for k in args.metrics.split(",") if k.strip()]
    unknown = [k for k in metric_keys if k not in ERROR_METRICS]
    if unknown:
        print(f"note: ignoring unknown metric key(s) {unknown}; available: "
              f"{sorted(ERROR_METRICS)}")
    metric_keys = [k for k in metric_keys if k in ERROR_METRICS]
    for mk in metric_keys:
        p1 = render_metric_figure(mk)
        print(f"wrote {p1}")

    # ---- spacing sweep ----------------------------------------------------
    if not args.no_sweep:
        sw = spacing_sweep(eps_px=args.eps, instrument=args.instrument,
                           margin_m=args.margin, half_width=args.half_width)
        print("\n" + "=" * 76)
        print("SPACING SWEEP -- where does matcher quality actually matter?")
        print("=" * 76)
        print(f"{'spacing':>8} {'span':>7} {'FIXED':>8} {'PESS':>8} {'OPT':>8} "
              f"{'pess x':>7} {'opt x':>7} {'bracket':>8}")
        for r in sw:
            print(f"{r['spacing']:8.1f} {r['span']:7.1f} {r['fixed']*100:8.3f} "
                  f"{r['pessimistic']*100:8.3f} {r['optimistic']*100:8.3f} "
                  f"{r['improvement_pess']:7.2f} {r['improvement_opt']:7.2f} {r['bracket']:8.2f}")
        kbest = int(np.argmax(sw["improvement_opt"]))
        print(f"\noptimum spacing for reconstruction quality : "
              f"{sw['spacing'][kbest]:.0f} m  ({sw['improvement_opt'][kbest]:.1f}x over FIXED)")
        print(f"best spacing for CALIBRATING the parameters: "
              f"{sw['spacing'][-1]:.0f} m  (bracket {sw['bracket'][-1]:.1f}x)")
        print("These are different experiments; do not run them on the same stations.")

        # "Second subplot on the bottom": stacked 2x1 rather than side-by-side
        # 1x2. Both axes still share the log-x station-spacing axis.
        fig2, axs = plt.subplots(2, 1, figsize=(8.2, 9.0))
        axs[0].plot(sw["spacing"], sw["improvement_opt"], "o-", label="optimistic", lw=2)
        axs[0].plot(sw["spacing"], sw["improvement_pess"], "s-", label="pessimistic", lw=2)
        axs[0].axvline(sw["spacing"][kbest], color="0.6", ls="--", lw=1)
        axs[0].annotate(f"optimum ~{sw['spacing'][kbest]:.0f} m",
                        (sw["spacing"][kbest], sw["improvement_opt"][kbest]),
                        xytext=(8, -18), textcoords="offset points", fontsize=9)
        axs[0].set_ylabel("median improvement over FIXED [x]")
        axs[0].legend(); axs[0].set_title("Reconstruction improvement peaks, then declines")
        axs[1].plot(sw["spacing"], sw["bracket"], "d-", color="crimson", lw=2)
        axs[1].set_ylabel("optimistic improvement / pessimistic improvement")
        axs[1].set_title("Sensitivity to the unmeasured matching parameters\n"
                         "grows monotonically with spacing")

        # set_xscale MUST happen before set_xlim: matplotlib autoscales the
        # view when the scale changes, which silently discards any xlim set
        # beforehand and collapses the whole plot to a sliver (this happened
        # here on the first attempt).
        for a in axs:
            a.set_xscale("log"); a.set_xlabel("station spacing [m]"); a.grid(alpha=.3)

        # Mark the FIXED-baseline point: spacing = the instrument's own stereo
        # baseline.  At exactly that separation, two independent stations
        # sitting b_inst apart give the SAME triangulation geometry as one
        # station's own stereo pair -- this is the point where "moving the
        # second eye out to a second waypoint" starts to pay off.  The sweep's
        # default spacings START at b_inst so this marker actually falls
        # inside the plotted range.
        b_inst = inst.baseline_m
        for a in axs:
            xlo, xhi = a.get_xlim()
            a.set_xlim(min(xlo, b_inst * 0.85), xhi)
            a.axvline(b_inst, color="tab:green", ls=":", lw=1.8, zorder=5)
        axs[0].annotate(f"{inst.name} stereo baseline\n({b_inst:.2f} m)",
                        (b_inst, axs[0].get_ylim()[1]), xytext=(6, -6),
                        textcoords="offset points", fontsize=8.5,
                        color="tab:green", va="top")

        fig2.suptitle(f"{inst.name}: where matcher quality matters", fontsize=13)
        fig2.tight_layout()
        p2 = os.path.join(args.out, "navcam_spacing_sweep.png")
        fig2.savefig(p2, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
