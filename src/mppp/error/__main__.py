"""
mppp command line.

    python -m mppp.error selftest
    python -m mppp.error demo   [--out DIR]
    python -m mppp.error sweep  --param eps --values 0.25,0.5,1,2
    python -m mppp.error run    --waypoints traverse.geojson --sol 1500 [options]
"""

from __future__ import annotations

import argparse
import sys
import os
import numpy as np


def _add_model_args(ap):
    ap.add_argument("--instrument", choices=["mcz34", "navcam"], default="mcz34")
    ap.add_argument("--eps", type=float, default=0.5, help="eps_intra [px]")
    ap.add_argument("--radius", type=float, default=25.0, help="grid half-width [m]")
    ap.add_argument("--select-radius", type=float, default=None,
                    help="station selection radius [m]; defaults to 2x --radius")
    ap.add_argument("--grid", type=int, default=201, help="grid samples per side")
    ap.add_argument("--cos-e-min", type=float, default=0.0,
                    help="emission-angle cutoff; 0 disables, 0.34 is ~70 deg")
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--pose", choices=["none", "telemetry", "ba"], default="none")
    ap.add_argument("--vo-drift", type=float, default=0.02)
    ap.add_argument("--att-sigma", type=float, default=2.0e-3, help="[rad]")
    # optional, off by default
    ap.add_argument("--theta-gate", type=float, default=None,
                    help="enable matchability gate at this theta_max [deg]")
    ap.add_argument("--tau-gate", type=float, default=None,
                    help="enable transition-tilt gate at this tau_max")
    ap.add_argument("--theta-c", type=float, default=None,
                    help="enable error correlation with this theta_c [deg]")
    ap.add_argument("--rho-inf", type=float, default=0.0)
    ap.add_argument("--coverage", type=float, default=None,
                    help="sigma_n threshold [m] for the coverage metric")
    ap.add_argument("--rho-adjacent", type=float, default=0.0)
    ap.add_argument("--out", default=".")


def _build(args):
    from .core import (ModelConfig, PoseModel, Grid, MASTCAM_Z_34, NAVCAM,
                       Instrument)
    base = MASTCAM_Z_34 if args.instrument == "mcz34" else NAVCAM
    inst = Instrument(name=base.name, ifov_rad=base.ifov_rad,
                      baseline_m=base.baseline_m, eps_intra_px=args.eps)
    cfg = ModelConfig(
        apply_occlusion=not args.no_occlusion,
        cos_e_min=args.cos_e_min,
        use_theta_gate=args.theta_gate is not None,
        theta_max_deg=args.theta_gate or 20.0,
        use_tau_gate=args.tau_gate is not None,
        tau_max=args.tau_gate or 2.0,
        use_correlation=args.theta_c is not None,
        theta_c_deg=args.theta_c or 15.0,
        rho_inf=args.rho_inf,
        store_station_dirs=args.theta_c is not None,
    )
    pm = PoseModel(mode=args.pose, vo_drift_frac=args.vo_drift,
                   att_sigma_rad=args.att_sigma)
    grid = Grid.square(args.radius, args.grid)
    return inst, cfg, pm, grid


def _render(field, args, tag, suptitle):
    import matplotlib
    matplotlib.use("Agg")
    from .metrics import compute_metrics, summarize
    from .plotting import plot_panel
    m = compute_metrics(field, rho_adjacent=args.rho_adjacent,
                        coverage_threshold_m=args.coverage)
    print(summarize(field, m))
    os.makedirs(args.out, exist_ok=True)
    fig, _ = plot_panel(field, m, suptitle=suptitle)
    png = os.path.join(args.out, f"{tag}.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    npz = os.path.join(args.out, f"{tag}.npz")
    np.savez_compressed(npz, x=field.grid.x, y=field.grid.y,
                        **{k: v for k, v in m.items() if isinstance(v, np.ndarray)})
    print(f"\nwrote {png}\nwrote {npz}")
    return m, png, npz


def cmd_selftest(args):
    from .selftest import run_all
    return 0 if run_all() else 1


def cmd_demo(args):
    from .core import Station, solve_precision_field
    inst, cfg, pm, grid = _build(args)
    stations = [
        Station(xyz=[0, 0, 1.9], az_deg=45.0, name="S1 anchor",
                instrument=inst, path_m=0.0, is_anchor=True),
        Station(xyz=[-14, 6, 2.1], az_deg=110.0, name="S2",
                instrument=inst, path_m=16.0),
        Station(xyz=[9, -12, 1.7], az_deg=330.0, name="S3",
                instrument=inst, path_m=34.0),
    ]
    f = solve_precision_field(grid, stations, cfg, pm)
    _render(f, args, "mppp_demo",
            f"Three-station demo  |  {inst.name}  eps={args.eps} px  pose={pm.mode}")
    return 0


def cmd_sweep(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .core import Station, solve_precision_field, ModelConfig, Instrument
    from .metrics import compute_metrics

    vals = [float(v) for v in args.values.split(",")]
    inst, cfg, pm, grid = _build(args)
    rows = []
    for v in vals:
        if args.param == "eps":
            i2 = Instrument(name=inst.name, ifov_rad=inst.ifov_rad,
                            baseline_m=inst.baseline_m, eps_intra_px=v)
            c2 = cfg
        elif args.param == "rho_inf":
            i2 = inst
            c2 = ModelConfig(**{**cfg.__dict__, "use_correlation": True,
                                "rho_inf": v, "store_station_dirs": True})
        elif args.param == "theta_c":
            i2 = inst
            c2 = ModelConfig(**{**cfg.__dict__, "use_correlation": True,
                                "theta_c_deg": v, "store_station_dirs": True})
        else:
            raise SystemExit(f"unknown sweep parameter {args.param}")
        sts = [Station(xyz=[0, 0, 1.9], az_deg=45.0, instrument=i2,
                       path_m=0.0, is_anchor=True),
               Station(xyz=[-14, 6, 2.1], az_deg=110.0, instrument=i2, path_m=16.0),
               Station(xyz=[9, -12, 1.7], az_deg=330.0, instrument=i2, path_m=34.0)]
        m = compute_metrics(solve_precision_field(grid, sts, c2, pm))
        rows.append((v, np.nanmedian(m["sigma_n"]), np.nanmedian(m["G_n"])))
        print(f"  {args.param}={v:<8g} median sigma_n = "
              f"{rows[-1][1]*100:8.3f} cm   median G_n = {rows[-1][2]:8.2f}")

    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    a = np.array(rows)
    ax[0].plot(a[:, 0], a[:, 1] * 100, "o-"); ax[0].set_ylabel("median sigma_n [cm]")
    ax[1].plot(a[:, 0], a[:, 2], "o-"); ax[1].set_ylabel("median G_n [-]")
    for x in ax:
        x.set_xlabel(args.param); x.grid(alpha=0.3)
    fig.suptitle(f"Sweep over {args.param}")
    fig.tight_layout()
    p = os.path.join(args.out, f"mppp_sweep_{args.param}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"\nwrote {p}")
    return 0


def cmd_run(args):
    from .core import solve_precision_field
    from .waypoints import build_stations, site_drive_for_sol, load_featurecollection
    inst, cfg, pm, grid = _build(args)
    fc = load_featurecollection(args.waypoints)
    site, drive, matched = site_drive_for_sol(fc, args.sol)
    if matched != args.sol:
        print(f"note: sol {args.sol} not present; using sol {matched}")
    sel = args.select_radius if args.select_radius is not None else 2.0 * args.radius
    stations, info = build_stations(
        fc, anchor_site=site, anchor_drive=drive, select_radius_m=sel,
        instrument=inst, camera_height_m=args.camera_height,
        flatten_elevation=args.flatten_elevation)
    print(f"anchor {info['anchor_name']}  ({info['n_stations']} stations "
          f"within {sel:.0f} m)")
    for s in stations:
        print(f"    {s.name:<34s} xyz=({s.xyz[0]:7.2f},{s.xyz[1]:7.2f},"
              f"{s.xyz[2]:6.2f})  az={s.az_deg:6.1f}  path={s.path_m:6.1f} m"
              + ("  [anchor]" if s.is_anchor else ""))
    f = solve_precision_field(grid, stations, cfg, pm)
    _render(f, args, f"mppp_sol{matched}",
            f"Sol {matched} | {inst.name} | eps={args.eps} px | pose={pm.mode}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mppp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("selftest", help="run analytic validation")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("demo", help="synthetic three-station example")
    _add_model_args(p); p.set_defaults(func=cmd_demo)

    p = sub.add_parser("sweep", help="sweep a parameter")
    _add_model_args(p)
    p.add_argument("--param", choices=["eps", "rho_inf", "theta_c"], default="eps")
    p.add_argument("--values", default="0.25,0.5,1.0,2.0")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("run", help="run on a real traverse")
    _add_model_args(p)
    p.add_argument("--waypoints", required=True)
    p.add_argument("--sol", type=int, default=-1)
    p.add_argument("--camera-height", type=float, default=1.9)
    p.add_argument("--flatten-elevation", action="store_true",
                   help="reproduce v1 behaviour (discards station elevations)")
    p.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
