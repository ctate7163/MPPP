"""
Notebook 05's measurements for many alignments, one alignment in memory at a time (v0p31).  The notebook keeps every
loaded model at once, which needs about 0.3 GB per Navcam alignment; this script keeps only the small per-alignment
results and the gate trials, so fifteen or more alignments run in a few GB.

Per alignment: epsilon (intra/cross, by range and by angle), the cross-station gate (sampled pairs, ML fit), the
decorrelation rho(theta), the view graph summary, the registration against the priors and the parameter row of
``mppp.error.alignment.parameters``.  Then the gate pooled over all alignments and the angle-form comparison.

Usage:
  python scripts/error_analysis_batch.py OUT_DIR NAME=PATH [NAME=PATH ...] [--min-track 3] [--n-points 5000]
         [--n-boot 50] [--form power] [--illumination dlmst] [--theta-max 30]
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _j(o):
    return o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray) else str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("alignments", nargs="+", help="NAME=PATH")
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--form", default="power")
    ap.add_argument("--illumination", default="dlmst")
    ap.add_argument("--theta-max", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from mppp.error import alignment as EA
    from mppp.error.viewgraph import graph_report
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"settings": vars(a), "alignments": {}, "eps": [], "eps_by_angle": [], "eps_by_range": {}, "gate_fits": [],
           "rho": {}, "rho_fits": [], "registration": [], "summary": [], "view_graph": {}}
    pairs = {}
    for spec in a.alignments:
        name, path = spec.split("=", 1)
        t = time.time()
        try:
            al = EA.load_alignment(path, label=name, min_track_length=a.min_track)
        except FileNotFoundError as e:
            print(f"{name}: skipped ({e})", flush=True)
            continue
        res["alignments"][name] = {"path": str(al.root), "images": len(al.model.images), "points": len(al.model.points),
                                   "points_all": al.points_before_track_filter, "stations": len(al.stations),
                                   "summary_tracks": al.summary.get("tracks"),
                                   "residual_rms_native_px": al.summary.get("residual_rms_native_px")}
        res["eps"] += EA.eps_table(al)
        res["eps_by_angle"] += EA.eps_by_angle(al)
        er = EA.eps_by_range(al, family="Navcam")
        res["eps_by_range"][name] = {k: v for k, v in er.items()}
        p = EA.pair_survival(al, n_points=a.n_points, seed=a.seed)
        pairs[name] = p
        g = EA.fit_gate(p, families=None, fit_tau=True, theta_max_deg=a.theta_max, n_boot=a.n_boot, seed=a.seed,
                        form=a.form, illumination=a.illumination)
        g = dict(g, alignment=name)
        res["gate_fits"].append(g)
        r = EA.decorrelation(al, n_points=min(a.n_points, 6000), seed=a.seed, n_boot=a.n_boot or 100)
        fr = EA.fit_rho(r)
        res["rho"][name] = {k: r[k] for k in ("theta_deg", "rho", "n", "se") if k in r}
        res["rho_fits"].append({"alignment": name, "n_tracks": r["n_tracks"], "first_bin": fr["first_bin"],
                                "gaussian": fr["gaussian"], "exponential": fr["exponential"]})
        vg = EA.view_graph(al)
        res["view_graph"][name] = graph_report(vg)
        res["registration"] += EA.registration(al)
        res["summary"].append(EA.parameters(al, g, r))
        print(f"{name}: {len(al.model.images)} images, {len(al.model.points)} points, eps "
              f"{res['summary'][-1]['eps_intra_px']:.3f}/{res['summary'][-1]['eps_cross_px']:.3f} px, gate A "
              f"{g.get('A', float('nan')):.3f} theta_half {g.get('theta_half_deg', float('nan')):.2f}, "
              f"{time.time() - t:.0f} s", flush=True)
        del al, r, vg
        gc.collect()
        (out / "error_analysis_batch.json").write_text(json.dumps(res, indent=1, default=_j), encoding="utf-8")
    if len(pairs) > 1:
        pooled = EA.combine_pairs(list(pairs.values()))
        g = EA.fit_gate(pooled, families="Navcam-Navcam", fit_tau=True, theta_max_deg=a.theta_max, n_boot=a.n_boot,
                        seed=a.seed, form=a.form, illumination=a.illumination)
        res["gate_fits"].append(dict(g, alignment="pooled"))
        res["gate_forms_pooled"] = EA.compare_gate_forms(pooled, families="Navcam-Navcam", theta_max_deg=a.theta_max)
        res["gate_forms"] = {n: EA.compare_gate_forms(p, theta_max_deg=a.theta_max) for n, p in pairs.items()}
        np.savez_compressed(out / "gate_pairs.npz", **{f"{n}__{k}": v for n, p in pairs.items() for k, v in p.items()
                                                      if isinstance(v, np.ndarray)})
    res["model_defaults"] = EA.model_defaults()
    (out / "error_analysis_batch.json").write_text(json.dumps(res, indent=1, default=_j), encoding="utf-8")
    print("wrote", out / "error_analysis_batch.json")


if __name__ == "__main__":
    main()
