"""
The Navcam rig drift checked on the rover deck across the mission (Christian, 4 Nov 2026: "Can you use the
fiducial data to check the roll trend? Can you get access to more deck images across the mission?").

Every Navcam stereo pair of the local PDS archive whose label puts the deck targets in frame is measured with the
cameras and the rig in use (the law at each image's temperature and sol; :func:`mppp_studies.calibration.fiducials.check_pds_pair`):
no block is needed, so every deck pair from sol 0 on counts.  Per pair the rig rotation the deck sees, with the rig
translation held (``rig_deckrot_*``: pitch, yaw, roll about NR's x, y, z, referred to the rig file's rotation
before its drift), is compared with the rig file's drift.  The tie points never enter, so this is an independent
test of the pitch, yaw and roll trends of the Navcam joint.

    python -m mppp_studies.calibration.deck_rig_trend                                     (the archive MPPP_PDS, the models in use)
    python -m mppp_studies.calibration.deck_rig_trend --joint D:\\rover_blocks\\camera_analysis\\navcal_tied_20261103
    python -m mppp_studies.calibration.deck_rig_trend --sols 600 700 --workers 2         (a trial on a few sols)

``--joint``: the drift and the reference rotation of that joint's rig file, and its block epochs on the plots.
Writes ``<out>/deck_pairs.csv``, ``deck_detections.csv``, ``deck_rig_trend.png``, ``summary.json``, ``report.md``.
The label scan is cached (``--cache``, default ``<root>/camera_analysis/deck_pds_scan.json``): a rerun reads only
new products.  Reads only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


from mppp_studies.calibration import fiducials as F

ANGLES = ("pitch", "yaw", "roll")


def _pair_job(args):
    fL, fR, cams, rig, extra = args
    try:
        return F.check_pds_pair(fL, fR, cams=cams, rig=rig, extra_rigid=extra)
    except Exception as exc:                      # noqa: BLE001 - one pair never stops the others
        return {"block": "pds", "pair": fL["sclk_key"], "sol": fL["sol"], "error": str(exc)}, [], []


def block_epochs(joint: Optional[Path], drift: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The joint's block epochs (rig_residuals.json) as rig angles about the rig file's sol-0 rotation."""
    if joint is None or not (joint / "rig_residuals.json").is_file() or not drift:
        return []
    from mppp.calib.constrain import drift_offset_rel
    out = []
    for b in json.loads((joint / "rig_residuals.json").read_text(encoding="utf-8")).get("blocks", []):
        s = float(b["sol_median"])
        out.append({"sol": s, "images": b.get("images"), "T": b.get("T_median_degC"),
                    **{a: float(b[f"{a}_mdeg"]) + float(drift_offset_rel(drift, a, [s], 0.0)[0]) for a in ANGLES}})
    return out


def plot(trend: Dict[str, Any], epochs: List[Dict[str, Any]], drift: Optional[Dict[str, Any]], out: Path,
         title: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None
    from mppp.calib.constrain import drift_offset_mdeg
    fig, axs = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    xs = np.linspace(0, max([p["sol"] for a in ANGLES for p in (trend.get(a) or {}).get("points", [])] + [2000]), 400)
    for ax, a in zip(axs, ANGLES):
        t = trend.get(a) or {}
        pts = t.get("points", [])
        if not pts:
            ax.set_ylabel(f"{a} (mdeg)")
            continue
        k = [p for p in pts if p["kept"]]
        d = [p for p in pts if not p["kept"]]
        Ts = [p["T"] if p["T"] is not None else np.nan for p in k]
        ax.errorbar([p["sol"] for p in k], [p["y"] for p in k], yerr=[p["sd"] for p in k], fmt="none", ecolor="0.6",
                    elinewidth=0.8, zorder=1)
        sc = ax.scatter([p["sol"] for p in k], [p["y"] for p in k], c=Ts, cmap="coolwarm", s=22, zorder=3,
                        edgecolors="k", linewidths=0.3, label=f"deck pairs ({len(k)})")
        if d:
            ax.scatter([p["sol"] for p in d], [p["y"] for p in d], marker="x", color="0.4", s=22, zorder=2,
                       label=f"dropped ({len(d)})")
        off = t["law"]["coef"][0] if drift and "law" in t else 0.0
        if epochs:                                 # shifted by the deck's offset from the law, to compare the trends
            ax.scatter([e["sol"] for e in epochs], [e[a] + off for e in epochs], marker="s", s=16, facecolors="none",
                       edgecolors="tab:green", zorder=2, label=f"block epochs (tie points) {off:+.1f}")
        if drift and "law" in t:
            c0 = t["law"]["coef"][0]
            ax.plot(xs, [c0 + drift_offset_mdeg(drift, x)[f"{a}_mdeg"] for x in xs], color="tab:green", lw=1.6,
                    label=f"rig law + deck offset {c0:+.1f}")
        if "line" in t:
            b0, b1 = t["line"]["coef"]
            s1 = t["line"]["se"][1]
            ax.plot(xs, b0 + b1 * (xs - 1000.0) / 1000.0, color="k", lw=1.2, ls="--",
                    label=f"deck line {b1:+.1f} +- {s1:.1f} mdeg/1000 sol")
        if "step" in t:
            c, s = t["step"]["coef"], t["step"]["se"]
            st = trend.get("step_sol")
            ax.plot([0, st, st, xs[-1]], [c[0], c[0], c[0] + c[1], c[0] + c[1]], color="tab:purple", lw=1.0,
                    label=f"step at sol {st:g}: {c[1]:+.1f} +- {s[1]:.1f}")
        good = [p["y"] for p in k if p["sd"] <= 2.0 * t.get("median_sd_mdeg", 1e9)] + [e[a] + off for e in epochs]
        if good:                                   # the scale of the well-measured pairs; the loose ones may run off
            lo, hi = float(np.min(good)), float(np.max(good))
            pad = 0.15 * (hi - lo) + 5.0
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(f"{a} (mdeg)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.colorbar(sc, ax=ax, pad=0.01).set_label("NL temperature (degC)")
    axs[-1].set_xlabel("sol")
    fig.suptitle(title)
    fig.tight_layout()
    f = out / "deck_rig_trend.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    return f.name


def report(summary: Dict[str, Any], fig: Optional[str], out: Path) -> Path:
    tr = summary["trend"]
    L = [f"# Navcam rig drift on the rover deck ({datetime.date.today():%d %b %Y})", "",
         f"Archive {summary['pds']}; {summary['frames']} Navcam full frames with the deck in frame, "
         f"{summary['pairs_found']} stereo pairs, {summary['pairs_solved']} with the rig rotation solved "
         f"(translation held), {summary['pairs_used']} used (rotation-fit rms <= {tr['max_rms']} px).  Cameras: the "
         f"law at each image's temperature; rig: `{summary['rig_file']}` at each sol.  Angles about NR's x (pitch), "
         f"y (yaw), z (roll), referred to the rig file's rotation before its drift.", "",
         "| angle | pairs | median sd | extra scatter | deck rate | law rate | deck - law | z | deck - law offset |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in ANGLES:
        t = tr.get(a) or {}
        if "line" not in t:
            L.append(f"| {a} | {t.get('n', 0)} | | | | | | | |")
            continue
        L.append(f"| {a} | {t['n']} | {t['median_sd_mdeg']:.1f} | {t['tau_mdeg']:.1f} | "
                 f"{t['line']['coef'][1]:+.2f} +- {t['line']['se'][1]:.2f} | {t.get('law_rate_mdeg_per_1000sol', float('nan')):+.2f} | "
                 f"{t.get('rate_minus_law', float('nan')):+.2f} | {t.get('rate_minus_law_z', float('nan')):+.1f} | "
                 f"{t.get('offset_vs_law_mdeg', float('nan')):+.1f} |")
    L += ["", "Rates in mdeg per 1000 sol, sd and offsets in mdeg.  The law's rate is the rig file's linear rate (for a "
          "hinged angle the late rate; the law fit below uses the full drift shape).", "",
          "| angle | BIC constant | BIC line | BIC law shape | BIC step |", "|---|---|---|---|---|"]
    for a in ANGLES:
        t = tr.get(a) or {}
        if "line" not in t:
            continue
        g = lambda k: f"{t[k]['bic']:.1f}" if k in t else ""          # noqa: E731
        L.append(f"| {a} | {g('constant')} | {g('line')} | {g('law')} | {g('step')} |")
    L += ["", f"Step at sol {tr.get('step_sol')}: " + ", ".join(
        f"{a} {tr[a]['step']['coef'][1]:+.1f} +- {tr[a]['step']['se'][1]:.1f}" for a in ANGLES
        if "step" in (tr.get(a) or {})) + " mdeg.", ""]
    u = summary.get("unsurveyed_rays") or {}
    for k, v in u.items():
        if v.get("X_nav_mm") is not None:
            se = v.get("se_jackknife_mm") or [float("nan")] * 3
            L.append(f"{k} from {v['n']} rays: RNAV ({', '.join(f'{x:.2f}' for x in v['X_nav_mm'])}) +- "
                     f"({', '.join(f'{x:.2f}' for x in se)}) mm; in use {tuple(v.get('in_use_nav_mm', []))}.")
    L += ["", "## How to read it", "",
          "* Each point is one stereo pair: NL's pose in the rover frame and NR's rotation relative to NL solved from "
          "the surveyed targets, the rig translation held at the rig file's.  The error bar is the fit's formal sd.",
          "* `extra scatter` is the scatter beyond the formal sd (pair-to-pair systematics: detection, the bottom of "
          "the field, the thermal state).",
          "* A constant `deck - law offset` is expected (the deck sits at the bottom edge of the frame, the tie points "
          "fill it); the test of the drift is the rate and the BIC of the law's shape against a constant and a line.",
          "* Green squares: the joint's block epochs (tie points), when `--joint` is given.", ""]
    if fig:
        L.append(f"![{fig}]({fig})")
    p = out / "report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pds", default=os.environ.get("MPPP_PDS"), help="the PDS archive (default MPPP_PDS)")
    ap.add_argument("--root", default=os.environ.get("MPPP_FREEZE_ROOT") or os.environ.get("MPPP_SCAPES"),
                    help="where camera_analysis is (default MPPP_FREEZE_ROOT, then MPPP_SCAPES)")
    ap.add_argument("--joint", default=None, help="a Navcam joint folder: its rig file and block epochs")
    ap.add_argument("--sols", nargs=2, type=int, default=None, metavar=("FIRST", "LAST"))
    ap.add_argument("--out", default=None, help="output folder (default <root>/camera_analysis/deck_rig_trend_<yyyymmdd>)")
    ap.add_argument("--cache", default=None, help="the label-scan cache (default <root>/camera_analysis/deck_pds_scan.json)")
    ap.add_argument("--workers", type=int, default=4, help="processes for the label scan and the pairs (default 4)")
    ap.add_argument("--max-rms", type=float, default=1.0, help="pairs whose rotation fit is above this rms (px) are not used")
    ap.add_argument("--step-sol", type=float, default=475.0, help="the sol of the step model (default 475)")
    ap.add_argument("--tdf4", action="store_true", help="use TDF4 (fitted, not surveyed) as a rigid target too")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not a.pds or not Path(a.pds).is_dir():
        print("ERROR: give --pds (or set MPPP_PDS)", file=sys.stderr)
        return 2
    pds = Path(a.pds)
    scan_root = pds / "datadrive" if (pds / "datadrive").is_dir() else pds
    root = Path(a.root) if a.root else pds
    ca = root / "camera_analysis"
    out = Path(a.out) if a.out else ca / f"deck_rig_trend_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    from mppp.calib.models import load_camera, load_rig
    joint = Path(a.joint) if a.joint else None
    rig_path = joint / "M2020_N_rig.json" if joint and (joint / "M2020_N_rig.json").is_file() else None
    rig = json.loads(rig_path.read_text(encoding="utf-8")) if rig_path else load_rig("N")
    if rig is None:
        print("ERROR: no Navcam rig file", file=sys.stderr)
        return 2
    cams = {e: load_camera("N", e) for e in ("L", "R")}
    t0 = time.time()
    print(f"scanning {scan_root} for Navcam frames with the deck in frame", flush=True)
    frames = F.scan_pds(scan_root, tuple(a.sols) if a.sols else None, cache=a.cache or ca / "deck_pds_scan.json",
                        progress=lambda s: print(s, flush=True), workers=a.workers)
    extra = ("TDF4",) if a.tdf4 else ()
    pairs_in = F.pds_pairs(frames, extra_rigid=extra)
    print(f"{len(pairs_in)} stereo pairs with >= 3 rigid targets predicted in each eye ({(time.time() - t0) / 60:.1f} min)",
          flush=True)
    jobs = [(fL, fR, cams, rig, extra) for fL, fR in pairs_in]
    rows, dets, rays = [], [], []
    if a.workers > 1 and len(jobs) > 4:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = ex.map(_pair_job, jobs, chunksize=2)
            for i, (r, d, ry) in enumerate(results, 1):
                rows.append(r)
                dets += d
                rays += ry
                if i % 25 == 0:
                    print(f"  {i} / {len(jobs)} pairs", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            r, d, ry = _pair_job(j)
            rows.append(r)
            dets += d
            rays += ry
            if i % 25 == 0:
                print(f"  {i} / {len(jobs)} pairs", flush=True)
    rows.sort(key=lambda r: (r.get("sol") or 0, r.get("pair") or ""))
    F.write_csv(out / "deck_pairs.csv", rows, F.PDS_COLUMNS + ["error"])
    F.write_csv(out / "deck_detections.csv", dets, F.DET_COLUMNS)
    drift = rig.get("drift")
    if isinstance(drift, str):
        drift = None
    trend = F.rig_trend(rows, drift, max_rms=a.max_rms, step_sol=a.step_sol)
    trend["step_sol"] = a.step_sol
    epochs = block_epochs(joint, drift)
    solved = [r for r in rows if F._num(r.get("rot_rms")) is not None]
    used = [r for r in solved if F._num(r.get("rot_rms")) <= a.max_rms]
    summary: Dict[str, Any] = {"pds": str(scan_root), "frames": len(frames), "pairs_found": len(pairs_in),
                               "pairs_solved": len(solved), "pairs_used": len(used),
                               "errors": len([r for r in rows if r.get("error")]),
                               "rig_file": str(rig_path or rig.get("file")), "trend": trend,
                               "minutes": round((time.time() - t0) / 60, 1)}
    summary["unsurveyed_rays"] = {}
    for k in F.UNSURVEYED:
        rk = [q for q in rays if q["target"] == k]
        if len(rk) >= 2:
            summary["unsurveyed_rays"][k] = F.fit_point_from_rays(rk, F.survey_nav_m([k])[k])
            summary["unsurveyed_rays"][k]["in_use_nav_mm"] = list(F.UNSURVEYED_NAV_MM[k])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    fig = plot(trend, epochs, drift, out, f"Navcam rig rotation seen by the rover deck ({len(used)} stereo pairs, "
                                        f"translation held) against the rig drift in use")
    rep = report(summary, fig, out)
    for ang in ANGLES:
        t = trend.get(ang) or {}
        if "line" in t:
            print(f"{ang:5s}: {t['n']} pairs, deck rate {t['line']['coef'][1]:+.2f} +- {t['line']['se'][1]:.2f}, "
                  f"law {t.get('law_rate_mdeg_per_1000sol', float('nan')):+.2f} mdeg/1000 sol "
                  f"(z {t.get('rate_minus_law_z', float('nan')):+.1f})")
    print(f"deck_rig_trend: {len(used)} pairs used; report {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
