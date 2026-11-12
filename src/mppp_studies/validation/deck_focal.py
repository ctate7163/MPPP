"""
mppp_studies.validation.deck_focal - test 4: the Navcam focal law against temperature on the rover deck.

The thermal term of the Navcam focal law (+36 ppm / degC) comes from terrain tie points.  The deck gives a test that
uses none: ONE Navcam frame of the surveyed deck targets, its pose free and one focal scale ``s`` free (fx, fy x
(1 + s); the law's camera at the frame's temperature otherwise held),

    minimise  sum |project(law(T) x (1 + s), pose, X_target) - uv|^2        over pose (6) and s

A single image with a free pose cannot see the size of the object - a uniform scale of the deck (its thermal
expansion, a survey scale error) is taken up by the camera's distance - nor the stereo baseline.  ``s`` comes from
the perspective of the target field alone (the targets lie 0.8 to 2.4 m from the camera in a 96 degree field).  So

    s(T) flat          the law's thermal term is right
    s(T) = -k T        the law over-states the focal change by k

and the same frames with the law frozen at T0 (``frozen=True``) must return the law's own slope: the sensitivity
check.  Frames: the archive deck frames of the deck_joint (``vic.jsonl``: per frame the eye, sol, temperature and the
measured target pixels at full resolution).

    python -m mppp_studies.validation.deck_focal --frames vic.jsonl --out DIR
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

log = logging.getLogger("mppp.validation")
PathLike = Union[str, Path]
SIGMA_PX = 0.5                   # a target's measurement sd (full-resolution px), for the weights
T0_DEGC = -20.0
MIN_TARGETS = 5
MAX_RMS_PX = 1.5


def _scaled(params: Sequence[float], s: float) -> List[float]:
    p = [float(v) for v in params]
    p[0] *= 1.0 + s
    p[1] *= 1.0 + s
    return p


def fit_frame(rec: Dict[str, Any], cams: Dict[str, Dict[str, Any]], frozen: bool = False,
              targets: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """One frame of ``vic.jsonl`` -> {"s_ppm", "sd_ppm", "rms_px", "n", "T", "sol", "eye"}; None with fewer than
    :data:`MIN_TARGETS` surveyed targets.  ``frozen``: the law at :data:`T0_DEGC` instead of the frame's temperature."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from mppp.calib.constrain import camera_at
    from ..calibration.fiducials import RIGID, Projector, survey_nav_m
    names = [k for k in rec["targets"] if k in (targets or RIGID)]
    if len(names) < MIN_TARGETS:
        return None
    X = np.array([survey_nav_m([k])[k] for k in names])
    uv = np.array([rec["targets"][k] for k in names], float)
    model, params, _ = camera_at(cams[rec["eye"]], T0_DEGC if frozen else rec["T"], rec["sol"])
    R0 = np.asarray(rec["R"], float)
    t0 = -R0 @ np.asarray(rec["C"], float)

    def res(v):
        R = Rotation.from_rotvec(v[:3]).as_matrix() @ R0
        pr = Projector(model, _scaled(params, v[6] * 1e-3))(X @ R.T + (t0 + v[3:6]))
        return ((pr - uv) / SIGMA_PX).ravel()

    r = least_squares(res, np.zeros(7), x_scale=[1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1.0])
    if not np.all(np.isfinite(r.fun)):
        return None
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac)
    except np.linalg.LinAlgError:
        return None
    rms = float(np.sqrt(np.mean((r.fun * SIGMA_PX) ** 2) * 2))
    return {"stem": rec["stem"], "eye": rec["eye"], "sol": rec["sol"], "T": float(rec["T"]), "n": len(names),
            "s_ppm": float(r.x[6] * 1e3), "sd_ppm": float(np.sqrt(cov[6, 6]) * 1e3), "rms_px": rms,
            "scale_native": rec.get("scale")}


def regress(rows: Sequence[Dict[str, Any]], with_sol: bool = False) -> Dict[str, Any]:
    """Weighted ``s = a + b (T - T0) [+ c (sol - 1000) / 1000]``, Huber weights on top of the formal ones; the se from
    the scatter, and from a jackknife over sols (frames of one sol share a rover state)."""
    T = np.array([r["T"] for r in rows])
    y = np.array([r["s_ppm"] for r in rows])
    w0 = 1.0 / np.array([max(r["sd_ppm"], 50.0) for r in rows])
    cols = [np.ones(len(rows)), T - T0_DEGC]
    if with_sol:
        cols.append((np.array([r["sol"] for r in rows], float) - 1000.0) / 1000.0)
    A = np.stack(cols, 1)

    def solve(m):
        w = w0[m].copy()
        for _ in range(10):
            x = np.linalg.lstsq(A[m] * w[:, None], y[m] * w, rcond=None)[0]
            z = (y[m] - A[m] @ x) * w0[m]
            sc = 1.4826 * np.median(np.abs(z)) + 1e-9
            w = w0[m] / np.maximum(1.0, np.abs(z) / (2.0 * sc))
        return x, sc

    m = np.ones(len(rows), bool)
    x, sc = solve(m)
    sols = sorted({r["sol"] for r in rows})
    jk = np.array([solve(np.array([r["sol"] != s for r in rows]))[0] for s in sols])
    n = len(sols)
    se = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
    return {"n": len(rows), "sols": n, "T_range": [float(T.min()), float(T.max())],
            "a_ppm_at_T0": float(x[0]), "slope_ppm_per_degC": float(x[1]), "se_a": float(se[0]), "se_slope": float(se[1]),
            "sol_rate_ppm_per_1000": float(x[2]) if with_sol else None, "se_sol_rate": float(se[2]) if with_sol else None,
            "chi_scale": float(sc)}


def law_slopes(cams: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """The law's own focal slope per eye (ppm / degC), by evaluating it 20 degC apart."""
    from mppp.calib.constrain import camera_at
    out = {}
    for e, m in cams.items():
        a = camera_at(m, T0_DEGC - 10.0, 1000)[1]
        b = camera_at(m, T0_DEGC + 10.0, 1000)[1]
        out[e] = float((b[0] / a[0] - 1.0) / 20.0 * 1e6)
    return out


def run(frames: PathLike, out: Optional[PathLike] = None, cmods: Optional[PathLike] = None) -> Dict[str, Any]:
    from mppp.calib.models import load_camera
    cams = {e: load_camera("N", e, cmods) for e in ("L", "R")}
    recs = [json.loads(l) for l in Path(frames).read_text(encoding="utf-8").splitlines() if l.strip()]
    res: Dict[str, Any] = {"frames_file": str(frames), "frames": len(recs), "law_slope_ppm_per_degC": law_slopes(cams),
                           "sigma_px": SIGMA_PX, "min_targets": MIN_TARGETS}
    for tag, frozen in (("law", False), ("law_frozen_at_T0", True)):
        rows = [r for r in (fit_frame(q, cams, frozen) for q in recs) if r and r["rms_px"] <= MAX_RMS_PX
                and np.isfinite(r["sd_ppm"])]
        res[tag] = {"all": regress(rows), "all_with_sol": regress(rows, True),
                    "L": regress([r for r in rows if r["eye"] == "L"]), "R": regress([r for r in rows if r["eye"] == "R"]),
                    "six_or_more": regress([r for r in rows if r["n"] >= 6]) if sum(r["n"] >= 6 for r in rows) > 20 else None,
                    "sd_ppm_median": float(np.median([r["sd_ppm"] for r in rows]))}
        res[tag + "_rows"] = rows
    if out:
        o = Path(out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "deck_focal.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        try:
            _figure(res, o / "deck_focal_vs_temperature.png")
        except Exception as e:                               # noqa: BLE001
            log.warning("figure: %s", e)
    return res


def _figure(res: Dict[str, Any], png: PathLike) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, tag, title in zip(axes, ("law", "law_frozen_at_T0"),
                              ("the law at each frame's temperature (flat = the law is right)",
                               "sensitivity: the law frozen at -20 degC (must return the law's slope)")):
        rows, f = res[tag + "_rows"], res[tag]["all"]
        for e, c in (("L", "#1f5fbf"), ("R", "#c8502a")):
            rr = [r for r in rows if r["eye"] == e]
            ax.scatter([r["T"] for r in rr], [r["s_ppm"] for r in rr], s=9, c=c, alpha=0.6, label=f"N{e}")
        t = np.array(f["T_range"])
        ax.plot(t, f["a_ppm_at_T0"] + f["slope_ppm_per_degC"] * (t - T0_DEGC), "k",
                label=f"{f['slope_ppm_per_degC']:+.1f} +- {f['se_slope']:.1f} ppm / degC")
        ax.axvline(T0_DEGC, c="0.5", ls="--", lw=1)
        ax.set_ylim(-6000, 6000)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Navcam temperature (degC)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("focal scale wanted by the deck targets, s (ppm)")
    fig.tight_layout()
    fig.savefig(png, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="vic.jsonl of the deck_joint")
    ap.add_argument("--cmods", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a.frames, a.out, a.cmods)
    print("law slope:", r["law_slope_ppm_per_degC"])
    for tag in ("law", "law_frozen_at_T0"):
        f = r[tag]["all"]
        print(f"{tag}: s slope {f['slope_ppm_per_degC']:+.1f} +- {f['se_slope']:.1f} ppm/degC, level {f['a_ppm_at_T0']:+.0f} +- "
              f"{f['se_a']:.0f} ppm at {T0_DEGC:g} degC ({f['n']} frames, {f['sols']} sols, T {f['T_range'][0]:.0f}..{f['T_range'][1]:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
