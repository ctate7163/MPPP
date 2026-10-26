"""
Figures of a law check (:mod:`mppp.calib.lawcheck`, ``law_validation.json``): how well each Navcam law predicts the
cameras of the blocks.  Left camera cool, right camera warm; the L / R pairs share their y axis.

    python studies/calibration/plot_lawcheck.py <folder with law_validation.json> [<out folder>] [--law <label>] [--min <images>]

1. ``lawcheck_residuals.png`` - per block, the median and rms residual with every unit held at each law against
   the units free: what holding the law costs the 3-D.
2. ``lawcheck_cameras_<law>.png`` - the free bin cameras' fx, fy (ppm) and k1 about the law against temperature,
   with their sd; NL left, NR right.
3. ``lawcheck_rig_<law>.png`` - the free rig units' pitch, yaw, roll (mdeg) about the law against sol.
4. ``lawcheck_pointing_<law>.png`` - the block principal points (cx, cy px) about the law against temperature, and
   the stereo pointing (NR - NL).
5. ``lawcheck_chi2.png`` - chi^2 / dof of every term for every law (1 = the law predicts the cameras to their sd).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EYE = {"NL": "C0", "NR": "C3"}            # left cool, right warm
LAWC = ["C0", "C2", "C4", "C9", "C8"]


def _load(folder):
    return json.loads((Path(folder) / "law_validation.json").read_text(encoding="utf-8"))


def residuals(res, out):
    import matplotlib.pyplot as plt
    laws = list(res["held"])
    fr = lambda k: res["free"][k] if k in res["free"] else res["free"]          # noqa: E731
    blocks = sorted(fr(laws[0])["blocks"])
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(blocks))
    w = 0.8 / (len(laws) + 1)
    for j, (stat, ax) in enumerate(zip(("median_px", "rms_px"), axs)):
        ax.bar(x - 0.4 + w / 2, [fr(laws[0])["blocks"][b][stat] for b in blocks], w, color="0.6",
               label=f"units free (from {laws[0]})")
        for i, k in enumerate(laws):
            ax.bar(x - 0.4 + w * (i + 1.5), [res["held"][k]["blocks"].get(b, {}).get(stat, np.nan) for b in blocks], w,
                   color=LAWC[i % len(LAWC)], label=f"held at {k}")
        ax.set_ylabel(stat.replace("_px", " residual (px)"))
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8, ncol=len(laws) + 1)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(blocks, rotation=35, ha="right", fontsize=8)
    axs[0].set_title("Navcam laws held against the blocks' own units (reprojection residuals)")
    fig.tight_layout()
    fig.savefig(out / "lawcheck_residuals.png", dpi=130)
    plt.close(fig)


def cameras(res, law, out, min_images=8):
    import matplotlib.pyplot as plt
    d = res["deviations"][law]
    terms = (("fx", "fx about the law (ppm)"), ("fy", "fy about the law (ppm)"), ("k1", "k1 about the law"))
    fig, axs = plt.subplots(len(terms), 2, figsize=(12, 9), sharex=True, sharey="row")
    for i, (k, lab) in enumerate(terms):
        for j, eye in enumerate(("NL", "NR")):
            ax = axs[i, j]
            rs = [r for r in d["cameras"] if r["eye"] == eye and r["images"] >= min_images]
            for early, mk in ((False, "o"), (True, "s")):
                rr = [r for r in rs if bool(r["early"]) == early]
                if rr:
                    ax.errorbar([r["T"] for r in rr], [r[f"d_{k}"] for r in rr],
                                yerr=[r[f"sd_{k}"] or 0 for r in rr], fmt=mk, ms=4, color=EYE[eye], alpha=0.75,
                                label=("early mission" if early else "after") if i == 0 else None)
            ax.axhline(0, color="k", lw=0.8)
            c = d["chi2"].get(f"{eye}_{k}")
            t = d["trends"].get(f"{eye}_{k}_vs_T")
            txt = []
            if c:
                txt.append(f"chi2/dof {c['chi2_dof']:.1f} (n {c['n']})")
            if t:
                txt.append(f"slope {t['slope']:+.3g} +- {t['se']:.2g} {t['units']}")
            ax.text(0.02, 0.95, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="white", alpha=0.75, lw=0))
            if j == 0:
                ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(f"{eye} bin cameras (free) about {law}")
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=8, loc="lower right")
            if i == len(terms) - 1:
                ax.set_xlabel("temperature (degC)")
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"lawcheck_cameras_{law}.png", dpi=130)
    plt.close(fig)


def rig(res, law, out, min_frames=8):
    import matplotlib.pyplot as plt
    d = res["deviations"][law]
    rs = [r for r in d["rigs"] if r["frames"] >= min_frames]
    fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, a in zip(axs, ("pitch", "yaw", "roll")):
        sc = ax.scatter([r["sol"] for r in rs], [r[f"d_{a}"] for r in rs], c=[r["T"] for r in rs], cmap="coolwarm",
                        s=18, zorder=3)
        ax.errorbar([r["sol"] for r in rs], [r[f"d_{a}"] for r in rs], yerr=[r[f"sd_{a}"] or 0 for r in rs], fmt="none",
                    ecolor="0.5", alpha=0.6, zorder=2)
        ax.axhline(0, color="k", lw=0.8)
        c, t = d["chi2"].get(f"rig_{a}"), d["trends"].get(f"rig_{a}_vs_sol")
        txt = [f"chi2/dof {c['chi2_dof']:.1f} (n {c['n']})"] if c else []
        if t:
            txt.append(f"trend {t['slope']:+.3g} +- {t['se']:.2g} {t['units']}")
        ax.text(0.02, 0.95, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8, bbox=dict(fc="white", alpha=0.75, lw=0))
        ax.set_ylabel(f"{a} about the law (mdeg)")
        ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=axs, label="temperature (degC)", shrink=0.6)
    axs[0].set_title(f"stereo rig units (free) about {law}")
    axs[-1].set_xlabel("sol")
    fig.savefig(out / f"lawcheck_rig_{law}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def pointing(res, law, out, min_images=8):
    import matplotlib.pyplot as plt
    d = res["deviations"][law]
    if not d.get("pointing"):
        return
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharey="row")
    for i, k in enumerate(("cx", "cy")):
        for j, eye in enumerate(("NL", "NR")):
            ax = axs[i, j]
            rs = [r for r in d["pointing"] if r["eye"] == eye and (r["images"] or 0) >= min_images]
            ax.errorbar([r["T"] for r in rs], [r[f"d_{k}"] for r in rs], yerr=[r[f"sd_{k}"] or 0 for r in rs],
                        fmt="o", ms=4, color=EYE[eye], alpha=0.8)
            ax.axhline(0, color="k", lw=0.8)
            c = d["chi2"].get(f"{eye}_{k}")
            if c:
                ax.text(0.02, 0.95, f"chi2/dof {c['chi2_dof']:.1f} (n {c['n']})", transform=ax.transAxes, va="top", bbox=dict(fc="white", alpha=0.75, lw=0),
                        fontsize=8)
            ax.set_title(f"{eye} {k} about {law}" if i == 0 else "")
            if j == 0:
                ax.set_ylabel(f"{k} (px)")
            ax.grid(alpha=0.3)
            ax.set_xlabel("temperature (degC)")
    for j, k in enumerate(("cx", "cy")):
        ax = axs[2, j]
        st = d.get("stereo") or []
        ax.errorbar([r["T"] for r in st], [r[f"d_{k}"] for r in st], yerr=[r[f"sd_{k}"] or 0 for r in st], fmt="D",
                    ms=4, color="C5")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"stereo pointing NR - NL {k} (px)")
        ax.set_xlabel("temperature (degC)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"lawcheck_pointing_{law}.png", dpi=130)
    plt.close(fig)


def chi2(res, out):
    import matplotlib.pyplot as plt
    laws = list(res["deviations"])
    keys = sorted({k for l_ in laws for k in res["deviations"][l_]["chi2"]})
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(keys))
    w = 0.8 / len(laws)
    for i, l_ in enumerate(laws):
        v = [res["deviations"][l_]["chi2"].get(k, {}).get("chi2_dof", np.nan) for k in keys]
        ax.bar(x - 0.4 + w * (i + 0.5), v, w, color=LAWC[i % len(LAWC)], label=l_)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("chi^2 / dof of the free units about the law")
    ax.set_title("How well each law predicts the units (1 = to their sd)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "lawcheck_chi2.png", dpi=130)
    plt.close(fig)


def main(argv):
    import matplotlib
    matplotlib.use("Agg")
    args = [a for a in argv if not a.startswith("--")]
    law = argv[argv.index("--law") + 1] if "--law" in argv else None
    mn = int(argv[argv.index("--min") + 1]) if "--min" in argv else 8
    for v in (law, str(mn)):
        if v in args:
            args.remove(v)
    res = _load(args[0])
    out = Path(args[1] if len(args) > 1 else args[0])
    out.mkdir(parents=True, exist_ok=True)
    residuals(res, out)
    chi2(res, out)
    for k in ([law] if law else list(res["deviations"])):
        cameras(res, k, out, mn)
        rig(res, k, out, mn)
        pointing(res, k, out, mn)
    print(f"figures in {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
