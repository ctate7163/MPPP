r"""
Where the Navcam rig's early drift ends: the knee of the pitch and yaw trends, profiled (0.99.5).

Christian, 16 Nov 2026: "where does the yaw and pitch knee prefer to be from the new freeze2 and deck joint results?"
and "to resolve the knee sol, add some deck station and blocks in the 250-400 range to run in the next freeze run".

Two independent sets of rig rotations against sol:
* the deck joint's station rigs (``all_rig.json`` of ``D:\rover_frame_eod_early``: one per station), and
* a Navcam joint's rig epochs (``navcam_joint\rig_residuals.json``: the epochs' residuals about the joint's law, put
  back on the law of ``navcam_joint\M2020_N_rig.json``).

For each angle a hinge - one rate before the knee, one after, yaw with a temperature term as in the deck law - is
fitted with the knee at every sol of a grid; the Delta chi^2 (scaled by the set's own scatter) gives the best knee,
its 68 % (Delta chi^2 <= 1) and 95 % (<= 4) ranges and how much worse the knee at sol 250 (the law's) fits.  Both
sets together: the sum of the two profiles.  Roll is fitted as a line and as "flat, then linear" (no drift before
the knee).  On 16 Nov the deck put the pitch knee at 355 (330-390) and the blocks at 220 (165-270): the sets are
thin exactly there (9 deck stations and 5 block epochs in sols 250-400), which the freeze-3 run fills.

    python -m rpp_studies.calibration.knee --deck D:\rover_frame_eod_early\all_rig.json ^
        --blocks D:\rover_blocks\camera_analysis\navcal_tied_<day>\navcam_joint --out <folder>

Writes ``knee_profile.json``, ``knee_profile.txt`` and ``knee_profile.png``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

KNEES = np.arange(60, 1001, 5)
ROLL_KNEES = np.arange(0, 1501, 10)
LAW_KNEE = 250


def deck_sets(all_rig: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for ang in ("pitch", "yaw", "roll"):
        lo = set((all_rig.get("angles", {}).get(ang) or {}).get("left_out") or [])
        r = [x for x in all_rig["rows"] if x["tag"] not in lo and x.get(f"{ang}_mdeg") is not None]
        out[ang] = (np.array([x["sol"] for x in r], float), np.array([x["T"] for x in r], float),
                    np.array([x[f"{ang}_mdeg"] for x in r], float))
    return out


def block_sets(residuals: Dict[str, Any], drift: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """The epochs' rotations: their residuals about the joint's law plus the law (hinge in the angles of
    ``hinge_angles``, a line in the others; the constant is free in every fit, so the law's offset is not needed)."""
    b = residuals["blocks"]
    s = np.array([x["sol_median"] for x in b], float)
    T = np.array([x["T_median_degC"] for x in b], float)
    knot = float(drift.get("knot_sol", LAW_KNEE))
    out = {}
    for ang in ("pitch", "yaw", "roll"):
        late = 1e3 * float(drift.get(f"{ang}_mdeg_per_sol", 0.0))
        early = 1e3 * float(drift.get(f"{ang}_early_mdeg_per_sol", drift.get(f"{ang}_mdeg_per_sol", 0.0)))
        law = 1e-3 * (late * s + (early - late) * np.minimum(s - knot, 0.0))
        out[ang] = (s, T, np.array([x[f"{ang}_mdeg"] for x in b], float) + law)
    return out


def _fit(cols, y):
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    dof = max(1, len(y) - A.shape[1])
    cov = np.linalg.inv(A.T @ A) * (r @ r) / dof
    return float(r @ r), c, np.sqrt(np.diag(cov))


def profile(s, T, y, temp: bool, knees: Sequence[float] = KNEES, kind: str = "hinge") -> Dict[str, Any]:
    """The knee profile of one angle in one set.  ``kind``: "hinge" (rates before and after) or "flat" (constant
    before, linear after)."""
    one = np.ones_like(s)
    temp = bool(temp) and float(np.ptp(T)) > 1e-6            # no temperature term when the set has one temperature
    tcol = [T + 20.0] if temp else []
    rss, fits = [], []
    for k in knees:
        if kind == "hinge":
            cols = [one, s / 1e3, np.minimum(s - k, 0) / 1e3] + tcol
        else:
            cols = [one, np.maximum(s - k, 0) / 1e3] + tcol
        f = _fit(cols, y)
        rss.append(f[0])
        fits.append(f)
    rss = np.array(rss)
    i = int(np.argmin(rss))
    p = (3 if kind == "hinge" else 2) + len(tcol)
    s2 = rss[i] / max(1, len(y) - p)
    d = (rss - rss[i]) / s2
    knees = np.asarray(knees)
    _, c, se = fits[i]
    line = _fit([one, s / 1e3] + tcol, y)
    n = len(y)
    bic = lambda r, q: n * np.log(r / n) + q * np.log(n)                    # noqa: E731
    out = {"n": n, "best": int(knees[i]), "r68": [int(knees[d <= 1].min()), int(knees[d <= 1].max())],
           "r95": [int(knees[d <= 4].min()), int(knees[d <= 4].max())], "sd_mdeg": float(np.sqrt(s2)),
           "line_minus_best_chi2": float((line[0] - rss[i]) / s2), "dbic_vs_line": float(bic(rss[i], p) - bic(line[0], p - 1)),
           "profile": [float(x) for x in d], "knees": [int(k) for k in knees],
           "before_n": int((s < knees[i]).sum()), "in_250_400": int(((s >= 250) & (s < 400)).sum())}
    if kind == "hinge":
        out.update(rate_after=float(c[1]), se_after=float(se[1]), rate_before=float(c[1] + c[2]), se_before=float(np.hypot(se[1], se[2])))
        if LAW_KNEE in set(int(k) for k in knees):
            out["law_knee_chi2"] = float(d[list(knees).index(LAW_KNEE)])
    else:
        out.update(rate_after=float(c[1]), se_after=float(se[1]), line_rate=float(line[1][1]))
    if temp:
        out["per_degC"] = float(c[-1])
    return out


def analyze(deck: Optional[Dict[str, Any]], blocks: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    sets = {}
    if deck is not None:
        sets["deck"] = deck_sets(deck)
    if blocks is not None:
        sets["blocks"] = block_sets(blocks["residuals"], blocks["drift"])
    res: Dict[str, Any] = {"sets": {k: {a: int(len(v[a][0])) for a in v} for k, v in sets.items()}, "pitch": {}, "yaw": {},
                           "roll": {}}
    for ang, temp in (("pitch", False), ("yaw", True)):
        for name, ss in sets.items():
            res[ang][name] = profile(*ss[ang], temp)
        if len(sets) == 2:
            tot = np.array(res[ang]["deck"]["profile"]) + np.array(res[ang]["blocks"]["profile"])
            i = int(np.argmin(tot))
            tot = tot - tot[i]
            k = KNEES
            res[ang]["both"] = {"best": int(k[i]), "r68": [int(k[tot <= 1].min()), int(k[tot <= 1].max())],
                                "r95": [int(k[tot <= 4].min()), int(k[tot <= 4].max())],
                                "law_knee_chi2": float(tot[list(k).index(LAW_KNEE)]), "profile": [float(x) for x in tot]}
    for name, ss in sets.items():
        res["roll"][name] = profile(*ss["roll"], True, ROLL_KNEES, kind="flat")
    return res


def text(res: Dict[str, Any]) -> str:
    lines = ["Navcam rig knee profile (rpp_studies.calibration.knee)", f"sets: {res['sets']}"]
    for ang in ("pitch", "yaw"):
        for name in ("deck", "blocks", "both"):
            r = res[ang].get(name)
            if not r:
                continue
            extra = (f"; rates {r['rate_before']:+.1f} +- {r['se_before']:.1f} / {r['rate_after']:+.2f} +- {r['se_after']:.2f} "
                     f"mdeg/1000 sol, sd {r['sd_mdeg']:.2f} mdeg, {r['in_250_400']} points in sols 250-400") if "rate_before" in r else ""
            lines.append(f"{ang:5s} {name:6s}: knee {r['best']} (68 % {r['r68'][0]}-{r['r68'][1]}, 95 % {r['r95'][0]}-{r['r95'][1]}), "
                         f"knee {LAW_KNEE} worse by {r.get('law_knee_chi2', float('nan')):+.1f}{extra}")
    for name, r in res["roll"].items():
        lines.append(f"roll  {name:6s}: flat then linear - knee {r['best']} (68 % {r['r68'][0]}-{r['r68'][1]}), rate after "
                     f"{r['rate_after']:+.2f} +- {r['se_after']:.2f}; a line ({r['line_rate']:+.2f}) is worse by only "
                     f"{r['line_minus_best_chi2']:+.1f} (BIC {r['dbic_vs_line']:+.1f} for the knee); T {r.get('per_degC', 0):+.3f} mdeg/degC")
    return "\n".join(lines) + "\n"


def figure(res: Dict[str, Any], png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, ang in zip(axs, ("pitch", "yaw")):
        for name, style in (("deck", "-"), ("blocks", "--"), ("both", "-")):
            r = res[ang].get(name)
            if r:
                ax.plot(KNEES, r["profile"], style, lw=2 if name == "both" else 1.2, label=f"{name}: {r['best']}")
        ax.axvline(LAW_KNEE, color="0.5", lw=0.8)
        ax.axhline(1, color="0.7", lw=0.6, ls=":")
        ax.axhline(4, color="0.7", lw=0.6, ls=":")
        ax.set_ylim(0, 20)
        ax.set_xlim(KNEES[0], 700)
        ax.set_xlabel("knee sol")
        ax.set_ylabel("Delta chi^2")
        ax.set_title(f"Navcam rig {ang}: where the early drift ends")
        ax.legend(fontsize=8)
    fig.savefig(png, dpi=130)
    plt.close(fig)


def load_blocks(folder: Path) -> Optional[Dict[str, Any]]:
    folder = Path(folder)
    rr, rig = folder / "rig_residuals.json", folder / "M2020_N_rig.json"
    if not rr.is_file() or not rig.is_file():
        return None
    return {"residuals": json.loads(rr.read_text(encoding="utf-8")),
            "drift": json.loads(rig.read_text(encoding="utf-8")).get("drift") or {}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", default=None, help="the deck joint's all_rig.json")
    ap.add_argument("--blocks", default=None, help="a Navcam joint's navcam_joint folder (rig_residuals.json, M2020_N_rig.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    deck = json.loads(Path(a.deck).read_text(encoding="utf-8")) if a.deck and Path(a.deck).is_file() else None
    blocks = load_blocks(Path(a.blocks)) if a.blocks else None
    if deck is None and blocks is None:
        print("knee: neither --deck nor --blocks was found")
        return 1
    res = analyze(deck, blocks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "knee_profile.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "knee_profile.txt").write_text(text(res), encoding="utf-8")
    try:
        figure(res, out / "knee_profile.png")
    except Exception as e:                                             # noqa: BLE001 - the numbers do not depend on it
        print(f"knee: no figure ({type(e).__name__}: {e})")
    print(text(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
