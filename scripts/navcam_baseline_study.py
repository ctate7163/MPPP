"""
The Navcam stereo baseline from the large blocks (MPPP v0p80; the camera-model recipe of 14 Oct 2026, section 4).

Over tens of metres the waypoint priors fix a block's scale independently of the 0.42 m baseline, so a block with
stations spread over >= 40 m in more than one direction can measure the baseline length against the localisations.
Each such block is aligned once more as a variant (``<WORK>/colmap_baseline``; the production alignment is not
touched), with:

- the Navcam cameras, distortion and thermal model held at the promoted consensus (``NAVCAM_INTRINSICS = "hold"``;
  the temperature bins held at the thermal model);
- the rig rotation free per epoch and the rig translation free (``NAVCAM_RIG_REFINE = "full"``, a weak prior of
  ``RIG_TRANSLATION_SIGMA_M`` per axis on the right centre; the epochs of a block share one translation);
- the station priors as in production (1 m per axis).

The final Navcam adjustment estimates the rig covariance; ``project.json`` ``settings.reconstruction.rig_fit`` holds
the baseline and its sd, and ``error_input/summary.json`` the block's scale against the priors.

    python scripts\\navcam_baseline_study.py run      [--root D:\\scapes\\colmap] [--sites ...] [--dry-run]
    python scripts\\navcam_baseline_study.py combine  [--root ...] [--out DIR] [--write-rig DIR]
    python scripts\\navcam_baseline_study.py all      (run, then combine)

``combine`` writes ``<root>/camera_analysis/navcam_baseline_<date>/baseline_study.json`` and ``.md``: per block the
baseline, its sd, scale_refined_over_prior, temperature and sol; the weighted mean with a between-block term
(DerSimonian-Laird), chi2/dof, and the regressions on temperature and sol.  The rule (recipe): adopt only if the
combined estimate differs from the baseline in use by more than 3 sigma AND the blocks agree (chi2/dof <= 2); then
``--write-rig DIR`` writes a candidate ``M2020_N_rig.json`` with the translation scaled to it (direction kept) for
``promote_cmods.py`` - and the joint consensus is refitted once (run_pipeline --redo navcam_consensus_2).
Caveat: this measures stereo scale against the waypoint scale; a systematic scale error of the localisations goes
straight into it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402

GROUP = "navcam_baseline"                      # sites.json group (recipe: >= 8 stations over >= 40 m in 2-D)
VARIANT = "baseline"
SETS = ["NAVCAM_INTRINSICS=hold", "NAVCAM_RIG_REFINE=full", "RIG_TRANSLATION_SIGMA_M=0.005"]
ADOPT_Z = 3.0
MAX_CHI2_DOF = 2.0


def run(a) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_sites.py"), "--root", a.root, "--variant", VARIANT,
           "--source", a.source]
    cmd += ["--sites", *a.sites] if a.sites else ["--group", GROUP]
    for s in SETS:
        cmd += ["--set", s]
    if a.dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _block_rows(root: Path, sites: Optional[List[str]]) -> List[Dict[str, Any]]:
    from mppp.sfm.sites import load_site_table, site_label, work_folder
    table = load_site_table()
    names = sites or list(dict.fromkeys(table.get("groups", {}).get(GROUP, [])))
    rows = []
    for s in names:
        if s not in table["sites"]:
            continue
        work = work_folder(root, s, False, table["sites"][s]["sols"])
        proj = work / f"colmap_{VARIANT}"
        pj = proj / "project.json"
        row: Dict[str, Any] = {"site": s, "label": site_label(s), "folder": str(proj)}
        if not pj.is_file():
            row["missing"] = "no colmap_baseline/project.json (run the study first)"
            rows.append(row)
            continue
        p = json.loads(pj.read_text(encoding="utf-8"))
        rs = (p.get("settings") or {}).get("reconstruction") or {}
        fit = rs.get("rig_fit") or {}
        if fit.get("baseline_m") is None or rs.get("refine_rig") is not True:
            row["missing"] = "no rig fit with the translation refined (NAVCAM_RIG_REFINE = 'full')"
            rows.append(row)
            continue
        ims = [r for r in p.get("images") or [] if str(r.get("instrument", "")).startswith("N")]
        Ts = [float(r["camera_temperature_degC"]) for r in ims if r.get("camera_temperature_degC") is not None]
        sols = [float(r["sol"]) for r in ims if r.get("sol") is not None]
        sm = proj / "error_input" / "summary.json"
        scale = None
        if sm.is_file():
            sims = json.loads(sm.read_text(encoding="utf-8")).get("prior_similarity_by_component") or []
            if sims:
                big = max(sims, key=lambda c: len(c.get("stations") or []))
                scale = big.get("scale_refined_over_prior")
        row.update({"baseline_m": float(fit["baseline_m"]), "baseline_sd_m": fit.get("baseline_sd_m"),
                    "baseline_start_m": fit.get("baseline_start_m"), "epochs": len(fit.get("epochs") or []),
                    "T_median_degC": float(np.median(Ts)) if Ts else None,
                    "sol_median": float(np.median(sols)) if sols else None, "images": len(ims),
                    "scale_refined_over_prior": scale, "variance_factor": fit.get("variance_factor")})
        rows.append(row)
    return rows


def _wls_line(x, y, w):
    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    X = np.c_[np.ones(len(x)), x - x.mean()]
    A = X.T @ (X * w[:, None])
    b = np.linalg.solve(A, X.T @ (w * y))
    r = y - X @ b
    dof = max(len(x) - 2, 1)
    s2 = max(float(np.sum(w * r ** 2) / dof), 1.0)            # never better than the formal errors
    se = float(np.sqrt(np.linalg.inv(A)[1, 1] * s2))
    return {"slope": float(b[1]), "se": se, "z": float(b[1] / se) if se > 0 else None, "mean_x": float(x.mean())}


def combine_rows(rows: List[Dict[str, Any]], baseline_in_use: float) -> Dict[str, Any]:
    """Weighted mean with a between-block term (DerSimonian-Laird), chi2/dof, trends and the adoption verdict."""
    ok = [r for r in rows if r.get("baseline_m") is not None and r.get("baseline_sd_m")]
    out: Dict[str, Any] = {"blocks_used": [r["site"] for r in ok], "baseline_in_use_m": float(baseline_in_use)}
    if len(ok) < 2:
        out["verdict"] = f"too few blocks with a baseline and its sd ({len(ok)})"
        out["adopt"] = False
        return out
    b = np.array([r["baseline_m"] for r in ok])
    s = np.array([float(r["baseline_sd_m"]) for r in ok])
    w = 1.0 / s ** 2
    m0 = float(np.sum(w * b) / np.sum(w))
    Q = float(np.sum(w * (b - m0) ** 2))
    k = len(b)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1.0 / (s ** 2 + tau2)
    m = float(np.sum(ws * b) / np.sum(ws))
    se = float(np.sqrt(1.0 / np.sum(ws)))
    chi2 = Q / max(k - 1, 1)
    d = m - float(baseline_in_use)
    z = d / se
    out.update({"mean_m": m, "se_m": se, "fixed_effect_mean_m": m0, "tau_m": float(np.sqrt(tau2)),
                "chi2_dof": chi2, "difference_mm": 1e3 * d, "difference_ppm": 1e6 * d / float(baseline_in_use),
                "z": z})
    T = [r.get("T_median_degC") for r in ok]
    S = [r.get("sol_median") for r in ok]
    if all(t is not None for t in T):
        out["vs_temperature_mm_per_degC"] = {k_: (1e3 * v if k_ in ("slope", "se") else v)
                                             for k_, v in (_wls_line(T, b, ws) or {}).items()}
    if all(x is not None for x in S):
        out["vs_sol_mm_per_1000sol"] = {k_: (1e6 * v if k_ in ("slope", "se") else v)
                                        for k_, v in (_wls_line(S, b, ws) or {}).items()}
    agree = chi2 <= MAX_CHI2_DOF
    out["adopt"] = bool(abs(z) > ADOPT_Z and agree)
    out["verdict"] = ("ADOPT: the combined baseline differs from the one in use by "
                      f"{1e3 * d:+.2f} mm ({z:+.1f} sigma) and the blocks agree (chi2/dof {chi2:.2f})" if out["adopt"] else
                      f"keep the baseline in use: {1e3 * d:+.2f} mm ({z:+.1f} sigma), chi2/dof {chi2:.2f}"
                      + ("" if agree else " - the blocks disagree (look at the per-block values before anything else)"))
    return out


def combine(a) -> int:
    from mppp.paths import cmods_dir
    root = Path(a.root)
    rig = json.loads((cmods_dir() / "M2020_N_rig.json").read_text(encoding="utf-8"))
    b_use = float(rig.get("baseline_m") or np.linalg.norm(rig["t_sensor_from_ref_m"]))
    rows = _block_rows(root, a.sites)
    res = combine_rows(rows, b_use)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"navcam_baseline_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    rep = {"rows": rows, "combined": res, "settings": SETS, "variant": VARIANT,
           "rig_in_use": {"file": str(cmods_dir() / "M2020_N_rig.json"), "baseline_m": b_use,
                          "translation": rig.get("translation")}}
    (out / "baseline_study.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    lines = [f"# Navcam baseline from the large blocks ({datetime.date.today()})", "",
             f"Baseline in use: {b_use:.5f} m ({rig.get('translation') or 'CAHV'}).", "",
             "| block | baseline m | sd mm | scale/prior | T °C | sol | epochs | note |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['label']} | - | - | - | - | - | - | {r['missing']} |")
            continue
        sd = r.get("baseline_sd_m")
        lines.append(f"| {r['label']} | {r['baseline_m']:.5f} | {1e3 * sd:.2f} | " if sd else f"| {r['label']} | {r['baseline_m']:.5f} | - | ")
        lines[-1] += (f"{r['scale_refined_over_prior']:.5f} | " if r.get("scale_refined_over_prior") else "- | ") + \
            f"{r.get('T_median_degC') or float('nan'):.1f} | {r.get('sol_median') or float('nan'):.0f} | {r.get('epochs')} | |"
    lines += ["", f"**{res['verdict']}**", ""]
    if res.get("mean_m") is not None:
        lines += [f"- combined {res['mean_m']:.5f} ± {1e3 * res['se_m']:.2f} mm (between-block {1e3 * res['tau_m']:.2f} mm), "
                  f"chi2/dof {res['chi2_dof']:.2f}, {res['difference_ppm']:+.0f} ppm from the one in use"]
        for k, u in (("vs_temperature_mm_per_degC", "mm/°C"), ("vs_sol_mm_per_1000sol", "mm/1000 sol")):
            v = res.get(k) or {}
            if v.get("slope") is not None:
                lines.append(f"- {k.split('_')[1]}: {v['slope']:+.4f} ± {v['se']:.4f} {u} (z {v['z']:+.1f})")
        lines.append("- expected thermal change of an aluminium bar (23 ppm/°C): 0.0098 mm/°C")
    (out / "baseline_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if a.write_rig:
        if not res.get("adopt") and not a.force:
            print("not written: the rule says keep the baseline in use (--force writes it anyway)", flush=True)
        else:
            d = Path(a.write_rig)
            d.mkdir(parents=True, exist_ok=True)
            t = np.asarray(rig["t_sensor_from_ref_m"], float)
            t_new = t * (res["mean_m"] / np.linalg.norm(t))
            r2 = dict(rig, t_sensor_from_ref_m=t_new.tolist(), baseline_m=float(res["mean_m"]), translation="fitted",
                      translation_source=f"navcam_baseline_study {datetime.date.today()} ({len(res['blocks_used'])} "
                                         f"large blocks, cameras held): {res['verdict']}")
            r2["translation_baseline_study"] = res
            (d / "M2020_N_rig.json").write_text(json.dumps(r2, indent=1, default=float), encoding="utf-8")
            print(f"candidate rig written: {d / 'M2020_N_rig.json'} (promote with scripts\\promote_cmods.py, then refit "
                  f"the joint once)", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "combine", "all"])
    ap.add_argument("--root", default="D:/scapes/colmap")
    ap.add_argument("--sites", nargs="*", help=f"default: the {GROUP} group of sites.json")
    ap.add_argument("--source", default="processed", choices=["processed", "pds"],
                    help="run: processed (default: the production run's images) or pds")
    ap.add_argument("--out", help="combine: output folder (default <root>/camera_analysis/navcam_baseline_<date>)")
    ap.add_argument("--write-rig", help="combine: write a candidate M2020_N_rig.json there if the rule says adopt")
    ap.add_argument("--force", action="store_true", help="combine --write-rig: write it even if the rule says keep")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    if a.cmd in ("run", "all"):
        rc = run(a)
    if a.cmd in ("combine", "all") and not a.dry_run:
        rc = combine(a) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
