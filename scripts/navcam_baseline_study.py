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

    python scripts\\navcam_baseline_study.py run      [--root D:\\scapes\\colmap] [--sites ...] [--flags waypoint_flags.json]
                                                     [--prior-sigma 0.5] [--dry-run]
    python scripts\\navcam_baseline_study.py combine  [--root ...] [--out DIR] [--write-rig DIR] [--promote]
    python scripts\\navcam_baseline_study.py all      (run, then combine)
    python scripts\\navcam_baseline_study.py confirm  [--root ...] [--out DIR]   (after a promotion: the blocks again with
                                                     the new baseline held; reverted if the scale agreement is not better)

v0p82 (17 Oct 2026, Christian): the blocks are the ``navcam_baseline`` group's sites with more than 4 waypoints over
more than 30 m (sites.json ``n_waypoints``, ``max_waypoint_separation_m``; others are left out with a note); the
stations flagged by the waypoint check (``--flags``) run without their position prior; the priors at 0.5 m per axis
(``--prior-sigma``; the localisations are good to ~0.3-0.5 m).  ``combine`` also measures the block scale agreement:
the robust similarity scale between the refined stations and their waypoints (mppp.sfm.waypoint_check) in the
production run (baseline held) and in the baseline run (free), the scale predicted with the combined baseline, and
an east/north scale test (a map-projection error of the waypoints would show east-west only).  Adopted only if the
combined baseline is > 3 sigma from the one in use, the blocks agree (chi2/dof <= 2), the predicted scale agreement
is better than now and the block scales are isotropic; ``--promote`` then promotes the candidate rig, and ``confirm``
checks it on the blocks.

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

GROUP = "navcam_baseline"                      # sites.json group
VARIANT = "baseline"
CHECK_VARIANT = "baseline_check"               # confirm: the blocks with the promoted baseline held
SETS = ["NAVCAM_INTRINSICS=hold", "NAVCAM_RIG_REFINE=full", "RIG_TRANSLATION_SIGMA_M=0.005"]
ADOPT_Z = 3.0
MAX_CHI2_DOF = 2.0
MIN_WAYPOINTS = 4                              # v0p82: more than this many waypoints ...
MIN_SEPARATION_M = 30.0                        # ... over more than this many metres (sites.json waypoint fields)
PRIOR_SIGMA_M = 0.5                            # v0p82: the baseline runs' waypoint prior (production: 1 m)
ISOTROPY_Z = 3.0                               # east and north block scales must agree within this many sigma


def qualifying_sites(sites=None, table=None):
    """(sites to run, [(site, reason) left out]) - the group's (or ``sites``) with more than MIN_WAYPOINTS
    waypoints over more than MIN_SEPARATION_M."""
    from mppp.sfm.sites import load_site_table
    t = table if table is not None else load_site_table()
    names = list(sites) if sites else list(dict.fromkeys(t.get("groups", {}).get(GROUP, [])))
    keep, out = [], []
    for s in names:
        v = t["sites"].get(s)
        if v is None:
            out.append((s, "not a site"))
            continue
        n, sep = v.get("n_waypoints"), v.get("max_waypoint_separation_m")
        if n is None or sep is None:
            keep.append(s)                         # a version-4 table: no waypoint fields - kept
        elif n > MIN_WAYPOINTS and sep > MIN_SEPARATION_M:
            keep.append(s)
        else:
            out.append((s, f"{n} waypoints over {sep:.1f} m (needs > {MIN_WAYPOINTS} over > {MIN_SEPARATION_M:g} m)"))
    return keep, out


def flagged_stations(path):
    """The ``flagged`` list of a waypoint_flags.json (or [])."""
    if not path or not Path(path).is_file():
        return []
    return list(json.loads(Path(path).read_text(encoding="utf-8")).get("flagged") or [])


def run(a, variant: str = VARIANT, sets=None) -> int:
    sites, left = qualifying_sites(a.sites)
    for x, why in left:
        print(f"left out of the baseline study: {x} ({why})", flush=True)
    if not sites:
        print("no site qualifies", flush=True)
        return 2
    cmd = [sys.executable, str(ROOT / "scripts" / "run_sites.py"), "--root", a.root, "--variant", variant,
           "--source", a.source, "--sites", *sites]
    sets = list(SETS if sets is None else sets) + [f"PRIOR_SIGMA_M={float(a.prior_sigma)}"]
    fl = flagged_stations(getattr(a, "flags", None))
    if fl:                                         # v0p82: flagged localisations run without their prior
        sets.append("NO_POSITION_PRIOR_AT=" + json.dumps(fl))
    for s_ in sets:
        cmd += ["--set", s_]
    if a.dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _wc(p: Path) -> Optional[Dict[str, Any]]:
    """The waypoint check of a project folder (None if it cannot be made)."""
    from mppp.sfm import waypoint_check as WC
    if not ((p / "project.json").is_file() and (p / "error_input" / "poses.csv").is_file()):
        return None
    try:
        return WC.check_block(p)
    except Exception:                                                      # noqa: BLE001
        return None


def scale_agreement(work: Path, proj: Path) -> Dict[str, Any]:
    """v0p82: the block scale against the waypoints (robust similarity, ppm) in the production run (``colmap``,
    baseline held) and in the baseline run (``proj``, free), with the east / north scales of the production run."""
    out: Dict[str, Any] = {}
    for key, p in (("held", work / "colmap"), ("free", proj)):
        r = _wc(p)
        if r and r.get("similarity") and not r["similarity"].get("translation_only"):
            out[f"scale_{key}_ppm"] = float(r["similarity"]["scale_ppm"])
            out[f"stations_{key}"] = int(r["similarity"]["stations"])
            if key == "held":
                out["anisotropy_held"] = r.get("anisotropy")
                out["flagged_held"] = r.get("flagged")
    return out


def _block_rows(root: Path, sites: Optional[List[str]]) -> List[Dict[str, Any]]:
    from mppp.sfm.sites import load_site_table, site_label, work_folder
    table = load_site_table()
    names = sites or qualifying_sites(None, table)[0]
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
        row.update(scale_agreement(work, proj))
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
    sa = scale_prediction(rows, m, float(baseline_in_use))
    out["scale_agreement"] = sa
    better = sa.get("rms_predicted_ppm") is None or sa["rms_predicted_ppm"] < sa["rms_held_ppm"]
    iso = sa.get("isotropy") or {}
    isotropic = iso.get("z") is None or abs(iso["z"]) <= ISOTROPY_Z
    out["adopt"] = bool(abs(z) > ADOPT_Z and agree and better and isotropic)
    why = [] if agree else ["the blocks disagree (look at the per-block values before anything else)"]
    if not better:
        why.append("the block scale agreement would not improve")
    if not isotropic:
        why.append(f"the block scales differ east-west and north-south ({iso['z']:+.1f} sigma: a map-projection error "
                   f"of the waypoints, not the baseline?)")
    out["verdict"] = ("ADOPT: the combined baseline differs from the one in use by "
                      f"{1e3 * d:+.2f} mm ({z:+.1f} sigma), the blocks agree (chi2/dof {chi2:.2f}), the scale agreement "
                      f"improves (rms {sa.get('rms_held_ppm', float('nan')):.0f} -> {sa.get('rms_predicted_ppm', float('nan')):.0f} ppm)"
                      if out["adopt"] else
                      f"keep the baseline in use: {1e3 * d:+.2f} mm ({z:+.1f} sigma), chi2/dof {chi2:.2f}"
                      + ("" if not why else " - " + "; ".join(why)))
    return out


def scale_prediction(rows: List[Dict[str, Any]], b_new: Optional[float], b_use: float) -> Dict[str, Any]:
    """v0p82: the block scales against the waypoints now (baseline held at ``b_use``) and predicted with ``b_new``
    (the reconstruction scales with the baseline: s_new = s_held x b_new / b_use), their rms over the blocks, and the
    east-minus-north scale over the blocks (weighted mean, z)."""
    held = [(r["site"], r["scale_held_ppm"]) for r in rows if r.get("scale_held_ppm") is not None]
    out: Dict[str, Any] = {"blocks": [h[0] for h in held]}
    if not held:
        return out
    s = np.array([h[1] for h in held]) * 1e-6
    out["rms_held_ppm"] = float(1e6 * np.sqrt(np.mean(s ** 2)))
    out["mean_held_ppm"] = float(1e6 * np.mean(s))
    if b_new:
        sp = (1 + s) * (b_new / b_use) - 1
        out["predicted_ppm"] = dict(zip(out["blocks"], (1e6 * sp).tolist()))
        out["rms_predicted_ppm"] = float(1e6 * np.sqrt(np.mean(sp ** 2)))
    free = [r["scale_free_ppm"] for r in rows if r.get("scale_free_ppm") is not None]
    if free:
        out["rms_free_ppm"] = float(np.sqrt(np.mean(np.square(free))))
    an = [r["anisotropy_held"] for r in rows if r.get("anisotropy_held") and r["anisotropy_held"].get("se_E_ppm")
          and r["anisotropy_held"].get("se_N_ppm")]
    if an:
        dif = np.array([a["scale_E_ppm"] - a["scale_N_ppm"] for a in an])
        se = np.array([np.hypot(a["se_E_ppm"], a["se_N_ppm"]) for a in an])
        w = 1 / se ** 2
        mu = float(np.sum(w * dif) / np.sum(w))
        e = float(np.sqrt(1 / np.sum(w)))
        out["isotropy"] = {"blocks": len(an), "east_minus_north_ppm": mu, "se_ppm": e, "z": mu / e}
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
    sa = res.get("scale_agreement") or {}
    if sa.get("rms_held_ppm") is not None:
        lines += ["", "## Block scale against the waypoints (robust similarity, ppm)", "",
                  "| block | held (now) | free (this study) | predicted with the combined baseline | east | north |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            if r.get("scale_held_ppm") is None:
                continue
            an = r.get("anisotropy_held") or {}
            pr = (sa.get("predicted_ppm") or {}).get(r["site"])
            lines.append(f"| {r['label']} | {r['scale_held_ppm']:+.0f} | "
                         + (f"{r['scale_free_ppm']:+.0f}" if r.get("scale_free_ppm") is not None else "-") + " | "
                         + (f"{pr:+.0f}" if pr is not None else "-") + " | "
                         + (f"{an['scale_E_ppm']:+.0f} ± {an['se_E_ppm']:.0f}" if an.get("se_E_ppm") else "-") + " | "
                         + (f"{an['scale_N_ppm']:+.0f} ± {an['se_N_ppm']:.0f}" if an.get("se_N_ppm") else "-") + " |")
        lines += ["", f"- rms now {sa['rms_held_ppm']:.0f} ppm (mean {sa['mean_held_ppm']:+.0f}); predicted "
                      + (f"{sa['rms_predicted_ppm']:.0f} ppm" if sa.get("rms_predicted_ppm") is not None else "-")
                      + (f"; free baseline {sa['rms_free_ppm']:.0f} ppm" if sa.get("rms_free_ppm") is not None else "")]
        iso = sa.get("isotropy")
        if iso:
            lines.append(f"- east minus north scale over {iso['blocks']} blocks: {iso['east_minus_north_ppm']:+.0f} ± "
                         f"{iso['se_ppm']:.0f} ppm (z {iso['z']:+.1f}; |z| > {ISOTROPY_Z:g}: not the baseline)")
    (out / "baseline_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    wr = a.write_rig or (str(out / "candidate_rig") if getattr(a, "promote", False) else None)
    if wr:
        if not res.get("adopt") and not a.force:
            print("not written: the rule says keep the baseline in use (--force writes it anyway)", flush=True)
        else:
            d = Path(wr)
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
            if getattr(a, "promote", False) and res.get("adopt"):
                promote_rig(d, out, res)
                rep["promoted"] = json.loads((out / "baseline_promotion.json").read_text(encoding="utf-8"))
                (out / "baseline_study.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    return 0


def promote_rig(candidate: Path, out: Path, res: Dict[str, Any]) -> Dict[str, Any]:
    """v0p82: promote the candidate rig (the rig in use first copied to ``out/rig_before`` for ``confirm``)."""
    import shutil
    from mppp.paths import cmods_dir
    from mppp.sfm.navcal_consensus import promote_candidate
    before = out / "rig_before"
    before.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cmods_dir() / "M2020_N_rig.json", before / "M2020_N_rig.json")
    note = (f"v0p82 baseline study {datetime.date.today()}: {res['mean_m']:.5f} m "
            f"({res['difference_mm']:+.2f} mm, {res['z']:+.1f} sigma, {len(res['blocks_used'])} large blocks)")
    p = promote_candidate(candidate, note, ROOT)
    info = {"candidate": str(candidate), "rig_before": str(before / "M2020_N_rig.json"), "note": note,
            "returncode": p.get("returncode"), "commit": p.get("commit"),
            "rms_held_ppm": (res.get("scale_agreement") or {}).get("rms_held_ppm"),
            "rms_predicted_ppm": (res.get("scale_agreement") or {}).get("rms_predicted_ppm"),
            "when": datetime.datetime.now().isoformat(timespec="seconds")}
    (out / "baseline_promotion.json").write_text(json.dumps(info, indent=1, default=float), encoding="utf-8")
    print(f"baseline promoted ({note}); exit {p.get('returncode')}", flush=True)
    return info


def confirm(a) -> int:
    """v0p82: after a promotion, the study's blocks once more with the new baseline held (production settings,
    ``<WORK>/colmap_baseline_check``); the scale agreement must be better than before the promotion, else the rig
    before it is promoted back.  Writes ``baseline_confirm.json`` / ``.md`` next to the study.  Exit 0 confirmed,
    1 reverted, 2 nothing to confirm."""
    from mppp.sfm.sites import load_site_table, work_folder
    out = Path(a.out) if a.out else Path(a.root) / "camera_analysis" / f"navcam_baseline_{datetime.date.today():%Y%m%d}"
    pf = out / "baseline_promotion.json"
    if not pf.is_file():
        print(f"no promotion in {out}: nothing to confirm", flush=True)
        return 2
    info = json.loads(pf.read_text(encoding="utf-8"))
    import copy
    b = copy.copy(a)
    b.prior_sigma = 1.0                                   # production settings (the "before" scales are from them),
    rc = run(b, variant=CHECK_VARIANT, sets=[])          # the new rig, flagged priors still out
    table = load_site_table()
    sites, _ = qualifying_sites(a.sites, table)
    rows = []
    for s_ in sites:
        w = work_folder(Path(a.root), s_, False, table["sites"][s_]["sols"])
        r = _wc(w / f"colmap_{CHECK_VARIANT}")
        if r and r.get("similarity") and not r["similarity"].get("translation_only"):
            rows.append({"site": s_, "scale_ppm": float(r["similarity"]["scale_ppm"])})
    rms = float(np.sqrt(np.mean(np.square([r["scale_ppm"] for r in rows])))) if rows else None
    before = info.get("rms_held_ppm")
    ok = rms is not None and before is not None and rms < before
    res = {"runs_exit": rc, "blocks": rows, "rms_after_ppm": rms, "rms_before_ppm": before,
           "rms_predicted_ppm": info.get("rms_predicted_ppm"), "confirmed": bool(ok)}
    if not ok and not a.dry_run:
        from mppp.sfm.navcal_consensus import promote_candidate
        p = promote_candidate(Path(info["rig_before"]).parent, "v0p82 baseline promotion reverted: the block scale "
                              f"agreement did not improve (rms {before} -> {rms} ppm)", ROOT)
        res["reverted"] = {"returncode": p.get("returncode"), "commit": p.get("commit")}
    (out / "baseline_confirm.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    md = [f"# Baseline confirmation ({datetime.date.today()})", "",
          f"Block scale against the waypoints, rms over {len(rows)} blocks: before {before} ppm, predicted "
          f"{info.get('rms_predicted_ppm')} ppm, after {rms} ppm -> "
          + ("**confirmed**" if ok else "**not better: the rig before the promotion is back**"), "",
          "| block | scale ppm |", "|---|---|"] + [f"| {r['site']} | {r['scale_ppm']:+.0f} |" for r in rows]
    (out / "baseline_confirm.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "combine", "all", "confirm"])
    ap.add_argument("--root", default="D:/scapes/colmap")
    ap.add_argument("--sites", nargs="*", help=f"default: the {GROUP} group of sites.json")
    ap.add_argument("--source", default="pds", choices=["processed", "pds"],
                    help="run: pds (default since v0p82: selected again, the processed images reused; a site without a "
                         "WORK folder is processed) or processed")
    ap.add_argument("--flags", help="v0p82: a waypoint_flags.json - its flagged stations run without position prior")
    ap.add_argument("--prior-sigma", type=float, default=PRIOR_SIGMA_M,
                    help=f"v0p82: the waypoint position prior per axis in the baseline runs (default {PRIOR_SIGMA_M} m)")
    ap.add_argument("--promote", action="store_true",
                    help="v0p82, combine: promote the candidate rig when the rule says adopt (then run confirm)")
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
    if a.cmd == "confirm":
        rc = confirm(a)
    return rc


if __name__ == "__main__":
    sys.exit(main())
