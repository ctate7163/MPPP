"""
The Navcam consensus studies (not part of the installed package).  The consensus fit itself, its candidate
folder, the promotion gate and the joint-side tools live in :mod:`mppp.calib.fit` (``fit_navcam``,
``write_navcam``, ``check_navcam``, ``leave_one_block_out``, ``k1_slopes_from_blocks``, ``fit_rig_translation``,
``consensus_change`` ...) and are re-exported here under their earlier names for the study scripts and notebook.
What stays here: :func:`rig_residuals` / :func:`plot_rig_residuals` - the per-block rig residuals against the
consensus rig, tested against temperature and sol.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from mppp.calib import fit as F
from mppp.calib.fit import (AXIS_TERMS, DEFAULT_TERMS, EARLY_APPLY_Z, EARLY_MISSION_SOL, EARLY_STEP_PX, EARLY_TERMS,  # noqa: F401
                            K1_APPLY_Z, LOO_LIMITS, PASS_LIMITS, PROMOTE_LIMITS, RAD_BINS, THERMAL_STEPS, THERMAL_TERMS,
                            UNITS, _axis_ppm, _fit_quadratic, _joint_kw, _pp_slopes, _quadratic_design, _thermal_values,
                            baseline_degeneracy, block_span_m, check_lines, consensus_change, early_start_from_blocks,
                            fit_early_offsets, fit_rig_translation, k1_slopes_from_blocks, large_blocks,
                            leave_one_block_out, ols as _ols, profile_thermal, radial_stats, read_cameras_bin as _read_cameras_bin)
from mppp.calib.joint import early_keypoint_map, rotation_mdeg as _rotation_mdeg  # noqa: F401
import navcal as NC

PathLike = Union[str, Path]
fit_consensus = F.fit_navcam
write_consensus = F.write_navcam
check_candidate = F.check_navcam
V0P73_TERMS = DEFAULT_TERMS


def promote_candidate(candidate_dir: PathLike, note: str, repo_root: PathLike = None, commit: bool = True,
                      verbose: bool = True) -> Dict[str, Any]:
    """Promote a Navcam candidate folder with :func:`mppp.calib.fit.promote_navcam` and (``commit``, the default,
    as the 0.86 pipeline did) commit ``src/mppp/data/cmods`` in ``repo_root`` so the promotion travels with the
    repository (claude_sync pushes it)."""
    import subprocess
    out = F.promote_navcam(candidate_dir, note, check=False)
    if verbose:
        print(f"promoted {sorted(out['files'])} into {out['folder']} ({out['stamp']})", flush=True)
    if commit and repo_root:
        try:
            subprocess.run(["git", "add", "src/mppp/data/cmods"], cwd=str(repo_root), check=True)
            subprocess.run(["git", "commit", "-q", "-m", f"cmods: Navcam consensus promoted ({note})"], cwd=str(repo_root), check=True)
            out["commit"] = "committed"
        except Exception as e:                                             # noqa: BLE001
            out["commit"] = f"not committed ({e}); commit src/mppp/data/cmods by hand"
            if verbose:
                print(out["commit"], flush=True)
    return out


# ------------------------------------------------------------------------------------- rig residuals
def rig_residuals(scapes_cfg: Dict[str, PathLike], cameras_dir: Optional[PathLike] = None, points: int = 8000,
                  iterations: int = 60, verbose: bool = True, epoch_gap: Optional[float] = 50.0,
                  epoch_breaks: Optional[Sequence[float]] = (300.0,)) -> Dict[str, Any]:
    """
    v0p73: does the rig follow the model?  Each block on its own, with the Navcam cameras of ``cameras_dir`` (default:
    in use) held and their whole model applied per image - fx, fy per axis and NR cx against temperature, the rig
    drift with its yaw rate, the early-mission offsets - and only the rig rotation free.  The block's rig minus the
    model's rig (its constant rotation; the temperature and sol terms are in the keypoints) is the residual: yaw,
    pitch, roll mdeg.  In the v0p73 form the yaw residual should not follow the block temperature (NR cx carries the
    stereo temperature change) and none should follow the sol (the drift rates).  Returns {"blocks": [{"block",
    "T_median_degC", "sol_median", "images", "yaw_mdeg", "pitch_mdeg", "roll_mdeg"}], "fits": {angle: {"vs_T": ...,
    "vs_sol": ...}}, "cameras": dir}: slopes mdeg/degC and mdeg per 1000 sol with se and z.
    v0p75: ``epoch_gap`` (sols; None = off): a block whose Navcam visits are more than that many sols apart gets one
    rig per visit (:func:`mppp.sfm.navcal.split_rig_by_epoch`) and one row per visit (``block`` "name [sols a-b]",
    ``site`` the block, ``epoch``) - the early visits of Chal Rocks Sid and Three Forks then test the drift on their own.
    v0p75.1: ``epoch_gap`` 50 sols (was 100) and ``epoch_breaks`` (sol 300, the pitch knot): images on both sides of
    the knot are separate rows even within one visit (Rochette 178-190 + 341).
    """
    import pycolmap
    from mppp.calib.constrain import thermal_terms
    from mppp.paths import cmods_dir
    from mppp.sfm.project import navcam_distortion_terms
    d = Path(cameras_dir) if cameras_dir else cmods_dir()
    nl = json.loads((d / "M2020_NL_fisheye_tangential.json").read_text(encoding="utf-8"))
    nr = json.loads((d / "M2020_NR_fisheye_tangential.json").read_text(encoding="utf-8"))
    rj = json.loads((d / "M2020_N_rig.json").read_text(encoding="utf-8"))
    R_model = np.asarray(rj["R_sensor_from_ref"], float)
    rig = (R_model, np.asarray(rj["t_sensor_from_ref_m"], float))
    tl, tr = thermal_terms(nl.get("thermal")), thermal_terms(nr.get("thermal"))
    t0 = tl["T0"]
    axis_ppm = {"NL": (tl["fx_ppm"], tl["fy_ppm"]), "NR": (tr["fx_ppm"], tr["fy_ppm"])}
    pp = {1: (tl["cx_px"], tl["cy_px"]), 2: (tr["cx_px"], tr["cy_px"])}
    drift = rj.get("drift")
    em = nl.get("early_mission") or {}
    rows = []
    t_all = time.time()
    for kb, (name, folder) in enumerate(scapes_cfg.items(), 1):
        t = time.time()
        try:
            sc = NC.load_scape(name, folder)
            cams = {k: pycolmap.Camera(model=j["model"], width=5120, height=3840, params=np.asarray(j["params"], float))
                    for k, j in (("NL", nl), ("NR", nr))}
            rec, proj, idx = NC.merge_scapes([sc], cams, rig, points_per_scape=points, seed=0)
        except Exception as e:                                                # noqa: BLE001
            rows.append({"block": name, "error": f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"  {name}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        temps = idx["temps"]
        for cid, key in ((1, "NL"), (2, "NR")):
            c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
            proj.cameras[key] = navcam_distortion_terms(c, "refine")
        proj.settings["navcam_rig_yaw"] = "refine"
        kw: Dict[str, Any] = {"pp_slopes": pp, "axis_ppm": axis_ppm}
        if drift:
            kw["drift"] = drift
        k1s = {"NL": float((nl.get("thermal") or {}).get("k1_per_degC") or 0.0),
               "NR": float((nr.get("thermal") or {}).get("k1_per_degC") or 0.0)}
        if any(k1s.values()):
            kw["k1_slopes"] = k1s
        applied = [k for k in (em.get("applied_terms") or []) if k in EARLY_TERMS]
        if applied:
            kw["extra_map"] = early_keypoint_map(proj, float(em["before_sol"]),
                                                 [float(em.get(f"d{k}_px", 0.0)) if k in applied else 0.0 for k in EARLY_TERMS])
        # v0p75: a block whose Navcam visits are more than ``epoch_gap`` sols apart (Chal Rocks Sid: sols 91-101 and
        # 360-378) gets one rig per visit, refined on its own, and one residual row per visit
        epochs = []
        if epoch_gap:
            rec_e, proj_e, eps = NC.split_rig_by_epoch(rec, proj, temps, gap=float(epoch_gap),
                                                       breaks=tuple(epoch_breaks or ()))     # v0p75.1: + sol 300
            if len(eps) > 1:
                rec, proj, epochs = rec_e, proj_e, eps
        ba = NC.joint_adjust(rec, proj, temps, 0.5 * (tl["fx_ppm"] + tl["fy_ppm"]), t0, max_iterations=iterations,
                             refine_rig="rotation", hold_cameras=("NL", "NR"), **kw)
        units = []
        if epochs:
            for e in epochs:
                if e.get("rig_id") is None:
                    continue
                rg = rec.rigs[int(e["rig_id"])]
                sid = next(iter(rg.non_ref_sensors), None)
                if sid is None:
                    continue
                units.append((f"{name} [sols {e['sol_min']}-{e['sol_max']}]", rg.sensor_from_rig(sid),
                              e["T_median_degC"], e["sol_median"], e["images"],
                              {"epoch": e["epoch"], "sols": [e["sol_min"], e["sol_max"]], "frames": e["frames"]}))
        if not units:
            _, _, Tr = NC.stereo_rig(rec)
            Ts = [v for n, v in temps.items() if n in {im.name for im in rec.images.values()}]
            sols = [float(r["sol"]) for r in proj.images if r.get("sol") is not None]
            units.append((name, Tr, float(np.median(Ts)) if Ts else None, float(np.median(sols)) if sols else None,
                          int(rec.num_reg_images()), None))
        for label, Tr, Tm, sm, n_im, ep in units:
            a = NC.rig_angles(np.asarray(Tr.rotation.matrix()), R_ref=R_model)
            row = {"block": label, "site": name, "T_median_degC": Tm, "sol_median": sm, "images": int(n_im),
                   "yaw_mdeg": a["yaw_mdeg"], "pitch_mdeg": a["pitch_mdeg"], "roll_mdeg": a["roll_mdeg"],
                   "cost": float(ba["final_cost"]), "iterations": ba["iterations"]}
            if ep:
                row["epoch"] = ep
            rows.append(row)
            if verbose:
                print(f"  {label}: T {(row['T_median_degC'] if row['T_median_degC'] is not None else float('nan')):+.1f} degC, "
                      f"sol {(row['sol_median'] or float('nan')):.0f}: residual yaw "
                      f"{row['yaw_mdeg']:+.2f}, pitch {row['pitch_mdeg']:+.2f}, roll {row['roll_mdeg']:+.2f} mdeg "
                      f"({time.time() - t:.0f} s)  {NC.progress(kb, len(scapes_cfg), t_all, 'block')}", flush=True)
    ok = [r for r in rows if "error" not in r]
    fits = {}
    for ang in ("yaw", "pitch", "roll"):
        y = np.array([r[f"{ang}_mdeg"] for r in ok], float)
        T = np.array([np.nan if r["T_median_degC"] is None else r["T_median_degC"] for r in ok], float)
        S = np.array([np.nan if r["sol_median"] is None else r["sol_median"] for r in ok], float)
        vt, vs = _ols(T, y), _ols(S / 1000.0, y)
        fits[ang] = {"mean_mdeg": float(np.mean(y)) if len(y) else None, "sd_mdeg": float(np.std(y)) if len(y) > 1 else None,
                     "vs_T": vt, "vs_sol": vs}
        if verbose and vt.get("slope") is not None:
            print(f"{ang}: residual mean {fits[ang]['mean_mdeg']:+.2f} (sd {fits[ang]['sd_mdeg']:.2f}) mdeg; vs block "
                  f"temperature {vt['slope']:+.3f} +- {vt['se']:.3f} mdeg/degC (z {vt['z']:+.1f}); vs sol "
                  f"{vs['slope']:+.2f} +- {vs['se']:.2f} mdeg/1000 sol (z {vs['z']:+.1f})", flush=True)
    return {"blocks": rows, "fits": fits, "cameras": str(d),
            "model": {"axis_ppm": axis_ppm, "pp_slopes": {"NL": list(pp[1]), "NR": list(pp[2])},
                      "yaw_rate_mdeg_per_1000sol": 1000.0 * float((drift or {}).get("yaw_mdeg_per_sol") or 0.0)}}


def plot_rig_residuals(res: Dict[str, Any], out: Optional[PathLike] = None, figsize=(12, 7)):
    """Rig residuals (yaw, pitch, roll) against the block temperature (top) and sol (bottom), with the fitted lines."""
    import matplotlib.pyplot as plt
    ok = [r for r in res["blocks"] if "error" not in r]
    fig, axs = plt.subplots(2, 3, figsize=figsize, squeeze=False)
    for j, ang in enumerate(("yaw", "pitch", "roll")):
        y = np.array([r[f"{ang}_mdeg"] for r in ok], float)
        for i, (k, lab, scale) in enumerate((("T_median_degC", "block Navcam temperature (degC)", 1.0),
                                              ("sol_median", "sol", 1.0))):
            ax = axs[i][j]
            x = np.array([np.nan if r[k] is None else r[k] for r in ok], float)
            ax.plot(x, y, "o", color="C0")
            for xx, yy, r in zip(x, y, ok):
                ax.annotate(r["block"][:12], (xx, yy), fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")
            f = res["fits"][ang]["vs_T" if i == 0 else "vs_sol"]
            if f.get("slope") is not None:
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 2)
                xm = f["mean_x"] * (1000.0 if i == 1 else 1.0)
                sl = f["slope"] / (1000.0 if i == 1 else 1.0)
                ax.plot(xs, f["intercept"] + sl * (xs - xm), "-", color="C3",
                        label=(f"{f['slope']:+.3f} +- {f['se']:.3f} mdeg/degC" if i == 0 else
                               f"{f['slope']:+.2f} +- {f['se']:.2f} mdeg/1000 sol") + f" (z {f['z']:+.1f})")
                ax.legend(fontsize=7)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xlabel(lab, fontsize=8)
            ax.set_ylabel(f"{ang} residual (mdeg)", fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("Navcam rig residuals per block (cameras and rig model held, rig rotation free)", fontsize=10)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
    return fig
