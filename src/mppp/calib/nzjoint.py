"""
The combined Navcam + Mastcam-Z joint: one adjustment of the finished Navcam + Mastcam-Z blocks in which the Navcam
law (:mod:`mppp.calib.tied`: per-eye cameras at the reference temperature, thermal slopes, early-mission offsets,
the rig on its sol trend) and the Mastcam-Z focus law (:mod:`mppp.calib.ztied`: per-eye focus lines, distortion,
the Mastcam-Z rig) are refined together.

Why: the Navcam law comes from Navcam-only blocks, and the Mastcam-Z law is then fitted with the Navcam held.  If
the two instruments disagree (a Navcam scale or pointing the Mastcam-Z frames pull on), this joint shows it: it
starts from the promoted laws, frees both, and compares the result with the Navcam-only solution - the Navcam terms
moved in units of their sd, the Navcam and Mastcam-Z residuals before and after, the rig.  It is a check, not a
calibration: nothing is promoted by default (``mppp calib nz``; ``zcam_joints.bat`` / run_pipeline run it last).

    res = fit_nzjoint(blocks)            # {name: WORK folder} of finished Navcam + Mastcam-Z blocks
    write_nzjoint(res, out_dir)          # nz_joint.json, report.md
"""
from __future__ import annotations

import copy
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from . import joint as J
from . import tied as T
from . import ztied as Z

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]

# the combined solution agrees with the Navcam-only one when no Navcam term moves by more than this many of its sd
# and the Navcam residuals do not get worse by more than this fraction
AGREE_MAX_Z = 3.0
NAVCAM_ONLY_POINTS_PER_BLOCK = 3000      # points per added Navcam-only block (the tied joint's thinning)
AGREE_MAX_NAV_RESIDUAL_RISE = 0.05
# ... and when no term moves more than these (also the test when the covariance fails: exact ties): the Navcam
# block-agreement scale (median 77 ppm between blocks, 20 Oct)
AGREE_MAX_F_PPM = 100.0
AGREE_MAX_PP_PX = 0.3
TERMS_COMPARED = ("fx", "fy", "cx", "cy", "k1", "k2")


def _hold_z_pp(proj) -> None:
    """The Mastcam-Z bin cameras keep their principal points (ZL pinned, ZR cx pinned, ZR cy on its tied line) while
    the Navcam principal points are refined - as in the two separate joints."""
    for key, c in proj.cameras.items():
        if str(key).startswith("Z"):
            c["fixed_params"] = sorted(set(c.get("fixed_params") or []) | {"cx", "cy"})


def adjust_nz(rec, proj, idx, st, units, zunits, sg, hold: Sequence[str] = T.HOLD_DEFAULT, iterations: int = 100,
              passes: int = 2, covariance: bool = True, verbose: bool = True, iso_f: bool = T.ISO_F,
              zhold: Optional[Dict[str, Sequence[str]]] = None):
    """Navcam bins tied to the Navcam law and Mastcam-Z bins tied to the focus law, both laws free, the Navcam rig
    on its trend and the Mastcam-Z rig rotation refined.  Returns (ba, Navcam globals, Mastcam-Z globals)."""
    from ..sfm.reconstruction import bundle_adjust
    hook_n, gn = T.make_ties(st, units, hold=hold, iso_f=iso_f)
    hook_z, gz = Z.make_zties(sg, zunits, hold=zhold or ())

    def hook(prob, **kw):
        a = hook_n(prob, **kw) or {}
        b = hook_z(prob, **kw) or {}
        return {"cov_blocks": {**(a.get("cov_blocks") or {}), **(b.get("cov_blocks") or {})},
                "ties": {"navcam": a.get("ties"), "zcam": b.get("ties")},
                "held": {**(a.get("held") or {}), **(b.get("held") or {})},
                "tie_ids": list(a.get("tie_ids") or []) + list(b.get("tie_ids") or []),    # 0.94.4: both sets
                "tie_specs": list(a.get("tie_specs") or []) + list(b.get("tie_specs") or []),
                "exact_solve": bool(a.get("exact_solve") and b.get("exact_solve"))}    # 26 Oct: off with the Mastcam-Z

    _hold_z_pp(proj)
    ba = None
    n_pass = max(1, int(passes))
    for p_ in range(n_pass):
        ks, kmap = T.keypoint_terms(units, proj, idx["temps"], gn)
        ba = bundle_adjust(rec, proj, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=True,
                           refine_principal_point=True, refine_rig="rotation", max_iterations=iterations,
                           attitude_prior_deg=5.0, covariance=covariance and p_ == n_pass - 1, navcam_aspect="free",
                           linear_solver="sparse_schur", keypoint_scale=ks, keypoint_map=kmap,
                           refine_zcam_rig=sg.get("rig_fit") or "rotation", extra_residuals=hook)
        if iso_f:
            T.sync_iso(gn)
        if T.TIE_K1_SLOPE:
            T.sync_k1(gn)
        if verbose:
            log.info(f"  N+Z pass {p_ + 1}: {ba['brief'][:160]}")
    if isinstance(ba.get("extra"), dict):
        ba["extra"]["iso_f"] = bool(iso_f)
    return ba, gn, gz


def family_stats(rec, proj, units, idx, gn) -> Dict[str, Dict[str, float]]:
    """Residual statistics (native px) of the Navcam and of the Mastcam-Z observations, the Navcam keypoints moved by
    the thermal law ``gn`` as in the adjustment."""
    ks, kmap = T.keypoint_terms(units, proj, idx["temps"], gn)
    reg = [int(i) for i in rec.reg_image_ids()]
    nav = [i for i in reg if rec.images[i].name[:1] == "N"]
    zc = [i for i in reg if rec.images[i].name[:1] == "Z"]
    out = {}
    for name, ids in (("navcam", nav), ("zcam", zc)):
        if ids:
            out[name] = J.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap, images=ids)
    return out


def compare(nav_only: Dict[str, Any], combined: Dict[str, Any], held: Dict[str, Any], nz: Dict[str, Any],
            ) -> Dict[str, Any]:
    """The combined solution against the Navcam-only law: per Navcam eye and term the change at the reference
    temperature in px and in units of the combined sd, the thermal slopes, the residuals of each instrument with the
    Navcam held (the Mastcam-Z joint) and free (this joint); ``agree`` when no term moves beyond
    :data:`AGREE_MAX_Z` sd and the Navcam residual median rises less than :data:`AGREE_MAX_NAV_RESIDUAL_RISE`."""
    out: Dict[str, Any] = {"navcam": {}, "residuals": {"navcam_held": held, "both_free": nz}}
    worst = 0.0
    beyond: List[str] = []
    for eye in ("NL", "NR"):
        a = (nav_only.get(eye) or {}).get("params_T0") or {}
        b = (combined.get(eye) or {}).get("params_T0") or {}
        sd = (combined.get(eye) or {}).get("sd_params_T0") or {}
        rows = {}
        for k in TERMS_COMPARED:
            if k not in a or k not in b:
                continue
            d = float(b[k]) - float(a[k])
            s = sd.get(k)
            z = d / s if s and np.isfinite(s) and s > 0 else None
            rows[k] = {"navcam_only": float(a[k]), "combined": float(b[k]), "change": d, "sd": s, "z": z}
            if z is not None and k in ("fx", "fy", "cx", "cy"):
                worst = max(worst, abs(z))
            if k in ("fx", "fy") and abs(1e6 * d / float(a[k])) > AGREE_MAX_F_PPM:
                beyond.append(f"{eye} {k} {1e6 * d / float(a[k]):+.0f} ppm")
            if k in ("cx", "cy") and abs(d) > AGREE_MAX_PP_PX:
                beyond.append(f"{eye} {k} {d:+.2f} px")
        for k in ("fx_ppm_per_degC", "fy_ppm_per_degC", "cx_px_per_degC", "cy_px_per_degC"):
            if k in (nav_only.get(eye) or {}) and k in (combined.get(eye) or {}):
                rows[k] = {"navcam_only": float(nav_only[eye][k]), "combined": float(combined[eye][k]),
                           "change": float(combined[eye][k]) - float(nav_only[eye][k])}
        if "fx" in rows:
            rows["fx"]["change_ppm"] = 1e6 * rows["fx"]["change"] / rows["fx"]["navcam_only"]
        out["navcam"][eye] = rows
    rise = None
    try:
        rise = nz["navcam"]["median_px"] / held["navcam"]["median_px"] - 1.0
    except (KeyError, TypeError, ZeroDivisionError):
        pass
    out["worst_navcam_z"] = worst
    out["beyond_limits"] = beyond
    out["navcam_residual_rise"] = rise
    out["agree"] = bool(worst <= AGREE_MAX_Z and not beyond and (rise is None or rise <= AGREE_MAX_NAV_RESIDUAL_RISE))
    return out


def navcam_only_law(st: Dict[str, Any]) -> Dict[str, Any]:
    """The promoted Navcam law in the shape of :func:`mppp.calib.tied.globals_result` (no sd)."""
    out = {}
    for eye in ("NL", "NR"):
        G, S = st["G"][eye], st["S"][eye]
        out[eye] = {"params_T0": dict(zip(T.TIED_PARAMS, map(float, G))),
                    "fx_ppm_per_degC": float(1e6 * S[0] / G[0]), "fy_ppm_per_degC": float(1e6 * S[1] / G[1]),
                    "cx_px_per_degC": float(S[2]), "cy_px_per_degC": float(S[3])}
    return out


def fit_nzjoint(blocks: Dict[str, PathLike], model: Optional[Dict[str, Any]] = None,
                navcam_dir: Optional[PathLike] = None, zoom: int = Z.ZOOM, nav_points: int = Z.NAV_POINTS_PER_BLOCK,
                z_points: int = Z.Z_POINTS_PER_BLOCK, iterations: int = 100, passes: int = 2, verbose: bool = True,
                scapes: Optional[Sequence[Any]] = None, st: Optional[Dict[str, Any]] = None,
                navcam_blocks: Optional[Dict[str, PathLike]] = None,
                navcam_points: int = NAVCAM_ONLY_POINTS_PER_BLOCK) -> Dict[str, Any]:
    """The Mastcam-Z joint first (Navcam held: the reference), then from the same merged blocks the combined
    adjustment (both laws free), and their comparison (:func:`compare`).  ``navcam_blocks`` (0.93): finished
    Navcam-only blocks {name: WORK} added to both adjustments with ``navcam_points`` points each, so the combined
    Navcam law is held by every Navcam block, not only by the Mastcam-Z blocks' Navcam frames (a block whose name is
    one of ``blocks`` is left out: its Navcam frames are there already)."""
    from ..sfm.project import zcam_focus_model_path
    if model is None:
        model = json.loads(zcam_focus_model_path(zoom=zoom).read_text(encoding="utf-8"))
    t = time.time()
    extra = {n: w for n, w in (navcam_blocks or {}).items() if n not in (blocks or {})}
    pts: Any = nav_points
    if extra:
        if scapes is None:
            scapes = [J.load_scape(n, w, single=False) for n, w in blocks.items()]
        scapes = list(scapes) + [J.load_scape(n, w, single=False) for n, w in extra.items()]
        pts = {n: int(navcam_points) for n in extra}
        pts["*"] = nav_points
        if verbose:
            log.info(f"N+Z joint: {len(extra)} Navcam-only blocks added ({navcam_points} points each): "
                     f"{', '.join(sorted(extra))}")
    rec, proj, idx, st, units, zunits, sg = Z.build(blocks, model, navcam_dir, zoom, pts, z_points,
                                                    scapes=scapes, st=st)
    sg["rig_fit"] = Z.ZJOINT_RIG_FIT                # 0.95.1: as the Mastcam-Z joint (rig translation refined)
    fixed = Z.fixed_globals(sg)                     # 0.93: the zoom's fixed Mastcam-Z terms, held in both steps
    zhold = {e: sorted(v) for e, v in fixed.items()}
    if not zunits:
        raise ValueError(f"no Mastcam-Z {zoom} mm bins in the blocks")
    if verbose:
        log.info(f"N+Z joint: {len(blocks or scapes or [])} blocks, {rec.num_reg_images()} images, "
                 f"{rec.num_points3D()} points; {len(units['cameras'])} Navcam bins, {len(zunits)} Mastcam-Z bins "
                 f"({time.time() - t:.0f} s)")
    # 1) the reference: Navcam held at its law, the Mastcam-Z law free
    t = time.time()
    rec_h, proj_h = copy.deepcopy(rec), copy.deepcopy(proj)
    _, gz_h, _ = Z.adjust_z(rec_h, proj_h, idx, st, units, zunits, sg, iterations=iterations, passes=passes,
                               covariance=False, verbose=verbose, hold=zhold)
    gn0 = {"G_NL": st["G"]["NL"], "G_NR": st["G"]["NR"], "S_NL": st["S"]["NL"], "S_NR": st["S"]["NR"],
           "E_NL": st["E"]["NL"], "E_NR": st["E"]["NR"], "Q0": st["Q0"], "D": st["D"], "H": st["H"]}
    held_stats = family_stats(rec_h, proj_h, units, idx, gn0)
    if verbose:
        log.info(f"  Navcam held: {_stats_line(held_stats)} ({time.time() - t:.0f} s)")
    # 2) both laws free, from the reference solution (the Mastcam-Z law starts where step 1 ended)
    t = time.time()
    sg2 = copy.deepcopy(sg)
    for eye, arr in gz_h.items():
        sg2["eyes"][eye]["G"] = arr.copy()
    ba, gn, gz = adjust_nz(rec_h, proj_h, idx, st, units, zunits, sg2, iterations=iterations, passes=passes,
                           verbose=verbose, zhold=zhold)
    nz_stats = family_stats(rec_h, proj_h, units, idx, gn)
    nav_res = T.globals_result(st, gn, ba, units)
    z_res = Z.globals_result(sg2, gz, ba, zunits)
    res: Dict[str, Any] = {"zoom": zoom, "blocks": sorted({u["scape"] for u in zunits.values()}),
                           "navcam_only_blocks": sorted(extra), "fixed_terms": fixed,
                           "navcam": nav_res, "zcam": z_res, "zcam_navcam_held": {e: dict(zip(Z.GLOBAL_TERMS, map(float, a)))
                                                                                 for e, a in gz_h.items()},
                           "rig_zcam": Z.rig_result(proj_h, sg2, zoom, ba), "final_cost": float(ba["final_cost"]),
                           "brief": ba["brief"], "seconds": time.time() - t, "navcam_models": str(st["folder"]),
                           "comparison": compare(navcam_only_law(st), nav_res, held_stats, nz_stats)}
    if verbose:
        for line in describe(res):
            log.info(line)
    return res


def _stats_line(s: Dict[str, Dict[str, float]]) -> str:
    return "; ".join(f"{k} median {v['median_px']:.3f} / rms {v['rms_px']:.3f} px ({v['observations']} obs)"
                     for k, v in s.items())


def describe(res: Dict[str, Any]) -> List[str]:
    c = res["comparison"]
    out = [f"residuals, Navcam held: {_stats_line(c['residuals']['navcam_held'])}",
           f"residuals, both free:   {_stats_line(c['residuals']['both_free'])}"]
    for eye, rows in c["navcam"].items():
        parts = []
        for k in ("fx", "fy", "cx", "cy"):
            r = rows.get(k)
            if r:
                parts.append(f"{k} {r['change']:+.3f} px" + (f" ({r['z']:+.1f} sd)" if r.get("z") is not None else ""))
        for k in ("fx_ppm_per_degC", "fy_ppm_per_degC"):
            r = rows.get(k)
            if r:
                parts.append(f"{k} {r['navcam_only']:+.1f} -> {r['combined']:+.1f}")
        out.append(f"Navcam {eye} combined minus Navcam-only: " + ", ".join(parts))
    rise = c.get("navcam_residual_rise")
    out.append(f"=> {'the combined solution agrees with the Navcam-only law' if c['agree'] else 'the Mastcam-Z frames pull the Navcam law'}"
               f" (largest Navcam change {c['worst_navcam_z']:.1f} sd, limit {AGREE_MAX_Z:g}; beyond "
               f"{AGREE_MAX_F_PPM:g} ppm / {AGREE_MAX_PP_PX:g} px: {', '.join(c.get('beyond_limits') or []) or 'none'}; "
               "Navcam residual median " + (f"{100 * rise:+.1f} %" if rise is not None else "n/a") + ")")
    for eye, tm in (res["zcam"].get("eyes") or {}).items():
        t0 = res["zcam_navcam_held"].get(eye) or {}
        t1 = tm["terms"]
        out.append(f"Mastcam-Z {eye}: fx {t0.get('a_fx_backlash', float('nan')):.2f} -> {t1['a_fx_backlash']:.2f} px, "
                   f"k1 {t0.get('k1', float('nan')):+.4f} -> {t1['k1']:+.4f}, k2 {t0.get('k2', float('nan')):+.4f} -> {t1['k2']:+.4f}")
    return out


LAW_FOLDER = "navcam_law"                  # 0.95.0: the combined Navcam law's files, for calib lawcheck


def write_nzjoint(res: Dict[str, Any], out_dir: PathLike) -> Dict[str, Any]:
    """``nz_joint.json`` and ``report.md`` in ``out_dir`` (nothing promoted), and (0.95.0, Christian, 29 Oct) the
    combined Navcam law as model files in ``out_dir/navcam_law/`` (:func:`mppp.calib.tied.write_law_files`), so that
    ``calib lawcheck --law combined=<out_dir>/navcam_law`` can test it on the Navcam-only blocks against the law in
    use - the question being whether the law the Mastcam-Z frames pull to costs the Navcam blocks anything."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    law = None
    nav = res.get("navcam") or {}
    if isinstance(nav.get("NL"), dict) and isinstance(nav.get("rig"), dict):
        try:
            law = T.write_law_files(dict(nav, blocks=list(res.get("blocks") or []) + list(res.get("navcam_only_blocks") or []),
                                         residuals=(res.get("comparison") or {}).get("residuals")),
                                    out / LAW_FOLDER, note=f"the Navcam law of the Navcam + Mastcam-Z {int(res['zoom'])} mm "
                                    "combined joint (a check: not promoted)",
                                    method="MPPP Navcam + Mastcam-Z combined joint (mppp.calib.nzjoint.fit_nzjoint")
            res["navcam_law_folder"] = str(law)
        except Exception as e:                       # noqa: BLE001 - the comparison stands without the files
            log.warning(f"N+Z joint: the combined Navcam law files were not written ({type(e).__name__}: {e})")
            res["navcam_law_folder_error"] = f"{type(e).__name__}: {e}"
    (out / "nz_joint.json").write_text(json.dumps(res, indent=1, default=T._json_default), encoding="utf-8")
    lines = [f"# Navcam + Mastcam-Z {int(res['zoom'])} mm combined joint ({datetime.date.today().isoformat()})", "",
             f"Blocks: {', '.join(res['blocks'])}; started from the Navcam law in `{res.get('navcam_models')}`. "
             + (f"Navcam-only blocks added ({len(res['navcam_only_blocks'])}): {', '.join(res['navcam_only_blocks'])}. "
                if res.get("navcam_only_blocks") else "")
             + "A check of the Navcam-only law against the Mastcam-Z frames: nothing is promoted.", ""]
    lines += [f"- {x}" for x in describe(res)]
    if law is not None:
        lines += ["", f"The combined Navcam law is in `{law}`: test it on the Navcam-only blocks with "
                      f"`mppp calib lawcheck --law in_use= --law combined={law}`."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"folder": str(out), "agree": res["comparison"]["agree"], "navcam_law": str(law) if law else None}
