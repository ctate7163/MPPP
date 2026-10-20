"""
Mastcam-Z 34 mm calibration from solved Navcam + Mastcam-Z blocks (MPPP v0p74).

The blocks (``..._zcam`` WORK folders, notebook 03) are read from disk without loading their points:
``project.json`` (cameras, settings: focus lines, Mastcam-Z rig, backlash classification), ``sparse/cahv_ba/
cameras.bin`` (refined cameras), ``run_done.json`` and ``health/health.json``.

* :func:`block_info` - one block: its focus-bin cameras (refined), Navcam focal scale, Mastcam-Z lens temperature,
  rig, health checks;
* :func:`grade_block` / :func:`select_blocks` - an automatic grade (pass / fail with reasons and a score) and the
  2-3 best blocks for the calibration (passing, highest score, different sols);
* :func:`low_backlash_groups` - the focus groups each block classified low-backlash (for the found list);
* :func:`start_model_from_blocks` - a start focus model from blocks solved before v0p74 (problematic as they are: fx,
  fy lines per eye from the refined bins, divided by each block's Navcam scale; the regular-state ratio);
* :func:`calibrate` - the calibration from v0p74 blocks: per eye and focus state one line in focus count for fx, fy,
  cx, cy (ZR), k1, k2, the weighted mean over the selected blocks with its scatter and its trend against the lens
  temperature (applied when significant), and the stereo rig (rotation, baseline) - written as a focus model that
  notebook 03 starts from (``lines``, ``rig``) plus a report;
* :func:`check_model` - sanity limits before a model is promoted;
* :func:`export_point_cloud` - a block's points as PLY (colours of the native model).
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
FULL = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
LINE_PARAMS = {"ZL": ("fx", "fy", "k1", "k2"), "ZR": ("fx", "fy", "cx", "cy", "k1", "k2")}

# grading (v0p74): what makes a block good enough to calibrate Mastcam-Z from
GRADE = {"min_registered_fraction": 0.95, "max_beyond_ground": 0.05, "max_residual_median_px": 0.40,
         "max_edge_to_centre": 1.6,          # v0p74.9: 1.5 -> 1.6 (Rio Chiquito 1.56, South Arm 1.54 failed on 5 Oct)
         "min_corner_triangulated": 0.4, "min_bins_per_eye": 5,
         "fail_checks_allowed": ("focal_change_pct", "ray_displacement_max_px", "within_station_shift_spread_m",
                                 "excluded_fraction", "attitude_residual_p95_deg")}
# sanity limits of a calibrated model against the start model (v0p74)
MODEL_LIMITS = {"max_f0_change_pct": 1.5, "max_aspect_dev": 0.01, "max_k1": 1.5, "baseline_m": (0.20, 0.29),
                "max_rig_change_deg": 1.0, "max_aspect_change": 0.003,
                "max_offset_scatter_pct": 0.5,      # start update: the blocks' f offsets about the model's f0
                "max_rig_scatter_mdeg": 150.0,      # calibration: the blocks' rig rotations about their mean
                "max_baseline_scatter_mm": 3.0,
                # v0p74.9 (the 5 Oct promotion had regular-state temperature terms of +8 px/degC and k2 -2.07):
                "max_T_slope_px_per_degC": 2.0, "max_T_slope_pp_px_per_degC": 0.5, "max_tau_f_px": 10.0,
                "max_regular_k_dev": {"k1": 0.05, "k2": 0.5}, "min_blocks": 2,
                "min_blocks_calibration": 3}


from mppp.calib.fit import (REGULAR_RATIO_DEFAULT, block_info, check_zcam as check_model, fit_zcam, huber_fit,  # noqa: F401
                            ols as _ols, pp_shift_to_rig, regular_ratio, weighted_mean as _wmean, zcam_leave_one_out as leave_one_out)


def calibrate(infos, model, **kw):
    """The calibration of :func:`mppp.calib.fit.fit_zcam` (``form``, ``regular`` and the temperature options of the
    earlier versions are gone: the form is fixed)."""
    return fit_zcam(infos, model, rig_translation=kw.get("rig_translation", "cahv"))


def _huber(X, y, w, k=1.345, floor=0.05):
    b, se, sig, _ = huber_fit(X, y, w, k, floor)
    return b, se, sig


def _json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def project_root(work: PathLike) -> Path:
    w = Path(work)
    return w / "colmap" if (w / "colmap" / "project.json").is_file() else w


def _check_value(info: Dict[str, Any], name: str) -> Optional[float]:
    c = info["checks"].get(name)
    try:
        return float(c["value"]) if c and c.get("value") is not None else None
    except (TypeError, ValueError):
        return None


def grade_block(info: Dict[str, Any], limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Is this block good enough to calibrate Mastcam-Z from?  Fails on: no finished run; health FAIL in a check other
    than the Mastcam-Z start-model ones (``fail_checks_allowed``: focal change, ray displacement - the start model is
    what is being calibrated); registered fraction, beyond-ground share, Navcam residual median, Navcam corners
    (edge-to-centre ratio, corner tie points) outside ``GRADE``; fewer than ``min_bins_per_eye`` refined backlash bins
    per eye.  Score (higher is better): registered fraction, small residual, many Mastcam-Z bins, a focus span, good
    corners.  Returns {"ok", "score", "reasons", "name"}.
    """
    lim = dict(GRADE, **(limits or {}))
    reasons = []
    rd = info.get("run_done") or {}
    if not rd:
        reasons.append("no run_done.json (the run did not finish)")
    bad = sorted(k for k, ss in info["statuses"].items() if "fail" in ss and k not in lim["fail_checks_allowed"])
    if bad:
        reasons.append("health FAIL: " + ", ".join(bad))
    v = {k: _check_value(info, k) for k in ("registered_fraction", "beyond_ground_fraction", "residual_median_px",
                                            "edge_to_centre_residual", "corner_triangulated_ratio")}
    if v["registered_fraction"] is not None and v["registered_fraction"] < lim["min_registered_fraction"]:
        reasons.append(f"registered {v['registered_fraction']:.3f}")
    if v["beyond_ground_fraction"] is not None and v["beyond_ground_fraction"] > lim["max_beyond_ground"]:
        reasons.append(f"beyond-ground {v['beyond_ground_fraction']:.3f}")
    if v["residual_median_px"] is not None and v["residual_median_px"] > lim["max_residual_median_px"]:
        reasons.append(f"residual median {v['residual_median_px']:.3f} px")
    if v["edge_to_centre_residual"] is not None and v["edge_to_centre_residual"] > lim["max_edge_to_centre"]:
        reasons.append(f"Navcam corners: edge/centre {v['edge_to_centre_residual']:.2f}")
    if v["corner_triangulated_ratio"] is not None and v["corner_triangulated_ratio"] < lim["min_corner_triangulated"]:
        reasons.append(f"Navcam corners: corner tie points {v['corner_triangulated_ratio']:.2f}")
    per_eye = {}
    for b in info["bins"]:
        if b["state"] == "backlash" and "fx" not in b["fixed"]:
            per_eye.setdefault(b["group"][:2], []).append(b)
    for eye in ("ZL", "ZR"):
        if len(per_eye.get(eye, [])) < lim["min_bins_per_eye"]:
            reasons.append(f"{eye}: {len(per_eye.get(eye, []))} refined backlash bins")
    foc = [b["focus"] for bs in per_eye.values() for b in bs if b["focus"] is not None]
    span = float(np.ptp(foc)) if foc else 0.0
    nb = sum(len(x) for x in per_eye.values())
    score = (10.0 * (v["registered_fraction"] or 0.9) - 10.0 * (v["residual_median_px"] or 0.4)
             + 2.0 * math.log1p(nb) + span / 500.0 - 2.0 * max(0.0, (v["edge_to_centre_residual"] or 1.0) - 1.0)
             + 2.0 * (v["corner_triangulated_ratio"] or 0.5) - 20.0 * (v["beyond_ground_fraction"] or 0.0))
    return {"name": info["name"], "ok": not reasons, "score": float(score), "reasons": reasons,
            "bins": nb, "focus_span": span, "values": v, "sol": info.get("sol")}


def select_blocks(infos: Sequence[Dict[str, Any]], n: int = 3, min_sol_gap: float = 30.0,
                  limits: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The ``n`` best passing blocks (score), preferring blocks at least ``min_sol_gap`` sols apart; returns
    (selected infos, every grade)."""
    grades = [dict(grade_block(i, limits), _i=k) for k, i in enumerate(infos)]
    ok = sorted((g for g in grades if g["ok"]), key=lambda g: -g["score"])
    if n is None or n <= 0:                              # v0p74.9: every passing block (the default now)
        n = len(ok)
    sel: List[Dict[str, Any]] = []
    for g in ok:
        s = g.get("sol")
        if all(s is None or x.get("sol") is None or abs(s - x["sol"]) >= min_sol_gap for x in sel):
            sel.append(g)
        if len(sel) >= n:
            break
    for g in ok:                                         # fill up with the next best if the sols were too close
        if len(sel) >= n:
            break
        if g not in sel:
            sel.append(g)
    return [infos[g["_i"]] for g in sel], [{k: v for k, v in g.items() if k != "_i"} for g in grades]


def block_sequence_states(info: Dict[str, Any], shipped: Optional[Dict[str, Any]] = None,
                          z_min: float = 3.0) -> Dict[str, Dict[str, Any]]:
    """
    v0p74.8: the focus state of every sequence of a solved block (:func:`mppp.sfm.backlash.apply_sequence_states`) from
    its saved groups (``zcam_focus_states.json``).  Blocks classified with v0p74.8 carry the verdicts; older ones are
    pooled here from their groups' ratios (all fitted images) and midpoints, the groups whose images are all on the
    shipped list (``shipped``) left out.  Returns {"sol|sequence": {"z", "state", ...}}.
    """
    from mppp.sfm.backlash import apply_sequence_states, is_low_backlash
    rows = [dict(g) for g in (info.get("backlash_groups") or [])]
    if info.get("backlash_fits") and rows and all(g.get("names") is not None for g in rows):
        rows = reclassify_block(info, shipped=shipped, z_min=z_min)            # the v0p74.8 rules on the saved fits
        out = {}
        for g in rows:
            out.setdefault(f"{g.get('sol')}|{g.get('sequence')}", {"z": g.get("sequence_z"), "state": g.get("sequence_state")})
        return out
    if any("sequence_state" in g for g in rows):
        out: Dict[str, Dict[str, Any]] = {}
        for g in rows:
            out.setdefault(f"{g.get('sol')}|{g.get('sequence')}", {"z": g.get("sequence_z"), "state": g.get("sequence_state")})
        return out
    for g in rows:
        names = g.get("names") or []
        g["all_listed"] = bool(names) and shipped is not None and all(
            is_low_backlash({"stem": str(n).rsplit(".", 1)[0], "sol": g.get("sol"), "sequence": g.get("sequence"),
                             "camera_group": g.get("camera_group")}, shipped) for n in names)
        ok = g.get("ratio") is not None and g.get("threshold_ratio") is not None and not g["all_listed"]
        g["ratio_fit"], g["sd_fit"] = (g.get("ratio"), g.get("sd")) if ok else (None, None)
    return apply_sequence_states(rows, z_min=z_min)


def reclassify_block(info: Dict[str, Any], shipped: Optional[Dict[str, Any]] = None, z_min: float = 3.0,
                     report: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """v0p74.8: a solved block's focus groups classified again with the current rules (measured backlash line, one
    state per sequence, the shipped list's images as the exceptions) from the per-image fits it saved
    (``zcam_focus_states.json``: ``fits_before``, the groups' images).  Returns the group rows
    (:func:`mppp.sfm.backlash.classify_groups`)."""
    from types import SimpleNamespace
    from mppp.sfm.backlash import classify_groups, load_low_backlash
    imgs = []
    for g in info.get("backlash_groups") or []:
        for n in g.get("names") or []:
            imgs.append({"name": n, "stem": str(n).rsplit(".", 1)[0], "camera_group": g.get("camera_group"),
                         "eye": g.get("eye"), "sol": g.get("sol"), "sequence": g.get("sequence"),
                         "focus_count": g.get("focus_count"), "instrument": (g.get("cameras") or ["?"])[0]})
    model = {"cameras": {k: {"f0_px": float(v), "label_f0_px": 1.0} for k, v in (info.get("backlash_ratios") or {}).items()}}
    return classify_groups(SimpleNamespace(images=imgs, settings={}), info.get("backlash_fits") or {}, model,
                           z_min=z_min, nav_scale=info.get("backlash_nav_scale") or 1.0,
                           known=shipped if shipped is not None else load_low_backlash(None), report=report)


def low_backlash_groups(infos: Sequence[Dict[str, Any]], z_min: float = 3.0,
                        listed: Optional[Dict[str, Any]] = None,
                        shipped: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The focus groups the blocks classified low-backlash (regular): decided, plausible, at least ``z_min`` sd below
    the midpoint between the two states; ``listed`` (:func:`mppp.sfm.backlash.load_low_backlash`): groups whose images
    are all on that list already are left out.  v0p74.8: and groups of a sequence that is backlash as a whole
    (:func:`block_sequence_states`; ``shipped``: the shipped list, whose images are the known exceptions)."""
    from mppp.sfm.backlash import is_low_backlash
    out = []
    for i in infos:
        seq = block_sequence_states(i, shipped=shipped, z_min=z_min)
        groups = reclassify_block(i, shipped=shipped, z_min=z_min) if i.get("backlash_fits") else \
            (i.get("backlash_groups") or [])
        for g in groups:
            own = g.get("own_state", g.get("state"))
            own_z = g.get("own_z", g.get("z"))
            if own == "regular" and not g.get("implausible") and own_z is not None and float(own_z) <= -abs(z_min) \
                    and g.get("names"):
                if (seq.get(f"{g.get('sol')}|{g.get('sequence')}") or {}).get("state") == "backlash":
                    continue
                if listed is not None and all(is_low_backlash({"stem": str(n).rsplit(".", 1)[0], "sol": g.get("sol"),
                                                               "sequence": g.get("sequence"),
                                                               "camera_group": g.get("camera_group")}, listed)
                                              for n in g["names"]):
                    continue
                out.append(dict(g, block=i["name"]))
    return out


def found_contradicted(infos: Sequence[Dict[str, Any]], found_path: PathLike,
                       shipped: Optional[Dict[str, Any]] = None, z_min: float = 3.0) -> Dict[str, Any]:
    """
    v0p74.8: the found-list entries (``<SCAPES_ROOT>/zcam_low_backlash_found.csv``) whose sequence is backlash as a
    whole in the block that holds them (:func:`block_sequence_states`) and that are not on the shipped list.  Returns
    {"remove": [(stem, block, "sol|sequence", z)], "keep": [stem, ...], "unknown": [stem, ...]} (unknown: no block
    holds the image).
    """
    from mppp.sfm.backlash import is_low_backlash
    f = Path(found_path)
    stems = []
    if f.is_file():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            tok = line.split("#", 1)[0].strip()
            if tok:
                stems.append(tok)
    where: Dict[str, Tuple[str, str, Any]] = {}
    for i in infos:
        seq = block_sequence_states(i, shipped=shipped, z_min=z_min)
        for g in i.get("backlash_groups") or []:
            k = f"{g.get('sol')}|{g.get('sequence')}"
            for n in g.get("names") or []:
                st = str(n).rsplit(".", 1)[0].upper()
                v = seq.get(k) or {}
                if st not in where or v.get("state") == "backlash":
                    where[st] = (i["name"], k, v.get("state"), v.get("z"), g)
    out: Dict[str, Any] = {"remove": [], "keep": [], "unknown": []}
    for s in stems:
        w = where.get(s.upper())
        if w is None:
            out["unknown"].append(s)
            continue
        name, k, st, z, g = w
        on_shipped = shipped is not None and is_low_backlash({"stem": s, "sol": g.get("sol"), "sequence": g.get("sequence"),
                                                              "camera_group": g.get("camera_group")}, shipped)
        if st == "backlash" and not on_shipped:
            out["remove"].append((s, name, k, z))
        else:
            out["keep"].append(s)
    return out


def start_model_from_blocks(infos: Sequence[Dict[str, Any]], model: Dict[str, Any], reference_focus: float = 600.0,
                            note: str = "") -> Dict[str, Any]:
    """
    v0p74: a start focus model from blocks solved before v0p74: per eye, one line fx, fy = a + b (focus - ref) over
    the refined (free) backlash-state bins of every block, each divided by its block's Navcam focal scale, with one
    offset per block (Huber, weights = images); the model's f0 is the offsets' mean (block offsets and their scatter
    are reported), ``aspect`` fy / fx; the regular-state ratio from the blocks' classified groups.  Everything else
    (distortion, principal point, rig, thermal) is kept from ``model``.  Returns the new model (a dict in the focus
    model's format) with ``start_update`` describing the fit.
    """
    import copy
    out = copy.deepcopy(model)
    rep: Dict[str, Any] = {"blocks": [i["name"] for i in infos], "eyes": {}}
    for g, gm in (out.get("cameras") or {}).items():
        rows = [(i, b) for i in infos for b in i["bins"]
                if b["group"] == g and b["state"] == "backlash" and "fx" not in b["fixed"] and b["focus"] is not None]
        blocks = sorted({i["name"] for i, _ in rows})
        if len(rows) < 6 or not blocks:
            rep["eyes"][g] = {"skipped": f"{len(rows)} usable bins"}
            continue
        res = {}
        for k, pn in ((0, "fx"), (1, "fy")):
            y = np.array([b["params"][k] / (i["navcam_scale"] or 1.0) for i, b in rows])
            F = np.array([b["focus"] - reference_focus for _, b in rows], float)
            w = np.array([b["n_images"] for _, b in rows], float)
            D = np.array([[1.0 if i["name"] == bl else 0.0 for bl in blocks] for i, _ in rows])
            X = np.c_[D, F]
            beta, se, sig = _huber(X, y, w, floor=0.3)
            offs = beta[:-1]
            res[pn] = {"a": float(np.mean(offs)), "b": float(beta[-1]), "sd_b": float(se[-1]),
                       "block_offsets": dict(zip(blocks, map(float, offs))), "scatter_offsets": float(np.std(offs)),
                       "robust_sd": float(sig), "bins": len(rows)}
        asp = res["fy"]["a"] / res["fx"]["a"]
        consistent = gm.get("aspect") is None or abs(asp - float(gm["aspect"])) <= MODEL_LIMITS["max_aspect_change"]
        if consistent:
            f0 = float(np.sqrt(res["fx"]["a"] * res["fy"]["a"]))
            slope = 0.5 * (res["fx"]["b"] + res["fy"]["b"])
            offs = {bl: float(np.sqrt(res["fx"]["block_offsets"][bl] * res["fy"]["block_offsets"][bl])) for bl in blocks}
        else:
            # fx and fy of the bins disagree (blocks solved on bad Navcam corners inflate fy): fx is the one to trust
            # - f0 from fx with the aspect in use; the slope in use is kept (the blocks refit their lines anyway)
            f0 = float(res["fx"]["a"] * math.sqrt(float(gm["aspect"])))
            slope = float(gm.get("slope_px_per_count") or 0.5 * (res["fx"]["b"] + res["fy"]["b"]))
            offs = {bl: float(res["fx"]["block_offsets"][bl] * math.sqrt(float(gm["aspect"]))) for bl in blocks}
        rep["eyes"][g] = {"f0_was": gm.get("f0_px"), "f0_now": f0, "slope_was": gm.get("slope_px_per_count"),
                          "slope_now": slope, "slope_fitted": 0.5 * (res["fx"]["b"] + res["fy"]["b"]),
                          "aspect_was": gm.get("aspect"), "aspect_now": asp, "fit": res,
                          "scatter_offsets_pct": 100.0 * float(np.std(list(offs.values()))) / f0}
        gm["f0_px"] = f0
        gm["reference_focus"] = float(reference_focus)
        gm["slope_px_per_count"] = slope
        if consistent:
            gm["slope_sd_px_per_count"] = 0.5 * math.hypot(res["fx"]["sd_b"], res["fy"]["sd_b"])
            gm["aspect"] = asp
        else:                                   # fx, fy of single bins drift apart: keep the aspect and slope in use
            rep["eyes"][g]["aspect_kept"] = f"fitted {asp:.5f} differs from {float(gm['aspect']):.5f} by more than " \
                                            f"{MODEL_LIMITS['max_aspect_change']}: aspect and slope kept, f0 from fx"
        gm["block_offsets_px"] = {bl: v - f0 for bl, v in offs.items()}
        gm["n_bins"] = len(rows)
        if gm.get("trend"):                     # the v0p42 sol trend was carried by one block: not kept
            gm["trend_v0p42"] = gm.pop("trend")
    # the regular state: its fitted f / label f relative to the backlash state's, over the classified groups
    reg = [g for i in infos for g in i.get("backlash_groups") or [] if g.get("state") == "regular" and g.get("decided")]
    bk = [g for i in infos for g in i.get("backlash_groups") or [] if g.get("state") == "backlash" and g.get("decided")]
    if reg and bk:
        rr = float(np.median([g["ratio"] for g in reg]))
        rb = float(np.median([g["ratio"] for g in bk]))
        rep["state_ratio"] = {"regular_median": rr, "backlash_median": rb, "regular_groups": len(reg),
                              "backlash_groups": len(bk), "backlash_over_regular": rb / rr}
    out["start_update"] = dict(rep, when=datetime.date.today().isoformat(),
                               note=note or "v0p74: start model from the blocks solved before v0p74 (Navcam scale divided out)")
    out["source"] = f"{out.get('source', '')}; v0p74 start update from {len(infos)} blocks ({datetime.date.today()})"
    return out


def report_lines(model: Dict[str, Any], grades: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    """A Markdown report of a calibration (or start update) and the block grades."""
    out = [f"# Mastcam-Z 34 mm calibration ({model.get('state', '')})", "", model.get("source", ""), ""]
    if grades:
        out += ["## Block grades", "", "| block | sol | ok | score | bins | focus span | reasons |", "|---|---|---|---|---|---|---|"]
        for g in sorted(grades, key=lambda g: -g["score"]):
            out.append(f"| {g['name']} | {g.get('sol') or ''} | {'yes' if g['ok'] else 'no'} | {g['score']:.2f} | {g['bins']} | "
                       f"{g['focus_span']:.0f} | {'; '.join(g['reasons'])} |")
        out.append("")
    cal = model.get("calibration")
    if cal:
        out += ["## Focus lines (p = a + b (focus - reference) [+ c (T - T0)])", "",
                "| eye | state | param | a | sd a | tau a | chi2/dof | b (per count) | sd b | T slope (per degC) | z | applied | blocks |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g, L in cal["eyes"].items():
            if "skipped" in L:
                out.append(f"| {g} | | | {L['skipped']} |||||||||")
                continue
            for st in ("backlash", "regular"):
                for pn, e in (L.get(st) or {}).items():
                    t = e.get("temperature") or {}
                    out.append(f"| {g} | {st} | {pn} | {e['a']:.4f} | {e['sd_a']:.4f} | {e['tau_a']:.4f} | {e['chi2_dof_a']:.2f} | "
                               f"{e['b']:+.6f} | {e['sd_b']:.6f} | {t.get('c_per_degC', float('nan')):+.4f} | "
                               f"{(t.get('z') if t.get('z') is not None else float('nan')):+.1f} | {t.get('applied', False)} | {e['n_blocks']} |")
        out.append("")
        if cal.get("regular_ratio"):
            out += ["Regular (low-backlash) state derived from the backlash lines (v0p74.9): fx, fy divided by "
                    + ", ".join(f"{g} {v['ratio']:.5f} ({v['source']}, {v['n_blocks']} blocks)" for g, v in cal["regular_ratio"].items()), ""]
        if cal.get("loo"):
            out += ["## Leave one out (backlash offsets at the reference focus)", "",
                    "| eye | param | blocks | rms (px) | max (px) | per block (residual px) |", "|---|---|---|---|---|---|"]
            for g, P in cal["loo"].items():
                for pn, e in P.items():
                    out.append(f"| {g} | {pn} | {e['n']} | {e['rms_px']:.2f} | {e['max_px']:.2f} | "
                               + ", ".join(f"{p['block'].replace('mars2020_', '').replace('_colmap_zcam', '')} {p['residual_px']:+.1f}"
                                           for p in e["per_block"]) + " |")
            out.append("")
        if cal.get("rig"):
            r = cal["rig"]
            out += ["## Stereo rig (right from left)", "",
                    f"- rotation vector {', '.join(f'{x:+.2f}' for x in r['rotvec_mdeg'])} mdeg"
                    + (f" (sd {', '.join(f'{x:.2f}' for x in r['sd_rotvec_mdeg'])})" if r.get("sd_rotvec_mdeg") else ""),
                    f"- baseline {r['baseline_m'] * 1000:.2f} mm" + (f" (sd {r['sd_baseline_mm']:.2f} mm)" if r.get("sd_baseline_mm") else ""),
                    f"- blocks: {', '.join(r['blocks'])}", ""]
            if r.get("per_block"):
                out += ["| block | sol | T optics | rotation vector (mdeg) | baseline (mm) | ZR pp offset moved into the rig (px) |",
                        "|---|---|---|---|---|---|"]
                for p in r["per_block"]:
                    T = p.get("T_optics_degC")
                    sh = p.get("pp_shift_px")
                    out.append(f"| {p['block']} | {p.get('sol')} | {'' if T is None else f'{T:.1f}'} | "
                               f"{', '.join(f'{x:+.1f}' for x in p['rotvec_mdeg'])} | {p['baseline_mm']:.2f} | "
                               f"{'' if sh is None else f'{sh[0]:+.2f}, {sh[1]:+.2f}'} |")
                out.append("")
            bc = r.get("baseline_vs_cahv")
            if bc:
                vs, vt = bc.get("vs_sol_per_1000") or {}, bc.get("vs_T") or {}
                out += [f"- refined baseline minus CAHV over {bc['n']} blocks: {bc['mean_mm']:+.2f} mm"
                        + (f" (sd {bc['sd_mm']:.2f} mm)" if bc.get("sd_mm") is not None else "")
                        + (f"; vs sol {vs['slope']:+.2f} +- {vs['se']:.2f} mm/1000 sol" if vs.get("slope") is not None else "")
                        + (f"; vs lens T {vt['slope']:+.3f} +- {vt['se']:.3f} mm/degC" if vt.get("slope") is not None else ""), ""]
            fits = (r.get("residuals") or {}).get("fits") or {}
            if fits:
                out += ["Rotation residuals of the blocks' rigs about the mean (mdeg):", ""]
                for ang, f in fits.items():
                    vt, vs = f.get("vs_T") or {}, f.get("vs_sol") or {}
                    out.append(f"- {ang}: sd {f.get('sd_mdeg') or float('nan'):.1f} mdeg"
                               + (f"; vs lens T {vt['slope']:+.2f} +- {vt['se']:.2f} mdeg/degC (z {vt.get('z') or 0:+.1f})" if vt.get("slope") is not None else "")
                               + (f"; vs sol {vs['slope']:+.2f} +- {vs['se']:.2f} mdeg/1000 sol (z {vs.get('z') or 0:+.1f})" if vs.get("slope") is not None else ""))
                out.append("")
            p2r = cal.get("pp_to_rig") or {}
            if p2r.get("pp0_px"):
                out += [f"- ZR principal point at the reference focus fixed at pp0 {p2r['pp0_px']} px; each block's offset "
                        f"moved into its rig rotation before averaging (the two are degenerate in a block's BA)", ""]
    su = model.get("start_update")
    if su and su.get("eyes"):
        out += ["## Start update (blocks before v0p74)", ""]
        for g, e in su["eyes"].items():
            if "skipped" in e:
                out.append(f"- {g}: {e['skipped']}")
            else:
                out.append(f"- {g}: f0 {e['f0_was']:.1f} -> {e['f0_now']:.1f} px, slope {e['slope_was']:.4f} -> "
                           f"{e['slope_now']:.4f} px/count, aspect {e['aspect_was']:.5f} -> {e['aspect_now']:.5f}; block "
                           f"offsets sd {e['fit']['fx']['scatter_offsets']:.1f} px")
        if su.get("state_ratio"):
            s = su["state_ratio"]
            out.append(f"- backlash / regular state f ratio {s['backlash_over_regular']:.4f} ({s['regular_groups']} regular, "
                       f"{s['backlash_groups']} backlash groups)")
        out.append("")
    return out


def export_point_cloud(work: PathLike, out: Optional[PathLike] = None) -> Dict[str, Any]:
    """A block's points as PLY (the native model with colours, ``gui_native/sparse``, else ``sparse/cahv_ba``), in
    the project's world frame (site ENU minus ``offset``, written beside it as JSON).  Returns {"ply", "points",
    "offset", "frame"}."""
    import pycolmap
    root = project_root(work)
    src = next((d for d in (root / "gui_native" / "sparse", root / "sparse" / "cahv_ba") if (d / "points3D.bin").is_file()
                or (d / "points3D.txt").is_file()), None)
    if src is None:
        raise FileNotFoundError(f"no solved points in {root}")
    rec = pycolmap.Reconstruction(str(src))
    out = Path(out) if out else root / "pointcloud" / f"{Path(work).name}.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec.export_PLY(str(out))
    proj = _json(root / "project.json") or {}
    meta = {"ply": str(out), "points": int(rec.num_points3D()), "source": str(src), "offset": proj.get("offset"),
            "frame": (proj.get("settings") or {}).get("world_frame"),
            "note": "coordinates = world frame minus offset (add offset for site coordinates)"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta
