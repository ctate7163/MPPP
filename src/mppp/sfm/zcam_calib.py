"""
Mastcam-Z 34 mm calibration from solved Navcam + Mastcam-Z blocks (MPPP v0p74).

The blocks (``..._colmap_zcam`` WORK folders, notebook 03) are read from disk without loading their points:
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


def _json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def project_root(work: PathLike) -> Path:
    w = Path(work)
    return w / "colmap" if (w / "colmap" / "project.json").is_file() else w


def block_info(work: PathLike, model: str = "cahv_ba") -> Dict[str, Any]:
    """Everything this module needs of one solved block (no points are read).  Raises FileNotFoundError without a
    solution."""
    from .navcal_consensus import _read_cameras_bin
    root = project_root(work)
    proj = _json(root / "project.json")
    cb = root / "sparse" / model / "cameras.bin"
    if proj is None or not cb.is_file():
        raise FileNotFoundError(f"no solved project in {root}")
    st = proj.get("settings") or {}
    db = (st.get("database") or {}).get("cameras") or {}
    refined = _read_cameras_bin(cb)
    bins, nav = [], []
    for key, cam in (proj.get("cameras") or {}).items():
        cid = db.get(key)
        p = refined.get(int(cid)) if cid is not None else None
        if p is None:
            continue
        if str(key).startswith("N") and cam.get("params"):
            p0 = np.asarray(cam["params"], float)
            nav.append((float(np.sqrt(p[0] * p[1]) / np.sqrt(p0[0] * p0[1])), int(cam.get("n_images") or 1)))
        if str(key).startswith("Z") and cam.get("group"):
            bins.append({"camera": key, "group": cam["group"], "focus": cam.get("focus_count_median"),
                         "state": "regular" if (cam.get("backlash_state") == "regular" or key.endswith("_reg")) else "backlash",
                         "params": [float(x) for x in p], "start": [float(x) for x in cam["params"]],
                         "fixed": list(cam.get("fixed_params") or []), "n_images": int(cam.get("n_images") or 1)})
    ns = (sum(r * w for r, w in nav) / sum(w for _, w in nav)) if nav else None
    imgs = proj.get("images") or []
    To = [float(r["zcam_optics_temperature_degC"]) for r in imgs
          if str(r.get("instrument", ""))[:1] == "Z" and r.get("zcam_optics_temperature_degC") is not None]
    Tf = [float(r["camera_temperature_degC"]) for r in imgs
          if str(r.get("instrument", ""))[:1] == "Z" and r.get("camera_temperature_degC") is not None]
    sols = [float(r["sol"]) for r in imgs if str(r.get("instrument", ""))[:1] == "Z" and r.get("sol") is not None]
    health = _json(root / "health" / "health.json") or {}
    checks = {c.get("check"): c for c in health.get("checks") or [] if isinstance(c, dict)}
    # several checks repeat per camera (focal_change_pct ...): keep every status of a name
    statuses: Dict[str, List[str]] = {}
    for c in health.get("checks") or []:
        if isinstance(c, dict):
            statuses.setdefault(c.get("check"), []).append(c.get("status"))
    states = _json(root / "zcam_focus_states.json") or {}
    return {"work": str(Path(work)), "root": str(root), "name": Path(work).name, "run_done": _json(root / "run_done.json"),
            "health_verdict": health.get("verdict"), "checks": checks, "statuses": statuses, "bins": bins,
            "navcam_scale": ns, "T_optics_degC": float(np.mean(To)) if To else None,
            "T_fpa_degC": float(np.mean(Tf)) if Tf else None, "sol": float(np.median(sols)) if sols else None,
            "lines": st.get("zcam_focus_lines") or {}, "rig": st.get("zcam_virtual_rig") or {},
            "rig_fit": st.get("zcam_rig_fit"), "backlash_groups": states.get("groups") or [],
            "backlash_fits": states.get("fits_before") or {}, "backlash_nav_scale": states.get("nav_scale"),
            "backlash_ratios": states.get("backlash_ratios") or {},
            "line_params": st.get("zcam_line_params"), "mppp": (_json(root / "run_done.json") or {}).get("mppp")}


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
    from .backlash import apply_sequence_states, is_low_backlash
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
    from .backlash import classify_groups, load_low_backlash
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
    from .backlash import is_low_backlash
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
    from .backlash import is_low_backlash
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


def _huber(X: np.ndarray, y: np.ndarray, w: np.ndarray, k: float = 1.345, floor: float = 0.05):
    wt = np.ones(len(y))
    b = np.zeros(X.shape[1])
    sig = floor
    for _ in range(30):
        sw = np.sqrt(w * wt)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = y - X @ b
        sig = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), floor)
        u = np.abs(r) / (k * sig)
        wt = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
    W = w * wt / max(np.mean(w * wt), 1e-12)
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(np.sum(W * (y - X @ b) ** 2) / dof)
    try:
        cov = np.linalg.inv(X.T @ (X * W[:, None])) * s2
    except np.linalg.LinAlgError:
        cov = np.full((X.shape[1], X.shape[1]), np.nan)
    return b, np.sqrt(np.clip(np.diag(cov), 0, None)), sig


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


def _ols(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """v0p75: slope, se and z of y against x (rows with a finite x), as the Navcam rig-residual tests."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) <= 0:
        return {"n": int(ok.sum()), "slope": None, "se": None, "z": None}
    X = np.c_[np.ones(ok.sum()), x[ok] - x[ok].mean()]
    c, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    r = y[ok] - X @ c
    s2 = float(np.sum(r ** 2) / max(ok.sum() - 2, 1))
    se = float(np.sqrt(s2 / np.sum((x[ok] - x[ok].mean()) ** 2)))
    return {"n": int(ok.sum()), "mean_x": float(x[ok].mean()), "intercept": float(c[0]), "slope": float(c[1]), "se": se,
            "z": float(c[1] / se) if se > 0 else None, "rms": float(np.sqrt(np.mean(r ** 2)))}


def _wmean(y: np.ndarray, sd: np.ndarray) -> Dict[str, float]:
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nanmedian(np.where(sd > 0, sd, np.nan)) if np.any(sd > 0) else 1.0)
    w = 1.0 / sd ** 2
    m = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - m) ** 2))
    dof = max(len(y) - 1, 1)
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - dof) / c) if c > 0 else 0.0
    w2 = 1.0 / (sd ** 2 + tau2)
    m2 = float(np.sum(w2 * y) / np.sum(w2))
    return {"value": m2, "se": float(np.sqrt(1.0 / np.sum(w2))), "tau": float(np.sqrt(tau2)), "chi2_dof": q / dof,
            "n": int(len(y))}


def pp_shift_to_rig(R_sensor_from_ref: np.ndarray, t_sensor_from_ref: np.ndarray, f_px: float,
                    dcx_px: float, dcy_px: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    The rig rotation that replaces a principal-point offset of the right (sensor) eye: a camera whose principal
    point is moved by (dcx, dcy) px sees the same pixels as the camera with the original principal point rotated
    about its own centre by (wx, wy) = (-dcy, dcx) / f (small angles, distant points).  Returns (R', t') with
    x_sensor' = R_d (R x_ref + t): R' = R_d R, t' = R_d t - the baseline is unchanged.
    """
    from scipy.spatial.transform import Rotation
    Rd = Rotation.from_rotvec([-dcy_px / f_px, dcx_px / f_px, 0.0]).as_matrix()
    return Rd @ np.asarray(R_sensor_from_ref, float), Rd @ np.asarray(t_sensor_from_ref, float)


def _label_pp_at(info: Dict[str, Any], group: str, ref: float) -> Optional[Tuple[float, float]]:
    """The eye's start (label) principal point at the reference focus: a line through the bins' start values."""
    rows = [(b["focus"], b["start"][2], b["start"][3]) for b in info["bins"]
            if b["group"] == group and b["focus"] is not None and len(b["start"]) >= 4]
    if not rows:
        return None
    F = np.array([r[0] for r in rows], float) - ref
    out = []
    for k in (1, 2):
        y = np.array([r[k] for r in rows], float)
        if len(rows) >= 3 and np.ptp(F) > 0:
            out.append(float(np.polyval(np.polyfit(F, y, 1), 0.0)))
        else:
            out.append(float(np.median(y)))
    return out[0], out[1]


def regular_ratio(infos: Sequence[Dict[str, Any]], group: str, min_regular_groups: int = 3) -> Dict[str, Any]:
    """
    v0p74.9: the ratio of the backlash to the regular focal length of an eye, measured by the blocks' focus-state
    classification (``zcam_focus_states.json``: the decided groups' fitted / label ratios, both states in one block):
    the median over the blocks with at least ``min_regular_groups`` regular and 3 backlash groups of that eye of
    (median backlash ratio / median regular ratio); else :data:`REGULAR_RATIO_DEFAULT`.  Returns {"ratio", "n_blocks",
    "per_block", "source"}.
    """
    from .backlash import backlash_ratio_at, measured_backlash_lines
    per = []
    for i in infos:
        gs = [g for g in (i.get("backlash_groups") or []) if g.get("camera_group") == group and g.get("ratio") is not None
              and not g.get("implausible")]
        b = [g for g in gs if g.get("state") == "backlash" and g.get("decided")]
        r = [g for g in gs if g.get("state") == "regular" and g.get("decided")]
        if len(r) >= min_regular_groups and len(b) >= 3:
            # the backlash ratio at each regular group's own focus count (it rises with the focus count, about
            # 0.45 % per 1000 counts on ZR034; the regular groups sit mostly at low counts)
            rows = [dict(g, ratio_fit=g.get("ratio_fit", g["ratio"]), fitted_unlisted=g.get("fitted_unlisted", g.get("fitted")))
                    for g in b]
            ln = measured_backlash_lines(rows, float(i.get("backlash_nav_scale") or 1.0)).get(group)
            q = [backlash_ratio_at(ln, g.get("focus_count")) / float(g["ratio"]) for g in r] if ln else \
                [float(np.median([g["ratio"] for g in b])) / float(g["ratio"]) for g in r]
            per.append({"block": i["name"], "ratio": float(np.median(q)), "regular": len(r), "backlash": len(b),
                        "line": (ln or {}).get("source")})
    if per:
        return {"ratio": float(np.median([p["ratio"] for p in per])), "n_blocks": len(per), "per_block": per,
                "source": "measured (backlash / regular groups)"}
    return {"ratio": REGULAR_RATIO_DEFAULT, "n_blocks": 0, "per_block": [], "source": "default"}


def leave_one_out(infos: Sequence[Dict[str, Any]], params: Sequence[str] = ("fx", "fy")) -> Dict[str, Any]:
    """
    v0p74.9: each block left out in turn: its backlash-state offset a (at the reference focus, divided by its Navcam
    scale) of each eye and parameter against the weighted mean (with between-block scatter) of the other blocks.
    Returns {group: {param: {"rms_px", "max_px", "n", "per_block": [{"block", "a", "predicted", "residual_px"}]}}} - the
    rms is what a block that took no part in the calibration should expect.
    """
    out: Dict[str, Any] = {}
    groups = sorted({g for i in infos for g in (i.get("lines") or {})})
    for g in groups:
        for pn in params:
            rows = []
            for i in infos:
                e = ((((i.get("lines") or {}).get(g) or {}).get("lines") or {}).get("backlash") or {}).get(pn)
                if isinstance(e, dict) and "a" in e:
                    k = (i.get("navcam_scale") or 1.0)
                    rows.append((i["name"], float(e["a"]) / k, float(e.get("sd_a") or np.nan) / k))
            if len(rows) < 3:
                continue
            per = []
            for j, (name, a, _) in enumerate(rows):
                others = [r for k_, r in enumerate(rows) if k_ != j]
                m = _wmean(np.array([r[1] for r in others]), np.array([r[2] for r in others], float))["value"]
                per.append({"block": name, "a": a, "predicted": m, "residual_px": a - m})
            res = np.array([p["residual_px"] for p in per])
            out.setdefault(g, {})[pn] = {"rms_px": float(np.sqrt(np.mean(res ** 2))), "max_px": float(np.max(np.abs(res))),
                                         "n": len(per), "per_block": per}
    return out


REGULAR_RATIO_DEFAULT = 1.0095          # v0p74.9: backlash / regular focal length when the blocks do not measure it


def calibrate(infos: Sequence[Dict[str, Any]], model: Dict[str, Any], z_temperature: float = 3.0,
              min_blocks_for_T: int = 5, regular: str = "derived", form: Any = "v0p75",
              rig_translation: str = "cahv") -> Dict[str, Any]:
    """
    v0p74: the Mastcam-Z calibration from the selected v0p74 blocks (their focus lines, ``settings
    ["zcam_focus_lines"][group]["lines"][state][param]``, and rigs).  Per eye, focus state and parameter the
    offset a (at the reference focus) and slope b are averaged over the blocks (weights 1 / sd^2 with the
    between-block scatter tau; fx, fy divided by the block's Navcam scale); a trend of a with the block's lens
    temperature is fitted and kept (``c_per_degC`` about ``T0_degC``) only when |z| >= ``z_temperature`` with at
    least ``min_blocks_for_T`` blocks.  The rig: the mean rotation and translation of the blocks' refined rigs.
    Returns the new focus model (``lines`` per eye, f0 / slope / aspect / pp / distortion for older readers,
    ``rig``) with ``calibration`` (every number, the blocks, the scatter).
    v0p74.9 (after the 5 Oct promotion): (1) ``regular="derived"`` - the regular (low-backlash) state is not averaged
    from the blocks' regular bins (a handful of 1-3 image bins per block, some misclassified: chi2/dof up to 2.7e5,
    temperature terms of +8 px/degC, ZR k2 -2.07 - 0.020/count promoted on 5 Oct) but derived: fx, fy = the backlash
    lines divided by the measured backlash / regular ratio (:func:`regular_ratio`), every other parameter the
    backlash state's, no temperature term; ``"fitted"`` keeps the old behaviour.  (2) A temperature term needs
    ``min_blocks_for_T`` (5) blocks.  (3) ``calibration["loo"]``: every block left out in turn - its backlash fx, fy
    offsets against the mean of the others (:func:`leave_one_out`).
    v0p75: ``form`` (default the v0p75 line form, :data:`mppp.sfm.zcam.LINE_FORM_V0P75`; ``"v0p74"`` / None for the
    free lines): the averaged lines are put on the form - fx and fy of an eye share one slope (the mean of both),
    k1, k2 have no slope, ZR cx sits at pp0 with no slope, ZR cy at pp0 with its slope; ZL's pp0 (the model's) is
    kept.  ``rig_translation`` ``"cahv"``: the rig translation is the blocks' CAHV baseline (held in the runs),
    ``"refined"`` the mean of the refined ones.  The rig consensus adds ``residuals``: each block's refined rotation
    against its start (the model's rig) as yaw / pitch / roll mdeg, with the slopes against the block's optics
    temperature and sol (``vs_T``, ``vs_sol``: slope, se, z) - the test for a temperature or sol term.
    """
    import copy
    from scipy.spatial.transform import Rotation
    from .zcam import LINE_FORM_V0P75
    if form in (None, "v0p74", "free"):
        form_of = {}
    elif form == "v0p75":
        form_of = LINE_FORM_V0P75
    elif isinstance(form, dict):
        form_of = form
    else:
        raise ValueError(f"form {form!r}: 'v0p75', 'v0p74' or {{eye: {{param: form}}}}")
    out = copy.deepcopy(model)
    cal: Dict[str, Any] = {"blocks": [i["name"] for i in infos], "when": datetime.date.today().isoformat(),
                           "eyes": {}, "z_temperature": z_temperature}
    # The right eye's principal-point offset at the reference focus and the rig rotation are the same thing to a
    # block's bundle adjustment (both free in v0p74): before anything is averaged every block is put on one
    # convention - ZR's cx, cy at the reference focus = pp0 (the model's, else the label's) and the difference moved
    # into that block's rig rotation (:func:`pp_shift_to_rig`).  The rigs are then comparable between blocks.
    shifts: Dict[str, Dict[str, Any]] = {}
    rig_adj: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for g, gm in (out.get("cameras") or {}).items():
        if not str(g).startswith("ZR"):
            continue
        ref0 = None
        for i in infos:
            ref0 = ((i["lines"].get(g) or {}).get("reference_focus")) or ref0
        ref0 = float(ref0 if ref0 is not None else gm.get("reference_focus", 600.0))
        pp0 = gm.get("pp0_px")
        if not pp0:
            lab = [p for p in (_label_pp_at(i, g, ref0) for i in infos) if p is not None]
            pp0 = [float(np.median([p[0] for p in lab])), float(np.median([p[1] for p in lab]))] if lab else None
        if not pp0:
            continue
        gm["pp0_px"] = [float(pp0[0]), float(pp0[1])]
        for i in infos:
            bk = (((i["lines"].get(g) or {}).get("lines") or {}).get("backlash") or {})
            has = {pn: isinstance(bk.get(pn), dict) and "a" in bk[pn] for pn in ("cx", "cy")}
            if not any(has.values()):
                continue
            d = (float(bk["cx"]["a"]) - float(pp0[0]) if has["cx"] else 0.0,
                 float(bk["cy"]["a"]) - float(pp0[1]) if has["cy"] else 0.0)
            f = float(bk["fx"]["a"]) if isinstance(bk.get("fx"), dict) and "a" in bk["fx"] else float(gm.get("f0_px") or 4720.0)
            shifts[i["name"]] = {"group": g, "dcx_px": d[0], "dcy_px": d[1], "f_px": f}
            v = (i.get("rig") or {}).get("Z034") or {}
            if v.get("R_refined") is not None and v.get("t_refined") is not None:
                rig_adj[i["name"]] = pp_shift_to_rig(np.asarray(v["R_refined"], float), np.asarray(v["t_refined"], float),
                                                     f, d[0], d[1])
    cal["pp_to_rig"] = {"pp0_px": {g: gm.get("pp0_px") for g, gm in (out.get("cameras") or {}).items() if str(g).startswith("ZR")},
                        "shifts": shifts,
                        "note": "ZR cx, cy at the reference focus put at pp0 in every block; the difference moved into "
                                "that block's rig rotation before the rigs were averaged (the two are degenerate in BA)"}
    for g, gm in (out.get("cameras") or {}).items():
        lines: Dict[str, Any] = {}
        ref = None
        for st in (("backlash",) if regular == "derived" else ("backlash", "regular")):
            pars: Dict[str, Any] = {}
            names = sorted({pn for i in infos for pn in (((i["lines"].get(g) or {}).get("lines") or {}).get(st) or {})
                            if pn in FULL})
            for pn in names:
                rows = []
                for i in infos:
                    L = (i["lines"].get(g) or {})
                    e = ((L.get("lines") or {}).get(st) or {}).get(pn)
                    if not isinstance(e, dict) or "a" not in e:
                        continue
                    ref = L.get("reference_focus", ref)
                    k = (i["navcam_scale"] or 1.0) if pn in ("fx", "fy") else 1.0
                    sh = shifts.get(i["name"]) if pn in ("cx", "cy") and str(g).startswith("ZR") else None
                    a_val = e["a"] / k - (sh[f"d{pn}_px"] if sh else 0.0)
                    rows.append({"block": i["name"], "a": a_val, "b": e["b"] / k,
                                 "sd_a": (e.get("sd_a") or np.nan) / k, "sd_b": (e.get("sd_b") or np.nan) / k,
                                 "T": i.get("T_optics_degC"), "n_bins": e.get("n_bins"), "slope_source": e.get("slope_source")})
                if not rows:
                    continue
                A = _wmean(np.array([r["a"] for r in rows]), np.array([r["sd_a"] for r in rows], float))
                fitted = [r for r in rows if r["slope_source"] == "fitted"] or rows
                B = _wmean(np.array([r["b"] for r in fitted]), np.array([r["sd_b"] for r in fitted], float))
                ent = {"a": A["value"], "b": B["value"], "sd_a": A["se"], "sd_b": B["se"], "tau_a": A["tau"],
                       "tau_b": B["tau"], "chi2_dof_a": A["chi2_dof"], "n_blocks": A["n"], "blocks": rows}
                Ts = np.array([np.nan if r["T"] is None else r["T"] for r in rows], float)
                ok = np.isfinite(Ts)
                if ok.sum() >= 2 and np.ptp(Ts[ok]) > 0:
                    T0 = float(np.mean(Ts[ok]))
                    y = np.array([r["a"] for r in rows])[ok]
                    sd = np.array([r["sd_a"] for r in rows], float)[ok]
                    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nanmedian(sd) if np.any(np.isfinite(sd)) else 1.0)
                    from .zcam_lines import _meta
                    try:                                    # the between-block scatter about the line in T, not the mean
                        mt = _meta(y, sd, Ts[ok] - T0)
                        c, sc, a_T = float(mt["slope"]), float(mt["slope_se"]), float(mt["value"])
                    except (np.linalg.LinAlgError, ValueError):
                        c, sc, a_T = float("nan"), float("nan"), float("nan")
                    z = c / sc if sc and np.isfinite(sc) and sc > 0 else None
                    ent["temperature"] = {"c_per_degC": c, "sd": sc, "z": z, "T0_degC": T0, "sensor": "HEAD_HTR (optics)",
                                          "applied": bool(z is not None and abs(z) >= z_temperature
                                                          and ok.sum() >= min_blocks_for_T)}
                    if ent["temperature"]["applied"]:
                        ent["a"] = a_T
                        ent["c_per_degC"], ent["T0_degC"] = c, T0
                pars[pn] = ent
            fm = form_of.get(str(g)[:2]) or form_of.get(str(g)) or {}
            if pars and fm:                                 # v0p75: the averaged lines on the form
                common = [pn for pn in pars if fm.get(pn) == "common"]
                if len(common) >= 2:
                    bs = np.array([pars[pn]["b"] for pn in common]); ss = np.array([pars[pn]["sd_b"] or np.nan for pn in common], float)
                    ss = np.where(np.isfinite(ss) & (ss > 0), ss, np.nanmax(ss) if np.any(np.isfinite(ss)) else 1.0)
                    wb = 1.0 / ss ** 2
                    b_c, sd_c = float(np.sum(wb * bs) / np.sum(wb)), float(1.0 / np.sqrt(np.sum(wb)))
                    for pn in common:
                        pars[pn].update({"b": b_c, "sd_b": sd_c, "form": "common", "common_slope_of": list(common)})
                pp0 = gm.get("pp0_px")
                for pn, e in pars.items():
                    f_ = fm.get(pn, "free")
                    if f_ in ("constant", "pinned"):
                        e.update({"b": 0.0, "sd_b": 0.0, "form": f_})
                    if f_ in ("pinned", "pinned_slope") and pn in ("cx", "cy") and pp0:
                        e.update({"a": float(pp0[0 if pn == "cx" else 1]), "sd_a": 0.0, "form": f_,
                                  "a_source": "pinned at pp0 (form)"})
                        e.pop("c_per_degC", None)
                        e.pop("T0_degC", None)
                    if f_ in ("constant", "pinned") and "c_per_degC" in e:
                        pass                                # a temperature term of a constant is still a constant
            if pars:
                lines[st] = pars
        if not lines:
            cal["eyes"][g] = {"skipped": "no block with v0p74 lines for this eye"}
            continue
        ref = float(ref if ref is not None else gm.get("reference_focus", 600.0))
        lines["reference_focus"] = ref
        bk = lines.get("backlash") or {}
        if regular == "derived" and bk:
            rr = regular_ratio(infos, g)
            reg = {}
            for pn, e in bk.items():
                if not isinstance(e, dict):
                    continue
                k = rr["ratio"] if pn in ("fx", "fy") else 1.0
                reg[pn] = {"a": e["a"] / k, "b": e["b"] / k, "sd_a": e["sd_a"] / k, "sd_b": e["sd_b"] / k,
                           "tau_a": e.get("tau_a", 0.0), "chi2_dof_a": e.get("chi2_dof_a", float("nan")),
                           "n_blocks": e.get("n_blocks"), "derived": f"backlash / {k:.5f}" if k != 1.0 else "backlash"}
                if "c_per_degC" in e:                       # the same temperature term as the backlash state
                    reg[pn]["c_per_degC"], reg[pn]["T0_degC"] = e["c_per_degC"] / k, e["T0_degC"]
            lines["regular"] = reg
            cal.setdefault("regular_ratio", {})[g] = rr
        if "fx" in bk and "fy" in bk:
            gm["f0_px"] = float(np.sqrt(bk["fx"]["a"] * bk["fy"]["a"]))
            gm["slope_px_per_count"] = 0.5 * (bk["fx"]["b"] + bk["fy"]["b"])
            gm["aspect"] = bk["fy"]["a"] / bk["fx"]["a"]
            gm["reference_focus"] = ref
        if "cx" in bk or "cy" in bk:
            pp = dict(gm.get("pp") or {}, note="v0p74 calibration: ZR principal point line (the R-L boresight; ZL the reference)")
            pp0 = list(gm.get("pp0_px") or [None, None])
            for k, pn in ((0, "cx"), (1, "cy")):
                if pn in bk:
                    pp[f"{pn}_px_per_count"] = bk[pn]["b"]
                    pp0[k] = bk[pn]["a"]
            gm["pp"] = pp
            if all(v is not None for v in pp0) and not gm.get("pp0_px"):
                gm["pp0_px"] = pp0
        if "k1" in bk and "k2" in bk:
            names = list((gm.get("distortion") or {}).get("names") or ["k1", "k2"])
            params = list((gm.get("distortion") or {}).get("params") or [0.0] * len(names))
            for pn in ("k1", "k2"):
                if pn in names:
                    params[names.index(pn)] = bk[pn]["a"]
                else:
                    names.append(pn)
                    params.append(bk[pn]["a"])
            gm["distortion"] = {"names": names, "params": params,
                                "note": "v0p74 calibration: k1, k2 at the reference focus (the lines give them per focus)"}
        gm["lines"] = {st: {pn: {k: v for k, v in e.items() if k in ("a", "b", "c_per_degC", "T0_degC")}
                            for pn, e in pars.items()} for st, pars in lines.items() if isinstance(pars, dict)}
        gm["lines"]["reference_focus"] = ref
        gm.pop("trend", None)                                 # the lines replace the sol trend
        cal["eyes"][g] = lines
    # the stereo rig: mean of the blocks' refined rigs (right from left)
    rigs = []
    for i in infos:
        for fam, v in (i.get("rig") or {}).items():
            if str(fam) == "Z034" and v.get("R_refined") is not None and v.get("t_refined") is not None:
                Ra, ta = rig_adj.get(i["name"], (np.asarray(v["R_refined"], float), np.asarray(v["t_refined"], float)))
                rigs.append({"block": i["name"], "rotvec": Rotation.from_matrix(Ra).as_rotvec(), "t": ta,
                             "T_optics_degC": i.get("T_optics_degC"), "sol": i.get("sol"),
                             "pp_shift_px": [shifts[i["name"]]["dcx_px"], shifts[i["name"]]["dcy_px"]] if i["name"] in shifts else None,
                             "rotvec_start": Rotation.from_matrix(np.asarray(v["R_sensor_from_ref"], float)).as_rotvec()})
    if rigs:
        rv = np.array([r["rotvec"] for r in rigs])
        tt = np.array([r["t"] for r in rigs])
        R = Rotation.from_rotvec(rv.mean(axis=0)).as_matrix()
        t = tt.mean(axis=0)
        t_source = "mean of the blocks' refined translations"
        if rig_translation == "cahv":                      # v0p75: the baseline held at CAHV
            tc = []
            for i in infos:
                v = (i.get("rig") or {}).get("Z034") or {}
                tv = v.get("t_sensor_from_ref_cahv") or v.get("t_sensor_from_ref")
                if tv is not None:
                    tc.append(np.asarray(tv, float))
            if tc:
                t = np.mean(np.array(tc), axis=0)
                t_source = f"CAHV median of the pairs, mean of {len(tc)} blocks (held in the runs)"
        # v0p75: each block's refined rotation against its start (the model's rig, the same for every block when the
        # runs start from the model): the residual angles and their trends with temperature and sol
        res_rows = []
        for r in rigs:
            info = next((i for i in infos if i["name"] == r["block"]), {})
            v = (info.get("rig") or {}).get("Z034") or {}
            R0 = np.asarray(v.get("R_sensor_from_ref"), float) if v.get("R_sensor_from_ref") is not None else None
            if R0 is None:
                continue
            M = Rotation.from_rotvec(r["rotvec"]).as_matrix() @ R0.T
            ang = Rotation.from_matrix(M).as_rotvec()
            res_rows.append({"block": r["block"], "T_optics_degC": r["T_optics_degC"], "sol": r["sol"],
                             "yaw_mdeg": float(np.degrees(ang[1]) * 1e3), "pitch_mdeg": float(np.degrees(ang[0]) * 1e3),
                             "roll_mdeg": float(np.degrees(ang[2]) * 1e3),
                             "start_source": v.get("rotation_source")})
        residuals: Dict[str, Any] = {"per_block": res_rows, "fits": {}}
        for ang_name in ("yaw", "pitch", "roll"):
            y = np.array([rr[f"{ang_name}_mdeg"] for rr in res_rows], float)
            T = np.array([np.nan if rr["T_optics_degC"] is None else rr["T_optics_degC"] for rr in res_rows], float)
            S = np.array([np.nan if rr["sol"] is None else rr["sol"] for rr in res_rows], float)
            residuals["fits"][ang_name] = {"mean_mdeg": float(np.mean(y)) if len(y) else None,
                                           "sd_mdeg": float(np.std(y, ddof=1)) if len(y) > 1 else None,
                                           "vs_T": _ols(T, y), "vs_sol": _ols(S / 1000.0, y)}
        sd_mdeg = (np.degrees(rv.std(axis=0, ddof=1)) * 1e3).tolist() if len(rigs) > 1 else None
        out["rig"] = {"R_sensor_from_ref": R.tolist(), "t_sensor_from_ref": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
                      "translation_source": t_source, "residuals": residuals,
                      "rotvec_mdeg": (np.degrees(rv.mean(axis=0)) * 1e3).tolist(), "sd_rotvec_mdeg": sd_mdeg,
                      "sd_baseline_mm": float(np.std(np.linalg.norm(tt, axis=1), ddof=1) * 1e3) if len(rigs) > 1 else None,
                      "blocks": [r["block"] for r in rigs],
                      "per_block": [{"block": r["block"], "rotvec_mdeg": (np.degrees(r["rotvec"]) * 1e3).tolist(),
                                     "baseline_mm": float(np.linalg.norm(r["t"]) * 1e3), "T_optics_degC": r["T_optics_degC"],
                                     "sol": r["sol"], "pp_shift_px": r["pp_shift_px"]} for r in rigs],
                      "note": f"v0p74 calibration: mean of {len(rigs)} blocks' refined Mastcam-Z rigs (right from left; the "
                              f"same for every focus - the focus motion is the principal points'; ZR cx, cy at the reference "
                              f"focus = pp0, each block's offset moved into its rig rotation first)"}
        cal["rig"] = {k: v for k, v in out["rig"].items() if k != "R_sensor_from_ref"}
    cal["loo"] = leave_one_out(infos)
    out["calibration"] = cal
    out["state"] = "calibrated (v0p74)"
    out["source"] = f"MPPP v0p74 Mastcam-Z 34 mm calibration from {len(infos)} blocks: {', '.join(cal['blocks'])}"
    return out


def check_model(new: Dict[str, Any], old: Dict[str, Any], limits: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Sanity limits of a new focus model against the one in use (``MODEL_LIMITS``): f0 within ``max_f0_change_pct``,
    aspect within ``max_aspect_dev`` of 1, |k1| below ``max_k1``, the rig baseline in ``baseline_m`` and within
    ``max_rig_change_deg`` of the old rig (or of CAHV), at least ``min_blocks`` blocks.  Returns {"ok", "checks"}."""
    lim = dict(MODEL_LIMITS, **(limits or {}))
    checks = []

    def add(name, ok, note):
        checks.append({"check": name, "ok": bool(ok), "note": note})
    cal, su = new.get("calibration"), new.get("start_update")
    nb = len(((cal or su or {}).get("blocks")) or [])
    need = lim["min_blocks_calibration"] if cal is not None else lim["min_blocks"]   # v0p74.9: 3 for a calibration
    add("blocks", nb >= need, f"{nb} blocks")
    for g, c in (new.get("cameras") or {}).items():
        o = (old.get("cameras") or {}).get(g) or {}
        if cal is not None:                     # a calibration must have fitted every eye
            e = (cal.get("eyes") or {}).get(g) or {}
            bk = e.get("backlash") if isinstance(e, dict) else None
            has = bool(bk) and all(isinstance(bk.get(pn), dict) for pn in ("fx", "fy"))
            add(f"{g} lines", has, "fx, fy lines fitted" if has else str(e.get("skipped") or "no backlash fx, fy lines"))
        if su is not None:
            e = (su.get("eyes") or {}).get(g) or {}
            if e.get("skipped"):
                add(f"{g} start", False, str(e["skipped"]))
            elif e.get("scatter_offsets_pct") is not None:
                add(f"{g} block offsets", e["scatter_offsets_pct"] <= lim["max_offset_scatter_pct"],
                    f"sd {e['scatter_offsets_pct']:.2f} % of f0")
        if c.get("f0_px") and o.get("f0_px"):
            d = 100.0 * abs(c["f0_px"] / o["f0_px"] - 1.0)
            add(f"{g} f0", d <= lim["max_f0_change_pct"], f"{o['f0_px']:.1f} -> {c['f0_px']:.1f} px ({d:.2f} %)")
        if c.get("aspect"):
            add(f"{g} aspect", abs(c["aspect"] - 1.0) <= lim["max_aspect_dev"], f"{c['aspect']:.5f}")
        dist = c.get("distortion") or {}
        if "k1" in (dist.get("names") or []):
            k1 = dist["params"][dist["names"].index("k1")]
            add(f"{g} k1", abs(k1) <= lim["max_k1"], f"{k1:+.4f}")
    if cal is not None:                         # v0p74.9: what was promoted on 5 Oct without a check
        for g, e in (cal.get("eyes") or {}).items():
            if not isinstance(e, dict) or "skipped" in e:
                continue
            for st in ("backlash", "regular"):
                for pn, x in (e.get(st) or {}).items():
                    if not isinstance(x, dict):
                        continue
                    t = x.get("temperature") or {}
                    if t.get("applied"):
                        c = abs(float(t.get("c_per_degC") or 0.0))
                        lim_c = lim["max_T_slope_px_per_degC"] if pn in ("fx", "fy") else lim["max_T_slope_pp_px_per_degC"]
                        add(f"{g} {st} {pn} temperature term", c <= lim_c, f"{c:.2f} px/degC applied")
                    if st == "backlash" and pn in ("fx", "fy") and np.isfinite(x.get("chi2_dof_a", np.nan)):
                        add(f"{g} {st} {pn} block scatter", float(x.get("tau_a") or 0.0) <= lim["max_tau_f_px"],
                            f"between-block sd {float(x.get('tau_a') or 0.0):.1f} px (chi2/dof {x['chi2_dof_a']:.1f})")
                    if pn in ("k1", "k2"):
                        bk = (e.get("backlash") or {}).get(pn) or {}
                        if st == "regular" and bk and abs(float(x["a"]) - float(bk["a"])) > lim["max_regular_k_dev"][pn]:
                            add(f"{g} regular {pn}", False, f"{x['a']:+.3f} against backlash {bk['a']:+.3f}")
    rig = new.get("rig")
    if rig:
        b = float(rig.get("baseline_m") or 0.0)
        add("rig baseline", lim["baseline_m"][0] <= b <= lim["baseline_m"][1], f"{b:.4f} m")
        if rig.get("sd_rotvec_mdeg"):
            s = max(float(x) for x in rig["sd_rotvec_mdeg"])
            add("rig scatter", s <= lim["max_rig_scatter_mdeg"],
                f"rotation sd {', '.join(f'{float(x):.1f}' for x in rig['sd_rotvec_mdeg'])} mdeg over {len(rig.get('blocks') or [])} blocks")
        if rig.get("sd_baseline_mm") is not None:
            add("rig baseline scatter", float(rig["sd_baseline_mm"]) <= lim["max_baseline_scatter_mm"],
                f"sd {float(rig['sd_baseline_mm']):.2f} mm")
        orig = old.get("rig") or {}
        if orig.get("R_sensor_from_ref") and rig.get("R_sensor_from_ref"):
            from scipy.spatial.transform import Rotation
            dR = Rotation.from_matrix(np.asarray(rig["R_sensor_from_ref"], float)) * \
                Rotation.from_matrix(np.asarray(orig["R_sensor_from_ref"], float)).inv()
            ang = float(np.degrees(np.linalg.norm(dR.as_rotvec())))
            add("rig change", ang <= lim["max_rig_change_deg"], f"{ang:.3f} deg from the rig in use")
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks}


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
