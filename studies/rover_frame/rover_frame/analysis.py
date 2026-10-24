"""
Study 3A, the rover frame (0.94.4, Christian 23 Oct 2026): what the adjusted blocks say about the rover body, the
mast and the camera mounts, with no new adjustment.

The Hazcams are fixed to the body, so the adjusted pose of a Hazcam frame is the body pose of its sequence.  The
mast cameras (Navcam, Mastcam-Z) at the same body pose give the mast-head pose; their attitude relative to that body,
in the label's RSM head frame, is a constant mount plus the RSM pointing error and flexure.  The camera positions in
the body frame (Hazcams) and in the head frame (mast cameras) are constants, and their scatter is the measurement.
Wheel settling and arm-induced body motion appear as body-pose changes between the sequences of one station.

Input per image (:func:`image_rows`): the adjusted pose ``R_c_w``, ``C_w``, the prior ``prior_R_c_w``, ``prior_C``,
the names (block, stem, family N/F/R/Z, eye L/R, group e.g. NL / FL / ZL034, station, sequence, sclk) and the label
state (:func:`rover_frame.labels.rover_state`).  Output: rows (one per image), body groups, body changes, mount
constants and pointing-law fits - plain dicts, written by ``run_rover_frame.py``.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as G

MDEG = 1000.0 * 180.0 / np.pi                    # rad -> mdeg
MM = 1000.0
MAX_PRIOR_CHECK_DEG = 0.05                       # the label pose rebuilt here must match the project's prior
AXES = ("x", "y", "z")


def camera_group(stem: str, instrument: str = "") -> Tuple[str, str, str]:
    """(family N/F/R/Z, eye L/R, group NL / FR / ZL034 ...) of a product."""
    fam, eye = stem[0].upper(), stem[1].upper()
    if fam == "Z":
        m = re.match(r"Z[LR](\d{3})", instrument or "")
        return fam, eye, f"Z{eye}{m.group(1) if m else ''}"
    return fam, eye, f"{fam}{eye}" if fam in "FR" else f"N{eye}"


def _vec(v, scale=1.0) -> Dict[str, float]:
    return {a: float(x) * scale for a, x in zip(AXES, v)}


def image_rows(images: Sequence[Dict[str, Any]], states: Dict[str, Dict[str, Any]],
               reasons: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """The label geometry of every image with a usable state (the body still to be chosen): telemetry attitude,
    the label camera in the rover and head frames, the prior check, and for a Hazcam the body it implies.
    ``reasons`` counts the images left out, by reason."""
    out = []
    reasons = {} if reasons is None else reasons

    def skip(why):
        reasons[why] = reasons.get(why, 0) + 1

    for im in images:
        st = states.get(im["stem"].upper()) or states.get(im["stem"])
        if not st:
            skip("no label found")
            continue
        if st.get("error") or "R_cam_rnav" not in st or not st.get("rover"):
            skip("label unreadable or without a camera model / rover frame")
            continue
        if st.get("camera_frame", "ROVER_NAV_FRAME") != "ROVER_NAV_FRAME":
            skip("camera model not in the rover nav frame")
            continue
        fam, eye, grp = camera_group(im["stem"], im.get("instrument", ""))
        R_tel = G.telemetry_rnav_from_world(st["rover"]["q_wxyz"])
        R_c_rnav = np.asarray(st["R_cam_rnav"], float)
        C_rnav = np.asarray(st["C_rnav"], float)
        M = G.project_correction(im["prior_R_c_w"], R_c_rnav, R_tel)
        chk = G.angle_deg(M)
        if fam != "Z" and chk > MAX_PRIOR_CHECK_DEG:
            # the label pose rebuilt here is not the project's prior: another frame convention - left out
            skip(f"label pose != the project's prior by > {MAX_PRIOR_CHECK_DEG} deg")
            continue
        R_c_rnav = M @ R_c_rnav                         # Mastcam-Z: the principal-point correction of the prior
        prior_C = np.asarray(im["prior_C"], float)
        r = dict(im)
        r.update(family=fam, eye=eye, group=grp, state=st, R_tel=R_tel, R_c_rnav=R_c_rnav, C_rnav=C_rnav,
                 O_tel=prior_C - R_tel.T @ C_rnav, prior_check_deg=chk,
                 rmc=st.get("rmc") or {}, sclk=st.get("sclk") or im.get("sclk"))
        if st.get("head"):
            R_rnav_head = G.quat_matrix(st["head"]["q_wxyz"])
            r.update(R_rnav_head=R_rnav_head, O_head=np.asarray(st["head"]["offset_m"], float))
        if fam in "FR":
            R_b, O_b = G.body_from_camera(im["R_c_w"], im["C_w"], R_c_rnav, C_rnav)
            r.update(R_body=R_b, O_body=O_b, omega_tel=G.residual_rnav(R_b, R_tel))
        out.append(r)
    return out


def body_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One body pose per Hazcam sequence and rover pose (block, station, family F / R, sequence, RMC POSE and ARM),
    from its left-eye frames (the right eye is posed by the rig)."""
    by: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["family"] in "FR" and r["eye"] == "L":
            k = (r["block"], r["station"], r["family"], r.get("sequence"), r["rmc"].get("POSE"), r["rmc"].get("ARM"))
            by[k].append(r)
    groups = []
    for k, rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], min(float(x["sclk"] or 0) for x in kv[1]))):
        R = G.mean_rotation([x["R_body"] for x in rs])
        O = np.mean([x["O_body"] for x in rs], axis=0)
        R_tel = G.mean_rotation([x["R_tel"] for x in rs])
        O_tel = np.mean([x["O_tel"] for x in rs], axis=0)
        spread = [G.angle_deg(x["R_body"] @ R.T) * 1000.0 for x in rs]
        st = rs[0]["state"]
        g = {"id": f"{k[0]}|{k[1]}|{k[2]}|{k[3]}|P{k[4]}|A{k[5]}", "block": k[0], "station": k[1], "family": k[2],
             "sequence": k[3], "POSE": k[4], "ARM": k[5], "RSM": st.get("rmc", {}).get("RSM"),
             "sclk": float(np.mean([float(x["sclk"] or 0) for x in rs])), "sol": rs[0].get("sol"), "frames": len(rs),
             "R_body": R, "O_body": O, "R_tel": R_tel, "O_tel": O_tel,
             "omega_tel_mdeg": _vec(G.residual_rnav(R, R_tel), MDEG),
             "spread_mdeg": float(np.max(spread)) if spread else 0.0,
             "chassis": st.get("chassis") or {}, "arm": st.get("arm") or {}}
        for x in rs:
            x["body_group"] = g["id"]
        groups.append(g)
    return groups


def choose_body(r: Dict[str, Any], groups: List[Dict[str, Any]],
                propagate: bool = True) -> Tuple[Optional[Dict[str, Any]], str]:
    """The body pose of an image: a front Hazcam group of the same block, station and rover pose (POSE; the same ARM
    count preferred, then the nearest in time), else a rear one ("front" / "rear", "..._arm_moved" when the ARM
    count differs); else (``propagate``) the nearest Hazcam group of the station at another rover pose, its body
    error carried over to this pose's telemetry ("..._propagated"); else None (the telemetry attitude)."""
    here = [g for g in groups if g["block"] == r["block"] and g["station"] == r["station"]]
    if r["family"] in "FR" and r["eye"] == "L":
        here = [g for g in here if g["family"] != r["family"]]          # the other pair (rear vs front)
    cands = [g for g in here if g["POSE"] == r["rmc"].get("POSE")]
    t = float(r["sclk"] or 0)
    if cands:
        cands.sort(key=lambda g: (g["family"] != "F", g["ARM"] != r["rmc"].get("ARM"), abs(g["sclk"] - t)))
        g = cands[0]
        return g, ("front" if g["family"] == "F" else "rear") + ("" if g["ARM"] == r["rmc"].get("ARM") else "_arm_moved")
    if propagate and here:
        # another rover pose of the station: its measured body error carried over to this pose's telemetry
        here.sort(key=lambda g: (g["family"] != "F", abs(g["sclk"] - t)))
        g = here[0]
        return g, ("front" if g["family"] == "F" else "rear") + "_propagated"
    return None, "telemetry"


def relate(rows: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every image relative to its body (:func:`choose_body`; the telemetry attitude and the prior position when
    there is none): the attitude residual in the rover frame (and the head frame for a mast camera) and the
    position in the body (head) frame minus the label's, as flat table rows (mdeg, mm)."""
    gid = {g["id"]: g for g in groups}
    out = []
    for r in rows:
        own = gid.get(r.get("body_group")) if r["family"] in "FR" else None
        if r["family"] in "FR" and r["eye"] == "R":
            # a right Hazcam against its own pair's body (the rig) - the left frame of the same sequence and pose
            own = next((g for g in groups if g["block"] == r["block"] and g["station"] == r["station"]
                        and g["family"] == r["family"] and g["sequence"] == r.get("sequence")
                        and g["POSE"] == r["rmc"].get("POSE")), None)
            g, src = own, ("own_pair" if own else "telemetry")
        else:
            g, src = choose_body(r, groups)
        if g is None:
            R_b, O_b = r["R_tel"], r["O_tel"]
        elif src.endswith("_propagated"):
            R_b = g["R_body"] @ g["R_tel"].T @ r["R_tel"]               # the station's body error, this pose's IMU
            O_b = g["O_body"] + (r["O_tel"] - g["O_tel"])
        else:
            R_b, O_b = g["R_body"], g["O_body"]
        R_c_rnav_meas = np.asarray(r["R_c_w"], float) @ R_b.T
        e_rnav = G.rotvec(r["R_c_rnav"].T @ R_c_rnav_meas)                # acts on rover coordinates
        # (the camera's own frame: R_c_rnav_meas = R_c_rnav exp(e_rnav))
        d_rnav = G.point_in_frame(r["C_w"], R_b, O_b) - r["C_rnav"]
        rsm = r["state"].get("rsm") or {}
        row = {"block": r["block"], "image": r["stem"], "family": r["family"], "eye": r["eye"], "group": r["group"],
               "station": r["station"], "sol": r.get("sol"), "sequence": r.get("sequence"), "sclk": r["sclk"],
               "POSE": r["rmc"].get("POSE"), "ARM": r["rmc"].get("ARM"), "RSM": r["rmc"].get("RSM"),
               "body": src, "body_group": g["id"] if g else "",
               "az_encoder": rsm.get("az_encoder"), "el_encoder": rsm.get("el_encoder"),
               "az_resolver": rsm.get("az_resolver"), "el_resolver": rsm.get("el_resolver"),
               "az_requested": rsm.get("az_requested"), "el_requested": rsm.get("el_requested"),
               "T_plate_L": (r["state"].get("navcam_plate_T") or {}).get("L"),
               "T_plate_R": (r["state"].get("navcam_plate_T") or {}).get("R"),
               "T_camera": r.get("camera_temperature_degC"), "T_zcam": r.get("zcam_optics_temperature_degC"),
               "observations": r.get("observations"), "prior_check_deg": r["prior_check_deg"]}
        row.update({f"e_rnav_{a}_mdeg": v for a, v in _vec(e_rnav, MDEG).items()})
        row.update({f"d_rnav_{a}_mm": v for a, v in _vec(d_rnav, MM).items()})
        if "omega_tel" in r:
            row.update({f"body_tel_{a}_mdeg": v for a, v in _vec(r["omega_tel"], MDEG).items()})
        if r["family"] in "NZ" and "R_rnav_head" in r:
            R_head_rnav = r["R_rnav_head"].T
            e_head = R_head_rnav @ e_rnav
            d_head = R_head_rnav @ (G.point_in_frame(r["C_w"], R_b, O_b) - r["O_head"]) - \
                R_head_rnav @ (r["C_rnav"] - r["O_head"])
            row.update({f"e_head_{a}_mdeg": v for a, v in _vec(e_head, MDEG).items()})
            row.update({f"d_head_{a}_mm": v for a, v in _vec(d_head, MM).items()})
        out.append(row)
    return out


def body_changes(groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Consecutive body poses of one station and Hazcam pair (in time): the adjusted change of attitude and
    origin in the first pose's rover frame, the telemetry change, what the telemetry missed, and the counters,
    suspension and arm angles that changed."""
    out = []
    by: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for g in groups:
        by[(g["block"], g["station"], g["family"])].append(g)
    for k, gs in by.items():
        gs = sorted(gs, key=lambda g: g["sclk"])
        for a, b in zip(gs, gs[1:]):
            dR_ba = b["R_body"] @ a["R_body"].T                   # rnav_a -> rnav_b
            dR_tel = b["R_tel"] @ a["R_tel"].T
            w_ba, w_tel = G.rotvec(dR_ba), G.rotvec(dR_tel)
            miss = G.rotvec(dR_ba @ dR_tel.T)
            dO = a["R_body"] @ (b["O_body"] - a["O_body"])
            row = {"block": k[0], "station": k[1], "family": k[2], "from": a["id"], "to": b["id"],
                   "dt_s": b["sclk"] - a["sclk"], "POSE": f"{a['POSE']}->{b['POSE']}", "ARM": f"{a['ARM']}->{b['ARM']}",
                   "pose_changed": a["POSE"] != b["POSE"], "arm_moved": a["ARM"] != b["ARM"],
                   "angle_ba_mdeg": float(np.linalg.norm(w_ba) * MDEG), "angle_tel_mdeg": float(np.linalg.norm(w_tel) * MDEG),
                   "missed_mdeg": float(np.linalg.norm(miss) * MDEG), "move_mm": float(np.linalg.norm(dO) * MM)}
            row.update({f"ba_{x}_mdeg": v for x, v in _vec(w_ba, MDEG).items()})
            row.update({f"tel_{x}_mdeg": v for x, v in _vec(w_tel, MDEG).items()})
            row.update({f"missed_{x}_mdeg": v for x, v in _vec(miss, MDEG).items()})
            row.update({f"move_{x}_mm": v for x, v in _vec(dO, MM).items()})
            for c in ("left_bogie", "right_bogie", "left_differential", "right_differential"):
                va, vb = (a["chassis"] or {}).get(c), (b["chassis"] or {}).get(c)
                row[f"d_{c}_mdeg"] = (vb - va) * MDEG if va is not None and vb is not None else None
            dj = [abs((b["arm"] or {}).get(j) - (a["arm"] or {}).get(j)) for j in ("j1", "j2", "j3", "j4", "j5")
                  if (a["arm"] or {}).get(j) is not None and (b["arm"] or {}).get(j) is not None]
            row["arm_max_joint_change_deg"] = float(np.degrees(max(dj))) if dj else None
            out.append(row)
    return out


def _robust(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    med = float(np.median(v))
    return {"n": int(len(v)), "mean": float(np.mean(v)), "median": med, "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "nmad": float(1.4826 * np.median(np.abs(v - med))), "p95_abs": float(np.percentile(np.abs(v - med), 95))}


def measured(r: Dict[str, Any]) -> bool:
    """A body pose from a Hazcam frame of the same rover pose (not propagated, not the telemetry)."""
    return r["body"] != "telemetry" and not r["body"].endswith("_propagated")


def mounts(rel: List[Dict[str, Any]], which: str = "measured") -> Dict[str, Any]:
    """Per camera group the constant attitude and position relative to the body (head frame for a mast camera,
    rover frame for a Hazcam) and their scatter - the measurement.  ``which``: "measured" (a Hazcam body of the same
    rover pose), "propagated" (those and the Hazcam body of another pose of the station carried over by the
    telemetry change) or "all" (the telemetry attitude where there is no Hazcam body)."""
    out: Dict[str, Any] = {}
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rel:
        if which == "measured" and not measured(r):
            continue
        if which == "propagated" and r["body"] == "telemetry":
            continue
        by[r["group"]].append(r)
    for grp, rs in sorted(by.items()):
        frame = "head" if rs[0]["family"] in "NZ" and f"e_head_x_mdeg" in rs[0] else "rnav"
        e = {a: _robust([x.get(f"e_{frame}_{a}_mdeg") for x in rs]) for a in AXES}
        d = {a: _robust([x.get(f"d_{frame}_{a}_mm") for x in rs]) for a in AXES}
        out[grp] = {"frame": frame, "images": len(rs), "blocks": len({x["block"] for x in rs}),
                    "attitude_mdeg": e, "position_mm": d,
                    "per_block": {b: {a: float(np.median([x[f"e_{frame}_{a}_mdeg"] for x in rs if x["block"] == b]))
                                      for a in AXES} for b in sorted({x["block"] for x in rs})}}
    return out


def pointing_fit(rel: List[Dict[str, Any]], group: str, angles: str = "encoder", bodies: str = "propagated",
                 per_block: bool = False) -> Dict[str, Any]:
    """The head-frame attitude residual of a mast camera group (NL, ZL034, ...) against the RSM angles and the
    temperature (:func:`rover_frame.geometry.fit_terms`): coefficients (mdeg) with standard errors, the rms before
    and after.  ``per_block``: one constant per block (the block's frame) instead of one overall.  ``bodies``: as
    :func:`mounts` ``which``."""
    ok_body = {"measured": measured, "propagated": lambda r: r["body"] != "telemetry", "all": lambda r: True}[bodies]
    rs = [r for r in rel if r["group"] == group and "e_head_x_mdeg" in r and ok_body(r)
          and r.get(f"az_{angles}") is not None and r.get(f"el_{angles}") is not None]
    if len(rs) < 12:
        return {"group": group, "n": len(rs), "note": "too few images"}
    az = np.array([r[f"az_{angles}"] for r in rs], float)
    el = np.array([r[f"el_{angles}"] for r in rs], float)
    tk = "T_zcam" if group.startswith("Z") else "T_plate_L" if group.endswith("L") else "T_plate_R"
    T = np.array([np.nan if r.get(tk) is None else float(r[tk]) for r in rs], float)
    X, names = G.fit_terms(az, el, T)
    if per_block:
        blocks = sorted({r["block"] for r in rs})
        B = np.array([[1.0 if r["block"] == b else 0.0 for b in blocks] for r in rs])
        X, names = np.hstack([B, X[:, 1:]]), [f"const[{b}]" for b in blocks] + names[1:]
    res = {"group": group, "n": len(rs), "blocks": len({r["block"] for r in rs}), "angles": angles, "bodies": bodies, "terms": names,
           "per_block": per_block, "cond": float(np.linalg.cond(X)), "axes": {}}
    for a in AXES:
        y = np.array([r[f"e_head_{a}_mdeg"] for r in rs], float)
        ok = np.isfinite(y)
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        e = y[ok] - X[ok] @ beta
        dof = max(int(ok.sum()) - X.shape[1], 1)
        s2 = float(e @ e / dof)
        try:
            cov = s2 * np.linalg.pinv(X[ok].T @ X[ok])
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), np.nan)
        res["axes"][a] = {"coef_mdeg": dict(zip(names, map(float, beta))), "se_mdeg": dict(zip(names, map(float, se))),
                          "rms_before_mdeg": float(np.std(y[ok])), "rms_after_mdeg": float(np.sqrt(np.mean(e ** 2)))}
    return res


def station_summary(rel: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per block and station: the frames, the Hazcam body poses, the share of mast frames with a measured body."""
    out = []
    keys = sorted({(r["block"], r["station"]) for r in rel})
    for b, s in keys:
        rs = [r for r in rel if r["block"] == b and r["station"] == s]
        mast = [r for r in rs if r["family"] in "NZ"]
        gs = [g for g in groups if g["block"] == b and g["station"] == s]
        out.append({"block": b, "station": s, "frames": len(rs), "mast_frames": len(mast),
                    "mast_with_body": sum(measured(r) for r in mast),
                    "mast_propagated": sum(r["body"].endswith("_propagated") for r in mast),
                    "front_bodies": sum(g["family"] == "F" for g in gs), "rear_bodies": sum(g["family"] == "R" for g in gs),
                    "POSE_values": sorted({r["POSE"] for r in rs if r["POSE"] is not None})})
    return out


def run(images: Sequence[Dict[str, Any]], states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The whole study on prepared images (see the module docstring)."""
    reasons: Dict[str, int] = {}
    rows = image_rows(images, states, reasons)
    groups = body_groups(rows)
    rel = relate(rows, groups)
    changes = body_changes(groups)
    mast_groups = sorted({r["group"] for r in rel if r["family"] in "NZ"})
    fits = {}
    for g in mast_groups:
        fits[g] = {"pooled": pointing_fit(rel, g), "measured": pointing_fit(rel, g, bodies="measured"),
                   "per_block": pointing_fit(rel, g, per_block=True), "resolver": pointing_fit(rel, g, angles="resolver"),
                   "telemetry": pointing_fit(rel, g, bodies="all", per_block=True)}
    flat_groups = [{k: v for k, v in g.items() if not k.startswith(("R_", "O_"))} |
                   {f"O_body_{a}_m": float(x) for a, x in zip(AXES, g["O_body"])} for g in groups]
    return {"rows": rel, "groups": flat_groups, "changes": changes, "stations": station_summary(rel, groups),
            "mounts": mounts(rel), "mounts_propagated": mounts(rel, "propagated"), "mounts_telemetry": mounts(rel, "all"),
            "fits": fits, "body_sources": {k: sum(r["body"] == k for r in rel) for k in sorted({r["body"] for r in rel})},
            "n_images": len(images), "n_used": len(rows), "left_out": reasons,
            "_rows": rows, "_groups": groups}            # with the matrices: rover_frame.estimate (not written)
