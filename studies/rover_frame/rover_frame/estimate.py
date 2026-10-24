"""
The rover attitude and position per rover pose, from the adjusted blocks (study 3A, 0.94.5, Christian 24 Oct 2026),
for the SPICE kernels of :mod:`rover_frame.spice`.  No new adjustment: the Hazcam body poses of
:func:`rover_frame.analysis.body_groups` combined by the end-of-drive rule.

- **End of drive (EOD)**: the first rover pose of a station (its first SITE / DRIVE / POSE / ARM in time) with a front
  and a rear Hazcam pair: one body attitude and position from both pairs (source ``F+R``).
- **Later poses**: the front Hazcams (``front``), else the rear ones (``rear``); a pose with no Hazcam frame gets the
  station's measured body error carried over by the telemetry change (``propagated``); a station with no Hazcam
  frame keeps the telemetry (``telemetry``).  A later measurement at the same pose starts a new interval.
- **Global terms over all blocks**: the rear-to-front Hazcam correction ``Q`` (the rear body is ``Q R_rear``; the
  labels' rear mounts disagree with the front ones by about 0.1 deg, Chal Rocks) with the rear-to-front origin offset
  ``t``, and the rover frame alignment ``A``.
- **Rover frame convention**: ``M2020_ROVER`` is the IMU frame on average - zero mean tilt (x, y) residual of the EOD
  bodies against the telemetry; the heading is the front Hazcams' (the block frame's, set by the position priors).
- **Uncertainty**: per measured interval the scatter of its frames; per propagated / telemetry interval the scatter
  of the EOD body minus telemetry over all stations (tilt and heading separately).

Rotations ``R_a_b`` map b coordinates to a coordinates; ``R_body`` = ``R_rnav_w`` (w: the block frame, ENU of the
landing site minus the project offset).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as G

MDEG = 1000.0 * 180.0 / np.pi
MARGIN_S = 0.5                       # an interval starts / ends this long before / after its first / last exposure


def _key(r: Dict[str, Any]) -> Tuple:
    m = r.get("rmc") or {}
    return (r["block"], r["station"], m.get("POSE"), m.get("ARM"))


def eod_pairs(rows: Sequence[Dict[str, Any]], groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (block, station) the front and rear Hazcam groups of its first rover pose, when both are there."""
    first: Dict[Tuple, Tuple[float, Any, Any]] = {}
    for r in rows:
        k = (r["block"], r["station"])
        t = float(r.get("sclk") or 0)
        if k not in first or t < first[k][0]:
            first[k] = (t, (r.get("rmc") or {}).get("POSE"), (r.get("rmc") or {}).get("ARM"))
    out = []
    for (b, s), (_, pose, arm) in sorted(first.items(), key=lambda kv: str(kv[0])):
        here = [g for g in groups if g["block"] == b and g["station"] == s and g["POSE"] == pose and g["ARM"] == arm]
        f = sorted([g for g in here if g["family"] == "F"], key=lambda g: g["sclk"])
        rr = sorted([g for g in here if g["family"] == "R"], key=lambda g: g["sclk"])
        if f and rr:
            out.append({"block": b, "station": s, "POSE": pose, "ARM": arm, "front": f[0], "rear": rr[0]})
    return out


def global_terms(pairs: Sequence[Dict[str, Any]], groups: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The rear-to-front correction Q, the rear-to-front origin offset t (front body coordinates, m), the rover frame
    alignment A and the telemetry scatter (sigmas), from the EOD pairs (identity / zero / defaults without any)."""
    out: Dict[str, Any] = {"n_eod": len(pairs)}
    if pairs:
        Qs = [p["front"]["R_body"] @ p["rear"]["R_body"].T for p in pairs]
        Q = G.mean_rotation(Qs)
        q = np.array([G.rotvec(x @ Q.T) for x in Qs]) * MDEG
        t = np.mean([p["front"]["R_body"] @ (p["rear"]["O_body"] - p["front"]["O_body"]) for p in pairs], axis=0)
    else:
        Q, q, t = np.eye(3), np.zeros((0, 3)), np.zeros(3)
    out.update(Q=Q, Q_mdeg=(G.rotvec(Q) * MDEG).tolist(), Q_sd_mdeg=(q.std(axis=0, ddof=1).tolist() if len(q) > 1 else None),
               t_m=t.tolist())
    # the EOD bodies (front and corrected rear) against the telemetry, in the rover frame
    pairs_BT = [(G.mean_rotation([p["front"]["R_body"], Q @ p["rear"]["R_body"]]),
                 G.mean_rotation([p["front"]["R_tel"], p["rear"]["R_tel"]])) for p in pairs]
    if not pairs_BT:                 # no EOD pair: the stations' front bodies
        pairs_BT = [(g["R_body"], g["R_tel"]) for g in groups if g["family"] == "F"]
    A = np.eye(3)
    if len(pairs_BT) >= 2:
        for _ in range(6):           # tilt only: M2020_ROVER = the IMU frame on average (iterated: exact mean)
            r = np.array([G.residual_rnav(A @ B, T) for B, T in pairs_BT])
            d = np.array([*r[:, :2].mean(axis=0), 0.0])
            A = G.exp(-d) @ A
            if np.abs(d).max() < 1e-13:
                break
    after = np.array([G.residual_rnav(A @ B, T) for B, T in pairs_BT], float).reshape(-1, 3)
    wbar = -G.rotvec(A)
    out.update(A=A, alignment_mdeg=(wbar * MDEG).tolist(), n_alignment=int(len(pairs_BT)))
    sd = after.std(axis=0, ddof=1) * MDEG if len(after) > 1 else np.array([np.nan] * 3)
    out.update(sigma_tilt_mdeg=float(np.sqrt(np.nanmean(sd[:2] ** 2))) if len(after) > 1 else None,
               sigma_heading_mdeg=float(sd[2]) if len(after) > 1 else None,
               tel_residual_mean_mdeg=(after.mean(axis=0) * MDEG).tolist() if len(after) else None)
    return out


def _measured_body(g: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(R_rover_w, O_w) of one Hazcam group: the rear ones through Q and t, both through A."""
    if g["family"] == "F":
        R, O = g["R_body"], g["O_body"]
    else:
        R = gt["Q"] @ g["R_body"]
        O = g["O_body"] - R.T @ np.asarray(gt["t_m"], float)
    return gt["A"] @ R, O


def intervals(rows: Sequence[Dict[str, Any]], groups: Sequence[Dict[str, Any]],
              gt: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The rover attitude intervals of every block with at least one Hazcam body (see the module docstring), and the
    global terms.  Each interval: block, station, SITE, DRIVE, POSE, ARM, source, sclk_start / sclk_end (s),
    R_rover_w, O_w, offset (the block's ENU offset), the residual against the telemetry and sigma (mdeg)."""
    pairs = eod_pairs(rows, groups)
    gt = gt or global_terms(pairs, groups)
    eod_ids = {p["front"]["id"]: p for p in pairs} | {p["rear"]["id"]: p for p in pairs}
    by_pose: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_pose[_key(r)].append(r)
    with_hazcam = {g["block"] for g in groups}
    out: List[Dict[str, Any]] = []
    for k, rs in sorted(by_pose.items(), key=lambda kv: (kv[0][0], min(float(x.get("sclk") or 0) for x in kv[1]))):
        block, station, pose, arm = k
        if block not in with_hazcam:
            continue
        ts = [float(x.get("sclk") or 0) for x in rs]
        R_tel = G.mean_rotation([x["R_tel"] for x in rs])
        O_tel = np.mean([x["O_tel"] for x in rs], axis=0)
        m = rs[0].get("rmc") or {}
        base = {"block": block, "station": station, "SITE": m.get("SITE"), "DRIVE": m.get("DRIVE"), "POSE": pose,
                "ARM": arm, "images": len(rs), "offset": rs[0].get("offset"), "sol": rs[0].get("sol")}
        here = sorted([g for g in groups if g["block"] == block and g["station"] == station and g["POSE"] == pose
                       and g["ARM"] == arm], key=lambda g: g["sclk"])
        events: List[Tuple[float, str, np.ndarray, np.ndarray, float, List[str], int]] = []
        used = set()
        for g in here:
            if g["id"] in used:
                continue
            p = eod_ids.get(g["id"])
            if p is not None:
                Rf, Of = _measured_body(p["front"], gt)
                Rr, Or = _measured_body(p["rear"], gt)
                R, O = G.mean_rotation([Rf, Rr]), (Of + Or) / 2.0
                used |= {p["front"]["id"], p["rear"]["id"]}
                sig = max(p["front"]["spread_mdeg"], p["rear"]["spread_mdeg"], G.angle_deg(Rf @ Rr.T) * 1000.0 / 2)
                events.append((min(p["front"]["sclk"], p["rear"]["sclk"]), "F+R", R, O, sig,
                               [p["front"]["id"], p["rear"]["id"]], p["front"]["frames"] + p["rear"]["frames"]))
            else:
                R, O = _measured_body(g, gt)
                used.add(g["id"])
                events.append((g["sclk"], "front" if g["family"] == "F" else "rear", R, O, g["spread_mdeg"], [g["id"]],
                               g["frames"]))
        # front before rear when both measure one pose at nearly the same time: keep the time order only
        events.sort(key=lambda e: e[0])
        if events and len(events) > 1:
            # a rear-only measurement right after a front one of the same pose adds nothing: the front one stays
            keep = [events[0]]
            for e in events[1:]:
                if e[1] == "rear" and keep[-1][1] in ("front", "F+R"):
                    continue
                keep.append(e)
            events = keep
        sig_prop = gt.get("sigma_heading_mdeg")
        if not events:
            st = sorted([g for g in groups if g["block"] == block and g["station"] == station],
                        key=lambda g: (g["family"] != "F", abs(g["sclk"] - min(ts))))
            if st:
                g = st[0]
                Rg, Og = _measured_body(g, gt)
                R = Rg @ g["R_tel"].T @ R_tel                # the station's body error, this pose's telemetry
                O = Og + (O_tel - g["O_tel"])
                events = [(min(ts), "propagated", R, O, sig_prop, [g["id"]], 0)]
            else:
                events = [(min(ts), "telemetry", R_tel, O_tel, sig_prop, [], 0)]
        t0 = max(0.0, min(ts + [e[0] for e in events]) - MARGIN_S)
        t1 = max(ts + [e[0] for e in events]) + MARGIN_S
        starts = [t0] + [e[0] for e in events[1:]]
        ends = starts[1:] + [t1]
        for (t, src, R, O, sig, ids, nfr), a, b in zip(events, starts, ends):
            out.append(dict(base, source=src, sclk_start=float(a), sclk_end=float(b), measured_sclk=float(t),
                            R_rover_w=R, O_w=np.asarray(O, float), sigma_mdeg=None if sig is None else float(sig),
                            groups=";".join(ids), hazcam_frames=int(nfr),
                            **{f"tel_residual_{x}_mdeg": float(v) for x, v in
                               zip("xyz", G.residual_rnav(R, R_tel) * MDEG)}))
    # consecutive intervals of one block must not overlap (a pose that began before the last one ended)
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        if a["block"] == b["block"] and a["sclk_end"] >= b["sclk_start"]:
            a["sclk_end"] = max(a["sclk_start"] + 1.0 / 65536.0 * 2, b["sclk_start"] - 1.0 / 65536.0)
    return out, gt


def hazcam_mounts(rows: Sequence[Dict[str, Any]], rel: Sequence[Dict[str, Any]], gt: Dict[str, Any]) -> Dict[str, Any]:
    """The four Hazcam frames in M2020_ROVER (R_c_rover: rover -> camera, C_rover in m) from the label mounts, the
    global terms and the right eyes' rig residuals against their own pair (``analysis.relate`` rows, ``own_pair``)."""
    lab: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for r in rows:
        if r["family"] in "FR" and r["group"] not in lab:
            lab[r["group"]] = (np.asarray(r["R_c_rnav"], float), np.asarray(r["C_rnav"], float))
    own: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for x in rel:
        if x["family"] in "FR" and x["eye"] == "R" and x["body"] == "own_pair":
            own[x["group"]].append((np.array([x[f"e_rnav_{a}_mdeg"] for a in "xyz"]) / MDEG,
                                    np.array([x[f"d_rnav_{a}_mm"] for a in "xyz"]) / 1000.0))
    A, Q, t = gt["A"], gt["Q"], np.asarray(gt["t_m"], float)
    out = {}
    for grp, (Rl, Cl) in sorted(lab.items()):
        fam, eye = grp[0], grp[1]
        R_c_body, C_body = Rl, Cl                        # in the frame of its own pair's body (left eye = label)
        n = 0
        if eye == "R" and own.get(grp):
            e = np.median([v[0] for v in own[grp]], axis=0)
            d = np.median([v[1] for v in own[grp]], axis=0)
            R_c_body, C_body, n = Rl @ G.exp(e), Cl + d, len(own[grp])
        if fam == "R":                                   # the rear body in the front body's frame: Q, t
            R_c_body = R_c_body @ Q.T
            C_body = Q @ C_body + t
        out[grp] = {"R_c_rover": R_c_body @ A.T, "C_rover_m": A @ C_body, "rig_images": n}
    return out
