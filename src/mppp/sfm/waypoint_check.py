"""
Waypoint localisation check (MPPP v0p82): which rover stations' waypoint positions disagree with the bundle adjustment.

A block's refined camera centres are compared with their waypoint priors station by station, after the best
similarity transform (scale, rotation, translation) between the two - fitted robustly (Huber) on the stations that
kept their prior, so that one bad localisation does not bend the fit.  Only *surveyed* stations - those with a
waypoint of their own (``position_source`` "waypoint site N drive D (exact)") - enter the fit: a station placed from
the rover's onboard estimate (site origin + label ORIGIN_OFFSET_VECTOR, :func:`mppp.sfm.reconstruction.is_unsurveyed`)
is checked against the fit but says nothing about the block's scale (0.90: the scale check of 19 Oct was skewed by
such stations - Hippo Pools -3418 ppm from 5 stations, 3 of them unsurveyed).  What is left per station is its localisation
error relative to the others: the manual localisations against the HiRISE basemap are good to about 0.3-0.5 m (JPL,
Censible, ICRA 2024), a station far beyond that - or far beyond the block's own scatter - is flagged.

    from mppp.sfm import waypoint_check as WC
    r = WC.check_block("<scapes root>/mars2020_sol_0072_chal_rocks_large/colmap")
    m = WC.mission_flags([r, ...])                  # stations flagged in any block; written by write_flags()

Flag rule (:data:`FLAG_ABS_M`, :data:`FLAG_MIN_M`, :data:`FLAG_K`): a station whose horizontal residual is above
``FLAG_ABS_M``, or above both ``FLAG_MIN_M`` and ``FLAG_K`` times the block's robust horizontal scatter.  Stations
without a prior in the run (short stops, ``LOCALIZE_MIN_IMAGES``) are checked the same way, against the fit of the
others, but marked so (their offsets come from the tie points alone).  The pipeline's ``waypoint_check`` stage writes
``<root>/camera_analysis/waypoint_flags_<date>/waypoint_flags.json`` (+ .csv, .md); notebook 03
``NO_POSITION_PRIOR_AT`` leaves the flagged stations' position priors out (the baseline stage does).
"""
from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .reconstruction import is_unsurveyed

PathLike = Union[str, Path]
FLAG_ABS_M = 1.0          # horizontal residual above this: always flagged (about 2-3x the manual localisation accuracy)
FLAG_MIN_M = 0.5          # ... and above this and FLAG_K x the block's robust scatter
FLAG_K = 3.0
MIN_OBSERVATIONS = 30     # images with fewer tie-point observations are not used (their pose is weak)
MIN_STATIONS_SIMILARITY = 5   # fewer stations with priors: translation only
MIN_EXTENT_SIMILARITY_M = 30.0  # stations with priors spread over less than this: translation only (scale and rotation
                              # of a small block are not determined by its waypoints; Chal Rocks Sid, 18 m: -1.6 %)
HUBER_M = 0.3


def _project_dir(p: PathLike) -> Path:
    p = Path(p)
    return p / "colmap" if (p / "colmap" / "project.json").is_file() else p


def station_rows(project_dir: PathLike, min_observations: int = MIN_OBSERVATIONS) -> List[Dict[str, Any]]:
    """Per station: image-weighted mean refined centre and prior centre (local frame, m), images, sols, prior kept /
    dropped - from ``error_input/poses.csv`` (refined position and refined - prior) and ``project.json``."""
    pdir = _project_dir(project_dir)
    P = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
    st = P.get("settings") or {}
    rs = st.get("reconstruction") or {}
    unloc = rs.get("unlocalized") or {}
    flagged = set(st.get("no_position_prior_at") or [])
    surveyed_images = {str(r.get("name")): not is_unsurveyed(r) for r in (P.get("images") or [])}
    rows: Dict[str, List[Dict[str, str]]] = {}
    with open(pdir / "error_input" / "poses.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(float(r.get("observations") or 0)) >= min_observations:
                rows.setdefault(r["station"], []).append(r)
    out = []
    for s, g in rows.items():
        ref = np.array([[float(r["E_m"]), float(r["N_m"]), float(r["U_m"])] for r in g])
        d = np.array([[float(r["dE_m"]), float(r["dN_m"]), float(r["dU_m"])] for r in g])
        prior = ref - d
        sols = [int(r["sol"]) for r in g]
        how = "dropped (flagged)" if s in flagged else (unloc.get(s) or {}).get("prior", "kept")
        surveyed = all(surveyed_images.get(r["name"], True) for r in g)   # no project rows (old runs): surveyed
        out.append({"station": s, "site": int(s[1:4]) if s[1:4].isdigit() else None,
                    "drive": int(s[5:]) if s[5:].isdigit() else None, "sols": [min(sols), max(sols)],
                    "images": len(g), "refined": ref.mean(0), "prior": prior.mean(0), "position_prior": how,
                    "surveyed": surveyed})
    return sorted(out, key=lambda r: (r["sols"][0], r["station"]))


def similarity_fit(src: np.ndarray, dst: np.ndarray, w: Optional[np.ndarray] = None, huber_m: float = HUBER_M,
                   iterations: int = 20, translation_only: bool = False, horizontal: bool = True) -> Dict[str, Any]:
    """Robust (Huber, IRLS) similarity ``dst ~ s R src + t`` (Umeyama per iteration); ``w`` base weights.
    ``horizontal`` (0.90, the default): the rotation is a yaw about the vertical axis and the scale is fitted in
    the east-north plane - the waypoint elevations are a few decimetres rough, and a free 3-D rotation turned
    them into a tilt that posed as scale and rotation (Enchanted Lake, 6 stations over 55 m: 1.7 deg, -4300 ppm,
    with horizontal offsets of 0.1 m)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    w0 = np.ones(n) if w is None else np.asarray(w, float)
    wr = np.ones(n)
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    dims = slice(0, 2) if horizontal else slice(0, 3)
    for _ in range(iterations):
        ww = w0 * wr
        ww = ww / ww.sum()
        ms, md = ww @ src, ww @ dst
        if translation_only:
            s, R, t = 1.0, np.eye(3), md - ms
        else:
            A, B = (src - ms)[:, dims], (dst - md)[:, dims]
            C = (B * ww[:, None]).T @ A
            U, S, Vt = np.linalg.svd(C)
            D = np.eye(len(S))
            D[-1, -1] = np.sign(np.linalg.det(U @ Vt))
            Rd = U @ D @ Vt
            var = float(np.sum(ww * np.sum(A ** 2, 1)))
            s = float(np.trace(np.diag(S) @ D) / var) if var > 0 else 1.0
            R = np.eye(3)
            R[dims, dims] = Rd
            t = md - s * R @ ms
        r = np.linalg.norm((dst - (s * src @ R.T + t))[:, dims], axis=1)
        wr = np.where(r <= huber_m, 1.0, huber_m / np.maximum(r, 1e-9))
        if _ >= iterations // 2:                  # second half: stations far beyond the scatter get no weight
            sd = 1.4826 * float(np.median(r))
            wr = np.where(r > max(3.0 * sd, 2 * huber_m), 0.0, wr)
            if wr.sum() < 3:
                wr = np.ones(n)
    rot = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return {"scale": float(s), "R": R, "t": t, "rotation_deg": float(rot), "yaw_deg": yaw,
            "translation_only": bool(translation_only), "horizontal": bool(horizontal)}


def check_block(project_dir: PathLike, name: Optional[str] = None, abs_m: float = FLAG_ABS_M, min_m: float = FLAG_MIN_M,
                k: float = FLAG_K) -> Dict[str, Any]:
    """The localisation check of one block (module docstring).  Returns {"block", "project", "stations": [...],
    "similarity": {scale_ppm, rotation_mdeg, yaw_mdeg, translation_m, translation_only}, "scatter_h_m", "flagged"}."""
    pdir = _project_dir(project_dir)
    rows = station_rows(pdir)
    out: Dict[str, Any] = {"block": name or pdir.parent.name, "project": str(pdir), "stations": [], "flagged": []}
    kept_all = [r for r in rows if r["position_prior"] == "kept" or str(r["position_prior"]).startswith("kept")]
    kept = [r for r in kept_all if r.get("surveyed", True)]           # the fit: waypoint stations only
    n_unsurveyed = sum(not r.get("surveyed", True) for r in rows)
    if len(kept) < 2:
        out["note"] = (f"{len(kept)} surveyed stations with a position prior ({n_unsurveyed} from the onboard "
                       f"estimate): nothing to compare")
        out["stations"] = [_jsonable(r) for r in rows]
        return out
    src = np.array([r["prior"] for r in kept])
    dst = np.array([r["refined"] for r in kept])
    w = np.sqrt([r["images"] for r in kept])
    ext = float(np.max(np.linalg.norm(src[:, None, :2] - src[None, :, :2], axis=-1)))
    fit = similarity_fit(src, dst, w, translation_only=len(kept) < MIN_STATIONS_SIMILARITY or ext < MIN_EXTENT_SIMILARITY_M)
    for r in rows:
        pred = fit["scale"] * fit["R"] @ r["prior"] + fit["t"]
        res = r["refined"] - pred
        r["raw_m"] = (r["refined"] - r["prior"]).tolist()
        r["residual_m"] = res.tolist()
        r["residual_h_m"] = float(np.hypot(res[0], res[1]))
    rk = np.array([[r["residual_m"][0], r["residual_m"][1]] for r in kept])
    mad = np.median(np.abs(rk - np.median(rk, 0)), 0) * 1.4826
    scatter = float(max(np.hypot(*mad), 0.05))
    # the scale's standard error from the scatter and the stations' horizontal spread about their centroid (the
    # lever arm of a scale): a block whose waypoints sit in two clusters 50 m apart has a 1000-ppm-class se, and a
    # localisation offset between rover sites then poses as scale (Enchanted Lake, 0.90)
    hc = src[:, :2] - np.average(src[:, :2], axis=0, weights=w ** 2)
    lever = float(np.sqrt(np.sum(hc ** 2)))
    scale_se_ppm = float(1e6 * scatter / lever) if lever > 0 and not fit["translation_only"] else None
    for r in rows:
        rh = r["residual_h_m"]
        r["flag"] = bool(r.get("surveyed", True) and (rh > abs_m or (rh > min_m and rh > k * scatter)))
        r["flag_reason"] = ("" if not r.get("surveyed", True) else      # an onboard estimate is expected to be off
                            f"{rh:.2f} m > {abs_m:g} m" if rh > abs_m else
                            f"{rh:.2f} m > {min_m:g} m and > {k:g} x scatter {scatter:.2f} m" if r["flag"] else "")
    n_kept = len(kept)
    out.update(similarity={"scale_ppm": 1e6 * (fit["scale"] - 1.0), "scale_se_ppm": scale_se_ppm,
                           "rotation_mdeg": 1e3 * fit["rotation_deg"],
                           "yaw_mdeg": 1e3 * fit["yaw_deg"], "translation_m": fit["t"].tolist(),
                           "translation_only": fit["translation_only"], "stations": n_kept,
                           "unsurveyed_stations": n_unsurveyed, "extent_m": ext},
               scatter_h_m=scatter, stations=[_jsonable(r) for r in rows],
               flagged=[r["station"] for r in rows if r["flag"]],
               mean_raw_offset_m=np.average([r["raw_m"] for r in kept], axis=0,
                                            weights=[r["images"] for r in kept]).tolist())
    out["anisotropy"] = scale_anisotropy(kept, fit)
    by_site: Dict[int, List[Dict[str, Any]]] = {}
    for r in kept:
        by_site.setdefault(r["site"], []).append(r)
    out["by_rover_site"] = {str(s): {"stations": len(g), "residual_m": np.average([x["residual_m"] for x in g], axis=0,
                                                                                   weights=[x["images"] for x in g]).tolist()}
                            for s, g in sorted(by_site.items())}
    return out


def scale_anisotropy(kept: Sequence[Dict[str, Any]], fit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """After the similarity: a further scale along east and along north fitted to the residuals (ppm, with se) -
    a stereo baseline error scales the block alike in every direction, a map-projection error of the waypoints
    (equirectangular maps stretch east-west; JPL, ICRA 2024: about 1 m per 500 m) mostly east-west.  None for a
    translation-only fit."""
    if fit.get("translation_only") or len(kept) < MIN_STATIONS_SIMILARITY:
        return None
    P = np.array([r["prior"] for r in kept])
    Rz = np.array([r["residual_m"] for r in kept])
    w = np.array([r["images"] for r in kept], float)
    w = w / w.mean()                                   # weights by images, normalised to a mean of 1
    out = {}
    for ax, name in ((0, "E"), (1, "N")):
        x = P[:, ax] - np.average(P[:, ax], weights=w)
        y = Rz[:, ax]
        sxx = float(np.sum(w * x * x))
        if sxx <= 0:
            return None
        b = float(np.sum(w * x * y) / sxx)
        r = y - b * x
        se = float(np.sqrt(np.sum(w * r * r) / (len(x) - 1) / sxx)) if len(x) > 2 else None
        out[f"scale_{name}_ppm"] = 1e6 * b
        out[f"se_{name}_ppm"] = None if se is None else 1e6 * se
    return out


def _jsonable(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def mission_flags(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Every station over the checked blocks: its worst residual, the blocks, and whether any block flagged it.
    ``flagged`` is the list for notebook 03 ``NO_POSITION_PRIOR_AT``."""
    st: Dict[str, Dict[str, Any]] = {}
    for b in results:
        for r in b.get("stations") or []:
            if "residual_h_m" not in r:
                continue
            e = st.setdefault(r["station"], {"station": r["station"], "site": r.get("site"), "drive": r.get("drive"),
                                             "sols": r["sols"], "blocks": [], "worst_residual_h_m": 0.0,
                                             "flag": False, "reasons": []})
            e["blocks"].append(b["block"])
            e["sols"] = [min(e["sols"][0], r["sols"][0]), max(e["sols"][1], r["sols"][1])]
            if r["residual_h_m"] > e["worst_residual_h_m"]:
                e["worst_residual_h_m"] = r["residual_h_m"]
                e["worst_residual_m"] = r["residual_m"]
                e["position_prior"] = r["position_prior"] + ("" if r.get("surveyed", True) else " (onboard estimate)")
            if r.get("flag"):
                e["flag"] = True
                e["reasons"].append(f"{b['block']}: {r['flag_reason']}")
    rows = sorted(st.values(), key=lambda e: (e["sols"][0], e["station"]))
    return {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "rule": {"abs_m": FLAG_ABS_M, "min_m": FLAG_MIN_M, "k": FLAG_K, "min_observations": MIN_OBSERVATIONS},
            "blocks": [{k: b.get(k) for k in ("block", "project", "similarity", "scatter_h_m", "flagged", "note",
                                              "mean_raw_offset_m", "by_rover_site", "anisotropy")} for b in results],
            "stations": rows, "flagged": [e["station"] for e in rows if e["flag"]]}


def write_flags(m: Dict[str, Any], out_dir: PathLike) -> Path:
    """``waypoint_flags.json``, ``.csv`` (one row per station) and ``.md`` in ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "waypoint_flags.json").write_text(json.dumps(m, indent=1, default=float), encoding="utf-8")
    with open(out / "waypoint_flags.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "site", "drive", "first_sol", "last_sol", "worst_residual_h_m", "dE_m", "dN_m", "dU_m",
                    "position_prior", "flag", "blocks", "reasons"])
        for e in m["stations"]:
            r = e.get("worst_residual_m") or [None] * 3
            w.writerow([e["station"], e["site"], e["drive"], e["sols"][0], e["sols"][1],
                        f"{e['worst_residual_h_m']:.3f}", *(f"{x:+.3f}" if x is not None else "" for x in r),
                        e.get("position_prior"), int(e["flag"]), " ".join(e["blocks"]), " | ".join(e["reasons"])])
    L = [f"# Waypoint localisation check ({m['created'][:10]})", "",
         f"Rule: horizontal residual after the block's robust similarity > {m['rule']['abs_m']:g} m, or > "
         f"{m['rule']['min_m']:g} m and > {m['rule']['k']:g} x the block's scatter.", "",
         "| block | stations (surveyed + onboard) | scale ppm | rotation mdeg | mean raw offset E N U m | scatter m | flagged |",
         "|---|---|---|---|---|---|---|"]
    for b in m["blocks"]:
        s = b.get("similarity") or {}
        mo = b.get("mean_raw_offset_m")
        se = s.get("scale_se_ppm")
        L.append(f"| {b['block']} | {s.get('stations', '-')} + {s.get('unsurveyed_stations', 0)} | "
                 f"{s.get('scale_ppm', float('nan')):+.0f}{'' if se is None else f' ± {se:.0f}'} | "
                 f"{s.get('rotation_mdeg', float('nan')):.0f} | "
                 + (" ".join(f"{x:+.2f}" for x in mo) if mo else "-")
                 + f" | {b.get('scatter_h_m') or float('nan'):.2f} | {', '.join(b.get('flagged') or []) or '-'} |")
    L += ["", f"**{len(m['flagged'])} flagged stations**: " + (", ".join(m["flagged"]) or "none"), "",
          "| station | sols | worst residual m (E, N, U) | prior | blocks | reason |", "|---|---|---|---|---|---|"]
    for e in m["stations"]:
        if e["flag"] or e["worst_residual_h_m"] > FLAG_MIN_M:
            r = e.get("worst_residual_m") or []
            L.append(f"| {e['station']}{' **flag**' if e['flag'] else ''} | {e['sols'][0]}-{e['sols'][1]} | "
                     f"{e['worst_residual_h_m']:.2f} ({', '.join(f'{x:+.2f}' for x in r)}) | {e.get('position_prior')} | "
                     f"{', '.join(e['blocks'])} | {'; '.join(e['reasons'])} |")
    (out / "waypoint_flags.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
