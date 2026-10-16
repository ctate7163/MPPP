"""
Diagnosis of one finished Navcam block (MPPP v0p80): what limits it, and does it differ between its visits?

Made for Chal Rocks Sid (sols 91-101 and 360-378, on both sides of the sol-300 pitch knot), whose cross-station
residual was the largest of the four error-analysis blocks (0.273 vs 0.226 px within stations) and whose registration
to the waypoints was twice as loose (0.25 m) - but it works on any WORK folder.

:func:`diagnose_block` reads ``<project>/project.json`` and ``<project>/error_input`` (the native-pixel model, poses.csv)
and returns:

- ``visits``: the Navcam visits (rig epochs of the project, else runs of sols split at gaps > 50 sols) with sols, LMST,
  camera temperature, images;
- ``residuals``: per-axis rms (= epsilon) of the observations of tie points seen within one station, across stations
  of one visit, and across visits; per eye; per camera-temperature bin of 5 degC;
- ``registration``: per visit the median offset of the refined camera centres from their waypoint priors (dE, dN,
  dU, |d|) - a visit whose localisation is off moves as a whole;
- ``rig``: the rig rotation per epoch after the run against its start (``reconstruction.rig_fit``, MPPP >= 0.80) and
  the thermal bins (fx, fy, k1 changes);
- ``settings``: the camera-model settings the run used.

:func:`compare` puts two runs (e.g. ``colmap`` of 5 Oct and ``colmap_v0p80``) side by side; :func:`write_report`
writes ``diagnose.json``, ``diagnose.md`` and ``diagnose.png`` (2 x 2: residual per image against sol and against
temperature, NL cool / NR warm; station offsets by visit; epsilon by track class for the runs compared).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]
VISIT_GAP_SOLS = 50.0
T_BIN_DEG = 5.0


def _visits(rows: Sequence[Dict[str, Any]], project_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The Navcam visits: the project's rig epochs if it has them (MPPP >= 0.75), else runs of sols."""
    sols = sorted({int(r["sol"]) for r in rows if r.get("sol") is not None and str(r.get("instrument", "")).startswith("N")})
    if not sols:
        return []
    ep = project_settings.get("navcam_rig_epochs") or {}
    table = ep.get("epochs") if isinstance(ep, dict) else None
    out = []
    if table:
        for e in table:
            lo, hi = e.get("sol_min", e.get("sols", [None, None])[0]), e.get("sol_max", e.get("sols", [None, None])[-1])
            if lo is None or hi is None:
                continue
            out.append({"name": str(e.get("rig") or e.get("key") or f"epoch {len(out)}"), "sols": [int(lo), int(hi)]})
    if not out:
        breaks = [float(x) for x in (project_settings.get("navcam_rig_epochs_breaks") or [300.0])]
        a = sols[0]
        prev = sols[0]
        for s in sols[1:]:
            if s - prev > VISIT_GAP_SOLS or any(prev < b <= s for b in breaks):
                out.append({"name": f"visit {len(out) + 1}", "sols": [a, prev]})
                a = s
            prev = s
        out.append({"name": f"visit {len(out) + 1}", "sols": [a, prev]})
    return out


def _visit_of(sol: Optional[int], visits: Sequence[Dict[str, Any]]) -> Optional[int]:
    if sol is None:
        return None
    for k, v in enumerate(visits):
        if v["sols"][0] <= int(sol) <= v["sols"][1]:
            return k
    return int(np.argmin([min(abs(int(sol) - v["sols"][0]), abs(int(sol) - v["sols"][1])) for v in visits]))


def _lmst_h(x) -> Optional[float]:
    try:
        from ..error.alignment import _lmst_hours
        return _lmst_hours(x)
    except Exception:                                             # noqa: BLE001
        return None


def _rms_axis(r: np.ndarray) -> Optional[float]:
    return float(np.sqrt(np.mean(r ** 2) / 2.0)) if len(r) else None


def diagnose_block(project_dir: PathLike, max_residual_px: float = 2.0) -> Dict[str, Any]:
    """See the module docstring.  ``project_dir``: ``<WORK>/colmap`` or ``<WORK>/colmap_<variant>``."""
    import pycolmap
    pdir = Path(project_dir)
    P = json.loads((pdir / "project.json").read_text(encoding="utf-8"))
    st = P.get("settings") or {}
    rows = P.get("images") or []
    meta = {r["name"]: r for r in rows}
    visits = _visits(rows, st)
    out: Dict[str, Any] = {"project": str(pdir), "site": pdir.parent.name, "visits": visits}
    for k, v in enumerate(visits):
        rr = [r for r in rows if str(r.get("instrument", "")).startswith("N") and _visit_of(r.get("sol"), visits) == k]
        Ts = [float(r["camera_temperature_degC"]) for r in rr if r.get("camera_temperature_degC") is not None]
        Ls = [h for h in (_lmst_h(r.get("lmst")) for r in rr) if h is not None]
        v.update({"images": len(rr), "stations": len({r.get("station") for r in rr}),
                  "T_degC": [float(np.min(Ts)), float(np.median(Ts)), float(np.max(Ts))] if Ts else None,
                  "lmst_h": [float(np.min(Ls)), float(np.max(Ls))] if Ls else None})
    rec = pycolmap.Reconstruction(str(pdir / "error_input" / "native"))
    xyz = {pid: p.xyz for pid, p in rec.points3D.items()}
    # every observation: residual (native px), its image's station / visit / eye / temperature
    obs_pid, obs_r, obs_img = [], [], []
    per_image: Dict[str, List[float]] = {}
    for iid, im in rec.images.items():
        if not im.has_pose:
            continue
        cam = rec.cameras[im.camera_id]
        pts = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
        if not pts:
            continue
        xy = np.array([q[0] for q in pts], float)
        X = np.array([xyz[q[1]] for q in pts], float)
        T = im.cam_from_world()
        Xc = X @ np.asarray(T.rotation.matrix()).T + np.asarray(T.translation)
        ok = Xc[:, 2] > 0
        uv = np.full_like(xy, np.nan)
        if ok.any():
            uv[ok] = np.asarray(cam.img_from_cam(Xc[ok]), float)
        r = xy - uv
        good = np.all(np.isfinite(r), 1) & (np.linalg.norm(r, axis=1) < max_residual_px)
        obs_pid += [q[1] for q, g in zip(pts, good) if g]
        obs_r.append(r[good])
        obs_img += [im.name] * int(good.sum())
        per_image[im.name] = [float(np.sqrt(np.mean(np.sum(r[good] ** 2, 1)) / 2.0)) if good.any() else float("nan"),
                              int(good.sum())]
    R = np.concatenate(obs_r) if obs_r else np.zeros((0, 2))
    pid = np.asarray(obs_pid)
    station = np.array([str(meta.get(n, {}).get("station")) for n in obs_img])
    visit = np.array([_visit_of(meta.get(n, {}).get("sol"), visits) if meta.get(n) else -1 for n in obs_img])
    eye = np.array([str(meta.get(n, {}).get("base_instrument") or meta.get(n, {}).get("instrument", ""))[:2] for n in obs_img])
    temp = np.array([float(meta.get(n, {}).get("camera_temperature_degC")) if meta.get(n, {}).get("camera_temperature_degC")
                     is not None else np.nan for n in obs_img])
    # track classes from the stations and visits of each point
    order = np.argsort(pid, kind="stable")
    cls = np.empty(len(pid), dtype=object)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and pid[order[j]] == pid[order[i]]:
            j += 1
        idx = order[i:j]
        nst = len(set(station[idx]))
        nvis = len(set(visit[idx]))
        c = "within station" if nst == 1 else ("across stations, one visit" if nvis == 1 else "across visits")
        cls[idx] = c
        i = j
    res: Dict[str, Any] = {"observations": int(len(R)), "all_px": _rms_axis(R), "by_class": {}, "by_eye": {},
                           "by_visit": {}, "by_temperature": []}
    for c in ("within station", "across stations, one visit", "across visits"):
        m = cls == c
        res["by_class"][c] = {"observations": int(m.sum()), "eps_px": _rms_axis(R[m]),
                              "points": int(len(set(pid[m]))) if m.any() else 0}
    for e in ("NL", "NR"):
        m = eye == e
        res["by_eye"][e] = {"observations": int(m.sum()), "eps_px": _rms_axis(R[m])}
    for k, v in enumerate(visits):
        m = visit == k
        res["by_visit"][v["name"]] = {"observations": int(m.sum()), "eps_px": _rms_axis(R[m]),
                                      "cross_station_eps_px": _rms_axis(R[m & (cls != "within station")])}
    if np.isfinite(temp).any():
        lo = np.floor(np.nanmin(temp) / T_BIN_DEG) * T_BIN_DEG
        hi = np.ceil(np.nanmax(temp) / T_BIN_DEG) * T_BIN_DEG
        for a in np.arange(lo, hi, T_BIN_DEG):
            for e in ("NL", "NR"):
                m = (temp >= a) & (temp < a + T_BIN_DEG) & (eye == e)
                if m.sum() >= 200:
                    res["by_temperature"].append({"bin_degC": [float(a), float(a + T_BIN_DEG)], "eye": e,
                                                  "observations": int(m.sum()), "eps_px": _rms_axis(R[m])})
    out["residuals"] = res
    # registration per visit (poses.csv: refined minus prior camera centre, m)
    reg: Dict[str, Any] = {}
    pcsv = pdir / "error_input" / "poses.csv"
    pose_rows = list(csv.DictReader(open(pcsv, encoding="utf-8"))) if pcsv.is_file() else []
    for k, v in enumerate(visits):
        rr = [p for p in pose_rows if _visit_of(int(p["sol"]), visits) == k and int(p.get("observations") or 0) >= 30]
        if not rr:
            continue
        d = np.array([[float(p["dE_m"]), float(p["dN_m"]), float(p["dU_m"])] for p in rr])
        reg[v["name"]] = {"images": len(rr), "median_dE_m": float(np.median(d[:, 0])), "median_dN_m": float(np.median(d[:, 1])),
                          "median_dU_m": float(np.median(d[:, 2])), "median_dC_m": float(np.median(np.linalg.norm(d, axis=1))),
                          "p90_dC_m": float(np.percentile(np.linalg.norm(d, axis=1), 90))}
    unloc = ((st.get("reconstruction") or {}).get("unlocalized") or {})
    stations = []
    by_st: Dict[str, list] = {}
    for p in pose_rows:
        by_st.setdefault(p["station"], []).append(p)
    for sname, g in sorted(by_st.items(), key=lambda kv: min(int(p["sol"]) for p in kv[1])):
        d3 = np.array([[float(p["dE_m"]), float(p["dN_m"]), float(p["dU_m"])] for p in g])
        stations.append({"station": sname, "sols": [min(int(p["sol"]) for p in g), max(int(p["sol"]) for p in g)],
                         "visit": visits[_visit_of(int(g[0]["sol"]), visits)]["name"] if visits else None,
                         "images": len(g), "dE_m": float(np.median(d3[:, 0])), "dN_m": float(np.median(d3[:, 1])),
                         "dU_m": float(np.median(d3[:, 2])), "dC_m": float(np.linalg.norm(np.median(d3, 0))),
                         "position_prior": (unloc.get(sname) or {}).get("prior", "kept")})
    names = [k for k in reg]
    if len(names) >= 2:
        a, b = reg[names[0]], reg[names[1]]
        reg["between_first_two_visits_m"] = float(np.linalg.norm([a["median_dE_m"] - b["median_dE_m"],
                                                                  a["median_dN_m"] - b["median_dN_m"],
                                                                  a["median_dU_m"] - b["median_dU_m"]]))
    # v0p80: by rover site (the localisation frame: a site change re-zeroes the rover frame - a whole site's priors
    # can be off together; only stations that kept their prior)
    by_site: Dict[str, Any] = {}
    for r in stations:
        if r["position_prior"] == "dropped":
            continue
        by_site.setdefault(r["station"][:4], []).append(r)
    reg["by_rover_site"] = {k: {"stations": len(v), "images": int(sum(x["images"] for x in v)),
                                "dE_m": float(np.average([x["dE_m"] for x in v], weights=[x["images"] for x in v])),
                                "dN_m": float(np.average([x["dN_m"] for x in v], weights=[x["images"] for x in v])),
                                "dU_m": float(np.average([x["dU_m"] for x in v], weights=[x["images"] for x in v]))}
                            for k, v in by_site.items()}
    out["registration"] = reg
    out["stations"] = stations
    out["pose_rows"] = [{"name": p["name"], "sol": int(p["sol"]), "station": p["station"], "E_m": float(p["E_m"]),
                         "N_m": float(p["N_m"]), "dE_m": float(p["dE_m"]), "dN_m": float(p["dN_m"]),
                         "dU_m": float(p["dU_m"])} for p in pose_rows]
    out["per_image"] = {n: {"eps_px": v[0], "observations": v[1], "sol": meta.get(n, {}).get("sol"),
                            "T_degC": meta.get(n, {}).get("camera_temperature_degC"),
                            "eye": str(meta.get(n, {}).get("base_instrument") or meta.get(n, {}).get("instrument", ""))[:2]}
                        for n, v in per_image.items()}
    rs = st.get("reconstruction") or {}
    th = st.get("thermal") or {}
    out["rig"] = {"rig_fit": rs.get("rig_fit"), "epochs_table": st.get("navcam_rig_epochs"),
                  "thermal_bins": [{k: r.get(k) for k in ("camera", "images", "T_median_degC", "fx", "fy", "cx", "cy", "k1",
                                                          "start", "free")} for r in th.get("rows") or []],
                  "thermal_after_px": (th.get("after") or {}).get("rms_px"), "thermal_before_px": (th.get("before") or {}).get("rms_px")}
    out["settings"] = {k: st.get(k) for k in ("navcam_pp", "navcam_k4", "navcam_distortion_fit", "navcam_rig_yaw",
                                              "navcam_aspect", "navcam_rig_epochs_gap", "navcam_rig_epochs_breaks",
                                              "navcam_rig_epochs_tie")}
    out["settings"].update({"round1_hold": rs.get("round1_hold"), "refine_rig": rs.get("refine_rig"),
                            "navcam_intrinsics": rs.get("navcam_intrinsics"),
                            "excluded_frames": len(rs.get("excluded") or [])})
    done = pdir / "run_done.json"
    if done.is_file():
        d = json.loads(done.read_text(encoding="utf-8"))
        out["run"] = {k: d.get(k) for k in ("mppp", "finished", "verdict", "images", "registered_images",
                                            "residual_rms_native_px", "camera_models")}
    return out


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Markdown lines: run ``a`` against run ``b`` (epsilon per class, per visit, registration per visit)."""
    la, lb = Path(a["project"]).name, Path(b["project"]).name
    out = [f"| | {la} | {lb} |", "|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.3f}"            # noqa: E731
    out.append(f"| ε all (px) | {fmt(a['residuals']['all_px'])} | {fmt(b['residuals']['all_px'])} |")
    for c in a["residuals"]["by_class"]:
        out.append(f"| ε {c} | {fmt(a['residuals']['by_class'][c]['eps_px'])} | "
                   f"{fmt((b['residuals']['by_class'].get(c) or {}).get('eps_px'))} |")
    for v in a["residuals"]["by_visit"]:
        out.append(f"| ε cross-station, {v} | {fmt(a['residuals']['by_visit'][v]['cross_station_eps_px'])} | "
                   f"{fmt((b['residuals']['by_visit'].get(v) or {}).get('cross_station_eps_px'))} |")
    for v, r in a["registration"].items():
        if isinstance(r, dict) and "median_dC_m" in r:
            rb = b["registration"].get(v) or {}
            out.append(f"| registration median dC, {v} (m) | {r['median_dC_m']:.3f} | "
                       f"{fmt(rb.get('median_dC_m'))} |")
    if "between_first_two_visits_m" in a["registration"]:
        out.append(f"| offset between the visits (m) | {a['registration']['between_first_two_visits_m']:.3f} | "
                   f"{fmt(b['registration'].get('between_first_two_visits_m'))} |")
    return out


def report_lines(d: Dict[str, Any]) -> List[str]:
    fmt = lambda v: "-" if v is None else f"{v:.3f}"            # noqa: E731
    out = [f"# {d['site']} — {Path(d['project']).name}", ""]
    if d.get("run"):
        r = d["run"]
        out.append(f"MPPP {r.get('mppp')}, finished {r.get('finished')}, verdict {r.get('verdict')}, "
                   f"{r.get('registered_images')}/{r.get('images')} images, rms {fmt(r.get('residual_rms_native_px'))} px")
        out.append("")
    out += ["## Visits", "", "| visit | sols | images | stations | T min / median / max °C | LMST h |", "|---|---|---|---|---|---|"]
    for v in d["visits"]:
        T = v.get("T_degC") or [np.nan] * 3
        L = v.get("lmst_h") or [np.nan] * 2
        out.append(f"| {v['name']} | {v['sols'][0]}–{v['sols'][1]} | {v.get('images')} | {v.get('stations')} | "
                   f"{T[0]:.0f} / {T[1]:.1f} / {T[2]:.0f} | {L[0]:.1f}–{L[1]:.1f} |")
    res = d["residuals"]
    out += ["", "## Residuals (ε = per-axis rms, native px)", "",
            f"all {fmt(res['all_px'])} px over {res['observations']} observations", "",
            "| tie points | points | observations | ε px |", "|---|---|---|---|"]
    for c, v in res["by_class"].items():
        out.append(f"| {c} | {v['points']} | {v['observations']} | {fmt(v['eps_px'])} |")
    out += ["", "| visit | ε all | ε cross-station |", "|---|---|---|"]
    for v, r in res["by_visit"].items():
        out.append(f"| {v} | {fmt(r['eps_px'])} | {fmt(r['cross_station_eps_px'])} |")
    out += ["", "| T bin °C | eye | observations | ε px |", "|---|---|---|---|"]
    for r in res["by_temperature"]:
        out.append(f"| {r['bin_degC'][0]:.0f}..{r['bin_degC'][1]:.0f} | {r['eye']} | {r['observations']} | {fmt(r['eps_px'])} |")
    out += ["", "## Registration to the waypoints (refined − prior camera centres, images with ≥ 30 observations)", "",
            "| visit | images | median dE | dN | dU | median |d| | p90 |d| |", "|---|---|---|---|---|---|---|"]
    for v, r in d["registration"].items():
        if isinstance(r, dict) and "median_dE_m" in r:
            out.append(f"| {v} | {r['images']} | {r['median_dE_m']:+.3f} | {r['median_dN_m']:+.3f} | {r['median_dU_m']:+.3f} | "
                       f"{r['median_dC_m']:.3f} | {r['p90_dC_m']:.3f} |")
    bs = d["registration"].get("by_rover_site") or {}
    if bs:
        out += ["", "By rover site (localisation frame; stations with a position prior, image-weighted):", "",
                "| site | stations | images | dE | dN | dU |", "|---|---|---|---|---|---|"]
        for k, r in bs.items():
            out.append(f"| {k} | {r['stations']} | {r['images']} | {r['dE_m']:+.3f} | {r['dN_m']:+.3f} | {r['dU_m']:+.3f} |")
    if "between_first_two_visits_m" in d["registration"]:
        out.append(f"\nThe median offsets of the first two visits differ by {d['registration']['between_first_two_visits_m']:.3f} m.")
    if d.get("stations"):
        out += ["", "| station | sols | visit | images | dE | dN | dU | |d| m | position prior |", "|---|---|---|---|---|---|---|---|---|"]
        for r in d["stations"]:
            out.append(f"| {r['station']} | {r['sols'][0]}–{r['sols'][1]} | {r['visit']} | {r['images']} | {r['dE_m']:+.3f} | "
                       f"{r['dN_m']:+.3f} | {r['dU_m']:+.3f} | {r['dC_m']:.3f} | {r['position_prior']} |")
    rf = (d.get("rig") or {}).get("rig_fit") or {}
    if rf.get("epochs"):
        out += ["", "## Rig per epoch (change from the start rig, mdeg)", "", "| camera | pitch | yaw | roll | baseline m |",
                "|---|---|---|---|---|"]
        for e in rf["epochs"]:
            out.append(f"| {e['camera']} | {e['pitch_mdeg']:+.2f} | {e['yaw_mdeg']:+.2f} | {e['roll_mdeg']:+.2f} | {e['baseline_m']:.5f} |")
    bins = (d.get("rig") or {}).get("thermal_bins") or []
    if bins:
        out += ["", "## Temperature bins (change from the start)", "", "| camera | images | T °C | d fx | d fy | d k1 ×1e-6 |",
                "|---|---|---|---|---|---|"]
        for b in bins:
            s0 = b.get("start") or {}
            dk = (1e6 * (b["k1"] - s0["k1"])) if (b.get("k1") is not None and s0.get("k1") is not None) else None
            out.append(f"| {b.get('camera')} | {b.get('images')} | {b.get('T_median_degC') or float('nan'):.1f} | "
                       f"{(b.get('fx') or 0) - (s0.get('fx') or 0):+.2f} | {(b.get('fy') or 0) - (s0.get('fy') or 0):+.2f} | "
                       + ("-" if dk is None else f"{dk:+.1f}") + " |")
    out += ["", "## Settings", "", "    " + json.dumps(d.get("settings"), default=str)]
    return out


def plot(d: Dict[str, Any], others: Sequence[Dict[str, Any]] = (), out_png: Optional[PathLike] = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EYE = {"NL": "C0", "NR": "C3"}
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6))
    pi = d["per_image"]
    for e, col in EYE.items():
        rows = [v for v in pi.values() if v["eye"] == e and np.isfinite(v["eps_px"])]
        axs[0][0].plot([v["sol"] for v in rows], [v["eps_px"] for v in rows], "o", ms=3, color=col, alpha=0.7, label=e)
        rows = [v for v in rows if v["T_degC"] is not None]
        axs[0][1].plot([v["T_degC"] for v in rows], [v["eps_px"] for v in rows], "o", ms=3, color=col, alpha=0.7, label=e)
    lo = min(axs[0][0].get_ylim()[0], axs[0][1].get_ylim()[0])
    hi = max(axs[0][0].get_ylim()[1], axs[0][1].get_ylim()[1])
    for a in axs[0]:
        a.set_ylim(lo, hi)
        a.set_ylabel("per-image ε (native px)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    axs[0][1].set_yticks(axs[0][0].get_yticks())
    axs[0][1].set_ylim(lo, hi)
    axs[0][0].set_xlabel("sol")
    axs[0][0].axvline(300, color="0.4", ls="--", lw=0.8)
    axs[0][1].set_xlabel("Navcam temperature (°C)")
    axs[0][1].axvline(-20, color="0.4", ls="--", lw=0.8)
    axs[0][0].set_title("residual per image against sol (dashed: pitch knot, sol 300)", fontsize=10)
    axs[0][1].set_title("residual per image against camera temperature", fontsize=10)
    ax = axs[1][0]
    visits = d["visits"]
    ARROW = 10.0
    allE, allN = [], []
    for k, v in enumerate(visits):
        rr = [p for p in d["pose_rows"] if _visit_of(p["sol"], visits) == k]
        by: Dict[str, list] = {}
        for p in rr:
            by.setdefault(p["station"], []).append(p)
        if by:
            m = np.array([[np.mean([q[c] for q in g]) for c in ("E_m", "N_m", "dE_m", "dN_m")] for g in by.values()])
            ax.quiver(m[:, 0], m[:, 1], m[:, 2], m[:, 3], color=("C0", "C1", "C2", "C4")[k % 4], angles="xy",
                      scale_units="xy", scale=1.0 / ARROW, width=0.004,
                      label=f"{v['name']} (sols {v['sols'][0]}–{v['sols'][1]})")
            allE += list(m[:, 0]) + list(m[:, 0] + ARROW * m[:, 2])
            allN += list(m[:, 1]) + list(m[:, 1] + ARROW * m[:, 3])
    if allE:
        pad = 2.0
        ax.set_xlim(min(allE) - pad, max(allE) + pad)
        ax.set_ylim(min(allN) - pad, max(allN) + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_title(f"refined − prior camera centre, mean per station (arrows x{ARROW:g})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axs[1][1]
    runs = [d] + list(others)
    classes = list(d["residuals"]["by_class"])
    w = 0.8 / len(runs)
    for k, r in enumerate(runs):
        vals = [(r["residuals"]["by_class"].get(c) or {}).get("eps_px") or 0.0 for c in classes]
        ax.bar(np.arange(len(classes)) + (k - (len(runs) - 1) / 2) * w, vals, w * 0.95,
               color=("C0", "0.6", "C3", "C2")[k % 4], label=Path(r["project"]).name)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylabel("ε (native px)")
    ax.set_title("ε by tie-point class", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{d['site']} — {Path(d['project']).name}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if out_png:
        fig.savefig(out_png, dpi=130)
    return fig


def write_report(project_dir: PathLike, compare_with: Sequence[PathLike] = (), out_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    """:func:`diagnose_block` of ``project_dir`` (and of each ``compare_with`` run), written to ``out_dir`` (default
    ``<project_dir>/diagnose``) as diagnose.json, diagnose.md, diagnose.png.  Returns the paths and the result."""
    d = diagnose_block(project_dir)
    others = []
    for c in compare_with:
        try:
            others.append(diagnose_block(c))
        except Exception as e:                                     # noqa: BLE001
            print(f"diagnose: {c} left out ({type(e).__name__}: {e})", flush=True)
    od = Path(out_dir) if out_dir else Path(project_dir) / "diagnose"
    od.mkdir(parents=True, exist_ok=True)
    lines = report_lines(d)
    for o in others:
        lines += ["", f"## Compared with {o['project']}", ""] + compare(d, o)
    (od / "diagnose.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    slim = {k: v for k, v in d.items() if k not in ("per_image", "pose_rows")}
    slim["compared"] = [{k: v for k, v in o.items() if k not in ("per_image", "pose_rows")} for o in others]
    (od / "diagnose.json").write_text(json.dumps(slim, indent=1, default=float), encoding="utf-8")
    import matplotlib.pyplot as plt
    fig = plot(d, others, od / "diagnose.png")
    plt.close(fig)
    return {"dir": str(od), "md": str(od / "diagnose.md"), "png": str(od / "diagnose.png"), "result": slim}
