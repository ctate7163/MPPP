"""
Tie-point studies S1-S6 (0.95.0, Christian 28 Oct 2026): which hyper-parameters drive tie-point quality and
completeness, one factor at a time from today's defaults on three small blocks of different character - Chal Rocks
tiny (near field, every zoom), Seitah Overlook tiny (far field), Rockytop example (low sun, weak cross-station) -
then the chosen setting on a full block.  Each variant is one ``mppp run --variant <study>_<variant>`` of a site
(``colmap_<variant>/``; a finished one is not run again), then every finished variant is measured the same way:

* completeness - registered fraction per family, Mastcam-Z frames with >= 10 / 50 anchor ties, anchor ties per
  Mastcam-Z frame (median), cross-station fraction, the track-length distribution, corner coverage;
* quality - residual median / rms per family, epsilon same- vs cross-station, the per-bin focal residuals about the
  focus lines (S6), every health check;
* runtime - matching, the stage-2 rounds, the pose-guided step, the final adjustment, the export (from the log).

S2 (point thinning) also compares each cap's solution with the all-points one: frame centres (cm), attitudes (mdeg)
and per-bin focal lengths (%) - the cap where the poses differ by < 1 cm / 0.1 mdeg and the focals by < 0.05 % is the
setting.

    python scripts/tie_point_studies.py S2                         the three tiny blocks, every S2 variant
    python scripts/tie_point_studies.py S1 S2 --site chal_rocks_tiny
    python scripts/tie_point_studies.py S2 --site chal_rocks --variants all cap1500      (the full block)
    python scripts/tie_point_studies.py S2 --compare-only          the report again from the finished runs
    python scripts/tie_point_studies.py S6 --site chal_rocks_tiny  (an analysis of the finished default runs: no alignment)
    python scripts/tie_point_studies.py --list

Writes ``<WORK>/studies/<study>.md`` and ``.json`` per site, and ``<root>/studies/<study>_<date>.md`` over the sites.
The variants run one after another (``--jobs`` for several at once through run_sites would oversubscribe the tiny
blocks); set ``MPPP_THREADS`` to share the cores with another run.
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TINY_SITES = ("chal_rocks_tiny", "seitah_overlook_tiny", "rockytop_example")


def _match(**over) -> Dict[str, Any]:
    """The default matching options with ``over`` (a dict setting replaces the whole dict)."""
    from mppp.pipeline import Settings
    return dict(Settings().match, **over)


# the studies: {study: {variant: {setting: value}}}; the first variant of each is today's default (an empty dict)
STUDIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    # S1 the feature budget (two passes by family: Navcam / the rest)
    "S1": {"n64_z32": {}, "n32_z32": {"max_num_features": {"N": 32768, "*": 32768}},
           "n16_z16": {"max_num_features": {"N": 16384, "*": 16384}},
           "n64_z16": {"max_num_features": {"N": 65536, "*": 16384}}},
    # S2 point thinning for the adjustments (every tie kept)
    "S2": {"all": {}, "cap4000": {"ba_points_cap": 4000}, "cap1500": {"ba_points_cap": 1500},
           "cap500": {"ba_points_cap": 500}},
    # S3 the matcher, the ratio test, the verification error
    "S3": {"sift": {}, "sift_lightglue": {"match_mode": "sift+lightglue", "lightglue_keypoints": "octaves"},
           "sift_lightglue16k": {"match_mode": "sift+lightglue", "lightglue_keypoints": "octaves",
                                 "match": _match(max_keypoints=16384)},
           "ratio08": {"match": _match(max_ratio=0.8)}, "err4": {"match": _match(max_error_px=4.0)},
           "err8": {"match": _match(max_error_px=8.0)}},
    # S4 pair selection: the pose-guided step's scope, and prior pairs instead of exhaustive
    "S4": {"pg_weak": {}, "pg_off": {"pose_guided": False}, "pg_all": {"pose_guided": {"scope": "all"}},
           "pg_final": {"pose_guided": {"scope": "final"}}, "prior_pairs": {"match_mode": "prior_pairs"}},
    # S5 the filters: the triangulation angle, the observation filter (the last round's max residual), the
    # isolated-point deletion, the outlier-frame thresholds
    "S5": {"default": {}, "tri1": {"min_tri_angle_deg": 1.0}, "tri0": {"min_tri_angle_deg": 0.1},
           "keep3px": {"schedule": [[24.0, 10.0, 8.0], [12.0, 4.0, 4.0], [8.0, 2.0, 3.0]]},
           "keep1p5px": {"schedule": [[24.0, 10.0, 8.0], [12.0, 4.0, 4.0], [8.0, 2.0, 1.5]]},
           "iso_off": {"isolated_points_factor": 0}, "iso6": {"isolated_points_factor": 6.0},
           "outlier_loose": {"outlier_thresholds": {"residual_factor": 4.0, "min_residual_px": 3.0}}},
    # S6 is an analysis of the finished default runs (the per-bin focal residuals about the lines vs observations)
    "S6": {"default": {}},
}
TITLES = {"S1": "feature budget", "S2": "point thinning for the adjustments", "S3": "matcher and verification",
          "S4": "pair selection and the pose-guided step", "S5": "filters",
          "S6": "per-bin focal residual vs observations (MIN_BIN_OBSERVATIONS)"}
# (step, the substrings a start line holds, the substrings an end line holds, which end line - the first after the
# start or the last): the first start line to that end line
STEP_MARKS = (("matching", ("| database:",), ("| matching:",), "last"),
              ("stage 1", ("| matching:",), ("stage 2:",), "first"),
              ("stage 2 rounds", ("stage 2:",), ("stage 2, pose-guided", "isolated two-view points", "before the final adjustment"), "first"),
              ("final adjustment", ("isolated two-view points", "before the final adjustment"), ("| export:",), "last"),
              ("database to export", ("| database:",), ("| export:",), "last"))


def variant_name(study: str, variant: str) -> str:
    return f"{study.lower()}_{variant}"


def _sets(settings: Dict[str, Any]) -> List[str]:
    return [x for k, v in settings.items() for x in ("--set", f"{k.upper()}={json.dumps(v)}")]


def run_variant(site: str, root: str, zcam: bool, study: str, variant: str, extra: Sequence[str]) -> Dict[str, Any]:
    from lightglue_ab import _newest_log
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    work = work_folder(Path(root), site, zcam, table["sites"][site]["sols"])
    cmd = [sys.executable, "-m", "mppp", "run", site, "--root", root, "--variant", variant_name(study, variant),
           *_sets(STUDIES[study][variant]), *extra]
    if zcam:
        cmd.insert(5, "--zcam")
    print(f"\n{datetime.datetime.now():%H:%M:%S}  {study} {variant}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env.setdefault(k, "8")
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    dt = time.time() - t0
    log = _newest_log(work, t0)
    print(f"{datetime.datetime.now():%H:%M:%S}  {study} {variant}: exit {rc} after {dt / 3600:.2f} h", flush=True)
    return {"exit": rc, "seconds": dt, "log": str(log) if log else None, "work": str(work)}


# ----------------------------------------------------------------------------------------------------- measuring
def _ts(line: str) -> Optional[datetime.datetime]:
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", line)
    return datetime.datetime.fromisoformat(m.group(1)) if m else None


def step_seconds(log: Optional[Path]) -> Dict[str, Optional[float]]:
    """The time of each step from the alignment log: the first line holding a start mark to the last line after it
    holding an end mark (:data:`STEP_MARKS`)."""
    out: Dict[str, Optional[float]] = {k: None for k, _, _, _ in STEP_MARKS}
    if log is None or not Path(log).is_file():
        return out
    lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    for name, start, end, which in STEP_MARKS:
        t0 = t1 = None
        for ln in lines:
            if t0 is None and any(m in ln for m in start):
                t0 = _ts(ln)
                continue
            if t0 is not None and any(m in ln for m in end):
                t1 = _ts(ln)
                if which == "first":
                    break
        out[name] = (t1 - t0).total_seconds() if t0 and t1 else None
    return out


def model_metrics(col: Path) -> Dict[str, Any]:
    """What the saved model says: registered fraction per family, the Mastcam-Z frames' anchor ties, the
    track-length distribution, residuals per family, corner coverage."""
    out: Dict[str, Any] = {}
    try:
        import pycolmap
        from mppp.sfm.project import SfmProject
        from mppp.sfm.reconstruction import native_residuals, zcam_anchor_ties
        proj = SfmProject.load(col)
        path = (proj.settings.get("reconstruction") or {}).get("path") or "sparse/cahv_ba"
        rec = pycolmap.Reconstruction(str(col / path))
    except Exception as e:                                               # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    fam_of = {r["name"]: str(r["instrument"])[:1] for r in proj.images}
    reg = {rec.images[i].name for i in rec.reg_image_ids()}
    fams = sorted(set(fam_of.values()))
    out["registered"] = {f: {"images": sum(1 for n, g in fam_of.items() if g == f),
                             "registered": sum(1 for n, g in fam_of.items() if g == f and n in reg)} for f in fams}
    ties, _ = zcam_anchor_ties(rec)
    if ties:
        v = np.array(sorted(ties.values()), float)
        out["zcam_frames"] = {"with_points": int((v > 0).sum()), "ties_ge_10": int((v >= 10).sum()),
                              "ties_ge_50": int((v >= 50).sum()), "images": len(v),
                              "ties_median": float(np.median(v)), "ties_p25": float(np.percentile(v, 25))}
    lens = np.array([pt.track.length() for pt in rec.points3D.values()], int)
    if lens.size:
        out["tracks"] = {"points": int(lens.size), "mean": float(lens.mean()), "two_view": int((lens == 2).sum()),
                         "three_plus": int((lens >= 3).sum()), "five_plus": int((lens >= 5).sum()),
                         "observations": int(lens.sum())}
    try:
        res = native_residuals(rec, proj)
        r, iid = np.asarray(res["residual_native_px"], float), np.asarray(res["image_id"], int)
        ok = np.isfinite(r)
        name_of = {int(i): rec.images[i].name for i in rec.reg_image_ids()}
        fam_i = np.array([fam_of.get(name_of.get(int(i), ""), "?") for i in iid])
        out["residual_px"] = {f: {"median": float(np.median(r[ok & (fam_i == f)])),
                                  "rms": float(np.sqrt(np.mean(r[ok & (fam_i == f)] ** 2))),
                                  "observations": int((ok & (fam_i == f)).sum())}
                              for f in fams if (ok & (fam_i == f)).any()}
    except Exception as e:                                               # noqa: BLE001
        out["residual_px"] = {"error": f"{type(e).__name__}: {e}"}
    out["frames"] = {int(f): [rec.images[d.id].name for d in rec.frames[f].data_ids] for f in rec.reg_frame_ids()}
    out["centres"] = {rec.images[i].name: np.asarray(rec.images[i].projection_center(), float).round(5).tolist()
                      for i in rec.reg_image_ids()}
    out["attitudes"] = {rec.images[i].name: np.asarray(rec.images[i].cam_from_world().rotation.matrix(), float).tolist()
                        for i in rec.reg_image_ids()}
    db = (proj.settings.get("database") or {}).get("cameras") or {}
    out["cameras"] = {str(k): np.asarray(rec.cameras[int(c)].params, float).tolist() for k, c in db.items()
                      if int(c) in rec.cameras}
    return out


def summary(col: Path, study: str, variant: str, run: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything measured on one alignment folder ``col`` (``colmap_<study>_<variant>/``, or the site's default
    ``colmap/`` for S6)."""
    from lightglue_ab import check_table
    col = Path(col)
    out: Dict[str, Any] = {"study": study, "variant": variant, "folder": str(col), "settings": STUDIES[study][variant]}
    rd = col / "run_done.json"
    if rd.is_file():
        out["run_done"] = json.loads(rd.read_text(encoding="utf-8"))
    pj = col / "project.json"
    if pj.is_file():
        st = (json.loads(pj.read_text(encoding="utf-8")).get("settings") or {})
        out["matching"] = {k: v for k, v in (st.get("matching") or {}).items() if not isinstance(v, dict)}
        for key in ("pose_guided", "pose_guided_weak", "isolated_points"):
            pg = st.get(key) or {}
            out[key] = {k: v for k, v in pg.items() if not isinstance(v, (list, dict))}
            if pg.get("after"):
                out[key]["navcam_zcam_points_after"] = pg["after"].get("navcam_zcam_points")
                out[key]["navcam_zcam_points_before"] = (pg.get("before") or {}).get("navcam_zcam_points")
        rs = st.get("reconstruction") or {}
        out["excluded_frames"] = len(rs.get("excluded") or [])
        out["convergence"] = rs.get("convergence")
        lines = st.get("zcam_focus_lines") or {}
        out["bins"] = [{"group": g, **{k: b.get(k) for k in ("camera", "state", "param", "focus", "value", "residual", "observations")}}
                       for g, r in lines.items() for b in (r.get("bins") or [])]
        out["lines_settled"] = [e["lines_settled"] for e in (rs.get("log") or []) if e.get("lines_settled")]
        out["ba"] = [{"round": e.get("round"), "stage": e.get("stage"), "points": (e.get("tracks") or {}).get("points"),
                      "seconds": e.get("elapsed_s")} for e in (rs.get("log") or []) if "round" in e and "tracks" in e]
        for e in rs.get("log") or []:
            if "final_ba" in e:
                out["final_ba"] = e["final_ba"]
    hj = col / "health" / "health.json"
    if hj.is_file():
        h = json.loads(hj.read_text(encoding="utf-8"))
        out["verdict"] = h.get("verdict")
        out["checks"] = check_table(h.get("checks") or [])
        eps = ((h.get("eps") or {}) if isinstance(h.get("eps"), dict) else {})
        out["eps"] = {k: eps.get(k) for k in ("eps_intra_px", "eps_cross_px")} if eps else None
    if (col / "project.json").is_file():
        out["model"] = model_metrics(col)
    log = Path(run["log"]) if run and run.get("log") else (max((col.parent / "runs").glob("*/log.txt"),
                                                               key=lambda p: p.stat().st_mtime, default=None)
                                                           if (col.parent / "runs").is_dir() else None)
    out["steps_s"] = step_seconds(log)
    out["alignment_seconds"] = (run or {}).get("seconds")
    return out


def compare_solutions(ref: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """S2: the other variant's solution against the reference (all points): frame centres, attitudes, per-bin
    focal lengths of the images and cameras both have."""
    from scipy.spatial.transform import Rotation
    m0, m1 = ref.get("model") or {}, other.get("model") or {}
    out: Dict[str, Any] = {}
    names = sorted(set(m0.get("centres") or {}) & set(m1.get("centres") or {}))
    if names:
        d = np.array([np.subtract(m1["centres"][n], m0["centres"][n]) for n in names], float)
        dist = np.linalg.norm(d, axis=1)
        out["centre_cm"] = {"median": float(100 * np.median(dist)), "p95": float(100 * np.percentile(dist, 95)),
                            "max": float(100 * dist.max()), "images": len(names)}
        ang = []
        for n in names:
            R0, R1 = np.asarray(m0["attitudes"][n]), np.asarray(m1["attitudes"][n])
            ang.append(np.degrees(np.linalg.norm(Rotation.from_matrix(R0.T @ R1).as_rotvec())))
        ang = np.array(ang)
        out["attitude_mdeg"] = {"median": float(1e3 * np.median(ang)), "p95": float(1e3 * np.percentile(ang, 95)),
                                "max": float(1e3 * ang.max())}
    cams = sorted(set(m0.get("cameras") or {}) & set(m1.get("cameras") or {}))
    if cams:
        pct = []
        for k in cams:
            p0, p1 = np.asarray(m0["cameras"][k], float), np.asarray(m1["cameras"][k], float)
            if len(p0) >= 2 and len(p1) >= 2 and p0[0] > 0:
                pct.append(100 * abs(np.sqrt(p1[0] * p1[1]) - np.sqrt(p0[0] * p0[1])) / np.sqrt(p0[0] * p0[1]))
        if pct:
            pct = np.array(pct)
            out["focal_pct"] = {"median": float(np.median(pct)), "max": float(pct.max()), "cameras": len(pct)}
    return out


# ------------------------------------------------------------------------------------------------------ reports
def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def report_site(work: Path, study: str, res: Dict[str, Any]) -> str:
    names = [v for v in STUDIES[study] if v in res["variants"]]
    vs = [res["variants"][v] for v in names]
    head = "| | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)

    def row(label, f):
        return f"| {label} | " + " | ".join(_f(f(v)) for v in vs) + " |"
    r = lambda v: v.get("run_done") or {}                      # noqa: E731
    m = lambda v: v.get("model") or {}                         # noqa: E731
    lines = [f"# {study} ({TITLES.get(study, study)}): {work.name} ({res['when']})", "",
             "Settings per variant: " + "; ".join(f"**{v}** {json.dumps(STUDIES[study][v]) or 'default'}" for v in names),
             "", "## Completeness", "", head, sep,
             row("registered / images", lambda v: f"{_f(r(v).get('registered_images'))} / {_f(r(v).get('images'))}")]
    fams = sorted({f for v in vs for f in (m(v).get("registered") or {})})
    for f in fams:
        lines.append(row(f"  {f} registered", lambda v, f=f: (lambda e: f"{e['registered']} / {e['images']}" if e else None)((m(v).get("registered") or {}).get(f))))
    lines += [row("Mastcam-Z frames with tie points", lambda v: (m(v).get("zcam_frames") or {}).get("with_points")),
              row("Mastcam-Z frames with >= 10 anchor ties", lambda v: (m(v).get("zcam_frames") or {}).get("ties_ge_10")),
              row("Mastcam-Z frames with >= 50 anchor ties", lambda v: (m(v).get("zcam_frames") or {}).get("ties_ge_50")),
              row("anchor ties per Mastcam-Z frame, median", lambda v: (m(v).get("zcam_frames") or {}).get("ties_median")),
              row("Navcam-Mastcam-Z tie points", lambda v: (v.get("pose_guided") or {}).get("navcam_zcam_points_after")
                  or (v.get("pose_guided_weak") or {}).get("navcam_zcam_points_after")),
              row("3-D points", lambda v: (m(v).get("tracks") or {}).get("points")),
              row("observations per point", lambda v: (m(v).get("tracks") or {}).get("mean")),
              row("points seen by 2 / >= 3 / >= 5 images", lambda v: (lambda t: f"{t['two_view']} / {t['three_plus']} / {t['five_plus']}" if t else None)(m(v).get("tracks"))),
              row("cross-station fraction", lambda v: ((v.get("checks") or {}).get("cross_station_fraction") or {}).get("value")),
              row("corner coverage (triangulated ratio)", lambda v: ((v.get("checks") or {}).get("corner_triangulated_ratio") or {}).get("value")),
              row("frames excluded", lambda v: v.get("excluded_frames")),
              "", "## Quality", "", head, sep]
    for f in fams:
        lines.append(row(f"residual {f} median / rms (native px)", lambda v, f=f: (lambda e: f"{e['median']:.3f} / {e['rms']:.3f}" if e and "median" in e else None)((m(v).get("residual_px") or {}).get(f))))
    lines += [row("epsilon same- / cross-station (px)", lambda v: (lambda e: f"{_f(e.get('eps_intra_px'))} / {_f(e.get('eps_cross_px'))}" if e else None)(v.get("eps"))),
              row("points beyond the ground", lambda v: ((v.get("checks") or {}).get("beyond_ground_fraction") or {}).get("value")),
              row("Mastcam-Z points below / above the terrain", lambda v: f"{_f(((v.get('checks') or {}).get('zcam_below_terrain_fraction') or {}).get('value'))} / {_f(((v.get('checks') or {}).get('zcam_above_terrain_fraction') or {}).get('value'))}"),
              row("per-bin focal residual rms about the lines (px)", lambda v: bin_rms(v.get("bins") or [])),
              row("health", lambda v: v.get("verdict")),
              "", "## Runtime", "", head, sep]
    for name, _, _, _ in STEP_MARKS:
        lines.append(row(f"{name} (min)", lambda v, name=name: (lambda s: s / 60 if s else None)((v.get("steps_s") or {}).get(name))))
    lines += [row("whole run (h)", lambda v: (lambda s: s / 3600 if s else None)(v.get("alignment_seconds"))),
              row("points in the final adjustment", lambda v: (v.get("final_ba") or {}).get("points_in_adjustment")
                  if isinstance(v.get("final_ba"), dict) else None),
              row("focus lines settled (round)", lambda v: ", ".join(str(e.get("round")) for e in (v.get("lines_settled") or [])) or "-")]
    if study == "S2" and names and "all" in res["variants"]:
        lines += ["", "## Against the all-points solution", "", head, sep]
        cmp = {v: compare_solutions(res["variants"]["all"], res["variants"][v]) for v in names}
        for label, key, sub in (("frame centres, median / p95 / max (cm)", "centre_cm", ("median", "p95", "max")),
                                ("attitudes, median / p95 / max (mdeg)", "attitude_mdeg", ("median", "p95", "max")),
                                ("per-bin focal length, median / max (%)", "focal_pct", ("median", "max"))):
            lines.append(f"| {label} | " + " | ".join(
                " / ".join(_f((cmp[v].get(key) or {}).get(s)) for s in sub) if cmp[v].get(key) else "-" for v in names) + " |")
        lines += ["", "The cap where the frame centres differ by < 1 cm, the attitudes by < 0.1 mdeg and the focal "
                  "lengths by < 0.05 % from the all-points solution is the setting (Christian, 28 Oct)."]
    if study == "S6":
        lines += ["", "## S6: per-bin focal residual about the lines vs observations", ""] + s6_table(vs)
    lines += ["", "## Every health check", "", head, sep]
    keys = list(dict.fromkeys(k for v in vs for k in (v.get("checks") or {})))
    for k in keys:
        lines.append(f"| {k.replace('|', ' ')} | " + " | ".join(
            f"{_f(((v.get('checks') or {}).get(k) or {}).get('value'))} "
            f"({((v.get('checks') or {}).get(k) or {}).get('status', '-')})" for v in vs) + " |")
    return "\n".join(lines) + "\n"


def bin_rms(bins: List[Dict[str, Any]]) -> Optional[float]:
    r = [b["residual"] for b in bins if b.get("param") in ("fx", "fy") and b.get("residual") is not None]
    return float(np.sqrt(np.mean(np.square(r)))) if r else None


S6_BANDS = ((0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10 ** 9))


def s6_table(vs: List[Dict[str, Any]]) -> List[str]:
    """The fx / fy residuals of the focus bins about their lines, by observation band, over the variants given
    (the default runs of the sites): what MIN_BIN_OBSERVATIONS should be."""
    rows = [b for v in vs for b in (v.get("bins") or []) if b.get("param") in ("fx", "fy") and b.get("residual") is not None]
    if not rows:
        return ["(no focus-line bins in these runs)"]
    out = ["| observations | bins | residual rms (px) | median abs (px) | p90 abs (px) |", "|---|---|---|---|---|"]
    for lo, hi in S6_BANDS:
        sel = [abs(float(b["residual"])) for b in rows if lo <= int(b.get("observations") or 0) < hi]
        if not sel:
            continue
        a = np.array(sel)
        out.append(f"| {lo}-{hi if hi < 10 ** 9 else ''} | {len(a)} | {np.sqrt(np.mean(a ** 2)):.2f} | "
                   f"{np.median(a):.2f} | {np.percentile(a, 90):.2f} |")
    out += ["", "The band where the rms stops falling is the floor for MIN_BIN_OBSERVATIONS (1000 now)."]
    return out


def report_all(root: Path, study: str, sites: Dict[str, Dict[str, Any]]) -> str:
    lines = [f"# {study} ({TITLES.get(study, study)}) over {', '.join(sites)} ({datetime.datetime.now():%Y-%m-%d %H:%M})", ""]
    for site, res in sites.items():
        lines += [f"- {site}: {res.get('report')}"]
    names = [v for v in STUDIES[study] if any(v in r["variants"] for r in sites.values())]
    lines += ["", "| site | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for site, res in sites.items():
        def cell(v):
            s = res["variants"].get(v)
            if not s:
                return "-"
            zf = ((s.get("model") or {}).get("zcam_frames") or {})
            t = (s.get("steps_s") or {}).get("database to export")
            return (f"{zf.get('ties_ge_10', '-')}/{zf.get('images', '-')} Z>=10; "
                    f"{_f((s.get('eps') or {}).get('eps_cross_px')) if s.get('eps') else '-'} px cross; "
                    f"{t / 60:.0f} min" if t else f"{zf.get('ties_ge_10', '-')}/{zf.get('images', '-')} Z>=10")
        lines.append(f"| {site} | " + " | ".join(cell(v) for v in names) + " |")
    if study == "S6":
        lines += ["", "## All sites"] + s6_table([s for r in sites.values() for s in r["variants"].values()])
    return "\n".join(lines) + "\n"


def study_site(a, study: str, site: str) -> Dict[str, Any]:
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    work = work_folder(Path(a.root), site, a.zcam, table["sites"][site]["sols"])
    out_dir = work / "studies"
    res: Dict[str, Any] = {"study": study, "site": site, "work": str(work),
                           "when": f"{datetime.datetime.now():%Y-%m-%d %H:%M}", "runs": {}, "variants": {}}
    prev = out_dir / f"{study}.json"
    if prev.is_file():
        res["runs"] = json.loads(prev.read_text(encoding="utf-8")).get("runs") or {}
    variants = a.variants or list(STUDIES[study])
    extra = [x for s in a.set for x in ("--set", s)]
    if not a.compare_only and study != "S6":
        for v in variants:
            if v not in STUDIES[study]:
                raise SystemExit(f"{study} has no variant {v!r}: {', '.join(STUDIES[study])}")
            if (work / f"colmap_{variant_name(study, v)}" / "run_done.json").is_file() and not a.rerun:
                print(f"{study} {v}: finished already; --rerun runs it again", flush=True)
                continue
            res["runs"][v] = run_variant(site, a.root, a.zcam, study, v, extra)
    for v in STUDIES[study]:
        col = work / f"colmap_{variant_name(study, v)}"
        if study == "S6":                                     # the default alignment of the site (colmap/)
            col = work / "colmap"
        if (col / "run_done.json").is_file() or (col / "project.json").is_file():
            res["variants"][v] = summary(col, study, v, res["runs"].get(v))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{study}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = report_site(work, study, res)
    (out_dir / f"{study}.md").write_text(md, encoding="utf-8")
    res["report"] = str(out_dir / f"{study}.md")
    print("\n" + md)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("studies", nargs="*", help="S1 ... S6")
    ap.add_argument("--site", nargs="+", default=list(TINY_SITES), help="the sites (default the three tiny blocks)")
    ap.add_argument("--zcam", action="store_true", default=True, help="the Navcam + Mastcam-Z block (default)")
    ap.add_argument("--navcam-only", dest="zcam", action="store_false")
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES") or "D:\\scapes")
    ap.add_argument("--variants", nargs="+", default=None, help="the variants of the study to run (default all)")
    ap.add_argument("--compare-only", action="store_true", help="the reports from the finished runs, no alignment")
    ap.add_argument("--rerun", action="store_true", help="run variants again that have finished")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="more settings for every variant")
    ap.add_argument("--list", action="store_true", help="the studies and their variants")
    a = ap.parse_args(argv)
    if a.list or not a.studies:
        for st, vs in STUDIES.items():
            print(f"{st} ({TITLES.get(st, '')}):")
            for v, s in vs.items():
                print(f"   {v:18s} {json.dumps(s) if s else 'today\'s default'}")
        return 0
    for st in a.studies:
        st = st.upper()
        if st not in STUDIES:
            raise SystemExit(f"no study {st!r}: {', '.join(STUDIES)}")
        sites = {site: study_site(a, st, site) for site in a.site}
        out = Path(a.root) / "studies"
        out.mkdir(parents=True, exist_ok=True)
        md = report_all(out, st, sites)
        path = out / f"{st}_{datetime.date.today():%Y%m%d}.md"
        path.write_text(md, encoding="utf-8")
        print("\n" + md + f"\n=> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
