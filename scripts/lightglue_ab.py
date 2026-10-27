"""
LightGlue A/B test (0.93.9, Christian 23 Oct 2026): one block aligned from the same processed images with each
matcher - COLMAP's SIFT (``colmap_sift``), SIFT + LightGlue on the largest-scale keypoints (``colmap_lightglue``),
LightGlue on keypoints of every octave (``colmap_lightglue_octaves``, 0.94.4) and SIFT and LightGlue together
(``colmap_sift_lightglue``, 0.94.4) - a variant already finished is not run again (``--rerun``), then compared: the matcher actually used, pairs and inlier matches, registered images, tie points, cross-station points,
residuals, every health check, and the time of the matching and of the whole alignment.

    python scripts/lightglue_ab.py                      (Chal Rocks tiny, sols 91-98, and Seitah Overlook tiny, 116-121; then the tiny joint)
    python scripts/lightglue_ab.py --site rockytop --zcam
    python scripts/lightglue_ab.py --compare-only       the report again from the two finished alignments

Writes ``<WORK>/lightglue_ab.md`` and ``lightglue_ab.json``.  First it checks that torch and kornia import and
whether CUDA is available: without them the "lightglue" alignment falls back to COLMAP's SIFT matcher (the report
says which matcher ran).
"""
from __future__ import annotations
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 27 Oct: as mppp_env.bat - two OpenMP runtimes (torch, pycolmap) abort a plain "python" start

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# the variants: their settings (0.94.4: LightGlue on keypoints of every octave, and SIFT + LightGlue together)
VARIANTS = {"sift": ["MATCH_MODE=exhaustive"],
            "lightglue": ["MATCH_MODE=lightglue"],
            "lightglue_octaves": ["MATCH_MODE=lightglue", "LIGHTGLUE_KEYPOINTS=octaves"],
            "sift_lightglue": ["MATCH_MODE=sift+lightglue", "LIGHTGLUE_KEYPOINTS=octaves"],
            # 0.94.4: what pose-guided matching and the isolated-point filter each do (SIFT; the run_weekend set)
            "sift_pg_off": ["MATCH_MODE=exhaustive", "POSE_GUIDED=false"],
            "sift_iso_off": ["MATCH_MODE=exhaustive", "ISOLATED_POINTS_FACTOR=0"]}
DEFAULT_VARIANTS = ("sift", "lightglue", "lightglue_octaves", "sift_lightglue")
LABELS = {"sift": "SIFT", "lightglue": "LightGlue", "lightglue_octaves": "LightGlue, every octave",
          "sift_lightglue": "SIFT + LightGlue", "sift_pg_off": "SIFT, no pose-guided", "sift_iso_off": "SIFT, no isolated filter"}


def gpu_check() -> Dict[str, Any]:
    """torch / kornia versions and CUDA, in a child process (a broken CUDA install must not kill this one)."""
    code = ("import json; out={}\n"
            "try:\n import torch; out['torch']=torch.__version__; out['cuda']=bool(torch.cuda.is_available())\n"
            " out['gpu']=torch.cuda.get_device_name(0) if out['cuda'] else None\n"
            "except Exception as e: out['torch_error']=repr(e)\n"
            "try:\n import kornia; out['kornia']=kornia.__version__\n"
            "except Exception as e: out['kornia_error']=repr(e)\n"
            "print(json.dumps(out))")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:                                               # noqa: BLE001
        return {"error": repr(e)}


def _newest_log(work: Path, after: float) -> Optional[Path]:
    logs = [p for p in (work / "runs").glob("*/log.txt") if p.stat().st_mtime >= after - 5]
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def _ts(line: str) -> Optional[datetime.datetime]:
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", line)
    return datetime.datetime.fromisoformat(m.group(1)) if m else None


def matching_seconds(log: Optional[Path]) -> Optional[float]:
    """From the alignment log: the time from the database line to the "matching:" line."""
    if log is None or not log.is_file():
        return None
    t0 = t1 = None
    for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "| database:" in ln or "prior pairs" in ln:
            t0 = t0 or _ts(ln)
        if "| matching:" in ln:
            t1 = _ts(ln)
    return (t1 - t0).total_seconds() if t0 and t1 else None


def after_features_seconds(log: Optional[Path]) -> Optional[float]:
    """From the alignment log: the database line to the export line - matching and adjustment, without the image
    processing and feature extraction that the second variant reuses from the first (0.94.3)."""
    if log is None or not log.is_file():
        return None
    t0 = t1 = None
    for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "| database:" in ln:
            t0 = t0 or _ts(ln)
        if "| export:" in ln:
            t1 = _ts(ln)
    return (t1 - t0).total_seconds() if t0 and t1 else None


def run_variant(site: str, root: str, zcam: bool, variant: str, extra) -> Dict[str, Any]:
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    work = work_folder(Path(root), site, zcam, table["sites"][site]["sols"])
    cmd = [sys.executable, "-m", "mppp", "run", site, "--root", root, "--variant", variant,
           *[x for v in VARIANTS[variant] for x in ("--set", v)], *extra]
    if zcam:
        cmd.insert(5, "--zcam")
    print(f"\n{datetime.datetime.now():%H:%M:%S}  {variant}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):     # 0.94.4: as run_sites (OpenBLAS thread table)
        env.setdefault(k, "8")
    rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
    dt = time.time() - t0
    log = _newest_log(work, t0)
    print(f"{datetime.datetime.now():%H:%M:%S}  {variant}: exit {rc} after {dt / 3600:.2f} h (log {log})", flush=True)
    return {"exit": rc, "seconds": dt, "log": str(log) if log else None, "work": str(work)}


def summary(work: Path, variant: str, run: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    col = work / f"colmap_{variant}"
    out: Dict[str, Any] = {"variant": variant, "folder": str(col)}
    rd = col / "run_done.json"
    if rd.is_file():
        out["run_done"] = json.loads(rd.read_text(encoding="utf-8"))
    pj = col / "project.json"
    if pj.is_file():
        st = (json.loads(pj.read_text(encoding="utf-8")).get("settings") or {})
        out["matching"] = st.get("matching") or {}
        out["isolated_points"] = st.get("isolated_points") or {}
        pg = st.get("pose_guided") or {}
        out["pose_guided"] = {k: v for k, v in pg.items() if not isinstance(v, (list, dict))}
        out["navcam_zcam_points"] = {"before": (pg.get("before") or {}).get("navcam_zcam_points"),
                                     "after": (pg.get("after") or {}).get("navcam_zcam_points")}
    hj = col / "health" / "health.json"
    if hj.is_file():
        h = json.loads(hj.read_text(encoding="utf-8"))
        out["verdict"] = h.get("verdict")
        out["checks"] = check_table(h.get("checks") or [])
    log = Path(run["log"]) if run and run.get("log") else None
    out["matching_seconds"] = matching_seconds(log)
    out["after_features_seconds"] = after_features_seconds(log)
    out["alignment_seconds"] = (run or {}).get("seconds")
    return out


def check_table(checks) -> Dict[str, Dict[str, Any]]:
    """Health checks keyed by the check name - with the camera or rig (the note) only for a check made once per
    camera - so the two variants line up (0.94.3: the notes hold the numbers, which differ)."""
    cs = [c for c in checks if isinstance(c, dict) and c.get("check")]
    count: Dict[str, int] = {}
    for c in cs:
        count[c["check"]] = count.get(c["check"], 0) + 1
    out: Dict[str, Dict[str, Any]] = {}
    for c in cs:
        note = str(c.get("note") or "")
        key = c["check"] if count[c["check"]] == 1 else f"{c['check']}|{note.split(':')[0].split(' ')[0][:24]}"
        out[key] = {"value": c.get("value"), "status": c.get("status"), "note": note[:120]}
    return out


QUALITY = (("points", "3-D tie points"), ("mean_track_length", "observations per point"),
           ("two_view_fraction", "points seen by only 2 images"), ("obs_per_image_p05", "observations, weakest 5 % of images"),
           ("cross_station_fraction", "points tying different stations"), ("tied_block_fraction", "largest tied block"),
           ("residual_median_px", "residual median (px)"), ("residual_p95_px", "residual p95 (px)"),
           ("outlier_image_fraction", "images with outlying residuals"), ("beyond_ground_fraction", "points beyond the ground"),
           ("zcam_below_terrain_fraction", "Mastcam-Z points below the Navcam terrain"),
           ("zcam_above_terrain_fraction", "Mastcam-Z points above the Navcam terrain"),
           ("station_shift_median_m", "station shift from the prior (m)"), ("attitude_residual_p95_deg", "attitude residual p95 (deg)"))


def quality_rows(vs: List[Dict[str, Any]]) -> List[str]:
    """The tie-point and alignment-quality rows of the comparison, one column per variant."""
    def row(label, vals):
        return f"| {label} | " + " | ".join(_f(v) for v in vals) + " |"
    rows = [row("Navcam-Mastcam-Z tie points", [(v.get("navcam_zcam_points") or {}).get("after") for v in vs]),
            row("... before pose-guided matching", [(v.get("navcam_zcam_points") or {}).get("before") for v in vs])]
    for k, label in QUALITY:
        if any(k in (v.get("checks") or {}) for v in vs):
            rows.append(row(label, [((v.get("checks") or {}).get(k) or {}).get("value") for v in vs]))
    rows.append(row("pose-guided matches added", [(v.get("pose_guided") or {}).get("new_matches") for v in vs]))
    rows.append(row("observations filtered after it", [(v.get("pose_guided") or {}).get("filtered") for v in vs]))
    rows.append(row("isolated points deleted", [(v.get("isolated_points") or {}).get("deleted") for v in vs]))
    return rows


def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def report(work: Path, res: Dict[str, Any]) -> str:
    names = [v for v in VARIANTS if v in res["variants"]]
    vs = [res["variants"][v] for v in names]
    head = "| | " + " | ".join(LABELS.get(v, v) for v in names) + " |"
    sep = "|---|" + "---|" * len(names)

    def row(label, f):
        return f"| {label} | " + " | ".join(f(v) for v in vs) + " |"
    m = lambda v: v.get("matching") or {}                      # noqa: E731
    r = lambda v: v.get("run_done") or {}                      # noqa: E731
    lines = [f"# LightGlue A/B: {work.name} ({res['when']})", "",
             f"GPU check: {json.dumps(res.get('gpu'))}", "", head, sep,
             row("matcher that ran", lambda v: str(m(v).get("matcher", m(v).get("mode", "-")))),
             row("matched / verified pairs", lambda v: f"{_f(m(v).get('matched_pairs'))} / {_f(m(v).get('verified_pairs'))}"),
             row("inlier matches", lambda v: _f(m(v).get("inlier_matches"))),
             row("LightGlue inliers added to SIFT's", lambda v: _f((m(v).get("lightglue") or {}).get("added_inliers"))),
             row("matching time (min)", lambda v: _f((v.get("matching_seconds") or 0) / 60 or None)),
             row("matching + adjustment, database to export (min)",
                 lambda v: _f((v.get("after_features_seconds") or 0) / 60 or None)),
             row("whole run (h; a later variant reuses the processed images and features)",
                 lambda v: _f((v.get("alignment_seconds") or 0) / 3600 or None)),
             row("registered / images", lambda v: f"{_f(r(v).get('registered_images'))} / {_f(r(v).get('images'))}"),
             row("residual rms (native px)", lambda v: _f(r(v).get("residual_rms_native_px"))),
             row("health", lambda v: str(v.get("verdict"))),
             row("MPPP / camera models", lambda v: f"{r(v).get('mppp', '-')} / "
                 f"{(r(v).get('camera_models') or {}).get('fingerprint', '-')}"), "",
             "## Tie points and alignment quality", "", head, sep, *quality_rows(vs), "",
             "## Every health check", "", head, sep]
    keys = list(dict.fromkeys(k for v in vs for k in (v.get("checks") or {})))
    for k in keys:
        lines.append(f"| {k.replace('|', ' ')} | " + " | ".join(
            f"{_f(((v.get('checks') or {}).get(k) or {}).get('value'))} "
            f"({((v.get('checks') or {}).get(k) or {}).get('status', '-')})" for v in vs) + " |")
    lines += ["", "What to look at: the Navcam-Mastcam-Z tie points (what the Mastcam-Z joints fit), the points tying "
              "different stations, the residuals, the Mastcam-Z points off the terrain, and the time."]
    return "\n".join(lines) + "\n"


def best_variant(works: Dict[str, Path], names=None) -> Optional[str]:
    """The finished variant with the most Navcam-Mastcam-Z tie points over the blocks (what the joints fit)."""
    score: Dict[str, int] = {}
    for w in works.values():
        for v in (names or VARIANTS):
            s = summary(w, v)
            if "run_done" in s:
                score[v] = score.get(v, 0) + int((s.get("navcam_zcam_points") or {}).get("after") or 0)
    return max(score, key=score.get) if score else None


TINY_SITES = ("chal_rocks_tiny", "seitah_overlook_tiny")
ZOOMS = (34, 48, 63, 79, 110)


def ab_site(a, site: str) -> Path:
    """The A/B of one site: both alignments (unless --compare-only), the comparison; returns its WORK folder."""
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    work = work_folder(Path(a.root), site, a.zcam, table["sites"][site]["sols"])
    res: Dict[str, Any] = {"site": site, "work": str(work), "when": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
                           "gpu": a.gpu, "runs": {}, "variants": {}}
    prev = work / "lightglue_ab.json"
    if prev.is_file():
        res["runs"] = (json.loads(prev.read_text(encoding="utf-8")).get("runs") or {})
    extra = [x for s in a.set for x in ("--set", s)]
    if not a.compare_only:
        for v in a.variants:
            if (work / f"colmap_{v}" / "run_done.json").is_file() and not a.rerun:
                print(f"{v}: finished already ({work / f'colmap_{v}'}); --rerun runs it again", flush=True)
                continue
            res["runs"][v] = run_variant(site, a.root, a.zcam, v, extra)
    for v in VARIANTS:
        if v in a.variants or (work / f"colmap_{v}" / "run_done.json").is_file():
            res["variants"][v] = summary(work, v, res["runs"].get(v))
    work.mkdir(parents=True, exist_ok=True)
    (work / "lightglue_ab.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = report(work, res)
    (work / "lightglue_ab.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"=> {work / 'lightglue_ab.md'}")
    return work


def tiny_joint(a, works: Dict[str, Path], variant: str = "lightglue") -> Dict[str, Any]:
    """0.94.1 (Christian, 23 Oct): the joints over the tiny blocks (their ``colmap_<variant>`` alignments) - the
    Hazcam fit and the Mastcam-Z joint of every zoom the blocks have - promoted when every check passes, with as few
    blocks as there are (``--min-blocks 1``), but only for models of which none is in use (``--promote-new-only``: a
    tiny joint does not replace a model made from more blocks)."""
    from mppp.waypoints.sites import load_site_table
    table = load_site_table()["sites"]
    day = f"{datetime.date.today():%Y%m%d}"
    out_root = Path(a.root) / "camera_analysis"
    done = {n: w for n, w in works.items() if (w / f"colmap_{variant}" / "run_done.json").is_file()}
    out: Dict[str, Any] = {"blocks": sorted(done), "variant": variant, "exit": {}}
    if not done:
        print(f"tiny joint: no finished colmap_{variant} block", flush=True)
        return out
    promo = ["--promote-if-ok", "--promote-new-only", "--min-blocks", "1"] if a.promote else ["--min-blocks", "1"]
    od = out_root / f"hazcam_fit_tiny_{day}"
    cmd = [sys.executable, "-m", "mppp", "--log", f"{od}_log.txt", "calib", "hazcam", "--variant", variant,
           "--blocks", *[f"{n}={w}" for n, w in done.items()], "--root", a.root, "--out", od, *promo]
    print("\n" + " ".join(map(str, cmd)), flush=True)
    out["exit"]["hazcam"] = subprocess.call([str(c) for c in cmd], cwd=str(ROOT))
    for z in ZOOMS:
        bl = {n: w / f"colmap_{variant}" for n, w in done.items() if z in (table.get(n, {}).get("zcam") or [])}
        if not bl:
            continue
        od = out_root / f"zcam{z}_tiny_joint_{day}"
        cmd = [sys.executable, "-m", "mppp", "--log", f"{od}_log.txt", "calib", "zcam", "--zoom", str(z),
               "--blocks", *[f"{n}={w}" for n, w in bl.items()], "--root", a.root, "--out", od, *promo]
        print("\n" + " ".join(map(str, cmd)), flush=True)
        out["exit"][f"z{z}"] = subprocess.call([str(c) for c in cmd], cwd=str(ROOT))
    (out_root / f"tiny_joint_{day}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs="+", default=list(TINY_SITES), help="the sites (default the two tiny ones)")
    ap.add_argument("--zcam", action="store_true", default=True, help="the Navcam + Mastcam-Z block (default)")
    ap.add_argument("--navcam-only", dest="zcam", action="store_false")
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES") or "D:\\scapes")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(DEFAULT_VARIANTS),
                    help="the variants (default the four matchers: sift, lightglue, lightglue_octaves, "
                         "sift_lightglue; also sift_pg_off, sift_iso_off)")
    ap.add_argument("--only", choices=list(VARIANTS), default=None, help="run one variant only")
    ap.add_argument("--rerun", action="store_true", help="run variants again that have finished")
    ap.add_argument("--joint-variant", default="auto",
                    help="the alignments the tiny joint uses (default auto: the variant with the most "
                         "Navcam-Mastcam-Z tie points over the blocks)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="more settings for both")
    ap.add_argument("--no-joint", dest="joint", action="store_false",
                    help="skip the tiny joint (the Hazcam fit and the Mastcam-Z joints over the sites' blocks)")
    ap.add_argument("--no-promote", dest="promote", action="store_false", help="the tiny joint promotes nothing")
    ap.add_argument("--joint-only", action="store_true", help="only the tiny joint (the alignments are done)")
    a = ap.parse_args(argv)
    if a.only:
        a.variants = [a.only]
    a.gpu = gpu_check()
    print("GPU check:", json.dumps(a.gpu), flush=True)
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    works = {}
    for site in a.site:
        if a.joint_only:
            works[site] = work_folder(Path(a.root), site, a.zcam, table["sites"][site]["sols"])
        else:
            works[site] = ab_site(a, site)
    if a.joint and a.zcam:
        jv = best_variant(works) if a.joint_variant == "auto" else a.joint_variant
        print(f"\ntiny joint over the {jv} alignments" + (" (the most Navcam-Mastcam-Z tie points)"
                                                           if a.joint_variant == "auto" else ""), flush=True)
        if jv:
            tiny_joint(a, works, jv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
