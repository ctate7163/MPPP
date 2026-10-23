"""
LightGlue A/B test (0.93.9, Christian 23 Oct 2026): one block aligned twice from the same processed images - COLMAP's
SIFT matcher and SIFT + LightGlue - into ``<WORK>/colmap_sift`` and ``<WORK>/colmap_lightglue``, then the two
compared: the matcher actually used, pairs and inlier matches, registered images, tie points, cross-station points,
residuals, every health check, and the time of the matching and of the whole alignment.

    python scripts/lightglue_ab.py                      (test_lightglue.bat: Chal Rocks tiny, sols 91-98, and Seitah Overlook tiny, 116-121; then the tiny joint)
    python scripts/lightglue_ab.py --site rockytop --zcam
    python scripts/lightglue_ab.py --compare-only       the report again from the two finished alignments

Writes ``<WORK>/lightglue_ab.md`` and ``lightglue_ab.json``.  First it checks that torch and kornia import and
whether CUDA is available: without them the "lightglue" alignment falls back to COLMAP's SIFT matcher (the report
says which matcher ran).
"""
from __future__ import annotations

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
VARIANTS = {"sift": "exhaustive", "lightglue": "lightglue"}


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
           "--set", f"MATCH_MODE={VARIANTS[variant]}", *extra]
    if zcam:
        cmd.insert(5, "--zcam")
    print(f"\n{datetime.datetime.now():%H:%M:%S}  {variant}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
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
        out["pose_guided"] = {k: v for k, v in (st.get("pose_guided") or {}).items() if not isinstance(v, (list, dict))}
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


def quality_rows(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """The tie-point and alignment-quality rows of the comparison."""
    ca, cb = a.get("checks") or {}, b.get("checks") or {}
    rows = []
    for k, label in QUALITY:
        if k in ca or k in cb:
            rows.append(f"| {label} | {_f((ca.get(k) or {}).get('value'))} | {_f((cb.get(k) or {}).get('value'))} |")
    pa, pb = a.get("pose_guided") or {}, b.get("pose_guided") or {}
    rows.append(f"| pose-guided matches added | {_f(pa.get('new_matches'))} | {_f(pb.get('new_matches'))} |")
    rows.append(f"| observations filtered after it | {_f(pa.get('filtered'))} | {_f(pb.get('filtered'))} |")
    return rows


def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def report(work: Path, res: Dict[str, Any]) -> str:
    a, b = res["variants"]["sift"], res["variants"]["lightglue"]
    ma, mb = a.get("matching") or {}, b.get("matching") or {}
    ra, rb = a.get("run_done") or {}, b.get("run_done") or {}
    lines = [f"# LightGlue A/B: {work.name} ({res['when']})", "",
             f"GPU check: {json.dumps(res.get('gpu'))}", "",
             "| | SIFT (COLMAP) | SIFT + LightGlue |", "|---|---|---|",
             f"| matcher that ran | {ma.get('matcher', ma.get('mode'))} | {mb.get('matcher', mb.get('mode'))} |",
             f"| matched / verified pairs | {_f(ma.get('matched_pairs'))} / {_f(ma.get('verified_pairs'))} | "
             f"{_f(mb.get('matched_pairs'))} / {_f(mb.get('verified_pairs'))} |",
             f"| inlier matches | {_f(ma.get('inlier_matches'))} | {_f(mb.get('inlier_matches'))} |",
             f"| matching time (min) | {_f((a.get('matching_seconds') or 0) / 60 or None)} | "
             f"{_f((b.get('matching_seconds') or 0) / 60 or None)} |",
             f"| matching + adjustment, database to export (min) | "
             f"{_f((a.get('after_features_seconds') or 0) / 60 or None)} | "
             f"{_f((b.get('after_features_seconds') or 0) / 60 or None)} |",
             f"| whole run (h; the second reuses the first's processed images and features) | "
             f"{_f((a.get('alignment_seconds') or 0) / 3600 or None)} | "
             f"{_f((b.get('alignment_seconds') or 0) / 3600 or None)} |",
             f"| registered / images | {_f(ra.get('registered_images'))} / {_f(ra.get('images'))} | "
             f"{_f(rb.get('registered_images'))} / {_f(rb.get('images'))} |",
             f"| residual rms (native px) | {_f(ra.get('residual_rms_native_px'))} | {_f(rb.get('residual_rms_native_px'))} |",
             f"| health | {a.get('verdict')} | {b.get('verdict')} |", "",
             "## Tie points and alignment quality", "", "| | SIFT | LightGlue |", "|---|---|---|",
             *quality_rows(a, b), "",
             "## Every health check", "", "| check | SIFT | LightGlue |", "|---|---|---|"]
    keys = list(dict.fromkeys(list((a.get("checks") or {})) + list((b.get("checks") or {}))))
    for k in keys:
        ca, cb = (a.get("checks") or {}).get(k) or {}, (b.get("checks") or {}).get(k) or {}
        lines.append(f"| {k.replace('|', ' ')} | {_f(ca.get('value'))} ({ca.get('status', '-')}) | "
                     f"{_f(cb.get('value'))} ({cb.get('status', '-')}) |")
    lines += ["", "What to look at: cross_station_fraction (more points seen from more than one station), the "
              "registered images, the residuals (LightGlue's extra matches should not raise them) and the time."]
    return "\n".join(lines) + "\n"


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
        for v in ([a.only] if a.only else list(VARIANTS)):
            res["runs"][v] = run_variant(site, a.root, a.zcam, v, extra)
    for v in VARIANTS:
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
    ap.add_argument("--only", choices=list(VARIANTS), default=None, help="run one variant only")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="more settings for both")
    ap.add_argument("--no-joint", dest="joint", action="store_false",
                    help="skip the tiny joint (the Hazcam fit and the Mastcam-Z joints over the sites' blocks)")
    ap.add_argument("--no-promote", dest="promote", action="store_false", help="the tiny joint promotes nothing")
    ap.add_argument("--joint-only", action="store_true", help="only the tiny joint (the alignments are done)")
    a = ap.parse_args(argv)
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
        tiny_joint(a, works)
    return 0


if __name__ == "__main__":
    sys.exit(main())
