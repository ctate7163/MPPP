"""
LightGlue A/B test (0.93.9, Christian 23 Oct 2026): one block aligned twice from the same processed images - COLMAP's
SIFT matcher and SIFT + LightGlue - into ``<WORK>/colmap_sift`` and ``<WORK>/colmap_lightglue``, then the two
compared: the matcher actually used, pairs and inlier matches, registered images, tie points, cross-station points,
residuals, every health check, and the time of the matching and of the whole alignment.

    python scripts/lightglue_ab.py                      (test_lightglue.bat: site chal_rocks_tiny, sols 91-98)
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
from typing import Any, Dict, Optional

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
        out["checks"] = {f"{c.get('check')}|{str(c.get('note', ''))[:24]}": {"value": c.get("value"), "status": c.get("status")}
                         for c in h.get("checks") or [] if isinstance(c, dict)}
    log = Path(run["log"]) if run and run.get("log") else None
    out["matching_seconds"] = matching_seconds(log)
    out["alignment_seconds"] = (run or {}).get("seconds")
    return out


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
             f"| whole alignment (h) | {_f((a.get('alignment_seconds') or 0) / 3600 or None)} | "
             f"{_f((b.get('alignment_seconds') or 0) / 3600 or None)} |",
             f"| registered / images | {_f(ra.get('registered_images'))} / {_f(ra.get('images'))} | "
             f"{_f(rb.get('registered_images'))} / {_f(rb.get('images'))} |",
             f"| residual rms (native px) | {_f(ra.get('residual_rms_native_px'))} | {_f(rb.get('residual_rms_native_px'))} |",
             f"| health | {a.get('verdict')} | {b.get('verdict')} |", "",
             "## Every health check", "", "| check | SIFT | LightGlue |", "|---|---|---|"]
    keys = list(dict.fromkeys(list((a.get("checks") or {})) + list((b.get("checks") or {}))))
    for k in keys:
        ca, cb = (a.get("checks") or {}).get(k) or {}, (b.get("checks") or {}).get(k) or {}
        lines.append(f"| {k.replace('|', ' - ')} | {_f(ca.get('value'))} ({ca.get('status', '-')}) | "
                     f"{_f(cb.get('value'))} ({cb.get('status', '-')}) |")
    lines += ["", "What to look at: cross_station_fraction (more points seen from more than one station), the "
              "registered images, the residuals (LightGlue's extra matches should not raise them) and the time."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="chal_rocks_tiny")
    ap.add_argument("--zcam", action="store_true", default=True, help="the Navcam + Mastcam-Z block (default)")
    ap.add_argument("--navcam-only", dest="zcam", action="store_false")
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES") or "D:\\scapes")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--only", choices=list(VARIANTS), default=None, help="run one variant only")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="more settings for both")
    a = ap.parse_args(argv)
    from mppp.waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    work = work_folder(Path(a.root), a.site, a.zcam, table["sites"][a.site]["sols"])
    res: Dict[str, Any] = {"site": a.site, "work": str(work), "when": f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
                           "gpu": gpu_check(), "runs": {}, "variants": {}}
    print("GPU check:", json.dumps(res["gpu"]), flush=True)
    prev = work / "lightglue_ab.json"
    if prev.is_file():
        res["runs"] = (json.loads(prev.read_text(encoding="utf-8")).get("runs") or {})
    extra = [x for s in a.set for x in ("--set", s)]
    if not a.compare_only:
        for v in ([a.only] if a.only else list(VARIANTS)):
            res["runs"][v] = run_variant(a.site, a.root, a.zcam, v, extra)
    for v in VARIANTS:
        res["variants"][v] = summary(work, v, res["runs"].get(v))
    work.mkdir(parents=True, exist_ok=True)
    (work / "lightglue_ab.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    md = report(work, res)
    (work / "lightglue_ab.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"=> {work / 'lightglue_ab.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
