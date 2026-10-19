"""
v0p85: what happens to the Mastcam-Z right-eye (ZR) frames, step by step, in the stage 2 of a block run with
``MPPP_ZCAM_TRACE`` set (``zr_diagnose_z34.bat``).  Reads ``<project>/zr_trace.jsonl`` of each block and writes
one Markdown report.

    python scripts\\zr_report.py D:\\scapes\\mars2020_sol_0178_rochette_zcam\\colmap_zrdiag ^
        D:\\scapes\\mars2020_sol_1408_south_arm_zcam\\colmap_zrdiag --out D:\\scapes\\mppp_pipeline\\zr_diagnosis.md
"""
from __future__ import annotations

import argparse
import json
import statistics as stx
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def read_trace(path: Path) -> List[Dict[str, Any]]:
    """[{"head": {...}, "rows": [...]}] in order."""
    steps: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "head" in d:
            steps.append({"head": d["head"], "rows": []})
        elif steps:
            steps[-1]["rows"].append(d)
    return steps


def _med(v):
    v = [x for x in v if x is not None]
    return f"{stx.median(v):.4g}" if v else "-"


def _rot_deg(q0, q1) -> float:
    from scipy.spatial.transform import Rotation
    return float(np.degrees((Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()).magnitude()))


def compare(a: Dict[str, Any], b: Dict[str, Any], eye: str) -> Dict[str, Any]:
    """What changed for one eye's images between two steps."""
    A = {r["name"]: r for r in a["rows"] if r["group"].startswith(eye)}
    B = {r["name"]: r for r in b["rows"] if r["group"].startswith(eye)}
    common = sorted(set(A) & set(B))
    out = {"images": len(common),
           "registered": (sum(bool(A[n].get("registered")) for n in common), sum(bool(B[n].get("registered")) for n in common)),
           "with_obs>=20": (sum((A[n].get("obs") or 0) >= 20 for n in common), sum((B[n].get("obs") or 0) >= 20 for n in common)),
           "obs": (sum(A[n].get("obs") or 0 for n in common), sum(B[n].get("obs") or 0 for n in common)),
           "keypoints_zero": sum((B[n].get("keypoints") or 0) == 0 for n in common),
           "frame_changed": sum(A[n].get("frame_id") != B[n].get("frame_id") for n in common),
           "rig_changed": sum(A[n].get("rig_id") != B[n].get("rig_id") for n in common),
           "camera_changed": sum(A[n].get("camera_id") != B[n].get("camera_id") for n in common),
           "frame_sensors_now": sorted({B[n].get("frame_sensors") for n in common if B[n].get("frame_sensors") is not None})}
    dC, dR, dcam = [], [], []
    for n in common:
        x, y = A[n], B[n]
        if x.get("C") and y.get("C"):
            dC.append(float(np.linalg.norm(np.subtract(y["C"], x["C"]))))
        if x.get("q") and y.get("q"):
            dR.append(_rot_deg(x["q"], y["q"]))
        if x.get("cam") and y.get("cam"):
            dcam.append(np.abs(np.subtract(y["cam"], x["cam"])))
    out["centre_change_m_median_max"] = (_med(dC), f"{max(dC):.4g}" if dC else "-")
    out["rotation_change_deg_median_max"] = (_med(dR), f"{max(dR):.4g}" if dR else "-")
    out["camera_change_max_fx_fy_cx_cy_k1_k2"] = (np.max(dcam, axis=0).round(4).tolist() if dcam else [])
    out["rel_minus_rig_deg_median_now"] = _med([B[n].get("rel_minus_rig_deg") for n in common])
    out["baseline_m_median_now"] = _med([B[n].get("baseline_m") for n in common])
    return out


def block_report(proj: Path) -> List[str]:
    p = proj / "zr_trace.jsonl"
    if not p.is_file():
        return [f"## {proj}", "", f"no {p.name} (was the run started by zr_diagnose_z34.bat?)", ""]
    steps = read_trace(p)
    starts = [i for i, s in enumerate(steps) if s["head"]["step"] == "stage2_start"]
    if starts:
        steps = steps[starts[-1]:]                      # the last run only (the trace is appended to)
    L = [f"## {proj.parent.name}", "", f"`{p}`: {len(steps)} steps", "",
         "| step | points | frames with >= 20 observations / registered, per eye and zoom |", "|---|---|---|"]
    for s in steps:
        cov = s["head"].get("coverage") or {}
        L.append(f"| {s['head']['step']} | {s['head'].get('points')} | "
                 + ", ".join(f"{g} {c['with_points']}/{c['frames']}" for g, c in sorted(cov.items())) + " |")
    L.append("")
    # the step where an eye loses more than half of its frames with tie points
    for eye in ("ZR", "ZL"):
        for i in range(1, len(steps)):
            a, b = steps[i - 1], steps[i]
            ca = sum(c["with_points"] for g, c in (a["head"].get("coverage") or {}).items() if g.startswith(eye))
            cb = sum(c["with_points"] for g, c in (b["head"].get("coverage") or {}).items() if g.startswith(eye))
            if ca >= 4 and cb < 0.5 * ca:
                L += [f"**{eye} lost its frames at `{b['head']['step']}`** (after `{a['head']['step']}`: {ca} -> {cb}).", ""]
                for e2 in ("ZR", "ZL"):
                    c = compare(a, b, e2)
                    L.append(f"- {e2}: " + "; ".join(f"{k} {v}" for k, v in c.items()))
                extra = {k: v for k, v in b["head"].items() if k not in ("coverage", "points", "step", "time")}
                L += [f"- step info: `{json.dumps(extra, default=str)[:600]}`", ""]
                # the same comparison one step earlier, for contrast
                if i >= 2:
                    c = compare(steps[i - 2], a, eye)
                    L += [f"- for contrast, `{steps[i - 2]['head']['step']}` -> `{a['head']['step']}` ({eye}): "
                          + "; ".join(f"{k} {v}" for k, v in c.items()), ""]
                break
    # the guard's own diagnosis lines from the run log
    logs = sorted((proj.parent / "runs").glob("*/log.txt"), key=lambda q: q.stat().st_mtime)
    if logs:
        hits = [ln.split("| ", 1)[-1] for ln in logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()
                if "lost" in ln and ("Mastcam-Z" in ln or "tie points" in ln) and "[sfm]" in ln]
        if hits:
            L += ["Guard messages in the run log:", ""] + [f"    {h[:900]}" for h in hits[:12]] + [""]
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("projects", nargs="+", help="<WORK>/colmap_<variant> folders")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    L = ["# Mastcam-Z right eye in stage 2 (zr_diagnose_z34)", ""]
    for p in a.projects:
        L += block_report(Path(p))
    text = "\n".join(L) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
