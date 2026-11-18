"""Navcam tied joints side by side (0.99.24).

Christian, 18 Nov 2026: "maybe m20 ncam's fy-fx is trading against k1 in temp. If they are degenerate, then i'd rather
let fy and fx fit separately while keeping k1 vs temp zero. can you test this?"  The test is four joints over the same
blocks, points and seed - the focal slope one per eye or one per axis (``--aniso-f``), the k1 slope free or zero
(``--zero S_NL_k1 S_NR_k1``) - and this table of them::

    python -m rpp_studies.calibration.compare_tied NAME=FOLDER [NAME=FOLDER ...] [--out FILE.md]

Per joint: the final cost, the cost against the first joint, the residuals' median and rms, and per eye the slopes
S_fx, S_fy (ppm/degC: the change of fx, fy per degC of the detector, over the focal length) and S_k1 (1/degC: the
change of the first radial term k1), each with its standard error.  Read with it: a term is worth its place when the
cost falls by much more than the number of parameters it adds (one unit of cost is about one unit of chi-square over
two, with the residuals in units of their errors), and two terms trade when either one alone gets most of the fall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EYES = ("NL", "NR")


def _find(d: Any, test) -> Optional[Dict[str, Any]]:
    """The first dict (breadth first) in ``d`` for which ``test`` holds."""
    queue = [d]
    while queue:
        x = queue.pop(0)
        if isinstance(x, dict):
            if test(x):
                return x
            queue.extend(x.values())
        elif isinstance(x, list):
            queue.extend(x[:50])
    return None


def joint_file(folder: Path) -> Optional[Path]:
    """``tied_joint.json`` of a joint's folder: in the folder or (as ``rpp calib navcam`` writes it) in its
    ``navcam_joint`` subfolder (0.99.27: the first table found none)."""
    for f in (Path(folder) / "tied_joint.json", Path(folder) / "navcam_joint" / "tied_joint.json"):
        if f.is_file():
            return f
    return None


def read(folder: Path) -> Dict[str, Any]:
    """{"cost", "median_px", "rms_px", "NL": {"fx", "fy", "k1": (value, sd)}, "NR": ...} of a joint's folder."""
    res = json.loads(joint_file(Path(folder)).read_text(encoding="utf-8"))
    out: Dict[str, Any] = {"cost": res.get("final_cost")}
    if out["cost"] is None:
        c = _find(res, lambda x: "final_cost" in x)
        out["cost"] = None if c is None else c["final_cost"]
    s = _find(res, lambda x: "median_px" in x and "rms_px" in x) or {}
    out["median_px"], out["rms_px"] = s.get("median_px"), s.get("rms_px")
    law = _find(res, lambda x: all(isinstance(x.get(e), dict) and "fx_ppm_per_degC" in x[e] for e in EYES)) or {}
    for e in EYES:
        o = law.get(e) or {}
        sd = o.get("sd") or {}
        out[e] = {k: (o.get(n), sd.get(n)) for k, n in (("fx", "fx_ppm_per_degC"), ("fy", "fy_ppm_per_degC"),
                                                        ("k1", "k1_per_degC"))}
    return out


def _pm(v, digits: int, scale: float = 1.0) -> str:
    val, sd = v
    if val is None:
        return "-"
    s = f"{scale * val:+.{digits}f}"
    return s + (f" ± {scale * sd:.{digits}f}" if sd is not None and sd == sd else "")


def table(rows: Dict[str, Dict[str, Any]]) -> str:
    names = list(rows)
    c0 = rows[names[0]]["cost"]
    L = ["| joint | cost | vs the first | median px | rms px | eye | S_fx ppm/°C | S_fy ppm/°C | S_k1 1e-5/°C |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = rows[n]
        for i, e in enumerate(EYES):
            head = ([n, f"{r['cost']:.1f}" if r["cost"] is not None else "-",
                     f"{r['cost'] - c0:+.1f}" if None not in (r["cost"], c0) else "-",
                     f"{r['median_px']:.4f}" if r["median_px"] is not None else "-",
                     f"{r['rms_px']:.4f}" if r["rms_px"] is not None else "-"] if i == 0 else [""] * 5)
            L.append("| " + " | ".join(head + [e, _pm(r[e]["fx"], 1), _pm(r[e]["fy"], 1), _pm(r[e]["k1"], 2, 1e5)]) + " |")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("joints", nargs="+", metavar="NAME=FOLDER")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows: Dict[str, Dict[str, Any]] = {}
    for j in a.joints:
        name, _, folder = j.rpartition("=")                       # a name may hold "=" ("k1(T)=0")
        if joint_file(Path(folder)) is None:
            print(f"{name}: no tied_joint.json in {folder} (left out)")
            continue
        rows[name] = read(Path(folder))
    if not rows:
        return 1
    txt = "Navcam tied joints side by side\n\n" + table(rows) + "\n"
    print(txt)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
