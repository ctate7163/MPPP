"""
Why does fy / fx of the Mastcam-Z bins depend on where the camera points? (1 Nov 2026, Christian: "Could fy be either
a block scale change for distant terrain or maybe an intrinsic camera sag from the optics changing shape with
gravity?")

For every Mastcam-Z bin camera of the finished blocks: its fy / fx (the block's refined model,
``gui_native/sparse``) and, over its images, the boresight elevation, the range of its tie points (median depth along
the optical axis), how much of the frame height the ties cover, and the share of ties in the upper half.  Then the
aspect against elevation and range together:

* gravity sag predicts a dependence on ELEVATION alone, the same at any range;
* a range-dependent (far-field) geometry effect predicts a dependence on RANGE, at any elevation;
* partial coverage with a missing distortion term predicts a dependence on the COVERAGE of the frame.

    python -m mppp_studies.calibration.zcam_aspect_geometry --root D:\\scapes --group zcam_consensus --zooms 34 110
    python -m mppp_studies.calibration.zcam_aspect_geometry --works D:\\scapes\\mars2020_sol_0460_rockytop_zcam --zooms 34

Writes ``<root>/camera_analysis/zcam_aspect_geometry_<date>/``: ``bins.csv`` (one row per bin camera),
``images.csv`` (one row per image) and ``report.md`` (weighted fits per zoom and the horizon / middle / down table
split by range).  Reads only; nothing in the blocks changes.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def image_geometry(rec, im, height: int) -> Optional[Dict[str, float]]:
    """Boresight elevation (deg; world z up), tie-point depth median (m), vertical coverage (5-95 % of the rows, as a
    share of the frame height), share of ties in the upper half, and the number of ties of one registered image."""
    T = im.cam_from_world()
    R = np.asarray(T.rotation.matrix(), float)
    t = np.asarray(T.translation, float)
    b = R.T @ np.array([0.0, 0.0, 1.0])
    el = float(np.degrees(np.arcsin(np.clip(b[2], -1, 1))))
    ys, xyz = [], []
    for p in im.points2D:
        if p.has_point3D():
            ys.append(float(p.xy[1]))
            xyz.append(rec.points3D[p.point3D_id].xyz)
    if len(ys) < 20:
        return None
    X = np.asarray(xyz, float) @ R.T + t
    z = X[:, 2]
    z = z[z > 0]
    ys = np.asarray(ys, float)
    lo, hi = np.percentile(ys, [5, 95])
    return {"elevation_deg": el, "range_m": float(np.median(z)) if z.size else float("nan"),
            "coverage": float((hi - lo) / height), "upper_share": float(np.mean(ys < height / 2.0)),
            "ties": int(len(ys))}


def wfit(y, X, w):
    """Weighted least squares: coefficients and their sd (scaled by the residual variance)."""
    W = np.sqrt(np.asarray(w, float))
    A, b = X * W[:, None], np.asarray(y, float) * W
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ c
    dof = max(1, len(b) - len(c))
    s2 = float(r @ r) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    return c, np.sqrt(np.diag(cov))


def block_rows(work: Path, zooms: Sequence[int], variant: str = "") -> List[Dict[str, Any]]:
    import pycolmap
    proj_dir = work / ("colmap" + (f"_{variant}" if variant else ""))
    pj = json.loads((proj_dir / "project.json").read_text(encoding="utf-8"))
    model = proj_dir / "gui_native" / "sparse"
    rec = pycolmap.Reconstruction(str(model))
    ids = (pj.get("settings", {}).get("database", {}) or {}).get("cameras", {})
    key_of = {int(v): k for k, v in ids.items()}
    rows = {r["name"]: r for r in pj["images"]}
    zset = {f"{int(z):03d}" for z in zooms}
    out = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        key = key_of.get(int(im.camera_id), "")
        if not (key.startswith(("ZL", "ZR")) and key[2:5] in zset):
            continue
        cam = rec.cameras[im.camera_id]
        g = image_geometry(rec, im, int(cam.height))
        if g is None:
            continue
        p = np.asarray(cam.params, float)
        r = rows.get(im.name, {})
        out.append(dict(block=work.name, image=im.name, bin=key, group=key[:5], fx=float(p[0]), fy=float(p[1]),
                        sequence=r.get("sequence"), sol=r.get("sol"), focus=r.get("focus_count"),
                        optics_T=r.get("zcam_optics_temperature_degC"), state=r.get("focus_state"), **g))
    return out


def summarise(images: List[Dict[str, Any]]):
    """Per bin (camera) means, and the aspect in ppm relative to its zoom and eye's weighted median."""
    bins: Dict[str, Dict[str, Any]] = {}
    for r in images:
        k = r["block"] + "|" + r["bin"]
        b = bins.setdefault(k, dict(block=r["block"], bin=r["bin"], group=r["group"], fx=r["fx"], fy=r["fy"],
                                    images=0, ties=0, _el=[], _rg=[], _cv=[], _up=[], _T=[], _w=[]))
        b["images"] += 1
        b["ties"] += r["ties"]
        for k2, v in (("_el", "elevation_deg"), ("_rg", "range_m"), ("_cv", "coverage"), ("_up", "upper_share"),
                      ("_T", "optics_T")):
            b[k2].append(r[v] if r[v] is not None else np.nan)
        b["_w"].append(r["ties"])
    out = []
    for b in bins.values():
        w = np.asarray(b.pop("_w"), float)
        for k2, name in (("_el", "elevation_deg"), ("_rg", "range_m"), ("_cv", "coverage"), ("_up", "upper_share"),
                         ("_T", "optics_T")):
            v = np.asarray(b.pop(k2), float)
            ok = np.isfinite(v)
            b[name] = float(np.average(v[ok], weights=w[ok])) if ok.any() else float("nan")
        b["aspect"] = b["fy"] / b["fx"]
        out.append(b)
    for g in sorted({b["group"] for b in out}):
        bb = [b for b in out if b["group"] == g]
        a = np.array([b["aspect"] for b in bb])
        ref = float(np.median(a))
        for b in bb:
            b["aspect_ppm"] = 1e6 * (b["aspect"] / ref - 1)
    return out


def report(bins: List[Dict[str, Any]]) -> List[str]:
    md = ["# Mastcam-Z bin aspect against pointing, range and frame coverage", "",
          "aspect_ppm = 1e6 (fy/fx / median of the zoom and eye - 1); weights = tie points; bins with >= 200 ties and "
          "|aspect| < 2 % (wild bins left out).", ""]
    for zoom in sorted({b["group"][2:5] for b in bins}):
        bb = [b for b in bins if b["group"][2:5] == zoom and b["ties"] >= 200 and abs(b["aspect_ppm"]) < 20000
              and np.isfinite(b["range_m"]) and b["range_m"] > 0]
        if len(bb) < 8:
            continue
        y = np.array([b["aspect_ppm"] for b in bb])
        w = np.array([b["ties"] for b in bb], float)
        el = np.array([b["elevation_deg"] for b in bb])
        lr = np.log10(np.array([b["range_m"] for b in bb]))
        cv = np.array([b["coverage"] for b in bb])
        md += [f"## {zoom} mm ({len(bb)} bins)", ""]
        for name, X in (("elevation only", np.c_[np.ones_like(el), el / 10]),
                        ("log10 range only", np.c_[np.ones_like(el), lr]),
                        ("coverage only", np.c_[np.ones_like(el), cv]),
                        ("elevation + log10 range + coverage", np.c_[np.ones_like(el), el / 10, lr, cv])):
            c, s = wfit(y, X, w)
            terms = ["const"] + {"elevation only": ["per 10 deg elevation"], "log10 range only": ["per decade of range"],
                                 "coverage only": ["per unit coverage"]}.get(
                name, ["per 10 deg elevation", "per decade of range", "per unit coverage"])
            md.append(f"- {name}: " + ", ".join(f"{t} {v:+.0f} +- {e:.0f} ppm" for t, v, e in zip(terms, c, s)))
        md += ["", "| pointing | near (< 10 m) | 10-30 m | far (> 30 m) |", "|---|---|---|---|"]
        for lab, sel in (("horizon (el > -12 deg)", el > -12), ("middle", (el <= -12) & (el >= -35)),
                         ("down (el < -35 deg)", el < -35)):
            cells = []
            for rs in ((lr < 1), (lr >= 1) & (lr < np.log10(30)), (lr >= np.log10(30))):
                m = sel & rs
                cells.append(f"{np.average(y[m], weights=w[m]):+.0f} ({m.sum()})" if m.any() else "-")
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
        md.append("")
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="the scapes root (with --group)")
    ap.add_argument("--group", default=None, help="a site group of sites.json, e.g. zcam_consensus")
    ap.add_argument("--works", nargs="*", default=None, help="WORK folders (the _zcam blocks) instead")
    ap.add_argument("--zooms", nargs="*", type=int, default=[34, 110])
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    works: List[Path] = [Path(w) for w in (a.works or [])]
    if a.group:
        from mppp.waypoints.sites import load_site_table, work_folder
        t = load_site_table()
        for s in t["groups"][a.group]:
            works.append(work_folder(a.root, s, True, t["sites"][s]["sols"]))
    if not works:
        ap.error("give --works or --root and --group")
    root = Path(a.root) if a.root else works[0].parent
    out = Path(a.out) if a.out else root / "camera_analysis" / f"zcam_aspect_geometry_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    images: List[Dict[str, Any]] = []
    for w in works:
        try:
            r = block_rows(w, a.zooms, a.variant)
            print(f"{w.name}: {len(r)} Mastcam-Z images", flush=True)
            images += r
        except Exception as e:                                        # noqa: BLE001
            print(f"{w.name}: skipped ({type(e).__name__}: {e})", flush=True)
    if not images:
        print("no Mastcam-Z images of these zooms", file=sys.stderr)
        return 1
    bins = summarise(images)
    for name, rows in (("images.csv", images), ("bins.csv", bins)):
        with open(out / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    md = report(bins)
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
