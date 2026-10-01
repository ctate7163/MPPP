"""
Mastcam-Z start models from the label CAHVOR models (MPPP v0p53): for each zoom (34, 48, 63, 79, 110 mm) and eye, a
Metashape frame calibration (``ZL048_frame.xml`` ...) and a provisional focus model
(``M2020_ZCAM048_focus_model.json``) estimated from the label camera models of the processed images.

The label models (``intrinsics`` in ``processed/mppp_manifest_v*.json``: CAHVOR converted to OpenCV) are
evaluated per image; per eye and zoom the script takes
  - the median distortion and principal point (``distortion``, ``pp0_px`` at the reference focus),
  - a line f = f0 + slope (focus - reference) through the label focal lengths (Huber; reference = the median focus),
  - f0 multiplied by ``--backlash-ratio`` (default 1.0: the label; at 34 mm the backlash state sits 0.9 % above it,
    the shipped 34 mm model was refined from aligned blocks - do not replace it with a label model).
These are the start for the 48, 63 and 110 mm blocks until notebook 04 §5b makes a refined consensus per zoom
from the aligned blocks.  Notebook 03 does not need the XMLs (``ZCAM_INTRINSICS = "focus_model"`` starts each eye
from the labels of its block); they are for Metashape and for ``ZCAM_INTRINSICS = "xml"``.

    python scripts\\zcam_start_models.py D:\\scapes\\colmap\\mars2020_sol_0361_sid_chal_rocks_colmap_zcam ^
        D:\\scapes\\colmap\\mars2020_sol_1150_overlook_mountain_colmap_zcam --zooms 48 63 110 ^
        --out D:\\scapes\\colmap\\camera_analysis\\zcam_start_models

Then check them and copy them into the models in use with ``scripts\\promote_cmods.py <out> --note "..."``
(focus models) - the XMLs are copied by hand into ``src\\mppp\\data\\cmods`` if wanted.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")


def label_rows(folders):
    """One row per Mastcam-Z image of the newest manifest of each WORK folder (or processed folder)."""
    from mppp.colmap import colmap_camera_params
    from mppp.sfm.workdir import load_manifest
    rows = []
    for d in map(Path, folders):
        proc = d / "processed" if (d / "processed").is_dir() else d
        try:
            man = load_manifest(proc)[0]
        except Exception as e:                                       # noqa: BLE001
            print(f"  {d}: no manifest ({e})")
            continue
        for m in man.get("images", []):
            fn = m.get("filename") or {}
            if "failed" in m or fn.get("family") != "Z" or not m.get("intrinsics"):
                continue
            s = float(fn.get("downsample_scale") or 1.0)
            model, p = colmap_camera_params(m["intrinsics"])
            p = np.asarray(p, float).copy()
            p[:4] /= s
            full = np.zeros(12)
            full[:len(p)] = p
            fc = m.get("focus_position_count")
            rows.append({"block": d.name, "group": fn["camera_group"], "zoom": fn.get("zoom_mm"), "eye": fn.get("eye"),
                         "focus": float(fc) if fc is not None else None, "params": full,
                         "size": (int(round(m["intrinsics"]["width"] / s)), int(round(m["intrinsics"]["height"] / s)))})
    return rows


def _huber_line(x, y, ref, k=1.345, it=20):
    X = np.c_[np.ones_like(x), x - ref]
    w = np.ones_like(y)
    for _ in range(it):
        b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        r = y - X @ b
        sig = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), 1e-9)
        u = np.abs(r) / (k * sig)
        w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
    return float(b[0]), float(b[1]), float(sig)


def eye_model(rows, backlash_ratio=1.0):
    P = np.array([r["params"] for r in rows])
    med = np.median(P, axis=0)
    W, H = rows[0]["size"]
    foc = np.array([r["focus"] for r in rows if r["focus"] is not None], float)
    f = np.array([np.sqrt(r["params"][0] * r["params"][1]) for r in rows if r["focus"] is not None], float)
    ref = float(np.round(np.median(foc))) if len(foc) else 0.0
    if len(foc) >= 3 and np.ptp(foc) > 50:
        f0, slope, rms = _huber_line(foc, f, ref)
    else:
        f0, slope, rms = float(np.median(f)) if len(f) else float(np.sqrt(med[0] * med[1])), 0.0, float("nan")
    return {"width": W, "height": H, "median_params": med, "f0_label_px": f0, "slope_px_per_count": slope,
            "reference_focus": ref, "label_rms_px": rms, "n_images": len(rows),
            "focus_range": [float(foc.min()), float(foc.max())] if len(foc) else None,
            "aspect": float(med[1] / med[0]), "backlash_ratio": float(backlash_ratio),
            "blocks": sorted({r["block"] for r in rows})}


def metashape_xml(e) -> str:
    p = e["median_params"]
    W, H = e["width"], e["height"]
    f = float(p[1])
    # FULL_OPENCV -> Metashape: f = fy, b1 = fx - fy, cx/cy from the centre, P1 = p2, P2 = p1 (see
    # mppp.sfm.project.camera_from_metashape_xml)
    vals = {"f": f, "cx": float(p[2] - W / 2), "cy": float(p[3] - H / 2), "b1": float(p[0] - p[1]),
            "k1": float(p[4]), "k2": float(p[5]), "k3": float(p[8]), "p1": float(p[7]), "p2": float(p[6])}
    body = "".join(f"  <{k}>{v!r}</{k}>\n" for k, v in vals.items() if k in ("f", "cx", "cy") or abs(v) > 0)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<calibration>\n  <projection>frame</projection>\n'
            f"  <width>{W}</width>\n  <height>{H}</height>\n{body}"
            f"  <date>{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}</date>\n</calibration>\n")


def focus_model(zoom: int, eyes) -> dict:
    cams = {}
    for g, e in sorted(eyes.items()):
        p = e["median_params"]
        cams[g] = {"f0_px": e["f0_label_px"] * e["backlash_ratio"], "reference_focus": e["reference_focus"],
                   "slope_px_per_count": e["slope_px_per_count"], "aspect": e["aspect"],
                   "fit_rms_px": e["label_rms_px"], "focus_range": e["focus_range"],
                   "label_f0_px": e["f0_label_px"], "label_slope_px_per_count": e["slope_px_per_count"],
                   "distortion": {"model": "FULL_OPENCV", "names": list(NAMES), "params": [float(x) for x in p[4:12]]},
                   "pp0_px": [float(p[2]), float(p[3])],
                   "pp": {"cx_px_per_count": 0.0, "cy_px_per_count": 0.0},
                   "n_images": e["n_images"], "blocks": e["blocks"],
                   "provisional": "label model (CAHVOR) - not refined"}
    return {"cameras": cams, "state": "single" if int(zoom) == 110 else "backlash", "pixel_mm": 0.0074,
            "units": "see mppp.sfm.project.zcam_model_focal / zcam_model_pp_shift",
            "source": (f"MPPP scripts/zcam_start_models.py {datetime.date.today().isoformat()}: label CAHVOR models of "
                       f"{sum(e['n_images'] for e in eyes.values())} images, blocks "
                       f"{sorted({b for e in eyes.values() for b in e['blocks']})}; f0 x backlash ratio "
                       f"{next(iter(eyes.values()))['backlash_ratio']:g}; provisional start model")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="WORK folders (or processed/ folders) with Mastcam-Z images")
    ap.add_argument("--zooms", nargs="*", type=int, default=[48, 63, 79, 110], help="zooms in mm (default 48 63 79 110)")
    ap.add_argument("--out", required=True, help="output folder (XMLs and focus models)")
    ap.add_argument("--backlash-ratio", type=float, default=1.0,
                    help="f0 = label f0 x this (default 1.0; 34 mm backlash state: about 1.009)")
    a = ap.parse_args(argv)
    rows = label_rows(a.folders)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for z in a.zooms:
        zr = [r for r in rows if r["zoom"] == z]
        if not zr:
            print(f"{z} mm: no images in these folders")
            continue
        eyes = {}
        for g in sorted({r["group"] for r in zr}):
            e = eye_model([r for r in zr if r["group"] == g], a.backlash_ratio if z != 110 else 1.0)
            eyes[g] = e
            (out / f"{g}_frame.xml").write_text(metashape_xml(e), encoding="utf-8")
            p = e["median_params"]
            print(f"{g}: {e['n_images']} images, f {e['f0_label_px']:.1f} px at focus {e['reference_focus']:.0f} "
                  f"(+{e['slope_px_per_count']:.4f} px/count), cx {p[2]:.1f} cy {p[3]:.1f}, k1 {p[4]:+.4f} k2 {p[5]:+.4f}"
                  f" -> {out / (g + '_frame.xml')}")
        fm = focus_model(z, eyes)
        f = out / f"M2020_ZCAM{z:03d}_focus_model.json"
        f.write_text(json.dumps(fm, indent=1, default=float), encoding="utf-8")
        report[z] = {g: {k: v for k, v in e.items() if k != "median_params"} | {"params": e["median_params"].tolist()}
                     for g, e in eyes.items()}
        print(f"{z} mm: provisional focus model {f}")
    (out / "zcam_start_models.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return 0 if report else 1


if __name__ == "__main__":
    sys.exit(main())
