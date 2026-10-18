"""
Rational FULL_OPENCV with k5 free (k6 held) against the Navcam form in use (MPPP 0.82.1, 18 Oct 2026; Christian:
"try again freeing k5 and holding k6").

Each block (a WORK folder with ``colmap/project.json`` and ``colmap/sparse/cahv_ba[_single]``) is loaded as solved
(:func:`mppp.sfm.navcal.load_scape`), merged on its own into one NL and one NR camera with the consensus rig and
thermal model of the cameras in use (fx, fy per axis and NR cx against temperature; no k1(T), drift or early-mission
terms in any variant, so that the lens families are compared on the same footing), and adjusted with:

  fisheye_t    THIN_PRISM_FISHEYE, k1-k4 + p1, p2 (sx1, sy1 held) - the v0p80 form, 10 parameters per eye
  rational_k4  FULL_OPENCV k1-k3 / (1 + k4 r^2), p1, p2 (k5 = k6 = 0) - 10 per eye
  rational_k5  + k5 r^4 in the denominator (k6 = 0 held) - 11 per eye
  rational_k6  + k5, k6 (reference) - 12 per eye

The rational start cameras are least-squares fits to the consensus fisheye cameras (64 x 48 pixel grid,
:func:`navcal.fit_camera_model`).  Reported per block and variant: cost, delta BIC (ln N per added parameter, two
eyes), residual median / rms / corner median, the refined k4-k6, whether the rational radial mapping stays monotonic
to 1.1 x the corner radius, and across blocks the repeatability of each variant's camera (rms difference between two
blocks' cameras over the frame, the best rotation removed: :func:`navcal.compare`).

Usage: python studies/experiments/lens_k5_experiment.py <label>=<WORK folder> [...] [--points 20000] [--out f.json]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

VARIANTS = {"fisheye_t": ("THIN_PRISM_FISHEYE", ["k4"]), "rational_k4": ("FULL_OPENCV", ["k4"]),
            "rational_k5": ("FULL_OPENCV", ["k4", "k5"]), "rational_k6": ("FULL_OPENCV", ["k4", "k5", "k6"])}


def monotonic_reach(params, width=5120, height=3840):
    """How far (in corner radii) the rational radial mapping r_d(r) keeps increasing."""
    fx, fy, cx, cy = params[:4]
    k1, k2, p1, p2, k3, k4, k5, k6 = params[4:12]
    rc = np.max(np.hypot(np.array([0, width]) - cx, np.array([0, height])[:, None] - cy)) / fx
    r = np.linspace(1e-4, 4.0, 40000)
    rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6) / (1 + k4 * r ** 2 + k5 * r ** 4 + k6 * r ** 6)
    inc = np.diff(rd) > 0
    stop = rd[np.argmax(~inc)] if (~inc).any() else rd[-1]
    return float(stop / rc)


def run_block(label, root, cams_in_use, rig, thermal, points):
    import pycolmap
    from mppp.sfm import navcal as NC
    from mppp.sfm.project import navcam_distortion_terms
    sc = NC.load_scape(label, root)
    out, cams_out = {}, {}
    for vname, (model, free) in VARIANTS.items():
        cams, fit_rms = {}, {}
        for k in ("NL", "NR"):
            c = cams_in_use[k]
            if model == "FULL_OPENCV":
                c, fit_rms[k] = NC.fit_camera_model(c, NC.RATIONAL)
            cams[k] = c
        rec, proj, idx = NC.merge_scapes([sc], cams, rig, points_per_scape=points, seed=0)
        for cid, key in ((1, "NL"), (2, "NR")):
            c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
            c = navcam_distortion_terms(c, "refine", k4="consensus")
            c["free_params"] = sorted(set(c.get("free_params") or []) | set(free))
            c["fixed_params"] = [n for n in (c.get("fixed_params") or []) if n not in free]
            proj.cameras[key] = c
        temps = idx["temps"]
        t = time.time()
        ba = NC.joint_adjust(rec, proj, temps, thermal["ppm"], thermal["T0"], max_iterations=150,
                             rig_slopes=(0.0, 0.0), pp_slopes=thermal["pp"], axis_ppm=thermal["axis"])
        ks = NC.keypoint_scales(proj, temps, thermal["ppm"], thermal["T0"], axis_ppm=thermal["axis"])
        kmap = NC.thermal_keypoint_map(proj, temps, thermal["T0"], (0.0, 0.0), thermal["pp"], None)
        st = NC.residual_stats(rec, proj, keypoint_scale=ks, keypoint_map=kmap)
        e = {"model": model, "free_extra": free, "cost": float(ba["final_cost"]), "iterations": ba["iterations"],
             "seconds": round(time.time() - t, 1), "start_fit_rms_px": fit_rms, **st, "cameras": {}}
        for cid, key in ((1, "NL"), (2, "NR")):
            q = np.asarray(rec.cameras[cid].params, float)
            names = NC.PARAMS[model]
            e["cameras"][key] = dict(zip(names, map(float, q)))
            if model == "FULL_OPENCV":
                e["cameras"][key]["monotonic_reach_corner_radii"] = monotonic_reach(q)
            cams_out[(vname, key)] = pycolmap.Camera(model=model, width=5120, height=3840, params=q)
        out[vname] = e
        print(f"{label:16s} {vname:12s} cost {e['cost']:.1f}  median {st['median_px']:.4f}  rms {st['rms_px']:.4f}  "
              f"corner {st['corner_median_px']:.4f}  ({e['seconds']} s)"
              + ("".join(f"  {k} k4 {e['cameras'][k]['k4']:+.4f} k5 {e['cameras'][k]['k5']:+.4f} k6 "
                         f"{e['cameras'][k]['k6']:+.4f} reach {e['cameras'][k]['monotonic_reach_corner_radii']:.2f}"
                         for k in ("NL", "NR")) if model == "FULL_OPENCV" else ""), flush=True)
    n = 2 * out["fisheye_t"]["observations"]
    base = out["rational_k4"]
    npar = {"fisheye_t": 10, "rational_k4": 10, "rational_k5": 11, "rational_k6": 12}
    for v, e in out.items():
        e["delta_cost_vs_rational_k4"] = e["cost"] - base["cost"]
        # cost = chi2 / 2 (whitened): delta BIC = 2 delta cost + delta k ln N, two eyes
        e["delta_bic_vs_rational_k4"] = 2 * e["delta_cost_vs_rational_k4"] + 2 * (npar[v] - 10) * np.log(n)
    return out, cams_out


def main(argv=None):
    import pycolmap
    from mppp.paths import cmods_dir
    from mppp.sfm import navcal as NC
    from mppp.sfm.cmods_form import thermal_terms
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blocks", nargs="+", help="label=WORK folder")
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--out", default="lens_k5_experiment.json")
    a = ap.parse_args(argv)
    d = cmods_dir()
    nl, nr, rj = (json.loads((d / n).read_text()) for n in
                  ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json"))
    cams = {k: pycolmap.Camera(model=j["model"], width=5120, height=3840, params=np.asarray(j["params"], float))
            for k, j in (("NL", nl), ("NR", nr))}
    rig = (np.asarray(rj["R_sensor_from_ref"], float), np.asarray(rj["t_sensor_from_ref_m"], float))
    tl, tr = thermal_terms(nl.get("thermal")), thermal_terms(nr.get("thermal"))
    thermal = {"ppm": float(np.mean([tl["fx_ppm"], tl["fy_ppm"], tr["fx_ppm"], tr["fy_ppm"]])), "T0": tl["T0"],
               "axis": {"NL": (tl["fx_ppm"], tl["fy_ppm"]), "NR": (tr["fx_ppm"], tr["fy_ppm"])},
               "pp": {1: (tl["cx_px"], tl["cy_px"]), 2: (tr["cx_px"], tr["cy_px"])}}
    res, allcams = {}, {}
    for spec in a.blocks:
        label, root = spec.split("=", 1)
        res[label], allcams[label] = run_block(label, root, cams, rig, thermal, a.points)
    labels = list(res)
    rep = {}
    for v in VARIANTS:                     # repeatability: each pair of blocks, camera difference with rotation removed
        rows = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                for k in ("NL", "NR"):
                    c = NC.compare(allcams[labels[i]][(v, k)], allcams[labels[j]][(v, k)])
                    rows.append({"pair": f"{labels[i]}-{labels[j]}", "eye": k, **c})
        rep[v] = rows
    summary = {}
    for v in VARIANTS:
        summary[v] = {"delta_bic": {b: round(res[b][v]["delta_bic_vs_rational_k4"], 1) for b in labels},
                      "corner_median_px": {b: round(res[b][v]["corner_median_px"], 4) for b in labels},
                      "median_px": {b: round(res[b][v]["median_px"], 4) for b in labels}}
        if rep[v]:
            key = next((k for k in ("rms_px", "rms") if k in rep[v][0]), None)
            if key:
                summary[v]["repeatability_rms_px"] = {e: round(float(np.sqrt(np.mean([r[key] ** 2 for r in rep[v]
                                                                                         if r["eye"] == e]))), 3)
                                                      for e in ("NL", "NR")}
    Path(a.out).write_text(json.dumps({"blocks": res, "repeatability": rep, "summary": summary}, indent=1, default=float))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
