"""
Navcam joint calibration variants (v0p52, 1 Oct 2026): rig yaw fixed at 0 and k4 / p1 = 0, on the blocks of the
v0p41 joint that are still on disk (colmap_old, 16 blocks; Marble Mountain from colmap, MPPP 0.44).

All variants: fisheye + tangential, f +38.1 ppm/degC about T0 = -20 degC, the v0p40 rig drift per image, NL cx
+0.0517 px/degC (the v0p41 form), 8000 points per block, start from the cameras in use (src/mppp/data/cmods).

  ref        : the v0p41 form (rig yaw, pitch, roll refined; drift yaw rate as fitted)
  yaw0       : rig yaw = 0 and held (the drift's yaw rate 0); NL/NR cx, cy absorb the stereo offset
  yaw0_k4    : yaw0 + k4 = 0 (held), the other lens terms free
  yaw0_k4p1  : yaw0 + k4 = p1 = 0

  python studies/navcal_v0p52/joint_variants.py SCAPES.json OUT_DIR [--points 8000] [--only ref yaw0 ...]
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

VARIANTS = {"ref": dict(yaw="refine", zero=()), "yaw0": dict(yaw="zero", zero=()),
            "yaw0_k4": dict(yaw="zero", zero=("k4",)), "yaw0_k4p1": dict(yaw="zero", zero=("k4", "p1"))}
RAD_BINS = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)


def radial_stats(rec, proj, ks, kmap):
    from mppp.sfm import navcal as NC
    by = {int(r["image_id"]): float(r.get("downsample_scale", 1.0)) for r in proj.images}
    blk = {}
    res, rad = [], []
    cache = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            iid = el.image_id
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            if iid not in cache:
                k = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
                if iid in ks:
                    c0 = np.asarray(cam.params[2:4]); k = c0 + (k - c0) / ks[iid]
                if kmap is not None and len(k):
                    k = kmap(iid, k, cam)
                cache[iid] = k
            p = im.project_point(pt.xyz)
            if p is None:
                continue
            xy = cache[iid][el.point2D_idx]
            res.append(np.linalg.norm(np.asarray(p) - xy) * by.get(iid, 1.0))
            rad.append(np.linalg.norm(xy - np.array([cam.width, cam.height]) / 2) / (np.hypot(cam.width, cam.height) / 2))
    res, rad = np.asarray(res), np.asarray(rad)
    bins = []
    for a, b in zip(RAD_BINS[:-1], RAD_BINS[1:]):
        m = (rad >= a) & (rad < b)
        bins.append({"r": [a, b], "n": int(m.sum()), "median_px": float(np.median(res[m])) if m.any() else None,
                     "rms_px": float(np.sqrt(np.mean(res[m] ** 2))) if m.any() else None})
    return {"observations": int(res.size), "median_px": float(np.median(res)), "rms_px": float(np.sqrt(np.mean(res ** 2))),
            "p95_px": float(np.percentile(res, 95)), "radial": bins}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scapes"); ap.add_argument("out")
    ap.add_argument("--points", type=int, default=8000)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--drift", default=None)
    ap.add_argument("--ppm", type=float, default=38.101192835265735)
    ap.add_argument("--pp-nl", type=float, default=0.051743758684936365)
    ap.add_argument("--t0", type=float, default=-20.0)
    ap.add_argument("--iterations", type=int, default=100)
    a = ap.parse_args(argv)
    import pycolmap
    from mppp.sfm import navcal as NC
    from mppp.sfm.project import navcam_distortion_terms, rig_without_yaw, PARAM_NAMES
    from mppp.paths import cmods_dir
    from navcam_calibration_study import start_state
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.scapes).read_text())
    cams, rig, _ = start_state(cfg, None, "fisheye_t", str(cmods_dir()))
    t = time.time()
    rec0, proj0, idx = NC.merge_scapes((NC.load_scape(n, cfg[n]) for n in cfg), cams, rig, points_per_scape=a.points, seed=0)
    temps = idx["temps"]
    print(f"merged {len(idx['images'])} blocks, {rec0.num_reg_images()} images, {rec0.num_points3D()} points "
          f"({time.time() - t:.0f} s)", flush=True)
    drift0 = json.loads(Path(a.drift).read_text()) if a.drift else None
    names = PARAM_NAMES["THIN_PRISM_FISHEYE"]
    results = {}
    rf = out / "variants.json"
    if rf.exists():
        results = json.loads(rf.read_text())
    for name, v in VARIANTS.items():
        if a.only and name not in a.only:
            continue
        t = time.time()
        rec = copy.deepcopy(rec0)
        proj = copy.deepcopy(proj0)
        drift = copy.deepcopy(drift0)
        proj.settings["navcam_rig_yaw"] = v["yaw"]
        if v["yaw"] == "zero":
            if drift:
                for k in ("yaw_mdeg_per_sol", "yaw_early_mdeg_per_sol"):
                    if k in drift:
                        drift[k] = 0.0
            _, _, T = NC.stereo_rig(rec)
            R0 = rig_without_yaw(np.asarray(T.rotation.matrix()))
            rec.rigs[1].set_sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=2),
                                            pycolmap.Rigid3d(pycolmap.Rotation3d(R0), np.asarray(T.translation)))
        for cid, key in ((1, "NL"), (2, "NR")):
            c = dict(proj.cameras[key], model=rec.cameras[cid].model.name, params=list(map(float, rec.cameras[cid].params)))
            c = navcam_distortion_terms(c, "refine", k4="zero" if "k4" in v["zero"] else "consensus",
                                        p1="zero" if "p1" in v["zero"] else "consensus")
            proj.cameras[key] = c
            rec.cameras[cid].params = np.asarray(c["params"], float)
        xkw = {"pp_slopes": {1: (a.pp_nl, 0.0)}}
        if drift:
            xkw["drift"] = drift
        ba = NC.joint_adjust(rec, proj, temps, a.ppm, a.t0, max_iterations=a.iterations, covariance=True,
                             rig_slopes=(0.0, 0.0), **xkw)
        ks = NC.keypoint_scales(proj, temps, a.ppm, a.t0)
        kmap = NC.thermal_keypoint_map(proj, temps, a.t0, (0.0, 0.0), xkw["pp_slopes"], drift)
        st = radial_stats(rec, proj, ks, kmap)
        _, _, T = NC.stereo_rig(rec)
        so = NC.stereo_offset(rec.cameras[1], rec.cameras[2], T)
        cov = ba.get("covariance") or {}
        vf = float(cov.get("variance_factor", 1.0))
        res = {"variant": name, "settings": v, "final_cost": float(ba["final_cost"]), "iterations": ba["iterations"],
               "observations": ba.get("observations"), "variance_factor": vf, "brief": ba["brief"],
               "NL": dict(zip(names, map(float, rec.cameras[1].params))),
               "NR": dict(zip(names, map(float, rec.cameras[2].params))),
               "rig_abs_mdeg": NC.rig_angles(np.asarray(T.rotation.matrix())),
               "rig_R": np.asarray(T.rotation.matrix()).tolist(), "rig_t": np.asarray(T.translation).tolist(),
               "stereo": so, "residuals": st, "seconds": time.time() - t,
               "blocks": {k: len(v2) for k, v2 in idx["images"].items()}}
        results[name] = res
        rf.write_text(json.dumps(results, indent=1, default=float))
        print(f"{name:10s} cost {res['final_cost']:.1f}  it {res['iterations']}  median {st['median_px']:.4f}  rms {st['rms_px']:.4f}"
              f"  r>0.9 median {st['radial'][-2]['median_px']:.3f}/{st['radial'][-1]['median_px']:.3f}  "
              f"rig yaw {res['rig_abs_mdeg']['yaw_mdeg']:+.2f}  disparity_inf {so['disparity_inf_px']:+.3f}  {res['seconds']:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
