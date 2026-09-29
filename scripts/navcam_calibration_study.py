"""
Navcam calibration study across scapes (v0p35, mppp.sfm.navcal; results in docs/results/v0p35/ and working notes
section 15).

  python scripts/navcam_calibration_study.py rig   OUT_DIR SCAPES.json [--samples label_temps.json] [--only NAME ...]
  python scripts/navcam_calibration_study.py joint OUT_DIR SCAPES.json [--samples ...] [--points 25000]
  python scripts/navcam_calibration_study.py loo   OUT_DIR SCAPES.json [--samples ...] [--lens rational|fisheye_t]
  python scripts/navcam_calibration_study.py all   OUT_DIR SCAPES.json [--samples ...]      (OUT_DIR/rig, OUT_DIR/joint)

SCAPES.json: {"name": "path to a notebook-03 folder (or its colmap/)", ...}.  Temperatures come from the project
records (MPPP >= 0.31) or from the label samples ({stem: {"NL", "NR"}}), interpolated in spacecraft clock.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def common_principal_points(scapes):
    """Median refined principal point of each eye over the blocks."""
    pts = {"NL": [], "NR": []}
    for sc in scapes:
        key_of = {int(v): k for k, v in sc.project.settings["database"]["cameras"].items()}
        for cid, cam in sc.rec.cameras.items():
            k = key_of.get(int(cid))
            if k in pts:
                pts[k].append(np.asarray(cam.params[2:4], float))
    return {k: np.median(np.array(v), axis=0).tolist() for k, v in pts.items()}


def cmd_rig(a, scapes_cfg, samples):
    from mppp.sfm import navcal as NC
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = a.only or list(scapes_cfg)
    # the common principal points come from all blocks (cheap: cameras only)
    import pycolmap
    pp = {"NL": [], "NR": []}
    for n, root in scapes_cfg.items():
        r = Path(root)
        r = r / "colmap" if (r / "colmap").is_dir() else r
        pj = json.loads((r / "project.json").read_text())
        key_of = {int(v): k for k, v in pj["settings"]["database"]["cameras"].items()}
        rec = pycolmap.Reconstruction()
        rec.read(str(r / "sparse" / "cahv_ba"))
        for cid, cam in rec.cameras.items():
            if key_of.get(int(cid)) in pp:
                pp[key_of[int(cid)]].append(np.asarray(cam.params[2:4], float))
    common = {k: np.median(np.array(v), axis=0).tolist() for k, v in pp.items()}
    (out / "common_principal_points.json").write_text(json.dumps(common, indent=1))
    print("common principal points", common, flush=True)
    for n in names:
        f = out / f"rig_{n.replace(' ', '_')}.json"
        if f.exists() and not a.force:
            print(f"{n}: done", flush=True)
            continue
        t = time.time()
        sc = NC.load_scape(n, scapes_cfg[n], samples)
        n_all = len(sc.rec.points3D)
        n_drop = NC.thin_points(sc.rec, a.rig_points, seed=0)
        res = NC.rig_study(sc, common_pp=common)
        res["points_used"] = {"all": n_all, "kept": n_all - n_drop}
        f.write_text(json.dumps(jsonable(res), indent=1))
        print(f"{n}: {time.time() - t:.0f} s", flush=True)
        del sc


def start_state(scapes_cfg, samples, lens="rational"):
    """Start cameras (the shipped consensus of the most recent project) and the start rig."""
    import pycolmap
    from mppp.sfm import navcal as NC
    newest = None
    for n, root in scapes_cfg.items():
        r = Path(root)
        r = r / "colmap" if (r / "colmap").is_dir() else r
        pj = json.loads((r / "project.json").read_text())
        if pj["cameras"].get("NL", {}).get("source", "").endswith("rational.json") and (pj.get("rig") or {}).get("N"):
            newest = pj
    cams = {k: pycolmap.Camera(model=newest["cameras"][k]["model"], width=5120, height=3840,
                               params=np.asarray(newest["cameras"][k]["params"], float)) for k in ("NL", "NR")}
    fits = {}
    if lens != "rational":
        for k in ("NL", "NR"):
            cams[k], fits[k] = NC.to_fisheye_tangential(cams[k])
    rig = (np.asarray(newest["rig"]["N"]["R_sensor_from_ref"], float), np.asarray(newest["rig"]["N"]["t_sensor_from_ref"], float))
    return cams, rig, fits


def build_merged(a, scapes_cfg, samples, lens="rational", exclude=()):
    from mppp.sfm import navcal as NC
    cams, rig, fits = start_state(scapes_cfg, samples, lens)
    gen = (NC.load_scape(n, scapes_cfg[n], samples) for n in scapes_cfg if n not in exclude)
    rec, proj, idx = NC.merge_scapes(gen, cams, rig, points_per_scape=a.points, seed=0)
    return rec, proj, idx, fits


def save_merged(out, tag, rec, proj, idx, extra=None):
    import pickle
    d = out / f"merged_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    rec.write(str(d))
    with open(d / "project_index.pkl", "wb") as f:
        pickle.dump({"project": proj, "index": idx, **(extra or {})}, f)


def load_merged(out, tag):
    import pickle
    import pycolmap
    d = out / f"merged_{tag}"
    rec = pycolmap.Reconstruction(str(d))
    with open(d / "project_index.pkl", "rb") as f:
        o = pickle.load(f)
    return rec, o["project"], o["index"], o


def camera_record(rec, cid, cov=None, vf=1.0):
    from mppp.sfm import navcal as NC
    cam = rec.cameras[cid]
    names = NC.PARAMS[cam.model.name]
    out = {"model": cam.model.name, "width": cam.width, "height": cam.height,
           "params": {n: float(v) for n, v in zip(names, cam.params)}}
    if cov is not None:
        free = [n for n in names if n not in ("k5", "k6", "sx1", "sy1") and not (cam.model.name == "THIN_PRISM_FISHEYE" and False)]
        free = free[:cov.shape[0]]
        out["free_params"] = free
        out["sd"] = {n: float(np.sqrt(cov[i, i] * vf)) for i, n in enumerate(free)}
        out["covariance"] = (np.asarray(cov) * vf).tolist()
    return out


def _cahv_rig(scapes_cfg):
    for n, root in scapes_cfg.items():
        r = Path(root)
        r = r / "colmap" if (r / "colmap").is_dir() else r
        pj = json.loads((r / "project.json").read_text())
        R = ((pj.get("rig") or {}).get("N") or {}).get("R_sensor_from_ref_cahv")
        if R is not None:
            return np.asarray(R, float)
    return None


def cmd_joint(a, scapes_cfg, samples):
    from mppp.sfm import navcal as NC
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lens = a.lens
    t = time.time()
    if (out / f"merged_{lens}_start").exists():
        rec, proj, idx, _ = load_merged(out, f"{lens}_start")
        fits = {}
    else:
        rec, proj, idx, fits = build_merged(a, scapes_cfg, samples, lens)
        save_merged(out, f"{lens}_start", rec, proj, idx)
    temps = idx["temps"]
    Ts = np.array([temps[r["name"]] for r in proj.images if r["name"] in temps])
    T0 = float(np.round(np.mean(Ts), 1))
    print(f"merged {len(idx['images'])} blocks, {rec.num_reg_images()} images, {rec.num_points3D()} points, "
          f"{sum(p.track.length() for p in rec.points3D.values())} observations, T0 {T0} degC  ({time.time() - t:.0f} s)",
          flush=True)
    # 0. the rig's temperature slopes measured inside the blocks (rig study, per-bin rigs), if available
    rig_dir = Path(a.rig_dir) if a.rig_dir else out.parent / "rig"
    within = {}
    studies = [json.loads(f.read_text()) for f in sorted(rig_dir.glob("rig_*.json"))] if rig_dir.is_dir() else []
    for ang in ("yaw", "pitch"):
        w = NC.rig_tests(studies, ang, mode="rotation_pp").get("within") if studies else None
        within[ang] = w
    ky = float(within["yaw"]["slope_mdeg_per_degC"]) if within.get("yaw") and a.rig_thermal else 0.0
    kp = float(within["pitch"]["slope_mdeg_per_degC"]) if within.get("pitch") and a.rig_thermal else 0.0
    print(f"rig temperature slopes from the per-bin rigs: yaw {ky:+.3f}, pitch {kp:+.3f} mdeg/degC", flush=True)
    # 1. converge the shared cameras from the per-block solutions at a provisional slope
    t = time.time()
    b0 = a.warm_slope
    ba = NC.joint_adjust(rec, proj, temps, b0, T0, max_iterations=200, rig_slopes=(ky, kp))
    print(f"warm-up at {b0} ppm/degC: {ba['brief']}  {time.time() - t:.0f} s", flush=True)
    save_merged(out, f"{lens}_warm", rec, proj, idx, {"T0": T0, "slope": b0})
    # 2. profile the slope
    if a.fixed_slope is not None:
        prof = {"rows": [], "best_ppm_per_degC": float(a.fixed_slope), "sd_ppm_per_degC": float("nan"), "fit": []}
    else:
        prof = NC.profile_slope(rec, proj, temps, T0, a.grid, max_iterations=100, rig_slopes=(ky, kp))
    best = prof["best_ppm_per_degC"]
    print(f"slope {best:.1f} +- {prof['sd_ppm_per_degC']:.1f} ppm/degC", flush=True)
    # 2b. profile the rig yaw slope at the best focal slope (0 = a temperature-independent rig)
    yaw_prof = None
    if a.rig_thermal and a.fixed_slope is None:
        import copy as _copy
        rows = []
        for k in sorted({0.0, ky - 1.0, ky - 0.5, ky, ky + 0.5, ky + 1.0}):
            r = _copy.deepcopy(rec)
            t1 = time.time()
            bb = NC.joint_adjust(r, proj, temps, best, T0, max_iterations=100, rig_slopes=(k, kp))
            rows.append({"yaw_mdeg_per_degC": k, "cost": float(bb["final_cost"]), "iterations": bb["iterations"],
                         "variance_factor": float(2 * bb["final_cost"] / max(2 * bb["observations"] - 1, 1))})
            print(f"  rig yaw slope {k:+.2f} mdeg/degC  cost {bb['final_cost']:.2f}  it {bb['iterations']}  {time.time() - t1:.0f} s", flush=True)
            del r
        xs = np.array([r["yaw_mdeg_per_degC"] for r in rows if r["yaw_mdeg_per_degC"] != 0.0 or True])
        cs = np.array([r["cost"] for r in rows])
        near = np.argsort(np.abs(xs - ky))[:5]
        c2 = np.polyfit(xs[near], cs[near], 2)
        kbest = float(-c2[1] / (2 * c2[0])) if c2[0] > 0 else float(xs[np.argmin(cs)])
        vf = float(np.median([r["variance_factor"] for r in rows]))
        yaw_prof = {"rows": rows, "best": kbest, "sd": float(np.sqrt(vf / (2 * c2[0]))) if c2[0] > 0 else float("nan"),
                    "cost_zero_minus_best": float(np.interp(0.0, xs, cs) - np.polyval(c2, kbest)) if 0.0 in xs else None}
        print(f"rig yaw slope {kbest:+.3f} +- {yaw_prof['sd']:.3f} mdeg/degC (within-block {ky:+.3f})", flush=True)
        ky = kbest
    elif a.rig_thermal and a.fixed_rig_yaw is not None:
        ky = float(a.fixed_rig_yaw)
    # 3. final adjustment at the best slope, with covariances
    t = time.time()
    ba = NC.joint_adjust(rec, proj, temps, best, T0, max_iterations=200, covariance=True, rig_slopes=(ky, kp))
    cov = ba.get("covariance") or {}
    vf = float(cov.get("variance_factor", 1.0))
    blocks = cov.get("blocks", {})
    save_merged(out, f"{lens}_final", rec, proj, idx, {"T0": T0, "slope": best, "rig_slopes": (ky, kp)})
    rows = NC._rig_rows(ba, rec, _cahv_rig(scapes_cfg), "rotation")
    res = {"lens": lens, "T0_degC": T0, "ppm_per_degC": best, "sd_ppm_per_degC": prof["sd_ppm_per_degC"],
           "rig_thermal": {"yaw_mdeg_per_degC": ky, "pitch_mdeg_per_degC": kp, "within_block": within,
                           "yaw_profile": yaw_prof},
           "profile": [{k: v for k, v in r.items() if k != "state"} for r in prof["rows"]],
           "profile_states": [r["state"] for r in prof["rows"]], "profile_fit": prof["fit"],
           "variance_factor": vf, "final": {k: ba[k] for k in ("final_cost", "iterations", "observations", "brief")},
           "NL": camera_record(rec, 1, blocks.get("camera:NL"), vf), "NR": camera_record(rec, 2, blocks.get("camera:NR"), vf),
           "rig": rows[0] if rows else None, "rig_R": NC.stereo_rig(rec)[2].rotation.matrix().tolist(),
           "rig_t": np.asarray(NC.stereo_rig(rec)[2].translation).tolist(),
           "blocks": {k: len(v) for k, v in idx["images"].items()},
           "points_per_block": idx["points"], "start_fits": fits, "seconds": time.time() - t}
    (out / f"joint_{lens}.json").write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps(jsonable({k: res[k] for k in ("ppm_per_degC", "sd_ppm_per_degC", "variance_factor")})), flush=True)


def _set_shared(rec, state):
    """Set the shared cameras (1 = NL, 2 = NR) and the stereo rig of a merged reconstruction."""
    import pycolmap
    rec.cameras[1].params = np.asarray(state["NL"], float)
    rec.cameras[2].params = np.asarray(state["NR"], float)
    T = pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(state["rig_R"], float)), np.asarray(state["rig_t"], float))
    rec.rigs[1].set_sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=2), T)


def _shared(rec):
    from mppp.sfm import navcal as NC
    _, _, T = NC.stereo_rig(rec)
    return {"NL": np.array(rec.cameras[1].params), "NR": np.array(rec.cameras[2].params),
            "rig_R": np.asarray(T.rotation.matrix()), "rig_t": np.asarray(T.translation)}


def cmd_loo(a, scapes_cfg, samples):
    """Leave one block out: the joint calibration of the others predicts the held-out block's cameras and rig."""
    from mppp.sfm import navcal as NC
    out = Path(a.out)
    lens = a.lens
    rec0, proj, idx, meta = load_merged(out, f"{lens}_final")
    temps, T0, slope = idx["temps"], meta["T0"], meta["slope"]
    rs = tuple(meta.get("rig_slopes") or (0.0, 0.0))
    kmap = NC.rig_keypoint_map(proj, temps, rs[0], rs[1], T0) if any(rs) else None
    names = [n for n in idx["images"] if not a.only or n in a.only]
    res_file = out / f"loo_{lens}.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}
    full = _shared(rec0)
    for s in names:
        if s in res and not a.force:
            continue
        t = time.time()
        others = [n for n in idx["images"] if n != s]
        # a. the joint calibration without s (warm start from the full joint solution)
        rm = NC.subset(rec0, idx, others)
        ba_m = NC.joint_adjust(rm, proj, temps, slope, T0, max_iterations=100, rig_slopes=rs)
        pred = _shared(rm)
        del rm
        ks = NC.keypoint_scales(proj, temps, slope, T0)
        imgs = idx["images"][s]
        # b. s with the predicted cameras and rig held
        rb = NC.subset(rec0, idx, [s])
        _set_shared(rb, pred)
        ba_b = NC.joint_adjust(rb, proj, temps, slope, T0, max_iterations=100, hold_cameras=("NL", "NR"), refine_rig=False,
                               rig_slopes=rs)
        st_b = NC.residual_stats(rb, proj, ks, imgs, kmap)
        # b0. the same without the rig's temperature dependence
        rb0 = NC.subset(rec0, idx, [s])
        _set_shared(rb0, pred)
        ba_b0 = NC.joint_adjust(rb0, proj, temps, slope, T0, max_iterations=100, hold_cameras=("NL", "NR"), refine_rig=False)
        st_b0 = NC.residual_stats(rb0, proj, ks, imgs)
        del rb0
        # b2. s with the predicted cameras held and its own rig rotation (the lens-model question alone)
        rb2 = NC.subset(rec0, idx, [s])
        _set_shared(rb2, pred)
        ba_b2 = NC.joint_adjust(rb2, proj, temps, slope, T0, max_iterations=100, hold_cameras=("NL", "NR"), rig_slopes=rs)
        st_b2 = NC.residual_stats(rb2, proj, ks, imgs, kmap)
        del rb2
        # c. s with its own cameras and rig rotation
        rc = NC.subset(rec0, idx, [s])
        ba_c = NC.joint_adjust(rc, proj, temps, slope, T0, max_iterations=100, rig_slopes=rs)
        st_c = NC.residual_stats(rc, proj, ks, imgs, kmap)
        own = _shared(rc)
        # d. how far the prediction is from the block's own calibration
        model = rc.cameras[1].model.name
        cmp = {}
        for k in ("NL", "NR"):
            ca = NC.pycolmap_camera(model, pred[k])
            cb = NC.pycolmap_camera(model, own[k])
            cmp[k] = {**NC.compare(cb, ca), "dfx_px": float(pred[k][0] - own[k][0]), "dcx_px": float(pred[k][2] - own[k][2]),
                      "dcy_px": float(pred[k][3] - own[k][3])}
        import pycolmap
        def _stereo(st):
            T = pycolmap.Rigid3d(pycolmap.Rotation3d(st["rig_R"]), st["rig_t"])
            return NC.stereo_offset(NC.pycolmap_camera(model, st["NL"]), NC.pycolmap_camera(model, st["NR"]), T)
        so_p, so_o = _stereo(pred), _stereo(own)
        res[s] = {"held": {"cost": ba_b["final_cost"], **st_b}, "own": {"cost": ba_c["final_cost"], **st_c},
                  "held_cameras": {"cost": ba_b2["final_cost"], **st_b2},
                  "held_no_rig_thermal": {"cost": ba_b0["final_cost"], **st_b0},
                  "cost_increase_no_rig_thermal_pct": 100 * (ba_b0["final_cost"] - ba_c["final_cost"]) / ba_c["final_cost"],
                  "cost_increase_pct": 100 * (ba_b["final_cost"] - ba_c["final_cost"]) / ba_c["final_cost"],
                  "cost_increase_cameras_pct": 100 * (ba_b2["final_cost"] - ba_c["final_cost"]) / ba_c["final_cost"],
                  "camera_difference": cmp,
                  "disparity_inf_diff_px": so_p["disparity_inf_px"] - so_o["disparity_inf_px"],
                  "vparallax_inf_diff_px": so_p["vparallax_inf_px"] - so_o["vparallax_inf_px"],
                  "rig_diff": NC.rig_angles(pred["rig_R"], own["rig_R"]),
                  "loo_joint": {"final_cost": ba_m["final_cost"], "iterations": ba_m["iterations"]},
                  "pred_state": {k: np.asarray(v).tolist() for k, v in pred.items()},
                  "own_state": {k: np.asarray(v).tolist() for k, v in own.items()},
                  "full_state": {k: np.asarray(v).tolist() for k, v in full.items()}, "seconds": time.time() - t}
        res_file.write_text(json.dumps(jsonable(res), indent=1))
        r = res[s]
        print(f"{lens} {s:16s} held cost +{r['cost_increase_pct']:.2f} % (cameras only +{r['cost_increase_cameras_pct']:.2f} %, no rig(T) +{r['cost_increase_no_rig_thermal_pct']:.2f} %)  median {st_b['median_px']:.4f} / own {st_c['median_px']:.4f}  "
              f"corner {st_b['corner_median_px']:.4f} / {st_c['corner_median_px']:.4f}  prediction rms NL {cmp['NL']['rms_px']:.3f} "
              f"NR {cmp['NR']['rms_px']:.3f} px  disparity {r['disparity_inf_diff_px']:+.3f} px  {time.time() - t:.0f} s", flush=True)


def cmd_all(a, scapes_cfg, samples):
    """The three studies in order: rig study, joint calibration (rational, then fisheye + tangential at the same
    thermal slopes), leave one block out for both lens models.  OUT/rig and OUT/joint."""
    import copy as _copy
    out = Path(a.out)
    b = _copy.copy(a)
    b.out = str(out / "rig")
    cmd_rig(b, scapes_cfg, samples)
    b = _copy.copy(a)
    b.out, b.lens, b.rig_dir = str(out / "joint"), "rational", str(out / "rig")
    cmd_joint(b, scapes_cfg, samples)
    j = json.loads((out / "joint" / "joint_rational.json").read_text())
    b.lens, b.fixed_slope, b.fixed_rig_yaw = "fisheye_t", j["ppm_per_degC"], j["rig_thermal"]["yaw_mdeg_per_degC"]
    cmd_joint(b, scapes_cfg, samples)
    for lens in ("rational", "fisheye_t"):
        b.lens = lens
        cmd_loo(b, scapes_cfg, samples)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rig", "joint", "loo", "all"])
    ap.add_argument("out")
    ap.add_argument("scapes")
    ap.add_argument("--samples")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--points", type=int, default=15000, help="joint/loo: points per block")
    ap.add_argument("--rig-points", type=int, default=120000, help="rig: at most this many points per block")
    ap.add_argument("--lens", default="rational")
    ap.add_argument("--warm-slope", type=float, default=40.0)
    ap.add_argument("--grid", type=float, nargs="*", default=[0, 15, 30, 45, 60, 75, 90])
    ap.add_argument("--fixed-slope", type=float, help="joint: skip the profile and use this slope (ppm/degC)")
    ap.add_argument("--fixed-rig-yaw", type=float, help="joint: the rig yaw slope (mdeg/degC) with --fixed-slope")
    ap.add_argument("--rig-dir", help="joint: the rig study folder (default OUT/../rig)")
    ap.add_argument("--no-rig-thermal", dest="rig_thermal", action="store_false",
                    help="joint: no temperature dependence of the rig")
    a = ap.parse_args(argv)
    scapes_cfg = json.loads(Path(a.scapes).read_text())
    samples = json.loads(Path(a.samples).read_text()) if a.samples else None
    if a.only and a.cmd == "joint":
        scapes_cfg = {k: v for k, v in scapes_cfg.items() if k in a.only}
    {"rig": cmd_rig, "joint": cmd_joint, "loo": cmd_loo, "all": cmd_all}[a.cmd](a, scapes_cfg, samples)


if __name__ == "__main__":
    main()
