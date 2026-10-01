"""
Navcam rig translation (v0p53, 1 Oct 2026): is there a first-order offset of the stereo baseline vector from the
CAHV (label) translation, and how well is it determined?

The merged blocks of the v0p52 joint (studies/navcal_v0p52/joint_variants.py, same scapes file) start from the
cameras and rig in use (src/mppp/data/cmods: v0p52 yawc_k4 - k4 = 0, one rig yaw, held), with the v0p40 drift for
pitch and roll (its yaw rate 0), f +38.1 ppm/degC, NL cx +0.0517 px/degC.  Variants:

  rot     : the rig rotation refined (pitch, roll; yaw held), translation held at CAHV - the reference
  trans   : as rot, plus the translation (tx along the baseline, ty, tz) refined with a weak prior on the right
            camera centre (--sigma-m, default 0.05 m, i.e. effectively free at the mm level)

Reported: the change of the right camera centre in the left camera frame (x along the baseline, y down, z forward)
with its formal sd scaled by the variance factor, the cost change, and the baseline length.  The scale of the block
comes from the waypoint priors (1 m per station): the baseline length (x) is tied to it and is the least
determined component.

  python studies/navcal_v0p53/rig_translation.py SCAPES.json OUT_DIR --drift rig_drift_model.json
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scapes"); ap.add_argument("out")
    ap.add_argument("--points", type=int, default=8000)
    ap.add_argument("--drift", default=None)
    ap.add_argument("--sigma-m", type=float, default=0.05)
    ap.add_argument("--ppm", type=float, default=38.101192835265735)
    ap.add_argument("--pp-nl", type=float, default=0.051743758684936365)
    ap.add_argument("--t0", type=float, default=-20.0)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    from scipy.spatial.transform import Rotation
    from mppp.sfm import navcal as NC
    from mppp.paths import cmods_dir
    from navcam_calibration_study import start_state
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.scapes).read_text())
    cams, rig, _ = start_state(cfg, None, "fisheye_t", str(cmods_dir()))
    t = time.time()
    rec0, proj0, idx = NC.merge_scapes((NC.load_scape(n, cfg[n]) for n in cfg), cams, rig, points_per_scape=a.points, seed=0)
    temps = idx["temps"]
    print(f"merged {len(idx['images'])} blocks, {rec0.num_reg_images()} images ({time.time() - t:.0f} s)", flush=True)
    drift = json.loads(Path(a.drift).read_text()) if a.drift else None
    if drift:
        for k in ("yaw_mdeg_per_sol", "yaw_early_mdeg_per_sol"):
            if k in drift:
                drift[k] = 0.0
    proj0.settings["navcam_rig_yaw"] = "hold"
    for key in ("NL", "NR"):                                     # the v0p52 consensus: k4 = 0 held
        c = proj0.cameras[key]
        c["fixed_params"] = sorted(set(c.get("fixed_params") or []) | {"k4"})
        c["free_params"] = [n for n in c.get("free_params") or [] if n != "k4"]
    rf = out / "rig_translation.json"
    res = json.loads(rf.read_text()) if rf.exists() else {}
    _, _, T0 = NC.stereo_rig(rec0)
    R0, t0 = np.asarray(T0.rotation.matrix()), np.asarray(T0.translation)
    C0 = -R0.T @ t0                                             # right centre in the left camera frame
    for name, rr, sig in (("rot", "rotation", None), ("trans", True, a.sigma_m)):
        if a.only and name not in a.only:
            continue
        t = time.time()
        rec, proj = copy.deepcopy(rec0), copy.deepcopy(proj0)
        kw = {"pp_slopes": {1: (a.pp_nl, 0.0)}}
        if drift:
            kw["drift"] = drift
        if sig:
            kw["rig_translation_sigma_m"] = float(sig)
        ba = NC.joint_adjust(rec, proj, temps, a.ppm, a.t0, max_iterations=a.iterations, covariance=(name == "trans"),
                             refine_rig=rr, rig_slopes=(0.0, 0.0), **kw)
        _, _, T = NC.stereo_rig(rec)
        R1, t1 = np.asarray(T.rotation.matrix()), np.asarray(T.translation)
        C1 = -R1.T @ t1
        cov = ba.get("covariance") or {}
        vf = float(cov.get("variance_factor", 1.0))
        rig_cov = [np.asarray(v) for k, v in (cov.get("blocks") or {}).items() if k.startswith("rig:")]
        sd_t = None
        if rig_cov and name == "trans":
            c = rig_cov[0]
            sd_t = (np.sqrt(np.clip(np.diag(c)[-3:], 0, None) * vf)).tolist()
        r = {"variant": name, "final_cost": float(ba["final_cost"]), "iterations": ba["iterations"],
             "variance_factor": vf, "observations": ba.get("observations"),
             "t_start_m": t0.tolist(), "t_m": t1.tolist(), "dt_mm": (1e3 * (t1 - t0)).tolist(),
             "centre_start_m": C0.tolist(), "centre_m": C1.tolist(), "dcentre_mm": (1e3 * (C1 - C0)).tolist(),
             "sd_t_mm": None if sd_t is None else [1e3 * x for x in sd_t],
             "baseline_start_m": float(np.linalg.norm(t0)), "baseline_m": float(np.linalg.norm(t1)),
             "rig_abs_mdeg": NC.rig_angles(R1), "rig_cov_shape": [list(np.shape(x)) for x in rig_cov],
             "seconds": time.time() - t, "sigma_m": sig}
        res[name] = r
        rf.write_text(json.dumps(res, indent=1, default=float))
        print(f"{name:6s} cost {r['final_cost']:.1f} it {r['iterations']} dcentre [mm] "
              f"{np.round(r['dcentre_mm'], 3).tolist()} sd_t [mm] {None if sd_t is None else np.round(r['sd_t_mm'], 3).tolist()} "
              f"baseline {r['baseline_m']:.5f} m ({1e3 * (r['baseline_m'] - r['baseline_start_m']):+.2f} mm) "
              f"{r['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
