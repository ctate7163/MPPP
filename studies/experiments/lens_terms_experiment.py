"""
Which radial and tangential terms does the Navcam camera need?  (MPPP 0.22.2)

For each refined Navcam block given on the command line (a project folder with ``project.json`` and
``sparse/cahv_ba``), the final bundle adjustment is repeated from the converged solution with the same
observations under several lens models:

  k4          the default rational model: (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2), p1, p2 refined
  k4k5        + k5 r^4 in the denominator
  k4k5k6      + k5 r^4 + k6 r^6 in the denominator
  k4 p=0      the default without tangential terms (p1 = p2 = 0, held)
  k4k5k6 p=0  both changes

Reported per variant: the robust cost (whitened, so 2 x cost difference ~ chi-square), BIC difference
(ln N per added parameter), residual rms / median overall and in rings of image radius (centre < 0.5,
edge 0.5-0.85, corner > 0.85 of the corner radius), the rms of the mean residual field (median residual
vector in 8 x 6 cells: a lens misfit shows here), the refined k4-k6, p1, p2, how far the camera moved
from the default variant's camera (rms over the frame after the best rotation, and in the corners),
and whether the radial mapping stays monotonic to 1.1 x the corner radius.

Usage:  python studies/experiments/lens_terms_experiment.py <label>=<project folder> [...] [--out lens_terms.json]
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import pycolmap                                                     # noqa: E402
from mppp.cmod import PixelCamera, compare_cameras                  # noqa: E402
from mppp.sfm.project import SfmProject, FULL_OPENCV_NAMES          # noqa: E402
from mppp.sfm.reconstruction import bundle_adjust, _scales          # noqa: E402

VARIANTS = {"k4": (["k4"], True), "k4k5": (["k4", "k5"], True), "k4k5k6": (["k4", "k5", "k6"], True),
            "k4 p=0": (["k4"], False), "k4k5k6 p=0": (["k4", "k5", "k6"], False)}
IP1, IP2 = FULL_OPENCV_NAMES.index("p1"), FULL_OPENCV_NAMES.index("p2")


def observations(rec, project):
    """Per observation: residual vector (native px), keypoint (full-res px), normalised radius, camera key, cell."""
    scale = _scales(rec, project)
    key_of = {int(v): k for k, v in project.settings["database"]["cameras"].items()}
    out = {k: [] for k in ("dx", "dy", "rad", "cam", "cx", "cy")}
    for iid, im in rec.images.items():
        if not im.has_pose:
            continue
        cam = rec.cameras[im.camera_id]
        key = key_of.get(int(im.camera_id))
        if key not in ("NL", "NR"):
            continue
        T = im.cam_from_world()
        R, t = T.rotation.matrix(), np.asarray(T.translation)
        p2 = [p for p in im.points2D if p.has_point3D()]
        if not p2:
            continue
        X = np.array([rec.points3D[p.point3D_id].xyz for p in p2]) @ R.T + t
        kp = np.array([p.xy for p in p2])
        ok = X[:, 2] > 1e-9
        uv = np.full_like(kp, np.nan)
        uv[ok] = cam.img_from_cam(X[ok])
        d = (uv - kp) * scale[iid]
        c = np.array([cam.params[2], cam.params[3]])
        rc = np.max(np.linalg.norm(np.array([[0, 0], [cam.width, 0], [0, cam.height], [cam.width, cam.height]]) - c, axis=1))
        out["dx"].append(d[:, 0]); out["dy"].append(d[:, 1])
        out["rad"].append(np.linalg.norm(kp - c, axis=1) / rc)
        out["cam"].append(np.full(len(kp), key))
        out["cx"].append(np.clip((kp[:, 0] / cam.width * 8).astype(int), 0, 7))
        out["cy"].append(np.clip((kp[:, 1] / cam.height * 6).astype(int), 0, 5))
    o = {k: np.concatenate(v) for k, v in out.items()}
    good = np.isfinite(o["dx"]) & np.isfinite(o["dy"])
    return {k: v[good] for k, v in o.items()}


def residual_stats(o):
    r = np.hypot(o["dx"], o["dy"])
    s = {"n_obs": int(r.size), "rms_px": float(np.sqrt(np.mean(r ** 2))), "median_px": float(np.median(r))}
    for name, lo, hi in (("centre", 0, 0.5), ("edge", 0.5, 0.85), ("corner", 0.85, 2.0)):
        m = (o["rad"] >= lo) & (o["rad"] < hi)
        s[f"{name}_n"] = int(m.sum())
        s[f"{name}_median_px"] = float(np.median(r[m])) if m.any() else float("nan")
        s[f"{name}_rms_px"] = float(np.sqrt(np.mean(r[m] ** 2))) if m.any() else float("nan")
    field = []
    for key in ("NL", "NR"):
        for i in range(8):
            for j in range(6):
                m = (o["cam"] == key) & (o["cx"] == i) & (o["cy"] == j)
                if m.sum() >= 200:
                    field.append((np.median(o["dx"][m]), np.median(o["dy"][m]), (i in (0, 7)) and (j in (0, 5))))
    f = np.array([(a, b) for a, b, _ in field])
    fc = np.array([(a, b) for a, b, corner in field if corner])
    s["residual_field_rms_px"] = float(np.sqrt(np.mean(np.sum(f ** 2, axis=1)))) if len(f) else float("nan")
    s["residual_field_corner_rms_px"] = float(np.sqrt(np.mean(np.sum(fc ** 2, axis=1)))) if len(fc) else float("nan")
    return s


def monotonic_to(params, width, height, factor=1.1):
    """Is the radial mapping r_d(r) increasing out to factor x the corner radius (normalised coordinates)?"""
    fx, fy, cx, cy = params[:4]
    k1, k2, p1, p2, k3, k4, k5, k6 = params[4:12]
    rc = np.max(np.hypot(np.array([0, width]) - cx, np.array([0, height])[:, None] - cy)) / fx
    # r_d is the distorted radius of the undistorted r; invertibility needs d r_d / d r > 0 up to the r whose r_d = factor*rc
    r = np.linspace(1e-4, 4.0, 40000)
    rd = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6) / (1 + k4 * r ** 2 + k5 * r ** 4 + k6 * r ** 6)
    inc = np.diff(rd) > 0
    first_stop = rd[np.argmax(~inc)] if (~inc).any() else rd[-1]
    return bool(first_stop >= factor * rc), float(first_stop / rc)


def run(label, root):
    proj = SfmProject.load(root)
    rec0 = pycolmap.Reconstruction(str(Path(root) / "sparse" / "cahv_ba"))
    key_of = {int(v): k for k, v in proj.settings["database"]["cameras"].items()}
    nav = [cid for cid in rec0.cameras if key_of.get(int(cid)) in ("NL", "NR")]
    res, cams = {}, {}
    for vname, (free, tangential) in VARIANTS.items():
        rec = copy.deepcopy(rec0)
        p = copy.deepcopy(proj)
        for k in ("NL", "NR"):
            p.cameras[k]["free_params"] = list(free)
        if not tangential:
            for cid in nav:
                c = rec.cameras[cid]
                q = np.array(c.params); q[IP1] = q[IP2] = 0.0; c.params = q
        t = time.time()
        ba = bundle_adjust(rec, p, sigma_px=0.5, loss_scale=2.0, refine_intrinsics=True, refine_tangential=tangential,
                           refine_rig="rotation", max_iterations=200, attitude_prior_deg=1.0)
        o = observations(rec, p)
        st = residual_stats(o)
        entry = {"free": free, "tangential": tangential, "final_cost": float(ba["final_cost"]),
                 "iterations": ba["iterations"], "seconds": round(time.time() - t, 1), **st, "cameras": {}}
        for cid in nav:
            c = rec.cameras[cid]; k = key_of[int(cid)]
            q = np.array(c.params, float)
            mono, reach = monotonic_to(q, c.width, c.height)
            entry["cameras"][k] = {n: float(v) for n, v in zip(FULL_OPENCV_NAMES, q)}
            entry["cameras"][k].update(monotonic_to_1p1_corner=mono, monotonic_reach_over_corner=reach)
            cams[(vname, k)] = PixelCamera.colmap(c.model.name, q, c.width, c.height, name=f"{vname} {k}")
        res[vname] = entry
        print(f"{label:14s} {vname:11s} cost {entry['final_cost']:.1f}  rms {st['rms_px']:.4f}  med {st['median_px']:.4f}  "
              f"corner med {st['corner_median_px']:.4f}  field {st['residual_field_rms_px']:.4f} "
              f"(corner cells {st['residual_field_corner_rms_px']:.4f})  {entry['seconds']} s", flush=True)
    base = res["k4"]
    n = base["n_obs"] * 2
    for vname, e in res.items():
        k_extra = (len(e["free"]) - 1) * 2 + (0 if e["tangential"] else -4)        # two cameras
        e["delta_cost"] = e["final_cost"] - base["final_cost"]
        e["delta_params"] = k_extra
        e["delta_bic"] = 2 * e["delta_cost"] + k_extra * np.log(n)
        e["camera_change_from_k4"] = {}
        for k in ("NL", "NR"):
            d = compare_cameras(cams[("k4", k)], cams[(vname, k)], step=96.0)
            e["camera_change_from_k4"][k] = {"rms_px": float(d["rms_px"]), "centre_rms_px": float(d["centre_rms_px"]),
                                             "corner_rms_px": float(d["corner_rms_px"]), "rotation_deg": float(d["rotation_deg"])}
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else Path("lens_terms.json")
    args = [a for a in args if a != str(out)]
    results = {}
    for a in args:
        label, root = a.split("=", 1)
        results[label] = run(label, root)
        out.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", out)
