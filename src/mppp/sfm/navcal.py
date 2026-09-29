"""
Navcam calibration across scapes (v0p35): rig stability, one joint calibration bundle, lens-model choice.

Three studies on solved notebook-03 blocks (``sparse/cahv_ba`` + ``project.json``), one camera temperature per
image (:func:`mppp.sfm.thermal.image_temperatures`):

* **Rig stability** (:func:`rig_study`): each block is re-adjusted with the right camera's rotation in the rig
  free (the pipeline's ``refine_rig="rotation"``), then with the whole right-from-left pose free (baseline length
  and direction set by the tie points and the position priors), and once more with one rig per 10 degC bin; every
  rig comes with its covariance (3-D points eliminated, frame poses and cameras marginalised).  :func:`rig_tests`
  tests the rigs against camera temperature, the left-right temperature difference, sol and network strength.
* **Joint calibration** (:func:`merge_scapes`, :func:`joint_calibration`): all blocks in one bundle adjustment
  with one camera per eye, one rig and each block's own poses and points.  The thermal model
  f(T) = f0 (1 + b (T - T0)) enters through the keypoints (``bundle_adjust(keypoint_scale=...)``), and b is
  profiled (the cost minimum over a grid of b).  The shared cameras and rig carry their covariance.
* **Lens model** (:func:`loo_lens`): the joint calibration without one block predicts that block's cameras; the
  rational (``FULL_OPENCV``, k4 free) and the fisheye + tangential (``THIN_PRISM_FISHEYE``, sx1 = sy1 = 0) models
  are compared on the held-out fit and on how far the prediction is from the block's own calibration.

Everything is in full-frame Navcam pixels (5120 x 3840); the covariances are for the assumed keypoint sigma and
are also given rescaled by the variance factor of the adjustment.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .project import SfmProject

PathLike = Union[str, Path]
RATIONAL = "FULL_OPENCV"
FISHEYE_T = "THIN_PRISM_FISHEYE"
PARAMS = {RATIONAL: ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
          FISHEYE_T: ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")}
FREE = {RATIONAL: ["k4"], FISHEYE_T: []}         # beyond reconstruction's defaults (fx fy cx cy k1 k2 k3 + p1 p2)
BA_DEFAULTS = dict(sigma_px=0.5, loss_scale=2.0, attitude_prior_deg=5.0, refine_tangential=True)


# ================================================================ loading
@dataclass
class Scape:
    name: str
    root: Path
    project: SfmProject
    rec: Any                                        # pycolmap.Reconstruction
    temps: Dict[str, Dict[str, Any]] = field(default_factory=dict)      # image name -> {"T", "T_NL", "T_NR"}

    def image_T(self) -> Dict[int, float]:
        return {int(iid): float(self.temps[im.name]["T"]) for iid, im in self.rec.images.items()
                if im.name in self.temps}


def align_project_to_model(project: SfmProject, rec) -> int:
    """Make the project describe exactly the model's images (by name): image, camera and frame ids from the
    model; model images the project does not know are deregistered.  Returns the number of such frames."""
    from .reconstruction import exclude_frames
    by_name = {r["name"]: r for r in project.images}
    missing = {im.frame_id for im in rec.images.values() if im.name not in by_name}
    if missing:
        exclude_frames(rec, sorted(missing))
    keep, cams = [], {}
    for iid, im in rec.images.items():
        r = by_name.get(im.name)
        if r is None:
            continue
        r["image_id"], r["camera_id"], r["frame_id"] = int(iid), int(im.camera_id), int(im.frame_id)
        cams.setdefault(r["instrument"], int(im.camera_id))
        keep.append(r)
    project.images = keep
    project.settings.setdefault("database", {})["cameras"] = dict(cams)
    return len(missing)


def load_scape(name: str, root: PathLike, samples: Optional[Dict[str, Dict[str, Any]]] = None,
               model: str = "cahv_ba") -> Scape:
    """A solved block with its camera temperatures (project records, else label ``samples``)."""
    import pycolmap
    from .thermal import image_temperatures
    root = Path(root)
    if (root / "colmap").is_dir():
        root = root / "colmap"
    project = SfmProject.load(root)
    rec = pycolmap.Reconstruction(str(root / "sparse" / model))
    align_project_to_model(project, rec)
    temps = image_temperatures(project, samples=samples)
    return Scape(name, root, project, rec, temps)


# ================================================================ rig geometry
def stereo_rig(rec) -> Tuple[Optional[int], Optional[int], Optional[Any]]:
    """(rig_id, right camera_id, sensor_from_rig) of the rig with a second sensor (the Navcam pair)."""
    for rid, rig in rec.rigs.items():
        for sid in rig.non_ref_sensors:
            return int(rid), int(sid.id), rig.sensor_from_rig(sid)
    return None, None, None


def rig_angles(R: np.ndarray, R_ref: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Right-from-left rotation as yaw (about the left camera's y axis - it shifts disparity), pitch (about x -
    vertical parallax) and roll (about z), mdeg; relative to ``R_ref`` if given."""
    from scipy.spatial.transform import Rotation
    M = R if R_ref is None else R @ np.asarray(R_ref).T
    rv = Rotation.from_matrix(M).as_rotvec()
    return {"yaw_mdeg": float(np.degrees(rv[1]) * 1e3), "pitch_mdeg": float(np.degrees(rv[0]) * 1e3),
            "roll_mdeg": float(np.degrees(rv[2]) * 1e3)}


def stereo_offset(left, right, sensor_from_rig) -> Dict[str, float]:
    """Disparity at infinity (x_left - x_right, px) and vertical parallax (y_left - y_right) of the ray through the
    left principal point: what the rig rotation and the two principal points together do to stereo."""
    R = np.asarray(sensor_from_rig.rotation.matrix())
    pl = np.asarray(left.params[2:4], float)
    d = np.array(list(left.cam_from_img(pl)) + [1.0])
    x = R @ d
    pr = np.asarray(right.img_from_cam(x), float).ravel()
    return {"disparity_inf_px": float(pl[0] - pr[0]), "vparallax_inf_px": float(pl[1] - pr[1])}


def _quat_xyz_cov_to_angles(cq: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Covariance of (qx, qy, qz) of a near-identity-relative quaternion -> covariance of (yaw, pitch, roll) in
    mdeg^2 (rotation vector = 2 q_xyz for |q_xyz| << 1; yaw = y, pitch = x, roll = z)."""
    J = np.zeros((3, 3))
    k = 2.0 * np.degrees(1.0) * 1e3
    J[0, 1] = J[1, 0] = J[2, 2] = k                  # rows yaw, pitch, roll <- columns qx, qy, qz
    return scale * (J @ cq @ J.T)


def _rig_rows(ba: Dict[str, Any], rec, R_ref: Optional[np.ndarray], mode: str) -> List[Dict[str, Any]]:
    """Rig parameters and their standard deviations after an adjustment, one row per rig with a second sensor."""
    rows = []
    cov = (ba.get("covariance") or {})
    vf = float(cov.get("variance_factor", 1.0))
    blocks = cov.get("blocks", {})
    for rid, rig in rec.rigs.items():
        for sid in rig.non_ref_sensors:
            T = rig.sensor_from_rig(sid)
            R = np.asarray(T.rotation.matrix())
            t = np.asarray(T.translation)
            C = -R.T @ t                               # right centre in the left camera frame
            ang = rig_angles(R, R_ref)
            row = {"rig_id": int(rid), "camera_id": int(sid.id), "mode": mode, **ang,
                   **{k.replace("_mdeg", "_abs_mdeg"): v for k, v in rig_angles(R).items()},
                   "baseline_m": float(np.linalg.norm(C)), "C_right_m": C.tolist(), "variance_factor": vf}
            c = blocks.get(f"rig:{rid}:{sid.id}")
            if c is not None:
                c = np.asarray(c)
                if mode == "rotation" and c.shape == (3, 3):
                    ca = _quat_xyz_cov_to_angles(c)
                    row.update({"sd_yaw_mdeg": float(np.sqrt(ca[0, 0] * vf)), "sd_pitch_mdeg": float(np.sqrt(ca[1, 1] * vf)),
                                "sd_roll_mdeg": float(np.sqrt(ca[2, 2] * vf)),
                                "sd_yaw_mdeg_nominal": float(np.sqrt(ca[0, 0])), "cov_ypr_mdeg2": (ca * vf).tolist()})
                elif mode == "full" and c.shape[0] >= 6:
                    # tangent (qx, qy, qz, tx, ty, tz) with w held (SubsetManifold(7, [3]))
                    ca = _quat_xyz_cov_to_angles(c[:3, :3])
                    # baseline length and direction from the translation t = -R C: C = -R^T t
                    Jc = -R.T
                    cC = Jc @ c[3:6, 3:6] @ Jc.T
                    u = C / np.linalg.norm(C)
                    var_b = float(u @ cC @ u)
                    P = np.eye(3) - np.outer(u, u)
                    var_dir = P @ cC @ P / np.linalg.norm(C) ** 2       # rad^2
                    row.update({"sd_yaw_mdeg": float(np.sqrt(ca[0, 0] * vf)), "sd_pitch_mdeg": float(np.sqrt(ca[1, 1] * vf)),
                                "sd_roll_mdeg": float(np.sqrt(ca[2, 2] * vf)),
                                "sd_baseline_mm": float(np.sqrt(var_b * vf) * 1e3),
                                "sd_baseline_direction_mdeg": float(np.degrees(np.sqrt(np.trace(var_dir) * vf)) * 1e3),
                                "cov_C_m2": (cC * vf).tolist()})
            rows.append(row)
    return rows


# ================================================================ rig study
def network_summary(sc: Scape) -> Dict[str, Any]:
    """Size and geometry of a block's Navcam network, and its temperatures and sols."""
    rec, p = sc.rec, sc.project
    reg = set(rec.reg_image_ids())
    names = {rec.images[i].name for i in reg}
    rows = [r for r in p.images if r["name"] in names]
    st = {}
    for r in rows:
        st.setdefault(r["station"], []).append(np.asarray(r["prior_C"], float))
    cs = np.array([np.mean(v, axis=0) for v in st.values()])
    span = float(np.max(np.linalg.norm(cs[:, None] - cs[None], axis=2))) if len(cs) > 1 else 0.0
    n_obs = n_cross = n_pts = 0
    stat_of = {r["image_id"]: r["station"] for r in rows}
    for pt in rec.points3D.values():
        ss = {stat_of.get(el.image_id) for el in pt.track.elements}
        n_pts += 1
        n_obs += pt.track.length()
        n_cross += len(ss - {None}) > 1
    T = [sc.temps[n]["T"] for n in names if n in sc.temps]
    TL = [sc.temps[r["name"]].get("T_NL") for r in rows if r["name"] in sc.temps]
    TR = [sc.temps[r["name"]].get("T_NR") for r in rows if r["name"] in sc.temps]
    dLR = [a - b for a, b in zip(TL, TR) if a is not None and b is not None]
    sols = [int(r["sol"]) for r in rows]
    full = [r for r in rows if float(r.get("downsample_scale", 1.0)) >= 0.99]
    return {"scape": sc.name, "images": len(reg), "frames": len(rec.reg_frame_ids()), "stations": len(st),
            "span_m": span, "points": n_pts, "observations": n_obs, "cross_station_fraction": n_cross / max(n_pts, 1),
            "full_res_fraction": len(full) / max(len(rows), 1), "sol_median": float(np.median(sols)),
            "sol_min": int(min(sols)), "sol_max": int(max(sols)), "T_median_degC": float(np.median(T)) if T else None,
            "T_min_degC": float(np.min(T)) if T else None, "T_max_degC": float(np.max(T)) if T else None,
            "dT_LR_median_degC": float(np.median(dLR)) if dLR else None, "n_dT_LR": len(dLR)}


def reference_rig(sc: Scape, which: str = "cahv") -> Optional[np.ndarray]:
    """A fixed reference for the rig angles: the label (CAHV) pairs' rig (``which="cahv"``, the same in every
    block to 1e-4 deg) or the rig the block started from (``"start"``: the shipped consensus)."""
    r = sc.project.rig or {}
    r = r.get("N", r)
    R = r.get("R_sensor_from_ref_cahv") if which == "cahv" else None
    R = R if R is not None else (r.get("R_sensor_from_ref") or r.get("R"))
    return np.asarray(R, float) if R is not None else None


def rig_study(sc: Scape, bin_deg: float = 10.0, min_images: int = 8, max_iterations: int = 100,
              modes: Sequence[str] = ("rotation", "rotation_pp", "full", "bins"),
              common_pp: Optional[Dict[str, Sequence[float]]] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Re-adjust one block several ways and return the rigs with their covariances:

    * ``rotation``: the pipeline's final adjustment (intrinsics and the right camera's rotation in the rig free);
    * ``rotation_pp``: the same with both principal points held at ``common_pp`` ({"NL": (cx, cy), "NR": ...}, the
      same for every block).  The rig yaw and the principal-point difference both shift the disparity at infinity
      and trade against each other, so only with common principal points is the yaw comparable between blocks: it
      then carries the block's whole disparity offset (and the pitch its vertical parallax);
    * ``full``: the right camera's rotation and position in the rig free (baseline length and direction);
    * ``bins``: one camera per eye and ``bin_deg`` temperature bin (fx, fy free, everything else held) and one rig
      per bin (rotation free) - the rig against temperature inside the block.
    """
    import time
    import pycolmap
    from .reconstruction import bundle_adjust
    from .thermal import split_by_temperature
    R_ref = reference_rig(sc)
    out: Dict[str, Any] = {"scape": sc.name, "network": network_summary(sc),
                           "R_reference": None if R_ref is None else R_ref.tolist()}
    for mode in modes:
        t0 = time.time()
        rec = copy.deepcopy(sc.rec)
        proj = copy.deepcopy(sc.project)
        extra = {}
        ba_kw = {}
        if mode == "rotation_pp":
            if not common_pp:
                continue
            key_of = {int(v): k for k, v in proj.settings.get("database", {}).get("cameras", {}).items()}
            for cid, cam in rec.cameras.items():
                k = key_of.get(int(cid))
                if k in common_pp:
                    q = np.array(cam.params, float)
                    q[2:4] = np.asarray(common_pp[k], float)
                    cam.params = q
            ba_kw["refine_principal_point"] = False
        if mode == "bins":
            temps = {n: {"T": v["T"]} for n, v in sc.temps.items()}
            rec, proj, bins = split_by_temperature(rec, proj, temps, bin_deg=bin_deg, min_images=min_images)
            extra["bins"] = bins
            if len({tuple(b["bin_degC"]) for b in bins}) < 2:
                out[mode] = {"skipped": "one temperature bin", **extra}
                continue
        ba = bundle_adjust(rec, proj, refine_rig=(True if mode == "full" else "rotation"), max_iterations=max_iterations,
                           covariance=True, **BA_DEFAULTS, **ba_kw)
        rows = _rig_rows(ba, rec, R_ref, "full" if mode == "full" else "rotation")
        key_of = {int(v): k for k, v in proj.settings.get("database", {}).get("cameras", {}).items()}
        for r in rows:
            ref_cam = rec.rigs[r["rig_id"]].ref_sensor_id.id
            sid = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(r["camera_id"]))
            r.update(stereo_offset(rec.cameras[int(ref_cam)], rec.cameras[int(r["camera_id"])],
                                   rec.rigs[r["rig_id"]].sensor_from_rig(sid)))
        if mode == "bins":
            # the bin of each rig: its reference sensor's camera -> bin row
            cam_bin = {b["camera_id"]: b for b in extra["bins"]}
            for r in rows:
                ref_cam = rec.rigs[r["rig_id"]].ref_sensor_id.id
                b = cam_bin.get(int(ref_cam)) or cam_bin.get(int(r["camera_id"]))
                if b:
                    r.update({"bin_degC": b["bin_degC"], "T_median_degC": b["T_median_degC"], "bin_images": b["images"]})
                r["frames"] = sum(1 for f in rec.frames.values() if f.rig_id == r["rig_id"] and f.has_pose)
        out[mode] = {"rigs": rows, "final_cost": ba["final_cost"], "iterations": ba["iterations"],
                     "observations": ba["observations"], "variance_factor": (ba.get("covariance") or {}).get("variance_factor"),
                     "seconds": time.time() - t0, **({"bins": extra["bins"]} if "bins" in extra else {})}
        if verbose:
            r0 = rows[0] if rows else {}
            print(f"  {sc.name} {mode:8s} {len(rows)} rig(s)  yaw {r0.get('yaw_mdeg', float('nan')):+.2f} "
                  f"+- {r0.get('sd_yaw_mdeg', float('nan')):.2f} mdeg  baseline {r0.get('baseline_m', float('nan')):.5f} m"
                  f" (+- {r0.get('sd_baseline_mm', float('nan')):.2f} mm)  {time.time() - t0:.0f} s", flush=True)
    return out


def _wls(y: np.ndarray, X: np.ndarray, var: np.ndarray) -> Dict[str, Any]:
    """Weighted least squares with an additive between-block variance tau^2 (method of moments, DerSimonian-Laird
    generalised to a regression).  Returns coefficients, their standard errors, tau and Q."""
    w = 1.0 / var
    W = np.diag(w)
    XtWX = X.T @ W @ X
    b = np.linalg.solve(XtWX, X.T @ W @ y)
    r = y - X @ b
    Q = float(np.sum(w * r ** 2))
    k, p = X.shape
    # tau^2 (moments): Q - (k - p) = tau^2 * tr(W - W X (X'WX)^-1 X' W)
    P = W - W @ X @ np.linalg.solve(XtWX, X.T @ W)
    tau2 = max(0.0, (Q - (k - p)) / float(np.trace(P)))
    w2 = 1.0 / (var + tau2)
    W2 = np.diag(w2)
    A = X.T @ W2 @ X
    b2 = np.linalg.solve(A, X.T @ W2 @ y)
    se = np.sqrt(np.diag(np.linalg.inv(A)))
    from scipy import stats
    pval = 2 * stats.norm.sf(np.abs(b2 / se))
    q_p = float(stats.chi2.sf(Q, k - p)) if k > p else float("nan")
    return {"coef": b2.tolist(), "se": se.tolist(), "p": pval.tolist(), "tau": math.sqrt(tau2), "Q": Q, "dof": k - p,
            "p_homogeneous": q_p}


def rig_tests(studies: Sequence[Dict[str, Any]], angle: str = "yaw", mode: str = "rotation_pp") -> Dict[str, Any]:
    """
    Across blocks: is the refined rig (``angle`` of the rotation-mode rig) the same everywhere, given its formal
    uncertainty (Q test, between-block scatter tau), and does it follow camera temperature, the left-right
    temperature difference, sol or network strength (one covariate at a time, weighted with var_i + tau^2)?
    Within blocks: the per-bin rigs against the bin temperature, with one offset per block.
    """
    rows = []
    for s in studies:
        r = (s.get(mode) or {}).get("rigs") or []
        if not r:
            continue
        n = s["network"]
        rows.append({"scape": s["scape"], "y": r[0][f"{angle}_mdeg"], "sd": r[0].get(f"sd_{angle}_mdeg", np.nan), **n})
    y = np.array([r["y"] for r in rows])
    sd = np.array([r["sd"] for r in rows])
    var = np.maximum(sd, 1e-6) ** 2
    k = len(rows)
    X0 = np.ones((k, 1))
    base = _wls(y, X0, var)
    out = {"angle": angle, "mode": mode, "n": k, "mean_mdeg": base["coef"][0], "se_mean_mdeg": base["se"][0], "tau_mdeg": base["tau"],
           "Q": base["Q"], "dof": base["dof"], "p_homogeneous": base["p_homogeneous"],
           "median_sd_mdeg": float(np.nanmedian(sd)), "scatter_sd_mdeg": float(np.std(y, ddof=1)) if k > 1 else float("nan"),
           "rows": [{"scape": r["scape"], "value_mdeg": r["y"], "sd_mdeg": r["sd"]} for r in rows], "covariates": {}}
    cov_names = {"T_median_degC": "camera temperature", "dT_LR_median_degC": "left-right temperature difference",
                 "sol_median": "sol", "stations": "stations", "span_m": "network span",
                 "cross_station_fraction": "cross-station fraction", "observations": "observations",
                 "full_res_fraction": "full-resolution fraction"}
    for c, label in cov_names.items():
        x = np.array([np.nan if r.get(c) is None else float(r[c]) for r in rows])
        ok = np.isfinite(x)
        if ok.sum() < 4:
            continue
        xs = x[ok]
        if c == "observations":
            xs = np.log10(xs)
        X = np.c_[np.ones(ok.sum()), xs - xs.mean()]
        f = _wls(y[ok], X, var[ok])
        from scipy import stats
        rho = stats.spearmanr(xs, y[ok])
        out["covariates"][c] = {"label": label + (" (log10)" if c == "observations" else ""), "slope": f["coef"][1],
                                "se": f["se"][1], "p": f["p"][1], "tau_mdeg": f["tau"], "spearman": float(rho.statistic),
                                "spearman_p": float(rho.pvalue), "n": int(ok.sum()),
                                "x_range": [float(xs.min()), float(xs.max())]}
    # within blocks: per-bin rigs
    wy, wx, wv, grp = [], [], [], []
    for s in studies:
        b = (s.get("bins") or {}).get("rigs") or []
        b = [r for r in b if r.get("T_median_degC") is not None and np.isfinite(r.get(f"sd_{angle}_mdeg", np.nan))]
        if len(b) < 2:
            continue
        for r in b:
            wy.append(r[f"{angle}_mdeg"])
            wx.append(r["T_median_degC"])
            wv.append(r[f"sd_{angle}_mdeg"] ** 2)
            grp.append(s["scape"])
    if len(set(grp)) >= 1 and len(wy) >= 3:
        g = sorted(set(grp))
        X = np.zeros((len(wy), len(g) + 1))
        for i, gg in enumerate(grp):
            X[i, g.index(gg)] = 1.0
        X[:, -1] = np.array(wx)
        wv = np.array(wv)
        f = _wls(np.array(wy), X, wv)
        out["within"] = {"slope_mdeg_per_degC": f["coef"][-1], "se": f["se"][-1], "p": f["p"][-1], "tau_mdeg": f["tau"],
                         "Q": f["Q"], "dof": f["dof"], "p_homogeneous": f["p_homogeneous"], "blocks": len(g), "bins": len(wy),
                         "rows": [{"scape": a, "T_degC": b, "value_mdeg": c, "sd_mdeg": float(np.sqrt(d))}
                                  for a, b, c, d in zip(grp, wx, wy, wv)]}
    return out


# ================================================================ merging blocks
def _subsample_points(rec, n: Optional[int], seed: int = 0, min_track: int = 2) -> List[int]:
    """Point ids to keep: all if ``n`` is None, else ``n`` drawn without replacement with probability
    proportional to min(track length, 6) (long tracks carry the intrinsics)."""
    ids = [pid for pid, pt in rec.points3D.items() if pt.track.length() >= min_track]
    if n is None or len(ids) <= n:
        return ids
    L = np.array([min(rec.points3D[p].track.length(), 6) for p in ids], float)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=n, replace=False, p=L / L.sum())
    return [ids[i] for i in sorted(pick)]


def thin_points(rec, n: Optional[int], seed: int = 0) -> int:
    """Keep at most ``n`` 3-D points of ``rec`` (in place; :func:`_subsample_points`, long tracks preferred).
    Memory of an adjustment with covariances grows by ~4 kB per observation, so blocks of 10^6 observations are
    thinned to fit in 8 GB.  Returns the number of points removed."""
    if n is None or len(rec.points3D) <= n:
        return 0
    keep = set(_subsample_points(rec, n, seed=seed, min_track=2))
    drop = [pid for pid in rec.points3D if pid not in keep]
    for pid in drop:
        rec.delete_point3D(pid)
    return len(drop)


def merge_scapes(scapes, cameras: Dict[str, Any], rig: Tuple[np.ndarray, np.ndarray],
                 points_per_scape: Optional[int] = 25000, seed: int = 0, exclude: Sequence[str] = ()) -> Tuple[Any, SfmProject, Dict[str, Any]]:
    """
    One reconstruction holding every block: camera 1 = NL, camera 2 = NR (``cameras``: {"NL", "NR"} ->
    pycolmap.Camera, shared by all blocks), rig 1 = the stereo pair (``rig`` = (R, t) of NR from NL), rig 2 / 3 =
    NR / NL alone; each block's frames, images, poses and (subsampled) points with fresh ids.  The merged project
    lists every image with its own block's priors (each block keeps its own world frame).  Returns
    (rec, project, index) with ``index`` = {"scape_of_image": {image_id: scape}, "images": {scape: [image_id]},
    "frames": {scape: [frame_id]}}.
    """
    import pycolmap
    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))      # noqa: E731
    new = pycolmap.Reconstruction()
    for cid, key in ((1, "NL"), (2, "NR")):
        c = cameras[key]
        new.add_camera(pycolmap.Camera(camera_id=cid, model=c.model, width=c.width, height=c.height,
                                       params=np.array(c.params, float)))
    r1 = pycolmap.Rig(rig_id=1)
    r1.add_ref_sensor(sensor(1))
    r1.add_sensor(sensor(2), pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(rig[0], float)), np.asarray(rig[1], float)))
    new.add_rig(r1)
    r2 = pycolmap.Rig(rig_id=2)
    r2.add_ref_sensor(sensor(2))
    new.add_rig(r2)
    r3 = pycolmap.Rig(rig_id=3)
    r3.add_ref_sensor(sensor(1))
    new.add_rig(r3)
    proj = None
    seen_names = set()
    idx: Dict[str, Any] = {"scape_of_image": {}, "images": {}, "frames": {}, "points": {}, "temps": {}}
    next_img, next_frame = 1, 1
    for sc in scapes:
        if sc.name in exclude:
            continue
        if proj is None:
            proj = copy.deepcopy(sc.project)
            proj.images = []
            proj.cameras = {k: {"model": cameras[k].model.name, "params": list(map(float, cameras[k].params)),
                                "width": cameras[k].width, "height": cameras[k].height,
                                "free_params": FREE.get(cameras[k].model.name, [])} for k in ("NL", "NR")}
            proj.settings = dict(proj.settings)
            proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
        idx["temps"].update({n: float(v["T"]) for n, v in sc.temps.items()})
        rec = sc.rec
        key_of = {int(v): k for k, v in sc.project.settings.get("database", {}).get("cameras", {}).items()}
        by_name = {r["name"]: r for r in sc.project.images}
        reg = set(rec.reg_image_ids())
        img_map: Dict[int, int] = {}
        idx["images"][sc.name], idx["frames"][sc.name] = [], []
        for fid, fr in rec.frames.items():
            if not fr.has_pose:
                continue
            ims = [d.id for d in fr.data_ids if d.id in reg and rec.images[d.id].name in by_name]
            if not ims:
                continue
            eyes = sorted({key_of.get(int(rec.images[i].camera_id), "") for i in ims})
            rid = 1 if eyes == ["NL", "NR"] else (2 if eyes == ["NR"] else 3)
            nf = pycolmap.Frame(frame_id=next_frame, rig_id=rid)
            pending = []
            for i in ims:
                im = rec.images[i]
                if im.name in seen_names:
                    raise ValueError(f"{im.name} is in more than one block")
                seen_names.add(im.name)
                cid = 1 if key_of.get(int(im.camera_id)) == "NL" else 2
                img_map[i] = next_img
                nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cid), id=next_img))
                kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
                ni = pycolmap.Image(name=im.name, keypoints=kps, camera_id=cid, image_id=next_img)
                ni.frame_id = next_frame
                pending.append(ni)
                r = dict(by_name[im.name])
                r.update({"image_id": next_img, "camera_id": cid, "frame_id": next_frame, "scape": sc.name})
                proj.images.append(r)
                idx["scape_of_image"][next_img] = sc.name
                idx["images"][sc.name].append(next_img)
                next_img += 1
            # the frame pose: the reference image's cam_from_world is the rig_from_world of rig 1/2/3
            ref = [i for i in ims if key_of.get(int(rec.images[i].camera_id)) == ("NL" if rid in (1, 3) else "NR")][0]
            nf.rig_from_world = rec.images[ref].cam_from_world()
            new.add_frame(nf)
            for ni in pending:
                new.add_image(ni)
            idx["frames"][sc.name].append(next_frame)
            next_frame += 1
        for fid in idx["frames"][sc.name]:
            new.register_frame(fid)
        keep = _subsample_points(rec, points_per_scape, seed=seed)
        n_pts = 0
        for pid in keep:
            pt = rec.points3D[pid]
            tr = pycolmap.Track()
            for el in pt.track.elements:
                if el.image_id in img_map:
                    tr.add_element(img_map[el.image_id], el.point2D_idx)
            if tr.length() >= 2:
                new.add_point3D(pt.xyz, tr, pt.color)
                n_pts += 1
        idx["points"][sc.name] = n_pts
    return new, proj, idx


def subset(rec, idx: Dict[str, Any], keep: Sequence[str]):
    """A copy of a merged reconstruction with only the blocks ``keep`` registered and their points."""
    out = copy.deepcopy(rec)
    keep = set(keep)
    s_of = idx["scape_of_image"]
    drop_frames = [f for sc, fs in idx["frames"].items() if sc not in keep for f in fs]
    drop_pts = [pid for pid, pt in out.points3D.items() if s_of.get(pt.track.elements[0].image_id) not in keep]
    for pid in drop_pts:
        out.delete_point3D(pid)
    reg = set(out.reg_frame_ids())
    for f in drop_frames:
        if f in reg:
            out.deregister_frame(f)
    return out


def keypoint_scales(proj: SfmProject, temps: Dict[str, float], ppm_per_degC: float, T0: float) -> Dict[int, float]:
    """{image_id: 1 + b (T - T0)} for the merged project (b in ppm/degC)."""
    out = {}
    for r in proj.images:
        T = temps.get(r["name"])
        if T is not None:
            out[int(r["image_id"])] = 1.0 + 1e-6 * float(ppm_per_degC) * (float(T) - float(T0))
    return out


def shared_state(rec) -> Dict[str, Any]:
    """The shared cameras and rig of a merged reconstruction."""
    _, _, T = stereo_rig(rec)
    R = np.asarray(T.rotation.matrix())
    t = np.asarray(T.translation)
    return {"NL": {"model": rec.cameras[1].model.name, "params": list(map(float, rec.cameras[1].params))},
            "NR": {"model": rec.cameras[2].model.name, "params": list(map(float, rec.cameras[2].params))},
            "rig_R": R.tolist(), "rig_t": t.tolist(), "rig_angles_abs": rig_angles(R), "baseline_m": float(np.linalg.norm(t))}


def rig_keypoint_map(proj: SfmProject, temps: Dict[str, float], yaw_mdeg_per_degC: float,
                     pitch_mdeg_per_degC: float, T0: float, right_camera_ids: Sequence[int] = (2,)) -> Callable:
    """
    The rig's temperature dependence as a keypoint map for :func:`bundle_adjust`: the right camera's rotation in
    the rig is R(T) = Rot(delta(T)) R(T0) with delta = (pitch, yaw, 0) slopes x (T - T0) (rotation vector in the
    right camera frame; yaw about y shifts disparity, pitch about x is vertical parallax), so a right keypoint x
    becomes pi(Rot(-delta) pi^-1(x)) - where the shared rig R(T0) would have put it (exact for the rotation; the
    baseline turns by delta, 0.15 mm per 20 mdeg, which is neglected).
    """
    from scipy.spatial.transform import Rotation
    T_of = {int(r["image_id"]): temps.get(r["name"]) for r in proj.images}
    right = set(int(c) for c in right_camera_ids)
    ky, kp = 1e-3 * math.radians(1.0) * float(yaw_mdeg_per_degC), 1e-3 * math.radians(1.0) * float(pitch_mdeg_per_degC)

    def fmap(iid, kps, cam):
        if int(cam.camera_id) not in right:
            return kps
        T = T_of.get(int(iid))
        if T is None:
            return kps
        dT = float(T) - float(T0)
        Rm = Rotation.from_rotvec([-kp * dT, -ky * dT, 0.0]).as_matrix()
        rays = np.array(cam.cam_from_img(kps), float)
        X = np.c_[rays[:, :2], np.ones(len(rays))] @ Rm.T
        out = np.array(cam.img_from_cam(X, check_cheirality=False), float)
        return np.where(np.isfinite(out), out, kps)
    return fmap


def joint_adjust(rec, proj: SfmProject, temps: Dict[str, float], ppm_per_degC: float, T0: float,
                 covariance: bool = False, max_iterations: int = 100, refine_rig: Union[bool, str] = "rotation",
                 hold_cameras: Sequence[str] = (), rig_slopes: Optional[Tuple[float, float]] = None,
                 **kw) -> Dict[str, Any]:
    """One adjustment of a merged reconstruction with the thermal model at slope ``ppm_per_degC`` and, with
    ``rig_slopes`` = (yaw, pitch) mdeg/degC, the rig's temperature dependence (:func:`rig_keypoint_map`)."""
    from .reconstruction import bundle_adjust
    ks = keypoint_scales(proj, temps, ppm_per_degC, T0)
    if rig_slopes and any(rig_slopes):
        kw["keypoint_map"] = rig_keypoint_map(proj, temps, rig_slopes[0], rig_slopes[1], T0)
    args = dict(BA_DEFAULTS)
    args.setdefault("linear_solver", "sparse_schur")        # blocks share only the cameras and the rig: very sparse
    args.update(kw)
    ba = bundle_adjust(rec, proj, refine_rig=refine_rig, max_iterations=max_iterations, covariance=covariance,
                       keypoint_scale=ks, hold_cameras=hold_cameras, **args)
    ba["ppm_per_degC"] = float(ppm_per_degC)
    ba["rig_slopes_mdeg_per_degC"] = list(rig_slopes) if rig_slopes else None
    ba["T0_degC"] = float(T0)
    return ba


def profile_slope(rec, proj: SfmProject, temps: Dict[str, float], T0: float, grid: Sequence[float],
                  max_iterations: int = 100, verbose: bool = True,
                  rig_slopes: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Cost of the joint adjustment over a grid of thermal slopes (each started from the same state); the slope is
    the minimum of a parabola through the three lowest points, its standard deviation var_factor / curvature."""
    import time
    rows = []
    for b in grid:
        t0 = time.time()
        r = copy.deepcopy(rec)
        ba = joint_adjust(r, proj, temps, b, T0, max_iterations=max_iterations, rig_slopes=rig_slopes)
        red = max(2 * ba["observations"] - 1, 1)
        rows.append({"ppm_per_degC": float(b), "cost": float(ba["final_cost"]), "iterations": ba["iterations"],
                     "variance_factor": float(2 * ba["final_cost"] / red), "state": shared_state(r)})
        if verbose:
            print(f"  slope {b:6.1f} ppm/degC  cost {ba['final_cost']:.2f}  it {ba['iterations']}  "
                  f"fx NL {r.cameras[1].params[0]:.3f} NR {r.cameras[2].params[0]:.3f}  {time.time() - t0:.0f} s", flush=True)
    return fit_profile(rows)


def fit_profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    x = np.array([r["ppm_per_degC"] for r in rows])
    c = np.array([r["cost"] for r in rows])
    A = np.c_[np.ones_like(x), x, x ** 2]
    a0, a1, a2 = np.linalg.lstsq(A, c, rcond=None)[0]
    best = -a1 / (2 * a2) if a2 > 0 else float(x[np.argmin(c)])
    vf = float(np.median([r["variance_factor"] for r in rows]))
    sd = math.sqrt(vf / (2 * a2)) if a2 > 0 else float("nan")          # cost = 1/2 chi2: var = vf / C''
    return {"rows": rows, "best_ppm_per_degC": float(best), "sd_ppm_per_degC": sd, "curvature": float(2 * a2),
            "variance_factor": vf, "fit": [float(a0), float(a1), float(a2)]}


# ================================================================ camera comparison and conversion
def pycolmap_camera(model: str, params: Sequence[float], width: int = 5120, height: int = 3840):
    import pycolmap
    return pycolmap.Camera(model=model, width=width, height=height, params=np.asarray(params, float))


def compare(a, b, step: float = 64.0, rotation_radius: float = 0.85) -> Dict[str, float]:
    """Where camera ``b`` puts the rays of a grid of ``a``'s pixels (pycolmap cameras, any model), after the best
    rotation (fitted inside ``rotation_radius`` of the half-diagonal) is removed: rms / centre / corner pixels."""
    w, h = a.width, a.height
    u, v = np.meshgrid(np.arange(step / 2, w, step), np.arange(step / 2, h, step))
    uv = np.c_[u.ravel(), v.ravel()]
    ra = np.array(a.cam_from_img(uv), float)
    rb = np.array(b.cam_from_img(uv), float)
    da = np.c_[ra, np.ones(len(ra))]
    db = np.c_[rb, np.ones(len(rb))]
    da /= np.linalg.norm(da, axis=1)[:, None]
    db /= np.linalg.norm(db, axis=1)[:, None]
    rr = np.linalg.norm(uv - np.array([w, h]) / 2, axis=1) / (np.hypot(w, h) / 2)
    ok = np.all(np.isfinite(da), 1) & np.all(np.isfinite(db), 1)
    sel = ok & (rr <= rotation_radius)
    H = da[sel].T @ db[sel]
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Q = Vt.T @ D @ U.T                                   # da -> rotated into b's frame
    X = (da[ok] @ Q.T)
    pb = np.array(b.img_from_cam(X, check_cheirality=False), float)
    d = np.linalg.norm(pb - uv[ok], axis=1)
    r_ok = rr[ok]
    from scipy.spatial.transform import Rotation
    return {"rms_px": float(np.sqrt(np.nanmean(d ** 2))), "centre_rms_px": float(np.sqrt(np.nanmean(d[r_ok < 0.3] ** 2))),
            "corner_rms_px": float(np.sqrt(np.nanmean(d[r_ok > 0.85] ** 2))), "max_px": float(np.nanmax(d)),
            "rotation_mdeg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(Q).as_rotvec())) * 1e3)}


def to_fisheye_tangential(cam) -> Tuple[Any, float]:
    """A THIN_PRISM_FISHEYE camera (sx1 = sy1 = 0) fitted to ``cam``'s rays over a pixel grid; p1, p2 are fitted
    too.  Returns (camera, fit rms px)."""
    from scipy.optimize import least_squares
    import pycolmap
    w, h = cam.width, cam.height
    u, v = np.meshgrid(np.linspace(1, w - 1, 64), np.linspace(1, h - 1, 48))
    px = np.stack([u.ravel(), v.ravel()], 1)
    rays = np.array(cam.cam_from_img(px), float)
    rays = np.c_[rays[:, :2], np.ones(len(rays))]
    p0 = np.asarray(cam.params, float)

    def model(q):
        c = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=w, height=h,
                            params=np.array([q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], 0.0, 0.0]))
        return np.array(c.img_from_cam(rays), float)

    q0 = np.array([p0[0], p0[1], p0[2], p0[3], 0, 0, p0[6] if len(p0) > 7 else 0, p0[7] if len(p0) > 7 else 0, 0, 0], float)
    res = least_squares(lambda q: (model(q) - px).ravel(), q0, method="lm", x_scale="jac")
    rms = float(np.sqrt(np.mean(np.sum((model(res.x) - px) ** 2, 1))))
    q = res.x
    return pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=w, height=h,
                           params=np.array([q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], 0.0, 0.0])), rms


def residual_stats(rec, proj: SfmProject, keypoint_scale: Optional[Dict[int, float]] = None,
                   images: Optional[Sequence[int]] = None, keypoint_map: Optional[Callable] = None) -> Dict[str, float]:
    """Reprojection residuals in native pixels (median, rms, corner median beyond 0.85 of the half-diagonal),
    with the keypoints thermally rescaled as in the adjustment."""
    by = {int(r["image_id"]): float(r.get("downsample_scale", 1.0)) for r in proj.images}
    want = set(images) if images is not None else None
    res, rad = [], []
    cache: Dict[int, np.ndarray] = {}
    for pid, pt in rec.points3D.items():
        for el in pt.track.elements:
            iid = el.image_id
            if want is not None and iid not in want:
                continue
            im = rec.images[iid]
            cam = rec.cameras[im.camera_id]
            if iid not in cache:
                k = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
                if keypoint_scale and iid in keypoint_scale:
                    c0 = np.asarray(cam.params[2:4])
                    k = c0 + (k - c0) / keypoint_scale[iid]
                if keypoint_map is not None and len(k):
                    k = keypoint_map(iid, k, cam)
                cache[iid] = k
            xy = cache[iid][el.point2D_idx]
            p = im.project_point(pt.xyz)
            if p is None:
                continue
            s = by.get(iid, 1.0)
            res.append(np.linalg.norm(np.asarray(p) - xy) * s)
            rad.append(np.linalg.norm(xy - np.array([cam.width, cam.height]) / 2) / (np.hypot(cam.width, cam.height) / 2))
    res, rad = np.asarray(res), np.asarray(rad)
    return {"observations": int(res.size), "median_px": float(np.median(res)), "rms_px": float(np.sqrt(np.mean(res ** 2))),
            "corner_median_px": float(np.median(res[rad > 0.85])) if np.any(rad > 0.85) else float("nan"),
            "p95_px": float(np.percentile(res, 95))}


# ================================================================ frozen camera files
def write_joint_cameras(joint: Dict[str, Any], out_dir: PathLike, loo: Optional[Dict[str, Any]] = None,
                        rig_test: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """
    Write the joint calibration (``joint``: the ``joint_<lens>.json`` of scripts/navcam_calibration_study.py) as
    start cameras for notebook 03 (``SfmProject.create(navcam_cameras=out_dir)``): ``M2020_NL_rational.json``,
    ``M2020_NR_rational.json`` (or ``M2020_N*_fisheye_tangential.json`` for the fisheye + tangential model,
    ``navcam_distortion="fisheye_tangential"``; the camera at T0 with its thermal slope and covariance) and ``M2020_N_rig.json``
    (the joint rig rotation; the translation stays the CAHV baseline).  ``loo``: the leave-one-out results, kept
    as the verification of each camera.
    """
    import json
    from scipy.spatial.transform import Rotation
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    lens = joint.get("lens", "rational")
    for g in ("NL", "NR"):
        c = joint[g]
        names = list(c["params"].keys())
        per = []
        for s, r in (loo or {}).items():
            d = (r.get("camera_difference") or {}).get(g) or {}
            per.append({"scape": s, "rms_px": d.get("rms_px"), "corner_rms_px": d.get("corner_rms_px"),
                        "dfx_px": d.get("dfx_px"), "held_out": True})
        d = {"model": c["model"], "width": int(c["width"]), "height": int(c["height"]),
             "params": [float(c["params"][n]) for n in names], "param_names": names,
             "distortion": ("rational: radial (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2); k5 = k6 = 0"
                            if c["model"] == "FULL_OPENCV" else "fisheye (theta polynomial k1-k4) + tangential p1, p2; sx1 = sy1 = 0"),
             "free_params": FREE.get(c["model"], []),
             "pixel_origin": "corner of the first pixel (COLMAP)",
             "source": (f"v0p35 joint calibration: one {g} camera shared by {len(joint.get('blocks', {}))} Navcam blocks "
                        f"({', '.join(joint.get('blocks', {}))}), {joint['final']['observations']} observations, "
                        f"thermal model in the keypoints; camera at T0 {joint['T0_degC']:.1f} degC"),
             "sd": c.get("sd"), "covariance": {"params": c.get("free_params"), "matrix": c.get("covariance"),
                                                "note": "rescaled by the variance factor of the joint adjustment"},
             "thermal": {"ppm_per_degC": float(joint["ppm_per_degC"]), "sd_ppm_per_degC": joint.get("sd_ppm_per_degC"),
                         "T0_degC": float(joint["T0_degC"]),
                         "note": "v0p35: joint profile over all blocks; SfmProject.create scales fx, fy by "
                                 "1 + ppm_per_degC 1e-6 (T - T0) to the block's median camera temperature"},
             "verification": {"per_scape": per, "method": "leave one block out (scripts/navcam_calibration_study.py loo)"}}
        path = out_dir / (f"M2020_{g}_rational.json" if c["model"] == "FULL_OPENCV" else f"M2020_{g}_fisheye_tangential.json")
        path.write_text(json.dumps(d, indent=1), encoding="utf-8")
        written[g] = path
    rg = joint.get("rig") or {}
    if "C_right_m" in rg:
        # the rig rotation of the joint adjustment; the translation from its centre (unchanged: rotation mode)
        pass
    R = np.asarray(joint.get("rig_R") or np.eye(3), float)
    t = np.asarray(joint.get("rig_t") or [0, 0, 0], float)
    d = {"ref": "NL", "sensor": "NR", "R_sensor_from_ref": R.tolist(),
         "rotvec_rad": [float(v) for v in Rotation.from_matrix(R).as_rotvec()],
         "t_sensor_from_ref_m": t.tolist(), "baseline_m": float(np.linalg.norm(t)),
         "sd_mdeg": {k: rg.get(f"sd_{k}_mdeg") for k in ("yaw", "pitch", "roll")},
         "rig_test": rig_test or {},
         "note": "x_NR = R x_NL + t. v0p35 joint calibration rig; SfmProject.create(navcam_cameras=<this folder>) starts "
                 "the rig rotation here and keeps the project's CAHV translation (the baseline sets the scale)."}
    th = joint.get("rig_thermal") or {}
    if th.get("yaw_mdeg_per_degC"):
        d["thermal"] = {"yaw_mdeg_per_degC": float(th["yaw_mdeg_per_degC"]),
                        "pitch_mdeg_per_degC": float(th.get("pitch_mdeg_per_degC") or 0.0),
                        "sd_yaw_mdeg_per_degC": (th.get("yaw_profile") or {}).get("sd"),
                        "T0_degC": float(joint["T0_degC"]),
                        "note": "v0p35: the right camera turns in the rig with temperature, R(T) = Rot(pitch dT, yaw dT, 0) "
                                "R(T0) (rotation vector in the right camera frame, mdeg/degC); SfmProject.create starts "
                                "the rig at the block's median camera temperature and the thermal stage turns each "
                                "bin's rig by the slope"}
    dr = joint.get("rig_drift")
    if dr:
        d["drift"] = dict(dr, note="v0p35: slow change of the rig over the mission (between-block regression on sol, "
                                   "with the camera temperature); SfmProject.create turns the start rig by these rates x "
                                   "(block median sol - sol0)")
    path = out_dir / "M2020_N_rig.json"
    path.write_text(json.dumps(d, indent=1), encoding="utf-8")
    written["rig"] = path
    return written


def rig_drift(studies: Sequence[Dict[str, Any]], mode: str = "rotation_pp") -> Dict[str, Any]:
    """Rig rotation against sol with the camera temperature as a second covariate (between blocks, weighted with
    var_i + tau^2): the per-sol rates of pitch, yaw and roll, their standard errors and p values, and sol0 (the mean
    sol of the blocks, where the joint rig applies)."""
    rows = [(s, (s.get(mode) or {}).get("rigs") or []) for s in studies]
    rows = [(s, r[0]) for s, r in rows if r]
    T = np.array([s["network"]["T_median_degC"] for s, _ in rows], float)
    sol = np.array([s["network"]["sol_median"] for s, _ in rows], float)
    X = np.c_[np.ones(len(rows)), T - T.mean(), sol - sol.mean()]
    out = {"sol0": float(sol.mean()), "T_mean_degC": float(T.mean()), "blocks": len(rows)}
    for ang in ("pitch", "yaw", "roll"):
        y = np.array([r[f"{ang}_mdeg"] for _, r in rows])
        v = np.array([r.get(f"sd_{ang}_mdeg", np.nan) for _, r in rows]) ** 2
        f = _wls(y, X, np.maximum(v, 1e-6))
        out[f"{ang}_mdeg_per_sol"] = float(f["coef"][2])
        out[f"{ang}_se"] = float(f["se"][2])
        out[f"{ang}_p"] = float(f["p"][2])
        out[f"{ang}_T_mdeg_per_degC"] = float(f["coef"][1])
        out[f"{ang}_tau_mdeg"] = float(f["tau"])
    return out
