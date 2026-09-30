"""
Navcam focal length against camera temperature (v0p31).

The flight team's Navcam camera models are interpolated by the camera-plate
temperature (``GEOMETRIC_CAMERA_MODEL.INTERPOLATION_VALUE``), but their focal
length hardly changes with it (less than 0.01 px over the scapes).  The refined
MPPP cameras of the scapes differ by up to ~2 px in focal length, in both eyes
together, and across scapes the difference follows the scape's median camera
temperature (notebook 04 section 2a).  This module measures the effect inside
one block and corrects for it.

Temperatures
    :func:`label_temperatures` reads the camera temperatures of one PDS label
    (``NAVCAM_LEFT_CAL`` / ``NAVCAM_RIGHT_CAL`` in
    ``INSTRUMENT_STATE_PARMS``: both eyes are in every Navcam label, and the
    image's own one is the model's interpolation temperature).
    :func:`interpolate_temperatures` fills the images of a block from a few
    sampled labels, linearly in spacecraft clock within a sol.
    :func:`image_temperatures` gives each project image its eye's temperature:
    from the project/manifest (``camera_temperature_degC``, MPPP >= 0.30), from
    a table of label samples, or by reading the labels under ``pds_dir``.

Temperature bins
    :func:`temperature_bins` puts every exposure (frame) in a bin of
    ``bin_deg`` by the mean temperature of its images; bins with fewer than
    ``min_images`` images are merged into their nearer neighbour.
    :func:`split_by_temperature` rebuilds a solved reconstruction with one
    Navcam camera per eye and bin (the stereo rig copied per bin, so both eyes
    of an exposure stay in one rig) and adds the bin cameras to a copy of the
    project with everything but the free parameters (default ``fx``, ``fy``)
    held.  :func:`thermal_adjust` runs the bundle adjustment on it, after the
    same adjustment with one camera per eye (the reference cost), and reports
    per bin the refined focal lengths.  :func:`fit_focal_temperature` fits
    ``f = a_scape + b T`` over the bins of one or several scapes: the scape
    offsets absorb everything that differs between scapes, so ``b`` is the
    within-block thermal coefficient.

Correction
    :func:`thermal_start_cameras` sets the bins' start focal lengths from a
    thermal model (``beta`` in ppm/degC about ``T0``) - with the bins held
    this is the temperature-corrected calibration for weak networks.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

LEFT_KEY, RIGHT_KEY = "NAVCAM_LEFT_CAL", "NAVCAM_RIGHT_CAL"


# ----------------------------------------------------------------------------- temperatures
def label_temperatures(path: PathLike) -> Dict[str, Optional[float]]:
    """Camera temperatures of one Navcam PDS product (label only): ``NL``, ``NR`` (degC) and ``interp``, the
    value the label's camera model was interpolated to (the image's own eye)."""
    from ..labels import label_get, read_pds
    L, _ = read_pds(path, load_image=False)
    names = label_get(L, "INSTRUMENT_STATE_PARMS.INSTRUMENT_TEMPERATURE_NAME") or []
    vals = label_get(L, "INSTRUMENT_STATE_PARMS.INSTRUMENT_TEMPERATURE") or []
    t = {}
    for k, v in zip(names, vals):
        try:
            t[str(k)] = float(getattr(v, "value", v))
        except (TypeError, ValueError):
            pass
    g = label_get(L, "GEOMETRIC_CAMERA_MODEL") or {}
    interp = None
    if str(g.get("INTERPOLATION_METHOD") or "").upper() == "TEMPERATURE":
        try:
            interp = float(g.get("INTERPOLATION_VALUE"))
        except (TypeError, ValueError):
            interp = None
    return {"NL": t.get(LEFT_KEY), "NR": t.get(RIGHT_KEY), "interp": interp}


def zcam_label_temperature(path: PathLike, key: str = "HEAD_FPA") -> Optional[float]:
    """v0p42: the Mastcam-Z camera temperature of one PDS product (label only): the focal-plane sensor ``HEAD_FPA``
    (also recorded: ``DEA``, ``HEAD_HTR_1``, ``HEAD_HTR_2``), degC; None without it."""
    from ..labels import label_get, read_pds
    L, _ = read_pds(path, load_image=False)
    names = label_get(L, "INSTRUMENT_STATE_PARMS.INSTRUMENT_TEMPERATURE_NAME") or []
    vals = label_get(L, "INSTRUMENT_STATE_PARMS.INSTRUMENT_TEMPERATURE") or []
    for k, v in zip(names, vals):
        if str(k) == key:
            try:
                return float(getattr(v, "value", v))
            except (TypeError, ValueError):
                return None
    return None


def _sol_sclk(stem: str) -> Tuple[int, float]:
    from ..filenames import parse_filename
    fn = parse_filename(stem + ".IMG")
    return int(fn.sol), float(fn.sclk)


def interpolate_temperatures(samples: Dict[str, Dict[str, Optional[float]]], stems: Iterable[str],
                             max_gap_s: float = 1800.0) -> Dict[str, Dict[str, Any]]:
    """
    Both eyes' camera temperatures for ``stems`` (PDS product stems) from label ``samples`` ({stem: {"NL", "NR"}}):
    exact where the stem (or its exposure) was sampled, else linear in spacecraft clock between the nearest samples
    of the same sol, else the nearest sample of the sol within ``max_gap_s``.  Returns {stem: {"NL", "NR", "source",
    "gap_s"}}; stems of sols without samples are left out.
    """
    by_sol: Dict[int, List[Tuple[float, float, float]]] = {}
    exact: Dict[Tuple[int, str], Dict[str, float]] = {}
    for s, v in samples.items():
        if v.get("NL") is None or v.get("NR") is None:
            continue
        sol, sclk = _sol_sclk(s)
        by_sol.setdefault(sol, []).append((sclk, float(v["NL"]), float(v["NR"])))
        exact[(sol, f"{sclk:.3f}")] = v
    for sol in by_sol:
        by_sol[sol].sort()
    out: Dict[str, Dict[str, Any]] = {}
    for s in stems:
        sol, sclk = _sol_sclk(s)
        e = exact.get((sol, f"{sclk:.3f}"))
        if e is not None:
            out[s] = {"NL": float(e["NL"]), "NR": float(e["NR"]), "source": "label", "gap_s": 0.0}
            continue
        pts = by_sol.get(sol)
        if not pts:
            continue
        t = np.array([p[0] for p in pts])
        i = int(np.searchsorted(t, sclk))
        if 0 < i < len(pts):
            (t0, l0, r0), (t1, l1, r1) = pts[i - 1], pts[i]
            w = (sclk - t0) / (t1 - t0) if t1 > t0 else 0.0
            out[s] = {"NL": l0 + w * (l1 - l0), "NR": r0 + w * (r1 - r0), "source": "interpolated",
                      "gap_s": float(min(sclk - t0, t1 - sclk))}
        else:
            j = 0 if i == 0 else len(pts) - 1
            gap = abs(pts[j][0] - sclk)
            if gap <= max_gap_s:
                out[s] = {"NL": pts[j][1], "NR": pts[j][2], "source": "nearest", "gap_s": float(gap)}
    return out


def image_temperatures(project, samples: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
                       pds_dir: Optional[PathLike] = None, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Each Navcam image's camera temperature (its own eye) keyed by image name: from the project record
    (``camera_temperature_degC``), else from ``samples`` (label temperatures, interpolated), else from the labels
    under ``pds_dir`` (every image's own label).  Returns {name: {"T", "T_NL", "T_NR", "source"}}.
    """
    out: Dict[str, Dict[str, Any]] = {}
    nav = [r for r in project.images if str(r.get("instrument", "")).startswith("N")]
    todo = []
    for r in nav:
        v = r.get("camera_temperature_degC")
        if v is not None:
            out[r["name"]] = {"T": float(v), "source": "manifest"}
        else:
            todo.append(r)
    if todo and samples:
        interp = interpolate_temperatures(samples, [r["stem"] for r in todo])
        for r in list(todo):
            v = interp.get(r["stem"])
            if v is None:
                continue
            eye = "NL" if str(r["instrument"]).startswith("NL") else "NR"
            out[r["name"]] = {"T": float(v[eye]), "T_NL": v["NL"], "T_NR": v["NR"], "source": v["source"],
                              "gap_s": v["gap_s"]}
        todo = [r for r in todo if r["name"] not in out]
    if todo and pds_dir:
        from ..select import iter_imgs
        want = {r["stem"]: r for r in todo}
        for fp, fn in iter_imgs(pds_dir):
            r = want.pop(fn.stem, None)
            if r is None:
                continue
            t = label_temperatures(fp)
            eye = "NL" if str(r["instrument"]).startswith("NL") else "NR"
            T = t.get(eye) if t.get(eye) is not None else t.get("interp")
            if T is not None:
                out[r["name"]] = {"T": float(T), "T_NL": t.get("NL"), "T_NR": t.get("NR"), "source": "label"}
            if not want:
                break
    if verbose:
        n = len(nav)
        print(f"[thermal] temperatures for {len(out)} of {n} Navcam images "
              f"({', '.join(f'{k} {v}' for k, v in _count(x['source'] for x in out.values()).items())})")
    return out


def _count(xs: Iterable[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


# ----------------------------------------------------------------------------- bins
def temperature_bins(frame_T: Dict[int, float], frame_n: Dict[int, int], bin_deg: float = 10.0,
                     min_images: int = 8) -> Dict[int, Tuple[float, float]]:
    """
    Bin of every frame (``frame_T``: frame id -> temperature; ``frame_n``: images per frame) as its (lo, hi)
    edges in degC.  Bins are multiples of ``bin_deg``; a bin with fewer than ``min_images`` images is merged
    into the neighbour whose centre is nearer (repeatedly, smallest first).
    """
    if not frame_T:
        return {}
    edges = {f: (np.floor(T / bin_deg) * bin_deg, np.floor(T / bin_deg) * bin_deg + bin_deg) for f, T in frame_T.items()}
    while True:
        bins = sorted(set(edges.values()))
        if len(bins) < 2:
            break
        n = {b: sum(frame_n.get(f, 1) for f, e in edges.items() if e == b) for b in bins}
        small = [b for b in bins if n[b] < min_images]
        if not small:
            break
        b = min(small, key=lambda x: n[x])
        k = bins.index(b)
        nb = [bins[j] for j in (k - 1, k + 1) if 0 <= j < len(bins)]
        c = 0.5 * (b[0] + b[1])
        tgt = min(nb, key=lambda x: abs(0.5 * (x[0] + x[1]) - c))
        merged = (min(b[0], tgt[0]), max(b[1], tgt[1]))
        for f, e in list(edges.items()):
            if e in (b, tgt):
                edges[f] = merged
    return {f: (float(e[0]), float(e[1])) for f, e in edges.items()}


def _tag(lo: float, hi: float) -> str:
    return f"T{int(round(lo)):+04d}{int(round(hi)):+04d}"


def rig_rotation_at(R, t, dT: float, yaw_mdeg_per_degC: float, pitch_mdeg_per_degC: float = 0.0):
    """v0p35: the stereo rig (x_R = R x_L + t) at a temperature dT from the one it describes, with the right camera's
    rotation changing by (pitch, yaw) slopes x dT (rotation vector in the right camera frame) and its centre fixed:
    R' = Rot R, t' = Rot t."""
    from scipy.spatial.transform import Rotation
    Q = Rotation.from_rotvec(np.radians(1e-3 * np.array([pitch_mdeg_per_degC * dT, yaw_mdeg_per_degC * dT, 0.0]))).as_matrix()
    return Q @ np.asarray(R, float), Q @ np.asarray(t, float)


def rig_slopes_for_project(project) -> Optional[Dict[str, float]]:
    """v0p35: the rig's temperature model of the project's start rig (``M2020_N_rig.json`` with a ``thermal``
    entry, recorded by ``SfmProject.create``): {"yaw_mdeg_per_degC", "pitch_mdeg_per_degC", "T0_degC"} or None."""
    rig = ((project.settings.get("navcam_cameras") or {}).get("rig") or {})
    th = rig.get("thermal")
    return th if th and th.get("yaw_mdeg_per_degC") is not None else None


def split_by_temperature(rec, project, temps: Dict[str, Dict[str, Any]], bin_deg: float = 10.0,
                         min_images: int = 8, free: Sequence[str] = ("fx", "fy"),
                         thermal_model: Optional[Dict[str, Dict[str, float]]] = None,
                         rig_slopes: Optional[Dict[str, float]] = None):
    """
    A copy of ``rec`` with one camera per Navcam eye and temperature bin, and a copy of ``project`` whose cameras
    and database mapping include them (``fixed_params``: all but ``free``).  ``thermal_model``: {"NL": {"ppm_per_degC",
    "T0_degC"}, ...} - the bins then start from the eye's camera scaled by 1 + ppm 1e-6 (T_bin - T0) (fx and fy).
    Frames without a temperature keep the original cameras.  ``rig_slopes`` (v0p35): {"yaw_mdeg_per_degC",
    "pitch_mdeg_per_degC"} - each bin's rig starts from the block's rig turned to the bin temperature
    (:func:`rig_rotation_at`, relative to the mean frame temperature of the block).  Returns (rec, project, bins)
    with ``bins``: one row per bin camera.
    """
    import pycolmap
    from .project import FULL_OPENCV_NAMES
    from .reconstruction import _PARAM_NAMES
    proj = copy.deepcopy(project)
    db_cams = dict(proj.settings.get("database", {}).get("cameras", {}))
    key_of = {int(v): k for k, v in db_cams.items()}
    nav_cids = {cid for cid, k in key_of.items() if str(k).startswith("N")}
    reg = set(rec.reg_image_ids())

    frame_T: Dict[int, float] = {}
    frame_n: Dict[int, int] = {}
    for fid, fr in rec.frames.items():
        ims = [d.id for d in fr.data_ids if d.id in reg]
        Ts = [temps[rec.images[i].name]["T"] for i in ims
              if rec.images[i].camera_id in nav_cids and rec.images[i].name in temps]
        if Ts:
            frame_T[fid] = float(np.mean(Ts))
            frame_n[fid] = len(ims)
    fbin = temperature_bins(frame_T, frame_n, bin_deg, min_images)
    bins = sorted(set(fbin.values()))

    new = pycolmap.Reconstruction()
    for cid, cam in rec.cameras.items():
        new.add_camera(cam)
    next_cid = max(rec.cameras) + 1
    bin_cam: Dict[Tuple[int, Tuple[float, float]], int] = {}
    rows = []
    for cid in sorted(nav_cids):
        if cid not in rec.cameras:
            continue
        base = rec.cameras[cid]
        key = key_of[cid]
        for b in bins:
            members = [f for f, e in fbin.items() if e == b]
            Tb = [temps[rec.images[d.id].name]["T"] for f in members for d in rec.frames[f].data_ids
                  if d.id in reg and rec.images[d.id].camera_id == cid and rec.images[d.id].name in temps]
            if not Tb:
                continue
            p = np.array(base.params, float)
            tm = float(np.median(Tb))
            if thermal_model and key in thermal_model:
                m = thermal_model[key]
                s = 1.0 + 1e-6 * float(m["ppm_per_degC"]) * (tm - float(m["T0_degC"]))
                p[0] *= s
                p[1] *= s
                p[2] += float(m.get("cx_px_per_degC") or 0.0) * (tm - float(m["T0_degC"]))     # v0p40
                p[3] += float(m.get("cy_px_per_degC") or 0.0) * (tm - float(m["T0_degC"]))
            c = pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width, height=base.height, params=p)
            new.add_camera(c)
            bin_cam[(cid, b)] = next_cid
            bkey = f"{key}_{_tag(*b)}"
            names = _PARAM_NAMES.get(base.model.name, FULL_OPENCV_NAMES)
            pc = dict(proj.cameras.get(key, {}))
            pc.update({"params": p.tolist(), "group": key, "temperature_bin_degC": list(b),
                       "temperature_median_degC": tm, "n_images": len(Tb),
                       "fixed_params": [n for n in names if n not in set(free)], "free_params": [],
                       "source": f"{pc.get('source', key)}; temperature bin {b[0]:g}..{b[1]:g} degC"})
            proj.cameras[bkey] = pc
            db_cams[bkey] = next_cid
            rows.append({"camera": bkey, "eye": key, "camera_id": next_cid, "bin_degC": list(b),
                         "T_median_degC": tm, "T_min_degC": float(np.min(Tb)), "T_max_degC": float(np.max(Tb)),
                         "images": len(Tb), "start_fx": float(p[0]), "start_fy": float(p[1])})
            next_cid += 1
    proj.settings.setdefault("database", {})["cameras"] = db_cams

    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))     # noqa: E731
    next_rid = max(rec.rigs) + 1
    rig_for: Dict[Tuple[int, Tuple[float, float]], int] = {}
    for rid, rig in rec.rigs.items():
        new.add_rig(rig)

    def _rig_of(rid: int, b) -> int:
        if (rid, b) in rig_for:
            return rig_for[(rid, b)]
        nonlocal next_rid
        old = rec.rigs[rid]
        ref = old.ref_sensor_id.id
        r = pycolmap.Rig(rig_id=next_rid)
        r.add_ref_sensor(sensor(bin_cam.get((ref, b), ref)))
        for sid in old.non_ref_sensors:
            T = old.sensor_from_rig(sid)
            if rig_slopes and frame_T:
                Tb = [frame_T[f] for f, e in fbin.items() if e == b and f in frame_T]
                if Tb:
                    dT = float(np.mean(Tb)) - float(np.mean(list(frame_T.values())))
                    R2, t2 = rig_rotation_at(np.asarray(T.rotation.matrix()), np.asarray(T.translation), dT,
                                             float(rig_slopes.get("yaw_mdeg_per_degC", 0.0)),
                                             float(rig_slopes.get("pitch_mdeg_per_degC", 0.0)))
                    T = pycolmap.Rigid3d(pycolmap.Rotation3d(R2), t2)
            r.add_sensor(sensor(bin_cam.get((sid.id, b), sid.id)), T)
        new.add_rig(r)
        rig_for[(rid, b)] = next_rid
        next_rid += 1
        return rig_for[(rid, b)]

    cam_of_image: Dict[int, int] = {}
    for fid, fr in rec.frames.items():
        b = fbin.get(fid)
        rid = fr.rig_id if b is None else _rig_of(fr.rig_id, b)
        nf = pycolmap.Frame(frame_id=fid, rig_id=rid)
        for d in fr.data_ids:
            im = rec.images[d.id]
            cid = bin_cam.get((im.camera_id, b), im.camera_id) if b is not None else im.camera_id
            cam_of_image[d.id] = cid
            nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cid), id=d.id))
        if fr.has_pose:
            nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
    for iid, im in rec.images.items():
        kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
        ni = pycolmap.Image(name=im.name, keypoints=kps, camera_id=cam_of_image.get(iid, im.camera_id), image_id=iid)
        ni.frame_id = im.frame_id
        new.add_image(ni)
    for fid in rec.reg_frame_ids():                        # deregistered frames stay unregistered
        if new.frames[fid].has_pose:
            new.register_frame(fid)
    for pid, pt in rec.points3D.items():
        tr = pycolmap.Track()
        for el in pt.track.elements:
            tr.add_element(el.image_id, el.point2D_idx)
        new.add_point3D(pt.xyz, tr, pt.color)
    return new, proj, rows


def _stats(rec, project) -> Dict[str, float]:
    from .reconstruction import native_residuals
    r = native_residuals(rec, project)["residual_native_px"]
    r = r[np.isfinite(r)]
    return {"observations": int(r.size), "median_px": float(np.median(r)), "rms_px": float(np.sqrt(np.mean(r ** 2))),
            "p95_px": float(np.percentile(r, 95))}


def thermal_adjust(rec, project, temps: Dict[str, Dict[str, Any]], bin_deg: float = 10.0, min_images: int = 8,
                   free: Sequence[str] = ("fx", "fy"), sigma_px: float = 0.5, loss_scale: float = 2.0,
                   max_iterations: int = 100, thermal_model: Optional[Dict[str, Dict[str, float]]] = None,
                   reference: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """
    The temperature-bin test on a solved block (see the module docstring).  ``reference``: first the same adjustment
    with one camera per eye (``free`` refined, everything else held, rig held), so that the bin result is compared
    with the same freedom minus the temperature dependence.  Returns {"reference", "reference_rec", "bins", "rows",
    "rec", "project"}.
    """
    from .reconstruction import bundle_adjust
    att = float((project.settings.get("reconstruction") or {}).get("attitude_prior_deg") or 1.0)
    out: Dict[str, Any] = {"bin_deg": bin_deg, "min_images": min_images, "free": list(free)}
    if reference:
        r0 = copy.deepcopy(rec)
        p0 = copy.deepcopy(project)
        from .project import FULL_OPENCV_NAMES
        from .reconstruction import _PARAM_NAMES
        for k, c in p0.cameras.items():
            if str(k).startswith("N"):
                names = _PARAM_NAMES.get(c.get("model", "FULL_OPENCV"), FULL_OPENCV_NAMES)
                c["fixed_params"] = [n for n in names if n not in set(free)]
                c["free_params"] = []
        ba0 = bundle_adjust(r0, p0, sigma_px=sigma_px, loss_scale=loss_scale, refine_rig=False,
                            max_iterations=max_iterations, attitude_prior_deg=att)
        cams0 = {k: [float(x) for x in r0.cameras[int(v)].params[:4]]
                 for k, v in p0.settings["database"]["cameras"].items() if str(k).startswith("N") and int(v) in r0.cameras}
        out["reference"] = {"cost": ba0["final_cost"], "initial_cost": ba0["initial_cost"], "seconds": ba0["seconds"],
                            "stats": _stats(r0, p0), "cameras": cams0}
        out["reference_rec"] = r0
        if verbose:
            print(f"[thermal] one camera per eye: cost {ba0['final_cost']:.1f} ({ba0['seconds']:.0f} s), "
                  f"{out['reference']['stats']}", flush=True)
    r1, p1, rows = split_by_temperature(rec, project, temps, bin_deg, min_images, free, thermal_model, rig_slopes)
    ba1 = bundle_adjust(r1, p1, sigma_px=sigma_px, loss_scale=loss_scale, refine_rig=False,
                        max_iterations=max_iterations, attitude_prior_deg=att)
    for row in rows:
        p = r1.cameras[row["camera_id"]].params
        row.update({"fx": float(p[0]), "fy": float(p[1]), "cx": float(p[2]), "cy": float(p[3])})
    obs = _obs_per_camera(r1)
    for row in rows:
        row["observations"] = int(obs.get(row["camera_id"], 0))
    out["bins"] = {"cost": ba1["final_cost"], "initial_cost": ba1["initial_cost"], "seconds": ba1["seconds"],
                   "stats": _stats(r1, p1), "n_bins": len({tuple(r["bin_degC"]) for r in rows})}
    out["rows"] = rows
    out["rec"] = r1
    out["project"] = p1
    if verbose:
        print(f"[thermal] {out['bins']['n_bins']} temperature bins: cost {ba1['final_cost']:.1f} "
              f"({ba1['seconds']:.0f} s), {out['bins']['stats']}", flush=True)
        for r in rows:
            print(f"   {r['camera']:18s} {r['images']:4d} images  T {r['T_median_degC']:6.1f} degC "
                  f"[{r['T_min_degC']:.1f}, {r['T_max_degC']:.1f}]  fx {r['fx']:.2f}  fy {r['fy']:.2f}  "
                  f"({r['observations']} obs)", flush=True)
    return out


def _obs_per_camera(rec) -> Dict[int, int]:
    n: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            c = rec.images[el.image_id].camera_id
            n[c] = n.get(c, 0) + 1
    return n


# ----------------------------------------------------------------------------- fits
def fit_focal_temperature(rows: Sequence[Dict[str, Any]], param: str = "fx", min_observations: int = 2000,
                          weight: bool = True) -> Dict[str, Any]:
    """
    ``param = a_scape + b T`` over temperature-bin rows (``scape``, ``eye``, ``T_median_degC``, ``param``,
    ``observations``) - one offset per scape and eye, one slope per eye; weights sqrt(observations).  Returns per
    eye {"px_per_degC", "sd", "ppm_per_degC", "rows", "scapes", "residual_rms_px", "offsets"}.
    """
    out: Dict[str, Any] = {}
    for eye in sorted({r["eye"] for r in rows}):
        rr = [r for r in rows if r["eye"] == eye and r.get("observations", 0) >= min_observations and r.get(param) is not None]
        scapes = sorted({r.get("scape", "") for r in rr})
        # scapes with a single bin carry no slope information but are kept for their offset
        if len(rr) < 3 or len(rr) <= len(scapes):
            continue
        A = np.zeros((len(rr), len(scapes) + 1))
        y = np.array([float(r[param]) for r in rr])
        for i, r in enumerate(rr):
            A[i, scapes.index(r.get("scape", ""))] = 1.0
            A[i, -1] = float(r["T_median_degC"])
        w = np.sqrt(np.array([float(r.get("observations", 1)) for r in rr])) if weight else np.ones(len(rr))
        w = w / w.mean()
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        res = y - A @ coef
        dof = max(1, len(rr) - A.shape[1])
        s2 = float(np.sum((w * res) ** 2)) / dof
        cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None])) * s2
        b = float(coef[-1])
        f0 = float(np.mean(y))
        out[eye] = {"param": param, "px_per_degC": b, "sd": float(np.sqrt(cov[-1, -1])), "ppm_per_degC": b / f0 * 1e6,
                    "ppm_sd": float(np.sqrt(cov[-1, -1])) / f0 * 1e6, "rows": len(rr), "scapes": len(scapes),
                    "residual_rms_px": float(np.sqrt(np.mean(res ** 2))),
                    "offsets": {s: float(c) for s, c in zip(scapes, coef[:-1])}}
    return out


def thermal_start_cameras(cameras: Dict[str, Dict[str, Any]], model: Dict[str, Dict[str, float]],
                          T: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Cameras (``{"NL": {...params...}}``) scaled to temperatures ``T`` ({"NL": degC}) with ``model``
    ({"NL": {"ppm_per_degC", "T0_degC"}}): fx and fy times 1 + ppm 1e-6 (T - T0)."""
    out = {}
    for k, c in cameras.items():
        c = copy.deepcopy(c)
        if k in model and k in T:
            s = 1.0 + 1e-6 * float(model[k]["ppm_per_degC"]) * (float(T[k]) - float(model[k]["T0_degC"]))
            c["params"] = list(map(float, c["params"]))
            c["params"][0] *= s
            c["params"][1] *= s
        out[k] = c
    return out


# ----------------------------------------------------------------------------- pipeline stage
def thermal_stage(rec, project, temps: Dict[str, Dict[str, Any]], bin_deg: float = 10.0, min_images: int = 8,
                  free: Sequence[str] = ("fx", "fy"), hold: bool = False,
                  thermal_model: Optional[Dict[str, Dict[str, float]]] = None, sigma_px: float = 0.5,
                  loss_scale: float = 2.0, max_iterations: int = 100, attitude_prior_deg: Optional[float] = None,
                  linear_solver: str = "auto", verbose: bool = True, rig_slopes: Optional[Dict[str, float]] = None):
    """
    The final stage of :func:`mppp.sfm.reconstruction.reconstruct` with ``thermal_bins_deg`` (v0p31): the solved
    block is split into Navcam temperature bins (:func:`split_by_temperature`) and adjusted once more with the bins'
    ``free`` parameters refined (or, with ``hold``, every bin held at its start: the eye's camera scaled to the bin
    temperature by ``thermal_model``), everything else of the cameras and the rig held, poses and points free.
    ``project`` is updated in place (bin cameras and their database ids).  Returns (rec, report).
    """
    from .reconstruction import bundle_adjust
    before = _stats(rec, project)
    r1, p1, rows = split_by_temperature(rec, project, temps, bin_deg, min_images, free, thermal_model, rig_slopes)
    project.cameras = p1.cameras
    project.settings.setdefault("database", {})["cameras"] = p1.settings["database"]["cameras"]
    bin_keys = [r["camera"] for r in rows]
    for k in bin_keys:
        project.cameras[k]["thermal_bin"] = True
    # every image of a binned frame now belongs to its bin camera (as Mastcam-Z images to their focus bins); the
    # eye's camera is kept in base_instrument so that strip_thermal_bins can undo the split before a rerun
    key_of_id = {int(v): k for k, v in project.settings["database"]["cameras"].items()}
    by_name = {r["name"]: r for r in project.images}
    for iid, im in r1.images.items():
        r = by_name.get(im.name)
        k = key_of_id.get(int(im.camera_id))
        if r is not None and k in bin_keys and r.get("instrument") != k:
            r.setdefault("base_instrument", r["instrument"])
            r["instrument"] = k
    held = [k for k in project.cameras if not str(k).startswith("N")] + (bin_keys if hold else [])
    kw = {} if attitude_prior_deg is None else {"attitude_prior_deg": attitude_prior_deg}
    ba = bundle_adjust(r1, project, sigma_px=sigma_px, loss_scale=loss_scale, refine_rig=False,
                       max_iterations=max_iterations, hold_cameras=held, linear_solver=linear_solver, **kw)
    obs = _obs_per_camera(r1)
    for row in rows:
        p = r1.cameras[row["camera_id"]].params
        row.update({"fx": float(p[0]), "fy": float(p[1]), "cx": float(p[2]), "cy": float(p[3]),
                    "observations": int(obs.get(row["camera_id"], 0))})
        project.cameras[row["camera"]]["refined_params"] = [float(x) for x in p]
    after = _stats(r1, project)
    report = {"bin_deg": float(bin_deg), "min_images": int(min_images), "free": list(free), "held": bool(hold),
              "rig_slopes": rig_slopes,
              "thermal_model": thermal_model, "bins": len({tuple(r["bin_degC"]) for r in rows}), "rows": rows,
              "before": before, "after": after, "initial_cost": ba["initial_cost"], "final_cost": ba["final_cost"],
              "seconds": ba["seconds"],
              "images_with_temperature": int(sum(1 for r in project.images if r["name"] in temps))}
    if verbose:
        print(f"[sfm] thermal stage: {report['bins']} temperature bins of {bin_deg:g} degC "
              f"({'held at the thermal model' if hold else 'f refined'}): median {before['median_px']:.4f} -> "
              f"{after['median_px']:.4f}, rms {before['rms_px']:.4f} -> {after['rms_px']:.4f} native px, "
              f"cost {ba['initial_cost']:.1f} -> {ba['final_cost']:.1f}", flush=True)
        for r in rows:
            print(f"      {r['camera']:18s} {r['images']:4d} images  T {r['T_median_degC']:6.1f} degC  "
                  f"fx {r['fx']:.2f}  fy {r['fy']:.2f}", flush=True)
    return r1, report


def strip_thermal_bins(project) -> int:
    """Undo :func:`thermal_stage` (and v0p40 the Mastcam-Z focus-state split, :mod:`mppp.sfm.backlash`) on a
    project (v0p31): images back to their eye's (focus bin's) camera, bin cameras removed,
    the bin table dropped.  Called before the database is built and before a reconstruction, so that a rerun starts
    from one camera per eye and the stereo rig.  Returns the number of bin cameras removed."""
    n = 0
    for r in project.images:
        if "base_instrument" in r:
            r["instrument"] = r.pop("base_instrument")
    for k in [k for k, c in project.cameras.items() if c.get("thermal_bin") or c.get("state_split")]:   # v0p40: also
        # the regular-focus-state Mastcam-Z cameras (mppp.sfm.backlash.split_by_state)
        project.cameras.pop(k)
        n += 1
    db = project.settings.get("database", {}).get("cameras")
    if isinstance(db, dict):
        for k in [k for k in db if k not in project.cameras]:
            db.pop(k)
    project.settings.pop("thermal", None)
    project.settings.pop("zcam_backlash", None)
    for r in project.images:
        r.pop("backlash_state", None)
    return n


def thermal_model_for_project(project) -> Optional[Dict[str, Dict[str, float]]]:
    """v0p31: the thermal model of the project's Navcam start cameras, referred to the temperature they were scaled
    to (``SfmProject.create`` from a consensus with a ``thermal`` entry): {"NL": {"ppm_per_degC", "T0_degC"}, ...}
    for :func:`thermal_stage` (bins held at the start camera scaled to the bin temperature).  None if the start
    cameras carry no thermal model."""
    nav = (project.settings.get("navcam_cameras") or {})
    out = {}
    for eye in ("NL", "NR"):
        th = (nav.get(eye) or {}).get("thermal")
        if th and th.get("ppm_per_degC") is not None and th.get("T_median_degC") is not None:
            out[eye] = {"ppm_per_degC": float(th["ppm_per_degC"]), "T0_degC": float(th["T_median_degC"])}
            for k in ("cx_px_per_degC", "cy_px_per_degC"):          # v0p40: principal point against temperature
                if th.get(k):
                    out[eye][k] = float(th[k])
    return out or None
