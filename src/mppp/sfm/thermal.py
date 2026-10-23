"""
The Navcam temperature bins of a solved block: the final stage of :func:`mppp.sfm.reconstruction.reconstruct`.

The refined Navcam cameras of a block follow the camera temperature (fx, fy by ~40 ppm/degC, the principal points
and k1 by their slopes - the thermal terms of the promoted model, :mod:`mppp.calib`).  :func:`temperature_bins` puts
every exposure (frame) in a bin of ``bin_deg`` by the mean temperature of its images (bins with fewer than
``min_images`` images merged into their nearer neighbour); :func:`split_by_temperature` rebuilds a solved
reconstruction with one Navcam camera per eye and bin (the stereo rig copied per bin), each bin started from the
eye's camera moved to the bin temperature by the thermal model; :func:`thermal_stage` adjusts it once more with the
bins' ``free`` parameters refined (fx, fy, k1), everything else held.  :func:`image_temperatures` gives each project
image its eye's temperature (:mod:`mppp.labels` reads the labels).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import calib
from ..labels import LEFT_KEY, RIGHT_KEY, camera_temperature, instrument_temperatures, label_temperatures  # noqa: F401

log = logging.getLogger("mppp.sfm")
PathLike = Union[str, Path]


# ----------------------------------------------------------------------------- temperatures

def eye_key(key: Any) -> str:
    """The Navcam eye of a camera key: ``"NL#1"`` (a rig-epoch copy) and ``"NL_T-20..-10"`` (a temperature
    bin) -> ``"NL"``; other keys unchanged."""
    k = str(key)
    return k.split("#", 1)[0].split("_T", 1)[0] if k.startswith("N") else k


def image_temperatures(project, samples: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
                       pds_dir: Optional[PathLike] = None, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Each Navcam image's camera temperature (its own eye) keyed by image name: from the project record
    (``camera_temperature_degC``), else from ``samples`` ({stem: {"NL", "NR"}} label temperatures of the same
    exposures), else from the labels under ``pds_dir`` (every image's own label).  Returns {name: {"T", "T_NL",
    "T_NR", "source"}}.
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
        for r in list(todo):
            v = samples.get(r["stem"])
            if v is None or v.get("NL") is None or v.get("NR") is None:
                continue
            eye = "NL" if str(r["instrument"]).startswith("NL") else "NR"
            out[r["name"]] = {"T": float(v[eye]), "T_NL": v["NL"], "T_NR": v["NR"], "source": "label"}
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
        log.info("temperatures for %d of %d Navcam images (%s)", len(out), len(nav),
                 ", ".join(f"{k} {v}" for k, v in _count(x["source"] for x in out.values()).items()))
    return out


def _count(xs: Iterable[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d


# ----------------------------------------------------------------------------- bins
# the defaults of the final (thermal) stage of notebook 03 - 5 degC bins, a bin with fewer than 5 images
# (both eyes counted) merged into its neighbour
THERMAL_BIN_DEG = 5.0
THERMAL_MIN_IMAGES = 5
# finer bins where the images are dense (most Navcam frames are taken at -25..-5 degC): 2.5 degC there,
# THERMAL_BIN_DEG outside; the merge of small bins (THERMAL_MIN_IMAGES) is unchanged
THERMAL_FINE_RANGE_DEGC = (-25.0, -5.0)
THERMAL_FINE_BIN_DEG = 2.5
# fx, fy and k1 per temperature bin (k1 changes with temperature, and the fx / fy slopes stay different with k1
# free); k1 is the one distortion term a bin may refine
THERMAL_FREE = ("fx", "fy", "k1")
THERMAL_FREE_DISTORTION = ("k1",)
THERMAL_K1_MIN_IMAGES = 6       # an eye's bin with fewer images keeps k1 at the eye's value (fx, fy still free)


def bin_edges(T: float, bin_deg: float = THERMAL_BIN_DEG, fine_range: Optional[Sequence[float]] = None,
              fine_deg: Optional[float] = None) -> Tuple[float, float]:
    """The (lo, hi) bin of temperature ``T``: ``fine_deg`` wide inside ``fine_range`` = (a, b) (edges at a,
    a + fine_deg, ...), ``bin_deg`` wide (multiples of ``bin_deg``) outside, cut at a and b so that no bin straddles
    the fine range.  Without a fine range: ``[floor(T / bin_deg) * bin_deg, + bin_deg)`` as before."""
    if fine_range and fine_deg:
        a, b = float(fine_range[0]), float(fine_range[1])
        if a <= T < b:
            lo = a + np.floor((T - a) / fine_deg) * fine_deg
            return float(lo), float(min(lo + fine_deg, b))
        lo = np.floor(T / bin_deg) * bin_deg
        hi = lo + bin_deg
        if T < a:
            return float(lo), float(min(hi, a))
        return float(max(lo, b)), float(hi)
    lo = np.floor(T / bin_deg) * bin_deg
    return float(lo), float(lo + bin_deg)


def temperature_bins(frame_T: Dict[int, float], frame_n: Dict[int, int], bin_deg: float = 10.0,
                     min_images: int = 8, fine_range: Optional[Sequence[float]] = None,
                     fine_deg: Optional[float] = None) -> Dict[int, Tuple[float, float]]:
    """
    Bin of every frame (``frame_T``: frame id -> temperature; ``frame_n``: images per frame) as its (lo, hi)
    edges in degC.

    1. A frame's temperature is the mean of its Navcam images' camera temperatures; the frame goes to the bin
       ``[floor(T / bin_deg) * bin_deg, + bin_deg)``, so the bins are fixed multiples of ``bin_deg`` (5 degC:
       ..., -20..-15, -15..-10, ...), not centred on the data.  Only bins holding frames exist.
    2. Images are counted per bin (a stereo frame counts 2, both eyes together).  While some bin has fewer than
       ``min_images``, the smallest of them is merged with the neighbouring occupied bin (the next one below or
       above in temperature, even across empty bins) whose centre is nearer to its own; the merged bin spans both
       (e.g. -20..-15 + -15..-10 -> -20..-10; -25..-20 + -15..-10 -> -25..-10).  Repeated until every bin has
       ``min_images`` or only one is left.  A tie between two neighbours goes to the colder one.

    ``fine_range`` (a, b) and ``fine_deg``: bins ``fine_deg`` wide inside [a, b) (:func:`bin_edges`), where
    the images are dense; the merge rule then works on those bins as on the others.
    """
    if not frame_T:
        return {}
    edges = {f: bin_edges(T, bin_deg, fine_range, fine_deg) for f, T in frame_T.items()}
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
    def f(x):                                    # half-degree edges of the fine bins keep their .5
        return f"{int(round(x)):+04d}" if abs(x - round(x)) < 1e-9 else f"{x:+06.1f}"
    return f"T{f(lo)}{f(hi)}"


def _database_ids(project) -> Tuple[List[int], List[int]]:
    """The camera and rig ids of the project database (empty without one)."""
    try:
        import pycolmap
        path = getattr(project, "database", None)
        if path is None or not Path(path).is_file():
            return [], []
        db = pycolmap.Database.open(str(path))
        try:
            return [int(c.camera_id) for c in db.read_all_cameras()], [int(r.rig_id) for r in db.read_all_rigs()]
        finally:
            db.close()
    except Exception:                                                      # noqa: BLE001
        return [], []


def split_by_temperature(rec, project, temps: Dict[str, Dict[str, Any]], bin_deg: float = 10.0,
                         min_images: int = 8, free: Sequence[str] = ("fx", "fy"),
                         thermal_model: Optional[Dict[str, Dict[str, float]]] = None,
                         fine_range: Optional[Sequence[float]] = None, fine_deg: Optional[float] = None,
                         k1_min_images: int = THERMAL_K1_MIN_IMAGES):
    """
    A copy of ``rec`` with one camera per Navcam eye and temperature bin, and a copy of ``project`` whose cameras
    and database mapping include them (``fixed_params``: all but ``free``).  ``thermal_model``: {"NL": {"T0_degC",
    "fx_ppm_per_degC", "fy_ppm_per_degC", "cx_px_per_degC", "cy_px_per_degC", "k1_per_degC"}, ...} - the bins then
    start from the eye's camera moved to the bin temperature by those terms (:func:`mppp.calib.camera_at`).
    Frames without a temperature keep the original cameras.  Returns (rec, project, bins) with ``bins``: one row per
    bin camera.  ``fine_range`` / ``fine_deg`` as in :func:`temperature_bins`; with ``k1`` in ``free``, a bin camera
    with fewer than ``k1_min_images`` images keeps k1 at the eye's value.
    """
    import pycolmap
    from .project import FULL_OPENCV_NAMES, PARAM_NAMES as _PARAM_NAMES
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
    fbin = temperature_bins(frame_T, frame_n, bin_deg, min_images, fine_range, fine_deg)
    bins = sorted(set(fbin.values()))

    new = pycolmap.Reconstruction()
    for cid, cam in rec.cameras.items():
        new.add_camera(cam)
    # above every camera id the project knows too - after stage 1 of a staged run the Mastcam-Z cameras are
    # no longer in ``rec`` but come back with their frames (restore_frames): their ids must stay free
    db_cam_ids, db_rig_ids = _database_ids(project)
    next_cid = max(list(rec.cameras) + [int(v) for v in (proj.settings.get("database", {}).get("cameras") or {}).values()]
                   + db_cam_ids) + 1
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
            mkey = key if (thermal_model and key in thermal_model) else eye_key(key)   # "NL#1" -> NL
            if thermal_model and mkey in thermal_model:
                _, p, _ = calib.camera_at({"model": base.model.name, "params": p, "thermal": thermal_model[mkey]}, tm)
                p = np.asarray(p, float)
            c = pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width, height=base.height, params=p)
            new.add_camera(c)
            bin_cam[(cid, b)] = next_cid
            bkey = f"{key}_{_tag(*b)}"
            names = _PARAM_NAMES.get(base.model.name, FULL_OPENCV_NAMES)
            free_b = [n for n in free if not (n == "k1" and len(Tb) < int(k1_min_images))]
            pc = dict(proj.cameras.get(key, {}))
            pc.update({"params": p.tolist(), "group": key, "temperature_bin_degC": list(b),
                       "temperature_median_degC": tm, "n_images": len(Tb),
                       "fixed_params": [n for n in names if n not in set(free_b)], "free_params": [],
                       "source": f"{pc.get('source', key)}; temperature bin {b[0]:g}..{b[1]:g} degC"})
            proj.cameras[bkey] = pc
            db_cams[bkey] = next_cid
            rows.append({"camera": bkey, "eye": key, "camera_id": next_cid, "bin_degC": list(b),
                         "T_median_degC": tm, "T_min_degC": float(np.min(Tb)), "T_max_degC": float(np.max(Tb)),
                         "images": len(Tb), "start_fx": float(p[0]), "start_fy": float(p[1]),
                         "free": list(free_b),
                         "start": {"fx": float(p[0]), "fy": float(p[1]), "cx": float(p[2]), "cy": float(p[3]),
                                   **({"k1": float(p[names.index("k1")])} if "k1" in names else {})}})
            next_cid += 1
    proj.settings.setdefault("database", {})["cameras"] = db_cams

    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))     # noqa: E731
    # above the database's rig ids too - after stage 1 the Mastcam-Z rigs are no longer in ``rec`` and come
    # back with their frames (restore_frames), which would otherwise attach them to a temperature-bin rig
    next_rid = max(list(rec.rigs) + db_rig_ids) + 1
    rig_for: Dict[Tuple[int, Tuple[float, float]], int] = {}
    for rid, rig in rec.rigs.items():
        new.add_rig(rig)

    def _bin_cam(cid: int, b) -> int:
        """The bin camera of a Navcam eye.  v0p86: an eye without registered images in the bin (Marble Mountain,
        19 Oct: an NL frame at -42 degC whose NR image was not registered) gets a held copy of its camera too - the
        bin rig would otherwise carry the eye's original camera, which also sits on the original rig, and the COLMAP
        database allows a camera on one rig only ("SQLite error: constraint failed" in sync_database)."""
        nonlocal next_cid
        if (cid, b) in bin_cam or cid not in nav_cids or cid not in rec.cameras:
            return bin_cam.get((cid, b), cid)
        base = rec.cameras[cid]
        key = key_of[cid]
        new.add_camera(pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width, height=base.height,
                                       params=np.array(base.params, float)))
        bin_cam[(cid, b)] = next_cid
        bkey = f"{key}_{_tag(*b)}"
        names = _PARAM_NAMES.get(base.model.name, FULL_OPENCV_NAMES)
        pc = dict(proj.cameras.get(key, {}))
        pc.update({"params": list(map(float, base.params)), "group": key, "temperature_bin_degC": list(b),
                   "n_images": 0, "fixed_params": list(names), "free_params": [], "thermal_bin": True,
                   "source": f"{pc.get('source', key)}; temperature bin {b[0]:g}..{b[1]:g} degC, no registered "
                             f"image of this eye: a held copy for the bin rig"})
        proj.cameras[bkey] = pc
        db_cams[bkey] = next_cid                         # not in the bin table: it has no images to fit
        next_cid += 1
        return bin_cam[(cid, b)]

    def _rig_of(rid: int, b) -> int:
        if (rid, b) in rig_for:
            return rig_for[(rid, b)]
        nonlocal next_rid
        old = rec.rigs[rid]
        ref = old.ref_sensor_id.id
        r = pycolmap.Rig(rig_id=next_rid)
        r.add_ref_sensor(sensor(_bin_cam(ref, b)))
        for sid in old.non_ref_sensors:
            r.add_sensor(sensor(_bin_cam(sid.id, b)), old.sensor_from_rig(sid))
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


def _obs_per_camera(rec) -> Dict[int, int]:
    n: Dict[int, int] = {}
    for pt in rec.points3D.values():
        for el in pt.track.elements:
            c = rec.images[el.image_id].camera_id
            n[c] = n.get(c, 0) + 1
    return n


# ----------------------------------------------------------------------------- pipeline stage
THERMAL_MIN_BIN_IMAGES = 3      # an eye's temperature bin with fewer images is held at the thermal model


def thermal_stage(rec, project, temps: Dict[str, Dict[str, Any]], bin_deg: float = THERMAL_BIN_DEG,
                  min_images: int = THERMAL_MIN_IMAGES,
                  free: Sequence[str] = THERMAL_FREE, hold: bool = False,
                  thermal_model: Optional[Dict[str, Dict[str, float]]] = None, sigma_px: float = 0.5,
                  loss_scale: float = 2.0, max_iterations: int = 100, attitude_prior_deg: Optional[float] = None,
                  linear_solver: str = "auto", verbose: bool = True,
                  fine_range: Optional[Sequence[float]] = THERMAL_FINE_RANGE_DEGC,
                  fine_deg: Optional[float] = THERMAL_FINE_BIN_DEG):
    """
    The final stage of :func:`mppp.sfm.reconstruction.reconstruct` with ``thermal_bins_deg``: the solved
    block is split into Navcam temperature bins (:func:`split_by_temperature`) and adjusted once more with the bins'
    ``free`` parameters refined (or, with ``hold``, every bin held at its start: the eye's camera scaled to the bin
    temperature by ``thermal_model``), everything else of the cameras and the rig held, poses and points free.
    ``project`` is updated in place (bin cameras and their database ids).  Returns (rec, report).
    ``free`` may include k1 (:data:`THERMAL_FREE_DISTORTION`; default fx, fy, k1); bins are ``fine_deg`` wide inside
    ``fine_range`` (default 2.5 degC over -25..-5 degC) and ``bin_deg`` outside.
    """
    # the Navcam distortion is one set per eye for all temperatures - a temperature bin fits at most the focal
    # length, the principal point and k1
    from .project import INTRINSIC_CORE
    bad = [n for n in free if n not in INTRINSIC_CORE and n not in THERMAL_FREE_DISTORTION]
    if bad:
        raise ValueError(f"thermal_free {tuple(free)}: {bad} are distortion terms; the Navcam distortion is not fitted "
                         f"per temperature bin except {THERMAL_FREE_DISTORTION} - use a subset of "
                         f"{tuple(INTRINSIC_CORE) + THERMAL_FREE_DISTORTION}")
    from .reconstruction import bundle_adjust
    before = _stats(rec, project)
    # the bins start from the refined eye cameras, so their refined / start ratio no longer holds the block's Navcam
    # focal scale - record the eyes' refined / start (promoted model) ratio here
    pre = {}
    for key, cid in ((project.settings.get("database") or {}).get("cameras") or {}).items():
        st = (project.cameras.get(key) or {}).get("params")
        if str(key).startswith("N") and "_T" not in str(key) and st and int(cid) in rec.cameras:
            p = rec.cameras[int(cid)].params
            pre[key] = float(np.sqrt(p[0] * p[1]) / np.sqrt(float(st[0]) * float(st[1])))
    project.settings["navcam_focal_scale_pre_thermal"] = pre
    r1, p1, rows = split_by_temperature(rec, project, temps, bin_deg, min_images, free, thermal_model,
                                        fine_range=fine_range, fine_deg=fine_deg)
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
    # an eye's bin with fewer than THERMAL_MIN_BIN_IMAGES images (binning is by frame, so one eye can be left with
    # one image) is held at the thermal model: its fx, fy are hardly observed and make the Schur complement singular
    weak = [r["camera"] for r in rows if int(r.get("images") or 0) < THERMAL_MIN_BIN_IMAGES]
    held = [k for k in project.cameras if not str(k).startswith("N")] + (bin_keys if hold else weak)
    kw = {} if attitude_prior_deg is None else {"attitude_prior_deg": attitude_prior_deg}
    ba = bundle_adjust(r1, project, sigma_px=sigma_px, loss_scale=loss_scale, refine_rig=False,
                       max_iterations=max_iterations, hold_cameras=held, linear_solver=linear_solver, **kw)
    obs = _obs_per_camera(r1)
    for row in rows:
        p = r1.cameras[row["camera_id"]].params
        row.update({"fx": float(p[0]), "fy": float(p[1]), "cx": float(p[2]), "cy": float(p[3]),
                    "observations": int(obs.get(row["camera_id"], 0))})
        if "k1" in (row.get("start") or {}):
            row["k1"] = float(p[4])
        project.cameras[row["camera"]]["refined_params"] = [float(x) for x in p]
    after = _stats(r1, project)
    report = {"bin_deg": float(bin_deg), "min_images": int(min_images), "free": list(free), "held": bool(hold),
              "fine_range_degC": list(fine_range) if fine_range else None,
              "fine_bin_deg": float(fine_deg) if fine_range and fine_deg else None,
              "k1_held_bins": [r["camera"] for r in rows if "k1" in free and "k1" not in r.get("free", [])],
              "weak_bins_held": [] if hold else weak, "ba_termination": ba.get("termination"), "ba_retried": ba.get("retried"),
              "thermal_model": thermal_model, "bins": len({tuple(r["bin_degC"]) for r in rows}), "rows": rows,
              "before": before, "after": after, "initial_cost": ba["initial_cost"], "final_cost": ba["final_cost"],
              "seconds": ba["seconds"],
              "images_with_temperature": int(sum(1 for r in project.images if r["name"] in temps))}
    if verbose:
        log.info(f"thermal stage: {report['bins']} temperature bins of {bin_deg:g} degC"
              + (f" ({fine_deg:g} degC over {fine_range[0]:g}..{fine_range[1]:g})" if fine_range and fine_deg else "")
              + f" ({'held at the thermal model' if hold else ', '.join(free) + ' refined'}): median {before['median_px']:.4f} -> "
              f"{after['median_px']:.4f}, rms {before['rms_px']:.4f} -> {after['rms_px']:.4f} native px, "
              f"cost {ba['initial_cost']:.1f} -> {ba['final_cost']:.1f}")
        if weak and not hold:
            log.info(f"thermal stage: held at the thermal model (fewer than {THERMAL_MIN_BIN_IMAGES} images): "
                  f"{', '.join(weak)}")
        for line in thermal_bin_lines(rows, free, hold):
            log.info(line)
    if after["median_px"] > 1.25 * before["median_px"] + 0.01:      # never silently worse
        report["warning"] = (f"the residual median rose {before['median_px']:.4f} -> {after['median_px']:.4f} px in the "
                             f"thermal stage")
        log.warning(report["warning"])
    return r1, report


def thermal_bin_lines(rows: Sequence[Dict[str, Any]], free: Sequence[str] = ("fx", "fy"), hold: bool = False) -> List[str]:
    """The temperature-bin table of the thermal stage - per bin camera its images, temperature, fx, fy, cx,
    cy (refined values; a trailing * marks a parameter held at its start: the eye's camera moved to the bin
    temperature by the thermal model) and the change from that start."""
    fr = set() if hold else set(free)
    mark = {k: ("" if k in fr else "*") for k in ("fx", "fy", "cx", "cy")}
    k1 = any("k1" in r for r in rows)                        # k1 per bin (1e-6 units)
    out = [f"      {'camera':20s} {'images':>6s} {'T degC':>7s} {'[min, max]':>14s} "
           + " ".join(f"{k + mark[k]:>10s}" for k in ("fx", "fy", "cx", "cy"))
           + "   d fx    d fy    d cx    d cy  (px, from the start)" + ("   d k1 [1e-6]" if k1 else "")]
    for r in rows:
        st = r.get("start") or {}
        d = [r[k] - st[k] if k in st else float("nan") for k in ("fx", "fy", "cx", "cy")]
        dk = ""
        if k1 and "k1" in r and "k1" in st:
            dk = f"  {1e6 * (r['k1'] - st['k1']):+9.1f}" + ("" if "k1" in r.get("free", fr) else "*")
        out.append(f"      {r['camera']:20s} {r['images']:6d} {r['T_median_degC']:7.1f} "
                   f"{'[' + format(r['T_min_degC'], '.1f') + ', ' + format(r['T_max_degC'], '.1f') + ']':>14s} "
                   + " ".join(f"{r[k]:10.2f}" for k in ("fx", "fy", "cx", "cy"))
                   + "  " + " ".join(f"{x:+7.2f}" for x in d) + dk)
    held = [k for k in ("fx", "fy", "cx", "cy") if mark[k]]
    if held:
        out.append(f"      * held at the start ({', '.join(held)}): the eye's camera of the block, moved to the bin "
                   f"temperature by the thermal model (f ppm/degC, NL cx px/degC)")
    return out


def strip_thermal_bins(project) -> int:
    """Undo :func:`thermal_stage` (and the Mastcam-Z focus-state split, :mod:`mppp.sfm.backlash`) on a
    project: images back to their eye's (focus bin's) camera, bin cameras removed,
    the bin table dropped.  Called before the database is built and before a reconstruction, so that a rerun starts
    from one camera per eye and the stereo rig.  Returns the number of bin cameras removed."""
    n = 0
    for r in project.images:
        if "base_instrument" in r:
            r["instrument"] = r.pop("base_instrument")
    for c in project.cameras.values():              # 0.93.5: a split's base camera back to its own count and state
        pre = c.pop("pre_split", None)
        if isinstance(pre, dict):
            c["n_images"] = pre.get("n_images", c.get("n_images"))
            if pre.get("backlash_state") is None:
                c.pop("backlash_state", None)
            else:
                c["backlash_state"] = pre["backlash_state"]
    for k in [k for k, c in project.cameras.items() if c.get("thermal_bin") or c.get("state_split")]:
        # also the regular-focus-state Mastcam-Z cameras (mppp.sfm.backlash.split_by_state)
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
    """The thermal model of the project's Navcam start cameras, referred to the temperature they were scaled to
    (``SfmProject.create`` records it under ``settings["navcam_cameras"][eye]["thermal"]``): {"NL": {"T0_degC",
    "fx_ppm_per_degC", "fy_ppm_per_degC", "cx_px_per_degC", "cy_px_per_degC", "k1_per_degC"}, ...} for
    :func:`thermal_stage` (bins started at the eye's camera moved to the bin temperature).  None if the start
    cameras carry no thermal model."""
    nav = (project.settings.get("navcam_cameras") or {})
    out = {}
    for eye in ("NL", "NR"):
        th = (nav.get(eye) or {}).get("thermal")
        if th and th.get("T_median_degC") is not None:
            t = calib.thermal_terms(th)
            out[eye] = {"T0_degC": float(th["T_median_degC"]), "fx_ppm_per_degC": t["fx_ppm"],
                        "fy_ppm_per_degC": t["fy_ppm"], "cx_px_per_degC": t["cx_px"], "cy_px_per_degC": t["cy_px"],
                        "k1_per_degC": t["k1"]}
    return out or None
