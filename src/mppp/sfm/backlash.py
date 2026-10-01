"""
Mastcam-Z focus backlash states (v0p40).

At the same focus motor count the Mastcam-Z focus mechanism sits in one of two states.  In the dominant **backlash**
state the refined focal length is about 1 % above the label (CAHVOR) model; the shipped focus model
(``cmods/M2020_ZCAM034_focus_model.json``, f0 / label f0 = 1.0086 left, 1.0088 right at focus 600 in the v0p42
refit; 1.0092 / 1.0105 before) was fitted in this state.
In the **regular** state the focal length is about the label's.  Regular-state images are mostly single frames or
small mosaics.  A focus-bin camera that holds images of both states is forced to split the difference, so:

1. :func:`image_focal_fits` - after the alignment, each Mastcam-Z image's focal length is measured on its own: its
   rotation and a focal scale free, its centre, the 3-D points and the rest of the camera held (Cauchy loss);
2. :func:`classify_groups` - images of one focus setting (a *focus group*: eye, sol, sequence and focus count;
   :func:`focus_groups`) share a state; the group's ratio of fitted to label focal length is compared with the
   midpoint between the two states (1 and the focus model's ratio) - clearly below: regular, clearly above:
   backlash, within ``z_min`` standard errors: undecided (kept in the dominant backlash state);
3. :func:`split_by_state` - regular-state images move to their own camera per focus bin (``<bin>_reg``, start f =
   the median label f of its images; bins of at most ``hold_f_images`` images hold f there);
4. :func:`backlash_stage` - the block is adjusted again (Navcam cameras held) and every image re-measured.

``strip_thermal_bins`` (mppp.sfm.thermal) also undoes the split, so a rerun starts from the focus bins.

v0p53: the focus groups of each zoom are classified with that zoom's focus model (``M2020_ZCAM<zoom>_focus_model.json``;
:data:`DEFAULT_BACKLASH_RATIO` for a zoom without one); 110 mm (the end of the zoom's mechanical range) has one focus
state only and is not classified (``single``).  Both states are scaled by the block's Navcam focal scale (refined /
start: the Mastcam-Z focal lengths inherit it through the shared points; audit item 2 of 30 Sep), and the stage
uses the project's focus model (``settings["zcam_focus_model"]["file"]``; audit item 4).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

STATES = ("backlash", "regular")
REGULAR_SUFFIX = "_reg"
DEFAULT_BACKLASH_RATIO = 1.0095      # f / label f in the backlash state when no focus model is at hand
SINGLE_STATE_ZOOMS = (110,)          # v0p53: no backlash state at the end of the zoom's mechanical range


LOW_BACKLASH_FILE = "mars2020_mastcam-z_low_backlash_observations.csv"   # v0p61: in mppp/data (Christian, 2 Oct 2026)


def low_backlash_path(path: Any = None):
    """The list in use: ``path``, else ``mppp/data/mars2020_mastcam-z_low_backlash_observations.csv`` (``False``: no
    list, see :func:`load_low_backlash`)."""
    from pathlib import Path
    if path:
        return Path(path)
    return Path(__file__).resolve().parents[1] / "data" / LOW_BACKLASH_FILE


def load_low_backlash(path: Any = None) -> Dict[str, Any]:
    """
    v0p61: the Mastcam-Z observations known to be in the low-backlash (regular) focus state.  One entry per line
    (several per line separated by commas; ``#`` starts a comment): a full image name or any part of one - a stem
    (``ZL0_0121_0677679408_053RAD_N0041250ZCAM08114_034085A03``), ``0121_0677679408_053`` (sol, clock), ``ZCAM07114_034``
    (a sequence at one zoom), ``ZCAM08114`` (a sequence at every zoom) - matched as a substring of the image's file name
    (case-insensitive); ``sol:sequence`` (``363:ZCAM08394``) also works.  A trailing ``?`` (uncertain) is kept as
    an entry and counted in ``uncertain``.  A full product name also matches the same eye and spacecraft clock
    whatever its product type and version (``ZL0_0091_0675031997_228RAD_...A03`` matches ``ZLF_0091_0675031997_228EBY_...J01``,
    key ``clocks``).  Every other Mastcam-Z image is in the high-backlash state, except at
    110 mm, which has a single state.  Returns {"entries", "sol_sequences", "clocks", "uncertain", "file"}.
    """
    import re
    if path is False:                                  # no list: every Mastcam-Z image high-backlash
        return {"entries": set(), "sol_sequences": set(), "clocks": set(), "uncertain": [], "file": None}
    f = low_backlash_path(path)
    out: Dict[str, Any] = {"entries": set(), "sol_sequences": set(), "clocks": set(), "uncertain": [], "file": str(f)}
    if not f.is_file():
        return out
    for line in f.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0]
        for tok in re.split(r"[,;\t]+", line):
            tok = tok.strip().strip('"').strip()
            if not tok:
                continue
            if tok.endswith("?"):
                tok = tok.rstrip("?").strip()
                out["uncertain"].append(tok)
            m = re.fullmatch(r"(\d+):(ZCAM\d{5})", tok, re.I)
            if m:
                out["sol_sequences"].add((int(m.group(1)), m.group(2).upper()))
                continue
            for ext in (".IMG", ".PNG", ".LBL"):
                if tok.upper().endswith(ext):
                    tok = tok[: -len(ext)]
            if len(tok) >= 6:                          # a too-short fragment would match everything
                out["entries"].add(tok.upper())
            c = _clock_key(tok)
            if c:
                out["clocks"].add(c)
    return out


def _clock_key(name: str):
    """(eye, spacecraft clock) of a Mastcam-Z product name (``ZL0_0091_0675031997_...`` -> ("L", "0675031997"))."""
    import re
    m = re.match(r"Z([LR])._\d{4}_(\d{10})_", str(name).strip().upper())
    return (m.group(1), m.group(2)) if m else None


def low_backlash_fingerprint(path: Any = None) -> Optional[str]:
    """SHA-256 of the list in use (None without one), so that notebook 03 rebuilds a project when the list changes."""
    import hashlib
    if path is False:
        return None
    f = low_backlash_path(path)
    return hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None


def is_low_backlash(row: Dict[str, Any], lst: Dict[str, Any]) -> bool:
    """Whether an image row (``stem``/``name``, ``sequence``, ``sol``, ``camera_group``) is on the list (never at
    110 mm, which has one focus state)."""
    if group_zoom(row.get("camera_group", row.get("instrument"))) in SINGLE_STATE_ZOOMS:
        return False
    name = str(row.get("stem") or row.get("name") or "").upper()
    if any(e in name for e in lst["entries"]):
        return True
    c = _clock_key(name)
    if c and c in lst.get("clocks", ()):
        return True
    seq = str(row.get("sequence") or "").upper()
    try:
        sol = int(row.get("sol"))
    except (TypeError, ValueError):
        sol = None
    return (sol, seq) in lst["sol_sequences"]


def group_zoom(group: Any) -> Optional[int]:
    """``"ZL034"`` -> 34, ``"ZR110_F01234"`` -> 110; None for a name that is not a Mastcam-Z camera."""
    g = str(group)
    try:
        return int(g[2:5]) if g.startswith("Z") else None
    except ValueError:
        return None


def navcam_scale(rec, project) -> Optional[float]:
    """v0p53: the block's Navcam focal scale now - refined / start focal length (sqrt(fx fy)) averaged over the Navcam
    eye cameras with images; None without Navcam cameras."""
    db = (project.settings.get("database") or {}).get("cameras") or {}
    r = []
    for key, cid in db.items():
        if not str(key).startswith("N") or "_T" in str(key) or int(cid) not in rec.cameras:
            continue
        start = (project.cameras.get(key) or {}).get("params")
        if not start:
            continue
        p = rec.cameras[int(cid)].params
        r.append(float(np.sqrt(p[0] * p[1]) / np.sqrt(float(start[0]) * float(start[1]))))
    return float(np.mean(r)) if r else None


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def backlash_ratios(model: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """{camera group: f / label f of the backlash state} from a focus model (:func:`mppp.sfm.project.zcam_focus_model`)."""
    out = {}
    for g, m in ((model or {}).get("cameras") or {}).items():
        if m.get("f0_px") and m.get("label_f0_px"):
            out[g] = float(m["f0_px"]) / float(m["label_f0_px"])
    return out


def focus_groups(images: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Mastcam-Z image rows by focus setting: eye, sol, sequence and focus count (one focus move, one state)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in images:
        if not str(r.get("camera_group", r.get("instrument", ""))).startswith("Z"):
            continue
        fc = r.get("focus_count")
        key = f"{r.get('camera_group')}|sol{r.get('sol')}|{r.get('sequence')}|F{'na' if fc is None else int(round(float(fc)))}"
        out.setdefault(key, []).append(r)
    return out


def station_shifts(rec, project) -> Dict[str, np.ndarray]:
    """{station: median of (solved - prior) centre over its registered Navcam images}: how far the alignment moved
    the station's CAHV positions (the Mastcam-Z label positions of the station share that error)."""
    by_name = {r["name"]: r for r in project.images}
    d: Dict[str, List[np.ndarray]] = {}
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        r = by_name.get(im.name)
        if r is None or not str(r.get("instrument", "")).startswith("N") or r.get("prior_C") is None:
            continue
        T = im.cam_from_world()
        C = -np.asarray(T.rotation.matrix()).T @ np.asarray(T.translation)
        d.setdefault(r["station"], []).append(C - np.asarray(r["prior_C"], float))
    return {k: np.median(np.array(v), axis=0) for k, v in d.items()}


def image_focal_fits(rec, project, min_observations: int = 20, loss_px: float = 2.0,
                     names: Optional[Iterable[str]] = None, center_sigma_m: Optional[float] = 0.05
                     ) -> Dict[str, Dict[str, Any]]:
    """
    {image name: fit} for every registered Mastcam-Z image with at least ``min_observations`` triangulated
    observations: the focal scale s (fx, fy x (1 + s)) and a small rotation that best fit its observations with its
    points and the rest of its camera held (scipy least squares, Cauchy loss at ``loss_px`` native pixels).  The
    centre: the adjustment can trade a focal-length error for a shift along the axis (1 % at 10 m range is 10 cm, well
    inside the 1 m position prior), so with ``center_sigma_m`` the centre is free with a prior of that sigma at the
    image's label position moved by its station's Navcam shift (:func:`station_shifts`; the solved centre where the
    station has no Navcam image, ``center_source``); None holds the solved centre.  Each fit: ``f_px`` (the fitted mean of fx, fy), ``f_camera_px``, ``label_f_px``, ``ratio_label``
    (fitted f / label f), ``sd_ratio``, ``observations``, ``rms_px`` (full-frame, after the fit).
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    by_name = {r["name"]: r for r in project.images}
    want = set(names) if names is not None else None
    shifts = station_shifts(rec, project) if center_sigma_m else {}
    out: Dict[str, Dict[str, Any]] = {}
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        r = by_name.get(im.name)
        if r is None or not str(r.get("instrument", "")).startswith("Z") or (want is not None and im.name not in want):
            continue
        pts = [(q.xy, q.point3D_id) for q in im.points2D if q.has_point3D()]
        if len(pts) < min_observations:
            continue
        xy = np.array([p[0] for p in pts], float)
        X = np.array([rec.points3D[p[1]].xyz for p in pts], float)
        cam = rec.cameras[im.camera_id]
        T = im.cam_from_world()
        R0 = np.asarray(T.rotation.matrix(), float)
        C = -R0.T @ np.asarray(T.translation, float)
        c0 = np.asarray(cam.params[2:4], float)
        s_nat = float(r.get("downsample_scale", 1.0))
        fs = loss_px / s_nat                                  # Cauchy scale in full-frame pixels
        free_c = bool(center_sigma_m) and r.get("station") in shifts and r.get("prior_C") is not None
        Cp = (np.asarray(r["prior_C"], float) + shifts[r["station"]]) if free_c else C
        sig_px = 0.5 / s_nat                                  # nominal keypoint sigma: weighs the centre prior
        npar = 7 if free_c else 4

        def proj_res(p):
            R = Rotation.from_rotvec(p[:3]).as_matrix() @ R0
            Cc = Cp + p[4:7] if free_c else C
            xc = (X - Cc) @ R.T
            uv = np.asarray(cam.img_from_cam(xc), float)
            uv = c0 + (1.0 + p[3]) * (uv - c0)
            d = uv - xy
            d[~np.isfinite(d)] = 1e3
            return d.ravel()

        def res(p):
            d = proj_res(p)
            if free_c:
                return np.r_[d, p[4:7] / float(center_sigma_m) * sig_px]
            return d

        x0 = np.zeros(npar)
        sol = least_squares(res, x0, loss="cauchy", f_scale=fs, x_scale=[1e-3] * 4 + [1e-2] * (npar - 4))
        J = sol.jac
        nres = 2 * len(pts)
        rr = sol.fun[:nres].reshape(-1, 2)
        e = np.linalg.norm(rr, axis=1)
        w = np.r_[np.repeat(1.0 / (1.0 + (e / fs) ** 2), 2), np.ones(len(sol.fun) - nres)]   # Cauchy weights
        Jw = J * w[:, None] ** 0.5
        dof = max(nres - 4, 1)
        s2 = float(np.sum(w[:nres] * sol.fun[:nres] ** 2) / dof)
        try:
            cov = np.linalg.inv(Jw.T @ Jw) * s2
            sd = float(np.sqrt(max(cov[3, 3], 0.0)))
        except np.linalg.LinAlgError:
            sd = float("nan")
        f_cam = float(0.5 * (cam.params[0] + cam.params[1]))
        f_fit = f_cam * (1.0 + float(sol.x[3]))
        lf = r.get("label_f_px")
        out[im.name] = {"image_id": int(iid), "camera": r["instrument"], "group": r.get("camera_group"),
                        "eye": r.get("eye"), "sol": r.get("sol"), "sequence": r.get("sequence"),
                        "focus_count": r.get("focus_count"), "observations": len(e),
                        "f_camera_px": f_cam, "f_px": f_fit, "scale": float(sol.x[3]),
                        "sd_scale": sd, "label_f_px": lf,
                        "ratio_label": (f_fit / float(lf)) if lf else None,
                        "sd_ratio": (sd * f_cam / float(lf)) if lf else None,
                        "rotation_mdeg": float(np.degrees(np.linalg.norm(sol.x[:3])) * 1e3),
                        "center_source": "navcam station" if free_c else "solved",
                        "center_offset_m": float(np.linalg.norm(Cp + sol.x[4:7] - C)) if free_c else 0.0,
                        "rms_px": float(np.sqrt(np.mean(e ** 2)))}
    return out


def classify_groups(project, fits: Dict[str, Dict[str, Any]], model: Optional[Dict[str, Any]] = None,
                    z_min: float = 3.0, floor: float = 0.001, tolerance: float = 0.006,
                    nav_scale: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    One row per focus group (:func:`focus_groups`): the observation-weighted median of its images' ratio of fitted to
    label focal length, its standard error (the images' formal errors, at least ``floor`` in the ratio, i.e. 0.1 %,
    for the scatter between images and the held centre) and the state: ``regular`` below the midpoint between 1
    and the group's backlash ratio (:func:`backlash_ratios`) by more than ``z_min`` standard errors, ``backlash``
    above it by as much, otherwise ``undecided`` (treated as backlash, the dominant state).  Groups without a fit
    (too few observations, no label f) are ``undecided`` too, and so are groups whose ratio lies more than
    ``tolerance`` (0.6 %) outside the two states (below 1 or above the backlash ratio): a failed bin rather than a
    focus state (``implausible``; at Airey Hill and Three Forks a few one- and two-image bins at 0.96-0.98).
    v0p53: ``nav_scale`` (the block's Navcam focal scale, :func:`navcam_scale`) multiplies both states (1 and the
    backlash ratio); groups of a :data:`SINGLE_STATE_ZOOMS` zoom are ``single`` (not classified, not split).
    """
    ratios = backlash_ratios(model)
    ns = float(nav_scale) if nav_scale else 1.0
    rows = []
    for key, members in sorted(focus_groups(project.images).items()):
        g = members[0].get("camera_group")
        if group_zoom(g) in SINGLE_STATE_ZOOMS:
            rows.append({"group_key": key, "camera_group": g, "eye": members[0].get("eye"), "sol": members[0].get("sol"),
                         "sequence": members[0].get("sequence"), "focus_count": members[0].get("focus_count"),
                         "images": len(members), "fitted": 0, "backlash_ratio": None, "threshold_ratio": None,
                         "names": [m["name"] for m in members], "cameras": sorted({m["instrument"] for m in members}),
                         "ratio": None, "sd": None, "z": None, "state": "single", "decided": False})
            continue
        rb = ratios.get(g, DEFAULT_BACKLASH_RATIO) * ns
        mid = 0.5 * (ns + rb)
        fs = [fits[m["name"]] for m in members if m["name"] in fits and fits[m["name"]].get("ratio_label")]
        row = {"group_key": key, "camera_group": g, "eye": members[0].get("eye"), "sol": members[0].get("sol"),
               "sequence": members[0].get("sequence"), "focus_count": members[0].get("focus_count"),
               "images": len(members), "fitted": len(fs), "backlash_ratio": rb, "threshold_ratio": mid,
               "nav_scale": ns,
               "names": [m["name"] for m in members], "cameras": sorted({m["instrument"] for m in members})}
        if not fs:
            row.update({"ratio": None, "sd": None, "z": None, "state": "undecided", "decided": False})
            rows.append(row)
            continue
        x = np.array([f["ratio_label"] for f in fs], float)
        w = np.array([f["observations"] for f in fs], float)
        sd_i = np.array([f["sd_ratio"] if f.get("sd_ratio") and np.isfinite(f["sd_ratio"]) else floor for f in fs])
        sd_i = np.maximum(sd_i, floor)
        med = _wmedian(x, w)
        sd = float(1.0 / np.sqrt(np.sum(1.0 / sd_i ** 2)))
        if len(x) > 2:                                       # the images' scatter, when there is enough of it
            sd = max(sd, float(1.4826 * np.median(np.abs(x - med)) / np.sqrt(len(x))))
        z = (med - mid) / sd
        implausible = med < ns - tolerance or med > rb + tolerance
        state = "undecided" if implausible else ("regular" if z < -z_min else ("backlash" if z > z_min else "undecided"))
        row.update({"ratio": med, "sd": sd, "z": float(z), "state": state, "decided": state != "undecided",
                    "implausible": bool(implausible),
                    "ratio_min": float(x.min()), "ratio_max": float(x.max()),
                    "offset_from_label_pct": 100.0 * (med - 1.0)})
        rows.append(row)
    return rows


def image_states(groups: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """{image name: "regular" | "backlash" | "single"} (undecided groups count as backlash; v0p53: "single" for the
    one-state zooms)."""
    out = {}
    for g in groups:
        for n in g["names"]:
            out[n] = g["state"] if g["state"] in ("regular", "single") else "backlash"
    return out


def split_by_state(rec, project, regular: Iterable[str], hold_f_images: int = 0):
    """
    A copy of ``rec`` in which the ``regular`` images (names) of every Mastcam-Z camera move to a new camera
    ``<camera>_reg`` (start f = the median label f of those images, the camera's fy/fx kept; f held when it has at
    most ``hold_f_images`` images), with their frames on new rigs of the new cameras.  ``project`` is updated in place
    (cameras, database ids, each moved image's ``instrument``, with the old one in ``base_instrument`` and its
    ``backlash_state``).  Returns (rec, rows) with one row per new camera.
    """
    import pycolmap
    from .project import FULL_OPENCV_NAMES
    regular = set(regular)
    by_name = {r["name"]: r for r in project.images}
    db = project.settings.setdefault("database", {}).setdefault("cameras", {})
    key_of = {int(v): k for k, v in db.items()}
    moved: Dict[int, List[int]] = {}                             # old camera id -> image ids
    for iid, im in rec.images.items():
        if im.name in regular and str(key_of.get(int(im.camera_id), "")).startswith("Z"):
            moved.setdefault(int(im.camera_id), []).append(int(iid))
    new = pycolmap.Reconstruction()
    for cam in rec.cameras.values():
        new.add_camera(cam)
    next_cid = max(list(rec.cameras) + [int(v) for v in db.values()]) + 1
    cam_map: Dict[int, int] = {}
    rows = []
    for cid, iids in sorted(moved.items()):
        old_key = key_of[cid]
        base = rec.cameras[cid]
        p = np.array(base.params, float)
        lf = [by_name[rec.images[i].name].get("label_f_px") for i in iids]
        lf = [float(v) for v in lf if v]
        fm = 0.5 * (p[0] + p[1])
        if lf:
            p[0:2] *= float(np.median(lf)) / fm
        new.add_camera(pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width, height=base.height,
                                       params=p))
        key = f"{old_key}{REGULAR_SUFFIX}"
        pc = copy.deepcopy(project.cameras.get(old_key, {}))
        fixed = list(pc.get("fixed_params") or [])
        held_f = len(iids) <= int(hold_f_images or 0)
        if held_f:
            fixed = [n for n in FULL_OPENCV_NAMES if n in set(fixed) | {"fx", "fy"}]
        pc.update({"params": p.tolist(), "fixed_params": fixed, "n_images": len(iids), "backlash_state": "regular",
                   "state_split": True, "base_camera": old_key,
                   "source": f"{pc.get('source', old_key)}; regular focus state (not backlash): f from the median "
                             f"label f of its {len(iids)} images" + ("; f held" if held_f else "")})
        project.cameras[key] = pc
        if old_key in project.cameras:
            project.cameras[old_key]["backlash_state"] = "backlash"
            project.cameras[old_key]["n_images"] = max(int(project.cameras[old_key].get("n_images", 0)) - len(iids), 0)
        db[key] = next_cid
        cam_map[cid] = next_cid
        rows.append({"camera": key, "base_camera": old_key, "camera_id": next_cid, "images": len(iids),
                     "start_f_px": float(0.5 * (p[0] + p[1])), "base_f_px": float(fm),
                     "f_held": held_f})
        next_cid += 1
    sensor = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))     # noqa: E731
    for rig in rec.rigs.values():
        new.add_rig(rig)
    next_rid = max(rec.rigs) + 1
    rig_for: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    moved_ids = {i for v in moved.values() for i in v}
    for fid, fr in rec.frames.items():
        rid = fr.rig_id
        ids = [d.id for d in fr.data_ids]
        remap = tuple(sorted(i for i in ids if i in moved_ids))
        if remap:
            old = rec.rigs[rid]
            mapping = tuple(sorted((int(d.sensor_id.id), cam_map[int(d.sensor_id.id)]) for d in fr.data_ids
                                   if d.id in moved_ids))
            if (rid, mapping) not in rig_for:
                m = dict(mapping)
                r = pycolmap.Rig(rig_id=next_rid)
                ref = old.ref_sensor_id.id
                r.add_ref_sensor(sensor(m.get(ref, ref)))
                for sid in old.non_ref_sensors:
                    r.add_sensor(sensor(m.get(sid.id, sid.id)), old.sensor_from_rig(sid))
                new.add_rig(r)
                rig_for[(rid, mapping)] = next_rid
                next_rid += 1
            rid = rig_for[(rid, mapping)]
        nf = pycolmap.Frame(frame_id=fid, rig_id=rid)
        for d in fr.data_ids:
            sid = int(d.sensor_id.id)
            nf.add_data_id(pycolmap.data_t(sensor_id=sensor(cam_map[sid] if d.id in moved_ids else sid), id=d.id))
        if fr.has_pose:
            nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
    new_key = {v: k for k, v in db.items()}
    for iid, im in rec.images.items():
        kps = np.array([q.xy for q in im.points2D], float).reshape(-1, 2)
        cid = cam_map[int(im.camera_id)] if int(iid) in moved_ids else int(im.camera_id)
        ni = pycolmap.Image(name=im.name, keypoints=kps, camera_id=cid, image_id=iid)
        ni.frame_id = im.frame_id
        new.add_image(ni)
        r = by_name.get(im.name)
        if r is not None and int(iid) in moved_ids:
            r.setdefault("base_instrument", r["instrument"])
            r["instrument"] = new_key[cid]
            r["backlash_state"] = "regular"
        elif r is not None and str(r.get("instrument", "")).startswith("Z"):
            r["backlash_state"] = "backlash"
    for fid in rec.reg_frame_ids():
        if new.frames[fid].has_pose:
            new.register_frame(fid)
    for pt in rec.points3D.values():
        tr = pycolmap.Track()
        for el in pt.track.elements:
            tr.add_element(el.image_id, el.point2D_idx)
        new.add_point3D(pt.xyz, tr, pt.color)
    return new, rows


def _state_summary(fits: Dict[str, Dict[str, Any]], states: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for s in STATES + ("single",):
        x = [f["ratio_label"] for n, f in fits.items() if f.get("ratio_label") and states.get(n, "backlash") == s]
        out[s] = {"images": len(x), "ratio_median": float(np.median(x)) if x else None,
                  "ratio_p10": float(np.percentile(x, 10)) if x else None,
                  "ratio_p90": float(np.percentile(x, 90)) if x else None}
    return out


def backlash_stage(rec, project, model: Optional[Dict[str, Any]] = None, mode: str = "split", sigma_px: float = 0.5,
                   loss_scale: float = 2.0, max_iterations: int = 100, attitude_prior_deg: Optional[float] = None,
                   linear_solver: str = "auto", refine_tangential: Any = True, hold_f_images: int = 0,
                   z_min: float = 3.0, min_observations: int = 20, verbose: bool = True):
    """
    Classify the Mastcam-Z focus groups of a solved block into the backlash and the regular state and (``mode``
    ``"split"``) give the regular images their own cameras (:func:`split_by_state`) and adjust again with the Navcam
    cameras held; ``"report"`` classifies only.  ``model``: the focus model (backlash ratios; default: the project's,
    ``settings["zcam_focus_model"]["file"]``, else the shipped ones of every zoom).  ``project`` is updated in place.  Returns (rec, report) with the groups, the per-image fits before (and
    after) the split and the new cameras.
    """
    from .project import zcam_focus_model
    from .reconstruction import bundle_adjust
    if mode not in ("split", "report", "list"):
        raise ValueError("zcam_backlash must be 'list', 'split', 'report' or None")
    if mode == "list":            # v0p61: the states come from the list (bins made at the start); classify to compare
        mode = "report"
    if model is None:
        try:
            f = (project.settings.get("zcam_focus_model") or {}).get("file")
            model = zcam_focus_model(model_file=f)
        except Exception:                                                # noqa: BLE001
            model = None
    ns = navcam_scale(rec, project)
    fits = image_focal_fits(rec, project, min_observations=min_observations, loss_px=loss_scale)
    groups = classify_groups(project, fits, model, z_min=z_min, nav_scale=ns)
    states = image_states(groups)
    report: Dict[str, Any] = {"mode": mode, "z_min": z_min, "backlash_ratios": backlash_ratios(model), "nav_scale": ns,
                              "groups": groups, "fits_before": fits, "before": _state_summary(fits, states),
                              "cameras": [], "after": None, "fits_after": None}
    n = {s: sum(1 for g in groups if g["state"] == s) for s in ("backlash", "regular", "undecided", "single")}
    report["group_counts"] = n
    report["implausible_groups"] = [g["group_key"] for g in groups if g.get("implausible")]
    regular = [nm for nm, s in states.items() if s == "regular"]
    if verbose:
        b = report["before"]
        print(f"[sfm] Mastcam-Z focus states: {len(groups)} focus groups - {n['backlash']} backlash, {n['regular']} "
              f"regular, {n['undecided']} undecided (kept as backlash), {n['single']} single-state (110 mm); "
              f"Navcam scale {(ns or 1.0):.5f}; fitted f / label f median "
              f"{(b['backlash']['ratio_median'] or float('nan')):.4f} (backlash), "
              f"{(b['regular']['ratio_median'] or float('nan')):.4f} (regular)", flush=True)
    if mode == "report" or not regular:
        return rec, report
    rec2, rows = split_by_state(rec, project, regular, hold_f_images=hold_f_images)
    report["cameras"] = rows
    held = [k for k in project.cameras if str(k).startswith("N")]
    kw = {} if attitude_prior_deg is None else {"attitude_prior_deg": attitude_prior_deg}
    ba = bundle_adjust(rec2, project, sigma_px=sigma_px, loss_scale=loss_scale, refine_rig=False,
                       max_iterations=max_iterations, hold_cameras=held, linear_solver=linear_solver,
                       refine_tangential=refine_tangential, **kw)
    report["ba"] = {k: ba[k] for k in ("initial_cost", "final_cost", "iterations") if k in ba}
    fits2 = image_focal_fits(rec2, project, min_observations=min_observations, loss_px=loss_scale)
    report["fits_after"] = fits2
    report["after"] = _state_summary(fits2, states)
    for row in rows:
        p = rec2.cameras[row["camera_id"]].params
        row["f_refined_px"] = float(0.5 * (p[0] + p[1]))
        project.cameras[row["camera"]]["refined_params"] = [float(x) for x in p]
    if verbose:
        a = report["after"]
        print(f"[sfm] {len(rows)} regular-state cameras ({len(regular)} images); after the adjustment fitted f / "
              f"label f median {(a['backlash']['ratio_median'] or float('nan')):.4f} (backlash), "
              f"{(a['regular']['ratio_median'] or float('nan')):.4f} (regular)", flush=True)
    return rec2, report


def strip_state_split(project) -> int:
    """Undo :func:`split_by_state` on a project: regular-state images back to their focus-bin camera, the ``_reg``
    cameras removed.  Returns the number of cameras removed."""
    n = 0
    split = {k for k, c in project.cameras.items() if c.get("state_split")}
    for r in project.images:
        if r.get("instrument") in split and "base_instrument" in r:
            r["instrument"] = r.pop("base_instrument")
    for k in split:
        project.cameras.pop(k)
        n += 1
    db = project.settings.get("database", {}).get("cameras")
    if isinstance(db, dict):
        for k in [k for k in db if k not in project.cameras]:
            db.pop(k)
    return n
