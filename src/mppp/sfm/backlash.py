"""
Mastcam-Z focus backlash states.

At the same focus motor count the Mastcam-Z focus mechanism sits in one of two states.  In the dominant
**backlash** state the refined focal length is about 1 % above the label (CAHVOR) model; the promoted focus model
(``cmods/M2020_ZCAM034_focus_model.json``) is in this state, and every image is taken to be in it unless the
low-backlash observation list (``mppp/data/mars2020_mastcam-z_low_backlash_observations.csv``), the found list
next to the WORK folders, or its own focal fit says otherwise (:func:`mppp.calib.focus_state`).  In the
**regular** state the focal length is about the label's.  Regular-state images are mostly single frames or small
mosaics.  A focus-bin camera that holds images of both states is forced to split the difference, so:

1. :func:`image_focal_fits` - after the alignment, each Mastcam-Z image's focal length is measured on its own: its
   rotation and a focal scale free, its centre, the 3-D points and the rest of the camera held (Cauchy loss);
2. :func:`classify_groups` - images of one focus setting (a *focus group*: eye, sol, sequence and focus count;
   :func:`focus_groups`) share a state; the group's ratio of fitted to label focal length is compared with the
   midpoint between the two states (1 and the measured backlash ratio) - clearly below: regular, clearly above:
   backlash, within ``z_min`` standard errors: undecided (kept in the dominant backlash state); one state per
   sequence (:func:`apply_sequence_states`);
3. :func:`split_by_state` - regular-state images move to their own camera per focus bin (``<bin>_reg``, start f =
   the median label f of its images; bins of at most ``hold_f_images`` images hold f there);
4. :func:`backlash_stage` - the block is adjusted again (Navcam cameras held) and every image re-measured; newly
   found regular groups go to the found list (:func:`append_found`).

``strip_thermal_bins`` (mppp.sfm.thermal) also undoes the split, so a rerun starts from the focus bins.  The focus
groups of each zoom are classified with that zoom's focus model (:data:`DEFAULT_BACKLASH_RATIO` for a zoom without
one); 110 mm (the end of the zoom's mechanical range) has one focus state only and is not classified (``single``).
Both states are scaled by the block's Navcam focal scale (refined / start: the Mastcam-Z focal lengths inherit it
through the shared points).
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .. import calib
from ..calib.fit import huber_fit, weighted_median as _wmedian

log = logging.getLogger("mppp.sfm")
STATES = calib.ZCAM_STATES
REGULAR_SUFFIX = "_reg"
DEFAULT_BACKLASH_RATIO = 1.0095      # f / label f in the backlash state when no focus model is at hand
SINGLE_STATE_ZOOMS = calib.ZCAM_SINGLE_STATE_ZOOMS   # (110,): no backlash state at the end of the zoom's range


LOW_BACKLASH_FILE = "mars2020_mastcam-z_low_backlash_observations.csv"   # in mppp/data


def low_backlash_path(path: Any = None):
    """The list in use: ``path``, else ``mppp/data/mars2020_mastcam-z_low_backlash_observations.csv`` (``False``: no
    list, see :func:`load_low_backlash`)."""
    from pathlib import Path
    if path:
        return Path(path)
    return Path(__file__).resolve().parents[1] / "data" / LOW_BACKLASH_FILE


FOUND_FILE = "zcam_low_backlash_found.csv"      # next to the WORK folders (SCAPES_ROOT), filled by notebook 03


def append_found(path: Any, groups: Sequence[Dict[str, Any]], site: str = "", when: Optional[str] = None) -> int:
    """
    add the images of newly classified low-backlash (regular) focus groups to the found list ``path``
    (``<SCAPES_ROOT>/zcam_low_backlash_found.csv``, read with the shipped list by :func:`load_low_backlash`), one
    stem per line with a comment: site, sol, sequence, focus, ratio, z, date.  Stems already in the file are skipped.
    Returns the number of lines added.
    """
    import datetime
    from pathlib import Path
    f = Path(path)
    have = set()
    if f.is_file():
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            tok = line.split("#", 1)[0].strip()
            if tok:
                have.add(tok.upper())
    when = when or datetime.date.today().isoformat()
    lines = []
    for g in groups:
        for nm in g.get("names") or []:
            stem = str(nm).rsplit(".", 1)[0]
            if stem.upper() in have:
                continue
            have.add(stem.upper())
            ratio = g.get("ratio") if g.get("ratio") is not None else float("nan")
            z = g.get("z") if g.get("z") is not None else (g.get("sequence_z") if g.get("sequence_z") is not None else float("nan"))
            how = "bin camera" if g.get("fit_source") == "bin camera" else "notebook 03 classification"
            if g.get("sequence_state") == "regular" and g.get("own_state") != "regular":
                how += ", by its sequence"                       # the sequence rule decided
            lines.append(f"{stem}  # {site} sol {g.get('sol')} {g.get('sequence')} focus {g.get('focus_count')} "
                         f"ratio {ratio:.4f} z {z:+.1f} ({how}, {when})")
    if lines:
        f.parent.mkdir(parents=True, exist_ok=True)
        new = not f.is_file()
        with f.open("a", encoding="utf-8") as fh:
            if new:
                fh.write("# Mastcam-Z observations classified low-backlash (regular focus state) by notebook 03; "
                         "read together with mppp/data/mars2020_mastcam-z_low_backlash_observations.csv\n")
            fh.write("\n".join(lines) + "\n")
    return len(lines)


def load_low_backlash(path: Any = None, found: Any = None) -> Dict[str, Any]:
    """
    the Mastcam-Z observations known to be in the low-backlash (regular) focus state.  One entry per line
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
    out: Dict[str, Any] = {"entries": set(), "sol_sequences": set(), "clocks": set(), "uncertain": [], "file": str(f),
                           "found_file": None, "found": 0}
    texts = [f.read_text(encoding="utf-8-sig")] if f.is_file() else []
    if found:                                          # the observations notebook 03 classified (found list)
        from pathlib import Path as _P
        ff = _P(found)
        out["found_file"] = str(ff)
        if ff.is_file():
            t = ff.read_text(encoding="utf-8-sig")
            out["found"] = sum(1 for ln in t.splitlines() if ln.split("#", 1)[0].strip())
            texts.append(t)
    if not texts:
        return out
    for line in "\n".join(texts).splitlines():
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


def low_backlash_fingerprint(path: Any = None, found: Any = None) -> Optional[str]:
    """SHA-256 of the list in use (None without one), so that notebook 03 rebuilds a project when the list changes
    (and the found list, ``found``)."""
    import hashlib
    from pathlib import Path
    if path is False:
        return None
    f = low_backlash_path(path)
    if not f.is_file():
        return None
    h = hashlib.sha256(f.read_bytes())
    if found and Path(found).is_file():
        h.update(Path(found).read_bytes())
    return h.hexdigest()


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
    """the block's Navcam focal scale now - refined / start focal length (sqrt(fx fy)) averaged over the Navcam
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


REGULAR_MARGIN = 0.004       # groups this far above the regular state (ratio 1) enter the backlash-ratio fit


def measured_backlash_lines(rows: Sequence[Dict[str, Any]], nav_scale: float = 1.0,
                            model_ratios: Optional[Dict[str, float]] = None, reference_focus: float = 600.0,
                            min_groups_line: int = 5, min_groups_const: int = 3, min_span: float = 300.0,
                            clip: Tuple[float, float] = (0.005, 0.016)) -> Dict[str, Dict[str, Any]]:
    """
    the backlash state's ratio of fitted to label focal length, measured on the block, per camera group, as a
    line in focus count, ratio = a + b (focus - ``reference_focus``).  The ratio is not one number: the refined focal
    lengths and the labels have different slopes in focus count (ZR034: about 0.065 against 0.041 px/count, so the
    ratio rises by about 0.45 % per 1000 counts), and a start model's f0 / label f0
    is not what the blocks measure (about 1.0095 at focus 600).  Fitted (Huber, weights = fitted images) to the groups
    at least :data:`REGULAR_MARGIN` above the regular state (``nav_scale``) and not more than 2 % above it, own
    ratios from the images not on the low-backlash list (``ratio_fit``).  With fewer than ``min_groups_line`` groups or
    a focus span below ``min_span`` the line is a constant (weighted median, at least ``min_groups_const`` groups);
    below that the model's ratio (``model_ratios``, else :data:`DEFAULT_BACKLASH_RATIO`) times ``nav_scale``.  The
    line is clipped to ``nav_scale`` + ``clip``.  Returns {group: {"a", "b", "n", "source", "reference_focus"}}.
    """
    ns = float(nav_scale or 1.0)
    by: Dict[str, List[Tuple[float, float, float]]] = {}
    for r in rows:
        x = r.get("ratio_fit")
        if x is None or r.get("focus_count") is None or r.get("state") == "single":
            continue
        if ns + REGULAR_MARGIN <= float(x) <= ns + 0.02:
            by.setdefault(str(r.get("camera_group")), []).append((float(r["focus_count"]), float(x),
                                                                  float(max(r.get("fitted_unlisted") or 1, 1))))
    groups = {str(r.get("camera_group")) for r in rows if r.get("state") != "single"}
    out: Dict[str, Dict[str, Any]] = {}
    for g in sorted(groups):
        pts = by.get(g, [])
        rb_model = float((model_ratios or {}).get(g, DEFAULT_BACKLASH_RATIO)) * ns
        if len(pts) >= min_groups_line and np.ptp([p[0] for p in pts]) >= min_span:
            F = np.array([p[0] for p in pts]) - reference_focus
            y = np.array([p[1] for p in pts])
            w = np.array([p[2] for p in pts])
            b, _, _, _ = huber_fit(np.c_[np.ones(len(F)), F], y, w, floor=2e-4, iterations=20)
            out[g] = {"a": float(b[0]), "b": float(b[1]), "n": len(pts), "source": "measured line"}
        elif len(pts) >= min_groups_const:
            out[g] = {"a": _wmedian(np.array([p[1] for p in pts]), np.array([p[2] for p in pts])), "b": 0.0,
                      "n": len(pts), "source": "measured constant"}
        elif abs(rb_model / ns - DEFAULT_BACKLASH_RATIO) <= 0.003:
            out[g] = {"a": rb_model, "b": 0.0, "n": len(pts), "source": "focus model"}
        else:                     # the model's f0 / label f0 is not the backlash ratio
            out[g] = {"a": DEFAULT_BACKLASH_RATIO * ns, "b": 0.0, "n": len(pts),
                      "source": f"default (the focus model's {rb_model / ns:.4f} is off)"}
        out[g].update({"reference_focus": reference_focus, "clip": [ns + clip[0], ns + clip[1]], "nav_scale": ns})
    return out


def backlash_ratio_at(line: Dict[str, Any], focus: Optional[float]) -> float:
    """The backlash ratio of a :func:`measured_backlash_lines` line at ``focus`` (clipped)."""
    x = float(line["a"]) + float(line.get("b") or 0.0) * ((float(focus) if focus is not None else line["reference_focus"])
                                                         - float(line["reference_focus"]))
    lo, hi = line.get("clip") or (-np.inf, np.inf)
    return float(min(max(x, lo), hi))


def apply_sequence_states(rows: List[Dict[str, Any]], z_min: float = 3.0) -> Dict[str, Dict[str, Any]]:
    """
    the images of one sequence (sol and sequence, both eyes) are in one focus state;
    the exceptions are on the low-backlash list.  Every row (:func:`classify_groups`) keeps its own verdict as
    ``own_state`` / ``own_z``; the sequence's verdict pools the groups' deviations from their midpoints (inverse
    variance, own ratios from the images not on the list): ``sequence_z`` below ``-z_min`` regular, above ``z_min``
    backlash, otherwise none.  A decided sequence gives its state to every group (undecided ones included); a group
    decided the other way on its own is a ``conflict`` (reported, not followed).  Images on the list stay regular
    whatever the sequence says (:func:`image_states`).  Undecided sequences leave the groups' own verdicts.
    Returns {"sol|sequence": {"z", "state", "groups", "conflicts"}}.
    """
    seqs: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("state") == "single":
            continue
        seqs.setdefault(f"{r.get('sol')}|{r.get('sequence')}", []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for key, rs in seqs.items():
        num = den = 0.0
        for r in rs:
            if r.get("ratio_fit") is None or r.get("implausible") or not r.get("sd_fit"):
                continue
            d = float(r["ratio_fit"]) - float(r["threshold_ratio"])
            v = float(r["sd_fit"]) ** 2
            num += d / v
            den += 1.0 / v
        z = num / np.sqrt(den) if den > 0 else None
        st = None if z is None else ("regular" if z < -z_min else ("backlash" if z > z_min else None))
        conflicts = []
        for r in rs:
            r["own_state"], r["own_z"] = r.get("state"), r.get("z")
            r["sequence_z"], r["sequence_state"] = (float(z) if z is not None else None), st
            if st is None or r.get("implausible"):       # 0.93.5: a failed bin (neither state) is not voted in
                continue
            if r.get("state") in STATES and r["state"] != st and not r.get("all_listed"):
                r["conflict"] = True
                conflicts.append(r["group_key"])
            r["state"], r["decided"] = st, True
        out[key] = {"z": (float(z) if z is not None else None), "state": st, "groups": len(rs), "conflicts": conflicts}
    return out


BIN_FALLBACK_SD = 0.001      # the ratio sd given to a group classified from its bin camera (no per-image fit):
#                              a bin of thousands of observations is formally far better, 0.1 % is the systematic part
#                              (the block's Navcam scale, the f / principal-point trade); the states are 0.95 % apart


def bin_focal_fallback(rec, project, min_observations: int = 200) -> Dict[str, Dict[str, Any]]:
    """
    {camera key: {"f_px", "observations"}} of every Mastcam-Z focus-bin camera with at least
    ``min_observations`` triangulated observations in ``rec`` - the fallback that classifies a focus group none of
    whose images got a per-image fit (:func:`image_focal_fits`: fewer than 20 observations each, or registered only
    through the bin): the bin's refined f against the group's label f.
    """
    from .zcam import _observations_per_camera
    db = (project.settings.get("database") or {}).get("cameras") or {}
    obs = _observations_per_camera(rec)
    out: Dict[str, Dict[str, Any]] = {}
    for key, cid in db.items():
        if not str(key).startswith("Z") or int(cid) not in rec.cameras:
            continue
        n = int(obs.get(int(cid), 0))
        if n < min_observations:
            continue
        p = rec.cameras[int(cid)].params
        out[str(key)] = {"f_px": float(0.5 * (p[0] + p[1])), "observations": n}
    return out


def classify_groups(project, fits: Dict[str, Dict[str, Any]], model: Optional[Dict[str, Any]] = None,
                    z_min: float = 3.0, floor: float = 0.001, tolerance: float = 0.006,
                    nav_scale: Optional[float] = None, known: Any = None, sequence_rule: bool = True,
                    measured_ratio: bool = True, report: Optional[Dict[str, Any]] = None,
                    bin_fallback: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    One row per focus group (:func:`focus_groups`): the observation-weighted median of its images' ratio of fitted to
    label focal length, its standard error (the images' formal errors, at least ``floor`` in the ratio, i.e. 0.1 %,
    for the scatter between images and the held centre) and the state: ``regular`` below the midpoint between the
    regular state (ratio 1) and the backlash state by more than ``z_min`` standard errors, ``backlash`` above it by as
    much, otherwise ``undecided`` (treated as backlash, the dominant state).  Groups without a fit (too few
    observations, no label f) are ``undecided`` too, and so are groups whose ratio lies more than ``tolerance`` (0.6 %)
    outside the two states: a failed bin rather than a focus state (``implausible``).
    ``nav_scale`` (the block's Navcam focal scale, :func:`navcam_scale`) multiplies both states; groups of a
    :data:`SINGLE_STATE_ZOOMS` zoom are ``single`` (not classified, not split).
    (1) the backlash ratio is measured on the block as a line in focus count
    (:func:`measured_backlash_lines`, ``measured_ratio``; else the focus model's f0 / label f0), so that the
    midpoint follows the focus count; (2) ``sequence_rule``: one state per sequence (:func:`apply_sequence_states`),
    the images on the low-backlash list (``known``: a :func:`load_low_backlash` dict, a path, False for none; default
    the project's list, without the found list) being the exceptions (``listed_names``, regular in
    :func:`image_states`).  The ratios that decide (``ratio_fit``, ``sd_fit``) come from the images not on that
    list; ``ratio`` / ``sd`` are every fitted image's, as before.  ``report`` (a dict) receives the lines and the
    sequences.  ``bin_fallback`` (:func:`bin_focal_fallback`) classifies a group without any per-image fit
    from its own bin camera's refined f (``fit_source`` "bin camera", sd :data:`BIN_FALLBACK_SD`).
    """
    ratios = backlash_ratios(model)
    ns = float(nav_scale) if nav_scale else 1.0
    if isinstance(known, dict):
        lst = known
    elif known is False or (known is None and ((getattr(project, "settings", None) or {}).get("zcam_low_backlash")
                                                or {}).get("disabled")):
        lst = load_low_backlash(False)                   # 0.93.5: a project built without a list keeps none
    else:
        f = known if known is not None else ((getattr(project, "settings", None) or {}).get("zcam_low_backlash") or {}).get("file")
        try:
            lst = load_low_backlash(f)
        except Exception:                                                  # noqa: BLE001
            lst = load_low_backlash(False)
    rows = []
    for key, members in sorted(focus_groups(project.images).items()):
        g = members[0].get("camera_group")
        base = {"group_key": key, "camera_group": g, "eye": members[0].get("eye"), "sol": members[0].get("sol"),
                "sequence": members[0].get("sequence"), "focus_count": members[0].get("focus_count"),
                "images": len(members), "names": [m["name"] for m in members],
                "cameras": sorted({m["instrument"] for m in members})}
        if group_zoom(g) in SINGLE_STATE_ZOOMS:
            rows.append(dict(base, fitted=0, backlash_ratio=None, threshold_ratio=None, ratio=None, sd=None, z=None,
                             state="single", decided=False, listed_names=[]))
            continue
        listed = [m["name"] for m in members if is_low_backlash(m, lst)]
        fs = [fits[m["name"]] for m in members if m["name"] in fits and fits[m["name"]].get("ratio_label")]
        fu = [fits[m["name"]] for m in members if m["name"] in fits and fits[m["name"]].get("ratio_label")
              and m["name"] not in listed]
        row = dict(base, fitted=len(fs), fitted_unlisted=len(fu), nav_scale=ns, listed_names=listed,
                   all_listed=bool(listed) and len(listed) == len(members))

        def stats(ff):
            x = np.array([f["ratio_label"] for f in ff], float)
            w = np.array([f["observations"] for f in ff], float)
            sd_i = np.array([f["sd_ratio"] if f.get("sd_ratio") and np.isfinite(f["sd_ratio"]) else floor for f in ff])
            sd_i = np.maximum(sd_i, floor)
            med = _wmedian(x, w)
            sd = float(1.0 / np.sqrt(np.sum(1.0 / sd_i ** 2)))
            if len(x) > 2:                                   # the images' scatter, when there is enough of it
                sd = max(sd, float(1.4826 * np.median(np.abs(x - med)) / np.sqrt(len(x))))
            return med, sd, x
        if fs:
            med, sd, x = stats(fs)
            row.update({"ratio": med, "sd": sd, "ratio_min": float(x.min()), "ratio_max": float(x.max()),
                        "offset_from_label_pct": 100.0 * (med - 1.0)})
        elif bin_fallback and not listed:
            # no per-image fit - the group's bin camera (its own, not shared with another sequence at the
            # same focus count) stands in: refined f / the group's median label f, with a wide sd
            cams = sorted((c for c in base["cameras"] if c in bin_fallback),
                          key=lambda c: calib.zcam_state(c, (getattr(project, "cameras", None) or {}).get(c)) == "regular")
            same = [m for m in members if m.get("label_f_px")]
            shared = any(r.get("instrument") in cams and r["name"] not in set(base["names"])
                         for r in project.images if r.get("camera_group") == g)
            if cams and same and not shared:
                lf = float(np.median([m["label_f_px"] for m in same]))
                fb = bin_fallback[cams[0]]
                row.update({"ratio": fb["f_px"] / lf, "sd": BIN_FALLBACK_SD, "ratio_fit": fb["f_px"] / lf,
                            "sd_fit": BIN_FALLBACK_SD, "fit_source": "bin camera",
                            "bin_observations": fb["observations"], "offset_from_label_pct": 100.0 * (fb["f_px"] / lf - 1.0)})
            else:
                row.update({"ratio": None, "sd": None})
        else:
            row.update({"ratio": None, "sd": None})
        fl = [fits[n] for n in listed if n in fits and fits[n].get("ratio_label")]
        if fl:                                               # 0.93.4: the listed (regular) images' own ratio
            ml, sl, _ = stats(fl)
            row.update({"ratio_listed": ml, "sd_listed": sl})
        if fu:
            mu, su, _ = stats(fu)
            row.update({"ratio_fit": mu, "sd_fit": su})
        elif fs:                                             # every fitted image on the list: its own ratio, for the record
            row.update({"ratio_fit": None, "sd_fit": None})
        rows.append(row)
    lines = measured_backlash_lines(rows, ns, ratios) if measured_ratio else {
        g: {"a": float(ratios.get(g, DEFAULT_BACKLASH_RATIO)) * ns, "b": 0.0, "n": 0, "source": "focus model",
            "reference_focus": 600.0, "clip": [-np.inf, np.inf], "nav_scale": ns}
        for g in {str(r.get("camera_group")) for r in rows}}
    for row in rows:
        if row.get("state") == "single":
            continue
        ln = lines.get(str(row["camera_group"]))
        rb = backlash_ratio_at(ln, row.get("focus_count")) if ln else DEFAULT_BACKLASH_RATIO * ns
        mid = 0.5 * (ns + rb)
        row.update({"backlash_ratio": rb, "threshold_ratio": mid})
        r_dec = row.get("ratio_fit") if row.get("ratio_fit") is not None else row.get("ratio")
        s_dec = row.get("sd_fit") if row.get("ratio_fit") is not None else row.get("sd")
        if r_dec is None:
            row.update({"z": None, "state": "undecided", "decided": False})
            continue
        z = (r_dec - mid) / s_dec
        implausible = r_dec < ns - tolerance or r_dec > rb + tolerance
        state = "undecided" if implausible else ("regular" if z < -z_min else ("backlash" if z > z_min else "undecided"))
        row.update({"z": float(z), "state": state, "decided": state != "undecided", "implausible": bool(implausible)})
    seqs = apply_sequence_states(rows, z_min=z_min) if sequence_rule else {}
    for row in rows:
        mixed_group_state(row, z_min=z_min)
    if report is not None:
        report["backlash_lines"] = lines
        report["sequences"] = seqs
        report["sequence_conflicts"] = [{k: r.get(k) for k in ("group_key", "own_state", "own_z", "sequence_state",
                                                                "sequence_z", "ratio_fit", "threshold_ratio")}
                                        for r in rows if r.get("conflict")]
    return rows


def mixed_group_state(row: Dict[str, Any], z_min: float = 3.0) -> Optional[str]:
    """0.93.4 (Christian, Seitah Overlook ZCAM08114): a focus group in which only some images are on the low-backlash
    list holds both states - the list marks the regular exceptions, e.g. pointings 2 and 4 of a five-pointing
    mosaic at one focus count.  Its unlisted images are judged against the listed ones of the same group, not against
    the absolute threshold (uncalibrated zooms put both states below it): backlash unless they fit clearly on the
    listed side of the midpoint (the listed ratio + half the block's state gap) - then the group keeps its own call
    (the rule never makes a group regular; ZCAM05086 at Three Forks has a listed image at the backlash ratio).  Sets
    and returns the group's state (for its unlisted images)."""
    listed = row.get("listed_names") or []
    if not listed or row.get("all_listed") or row.get("state") == "single" or row.get("implausible"):
        return None                                      # (implausible: a failed bin, neither state - left as it was)
    ml, sl, mu, su = row.get("ratio_listed"), row.get("sd_listed"), row.get("ratio_fit"), row.get("sd_fit")
    if ml is None or mu is None:
        st, z = "backlash", None
        note = "the list marks the group's regular images; the unlisted ones, without a fit to compare, backlash"
    else:
        # the midpoint between the listed images' ratio and that ratio plus the block's state gap: an unlisted image
        # clearly on the listed side is regular too, anything else backlash (undecided counts as backlash)
        ns, rb = float(row.get("nav_scale") or 1.0), row.get("backlash_ratio")
        gap = (float(rb) - ns) if rb is not None else (DEFAULT_BACKLASH_RATIO - 1.0) * ns
        z = (mu - (ml + 0.5 * gap)) / max(float(np.hypot(sl or 0.0, su or 0.0)), 1e-6)
        st = "regular" if z < -z_min else "backlash"
        note = (f"the unlisted images fit {100 * (mu - ml):+.2f} % from the listed (regular) ones of the same group "
                f"(state gap {100 * gap:.2f} %, z {z:+.1f}): {st}")
    if st == "regular":                                  # the rule only takes images out of the regular state: a
        row.update({"mixed_group": True, "mixed_z": z,   # group it finds like its listed images keeps its own call
                    "mixed_note": note + f" - kept {row.get('state')}"})
        return row.get("state")
    if row.get("state") != st:
        row["state_before_mixed_rule"] = row.get("state")
    row.update({"state": st, "decided": True, "mixed_group": True, "mixed_z": z, "mixed_note": note})
    return st


def image_states(groups: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """{image name: "regular" | "backlash" | "single"} (undecided groups count as backlash; "single" for the
    one-state zooms; images on the low-backlash list (``listed_names``) are regular whatever their group)."""
    out = {}
    for g in groups:
        listed = set(g.get("listed_names") or [])
        for n in g["names"]:
            if g["state"] == "single":
                out[n] = "single"
            elif n in listed or g["state"] == "regular":
                out[n] = "regular"
            else:
                out[n] = "backlash"
    return out


def regular_pp_offsets(model: Optional[Dict[str, Any]]) -> Dict[str, List[float]]:
    """{eye group: [dcx, dcy]} - the regular state's principal point minus the backlash state's in a focus model
    (``regular_pp_offset_px``, 0.93.2; a model without it: none)."""
    out = {}
    for g, gm in ((model or {}).get("cameras") or {}).items():
        off = gm.get("regular_pp_offset_px")
        if off and any(float(x) for x in off[:2]):
            out[g] = [float(off[0]), float(off[1])]
    return out


def split_by_state(rec, project, regular: Iterable[str], hold_f_images: int = 0,
                   pp_offsets: Optional[Dict[str, Sequence[float]]] = None):
    """
    A copy of ``rec`` in which the ``regular`` images (names) of every Mastcam-Z camera move to a new camera
    ``<camera>_reg`` (start f = the median label f of those images, the camera's fy/fx kept; f held when it has at
    most ``hold_f_images`` images), with their frames on new rigs of the new cameras.  ``project`` is updated in place
    (cameras, database ids, each moved image's ``instrument``, with the old one in ``base_instrument`` and its
    ``backlash_state``).  ``pp_offsets``: {eye group: [dcx, dcy]} (:func:`regular_pp_offsets`) - the new camera's
    principal point is the old one's plus its eye's offset (0.93.2: each state its own camera, the pose shared).
    Returns (rec, rows) with one row per new camera.
    """
    import pycolmap
    from .project import FULL_OPENCV_NAMES
    regular = set(regular)
    by_name = {r["name"]: r for r in project.images}
    db = project.settings.setdefault("database", {}).setdefault("cameras", {})
    key_of = {int(v): k for k, v in db.items()}
    moved: Dict[int, List[int]] = {}                             # old camera id -> image ids
    for iid, im in rec.images.items():
        k0 = str(key_of.get(int(im.camera_id), ""))
        # 0.93.5: only a backlash-state camera is split - an image already in a regular camera (the list's _reg
        # bins) stays, and a single-state (110 mm) camera has nothing to split
        if im.name in regular and k0.startswith("Z") and calib.zcam_state(k0, project.cameras.get(k0)) == "backlash":
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
        pc0 = project.cameras.get(old_key, {})
        off = (pp_offsets or {}).get(str(pc0.get("group") or old_key)[:5])
        if off:
            p[2] += float(off[0])
            p[3] += float(off[1])
        new.add_camera(pycolmap.Camera(camera_id=next_cid, model=base.model, width=base.width, height=base.height,
                                       params=p))
        key = f"{old_key}{REGULAR_SUFFIX}"
        k_ = 2
        while key in project.cameras or key in db:          # 0.93.4: the list's own regular bin has this name
            key = f"{old_key}_{k_}{REGULAR_SUFFIX}"
            k_ += 1
        pc = copy.deepcopy(project.cameras.get(old_key, {}))
        fixed = list(pc.get("fixed_params") or [])
        held_f = len(iids) <= int(hold_f_images or 0)
        if held_f:
            fixed = [n for n in FULL_OPENCV_NAMES if n in set(fixed) | {"fx", "fy"}]
        else:                                            # 0.93.5: its f starts at the labels - free to find its state
            fixed = [n for n in fixed if n not in ("fx", "fy")]
        pc.update({"params": p.tolist(), "fixed_params": fixed, "n_images": len(iids), "backlash_state": "regular",
                   "state_split": True, "base_camera": old_key,
                   "source": f"{pc.get('source', old_key)}; regular focus state (not backlash): f from the median "
                             f"label f of its {len(iids)} images" + ("; f held" if held_f else "")
                             + (f"; principal point {off[0]:+.2f}, {off[1]:+.2f} px (the regular state's)" if off else "")})
        project.cameras[key] = pc
        if old_key in project.cameras:
            oc = project.cameras[old_key]
            oc.setdefault("pre_split", {"n_images": oc.get("n_images"), "backlash_state": oc.get("backlash_state")})
            oc["backlash_state"] = "backlash"
            oc["n_images"] = max(int(oc.get("n_images", 0)) - len(iids), 0)
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
        elif r is not None and str(r.get("instrument", "")).startswith("Z"):   # 0.93.5: its camera's state
            r["backlash_state"] = calib.zcam_state(r["instrument"], project.cameras.get(r["instrument"]))
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
                   z_min: float = 3.0, min_observations: int = 20, verbose: bool = True,
                   hold_cameras: Sequence[str] = (),
                   sequence_rule: bool = True, measured_ratio: bool = True, bin_fallback: bool = True):
    """
    Classify the Mastcam-Z focus groups of a solved block into the backlash and the regular state and (``mode``
    ``"split"``) give the regular images their own cameras (:func:`split_by_state`) and adjust again with the Navcam
    cameras held; ``"report"`` classifies only.  ``model``: the focus model (backlash ratios; default: the project's,
    ``settings["zcam_focus_model"]["file"]``, else the shipped ones of every zoom).  ``project`` is updated in place.  Returns (rec, report) with the groups, the per-image fits before (and
    after) the split and the new cameras.
    """
    from .project import zcam_focus_model
    from .reconstruction import bundle_adjust
    if mode not in ("split", "report", "list", "list_split"):
        raise ValueError("zcam_backlash must be 'list_split', 'list', 'split', 'report' or None")
    list_split = mode == "list_split"      # the list's states, plus groups classified regular now (split)
    if mode == "list":            # the states come from the list (bins made at the start); classify to compare
        mode = "report"
    if model is None:
        try:
            f = (project.settings.get("zcam_focus_model") or {}).get("file")
            model = zcam_focus_model(model_file=f)
        except Exception:                                                # noqa: BLE001
            model = None
    ns = navcam_scale(rec, project)
    fits = image_focal_fits(rec, project, min_observations=min_observations, loss_px=loss_scale)
    cls: Dict[str, Any] = {}
    groups = classify_groups(project, fits, model, z_min=z_min, nav_scale=ns, sequence_rule=sequence_rule,
                             measured_ratio=measured_ratio, report=cls,
                             bin_fallback=bin_focal_fallback(rec, project) if bin_fallback else None)
    states = image_states(groups)
    report: Dict[str, Any] = {"mode": mode, "z_min": z_min, "backlash_ratios": backlash_ratios(model), "nav_scale": ns,
                              "groups": groups, "fits_before": fits, "before": _state_summary(fits, states),
                              "cameras": [], "after": None, "fits_after": None,
                              "backlash_lines": cls.get("backlash_lines"), "sequences": cls.get("sequences"),
                              "sequence_conflicts": cls.get("sequence_conflicts")}
    n = {s: sum(1 for g in groups if g["state"] == s) for s in ("backlash", "regular", "undecided", "single")}
    report["group_counts"] = n
    report["implausible_groups"] = [g["group_key"] for g in groups if g.get("implausible")]
    regular = [nm for nm, s in states.items() if s == "regular"]
    if list_split:
        # the groups the list already had (their images start in _reg cameras) are not split again; groups
        # classified regular that the list did not have are the new ones (flagged, split, and for the found list)
        listed = {r["name"] for r in project.images if r.get("focus_state") == "regular"}
        new_groups = [g for g in groups if g["state"] == "regular" and not g.get("implausible")
                      and not set(g.get("names") or []) <= listed]
        # an image on the list (or the found list) whose group the fit and its sequence call backlash -
        # the shipped list's images are the known exceptions (kept regular); found-list ones are contradicted
        disagree = [g for g in groups if g["state"] == "backlash"
                    and (set(g.get("names") or []) & listed) - set(g.get("listed_names") or [])]
        report["new_regular_groups"] = [{k: v for k, v in g.items()} for g in new_groups]
        report["list_disagrees"] = [{k: v for k, v in g.items() if k != "names"} for g in disagree]
        regular = sorted({nm for g in new_groups for nm in g.get("names") or []} - listed)
        mode = "split" if regular else "report"
        if verbose:
            for g in new_groups:
                log.info(f"LOW BACKLASH (new): {g['camera_group']} sol {g['sol']} {g['sequence']} focus "
                      f"{g['focus_count']}: f / label f {g['ratio']:.4f} (z {g['z']:+.1f}), {g['images']} images - not on "
                      f"the list: split now, and added to the found list")
            for g in disagree:
                zz = g.get("z")
                log.info(f"found list says low backlash, the fit and its sequence say backlash: {g['camera_group']} "
                      f"sol {g['sol']} {g['sequence']} focus {g['focus_count']} (f / label f "
                      f"{(g.get('ratio_fit') or g.get('ratio') or float('nan')):.4f}, z "
                      f"{(zz if zz is not None else float('nan')):+.1f}) - remove it from the found list")
    if verbose:
        for g, ln in sorted((report.get("backlash_lines") or {}).items()):
            log.info(f"{g} backlash state: f / label f {ln['a']:.4f} {ln['b'] * 1000:+.4f} per 1000 counts about "
                  f"focus {ln['reference_focus']:.0f} ({ln['source']}, {ln['n']} groups)")
        for c in report.get("sequence_conflicts") or []:
            log.info(f"sequence rule: {c['group_key']} fits {c['own_state']} (z {c['own_z']:+.1f}) but its sequence "
                  f"is {c['sequence_state']} (z {c['sequence_z']:+.1f}): {c['sequence_state']}")
        b = report["before"]
        log.info(f"Mastcam-Z focus states: {len(groups)} focus groups - {n['backlash']} backlash, {n['regular']} "
              f"regular, {n['undecided']} undecided (kept as backlash), {n['single']} single-state (110 mm); "
              f"Navcam scale {(ns or 1.0):.5f}; fitted f / label f median "
              f"{(b['backlash']['ratio_median'] or float('nan')):.4f} (backlash), "
              f"{(b['regular']['ratio_median'] or float('nan')):.4f} (regular)")
    if mode == "report" or not regular:
        return rec, report
    rec2, rows = split_by_state(rec, project, regular, hold_f_images=hold_f_images,
                                pp_offsets=regular_pp_offsets(model))
    report["cameras"] = rows
    # the Navcam, and the block's other held cameras (0.93.1: a weak Hazcam eye stays at its start)
    held = sorted({k for k in project.cameras if str(k).startswith("N")} | set(hold_cameras))
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
        log.info(f"{len(rows)} regular-state cameras ({len(regular)} images); after the adjustment fitted f / "
              f"label f median {(a['backlash']['ratio_median'] or float('nan')):.4f} (backlash), "
              f"{(a['regular']['ratio_median'] or float('nan')):.4f} (regular)")
    return rec2, report
