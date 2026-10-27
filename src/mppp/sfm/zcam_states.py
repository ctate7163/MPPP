"""
The Mastcam-Z focus-backlash states of every image, from the blocks' own focal lengths (0.94.4, Christian 25 Oct):
which frames belong on the low-state (found) list, which do not, and which cannot be told yet.

Each block's ``zcam_focus_states.json`` holds, per Mastcam-Z image, its fitted focal length over its label focal
length (:func:`mppp.sfm.backlash.image_focal_fits`).  In a states run (``run_zcam_states.bat``: variant ``states``,
``zcam_bin_by = "image"``, no focus lines) that is each image's own camera.  Per block, eye and zoom:

- the high (backlash) state is a line in focus count, fitted robustly to the images not on the lists (the dominant
  state), refitted without the images it then finds low;
- the low state is that line minus the block's state gap - measured from the listed images that have a fit, else
  the median gap of the other blocks at that zoom, else the default 0.95 %;
- each image is ``low`` when its midpoint distance is below -z_min, ``high`` above +z_min, else ``undecided``
  (``implausible`` when it is far outside both states: a failed fit).

Then per sequence (block, sol, sequence, eye and zoom): its images' states in clock order (``LHLHL`` for sol 121
ZCAM08114), the verdict ``low`` / ``high`` / ``mixed`` / ``undecided``.  Most sequences are in one state; a mixed one
is reported for a look.  The proposal for the found list: images found low that no list has (``add``); a sequence
whose fitted images are all low adds its unfitted images too ("by its sequence"); listed images found high
(``remove``: a ``-`` line); and the cases to review.  ``apply_proposal`` writes it into the found list (a dated
backup first).
"""
from __future__ import annotations

import csv
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.sfm")
PathLike = Union[str, Path]

Z_MIN = 3.0
DEFAULT_GAP = 0.0095 / 1.0095        # (high - low) / high when nothing measures it (the 1.0095 default ratio)
SCATTER_FLOOR = 0.0005               # the least scatter of an image's ratio about its state (0.05 %)
IMPLAUSIBLE = 0.004                  # this far outside both states: a failed fit, not a state
MIN_GAP = 0.002                      # a measured gap below 0.2 % is not trusted (the pooled gap is used)
REF_FOCUS = 600.0
MIN_HIGH_IMAGES = 3                  # fewer unlisted fitted images than this: the listed (low) images set the levels
MIN_SLOPE_SPAN = 500.0               # a block's own focus slope needs this span of focus counts (and 10 images);
MIN_SLOPE_IMAGES = 10                # else the slope pooled over the blocks for that eye and zoom


def _zoom(group: str) -> Optional[int]:
    try:
        return int(str(group)[2:5])
    except ValueError:
        return None


def _clock(name: str) -> str:
    p = str(name).split("_")
    return p[2] if len(p) > 2 else str(name)


# ====================================================================================== the blocks' images
def _manifest_moves(work: PathLike) -> Dict[str, Dict[str, Any]]:
    """The motor moves of a block's Mastcam-Z images (:func:`mppp.sfm.backlash.motor_moves`) from its processed
    manifest (the newest ``processed/mppp_manifest_*.json``); {} without one."""
    from .backlash import motor_moves
    ms = sorted((Path(work) / "processed").glob("mppp_manifest_*.json"), key=lambda p: p.stat().st_mtime)
    if not ms:
        return {}
    try:
        man = json.loads(ms[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return motor_moves(man.get("images") or [])


def block_images(work: PathLike, variant: str = "", lists: Optional[Dict[str, Any]] = None,
                 focus_list: Optional[Dict[str, Any]] = None, single: bool = False) -> List[Dict[str, Any]]:
    """One row per Mastcam-Z image of a block (110 mm left out: one state; ``single``: only the 110 mm images):
    name, block, group, eye, zoom, sol, sequence, focus, ratio (fitted / label f; None without a fit), sd,
    observations, listed (shipped or found list), camera; 0.94.4 (Christian, 26 Oct): the motor moves from the
    previous frame (``zoom_dir``, ``focus_dir``, ``dzoom``, ``dfocus``) and ``focus_below`` (on the focus-approach
    list)."""
    from .backlash import SINGLE_STATE_ZOOMS, is_focus_from_below, is_low_backlash
    from .workdir import project_dir
    root = project_dir(Path(work), variant or None)
    fs = root / "zcam_focus_states.json"
    pj = root / "project.json"
    if not fs.is_file() or not pj.is_file():
        return []
    st = json.loads(fs.read_text(encoding="utf-8"))
    proj = json.loads(pj.read_text(encoding="utf-8"))
    fits = dict(st.get("fits_before") or {})
    fits.update(st.get("fits_after") or {})                  # after a split: the images' last fits
    out = []
    moves = _manifest_moves(work)
    for r in proj.get("images") or []:
        g = str(r.get("camera_group") or "")
        if not g.startswith("Z") or ((_zoom(g) in SINGLE_STATE_ZOOMS) != bool(single)):
            continue
        f = fits.get(r["name"]) or {}
        mv = moves.get(r.get("stem") or Path(r["name"]).stem) or {}
        out.append({"block": Path(work).name, "name": r["name"], "stem": r.get("stem") or Path(r["name"]).stem,
                    "group": g, "eye": g[:2], "zoom": _zoom(g), "sol": r.get("sol"), "sequence": r.get("sequence"),
                    "focus": r.get("focus_count"), "clock": _clock(r["name"]),
                    "ratio": f.get("ratio_label"), "sd": f.get("sd_ratio"), "observations": f.get("observations"),
                    "camera": f.get("camera"), "cx": f.get("cx_px"), "cy": f.get("cy_px"),
                    "listed": bool(lists is not None and not single and is_low_backlash(r, lists)),
                    "start_state": r.get("focus_state"),
                    "zoom_dir": mv.get("zoom_dir"), "focus_dir": mv.get("focus_dir"), "dzoom": mv.get("dzoom"),
                    "dfocus": mv.get("dfocus"),
                    "focus_below": bool(focus_list is not None and is_focus_from_below(r, focus_list))})
    return out


# ====================================================================================== the two states per block
def _robust_line(F: np.ndarray, y: np.ndarray, w: np.ndarray, it: int = 6, slope: Optional[float] = None
                 ) -> Tuple[float, float, float, np.ndarray]:
    """y = a + b (F - REF) / 1000, iteratively 3-sigma clipped (weights w); ``slope`` given: b held there.  Returns
    a, b, robust sd, kept mask."""
    keep = np.isfinite(y) & np.isfinite(F)
    a, b, s = float(np.nanmedian(y)) if keep.any() else 1.0, 0.0 if slope is None else float(slope), SCATTER_FLOOR
    for _ in range(it):
        if slope is not None and keep.any():
            a = float(np.median(y[keep] - b * (F[keep] - REF_FOCUS) / 1000.0))
        elif keep.sum() >= 3 and np.ptp(F[keep]) > 50:
            X = np.column_stack([np.ones(keep.sum()), (F[keep] - REF_FOCUS) / 1000.0])
            W = w[keep]
            coef = np.linalg.lstsq(X * np.sqrt(W)[:, None], y[keep] * np.sqrt(W), rcond=None)[0]
            a, b = float(coef[0]), float(coef[1])
        elif keep.any():
            a, b = float(np.median(y[keep])), 0.0
        r = y - (a + b * (F - REF_FOCUS) / 1000.0)
        s = max(1.4826 * float(np.median(np.abs(r[keep]))) if keep.any() else SCATTER_FLOOR, SCATTER_FLOOR)
        new = np.isfinite(r) & (np.abs(r) <= 3 * s)
        if new.sum() < 2 or (new == keep).all():
            keep = new if new.sum() >= 2 else keep
            break
        keep = new
    return a, b, s, keep


def classify(rows: List[Dict[str, Any]], z_min: float = Z_MIN, pooled_gap: Optional[Dict[int, float]] = None,
             pooled_slope: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Classify the images of one block, eye and zoom (``rows`` changed in place: ``d`` the ratio minus the high
    line, ``gap`` (px ratio units), ``z_mid``, ``state``).  Returns the levels {a, b, scatter, gap (relative),
    gap_source, n_high, n_low, ...}."""
    fitted = [r for r in rows if r.get("ratio") is not None and r.get("focus") is not None]
    for r in rows:
        if r not in fitted:
            r.update(state="no fit", d=None, z_mid=None, gap=None)
    lv: Dict[str, Any] = {"images": len(rows), "fitted": len(fitted)}
    if not fitted:
        return dict(lv, a=None, b=None, scatter=None, gap=None, gap_source=None, n_high=0, n_low=0)
    F = np.array([float(r["focus"]) for r in fitted])
    y = np.array([float(r["ratio"]) for r in fitted])
    sd = np.array([float(r["sd"]) if r.get("sd") else SCATTER_FLOOR for r in fitted])
    listed = np.array([bool(r["listed"]) for r in fitted])
    w = 1.0 / np.maximum(sd, SCATTER_FLOOR) ** 2
    pg = (pooled_gap or {}).get(fitted[0]["zoom"])
    own_slope = len(fitted) >= MIN_SLOPE_IMAGES and float(np.ptp(F)) >= MIN_SLOPE_SPAN
    slope = None if own_slope else (pooled_slope or {}).get(fitted[0]["group"], 0.0)
    low = np.zeros(len(fitted), bool)
    for _ in range(4):
        base = ~listed & ~low
        if base.sum() >= MIN_HIGH_IMAGES or (base.any() and not listed.any()):
            # the high line from the unlisted high images
            a, b, s, _k = _robust_line(F[base], y[base], w[base], slope=slope)
            gap, src = None, None
            ref = listed if listed.any() else low
            if ref.any():
                g = -float(np.median(y[ref] - (a + b * (F[ref] - REF_FOCUS) / 1000.0))) / a
                if g >= MIN_GAP:
                    gap, src = g, ("measured (listed images)" if listed.any() else "measured (images found low)")
            if gap is None:
                gap, src = (pg, "pooled (other blocks)") if pg else (DEFAULT_GAP, "default 0.95 %")
        else:                                                # (nearly) every fitted image listed: the listed
            a0, b, s, _k = _robust_line(F[listed], y[listed], w[listed], slope=slope)   # ones are the low state
            gap, src = (pg, "pooled (other blocks)") if pg else (DEFAULT_GAP, "default 0.95 %")
            a = a0 / (1.0 - gap)
        line = a + b * (F - REF_FOCUS) / 1000.0
        d = y - line
        G = gap * a
        sig = np.hypot(sd, max(s, SCATTER_FLOOR))
        zm = (d + 0.5 * G) / sig
        new_low = (zm < -z_min) & ~listed
        if (new_low == low).all():
            break
        low = new_low
    for r, di, zi, sgi in zip(fitted, d, zm, sig):
        implausible = di > IMPLAUSIBLE or di < -G - IMPLAUSIBLE
        st = "implausible" if implausible else ("low" if zi < -z_min else ("high" if zi > z_min else "undecided"))
        r.update(d=float(di), gap=float(G), z_mid=float(zi), sd_eff=float(sgi), state=st)
    return dict(lv, a=float(a), b=float(b), scatter=float(s), gap=float(gap), gap_source=src,
                slope_source="own" if own_slope else "pooled",
                focus_span=float(np.ptp(F)),
                n_high=sum(1 for r in fitted if r["state"] == "high"),
                n_low=sum(1 for r in fitted if r["state"] == "low"),
                n_listed_fitted=int(listed.sum()))


def survey(blocks: Dict[str, PathLike], variant: str = "", lists: Optional[Dict[str, Any]] = None,
           z_min: float = Z_MIN, focus_list: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Every block's Mastcam-Z images classified; the per-zoom gap pooled over the blocks that measure one; the
    focal ratios by motor approach (:func:`approach_table`, 110 mm included).
    Returns {"images", "levels", "sequences", "pooled_gap", "proposal", "approach"}."""
    rows: List[Dict[str, Any]] = []
    single: List[Dict[str, Any]] = []
    for name, w in blocks.items():
        b = block_images(w, variant, lists, focus_list)
        s1 = block_images(w, variant, lists, focus_list, single=True)
        for r in b + s1:
            r["block"] = name
        rows += b
        single += s1
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["block"], r["group"]), []).append(r)
    # pass 1: each block's own gaps; pooled per zoom from those measured on listed images
    lv0 = {k: classify(v, z_min) for k, v in by.items()}
    slopes: Dict[str, float] = {}                            # the focus slope per eye and zoom, from the blocks
    for grp in sorted({k[1] for k in by}):                   # that span enough focus
        bs = [x["b"] for (blk, g), x in lv0.items() if g == grp and x.get("slope_source") == "own"]
        if bs:
            slopes[grp] = float(np.median(bs))
    lv1 = {k: classify(v, z_min, None, slopes) for k, v in by.items()}
    pooled: Dict[int, float] = {}
    for zoom in sorted({r["zoom"] for r in rows}):
        g = [x["gap"] for (blk, grp), x in lv1.items() if _zoom(grp) == zoom and x.get("gap")
             and str(x.get("gap_source", "")).startswith("measured") and x["gap"] >= MIN_GAP]
        if g:
            pooled[zoom] = float(np.median(g))
    levels = {f"{k[0]}|{k[1]}": classify(v, z_min, pooled, slopes) for k, v in by.items()}
    centre_offsets(by)
    seqs = sequences(rows)
    return {"images": rows, "levels": levels, "pooled_gap": pooled, "pooled_slope": slopes, "sequences": seqs,
            "proposal": proposal(rows, seqs, lists), "z_min": z_min, "single_images": single,
            "approach": approach_table(rows, single), "focus_list_proposal": focus_list_proposal(rows + single)}


CENTRE_FLAG_PX = 1.0                 # a sequence whose camera centre sits this far from its eye's high-state centre


def centre_offsets(by: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> None:
    """0.94.5 (Christian, 27 Oct: the right eye's backlash state shows mainly in its principal point - Seitah Overlook
    ZCAM08114 pointings 1, 3, 5 fit high in f but 2-3.7 px off in the high-state bins): per (block, eye and zoom),
    each image's cx, cy (``dcx``, ``dcy``, full-frame px) about the median of the block's unlisted high-state images
    of that camera.  Only where the centres vary (the states run with ``ZCAM_BIN_REFINE=focal_pp`` frees ZR's; ZL holds
    its centre as the reference eye).  Sets ``dcx``, ``dcy`` on the rows (None otherwise)."""
    for (_blk, _grp), v in by.items():
        c = [(r["cx"], r["cy"]) for r in v if r.get("cx") is not None and r.get("cy") is not None]
        if len(c) < 2 or float(np.ptp([x for x, _ in c])) + float(np.ptp([y for _, y in c])) < 1e-6:
            for r in v:
                r["dcx"] = r["dcy"] = None
            continue
        ref = [(r["cx"], r["cy"]) for r in v if r.get("state") == "high" and not r.get("listed") and r.get("cx") is not None]
        if len(ref) < 2:                                   # the unlisted high-state images, else every high one
            ref = [(r["cx"], r["cy"]) for r in v if r.get("state") == "high" and r.get("cx") is not None]
        ref = ref if len(ref) >= 2 else c
        mx, my = float(np.median([x for x, _ in ref])), float(np.median([y for _, y in ref]))
        for r in v:
            ok = r.get("cx") is not None and r.get("cy") is not None
            r["dcx"] = float(r["cx"]) - mx if ok else None
            r["dcy"] = float(r["cy"]) - my if ok else None


FOCUS_MOVE_MIN = 100.0               # a focus move up this large (counts, from the previous frame) proposes the frame


def approach_table(rows: List[Dict[str, Any]], single: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """0.94.4 (Christian, 26 Oct: the zoom and the focus backlash superimposed): per zoom and eye, the frames'
    focal ratios by how the motors came into place - zoom up / down / none and focus up / down / none from the
    previous frame (within 30 min) - and by the lists (zoom low state, focus from below).  ``d`` is the ratio about
    the block's high line (34-79 mm, :func:`classify`); at 110 mm (no high line) about the median of the block's
    110 mm frames of that eye.  Returns rows {zoom, eye, zoom_dir, focus_dir, listed, focus_below, n, d_median_pct,
    d_mean_pct, d_se_pct}."""
    s_rows = []
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in single:
        if r.get("ratio") is not None:
            by.setdefault((r["block"], r["group"]), []).append(r)
    for v in by.values():
        med = float(np.median([r["ratio"] for r in v]))
        for r in v:
            s_rows.append(dict(r, d=float(r["ratio"]) - med, state="single"))
    groups: Dict[Tuple[Any, ...], List[float]] = {}
    for r in list(rows) + s_rows:
        if r.get("d") is None or r.get("state") in ("implausible", "no fit"):
            continue
        k = (r["zoom"], r["eye"], r.get("zoom_dir") or "-", r.get("focus_dir") or "-", bool(r.get("listed")),
             bool(r.get("focus_below")))
        groups.setdefault(k, []).append(float(r["d"]))
    out = []
    for k, v in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        a = np.array(v)
        out.append({"zoom": k[0], "eye": k[1], "zoom_dir": k[2], "focus_dir": k[3], "listed": k[4],
                    "focus_below": k[5], "n": int(a.size), "d_median_pct": 100 * float(np.median(a)),
                    "d_mean_pct": 100 * float(a.mean()),
                    "d_se_pct": 100 * float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else None})
    return out


def focus_list_proposal(rows: Sequence[Dict[str, Any]], min_move: float = FOCUS_MOVE_MIN) -> Dict[str, List[Dict[str, Any]]]:
    """Frames whose focus moved UP by at least ``min_move`` counts from the previous frame (same eye, within 30
    min) and are not on the focus-approach list (``add``); listed frames whose focus moved down by that much
    (``check``).  Only the frames in the blocks count (a sweep or a sequence in between is not seen)."""
    add, check = [], []
    for r in rows:
        dfc = r.get("dfocus")
        if dfc is None:
            continue
        if dfc >= min_move and not r.get("focus_below"):
            add.append({k: r.get(k) for k in ("block", "sol", "sequence", "group", "stem", "dzoom", "dfocus")})
        elif dfc <= -min_move and r.get("focus_below"):
            check.append({k: r.get(k) for k in ("block", "sol", "sequence", "group", "stem", "dzoom", "dfocus")})
    return {"add": add, "check": check}


def sequences(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (block, sol, sequence, eye and zoom): the images' states in clock order and the verdict."""
    by: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["block"], r["sol"], r["sequence"], r["group"]), []).append(r)
    out = []
    code = {"low": "L", "high": "H", "undecided": "?", "no fit": ".", "implausible": "x"}
    for (blk, sol, seq, grp), v in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0, str(kv[0][2]), kv[0][3])):
        v = sorted(v, key=lambda r: r["clock"])
        n = {s: sum(1 for r in v if r["state"] == s) for s in code}
        nl = sum(1 for r in v if r["listed"])
        if n["low"] and n["high"]:
            verdict = "mixed"
        elif n["low"]:
            verdict = "low"
        elif n["high"]:
            verdict = "high"
        elif n["undecided"] or n["implausible"]:
            verdict = "undecided"
        else:
            verdict = "no fit"
        ds = [r["d"] for r in v if r.get("d") is not None]
        out.append({"block": blk, "sol": sol, "sequence": seq, "group": grp, "images": len(v), "listed": nl,
                    "low": n["low"], "high": n["high"], "undecided": n["undecided"], "no_fit": n["no fit"],
                    "implausible": n["implausible"], "verdict": verdict,
                    "pattern": "".join(code[r["state"]] for r in v),
                    "listed_pattern": "".join("*" if r["listed"] else "-" for r in v),
                    "d_median_pct": 100 * float(np.median(ds)) if ds else None,
                    "dcx_median_px": (float(np.median([r["dcx"] for r in v if r.get("dcx") is not None]))
                                      if any(r.get("dcx") is not None for r in v) else None),
                    "dcy_median_px": (float(np.median([r["dcy"] for r in v if r.get("dcy") is not None]))
                                      if any(r.get("dcy") is not None for r in v) else None),
                    "gap_pct": 100 * float(v[0]["gap"]) if v[0].get("gap") else None,
                    "focus": sorted({r["focus"] for r in v if r.get("focus") is not None})[:3]})
    return out


def proposal(rows: List[Dict[str, Any]], seqs: List[Dict[str, Any]], lists: Optional[Dict[str, Any]] = None
             ) -> Dict[str, List[Dict[str, Any]]]:
    """add (found low, on no list, in a sequence whose fitted images are all low; the unfitted members of such a
    sequence when it has two low images and no other zoom of the sol and sequence is mixed - never a frame a ``-``
    line takes off), remove (listed, found high), review (a low image in a sequence otherwise high, mixed
    sequences, a sequence whose zooms disagree, listed images undecided, implausible fits)."""
    from .backlash import _matches
    sv = {(s["block"], s["sol"], s["sequence"], s["group"]): s for s in seqs}
    mixed_seq = {(s["block"], s["sol"], s["sequence"]) for s in seqs if s["verdict"] == "mixed"}
    zoom_states: Dict[Tuple[Any, ...], set] = {}
    for s in seqs:
        if s["verdict"] in ("low", "high"):
            zoom_states.setdefault((s["block"], s["sol"], s["sequence"]), set()).add(s["verdict"])
    ex = (lists or {}).get("excluded")
    add, remove, review = [], [], []
    for r in rows:
        s = sv[(r["block"], r["sol"], r["sequence"], r["group"])]
        off = bool(ex and _matches(r, ex))
        if r["state"] == "low" and not r["listed"] and not off:
            why = (f"f / label {r['ratio']:.4f}, {100 * r['d']:+.2f} % from the high line "
                   f"(gap {100 * r['gap']:.2f} %, z {r['z_mid']:+.1f})")
            if s["verdict"] == "low":
                add.append(dict(r, why=why))
            else:
                review.append(dict(r, why=f"low in a {s['verdict']} sequence `{s['pattern']}`: {why}"))
        elif (r["state"] == "no fit" and not r["listed"] and not off and s["verdict"] == "low" and s["low"] >= 2
              and (r["block"], r["sol"], r["sequence"]) not in mixed_seq):
            add.append(dict(r, why=f"by its sequence ({s['low']} of its images low, none high)"))
        elif r["state"] == "high" and r["listed"]:
            remove.append(dict(r, why=f"f / label {r['ratio']:.4f}, {100 * r['d']:+.2f} % from the high line "
                                      f"(gap {100 * r['gap']:.2f} %, z {r['z_mid']:+.1f})"))
        elif r["listed"] and r["state"] in ("undecided", "implausible"):
            review.append(dict(r, why=f"listed, {r['state']} (z {r['z_mid']:+.1f})" if r.get("z_mid") is not None
                               else f"listed, {r['state']}"))
        elif r["state"] == "implausible":
            review.append(dict(r, why=f"implausible fit (f / label {r['ratio']:.4f}, {100 * r['d']:+.2f} %)"))
    for s in seqs:
        if s["verdict"] == "mixed":
            review.append({"block": s["block"], "sol": s["sol"], "sequence": s["sequence"], "group": s["group"],
                           "name": None, "why": f"mixed sequence {s['pattern']} (listed {s['listed_pattern']})"})
    for (blk, sol, seq), v in sorted(zoom_states.items(), key=lambda kv: str(kv[0])):
        if len(v) > 1:
            zs = ", ".join(f"{s['group']} {s['verdict']}" for s in seqs
                           if (s["block"], s["sol"], s["sequence"]) == (blk, sol, seq) and s["verdict"] in ("low", "high"))
            review.append({"block": blk, "sol": sol, "sequence": seq, "group": "-", "name": None,
                           "why": f"its zooms disagree: {zs}"})
    return {"add": add, "remove": remove, "review": review}


# ====================================================================================== files
def write_survey(res: Dict[str, Any], out: PathLike) -> Dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    cols = ["block", "group", "sol", "sequence", "clock", "focus", "ratio", "sd", "observations", "listed",
            "start_state", "d", "gap", "z_mid", "state", "cx", "cy", "dcx", "dcy", "name"]
    with (out / "images.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(res["images"])
    files["images"] = out / "images.csv"
    scols = ["block", "sol", "sequence", "group", "images", "listed", "low", "high", "undecided", "no_fit",
             "implausible", "verdict", "pattern", "listed_pattern", "d_median_pct", "gap_pct", "dcx_median_px",
             "dcy_median_px"]
    with (out / "sequences.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, scols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(res["sequences"])
    files["sequences"] = out / "sequences.csv"
    (out / "levels.json").write_text(json.dumps({"levels": res["levels"], "pooled_gap": res["pooled_gap"],
                                                 "pooled_slope": res.get("pooled_slope")}, indent=1),
                                     encoding="utf-8")
    files["proposal"] = out / "found_list_proposal.txt"
    files["proposal"].write_text("\n".join(proposal_lines(res["proposal"])) + "\n", encoding="utf-8")
    files["report"] = out / "report.md"
    files["report"].write_text("\n".join(report_lines(res)) + "\n", encoding="utf-8")
    return files


def proposal_lines(p: Dict[str, List[Dict[str, Any]]], when: Optional[str] = None) -> List[str]:
    """The lines for the found list: additions (stems), removals (``-`` stems), each with its reason."""
    when = when or datetime.date.today().isoformat()
    out = [f"# {when}: mppp zcam-states proposal - {len(p['add'])} to add, {len(p['remove'])} to take off "
           f"({len(p['review'])} cases to review are in report.md, not here)"]
    for r in p["add"]:
        out.append(f"{r['stem']}  # {when} states: {r['block']} sol {r['sol']} {r['sequence']} {r['group']} focus "
                   f"{r['focus']}: low - {r['why']}")
    for r in p["remove"]:
        out.append(f"-{r['stem']}  # {when} states: {r['block']} sol {r['sol']} {r['sequence']} {r['group']} focus "
                   f"{r['focus']}: listed but high - {r['why']}")
    return out


def apply_proposal(p: Dict[str, List[Dict[str, Any]]], found: PathLike, when: Optional[str] = None) -> Dict[str, Any]:
    """Write the proposal into the found list ``found``: a backup ``<name>_<yyyymmdd_HHMMSS>.bak.csv`` first, then
    the lines appended (stems already there skipped).  Returns {"backup", "added", "removed"}."""
    f = Path(found)
    have = set()
    if f.is_file():
        for ln in f.read_text(encoding="utf-8-sig").splitlines():
            t = ln.split("#", 1)[0].strip().upper()
            if t:
                have.add(t)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = f.with_name(f"{f.stem}_{stamp}.bak{f.suffix}") if f.is_file() else None
    if bak:
        shutil.copy2(f, bak)
    lines = proposal_lines(p, when)
    new = [ln for ln in lines[1:] if ln.split("#", 1)[0].strip().upper() not in have]
    if new:
        with f.open("a", encoding="utf-8") as fh:
            fh.write("\n" + lines[0] + "\n" + "\n".join(new) + "\n")
    return {"backup": str(bak) if bak else None, "added": sum(1 for ln in new if not ln.startswith("-")),
            "removed": sum(1 for ln in new if ln.startswith("-"))}


def report_lines(res: Dict[str, Any]) -> List[str]:
    p = res["proposal"]
    imgs = res["images"]
    n = {s: sum(1 for r in imgs if r["state"] == s) for s in ("low", "high", "undecided", "implausible", "no fit")}
    out = ["# Mastcam-Z backlash states, image by image", "",
           f"{len(imgs)} images (110 mm left out: one state): {n['high']} high, {n['low']} low, {n['undecided']} "
           f"undecided, {n['implausible']} implausible, {n['no fit']} without a fit. z_min {res['z_min']:g}.", "",
           "The high (backlash) state is each block's line in focus count; the low state that line minus the state "
           "gap. `d` is an image's f / label f minus the high line.", "",
           "## State gaps (high - low) / high per zoom, pooled over the blocks", ""]
    for z, g in sorted(res["pooled_gap"].items()):
        out.append(f"- {z} mm: {100 * g:.2f} %")
    out += ["", "## Levels per block, eye and zoom", "",
            "| block | camera | images | fitted | high line at focus 600 | slope /1000 counts | scatter | gap | gap from | high | low |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, x in sorted(res["levels"].items()):
        blk, grp = k.split("|")
        if x.get("a") is None:
            out.append(f"| {blk} | {grp} | {x['images']} | 0 | - | - | - | - | - | - | - |")
            continue
        out.append(f"| {blk} | {grp} | {x['images']} | {x['fitted']} | {x['a']:.4f} | {x['b']:+.4f} | "
                   f"{100 * x['scatter']:.2f} % | {100 * x['gap']:.2f} % | {x['gap_source']} | {x['n_high']} | "
                   f"{x['n_low']} |")
    out += ["", "## Sequences in the low state or mixed", "",
            "Pattern: the images in clock order - L low, H high, ? undecided, . no fit, x implausible; "
            "listed: * on a list.", "",
            "| block | sol | sequence | camera | images | pattern | listed | verdict | d median | gap |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["sequences"]:
        if s["verdict"] in ("low", "mixed") or s["listed"]:
            out.append(f"| {s['block']} | {s['sol']} | {s['sequence']} | {s['group']} | {s['images']} | "
                       f"`{s['pattern']}` | `{s['listed_pattern']}` | {s['verdict']} | "
                       + (f"{s['d_median_pct']:+.2f} %" if s["d_median_pct"] is not None else "-") + " | "
                       + (f"{s['gap_pct']:.2f} %" if s["gap_pct"] is not None else "-") + " |")
    cen = [s for s in res["sequences"] if s.get("dcx_median_px") is not None
           and (s["listed"] or np.hypot(s["dcx_median_px"], s["dcy_median_px"] or 0.0) >= CENTRE_FLAG_PX)]
    if any(s.get("dcx_median_px") is not None for s in res["sequences"]):
        out += ["", "## Camera centre by sequence (where the centres are free: the right eye in the states run)", "",
                f"The centre (cx, cy, full-frame px) about the median of the block's high-state images of that camera; "
                f"listed sequences, and those {CENTRE_FLAG_PX:g} px or more off. The right eye's backlash state shows "
                "mainly here (Seitah Overlook, 27 Oct).", "",
                "| block | sol | sequence | camera | images | listed | f pattern | dcx | dcy | d f |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in cen:
            out.append(f"| {s['block']} | {s['sol']} | {s['sequence']} | {s['group']} | {s['images']} | `{s['listed_pattern']}` | "
                       f"`{s['pattern']}` | {s['dcx_median_px']:+.2f} | {(s['dcy_median_px'] or 0.0):+.2f} | "
                       + (f"{s['d_median_pct']:+.2f} %" if s["d_median_pct"] is not None else "-") + " |")
    for title, key in (("To add to the found list", "add"), ("To take off (listed, found high)", "remove"),
                       ("To review", "review")):
        out += ["", f"## {title} ({len(p[key])})", ""]
        for r in p[key][:400]:
            nm = r.get("stem") or r.get("name") or "-"
            out.append(f"- {r['block']} sol {r['sol']} {r['sequence']} {r['group']}: {nm} - {r['why']}")
        if len(p[key]) > 400:
            out.append(f"- ... {len(p[key]) - 400} more in found_list_proposal.txt")
    if res.get("approach"):
        out += ["", "## Focal ratio by how the motors came into place", "",
                "From the previous frame of the same eye (within 30 min; frames outside the blocks are not seen). "
                "`d` about the block's high line (110 mm: about the block's median).  zoom list: on the low "
                "(zoom) backlash list; focus list: on the focus-from-below list.", "",
                "| zoom | eye | zoom | focus | zoom list | focus list | n | d median | d mean +- se |",
                "|---|---|---|---|---|---|---|---|---|"]
        for a in res["approach"]:
            se = f" +- {a['d_se_pct']:.3f}" if a.get("d_se_pct") is not None else ""
            out.append(f"| {a['zoom']} | {a['eye']} | {a['zoom_dir']} | {a['focus_dir']} | {'yes' if a['listed'] else '-'} "
                       f"| {'yes' if a['focus_below'] else '-'} | {a['n']} | {a['d_median_pct']:+.3f} % | "
                       f"{a['d_mean_pct']:+.3f}{se} % |")
    fp = res.get("focus_list_proposal") or {}
    if fp:
        out += ["", f"## Focus-from-below list: frames whose focus moved up >= {FOCUS_MOVE_MIN:g} counts, not listed "
                f"({len(fp.get('add', []))}); listed frames whose focus moved down ({len(fp.get('check', []))})", ""]
        for r in (fp.get("add") or [])[:200]:
            out.append(f"- add? {r['block']} sol {r['sol']} {r['sequence']} {r['group']}: {r['stem']} (focus "
                       f"{r['dfocus']:+.0f}, zoom {r['dzoom']:+.0f})")
        for r in (fp.get("check") or [])[:200]:
            out.append(f"- check: {r['block']} sol {r['sol']} {r['sequence']} {r['group']}: {r['stem']} (focus "
                       f"{r['dfocus']:+.0f}, zoom {r['dzoom']:+.0f})")
    out += ["", "## Undecided and unfitted sequences", ""]
    und = [s for s in res["sequences"] if s["verdict"] in ("undecided", "no fit")]
    for s in und[:300]:
        out.append(f"- {s['block']} sol {s['sol']} {s['sequence']} {s['group']}: {s['images']} images `{s['pattern']}`"
                   f" (listed `{s['listed_pattern']}`) - {s['verdict']}")
    return out


def find_blocks(root: PathLike, variant: str = "", sites: Optional[Iterable[str]] = None) -> Dict[str, Path]:
    """The ``*_zcam`` WORK folders under ``root`` with a ``zcam_focus_states.json`` in the variant's project
    (``sites``: only those whose folder name contains one of these site names)."""
    from .workdir import project_dir
    out = {}
    for w in sorted(Path(root).glob("mars2020_sol_*_zcam")):
        if sites and not any(f"_{s}_zcam" in w.name for s in sites):
            continue
        if (project_dir(w, variant or None) / "zcam_focus_states.json").is_file():
            out[w.name] = w
    return out
