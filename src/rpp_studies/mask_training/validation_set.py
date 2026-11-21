"""
A validation set for the rover-mask model from frames no model has trained on (0.99.21; Christian, 17 Nov 2026:
"prepare 100 images from the blocks that are not in the training set. Make sure these are representative and span the
cameras, time of day and mission sol. I will spot clean the auto generated masks ... they will become a new validation
dataset ... run inferences with v3, 4 and 5 and make an educated guess of what their true masks should be").

Why: every comparison so far scored the models on held-out frames of a later training run, most of which the released
model had trained on; the fine-tuned models start from its weights.  A set kept out of every training is the only fair
test (project doc mask-v5-and-v3-2026-11-17).

    python -m rpp_studies.mask_training.validation_set --blocks D:\\rover_blocks --training D:\\masks\\masks_training_set_v8
        --pds D:\\data\\m2020 --out D:\\masks\\validation_set_v1 --models rpp_mask_v3 <v4.2 .pt> <v5 .pt>
    ... --select-only        the selection only (selection.csv; no image is read, no GPU)

1. Candidates: every image of the aligned blocks under ``--blocks`` (``mars2020_sol_*\\colmap\\project.json`` and
   Curiosity's ``msl_sol_*\\manifest.json``).
2. Out: frames of the training set, and - unless a camera runs short - their stereo partners and every frame of a
   sequence the training set has a frame of at the same station and sol (the same scene and rover pose).
3. Selection (:func:`select`): a quota per camera (:data:`QUOTAS`; Mastcam-Z by zoom), and within a camera the frames
   spread over mission sol and local time (farthest point first), one frame per station and sequence, the two eyes
   balanced.
4. Images as the training set's (:func:`rpp_studies.mask_training.dataset.regen_config`: the pipeline's linear 8-bit
   image, full detector frame, not undistorted; Curiosity: the block's image).
5. The guess: the mean of the models' terrain probabilities above 0.5 (:func:`guess_mask`), invalid pixels excluded.
   Where the models disagree is written beside it: that is where to look first.

Written to ``--out``: ``images\\<stem>.png`` (RGB, the guess in alpha), ``masks\\<stem>.png`` (255 = terrain; the file
to clean), ``masks_auto\\`` (the guess as made, for the record), ``review\\NNN_<stem>.jpg`` (most disagreement first:
the excluded area tinted orange, the disagreement magenta), ``manifest.csv`` and ``README.txt``.  Keep the folder out
of every training set.

The guess leans toward the models that made it: a mask cleaned only where the models disagree favors them.  The
manifest's ``disagree_pct`` and the review order are a help, not a limit - look at every frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# frames per camera for a set of 100 (scaled to --n).  Mastcam-Z by zoom; MSL_N / MSL_F: Curiosity's Navcams and front
# Hazcams.  A camera with too few candidates gives its remainder to the others.
QUOTAS = {"N": 28, "Z026": 3, "Z034": 9, "Z048": 4, "Z063": 5, "Z079": 4, "Z100": 3, "Z110": 6, "F": 10, "R": 6,
          "MSL_N": 16, "MSL_F": 6}
CAMERA_NAMES = {"N": "Navcam", "F": "Front Hazcam", "R": "Rear Hazcam", "MSL_N": "Curiosity Navcam",
                "MSL_F": "Curiosity front Hazcam", "MSL_R": "Curiosity rear Hazcam"}
GUESS_THRESHOLD = 0.5             # on the mean terrain probability
UNSURE_BAND = 0.3                 # |mean - 0.5| below this counts as unsure
REVIEW_LONG_SIDE = 1600
_MSL = re.compile(r"^(?P<fam>[NFR])(?P<eye>[LR])(?P<side>[AB])_(?P<sclk>\d{9})(?P<prod>[A-Z0-9]{3})_"
                  r"(?P<samp>[A-Z])(?P<site>\d{3})(?P<drive>\d{4})(?P<seq>[A-Z]{4}\d{5})M(?P<ver>\d)$", re.I)
_LMST = re.compile(r"M(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)")


def lmst_hours(text: Any) -> Optional[float]:
    """Local mean solar time in hours from a label string (``Sol-01608M13:37:59.331``) or a number of hours."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = _LMST.search(str(text))
    return None if m is None else int(m.group(1)) + int(m.group(2)) / 60.0 + float(m.group(3)) / 3600.0


def look_elevation_deg(R_w2c: Any) -> Optional[float]:
    """The elevation of the optical axis above the horizon from a world-to-camera rotation (world z up; the camera
    looks along its +z): negative looks down, at the rover's deck or the ground near it."""
    try:
        z = float(np.asarray(R_w2c, float).reshape(3, 3)[2, 2])
    except (TypeError, ValueError):
        return None
    return float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))


def frame_key(stem: str) -> Optional[Dict[str, Any]]:
    """What a product name says about a frame: {"stem", "mission", "stratum" (a key of :data:`QUOTAS`), "eye", "sol",
    "pair" (the same for the two eyes of a stereo pair), "group" (station, sol and sequence: the same scene and rover
    pose), "station", "base" (the stem without its version)}.  None for a name of neither mission."""
    stem = Path(stem).stem
    m = _MSL.match(stem)
    if m:
        st = f"{int(m['site']):03d}_{int(m['drive']):04d}"
        return {"stem": stem, "mission": "MSL", "stratum": "MSL_" + m["fam"].upper(), "eye": m["eye"].upper(),
                "sol": None, "pair": f"MSL:{m['fam'].upper()}{m['side'].upper()}:{m['sclk']}",
                "group": f"MSL:{st}:{m['seq'].upper()}", "station": "MSL:" + st, "base": stem[:-1].upper()}
    from rpp.filenames import parse_filename
    try:
        f = parse_filename(stem)
    except ValueError:
        return None
    if f.family not in ("N", "F", "R", "Z"):
        return None
    stratum = f"Z{f.zoom_mm:03d}" if f.family == "Z" and f.zoom_mm is not None else f.family
    st = f"{f.site}_{f.drive}"
    return {"stem": stem, "mission": "M2020", "stratum": stratum, "eye": f.eye, "sol": f.sol,
            "pair": f"M20:{f.family}:{f.sclk_key}", "group": f"M20:{st}:{f.sol}:{f.sequence}", "station": "M20:" + st,
            "base": stem[:52].upper()}


def training_keys(names: Iterable[str]) -> Dict[str, set]:
    """The training set's frames as {"base", "pair", "group"} sets (see :func:`frame_key`)."""
    out: Dict[str, set] = {"base": set(), "pair": set(), "group": set()}
    for n in names:
        k = frame_key(n)
        if k is None:
            out["base"].add(Path(n).stem.upper())
            continue
        for key in out:
            out[key].add(k[key])
    return out


def independence(key: Dict[str, Any], train: Dict[str, set]) -> int:
    """0: the frame is in the training set; 1: its stereo partner or another frame of its sequence at the same station
    and sol is; 2: neither."""
    if key["base"] in train["base"] or key["stem"].upper() in train["base"]:
        return 0
    return 1 if (key["pair"] in train["pair"] or key["group"] in train["group"]) else 2


def block_candidates(blocks_roots: Any) -> List[Dict[str, Any]]:
    """Every image of the aligned blocks below the folder(s) ``blocks_roots``: :func:`frame_key` plus {"sol", "lmst_h",
    "sun_el_deg", "look_el_deg", "block", "source" (Curiosity: the block's image file; Mars 2020: None, the image comes
    from the archive)}.  A frame in several blocks counts once."""
    roots = [Path(blocks_roots)] if isinstance(blocks_roots, (str, Path)) else [Path(r) for r in blocks_roots]
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for root in roots:
        _block_candidates(root, out, seen)
    return out


def _block_candidates(root: Path, out: List[Dict[str, Any]], seen: set) -> None:
    projects = sorted(set(root.glob("mars2020_sol_*/colmap/project.json")) | set(root.glob("mars2020_sol_*/project.json")))
    for pj in projects:
        try:
            images = json.loads(pj.read_text(encoding="utf-8")).get("images") or []
        except (OSError, ValueError):
            continue
        for r in images:
            k = frame_key(r.get("stem") or r.get("name") or "")
            if k is None or k["stem"] in seen:
                continue
            seen.add(k["stem"])
            out.append(dict(k, lmst_h=lmst_hours(r.get("lmst")), sun_el_deg=r.get("solar_elevation_deg"),
                            look_el_deg=look_elevation_deg(r.get("prior_R_w2c")),
                            block=(pj.parent.parent if pj.parent.name == "colmap" else pj.parent).name, source=None))
    for mj in sorted(root.glob("msl_sol_*/manifest.json")):
        try:
            images = json.loads(mj.read_text(encoding="utf-8")).get("images") or []
        except (OSError, ValueError):
            continue
        for r in images:
            k = frame_key(r.get("stem") or "")
            if k is None or k["stem"] in seen:
                continue
            seen.add(k["stem"])
            src = mj.parent / "images" / str(r.get("image") or (k["stem"] + ".png"))
            out.append(dict(k, sol=r.get("sol"), lmst_h=lmst_hours(r.get("lmst_h")), sun_el_deg=None,
                            look_el_deg=look_elevation_deg(r.get("R_w2c")), block=mj.parent.name, source=str(src)))


def scaled_quotas(n: int, quotas: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """:data:`QUOTAS` scaled to ``n`` frames (largest remainders first)."""
    q = dict(quotas or QUOTAS)
    tot = sum(q.values())
    raw = {k: v * n / tot for k, v in q.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (out[k] - raw[k], k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def _spread(cands: List[Dict[str, Any]], n: int, taken: Sequence[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """``n`` of ``cands`` spread over sol, local time and pointing: farthest point first in (sol, LMST, half the
    elevation of the optical axis), each scaled to the range over ``cands`` and ``taken``; a station already taken
    counts half as far, a sequence already taken is passed over while others remain, the eye in the majority counts 0.9
    as far."""
    if n <= 0 or not cands:
        return []
    both = list(cands) + list(taken)

    def col(key, weight=1.0):
        v = np.array([np.nan if c.get(key) is None else float(c[key]) for c in both], float)
        if np.all(np.isnan(v)):
            return np.zeros(len(both))
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        rng = v.max() - v.min()
        return weight * (v - v.min()) / rng if rng > 0 else np.zeros(len(both))
    XA = np.c_[col("sol"), col("lmst_h"), col("look_el_deg", 0.5)]
    X, XT = XA[: len(cands)], XA[len(cands):]
    order = sorted(range(len(cands)), key=lambda i: (X[i, 0], X[i, 1], cands[i]["stem"]))
    chosen: List[int] = []
    groups = {c["group"] for c in taken}
    stations = {c["station"] for c in taken}
    eyes = {"L": 0, "R": 0}
    for c in taken:
        if c.get("eye") in eyes:
            eyes[c["eye"]] += 1
    dmin = np.full(len(cands), np.inf)
    for t in XT:
        dmin = np.minimum(dmin, np.linalg.norm(X - t, axis=1))
    while len(chosen) < min(n, len(cands)):
        best, best_score = None, -1.0
        fresh = any(i not in chosen and cands[i]["group"] not in groups for i in order)
        for i in order:
            if i in chosen or (fresh and cands[i]["group"] in groups):
                continue
            d = 1.0 if not np.isfinite(dmin[i]) else float(dmin[i])
            if cands[i]["station"] in stations:
                d *= 0.5
            e = cands[i].get("eye")
            if e in eyes and eyes[e] > eyes["R" if e == "L" else "L"]:
                d *= 0.9
            if d > best_score + 1e-12:
                best, best_score = i, d
        chosen.append(best)
        c = cands[best]
        groups.add(c["group"])
        stations.add(c["station"])
        if c.get("eye") in eyes:
            eyes[c["eye"]] += 1
        dmin = np.minimum(dmin, np.linalg.norm(X - X[best], axis=1))
    return [cands[i] for i in chosen]


def select(cands: Sequence[Dict[str, Any]], train: Dict[str, set], n: int = 100,
           quotas: Optional[Dict[str, int]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The validation frames: per camera its quota (:func:`scaled_quotas`), spread over sol and local time
    (:func:`_spread`), frames independent of the training set first (``independence`` 2), then - only when a camera runs
    short - frames whose stereo partner or sequence is in it (1, marked ``near_training``).  Quota a camera cannot fill
    goes to the cameras with independent frames to spare, largest first.  Returns (frames, report)."""
    q = scaled_quotas(n, quotas)
    by: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    dropped = 0
    for c in cands:
        lvl = independence(c, train)
        if lvl == 0:
            dropped += 1
            continue
        by.setdefault(c["stratum"], {1: [], 2: []})[lvl].append(c)
    picked: Dict[str, List[Dict[str, Any]]] = {}
    for s, want in q.items():
        pool = by.get(s, {1: [], 2: []})
        got = _spread(pool[2], want)
        if len(got) < want:
            near = _spread(pool[1], want - len(got), got)
            got += [dict(c, near_training=True) for c in near]
        picked[s] = got
    short = n - sum(len(v) for v in picked.values())
    while short > 0:                                    # the remainder to the cameras with independent frames left
        spare = sorted(((len(by.get(s, {2: []})[2]) - sum(1 for c in picked[s] if not c.get("near_training")), s)
                        for s in q), reverse=True)
        if not spare or spare[0][0] <= 0:
            break
        s = spare[0][1]
        have = {c["stem"] for c in picked[s]}
        more = _spread([c for c in by[s][2] if c["stem"] not in have], 1, picked[s])
        if not more:
            break
        picked[s] += more
        short -= 1
    frames = [dict(c, near_training=bool(c.get("near_training"))) for s in q for c in picked[s]]
    rep = {"wanted": n, "selected": len(frames), "candidates": len(cands), "in_training": dropped,
           "by_camera": {s: {"quota": q[s], "selected": len(picked[s]),
                             "near_training": sum(1 for c in picked[s] if c.get("near_training")),
                             "independent_candidates": len(by.get(s, {2: []})[2]),
                             "near_candidates": len(by.get(s, {1: []})[1])} for s in q}}
    return frames, rep


def guess_mask(probs: Sequence[np.ndarray], thresholds: Sequence[float], valid: Optional[np.ndarray] = None
               ) -> Dict[str, Any]:
    """The educated guess from the models' terrain probabilities: {"mask" (uint8, 255 = terrain: the mean probability
    above :data:`GUESS_THRESHOLD`, invalid pixels 0), "mean" (float32), "disagree" (bool: the models' own masks differ),
    "unsure" (bool: disagree, or the mean within :data:`UNSURE_BAND` of 0.5), "disagree_pct", "unsure_pct",
    "rover_pct" (of the valid pixels)}."""
    P = np.stack([np.asarray(p, np.float32) for p in probs])
    mean = P.mean(axis=0)
    own = np.stack([p > float(t) for p, t in zip(P, thresholds)])
    v = np.ones(mean.shape, bool) if valid is None else np.asarray(valid, bool)
    disagree = (own.any(axis=0) != own.all(axis=0)) & v
    unsure = (disagree | (np.abs(mean - 0.5) < UNSURE_BAND)) & v
    terrain = (mean > GUESS_THRESHOLD) & v
    nv = max(int(v.sum()), 1)
    return {"mask": np.where(terrain, 255, 0).astype(np.uint8), "mean": mean, "disagree": disagree, "unsure": unsure,
            "disagree_pct": 100.0 * float(disagree.sum()) / nv, "unsure_pct": 100.0 * float(unsure.sum()) / nv,
            "rover_pct": 100.0 * float((~terrain & v).sum()) / nv}


def review_panel(rgb: np.ndarray, mask: np.ndarray, disagree: np.ndarray, long_side: int = REVIEW_LONG_SIDE
                 ) -> np.ndarray:
    """The frame beside itself with the guess drawn in (RGB uint8): the excluded area tinted orange, the pixels the
    models disagree on magenta.  The linear image is shown with a gamma so the shadows can be read."""
    import cv2
    h, w = rgb.shape[:2]
    s = min(1.0, long_side / float(max(h, w)))
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    im = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    im = np.power(np.clip(im / max(float(np.percentile(im, 99.5)), 1e-3), 0, 1), 1 / 2.2)
    ex = cv2.resize((mask == 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    dg = cv2.resize(disagree.astype(np.uint8), size, interpolation=cv2.INTER_AREA) > 0
    ov = im.copy()
    ov[ex] = 0.55 * ov[ex] + 0.45 * np.array([1.0, 0.55, 0.0], np.float32)
    ov[dg] = np.array([1.0, 0.0, 1.0], np.float32)
    gap = np.ones((size[1], 12, 3), np.float32)
    return (np.clip(np.hstack([im, gap, ov]), 0, 1) * 255).astype(np.uint8)


def training_names(training: Path) -> List[str]:
    """The stems of a training set: its ``masks`` folder (else ``images``, else the folder itself)."""
    t = Path(training)
    for d in (t / "masks", t / "images", t):
        if d.is_dir():
            names = [p.stem for p in d.iterdir() if p.suffix.lower() == ".png" and not p.name.endswith(".tmp.png")]
            if names:
                return names
    return []


def _load_m2020(stem: str, index: Dict[str, Path], cfg: Dict, waypoints) -> Tuple[Optional[np.ndarray], Any, str]:
    import warnings
    from rpp.image import RPPImage
    from rpp_studies.mask_training.dataset import find_pds
    path, status = find_pds(stem, index)
    if path is None:
        return None, None, "not in the archive"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = RPPImage(str(path), cfg, waypoints)
    return np.asarray(im.image_int8), np.asarray(im.mask_valid) > 0, status


def _load_file(path: str) -> Tuple[Optional[np.ndarray], Any, str]:
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None, None, "image not read"
    if a.ndim == 2:
        a = np.dstack([a, a, a])
    elif a.shape[2] == 2:
        a = np.dstack([a[..., 0]] * 3)
    rgb = a[..., :3][..., ::-1].copy()
    if rgb.dtype != np.uint8:
        rgb = (rgb.astype(np.float32) / 257.0).astype(np.uint8)
    return rgb, rgb.max(axis=2) > 0, "ok"


def write_selection(frames: Sequence[Dict[str, Any]], path: Path) -> None:
    cols = ["stem", "camera", "stratum", "eye", "sol", "lmst_h", "sun_el_deg", "look_el_deg", "station", "group",
            "block", "near_training"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in frames:
            w.writerow(dict(c, camera=camera_name(c["stratum"]), **_rounded(c)))


def _rounded(c: Dict[str, Any]) -> Dict[str, Any]:
    return {k: "" if c.get(k) is None else round(float(c[k]), d) for k, d in (("lmst_h", 2), ("look_el_deg", 1),
                                                                               ("sun_el_deg", 1))}


def camera_name(stratum: str) -> str:
    return CAMERA_NAMES.get(stratum) or (f"Mastcam-Z {int(stratum[1:])} mm" if stratum[:1] == "Z" else stratum)


def report_text(rep: Dict[str, Any]) -> List[str]:
    L = [f"{rep['selected']} of {rep['wanted']} frames from {rep['candidates']} block images "
         f"({rep['in_training']} of them are training frames)",
         f"{'camera':26s} {'quota':>5s} {'taken':>5s} {'near':>5s}   candidates independent / near the training set"]
    for s, r in rep["by_camera"].items():
        L.append(f"{camera_name(s):26s} {r['quota']:5d} {r['selected']:5d} {r['near_training']:5d}   "
                 f"{r['independent_candidates']} / {r['near_candidates']}")
    return L


def available(cands: Sequence[Dict[str, Any]], index: Optional[Dict[str, Path]]) -> List[Dict[str, Any]]:
    """The candidates whose image can be made: Mars 2020 frames with a product in the archive index (all of them when
    there is no index), Curiosity frames whose block image exists."""
    from rpp_studies.mask_training.dataset import find_pds
    out = []
    for c in cands:
        if c["mission"] == "M2020":
            if index is None or find_pds(c["stem"], index)[0] is not None:
                out.append(c)
        elif c.get("source") and Path(c["source"]).is_file():
            out.append(c)
    return out


def build(frames: Sequence[Dict[str, Any]], out: Path, models: Sequence[str], index: Optional[Dict[str, Path]] = None,
          device: str = "auto", force: bool = False) -> Dict[str, Any]:
    """The images, the models' guess and the review panels of the selected frames (see the module text).  A frame
    already written is kept unless ``force``.  Returns {"written", "kept", "failed": {stem: why}, "rows"}."""
    import cv2
    from rpp.mask.infer import get_model, predict_probability
    from rpp_studies.mask_training.dataset import regen_config
    out = Path(out)
    for d in ("images", "masks", "masks_auto", "review"):
        (out / d).mkdir(parents=True, exist_ok=True)
    loaded = [get_model(m, device) for m in models]
    labels = [Path(m).stem if Path(m).suffix else str(m) for m in models]
    cfg, waypoints = None, None
    if any(c["mission"] == "M2020" for c in frames):
        if index is None:
            raise SystemExit("the archive index is needed for the Mars 2020 frames (--pds)")
        from rpp.waypoints import load_waypoints
        cfg, waypoints = regen_config(), load_waypoints()
    rows, failed, kept = [], {}, 0
    for k, c in enumerate(frames, 1):
        stem = c["stem"]
        mask_file = out / "masks" / f"{stem}.png"
        try:
            if c["mission"] == "M2020":
                rgb, valid, status = _load_m2020(stem, index, cfg, waypoints)
            else:
                rgb, valid, status = _load_file(c["source"])
            if rgb is None:
                failed[stem] = status
                continue
            probs = [predict_probability(m, card, rgb) for m, card in loaded]
            g = guess_mask(probs, [float(card["threshold"]) for _, card in loaded], valid)
            row = dict(c, camera=camera_name(c["stratum"]), width=rgb.shape[1], height=rgb.shape[0], source_status=status,
                       rover_pct=round(g["rover_pct"], 2), disagree_pct=round(g["disagree_pct"], 3),
                       unsure_pct=round(g["unsure_pct"], 3))
            for lab, p, (_, card) in zip(labels, probs, loaded):
                row[f"rover_pct {lab}"] = round(100.0 * float(((p <= float(card["threshold"])) & valid).sum())
                                                / max(int(valid.sum()), 1), 2)
            rows.append(row)
            cv2.imwrite(str(out / "masks_auto" / f"{stem}.png"), g["mask"])
            if mask_file.is_file() and not force:       # a mask already there may have been cleaned: never replaced
                kept += 1
            else:
                cv2.imwrite(str(mask_file), g["mask"])
                cv2.imwrite(str(out / "images" / f"{stem}.png"), np.dstack([rgb[..., ::-1], g["mask"]]))
            row["_panel"] = review_panel(rgb, g["mask"], g["disagree"])
        except Exception as e:                           # noqa: BLE001 - one frame never stops the others
            failed[stem] = f"{type(e).__name__}: {e}"[:200]
        if k % 10 == 0 or k == len(frames):
            print(f"  {k}/{len(frames)} frames", flush=True)
    rows.sort(key=lambda r: -r["disagree_pct"])
    for old in (out / "review").glob("*.jpg"):
        old.unlink()
    for rank, r in enumerate(rows, 1):
        r["review_rank"] = rank
        cv2.imwrite(str(out / "review" / f"{rank:03d}_{r['stem']}.jpg"), r.pop("_panel")[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    cols = ["review_rank", "stem", "camera", "stratum", "eye", "sol", "lmst_h", "sun_el_deg", "look_el_deg", "station", "block",
            "near_training", "width", "height", "rover_pct", "disagree_pct", "unsure_pct"] + \
           [f"rover_pct {lab}" for lab in labels] + ["source_status"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, **_rounded(r)))
    return {"written": len(rows) - kept, "kept": kept, "failed": failed, "rows": rows, "models": labels}


README = """Rover-mask validation set ({n} frames; made {when} by rpp_studies.mask_training.validation_set, RPP {version})

Frames of the aligned blocks that are not in the training set {training}.
The masks are a guess: the mean terrain probability of {models} above 0.5.  They are to be cleaned by hand.

  images\\<stem>.png      the image as the training set's (RGB; the guess in alpha)
  masks\\<stem>.png       255 = terrain, 0 = excluded: THE FILE TO CLEAN (a rerun never replaces it; --force does)
  masks_auto\\<stem>.png  the guess as made, for the record
  review\\NNN_<stem>.jpg  the frame and the guess (excluded orange, the models disagree magenta), most disagreement first
  manifest.csv           one row per frame: camera, sol, local time, the share excluded, the share the models disagree on
  selection.csv          the selection and why (near_training: its stereo partner or sequence is in the training set)

Keep this folder out of every training set.  The guess leans toward the models that made it: look at every frame,
not only where they disagree.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", required=True, nargs="+",
                    help="the folder(s) of aligned blocks (D:\\rover_blocks; D:\\rover_frame_eod_early\\scapes for the "
                         "deck stations, whose Navcam frames look at the rover)")
    ap.add_argument("--training", required=True, nargs="+",
                    help="the training set(s) whose frames are left out (their masks\\ folders are read)")
    ap.add_argument("--out", required=True, help="the validation set's folder (not inside a training set)")
    ap.add_argument("--pds", default=None, help="the Mars 2020 archive the images are made from (default RPP_PDS)")
    ap.add_argument("--models", nargs="+", default=["rpp_mask_v5"],
                    help="registry names or checkpoints whose mean probability is the guess")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--select-only", action="store_true", help="write selection.csv and stop (no image, no GPU)")
    ap.add_argument("--force", action="store_true", help="replace masks\\ files already there (cleaned masks are lost)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    import datetime
    from rpp import __version__, env as rpp_env
    out = Path(a.out)
    for t in a.training:
        if out.resolve() == Path(t).resolve() or Path(t).resolve() in out.resolve().parents:
            raise SystemExit(f"--out {out} is inside the training set {t}: it would be trained on")
    names = [n for t in a.training for n in training_names(Path(t))]
    if not names:
        raise SystemExit(f"no training frames found in {a.training}: nothing could be left out")
    cands = block_candidates(a.blocks)
    if not cands:
        raise SystemExit(f"no block images found below {a.blocks}")
    pds = a.pds or rpp_env.get("RPP_PDS")
    index = None
    if pds and Path(pds).is_dir():
        from rpp_studies.mask_training.dataset import index_pds
        print(f"indexing {pds} ...", flush=True)
        index = index_pds(pds)
        print(f"  {len(index)} products", flush=True)
    elif not a.select_only:
        raise SystemExit(f"the Mars 2020 archive was not found ({pds}): --pds")
    n_all = len(cands)
    cands = available(cands, index)
    print(f"{len(cands)} of {n_all} block images can be made (in the archive, or a Curiosity block image on disk)")
    frames, rep = select(cands, training_keys(names), a.n)
    out.mkdir(parents=True, exist_ok=True)
    write_selection(frames, out / "selection.csv")
    text = [f"training set: {len(names)} frames"] + report_text(rep)
    print("\n".join(text))
    (out / "selection.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    if a.select_only:
        print(f"selection only: {out / 'selection.csv'}")
        return 0
    res = build(frames, out, a.models, index, a.device, a.force)
    (out / "README.txt").write_text(README.format(
        n=len(res["rows"]), when=datetime.datetime.now().isoformat(timespec="minutes"), version=__version__,
        training=", ".join(a.training), models=", ".join(res["models"])), encoding="utf-8")
    print(f"{res['written']} masks written, {res['kept']} kept as they were, {len(res['failed'])} failed -> {out}")
    for s, why in res["failed"].items():
        print(f"  failed: {s}: {why}")
    return 0 if res["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
