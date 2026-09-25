"""
Frame audit (v0p10; statistics v0p12): run a trained mask model over the
training set and rank frames by agreement with their labels, to find bad
labels and bad images.

v0p12: ranked by the % of the frame that is wrong (``error_pct``) by default.
IoU is meaningless for frames without terrain (sky, calibration target, rover
deck): a single false pixel gives 0, so the IoU ranking was all such frames.
Also recorded: ``interior_error_pct`` (errors away from label edges = wrong
regions, not boundary jitter) and ``confident_error_pct`` (the model is sure
and the label disagrees).

The model cannot beat its labels, so the frames it disagrees with most are the
first ones to look at: a coarse or wrong mask, a mask exported for the wrong
product, an image regenerated from a different version, a frame with no
terrain.  Scores are whole-frame, full-resolution, exactly as in production
(``infer_mask``: the card's canvas, threshold and quad rule; no dilation).

Training frames score optimistically (the model has seen them), so the split
column matters: a *train* frame near the bottom of the list is a strong
candidate for a label problem.

    rows = audit_frames(items, ckpt, image_dirs=("images",))     # CSV beside the checkpoint
    view = rank_rows(rows, "interior_error_pct", split="train")   # re-rank without re-running
    pages = render_worst(view, ckpt, n=24)                        # PNG panels of the worst frames
"""
from __future__ import annotations

import csv
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

FIELDS = ("rank", "name", "split", "variant", "camera", "iou", "error_pct", "interior_error_pct",
          "confident_error_pct", "missed_pct", "false_pct", "terrain_truth_pct", "terrain_pred_pct",
          "band_px", "width", "height", "status", "image", "mask")
FLOATS = ("iou", "error_pct", "interior_error_pct", "confident_error_pct", "missed_pct", "false_pct",
          "terrain_truth_pct", "terrain_pred_pct")
# v0p12 ranking statistics (all % of the frame, larger = worse, except iou)
SORT_KEYS = {
    "error_pct": "all wrong pixels (missed + false): what the frame adds to the training error",
    "interior_error_pct": "wrong pixels farther than band_px from any label edge: wrong or missing regions, "
                          "not boundary jitter",
    "confident_error_pct": "pixels where the model is confident (P > 0.9 or < 0.1) and the label disagrees: "
                           "the best single pointer to a wrong label",
    "iou": "IoU of the terrain class (ascending); frames with little terrain are excluded by min_terrain_pct",
}


def frame_stats(truth: np.ndarray, prob: np.ndarray, threshold: float, band_frac: float = 0.004,
                confident: float = 0.9) -> Dict[str, float]:
    """
    Per-frame agreement statistics at full resolution (``truth`` bool, ``prob`` float).
    ``band_px`` = max(3, band_frac x long side): 20 px on a 5120 px frame, 7 px at 1648.
    """
    from .monitor import panel_stats
    iou, missed, false, diff = panel_stats(truth, prob, threshold)
    h, w = truth.shape
    band = max(3, int(round(band_frac * max(h, w))))
    t8 = truth.astype(np.uint8)
    edge = cv2.morphologyEx(t8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    near = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band + 1,) * 2)) > 0
    wrong = diff != 0
    conf = (truth & (prob < 1.0 - confident)) | (~truth & (prob > confident))
    return {"iou": iou, "missed_pct": missed, "false_pct": false, "error_pct": missed + false,
            "interior_error_pct": 100.0 * float((wrong & ~near).mean()),
            "confident_error_pct": 100.0 * float(conf.mean()),
            "terrain_truth_pct": 100.0 * float(truth.mean()),
            "terrain_pred_pct": 100.0 * float((prob > threshold).mean()), "band_px": band}


def rank_rows(rows: Sequence[Dict[str, Any]], sort_by: str = "error_pct", min_terrain_pct: Optional[float] = None,
              split: Optional[str] = None, include_unscored: bool = True) -> List[Dict[str, Any]]:
    """
    Re-rank audit rows (e.g. from ``read_audit_csv``) without re-running the model.
    ``sort_by``: a key of ``SORT_KEYS``.  ``min_terrain_pct``: drop frames whose
    label has less terrain than this (% of the frame; default 1 for "iou", where
    frames without terrain are meaningless, else none).  ``split``: "train" /
    "val" / None.  Unreadable or mismatched frames stay first.  Returns copies
    with a new ``rank``.
    """
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {list(SORT_KEYS)}")
    ok = [r for r in rows if r.get("status") in (None, "ok")]
    if ok and ok[0].get(sort_by) is None:
        raise KeyError(f"these audit rows have no {sort_by!r} (audited before v0p12): rerun audit_frames")
    if min_terrain_pct is None and sort_by == "iou":
        min_terrain_pct = 1.0
    keep = [dict(r) for r in ok
            if (split is None or r.get("split") == split)
            and (min_terrain_pct is None or (r.get("terrain_truth_pct") or 0.0) >= min_terrain_pct)]
    if sort_by == "iou":
        keep.sort(key=lambda r: (r["iou"], -r["error_pct"]))
    else:
        keep.sort(key=lambda r: (-r[sort_by], -r["error_pct"]))
    bad = [dict(r) for r in rows if r.get("status") not in (None, "ok")] if include_unscored else []
    out = bad + keep
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out


def _read_pair(it) -> tuple:
    # same rule as training: RGB of the image, labels from the separate masks/ file (alpha ignored)
    im = cv2.imread(it.image, cv2.IMREAD_COLOR)
    ms = cv2.imread(it.mask, cv2.IMREAD_GRAYSCALE)
    return it, im, ms


def split_labels(items: Sequence, card: Dict[str, Any]) -> Dict[str, str]:
    """mask path -> "train" / "val", reproducing the training split from the card."""
    from .train import grouped_split
    tr_cfg = card.get("training") or {}
    tr, va = grouped_split(list(items), float(tr_cfg.get("val_ratio", 0.1)), int(tr_cfg.get("seed", 42)))
    lab = {it.mask: "train" for it in tr}
    lab.update({it.mask: "val" for it in va})
    return lab


def audit_frames(items: Sequence, checkpoint: PathLike, out_csv: Optional[PathLike] = "auto",
                 image_dirs: Optional[Sequence[str]] = ("images",), which: str = "all",
                 device: str = "auto", threshold: Optional[float] = None, amp: bool = True,
                 read_workers: int = 4, limit: Optional[int] = None, sort_by: str = "error_pct",
                 progress_every: int = 250, band_frac: float = 0.004) -> List[Dict[str, Any]]:
    """
    Score every frame of ``items`` (``scan_dataset`` output) with ``checkpoint``.

    * ``image_dirs``: keep only frames from these sub-folders.  ``("images",)``
      (default) scores each label once on its production image; None = all
      (``images_variable`` too, about twice the time).
    * ``which``: "all" | "train" | "val" — the split is rebuilt from the card
      (same val_ratio and seed as training, grouped by mask).
    * ``sort_by``: see ``SORT_KEYS`` (v0p12 default "error_pct"; "iou" ranked
      every frame without terrain first).  Re-rank later with :func:`rank_rows`.
    * ``band_frac``: boundary band for ``interior_error_pct`` (x long side).
    * ``out_csv``: "auto" = ``<checkpoint stem>_debug/frame_audit.csv``; None = no file.

    Returns the rows, worst first, with ``rank``.  An unreadable image or a
    size mismatch is kept with ``status`` set and ranked first (IoU -1).
    """
    import torch
    from .infer import get_model, predict_probability

    if which not in ("all", "train", "val"):
        raise ValueError("which must be 'all', 'train' or 'val'")
    if sort_by not in SORT_KEYS:
        raise ValueError(f"sort_by must be one of {list(SORT_KEYS)}")
    ckpt = Path(checkpoint)
    model, card = get_model(ckpt, device)
    thr = float(card["threshold"] if threshold is None else threshold)
    split = split_labels(items, card)
    sel = [it for it in items if getattr(it, "tile", -1) < 0
           and (image_dirs is None or Path(it.image).parent.name in image_dirs)
           and (which == "all" or split.get(it.mask) == which)]
    if limit:
        sel = sel[:int(limit)]
    if not sel:
        raise ValueError(f"no frames to audit (image_dirs={image_dirs}, which={which})")
    dev = next(model.parameters()).device
    use_amp = bool(amp and dev.type == "cuda" and torch.cuda.is_bf16_supported())

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    print(f"[mask.audit] {ckpt.name}: {len(sel)} frames (image_dirs={image_dirs}, which={which}), "
          f"threshold {thr}, {'bf16' if use_amp else 'fp32'} on {dev}")
    with ThreadPoolExecutor(max(1, int(read_workers))) as pool:
        pending: deque = deque()
        it_sel = iter(sel)
        for it in it_sel:                                    # bounded prefetch: RAM ~ 2 x workers frames
            pending.append(pool.submit(_read_pair, it))
            if len(pending) >= 2 * max(1, int(read_workers)):
                break
        k = 0
        while pending:
            it, im, ms = pending.popleft().result()
            nxt = next(it_sel, None)
            if nxt is not None:
                pending.append(pool.submit(_read_pair, nxt))
            k += 1
            name = Path(it.image).name
            row: Dict[str, Any] = {"name": name, "split": split.get(it.mask, "?"),
                                   "variant": Path(it.image).parent.name, "camera": name[:3],
                                   "image": it.image, "mask": it.mask, "status": "ok"}
            if im is None or ms is None:
                row.update(status="unreadable image" if im is None else "unreadable mask", iou=-1.0)
            elif im.shape[:2] != ms.shape[:2]:
                row.update(status=f"size mismatch {im.shape[1]}x{im.shape[0]} vs mask {ms.shape[1]}x{ms.shape[0]}",
                           iou=-1.0)
            else:
                rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    prob = predict_probability(model, card, rgb)
                row.update(frame_stats(ms > 127, prob, thr, band_frac),
                           width=int(ms.shape[1]), height=int(ms.shape[0]))
            rows.append(row)
            if progress_every and (k % progress_every == 0 or k == len(sel)):
                el = time.time() - t0
                print(f"  {k}/{len(sel)}  {el:.0f} s  (~{el / k * (len(sel) - k) / 60:.1f} min left)")

    rows = rank_rows(rows, sort_by, min_terrain_pct=0.0 if sort_by == "iou" else None)   # CSV keeps every frame
    if out_csv is not None:
        path = ckpt.parent / f"{ckpt.stem}_debug" / "frame_audit.csv" if out_csv == "auto" else Path(out_csv)
        write_audit_csv(rows, path)
        print(f"[mask.audit] -> {path}")
    return rows


def write_audit_csv(rows: Sequence[Dict[str, Any]], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def read_audit_csv(path: PathLike) -> List[Dict[str, Any]]:
    num = {"rank": int, "width": int, "height": int, "band_px": int}
    out = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if v in ("", None):
                    r[k] = None
                elif k in num:
                    r[k] = num[k](v)
                elif k in FLOATS:
                    r[k] = float(v)
            out.append(r)
    return out


def summarize(rows: Sequence[Dict[str, Any]], cuts: Sequence[float] = (1.0, 5.0, 20.0),
              min_terrain_pct: float = 1.0) -> str:
    """Per split: frames, mean / median error %, counts above error cuts, mean IoU of frames with terrain."""
    lines = []
    for sp in ("train", "val"):
        rs = [r for r in rows if r.get("split") == sp and r.get("status") in (None, "ok")]
        if not rs:
            continue
        e = np.array([r["error_pct"] for r in rs])
        iou = np.array([r["iou"] for r in rs if (r.get("terrain_truth_pct") or 0) >= min_terrain_pct])
        n_empty = sum((r.get("terrain_truth_pct") or 0) < min_terrain_pct for r in rs)
        above = ", ".join(f">{c:g}%: {int((e > c).sum())}" for c in cuts)
        lines.append(f"{sp:5s} {len(rs):5d} frames  error % mean {e.mean():.3f} median {np.median(e):.3f} ({above})  "
                     f"mean IoU {iou.mean():.4f} over {iou.size} frames with >= {min_terrain_pct:g}% terrain "
                     f"({n_empty} with less)")
    bad = [r for r in rows if r.get("status") not in (None, "ok")]
    if bad:
        lines.append(f"{len(bad)} frames not scored (unreadable / size mismatch), ranked first")
    return "\n".join(lines)


def worst_table(rows: Sequence[Dict[str, Any]], n: int = 30) -> str:
    """Plain-text table of the first ``n`` rows (err = missed + false; int = interior; conf = confident)."""
    f2 = lambda v: "-" if v is None else f"{v:.2f}"                           # noqa: E731
    head = (f"{'rank':>4}  {'err%':>6}  {'int%':>6}  {'conf%':>6}  {'miss%':>6}  {'false%':>6}  {'IoU':>5}  "
            f"{'terr%':>5}  {'split':5}  name")
    out = [head]
    for r in rows[:n]:
        if r.get("status") not in (None, "ok"):
            out.append(f"{r['rank']:>4}  {'':>52}  {r['split']:5}  {r['name']}  [{r['status']}]")
            continue
        out.append(f"{r['rank']:>4}  {r['error_pct']:6.2f}  {f2(r.get('interior_error_pct')):>6}  "
                   f"{f2(r.get('confident_error_pct')):>6}  {r['missed_pct']:6.2f}  {r['false_pct']:6.2f}  "
                   f"{r['iou']:5.3f}  {r['terrain_truth_pct']:5.1f}  {r['split']:5}  {r['name']}")
    return "\n".join(out)


def render_worst(rows: Sequence[Dict[str, Any]], checkpoint: PathLike, n: int = 24, per_page: int = 6,
                 out_dir: Optional[PathLike] = "auto", device: str = "auto", tile_h: int = 300,
                 sort_by: Optional[str] = None) -> List[Path]:
    """
    Panels (image | truth | P(terrain) | prediction − truth) of the ``n`` worst
    scored frames, ``per_page`` rows per PNG, written as
    ``worst_frames_01.png ...`` (+ ``worst_frames.txt``, one image name per
    line, e.g. for selecting them in Metashape).  ``rows`` in the order to show
    (see :func:`rank_rows`); ``sort_by`` only labels the title.  Returns the PNG paths.
    """
    import torch
    from .infer import get_model, predict_probability
    from .monitor import render_rows
    ckpt = Path(checkpoint)
    odir = ckpt.parent / f"{ckpt.stem}_debug" if out_dir == "auto" else Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    model, card = get_model(ckpt, device)
    dev = next(model.parameters()).device
    use_amp = bool(dev.type == "cuda" and torch.cuda.is_bf16_supported())
    pick = [r for r in rows if r.get("status") in (None, "ok")][:n]
    (odir / "worst_frames.txt").write_text("".join(r["name"] + "\n" for r in pick), encoding="utf-8")
    pages: List[Path] = []
    for p0 in range(0, len(pick), per_page):
        panel_rows = []
        for r in pick[p0:p0 + per_page]:
            img = cv2.cvtColor(cv2.imread(r["image"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 127
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                prob = predict_probability(model, card, img)
            panel_rows.append((img, gt, prob, f"#{r['rank']}  [{r['split']}]  err {r['error_pct']:.2f}%  {r['name']}"))
        panel, _ = render_rows(panel_rows, f"{ckpt.name}  worst frames{' by ' + sort_by if sort_by else ''}  "
                               f"{p0 + 1}-{p0 + len(panel_rows)} of {len(rows)}", float(card["threshold"]), tile_h)
        f = odir / f"worst_frames_{p0 // per_page + 1:02d}.png"
        cv2.imwrite(str(f), panel)
        pages.append(f)
    return pages
