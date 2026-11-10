"""
A/B of mask models on the same held-out frames (31 Oct 2026, before the freeze: is the v8-trained model,
``mppp_mask_v4``, better than the default ``mppp_mask_v3``?).

    python -m mppp_studies.mask_training.compare_mask_models --data D:\\masks\\masks_training_set_v8
        --split-ckpt D:\\code\\MPPP\\checkpoints\\convnext_tiny_s4_seg_20261030_v8.pt
        --models mppp_mask_v3 D:\\code\\MPPP\\checkpoints\\convnext_tiny_s4_seg_20261030_v8.pt
        --seen D:\\masks\\masks_training_set_v7 --out "D:\\code\\MPPP\\Claude outputs\\mask_ab_v3_v4"

The test frames are the held-out frames of ``--split-ckpt`` (its card's split, rebuilt as in training).  Every model
in ``--models`` (registry names, checkpoint or .safetensors paths) is scored on them with the metrics of
``make_mask_figure.py``: exclusion IoU, recall ("hardware recall": the rover pixels also excluded) and precision,
valid pixels only, pooled per camera.  ``--seen``: the training set (or its masks/ folder) of an older model; its
frames may be in that model's training data, so the table is also given for the frames NOT in it - the fair
comparison.  Writes ``mask_ab.md`` / ``mask_ab.csv`` (and ``mask_ab_frames.csv``, one row per frame and model).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def seen_names(path) -> set:
    """Mask file names of an older training set (its root with masks/, or the masks folder itself)."""
    p = Path(path)
    d = p / "masks" if (p / "masks").is_dir() else p
    return {f.name for f in d.glob("*.png")}


def items_as_trained(items, card) -> list:
    """The items as they were when the model of ``card`` trained (2 Nov 2026: the v8 set grew from 4,095 to 4,099
    masks between v4 and v4.1, and the split - a shuffle of the sorted mask names - changes with every mask added):
    the newest masks (file time) left out until the count is the card's n_train + n_val."""
    tr = card.get("training") or {}
    n = int(tr.get("n_train") or 0) + int(tr.get("n_val") or 0)
    masks = sorted({it.mask for it in items})
    if not n or len(masks) <= n:
        return list(items)
    newest = sorted(masks, key=lambda m: (Path(m).stat().st_mtime if Path(m).exists() else 0.0, m))[n:]
    drop = set(newest)
    print(f"  {len(drop)} newest masks left out to rebuild the split of {card.get('name') or 'the model'} "
          f"({len(masks)} -> {n}): {', '.join(Path(m).name for m in sorted(drop))[:300]}")
    return [it for it in items if it.mask not in drop]


def table(rows, subset_label: str) -> list:
    md = [f"### {subset_label}", "", "| camera | frames | model | IoU (exclude) | recall (exclude) | precision (exclude) |",
          "|---|---:|---|---:|---:|---:|"]
    md += [f"| {r['camera']} | {r['frames']} | {r['model']} | {r['exclusion_IoU']:.4f} | {r['hardware_recall']:.4f} | "
           f"{r['exclusion_precision']:.4f} |" for r in rows]
    return md + [""]


def pool(frame_rows, models, keep=lambda r: True):
    acc = {}
    for r in frame_rows:
        if not keep(r):
            continue
        for cam in (r["camera"], "all"):
            d = acc.setdefault((cam, r["model"]), {"frames": 0, "tp": 0, "fp": 0, "fn": 0})
            d["frames"] += 1
            for k in ("tp", "fp", "fn"):
                d[k] += r[k]
    out = []
    cams = sorted({c for c, _ in acc if c != "all"}) + ["all"]
    for cam in cams:
        for m in models:
            d = acc.get((cam, m))
            if not d:
                continue
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            out.append({"camera": cam, "frames": d["frames"], "model": m,
                        "exclusion_IoU": tp / max(1, tp + fp + fn), "hardware_recall": tp / max(1, tp + fn),
                        "exclusion_precision": tp / max(1, tp + fp)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the training set whose held-out frames are the test (images/, masks/)")
    ap.add_argument("--split-ckpt", required=True, help="the checkpoint (or .safetensors) whose card defines the split")
    ap.add_argument("--models", nargs="+", required=True, help="registry names or checkpoint paths to compare")
    ap.add_argument("--seen", default=None, help="an older training set (or masks folder): report frames not in it")
    ap.add_argument("--seen-split", nargs=2, default=None, metavar=("CKPT", "DATA"),
                    help="the older model's checkpoint and the set it was trained on: its held-out frames (the split "
                         "rebuilt from its card) count as unseen too - with --seen, the frames neither model trained "
                         "on (31 Oct: the v8 held-out frames are 98 %% v7 frames)")
    ap.add_argument("--also-held-out", nargs="+", default=[], metavar="CKPT",
                    help="2 Nov 2026: keep only the test frames these models (checkpoints trained on --data) held out "
                         "too - their splits rebuilt as they trained (the newest masks left out to their count)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="only the first N held-out frames (a quick look)")
    a = ap.parse_args(argv)

    import cv2
    from .make_mask_figure import family, pooled
    from mppp.mask.infer import get_model, predict_probability
    from mppp_studies.mask_training.audit import split_labels
    from mppp_studies.mask_training.train import scan_dataset

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from mppp_studies.mask_training.audit import recorded_split
    _, split_card = get_model(a.split_ckpt, a.device)
    items = scan_dataset(a.data, image_dirs=("images",), mask_dir="masks", missing="skip")

    def split_of(ck, card):
        # 0.95.4: the split training recorded (split.json, frame_audit.csv, the card's val_masks); the rebuild from
        # file times only when nothing was recorded - on 2 Nov it took 54 of 59 "held-out" frames from v4.1's training
        if recorded_split(ck, card) is not None:
            print(f"  {Path(ck).name}: the split recorded in training")
            return split_labels(items, card, ck)
        print(f"  {Path(ck).name}: no recorded split - rebuilt from the card (newest masks left out); unreliable "
              f"once the set has changed")
        return split_labels(items_as_trained(items, card), card)

    split = split_of(a.split_ckpt, split_card)
    val = [it for it in items if split.get(it.mask) == "val"]
    for ck in a.also_held_out:
        _, card_o = get_model(ck, a.device)
        sp_o = split_of(ck, card_o)
        before = len(val)
        val = [it for it in val if sp_o.get(it.mask) == "val"]
        print(f"--also-held-out {Path(ck).name}: {len(val)} of {before} test frames held out by it too")
    val = val[: a.limit]
    if not val:
        raise SystemExit("no held-out frames: is --split-ckpt a model trained on --data?")
    seen = seen_names(a.seen) if a.seen else set()
    if a.seen_split and seen:
        # the older model's own held-out frames were not in its training either (its split, rebuilt from its card
        # over the set it was trained on - images/ and images_variable/ as in training then)
        _, old_card = get_model(a.seen_split[0], a.device)
        old_items = scan_dataset(a.seen_split[1], image_dirs=("images", "images_variable"), mask_dir="masks",
                                 missing="skip")
        old_split = split_labels(old_items, old_card, a.seen_split[0])
        held = {Path(m).name for m, v in old_split.items() if v == "val"}
        print(f"--seen-split: {len(held)} masks held out by {Path(a.seen_split[0]).name}")
        seen = seen - held
    labels = [Path(m).stem if Path(m).suffix else m for m in a.models]
    print(f"{len(val)} held-out frames; {sum(Path(it.mask).name in seen for it in val)} of them in --seen")

    frame_rows = []
    for spec, lab in zip(a.models, labels):
        model, card = get_model(spec, a.device)
        thr = float(card["threshold"])
        for k, it in enumerate(val, 1):
            rgb = cv2.cvtColor(cv2.imread(it.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(it.mask, cv2.IMREAD_GRAYSCALE) > 127
            r = pooled(gt, predict_probability(model, card, rgb) > thr, rgb.max(axis=2) > 0)
            name = Path(it.image).name
            frame_rows.append({"frame": name, "camera": family(name), "model": lab,
                               "in_seen": Path(it.mask).name in seen, **r})
            if k % 50 == 0:
                print(f"  {lab}: {k}/{len(val)}", flush=True)

    with open(out / "mask_ab_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0]))
        w.writeheader()
        w.writerows(frame_rows)
    md = [f"# Mask model A/B on {len(val)} held-out frames of {Path(a.split_ckpt).name}", "",
          f"Test set: {a.data}; models: {', '.join(labels)}" + (f"; --seen {a.seen}" if a.seen else ""), ""]
    allrows = pool(frame_rows, labels)
    md += table(allrows, "All held-out frames")
    if seen:
        unseen = pool(frame_rows, labels, keep=lambda r: not r["in_seen"])
        md += table(unseen, "Held-out frames the older model did not train on (not in --seen, or in its own held-out split with --seen-split): the fair comparison")
    with open(out / "mask_ab.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subset"] + list(allrows[0]))
        w.writeheader()
        w.writerows([dict(r, subset="all") for r in allrows]
                    + ([dict(r, subset="not_seen") for r in unseen] if seen else []))
    (out / "mask_ab.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
