"""
The JOSS paper's mask figure.  Held-out frames only (the split is rebuilt
from the checkpoint's card, as mppp.mask.training.audit does), one or more per camera family, with the predicted
exclusion tinted orange and the hand label's boundary in blue; plus per-camera metrics over every held-out frame.

    python scripts\\make_mask_figure.py --ckpt D:\\code\\MPPP\\checkpoints\\convnext_tiny_s4_seg_20261021_v8.pt
                          --data D:\\masks\\masks_training_set_v8 --out paper\\mask_fig
                          [--names NLF_....png ZL0_....png ...] [--per-family 2] [--dpi 300] [--no-metrics]

Writes <out>/figure1_masks.png (+ .pdf), <out>/mask_metrics_by_camera.csv / .md and <out>/figure1_frames.txt.
Metrics ignore invalid pixels (RGB exactly 0: padding, dark columns), which every label marks "exclude" and which
would otherwise inflate the scores.  "Hardware recall" = excluded label pixels the model also excludes (the
error that matters for SfM: a missed rover pixel can tie stations on the rover).
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

FAMILIES = {"N": "Navcam", "F": "Front Hazcam", "R": "Rear Hazcam", "Z": "Mastcam-Z", "S": "SHERLOC WATSON"}
EXCL_RGB = (0xEB / 255, 0x68 / 255, 0x34 / 255)      # predicted exclusion (orange)
LABEL_HEX = "#2a78d6"                               # label boundary (blue)


def family(name: str) -> str:
    return FAMILIES.get(name[:1].upper(), name[:3])


def sol_of(name: str) -> str:
    s = name[4:8]
    return f"sol {int(s)}" if s.isdigit() else ""


def pooled(truth_incl, pred_incl, valid):
    import numpy as np
    t, p = ~truth_incl & valid, ~pred_incl & valid                 # exclusion class, valid pixels only
    return {"tp": int((t & p).sum()), "fp": int((~t & p & valid).sum()), "fn": int((t & ~p).sum()),
            "n_valid": int(valid.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="*", default=None, help="images/ file names to show (default: auto-pick)")
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-metrics", action="store_true")
    a = ap.parse_args(argv)

    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mppp.mask.infer import get_model, predict_probability
    from mppp.mask.training.train import scan_dataset
    from mppp.mask.training.audit import split_labels

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, card = get_model(a.ckpt, a.device)
    thr = float(card["threshold"])
    items = scan_dataset(a.data, image_dirs=("images",), mask_dir="masks", missing="skip")
    split = split_labels(scan_dataset(a.data, image_dirs=("images", "images_variable"), mask_dir="masks",
                                      missing="skip"), card)
    val = [it for it in items if split.get(it.mask) == "val"]
    if not val:
        raise SystemExit("no held-out frames: is this the card of a model trained on this set?")

    cache = {}

    def run(it):
        if it.image not in cache:
            bgr = cv2.imread(it.image, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prob = predict_probability(model, card, rgb)
            gt = cv2.imread(it.mask, cv2.IMREAD_GRAYSCALE) > 127
            cache[it.image] = (rgb, gt, prob, rgb.max(axis=2) > 0)
        return cache[it.image]

    # ---- metrics over every held-out frame, pooled per camera family
    if not a.no_metrics:
        acc = {}
        for k, it in enumerate(val, 1):
            rgb, gt, prob, valid = run(it)
            r = pooled(gt, prob > thr, valid)
            d = acc.setdefault(family(Path(it.image).name), {"frames": 0, "tp": 0, "fp": 0, "fn": 0})
            d["frames"] += 1
            for key in ("tp", "fp", "fn"):
                d[key] += r[key]
            if len(cache) > 8 and (a.names is None or Path(it.image).name not in a.names):
                cache.pop(it.image, None)                            # keep memory bounded
        rows = []
        for fam, d in sorted(acc.items()):
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            rows.append({"camera": fam, "frames": d["frames"],
                         "exclusion_IoU": round(tp / max(1, tp + fp + fn), 4),
                         "hardware_recall": round(tp / max(1, tp + fn), 4),
                         "exclusion_precision": round(tp / max(1, tp + fp), 4)})
        with open(out / "mask_metrics_by_camera.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        md = ["| camera | held-out frames | IoU (exclude) | recall (exclude) | precision (exclude) |",
              "|---|---:|---:|---:|---:|"]
        md += [f"| {r['camera']} | {r['frames']} | {r['exclusion_IoU']:.3f} | {r['hardware_recall']:.3f} | "
               f"{r['exclusion_precision']:.3f} |" for r in rows]
        (out / "mask_metrics_by_camera.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))

    # ---- pick the frames to show
    by_name = {Path(it.image).name: it for it in val}
    if a.names:
        missing = [n for n in a.names if n not in by_name]
        if missing:
            raise SystemExit(f"not held-out (or not in images/): {missing}")
        picks = [by_name[n] for n in a.names]
    else:                                # per family: frames whose label excludes 10-60 % of the valid pixels
        picks = []
        for fam in FAMILIES.values():
            cands = [it for it in val if family(Path(it.image).name) == fam]
            scored = []
            for it in cands[:200]:
                gt = cv2.imread(it.mask, cv2.IMREAD_GRAYSCALE) > 127
                im = cv2.imread(it.image, cv2.IMREAD_COLOR)
                valid = im.max(axis=2) > 0
                f = float((~gt & valid).sum() / max(1, valid.sum()))
                if 0.10 <= f <= 0.60:
                    scored.append((abs(f - 0.3), Path(it.image).name, it))
            picks += [s[2] for s in sorted(scored)[:a.per_family]]
    (out / "figure1_frames.txt").write_text("".join(Path(p.image).name + "\n" for p in picks), encoding="utf-8")

    # ---- the figure
    n = len(picks)
    cols = min(a.cols, n)
    rws = (n + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(3.4 * cols, 2.75 * rws), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, it in zip(axes.ravel(), picks):
        rgb, gt, prob, valid = run(it)
        excl = (prob <= thr) & valid
        im = rgb.astype(np.float32) / 255.0
        tint = np.array(EXCL_RGB, np.float32)
        im[excl] = 0.55 * im[excl] + 0.45 * tint
        ax.imshow(np.clip(im, 0, 1), interpolation="lanczos")
        ax.contour(gt.astype(float), levels=[0.5], colors=[LABEL_HEX], linewidths=0.8)
        name = Path(it.image).name
        ax.set_title(f"{family(name)}  {sol_of(name)}", fontsize=8, color="#0b0b0b", loc="left")
    fig.text(0.01, 0.005, "orange: excluded by the model;  blue line: hand-label boundary;  held-out frames",
             fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure1_masks.{ext}", dpi=a.dpi)
    print("->", out / "figure1_masks.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
