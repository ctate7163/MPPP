"""
Headless mask retraining (a GPU machine), equivalent to notebooks/training/train_mask.ipynb steps 1-3.  Run it from
a Python that has mppp and a CUDA torch (on Windows: the notebooks' kernel; the pip torch of envs/mppp.yml is CPU only):

    set MPPP_MASK_DATA=D:\\masks\\masks_training_set_v8
    set MPPP_PDS=D:\\data\\m2020
    python scripts\\train_mask.py --regen                (images/ from the PDS; --variable also images_variable/)
    python scripts\\train_mask.py --epochs 10            (images/ already regenerated)

Writes checkpoints\\convnext_tiny_s4_seg_<yyyymmdd>[_tag].pt + .json card, <stem>_debug\\ (curves.png, log.csv,
val\\, batch\\, frame_audit.csv, worst_frames_NN.png); --paper also the JOSS outputs (scripts\\windows\\train_masks.bat
does it all).  Augmentation: horizontal flips, and large Navcam/Hazcam frames zoomed in up to 2x (--zoom-max).
Never promotes or releases: do that after looking.
--smoke runs a tiny CPU training on synthetic data (checks the script, nothing else).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # duplicate Intel OpenMP on Windows/conda; before torch


def _smoke_data(root: Path) -> None:
    import cv2
    import numpy as np
    rng = np.random.default_rng(0)
    for d in ("images", "masks"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for k in range(12):
        im = rng.integers(20, 230, (96, 128, 3), dtype=np.uint8)
        m = np.full((96, 128), 255, np.uint8)
        m[60:, 30 + k:90] = 0                                   # a "rover" block
        im[m == 0] = (90, 90, 90)
        name = f"NLF_{k:04d}_0000000000_000FDR_N0000000NCAM00000_00_095J01.png"
        cv2.imwrite(str(root / "images" / name), im)
        cv2.imwrite(str(root / "masks" / name), m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("MPPP_MASK_DATA"), help="training set root (masks/, images/)")
    ap.add_argument("--pds", default=os.environ.get("MPPP_PDS"), help="PDS archive (for --regen)")
    ap.add_argument("--regen", action="store_true", help="regenerate images/ from the PDS, then images_variable/")
    ap.add_argument("--reuse-variable", nargs="*", default=[], help="older images_variable/ folders to reuse "
                    "(with --variable)")
    ap.add_argument("--variable", action="store_true", help="also train on images_variable/ (brightness-varied copies, "
                    "built by --regen); 0.95.2 (Christian, 30 Oct): off - only the images of the standard pipeline")
    ap.add_argument("--workers", type=int, default=8, help="--regen processes (default 8; 1 if spawn gives trouble)")
    ap.add_argument("--epochs", type=int, default=10, help="10 = the mppp_mask_v3 recipe (TrainConfig default 6)")
    ap.add_argument("--quad-split-above", type=int, default=None, help="e.g. 4000: 5120 px frames as quadrants")
    ap.add_argument("--init-from", default=None, help="start from a ConvNeXtSeg checkpoint (fine-tune)")
    ap.add_argument("--tag", default="", help="checkpoint name suffix, e.g. v8")
    ap.add_argument("--zoom-max", type=float, default=2.0, help="0.95.2: large Navcam/Hazcam frames zoomed in by up "
                    "to this factor (a random crop, then downsampled; default 2; 1 = off)")
    ap.add_argument("--zoom-prob", type=float, default=0.5, help="share of those frames zoomed (default 0.5)")
    ap.add_argument("--no-hflip", action="store_true", help="no random left-right flips (on by default)")
    ap.add_argument("--paper", nargs="?", const="auto", default=None, metavar="DIR",
                    help="0.95.2: then the JOSS outputs into DIR (default <repo>/Claude outputs/mask_<checkpoint>): "
                         "figure1_masks.png/.pdf, per-camera metrics, training curves, the card, the training-set "
                         "summary and the release candidate .safetensors")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic CPU run (script check)")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")   # mppp logs to "mppp.*"; notebooks show nothing
    import torch
    from mppp.paths import checkpoints_dir
    from mppp.mask.training.train import TrainConfig, scan_dataset, train
    from mppp.mask.training.audit import audit_frames, summarize, worst_table, rank_rows, render_worst

    if a.smoke:
        import tempfile
        root = Path(tempfile.mkdtemp(prefix="mppp_mask_smoke_"))
        _smoke_data(root / "set")
        data = root / "set"
        cfg = TrainConfig(epochs=1, batch_size=2, input_size=128, canvas=(128, 96), pretrained=False,
                          num_workers=0, val_preview_every=0, print_every=2, val_ratio=0.25)
        out = root / "ckpt" / "smoke.pt"
    else:
        if not a.data:
            ap.error("--data or MPPP_MASK_DATA is required")
        data = Path(a.data)
        if not torch.cuda.is_available():
            print("WARNING: no CUDA in this Python's torch - training on the CPU would take days. "
                  "Run from the notebooks' kernel (python -c \"import torch; print(torch.cuda.is_available())\").")
            return 2
        if a.regen:
            from mppp.mask.training.dataset import regenerate_images_from_pds, build_variable_images
            if not a.pds:
                ap.error("--regen needs --pds or MPPP_PDS")
            print(regenerate_images_from_pds(data, a.pds, workers=a.workers))
            if a.variable:
                n_img = len([f for f in (data / "images").glob("*.png")])
                vdir = data / "images_variable"
                n_var = len([f for f in vdir.glob("*.png")]) if vdir.is_dir() else 0
                if n_var >= n_img > 0:                  # 0.95.2: a rerun keeps the complete images_variable/
                    print(f"images_variable/ complete ({n_var} of {n_img}): kept")
                else:
                    print(build_variable_images(data, reuse_from=a.reuse_variable))
        cfg = TrainConfig(epochs=a.epochs, quad_split_above=a.quad_split_above, init_from=a.init_from,
                          hflip=not a.no_hflip, zoom_max=a.zoom_max, zoom_prob=a.zoom_prob)
        stem = f"{cfg.backbone}{'_s4' if cfg.stride4 else ''}_seg_{date.today():%Y%m%d}{('_' + a.tag) if a.tag else ''}"
        out = checkpoints_dir(create=True) / f"{stem}.pt"
        k = 2
        while out.exists():                             # 0.95.2: a second run the same day gets _2, _3 ...
            out = out.with_name(f"{stem}_{k}.pt")
            k += 1

    image_dirs = ("images", "images_variable") if (a.variable or a.smoke) else ("images",)
    items = scan_dataset(data, image_dirs=image_dirs, mask_dir="masks", missing="skip")
    print(f"{len(items)} images, {len({i.mask for i in items})} masks -> {out}")
    ckpt = train(items, out, cfg, live=False)
    print("saved", ckpt)
    if not a.no_audit:
        rows = audit_frames(items, ckpt, image_dirs=("images",), which="all")
        print(summarize(rows))
        view = rank_rows(rows, "confident_error_pct", split="train")
        print(worst_table(view, n=30))
        if not a.smoke:
            render_worst(view, ckpt, n=24, sort_by="confident_error_pct")
    if a.paper:
        pdir = (Path(__file__).resolve().parent.parent / "Claude outputs" / f"mask_{Path(ckpt).stem}") if a.paper == "auto" else Path(a.paper)
        print(paper_outputs(Path(ckpt), data, pdir, items, smoke=a.smoke, image_dirs=image_dirs))
    return 0


def training_set_summary(items, card) -> dict:
    """The training set as the paper describes it: frames per camera, sols, the split (from the card)."""
    import re
    fam = {"N": "Navcam", "F": "Front Hazcam", "R": "Rear Hazcam", "Z": "Mastcam-Z", "S": "SHERLOC WATSON"}
    masks = sorted({i.mask for i in items})
    by: dict = {}
    sols = []
    for m in masks:
        n = Path(m).name
        f = fam.get(n[:1].upper(), n[:3])
        by[f] = by.get(f, 0) + 1
        s = re.match(r"^\w{3}_(\d{4})_", n)
        if s:
            sols.append(int(s.group(1)))
    tr = card.get("training") or {}
    return {"masks": len(masks), "samples": len(items), "by_camera": dict(sorted(by.items())),
            "sols": [min(sols), max(sols)] if sols else None, "n_train": tr.get("n_train"), "n_val": tr.get("n_val"),
            "n_train_frames": tr.get("n_train_frames"), "n_val_frames": tr.get("n_val_frames"),
            "val_iou": card.get("val_iou"), "epochs": tr.get("epochs"), "hflip": tr.get("hflip"),
            "zoom_max": tr.get("zoom_max"), "zoom_prob": tr.get("zoom_prob"), "backbone": card.get("backbone")}


def paper_outputs(ckpt: Path, data: Path, out: Path, items, smoke: bool = False,
                  image_dirs=("images",)) -> dict:
    """0.95.2 (Christian, 30 Oct: "generates the required figures and outputs needed for the joss paper"): the mask
    figure and per-camera metrics on the held-out frames (scripts/make_mask_figure.py), the training curves and
    log, the model card, the training-set summary (summary.md / .json) and the release candidate .safetensors
    (not registered: promote and release after looking)."""
    import json
    import shutil
    out.mkdir(parents=True, exist_ok=True)
    card = json.loads(ckpt.with_suffix(".json").read_text(encoding="utf-8"))
    res: dict = {"folder": str(out)}
    import make_mask_figure
    make_mask_figure.main(["--ckpt", str(ckpt), "--data", str(data), "--out", str(out)]
                          + (["--dpi", "60", "--per-family", "1"] if smoke else []))
    res["figure"] = str(out / "figure1_masks.png")
    dbg = ckpt.parent / f"{ckpt.stem}_debug"
    for f in ("curves.png", "log.csv"):
        if (dbg / f).is_file():
            shutil.copyfile(dbg / f, out / f"training_{f}")
    shutil.copyfile(ckpt.with_suffix(".json"), out / f"{ckpt.stem}.card.json")
    summ = training_set_summary(items, card)
    summ["image_dirs"] = list(image_dirs)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    lines = [f"# Mask model {ckpt.stem}", "",
             f"- training set: {summ['masks']} image-mask pairs ({summ['samples']} samples from "
             f"{' + '.join(d + '/' for d in summ['image_dirs'])}), "
             f"sols {summ['sols'][0]}-{summ['sols'][1]}" if summ["sols"] else f"- training set: {summ['masks']} pairs",
             "- by camera: " + ", ".join(f"{k} {v}" for k, v in summ["by_camera"].items()),
             f"- split (grouped by mask): {summ['n_train_frames']} train / {summ['n_val_frames']} held-out masks "
             f"({summ['n_train']} / {summ['n_val']} samples)",
             f"- held-out IoU (include class, threshold {card.get('threshold')}): {summ['val_iou']}",
             f"- {summ['backbone']}, {summ['epochs']} epochs; augmentation: horizontal flip {summ['hflip']}, zoom of "
             f"large Navcam/Hazcam frames up to {summ['zoom_max']} x (share {summ['zoom_prob']})", ""]
    if (out / "mask_metrics_by_camera.md").is_file():
        lines += ["## Held-out frames by camera (valid pixels only)", "",
                  (out / "mask_metrics_by_camera.md").read_text(encoding="utf-8")]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        from mppp.mask.hub import export_safetensors
        r = export_safetensors(ckpt, out=out / f"{ckpt.stem}.safetensors")
        res["safetensors"] = r.get("path")
        res["sha256"] = r.get("sha256")
    except Exception as e:                                   # noqa: BLE001
        res["safetensors_error"] = f"{type(e).__name__}: {e}"
    res["summary"] = str(out / "summary.md")
    return res


if __name__ == "__main__":                 # required on Windows: DataLoader workers start by spawn
    sys.exit(main())
