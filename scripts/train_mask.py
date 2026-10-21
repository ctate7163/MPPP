"""
Headless mask retraining (a GPU machine), equivalent to notebooks/training/train_mask.ipynb steps 1-3.  Run it from
a Python that has mppp and a CUDA torch (on Windows: the notebooks' kernel; the pip torch of envs/mppp.yml is CPU only):

    set MPPP_MASK_DATA=D:\\masks\\masks_training_set_v8
    set MPPP_PDS=D:\\data\\m2020
    python scripts\\train_mask.py --regen --reuse-variable D:\\masks\\masks_training_set_v7\\images_variable
    python scripts\\train_mask.py --epochs 10            (images/ already regenerated)

Writes checkpoints\\convnext_tiny_s4_seg_<yyyymmdd>[_tag].pt + .json card, <stem>_debug\\ (curves.png, log.csv,
val\\, batch\\, frame_audit.csv, worst_frames_NN.png).  Never promotes or releases: do that after looking.
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
    ap.add_argument("--reuse-variable", nargs="*", default=[], help="older images_variable/ folders to reuse")
    ap.add_argument("--workers", type=int, default=4, help="--regen processes (1 if spawn gives trouble)")
    ap.add_argument("--epochs", type=int, default=10, help="10 = the mppp_mask_v3 recipe (TrainConfig default 6)")
    ap.add_argument("--quad-split-above", type=int, default=None, help="e.g. 4000: 5120 px frames as quadrants")
    ap.add_argument("--init-from", default=None, help="start from a ConvNeXtSeg checkpoint (fine-tune)")
    ap.add_argument("--tag", default="", help="checkpoint name suffix, e.g. v8")
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
            print(build_variable_images(data, reuse_from=a.reuse_variable))
        cfg = TrainConfig(epochs=a.epochs, quad_split_above=a.quad_split_above, init_from=a.init_from)
        stem = f"{cfg.backbone}{'_s4' if cfg.stride4 else ''}_seg_{date.today():%Y%m%d}{('_' + a.tag) if a.tag else ''}"
        out = checkpoints_dir(create=True) / f"{stem}.pt"

    items = scan_dataset(data, image_dirs=("images", "images_variable"), mask_dir="masks", missing="skip")
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
    return 0


if __name__ == "__main__":                 # required on Windows: DataLoader workers start by spawn
    sys.exit(main())
