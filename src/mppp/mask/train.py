"""
Mask-model training (ConvNeXt tiny/base + FPN + lite ASPP), from the
``Tate_training_convnext_20251002_edited`` notebook, with the defects that
produced NaN losses in the September 2026 run removed.

Why that run failed (diagnosis in CHANGELOG v0p2)
--------------------------------------------------
1. **fp16 overflow in the decoder, not the backbone.**  ``fpn.smooth ->
   aspp.b* -> aspp.proj`` has no normalisation until ``aspp.bn``; the loss is
   invariant to the scale of those weights, so their norms grow during
   training (x1.3-1.7 after one epoch) and the pre-BN activations grow with the
   product.  On an ordinary Mastcam-Z frame the epoch-1 checkpoint already
   reaches 1.2e4 and the 2025 checkpoint 2.3e4; fp16 overflows at 6.55e4.  The
   brightest/most textured training images crossed first (intermittent
   non-finite losses from epoch 2, iteration ~570), then all of them.
2. **Skipping non-finite batches is an absorbing state.**  A skipped batch
   produces no update, so once every batch overflows the weights can never
   change again (epochs 3-4: 1735/1748 and 1741/1748 skipped).  The forward
   pass had already written inf/NaN into the BatchNorm running statistics, so
   evaluation returned NaN on every validation batch from the end of epoch 2.
3. Gradients were clipped *before* ``GradScaler.unscale_``, i.e. clipped to
   norm 1 in scaled units (~1/65536 in real units), making the clip threshold
   meaningless.
4. The random train/val split put ``images/X`` and ``images_variable/X``
   (same scene, same mask) on both sides, so validation IoU is optimistic.

The backbone size is not the cause: the overflow is in the decoder, which is
the same for convnext_base.

Fixes here: precision ``auto`` = bf16 when the GPU supports it, otherwise the
backbone in fp16 and the decoder + loss in fp32; unscale before clip; any
non-finite loss stops training with the offending files named (no silent
skipping); split grouped by mask; the pre-BN ASPP peak is logged so drift is
visible; checkpoints are never overwritten and always get a model card.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .model import DEFAULT_CARD, ConvNeXtSeg, canvas_of, fit_to_canvas, needs_quad_split, quad_boxes, write_card

PathLike = Union[str, Path]
FP16_MAX = 65504.0


class NonFiniteLoss(RuntimeError):
    pass


# ------------------------------------------------------------------ data
@dataclass(frozen=True)
class Item:
    image: str
    mask: str
    tile: int = -1                      # -1 whole frame; 0-3 quadrant (TL, TR, BL, BR), see model.quad_boxes

    @property
    def label(self) -> str:
        return Path(self.image).name + ("" if self.tile < 0 else f"#q{self.tile}")


def scan_dataset(data_dir: PathLike, image_dirs: Sequence[str] = ("images", "images_variable"),
                 mask_dir: str = "masks", missing: str = "raise") -> List[Item]:
    """
    ``data_dir/<image_dir>/<name>`` paired with ``data_dir/<mask_dir>/<name>``.
    ``missing``: what to do with images that have no mask — "raise" (default;
    the notebook would have failed later, inside a DataLoader worker) or
    "skip" (e.g. frames removed from the Metashape project; they are counted
    in a warning).
    """
    if missing not in ("raise", "skip"):
        raise ValueError("missing must be 'raise' or 'skip'")
    root = Path(data_dir)
    items, missing_items = [], []
    for d in image_dirs:
        folder = root / d
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff") or p.name.endswith(".tmp.png"):
                continue
            m = root / mask_dir / p.name
            (items if m.is_file() else missing_items).append(Item(str(p), str(m)))
    if missing_items and missing == "raise":
        raise FileNotFoundError(f"{len(missing_items)} images have no mask, e.g. {missing_items[0].image} "
                                f"(scan_dataset(..., missing='skip') ignores them)")
    if missing_items:
        import warnings
        warnings.warn(f"scan_dataset: skipped {len(missing_items)} images without a mask, "
                      f"e.g. {Path(missing_items[0].image).name}")
    if not items:
        raise FileNotFoundError(f"No images found under {root} / {list(image_dirs)}")
    return items


def grouped_split(items: Sequence[Item], val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Item], List[Item]]:
    """Split by mask file, so every variant of a scene lands on the same side."""
    groups: Dict[str, List[Item]] = {}
    for it in items:
        groups.setdefault(it.mask, []).append(it)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(round(val_ratio * len(keys))))
    val_keys = set(keys[:n_val])
    train = [it for k in keys if k not in val_keys for it in groups[k]]
    val = [it for k in keys if k in val_keys for it in groups[k]]
    return train, val


def _image_size(path: PathLike) -> Tuple[int, int]:
    """(h, w) from the PNG header without decoding; other formats are decoded."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(head[20:24], "big"), int.from_bytes(head[16:20], "big")
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"cannot read {path}")
    return im.shape[:2]


def expand_quads(items: Sequence[Item], quad_split_above: Optional[int], overlap: int = 64,
                 skip_empty: bool = True) -> Tuple[List[Item], Dict[str, int]]:
    """
    Replace every frame whose long side exceeds ``quad_split_above`` by its four
    overlapping quadrants (``Item.tile`` 0-3), as inference will see it.
    ``skip_empty`` drops quadrants whose core is pure padding (image and mask
    all zero, e.g. the unused half of a sub-frame padded to the detector
    frame), the rule inference uses; the image is only read when the mask
    core is empty.
    Returns (items, counts).
    """
    out: List[Item] = []
    n = {"frames": 0, "split": 0, "tiles": 0, "empty_skipped": 0}
    mask_cache: Dict[str, np.ndarray] = {}
    for it in items:
        n["frames"] += 1
        h, w = _image_size(it.mask)
        if not needs_quad_split(h, w, quad_split_above):
            out.append(it)
            continue
        n["split"] += 1
        boxes = quad_boxes(h, w, overlap)
        ms = None
        if skip_empty:
            ms = mask_cache.get(it.mask)
            if ms is None:
                ms = cv2.imread(it.mask, cv2.IMREAD_GRAYSCALE)
                mask_cache = {it.mask: ms}            # items of one mask are adjacent after sorting
        im = None
        for q, (_, (cy0, cy1, cx0, cx1)) in enumerate(boxes):
            # same rule as inference: a quadrant whose core is pure padding is never run
            if skip_empty and not np.any(ms[cy0:cy1, cx0:cx1]):
                if im is None:
                    im = cv2.imread(it.image, cv2.IMREAD_COLOR)
                if im is not None and not np.any(im[cy0:cy1, cx0:cx1]):
                    n["empty_skipped"] += 1
                    continue
            out.append(Item(it.image, it.mask, q))
            n["tiles"] += 1
    return out, n


def letterbox(im: np.ndarray, ms: np.ndarray, size: int,
              canvas: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize long side to ``size`` (shrinking further if needed to fit ``canvas``
    = (W, H)), zero-pad bottom/right — identical to inference.  ``canvas=None``
    is the square ``size`` x ``size`` of pre-v0p5 checkpoints.
    """
    cw, ch = canvas or (size, size)
    nh, nw = fit_to_canvas(im.shape[0], im.shape[1], size, (cw, ch))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ms = cv2.resize(ms, (nw, nh), interpolation=cv2.INTER_NEAREST)
    im = np.pad(im, ((0, ch - nh), (0, cw - nw), (0, 0)))
    ms = np.pad(ms, ((0, ch - nh), (0, cw - nw)))
    return im, (ms > 127).astype(np.uint8)


class MaskDataset(Dataset):
    def __init__(self, items: Sequence[Item], size: int = 1648, hflip: bool = False,
                 canvas: Optional[Tuple[int, int]] = None, quad_overlap: int = 64):
        self.items, self.size, self.hflip, self.canvas = list(items), int(size), hflip, canvas
        self.quad_overlap = int(quad_overlap)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        im = cv2.imread(it.image, cv2.IMREAD_COLOR)
        ms = cv2.imread(it.mask, cv2.IMREAD_GRAYSCALE)
        if im is None or ms is None:
            raise RuntimeError(f"Failed to read {it.image if im is None else it.mask}")
        if im.shape[:2] != ms.shape[:2]:
            raise RuntimeError(f"{it.image} is {im.shape[1]}x{im.shape[0]} but its mask is {ms.shape[1]}x{ms.shape[0]}")
        if it.tile >= 0:
            (y0, y1, x0, x1), _ = quad_boxes(ms.shape[0], ms.shape[1], self.quad_overlap)[it.tile]
            im, ms = im[y0:y1, x0:x1], ms[y0:y1, x0:x1]
        im, ms = letterbox(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), ms, self.size, self.canvas)
        if self.hflip and random.random() < 0.5:
            im, ms = im[:, ::-1].copy(), ms[:, ::-1].copy()
        return im, ms, i


class Collate:
    """
    Batch -> normalised NCHW tensor.  A module-level class, not a closure, so it
    pickles: with ``num_workers > 0`` on Windows the DataLoader starts worker
    processes by *spawn* and must pickle it (v0p6 failed there with
    "Can't get local object 'make_collate.<locals>.collate'").
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(list(mean), dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(list(std), dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, batch):
        x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        y = torch.from_numpy(np.stack([b[1] for b in batch])).float()
        return x, y, [b[2] for b in batch]


def make_collate(card: Dict) -> Collate:
    return Collate(card["normalisation"]["mean"], card["normalisation"]["std"])


# ---------------------------------------------------------------- losses
def bce_tversky_loss(logits: torch.Tensor, gt: torch.Tensor, alpha: float = 0.3, beta: float = 0.7,
                     eps: float = 1e-6):
    """Unchanged from the notebook, but always evaluated in fp32."""
    logits = logits.float().squeeze(1)
    B = logits.shape[0]
    pw = (((1 - gt).flatten(1).sum(1) + eps) / (gt.flatten(1).sum(1) + eps)).clamp(1.0, 10.0).view(B, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pw)
    p = torch.sigmoid(logits)
    tp = (p * gt).flatten(1).sum(1)
    fp = (p * (1 - gt)).flatten(1).sum(1)
    fn = ((1 - p) * gt).flatten(1).sum(1)
    tv = 1 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()
    return bce + tv, float(bce.detach()), float(tv.detach())


@torch.no_grad()
def batch_iou(logits: torch.Tensor, gt: torch.Tensor, threshold: float) -> float:
    pb = (torch.sigmoid(logits.float().squeeze(1)) > threshold).float()
    inter = (pb * gt).flatten(1).sum(1)
    union = (pb + gt - pb * gt).flatten(1).sum(1).clamp_min(1)
    return float((inter / union).mean())


# ------------------------------------------------------------- precision
def resolve_precision(precision: str, device: str) -> str:
    """'auto' -> 'bf16' on GPUs that support it, else 'fp16-backbone'; CPU -> 'fp32'."""
    if precision not in ("auto", "bf16", "fp16-backbone", "fp32"):
        raise ValueError(f"precision must be auto | bf16 | fp16-backbone | fp32, got {precision!r}")
    if not device.startswith("cuda"):
        return "fp32" if precision == "auto" else precision
    if precision == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16-backbone"
    return precision


def forward(model: ConvNeXtSeg, x: torch.Tensor, precision: str, device_type: str) -> torch.Tensor:
    if precision == "fp32":
        return model(x)
    if precision == "bf16":
        with torch.autocast(device_type, dtype=torch.bfloat16):
            return model(x)
    with torch.autocast(device_type, dtype=torch.float16):          # fp16-backbone
        feats = model.features(x)
    return model.decode([f.float() for f in feats], x.shape[-2:])


# ------------------------------------------------------------------ train
@dataclass
class TrainConfig:
    backbone: str = "convnext_tiny"       # v0p6 default (base showed no overfitting and gains little)
    input_size: int = 1648                # long side: Mastcam-Z at native resolution
    # (W, H), multiples of 32 (ConvNeXt stride): 4:3 frames fit with 16/48 px of padding
    # instead of the 448 px of a 1648 square (-24 % pixels).  None = square input_size.
    canvas: Optional[Tuple[int, int]] = (1664, 1248)
    fpn_width: int = 256
    stride4: bool = True                  # v0p6: stride-4 skip in the decoder (sharper boundaries)
    # v0p6 option, OFF by default: frames with a long side > quad_split_above px are trained -
    # and inferred - as four quadrants overlapping by quad_overlap px.  4000 would split only the
    # 5120 px Navcam/Hazcam frames (whole: 0.32x on the canvas; quadrants: 0.63x, the scale 2560 px
    # frames get whole); 2500 also the 2560 px frames (quadrants enlarged 1.22x, ~2x samples per
    # epoch).  Quadrants whose core is pure padding are skipped (skip_empty_quads).
    quad_split_above: Optional[int] = None
    quad_overlap: int = 64
    skip_empty_quads: bool = True
    # v0p10 defaults (requested 24 Sep 2026 after the base/tiny comparison): 6 epochs, lr 5e-5
    # (1e-4 was hot for batch 4: val loss only fell as the cosine schedule annealed), weight
    # decay 5e-3 (AdamW decays by lr*wd per step, so 5e-4 was effectively off), hflip on.
    epochs: int = 6
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 5e-3
    warmup_fraction: float = 0.05
    grad_clip: float = 1.0
    precision: str = "auto"
    threshold: float = 0.5                # inference threshold, also used for IoU
    dilate_kernel: int = 3
    val_ratio: float = 0.1
    seed: int = 42
    hflip: bool = True                    # v0p10: random left-right flip of training frames
    num_workers: int = 2                  # v0p10: background image loading (0 = in this process)
    print_every: int = 50
    val_preview_every: int = 200          # iterations; 0 disables the held-out preview
    n_val_preview: int = 10               # fixed held-out frames shown every time
    pretrained: bool = True               # ImageNet-22k backbone (pretrained_file, hub, or local HF cache)
    # local timm model.safetensors for offline machines.  "auto" (v0p10) = checkpoints/<backbone>.fb_in22k.safetensors
    # if that file exists (e.g. checkpoints/convnext_tiny.fb_in22k.safetensors), else the hub / HF cache.
    pretrained_file: Optional[str] = "auto"
    # Start from an existing ConvNeXtSeg checkpoint instead: every tensor whose name and
    # shape match is loaded (the backbone of any same-backbone checkpoint; the decoder
    # too if fpn_width matches).  Overrides `pretrained`.
    init_from: Optional[str] = None


def resolve_pretrained_file(pretrained_file: Optional[PathLike], backbone: str) -> Optional[str]:
    """``"auto"`` -> ``checkpoints/<backbone>.fb_in22k.safetensors`` if it exists, else None
    (hub / local HF cache); any other value is returned as a string (None stays None)."""
    if pretrained_file is None:
        return None
    if str(pretrained_file) != "auto":
        return str(pretrained_file)
    try:
        from ..paths import checkpoints_dir
        f = checkpoints_dir() / f"{backbone}.fb_in22k.safetensors"
    except FileNotFoundError:
        return None
    return str(f) if f.is_file() else None


def _same_bytes(a: Path, b: Path, chunk: int = 1 << 22) -> bool:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                return False
            if not x:
                return True


def best_checkpoint_name(card: Dict) -> str:
    """Default-model file name for a card: ``<backbone>[_s4]_seg_best.pt``."""
    return f"{card['backbone']}{'_s4' if card.get('stride4') else ''}_seg_best.pt"


def promote_checkpoint(checkpoint: PathLike, best_name: Optional[str] = None, force: bool = False,
                       archive_dir: Optional[PathLike] = None) -> Dict[str, object]:
    """
    Copy a trained checkpoint (+ model card and ``.check.png``) to the default
    name that ``mppp.default_config()`` uses — ``convnext_tiny_s4_seg_best.pt``
    for a tiny stride-4 model — in the same folder.

    An existing best is moved to ``archive/`` (``<stem>_<UTC time>.pt``), never
    deleted.  If it was trained on the same dataset (same fingerprint) with a
    higher validation IoU, nothing happens unless ``force=True``; val IoUs of
    different datasets are not comparable, so then the new one wins.
    Returns a report dict (``promoted``: bool, ``reason``, paths).
    """
    import shutil
    from .model import card_path, read_card
    src = Path(checkpoint)
    if not src.is_file():
        raise FileNotFoundError(f"checkpoint not found: {src}")
    card = read_card(src)
    dst = src.parent / (best_name or best_checkpoint_name(card))
    if dst.resolve() == src.resolve():
        return {"promoted": False, "reason": "already the default", "best": str(dst)}
    rep: Dict[str, object] = {"source": str(src), "best": str(dst), "val_iou": card.get("val_iou")}
    if dst.is_file():
        old = read_card(dst)
        if old.get("promoted_from") == src.name and dst.stat().st_size == src.stat().st_size \
                and _same_bytes(src, dst):
            rep.update(promoted=False, reason=f"{src.name} is already the default")
            return rep
        same_data = (old.get("training", {}) or {}).get("dataset_fingerprint") == \
                    (card.get("training", {}) or {}).get("dataset_fingerprint") and old.get("training")
        if same_data and not force and float(old.get("val_iou") or -1) > float(card.get("val_iou") or -1):
            rep.update(promoted=False, reason=f"{dst.name} ({old.get('name')}) has a higher val IoU "
                       f"{old.get('val_iou'):.4f} on the same dataset (force=True to replace it)")
            return rep
        adir = Path(archive_dir) if archive_dir else src.parent / "archive"
        adir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        moved = []
        for f in (dst, card_path(dst), dst.with_suffix(".check.png")):
            if f.is_file():
                target = adir / f"{f.name.split('.')[0]}_{stamp}{''.join(Path(f.name).suffixes)}"
                k = 1
                while target.exists():                     # never overwrite an archived file
                    target = adir / f"{f.name.split('.')[0]}_{stamp}_{k}{''.join(Path(f.name).suffixes)}"
                    k += 1
                shutil.move(str(f), str(target))
                moved.append(str(target))
        rep["archived"] = moved
    shutil.copy2(src, dst)
    if card_path(src).is_file():
        c = json.loads(card_path(src).read_text(encoding="utf-8"))
        c["promoted_from"] = src.name
        card_path(dst).write_text(json.dumps(c, indent=2), encoding="utf-8")
    if src.with_suffix(".check.png").is_file():
        shutil.copy2(src.with_suffix(".check.png"), dst.with_suffix(".check.png"))
    rep.update(promoted=True, reason="promoted")
    return rep


def init_from_checkpoint(model: ConvNeXtSeg, checkpoint: PathLike) -> Dict[str, object]:
    """
    Copy every tensor whose name and shape match from a ConvNeXtSeg checkpoint.
    E.g. the Oct-2025 ``convnext_base_seg_best.pt`` (fpn_width 192) supplies the
    whole backbone but no decoder tensors for fpn_width 256.  Refuses if no
    backbone tensor matches (wrong backbone).

    An ImageNet weights file (timm/Hugging Face ``model.safetensors`` or a FAIR
    ``.pth``, i.e. no ``bb.`` keys) is accepted too and loaded into the backbone
    exactly as ``pretrained_file=`` would (v0p6; before, a .safetensors here
    failed with "UnpicklingError: invalid load key").
    """
    from .model import load_backbone_weights, read_state_dict
    src = read_state_dict(checkpoint)
    if not any(k.startswith("bb.") for k in src):
        rep = load_backbone_weights(model.bb, model.backbone_name, checkpoint)
        n_dec = sum(not k.startswith("bb.") for k in model.state_dict())
        return {"checkpoint": str(checkpoint), "backbone_tensors": f"{rep['tensors']}/{rep['tensors']}",
                "decoder_tensors": f"0/{n_dec}", "summary": rep["summary"] + " (decoder random)"}
    own = model.state_dict()
    take = {k: v for k, v in src.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    # decoder groups load all-or-nothing: the pre-v0p6 decoder (fpn, aspp, head) and the
    # v0p6 stride-4 skip (s4, fuse), so a pre-v0p6 checkpoint still supplies fpn/aspp/head.
    for group in (("fpn.", "aspp.", "head."), ("s4.", "fuse.")):
        keys = [k for k in own if k.startswith(group)]
        if keys and not all(k in take for k in keys):
            take = {k: v for k, v in take.items() if not k.startswith(group)}
    n_bb = sum(k.startswith("bb.") for k in own)
    got_bb = sum(k.startswith("bb.") for k in take)
    if got_bb == 0:
        raise ValueError(f"{checkpoint}: no backbone tensor matches {model.bb.__class__.__name__} — different backbone?")
    own.update(take)
    model.load_state_dict(own)
    n_dec, got_dec = len(own) - n_bb, len(take) - got_bb
    return {"checkpoint": str(checkpoint), "backbone_tensors": f"{got_bb}/{n_bb}", "decoder_tensors": f"{got_dec}/{n_dec}",
            "summary": f"backbone {got_bb}/{n_bb} tensors, decoder {got_dec}/{n_dec} tensors from {Path(checkpoint).name}"}


def _dataset_fingerprint(items: Sequence[Item]) -> str:
    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: i.image):
        h.update(Path(it.image).name.encode())
        h.update(Path(it.mask).name.encode())
    return h.hexdigest()[:16]


def train(items: Sequence[Item], out_checkpoint: PathLike, cfg: Optional[TrainConfig] = None,
          device: Optional[str] = None, overwrite: bool = False,
          on_progress: Optional[Callable[[Dict], None]] = None,
          debug_dir: Optional[PathLike] = "auto", live: bool = False) -> Path:
    """
    Train and save the best-validation checkpoint plus its model card.

    ``debug_dir``: where ``TrainingMonitor`` writes curves, CSV logs and panel
    PNGs ("auto" = ``<checkpoint stem>_debug/`` beside the checkpoint; None
    disables).  ``live``: also open a viewer window, run as a separate
    process (``monitor.py watch``).  matplotlib is never imported here.
    ``on_progress`` receives the raw record dict every ``print_every``
    iterations, for custom monitoring.  Returns the checkpoint path.
    """
    cfg = cfg or TrainConfig()
    out = Path(out_checkpoint)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; choose a new name (e.g. with the date) or pass overwrite=True")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    precision = resolve_precision(cfg.precision, device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    card = dict(DEFAULT_CARD, backbone=cfg.backbone, fpn_width=cfg.fpn_width, input_size=cfg.input_size,
                threshold=cfg.threshold, dilate_kernel=cfg.dilate_kernel, stride4=bool(cfg.stride4),
                quad_split_above=cfg.quad_split_above or None, quad_overlap=int(cfg.quad_overlap))
    if cfg.canvas:
        cw, ch = (int(v) for v in cfg.canvas)
        if cw % 32 or ch % 32:
            raise ValueError(f"canvas {cfg.canvas} must be multiples of 32 (ConvNeXt output stride)")
        card["canvas"] = [cw, ch]
    canvas = canvas_of(card)
    tr_frames, va_frames = grouped_split(items, cfg.val_ratio, cfg.seed)     # split by mask BEFORE tiling
    tr_items, tr_q = expand_quads(tr_frames, cfg.quad_split_above, cfg.quad_overlap, cfg.skip_empty_quads)
    va_items, va_q = expand_quads(va_frames, cfg.quad_split_above, cfg.quad_overlap, cfg.skip_empty_quads)
    if cfg.quad_split_above:
        print(f"[mask.train] quad split > {cfg.quad_split_above} px: train {tr_q['split']}/{tr_q['frames']} frames "
              f"-> {len(tr_items)} samples, val {va_q['split']}/{va_q['frames']} -> {len(va_items)} "
              f"({tr_q['empty_skipped'] + va_q['empty_skipped']} empty padding quadrants skipped)")
    collate = make_collate(card)
    # workers > 0: kept alive between epochs (on Windows each new worker re-imports torch)
    loader_kw = dict(collate_fn=collate, num_workers=cfg.num_workers, pin_memory=(dev_type == "cuda"),
                     persistent_workers=cfg.num_workers > 0)
    tl = DataLoader(MaskDataset(tr_items, cfg.input_size, cfg.hflip, canvas, cfg.quad_overlap),
                    batch_size=cfg.batch_size, shuffle=True, **loader_kw)
    vl = DataLoader(MaskDataset(va_items, cfg.input_size, canvas=canvas, quad_overlap=cfg.quad_overlap),
                    batch_size=cfg.batch_size, shuffle=False, **loader_kw)

    if cfg.init_from:
        model = ConvNeXtSeg(cfg.backbone, pretrained=False, fpn_width=cfg.fpn_width, stride4=cfg.stride4)
        init_report = init_from_checkpoint(model, cfg.init_from)
        print(f"[mask.train] init from {Path(cfg.init_from).name}: {init_report['summary']}")
    else:
        pfile = resolve_pretrained_file(cfg.pretrained_file, cfg.backbone) if cfg.pretrained else None
        if pfile:
            print(f"[mask.train] ImageNet-22k backbone weights from {pfile}")
        model = ConvNeXtSeg(cfg.backbone, pretrained=cfg.pretrained, fpn_width=cfg.fpn_width,
                            pretrained_file=pfile, stride4=cfg.stride4)
        init_report = {"summary": ("ImageNet-22k" + (f" ({Path(pfile).name})" if pfile else "")) if cfg.pretrained
                       else "random"}
    model = model.to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = max(1, cfg.epochs * len(tl))
    warm = max(1, int(cfg.warmup_fraction * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(dev_type, enabled=(precision == "fp16-backbone"))

    monitor = None
    if debug_dir is not None:
        from .monitor import TrainingMonitor
        ddir = out.parent / f"{out.stem}_debug" if debug_dir == "auto" else Path(debug_dir)
        monitor = TrainingMonitor(ddir, threshold=cfg.threshold, live=live)
        print(f"[mask.train] debug plots -> {ddir}")
    mean = torch.tensor(card["normalisation"]["mean"]).view(3, 1, 1)
    std = torch.tensor(card["normalisation"]["std"]).view(3, 1, 1)
    to_uint8 = lambda xt: (((xt.detach().float().cpu() * std + mean).clamp(0, 1) * 255)
                           .byte().permute(1, 2, 0).numpy())
    # fixed held-out preview frames, evenly spaced through the validation list
    prev_idx = sorted({int(round(k)) for k in np.linspace(0, len(va_items) - 1, max(1, cfg.n_val_preview))})
    preview = [vl.dataset[k] for k in prev_idx] if (monitor and cfg.val_preview_every) else []

    def val_preview(ep, it):
        model.eval()
        samples = []
        with torch.no_grad():
            for (im, ms, k) in preview:
                x1, _, _ = collate([(im, ms, k)])
                p1 = torch.sigmoid(forward(model, x1.to(device), precision, dev_type).float())[0, 0].cpu().numpy()
                samples.append((im, ms, p1, va_items[k].label))
        model.train()
        m = monitor.on_val_preview(ep, it, len(tl), samples)
        print(f"    held-out preview ({len(samples)} fixed frames) IoU {m:.4f}")

    peak = {"aspp_pre_bn": 0.0}
    model.aspp.proj.register_forward_hook(
        lambda m, i, o: peak.__setitem__("aspp_pre_bn", max(peak["aspp_pre_bn"], float(o.detach().abs().max()))))

    history: List[Dict] = []
    best = -1.0
    t0 = time.time()
    print(f"[mask.train] {cfg.backbone}{' +stride4' if cfg.stride4 else ''}  precision={precision}  device={device}  "
          f"train={len(tr_items)} val={len(va_items)} samples (grouped by mask)")
    for ep in range(cfg.epochs):
        model.train()
        s_loss = s_iou = 0.0
        i_loss = i_iou = 0.0
        i_n = 0
        for it, (x, gt, idx) in enumerate(tl, start=1):
            x, gt = x.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            logits = forward(model, x, precision, dev_type)
            loss, bce, tv = bce_tversky_loss(logits, gt)
            if not torch.isfinite(loss):
                names = [tl.dataset.items[i].label for i in idx]
                raise NonFiniteLoss(
                    f"non-finite loss at epoch {ep + 1} iteration {it} (precision {precision}, "
                    f"peak pre-BN ASPP activation {peak['aspp_pre_bn']:.3g}); batch: {names}. "
                    f"BatchNorm running statistics are now contaminated: restart from the last checkpoint.")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                                  # clip real, not scaled, gradients
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
            sched.step()
            b_loss, b_iou = float(loss.detach()), batch_iou(logits, gt, cfg.threshold)
            s_loss += b_loss
            s_iou += b_iou
            i_loss += b_loss
            i_iou += b_iou
            i_n += 1
            if it % cfg.print_every == 0 or it == len(tl):
                rec = {"epoch": ep + 1, "iter": it, "n_iter": len(tl), "loss": s_loss / it, "iou": s_iou / it,
                       "loss_interval": i_loss / i_n, "iou_interval": i_iou / i_n,
                       "bce": bce, "tversky": tv, "grad_norm": gnorm, "lr": opt.param_groups[0]["lr"],
                       "aspp_pre_bn_peak": peak["aspp_pre_bn"], "fp16_headroom": FP16_MAX / max(peak["aspp_pre_bn"], 1e-9),
                       "elapsed_s": time.time() - t0}
                print(f"  ep {ep + 1} it {it}/{len(tl)}  loss {rec['loss']:.4f}  IoU {rec['iou']:.4f}  "
                      f"(last {i_n}: loss {rec['loss_interval']:.4f} IoU {rec['iou_interval']:.4f})  "
                      f"|g| {gnorm:.3g}  lr {rec['lr']:.2e}  ASPP pre-BN peak {peak['aspp_pre_bn']:.3g}")
                i_loss = i_iou = 0.0
                i_n = 0
                if monitor or on_progress:
                    rec.update(image0=to_uint8(x[0]), gt0=gt[0].cpu().numpy(),
                               prob0=torch.sigmoid(logits[0, 0].detach().float()).cpu().numpy())
                if monitor:
                    monitor.on_progress(rec)
                if on_progress:
                    on_progress({**rec, "logits": logits.detach(), "gt": gt, "x_index": idx})
            if preview and (it % cfg.val_preview_every == 0 or it == len(tl)):
                val_preview(ep + 1, it)

        model.eval()
        v_loss = v_iou = 0.0
        with torch.no_grad():
            for x, gt, _ in vl:
                x, gt = x.to(device), gt.to(device)
                logits = forward(model, x, precision, dev_type)
                l, _, _ = bce_tversky_loss(logits, gt)
                v_loss += float(l)
                v_iou += batch_iou(logits, gt, cfg.threshold)
        v_loss /= max(1, len(vl))
        v_iou /= max(1, len(vl))
        if not math.isfinite(v_loss):
            raise NonFiniteLoss(f"non-finite validation loss after epoch {ep + 1}")
        history.append({"epoch": ep + 1, "train_loss": s_loss / len(tl), "train_iou": s_iou / len(tl),
                        "val_loss": v_loss, "val_iou": v_iou, "aspp_pre_bn_peak": peak["aspp_pre_bn"]})
        print(f"[ep {ep + 1}/{cfg.epochs}] train IoU {s_iou / len(tl):.4f}  val IoU {v_iou:.4f}  val loss {v_loss:.4f}")
        if v_iou > best:
            best = v_iou
            torch.save({"model": model.state_dict(), "val_iou": best, "epoch": ep + 1}, out)
            write_card(out, **card, name=out.stem, val_iou=best, epoch=ep + 1,
                       trained_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                       training={**asdict(cfg), "precision_resolved": precision, "device": device,
                                 "init": init_report["summary"],
                                 "n_train": len(tr_items), "n_val": len(va_items), "split": "grouped by mask file",
                                 "n_train_frames": len(tr_frames), "n_val_frames": len(va_frames),
                                 "dataset_fingerprint": _dataset_fingerprint(items)},
                       history=history)
            print(f"  -> saved {out.name} (val IoU {best:.4f})")
    if monitor:
        monitor.finish()
    return out
