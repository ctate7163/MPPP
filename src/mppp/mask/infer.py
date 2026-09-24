"""Mask inference with a cached model (loaded once per checkpoint and device)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import canvas_of, fit_to_canvas, load_model, needs_quad_split, quad_boxes

PathLike = Union[str, Path]


def pick_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


@lru_cache(maxsize=4)
def _cached(checkpoint: str, device: str):
    return load_model(checkpoint, device)


def get_model(checkpoint: Optional[PathLike] = None, device: str = "auto"):
    """(model, card) for a checkpoint path, a registry name (default: the released model) or a file
    name in ``checkpoints_dir()`` - see :func:`mppp.mask.hub.resolve_checkpoint`.  Cached."""
    from .hub import resolve_checkpoint
    path = resolve_checkpoint(checkpoint)
    return _cached(str(path), pick_device(device))


_CARD = object()          # sentinel: "take the value from the model card"


@torch.no_grad()
def predict_probability(model, card: Dict[str, Any], image_0_255: np.ndarray,
                        quad_split_above: Any = _CARD) -> np.ndarray:
    """
    ``image_0_255``: H x W x 3 float/uint8 RGB on a 0-255 scale.
    Returns the H x W float32 probability of 'include' (terrain).

    Frames whose long side exceeds ``quad_split_above`` (default: the card's
    value; None in pre-v0p6 cards = never) are run as four overlapping
    quadrants, exactly as the model was trained, and stitched from the
    quadrant cores.  A quadrant whose core is all zero (padding) is not run:
    probability 0 there.
    """
    if image_0_255.ndim != 3 or image_0_255.shape[2] != 3:
        raise ValueError("image must be H x W x 3")
    thr = card.get("quad_split_above") if quad_split_above is _CARD else quad_split_above
    H0, W0 = image_0_255.shape[:2]
    if not needs_quad_split(H0, W0, thr):
        return _predict_frame(model, card, image_0_255)
    out = np.zeros((H0, W0), np.float32)
    for (y0, y1, x0, x1), (cy0, cy1, cx0, cx1) in quad_boxes(H0, W0, int(card.get("quad_overlap", 64))):
        if not np.any(image_0_255[cy0:cy1, cx0:cx1]):      # quadrant is pure padding
            continue
        crop = image_0_255[y0:y1, x0:x1]
        p = _predict_frame(model, card, crop)
        out[cy0:cy1, cx0:cx1] = p[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    return out


@torch.no_grad()
def _predict_frame(model, card: Dict[str, Any], image_0_255: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    cw, ch = canvas_of(card)
    H0, W0 = image_0_255.shape[:2]
    nh, nw = fit_to_canvas(H0, W0, int(card["input_size"]), (cw, ch))
    img = cv2.resize(np.clip(image_0_255, 0, 255).astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = np.pad(img, ((0, ch - nh), (0, cw - nw), (0, 0)))
    x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device) / 255.0
    mean = torch.tensor(card["normalisation"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(card["normalisation"]["std"], device=device).view(1, 3, 1, 1)
    p = torch.sigmoid(model((x - mean) / std))[:, :, :nh, :nw]
    p = F.interpolate(p, size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
    return p.float().cpu().numpy()


def probability_to_mask(prob: np.ndarray, threshold: float, dilate_kernel: int = 0) -> np.ndarray:
    """uint8 mask, 255 = include, 0 = exclude; the include region is dilated."""
    m = (prob > threshold).astype(np.uint8)
    if dilate_kernel and dilate_kernel > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel,) * 2))
    return m * np.uint8(255)


def infer_mask(image_0_255: np.ndarray, checkpoint: PathLike, device: str = "auto",
               threshold: Optional[float] = None, dilate_kernel: Optional[int] = None,
               quad_split_above: Any = _CARD) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """-> (mask uint8 0/255, probability float32, model card).  ``quad_split_above``:
    see :func:`predict_probability` (default from the card; None disables)."""
    model, card = get_model(checkpoint, device)
    prob = predict_probability(model, card, image_0_255, quad_split_above)
    thr = card["threshold"] if threshold is None else threshold
    dk = card["dilate_kernel"] if dilate_kernel is None else dilate_kernel
    return probability_to_mask(prob, float(thr), int(dk)), prob, card
