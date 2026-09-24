"""
ConvNeXt segmentation network (terrain = 1 / exclude = 0) and checkpoint I/O.

Architecture of the network that produced ``convnext_tiny_seg_best.pt``:
timm ConvNeXt backbone (features_only) -> 3-level FPN (strides 8/16/32) ->
lite ASPP (stride 8) -> x2 upsample -> 1-channel logits at stride 4, bilinearly
resized to the input size.

``stride4=True`` (v0p6, the training default) adds the backbone's first stage
(stride 4) as a DeepLabv3+-style skip: 1x1 projection to 48 channels,
concatenated with the upsampled ASPP output, fused by a 3x3 conv.  Without it
the boundary is decided on the stride-8 grid (8 x 8 input pixels per cell).
Cards written before v0p6 have no ``stride4`` key and load as ``False``.

Every checkpoint ``<name>.pt`` travels with a model card ``<name>.json``.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

PathLike = Union[str, Path]

SUPPORTED_BACKBONES = ("convnext_tiny", "convnext_base")

DEFAULT_CARD: Dict[str, Any] = {
    "card_version": 1,
    "architecture": "ConvNeXtSeg",
    "backbone": "convnext_tiny",
    "fpn_width": 256,
    "input_size": 1648,              # long side after resizing (never enlarged beyond the canvas)
    # canvas [W, H] the resized image is zero-padded into (bottom/right).  Absent in
    # cards written before v0p5 -> square [input_size, input_size], the old behaviour.
    "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "input_range": "0-255 linear 8-bit-equivalent RGB (MPPP colour pipeline)",
    "output": "logit; sigmoid > threshold = include in reconstruction (terrain)",
    "threshold": 0.4,
    "dilate_kernel": 3,
    "state_dict_key": "model",
    "stride4": False,                # v0p6: stride-4 skip in the decoder (absent in older cards)
    # v0p6: frames whose long side exceeds this are processed as four overlapping quadrants
    # (training and inference).  None (absent in older cards) = never split.
    "quad_split_above": None,
    "quad_overlap": 64,              # px at full resolution, each side of the centre lines
}

S4_CHANNELS = 48


class TinyFPN(nn.Module):
    def __init__(self, chs, out_ch=192):
        super().__init__()
        c2, c3, c4 = chs
        self.l2 = nn.Conv2d(c2, out_ch, 1, bias=False)
        self.l3 = nn.Conv2d(c3, out_ch, 1, bias=False)
        self.l4 = nn.Conv2d(c4, out_ch, 1, bias=False)
        self.bn2, self.bn3, self.bn4 = (nn.BatchNorm2d(out_ch) for _ in range(3))
        self.act = nn.ReLU(inplace=True)
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, c2, c3, c4):
        p4 = self.act(self.bn4(self.l4(c4)))
        p3 = self.act(self.bn3(self.l3(c3))) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.act(self.bn2(self.l2(c2))) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(p2)


class LiteASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.b2 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2, bias=False)
        self.b3 = nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4, bias=False)
        self.proj = nn.Conv2d(out_ch * 3, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.proj(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))
        return self.act(self.bn(y))


def canvas_of(card: Dict[str, Any]) -> Tuple[int, int]:
    """(W, H) of the network input canvas described by a model card."""
    c = card.get("canvas")
    s = int(card["input_size"])
    return (int(c[0]), int(c[1])) if c else (s, s)


def fit_to_canvas(h: int, w: int, input_size: int, canvas: Tuple[int, int]) -> Tuple[int, int]:
    """New (h, w): long side -> input_size, then shrunk further if needed to fit the canvas."""
    cw, ch = canvas
    r = min(input_size / max(h, w), cw / w, ch / h)
    return min(ch, int(round(h * r))), min(cw, int(round(w * r)))


Box = Tuple[int, int, int, int]            # y0, y1, x0, x1 (end-exclusive)


def needs_quad_split(h: int, w: int, quad_split_above: Optional[int]) -> bool:
    """True when the long side exceeds ``quad_split_above`` (None / 0 = never)."""
    return bool(quad_split_above) and max(h, w) > int(quad_split_above)


def quad_boxes(h: int, w: int, overlap: int = 64) -> List[Tuple[Box, Box]]:
    """
    The four quadrants of an h x w frame as ``(crop, core)`` pairs, in the order
    top-left, top-right, bottom-left, bottom-right.  ``crop`` extends ``overlap``
    px past the centre lines (context for the network); ``core`` is the exact
    quadrant, in frame coordinates, that the crop's prediction is kept for.
    The cores tile the frame exactly.
    """
    ym, xm = h // 2, w // 2
    oy, ox = min(int(overlap), h - ym), min(int(overlap), w - xm)
    out = []
    for (cy0, cy1, y0, y1) in ((0, ym, 0, min(h, ym + oy)), (ym, h, max(0, ym - oy), h)):
        for (cx0, cx1, x0, x1) in ((0, xm, 0, min(w, xm + ox)), (xm, w, max(0, xm - ox), w)):
            out.append(((y0, y1, x0, x1), (cy0, cy1, cx0, cx1)))
    return out


def _hub_reachable(host: str = "huggingface.co", timeout: float = 3.0) -> bool:
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False
    finally:
        socket.setdefaulttimeout(None)


def _cached_weights(model_name: str):
    """Path of timm/<model_name> weights already in the local Hugging Face cache, or None."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        p = try_to_load_from_cache(f"timm/{model_name}", fname)
        if isinstance(p, str) and Path(p).is_file():
            return p
    return None


def read_state_dict(path: PathLike) -> Dict[str, torch.Tensor]:
    """Tensors from a ``.safetensors`` or torch (``.pt``/``.pth``/``.bin``) file; unwraps {"model": ...}."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"weights file not found: {path}")
    with open(path, "rb") as f:
        head = f.read(16)
    is_st = path.suffix.lower() == ".safetensors" or (len(head) >= 9 and head[8:9] == b"{")
    if is_st:
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:                                          # noqa: BLE001  e.g. FAIR .pth with argparse args
        warnings.warn(f"{Path(path).name} holds pickled Python objects; loading it with weights_only=False "
                      f"(only do this for files you trust)")
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "model_state"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    return state


def load_backbone_weights(bb: nn.Module, backbone: str, path: PathLike) -> Dict[str, Any]:
    """
    Load ImageNet weights for a timm ConvNeXt into the ``features_only`` backbone
    ``bb``: a timm/Hugging Face ``model.safetensors`` or an original FAIR
    ``convnext_*.pth`` (converted with timm's filter).  The classifier head is
    dropped, so in1k, in22k and fine-tuned files all work.  Every backbone
    tensor must be matched, else ValueError (e.g. tiny weights for base).
    """
    import timm
    from timm.models.convnext import checkpoint_filter_fn
    state = read_state_dict(path)
    if any(k.startswith("bb.") for k in state):
        raise ValueError(f"{Path(path).name} is a ConvNeXtSeg checkpoint: use init_from=, not pretrained_file=")
    full = timm.create_model(backbone, pretrained=False, num_classes=0)
    state = checkpoint_filter_fn(state, full)                       # FAIR -> timm names; no-op for timm files
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    renamed = {}
    for k, v in state.items():
        top, _, rest = k.partition(".")
        if top in ("stem", "stages") and rest[:1].isdigit():
            idx, _, tail = rest.partition(".")
            k = f"{top}_{idx}.{tail}"
        renamed[k] = v
    own = bb.state_dict()
    bad = [k for k in own if k not in renamed or tuple(renamed[k].shape) != tuple(own[k].shape)]
    if bad:
        raise ValueError(f"{Path(path).name} does not fit timm {backbone}: {len(bad)} of {len(own)} backbone "
                         f"tensors missing or mis-shaped (e.g. {bad[0]}) — wrong model size?")
    bb.load_state_dict({k: renamed[k] for k in own}, strict=True)
    return {"file": str(path), "tensors": len(own), "summary": f"ImageNet backbone {len(own)}/{len(own)} tensors "
            f"from {Path(path).name}"}


def create_backbone(backbone: str, pretrained: bool, pretrained_file: str = None):
    """
    timm ConvNeXt (features_only).  ImageNet weights, in order: ``pretrained_file``
    (a local timm/HF ``model.safetensors`` or ``.pth``), the Hugging Face hub, the
    local HF cache (used automatically when the hub is unreachable).  Fails with
    instructions rather than silently training from random weights.
    """
    import os
    import timm
    if not pretrained:
        return timm.create_model(backbone, features_only=True, pretrained=False)
    if pretrained_file:
        bb = timm.create_model(backbone, features_only=True, pretrained=False)
        load_backbone_weights(bb, backbone, pretrained_file)
        return bb
    offline = not _hub_reachable()
    errors = []
    for name in (f"{backbone}.fb_in22k", backbone):
        if offline:                                           # no network: look in the HF cache only
            cached = _cached_weights(name)
            if cached:
                return timm.create_model(backbone, features_only=True, pretrained=True,
                                         pretrained_cfg_overlay=dict(file=cached))
            errors.append(f"{name}: not in local cache")
            continue
        try:
            return timm.create_model(name, features_only=True, pretrained=True)
        except Exception as e:                                # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}")
    raise RuntimeError(
        f"No ImageNet weights for {backbone}: huggingface.co is {'unreachable' if offline else 'reachable'} "
        f"and the local cache has none ({'; '.join(errors)}).  Options: "
        f"(1) TrainConfig(init_from=<existing ConvNeXtSeg checkpoint with the same backbone>);  "
        f"(2) TrainConfig(pretrained_file=<model.safetensors of timm/{backbone}.fb_in22k, or a FAIR .pth>) downloaded on a "
        f"networked machine, e.g. https://huggingface.co/timm/{backbone}.fb_in22k/resolve/main/model.safetensors "
        f"saved as checkpoints/{backbone}.fb_in22k.safetensors;  (3) pretrained=False (random init, not recommended).")


class ConvNeXtSeg(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = False, fpn_width: int = 256,
                 pretrained_file: str = None, stride4: bool = False):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"backbone must be one of {SUPPORTED_BACKBONES}; got {backbone!r}")
        self.backbone_name = backbone
        self.bb = create_backbone(backbone, pretrained, pretrained_file)
        chs = self.bb.feature_info.channels()
        self.stride4 = bool(stride4)
        self.pick = [-4, -3, -2, -1] if self.stride4 else [-3, -2, -1]
        self.fpn = TinyFPN([chs[i] for i in (-3, -2, -1)], out_ch=fpn_width)
        self.aspp = LiteASPP(fpn_width, fpn_width)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        if self.stride4:
            self.s4 = nn.Sequential(nn.Conv2d(chs[-4], S4_CHANNELS, 1, bias=False),
                                    nn.BatchNorm2d(S4_CHANNELS), nn.ReLU(inplace=True))
            self.fuse = nn.Sequential(nn.Conv2d(fpn_width + S4_CHANNELS, fpn_width, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(fpn_width), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Conv2d(fpn_width, fpn_width // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_width // 2), nn.ReLU(inplace=True),
            nn.Conv2d(fpn_width // 2, 1, 1))
        with torch.no_grad():
            self.head[-1].bias.fill_(math.log(0.55 / 0.45))

    def features(self, x):
        """Backbone only (safe in fp16: residual stream stays < ~2e4 on M2020 imagery)."""
        feats = self.bb(x)
        return [feats[i] for i in self.pick]

    def decode(self, feats, size):
        """
        FPN -> ASPP -> head.  fpn.smooth -> aspp.b* -> aspp.proj is a chain of
        three convolutions with no normalisation until aspp.bn, so the loss is
        invariant to their scale and their weight norms drift upward during
        training.  Pre-BN activations reached 2.3e4 on an ordinary frame with
        the 2025 checkpoint; fp16 overflows at 6.55e4.  Run this part in fp32
        or bf16, never fp16.
        """
        if self.stride4:
            c1, c2, c3, c4 = feats
            z = self.aspp(self.fpn(c2, c3, c4))
            z = F.interpolate(z, size=c1.shape[-2:], mode="bilinear", align_corners=False)
            z = self.fuse(torch.cat([z, self.s4(c1)], dim=1))
        else:
            c2, c3, c4 = feats
            z = self.up2(self.aspp(self.fpn(c2, c3, c4)))
        return F.interpolate(self.head(z), size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.decode(self.features(x), x.shape[-2:])


def card_path(checkpoint: PathLike) -> Path:
    return Path(checkpoint).with_suffix(".json")


SAFETENSORS_CARD_KEY = "mppp_card"          # v0p13: the model card travels inside the .safetensors metadata


def _safetensors_card(checkpoint: PathLike) -> Optional[Dict[str, Any]]:
    from safetensors import safe_open
    with safe_open(str(checkpoint), framework="pt") as f:
        meta = f.metadata() or {}
    return json.loads(meta[SAFETENSORS_CARD_KEY]) if SAFETENSORS_CARD_KEY in meta else None


def read_card(checkpoint: PathLike) -> Dict[str, Any]:
    """Model card: embedded in a ``.safetensors`` file, else the ``<name>.json`` beside the checkpoint."""
    card = dict(DEFAULT_CARD)
    p = card_path(checkpoint)
    emb = _safetensors_card(checkpoint) if Path(checkpoint).suffix == ".safetensors" and Path(checkpoint).is_file() \
        else None
    if emb is not None:
        card.update(emb)
    elif p.is_file():
        card.update(json.loads(p.read_text(encoding="utf-8")))
    else:
        warnings.warn(f"No model card {p.name} next to the checkpoint; assuming the "
                      f"convnext_tiny defaults.")
    return card


def read_state_dict_checkpoint(checkpoint: PathLike, card: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Weights of a ConvNeXtSeg checkpoint.  ``.safetensors``: plain tensors, no
    code.  ``.pt``: loaded with ``torch.load(weights_only=True)`` (v0p13; MPPP
    checkpoints hold only tensors and numbers), so a file from elsewhere
    cannot execute code on load.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(checkpoint), device="cpu")
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except Exception as e:                                    # noqa: BLE001
        raise RuntimeError(f"{checkpoint.name} is not a plain-tensor checkpoint ({type(e).__name__}: {e}). "
                           f"If you trust it, convert it once with torch.load(..., weights_only=False) and "
                           f"mppp.mask.hub.export_safetensors.") from e
    key = (card or {}).get("state_dict_key", "model")
    return state[key] if isinstance(state, dict) and key in state else state


def write_card(checkpoint: PathLike, **fields: Any) -> Path:
    card = dict(DEFAULT_CARD)
    card.update(fields)
    p = card_path(checkpoint)
    p.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return p


def load_model(checkpoint: PathLike, device: str = "cpu") -> Tuple[ConvNeXtSeg, Dict[str, Any]]:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Mask checkpoint not found: {checkpoint}")
    card = read_card(checkpoint)
    model = ConvNeXtSeg(backbone=card["backbone"], pretrained=False, fpn_width=int(card["fpn_width"]),
                        stride4=bool(card.get("stride4", False)))
    model.load_state_dict(read_state_dict_checkpoint(checkpoint, card), strict=True)
    return model.to(device).eval(), card
