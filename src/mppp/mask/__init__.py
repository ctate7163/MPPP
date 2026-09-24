"""
mppp.mask — reconstruction masks (terrain vs rover hardware, sky, artefacts).

Inference with the released model (``infer_mask``; the model is downloaded
once, see :mod:`mppp.mask.hub`) or with any ConvNeXtSeg checkpoint and its
model card.  Training (``mask.train``), dataset tools (``mask.dataset``),
progress plots (``mask.monitor``) and the label audit (``mask.audit``) are
included for completeness; they are not needed to process images.

torch / timm are only imported when this sub-package is used.
"""
from .infer import infer_mask, get_model, predict_probability, probability_to_mask  # noqa: F401
from .model import ConvNeXtSeg, load_model, read_card, write_card, DEFAULT_CARD      # noqa: F401
from .hub import fetch_model, install_model, export_safetensors, resolve_checkpoint  # noqa: F401
