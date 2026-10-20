"""
mppp.mask — reconstruction masks (terrain vs rover hardware, rover shadows and other artificial objects; the sky is not masked).

Inference with the released model (``infer_mask``; the model is downloaded
once, see :mod:`mppp.mask.hub`) or with any ConvNeXtSeg checkpoint and its
model card.  Retraining (:mod:`mppp.mask.training`: the data set, the training
loop, progress plots and the label audit) is included for completeness; it is
not needed to process images.

torch / timm are only imported when this sub-package is used.
"""
from .infer import infer_mask, get_model, predict_probability, probability_to_mask  # noqa: F401
from .model import ConvNeXtSeg, load_model, read_card, write_card, DEFAULT_CARD      # noqa: F401
from .hub import fetch_model, install_model, export_safetensors, resolve_checkpoint  # noqa: F401
