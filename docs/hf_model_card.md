---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - mars
  - mars-2020
  - perseverance
  - photogrammetry
  - segmentation
  - convnext
  - safetensors
---

# MPPP mask model v1 (`mppp_mask_v1`)

This is the reconstruction-mask model of [MPPP](https://github.com/ctate7163/MPPP), the Mars Photogrammetry Preprocessing Pipeline. For each pixel of a Mars 2020 Perseverance image it predicts whether that pixel should be used for photogrammetric reconstruction. **Include** means terrain. **Exclude** means rover hardware, sky, calibration targets and image artefacts.

| | |
|---|---|
| file | `mppp_mask_convnext_tiny_s4_v1.safetensors` (124,825,148 bytes) |
| SHA-256 | `678aa6ec0b9242b5359d2e3bf8c89d8325203c84d35259555a45e58ef6f67aa4` |
| architecture | ConvNeXt-tiny backbone (ImageNet-22k initialisation) + FPN + light ASPP + stride-4 decoder skip, one logit |
| input | linear 8-bit-equivalent RGB as produced by MPPP, resized to fit a 1664 × 1248 canvas (long side ≤ 1648), ImageNet normalisation |
| output | sigmoid > 0.5 = include; MPPP dilates the include region by 3 × 3 |
| training data | 7,056 training / 782 validation frames (3,921 hand-edited masks of Mars 2020 engineering-camera and Mastcam-Z frames, with brightness-varied copies), split by mask so variants never cross the split |
| training | 3 epochs, AdamW lr 1e-4, bf16, weighted BCE + Tversky loss (24 Sep 2026) |
| validation | IoU 0.971 (mean per batch; frames without terrain count as 0 in this metric, so the IoU over frames with terrain is higher) |

The model card, with every inference setting and the training history, is embedded in the safetensors metadata under `mppp_card`. `mppp.mask.model.load_model(path)` reads it.

## Use

With MPPP, the model is downloaded and checked automatically on first use:

```python
import mppp
cfg = mppp.load_config({"masking": {"infer_mask": True, "checkpoint": "mppp_mask_v1"}})
manifest = mppp.process_images(paths, "out/", cfg, mppp.load_waypoints())
```

Direct inference:

```python
from mppp.mask import infer_mask
mask, probability, card = infer_mask(rgb_uint8_image, "mppp_mask_v1")
```

## Limitations

The labels are hand-drawn polygons, so boundaries are accurate to a few pixels and some rover parts are drawn coarsely. Frames unlike the training set (other cameras, unusual illumination, dust on the optics) may be masked less reliably. The masks are meant to remove rover and sky features from photogrammetry, not to serve as precise segmentations.

## License

Apache-2.0, as for MPPP. The Mars 2020 images used for training are NASA/JPL-Caltech public data (PDS).
