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

# MPPP mask models (`mppp_mask_v4`, `mppp_mask_v3`, ...)

These are the reconstruction-mask models of [MPPP](https://github.com/ctate7163/MPPP), the Mars Photogrammetry
Preprocessing Pipeline ([documentation](https://ctate7163.github.io/MPPP/)). For each pixel of a Mars 2020
Perseverance image, a model predicts whether that pixel should be used for photogrammetric reconstruction:

- **Include** means terrain and sky.
- **Exclude** means rover hardware, rover shadows and other artificial objects: the robotic arm, drill bits,
  Ingenuity, the belly pan, sample tubes and debris.

The masks keep features on the rover from tying rover stations together in structure-from-motion.

## Newest: `mppp_mask_v4` (30 Oct 2026, trained on the v8 mask set)

| | |
|---|---|
| file | `mppp_mask_convnext_tiny_s4_v4.safetensors` (124,827,092 bytes) |
| SHA-256 | `9ecb3dd5fb25fa4f21df0034c575a66d46a8a49b6f500129409bc2cf3ec8a467` |
| architecture | ConvNeXt-tiny backbone (ImageNet-22k initialisation) + FPN (256) + light ASPP + stride-4 decoder skip, one logit (31.2 M parameters) |
| input | linear 8-bit-equivalent RGB as produced by MPPP, letterboxed to a 1664 × 1248 canvas (long side ≤ 1648), ImageNet normalisation |
| output | sigmoid > 0.5 = include; MPPP dilates the include region by 3 × 3 |
| training data | 4,095 hand-edited masks of Mars 2020 frames, sols 9–2022: Navcam 1,925, Mastcam-Z 1,983, Front Hazcam 82, Rear Hazcam 31, SHERLOC WATSON 74. The images come from the standard MPPP pipeline only (no brightness-varied copies). |
| split | 3,685 training / 410 held-out frames, grouped by mask (seed 42) |
| training | AdamW, lr 5e-5 (cosine, 5 % warm-up), weight decay 5e-3, batch 4, gradient clip 1.0, bf16, BCE + Tversky (α 0.3, β 0.7) loss; 20 epochs, best held-out IoU at epoch 13 (kept) |
| augmentation | horizontal flips; for the large Navcam and Hazcam frames, a random zoom crop in half of the samples (a crop of 1/z of the frame, z uniform in 1–2) so the model also sees them at closer to native scale |
| held-out IoU | 0.978 (include class, mean per batch) |

Held-out frames by camera. The scores use valid pixels only, for the **exclusion** class (the error that matters for SfM). Recall is the share of rover pixels the model also excludes.

| camera | held-out frames | IoU | recall | precision |
|---|---:|---:|---:|---:|
| Navcam | 186 | 0.976 | 0.985 | 0.990 |
| Mastcam-Z | 200 | 0.945 | 0.969 | 0.974 |
| Front Hazcam | 15 | 0.967 | 0.985 | 0.981 |
| Rear Hazcam | 2 | 0.976 | 0.987 | 0.989 |
| SHERLOC WATSON | 7 | 0.974 | 0.983 | 0.991 |

![Held-out frames: orange = excluded by the model, blue line = hand-label boundary](figures/figure1_masks.png)

![Training curves](figures/training_curves.png)

## Versions

| name | file | trained | data | held-out IoU | notes |
|---|---|---|---|---|---|
| `mppp_mask_v4` | `mppp_mask_convnext_tiny_s4_v4.safetensors` | 30 Oct 2026, best of 20 epochs (13) | v8 set, 4,095 masks, pipeline images only, zoom crops | 0.978 | newest |
| `mppp_mask_v3` | `mppp_mask_convnext_tiny_s4_v3.safetensors` | 26 Sep 2026, best of 10 epochs (9) | v7 set, 3,931 masks + brightness-varied copies | 0.979 | MPPP's default up to 0.95 |
| `mppp_mask_v2` | `mppp_mask_convnext_tiny_s4_v2.safetensors` | 25 Sep 2026, 4 epochs | v7 set | 0.977 | |
| `mppp_mask_v1` | `mppp_mask_convnext_tiny_s4_v1.safetensors` | 24 Sep 2026, 3 epochs | v7 set | 0.971 | |

The held-out IoUs of v3 and v4 come from different splits of different sets, so they are not a comparison. MPPP's
`mppp_studies.mask_training.compare_mask_models` scores both on the same held-out frames, including the subset that v3 never saw.
The default is set in MPPP's `src/mppp/data/models.json`; select a model with `"checkpoint": "mppp_mask_v4"`.

All versions have the same architecture and input. Each file carries its full model card (every inference setting
and the training history) in its safetensors metadata under `mppp_card`; `mppp.mask.model.load_model(path)` reads it.

## Use

With MPPP, the model is downloaded from this repository on first use, checked against its SHA-256 and kept in the
user cache:

```python
import mppp
cfg = mppp.load_config({"masking": {"infer_mask": True, "checkpoint": "mppp_mask_v4"}})
manifest = mppp.process_images(paths, "out/", cfg, mppp.load_waypoints())
```

Direct inference:

```python
from mppp.mask import infer_mask
mask, probability, card = infer_mask(rgb_uint8_image, "mppp_mask_v4")
```

Without MPPP:
- `safetensors.torch.load_file(path)` gives the weights.
- `safetensors.safe_open(path, "pt").metadata()["mppp_card"]` gives the card (JSON).

To retrain on a newer mask set, use MPPP's `mppp_studies.mask_training.train_mask`, or `scripts/windows/train_masks.bat` on Windows.

## Limitations

- **Boundaries:** the labels are hand-drawn polygons, so boundaries are accurate to a few pixels, and some rover parts are drawn coarsely.
- **Mastcam-Z:** it is the hardest camera (exclusion IoU 0.945). Its narrow fields show small rover parts and shadows at high magnification, and those are where most of the errors are.
- **Hazcams and WATSON:** they have few held-out frames (2–15), so their scores are uncertain.
- **Unfamiliar frames:** frames unlike the training set (other cameras, unusual illumination, dust on the optics) may be masked less reliably.
- **Intended use:** the masks are meant to remove rover features from photogrammetry, not to serve as precise segmentations.

## Provenance and license

Apache-2.0, as for MPPP, released with the approval of Malin Space Science Systems. The training images are public
Mars 2020 products from the NASA Planetary Data System (Navcam and Hazcams: NASA/JPL-Caltech; Mastcam-Z:
NASA/JPL-Caltech/ASU/MSSS; SHERLOC WATSON: NASA/JPL-Caltech). The masks (labels) and the models are the author's own
work.

## Citation

Please cite MPPP as described in its
[CITATION.cff](https://github.com/ctate7163/MPPP/blob/main/CITATION.cff).
