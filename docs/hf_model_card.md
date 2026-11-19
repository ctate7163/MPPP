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

# RPP mask models (`rpp_mask_v5`, `rpp_mask_v4`, `rpp_mask_v3`, ...)

These are the reconstruction-mask models of [RPP](https://github.com/ctate7163/rpp), the Rover Photogrammetry
Pipeline ([documentation](https://ctate7163.github.io/rpp/)). For each pixel of a Mars 2020 Perseverance image
(and, from `rpp_mask_v5`, a Curiosity Navcam or Hazcam image), a model predicts whether that pixel should be used
for photogrammetric reconstruction:

- **Include** means terrain and sky.
- **Exclude** means rover hardware, rover shadows and other artificial objects: the robotic arm, drill bits,
  Ingenuity, the belly pan, sample tubes and debris.

The masks keep features on the rover from tying rover stations together in structure-from-motion.

Formerly `ctate7163/mppp-mask`: the pipeline was called MPPP (Mars Photogrammetry Preprocessing Pipeline) until
0.98, and the models `mppp_mask_v1` .. `mppp_mask_v4`. The files under their old names
(`mppp_mask_convnext_tiny_s4_v*.safetensors`) stay in this repository; the `rpp_mask_convnext_tiny_s4_v*` files are
the same bytes, with the same SHA-256. `rpp_mask_v5` was released after the rename and has no old name.

RPP's default is still `rpp_mask_v3` (see *Versions*); select another with `"checkpoint": "rpp_mask_v5"`.

## Newest: `rpp_mask_v5` (18 Nov 2026, with Curiosity frames)

| | |
|---|---|
| file | `rpp_mask_convnext_tiny_s4_v5.safetensors` ({{bytes:rpp_mask_v5}} bytes) |
| SHA-256 | `{{sha256:rpp_mask_v5}}` |
| architecture | as `rpp_mask_v4`: ConvNeXt-tiny + FPN (256) + light ASPP + stride-4 decoder skip, one logit (31.2 M parameters) |
| input, output | as `rpp_mask_v4`; sigmoid > 0.5 = include, no dilation of the include region |
| training data | the v8 mask set as of 18 Nov 2026, with Curiosity (MSL) Navcam and front Hazcam frames added: 4,965 training frames (198 of them Curiosity) and 479 held-out frames (RPP's frozen test list) |
| training | fine-tuned from the v5.1 run, itself fine-tuned from `rpp_mask_v3`; AdamW, lr 1e-5 (cosine, 5 % warm-up), weight decay 5e-3, batch 4, gradient clip 1.0, bf16; BCE + Tversky (α 0.3, β 0.7) with a 2 px band at the label edges left out of the loss; the checkpoint is from the first epoch (its card records one epoch) |
| augmentation | horizontal flips; a random zoom crop (1–2×) in half of the Navcam and Hazcam samples; a mild photometric change in half of the samples; gray in 10 % |
| held-out IoU | 0.978 (include class, mean per batch, the frozen test list) |

**Validation on frames no model trained on.** `validation_set_v1`: 100 frames of RPP's aligned blocks that are in
no training set (and, for the Mars 2020 cameras, no stereo partner or sequence of a training frame either),
stratified by camera; 90 were cleaned by hand and scored. Exclusion class, valid pixels only:

| camera | frames | `rpp_mask_v3` IoU | `rpp_mask_v5` IoU | `rpp_mask_v5` recall | `rpp_mask_v5` precision |
|---|---:|---:|---:|---:|---:|
| Navcam | 25 | 0.972 | 0.976 | 0.984 | 0.992 |
| Mastcam-Z | 29 | 0.961 | 0.962 | 0.984 | 0.977 |
| Front Hazcam | 10 | 0.978 | 0.972 | 0.980 | 0.991 |
| Rear Hazcam | 6 | 0.987 | 0.987 | 0.995 | 0.992 |
| Curiosity Navcam | 14 | 0.971 | 0.983 | 0.989 | 0.993 |
| Curiosity front Hazcam | 6 | 0.934 | 0.968 | 0.978 | 0.990 |
| all | 90 | 0.973 | 0.974 | 0.983 | 0.991 |

Over all 90 frames `rpp_mask_v5` is 0.0017 higher than `rpp_mask_v3` (95 % interval -0.0063 to +0.0091, paired by
frame): no difference overall. What it adds is the Curiosity cameras, which `rpp_mask_v3` never saw (Navcam 0.983
against 0.971, front Hazcam 0.968 against 0.934); on the Mars 2020 front Hazcam it is a little lower (0.972 against
0.978). Read the numbers with three cautions:

- The hand-cleaned masks started from the mean of three other models (v3 and two unreleased runs), so any lean of
  the labels favors those, not v5.
- It is one training run. An earlier run of the recipe without the Curiosity frames scored 0.977 (its checkpoint was
  not kept), and runs kept at the end of a full schedule 0.965 to 0.971: run-to-run spread is larger than the
  difference from v3.
- Some Curiosity validation frames may share sequences with Curiosity training frames; for those cameras the
  independence is weaker than for the Mars 2020 cameras.

## Versions

| name | file | trained | data | held-out IoU | notes |
|---|---|---|---|---|---|
| `rpp_mask_v5` | `rpp_mask_convnext_tiny_s4_v5.safetensors` | 18 Nov 2026, fine-tune of v5.1 (from v3), epoch 1 | v8 set with Curiosity frames, 4,965 training frames | 0.978 | newest; 0.974 on `validation_set_v1` (v3: 0.973); the model for Curiosity frames |
| `rpp_mask_v4` | `rpp_mask_convnext_tiny_s4_v4.safetensors` | 30 Oct 2026, best of 20 epochs (13) | v8 set, 4,095 masks, pipeline images only, zoom crops | 0.978 | |
| `rpp_mask_v3` | `rpp_mask_convnext_tiny_s4_v3.safetensors` | 26 Sep 2026, best of 10 epochs (9) | v7 set, 3,931 masks + brightness-varied copies | 0.979 | RPP's default |
| `rpp_mask_v2` | `rpp_mask_convnext_tiny_s4_v2.safetensors` | 25 Sep 2026, 4 epochs | v7 set | 0.977 | |
| `rpp_mask_v1` | `rpp_mask_convnext_tiny_s4_v1.safetensors` | 24 Sep 2026, 3 epochs | v7 set | 0.971 | |

The held-out IoUs in this table come from different splits of different sets, so they are not a comparison:
`validation_set_v1` above is the one set on which the models are compared fairly. RPP's
`rpp_studies.mask_training.compare_mask_models` scores any models on the same frames. The default is set in RPP's
`src/rpp/data/models.json`; select a model with `"checkpoint": "rpp_mask_v5"`.

All versions have the same architecture and input. Each file carries its full model card (every inference setting
and the training history) in its safetensors metadata under `mppp_card`; `rpp.mask.model.load_model(path)` reads it.

## Use

With RPP, the model is downloaded from this repository on first use, checked against its SHA-256 and kept in the
user cache:

```python
import rpp
cfg = rpp.load_config({"masking": {"infer_mask": True, "checkpoint": "rpp_mask_v5"}})
manifest = rpp.process_images(paths, "out/", cfg, rpp.load_waypoints())
```

Direct inference:

```python
from rpp.mask import infer_mask
mask, probability, card = infer_mask(rgb_uint8_image, "rpp_mask_v5")
```

Without RPP:
- `safetensors.torch.load_file(path)` gives the weights.
- `safetensors.safe_open(path, "pt").metadata()["mppp_card"]` gives the card (JSON).

To retrain on a newer mask set, use RPP's `rpp_studies.mask_training.train_mask`, or `scripts/windows/train_masks.bat` on Windows.

## Limitations

- **Boundaries:** the labels are hand-drawn polygons, so boundaries are accurate to a few pixels, and some rover parts are drawn coarsely.
- **Mastcam-Z:** it is the hardest camera (exclusion IoU 0.945 for v4 on its held-out frames, 0.968 for v5 on `validation_set_v1`). Its narrow fields show small rover parts and shadows at high magnification, and those are where most of the errors are.
- **Hazcams and WATSON:** they have few held-out frames (2–15), so their scores are uncertain.
- **Curiosity:** only `rpp_mask_v5` has seen Curiosity frames, and only Navcam and front Hazcam ones; Mastcam and the rear Hazcams of Curiosity are untested.
- **Unfamiliar frames:** frames unlike the training set (other cameras, unusual illumination, dust on the optics) may be masked less reliably.
- **Intended use:** the masks are meant to remove rover features from photogrammetry, not to serve as precise segmentations.

## Provenance and license

Apache-2.0, as for RPP, released with the approval of Malin Space Science Systems. The training images are public
Mars 2020 and Mars Science Laboratory products from the NASA Planetary Data System (Navcam and Hazcams:
NASA/JPL-Caltech; Mastcam-Z: NASA/JPL-Caltech/ASU/MSSS; SHERLOC WATSON: NASA/JPL-Caltech). The masks (labels) and
the models are the author's own work.

## Citation

Please cite RPP as described in its
[CITATION.cff](https://github.com/ctate7163/rpp/blob/main/CITATION.cff).
