# Rover masks

Every image MPPP writes has a reconstruction mask. Pixels of terrain and sky are **included**. Rover hardware,
rover shadows and other artificial objects are **excluded**: the arm, drill bits, Ingenuity, the belly pan, sample
tubes and debris. The mask keeps features on the rover from tying the stations of a block together.

The masks come from a segmentation network: a ConvNeXt-tiny backbone, a feature pyramid, a light ASPP block and a
stride-4 decoder with one logit. The models are released on Hugging Face,
[`ctate7163/mppp-mask`](https://huggingface.co/ctate7163/mppp-mask). MPPP downloads the default model on first use,
checks its SHA-256 against `src/mppp/data/models.json` and keeps it in the user cache.

## Models

| name | trained on | held-out IoU | status |
|---|---|---|---|
| `mppp_mask_v4` | v8 mask set: 4,095 masks, sols 9–2022 (Navcam, Mastcam-Z, Hazcams, WATSON) | 0.978 | prepared 31 Oct 2026; candidate default |
| `mppp_mask_v3` | v7 mask set: 3,931 masks + brightness-varied copies | 0.979 | default up to 0.95 |
| `mppp_mask_v2`, `mppp_mask_v1` | v7 mask set | 0.977, 0.971 | kept for reproducibility |

The two held-out IoUs come from different splits of different sets, so they are not a comparison;
`mppp_studies.mask_training.compare_mask_models` is (below). Held-out exclusion IoU of `mppp_mask_v4`, by camera (valid pixels only):

| camera | IoU | recall | precision |
|---|---:|---:|---:|
| Navcam | 0.976 | 0.985 | 0.990 |
| Mastcam-Z | 0.945 | 0.969 | 0.974 |
| Front Hazcam | 0.967 | 0.985 | 0.981 |

Choose a model in the processing configuration:

```python
cfg = mppp.load_config({"masking": {"checkpoint": "mppp_mask_v4"}})
```

You can also run one image directly with `mppp.mask.infer_mask(rgb, "mppp_mask_v4")`. `mppp mask list` shows
the models and what is cached, and `mppp mask fetch NAME` downloads one.

## Inference

1. The image is taken in the 8-bit-equivalent range of the MPPP colour pipeline, letterboxed to a 1664 × 1248 canvas
   (long side ≤ 1648 px) and normalised with ImageNet statistics.
2. The sigmoid output is resized back and thresholded at 0.5. The included region is then dilated by 3 × 3.
3. Invalid pixels (padding, dark columns) are excluded in any case.

All of these settings come from the model card stored in the safetensors metadata (`mppp_card`). See
[Methods §5](methods.md#5-masks).

## Training a new model

The training images are regenerated from the PDS products by the MPPP pipeline itself, so the network sees exactly
the images it will be given in production. On Windows:

```bat
D:\code\MPPP\scripts\windows\train_masks.bat
```

This batch file does the following:

1. It finds the newest `D:\masks\masks_training_set_vN`.
2. It regenerates `images\` from the PDS, without selection rules and without a landing position, so every labelled
   frame is used.
3. It trains a ConvNeXt-tiny with:
   - horizontal flips;
   - a random zoom crop (up to 2×) of the large Navcam and Hazcam frames;
   - bf16 precision;
   - a train/held-out split grouped by mask.
4. It writes the paper outputs to `Claude outputs\mask_<checkpoint>\`: the mask figure, per-camera metrics,
   training curves, the card, a training-set summary and a release-candidate `.safetensors`.

Running `mppp_studies.mask_training.train_mask` with the same options does the same on any platform.

`--zoom-pairs` (mask v4.2) puts every full-resolution Navcam frame (5,120 px) into each epoch twice: whole, downsampled
to the 1,648 px input as at inference, and as an exactly 2× crop at a random place, downsampled the same way. The
model always sees the inference view and also sees rover hardware near the Mastcam-Z pixel scale. Binned Navcam and
Hazcam frames are used once; the random zoom is off. On Windows:

```bat
D:\code\MPPP\scripts\windows\train_masks.bat --tag v8_v4.2 --zoom-pairs
```

## Comparing two models

```bat
python D:\code\MPPP\mppp_studies.mask_training.compare_mask_models --data D:\masks\masks_training_set_v8 --split-ckpt D:\code\MPPP\checkpoints\convnext_tiny_s4_seg_20261030_v8.pt --models mppp_mask_v3 D:\code\MPPP\checkpoints\convnext_tiny_s4_seg_20261030_v8.pt --seen D:\masks\masks_training_set_v7 --out "D:\code\MPPP\Claude outputs\mask_ab_v3_v4"
```

This scores both models on the held-out frames of the new model, by camera. With `--seen`, it scores them again on
the frames the older model never saw in training. That second table is the fair comparison.

## Releasing

Releasing a model takes three steps:

1. Export it to safetensors under a new registry name.
2. Add its Hugging Face URL to `models.json`.
3. Upload the file and the model card (`docs/hf_model_card.md`) with `mppp_studies.mask_training.release_mask_model`.

See [Releasing](RELEASING.md), sections 4–6.
