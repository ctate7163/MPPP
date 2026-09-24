# Changelog

## v0p12 — 2026-09-24

### Frame audit: statistics that work for frames without terrain
- **Problem:** the ranking by IoU put every frame without terrain first (sky, calibration target, rover deck). One false pixel gives IoU 0, and empty truth + empty prediction also gave 0.
- **New default ranking: `error_pct`**, the % of the frame that is wrong (missed + false). This is what the frame adds to the training error.
- Two new per-frame statistics, both in % of the frame (`frame_stats`):
  - `interior_error_pct`: wrong pixels farther than `band_px` from any label edge (0.4 % of the long side: 20 px at 5120, 7 px at 1648). It finds wrong or missing regions, not boundary jitter.
  - `confident_error_pct`: the model is sure (P > 0.9 or < 0.1) and the label disagrees.
- **`rank_rows(rows, sort_by, split=, min_terrain_pct=)`** re-ranks a saved audit without re-running the model. `sort_by="iou"` leaves out frames with < 1 % terrain.
- `summarize` now reports the error % per split (mean, median, counts above 1/5/20 %) and the mean IoU of frames with terrain only.
- `worst_table` shows all the statistics. `render_worst(view, ckpt, sort_by=)` draws the rows in the given order.
- **`panel_stats`** (training preview and panels): no terrain in both truth and prediction now scores IoU 1, not 0. This slightly changes the held-out preview mean when a preview frame has no terrain.
- **Not changed: the training `batch_iou`** still scores empty/empty as 0. On the 24 Sep audit that lowers val IoU by about 0.015 (0.968 over all val frames vs 0.983 over frames with terrain). It is left as is so that val IoUs stay comparable with the base/tiny runs.
- **Notebook `02_train_mask_v0p12`:** `SORT_BY` / `SPLIT` in the audit cell. A pre-v0p12 CSV is recomputed once, since it lacks the new columns. Notebooks 01 and 03 are renamed to v0p12.
- Tests: `tests/test_v0p12.py`:
  - empty/empty = 1;
  - boundary jitter is not interior error, but a missed region is;
  - confident error;
  - re-ranking, filters, a pre-v0p12 CSV.

## v0p11 — 2026-09-24

### GPU extraction and matching on Windows (`mppp.sfm.gpu`)
- **Problem:** the pip `pycolmap` wheel for Windows has no CUDA. The conda-forge CUDA build (`has_cuda` True) cannot be used with the pip `pyceres` wheel. The weighted BA takes pycolmap's `ceres::Problem` and cost functions into pyceres, and the two packages are compiled separately, so pybind11 keeps separate type registries. The result was `TypeError: Unregistered type : ceres::Problem` (2 of 7 `test_sfm.py` tests in the CUDA env).
- **Fix:** keep the notebook kernel on pip pycolmap 4.2 + pyceres 2.6, and run only the two GPU steps in the CUDA environment as a subprocess.
  - Usage: `extract_features(proj, ..., python=GPU_PY)` and `match(proj, ..., python=GPU_PY)`.
  - The two environments share only files (`features.db`, `database.db`, `project.json`, `pairs_prior.txt`). The settings each step records are reloaded into the kernel's project.
  - COLMAP's log streams into the notebook.
  - The subprocess gets the conda environment's DLL folders on PATH, so `conda activate` is not needed.
- **`check_gpu_python(GPU_PY)`** reports the pycolmap version and `has_cuda` of the CUDA environment. It refuses a different COLMAP major.minor, because the database format must match.
- `bundle_adjust` now explains the mismatch, instead of the bare pybind11 error, when pycolmap and pyceres don't share Ceres types.
- Notebook `03_colmap_alignment_v0p11`: `GPU_PY` in the first cell (None = CPU in the kernel). Notebooks 01 and 02 are renamed to v0p11 without other changes.
- Test: extraction and matching through a subprocess (this interpreter as the second environment); settings round trip; version check.
- `check_ba_environment()` (first cell of notebook 03) stops at once, with instructions, if the notebook kernel cannot run the weighted BA. The typical cause is running the notebook in the CUDA environment itself. `check_gpu_python` warns when `GPU_PY` is the kernel's own python.
- Test fix: the fp16-overflow reproduction now always uses the 2025 checkpoint (`convnext_tiny_seg_best.pt`, skipped if absent). After `promote_checkpoint`, the default model (ASPP peak ~44) never overflows, so the test failed for the wrong reason.

## v0p10 — 2026-09-24

### Mask training defaults (requested after the base vs tiny comparison)
- `TrainConfig()` now equals the notebook cell: convnext_tiny + stride-4 decoder, `hflip=True`, `epochs=6`, `lr=5e-5`, `weight_decay=5e-3`, `val_preview_every=200`, `n_val_preview=10`, `num_workers=2`.
  - Why lr 5e-5: with 1e-4 at batch 4, validation loss fell mainly while the cosine schedule annealed.
  - Why weight decay 5e-3: AdamW decays by lr × wd per step, so 5e-4 was effectively off (about 0.01 % over a run).
- `pretrained_file="auto"` (new default) uses `checkpoints/<backbone>.fb_in22k.safetensors` when that file exists. Otherwise it falls back to the hub / local HF cache as before. The card records which file was used.

### Frame audit: `mppp.mask.audit` and a notebook-02 section
- `audit_frames(items, ckpt)` runs the model over every frame, exactly as in production, and ranks the frames by IoU with their mask (or by `error_pct`).
  - Defaults: `images/` only, so each label is scored once on its production image.
  - Each row gets its train/val label; the split is rebuilt from the card.
  - Output: `<checkpoint>_debug/frame_audit.csv`.
  - Unreadable files and image/mask size mismatches are ranked first.
- `render_worst` writes `worst_frames_NN.png`: the four-panel view (image | truth | P(terrain) | prediction − truth) of the worst frames. It also writes `worst_frames.txt` with their names, e.g. for selecting them in Metashape.
- `summarize` and `worst_table` print text summaries.
- Train frames score optimistically: a train frame near the bottom of the list is a strong label suspect.
- Tested on real v7 frames with the 24 Sep tiny model: the SIF WATSON wheel frame (blocky polygon label) ranks first.

### Default mask model: `checkpoints/convnext_tiny_s4_seg_best.pt`
- `default_config()["masking"]["checkpoint"]` changed from `convnext_tiny_seg_best.pt`, which is no longer in `checkpoints/`. With the old default, the COLMAP notebook failed on every image with "Mask checkpoint not found".
- `promote_checkpoint(ckpt)` copies a trained model (+ card, `.check.png`) to `<backbone>[_s4]_seg_best.pt`.
  - The previous best is moved to `checkpoints/archive/<stem>_<UTC time>…`, never overwritten or deleted.
  - Promoting the same file twice does nothing.
  - A best trained on the same dataset (same fingerprint) with a higher val IoU is kept unless `force=True`. Val IoUs of different datasets are not comparable, so in that case the new model wins.
- `process_images` checks the mask checkpoint once, before any image. A missing file now raises a single error that lists the checkpoints that do exist and shows the `promote_checkpoint` fix.

### Notebooks
- `01_process_images_v0p10` and `03_colmap_alignment_v0p10` (renamed from `04_`, as requested; the number 03 is free since v0p9 removed the old notebook 03): default checkpoint as above.
- `02_train_mask_v0p10`: the requested cfg cell, then at the end:
  - **Lowest-agreement frames:** table + panels. The CSV is reused on rerun unless `RERUN_AUDIT=True`.
  - **Promote**.
- Tests: `tests/test_v0p10.py`:
  - A deliberately inverted label ranks first; CSV round trip; size-mismatch reporting.
  - Promotion rules and archive naming; the preflight error; `pretrained_file="auto"`.
  - The defaults test is updated.

## v0p9 — 2026-09-23

### COLMAP bridge: `mppp.sfm` and notebook `04_colmap_alignment_v0p9`
From MPPP-processed images to a COLMAP 4.2 database and a refined, prior-aligned reconstruction for `mppp.error`. Details and formulas are in `docs/MPPP_methods.md` §10.
- **Selection:** `select_best_products` keeps one product per exposure (instrument + SCLK): the largest file, then the highest version. It keeps NCAM sequences only (SAPP sun and SCAM support frames are dropped).
- **Cameras:** one COLMAP camera per Navcam eye at full resolution, from `M2020_N{L,R}0_frame.xml` with p1 = p2 = b1 = b2 = 0 (FULL_OPENCV with k1–k3, k4–k6 = 0). This matches Metashape's projection to 1e-6 px (test).
  - Half- and quarter-resolution frames use the same camera: native keypoints are scaled by 1/s. This is exact for MPPP's detector-padded, binned frames (test).
- **Rig:** left is the reference. The right offset is the median CAHV left→right pose, one value for every pair and every resolution.
  - Belva: 100 pairs agree to 1.3e-4° and 10 µm; baseline 0.42436 m.
- **Database:** frames are exposures; position priors come from the waypoints (σ = 1 m, configurable); keypoints are in full-resolution pixels.
- **Matching:**
  - `exhaustive` is the baseline.
  - `prior_pairs` (`prior_overlap_pairs`) casts each image's valid pixels onto a local ground plane and projects the footprint into the other images.
  - Geometric verification uses a 6 px threshold in full-resolution pixels.
- **Reconstruction:** CAHV-posed triangulation, then a weighted bundle adjustment (pycolmap cost functions + pyceres).
  - Each observation has σ/s in full-resolution pixels. COLMAP's own BA would over-weight quarter-resolution frames 16×.
  - Frame poses are free; waypoint priors; per-camera f, c and k1–k3 free, with p1, p2 and k4–k6 held at zero.
  - The rig rotation is free and the CAHV baseline is held: it sets the scale.
  - Graduated schedule of three rounds, then a final adjustment.
  - `register_stations` (a generalized absolute pose per station) is available for priors that are far off; it is not needed at Belva.
- **Scale finding:** when the baseline was also free, the first test shrank it from 0.424 m to 0.14 m. The position priors (1 m over stations ~50 m apart) do not hold the scale, so `refine_rig="rotation"` is the default.
- **Export** (`export_for_error`): a native-pixel text model (one camera per eye and resolution), `stations.csv`, `poses.csv` (prior − refined), `residuals.npz` and `summary.json`, which includes the `mppp.error` epsilon estimates.
- **pycolmap pitfalls found on the way (handled):**
  - `Rig.sensor_from_rig()` returns a copy. Used directly as a Ceres parameter block, it gave every right-camera observation its own offset. The rig offset is now one explicit block, written back after solving.
  - `triangulate_points` also refines the rig offset by default; it moved the baseline 4 mm, and that is now switched off.
  - Removing pycolmap's point blocks from its adjuster problem is O(residuals) each in Ceres, over an hour at 10⁶ observations. The problem is now built on a copy of the reconstruction with dummy points (58 s on 133k observations).

### `mppp.error`
- **`read_colmap` residuals now include lens distortion.** `_project` used a pinhole for every model, which is off by hundreds of pixels for the Navcam FULL_OPENCV model (k1 = −0.27). The new `project_camera` implements SIMPLE_RADIAL, RADIAL, OPENCV and FULL_OPENCV exactly as COLMAP does (test against pycolmap) and refuses other models. This affects `observation_residuals`, `calibrate_eps_from_residuals`, `match_survival` and everything else that projects.

### Removed
- Notebook `03_build_mask_training_set` (as requested). `mppp.mask.dataset.build_variable_images` stays in the library.

### Packaging
- New optional dependencies `sfm = ["pycolmap>=4.2", "pyceres>=2.6"]`. Notebooks renamed `*_v0p9`.
- Tests: `tests/test_sfm.py`:
  - Metashape ↔ COLMAP projection, and exact keypoint scaling.
  - `mppp.error` projection vs pycolmap for four models.
  - Weighted BA on a synthetic two-station rig with half- and quarter-resolution frames: recovers f to <1 px, principal point to <1 px, k1 to 1e-3 and camera centres to <2 cm, with the baseline held.
  - Whitened cost ≈ number of observations at the truth (the weights follow native resolution).
  - Prior-overlap pairs keep stereo partners and reject views 150° apart.
  - Outlier filtering removes every observation above the limit.

### Belva test (Navcam, sols 748–815; run in the cloud sandbox, 2 CPUs)
- **Data:** 478 exposures (NCAM only, best product per exposure) and 13 site/drive stations. There are 100 stereo pairs; 278 frames are left-only, mostly quarter-resolution post-drive images.
- **Matching:**
  - 2.33 M keypoints (8192 max per image, masks from `convnext_tiny_seg_best.pt`).
  - Prior-guided pairs at min overlap 0.02: 35,794 of 114,003 pairs, 3.13 M inlier matches.
  - Timing: 16 min matching, 23 min features, 35 min reconstruction.
- **Reconstruction:** all 478 images registered; 297,378 points; 1.03 M observations; mean track length 3.47; 12.8% of points cross-station.
  - Residuals: median 0.221 and rms 0.415 native px; by resolution the medians are 0.209 (half), 0.245 (full) and 0.264 (quarter).
- **Intrinsics (initial → refined):**
  - NL: f 2950.9 → 2948.5 / 2948.0, c (2594.2, 1942.2) → (2591.4, 1945.5), k1 −0.2755 → −0.2692.
  - NR: f 2943.6 → 2942.3 / 2941.8, c (2578.4, 1948.1) → (2579.3, 1949.9).
  - The rig rotation went from 0.121° to 0.171° (baseline held).
- **Station connectivity (components tied by ≥ 50 shared points):**
  - Site 39 D0650–D1170 is one block of 7 stations. S038D1808 and S038D2208 form a second block, tied by 153 points only.
  - S037D4972, S038D0000, S038D0944 and S039D0000 have no cross-station points, so their prior offsets carry no information.
- **Waypoint priors vs refined, site-39 block:**
  - Station centres differ by 0.03–0.44 m.
  - After the best similarity: scale 1.0022, rotation 0.97°, station residuals 0.06–0.34 m.
  - The block's absolute orientation is set only by the 1 m position priors over ~100 m, so the 0.97° is not significant yet. **Provisional.**
- **`mppp.error`:** ε_intra 0.255 px, ε_cross 0.385 px (native px, post-fit residuals / √2), ratio 1.51. **Provisional:** the pairs were prior-guided, not exhaustive, and the value is higher than the 0.169 px measured earlier from Metashape tie points.
- **Found and fixed during the run:**
  - Frames with no observations rotated freely (9° and 36°). Frames with < 30 observations are now held (55 images, flagged in `poses.csv`).
  - Outlier filtering was a no-op: pycolmap's maps do not recognise numpy integer ids.
  - The final adjustment could push a few points behind a camera; a filter now runs after it.

## v0p8 — 2026-09-23

- **`build_variable_images` / notebook 03 run before `images/` existed.** v0p7 moved `images_variable/` aside to `images_variable_prev_<time>` first, then found no images and wrote nothing. That left an empty `images_variable/` (seen on v7 at 10:44). Nothing was lost: the old folder was intact under the `_prev_` name.
  - Inputs are now checked **before** anything is moved. If `images/` has no PNGs, or none of them has a mask, it raises and says to regenerate `images/` first (notebook 02, step 1).
  - It warns when fewer images than masks exist (regeneration unfinished).
  - Interrupted `*.tmp.png` writes are ignored here and in `scan_dataset`.
- Notebook `03_build_mask_training_set_v0p8` states the order (02 step 1, then 03). It prints the image and mask counts and stops if `images/` is empty. The spot check copes with fewer than 4 images. Notebook 02 points to 03 after step 1.
- Test: an empty `images/` (and one with only a half-written file, or no masks) raises and leaves `images_variable/` in place.
- Notebooks renamed `*_v0p8`.

## v0p7 — 2026-09-23

- **`num_workers > 0` on Windows.** Training failed at the first batch with `AttributeError: Can't get local object 'make_collate.<locals>.collate'`. Windows starts DataLoader workers by *spawn*, which pickles the collate function, and a function defined inside another function cannot be pickled. The collate is now a module-level class (`Collate`). Workers are kept alive between epochs (`persistent_workers`), because each new Windows worker re-imports torch.
- Tests reproduce the Windows behaviour on Linux: `train(num_workers=2)` and `regenerate_images_from_pds(workers=2)`, each under the spawn start method. The training test fails on v0p6 with the reported error.
- Notebook `02_train_mask_v0p7`: `num_workers=2`, and the browser download links for the timm ImageNet-22k weights of tiny and base.
- Notebooks renamed `*_v0p7`.

## v0p6 — 2026-09-23

### `images/` regenerated from the PDS products (`mppp.mask.dataset.regenerate_images_from_pds`; notebook 02, step 1)
- For every `masks/<stem>.png`, the matching `<stem>.IMG` is found anywhere under the archive (`D:/data/m2020`: `datadrive/<sol>/ids/rdr/<cam>/` and the flat `zcam/`). If the exact version is missing, the highest version of the same product is used and logged as `version_substituted`.
- The image is run through `MPPPImage` with no mask inference, linear 8-bit, and padding to the full detector frame. That gives the same geometry as the Metashape masks and the same radiance → 8-bit mapping `MPPPImage` feeds the network at inference.
  - Checked on two v7 frames (Navcam 2560×1920 padded half-height sub-frame; Mastcam-Z 1648×1200): size and valid area match the masks exactly.
- Written as RGBA with alpha = your mask (the v7 convention; `alpha="valid"` or `None` are options). An image whose size differs from its mask is not written.
- Resumable: a marker file in `images/`, and images already written are skipped. An `images/` folder not made by this function is renamed `images_prev_<time>`. `_mppp_regen_log.csv` has one row per mask. `workers>1` runs separate processes.
- **Finding:** the old v5/v6 `images/` are ≈1.28× brighter in R and B and ≈1.37× in G than the current pipeline output. The correlation is 0.9999, so it is a constant rescale: consistent with an older 8-bit scale (≈50 instead of 64) and green white balance (1.5/1.4 instead of 1.4/1.3). Inference has been using the current scale all along, so the old training images did not match what the model sees in production. The v7 `images_variable/` (= v6 `images/`) now acts as brightness augmentation.
- `scan_dataset(..., missing="skip")` skips images that have no mask, with a warning. v7 `images_variable/` holds a few frames you removed from the Metashape project.

### Model and training
- **Default backbone `convnext_tiny`** (as requested; the base run of 22 Sep showed no overfitting). The other defaults are unchanged.
- **Stride-4 decoder** (`stride4=True`, default for training). The first ConvNeXt stage (stride 4) is projected to 48 channels, concatenated with the upsampled ASPP output and fused by a 3×3 conv (DeepLabv3+ style). Before, the boundary was decided on the stride-8 grid.
  - Cards get `stride4`; cards without it load as before (strict state-dict load).
  - `init_from` loads the pre-v0p6 decoder (fpn, aspp, head) and the stride-4 layers as separate all-or-nothing groups, so a v0p5 checkpoint supplies everything except the new layers.
- **Quad split option, off by default** (`quad_split_above=None`; `quad_overlap` 64 px). Turned off at your request after the trade-off below; the code and tests stay, so it is one setting to try. When set, a frame with a longer side is trained and inferred as four quadrants. Each crop extends 64 px past the centre lines and only its core quadrant is kept; the cores tile the frame exactly.
  - Quadrants whose core is pure padding are skipped in training and not run at inference.
  - The split happens after the grouped train/val split, so all quadrants of a frame stay on one side.
  - Card fields `quad_split_above` and `quad_overlap`; old cards have none, so they never split. `infer_mask(..., quad_split_above=...)` overrides it.
  - 4000 would split only the 5120 px Navcam/Hazcam frames (237 in v7). Whole, they reach the network at 0.32×; as quadrants at 0.63×, the same scale 2560 px frames get whole. With 2500 the 2560 px frames are split too, but their quadrants are enlarged 1.22× (no new detail) and v7 grows from 3925 to ≈7.7–8.1k samples per image folder.
  - Validation IoU is computed on quadrants, so it is not comparable across split settings.
- `curves.png`: the loss and IoU panels start at zero.
- **ImageNet weights files:** `init_from=<model.safetensors>` failed with `UnpicklingError: invalid load key '\xac'`: it expected a pickled ConvNeXtSeg checkpoint.
  - Both `init_from=` and `pretrained_file=` now accept a timm/Hugging Face `model.safetensors` or a FAIR `.pth`. The file is read with safetensors when it is one; FAIR names are converted with timm's filter; the classifier head is dropped (in1k, in22k and fine-tuned files all work); every backbone tensor must match, or the error names the size mismatch.
  - `pretrained_file` no longer goes through timm's `pretrained_cfg_overlay`, which could fail on an in22k head (21841 classes) against the in1k default config.
  - Weights must match the backbone size: `convnext_base_model.safetensors` works with `backbone="convnext_base"`; the tiny default needs `convnext_tiny` weights (or the HF cache).

### Notebooks
- `01`, `02`, `03` renamed `*_v0p6`.
- `02` gains step 1 (regeneration and a visual check) and a whole-frame held-out check with one frame per camera family.
- `03` now defaults to `psx = None`, so it rebuilds only `images_variable/`. You export `masks/` yourself, and a re-export would archive that folder.

### Tests
- 14 new test cases: stride-4 shapes and old-card loading, ImageNet `.safetensors` via `init_from`/`pretrained_file`, init from a pre-v0p6 checkpoint, quadrant tiling (4 geometries), quadrant stitching and padding skip, `expand_quads`, a train-and-infer run with quadrants, `scan_dataset(missing="skip")`, the curves y-limits, PDS regeneration (exact pixels, alpha, archiving, resume, version substitution) and version lookup.

## v0p5 — 2026-09-22

### Mask training-set builder (`mppp.mask.dataset`, notebook `03_build_mask_training_set_v0p5`)
The code that produced `masks/` and `images_variable/` was not in MPPP, MarsMask, or the workspace and training notebooks. It was reconstructed from the data in `D:/masks/masks_training_set_v5`–`v7`:
- **`images/`:** MPPP "standard" 8-bit RGBA (fixed radiometric scale), plus `images_standard_16-bit/` and `masks_valid/` from the same run (identical timestamps).
- **`masks/`:** drawn and edited in Metashape (`masks_v*.psx`, whose cameras point at `images/`) and exported. Metashape keeps them as `c<camera_id>.png` in `<project>.files/0/0/masks/masks.zip`, with `frame.zip` mapping camera id to image path.
  - `export_masks_from_psx` reads them directly, with no Metashape needed.
  - Checked on the v7 project: 3925 cameras, 3910 in `images/`; masks bit-identical to your 22 Sep export.
  - 15 cameras point at files in the training-set root rather than `images/`; they are reported, not written.
- **`images_variable/`:** in v6, the RGB is bit-identical to v5 `images/` (an older processing), with v6 `masks/` in the alpha channel.
  - Fitted on 15 pairs from all six cameras, it is ≈ clip((x−lo)/(hi−lo))^g per image, nearly the same for all channels: lo 0–0.11, hi 0.86–1.76, g 0.47–0.85, fit rms 0.4–8 DN.
  - `build_variable_images` keeps an original variable RGB wherever one exists (bit-exact) and synthesizes the rest from that parameter range, seeded by file name (`images_variable_manifest.csv` records the parameters). The alpha is always the current mask.
  - Synthesized versions are milder than the originals (no warm-tint shift). They are augmentation, not a reproduction of the lost code.
  - Validity comes from non-zero RGB, **not alpha**: in v7 `images/` the alpha is the training mask, and using it would have blacked out the rover in every synthesized image (caught by the spot-check panel; test added).
- `build_training_set(root, psx, reuse_variable_from)` does both steps and writes `build_report.json`. Existing `masks/` and `images_variable/` folders are renamed `*_prev_<time>`, never deleted, and `images/` is never modified.

### Training
- **New defaults** (as requested): `convnext_base`, 5 epochs, batch 4, lr 1e-4, weight decay 5e-4, threshold 0.5, `input_size` 1648.
- **Rectangular canvas (1664 × 1248 by default).**
  - Images are resized so the long side is 1648 and zero-padded bottom/right into a 4:3 canvas whose sides are multiples of 32 (the ConvNeXt stride). Mastcam-Z 1648×1200 stays at native resolution with 16/48 px of padding (the 1648 square added 448 rows).
  - Navcam/Hazcam frames map to 1648×1236. That is 24 % fewer pixels per sample, and 1648 is not a multiple of 32.
  - The card records `canvas`; cards without it are treated as the old square, so existing checkpoints infer exactly as before. `canvas=None` restores the square for training.
- **Offline machines** (huggingface.co unreachable: the 5× retry loop seen on 22 Sep):
  - The hub is probed first. If it is unreachable, only the local HF cache is used; otherwise training stops at once with the options instead of retrying twice for ~45 s.
  - `init_from=` copies every name- and shape-matched tensor from an existing ConvNeXtSeg checkpoint (the decoder all or nothing). `D:/code/MarsMask/mask_train/convnext_base_seg_best.pt` (Oct 2025, `fpn_width` 192) supplies all 340 backbone tensors and no decoder tensors at `fpn_width` 256.
  - `pretrained_file=` loads a local `model.safetensors`.
- Notebooks renamed `*_v0p5`; `02_train_mask_v0p5` uses the new defaults and data from v7.

## v0p4 — 2026-09-22

### Training monitor fixes (from the first full GPU run of v0p3)
- **Logs from restarted runs were mixed.** Every run with the same checkpoint name (the date) appended to one `log.csv`. The 22 Sep folder held rows from four starts, so `curves.png` drew doubled lines. Now an existing debug folder is moved to `<name>_debug_prev_<time>` before a run starts. If Windows refuses the rename because a file is open, its files move to a `_prev_<time>` subfolder instead. Nothing is deleted.
- **Live window no longer depends on OpenCV's GUI.** `cv2.imshow` failed with the installed OpenCV ("function is not implemented"; the pip wheel is built without GUI support). `live=True` now starts `monitor.py watch <debug_dir>` as a separate process: a matplotlib window showing curves, the newest batch panel and the newest held-out panel, refreshed every 5 s. The training kernel still never imports matplotlib. The viewer can also be started by hand from any terminal, for a running or finished run.
- Panel PNGs are written atomically (`*.tmp.png` then rename), so the viewer and Photos never read half-written files.
- Notebooks renamed `*_v0p4`; `02_train_mask_v0p4` uses `live=True`.
- Tests: folder archiving, newest-image selection ignoring partial writes, and the viewer process starting and polling.

## v0p3 — 2026-09-22

### Training progress plots (`mppp.mask.monitor`)
`train()` now writes `<checkpoint stem>_debug/` by default (`debug_dir=None` disables it).
- **Kernel-crash fix, same release.** The first form of this monitor drew with matplotlib inside the training process. On the Windows/conda GPU run, the kernel died at iteration 50: `log.csv` had one row and there was no PNG, so it died at the first in-process figure after torch/CUDA had loaded. The crash was native, with no traceback, which is consistent with a duplicate Intel OpenMP runtime ("OMP: Error #15").
  - Panels are now drawn with OpenCV only (already loaded by the data loader, same interpolation modes).
  - `curves.png` is rendered by a separate Python process running `monitor.py` (written atomically).
  - The trainer never imports matplotlib; a test runs training in a fresh interpreter and asserts this.
  - Both notebooks set `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch, as the 2025 training notebook did, for their own matplotlib cells.
  - The post-training check cell uses the OpenCV panel instead of pyplot.
- `curves.png`: four panels, each with one y-axis — train loss (mean over each print interval), IoU (train batches vs a held-out preview), learning rate, and the pre-BN ASPP peak against the fp16 limit.
- `val/ep*_it*.png`: the **same** fixed held-out frames every `val_preview_every` iterations (default 250, `n_val_preview` = 6), in eval mode, exactly as inference sees them. Columns are image | truth | P(terrain) | prediction − truth, with IoU / missed % / false % per frame at full resolution.
- `batch/ep*_it*.png`: the first frame of the current training batch.
- `log.csv` and `val_preview.csv`: the numbers.
- `live=True` also shows the latest panel in an OpenCV window; display errors disable it rather than stopping training.
- `python src/mppp/mask/monitor.py curves <debug_dir>` redraws the curves for any run, including an interrupted one.
- Console lines now show the mean over the last interval as well as the running epoch mean, which lags.
- Notebook `02_train_mask_v0p3` gains a cell that shows the latest curves and held-out panel.
- The v0p2 notebook dropped the Qt figure that the 2025 notebook drew; this restores progress display in a more robust form.

### Scapes in the process notebook (`mppp.scapes`)
- The five scapes from `workspace.ipynb` are transcribed as data: Rockytop 460-535, Landing 9-48, Belva 770-835, Bunsen 1055-1095, Hellandfjellet 1601-1645.
- Each selection line is tagged with its workspace group. Where a workspace cell assigned `camera_codes` twice, only the last assignment took effect, and that is what is transcribed.
- `select_scape(name, input_dir)` takes groups 1 and 2 only. Group 3 (the Rockytop Z110/Z063 mosaics) is kept for provenance but not selected.
- 110 mm Mastcam-Z frames are dropped wherever a selection takes "all zooms" (Belva, Bunsen, Hellandfjellet).
- The result is de-duplicated to the highest product version; nothing is deleted. The legacy Belva/Bunsen cells ran `dedupe_by_version_index(delete=True)` on the archive.
- The report — definition, counts per selection, camera groups, dropped Z110, superseded versions — goes into the manifest and config snapshot through the new `process_images(..., provenance=...)` argument.
- `01_process_images_v0p3`: a named scape is option A; sol range and waypoint radius remain as B and C.
- `tests/test_scapes.py` checks group/zoom/sol/version filtering on a synthetic archive and the sol ranges against the workspace.

## v0p2 — 2026-09-22

### Mask training: NaN losses diagnosed and fixed (`mppp.mask.train`, notebook `02_train_mask_v0p2`)
The September 2026 run (`Tate_training_convnext_20251002_edited.ipynb`) trained normally for 1.3 epochs (val IoU 0.975 after epoch 1). Non-finite losses started at epoch 2, iteration ~570, grew in frequency, and then covered every batch (epochs 3-4: 1735/1748 and 1741/1748 skipped; val loss NaN from the end of epoch 2).
- **Cause: fp16 overflow in the decoder.** `fpn.smooth → aspp.b* → aspp.proj` has no normalisation until `aspp.bn`, so the loss is invariant to the scale of those weights and their norms drift upward. From the two checkpoints:
  - fpn.smooth: 9.2 at init, 13.4 after epoch 1, 15.7 in 2025.
  - aspp.proj: 9.2 at init, 12.4 after epoch 1, 13.2 in 2025.
  - Layers followed directly by BatchNorm barely move (9.2 → 9.8).
  - Pre-BN `aspp.proj` peak on an ordinary Mastcam-Z frame: 1.2e4 (epoch 1) and 2.3e4 (2025). fp16 overflows at 6.55e4. The backbone peaks at 1.1e4 and is not the problem.
- **Why it never recovered:** a skipped batch makes no update, so once every batch overflowed the weights were frozen. The overflowing forward passes had already put inf/NaN into the BatchNorm running statistics, which is why validation was NaN.
- **Not the Tiny backbone.** The decoder is identical for convnext_base, so Base would fail the same way.
- **Fixes:**
  - `precision="auto"` uses bf16 where supported, otherwise an fp16 backbone with an fp32 decoder and loss; plain fp16 is no longer offered.
  - Gradients are unscaled before clipping (they were clipped at norm 1 in *scaled* units).
  - A non-finite loss raises `NonFiniteLoss`, naming the batch files, instead of being skipped.
  - The train/val split is grouped by mask, so `images/X` and `images_variable/X` no longer straddle it; earlier val IoUs were optimistic by an unknown amount.
  - The IoU metric uses the inference threshold (0.4, was 0.5).
  - Checkpoints are never overwritten and always get a model card with history, config, split and dataset fingerprint.
  - The pre-BN ASPP peak is logged every `print_every` iterations.
- `ConvNeXtSeg.forward` is split into `features()` + `decode()`; the weights are unchanged, so existing checkpoints load unmodified.
- `tests/test_mask_train.py` reproduces the failure with the real checkpoint: scaling `aspp.proj` ×4 leaves fp32 train-mode output unchanged, full fp16 overflows, and the new path gives an identical result before and after scaling. It also covers a CPU smoke run, stop-on-NaN, the grouped split and letterbox geometry.
- The epoch-1 checkpoint from the failed run (`unorganized/convnext_tiny_seg_best.pt`, val IoU 0.9751 on the leaky split) is usable but is not the default; the default stays the 2025 checkpoint (0.979).

### Error model merged (`mppp.error` ← `mppp_error` v0p15)
- Code unchanged except package-relative imports, subprocess self-tests finding the merged package, and one crash fix: a duplicate `waypoints._last_per_sol` shadowed the first, so `build_stations` / `python -m mppp.error run` raised `TypeError`.
- `python -m mppp.error selftest` gives 186/186; also run by `pytest` (marked `slow`).
- Open findings that change published numbers (not changed, decisions needed) are in `docs/mppp_error_review_v0p2.md`: the E[Λ] cell model, unweighted correlation inflation, the residual degrees-of-freedom bias in the ε estimators, the pixel scale of ε = 0.169 px, and the sol-1842 "last per sol" rule relevant to the frozen site-87 prediction.
- Lab notebook moved to `docs/mppp_error_README_v0p15.md`.

### Other
- Real waypoints (`params/M20_waypoints.json`, 698 features, SHA-256 a1a2d082…) replace the synthetic fixture in tests.
  - Confirms the waypoint properties `lon`/`lat`, and that site 3 drive 0 is the first feature (flagged assumptions 5-6 in the methods doc).
  - New independent check: the exact (33, 2864) waypoint and (33, 0) + label offset agree to 5.6 m E, 0.4 m N, 0.7 m U.
- Notebooks renamed to the new version (`01_process_images_v0p2`, `02_train_mask_v0p2`).

## v0p1 — 2026-09-21

First packaged version, assembled from the working pre-package code (`src/image.py`, `readers.py`, `writers.py`, `workspace.ipynb`, `config.json`). Verified against that code on the two example products: valid masks identical, camera positions identical to 1e-6 m, integer images within one count, yaw/pitch/roll within 0.02° (`tests/test_legacy_regression.py`).

### Behaviour kept
Processing order and arithmetic; τ(L_s) table and zenith scaling; white-balance gains and integer scales; Malvar-2004 demosaicing of raw-Bayer Mastcam-Z; padding of sub-frames to the full detector frame; Navcam intrinsics from `M2020_N?1_frame.xml`; network architecture, 1648 px input, threshold 0.4 and 3×3 dilation of the mask model; the Metashape yaw/pitch/roll formula (verbatim); reference offset `floor(mean/10)·10`.

### Deliberate changes
- **Default output is 16-bit linear RGBA PNG** (was 8-bit). 8-bit PNG and 16-bit TIFF remain available through `export.formats`.
- **Rounding**: integer products are rounded to nearest (was truncation): ±1 count.
- **Metashape P1/P2 → OpenCV p2/p1.** The legacy code copied P1, P2 straight into the OpenCV slots; the two conventions are swapped. Effect with the current Navcam XML: about 0.01 px at the frame corner.
- **Pixel-origin convention made explicit.** Internally the first pixel centre is (0, 0) (CAHVOR/OpenCV); XML principal points are converted (−0.5 px) and the COLMAP/Metashape exporters add 0.5 px. The legacy code mixed the two.
- **k3 is exported**: COLMAP model `FULL_OPENCV` is selected when k3 ≠ 0 (the legacy `OPENCV` choice silently dropped the Navcam k3 = −0.022).
- CAHV axis **A is normalised** and the rotation orthonormalised (SVD): K changes at the 1e-5 level.
- `radiometry.zenith_min` (present but unused in the legacy config) is now applied as a floor on μ.
- The τ table is interpolated periodically in L_s (was linear extrapolation outside 0–360).
- Sub-frame padding uses `IMAGE.FIRST_LINE(_SAMPLE)` scaled by the downsample factor. This reproduces, and explains, the legacy special case "`pad_left >= full_width` → halve it".
- `np.prod` of uint8 masks (worked only through integer overflow) replaced by `np.all`.
- **Archive safety**: `dedupe_by_version` never deletes files (the legacy notebook called `dedupe_by_version_index(delete=True)` on the PDS archive). It now compares the full two-digit version.
- `waypoints_within_radius`: an anchor sol without a waypoint uses the last earlier waypoint (was: the latest waypoint of the mission).
- `find_waypoint_for_site_drive` always returns a feature (was feature or properties depending on the branch).
- Waypoints are cached on disk with a SHA-256 in the run records; processing works offline, and without waypoints (site-frame priors).
- Labels are read through `pdr` metadata with units stripped; the legacy indexing (`label[...][0]`) fails with current `pdr`.
- Mask model is loaded lazily and once per checkpoint/device (was loaded at import from a hard-coded `D:/` path); a model card JSON accompanies each checkpoint; `convnext_tiny` and `convnext_base` backbones are supported.
- Resource folders are located from the package, not from the current working directory.
- The undocumented Mastcam-Z fix-up `mask[0:5, 0] = 0; mask[-1:, 0] = 0` was dropped: those pixels lie in the dark columns, which are already invalid (verified on the example frame).

### New
Per-run manifest with per-image metadata (LMST, LTST, L_s, solar geometry, τ, focus/zoom motor counts, stereo partner, camera group, intrinsics, pose, Mars lat/lon/elevation); config + code-version snapshot with every output set; iTXt + XMP + EXIF-GPS metadata in PNG; COLMAP text model with priors and rig configuration; combined notebook `01_process_images_v0p1.ipynb`; test suite.

### Not yet included
`MPPP_import.py`, `mppp_error` v0p15, and the mask training script were not available when v0p1 was assembled; `mppp.error` and `mppp.mask.train` are documented slots.
