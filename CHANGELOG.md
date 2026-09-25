# Changelog

The development history v0p1–v0p12 (21–24 September 2026) is in [docs/history/CHANGELOG_v0p1-v0p12.md](docs/history/CHANGELOG_v0p1-v0p12.md).

## 0.14.3 — 2026-09-25

COLMAP alignment settings after the Three Forks test.

- **Tangential distortion can be refined.** `reconstruct(..., refine_tangential=True)` (and `bundle_adjust`) frees p1, p2 of the FULL_OPENCV cameras; otherwise they stay at their initial values.
  - To start them from the Metashape calibration rather than zero, create the project with `zero_terms=("b1", "b2")`. For NL0 that is P1 = 1.72e-4, P2 = 1.70e-4, about 2 px at a full-resolution corner.
  - The library default is unchanged (`False`, p1 = p2 = 0), so earlier results reproduce.
  - The camera model stays FULL_OPENCV. `fixed_camera_params(model, refine_principal_point, refine_tangential)` returns the held indices.
- **Fourth round with tighter cut-offs.** The default schedule is `(24, 10, 8), (12, 2, 4), (8, 2, 2), (6, 1.5, 1.5)`: triangulation threshold [full-res px], Cauchy scale [σ], maximum residual kept [native px].
  - The last round keeps residuals up to 1.5 native px (3σ at σ = 0.5 px).
  - The final adjustment uses the last round's values.
  - `DEFAULT_SCHEDULE` in `mppp.sfm.reconstruction`.
- **`min_tri_angle_deg` default 0.5** (was 1.5). This keeps stereo-only points out to about 49 m instead of 16 m.
- **`sigma_px`** stays 0.5 (native px). The notebook now passes it explicitly.
- **SIFT `max_num_features` default 16380** (was 8192).
  - `features.db` now has a `features.json` record beside it. An existing `features.db` is reused only if the record matches the requested settings and covers every project image; otherwise it is extracted again.
  - A `features.db` from 0.14.2 has no record, so it is extracted once more.
  - `features_up_to_date(project, ...)` checks this.
- **Health report:** the camera-model section also lists p1 and p2 (initial → refined).
- **Notebook 03:**
  - `TANGENTIAL = "refine"` (or `"xml"`: from the calibration, held; `"zero"`: the 0.14.2 behaviour). The project is rebuilt when this changes the start values.
  - `MAX_NUM_FEATURES = 16380`, `SCHEDULE` with four rounds, `min_tri_angle_deg=0.5`.
  - Prints the initial → refined camera parameters.
- Tests in `tests/test_v0p14.py`:
  - p1/p2 are recovered on the synthetic rig only when asked;
  - the defaults;
  - the XML tangential terms with the OpenCV/Metashape swap;
  - feature re-extraction on changed settings.

## 0.14.2 — 2026-09-25

- **Health thresholds doubled** (warn / fail), as requested after the Three Forks test:
  - `rig_rotation_change_deg`: 0.06 / 0.2
  - `outlier_image_fraction`: 0.04 / 0.20
  - `ray_displacement_max_px`: 20 / 60
- **Weak-image diagnosis** (`diagnose_weak_images`, part of `assess_alignment` and printed by `health_table`).
  - For every image with fewer than 30 observations, it reports the keypoints, the verified inlier matches with its stereo partner, with other images and with other stations, and the most likely cause, with advice:
    - `few_keypoints` (masked or featureless);
    - `unmatched`;
    - `stereo_only_far` (lower `min_tri_angle_deg`);
    - `same_station_only` (a left-only mast pan: no baseline to triangulate);
    - `lost_in_triangulation`.
  - On the Belva v0p9 result the 55 weak images split into 20 `few_keypoints` (most with 0 keypoints), 17 `same_station_only`, 16 `unmatched` (prior-pair matching) and 2 `lost_in_triangulation`.

## 0.14.1 — 2026-09-25

- **Local fallback for the mask model.** If `mppp_mask_v1` is not in the cache and cannot be downloaded (offline, or not yet published: Hugging Face 401, GitHub 404), `fetch_model` now installs a local copy into the cache.
  - It looks in `checkpoints_dir()` (`MPPP_CHECKPOINTS`, else `checkpoints/` in the source folder) for the released `.safetensors`, then the checkpoint it was exported from (`convnext_tiny_s4_seg_best.pt`), then the promoted best of the same architecture.
  - The best checkpoint prepared with 0.13 converts to exactly the registered file (same SHA-256), so there is no warning. A different checkpoint is installed with a warning.
  - `local_candidates(name)` lists what would be used, and `fetch_model(local_fallback=False)` turns the fallback off.

## 0.14.0 (v0p14) — 2026-09-25

### Keep only the images you left in the output folder
- `process_images(..., only_existing="PNG8")` processes only the selected products whose image is still in `images_png8/`; `"PNG16"`, `"TIFF16"` or any sub-folder name also work.
- The manifest, `references.txt`, the COLMAP priors and the config snapshot are built from those images alone.
- The manifest records what was kept and removed, and which images in the folder were not in the selection.
- Workflow: process a sol range, delete the unsuitable images from `images_png8/` (or `images_png16/`), then rerun with the same selection and the option set.
- Nothing is deleted. Outputs of removed images in other folders (masks, other formats) stay on disk but are no longer in the manifest.
- `filter_to_existing(paths, out_dir, fmt)` is available on its own.
- **Notebook 01:** `ONLY_REMAINING = "PNG16"` (or None).
- **Notebook 03:** named test sites (`SITES = {"belva": (748, 815), "rockytop": (461, 530), "threeforks": (684, 692)}`, `SITE = ...`), with the work folder `D:/scapes/<site>_colmap` or `<site>_colmap_nav_zcam34`.
- **Notebook 03:** `KEEP_ONLY_REMAINING = True` reprocesses the images left in `processed/images_png8/`, then rebuilds the COLMAP project from them (`project.json`). The features already extracted are reused, and the database is rebuilt as usual.

### Mask model export no longer records the MPPP version
- The embedded card no longer contains `exported_with_mppp`, so the safetensors file (and its SHA-256) depends only on the checkpoint.
- The registry value is now `59b8f29b…21f5f` (124,825,116 bytes); it was `678aa6ec…67aa4` in 0.13.0, before the model was published.
- If you installed the model with 0.13.0, install it again (`python -m mppp.mask.hub install …`) so that the cached file matches the registry.

## 0.13.0 (v0p13) — 2026-09-24

First version prepared for the public repository `github.com/ctate7163/MPPP`. Scope for 1.0: process Mars 2020 **Navcam and Mastcam-Z** products and prepare them for **Metashape and COLMAP**. The mask model ships ready to use. Mask training and the error model are included, but they are not part of the main workflow.

### Repository and packaging
- **git:** the notebooks no longer carry a version suffix (`notebooks/01_process_images`, `03_colmap_alignment`, `training/02_train_mask`). The v0pN notebook copies stay in your old folder.
- **One version source:** `mppp.__version__ = "0.13.0"`. `pyproject.toml` reads it (`dynamic = ["version"]`), and `VERSION_TAG` (`v0p13`) is derived from it. Before this, `pyproject.toml` said 0.9 while the code said v0p12.
- **Legacy code** moved to `src/legacy/`, used only by the regression test:
  - the pre-package `image.py`, `readers.py`, `writers.py` and `config.json`;
  - `mppp.error.compat` (the v1 error-map shim), now `src/legacy/error_compat.py`.
- **Research scripts** `mppp.error.study_navcam` and `study_cross_station` moved to `studies/error/` (not installed). The error-model self-test runs them from there when present.
- **Package data** (`src/mppp/data/`, installed with the package) replaces `params/`:
  - Metashape calibrations (`m20_cmods/`);
  - the optical-depth table;
  - the waypoint snapshot (previously stored twice, in `params/` and `mppp/error/data/`);
  - the occlusion profiles;
  - the model registry `models.json`.
- **User cache** (`mppp.paths.cache_dir()`: `MPPP_CACHE`, else `%LOCALAPPDATA%\mppp` / `~/.cache/mppp` / `~/Library/Caches/mppp`) holds downloaded models and refreshed waypoints.
- **Checkpoints you train** go to `mppp.paths.checkpoints_dir()`: `MPPP_CHECKPOINTS`, else `checkpoints/` in a clone.
- **Waypoints:** `load_waypoints()` uses the cached copy if there is one, else the packaged snapshot. It no longer downloads implicitly, which keeps runs reproducible offline. `refresh=True` downloads into the cache.
  - A missing site now suggests a refresh when the snapshot was used.
  - The manifest records which copy was used.
- **Tests:** the example PDS products moved to `tests/data/m20/`. The tests find a mask model via `MPPP_TEST_CHECKPOINT`, the cached released model, or `checkpoints_dir()`.
- `LICENSE` (Apache-2.0, as already declared), `CITATION.cff`, `.gitignore`, `.gitattributes`, and conda environment files `envs/mppp.yml` and `envs/mppp_gpu.yml`. The `sfm` extra pins a matching pycolmap 4.2 / pyceres 2.6 pair.

### Released mask model: safetensors, registry, download
- **`mppp.mask.hub`:**
  - `export_safetensors` writes one `.safetensors` file with the model card embedded in its metadata. It is deterministic: the header keys are sorted, because the writer's order is not stable.
  - `fetch_model` downloads a registry model into the cache and verifies its SHA-256; a mismatching download is kept aside, never used.
  - `install_model` converts or copies a local model into the cache.
  - `resolve_checkpoint`: a path, a registry name, or a file in `checkpoints_dir()`.
  - CLI: `python -m mppp.mask.hub export|install|fetch|list`.
- `mppp_mask_v1` = the 24 Sep 2026 ConvNeXt-tiny stride-4 model (val IoU 0.971), SHA-256 `678aa6ec…67aa4` (superseded in 0.14.0). URLs: Hugging Face `ctate7163/mppp-mask`, and the GitHub release `mask-v1`.
- **Default `config["masking"]["checkpoint"]` is now `"mppp_mask_v1"`**, downloaded on first use. `process_images` resolves it once, before the first image.
- **Loading is pickle-free:** `.safetensors` directly, and `.pt` with `torch.load(weights_only=True)`. A `.pt` holding Python objects is refused with instructions. User-supplied ImageNet `.pth` files still fall back to a full load, with a warning.
- `docs/RELEASING.md`: pushing to GitHub, exporting to safetensors, the GitHub release, Hugging Face (`docs/hf_model_card.md`).

### Merged duplicates
- **COLMAP camera math and text I/O** now live only in `mppp.colmap`:
  - `project_camera` (vectorised) was in `mppp.error.colmap`, which now re-exports it;
  - `scale_camera_params` was a private copy in `mppp.sfm.export`;
  - `write_cameras_txt` / `write_images_txt` are shared by the prior-model writer and the native-pixel export.
- **Metashape XML:** `mppp.sfm.project.read_metashape_calibration` now calls `mppp.camera.read_metashape_xml` instead of its own parser.
- **Waypoints:** `mppp.error.waypoints.load_featurecollection()` with no argument uses `mppp.load_waypoints`. There is one snapshot file.

### COLMAP pipeline
- **`build_database(overwrite=True)` is the default:** rerunning the notebook replaces `database.db` instead of stopping with FileExistsError. A file still open elsewhere on Windows gets a clear message.
- **Two-view tie points:** `reconstruct(min_track_length=2)` keeps points seen in only two images (e.g. one stereo pair); 3 drops them.
  - COLMAP already triangulated them (`ignore_two_view_tracks=False`). The new parameter makes this explicit, and the track statistics now report their number and share.
  - `min_tri_angle_deg` (default 1.5) is exposed: at the 0.424 m baseline it limits stereo-only points to about 16 m.
- **Alignment health (`mppp.sfm.health`):** `assess_alignment(project, rec)` reports each check with provisional warn/fail thresholds and an overall verdict. The checks cover:
  - registration;
  - the tie-point population: track-length histogram, observations per image, cross-station ties, tied station blocks;
  - reprojection: median and p95; left vs right eye; outlier images; edge vs centre residual;
  - camera-model change: focal length, principal point, and the image displacement of the same ray;
  - stereo-rig stability: rotation vs CAHV, and the CAHV pair spread;
  - pose change vs the priors: per-station shift, within-station spread, attitude change, scale vs priors.
  - On the Belva result (v0p9 run) the verdict is FAIL: 55 of 478 frames are held at their prior; the refined right camera is rotated 0.13° from the CAHV rig while the CAHV pairs agree to 1e-4°; residuals rise from 0.18 to 0.31 px towards the image edge; 6 of 13 stations are outside the main tied block. The thresholds are provisional.
  - `write_health` writes `health.json`, `health.md` and `health.png`. The COLMAP notebook runs it before the error analysis and stops the analysis on FAIL. Methods: `docs/methods.md` §11.

### Mastcam-Z 34 mm in the COLMAP pipeline (option)
- `SfmProject.create` accepts Mastcam-Z frames. There is one COLMAP camera per eye and zoom (`ZL034`, `ZR034`, see `camera_key`), at the 1648 × 1200 full frame.
- **Initial camera:** `zcam_intrinsics="label"` (default) uses the median of the per-image label CAHVOR models, with p1 = p2 = 0. Focus changes f by a few pixels between images; the spread is recorded as `label_f_spread_px`. `"xml"` uses `m20_cmods/ZL034_frame.xml`, whose values look rounded (f 4720, k1 −0.45, k2 0.28) and are therefore only a rough start.
- **Rig:** Mastcam-Z left and right exposures carry different spacecraft clocks, so they are separate frames with no rig constraint. The rig code now names rigs by camera pair (`N`, `Z034`) and would build a Mastcam-Z rig if pairs ever share a clock.
- `select_best_products(sequence_prefix=("NCAM", "ZCAM"))` accepts several sequence prefixes.
- **Notebook 03:** `INCLUDE_ZCAM34 = True` adds the `_034` ZL0/ZR0 frames of the same sols, in a separate work folder.
- The health report's eye-balance check is now per camera pair (NL/NR, ZL034/ZR034).

### Tests
- `tests/test_v0p13.py`:
  - version source and package data;
  - cache and waypoint refresh;
  - legacy and studies outside the package;
  - deterministic, equivalent safetensors export;
  - pickle refusal;
  - registry download, SHA check, install, resolve and export → registry;
  - the released registry entry;
  - one projection for all modules;
  - the database overwrite default;
  - two-view tracks;
  - alignment health on a synthetic block, including a broken rig.
