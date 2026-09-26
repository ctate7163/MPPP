# Changelog

The development history v0p1–v0p12 (21–24 September 2026) is in [docs/history/CHANGELOG_v0p1-v0p12.md](docs/history/CHANGELOG_v0p1-v0p12.md).

## 0.22.0 — 2026-09-26

Best starting cameras (the refined results of earlier runs now start new ones):
- **Navcam rational cameras** (`M2020_N{L,R}_rational.json`) are the observation-weighted consensus of the Three Forks, Bell Island and Rockytop rational solutions. They differ from the v0p20 values by 0.15–0.17 px rms.
- **Navcam rig rotation** starts from the mean refined rig (`M2020_N_rig.json`, `navcam_rig="consensus"`, the default). The baseline vector stays the CAHV value; `"cahv"` restores the old start.
- **Mastcam-Z 34 mm focus bins** start from the focal-length-against-focus model of notebook 05 (`M2020_ZCAM034_focus_model.json`, `zcam_intrinsics="focus_model"`, the default), not from the label, which is about 1 % short. Bins of at most `zcam_hold_f_images` = 2 images hold f at the model. Bins outside the fitted focus range (−2000 to 1300 counts) use their label f scaled by the model's refined/label ratio.

Alignment:
- **Outlier frames can be excluded** (`reconstruct(exclude_outliers=True)`, notebook 03 `EXCLUDE_OUTLIERS = True`). `find_outlier_frames` flags a frame (a Navcam pair or one Mastcam-Z image) when:
  - its median residual is more than 3× the median of all frames (and more than 0.6 px);
  - it has fewer than 30 observations;
  - its shift or attitude change from the priors differs from the rest of its station by more than 5 scaled MADs (and 0.25 m or 0.5°). The attitude change is taken in the world frame, so a rover tilt common to the station is not an outlier.
  Flagged frames are deregistered after round 2 and again before the final adjustment. They are listed in the log and in `project.settings["reconstruction"]["excluded"]`, and the health report counts them separately (`excluded_fraction`, warn 5 %, fail 20 %) instead of as unregistered images. On the Three Forks, Bell Island and Rockytop rational solutions the defaults flag 0, 1 and 1 frames, all weakly tied (26 and 0 observations).
- **Tie points by convergence angle** (`convergence_statistics`): points per bin of the largest ray angle (0–2–5–10–20–40–180°), cross-station points, and points and observations above 10° and 20°. Notebook 03 prints them and stores them in the project settings.
- **More points at high convergence.** Notebook 03 now matches with a SIFT ratio test of 0.9 (COLMAP's default is 0.8). In the whole pipeline on the Three Forks test set this gave 8 % more tie points above 10° and 16 % more above 20°, for a 4 % higher median residual. Triangulation options (`TRIANGULATION`: transitivity, angle tolerances), DSP-SIFT and affine-adapted SIFT (`SIFT`, new `estimate_affine_shape` in `extract_features`) were tested too and added no high-convergence points (methods §13).
- **Rig translation prior** (`bundle_adjust(refine_rig=True, rig_translation_sigma_m=...)`) for experiments. The stereo baseline stays fixed by default: freeing it changes its length by ≤ 0.07 mm, because the images carry no scale (methods §13).
- `triangulate` takes COLMAP's track options (`max_transitivity`, `create/continue_max_angle_error_deg`, `complete_max_transitivity`); `reconstruct(triangulation_options=...)`.
- `match` passes `max_ratio`, `max_distance` and `cross_check` to the SIFT matcher and records them.

Error analysis:
- **Tracks of three or more images** (`load_alignment(min_track_length=3)`, notebook 04 `MIN_TRACK_LENGTH = 3`, the batch default). The ε table gives the number of points and a DOF-corrected ε = ε √(2N/(2N − 3P)).
- Residuals recomputed from the model (no `residuals.npz`) now carry point ids, so the DOF correction and cross-station split work there too.

Batch runs:
- **`scripts/run_scapes.py`** runs notebook 03 for several sites, then notebooks 04 and 05 on the ones that finished. Settings are injected after each notebook's `parameters` cell, each executed notebook is saved beside its results, and progress goes to `<root>/batch_log.txt`, one line per cell.
- Notebook 03 has a `threeforks_large` site (sols 670–694).

Fixes:
- `exclude_frames` used a `Frame` attribute pycolmap does not have.
- Tests: the Mastcam-Z binning test names `zcam_intrinsics="label"`; the rational-camera k4 test accepts 0.01 of the 0.02 start offset left, since k4 trades off against k1–k3 on the small synthetic block.

## 0.21.1 — 2026-09-26

Notebook 05:
- **Full-frame example images.** The examples now prefer images that cover the whole detector at full resolution (`pick_example(full_frame=True)`, using the manifest's padding); many Navcam products are sub-frames or downsampled. An image named in `EXAMPLE_FILES` that belongs to the scape uses its own refined camera, which for Mastcam-Z is its focus bin.
- **Undistortion stops where a lens model folds back.** `radial_limit` finds where the radial mapping stops increasing (57° off-axis for the three-term Navcam polynomial; the rational model has no such limit). Beyond that angle `undistort` leaves the output black and the grid is not drawn, instead of repeating interior pixels near the corners.
- **Mastcam-Z focus fit** leaves out bins below `FOCUS_MIN` = −2000 motor counts, for the label fit too (`fit_focus_model(min_focus=-2000)`).
- **Mastcam-Z focus plot.** Left and right eyes share both axes, and a right-hand axis gives the focal length in mm for 7.4 µm pixels (`ZCAM_PIXEL_MM`). The fitted slope is also given in mm per 1000 counts.

## 0.21.0 — 2026-09-26

- **Notebook 05, camera models across scapes** (`notebooks/05_camera_models.ipynb`, `mppp.sfm.calibration`). Takes the solutions of notebook 03 and puts them side by side:
  - Navcam intrinsics per scape, as pixel differences over the whole frame from the shipped model and from the consensus of each lens model, after removing the rotation a pose absorbs. Includes difference maps and radial profiles.
  - The label (flight) calibration against the refined cameras.
  - The Navcam rig. Also the effect of any two geometries on disparity and range: label vs refined, start vs refined, consensus vs each scape.
  - Mastcam-Z focal length against focus across scapes, with scape-to-scape offsets and the label line.
  - Mastcam-Z stereo pairs against the label geometry and against one fixed rig.
  - A table of what limits the geometry, next to the image precision ε.
  - Updated CAHVORE (Navcam, type 2 and 3) and CAHVOR (Mastcam-Z) models, written as JSON and as label text.
  - One example image per camera, original and undistorted, with the hardware mask under a white screen (`SCREEN_ALPHA`, default 0.3).
- **`mppp.cmod`: JPL camera models.**
  - `CameraModel` (CAHV, CAHVOR, CAHVORE) projects exactly as the JPL library does, including CAHVORE types 1–3.
  - `fit_to_colmap` fits CAHVOR or CAHVORE to a COLMAP camera.
  - `PixelCamera` and `compare_cameras` compare any two cameras in pixels, with the rotation removed and fold-over excluded. The rotation is fitted inside 0.85 of the half-diagonal.
  - The Navcam label is CAHVORE type 2 (fisheye). It agrees with MPPP's rational model to 2.7 px rms; read as perspective CAHVOR it would be about 250 px off. CAHVORE fits the rational model to 0.4 px (type 2) or 0.2 px (type 3), whereas CAHVOR only reaches 9 px rms.
- **Manifest: the full label camera model** (`camera_model_label`) is now kept for every image: C, A, H, V, O, R, E, type and parameter, with the product size. Older manifests keep only the CAHV part with R1 and R2; `attach_pds_labels` reads the exact models from the PDS archive for them.
- **First results** (seven v0p15 scapes with the polynomial Navcam model, plus three v0p20 rational solutions; see docs/methods.md §12):
  - The rational Navcam cameras repeat from scape to scape to 0.1–0.55 px rms; the polynomial ones to 0.6–4.7 px, with up to 21 px in the corners.
  - The label Navcam stereo geometry predicts about 1.3–1.9 px less disparity than the refined one, so ranges come out about 1 % short at 10 m and 2 % at 20 m.
  - Mastcam-Z 34 mm focal length lies about 1 % above the label value at the same focus (43–49 px) and repeats from scape to scape to 4–5 px.
  - One fixed Mastcam-Z rig would leave ±1 px of pair-to-pair disparity scatter at 10 m.

## 0.20.1 — 2026-09-26

- **Fix: residuals were attributed to the wrong tie points in notebook 04** (`mppp.error.alignment`, exports from v0p14.5 on).
  - `residuals.npz` stores the keypoint index of the reconstruction. The native model written by `export_for_error` keeps only the observed keypoints of each image, so these indices did not fit its shortened lists. About a third of the observations (35–52 %) fell off the end and were dropped, and the rest were matched to the wrong tie points.
  - The native index is now the rank of the keypoint index among the image's residual rows. Recomputing each residual from the model now reproduces the stored value to 1e-11 px; before, the median mismatch was 0.12–0.16 px.
  - Affected: the ε table (intra/cross split and totals), ε by range, and the `eps_*` rows of the parameter summary. Pair survival, the gate fit, decorrelation, the view graph and registration read the model directly and were not affected. Existing exports do not need to be redone.
  - With the fix, notebook 04 agrees with the ε that `export_for_error` reports. For example, Rockytop (rational) gives 0.235 / 0.315 px intra / cross, where the old mapping gave 0.207 / 0.199 px.

## 0.20.0 — 2026-09-26

Scope: Mars 2020 Navcam and Mastcam-Z at 34 mm.

- **Rational Navcam lens model (default).**
  - The three-term polynomial of the Metashape calibration cannot be inverted beyond ~0.88 of the corner radius (~53° off-axis). COLMAP could not undistort those pixels, so corner keypoints were never triangulated. At Three Forks: 45–57 % of corner keypoints matched, but 19 % (0.85–0.90 of the radius) and 0 % (beyond 0.90) became tie points.
  - A fourth polynomial term (Metashape K4) does not fit the corners either. One denominator term does: FULL_OPENCV (1 + k1 r² + k2 r⁴ + k3 r⁶)/(1 + k4 r²).
  - Shipped as `m20_cmods/M2020_N{L,R}_rational.json`: the author's calibration re-fitted with the corners, then the mean of two full bundle-adjusted solutions (Three Forks, 52 images; Bell Island, 145 images). These agree to 0.5 % in k1 and k4 and to 1 px in f. It is invertible over the whole frame (corners ≈ 61° off-axis). The bundle adjustment refines k1–k4, p1, p2 per project (`free_params`).
  - Bell Island, compared with the polynomial run: median / RMS residual 0.151 / 0.334 px (was 0.193 / 0.374 px), edge-to-centre residual ratio 1.41 (was 1.57), corner check 0.67.
  - Three Forks Navcam, same matches and settings:

    | | 3-term polynomial | rational |
    |---|---|---|
    | median / RMS residual | 0.232 / 0.382 px | 0.176 / 0.322 px |
    | tie points at 0.85–0.90 / 0.90–0.95 / 0.95–1.0 of the corner radius | 20 / 0 / 0 % | 44 / 41 / 31 % |
    | observations | 423,044 | 431,120 |
    | principal-point change | 3–5 px | ~1 px |

  - `SfmProject.create(navcam_distortion="polynomial")` keeps the old model. Notebook 03: `NAVCAM_DISTORTION`.
- **Weak attitude prior per frame** (`reconstruct(attitude_prior_deg=1.0)`, the CAHV pointing).
  - Block orientation was held only by the waypoint position priors, so a block of a few nearly collinear stations could rotate about the line through them. Three Forks: 0.59° about East with the rational model, 0.17° with the polynomial; 2.4° with Mastcam-Z.
  - With the prior the attitude change is 0.07° (median). Residuals are unchanged, because relative orientations come from the tie points.
- **Alignment health:**
  - Tie-point coverage in rings of image radius per camera, and the check `corner_triangulated_ratio` (triangulated share beyond 0.85 of the corner radius over that inside 0.6; warn < 0.5, fail < 0.25). It was 0.20 with the polynomial and is 0.63 with the rational model.
  - `station_shift_median_m` and `within_station_shift_spread_m` thresholds doubled (2/6 m, 0.1/0.4 m).
  - The camera-change check now samples the whole frame, corners included (it stopped at ~0.78 of the corner radius), and reports k4.
  - The edge residual uses the corner radius.
  - The report header names the project, the time and the MPPP version.
- **Defaults:**
  - `min_tri_angle_deg` 0.25 (stereo-only points to ≈ 97 m; was 0.5).
  - SIFT at native resolution for every product (`max_image_size` 5120; 3200 shrank full-resolution Navcam frames to 0.625×). Features are extracted once more.
  - p1, p2 are kept from the calibration and refined (`zero_terms=("b1", "b2")`, `refine_tangential=True`), as notebook 03 already did. The old library default zeroed them, and with the rational model that shifted the corners by up to 7.7 px.
- **Scope enforced:**
  - `SfmProject.create` accepts only NLF/NRF and ZL0/ZR0 at 34 mm, checked before any file is linked. It also refuses images processed with `resize.undistort=True`.
  - `process_images` warns about other products.
- **Fixes from a code review:**
  - `reuse_existing` now also notices a new waypoint table.
  - The bundle adjustment refuses to run without the database camera mapping when cameras carry `free_params`/`fixed_params` (they would have been ignored silently).
  - `mppp.colmap.unproject_camera` uses Newton iteration (the fixed-point version left 0.15 px in the rational corners); it now agrees with COLMAP to 1e-10.
  - `SfmProject.refresh_images` updates copied (not hard-linked) images when a project is reused (notebook 03).
  - `gate_curve` no longer overflows for CV → 0.
- **Provenance and licence:** NOTICE file and a README section. Images come from the PDS; the masks, mask model and camera models are the author's own work; the package is released under Apache-2.0 with MSSS approval and contains no sensitive data.
- Notebook 03: sites Bell Island (1451–1467) and Taylor Fjellet (1601–1646); `NAVCAM_DISTORTION`; attitude prior; k4 in the camera printouts; the last cell points to notebook 04. Notebook 04: the two new sites. Mastcam-Z focus-breathing plot: smaller markers.

## 0.15.0 — 2026-09-25

- **Notebook 04, error analysis of COLMAP alignments** (`notebooks/04_error_analysis.ipynb`, `mppp.error.alignment`). For one or more `error_input/` folders from notebook 03, it measures the numbers the error model runs on and sets them beside the values the model assumes:
  - image precision ε per instrument, same-station vs cross-station tracks, against range;
  - the cross-station match gate against convergence angle and |ΔLMST|: a maximum-likelihood fit of A (1 + θ/θ_c)^−k exp(−ΔL/L0), with bootstrap ranges, χ²/dof and a shape-free half-survival angle, per camera pairing and pooled over alignments;
  - decorrelation ρ(θ), the measured station view graph, and bundle adjustment vs telemetry per station.
  - Checked on Rockytop: θ̄ 5.6°, CV 0.38 (archive: 4.3° / 0.36 pooled, 2.4° / 0.50 for Rockytop). At Three Forks the cross-station rate is flat to ~15° and then drops sharply (half at 16°), a shape the power law cannot follow (χ²/dof 33); the notebook reports it as such.
  - A trial is "matched in image a; also matched in image b?" among the geometrically possible pairs, so terrain that never matched (occluded, masked, textureless) stays out of the denominator.
- **Bug fix, `mppp.error.colmap.measure_theta_c`:** the pairwise triangulations used rays aimed at the fitted 3D point, so every pair returned that point exactly; the residuals were rounding noise and ρ(θ) meaningless. They now use the observed keypoints, undistorted through the camera model (`observed_rays`, `mppp.colmap.unproject_camera`). Measured on real data: median pair residual 15 mm (Three Forks), 127 mm (Rockytop); ρ 0.00-0.03 at 0.25-5° (the model assumes ρ_0 = 0.22 decaying with θ_c = 0.4°).
- `residuals.npz` of 0.14.5+ exports are matched to the native model by keypoint index (its points are numbered afresh).
- `process_images(reuse_existing=True)` also reuses the newest manifest of an earlier version (`mppp_manifest_v0p14.json`), so a version change alone does not process everything again.

## 0.14.7 — 2026-09-25

- **New default mask model `mppp_mask_v2`**: `convnext_tiny_s4_seg_20260925.pt` (25 Sep 2026, 4 epochs, lr 5e-5, bf16; val IoU 0.9771, val loss 0.049 on the same split as v1's successor runs).
  - Exported to `mppp_mask_convnext_tiny_s4_v2.safetensors` (124,825,300 bytes, SHA-256 `227e4833…cef569b9`); the weights are identical to the checkpoint, and the export is deterministic.
  - `config["masking"]["checkpoint"]` defaults to `"mppp_mask_v2"`. Until it is uploaded (Hugging Face, GitHub release `mask-v2`), MPPP installs it from `checkpoints_dir()` into the cache on first use.
  - `mppp_mask_v1` stays in the registry: `"checkpoint": "mppp_mask_v1"`.
- **`process_images(..., reuse_existing=True)`**, used by notebook 03 (`KEEP_ONLY_REMAINING` was not reliable):
  - Before, `KEEP_ONLY_REMAINING = False` loaded whatever manifest was on disk. After a `True` run that was the reduced set (and it still printed "kept N of M … removed"). Frames deleted since, a changed configuration or a new mask model were not noticed. `True` processed every remaining image again (about 25 min at Three Forks) and rebuilt the COLMAP project on every run.
  - Now the manifest always describes the selection: `False` = every selected product, `True` = only those still in `images_png8`. Images already processed with the same configuration and with their files present are reused from the manifest (references and priors are rebuilt from it); only missing images are processed. A changed configuration processes everything and names the changed keys.
  - Notebook 03 rebuilds the COLMAP project when its images differ from the manifest (not on every `True` run).
- **Features are extracted again when an image or mask file changes** (e.g. processed with another mask model): `features.json` records each image's and mask's size and modification time. Records from before 0.14.7 have none, so the next run extracts once.
- **Project images that are copies (not hard links) are refreshed** when the processed image is newer.
- **Mask inference off at chosen rover stations**: `config["masking"]["skip_inference_at"]`, e.g. `["S032D1184"]` (also `[32, 1184]`, `"32/1184"` or a station label). Those images keep the rover; only invalid (black) pixels are masked. Recorded per image as `mask.inference_skipped`. With `reuse_existing`, changing the list processes only the images of the stations concerned. Notebook 03: `NO_MASK_INFERENCE_AT`.

## 0.14.6 — 2026-09-25

- **Mask training labels always come from `masks/`.** This was already how training read them; it is now enforced and tested.
  - `read_pair` (training and `audit_frames`) reads the image as 3-channel colour, so its alpha channel is discarded, and the label from the separate `masks/<name>` file.
  - The alpha of `images/` and `images_variable/` is only a copy written when those images were made; it goes stale when `masks/` is edited, and is never used.
  - `scan_dataset` refuses a `mask_dir` that is also an image dir; the checkpoint card records `training.labels`.
  - Test: an RGBA image whose alpha disagrees with its `masks/` file trains on the `masks/` file, with RGB unchanged.

## 0.14.5 — 2026-09-25

Fixes after the first Three Forks run with Mastcam-Z.

- **Mastcam-Z prior attitudes fitted to the camera's principal point.**
  - The label CAHVOR models move the principal point with focus and rotate the pointing to compensate. At Three Forks, ZR034's principal point moves ~160 px in x and ~120 px in y over focus counts −48 to 1266 (ZL034: ~12 px). A label attitude therefore fits only its own principal point.
  - With one principal point per eye (or per focus bin, held at the eye median), the priors were inconsistent by up to 1.3°. The stereo-pair spread was 1.29° and the rig rotation changed 1.34°. Mastcam-Z points failed the residual filter after round 1, and 223 of the 299 images (all the Mastcam-Z images) ended with < 30 observations.
  - Each Mastcam-Z prior is now rotated so that the pixel at its camera's principal point sees the ray its label model sees there (`prior_rotation_correction`). The correction is recorded per image as `prior_R_correction_deg` (Three Forks: median 0.07°, max 1.31°).
  - Checked on the Three Forks priors: the left/right relative rotations then agree to 0.08° (max) instead of 1.29°.
- **No stereo rig for Mastcam-Z by default** (`SfmProject.create(zcam_rig=False)`). Each Mastcam-Z image is its own frame: even corrected, the pairs disagree by up to 0.08° (~7 px at f = 4700 px), too much for a rigid constraint. Navcam keeps its rig (spread 0.0001°).
- **Round 4 repeats round 3** (8, 2, 2): it only shows whether another pass changes anything. The default schedule is `(24, 10, 8), (12, 2, 4), (8, 2, 2), (8, 2, 2)`.
- **Native model fixed:** `error_input/native` crashed the COLMAP GUI.
  - `images.txt` listed only the observed keypoints, but the tracks in `points3D.txt` used the full keypoint indices, so COLMAP stopped with `Check failed: point2D.point3D_id == point3D_id`.
  - It is now a complete COLMAP 4 model (cameras, rigs, frames, images, points), built by `native_reconstruction` and verified by reading it back with COLMAP's reader.
  - The error analysis was not affected: `mppp.error` matches observations by point ID.
- **COLMAP GUI copy in native pixels:** `reconstruct` also writes `gui_native/` (`write_gui_native`).
  - It contains the refined model and a database with native-pixel cameras (one per camera and resolution), keypoints and verified matches, so keypoints, tie points and matches line up with the half- and quarter-resolution images. Point colours are taken from the images.
  - It is for viewing only; the bundle adjustment works on the full-resolution project.
  - `open_in_colmap.bat` now opens this copy; `open_in_colmap_fullres.bat` opens the full-resolution project.
- **Notebook 03:** `ZCAM_RIG = False`. The project is rebuilt when a project from ≤ 0.14.4 is found, and the Mastcam-Z prior correction is printed.

## 0.14.4 — 2026-09-25

Still `v0p14`: manifests and notebooks keep the `v0p14` tag.

- **Mastcam-Z focus breathing: one camera per focus bin.**
  - `SfmProject.create(zcam_focus_bin=30)` (the default) splits each Mastcam-Z eye and zoom into cameras by focus motor count, named e.g. `ZL034_F02312` after the bin's median count.
  - Bins are at most 30 counts wide. They are grouped greedily from the lowest count (`focus_bins`), so a cluster of nearly equal counts is never split by a grid line.
  - Each bin starts from the median label focal length of its images.
  - `zcam_bin_refine="focal"` (the default): a bin refines fx and fy only. The principal point, k1–k3 and p1/p2 are held at the median of the whole eye and zoom, because for a ~25° field the principal point of a few images is nearly degenerate with their attitude. `"all"` refines everything, like Navcam.
  - `zcam_focus_bin=None` gives the 0.14.3 behaviour: one camera per eye and zoom.
  - Images record `camera_group`, `focus_count` and `label_f_px`.
- **Per-camera held parameters:** `project.cameras[key]["fixed_params"]`, honoured by `bundle_adjust`.
- **Focal length vs focus** (`mppp.sfm.zcam`, notebook section 7b).
  - `write_focus_breathing(proj, rec)` plots, per eye and zoom: refined f per bin (marker size ~ observations), the bin start, the per-image label f, and linear fits (refined bins weighted by observations, bins with < 100 observations excluded).
  - Slopes are given in px/count and %/1000 counts.
  - Output: `health/zcam_focus_breathing.{png,csv,json}`.
- **Top-down camera shifts** (`plot_camera_shifts`, notebook section 8, first cell). It replaces the per-station plot. Stations are tens of metres apart but the cameras of one station lie within a metre, so the figure has three parts:
  - an overview of the station median shifts;
  - every camera's total shift per station, coloured by attitude change;
  - one panel per station in local coordinates, with an arrow per image from its prior (CAHV + waypoint) to its refined centre, a shared exaggeration, and colour = vertical shift.
  - Navcam and Mastcam-Z have different markers; held images are grey crosses. Saved as `error_input/camera_shifts.png`.
- **COLMAP GUI files** in the project folder, written by `build_database` and `reconstruct` (`write_gui_project`):
  - `colmap_gui.ini`: File > Open project (database, images, masks);
  - `open_in_colmap.bat`: opens the GUI with the project and `sparse/cahv_ba` in one step. It uses `COLMAP_BAT`, e.g. `setx COLMAP_BAT D:\tools\COLMAP\COLMAP.bat`.
- **Health:** the left/right eye ratio compares eye groups (`ZL034` vs `ZR034`), not individual focus bins.
- **Station labels start with the sol:** `Sol0686 S032D1184`, or `Sol0684-0685 S032D1174` for a station occupied over several sols (`station_labels`, `SfmProject.station_label`).
  - Used in the health table, weak-image list and health plot, the camera-shift plots and the export summary.
  - Added as a `station_label` column to `stations.csv` and `poses.csv`.
  - The station ID (site/drive) is unchanged and still groups the images.
- **Notebook 03:**
  - `ZCAM_FOCUS_BIN = 30`, `ZCAM_BIN_REFINE = "focal"`; the project is rebuilt when these change.
  - The focus-bin summary is printed after the project is built.
  - With `GPU_PY = None`, extraction and matching no longer request the GPU, so the "no CUDA" warning is gone.
  - The default test case is Three Forks (`SITE = "threeforks"`, sols 684–693).

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
