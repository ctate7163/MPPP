# Changelog

The development history v0p1–v0p12 (21–24 September 2026) is in [docs/history/CHANGELOG_v0p1-v0p12.md](docs/history/CHANGELOG_v0p1-v0p12.md).

## 0.35.2 — 2026-09-29

- **`LOCALIZE_MIN_IMAGES`** (notebook 03, default 4; `reconstruct(localize_min_images=...)`, `bundle_adjust`, `project.settings["localize_min_images"]`): stations (site, drive) with fewer images than this get no waypoint position prior. They are short stops during a drive, and their waypoints can be metres off: at Seitah North four two-image drives on sols 238–239 sat 3.0–3.1 m from their waypoints. Before the first round these stations are placed on the block from their tie points (`register_stations(only=...)`). After that every adjustment, including the thermal stage, leaves out their position priors; their attitude priors stay. A station keeps its prior when it shares fewer than 20 tie points with the others, or when every station is below the minimum (`unlocalized_stations`). Recorded in `project.settings["reconstruction"]["unlocalized_stations"]` and `["unlocalized"]`, and listed after the alignment.
- **Health:** stations without a position prior are left out of `station_shift_median_m` and of the waypoint-layout fit (`prior_scale_error_*`). Their largest shift is reported as `unlocalized_station_shift_max_m`, with no threshold. In `stations`, `position_prior` is False for them. On the Seitah North block, re-adjusting without the five two-image stations' priors takes the prior scale error from 8.6 % (1.91 m, FAIL) to 1.5 % (0.27 m, pass). The block itself does not move: the station shifts agree to 1 mm, because the tie points already outweighed those priors. The health verdict goes from FAIL to WARN; `block_rotation_deg` 0.42° remains.
- **`station_map.png` / `camera_shifts.png`** (`plot_camera_shifts`): only the two overview panels are drawn: station arrows, and the per-camera shift by station. The per-station panels are available with `station_panels=True`. Stations without a position prior are drawn with dashed arrows and marked "(no prior)".
- Tests: `tests/test_v0p35p1.py`.

## 0.35.1 — 2026-09-29

Is the Navcam rig's drift robust? Six more blocks, a review of the sites, and the resolutions (working notes §16; results in `docs/results/v0p35p1/`):
- **Notebook 03 fix:** the camera print-out after the alignment raised `IndexError` when every frame of a block sat in a temperature bin (Whale Mountain): the eye's camera then has no image. `reconstruction.camera_changes` / `print_camera_changes` look cameras up by database id and mark bin cameras and eyes without images. Notebook 03 also warns when a site's WORK folder holds a block with no image in its sol range (Origny in `olifants_colmap`), lists the new sites (van_zyl, seitah_north, whale_mountain, pico_turquino, origny = the former olifants range, olifants 1880–1889) and points `NAVCAM_CAMERAS` at `navcal_v0p35p1`.
- **Rig study on new blocks:** `navcal.load_scape` reads a block solved with thermal bins from `sparse/cahv_ba_single` with the bins stripped, and refits cameras of another lens model (`lens=`, `fit_camera_model`: rational ↔ fisheye + tangential); `navcam_calibration_study.py rig --rig-lens --common-pp` keeps new blocks comparable with an earlier study.
- **One label reference** (`navcal.common_reference`, applied by `navcal_report.load_rig`): the label rig of blocks before sol ~250 differs by 0.23 mdeg in pitch and 0.58 in roll; all angles are now referred to one.
- **Findings** (Butler Landing, Van Zyl, Seitah North, Three Forks South, Pico Turquino added; Whale Mountain shown, not fitted): the v0p35 drift predicts every block after sol 240 within 2 σ. **Pitch drift is robust but faster early**: the two blocks before sol 100 (Butler Landing, Van Zyl) lie 4–6 mdeg below the line; a hinge at sol 300 (+25.8 mdeg / 1000 sol before, +4.46 after) cuts the leave-one-block-out error from 1.8 to 1.1 mdeg. **Roll drift holds** (−6.1 ± 1.1) with a ~3 mdeg block floor (Three Forks South vs Three Forks: pitch 0.08, yaw 1.3, roll 3.4 mdeg apart). **Yaw drift is marginal** (+3.9 ± 1.6, p 0.015; small prediction gain) and kept. **Yaw vs temperature is robust** (within blocks −1.05 ± 0.03 mdeg/°C, between −1.17 ± 0.31).
- **Rig file drift** (`navcal.rig_drift_model` with `knot_sol`, `drift_offset_mdeg`; `start_rig_rotation` applies the hinge): `D:\scapes\colmap\camera_analysis\navcal_v0p35p1\navcam_joint` (cameras as v0p35). Notebook 04 §2f (`drift_robustness` with leave-one-block-out prediction, `navcal_report.drift_figure`) computes it; §2e writes it.
- **Resolutions** (notebook 04 §2g, `navcal.split_by_scale`, `navcal.scale_offsets`, `navcam_calibration_study.py scale`, `navcal_report.scale_summary`, `label_scale_consistency`): no systematic pixel offset between full, half and quarter resolution. The label models of 1,760 products agree across scales under MPPP's mapping (x_full = x / scale, corner origin) to ≤ 0.03 px; per-scale cameras in 20 blocks show block-specific common shifts (±1 px, the same in both eyes: frame attitude) but no common offset (quarter − half +0.02 ± 0.14 px) and left − right differences of 0.13 px rms averaging zero.
- **Sites flagged:** butler_landing (weak span, rig held), whale_mountain (2 stations), seitah_north (health FAIL on the waypoint scale), threeforks_south (contains Three Forks), origny (lives in `olifants_colmap`), olifants 1880–1889 (no PDS products on disk), groloy (not run), van_zyl (quarter resolution, early labels).
- Tests: `tests/test_v0p35p1.py`.

## 0.35.0 — 2026-09-29

Navcam calibration across fifteen blocks: rig stability, one joint calibration, lens-model choice (`mppp.sfm.navcal`, `scripts/navcam_calibration_study.py`; working notes §15, methods §16; results in `docs/results/v0p35/`):
- **Rig study** (`rig_study`, `rig_tests`, `rig_drift`): each block re-adjusted with the rig rotation free (with free and with common principal points), with the rig translation free, and with one rig per 10 °C bin, every rig with its covariance (`bundle_adjust(covariance=True)`: 3-D points eliminated, frame poses marginalised). Findings: the rig yaw turns with camera temperature, **−1.13 ± 0.04 mdeg/°C within blocks** and −1.11 ± 0.31 between blocks (0.05 px of disparity at infinity per °C); pitch (+0.0050 mdeg/sol) and roll (−0.0060 mdeg/sol) drift over the mission; the left–right temperature difference and the network strength do not explain the rig; the baseline length is not observable (σ 2–26 mm) and stays at the CAHV value; a block-to-block residual of 2–4 mdeg remains (formal σ 0.1–0.3 mdeg).
- **Joint calibration** (`merge_scapes`, `joint_adjust`, `profile_slope`): one NL and one NR camera, one rig and each block's poses and points in one adjustment; the focal thermal model and the rig's temperature term enter through the keypoints (`bundle_adjust(keypoint_scale=..., keypoint_map=...)`). Focal slope **38.6 ppm/°C** (the start-camera slope; within-block 30, across-block 55–60), rig yaw slope **−1.03 mdeg/°C**; cameras with covariance (fx σ 0.02 px).
- **Leave one block out** for the rational and the fisheye + tangential (`THIN_PRISM_FISHEYE`, sx1 = sy1 = 0) models: the shared cameras predict a new block to +0.57 % in cost and 0.25 px rms over the frame; the rig transfers less well (+3.4 % held, +4.5 % without the temperature term; the drift halves the rest). The fisheye + tangential model fits all 15 held-out blocks better (−0.6 % cost, corners −8 %) with the same transfer: **it is the lens model of the frozen cameras.**
- **Frozen cameras** (`write_joint_cameras`, notebook 04 §2e): `M2020_NL/NR_fisheye_tangential.json` (or `_rational.json`) at T0 with the thermal slope, covariance and leave-one-out verification, and `M2020_N_rig.json` with the rig's `thermal` (yaw, pitch mdeg/°C) and `drift` (mdeg/sol) models. Written to `D:\scapes\colmap\camera_analysis\navcal_v0p35\navcam_joint` (fisheye + tangential) and `...\navcam_joint_rational`.
- **Pipeline:** `SfmProject.create(navcam_distortion="fisheye_tangential")` (needs `navcam_cameras`); the start rig is turned to the block's median camera temperature and sol (`start_rig_rotation`); the thermal stage turns each bin's rig by the slope (`split_by_temperature(rig_slopes=...)`, `rig_slopes_for_project`); `reconstruct(refine_rig="auto")` refines the rig rotation on a strong network and holds it on a weak one. `mppp.colmap.project_camera` handles any COLMAP model through pycolmap (fisheye models in the error analysis and notebook 04); `calibration.Camera.distortion` names the fisheye lenses.
- **Notebooks v0p35** (`*_v0p35.ipynb`, and the unversioned copies): 03 defaults `NAVCAM_DISTORTION = "fisheye_tangential"`, `NAVCAM_CAMERAS = ...navcal_v0p35\navcam_joint`, `NAVCAM_RIG_REFINE = "auto"`; 04 gains §2d (rig tests, joint calibration, leave-one-out, figure) and §2e (frozen cameras); 01 and 05 carry the version.
- Tests: `tests/test_v0p35.py`.

## 0.31.2 — 2026-09-28

The default mask model always comes from Hugging Face (`ctate7163/mppp-mask`):
- **`mppp_mask_v3` registry entry corrected.** The SHA-256 recorded in 0.22.4 (`daa34ffb…`) was of an export made without the checkpoint's `.json` card: that file carries the default card (no stride-4 decoder, threshold 0.4) and does not load. The released file, exported with its card, is `46830126…` (124,826,276 bytes); it loads and gives the same output as the `.pt`. Its only URL is `https://huggingface.co/ctate7163/mppp-mask/resolve/main/mppp_mask_convnext_tiny_s4_v3.safetensors`; `"hf_repo"` names the repository.
- **A registry name means the released file.** `fetch_model` / `resolve_checkpoint` check a cached copy against the registry SHA-256 (once per process) and download it again if it differs (the old copy is kept as `*.sha256-mismatch`). A local checkpoint no longer stands in when the download fails, unless `MPPP_MASK_LOCAL_FALLBACK=1` (or `local_fallback=True`). To run another model, give its path in `config["masking"]["checkpoint"]`.
- **`python -m mppp.mask.hub upload`** (`upload_model`): exports the model if needed, refuses a file whose SHA-256 is not the registry's, creates the Hugging Face repository and uploads the file and `docs/hf_model_card.md` as `README.md`. **`verify`** (`verify_model`) downloads from every registry URL and checks the SHA-256.
- `HF_TOKEN` (or a `hf auth login` token) is sent to huggingface.co only, for a private repository.
- `export_safetensors` refuses a `.pt` without its `.json` card (`allow_default_card=True` to override).
- Model card `docs/hf_model_card.md` and `docs/RELEASING.md` §4–6 rewritten for v3.

## 0.31.1 — 2026-09-28

Checks of the thermal model and the error-model inputs of fifteen Navcam scapes (working notes §14):
- **Leave-one-scape-out** prediction of each scape's Navcam camera from the other eight: the thermal model cuts the fx prediction error from 0.78 to 0.45 px rms (within-block slope) or 0.33 px (across-scape slope); over the whole frame both give 0.31–0.32 px against 0.44 px without a model.
- **Geometry of the temperature bins**: against one camera per eye, the binned blocks change shape by 0.1–3 mm within 6 m and 1–28 mm at 12–25 m (60–1600 ppm of range; largest at Pearce Canyon); Taylorfjellet also turns by 0.05°. `thermal_adjust` returns the reference reconstruction (`reference_rec`).
- **`scripts/error_analysis_batch.py`**: notebook 05's measurements (ε, gate, decorrelation, view graph, registration, parameter row, pooled gate and form comparison) one alignment at a time; notebook 05 with all fifteen alignments loaded at once needs more than 8 GB.
- **`scripts/site_appearance.py`**: texture and contrast of processed images inside the terrain mask and below −3° elevation (band-pass contrast at the SIFT octaves, rms contrast, spectral slope, coherence, SIFT density and response, repetitiveness, shadow fraction, dynamic range).
- Findings: ε (0.27 / 0.32 px same / cross station), the zero-angle cross-station rate (0.63) and the gate form are common to all sites; θ½ (3.6–15°) and τ (1.4–9 h) are site parameters, θ½ following the network (cross-station fraction), not the texture. Single-image contrast mostly follows the sun elevation; after removing it, only the SIFT keypoint strength relates to the cross-station rate (ρ −0.71).
- Notebook 05 defaults: the fifteen Navcam scapes under `D:\scapes\colmap`. Results in `docs/results/v0p31/`.

## 0.31.0 — 2026-09-28

Navcam focal length and camera temperature (`mppp.sfm.thermal`; working notes §13, methods §15):
- **Temperature-bin experiment** (`scripts/temperature_bins_experiment.py`): on a solved block, every Navcam image gets its camera-plate temperature (`NAVCAM_LEFT_CAL` / `NAVCAM_RIGHT_CAL` of the label, interpolated in spacecraft clock between the labels of the sol where needed; leave-one-out error 0.4–0.6 °C median), each eye is split into one camera per 10 °C bin (bins under 8 images merged into a neighbour) with fx, fy free and everything else of the camera and the rig shared, and the block is re-adjusted; a reference adjustment with one camera per eye and the same freedom is run first. Nine scapes, 21 bins: **fx rises 0.094 ± 0.006 px/°C (NL) and 0.086 ± 0.007 px/°C (NR), about 30 ppm/°C**, with one offset per scape; fy follows with more noise (0.06 ± 0.02 px/°C). Across scapes the slope is 0.16–0.18 px/°C.
- **Thermal stage in the pipeline**: `reconstruct(temperatures=..., thermal_bins_deg=10)` ends with the same split and adjustment (bins held at the start camera scaled by the thermal model when the Navcam intrinsics are held). The binned block is the delivered model; the one-camera solution is kept in `sparse/<out>_single`. Images of a bin carry the bin camera as `instrument` (the eye in `base_instrument`); `strip_thermal_bins` undoes the split before the database is built and at the start of every reconstruction, so reruns start from one camera per eye. The report is in `project.settings["thermal"]`. Notebook 03: `THERMAL_BINS_DEG = 10`, `THERMAL_MIN_IMAGES = 8` (None = off); temperatures from the manifest, else the labels under `PDS_DIR`.
- **Temperature-corrected consensus**: notebook 04 section 2a fits both slopes, and `CAL.thermal_model` (`THERMAL_SOURCE = "auto"`: within-block if measured, else across scapes; or `"fixed"` with `THERMAL_PPM`) scales every refined camera to a reference temperature T0 (the observation-weighted mean) before `consensus_camera` averages them; `reference_differences` and the consensus verification scale the consensus to each camera's temperature. The consensus JSON carries `thermal` {ppm_per_degC, T0_degC}, and `SfmProject.create(navcam_cameras=...)` scales the start fx, fy to the block's median camera temperature. On the nine scapes the mean distance of a scape's camera from the consensus drops from 0.39 to 0.27 px rms (Taylorfjellet 0.68 → 0.29 / 0.32 px).
- Manifests: `camera_temperature_degC` is filled from the label when an older manifest entry is reused; project image records carry it.
- Notebook 03 rebuilds the project when the consensus files in `NAVCAM_CAMERAS` change (`navcam_cameras_fingerprint`), since a consensus rewritten in the same folder has the same path.

Other changes:
- **Outlier frames**: `OUTLIER_DEFAULTS["min_residual_px"]` 0.6 → **1.0 px** (the residual test flags a frame whose median residual exceeds both 3 × the block median and 1 native px; the shift and attitude tests are unchanged).
- **Notebook order**: 04 is now camera models, 05 error analysis (constrain the cameras before the error analysis); `run_scapes.py` runs them in that order.
- Notebook 04: the difference maps share one colour scale (98th percentile over all panels) and are drawn as images of their sample grid (the square markers aliased into moiré when scaled); bin cameras are labelled; stereo and radial-profile sections use the most observed camera of an eye.
- Notebook 03 defaults (from the user's settings): `KEEP_ONLY_REMAINING = 1`, `SKY_ELEVATION_DEG = 20`, `LMST_WINDOW_H = (8, 17)`, `NAVCAM_INTRINSICS = "auto"`, `NAVCAM_CAMERAS = D:\scapes\colmap\camera_analysis\2026-09-28\navcam_consensus`, `NO_MASK_INFERENCE_AT = []`; sites: `rochette` (179–190) and `threeforks_south` (652–683) added, `threeforks_large` removed; `butler_landing` 1–14, `sid` 361–378, `airey_hill` 960–991, `rio_chiquito` 1333–1337. Notebook 04 defaults: the 15 scapes of the user's list.
- `bundle_adjust` skips the prior of a frame whose reference image the project does not list, and `split_by_temperature` keeps deregistered frames unregistered.

## 0.30.0 — 2026-09-28

Image selection and processing:
- **Navcam 10 % darker** (`color.brightness_by_family = {"N": 0.9}`, notebook 03 `NAVCAM_BRIGHTNESS`): a per-family brightness factor applied with the white balance keeps nominally exposed Navcam frames off the top of the 8-bit range.
- **LMST window** (`selection.lmst_window_h = [9, 17]`, inclusive) and **saturation limit** (`selection.max_saturated_fraction = 0.05`, of the valid pixels at the product's maximum DN): images outside the window or above the limit are not processed and are listed in the manifest under `skipped` with the reason, like sky-pointing frames. Every image's `lmst_h` and `saturated_fraction` are in the manifest.
- **`process_images(workers=4)`**: images are processed in worker processes (spawned; each loads the mask model once); results keep the selection order. `workers=1` or `stop_on_error` processes in the calling process; if the pool fails (a dead worker, or a `__main__` that cannot be re-imported) the remaining images are processed in the calling process instead of failing.

Reconstruction:
- **Bundle adjustment**: the problem is built about twice as fast (per-image lookups hoisted out of the observation loop), and the linear solver is chosen by block size (`linear_solver="auto"`: dense Schur up to 600 frames, where the Schur complement is small and the dense factorisation is multi-threaded; iterative Schur with a Schur-Jacobi preconditioner beyond; `"sparse_schur"` remains). The solver, thread count and Ceres time are in the log.
- **Exhaustive matching** takes `block_size` (default 100, COLMAP's 50): fewer descriptor reloads and GPU pipeline drains per block.
- **Five-site consensus start cameras**: `M2020_N{L,R}_rational.json` and `M2020_N_rig.json` are now the observation-weighted consensus of the 0.22.4 solutions of Three Forks, Belva Crater, Bell Island, Olifants and Marble Mountain (notebook 05, 28 Sep 2026), with each site's distance from the consensus recorded in the files.
- `THIN_PRISM_FISHEYE` and `OPENCV_FISHEYE` cameras are handled by `bundle_adjust` (tangential and thin-prism terms held unless freed by `free_params`; `_PARAM_NAMES` gives the names for any model). **Lens-model test** (`scripts/lens_model_experiment.py`): repeats the final adjustment of a refined block with the Navcam in the rational, the θ-polynomial fisheye, the fisheye + tangential and the thin-prism model (working notes §11). The rational model remains the default.

Health and viewing:
- **`health/station_map.png`**: a top-down view of the site with every camera's prior-to-refined shift (the figure of `plot_camera_shifts`), written with the health report and shown in notebook 03 section 7.
- **`block_rotation_deg`** check: the median world-frame attitude change of all frames (about E, N and U) is the rotation of the whole block away from the East-North-Up frame the priors define; warn 0.3°, fail 1°. The report records the world frame and its offset.
- **Tie-point colours** in the GUI copy are the mean over all valid observations of each point's track (`export.point_colors_from_tracks`); black (invalid) pixels are left out, and a point with no valid sample is mid-grey. COLMAP's own extractor left points black when the image it happened to sample could not be read or the keypoint fell on an invalid pixel.
- **`open_in_colmap.bat`** starts COLMAP with the database, the images and the refined model in one go. It looks for `COLMAP.bat` at the path given in notebook 03 (`COLMAP_BAT`, kept in `project.settings["colmap_bat"]`), then at the usual install folders, then in the `COLMAP_BAT` environment variable, then on the PATH.

From the review of the nine 0.22.4 alignments (Three Forks, Rockytop, Belva Crater, Pearce Canyon, South Arm, Bell Island, Taylorfjellet, Olifants, Marble Mountain; working notes §12):
- **Right-only Navcam exposures are kept**: a right image whose left partner is missing (16 of 176 at Pearce Canyon, where they made up the whole `registered_fraction` warning) goes into a one-sensor rig with the right camera as reference, posed from its own CAHV prior with the shared right-camera intrinsics. The database summary lists them (`single_eye_images`).
- **Pose checks split**: `block_rotation_deg` now uses the world-frame rotation taking the prior attitude to the refined one (the sign was reversed); `attitude_residual_p95_deg` (warn 0.75°, fail 2°) judges each frame's attitude change about the block rotation; the undivided `attitude_change_p95_deg`, which warned at seven of nine sites on ordinary label pointing scatter, is reported without a verdict. The layout scale error is judged as the displacement it makes at the stations (`prior_scale_error_m`, warn 0.5 m, fail 1.5 m) instead of a percentage that flagged every small block; `prior_scale_error_pct` and the layout rotation (deg and m) are reported.
- **Camera temperature**: the manifest records `camera_temperature_degC` (the temperature the label camera model was interpolated to); `attach_pds_labels` adds it to older manifests; `calibration.camera_temperatures` / `focal_temperature_fit` and notebook 05 section 2a test whether the scape-to-scape focal-length spread (Taylorfjellet +1.4 px in both eyes) follows it. Notebook 05 defaults: the nine scapes under `D:\scapes\colmap`, `PDS_DIR = D:/data/m2020`.
- **CAHVORE type 3 fit** (`cmod.fit_to_colmap`): also started from the type-2 geometry at linearity 0, 0.5 and 1, keeping the best finite solution (the nine-scape NL fit had ended at linearity 0.99 with a 51° O tilt and an rms of NaN).
- Notebook 03: every results cell starts with the site, its sol range and its COLMAP folder (`banner()`).

Notebook 03: rewritten first cell; the site table of 17 sites; `KEEP_ONLY_REMAINING = False`; `ADD_NEARBY_WAYPOINTS = 10`; `SKY_ELEVATION_DEG = 10`; `GPU_PY` and `COLMAP_BAT` set; `MAX_NUM_FEATURES = 16000`; `MATCH max_distance = 1.0`; three-round schedule with Cauchy scale 4 in round 2; `ATTITUDE_PRIOR_DEG = 5`; `LINEAR_SOLVER`, `MATCH_BLOCK_SIZE`, `WORKERS`.

## 0.22.4 — 2026-09-28

- **Mask model v3 is the default** (`mppp_mask_v3`, exported from `checkpoints/convnext_tiny_s4_seg_20260925b.pt`: 9 of 10 epochs, val IoU 0.979; SHA-256 `daa34ffb…`). Until the safetensors is published, `fetch_model` installs the local `.pt` from `checkpoints/` automatically; `python -m mppp.mask.hub export checkpoints/convnext_tiny_s4_seg_20260925b.pt --name mppp_mask_v3` writes the release file.
- **Sky-pointing Navcam frames are not processed** (`selection.max_boresight_elevation_deg`, 45°; notebook 03 `SKY_ELEVATION_DEG`): a frame whose label boresight points higher raises `SkyImage` before radiometry and mask inference and is listed in the manifest under `skipped`. Mastcam-Z is not filtered.
- **`only_existing` tolerates a first run**: when the image folder does not exist yet, or holds none of the selection, everything is processed and the manifest says the filter was ignored; the next run applies it as before (`KEEP_ONLY_REMAINING = True` is now the notebook default).
- **Navcam intrinsics are refined by default** (`reconstruct(navcam_intrinsics="refine")`, notebook 03 `NAVCAM_INTRINSICS = "refine"`); `"auto"` (hold on a weak network) and `"hold"` remain options, and the network verdict is still logged.
- **`ADD_NEARBY_WAYPOINTS`** replaces `NEARBY_M` in notebook 03: metres, inclusive, default 5; 0 turns it off.
- Notebook 03 defaults: `SCAPES_ROOT = D:/scapes/colmap`; sites `butler_landing`, `rockytop`, `threeforks` (685–692), `threeforks_large` (652–693), `belva_crater`, `tuxedo_park`, `airey_hill`, `bunsen_peak`, `pearce_canyon`, `rio_chiquito`, `south_arm`, `bell_island`, `taylorfjellet`, `olifants`, `groloy`, `marble_mountain`; `SITE = "rockytop"`; `STORE_MASK_IN_ALPHA = True`; `ZCAM_RIG = True`; `MAX_NUM_FEATURES = 12000`; round 2 of the schedule at Cauchy scale 4; `NO_MASK_INFERENCE_AT = ["S032D1184"]`.

## 0.22.3 — 2026-09-27

- **Other visits to the same spot** (`select.find_imgs_near`, `waypoints.stations_near`; notebook 03 `NEARBY_M`): the products of the sol range plus those of every waypoint station within `NEARBY_M` metres of a station imaged in the range, whatever their sol. The report lists each added station, its distance and nearest in-range station, and its images. Off by default; with a value the work folder gets a `_near<N>m` suffix.
- **Lens-term test** (`scripts/lens_terms_experiment.py`): repeats the final adjustment of a refined Navcam block with the same observations and k4 alone, + k5, + k5 + k6, without p1 and p2, and both, and reports cost, BIC, residuals by image radius, the mean residual field, the refined terms, the camera change and the monotonic range of the radial mapping.
- **Figures.** Notebook 04: ε against convergence angle draws the cross-station curves first, with fixed colours per series and the legend ordered as the curves lie; the gate panels share their x and y axes. `scripts/sites_table.py`: one LMST panel per site (the union of its scapes, each image once) plus one for all sites, Navcam and Mastcam-Z each normalised to unit area, 5–19 h with a dashed line at noon and no y ticks; a per-site table (`sites_by_site.csv`, and in `sites.md`).

## 0.22.2 — 2026-09-27

Methods only: numbers measured on particular scapes are no longer stated here, in `docs/methods.md`, in the notebooks or in code comments. They are collected in `docs/results/working_notes.md` (with the statements they supersede marked as withdrawn) until the paper's results and discussion sections take them over. Earlier entries below keep their numbers as history; where a later measurement contradicts one, the working notes say so.

- **Every gate parameter is fitted from the data** (`fit_gate`): the amplitude, the angle form's parameters and the illumination e-fold (`fit_tau=True` by default). Four angle forms — power / gamma-mixture `(1 + θ/θ_c)^−k` (the model's), exponential, stretched exponential, logistic — and four illumination covariates — `|ΔLMST|` (e-fold `tau_h`), the angle between the sun vectors (`sun0_deg`), the shadow-tip distance of a unit post from the label's solar azimuth and elevation (`s0`), none — are fitted to the same binned trials and compared by AIC (`compare_gate_forms`; notebook 04 section 3b). Log-parameters are bounded (`Q_LIMIT`); a fit at a bound or with its angle scale outside the fitted range is reported as `constrained = False` for every form. `gate_expected` takes the fit dict.
- **ρ(θ) re-estimated** (`decorrelation`, `fit_rho`): pair-triangulation residuals of tracks of 4–12 images, correlated only between pairs that share no image (pairs sharing an image are correlated by that image's own error), the −1/(M−1) constraint bias of residuals about their mean removed, fine bins near zero (`RHO_EDGES_DEG`), standard errors from a bootstrap over tracks, and Gaussian / exponential fits of ρ_∞ + ρ_0 f(θ/θ_c). The first bin says whether ρ is consistent with zero at all.
- **ε against convergence angle** (`eps_by_angle`; notebook 04 section 2b): per observation the largest ray angle of its point and the angle to the nearest other ray, same-station and cross-station apart, so that precision degrading with angle is measured separately from the completeness loss the gate describes.
- **Solar geometry per image**: `stations.csv` carries the label's solar azimuth (with the elevation and LMST it already had); `load_alignment` attaches the sun vector and shadow-tip vector per image (from `stations.csv`, else the newest manifest), and `pair_survival` reports `dsun_deg` and `dshadow` per pair.
- **`L0` renamed `tau`** throughout `mppp.error` and notebook 04 (`tau_h`, `fit_tau`, `FIT_TAU`): L_s / L_0 is the solar longitude and is not an illumination e-fold.
- **Navcam intrinsics held where the network is weak.** `navcam_network(project)` counts the Navcam stations and their span; `reconstruct(navcam_intrinsics="auto")` (default) holds the Navcam cameras at their start values when there are fewer than `NETWORK_MIN_STATIONS` (4) stations or less than `NETWORK_MIN_SPAN_M` (5 m) of span, and refines them otherwise; `"hold"` / `"refine"` force it; `hold_cameras` holds any camera; `refine_rig=False` holds the rig rotation (`bundle_adjust(hold_cameras=...)`). Notebook 03: `NAVCAM_INTRINSICS`, `NAVCAM_RIG_REFINE`, `HOLD_CAMERAS`.
- **Staged Navcam-then-Mastcam-Z** (`reconstruct(staged=True)`; notebook 03 `STAGED = True`): the Navcam frames are solved on their own through the whole schedule and exported (`sparse/<name>_navcam`, `error_input_navcam/`), then the Mastcam-Z frames are registered at their priors and solved with the Navcam cameras and rig held. The Navcam-only and the combined solutions share images, features, matches and database.
- **Verified consensus cameras from notebook 05** (`calibration.write_navcam_consensus`): the consensus rational Navcam cameras and mean rig in the format of the shipped start cameras, each scape's distance from the consensus recorded as `verification`; `SfmProject.create(navcam_cameras=<folder>)` (notebook 03 `NAVCAM_CAMERAS`) starts a project from them.
- **Sites table and LMST histogram** (`scripts/sites_table.py` → `docs/results/sites.md`, `sites.csv`, `sites_lmst.png`): the scapes with site numbers, sol ranges, images per camera, stations, span, LMST and sun-elevation spread, and the LMST-of-day distribution per scape.
- Notebook 04 no longer compares against the archive's case shapes or its pooled ε; the model's values are `ModelConfig`'s. Notebook 05's repeatability plot draws `EPS_REF_PX` instead of a fixed line.
- `_observations` returns the image id of each observation (`image`).

## 0.22.1 — 2026-09-26

Fixes from the first v0p22 site runs (Airey Hill, Bell Island, Three Forks 684–693, all Navcam + Mastcam-Z 34):
- **Outlier exclusion told apart from unconstrained frames.** Frames with no tie points at all (defocused focus-stack members, calibration-target and sky images: 79, 16 and 37 images at the three sites) are reported as `kind: "unconstrained"` ("no tie points") and counted by the health check as `unconstrained_fraction` (warn 25 %, fail 60 %), not as outliers. `excluded_fraction` now counts real outliers only (17, 0 and 15 images).
- **The attitude test of `find_outlier_frames` applies to Navcam frames only.** A Mastcam-Z prior attitude is one mast pointing, whose error is per image, so a Mastcam-Z frame 0.5–1.5° from its station's median (with up to 1300 tie points) was wrongly excluded.
- **DOF-corrected ε for subsets.** For one family, instrument or track kind, a point shared with other observations now absorbs only its share of 3 degrees of freedom (3 Σ 1/track length over the subset), instead of 3 per point touched; one eye of a stereo pair was overcorrected (0.86 px instead of 0.44).
- **Notebook 05 with mixed Navcam + Mastcam-Z runs.** `rig_geometry` picks the Navcam rig (a Mastcam-Z pair sharing a clock also becomes a frame with a rig; the label stereo comparison had used the 0.243 m Mastcam-Z rig). `consensus_camera(max_rms_px=1.5)` leaves out cameras far from the mean of the others (`excluded`), and `mean_rig(max_dev_mdeg=50)` leaves out rigs far from the median: Three Forks 684–693 with Mastcam-Z has only 35 left Navcam images at three stations 3 m apart, and its Navcam intrinsics drifted 8.5 px (cy −45 px, rig yaw −146 mdeg) with the block still fitting to 0.13 px — the intrinsics are not observable from such a network (methods §14).

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
