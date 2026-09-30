# Changelog

The development history v0p1–v0p12 (21–24 September 2026) is in [docs/history/CHANGELOG_v0p1-v0p12.md](docs/history/CHANGELOG_v0p1-v0p12.md).

## 0.51.0 — 2026-09-30

Navcam rig yaw held or zero; partly transparent masks.
- **`NAVCAM_RIG_YAW`** (notebook 03; `SfmProject.create(navcam_rig_yaw=)`, `project.settings["navcam_rig_yaw"]`): `"refine"` (default, as before), `"hold"` (the start rig's yaw, i.e. the joint rig at the block's temperature and sol, kept), or **`"zero"`** (the yaw - rotation of the right camera about the left camera's y axis, which shifts the disparity - set to 0 in the start rig and held; the principal points absorb the ~10 mdeg block-to-block differences). Pitch and roll are still refined with `NAVCAM_RIG_REFINE = "refine"`. In the bundle adjustment the rig quaternion's y component is held (at 0 for "zero"); only the Navcam rig is affected, not the Mastcam-Z rigs. With "hold" or "zero" the thermal stage no longer turns the rig by the yaw temperature slope (`thermal.rig_slopes_for_project`). `project.rig_without_yaw`, `rig_yaw_mdeg`; the removed yaw is recorded as `rig["N"]["yaw_removed_mdeg"]`.
- **`STORE_MASK_IN_ALPHA = 0.5`:** the masked pixels are half transparent (alpha 128 of 255; 16-bit 32768) instead of fully transparent; `True` / 1 is fully transparent as before, `False` / 0 writes no alpha. `process.alpha_transparency`, `MPPPImage.rgba(bits, transparency)`. `calibration.load_example` now reads only fully opaque alpha as "included". A changed value reprocesses the images (the processing settings change).
- Notebooks copied as `*_v0p51.ipynb`.

## 0.50.1 — 2026-09-30

GitHub: pull before push.
- `setup_github.bat` stopped at "Updates were rejected because the remote contains work that you do not have locally": `github.com/ctate7163/MPPP` already held commits. New **`scripts/windows/github_push.bat`** fetches GitHub first: nothing new -> push; commits of the same history -> `git merge` them, then push (stops on a conflict); an unrelated history (an earlier upload) -> you choose **K** keep it as the branch `github-before-v0p50` and make `main` this history (recommended), **M** merge it (`--allow-unrelated-histories -X ours`), or **Q** quit. It then writes `_transfer\mppp_pc.bundle` (this copy's history, for Claude's next session). `setup_github.bat` and `sync_from_claude.bat` call it; `sync_from_claude.bat` merges a delivery into a copy that has its own commits instead of refusing.

## 0.50.0 — 2026-09-30

Navcam distortion one set for all sols and temperatures, one camera-model folder, zcam34 sites, tests by module, git + GitHub.
- **Navcam distortion held (notebook 03 `NAVCAM_DISTORTION_FIT = "hold"`, default):** k1-k4, p1, p2 (and sx1, sy1) stay at the start (consensus) camera in every block and temperature bin - one distortion per eye for all sols and temperatures; only fx, fy, cx, cy are refined. `"refine"` fits it per block as before (then `TANGENTIAL` decides p1, p2). `mppp.sfm.project.navcam_distortion_terms`, `SfmProject.create(navcam_distortion_fit=, navcam_k4=, navcam_p1=)`; recorded in `project.settings` (a project made before is rebuilt once; features and matches are reused).
- **`NAVCAM_K4`, `NAVCAM_P1` = `"consensus"` (default) or `"zero"`:** the term is set to 0 and held, with either fit setting.
- **No distortion per temperature bin:** `thermal_stage` refuses `THERMAL_FREE` terms other than fx, fy, cx, cy.
- **Fix:** `TANGENTIAL = "zero"` did not zero p1, p2 of a fisheye + tangential (THIN_PRISM_FISHEYE) start camera; `camera_from_colmap_json` now zeroes them for any model. One parameter-name table (`project.PARAM_NAMES`).
- **Outlier frames:** the median-residual floor of the frame exclusion (`OUTLIER_DEFAULTS["min_residual_px"]`) 1.0 -> **1.2** native px.
- **Features:** notebook 03 `MAX_NUM_FEATURES` 16000 -> **16384** (`database.DEFAULT_MAX_NUM_FEATURES` 16380 -> 16384); features are extracted once more.
- **One camera-model folder: `src/mppp/data/cmods/`.** `params/` is retired; `params/cmods` (models in use), `data/m20_cmods` (flight Metashape calibrations, rational Navcam cameras) and `data/navcam_consensus` (a byte-identical copy) are one folder, files unchanged. The v0p22 consensus rig that `m20_cmods/M2020_N_rig.json` held is in `cmods/history/v0p22_consensus/`; the rational start cameras now pair with the current rig. `paths.cmods_dir()` = `MPPP_CMODS` or `data/cmods`; `paths.package_cmods_dir()`. `promote_cmods.py` writes there (commit afterwards). `M2020_occlusion_profiles.csv`, `M2020_taus_versus_L_s.csv` and `M20_waypoints.json` were already in `src/mppp/data` (identical to the `params/` copies).
- **Sites: `"zcam34": true/false`** per site in `sites.json` (Christian's `z34` field, as a JSON boolean) replaces the `nav_zcam34` group; Mastcam-Z WORK folders are **`<site>_colmap_zcam34`** (old `<site>_colmap_nav_zcam34` folders are used while no new one exists; `scripts/rename_zcam34_folders.py [--apply]` renames them and the paths in their project files). `run_sites.py` / `process_sites.py --zcam` take only the zcam34 sites. **`run_all_sites.bat` runs the Navcam blocks only; new `run_all_sites_zcam34.bat`** runs the zcam34 sites. `sites.json` is Christian's 30 Sep edit with the errors fixed (`sid` -> `sid_chal_rocks` in `navcam_consensus`, the `nav_z3434` search/replace typo). `check_sites` shows zcam34 and warns about sites with identical sol ranges (rockytop, rockytop_skinner, rockytop_wildcat select the same 139 images).
- **Runs that looked alive but did nothing:** on 30 Sep `run_all_sites.bat` sat at butler_landing "starting" for 47 min with a live heartbeat and no `log.txt` - most likely its console was paused by a QuickEdit selection (a click in the window), which blocks every print. The log now writes its file before the console, MPPP's runs switch QuickEdit off (`runner.disable_quickedit`), and `sites_status` flags a run "starting" for more than 10 min.
- **Tests by module:** the 30 `test_v0pNN.py` files and `test_units.py` / `test_sfm.py` are one file per module (`test_sfm_reconstruction.py`, `test_sfm_project.py`, `test_processing.py`, `test_runner.py`, `test_scripts.py`, ...), shared helpers in `tests/helpers.py`. The 19 tests that checked notebook text are gone; `test_notebooks_compile_and_have_a_parameters_cell` checks only that every code cell parses and the runner's tagged cell and section-3 heading exist.
- **Experiments moved to `studies/experiments/`:** `pair_experiment.py`, `lens_model_experiment.py`, `lens_terms_experiment.py`, `temperature_bins_experiment.py`, `error_sources.py`, `run_v0p22_batch.bat`.
- **Git and GitHub:** `scripts/windows/setup_github.bat` (once: makes `D:\code\MPPP` a git working copy of the delivered history, removes files MPPP no longer has, creates the private `github.com/ctate7163/MPPP` and pushes, offers to delete the old bundles) and `sync_from_claude.bat` (after each delivery: fast-forwards to `_transfer\mppp_latest.bundle`, removes deleted files, pushes). `.gitignore`: `_transfer/`, bundles, the notebook working copies `*_v0pNN.ipynb`, `data/`.
- Versions: 0.44 -> **0.50** (so that "v0p5" sorts after v0p44: `VERSION_TAG` v0p50). Notebooks copied as `*_v0p50.ipynb`.

## 0.44.0 — 2026-09-30

Navcam tiles below a quarter frame, thermal-stage defaults, the bin table with principal points, the LMST histogram.
- **Tiles:** `NAVCAM_MIN_FRAME_FRACTION` 0.5 -> **0.25**: `select_best_products` leaves out Navcam products covering less than a quarter of the detector frame (a quarter-frame product, exactly 1/4, is kept). At Van Zyl (sols 49-71) the v0p43 manifest held 473 such products of 564: full-resolution 1280 x 960 tiles (1/16, e.g. NCAM00603, NCAM00297), 1280 x 424 strips and quarter-resolution 1280 x 224 strips (0.23, NCAM00500-00504). `PROCESS_RULES = 3`: `process_sites.py` checks every site once more.
- **Thermal stage defaults:** notebook 03 `THERMAL_BINS_DEG` 10 -> **5** degC, `THERMAL_MIN_IMAGES` 8 -> **5** (`thermal.THERMAL_BIN_DEG`, `THERMAL_MIN_IMAGES`; `reconstruct(thermal_min_images=5)`). How bins form and merge is now in `thermal.temperature_bins`' docstring and the notebook comments: fixed multiples of the bin width; a frame by the mean temperature of its Navcam images; images of both eyes counted; the smallest bin under the minimum joins the neighbouring occupied bin whose centre is nearer (a tie goes to the colder one), repeated.
- **Principal points per bin:** only fx, fy are refined per temperature bin (and nothing when the Navcam intrinsics are held); cx, cy stay at the eye's refined values moved by the thermal model (NL cx +0.0517 px/degC). New notebook 03 setting `THERMAL_FREE = ("fx", "fy")` (add "cx", "cy" to fit them per bin). The thermal-stage table (`thermal.thermal_bin_lines`) now shows per bin camera: images, T median [min, max], fx, fy, cx, cy (* = held) and each one's change from its start; `split_by_temperature` rows carry `start`.
- **LMST histogram: `scripts/lmst_histogram.py`** - one panel per site plus all sites, Navcam and Mastcam-Z 34 each normalised to unit area, noon and the processing window marked, and `<name>_by_site.csv` (sols, images, stations, span, LMST min / median / max, Navcam LMST IQR). Reads the newest manifest of every WORK folder under `--roots` (default `D:/scapes/colmap`, `D:/scapes/colmap_old`; a folder name under several roots counts once), leaves out Navcam tiles below the selection rule, writes to `--out` (default `<MPPP>/Claude outputs`). Replaces the cloud-only `scripts/sites_table.py` figure. Run on 30 Sep: 21 sites, 2356 Navcam + 711 Mastcam-Z images.
- Notebooks copied as `*_v0p44.ipynb`.

## 0.43.3 — 2026-09-30

Stopping runs, one batch at a time, interrupted processing completed, Navcam tiles left out.
- **What happened (30 Sep):** `run_all_sites.bat` (process *and align* every site) and three `process_sites.bat` starts ran at the same time on `D:\scapes\colmap`, with no way to stop them from the .bat files. Sites being written by one run looked half-processed to the next.
- **`stop_mppp.bat` / `scripts/stop_runs.py`:** lists every MPPP run on the computer (the minimised `_run_*.bat` windows and `process_sites.py`, `run_sites.py`, `align_scape.py`, `run_scapes.py`, found by command line), asks, and stops each with everything it started (`taskkill /T /F`: notebook kernels, image workers). Notebooks open in Jupyter are not touched. The status files under the scapes folder are then marked stopped.
- **One batch per scapes folder:** `process_sites.py` and `run_sites.py` hold `<root>/mppp_batch.json` (pid, heartbeat, current site). A second start stops at once and says which batch runs (`--force-start` overrides). A site whose WORK folder has a run going on (e.g. `align_here.bat`) is skipped, not "FAILED". `sites_status.bat` shows the batch first.
- **A run whose process is gone is no longer "alive"** (`runner.pid_alive`): no 5-minute wait after a crash or a stop.
- **Interrupted processing is completed, not trimmed:** with `KEEP_ONLY_REMAINING` (`only_existing="PNG8"`), an image counts as removed only if it is in one of the folder's manifests (processed before) and its PNG is gone. Before, a run stopped half-way left a partial `images_png8`, and every later run processed only that part. The report now gives `never_processed`.
- **Navcam tiles left out:** `select_best_products(min_frame_fraction=0.5)` drops Navcam products that cover less than half of the detector frame, e.g. the 16 single full-resolution 1280 x 960 tiles of NCAM00698 on sol 92 (`NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01`, 1/16); the assembled full frame of the same acquisition (`..._000RAD...`) stays. The label is read only for files too small to be a full frame. `min_frame_fraction=None` turns it off; Mastcam-Z is not filtered. Notebook 03 (`_v0p43p3`) prints the count.
- **Checking the site definitions:** `scripts/windows/check_sites.bat` / `scripts/check_sites.py` checks `src/mppp/data/sites.json` after an edit: JSON errors with the line (and the line before), sol ranges, groups naming unknown sites, `no_mask_inference_at` stations, settings that are not notebook 03 settings, partly overlapping blocks. A broken file no longer breaks `import mppp` (a warning; notebook 03 then shows the JSON error).
- **Camera models in use: `params/cmods`** (`<MPPP>/params/cmods`, e.g. `D:\code\MPPP\params\cmods`; `MPPP_CMODS` overrides): the Navcam cameras and rig (`M2020_NL/NR_fisheye_tangential.json`, `M2020_N_rig.json`, the v0p41 joint) and the Mastcam-Z focus model (`M2020_ZCAM034_focus_model.json`, v0p42), with `README.md` and `CHANGES.md`. Notebook 03 starts from them by default (`NAVCAM_CAMERAS = NAVCAM_CONSENSUS_DIR` = `mppp.sfm.project.navcam_consensus_dir()`; `ZCAM_FOCUS_MODEL = None` = `zcam_focus_model_path()`), falling back to the copies in the package data. `scripts/promote_cmods.py <folder or files> --note ...` checks a new consensus, copies it in, keeps the replaced files in `history/<time>/` and logs it in `CHANGES.md`; `--list` shows what is in use.
- **Frame observation limit 30 -> 20, one setting:** frames with fewer tie-point observations are held at their prior in the bundle adjustment and excluded as outliers. Both used 30 separately; now both follow `reconstruction.MIN_FRAME_OBSERVATIONS` (20), set per run with notebook 03 `MIN_FRAME_OBSERVATIONS` (`reconstruct(min_frame_observations=)`, also for the thermal and focus-state stages) and recorded in `settings["reconstruction"]`. On 30 Sep the outlier exclusion found every bad frame but also dropped a few good, weakly tied ones.
- **`PROCESS_RULES = 2`** in the processing run key: `process_sites.py` checks every site again once (reusing processed images).

## 0.43.2 — 2026-09-30

Windows .bat files: finding Python, and running from any folder; the default Navcam cameras ship with MPPP.
- **Navcam consensus shipped:** `src/mppp/data/navcam_consensus/` holds the v0p41 joint cameras (byte-identical to `camera_analysis/navcal_v0p41/navcam_joint`), `mppp.sfm.project.NAVCAM_CONSENSUS_DIR`. Notebook 03 (`03_colmap_alignment_v0p43p2.ipynb`) uses it as the `NAVCAM_CAMERAS` default instead of `D:\scapes\colmap\camera_analysis\...`, which had moved to `colmap_old` on 30 Sep. A `NAVCAM_CAMERAS` folder without the camera files now stops with a clear message in the settings cell. Projects made with the old folder are rebuilt once (the folder is part of the project check); features and matches are reused.
- **Symptoms (30 Sep):** "MPPP: no Python with pycolmap, pyceres and nbclient was found", and `'"D:\scapes\colmap\mppp_env.bat"' is not recognized` when a .bat was copied into `D:\scapes\colmap`.
- **`mppp_env.bat` searches properly:** `MPPP_PYTHON` or `scripts\windows\mppp_python.txt` (one line: the notebooks' `python.exe`, from `import sys; print(sys.executable)` in a notebook); then `python` on PATH; then every environment in `%USERPROFILE%\.conda\environments.txt`; then `MPPP_CONDA` and the usual miniconda / anaconda / miniforge folders (`MPPP_ENV` first, base, then every `envs\*`). The chosen environment is put first on PATH as `conda activate` does (no `activate.bat` needed).
- **The test is `scripts\windows\check_env.py`:** pycolmap, pyceres, nbclient, nbformat and ipykernel must import, with `KMP_DUPLICATE_LIB_OK=TRUE` set as notebook 03 does (the earlier check imported pycolmap and pyceres without it, which can abort on the duplicate Intel OpenMP runtime).
- **When nothing passes,** the search runs again and prints every Python it tried and what it is missing, with the two fixes (`pip install nbclient ipykernel` into the notebooks' Python, or its path in `mppp_python.txt`).
- **`process_sites.bat`, `run_all_sites.bat` and `sites_status.bat` work from any folder:** inside MPPP they use their own folder; copied elsewhere they use `MPPP_HOME` (default `D:\code\MPPP`) for `mppp_env.bat` and the `_run_*.bat` helpers.

## 0.43.1 — 2026-09-30

Fix for staged Nav+Zcam runs; image processing for many sites without alignment.

**Processing only: `scripts/process_sites.py`, `scripts/windows/process_sites.bat`.**
- Selects and processes the images of `--all` sites, a `--group` or `--sites a b c` (`--zcam` for Nav+Zcam folders) into `<root>/<site>_colmap[_nav_zcam34]/processed`, one site after the other, and stops before the alignment.
- It runs sections 1–2 of the newest notebook 03 with its default settings (`runner.align(process_only=True)`, `run_notebook(stop_before="## 3")`), so it always uses the current code and defaults. The site's settings in `sites.json`, `<WORK>/mppp_settings.json`, `--settings` and `--set` apply as for an alignment.
- A finished site gets `processed/process_done.json` (run key, image counts); sites processed with the same settings are skipped (`--force` redoes them, still reusing unchanged images). A failed site is logged and the next one starts.
- Logs: `<root>/process_sites_log.txt`, `<WORK>/runs/<time>_process/log.txt`, `<WORK>/mppp_status.json` (so `align_here.bat` will not start while a folder is being processed). `--status` and `sites_status.bat` show "(processing)".
- Then `align_here.bat`, copied into a WORK folder, aligns those images with the newest notebook 03's defaults.
- The .bat files carry no version: they always run the current MPPP in `MPPP_HOME`.

**Fix: staged Nav+Zcam runs stopped at the start of stage 2.**
- **Symptom:** `ValueError: Check failed: ExistsCamera(sensor_id.id) Camera 28 from rig 27 not found in the reconstruction` in `reconstruct` → `restore_frames`, right after "before the final adjustment" of stage 1 (threeforks_south_colmap_nav_zcam34, 30 Sep). Nothing after `sparse/cahv_ba_navcam` is written.
- **Cause:** COLMAP's `tear_down` (inside `triangulate_points`) drops not only the deregistered Mastcam-Z frames but also the rigs and cameras no registered frame uses. With `ZCAM_RIG` on, the Mastcam-Z stereo rig and both its cameras were gone after stage 1. `restore_frames` added the rig before its cameras, and `AddRig` requires every camera of the rig to exist. The v0p40 test used a rig that survived stage 1, and the staged-reconstruct test replaced `triangulate` with a no-op, so neither saw it.
- **Fix:** `restore_frames` adds every camera of the frame's images and of its rig's sensors (from the start reconstruction) before the rig. Cameras still in the block keep their stage-1 values.
- **Tests:** `tests/test_v0p43p1.py`: a torn-down stereo rig is restored; refined cameras are not overwritten; staged `reconstruct` end to end with a two-camera Mastcam-Z rig and a `triangulate` that tears the block down.
- Notebooks unchanged (the `_v0p43` copies stay current).

## 0.43.0 — 2026-09-30

Runs without Jupyter, stable site definitions, and cheap reruns.
- **Why.** On 30 Sep the four Nav+Zcam notebook runs showed nothing on disk after their Navcam stage. The saved notebooks still held the outputs from when the runs started, and COLMAP writes nothing until the end of a run, so it could not be told whether they were running or had stopped.
- **Site definitions:** `src/mppp/data/sites.json`, one line per site: sols, label, note, `no_mask_inference_at`, `settings` (notebook 03 overrides for that site). It also has groups: `navcam_consensus` (the 23 joint blocks), `nav_zcam34`, `rerun_rational`.
  - `mppp.sfm.sites.SITES`, `load_sites`, `load_site_table`, `site_group`, `work_folder`, `parse_work_folder`.
  - Notebook 03 reads it: `SITES_FILE`, and `SITES_EXTRA` for sites of one run.
- **Notebook 03:**
  - `SOURCE = "processed"` aligns the images already in `WORK/processed` from its newest manifest: no PDS search, no image processing. Images deleted from `processed/images_png8` are left out.
  - `WORK_DIR`: any WORK folder. The site and `INCLUDE_ZCAM34` follow its name.
  - Exclusions with `WORK/exclude_images.txt` and `EXCLUDE`: stations `S032D1184`, `sol:658` or `sol:654-693`, `seq:NCAM08111`, or file-name patterns. An entry that selects no image is reported.
  - `VARIANT = "name"` writes to `WORK/colmap_name` and starts from `colmap/`'s features and matches (`mppp.sfm.workdir`).
  - `run_settings.json` and `run_done.json` (with `RUN_KEY`) are written in the project folder.
- **Match reuse:**
  - `build_database(reuse_matches_from=, matching=)` moves the old `database.db` aside and copies its matches into the new one, by image name (`database.reuse_matches`).
  - Matches are carried over only when `database_matches.json` (written by `match`) records the same features and matching settings. COLMAP then skips those pairs.
  - Verified two-view geometries are carried over only when both images' start cameras are unchanged. A camera-model change therefore re-verifies but does not re-match.
  - Tested: after removing images, matching took 0 s and gave identical matches; a camera-model variant re-verified only.
- **Headless runner** (`mppp.runner`):
  - `run_notebook` saves the executed notebook after every cell and logs every cell and every printed `[sfm]` line to `log.txt` as it happens.
  - `RunStatus` writes a status file with a 30 s heartbeat; a "running" status without a heartbeat for 5 min reads as stopped.
  - `align(work, source, variant, …)` puts runs in `WORK/runs/<time>[_variant]/`, the status in `WORK/mppp_status.json`, and a one-line history in `WORK/runs/runs.txt`. It refuses to start while another run of the folder is alive.
  - Settings are applied in this order, later winning: the site's `settings`, `WORK/mppp_settings.json`, `--settings FILE`, `--set NAME=VALUE`.
- **Scripts:**
  - `scripts/align_scape.py WORK` (default `--source processed`; `--variant`, `--set`, `--settings`, `--status`).
  - `scripts/run_sites.py`:
    - `--all`, `--group`, `--sites`, `--zcam`, `--source`, `--variant`, `--jobs`, `--then 04 05`, `--dry-run`, `--status`.
    - One `align_scape.py` process per site.
    - Skips sites finished with the same run key, and folders finished before 0.43; `--force` reruns them.
  - `scripts/run_scapes.py` now uses the shared runner.
- **Windows** (`scripts/windows/`):
  - `align_here.bat`: copy it into a WORK folder and double-click. It runs in a minimised window; double-click again (or `align_here.bat status`) for the status.
  - `run_all_sites.bat` (the settings are at its top) and `sites_status.bat`.
  - `mppp_env.bat` finds the conda environment with pycolmap, pyceres and nbclient (`MPPP_HOME`, `MPPP_CONDA`, `MPPP_ENV`).
- **Fix:** rerunning notebook 03 on an existing project after a thermal stage stopped with `KeyError: 'focus_count_range'` (the temperature-bin cameras of the last run were still in `project.json`). The project now drops them (`strip_thermal_bins`) when it is reused.
- Tests: `tests/test_v0p43.py`.
- Checked end to end in the cloud on 10 Navcam images of sol 1307, all through `run_sites.py` / `align_scape.py`:
  - PDS → process → align;
  - a rerun skipped;
  - a processed-source rerun with exclusions;
  - a rerun that reused every match;
  - a rational-camera variant.

## 0.42.0 — 2026-09-30

The Mastcam-Z focus model gets temperature and sol terms and a principal point that moves with focus. The data are the backlash-state focus bins (≥ 1000 observations) and 440 simultaneous stereo pairs of the Rockytop (sols 461–530), Three Forks (684–692) and Airey Hill (961–991) Navcam + Mastcam-Z blocks (MPPP 0.22). HEAD_FPA temperatures come from 86 Zcam labels, interpolated in spacecraft clock within each sol and eye. The shipped model is built by the library functions notebook 04 §5b uses. Study scripts and results: `docs/results/v0p42/zcam_focus_study/` (`build_model.py`).
- **Mastcam-Z focal length follows the block's Navcam scale.** In the 0.22 Three Forks block the Navcam focal lengths refined +0.33 % (NR) and +0.38 % (NL), and every Mastcam-Z bin sat +0.40 % above Rockytop. The Zcam bins inherit the Navcam angular scale through the shared points. Dividing each block's Zcam f by its Navcam refined/start ratio (Rockytop 1.00025, Three Forks 1.00351, Airey Hill 1.00003) brings Three Forks to within 4 px of Rockytop (`calibration.navcam_focal_scale`, `fit_focus_model(navcam_normalise=True)`).
- **Temperature: no measurable focal-length term.** HEAD_FPA spans −28 to −9 °C over the bins.
  - Without the Navcam normalisation, a pooled fit gives +1.6 to +2.3 px/°C. This is the Three Forks block effect (a warmer and "longer" block), not temperature.
  - Normalised, with the sol term: +0.15 ± 0.26 (ZL) and −0.10 ± 0.17 (ZR) px/°C.
  - Within Rockytop: +0.34 ± 0.22 and −0.01 ± 0.14 px/°C.
  - The model's thermal slope is 0, and the measurements are recorded in the model file.
- **Sol: a small positive trend, provisional.** ZL +0.025 ± 0.005 px/sol (+5 ppm/sol); ZR +0.005 ± 0.005 (not significant).
  - It is carried by Airey Hill, which sits +17.6 (ZL) and +8.6 (ZR) px above the Rockytop/Three Forks level (per-block fit).
  - With three blocks a sol trend and a block offset cannot be told apart, and the values depend on the weighting: capping the bin weights at 5000 observations gives +0.035 and +0.021 px/sol.
  - The model evaluates the trend at the sol clamped to 461–991 and does not extrapolate.
- **Focus slope refit** (weights = observations; close to linear: a quadratic term would add −0.2 ± 4 px (ZL) and +5 ± 2 px (ZR) at 1000 counts from focus 600):

  | eye | f at focus 600, sol 700 | slope (px/count) | was | label slope | f / label | rms |
  |---|---|---|---|---|---|---|
  | ZL034 | 4720.2 px | 0.0541 ± 0.0015 | 0.0463 | 0.0505 | 1.0086 | 2.6 px |
  | ZR034 | 4723.4 px | 0.0646 ± 0.0014 | 0.0482 | 0.0461 | 1.0088 | 2.7 px |

  The v0.21.1 model came from polynomial-Navcam solutions with the backlash and regular states mixed. Fitted range: ZL −400..1300, ZR −1100..1300 counts. Outside it, the bin's label f is scaled by the model's refined/label ratio, as before.
- **Principal point against focus.** With a ~20° × 15° field, a principal point trades with the pointing, so only the right-minus-left difference is observable. It is measured as the equivalent boresight, where the left principal point's ray lands in the right image: eqx = (cxR − cxL) + fR·yaw, eqy = (cyR − cyL) − fR·pitch (`calibration.zcam_boresight_table`, `fit_zcam_boresight`: Huber, per-block offsets, bootstrap errors).
  - eqx +2.49 ± 0.29 px and eqy +1.39 ± 0.34 px per 1000 counts: 3.2 and 1.8 px over focus 0–1300. Sigma clipping moved the eqy slope between 0.9 and 1.9, which is why the fit is now Huber.
  - Roll +20.5 ± 3.0 mdeg per 1000 counts.
  - No temperature term in the boresight: −0.020 ± 0.016 and 0.000 ± 0.016 px/°C.
  - Block levels: eqx 193.0 / 192.9 / 192.2 px, eqy 11.0 / 10.7 / 12.8 px, roll −624 / −635 / −634 mdeg.
  - The model puts the slope on ZR (ZL is the reference). Each ZR focus bin's cx, cy move from the eye's median label principal point by the slope times (bin focus − the eye's median focus in the block). The per-image label ZR principal point (−0.2 px/count in x) is an artifact that the label pointing compensates. It is not used.
  - With free per-image poses (no `ZCAM_RIG`) the shift is absorbed by the pointing. It makes the bins consistent with one another and with a Mastcam-Z rig.
- **Focus model file** (`m20_cmods/M2020_ZCAM034_focus_model.json`): per eye `thermal` (`T0_degC` −15, `sensor` HEAD_FPA, `f_px_per_degC`, measured values), `trend` (`sol0` 700, `f_px_per_sol`, `sol_range`), `pp` (`cx_px_per_count`, `cy_px_per_count`) and `block_offsets_px`; `stereo` (roll slope, block levels, temperature terms); the v0.21.1 values under `previous`.
  - `project.zcam_model_focal` and `zcam_model_pp_shift` evaluate the model.
  - `_split_by_focus` evaluates it at each bin's median focus, HEAD_FPA temperature and sol, and records `temperature_median_degC`, `sol_median`, `focus_model_terms_px` and `pp_shift_px` on the bin camera.
  - `backlash_ratios` is unchanged (f0 / label f0 at focus 600): 1.0086 / 1.0088.
- **Mastcam-Z temperature in the manifest.** `camera_temperature_degC` of a Zcam image is now its HEAD_FPA (`image.ZCAM_TEMPERATURE_KEY`). It is read from the label for reused manifests (`process.py`, `thermal.zcam_label_temperature`) and by `calibration.attach_pds_labels` (notebook 04 §2a). Navcam code filters on the family, so the Zcam values do not enter the Navcam thermal model.
- **Refit tools** (notebook 04 §5b; settings `FOCUS_THERMAL`, `FOCUS_TREND`, `FOCUS_NAVCAM_NORMALISE`):
  - `focus_table` rows carry `sol`, `temperature_degC` and `navcam_scale`.
  - `fit_focus_model(thermal=, trend=, navcam_normalise=, T0=, sol0=, reference_focus=)` returns standard errors and `aspect`.
  - `focus_model_json` writes a candidate model, which notebook 03 uses through `ZCAM_FOCUS_MODEL` (`SfmProject.create(zcam_focus_model_file=)`).
  - A project records the model's SHA-256 (`settings["zcam_focus_model"]`). Notebook 03 prints a note when an existing project was built with another model; its bins are rebuilt only with `REPROCESS_ALL`.
- Tests: `tests/test_v0p42.py`.

## 0.41.0 — 2026-09-30

The Navcam consensus with first-order temperature and sol terms, the lens-term test, and an exposure rule (working notes §19):
- **The rig's temperature term becomes a principal-point term.** Tests in the joint adjustment of 23 blocks, all from the same state, with the rig's mission drift applied per image (`navcal.thermal_keypoint_map`, `joint_adjust(pp_slopes=, drift=)`, `navcam_calibration_study.py joint --pp-thermal NL|NR|split --drift FILE`):

  | model | joint cost |
  |---|---|
  | no drift, no temperature term | 205,392 |
  | drift | 198,201 |
  | drift + rig yaw −1.018 mdeg/°C | 195,847 |
  | drift + NR cx slope | 195,864 |
  | drift + NL and NR split | 195,665 |
  | **drift + NL cx +0.0517 ± 0.0004 px/°C** | **195,596** |

  The NL principal point moving with its temperature fits best, with the same number of parameters (Δ cost −251 against the yaw slope). Putting the drift into the joint is itself worth −7,191.
- **First-order temperature and sol terms of the intrinsics** (`navcal.split_by_unit`: one fx, fy, cx, cy per block and sol epoch, distortion shared, with the model above applied, then regressed on camera temperature and sol).
  - Nothing remains in f: −4 ± 5 (NL) and −2 ± 5 (NR) ppm/°C, −6 ± 34 and −26 ± 35 ppm per 1000 sols. The 38.1 ppm/°C focal slope holds.
  - Nothing remains in cx with temperature: 0.000 ± 0.014 px/°C.
  - cy: −0.08 ± 0.04 px/°C in both eyes (p 0.05–0.07; the block-to-block scatter is 0.95 px). Not used.
  - cx over the mission: −0.41 and −0.44 ± 0.11 px per 1000 sols in both eyes (common mode). Imposed in the joint it *raised* the cost by 20 (the frame attitudes absorb a common shift), so it is left out.
- **Consensus v0p41** (`D:\scapes\colmap\camera_analysis\navcal_v0p41\navcam_joint`; notebook 03 `NAVCAM_CAMERAS`):
  - **Intrinsics:** fisheye + tangential at −20 °C; fx, fy +38.1 ppm/°C; NL cx +0.0517 px/°C (`thermal.cx_px_per_degC` in the camera file).
  - **Rig:** no temperature term; the mission drift only (pitch hinge at sol 300, yaw, roll).
  - **Pipeline:** `SfmProject.create`, the thermal-stage bins and `write_joint_cameras` carry the principal-point terms. `trend` (principal point per sol about sol0) is supported but not written.
- **k4, p1, p2 stay** (joint of 23 blocks, same model and state, `fixed_params`):
  - k4 = 0: cost +285, 0.07 px rms, 0.25–0.30 px in the corners, 1.3–1.5 px at the far corners.
  - p1 = p2 = 0: cost +42,066 (+21 %), 0.47–0.88 px rms.
  - All three at 0: +42,412.
- **Exposure rule** (`selection.max_exposure_ms` 40, `selection.max_centre_tint` 1.2; notebook 03 `MAX_EXPOSURE_MS`, `MAX_CENTRE_TINT`; `ExposureImage`).
  - The NCAM08111 sequence at the Three Forks depot (sols 654–693) brackets each view at ~2, ~18 and ~50–65 ms.
  - The long member has a dark blue disk in the centre. At sol 658, the frame labelled 54.1 ms has half the others' centre radiance and B/R 1.74× the edge (2 and 18 ms: 1.01, 1.00), as if it was exposed much shorter than labelled. Other Navcam frames there stay below 30 ms.
  - Frames above either limit are skipped and listed under `skipped`. A manifest entry is reused unless it falls under the rules; a change of the limits does not reprocess the other images. `centre_tint` is recorded in the manifest.
- **Matching:** `prior_pairs` keeps 59–65 % of the exhaustive pairs on Navcam-only blocks and 23 % on Three Forks North with Mastcam-Z (480 images).
- Tests: `tests/test_v0p41.py`.

## 0.40.0 — 2026-09-29

Closing out the Navcam analysis: site discovery, a new consensus from 23 blocks, rig epochs, and the first Mastcam-Z step, the focus backlash states (working notes §18):
- **Site discovery for notebooks 04 and 05** (`mppp.sfm.sites`: `SITES`, `scan_scapes`, `discover_scapes`, `site_label`, `check_sites`): with `SCAPES = None` (04) / `ALIGNMENTS = None` (05), every `<site>_colmap` (and, with `INCLUDE_ZCAM34`, `<site>_colmap_nav_zcam34`) folder below `SCAPES_ROOT` with a finished alignment is used, in sol order. The table printed with it gives the reason for each folder left out: no project.json; a run in progress, whose project.json has no reconstruction yet; model missing; no `error_input` (05); fewer than 10 images; or no image in the site's sol range. `SCAPE_EXCLUDE` leaves out named sites.
- **Notebook 03 defaults** are the settings block of 29 Sep. `SITES` has 32 sites, including enchanted_lake, rose_river_falls, knob_mountain, berea, hippo_pools, threeforks_north/south, taylorfjellet_large and origny_large, and the markdown table is regenerated from it. `ATTITUDE_PRIOR_DEG = 2`, `LOCALIZE_MIN_IMAGES = 3`, `NAVCAM_INTRINSICS = "auto"`, `NAVCAM_RIG_REFINE = "refine"`. A reversed sol range stops the notebook.
- **Fix: `NAVCAM_RIG_REFINE = "refine"` held the rig.** Only "rotation" and "auto" reached `reconstruct`; "refine" became False. It now maps to "rotation", and an unknown value is an error. Sid and Origny were run with the rig held on strong networks.
- **`ZCAM_TANGENTIAL`** (notebook 03, default "zero"): Mastcam-Z p1, p2 are set separately from the Navcam's `TANGENTIAL`. The choices are "refine", "xml", "zero", or None to follow `TANGENTIAL`. `SfmProject.create(zcam_zero_terms=...)`, `bundle_adjust` / `reconstruct(refine_tangential={"N": bool, "Z": bool})`, `reconstruction.tangential_for`.
- **Mastcam-Z focus backlash states** (`mppp.sfm.backlash`; notebook 03 `ZCAM_BACKLASH = "split"`; `reconstruct(zcam_backlash=...)`). At the same focus count the focus mechanism is either in the dominant backlash state (f about 1 % above CAHVOR; the focus model's state) or in the regular state (f about CAHVOR; mostly single frames and small mosaics). After the alignment each Mastcam-Z image's focal length is measured with its rotation, focal scale and centre free, the centre held by a 5 cm prior at its label position moved by the station's Navcam shift, and the points held. Images of one focus setting (eye, sol, sequence, focus count) are classified together against the midpoint between the two states. Regular groups move to their own camera per focus bin (`<bin>_reg`, start f = label; f held for ≤ `ZCAM_HOLD_F_IMAGES` images), and the block is adjusted again with the Navcam held. Unclear groups (within 3 σ of the midpoint) stay in the backlash state, as do implausible ones (more than 0.6 % outside both states). Reported in `project.settings["zcam_backlash"]` and `colmap/zcam_focus_states.json`. The focus-breathing plot and fit (`zcam.py`, `calibration.fit_focus_model(state="backlash")`) keep the regular cameras apart. `strip_thermal_bins` also undoes the split before a rerun.
- **Fix: staged Navcam + Mastcam-Z runs stopped in stage 2** with `KeyError` in `reconstruct` (`rec.frames[fid]`). pycolmap 4 drops deregistered frames and their images when a model is written and read back, and `triangulate_points` does this, so the Mastcam-Z frames set aside in stage 1 were gone. They are now put back from the initial reconstruction (`reconstruction.restore_frames`).
- **Notebook 03 after a kernel restart:** the health and focus-breathing cells read the saved alignment when `rec` is not defined (`reconstruction.load_alignment`). They say so when section 6 has not finished; before, this raised `NameError: name 'rec' is not defined`.
- **Navcam consensus v0p40** (`D:\scapes\colmap\camera_analysis\navcal_v0p40\navcam_joint`, and `navcam_joint_rational`; notebook 03 `NAVCAM_CAMERAS` now points here):
  - Joint calibration of 23 blocks (Butler Landing to Marble Mountain, 8,000 points per block, 0.80 M observations).
  - Focal slope **38.1 ± 0.2 ppm/°C** (v0p35: 38.6). Rig yaw slope **−1.018 ± 0.007 mdeg/°C** (within blocks −1.048).
  - Cameras and rig are written at the **reference temperature −20 °C** (`write_joint_cameras(T_ref=-20)`, `NAVCAM_T_REF_DEGC`; `navcam_calibration_study.py joint --t0`, default −20).
  - Against the v0p35.1 cameras at −20 °C: 0.48 px (NL) and 0.62 px (NR) rms over the frame. Most of this is a common principal-point shift of +0.5 px in x, which trades with the rig yaw (−3.6 mdeg).
  - The joint of the 13 blocks aligned with the fisheye cameras differs from the full one by 0.28–0.36 px rms.
  - Fisheye + tangential still fits better than rational (cost 203,054 vs 204,040).
  - Drift, from 22 blocks with the multi-epoch blocks split (25 entries; Whale Mountain left out): pitch +4.9 ± 0.4 mdeg/1000 sol after sol 300 and +22.5 before; roll −5.5 ± 1.0; yaw +2.6 ± 1.4 (p 0.06).
  - Leave-one-block-out rig prediction rms: pitch 1.1, yaw 4.8, roll 3.3 mdeg.
- **Early Navcam products without a temperature** (notebook 04 §2a showed NaN before about sol 200): their label camera model is not interpolated to temperature (`INTERPOLATION_METHOD = NONE`), so there was no interpolation value to read. `MPPPImage.camera_temperature_degC` now falls back to the eye's `NAVCAM_LEFT_CAL` / `NAVCAM_RIGHT_CAL` label temperature. Rerunning notebook 01 with reuse fills the missing values in existing manifests from the labels, without reprocessing (`process.py`: also where the value is None).
- **Rig epochs** (`navcal.sol_epochs`, `split_rig_by_epoch`, `expand_epochs`, rig-study mode `epochs`, `navcam_calibration_study.py rig --modes epochs`): nearby waypoints can bring images from another part of the mission into a block. Sid has sols 91–101 and 360–371, on either side of the pitch knee. The block is then solved with one rig per group of sols more than 30 sols apart (cameras shared, common principal points), and each epoch with at least 6 stereo frames enters the drift fits at its own sol and temperature (notebook 04 §2f `EPOCHS = True`). Sid's two epochs differ by +5.0 mdeg in pitch (the hinge drift predicts +5.5) and −14.3 mdeg in yaw (the thermal term for their 13.5 °C difference predicts −14).
- **Drift figure** (`navcal_report.drift_figure`): draws an older hinge model as a hinge; `labels=` sets the legend.
- Tests: `tests/test_v0p40.py`.

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
