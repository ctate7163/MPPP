# Changelog

The development history before 0.90 is in `docs/history/` (`CHANGELOG_v0p1-v0p12.md`, `CHANGELOG_v0p13-v0p85.md`).

## 0.91.0 — 2026-10-20 (in development)

The joint as one bundle adjustment, the rig fitted for yaw and roll, Mastcam-Z p1, p2, the site maps and the CLI.

- **Navcam joint in one adjustment** (`mppp.calib.tied`): every term of the frozen form is a parameter of one
  bundle adjustment over the blocks - the intrinsics at −20 °C with one slope each in temperature (fx, fy, cx, cy,
  k1 per eye), the distortion (k2, p1, p2, k3, k4), the early-mission offsets *per eye*, and the rig as its rotation
  at sol 0 with linear pitch, yaw and roll rates and the early pitch hinge - tied to the block cameras and rig
  epochs by constraint residuals, with the poses and points; one normal matrix gives the covariance of the whole
  set (finite sd: the v0p83 prototype's were not).  Afterwards a diagnostic pass holds the solution and frees one
  camera per block, eye, temperature bin and era and one rig per block and epoch: those per-block, per-bin and
  per-sol values are what the refined figures draw about the global law, and their agreement is the promotion
  gate (`block agreement` replaces the leave-one-block-out).  `navcam_calibration_study.py consensus` runs it by
  default (`--joint profiles` gives the 0.90 joint); `mppp calib navcam` runs it from the shell.  The candidate
  files carry `early_mission` per eye, `drift` with `sol0 = 0`, `knot_ref = 0`, the three rates and the hinge.
- **Block runs**: the rig roll is held at the rig file's trend like the yaw (`navcam_rig_roll`, per epoch it was
  ±3 mdeg noise); a rig epoch with fewer than 16 Navcam images or a single station joins the nearest epoch
  (Rochette's sol-341 station).  Every site aligns once more (the run key changed).
- **Mastcam-Z `zcam-v2`**: p1, p2 refined per eye and constant over focus (`zcam_tangential = "refine"`; on the
  focus lines as constants; in the focus model's `distortion` with their sd and block scatter; checked against
  `max_p` 5e-3 and the scatter); `zcam-v1` files still load.  `zcam34_consensus` has 11 blocks.
- **Site maps**: the 20 Oct figure code (`studies/m20_sites`: blocks from `sites.json`, 1 m DTM and orthomosaic
  backgrounds, the table sized to the block count, `M20_TAG`) and `scripts\windows\make_maps.bat` /
  `mppp maps`; the `[maps]` extra (adjustText, rasterio, pyproj).
- **CLI**: `mppp maps`, `mppp calib navcam`, `mppp doctor`; LightGlue logs its progress.

## 0.90.0 — 2026-10-19

The beta-test and JOSS release: the package is reduced to what runs the pipeline from PDS products to an
aligned COLMAP block, with the calibration studies moved out of it.

- **Layout**: `mppp.calib` (the frozen camera-model forms, their loading, constraints and fits), `mppp.waypoints`
  (waypoints, sites and blocks; absorbs `mppp.sfm.sites` and `mppp.maps`), `mppp.pipeline` + the `mppp` command
  (the notebook-03 workflow as an importable function and a CLI); the six calibration modules, the map prototype
  and the one-off scripts are in `studies/`; `src/legacy` is gone (its regression test compares against stored
  fixtures in `tests/data/`).
- **Camera models**: one form per camera family - Navcam and Hazcam `THIN_PRISM_FISHEYE` (sx1, sy1 held) with
  fx, fy, cx, cy, k1 linear in temperature and the stereo-rig rotation linear in sol; Mastcam-Z per zoom, a
  function of focus and backlash state, p1 = p2 = k3 = 0.  The polynomial and rational Navcam forms, the Metashape
  XML flight calibrations and the model history are removed from the package data.
- **Pipeline**: `mppp.pipeline.Settings` holds every setting of a run (the notebook's parameters cell, with the
  author-machine paths read from `MPPP_PDS`, `MPPP_SCAPES`, `MPPP_GPU_PYTHON` and `COLMAP_BAT`);
  `select_images`, `process_site`, `align` and `run_site` are the former notebook sections; `run_key` names a run by
  its settings, the camera models in use and the alignment rules.  `mppp process|align|run|sites|cmods|mask|version`
  replaces running the notebook headless (`nbclient` is no longer needed); `mppp.runner.run_logged` keeps the log
  file, status file and heartbeat of a run.  The notebooks are renumbered (`01_process_images`,
  `02_colmap_alignment`, `03_error_analysis`, `training/train_mask`) and 01 and 02 are thin callers that display
  the results.  Settings files and `--set` take the field names in either case (`attitude_prior_deg` or
  `ATTITUDE_PRIOR_DEG`); run keys of runs made before 0.90 no longer match (every site aligns once more).
- **Matching**: SIFT + LightGlue by default (`match(mode="lightglue")`); COLMAP's nearest-neighbour matcher remains
  available.
- **Removed**: dead functions and options, compatibility branches, the `v0pNN` change annotations in the source,
  author-machine paths and the Claude-sync workflow files.
- **Tests** by module; a 16-bit radiometric traceability test (PNG16 linear in PDS RAD radiance).
