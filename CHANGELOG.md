# Changelog

The development history before 0.90 is in `docs/history/` (`CHANGELOG_v0p1-v0p12.md`, `CHANGELOG_v0p13-v0p85.md`).

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
