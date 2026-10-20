# Changelog

The development history before 0.90 is in `docs/history/` (`CHANGELOG_v0p1-v0p12.md`, `CHANGELOG_v0p13-v0p85.md`).

## 0.91.0 — 2026-10-20 (in development)

The joint as one bundle adjustment, the rig fitted for yaw and roll, Mastcam-Z p1, p2, the site maps and the CLI.

- **20 Oct, evening - the Navcam models in use are the tied joint `navcal_tied_20261020_2`** (22 blocks, every
  check passed; `src/mppp/data/cmods/CHANGES.md`).  `write_tied` now lists `early_mission.applied_terms` (fx, fy,
  cx, cy of each eye): without it the projects dropped the early-mission offsets.  Gate of the tied joint from the
  first real run: block agreement fx 250 ppm, fy 500 ppm, rig 8 mdeg, at most 10 % of the units off; the thermal
  f slope change against the files in use is reported, not gated.  Next joints: one focal slope per eye (fy's
  ppm/°C tied to fx's, `ISO_F`, `--aniso-f` frees it; 20 Oct: fy − fx +1.1 ± 1.4, −0.25 ± 1.4 ppm/°C), the k1(T)
  significance line (z −10 / −5.8), rig epochs of up to 50 sols (was 25), the pitch transition tested at 25
  knots over sols 200-400.
- **Mastcam-Z frames need 10 valid Navcam ties**: a Navcam-Mastcam-Z point needs a third look (0.86, unchanged);
  a Mastcam-Z frame with fewer than `ZCAM_MIN_NAVCAM_TIES` (10) such points is excluded before the stage-2 final
  adjustment (after the pose-guided matching).  `mppp.sfm.health.zcam_terrain`: the Mastcam-Z points (and the
  Navcam-Mastcam-Z ties) above and below the Navcam-only terrain (0.5 m grid, 0.15 m + 1 % of range) - logged in
  every stage-2 round and after the pose-guided matching, and in the health report.
- **Figures**: `make_figures.bat` (the standard set at 200 dpi: Navcam rig trends with pitch −5..20 and roll
  −15..15 mdeg, intrinsics vs temperature with the joint's cy slopes drawn, block table, LMST histograms, station
  maps, the Mastcam-Z rig trends and focus lines, the site map = figure 2 of the maps); `m20_dem_env.bat` shared
  with `make_maps.bat` (the TRN orthomosaic first).  Cluster zooms on the orthomosaic in colour with one stretch for
  all panels (a grey mosaic tinted by the DTM height); figure 3's regional titles as figure 6's; Three Forks
  North's sol 413 left out of the figures' ranges.  `zr_diagnose_z34.bat` removed.
- **Sites**: plateau_point and vingeanne in navcam_consensus (26); olifants in zcam34_consensus (12).
- **Mastcam-Z joint** (`mppp.calib.ztied`, `mppp calib zcam`, `zcam_tied_joint.bat`): one bundle adjustment over
  the finished Navcam + Mastcam-Z blocks of a zoom with the Navcam held at the models in use (the promoted law per
  temperature bin and era, the rig on its sol trend) and the zcam-v2 focus model fitted: every block's focus-bin
  camera tied to fx, fy = a[state] + b (focus − ref) (one slope per eye), k1, k2, p1, p2 per eye, ZR's cy with its
  focus slope, one Mastcam-Z rig (rotation refined, CAHV baseline held); a diagnostic pass frees each block's bins
  and rig - the per-block focus lines and the rig's sol / lens-temperature trends.  Writes the candidate focus
  model, its check against the one in use (`check_zcam`), `report.md`, the figures; `--promote-if-ok`.
  `merge_scapes(include_zcam=True)`, `tied_reconstruction(passthrough=True)`, `load_scape(single=False)`;
  `bundle_adjust` drops the residual-less pose blocks of right Mastcam-Z frames posed through the rig (the
  covariance was rank-deficient).
- **Cluster zooms**: only the block's own waypoints (its primary sols plus every waypoint within 10 m, the site's
  NEARBY_FROM_FIRST_SOL respected) in colour and in the N× / Z× counts; the other waypoints in the square in grey;
  thicker 10 m circles.
- **claude_sync.bat**: the paper is no longer rendered after every sync; uncommitted edits are stashed before the
  merge and popped right after it (they were left in the stash - paper.md, 20 Oct).  The bundles carry
  `refs/heads/main` again.

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
- **CLI**: `mppp maps`, `mppp calib navcam`, `mppp doctor`; LightGlue logs its progress (and is off by default
  until the A/B test: `match_mode = "exhaustive"`).
- **One folder of batch files**: every `.bat` is in `scripts\windows` (the calibration study's moved from
  `studies\calibration\windows`; the two forwarders and the v0p22 experiment batch are gone).

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
