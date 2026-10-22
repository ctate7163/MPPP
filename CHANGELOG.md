# Changelog

The development history before 0.90 is in `docs/history/` (`CHANGELOG_v0p1-v0p12.md`, `CHANGELOG_v0p13-v0p85.md`).

## 0.93.0 — 2026-10-22

- **Front and rear Hazcams in the alignment** (`hazcams` setting; `mppp process|align|run --hazcam front|rear|both`,
  `run_sites.py --hazcam ...`, or `--set HAZCAMS=F`).  The selection adds the `FLF` / `FRF` (`FHAZ` sequences) and / or
  `RLF` / `RRF` (`RHAZ`) products of the block's stations (the same nearby-waypoint rule as Mastcam-Z; the 1/3
  sub-frame rule holds for them too).  The run goes to `<WORK>/colmap_hazcam` unless a variant is named, so
  `colmap/` keeps the Navcam (+ Mastcam-Z) result; a run without `hazcams` leaves out Hazcam images found in
  `processed/`.  `FLF`, `FRF`, `RLF`, `RRF` are in the processing and project scope.
- **The Hazcam frames come in at stage 2**, with the Mastcam-Z frames: the Navcam cameras and rig are held (stage 1 is
  Navcam only, as before), the Hazcam cameras are held in the first stage-2 round and then refine fx, fy, k1, k2
  (`hazcam_intrinsics`: "radial" default, "focal", "all", "hold"); the stereo rig of each pair stays at its CAHV
  value (`hazcam_rig_refine`: "hold" default, or "rotation").  Each round logs the Hazcam frames with tie points and
  the Navcam-Hazcam tie points (`settings["reconstruction"]["hazcam"]`, `mppp.sfm.reconstruction.hazcam_coverage`).
  Prior-overlap pairs put the Hazcam ground 0.8 m below the camera (`pairs.CAMERA_HEIGHT_BY_FAMILY_M`), not the
  mast's 1.95 m.
- **Hazcam frames ride on their station** (stage 2): a Hazcam camera is fixed to the rover body, so its position
  prior moves by the shift the Navcam block found for its station (the rover's localisation error) and tightens to
  0.1 m per axis (`HAZCAM_STATION_PRIOR_SIGMA_M`; the label prior stays in `prior_C_label`).  A Hazcam eye with
  fewer than 1000 tie-point observations is held at its start camera (`HAZCAM_MIN_REFINE_OBSERVATIONS`), and the
  Hazcam eyes keep their pixel aspect fy / fx like the Navcam (`navcam_aspect = "hold"`).  The corner-coverage health
  check leaves the Hazcams out (their 136-degree corners are sky, horizon and rover deck).
- **Tested end to end on the Rockytop example set** (sols 461 and 464, 32 Navcam images, the FHAZ02008 and
  RHAZ02419 pairs; CPU, 5 min): all 40 images registered.  Front Hazcams: 9500 observations per eye, median
  residual 0.29-0.33 px, f refined by 0.3-0.4 %, frames within 1-3 cm and 0.1 deg of their station-shifted priors.
  Rear Hazcams: 150-250 observations per eye (quarter-resolution frames, about half of each masked as rover
  shadow), held at the label fits, median residual 1.2-2.9 px full frame (0.3-0.7 px in the image), within 5 cm.
  Before the station priors and the weak-eye hold, the rear eyes refined fx by 2.3 % and fy by 4.6 % and one rear
  pair slid 1.0-1.3 m from its station and was excluded as an outlier.
- **Hazcam start cameras from the labels** (`mppp.calib.labelfit`, new): the `THIN_PRISM_FISHEYE` fitted by least
  squares to each product's own label CAHVORE over the whole full frame, in the label model's camera frame (the pose
  prior is unchanged).  Sol 461 labels: front Hazcams 0.13 / 0.19 px rms (the closed form f (1 + r0), r1, r2: 1.4 /
  2.5 px - it ignores the 0.05-0.1 deg O tilt); rear Hazcams 2.3 / 3.9 px rms over 2, 5 and 20 m (closed form 6.0 /
  6.5 px) - their CAHVOREs carry entrance-pupil terms of 3-9 mm, which no central camera follows.  The processed
  image's COLMAP prior camera is this fit; a block starts each eye from the median of its images' fits
  (`labelfit.block_camera`, `start_cameras.hazcam_start_camera`).  `calib.label_camera` (the fallback for any
  Navcam / Hazcam product without a promoted model) uses the fit too.  Baselines from the CAHV pairs: front 0.249 m,
  rear 0.934 m.
- **Fixed: the generic Hazcam start model** (`calib.hazcam_start_model`, used now only without a label model) had
  f = 2130 px, k1 = -0.25, k2 = +0.08 - an OpenCV-pinhole guess on the fisheye theta polynomial, 330 px rms (550 px in
  the corners) from the flight calibration.  It is f = 2148 px, k1 = +0.0145, k2 = -0.0083 (the mean of the four sol
  461 label fits).  Any Hazcam image processed before 0.93 has the old prior camera in its manifest; the COLMAP
  project does not use it (the block start comes from the labels).
- **Run keys unchanged**: the Hazcam settings enter `Settings.key_dict` only when `hazcams` is set, so every Navcam and
  Mastcam-Z block keeps its run key (checked against 0.92.3: default, Z34 and variant keys identical) and nothing
  realigns.  `ALIGN_RULES` and `PROCESS_RULES` are unchanged.
- `run_sites.py` checks a block's finished state in the run's own variant (a Hazcam run's `colmap_hazcam`).
- Stage 2 without Mastcam-Z frames no longer logs empty Mastcam-Z coverage, terrain and focus-line lines.
- Tests: `tests/test_hazcam.py` (label fits on the sol 461 FLF / RLF models, start cameras, settings and run keys,
  selection filter, CLI, the staged alignment with a Hazcam rig held and refined, and with the Hazcam cameras held).

## 0.92.3 — 2026-10-22

- **Figures** (Christian, 22 Oct): every subplot of the Navcam and Mastcam-Z figures is lettered A, B, C ...
  (`constrain.panel_letters`: rig trends, intrinsics vs temperature, Mastcam-Z rig trends; the focus lines in
  `studies/calibration/zcam_lines.py`).  Navcam rig trends: pitch -10 to +30, yaw and roll -20 to +20 mdeg
  (`RIG_TREND_YLIMS`), the stereo cx panel's primary axis -20 to -17 px (`STEREO_CX_YLIM`); points outside an axis
  are counted on the panel.  `make_figures` also writes `navcam_rig_trends_<tag>_nolabels.png` and
  `zcam_rig_trends_<tag>_nolabels.png` (no block name at the points; `point_labels=False`).  Rig trends layout: pitch (A), roll (B, upper right), yaw (C), stereo cx (D); the sol axes read "sol epoch".
- **Sites** (unofficial blocks, Christian 22 Oct): `tatarka` (sols 1690-1697, S081D3608) and `crocodile_bridge`
  (1704-1706, S082D2786-D2866), both with a Mastcam-Z 34 mm 360 (`zcam: [34]`); `seitah_huge` (201-340 without the
  sol 340 waypoint S009D1064, site setting `EXCLUDE`; 20 waypoints, 193 m); `delta_front_huge` (408-708, 127
  waypoints over 843 m, Navcam only); `turtle_rock` (1280-1282, 4 waypoints, 14 m) and `curtis_ridge` (1298-1302,
  6 waypoints, 28 m); `bessieres_island` (1018-1023, Mastcam-Z 34 mm 360);
  `la_orotava_crater` (384-385, 2 waypoints, 14 m).  None of these is in a group yet.  Pico Turquino joins `navcam_consensus` (28 blocks;
  Hippo Pools was already in it).
- **More cold points in the intrinsics figure.**  No bin camera below -30 °C was drawn: the cold morning images
  are spread 1-3 per 2 °C bin of a block, and the figure took bins of >= 8 images (2 of 106 cold bin cameras).
  The figure now takes bins of >= 4 images (`tied.FIG_BIN_MIN_IMAGES`; `make_figures` re-cuts them from the
  study's `navcam_joint/tied_joint.json`, so the promoted 20 Oct joint gives 263 bins, 24 below -30 °C, down to
  -47 °C, without a rerun), and the next joint's diagnostic pass merges a block's adjacent thin 2 °C bins until each
  eye has 8 images, at most 10 °C wide (`tied.merge_bins`, `DIAG_BIN_MIN_IMAGES`, `DIAG_BIN_MAX_DEG`; simulated on
  the 20 Oct joint: about 42 bins below -30 °C).  The global fit's bins are unchanged.
- **Navcam rig trends layout:** pitch (A), roll (B, upper right), yaw (C), stereo cx (D); the sol axes read
  "sol epoch".
- **New unofficial blocks in `sites.json`** (Christian, 22 Oct): `tatarka` (sols 1690-1697, S081D3608) and
  `crocodile_bridge` (sols 1704-1706, S082D2786-D2866), both with a Mastcam-Z 34 mm 360 (`zcam: [34]`);
  `seitah_huge` (sols 201-340 without the waypoint S009D1064: site setting `EXCLUDE`); `delta_front_huge` (sols
  408-708, 127 waypoints, 843 m across: prior-overlap pairs).  None is in a group yet.

## 0.92.2 — 2026-10-22

- **One shipped exclusion list instead of movie filters** (`src/mppp/data/exclude_images.txt`, read by
  `pipeline.shipped_exclusions`; on by default, `EXCLUDE_LIST=false` keeps the images; not in the run key): the
  Mastcam-Z movies ZCAM05179, ZCAM05180 (Bunsen Peak), ZCAM05114 (Rockytop), ZCAM05093 (Issole), ZCAM05062; every
  ZCAM01* sequence; the Navcam stereo pairs 0693_0728460284_145, 0684_0727665222_395, 0693_0728460255_160 and
  0681_0727400691_868 (Three Forks North); ZCAM00* (but Chal Rocks' ZCAM00033 and ZCAM00034 are kept: an entry
  starting `!` always keeps its images) and ZCAM08512 (Rockytop).  The fixed-pointing movie filter and the sites' movie `EXCLUDE` entries of
  0.92.1 are gone.  `seq:` entries take wildcards (`seq:ZCAM01*`).  The log names each entry that removed images.
- **COLMAP GUI files by default**: `open_in_colmap.bat`, `open_in_colmap_fullres.bat` and `colmap_gui.ini` are written
  into every project (`gui_files` on; it was only `--gui`, so batch runs had none).  `mppp gui [--root DIR | WORK ...]`
  writes them for projects already aligned.
- **Prior-overlap pairs from 300 images** (`max_exhaustive_images` 300, was 1000; not in the run key).
- Because neither is in the run key, finished blocks the list touches keep their results until aligned again
  (`--force`): Three Forks North, Belva Crater (low-backlash list), Rockytop, Bunsen Peak; Issole is not finished.

## 0.92.1 — 2026-10-22

- **Mastcam-Z movies are left out by their fixed pointing** (`pipeline.drop_zcam_movies`; `zcam_movie_min_frames` 20,
  `zcam_movie_pointing_deg` 0.05, not in the run key): 20 or more frames of one sequence and eye whose label
  boresights lie within 0.05 deg.  The Z34 run of 21-22 Oct had four: Issole ZCAM05093 (~2000 frames), Bunsen Peak
  ZCAM05179 (1088 frames at 2 pointings) and ZCAM05180 (416 at one), Rockytop ZCAM05114 (144 at one) - one eye, no
  tie points, hours of matching, and the health check FAILed (Bunsen: 576 images without tie points; Rockytop: 21 %
  excluded).  A mosaic has one frame per eye per pointing (checked on four blocks) and stays.  Rockytop and Bunsen
  Peak also exclude their movies by name (`EXCLUDE` in sites.json), which changes their run key: they run again.
- **Low-backlash lists corrected**: the left-eye images of Three Forks North's ZCAM07114 (sols 413, 679-693), ZCAM08692
  (sol 690) and ZCAM05086 (sol 692), and Belva Crater's ZCAM08808 (sol 789, both eyes) are taken off (commented out,
  with the reason) - their "regular" bins refined 1.0-1.2 % in f, the backlash state.  8 entries in the shipped list,
  98 in the PC's `zcam_low_backlash_found.csv`.  Those two blocks must be aligned again for it to count.
- **The Mastcam-Z joint on the good blocks**: group `zcam34_joint` (Chal Rocks, Rochette, Enchanted Lake, South Arm,
  Taylorfjellet, Olifants, Marble Mountain) is the joints' default (`calib zcam` / `calib nz`, so
  `zcam_tied_joint.bat`), and a block whose alignment health is FAIL is left out unless `--include-failed`.
- `run_multizoom_test.bat`: Taylorfjellet with all five Mastcam-Z zooms in one alignment (variant `multizoom`).

## 0.92.0 — 2026-10-21

The combined Navcam + Mastcam-Z check, the zoom tie ladder, k1/k2 tests, the documentation site and the Zenodo setup.

- **Combined Navcam + Mastcam-Z joint** (`mppp.calib.nzjoint`, `mppp calib nz`): from the merged finished blocks, the
  Mastcam-Z joint with the Navcam held (the reference), then one adjustment with both laws free (Navcam bins tied to
  the Navcam law, Mastcam-Z bins to the focus law, both rigs refined), and its comparison with the Navcam-only law:
  each Navcam term's change at the reference temperature (px, ppm, in sd), the thermal slopes, the Navcam and
  Mastcam-Z residuals before and after.  "Agrees" when no term moves beyond 3 sd, 100 ppm (f) or 0.3 px (cx, cy) and
  the Navcam residual median rises less than 5 %.  A check, never promoted; `zcam_tied_joint.bat` runs it last
  (`nz34_joint_<date>/report.md`; `set MPPP_NZ=none` skips it), so `run_sites_zcam34.bat` ends with it.
- **The tie rule by zoom**: 34 and 48 mm frames need 10 valid tie points with Navcam (as before); 63-110 mm frames need
  10 with Navcam or with a Mastcam-Z image at a wider zoom (`reconstruction.navcam_tie_counts`,
  `ZCAM_NAVCAM_ONLY_MAX_ZOOM = 48`): a long-zoom frame ties through 34 or 48 mm, whose pixels are close to its own.
- **k1, k2 of Mastcam-Z as curves, with tests** (`ztied.distortion_curves`, `distortion_tests`): each eye's radial
  displacement at 1/4, 1/2, 3/4 and 1 of the corner radius, the right-minus-left difference with its sd, and two
  likelihood-ratio tests from the joint's solution - k2 = 0 in both eyes, and ZR's k1, k2 = ZL's - each with the
  largest change of the curve it causes.  Over the 34 mm frame r^3 and r^5 correlate 0.98, so the eyes' different k1,
  k2 (20 Oct: ZL -0.46/0.55, ZR -0.48/0.95) give curves within 0.1-0.2 px of each other.  `--no-k-tests` skips the two
  extra adjustments.
- **Zoom plumbing**: `fit_zjoint` / `write_zjoint` load the focus model of the zoom asked for (they always loaded 34 mm).
- **Documentation site** (MkDocs Material + mkdocstrings, `mkdocs.yml`, `docs/*.md`, `.github/workflows/docs.yml` to
  GitHub Pages; `pip install -e ".[docs]"`), **Zenodo** (`.zenodo.json`; `CITATION.cff` valid: cff-version 1.2.0) and
  the release steps for the DOI in `docs/RELEASING.md`.
- The example data: `MPPP_PDS` is the unpacked folder (it holds `datadrive/` and `zcam/`); `mppp example fetch` said
  `datadrive/`, which hides the Mastcam-Z frames.  The matcher docstrings say what the default is (exhaustive SIFT).

## 0.91.0 — 2026-10-20

The joint as one bundle adjustment, the rig fitted for yaw and roll, Mastcam-Z p1, p2, the site maps and the CLI.

- **21 Oct, 08:30 - Issole / South Seitah back in `zcam34_consensus` (12 blocks)** without its Mastcam-Z 34 mm video
  (sequence ZCAM05093: the site's `settings` carry `EXCLUDE: ["seq:ZCAM05093"]`).
- **21 Oct - Issole / South Seitah out of `zcam34_consensus` (11 blocks; 2713 images, 2585 of them Mastcam-Z 34 mm: 3.7 million
  exhaustive pairs, over 9 h of matching), Olifants kept.  `max_exhaustive_images` (1000, not in the run key):
  "exhaustive" over more images matches the prior-overlap pairs instead, and the log says so.  A `SKIP` file in a WORK folder leaves the block out**
  of `run_sites` - also of a batch already running, since each block's `align_scape.py` exits (code 5) before doing
  anything.  Delete the file to align the block again.
- **20 Oct, evening - mask retraining without the notebook:** `scripts/train_mask.py` (headless training on a GPU
  machine, `--regen` from the PDS, `--tag`, the frame audit afterwards; never promotes or releases; `--smoke` checks
  it on synthetic data) and `scripts/make_mask_figure.py` (the paper's mask figure: held-out frames per camera with
  the predicted exclusion and the label's boundary, metrics per camera).  pytest keeps temporary folders only of
  failed tests.
- **20 Oct, 20:00 - `run_sites_zcam34.bat` runs the Mastcam-Z joint after the blocks** (`zcam_tied_joint.bat`, which
  promotes only when every check passes; `set MPPP_THEN=none` aligns only, `set MPPP_THEN_ARGS=--no-promote`).
  `_run_sites.bat` calls `MPPP_THEN` when the batch ran (exit 0 or 1).  The run_sites / process_sites /
  sites_status / stop_mppp .bat files check the scapes root after `mppp_env.bat` (`mppp_scapes.txt`), with
  `D:\scapes` as the fallback - before, they closed at once when MPPP_SCAPES was not in the environment.
- **20 Oct, 19:40 - Mastcam-Z joint gate; 40 official blocks.**  The first real joint (4 blocks, 10 min) was promoted
  with its Mastcam-Z residuals NaN (points behind a camera or outside the lens model) and four blocks aligned with
  older camera models: the residual statistics are now finite-only with the non-finite count, and the joint's
  check adds the Mastcam-Z median (<= 0.6 px), the non-finite share (<= 0.5 %) and "blocks aligned with the models
  in use" (their `run_done.json` fingerprint).  Seitah Overlook is no longer an official block (40, renumbered:
  Artuby Ridge 4 ... Hans Amundsen Memorial 21 ... Marble Mountain 40); "Hans Amundsen Memorial Workspace" is
  now "Hans Amundsen Memorial", and it joins `navcam_consensus` (27).  `taylorfjellet_large` is now `taylorfjellet`
  (official block 33, `zcam34_consensus`, `zcam_consensus`; its WORK folder becomes `mars2020_sol_1601_taylorfjellet*`).
- **20 Oct, 19:30 - the Z34 run's failures.**  Rochette: `KeyError: 'p1'` after stage 2 round 1 - zcam-v2 put
  p1, p2 on the focus lines but `mppp.sfm.zcam.refit_param_lines` had no index or model slope for them (the staged
  N+Z test now uses the pipeline's line parameters).  Issole: died unseen in the CPU feature extraction of 2713
  images - extraction, like matching, now always runs in a child process.  Matches of another matcher are kept:
  the replaced `database.db` moves to `colmap/matches/database_<matcher>_<key>.db` (with its record) instead of
  being deleted, and a run with those settings again copies them back (`mppp.sfm.database.matches_library`) - an
  A/B of SIFT against LightGlue matches each block once per matcher.  `claude_sync.bat` also pushes the bundle's
  working branch (e.g. `v0p91`) to GitHub.
- **20 Oct, 18:30 - batch runs use the GPU environment again.**  `run_sites` / `align_scape` looked only at
  `MPPP_GPU_PYTHON`; the notebook also read `MPPP_GPU_PY` and `%LOCALAPPDATA%\mppp\gpu_python.txt` (from
  `install_lightglue.bat`).  With LightGlue off by default, Chal Rocks Z34 then matched 60031 pairs (32768 features)
  with COLMAP's SIFT on the CPU inside the run's own process, silently for 80 min, until the process died with no
  traceback and the status left at "running" (07:00 the same block took 9 min on the GPU).  `Settings.gpu_python` now
  looks in the same places, the log warns when there is none, and COLMAP's matcher always runs in a child process
  (the CUDA environment, else this python): a COLMAP abort fails the step with its message, and the calibrated
  verification fallback can take over.
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
- **Example data** (`mppp.example`, `mppp example fetch | build`, `build_example_data.bat`): Rockytop sols 461-464 -
  the 32 main Navcam images, one front Hazcam pair on each of sols 461 and 464, the eight Mastcam-Z 34 mm frames of
  ZCAM03381 (an explicit list, `mppp.example.EXAMPLE_FILES`) -
  zipped in the archive layout with a manifest, for the GitHub release `example-data-v1`; site `rockytop_example`.
- **Sites**: Seitah Overlook and Hans Amundsen Memorial Workspace are official blocks (41, renumbered by first sol).
  Maps: "Octavia E. Butler Landing Site"; cluster zooms with the legend in two columns beside the title, the
  traverse blue as in the maps, no 10 m circles on the orthomosaic panels, a grey mosaic in grey (one stretch for
  the traverse; a 3-band colour mosaic is shown in colour).  `zcam_tied_joint.bat` promotes by default
  (`--no-promote` to only check).
- **Maps**: figure 7, the GitHub banner (`m20_traverse_banner_<tag>.png`: the traverse panel wider and shorter, the
  title in the middle, the legend lower right, the landing site starred; `M20_BANNER_BOX`, `M20_BANNER_TEXT_Y`);
  the cluster zooms on white (`--background white`, made by `make_maps.bat`); legends: "Rover Traverse",
  "Mastcam-Z 360°", "Sample Core Acquired".
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
