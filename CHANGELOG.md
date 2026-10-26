# Changelog

The development history before 0.90 is in `docs/history/` (`CHANGELOG_v0p1-v0p12.md`, `CHANGELOG_v0p13-v0p85.md`).

## 0.94.4 — 2026-10-23

- Sites (Christian, 26 Oct): Taylorfjellet leaves out ZCAM05065 (`settings.exclude`). A Mastcam-Z sequence left out (`seq:ZCAM...`) no longer changes the run key of a Navcam-only run. A front Hazcam frame of Marble Mountain (sol 1976) is on the global exclude list.
- **Focus from below in the blocks and the Mastcam-Z joint**: the frames on the focus-approach list get their own bin cameras (`_fb`, before `_reg`); the joint fits one intercept offset per eye (px on fx, fy x the aspect; same focus slope), reported with its sd and written into the focus model (`focus_below_offset_px`).
- **The focus approach, a second backlash list** (Christian, 26 Oct: "two different backlash effects superimposed"): `<scapes>/zcam_focus_from_below.csv` (and the shipped `mars2020_mastcam-z_focus_from_below.csv`) lists the frames whose focus came into place from below (the motor moving up); the default is the focus coming down (sol 93 ZCAM00034, sol 102 ZCAM08062). Independent of the zoom low / high list, every zoom including 110 mm (`backlash.load_focus_below`, `is_focus_from_below`). `backlash.motor_moves`: each Mastcam-Z frame's zoom and focus move from the previous frame of its eye (manifest counts). `mppp zcam-states` reports the focal ratios by how the motors came into place (110 mm included) and proposes focus-list frames from moves up of 100+ counts.
- Sites (Christian, 26 Oct): Chal Rocks' Mastcam-Z block takes its 110 mm frames again (the tiny copy stays without them).
- The exact-tie solve is off in the Mastcam-Z and combined joints (`ztied.EXACT_TIE_SOLVE`): on the first real one (Z34, 7 blocks, 26 Oct) it found no decrease and took 524 s per call; the Navcam joint keeps it.
- Restart after the PC went down on 25 Oct (Christian, 26 Oct: results by the end of the day): `run_pipeline --zcam-sites SITE ...` aligns only those Mastcam-Z sites; the joints take the other sites' finished blocks as they are (Rochette and Enchanted Lake were aligned on 25 Oct with the Navcam models in use; a block never finished, Olifants, is left out). Marble Mountain's Mastcam-Z block without its 110 mm frames.
- **The ties as exact constraints in the solve** (`mppp.calib.tiesolve`, `tied.EXACT_TIE_SOLVE`, `adjust_tied(exact_solve=)`; Christian, 25 Oct: "write each unit's camera as a function of the globals"): after Ceres, Levenberg-Marquardt continues with the tie residuals removed, every tied unit set from the globals, and the steps taken in the reduced variables (points by the Schur complement, damped like Ceres; poses, untied unit terms, the globals' null space); the ties go back at zero residual. On the synthetic joint the converged pitch rate no longer depends on the tie sigma (0.224 .. 0.247 .. 0.012 mdeg/1000 sol at 1e-5 .. 1e-12 rad before, 0.238 at every sigma after). The Mastcam-Z and combined joints pass their tie specs too. On by default (`EXACT_TIE_SOLVE`). The points are damped like Ceres' (undamped, a poorly seen far point stepped by a metre and brought an observation back from behind its camera: +34 in cost at any step), a failed evaluation is never accepted, and the joint reports the refined cost.
- **Navcam joint defaults from the 12-block study** (25 Oct): the tie sigmas of k and the rig 1e-6 (were 1e-7, 1e-8) - Ceres converges further and in half the time, the exact solve then puts the units on the law; k4 of each eye held at the models in use (`HOLD_DEFAULT`): k1-k4 trade off (|r| 0.88-0.99), held the scaled condition falls from 4.2e5 to 3.0e4, the k1-k3 sd 3-8x and fx's 15 %, the radial curve moves at most 0.08 / 0.18 px (NL / NR) at the very corner; k4 = 0 is rejected (cost +50). Tested and NOT adopted: the common-mode principal-point slope at 0 (cost +22, pitch_early +3.2 sd), both k1 slopes at 0 (cost +30, NL fx slope -3.3 sd); NR's k1 slope at 0 is consistent (0.45 sd) and halves NR's fx-slope sd (1.04 -> 0.58 ppm/degC) - left free until the full joint confirms it.
- A held thermal slope no longer sends every global sd to the bin-camera fallback (and skips the early-mission test).
- **The law check** (`mppp.calib.lawcheck`; Christian, 25 Oct: how good the global solutions are at predicting the cameras): over finished blocks, every unit held at a law (per-block residuals the law leaves), the same units free (each unit's deviation from the law in its own sd, chi^2 / dof per term, trends with temperature and sol) and the pointing pass (principal points, stereo pointing); one free pass compared with any number of laws.

- **The covariance of the joints' global terms with the ties eliminated exactly** (`mppp.calib.tiecov`, Christian 25 Oct: the 24 and 25 Oct Navcam joints failed promotion on infinite rig sd; 20 Oct had needed the bin-camera fallback). The joints tie every unit to the global terms with residuals of sigma 1e-3 px .. 1e-8 rad - weights up to 1e16 next to data weights of 1e7 - and COLMAP's covariance estimator, which inverts the tied problem as it is, returned infinite sd for the global blocks from about a dozen blocks on (fine on 3 and 12 real blocks, infinite on 22-29). After the solve the tie residuals now leave the problem, the points are eliminated, the tied unit parameters are substituted by the globals they equal, and the globals' marginal information is inverted (scaled): no number in it is far from the data's size. It agrees with COLMAP's estimator where that works - within 1 % on the synthetic joint and within 0.6 % on 12 real blocks (pitch rate sd 0.0910 / 0.0911 mdeg per 1000 sol, NL fx slope 1.0416 / 1.0412 ppm/degC) - and does not depend on the tie sigma. Used by the Navcam joint, the Mastcam-Z joint and the combined joint (`EXACT_TIE_COVARIANCE`); COLMAP's estimate stays for the units and is kept for the globals if the exact one fails (`covariance_method`, `covariance_exact` in the results). The log said "covariance ok" while every rig sd was infinite: it now names the method, the scaled condition and says NOT USABLE when an sd is not finite or comes from the fallback; the strongest correlations between global terms (|r| >= 0.8) are reported (`global_correlations`).

- **The found list is read again, always** (Christian, 25 Oct: "Always use this list. Add and subtract frames as needed"): `zcam_low_backlash_found` is "auto" again (`<scapes root>/zcam_low_backlash_found.csv`, off since 0.93.6), in the blocks, in the classification of their focus groups, in the Mastcam-Z joints' list check (`ztied.build(found=...)`, `found_list` in the result) and in the per-image start cameras. A line starting with `-` takes frames off the list - it wins over every match in either file (`-0121_0677678972_053`). The PC's found list: sol 121 ZCAM08114 pointings 2 and 4 (34-79 mm, both eyes) commented out and taken off with `-` lines (the sequence alternates low, high, low, high, low: 0.996-0.998 against 1.001-1.004); backup `zcam_low_backlash_found_20261025_before.bak.csv`. The found list is part of a Mastcam-Z block's run key (every `_zcam` block realigns); Navcam-only keys are unchanged.
- **The states run** (`run_zcam_states.bat`, then `run_zcam_states.bat report`): the Mastcam-Z sites aligned again as the variant `states` with every Mastcam-Z image a camera of its own (`zcam_bin_by = "image"`; also `"sequence"`), its focal length free (`ZCAM_HOLD_F_IMAGES=0`, `ZCAM_BIN_REFINE=focal`) and no focus lines - each image's f / label f is its own.
- **`mppp zcam-states`** (`mppp.sfm.zcam_states`): per block, eye and zoom the high-state line in focus count (robust; the slope pooled over the blocks where a block spans under 500 counts) and the state gap (from the listed images, else pooled per zoom, else 0.95 %); each image low / high / undecided / implausible; each sequence's pattern in clock order (`LHLHL`) and verdict; the proposal for the found list (additions - also the unfitted images of an all-low sequence; `-` lines for listed images found high; the mixed sequences, zooms that disagree and failed fits to review); `--apply` writes it (a dated backup first). On this weekend's blocks (7, the production alignments): pooled gaps 34 mm 1.06 %, 63 mm 0.41 %; to add ZCAM00034 sol 93 at 63 and 79 mm; nothing listed is found high; to review Chal Rocks ZCAM08396 sol 364 ZR034 (16 high, the last image low), ZCAM00034 sol 93 (34 mm high, 63 and 79 mm low).

- Mastcam-Z joint (Christian, 24 Oct): the high / low backlash ratio can vary by block or sol (`calib zcam --zcam-ratio-by block|sol|global`, default block, `ztied.REGULAR_RATIO_BY`). Each group of low-state bins gets its own offset on a_fx_regular (fy by the aspect), drawn to 0 with an sd of 0.3 % of f; the eye's ratio is the groups' mean. The blocks' 34 mm low frames against their own high-state lines: Chal Rocks ZL 1.0145-1.0153 / ZR 1.0131-1.0147 (sols 91-94 agree), Seitah Overlook 1.0152 / 1.0126, Three Forks North 1.0102 / 1.0098-1.0117, Belva Crater ZL 1.0167. The groups' ratios are in the result (`regular_ratio_groups`), the report lines and the candidate model (for the record).

- Mastcam-Z long zooms (Christian, 24 Oct; Chal Rocks: 136 of 214 79 mm frames excluded with "0 tie points with Navcam or a wider zoom", 53 more on residuals): a tie with a wider zoom (34 or 48 mm for 79 mm) counts with two looks; a long-zoom frame with fewer than 10 direct ties is anchored through its mosaic (50 tie points with same-zoom frames that are anchored, directly or in turn) - at Chal Rocks a 79 mm frame shares a median 13,400 points, nearly all with the other 79 mm frames, and 1-2 with a 34 mm one (57 of 170 anchored directly, the other 113 through the mosaic); the zooms without a calibrated focus model (48, 79 mm) are not excluded on residuals before the final test (their focal lengths start 0.5-0.7 % off at the labels and are fitted from round 2). `ZCAM_RULES` 2: the Mastcam-Z blocks realign.

- Sites (Christian, 24 Oct): Chal Rocks (and its tiny copy) take the 110 mm Mastcam-Z frames too (`zcam` 34, 48, 63, 79, 110): its Mastcam-Z block realigns on the next run.

- Navcam joint (Christian, 24 Oct, from the 20 and 24 Oct joints):
  - the rig roll is refined in the joints (`tied_reconstruction`, `merge_scapes`): the blocks hold it and their settings came through, so every unit's roll was held and the global roll and roll rate were never refitted (-56.795270 mdeg and -4.933370 mdeg/1000 sol to 7 digits on 20 and 24 Oct, sd 6e-5 mdeg; the diagnostic roll profile had Q2 0.0);
  - with one relative focal slope the keypoint map scales fy by fx's slope (the ties'), not the held fy slope of the last sync; a pass that starts more than 1.5x above the last one's end gets one more pass (24 Oct: 2.54e6 after 1.82e5); the pass costs are recorded;
  - the promotion check's covariance needs the joint's own sd: the bin-camera fallback (correlations ignored, 3-4x small) and a non-finite sd of a free rig term fail it (24 Oct: pitch and yaw rates inf passed);
  - the early-mission offsets are kept only where |z| >= 3 on the joint's covariance; the others are set to 0, held, and the joint refitted (`early_test` in the result; `applied_terms` of the camera files);
  - the joints' points (`joint._subsample_points`): drawn by the stations they are seen from (capped at 4) instead of track length; none under 0.25 deg (drifted below the blocks' cut: 1.7 px median at Bunsen Peak) or over 60 deg (1.3 px); 20-60 deg (0.54 px) at half the weight.

- Sites (Christian, 24 Oct): Enchanted Lake's second visit starts at sol 422 (NEARBY_FROM_FIRST_SOL keeps sol 421's station, 4.3 m away, out); Knob Mountain Large includes sols 440-459 and 535-537 (149 m wide, 30 waypoints). The zooms map figure's title no longer says "one stretch for all panels".
- Health: the excluded-frames FAIL limit is 33 % (Christian, 24 Oct; was 30 %). Chal Rocks Mastcam-Z, 30.2 % excluded, is a WARN. Thresholds are not part of a run key: finished blocks are judged again where a joint picks its blocks.

- A Mastcam-Z joint judges a FAIL block on its own zoom (`health.restrict_to_zoom`, `cli._rejudged(zoom=...)`): the excluded fraction counts the Navcam, Hazcam and that zoom's frames, the corner check that zoom's cameras. Chal Rocks Mastcam-Z (24 Oct): 30.2 % excluded, 180 of the 194 frames at 79 mm (too few ties to the Navcam or a wider zoom, residuals over 2 px) and the corner check failed on ZL063 - so the Z34 joint left out a block whose 34 mm frames are fine (3.4 % excluded at 34 mm).

- Hazcam joint (Christian, 24 Oct; `mppp.calib.hazjoint`, now the default of `mppp calib hazcam`, `--hazcam-method blocks` for the old fit): the Hazcam law fitted on the image observations of every block at once, each frame at its own camera temperature, the poses free and the points held (only points seen from two other frames), instead of a line through one refined camera per block. One ppm/degC for all four eyes, informed by FL (`--hazcam-slope FL`, default; `all` shares it, `per_eye` frees it; the per-eye slopes are reported for comparison); the rear eyes' k1 and k2 constant. Diagnostic temperature bins per block and eye: 5 degC, at least 3 images, at most 15 degC wide (the Navcam's: 2 degC, 8 images, 10 degC). Early-mission offsets of the Hazcams (joint and block fit) are kept only where significant (|z| >= 3), else 0.

- Rover-frame study: every station is tagged by its place in its rover site (`last_before_site_change`, `site_start`, `within_site`; Christian 24 Oct: the last waypoint before a site index change is the more likely to have a rotation error), and the report and `rover_frame.json` give the front-Hazcam heading error about the block median by tag.

- Rover kernels (Christian, 24 Oct): the rover-frame study estimates the rover attitude and position per rover pose (end-of-drive pose from both Hazcam pairs, later poses from the front pair, propagated poses; the rear-to-front Hazcam correction and the IMU-mean rover frame over all blocks) and writes them with spiceypy: a CK per block (`M2020_ROVER` relative to `M2020_LOCAL_LEVEL`, constant per interval), an SPK per block (with NAIF's clock kernel), an FK of the four Hazcams, the sigma / source sidecar and a meta-kernel (`studies/rover_frame/rover_frame/estimate.py`, `spice.py`; `--no-kernels`; the `rover` extra).

- The calibration joints take blocks that fail only the labels' pointing checks (`block_rotation_deg`, `attitude_residual_p95_deg`): Tuxedo Park (eight stations' headings 1.6-1.9 deg off) and Witch Hazel Hill (one site's heading 3.3 deg off) have median residuals of 0.12-0.15 px.  Their health verdict stays FAIL elsewhere (`sfm.health.LABEL_POINTING_CHECKS`, `rejudge(warn_only=...)`).

- **Mastcam-Z sites for a faster run** (Christian, 24 Oct): `zcam_consensus` and `zcam34_joint` take Chal Rocks instead
  of Chal Rocks Large (1601 images); Pico Turquino left out of `zcam_consensus` (Z48 keeps Chal Rocks, Seitah
  Overlook and Taylorfjellet - the minimum of three blocks).
- **`run_pipeline.py`: a `lightglue_ab` stage after `rover_frame`** - `scripts/lightglue_ab.py` on Chal Rocks tiny and
  Seitah Overlook tiny, every variant again with the models just promoted (`--rerun`), no tiny joint and no
  promotion (`--no-joint --no-promote`); its tables go into REPORT.md.  The state tag stays v0944: a pipeline started
  before the stage existed runs it (with `rover_frame` and the report) when `run_pipeline.bat` is started again.
- **Good camera models are promoted** (24 Oct, the checks of 20-23 Oct):
  - the Hazcam fit leaves out a block far off an eye's law (more than 4 robust sd and 1500 ppm; 23 Oct: Belva
    Crater's rear eyes 0.7 % low on 1.5k observations) and refits;
  - an eye of 8 or more blocks passes its block-scatter check on scatter / sqrt(blocks) <= 400 ppm (the rear eyes
    scatter 1600-2500 ppm per block, so a rear law of one tiny block stayed in use);
  - the Navcam promotion gates on the median residual against the models in use; the rms and the p95 follow the
    block mix of the two joints and are reported only.
- **A Hazcam pair's eye ratio no longer fails a block** (24 Oct: Knob Mountain Large FAIL on RL 0.184 / RR 0.422 px,
  ratio 2.29 - it would have left the joints): `hazcam_residual_eye_ratio`, warn 2 / fail 10 (the Navcam and
  Mastcam-Z pairs keep 1.3 / 2.0); saved reports are judged again with it.
- **Fix: Hazcam frames excluded as outliers for their station's shift** (weekend run, 23 Oct).  Stage 2 moves a
  Hazcam image's position prior by its station's Navcam shift (`hazcam_station_priors`, `prior_C_label` keeps the
  label prior), but the outlier test measured every frame from `prior_C` and grouped the Hazcam frames with the
  Navcam ones: a well-placed Hazcam frame then looked its station's shift (0.25-2.1 m) away and was excluded -
  4 frames in OEB Landing Site, 17 in Chal Rocks Large, 48 in Van Zyl Overlook Huge, and the lone right Hazcam
  frames whose exclusion crashed Seitah Overlook and Artuby Ridge.  Shifts are now measured from the label prior
  (`reconstruction.label_prior_C`; also `poses.csv`, the station shift health checks and the prior similarity),
  and Hazcam frames form their own group ("H"), tested against their station's Navcam frames, never on attitude.
- **Blocks already run are judged again from the label priors** (`health.restation`, used by `calib`'s check of
  a FAIL block): Van Zyl Overlook Huge's `within_station_shift_spread_m` FAIL (1.06 m) was this artifact, and would
  have kept the block out of the Navcam joint and the Hazcam fit.
- **Stereo exposures with one eye only in the local archive are reported** (`pipeline.one_eye_exposures`, a log
  line and `one_eye_exposures` in the selection report).  The lone right Hazcams of Seitah Overlook (sols 122-124)
  and Artuby Ridge (sol 173) are J01 products of exposures whose left and right images the PDS has as J02: the
  local archive is incomplete, not the rover.  New `scripts/archive_gaps.py` lists every such exposure of the
  archive (`<pds>/archive_gaps.csv`) and with `--download` fetches the missing eye from the PDS Imaging Node (the
  highest version of the same product kind, into the folder of the eye that is there; nothing replaced).
- **Fix: a right camera dropped by the triangulation** (weekend run, 23 Oct: Seitah Overlook crashed, Artuby Ridge
  "Camera model does not exist").  A right Hazcam exposure without its left image sits in a one-camera rig of its
  own; when stage 2 excluded that rig's last frame as an outlier, COLMAP's `triangulate_points` dropped the rig with
  its camera although the stereo rig still used the camera.  `reconstruction.triangulate` now puts such cameras
  back (`restore_shared_cameras`; reproduced on the Rockytop test block with a forced exclusion).
- **Fix: a right camera dropped by the triangulation** (weekend run, 23 Oct: Seitah Overlook crashed, Artuby Ridge
  "Camera model does not exist").  A right Hazcam exposure without its left image sits in a one-camera rig of its
  own; when stage 2 excluded that rig's last frame as an outlier, COLMAP's `triangulate_points` dropped the rig with
  its camera although the stereo rig still used the camera.  `reconstruction.triangulate` now puts such cameras
  back (`restore_shared_cameras`; reproduced on the Rockytop test block with a forced exclusion: "Camera with ID 2
  does not exist" before, the block finishes after).
- **SIFT is the default matcher again** (`match_mode` "exhaustive", Christian 23 Oct; 0.93.9 had "lightglue").  The
  finished A/B on Chal Rocks tiny: SIFT 13,604 Navcam - Mastcam-Z tie points, LightGlue 7,578, LightGlue on every
  octave 7,043; residual median 0.22 vs 0.37 px, p95 0.94 vs 1.39 px.  LightGlue tied 3x more points across
  stations (12 % vs 4 %) and kept the rear Hazcams steadier (RL / RR rays 4-5 px vs 27-29 px), so it stays as
  `--set MATCH_MODE=lightglue` or `sift+lightglue` for the ecam blocks.  Above `max_exhaustive_images` (300) images
  "exhaustive" is SIFT over the prior-overlap pairs, as before; pose-guided matching follows either way.
- **Exclusions** (Christian, 23 Oct 16:48): `seq:FHAZ00202`, `seq:FHAZ00215`, `seq:FHAZ00216`; `ZCAM01023` is no
  longer kept (`seq:ZCAM01*` covers it).  Three Forks North: no mask inference for NCAM03688 and NCAM13688.
- **Study 3A, the rover frame** (Christian, 23 Oct; `studies/rover_frame`, `run_pipeline.py`'s new `rover_frame`
  stage after the error analysis): from the labels the RSM angles (encoder, resolver, requested), the RSM head frame,
  the rover attitude and motion counters (POSE, ARM), the Navcam plate temperatures, the bogie, differential and arm
  angles (keyword names checked on sol 709-711 labels); per Hazcam sequence and rover pose the body pose (the Hazcams
  are body-fixed) and its residual against the telemetry; per Navcam / Mastcam-Z frame the attitude relative to that
  body in the head frame (label kinematics removed) fitted against cos / sin el, cos / sin az and T; the camera
  positions in the body and head frames as constants with their scatter; the body-pose changes between the
  sequences of one station (wheel settling, the arm).  No new adjustment.  The state tag stays v0944, so after the
  weekend run `run_pipeline.bat` runs only `rover_frame` and the report.
- **`scripts\windows` trimmed to 19 files** (Christian, 23 Oct: "too many bat files"): removed `run_weekend.bat` and
  `_run_weekend_loop.bat`, `build_example_data.bat`, `run_multizoom_test.bat`, `run_hazcam.bat`,
  `hazcam_calibration.bat`, `test_lightglue.bat`, `navcam_tied_joint.bat`, `zcam_tied_joint.bat`,
  `run_sites_zcam34.bat`, `run_z34.bat`, `run_v0p87.bat`, `move_scapes.bat` and `sites_status.bat` (`check_sites.bat`
  shows the same); `check_env.py` and `lightglue_check.py` moved up to `scripts\`.  `docs/WINDOWS.md` lists what
  is left by use and what replaces each removed file.
- The block's own `EXCLUDE` entries are reported at the processing, where they now act (`not processed (EXCLUDE
  'sol:15-31'): N products`, with the typo warning there); the alignment no longer calls them a typo.
- `tests/test_calib.py`: the Hazcam start-model tests use a models folder without RL, now that FL, FR, RL and RR
  are all in use.

- **Two new matchers for the LightGlue test** (Christian; the Chal Rocks tiny A/B: LightGlue had 3.6x fewer
  Navcam - Mastcam-Z tie points than SIFT, 13,512 vs 3,786, and higher residuals, 0.35 vs 0.21 px median):
  - **LightGlue on keypoints of every octave** (`LIGHTGLUE_KEYPOINTS=octaves`; `lightglue.select_keypoints`): the
    8192 keypoints are an equal share of every octave of scale (small octaves keep all theirs), not the 8192 largest.
    The largest-scale keypoints are the least precisely located, and a Navcam frame loses exactly the fine keypoints
    that meet a Mastcam-Z frame of 3-10x finer ground resolution.
  - **SIFT and LightGlue together** (`MATCH_MODE=sift+lightglue`): COLMAP's SIFT matcher as "exhaustive" (or the
    prior pairs of a big block), then LightGlue over the same pairs in a copy of the database, verified there, its
    inliers added to each pair's SIFT inliers (`matching.merge_inliers`; the SIFT match wins where both use a
    keypoint; a pair SIFT did not verify gets LightGlue's geometry). Without LightGlue (or over its 2 h budget) the
    SIFT matches stand, recorded as such.
  - The run key and the matches record carry the keypoint selection only when it is not "largest", so finished runs
    and their matches stay valid. The default matcher is unchanged (`lightglue`, largest scales).
- **Pose-guided matching runs in one thread by default** (`GUIDED_DEFAULTS["workers"]` 1; 0 = min(8, CPU count)):
  the Chal Rocks tiny SIFT + LightGlue run of 14:55 stopped writing at this step (15:11) and seems to have held its
  `database_guided.db` open, so the 16:05 run failed on it (WinError 32); on the tiny blocks the threads saved
  nothing (120 s either way). A `database_guided.db` held by another process is now left alone and the step writes
  `database_guided_<pid>.db`.
- **PyPI** (Christian: <https://pypi.org/project/mppp/> for the JOSS submission): the project metadata PyPI shows
  (SPDX license with LICENSE and NOTICE, author, keywords, classifiers, documentation / repository / issues /
  changelog links; setuptools 77), `MANIFEST.in` (the sdist: package, data, tests and their two PDS products; not
  the studies, notebooks, Windows scripts, paper or documentation site), `.github/workflows/publish.yml` (on a
  published GitHub release: build, `twine check --strict`, the version against the tag, a clean install of the
  wheel, then the upload with PyPI trusted publishing - no token; "Run workflow" rehearses on TestPyPI),
  `docs/RELEASING.md` section 8 (the one-time pending-publisher setup), `pip install mppp` in the README and the
  installation page (absolute links: relative ones break on PyPI), `mppp --version`. Built and checked here: wheel
  760 KB, sdist 12 MB, both pass `twine check --strict`, the wheel installs and runs in a clean environment.
- **Two crashes of the 23 Oct afternoon run fixed** (run_sites_zcam, two blocks at once): LightGlue's SIFT fallback
  (Chal Rocks Large Mastcam-Z: 152,059 prior pairs projected at 3.0 h) ran COLMAP's CPU matcher inside the alignment
  process, where OpenBLAS overflowed its thread table on the many-core PC ("precompiled NUM_THREADS exceeded") and
  the process died; the matcher now always runs in the CUDA environment's child process, as for "exhaustive". The
  alignments get `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` 8 unless set (run_sites, lightglue_ab). Seitah Overlook
  Mastcam-Z died in native code at the start of stage-2 round 3 with no message (2.5 min after, while Three Forks
  North started processing beside it). **`run_sites` runs a failed site once more, on its own, when the others are
  done** (`--retries`, default 1).
- **A Hazcam + Mastcam-Z point needs a Navcam look** (Christian's question, 14:50: the third-look rule covered only
  Navcam + Mastcam-Z tracks; `reconstruction.HAZCAM_ZCAM_NEED_NAVCAM`, ALIGN_RULES 92): a point whose track holds a
  Hazcam and a Mastcam-Z image but no Navcam image is deleted, whatever its length. Those points were most of the
  scattered Mastcam-Z points: Chal Rocks tiny 46 / 23 % above / below the Navcam terrain (two-view; 7 / 5 % with
  three looks or more), Seitah Overlook tiny 67 / 24 %; with a Navcam look too they are on it (0.1 / 0.0 %). With the
  isolated-point filter after it: Mastcam-Z above / below 2.85 / 1.62 -> 0.17 / 0.14 % (Chal Rocks tiny, 6.4 % of the
  points deleted) and 1.30 / 0.38 -> 0.58 / 0.14 % (Seitah Overlook tiny, 0.7 %); no Navcam - Mastcam-Z point deleted.
- **OEB Landing Site without sols 15-31** (`EXCLUDE ["sol:15-31"]`) and **Van Zyl Overlook Huge without the first three
  site-3 waypoints** (`EXCLUDE ["S003D0000", "S003D0008", "S003D0110"]`) (Christian).
- **Isolated two-view points deleted before the final stage-2 adjustment** (Christian: "a few points scattered
  above and below the terrain"; `Settings.isolated_points_factor` 4.0, 0 = off; `reconstruction.drop_isolated_points`;
  ALIGN_RULES 91). A two-view point whose mean distance to its 8 nearest points, divided by its range, is more than
  4x the median of the points at a similar range (range bins of 1.5x) is deleted; a point a third image confirms
  stays, so no Navcam - Mastcam-Z point (three images at least) is deleted. The scattered points are false matches at
  the wrong range (mostly from pose-guided matching; 73-94 % two-view, median isolation 9-16x against 0.5x for the
  others). On the Chal Rocks tiny / Seitah Overlook tiny SIFT blocks the Mastcam-Z points above / below the Navcam
  terrain go from 2.81 / 1.60 % to 0.29 / 0.21 % and from 1.30 / 0.37 % to 0.58 / 0.10 %, deleting 5.6 / 3.7 % of
  the points and 1.3 / 6.2 % of those beyond 40 m. Only points within 1.5 radii of the cameras' bounding sphere are
  judged (Christian, 14:39; `ISOLATED_POINTS_RADII`): 0.33 / 0.24 % and 0.76 / 0.17 %, deleting 5.3 / 0.5 % of the
  points and none beyond 40 m - the default. (First version, one block-wide median over all points: it
  deleted 40 / 28 % of the points beyond 40 m and up to 732 Navcam - Mastcam-Z points - Christian's question, 14:36.) Pose-guided matching itself was checked on the real blocks: the 0.94.3
  candidate search and its threads give the same matches as before (131,930 at Chal Rocks tiny); stricter match
  settings (both images' depth windows) removed far fewer of the false matches than the filter.
- **`run_weekend.bat`: several days unattended** (Christian): waits until nothing else runs, the A/B on Chal Rocks
  tiny (every variant again: its masks changed) and Seitah Overlook tiny - SIFT, LightGlue, LightGlue on every
  octave, SIFT + LightGlue, SIFT without pose-guided matching, SIFT without the isolated-point filter - then the whole
  `run_pipeline` with SIFT (`--set MATCH_MODE=exhaustive`), then `WEEKEND.md`. Restartable (state in
  `weekend_state.json`: the pipeline is reset only the first time; the A/B keeps the variants done since), restarted
  after a crash by its loop (at most 20 times, never after `stop_mppp.bat`), the computer kept awake.
  `run_pipeline --set NAME=VALUE` passes block settings to every run.
- **`--set MATCH_MODE=sift+lightglue` works** (the `+` made it Python source; the A/B's SIFT + LightGlue variant
  failed at once).
- **`navcam_consensus` is now `ecam_consensus`** (Christian): the engineering-camera blocks - Navcam with the front
  and rear Hazcams. The old name still works everywhere (`waypoints.sites.GROUP_ALIASES`); the file has the new one.
- **`run_pipeline` fits the Navcam, the front and the rear Hazcams before the Mastcam-Z runs** (Christian): the
  stages `navcam_runs` / `navcam_joint` are `ecam_runs` / `ecam_joint` (the old names still work for `--from`,
  `--redo`, `--stop-before`); `ecam_joint` runs the Navcam tied joint and then the Hazcam fit (FL, FR, RL, RR and the
  F and R rigs, `calib hazcam --group ecam_consensus`) over the same blocks, each promoted when its checks pass, so
  the Mastcam-Z blocks start from both. `zcam_joints` no longer fits the Hazcams again (the Mastcam-Z blocks hold
  the Hazcam frames of the ecam blocks of the same sols, which would count twice); `--zcam-hazcam-fit` does it anyway.
  The state file starts afresh (`state_before_v0944.json`).
- **LightGlue's time budget per block is 3 h** (Christian; was 2 h): a block whose LightGlue matching is projected
  (from its first 20 pairs) to take longer falls back to SIFT (`lightglue`) or keeps its SIFT matches alone
  (`sift+lightglue`). Not in the run key.
- **`check_sites.bat` shows each block's last run** (Christian): per site, when its Navcam block and its Navcam +
  Mastcam-Z block last finished, the MPPP version, variant and verdict (the newest `colmap*/run_done.json`), or
  "running since"; `--root` or `MPPP_SCAPES`.
- **Pico Turquino in `zcam_consensus`** (Christian; Mastcam-Z 48 mm): a second 48 mm block for the Z48 joint.
- **Chal Rocks (all three blocks): no mask inference on exposure `0093_0675177115_000`** (Christian: the model
  masked a rock there), next to `0093_0675176765_000`.
- **Eight image-processing workers by default** (Christian; were 4): `Settings.workers` (`WORKERS`, not in the run
  key), `process_images`, and `train_mask.py --regen`. With two sites at a time that is up to 16 worker processes.
- **`seq:FHAZ00206` on the exclude list** (Christian): its front Hazcam frames are left out of every alignment.
- **Seitah Overlook in `navcam_consensus`** (Christian): its Navcam block (`mars2020_sol_0113_seitah_overlook`) runs
  with the other Navcam blocks and enters the Navcam joint.
- **Belva Crater gets Mastcam-Z 110 mm** (`belva_crater` zcam [34, 110]; Christian): its zcam block (in
  zcam_consensus) aligns the 110 mm frames too, so the next 110 mm joint has a third block.
- **`test_lightglue.bat` runs four variants per tiny block**: `sift`, `lightglue`, `lightglue_octaves`,
  `sift_lightglue` (a variant already finished is not run again; `--rerun`, `--variants`), compared side by side
  with the Navcam - Mastcam-Z tie points (before and after pose-guided matching), the LightGlue inliers added, the
  time from the database to the export (the image processing and feature extraction are done once, by the first
  variant), and the MPPP version and camera models of each run. The tiny joint uses the variant with the most
  Navcam - Mastcam-Z tie points over the blocks (`--joint-variant`).

## 0.94.3 — 2026-10-23

- **Pose-guided matching 3.5x faster on one thread, and memory-bounded** (Christian: run the blocks faster, test it
  for robustness): the epipolar candidates are found by sorting the rays of B by their epipolar-plane angle about the
  baseline and searching windows (`guided.epipolar_candidates`), not by testing every keypoint pair (a 20k x 20k
  product per pair) - the same matches (test against the old every-pair test, 16 of 16 pairs of a 20k-keypoint
  synthetic block identical; random geometry with rays along the baseline). Every stage is chunked, so a thread needs
  about 230 MB whatever the geometry (the old every-pair test needed about 1.1 GB per thread, 9 GB at 8 threads).
  Threads scale 1.6-1.8x on 2 cores (the old dense test hardly scaled: OpenBLAS serialised it).
- **Two sites at a time by default** (`run_sites`, `run_pipeline`; `--jobs 1` or `MPPP_JOBS` to change):
  - the GPU steps (COLMAP GPU SIFT extraction and matching, LightGlue) take turns through an OS file lock
    (`mppp.sfm.gpu.gpu_lock`, `<cache>/gpu.lock`, freed if its holder dies; gives up waiting after 4 h;
    `MPPP_GPU_LOCK=0` off), so two alignments neither run out of GPU memory nor skew LightGlue's 2 h projection;
  - LightGlue out of GPU memory falls back to COLMAP's SIFT matcher for the pairs not matched yet;
  - the shared downloads (waypoint table, mask model, LightGlue weights) are fetched once before the sites start, and
    the waypoint cache and mask download are written atomically (`paths.atomic_write_bytes`, `replace_retry`) - two
    processes refreshing at once could leave a half-written table;
  - each site's console output goes to `<WORK>/runs/console_<time>.txt` (was thrown away with more than one job);
    a site that cannot start is logged as failed instead of stopping the batch.
- `lightglue_ab.py`: the health checks line up by check name (the notes hold numbers, so most rows were split), and a
  "Tie points and alignment quality" table (points, track length, cross-station ties, weakest images, residuals,
  points off the terrain, pose-guided matches and filtered observations).
- Tests: `tests/test_parallel.py` (15).

## 0.94.2 — 2026-10-23

- **Pose-guided matching in threads** (Christian: run the blocks faster): the pairs are matched by up to 8 threads
  (`GUIDED_DEFAULTS["workers"]`, 0 = min(8, CPU count); numpy releases the GIL in the array work), the database written
  by the main thread in pair order - the same matches as one thread (test). It took 23-83 min of a Mastcam-Z block on
  22-23 Oct, single-threaded.
- `run_pipeline.bat --jobs N`: the Navcam and Mastcam-Z block runs align N sites at the same time (`run_sites --jobs`;
  each WORK folder is its own, the shared files are read-only; memory is the limit - default 1; 2 since 0.94.3).

## 0.94.1 — 2026-10-23

- **Seitah Overlook tiny** (`seitah_overlook_tiny`, sols 116-121, every zoom; in no group) next to Chal Rocks tiny.
- **`test_lightglue.bat` runs both tiny blocks**, then **the tiny joint** (Christian): the Hazcam fit and the
  Mastcam-Z joint of every zoom over their `colmap_lightglue` alignments (`lightglue_ab.tiny_joint`), promoted when
  every check passes with as few blocks as there are (`--min-blocks 1`) - only models of which none is in use
  (`--promote-new-only`), so a tiny joint never replaces a model made from more blocks (Z34, Z63, FL, FR, the rigs).
  `--joint-only`, `--no-joint`, `--no-promote`.
- **The fy / fx aspect limit is 2 %** (Christian; was 1 %), for the Mastcam-Z models (`max_aspect_dev`) and the
  Hazcam laws.
- **Mastcam-Z 110 mm in use** (a first calibration, single state): the joint `zcam110_joint_20261023` of 2 blocks,
  which failed only the form (fixed in 0.94.0) and the 1 % aspect limit - fy / fx 0.982 in both eyes, worth a look
  with more blocks. Z79 stays out (one block in its joint, fy / fx 0.992, its low-state ratio fit 0.9955).
- `mppp calib zcam / hazcam --min-blocks N --promote-new-only`; `calib zcam --blocks NAME=<project folder>` takes a
  `colmap_<variant>` folder (its Navcam fingerprint read there).

## 0.94.0 — 2026-10-23

The release of the 22-23 Oct work (0.93.1-0.93.10 below): Hazcams in every block with their own law, the
Mastcam-Z backlash states from the low-state list only, Z110 single-state, the joints of every zoom, LightGlue on,
`run_pipeline.bat` as the whole run with the error analysis.  New in 0.94.0:

- **The shipped exclusion list keeps its images out of the processing too** (Christian: ZCAM03017 - already on the
  list since 22 Oct, it was processed and then left out of the alignment; so were the Mastcam-Z movies of a thousand
  frames). `pipeline.exclude_products` reads sol, site, drive and sequence from the product name. `PROCESS_RULES` 6:
  every block selects again (processed images reused).
- **`no_mask_inference_at` takes single exposures** by sol and spacecraft clock (`0093_0675176765_000`: every camera
  of that exposure, e.g. one Navcam stereo pair; `config.parse_exposures`). Chal Rocks, Chal Rocks Large and Chal
  Rocks tiny: `0093_0675176765_000` in place of `NCAM00699` (only that stereo pair unmasked), `0360_0698912834_738`
  added; ZCAM00034, ZCAM08060, RHAZ0* stay.
- `VERSION_TAG` v0p94 (the processing manifests of v0p93 are read and reused).
- **Chal Rocks tiny** without station S004D0218 and sequence ZCAM08390 (site `EXCLUDE`, this block only); a site's or
  run's `EXCLUDE` entries now keep their images out of the processing too.
- **A first Mastcam-Z calibration without low-state bins gets the default high / low ratio** (1.0095), not 1.0000
  (`ztied.candidate_model`, `start_model_from_blocks`): the Z63 joint of 23 Oct failed only its ratio check on that.
  A 110 mm model needs no low-state lines in its form (`forms.validate_zcam`: the Z110 joint failed "form").
- **Models in use** (Christian: promote what is available):
  - Mastcam-Z **63 mm** (first calibration): the joint `zcam63_joint_20261023` (chal_rocks, marble_mountain; 12 / 14
    bins; f0 8624.7 / 8623.7 px, aspect 1.00003 / 1.00023, rig 21.7 mdeg from the start, residual median 0.22 px),
    its low-state ratio set to the default 1.0095; every check passes.
  - Hazcam **FL athermal**: fx 2141.07, fy 2140.44 px, k1 +0.0105 constant over the 4 blocks of
    `hazcam_fit_zcam_20261023` (block scatter 538 ppm), every check passes.
  - Not promoted: Z48 (one bin per eye), Z79 (2 bins per eye, rig 0.70 deg from the start, fy / fx 0.992), Z110
    (fy / fx 0.982 - beyond the 1 % aspect limit; two blocks), RL / RR (scatter 1600-2500 ppm, athermal too).

## 0.93.10 — 2026-10-23

- **Hazcam eyes fall back to an athermal law** (Christian): when an eye's temperature law fails its thermal check,
  the fit is made again with no temperature terms (fx, fy and k1 constant; `athermal`, `fallback`, the temperature
  fit kept in `thermal_fit`) and that law is the candidate if it passes every check - promoted with the others. If it
  fails too, the temperature law stays (failed) and the attempt is recorded (`athermal_tried`). Replayed on the fit of
  23 Oct 08:49: FL (+151 ppm/degC) passes athermal with a block scatter of 538 ppm (limit 600); RL and RR fail
  athermal too (scatter 2541 / 1874 ppm); FR keeps its temperature law (+67 ppm/degC).

## 0.93.9 — 2026-10-23

- **LightGlue is the matcher** (Christian, 23 Oct): `match_mode` "lightglue" (SIFT + LightGlue on the GPU through
  kornia; COLMAP's SIFT matcher when torch / kornia are missing or the block's projected time is over 2 h; the
  prior-overlap pairs above `max_exhaustive_images`). The run key changes: every block aligns again.
- **`test_lightglue.bat`** (`scripts/lightglue_ab.py`): the A/B test on **Chal Rocks tiny** (`chal_rocks_tiny`, sols
  91-98 only, Navcam + Mastcam-Z + Hazcams; in no group): a GPU check (torch, kornia, CUDA), the block processed once
  and aligned twice - COLMAP's SIFT matcher into `colmap_sift`, SIFT + LightGlue into `colmap_lightglue` - then
  `<WORK>/lightglue_ab.md`: the matcher that ran, pairs, inlier matches, matching and total time, registered images,
  residuals and every health check side by side.
- **`run_pipeline.bat` is the whole run again** (Christian): wait, `navcam_runs` (run_sites over navcam_consensus),
  `navcam_joint` (the tied joint, promoted when its checks pass), `zcam_runs` (run_sites --zcam over zcam_consensus
  and the other zcam34_joint sites), `zcam_joints` (the Hazcam fit, the Mastcam-Z joint of every zoom, the N+Z
  check), `error_analysis` (studies/analyses/error_analysis_batch.py over every finished block: epsilon intra /
  cross station, by range and angle, the cross-station gate, rho(theta), the view graph, the registration ->
  `camera_analysis/error_analysis_<date>/report.md`), report. The old stages (consensus passes, baseline study,
  Z34-only calibration from per-block lines, found list) are gone; the old state file is kept aside. Options:
  `--no-promote`, `--no-hazcam-fit`, `--no-nz`, `--zooms`, `--from`, `--stop-before`, `--redo`. The error-analysis
  batch goes on past a block that fails. `run_v0p91.bat` (the old pipeline's options) removed.
- **Mask inference off** for `ZCAM08060` too at Chal Rocks, Chal Rocks Large and Chal Rocks tiny (with NCAM00699,
  ZCAM00034), and for every `RHAZ0*` sequence (rear Hazcams) there and at Bell Island and Bell Island Large
  (Christian, for now). A `no_mask_inference_at` entry may be a sequence prefix: `RHAZ0*` (or `RHAZ0`).
- `zcam_consensus` (Christian): chal_rocks_large in place of chal_rocks, and taylorfjellet added; `zcam34_joint` too
  takes chal_rocks_large (so the Z34 joint does not hold the same Chal Rocks frames twice).
- sites.json (Christian): `crocodile_bridge` has a Mastcam-Z **110 mm** 360, not 34 mm (`"zcam": [110]`);
  `observation_mountain`'s label "Observation Mountain" (was "Hidden Harbor Observation Mountain"); Origny and
  Origny Large stay Navcam only (no Mastcam-Z 34 mm 360, 22 Oct).
- **Models in use** (Christian: promote what is available): the PC's promotions of this morning, byte-identical -
  the Mastcam-Z 34 mm joint of 8 blocks (`zcam34_joint_20261023`, every check passed; low-state ratio 1.0146 ZL /
  1.0139 ZR, rig 1.8 mdeg from the one before) and `CHANGES.md`; and the Hazcam **front and rear rigs** of the Hazcam
  fit of 6 blocks (`hazcam_fit_zcam_20261023`; each passed its own check: rear 0.267 deg from the CAHV rig, front
  0.026 deg). Not in the repository: `M2020_FR.json` (promoted on the PC at 08:55, kept there untracked - a tracked copy
  would stop the sync's fast-forward); FL, RL, RR failed their checks; the 48 mm joint had one bin per eye (89
  observations) and is not usable; 63, 79, 110 mm were still running.
- **The Mastcam-Z joint keeps its result when a check cannot run** (`zcam_joint.json` first; the check is then a failed
  check) and the rig-scatter check takes a scatter that could not be measured (Z48, 23 Oct: a crash after the
  adjustment).
- **Excluded** (bad colours, Three Forks North front Hazcams): `*_0681_0727400287_565*`, `*_0684_0727664729_880*`,
  `*_0693_0728459809_940*`.

## 0.93.8 — 2026-10-23

- **`no_mask_inference_at` takes sequence ids** (Christian): besides stations (`S032D1184`) an entry can name a
  sequence (`NCAM00699`, `ZCAM00034`, also `seq:NCAM00699`); the mask model is off for every image of it
  (`config.mask_inference_skipped`, `parse_sequences`; the processing reuse check follows it, so only those images
  are processed again). Chal Rocks and Chal Rocks Large: `["NCAM00699", "ZCAM00034"]`; Three Forks North:
  `["ZCAM05086", "ZCAM08692", "NCAM02688"]` (was `["S032D1184"]`). The global default is now empty (it was
  `["S032D1184"]` for every site).
- **Kept**: `!seq:ZCAM01023`, `!seq:ZCAM01025` (the `seq:ZCAM01*` exclusion no longer drops them).
- **`zcam_joints.bat` runs one copy at a time** (a lock folder `camera_analysis\zcam_joints.lock`): on 23 Oct two
  copies started 2 s apart ran the Hazcam fit and the Z34 joint twice into the same folders; the second promotion
  failed with "WinError 32 ... being used by another process". `set MPPP_ZOOMS=none` runs only the Hazcam fit (and
  the N+Z check unless `MPPP_NZ=none`). A promotion copy now waits out a file held for a moment (20 tries), writes
  through a temporary file, and a failed Hazcam promotion says so instead of a traceback.
- **A Hazcam rig is promoted on its own checks**: the rear rig (0.267 deg from the CAHV rig, five blocks) was held
  back because the rear eyes' laws failed (thermal slope, block scatter 1600-2500 ppm). The blocks hold the
  principal points at their start, so the rig rotation does not depend on the eyes' focal laws.
- `promote_models.bat`: drag model files or a candidate folder onto it (`mppp cmods promote`).
- **LightGlue on a big block gets the prior-overlap pairs** (`match_mode="lightglue"` with more than
  `max_exhaustive_images` images): it matched every pair (564,453 at Marble Mountain's 1063 images), which is over
  its 2 h budget, and then fell back to COLMAP over every pair.

## 0.93.7 — 2026-10-23

- **The Mastcam-Z joints take the low backlash state from the shipped low-state list only** (Christian, 23 Oct): a
  joint does not judge states again - it takes each camera's state from its block - and the blocks run before
  0.93.6 also put the found list's frames in low-state cameras. A low-state camera now enters a joint only when every
  one of its frames is on the shipped list (`low_cameras_not_on_list` in the result and the report; `ztied.
  LOW_LIST_CHECK`). With the split cameras left out (0.93.6), the 22-23 Oct blocks give the joints only listed
  frames in the low state; every other frame is in the high state. Seitah Overlook: of its nine low-state cameras
  only ZR034_F01296_reg (3 listed frames) enters.

## 0.93.6 — 2026-10-23

- **A frame is in the high backlash state unless it is on the low-state list** (Christian, 23 Oct): the default
  `zcam_backlash` is `"list"` (was `"list_split"`) - the list's frames get their own low-state cameras at the start,
  and a block's focal fits only report groups that look low; nothing is split off or written to the found list. The
  found list is no longer read (`zcam_low_backlash_found` None, was "auto"): since 22 Oct it held the five Z34 left
  pointings of Seitah Overlook ZCAM08114, two of them high-state frames. Entries worth keeping belong in the shipped
  list.
- **The Mastcam-Z joints are safe on blocks run before this version**: cameras a block split off by its focal fits
  (`state_split`) are left out (`split_cameras_left_out`) - before 0.93.4 they could hold high-state frames under a
  low-state name. With 0.93.4-0.93.5 (seitah_overlook and chal_rocks judged again to WARN, the low state's aspect
  tied and its ratio bounded, 110 mm single-state) the joints can run on the 22-23 Oct blocks.
- **`zcam_joints.bat` fits the Hazcams first**: `mppp calib hazcam --zcam-blocks --group zcam_consensus` (the
  sites' `_zcam` blocks, whose Hazcams and rigs are refined), promoted when its checks pass (`MPPP_HAZ=none` skips
  it). The rear Hazcam rig moved 0.27-0.28 deg from its start in all six blocks (22-23 Oct).
- **A Hazcam rig is judged against its own limit**: `hazcam_rig_rotation_change_deg` warns at 0.06 and fails at
  0.5 deg (the Hazcam fit's limit); the Navcam and Mastcam-Z rigs keep 0.3. Its start is the label CAHV rig until
  the Hazcam law is promoted - Marble Mountain's rear rig moved 0.304 deg and failed the block (23 Oct), which kept it
  out of the joints. Reports from before are judged again the same way at the joint gate: all six blocks of 22-23
  Oct enter the joints (chal_rocks, seitah_overlook and marble_mountain were FAIL).
- Joint reports say "high backlash state" and "low backlash state".

## 0.93.5 — 2026-10-22

- **110 mm has no focus state** (Christian): its images and cameras are `"single"` everywhere -
  `calib.focus_state(listed, group)`, the selection's bins (`backlash_state`), the block reports, the joints and the
  checks. The low-backlash lists never apply to it, no regular ratio, regular lines or regular principal point are
  derived or written for it (`fit.regular_ratio` returns none; the joint's candidate has one set of lines, its bins on
  the backlash terms), and `check_zcam` fails a 110 mm model with a regular state. Its frames enter every block
  without a state of their own and so anchor the per-state corrections of the other zooms.
- **One rule for a camera's state** (`calib.zcam_state`): single at 110 mm, else the camera's own `backlash_state`,
  else `_reg` at the end of the key (also `<bin>_2_reg`). It replaces four copies that disagreed (an explicit
  "backlash" beat a `_reg` key in one, not in the others). Every selection bin now records its state.
- **The shipped Z34 model's regular state repaired**: the 22 Oct joint (8 ZL / 3 ZR regular bins, mislabelled
  states) wrote a regular ratio of 1.024 (regular fy / fx 0.984: fx and fy of the regular state were free on their
  own). The model before it had 1.009 (ZL) / 1.0109 (ZR); those are restored and the regular lines are the backlash
  lines divided by them (`regular_ratio_note`). Blocks start their listed `_reg` bins on these lines, and bins of up
  to 2 images hold f there.
- **The joint's regular state scales f only**: `a_fy_regular` is tied to `a_fx_regular` times the backlash aspect;
  the ratio is fx-based and kept at the start value when the fit leaves 1.004–1.016 (`regular_ratio_note`).
  `check_zcam` adds "regular ratio" and "regular aspect" (within 5e-4 of the backlash aspect).
- **Bugs found in a sweep of the state code**:
  - `split_by_state` split cameras already in the regular state: in `split` mode the listed `_reg` bins moved to a
    `_reg_reg` camera, the `_reg` bin was marked backlash with 0 images and the principal-point offset added twice.
    Only backlash-state cameras are split now.
  - A project built without a list (`zcam_low_backlash=False`) got the shipped list back when its groups were
    classified; the project now records `"disabled"`.
  - The sequence rule gave a failed (implausible) group its sequence's state; such a group could then be split and
    written to the found list. It keeps its own state now.
  - After a split every other Mastcam-Z image was labelled "backlash" (the listed regular images and 110 mm too):
    each now takes its camera's state.
  - `strip_thermal_bins` left a split's base camera with its reduced image count and state, so each rerun lowered
    it again: the values before the split are kept (`pre_split`) and restored.
  - The measured regular ratio (`fit.regular_ratio`) counted groups the sequence rule had moved and partly listed
    groups; it now uses groups in the state of their own fit and the listed images' own ratio.
  - `refit_focus_lines(apply=True)`: a regular bin set on the backlash lines lost the regular principal point.
  - The start model of a new zoom (`ztied.start_model_from_blocks`) averaged regular bins into the backlash lines.
  - A split regular camera kept its base's held f; its f starts at the labels and is now free (unless it has at most
    `hold_f_images` images). The bin fallback takes the group's backlash-state camera, not the first by name.
- **A Hazcam eye refines its intrinsics from 500 tie-point observations** (Christian; was 1000:
  `HAZCAM_MIN_REFINE_OBSERVATIONS`).
- `ALIGN_RULES` 89; with the model repair every block aligns again.

## 0.93.4 — 2026-10-22

- **A partly listed Mastcam-Z focus group holds both states** (Christian, Seitah Overlook ZCAM08114, sol 121): a
  five-pointing mosaic with one stereo pair per zoom at one focus count, pointings 1, 3 and 5 in the low-backlash
  (regular) state and on the list, 2 and 4 in the backlash state. All five form one focus group; its unlisted images
  fit f / label f 1.000 against 0.994 for the listed ones, below the absolute midpoint (1.0047), so the group was
  called regular and 2 and 4 were moved into a regular camera. The list now marks the group's regular exceptions:
  the unlisted images are judged against the listed ones of the same group (midpoint = listed ratio + half the
  block's state gap) and are backlash unless they clearly fit like the listed ones, in which case the group keeps its
  own call - the rule never makes a group regular (`mixed_group_state`; `ratio_listed`, `mixed_group`, `mixed_note`
  in the group rows). Replayed on the Seitah block: all 8 groups (Z34, Z48, Z63, Z79, both eyes) become backlash.
- **The state split never overwrites a camera**: the new regular camera of a split was always `<bin>_reg`, the name
  the selection already gives the list's own regular bin, so the split replaced that camera's entry and database id
  (7 cameras at Seitah Overlook, 2 Z79 cameras in the 0.93.1 Chal Rocks block). A taken name now gets
  `<bin>_2_reg`, `<bin>_3_reg`, ... (still ending in `_reg`).
- `ZCAM_RULES` 1: a new part of the run key of the Mastcam-Z blocks only, so every `_zcam` block aligns again and
  the Navcam-only blocks keep their keys.

## 0.93.3 — 2026-10-22

- **`open_in_colmap.bat` finds COLMAP again on the MPPP PC**: the 0.90 cleanup dropped the `D:\tools` folders from
  `COLMAP_BAT_CANDIDATES`, so the GUI files of blocks run since then said "COLMAP.bat not found" unless `COLMAP_BAT`
  was set. The candidates now include the official release zips unpacked in `C:\tools` or `D:\tools`
  (`colmap-x64-windows-nocuda`, `colmap-x64-windows-cuda`, `COLMAP`). `mppp gui` rewrites the files of finished blocks.
- **chal_rocks: sol 91 first** (Christian): `"sols": [[91, 101], [361, 378]]`, so the block folders are
  `mars2020_sol_0091_chal_rocks[_zcam]` (were `0361`). `move_scapes.bat` (`studies/calibration/move_scapes.py
  --apply`) renames the existing `mars2020_sol_0361_chal_rocks_zcam` and rewrites its project paths, so the next run
  reuses its processed images.
- **A block's own stations on other sols are selected** (Christian): an image taken at one of the block's stations on
  a sol outside its sol ranges (the rover had not moved yet) was left out - the nearby-waypoint search adds only
  other stations. Chal Rocks lost its sol 102 Mastcam-Z sequences at the sol 99 station S004D0218 (ZCAM08062, 8 Z34
  pairs; ZCAM08061, 75 Z110 pairs). `find_imgs_near` now adds them (`own_station_other_sols` in the report; the log
  lists them; `NEARBY_FROM_FIRST_SOL` still applies), for the Navcam, Mastcam-Z and Hazcam selection.
- **Thresholds** (Christian, 22 Oct): a frame is an outlier when its median residual exceeds 3 x the block's and
  2.0 px (`OUTLIER_DEFAULTS["min_residual_px"]`, was 1.6 - Chal Rocks lost 17 Z79 frames at 1.6-1.9 px); the health
  check fails a block at 30 % frames excluded (`excluded_fraction`, was 20 %) and at a stereo-rig rotation change of
  0.3 deg (`rig_rotation_change_deg`, was 0.2; the rear Hazcam rig moved 0.26 deg at Chal Rocks). `ALIGN_RULES` 88
  and `PROCESS_RULES` 5 (the own-station selection): every block selects and aligns again (processed images reused).
- `zcam_consensus`: seitah_overlook, chal_rocks, threeforks_north, belva_crater, south_arm, marble_mountain.
- **Kept**: `!seq:ZCAM00015` (the `seq:ZCAM00*` exclusion no longer drops it). **Origny** has no Mastcam-Z 34 mm
  360 (Christian): `origny` and `origny_large` are Navcam only (`"zcam": []`) and official block 35 no longer lists
  360 #51 (the maps' 360 table, `studies/m20_sites/m20_sites/refdata.py`, dates #51 "Olifants" sol 1775, inside
  Origny's square - its sol is probably wrong).
- **Corner check without the uncalibrated Mastcam-Z zooms; FAIL blocks judged again at the joint gate**: the
  corner-coverage health check (keypoints beyond 0.85 of the corner radius that become tie points) leaves out the
  Mastcam-Z zooms without a promoted focus model, as it does the Hazcams - their distortion is the label start and
  their few-frame mosaics barely overlap at the corners (Seitah Overlook, 22 Oct: ZR063 with 5 images was the
  block's only FAIL, which would have kept it out of every joint). `mppp calib` judges a block whose run recorded
  FAIL again from its `health/health.json` under the current thresholds and rules (`health.rejudge`) and takes it
  when it no longer fails (logged); `--include-failed` still takes every block.
- **Residual summary**: `residual_rms_native_px` and the median use the finite residuals only (`residual_nonfinite`
  counts the others: a point behind a camera or outside a lens model's domain). The 22 Oct chal_rocks block reported
  an RMS of Infinity.

## 0.93.2 — 2026-10-22

- **Each backlash state its own camera, the pose shared** (Christian): the Mastcam-Z regular (low-backlash) state
  gets its own principal point - `d_cx_regular`, `d_cy_regular` per eye in the Mastcam-Z joint (`ztied.GLOBAL_TERMS`,
  now 12), tied to every regular-state bin (cx = pin + d_cx, cy = pin + b_cy (focus - ref) + d_cy) - with the exposure
  pose and the Mastcam-Z rig shared by both states, so the state no longer moves the frame rotation or the rig. The
  state is per eye (Three Forks North: 96 ZL images regular, their ZR partners backlash); the exposures with one eye
  regular and the other in backlash make the offset observable, and it is fitted when an eye has at least 5 of them
  (`MIN_MIXED_EXPOSURES`, `mixed_exposures` in the result), else held. The candidate model writes it as the regular
  state's cx, cy lines (pp0 + the offset, both eyes) and `regular_pp_offset_px` (with sd, held flag and the
  exposure count); `zcam_camera_at` and the block bins read the regular state's principal point from those lines, the
  backlash split (`split_by_state(pp_offsets=)`) starts a new regular camera at it, and `check_zcam` adds
  "regular principal point" (|offset| <= 10 px). Models without it behave as before (offset 0).
- **Why**: in the 22 Oct Z34 joint, chal_rocks - the only block with regular-state images (8 % of the observations
  in each eye) - was the outlier of the per-block rigs (pitch -28.7, yaw -11.9 mdeg; the other five -3.5..+5.3 and
  -6.5..+2.9) and anchors the rig's yaw trend (+6.4 +- 2.1 mdeg / 1000 sol): with one principal point for both
  states, a boresight change of the regular state can only go into the rig. Within a block the state does not change
  the frames' world attitude (Three Forks North, the same sequence: regular - backlash (+14, +17, -18) +- (19, 24, 18)
  mdeg). In a synthetic joint (`test_zcam_joint_gives_the_regular_state_its_own_principal_point`) a (+6, -4) px
  regular offset in a third of the ZL images moved the rig by (-18, -22, +9) mdeg and left 0.57 px residuals with the
  old form; with the offset fitted it is recovered to 0.01 px and the rig to 0.03 mdeg (0.055 px residuals).
- **Measured on Three Forks North** (block re-adjusted from its result with every bin's principal point held, then
  with the ZL regular bins' free): the four regular bins with 17-22 images each move cx +0.5..+1.1 px, cy 0..-0.4
  px, all the same way, and the block's Mastcam-Z rig yaw moves by ~8 mdeg with them (0.8 px at f 4700 px) - the
  size of chal_rocks' -12 mdeg. A 4-image regular bin runs away with its own principal point free (+71, +34 px),
  which is why the joint fits one offset per eye from all regular bins.
- The run keys change only with a promoted model that carries the offset (the next Z34 joint).

## 0.93.1 — 2026-10-22

- **Hazcams in every block** (Christian, 22 Oct): `hazcams` defaults to both pairs (`("F", "R")`), so the front and
  rear Hazcam frames of a block's stations are selected, processed and aligned with the Navcam (stage 2, Navcam held,
  as in 0.93.0). A Hazcam run no longer goes to `colmap_hazcam` by itself: the results are in `colmap/` like any
  block. `--hazcam none` (new choice; `--set HAZCAMS=none`) leaves them out. The Hazcam rig rotations are refined by
  default (`hazcam_rig_refine = "rotation"`); a pair whose two eyes are held as weak (< 1000 observations) keeps its
  start rig (`settings["hazcam"]["rig_hold"]`, log `hazcam_rig_held`). `ALIGN_RULES` 87: **every block realigns**
  (and reprocesses to add the Hazcam products; the Navcam images already processed are reused).
- **Hazcam law in the Navcam form** (`mppp calib hazcam`, `mppp.calib.hazfit`, new): from the finished blocks'
  refined Hazcam cameras and rigs, per eye fx and fy with one ppm/degC slope about T0 = -20 degC, an early-mission
  offset (before sol 380) on fx, fy, k1 linear in temperature, k2 the blocks' mean (cx, cy, p1, p2, k3, k4 the label
  fits' median); per pair the rig rotation at sol 0 with a linear drift in sol (>= 3 blocks with a refined rig).
  Weak eyes are left out. Each eye and pair is checked on its own (`HAZCAM_LIMITS`) and the passing files
  (`M2020_FL.json` ..., `M2020_F_rig.json`, `M2020_R_rig.json`) are promoted with `--promote-if-ok`. A block reads
  them as it reads the Navcam's (thermal law at the block's median temperature, rig at its median sol).
- **`run_hazcam.bat`** (new): aligns the `hazcam_consensus` group (new: chal_rocks_large, threeforks_north,
  belva_crater, bell_island, marble_mountain) with both Hazcam pairs and the Navcam held at its promoted law
  (`NAVCAM_INTRINSICS=hold`, variant `hazcam` -> `colmap_hazcam/`), then `hazcam_calibration.bat` (the fit,
  promoted if it passes).
- **`run_sites_zcam.bat` now ends in the joints**: it aligns `zcam_consensus` (now chal_rocks and marble_mountain;
  chal_rocks_large, taylorfjellet and olifants out; `zcam34_consensus` has chal_rocks_large in place of chal_rocks) at every zoom of each site, then `zcam_joints.bat` (new) runs
  `mppp calib zcam --promote-if-ok` for 34, 48, 63, 79 and 110 mm (`MPPP_ZOOMS`; 34 over
  `zcam34_joint,zcam_consensus`, the others over `zcam_consensus`), the Z34 figures and `mppp calib nz` with
  `--navcam-group navcam_consensus` (`MPPP_NZ=none` skips it; `MPPP_THEN=none` stops after the alignment).
- **Fixed: weak Hazcam eyes drifted in Mastcam-Z blocks.** An eye with too few ties was held in the stage-2 rounds
  only; the pose-guided adjustment and the Mastcam-Z backlash split (both only with Mastcam-Z frames) refined it
  (FL, FR +1.2 % in a sol 709-711 Navcam + Mastcam-Z + front Hazcam test block), and it was then held at the
  drifted value. A weak eye is now put back to its start camera and held in every later adjustment
  (`reconstruction.reset_cameras_to_start`, `backlash_stage(hold_cameras=...)`), and the Hazcam rigs are held in
  round 1 with the eyes. The test block afterwards: FL, FR 0 % change, F rig 0 deg; all Mastcam-Z zooms (34, 63,
  110) and the front Hazcams registered together (Z48 had no tie points).
- **Fixed Mastcam-Z distortion terms per zoom** (`forms.ZCAM_FIXED_TERMS`): Z34 k2 = 0.5 (ZL), 0.6 (ZR), p1, p2 free;
  Z48 k2, p1, p2 free; Z63, Z79, Z110 k2 = p1 = p2 = 0. The block BA holds them in the bin cameras, the joint holds
  the global terms at them (`ztied.fixed_globals`) and its likelihood-ratio tests add "the fixed terms free"; the
  per-image prior keeps the promoted model's own values (its k1 belongs to its k2).
- **Mastcam-Z joints for new zooms**: a zoom without a focus model starts from the blocks
  (`ztied.start_model_from_blocks`: per eye the median bin camera with one focus slope, the median rig); its first
  calibration needs 2 blocks (`min_blocks_start`), later ones 3. The joint accepts blocks aligned with an older
  Mastcam-Z model (the check is now "aligned with the Navcam models in use"). `--group` takes comma-separated
  groups; zcam / nz keep the sites whose `zcam` list has the zoom. Zooms with one block run but are not promoted.
- **`mppp calib nz --navcam-group G`** (new): adds the Navcam-only blocks of group G (3000 points each) so the
  combined Navcam law is held by every Navcam block, for comparing the Navcam-only and Navcam + Mastcam-Z fits.
- **Z34 joint review (22 Oct, chal_rocks, enchanted_lake, marble_mountain, rochette, south_arm, taylorfjellet;
  promoted on the PC at 11:02 and committed here unchanged)**: k2 0.484 +- 0.010 (ZL), 0.615 +- 0.009 (ZR), k1-k2
  correlation 0.98, ZL and ZR curves within 0.06 px - hence the fixed values; the k2 = 0 test is significant but
  moves the curve by <= 0.27 px; ZL p1 = +3.1e-3 (near the 5e-3 limit); f0 +0.16-0.19 %; Mastcam-Z fy / fx
  1.0008-1.0010 in both eyes (suspicious for square pixels: an aspect term may be absorbing a principal-point or
  attitude effect); rig yaw +6.4 +- 2.1 mdeg / 1000 sol (3 sd), roll -2.6 +- 1.0, no temperature term. The combined
  N+Z check moved NL cx by -1.3 px (38 sd), NL and NR fy by about +250 ppm and the thermal slope from 36 to 41
  ppm/degC while the Navcam residual changed by only -0.7 %: the six blocks alone do not pin the Navcam principal
  point against attitude - `--navcam-group` is the test.
- **Official blocks run by the plain sites** (Christian): block 3 Chal Rocks by `chal_rocks` (was
  `chal_rocks_large`), block 18 Belva Crater by `belva_crater` (was `belva_crater_large`); `blocks_all` and the region
  groups follow. `waypoints.blocks.restructure` prefers the plain site over its `_large` / `_huge` form when both
  list the block.
- **sites.json**: `delta_front_huge` was defined twice (the same sols, two notes; JSON kept the second silently) -
  the second copy is removed. `mppp sites check` (`validate_site_table`) now reports a key defined twice in one
  object as an error.
- **Excluded stations leave the selection** (Christian, 13:08): rockytop drops the sol 460 mid-drive waypoint
  S026D0282 (site setting `EXCLUDE`, as seitah_huge's S009D1064). A station in `exclude` (`S###D####`) is now left
  out of the product selection and does not anchor the nearby-waypoint search (`find_imgs_near(exclude_stations=)`),
  so the two sol 456 waypoints within 10 m of S026D0282 (S026D0218, S026D0160) no longer join rockytop; rochette
  drops station S006D2594 (sol 178 mid-drive, not in the waypoint table); the waypoint
  fields honour it too (rockytop 58.7 -> 23.0 m, 13 -> 10 waypoints). `zcam_consensus` gains seitah_overlook
  (zooms 34, 48, 63, 79, 110: a second block for Z48, Z79 and Z110, so their first calibrations can be promoted).
  Shipped exclusions: `seq:ZCAM05069`, `seq:ZCAM05070`, `seq:ZCAM05071` (`seq:ZCAM05114` was already listed).
- **Shipped exclusions** (`mppp/data/exclude_images.txt`, Christian): `seq:ZCAM05060`, `seq:ZCAM03017` and the sol 76
  exposure `*_0076_0673685057_000*` (every camera at that SCLK); `seq:ZCAM05062` was already listed.
- **Figures**: the LMST histograms are laid out in reading order (first row = the first two blocks), lettered A, B,
  ..., with the legend just above the bottom all-sites panel. The map banner (figure 7) is the terrain with "Mars
  Photogrammetry Preprocessing Pipeline" centred, without traverse, markers, labels or legend (`M20_BANNER_TEXT`,
  `M20_BANNER_TEXT_Y`, `M20_BANNER_SCALEBAR=1`).
- Tests: `tests/test_hazfit_zfixed.py` (fixed terms, joint globals, the Hazcam fit, check, files and `camera_at` of
  the result, scripts, groups, CLI); Hazcam default and Z34 tests updated.

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
