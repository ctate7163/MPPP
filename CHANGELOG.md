# Changelog

The development history before 0.95 is in `docs/history/` (`CHANGELOG_v0p90-v0p94.md`, `CHANGELOG_v0p13-v0p85.md`,
`CHANGELOG_v0p1-v0p12.md`).

## 0.96.0 — 2026-11-10

**In short.** The core and the studies apart (Christian, 9 Nov: "The core is the pipeline ... The code that solves
the camera models from the blocks or deck alignments should be a mature study ... the tie point error analysis ...
Training the mask model should also be a study"): `mppp` is the pipeline alone - PDS products in, the waypoints,
the promoted camera models, processed images, the COLMAP block, its health, the dense cloud, the exports - and the
research lives in `mppp_studies` (`calibration`, `error`, `mask_training`, `sites`, `paper`, `misc`), which imports
the core and which the core never imports.  Dead code out of the core.  Two new products of the pipeline: per-image
depth maps and the photometric cube from a cleaned station mesh, and a prototype of the stereoscopic panorama
(`mppp mesh`, `mppp depthmaps`, `mppp ods`).  The freeze run as a driver of its own (`freeze.bat`) with the mask
comparison in front of it, and one way to sync code (`sync_code.bat`).  One pipeline notebook and one per study.
Below them, the Mastcam-Z joint diagnostics of 5-9 Nov (items A-F), the mask augmentations and the site-map
changes made on the way to the build.

- **The split** (`src/mppp_studies/`, 1/n). `mppp.calib` keeps what the pipeline reads and the forms the studies
  share (`forms`, `models`, `constrain`'s thermal and focus laws, `labelfit`, `tiecov`, `tiesolve` with the linear
  tie class, the new `robust` - Huber fits, weighted medians, `meta_line`, the OLS line - that `sfm.backlash` and
  `sfm.zcam` use); the joints, the tied fits, the Hazcam and NZ joints, the law check, the fiducials, the deck
  adjustment, the backlash states, the pipeline runner, the first freeze chain, the promotions, the plots and the
  block screening moved to `mppp_studies.calibration` (`python -m mppp_studies.calibration calib|zcam-states`;
  `mppp calib` and `mppp zcam-states` forward there, `_study_parsers` / `_study_command`); `mppp.error` and the
  error batch, the Navcam and cross-station studies and the site map became `mppp_studies.error`; the mask training
  (train, dataset, audit, monitor, augment, `train_mask`, `compare_mask_models`, `make_mask_figure`,
  `release_mask_model`) `mppp_studies.mask_training` - `mppp.mask` keeps the inference and the hub; the M20 site
  figures, the maps and the waypoint figures `mppp_studies.sites` (`mppp maps` forwards; `python -m
  mppp_studies.sites maps`); the paper's figures and build `mppp_studies.paper`; the old calibration study, the
  experiments, the v0.52-53 navcal and the one-off scripts `mppp_studies.misc` (kept, not maintained). The one hook
  the pipeline keeps is lazy: `pipeline.deck_in_ba` imports `mppp_studies.calibration.deck` when a block asks for the
  deck in its adjustment, with a plain message when the studies are not installed. `mppp_studies.paths`
  (`repo_root`, `outputs_dir`, `docs_dir`): where the studies write. README, CONTRIBUTING, `docs/WINDOWS.md`,
  `docs/methods.md`, `docs/cli.md` follow the move.
- **Dead code out of the core** (2/n; vulture at 80 %): `forms.distortion_names`, `colmap.station_center`,
  `labels.as_vector`, `select.find_imgs_in_sol_range` / `find_imgs_for_waypoint` / `dedupe_by_version`,
  `terrain.shaded_relief`, `camera.is_distorted` / `project_cahv`, `runner.python_literal`,
  `project.zcam_focus_model_files`, `reconstruction.print_camera_changes`, `sites.zcam34_sites`,
  `blocks.site_official_blocks` / `rename_site` / `remove_site` / `site_waypoints`, `cli._work_of`, the waypoint
  figures of `waypoints.blocks`; `constrain`'s plots and `health`'s re-judging into the calibration study. pyflakes
  clean (`flake8 --select F src/mppp`); the split's loose ends it found fixed (`run_rover_frame`'s root,
  `study_navcam`'s data folder, `screening`'s named imports).
- **The freeze driver** (3/n; `mppp_studies.calibration.freeze`, `scripts\windows\freeze.bat`; Christian, 9 Nov:
  "please prepare a freeze.bat that will run on 0.96"). The freeze-2 steps - wait, cmods, zjoints, zfinal, ecam, nz,
  deck, ztests, studies, report - on the 0.96 layout, with the first chain's robustness (atomic status with a
  heartbeat, one run per root, keep-awake, RunOnce after a reboot, a disk-space wait, a failed step logged and the
  run goes on) and three fixes from the 9 Nov restarts: a stop by the user (Ctrl-C, `stop_mppp.bat`) is logged
  and `--status` says "stopped by the user"; `--status` tells running / not running / done apart (heartbeat and
  pid); `--update-code` only updates the checkout (`git pull --ff-only` from `D:\code\MPPP`) and refuses under a
  live run. New step **mask** (`--mask-model keep|auto|FILE`; Christian: "can you delay the start until the mask
  comparison is done and we know which CNN is best?"): `keep` is the model the blocks on disk were processed with
  (read from their `run_settings.json`; no image processed again); `auto` waits for the training under way and its
  comparison and uses the new model when it beats the blocks' model by `--mask-margin` (0.01) exclusion IoU - then
  every image is processed again and every block aligned again (the checkpoint is in the run key). The choice is
  kept across restarts. The run requires MPPP 0.96.x in its checkout (`--any-version`). `freeze.bat` runs it on
  its own checkout `D:\code\MPPP_freeze`, cloned from the main tree the first time. **`sync_code.bat`** (replaces
  `claude_sync.bat`; Christian: "let's clean up the MPPP directory so that these conflicts don't keep happening"):
  a delivered bundle or GitHub, one fast-forward of `main`, the quick tests one file per process with MKL
  sequential (torch's and MKL's OpenMP in one process aborted the 9 Nov runs), the freeze checkout brought up
  through the driver unless a freeze runs on it; nothing started, nothing pushed. `freeze_chain.bat` retired (the
  first chain's module stays, tested); `scripts\windows` stays at 25 files. **`train_mask --compare [MODEL ...]`**
  (`train_masks.bat` forwards it; Christian: "will the mask comparison run after the training so that we will know
  which is better?"): right after the training, `compare_mask_models` of the new checkpoint against the blocks'
  model (`mppp_mask_v3`) on its held-out frames -> `Claude outputs/mask_<ckpt>/mask_ab/` and `mask_verdict.json`
  (exclusion IoU per model, winner, margin), which the freeze reads. `mppp doctor` prints the checkout's git head,
  the studies package and the live runs on `MPPP_SCAPES`. Tests: `tests/test_freeze.py` (14).
- **`mppp.ibr` - the station mesh, the depth maps, the photometric cube, the stereoscopic panorama** (4/n;
  Christian, 9 Nov: "0.96.0 should build the pipeline that generates the per image depth maps and photometric cube
  from the cleaned up 3d mesh. 0.96.0 should prototype the stereoscopic pipeline. The output should just be two
  images. Don't build the webviewer"; the plans of 29 and 31 Oct). `station`: the station of a block (site, drive),
  its frames by camera family, the mast pivot from their centres and rotations (least squares C_i = P + R_i^T d_g
  per family; the mean centre below 6 frames or 60 deg of azimuth), the block's up / north / east from
  `error_input/block_frame.json` or the sun model. `mesh`: Poisson (open3d; COLMAP's Delaunay mesher through the
  executable when given) over the fused cloud, then the cleanup - components under 1 %, rubber sheets, surface no
  fused point supports within tau(z) = max(5 cm, 2 % z, 3 sigma_z), the 60 m radius - decimation to 3 M
  triangles, Taubin smoothing, the far sphere at 100 m; `mesh_report.json` with the coverage by range, a preview.
  `raycast`: open3d's RaycastingScene over mesh + sphere; per image the depth, range, normal and label maps (0
  none, 1 mesh, 2 far) through the camera's own model (`cam_from_img` rays) at a fraction of the native size; npz
  and 16-bit previews; validation against the sparse points the image observes; `maps_report.{json,md}`.
  `photometry`: incidence, emission, phase and the cast shadow from the sun direction of the image's row
  (`sun_enu` through `R_block_from_enu`); npz, COLMAP `.bin` arrays, previews, `photometry_report.json`. `ods`:
  the omni-directional stereo panorama (equirectangular 8192 x 4096, 64 mm between the eyes): each eye's rays cast
  into the scene, every hit coloured from the source frame that sees it best (the smallest angle to the frame's
  axis, inside the frame and its mask), chunked by rows; `ods_left.png`, `ods_right.png`, `provenance.png`,
  `ods.json` with the coverage. Rockytop sols 461-464 (the example data) is the example station. `mppp mesh |
  depthmaps | ods WORK --station S003D1234` (also `python -m mppp.ibr`); the products under
  `<project>/ods/<station>/`; the `[ibr]` extra (open3d). Not built: the web viewer (future work; any VR photo
  viewer shows the pair). SPICE stays future work (Christian, 9 Nov). Tests: `tests/test_ibr.py` (12: the pivot
  fit, the cleanup, the far sphere and tau, Poisson + decimation, the cast depth against the analytic plane through
  THIN_PRISM_FISHEYE and FULL_OPENCV, the angles on a plane and a shadow, the array round trip, the ODS geometry
  and render, the three commands end to end on a saved block).
- **Tested on the Rockytop block (9 Nov evening, the sandbox)** with the real model, poses and cameras of
  `D:\rover_blocks\mars2020_sol_0460_rockytop` (315 frames, 1.05 M tie points; the image files could not be read
  through the bridge - hard links - so one real sol-461 frame from `tests/data` and rendered stand-ins for the
  others): the Metashape conversion exact to 1.8e-12 px over the 54 sensors, Metashape's own camera equations
  reproducing pycolmap's projection to 2.5e-11 px on 20,000 observations (0.32 px rms against the keypoints, mean
  0.0007 px: no half-pixel offset), every Bundler observation in front of its camera; the dense workspace of the
  16 sol-461 half-resolution Navcam frames; COLMAP's fusion and Poisson mesher on stand-in depth maps (no CUDA
  here: the patch-match is the one step not run), the fused cloud within 1.5 cm (median) of the tie points to
  10 m; the station pivot from 20 frames to 0.4 mm rms; the depth maps through the native fisheye models within
  0.3-0.4 % of the tie points' depth to 10 m; the photometric cube; the 4096 x 2048 ODS pair, whose left / right
  disparity is 99 % of the 64 mm prediction from 2 to 7 m with no vertical disparity. Two fixes from it:
  **`mppp dense` fusion wrote no `fused.ply`** on pycolmap 4.2 (its `stereo_fusion` writes a model folder unless
  `output_type="ply"`; the call raised after the fusion) - fixed; **`mppp dense --station / --sequence / --scale /
  --images`**: a sub-model of those frames in its own workspace `dense_<station>[_<sequence>]` (the whole block's
  317 frames otherwise), which `mppp mesh` takes first. Tests: the ODS parallax with a ground-fixed texture, the
  dense subset and the fusion on a synthetic plane.
- **The first real station run, on the PC (10 Nov)**: Rockytop S026D0630, the 16 sol-461 half-resolution Navcam
  frames, `mppp dense --station 26_630 --scale 0.5` with patch-match on CUDA (323 s) and the fusion (150,014
  points); then `mesh`, `depthmaps --photometry`, `ods`. The sparse points agree with the cast depth to 0.1-0.5 %
  (median |dz|/z, 0-10 m). **But the fused cloud stopped at 7.7 m** (99th percentile; none beyond 10 m), so the
  mesh covered 2-10 m (55 m^2) and the ODS pair is 7 % mesh, 93 % far sphere. The cause is COLMAP's default
  patch-match filter: a pixel needs a 3 deg triangulation angle (Navcam's 0.424 m baseline: 8.1 m) and two
  consistent sources, and a fused point 5 images, where one station's far terrain is seen by one stereo pair.
  **`mppp dense --station / --images` now uses `STATION_STEREO_OPTIONS`** (0.4 deg - 61 m, the mesh radius - and
  one consistent source) **and `STATION_FUSION_OPTIONS`** (2 images); `--min-angle DEG` and `--min-views N` set
  them; a whole block keeps COLMAP's defaults; the options used are in `mppp_dense_run.json`. The stereo has to
  be run again (the filter is applied in patch-match). Also: the open3d message names `open3d>=0.20` (the first
  with a Python 3.13 wheel; the `[ibr]` extra too) with quotes that work in cmd.
- **The station's surface: the mesh to 20 m from the mast axis, a ground plane to 100 m, the sky at infinity**
  (10 Nov; Christian: "cut off at 20 m from the axis of the mast rotation, beyond that, project onto a flat plain out
  to 100 m, and beyond that onto a sphere at infinity ... Many parts of the terrain will only be seen by two images,
  so make sure those get good depth coverage"; "even if you don't get a solid constraint of stereo range on the mid
  to far field terrain, you can still assume that it is distant with little parallax"). `mppp mesh`: the mesh is
  clipped at `--mesh-radius` 20 m of horizontal distance from the axis (was 60 m of range); only the fused points
  within 25 m are meshed. `fit_ground_plane`: the fused points and the block's tie points (3+ images) 3-100 m from
  the axis, the median point of each 10 deg x range-ring bin, a Huber fit z = a x + b y + c over the bins - the
  regional slope (an 8 deg test slope with rocks and wild points comes back to 0.3 deg); horizontal at the median
  height with `--horizontal-plane`, a tilt over 20 deg or under 8 bins / 90 deg of azimuth; `ground_plane.json` has
  the plane, its tilt, its downhill direction and the misfit per ring. `raycast.Scene`: the mesh first, the plane
  where a ray misses it (out to `--plane-radius` 100 m; it fills the mesh's holes too), the sky at infinity where it
  misses both (label 2, no depth in the maps; 1e6 m for the ODS); priority, not the nearest hit, and only the mesh
  occludes. New label 4 = plane in the maps, the reports and `ods.json`; the depth maps' report checks the tie points
  on the plane as well (how far the flat stand-in is from the terrain it replaces). `mesh_report.json`: the fused
  points per ring from the axis, the scene. `--far-radius R` keeps the prototype's finite sphere. Two-view coverage:
  a stereo-pair test with COLMAP's real fusion - 0 points with its defaults, 239,704 with the station options,
  on the ground to 4.7 mm (median), out to 60 m (11,483 at 15-20 m, 23,962 beyond 20 m).
- **The rover shown, or not** (10 Nov; Christian: "We should try this with and without the rover body and deck
  shown. The rover deck should always be aligned though."). `mppp dense --station ... --fusion --keep-rover`: the
  same depth maps fused without the rover masks into `fused_rover.ply` (the stereo takes no masks, so no new
  stereo); `mppp mesh / depthmaps / ods --rover`: that cloud meshed with the deck (the plane still from the masked
  `fused.ply`, so the deck does not lift it), the rover's pixels used, products in `ods/<station>_rover/` beside the
  rover-removed ones. The deck comes from the same posed frames as the terrain, so it sits where the block puts it
  in both eyes. Not handled yet: the wind-sensor booms below the mast head, too close to the eyes (Christian: "In the
  future, we will need a way to deal with those by somehow removing them from the point cloud and masking out that
  part of the images"); and an arm moved between the station's sols shows twice.
- **Why the mesh still had no coverage after the fix** (10 Nov, the PC's 10:34 rerun). One frame's maps settle it:
  its photometric depth map has depth nearly everywhere (17 % of its pixels at 8-60 m, 15 % beyond), the geometric
  map nothing beyond 8 m (99th percentile 7.0 m) - the 3 deg filter. The rerun did not get the fix: the PC was still
  on ecf9a5a (bundles h and i not synced), and even with it COLMAP's patch-match skips every image whose maps exist
  (the stereo step took 0.3 s). `stereo_outputs_ready`: the options of the maps are recorded
  (`stereo/mppp_stereo.json`); maps made with other options, or unrecorded ones when the run sets options, are
  deleted before the stereo; the same options keep them (COLMAP resumes); `mppp dense --restereo` forces it.
- **The ODS blend** (10 Nov; Christian: "the mosaic should be blended together to avoid jagged borders"). The
  prototype took the best two sources by a 40 px border feather over the parallax angle - hard steps at every frame
  boundary and along the rover mask's outline. Now: the feather counts from every edge a source may not cross (the
  image border and the rover mask: a distance transform per source), rising as a smoothstep; two bands - the low
  band (each source blurred, sigma 0.6 % of its short side) over a wide feather (15 % of the short side), the detail
  over a narrow one (3 %), so brightness changes across a wide strip while the texture stays sharp and is not
  ghosted; the best four sources per pixel; a soft parallax term 1 / (1 + angle / 0.02 rad); and one gain per source
  from the low-band brightness of the pixels two sources share (a 1024 px pass; median log ratios per pair, least
  squares with a weak pull to 1, clipped 0.6-1.6; `ods.json` "gains", `--no-gains`). Test: six frames, every other
  one 30 % darker - the gains come back at 1/0.70, the band around the station is level to 5 % against a
  same-exposure render with no step over 1 %, where without gains the frames' levels show (> 20 %). With the plane
  the terrain past 8 m is no longer painted on a 100 m sphere, which was the other source of breaks at the seams
  (two frames disagreeing on where that terrain is).
- **Milder lighting augmentation** (10 Nov; Christian: "lower the brightness contrast jitter augmentation ... the
  bright rover deck is one of the best cues it has"). The A/B of 10 Nov on the 480 frozen frames: v4.4 (recipe A)
  0.9711 against v4.2 0.9724 on the same data and frames (-0.0013, 95 % -0.0023..-0.0004; -0.0043 on the 120 frames
  v3 never saw), the loss nearly all missed hardware; v3 0.9787 (0.9731 on the 120, tied with v4.2). The photometric
  defaults are now mild - exposure x0.7-1.15 (brightening small: it clips the deck first), gamma 0.85-1.2, gains
  +-8 %, black level +-1 %, on half the samples (`--photometric-prob 0.5`); a 215-grey deck patch never clips and
  keeps 92 % of its contrast to the terrain (median), against 29 % clipped under v4.4's settings, kept as
  `PHOTOMETRIC_STRONG` (`--photometric-strength strong`). Test: `test_mild_photometric_keeps_the_bright_deck_a_cue`.
- **Notebooks** (Christian: "clean up the notebooks to have one example of the pipeline and other notebooks for
  each study"): `notebooks/pipeline_example.ipynb` (the whole core on the example data: processing, the alignment
  and its health, cameras and tie points, the dense cloud, the station mesh, the depth and photometric maps, the
  stereoscopic panorama, the Metashape and Nerfstudio exports; replaces `01_process_images` and
  `02_colmap_alignment`) and `notebooks/studies/` - `camera_models` (the joints, the deck, the promotion; rewritten
  for the study package), `error_analysis`, `mask_training` (+ the recipes, the frozen test list, `--compare`),
  `sites_maps` (new). `scripts/_make_notebooks_0p96.py` regenerates the new ones.

**On the way to the build (5-9 Nov).** The Mastcam-Z joint's diagnostics after the 4 Nov Z34 joint failed only
the per-block aspect gate on Chal Rocks large (items A-D of
`claude/handoff-2026-11-05-zcam-joint-diagnostics-pooling-states-residuals.md`, Christian 5 Nov: "yes to pooling on
the block or sequence level"): a pooled diagnostic pass the gate reads, the residual direction per sequence, a
warning for opposite backlash states between the eyes, and f0's sd from the sequence scatter.

- **A. Pooled diagnostic pass** (`mppp_studies.calibration.ztied.pooled_pass`; `calib zcam --zcam-pool sequence|block|none`,
  default sequence; `fit_zjoint(pool=...)`). One pool per (eye, sequence) - or (eye, block, backlash state) with
  "block" - over the bins at the joint's law: per pool one scale on fx, one on fy (so f and fy / fx) and cx, cy free,
  the distortion on the line, each frame on the line at its own focus count (its keypoints scaled by the eye's slope
  from the bin's focus to the frame's). The per-bin pass still runs (the lines, the rig trends, `zcam_efl_vs_focus`),
  so `pool` costs one more adjustment. The per-block aspect gate (`block_aspect_rows`) now reads the pooled value -
  the observation-weighted mean of the block's pools' dfy - dfx (20,000-observation floor as before) - with the
  per-bin line beside it (`per_bin_ppm`, `per_bin_diff_ppm`); a block with two visits (Chal Rocks large, sols 72-100
  and 360-379; `visits_of`, gaps over 100 sols) gets a row per visit, report only. The report lists the 10 pools with
  the largest |dfy - dfx| with their dfx, dfy. Why: ZCAM08401 (sol 369, a 6-row 200-degree pan) makes each row its
  own 30-count focus bin, whose fy is held only by the rows above and below - other bins; the rows' fy drifted
  together by -1.5 to -1.8 % in the joint's per-bin pass while the block alignment (aspect held) and the mini-block
  of 5 Nov evening showed nothing. The joint's own fit is unchanged. Unit test: a sequence with fy +3000 ppm off the
  law in both synthetic blocks - the pools find it (about the fitted line, which is the sequences' mean), the per-bin
  bins agree, the gate reads the block mean.
- **B. Residual direction per sequence** (`residual_directions`, from the final joint pass, in `zcam_joint.json`
  and `report.md`): per (eye, sequence) the rms of the x and y residuals apart, the scale left in fx and fy (the slope
  of the residual against the distance from the principal point, as ppm the frames want f changed by, with its sd),
  and the same per row (elevation tier, clustered at gaps over half the vertical field of view) for pans. The 10
  sequences with the largest rms and every sequence flagged by A or C. Mostly vertical residuals growing from the
  centre mean a real pull (pose or structure, e.g. the visit-2 site-frame offset) that pooling would hide; no
  direction means the bins were the problem. Unit test: ZL's fy made 2000 ppm too large in one sequence - the row
  says fy -2000 ppm, x untouched.
- **C. Opposite backlash states between the eyes** (`opposite_state_flags`, `OPPOSITE_STATE_TOL` 35 %): per sequence,
  the two eyes' dfx from the diagnostic pass (the pools, else the per-bin means) against each eye's low-state step
  1e6 (1 / ratio - 1); an eye labelled high within 35 % of its step while the other eye is within 35 % of 0, fx and
  fy moving together, is "possible low state, eye X" (the mirror: "possible high state"). A WARN line in the report,
  `check.json` and `calib zcam`'s output (`check_label`), never a failure. The case: Z63 ZCAM08038 (sol 78), ZL at
  -3,450 / -3,760 ppm against a step of -4,170, ZR within +900 of 0; confirmed by the mini-block, and its 24 ZL frames
  are on the found list since 5 Nov.
- **D. f0's sd from the sequence scatter** (`f0_uncertainty`; `eyes.<eye>.f0_sd` in the result, `f0_sd_px` and
  `f0_uncertainty` in the candidate model, a report line). The sequence offsets are random effects with the prior sd
  0.5 % of f, so with few sequences the covariance's sd of f0 is the prior / sqrt(sequences) (Z48: +-13.39 px, the
  same as each offset's sd) and reflects the assumed scatter, not the observed one. Now the fitted offsets' sample sd
  (shrunk by the prior where it binds - flagged `prior_bound`) gives f0 an sd of its own, and the larger of the two is
  adopted. The prior in the fit is unchanged; the lines' `sd_a` stay the covariance's.
- **F. Weak-geometry sequences** (`weak_sequences`, `WEAK_SEQUENCE_FOV` 2, `WEAK_FY_SD` 1 %; Christian, 9 Nov: the Z110
  slope — the bins' fy slope 0.11–0.13 px/count against fx's 0.31, the shared slope pulled 5–9 % shallow by the
  single-row horizon pans; ZCAM08038's 2-row strip at fy/fx −0.5 % with no change in the residuals). A sequence whose
  frames' boresight elevations cover (span + one field of view) under 2 vertical fields of view is weak: in the joint
  its fy gets an offset of its own (px, drawn to 0 with sd 1 % of fy; `weak_fy_px`, `res["weak_fy"]`, a report line),
  so its fy does not pull the slope or the aspect; its pools stay out of the per-block aspect gate's mean and the
  per-block fy line (reported as weak). `WEAK_SEQUENCE_FOV = 0` turns it off. Unit test: a +3000 ppm fy departure of
  one sequence lands in its offset.
- **The focus-approach offset is not fitted any more** (`FOCUS_BELOW_FIT = False`; Christian, 9 Nov: "remove focus
  breathing as a fit parameter"). The 0.94.4 offset on f for bins whose focus came into place from below is held at 0
  and the report says so; `calib zcam --zcam-focus-below` / `fit_zjoint(focus_below=True)` fit it again. The focus
  slope itself (the line) is unchanged.
- **Low-backlash list** (`mars2020_mastcam-z_low_backlash_observations.csv`, Christian 9 Nov): on — Chal Rocks sol 90
  ZCAM08051 and Enchanted Lake sol 586 ZCAM08602 at 34 mm (the 9 Nov states study: −0.92 / −0.96 % and −0.86 / −0.92 %
  against a 1.04 % gap), Enchanted Lake sol 556 ZCAM08577 at 63 mm (the 4 Nov joint's bins at 90 / 88 % of the gap;
  invisible to the block-relative states study as its block's only 63 mm sequence), Chal Rocks sol 78 ZCAM08038 left
  eye only; off — Rockytop sols 507 / 513 ZCAM08530 / ZCAM08538 at 63 mm (0–30 % of the gap). New entry syntax
  `ZL:78:ZCAM08038` (one eye of a sequence on a sol; `-` takes it off), key `eye_sol_sequences`. The blocks concerned
  (Chal Rocks large, Enchanted Lake, Rockytop) need realigning; freeze-2's Z step does that.
- `sites.json`: Rochette without the sol 180 frame(s) at SCLK 0682923713 (`*F_0180_0682923713*`).
- `sites.json` (Christian, 9 Nov): Belva Crater large from sol 768 - the approach station 39_0000, 100 m south of the
  crater stations, with its Mastcam-Z sequences ZCAM08785 (63 mm; `zcam_include` keeps the 63 mm list to it),
  ZCAM08788, ZCAM08796, ZCAM08810, ZCAM08812 and ZCAM08817; `zcam` [34, 63, 110], 14 waypoints, 132.5 m across.
  Official block 18 keeps its 770–821 visits.
- `sites.json` (Christian, 9 Nov): Belva Crater large from sol 768 - the approach station 39_0000, 100 m south of the
  crater stations, with its Mastcam-Z sequences ZCAM08785 (63 mm; `zcam_include` keeps the 63 mm list to it),
  ZCAM08788, ZCAM08796, ZCAM08810, ZCAM08812 and ZCAM08817; `zcam` [34, 63, 110], 14 waypoints, 132.5 m across.
  Official block 18 keeps its 770–821 visits.
- Not in this version: item E (the `navcal_tied_20261103` exact-tie LinAlgError); the Z79 Chal Rocks relabels
  (ZCAM07105, ZCAM08390 to low) are a test, not a list change yet — see the 9 Nov notes.
- **Mask training: online augmentation** (`mppp_studies.mask_training.augment`, the 9 Nov plan
  `claude/mask-next-run-2026-11-09-worst-frames-and-augmentation-plan.md`): lighting (exposure ×0.4–2.5 log-uniform,
  gamma 0.6–1.8, channel gains ±15 %, black level ±2 %, Poisson-Gaussian noise, soft shadows ×0.3–0.6 with the label
  unchanged, haze veil, grey↔colour) drawn per sample in linear space — replaces `images_variable/` and the v4.1 tone
  curve; an LMST-weighted sampler (the frame's clock → LMST within ±2 min; before 09:00 / after 16:00 drawn 4.5×);
  angular-scale normalisation (one target mrad/px per sample from 0.08–1.2, Navcam zoomed in, Z110 out; replaces the
  pixel zooms); mask-conditioned defocus (2–12 px, p 0.3); cross-camera copy-paste of hardware cutouts at the matched
  angular scale (p 0.3); a fitted colour matrix (`ColorMatrix` json, identity until fitted) applied N→Z and inverted;
  family-stratified batches (half Mastcam-Z); `train_families` for the transfer runs, the validation IoU per camera
  family in the log and the card history. `TrainConfig` fields of the same names; `mppp_studies.mask_training.train_mask --recipe
  A|B|C` (A photometric + sampler; B + angular scale, defocus, quad split 4000; C + copy-paste, colour matrix,
  family batches) and the single flags; `train_masks.bat --recipe A --epochs 30 --tag v9A`. Tests: numpy parts
  (`tests/test_mask_augment.py`); the torch wiring is smoke-tested with `train_mask.py --smoke --recipe C`.
  The augmentations run as a pipeline object, `augment.Augment` (a geometry step - angular scale or the letterbox -
  then named stages; `from_config` builds the ones the config turns on, `add_stage` adds one without touching the
  dataset), and `MaskDataset` only calls it. **Frozen test list** (`--test-list FILE`, `TrainConfig.test_list`,
  `read_test_list` / `write_test_list`): the listed mask files are the held-out side of every run in place of the
  random draw, so runs are scored on the same frames; the card records path, count and sha256; `train_mask.py
  --write-test-list FILE --from-checkpoint CKPT` writes one from a checkpoint's recorded held-out split.
- **Site maps** (Christian, 9 Nov): `make_maps.bat` downloads the current waypoint table first (`mppp maps
  --waypoints refresh|auto|study`, `cli.sync_study_waypoints` copies it into the study folder); no contour lines on
  the orthomosaic panels (`M20_CONTOURS=0` drops them from the DTM panels too); figure 6
  (`m20_mars_to_rover_sites_no_I`): E–H 1.25× wider and zoomed in 1.2× (less where a region's blocks would not fit),
  the block list moved right and without "sols ended inside"; the cluster zooms (figure 1) carry a translucent box
  behind every sol number. `sites.json`: **Curtis Ridge** (sols 1298–1304, N×4) is official block 28; Pico Turquino
  to Marble Mountain are 29–41 (the sites' `official_blocks` lists, `blocks_all` and
  `blocks_margin_unit_neretva_vallis` follow).

## 0.95.5 — 2026-11-08

**In short.** What the second freeze run needs (Christian, 8 Nov: "prepare a new freeze run with the Nov 3 ncam
models and the new rig model"): the deck baseline reaches the blocks, the Navcam joint reads and fits a yaw knee,
the Mastcam-Z joints can promote with the per-block aspect gate waived, and the deck adjustment frees the Z rig.

- **Mis-tie exclusions carry their station** (bug, 9 Nov). The far-field and small-sequence mis-tie exclusions
  (`find_mistied_bins`, 0.95.0 / 0.95.3) had no `station`, and the site log's list of excluded frames
  (`pipeline.run_site`) read `e["station"]`: a block whose adjustment excluded a mis-tie ended with
  `KeyError: 'station'` after its last adjustment, before the export, the health report and `run_done.json`. Every
  failed states run of freeze-1 ended this way (Rockytop 9.2 h, Chal Rocks large 7.9 h, Taylorfjellet 10.3 h, Marble
  Mountain 3.0 h, Airey Hill 0.3 h, Overlook Mountain, Rochette, Bright Angel's first try): per-image free focal
  lengths make mis-ties common; with the focus lines they are rare. The blocks' `zcam_focus_states.json` and
  `sparse/` were written before the failure, so `mppp zcam-states` reads them as they are. The exclusions now name
  the image's station (`reconstruction._image_station`) and the log line takes a missing one as `-`.
- **Rig translation "deck" is used by the blocks** (bug). `calib.rig_at` returned the file's translation only for
  `"translation": "fitted"`; the 6 Nov 2026 promotion wrote `"deck"` (423.60 mm), so every block started after it
  took its CAHV baseline instead (Rockytop's states run of 8 Nov: 424.36 mm). The freeze blocks of 2-6 Nov ended
  before that promotion and used the fitted 424.01 mm. `RIG_TRANSLATIONS_USED = ("fitted", "deck")`; the Navcam
  promotion keeps a deck length (`fit.promote_navcam`'s carry-over).
- **Navcam rig yaw knee** (Christian, 7 Nov: "the knee looks real, so the yaw and pitch ... should use the knee").
  `tied.start_values` reads `yaw_early_mdeg_per_sol` of the rig file (H's yaw) and records `hinge_angles`; the joint
  then fits H's yaw (it was always held at 0; roll stays linear) and writes `yaw_early_mdeg_per_sol` and
  `hinge_angles: ["pitch", "yaw"]`. Without a yaw knee in the file nothing changes. `rig_at` / `drift_offset_mdeg`
  applied a yaw knee already.
- **`calib zcam --promote-waive CHECK ...`**: promote when the only failed checks are the ones named; the waiver
  is written to `promotion_waiver.json` and the promotion note. `run_pipeline.py --zcam-waive-aspect ZOOM ...`
  passes `--promote-waive "per-block aspect vs the Navcam"` to those zooms' joints, `--zcam-aspect-hold ZOOM ...`
  passes `--zcam-aspect 1.0`.
- **`python -m mppp_studies.calibration.deck --zcam-rig full`**: the Mastcam-Z rig refined in the deck adjustment, so a Z34 deck pan
  gives the Z baseline against the survey; `--no-zcam` for the Navcam targets only.
- **Standard figure: the cameras on the mast head** (Christian, 8 Nov: "I love zcam rig figure. Add that to the list of
  standard figure ... also do this for the other zooms").  `mppp_studies.calibration.zcam_mast_head`: the Z frames of the Mastcam-Z
  blocks through the deck_joint mast chain (resolver angles; per station the block <- RNAV similarity from its Navcam
  frames) into the head frame; per zoom a 2 x 2 (top view, side view, pair centre / height against sol and
  temperature, centre along the boresight against focus), the centre and image-centre pointing against zoom, and the
  Z mount per zoom (Q_Z per zoom and eye; c_Z from 34 mm, along the boresight held at its near-focus median) in
  `zcam_mast_head_<tag>.json`.  `paper_figures.py --mast-chain` (default: the newest deck_joint_mast_model.json under
  `<root>\camera_analysis` or `--out`).
- `sites.json`: Chal Rocks large without sol 99 ZCAM08060 at 79 mm (the Z79 joint's +1.2 % sequence, a near-focus
  pan of the deck and the ground beside it); group `freeze2_z110_without_marble` for the Z110 test without Marble
  Mountain.

## 0.95.4 — 2026-11-02

**In short.** The unattended freeze chain (the new blocks root `D:\rover_blocks`, mask v4.2 compared before the
blocks, a final pass with the promoted models, the existing studies at the end, resumable after any fault), and the
Hazcam early-mission step tested on the blocks.

- **Rover deck fiducial check** (new; reads only, not part of the freeze chain; Christian, 3 Nov: "build the
  fiducial check for the aligned blocks"). `mppp_studies.calibration.fiducials` and `mppp_studies.calibration.fiducial_check`: the six surveyed
  rigid deck targets (AFT1, RPFA1, RPFA2, TDF1-3; RMECH survey, RNAV = RMECH + (90.02, 0, -1133.38) mm) predicted
  through each Navcam image's label CAHVORE, found by point symmetry (phase correlation of the window with its
  180-degree rotation, then `cv2.cornerSubPix`), and fitted with each block's own refined cameras and NL-NR pose:
  one-eye resections, the pair with the block's rig held (`rig_rms`), the baseline the deck asks for
  (`scale_ppm` +- sd), the rig translation and rotation it sees, the triangulated deck's scale and shape, the
  LDIFF-RDIFF separation (the differential bar turns, its length does not), the NL pose in the rover frame and the
  rover pose in the block (heading, pitch, roll), and its spread within each station. AFT2 is behind the hexagonal
  plate from the mast. Writes `<root>/camera_analysis/fiducials_<date>/` (pairs.csv, detections.csv, summary.json,
  report.md, plots, annotated images with all six targets). First 12 pairs (Seitah huge, Three Forks large, Bell
  Island large; sols 203-1549, the images reachable from the mask set): one eye with the block camera 0.2-0.7 px,
  pair with the block rig 0.5-1.1 px, **baseline -978 +- 116 ppm** (the deck asks for 423.6 mm against the block
  rig's 424.0; chi2/dof 1.0; no significant temperature or sol trend yet); the deck's rig pitch per pair is
  +-50-170 mdeg, far too coarse for the 5 mdeg/1000 sol pitch drift. Annotated images: sub-pixel marks with the
  pixel-centre mapping of the zoomed insets (the 3 Nov prototype drew them half a native pixel up-left), larger
  text, the block solution's reprojection, TDF4. **TDF4** (rigid, not surveyed; Christian, 3 Nov): tracked, and
  fitted from the rays of images whose poses are resected on the surveyed targets (`fit_point_from_rays`):
  RNAV (335.75, 126.60, -1133.43) mm +- (0.08, 0.08, 0.16) from 24 images with >= 5 targets (4 Nov; the 9 with all
  seven: (335.84, 126.64, -1133.37) +- (0.07, 0.11, 0.24)); now the position in use (`UNSURVEYED_NAV_MM`, in the rover
  navigation frame - Christian, 4 Nov: "work consistently in rover frame, not rmech"), refitted by every check, not
  in the fits until promoted. **Which eye moves** (4 Nov): each eye's deck resection minus its label pose, in the
  label camera axes (`minus_label_C_*_mm`, `minus_label_rot_*_mdeg`, the report's per-eye table), and the baseline of
  the two eyes resected independently (`baseline_independent_mm` against `baseline_label_mm`). 54 images (31 NL,
  23 NR, sols 29-2006): both eyes sit about 2-2.7 mm low and 1-2.5 mm back of their labels and point about 0.1 deg
  low (the mast head or its kinematics); along the baseline NL is at its label (-0.1 +- 0.2 mm) and NR is
  1.3-1.7 +- 0.5 mm toward NL; no rotation differs between the eyes beyond its error.
- **The rig drift checked on the rover deck, mission-wide** (new; reads only; Christian, 4 Nov: "Can you use the
  fiducial data to check the roll trend? Can you get access to more deck images across the mission?").
  `fiducials.solve_pair(mode="rotation")`: NR's rotation free with the rig translation held (`rot_*_mdeg` +- sd,
  `rig_deckrot_*` about the rig file's sol-0 rotation, next to the law's `rig_block_*`) - a free translation trades
  with the rotation at 1-2 m (synthetic, 0.3 px: pitch 40 -> 3.5 mdeg, yaw 16 -> 3.3; roll 5-6 either way).
  `fiducials.scan_pds` / `pds_frame` / `pds_pairs` / `check_pds_pair` / `rig_trend` and `mppp_studies.calibration.deck_rig_trend`:
  every Navcam full-frame stereo pair of the local PDS archive (MPPP_PDS) whose label puts >= 3 rigid targets in
  each eye, measured with the law at each image's temperature and the rig at its sol - no block needed; the label
  scan is cached (`camera_analysis/deck_pds_scan.json`); fits of a constant, a line, the law's own drift shape and
  a step against sol; `deck_rig_trend.png`, report.md. First run (4 Nov, 88 pairs staged from the archive, sols
  15-2022, 71 used, the 3 Nov joint's cameras and rig): a pair gives pitch and yaw to ~10 mdeg and roll to ~15
  (6-14 with targets fore and aft in both eyes). **Pitch** +6.5 +- 1.7 mdeg/1000 sol against the law's +5.5, and
  the law's hinged shape is the best model (BIC 54 vs line 62, constant 73): the tie-point pitch drift confirmed
  independently. **Roll** -13.5 +- 2.9 against the law's -5.4 (z -2.8); a constant roll is rejected (BIC +19 against
  a line); a line and a step at sol 475 (-16 +- 3.5) fit equally. **Yaw**: a step of +10.4 +- 2.7 at sol 475 that
  the tie points do not show (+0.1 +- 0.7) - with the translation held the deck's yaw also carries the baseline and
  focal scale (0.5 px of disparity at 1.5 m), so it is not taken as a rotation. Offsets deck - tie points: pitch
  +5.5, yaw +5.6, roll -13.6 mdeg (the deck is at the bottom edge of the frame). `--tdf4` (`extra_rigid=("TDF4",)`): the fitted TDF4 as a
  rigid target too - simulated on the block sols' label geometry it adds ~1 % of the pairs and nothing to the roll
  trend; all 339 block-sol pairs with >= 3 targets per eye would give rate se ~1.0 / 1.2 / 2.1 mdeg/1000 sol (pitch,
  yaw, roll; the 71 used: 1.7 / 2.1 / 2.9) - roll needs targets fore and aft, i.e. the full archive.
- **The rover deck inside the bundle adjustment** (new; Christian, 4 Nov: "Look into ways of putting the fiducial
  locations directly in the bundle adjustment ... Then build and test the deck-in-the-adjustment"). `mppp_studies.calibration.deck`:
  one rover pose T_j (block <- RNAV) per rest interval (per station, or `split="sol"`), a 3-D point per (interval,
  target) observed by its deck pixels with COLMAP's own reprojection costs (the tie points' camera, pose, rig and
  intrinsic blocks) and tied to T_j P_k by the survey (0.05 mm per axis; TDF4 0.4 mm), through the
  `extra_residuals` hook of `bundle_adjust`; the prior-gauge recentre is applied to T_j. With a rigid T_j the survey
  sets the block's metric scale, so a translation-refined rig measures the stereo baseline against the survey inside
  the solve. `python -m mppp_studies.calibration.deck WORK` on a solved block (writes `sparse/cahv_ba_deck`, `colmap/deck/`), or the
  setting `deck_in_ba` (`True`/`"station"`/`"sol"`; off by default and then not in the run key). Tests: the survey
  tie's Jacobian, a synthetic block whose 1 % long baseline the deck brings back to < 0.1 %, poses held -> only T
  moves. First block, Three Forks north ecams sols 686-688 (one station, deck unmasked, 16k features, Navcam and
  front-Hazcam baselines refined): deck 1.15 -> 1.04 px over 40 targets in 10 images; **Navcam baseline 423.56 mm
  (-1058 ppm), front Hazcam -1056 ppm** (the two move together: one station has no other scale); the deck's own
  scale left over +204 +- 294 ppm. Without the deck the same adjustment drifts in scale (-6000 ppm: weak priors only).
  A target more than 3 px (and 4 x the median) off the start fit is dropped (`DECK_REJECT_PX`; one false AFT1, 11 px).
  Two rest intervals (sol 684 at S032D1174 masked, 686-688 at S032D1184): both at 1.06-1.08 px, Navcam baseline
  423.57 mm (-1030 ppm); the rover moved 408.4 mm between them by the deck, 408.5 (front) / 407.8 (rear) by the
  Hazcams.
  Mastcam-Z in the deck adjustment (5 Nov): Z deck targets are found where the block pose predicts them
  (`Z_SEARCH_PX`), and with `include_zcam` the Z rig's translation is refined against the survey too (its start
  baseline read from `zcam_virtual_rig`). Three Forks north, Navcam and the Z34 deck pan (sols 686-692, one interval
  per sol): 63 targets in 28 images at 0.89 px; **Z34 baseline 242.85 -> 242.56 mm (-1184 ppm)**, Navcam -1042 ppm in
  the same solve. Study A1 (seven stations, sols 477-1831) in the project doc of 5 Nov.
- **Hazcam rigs: `hazcam_rig_refine = "full"`** (Christian, 4 Nov: "the hazcams might not have correct baselines, so
  let those refine too"): rotation and translation, with the rig translation prior (`rig_translation_sigma_m`) on
  the right centre; a weak pair (`rig_hold`, < 500 observations) stays held.
- **TDF4 resolved; the deck with all seven targets** (Christian, 4 Nov: "how many images are there with 4 or more
  of the seven fiducials? Resolve the tdf4 location and remeasure the rig rotation"). The archive holds 567 Navcam
  full-frame images with >= 4 of the seven predicted in frame (NL 281, NR 286; 230 stereo pairs); 410 have >= 4
  found. **TDF4** `UNSURVEYED_NAV_MM` = RNAV (335.83, 126.70, -1133.66) mm +- (0.05, 0.05, 0.09) from 80 rays
  (+0.08, +0.10, -0.23 from the 24-image value; early and late halves agree to 0.05 mm). **Rig rotation** with TDF4
  rigid: 200 of 230 pairs solve (164 without - the simulation's "~1 %" was wrong for this selection, where NR
  often has 3 surveyed targets), and the scatter beyond the formal sd falls from 40-88 mdeg to 0-5. With the 71 earlier
  pairs, 192-198 used: pitch +3.6 +- 1.0 (law +5.5, z -1.8; the law's hinged shape and a knee at 250 are best, a
  quadratic +12 BIC), yaw +5.1 +- 1.6 (law +2.7; a step at 475 and the law equal), **roll -9.7 +- 2.0 (law -5.4,
  z -1.9; the law's line is best, a knee at 250 +3.9 and a quadratic +5.8 BIC)** mdeg/1000 sol. **A static focal
  term at the deck**: one adjustment of all 410 images (226 pairs on the law rig, 70 lone images; poses, rig
  rotation offset and rate, TDF4, a focal scale per eye) wants f +1840 / +1740 +- 100 ppm (NL / NR) - or, with f
  held, k1 +0.0016 / +0.0015 (law 0.045 / 0.047) - the same in every sol third, cold and warm half, full-res only
  and lone images; rms 0.80 -> 0.65 px (f) or 0.58 (k1). The label CAHVORE's pupil motion is < 0.1 px at the deck,
  so not that. Either the law's radial profile in the lower field or a near-range effect: a diagnostic for the
  post-freeze Navcam joint (deck observations as an extra group with their own near-field term), not a change to
  the law now.
- **Navcam joint diagnostic bins hold k1 at the law** (`tied.DIAG_FREE = ("fx", "fy")`; Christian, 4 Nov: "Since df
  and dk1 trade against each other ... Ok, hold dk1 at the law"). The block-agreement pass freed fx, fy and k1 per
  block bin; in one Navcam image f and k1 correlate at -0.92 (f's sd x2.5; -0.69 with k3), and Threeforks large,
  Vingeanne and Belva Crater large paired +-400 ppm in fy with dk1 of the opposite sign. Not used by the running
  freeze chain (its Navcam joint is done); takes effect at the next `calib navcam` and `calib lawcheck` (its free units are
  the same bins: `lawcheck.FREE_ORDER = tied.DIAG_FREE`).
- **Which eye moves, 4 Nov with the deck pairs** (57 NL and 50 NR resections, the 3 Nov law cameras, label camera
  axes): along the baseline NL +0.24 +- 0.13 mm and NR -0.83 +- 0.18 mm from their labels (pairs: baseline
  -0.95 +- 0.17 mm shorter than the labels'); both eyes 2.0-2.3 mm low and 1.0-1.4 mm back, and turned (pitch
  -86/-69, yaw +88/+102 mdeg) - the mast head. NR - NL: 0.43 +- 0.07 mm forward (the size the common yaw makes),
  roll +15 +- 4 mdeg. The roll drift against sol splits evenly between the eyes against their labels (NL +15 +- 8,
  NR -16 +- 7 mdeg/1000 sol; NR - NL -15 +- 7, as the deck rig's -13.5 +- 2.9): which eye turns stays unresolved.
- **Navcam joint: `--zero TERM ...`** (`tied.zero_terms`, `fit_tied(zero=...)`; Christian, 3 Nov): global terms set
  to 0 and held, for the post-freeze runs - `D_roll` (one roll for the mission), `S_NL_cx S_NL_cy S_NR_cy` (those
  principal points constant with temperature, NR cx carrying the stereo term).
- **No early-mission offsets in the Navcam and Hazcam laws** (Christian, 3 Nov: "let's remove early mission offset
  altogether"). `tied.EARLY_OFFSETS`, `hazjoint.EARLY_OFFSETS`, `hazfit.EARLY_OFFSETS` = False: every early term at 0
  and held (no |z| test); `fit_tied(early_offsets=True)`, `run_joint(early=True)`, `fit_eye(early_offsets=True)` fit
  them again. Why: their z was per image observation. Navcam (3 Nov joint): NR fy +0.057 px, NR cy +0.039 px; on
  block means the fy step is -223 +- 112 ppm (NL) and -165 +- 119 ppm (NR), not significant and of the other sign;
  the cy offsets of the models in use (+0.137 / +0.152 px) are common to both eyes, i.e. absorbed by the image
  attitudes; at 0 no image point moves by more than 0.04 px. Hazcams: the step is real in the data for RL (+1,230 ppm
  on all seven early blocks) and FR (+444 ppm) - `early_check` keeps reporting it; with the offsets at 0 the early
  RL blocks sit about 0.12 % off the law. The Mastcam-Z early principal-point term (ZR before sol 150) is unchanged.
- **`ecam_consensus`: Witch Hazel Hill back in place of Witch Hazel Hill Large** (Christian, 3 Nov). The large block
  (sols 1355-1428: Witch Hazel Hill and South Arm, 281 m) failed its health check in the freeze run - 9 of 20 stations
  in one tied piece, 2 stations without cross-station ties, attitude p95 3.9 deg - and the Navcam joint left it out.
  The freeze chain's final pass runs `witch_hazel_hill` (its block is new, so it aligns from the start).
- **Navcam intrinsics-vs-temperature figure**: the fy model in use is drawn (grey dash-dot, beside fx's dotted) and
  the fy bins (squares) have a legend entry (Christian, 3 Nov: "Is the starting model the black dashed line for both
  fx and fy?").
- **Mask comparisons use the split training recorded** (2 Nov, "study why the training did not improve"). The A/B
  of v3, v4.1 and v4.2 rebuilt v4.1's split by leaving out the newest masks by file time; the v8 set had grown by
  700 masks (656 Mastcam-Z, 320 of them 110 mm) and 54 of the 59 "held-out" test frames were in fact v4.1 training
  frames, so v4.1 was scored on frames it had learned. Training now writes `<checkpoint>.split.json` (the mask
  names on each side) and the card's `training.val_masks`; `mppp_studies.mask_training.audit.recorded_split` reads those,
  else the debug folder's `frame_audit.csv` (written at the end of training, on the set as it was), and
  `split_labels(items, card, checkpoint)` marks masks added later as "unseen". `compare_mask_models.py` uses it for
  `--split-ckpt`, `--also-held-out` and `--seen-split`, and says when it has to fall back to the rebuild.
- **The Hazcam early-mission step is tested on the blocks** (`mppp_studies.calibration.hazjoint.early_check`; Christian, 2 Nov:
  "It's weird that the hazcams have such a large early mission offset and that it is at 380. I suspect that is not
  real, so please keep an eye on it"). The joint's z for a step is per image observation (~10^5), so it overstates
  the evidence for an effect shared by whole blocks: the 27 Oct fit kept FL's -30 ppm step at z 17 that the blocks do
  not show at all. The check takes each eye's diagnostic bins back to the law without the step, averages them per
  block, and tests the early blocks against the late ones (Welch, block-level z), scans every break between the
  blocks' sols for the best step (380 is the Navcam law's sol, not a Hazcam event), compares the two groups'
  temperatures (a step can stand in for a slope held at another eye's), and refits the step with each eye's own slope.
  Each eye's `early_mission.block_check` in `hazcam_fit.json`, one line per eye in `report.md`, a warning in the log
  when doubtful. The law is unchanged (diagnostic only). On the 27 Oct fit: RL +1,230 ppm, block z +51, all seven
  early blocks +1,186 to +1,318 ppm and every late block within 170 ppm of 0, best break sol 387 (between the blocks
  of sols 343 and 469); FR +444 ppm, z +19, best break 425; FL -7 ppm, z -0.3 (its applied -30 ppm is not real); RR
  +41 ppm, z +1.9 (held at 0, rightly). So the step in RL and FR is in the data on every early block, not one
  odd block; RR shows none, so it is not the rear rig as a whole - what changed for those two eyes between sols 343
  and 469 is the open question.
- **`freeze_chain.bat`** (Christian, 2 Nov: away while it runs; 14:52-15:30: the blocks in `D:\rover_blocks`, mask
  v4.2 compared before the blocks, S2 at the end, "robust against faults ... a standard set of blocks using the best
  camera and rig models available followed by the time consuming tests and studies"): copies the found list and the
  focus-from-below list from `D:\scapes` into the new root, checks `<root>\dem` and the free disk space, waits for
  mask v4.2's training (started by hand; done when its release candidate or summary is written), runs the mask A/B of
  v3, v4.1 and v4.2 (v4.2's split, v4.1 with `--also-held-out`, v3 with `--seen-split`) and passes
  `--set mask_checkpoint=<v4.2 .pt>` to the blocks when v4.2's exclusion IoU on the frames no model trained on beats
  v3's. The pipeline then runs in three calls: `--reset --stop-before error_analysis` (the blocks, the Navcam joint,
  the Hazcam fit, the Mastcam-Z joints - promoted when their checks pass), `--redo ecam_runs zcam_runs --stop-before
  error_analysis` (the final pass: the camera-model fingerprint is in the run key, so only the blocks whose models
  were promoted are aligned again, with their features and matches reused), then the studies (error_analysis,
  rover_frame, lightglue_ab, zcam_states, report). Then the studies that exist (15:46: "use past studies instead of
  writing new ones") on those blocks - `make_figures.bat`, `make_maps.bat`, `zcam_aspect_geometry.py` over every
  zoom, `tie_point_studies.py S6` over the zcam_consensus blocks - and last the tie-point studies on the tiny blocks in
  `D:\scapes` (finished variants skipped): S2 with its verdict (would a cap of 1500 do: centres < 1 cm p95,
  attitudes < 0.1 mdeg p95, focal lengths < 0.05 % max), then S1, S3, S4, S5 (`--skip-tie-studies`).
  `make_maps.bat` honours `MPPP_NOPAUSE`. Faults: a failed step is logged and the
  chain goes on; a step past its time limit is stopped with its process tree; the computer is kept awake
  (SetThreadExecutionState) and QuickEdit switched off; the chain's state is
  `<root>\mppp_pipeline\freeze_chain_status.json` - started again after a crash, a reboot or a closed window it
  skips the steps done, keeps the mask choice (another would change the run key) and continues the pipeline without
  `--reset`; a second chain on the same root does not start while one is alive. No point cap (`--cap` adds it).
  0.95.4, from an independent review (16:02): the status file carries a tag (`CHAIN_TAG`), and the status of another
  chain - the 14:17 one, with "s2" done - is set aside as `freeze_chain_status_before_<tag>.json` instead of being
  taken as this chain's; an exception in the chain's own code for a step is logged as `exit chain-error` and the chain
  goes on (`guarded`); a scheduled task "MPPP freeze chain" (`schtasks /SC ONLOGON`, a one-line launcher next to the
  status file) starts it again at the next logon after a reboot and is removed when the chain ends (`--no-logon-task`);
  the free disk space is logged before every step; `freeze_chain.bat --status`; the title and the log path are set
  before `mppp_env.bat`, and an environment failure is written to the chain's log.
- **Freeze chain audit (3 Nov, 06:20, "full spectrum audit", while the chain ran its blocks)**: (1) the reboot
  guard never armed - `schtasks /SC ONLOGON` needs elevation ("Access is denied" in the 20:21 log); now a RunOnce
  entry of the user (`reg add HKCU\...\RunOnce`), removed when the chain ends on its own; (2) `stop_mppp.bat` left
  the chain alive, which then started the next pipeline step: `freeze_chain.py` is a run marker (`_batch.RUN_MARKERS`),
  and the chain also ends when `mppp_stopped.txt` appears while a step runs (`ChainStopped`); (3) a status file of
  another chain version is continued when its step names are all known (a tag bump no longer restarts the joints);
  (4) a chain-side exception after the child started kills the child (no orphan next to the next step); (5) under
  25 GB free a step waits for space (24 h) instead of starting. Found, not fixed: the tiny blocks
  (`chal_rocks_tiny`, `seitah_overlook_tiny`) are no longer in `D:\scapes` - the lightglue_ab stage and S1/S3/S4/S5
  will rebuild them from PDS. Tests: 13 in `tests/test_freeze_chain.py` (the disk wait off in the tests; its own test).
  The audit's edits reached the PC's `freeze_chain.py` only as an older copy (without the studies at the end and with
  the 0.95.3 tag); the five fixes are applied here to the 0.95.4 chain instead.
- **The blocks root is `D:\rover_blocks`** (Christian, 2 Nov; was `D:\scapes`): the default of every Windows script
  (`run_pipeline.bat` ROOT, the `MPPP_SCAPES` fallbacks, `m20_dem_env.bat`'s DEMs in `<root>\dem`). The camera
  models stay in `D:\scapes\cmods` (`mppp_cmods.txt`). An `MPPP_SCAPES` set with `setx` still wins:
  `setx MPPP_SCAPES D:\rover_blocks`.

## 0.95.3 — 2026-10-31 (the freeze)

**In short.** The code for the fresh run of every block: two batch fixes, the v8-trained mask model prepared for
Hugging Face (not yet the default), a mask A/B script, and the documentation brought up to date.

- **Blocks finished without a run key are run again** (`scripts/_batch.py finished`). A block with
  `error_input/summary.json` but no `run_done.json` (an older MPPP) used to be skipped as "finished without a run
  key"; for the freeze every block runs with the frozen models, so it is now queued ("finished by an older MPPP
  without a run key - run again").
- **No Mastcam-Z joints after `stop_runs`.** `stop_runs.py` (`stop_mppp.bat`) writes `mppp_stopped.txt` in the
  scapes root; `_run_sites.bat` then does not start `MPPP_THEN` (`zcam_joints.bat`) when the batch it waited for was
  stopped. A new batch removes the marker.
- **`zcam_backlash_rerun.bat`** (Christian, 31 Oct: "one good zcam joint to capture all of the problematic
  backlash anomalies ... the rerun of these three blocks with all zooms including z110. Allow more joint iterations
  for convergence"): in one minimised window, (1) the per-image backlash states run of Chal Rocks, Seitah Overlook
  and Rockytop (a new variant, `colmap_states_1031`, seeded from the blocks' current features), (2) `mppp
  zcam-states --apply` on those blocks (`MPPP_STATES_APPLY=0` stops after the report), (3) the three blocks again
  with `--force` (the found list is not in the run key) and every zoom of their lists, (4) `zcam_joints.bat` for
  34 48 63 79 110 mm with `--no-promote --iterations 300`, no Hazcam fit, then the N+Z check.  `stop_mppp.bat`
  stops the chain (`zcam_backlash_rerun.bat` is a run marker).  Why: the joints of 31 Oct - Z34 Chal Rocks
  ZCAM08401 (sol 369) -4,783 / -4,154 ppm and ZCAM08396 -2,300 ppm, Z63 Rockytop ZCAM08530 / 08538 -1.0 to
  -1.1 %, about the low/high backlash ratio; Z48 not converged (101 iterations in both passes); Z110 not run.
- **Small mis-tied Mastcam-Z sequences are caught automatically** (Christian, 1 Nov). `find_mistied_bins`: a bin
  camera of at most `ZCAM_MISTIE_SMALL_FRAMES` = 3 frames whose focal length moved more than 2 % from the model is a
  mis-tie whatever its shift (the far-field rule still needs 2 % and a 0.5 m shift). Checked before the final
  adjustment as before and, new, again after it: the frames are excluded, the bins reset to the model and the final
  adjustment runs once more. Rockytop ZCAM03391 / ZCAM03400 and Seitah Overlook's two ZCAM08114 frames were found
  by hand; Rockytop's ZCAM03400 at 79 mm is also in its EXCLUDE list now.
- **The joint's distortion tests attribute their delta chi2** (Christian, 1 Nov: "Does z63 have a significant k2 or
  tangential distortion parameters?"): with several fixed terms, k1, k2 and the tangential pair (p1, p2) are each
  freed alone too; every test reports the refit k1, k2, p1, p2 and the largest tangential displacement at the frame
  corners next to the radial curve change.
- **`mppp_studies.calibration.zcam_aspect_geometry`** (Christian, 1 Nov: "Could fy be either a block scale change for distant
  terrain or maybe an intrinsic camera sag from the optics changing shape with gravity?"): every Mastcam-Z bin's
  fy / fx against its images' boresight elevation, tie-point range and frame coverage, with weighted fits and a
  pointing x range table - gravity predicts elevation alone, a far-field geometry effect range, partial coverage the
  coverage. Reads the blocks only.
- **`zcam_joints.bat`: 300 Ceres iterations per pass** in every Mastcam-Z joint and in the N+Z check
  (`MPPP_JOINT_ITERATIONS`; the 1 Nov N+Z first pass still stopped at 101).
- **110 mm k1 held at 0** (Christian, 1 Nov: "hold z110 k1 at zero for now and test if k1 is significant"):
  `forms.ZCAM_FIXED_TERMS[110]` now k1, k2, p1, p2 = 0 (the 1 Nov joint fitted k1 +0.42 / +0.39 against -0.23 at
  79 mm, with fy / fx 0.993). The joint's distortion tests free the fixed terms together and, new, k1 alone
  (delta chi2 and the curve change it makes).
- **Per-bin Mastcam-Z aspect hold** (Christian, 2 Nov: "yes to the per-bin aspect hold"; `Settings.zcam_aspect`
  "hold", a Mastcam-Z run-key setting): every Mastcam-Z bin camera's fy / fx is held (relative sigma 2e-5, the
  Navcam's) at its zoom model's pixel aspect, or at the eye's detector aspect - the median of the 34, 48 and 63 mm
  models - where the zoom model's is more than 0.2 % from it (`ZCAM_ASPECT_MAX_DEV`; 110 mm, 0.993) or there is no
  model (`reconstruction.zcam_aspect_targets`, logged per group; `focus_info["aspect"]` in the project). The 2 Nov
  analysis (`mppp_studies.calibration.zcam_aspect_geometry`, 1,151 bins): no gravity sag and no range bias - within blocks the
  34 mm slope with elevation is -67 +- 32 ppm per 10 deg outside Rockytop; the aspect outliers are single sequences
  (Chal Rocks nearly all, -2,500 to -21,500 ppm; Rockytop ZCAM03415 / 03416 sols 520-525; Seitah ZCAM08114) where a
  free fy took up geometry errors. The joints (`calib zcam`, `nz`, the Navcam joints) pass `zcam_aspect="free"`: they
  estimate the models' aspect. "free" restores the per-bin fx, fy. **Softened** the same morning (Christian: "soften
  the aspect hold to about 1000 ppm. same for the ecams"; a free fy / fx also takes up some tangential distortion):
  `Settings.zcam_aspect_sigma` and `Settings.navcam_aspect_sigma` (Navcam and Hazcams) 1e-3 relative - about 1 px of
  tangential-like effect per bin, while the 3,000-21,000 ppm sequence outliers are 3-20 sigma (the Navcam hold was
  2e-5). Both are run-key settings, so every block runs again.
- **Seitah Overlook: sol 124 ZCAM08122 at 34 mm excluded** (2 Nov): -3.8 % in focal length in both eyes (z -700) in
  the 34 mm joint without Chal Rocks - the only check that joint failed.
- **Mask training: zoom pairs** (Christian, 2 Nov: "the high res navcams should both downsample to the standard
  1,648 px width and zoom in 2x before down sampling to 1,648 px ... not for hazcams. Make it exactly 2x"):
  `train_mask.py --zoom-pairs` (`TrainConfig.zoom_pairs`, `MaskDataset`): every Navcam frame whose long side is at
  least 2 x the input size is in the training list twice - whole, and as an exactly 2x crop at a random place
  (`zoom_crop(exact=True)`); binned Navcam and Hazcam frames once; the random zoom off. Before (v4, v4.1) half of the
  large Navcam and Hazcam frames were replaced by a crop zoomed 1-2x, so the inference view of a full-resolution frame
  was seen half as often. For mask v4.2: `train_masks.bat --tag v8_v4.2 --zoom-pairs`, before the freeze run's blocks (`freeze_chain.bat` waits for it).
- **Mask A/B across sets that grew** (2 Nov): `compare_mask_models.py --also-held-out CKPT ...` keeps only the frames
  those models held out too, each split rebuilt as it trained (`items_as_trained`: the newest masks left out to the
  card's n_train + n_val - the v8 set grew from 4,095 to 4,099 masks between v4 and v4.1, and the shuffle of the
  sorted names changes with every mask). The chain's A/B uses v4.1's split with `--also-held-out` v4.
- **The freeze run writes the full export by default** (`run_pipeline.py --light` for the calibration-run export):
  its error_analysis stage reads `error_input\native`, which the light export of 28 Oct left out.
- **Navcam joint points by stations** (Christian, 2 Nov: "for the joint, scale to 500-1000 points per station"):
  `calib navcam` draws 750 Navcam points per station of a block (`--points-per-station`; at least 2,000;
  `tied.POINTS_PER_STATION`, `joint.merge_scapes` with `{"per_station", "min"}`; the per-block numbers in the result's
  `points_per_block`), so a large block weighs by its size; `--points N` is N per block again. The freeze run
  (`run_pipeline.py`) passes `--points-per-station 1500 --iterations 300` instead of `--points 8000` (Christian: the
  final run's joint can take longer); `calib navcam` now takes `--iterations` too (the 27 Oct joint converged in 4).
- **Mastcam-Z lists** (Christian, 2 Nov): `zcam_consensus` starts with the three longest blocks, Chal Rocks Large,
  Rockytop and Taylorfjellet; Rockytop in the 79 and 110 mm lists, Marble Mountain in the 110 mm list; Cannery Passage
  out of every list (its site stays). The run_pipeline tests follow the lists (they still expected the 27 Oct ones).
- **Navcam joint list (`ecam_consensus`) 31 -> 21 blocks** (Christian, 2 Nov: fewer small blocks, more large and huge
  ones): no image twice any more (oeb_landing_site, chal_rocks and chal_rocks_large were inside
  van_zyl_overlook_huge, taylorfjellet inside taylorfjellet_large); seitah_huge for issole_seitah_south and
  seitah_north_large, bright_angle_large for cheyava_falls; rochette, groloy, airey_hill and hans_amundsen_workspace
  out; three new unofficial large blocks - `enchanted_lake_large` (sols 421-425, 556-593; 12 waypoints, 57 m),
  `pico_turquino_large` (1303-1321; 11, 93 m) and `witch_hazel_large` (1355-1428, Witch Hazel Hill and South Arm with
  the sol 1355 end-of-drive station; 17, 281 m) for enchanted_lake, pico_turquino, south_arm and witch_hazel_hill.
  The Mastcam-Z and Hazcam lists are unchanged.
- **Cannery Passage**, an unofficial block (Christian, 2 Nov): sols 543-552, 4 waypoints (rover sites 26, 31; 13.7 m
  apart), Mastcam-Z 48, 63 and 110 mm; in `zcam_consensus` and the 48, 63 and 110 mm consensus groups.
- **Rockytop: eight 63 / 79 mm sequences excluded** (Christian, 2 Nov: "frames that should not be there"):
  ZCAM05105, ZCAM05108, ZCAM05109, ZCAM05110, ZCAM05111, ZCAM05112, ZCAM05115, ZCAM05116 (`seq:` entries; Navcam run
  keys unchanged).
- **Rockytop 110 mm: ZCAM08548 added** to its `zcam_include` (Christian, 2 Nov): ZCAM08529, ZCAM07101, ZCAM08486,
  ZCAM08548.
- **`mppp process --dry-run` counts what processing would process** (2 Nov): it applies the shipped exclusion list
  and the site's / run's EXCLUDE entries, a site's `zcam_include` among them (`pipeline.apply_exclusions`, shared
  with `process_site`), and prints the products by camera and, per Mastcam-Z zoom, every sequence with its sols and
  frame count (`pipeline.product_summary`), to choose a site's `zcam_include`. Before, it counted the selection
  before the exclusions: Rockytop showed 3,550 110 mm frames, while its block has the 448 of its three listed
  sequences.
- **Belva Crater Large replaces Belva Crater** in `zcam_consensus`, `zcam34_consensus`, `zcam110_consensus` and
  `hazcam_consensus` (Christian, 2 Nov), with 110 mm (no 79 mm, as Belva Crater): sols 770-821, 14 stations instead
  of 5, for more 110 mm frames and stations in the 110 mm joint. Its Mastcam-Z block is built by the freeze run.
- **Chal Rocks Large replaces Chal Rocks** in `zcam_consensus` and `zcam34_joint` (Christian, 1 Nov), with 79 and
  110 mm added to its zooms: its second 34 mm 360 (sols 72-90) at the same site separates the second visit's
  anomaly from the place. 1 Nov analysis (all 13 Z34 blocks' bins): frames looking at the horizon have fy / fx
  1,000-1,500 ppm lower than frames looking down, in nearly every block (Taylorfjellet, whose horizon views end on
  a near hillside, not); Chal Rocks fails the per-block aspect check because its largest sequence (ZCAM08401) is a
  horizon panorama. Not a backlash state, not mainly temperature.
- **Mask training: a random tone curve per sample** (Christian, 1 Nov: "v4 should have better augmentation using
  color changes as well as h flips and zooms ... call it v4.1"). `TrainConfig.tone_prob` / `train_mask.py
  --tone-prob` (default 0.5 in the script and `train_masks.bat`): a share of the training samples get a tone curve
  from v3's `images_variable` recipe (`dataset.random_tone`: black point 0-0.11, white point 0.90-1.50, gamma
  0.50-0.85, channel gains 0.95-1.05), drawn anew every time - the colour diversity v3 had from its fixed copies
  on disk, with the images still the standard pipeline's. `train_masks.bat` trains 20 epochs (as v4).
- **Alignment health: cameras without tie points are listed, not judged** (`unconstrained_cameras`, fewer than
  `MIN_CAMERA_OBSERVATIONS` = 30). Rockytop's ZCAM03391 79 mm bins had 0 observations and drifted with the priors
  (fx +9.9 %, fy +33 %); their focal and ray checks failed the block, and the FAIL kept Rockytop - with its good
  79 mm mosaic ZCAM08522 (23 frames per eye, 26-87 k observations per bin) - out of the Z79 joint. The run key is
  unchanged (the alignment is not).
- **`mppp calib zcam / nz --iterations N --passes N`**: Ceres iterations per pass (default 100) and passes
  (default 2); the joint's json records `max_iterations` and `passes`.
- **sites.json**: Seitah Overlook without the two right-eye 79 mm frames of sol 121 ZCAM08114 (0677679250,
  0677678900: 258-261 px in the Z79 joint - mis-tied, not a state; the rig looked 0.7 deg off); Rockytop without
  ZCAM03391 at 79 mm (f 10-11.5 % off the label).
- **`mppp_mask_v4`** in the model registry (`src/mppp/data/models.json`): the ConvNeXt-tiny trained on the v8 mask
  set (4,095 masks, sols 9-2022, pipeline images only, flips and zoom crops; best of 20 epochs at 13, held-out IoU
  0.978; exclusion IoU Navcam 0.976, Mastcam-Z 0.945), exported as `mppp_mask_convnext_tiny_s4_v4.safetensors`
  (SHA-256 `9ecb3dd5…`), URL on Hugging Face `ctate7163/mppp-mask`. The default stays `mppp_mask_v3` until the A/B.
  `docs/hf_model_card.md` describes v4 and the version table (and no longer says the sky is excluded: it is
  included). `mppp_studies.mask_training.release_mask_model upload --extra` uploads the card's figures.
- **`mppp_studies.mask_training.compare_mask_models`**: two or more mask models on the same held-out frames, per camera, with
  `--seen` (the older model's training set) for the frames neither model trained on.
- **Documentation**: new pages *Downstream exports* (dense, Metashape, Nerfstudio), *Rover masks* and *Roadmap*;
  the API reference adds `mppp_studies.calibration.hazfit`, `mppp.sfm.dense`, `metashape`, `nerfstudio`; Methods §5 (masks)
  updated (threshold 0.5, the registry models, sky included); the index's scope; `docs/RELEASING.md` names the right
  upload command (`mppp_studies.mask_training.release_mask_model`, not `mppp.mask.hub`). The working notes (`AUDIT_v0p90`,
  `PLAN_v0p91`, `DELIVERY_v0p91`, `HANDOFF_v0p87`, `CI_ROADMAP`) moved to `docs/history/notes/`. `mkdocs build
  --strict` passes.

## 0.95.2 — 2026-10-30

**In short.** Fewer SIFT features on the narrow Mastcam-Z zooms (63, 79, 110 mm); `make_figures` draws the newest 34, 48 and 63 mm
joints and the Hazcams by default; a block can be opened as a Metashape project (`mppp metashape`).

- **Fewer SIFT features on the narrow zooms: 16,384 at 63 and 79 mm, 8,192 at 110 mm** (Christian, 30 Oct: "speed
  up the adjustments and improve the tie point quality between ncam and Zcam ... a poor man's octree").
  `max_num_features` takes a zoom key: `{"N": 65536, "Z063": 16384, "Z079": 16384, "Z110": 8192, "*": 32768}`
  (`database.image_zoom` from the camera group or the product name). COLMAP keeps the larger-scale features, which
  make the ties to the wider cameras; the finer ones have no counterpart in a Navcam frame and only tie frames of the
  same zoom (and crowd the ratio test). The GPU matcher also clamps every image to `max_num_matches` 32,768 and
  COLMAP lists the keypoints finest first (checked on Chal Rocks tiny, 30 Oct): an image above 32,768 lost its
  coarsest features - the Navcam-tie ones - in the pair matching. Taylorfjellet 27 Oct: Z63 median 32,844 features
  a frame (210 of 418 above 32,768). Only Mastcam-Z blocks with these zooms change: the zoom keys of zooms a block
  does not have are dropped from its features record (`features_for_project`) and from the Navcam run key, so the
  Navcam blocks keep their keys and features.
- **Metashape project of a block** (Christian, 30 Oct: "Convert the colmap alignment a metashape project with just
  cameras, tie points, masks. The frames should be linked to the colmap images folder"). `mppp metashape WORK`
  writes `<WORK>\colmap\metashape\`: `metashape_job.json` (one sensor per COLMAP camera of the native-pixel model
  `gui_native/sparse`, every pose, the masks, the region, the block offset and frame), the tie points as Bundler v0.3
  (`<block>.bundler.out` and its image list), `build_metashape_project.py` (`mppp.sfm.metashape_build`, runs in
  Metashape Professional's Python) and `make_metashape_project.bat`, which makes `<block>.psx`: the photos linked from
  `images\`, the tie points imported, the sensors and poses set and the calibration fixed (no alignment, no
  optimisation), the masks from `masks\`, a local metric coordinate system (MPPP's world frame: x east, y north,
  z up, the chunk transform adds the block offset), and a check (`metashape_check.json`: COLMAP's observations and
  the imported tie points reprojected by Metashape). The camera models translate exactly: THIN_PRISM_FISHEYE ->
  Fisheye and FULL_OPENCV -> Frame with f = fy, b1 = fx - fy, (cx, cy) from the frame centre, K1..K4, and P1 = p2,
  P2 = p1 (checked to 1e-11 px against pycolmap); terms Metashape has no place for (sx1, sy1; k4-k6), zero in every
  MPPP model, would be fitted and reported. Chal Rocks: 137 images, 40 sensors, 383,851 tie points with 1.32 M
  observations (55 MB Bundler file); the job's sensors and poses reproject COLMAP's observations at its own
  residuals (median 0.154 px). First run in Metashape 2.2.3 (Chal Rocks): COLMAP's observations reprojected at
  0.155 px median, mean offset (-0.002, -0.002) px - the calibration, the poses and the pixel origin carry over -
  but the Bundler import triangulates its own tie points with Bundler's pinhole cameras (8 m off, 681 px median
  reprojection error once the real cameras are set: the "large pixel residual that is not real", and 28 of 83
  enabled Navcam frames without a depth map). The build now gives the points COLMAP's coordinates (triangulated
  again with the real cameras when the API does not take them) and ports the camera references (the PDS label
  priors: position with the prior sigma, yaw/pitch/roll by Metashape's own conversion), and saves before the check.
  Second run: the import also renumbers its tracks (12 dropped, ids shifted by 0-12), so COLMAP's coordinates by
  track id landed on the wrong tracks for part of the points (median 0.24 px, rms 480 px). The build now
  triangulates the points again from COLMAP's observations with the job's cameras, and the export keeps one
  observation per image and track (COLMAP lets a track see an image twice: 6,636 observations in Chal Rocks; the
  one nearer the projection is kept). Metashape's yaw/pitch/roll of the refined poses agree with MPPP's reference
  formula to 1e-13 deg.
- **Nerfstudio export** (Christian, 30 Oct: "undistorted in a perspective frame, with images and masks all of one
  size and an accurate transformations.json"; 2048 x 1536, "Zcam frames can be clipped 22 columns on the left and 16
  on the right"). `mppp nerfstudio WORK` writes `<project>\nerfstudio\` (`mppp.sfm.nerfstudio`): every registered
  frame resampled through COLMAP's exact camera model into a 2048 x 1536 pinhole frame (per-frame focal length and
  principal point: the camera's centre resolution scaled by 2048 / width; Mastcam-Z without columns 0-21 and
  1632-1647, scaled by 1536 / 1200), the masks likewise (0 where there is no source pixel), `transforms.json`
  (OpenGL camera-to-world in MPPP's world frame) and `sparse_pc.ply`. Checked on a synthetic block: the resampled
  value at Nerfstudio's projection of a tie point is the source value at COLMAP's projection.
- **`ecam_block.bat SITE`** (`mppp run SITE --metashape --nerfstudio`): one site's Navcam + Hazcam block, then the
  Metashape job and the Nerfstudio export. Hippo Pools' second visit runs out to sol 2022 (new images).
- **`train_masks.bat`** (Christian, 30 Oct: "looks for D:\masks\masks_training_set_v8 or the highest v and trains a
  convnext tiny model all the way through and generates the required figures and outputs needed for the joss
  paper"): the newest `masks_training_set_vN` (or `MPPP_MASK_DATA`), `images\` regenerated from the PDS (resumable)
  and `images_variable\` (the previous set's reused; kept when complete), 10 epochs of ConvNeXt-tiny with a Python
  that has CUDA torch, then `mppp_studies.mask_training.train_mask --paper`: `figure1_masks.png/.pdf` and per-camera metrics on the
  held-out frames, the training curves, the card, `summary.md` (pairs by camera, sols, split, held-out IoU,
  augmentation) and the release candidate `.safetensors`, in `Claude outputs\mask_<checkpoint>\`. Nothing promoted.
- **Training images keep the frames the alignment skips**: `regen_config` turns the pipeline's frame selection off
  (v8's first regeneration lost 208 Navcam frames - sky-pointing, dark blue centre, saturated, outside the LMST
  window - and 4 Mastcam-Z frames of site 1 have no landing-frame position and stay out).
- **Training images need no landing-frame position** (Christian, 30 Oct: "a landing position should not be
  necessary to color process them"): `regen_config` takes the pose prior from the label's site frame
  (`extrinsics_from_waypoints` off), so the four sol-3 Mastcam-Z frames of site 1 are regenerated too.
- **`train-ns.bat` in every Nerfstudio folder** (Christian, 30 Oct: "a small and zcam optimized splatfacto training run
  starting from the sparse point cloud and training on down sampled by two images. Make sure to prune floating blobs
  regularly"): `mppp nerfstudio` also writes `images_2\` and `masks_2\` (1024 x 768; a half-size mask pixel is valid
  only where its four source pixels are) and the batch file: splatfacto, 15000 steps, `--load-3D-points` (the tie
  points), camera optimizer off (MPPP's poses held), densification threshold 0.0006 (finer for Mastcam-Z) until step
  10000, opacity reset every 1500 steps with culling below 0.2, large blobs culled (0.3 of the scene scale), scale
  regularisation (axis ratio at most 6), images cached on the CPU; extra splatfacto options as arguments.
- **Hippo Pools: sol 1947's end-of-drive station S091D0000 in** (Christian, 30 Oct): `NEARBY_FROM_FIRST_SOL` off, so
  its sol-1947 images come in as another visit within 10 m; the mid-drive stops at site 90 do not.
- **Mask training augmentation**: horizontal flips (as before) and, new, large Navcam/Hazcam frames (long side above
  the input size) zoomed in by up to 2x: half of them cropped to 1/z of the frame (z uniform in [1, 2]) at a random
  place before the letterbox downsamples them (`TrainConfig.zoom_max`, `zoom_prob`, `zoom_families`; crops mostly of
  sub-frame padding are drawn again). A second run the same day gets `_2` instead of stopping on the existing name.
- **Mask training on the pipeline's images only** (Christian, 30 Oct: "only trains on the images that went through
  the standard mppp pipeline"): `train_mask.py` and `train_masks.bat` use `images\` alone; the brightness-varied
  `images_variable\` copies (about twice the samples per epoch) are built and used only with `--variable`.
- **make_figures** (`mppp_studies.paper.paper_figures`): the efl vs focus figure of the newest joint folder of each of 34, 48,
  63 mm (`zcam<zoom>_joint_<date>` or `<study>\zcam<zoom>_joint`, with its `zcam_joint.json`; `--zcam-joints`,
  `--zcam-zooms`), and of the newest Hazcam fit (`hazcam_fit_*` or `<study>\hazcam_fit`; `--hazcam`): the rig trends
  of each pair and a new `plot_hazcam_efl_vs_temperature` (the four eyes' focal length against temperature, law and
  block bins, early bins open), also written by the Hazcam fit itself. `make_figures.bat` passes `MPPP_FIG_ARGS`.

## 0.95.1 — 2026-10-29

**In short.** The Mastcam-Z joint refines the rig translation instead of holding it at the CAHV value, reports how
significant the move is, and gives ZR an early-mission principal-point offset; the joints' outputs gain what the
Navcam-anchor question needs; the trial can be resumed part by part. Run keys do not change: the 0.94 and 0.95.0
blocks stay valid.

### Mastcam-Z joint

- **Rig translation refined** (Christian, 29 Oct: "the z34 rig translation should not be held ... let these rig
  parameters change and optimize while keeping a close eye on their significance"). Z34 28 Oct: ZR's focus slope was
  0.0105 px/count steeper than ZL's in all ten blocks (Z48 +0.013, Z63 +0.0125, Z110 +0.009) - the signature of a
  right centre that is wrong along the boresight (it scales ZR's image by dz / D, and D goes with the focus count; a
  synthetic block with +6 mm gave the same pattern and the free rig recovered +6.0 mm). `ztied.ZJOINT_RIG_FIT = "full"`:
  rotation and translation refined, a 1 cm prior per axis on the start rig's right centre (`--zcam-rig-fit rotation`
  for the 0.94 joint). The refined rig gets a covariance of its own (`extra:zrig|Z0nn`); the report and
  `zcam_joint.json` (`rig`) give the right centre's move (x, y, z along the boresight, mm) with its sd and z per axis
  and the baseline; the candidate model's rig carries `translation_change_mm`, `translation_sd_mm`. The N+Z joint
  refines the rig the same way.
- **Early-mission principal-point offset of ZR** (Christian: "fix the Zcam rig drift for all zooms. Put this into an
  early mission principal point offset if it is significant"). The per-block rig pitch of Z34 was -28 / -36 mdeg at
  Chal Rocks (sol 102) and Seitah Overlook (119) and within +-7 mdeg from Rochette (182) on: a step, not a drift. ZR's
  bins before `ZCAM_EARLY_SOL` 150 (their median sol) share one offset (dcx, dcy) on top of their principal-point ties
  (their cx is freed for it), from three early bins on (`ZCAM_EARLY_MIN_BINS`); its sd from the covariance
  (`extra:dearly|ZR0nn`); applied in the candidate model (`pp.early`, `applied`) where |z| >= 3 and recorded otherwise.
  The diagnostic pass applies it, so the per-block rig rows show what is left. The block runs start an early ZR bin
  with the offset (`early_pp_offset_px` in the bin spec), after the focus lines. Every zoom; `--zcam-early-sol SOL` (0:
  none).
- **Eye-slope tests** (`--eye-slope-tests`) against the joint's own rig: with the translation refined they are "rig
  translation held" (the start rig; 3 dof) and "one slope for both eyes" (1 dof); with `--zcam-rig-fit rotation` the
  three tests of 0.95.0's first form (translation free, one slope, both).
- **Thin bins held at their start** (the 1000-observation cut). The Z79 joint of 28 Oct: a free Seitah Overlook bin of
  389 observations went to fx 10247, fy 6701 px and the rms to 34.9 px.
- **Per-block aspect check with an observation floor** (`ZJOINT_LIMITS["min_block_aspect_check_observations"]`
  20000): an eye-block line with fewer observations is listed, not judged (Seitah Overlook's far-field 9-15 k in Z34
  and Z63 failed the check every time).

- **k2 held at every zoom** (Christian, 29 Oct: "hold k2 at its last fit value rounded down to the lowest tenth ...
  We don't want the k1/k2 degeneracy getting into the rig refinement"): `forms.ZCAM_FIXED_TERMS` 34 mm ZL 0.5 and ZR 0.6 as
  before (last free fits 0.484 / 0.615; Christian: "held at a reasonable value until there is evidence of another
  significant value"); 48 mm 1.4 / 1.4 (was free; 28 Oct 1.474 / 1.430); 63, 79, 110 mm 0 as
  before (with p1, p2).
- **The "free (vs the fixed values)" test works**: from the tied solution with every bin at the fixed value the
  adjustment did not move the freed term (28 Oct Z34: k2 0.5 -> 0.5000013) and the delta chi2 (810) was convergence
  noise made positive by `abs()`. The freed term now starts `FREED_TERM_NUDGE` off (k2 0.05) and the delta chi2 is
  signed (`converged` false when the variant ends worse than the joint); synthetic: held 0.3, truth 0.40 -> refit
  0.415 / 0.418, delta chi2 145.

- **No "ZR k1 = ZL k1" test** (Christian, 29 Oct: "ZR034 k1 should not be equal to ZL034 k1. remove that test in the
  joint"): the distortion tests are k2 = 0 (when k2 is free) and the fixed terms freed; the curves' right-minus-left
  difference with its z is still in the report.

- **Focal length against focus figure** (Christian, 29 Oct): `zcam_efl_vs_focus.png` in every Mastcam-Z joint folder
  and `zcam<zoom>_efl_vs_focus_<tag>.png` from `make_figures.bat` for a joint folder: per eye (ZL cool, ZR warm, the
  same y range), the joint's law (high state solid, low dashed), every bin's own fx from the diagnostic pass (a dot
  per bin, all blocks alike, open = low state, x = thin), efl in mm on the right axis, the bins about
  the law in ppm below, the reference focus dashed (`constrain.plot_zcam_efl_vs_focus`). The diagnostic pass records
  the bins (`diagnostics.bins`); a joint written before has the blocks' own lines drawn instead.

### Navcam

- **The combined law as model files**: the N+Z joint writes its Navcam law into `<out>\navcam_law\` (camera and rig
  files, `tied.write_law_files`, as `calib navcam` writes its candidate), so `calib lawcheck --law combined=...` checks
  it against the law in use on the Navcam blocks (the Navcam / Mastcam-Z y-scale tension, +200 / +230 ppm in fy).

### Hazcams

- **FL and FR fit their own thermal slopes, the rear eyes take their mean** (Christian, 29 Oct: "The FL and FR should
  fit separately, and we can assume that both RL and RR are the average of FL and FR"). `hazjoint` slope mode
  `"front"`, the default (`--hazcam-slope front`; `FL`, `all`, `per_eye` as before): terms `s_FL`, `s_FR`; RL and RR use
  (s_FL + s_FR) / 2 without pulling on it, with its sd from the covariance; the report says "own slope" / "the mean of
  FL's and FR's slopes". Before, every eye took FL's slope (+20.0 ppm/degC; FR's own +35.2).

### Dense reconstruction

- `dense_reconstruct.bat` prepares the workspace with MPPP's Python (`mppp dense --prepare --no-bat`: the images
  **and the rover masks** undistorted, so the fusion leaves the rover out), falling back to COLMAP's
  `image_undistorter` (no masks) only when MPPP's Python is not found; it says whether the masks are there.
- `mppp dense <WORK> --maps [png|jpg]` (`--maps-kind geometric|photometric`): COLMAP's depth and normal maps
  (`dense\stereo\depth_maps`, `normal_maps`, `*.bin`) as images in `dense\stereo\maps_png\` - depth in a colour scale
  over its 2-98th percentile (range in the file name), normals as RGB in the camera frame (`sfm.dense.read_colmap_array`,
  `export_dense_maps`). No value (transparent in a PNG, black in a JPEG) inside the rover mask (`dense\masks`) and
  beyond 100 m (`--max-depth`), in the depth and the normal images alike (Christian, 29 Oct).

### Sites and scripts

- Taylorfjellet first in `zcam_consensus` and `zcam34_joint` (Christian, 29 Oct: the longest block, 11-13 h, starts first).
- `run_sites.py --group a,b`: several groups, comma-separated, each site once (as `mppp calib`).
- **One models folder for prompts and batch files**: without `MPPP_CMODS` in the environment, `paths.cmods_dir()` reads
  `scripts\windows\mppp_cmods.txt` (written by `mppp cmods init`) as the batch files do. On 29 Oct `mppp cmods
  promote` from a prompt wrote into the package copy while the site runs aligned with `D:\scapes\cmods`.
- `render_paper.bat` is `make_paper.bat` (log `_transfer\make_paper_log.txt`); `sites_status.bat` is back (the run status
  of every block under `MPPP_SCAPES`: `run_sites.py --status`); `CODE_OF_CONDUCT.md` moves to `docs/`.
- **Include lists** (Christian, 29 Oct: "some z110 into Rockytop, but not all of them, because that would be
  thousands"): a site's `"zcam_include": {"110": ["ZCAM08529", ...]}` keeps only those sequences at that zoom (the
  zoom in the site's `zcam` list); it becomes the exclusion `zoom:110 except seq:...` (new entry form `zoom:NNN`, in
  `exclude_images.txt` too), left out of the Navcam keys. Rockytop `[34, 63, 79, 110]` with ZCAM08529, ZCAM07101,
  ZCAM08486 at 110 mm and in `zcam_consensus` (a third Z79 block); Taylorfjellet `[34, 48, 63, 79, 110]` with
  ZCAM09672, ZCAM09695, ZCAM09723. Marble Mountain leaves out ZCAM10008 at 110 mm (`"exclude": ["*ZCAM10008_110*"]`);
  a Mastcam-Z exclusion pattern (`ZCAM` in it) stays out of the Navcam keys.
- **Chal Rocks 79 mm off the low-state list** (Christian, 29 Oct, from `mppp zcam-states --sites chal_rocks`: every
  fitted frame of both sequences sits on the high line, +0.2 to +0.6 %, z +7 to +17): `-99:ZCAM08060` and
  `-361:ZCAM07105` in the shipped list (the shipped `ZCAM08060_079`, on the list since 0.61, removed; a `-` entry wins
  over the found list too, so the found list on the PC needs no edit). The Z79 joint of 28 Oct had 14 of 17 ZL bins
  low and a block ratio of 1.0003. A row without a `sequence` takes it from the product name for the `sol:sequence`
  entries.
- `sites.json`: the sol ranges of Rose River Falls (449-454), Hidden Harbor (606-619) and Bessieres Island (1018-1039)
  take in their Z34 360s; no waypoint added.
- **79 and 110 mm back in the site runs** (Christian, 29 Oct: "leave z79 on so that we get a first promotable model to
  improve following models. Same with Z110"): `run_pipeline.py --zooms` defaults to 34 48 63 79 110 (the site runs'
  `zcam_zooms_only` and the joints); the 79 / 110 mm bins still keep their refined values per bin in the block runs
  (`zcam_line_zooms` 34, 48, 63) until they have models. Marble Mountain gets 110 mm (`"zcam": [34, 63, 110]`) for a
  third Z110 block; Belva Crater is 110 mm and not 79 mm (`[34, 110]`; Christian, 29 Oct). Chal Rocks drops 79 mm (`[34, 48, 63, 110]`; Christian: "better just to not fit z79 from that block"): the joints leave a site out of a zoom not in its list, so its existing 79 mm bins are not fitted, and the rerun does not align them (79 mm made it 4.7x longer).
- `trial_joints_0p95.bat`: `MPPP_TRIAL` is a list of parts run in order (`navcam tiefslope nz lawcheck hazcam zcam`;
  `all` the default) to resume after a stop; each part a subroutine; the N+Z joint before the law check, which also
  checks the combined law; zooms 34 48 63 79 110 by default.
- `claude_sync.bat` runs from a copy in `%TEMP%` (the merge replaced it mid-run on 29 Oct and the stash was not
  popped).

## 0.95.0 — 2026-10-28 (the freeze release)

**In short.** 0.95.0 is the code every site is rerun on before the cameras of the Navcam, the front Hazcams and
Mastcam-Z 34 mm are frozen for the paper. What it changes, in order of weight: the Mastcam-Z joint fits one focus slope
per eye with an intercept offset per sequence (the focal length is not curved over focus; sequence bins are the main
runs' default, so every Mastcam-Z block runs again), gives the right eye's low-state centre its own offset relative
to the left eye, and holds the focus-from-below offset where it is one with the high/low ratio; the site runs hold a
zoom with fewer than six images at a site at the model, drop far-field mis-ties before the final adjustment instead
of failing the block, extract twice the SIFT features on the Navcam frames, and fail the Navcam corner check only
below 0.15; the Navcam joint carries the stereo differences of the principal point and its thermal slopes as terms of
their own (and the law check tests the stereo law with a real uncertainty), skips the exact-tie step when the ties
are met, and no longer forces an extra pass; every rerun records the Sun-Mars distance, the label sun vector and the
block's tilt with the sun in the block frame for the paper-3 work; and the camera models in use live outside git
(`MPPP_CMODS`), the sync script stops when a delivery is not on `main`, side runs have status files of their own,
the database copies shrink and go, and the hand exclusions of Taylorfjellet and Marble Mountain are gone.

### Mastcam-Z joint and the blocks' focus lines

- **Per-sequence focus offsets, one slope per eye** (Christian, 27 Oct: "the focal length is not curved over focus;
  different sequences sit at different linear offsets with the same slope" - Seitah Overlook's fixed infinity focus
  -0.26 % from the Z34 line, Taylorfjellet's close look-down sequences about -0.5 %). `Settings.zcam_bin_by =
  "sequence"` is the main runs' default (one bin per focus bin, sol, sequence and state; the bin spec records
  `sequence`, `sequence_sol`), so a bin belongs to one sequence. In the joint each (block, sol, sequence, eye) gets an
  intercept offset on fx (fy x the aspect) on top of the eye's line, shared by its states (a mixed sequence measures the
  high / low ratio within itself), drawn to 0 with an sd of `forms.SEQ_OFFSET_SD` = 0.5 % of f (the line is the
  sequences' weighted mean); sequences under `MIN_SEQ_IMAGES` = 5 frames or `MIN_SEQ_OBS` = 500 observations hold
  their offset at 0; thin bins (0.94.5) get none. The offsets are recorded (`sequence_offsets` in `zcam_joint.json`,
  the report's table, `sequence_offsets` on each eye of the candidate model - block facts, not model terms) and
  checked (`max_sequence_offset_pct` = 1 %: more is a mis-tie); the diagnostic pass puts each bin at the law plus its
  offset, so the per-block lines and the block-aspect check do not read the offsets as aspect; the report says how
  many sequences carry a focus span (the slope's support). In the blocks, `refit_param_lines` fits the fx, fy lines
  with one intercept per sequence (weighted zero-sum) and sets each bin on its sequence's line.
- **The right eye's low-state principal point** (Christian, 27 Oct: "the state shows in the centre"; Seitah Overlook's
  ZR063 low-state frames 0.65 px): ZR's `d_cx_regular`, `d_cy_regular` are fitted relative to ZL's (held: the
  reference eye) from every exposure in which ZR is low and ZL is in the problem, whatever ZL's state
  (`paired_low_exposures`, `MIN_PAIRED_LOW_EXPOSURES` = 10) - not only the mixed exposures of 0.93.2, which the 63 mm
  blocks lack. The blocks' line refit pins a regular-state ZR line at pp0 + `regular_pp_offset_px` (before, the
  own-line path dropped the offset the no-own-line path added).
- **Focus-from-below** held at 0 unless the eye has an fb bin in the high state (every fb bin low: the offset is one
  with the high / low ratio); the reason is in the rows and the candidate (`focus_below.held`).
- **The worst frames**: per-image residuals in the joint's result (`zcam.worst_images`), a table in the report, and the
  frames behind a failed rms in the check's note.

### Site runs

- **A zoom with too few images at a site stays at the model** (`Settings.zcam_min_zoom_images` = 6 per eye;
  `held_at_model` on its bins, every parameter fixed; the lines and the camera-model checks leave it alone).
  Taylorfjellet's one 48 mm pair held f but its free principal point went +90 px and failed the block; 48 mm is back
  on Taylorfjellet.
- **Far-field mis-ties** (Marble Mountain ZCAM04469: 63 mm at the horizon from a station 2 m off, f +7.9 %, 1.7 m):
  before the final adjustment a Mastcam-Z bin whose focal length is more than `ZCAM_MISTIE_FOCAL_PCT` = 2 % off the
  model while its frames sit more than `ZCAM_MISTIE_SHIFT_M` = 0.5 m from the station's Navcam shift loses those
  frames (`kind: mistie`) and goes back to the model; a warn-only health note (`mistied_frames`). Marble Mountain's
  `seq:ZCAM04469` exclusion is gone; Seitah Overlook's ZCAM08111 stays (a primary image set).
- **SIFT features per family**: `Settings.max_num_features = {"N": 65536, "*": 32768}` (Christian, 27 Oct: twice the
  features on the full-resolution Navcam frames, which the zoomed-in Mastcam-Z frames tie to; one extraction pass per
  distinct count; `features.json` records the dict, so every site's features are extracted again).
- Health: the Navcam `corner_triangulated_ratio` fails below 0.15 (0.94.5).
- **Hazcam pairs off their station are dropped, not the block** (Three Forks North: one front pair tied by 1100 points
  slid 0.9 m and failed the block on its station's shift spread): a Hazcam pair farther than `hazcam_shift_m` = 0.3 m
  from the median shift of its station's Navcam frames with fewer than `hazcam_strong_observations` = 3000 is excluded
  like a Mastcam-Z frame short of Navcam ties, from a single Navcam frame on (the station MAD test needed three).
- **79 / 110 mm bins off the focus lines** (`Settings.zcam_line_zooms` = (34, 48, 63)): a zoom without a real model
  keeps its refined values per bin (Chal Rocks ZR079 fy on lines: rms 42 px).
- **Frames without tie points** carry their label boresight elevation (sky or far field above 3 deg) and are counted
  per sequence in the run log (Overlook Mountain: 101 frames of four 48 mm sequences).

### Speed (Christian, 28 Oct: Chal Rocks 8.6 h with 79/110 mm, Taylorfjellet > 11 h)

- **`MPPP_THREADS`**: `run_sites` divides the cores among its jobs; the bundle adjustments, the matcher, the extractor
  and the guided matcher take that many threads (three Ceres solvers no longer oversubscribe the CPU).
- **`Settings.zcam_zooms_only`** (`run_pipeline --zooms`, default 34 48 63; `--all-site-zooms` for every zoom): the
  freeze runs keep the zooms being calibrated; 79 / 110 mm wait for a pass of their own.
- **Mastcam-Z pairs verified uncalibrated from the start** (`Settings.zcam_uncalibrated_matching`; `match
  (uncalibrated_families=("Z",))`): COLMAP's calibrated estimators abort on a pair whose keypoints fail to unproject and
  the pass before the retry was lost (0.4 h at Chal Rocks). Recorded in the matches record; a Mastcam-Z run key.
- **Two stage-2 rounds instead of three when the focus lines settle** (`Settings.zcam_line_settle_px` = 0.1): each
  refit records how far its lines moved the bins from where the round started (`focus_line_change_px`); below the
  limit the remaining rounds are skipped. 0 never skips.
- **Pose-guided matching for the weak frames, before their exclusion** (`GUIDED_DEFAULTS["scope"]` = "weak", through
  `Settings.pose_guided = {"scope": ...}`): the Mastcam-Z frames with fewer than `weak_navcam_ties` = 50 anchor ties, or
  no tie points at all, get their guided matches in the exclusion round, before the exclusion that would drop them
  (Overlook Mountain: 101 frames "no tie points" lost in round 2, then 735 k guided Navcam-Z matches they could not
  use); a frame without points gets its pairs from the partners' points in its frame and a depth window from them.
  "all" adds the every-frame pass after the rounds (0.94's +15-19 % ties at ~2 h), "final" is that pass alone. Report
  in `pose_guided_weak`, `database_guided_weak.db`.
- **Point thinning in the adjustments** (`Settings.ba_points_cap`, 0 = all; study S2): at most that many two-view,
  same-family, same-station points per image, spread over an 8 x 6 grid, every tie kept (`thin_points`); the
  adjustment reports `points_in_adjustment`. The points stay in the model.
- **`Settings.export_level = "light"`** (`run_pipeline`'s site runs unless `--export`): no native text model,
  residuals, `error_input_navcam/` or `gui_native/` - the joints read `sparse/<model>`, `project.json` and the health
  report; `summary.json`, `stations.csv`, `poses.csv` and `block_frame.json` stay.

- **Overlook Mountain's 48 mm frames are low-state** (Christian, 28 Oct): the low-backlash list takes
  ``<sol>[-<sol>]:<zoom>`` entries (`1150-1155:048`: every frame of that zoom on those sols; `-` takes them off), key
  `sol_zooms`. The list's fingerprint changes, so every Mastcam-Z run key does too (they realign anyway).
- **`scripts/windows/trial_joints_0p95.bat`**: the 0.95.0 joints over the blocks aligned on 0.94 (the Navcam joint with
  and without `--tie-f-slope`, the law check of the candidate, the Hazcam fit, the Mastcam-Z joints of 34 / 48 / 63 mm,
  the N+Z check on request), nothing promoted, no waiting for running blocks - into `camera_analysis\trial_0p95_<date>\`.
  The 0.94 blocks have focus bins, so the Mastcam-Z joint runs without sequence offsets.

- **Dense reconstruction, preloaded** (Christian, 28 Oct): `mppp dense WORK --prepare` writes `<project>/dense/`
  - the images undistorted to pinhole cameras from the native-pixel model, COLMAP's stereo configs and the rover
  masks undistorted the same way for the fusion; `--stereo` / `--fusion` / `--mesh` run the steps (a CUDA pycolmap
  for the stereo, `--python`). `dense_reconstruct.bat`, written next to `open_in_colmap.bat`, does the same with the
  COLMAP executable (a CUDA build first); the GUI's dense dialog takes `dense\` as its workspace.

### Studies (`mppp_studies.calibration.tie_point_studies`)

- S1 feature budget, S2 point thinning (against the all-points solution: frame centres, attitudes, per-bin focals), S3
  matcher and verification, S4 pair selection and the pose-guided scope, S5 filters, S6 the per-bin focal residual
  about the lines by observation band (for `MIN_BIN_OBSERVATIONS`): variant sets run as `mppp run --variant
  <study>_<variant>` on the three tiny blocks, measured the same way (completeness, quality, time per step).

### Navcam joint

- **Stereo differences as terms of their own**: `DS` = (NR - NL) of cx, cy and of their thermal slopes, tied exactly to
  the eyes' terms, with sd from the exact-tie covariance (`stereo_pp` in the result and the report). The two eyes'
  terms correlate at 0.98-0.997 (shared head pointing); the difference is the stereo property and is now readable.
  Holding NL's whole principal point and its slopes stays a `--hold` choice (`G_NL_cx`, `S_NL_cx`, `S_NL_cy`).
- **The law check's stereo uncertainty**: the pointing pass carries each block-era's stereo principal-point
  difference as a block of its own; its sd replaces `hypot(sd_L, sd_R)`, which was 14-18x too large with the eyes
  correlated at 0.997-0.999 (stereo chi^2/dof 0.01-0.13: the stereo law was never tested). `sd_source` on the rows.
- **The exact-tie step** is skipped when the ties are already met (their share of the cost below
  `EXACT_TIE_MET_SHARE` = 1e-6) and, once it found no decrease in a run, for the rest of that run (27 Oct: six calls
  of 4 minutes, never a decrease). **One more pass** is added only while the final costs have not settled
  (`PASS_SETTLE_TOL` = 0.1 %): every pass starts above the last one's end because the keypoint map is re-applied with
  the new slopes, which is not a reason for a pass.
- `--tie-f-slope` (`TIE_F_SLOPE`, off): NR's focal thermal slope tied to NL's ppm/degC, for the comparison run (27
  Oct: 35.80 and 35.87, 0.07 +- 0.13 apart).
- A held early-mission offset of the reference eye is held at 0 (`EARLY_HELD_AT_ZERO`: `E_NL_cy`), not at the 0.137 px
  the 24 Oct fit left in the model in use; NR's early cy is then the stereo early offset.

### Recorded for the paper-3 work (`mppp.sun`)

- Per image the **Sun-Mars distance** `d_mars_au` (SPICE with `spiceypy` and the kernels in `MPPP_KERNELS`, else the
  L_s ellipse; `d_mars_source` says which) and the **label sun vector in ENU** (`sun_enu`) - in `manifest.json`, on the
  project rows (computed from the label values of manifests before 0.95.0) and in `error_input/stations.csv`
  (`d_mars_au`, `sun_e/n/u`); per block `error_input/block_frame.json` (`R_block_from_enu`: the block's rotation from
  the label ENU frame, the health check's `block_rotation_deg`) and the sun in the block frame (`sun_bx/by/bz`).

### Operations

- **Camera models in use outside git** (`MPPP_CMODS`): `mppp cmods init DIR` seeds the folder with the package set and
  writes `scripts/windows/mppp_cmods.txt` for the batch files; promotions keep the folder complete; `mppp cmods
  status` compares it with the package copy; `mppp cmods ship` copies the models in use into `src/mppp/data/cmods`
  for the one commit that ships them. A moved folder does not rebuild the projects (the fingerprint decides). The
  shipped-model tests check form and provenance, not values.
- **`claude_sync.bat`**: fetches `refs/heads/main` only (no tags), prints the bundle's heads and the delivery's newest
  commit, stops before touching anything when the delivery is already here, not on `main` or behind this copy, and
  after the merge when `main` did not move; pushes `refs/heads/*` only; pops only its own stash; keeps the last runs
  in the log; refuses to run from a copy outside `scripts/windows`.
- **Side runs**: one status file per batch (`mppp_batches/<kind>_<pid>.json`; a batch is refused only when it shares a
  site with a running one), `run_done.json` is removed when a block is aligned again, and the joints wait
  (`--wait-h`, 36 h) for sites of their groups whose alignment is running.
- **Database copies** (`database_thermal.db`, `database_guided.db`) are made without the descriptors (SQLite attach /
  insert-select; the guided matcher reads them from the original) and removed at the end of the run
  (`Settings.keep_databases` keeps them).
- Version 0.95.0 (`v0p95`: output names `mppp_config_v0p95.json` ...; run keys do not carry the version).
