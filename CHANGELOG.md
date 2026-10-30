# Changelog

The development history before 0.95 is in `docs/history/` (`CHANGELOG_v0p90-v0p94.md`, `CHANGELOG_v0p13-v0p85.md`,
`CHANGELOG_v0p1-v0p12.md`).

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
- **make_figures** (`scripts/paper_figures.py`): the efl vs focus figure of the newest joint folder of each of 34, 48,
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

### Studies (`scripts/tie_point_studies.py`)

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
