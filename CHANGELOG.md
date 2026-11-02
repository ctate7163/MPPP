# Changelog

The development history before 0.95 is in `docs/history/` (`CHANGELOG_v0p90-v0p94.md`, `CHANGELOG_v0p13-v0p85.md`,
`CHANGELOG_v0p1-v0p12.md`).

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
- **`scripts/zcam_aspect_geometry.py`** (Christian, 1 Nov: "Could fy be either a block scale change for distant
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
  analysis (`scripts/zcam_aspect_geometry.py`, 1,151 bins): no gravity sag and no range bias - within blocks the
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
  was seen half as often. For mask v4.2: `train_masks.bat --tag v8_v4.2 --zoom-pairs` (after the freeze run).
- **Mask A/B across sets that grew** (2 Nov): `compare_mask_models.py --also-held-out CKPT ...` keeps only the frames
  those models held out too, each split rebuilt as it trained (`items_as_trained`: the newest masks left out to the
  card's n_train + n_val - the v8 set grew from 4,095 to 4,099 masks between v4 and v4.1, and the shuffle of the
  sorted names changes with every mask). The chain's A/B uses v4.1's split with `--also-held-out` v4.
- **`freeze_chain.bat`** (Christian, 2 Nov: away while it runs): waits for the mask training, the mask A/B of v3, v4
  and v4.1 (`--seen-split`), the point-thinning study S2 with its verdict (cap 1500 when centres < 1 cm p95,
  attitudes < 0.1 mdeg p95, focal lengths < 0.05 % max), then `run_pipeline.py --reset` (the state of 28 Oct had the
  ecam stages done: without `--reset` the freeze run would have started at zcam_runs), then
  `<root>\paper_studies\run_paper_studies.py` if it exists. Log and status in `<root>\mppp_pipeline\freeze_chain_*`.
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
  included). `scripts/release_mask_model.py upload --extra` uploads the card's figures.
- **`scripts/compare_mask_models.py`**: two or more mask models on the same held-out frames, per camera, with
  `--seen` (the older model's training set) for the frames neither model trained on.
- **Documentation**: new pages *Downstream exports* (dense, Metashape, Nerfstudio), *Rover masks* and *Roadmap*;
  the API reference adds `mppp.calib.hazfit`, `mppp.sfm.dense`, `metashape`, `nerfstudio`; Methods §5 (masks)
  updated (threshold 0.5, the registry models, sky included); the index's scope; `docs/RELEASING.md` names the right
  upload command (`scripts/release_mask_model.py`, not `mppp.mask.hub`). The working notes (`AUDIT_v0p90`,
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
  that has CUDA torch, then `scripts/train_mask.py --paper`: `figure1_masks.png/.pdf` and per-camera metrics on the
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
