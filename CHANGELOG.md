# Changelog

The development history before 0.95 is in `docs/history/` (`CHANGELOG_v0p90-v0p94.md`, `CHANGELOG_v0p13-v0p85.md`,
`CHANGELOG_v0p1-v0p12.md`).

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
