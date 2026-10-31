# MPPP handoff: 0.87 findings, read with the 0.90 layout

**Updated 19 Oct 2026 (evening) for 0.90.0.** The findings below (sections 3-4) are current; the code references in
sections 1-2 were written for 0.87 and are translated here:

| 0.87 said | 0.90 is |
|---|---|
| notebook 03 is the pipeline; `runner.run_notebook`; `ALIGN_RULES` in `runner` | `mppp.pipeline` (`Settings`, `process_site`, `align`, `run_site`, `run_key`) and the `mppp` command (`mppp process`, `mppp align`, `mppp run`); notebooks `01_process_images`, `02_colmap_alignment`, `03_error_analysis` call it |
| `mppp.sfm.sites`, `mppp.maps` | `mppp.waypoints` (`.table`, `.sites`, `.blocks`); the map prototype is `studies/m20_sites/` |
| `mppp.sfm.cmods_form`, `navcal*`, `zcam_calib`, `zcam_lines`, `diagnose`, `navcal_tied` | `mppp.calib` (`forms`, `models`, `constrain`, `fit`, `joint`) holds the frozen forms, the promoted models, their constraints and the streamlined fits; the studies are in `studies/calibration/` with their scripts (`run_pipeline.py`, `navcam_*`, `zcam_*`, `run_v0p87.bat`, `zr_diagnose_z34.bat`, `install_lightglue.bat`) |
| `matching.match(matcher="sift_lightglue")` through COLMAP's ONNX LightGlue, probed in a subprocess | `match(mode="lightglue")`: COLMAP SIFT features matched by LightGlue through kornia/torch (`mppp.sfm.lightglue`), verified by COLMAP; no ONNX build needed. Without kornia, or above `max_hours` per block (projected from the first 20 pairs), the block uses COLMAP's SIFT matcher and the matches record says `matcher: sift` against `matcher_requested: lightglue` (the 0.86 record bug is fixed). `install_lightglue.bat` (ONNX route) is kept in `studies/calibration/windows/` only |
| `cmods/history/`, `*_frame.xml`, `*_rational.json` | gone from the package; git history keeps replaced models; `scripts/promote_cmods.py` logs to `cmods/CHANGES.md` |
| `_transfer/`, `claude_sync.bat`, `render_paper.bat` after sync | no bundle workflow in the repo; the paper renders with `python scripts/build_paper.py` (inara docker or pandoc); `render_paper.bat` wraps it on Windows |
| `D:\scapes`, `D:\data\m2020`, `GPU_PY` path defaults | environment: `MPPP_SCAPES`, `MPPP_PDS`, `MPPP_GPU_PYTHON`, `COLMAP_BAT`; `MPPP_CMODS` for a model folder |
| `test_v0pNN.py` | tests named by module; the 0.86/0.87 behaviour tests live in `test_sfm_reconstruction.py` (third-view rule, unsurveyed priors, k2, bogus-camera limits) and `test_sfm_thermal.py` (bin camera for an eye without images) |

Everything from 0.86 and 0.87 is in 0.90: the third-view rule for Navcam + Mastcam-Z points, the 10 m / 3 deg
priors of unsurveyed stations, the bogus-camera limits and `clamp_k2`, the temperature-bin camera fix, the
`zcam_guard`, Three Forks Large out of `navcam_consensus`, and the South Arm Mastcam-Z 34 mm model.  Still open from
0.87: the `backlash.py` rig pattern (same shape as the thermal bug), approach-station start positions, 3 vs 4 views
for mixed points, the upright-SIFT test, Hazcam calibration (0.90 ships a start model).

---

# MPPP v0p87: handoff as written on 19 Oct 2026 (evening)

Read this first. It covers how this project is worked on, what changed since the v0p85 delivery, what the data showed since v0p80, what is running, and what is open. Older handoffs are in the claude.ai project "Photogrammetry" (`claude/mppp-*.md`). Read the latest two (`mppp-v0p80-start-here-...`, `navcam-review-2026-10-18-pass2-...`) for the background.

- **Code:** 0.87.0, on top of 5d7a689 (0.86.0). Delivered as `D:\code\MPPP\_transfer\mppp_latest.bundle`. GitHub `ctate7163/MPPP` gets it when Christian runs `claude_sync.bat`.
- **Tests:** full suite 548+ passed, 7 skipped, about 11 min. Run it detached (`nohup ... &`) and poll, because the cloud shell times out at 10 min.
- **Next run:** `scripts\windows\run_v0p87.bat` (was `run_v0p86.bat`): 16 Navcam blocks, then the 7 Mastcam-Z 34 mm blocks. Estimated 20-22 h, or 27-29 h if the Z34 calibration passes and the 7 Z blocks rerun (`zcam_final_runs`).

## 1. How this project is worked on

- **Two machines.** Claude works in a cloud container with a clone of the repo (`/home/claude/MPPP_repo`). Christian's PC (Windows, `D:\code\MPPP`, data in `D:\scapes`, PDS archive `D:\data\m2020`, GPU conda env `C:\Users\ctate\AppData\Local\miniconda3\envs\mppp_gpu`) runs everything heavy.
- **The PC link.** It gives list, stage (PC → cloud) and commit (cloud → PC) of files only. There is **no shell on the PC**, so nothing can be run there by Claude. Christian double-clicks the .bat files.
- **Delivery.**
  1. `git bundle create ... --all` here, then commit the bundle to `D:\code\MPPP\_transfer\mppp_latest.bundle` (force).
  2. Christian runs `_transfer\claude_sync.bat` (or `scripts\windows\claude_sync.bat`). It fast-forwards, pushes to GitHub, writes `_transfer\mppp_pc.bundle`, and since v0p86 renders the JOSS paper.
  3. If the PC has its own commits (a promotion made by a run), the sync stops and changes nothing. Merge `_transfer\mppp_pc.bundle` here, then deliver again.
- **Never sync while a run is going.** Each block starts a fresh Python process, so code synced mid-run changes the rest of the run.
- **The stop hook** says there are unpushed commits after every turn. That is expected; pushing happens through `claude_sync.bat`.
- **Logs to stage when following a run:**
  - `D:\scapes\run_sites_log.txt` (one line per block start and finish);
  - `D:\scapes\<block>\runs\<timestamp>\log.txt` (the notebook log; the executed notebook is next to it, with the traceback);
  - `D:\scapes\mppp_pipeline\{pipeline_log.txt, state.json, pipeline_status.json, *.log, REPORT.md}`;
  - `D:\scapes\camera_analysis\...` (consensus and calibration studies).
- **Block folders** are `D:\scapes\mars2020_sol_<first sol>_<site>[_zcam]\colmap` (since v0p85; `D:\scapes\colmap\..._colmap` before). The final model is `sparse\cahv_ba`. `project.json` has every image with its station, temperature, priors and camera.
- **Christian's preferences:**
  - Plain, direct answers; one question at a time.
  - Plots: left camera cool colour, right camera warm; left/right subplot pairs share y range and ticks; key x values (reference temperature or focus) as vertical dashed lines; the notebooks' plain matplotlib style.
  - Christian decides site and group membership. Overlapping blocks in `navcam_consensus` are intended.

## 2. What changed since v0p85 (8ca72e2) — read before touching the code

### 0.85.x (19 Oct, during the day)
- **Block folders without `_colmap`** in `D:\scapes`. `scripts/move_scapes.py` renames old ones; every run .bat calls it first.
- **Sub-frame rule:** `NAVCAM_MIN_FRAME_FRACTION` 1/3 (was 0.2). The v0p82 mask rule `MIN_INCLUDED_FRACTION` is off. ALIGN_RULES 85.
- **`NEARBY_FROM_FIRST_SOL`** (notebook 03): nearby stations add nothing before the block's first sol. Site edits by Christian are in `sites.json`, with notes last in each site.
- **The ZR dropout, found and fixed** (`reconstruction.triangulate`, ALIGN_RULES 86).
  - COLMAP's triangulator silently skips every image of a "bogus" camera: any extra parameter beyond ±1 (`max_extra_param`), or a focal length beyond 10× the image size.
  - The Mastcam-Z 34 mm FULL_OPENCV k2 is about 1.0–1.05. Whenever the ZR focus line set k2 > 1, every ZR frame lost all its tie points at the next triangulation.
  - This explains every right-eye loss since 5 Oct.
  - `triangulate()` now sets `max_extra_param` 1e6 and focal ratio 1e-3..1e3 (`BOGUS_*`).
- **`clamp_k2`:** k2 kept within ±10 after every adjustment and on the focus lines (`K2_LIMIT`).
- **`mppp.sfm.zcam_guard`:** after every stage-2 triangulation it counts the frames with tie points per eye and zoom. A collapse is diagnosed and recovered: line values undone, then poses, then the pose-guided step. `MPPP_ZCAM_TRACE=1/2` writes `colmap\zr_trace.jsonl` (1 = records only; 2 = also models after each triangulation). `scripts/zr_report.py` makes `zr_diagnosis.md`.
- **Groups:**
  - `navcam_consensus` now 17 blocks (see 0.86 below);
  - `zcam34_consensus` 7 blocks (chal_rocks, rochette, threeforks_north, rio_chiquito, south_arm, origny, marble_mountain);
  - `zcam34_diag` (rochette, south_arm, marble_mountain).
- **`zr_diagnose_z34.bat`:** aligns the `zcam34_diag` blocks into `colmap\` with the trace, writes the report, then runs the Z34 calibration with `--promote`.
- **Other:** `run_pipeline.py --stop-before STAGE`; 12 unused .bat files removed.

### 0.86.0 (19 Oct, afternoon)
- **SIFT + LightGlue matching is the default** (`matching.match(matcher="sift_lightglue")`; notebook 03 `MATCH["matcher"]`).
  - Before matching, `lightglue_probe()` runs LightGlue in a *separate process* on synthetic images (a pycolmap without ONNX aborts the whole process). Round 1 is small images; round 2, only if round 1 is fast, uses dense images at the pipeline's feature setting.
  - The result is cached in `%LOCALAPPDATA%\mppp\lightglue_probe.json`, keyed by python, pycolmap version and the `_core` build.
  - The block uses SIFT brute force, with a log note, when LightGlue is unavailable or its projected time for the block's pairs exceeds `LIGHTGLUE_MAX_HOURS_PER_BLOCK` (2 h).
  - **On the PC it currently always falls back, but not for lack of ONNX.** The probe in the v0p86 run (19 Oct 17:34) shows the GPU env `mppp_gpu` (conda-forge) already has ONNX Runtime with the CUDA execution provider; LightGlue failed with "cuDNN is unavailable for CUDA Execution Provider: LoadLibrary failed for cudnn64_9.dll". **Likely fix: `conda install -n mppp_gpu -c conda-forge cudnn` (cuDNN 9 for CUDA 12)**, then delete `%LOCALAPPDATA%\mppp\lightglue_probe.json` so the probe runs again. That would make `install_lightglue.bat` unnecessary. Not done: never change the GPU env while a run is going. Then check the probe's ms per pair (keypoints ~50k per image: probably too slow for the 2 h limit on big blocks).
- **`install_lightglue.bat`** (not run; Christian decided not to before this run).
  - Clones `mppp_gpu` to `mppp_gpu_lg` and installs conda-forge `colmap=4.2.1=cuda_<CU>*_1` (the `_1` builds depend on `onnxruntime-cpp`), the CUDA `pycolmap` 4.2.1 and CUDA `onnxruntime-cpp`.
  - Tests with `lightglue_check.py`. Only on PASS, and with no MPPP run going, it writes `%LOCALAPPDATA%\mppp\gpu_python.txt`. `--off` deletes it.
  - Notebook 03: `GPU_PY = gpu_python(default)` (`MPPP_GPU_PY`, then that file, then the old path).
  - Unverified: whether the win-64 `_1` build really has ONNX (Linux `_1` builds list `onnxruntime-cpp`; win-64 could not be inspected from the cloud).
- **Three Forks Large left out of `navcam_consensus`** (16 blocks). It cost 1.9 h per run and moved the cameras by only 0.003 px (centre) and 0.008 px (corners) in the 18 Oct leave-one-out. It stays in `navcam_baseline`.
- **Navcam + Mastcam-Z points need a third view** (`reconstruction.drop_mixed_short_tracks`, `MIN_MIXED_TRACK_LENGTH = 3`).
  - Applied after every triangulation (inside `triangulate()`) and in `drop_short_tracks`. Project setting `min_mixed_track_length` overrides.
  - Same-camera two-view points (the stereo pairs) are kept.
  - Why: §3.4.
- **Fixed: Marble Mountain stage-2 crash** ("SQLite error: constraint failed" in `sync_database`).
  - `thermal.split_by_temperature` built a temperature-bin rig with the eye's *original* camera when that eye had no registered image in the bin, putting one camera on two rigs. The COLMAP database allows a camera on one rig only.
  - Such an eye now gets a held copy of its camera (`_bin_cam`; all parameters fixed, `thermal_bin: True`, not in the bin table).
  - Regression test: `tests/test_sfm_thermal.py::test_bin_rig_gets_its_own_camera_for_an_eye_without_images`.
  - **The same pattern exists in `backlash.py` (~line 690, `m.get(ref, ref)`) for the Mastcam-Z regular-state split.** It has not bitten yet, because the blocks so far had no regular-state frames in a rig with a backlash partner. Watch for the same SQLite error there.
- **`render_paper.bat`:** pandoc + citeproc, xelatex first, renders `paper\paper.md` → `paper\paper.pdf` after every `claude_sync.bat` (only then). `paper.pdf` is git-ignored. **`paper\paper.md` is not in the repo yet, and `D:\code\MPPP\paper` was empty.** Christian is to put `paper.md` and `paper.bib` there; then commit them.

### 0.87.0 (19 Oct, evening)
- **Stations without a waypoint of their own: 10 m position and 3° attitude priors** (Christian), against 1 m / 1° for surveyed stations.
  - `reconstruction.is_unsurveyed(row)` is true for `position_source` "waypoint site N drive 0 + label ORIGIN_OFFSET_VECTOR".
  - `UNSURVEYED_PRIOR_SIGMA_M`, `UNSURVEYED_ATTITUDE_PRIOR_DEG`; project settings override.
  - The prior costs are our own (pycolmap cost functions in `bundle_adjust`), so each frame simply gets its own covariance.
  - The gauge recentring (`prior_gauge`) weights by 1/σ².
  - The record has `frames_unsurveyed`. The COLMAP database pose priors (`database.py`) still carry 1 m for every image; the adjustment does not use them.
  - Stations with fewer than 3 images still lose their position prior when tied (`LOCALIZE_MIN_IMAGES`).
- **Mastcam-Z 34 mm start model promoted from South Arm alone** (Christian: better than the v0p75 start).
  - `zcam_calib.calibrate([South Arm], form v0p75, regular derived)`, run in the cloud on the staged block.
  - The calibration's "blocks ≥ 2" check fails and was waived on purpose.
  - Report: `docs/results/zcam34_calibration_20261019_south_arm/`. The old file is in `cmods/history/20261019-155805/`.
  - Numbers in §3.3.
- **`run_v0p87.bat`**, version bump, this file.

### Known issue introduced in 0.86 (not fixed)
- `matching.match` writes the *requested* matcher (`sift_lightglue`) into the matches record, even when the block fell back to SIFT.
- If LightGlue later becomes available, those blocks' SIFT matches would be reused as if they were LightGlue matches. That would spoil a SIFT-vs-LightGlue comparison.
- Fix: record the matcher actually used and compare against what this PC would use. It is harmless as long as LightGlue is never available.

## 3. What the data showed since v0p80

### 3.1 Navcam camera models (pass 2 of 18 Oct; project doc `navcam-review-2026-10-18-...`)
- **Residuals and lens:**
  - Median 0.164 px, rms 0.337 px.
  - Residuals rise from 0.155 px (centre) to 0.21 px (corners).
  - Leave-one-block-out: the worst block moves the lens 0.011 px (centre) and 0.027 px (corners). **More of the same data no longer changes the lens.**
- **Thermal:**
  - fx/fy: NL +40.7/+45.2, NR +37.6/+43.5 ppm/°C. fy − fx ≈ +5 ppm/°C is a real anisotropy.
  - k1(T): NL −7.2e-6/°C (z −4.5); NR not significant.
  - Bins colder than about −35 °C sit up to 1 px above the line: curvature or thermal lag.
- **Stereo:**
  - NR cx(T) −0.0525 px/°C. The 0.80.1 value of −0.0714 was too steep, and that alone explains the +300 ppm/°C correlation of block scale with temperature seen on 17–18 Oct.
  - Rig yaw rate +2.56 ± 0.10 mdeg/1000 sol. No yaw-vs-temperature term (z −0.2).
  - Rig epoch scatter: pitch 1.0, yaw 2.3, roll 3.9 mdeg. Rochette's sol-341 epoch is a +20 mdeg roll outlier.
- **Early mission** (sol < 380): fx +0.20, fy +0.25, cx +0.32, cy −0.57 px, all |z| ≥ 5.8.
- **Baseline:** 0.42401 m in use. The 14 large blocks with cameras held give 0.42438 m. Per-block baselines span 0.42188–0.42435 m. The baseline study produced no sd (covariance bug, fixed in 0.82.1).
- **Lens form:** THIN_PRISM_FISHEYE (angle-space tangential) beats FULL_OPENCV/rational on every test. k5 adds nothing. Keep it.
- **Full-resolution frames:**
  - Median residual 0.13–0.18 px against 0.25–0.30 px at half resolution and ~1 px at quarter.
  - About 5× the information per image.
  - Most frames are downsampled.
- **Which blocks constrain what:**
  - Corners: compact close-range blocks (Cheyava Falls, Bunsen Peak, South Arm).
  - Thermal: wide temperature span on the same terrain (Rochette/Artuby, Taylorfjellet, Bunsen Peak).
  - Yaw and scale: long views and many waypoints (Three Forks Large, Bell Island Large, Taylorfjellet Huge, Origny Large).
  - Drift: revisits (Chal Rocks, Rochette + Artuby).
  - Image and station counts predict little.
- **Covariance failure** in v0.74.7–0.82.0 was a code regression (owned camera arrays), fixed in 0.82.1. "Not converged" in pass 2 was the form change (k4 free) moving f and the radial terms together, not iteration noise.

### 3.2 Timing (for planning)
- **Navcam runs, 18 Oct, per block (h):**

  | block | h | block | h | block | h |
  |---|---|---|---|---|---|
  | Van Zyl Large | 0.63 | Rockytop | 0.87 | Witch Hazel | 0.48 |
  | Chal Rocks Large | 0.61 | Enchanted Lake | 0.26 | South Arm | 0.27 |
  | Rochette | 0.27 | Three Forks Large | 1.87 | Bell Island Large | 1.11 |
  | Seitah North | 0.27 | Belva Large | 0.56 | Taylorfjellet Large | 0.95 |
  | Issole | 0.32 | Tuxedo Park | 0.29 | Taylorfjellet Huge | 1.66 |
  | Artuby | 0.25 | Bunsen Peak | 0.70 | Origny Large | 1.00 |
  | Chal Rocks | 0.20 | Cheyava Falls | 0.64 | Marble Mountain | 0.38 |
  | Pico Turquino | 0.17 | Hippo Pools | 0.30 | | |

- **Consensus (23 blocks, 4024 images):** 8.2 h. ~280 s per joint adjustment:
  - thermal profiles 60 fits = 4.7 h (70 fits since NL cx(T) was added);
  - early offsets 1.4 h for 3 rounds (now 1 round);
  - leave-one-out 1.7 h.
  - The time scales with the images in the merge.
- **Navcam + Z34 blocks (19 Oct diagnostic):** Rochette 1.34 h, South Arm 1.79 h.
- **SIFT matching on the GPU:** ~10 ms per pair (Rochette Z34: 28,680 pairs in ~5 min).
- **LightGlue:** the pipeline extracts up to 32,768 features, which gave ~52,000 keypoints on dense synthetic texture (the cap did not bind as expected). Expect 0.3–1.5 s per pair on a GPU.

### 3.3 Mastcam-Z 34 mm (19 Oct diagnostic, right-eye fix in)
- **Right eye kept to the end:** Rochette ZL 95/95, ZR 95/95; South Arm ZL 97/97, ZR 96/96 frames with tie points.
- **The two blocks agree** (figure: `D:\scapes\mppp_pipeline\zcam34_intrinsics_vs_focus_20261019.png`, CSV next to it):
  - fx(600): ZL 4720.9 / 4720.7 px, ZR 4727.8 / 4726.4 px (Rochette / South Arm).
  - Slopes: ZL 0.032 px/count in both, ZR 0.050 / 0.049. **The v0p75 start model was too steep: 0.0505 and 0.064.**
  - fy − fx: 15 px at Rochette, 11 px at South Arm, in both eyes. **The aspect is the parameter that differs between blocks.** In the 18 Oct run the calibration failed on "ZL fy between-block sd 10.8 px", and the 2-block calibration of 19 Oct has fy chi²/dof 13.8 (ZL) and 9.5 (ZR).
  - k1: ZL −0.458/−0.459, ZR −0.477/−0.473. k2: ZL 0.654/0.678, **ZR 1.043/1.050** (the start model had 0.875).
  - ZR cy follows focus at 0.0012–0.0014 px/count (model 0.0017). cx is held at pp0 in both eyes, and ZL cy too.
  - Rig rotation vector: (+1131.4, +3960.6, −610.2) and (+1129.6, +3953.7, −608.2) mdeg; baseline 243.56 mm.
- **Promoted model (South Arm):** ZL fx(600) 4720.58 px, fy 4731.35 px, slope 0.0320; ZR fx 4726.30 px, fy 4737.02 px, slope 0.0485; k1/k2 ZL −0.4589/0.6776, ZR −0.4732/1.0504; ZR cy slope 0.00122; regular state = backlash/1.0095.
- **The 19 Oct 2-block calibration** (`D:\scapes\camera_analysis\zcam34_calibration_20261019_2`) failed only on the block count. Marble Mountain had crashed (fixed in 0.86).

### 3.4 Two-view points and false Navcam–Mastcam-Z matches (Rochette, South Arm)
- **Method:** height of each point above a terrain surface built from points seen in 4+ images (0.5 m cells). Script logic is in this session's notes; easy to redo with `pycolmap.Reconstruction` + numpy.
- **Off by more than 0.25 m:**

  | point type | Rochette | South Arm |
  |---|---|---|
  | two-view Navcam + Mastcam-Z | 36% | 47% |
  | three-view Navcam + Mastcam-Z | 7.8% | 8.8% |
  | four-view Navcam + Mastcam-Z | 1.8% | 2.7% |
  | two-view, one camera type (stereo pairs) | 1.1% | 0.9% |
  | three-view, one camera type | 0.4% | 0.3% |

- The bad ones have *low* reprojection error: epipolar-consistent false matches.
- Stage 2's pose-guided Navcam–Mastcam-Z matching roughly doubles those ties (Rochette 59k → 105k, South Arm 48k → 103k) and is the likely source, not proven. Same failure mode as COLMAP guided matching at Three Forks (v0p64.1).
- A global minimum track length of 3 was considered and rejected. It removes 54% of the points and 38–43% of the Mastcam-Z observations (mostly good ZL–ZR stereo), and it would leave South Arm frames with < 30 tie points. Hence the mixed-only third-view rule.
- **Possible next step:** require 4 views for mixed points (1.8–2.7% off), or tighten the pose-guided matching's epipolar threshold.

### 3.5 Stations without a waypoint (approach stops) — Christian wants to keep them
- **What they are:** mid-drive and approach stops have no localised waypoint. Their prior is the site's drive-0 waypoint plus the label `ORIGIN_OFFSET_VECTOR` (onboard odometry since the site start). The project's `position_source` says "waypoint site N drive 0 + label ORIGIN_OFFSET_VECTOR".
- **They get the same priors as surveyed stations:** 1 m per axis (`prior_sigma_m`) and 1° attitude (`ATTITUDE_PRIOR_DEG`, isotropic in the camera frame).
- **Rochette** (sites S006D2594, S006D2648, S006D2684 without waypoints):
  - S006D2594 moved 0.9 m N and turned **+1.53° in yaw** (tilt 0.2°). Every other Navcam frame is within 0.3°.
  - S006D2684 moved **6.3 m E** against its 1 m prior.
  - Their Navcam stereo points sit −0.6 to +0.5 m off the terrain, growing with distance. 10% of these stations' stereo points are off, against 0.5% at South Arm's secondary stations.
  - Figure: `D:\scapes\mppp_pipeline\rochette_two_track_points_20261019.png`.
- **S006D2594's ties to the rest were almost all Mastcam-Z:** 3,846 two-view N–Z, 1,038 three-view Z, 321 four-view, and only 90 Navcam–Navcam. So its 1.53° may have been forced by false matches. The third-view rule leaves it ~1,600 ties; the v0p87 run will show whether its stereo lands on the terrain.
- **v0p82 run, same pattern:** Van Zyl's 14 two-image stops (NCAM00301) drifted −1.6 E / +0.7 U m at sol 20 to −3.9 E / +1.8 U m at sol 73. Their priors are dropped (`LOCALIZE_MIN_IMAGES` 3).
- **Built in 0.87.0:** 10 m / 3° priors for these stations (isotropic attitude; Christian's numbers).
- **Proposed, not built:**
  1. **Better start:** chain from the nearest surveyed drive of the same site instead of drive 0: C = waypoint(surveyed) + [offset(this) − offset(surveyed)], using the labels of that drive's images in the block. For S006D2594 that is S006D2666 (72 drive counts later, same sol).
     - The label offset is not stored in `project.json` now. Add it in `image.py _position` / `project.py`.
     - Alternative: `Mars2020_waypoint_shifts.csv` interpolated between localised drives (idea from 17 Oct).
  2. **Optional refinement of the built rule:** tilt is better known than heading (IMU), so an anisotropic attitude prior (tilt 1°, heading 3–5°) would keep more information. World-frame covariance diag(σt², σt², σψ²) rotated into the camera frame, Σ_cam = R_w2c Σ_world R_w2cᵀ; check pycolmap `AbsolutePosePriorCost`'s perturbation convention first. Code: the prior loop in `reconstruction.bundle_adjust`.
  3. Health report: unsurveyed stations with their solved shift and rotation in prior σ.

### 3.6 Earlier findings still relevant
- **Chal Rocks Sid:** site 11 stations sit 0.25 m off their waypoints (a localisation-frame offset at the site 10 → 11 change). Idea, not built: one translation per rover site.
- **Label pointing:** one rover site at Bell Island (S072D0000) and Witch Hazel (S066D0000/0070) is ~3.2° off in the label pointing (health `attitude_residual_p95`).
- **Waypoint check, 13 blocks:** mean scale +73 ± 640 ppm, block-to-block sd 2300 ppm. Three Forks Large +3503 ppm, Hippo Pools −3418 ppm.

## 4. What is running / next
- **The v0p86 full run is going** (`run_v0p86.bat`, 0.86.0 = 5d7a689; `run_pipeline.py --reset --from navcam_runs --no-baseline-study`, started 19 Oct 17:25).
  - It runs with the v0p75 Mastcam-Z start model and 1 m / 1° priors everywhere. The v0p87 changes (South Arm Z34 model, 10 m / 3° priors for unsurveyed stations) are for the run after it.
  - **Do not sync v0p87 until this run has finished.**
  - Progress at 19:47: OEB Landing Site 1.30 h, Chal Rocks 0.18, Artuby 0.13, Rochette 0.11, Issole 0.28; Seitah North Large running.
  - Issole's health: 128/128 registered, median 0.153 px, one WARN (8 outlier images).
  - Matching: SIFT (LightGlue fell back, see §2). The third-view rule is in.
  - An earlier start at 16:45 began the pass-1 consensus with 14 blocks; it was stopped and restarted at 17:25 with `--from navcam_runs`.
  - `zr_diagnose_z34.bat` was started twice at 17:27 and 17:30 while the run was going; `run_sites` refused both ("already running"). That guard works.
- **What to look at when the run finishes:**
  1. Navcam consensus (pass 3 effectively; first with NL cx(T) and k1(T) from bins): `pass_change.json` and convergence; rig residuals yaw vs T flat.
  2. Waypoint check: block scale against temperature, which should no longer correlate.
  3. Z34: the zcam_guard counts per round (no eye lost); the calibration from 7 blocks, especially fy and aspect scatter between blocks; whether it promotes.
  4. Rochette: whether the approach stations' stereo now sits on the terrain (third-view rule).
  5. Marble Mountain completes.
- **Open items:**
  - Approach stations: better start position by chaining from the nearest surveyed drive (§3.5).
  - Mixed points: 3 vs 4 views.
  - The `backlash.py` rig pattern (§2).
  - The matches-record matcher bug (§2).
  - LightGlue install (decide after the run).
  - Upright SIFT test (Christian asked; estimated 10–30% more correct matches in level scenes; untested).
  - The JOSS paper: `D:\code\MPPP\paper` now has `paper.md`, `paper.bib`, figures (PNG up to 10 MB, a JPG, two Mastcam-Z frames) and a rendered `paper.pdf`, all untracked. Decide with Christian what goes into git (at least `paper.md`, `paper.bib` and the figures `paper.md` uses; large PNGs maybe as JPG).
  - Joint thermal fit in one adjustment instead of ~110 profiles (prototype `mppp.sfm.navcal_tied`; its global sds come out infinite).
  - Per-rover-site frame offset.
  - Hazcam models (planned).
