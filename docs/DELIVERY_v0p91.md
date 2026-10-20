# MPPP 0.91.0 — what is in it, what was tested, what to decide (20 Oct 2026)

Branch `v0p91` (on top of 0.90 + the 20 Oct pass-2 Navcam models), delivered as `_transfer\mppp_latest.bundle`.
Adopt with `claude_sync.bat` once the Z34 run has finished (the running sites keep their code; the new Settings
change every run key, so the next `run_sites` aligns every site again).

## 1. The Navcam joint as one bundle adjustment — `mppp.calib.tied`

What Christian asked for on 20 Oct: one Navcam parameter set for every temperature and sol, each thermal term an
offset at −20 °C and one slope, everything optimised with the block poses at once, then the per-block, per-bin and
per-sol values plotted against the global solution.

| piece | where | state |
|---|---|---|
| the units: one camera per (eye, 2 °C bin, era), one rig per (block, epoch ≤ 25 sols, bin, era), every unit started at the law | `tied.tied_reconstruction` | tested (synthetic block) |
| the ties: constraint residuals (σ 1e-3 px, 1e-7, 1e-8) to G (10 per eye), S (5 slopes per eye), **E per eye** (4), Q0 (rig at sol 0), D (pitch, yaw, roll rates), H (the pitch hinge; yaw and roll linear) | `tied.make_ties` | tested |
| within a bin each image's own temperature counts (keypoint map of the current slopes, updated per pass) | `tied.keypoint_terms` | as in the prototype |
| one normal matrix → **finite covariance of every global term** (the v0p83 prototype's sd were infinite: the Q0 yaw ↔ NR cx null space; Q0's yaw is held, like the runs hold it) | `tied.adjust_tied`, `globals_result` | tested: sd finite on the synthetic block |
| the diagnostic pass: globals held, one camera per (block, eye, bin, era) and one rig per (block, epoch) free, no ties, one adjustment | `tied.diagnostic_pass` | tested |
| the candidate files in the frozen form: params at T0, `thermal`, `early_mission` per eye with sd, `drift` with `sol0 = 0`, `knot_ref = 0`, three rates and the hinge; `consensus.json` for the gate; `rig_residuals.json` + `thermal_bins.json` for the figures | `tied.write_tied` | tested; `forms.validate_rig` passes |
| the promotion gate: `block agreement` (share of block-bin cameras off the law by > 150 ppm and of epochs off the trend by > 5 mdeg, each ≤ 25 %) in place of the leave-one-block-out; the other 0.90 checks unchanged | `fit.check_navcam` | tested |
| the figures: the refined rig-trends and intrinsics-vs-T figures take the diagnostic pass's epochs and bins (`figure_inputs`); `paper_figures.py` reads `thermal_bins.json` when present | `tied.figure_inputs` | tested (both figures render) |
| the commands: `navcam_calibration_study.py consensus` (`--joint tied` default, `--joint profiles` = 0.90), `run_pipeline.py` passes `--joint tied`; `mppp calib navcam --root D:\scapes [--promote-if-ok]` | studies, `mppp.cli` | arguments tested; **not run on real blocks here** (no scapes in this session) |

Cost: the tied joint thins to 3000 points per block (the profile joint used 8000 and ~90 adjustments);
expect ~2 passes × one adjustment of 16–23 blocks + the diagnostic pass — tens of minutes, not 5–8 h.

## 2. The block runs

- `navcam_rig_roll = "hold"` (Settings; `project.settings`; `bundle_adjust` holds quaternion z with the yaw's y).
- Rig epochs: fewer than 16 Navcam images or one station → joined to the nearest epoch
  (`project.merge_small_epochs`, recorded in `navcam_rig_epochs.merged_small_epochs`).
- `zcam-v2`: `zcam_tangential = "refine"` — p1, p2 refined per eye in stage 1, held in the focus bins, on the lines
  as constants, averaged over the blocks in `fit_zcam` (with sd and between-block τ), checked (`max_p` 5e-3,
  `max_tau_p` 1e-3). The form id of a new calibration is `zcam-v2`; the 0.87/0.90 file (`zcam-v1`) loads unchanged.

## 3. Maps and CLI

- The 20 Oct figure code is in `studies/m20_sites/m20_sites` (six figures, orthomosaic background, the table
  sized to the block count, `M20_TAG`); `mppp maps all|zooms|figures …` runs it (one process per step, the
  `M20_*` settings from the options); `make_maps.bat` is one line over it; `[maps]` extra.
- `mppp doctor` prints the environment, dependencies, camera models and waypoint age (exit 1 when
  `MPPP_SCAPES`/`MPPP_PDS` are missing — the 10:43 failure of 20 Oct).
- Not done: the port of the figure bodies into `mppp.waypoints.maps` with a `MapContext` (handoff §6.1–6.2). The
  figures are paper material, not pipeline, and the study scripts are what Christian refined; see decision 6.

## 4. Tests

`tests/test_calib_tied.py` (units and law, adjustment with covariance and files and the gate, figure inputs),
`test_sfm_project.py` (small-epoch rule), `test_sfm_reconstruction.py` (roll hold), zcam-v2 expectations in
`test_calib`, `test_sfm_zcam`, `test_pipeline`; `test_cli` (doctor, maps, calib arguments); the study's
`test_from_sites`. Full suite: see the delivery message.

## 5. Decisions before the final delivery

1. **Q0 yaw held** in the tied joint (default `hold = ("Q0_yaw",)`): a constant yaw and NR's cx0 are one null
   direction; holding the yaw at the file's value (as every run does) makes the covariance finite. Alternative:
   hold NR cx0 instead and free the yaw. Recommendation: keep the yaw held.
2. **The pitch knot**: the tied joint cannot fit the knot (non-linear); it takes the rig file's `knot_sol` (300
   today). The 20 Oct free-knot refit on the epochs gave 250 (200–290). Recommendation: run with `--knot 250`
   (`mppp calib navcam --knot 250`), and have the diagnostic pass's epochs re-estimate the knot each time
   (`fit_rig_drift(knot_sol="free")` on `rig_residuals.json`, already in `paper_figures.py`).
3. **The gate**: block agreement ≤ 25 % of block-bin cameras off by > 150 ppm and of epochs off by > 5 mdeg, in
   place of the leave-one-block-out (which would cost N tied joints). The 20 Oct block table had 7 of 16 blocks
   "no" at 100 ppm / 5 mdeg, most of them NR's early offset (now per eye) and fy; 150 ppm / 25 % is my guess at
   where a healthy solution lands. Set the limits after the first real run.
4. **Points per block** 3000 (the prototype's "shrink the blocks") vs 8000: 8000 gives the same cameras with
   tighter sd at ~3× the solve time. Recommendation: 3000 for the pipeline passes, 8000 for the final one.
5. **The 0.91 run**: every site realigns (run key), ~8 h Navcam + ~1.5 h per Z34 block. Recommendation: before
   that, run the tied joint once on the finished 0.90 blocks as they are (`mppp calib navcam --root D:\scapes
   --knot 250`, no promotion) to see the solution, its sd and the diagnostic scatter — ~1 h on the PC, can run
   while Z34 is going (CPU only).
6. **Maps**: keep the figures as the study scripts behind `mppp maps` (decision: done), or port into the package
   per the handoff. The wheel does not ship studies; JOSS only needs the figures. Recommendation: keep.
7. **zcam-v2** changes the Z34 start model's form on the next calibration; the running Z34 (v1) is unaffected.
   If p1, p2 come out beyond 5e-3 or scatter > 1e-3 between blocks the check fails and v1's model stays.
