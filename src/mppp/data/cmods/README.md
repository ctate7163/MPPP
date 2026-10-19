# MPPP camera models (`src/mppp/data/cmods`)

The one folder of the camera models in use.  `MPPP_CMODS` points MPPP to another folder.  Everything that reads
them goes through `mppp.calib` (`load_camera`, `load_rig`, `load_zcam`; `camera_at`, `rig_at`, `zcam_camera_at`
constrain them to an image or block; `check_form` and `describe` say what is in use).

| file | what |
|---|---|
| `M2020_NL_fisheye_tangential.json`, `M2020_NR_fisheye_tangential.json` | the Navcam eyes: COLMAP `THIN_PRISM_FISHEYE` at the full frame 5120 x 3840 (`params` in `param_names` order, sx1 = sy1 = 0 held), at `thermal.T0_degC`, with `thermal.fx_ppm_per_degC`, `fy_ppm_per_degC`, `cx_px_per_degC`, `cy_px_per_degC`, `k1_per_degC` and the `early_mission` offsets (`dfx_px` ... before `before_sol`, `applied_terms`); `"form": "navcam-v1"`, `"convention": "NL reference"` (NL keeps the label CAHVORE pointing and position); `sd`, `verification` (the leave-one-block-out and residuals of the consensus) |
| `M2020_N_rig.json` | the Navcam stereo rig, x_NR = R x_NL + t: `R_sensor_from_ref`, `t_sensor_from_ref_m` (`"translation": "fitted"` when the projects use it, else each block keeps its CAHV baseline), `drift` (`<angle>_mdeg_per_sol` about `sol0`, the pitch hinge `knot_sol` / `knot_ref` / `<angle>_early_mdeg_per_sol`), one yaw for every temperature (`yaw_constant`) |
| `M2020_FL.json`, `M2020_FR.json`, `M2020_RL.json`, `M2020_RR.json` | the Hazcam eyes in the same form, when calibrated.  Without a file `mppp.calib.hazcam_start_model` is used (f = 2130 px, principal point at the frame centre, k1 = -0.25, k2 = +0.08, no thermal term; `"source": "start model (not calibrated)"`) with a warning |
| `M2020_ZCAM034_focus_model.json` | the Mastcam-Z 34 mm calibration (`"form": "zcam-v1"`): per eye (`cameras.ZL034`, `ZR034`) the `lines` per backlash state (`backlash`, `regular`) - `fx`, `fy` (`a + b (focus - lines.reference_focus)`), `k1`, `k2` (constant), ZR `cx`, `cy` (pinned at `pp0_px`, cy with its slope); `pp0_px`, `pp` (`cx_px_per_count`, `cy_px_per_count`), `distortion`, `regular_ratio` (backlash / regular focal length), `aspect`; the zoom's `rig`.  p1 = p2 = k3 = 0 held, no temperature term.  `M2020_ZCAM048_...`, `_063`, `_079`, `_110`: the other zooms when calibrated - a zoom without a file starts from the label CAHVOR of each image |

Backlash state: an image is in the `backlash` state (the higher focal length, about 1 % above the label) unless
`mppp/data/mars2020_mastcam-z_low_backlash_observations.csv`, the found list next to the WORK folders or its own
focal fit puts it in the `regular` state (the lower focal length).

**Before a batch:** `studies/calibration/prepare_next_run.py` checks that the models in use follow the forms
(`mppp.calib.check_form`); `run_sites.py` and notebook 03 refuse to start otherwise.  The models' fingerprint
(`mppp.calib.models_fingerprint`) is part of the run key: a promotion makes `run_sites.py` align every site again.

**Change the models in use only with `scripts/promote_cmods.py`** (`mppp.calib.models.promote`: checks each file
against its form, replaces the file, logs the change in `CHANGES.md`), then commit - the replaced models are in the
git history, there is no `history/` folder:

    python scripts/promote_cmods.py <candidate folder or files> --note "why these are better"
    python scripts/promote_cmods.py --list
    git add src/mppp/data/cmods && git commit -m "cmods: ..."

A Navcam candidate comes from `mppp.calib.fit.fit_navcam` + `write_navcam` (gated by `check_navcam`), a Mastcam-Z
one from `mppp.calib.fit.fit_zcam` (gated by `check_zcam`); the study scripts that run them are in
`studies/calibration/`.  Projects built from other models are rebuilt from these on their next notebook 03 run;
processed images, features and matches are reused.
