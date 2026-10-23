# Changes to the camera models in use

## 20260930 - start of params/cmods (MPPP 0.43.3)
  - `M2020_NL_fisheye_tangential.json` 15389f18ab97, `M2020_NR_fisheye_tangential.json` 8c114d80b388,
    `M2020_N_rig.json` 949b9c26aca7: the v0p41 Navcam joint of 23 blocks (fisheye + tangential at -20 degC,
    f +38.1 ppm/degC, NL cx +0.0517 px/degC, rig with the mission drift only), from
    `D:\scapes\colmap\camera_analysis\navcal_v0p41\navcam_joint`
  - `M2020_ZCAM034_focus_model.json` f0586ba95ddd: the v0p42 Mastcam-Z focus model (3 blocks; temperature slope 0;
    sol trend provisional)

## 20260930 - moved to src/mppp/data/cmods (MPPP 0.50.0)
  - `params/` retired: the models in use, the flight calibrations (`m20_cmods/`) and the shipped consensus copy
    (`navcam_consensus/`, identical to the models in use) are one folder. Files unchanged; the v0p22 consensus rig
    that `m20_cmods/M2020_N_rig.json` held is in `history/v0p22_consensus/`.

## 20261001-093421 - v0p52 Navcam joint 'yawc_k4' (17 blocks): one constant rig yaw +35.21 mdeg (no drift / temperature yaw), k4 = 0; cost +0.25 % vs the v0p41 form on the same blocks
  - `M2020_NL_fisheye_tangential.json` da1b4f7dc3ad from `D:\scapes\colmap\camera_analysis\navcal_v0p52\navcam_joint_yawc_k4\M2020_NL_fisheye_tangential.json`
  - `M2020_NR_fisheye_tangential.json` 85167b0eb312 from `D:\scapes\colmap\camera_analysis\navcal_v0p52\navcam_joint_yawc_k4\M2020_NR_fisheye_tangential.json`
  - `M2020_N_rig.json` 4bef659a839e from `D:\scapes\colmap\camera_analysis\navcal_v0p52\navcam_joint_yawc_k4\M2020_N_rig.json`
  - replaced files (the v0p41 joint of 23 blocks) kept in `history/v0p41_joint/`

## 20261001-121403 - v0p60 Navcam consensus: 18 blocks (16 of the v0p52 joint + Bell Island and Taylorfjellet from the 0.53 runs), k4 = 0, one rig yaw +34.94 mdeg; cy +0.14 px, fy +0.07-0.10 px, yaw -0.27 mdeg vs v0p52
  - `M2020_NL_fisheye_tangential.json` 03d1bc4d1b35 from `D:\scapes\colmap\camera_analysis\navcal_v0p60\navcam_joint\M2020_NL_fisheye_tangential.json`
  - `M2020_NR_fisheye_tangential.json` 1ce955db253b from `D:\scapes\colmap\camera_analysis\navcal_v0p60\navcam_joint\M2020_NR_fisheye_tangential.json`
  - `M2020_N_rig.json` eefda5eb2fac from `D:\scapes\colmap\camera_analysis\navcal_v0p60\navcam_joint\M2020_N_rig.json`
  - replaced files (the v0p52 joint yawc_k4, 17 blocks) kept in `history/v0p52_joint/`

## 20261003-120509 - v0p73: the v0p72 Navcam consensus of 2 Oct 2026 (navcal_v0p72_20261002_2, 16 blocks; f +37.27 ppm/degC, stereo cx +0.0529 px/degC refitted, early-mission fx +0.15, fy +0.26, cy -0.39 px before sol 380, rig yaw +36.47 mdeg, baseline 0.42343 m fitted on 6 large blocks) in the NL-reference form: the stereo temperature change moved from NL cx to NR cx (-0.05280 px/degC), fx and fy each their own ppm/degC (both 37.27 until the v0p73 consensus fits them), k4 = 0. Replaces the v0p60 consensus; the hand promotions of the notebook-04 section 2e joint on 2 Oct 18:43-19:07 (k4 free, no stereo temperature term) were never committed
  - `M2020_NL_fisheye_tangential.json` d0e76fa53f42 from `D:\scapes\colmap\camera_analysis\navcal_v0p72_20261002_2\navcam_joint\M2020_NL_fisheye_tangential.json` (v0p73 NL-reference form)
  - `M2020_NR_fisheye_tangential.json` b15788a0fec7 from `D:\scapes\colmap\camera_analysis\navcal_v0p72_20261002_2\navcam_joint\M2020_NR_fisheye_tangential.json` (v0p73 NL-reference form)
  - `M2020_N_rig.json` 542d3156ca7e from `D:\scapes\colmap\camera_analysis\navcal_v0p72_20261002_2\navcam_joint\M2020_N_rig.json` (v0p73 NL-reference form)
  - replaced files (the v0p60 consensus, 18 blocks) kept in `history/v0p60_consensus/`

## 20261004-124857 - v0p74 pipeline consensus of 16 blocks (v0p73 form)
  - `M2020_N_rig.json` 84500425009a from `D:\scapes\colmap\camera_analysis\navcal_v0p74_20261004_pipeline\navcam_joint\M2020_N_rig.json`
  - `M2020_NL_fisheye_tangential.json` 6cc50f0b2854 from `D:\scapes\colmap\camera_analysis\navcal_v0p74_20261004_pipeline\navcam_joint\M2020_NL_fisheye_tangential.json`
  - `M2020_NR_fisheye_tangential.json` ae0d5aa195c1 from `D:\scapes\colmap\camera_analysis\navcal_v0p74_20261004_pipeline\navcam_joint\M2020_NR_fisheye_tangential.json`
  - replaced files kept in `history/20261004-124857/`

## 20261004-125252 - v0p74 Mastcam-Z 34 mm start model from 10 blocks (start-update)
  - `M2020_ZCAM034_focus_model.json` cd1b45b942a4 from `D:\scapes\colmap\camera_analysis\zcam34_start_20261004\M2020_ZCAM034_focus_model.json`
  - replaced files kept in `history/20261004-125252/`

## 20261005-222622 - v0p74 Mastcam-Z 34 mm calibration from 3 blocks (made on the PC; replaced on 8 Oct)
  - `M2020_ZCAM034_focus_model.json` 589aea1eae5e from `D:\scapes\colmap\camera_analysis\zcam34_calibration_20261005\M2020_ZCAM034_focus_model.json` (Rockytop, Marble Mountain, Belva Crater Large N+Z)
  - promoted on the PC (commit e8307c9, merged into this history on 17 Oct); replaced on 8 Oct by the v0p75 start model below - the file is kept in `history/20261005-222622_calibration/`

## 20261008-175008 - v0p75 start model: one df/dfocus per eye, ZL pp0 fixed, ZR pp0 pinned (cy slope), k1/k2 constant, rig mean of 3 first-run blocks, baseline CAHV (8 Oct 2026, from the archived 4-5 Oct first runs)
  - `M2020_ZCAM034_focus_model.json` 399f9eee93a9 from `/home/claude/v0p75_start/M2020_ZCAM034_focus_model.json`
  - replaced files kept in `history/20261008-175008/`

## 20261017-092358 - v0p80 best: residual rig yaw(T) of the 4 Oct rig residuals (D:\scapes\colmap\camera_analysis\navcal_v0p74_20261004_pipeline\rig_residuals.json, -0.339 +- 0.134 mdeg/degC) moved into NR cx(T): -0.0540 -> -0.0714 px/degC
  - `M2020_NL_fisheye_tangential.json` 20844b7af608 from `/home/claude/navcal_v0p80_best_20261017/navcam_joint/M2020_NL_fisheye_tangential.json`
  - `M2020_NR_fisheye_tangential.json` 47ae2675446c from `/home/claude/navcal_v0p80_best_20261017/navcam_joint/M2020_NR_fisheye_tangential.json`
  - `M2020_N_rig.json` 2bed6f3c54cc from `/home/claude/navcal_v0p80_best_20261017/navcam_joint/M2020_N_rig.json`
  - replaced files kept in `history/20261017-092358/`

## 20261019-070226 - v0p85: the 18 Oct pass-2 consensus of 23 v0p82 blocks (v0p80 form: k3+k4, k1(T) NL, NR cx -0.0525 px/degC, yaw rate +2.56 mdeg/1000 sol, early offsets fx +0.197 fy +0.254 cx +0.324 cy -0.572 px; leave-one-out 0.011 / 0.027 px; not promoted by the pipeline only because of the covariance bug fixed in 0.82.1; rig residuals: no yaw temperature term left)
  - `M2020_NL_fisheye_tangential.json` b59e2267b62e from `/mnt/user-data/uploads/colmap/camera_analysis/navcal_v0p82_20261018_pipeline_pass2/navcam_joint/M2020_NL_fisheye_tangential.json`
  - `M2020_NR_fisheye_tangential.json` 7d4a299dd60a from `/mnt/user-data/uploads/colmap/camera_analysis/navcal_v0p82_20261018_pipeline_pass2/navcam_joint/M2020_NR_fisheye_tangential.json`
  - `M2020_N_rig.json` 49198e58ec14 from `/mnt/user-data/uploads/colmap/camera_analysis/navcal_v0p82_20261018_pipeline_pass2/navcam_joint/M2020_N_rig.json`
  - replaced files kept in `history/20261019-070226/`

- 2026-10-19 (v0p85): M2020_N_rig.json of the promotion above edited by hand: "translation": "fitted" restored (the pass-2 writer dropped it because the new large-block fit was recorded, not applied; t and the baseline 0.42401 m are unchanged) - sha 4aa6455701fc

## 20261019-155805 - v0p87: Mastcam-Z 34 mm calibration from South Arm alone (19 Oct ZR-fix diagnostic run; Christian: better than the v0p75 start in the next run). One block, so the 'blocks' check (>= 2) is waived; Rochette left out (approach-station poses), Marble Mountain failed (fixed in v0p86)
  - `M2020_ZCAM034_focus_model.json` a043c8e52953 from `D:\scapes\mars2020_sol_1408_south_arm_zcam` (zcam_calib.calibrate on that block alone, form v0p75, regular state derived; report in docs/results/zcam34_calibration_20261019_south_arm/)
  - replaced files kept in `history/20261019-155805/`
## 0.90.0 - the forms frozen (mppp.calib)
  - one form per family: Navcam / Hazcam THIN_PRISM_FISHEYE with sx1 = sy1 = 0, thermal terms on fx, fy, cx, cy, k1,
    early-mission offsets, the rig drift in sol (`"form": "navcam-v1"`); Mastcam-Z per zoom, focus lines per
    backlash state, p1 = p2 = k3 = 0, no temperature term (`"form": "zcam-v1"`)
  - `M2020_NL_fisheye_tangential.json`, `M2020_NR_fisheye_tangential.json`: the shared `ppm_per_degC` and the
    `v0p80` bookkeeping keys dropped from `thermal` (the per-axis terms are the model), `form`, `family`, `eye` added;
    parameters unchanged (the 18 Oct 2026 consensus)
  - `M2020_ZCAM034_focus_model.json`: the superseded `trend_v0p42` entries and the temperature terms dropped;
    parameters unchanged
  - removed: the Metashape flight calibrations `*_frame.xml`, the rational Navcam cameras `M2020_N?_rational.json`,
    the `history/` folder (the git history keeps replaced models)
  - Hazcam start models are made by `mppp.calib.hazcam_start_model` until a calibrated file is promoted

## 20261020-065411 - v0p86 pipeline consensus pass2 of 16 blocks (v0p80 form)
  - `M2020_N_rig.json` 3b2f5b0638f9 from `D:\scapes\camera_analysis\navcal_v0p86_20261020_pipeline_pass2\navcam_joint\M2020_N_rig.json`
  - `M2020_NL_fisheye_tangential.json` c54ee6cb4d5a from `D:\scapes\camera_analysis\navcal_v0p86_20261020_pipeline_pass2\navcam_joint\M2020_NL_fisheye_tangential.json`
  - `M2020_NR_fisheye_tangential.json` b6ec31a08d2d from `D:\scapes\camera_analysis\navcal_v0p86_20261020_pipeline_pass2\navcam_joint\M2020_NR_fisheye_tangential.json`
  - replaced files kept in `history/20261020-065411/`
  - 0.90: the three files converted to the frozen form on adoption (`form`, `family`, `eye`; the shared
    `ppm_per_degC` and v0p80 bookkeeping dropped from `thermal`); parameters unchanged; no history folder (git)

## 20261020-154310 - Navcam tied joint navcal_tied_20261020_2 (0.91, 22 blocks, 4358 images, knot 250, yaw held; fx/fy slopes fitted separately - agree within 1 sd; early-mission offsets of both eyes applied)
  - `M2020_NL_fisheye_tangential.json` 3a929587a1cc from `D:\scapes\camera_analysis\navcal_tied_20261020_2\navcam_joint\M2020_NL_fisheye_tangential.json` (rewritten with applied_terms) (THIN_PRISM_FISHEYE 5120x3840, fx 2956.17 at -20.0 degC)
  - `M2020_NR_fisheye_tangential.json` eb9633c261ff from `D:\scapes\camera_analysis\navcal_tied_20261020_2\navcam_joint\M2020_NR_fisheye_tangential.json` (rewritten with applied_terms) (THIN_PRISM_FISHEYE 5120x3840, fx 2948.69 at -20.0 degC)
  - `M2020_N_rig.json` cf7cd2204615 from `D:\scapes\camera_analysis\navcal_tied_20261020_2\navcam_joint\M2020_N_rig.json` (rewritten with applied_terms) (rig, baseline 0.4240102397823094 m)

## 20261022-110217 - Mastcam-Z 34 mm joint of 6 blocks
  - `M2020_ZCAM034_focus_model.json` e82c2e0da6d6 from `D:\scapes\camera_analysis\zcam34_joint_20261022\M2020_ZCAM034_focus_model.json` (focus model for ['ZL034', 'ZR034'] and rig)

## 20261023-085533 - Hazcam fit of 6 blocks (belva_crater, chal_rocks, marble_mountain, seitah_overlook, south_arm, threeforks_north)
  - `M2020_FR.json` b2d331b50421 from `D:\scapes\camera_analysis\hazcam_fit_zcam_20261023\candidate\M2020_FR.json` (THIN_PRISM_FISHEYE 5120x3840, fx 2139.08 at -20.0 degC)

## 20261023-100444 - Mastcam-Z 34 mm joint of 8 blocks
  - `M2020_ZCAM034_focus_model.json` f732cdadf193 from `D:\scapes\camera_analysis\zcam34_joint_20261023\M2020_ZCAM034_focus_model.json` (focus model for ['ZL034', 'ZR034'] and rig)

## 20261023-110006 - Mastcam-Z 63 mm joint of 2 blocks (zcam63_joint_20261023, chal_rocks and marble_mountain; a first calibration): every check passed except the low-state ratio, which the joint wrote as 1.0000 for want of low-state bins - set to the default 1.0095 (0.94.0)
  - `M2020_ZCAM063_focus_model.json` 80abe5bb1562 from `/tmp/claude-0/-home-claude/3f5549f9-2d32-5ba5-952a-af5a6dabd550/scratchpad/z63/M2020_ZCAM063_focus_model.json` (focus model for ['ZL063', 'ZR063'] and rig)

## 20261023-110047 - Hazcam FL athermal law (0.94.0 fallback; Christian 23 Oct): the temperature law of hazcam_fit_zcam_20261023 failed its thermal check (+150.8 ppm/degC); fx, fy, k1 constant over the same 4 blocks, block scatter 538 ppm, every check passed
  - `M2020_FL.json` e449940994af from `/tmp/claude-0/-home-claude/3f5549f9-2d32-5ba5-952a-af5a6dabd550/scratchpad/fl/M2020_FL.json` (THIN_PRISM_FISHEYE 5120x3840, fx 2141.07 at -20.0 degC)

## 20261023-113000 - Mastcam-Z 110 mm joint of 2 blocks (zcam110_joint_20261023; a first calibration, single state): it failed only the form (no low-state lines - not needed at 110 mm since 0.94.0) and the aspect at the 1 % limit (fy / fx 0.982); every check passes with the 2 % limit of 0.94.1 (Christian)
  - `M2020_ZCAM110_focus_model.json` cf9de96f5f94 from `/mnt/user-data/uploads/camera_analysis/zcam110_joint_20261023/M2020_ZCAM110_focus_model.json` (focus model for ['ZL110', 'ZR110'] and rig)

## 20261023-142018 - Hazcam fit of 2 blocks (chal_rocks_tiny, seitah_overlook_tiny)
  - `M2020_RL.json` 65c4a274490e from `D:\scapes\camera_analysis\hazcam_fit_tiny_20261023\candidate\M2020_RL.json` (THIN_PRISM_FISHEYE 5120x3840, fx 2156.86 at -20.0 degC)
  - `M2020_RR.json` 5b5342f5a8b0 from `D:\scapes\camera_analysis\hazcam_fit_tiny_20261023\candidate\M2020_RR.json` (THIN_PRISM_FISHEYE 5120x3840, fx 2148.31 at -20.0 degC)
