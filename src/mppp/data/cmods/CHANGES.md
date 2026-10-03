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
