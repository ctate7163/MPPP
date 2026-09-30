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
