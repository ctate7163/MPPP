# Changes to the camera models in use

## 20260930 - start of params/cmods (MPPP 0.43.3)
  - `M2020_NL_fisheye_tangential.json` 15389f18ab97, `M2020_NR_fisheye_tangential.json` 8c114d80b388,
    `M2020_N_rig.json` 949b9c26aca7: the v0p41 Navcam joint of 23 blocks (fisheye + tangential at -20 degC,
    f +38.1 ppm/degC, NL cx +0.0517 px/degC, rig with the mission drift only), from
    `D:\scapes\colmap\camera_analysis\navcal_v0p41\navcam_joint`
  - `M2020_ZCAM034_focus_model.json` f0586ba95ddd: the v0p42 Mastcam-Z focus model (3 blocks; temperature slope 0;
    sol trend provisional)
