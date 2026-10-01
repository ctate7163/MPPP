# MPPP camera models (`src/mppp/data/cmods`)

The one folder of MPPP camera models (v0p50; before: `params/cmods`, `src/mppp/data/m20_cmods`,
`src/mppp/data/navcam_consensus`). `MPPP_CMODS` points MPPP to another folder for the models in use.

**In use** (notebook 03 starts every alignment from them: `NAVCAM_CAMERAS`, `ZCAM_FOCUS_MODEL = None`):

| file | what |
|---|---|
| `M2020_NL_fisheye_tangential.json`, `M2020_NR_fisheye_tangential.json` | Navcam cameras (fisheye + tangential, full frame 5120 x 3840) at the reference temperature, with the thermal terms (`thermal`: f ppm/degC, NL cx px/degC). The distortion is one set per eye for all sols and temperatures (v0p50: held in every block, `NAVCAM_DISTORTION_FIT = "hold"`); v0p52: k4 = 0 (`NAVCAM_K4 = "zero"`) |
| `M2020_N_rig.json` | Navcam stereo rig (right from left): v0p52 one constant yaw (+35.21 mdeg, `NAVCAM_RIG_YAW = "hold"`), pitch and roll with the mission drift |
| `M2020_ZCAM034_focus_model.json` | Mastcam-Z 34 mm: focal length against focus count per eye, principal-point shift with focus |

**Flight and earlier start cameras:** `M2020_*_frame.xml` (Metashape calibrations from the flight CAHV(OR)(E)
models, Navcam, Hazcam, Mastcam-Z 34/110 mm) and `M2020_NL_rational.json`, `M2020_NR_rational.json` (the v0p20
rational Navcam cameras, `NAVCAM_DISTORTION = "rational"`). `history/` keeps replaced models (e.g. the v0p22
consensus rig, the v0p41 joint).

**Change the models in use only with `scripts/promote_cmods.py`** (checks the files, keeps the replaced ones in
`history/`, logs the change in `CHANGES.md`), then commit:

    python scripts\promote_cmods.py <folder or files> --note "why these are better"
    python scripts\promote_cmods.py --list
    git add src/mppp/data/cmods && git commit -m "cmods: ..."

Projects built from other models are rebuilt from these on their next notebook 03 run (their fingerprints are part
of the project check); processed images, features and matches are reused.
