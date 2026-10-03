# MPPP camera models (`src/mppp/data/cmods`)

The one folder of MPPP camera models (v0p50; before: `params/cmods`, `src/mppp/data/m20_cmods`,
`src/mppp/data/navcam_consensus`). `MPPP_CMODS` points MPPP to another folder for the models in use.

**In use** (notebook 03 starts every alignment from them: `NAVCAM_CAMERAS`, `ZCAM_FOCUS_MODEL = None`):

| file | what |
|---|---|
| `M2020_NL_fisheye_tangential.json`, `M2020_NR_fisheye_tangential.json` | Navcam cameras (fisheye + tangential, full frame 5120 x 3840) at the reference temperature, with the thermal terms. **v0p73 form, "NL reference"** (`mppp.sfm.cmods_form`): NL keeps the label CAHVORE's pointing and position, everything that moves is NR's - fx and fy of each eye against temperature (`thermal.fx_ppm_per_degC`, `fy_ppm_per_degC`), the stereo temperature change on NR cx (`cx_px_per_degC` of NR; NL's is 0), k4 = 0 (`NAVCAM_K4 = "zero"`). The distortion is one set per eye for all sols and temperatures (held in every block, `NAVCAM_DISTORTION_FIT = "hold"`). In use: the v0p72 consensus of 2 Oct 2026 (16 blocks) in that form |
| `M2020_N_rig.json` | Navcam stereo rig (right from left): one yaw for every temperature (`yaw_constant`, `NAVCAM_RIG_YAW = "hold"`), pitch, roll and (v0p73) yaw with the mission drift (`drift.<angle>_mdeg_per_sol`; the yaw rate is 0 until a v0p73 consensus measures it), translation fitted on the large blocks |
| `M2020_ZCAM034_focus_model.json` | Mastcam-Z 34 mm: focal length against focus count per eye, principal-point shift with focus. v0p53: one file per zoom (`M2020_ZCAM048_...`, `_063`, `_110`); a file may also hold each eye's `distortion` and `pp0_px` and the zoom's `rig` (notebook 04 §5b), which notebook 03 then holds in every block. A zoom without a file starts from the labels of its block (`scripts/zcam_start_models.py` makes provisional files and XMLs from the labels) |

**Flight and earlier start cameras:** `M2020_*_frame.xml` (Metashape calibrations from the flight CAHV(OR)(E)
models, Navcam, Hazcam, Mastcam-Z 34/110 mm) and `M2020_NL_rational.json`, `M2020_NR_rational.json` (the v0p20
rational Navcam cameras, `NAVCAM_DISTORTION = "rational"`). `history/` keeps replaced models (e.g. the v0p22
consensus rig, the v0p41 and v0p52 joints). Make a new Navcam consensus with notebook 04 §2d (`STUDY_CMD = "consensus"`) or `scripts/navcam_calibration_study.py consensus`.

**Before a batch:** `scripts/prepare_next_run.py` (`prepare_next_run.bat`) checks that the models in use are the
expected consensus and follow the v0p73 form; `run_sites.py` and notebook 03 refuse to start otherwise. The models'
fingerprint is part of the run key (v0p73): a promotion makes `run_sites.py` align every site again.

**Change the models in use only with `scripts/promote_cmods.py`** (checks the files, keeps the replaced ones in
`history/`, logs the change in `CHANGES.md`), then commit:

    python scripts\promote_cmods.py <folder or files> --note "why these are better"
    python scripts\promote_cmods.py --list
    git add src/mppp/data/cmods && git commit -m "cmods: ..."

Projects built from other models are rebuilt from these on their next notebook 03 run (their fingerprints are part
of the project check); processed images, features and matches are reused.
