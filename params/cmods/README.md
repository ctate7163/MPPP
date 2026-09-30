# MPPP camera models in use (`params/cmods`)

Notebook 03 starts every alignment from the models in this folder: the Navcam cameras and stereo rig
(`NAVCAM_CAMERAS`, default this folder) and the Mastcam-Z focus model (`ZCAM_FOCUS_MODEL = None` means the file
here). `MPPP_CMODS` points MPPP to another folder. If a file is missing here, MPPP falls back to the copy shipped
in `src/mppp/data` (`navcam_consensus/`, `m20_cmods/`).

| file | what |
|---|---|
| `M2020_NL_fisheye_tangential.json`, `M2020_NR_fisheye_tangential.json` | Navcam cameras (fisheye + tangential, full frame 5120 x 3840) at the reference temperature, with the thermal terms (`thermal`: f ppm/degC, NL cx px/degC) |
| `M2020_N_rig.json` | Navcam stereo rig (right from left), with the mission drift |
| `M2020_ZCAM034_focus_model.json` | Mastcam-Z 34 mm: focal length against focus count per eye, principal-point shift with focus |

**Change them only with `scripts/promote_cmods.py`** (checks the files, keeps the replaced ones in `history/`,
logs the change in `CHANGES.md`):

    python scripts\promote_cmods.py <folder or files> --note "why these are better"
    python scripts\promote_cmods.py --list

Projects built from other models are rebuilt from these on their next notebook 03 run (their fingerprints are part
of the project check); processed images, features and matches are reused.
