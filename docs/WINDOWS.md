# The Windows batch files (0.96.0: 25 files)

One folder, `scripts\windows\`, every double-click file in one place; it finds the repository two levels up.
0.94.4 (Christian, 23 Oct): trimmed to the files in use - the one-off runs, the single-zoom and single-camera
batches and the weekend runner went (git history has them; `run_pipeline.bat` does what they did).

| what for | files |
|---|---|
| **the freeze** | `freeze.bat` (0.96.0: the freeze run, unattended and resumable, on its own checkout `D:\code\MPPP_freeze` cloned from this tree - `mppp_studies.calibration.freeze`: wait, mask, cmods, zjoints, zfinal, ecam, nz, deck, ztests, studies, report; `--status`, `--mask-model keep\|auto\|FILE`, `--update-code`, `--redo STEP`, `--only STEP`), `stop_mppp.bat` (also stops it: "stopped by the user") |
| **the runs** | `run_pipeline.bat` (the whole calibration study: the ecam_consensus blocks - Navcam with both Hazcam pairs; the Navcam joint and the Hazcam fit, promoted; the Mastcam-Z blocks; the Mastcam-Z joints; the error analysis; the rover-frame study 3A), `run_sites.bat` (the Navcam blocks of a group), `run_sites_zcam.bat` (the Navcam + Mastcam-Z blocks of zcam_consensus, then `zcam_joints.bat`), `zcam_joints.bat`, `zcam_backlash_rerun.bat` (0.95.3: the backlash states run of Chal Rocks, Seitah Overlook and Rockytop, the found list updated, the three blocks again with every zoom, then the 34-110 mm joints with `--iterations 300`, nothing promoted), `run_zcam_states.bat`, `promote_models.bat`, `stop_mppp.bat` |
| **status** | `check_sites.bat` (the site definitions; when each block last finished, with which version, or that it runs now) |
| **one folder** | `process_sites.bat` (process a group's images), `align_here.bat` (align the folder it sits in) |
| **figures** | `make_figures.bat` (the standard figures at 200 dpi), `make_maps.bat` (the site maps), `make_paper.bat` (the JOSS paper as a PDF); **status**: `sites_status.bat` (the run status of every block) |
| **setup** | `mppp_env.bat` (finds the Python with mppp, pycolmap, pyceres; reads `mppp_python.txt`, `mppp_scapes.txt`, `mppp_pds.txt`; checks it with `scripts\check_env.py`), `install_lightglue.bat`, `sync_code.bat` (0.96.0, replaces `claude_sync.bat`: this tree's `main` fast-forwarded from a delivered git bundle or from GitHub, the quick tests one file per process with MKL sequential, the freeze checkout brought up unless a freeze runs on it; nothing started, nothing pushed), `update_and_run_single_sites.bat` (0.96.1: the tree fast-forwarded to the newest bundle in `D:\rover_blocks\_transfer` when that is ahead, then `mppp single-sites` - the test stations' dense cloud, mesh, depth maps and 16-bit stereo panoramas; `--steps mesh ods --variants rover` for the panorama alone) |
| helpers (not double-clicked) | `_run_sites.bat`, `_run_process.bat`, `_run_align.bat` (the minimised batch windows), `m20_dem_env.bat` (the DTM / orthomosaic paths of the figures and maps) |

Removed in 0.94.4: `run_weekend.bat` (`python -m mppp_studies.misc.calibration_old.run_weekend` still works), `test_lightglue.bat`
(`python -m mppp_studies.calibration.lightglue_ab`), `run_hazcam.bat` and `hazcam_calibration.bat` (the Hazcam fit is in
`run_pipeline.bat`), `navcam_tied_joint.bat` and `zcam_tied_joint.bat` (`mppp calib navcam`, `mppp calib zcam`;
in `run_pipeline.bat`), `run_sites_zcam34.bat`, `run_z34.bat`, `run_multizoom_test.bat`, `run_v0p87.bat`,
`move_scapes.bat`, `build_example_data.bat` (`python -m mppp example build --pds ... --zcam ... --out _transfer`), `sites_status.bat` (`check_sites.bat`).

The package wrappers are what a user of `mppp` runs, the study wrappers are the named experiments of the paper. Every wrapper takes its folders from the environment — `MPPP_SCAPES` (the block
folders), `MPPP_PDS` (the PDS products), optionally `MPPP_GPU_PYTHON` (else `%LOCALAPPDATA%\mppp\gpu_python.txt`, written by `install_lightglue.bat`) and `COLMAP_BAT` — set once with
`setx MPPP_SCAPES D:\rover_blocks` (new windows see it), or written one per line into `scripts\windows\mppp_scapes.txt`
and `mppp_pds.txt` (git-ignored). `run_pipeline.bat` falls back to `D:\rover_blocks` (the blocks root since 2 Nov 2026; the camera models stay in `D:\scapes\cmods`).

Since 0.94.3 the site batches (`run_sites*.bat`, `run_pipeline.bat`) align **two sites at a time** (`--jobs 1` for one,
or set `MPPP_JOBS`). Each site's console output is then in `<WORK>\runs\console_<time>.txt`; the GPU steps (SIFT
extraction and matching, LightGlue) take turns through `%LOCALAPPDATA%\mppp\gpu.lock` (`MPPP_GPU_LOCK=0` turns that off).

## The station products (`mppp mesh`, `depthmaps`, `ods`)

They need open3d in the Python that runs mppp (the base Python): `python -m pip install "open3d>=0.20"` (0.20 is the
first release with a Windows wheel for Python 3.13). The dense step before them runs its patch-match in the CUDA
python (`mppp_gpu`) by itself.

## Site maps: `scripts\windows\make_maps.bat`

Double-click it (or `make_maps.bat dtm 300`, `make_maps.bat ortho`, `make_maps.bat both 450`): the six M20 site-map
figures of `mppp_studies.sites` (`m20_sites`) from the official blocks of `src\mppp\data\sites.json`, into
`Claude outputs\maps_<yyyymmdd>\m20_<figure>_<dtm|ortho>.png`.  Backgrounds: the 1 m HiRISE DTMs and the
orthomosaic under `D:\rover_blocks\dem` (or `M20_DEMS` / `M20_ORTHOS`), the ~70 m terrain tiles elsewhere.  Needs
`pip install -e ".[maps]"` (adjustText, rasterio, pyproj).  Each figure is its own process; at 300 dpi with the
DTMs a figure takes a few minutes and 1-3 GB.
