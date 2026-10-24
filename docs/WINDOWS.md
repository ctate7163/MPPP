# The Windows batch files (0.94.4: 19 files)

One folder, `scripts\windows\`, every double-click file in one place; it finds the repository two levels up.
0.94.4 (Christian, 23 Oct): trimmed to the files in use - the one-off runs, the single-zoom and single-camera
batches and the weekend runner went (git history has them; `run_pipeline.bat` does what they did).

| what for | files |
|---|---|
| **the runs** | `run_pipeline.bat` (the whole calibration study: the ecam_consensus blocks - Navcam with both Hazcam pairs; the Navcam joint and the Hazcam fit, promoted; the Mastcam-Z blocks; the Mastcam-Z joints; the error analysis), `run_sites.bat` (the Navcam blocks of a group), `run_sites_zcam.bat` (the Navcam + Mastcam-Z blocks of zcam_consensus, then `zcam_joints.bat`), `zcam_joints.bat`, `promote_models.bat`, `stop_mppp.bat` |
| **status** | `check_sites.bat` (the site definitions; when each block last finished, with which version, or that it runs now) |
| **one folder** | `process_sites.bat` (process a group's images), `align_here.bat` (align the folder it sits in) |
| **figures** | `make_figures.bat` (the standard figures at 200 dpi), `make_maps.bat` (the site maps), `render_paper.bat` |
| **setup** | `mppp_env.bat` (finds the Python with mppp, pycolmap, pyceres; reads `mppp_python.txt`, `mppp_scapes.txt`, `mppp_pds.txt`; checks it with `scripts\check_env.py`), `install_lightglue.bat`, `claude_sync.bat` (uncommitted edits stashed before the merge and popped after it) |
| helpers (not double-clicked) | `_run_sites.bat`, `_run_process.bat`, `_run_align.bat` (the minimised batch windows), `m20_dem_env.bat` (the DTM / orthomosaic paths of the figures and maps) |

Removed in 0.94.4: `run_weekend.bat` (`python studies\calibration\run_weekend.py` still works), `test_lightglue.bat`
(`python scripts\lightglue_ab.py`), `run_hazcam.bat` and `hazcam_calibration.bat` (the Hazcam fit is in
`run_pipeline.bat`), `navcam_tied_joint.bat` and `zcam_tied_joint.bat` (`mppp calib navcam`, `mppp calib zcam`;
in `run_pipeline.bat`), `run_sites_zcam34.bat`, `run_z34.bat`, `run_multizoom_test.bat`, `run_v0p87.bat`,
`move_scapes.bat`, `build_example_data.bat` (`python -m mppp example build --pds ... --zcam ... --out _transfer`), `sites_status.bat` (`check_sites.bat`).

The package wrappers are what a user of `mppp` runs, the study wrappers are the named experiments of the paper. Every wrapper takes its folders from the environment — `MPPP_SCAPES` (the block
folders), `MPPP_PDS` (the PDS products), optionally `MPPP_GPU_PYTHON` (else `%LOCALAPPDATA%\mppp\gpu_python.txt`, written by `install_lightglue.bat`) and `COLMAP_BAT` — set once with
`setx MPPP_SCAPES D:\scapes` (new windows see it), or written one per line into `scripts\windows\mppp_scapes.txt`
and `mppp_pds.txt` (git-ignored). `run_pipeline.bat` falls back to `D:\scapes`.

Since 0.94.3 the site batches (`run_sites*.bat`, `run_pipeline.bat`) align **two sites at a time** (`--jobs 1` for one,
or set `MPPP_JOBS`). Each site's console output is then in `<WORK>\runs\console_<time>.txt`; the GPU steps (SIFT
extraction and matching, LightGlue) take turns through `%LOCALAPPDATA%\mppp\gpu.lock` (`MPPP_GPU_LOCK=0` turns that off).

## Site maps: `scripts\windows\make_maps.bat`

Double-click it (or `make_maps.bat dtm 300`, `make_maps.bat ortho`, `make_maps.bat both 450`): the six M20 site-map
figures of `studies\m20_sites\m20_sites` from the official blocks of `src\mppp\data\sites.json`, into
`Claude outputs\maps_<yyyymmdd>\m20_<figure>_<dtm|ortho>.png`.  Backgrounds: the 1 m HiRISE DTMs and the
orthomosaic under `D:\scapes\dem` (or `M20_DEMS` / `M20_ORTHOS`), the ~70 m terrain tiles elsewhere.  Needs
`pip install -e ".[maps]"` (adjustText, rasterio, pyproj).  Each figure is its own process; at 300 dpi with the
DTMs a figure takes a few minutes and 1-3 GB.
