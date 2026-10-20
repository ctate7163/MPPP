# The Windows batch files (0.91)

One folder, `scripts\windows\` (0.91: the calibration study's wrappers moved here from
`studies\calibration\windows`, so every double-click file is in one place and finds the repository two levels up):

| group | files |
|---|---|
| environment | `mppp_env.bat` (finds the Python with mppp, pycolmap, pyceres; reads `mppp_python.txt`, `mppp_scapes.txt`, `mppp_pds.txt`), `install_lightglue.bat`, `claude_sync.bat` (uncommitted edits stashed before the merge and popped after it), `render_paper.bat` (on demand) |
| the package over a site group | `process_sites.bat`, `run_sites.bat`, `run_sites_zcam.bat`, `run_sites_zcam34.bat`, `align_here.bat`, `sites_status.bat`, `stop_mppp.bat`, `check_sites.bat`, `make_maps.bat`, `make_figures.bat` (the standard figures at 200 dpi), `m20_dem_env.bat` (the DTM / orthomosaic paths for both) |
| the calibration study (`studies\calibration\*.py`) | `run_pipeline.bat` (the stage runner), the named runs `run_v0p91.bat`, `run_v0p87.bat`, `run_z34.bat`, `navcam_tied_joint.bat` (the one-adjustment Navcam joint on the finished blocks), `zcam_tied_joint.bat` (the Mastcam-Z joint: Navcam held at the models in use, the focus model fitted), `move_scapes.bat` |

The package wrappers are what a user of `mppp` runs, the study wrappers are the named experiments of the paper. Every wrapper takes its folders from the environment — `MPPP_SCAPES` (the block
folders), `MPPP_PDS` (the PDS products), optionally `MPPP_GPU_PYTHON` and `COLMAP_BAT` — set once with
`setx MPPP_SCAPES D:\scapes` (new windows see it), or written one per line into `scripts\windows\mppp_scapes.txt`
and `mppp_pds.txt` (git-ignored). `run_pipeline.bat` and `run_z34.bat` fall back to `D:\scapes`.

## Site maps: `scripts\windows\make_maps.bat`

Double-click it (or `make_maps.bat dtm 300`, `make_maps.bat ortho`, `make_maps.bat both 450`): the six M20 site-map
figures of `studies\m20_sites\m20_sites` from the official blocks of `src\mppp\data\sites.json`, into
`Claude outputs\maps_<yyyymmdd>\m20_<figure>_<dtm|ortho>.png`.  Backgrounds: the 1 m HiRISE DTMs and the
orthomosaic under `D:\scapes\dem` (or `M20_DEMS` / `M20_ORTHOS`), the ~70 m terrain tiles elsewhere.  Needs
`pip install -e ".[maps]"` (adjustText, rasterio, pyproj).  Each figure is its own process; at 300 dpi with the
DTMs a figure takes a few minutes and 1-3 GB.
