# Windows batch files

The folder `scripts\windows\` holds 24 batch files for running RPP on a Windows PC by double-click. Each one finds the
repository two levels up. They are wrappers: everything they do is an `rpp` command or a Python module that also
runs on Linux and macOS. A new user needs none of them; start with [Quick start](quickstart.md).

| What for | Files |
|---|---|
| **Setup** | `rpp_env.bat` (finds the Python that has rpp, pycolmap and pyceres; reads `rpp_python.txt`, `rpp_scapes.txt`, `rpp_pds.txt`, `rpp_cmods.txt`), `install_lightglue.bat` (the CUDA environment for matching), `sync_code.bat` (brings this tree up to date from a git bundle or from GitHub, runs the quick tests, updates the freeze checkout unless a freeze runs) |
| **One block** | `process_sites.bat` (process a group's images), `align_here.bat` (align the folder it sits in), `ecam_block.bat` (one Navcam and Hazcam block, then the Metashape and Nerfstudio exports) |
| **Many blocks** | `run_sites.bat` (the Navcam blocks of a group), `run_sites_zcam.bat` (the Navcam + Mastcam-Z blocks, then `zcam_joints.bat`), `check_sites.bat` (the site definitions, and when each block last finished or that it runs now), `stop_rpp.bat` (stops the runs) |
| **Station products** | `rpv_process.bat` (the station products of every aligned block for the viewer, Rockytop first, then the latest sol back; `--recolor` for the colors alone, `--unpublish LABEL` to take a station off the page; see below), `run_single_site_ibr.bat` (`rpp single-sites` for a few stations of `single_sites.json`) |
| **Calibration study** | `run_pipeline.bat` (the whole study: blocks, joints, checks), `zcam_joints.bat` (the Mastcam-Z joints), `promote_models.bat`, `freeze.bat` (a freeze run: every block aligned again with one version of the code and the models, unattended and resumable) |
| **Mask training** | `train_masks.bat` |
| **Figures and paper** | `make_figures.bat` (the standard figures), `make_maps.bat` (the site maps), `make_paper.bat` (the paper as a PDF) |
| Helpers (not double-clicked) | `_run_sites.bat`, `_run_process.bat`, `_run_align.bat` (the minimized batch windows), `m20_dem_env.bat` (the terrain-model and orthomosaic paths of the maps) |

Batch files of finished studies and of the rename from MPPP (0.99) were removed in 1.0. The git tag
`pre-v1.0-archive` has them.

The package wrappers are what a user of `rpp` runs, the study wrappers are the named experiments of the paper. Every wrapper takes its folders from the environment — `RPP_SCAPES` (the block
folders), `RPP_PDS` (the PDS products), optionally `RPP_GPU_PYTHON` (else `%LOCALAPPDATA%\mppp\gpu_python.txt`, written by `install_lightglue.bat`) and `COLMAP_BAT` — set once with
`setx RPP_SCAPES D:\rover_blocks` (new windows see it), or written one per line into `scripts\windows\rpp_scapes.txt`
and `rpp_pds.txt` (git-ignored). `run_pipeline.bat` falls back to `D:\rover_blocks` (the blocks root since 2 Nov 2026; the camera models stay in `D:\scapes\cmods`).

Since 0.94.3 the site batches (`run_sites*.bat`, `run_pipeline.bat`) align **two sites at a time** (`--jobs 1` for one,
or set `RPP_JOBS`). Each site's console output is then in `<WORK>\runs\console_<time>.txt`; the GPU steps (SIFT
extraction and matching, LightGlue) take turns through `%LOCALAPPDATA%\mppp\gpu.lock` (`RPP_GPU_LOCK=0` turns that off).

## The viewer's stations: `rpv_process.bat`

`scripts\windows\rpv_process.bat` makes the station products of every aligned block of a root
(`RPP_RPV_ROOT`, default `D:\rover_blocks_f3`) and puts them on the viewer: the Rockytop stations first, then the
others from the latest sol back. A station already made by RPP 0.99 or later with the same station code is skipped.
It waits for blocks that are not aligned yet, and stops when nothing is left. Several windows can run at once
(up to four, for example one run and one `--recolor`): each takes the first free checkout of `D:\code\RPP_rpv`,
`RPP_rpv_b`, `RPP_rpv_c` or `RPP_rpv_d`, brought to this tree's branch before its run, and the windows share the
stations through their locks. Stop station runs of 0.99.52 or older first: they take no locks.

```bat
D:\code\RPP\scripts\windows\rpv_process.bat
D:\code\RPP\scripts\windows\rpv_process.bat --recolor
D:\code\RPP\scripts\windows\rpv_process.bat --dry-run
D:\code\RPP\scripts\windows\rpv_process.bat --unpublish Sol0362_S011D0108
```

`--recolor` does no meshing, projection or rendering: it remakes the colors of every made station from its 16-bit
panoramas and textures, and uploads them. The log is `<root>\single_sites\rpv_process_log_<checkout>.txt`.

## The station products (`rpp mesh`, `depthmaps`, `ods`)

They need open3d in the Python that runs rpp (the base Python): `python -m pip install "open3d>=0.20"` (0.20 is the
first release with a Windows wheel for Python 3.13). The dense step before them runs its patch-match in the CUDA
python (`rpp_gpu`) by itself.

`scripts\windows\run_single_site_ibr.bat [STATION ...]` updates `D:\code\RPP` from the newest `rpp_*.bundle` in
`D:\rover_blocks\_transfer` and runs `rpp single-sites` (dense, mesh, depth maps, panorama; both variants) into
`D:\rover_blocks\single_sites`. Steps that are up to date are skipped; `single_sites_status.json` there says what is
running. `set RPP_NO_PAUSE=1` first for an unattended run. `--publish DIR --publish-only` puts the
stereo JPEGs and the viewer page into `DIR` (see [Station products](station-products.md)).

## Site maps: `scripts\windows\make_maps.bat`

Double-click it (or `make_maps.bat dtm 300`, `make_maps.bat ortho`, `make_maps.bat both 450`): the six M20 site-map
figures of `rpp_studies.sites` (`m20_sites`) from the official blocks of `src\rpp\data\sites.json`, into
`Claude outputs\maps_<yyyymmdd>\m20_<figure>_<dtm|ortho>.png`.  Backgrounds: the 1 m HiRISE DTMs and the
orthomosaic under `D:\rover_blocks\dem` (or `M20_DEMS` / `M20_ORTHOS`), the ~70 m terrain tiles elsewhere.  Needs
`pip install -e ".[maps]"` (adjustText, rasterio, pyproj).  Each figure is its own process; at 300 dpi with the
DTMs a figure takes a few minutes and 1-3 GB.
