# The Windows batch files (0.90)

Two folders, by what they wrap:

| folder | what | files |
|---|---|---|
| `scripts\windows\` | the package: environment, the `mppp` commands over a site group, status, stop, sync | `mppp_env.bat` (finds the Python with mppp, pycolmap, pyceres; reads `mppp_python.txt`, `mppp_scapes.txt`, `mppp_pds.txt`), `process_sites.bat`, `run_sites.bat`, `run_sites_zcam.bat`, `run_sites_zcam34.bat`, `align_here.bat`, `sites_status.bat`, `stop_mppp.bat`, `check_sites.bat`, `claude_sync.bat`, `render_paper.bat` |
| `studies\calibration\windows\` | the calibration study: the whole pipeline and the named runs | `run_pipeline.bat`, `run_v0p87.bat`, `run_z34.bat`, `zr_diagnose_z34.bat`, `install_lightglue.bat`, `move_scapes.bat`; `stop_mppp.bat` and `sites_status.bat` here forward to the package ones |

Nothing is deprecated: the package wrappers are what a user of `mppp` runs, the study wrappers are the named
experiments of the paper. Every wrapper takes its folders from the environment — `MPPP_SCAPES` (the block
folders), `MPPP_PDS` (the PDS products), optionally `MPPP_GPU_PYTHON` and `COLMAP_BAT` — set once with
`setx MPPP_SCAPES D:\scapes` (new windows see it), or written one per line into `scripts\windows\mppp_scapes.txt`
and `mppp_pds.txt` (git-ignored). `run_pipeline.bat` and `run_z34.bat` fall back to `D:\scapes`.
