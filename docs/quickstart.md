# Quick start

This page runs MPPP on the example data set: Rockytop, sols 461-464 (Mars 2020 rover site 26). The set holds

- the main Navcam images at the two end-of-drive waypoints 26/630 (sol 461) and 26/756 (sol 464): the Navcam 360
  sequences `NCAM03461` and `NCAM03464` and their companion sequences `NCAM02xxx` and `NCAM13xxx`;
- one front Hazcam pair at each waypoint (`FHAZ02008`);
- eight Mastcam-Z 34 mm frames, four stereo pairs of sequence `ZCAM03381` (sol 464).

The files are laid out as a local PDS archive (`datadrive/<sol>/ids/rdr/ncam|fcam/` and `zcam/`) and published as
the GitHub release asset `example-data-v1`. The site `rockytop_example` in the site definitions selects them.

## 1. Install

```bash
pip install -e ".[mask,sfm,plots]"
```

See [Installation](installation.md) for the dependency groups and the GPU setup.

## 2. Fetch the example data

```bash
mppp example fetch                 # downloads the release asset, checks its SHA-256, unpacks to ~/mppp_example
```

`--dest DIR` unpacks somewhere else; `--force` downloads again. The zip contains `example_manifest.json` with the size
and SHA-256 of every file.

## 3. Point MPPP at the data and an output folder

```bash
export MPPP_PDS=~/mppp_example     # searched recursively: datadrive/ (Navcam, Hazcam) and zcam/ (Mastcam-Z)
export MPPP_SCAPES=~/mppp_scapes   # the WORK folders are created here
```

On Windows: `setx MPPP_PDS %USERPROFILE%\mppp_example` and `setx MPPP_SCAPES D:\scapes`, then open a new window.

`MPPP_PDS` set to `~/mppp_example/datadrive` (as `mppp example fetch` printed before 0.92) finds no Mastcam-Z frames; the Navcam-only run works
with it, but the Mastcam-Z frames in `~/mppp_example/zcam` are then outside the searched folder.

## 4. Run the block

```bash
mppp run rockytop_example --zcam
```

`run` selects the PDS products of the site's sols, processes them into `<WORK>/processed/`, and aligns them. With
`--zcam` the Mastcam-Z 34 mm frames of the same sols are added, and the WORK folder is
`$MPPP_SCAPES/mars2020_sol_0461_rockytop_example_zcam`. Without `--zcam` it is
`mars2020_sol_0461_rockytop_example` and only Navcam is aligned.

The two halves can also be run separately:

```bash
mppp process rockytop_example --zcam --dry-run     # list what would be processed
mppp process rockytop_example --zcam
mppp align   $MPPP_SCAPES/mars2020_sol_0461_rockytop_example_zcam
```

Each run writes `<WORK>/runs/<time>/log.txt` and `<WORK>/mppp_status.json`; `mppp align <WORK> --status` prints the
status. Without a GPU environment, SIFT extraction and matching run on the CPU.

## 5. Look at the results

In the WORK folder:

| path | content |
|---|---|
| `processed/images_png16/<PDS stem>.png` | 16-bit linear images, the mask in the alpha channel |
| `processed/references.txt` | positions and angles for Metashape *Import Reference* |
| `processed/colmap/sparse_prior/` | COLMAP cameras, prior poses and rig |
| `processed/mppp_manifest_<version>.json` | the per-image record (provenance) |
| `colmap/sparse/cahv_ba/` | the refined COLMAP block |
| `colmap/health/` | the alignment health report (`health.json`, `health.md`, `health.png`) |
| `colmap/error_input/` | exports for the error analysis (`summary.json`, `poses.csv`, `residuals.npz`, ...) |
| `colmap/run_done.json` | the run key, verdict and image counts of the finished run |

`mppp align <WORK> --gui` writes `colmap_gui.ini` and `open_in_colmap.bat` to open the project in the COLMAP GUI.

## The same from Python

```python
from pathlib import Path
from mppp.pipeline import Settings, run_site

home = Path.home()
s = Settings(pds_dir=home / "mppp_example", scapes_root=home / "mppp_scapes", include_zcam=True)
report = run_site("rockytop_example", s)
print(report["verdict"])
```

The notebooks in `notebooks/` (`01_process_images`, `02_colmap_alignment`, `03_error_analysis`) call the same
functions and show the results.
