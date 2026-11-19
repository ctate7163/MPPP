# Quick start

This page runs RPP on the example data set: Rockytop, sols 461-464 (Mars 2020 rover site 26). The set holds

- the main Navcam images at the two end-of-drive waypoints 26/630 (sol 461) and 26/756 (sol 464): the Navcam 360
  sequences `NCAM03461` and `NCAM03464` and their companion sequences `NCAM02xxx` and `NCAM13xxx`;
- one front Hazcam pair at each waypoint (`FHAZ02008`);
- eight Mastcam-Z 34 mm frames, four stereo pairs of sequence `ZCAM03381` (sol 464).

The files are laid out as a local PDS archive (`datadrive/<sol>/ids/rdr/ncam|fcam/` and `zcam/`), as the PDS
Imaging Node has them; they are also published as the GitHub release asset `example-data-v1`. The site `rockytop_example` in the site definitions selects them.

## 1. Install

```bash
pip install -e ".[mask,sfm,plots]"
```

See [Installation](installation.md) for the dependency groups and the GPU setup.

## 2. Fetch the example data

```bash
rpp example fetch                 # the 36 products from the PDS Imaging Node into ~/rpp_example
```

`--dest DIR` puts them somewhere else. Files already there are not downloaded again. If the PDS cannot be reached,
the release zip (`example-data-v1`) is used instead; `--source release` asks for it directly.

Any other sols: `rpp pds fetch --sols FIRST LAST [--cameras ncam fcam rcam zcam] [--match TEXT] [--list]`.

## 3. Point RPP at the data and an output folder

Nothing to do for the example: `rockytop_example` reads `~/rpp_example` and writes to `~/rpp_scapes`. For your own
data, or other folders:

```bash
export RPP_PDS=~/rpp_example     # searched recursively: datadrive/ (Navcam, Hazcam) and zcam/ (Mastcam-Z)
export RPP_SCAPES=~/rpp_scapes   # the WORK folders are created here
```

On Windows: `setx RPP_PDS %USERPROFILE%\rpp_example` and `setx RPP_SCAPES D:\rover_blocks`, then open a new window.

`RPP_PDS` set to `~/rpp_example/datadrive` (as `rpp example fetch` printed before 0.92) finds no Mastcam-Z frames; the Navcam-only run works
with it, but the Mastcam-Z frames in `~/rpp_example/zcam` are then outside the searched folder.

## 4. Run the block

```bash
rpp run rockytop_example --zcam
```

`run` selects the PDS products of the site's sols, processes them into `<WORK>/processed/`, and aligns them. With
`--zcam` the Mastcam-Z 34 mm frames of the same sols are added, and the WORK folder is
`$RPP_SCAPES/mars2020_sol_0461_rockytop_example_zcam`. Without `--zcam` it is
`mars2020_sol_0461_rockytop_example` and only Navcam is aligned.

The two halves can also be run separately:

```bash
rpp process rockytop_example --zcam --dry-run     # list what would be processed
rpp process rockytop_example --zcam
rpp align   $RPP_SCAPES/mars2020_sol_0461_rockytop_example_zcam
```

Each run writes `<WORK>/runs/<time>/log.txt` and `<WORK>/mppp_status.json`; `rpp align <WORK> --status` prints the
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

`rpp align <WORK> --gui` writes `colmap_gui.ini` and `open_in_colmap.bat` to open the project in the COLMAP GUI.

## The same from Python

```python
from pathlib import Path
from rpp.pipeline import Settings, run_site

home = Path.home()
s = Settings(pds_dir=home / "rpp_example", scapes_root=home / "rpp_scapes", include_zcam=True)
report = run_site("rockytop_example", s)
print(report["verdict"])
```

`notebooks/pipeline_example.ipynb` calls the same functions on the example data and shows the results;
`notebooks/van_zyl_ods_example.ipynb` downloads one station from the PDS (16 frames of sol 52: the station and its
mid-drive pair) and goes on to the panoramas, the 3D models and the viewer; the
studies (`rpp_studies`) have a notebook each under `notebooks/studies/`.
