# MPPP — Mars Photogrammetry Preprocessing Pipeline

[![docs](https://github.com/ctate7163/MPPP/actions/workflows/docs.yml/badge.svg)](https://ctate7163.github.io/MPPP/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Documentation: <https://ctate7163.github.io/MPPP/>
<!-- DOI badge: replace XXXXXXX with the Zenodo concept DOI after the first archived release (docs/RELEASING.md) -->

*Beyond the stereo baseline with Mars rover images.*

Every time a Mars rover stops, it looks around from a new viewpoint. The Mars 2020 Perseverance rover captures the
Martian landscape in stereo, often in 360° panoramas, and the images and their metadata are public in NASA's
Planetary Data System (PDS) — but in a format that common 3D reconstruction software cannot use. MPPP is the
PDS-to-SfM pipeline that fills this gap: it turns **Navcam** and **Mastcam-Z** PDS products (`*.IMG`, RAD) into
aligned structure-from-motion blocks in **COLMAP**, and into inputs for **Agisoft Metashape**.

Fixed-baseline stereo gives each rover stop its own range map; MPPP lets the images from *nearby* stops triangulate
together. A **block** is a set of rover stations within a few tens of metres of each other — Perseverance is
parked within 10 m of three or more earlier or later stops on about 20 % of sols — and several dozen blocks of
four or more panoramas have accumulated over the first 2,000 sols.

Five processes take a block from PDS inputs to SfM outputs:

1. **Images** — radiometrically linear 16-bit images (and 8-bit previews) that keep their PDS file names and are
   traceable, count for count, to the PDS radiance, with colours harmonised between instruments and brightness
   between times of day;
2. **Masks** — rover hardware and rover shadows (the sky is not masked), from a fine-tuned ConvNeXt-tiny
   segmentation model downloaded on first use, so that the matcher ties images together on the terrain, not on the
   rover;
3. **Cameras and stations** — the label CAHV/CAHVOR/CAHVORE models converted to COLMAP cameras, or the promoted
   temperature-, sol- and focus-aware calibrations of `mppp.calib`; each station placed in a local East–North–Up
   frame (metres) from the rover waypoints, which with the stereo baseline fixes the scale;
4. **Alignment** — a COLMAP project with stereo rigs, SIFT features matched by LightGlue, and a prior-weighted
   bundle adjustment on pyceres;
5. **Health and exports** — an alignment health report, and the measurements for the 3D reconstruction error
   analysis (`mppp.error`, under development).

The same alignments are measurements of the cameras: each block refines focal lengths, principal points,
distortion and the stereo rig to about 0.2 px, so the blocks across the mission form a calibration data set that
no single sol can provide. MPPP versions those refined models, checks them across blocks, and feeds them back as
the default cameras for the next run.

Version **0.94.4** (`v0p94`), a beta for the JOSS submission (`paper/paper.md`). Versions before 1.0 are
development releases; see [CHANGELOG.md](https://github.com/ctate7163/MPPP/blob/main/CHANGELOG.md).

## Install

From PyPI (<https://pypi.org/project/mppp/>):

```bash
pip install "mppp[mask,sfm,plots]"     # what `mppp run` needs; plain `pip install mppp` reads and processes PDS products
mppp doctor                            # what is installed, which camera models are in use
```

From the repository, with the tests, studies, notebooks and Windows scripts:

```bash
git clone https://github.com/ctate7163/MPPP.git
cd MPPP
pip install -e ".[all]"        # mask (torch, timm), sfm (pycolmap 4.2, pyceres 2.6, kornia), plots, notebooks, dev
pytest -q                      # the test suite runs on the two PDS products in tests/data/m20
```

Python ≥ 3.10. `envs/mppp.yml` is a conda environment with the same packages. The mask model (`mppp_mask_v3`,
125 MB) comes from Hugging Face, [`ctate7163/mppp-mask`](https://huggingface.co/ctate7163/mppp-mask), into the user
cache on first use (`MPPP_CACHE` overrides the location); `mppp mask install <file.safetensors>` installs it
offline. The LightGlue weights (45 MB) are fetched by kornia the same way; `MPPP_LIGHTGLUE_WEIGHTS` names a local
file.

Set three environment variables once: `MPPP_PDS` (the folder of PDS products), `MPPP_SCAPES` (where the block
folders go) and, optionally, `MPPP_GPU_PYTHON` (see *GPU on Windows*).

## Example data

A small Mars 2020 set to try the pipeline on: Rockytop (sols 461-464) - the main Navcam images, one front Hazcam pair
at each of the two waypoints (sols 461 and 464) and eight Mastcam-Z 34 mm frames (ZCAM03381, sol 464), in the layout of a local PDS archive.

```bash
mppp example fetch               # downloads the release asset (example-data-v1), checks it, unpacks to ~/mppp_example
export MPPP_PDS=~/mppp_example          # the folder holding datadrive/ and zcam/
mppp run rockytop_example --zcam
```

## Use

A **site** is a block of rover stations — where Perseverance ended its drives repeatedly within a few tens of
metres — defined by its sol ranges in `src/mppp/data/sites.json` (`mppp sites list`). Each site has a WORK folder
`<MPPP_SCAPES>/mars2020_sol_<first sol>_<site>[_zcam]`.

```bash
mppp process rockytop            # select the site's products, process them into <WORK>/processed/
mppp align  <WORK>               # COLMAP project, features, exhaustive matching, bundle adjustment, health report
mppp run    rockytop --zcam      # both, with the Mastcam-Z 34 mm frames of the same sols
mppp align  <WORK> --set attitude_prior_deg=1 --variant tight      # another setting, into <WORK>/colmap_tight/
mppp cmods list                  # the camera models in use
```

The same functions are importable (`mppp.pipeline.Settings`, `process_site`, `align`, `run_site`), and the
notebooks in `notebooks/` call them and show the results: `01_process_images` (selection and processing),
`02_colmap_alignment` (the alignment, its health table and camera changes), `03_error_analysis` (the error model's
inputs measured from alignments; `mppp.error` is under development), `training/train_mask` (retraining the mask
model). Reruns reuse processed images, features and matches; a changed setting or a promoted camera model changes
the run key and realigns.

Outputs of `process`: `images_png16/<PDS stem>.png` (the mask in the alpha channel), `references.txt` for
Metashape *Import Reference*, `colmap/sparse_prior/` (cameras, poses, rig), `mppp_manifest_<version>.json` and
`mppp_config_<version>.json` for provenance. Outputs of `align`: `colmap/sparse/cahv_ba/` (the refined block),
`health.json`, `summary.json`, `error_input/` (the exports for the error analysis), `runs/<time>/log.txt`.

For many sites at once, `scripts/run_sites.py --group ecam_consensus` and `scripts/process_sites.py --all`
(Windows: `scripts\windows\run_sites.bat`, `process_sites.bat`, `align_here.bat`, `check_sites.bat`,
`stop_mppp.bat`) run the command site after site with logs and status files.

## Camera models

`mppp.calib` defines one frozen model form per camera family and ships the promoted models in
`src/mppp/data/cmods/` (`mppp cmods list`; `MPPP_CMODS` points at another folder, whose missing files fall back to
the package copies):

* **Navcam** — COLMAP `THIN_PRISM_FISHEYE` at the full 5120 × 3840 frame (sx1 = sy1 = 0 held); focal lengths,
  principal point and k1 linear in camera temperature, offsets for the early mission, and a stereo rig whose
  rotation drifts linearly with sol. NL is the rig reference.
* **Hazcam** (front and rear) — the same form, with fx, fy and k1 linear in temperature, an early-mission offset and
  a rig drift in sol (`mppp calib hazcam`, fitted from the blocks; not promoted yet). Without a model each image
  starts from a least-squares fit of the fisheye to its own label CAHVORE (0.13–0.19 px rms for the front pair; 2–4
  px for the rear pair, whose models carry several mm of entrance-pupil movement), and a block refines fx, fy, k1, k2
  (`hazcam_intrinsics`) and the rig rotation. Since 0.93.1 the Hazcam frames of a block's stations are in every
  block (`--hazcam none` leaves them out); they come in after the Navcam block, with the Navcam cameras and rig held.
* **Mastcam-Z** — one model per zoom, a function of the focus count and the focus-motor backlash state only
  (lines in focus per eye and state; p1 = p2 = k3 = 0 held). 34 mm is calibrated; the other zooms start from the
  label CAHVOR. An image is in the high-focal-length state unless the low-backlash list or its own focal fit says
  otherwise.

New models enter with `scripts/promote_cmods.py <files> --note "..."` (validated against the form, logged in
`cmods/CHANGES.md`; the replaced files stay in git history). The fits that produce them are in `mppp.calib.fit`;
the calibration campaigns themselves are research code in `studies/calibration/`.

## GPU on Windows

The pip `pycolmap` wheel for Windows is CPU-only, and the conda-forge CUDA build cannot be combined with the pip
`pyceres` that the bundle adjustment needs. Keep the main environment on the pip wheels and create
`envs/mppp_gpu.yml` for SIFT extraction only; point `MPPP_GPU_PYTHON` at its `python.exe` and extraction runs there
as a subprocess (`mppp.sfm.check_gpu_python`). LightGlue matching runs on the main environment's torch GPU when
there is one, and falls back to COLMAP's SIFT matcher (recorded as such) when kornia is missing or the projected
time exceeds the per-block budget. `mppp align --gui` writes the files that open a project in the COLMAP GUI
(`COLMAP_BAT` names the COLMAP.bat).

## Layout

| path | content |
|---|---|
| `src/mppp/` | the package: `pipeline` and `cli` (the workflow), `process` and `image` (PDS → images), `mask` (inference; `mask/training`), `calib` (camera-model forms, models, constraints, fits), `sfm` (COLMAP project, features, LightGlue matching, bundle adjustment, health, export), `waypoints` (waypoint table, sites, blocks), `error` (the error model, under development), `camera`/`cmod`/`colmap`/`labels`/`radiometry`/`writers`/`config`/`paths` |
| `src/mppp/data/` | site definitions, waypoint snapshot, optical-depth table, the camera models in use (`cmods/`), the mask-model registry |
| `notebooks/` | the workflows above as thin callers of the pipeline |
| `scripts/` | batch runs (`run_sites.py`, `process_sites.py`, `align_scape.py`, `stop_runs.py`), `promote_cmods.py`, `check_sites.py`, `build_paper.py`, `release_mask_model.py`; `windows/` the .bat wrappers |
| `studies/` | research code that imports the package and is not installed: `calibration/` (Navcam consensus, Mastcam-Z calibration, their scripts and notebook), `m20_sites/` (the waypoint-map study), `error/`, `analyses/`, `experiments/` |
| `tests/` | pytest, one file per module; `tests/data/m20/` holds two public PDS products and the stored regression fixtures |
| `docs/` | `methods.md` (conventions, equations, flagged assumptions), `RELEASING.md`, `HANDOFF_v0p87.md` (the state of the calibration work), `history/` |
| `paper/` | the JOSS paper (`python scripts/build_paper.py` renders it) |

## Scope and limits

Calibrated models exist for Navcam and Mastcam-Z at 34 mm; the Hazcams (label-fit start cameras, refined per block)
and the other Mastcam-Z zooms are processed with start models and are not part of the tested scope yet. Scale comes from the Navcam stereo baseline together with
the waypoint position priors (1 m; 10 m for stations without a waypoint of their own); orientation from a weak
attitude prior per frame. The alignment-health thresholds are provisional. See `docs/methods.md` for the flagged
assumptions and `docs/HANDOFF_v0p87.md` for the open calibration questions.

## Data provenance

All Mars 2020 images used by MPPP, including the two test products in `tests/data/m20/` and the images the mask
model was trained on, are public products of the NASA Planetary Data System (PDS). Credits: Navcam
NASA/JPL-Caltech; Mastcam-Z NASA/JPL-Caltech/ASU/MSSS. The reconstruction masks, the mask model and the camera
models in `src/mppp/data/cmods/` are the author's own work. The waypoint snapshot is a copy of the public Mars 2020
waypoint table. The package contains no sensitive or non-public data.

## License and citation

Apache-2.0 (see [LICENSE](https://github.com/ctate7163/MPPP/blob/main/LICENSE) and [NOTICE](https://github.com/ctate7163/MPPP/blob/main/NOTICE)); released as open source with the approval of Malin Space
Science Systems. If you use MPPP, please cite it as described in [CITATION.cff](https://github.com/ctate7163/MPPP/blob/main/CITATION.cff).
