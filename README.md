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

Version **0.97.20** (`v0p97`): the core (`mppp`, the pipeline) and the research studies (`mppp_studies`) apart, the
freeze run as a driver of its own, and the products of one rover station past the block - a cleaned mesh with a
model of the rover, per-image depth and photometric maps, and a stereoscopic 360 degree panorama with a web
viewer (below). The paper's freeze release is 0.95 (`paper/paper.md`). Versions before 1.0 are
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

The same functions are importable (`mppp.pipeline.Settings`, `process_site`, `align`, `run_site`), and
`notebooks/pipeline_example.ipynb` calls them on the example data (Rockytop, sols 461-464) and shows the results,
start to finish: processing, the alignment and its health, the dense cloud, the station mesh, the depth and
photometric maps, the stereoscopic panorama, the exports.  The research studies have a notebook each under
`notebooks/studies/` (`camera_models`, `error_analysis`, `mask_training`, `sites_maps`).  Reruns reuse processed
images, features and matches; a changed setting or a promoted camera model changes the run key and realigns.

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
the package copies; on a working machine the models in use live there - `mppp cmods init DIR`, `mppp cmods status`,
`mppp cmods ship` - see docs/calibration.md):

* **Navcam** — COLMAP `THIN_PRISM_FISHEYE` at the full 5120 × 3840 frame (sx1 = sy1 = 0 held); focal lengths,
  principal point and k1 linear in camera temperature, offsets for the early mission, and a stereo rig whose
  rotation drifts with sol, with a knee in the pitch rate at sol 250. NL is the rig reference.
* **Hazcam** (front and rear) — the same form, with fx, fy and k1 linear in temperature, an early-mission offset and
  a rig drift in sol (`mppp calib hazcam`, fitted from the blocks). First models of the four eyes and both rigs
  ship since 0.94 (23 Oct 2026; one to six blocks each, three of the four eyes without a temperature term).
  Without a model each image
  starts from a least-squares fit of the fisheye to its own label CAHVORE (0.13–0.19 px rms for the front pair; 2–4
  px for the rear pair, whose models carry several mm of entrance-pupil movement), and a block refines fx, fy, k1, k2
  (`hazcam_intrinsics`) and the rig rotation. Since 0.93.1 the Hazcam frames of a block's stations are in every
  block (`--hazcam none` leaves them out); they come in after the Navcam block, with the Navcam cameras and rig held.
* **Mastcam-Z** — one model per zoom, a function of the focus count and the focus-motor backlash state only
  (lines in focus per eye and state; p1 = p2 = k3 = 0 held). Models ship for 34 mm (eight blocks) and, as first calibrations of two blocks each, for 63 and
  110 mm; 48 and 79 mm start from the label CAHVOR. The freeze run fits all five zooms again; its models are not
  in the package yet. An image is in the high-focal-length state unless the low-backlash list or its own focal fit says
  otherwise.

New models enter with `python -m mppp_studies.calibration.promote_cmods <files> --note "..."` (validated against the
form, logged in `cmods/CHANGES.md`; the replaced files stay in git history). The fits that produce them are the
calibration study, `mppp_studies.calibration` (0.96.0: the core package reads the promoted models and never solves
for them).

## Station products and the stereo panorama viewer

`mppp single-sites` (or `mppp dense`, `mesh`, `depthmaps`, `ods` one by one) makes the products of one rover station
from a finished block: a cleaned terrain mesh out to 75 m with a ground plane and a far field (100 m, 300 m or infinity per direction), the rover
model in it, per-image depth, normal, incidence, emission, phase and shadow maps, and an omnidirectional stereo
panorama, 8192 x 4096 per eye (16K with `--width 16384`), as PNG (8-bit and 16-bit linear) and JPEG. The stacked left-over-right files are
named `..._360_3DTD` and carry XMP metadata for headset viewers.

* The panoramas in a browser or a headset: **<https://ctate7163.github.io/MPPP/ods/>** (`docs/ods/`, a prototype;
  `mppp single-sites --publish docs/ods` adds stations).
* How they are made and what the files are: [docs/station-products.md](https://ctate7163.github.io/MPPP/station-products/).
* The rover model (derived from NASA's Perseverance model, credit NASA/JPL-Caltech), for use by others:
  [docs/rover-model.md](https://ctate7163.github.io/MPPP/rover-model/).

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
| `src/mppp/` | the pipeline: `pipeline` and `cli` (the workflow), `process` and `image` (PDS → images), `mask` (inference, the model files), `calib` (camera-model forms, the promoted models, their constraint to an image or block), `sfm` (COLMAP project, features, matching, the adjustment, exports, dense), `ibr` (the station products: mesh, rover model, depth and photometric maps, the stereoscopic panorama, its publication for the web viewer), `waypoints` |
| `src/mppp_studies/` | the studies (installed with the package, importing it, never imported by it): `calibration` (the camera-model solvers, their gates and promotion, the deck and fiducial studies, the chain driver of a freeze run), `error` (the reconstruction-error model), `mask_training` (training, audit, release of the mask model), `sites` (the site-map figures), `paper`, `misc` (what did not clearly belong anywhere) |
| `src/mppp/data/` | site definitions, waypoint snapshot, optical-depth table, the camera models in use (`cmods/`), the mask-model registry, the rover model (`rover/`), the single-station list (`single_sites.json`) |
| `notebooks/` | `pipeline_example.ipynb` (the pipeline end to end on the example data) and `studies/` (one notebook per study) |
| `scripts/` | the batch runs of the pipeline (`run_sites.py`, `process_sites.py`, `align_scape.py`, `stop_runs.py`, `check_sites.py`, `check_env.py`, `waypoint_check.py`), `build_rover_model.py` and `export_rover_glb.py`; `windows/` the .bat wrappers |
| `tests/` | pytest, one file per module; `tests/data/m20/` holds two public PDS products and the stored regression fixtures |
| `docs/` | `methods.md` (conventions, equations, flagged assumptions), `calibration.md`, `exports.md`, `station-products.md`, `rover-model.md`, `masks.md`, `roadmap.md`, `ods/` (the panorama viewer and its images), `rover/` (the rover model as glTF), `RELEASING.md`, `history/` (older changelogs and working notes) |
| `paper/` | the JOSS paper (`python -m mppp_studies.paper.build_paper` renders it) |

## Scope and limits

The Navcam and Mastcam-Z 34 mm models are the tested scope. The Hazcam models and the Mastcam-Z 63 and 110 mm
models are first calibrations from a few blocks; 48 and 79 mm start from the label models. The station products and
the viewer are prototypes: the rover model is one pose (arm stowed), and the viewer has not been checked in a
headset. Scale comes from the Navcam stereo baseline together with
the waypoint position priors (1 m; 10 m for stations without a waypoint of their own); orientation from a weak
attitude prior per frame. The alignment-health thresholds are provisional. See `docs/methods.md` for the flagged
assumptions and `docs/calibration.md` for the calibration loop and its checks.

## Data provenance

All Mars 2020 images used by MPPP, including the two test products in `tests/data/m20/` and the images the mask
model was trained on, are public products of the NASA Planetary Data System (PDS). Credits: Navcam
NASA/JPL-Caltech; Mastcam-Z NASA/JPL-Caltech/ASU/MSSS. The reconstruction masks, the mask model and the camera
models in `src/mppp/data/cmods/` are the author's own work. The rover model is derived from NASA's "Mars
Perseverance Rover, 3D Model" (credit NASA/JPL-Caltech), modified as described in `docs/rover-model.md`.
The panorama viewer includes three.js (MIT license). The waypoint snapshot is a copy of the public Mars 2020
waypoint table. The package contains no sensitive or non-public data.

## License and citation

Apache-2.0 (see [LICENSE](https://github.com/ctate7163/MPPP/blob/main/LICENSE) and [NOTICE](https://github.com/ctate7163/MPPP/blob/main/NOTICE)); released as open source with the approval of Malin Space
Science Systems. If you use MPPP, please cite it as described in [CITATION.cff](https://github.com/ctate7163/MPPP/blob/main/CITATION.cff).
