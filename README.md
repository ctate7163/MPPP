# MPPP — Mars Photogrammetry Preprocessing Pipeline

MPPP turns Mars 2020 Perseverance **Navcam** and **Mastcam-Z** PDS image products (`*.IMG`, RAD) into photogrammetry-ready inputs for **Agisoft Metashape** and **COLMAP**:

* radiometrically normalised 16-bit (and optionally 8-bit) images that keep their PDS file names, padded to the full detector frame;
* reconstruction masks (terrain vs rover hardware, sky and artefacts) from a trained segmentation model, shipped ready to use;
* per-image camera models from calibrated Metashape frame calibrations or the label CAHVOR model;
* pose priors in a local East–North–Up frame (metres, origin at the landing site) from the label CAHV model and the mission waypoints, plus Mars longitude/latitude;
* a Metashape reference file, a COLMAP prior model, and a manifest and configuration snapshot for full provenance.

For Navcam (optionally with the Mastcam-Z 34 mm frames of the same sols), MPPP also builds a complete COLMAP project — database, stereo rig, matches — and a prior-aligned, weighted bundle adjustment, with an **alignment health report** (tie points, reprojection error, camera-model change, stereo-rig stability, pose change).

Version **0.21.0** (`v0p21`). Versions before 1.0 are development releases; see [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
git clone https://github.com/ctate7163/MPPP.git
cd MPPP
pip install -e .[mask,sfm]          # masks (torch, timm, safetensors) and the COLMAP pipeline (pycolmap 4.2, pyceres 2.6)
pytest -m "not slow"                # optional: the test suite (the example PDS products are in tests/data)
```

A conda environment file is in `envs/mppp.yml`. Python ≥ 3.10.

**The mask model** (`mppp_mask_v2`, about 125 MB, safetensors; `mppp_mask_v1` = the previous one) is downloaded on first use into the user cache (`%LOCALAPPDATA%\mppp` on Windows, `~/.cache/mppp` on Linux, `~/Library/Caches/mppp` on macOS; `MPPP_CACHE` overrides) and checked against its SHA-256. Offline: `python -m mppp.mask.hub install <file.safetensors>`. `python -m mppp.mask.hub list` shows the status.

## Use

Notebooks (in `notebooks/`):

| notebook | what it does |
|---|---|
| `01_process_images.ipynb` | select PDS products (named scapes, sol ranges, waypoint radius, lists), process them, write Metashape and COLMAP inputs |
| `03_colmap_alignment.ipynb` | Navcam (+ optional Mastcam-Z 34 mm): COLMAP database with a left-referenced stereo rig and position priors, matching, CAHV-initialised weighted bundle adjustment, alignment health |
| `04_error_analysis.ipynb` | one or more alignments from 03: the error model's inputs measured from them (image precision ε, cross-station match gate vs angle and ΔLMST, decorrelation, view graph, registration) next to the values the model assumes |
| `05_camera_models.ipynb` | the refined cameras of several alignments from 03 side by side. Covers Navcam intrinsics and stereo rig, and Mastcam-Z focal length against focus. Differences are shown in pixels over the whole frame, with their effect on disparity and range. Also reports the distance from the flight (label) calibration, writes updated CAHVORE / CAHVOR models, and shows example images undistorted with the hardware mask screened |
| `training/02_train_mask.ipynb` | optional: retrain the mask model from a labelled mask set |

Or in Python:

```python
import mppp
from mppp.scapes import select_scape

paths, provenance = select_scape("belva", "D:/data/m2020")        # or mppp.select.find_imgs(...)
manifest = mppp.process_images(paths, "D:/scapes/belva", {"export": {"formats": ["PNG16"]}},
                               mppp.load_waypoints(), provenance=provenance)
```

To drop unsuitable frames, delete them from the output image folder and rerun with the same selection and `only_existing="PNG16"` (or `"PNG8"`): the manifest, references and COLMAP priors are rebuilt from the remaining images. With `reuse_existing=True`, images already processed with the same configuration are taken from the manifest instead of being processed again. `config["masking"]["skip_inference_at"] = ["S032D1184"]` keeps the rover in the images of chosen stations (only invalid pixels masked).

Outputs: `images_png16/<PDS stem>.png` (mask in the alpha channel), `references.txt` (+ `_absolute`) for Metashape *Import Reference*, `colmap/sparse_prior/` (cameras, image poses, stereo rig), `mppp_manifest_<version>.json`, `mppp_config_<version>.json`.

**Waypoints:** a snapshot of the M2020 waypoint table ships with MPPP; `mppp.load_waypoints(refresh=True)` downloads the current table into the user cache, which is used from then on (needed for recent sols).

## GPU on Windows (COLMAP feature extraction and matching)

The pip `pycolmap` wheel for Windows is CPU-only. The conda-forge build has CUDA, but it cannot be combined with the pip `pyceres` that the weighted bundle adjustment needs. Keep the main environment on the pip wheels and create a second environment for the GPU steps only (`envs/mppp_gpu.yml`); set `GPU_PY` in `03_colmap_alignment.ipynb` to that environment's `python.exe`, and extraction and matching run there as a subprocess. `mppp.sfm.check_gpu_python` and `check_ba_environment` verify the setup.

## Layout

| path | content |
|---|---|
| `src/mppp/` | the package |
| `src/mppp/data/` | package data: camera models (`m20_cmods/`: Metashape calibrations and the rational Navcam cameras), optical-depth table, waypoint snapshot, occlusion profiles, model registry (`models.json`) |
| `notebooks/` | the workflows above; `notebooks/training/` retrains the mask model |
| `docs/methods.md` | methods, conventions, equations and flagged assumptions |
| `docs/RELEASING.md` | how to release code and models (GitHub, Hugging Face, safetensors) |
| `tests/` | pytest suite; `tests/data/m20/` holds two public PDS products |
| `src/legacy/` | the pre-package code, kept only for the regression test |
| `studies/` | research scripts for the error model (not installed) |

| module | role |
|---|---|
| `mppp.process`, `mppp.image` | batch driver and the per-image pipeline (`MPPPImage`) |
| `mppp.filenames`, `mppp.select`, `mppp.scapes`, `mppp.labels` | file names, product selection, PDS labels |
| `mppp.camera`, `mppp.radiometry`, `mppp.waypoints` | camera models and poses, radiometric normalisation, rover positions |
| `mppp.writers`, `mppp.colmap` | image and metadata writers; COLMAP text I/O and camera math |
| `mppp.mask` | mask inference and the model registry (`mask.hub`) |
| `mppp.sfm` | the COLMAP pipeline and the alignment health report |
| `mppp.paths` | package data, user cache and checkpoint locations |

**Also included, not needed for processing:** mask-model training and label auditing (`mppp.mask.train`, `mask.dataset`, `mask.audit`, `notebooks/training/`), and a photogrammetric error model (`mppp.error`, under development; `docs/error/`).

## Scope and limits

MPPP supports two Mars 2020 cameras: **Navcam** (`NLF`/`NRF`) and **Mastcam-Z at 34 mm** (`ZL0`/`ZR0`, sequence `_034`), RAD products. The COLMAP pipeline (`mppp.sfm`) accepts only these and stops on anything else. Image processing will run on other Mars 2020 cameras and Mastcam-Z zooms, but they are outside the supported and tested scope.

- Navcam cameras start from a rational lens model (`M2020_N{L,R}_rational.json`, v0p20) that is valid to the frame corners (about 61° off-axis). The Metashape three-term calibration remains available (`navcam_distortion="polynomial"`), but it cannot be inverted beyond ~0.88 of the corner radius, so the image corners are lost.
- Mastcam-Z 34 mm: one camera per eye and focus bin, initialised from the label CAHVOR models, no stereo-rig constraint.
- Scale of a COLMAP block comes from the CAHV stereo baseline; position from loose (1 m) waypoint priors; orientation, since v0p20, from a weak (1°) CAHV attitude prior per frame. Without it, a block of a few nearly collinear stations could rotate about the line through them (0.6° at Three Forks with Navcam, 2.4° with Mastcam-Z).
- The alignment-health thresholds are provisional. See `docs/methods.md` §8 for the flagged assumptions.

## Data provenance

All Mars 2020 images used by MPPP, including the two test products in `tests/data/m20/` and the images the mask model was trained on, are public products of the NASA Planetary Data System (PDS). Credits: Navcam NASA/JPL-Caltech; Mastcam-Z NASA/JPL-Caltech/ASU/MSSS. The reconstruction masks, the mask model and the camera models in `src/mppp/data/m20_cmods/` are the author's own work. The waypoint snapshot is a copy of the public Mars 2020 waypoint table. The package contains no sensitive or non-public data.

## License and citation

Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)); released as open source with the approval of Malin Space Science Systems. If you use MPPP, please cite it as described in [CITATION.cff](CITATION.cff).
