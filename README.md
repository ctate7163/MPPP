# MPPP — Mars Photogrammetry Preprocessing Pipeline

MPPP turns Mars 2020 Perseverance **Navcam** and **Mastcam-Z** PDS image products (`*.IMG`, RAD) into photogrammetry-ready inputs for **Agisoft Metashape** and **COLMAP**:

* radiometrically normalised 16-bit (and optionally 8-bit) images that keep their PDS file names, padded to the full detector frame;
* reconstruction masks (terrain vs rover hardware, sky and artefacts) from a trained segmentation model, shipped ready to use;
* per-image camera models from calibrated Metashape frame calibrations or the label CAHVOR model;
* pose priors in a local East–North–Up frame (metres, origin at the landing site) from the label CAHV model and the mission waypoints, plus Mars longitude/latitude;
* a Metashape reference file, a COLMAP prior model, and a manifest and configuration snapshot for full provenance.

For Navcam (optionally with the Mastcam-Z 34 mm frames of the same sols), MPPP also builds a complete COLMAP project — database, stereo rig, matches — and a prior-aligned, weighted bundle adjustment, with an **alignment health report** (tie points, reprojection error, camera-model change, stereo-rig stability, pose change).

Version **0.14.3** (`v0p14`). Versions before 1.0 are development releases; see [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
git clone https://github.com/ctate7163/MPPP.git
cd MPPP
pip install -e .[mask,sfm]          # masks (torch, timm, safetensors) and the COLMAP pipeline (pycolmap 4.2, pyceres 2.6)
pytest -m "not slow"                # optional: the test suite (the example PDS products are in tests/data)
```

A conda environment file is in `envs/mppp.yml`. Python ≥ 3.10.

**The mask model** (`mppp_mask_v1`, about 125 MB, safetensors) is downloaded on first use into the user cache (`%LOCALAPPDATA%\mppp` on Windows, `~/.cache/mppp` on Linux, `~/Library/Caches/mppp` on macOS; `MPPP_CACHE` overrides) and checked against its SHA-256. Offline: `python -m mppp.mask.hub install <file.safetensors>`. `python -m mppp.mask.hub list` shows the status.

## Use

Notebooks (in `notebooks/`):

| notebook | what it does |
|---|---|
| `01_process_images.ipynb` | select PDS products (named scapes, sol ranges, waypoint radius, lists), process them, write Metashape and COLMAP inputs |
| `03_colmap_alignment.ipynb` | Navcam (+ optional Mastcam-Z 34 mm): COLMAP database with a left-referenced stereo rig and position priors, matching, CAHV-initialised weighted bundle adjustment, alignment health |

Or in Python:

```python
import mppp
from mppp.scapes import select_scape

paths, provenance = select_scape("belva", "D:/data/m2020")        # or mppp.select.find_imgs(...)
manifest = mppp.process_images(paths, "D:/scapes/belva", {"export": {"formats": ["PNG16"]}},
                               mppp.load_waypoints(), provenance=provenance)
```

To drop unsuitable frames, delete them from the output image folder and rerun with the same selection and `only_existing="PNG16"` (or `"PNG8"`): only the remaining images are processed again, and the manifest, references and COLMAP priors are rebuilt from them.

Outputs: `images_png16/<PDS stem>.png` (mask in the alpha channel), `references.txt` (+ `_absolute`) for Metashape *Import Reference*, `colmap/sparse_prior/` (cameras, image poses, stereo rig), `mppp_manifest_<version>.json`, `mppp_config_<version>.json`.

**Waypoints:** a snapshot of the M2020 waypoint table ships with MPPP; `mppp.load_waypoints(refresh=True)` downloads the current table into the user cache, which is used from then on (needed for recent sols).

## GPU on Windows (COLMAP feature extraction and matching)

The pip `pycolmap` wheel for Windows is CPU-only. The conda-forge build has CUDA, but it cannot be combined with the pip `pyceres` that the weighted bundle adjustment needs. Keep the main environment on the pip wheels and create a second environment for the GPU steps only (`envs/mppp_gpu.yml`); set `GPU_PY` in `03_colmap_alignment.ipynb` to that environment's `python.exe`, and extraction and matching run there as a subprocess. `mppp.sfm.check_gpu_python` and `check_ba_environment` verify the setup.

## Layout

| path | content |
|---|---|
| `src/mppp/` | the package |
| `src/mppp/data/` | package data: Metashape calibrations (`m20_cmods/`), optical-depth table, waypoint snapshot, occlusion profiles, model registry (`models.json`) |
| `notebooks/` | the two workflows above; `notebooks/training/` retrains the mask model |
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

Supported: Mars 2020 Navcam and Mastcam-Z RAD products (other engineering cameras are processed by the same code but are less tested). The full COLMAP pipeline (`mppp.sfm`) handles Navcam, and Mastcam-Z 34 mm as an experimental option (one camera per eye, no stereo-rig constraint); other Mastcam-Z zooms get a COLMAP prior model and Metashape references. Hazcam XML calibrations have a K4 term that COLMAP's FULL_OPENCV model cannot represent. Orientation and scale of a COLMAP block come from the CAHV stereo baseline and loose (1 m) waypoint position priors; no attitude priors are used yet. The alignment-health thresholds are provisional. See `docs/methods.md` §8 for the flagged assumptions.

## License and citation

Apache-2.0 (see [LICENSE](LICENSE)). If you use MPPP, please cite it as described in [CITATION.cff](CITATION.cff).
