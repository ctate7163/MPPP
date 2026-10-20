# MPPP — Mars Photogrammetry Preprocessing Pipeline

MPPP turns Mars 2020 Perseverance **Navcam** and **Mastcam-Z** PDS image products (`*.IMG`, RAD) into photogrammetry-ready inputs for **Agisoft Metashape** and **COLMAP**:

* radiometrically normalised 16-bit (and optionally 8-bit) images that keep their PDS file names, padded to the full detector frame;
* reconstruction masks (rover hardware, rover shadows and other artificial objects excluded; the sky is not masked) from a trained segmentation model, shipped ready to use;
* per-image camera models from calibrated Metashape frame calibrations or the label CAHVOR model;
* pose priors in a local East–North–Up frame (metres, origin at the landing site) from the label CAHV model and the mission waypoints, plus Mars longitude/latitude;
* a Metashape reference file, a COLMAP prior model, and a manifest and configuration snapshot for full provenance.

For Navcam (optionally with the Mastcam-Z 34 mm frames of the same sols), MPPP also builds a complete COLMAP project — database, stereo rig, matches — and a prior-aligned, weighted bundle adjustment, with an **alignment health report** (tie points, reprojection error, camera-model change, stereo-rig stability, pose change).

Version **0.21.1** (`v0p21`). Versions before 1.0 are development releases; see [CHANGELOG.md](CHANGELOG.md).

## Install

```bash
git clone https://github.com/ctate7163/MPPP.git
cd MPPP
pip install -e .[mask,sfm]          # masks (torch, timm, safetensors) and the COLMAP pipeline (pycolmap 4.2, pyceres 2.6)
pytest -m "not slow"                # optional: the test suite (the example PDS products are in tests/data)
```

**Git and GitHub (v0p50).** The history is in git; the private repository is `github.com/ctate7163/MPPP`. After each delivery from Claude: `_transfer\claude_sync.bat` (fast-forwards to `_transfer\mppp_latest.bundle`, stashes local changes to tracked files, pushes to GitHub, writes `_transfer\mppp_pc.bundle`; v0p82 - it replaces sync_from_claude.bat, adopt_claude.bat and github_push.bat, removed in v0p85).


A conda environment file is in `envs/mppp.yml`. Python ≥ 3.10.

**The mask model** (`mppp_mask_v3`, about 125 MB, safetensors; `mppp_mask_v2` and `v1` = the previous ones) comes from Hugging Face, [`ctate7163/mppp-mask`](https://huggingface.co/ctate7163/mppp-mask). It is downloaded on first use into the user cache (`%LOCALAPPDATA%\mppp` on Windows, `~/.cache/mppp` on Linux, `~/Library/Caches/mppp` on macOS; `MPPP_CACHE` overrides) and checked against its SHA-256; a cached copy that is not the released file is downloaded again. Offline: `python -m mppp.mask.hub install <file.safetensors>`. `python -m mppp.mask.hub list` shows the status.

## Use

The pipeline is importable (`mppp.pipeline`: `Settings`, `select_images`, `process_site`, `align`, `run_site`) and a command-line tool:

```
mppp process SITE [--root DIR] [--pds DIR] [--zcam] [--set NAME=VALUE ...] [--settings FILE]   # select and process
mppp align WORK [--variant NAME] [--set ...] [--gui] [--stop-after STEP]                         # align what is processed
mppp run SITE [...]                                                                             # both
mppp sites [list|check|folder SITE]   mppp cmods [list|promote PATH --note ...]   mppp mask [list|fetch|install FILE]
```

`MPPP_PDS` (the PDS archive) and `MPPP_SCAPES` (the root of the site folders) are the defaults of `--pds` and `--root`; `MPPP_GPU_PYTHON` names a CUDA pycolmap environment for the feature and matching steps. Every setting is a field of `mppp.pipeline.Settings` (`help(Settings)`); a site's `settings` in `sites.json`, `<WORK>/mppp_settings.json`, `--settings FILE` and `--set` override it, in that order.

Notebooks (in `notebooks/`) are thin callers of the same functions that show the results:

| notebook | what it does |
|---|---|
| `01_process_images.ipynb` | `select_images` + `process_site`: one PDS product per exposure of a site, processed into `<WORK>/processed` (8-bit PNG, terrain mask, CAHV + waypoint pose, manifest) |
| `02_colmap_alignment.ipynb` | `align`: COLMAP database with a left-referenced stereo rig and position priors, matching, CAHV-initialised weighted bundle adjustment, the alignment health report, the camera changes and the export for the error analysis |
| `03_error_analysis.ipynb` | one or more alignments from 02: the error model's inputs measured from them (image precision ε, cross-station match gate vs angle and ΔLMST, decorrelation, view graph, registration) next to the values the model assumes |
| `training/train_mask.ipynb` | optional: retrain the mask model from a labelled mask set |

**Batch runs.** The sites and their sol ranges are in `src/mppp/data/sites.json`. Each site has its own WORK folder, `<root>/mars2020_sol_<first sol>_<site>` or, with Mastcam-Z, `..._zcam`. `mppp process` / `mppp align` / `mppp run` and the scripts below write a log (`<WORK>/runs/<time>/log.txt`) and a status file with a heartbeat (`<WORK>/mppp_status.json`).

- **Python for the .bat files:** `scripts\windows\mppp_env.bat` looks for a Python with pycolmap and pyceres (every conda environment it can find). If it finds none it lists what it tried; then write the notebooks' Python (`import sys; print(sys.executable)` in a notebook) into `scripts\windows\mppp_python.txt`. The .bat files can be copied anywhere; they find MPPP through `MPPP_HOME` (default `D:\code\MPPP`).
- **Process only (many sites):** `scripts\windows\process_sites.bat` (choose the sites at its top), or `python scripts\process_sites.py --all` (also `--all --zcam` for the zcam34 sites, or `--sites rockytop sid_chal_rocks`). It selects and processes each site's images into `<WORK>\processed\` with the pipeline's defaults, one site after the other, without aligning. Sites already processed with the same settings are skipped.
- **Camera models:** `src\mppp\data\cmods\` (v0p50: the one folder of camera models; `params\` is retired): the Navcam cameras and rig and the Mastcam-Z focus model in use, and the flight calibrations. Every alignment starts from them; replace the models in use only with `mppp cmods promote <folder or files> --note "..."` (`scripts\promote_cmods.py` does the same; the git history keeps the old ones), then commit.
- **Site list:** `src\mppp\data\sites.json`; check it after editing with `scripts\windows\check_sites.bat`.
- **Time of day of the images:** `python scripts\lmst_histogram.py` (all sites under `D:\scapes` and `colmap_old`; `--sites a b`) writes `lmst_histogram.png` and a per-site table into `Claude outputs\`.
- **Stop runs:** `scripts\windows\stop_mppp.bat` lists every MPPP run on the computer and stops them (with their worker processes) after asking. Only one batch (`process_sites` or `run_sites`) runs on a scapes folder at a time; `sites_status.bat` shows it.
- **One WORK folder:** copy `scripts\windows\align_here.bat` into the folder that holds `processed\` and double-click it. It aligns the images already processed there with the pipeline's default settings, in a minimised window.
  - Double-click it again to see the status.
  - Progress is in `runs\<time>\log.txt`, and the results are in `colmap\`.
  - The same from a prompt: `mppp align <WORK>` (or `python scripts\align_scape.py <WORK>`).
- **Every site:** `scripts\windows\run_sites.bat` (Navcam blocks, group `navcam_consensus`), `run_sites_zcam.bat` (Navcam + every Mastcam-Z zoom of each site's `"zcam"` list, group `zcam_consensus`) and `run_sites_zcam34.bat` (only the 34 mm frames, group `zcam34_consensus`; v0p61; for another zoom: `python scripts\run_sites.py --group zcam63_consensus --zcam --set zcam_zooms=[63]`). Or `python scripts\run_sites.py --group navcam_consensus` (also `--group zcam34_consensus --zcam --set zcam_zooms=[34]`, or `--sites rockytop chal_rocks_sid`). It selects, processes and aligns each site into its folder, one after the other.
  - Sites already finished with the same settings are skipped.
  - `sites_status.bat` (or `run_sites.py --status`) lists what is running, finished, failed or stopped.
- **Reruns redo only the alignment.** The processed images, the SIFT features and the matches are reused. So:
  - **To leave images out,** list them in `<WORK>\exclude_images.txt`: a station `S032D1184`, `sol:658`, `seq:NCAM08111` or a file-name pattern such as `ZR0_0690_*`.
  - **To change SfM settings,** use `<WORK>\mppp_settings.json` (`{"ATTITUDE_PRIOR_DEG": 1.0}`) or `--set NAME=VALUE`.
  - **To try a camera model** without losing the default results, use `--variant NAME`. It writes to `<WORK>\colmap_NAME` and starts from `colmap\`'s features and matches.

`scripts/run_scapes.py` (the v0p22 batch runner) still works:

```
python scripts\run_scapes.py --sites taylorfjellet rockytop belva threeforks_south landing --zcam --reprocess ^
    --root D:\scapes\v0p22 --gpu-py C:\Users\<you>\AppData\Local\miniconda3\envs\mppp_gpu\python.exe
```

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

The pip `pycolmap` wheel for Windows is CPU-only. The conda-forge build has CUDA, but it cannot be combined with the pip `pyceres` that the weighted bundle adjustment needs. Keep the main environment on the pip wheels and create a second environment for the GPU steps only (`envs/mppp_gpu.yml`); set `MPPP_GPU_PYTHON` (`Settings.gpu_python`) to that environment's `python.exe`, and extraction and matching run there as a subprocess. `mppp.sfm.check_gpu_python` and `check_ba_environment` verify the setup.

## Layout

| path | content |
|---|---|
| `src/mppp/` | the package |
| `src/mppp/data/` | package data: the site definitions (`sites.json`), camera models (`cmods/`: the models in use - Navcam cameras and rig, Mastcam-Z focus model - plus the flight Metashape calibrations and the rational Navcam cameras; README and CHANGES there), optical-depth table, waypoint snapshot, occlusion profiles, model registry (`models.json`) |
| `notebooks/` | the workflows above; `notebooks/training/` retrains the mask model |
| `scripts/` | `align_scape.py` (one WORK folder) and `run_sites.py` (many sites, one `mppp align` process each): the pipeline with logs and status; `process_sites.py` (select and process only); `_batch.py` (batch lock, stopping runs); `windows/`: `process_sites.bat`, `align_here.bat`, `run_sites.bat`, `sites_status.bat` |
| `docs/methods.md` | methods, conventions, equations and flagged assumptions |
| `docs/RELEASING.md` | how to release code and models (GitHub, Hugging Face, safetensors) |
| `tests/` | pytest suite, one file per module (`test_sfm_reconstruction.py`, `test_processing.py`, ...; v0p50), shared helpers in `tests/helpers.py`; `tests/data/m20/` holds two public PDS products |
| `studies/` | analyses and one-off experiments (`studies/experiments/`), not part of the package |
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

**Also included, not needed for processing:** mask-model training and label auditing (`mppp.mask.training`, `notebooks/training/`), and a photogrammetric error model (`mppp.error`, under development; its original notes are in `docs/history/`).

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
