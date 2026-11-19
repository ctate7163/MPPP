# Installation

RPP needs Python 3.10 or later. Install RPP into a conda environment of its own (Miniforge, Miniconda or Anaconda), from the GitHub repository.
It will be on PyPI as `rover-photogrammetry` (the import name and the command are `rpp`). The same
commands work in a Windows Anaconda Prompt, PowerShell and a Linux or macOS shell:

```bash
conda create -n rpp -c conda-forge python=3.12 git -y
conda activate rpp
git clone --depth 1 https://github.com/ctate7163/rpp.git
cd rpp
pip install -e ".[mask,sfm,plots]"
rpp doctor
```

- `--depth 1` downloads the current files only (about 40 MB) instead of the whole history.
- `[mask,sfm,plots]` is what `rpp run` needs: torch and timm (masks), pycolmap 4.2, pyceres 2.6 and kornia (the
  alignment), matplotlib (plots). PyTorch is the largest download; for a CUDA build install torch first as
  <https://pytorch.org> says, and pip keeps it. `pip install -e .` alone reads and processes PDS products;
  `".[all]"` adds the notebooks, the maps, the tests and the studies' packages.
- `rpp doctor` reports what is installed and which camera models are in use.
- `envs/rpp.yml` makes the same environment in one step: `conda env create -f envs/rpp.yml`, then
  `conda activate rpp` and `pip install -e .` in the clone.

No PDS data comes with the code: the example set, the products of other sols and the two products the tests use are
downloaded from the PDS Imaging Node when they are needed (below; for the tests, `RPP_TEST_FETCH=1`).

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The two PDS products some tests read (a Navcam frame of sol 461 and a Mastcam-Z frame of sol 464, 11 MB) are not in
the repository. Set `RPP_TEST_FETCH=1` (`set RPP_TEST_FETCH=1` in a Windows prompt) once to download them from the
PDS into the user cache, or `RPP_TEST_DATA` to a folder that holds them; without them those tests are skipped.

## Optional dependency groups

The base install reads and processes PDS products. The other parts of the pipeline are optional groups in
`pyproject.toml`:

| group | packages | used for |
|---|---|---|
| `mask` | torch, timm, safetensors | mask inference |
| `sfm` | pycolmap 4.2, pyceres 2.6, torch, kornia | the COLMAP alignment; kornia for SIFT + LightGlue |
| `plots` | matplotlib, pillow | health and camera plots |
| `error` | matplotlib, pillow | the error model (under development) |
| `maps` | matplotlib, scipy, pillow, adjustText, rasterio, pyproj | the M20 site-map figures |
| `notebooks` | jupyterlab, matplotlib, pillow | the notebooks in `notebooks/` |
| `dev` | pytest, flake8, nbformat, pillow | tests and static checks |
| `docs` | mkdocs, mkdocs-material, mkdocstrings | this documentation site |
| `all` | mask, sfm, error, plots, maps, notebooks, dev | everything except `release` and `docs` |

For example, `pip install -e ".[mask,sfm,plots]"` in the clone installs what `rpp run` needs.

## Models downloaded on first use

- The mask model (`rpp_mask_v3`, 125 MB) comes from Hugging Face,
  [`ctate7163/rpp-mask`](https://huggingface.co/ctate7163/rpp-mask), into the user cache. `RPP_CACHE` overrides
  the location; `rpp mask install <file.safetensors>` installs it offline.
- The LightGlue weights (45 MB) are fetched by kornia the same way; `RPP_LIGHTGLUE_WEIGHTS` names a local file.

## Environment variables

| variable | meaning |
|---|---|
| `RPP_PDS` | the folder of PDS products (searched recursively) |
| `RPP_SCAPES` | where the block (WORK) folders are written |
| `RPP_GPU_PYTHON` | optional: the `python` of a CUDA pycolmap environment used for SIFT extraction and matching |
| `RPP_CACHE` | optional: the cache folder for the mask model and the waypoint table |
| `RPP_CMODS` | optional: another camera-model folder (missing files fall back to the package copies) |
| `COLMAP_BAT` | optional: the `COLMAP.bat` that `rpp align --gui` uses |

`rpp doctor` reports which of these are set, which dependencies are missing, and the state of the models and the
waypoint table.

## GPU on Windows

The pip `pycolmap` wheel for Windows is CPU-only, and the conda-forge CUDA build cannot be combined with the pip
`pyceres` that the bundle adjustment needs. Keep the main environment on the pip wheels and create
`envs/rpp_gpu.yml` for SIFT extraction only; point `RPP_GPU_PYTHON` at its `python.exe` and extraction runs there as
a subprocess. LightGlue matching runs on the main environment's torch GPU when there is one. The Windows batch files
are described in [Windows batch files](WINDOWS.md).

## Building this documentation

```bash
pip install -e ".[docs]"
mkdocs serve                   # preview at http://127.0.0.1:8000
mkdocs build --strict          # what the docs workflow runs
```
