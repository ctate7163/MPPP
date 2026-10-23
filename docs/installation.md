# Installation

MPPP needs Python 3.10 or later. From PyPI (<https://pypi.org/project/mppp/>, 0.94.4 onwards):

```bash
pip install "mppp[mask,sfm,plots]"
```

From the repository (the tests, studies, notebooks and Windows scripts are only there):

```bash
git clone https://github.com/ctate7163/MPPP.git
cd MPPP
pip install -e ".[all]"
pytest -q                      # the test suite runs on the two PDS products in tests/data/m20
```

`envs/mppp.yml` is a conda environment with the same packages.

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

For example, `pip install "mppp[mask,sfm,plots]"` (or `pip install -e ".[mask,sfm,plots]"` in a clone) installs what
`mppp run` needs.

## Models downloaded on first use

- The mask model (`mppp_mask_v3`, 125 MB) comes from Hugging Face,
  [`ctate7163/mppp-mask`](https://huggingface.co/ctate7163/mppp-mask), into the user cache. `MPPP_CACHE` overrides
  the location; `mppp mask install <file.safetensors>` installs it offline.
- The LightGlue weights (45 MB) are fetched by kornia the same way; `MPPP_LIGHTGLUE_WEIGHTS` names a local file.

## Environment variables

| variable | meaning |
|---|---|
| `MPPP_PDS` | the folder of PDS products (searched recursively) |
| `MPPP_SCAPES` | where the block (WORK) folders are written |
| `MPPP_GPU_PYTHON` | optional: the `python` of a CUDA pycolmap environment used for SIFT extraction and matching |
| `MPPP_CACHE` | optional: the cache folder for the mask model and the waypoint table |
| `MPPP_CMODS` | optional: another camera-model folder (missing files fall back to the package copies) |
| `COLMAP_BAT` | optional: the `COLMAP.bat` that `mppp align --gui` uses |

`mppp doctor` reports which of these are set, which dependencies are missing, and the state of the models and the
waypoint table.

## GPU on Windows

The pip `pycolmap` wheel for Windows is CPU-only, and the conda-forge CUDA build cannot be combined with the pip
`pyceres` that the bundle adjustment needs. Keep the main environment on the pip wheels and create
`envs/mppp_gpu.yml` for SIFT extraction only; point `MPPP_GPU_PYTHON` at its `python.exe` and extraction runs there as
a subprocess. LightGlue matching runs on the main environment's torch GPU when there is one. The Windows batch files
are described in [Windows batch files](WINDOWS.md).

## Building this documentation

```bash
pip install -e ".[docs]"
mkdocs serve                   # preview at http://127.0.0.1:8000
mkdocs build --strict          # what the docs workflow runs
```
