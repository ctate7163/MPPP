# Installation

RPP needs Python 3.10 or later. It runs on Windows, Linux and macOS. A GPU is optional.

## From PyPI

Install RPP into a conda environment of its own (Miniforge, Miniconda or Anaconda). The commands are the same in a
Windows Anaconda Prompt, PowerShell, and a Linux or macOS shell.

```bash
conda create -n rpp -c conda-forge python=3.12 -y
conda activate rpp
pip install "rover-photogrammetry[pipeline]"
rpp doctor
```

- The distribution is `rover-photogrammetry`. The import name and the command are `rpp`.
- `[pipeline]` is what `rpp run` needs: torch and timm (masks), pycolmap 4.2, pyceres 2.6 and kornia (the
  alignment), matplotlib (plots). Without it, `pip install rover-photogrammetry` reads and processes PDS products
  only.
- `rpp doctor` reports what is installed, what is missing and which camera models are in use.
- The PyPI package named `rpp` is a different project (a REAPER file parser) that uses the same import name. Do not
  install both in one environment.

**GPU.** PyTorch is the largest download. For a CUDA build, install torch first with the command
<https://pytorch.org> gives for your system; pip then keeps the torch that is there. See also
[GPU on Windows](#gpu-on-windows).

## From the repository

Install from a clone to change the code, to run the notebooks, or to use the research studies (`rpp_studies`),
which are not in the PyPI package.

```bash
conda create -n rpp -c conda-forge python=3.12 git -y
conda activate rpp
git clone --depth 1 https://github.com/ctate7163/rpp.git
cd rpp
pip install -e ".[all]"
rpp doctor
```

- `--depth 1` downloads the current files only instead of the whole history.
- `envs/rpp.yml` makes the same environment in one step: `conda env create -f envs/rpp.yml`, then
  `conda activate rpp` and `pip install -e .` in the clone.

## Data

No PDS data comes with the code. RPP downloads what it needs from the PDS Imaging Node:

- `rpp example fetch` downloads the example set (Rockytop, sols 461-464);
- `rpp pds fetch --sols FIRST LAST` downloads the products of other sols into `RPP_PDS`;
- the tests download two products when `RPP_TEST_FETCH=1` is set.

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
| `pipeline` | the groups `mask`, `sfm` and `plots` | PDS products to an aligned block: `rpp run` |
| `mask` | torch, timm, safetensors | mask inference |
| `sfm` | pycolmap 4.2, pyceres 2.6, torch, kornia | the COLMAP alignment; kornia for SIFT + LightGlue |
| `plots` | matplotlib, pillow | health and camera plots |
| `stations` | open3d, huggingface_hub | station products and the viewer (`ibr` is the name of this group before 1.0) |
| `notebooks` | jupyterlab, matplotlib, pillow | the notebooks in `notebooks/` |
| `error`, `maps`, `rover`, `release` | matplotlib, rasterio, pyproj, spiceypy, ... | the research studies of a source checkout |
| `dev` | pytest, flake8, nbformat, pillow | tests and static checks |
| `docs` | mkdocs, mkdocs-material, mkdocstrings | this documentation site |
| `all` | pipeline, stations, notebooks, error, maps, rover, dev | everything except `release` and `docs` |

For example, `pip install "rover-photogrammetry[pipeline,stations]"` installs the pipeline and the station products.

## Models downloaded on first use

- The mask model (`rpp_mask_v5`, 125 MB) comes from Hugging Face,
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
