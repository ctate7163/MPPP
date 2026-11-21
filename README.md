# RPP: Rover Photogrammetry Pipeline

[![tests](https://github.com/ctate7163/rpp/actions/workflows/tests.yml/badge.svg)](https://github.com/ctate7163/rpp/actions/workflows/tests.yml)
[![docs](https://github.com/ctate7163/rpp/actions/workflows/docs.yml/badge.svg)](https://ctate7163.github.io/rpp/)
[![PyPI](https://img.shields.io/pypi/v/rover-photogrammetry.svg)](https://pypi.org/project/rover-photogrammetry/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/ctate7163/rpp/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24408630.svg)](https://doi.org/10.5281/zenodo.24408630)

**Mars rover images to true-scale 3D, across many rover stops.**

Every time a Mars rover stops, it sees the terrain from a new viewpoint. The mission's own stereo products use one
stop at a time. RPP combines the images of nearby stops: it turns the public images of NASA's Planetary Data System
(PDS) into an aligned structure-from-motion project, in meters, with north and up known.

- **Documentation:** <https://ctate7163.github.io/rpp/>
- **See the results:** the Rover Photogrammetry Viewer (RPV), <https://ctate7163.github.io/rpv/>, in a browser or a
  VR headset
- **Rover mask model:** <https://huggingface.co/ctate7163/rpp-mask>
- **Cite:** RPP 1.0.0rc1, [doi:10.5281/zenodo.24408630](https://doi.org/10.5281/zenodo.24408630) (Zenodo; see [Citing RPP](https://ctate7163.github.io/rpp/citing/))

## What RPP does

RPP takes a **block**, a set of rover stations within a few tens of meters of each other, through five steps:

1. **Images.** PDS radiance products (`*.IMG`) become 16-bit linear images that keep their PDS names, with the
   colors of the cameras brought into one color space.
2. **Masks.** A neural network marks the rover hardware and the rover's shadow in every image, so that the matcher
   ties images together on the terrain. The sky is not masked.
3. **Cameras and stations.** RPP applies its own in-flight camera models, which vary with temperature, focus, zoom
   and sol, or converts the mission's CAHV, CAHVOR and CAHVORE models. Each station is placed from the rover's
   localization in a local east-north-up frame.
4. **Alignment.** COLMAP features and matching, then RPP's weighted bundle adjustment with position and attitude
   priors and the stereo rigs.
5. **Health report and exports.** A report says how well the block aligned. The block opens in COLMAP, Agisoft
   Metashape and Nerfstudio.

From an aligned block, RPP can also make **station products**: a terrain mesh around one rover station, depth and
normal maps for every image, a stereo 360° panorama and a 3D station model. RPV shows them.

## Supported cameras

| Rover | Camera | Status in 1.0 |
|---|---|---|
| Mars 2020 Perseverance | Navcam (left, right) | calibrated in flight: temperature, sol, stereo rig |
| Mars 2020 Perseverance | Mastcam-Z 34 mm and 110 mm | calibrated in flight: focus, focus-motor backlash state |
| Mars 2020 Perseverance | Mastcam-Z 26, 48, 63, 79, 100 mm | first models, from one to a few blocks each |
| Mars 2020 Perseverance | Front and rear Hazcam | first models |
| MSL Curiosity | Navcam, front Hazcam | experimental (a research study in the source repository) |

[What is validated and what is experimental](https://ctate7163.github.io/rpp/status/) has the details and the
numbers.

## Install

RPP needs Python 3.10 or later. Install it into a conda environment of its own (Miniforge, Miniconda or Anaconda).
The commands are the same in a Windows Anaconda Prompt, PowerShell, and a Linux or macOS shell.

```bash
conda create -n rpp -c conda-forge python=3.12 -y
conda activate rpp
pip install "rover-photogrammetry[pipeline]"
rpp doctor
```

The distribution is `rover-photogrammetry`; the import name and the command are `rpp`. `rpp doctor` reports what is
installed and what is missing.

To work on the code, or to run the notebooks and the research studies, install from the repository instead:

```bash
conda create -n rpp -c conda-forge python=3.12 git -y
conda activate rpp
git clone --depth 1 https://github.com/ctate7163/rpp.git
cd rpp
pip install -e ".[all]"
rpp doctor
```

Notes:

- **GPU.** Install the CUDA build of PyTorch first, with the command <https://pytorch.org> gives for your system.
  pip then keeps it. RPP also runs on a CPU; masking and matching are slower.
- **Extras.** `[pipeline]` is what `rpp run` needs (masks, alignment, plots). `[stations]` adds the station products
  and the viewer. `[all]` adds the notebooks, the tests and what the studies need.
- **No data comes with the code.** RPP downloads PDS products, the mask model (125 MB, from Hugging Face) and the
  matcher weights when it first needs them.
- The PyPI package named `rpp` is a different project that uses the same import name. Do not install both.

[Installation](https://ctate7163.github.io/rpp/installation/) covers the environment variables, offline use and the
GPU setup on Windows.

## First run

This downloads a small Mars 2020 example (Rockytop, sols 461-464: 36 PDS products) and aligns it.

```bash
rpp example fetch
rpp run rockytop_example --zcam
```

The results are in `~/rpp_scapes/`. The block folder holds the processed images, the COLMAP project
(`colmap/sparse/cahv_ba/`), the health report (`colmap/health/`) and files that open the block in the COLMAP GUI.

Next steps:

- [Quick start](https://ctate7163.github.io/rpp/quickstart/): your own sols and sites, and what each output is.
- [`notebooks/pipeline_example.ipynb`](https://github.com/ctate7163/rpp/blob/main/notebooks/pipeline_example.ipynb):
  the same run from Python, with figures.
- [`notebooks/van_zyl_ods_example.ipynb`](https://github.com/ctate7163/rpp/blob/main/notebooks/van_zyl_ods_example.ipynb):
  one rover station from the PDS download to a stereo panorama, a 3D model and a local copy of the viewer.

## The repository

| Path | Content |
|---|---|
| `src/rpp/` | the installed package: the pipeline, the camera models, the station products, the viewer page |
| `src/rpp_studies/` | research studies that use the pipeline: camera calibration, reconstruction error, mask training, site maps, Curiosity. Not in the PyPI package |
| `notebooks/` | two examples of the pipeline, and one notebook per study |
| `scripts/` | batch runs over many sites, and Windows batch files |
| `docs/` | the documentation site |
| `paper/` | the software paper |
| `tests/` | the test suite (`pytest -q`) |

## Roadmap

1.0 is the validated pipeline with an experimental viewer. The
[roadmap](https://ctate7163.github.io/rpp/roadmap/) lists what comes next: measurement and source look-up in the
viewer, downloads with a DOI, more Curiosity support, and a catalog of stations along the traverse.

## Contributing and help

Questions, bug reports and requests are welcome in the
[issue tracker](https://github.com/ctate7163/rpp/issues). See
[CONTRIBUTING.md](https://github.com/ctate7163/rpp/blob/main/CONTRIBUTING.md) for how to set up a development
environment and run the tests.

## Citing

If you use RPP, please cite it as [CITATION.cff](https://github.com/ctate7163/rpp/blob/main/CITATION.cff)
describes. GitHub shows the citation under "Cite this repository".

## Credits and license

RPP is written by Christian Tate at Malin Space Science Systems and released as open source, under the
[Apache-2.0 license](https://github.com/ctate7163/rpp/blob/main/LICENSE), with the company's approval.

All images RPP uses are public PDS products. Image credits: Navcam and Hazcam NASA/JPL-Caltech; Mastcam-Z
NASA/JPL-Caltech/ASU/MSSS. The rover model is derived from NASA's "Mars Perseverance Rover, 3D Model"
(NASA/JPL-Caltech). The viewer includes three.js (MIT license). See
[NOTICE](https://github.com/ctate7163/rpp/blob/main/NOTICE).

Until version 0.98 the package was called MPPP (Mars Photogrammetry Preprocessing Pipeline). Work folders made by
MPPP are read as they are.
