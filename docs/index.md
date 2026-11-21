# RPP: Rover Photogrammetry Pipeline

**Mars rover images to true-scale 3D, across many rover stops.**

RPP turns the public Mars rover images of NASA's Planetary Data System (PDS) into aligned structure-from-motion
projects that combine the images of many rover stops. The result is in meters, with north and up known, and opens in
COLMAP, Agisoft Metashape and Nerfstudio.

Every time a rover stops, it sees the terrain from a new viewpoint. The mission's stereo products use one stop at a
time, with a fixed baseline between the two eyes of a camera. RPP lets the images from nearby stops triangulate
together. A **block** is a set of rover stations within a few tens of meters of each other.

[Install RPP](installation.md){ .md-button .md-button--primary }
[Run the example](quickstart.md){ .md-button }
[Open the viewer](https://ctate7163.github.io/rpv/){ .md-button target="_blank" rel="noopener" }

## What it does

1. **Images.** PDS radiance products (`*.IMG`) become 16-bit linear images that keep their PDS names, with the colors
   of the cameras brought into one color space.
2. **Masks.** A neural network marks the rover hardware and the rover's shadow in every image. The sky is not
   masked. See [Rover masks](masks.md).
3. **Cameras and stations.** RPP applies its own in-flight camera models, which vary with temperature, focus, zoom
   and sol, or converts the mission's CAHV, CAHVOR and CAHVORE models. Each station is placed from the rover's
   localization in a local east-north-up frame. See [Camera models and calibration](calibration.md).
4. **Alignment.** COLMAP features and matching, then RPP's weighted bundle adjustment with position and attitude
   priors and the stereo rigs. See [Pipeline](pipeline.md).
5. **Health report and exports.** A report says how well the block aligned. See [Downstream exports](exports.md).

From an aligned block, RPP also makes the products of one rover station: a terrain mesh, depth and normal maps for
every image, a stereo 360° panorama and a 3D station model. The
[Rover Photogrammetry Viewer](viewer.md) (RPV) shows them in a browser or a VR headset.

## Who it is for

- **Scientists** who want to see rocks in their context, measure on a model with a known scale, or start their own
  reconstruction from clean inputs.
- **Photogrammetry and 3D graphics users** who want Mars rover images in a form that COLMAP, Metashape and
  Nerfstudio can read, with masks, cameras and poses.
- **Mission teams** who want in-flight camera models that are fitted across the mission and checked independently.

## Where to start

| If you want to | Read |
|---|---|
| install RPP | [Installation](installation.md) |
| run a first block | [Quick start](quickstart.md) |
| know what you can rely on | [What is validated and what is experimental](status.md) |
| understand each step and its outputs | [Pipeline](pipeline.md) |
| take a block into another program | [Downstream exports](exports.md) |
| make panoramas and 3D station models | [Station products](station-products.md), [Viewer](viewer.md) |
| look up a command | [Command line](cli.md) |
| call RPP from Python | [API reference](api.md) |
| see the equations and conventions | [Methods](methods.md) |
| know what is planned | [Roadmap](roadmap.md) |

Two notebooks in the repository run the pipeline from Python:
[`pipeline_example.ipynb`](https://github.com/ctate7163/rpp/blob/main/notebooks/pipeline_example.ipynb) (the Rockytop
example) and
[`van_zyl_ods_example.ipynb`](https://github.com/ctate7163/rpp/blob/main/notebooks/van_zyl_ods_example.ipynb) (one
station from the PDS download to the viewer).

## Supported cameras

| Rover | Camera | Status in 1.0 |
|---|---|---|
| Mars 2020 Perseverance | Navcam (left, right) | calibrated in flight: temperature, sol, stereo rig |
| Mars 2020 Perseverance | Mastcam-Z 34 mm and 110 mm | calibrated in flight: focus, focus-motor backlash state |
| Mars 2020 Perseverance | Mastcam-Z 26, 48, 63, 79, 100 mm | first models, from one to a few blocks each |
| Mars 2020 Perseverance | Front and rear Hazcam | first models |
| MSL Curiosity | Navcam, front Hazcam | experimental (a research study in the source repository) |

`rpp cmods list` prints the camera models in use.

## License, credits and citation

RPP is released under the Apache-2.0 license. Source code:
[github.com/ctate7163/rpp](https://github.com/ctate7163/rpp). To cite RPP, see [Citing](citing.md).

All images RPP uses are public PDS products. Image credits: Navcam and Hazcam NASA/JPL-Caltech; Mastcam-Z
NASA/JPL-Caltech/ASU/MSSS. The rover model is derived from NASA's Perseverance model (credit NASA/JPL-Caltech; see
[Rover model](rover-model.md)).

Until version 0.98 the package was called MPPP (Mars Photogrammetry Preprocessing Pipeline). Work folders made by
MPPP are read as they are; `rpp.ondisk` lists the file names that keep the old spelling.
