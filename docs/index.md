# MPPP

MPPP (Mars Photogrammetry Preprocessing Pipeline) turns Mars 2020 Perseverance **Navcam** and **Mastcam-Z** PDS
products (`*.IMG`, RAD) into aligned structure-from-motion blocks in **COLMAP**, and into inputs for **Agisoft
Metashape**.

The Mars 2020 images and their metadata are public in NASA's Planetary Data System (PDS), in a format that common 3D
reconstruction software cannot read directly. Fixed-baseline stereo gives each rover stop its own range map; MPPP lets
the images from nearby stops triangulate together. A **block** (or **site**) is a set of rover stations within a few
tens of metres of each other.

## What it does

Five processes take a block from PDS inputs to SfM outputs:

1. **Images**: radiometrically linear 16-bit images (and 8-bit previews) that keep their PDS file names and are
   traceable to the PDS radiance.
2. **Masks**: rover hardware and rover shadows (the sky is not masked), from a fine-tuned ConvNeXt-tiny segmentation
   model downloaded on first use.
3. **Cameras and stations**: the label CAHV/CAHVOR/CAHVORE models converted to COLMAP cameras, or the promoted
   calibrations of `mppp.calib`; each station placed in a local East-North-Up frame (metres) from the rover
   waypoints.
4. **Alignment**: a COLMAP project with stereo rigs, SIFT features, matching, and a prior-weighted bundle adjustment
   on pyceres.
5. **Health and exports**: an alignment health report and the measurements for the reconstruction error analysis.

A finished block also goes on to COLMAP's dense reconstruction, an Agisoft Metashape project and Nerfstudio's
Gaussian splatting (see [Downstream exports](exports.md)).

From a finished block MPPP also makes the products of one rover station: a cleaned mesh with a model of the rover,
per-image depth and photometric maps, and a stereoscopic 360 degree panorama for a headset (see
[Station products and the stereo panorama viewer](station-products.md)). The panoramas can be looked at on the
[stereo panorama page](ods/index.html) (prototype).

The alignments also measure the cameras. MPPP versions the refined camera models, checks them across blocks and uses
them as the default cameras for the next run (see [Camera models and calibration](calibration.md)).

## Where to start

- [Installation](installation.md): the package, its optional dependency groups and the environment variables.
- [Quick start](quickstart.md): fetch the Rockytop example data and run one block.
- [Pipeline](pipeline.md): the stages and their outputs.
- [Downstream exports](exports.md): dense reconstruction, Metashape, Nerfstudio.
- [Station products and viewer](station-products.md): the station mesh, depth and photometric maps, the stereo
  panorama and the web viewer.
- [Rover model](rover-model.md): the Perseverance model used in the station mesh, its source and its changes.
- [Rover masks](masks.md): the mask models, training a new one, the Hugging Face releases.
- [Command line](cli.md): every `mppp` subcommand and its options.
- [API reference](api.md): the importable functions (`mppp.pipeline.Settings`, `process_site`, `align`, `run_site`).
- [Methods](methods.md): conventions, equations and flagged assumptions.
- [Roadmap](roadmap.md): the freeze run, the panoramas and what follows.

## Scope

Models shipped with the package (`mppp cmods list`): Navcam (a temperature-dependent tied joint, with a stereo rig
that drifts with sol and has a knee at sol 250 in pitch), front and rear Hazcam, and Mastcam-Z at 34, 63 and 110 mm.
The Mastcam-Z joints at 34 to 110 mm and the rigs are being calibrated again in the current freeze run; its models
are not in the package yet (see [Camera models and calibration](calibration.md)). Scale comes from the Navcam stereo
baseline together with the waypoint position priors. The alignment-health thresholds are provisional.

## License and data

MPPP is released under the Apache-2.0 license. All Mars 2020 images used by MPPP are public PDS products (Navcam:
NASA/JPL-Caltech; Mastcam-Z: NASA/JPL-Caltech/ASU/MSSS). The rover model is derived from NASA's Perseverance model
(credit NASA/JPL-Caltech; see [Rover model](rover-model.md)). Source code:
[github.com/ctate7163/MPPP](https://github.com/ctate7163/MPPP). To cite MPPP, see [Citing](citing.md).

