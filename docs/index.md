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

The alignments also measure the cameras. MPPP versions the refined camera models, checks them across blocks and uses
them as the default cameras for the next run (see [Camera models and calibration](calibration.md)).

## Where to start

- [Installation](installation.md): the package, its optional dependency groups and the environment variables.
- [Quick start](quickstart.md): fetch the Rockytop example data and run one block.
- [Pipeline](pipeline.md): the stages and their outputs.
- [Command line](cli.md): every `mppp` subcommand and its options.
- [API reference](api.md): the importable functions (`mppp.pipeline.Settings`, `process_site`, `align`, `run_site`).
- [Methods](methods.md): conventions, equations and flagged assumptions.

## Scope

Calibrated models exist for Navcam and Mastcam-Z at 34 mm. The Hazcams and the other Mastcam-Z zooms are processed
with start models and are not part of the tested scope. Scale comes from the Navcam stereo baseline together with the
waypoint position priors. The alignment-health thresholds are provisional.

## License and data

MPPP is released under the Apache-2.0 license. All Mars 2020 images used by MPPP are public PDS products (Navcam:
NASA/JPL-Caltech; Mastcam-Z: NASA/JPL-Caltech/ASU/MSSS). Source code:
[github.com/ctate7163/MPPP](https://github.com/ctate7163/MPPP). To cite MPPP, see [Citing](citing.md).
