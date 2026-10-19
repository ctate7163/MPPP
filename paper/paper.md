---
title: 'MPPP: beyond the stereo baseline with Mars rover images'
tags:
  - Python
  - planetary science
  - Mars rover
  - photogrammetry
  - structure from motion (SfM)
  - convolutional neural networks (CNN)
authors:
  - name: Christian D. Tate
    orcid: 0000-0003-3467-0099
    affiliation: "1"
affiliations:
  - name: Malin Space Science Systems, San Diego, CA, USA
    index: 1
date: 01 November 2026
bibliography: paper.bib
---

# Summary

Every time a Mars rover stops, it looks around from a new viewpoint. The Mars 2020 Perseverance rover [@Farley:2020] captures the Martian landscape in stereo, often in 360° panoramas [@Bell:2021; @Maki:2020]. Although these images and their metadata are publicly available through NASA's Planetary Data System (PDS), they are in a PDS format that is not easily used in common 3D reconstruction software. If properly preprocessed, however, these image sets can fuse into 3D models using structure from motion (SfM) tools such as COLMAP [@Schoenberger:2016] or Agisoft Metashape [@Agisoft:2024]. MPPP, the Mars Photogrammetry Preprocessing Pipeline, fills this gap with a PDS-to-SfM pipeline.

# Statement of need

The target audience for MPPP is the Mastcam-Z Stereo-DTM working group and other researchers who model Mars' 3D terrain. Fixed-baseline stereo gives each rover stop its own range map; MPPP lets images from nearby rover stops triangulate to better see the terrain.

The same alignments are measurements of the cameras. Each block solution refines the focal lengths, principal points, distortion terms and stereo rig of the cameras that imaged it, with an image-measurement precision near 0.2 px, so a collection of blocks across the mission becomes a calibration data set that no single sol can provide. MPPP uses this feature in both directions: block solutions constrain camera model forms and parameters and measure how they vary with temperature, focus and time since landing. The refined models are then versioned, checked across blocks, and fed back as the default cameras for the next run. A follow-on study will use this machinery to tie a Mastcam-Z calibration to a refined Navcam calibration and to measure how camera-model error propagates into 3D reconstruction error.

At the other end of the pipeline, an aligned block is the foundation for 3D models made for exploration as well as measurement. MPPP emphasizes realism and data provenance: colors come from radiometrically calibrated products under one convention, scale comes from the rover's own localization, and every model traces back to the PDS product identifiers and the code version that made it. The result is useful alike to scientists who want to see the rocks in context and to space enthusiasts who want to see Mars. The Jezero crater landscape models published on Sketchfab [@Tate:Sketchfab] were preprocessed with MPPP; the models before 2026 were aligned in Agisoft Metashape, whereas the current pipeline aligns in COLMAP 4.

# State of the field

Several tools process rover data into depth maps, point clouds or 3D models: VICAR [@VICAR], PRoViP [@Barnes:2018] and the Ames Stereo Pipeline [@Beyer:2018]. They primarily reconstruct from fixed-baseline stereo pairs. Although PRoViP can fuse stereo products from several rover positions and the Ames Stereo Pipeline can bundle-adjust multiple images, neither offers an open path from the archive to SfM. The nearest relative is the Metashape workflow of @Schmauss:2025 for Curiosity's Mastcam and Navcam images, which embeds the camera metadata in image tags and builds a digital outcrop model of Mont Mercou; MPPP is the Mars 2020 counterpart, with the masking, camera-model translation and prior-weighted bundle adjustment done in code and the alignment in COLMAP.

SfM requires numerous overlapping images for a robust 3D reconstruction, and it often fails with rover images because their image sets are sparse. Fortunately, some of Perseverance's image sequences are amenable to SfM because they come from neighboring rover locations less than 10 meters apart. We call such a set of complementary rover locations a block. Several dozen blocks of four or more end-of-drive panoramas have accumulated over the first 2,000 sols, and the rover is parked within 10 m of three or more earlier or later stops on about 20 % of sols. These blocks fuse into intricate and expansive 3D models of the Martian terrain.

# Pipeline

MPPP runs as one resumable command (`mppp process`, `mppp align`) over a block or a batch of blocks, and the same functions are importable from Python (`mppp.pipeline`). Its bundle adjustment carries waypoint position and attitude priors, resolution-aware weighting and stereo rigs, starts from the label CAHV cameras or from versioned refined models, and takes its color processing from a configuration file. Five processes take Perseverance rover images from PDS inputs to SfM outputs:

1. loading radiance-calibrated images in PDS format and harmonizing their colors between instruments and their brightness between times of day;
2. masking rover hardware in the frame and rover shadows cast on the terrain;
3. converting CAHV, CAHVOR and CAHVORE camera models to COLMAP camera models and placing each station in a local east-north-up frame from the rover waypoints, which, together with the stereo baseline, fixes the scale in meters;
4. assembling the COLMAP project and running feature matching and prior-weighted bundle adjustment;
5. checking alignment health and exporting the measurements for 3D reconstruction error analysis (`mppp.error`, a library under development that is not yet part of the advertised workflow).

MPPP loads PDS data and metadata with the Planetary Data Reader [@Brown:2024]. Color processing applies a static RGB rescaling so that Navcam, Hazcam, Mastcam-Z and WATSON images roughly match; the current release supports Navcam and Mastcam-Z at the 34 mm focal length with calibrated camera models, and processes the Hazcams and the other Mastcam-Z zooms with start models derived from the labels. Time-of-day and weather-related changes in illumination are modeled with zenith scaling and an approximate atmospheric opacity correction from a seasonal optical-depth table [@TODO:opacity].

Rover masking is necessary because the rover body is the most reliable feature in every frame: its features are static from stop to stop, so a matcher ties images together on the rover rather than on the terrain. To correct this, we fine-tune a ConvNeXt-tiny convolutional neural network (CNN) [@Liu:2022; @Wightman:2019; @Paszke:2019] on thousands of rover image-mask pairs and use its inference to exclude those image regions from the SfM process. The training set comes from several Mars 2020 instruments on a variety of terrain types throughout the first 2,000 sols of the mission. We mask rover shadows on the terrain at the same time as rover hardware so that imagery from different times of day will more easily align. Figure 1 shows several image-mask pairs withheld during training. The weights are hosted on Hugging Face (https://huggingface.co/ctate7163/mppp-mask), and the training code is in `mppp.mask.training`.

Camera locations are necessary for scale and relative placement. The initial camera positions come from the JPL-supplied waypoints for each rover location [@M2020:waypoints]. These are in meters, with XYZ coordinates aligned with east, north and up, respectively, and recentered for each block. The per-image camera rotations and intrinsic parameters come from the PDS metadata, converted to COLMAP camera models (`THIN_PRISM_FISHEYE` for the Navcam and Hazcams; the OpenCV model with the tangential and the sixth-order radial terms held at zero for Mastcam-Z). Each stereo pair is a rig, so that the two cameras share one relative translation and rotation for every pair of that instrument in the block. Besides loading cameras from the PDS metadata, `mppp.calib` defines one frozen model form per camera family and instantiates the versioned refined models per image from the temperature, focus, focus-motor backlash state and sol in the label; these are the default start for a run. The Navcam form is a fisheye model whose focal lengths, principal point and first radial term are linear in camera temperature and whose stereo-rig rotation drifts linearly with sol; the Mastcam-Z form is a per-zoom model that depends only on focus count and backlash state.

MPPP extracts SIFT features [@Lowe:2004] with COLMAP through pycolmap, matches them with LightGlue [@Lindenberger:2023] by default (COLMAP's nearest-neighbour matcher remains available), verifies the matches geometrically in COLMAP, and runs its own bundle adjustment on pyceres [@Agarwal:2023], where the waypoint priors and the resolution-aware weights enter. A few matching and triangulation parameters are set to suit sparse rover image sets.

# Development roadmap

Future work could develop the following features.

1. Calibrated camera models for the body-mounted Hazcams, the arm-mounted WATSON camera [@Bhartia:2021] and Mastcam-Z's other zoom positions (the current release carries only start models for these).
2. Automatic download of the data through the PDS API instead of requiring local PDS products.
3. Extension from SfM to automatic multi-view stereo [@Schoenberger:2016b], a 3D triangle mesh and a blended texture image.
4. Learned feature detectors and descriptors that outperform SIFT on variable illumination and large convergence angles.
5. Conversion of SfM results for Nerfstudio [@Tancik:2023], to train neural radiance fields, 3D Gaussian splats [@Kerbl:2023] and other emerging 3D representations.
6. Support for MSL Curiosity rover images.


# Figures

![Rover hardware and shadow masks predicted by the fine-tuned ConvNeXt-tiny model on frames withheld from training.](figure1.png)

![A Navcam frame as recorded through the fisheye lens (left) and resampled through its refined camera model (right).](figure2.png)

![An aligned block in plan view: stations from telemetry and after adjustment, with the sparse point cloud.](figure3.png)

# Acknowledgements

We acknowledge contributions from the Mastcam-Z team at Arizona State University.

# References
