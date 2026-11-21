# What is validated and what is experimental

RPP 1.0 is a validated pipeline with an experimental viewer. This page says which parts you can rely on, how each
was checked, and which parts are still research.

!!! note "Release candidate"
    This page describes 1.0.0rc1. The camera models of 1.0.0 come from the third freeze run (November 2026), which
    aligns every Mars 2020 block again with one version of the code and the models. The numbers marked *freeze 3*
    below are filled in when that run is reviewed.

## Validated

These parts have tests, have run on every block of the freeze runs, and have independent checks.

| Part | What it does | How it was checked |
|---|---|---|
| PDS import (`rpp process`) | Reads Mars 2020 Navcam, Hazcam and Mastcam-Z radiance products; writes 16-bit linear images that keep their PDS names | Pixel values are traceable to the PDS radiance; regression fixtures for a Navcam and a Mastcam-Z product |
| Rover masks (`rpp.mask`) | Marks rover hardware and rover shadows | 90 hand-cleaned frames that no model trained on: exclusion IoU 0.974 for the default model `rpp_mask_v5` ([Rover masks](masks.md)) |
| Camera models (`rpp.calib`) | Navcam, and Mastcam-Z at 34 and 110 mm, as functions of temperature, sol, focus and backlash state | Fitted jointly over many blocks, then tested against data the fit did not use: the surveyed rover-deck fiducials, the Mastcam-Z stereo pairs alone, and the azimuth and elevation resolvers ([Camera models and calibration](calibration.md)) |
| Station positions | Each station placed from the rover's localization, in meters, east-north-up | Every block's health report compares its scale and orientation with these priors |
| Alignment (`rpp align`) | COLMAP matching and RPP's weighted bundle adjustment | A health report for every block: reprojection error, registered share, scale and orientation against the priors |
| Exports | COLMAP, Agisoft Metashape, Nerfstudio | Tests of the written cameras, poses and images |

**Freeze 3 so far** (six of twenty blocks, 20 November 2026): every block aligned with a warning-level health report
and none failed. The root-mean-square reprojection error is 0.40 to 0.70 pixels at full resolution, and 95 % or more
of each block's images registered, except at Seitah Overlook (89 %).

The independent checks are kept in a ledger, `src/rpp_studies/validation/ledger.json`: one entry per claim of the
camera and rig models, with the test, the value, its standard error and whether it agrees. For example, the Navcam
stereo baseline of the model is 423.53 mm at -20 °C, and a length surveyed on the rover deck gives
423.66 ± 0.07 mm; the ledger lists the two as consistent.

## First models

These parts work and ship with 1.0, but rest on fewer data. Use their results with their stated uncertainty.

| Part | State |
|---|---|
| Mastcam-Z 63 mm | Close to the 34 and 110 mm models; fewer blocks |
| Mastcam-Z 26, 48, 79 and 100 mm | First models, from one to a few blocks each. The 26 and 100 mm models come from one block each |
| Front and rear Hazcam | First models of the four eyes and both rigs |
| Alignment-health thresholds | Provisional. A "warn" verdict asks you to look at the report; it does not mean that the block is wrong |

## Experimental

These parts are useful, change quickly, and have not been validated. Their file formats and options may change in a
1.x release.

| Part | State |
|---|---|
| Station products (`rpp mesh`, `depthmaps`, `ods`, `single-sites`) | Terrain mesh, depth and normal maps, stereo panoramas and 3D station models of one rover station |
| RPV, the Rover Photogrammetry Viewer | The stereo panoramas are the most tested view. The 3D terrain modes are newer. The rover drive is a simulation, shown only with `?experimental=1` |
| Color views | "Enhanced color", "enhanced tone" and the decorrelation stretch are display transforms of the 16-bit linear panoramas, not calibrated color |
| Rover model in the station products | One pose (arm stowed); the suspension is not fitted to each station |
| Curiosity (MSL) | Navcam and front Hazcam blocks align (Yellowknife Bay). A research study in the source repository, not part of the installed package |
| Reconstruction-error analysis (`rpp_studies.error`) | Under development |
| Mask training (`rpp_studies.mask_training`) | Works; written for the author's data layout |

## What RPP does not do

- It does not make calibrated color (I/F) for the engineering cameras; their colors are matched to Mastcam-Z.
- It does not correct for the time of day. Images of one place taken at different times keep their shadows.
- It does not read Mars 2020 SuperCam, SHERLOC, WATSON or PIXL images, or rovers other than Perseverance and
  Curiosity.
- It does not replace the mission's localization. It starts from it.

## Version policy

From 1.0.0 the command line, the `rpp.pipeline` functions, the camera-model file format and the names RPP writes
into a work folder are stable within the 1.x series. Experimental parts may change in a minor release; the
[changelog](https://github.com/ctate7163/rpp/blob/main/CHANGELOG.md) says when they do.
