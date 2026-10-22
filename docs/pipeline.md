# Pipeline

The pipeline is `mppp.pipeline`; the `mppp` command (`process`, `align`, `run`) calls it. Every setting of a run is
a field of [`Settings`](api.md#mppp.pipeline.Settings), whose defaults are the production values. Settings come
from, later ones winning: the site's `settings` in the site definitions (`mppp/data/sites.json`),
`<WORK>/mppp_settings.json`, a `--settings` file, and `--set NAME=VALUE` overrides.

A site's WORK folder is `<MPPP_SCAPES>/mars2020_sol_<first sol>_<site>[_zcam]`. It holds `processed/` (images,
masks, manifest) and one COLMAP project per variant (`colmap/`, `colmap_<variant>/`).

```text
process_site  ->  select products  ->  process images (radiometry, masks, cameras, pose priors)
align         ->  manifest -> project -> features -> database -> match -> reconstruct -> health -> export
run_site      =   process_site, then align
```

`mppp align --stop-after STEP` (or `align(..., stop_after=STEP)`) stops after one of these steps.

## 1. Select the PDS products

`select_products` finds, under `MPPP_PDS`, the Navcam RAD products (`NLF`, `NRF`) of the site's sol ranges and, with
Mastcam-Z, the `ZL0` / `ZR0` products of each of the site's zooms (34 mm unless the site lists others). By default it
also adds every waypoint station within 10 m of a station of the sol range, from any sol
(`add_nearby_waypoints`). One product is kept per exposure (the largest), Navcam tiles and sub-frames covering less
than a third of the detector are dropped, and only `NCAM` sequences are kept (plus `ZCAM` with Mastcam-Z). The
mission waypoint table is downloaded and cached for a day; without a connection the packaged snapshot is used.

With `hazcams` (0.93; `--hazcam front|rear|both`, or `--set HAZCAMS=F`), the front (`FLF` / `FRF`, `FHAZ` sequences)
and / or rear (`RLF` / `RRF`, `RHAZ` sequences) Hazcam products of the same stations are added, under the same
one-product-per-exposure and one-third-of-the-frame rules. The run goes to `<WORK>/colmap_hazcam` unless a variant
is named, so the Navcam (+ Mastcam-Z) result in `colmap/` stays; a run without `hazcams` leaves out any Hazcam images
in `processed/`. The Hazcam settings enter the run key only when `hazcams` is set: the keys of every other run are
unchanged.

## 2. Process the images

`process_site` writes `<WORK>/processed/`. For each product:

- **Rules before image work**: Navcam frames pointing more than 20° above the horizon, images outside 06:00-18:00
  local mean solar time, images with more than 20 % saturated valid pixels, Navcam exposures longer than 40 ms, and
  frames whose centre is bluer than the edge by more than a B/R ratio of 1.2 are not processed.
- **Radiometry**: radiance from the label scaling, normalised to an overhead Sun at a reference optical depth;
  written as 16-bit linear PNG (and 8-bit previews). The PDS file name is kept.
- **Masks**: the terrain mask from the released ConvNeXt-tiny model (`mppp_mask_v3`), stored in the PNG alpha
  channel. The model masks rover hardware and rover shadows; the sky is not masked.
- **Cameras**: the promoted models of `mppp.calib` (Navcam, Mastcam-Z 34 mm) or the label CAHVOR start models;
  every image is padded to the full detector frame so that one camera serves all sub-frames of an eye.
- **Pose priors**: the label pose (CAHV) placed in East-North-Up metres of the landing-site frame through the
  waypoint table.

Outputs: `images_png16/`, `references.txt` for Metashape, `colmap/sparse_prior/` (cameras, poses, rig),
`mppp_manifest_<version>.json` and `mppp_config_<version>.json`. Images already processed with the same
configuration are reused. Details: [Methods](methods.md), sections 1-7.

## 3. Project, features and matching

`build_project` creates the COLMAP project: one camera per Navcam eye, one per Mastcam-Z eye and focus bin (30 focus
counts by default), and stereo rigs (Navcam left is the rig reference).

`extract_and_match` extracts SIFT features (up to 32768 per image, at native resolution) and matches them:

| `match_mode` | matcher |
|---|---|
| `exhaustive` (default) | COLMAP's SIFT matcher over all image pairs (ratio test 0.9, cross check), in blocks of 100 images |
| `prior_pairs` | the same matcher over the pairs that the prior poses predict to overlap (`prior_overlap_pairs`) |
| `lightglue` | SIFT features matched by LightGlue (kornia); optional, best on a GPU |

Above `max_exhaustive_images` (300) images, `exhaustive` matches the prior-overlap pairs instead: exhaustive
matching grows as n²/2 pairs (3.7 million at 2713 images). A pair is kept when at least 5 % of either image's
ground footprint (its valid pixels on a 32 × 24 grid, cast onto flat ground 1.95 m below the camera, at most 40 m
out) falls inside the other image, so a narrow zoom inside a wide Navcam frame counts as nearly 100 %; stereo
partners are always kept. The pairs only choose which images are compared: the matcher still compares all of the
features of both images. Geometric verification runs in full-resolution pixels
(6 px threshold). Features and matches are reused while the images and matching settings are unchanged.

COLMAP's matcher runs in a child process: the CUDA environment named by `MPPP_GPU_PYTHON` when set, otherwise the
current Python on the CPU.

Before any of this, the images of the shipped exclusion list `src/mppp/data/exclude_images.txt` are left out
(Mastcam-Z movies, ZCAM01* sequences, a few Navcam stereo pairs that do not work for photogrammetry);
`EXCLUDE_LIST=false` keeps them. A WORK folder's own `exclude_images.txt` and the site's `EXCLUDE` setting add to it.

## 4. Staged bundle adjustment

`reconstruct_project` triangulates the verified matches from the rover poses and runs a weighted bundle adjustment
built on COLMAP's cost functions with pyceres:

- every observation has a standard deviation of 0.5 px in its own native pixels; Cauchy loss;
- each frame has a waypoint position prior (1 m per axis) and a weak attitude prior (2° per axis);
- the stereo baseline is held, which with the position priors fixes the scale;
- rounds of (triangulate, adjust, drop observations above a residual limit) follow `schedule`; frames that do not
  align with the majority are excluded and the block is adjusted again.

With `staged` (default), the adjustment runs in two stages:

1. **Navcam**: the Navcam block alone. Cameras are held in round 1 and refined from round 2, with the intrinsics held
   where the network is weak (`navcam_intrinsics="auto"`). At the end of stage 1 the Navcam cameras are fitted per
   camera-temperature bin.
2. **Mastcam-Z**: the Mastcam-Z frames join with the Navcam cameras and rig held. The Mastcam-Z focus-bin cameras are
   put on lines in focus count per eye, the Mastcam-Z rig rotation is refined, and pose-guided matching adds tie
   points between images whose poses are known. The Hazcam frames (0.93) join here too: their cameras (each eye
   from the median of its images' label CAHVORE fits) are held in the first round and then refine fx, fy, k1, k2
   (`hazcam_intrinsics`; an eye with fewer than 1000 tie-point observations stays at its start); their position
   priors follow their station's Navcam solution to 0.1 m (the cameras are fixed to the rover body); their stereo
   rigs keep the CAHV values (`hazcam_rig_refine = "hold"`, or "rotation"). The
   log reports, per round, the Hazcam frames with tie points and the Navcam-Hazcam tie points
   (`settings["reconstruction"]["hazcam"]`).

The Navcam-only stage is exported as well (`error_input_navcam/`).

## 5. Health checks

`health` evaluates the refined block in groups: registration, tie points (track lengths, cross-station fraction,
tied station blocks), reprojection (residuals per camera, left/right balance, growth towards the image edge, corner
coverage), camera-model changes, the stereo rig, and pose shifts against the priors. Each value has a warn and a fail
threshold; the verdict is the worst status. The thresholds are provisional. A `fail` verdict means the alignment
should be fixed before any error analysis. The report is written to `<project>/health/` (`health.json`,
`health.md`, `health.png`); with Mastcam-Z focus bins, the focus-breathing figure is added.

## 6. Exports

`export_for_error` writes `<project>/error_input/`:

| file | content |
|---|---|
| `native/` | COLMAP text model in each image's native pixels |
| `stations.csv` | image to station (site/drive), sol, SCLK, LMST, solar elevation, instrument, scale |
| `poses.csv` | prior versus refined camera centre (ENU, m) and attitude difference (deg) |
| `residuals.npz` | per-observation residuals in native pixels |
| `summary.json` | cameras and rig before and after, track statistics, epsilon estimates |
| `camera_shifts.png` | the camera-shift figure |

`run_done.json` records the run key: a hash of the settings, the camera models in use and the alignment rules. A
WORK folder whose `run_done.json` has the same key is skipped by the batch scripts; promoting a camera model changes
the key and realigns every block.

## Batch runs

`scripts/run_sites.py --group navcam_consensus` and `scripts/process_sites.py --all` run site after site with logs
and status files. On Windows the same runs are double-click batch files; see
[Windows batch files](WINDOWS.md).
