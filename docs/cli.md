# Command line

The `mppp` command is installed with the package (`python -m mppp` is the same). The text below is the
output of `--help` for the command and each subcommand. Settings of `process`, `align` and `run` are the
fields of `mppp.pipeline.Settings` (see the [API reference](api.md)); `--set NAME=VALUE` overrides one.

## mppp

```text
usage: mppp [-h] [-v | -q] [--log FILE]
            {process,align,run,sites,gui,cmods,mask,example,waypoints,maps,calib,mesh,depthmaps,ods,dense,metashape,nerfstudio,doctor,version} ...

The ``mppp`` command: the pipeline (:mod:`mppp.pipeline`) and its bookkeeping from the shell.

::

    mppp process SITE|WORK [--root DIR] [--pds DIR] [--zcam] [--hazcam front|rear|both|none] [--set NAME=VALUE ...]
                 [--settings FILE] [--dry-run]
    mppp align WORK [--variant NAME] [--set ...] [--settings FILE] [--gui] [--stop-after STEP] [--force] [--status]
    mppp run SITE|WORK [...]                  process, then align
    mppp sites [list | check | folder SITE]   the site definitions (mppp/data/sites.json)
    mppp cmods [list | promote PATH ... --note TEXT]   the camera models in use
    mppp mask [list | fetch [NAME] | install FILE]      the terrain mask model
    mppp example [fetch [--dest DIR] | build --pds DIR --zcam DIR --out DIR]   the Mars 2020 example data (Rockytop)
    mppp maps [all | zooms | figures 2 3 ...] [--out DIR] [--dpi N] [--background dtm|ortho] [--tag T] [--dems F;F] [--orthos F;F]
    mppp calib navcam [--root DIR] [--group ecam_consensus | --blocks NAME=WORK ...] [--out DIR] [--promote-if-ok]
    mppp calib zcam|nz|hazcam [--zoom Z] [--group G1,G2] [--navcam-group G] [--variant V] [--promote-if-ok]
    mppp mesh WORK --station S003D1234 [--points PLY] [--colmap EXE] [--mesh-radius 30] [--plane-radius 100]   0.96: one clean surface
    mppp depthmaps WORK --station ... [--cameras N] [--scale 1] [--photometry]   0.96: per-image depth / normal maps, the photometric cube
    mppp ods WORK --station ... [--width 8192] [--ipd 0.064]                       0.96: the stereoscopic panorama, two eye images
    mppp doctor                               what is set, what is missing (environment, dependencies, models)
    mppp version

``-v`` / ``-q`` set the logging level, ``--log FILE`` tees the log to a file.  ``process``, ``align`` and ``run``
write ``<WORK>/runs/<time>/log.txt`` and ``<WORK>/mppp_status.json`` as well (:func:`mppp.runner.run_logged`).
Every subcommand returns an exit code (0 ok, 1 failed, 2 bad arguments, 3 a run is already going on).

positional arguments:
  {process,align,run,sites,gui,cmods,mask,example,waypoints,maps,calib,mesh,depthmaps,ods,dense,metashape,nerfstudio,doctor,version}
    process             select the PDS products of a site and process them into <WORK>/processed
    align               align the images processed in a WORK folder
    run                 process, then align a site
    sites               the site definitions
    gui                 write the COLMAP GUI files (open_in_colmap.bat ...) into finished projects
    dense               the COLMAP dense reconstruction of a finished block (0.95.0)
    cmods               the camera models in use
    mask                the terrain mask model
    example             the Mars 2020 example data (Rockytop, sols 461-464): fetch or build
    waypoints           the Mars 2020 waypoint table in use (cached download or packaged snapshot)
    maps                the M20 site-map figures (mppp_studies.sites) from the official blocks of
                        sites.json
    calib               the camera calibrations (0.91: the Navcam tied joint, the Mastcam-Z joint)
    doctor              what is set, what is missing: environment variables, dependencies, models,
                        waypoints
    version             the MPPP version

options:
  -h, --help            show this help message and exit
  -v, --verbose         debug messages
  -q, --quiet           warnings only
  --log FILE            also write the log to this file
```

## mppp process

```text
usage: mppp process [-h] [--set NAME=VALUE] [--settings FILE] [--sites-file FILE] [--root DIR]
                    [--pds DIR] [--zcam] [--hazcam {front,rear,both,none}] [--force] [--no-echo]
                    [--dry-run]
                    SITE|WORK

positional arguments:
  SITE|WORK             a site of the site definitions, or its WORK folder

options:
  -h, --help            show this help message and exit
  --set NAME=VALUE      a pipeline setting (mppp.pipeline.Settings field), VALUE as JSON or
                        Python; repeatable
  --settings FILE       a JSON file of settings {NAME: value}
  --sites-file FILE     site definitions (default mppp/data/sites.json)
  --root DIR            the scapes root (default: MPPP_SCAPES)
  --pds DIR             the PDS archive (default: MPPP_PDS)
  --zcam                the Navcam + Mastcam-Z block of the site (..._zcam)
  --hazcam {front,rear,both,none}
                        the Hazcam frames of the block's stations: front, rear, both (the default
                        since 0.93.1) or none
  --force               start even if the status file says a run is alive
  --no-echo             log to the file only
  --dry-run             select only: list what would be processed
```

## mppp align

```text
usage: mppp align [-h] [--variant VARIANT] [--set NAME=VALUE] [--settings FILE]
                  [--sites-file FILE] [--hazcam {front,rear,both,none}] [--force] [--no-echo]
                  [--gui] [--stop-after STEP] [--status]
                  WORK

positional arguments:
  WORK                  the WORK folder (the one holding processed/)

options:
  -h, --help            show this help message and exit
  --variant VARIANT     results in <WORK>/colmap_<variant> instead of colmap/
  --set NAME=VALUE      a pipeline setting (mppp.pipeline.Settings field), VALUE as JSON or
                        Python; repeatable
  --settings FILE       a JSON file of settings {NAME: value}
  --sites-file FILE     site definitions (default mppp/data/sites.json)
  --hazcam {front,rear,both,none}
                        the Hazcam frames of the block's stations: front, rear, both (the default
                        since 0.93.1) or none
  --force               start even if the status file says a run is alive
  --no-echo             log to the file only
  --gui                 write the COLMAP GUI files (colmap_gui.ini, open_in_colmap.bat)
  --stop-after STEP     stop after this step: manifest, project, features, database, match,
                        reconstruct, health
  --status              print the folder's run status and exit
```

## mppp run

```text
usage: mppp run [-h] [--variant VARIANT] [--set NAME=VALUE] [--settings FILE] [--sites-file FILE]
                [--root DIR] [--pds DIR] [--zcam] [--hazcam {front,rear,both,none}] [--force]
                [--no-echo] [--gui] [--stop-after STEP]
                SITE|WORK

positional arguments:
  SITE|WORK             a site of the site definitions, or its WORK folder

options:
  -h, --help            show this help message and exit
  --variant VARIANT     results in <WORK>/colmap_<variant> instead of colmap/
  --set NAME=VALUE      a pipeline setting (mppp.pipeline.Settings field), VALUE as JSON or
                        Python; repeatable
  --settings FILE       a JSON file of settings {NAME: value}
  --sites-file FILE     site definitions (default mppp/data/sites.json)
  --root DIR            the scapes root (default: MPPP_SCAPES)
  --pds DIR             the PDS archive (default: MPPP_PDS)
  --zcam                the Navcam + Mastcam-Z block of the site (..._zcam)
  --hazcam {front,rear,both,none}
                        the Hazcam frames of the block's stations: front, rear, both (the default
                        since 0.93.1) or none
  --force               start even if the status file says a run is alive
  --no-echo             log to the file only
  --gui                 write the COLMAP GUI files
  --stop-after STEP     stop the alignment after this step
```

## mppp sites

```text
usage: mppp sites [-h] [--sites-file FILE] [--root DIR] [--zcam] [{list,check,folder}] [site]

positional arguments:
  {list,check,folder}
  site                 folder: the site name

options:
  -h, --help           show this help message and exit
  --sites-file FILE
  --root DIR           folder: the scapes root (default: MPPP_SCAPES)
  --zcam               folder: the Navcam + Mastcam-Z block
```

## mppp cmods

```text
usage: mppp cmods [-h] [--note NOTE] [--dry-run] [{list,promote}] [paths ...]

positional arguments:
  {list,promote}
  paths           promote: folders or files with the new models

options:
  -h, --help      show this help message and exit
  --note NOTE     promote: why these are the better models (goes into CHANGES.md)
  --dry-run       promote: check and show, copy nothing
```

## mppp mask

```text
usage: mppp mask [-h] [--name NAME] [--force] [{list,fetch,install}] [arg]

positional arguments:
  {list,fetch,install}
  arg                   fetch: the model name; install: a .pt/.safetensors file

options:
  -h, --help            show this help message and exit
  --name NAME           install: the registry name
  --force               fetch: download again
```

## mppp example

```text
usage: mppp example [-h] [--dest DIR] [--force] [--pds DIR] [--zcam DIR] [--out DIR] {fetch,build}

positional arguments:
  {fetch,build}  fetch: download the release asset and unpack it; build: zip it from a local PDS
                 archive (for the release)

options:
  -h, --help     show this help message and exit
  --dest DIR     fetch: where to unpack (default ~/mppp_example)
  --force        fetch: download again
  --pds DIR      build: the PDS archive (default MPPP_PDS)
  --zcam DIR     build: the Mastcam-Z folder (default <PDS>/../zcam)
  --out DIR      build: where to write the zip
```

## mppp waypoints

```text
usage: mppp waypoints [-h] [--refresh]

options:
  -h, --help  show this help message and exit
  --refresh   download the current table now (every sync should)
```

## mppp maps

```text
usage: mppp maps [-h] [--out DIR] [--dpi DPI] [--background {dtm,ortho,white}] [--tag TAG]
                 [--sites-file FILE] [--dems DEMS] [--orthos ORTHOS]
                 [what ...]

positional arguments:
  what                  all (default) | zooms | figures [2 3 4 5 6]

options:
  -h, --help            show this help message and exit
  --out DIR             output folder (default <repo>/Claude outputs/maps_<yyyymmdd>)
  --dpi DPI
  --background {dtm,ortho,white}
  --tag TAG             appended to every figure name
  --sites-file FILE
  --dems DEMS           1 m DTM GeoTIFFs, os.pathsep-separated (M20_DEMS)
  --orthos ORTHOS       orthomosaic GeoTIFFs (M20_ORTHOS)
```

## mppp calib

```text
usage: mppp calib [-h] [--variant VARIANT] [--root DIR] [--group GROUP] [--zoom ZOOM]
                  [--navcam-group NAVCAM_GROUP] [--blocks [NAME=WORK ...]] [--sites-file FILE]
                  [--out DIR] [--points POINTS] [--zpoints ZPOINTS] [--hold [HOLD ...]]
                  [--knot KNOT] [--aniso-f] [--no-diagnostics] [--include-failed] [--no-k-tests]
                  [--promote-if-ok] [--no-promote] [--note NOTE]
                  {navcam,zcam,nz,hazcam}

positional arguments:
  {navcam,zcam,nz,hazcam}
                        navcam: the one-adjustment joint over the finished Navcam blocks; zcam:
                        the Mastcam-Z joint over the finished Navcam + Mastcam-Z blocks, the
                        Navcam held at the models in use; nz: the combined joint (both laws free)
                        compared with the Navcam-only law - a check, never promoted; hazcam: the
                        Hazcam law (Navcam form) from the finished blocks with Hazcams (0.93)

options:
  -h, --help            show this help message and exit
  --variant VARIANT     the blocks' project colmap_<variant> (default colmap; the Hazcam-only
                        alignments: hazcam)
  --root DIR            the scapes root (default: MPPP_SCAPES)
  --group GROUP         site group(s) of the blocks, comma-separated (default ecam_consensus /
                        zcam<zoom>_joint, else zcam<zoom>_consensus); zcam / nz take the sites
                        whose "zcam" list has the zoom
  --zoom ZOOM           zcam: the Mastcam-Z zoom (default 34)
  --navcam-group NAVCAM_GROUP
                        nz: also the finished Navcam-only blocks of this group (e.g.
                        ecam_consensus), 3000 points each, so the combined Navcam law is held by
                        every Navcam block
  --blocks [NAME=WORK ...]
                        blocks given explicitly instead
  --sites-file FILE
  --out DIR             the study folder (default <root>/camera_analysis/navcal_tied_<date>)
  --points POINTS       Navcam points per block (default 8000)
  --zpoints ZPOINTS     zcam: Mastcam-Z-only points per block (default 8000; every Navcam-
                        Mastcam-Z point with a third look is kept besides)
  --hold [HOLD ...]     global terms held (default Q0_yaw)
  --knot KNOT           the pitch hinge sol (default the rig file's)
  --aniso-f             separate fx and fy thermal slopes (default: one ppm/degC per eye, fy tied
                        to fx)
  --no-diagnostics      skip the per-block diagnostic pass (no gate then)
  --include-failed      also blocks whose alignment health is FAIL (left out by default)
  --no-k-tests          zcam: skip the k1/k2 likelihood-ratio tests (2 more adjustments)
  --promote-if-ok       promote the candidate when every check passes
  --no-promote          never promote (overrides --promote-if-ok)
  --note NOTE           promotion note
```

`--iterations N` and `--passes N` (0.95.3; `zcam` and `nz`): Ceres iterations per pass of the adjustment and of its
tests (default 100) and the number of passes (default 2, the covariance on the last). `zcam_backlash_rerun.bat`
runs the joints with `--iterations 300`.

## mppp doctor

```text
usage: mppp doctor [-h]

options:
  -h, --help  show this help message and exit
```

## mppp version

```text
usage: mppp version [-h]

options:
  -h, --help  show this help message and exit
```

## mppp gui

Writes `open_in_colmap.bat`, `open_in_colmap_fullres.bat`, `colmap_gui.ini` and (0.95.0) `dense_reconstruct.bat`
into finished projects.

## mppp single-sites

0.96.0 (Christian, 10 Nov 2026: "We'll use these three as the test sites ... the single sites should come from a
config file"). The single-station test sites of `mppp.ibr`, listed in `src/mppp/data/single_sites.json` (per
station: `site` - a site of `sites.json`, whose Navcam WORK folder is the block - or `work`; `station`; `frames`:
`images`, `scale` or `sequences` for the dense step; optional `variants`, `mesh_args` / `depthmaps_args` /
`ods_args`, `note`).

```text
usage: mppp single-sites [LABEL ...] [--config FILE] [--root D:\rover_blocks] [--out DIR]
                         [--steps dense mesh depthmaps ods] [--variants masked rover]
                         [--squares-only] [--list] [--dry-run] [--install DIR]
```

Per station and variant the steps run as their own processes: `mppp dense` (stereo and fusion, then the fusion with
the rover kept), then `mesh`, `depthmaps --photometry` and `ods` into `<root>\single_sites\<label>\` (rover masked
out) and `...\<label>\with_rover\` (rover shown). A failed step skips the rest of its variant (a failed dense step
the whole station). Each station's output also goes to `single_site_log.txt` in its folder;
`single_sites_summary.md` has a line per station and variant: the exit codes, how far the fused points reach, the
ground plane, the panorama's coverage, whether the square is there. `--squares-only` only stacks the eye images
already there. `--install DIR` writes `run_single_site_ibr.bat` into DIR: it fast-forwards `D:\code\MPPP`
to the newest bundle in `D:\rover_blocks\_transfer` when that bundle is ahead (never back), then runs
`mppp single-sites` with its arguments - it names no version, so it never needs replacing.

## mppp dense

0.95.0 (Christian, 28 Oct 2026): the COLMAP dense reconstruction of a finished block, preloaded from the alignment.

```text
usage: mppp dense [-h] [--variant VARIANT] [--prepare] [--stereo] [--fusion] [--mesh]
                  [--max-image-size N] [--python PYTHON] [--overwrite] work
```

`--prepare` (the default with no step given) writes the workspace `<project>/dense/`: the images undistorted to
pinhole cameras from the **native-pixel** model `gui_native/sparse` (the image files are at their native
resolution; the full-resolution model's cameras would not fit them), the pinhole model `dense/sparse/`, COLMAP's
`stereo/patch-match.cfg` and `fusion.cfg`, and `dense/masks/` - the rover masks undistorted the same way, binary
again, for the fusion (`StereoFusion.mask_path`: a pixel at 0 is never fused; the patch-match stereo itself takes
no masks). `--max-image-size` (default 2000 px on the long side) sets the working resolution. `--stereo` runs the
patch-match stereo (a CUDA pycolmap: `--python` or `GPU_PYTHON`, the environment the matcher uses), `--fusion`
the fusion into `dense/fused.ply` with the masks, `--mesh` a Poisson mesh.

**A station or a sequence only** (0.96.0): `--station S026D0630` (or `26_630`, `sol:461`) with `--cameras N`,
`--sequence NCAM03461 ...` and `--scale 0.5` (the half-resolution frames), or `--images NAME ...`, works on those
frames alone: a sub-model of them (the other frames deregistered, their observations dropped) is undistorted into
its own workspace `<project>/dense_<station>[_<sequence>]`, so the patch-match stereo and the fusion see nothing
else, and `mppp mesh --station` takes that workspace's `fused.ply` first. For example the sol-461 half-resolution
Navcam frames of Rockytop: `mppp dense <WORK> --station 26_630 --scale 0.5 --prepare --stereo --fusion`.
A station run loosens COLMAP's patch-match and fusion filters (10 Nov 2026): a pixel needs a 0.4 deg triangulation
angle instead of 3 deg (with Navcam's 0.424 m baseline, 61 m instead of 8 m) and one consistent source view instead
of two, and a fused point 2 images instead of 5 - the far terrain of one station is seen by one stereo pair only.
`--min-angle DEG` and `--min-views N` change them; a whole-block run keeps COLMAP's defaults.
The options of the patch-match maps are recorded in `stereo/mppp_stereo.json`: maps made with other options are
deleted before `--stereo` (COLMAP skips every image whose maps exist), `--restereo` deletes them in any case.
`--keep-rover` fuses the same depth maps without the rover masks into `fused_rover.ply` (for `mppp mesh / depthmaps /
ods`, which show the rover body and deck by default since 0.96.1 - products in `ods/<station>_rover/`;
`--mask-rover` for the rover-masked products in `ods/<station>/`); after a run, `--fusion --keep-rover` alone is
enough.

`dense_reconstruct.bat`, written next to `open_in_colmap.bat`, does the same with the COLMAP executable (a CUDA
build is looked for first): double-click runs undistortion (without the masks: `mppp dense --prepare` for those),
stereo and fusion; `set MPPP_DENSE_STEP=prepare|stereo|fusion|mesh` runs one step, `MPPP_DENSE_MAX_SIZE` the
resolution, `MPPP_DENSE_MESH=1` adds the mesh. The GUI's Reconstruction > Dense reconstruction dialog takes the same
`dense\` folder as its workspace, with the undistortion already done.

```text
usage: mppp gui [-h] [--root DIR] [--variant VARIANT] [works ...]

positional arguments:
  works              WORK folders (default: every WORK folder under --root
                     with a model)

options:
  -h, --help         show this help message and exit
  --root DIR         the scapes root (default: MPPP_SCAPES)
  --variant VARIANT  the project colmap_<variant> instead of colmap
```

## mppp mesh, mppp depthmaps, mppp ods

0.96.0 (Christian, 9 Nov 2026: "the pipeline that generates the per image depth maps and photometric cube from the
cleaned up 3d mesh ... prototype the stereoscopic pipeline. The output should just be two images"): the products of
one **station** (a site and drive of the rover) of a finished block, under `<project>/ods/<station>/`
(`mppp.ibr`; the `[ibr]` extra: open3d).

```text
usage: mppp mesh      WORK --station S026D0392 [--points dense/fused.ply] [--colmap colmap.exe] [--dense-dir DIR]
                      [--mesh-radius 30] [--plane-radius 100] [--horizontal-plane] [--no-tie-points]
                      [--far-radius R] [--target-triangles 3000000] [--poisson-depth 10] [--cameras N] [--no-preview]
usage: mppp depthmaps WORK --station ... [--cameras N] [--scale 1] [--zcam-scale 1] [--images NAME ...]
                      [--photometry] [--no-shadow] [--no-preview] [--no-validate]
usage: mppp ods       WORK --station ... [--width 8192] [--ipd 0.064] [--cameras N] [--scale 1] [--images ...]
                      [--rows-per-chunk 128] [--eyes left right] [--exclude STEM ...] [--stack-top left]
                      [--prefix S003D2046] [--bits 16] [--pds DIR] [--tone soft]
```

`--station` is `S026D0392`, `26_392` or `sol:461`.  **`mesh`**: the station's frames by camera family, the mast
pivot fitted from their centres and rotations (`station.json`), then the surface over the fused cloud (the newest
`dense*/fused.ply`, or `--points`): Poisson (open3d), or COLMAP's Delaunay mesher through `--colmap` over
`--dense-dir`; then the cleanup of the 31 Oct plan — components under 1 %, rubber sheets (long thin triangles),
surface no fused point supports within τ(z) = max(5 cm, 2 % z, 3 σz), everything farther than `--mesh-radius`
(30 m) from the mast axis (horizontal distance; 10 Nov 2026) — the decimation to `--target-triangles`, Taubin
smoothing. Beyond the mesh: **the ground plane** out to `--plane-radius` (100 m) from the axis, fitted to the fused
points and the block's tie points (seen by 3+ images) as the medians of 10 deg x range-ring bins and a Huber fit,
so it follows the regional slope (`--horizontal-plane` for a level one; a tilt over 20 deg or too few bins falls
back to it); it also fills the mesh's holes. Beyond the plane **the sky at infinity**. The mesh has priority over
the plane even where the plane is nearer, and only the mesh occludes. `mesh.ply`, `ground_plane.json` (the plane,
its tilt and its misfit per ring), `mesh_report.json` (the fused points per ring from the axis, the coverage by
range), `mesh_preview.png` (the plane outlined in the range panorama). `--far-radius R` gives the 0.96.0
prototype's finite sphere instead of the sky.
**`depthmaps`**: every frame of the station ray-cast through its own camera model into mesh, plane and sky at
`--scale` of the native size: `maps/<image>.npz` (depth, range, normal, label 0 none / 1 mesh / 2 sky at infinity,
no depth / 3 masked / 4 plane) and 16-bit previews (the plane darker), validated against the sparse points the frame
observes, on the mesh and on the plane separately (`maps_report.md`: how far the flat stand-in is from the terrain); `--photometry` adds the
photometric cube from the sun direction of the frame's row (`sun_enu` turned into the block frame): incidence,
emission, phase and the cast shadow — `photometry/` (npz), `photometry_maps/` (COLMAP `.bin` arrays),
`photometry_jpg/`, `photometry_report.json`.  **`ods`**: the omni-directional stereo panorama — equirectangular,
`--width` × half, the eyes `--ipd` apart on the circle around the pivot — each eye's rays cast into the scene and
every hit coloured from the source frame that sees it best (the smallest angle to the frame's axis, inside the
frame and its mask): `<station>_ods_left.png`, `<station>_ods_right.png`, `<station>_360_3DTD.png/.jpg`
(the left eye above the right; `--stack-top right`), `provenance.png` (which frame coloured what), `ods.json` (the
coverage: coloured, mesh, far, none); with the rover shown (the default since 0.96.1) they are
`<station>_rover_ods_...`, with `--mask-rover` `<station>_ods_...`; `--exclude STEM ...` leaves
frames out (both eyes of a pair).  **`mesh`** with the rover shown puts a floor at the rover deck: nothing below the deck top
inside its footprint (RMECH x -1.10..+0.95 m, y +-0.60 m, z 0 = the deck top; the rover placed by the frames' label
CAHVORE poses), the floor added to the mesh, `rover_deck.json`; `--no-deck-floor` to leave it out.  With the rover shown the mesh also takes NASA's Perseverance
model (simplified, the arm stowed; `mppp.ibr.rover_model`) at the station's rover pose, the fused points within 4 cm
of it dropped (`--no-rover-model`, `--rover-model-parts`).  **`ods --bits 16`** (0.96.1) blends the panorama from 16-bit linear frames made
again from the PDS products (`<project>/images16/`; `--pds` or `MPPP_PDS`): `<station>_ods_left_16.png`,
`_right_16.png`, `_360_3DTD_16.png` (uint16 linear, for colour work; what the 8-bit images clip at 255 is kept,
saturated pixels are neutral) next to the 8-bit ones, which get a soft highlight roll-off (`--tone clip` for the old
cut).  `mesh` also removes floating surface: anything above the cameras, and beyond
3 m of the mast axis anything less than 1 m below them, heights above the ground plane (`--keep-floating`).  Any VR photo viewer that takes an equirectangular pair shows it; the
viewer itself is future work.

## mppp metashape

0.95.2 (Christian, 30 Oct 2026): a finished block as an Agisoft Metashape project - cameras, tie points and masks,
the photos linked from the COLMAP project's `images\`.

```text
usage: mppp metashape [-h] [--variant VARIANT] [--model MODEL] [--out OUT] [--max-points MAX_POINTS]
                      [--metashape METASHAPE] [--run] work
```

Writes `<project>\metashape\`: `metashape_job.json`, the tie points as Bundler v0.3 (`<block>.bundler.out` and
`<block>.bundler_images.txt`), `build_metashape_project.py` and `make_metashape_project.bat`. Double-click the batch
file (or `--run`): Metashape Professional (`MPPP_METASHAPE`, else the standard install) builds `<block>.psx` there
and leaves it open (`set MPPP_METASHAPE_HEADLESS=1`: without the window). Nothing is aligned or optimised: the
sensors (one per COLMAP camera of `gui_native\sparse`) have their calibration fixed and every camera its COLMAP
pose. The chunk's coordinate system is local and metric - MPPP's world frame of the block (x east, y north, z up;
the chunk transform adds the block offset). `metashape_check.json` reports COLMAP's observations and the imported
tie points reprojected by Metashape (they should match COLMAP's residuals, about 0.15 px median).

| COLMAP | Metashape |
|---|---|
| THIN_PRISM_FISHEYE (Navcam, Hazcam) | Fisheye |
| FULL_OPENCV (Mastcam-Z) | Frame |
| fx, fy | f = fy, b1 = fx - fy |
| cx, cy (from the top-left corner) | cx - w/2, cy - h/2 (from the frame centre) |
| k1, k2, k3, k4 | K1, K2, K3, K4 |
| p1, p2 | P2, P1 (swapped: Metashape's P1 goes with x) |
| sx1, sy1; k4-k6 of FULL_OPENCV | no place in Metashape: zero in every MPPP model (else fitted and reported) |

## mppp nerfstudio

0.95.2 (Christian, 30 Oct 2026): a finished block for Nerfstudio's splatfacto - every frame in one pinhole size.

```text
usage: mppp nerfstudio [-h] [--variant VARIANT] [--model MODEL] [--out OUT] [--size SIZE] [--zcam-clip ZCAM_CLIP]
                       [--families FAMILIES] [--jpg] [--threads THREADS] [--overwrite] work
```

Writes `<project>\nerfstudio\`: `images\` and `masks\` (2048 x 1536 by default), `transforms.json` and
`sparse_pc.ply` (the tie points, splatfacto's start). Each frame is resampled through COLMAP's exact camera model
(`gui_native\sparse`) into a pinhole frame with its own focal length and principal point (per frame in
`transforms.json`): Navcam and Hazcam keep their centre resolution scaled by 2048 / width (about 82 deg of a Navcam,
100 deg of a Hazcam; the fisheye edges beyond are left out); Mastcam-Z frames lose their first 22 and last 16
columns (`--zcam-clip`) and are scaled by 1536 / 1200. The masks are resampled the same way and are 0 wherever the
frame has no source pixel. `transform_matrix` is camera to world with OpenGL camera axes, in MPPP's world frame
(x east, y north, z up, metres; the block offset in `mppp`).

```text
ns-train splatfacto --data <project>\nerfstudio nerfstudio-data
```

`mppp run SITE --metashape --nerfstudio` - and `scripts\windows\ecam_block.bat SITE` - align a site's Navcam +
Hazcam block and write both exports.
