# Command line

The `mppp` command is installed with the package (`python -m mppp` is the same). The text below is the
output of `--help` for the command and each subcommand. Settings of `process`, `align` and `run` are the
fields of `mppp.pipeline.Settings` (see the [API reference](api.md)); `--set NAME=VALUE` overrides one.

## mppp

```text
usage: mppp [-h] [-v | -q] [--log FILE]
            {process,align,run,sites,gui,cmods,mask,example,waypoints,maps,calib,doctor,version} ...

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
    mppp doctor                               what is set, what is missing (environment, dependencies, models)
    mppp version

``-v`` / ``-q`` set the logging level, ``--log FILE`` tees the log to a file.  ``process``, ``align`` and ``run``
write ``<WORK>/runs/<time>/log.txt`` and ``<WORK>/mppp_status.json`` as well (:func:`mppp.runner.run_logged`).
Every subcommand returns an exit code (0 ok, 1 failed, 2 bad arguments, 3 a run is already going on).

positional arguments:
  {process,align,run,sites,gui,cmods,mask,example,waypoints,maps,calib,doctor,version}
    process             select the PDS products of a site and process them into <WORK>/processed
    align               align the images processed in a WORK folder
    run                 process, then align a site
    sites               the site definitions
    gui                 write the COLMAP GUI files (open_in_colmap.bat ...) into finished projects
    cmods               the camera models in use
    mask                the terrain mask model
    example             the Mars 2020 example data (Rockytop, sols 461-464): fetch or build
    waypoints           the Mars 2020 waypoint table in use (cached download or packaged snapshot)
    maps                the M20 site-map figures (studies/m20_sites) from the official blocks of
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
  --variant VARIANT     the blocks' project colmap_<variant> (default colmap; run_hazcam.bat:
                        hazcam)
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

Writes `open_in_colmap.bat`, `open_in_colmap_fullres.bat` and `colmap_gui.ini` into finished projects.

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
