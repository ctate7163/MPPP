# Changelog

All notable changes to RPP from 1.0 on. The format follows [Keep a Changelog](https://keepachangelog.com/), and the
versions follow [Semantic Versioning](https://semver.org/): within 1.x the command line, the `rpp.pipeline`
functions, the camera-model file format and the names RPP writes into a work folder stay compatible. Parts marked
experimental in the documentation may change in a minor release.

The development history before 1.0 is in
[`docs/history/`](https://github.com/ctate7163/rpp/tree/main/docs/history): `CHANGELOG_v0p95-v0p99.md`,
`CHANGELOG_v0p90-v0p94.md`, `CHANGELOG_v0p13-v0p85.md` and `CHANGELOG_v0p1-v0p12.md`.

## 1.0.0rc1 — 2026-11-20

The first release candidate of 1.0. The pipeline's features and the station products are those of 0.99.53. This release cleans the code,
removes what earlier studies left behind, and makes RPP easier to install and to start with. The camera models are
still those of 0.99; 1.0.0 will carry the models of the third freeze run.

### Changed

- **The default rover mask model is `rpp_mask_v5`** (it was `rpp_mask_v3`). The third freeze run uses it.
  - A run key now names the mask model, so a block made with the old default is processed and aligned again under
    the new one. `--set mask_checkpoint=rpp_mask_v3` gives the keys of 0.99 and earlier and reuses those blocks.
  - A block processed with the checkpoint file that `rpp_mask_v5` was exported from
    (`convnext_tiny_s4_seg_20261118_v5p2_2.pt`, as in the third freeze run) has the same processing configuration
    as the default: its images are not processed again.
  - The Curiosity blocks (`rpp_studies.msl`) use `rpp_mask_v5` by default.
- **The PyPI package holds the pipeline only.** `pip install rover-photogrammetry` installs `rpp`. The research
  studies (`rpp_studies`) come with a source checkout (`git clone`, then `pip install -e .`), as before. `rpp calib`,
  `rpp zcam-states` and `rpp maps` say so when the studies are not there.
- **The viewer page is part of the package**: `rpp.viewer` (`index.html`, `vendor/`), moved from `docs/ods/`. An
  installed RPP can now write a local copy of the viewer (`rpp single-sites ... --publish DIR`). Without
  `RPP_VIEWER_DIR` and without a checkout named `rpv` beside a source checkout, `--upload` publishes into `./rpv`.
- **Install groups**: `[pipeline]` (masks, alignment, plots) is what `rpp run` needs; `[stations]` is the station
  products and the viewer (`[ibr]` still works). `[all]` is unchanged in what it installs.
- `rpp --help` is grouped by task and starts with the first steps.
- **`scripts\windows\rpv_process.bat`** is the station run for the viewer: `rpv_products_0p99p53.bat` of 0.99.53,
  with `update_rpp_rpv.bat` and `unpublish_station.bat` in it (`--unpublish LABEL ...`). Each window takes the first
  free checkout of `D:\code\RPP_rpv`, `_rpv_b`, `_rpv_c`, `_rpv_d`, so several runs share the stations through their
  locks. A checkout follows the branch of the tree the batch file is in, not a bundle. Log:
  `rpv_process_log_<slot>.txt`.
- **No nadir mark in the panoramas** (0.99.53, `nadir.MARK_IN_PANORAMA`): the mark is the picture
  `<prefix>_nadir_mark.png` alone, which the viewer lays on the ground (`ods.json`: `in_panorama: false`).
- **The rover's texture in the 3D modes has larger islands** (Christian: "the mesh_rover texture should also have
  larger texture islands if possible"): connected triangles that face within 35° share a chart of up to 0.6 m
  (`station_mesh.layout_charts`), in place of a cell per triangle. About 5 times finer texels in the same 4096²
  texture on a test mesh, and seams only at the charts' edges.
- **Every RPV product says who made it** (0.99.53): XMP `dc:creator` and `xmp:CreatorTool` in every JPEG and PNG of a
  station, `asset.generator` and `asset.extras` in the `.glb` files, `author` and `software` in `ods.json` and the
  `panos.json` entries.
- **The fused cloud is keyed on its own code** (`sfm/dense.py`), not on the station steps' code: a change to the
  panorama or mesh code no longer fuses every block again. The clouds of 0.99.51 to 0.99.53 are kept.
- The station code changed (the rover's texture charts), so every station is made once again under 1.0, from the
  mesh step on.
- **RPV:** B (headset) and the B key switch between the stereo terrain and the full 3D terrain and show the standard
  color each time (Christian, 20 Nov 2026).
- The freeze run needs RPP 1.0 (`rpp_studies.calibration.freeze.REQUIRED_VERSION`).
- Documentation: a new README, home page, installation page and roadmap; new pages
  [What is validated and what is experimental](https://ctate7163.github.io/rpp/status/) and
  [RPV: the Rover Photogrammetry Viewer](https://ctate7163.github.io/rpp/viewer/); the navigation is grouped.
- `CITATION.cff` carries the abstract, the keywords and the author's ORCID, and is the only source of the Zenodo
  record's metadata.
- The tests run on Linux and Windows in GitHub Actions.
- **Faster depth maps and a two-pass fusion** (`rpp dense`, every station and block workspace):
  - New depth maps match each image against its 12 best-connected images (COLMAP's default is 20) and sample every
    other pixel of the 11 x 11 matching window (`window_step 2`). `--sources N` (0: COLMAP's 20) and
    `--window-step 1` give the full work. The speed-up is an estimate (about 2-4 x) until the first block is timed.
  - A workspace that already holds depth maps (the 0.99 blocks, a block stopped part way) keeps the options its maps
    were made with, so no map is made again. `--restereo` makes them again with the new options.
  - `rpp dense --two-pass-fusion` (experimental, off by default): a loose fusion (two agreeing pixels: the far
    terrain that one stereo pair sees) and a strict one (four pixels, 1 px, 8°), merged: every strict point plus the
    loose points more than 2 working pixels from any strict point (`fused_loose.ply`, `fused_strict.ply`,
    `rpp_fusion_merge.json`). The default fusion is 0.99's, and the fused clouds of 0.99.51 to 0.99.53 are kept.
- **`rpp fusion-compare`** (experimental): fusion and mesh variants of a station's block cloud, made from the existing
  depth maps into `compare/` beside the cloud, and scored against the block's tie points ring by ring, with a "worse",
  "not worse" or "better" verdict. The variants are the two-pass fusion, a depth-map filter before the fusion
  (`rpp.sfm.depth_filter`: depth edges, small islands, back-facing normals, an optional range limit per camera), both
  together, and COLMAP's Delaunay mesher. The station products, the block's depth maps and its `fused.ply` are not
  changed. See [Station products](docs/station-products.md).
- **The JOSS figures 1-4 are in the standard figure set** (`make_figures.bat`, `rpp_studies.paper.joss_figures`): the
  pipeline diagram, the Navcam pairs of sols 9 and 52, the aligned blocks of Butler Landing, Van Zyl Overlook and
  Rockytop, and the 3D assets of the stations of sols 9, 461 and 52, one entry per file at 400 dpi. An entry whose
  input is missing is skipped with one line saying why. The oblique views and renders are ray cast, and the rover
  models are RPP's own model with the arm in its stowed pose from the image labels.
- **Belva Crater (`belva_crater_large`) starts at sol 770**, with no nearby frames from before it
  (`NEARBY_FROM_FIRST_SOL`). The freeze-3 block, `mars2020_sol_0768_belva_crater_large_zcam`, held 42 frames of
  sols 766 and 768. The block made from now on is `mars2020_sol_0770_belva_crater_large_zcam`.
- The Sol 784 station (`Sol0784_S039D0926`) takes the sol 797 NCAM03795 and NCAM02797 pairs as context, besides
  the sol 784 NCAM00301 pair. A station's new `zcam_context` list is context for its Mastcam-Z 34 mm version only:
  `Sol0789_S039D0926_Z34` takes the ZCAM08826 frames (sol 805, 110 frames).
- `rpp dense --add-images` takes name patterns (`N?F_*_N0390926NCAM*.png`), as `rpp ods --context-images` does, so
  a Mastcam-Z version run with `--terrain own` meshes its station and context frames. A pattern that matches nothing
  is left out with a note.
- **A station is named by its sol** (Christian, 20 Nov 2026: "there cannot be two eod locations on one sol"):
  `Sol0052`, not `Sol0052_S003D2046`; a Mastcam-Z version is `Sol0789_Z34`. The site and drive stay in the station
  list's `"station"` field and in `station.json`. The old names still select a station (`rpv_process.bat
  Sol0052_S003D2046`, `--unpublish`), a station list with old names is read with the new ones (two stations of one
  sol keep their long names), the viewer lists a station once under its new name, and old viewer links
  (`#Sol0052_S003D2046/...`) open the renamed station.
- The full 3D model's texture charts take triangles without area (Van Zyl: "the station mesh's part was not made
  (OverflowError: int too big to convert)").
- The release v1.0.0rc1 is archived on Zenodo: doi:10.5281/zenodo.24408630 (README badge, `CITATION.cff`,
  `docs/citing.md`).

### Removed

- **The names from before 0.99**: the alias packages `mppp` and `mppp_studies`, the `mppp` command, `MPPPImage` and
  `scripts/windows/mppp_env.bat`. Use `rpp`, `rpp_studies` and `RPPImage`. Still read, so that earlier work is
  reused: the `MPPP_*` environment variables, the old mask-model names in a block's configuration, and the file
  names RPP writes into a work folder (`rpp.ondisk`).
- **Earlier studies** in `rpp_studies.misc`: the calibration scripts from before 0.91 (`calibration_old`), the lens
  and temperature-bin experiments (`experiments`), the Navcam calibration notes of 0.52 and 0.53, the
  site-appearance analysis and their tests. `zcam_lines` moved to `rpp_studies.calibration`.
- **Batch files and scripts of finished work**: `zcam_joints_097.bat`, `zcam_backlash_rerun.bat`,
  `run_zcam_states.bat`, `check_rpp_reuse.bat`, `sync_rpp.bat`, `sites_status.bat`, `scripts/check_reuse.py` and
  `scripts/_make_notebooks_0p96.py`. `scripts/windows` holds 23 files.
- `.zenodo.json`.

Everything removed is in the git history at the tag `pre-v1.0-archive` (0.99.53).
