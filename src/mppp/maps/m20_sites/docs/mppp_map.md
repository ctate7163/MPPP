# mppp.map — M2020 blocks of rover stations: figures and MPPP sites export

**Status:** working prototype, ready to port into the MPPP package as `mppp.map`. Entry point `mppp_map.py`.
**Data:** `M20_waypoints.json`, sols 13–2009 (706 waypoints); Mastcam-Z numbered 360s #1–#56.
**Last updated:** 2026-10-16.

## 1. What it does

1. Finds **blocks of rover stations**: places where Perseverance ended the drive repeatedly in a small area, plus sites added by hand. Each end-of-drive stop has a full Navcam 360.
2. Draws six figures that place the 34 blocks in context, from global Mars down to ±30 m.
3. Writes the blocks as an **MPPP sites file** (`sites_blocks.json`, sites.json version 4), so the photogrammetry notebooks can process each block.

## 2. Quick start

```bash
pip install numpy scipy matplotlib rasterio pyproj adjustText pillow
python mppp_map.py all out/               # blocks -> zooms -> figures 2-6 -> sites_blocks.json  (~6 min at 300 dpi)
python mppp_map.py figures out/ 5 6       # selected map figures only (2, 3, 4, 5, 6)
python mppp_map.py zooms out/             # the ±30 m zoom figure only
python mppp_map.py export out/            # sites file only (needs sites_zoom.json from `blocks`)
python mppp_map.py clusters               # re-run raw clustering after a new waypoints file, then `all`
M20_DPI=450 python mppp_map.py all out/   # higher resolution (default 300 dpi; images over 8000 px cannot be previewed in chat)
```

**Optional HiRISE DTM background:** put DTM GeoTIFFs in `./dem/` or set `M20_DEMS` to a list of paths.

- Subfolders are fine, and only files with DTM or DEM in the name are read.
- Separate the `M20_DEMS` paths with `;` on Windows and `:` on Linux/Mac.
- Without a DTM, the maps use the bundled ~70 m terrain tiles and say so in the titles or footnotes.

## 3. Pipeline

| Step | Script | Inputs | Output |
|---|---|---|---|
| Raw clusters | `cluster.py`, `analyze.py` | `M20_waypoints.json` | `clusters.json`, `clusters_big.json` |
| Blocks | `sitesdef.py` | clusters, `sites.json` (legacy MPPP v3), `refdata.py` | `sites_zoom.json` |
| Zoom figure | `zooms2.py` | `sites_zoom.json`, waypoints, DTM (opt.) | `m20_cluster_zooms.png` |
| Map figures | `mapplot3.py OUTDIR [2 3 4 5 6]` | `sites_zoom.json`, `terrain/` tiles, DTM (opt.) | figures 2–6, one process each |
| Sites export | `export_sites.py` | `sites_zoom.json`, `sites.json` | `sites_blocks.json` |

**Support modules:**

- `refdata.py`: the Mastcam-Z 360 list #1–#56 with sols, sample cores 1–30, and the depot.
- `dem.py`: DTM discovery, the lon/lat ↔ waypoint easting/northing frame, bilinear sampling and hillshade.
- `terrain_tiles.py` and `qmesh.py`: turn quantized-mesh terrain tiles into a gridded DEM. The tile cache is in `terrain/` with `layer.json`.

## 4. Block rules (sitesdef.py)

- **Waypoint classes** (`F` flag): `y` = end-of-drive (full Navcam 360); `m` = mid-drive (usually a partial Navcam); `i` = interpolated (no images).
- **Clusters:** single-linkage of end-of-drive waypoints at 10 m, keeping clusters with ≥4 stations. OEB sols 13 and 14 are merged by hand (`MANUAL_MERGE`).
- **Added sites:**
  - From legacy sites.json windows (`ADDED`): Rochette, Three Forks North, Chal Rocks, Airey Hill, Overlook Mountain, Bright Angel, Rio Chiquito, Groloy, Tenby/Berea and Marble Mountain.
  - From user sol windows (`EXTRA_WINDOWS`): Wild Basin Lake 838–853 and Plateau Point 1252–1263.
  - Removed: unnamed cluster 20 (`REMOVED_CLUSTERS`).
- **Centre:** the mean of the cluster's stations, or the densest stop if they spread more than 40 m. `CENTRE_OFFSETS` shifts Tuxedo Park 20 m west so the sol 905 and 907 stops fall inside.
- **Block:** a ±30 m square (`HALF = 30`) around the centre. Every waypoint in the square counts, not only cluster members.
- **Visits:** the sols on which the rover ended inside the square, split into continuous runs. `RUN_EXCLUDE` hides single-sol runs from the listing (sol 413 at Three Forks North) without renumbering.
- **Numbering:** by first sol of residency (`first_all`, which includes hidden sols), giving blocks 1–34.
- **Counts:** `N×n` = end-of-drive Navcam 360s in the square; `Z×n` = numbered Mastcam-Z 360s taken in the square.
- **Regions** (used by figures 4–6 and the export groups):

| Region | Blocks |
|---|---|
| Crater floor and Séítah | 1–7 |
| Delta front | 8–13 |
| Upper fan and Belva | 14–18 |
| Margin unit and Neretva Vallis | 19–25 |
| Crater rim (Witch Hazel Hill) | 26–29 |
| Crater rim west | 30–34 |

## 5. Figures

All figures are rendered at 300 dpi (set with `M20_DPI`) and use a global font scale of 1.15 (`FS`).

| Marker | Meaning |
|---|---|
| Orange square | Block of rover stations |
| Triangle | Mastcam-Z numbered 360 |
| Diamond | Sample core |
| Blue line | Traverse |
| Star | Octavia E. Butler landing site |

| # | File | Function | Size (px) | Layout |
|---|---|---|---|---|
| 1 | `m20_cluster_zooms.png` | `zooms2.py` | 5175 × 7560 | 5 columns × 7 rows: one ±30 m panel per block (orange border) plus the legend (details below) |
| 2 | `m20_jezero_traverse_clusters_map.png` | `fig2` | 5400 × 2220 | A crater (77.1–78.1 E, 17.9–18.8 N) \| B traverse (77.205–77.47 E, 18.38–18.515 N); C block table in 3 columns |
| 3 | `m20_jezero_traverse_sites_regions.png` | `fig3` | 5400 × 5783 | A–C, then regional panels D–I |
| 4 | `m20_jezero_traverse_sites_regions_wide.png` | `fig4` | 7200 × 3367 | A \| B \| one-column table; a row of six regional panels D–I in reverse chronological order |
| 5 | `m20_mars_to_rover_sites.png` | `fig5` | 5400 × 7112 | Mars to rover sites, four rows (below) |
| 6 | `m20_mars_to_rover_sites_no_I.png` | `fig6` = `fig5(out, with_I=False)` | 5400 × 6794 | Same rows 1–2; E–H two over two with the block list on the right (below) |

**Figure 1 panel details:** every waypoint is labelled by sol, the title gives the visits, and the footer gives the counts. "Relative northing [m]" labels the left column, and "Relative easting [m]" labels the bottom panel of each column. The text is enlarged (titles 9.6 pt, sol labels 7.2 pt).

### Figure 5: Mars to rover sites

Every row ends at the same right edge, where row 2's colour bar ends (W − 0.93 in).

**Row 1**

- **A, global Mars** (terrain level 4): box B and the landing star.
- **B, Isidis basin square** (centre 85.5 E, 14 N, ±15°): a small landing star and the Jezero box C (thin line). B merges the z7/z8 terrain tiles around Jezero into the z5/z6 basin tiles; without them Jezero is smoothed away and the box appears to sit on featureless slope.
- In both panels the star is drawn above the black box.
- A and B share one colour range, clipped at the 1st/99th percentile (about ±5.9 km, with arrows on the colour bar).
- The shading is the original (exaggeration 0.02× and 0.3×). The images are then scaled so their mean brightness matches C and D (gain about 1.2).

**Row 2**

- **C, Jezero crater**, square (1:1, `BOX_C5`): 77.1–78.1 E, from 18.8 N down to about 17.85 N. The landing star is labelled "Octavia E. Butler Landing Site" and the crater "Jezero crater", both in white with a dark outline (`panel_A(..., names=True)`). The 10 km scale bar is at the bottom right.
- **D, "Blocks of rover stations"**: the traverse at 2:1 (`BOX_B5`), with region boxes E–I and the legend.
- Block numbering by first sol and the background note are in the footnote.

**Row 3:** four tall panels, 1.5 km wide, widened to fill the row. The gap between neighbouring panels (0.75 in) leaves room for F's and H's y tick labels.

- E Crater rim west (blocks 30–34) | F Crater rim (26–29) | G Delta Front (8–13) | H Crater Floor and Séítah (1–7).
- Panel height is fixed (`sh`), so each shows about 2.0 km; the bottom edge stays where a centred 2 km box would put it.
- E+F share one colour bar, and G+H share another.

**Row 4:** one wide panel I, Upper Fan, Belva Crater, Margin Unit and Neretva Vallis (blocks 14–25).

- Full row width (about 7.4 km) at the same scale as E–H, about 2.5 km tall, with its own colour bar.
- Its box in D overlaps F and G, which is intended.

**Rows 3–4, all panels**

- Block numbers are 10 pt, and tick labels are 7.8×FS pt.
- Titles are one line, `X. Region, sols a–b, blocks m–n`. They shrink only if they would run into the next panel.
- Latitude labels appear only on the left-most panels, and longitude labels only on the bottom row.
- The script asserts that every block centre lies inside its panel.

### Figure 6: compact version with the block list

- **Rows 1–2:** as in figure 5, without the I box in D.
- **Rows 3–4:** E F over G H, at the same scale and height as figure 5.
- **Right side, panel I:** the list of all 34 blocks (number, name, N×/Z× counts, sols). Its monospace font is fitted to the space (about 12 pt).

### Main settings in `mapplot3.py`

| Setting | What it controls |
|---|---|
| `DPI` (env `M20_DPI`) | Output resolution |
| `FS` | Font scale |
| `BOX_A`, `BOX_B` | Crater and traverse boxes (figures 2–4) |
| `BOX_C5`, `BOX_B5` | Square crater box and 2:1 traverse box (figures 5–6) |
| `sh`, `sh4` | Row 3 and 4 panel heights in inches (`BH`, `BH4` in metres follow from the scale) |
| `REGION_KEYS`, `REGION_TITLE` | Region definitions and titles |
| `scalebar(..., where=)` | Scale-bar side per panel |
| `panel_B(..., landing_label=True)` | Landing-site label |
| `draw_terrain(..., exag, flat)` | Hillshade exaggeration and brightness |

## 6. Sites export (`sites_blocks.json`)

The file uses MPPP sites.json **version 4**: `{"_about", "version": 4, "sites": {...}, "groups": {...}}`. Keys are `blk{NN}_{slug(name)}`, for example `blk09_three_forks_north`.

| Field | Meaning |
|---|---|
| `sols` | Visits as `[first, last]`, longest visit first; that visit names the folder `mars2020_sol_<first>_<key>_colmap`. All visits are selected. |
| `label` | Block name. |
| `zcam` | Mastcam-Z zoom settings copied from the matching legacy sites.json entry; `[]` means Navcam only. |
| `note` | One line: block number, counts, origin, visits. |
| `no_mask_inference_at` | Copied only when the block came directly from that sites.json key. |
| `settings` | `{}`; per-site overrides go here. |
| `block` | Geometry and provenance (fields below). |

The `block` record holds:

- `number`, `name`, `region`, `kind` (cluster, sites.json or sol window), `source_key`, `legacy_site`
- `centre` {lon, lat, easting, northing, frame} and `half_size_m`
- `visits` (chronological) and `end_of_drive_sols`
- `n_navcam_360`, `n_partial_navcam` and `mastcam_z_360` (the numbered panoramas)

**Groups:** `blocks_all`, plus one per region: `blocks_crater_floor_seitah`, `blocks_delta_front`, `blocks_upper_fan_belva`, `blocks_margin_unit_neretva_vallis`, `blocks_crater_rim`, `blocks_crater_rim_west`.

**Using it in MPPP:**

1. Either set notebook 03's `SITES_EXTRA = "sites_blocks.json"` and pick a group (for example `blocks_delta_front`), or merge `sites` and `groups` into `src/mppp/data/sites.json`.
2. Run `check_sites.py` on the merged file.

The `block.centre` and `half_size_m` fields let the image selector filter by position as well as by sol. A recommended next change to notebook 03 is to drop images whose rover position falls outside the square.

## 7. Proposed API (`mppp.map`)

```
mppp/map/
  __init__.py      # re-exports below
  waypoints.py     # load_waypoints(path) -> Waypoints(sol, x, y, flag, lon, lat); LocalFrame
  blocks.py        # find_clusters(wp, radius=10, min_n=4); build_blocks(wp, legacy_sites, added, half=30) -> list[Block]
  terrain.py       # terrain_tiles(box, step, levels); load_dtm(paths); sample_dtm(lon, lat); hillshade(z)
  figures.py       # zooms, traverse_map, regions, regions_wide, mars_to_rover(with_I=True|False)
  export.py        # sites_json(blocks, legacy_sites, out, version=4)
  cli.py           # `mppp map {all,clusters,blocks,zooms,figures,export}` (same verbs as mppp_map.py)
```

```python
from mppp import map as mmap
wp = mmap.load_waypoints("M20_waypoints.json")
blocks = mmap.build_blocks(wp, legacy_sites="src/mppp/data/sites.json")
mmap.figures.mars_to_rover(blocks, "out/m20_mars_to_rover_sites.png", dem=mmap.load_dtm(), dpi=300)
mmap.figures.mars_to_rover(blocks, "out/m20_mars_to_rover_sites_no_I.png", with_I=False)
mmap.sites_json(blocks, out="out/sites_blocks.json")
```

**Porting notes**

- The scripts do their work at import time and pass data through `sites_zoom.json`.
- To port them:
  - Wrap the module-level code in functions that take the waypoints and block list as arguments.
  - Move the constants (`HALF`, `MANUAL_MERGE`, `ADDED`, `EXTRA_WINDOWS`, `CENTRE_OFFSETS`, `RUN_EXCLUDE`, region lists) into a config dict.
- Keep one process per figure, or call `plt.close('all')` and `gc.collect()` between figures. At 300–450 dpi each figure needs one to a few GB.

## 8. Known limits and open items

- The background terrain is about 70 m resolution (HiRISE-DTM-based tiles plus MOLA) until the 1 m DTMs are connected:
  - MSR TRN HiRISE DTM: doi:10.5066/P13CPYYU, 77.057–77.380 E, covers blocks 14–34.
  - Mars 2020 TRN HiRISE DTM: doi:10.5066/P9REJ9JN, covers the crater floor and delta.
- Mastcam-Z #56 (Marble Mountain) has no published sol yet and is placed at sol 1962.
- Legacy `zcam` lists are matched by sol overlap; blocks with no legacy entry get `[]`.
- `end_of_drive_sols` includes sols hidden from the visit listing (413 at Three Forks North).
- Tuxedo Park is shifted 20 m west, the side where the sol 905 and 907 stops actually lie.
- Plateau Point uses the standard spelling.
- Figures 2–4 keep the original crater box (17.9–18.8 N), the "Landing (OEB)" label and their layouts; the square crater, 2:1 traverse and white place names are used only in figures 5–6.
