M2020 blocks of rover stations: figures and MPPP sites export (prototype for mppp.map, 16 Oct 2026)

Figures (300 dpi; set M20_DPI=450 for higher resolution):
  m20_cluster_zooms.png                       1. ±30 m panels for the 34 blocks, 5 columns x 7 rows
  m20_jezero_traverse_clusters_map.png        2. A crater | B traverse; C block table
  m20_jezero_traverse_sites_regions.png       3. A-C plus regional panels D-I
  m20_jezero_traverse_sites_regions_wide.png  4. A | B | table over one row of regional panels
  m20_mars_to_rover_sites.png                 5. A Mars | B Isidis; C crater | D traverse; E-H regional; I upper fan to margin unit
  m20_mars_to_rover_sites_no_I.png            6. as 5, with E F / G H two over two and the block list on the right
Data:
  sites_blocks.json   the 34 blocks as an MPPP sites file (version 4) -> notebook 03 SITES_EXTRA

Run (from this folder):
  pip install numpy scipy matplotlib rasterio pyproj adjustText pillow
  python mppp_map.py all out/        blocks -> zooms -> figures 2-6 -> sites_blocks.json
  python mppp_map.py figures out/ 5 6   (any of 2 3 4 5 6)
  python mppp_map.py --help          all sub-commands

Optional 1 m HiRISE DTM background: put DTM GeoTIFFs in ./dem/ (subfolders fine; only names with DTM or DEM are read),
or set M20_DEMS to a list of paths (';' on Windows, ':' on Linux/Mac). MSR TRN HiRISE DTM (doi:10.5066/P13CPYYU) is used first,
the Mars 2020 TRN HiRISE DTM (doi:10.5066/P9REJ9JN) fills in; the bundled ~70 m terrain tiles (./terrain) fill the rest.
  MSR mosaic: https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/MSR_TRN_HiRISE_soc_003_USGS_release_aug2024/MSR_TRN_HiRISE_soc_003_USGS_release_aug2024_mosaics.zip (5.3 GB)

Full documentation (block rules, every figure, sites file schema, proposed mppp.map API, open items): docs/mppp_map.md
