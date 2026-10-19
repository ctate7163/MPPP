"""
mppp.waypoints - where the rover was, and the sites (blocks of rover stations) MPPP aligns.

* :mod:`.table` - the Mars 2020 waypoint table (a GeoJSON snapshot ships with the package; ``load_waypoints(refresh=True)``
  downloads the current one into the user cache), look-ups by rover site/drive and by radius, flat rows and the
  map-frame <-> lon/lat relation;
* :mod:`.sites` - the site definitions (``mppp/data/sites.json``): sol ranges, Mastcam-Z zooms, groups, the WORK
  folder of a site and the discovery of finished alignments;
* :mod:`.blocks` - the official numbered blocks of stations in the site file (version 5) and the waypoint statistics
  of every site.

Nothing is read at import time.
"""
from .table import (cached_path, load_waypoints, lonlat_fit, lonlat_of, offset_lonlat, snapshot_path,  # noqa: F401
                    stations_near, waypoint_for_site_drive, waypoint_rows, waypoints_within_radius)
from .sites import (SITES_FILE, SITES_FILE_VERSION, DEFAULT_SCAPES_ROOT, block_sites, check_block_record, check_sites,  # noqa: F401
                    discover_scapes, folder_name, in_sol_ranges, load_site_ranges, load_site_table, load_sites,
                    parse_work_folder, scan_scapes, site_block, site_group, site_label, site_sol_ranges, site_zooms,
                    sol_ranges, sol_span, validate_site_table, work_folder, zcam_sites)
from .blocks import (order_by_first_sol, restructure, update_waypoint_stats, waypoint_points, waypoint_stats,  # noqa: F401
                     write_table)
