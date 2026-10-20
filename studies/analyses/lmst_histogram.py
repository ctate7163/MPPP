"""
Time of day (LMST) of the processed images, per site: one histogram panel per site plus one for all sites, Navcam and
Mastcam-Z 34 mm each normalised to unit area, and a table per site.

    python studies/analyses/lmst_histogram.py                           # MPPP_SCAPES
    python studies/analyses/lmst_histogram.py --roots ROOT --sites rockytop sid threeforks_south
    python studies/analyses/lmst_histogram.py --out OUT --name lmst_2026-09-30

Images come from the newest ``processed/mppp_manifest_v*.json`` of every WORK folder (``<site>_colmap``,
``<site>_colmap_nav_zcam34``) under ``--roots``; a folder name found under several roots counts once (its newest
manifest); a site is the union of its folders, each image counted once.  LMST is per image from its label (``LOCAL_MEAN_SOLAR_TIME``).  The dotted lines mark the
processing window ``selection.lmst_window_h`` (images outside it are not processed since v0p30; older manifests
may still hold some).  Navcam tiles below ``--min-frame-fraction`` of the frame (default: the current selection
rule, 1/5 since v0p65) are left out, as processing does.  Writes ``<out>/<name>.png`` and ``<out>/<name>_by_site.csv``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 0.90: the figure lives in the package (mppp.waypoints.blocks: lmst_images, lmst_site_row, plot_lmst_histograms,
# write_lmst_table); this script is its command line.
from mppp.waypoints.blocks import lmst_hours, lmst_images, lmst_site_row, plot_lmst_histograms, write_lmst_table  # noqa: F401


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[os.environ.get("MPPP_SCAPES", ".")],
                    help="folders holding the WORK folders")
    ap.add_argument("--sites", nargs="+", default=None, help="only these sites")
    ap.add_argument("--out", default=".", help="output folder")
    ap.add_argument("--name", default="lmst_histogram", help="file name stem")
    ap.add_argument("--min-frame-fraction", type=float, default=None,
                    help="leave out Navcam tiles below this fraction of the frame, as the current selection does "
                         "(default: select_best_products' NAVCAM_MIN_FRAME_FRACTION; 0: keep all)")
    a = ap.parse_args(argv)
    from mppp.config import default_config
    from mppp.sfm.project import NAVCAM_MIN_FRAME_FRACTION
    mff = NAVCAM_MIN_FRAME_FRACTION if a.min_frame_fraction is None else a.min_frame_fraction
    images, used = lmst_images(a.roots, a.sites, mff or None)
    if not images:
        print("no processed images found under", a.roots)
        return 1
    rows = sorted((lmst_site_row(k, list(v.values())) for k, v in images.items()), key=lambda r: r["first_sol"])
    window = (default_config().get("selection") or {}).get("lmst_window_h")
    out = Path(a.out)
    nN, nZ = plot_lmst_histograms(images, rows, out / f"{a.name}.png", tuple(window) if window else None)
    write_lmst_table(rows, out / f"{a.name}_by_site.csv")
    for site, d, m, n, n_small in used:
        print(f"  {site:22s} {n:5d} images  {m}  {d}" + (f"  ({n_small} tiles below {mff:g} of the frame left out)"
                                                       if n_small else ""))
    print(f"wrote {out / (a.name + '.png')} and {a.name}_by_site.csv: {len(rows)} sites, {nN} Navcam + {nZ} Mastcam-Z")
    return 0


if __name__ == "__main__":
    sys.exit(main())
