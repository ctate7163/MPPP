"""
Blocks of rover stations: from the waypoints to the site definitions and maps (MPPP v0p80, mppp.maps).

    python scripts\\mppp_maps.py blocks  [--waypoints FILE] [--zcam360 CSV]        the block list
    python scripts\\mppp_maps.py figures [--waypoints FILE] --out DIR [--dem DEM]   traverse map + zoom grid
    python scripts\\mppp_maps.py export  [--waypoints FILE] --out DIR              DIR/sites_blocks.json (sites v4)
    python scripts\\mppp_maps.py merge   (--from sites_blocks.json | --waypoints FILE) [--write]
                                         the blocks into src/mppp/data/sites.json (--write; else a dry run), then
                                         check_sites; --compare FILE shows how two block files differ
    python scripts\\mppp_maps.py all     [--waypoints FILE] --out DIR              blocks, figures, export

--waypoints default: the cached waypoints (mppp.load_waypoints; refresh with mppp.load_waypoints(refresh=True)), else the
packaged snapshot (sols to 1962).  The fresh file of 15 Oct (M20_waypoints_sol2009_all) gives Hippo Pools and Marble
Mountain their full counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["blocks", "figures", "export", "merge", "all"])
    ap.add_argument("--waypoints")
    ap.add_argument("--zcam360", help="CSV: number, sol, name of the numbered Mastcam-Z 360s (Z× counts)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--dem", help="a DEM GeoTIFF in the waypoint frame for the traverse map (needs rasterio)")
    ap.add_argument("--from", dest="src", help="merge: a sites_blocks.json to merge instead of deriving the blocks")
    ap.add_argument("--compare", help="merge: a second sites_blocks.json to compare with")
    ap.add_argument("--write", action="store_true", help="merge: write src/mppp/data/sites.json")
    a = ap.parse_args(argv)
    from mppp import maps
    out = Path(a.out)
    res = None
    if a.cmd != "merge" or not a.src:
        res = maps.build_from_waypoints(a.waypoints, a.zcam360)
        print(f"waypoints {res['waypoints']}: sols {res['sol_range'][0]}-{res['sol_range'][1]}, {len(res['rows'])} "
              f"waypoints, {len(res['clusters'])} clusters of >= {maps.blocks.MIN_STOPS}")
    if a.cmd in ("blocks", "all"):
        print("\n".join(maps.block_lines(res["blocks"])))
    if a.cmd in ("figures", "all"):
        import matplotlib.pyplot as plt
        out.mkdir(parents=True, exist_ok=True)
        plt.close(maps.figures.traverse_map(res["rows"], res["blocks"], out / "m20_blocks_traverse.png", dem=a.dem))
        plt.close(maps.figures.zoom_grid(res["rows"], res["blocks"], out / "m20_block_zooms.png"))
        print(f"figures in {out}")
    if a.cmd in ("export", "all"):
        out.mkdir(parents=True, exist_ok=True)
        maps.write_sites({"_about": "MPPP blocks of rover stations (mppp.maps, scripts/mppp_maps.py export)",
                          "version": 4, "waypoints": res["waypoints"], "sites": res["sites"], "groups": res["groups"]},
                         out / "sites_blocks.json")
        (out / "blocks.json").write_text(json.dumps(res["blocks"], indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"written {out / 'sites_blocks.json'} ({len(res['sites'])} blocks) and blocks.json")
    if a.cmd == "merge":
        from mppp.sfm.sites import SITES_FILE, load_site_table, validate_site_table
        sites, groups = maps.load_blocks_file(a.src) if a.src else (res["sites"], res["groups"])
        if a.compare:
            other, _ = maps.load_blocks_file(a.compare)
            print("\n".join(maps.compare_blocks(sites, other)))
        merged = maps.merge_sites(load_site_table(), sites, groups)
        tmp = out / "sites_merged.json"
        out.mkdir(parents=True, exist_ok=True)
        maps.write_sites(merged, tmp)
        err, warn = validate_site_table(tmp)
        print(f"{len(sites)} blocks merged -> {tmp}: {len(err)} errors, {len(warn)} warnings")
        for e in err:
            print("  ERROR", e)
        if a.write and not err:
            maps.export.merge_into_sites_file(SITES_FILE, sites, groups)
            print(f"written {SITES_FILE} (commit it; run_sites.py --group blocks_all runs them)")
        elif a.write:
            print("not written: fix the errors first")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
