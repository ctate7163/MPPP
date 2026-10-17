"""
Blocks of rover stations: the waypoint-map pipeline and the site definitions (MPPP v0p80, mppp.maps).

    python scripts\\mppp_maps.py all      OUT [--waypoints FILE] [--recluster] [--dpi 450]
                                          blocks -> zooms (figure 1) -> figures 2-6 -> OUT\\sites_blocks.json
    python scripts\\mppp_maps.py blocks   OUT          the block list (OUT\\_m20_sites\\sites_zoom.json)
    python scripts\\mppp_maps.py zooms    OUT          figure 1, the +-30 m panels
    python scripts\\mppp_maps.py figures  OUT [2 3 4 5 6]
    python scripts\\mppp_maps.py export   OUT          OUT\\sites_blocks.json (sites v4)
    python scripts\\mppp_maps.py clusters OUT          raw 10 m clusters again (after a new waypoints file)
    python scripts\\mppp_maps.py merge    [--from sites_blocks.json] [--write]
                                          the blocks into src/mppp/data/sites.json (default: the packaged 16 Oct
                                          blocks; without --write a dry run into OUT\\sites_merged.json), checked
    python scripts\\mppp_maps.py compare  A.json B.json      how two block files differ

The pipeline is the prototype of 16 Oct (src/mppp/maps/m20_sites, documentation docs/mppp_map.md there), run on a
copy in OUT\\_m20_sites.  Figures need adjustText (pip install adjustText); 1 m HiRISE DTMs via M20_DEMS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "blocks", "zooms", "figures", "export", "clusters", "merge", "compare"])
    ap.add_argument("args", nargs="*", help="OUT [figure numbers] / compare: A B")
    ap.add_argument("--waypoints", help="a waypoint GeoJSON to use instead of the prototype's (sols 13-2009)")
    ap.add_argument("--legacy-sites", help="the legacy sites.json the added blocks come from (default the prototype's)")
    ap.add_argument("--recluster", action="store_true", help="run the raw clustering first (new waypoints)")
    ap.add_argument("--dpi", type=int, help="M20_DPI (default 300)")
    ap.add_argument("--from", dest="src", help="merge: the sites_blocks.json to merge (default the packaged one)")
    ap.add_argument("--write", action="store_true", help="merge: write src/mppp/data/sites.json")
    a = ap.parse_args(argv)
    from mppp import maps
    if a.cmd == "compare":
        if len(a.args) != 2:
            ap.error("compare A.json B.json")
        x, _ = maps.load_blocks_file(a.args[0])
        y, _ = maps.load_blocks_file(a.args[1])
        print("\n".join(maps.compare_blocks(x, y)))
        return 0
    out = Path(a.args[0]) if a.args else Path("maps_out")
    if a.cmd == "merge":
        import json
        import shutil
        from mppp.sfm.sites import SITES_FILE, validate_site_table
        sites, groups = maps.load_blocks(a.src)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "sites_merged.json"
        shutil.copy2(SITES_FILE, tmp)
        maps.merge_into_sites_file(tmp, sites, groups)
        err, warn = validate_site_table(tmp)
        print(f"{len(sites)} blocks merged -> {tmp}: {len(err)} errors, {len(warn)} warnings")
        for e in err:
            print("  ERROR", e)
        if a.write and not err:
            maps.merge_into_sites_file(SITES_FILE, sites, groups)
            print(f"written {SITES_FILE} (commit it; run_sites.py --group blocks_all runs them)")
        elif a.write:
            print("not written: fix the errors first")
            return 1
        json.loads(tmp.read_text(encoding="utf-8"))
        return 0
    maps.prototype.run(a.cmd, out, figures=tuple(a.args[1:]), waypoints=a.waypoints, legacy_sites=a.legacy_sites,
                       recluster=a.recluster, dpi=a.dpi)
    print(f"done: {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
