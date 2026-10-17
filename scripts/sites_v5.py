"""
sites.json version 5 (MPPP v0p81, 17 Oct 2026): the official numbered blocks as a table, no blk.. sites (mppp.maps.official).

    python scripts\\sites_v5.py restructure SRC.json [--write]     a version-4 file -> version 5 (dry run: OUT\\sites_v5.json)
    python scripts\\sites_v5.py waypoints  [--file F] [--refresh] [--write]
                                   the waypoint fields (max_waypoint_separation_m, n_waypoints, rover_sites) again from
                                   the newest waypoints file: --refresh downloads it from JPL into the user cache, and
                                   a newer file than the packaged one replaces src/mppp/data/M20_waypoints.json and
                                   src/mppp/maps/m20_sites/M20_waypoints.json
    python scripts\\sites_v5.py show [SITE]   the official blocks (number, site, visits, max separation, rover sites)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 17 Oct 2026: sites renamed in sites.json (old -> new), applied to the groups
RENAMES = {"van_zyl": "van_zyl_overlook", "chal_rocks_sid": "chal_rocks", "pearce_canyon": "cheyava_falls",
           "chal_rocks_all": "chal_rocks_large", "taylorfjellet_all": "taylorfjellet_large",
           "marble_mountain_all": "marble_mountain"}
PROTO_WP = ROOT / "src/mppp/maps/m20_sites/M20_waypoints.json"
PKG_WP = ROOT / "src/mppp/data/M20_waypoints.json"


def _last_sol(path) -> int:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return max(int((f.get("properties") or {}).get("sol") or 0) for f in d["features"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["restructure", "waypoints", "show"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--file", help="waypoints: a waypoint GeoJSON to use")
    ap.add_argument("--refresh", action="store_true", help="waypoints: download the current file from JPL first")
    ap.add_argument("--out", default="maps_out")
    a = ap.parse_args(argv)
    from mppp.maps import load_blocks, official
    from mppp.sfm.sites import SITES_FILE, validate_site_table
    if a.cmd == "show":
        t = json.loads(SITES_FILE.read_text(encoding="utf-8"))
        for n, b in sorted(t["official_blocks"].items(), key=lambda x: int(x[0])):
            if a.args and b["site"] not in a.args:
                continue
            print(f"{int(n):3d} {b['name']:22s} {b['site']:24s} {str(b['visits']):44s} "
                  f"{b['max_waypoint_separation_m']:7.1f} m  rover sites {b['rover_sites']}")
        return 0
    if a.cmd == "restructure":
        src = Path(a.args[0]) if a.args else SITES_FILE
        t = json.loads(src.read_text(encoding="utf-8"))
        blocks, _ = load_blocks()
        new, rep = official.restructure(t, blocks, PROTO_WP, RENAMES)
        z = new["groups"].get("zcam110_consensus") or []          # chal_rocks has no 110 mm: its large block has
        if "chal_rocks" in z and 110 not in (new["sites"]["chal_rocks"].get("zcam") or []):
            new["groups"]["zcam110_consensus"] = ["chal_rocks_large" if m == "chal_rocks" else m for m in z]
            rep.append("group zcam110_consensus: chal_rocks (no 110 mm) -> chal_rocks_large")
        print("\n".join(rep))
    else:
        new = json.loads(SITES_FILE.read_text(encoding="utf-8"))
        wp = Path(a.file) if a.file else None
        if a.refresh:
            from mppp.waypoints import cached_path, load_waypoints
            load_waypoints(refresh=True)
            wp = cached_path() if cached_path().is_file() else wp
        wp = wp or PROTO_WP
        last, have = _last_sol(wp), _last_sol(PROTO_WP)
        print(f"waypoints {wp}: to sol {last} (packaged: sol {have})")
        if last > have and a.write:
            for dst in (PROTO_WP, PKG_WP):
                shutil.copy2(wp, dst)
            print(f"  newer: copied to {PROTO_WP} and {PKG_WP} (run scripts/mppp_maps.py all OUT --recluster to "
                  f"see whether the blocks change)")
        print("\n".join(official.update_waypoint_stats(new, wp)) or "  the waypoint fields are unchanged")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = official.write_table(new, out / "sites_v5.json")
    err, warn = validate_site_table(tmp)
    print(f"-> {tmp}: {len(err)} errors, {len(warn)} warnings")
    for e in err:
        print("  ERROR", e)
    if a.write and not err:
        official.write_table(new, SITES_FILE)
        print(f"written {SITES_FILE}")
    elif a.write:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
