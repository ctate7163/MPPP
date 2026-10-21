"""Tests of the waypoint-map study (studies/m20_sites): the packaged blocks, the prototype, the sites file v5 rules."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))              # the repository, for "studies.m20_sites"
import studies.m20_sites as maps  # noqa: E402


def test_maps_packaged_blocks_and_prototype(tmp_path):
    sites, groups = maps.load_blocks()
    assert len(sites) == 36 and len(groups["blocks_all"]) == 36                # the prototype's own list (sites.json has 38 since 0.91)
    b9 = sites["blk09_three_forks_north"]["block"]                        # 17 Oct: Christian's visits (413, 433-435)
    assert b9["n_navcam_360"] == 8 and 413 in b9["end_of_drive_sols"] and [413, 413] in b9["visits"]
    assert [433, 435] in b9["visits"] and sites["blk08_three_forks_south"]["block"]["visits"][0] == [413, 414]
    assert sites["blk14_observation_mountain"]["block"]["visits"][0][0] == 592
    assert sites["blk15_whale_mountain"]["block"]["visits"][0][0] == 606
    assert sites["blk35_hippo_pools"]["sols"] == [[1947, 1961], [2006, 2009]]
    out = maps.prototype.run("export", tmp_path / "out")                 # blocks are in the package; export only
    again, _ = maps.load_blocks_file(out / "sites_blocks.json")
    # the export follows sites.json (38 blocks since 0.91, numbered by first sol), the prototype's own list has 36
    assert len(again) == 41 and any(k.endswith("rose_river_falls") for k in again) and any(k.endswith("olifants") for k in again)
    assert (tmp_path / "out" / "_m20_sites" / "mppp_map.py").is_file()


def test_maps_blocks_rerun_from_waypoints(tmp_path):
    out = maps.prototype.run("blocks", tmp_path / "out")
    sz = __import__("json").loads((out / "_m20_sites" / "sites_zoom.json").read_text(encoding="utf-8"))
    assert len(sz) == 36


def test_sites_json_holds_the_official_blocks():
    from mppp.waypoints.sites import load_site_table
    t = load_site_table()
    assert not any(k.startswith("blk") for k in t["sites"])               # v0p81: no blk.. sites
    sites, _ = maps.load_blocks()
    ob = t["official_blocks"]
    assert len(ob) == 41                                     # 0.91: + Rose River Falls, Olifants, Plankeholmane, Seitah Overlook, Hans Amundsen
    assert ob["13"]["site"] == "rose_river_falls" and ob["13"]["mastcam_z_360"] == [17] and ob["38"]["site"] == "olifants" and ob["35"]["site"] == "plankeholmane"
    assert ob["17"]["name"] == "Hidden Harbor" and ob["17"]["site"] == "hidden_harbor"
    assert ob["4"]["site"] == "seitah_overlook" and ob["22"]["site"] == "hans_amundsen_workspace"
    by_centre = {(round(b["centre"]["easting"], 1), round(b["centre"]["northing"], 1)): (int(n), b) for n, b in ob.items()}
    for k, v in sites.items():                                            # the prototype's 36 blocks are all there
        c = v["block"]["centre"]
        n, b = by_centre[(round(c["easting"], 1), round(c["northing"], 1))]
        assert (b["name"] == v["block"]["name"] or k == "blk15_whale_mountain") and sorted(b["visits"]) == sorted(v["sols"])
        assert b["site"] in t["sites"] and n in t["sites"][b["site"]]["official_blocks"]
    for k, v in t["sites"].items():                                       # Christian's v5 rules
        assert isinstance(v["sols"][0], list) and isinstance(v["notes"], str) and "note" not in v
        assert isinstance(v["official_blocks"], list) and v["max_waypoint_separation_m"] >= 0
    assert t["sites"]["observation_mountain"]["sols"] == [[592, 600]] and ob["16"]["site"] == "observation_mountain"
    assert t["sites"]["threeforks_large"]["official_blocks"] == [9, 10]
    assert {3, 4, 11} <= set(t["sites"]["chal_rocks_large"]["rover_sites"])      # 0.90: + the nearby waypoints' sites


def test_official_restructure_v4_to_v5(tmp_path):
    import json as _json
    from mppp.waypoints import blocks as official
    v4 = {"version": 4, "sites": {"a": {"sols": [72, 100], "label": "A", "note": "x"},
                                  "b": {"sols": [[361, 378], [91, 101]], "label": "B"},
                                  "blk03_chal_rocks": {"sols": [[72, 101]], "label": "C", "block": {"number": 3}},
                                  "r": {"sols": [5, 3], "label": "reversed"}},
          "groups": {"g": ["b", "old"], "blocks_all": ["blk03_chal_rocks"]}}
    blocks, _ = maps.load_blocks()
    wp = HERE / "m20_sites" / "M20_waypoints.json"
    t, rep = official.restructure(v4, blocks, wp, {"old": "a"})
    assert t["version"] == 5 and "blk03_chal_rocks" not in t["sites"] and len(t["official_blocks"]) == 36
    assert t["sites"]["a"]["sols"] == [[72, 100]] and t["sites"]["a"]["notes"] == "x" and "note" not in t["sites"]["a"]
    assert t["sites"]["r"]["sols"] == [[3, 5]] and any("reversed" in x for x in rep)
    assert t["sites"]["b"]["official_blocks"] == [3] and t["groups"]["g"] == ["b", "a"]
    assert all(v["notes"] for v in t["sites"].values()) and len(t["groups"]["blocks_all"]) == 36
    again, rep2 = official.restructure(t, blocks, wp)                     # version 5 in, the same out
    assert again["sites"] == t["sites"] and again["official_blocks"] == t["official_blocks"]
    f = official.write_table(t, tmp_path / "s.json")
    assert _json.loads(f.read_text())["official_blocks"]["3"]["name"] == "Chal Rocks"
    W = official.waypoint_points(wp)
    st = official.waypoint_stats([[592, 593]], W)
    assert st["n_primary"] == 3 and st["n_waypoints"] >= 3 and st["rover_sites"][:2] == [29, 30]   # 0.90: + the nearby ones
    assert official.update_waypoint_stats(t, wp) == []


