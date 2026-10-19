"""Tests of the waypoint-map study (studies/m20_sites): the packaged blocks, the prototype, the sites file v5 rules."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1]))              # the repository, for "studies.m20_sites"
import studies.m20_sites as maps  # noqa: E402


def test_maps_packaged_blocks_and_prototype(tmp_path):
    sites, groups = maps.load_blocks()
    assert len(sites) == 36 and len(groups["blocks_all"]) == 36
    b9 = sites["blk09_three_forks_north"]["block"]                        # 17 Oct: Christian's visits (413, 433-435)
    assert b9["n_navcam_360"] == 8 and 413 in b9["end_of_drive_sols"] and [413, 413] in b9["visits"]
    assert [433, 435] in b9["visits"] and sites["blk08_three_forks_south"]["block"]["visits"][0] == [413, 414]
    assert sites["blk14_observation_mountain"]["block"]["visits"][0][0] == 592
    assert sites["blk15_whale_mountain"]["block"]["visits"][0][0] == 606
    assert sites["blk35_hippo_pools"]["sols"] == [[1947, 1961], [2006, 2009]]
    out = maps.prototype.run("export", tmp_path / "out")                 # blocks are in the package; export only
    again, _ = maps.load_blocks_file(out / "sites_blocks.json")
    assert maps.compare_blocks(again, sites)[0].startswith("the same")
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
    assert len(ob) == 36
    for k, v in sites.items():
        b = ob[str(v["block"]["number"])]
        assert b["name"] == v["block"]["name"] and sorted(b["visits"]) == sorted(v["sols"])
        assert b["centre"] == v["block"]["centre"] and b["site"] in t["sites"]
        assert v["block"]["number"] in t["sites"][b["site"]]["official_blocks"]
    for k, v in t["sites"].items():                                       # Christian's v5 rules
        assert isinstance(v["sols"][0], list) and isinstance(v["notes"], str) and "note" not in v
        assert isinstance(v["official_blocks"], list) and v["max_waypoint_separation_m"] >= 0
    assert t["sites"]["observation_mountain"]["sols"] == [[592, 600]] and ob["14"]["site"] == "observation_mountain"
    assert ob["15"]["site"] == "whale_mountain" and t["sites"]["threeforks_large"]["official_blocks"] == [8, 9]
    assert t["sites"]["chal_rocks_large"]["rover_sites"] == [3, 4, 11]


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
    assert st["n_waypoints"] == 3 and 20 < st["max_waypoint_separation_m"] < 22 and st["rover_sites"] == [29, 30]
    assert official.update_waypoint_stats(t, wp) == []


