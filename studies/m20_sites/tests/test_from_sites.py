"""The figures' block list comes from the packaged sites.json (official_blocks, version 5)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "m20_sites"
sys.path.insert(0, str(HERE))


def test_blocks_from_sites_packaged():
    from from_sites import blocks_from_sites, sites_file
    path = sites_file()
    doc, blocks = blocks_from_sites(path)
    assert len(blocks) == len(doc["official_blocks"])
    assert [b["id"] for b in blocks] == sorted(b["id"] for b in blocks)
    for b in blocks:
        assert b["runs"] == sorted(b["runs"]) and b["first"] <= b["runs"][0][0]   # a visit left out of the figures only (MAP_OMIT_VISITS) keeps first
        assert b["region"] in {"crater_rim_west", "crater_rim", "margin_unit_neretva_vallis", "upper_fan_belva",
                               "delta_front", "crater_floor_seitah"}


def test_blocks_from_sites_fixture(tmp_path):
    from from_sites import blocks_from_sites
    doc = {"version": 5, "official_blocks": {
        "2": {"name": "B", "region": "crater_rim", "centre": {"easting": 10.0, "northing": 20.0}, "visits": [[300, 310], [100, 120]],
              "n_navcam_360": 3, "mastcam_z_360": [4, 5], "source_key": "b"},
        "1": {"name": "A", "region": "delta_front", "centre": {"easting": 1.0, "northing": 2.0}, "visits": [[50, 60]], "site": "a"}}}
    p = tmp_path / "sites.json"; p.write_text(json.dumps(doc))
    _, blocks = blocks_from_sites(str(p))
    assert [b["id"] for b in blocks] == [1, 2]
    assert blocks[1]["runs"] == [[100, 120], [300, 310]] and blocks[1]["first"] == 100 and blocks[1]["nz_sites_json"] == 2 and "members" in blocks[1]   # nz: counted from the block's waypoints (20 Oct)
    assert blocks[0]["key"] == "a" and blocks[0]["n360"] == 0
