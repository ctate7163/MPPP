"""Product selection and waypoints (mppp.select, waypoints, sfm select_best_products). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import numpy as np
import pytest
from conftest import synthetic_waypoints
import json
from pathlib import Path


def test_dedupe_keeps_highest_version_and_never_deletes(tmp_path):
    from mppp.select import dedupe_by_version, find_imgs
    a = tmp_path / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ01.IMG"
    b = tmp_path / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ02.IMG"
    c = tmp_path / "ZL0_0710_0729888971_069RAD_N0332864ZCAM07114_1100LMA01.IMG"
    for p in (a, b, c):
        p.write_bytes(b"x")
    kept, dropped = dedupe_by_version([a, b, c])
    assert kept == [b, c] and dropped == [a]
    assert a.exists()                                            # archive untouched
    assert find_imgs(tmp_path, ["ZL0"], sequ_id="_110") == [c]
    assert find_imgs(tmp_path, ["ZL0"], sequ_id="_034") == []
    assert find_imgs(tmp_path, ["NLF", "ZL0"], sol_range=(709, 709)) == [a, b]


def test_waypoint_lookup_and_radius():
    from mppp.waypoints import waypoint_for_site_drive, waypoints_within_radius, offset_lonlat
    wp = synthetic_waypoints()
    assert waypoint_for_site_drive(wp, 33, 1000)["properties"]["sol"] == 705
    assert waypoint_for_site_drive(wp, 33, 900)["properties"]["drive"] == 1000     # nearest
    assert waypoint_for_site_drive(wp, 33, 900, exact=True) is None
    assert waypoint_for_site_drive(wp, 99, 0) is None
    assert len(waypoints_within_radius(wp, 13, 10.0)) == 2
    assert len(waypoints_within_radius(wp, 20, 10.0)) == 2          # rover parked since sol 14
    lon, lat = offset_lonlat(77.0, 18.0, d_east=0.0, d_north=3396190.0 * np.pi / 180)
    assert (lon, lat) == (pytest.approx(77.0), pytest.approx(19.0))


def test_waypoints_snapshot_cache_and_refresh(tmp_path, monkeypatch):
    from mppp import waypoints as W
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path))
    wp = W.load_waypoints()
    assert wp["_mppp_source"]["packaged_snapshot"] and wp["_mppp_source"]["path"] == str(W.snapshot_path())
    src = tmp_path / "remote.json"
    src.write_text(json.dumps({"type": "FeatureCollection", "features": [{"properties": {"site": 1, "drive": 0}}]}))
    wp = W.load_waypoints(refresh=True, url=src.as_uri())
    assert wp["_mppp_source"]["path"] == str(tmp_path / "M20_waypoints.json") and wp["_mppp_source"]["n_features"] == 1
    assert W.load_waypoints()["_mppp_source"]["n_features"] == 1                # the cache now wins
    with pytest.warns(UserWarning, match="could not download"):
        assert W.load_waypoints(refresh=True, url=(tmp_path / "missing.json").as_uri())["_mppp_source"]["n_features"] == 1
    from mppp.error.waypoints import load_featurecollection                     # one loader for both
    assert len(load_featurecollection()["features"]) == 1


def test_select_best_products_several_sequences():
    from mppp.sfm.project import select_best_products
    names = ["NLF_0770_0736000000_000RAD_N0390000NCAM00500_0A0195J01.IMG",
             "ZL0_0770_0736000100_000RAD_N0390000ZCAM08000_0340LMA01.IMG",
             "NLF_0770_0736000200_000RAD_N0390000SAPP00500_0A0195J01.IMG"]
    kept, rep = select_best_products([Path(n) for n in names], sizes={n: 1 for n in names},
                                     sequence_prefix=("NCAM", "ZCAM"))
    assert [p.name for p in kept] == names[:2] and rep["n_dropped_sequence"] == 1
    kept, _ = select_best_products([Path(n) for n in names], sizes={n: 1 for n in names})
    assert [p.name for p in kept] == names[:1]                                           # default: NCAM only


def _wp(site, drive, sol, e, n):
    return {"type": "Feature", "properties": {"site": site, "drive": drive, "sol": sol, "easting": e, "northing": n}}


def test_stations_near_and_find_imgs_near(tmp_path):
    from mppp.waypoints import stations_near
    from mppp.select import find_imgs_near
    wps = {"features": [_wp(26, 500, 461, 1000.0, 2000.0), _wp(26, 600, 470, 1010.0, 2000.0),
                        _wp(30, 100, 600, 1003.0, 2004.0),       # 5.0 m from S026D0500: a later visit
                        _wp(30, 200, 610, 1016.0, 2000.0),       # 6 m from S026D0600: outside 5 m
                        _wp(31, 0, 700, 2000.0, 2000.0)]}
    rows = stations_near(wps, [(26, 500), (26, 600)], 5.0)
    keys = {(r["site"], r["drive"]): r for r in rows}
    assert set(keys) == {(26, 500), (26, 600), (30, 100)}
    assert keys[(30, 100)]["distance_m"] == pytest.approx(5.0) and keys[(30, 100)]["nearest"] == [26, 500]
    assert not keys[(30, 100)]["anchor"] and keys[(26, 500)]["anchor"]
    assert {(r["site"], r["drive"]) for r in stations_near(wps, [(26, 500), (26, 600)], 6.5)} >= {(30, 200)}
    # an archive of empty .IMG files named like PDS products
    def name(cam, sol, site, drive):
        return f"{cam}_{sol:04d}_0700000000_000RAD_N{site:03d}{drive:04d}NCAM00100_0A0095J01.IMG"
    for cam, sol, site, drive in [("NLF", 461, 26, 500), ("NRF", 461, 26, 500), ("NLF", 470, 26, 600),
                                  ("NLF", 600, 30, 100), ("NLF", 610, 30, 200), ("NLF", 700, 31, 0)]:
        (tmp_path / name(cam, sol, site, drive)).write_bytes(b"")
    paths, rep = find_imgs_near(tmp_path, ["NLF", "NRF"], (455, 480), wps, radius_m=5.0)
    names = sorted(p.name for p in paths)
    assert len(names) == 4 and any("_0600_" in n for n in names) and not any("_0610_" in n for n in names)
    assert rep["n_in_range"] == 3 and rep["n_added"] == 1
    assert rep["stations_added"] == [{"station": "S030D0100", "distance_m": 5.0, "nearest": "S026D0500", "images": 1,
                                      "sols": [600, 600]}]
    paths0, rep0 = find_imgs_near(tmp_path, ["NLF", "NRF"], (455, 480), wps, radius_m=None)
    assert len(paths0) == 3 and rep0["n_added"] == 0
    # v0p53.1: Mastcam-Z with no image in the sol range still gets the nearby visit, from the Navcam stations
    zname = "ZL0_0600_0700000000_000RAD_N0300100ZCAM00100_0340LMJ01.IMG"
    (tmp_path / zname).write_bytes(b"")
    zp, _ = find_imgs_near(tmp_path, ["ZL0"], (455, 480), wps, radius_m=5.0, sequ_id="_034")
    assert zp == []                                                     # no Mastcam-Z station in range: nothing
    zp, zr = find_imgs_near(tmp_path, ["ZL0"], (455, 480), wps, radius_m=5.0, sequ_id="_034",
                            anchor_stations=rep["stations_in_range"])
    assert [p.name for p in zp] == [zname] and zr["stations_added"][0]["station"] == "S030D0100"


def test_select_best_products_leaves_out_navcam_tiles(tmp_path, monkeypatch):
    from mppp.sfm import project as P
    tile = tmp_path / "NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    full = tmp_path / "NRF_0092_0675115592_000RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    half = tmp_path / "NLF_0092_0675110134_691RAD_N0040136NCAM00507_0A0295J01.IMG"
    for f in (tile, full, half):
        f.write_bytes(b"")
    sizes = {tile.name: 7426560, full.name: 118046720, half.name: 1774080}
    fractions = {tile.name: 1 / 16, half.name: 1.0}
    calls = []

    def fake(path, size=None):
        calls.append(Path(path).name)
        return fractions.get(Path(path).name)
    monkeypatch.setattr(P, "frame_fraction", fake)
    kept, rep = P.select_best_products([tile, full, half], sizes=sizes)
    assert {p.name for p in kept} == {full.name, half.name}
    assert rep["n_dropped_subframes"] == 1 and rep["dropped_subframes"][0]["file"] == tile.name
    assert rep["n_superseded"] == 0
    kept, rep = P.select_best_products([tile, full, half], sizes=sizes, min_frame_fraction=None)
    assert len(kept) == 3 and rep["n_dropped_subframes"] == 0


def test_frame_fraction_reads_labels_only_for_small_files(tmp_path, monkeypatch):
    from mppp.sfm import project as P
    import mppp.labels as L
    big = tmp_path / "NRF_0092_0675115592_000RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    big.write_bytes(b"")
    monkeypatch.setattr(L, "read_pds", lambda *a, **k: pytest.fail("label read for a full-size file"))
    assert P.frame_fraction(big, size=118046720) is None
    tile = tmp_path / "NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    tile.write_bytes(b"")
    monkeypatch.setattr(L, "read_pds", lambda *a, **k: ({"IMAGE": {"LINES": 960, "LINE_SAMPLES": 1280}}, None))
    assert P.frame_fraction(tile, size=7426560) == pytest.approx(1 / 16)
    half = tmp_path / "NLF_0092_0675110134_691RAD_N0040136NCAM00507_0A0195J01.IMG"   # downsample 1: 2560 x 1920
    half.write_bytes(b"")
    monkeypatch.setattr(L, "read_pds", lambda *a, **k: ({"IMAGE": {"LINES": 1920, "LINE_SAMPLES": 2560}}, None))
    assert P.frame_fraction(half, size=1774080) == pytest.approx(1.0)
    zcam = tmp_path / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"
    zcam.write_bytes(b"")
    kept, rep = P.select_best_products([zcam], sizes={zcam.name: 100}, sequence_prefix=("ZCAM",))
    assert len(kept) == 1 and rep["n_dropped_subframes"] == 0            # Mastcam-Z is not filtered


def test_real_tile_label_if_staged():
    f = Path("/mnt/user-data/uploads/m2020/datadrive/00092/ids/rdr/ncam/"
             "NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01.IMG")
    if not f.is_file():
        pytest.skip("tile product not available here")
    from mppp.sfm.project import frame_fraction
    assert frame_fraction(f) == pytest.approx(1 / 16)


def test_quarter_frame_rule(tmp_path, monkeypatch):
    from mppp.sfm import project as P
    assert P.NAVCAM_MIN_FRAME_FRACTION == 0.25
    names = {"a": "NLF_0049_0000000001_000RAD_N0010000NCAM00603_0A00LLJ01.IMG",     # 1280x960 full-res tile, 1/16
             "b": "NLF_0049_0000000002_000RAD_N0010000NCAM00500_0A0295J01.IMG",     # quarter-res strip 1280x224
             "c": "NLF_0049_0000000003_000RAD_N0010000NCAM00414_0A00LLJ01.IMG",     # exactly 1/4: kept
             "d": "NLF_0049_0000000004_000RAD_N0010000NCAM00415_0A0195J01.IMG"}     # half-res full frame
    frac = {names["a"]: 1 / 16, names["b"]: 0.2333, names["c"]: 0.25, names["d"]: None}
    paths = []
    for n in names.values():
        (tmp_path / n).write_bytes(b"")
        paths.append(tmp_path / n)
    monkeypatch.setattr(P, "frame_fraction", lambda p, size=None: frac[Path(p).name])
    kept, rep = P.select_best_products(paths, sizes={n: 1 for n in names.values()})
    assert {p.name for p in kept} == {names["c"], names["d"]} and rep["n_dropped_subframes"] == 2
    from mppp.runner import PROCESS_RULES
    assert PROCESS_RULES == 3
