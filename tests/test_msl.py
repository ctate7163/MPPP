"""0.99.4 (Christian, 16 Nov 2026: "Prepare a test msl block"): the MSL first-block study's names, archive listing and
block choice (rpp_studies.msl.first_block); the products themselves are tested on the PC."""
from __future__ import annotations

import numpy as np

from rpp_studies.msl import first_block as F


def test_msl_names_lmst_and_what_is_fetched():
    f = F.parse_name("NLB_668441715RAD_F0870792NCAM00252M1.IMG")
    assert f["camera"] == "NLB" and f["family"] == "N" and f["eye"] == "L" and f["side"] == "B" and f["sclk"] == 668441715
    assert f["product_type"] == "RAD" and f["sampling"] == "F" and (f["site"], f["drive"]) == (87, 792) and f["sequence"] == "NCAM00252"
    assert f["group"] == "NLB" and F.parse_name("NRB_668441715RAD_F0870792NCAM00252M1")["pair_key"] == f["pair_key"]
    assert F.parse_name("FLB_668441382RAD_F0870792FHAZ00302M1")["family"] == "F"
    assert F.parse_name("NLF_0052_0671954445_000RAD_N0032046NCAM00500_0A00LLJ01") is None        # an M2020 name
    assert F.parse_name("1408ML0068980090601986C00_DRCL") is None                                 # Mastcam
    assert abs(F.lmst_hours("Sol-03052M15:04:27.460") - (15 + 4 / 60 + 27.46 / 3600)) < 1e-9 and F.lmst_hours(None) is None
    names = ["NLB_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715RAD_F0870792NCAM00252M1.LBL",
             "NRB_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715MXY_F0870792NCAM00252M1.IMG",
             "NLB_668441715XYZ_F0870792NCAM00252M1.IMG", "NLB_670399104RAD_S0871078NCAM00263M1.IMG",     # XYZ, subframe: no
             "NLA_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715RAD_F0880792NCAM00252M1.IMG",     # side A, site 88: no
             "NLB_668441999MXY_F0870792NCAM00252M1.IMG"]                                                 # MXY of no RAD: no
    assert F.wanted(names, 87, "B") == ["NLB_668441715RAD_F0870792NCAM00252M1", "NRB_668441715RAD_F0870792NCAM00252M1",
                                         "NLB_668441715MXY_F0870792NCAM00252M1"]
    assert F.wanted(names, 87, "B", drives=[1078]) == []


def test_the_block_is_the_richest_station_and_its_neighbors():
    rows = []
    for (site, drive, pairs, x) in ((87, 0, 10, 0.0), (87, 792, 12, 30.0), (87, 1078, 9, 200.0), (87, 1200, 3, 10.0)):
        for k in range(pairs):
            for eye in "LR":
                rows.append({"site": site, "drive": drive, "eye": eye, "sol": 3047 + drive // 100, "pair_key": f"{drive}_{k}",
                             "C_enu_m": [x + (0.2 if eye == "R" else 0.0), 0.0, 1.9]})
    frames, rep = F.choose_block(rows, min_pairs=8, radius_m=50.0)
    assert rep["anchor"] == "S087D0792" and {(r["site"], r["drive"]) for r in frames} == {(87, 0), (87, 792)}
    st = {s["station"]: s for s in rep["stations"]}
    assert st["S087D1078"]["in_block"] is False and st["S087D1200"]["in_block"] is False and st["S087D0000"]["pairs"] == 10
    frames, rep = F.choose_block(rows, min_pairs=8, radius_m=500.0, station=(87, 1078))
    assert rep["anchor"] == "S087D1078" and len({r["drive"] for r in frames}) == 3
    a = np.linspace(100, 6000, 1000).reshape(10, 100)
    s = F.stretch8(a)
    assert s.dtype == np.uint8 and s.min() == 0 and s.max() == 255
    pairs = F.stereo_baselines([{"pair_key": "p", "eye": "L", "image": "l.png", "C_enu_m": [0, 0, 0]},
                                {"pair_key": "p", "eye": "R", "image": "r.png", "C_enu_m": [0.424, 0, 0]}],
                               {"l.png": np.zeros(3), "r.png": np.array([0.4, 0.0, 0.0])})
    assert len(pairs) == 1 and abs(pairs[0]["ratio"] - 0.4 / 0.424) < 1e-12


def test_archive_listings_with_relative_and_absolute_links(monkeypatch):
    page = ('<a href="../">Parent</a> <a href="NLB_668441715RAD_F0870792NCAM00252M1.IMG">x</a> '
            "<a href='/data/msl/MSLNAV_1XXX/DATA/SOL03052/NLB_668441715RAD_F0870792NCAM00252M1.LBL'>y</a> "
            '<a href="?C=M;O=A">Last modified</a> <a href="SUB/">sub</a>')
    monkeypatch.setattr(F, "_get", lambda url, timeout=120.0: page.encode())
    assert F.listing("https://example/") == ["NLB_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715RAD_F0870792NCAM00252M1.LBL"]
    url, names, tried = F.find_sol_dir(3052, bases=("https://a/MSLNAV_1XXX",))
    assert url == "https://a/MSLNAV_1XXX/DATA/SOL03052/" and len(names) == 2 and len(tried) == 1
