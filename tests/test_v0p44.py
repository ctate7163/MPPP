"""MPPP v0p44: Navcam tiles below 1/4 of the frame, thermal-stage defaults (5 degC, 5 images) with cx, cy shown per
bin and THERMAL_FREE, the LMST histogram script."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_version():
    import mppp
    assert mppp.__version__ == "0.44.0"


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


def test_thermal_defaults_and_notebook():
    import inspect
    from mppp.sfm import thermal as T
    from mppp.sfm.reconstruction import reconstruct
    assert T.THERMAL_BIN_DEG == 5.0 and T.THERMAL_MIN_IMAGES == 5
    sig = inspect.signature(T.thermal_stage).parameters
    assert sig["bin_deg"].default == 5.0 and sig["min_images"].default == 5
    assert inspect.signature(reconstruct).parameters["thermal_min_images"].default == 5
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    ns = {}
    params = next("".join(c["source"]) for c in nb["cells"] if "parameters" in c.get("metadata", {}).get("tags", []))
    assert "THERMAL_BINS_DEG  = 5 " in params and "THERMAL_MIN_IMAGES = 5 " in params
    assert 'THERMAL_FREE = ("fx", "fy")' in params and "thermal_free=THERMAL_FREE" in src


def test_temperature_bins_merge_as_documented():
    from mppp.sfm.thermal import temperature_bins
    # 5 degC bins: -20..-15 (6 images), -15..-10 (2 images), -5..0 (8 images)
    T = {1: -19.0, 2: -18.0, 3: -17.0, 4: -12.0, 5: -3.0, 6: -2.0, 7: -1.5, 8: -1.0}
    n = {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}
    b = temperature_bins(T, n, 5.0, 5)
    assert b[4] == (-20.0, -10.0) and b[1] == (-20.0, -10.0)       # the small bin joins its nearer neighbour
    assert b[5] == (-5.0, 0.0)
    # across an empty bin: -25..-20 (2 images) joins -15..-10 (the next occupied one)
    b = temperature_bins({1: -22.0, 2: -12.0, 3: -11.0, 4: -13.0}, {1: 2, 2: 2, 3: 2, 4: 2}, 5.0, 5)
    assert set(b.values()) == {(-25.0, -10.0)}
    # a tie goes to the colder neighbour
    b = temperature_bins({1: -17.0, 2: -12.0, 3: -7.0}, {1: 6, 2: 2, 3: 6}, 5.0, 5)
    assert b[2] == (-20.0, -10.0) and b[3] == (-10.0, -5.0)


def test_thermal_bin_lines_show_cx_cy_and_held_marks():
    from mppp.sfm.thermal import thermal_bin_lines
    rows = [{"camera": "NL_T-020-015", "images": 12, "T_median_degC": -17.2, "T_min_degC": -19.0, "T_max_degC": -15.1,
             "fx": 2956.5, "fy": 2956.4, "cx": 2561.0, "cy": 1920.5,
             "start": {"fx": 2956.0, "fy": 2956.0, "cx": 2561.0, "cy": 1920.5}}]
    lines = thermal_bin_lines(rows, ("fx", "fy"))
    head, row = lines[0], lines[1]
    assert "cx*" in head and "cy*" in head and "fx*" not in head
    assert "2561.00" in row and "1920.50" in row and "+0.50" in row
    assert "held at the start (cx, cy)" in lines[-1]
    lines = thermal_bin_lines(rows, ("fx", "fy", "cx", "cy"))
    assert "*" not in lines[0] and not lines[-1].lstrip().startswith("*")
    lines = thermal_bin_lines(rows, ("fx", "fy"), hold=True)
    assert "fx*" in lines[0]


def test_split_by_temperature_records_the_start(tmp_path):
    pytest.importorskip("pycolmap")
    src = (ROOT / "src" / "mppp" / "sfm" / "thermal.py").read_text(encoding="utf-8")
    assert '"start": {"fx": float(p[0]), "fy": float(p[1]), "cx": float(p[2]), "cy": float(p[3])}' in src


def test_lmst_histogram(tmp_path):
    pytest.importorskip("matplotlib")
    import importlib.util
    spec = importlib.util.spec_from_file_location("lmst_histogram", ROOT / "scripts" / "lmst_histogram.py")
    lh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lh)

    def rec(stem, lmst, fam="N", ds=0.5, size=(2560, 1920), site=1, drive=10, sol=100):
        return {"source_product": f"/x/{stem}.IMG", "LMST": f"Sol-00100M{lmst}", "sol": sol, "site": site,
                "drive": drive, "native_size": list(size), "pose": {"C_enu_m": [drive * 1.0, 0.0, 0.0]},
                "filename": {"family": fam, "downsample_scale": ds, "stem": stem}}
    for root, folder, ver, recs in (
            ("old", "sid_colmap", "v0p35", [rec("A", "10:00:00"), rec("B", "11:00:00"), rec("OLDONLY", "12:00:00")]),
            ("new", "sid_colmap", "v0p43", [rec("A", "10:00:00"), rec("B", "11:00:00"), rec("C", "13:30:00", drive=11),
                                            rec("TILE", "14:00:00", ds=1.0, size=(1280, 960))]),
            ("new", "sid_colmap_nav_zcam34", "v0p43", [rec("Z1", "14:30:00", fam="Z", size=(1648, 1200)),
                                                       rec("A", "10:00:00")])):
        d = tmp_path / root / folder / "processed"
        d.mkdir(parents=True)
        (d / f"mppp_manifest_{ver}.json").write_text(json.dumps({"images": recs}))
    out = tmp_path / "out"
    assert lh.main(["--roots", str(tmp_path / "new"), str(tmp_path / "old"), "--out", str(out), "--name", "t"]) == 0
    assert (out / "t.png").stat().st_size > 10000
    import csv
    rows = list(csv.DictReader((out / "t_by_site.csv").open()))
    assert len(rows) == 1 and rows[0]["site"] == "sid"
    assert rows[0]["navcam"] == "3" and rows[0]["mastcam_z34"] == "1"      # A, B, C; OLDONLY and TILE left out
    assert rows[0]["stations"] == "2"
    assert lh.lmst_hours("Sol-01451M13:00:25.313") == pytest.approx(13.007, abs=1e-3)
