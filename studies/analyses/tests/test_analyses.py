"""Tests of the analysis scripts in studies/analyses (lmst_histogram, site_appearance)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def test_site_appearance_measures_on_synthetic_texture():
    pytest.importorskip("cv2")
    sys.path.insert(0, str(HERE))
    from site_appearance import measures, elevation_map
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    base = 20000 + 4000 * gaussian_filter(rng.normal(size=(600, 800)), 2.0)
    smooth = 20000 + 4000 * gaussian_filter(rng.normal(size=(600, 800)), 8.0)
    mask = np.zeros((600, 800)); mask[50:550, 50:750] = 255
    u8 = lambda x: 255 * (x - x.min()) / np.ptp(x)                               # noqa: E731
    a = measures(base, u8(base), mask)
    b = measures(smooth, u8(smooth), mask)
    assert a["band_contrast_s1"] > b["band_contrast_s1"] and a["sift_per_mpix"] > b["sift_per_mpix"]
    assert 0 < a["mask_frac"] <= 1 and a["spectral_tiles"] > 0 and b["spectral_slope"] > a["spectral_slope"]
    meta = {"intrinsics": {"K": [[500, 0, 400], [0, 500, 300], [0, 0, 1]], "dist_opencv": {}},
            "pose": {"R_world_to_cam": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]}}      # looking north, level
    el = elevation_map(meta, (600, 800))
    assert abs(el[300, 400]) < 0.2 and el[0, 400] > 25 and el[-1, 400] < -25

def test_lmst_histogram(tmp_path):
    pytest.importorskip("matplotlib")
    import importlib.util
    spec = importlib.util.spec_from_file_location("lmst_histogram", HERE / "lmst_histogram.py")
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
