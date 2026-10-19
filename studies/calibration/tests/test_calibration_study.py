"""Tests of the calibration study modules (studies/calibration; run with ``pytest studies/calibration/tests``).
They import the study modules as siblings and ``mppp`` as the installed package."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
TESTS = HERE.parents[1] / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
from conftest import NLF, ZL0, cfg_nomask  # noqa: E402,F401

needs_data = pytest.mark.skipif(not (NLF.is_file() and ZL0.is_file()), reason="example IMGs not present")


@needs_data
def test_manifest_keeps_the_full_label_model(cfg_nomask):
    from mppp.cmod import PixelCamera, compare_cameras
    from mppp.image import MPPPImage
    from calibration import label_model_from_meta
    for path, tol in ((ZL0, 0.6), (NLF, 5.0)):
        m = json.loads(json.dumps(MPPPImage(path, cfg_nomask, None).meta, default=float))
        assert m["camera_model_label"]["MODEL_TYPE"] in ("CAHVOR", "CAHVORE")
        exact = label_model_from_meta(m)
        m.pop("camera_model_label")
        approx = label_model_from_meta(m)                       # older manifests: O along A
        d = compare_cameras(PixelCamera.cahv(exact, as_is=True), PixelCamera.cahv(approx, as_is=True), 96, False)
        assert d["max_px"] < tol
        _, li = exact.decompose()
        W, H = (5120, 3840) if path == NLF else (1648, 1200)
        assert (exact.width, exact.height) == (W, H) and 0.4 * W < li["hc"] < 0.6 * W





def test_interpolate_temperatures_in_sclk():
    from calibration import interpolate_temperatures
    mk = lambda sol, sclk: f"NLF_{sol:04d}_{sclk:010d}_000ECM_N0000000NCAM00000_01_095J01"       # noqa: E731
    samples = {mk(100, 1000000): {"NL": -30.0, "NR": -31.0}, mk(100, 1001000): {"NL": -20.0, "NR": -21.0}}
    out = interpolate_temperatures(samples, [mk(100, 1000000), mk(100, 1000500), mk(100, 1002000), mk(100, 1009000),
                                             mk(101, 1100000)])
    assert out[mk(100, 1000000)]["source"] == "label" and out[mk(100, 1000000)]["NL"] == -30.0
    assert out[mk(100, 1000500)]["source"] == "interpolated" and abs(out[mk(100, 1000500)]["NR"] + 26.0) < 1e-9
    assert out[mk(100, 1002000)]["source"] == "nearest" and out[mk(100, 1002000)]["NL"] == -20.0
    assert mk(100, 1009000) not in out and mk(101, 1100000) not in out          # beyond the gap / sol without samples


def test_fit_focal_temperature_uses_within_scape_changes_only():
    from calibration import fit_focal_temperature
    rows = []
    for sc, off, temps in (("A", 0.0, (-40, -30, -20)), ("B", 3.0, (-25, -10)), ("C", -2.0, (-35, -15))):
        for T in temps:
            rows.append({"scape": sc, "eye": "NL", "T_median_degC": T, "fx": 2955.0 + off + 0.08 * T, "observations": 1e5})
    f = fit_focal_temperature(rows, "fx")["NL"]
    assert abs(f["px_per_degC"] - 0.08) < 1e-9 and f["scapes"] == 3 and f["residual_rms_px"] < 1e-9
    assert abs(f["offsets"]["B"] - f["offsets"]["A"] - 3.0) < 1e-9


def test_found_contradicted_and_clean(tmp_path):
    from zcam_calib import found_contradicted
    groups = [{"group_key": "ZL034|sol1337|S1|F180", "camera_group": "ZL034", "eye": "L", "sol": 1337, "sequence": "S1",
               "focus_count": 180, "ratio": 1.0057, "sd": 0.001, "threshold_ratio": 1.0071, "state": "regular",
               "z": -1.4, "names": ["ZL0_1337_A.png"]},
              {"group_key": "ZL034|sol1337|S1|F930", "camera_group": "ZL034", "eye": "L", "sol": 1337, "sequence": "S1",
               "focus_count": 930, "ratio": 1.0102, "sd": 0.0005, "threshold_ratio": 1.0071, "state": "backlash",
               "z": 6.2, "names": ["ZL0_1337_B.png"]},
              {"group_key": "ZR034|sol483|S2|F1278", "camera_group": "ZR034", "eye": "R", "sol": 483, "sequence": "S2",
               "focus_count": 1278, "ratio": 1.0015, "sd": 0.001, "threshold_ratio": 1.0076, "state": "regular",
               "z": -6.1, "names": ["ZR0_0483_C.png"]}]
    f = tmp_path / "found.csv"
    f.write_text("# header\nZL0_1337_A  # rio\nZR0_0483_C  # rockytop\nZL0_9999_X  # elsewhere\n", encoding="utf-8")
    res = found_contradicted([{"name": "blk", "backlash_groups": groups}], f, shipped=None)
    assert [r[0] for r in res["remove"]] == ["ZL0_1337_A"]
    assert res["keep"] == ["ZR0_0483_C"] and res["unknown"] == ["ZL0_9999_X"]
