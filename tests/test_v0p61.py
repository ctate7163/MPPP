"""v0p61: the low-backlash list, regular-state focus bins, the focus-line refit, k3 = 0, the notebook 04 guards and
the Windows scripts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_low_backlash_list_shipped_and_matching(tmp_path):
    from mppp.sfm.backlash import is_low_backlash, load_low_backlash, low_backlash_fingerprint, low_backlash_path
    lst = load_low_backlash()
    assert low_backlash_path().is_file() and len(lst["entries"]) >= 60 and lst["clocks"]
    assert low_backlash_fingerprint() and len(lst["uncertain"]) == 1
    row = lambda stem, g: {"stem": stem, "camera_group": g}                     # noqa: E731
    # a sequence at one zoom (ZCAM08045_034), any product type / version
    assert is_low_backlash(row("ZLF_0091_0675032123_223EBY_N0040136ZCAM08045_034085J01", "ZL034"), lst)
    # a full product name: same eye and clock, other product type
    assert is_low_backlash(row("ZRF_0094_0675283792_738EBY_N0040136ZCAM03144_034085J01", "ZR034"), lst)
    # (sol, clock) fragments at 48 mm; never at 110 mm (one focus state)
    assert is_low_backlash(row("ZLF_0121_0677678747_053EBY_N0041250ZCAM08114_048085J01", "ZL048"), lst)
    assert not is_low_backlash(row("ZLF_0121_0677678823_053EBY_N0041250ZCAM08114_110085J01", "ZL110"), lst)
    assert not is_low_backlash(row("ZLF_0363_0700000000_000EBY_N0040136ZCAM08394_034085J01", "ZL034"), lst)
    # own list: comments, separators, sol:sequence, extensions, too-short fragments ignored
    f = tmp_path / "low.csv"
    f.write_text("# list\nZCAM01234_063, 12:ZCAM00001\nABC\nZL0_0005_0600000000_000RAD_N0000000ZCAM00002_034085A01.IMG?\n")
    own = load_low_backlash(f)
    assert "ABC" not in own["entries"] and (12, "ZCAM00001") in own["sol_sequences"] and own["uncertain"]
    assert is_low_backlash({"stem": "x", "sequence": "ZCAM00001", "sol": 12, "camera_group": "ZL034"}, own)
    assert is_low_backlash(row("ZLF_0005_0600000000_000EBY_N0000000ZCAM00002_034085J01", "ZL034"), own)
    assert is_low_backlash(row("ZRF_0010_0610000000_000EBY_N0000000ZCAM01234_063085J01", "ZR063"), own)
    assert low_backlash_fingerprint(tmp_path / "none.csv") is None


def _zrows(group, counts, regular=()):
    return [{"camera_group": group, "instrument": group, "focus_count": float(c), "label_f_px": 5000.0 + 0.1 * c,
             "focus_state": "regular" if i in regular else "backlash", "sol": 100, "name": f"{group}_{i}"}
            for i, c in enumerate(counts)]


def test_split_by_focus_regular_state_bins():
    from mppp.sfm.project import _split_by_focus
    base = {"model": "FULL_OPENCV", "width": 1648, "height": 1200, "source": "median of label CAHVOR models",
            "params": [5000.0, 5000.0, 824.0, 600.0, 0, 0, 0, 0, 0, 0, 0, 0], "fixed_params": ["k3"]}
    rows = _zrows("ZL034", [1000, 1005, 1010, 1012], regular=(2, 3))
    gm = {"f0_px": 5045.0, "label_f0_px": 5000.0, "reference_focus": 1000.0, "slope_px_per_count": 0.1,
          "aspect": 1.0, "focus_range": [0, 3000]}
    cams = _split_by_focus(rows, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal", {"cameras": {"ZL034": gm}})
    reg = [k for k in cams if k.endswith("_reg")]
    assert len(cams) == 2 and len(reg) == 1
    c = cams[reg[0]]
    assert c["backlash_state"] == "regular" and {"fx", "fy", "k3"} <= set(c["fixed_params"])
    other = cams[next(k for k in cams if not k.endswith("_reg"))]
    assert c["params"][0] == pytest.approx(other["params"][0] / 1.009 + 0.1 * (c["focus_count_median"] - other["focus_count_median"]) / 1.009, rel=1e-6)
    assert "k3" in other["fixed_params"]                      # the eye's fixed terms are inherited (ZCAM_K3 = "zero")
    assert {r["instrument"] for r in rows if r["focus_state"] == "regular"} == set(reg)


def test_label_median_k3_zero():
    from mppp.sfm.project import _camera_from_label_median
    p = [("FULL_OPENCV", [5000, 5000, 824, 600, 0.1, 0.2, 1e-4, 2e-4, 0.3, 0, 0, 0])]
    assert _camera_from_label_median("ZL034", p, (1648, 1200), ("p1", "p2"))["params"][8] == pytest.approx(0.3)
    c = _camera_from_label_median("ZL034", p, (1648, 1200), ("p1", "p2", "k3"))
    assert c["params"][6:9] == [0.0, 0.0, 0.0]


class _Cam:
    def __init__(self, f, a=1.0):
        self.params = np.array([f / np.sqrt(a), f * np.sqrt(a), 824, 600, 0, 0, 0, 0, 0, 0, 0, 0], float)


def test_refit_focus_lines_flags_the_outlier_bin(monkeypatch):
    from mppp.sfm import zcam
    focus = [500, 800, 1100, 1400, 1700, 2000]
    f_true = lambda c: 5000.0 + 0.2 * (c - 1100)               # noqa: E731
    fs = [f_true(c) for c in focus]
    fs[3] += 40.0                                              # an outlier bin
    keys = [f"ZL034_F{c:05d}" for c in focus] + ["ZL034_F01100_reg"]
    cams = {i + 1: _Cam(f) for i, f in enumerate(fs)}
    cams[len(keys)] = _Cam(4950.0)
    rec = SimpleNamespace(cameras=cams)
    proj = SimpleNamespace(
        settings={"database": {"cameras": {k: i + 1 for i, k in enumerate(keys)}}},
        cameras={k: {"group": "ZL034", "focus_count_median": float(c),
                     "backlash_state": "regular" if k.endswith("_reg") else None}
                 for k, c in zip(keys, focus + [1100])},
        images=[])
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: {i + 1: 1000 for i in range(len(keys))})
    model = {"cameras": {"ZL034": {"f0_px": 5045.0, "label_f0_px": 5000.0, "reference_focus": 1100.0,
                                   "slope_px_per_count": 0.19}}}
    out = zcam.refit_focus_lines(rec, proj, model=model)["ZL034"]
    assert out["slope_source"] == "fitted" and out["n_bins"] == 6
    assert out["f0_px"] == pytest.approx(5000.0, abs=1.0) and out["slope_px_per_count"] == pytest.approx(0.2, abs=0.005)
    assert [o["camera"] for o in out["outliers"]] == ["ZL034_F01400"]
    # applied: every bin on the line, the regular bin at line / (f0 / label f0)
    assert np.sqrt(cams[4].params[0] * cams[4].params[1]) == pytest.approx(f_true(1400), abs=1.0)
    assert np.sqrt(cams[7].params[0] * cams[7].params[1]) == pytest.approx(f_true(1100) / 1.009, abs=1.0)
    assert any("outlier bins: ZL034_F01400" in s for s in zcam.focus_line_lines({"ZL034": out}))
    # two bins: f0 only, slope from the model
    proj2 = SimpleNamespace(settings={"database": {"cameras": {keys[0]: 1, keys[1]: 2}}},
                            cameras={k: proj.cameras[k] for k in keys[:2]}, images=[])
    o2 = zcam.refit_focus_lines(SimpleNamespace(cameras={1: _Cam(fs[0]), 2: _Cam(fs[1])}), proj2, model=model,
                                apply=False)["ZL034"]
    assert o2["slope_source"] == "focus model" and o2["slope_px_per_count"] == 0.19


def test_wls_ignores_nan_rows():
    from mppp.sfm.navcal import _wls
    y = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    X = np.c_[np.ones(5), np.arange(5.0)]
    r = _wls(y, X, np.array([0.1, 0.1, 0.1, np.nan, 0.1]))
    assert r["n_used"] == 3 and np.all(np.isfinite(r["coef"]))
    assert np.isnan(_wls(y[:2], X[:2], np.array([0.1, np.nan]))["coef"][0])


def test_outlier_frame_threshold_and_reconstruct_options():
    import inspect
    from mppp.sfm import reconstruction as R
    assert R.OUTLIER_DEFAULTS["min_residual_px"] == 1.6
    sig = inspect.signature(R._reconstruct).parameters
    assert sig["zcam_focus_line"].default is True and sig["zcam_line_cycles"].default == 2
    assert sig["thermal_after_stage1"].default is True


def test_windows_scripts_v0p61():
    w = ROOT / "scripts" / "windows"
    a = (w / "adopt_claude.bat").read_bytes()
    assert b"\r\n" in a and b"\n" not in a.replace(b"\r\n", b"")
    assert b"git stash push -u" in a and b"claude/main" in a and b"github_push.bat" in a
    g = (w / "github_push.bat").read_bytes()
    assert b"\n" not in g.replace(b"\r\n", b"") and b"git stash push -u" in g and b"--force-with-lease" in g
    assert b"sync_from_claude.bat" in g
    z = (w / "run_sites_zcam.bat").read_text()
    assert 'set "EXTRA="' in z and "ZCAM_ZOOMS" not in z.split("set \"EXTRA=")[1].splitlines()[0]


def test_notebook_defaults_v0p61():
    import json
    import re
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for s in ("STORE_MASK_IN_ALPHA = 0.5", 'ZCAM_K3           = "label"',
              'ZCAM_BACKLASH', "ZCAM_LOW_BACKLASH", "ZCAM_FOCUS_LINE", "THERMAL_AFTER_STAGE1"):
        assert s.replace(" ", "") in src.replace(" ", ""), s
    i = src.index("NO_MASK_INFERENCE_AT = [")
    assert re.search(r'\n\s+"S032D1184"', src[i:i + 400])
    nb4 = json.loads((ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8"))
    src4 = "".join("".join(c["source"]) for c in nb4["cells"] if c["cell_type"] == "code")
    assert "navcal_v0p40" not in src4
