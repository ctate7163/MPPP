"""v0p61: the low-backlash list, regular-state focus bins and the notebook 03 guards (the focus-line refit, k3 = 0
and the notebook 04 guards were retired with the calibration studies in 0.90)."""
from pathlib import Path
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


def test_outlier_frame_threshold_and_reconstruct_options():
    import inspect
    from mppp.sfm import reconstruction as R
    assert R.OUTLIER_DEFAULTS["min_residual_px"] == 1.6
    sig = inspect.signature(R._reconstruct).parameters
    assert sig["zcam_focus_line"].default is True and sig["zcam_line_cycles"].default == 2
    assert sig["thermal_after_stage1"].default is True


def test_windows_scripts_v0p61():
    w = ROOT / "scripts" / "windows"
    # v0p85: adopt_claude.bat, github_push.bat and sync_from_claude.bat removed (claude_sync.bat replaces them)
    assert not (w / "adopt_claude.bat").exists() and (w / "claude_sync.bat").is_file()
    z = (w / "run_sites_zcam.bat").read_text()
    assert 'set "EXTRA="' in z and "ZCAM_ZOOMS" not in z.split("set \"EXTRA=")[1].splitlines()[0]


def test_pipeline_defaults_v0p61():
    from mppp.pipeline import Settings
    s = Settings()
    assert s.store_mask_in_alpha == 0.5 and s.zcam_k3 == "label" and s.zcam_backlash == "list_split"
    assert s.zcam_low_backlash is None and s.zcam_focus_line is True and s.thermal_after_stage1 is True
    assert s.no_mask_inference_at == ["S032D1184"]
