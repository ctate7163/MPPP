"""0.94.4 (Christian, 25 Oct): the found list is always read (with "-" lines taking frames off), a states run gives
every Mastcam-Z image its own camera, and `mppp zcam-states` tells which frames are in the low backlash state."""
import json

import numpy as np
import pytest


def _name(eye, sol, clock, seq, zoom=34):
    return f"Z{eye}0_{sol:04d}_{clock:010d}_000RAD_N0000000{seq}_{zoom:03d}0LMA01"


def test_found_list_lines_and_exclusions(tmp_path):
    from mppp.sfm.backlash import default_found_path, is_low_backlash, load_low_backlash
    found = tmp_path / "zcam_low_backlash_found.csv"
    a, b = _name("L", 121, 677678707, "ZCAM08114"), _name("L", 121, 677678972, "ZCAM08114")
    found.write_text(f"# test\n{a}  # low\nZCAM07114_034\n-0121_0677678972_000\n-{_name('L', 501, 711425107, 'ZCAM07114')}\n",
                     encoding="utf-8")
    lst = load_low_backlash(None, found=found)
    row = lambda n, sol, seq: {"name": n, "camera_group": n[:2] + n[45:48], "sol": sol, "sequence": seq}   # noqa: E731
    assert is_low_backlash(row(a, 121, "ZCAM08114"), lst)
    assert not is_low_backlash(row(_name("L", 121, 677678972, "ZCAM08114").replace("ZCAM08114", "ZCAM07114"), 121,
                                   "ZCAM07114"), lst)            # a "-" line wins over a pattern
    assert is_low_backlash(row(_name("R", 684, 728371111, "ZCAM07114"), 684, "ZCAM07114"), lst)
    assert not is_low_backlash(row(_name("L", 501, 711425107, "ZCAM07114"), 501, "ZCAM07114"), lst)
    assert not is_low_backlash(row(b, 121, "ZCAM08114"), lst)
    assert {k: len(v) for k, v in lst["excluded"].items()} == {"entries": 2, "sol_sequences": 0, "clocks": 1}
    assert default_found_path(tmp_path) == found


def test_found_list_on_for_mastcamz_keys_only():
    from mppp.pipeline import Settings
    s = Settings()
    assert s.zcam_low_backlash_found == "auto" and s.zcam_bin_by == "focus"
    k = s.key_dict()
    assert k["zcam_low_backlash_found"] is None and "zcam_bin_by" not in k      # Navcam keys as before 0.94.4
    kz = s.replace(include_zcam=True).key_dict()
    assert kz["zcam_low_backlash_found"] == "auto"
    assert "zcam_bin_by" in s.replace(include_zcam=True, zcam_bin_by="image").key_dict()


def test_cameras_per_image_and_per_sequence():
    from mppp.sfm.project import _split_by_focus
    base = {"model": "FULL_OPENCV", "width": 1648, "height": 1200, "source": "median of label CAHVOR models",
            "params": [5000.0, 5000.0, 824.0, 600.0, 0, 0, 0, 0, 0, 0, 0, 0], "fixed_params": ["k3"]}
    rows = []
    for i, (sol, seq) in enumerate(((100, "ZCAM00001"), (100, "ZCAM00001"), (101, "ZCAM00002"))):
        n = _name("L", sol, 700000000 + i, seq)
        rows.append({"camera_group": "ZL034", "instrument": "ZL034", "focus_count": 1000.0 + i, "label_f_px": 5000.0,
                     "focus_state": "backlash", "sol": sol, "sequence": seq, "name": n, "stem": n})
    cams = _split_by_focus([dict(r) for r in rows], {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal")
    assert len(cams) == 1
    cams = _split_by_focus([dict(r) for r in rows], {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal", by="sequence")
    assert sorted(cams) == ["ZL034_F01000_s0100_ZCAM00001", "ZL034_F01002_s0101_ZCAM00002"]
    fbr = [dict(r) for r in rows]
    fbr[1]["focus_below"] = True                     # 0.94.4: focus from below - a bin camera of its own ("_fb")
    cams = _split_by_focus(fbr, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal")
    fbk = [k for k in cams if k.endswith("_fb")]
    assert len(cams) == 2 and fbk == ["ZL034_F01001_fb"] and cams[fbk[0]]["focus_approach"] == "below"
    rr = [dict(r) for r in rows]
    cams = _split_by_focus(rr, {"ZL034": base}, {"ZL034": "Z"}, 30.0, "focal", by="image", hold_f_images=0)
    assert len(cams) == 3 and all("_c07" in k for k in cams) and len({r["instrument"] for r in rr}) == 3
    assert all("fx" not in c["fixed_params"] for c in cams.values())
    from mppp.calib.forms import zcam_key_zoom, zcam_state
    assert {zcam_key_zoom(k) for k in cams} == {34} and {zcam_state(k, {}) for k in cams} == {"backlash"}


def _block(tmp_path, name, rows):
    """A block folder with colmap/project.json and colmap/zcam_focus_states.json from (name, group, sol, seq,
    focus, ratio) rows (ratio None: no fit)."""
    w = tmp_path / name
    (w / "colmap").mkdir(parents=True)
    imgs, fits = [], {}
    for n, g, sol, seq, foc, ratio in rows:
        imgs.append({"name": n + ".png", "stem": n, "camera_group": g, "sol": sol, "sequence": seq, "focus_count": foc})
        if ratio is not None:
            fits[n + ".png"] = {"ratio_label": ratio, "sd_ratio": 0.0002, "observations": 2000, "camera": g}
    (w / "colmap" / "project.json").write_text(json.dumps({"images": imgs}), encoding="utf-8")
    (w / "colmap" / "zcam_focus_states.json").write_text(json.dumps({"fits_before": fits, "fits_after": None}),
                                                         encoding="utf-8")
    return w


def test_states_survey_low_high_mixed_and_proposal(tmp_path):
    from mppp.sfm import zcam_states as ZS
    from mppp.sfm.backlash import load_low_backlash
    rng = np.random.default_rng(1)
    rows = []
    hi = lambda F: 1.0115 - 0.006 * (F - 600) / 1000            # noqa: E731 - the high line
    k = 0
    for s in range(12):                                          # twelve high sequences of four images
        for j in range(4):
            k += 1
            F = 200 + 80 * s
            rows.append((_name("L", 300 + s, 700000000 + k, f"ZCAM0{s:04d}"), "ZL034", 300 + s, f"ZCAM0{s:04d}", F,
                         hi(F) + rng.normal(0, 0.0003)))
    low = lambda F: hi(F) * (1 - 0.0145)                         # noqa: E731
    listed_seq = [(_name("L", 91, 675031000 + j, "ZCAM08045"), "ZL034", 91, "ZCAM08045", 300, low(300)) for j in range(3)]
    new_low = [(_name("L", 361, 698990000 + j, "ZCAM07105"), "ZL034", 361, "ZCAM07105", 500, low(500)) for j in range(3)]
    new_low.append((_name("L", 361, 698990010, "ZCAM07105"), "ZL034", 361, "ZCAM07105", 500, None))   # unfitted
    alt = [(_name("L", 121, 677678000 + j, "ZCAM08114"), "ZL034", 121, "ZCAM08114", 900,
            low(900) if j % 2 == 0 else hi(900)) for j in range(5)]
    wrong = [(_name("L", 140, 680000000, "ZCAM03018"), "ZL034", 140, "ZCAM03018", 700, hi(700))]   # listed, high
    w = _block(tmp_path, "mars2020_sol_0091_test_zcam", rows + listed_seq + new_low + alt + wrong)
    found = tmp_path / "zcam_low_backlash_found.csv"
    found.write_text("\n".join(["ZCAM08045_034", "ZCAM03018_034"] + [alt[j][0] for j in (0, 2, 4)]) + "\n",
                     encoding="utf-8")
    lists = load_low_backlash(False)
    lists = load_low_backlash(tmp_path / "none.csv", found=found)          # only the found list (no shipped one)
    blocks = ZS.find_blocks(tmp_path)
    assert list(blocks) == ["mars2020_sol_0091_test_zcam"]
    res = ZS.survey(blocks, "", lists)
    lv = res["levels"]["mars2020_sol_0091_test_zcam|ZL034"]
    assert lv["gap"] == pytest.approx(0.0145, abs=0.0008) and lv["b"] == pytest.approx(-0.006, abs=0.0015)
    seq = {(s["sol"], s["sequence"]): s for s in res["sequences"]}
    assert seq[(91, "ZCAM08045")]["verdict"] == "low" and seq[(305, "ZCAM00005")]["verdict"] == "high"
    assert seq[(121, "ZCAM08114")]["pattern"] == "LHLHL" and seq[(121, "ZCAM08114")]["verdict"] == "mixed"
    assert seq[(361, "ZCAM07105")]["pattern"] == "LLL." and seq[(361, "ZCAM07105")]["verdict"] == "low"
    p = res["proposal"]
    assert sorted(r["stem"] for r in p["add"]) == sorted(r[0] for r in new_low)       # 3 low + 1 by its sequence
    assert [r["stem"] for r in p["remove"]] == [wrong[0][0]]
    assert any("LHLHL" in r["why"] for r in p["review"])
    files = ZS.write_survey(res, tmp_path / "out")
    assert "LHLHL" in files["report"].read_text() and files["images"].is_file()
    ap = ZS.apply_proposal(p, found, when="2026-10-25")
    assert ap["added"] == 4 and ap["removed"] == 1 and ap["backup"]
    again = load_low_backlash(tmp_path / "none.csv", found=found)
    from mppp.sfm.backlash import is_low_backlash
    r_new = {"name": new_low[0][0], "camera_group": "ZL034", "sol": 361, "sequence": "ZCAM07105"}
    r_wrong = {"name": wrong[0][0], "camera_group": "ZL034", "sol": 140, "sequence": "ZCAM03018"}
    assert is_low_backlash(r_new, again) and not is_low_backlash(r_wrong, again)
    assert ZS.apply_proposal(p, found)["added"] == 0                    # nothing twice
    from mppp.cli import main
    assert main(["zcam-states", "--root", str(tmp_path), "--found", str(found), "--out", str(tmp_path / "o2")]) == 0


def test_focus_from_below_list_and_motor_moves(tmp_path):
    """0.94.4 (Christian, 26 Oct: two backlash effects superimposed): the focus-approach list is separate from the
    zoom (low / high) list and applies at every zoom; the motor moves come from the previous frame of the eye."""
    from mppp.sfm.backlash import is_focus_from_below, is_low_backlash, load_focus_below, load_low_backlash, motor_moves
    fb = tmp_path / "zcam_focus_from_below.csv"
    fb.write_text("92:ZCAM00033\n0121_0677678900\n-0121_0677678707\n", encoding="utf-8")
    lst = load_focus_below(found=fb)
    a = {"stem": _name("L", 92, 675116815, "ZCAM00033"), "sol": 92, "sequence": "ZCAM00033", "camera_group": "ZL034"}
    b = {"stem": _name("R", 121, 677678900, "ZCAM08114", 79), "sol": 121, "sequence": "ZCAM08114", "camera_group": "ZR079"}
    c = {"stem": _name("L", 121, 677678707, "ZCAM08114"), "sol": 121, "sequence": "ZCAM08114", "camera_group": "ZL034"}
    d = {"stem": _name("L", 93, 675176651, "ZCAM00034"), "sol": 93, "sequence": "ZCAM00034", "camera_group": "ZL034"}
    e = {"stem": _name("L", 121, 677678847, "ZCAM08114", 110), "sol": 121, "sequence": "ZCAM08114", "camera_group": "ZL110"}
    assert is_focus_from_below(a, lst) and is_focus_from_below(b, lst)
    assert not is_focus_from_below(c, lst) and not is_focus_from_below(d, lst)          # "-" line; the default
    fb.write_text("0121_0677678847\n", encoding="utf-8")
    assert is_focus_from_below(e, load_focus_below(found=fb))                           # 110 mm too
    assert not is_low_backlash(e, load_low_backlash(None, found=fb))                    # the zoom list: one state
    man = []
    for i, (eye, sclk, zm, fc) in enumerate((("L", 100.0, 2448, 1254), ("L", 140.0, 3834, 2316), ("L", 200.0, 3834, 2100),
                                             ("L", 9000.0, 3834, 2100), ("R", 150.0, 2448, 1296))):
        man.append({"filename": {"family": "Z", "eye": eye, "stem": f"s{i}"}, "sclk": sclk,
                    "zoom_position_count": zm, "focus_position_count": fc})
    mv = motor_moves(man)
    assert mv["s0"]["zoom_dir"] is None and mv["s4"]["previous"] is None
    assert (mv["s1"]["zoom_dir"], mv["s1"]["focus_dir"], mv["s1"]["dfocus"]) == ("up", "up", 1062.0)
    assert (mv["s2"]["zoom_dir"], mv["s2"]["focus_dir"]) == ("none", "down")
    assert mv["s3"]["zoom_dir"] is None                                                 # more than 30 min later
