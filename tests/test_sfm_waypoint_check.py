"""The waypoint check of a finished block (mppp.sfm.waypoint_check): the similarity fit, flagged stations, anisotropy."""
import csv
import json

import numpy as np
import pytest
def test_similarity_fit_is_robust():
    from mppp.sfm.waypoint_check import similarity_fit
    rng = np.random.default_rng(1)
    src = rng.uniform(-50, 50, (12, 3))
    src[:, 2] *= 0.05
    a = np.radians(0.3)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    dst = 1.0025 * src @ R.T + [0.2, -0.1, 0.05] + rng.normal(0, 0.03, src.shape)
    dst[3] += [2.0, -1.5, 0]                                   # one bad localisation
    f = similarity_fit(src, dst)
    assert f["scale"] == pytest.approx(1.0025, abs=4e-4) and f["yaw_deg"] == pytest.approx(0.3, abs=0.05)


def _fake_block(tmp, stations, bad=None, stretch_e=0.0):
    """A project folder with project.json and error_input/poses.csv: refined = prior (+ the bad offset, + an east
    stretch) per station, 10 images each."""
    p = tmp / "mars2020_sol_0001_fake_colmap" / "colmap"
    (p / "error_input").mkdir(parents=True)
    (p / "project.json").write_text(json.dumps({"settings": {"reconstruction": {"unlocalized": {}}}, "images": []}))
    with open(p / "error_input" / "poses.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "station", "station_label", "sol", "instrument", "downsample_scale", "dE_m", "dN_m", "dU_m",
                    "dC_m", "dAttitude_deg", "E_m", "N_m", "U_m", "observations"])
        for k, (st, e, n) in enumerate(stations):
            d = np.array([stretch_e * e, 0.0, 0.0]) + (np.array(bad[1]) if bad and bad[0] == st else 0.0)
            for i in range(10):
                w.writerow([f"NLF_{k}_{i}.png", st, st, 100 + k, "NL", 0.25, *d, float(np.linalg.norm(d)), 0.1,
                            e + d[0], n + d[1], d[2], 500])
    return p


def test_check_block_flags_a_bad_waypoint(tmp_path):
    from mppp.sfm import waypoint_check as WC
    st = [(f"S010D{i:04d}", float(x), float(y)) for i, (x, y) in
          enumerate([(0, 0), (20, 5), (40, -3), (10, 30), (35, 25), (-5, 45)])]
    p = _fake_block(tmp_path, st, bad=("S010D0002", (1.4, -0.6, 0.0)))
    r = WC.check_block(p)
    assert r["flagged"] == ["S010D0002"] and not r["similarity"]["translation_only"]
    m = WC.mission_flags([r])
    assert m["flagged"] == ["S010D0002"]
    out = WC.write_flags(m, tmp_path / "flags")
    assert "S010D0002 **flag**" in (out / "waypoint_flags.md").read_text()
    assert len(list(csv.reader(open(out / "waypoint_flags.csv")))) == 7


def test_check_block_east_west_stretch_is_anisotropic(tmp_path):
    from mppp.sfm import waypoint_check as WC
    rng = np.random.default_rng(3)
    st = [(f"S020D{i:04d}", float(x), float(y)) for i, (x, y) in enumerate(rng.uniform(-60, 60, (10, 2)))]
    r = WC.check_block(_fake_block(tmp_path, st, stretch_e=0.004))
    an = r["anisotropy"]
    assert an["scale_E_ppm"] - an["scale_N_ppm"] == pytest.approx(4000, rel=0.05)


def test_small_block_translation_only(tmp_path):
    from mppp.sfm import waypoint_check as WC
    st = [(f"S030D{i:04d}", float(i * 3), 0.0) for i in range(6)]          # 15 m
    r = WC.check_block(_fake_block(tmp_path, st))
    assert r["similarity"]["translation_only"] and r["anisotropy"] is None
