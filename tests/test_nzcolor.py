"""0.99.13 (Christian, 17 Nov 2026: "what is the approximate color conversion between zcam and ncam? estimate the white
balances and apply it to ncam and hazcams"): rpp.ibr.nzcolor on synthetic samples."""
from __future__ import annotations

import numpy as np


def test_the_navcam_factor_makes_the_navcams_look_like_mastcam_z():
    from rpp.ibr import nzcolor as C
    from rpp.config import default_config
    rng = np.random.default_rng(1)
    true = np.array([1.06, 0.97, 0.81])                     # Z / N radiance
    blocks = []
    for k, (s0, s1) in enumerate(((684, 684), (688, 688), (693, 693))):
        LN = rng.uniform(0.02, 0.2, (400, 3))
        LZ = LN * true * np.exp(rng.normal(0, 0.03, (400, 3)))
        blocks.append({"block": f"b{k}", "pairs": 400, "frames": 4, "missing": {},
                       "samples": [list(a) + list(b) + [s0, s1, 600] for a, b in zip(LN, LZ)]})
    s = C.summarize(blocks)
    assert np.allclose(s["radiance_ratio_Z_over_N"], true, rtol=0.01)
    col = default_config()["color"]
    wbN = np.asarray(col["white_balance_ecam"]) * col["brightness_by_family"].get("N", 1.0)
    wbZ = np.asarray(col["white_balance_zcam"])
    disp = true * wbZ / wbN
    assert np.allclose(s["navcam_factor"], disp / disp[1], rtol=0.01) and s["navcam_factor"][1] == 1.0
    # the new Navcam balance displays the Navcam radiance as Mastcam-Z displays its own, up to the brightness
    newN = np.asarray(s["white_balance_ecam_matching_zcam"]) * col["brightness_by_family"].get("N", 1.0)
    shown_N, shown_Z = newN, true * wbZ
    assert np.allclose(shown_N / shown_N[1], shown_Z / shown_Z[1], rtol=0.01)
    assert np.allclose(np.asarray(s["white_balance_hazcam_matching_zcam"]) / np.asarray(col["white_balance_hazcam"]),
                       s["navcam_factor"], rtol=1e-3)
    assert set(s["by_sol_pair"]) == {"N684-Z684", "N688-Z688", "N693-Z693"}
    assert any("white_balance_ecam" in ln for ln in C.lines(s))


def test_samples_skip_saturated_dark_and_edge_pixels():
    from rpp.ibr import nzcolor as C
    img = np.full((20, 20, 3), 5000, np.uint16)
    sat = np.zeros((20, 20), bool)
    assert np.allclose(C._sample(img, sat, 10.5, 10.5), 5000)
    assert C._sample(img, sat, 1.0, 10.5) is None             # the window leaves the image
    sat[10, 10] = True
    assert C._sample(img, sat, 10.5, 10.5) is None
    img[:] = 100
    assert C._sample(img, None, 10.5, 10.5) is None            # too dark
    assert C.summarize([])["samples"] == 0
