"""
Regression against the pre-package pipeline that produced the first scapes (2025).

The expected values were produced once from that code (``src/legacy`` up to 0.85) and stored in
``tests/data/legacy_expected_{ZL0,NLF}.npz``: the valid-pixel mask (packed bits), every fourth row
and column of the 16-bit and 8-bit products, the pose reference (X Y Z yaw pitch roll) and the
pinhole K and distortion vector.  Accepted differences: integer products differ by at most one
count (round-to-nearest instead of truncation); K differs at the 1e-5 level (CAHV axis A is
normalised); the XML principal point differed by exactly 0.5 px (pixel-origin convention) and
OpenCV p1/p2 were swapped in the old code.
"""
import numpy as np
import pytest

from conftest import NLF, ROOT, ZL0, needs_data, synthetic_waypoints

FIXTURES = ROOT / "tests" / "data"
pytestmark = needs_data


@pytest.mark.parametrize("name,img", [("ZL0", ZL0), ("NLF", NLF)], ids=["ZL0", "NLF"])
def test_matches_legacy(name, img):
    from mppp import MPPPImage, load_config
    f = FIXTURES / f"legacy_expected_{name}.npz"
    if not f.is_file():
        pytest.skip(f"{f.name} not present")
    old = np.load(f)
    # the Navcam brightness factor (0.9) and the selection rules are newer than the legacy code; compare at 1.0
    cfg = load_config({"masking": {"infer_mask": False}, "color": {"brightness_by_family": {"N": 1.0}},
                       "selection": {"max_boresight_elevation_deg": None, "lmst_window_h": None,
                                     "max_saturated_fraction": None}})
    new = MPPPImage(img, cfg, synthetic_waypoints())
    shape = tuple(int(s) for s in old["mask_shape"])
    mask_old = np.unpackbits(old["mask_valid"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    assert np.array_equal(mask_old, new.mask_valid > 0)
    assert int(np.abs(old["image_int16"].astype(int) - new.image_int16[::4, ::4].astype(int)).max()) <= 1
    assert int(np.abs(old["image_int8"].astype(int) - new.image_int8[::4, ::4].astype(int)).max()) <= 1
    ref = old["reference"]
    assert np.allclose(ref[:3], new.reference[1:4], atol=1e-6)                   # position: identical
    assert np.allclose(ref[3:6], new.reference[4:7], atol=0.02)                   # YPR, degrees
    K_new = new.intrinsics.K.copy()
    if "XML" in new.intrinsics.source:
        K_new[:2, 2] += 0.5
        assert (old["d_cam"][2], old["d_cam"][3]) == (new.intrinsics.dist["p2"], new.intrinsics.dist["p1"])
    assert np.allclose(old["K_cam"], K_new, rtol=2e-5, atol=2e-2)
