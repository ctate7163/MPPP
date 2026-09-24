"""
Regression against the pre-package code (src/legacy/image.py + readers.py),
which produced the scapes processed before v0p2.  Kept in src/legacy (v0p13).

Accepted differences (see CHANGELOG): integer products differ by at most one
count (round-to-nearest instead of truncation); K differs at the 1e-5 level
(CAHV axis A is normalised); the XML principal point differs by exactly 0.5 px
(pixel-origin convention) and OpenCV p1/p2 are no longer swapped.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

from conftest import NLF, ROOT, ZL0, needs_data, synthetic_waypoints

LEGACY_DIR = ROOT / "src" / "legacy"
LEGACY = LEGACY_DIR / "image.py"
pytestmark = [needs_data, pytest.mark.skipif(not (LEGACY.is_file() and (LEGACY_DIR / "readers.py").is_file()),
                                             reason="legacy src/legacy/image.py not present")]


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    work = tmp_path_factory.mktemp("legacy") / "src"
    work.mkdir()
    code = LEGACY.read_text(encoding="utf-8")
    code = code[:code.index("# load the model once")]              # drop the hard-coded checkpoint load
    (work / "image_legacy.py").write_text(code, encoding="utf-8")
    (work / "readers.py").write_text((LEGACY_DIR / "readers.py").read_text(encoding="utf-8"), encoding="utf-8")
    cfg = json.loads((LEGACY_DIR / "config.json").read_text())
    cfg["masking"]["infer_mask"] = False
    (work / "config.json").write_text(json.dumps(cfg))
    import shutil
    from mppp.paths import data_dir
    shutil.copytree(data_dir(), work.parent / "params")            # legacy reads ../params (now package data)
    cwd = os.getcwd()
    os.chdir(work)                                                  # legacy resolves ../params from the cwd
    sys.path.insert(0, str(work))
    try:
        spec = importlib.util.spec_from_file_location("image_legacy", work / "image_legacy.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield lambda f: mod.MPPP_Image(str(f), str(work / "config.json"), synthetic_waypoints())
    finally:
        os.chdir(cwd)
        sys.path.remove(str(work))


@pytest.mark.parametrize("img", [ZL0, NLF], ids=["ZL0", "NLF"])
def test_matches_legacy(legacy, img):
    from mppp import MPPPImage, load_config
    try:
        old = legacy(img)
    except OSError as e:                                            # e.g. symlinks unavailable on Windows
        pytest.skip(str(e))
    new = MPPPImage(img, load_config({"masking": {"infer_mask": False}}), synthetic_waypoints())
    assert np.array_equal(old.mask_valid > 0, new.mask_valid > 0)
    assert np.abs(old.image_int16.astype(int) - new.image_int16.astype(int)).max() <= 1
    assert np.abs(old.image_int8.astype(int) - new.image_int8.astype(int)).max() <= 1
    assert np.allclose(old.reference[1:4], new.reference[1:4], atol=1e-6)          # position: identical
    assert np.allclose(old.reference[4:7], new.reference[4:7], atol=0.02)          # YPR, degrees
    K_old, K_new = np.array(old.K_cam, float), new.intrinsics.K.copy()
    if "XML" in new.intrinsics.source:
        K_new[:2, 2] += 0.5
        assert (old.d_cam[2], old.d_cam[3]) == (new.intrinsics.dist["p2"], new.intrinsics.dist["p1"])
    assert np.allclose(K_old, K_new, rtol=2e-5, atol=2e-2)
