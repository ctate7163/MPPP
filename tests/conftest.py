import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mppp.paths import checkpoints_dir  # noqa: E402

DATA = ROOT / "tests" / "data" / "m20"                                     # two public PDS products (v0p13)
NLF = DATA / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ01.IMG"
ZL0 = DATA / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"


def _find_ckpt():
    """A mask model for the tests: MPPP_TEST_CHECKPOINT, the released model if cached, or a local best."""
    env = os.environ.get("MPPP_TEST_CHECKPOINT")
    if env:
        return Path(env)
    try:
        from mppp.mask.hub import default_model_name, model_path
        p = model_path(default_model_name())
        if p.is_file():
            return p
    except Exception:                                                        # noqa: BLE001
        pass
    for n in ("convnext_tiny_s4_seg_best.pt", "convnext_tiny_seg_best.pt"):
        if (checkpoints_dir() / n).is_file():
            return checkpoints_dir() / n
    return checkpoints_dir() / "convnext_tiny_s4_seg_best.pt"


CKPT = _find_ckpt()

# The Navcam test product is a Sun-pointing tile (boresight +78 deg).  The v0p22.4 sky rule
# (selection.max_boresight_elevation_deg = 45) would refuse it, so the tests run with the rule off;
# test_v0p22_4 turns it on explicitly.
import mppp.config as _cfg  # noqa: E402
_cfg._DEFAULTS["selection"]["max_boresight_elevation_deg"] = None
_cfg._DEFAULTS["selection"]["max_saturated_fraction"] = None       # the Sun tile is 99.99 % saturated
_cfg._DEFAULTS["selection"]["lmst_window_h"] = None

needs_data = pytest.mark.skipif(not (NLF.is_file() and ZL0.is_file()), reason="example IMGs not present")
needs_ckpt = pytest.mark.skipif(not CKPT.is_file(), reason="mask checkpoint not present")
# the 2025 checkpoint whose decoder activations overflow fp16 (the Sept-2026 failure); v0p11
LEGACY_CKPT = checkpoints_dir() / "convnext_tiny_seg_best.pt"
needs_legacy_ckpt = pytest.mark.skipif(not LEGACY_CKPT.is_file(), reason="2025 mask checkpoint not present")


def synthetic_waypoints():
    """Minimal GeoJSON with the landing frame (site 3) and site 33 origins. NOT real positions."""
    def feat(sol, site, drive, e, n, z, lon, lat):
        return {"type": "Feature",
                "properties": {"sol": sol, "site": site, "drive": drive, "easting": e, "northing": n,
                               "elev_geoid": z, "lon": lon, "lat": lat},
                "geometry": {"type": "Point", "coordinates": [lon, lat, z]}}
    return {"type": "FeatureCollection", "features": [
        feat(13, 3, 0, 4354494.0, 1093299.0, -2569.9, 77.45089, 18.44463),
        feat(14, 3, 38, 4354497.4, 1093294.0, -2569.9, 77.45095, 18.44455),
        feat(700, 33, 0, 4350000.0, 1096000.0, -2500.0, 77.37100, 18.49020),
        feat(705, 33, 1000, 4349900.0, 1096100.0, -2499.0, 77.36920, 18.49190),
    ]}


@pytest.fixture(scope="session")
def waypoints():
    from mppp.waypoints import load_waypoints, snapshot_path
    return load_waypoints(snapshot_path())                                     # the packaged snapshot


@pytest.fixture(scope="session")
def cfg_nomask():
    from mppp import load_config
    return load_config({"masking": {"infer_mask": False}})
