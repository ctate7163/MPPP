import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mppp.paths import checkpoints_dir  # noqa: E402

DATA = ROOT / "tests" / "data" / "m20"                                     # two public PDS products (0.91: Rockytop)
# 0.91: Rockytop, the waypoint of sols 461-464 (site 26 drive 630) - a quarter-resolution frame of the sol-461 Navcam
# 360 and a full 34 mm Mastcam-Z frame of ZCAM03381 (sol 464); the example data set of the same place is a release
# asset (mppp fetch example)
NLF = DATA / "NLF_0461_0707870987_044RAD_N0260630NCAM03461_0A0195J02.IMG"
ZL0 = DATA / "ZL0_0464_0708130229_159RAD_N0260630ZCAM03381_0340LMA01.IMG"


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

# The Navcam test product is a Sun-pointing tile (boresight +78 deg, 99.99 % saturated, taken outside the LMST
# window).  The three selection rules of the shipped defaults would refuse it, so the tests that process it pass
# SELECTION_OFF (``cfg_nomask``, ``config_selection_off``) explicitly; the defaults themselves are not changed.
SELECTION_OFF = {"selection": {"max_boresight_elevation_deg": None, "max_saturated_fraction": None, "lmst_window_h": None}}

needs_data = pytest.mark.skipif(not (NLF.is_file() and ZL0.is_file()), reason="example IMGs not present")
needs_ckpt = pytest.mark.skipif(not CKPT.is_file(), reason="mask checkpoint not present")
# the 2025 checkpoint whose decoder activations overflow fp16 (the Sept-2026 failure); v0p11
LEGACY_CKPT = checkpoints_dir() / "convnext_tiny_seg_best.pt"
needs_legacy_ckpt = pytest.mark.skipif(not LEGACY_CKPT.is_file(), reason="2025 mask checkpoint not present")


def synthetic_waypoints():
    """Minimal GeoJSON with the landing frame (site 3) and the site 26 (the test products) and site 33 origins."""
    def feat(sol, site, drive, e, n, z, lon, lat):
        return {"type": "Feature",
                "properties": {"sol": sol, "site": site, "drive": drive, "easting": e, "northing": n,
                               "elev_geoid": z, "lon": lon, "lat": lat},
                "geometry": {"type": "Point", "coordinates": [lon, lat, z]}}
    return {"type": "FeatureCollection", "features": [
        feat(13, 3, 0, 4354494.0, 1093299.0, -2569.9, 77.45089, 18.44463),
        feat(14, 3, 38, 4354497.4, 1093294.0, -2569.9, 77.45095, 18.44455),
        feat(449, 26, 0, 4351987.44, 1094063.3, -2534.78, 77.40630, 18.45751),
        feat(700, 33, 0, 4350000.0, 1096000.0, -2500.0, 77.37100, 18.49020),
        feat(705, 33, 1000, 4349900.0, 1096100.0, -2499.0, 77.36920, 18.49190),
    ]}


@pytest.fixture(scope="session")
def waypoints():
    from mppp.waypoints import load_waypoints, snapshot_path
    return load_waypoints(snapshot_path())                                     # the packaged snapshot


def config_selection_off(over=None):
    """``mppp.load_config`` with the three selection rules off (the test products), plus ``over``."""
    from mppp import load_config
    cfg = {"selection": dict(SELECTION_OFF["selection"])}
    for k, v in (over or {}).items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return load_config(cfg)


@pytest.fixture(scope="session")
def cfg_nomask():
    return config_selection_off({"masking": {"infer_mask": False}})
