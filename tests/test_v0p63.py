"""v0p63: more features, five SIFT octaves and guided matching by default (Navcam - Mastcam-Z tie points)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_feature_and_matching_defaults():
    import inspect
    from mppp.sfm import database as D
    from mppp.sfm.matching import match
    assert D.DEFAULT_MAX_NUM_FEATURES == 32768 and D.DEFAULT_NUM_OCTAVES == 4 and D.DEFAULT_FIRST_OCTAVE == -1
    assert inspect.signature(match).parameters["guided_matching"].default is False   # v0p64.1: off again
    assert D._feature_settings(32768, 5120, False) == {"max_num_features": 32768, "max_image_size": 5120,
                                                        "domain_size_pooling": False}     # old records stay valid
    assert D._feature_settings(32768, 5120, False, num_octaves=5)["num_octaves"] == 5
    from mppp.pipeline import Settings
    s = Settings()
    assert s.max_num_features == 32768 and "num_octaves" in s.sift and s.match["guided_matching"] is False
