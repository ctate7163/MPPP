"""mppp.error (merged mppp_error v0p15): the full self-test plus merge regressions."""
import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "src/mppp/data/M20_waypoints.json"   # packaged snapshot (v0p13)


@pytest.mark.slow
def test_error_model_selftest_186():
    import matplotlib
    matplotlib.use("Agg")
    from mppp.error.selftest import run_all
    assert run_all(verbose=False)


def test_build_stations_no_longer_shadowed():
    """v0p15: a second `_last_per_sol` shadowed the first -> build_stations raised TypeError."""
    from mppp.error.waypoints import build_stations, load_featurecollection
    stations, info = build_stations(load_featurecollection(str(DATA)), anchor_site=3, anchor_drive=0)
    assert len(stations) >= 2


def test_sol_1842_mid_drive_rule_is_documented_behaviour():
    """
    OPEN ISSUE (docs/mppp_error_review_v0p2.md #7), pinned, NOT fixed: on sol 1842
    the 'highest drive' rule keeps 87_5286 (final='m', mid-drive) over 88_0
    (final='y').  If this changes, the frozen site-87 prediction must be re-checked.
    """
    from mppp.error.waypoints import _last_per_sol
    feats = json.loads(DATA.read_text())["features"]
    by = {f["properties"]["RMC"]: f["properties"] for f in feats}
    assert by["87_5286"]["final"] == "m" and by["88_0"]["final"] == "y"
    assert _last_per_sol(by, ["87_5286", "88_0"]) == ["87_5286"]
