"""The waypoint snapshot, its table and the site blocks (mppp.waypoints.table, blocks)."""


def test_order_by_first_sol():
    from mppp.waypoints.blocks import order_by_first_sol
    assert list(order_by_first_sol({"b": {"sols": [[5, 6], [1, 2]]}, "a": {"sols": [[3, 4]]}})) == ["b", "a"]


def test_load_waypoints_auto_refresh(tmp_path, monkeypatch):
    """0.90: refresh="auto" downloads when the cached table is missing or older than max_age_days, and leaves a
    fresh one alone; a failed download warns and keeps the file."""
    import json, time, warnings
    import urllib.request
    from mppp.waypoints import table as T
    calls = []
    body = json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"sol": 1, "site": 3, "drive": 0,
                       "easting": 0, "northing": 0, "elev_geoid": 0}, "geometry": {"type": "Point", "coordinates": [77.45, 18.44]}}]}).encode()

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return body

    def fake_open(url, timeout=30.0):
        calls.append(url)
        return _R()
    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    target = tmp_path / "wp.json"
    w = T.load_waypoints(target, refresh="auto")                       # missing -> downloaded
    assert len(calls) == 1 and w["_mppp_source"]["n_features"] == 1 and T.waypoints_age_days(target) < 0.01
    T.load_waypoints(target, refresh="auto")                           # fresh -> left alone
    assert len(calls) == 1
    old = time.time() - 3 * 86400
    import os
    os.utime(target, (old, old))
    T.load_waypoints(target, refresh="auto", max_age_days=1.0)         # stale -> downloaded again
    assert len(calls) == 2

    def failing(url, timeout=30.0):
        raise OSError("offline")
    monkeypatch.setattr(urllib.request, "urlopen", failing)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w = T.load_waypoints(target, refresh=True)
    assert w["_mppp_source"]["n_features"] == 1 and any("could not download" in str(x.message) for x in rec)
