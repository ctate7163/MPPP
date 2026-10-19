"""The waypoint snapshot, its table and the site blocks (mppp.waypoints.table, blocks)."""


def test_order_by_first_sol():
    from mppp.waypoints.blocks import order_by_first_sol
    assert list(order_by_first_sol({"b": {"sols": [[5, 6], [1, 2]]}, "a": {"sols": [[3, 4]]}})) == ["b", "a"]
