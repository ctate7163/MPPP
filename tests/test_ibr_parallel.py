import cv2
import numpy as np
import pytest

from test_ibr import _ring_station, _station_scene


def test_render_processes_give_the_same_picture_bit_for_bit(tmp_path, caplog):
    """0.99.25 (Christian, 18 Nov 2026: "rewrite the ibr ods pipeline to use multiple cores. Share memory between the
    workers"): the bands rendered by worker processes from the sources in shared memory are the picture of the loop in
    one process, bit for bit - every array of an eye and the files of a station."""
    from rpp.ibr import ods as O
    from rpp.ibr.parallel import RenderPool
    proj, rec, st, scene = _ring_station(tmp_path)
    axes = dict(up=st.up, north=st.north, east=st.east)
    src1 = O.station_sources(proj, rec, st, scale=1.0)
    one = O.render_eye(scene, src1, st.pivot, "left", 720, 360, 0.064, rows_per_chunk=45, **axes)
    src2 = O.station_sources(proj, rec, st, scale=1.0)
    with caplog.at_level("INFO", logger="rpp.ibr"):
        with RenderPool(scene, src2, 2) as pool:
            assert pool.failed is None and src2[0]._adopted
            seen = []
            two = O.render_eye(scene, src2, st.pivot, "left", 720, 360, 0.064, rows_per_chunk=45, pool=pool,
                               progress=lambda eye, r, h, dt: seen.append(r), **axes)
            right = O.render_eye(scene, src2, st.pivot, "right", 720, 360, 0.064, rows_per_chunk=45, pool=pool, **axes)
    assert seen[-1] == 360 and seen == sorted(seen) and "render processes ready" in caplog.text
    for k in ("rgb", "source", "label", "range", "hidden_fallback", "rover", "provenance", "filled", "sky_filled", "ground_filled"):
        assert np.array_equal(one[k], two[k]), k
    assert one["effective_frames"] == two["effective_frames"] and one["surface_check_rejected"] == two["surface_check_rejected"]
    assert (two["source"] >= 0).mean() > 0.3 and not np.array_equal(two["rgb"], right["rgb"])
    # (a scene's first station render differs from its later ones in a few pixels, as before 0.99.25: the first sets
    # the scene's sky sphere and plane reach; so one render first)
    O.render_station(proj, rec, st, scene, tmp_path / "w0", width=720, scale=1.0, rows_per_chunk=45, workers=1)
    a = O.render_station(proj, rec, st, scene, tmp_path / "w1", width=720, scale=1.0, rows_per_chunk=45, workers=1)
    b = O.render_station(proj, rec, st, scene, tmp_path / "w2", width=720, scale=1.0, rows_per_chunk=45, workers=2)
    assert a["render_processes"] == 1 and b["render_processes"] == 2
    for eye in ("left", "right"):
        fa, fb = O.ods_file(tmp_path / "w1", eye, a["prefix"]), O.ods_file(tmp_path / "w2", eye, b["prefix"])
        assert np.array_equal(cv2.imread(str(fa), cv2.IMREAD_UNCHANGED), cv2.imread(str(fb), cv2.IMREAD_UNCHANGED))
    assert a["eyes"]["left"]["effective_frames"] == b["eyes"]["left"]["effective_frames"]


def test_a_pool_that_fails_leaves_the_render_to_one_process(tmp_path, monkeypatch, caplog):
    """A worker that dies does not cost the panorama: the rest of the bands are rendered in the parent."""
    from rpp.ibr import ods as O
    from rpp.ibr import parallel as P
    proj, rec, st, scene, _P, _colors = _station_scene(tmp_path)
    axes = dict(up=st.up, north=st.north, east=st.east)
    one = O.render_eye(scene, O.station_sources(proj, rec, st, scale=0.5), st.pivot, "left", 256, 128, 0.064, rows_per_chunk=32, **axes)
    src = O.station_sources(proj, rec, st, scale=0.5)
    pool = P.RenderPool(scene, src, 2)

    class Broken:
        def imap(self, *a, **k):
            raise RuntimeError("a worker was lost")

        def terminate(self):
            pass
    real, pool.pool = pool.pool, Broken()
    with caplog.at_level("WARNING", logger="rpp.ibr"):
        two = O.render_eye(scene, src, st.pivot, "left", 256, 128, 0.064, rows_per_chunk=32, pool=pool, **axes)
    real.terminate()
    pool.close()
    assert pool.failed and "rendering in one process" in caplog.text and np.array_equal(one["rgb"], two["rgb"])
    assert P.default_workers(3) == 3 and P.default_workers(0) == 1
    monkeypatch.setenv("RPP_IBR_WORKERS", "5")
    assert P.default_workers() == 5


def test_scene_state_round_trip_and_arena():
    from rpp.ibr import parallel as P
    from rpp.ibr.raycast import Scene
    V = np.array([[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    sc = Scene((V, F), plane={"normal": (0, 0, 1), "offset": 0.0, "centre": (0, 0, 0), "radius_m": 500.0}, sky=True)
    sc.rover_triangles = (0, 1)
    big = {}
    st = P._strip(dict(sc.state(), huge=np.arange(300000.0)), (), big)
    assert list(big) == [("huge",)] and st["huge"] == ("__arena__", ("huge",))
    ar = P.Arena.create({p: (a.shape, a.dtype.str) for p, a in big.items()})
    ar.view(("huge",))[...] = big[("huge",)]
    other = P.Arena.attach(ar.ref)
    back = P._restore(st, other)
    assert back["huge"][123456] == 123456.0
    back.pop("huge")
    sc2 = Scene.from_state(back, nthreads=1)
    o = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    d = np.array([[0.0, 0.0, -1.0], [0.6, 0.0, 0.8]])
    h1, h2 = sc.cast(o, d), sc2.cast(o, d)
    assert np.array_equal(h1["t"], h2["t"]) and np.array_equal(h1["label"], h2["label"]) and sc2.nthreads == 1
    assert sc2.rover_triangles == (0, 1) and sc2.ids == sc.ids
    del back, st
    other.close()
    ar.close()
