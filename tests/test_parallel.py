"""0.94.3: robustness of running faster - the pose-guided candidate search, threads, two alignments at once
(the GPU lock, shared caches written atomically, the run_sites scheduler), LightGlue's out-of-memory fallback."""
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


# ------------------------------------------------------------------ the epipolar candidate search


def _normals(A, base):
    n = np.cross(A, base)
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    return n


def test_epipolar_candidates_equal_every_pair_test():
    """The sorted-window search finds exactly the pairs of the test of every pair, on random geometry with rays
    along the baseline (either way), exactly parallel ones, unnormalised rays and tolerances up to 0.3 rad."""
    from mppp.sfm.guided import epipolar_candidates, epipolar_dense
    rng = np.random.default_rng(7)
    for k in range(150):
        base = rng.normal(size=3) * rng.uniform(0.01, 40)
        A = rng.normal(size=(int(rng.integers(1, 1500)), 3))
        B = rng.normal(size=(int(rng.integers(1, 1500)), 3))
        if k % 3 == 0:
            eb = base / np.linalg.norm(base)
            m = min(20, len(A))
            A[:m] = eb + rng.normal(size=(m, 3)) * 10 ** rng.uniform(-9, -2)
            m = min(20, len(B))
            B[:m] = -eb + rng.normal(size=(m, 3)) * 10 ** rng.uniform(-9, -2)
            A[0] = eb                                             # exactly along the baseline: no plane
        if k % 2:
            A /= np.linalg.norm(A, axis=1, keepdims=True)
            B /= np.linalg.norm(B, axis=1, keepdims=True)
        tol = float(np.sin(10 ** rng.uniform(-4, -0.5)))
        n = _normals(A, base)
        i1, j1 = epipolar_dense(A, B, n, tol)
        i2, j2 = epipolar_candidates(A, B, base, n, tol)
        assert np.all(np.diff(i2) >= 0)                           # sorted by (i, j), as the dense test gives them
        s1, s2 = set(zip(i1.tolist(), j1.tolist())), set(zip(i2.tolist(), j2.tolist()))
        diff = s1 ^ s2                                            # only floating-point ties at the tolerance
        assert all(abs(abs(float(n[a] @ B[b])) - tol) < 1e-12 for a, b in diff), (k, len(diff))
    e = epipolar_candidates(np.zeros((0, 3)), np.ones((3, 3)), np.array([1.0, 0, 0]), np.zeros((0, 3)), 0.01)
    assert len(e[0]) == len(e[1]) == 0


def test_epipolar_dense_bounds_its_memory():
    """At most about 2**21 products at a time, whatever the chunk."""
    from mppp.sfm import guided as G
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(3000, 3)), rng.normal(size=(5000, 3))
    base = np.array([0.3, 1.0, 0.1])
    n = _normals(A, base)
    seen = []
    real = np.ndarray.__matmul__

    class Spy(np.ndarray):
        def __matmul__(self, other):
            seen.append(self.shape[0] * other.shape[1])
            return real(np.asarray(self), other)
    i, j = G.epipolar_dense(A, B, n.view(Spy), 0.01, chunk=100000)
    assert seen and max(seen) <= (1 << 21) + 5000 and len(i)


# ------------------------------------------------------------------ the GPU lock across processes

_HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    from mppp.sfm.gpu import gpu_lock
    with gpu_lock("holder", path=sys.argv[2], poll_s=0.1):
        open(sys.argv[3], "w").write(str(time.time()))
        time.sleep(float(sys.argv[4]))
""")


def _hold(tmp_path, seconds):
    flag = tmp_path / f"held_{time.time_ns()}"
    p = subprocess.Popen([sys.executable, "-c", _HOLDER, str(SRC), str(tmp_path / "gpu.lock"), str(flag), str(seconds)])
    t0 = time.time()
    while not flag.exists():
        assert time.time() - t0 < 60 and p.poll() is None
        time.sleep(0.05)
    return p


def test_gpu_lock_makes_a_second_process_wait(tmp_path, caplog):
    from mppp.sfm.gpu import gpu_lock
    p = _hold(tmp_path, 2.0)
    t0 = time.time()
    with caplog.at_level("INFO", logger="mppp.sfm"):
        with gpu_lock("second", path=tmp_path / "gpu.lock", poll_s=0.1) as g:
            waited = time.time() - t0
    p.wait(30)
    assert waited > 1.0 and g.waited_s > 1.0
    assert "waiting for the GPU (in use by pid" in caplog.text and "GPU free after" in caplog.text
    # free now: no wait
    with gpu_lock("third", path=tmp_path / "gpu.lock", poll_s=0.1) as g:
        assert g.waited_s < 0.5


def test_gpu_lock_nested_in_one_process_does_not_wait(tmp_path):
    from mppp.sfm.gpu import gpu_lock
    t0 = time.time()
    with gpu_lock("outer", path=tmp_path / "gpu.lock", poll_s=0.1, max_wait_h=0.01):
        with gpu_lock("inner", path=tmp_path / "gpu.lock", poll_s=0.1, max_wait_h=0.01) as g:
            assert g.nested
    assert time.time() - t0 < 2
    p = _hold(tmp_path, 1.0)                                       # released after the outer one
    p.wait(30)
    assert p.returncode == 0


def test_gpu_lock_is_freed_when_its_holder_dies(tmp_path):
    from mppp.sfm.gpu import gpu_lock
    p = _hold(tmp_path, 60.0)
    p.kill()
    p.wait(30)
    t0 = time.time()
    with gpu_lock("after a crash", path=tmp_path / "gpu.lock", poll_s=0.1):
        assert time.time() - t0 < 5


def test_gpu_lock_gives_up_waiting_and_can_be_off(tmp_path, monkeypatch, caplog):
    from mppp.sfm.gpu import gpu_lock
    p = _hold(tmp_path, 20.0)
    try:
        t0 = time.time()
        with caplog.at_level("INFO", logger="mppp.sfm"):
            with gpu_lock("impatient", path=tmp_path / "gpu.lock", poll_s=0.1, max_wait_h=1.0 / 3600):
                pass
        assert time.time() - t0 < 10 and "going ahead anyway" in caplog.text
        monkeypatch.setenv("MPPP_GPU_LOCK", "0")
        t0 = time.time()
        with gpu_lock("off", path=tmp_path / "gpu.lock") as g:
            assert not g.enabled and time.time() - t0 < 1
        monkeypatch.delenv("MPPP_GPU_LOCK")
        with gpu_lock("cpu step", enabled=False, path=tmp_path / "gpu.lock") as g:
            assert not g.enabled
    finally:
        p.kill()
        p.wait(30)


def test_run_step_takes_the_gpu_lock(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from mppp.sfm import gpu as GP
    from mppp.sfm import project as PR
    used = []

    class Lock:
        def __init__(self, what, enabled=True, **k):
            used.append((what, enabled))

        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass
    monkeypatch.setattr(GP, "gpu_lock", Lock)
    monkeypatch.setattr(GP, "_run", lambda python, args, echo=True: 'MPPP_RESULT {"ok": 1}\n')
    monkeypatch.setattr(PR.SfmProject, "load", staticmethod(lambda root: SimpleNamespace(settings={})))
    proj = SimpleNamespace(root=tmp_path / "site" / "colmap", save=lambda: None, settings={})
    assert GP.run_step("py", "match", proj, use_gpu=True) == {"ok": 1}
    assert GP.run_step("py", "extract_features", proj, use_gpu=False) == {"ok": 1}
    assert used == [("match (site)", True), ("extract_features (site)", False)]


# ------------------------------------------------------------------ shared caches written atomically

_WRITER = textwrap.dedent("""
    import sys, json
    sys.path.insert(0, sys.argv[1])
    from mppp.paths import atomic_write_bytes
    k = int(sys.argv[3])
    for r in range(40):
        atomic_write_bytes(sys.argv[2], json.dumps({"w": k, "r": r, "x": [k] * 200000}).encode())
""")


def test_atomic_write_never_shows_half_a_file(tmp_path):
    """Two writers and a reader of one cache file at the same time (two alignments refreshing the waypoints)."""
    from mppp.paths import atomic_write_bytes
    f = tmp_path / "cache.json"
    atomic_write_bytes(f, json.dumps({"w": -1}).encode())
    ws = [subprocess.Popen([sys.executable, "-c", _WRITER, str(SRC), str(f), str(k)]) for k in range(2)]
    reads = bad = 0
    while any(w.poll() is None for w in ws):
        try:
            json.loads(f.read_bytes())
            reads += 1
        except ValueError:
            bad += 1
        except OSError:
            pass
    assert all(w.wait(60) == 0 for w in ws)
    assert reads > 10 and bad == 0
    assert not list(tmp_path.glob(".*.tmp"))                     # no temporary files left


def test_replace_retry_waits_for_a_busy_file(tmp_path, monkeypatch):
    from mppp import paths as P
    tmp, dst = tmp_path / "a.tmp", tmp_path / "a"
    tmp.write_text("new")
    dst.write_text("old")
    calls = []
    real = os.replace

    def busy(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(32, "in use")
        real(a, b)
    monkeypatch.setattr(P.os, "replace", busy)
    assert P.replace_retry(tmp, dst, wait_s=0.01) and dst.read_text() == "new" and len(calls) == 3
    tmp.write_text("newer")
    monkeypatch.setattr(P.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError(32, "in use")))
    assert not P.replace_retry(tmp, dst, tries=3, wait_s=0.01)
    assert dst.read_text() == "new" and not tmp.exists()


def test_waypoint_refresh_writes_atomically(tmp_path, monkeypatch):
    import io
    from mppp.waypoints import table as T
    good = json.dumps({"type": "FeatureCollection", "features": [{"properties": {"site": 1}}]}).encode()

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *e):
            pass
    monkeypatch.setattr(T.urllib.request, "urlopen", lambda url, timeout=0: R(good))
    target = tmp_path / "wp.json"
    d = T.load_waypoints(target, refresh=True)
    assert d["_mppp_source"]["n_features"] == 1 and not list(tmp_path.glob(".*.tmp"))
    # the cache in use by another process: the old copy stays, with a warning
    monkeypatch.setattr(T, "atomic_write_bytes", lambda p, b: False)
    with pytest.warns(UserWarning, match="in use"):
        T.load_waypoints(target, refresh=True)


# ------------------------------------------------------------------ LightGlue out of GPU memory


def test_lightglue_out_of_memory_falls_back_to_sift(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("kornia")
    from types import SimpleNamespace
    from mppp.sfm import lightglue as LG
    from mppp.sfm import matching as M
    proj = SimpleNamespace(root=tmp_path / "site" / "colmap")

    def oom(*a, **k):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    monkeypatch.setattr(LG, "match_pairs_lightglue", oom)
    monkeypatch.setattr(LG, "pick_device", lambda d=None: "cpu")
    assert M._try_lightglue(proj, None, 8192, 0.1, None, 6.0, 15, 2.0) is None

    def other(*a, **k):
        raise RuntimeError("a bug")
    monkeypatch.setattr(LG, "match_pairs_lightglue", other)
    with pytest.raises(RuntimeError, match="a bug"):
        M._try_lightglue(proj, None, 8192, 0.1, None, 6.0, 15, 2.0)


# ------------------------------------------------------------------ run_sites: two sites at a time


def _run_sites():
    spec = importlib.util.spec_from_file_location("run_sites_par", ROOT / "scripts" / "run_sites.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    return rs


_STUB = textwrap.dedent("""
    import sys, time, json, pathlib
    work = pathlib.Path(sys.argv[1])
    t0 = time.time()
    print("stub alignment of", work.name, flush=True)
    time.sleep(3)
    (work / "stub.json").write_text(json.dumps({"t0": t0, "t1": time.time()}))
    sys.exit(1 if "rochette" in work.name else 0)
""")


def test_run_sites_runs_two_sites_at_once(tmp_path, monkeypatch):
    rs = _run_sites()
    monkeypatch.delenv("MPPP_JOBS", raising=False)
    assert rs.default_jobs() == 2
    monkeypatch.setenv("MPPP_JOBS", "3")
    assert rs.default_jobs() == 3
    monkeypatch.setenv("MPPP_JOBS", "x")
    assert rs.default_jobs() == 2
    monkeypatch.delenv("MPPP_JOBS")
    stub = tmp_path / "stub.py"
    stub.write_text(_STUB)
    fetched = []
    monkeypatch.setattr(rs, "child_cmd", lambda a, work: [sys.executable, str(stub), str(work)])
    monkeypatch.setattr(rs, "prefetch", lambda log, sets: fetched.append(1))
    root = tmp_path / "scapes"
    t0 = time.time()
    rc = rs.main(["--sites", "rockytop", "rochette", "--root", str(root)])
    took = time.time() - t0
    assert rc == 1                                                 # rochette failed
    log = (root / "run_sites_log.txt").read_text()
    assert "jobs 2" in log and "site rockytop: finished" in log and "site rochette: FAILED" in log
    assert "1 ok, 1 failed" in log and fetched == [1]
    w = {p.parent.name: json.loads(p.read_text()) for p in root.glob("*/stub.json")}
    assert len(w) == 2
    (a, b) = w.values()
    assert a["t0"] < b["t1"] and b["t0"] < a["t1"] and took < 12        # at the same time
    cons = sorted(root.glob("*/runs/console_*.txt"))
    assert len(cons) == 2 and all("stub alignment of" in c.read_text() for c in cons)
    # one at a time: no console files, no prefetch, one after the other
    for c in cons:
        c.unlink()
    for p in root.glob("*/stub.json"):
        p.unlink()
    assert rs.main(["--sites", "rockytop", "rochette", "--root", str(root), "--jobs", "1"]) == 1
    w = [json.loads(p.read_text()) for p in root.glob("*/stub.json")]
    assert len(w) == 2 and (w[0]["t1"] <= w[1]["t0"] or w[1]["t1"] <= w[0]["t0"])
    assert not list(root.glob("*/runs/console_*.txt")) and fetched == [1]


def test_run_sites_survives_a_child_that_cannot_start(tmp_path, monkeypatch):
    rs = _run_sites()
    monkeypatch.setattr(rs, "child_cmd", lambda a, work: [str(tmp_path / "no_such_program.exe")])
    monkeypatch.setattr(rs, "prefetch", lambda log, sets: None)
    root = tmp_path / "scapes"
    assert rs.main(["--sites", "rockytop", "rochette", "--root", str(root)]) == 1
    log = (root / "run_sites_log.txt").read_text()
    assert log.count("FAILED to start") == 2 and "0 ok, 2 failed" in log


def test_prefetch_only_logs_failures(monkeypatch):
    rs = _run_sites()
    from mppp.waypoints import table as T
    from mppp.mask import hub as H
    lines = []
    monkeypatch.setattr(T, "load_waypoints", lambda refresh=None: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(H, "resolve_checkpoint", lambda c=None: (_ for _ in ()).throw(FileNotFoundError("no model")))
    rs.prefetch(lines.append, {"MATCH_MODE": "exhaustive"})
    assert any("waypoints not refreshed" in x for x in lines) and any("mask model not fetched" in x for x in lines)
    assert not any("LightGlue" in x for x in lines)
    lines.clear()
    rs.prefetch(lines.append, {"INFER_MASK": False, "MATCH_MODE": "exhaustive"})
    assert not any("mask" in x for x in lines)


def test_epipolar_candidates_memory_is_bounded_driving_toward_the_scene():
    """Rays along the baseline (the rover drove toward what it photographs) put many rays in wide windows: the
    gathered candidates stay bounded, and the result does not depend on the bound."""
    import tracemalloc
    from mppp.sfm.guided import epipolar_candidates
    rng = np.random.default_rng(3)
    base = np.array([0.0, 0.0, 5.0])                              # along the view direction
    A = np.c_[rng.normal(scale=0.3, size=(12000, 2)), np.ones(12000)]
    B = np.c_[rng.normal(scale=0.3, size=(12000, 2)), np.ones(12000)]
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    n = _normals(A, base)
    tol = float(np.sin(0.003))
    tracemalloc.start()
    i1, j1 = epipolar_candidates(A, B, base, n, tol)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    i2, j2 = epipolar_candidates(A, B, base, n, tol, max_candidates=5000)
    assert np.array_equal(i1, i2) and np.array_equal(j1, j2) and len(i1) > 10000
    assert peak < 600e6, peak / 1e6
