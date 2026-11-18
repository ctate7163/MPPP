"""The panorama render on several cores (0.99.25).

Christian, 18 Nov 2026: "rewrite the ibr ods pipeline to use multiple cores. Share memory between the workers."

A panorama eye is rendered in bands of rows (:func:`rpp.ibr.ods.render_rows`), and a band needs nothing of another.
:class:`RenderPool` gives the bands to worker processes:

* **The sources' pixels are in shared memory, once.**  The image, its low band, its edge distance and - where a
  source has them - its saturation and rover mask (:meth:`rpp.ibr.ods.SourceImage.shared_layout`) are computed in
  the parent (on threads: reading, blurring and the distance transform run outside Python's lock) into one block
  that every worker maps.  A worker holds no copy of any image.
* **The surface is in shared memory too** (vertices, triangles, the far field); each worker builds its own
  ray-casting structure from it (that structure cannot be shared between processes), with one thread.
* **The bands come back through shared memory**: a worker writes its rows straight into the eye's output block.
* The pool lives for all the eyes of one render (:func:`rpp.ibr.ods.render_station`): the sources are read and
  blurred once, not once per eye as before.

The picture is the same, bit for bit, as from the loop in one process: the bands are the same code on the same
numbers, and they are collected in row order.  Should the pool fail (a worker lost, no memory for the blocks), the
render goes on in the parent process and says so in the log.

Workers: ``workers`` of :func:`rpp.ibr.ods.render_station` (``--workers`` of ``rpp ods``), else ``RPP_IBR_WORKERS``,
else half the machine's logical processors, at most 8.  1: the loop in one process, no pool.

Processes are started with "spawn" on every system (Windows has nothing else), so a script that renders must keep
its entry point under ``if __name__ == "__main__":``.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .. import env as rpp_env

log = logging.getLogger("rpp.ibr")

MAX_AUTO_WORKERS = 8
SHARE_FROM_BYTES = 1 << 20       # arrays of the scene at least this large go into shared memory
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def default_workers(workers: Optional[int] = None) -> int:
    """The number of render processes: ``workers`` when given, else ``RPP_IBR_WORKERS``, else half the logical
    processors (at most ``MAX_AUTO_WORKERS``); at least 1."""
    if workers is None:
        v = rpp_env.get("RPP_IBR_WORKERS")
        try:
            workers = int(v) if v not in (None, "") else None
        except ValueError:
            workers = None
    if workers is None:
        workers = min(MAX_AUTO_WORKERS, (os.cpu_count() or 2) // 2)
    return max(1, int(workers))


class Arena:
    """Named arrays in one block of shared memory.  ``create`` in the parent, ``attach`` (by ``ref``) in a worker."""

    def __init__(self, shm, table: Dict[Any, Tuple[int, Tuple[int, ...], str]], owner: bool):
        self.shm, self.table, self.owner = shm, table, owner

    @classmethod
    def create(cls, layout: Dict[Any, Tuple[Tuple[int, ...], str]]) -> "Arena":
        from multiprocessing import shared_memory
        table, off = {}, 0
        for key, (shape, dtype) in layout.items():
            table[key] = (off, tuple(int(x) for x in shape), str(dtype))
            n = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
            off += (n + 63) // 64 * 64
        return cls(shared_memory.SharedMemory(create=True, size=max(off, 64)), table, True)

    @classmethod
    def attach(cls, ref: Tuple[str, Dict]) -> "Arena":
        from multiprocessing import shared_memory
        try:
            shm = shared_memory.SharedMemory(name=ref[0], track=False)       # Python 3.13: the parent alone owns it
        except TypeError:
            shm = shared_memory.SharedMemory(name=ref[0])
        return cls(shm, ref[1], False)

    @property
    def ref(self) -> Tuple[str, Dict]:
        return self.shm.name, self.table

    @property
    def nbytes(self) -> int:
        return int(self.shm.size)

    def view(self, key) -> np.ndarray:
        off, shape, dtype = self.table[key]
        return np.ndarray(shape, dtype=dtype, buffer=self.shm.buf, offset=off)

    def close(self) -> None:
        try:
            self.shm.close()
        except BufferError:                      # a view is still alive somewhere
            import gc
            gc.collect()
            try:
                self.shm.close()
            except BufferError:                  # ... then the mapping goes with the process
                pass
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


# ---- the scene's large arrays in and out of an arena ----
def _strip(obj: Any, path: Tuple, big: Dict[Tuple, np.ndarray]) -> Any:
    if isinstance(obj, np.ndarray) and obj.nbytes >= SHARE_FROM_BYTES:
        big[path] = obj
        return ("__arena__", path)
    if isinstance(obj, dict):
        return {k: _strip(v, path + (k,), big) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_strip(v, path + (i,), big) for i, v in enumerate(obj))
    return obj


def _restore(obj: Any, arena: Arena) -> Any:
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], str) and obj[0] == "__arena__":
        return arena.view(obj[1])
    if isinstance(obj, dict):
        return {k: _restore(v, arena) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_restore(v, arena) for v in obj)
    return obj


# ---- the worker process ----
_W: Dict[str, Any] = {}


def _init(payload: Dict[str, Any]) -> None:
    from . import ods
    from .raycast import Scene
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:                            # noqa: BLE001
        pass
    sa = Arena.attach(payload["scene_arena"])
    ia = Arena.attach(payload["source_arena"])
    scene = Scene.from_state(_restore(payload["scene"], sa), nthreads=1)
    sources = []
    for k, (state, has) in enumerate(payload["sources"]):
        s = ods.SourceImage.from_light_state(state)
        s.adopt({f: ia.view((k, f)) for (kk, f) in ia.table if kk == k}, has)
        sources.append(s)
    _W.update(scene=scene, sources=sources, arenas=(sa, ia), out=None)


def _band(task: Tuple) -> Tuple[int, int, List[int], Optional[np.ndarray], float]:
    from . import ods
    out_ref, kind, args, r0, r1 = task
    t0 = time.time()
    cur = _W.get("out")
    if cur is None or cur.shm.name != out_ref[0]:
        if cur is not None:
            cur.close()
        cur = _W["out"] = Arena.attach(out_ref)
    if kind == "ods":
        P, eye, width, height, ipd_m, kw = args
        band = ods.render_rows(_W["scene"], _W["sources"], P, eye, width, height, ipd_m, r0, r1, **kw)
    elif kind == "texture":                                     # 0.99.26: a station mesh's texture (rpp.ibr.station_mesh)
        from . import station_mesh
        band = station_mesh.texture_rows(_W["scene"], _W["sources"], args, r0, r1)
    else:
        raise ValueError(f"unknown band kind {kind!r}")
    for f in ods.BAND_FIELDS:
        cur.view(f)[r0:r1] = band[f]
    return r0, r1, band["rejected"], band["neff"], time.time() - t0


def _ready(_: int) -> int:
    return os.getpid()


class RenderPool:
    """Worker processes for the bands of one scene and one list of sources (see the module).  Made when the sources
    are final (gains, masks, fill colors set): the workers get them as they are then.  ``close`` it when done."""

    def __init__(self, scene, sources: Sequence[Any], workers: int, prepare_threads: Optional[int] = None):
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor
        self.workers = int(workers)
        self.failed: Optional[str] = None
        self.pool = None
        self.arenas: List[Arena] = []
        self._outs: List[Arena] = []                             # the eyes' output blocks, closed when no view is left
        t0 = time.time()
        try:
            layout = {(k, f): spec for k, s in enumerate(sources) for f, spec in s.shared_layout().items()}
            ia = Arena.create(layout)
            self.arenas.append(ia)
            views = [{f: ia.view((k, f)) for f in s.shared_layout()} for k, s in enumerate(sources)]
            for s in sources:                                    # the parent's own copies, if an earlier pass left any
                s.release()
            with ThreadPoolExecutor(max(1, int(prepare_threads or self.workers))) as ex:
                has = list(ex.map(lambda k: sources[k].fill_shared(views[k]), range(len(sources))))
            for s, v, h in zip(sources, views, has):
                s.adopt(v, h)
            t1 = time.time()
            big: Dict[Tuple, np.ndarray] = {}
            state = _strip(scene.state(), (), big)
            sa = Arena.create({p: (a.shape, a.dtype.str) for p, a in big.items()})
            self.arenas.append(sa)
            for p, a in big.items():
                sa.view(p)[...] = a
            payload = {"scene": state, "scene_arena": sa.ref, "source_arena": ia.ref,
                       "sources": [(s.light_state(), h) for s, h in zip(sources, has)]}
            old = {v: os.environ.get(v) for v in _THREAD_VARS}
            os.environ.update({v: "1" for v in _THREAD_VARS})    # one thread per worker: the workers are the cores
            try:
                self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_init, initargs=(payload,))
                pids = set(self.pool.map(_ready, range(4 * self.workers), chunksize=1))
            finally:
                for v, x in old.items():
                    if x is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = x
            log.info("ods: %d render processes ready (%d answered) - %d sources in %.1f GB of shared memory (%.0f s), "
                     "the surface in %.2f GB (%.0f s to start the processes)", self.workers, len(pids), len(sources),
                     ia.nbytes / 2 ** 30, t1 - t0, sa.nbytes / 2 ** 30, time.time() - t1)
        except Exception as e:                                   # noqa: BLE001 - then the render is in this process
            self._give_up(f"{type(e).__name__}: {e}")

    def _give_up(self, why: str) -> None:
        self.failed = why
        log.warning("ods: the render processes are given up (%s): rendering in one process", why)
        if self.pool is not None:
            try:
                self.pool.terminate()
            except Exception:                                    # noqa: BLE001
                pass
            self.pool = None

    def bands(self, P, eye: str, width: int, height: int, ipd_m: float, spans: Sequence[Tuple[int, int]],
              kw: Dict[str, Any], fallback: Callable[[Sequence[Tuple[int, int]]], Iterator]) -> Iterator:
        """(r0, r1, band) of every span of a panorama eye, in row order - from the workers, or (after a failure) from
        ``fallback``."""
        args = (np.asarray(P, float), eye, int(width), int(height), float(ipd_m), kw)
        yield from self.run("ods", [args] * len(spans), spans, width, height, bool(kw.get("linear")), fallback, eye)

    def run(self, kind: str, args: Sequence[Any], spans: Sequence[Tuple[int, int]], width: int, height: int,
            linear: bool, fallback: Callable[[Sequence[Tuple[int, int]]], Iterator], what: str = "") -> Iterator:
        """(r0, r1, band) of every span, in row order: ``kind`` "ods" (:func:`rpp.ibr.ods.render_rows`) or "texture"
        (:func:`rpp.ibr.station_mesh.texture_rows`), ``args[i]`` the arguments of span i."""
        from . import ods
        if self.pool is None:
            yield from fallback(spans)
            return
        eye = what
        spec = {"rgb": ((height, width, 3), "float32" if linear else "uint8"), "source": ((height, width), "int16"),
                "label": ((height, width), "uint8"), "range": ((height, width), "float32"),
                "hidden": ((height, width), "bool"), "ground": ((height, width), "bool"),
                "rover": ((height, width), "bool"), "rej_map": ((height, width), "bool"),
                "prov": ((height, width, 3), "uint8")}
        assert set(spec) == set(ods.BAND_FIELDS)
        done = 0
        self._close_outs()
        try:
            out = Arena.create(spec)
            self._outs.append(out)
            tasks = [(out.ref, kind, a, int(r0), int(r1)) for a, (r0, r1) in zip(args, spans)]
            busy = 0.0
            t0 = time.time()
            for r0, r1, rejected, neff, dt in self.pool.imap(_band, tasks, chunksize=1):
                busy += dt
                band = {f: out.view(f)[r0:r1] for f in ods.BAND_FIELDS}
                band.update(rejected=rejected, neff=neff)
                yield r0, r1, band
                del band
                done += 1
            wall = max(time.time() - t0, 1e-6)
            log.info("ods %s: %d bands in %.0f s on %d processes (%.0f s of band time: %.1f x one process)",
                     eye, len(spans), wall, self.workers, busy, busy / wall)
        except GeneratorExit:
            raise
        except Exception as e:                                   # noqa: BLE001
            self._give_up(f"{type(e).__name__}: {e}")
            yield from fallback(spans[done:])

    def _close_outs(self) -> None:
        for a in self._outs:
            a.close()
        self._outs = []

    def close(self) -> None:
        self._close_outs()
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        for a in self.arenas:
            a.close()
        self.arenas = []

    def __enter__(self) -> "RenderPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
