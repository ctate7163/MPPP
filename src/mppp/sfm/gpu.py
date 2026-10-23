"""
GPU feature extraction and matching in a second Python environment.

Why: on Windows the pip ``pycolmap`` wheel has no CUDA.  conda-forge has a
CUDA build, but its ``ceres::Problem`` / cost-function types are not
interchangeable with the pip ``pyceres`` wheel (they are compiled separately,
so pybind11 keeps separate type registries), and the weighted bundle
adjustment fails there with "Unregistered type : ceres::Problem".  The two
steps that need the GPU — SIFT extraction and matching — use pycolmap only.
So they run as a subprocess in the CUDA environment, and everything else,
including the bundle adjustment, stays in the notebook's own environment
(pip pycolmap + pyceres).  The two share only files (``features.db``,
``database.db``, ``project.json``), and both must be the same COLMAP
version (checked).

    GPU_PY = r"C:\\Users\\me\\miniconda3\\envs\\mppp_gpu\\python.exe"
    check_gpu_python(GPU_PY)                                 # pycolmap version + has_cuda
    extract_features(proj, use_gpu=True, python=GPU_PY)
    match(proj, mode="exhaustive", use_gpu=True, python=GPU_PY)
"""
from __future__ import annotations

import logging

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

log = logging.getLogger("mppp.sfm")

PathLike = Union[str, Path]
_SRC = str(Path(__file__).resolve().parents[2])          # .../src
_BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); "
         "from mppp.sfm.gpu import _main; _main(sys.argv[2:])")


def gpu_python(default: Optional[PathLike] = None) -> Optional[str]:
    """v0p86: the GPU environment's python.exe - ``MPPP_GPU_PY``, else the one line of
    ``%LOCALAPPDATA%/mppp/gpu_python.txt`` (written by ``install_lightglue.bat`` once its environment passed the
    LightGlue test; delete the file to go back), else ``default``.  A path that does not exist is skipped."""
    cands = [os.environ.get("MPPP_GPU_PY")]
    f = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "mppp" / "gpu_python.txt"
    if f.is_file():
        lines = [ln.strip().strip('"') for ln in f.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        cands.append(lines[0] if lines else None)
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    return None if default is None else str(default)


def conda_env_python(name: str = "mppp_gpu") -> Optional[str]:
    """The python.exe of the conda environment ``name`` next to this one (``<conda>/envs/<name>``), the place
    the usual place - the old notebook's fallback (``...\\miniconda3\\envs\\mppp_gpu\\python.exe``)."""
    here = Path(sys.executable).resolve().parent
    bases = [here, here.parent, here.parent.parent]                  # base python, Scripts/, envs/<x>/
    for b in bases:
        for c in (b / "envs" / name / "python.exe", b / "envs" / name / "bin" / "python"):
            if c.is_file() and c.resolve() != Path(sys.executable).resolve():
                return str(c)
    return None


def _env_for(python: Path) -> Dict[str, str]:
    """Environment for a conda python run without ``conda activate``: its DLL folders on PATH."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX")}
    prefix = python.parent if python.parent.name.lower() != "bin" else python.parent.parent
    extra = [prefix, prefix / "Library" / "mingw-w64" / "bin", prefix / "Library" / "usr" / "bin",
             prefix / "Library" / "bin", prefix / "Scripts", prefix / "bin"]
    env["PATH"] = os.pathsep.join([str(p) for p in extra if p.is_dir()] + [env.get("PATH", "")])
    env["CONDA_PREFIX"] = str(prefix)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run(python: PathLike, args, echo: bool = True) -> str:
    python = Path(python)
    if not python.is_file():
        raise FileNotFoundError(f"python not found: {python}")
    proc = subprocess.Popen([str(python), "-c", _BOOT, _SRC, *args], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=_env_for(python), text=True, errors="replace")
    out = []
    for line in proc.stdout:                                   # stream COLMAP's log into the notebook
        out.append(line)
        if echo:
            print(line, end="", flush=True)                  # COLMAP's own log, streamed through
    if proc.wait() != 0:
        tail = "".join(out[-30:])
        raise RuntimeError(f"GPU step failed in {python} (exit {proc.returncode}):\n{tail}")
    return "".join(out)


def check_gpu_python(python: PathLike, require_cuda: bool = True) -> Dict[str, Any]:
    """pycolmap version / CUDA in ``python``; refuses a different COLMAP major.minor than this kernel's."""
    out = _run(python, ["check"], echo=False)
    info = json.loads(out.strip().splitlines()[-1])
    try:
        import pycolmap
        here = pycolmap.__version__
    except ImportError:
        here = None
    info["pycolmap_here"] = here
    if Path(info["python"]).resolve() == Path(sys.executable).resolve():
        import warnings
        warnings.warn("GPU_PY is this kernel's own python: the notebook kernel should be the pip environment "
                      "(pycolmap + pyceres wheels), not the CUDA one")
    if here and here.split(".")[:2] != info["pycolmap"].split(".")[:2]:
        raise RuntimeError(f"pycolmap {info['pycolmap']} in {python} but {here} here: the database format "
                           f"may differ; install the same COLMAP version in both")
    if require_cuda and not info["has_cuda"]:
        raise RuntimeError(f"pycolmap in {python} has no CUDA")
    return info


def check_ba_environment() -> Dict[str, Any]:
    """
    Can this kernel run the weighted bundle adjustment?  pycolmap must hand its
    ``ceres::Problem`` to pyceres, which only works for a pycolmap/pyceres pair
    built together (the pip wheels).  Raises with instructions otherwise, e.g.
    when the notebook kernel is the conda CUDA environment itself.
    """
    import pycolmap
    import pyceres
    info = {"python": sys.executable, "pycolmap": pycolmap.__version__,
            "has_cuda": bool(getattr(pycolmap, "has_cuda", False)), "pyceres": getattr(pyceres, "__version__", "?")}
    try:
        pycolmap.create_default_ceres_bundle_adjuster(pycolmap.BundleAdjustmentOptions(),
                                                      pycolmap.BundleAdjustmentConfig(),
                                                      pycolmap.Reconstruction()).problem
    except TypeError as e:
        raise RuntimeError(
            f"This kernel ({sys.executable}: pycolmap {info['pycolmap']}, CUDA {info['has_cuda']}, pyceres "
            f"{info['pyceres']}) cannot run the weighted bundle adjustment: its pycolmap and pyceres do not share "
            f"Ceres types ({str(e).splitlines()[0]}). Switch the notebook kernel to the environment with the pip "
            f"wheels pycolmap 4.2 + pyceres 2.6 (CUDA False there is fine) and keep GPU_PY pointing at this CUDA "
            f"environment for extraction and matching.") from e
    return info



# ------------------------------------------------------------------ one GPU user at a time (0.94.3)

GPU_LOCK_MAX_WAIT_H = 4.0          # after this long the step runs anyway (a hung holder must not stop the batch)
_HELD = {"depth": 0}               # this process holds the lock already: a nested step does not wait for itself


class gpu_lock:
    """
    Context manager: the GPU steps of two alignments running at once (``run_sites --jobs 2``) take turns - COLMAP's
    GPU SIFT extraction and matching, LightGlue.  An OS file lock on ``<cache>/gpu.lock`` (released by the OS if
    the holder dies); a waiter logs once who holds it.  Two users at once could run out of GPU memory, and
    LightGlue's time projection (its 3 h budget) would be skewed by the other job.  ``MPPP_GPU_LOCK=0`` turns it
    off; ``enabled=False`` (a CPU step) does nothing.
    """

    def __init__(self, what: str, enabled: bool = True, max_wait_h: Optional[float] = None,
                 path: Optional[PathLike] = None, poll_s: float = 5.0):
        self.what = what
        self.enabled = bool(enabled) and os.environ.get("MPPP_GPU_LOCK", "1").strip() not in ("0", "false", "no")
        self.max_wait_s = 3600.0 * (GPU_LOCK_MAX_WAIT_H if max_wait_h is None else float(max_wait_h))
        self.path = Path(path) if path is not None else None
        self.poll_s = float(poll_s)
        self.fh = None
        self.waited_s = 0.0

    def _try(self) -> bool:
        try:
            if sys.platform.startswith("win"):
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def __enter__(self) -> "gpu_lock":
        if not self.enabled:
            return self
        if _HELD["depth"] > 0:
            _HELD["depth"] += 1
            self.nested = True
            return self
        self.nested = False
        import time
        from ..paths import cache_dir
        p = self.path or (cache_dir() / "gpu.lock")
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(p, "a+b")
            if self.fh.seek(0, 2) == 0:
                self.fh.write(b"\0")                    # one byte to lock (msvcrt locks a byte range)
                self.fh.flush()
        except OSError as e:
            log.warning("GPU lock %s not usable (%s); %s without it", p, e, self.what)
            self.fh = None
            return self
        info = p.with_suffix(".holder")
        t0 = time.time()
        said = False
        while not self._try():
            if not said:
                try:
                    who = info.read_text(encoding="utf-8").strip()
                except OSError:
                    who = "another process"
                log.info("%s: waiting for the GPU (in use by %s)", self.what, who)
                said = True
            if time.time() - t0 > self.max_wait_s:
                log.warning("%s: the GPU has been busy for %.1f h; going ahead anyway", self.what,
                            (time.time() - t0) / 3600)
                self.fh.close()
                self.fh = None
                return self
            time.sleep(self.poll_s)
        self.waited_s = time.time() - t0
        _HELD["depth"] = 1
        if said:
            log.info("%s: GPU free after %.0f min", self.what, self.waited_s / 60)
        try:
            info.write_text(f"pid {os.getpid()} ({self.what}, {Path.cwd().name}, since {time.strftime('%H:%M')})",
                            encoding="utf-8")
        except OSError:
            pass
        return self

    def __exit__(self, *exc) -> None:
        if getattr(self, "nested", False):
            _HELD["depth"] -= 1
            return
        if self.fh is None:
            return
        _HELD["depth"] = 0
        try:
            if sys.platform.startswith("win"):
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self.fh.close()
            self.fh = None


def run_step(python: PathLike, step: str, project, **kwargs: Any) -> Dict[str, Any]:
    """Run ``extract_features`` or ``match`` on ``project`` in ``python``; reloads the project settings."""
    from .project import SfmProject
    if step not in ("extract_features", "match"):
        raise ValueError(step)
    project.save()                                            # the subprocess reads project.json
    with gpu_lock(f"{step} ({project.root.parent.name})", enabled=bool(kwargs.get("use_gpu", True))):
        out = _run(python, [step, str(project.root), json.dumps(kwargs)])
    project.settings = SfmProject.load(project.root).settings  # what the subprocess recorded
    last = [ln for ln in out.strip().splitlines() if ln.startswith("MPPP_RESULT ")]
    return json.loads(last[-1][len("MPPP_RESULT "):]) if last else {}


def _main(argv) -> None:
    import pycolmap
    if argv[0] == "check":
        print(json.dumps({"pycolmap": pycolmap.__version__, "has_cuda": bool(getattr(pycolmap, "has_cuda", False)),
                          "python": sys.executable}))
        return
    from .project import SfmProject
    step, root, kwargs = argv[0], argv[1], json.loads(argv[2])
    proj = SfmProject.load(root)
    print(f"[mppp.sfm.gpu] {step} in {sys.executable} (pycolmap {pycolmap.__version__}, "
          f"CUDA {getattr(pycolmap, 'has_cuda', False)})")
    if step == "extract_features":
        from .database import extract_features
        res: Any = str(extract_features(proj, **kwargs))
    else:
        from .matching import match
        res = match(proj, **kwargs)
    print("MPPP_RESULT " + json.dumps(res, default=str), flush=True)   # the protocol: the parent reads this line
