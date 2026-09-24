"""
GPU feature extraction and matching in a second Python environment (v0p11).

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

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]
_SRC = str(Path(__file__).resolve().parents[2])          # .../src
_BOOT = ("import sys; sys.path.insert(0, sys.argv[1]); "
         "from mppp.sfm.gpu import _main; _main(sys.argv[2:])")


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
            print(line, end="")
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


def run_step(python: PathLike, step: str, project, **kwargs: Any) -> Dict[str, Any]:
    """Run ``extract_features`` or ``match`` on ``project`` in ``python``; reloads the project settings."""
    from .project import SfmProject
    if step not in ("extract_features", "match"):
        raise ValueError(step)
    project.save()                                            # the subprocess reads project.json
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
          f"CUDA {getattr(pycolmap, 'has_cuda', False)})", flush=True)
    if step == "extract_features":
        from .database import extract_features
        res: Any = str(extract_features(proj, **kwargs))
    else:
        from .matching import match
        res = match(proj, **kwargs)
    print("MPPP_RESULT " + json.dumps(res, default=str), flush=True)
