"""
Where MPPP finds its files (v0p13).

* **Package data** (``data_dir()``, installed with the package): the Metashape
  camera models ``cmods/`` (v0p50: the flight calibrations and the camera models in use), the optical-depth table, a snapshot of
  the M2020 waypoints, the Mastcam-Z occlusion profiles and the model
  registry ``models.json``.  Nothing here is written at run time.
* **User cache** (``cache_dir()``): refreshed waypoints and downloaded or
  installed mask models.  ``MPPP_CACHE`` overrides the location; otherwise
  ``%LOCALAPPDATA%\\mppp`` on Windows, ``~/Library/Caches/mppp`` on macOS and
  ``$XDG_CACHE_HOME/mppp`` (``~/.cache/mppp``) elsewhere.
* **Checkpoints you train** (``checkpoints_dir()``): ``MPPP_CHECKPOINTS``, else
  ``checkpoints/`` in a source checkout (git clone), else
  ``<cache>/checkpoints``.

Before v0p13 all of these lived in ``params/`` and ``checkpoints/`` beside the
source, found by walking up from the working directory; that only worked in
the author's folder layout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def data_dir() -> Path:
    """Package data (read-only)."""
    return Path(__file__).resolve().parent / "data"


def data_file(name: str) -> Path:
    """A file in the package data; raises if absent."""
    p = data_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"package data file missing: {p}")
    return p


def params_dir() -> Path:
    """Alias of :func:`data_dir` (the calibration XMLs and tables were in ``params/`` before v0p13)."""
    return data_dir()


REPO_ROOT = Path(__file__).resolve().parents[2]      # the MPPP folder of a source checkout (src/mppp/paths.py)


CMODS = "cmods"      # v0p50: the one camera-model folder of the package data (was params/cmods, data/m20_cmods,
                     # data/navcam_consensus)


def package_cmods_dir() -> Path:
    """v0p50: ``mppp/data/cmods`` - the flight calibrations (``*_frame.xml``), the rational Navcam start cameras and
    the camera models in use (Navcam cameras and rig, Mastcam-Z focus model; README.md, CHANGES.md, history/)."""
    return data_dir() / CMODS


def cmods_dir() -> Path:
    """The folder of the camera models in use: ``MPPP_CMODS`` if set, else ``mppp/data/cmods`` (v0p50; before:
    ``<MPPP>/params/cmods``).  Notebook 03 starts from the models here (``NAVCAM_CAMERAS``, ``ZCAM_FOCUS_MODEL``);
    ``scripts/promote_cmods.py`` puts a new consensus here and keeps the old one in ``history/``."""
    env = os.environ.get("MPPP_CMODS")
    return Path(env) if env else package_cmods_dir()


def cache_dir(create: bool = True) -> Path:
    env = os.environ.get("MPPP_CACHE")
    if env:
        p = Path(env)
    elif sys.platform.startswith("win"):
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "mppp"
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Caches" / "mppp"
    else:
        p = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mppp"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def source_root() -> Optional[Path]:
    """The repository root when running from a source checkout (``pyproject.toml`` two levels above the package)."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "mppp").is_dir() else None


def checkpoints_dir(create: bool = False) -> Path:
    env = os.environ.get("MPPP_CHECKPOINTS")
    if env:
        p = Path(env)
    elif source_root() is not None:
        p = source_root() / "checkpoints"
    else:
        p = cache_dir() / "checkpoints"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_resource(name: PathLike, base: Path) -> Path:
    """Return ``name`` if it is an existing/absolute path, else ``base/name``."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return base / p
