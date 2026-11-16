"""
Where RPP finds its files.

* **Package data** (``data_dir()``, installed with the package): the camera models in use ``cmods/``
  (:mod:`rpp.calib`), the optical-depth table, a snapshot of the M2020 waypoints, the Mastcam-Z occlusion
  profiles, the low-backlash observation list and the model registry ``models.json``.  Nothing here is written
  at run time except by a promotion (``rpp_studies.calibration.promote_cmods``).
* **User cache** (``cache_dir()``): refreshed waypoints and downloaded or
  installed mask models.  ``RPP_CACHE`` overrides the location; otherwise
  ``%LOCALAPPDATA%\\mppp`` on Windows, ``~/Library/Caches/mppp`` on macOS and
  ``$XDG_CACHE_HOME/mppp`` (``~/.cache/mppp``) elsewhere.  The folder keeps the old name in 0.99
  (:data:`rpp.ondisk.CACHE_DIR`): v0p98 shares it, and with it the downloaded models and the GPU lock.
* **Checkpoints you train** (``checkpoints_dir()``): ``RPP_CHECKPOINTS``, else
  ``checkpoints/`` in a source checkout (git clone), else
  ``<cache>/checkpoints``.

"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

from . import env as rpp_env
from . import ondisk

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


REPO_ROOT = Path(__file__).resolve().parents[2]      # the RPP folder of a source checkout (src/rpp/paths.py)


CMODS = "cmods"      # the one camera-model folder of the package data


def package_cmods_dir() -> Path:
    """``rpp/data/cmods`` - the camera models in use (Navcam cameras and rig, Hazcam start models when promoted,
    the Mastcam-Z focus models; README.md, CHANGES.md)."""
    return data_dir() / CMODS


def cmods_dir() -> Path:
    """The folder of the camera models in use: ``RPP_CMODS`` if set, else ``rpp/data/cmods``.  Every start
    camera comes from here (:mod:`rpp.calib`); ``rpp_studies.calibration.promote_cmods`` puts a new model here."""
    env = rpp_env.get("RPP_CMODS")
    if env:
        return Path(env)
    # 0.95.1 (29 Oct: "cmods promote" from a plain prompt wrote into the package copy while the batch files, which
    # read scripts/windows/rpp_cmods.txt, aligned with D:\scapes\cmods): the same one-line file here
    root = source_root()
    # 0.99.0: rpp_cmods.txt, else the mppp_cmods.txt of a tree set up before the rename
    for name in ("rpp_cmods.txt", "mppp_cmods.txt"):
        txt = (root / "scripts" / "windows" / name) if root else None
        if txt is not None and txt.is_file():
            for line in txt.read_text(encoding="utf-8-sig").splitlines():
                line = line.split("#", 1)[0].strip().strip('"')
                if line and Path(line).is_dir():
                    return Path(line)
    return package_cmods_dir()


def cache_dir(create: bool = True) -> Path:
    env = rpp_env.get("RPP_CACHE")
    if env:
        p = Path(env)
    elif sys.platform.startswith("win"):
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / ondisk.CACHE_DIR
    elif sys.platform == "darwin":
        p = Path.home() / "Library" / "Caches" / ondisk.CACHE_DIR
    else:
        p = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / ondisk.CACHE_DIR
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def source_root() -> Optional[Path]:
    """The repository root when running from a source checkout (``pyproject.toml`` two levels above the package)."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "rpp").is_dir() else None


def checkpoints_dir(create: bool = False) -> Path:
    env = rpp_env.get("RPP_CHECKPOINTS")
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


def replace_retry(tmp: PathLike, dst: PathLike, tries: int = 20, wait_s: float = 0.5) -> bool:
    """``os.replace(tmp, dst)``, again while Windows refuses because another process has ``dst`` open (two
    alignments at once, 0.94.3).  False (and ``tmp`` removed) when it never succeeded: ``dst`` is unchanged."""
    import time
    for k in range(max(1, tries)):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError:
            if k + 1 < tries:
                time.sleep(wait_s)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def atomic_write_bytes(path: PathLike, data: bytes) -> bool:
    """Write ``path`` so that a reader in another process sees the old or the new file, never half of one: a
    temporary file of this process next to it, then :func:`replace_retry`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    return replace_retry(tmp, path)
