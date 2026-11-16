"""
Where the studies read and write outside the package (0.96.0).

The studies live in ``src/rpp_studies/`` of a source checkout; their outputs go next to the checkout, not into the
package: ``checkpoints/`` (mask models; ``rpp.paths.checkpoints_dir``) and ``Claude outputs/`` (figures, tables, the
mask A/B, the release candidates).  ``RPP_OUTPUTS`` moves the outputs folder; ``RPP_CHECKPOINTS`` the checkpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from rpp.paths import checkpoints_dir  # noqa: F401  (re-exported: the studies import both from here)
from rpp import env as rpp_env

PKG = Path(__file__).resolve().parent               # src/rpp_studies


def repo_root() -> Optional[Path]:
    """The repository root of a source checkout (``pyproject.toml`` and ``src/rpp`` two levels up), else None."""
    root = PKG.parents[1]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "rpp").is_dir() else None


def outputs_dir(create: bool = False) -> Path:
    """``Claude outputs/`` next to the checkout (``RPP_OUTPUTS`` overrides; without a checkout: the user cache)."""
    env = rpp_env.get("RPP_OUTPUTS")
    if env:
        p = Path(env)
    elif repo_root() is not None:
        p = repo_root() / "Claude outputs"
    else:
        from rpp.paths import cache_dir
        p = cache_dir() / "outputs"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def docs_dir() -> Optional[Path]:
    root = repo_root()
    return root / "docs" if root is not None else None
