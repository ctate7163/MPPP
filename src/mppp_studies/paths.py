"""
Where the studies read and write outside the package (0.96.0).

The studies live in ``src/mppp_studies/`` of a source checkout; their outputs go next to the checkout, not into the
package: ``checkpoints/`` (mask models; ``mppp.paths.checkpoints_dir``) and ``Claude outputs/`` (figures, tables, the
mask A/B, the release candidates).  ``MPPP_OUTPUTS`` moves the outputs folder; ``MPPP_CHECKPOINTS`` the checkpoints.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from mppp.paths import checkpoints_dir  # noqa: F401  (re-exported: the studies import both from here)

PKG = Path(__file__).resolve().parent               # src/mppp_studies


def repo_root() -> Optional[Path]:
    """The repository root of a source checkout (``pyproject.toml`` and ``src/mppp`` two levels up), else None."""
    root = PKG.parents[1]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "mppp").is_dir() else None


def outputs_dir(create: bool = False) -> Path:
    """``Claude outputs/`` next to the checkout (``MPPP_OUTPUTS`` overrides; without a checkout: the user cache)."""
    env = os.environ.get("MPPP_OUTPUTS")
    if env:
        p = Path(env)
    elif repo_root() is not None:
        p = repo_root() / "Claude outputs"
    else:
        from mppp.paths import cache_dir
        p = cache_dir() / "outputs"
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def docs_dir() -> Optional[Path]:
    root = repo_root()
    return root / "docs" if root is not None else None
