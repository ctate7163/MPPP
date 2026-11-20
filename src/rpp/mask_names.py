"""
The released mask models' names (0.99.0, the rename MPPP -> RPP), without importing torch.

``rpp_mask_v1`` .. ``v4`` were ``mppp_mask_v1`` .. ``v4`` before 0.99: the registry (``rpp/data/models.json``)
lists the old names as ``aliases``.  The processing configuration compares names through :func:`canonical`
(``rpp.process``), and a run key hashes a model under its old name (:func:`key_name`, ``rpp.pipeline``), so no
block is processed or aligned again because of the rename.  :mod:`rpp.mask.hub` downloads and caches the files.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, Optional

from .paths import data_dir


@lru_cache(maxsize=1)
def _registry() -> Dict[str, Any]:
    return json.loads((data_dir() / "models.json").read_text(encoding="utf-8"))


def canonical(name: Optional[str], reg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The registry name of ``name``: itself, the model that lists it among its ``aliases``, or the model exported
    from that checkpoint file where the registry marks them as the same (``same_as_source``).  Anything else (another
    path or file name, an unknown name, ``None``) comes back as it is."""
    if not isinstance(name, str) or not name:
        return name
    try:
        reg = reg or _registry()
    except (OSError, ValueError):                         # no registry: compare as written
        return name
    if name in reg.get("models", {}):
        return name
    for k, e in reg.get("models", {}).items():
        if name in (e.get("aliases") or []):
            return k
    # 1.0: the checkpoint a released model was exported from, where the registry says the two give the same masks
    # (``same_as_source``: rpp_mask_v5 and the file freeze 3 named) - a block processed with that file is not
    # processed again under the model's name
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    for k, e in reg.get("models", {}).items():
        if e.get("same_as_source") and base == e.get("source_checkpoint"):
            return k
    return name


def key_name(name: Optional[str], reg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """``name`` as a run key hashes it: a model released before 0.99 under its old name (``rpp_mask_v4`` and
    ``mppp_mask_v4`` both give ``mppp_mask_v4``), so the keys of the blocks on disk stay what they were."""
    try:
        reg = reg or _registry()
    except (OSError, ValueError):
        return name
    c = canonical(name, reg)
    if not isinstance(c, str):
        return name
    old = (reg.get("models", {}).get(c) or {}).get("aliases") or []
    return old[0] if old else name
