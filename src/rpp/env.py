"""
Environment variables: ``RPP_<NAME>``, else the old ``MPPP_<NAME>`` (0.99.0).

The pipeline was called MPPP until 0.98; its variables (``MPPP_SCAPES``, ``MPPP_PDS``, ``MPPP_CMODS``, ...) may
still be set on a PC, in the Windows environment or in a batch file outside git.  Every read of an ``RPP_*``
variable goes through :func:`get`, which falls back to the ``MPPP_*`` spelling when the ``RPP_*`` one is not set::

    from rpp import env as rpp_env
    root = rpp_env.get("RPP_SCAPES")          # RPP_SCAPES, else MPPP_SCAPES, else None

``scripts/windows/rpp_env.bat`` does the same for the batch files (it copies each ``MPPP_X`` to ``RPP_X`` when
``RPP_X`` is unset).  ``RPP_HOME`` is the one exception there: a batch file's own tree is its home, whatever an
old ``MPPP_HOME`` says.
"""
from __future__ import annotations

import os
from typing import Dict, List, Mapping, MutableMapping, Optional, Tuple

PREFIX = "RPP_"
LEGACY_PREFIX = "MPPP_"
NOT_ADOPTED = ("HOME",)          # adopt_legacy leaves MPPP_HOME alone (the tree in use is the one that runs)


def names(name: str) -> Tuple[str, str]:
    """(``RPP_<NAME>``, ``MPPP_<NAME>``) for ``"SCAPES"``, ``"RPP_SCAPES"`` or ``"MPPP_SCAPES"``."""
    n = str(name)
    for p in (PREFIX, LEGACY_PREFIX):
        if n.startswith(p):
            n = n[len(p):]
            break
    return PREFIX + n, LEGACY_PREFIX + n


def get(name: str, default: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``RPP_<NAME>`` if it is set (even to ""), else ``MPPP_<NAME>``, else ``default`` - as ``os.environ.get``."""
    env = os.environ if environ is None else environ
    new, old = names(name)
    if new in env:
        return env[new]
    if old in env:
        return env[old]
    return default


def is_set(name: str, environ: Optional[Mapping[str, str]] = None) -> bool:
    """Either spelling is in the environment."""
    env = os.environ if environ is None else environ
    return any(n in env for n in names(name))


def which(name: str, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The spelling that :func:`get` reads (``None`` when neither is set) - for messages such as ``rpp doctor``."""
    env = os.environ if environ is None else environ
    for n in names(name):
        if n in env:
            return n
    return None


def adopt_legacy(environ: Optional[MutableMapping[str, str]] = None) -> List[str]:
    """Copy every ``MPPP_X`` to ``RPP_X`` where ``RPP_X`` is unset (not ``MPPP_HOME``), in ``environ`` (default
    ``os.environ``): for a child environment built by hand.  Returns the ``RPP_*`` names it set."""
    env = os.environ if environ is None else environ
    done = []
    for k in [k for k in env if k.startswith(LEGACY_PREFIX)]:
        base = k[len(LEGACY_PREFIX):]
        if base in NOT_ADOPTED or not base:
            continue
        if PREFIX + base not in env:
            env[PREFIX + base] = env[k]
            done.append(PREFIX + base)
    return done


def legacy_only(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The ``MPPP_*`` variables that are set without their ``RPP_*`` spelling (what ``rpp doctor`` reports)."""
    env = os.environ if environ is None else environ
    return {k: v for k, v in env.items()
            if k.startswith(LEGACY_PREFIX) and PREFIX + k[len(LEGACY_PREFIX):] not in env}
