"""
``mppp``: the name of :mod:`rpp` (Rover Photogrammetry Pipeline) before 0.99 - kept so that old imports, pickles,
notebooks and scripts outside the repository keep working.

``import mppp.X`` gives the module ``rpp.X`` itself (the same object, so classes, ``isinstance`` and pickles that
name ``mppp.sfm.project.SfmProject`` all work), and ``python -m mppp ...`` runs ``rpp`` with a note.  The same
goes for ``mppp_studies`` -> :mod:`rpp_studies`.  New code imports ``rpp``.  The alias packages go one version
after ``rpp`` reaches main (after freeze-3).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import rpp as _rpp

__version__ = _rpp.__version__
VERSION_TAG = _rpp.VERSION_TAG
NEW_NAME = "rpp"


class AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``<old>.X`` -> the module ``<new>.X`` (imported if needed), registered under both names."""

    def __init__(self, old: str, new: str):
        self.old, self.new = old, new

    def target(self, fullname: str):
        if fullname == self.old + ".__main__":                # the alias package's own __main__ (the note)
            return None
        return self.new + fullname[len(self.old):] if fullname.startswith(self.old + ".") else None

    def find_spec(self, fullname, path=None, target=None):
        t = self.target(fullname)
        if t is None:
            return None
        real = importlib.util.find_spec(t)
        if real is None:
            return None
        return importlib.util.spec_from_loader(fullname, self, origin=real.origin,
                                               is_package=real.submodule_search_locations is not None)

    def create_module(self, spec):
        return importlib.import_module(self.target(spec.name))

    def exec_module(self, module):
        pass                                              # the module is the real one, already executed

    # ``python -m mppp.X`` (runpy) asks the loader for the code of the module it runs as __main__
    def get_code(self, fullname):
        t = self.target(fullname)
        real = importlib.util.find_spec(t)
        return real.loader.get_code(t)

    def is_package(self, fullname):
        real = importlib.util.find_spec(self.target(fullname))
        return real is not None and real.submodule_search_locations is not None

    def get_source(self, fullname):
        t = self.target(fullname)
        real = importlib.util.find_spec(t)
        return real.loader.get_source(t)


def install(old: str, new: str) -> AliasFinder:
    """Put the alias finder ``old`` -> ``new`` first on ``sys.meta_path`` (once)."""
    for f in sys.meta_path:
        if isinstance(f, AliasFinder) and f.old == old:
            return f
    f = AliasFinder(old, new)
    sys.meta_path.insert(0, f)
    return f


install("mppp", "rpp")
install("mppp_studies", "rpp_studies")
warnings.warn("the package mppp is called rpp since 0.99: import rpp", DeprecationWarning, stacklevel=2)


def __getattr__(name):
    """``mppp.<anything>`` is ``rpp.<anything>`` (a submodule not yet imported comes through the finder)."""
    try:
        return getattr(_rpp, name)
    except AttributeError:
        raise AttributeError(f"module 'mppp' (now 'rpp') has no attribute {name!r}") from None


def __dir__():
    return sorted(set(dir(_rpp)) | set(globals()))


def cli_main(argv=None) -> int:
    """The ``mppp`` command of 0.99: ``rpp`` with a note on stderr."""
    print("note: the mppp command is called rpp since 0.99 (this alias goes in a later version)", file=sys.stderr)
    from rpp.cli import main
    return main(argv)
