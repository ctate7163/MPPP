"""
``mppp_studies``: the name of :mod:`rpp_studies` before 0.99 - kept so that old imports, pickles and scripts
outside the repository keep working (``mppp_studies.X`` is the module ``rpp_studies.X`` itself; see
:mod:`mppp`).  New code imports ``rpp_studies``.
"""
import warnings

import mppp as _mppp  # noqa: F401  - installs the finders for mppp.* and mppp_studies.*
import rpp_studies as _studies

warnings.warn("the package mppp_studies is called rpp_studies since 0.99: import rpp_studies",
              DeprecationWarning, stacklevel=2)


def __getattr__(name):
    try:
        return getattr(_studies, name)
    except AttributeError:
        raise AttributeError(f"module 'mppp_studies' (now 'rpp_studies') has no attribute {name!r}") from None
