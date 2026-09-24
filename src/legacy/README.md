# Legacy code (not part of the package)

The pre-package pipeline (`image.py`, `readers.py`, `writers.py`, `config.json`), which produced the scapes processed before MPPP v0p2, and `error_compat.py`, the shim that kept the v1 error-map notebook running. They are kept only so that `tests/test_legacy_regression.py` can compare the package with the old results. Do not import them in new code.
