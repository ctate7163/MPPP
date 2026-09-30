"""MPPP: can this Python run the MPPP notebooks headless?  Exit 0 if pycolmap, pyceres, nbclient, nbformat and
ipykernel all import (used by mppp_env.bat; prints one line: this python and what is missing)."""
import importlib
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # as notebook 03: duplicate Intel OpenMP on Windows/conda
missing = []
for name in ("pycolmap", "pyceres", "nbclient", "nbformat", "ipykernel"):
    try:
        importlib.import_module(name)
    except Exception as e:                                  # noqa: BLE001 - ImportError, DLL load failures, ...
        missing.append(f"{name} ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''})")
if missing:
    print(f"   {sys.executable}: missing {'; '.join(missing)}")
    sys.exit(1)
print(f"   {sys.executable}: ok")
