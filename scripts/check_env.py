"""RPP: can this Python run the RPP pipeline?  Exit 0 if rpp, pycolmap and pyceres all import (used by
rpp_env.bat; prints one line: this python and what is missing)."""
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # duplicate Intel OpenMP on Windows/conda
# 0.99.0: this tree's own src first, as rpp_env.bat then sets PYTHONPATH - a Python needs pycolmap and pyceres, not an
# installed rpp (a PC set up before the rename has only an editable install of mppp)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
missing = []
for name in ("rpp", "pycolmap", "pyceres"):
    try:
        importlib.import_module(name)
    except Exception as e:                                  # noqa: BLE001 - ImportError, DLL load failures, ...
        missing.append(f"{name} ({type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''})")
if missing:
    print(f"   {sys.executable}: missing {'; '.join(missing)}")
    sys.exit(1)
print(f"   {sys.executable}: ok")
