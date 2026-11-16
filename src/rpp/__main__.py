"""``python -m rpp`` - the same as the ``rpp`` command."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 27 Oct: as rpp_env.bat (torch and pycolmap both bring OpenMP)

from .cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
