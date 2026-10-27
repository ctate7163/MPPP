"""``python -m mppp`` - the same as the ``mppp`` command."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # 27 Oct: as mppp_env.bat (torch and pycolmap both bring OpenMP)

from .cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
