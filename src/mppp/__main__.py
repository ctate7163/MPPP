"""``python -m mppp`` - the name before 0.99: runs ``python -m rpp`` with a note."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # as rpp/__main__.py (torch and pycolmap both bring OpenMP)

from mppp import cli_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli_main())
