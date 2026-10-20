"""``python -m mppp`` - the same as the ``mppp`` command."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
