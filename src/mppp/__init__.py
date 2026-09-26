"""
MPPP — Mars Photogrammetry Preprocessing Pipeline.

Pre-processes Mars 2020 PDS image products (``.IMG``) from Navcam and
Mastcam-Z into photogrammetry-ready imagery, masks, camera models and pose
priors for Agisoft Metashape and COLMAP.

Versioning: ``__version__`` (PEP 440) is the single source of the version;
``pyproject.toml`` reads it, and ``VERSION_TAG`` (``v0p13``) is derived from it
for file names, notebooks and output snapshots.  Releases before 1.0 were
tagged ``v0pN``; see CHANGELOG.md and docs/history/.
"""

__version__ = "0.20.0"


def _tag(version: str) -> str:
    major, minor = version.split(".")[:2]
    return f"v{major}p{minor}"


VERSION_TAG = _tag(__version__)

# Light imports only: torch / timm are imported lazily inside ``mppp.mask``.
from .config import load_config, default_config, save_config_snapshot  # noqa: E402
from .filenames import parse_filename                                   # noqa: E402
from .waypoints import load_waypoints                                   # noqa: E402
from .image import MPPPImage                                            # noqa: E402
from .process import process_images                                     # noqa: E402

__all__ = [
    "__version__", "VERSION_TAG",
    "load_config", "default_config", "save_config_snapshot",
    "parse_filename", "load_waypoints", "MPPPImage", "process_images",
]
