"""
RPP - the Rover Photogrammetry Pipeline.

RPP turns Mars rover images from NASA's Planetary Data System (PDS) into aligned, true-scale
structure-from-motion blocks that combine the images of many rover stops.  It reads Mars 2020 Perseverance Navcam,
Hazcam and Mastcam-Z products (Curiosity support is experimental), converts their color, masks the rover hardware
and its shadows, applies versioned in-flight camera models, places each stop from the rover's localization, and
aligns the images with COLMAP feature matching and its own weighted bundle adjustment.  The results open in COLMAP,
Agisoft Metashape and Nerfstudio.

The parts::

    rpp.pipeline    the pipeline as functions: process_site, align, run_site (the ``rpp`` command calls these)
    rpp.process     PDS products to 16-bit images with masks, poses and a manifest
    rpp.mask        the rover mask model (downloaded from Hugging Face on first use)
    rpp.calib       the camera, rig and mast model forms and the released model files (rpp/data/cmods)
    rpp.sfm         the COLMAP project, matching, bundle adjustment, health report, exports
    rpp.ibr         station products: mesh, depth and normal maps, stereo panoramas, 3D station models
    rpp.viewer      RPV, the Rover Photogrammetry Viewer (a static web page for those products)

The research studies that fit the camera models and analyze the results (``rpp_studies``) are in the source
repository, not in the installed package.

``__version__`` (PEP 440) is the single source of the version; ``VERSION_TAG`` (``v1p0``) is derived from it for
file names and output records.  Until 0.98 the package was called MPPP; the names it wrote into work folders keep
that spelling (:mod:`rpp.ondisk`), so earlier blocks are read as they are.
"""

__version__ = "1.0.0rc1"


def _tag(version: str) -> str:
    major, minor = version.split(".")[:2]
    return f"v{major}p{minor}"


VERSION_TAG = _tag(__version__)

# Light imports only: torch / timm are imported lazily inside ``rpp.mask``.
from .config import load_config, default_config, save_config_snapshot  # noqa: E402
from .filenames import parse_filename                                   # noqa: E402
from .waypoints import load_waypoints                                   # noqa: E402
from .image import RPPImage                                             # noqa: E402
from .process import process_images                                     # noqa: E402

__all__ = [
    "__version__", "VERSION_TAG",
    "load_config", "default_config", "save_config_snapshot",
    "parse_filename", "load_waypoints", "RPPImage", "process_images",
]
