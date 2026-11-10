"""
mppp.ibr - image-based rendering products of an aligned block (0.96.0, the prototypes of the 31 Oct and 29 Oct plans):

* :mod:`.station` - one station (site, drive) of a finished block and the pivot of its mast 360 (the centre the
  camera centres turn about, fitted by least squares);
* :mod:`.mesh` - the fused dense cloud of the station's Navcam frames made a clean mesh (Poisson, or COLMAP's
  Delaunay mesher when the executable is at hand), cleaned (floaters, rubber sheets across occlusion edges, surface
  without support, the radius clip, decimation, smoothing) and a far-field sphere behind it;
* :mod:`.raycast` - per-image depth, range, normal and label maps by ray casting every pixel of the image's exact
  camera model through the mesh and the sphere: one surface, many views, so the maps agree by construction;
* :mod:`.photometry` - the photometric cube per image: incidence, emission and phase angles (and cast shadow) from
  the ray-cast normals, the pixel rays and the Sun vector of the frame;
* :mod:`.ods` - the stereoscopic panorama of the station: two equirectangular eye images from the mesh and the
  station's own frames (the omnidirectional stereo of a 64 mm IPD circle about the pivot), with a provenance map.

Everything is in the block frame (metres) of the model the block ended with (``gui_native/sparse``); nothing is
re-aligned.  The optional dependency set is ``mppp[ibr]`` (open3d, trimesh); the CLI is ``mppp mesh``,
``mppp depthmaps`` and ``mppp ods``.
"""
from .station import Station, fit_pivot, select_station, station_images  # noqa: F401
