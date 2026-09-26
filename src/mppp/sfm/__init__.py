"""
mppp.sfm — the bridge from MPPP-processed images to a COLMAP database and an
aligned reconstruction for error analysis (``mppp.error``).

Steps (notebook ``03_colmap_alignment``, ``04_`` until v0p9)::

    project  = SfmProject.create(metas, processed_dir, root)    # images, masks, cameras, rig, priors
    extract_features(project)                                    # SIFT at native resolution -> features.db
    build_database(project)                                      # -> database.db (full-resolution keypoints)
    match(project, mode="exhaustive" | "prior_pairs")            # matches + two-view geometries
    rec = reconstruct(project)                                   # CAHV-initialised, weighted BA
    report = assess_alignment(project, rec)                      # alignment health, before any analysis
    export_for_error(project, rec)                               # native-resolution text model + tables

Design (see ``docs/methods.md`` §9):

* **One COLMAP camera per Navcam eye**, at full detector resolution
  (5120 x 3840), FULL_OPENCV.  v0p20: initialised from the rational model
  ``mppp/data/m20_cmods/M2020_N{L,R}_rational.json`` (radial
  (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2), valid to the frame corners;
  k1-k4, p1, p2 refined); ``navcam_distortion="polynomial"`` uses the
  Metashape calibration ``M2020_N{L,R}0_frame.xml`` (b1 = b2 = 0, k4-k6 = 0),
  which cannot be inverted beyond ~0.88 of the corner radius.  Half- and quarter-resolution frames are the
  same physical camera: their keypoints are scaled to full-resolution pixels
  (exact for binned, detector-frame-padded MPPP images with a corner pixel
  origin), so every resolution shares one set of intrinsics.
* **Mastcam-Z** (v0p13, experimental): one camera per eye, zoom and focus bin
  (v0p14.4: ``ZL034_F02312``, bins of <= 30 focus motor counts), so focus
  breathing is refined as one focal length per bin (:mod:`mppp.sfm.zcam`).
* **One rig**: Navcam left is the reference sensor; the right camera's
  ``sensor_from_rig`` is the median left->right pose of all CAHV pairs (it
  varies by micro-radians / micrometres).  Because every resolution uses the
  same two cameras, the offset is the same for all stereo pairs.
* **Frames** = exposures (shared SCLK); left-only exposures are frames with one
  image.  Extrinsics are initialised from CAHV (``MPPPImage.pose``).
* **Weighting**: a keypoint at downsample scale s has a full-resolution
  standard deviation sigma/s, which the custom bundle adjustment uses (COLMAP's
  own BA weights every residual equally, which would over-weight quarter-
  resolution frames 16x).
"""
from .project import SfmProject, camera_from_metashape_xml, select_best_products  # noqa: F401
from .database import extract_features, build_database  # noqa: F401
from .pairs import prior_overlap_pairs  # noqa: F401
from .matching import match  # noqa: F401
from .reconstruction import reconstruct, bundle_adjust, initial_reconstruction  # noqa: F401
from .export import export_for_error, plot_camera_shifts, pose_residual_table  # noqa: F401
from .gpu import check_ba_environment, check_gpu_python  # noqa: F401  (v0p11)
from .health import assess_alignment, health_table, write_health  # noqa: F401  (v0p13)
from .zcam import focus_breathing_table, write_focus_breathing  # noqa: F401  (v0p14.4)
