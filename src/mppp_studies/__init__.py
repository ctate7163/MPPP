"""
mppp_studies - the research studies built on the MPPP pipeline (0.96.0).

The core package :mod:`mppp` is the pipeline: PDS products in, processed images, the COLMAP project and its
alignment with the promoted camera models, the dense products (mesh, depth, photometric cube, the stereoscopic
panorama).  Everything that *produces* those camera models, or analyses the pipeline's output, is a study here:

* :mod:`mppp_studies.calibration` - the camera-model solvers (the Navcam tied joint, the Mastcam-Z joint, the
  Hazcam fits), their gates and promotion, the deck-target adjustment, the fiducial and backlash-state studies, and
  the chain driver of the freeze runs.  Its products are the promoted model files the pipeline reads.
* :mod:`mppp_studies.error` - the reconstruction-error model and the tie-point error analysis of aligned blocks.
* :mod:`mppp_studies.mask_training` - training, auditing and releasing the rover-mask model the pipeline runs.
* :mod:`mppp_studies.sites` - the site-map figures and the official blocks of rover stations.
* :mod:`mppp_studies.misc` - what did not clearly belong anywhere; a study that stays unused here is clutter.

The studies import the core; the core never imports a study (the few optional hooks, such as the deck adjustment
in the alignment, import lazily by name and fail with a clear message when the study is not installed).
"""
__all__ = ["calibration", "error", "mask_training", "sites", "misc"]
