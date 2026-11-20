"""
rpp_studies - the research studies built on the RPP pipeline.

The core package :mod:`rpp` is the pipeline: PDS products in, processed images, the COLMAP project and its
alignment with the promoted camera models, the dense products (mesh, depth, photometric cube, the stereoscopic
panorama).  Everything that *produces* those camera models, or analyses the pipeline's output, is a study here:

* :mod:`rpp_studies.calibration` - the camera-model solvers (the Navcam tied joint, the Mastcam-Z joint, the
  Hazcam fits), their gates and promotion, the deck-target adjustment, the fiducial and backlash-state studies, and
  the chain driver of the freeze runs.  Its products are the promoted model files the pipeline reads.
* :mod:`rpp_studies.error` - the reconstruction-error model and the tie-point error analysis of aligned blocks.
* :mod:`rpp_studies.mask_training` - training, auditing and releasing the rover-mask model the pipeline runs.
* :mod:`rpp_studies.sites` - the site-map figures and the official blocks of rover stations.
* :mod:`rpp_studies.msl` - Curiosity (MSL) blocks: the first alignments and camera models of a second rover.
* :mod:`rpp_studies.validation` - the independent checks of the camera and rig models, and their ledger.
* :mod:`rpp_studies.paper` - the figures and the build of the papers.
* :mod:`rpp_studies.misc` - small tools: the display-color study, the disk-space report, the archive-gap report,
  the LightGlue check, the time-of-day histogram.

The studies are part of the source repository, not of the installed package (``pip install rover-photogrammetry``
installs :mod:`rpp` alone); a checkout installed with ``pip install -e .`` has both.

The studies import the core; the core never imports a study (the few optional hooks, such as the deck adjustment
in the alignment, import lazily by name and fail with a clear message when the study is not installed).
"""
__all__ = ["calibration", "error", "mask_training", "sites", "msl", "validation", "paper", "misc"]
