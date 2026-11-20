"""
rpp_studies.misc - small tools that belong to no study.  Nothing here is imported by the core.

* ``color_enhance`` - three automatic display transforms of a 16-bit linear panorama (the study behind the
  viewer's color views);
* ``free_space`` - what takes the space in a blocks root, and the removal of what can be made again;
* ``archive_gaps`` - stereo exposures with one eye missing from a local PDS archive, and their download;
* ``lightglue_check`` - does this Python's pycolmap match with LightGlue on the GPU (run by install_lightglue.bat);
* ``lmst_histogram`` - the time of day of the processed images, per site.

The studies that were here until 0.99 (the pre-0.91 calibration scripts, the lens and temperature-bin experiments,
the 0.52 / 0.53 Navcam calibration notes) are in the git history at the tag ``pre-v1.0-archive``.
"""
