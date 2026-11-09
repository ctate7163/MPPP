"""
mppp_studies.calibration - the camera-model solvers of MPPP and the chain that runs them (0.96.0: moved out of
``mppp.calib`` and ``mppp.sfm``; the core keeps only the forms, the promoted models and their constraint to an
image or block, :mod:`mppp.calib`).

* :mod:`.fit` - the Navcam consensus (``fit_navcam`` / ``write_navcam`` / ``check_navcam`` / ``promote_navcam``), the
  Mastcam-Z per-zoom fit from solved blocks (``fit_zcam`` / ``check_zcam``), the rig drift fit;
* :mod:`.joint`, :mod:`.tied` - the one-adjustment Navcam joint over many blocks and its tied form;
* :mod:`.ztied` - the Mastcam-Z joint (focus lines per eye and backlash state, the sequence offsets, the pooled
  diagnostic pass, the weak-sequence offsets, the gates);
* :mod:`.hazfit`, :mod:`.hazjoint` - the Hazcam fits;  :mod:`.nzjoint` - the N+Z check;  :mod:`.lawcheck` - a
  promoted law against a block;
* :mod:`.fiducials`, :mod:`.deck` - the surveyed rover-deck targets: detection, the deck adjustment in the BA, the
  deck rig trend;  :mod:`.zcam_states` - the Mastcam-Z backlash state of every image from the blocks;
* :mod:`.rover_frame` - the rover body pose per Hazcam (study 3A);
* the drivers: :mod:`.run_pipeline` (the calibration chain of a freeze run), :mod:`.freeze_chain`,
  :mod:`.promote_cmods`; the reports :mod:`.zcam_mast_head`, :mod:`.zcam_aspect_geometry`, :mod:`.deck_rig_trend`,
  :mod:`.fiducial_check`, :mod:`.tie_point_studies`, :mod:`.lightglue_ab`.

Run a tool as a module: ``python -m mppp_studies.calibration.run_pipeline --root D:\\rover_blocks ...``; the ``mppp
calib`` and ``mppp zcam-states`` commands of the core CLI forward here.
"""
