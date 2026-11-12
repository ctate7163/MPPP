"""
mppp_studies.validation - independent checks of the camera and rig models (0.97.0).

Christian, 12 Nov 2026: "Independent validation is very important. keep track of these and quantify them for the
final report."  Each test here uses a measurement the joints do not: Mastcam-Z stereo pairs alone (``zpairs``, test
3 of the 12 Nov note), single Navcam frames on the surveyed deck targets (``deck_focal``, test 4).  ``ledger`` keeps
the list of claims and their independent evidence (``ledger.json``) and writes the table of the final report.
"""
