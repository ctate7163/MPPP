"""
mppp.calib - the camera models of the pipeline: the frozen forms, the promoted models, their constraint to an
image or block, and their fits.

The forms (:mod:`mppp.calib.forms`):

* **Navcam** (family N, eyes L and R): COLMAP ``THIN_PRISM_FISHEYE`` at the full 5120 x 3840 frame with sx1 = sy1 = 0
  held.  fx and fy are linear in the camera temperature (ppm/degC about T0), cx and cy too (px/degC), and so is k1
  (per degC); fx, fy, cx, cy carry offsets for the images before the early-mission sol.  NL is the rig reference
  (it keeps the label CAHVORE pointing and position); the stereo rig (NR from NL) has one yaw for every temperature
  and a rotation drift linear in sol (with the knot of the pitch rate).
* **Hazcam** (families F and R): the same form, not calibrated yet; each image starts from the fisheye fitted to its
  own label CAHVORE (:mod:`mppp.calib.labelfit`, 0.93), a block from the median of its images' fits.
* **Mastcam-Z** (family Z, zooms 34, 48, 63, 79, 110 mm): one calibration per zoom, ``FULL_OPENCV`` at 1648 x 1200
  with p1 = p2 = k3 = 0 held, a function of the focus count and the backlash state only: fx and fy are lines in
  focus count per eye and state, the principal point follows the ``pp`` terms about ``pp0_px``, k1 and k2 are
  constant; no temperature term.  A zoom without a promoted file starts from the label CAHVOR of each image.  An
  image is in the high-focal-length ``"backlash"`` state unless the low-backlash list, the found list or its own
  focal fit puts it in the low-focal-length ``"regular"`` state.

:mod:`mppp.calib.models` loads the promoted models (``mppp.paths.cmods_dir()``) and promotes new ones;
:mod:`mppp.calib.constrain` gives the camera of one image or block; :mod:`mppp.calib.fit` fits the forms to solved
blocks (one streamlined fit per form) and gates the candidates.
"""
from .forms import (FULL_FRAME, FULL_OPENCV_PARAMS, HAZCAM, INTRINSIC_CORE, NAVCAM, NAVCAM_PARAMS, PARAM_NAMES, ZCAM,  # noqa: F401
                    ZCAM_DEFAULT_STATE, ZCAM_LINE_FORM, ZCAM_LINE_PARAMS, ZCAM_STATES, ZCAM_ZOOMS, Form, form_for, validate,
                    zcam_line_form)
from .models import (NAVCAM_FILES, check_form, describe, hazcam_start_model, load_camera, load_rig, load_zcam,  # noqa: F401
                     load_zcam_all, models_fingerprint, navcam_fingerprint, promote, write_model, zcam_files,
                     zcam_fingerprint)
from . import labelfit  # noqa: F401,E402  (the Navcam / Hazcam label CAHVORE fit, 0.93)
from .constrain import (camera_at, drift_offset_mdeg, early_share, focus_state, label_camera, rig_at, scaled,  # noqa: F401
                        sol_epochs, thermal_scale, thermal_terms, zcam_camera_at, zcam_line_value, zcam_lines_at)

__all__ = [
    "Form", "NAVCAM", "HAZCAM", "ZCAM", "PARAM_NAMES", "NAVCAM_PARAMS", "FULL_OPENCV_PARAMS", "FULL_FRAME",
    "INTRINSIC_CORE", "ZCAM_ZOOMS", "ZCAM_STATES", "ZCAM_DEFAULT_STATE", "ZCAM_LINE_FORM", "ZCAM_LINE_PARAMS",
    "form_for", "validate", "zcam_line_form",
    "NAVCAM_FILES", "load_camera", "load_rig", "load_zcam", "load_zcam_all", "zcam_files", "hazcam_start_model",
    "models_fingerprint", "navcam_fingerprint", "zcam_fingerprint", "check_form", "describe", "promote", "write_model",
    "camera_at", "scaled", "rig_at", "drift_offset_mdeg", "early_share", "sol_epochs", "thermal_terms", "thermal_scale",
    "zcam_camera_at", "zcam_lines_at", "zcam_line_value", "focus_state", "label_camera",
]
