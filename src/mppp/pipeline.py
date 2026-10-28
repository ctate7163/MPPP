"""
The MPPP pipeline as functions: select the PDS products of a site, process them, align them with COLMAP.

::

    from mppp.pipeline import Settings, run_site, process_site, align
    s = Settings(pds_dir="/data/m2020", scapes_root="/scapes")
    report = run_site("threeforks", s)            # select + process + align -> the health report

Every setting of a run is a field of :class:`Settings` (the defaults are the production ones).  Settings come
from, later ones winning: the site's ``settings`` in the site definitions (``mppp/data/sites.json``),
``<WORK>/mppp_settings.json``, a ``--settings`` file and ``--set NAME=VALUE`` overrides
(:func:`load_settings`; names are case-insensitive, so the upper-case names of older settings files still work).

A site's WORK folder is ``<scapes_root>/mars2020_sol_<first sol>_<site>[_zcam]`` and holds ``processed/`` (images,
masks, manifest) and one COLMAP project per variant (``colmap/``, ``colmap_<variant>/``).  :func:`run_key` names
the result a run would give (settings, camera models, alignment rules): a folder whose ``run_done.json`` records
the same key is finished and skipped by the batch scripts.

Progress goes to the ``mppp.pipeline`` logger; the command-line tool (:mod:`mppp.cli`) and
:func:`mppp.runner.run_logged` add a log file and a status file with a heartbeat.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]
log = logging.getLogger("mppp.pipeline")

SOURCES = ("pds", "processed")
STEPS = ("manifest", "project", "features", "database", "match", "reconstruct", "health", "export")
ZCAM_ZOOMS_ALL = (34, 48, 63, 79, 110)
RUN_DONE = "run_done.json"
PROCESS_DONE = "process_done.json"      # <WORK>/processed/process_done.json: written by a finished processing run
RUN_SETTINGS = "run_settings.json"

# Bumped when the alignment code changes its results under unchanged settings (a default changed in the code, not
# in the settings), so that the batch scripts align every site again instead of skipping it as done.
ZCAM_RULES = 2                   # 0.94.4: a long zoom anchored through a wider zoom with 2 looks or through its mosaic (1, 0.93.4: a partly listed focus group, unique _reg names)
ALIGN_RULES = 92                 # 0.94.4: Hazcam + Mastcam-Z points need a Navcam look (91: isolated two-view points deleted; 89: Hazcam eye from 500 observations)
# Likewise for the selection and processing rules (process_done.json).
PROCESS_RULES = 6                # 0.94.0: the shipped exclusion list before the processing (5: own stations on other sols)


def _env_path(name: str) -> Optional[Path]:
    v = os.environ.get(name)
    return Path(v) if v else None


def _env_str(name: str) -> Optional[str]:
    return os.environ.get(name) or None


def _default_gpu_python() -> Optional[str]:
    """0.91: the batch runs (run_sites, align_scape) found the GPU environment only through MPPP_GPU_PYTHON, the
    notebook through the pointer file - Chal Rocks Z34 (20 Oct) then matched 60031 pairs with COLMAP's SIFT on the
    CPU inside the run's own process for 80 min until COLMAP aborted and took the run with it (no traceback, the
    status left 'running').  Both now look in the same places."""
    v = _env_str("MPPP_GPU_PYTHON")
    if v:
        return v
    from .sfm.gpu import conda_env_python, gpu_python
    return gpu_python() or conda_env_python("mppp_gpu")


@dataclass
class Settings:
    """Every setting of a run.  Paths may be strings; ``None`` paths come from the environment where noted."""

    # --- data and output ---------------------------------------------------------------------------------------
    pds_dir: Optional[Path] = field(default_factory=lambda: _env_path("MPPP_PDS"))        # the local PDS archive (MPPP_PDS)
    scapes_root: Optional[Path] = field(default_factory=lambda: _env_path("MPPP_SCAPES"))  # one WORK folder per site below it (MPPP_SCAPES)
    gpu_python: Optional[str] = field(default_factory=lambda: _default_gpu_python())  # python.exe of a CUDA pycolmap environment for features and matching (MPPP_GPU_PYTHON, else MPPP_GPU_PY, %LOCALAPPDATA%\mppp\gpu_python.txt from install_lightglue.bat, <conda>\envs\mppp_gpu); None: the CPU
    colmap_bat: Optional[str] = field(default_factory=lambda: _env_str("COLMAP_BAT"))      # COLMAP.bat for open_in_colmap.bat; None: search the usual places (COLMAP_BAT)
    workers: int = 8                              # image-processing worker processes (8 since 0.94.4, Christian)
    gui_files: bool = True                        # write colmap_gui.ini and open_in_colmap.bat into the project (22 Oct: on by default - batch runs had none)

    # --- site ----------------------------------------------------------------------------------------------------
    sites_file: Optional[Path] = None             # the site definitions (name -> sols); None: mppp/data/sites.json
    sites_extra: Dict[str, Any] = field(default_factory=dict)   # more sites for this run, {"my_site": (700, 712)}
    source: str = "pds"                           # "pds": select and process (reusing what is processed); "processed": align <WORK>/processed as it is
    variant: str = ""                             # "" -> results in WORK/colmap; "name" -> WORK/colmap_name (starts from colmap/'s features and matches)
    exclude: List[str] = field(default_factory=list)   # images left out, as in WORK/exclude_images.txt: "S032D1184", "sol:658", "seq:NCAM08111", "ZR0_0690_*"
    reuse_matches: bool = True                    # keep the matches of the previous database.db (same features and matching settings)
    add_nearby_waypoints: float = 10.0            # also every waypoint station within this many metres of a station of the sol range, any sol; 0: off
    nearby_from_first_sol: bool = False           # the nearby stations add no image from before the block's first sol
    sequence_prefixes: Tuple[str, ...] = ("NCAM",)   # Navcam sequences kept (SAPP sun images, SCAM support images dropped); "ZCAM" is added with Mastcam-Z

    # --- processing ----------------------------------------------------------------------------------------------
    reprocess_all: bool = False                   # process every image again (new masks, radiometry, poses) and rebuild the project
    keep_only_remaining: bool = False             # only the images still in processed/images_png8 (deleted ones stay out); False: every selected product
    sky_elevation_deg: Optional[float] = 20.0     # Navcam frames pointing higher than this (sky surveys) are not processed
    lmst_window_h: Optional[Tuple[float, float]] = (6.0, 18.0)   # images taken outside this local mean solar time window are not processed
    max_saturated_fraction: Optional[float] = 0.2  # images with more than this fraction of their valid pixels saturated are not processed
    navcam_min_frame_fraction: Optional[float] = 1 / 3   # sub-frames / tiles covering less of the detector frame are left out (the full frame is kept)
    subframe_families: Tuple[str, ...] = ("N",)   # the camera families the sub-frame rule applies to
    min_included_fraction: Optional[float] = None  # images whose mask keeps less than this share of the image are left out of the alignment; None: keep all
    max_exposure_ms: Optional[float] = 40.0       # Navcam frames with a longer exposure are not processed (the long member of the exposure brackets)
    max_centre_tint: Optional[float] = 1.2        # nor frames whose centre is bluer than the edge by more than this (B/R ratio)
    navcam_brightness: float = 0.9                # brightness factor for Navcam
    no_mask_inference_at: List[str] = field(default_factory=list)   # stations (S032D1184) or sequences (NCAM00699; 0.93.8) where the mask model is off (the rover stays in the images); per site in sites.json
    store_mask_in_alpha: Any = 0.5                # the mask in the PNG alpha channel: 1 = masked pixels transparent, 0.5 = half, 0 = no alpha
    infer_mask: bool = True                       # run the mask model (False: no terrain masks)
    mask_checkpoint: Optional[str] = None         # a mask model or checkpoint file; None: the released default

    # --- cameras -------------------------------------------------------------------------------------------------
    navcam_cameras: Optional[Path] = None         # a folder of Navcam camera models to start from; None: the models in use (mppp.calib)
    navcam_intrinsics: str = "auto"               # "refine", "hold" (keep the start cameras) or "auto" (hold where the network is weak)
    navcam_rig: str = "consensus"                 # rig rotation start: "consensus" (the promoted rig) or "cahv"; the baseline is CAHV
    navcam_rig_refine: str = "refine"             # "refine" (rotation), "auto" (strong network only), "hold", or "full" (rotation and translation, baseline studies)
    navcam_rig_yaw: str = "hold"                  # one constant rig yaw held in every block ("hold") or refined with the rig ("refine")
    navcam_rig_roll: str = "hold"                 # 0.91: the rig roll held at the rig file's trend ("hold"; per epoch it is poorly determined) or refined ("refine")
    navcam_distortion_fit: str = "hold"           # Navcam distortion held at the models in every block ("hold") or fitted per block ("refine")
    navcam_pp: str = "hold"                       # the Navcam principal point held at the models ("hold") or refined per block ("refine")
    navcam_aspect: str = "hold"                   # every Navcam camera keeps fy / fx at its start ("hold") or refines fx, fy apart ("free")
    round1_hold: bool = True                      # round 1 adjusts poses and points only; the cameras refine from round 2 on
    rig_translation_sigma_m: float = 0.005        # with navcam_rig_refine "full": the prior on the right camera centre per axis (m)
    tangential: str = "refine"                    # Navcam p1, p2 when the distortion is refined: "refine" or "zero"
    zcam_k3: str = "label"                        # Mastcam-Z k3: "label" (the start camera's, held) or "zero"
    zcam_tangential: Optional[str] = "refine"     # Mastcam-Z p1, p2: "refine" (0.91, zcam-v2: per eye, constant over focus), "zero" (zcam-v1), or None (as tangential)
    hazcams: Tuple[str, ...] = ("F", "R")         # 0.93: the Hazcam frames of the block's stations: "F" (front), "R" (rear); () = Navcam only (Christian 22 Oct: in every Navcam block)
    hazcam_intrinsics: str = "radial"             # what a block refines of each Hazcam eye: "radial" (fx, fy, k1, k2), "focal", "all" or "hold" (the start is the label CAHVORE fit)
    hazcam_rig_refine: str = "rotation"           # the Hazcam stereo rigs: "rotation" (refined when the Hazcams come in; a pair of weak eyes stays at its start) or "hold"
    include_zcam: bool = False                    # add the Mastcam-Z frames of the same sols (folder ..._zcam)
    zcam_zooms: Any = None                        # their zooms (mm), e.g. [34, 48]; None: the site's "zcam" list ([34] without one); "all": every zoom
    zcam_zooms_only: Any = None                   # 0.95.0 (Christian, 28 Oct: 79/110 mm made Chal Rocks 4.7x longer): keep only these zooms of the site's list, e.g. [34, 48, 63] for the freeze runs (run_pipeline --zooms passes its list); None: all of the site's
    zcam_intrinsics: str = "focus_model"          # Mastcam-Z start: "focus_model" (the promoted model) or "label"
    zcam_focus_model: Optional[Path] = None       # a focus-model JSON to start from; None: the one in use
    zcam_focus_bin: Optional[float] = 30          # focus motor counts per Mastcam-Z camera (None: one camera per eye)
    zcam_bin_by: str = "sequence"                 # 0.94.4: the Mastcam-Z cameras - "focus" (focus bins), "sequence" (focus bins of each sol and sequence), "image" (one camera per image: its own focal length - the states run); 0.95.0 (Christian, 27 Oct): "sequence" - each sequence's bins carry its own focus offset (mppp.calib.ztied)
    zcam_bin_refine: str = "focal_pp"             # a focus-bin camera refines fx, fy and on ZR cx, cy ("focal_pp"); "focal", "all", "model"
    zcam_line_params: Dict[str, List[str]] = field(default_factory=lambda: {"ZL": ["fx", "fy", "k1", "k2", "p1", "p2"],
                                                                            "ZR": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]})   # the focus parameters stage 2 puts on a line per eye
    zcam_hold_f_images: int = 2                   # focus bins with at most this many images hold f at the focus model
    zcam_min_zoom_images: int = 6                 # 0.95.0 (Christian, 27 Oct): an eye and zoom with fewer images at a site keeps every bin at the model (Taylorfjellet's one 48 mm pair: a free principal point went +90 px)
    # 0.93.6 (Christian, 23 Oct): a frame is in the high backlash state unless it is on the low-state list - the
    # list's frames get their own (low-state) cameras; a block's focal fits only report groups that look low
    zcam_backlash: Optional[str] = "list"         # focus backlash states: "list", "list_split", "split", "report" or None
    zcam_low_backlash: Any = None                 # the low-backlash observation list; None: the shipped CSV; False: none
    # 0.94.4 (Christian, 25 Oct: "always use this list"): the found list again, with the shipped one - curated by
    # hand and by `mppp zcam states --apply` ("-" lines take frames off; 0.93.6 had it off after Seitah Overlook's
    # high-state pointings 2 and 4 of ZCAM08114 got in); "auto" = <scapes_root>/zcam_low_backlash_found.csv
    zcam_low_backlash_found: Any = "auto"
    zcam_focus_line: bool = True                  # stage 2 puts the focus-bin cameras on lines in focus count (False: f per bin)
    zcam_uncalibrated_matching: bool = True       # 0.95.0 (Christian, 28 Oct): verify the Mastcam-Z pairs uncalibrated (fundamental matrix) from the start - COLMAP's calibrated estimators abort on a pair whose keypoints fail to unproject, and the pass before the retry was lost; False: calibrated with the retry (a Mastcam-Z setting: Navcam keys stay)
    zcam_line_zooms: Any = (34, 48, 63)           # 0.95.0 (Christian, 28 Oct): the zooms whose bins go on focus lines; 79 and 110 mm keep their refined values per bin until they have real models (Chal Rocks ZR079 fy 42 px rms)
    zcam_line_cycles: int = 2                     # rounds of freeing the bins and refitting the lines
    ba_points_cap: int = 0                        # 0.95.0 (Christian, 28 Oct; study S2): at most this many two-view same-family same-station points per image in the bundle adjustments, every tie kept (mppp.sfm.reconstruction.thin_points); 0: every point (4-5 M at Chal Rocks)
    zcam_line_settle_px: float = 0.1              # 0.95.0 (Christian, 28 Oct): the remaining stage-2 rounds are skipped once a refit moves the focus lines by less than this (px at the bins) - two rounds instead of three on a settled block; 0: never (a Mastcam-Z setting: Navcam keys stay)
    zcam_station_prior_sigma_m: float = 0.1       # 0.94.5 (27 Oct): a Mastcam-Z frame's position prior moves with its station's Navcam shift, +-this per axis (m); 0: the waypoint prior (prior_sigma_m)
    zcam_line_form: Optional[Dict[str, Any]] = None   # the form of each focus line {eye: {param: form}} for a study; None: the shipped form
    zcam_rig: bool = True                         # the two Mastcam-Z eyes of an exposure form a stereo rig, one per zoom
    zcam_rig_fit: Optional[str] = "rotation"      # the Mastcam-Z rig refined in stage 2: "rotation", "full" (and translation), None
    zcam_rig_refine: str = "refine"               # "refine" (from the model's rig), "auto" (held when the model has one), "hold"
    pose_guided: Any = True                       # after the stage-2 rounds, match again guided by the poses (a dict overrides mppp.sfm.guided.GUIDED_DEFAULTS)
    isolated_points_factor: float = 4.0           # 0.94.4: before the final stage-2 adjustment, delete the two-view points more than this x the median spacing (at their range) from their 8 nearest points - the scattered false points above and below the terrain; 0 = off
    thermal_after_stage1: bool = True             # the Navcam temperature bins are fitted at the end of stage 1, before Mastcam-Z comes in
    navcam_rig_epoch_gap_sols: float = 50         # Navcam visits more than this many sols apart get their own rig rotation
    navcam_rig_epoch_breaks_sols: Tuple[float, ...] = (300,)   # images on both sides of these sols get separate rig rotations

    # --- features and matching -----------------------------------------------------------------------------------
    max_num_features: Any = field(default_factory=lambda: {"N": 65536, "*": 32768})   # SIFT features per image, by family ("*": the rest); 0.95.0 (Christian, 27 Oct): twice the 32768 on the full-resolution Navcam frames, for the Mastcam-Z ties
    sift: Dict[str, Any] = field(default_factory=lambda: dict(domain_size_pooling=False, estimate_affine_shape=False,
                                                              num_octaves=None, first_octave=-1))   # num_octaves None: 6 with Mastcam-Z, else 4
    match_mode: str = "exhaustive"                # the matcher (0.94.4, Christian 23 Oct: SIFT again - the LightGlue A/B on Chal Rocks tiny: 13.6k Navcam-Mastcam-Z ties with SIFT, 7.6k with LightGlue, residual median 0.22 vs 0.37 px; 0.93.9 had LightGlue): "lightglue" (SIFT + LightGlue on the GPU; COLMAP's SIFT matcher when torch / kornia are missing or it would take over 3 h), "exhaustive" (COLMAP's SIFT matcher), "sift+lightglue" (0.94.4: both, LightGlue's inliers added to SIFT's) or "prior_pairs" (quick / CPU); over max_exhaustive_images images the prior-overlap pairs
    max_exhaustive_images: int = 300              # 21-22 Oct: above this many images "exhaustive" matches the prior-overlap pairs instead (Issole Z34: 2713 images, 3.7 M pairs, > 9 h; 300 from 22 Oct, Christian); 0 = never (not in the run key)
    exclude_list: bool = True                     # 22 Oct: leave out the images of mppp/data/exclude_images.txt (Mastcam-Z movies and other observations unfit for photogrammetry); False: keep them (not in the run key)
    match: Dict[str, Any] = field(default_factory=lambda: dict(max_ratio=0.9, max_distance=1.0, cross_check=True,
                                                               guided_matching=False, max_error_px=6.0,
                                                               max_keypoints=8192, filter_threshold=0.1))   # matching options (the last two: LightGlue)
    lightglue_keypoints: str = "largest"          # 0.94.4: which max_keypoints keypoints LightGlue sees: "largest" scales, or "octaves" (a share of every octave; in the run key only when not "largest")
    match_block_size: int = 100                   # images per exhaustive-matching block
    export_level: str = "full"                    # 0.95.0 (Christian, 28 Oct): "light" leaves out what a calibration run does not use - the native-pixel text model and residuals of error_input/, error_input_navcam/ and the COLMAP GUI copy (gui_native/); "full": everything (not in the run key; run_pipeline's site runs are light unless --export)
    keep_databases: bool = False                  # 0.95.0: keep the run's database copies (database_thermal.db, database_guided.db; without descriptors) after the run - deleted by default (not in the run key)

    # --- reconstruction ------------------------------------------------------------------------------------------
    schedule: Any = ((24.0, 10.0, 8.0), (12.0, 4.0, 4.0), (8.0, 2.0, 2.0))   # rounds of (triangulation threshold px, Cauchy scale, max residual kept px)
    min_tri_angle_deg: float = 0.25               # keeps stereo-only points out to ~97 m at the 0.42 m baseline
    triangulation: Dict[str, Any] = field(default_factory=lambda: dict(max_transitivity=1))   # COLMAP track building
    attitude_prior_deg: float = 2.0               # CAHV attitude prior per frame
    localize_min_images: int = 3                  # stations with fewer images get no waypoint position prior
    prior_sigma_m: float = 1.0                    # waypoint position prior, 1-sigma per axis (m)
    no_position_prior_at: List[str] = field(default_factory=list)   # stations (S###D####) whose position priors are left out (flagged localisations)
    prior_gauge_recentre: bool = True             # after each adjustment the block is moved onto its position priors (their mean residual)
    exclude_outliers: bool = True                 # drop frames that do not align with the majority, then re-adjust
    outlier_thresholds: Dict[str, Any] = field(default_factory=dict)   # e.g. {"residual_factor": 3.0, "min_shift_m": 0.25}
    min_frame_observations: int = 20              # frames with fewer tie-point observations are held at their prior and excluded as outliers
    staged: bool = True                           # Navcam block first, then Mastcam-Z with the Navcam cameras and rig held
    hold_cameras: Tuple[str, ...] = ()            # further camera keys held at their start values, e.g. ("ZL034_f0",)
    linear_solver: str = "auto"                   # "auto" (dense Schur up to 600 frames, else iterative), "sparse_schur", "dense_schur", "iterative_schur"
    thermal_bins_deg: Optional[float] = 5         # Navcam camera-temperature bins for the final stage (one camera per eye and bin); None: off
    thermal_fine_range: Any = (-25.0, -5.0)       # finer bins in this temperature range
    thermal_fine_deg: Any = 2.5                   # ... this wide; None: uniform bins
    thermal_min_images: int = 5                   # a bin with fewer images is merged into the nearer neighbouring bin
    thermal_free: Tuple[str, ...] = ("fx", "fy", "k1")   # refined per bin (unless the Navcam intrinsics are held)

    # --- helpers
    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in PATH_FIELDS and isinstance(v, str):
                setattr(self, f.name, Path(v))
            if f.name in TUPLE_FIELDS and isinstance(v, list):
                setattr(self, f.name, tuple(v))
        self.hazcams = hazcam_families(self.hazcams)

    def replace(self, **changes: Any) -> "Settings":
        """A copy with ``changes`` (case-insensitive names) applied."""
        return dataclasses.replace(self, **normalize(changes))

    def to_dict(self) -> Dict[str, Any]:
        """Every field, JSON-ready (paths as strings, tuples as lists)."""
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))

    def key_dict(self) -> Dict[str, Any]:
        """The fields that decide a run's result (machine paths and bookkeeping left out).  The Hazcam settings
        (0.93) count only for a run with Hazcams, so the keys of every other run stay what they were."""
        d = {k: v for k, v in self.to_dict().items() if k not in NOT_IN_KEY}
        if not d.get("hazcams"):
            for k in HAZCAM_FIELDS:
                d.pop(k, None)
        if d.get("lightglue_keypoints") in (None, "largest"):
            d.pop("lightglue_keypoints", None)       # 0.94.4: the keys of every run before it stay what they were
        if d.get("zcam_bin_by") in (None, "focus"):
            d.pop("zcam_bin_by", None)               # 0.94.4: likewise
        if not d.get("zcam_zooms_only"):
            d.pop("zcam_zooms_only", None)           # 0.95.0: unset, the keys stay what they were
        if not d.get("include_zcam"):
            d.pop("zcam_zooms_only", None)           # 0.95.0: a Mastcam-Z setting (Navcam keys stay)
            d.pop("zcam_line_zooms", None)           # 0.95.0: likewise
            d.pop("zcam_uncalibrated_matching", None)   # 0.95.0: likewise
            d.pop("zcam_line_settle_px", None)       # 0.95.0: likewise
            d.pop("zcam_bin_by", None)               # 0.95.0: a Mastcam-Z setting (Navcam keys stay)
            d.pop("zcam_station_prior_sigma_m", None)   # 0.94.5: a Mastcam-Z setting (Navcam keys stay)
            d["zcam_low_backlash_found"] = None      # 0.94.4: the found list is a Mastcam-Z setting (Navcam keys stay)
            if d.get("exclude"):                     # 0.94.4: a Mastcam-Z sequence left out does not change a Navcam run
                d["exclude"] = [e for e in d["exclude"] if not str(e).strip().lower().startswith("seq:zcam")]
        return d


PATH_FIELDS = {"pds_dir", "scapes_root", "sites_file", "navcam_cameras", "zcam_focus_model"}
TUPLE_FIELDS = {"sequence_prefixes", "subframe_families", "hold_cameras", "thermal_free", "navcam_rig_epoch_breaks_sols",
                "hazcams", "zcam_line_zooms"}
HAZCAM_FIELDS = ("hazcams", "hazcam_intrinsics", "hazcam_rig_refine")
# the Hazcam products of a family: camera codes and the sequences kept (Hazcam frames come in FHAZ / RHAZ sequences)
HAZCAM_CODES = {"F": ("FLF", "FRF"), "R": ("RLF", "RRF")}
HAZCAM_SEQUENCES = {"F": ("FHAZ",), "R": ("RHAZ",)}
HAZCAM_VARIANT = "hazcam"                         # the Hazcam-only alignments' variant (the Navcam held)


def hazcam_families(v: Any) -> Tuple[str, ...]:
    """The ``hazcams`` setting as a tuple of families ("F", "R"): accepts "F", "R", "FR", "F,R", "front", "rear",
    "both", "all", a list of these, or nothing (``()``, None, False, "", "none")."""
    if v is None or v is False or v == "":
        return ()
    items = [v] if isinstance(v, str) else list(v)
    out = set()
    for it in items:
        t = str(it).strip().lower()
        if t in ("", "none", "no", "false", "0"):
            continue
        if t in ("both", "all", "true", "1", "fr", "rf", "f,r", "r,f", "front,rear", "rear,front"):
            out |= {"F", "R"}
        elif t in ("f", "front", "fhaz", "fcam", "front_hazcam"):
            out.add("F")
        elif t in ("r", "rear", "rhaz", "rcam", "rear_hazcam"):
            out.add("R")
        elif "," in t:
            out |= set(hazcam_families(t.split(",")))
        else:
            raise ValueError(f"hazcams = {v!r}: use 'F' (front), 'R' (rear) or both")
    return tuple(f for f in ("F", "R") if f in out)
NOT_IN_KEY = {"pds_dir", "scapes_root", "gpu_python", "colmap_bat", "workers", "gui_files", "sites_file",
              "source", "variant", "max_exhaustive_images", "exclude_list", "keep_databases", "export_level"}
SETTING_NAMES = tuple(f.name for f in fields(Settings))


def setting_names(upper: bool = False) -> List[str]:
    """The setting names (upper case: as older settings files and ``sites.json`` write them)."""
    return [n.upper() if upper else n for n in SETTING_NAMES]


def normalize(d: Dict[str, Any]) -> Dict[str, Any]:
    """``{NAME: value}`` with the names lower-cased and checked; a ``py:`` string value (from ``--set``) is
    evaluated as a Python literal."""
    out = {}
    for k, v in d.items():
        name = str(k).lower()
        if name == "include_zcam34":                   # the name before zcam_zooms
            name, v = "include_zcam", bool(v)
        if name not in SETTING_NAMES:
            raise KeyError(f"unknown setting {k!r}; the settings are {', '.join(SETTING_NAMES)}")
        if isinstance(v, str) and v.startswith("py:"):
            v = _python_value(v[3:].strip())
        out[name] = v
    return out


def _python_value(src: str) -> Any:
    """A ``--set`` value written as Python (``dict(max_ratio=0.85)``, ``((24, 10, 8),)``): literals and the
    ``dict`` / ``list`` / ``tuple`` / ``Path`` constructors, nothing else."""
    import ast
    try:
        return ast.literal_eval(src)
    except (ValueError, SyntaxError):
        pass
    names = {"dict": dict, "list": list, "tuple": tuple, "set": set, "Path": Path, "True": True, "False": False,
             "None": None}
    return eval(compile(src, "<set>", "eval"), {"__builtins__": {}}, names)     # noqa: S307 - the user's own option


def load_settings(work: Optional[PathLike] = None, site: Optional[str] = None,
                  settings_file: Optional[PathLike] = None, sets: Optional[Dict[str, Any]] = None,
                  sites_file: Optional[PathLike] = None, base: Optional[Settings] = None) -> Settings:
    """
    The settings of a run, later ones winning: ``base`` (default :class:`Settings`), the site's ``settings`` (and
    ``no_mask_inference_at``) in the site definitions, ``<work>/mppp_settings.json``, ``settings_file``, ``sets``
    (``--set NAME=VALUE``, parsed by :func:`mppp.runner.parse_set`).
    """
    from .sfm.workdir import SETTINGS_FILE, load_settings as read_json
    from .waypoints.sites import load_site_table
    over: Dict[str, Any] = {}
    rec = (load_site_table(sites_file).get("sites", {}).get(site) or {}) if site else {}
    if isinstance(rec, dict):
        if rec.get("no_mask_inference_at"):
            over["no_mask_inference_at"] = list(rec["no_mask_inference_at"])
        over.update(rec.get("settings") or {})
    if work:
        over.update(read_json(Path(work) / SETTINGS_FILE))
    if settings_file:
        over.update(read_json(settings_file))
    over.update(sets or {})
    if sites_file:
        over["sites_file"] = Path(sites_file)
    s = base if base is not None else Settings()
    return s.replace(**over)


work_settings = load_settings


# ----------------------------------------------------------------------------------------------------- run keys
EXCLUDE_LIST_FILE = "exclude_images.txt"


def shipped_exclusions() -> List[str]:
    """The entries of the shipped exclusion list (``mppp/data/exclude_images.txt``): Mastcam-Z movies and other
    observations that do not work for photogrammetry, left out of every alignment unless ``exclude_list`` is False."""
    from .sfm.workdir import read_exclusions
    return read_exclusions(Path(__file__).resolve().parent / "data" / EXCLUDE_LIST_FILE)


def effective_match_mode(s: "Settings", n_images: int) -> str:
    """The matcher a block gets: ``match_mode``, except that "exhaustive" over more than ``max_exhaustive_images``
    images matches the prior-overlap pairs (n^2 / 2 pairs: 3.7 million at 2713 images)."""
    if s.match_mode == "exhaustive" and s.max_exhaustive_images and n_images > int(s.max_exhaustive_images):
        return "prior_pairs"
    return s.match_mode


def run_key(settings: Union[Settings, Dict[str, Any]], source: Optional[str] = None, variant: Optional[str] = None
            ) -> str:
    """A short hash of what decides a run's result: the settings (as :func:`resolve_site` returns them, so the
    site's Mastcam-Z zooms count), the source and variant, the alignment rules and the camera models in use (a
    promotion makes every site run again)."""
    if isinstance(settings, Settings):
        d: Dict[str, Any] = {"settings": settings.key_dict(),
                             "source": settings.source if source is None else source,
                             "variant": (settings.variant if variant is None else variant) or ""}
    else:
        d = {"settings": settings, "source": source or "pds", "variant": variant or ""}
    if not str(d["source"]).startswith("process:"):
        d["rules"] = ALIGN_RULES
        if isinstance(d["settings"], dict) and d["settings"].get("include_zcam"):
            d["zcam_rules"] = ZCAM_RULES             # a Mastcam-Z block's own rules: the Navcam-only keys stay
        try:
            from .calib import models_fingerprint
            zc = bool(isinstance(d["settings"], dict) and d["settings"].get("include_zcam"))
            d["cmods"] = models_fingerprint(include_zcam=zc)    # 27 Oct: a Navcam block's key without the Z models
        except Exception:                                                  # noqa: BLE001
            d["cmods"] = None
    txt = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(txt.encode()).hexdigest()[:16]


def process_key(settings: Union[Settings, Dict[str, Any]]) -> str:
    """The run key of a processing-only run (``process_done.json``)."""
    return run_key(settings, f"process:{PROCESS_RULES}", "")


def processed_done(work: PathLike, key: str) -> Optional[Dict[str, Any]]:
    """The ``process_done.json`` of ``work`` if it records ``key``, else None."""
    p = Path(work) / "processed" / PROCESS_DONE
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if d.get("run_key") == key else None


def run_done(work: PathLike, variant: str = "") -> Optional[Dict[str, Any]]:
    """The ``run_done.json`` of a project (None if there is none)."""
    from .sfm.workdir import project_dir
    p = project_dir(work, variant) / RUN_DONE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------------ the site
@dataclass
class Site:
    """A site resolved against the settings: its sol ranges and WORK folder."""
    name: str
    sol_ranges: List[Tuple[int, int]]        # every visit, in the order listed
    work: Path
    zcam: bool
    zooms: List[int]

    @property
    def sols(self) -> Tuple[int, int]:
        return self.sol_ranges[0][0], max(b for _, b in self.sol_ranges)

    @property
    def sols_text(self) -> str:
        return ", ".join(f"{a}-{b}" for a, b in self.sol_ranges)


def site_table(settings: Settings) -> Dict[str, Any]:
    """The site definitions with ``sites_extra`` added."""
    from .waypoints.sites import load_site_table
    t = load_site_table(settings.sites_file)
    sites = dict(t.get("sites", {}))
    for k, v in (settings.sites_extra or {}).items():
        sites[k] = {"sols": list(v)} if not isinstance(v, dict) else dict(v)
    return dict(t, sites=sites)


def zcam_zooms(settings: Settings, site_rec: Optional[Dict[str, Any]]) -> List[int]:
    """The Mastcam-Z zooms of a run: ``zcam_zooms``, the site's list, ``[34]``; ``[]`` without ``include_zcam``."""
    from .waypoints.sites import site_zooms
    if not settings.include_zcam:
        return []
    z = settings.zcam_zooms
    if z is None:
        z = site_zooms(site_rec) or [34]
    if isinstance(z, str) and z.lower() == "all":
        z = list(ZCAM_ZOOMS_ALL)
    zooms = sorted(int(x) for x in z)
    only = settings.zcam_zooms_only
    if only:                                     # 0.95.0: the run keeps the zooms being calibrated
        keep = {int(x) for x in ([only] if isinstance(only, (int, float)) else only)}
        zooms = [x for x in zooms if x in keep]
    if set(zooms) - set(ZCAM_ZOOMS_ALL):
        raise ValueError(f"zcam_zooms = {zooms}: MPPP aligns the Mastcam-Z zooms {list(ZCAM_ZOOMS_ALL)} mm")
    return zooms


def resolve_site(site_or_work: PathLike, settings: Settings) -> Tuple[Site, Settings]:
    """
    ``(site, settings)`` for a site name or a WORK folder: a folder's name (``mars2020_sol_<sol>_<site>[_zcam]``)
    names the site and whether Mastcam-Z is included; a folder of another name must hold ``processed/`` and gets
    the sol range of its manifest.  The returned settings carry ``include_zcam``, ``zcam_zooms`` and the SIFT
    octaves as derived.
    """
    from .sfm.workdir import load_manifest, sol_range
    from .waypoints.sites import check_sites, parse_work_folder, sol_ranges, work_folder
    table = site_table(settings)
    sites = table["sites"]
    for w in check_sites({k: v.get("sols") for k, v in sites.items()}):
        log.warning("sites: %s", w)
    p = Path(site_or_work)
    s = settings
    is_folder = isinstance(site_or_work, Path) or any(c in str(site_or_work) for c in "/\\") or p.is_dir()
    if is_folder:
        work = p.expanduser().resolve()
        name, zcam = parse_work_folder(work)
        if name:
            s = s.replace(include_zcam=zcam)
        name = name or work.name
        rec = sites.get(name)
        if rec is not None:
            ranges = sol_ranges(rec["sols"])
        elif (work / "processed").is_dir():
            ranges = [sol_range(load_manifest(work / "processed")[0]["images"])]
        else:
            raise KeyError(f"{work.name}: not a site of {settings.sites_file or 'mppp/data/sites.json'} and no "
                           f"processed/ folder in it")
    else:
        name = str(site_or_work)
        rec = sites.get(name)
        if rec is None:
            raise KeyError(f"site {name!r} is not in {settings.sites_file or 'mppp/data/sites.json'} or sites_extra")
        ranges = sol_ranges(rec["sols"])
        if not s.scapes_root:
            raise ValueError("no scapes root: set Settings.scapes_root (or the MPPP_SCAPES environment variable)")
        work = work_folder(Path(s.scapes_root), name, bool(s.include_zcam), rec["sols"])
    for a, b in ranges:
        if b < a:
            raise ValueError(f"site {name!r}: a visit ends before it starts ({ranges})")
    zooms = zcam_zooms(s, rec)
    if s.sift.get("num_octaves") is None:              # 6 octaves with Mastcam-Z (coarse features on the Zcam frames), else COLMAP's 4
        s = s.replace(sift=dict(s.sift, num_octaves=6 if zooms else 4))
    s = s.replace(zcam_zooms=zooms)
    return Site(name, ranges, work, bool(s.include_zcam), zooms), s


# ------------------------------------------------------------------------------------------- 1. select the products
def _own_note(rep: Dict[str, Any]) -> str:
    """" (n of them at the block's own stations on sols ...)" for a nearby-search report (0.93.3)."""
    own = rep.get("own_station_other_sols") or []
    if not own:
        return ""
    n = sum(a["images"] for a in own)
    sols = sorted({s for a in own for s in a["sols"]})
    return f" ({n} of them at the block's own stations on sols {', '.join(map(str, sols))})"


def select_products(site: Union[str, Site], settings: Settings) -> Tuple[List[Path], Dict[str, Any]]:
    """One PDS product per exposure of the site's sol ranges (and the nearby stations), with the selection
    report (``provenance`` of the manifest).  :func:`select_images` returns the paths alone."""
    from . import load_waypoints
    from .select import find_imgs, find_imgs_near
    from .sfm import select_best_products
    from .waypoints.sites import sol_ranges
    st = site if isinstance(site, Site) else resolve_site(site, settings)[0]
    if not settings.pds_dir:
        raise ValueError("no PDS archive: set Settings.pds_dir (or the MPPP_PDS environment variable)")
    pds = Path(settings.pds_dir)
    ranges = [list(r) for r in st.sol_ranges]
    near = None
    radius = float(settings.add_nearby_waypoints or 0)
    waypoints = load_waypoints(refresh="auto")                 # 0.90: the current mission table (cache <= 1 day old)
    src = waypoints["_mppp_source"]
    log.info("waypoints: %d features, %s%s", src["n_features"], src["file_time"],
             " (packaged snapshot: no download possible - new stations get the onboard estimate)" if src["packaged_snapshot"] else "")
    # 0.93.1: stations excluded by the settings (a site's EXCLUDE "S026D0282") are not selected and do not anchor
    # the nearby-waypoint search
    ex_st = [str(e).upper() for e in (settings.exclude or []) if re.fullmatch(r"S\d{3}D\d{4}", str(e).strip().upper())]
    if radius:
        min_sol = min(int(r[0]) for r in sol_ranges(ranges)) if settings.nearby_from_first_sol else None
        paths, near = find_imgs_near(pds, ["NLF", "NRF"], ranges, waypoints, radius_m=radius, min_sol=min_sol,
                                     exclude_stations=ex_st)
        if ex_st:
            log.info("stations excluded from the selection and the nearby search: %s", ", ".join(ex_st))
        log.info("Navcam: %d in sols %s + %d from other visits within %g m%s", near["n_in_range"], st.sols_text,
                 near["n_added"], radius, f" (none before sol {min_sol})" if min_sol is not None else "")
        for a in near["stations_added"]:
            log.info("   %s (%s m from %s): %s images, sols %s-%s", a["station"], a["distance_m"], a["nearest"],
                     a["images"], a["sols"][0], a["sols"][1])
        for a in near.get("own_station_other_sols") or []:            # 0.93.3
            log.info("   %s (a block station, the rover not yet moved): %s images on sols %s", a["station"],
                     a["images"], ", ".join(map(str, a["sols"])))
    else:
        paths = find_imgs(pds, ["NLF", "NRF"], sol_range=ranges)
        log.info("Navcam: %d products in sols %s", len(paths), st.sols_text)
    prefixes = tuple(settings.sequence_prefixes)
    for zoom in st.zooms:                      # every zoom of the site (sequ_id "_034", "_048", ...)
        if radius:
            # from the Navcam block's stations, so every zoom gets the same visits
            zp, znear = find_imgs_near(pds, ["ZL0", "ZR0"], ranges, waypoints, radius_m=radius,
                                       sequ_id=f"_{zoom:03d}", anchor_stations=near["stations_in_range"], min_sol=min_sol,
                                       exclude_stations=ex_st)
            log.info("Mastcam-Z %d: %d in range + %d from other visits%s", zoom, znear["n_in_range"], znear["n_added"],
                     _own_note(znear))
        else:
            zp = find_imgs(pds, ["ZL0", "ZR0"], sol_range=ranges, sequ_id=f"_{zoom:03d}")
            log.info("Mastcam-Z %d: %d products", zoom, len(zp))
        paths += zp
    if st.zooms:
        prefixes += ("ZCAM",)
    sub_fams = tuple(settings.subframe_families)
    for fam in hazcam_families(settings.hazcams):    # 0.93: the Hazcam frames of the same stations
        if radius:
            hp, hnear = find_imgs_near(pds, list(HAZCAM_CODES[fam]), ranges, waypoints, radius_m=radius,
                                       anchor_stations=near["stations_in_range"], min_sol=min_sol,
                                       exclude_stations=ex_st)
            log.info("%s Hazcam: %d in range + %d from other visits%s", "front" if fam == "F" else "rear",
                     hnear["n_in_range"], hnear["n_added"], _own_note(hnear))
        else:
            hp = find_imgs(pds, list(HAZCAM_CODES[fam]), sol_range=ranges)
            log.info("%s Hazcam: %d products", "front" if fam == "F" else "rear", len(hp))
        paths += hp
        prefixes += HAZCAM_SEQUENCES[fam]
        sub_fams += (fam,)                           # the 1/3 rule holds for the Hazcam sub-frames too
    paths, sel = select_best_products(paths, sequence_prefix=prefixes,
                                      min_frame_fraction=settings.navcam_min_frame_fraction,
                                      frame_fraction_families=sub_fams)
    log.info("selection: %s", json.dumps({k: v for k, v in sel.items() if k != "dropped_subframes"}))
    if sel.get("n_dropped_subframes"):
        log.info("%d Navcam tiles / sub-frames below %s of the frame left out, e.g. %s", sel["n_dropped_subframes"],
                 sel["min_frame_fraction"], sel["dropped_subframes"][0]["file"])
    one_eye = one_eye_exposures(paths)
    if one_eye:
        # 23 Oct (Seitah Overlook, Artuby Ridge): the local archive held only the right image of some Hazcam
        # exposures (an older J01 version) while the PDS has both eyes (J02) - download the missing eye
        log.info("%d stereo exposures with one eye only in the PDS archive (the other eye missing - download it), "
                 "e.g. %s", len(one_eye), ", ".join(one_eye[:3]))
        sel["one_eye_exposures"] = one_eye
    log.info("%d products", len(paths))
    sel["site"] = st.name
    sel["sol_ranges"] = ranges
    return paths, sel


def one_eye_exposures(paths: Sequence[PathLike]) -> List[str]:
    """The stems of Navcam and Hazcam products whose stereo partner (the other eye, same SCLK) is not among
    ``paths`` - in the M2020 archive both eyes of these exposures are normally there."""
    by: Dict[str, set] = {}
    stems: Dict[str, str] = {}
    for p in paths:
        stem = Path(p).stem.upper()
        if len(stem) < 23 or stem[0] not in "NFR" or stem[1] not in "LR":
            continue
        k = stem[0] + stem[4:23]                        # family, sol, SCLK with its milliseconds
        by.setdefault(k, set()).add(stem[1])
        stems.setdefault(k, Path(p).stem)
    return sorted(stems[k] for k, eyes in by.items() if len(eyes) == 1)


_PRODUCT = re.compile(r"^[A-Z0-9]{3}_(\d{4})_\d{10}_\d{3}[A-Z0-9]{3}_[A-Z](\d{3})(\d{4})([A-Z]{4}\d{5})_", re.I)


def product_meta(path: PathLike) -> Dict[str, Any]:
    """{stem, sol, site, drive, sequence} of a PDS product from its file name (as the manifest records them)."""
    stem = Path(path).stem
    m = _PRODUCT.match(stem)
    if not m:
        return {"filename": {"stem": stem}}
    return {"filename": {"stem": stem, "sol": int(m.group(1)), "site": int(m.group(2)), "drive": int(m.group(3)),
                         "sequence": m.group(4).upper()}}


def exclude_products(paths: Sequence[PathLike], entries: Sequence[str]) -> Tuple[List[Path], List[str]]:
    """(kept, excluded stems) of PDS products under exclusion entries (``mppp.sfm.workdir`` syntax)."""
    from .sfm.workdir import apply_exclusions
    metas = [dict(product_meta(p), _path=p) for p in paths]
    kept, removed = apply_exclusions(metas, entries)
    return [m["_path"] for m in kept], [r["image"] for r in removed if r["image"]]


def select_images(site: Union[str, Site], settings: Settings) -> List[Path]:
    """The PDS products of a site (section 1 of the pipeline): see :func:`select_products`."""
    return select_products(site, settings)[0]


# ------------------------------------------------------------------------------------------------------ 2. process
def processing_config(settings: Settings) -> Dict[str, Any]:
    """The image-processing configuration of a run (:func:`mppp.default_config` with the settings applied)."""
    from . import default_config
    cfg = default_config()
    cfg["export"].update(formats=["PNG8"], store_mask_in_alpha=settings.store_mask_in_alpha, write_mask_files=True,
                         write_colmap=False)
    cfg["masking"]["skip_inference_at"] = list(settings.no_mask_inference_at or [])
    cfg["masking"]["infer_mask"] = bool(settings.infer_mask)
    if settings.mask_checkpoint:
        cfg["masking"]["checkpoint"] = str(settings.mask_checkpoint)
    cfg["selection"].update(max_boresight_elevation_deg=settings.sky_elevation_deg,
                            lmst_window_h=list(settings.lmst_window_h) if settings.lmst_window_h else None,
                            max_saturated_fraction=settings.max_saturated_fraction,
                            max_exposure_ms=settings.max_exposure_ms, max_centre_tint=settings.max_centre_tint)
    cfg["color"]["brightness_by_family"] = {"N": settings.navcam_brightness}
    return cfg


def process_site(site_or_work: PathLike, settings: Settings) -> Path:
    """
    Select and process the images of a site (sections 1-2): one product per exposure from the PDS archive into
    ``<WORK>/processed`` (8-bit PNG, terrain mask, CAHV + waypoint pose, manifest).  Images already processed with
    the same configuration are reused (``reprocess_all`` processes them again).  Writes
    ``processed/process_done.json`` with the processing key and returns the ``processed`` folder.
    """
    from . import load_waypoints, process_images
    st, s = resolve_site(site_or_work, settings)
    paths, sel = select_products(st, s)
    if s.exclude_list:
        # 0.94.0 (Christian, 23 Oct): the shipped exclusion list also keeps its images out of the processing (they
        # were processed, then left out of the alignment: Mastcam-Z movies of a thousand frames)
        # (with the site's / run's EXCLUDE entries: "seq:ZCAM08390" for one block)
        own = [str(e) for e in (s.exclude or [])]
        for e in own:                                     # 0.94.4: what each of the block's own entries leaves out
            k = len(exclude_products(paths, [e])[1])
            log.info("not processed (EXCLUDE %r): %d products%s", e, k, "" if k else "  <- matches no product (typo?)")
        paths, ex = exclude_products(paths, shipped_exclusions() + own)
        if ex:
            log.info("not processed (the shipped exclusion list%s): %d products, e.g. %s",
                     " and EXCLUDE" if own else "", len(ex), ex[0])
            sel["excluded_before_processing"] = len(ex)
    processed = st.work / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    cfg = processing_config(s)
    manifest = process_images(paths, processed, cfg, load_waypoints(refresh="auto"), provenance=sel,
                              only_existing="PNG8" if s.keep_only_remaining else None,
                              reuse_existing=not s.reprocess_all, workers=s.workers)
    log.info("%d images in the manifest, %d failed, %d skipped by the selection rules (sky, LMST window, "
             "saturation, exposure)", manifest["n_processed"], len(manifest["failed"]), len(manifest.get("skipped", [])))
    skipped = sorted({f"S{m['site']:03d}D{m['drive']:04d}" for m in manifest["images"]
                      if (m.get("mask") or {}).get("inference_skipped")})
    if skipped:
        log.info("images without mask inference (rover visible) at %s", skipped)
    if manifest.get("only_existing"):
        o = manifest["only_existing"]
        log.info("kept %d of %d selected; %d removed", o["n_kept"], o["n_selected"], o["n_removed"])
    from . import __version__
    done = {"mppp": __version__, "run_key": process_key(s), "site": st.name, "finished": _now(),
            "images": len(manifest.get("images", [])), "failed": len(manifest.get("failed", [])),
            "skipped": len(manifest.get("skipped", []))}
    (processed / PROCESS_DONE).write_text(json.dumps(done, indent=1), encoding="utf-8")
    log.info("processed: %d images -> %s", done["images"], processed)
    return processed


# --------------------------------------------------------------------------------------------------- 3. the manifest
def load_images(work: PathLike, settings: Settings, zooms: Sequence[int] = ()) -> List[Dict[str, Any]]:
    """The manifest images of ``<work>/processed`` that go into the alignment: the newest manifest, minus the
    images deleted from ``images_png8`` (``keep_only_remaining``), the exclusions (``exclude_images.txt`` and
    ``exclude``), the images with too little unmasked area, and the Mastcam-Z zooms not in ``zooms``."""
    from .sfm.workdir import (EXCLUDE_FILE, apply_exclusions, filter_included_fraction, keep_existing, load_manifest,
                              read_exclusions)
    work = Path(work)
    processed = work / "processed"
    if not processed.is_dir():
        raise FileNotFoundError(f"{work} has no processed/ folder: process the images first (mppp process)")
    manifest, mf = load_manifest(processed)
    images = list(manifest["images"])
    n0 = len(images)
    if settings.keep_only_remaining:
        images, gone = keep_existing(images, processed, "PNG8")
        if gone:
            log.info("%d images of the manifest are no longer in processed/images_png8 - left out", len(gone))
    log.info("%d of %d images from %s", len(images), n0, mf.name)
    entries = read_exclusions(work / EXCLUDE_FILE) + list(settings.exclude or [])
    if entries:
        images, removed = apply_exclusions(images, entries)
        by: Dict[str, List[str]] = {}
        for r in removed:
            by.setdefault(r["entry"], []).append(r["image"])
        for e, ims in by.items():
            ims = [x for x in ims if x]
            gone = settings.exclude_list and e in [str(x) for x in (settings.exclude or [])]
            log.info("excluded by %r: %d images%s", e, len(ims), "" if ims else (
                "  (its products were left out before the processing)" if gone else "  <- matches no image (typo?)"))
        log.info("%d images after the exclusions", len(images))
    if settings.exclude_list:
        images, removed = apply_exclusions(images, shipped_exclusions())
        by: Dict[str, int] = {}
        for r in removed:
            if r["image"]:
                by[r["entry"]] = by.get(r["entry"], 0) + 1
        for e, n in by.items():
            log.info("excluded by the shipped list (mppp/data/exclude_images.txt) %r: %d images", e, n)
    images, small = filter_included_fraction(images, settings.min_included_fraction)
    if small:
        log.info("%d images with less than %.3f of the image unmasked left out, e.g. %s", len(small),
                 settings.min_included_fraction or 0,
                 ", ".join(f"{r['image']} ({r['included_fraction']:.2f})" for r in small[:3]))
    n = len(images)
    images = [m for m in images if "failed" in m or (m.get("filename") or {}).get("family") != "Z"
              or m["filename"].get("zoom_mm") in list(zooms)]
    if len(images) < n:
        log.info("%d Mastcam-Z images of other zooms than %s left out", n - len(images), list(zooms))
    n = len(images)
    fams = hazcam_families(settings.hazcams)
    images = [m for m in images if "failed" in m or (m.get("filename") or {}).get("family") not in ("F", "R")
              or m["filename"]["family"] in fams]
    if len(images) < n:
        log.info("%d Hazcam images left out (hazcams = %s)", n - len(images), list(fams) or "none")
    return images


# ----------------------------------------------------------------------------------------------------- 4. project
def _zero_terms(s: Settings) -> Tuple[Tuple[str, ...], Optional[Tuple[str, ...]]]:
    """(Navcam zero terms, Mastcam-Z zero terms or None when they follow the Navcam ones)."""
    for k, v in (("tangential", s.tangential), ("zcam_tangential", s.zcam_tangential)):
        if v not in ("refine", "zero") and not (k == "zcam_tangential" and v is None):
            raise ValueError(f"{k} = {v!r}: use 'refine' or 'zero'")
    zt = s.zcam_tangential or s.tangential
    zero = ("p1", "p2", "b1", "b2") if s.tangential == "zero" else ("b1", "b2")
    zzero = None if zt == s.tangential else (("p1", "p2", "b1", "b2") if zt == "zero" else ("b1", "b2"))
    if s.zcam_k3 not in ("label", "zero"):
        raise ValueError(f"zcam_k3 = {s.zcam_k3!r}: use 'label' or 'zero'")
    if s.zcam_k3 == "zero":
        zzero = tuple(zzero if zzero is not None else zero) + ("k3",)
    return zero, zzero


def _refine_tangential(s: Settings) -> Any:
    """p1, p2 refined per camera family: a bool when both follow ``tangential``."""
    zt = s.zcam_tangential or s.tangential
    return (s.tangential == "refine") if zt == s.tangential else {"N": s.tangential == "refine", "Z": zt == "refine"}


def _rig_refine(s: Settings) -> Any:
    r = {"refine": "rotation", "rotation": "rotation", "auto": "auto", "hold": False, "full": True}.get(s.navcam_rig_refine)
    if r is None:
        raise ValueError(f"navcam_rig_refine = {s.navcam_rig_refine!r}: use 'refine', 'auto', 'hold' or 'full'")
    return r


def _check_settings(s: Settings) -> None:
    for k, v, ok in (("navcam_distortion_fit", s.navcam_distortion_fit, ("hold", "refine")),
                     ("navcam_rig_yaw", s.navcam_rig_yaw, ("hold", "refine")),
                     ("navcam_rig_roll", s.navcam_rig_roll, ("hold", "refine")),
                     ("navcam_pp", s.navcam_pp, ("hold", "refine")),
                     ("navcam_intrinsics", s.navcam_intrinsics, ("refine", "hold", "auto")),
                     ("hazcam_intrinsics", s.hazcam_intrinsics, ("radial", "focal", "all", "hold")),
                     ("hazcam_rig_refine", s.hazcam_rig_refine, ("hold", "rotation")),
                     ("source", s.source, SOURCES)):
        if v not in ok:
            raise ValueError(f"{k} = {v!r}: use one of {ok}")
    _rig_refine(s)
    _zero_terms(s)
    if s.navcam_cameras and not (Path(s.navcam_cameras) / "M2020_N_rig.json").is_file():
        raise FileNotFoundError(f"navcam_cameras = {s.navcam_cameras}: no Navcam cameras there (moved?); None uses "
                                f"the models in use (mppp.calib)")


def _low_backlash_found(s: Settings, work: Path) -> Optional[Path]:
    from .sfm.backlash import FOUND_FILE
    if not s.include_zcam:
        return None
    if s.zcam_low_backlash_found == "auto":
        return (Path(s.scapes_root) if s.scapes_root else work.parent) / FOUND_FILE
    return Path(s.zcam_low_backlash_found) if s.zcam_low_backlash_found else None


def _focus_below_sha(found: Optional[Path]) -> Optional[str]:
    """The fingerprint of the focus-approach list next to the found list (None without one)."""
    from .sfm.backlash import FOCUS_BELOW_FILE, low_backlash_fingerprint
    if not found:
        return None
    f = Path(found).parent / FOCUS_BELOW_FILE
    return low_backlash_fingerprint(f) if f.is_file() else None


def wanted_project_settings(s: Settings, work: Path) -> Dict[str, Any]:
    """What ``project.settings`` must record for the project to be reused rather than built again."""
    from .sfm.backlash import low_backlash_fingerprint
    zero, zzero = _zero_terms(s)
    z = bool(s.include_zcam)
    found = _low_backlash_found(s, work)
    return {"zero_terms": list(zero), "zcam_intrinsics": s.zcam_intrinsics,
            "zcam_zero_terms": list(zzero) if (zzero is not None and z) else None,
            "zcam_focus_bin": float(s.zcam_focus_bin) if s.zcam_focus_bin else None, "zcam_bin_refine": s.zcam_bin_refine,
            "zcam_bin_by": (s.zcam_bin_by or "focus") if z else None,
            "zcam_rig": s.zcam_rig, "zcam_rig_refine": s.zcam_rig_refine,
            "zcam_low_backlash_sha256": low_backlash_fingerprint(s.zcam_low_backlash, found) if z else None,
            "zcam_focus_below_sha256": _focus_below_sha(found) if z else None,      # 0.94.4
            "zcam_line_params": ({str(k): list(v) for k, v in s.zcam_line_params.items()}
                                 if (s.zcam_line_params and z) else None),
            "zcam_rig_fit": s.zcam_rig_fit if z else None,
            "zcam_line_form": s.zcam_line_form if z else None,
            "navcam_rig_epochs_gap": float(s.navcam_rig_epoch_gap_sols),
            "navcam_rig_epochs_breaks": [float(x) for x in s.navcam_rig_epoch_breaks_sols],
            "navcam_distortion": "fisheye_tangential", "navcam_rig": s.navcam_rig,
            "zcam_hold_f_images": int(s.zcam_hold_f_images) if s.zcam_intrinsics == "focus_model" else None,
            "zcam_min_zoom_images": int(s.zcam_min_zoom_images) if z else None,       # 0.95.0
            "prior_R_corrected": True,
            "navcam_distortion_fit": s.navcam_distortion_fit, "navcam_k4": "consensus",
            "navcam_rig_yaw": s.navcam_rig_yaw, "navcam_rig_roll": s.navcam_rig_roll, "navcam_pp": s.navcam_pp,
            "prior_sigma_m": [float(s.prior_sigma_m)] * 3}


def _hazcam_settings_changed(proj, s: Settings) -> bool:
    """A project with Hazcam images built with other Hazcam settings (0.93)."""
    h = proj.settings.get("hazcam") if proj is not None else None
    if not h:
        return False
    return h.get("intrinsics") != s.hazcam_intrinsics or h.get("rig_refine") != s.hazcam_rig_refine


def check_camera_models(s: Settings) -> Dict[str, Any]:
    """The camera models this run starts from, logged; raises when they do not follow the frozen forms."""
    from .calib import check_form, describe
    for line in describe(s.navcam_cameras):
        log.info("%s", line)
    form = check_form(s.navcam_cameras)
    for w in form["warnings"]:
        log.warning("camera models: %s", w)
    if not form["ok"]:
        raise RuntimeError("the camera models do not follow the frozen forms: "
                           + "; ".join(form["errors"]))
    return form


def build_project(work: PathLike, images: List[Dict[str, Any]], settings: Settings):
    """
    The COLMAP project of a run (``<work>/colmap`` or ``colmap_<variant>``): created from the manifest images with
    the start cameras, rig and priors, or reused when it holds the same images with the same camera settings and
    models (``reprocess_all`` rebuilds it).  A variant starts from the default project's features and matches.
    Returns ``(project, info)`` with ``info["reuse_matches_from"]``.
    """
    from .calib import navcam_fingerprint
    from .paths import cmods_dir
    from .sfm import SfmProject
    from .sfm.workdir import project_dir, seed_variant
    s = settings
    work = Path(work)
    root = project_dir(work, s.variant)
    info: Dict[str, Any] = {"reuse_matches_from": [], "created": False}
    if s.variant:
        seed = seed_variant(root, project_dir(work, None))
        info["reuse_matches_from"] = seed["reuse_matches_from"] if s.reuse_matches else []
        if seed["copied"]:
            log.info("variant %s: features copied from %s", s.variant, seed["base"])
    info["form"] = check_camera_models(s)
    found = _low_backlash_found(s, work)
    zero, zzero = _zero_terms(s)
    z = bool(s.include_zcam)
    proj = SfmProject.load(root) if (root / "project.json").is_file() else None
    wanted = wanted_project_settings(s, work)
    names = {Path(m["source_product"]).stem for m in images}
    nav_dir = Path(s.navcam_cameras) if s.navcam_cameras else cmods_dir()
    have = (proj.settings.get("navcam_cameras") or {}) if proj is not None else {}
    rebuild = (s.reprocess_all or proj is None or names != {r["stem"] for r in proj.images}
               or any(proj.settings.get(k) != v for k, v in wanted.items())
               # 0.95.0: the models' fingerprint decides, not the folder (MPPP_CMODS moved: nothing to rebuild)
               or have.get("fingerprint") != navcam_fingerprint(nav_dir)      # rewritten models rebuild the project
               or _hazcam_settings_changed(proj, s))
    if rebuild:
        proj = SfmProject.create(images, work / "processed", root, prior_sigma_m=(s.prior_sigma_m,) * 3,
                                 zcam_intrinsics=s.zcam_intrinsics, zero_terms=zero,
                                 zcam_focus_bin=s.zcam_focus_bin, zcam_bin_refine=s.zcam_bin_refine, zcam_rig=s.zcam_rig,
                                 zcam_bin_by=s.zcam_bin_by or "focus",
                                 navcam_rig=s.navcam_rig, zcam_hold_f_images=s.zcam_hold_f_images,
                                 zcam_min_zoom_images=s.zcam_min_zoom_images,
                                 navcam_cameras=s.navcam_cameras,
                                 zcam_zero_terms=zzero if z else None, zcam_rig_refine=s.zcam_rig_refine,
                                 zcam_low_backlash=s.zcam_low_backlash, zcam_low_backlash_found=found,
                                 zcam_line_params=s.zcam_line_params if z else None,
                                 zcam_rig_fit=s.zcam_rig_fit if z else None,
                                 zcam_focus_model_file=s.zcam_focus_model,
                                 navcam_distortion_fit=s.navcam_distortion_fit, navcam_rig_yaw=s.navcam_rig_yaw,
                                 navcam_rig_roll=s.navcam_rig_roll,
                                 navcam_rig_epoch_gap=s.navcam_rig_epoch_gap_sols,
                                 navcam_rig_epoch_breaks=s.navcam_rig_epoch_breaks_sols,
                                 zcam_line_form=s.zcam_line_form if z else None, navcam_pp=s.navcam_pp,
                                 hazcam_intrinsics=s.hazcam_intrinsics, hazcam_rig_refine=s.hazcam_rig_refine)
        info["created"] = True
    else:
        from .sfm.thermal import strip_thermal_bins
        strip_thermal_bins(proj)                     # back to one camera per eye / focus bin before a rerun
        proj.refresh_images(images)                  # copies (no hard links) follow images processed again
        if z and s.zcam_intrinsics == "focus_model":
            from .calib import zcam_fingerprint
            fm = (proj.settings.get("zcam_focus_model") or {}).get("fingerprint")
            if fm != zcam_fingerprint(None, s.zcam_focus_model):
                log.warning("this project's Mastcam-Z focus bins were built with %s; set reprocess_all to rebuild "
                            "them from the current one (a new alignment)",
                            "an earlier focus model" if fm is None else "another focus model")
    if s.colmap_bat:
        proj.settings["colmap_bat"] = str(s.colmap_bat)
    proj.settings["gui_files"] = bool(s.gui_files)
    proj.save()
    _log_project(proj)
    return proj, info


def _log_project(proj) -> None:
    import numpy as np
    log.info("%d images; baseline %s m; CAHV pairs %s; rig rotation %s", len(proj.images),
             {k: round(v["baseline_m"], 4) for k, v in proj.rig.items()},
             {k: v["n_pairs"] for k, v in proj.rig.items()},
             {k: v.get("rotation_source", "CAHV") for k, v in proj.rig.items()})
    for k, c in proj.cameras.items():
        if not c.get("group"):
            log.info("%s %s p1 %.2e p2 %.2e %s%s k3 %.3g k4 %.3g", k, [round(x, 3) for x in c["params"][:5]],
                     c["params"][6], c["params"][7], "distortion held" if c.get("distortion_fit") == "hold" else "",
                     f" ({', '.join(c['zeroed_terms'])} = 0)" if c.get("zeroed_terms") else "",
                     c["params"][8], c["params"][9])
    bins = Counter(c["group"] for c in proj.cameras.values() if c.get("group") and "focus_count_range" in c)
    for g, n in sorted(bins.items()):
        cs = [c for c in proj.cameras.values() if c.get("group") == g and "focus_count_range" in c]
        held = sum("fx" in (c.get("fixed_params") or []) for c in cs)
        log.info("%s: %d focus bins (%d with f held), counts %.0f-%.0f, images per bin %s", g, n, held,
                 min(c["focus_count_range"][0] for c in cs), max(c["focus_count_range"][1] for c in cs),
                 sorted(c["n_images"] for c in cs))
    corr = [r["prior_R_correction_deg"] for r in proj.images if "prior_R_correction_deg" in r]
    if corr:
        log.info("Mastcam-Z prior attitude correction: median %.3f deg, max %.3f deg (%d images)",
                 float(np.median(corr)), max(corr), len(corr))
    log.info("images per camera: %s", dict(Counter(r["instrument"] for r in proj.images)))


# --------------------------------------------------------------------------------------- 5. features, database, match
def extract_and_match(proj, settings: Settings, reuse_from: Sequence[PathLike] = (), stop_after: Optional[str] = None
                      ) -> Dict[str, Any]:
    """SIFT features (reused while the images are the same), the database (matches carried over while the
    features and matching settings are the same), then the matching."""
    from .sfm import build_database, extract_features, match, prior_overlap_pairs
    from .sfm.matching import match_settings
    s = settings
    gpu = True if s.gpu_python else None
    # in a child process like the matching (Issole Z34, 20 Oct: 2713 images extracted on the CPU inside the run's
    # process, which died with no message)
    extract_features(proj, max_num_features=s.max_num_features, use_gpu=gpu, python=s.gpu_python or sys.executable,
                     overwrite=bool(s.reprocess_all), **s.sift)
    out: Dict[str, Any] = {}
    if stop_after == "features":
        return out
    mode = effective_match_mode(s, len(proj.images))
    # 0.95.0: the Mastcam-Z pairs verified uncalibrated from the start (the calibrated pass that aborted was lost)
    unc = {"uncalibrated_families": ("Z",)} if s.zcam_uncalibrated_matching and s.include_zcam else {}
    db = build_database(proj, reuse_matches_from=list(reuse_from),
                        matching=match_settings(mode, **s.match, **unc,
                                                **({"keypoint_selection": str(s.lightglue_keypoints)}
                                                   if s.lightglue_keypoints not in (None, "largest")
                                                   else {})) if s.reuse_matches else None)
    if db.get("reused_matches"):
        log.info("matches reused: %s %s", {k: db["reused_matches"][k] for k in ("matches", "geometries")},
                 [x.get("skipped") or "used" for x in db["reused_matches"]["sources"]])
    log.info("database: %s", {k: v for k, v in db.items() if k != "reused_matches"})
    out["database"] = db
    if stop_after == "database":
        return out
    pairs = None
    big = bool(s.max_exhaustive_images) and len(proj.images) > int(s.max_exhaustive_images)
    if mode == "prior_pairs" or (mode in ("lightglue", "sift+lightglue") and big):
        # 0.93.8: LightGlue on a big block gets the prior-overlap pairs too (it matched every pair: 564,453 at
        # Marble Mountain's 1063 images, over its time budget, and then fell back to COLMAP over every pair)
        pairs = prior_overlap_pairs(proj, min_overlap=0.05)
        n = len(proj.images)
        log.info("%d prior pairs%s", len(pairs), f" (instead of {n * (n - 1) // 2} exhaustive: {n} images > "
                 f"max_exhaustive_images {s.max_exhaustive_images})" if big else "")
    kw = dict(use_gpu=gpu, block_size=s.match_block_size, **s.match, **unc)
    if s.lightglue_keypoints not in (None, "largest"):
        kw["keypoint_selection"] = str(s.lightglue_keypoints)
    # COLMAP's matcher runs in a child process: the CUDA environment, else this python on the CPU - a COLMAP
    # abort (glog "Check failed") then fails the step with its message instead of killing the run unseen, and
    # the calibrated-verification fallback of mppp.sfm.matching can take over.  0.94.4: for "lightglue" too - its
    # SIFT fallback (over the time budget) ran COLMAP's CPU matcher inside the run, where OpenBLAS overflowed its
    # thread table on the 23 Oct PC and the process died (Chal Rocks Large Mastcam-Z, 152,059 pairs)
    kw["python"] = s.gpu_python or sys.executable
    if pairs is not None:
        kw["pairs"] = pairs
    out["match"] = match(proj, mode=mode, **kw)
    log.info("matching: %s", out["match"])
    return out


# ------------------------------------------------------------------------------------------------- 6. reconstruct
def reconstruct_project(proj, settings: Settings) -> Tuple[Any, Dict[str, Any]]:
    """Triangulation from the rover poses and the weighted bundle adjustment; the export for the error analysis
    (``error_input/``, ``summary.json``) and the convergence statistics.  Returns ``(rec, summary)``."""
    from .sfm import export_for_error, reconstruct
    from .sfm.reconstruction import camera_changes, convergence_statistics, navcam_network
    from .sfm.thermal import image_temperatures, thermal_model_for_project
    s = settings
    net = navcam_network(proj)
    log.info("Navcam network: %s stations over %.1f m, %s images -> %s (intrinsics setting %r)", net["stations"],
             net["span_m"], net["navcam_images"], net["verdict"], s.navcam_intrinsics)
    temps = image_temperatures(proj, pds_dir=s.pds_dir if s.thermal_bins_deg else None, verbose=bool(s.thermal_bins_deg))
    proj.settings["no_position_prior_at"] = list(s.no_position_prior_at or [])
    proj.settings["prior_gauge_recentre"] = bool(s.prior_gauge_recentre)
    if s.no_position_prior_at:
        log.info("waypoint position priors left out (flagged): %s", ", ".join(s.no_position_prior_at))
    rig_refine = _rig_refine(s)
    z = bool(s.include_zcam)
    light = str(s.export_level or "full") == "light"      # 0.95.0: a calibration run's export
    if light:
        log.info("export level light: no native text model, residuals, Navcam-stage export or COLMAP GUI copy")
    rec = reconstruct(proj, sigma_px=0.5, schedule=s.schedule, refine_rig=rig_refine,
                      refine_tangential=_refine_tangential(s), min_track_length=2,
                      min_tri_angle_deg=s.min_tri_angle_deg, attitude_prior_deg=s.attitude_prior_deg,
                      triangulation_options=s.triangulation, exclude_outliers=s.exclude_outliers,
                      outlier_thresholds=s.outlier_thresholds, min_frame_observations=s.min_frame_observations,
                      navcam_intrinsics=s.navcam_intrinsics, staged=s.staged, hold_cameras=tuple(s.hold_cameras),
                      linear_solver=s.linear_solver, temperatures=temps, thermal_bins_deg=s.thermal_bins_deg,
                      thermal_min_images=s.thermal_min_images, thermal_free=tuple(s.thermal_free),
                      thermal_model=thermal_model_for_project(proj), thermal_fine_range=s.thermal_fine_range,
                      thermal_fine_deg=s.thermal_fine_deg, localize_min_images=s.localize_min_images,
                      zcam_backlash=s.zcam_backlash if z else None, zcam_focus_line=bool(s.zcam_focus_line),
                      zcam_line_cycles=int(s.zcam_line_cycles), pose_guided=s.pose_guided,
                      zcam_line_settle_px=float(s.zcam_line_settle_px or 0), ba_points_cap=int(s.ba_points_cap or 0),
                      isolated_points_factor=float(s.isolated_points_factor or 0),
                      navcam_aspect=s.navcam_aspect, thermal_after_stage1=bool(s.thermal_after_stage1),
                      round1_hold=bool(s.round1_hold), zcam_station_prior_sigma_m=float(s.zcam_station_prior_sigma_m or 0),
                      rig_translation_sigma_m=s.rig_translation_sigma_m if rig_refine is True else None,
                      gui_native=not light)
    rs = proj.settings["reconstruction"]
    log.info("Navcam intrinsics: %s | held cameras: %s | staged: %s", rs.get("navcam_intrinsics"),
             rs.get("hold_cameras") or "none", rs.get("staged"))
    excluded = rs.get("excluded", [])
    log.info("%d frames excluded as outliers", len(excluded))
    for st, v in (rs.get("unlocalized") or {}).items():
        log.info("   no position prior: %s (%s images, %s cross-station points): %s", st, v["images"],
                 v["cross_points"], v["prior"])
    for e in excluded:
        log.info("   %s | %s | %s", ", ".join(e["images"]), e["station"], "; ".join(e["reasons"]))
    import numpy as np
    by_seq: Dict[str, List[Dict[str, Any]]] = {}
    for e in excluded:                                    # 0.95.0: the frames without tie points by sequence
        if e.get("kind") == "unconstrained" and e.get("sequence"):
            by_seq.setdefault(str(e["sequence"]), []).append(e)
    if by_seq:
        log.info("frames without tie points by sequence: %s", "; ".join(
            f"{sq} {len(v)}" + (f" (boresight {np.median([x['boresight_elevation_deg'] for x in v if x.get('boresight_elevation_deg') is not None]):+.0f} deg)"
                                if any(x.get("boresight_elevation_deg") is not None for x in v) else "")
            for sq, v in sorted(by_seq.items(), key=lambda kv: -len(kv[1]))))
    summary = export_for_error(proj, rec, level="light" if light else "full")
    log.info("export: %s", json.dumps({k: summary[k] for k in ("tracks", "residual_rms_native_px", "registered_images",
                                                                "project_images", "prior_offset_median_m",
                                                                "station_components")}, default=str))
    if rs.get("navcam_stage") and not light:
        import pycolmap
        rec_nav = pycolmap.Reconstruction(str(proj.root / rs["navcam_stage"]["path"]))
        nav = export_for_error(proj, rec_nav, out_dir=proj.root / "error_input_navcam")
        log.info("Navcam-only stage exported to error_input_navcam/: %s",
                 {k: nav[k] for k in ("tracks", "residual_rms_native_px", "registered_images")})
    conv = convergence_statistics(rec, proj)
    proj.settings["reconstruction"]["convergence"] = conv
    proj.save()
    log.info("tie points by convergence angle %s deg: %s | cross-station: %s | > 10 deg: %s, > 20 deg: %s",
             conv["bins_deg"], conv["points"], conv["cross_station_points"], conv["points_over_10deg"],
             conv["points_over_20deg"])
    for r in camera_changes(rec, proj):
        if r["missing"]:
            log.info("%s: not in the reconstruction", r["camera"])
            continue
        log.info("%s (%d images)%s: %s", r["camera"], r["images"],
                 "  [temperature bin]" if r["thermal_bin"] else "", " ".join(
                     f"{n} {a:.4g}->{b:.4g}" for n, a, b in zip(r["names"][:10], r["start"], r["refined"])))
    hz = (proj.settings.get("reconstruction") or {}).get("hazcam")
    if hz:
        from .sfm.reconstruction import hazcam_line
        log.info("Hazcam: %s", hazcam_line(hz))
    bl = proj.settings.get("zcam_backlash")
    if bl:
        log.info("Mastcam-Z focus groups: %s | f / label f: %s", bl["group_counts"],
                 {k: (round(v["ratio_median"], 4) if v["ratio_median"] else None) for k, v in (bl.get("after") or bl["before"]).items()})
    return rec, summary


# ------------------------------------------------------------------------------------------------- 7-8. health, export
def health(proj, rec) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """The alignment health report, logged and written to ``<project>/health/``; the Mastcam-Z focus breathing
    when there are focus-bin cameras.  Returns ``(report, files)``."""
    from .sfm import assess_alignment, health_table, write_health
    report = assess_alignment(proj, rec)
    for line in health_table(report).splitlines():
        log.info("%s", line)
    files = write_health(report, proj.root / "health", rec, proj)
    if report["verdict"] == "fail":
        log.warning("FAIL: fix the alignment (matching, priors, masks, thresholds) before any error analysis")
    if any(c.get("group") for c in proj.cameras.values()):
        from .sfm import write_focus_breathing
        fb = write_focus_breathing(proj, rec)
        files["focus_breathing"] = fb["png"]
        for g, f in fb["fits"].items():
            for kind in ("refined", "label"):
                if f[kind]:
                    log.info("%s %7s: f = %.1f px at count %.0f, slope %+.4f px/count, rms %.2f px, n = %d", g, kind,
                             f[kind]["f0_px"], f["reference_count"], f[kind]["slope_px_per_count"], f[kind]["rms_px"],
                             f[kind]["n"])
    return report, files


def finish(proj, report: Dict[str, Any], summary: Dict[str, Any], settings: Settings, site: str, key: str,
           form: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The camera-shift figure beside ``error_input/`` and ``run_done.json`` (the run key: a rerun with the same
    settings and camera models is skipped by the batch scripts)."""
    from . import __version__
    from .calib import models_fingerprint
    from .sfm import plot_camera_shifts
    import matplotlib.pyplot as plt
    with (proj.root / "error_input" / "poses.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig = plot_camera_shifts(proj, rows=rows, out_png=proj.root / "error_input" / "camera_shifts.png")
    plt.close(fig)
    eps = summary.get("eps", {})
    log.info("error_input: %s | eps (rms residual / sqrt 2): same-station %.3f px, cross-station %.3f px",
             proj.root / "error_input", eps.get("eps_intra_px", float("nan")), eps.get("eps_cross_px", float("nan")))
    done = {"mppp": __version__, "run_key": key, "site": site, "variant": settings.variant or "",
            "source": settings.source, "finished": _now(), "verdict": report.get("verdict"),
            "images": len(proj.images), "registered_images": summary.get("registered_images"),
            "residual_rms_native_px": summary.get("residual_rms_native_px"),
            "camera_models": {"fingerprint": models_fingerprint(), "form_ok": (form or {}).get("ok", True),
                              "dir": str(settings.navcam_cameras or (proj.settings.get("navcam_cameras") or {}).get("dir"))},
            "zcam_new_low_backlash": len((proj.settings.get("zcam_backlash") or {}).get("new_regular_groups") or [])}
    (proj.root / RUN_DONE).write_text(json.dumps(done, indent=1, default=str), encoding="utf-8")
    log.info("done: %s", proj.root / RUN_DONE)
    if not getattr(settings, "keep_databases", False):      # 0.95.0 (Christian, 27 Oct): 5-20 GB per site on the RAID
        freed = 0
        for f in list(proj.root.glob("database_thermal.db")) + list(proj.root.glob("database_guided*.db")) \
                + list(proj.root.glob("database_lightglue_add*.db")):
            try:
                freed += f.stat().st_size
                f.unlink()
            except OSError as e:                                # noqa: BLE001 - held by a viewer: next time
                log.warning("not removed: %s (%s)", f.name, e)
        if freed:
            log.info("the run's database copies removed (%.1f GB; keep_databases keeps them)", freed / 1e9)
    return done


# ------------------------------------------------------------------------------------------------------- 3-8. align
def align(work: PathLike, settings: Settings, stop_after: Optional[str] = None) -> Dict[str, Any]:
    """
    Align the images processed in ``<work>/processed`` (sections 3-8): project, features, database, matching,
    reconstruction, health report, export.  ``stop_after`` (one of :data:`STEPS`) stops after that step and returns
    what is there.  Returns the health report (``report["summary"]`` holds the export summary, ``report["run_key"]``
    the key written to ``run_done.json``).
    """
    from . import __version__
    from .sfm.workdir import project_dir
    if stop_after is not None and stop_after not in STEPS:
        raise ValueError(f"stop_after = {stop_after!r}: use one of {STEPS}")
    work = Path(work).resolve()
    if not (work / "processed").is_dir():
        raise FileNotFoundError(f"{work} has no processed/ folder: process the images first "
                                f"(mppp process {work.name}, or run with source 'pds')")
    st, s = resolve_site(work, settings)
    _check_settings(s)
    key = run_key(s)
    root = project_dir(work, s.variant)
    log.info("== %s (sols %s) | %s | variant %s | run key %s", st.name, st.sols_text, root, s.variant or "(default)", key)
    _warn_other_block(root, st)
    if s.gpu_python:
        from .sfm import check_gpu_python
        log.info("GPU env: %s", check_gpu_python(s.gpu_python))
    else:
        log.warning("no GPU environment (MPPP_GPU_PYTHON, MPPP_GPU_PY, %%LOCALAPPDATA%%\\mppp\\gpu_python.txt or "
                    "<conda>\\envs\\mppp_gpu): "
                    "SIFT extraction and matching on the CPU - hours on a large block")
    images = load_images(work, s, st.zooms)
    if stop_after == "manifest":
        return {"images": len(images)}
    # 0.95.0 (Christian, 27 Oct): a block being aligned again is not finished - run_done.json goes now, so a joint
    # that collects the group's blocks meanwhile waits for it (mppp calib) instead of reading a half-written project
    stale = project_dir(work, s.variant) / RUN_DONE
    if stale.is_file():
        try:
            stale.unlink()
            log.info("run_done.json removed: the block is being aligned again")
        except OSError as e:                                  # noqa: BLE001
            log.warning("run_done.json not removed (%s)", e)
    proj, info = build_project(work, images, s)
    if s.zcam_line_zooms is not None:                       # 0.95.0: the zooms whose bins go on focus lines
        proj.settings["zcam_line_zooms"] = sorted(int(z) for z in s.zcam_line_zooms)
        proj.save()
    (proj.root / RUN_SETTINGS).write_text(json.dumps({"mppp": __version__, "run_key": key, "settings": s.to_dict()},
                                                     indent=1), encoding="utf-8")
    if stop_after == "project":
        return {"project": str(proj.root), "images": len(proj.images), "run_key": key}
    out = extract_and_match(proj, s, info["reuse_matches_from"], stop_after)
    if stop_after in ("features", "database", "match"):
        return dict(out, project=str(proj.root), run_key=key)
    rec, summary = reconstruct_project(proj, s)
    if stop_after == "reconstruct":
        return {"project": str(proj.root), "summary": summary, "run_key": key}
    report, files = health(proj, rec)
    report["summary"] = summary
    report["files"] = files
    report["run_key"] = key
    report["project"] = str(proj.root)
    if stop_after == "health":
        return report
    report["run_done"] = finish(proj, report, summary, s, st.name, key, info.get("form"))
    return report


def _warn_other_block(root: Path, st: Site) -> None:
    """A project folder left by another sol range (e.g. a site renamed in the site definitions) would mix two
    blocks."""
    if not (root / "project.json").is_file():
        return
    try:
        sols = [int(r["sol"]) for r in json.loads((root / "project.json").read_text(encoding="utf-8")).get("images", [])
                if "sol" in r]
    except (OSError, ValueError):
        return
    if sols and not any(a <= x <= b for x in sols for a, b in st.sol_ranges):
        log.warning("%s holds a block of sols %d-%d, none in %s: rename or move that folder before running %r "
                    "with this sol range", root, min(sols), max(sols), st.sols_text, st.name)


def run_site(site_or_work: PathLike, settings: Settings, stop_after: Optional[str] = None) -> Dict[str, Any]:
    """Select, process and align a site (or a WORK folder): :func:`process_site` when ``source`` is ``"pds"``, then
    :func:`align`.  Returns the health report."""
    st, s = resolve_site(site_or_work, settings)
    if s.source == "pds":
        process_site(st.work, s)
    elif not (st.work / "processed").is_dir():
        raise FileNotFoundError(f"{st.work} has no processed/ folder (source 'processed'): process the images first")
    return align(st.work, s, stop_after=stop_after)


def load_results(work: PathLike, variant: str = ""):
    """``(project, rec, report, summary)`` of a finished alignment, read from disk (for the notebooks)."""
    from .sfm import SfmProject
    from .sfm.reconstruction import load_alignment
    from .sfm.workdir import project_dir
    root = project_dir(work, variant)
    proj = SfmProject.load(root)
    rec = load_alignment(proj)
    report = json.loads((root / "health" / "health.json").read_text(encoding="utf-8")) \
        if (root / "health" / "health.json").is_file() else None
    summary = json.loads((root / "error_input" / "summary.json").read_text(encoding="utf-8")) \
        if (root / "error_input" / "summary.json").is_file() else None
    return proj, rec, report, summary


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


__all__ = ["Settings", "Site", "hazcam_families", "ALIGN_RULES", "ZCAM_RULES", "PROCESS_RULES", "STEPS", "load_settings", "work_settings", "normalize",
           "setting_names", "run_key", "process_key", "processed_done", "run_done", "resolve_site", "select_images",
           "select_products", "process_site", "processing_config", "load_images", "build_project", "extract_and_match",
           "reconstruct_project", "health", "finish", "align", "run_site", "load_results"]
