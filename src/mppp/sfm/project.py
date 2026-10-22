"""
The SfM project: which images, which cameras, the stereo rig and the priors.

A project directory holds::

    images/<stem>.png         8-bit RGB, as processed by MPPP (hard link or copy)
    masks/<stem>.png.png      COLMAP masks (0 = ignore), from the MPPP masks
    project.json              image table, cameras, rig, world offset, settings
    features.db               SIFT features at native resolution (any camera)
    database.db               the final database (cameras, rig, frames, priors,
                              full-resolution keypoints, matches)
    sparse/                   reconstructions;  error_input/  exports

Every start camera and rig comes from the promoted models through :mod:`mppp.calib`
(:mod:`mppp.sfm.start_cameras`): the Navcam eyes at the block's median temperature and sol, the stereo rig at
its median sol, the Mastcam-Z eyes from the zoom's focus model (or the labels for a zoom without one), split into
one camera per focus bin and backlash state.
"""
from __future__ import annotations

import logging

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import calib
from ..calib import forms
from ..filenames import parse_filename
from ..paths import cmods_dir

log = logging.getLogger("mppp.sfm")

PathLike = Union[str, Path]

FULL_FRAME = forms.FULL_FRAME                      # full detector frame (width, height) at scale 1, by family
SCOPE = ("Navcam (NLF/NRF), the front and rear Hazcams (FLF/FRF, RLF/RRF) and Mastcam-Z at 34, 48, 63, 79, 110 mm "
         "(ZL0/ZR0 _034, _048, _063, _079, _110)")
SCOPE_ZCAM_ZOOMS = forms.ZCAM_ZOOMS
HAZCAM_CAMERA_CODES = ("FLF", "FRF", "RLF", "RRF")          # 0.93
SCOPE_CAMERA_CODES = ("NLF", "NRF", "ZL0", "ZR0") + HAZCAM_CAMERA_CODES
HAZCAM_FAMILIES = ("F", "R")
PARAM_NAMES = forms.PARAM_NAMES                    # parameter names per COLMAP model
INTRINSIC_CORE = forms.INTRINSIC_CORE              # focal length and principal point; everything else is distortion
FULL_OPENCV_NAMES = forms.FULL_OPENCV_PARAMS
# the Navcam distortion (k1-k4, p1, p2) is one set per eye for every sol and temperature: "hold" keeps it at the
# promoted camera in every block and temperature bin; "refine" fits it per block (the joint fit does)
NAVCAM_DISTORTION_FIT = "hold"
# the Navcam principal point is held in every block at the start camera (the promoted model moved to the block's
# temperature) - a block cannot separate it from the frame attitudes (a 1 px cx shift is 0.087 px rms of yaw on the
# equidistant fisheye); "refine": fitted per block.  The joint fit still fits NL cx, cy, NR cx, cy and their slopes.
NAVCAM_PP_SETTINGS = ("hold", "refine")
NAVCAM_PP = "hold"
# the Navcam rig yaw (rotation of the right camera about the left camera's y axis, the one that shifts the
# disparity): "hold" (the rig file's yaw at the block's sol, held in the adjustment and the thermal stage, so the
# principal points absorb the differences) or "refine" (with NAVCAM_RIG_REFINE); pitch and roll follow NAVCAM_RIG_REFINE
NAVCAM_RIG_YAW_SETTINGS = ("hold", "refine")
ZEROED_TERMS = ("b1", "b2")          # p1, p2 are kept from the model
ZCAM_FOCUS_BIN = 30.0                               # focus motor counts per Mastcam-Z camera bin
ZCAM_FOCUS_MODEL = calib.models.ZCAM_FILE_PATTERN.format(zoom=34)
ZCAM_FOCUS_MODEL_PATTERN = calib.models.ZCAM_FILE_PATTERN
ZCAM_BIN_REFINE_SETTINGS = ("focal", "all", "model", "focal_pp")  # "model": f held at the focus model;
# "focal_pp": a bin refines fx, fy - and on the right eye (ZR) also cx, cy, the R-L boresight that the narrow field
# makes observable (ZL keeps its principal point: the Mastcam-Z reference eye, like NL); the regular-state bins too
ZCAM_HOLD_F_IMAGES = 2                              # focus bins with <= this many images hold f at the model
NAVCAM_RIG = "consensus"                            # Navcam rig rotation starts from the promoted rig ("cahv": the labels)
NAVCAM_RIG_FILE = calib.models.RIG_FILE["N"]
NAVCAM_FISHEYE_PATTERN = "M2020_{instrument}_fisheye_tangential.json"
# the folder of the models in use (mppp/data/cmods; MPPP_CMODS overrides)
NAVCAM_PACKAGE_CONSENSUS_DIR = calib.models._folder(None)


def navcam_consensus_dir() -> Path:
    """Where the start cameras and rig come from by default: the models in use (:func:`mppp.paths.cmods_dir`)."""
    return cmods_dir()


NAVCAM_CONSENSUS_DIR = navcam_consensus_dir()
# parameters a focus-bin camera holds with zcam_bin_refine="focal": all but the focal length
ZCAM_BIN_HELD = ("cx", "cy", "k1", "k2", "p1", "p2", "k3")
NAVCAM_RIG_EPOCH_GAP_SOLS = calib.constrain.RIG_EPOCH_GAP_SOLS    # Navcam visits of a block more than this many sols
                                                                 # apart get their own rig rotation
NAVCAM_RIG_EPOCH_BREAKS_SOLS = calib.constrain.RIG_EPOCH_BREAK_SOLS
NAVCAM_RIG_EPOCH_MAX_SPAN_SOLS = calib.constrain.RIG_EPOCH_MAX_SPAN_SOLS   # an epoch spanning more sols is split
NAVCAM_RIG_EPOCH_MIN_IMAGES = 16     # 0.91: an epoch with fewer Navcam images (8 stereo frames) ...
NAVCAM_RIG_EPOCH_MIN_STATIONS = 2    # ... or a single station gets no rig of its own: joined to the nearest epoch


def focus_bins(counts: Sequence[Optional[float]], width: float = ZCAM_FOCUS_BIN) -> List[int]:
    """
    Bin index per image from its focus motor count: sorted counts are grouped
    greedily, each bin spanning at most ``width`` counts from its first
    (lowest) member, so a cluster of nearly equal counts is never split by a
    fixed grid line.  Images without a count share one extra bin (the last).
    Returns one bin index per input, bins numbered by increasing focus.
    """
    idx = [i for i, c in enumerate(counts) if c is not None and np.isfinite(float(c))]
    order = sorted(idx, key=lambda i: float(counts[i]))
    out = [-1] * len(counts)
    b, start = -1, None
    for i in order:
        c = float(counts[i])
        if start is None or c - start > width:
            b, start = b + 1, c
        out[i] = b
    missing = [i for i in range(len(counts)) if out[i] < 0]
    for i in missing:
        out[i] = b + 1
    return out


def camera_key(fn: Dict[str, Any]) -> str:
    """
    The COLMAP camera an image belongs to: the instrument for the engineering
    cameras (``NL``, ``NR``: every resolution shares it), the camera group for
    Mastcam-Z (``ZL034``: one camera per eye and zoom).
    """
    return fn["camera_group"] if fn["family"] == "Z" else fn["instrument"]


# ------------------------------------------------------------------ selection
NAVCAM_MIN_FRAME_FRACTION = 1.0 / 3.0   # select_best_products drops sub-frames / tiles below this share of the full
                                        # frame (the 1/3 rule); notebook 03 NAVCAM_MIN_FRAME_FRACTION


def frame_fraction(path: PathLike, size: Optional[int] = None) -> Optional[float]:
    """The fraction of the detector frame a Navcam product covers (image lines x samples against the full
    frame at its downsampling); None if unknown.  A full-resolution tile such as
    ``NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01`` (1280 x 960 of 5120 x 3840) gives 1/16.  The label is
    read only when the file is too small to be a full 3-band 16-bit frame (so ordinary frames cost a stat)."""
    from ..image import FULL_FRAME
    p = Path(path)
    fn = parse_filename(p)
    if fn.family not in FULL_FRAME:
        return None
    W, H = FULL_FRAME[fn.family]
    s = float(fn.downsample_scale)
    full = W * H * s * s
    size = p.stat().st_size if size is None else int(size)
    if size >= 0.5 * full * 3 * 2:                  # at least half a 3-band 16-bit frame: not a small sub-frame
        return None
    try:
        from ..labels import label_get, read_pds, first
        lab, _ = read_pds(p, load_image=False)
        lines = float(first(label_get(lab, "IMAGE.LINES")))
        samples = float(first(label_get(lab, "IMAGE.LINE_SAMPLES")))
    except Exception:                                   # noqa: BLE001 - unreadable label: keep the product
        return None
    return lines * samples / full


def select_best_products(paths: Iterable[PathLike], sizes: Optional[Dict[str, int]] = None,
                         sequence_prefix: Union[None, str, Sequence[str]] = "NCAM",
                         min_frame_fraction: Optional[float] = NAVCAM_MIN_FRAME_FRACTION,
                         frame_fraction_families: Sequence[str] = ("N",)) -> Tuple[List[Path], Dict[str, Any]]:
    """
    One product per exposure (instrument + SCLK): the largest file (highest
    resolution / largest sub-frame), then the highest version.  ``sizes``
    maps file name -> bytes (default: stat the files).  ``sequence_prefix``
    keeps only e.g. NCAM sequences (drops SAPP sun images, SCAM support images);
    a tuple keeps several, e.g. ``("NCAM", "ZCAM")`` for Navcam + Mastcam-Z.
    ``min_frame_fraction`` (default 1/3; None: off): products of ``frame_fraction_families`` (Navcam)
    covering less of the detector frame than this are left out - the single full-resolution tiles (1/16 of the
    frame, and a quarter-frame tile, 1/4) of a full-resolution acquisition, which also comes as one full frame.  Returns (paths, report).
    """
    prefixes = None if not sequence_prefix else tuple(
        x.upper() for x in ([sequence_prefix] if isinstance(sequence_prefix, str) else sequence_prefix))
    best: Dict[Tuple[str, str], Tuple[int, int, Path]] = {}
    n_in, dropped_seq = 0, 0
    fams = {str(f).upper() for f in (frame_fraction_families or ())}
    subframes: List[Dict[str, Any]] = []
    for p in map(Path, paths):
        n_in += 1
        fn = parse_filename(p)
        if prefixes and not fn.sequence.upper().startswith(prefixes):
            dropped_seq += 1
            continue
        size = (sizes or {}).get(p.name)
        if size is None:
            size = p.stat().st_size
        if min_frame_fraction and fn.family in fams:
            frac = frame_fraction(p, size)
            if frac is not None and frac < float(min_frame_fraction):
                subframes.append({"file": p.name, "frame_fraction": round(frac, 4)})
                continue
        key = (fn.instrument, fn.sclk_key)
        cand = (int(size), fn.version, p)
        if key not in best or cand[:2] > best[key][:2]:
            best[key] = cand
    kept = sorted((v[2] for v in best.values()), key=lambda q: (parse_filename(q).sclk, q.name))
    return kept, {"n_input": n_in, "n_dropped_sequence": dropped_seq, "n_exposure_products": len(kept),
                  "n_superseded": n_in - dropped_seq - len(subframes) - len(kept),
                  "sequence_prefix": list(prefixes) if prefixes else None,
                  "min_frame_fraction": min_frame_fraction, "n_dropped_subframes": len(subframes),
                  "dropped_subframes": subframes}


# -------------------------------------------------------------------- cameras
def camera_from_colmap_json(path: PathLike, zero_terms: Sequence[str] = ()) -> Dict[str, Any]:
    """
    A COLMAP camera stored as JSON (``model``, ``width``, ``height``, ``params``, optional ``free_params``).
    ``p1``/``p2`` in ``zero_terms`` are set to 0.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    params = [float(v) for v in d["params"]]
    names = PARAM_NAMES.get(d["model"], ())
    for n in ("p1", "p2"):
        if n in zero_terms and n in names:
            params[names.index(n)] = 0.0
    zeroed = [n for n in ("p1", "p2") if n in zero_terms]
    return {"model": d["model"], "width": int(d["width"]), "height": int(d["height"]), "params": params,
            "free_params": list(d.get("free_params") or []), "distortion": d.get("distortion"),
            "source": f"{Path(path).name}" + (f" ({', '.join(zeroed)} set to 0)" if zeroed else "")}


# -------------------------------------------------------------------- project
@dataclass
class SfmProject:
    root: Path
    images: List[Dict[str, Any]]
    cameras: Dict[str, Dict[str, Any]]
    rig: Dict[str, Any]
    offset: List[float]
    settings: Dict[str, Any] = field(default_factory=dict)

    # --- paths
    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def features_db(self) -> Path:
        return self.root / "features.db"

    @property
    def database(self) -> Path:
        return self.root / "database.db"

    def station_label(self, station: str) -> str:
        """``S032D1184`` -> ``Sol0686 S032D1184`` (see :func:`station_labels`)."""
        return station_labels(self.images).get(station, station)

    def image(self, name: str) -> Dict[str, Any]:
        for r in self.images:
            if r["name"] == name:
                return r
        raise KeyError(name)

    # --- persistence
    def save(self) -> Path:
        p = self.root / "project.json"
        d = {"images": self.images, "cameras": self.cameras, "rig": self.rig, "offset": self.offset,
             "settings": self.settings}
        p.write_text(json.dumps(d, indent=1, default=_json_default), encoding="utf-8")
        return p

    @classmethod
    def load(cls, root: PathLike) -> "SfmProject":
        root = Path(root)
        d = json.loads((root / "project.json").read_text(encoding="utf-8"))
        return cls(root, d["images"], d["cameras"], d["rig"], d["offset"], d.get("settings", {}))

    def refresh_images(self, metas: Sequence[Dict[str, Any]], link: bool = True) -> int:
        """
        Bring the project's ``images/`` and ``masks/`` up to date with the
        processed files (a hard link follows them already; a copy is replaced
        when the processed file is newer).  For a project that is reused rather
        than created again.  Returns the number of project images checked.
        """
        processed = Path(self.settings.get("processed_dir", ""))
        if not processed.is_dir() and (self.root.parent / "processed").is_dir():    # a renamed WORK folder
            processed = self.root.parent / "processed"
            self.settings["processed_dir"] = str(processed)
        fmt = self.settings.get("image_format", "PNG8")
        names = {r["name"] for r in self.images}
        n = 0
        for m in metas:
            name = Path(m["source_product"]).stem + ".png"
            if name not in names or fmt not in (m.get("outputs") or {}):
                continue
            _link_or_copy(processed / m["outputs"][fmt], self.images_dir / name, link)
            if "mask" in m["outputs"]:
                _link_or_copy(processed / m["outputs"]["mask"], self.masks_dir / (name + ".png"), link)
            n += 1
        return n

    # --- creation
    # --- creation
    @classmethod
    def create(cls, metas: Sequence[Dict[str, Any]], processed_dir: PathLike, root: PathLike,
               image_format: str = "PNG8", zero_terms: Sequence[str] = ZEROED_TERMS, link: bool = True,
               prior_sigma_m: Sequence[float] = (1.0, 1.0, 1.0),
               zcam_intrinsics: str = "focus_model", zcam_focus_bin: Optional[float] = ZCAM_FOCUS_BIN,
               zcam_bin_refine: str = "focal", zcam_rig: bool = False, zcam_rig_refine: str = "auto",
               zcam_low_backlash: Optional[PathLike] = None, zcam_line_params: Optional[Dict[str, Any]] = None,
               zcam_low_backlash_found: Optional[PathLike] = None, zcam_rig_fit: Optional[str] = None,
               navcam_distortion: str = "fisheye_tangential", zcam_hold_f_images: int = ZCAM_HOLD_F_IMAGES,
               navcam_rig: str = NAVCAM_RIG, navcam_cameras: Optional[PathLike] = None,
               zcam_zero_terms: Optional[Sequence[str]] = None,
               zcam_focus_model_file: Optional[PathLike] = None,
               navcam_distortion_fit: str = NAVCAM_DISTORTION_FIT, navcam_k4: str = "consensus",
               navcam_rig_yaw: str = "hold", navcam_rig_epoch_gap: Optional[float] = None,
               zcam_line_form: Any = None, navcam_rig_epoch_breaks: Optional[Sequence[float]] = None,
               navcam_rig_epochs_tie: bool = True, navcam_pp: str = NAVCAM_PP,
               navcam_rig_roll: str = "hold", hazcam_intrinsics: str = "radial",
               hazcam_rig_refine: str = "hold") -> "SfmProject":
        """
        ``metas``: ``MPPPImage.meta`` dicts with ``outputs`` (as in the MPPP manifest; paths relative to
        ``processed_dir``), padded to the detector frame.  One COLMAP camera per Navcam eye (from the promoted
        model at the block's median temperature and sol, :mod:`mppp.sfm.start_cameras`) and one per Mastcam-Z eye
        and zoom (``ZL034``/``ZR034``), split into focus bins.
        ``prior_sigma_m``: 1-sigma of the CAHV/waypoint camera positions (east, north, up) used as position priors.
        ``zcam_intrinsics``: ``"focus_model"`` (default): the Mastcam-Z eye starts from the zoom's promoted model
        (distortion, principal point at the reference focus) and each focus bin's focal length from its lines;
        ``"label"``: the median of the per-image label CAHVOR intrinsics of that eye and zoom.
        ``zcam_zero_terms``: the terms set to 0 in the Mastcam-Z cameras (e.g. ``("p1", "p2", "b1", "b2")``) when
        they differ from the Navcam ones (``zero_terms``); None: ``zero_terms`` for both.
        ``navcam_cameras``: a folder with Navcam model files to start from instead of the models in use (a candidate
        of :func:`mppp.calib.fit.write_navcam`); the files' ``verification`` blocks are kept in
        ``settings["navcam_cameras"]``.
        ``navcam_rig``: ``"consensus"`` (default) starts the rig rotation from the promoted rig at the block's median
        sol, the translation the fitted one when the file has it; ``"cahv"``: both from the label CAHV pairs.
        ``zcam_focus_model_file``: a focus-model JSON to use instead of the promoted one of its zoom (a candidate);
        the project records its path and SHA-256 (``settings["zcam_focus_model"]``).
        ``zcam_hold_f_images`` (default 2): with ``"focus_model"``, bins of at most this many images hold their
        focal length at the model (a bin of one or two images constrains it poorly).
        Mastcam-Z left and right exposures have different spacecraft clocks, so they are separate frames (no rig
        constraint unless ``zcam_rig``); Navcam pairs share the clock and form the left-referenced rig.
        ``zcam_focus_bin`` (default 30): each eye and zoom is split into cameras by focus motor count, bins at most
        this many counts wide (:func:`focus_bins`), named e.g. ``ZL034_F02312`` (the bin's median count).  None or 0:
        one camera per eye and zoom.
        ``zcam_bin_refine``: ``"focal"`` - a bin camera refines fx, fy only; ``"focal_pp"`` - and on ZR cx, cy;
        ``"model"`` - f held at the model; ``"all"`` - every parameter.  Each image records ``camera_group``
        (``ZL034``), ``focus_count``, ``label_f_px`` and ``focus_state``.
        Mastcam-Z priors: the label CAHVOR models move the principal point with focus and rotate the pointing to
        compensate, so each prior attitude is rotated to fit the principal point of its COLMAP camera
        (``prior_R_correction_deg``).
        ``zcam_rig`` (default False): True gives the two Mastcam-Z eyes of an exposure a stereo rig per zoom
        (the zoom model's rotation when it has one, held with ``zcam_rig_refine="auto"``, else the CAHV median).
        ``navcam_rig_yaw``: ``"hold"`` (default) keeps the rig file's yaw (at the block's sol) in every block;
        ``"refine"`` refines it with the rig.  ``navcam_pp`` / ``navcam_distortion_fit`` / ``navcam_k4``: see the
        module constants; ``navcam_distortion`` is ``"fisheye_tangential"`` (the only form).
        ``zcam_line_form``: the form of each Mastcam-Z focus line (default :data:`mppp.calib.ZCAM_LINE_FORM`; a dict
        {eye: {param: form}} overrides it for a study).
        ``hazcam_intrinsics`` (0.93): what a block refines of each Hazcam eye - "radial" (fx, fy, k1, k2; the
        default), "focal", "all" or "hold" (:data:`mppp.sfm.start_cameras.HAZCAM_INTRINSICS`); the eyes start from
        the fit to their label CAHVOREs (no Hazcam model is promoted yet).  ``hazcam_rig_refine``: "hold" (the CAHV
        stereo rig of each Hazcam pair) or "rotation" (refined in the adjustments that bring the Hazcams in).
        Both are recorded in ``settings["hazcam"]`` only when the block has Hazcam images.
        Scope: Navcam, the front and rear Hazcams, and Mastcam-Z at 34, 48, 63, 79 and 110 mm; anything else raises.
        """
        if navcam_rig not in ("consensus", "cahv"):
            raise ValueError("navcam_rig must be 'consensus' or 'cahv'")
        if navcam_distortion != "fisheye_tangential":
            raise ValueError("navcam_distortion: the Navcam form is 'fisheye_tangential' (THIN_PRISM_FISHEYE); "
                             "other lens models are no longer supported")
        if zcam_intrinsics not in ("label", "focus_model"):
            raise ValueError("zcam_intrinsics must be 'label' or 'focus_model'")
        if zcam_rig_refine not in ("auto", "hold", "refine"):
            raise ValueError("zcam_rig_refine must be 'auto', 'hold' or 'refine'")
        if zcam_bin_refine not in ZCAM_BIN_REFINE_SETTINGS:
            raise ValueError(f"zcam_bin_refine must be one of {ZCAM_BIN_REFINE_SETTINGS}")
        if navcam_distortion_fit not in ("hold", "refine"):
            raise ValueError("navcam_distortion_fit must be 'hold' or 'refine'")
        if navcam_rig_yaw not in NAVCAM_RIG_YAW_SETTINGS:
            raise ValueError(f"navcam_rig_yaw must be one of {NAVCAM_RIG_YAW_SETTINGS}")
        if navcam_rig_roll not in NAVCAM_RIG_YAW_SETTINGS:
            raise ValueError(f"navcam_rig_roll must be one of {NAVCAM_RIG_YAW_SETTINGS}")
        if navcam_k4 != "consensus":
            raise ValueError("navcam_k4: k3 and k4 come as a pair from the promoted model ('consensus'); k4 = 0 is "
                             "no longer a form")
        if navcam_pp not in NAVCAM_PP_SETTINGS:
            raise ValueError(f"navcam_pp must be one of {NAVCAM_PP_SETTINGS}")
        if hazcam_rig_refine not in ("hold", "rotation"):
            raise ValueError("hazcam_rig_refine must be 'hold' or 'rotation'")
        from .start_cameras import (HAZCAM_INTRINSICS, hazcam_start_camera, navcam_start_camera, navcam_start_rig,
                                    zcam_eye_camera)
        if hazcam_intrinsics not in HAZCAM_INTRINSICS:
            raise ValueError(f"hazcam_intrinsics must be one of {tuple(HAZCAM_INTRINSICS)}")
        processed_dir, root = Path(processed_dir), Path(root)
        nav_dir = Path(navcam_cameras) if navcam_cameras else cmods_dir()
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
        metas = [m for m in metas if "failed" not in m]
        if not metas:
            raise ValueError("no processed images")
        out_of_scope = [m["source_product"] for m in metas
                        if m["filename"]["camera_code"] not in SCOPE_CAMERA_CODES
                        or (m["filename"]["family"] == "Z" and m["filename"].get("zoom_mm") not in SCOPE_ZCAM_ZOOMS)]
        if out_of_scope:
            raise ValueError(f"{len(out_of_scope)} images outside MPPP's scope ({SCOPE}), e.g. {out_of_scope[0]}")
        undist = [m["source_product"] for m in metas if m.get("undistorted")]
        if undist:
            raise ValueError(f"{len(undist)} images were processed with resize.undistort=True (e.g. {undist[0]}); "
                             f"the COLMAP pipeline needs the original, distorted pixels")

        C = np.array([m["pose"]["C_enu_m"] for m in metas], float)
        offset = (np.floor(C.mean(axis=0) / 10.0) * 10.0).tolist()
        frames = {m["pose"]["frame"] for m in metas}
        if len(frames) != 1:
            raise ValueError(f"images are in different world frames {frames}; priors would be inconsistent")

        images, instruments, label_params = [], {}, {}
        for m in metas:
            fn = m["filename"]
            fam = fn["family"]
            s = float(fn["downsample_scale"])
            W, H = (int(round(v * s)) for v in FULL_FRAME[fam])
            iw, ih = int(m["intrinsics"]["width"]), int(m["intrinsics"]["height"])
            if (iw, ih) != (W, H):
                raise ValueError(f"{m['source_product']}: {iw}x{ih} is not the padded detector frame {W}x{H} at "
                                 f"scale {s:g} (process with resize.apply_padding=True, undistort=False)")
            stem = Path(m["source_product"]).stem
            src = processed_dir / m["outputs"][image_format]
            name = stem + ".png"
            _link_or_copy(src, root / "images" / name, link)
            if "mask" in m["outputs"]:
                _link_or_copy(processed_dir / m["outputs"]["mask"], root / "masks" / (name + ".png"), link)
            key = camera_key(fn)
            instruments[key] = fam
            label_f, label_c = None, None
            if fam == "Z":
                from ..colmap import colmap_camera_params
                model, params = colmap_camera_params(m["intrinsics"])
                full = np.asarray(params, float).copy()
                full[:4] /= s                              # to full resolution; distortion is resolution-free
                label_params.setdefault(key, []).append((model, full))
                label_f = float(full[0])
                label_c = [float(full[2]), float(full[3])]
            R = np.asarray(m["pose"]["R_world_to_cam"], float)
            images.append({
                "name": name, "stem": stem, "instrument": key, "eye": fn["eye"],
                "sclk_key": fn["sclk_key"], "sol": fn["sol"], "site": m.get("site"), "drive": m.get("drive"),
                "station": f"S{int(m['site']):03d}D{int(m['drive']):04d}",
                "sequence": fn["sequence"], "downsample_scale": s, "native_size": [iw, ih],
                "lmst": m.get("LMST"), "solar_elevation_deg": m.get("solar_elevation_deg"),
                "solar_azimuth_deg": m.get("solar_azimuth_deg"),
                "prior_C": (np.asarray(m["pose"]["C_enu_m"], float) - offset).tolist(),
                "prior_R_w2c": R.tolist(), "position_source": m["pose"].get("position_source"),
                "has_mask": "mask" in m["outputs"],
                "camera_group": key, "focus_count": _num(m.get("focus_position_count")), "label_f_px": label_f,
                "label_c_px": label_c,
                "camera_temperature_degC": m.get("camera_temperature_degC"),
                "zcam_optics_temperature_degC": m.get("zcam_optics_temperature_degC"),
            })

        _fm_cache: Dict[str, Any] = {}

        def _zcam_fmodel() -> Dict[str, Any]:
            if "m" not in _fm_cache:
                _fm_cache["m"] = zcam_focus_model(None, zcam_focus_model_file)
            return _fm_cache["m"]

        # the known low-backlash (regular focus state) images; every other one is in the backlash state
        from .backlash import is_low_backlash, load_low_backlash, low_backlash_fingerprint
        _lb = load_low_backlash(zcam_low_backlash, found=zcam_low_backlash_found)
        n_low = 0
        for r in images:
            if instruments.get(r["camera_group"]) == "Z":
                r["focus_state"] = calib.focus_state(is_low_backlash(r, _lb))
                n_low += r["focus_state"] == "regular"

        cameras = {}
        nav_info: Dict[str, Any] = {}
        for instr, fam in sorted(instruments.items()):
            zt = tuple(zcam_zero_terms) if (fam == "Z" and zcam_zero_terms is not None) else tuple(zero_terms)
            if fam == "Z":
                gm = _zcam_fmodel().get("cameras", {}).get(instr) if zcam_intrinsics == "focus_model" else None
                cameras[instr] = zcam_eye_camera(instr, label_params[instr], FULL_FRAME[fam], zt, gm)
                if "k3" in zt:
                    cameras[instr]["fixed_params"] = sorted(set(cameras[instr].get("fixed_params") or []) | {"k3"},
                                                            key=FULL_OPENCV_NAMES.index)
                continue
            if fam in HAZCAM_FAMILIES:
                cam, nav_info[instr] = hazcam_start_camera(
                    instr, images, [m for m in metas if camera_key(m["filename"]) == instr], nav_dir, hazcam_intrinsics)
                if (cam["width"], cam["height"]) != FULL_FRAME[fam]:
                    raise ValueError(f"{instr}: the camera is {cam['width']}x{cam['height']}, expected the full frame "
                                     f"{FULL_FRAME[fam]}")
                cameras[instr] = cam
                continue
            cam, nav_info[instr] = navcam_start_camera(instr, images, nav_dir, zt)
            if (cam["width"], cam["height"]) != FULL_FRAME[fam]:
                raise ValueError(f"{instr}: the model is {cam['width']}x{cam['height']}, expected the full frame {FULL_FRAME[fam]}")
            cameras[instr] = navcam_distortion_terms(cam, navcam_distortion_fit, pp=navcam_pp)
        focus_info = None
        if zcam_intrinsics == "focus_model" and any(f == "Z" for f in instruments.values()):
            fm = _zcam_fmodel()
            focus_info = {"file": str(zcam_focus_model_file) if zcam_focus_model_file else None,
                          "files": fm.get("sources", {}), "zooms": sorted(fm.get("zooms", {})),
                          "fingerprint": zcam_focus_model_fingerprint(zcam_focus_model_file)}
            missing = sorted({k for k, f in instruments.items() if f == "Z"} - set(fm.get("cameras", {})))
            if missing:
                focus_info["no_model"] = missing                  # start from the labels (f per bin)
        if zcam_focus_bin:
            fmodel = _zcam_fmodel() if zcam_intrinsics == "focus_model" else None
            cameras = _split_by_focus(images, cameras, instruments, float(zcam_focus_bin), zcam_bin_refine,
                                      fmodel, int(zcam_hold_f_images), line_params=zcam_line_params,
                                      line_form=zcam_line_form)

        _align_priors_to_cameras(images, cameras)
        rig = _rig_from_pairs([r for r in images if instruments.get(r["camera_group"]) != "Z"])
        zrig: Dict[str, Any] = {}
        if zcam_rig:
            # one Mastcam-Z rig per zoom for every focus bin (a COLMAP rig has one camera per eye, the focus
            # bins have many): the bundle adjustment poses a right image from its left partner of the same exposure
            # through this rig (``settings["zcam_virtual_rig"]``).  Rotation: the zoom's focus model ``rig`` when it
            # has one (held, ZCAM_RIG_REFINE = "auto"), else the CAHV median (refined); translation CAHV.
            zrows = [dict(r, instrument=r["camera_group"]) for r in images if instruments.get(r["camera_group"]) == "Z"]
            for fam_key, rg in _rig_from_pairs(zrows).items():
                z = _rig_zoom(fam_key)
                zr = (((_zcam_fmodel().get("zooms") or {}).get(z) or {}).get("rig")
                      if (z and zcam_intrinsics == "focus_model") else None)
                ent = {"ref": rg["ref"], "sensor": rg["sensor"], "n_pairs": rg["n_pairs"],
                       "R_sensor_from_ref": rg["R_sensor_from_ref"], "t_sensor_from_ref": rg["t_sensor_from_ref"],
                       "baseline_m": rg["baseline_m"], "rot_spread_deg": rg["rot_spread_deg"],
                       "rotation_source": "CAHV median of the pairs"}
                if zr and zr.get("R_sensor_from_ref"):
                    ent["R_sensor_from_ref_cahv"] = ent["R_sensor_from_ref"]
                    ent["R_sensor_from_ref"] = zr["R_sensor_from_ref"]
                    ent["rotation_source"] = f"Mastcam-Z {z} mm focus model rig ({zr.get('note', '')})"
                if zr and zr.get("t_sensor_from_ref"):        # the calibrated baseline (right from left)
                    ent["t_sensor_from_ref_cahv"] = ent["t_sensor_from_ref"]
                    ent["t_sensor_from_ref"] = [float(x) for x in zr["t_sensor_from_ref"]]
                    ent["baseline_m"] = float(np.linalg.norm(zr["t_sensor_from_ref"]))
                    ent["rotation_source"] += "; translation of the focus model"
                ent["hold"] = (bool(zr) if zcam_rig_refine == "auto" else zcam_rig_refine == "hold")
                zrig[fam_key] = ent
        for fam_key in [k for k in rig if k in ("N", "F", "R")]:
            if navcam_rig == "consensus":
                info = navcam_start_rig(rig[fam_key], images, fam_key, nav_dir)
                if info is not None:
                    nav_info["rig" if fam_key == "N" else f"rig_{fam_key}"] = info
        # rig epochs - the Navcam visits of a block more than ``navcam_rig_epoch_gap`` sols apart each get their own
        # rig rotation: "N" for the first visit in sol order, "N#1", "N#2" ... for the later ones, each started at the
        # drift model's value for its own median sol; a visit with images on both sides of the pitch knot is cut
        # there; the copies' intrinsics, distortion and baseline are tied to the eye's (``navcam_rig_epochs_tie``)
        gap = NAVCAM_RIG_EPOCH_GAP_SOLS if navcam_rig_epoch_gap is None else float(navcam_rig_epoch_gap)
        breaks = tuple(NAVCAM_RIG_EPOCH_BREAKS_SOLS if navcam_rig_epoch_breaks is None else navcam_rig_epoch_breaks)
        epochs_info = navcam_rig_epochs(images, rig, gap, nav_info, cameras=cameras, breaks=breaks,
                                        max_span=NAVCAM_RIG_EPOCH_MAX_SPAN_SOLS)
        proj = cls(root, images, cameras, rig, offset,
                   {"world_frame": frames.pop(), "image_format": image_format,
                    "prior_sigma_m": list(map(float, prior_sigma_m)), "zero_terms": list(zero_terms),
                    "zcam_zero_terms": None if zcam_zero_terms is None else list(zcam_zero_terms),
                    "processed_dir": str(processed_dir), "zcam_intrinsics": zcam_intrinsics,
                    "zcam_hold_f_images": int(zcam_hold_f_images) if zcam_intrinsics == "focus_model" else None,
                    "zcam_focus_model": focus_info,
                    "zcam_focus_bin": float(zcam_focus_bin) if zcam_focus_bin else None,
                    "zcam_bin_refine": zcam_bin_refine, "zcam_rig": bool(zcam_rig),
                    # the focus parameters of each eye that stage 2 puts on lines in focus count, per state
                    "zcam_line_params": ({str(k): list(v) for k, v in zcam_line_params.items()}
                                         if zcam_line_params else None),
                    "zcam_virtual_rig": zrig or None, "zcam_rig_refine": zcam_rig_refine,
                    "zcam_rig_fit": zcam_rig_fit,                       # "rotation" / "full" in stage 2
                    "navcam_rig_epochs": epochs_info,                   # the Navcam visits with their own rig
                    "navcam_rig_epochs_gap": gap,
                    "navcam_rig_epochs_breaks": list(breaks),
                    "navcam_rig_epochs_tie": bool(navcam_rig_epochs_tie),   # one camera per eye for all epochs
                    "zcam_line_form": zcam_line_form,                   # the form of every focus line
                    "zcam_low_backlash_sha256": (low_backlash_fingerprint(zcam_low_backlash, zcam_low_backlash_found)
                                                 if any(f == "Z" for f in instruments.values()) else None),
                    "zcam_low_backlash": {"file": _lb["file"], "images": int(n_low),
                                          "entries": len(_lb["entries"]) + len(_lb["sol_sequences"]),
                                          "found_file": _lb.get("found_file"), "found": _lb.get("found", 0)},
                    "navcam_distortion": navcam_distortion, "navcam_rig": navcam_rig, "prior_R_corrected": True,
                    "navcam_distortion_fit": navcam_distortion_fit, "navcam_k4": navcam_k4,
                    "navcam_pp": navcam_pp,
                    "navcam_rig_yaw": navcam_rig_yaw,
                    "navcam_rig_roll": navcam_rig_roll,                 # 0.91: the roll held at the file's trend
                    "navcam_rig_thermal": False,                      # no rig temperature term
                    "navcam_cameras": {"dir": str(nav_dir), "fingerprint": navcam_cameras_fingerprint(nav_dir),
                                       "models_fingerprint": calib.models_fingerprint(nav_dir), **nav_info}})
        haz = sorted(k for k, f in instruments.items() if f in HAZCAM_FAMILIES)
        if haz:                                       # 0.93: recorded only for a block with Hazcam images
            proj.settings["hazcam"] = {
                "intrinsics": hazcam_intrinsics, "rig_refine": hazcam_rig_refine, "cameras": haz,
                "rigs": sorted(k for k in rig if k in HAZCAM_FAMILIES),
                "images": {k: sum(1 for r in images if r["instrument"] == k) for k in haz},
                "start": {k: nav_info.get(k) for k in haz}}
        proj.save()
        return proj


def rig_yaw_mdeg(R) -> float:
    """The yaw of a rig rotation (right from left): its rotation-vector component about y, mdeg."""
    from scipy.spatial.transform import Rotation
    return float(np.degrees(Rotation.from_matrix(np.asarray(R, float)).as_rotvec()[1]) * 1e3)


def navcam_distortion_terms(cam: Dict[str, Any], fit: str = NAVCAM_DISTORTION_FIT, pp: str = "refine") -> Dict[str, Any]:
    """
    The Navcam distortion settings of one start camera (a ``project.cameras`` dict, changed and returned).

    ``fit="hold"``: every distortion parameter (all but fx, fy, cx, cy) goes into ``fixed_params`` and out of
    ``free_params`` - the distortion is the promoted camera's for every block, sol and temperature bin (thermal
    bins inherit ``fixed_params``); ``"refine"``: the block refines it (k1-k4 and k3 always, p1, p2 with
    TANGENTIAL = "refine").  The form's held zeros (sx1, sy1) are fixed in either case.
    ``pp="hold"``: cx and cy go into ``fixed_params`` too - a block keeps the promoted principal point (moved to the
    block temperature), the one carrier of the stereo temperature change.
    """
    names = PARAM_NAMES.get(cam["model"], FULL_OPENCV_NAMES)
    params = list(map(float, cam["params"]))
    fixed = list(cam.get("fixed_params") or [])
    free = list(cam.get("free_params") or [])
    if pp not in NAVCAM_PP_SETTINGS:
        raise ValueError(f"pp must be one of {NAVCAM_PP_SETTINGS}")
    if fit not in ("hold", "refine"):
        raise ValueError("fit must be 'hold' or 'refine'")
    held = [n for n in names if n not in INTRINSIC_CORE] if fit == "hold" else []
    held += [n for n in forms.NAVCAM.held_zero if n in names]
    if pp == "hold":
        held += ["cx", "cy"]
    for n in held:
        if n not in fixed:
            fixed.append(n)
    for n in forms.NAVCAM.held_zero:
        if n in names:
            params[names.index(n)] = 0.0
    free = [n for n in free if n not in fixed]
    cam = dict(cam, params=params, fixed_params=fixed, free_params=free, distortion_fit=fit, pp=pp)
    note = (["distortion held"] if fit == "hold" else []) + (["principal point held"] if pp == "hold" else [])
    if note:
        cam["source"] = f"{cam.get('source', '')}; " + "; ".join(note)
    return cam


def station_labels(images: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """
    Readable station names, sol first: ``S032D1184`` -> ``Sol0686 S032D1184``
    (``Sol0686-0688 S032D1184`` when the rover stayed there several sols).  The
    station ID itself (site/drive) is unchanged and still what groups images.
    """
    sols: Dict[str, set] = {}
    for r in images:
        if r.get("station") is None:
            continue
        s = sols.setdefault(r["station"], set())
        try:
            s.add(int(r["sol"]))
        except (KeyError, TypeError, ValueError):
            pass
    out = {}
    for st, s in sols.items():
        sol = "" if not s else (f"Sol{min(s):04d}" if len(s) == 1 else f"Sol{min(s):04d}-{max(s):04d}")
        out[st] = f"{sol} {st}".strip()
    return out


def _focus_tag(count: Optional[float]) -> str:
    """Camera-name suffix for a focus count: F02312, Fm00150 (negative), Fna (unknown)."""
    if count is None:
        return "Fna"
    m = int(round(count))
    return f"F{m:05d}" if m >= 0 else f"Fm{-m:05d}"


def _num(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def zcam_focus_model_files(folder: Optional[PathLike] = None) -> Dict[int, Path]:
    """{zoom mm: focus-model file} of the models in use (:func:`mppp.calib.zcam_files`)."""
    return calib.zcam_files(folder)


def _zoom_of_group(g: str) -> Optional[int]:
    try:
        return int(str(g)[2:5]) if str(g).startswith("Z") else None
    except ValueError:
        return None


def zcam_focus_model(folder: Optional[PathLike] = None, model_file: Optional[PathLike] = None) -> Dict[str, Any]:
    """The Mastcam-Z focus models of every zoom merged (:func:`mppp.calib.load_zcam_all`): ``cameras`` holds ZL034,
    ZR034, ZL048 ...; ``zooms[zoom]`` the rest of each file (``rig``, ``stereo`` ...); ``sources[zoom]`` the file.
    ``model_file``: a file (a candidate) whose cameras replace those of the same zoom."""
    out = calib.load_zcam_all(folder, model_file)
    if not out["cameras"] and not model_file:
        raise FileNotFoundError(f"no Mastcam-Z focus model ({ZCAM_FOCUS_MODEL_PATTERN.format(zoom=34)}) found")
    return out


def zcam_focus_model_path(folder: Optional[PathLike] = None, zoom: int = 34) -> Path:
    """The focus-model file of a zoom (default 34 mm) in use."""
    return calib.models.zcam_file(zoom, folder)


def _rig_zoom(fam_key: str) -> Optional[int]:
    """``"Z034"`` (a Mastcam-Z rig family) -> 34."""
    try:
        return int(str(fam_key)[1:4]) if str(fam_key).startswith("Z") else None
    except ValueError:
        return None


def zcam_model_focal(gm: Dict[str, Any], focus: float, T: Optional[float] = None,
                     sol: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
    """Focal length [px] of one eye's focus model (``zcam_focus_model()["cameras"][group]``) at ``focus``
    (motor counts), focal-plane temperature ``T`` (degC; the thermal term needs it) and ``sol`` (the trend term is
    evaluated at the sol clamped to the model's ``sol_range``).  Returns (f, {"focus", "thermal", "trend"} terms)."""
    terms = {"focus": float(gm["slope_px_per_count"]) * (float(focus) - float(gm["reference_focus"]))}
    th = gm.get("thermal") or {}
    terms["thermal"] = (float(th.get("f_px_per_degC") or 0.0) * (float(T) - float(th.get("T0_degC", 0.0)))
                        if T is not None and th else 0.0)
    tr = gm.get("trend") or {}
    if sol is not None and tr and tr.get("f_px_per_sol"):
        lo, hi = tr.get("sol_range") or (-np.inf, np.inf)
        terms["trend"] = float(tr["f_px_per_sol"]) * (float(np.clip(float(sol), lo, hi)) - float(tr.get("sol0", 0.0)))
    else:
        terms["trend"] = 0.0
    return float(gm["f0_px"]) + sum(terms.values()), terms


def zcam_model_pp_shift(gm: Optional[Dict[str, Any]], focus: Optional[float],
                        focus_ref: Optional[float]) -> Tuple[float, float]:
    """The principal-point shift (dcx, dcy) [px] of a focus bin at ``focus`` relative to ``focus_ref`` (the
    eye's median focus in the block, where the camera starts at the median label principal point).  Only the
    right-minus-left difference is observable with the narrow Mastcam-Z field (the absolute one trades with pointing);
    the shipped model puts it on ZR."""
    pp = (gm or {}).get("pp") or {}
    if focus is None or focus_ref is None or not pp:
        return 0.0, 0.0
    d = float(focus) - float(focus_ref)
    return float(pp.get("cx_px_per_count") or 0.0) * d, float(pp.get("cy_px_per_count") or 0.0) * d


def _split_by_focus(images: List[Dict[str, Any]], cameras: Dict[str, Dict[str, Any]], families: Dict[str, str],
                    width: float, refine: str, model: Optional[Dict[str, Any]] = None,
                    hold_f_images: int = 0, line_params: Optional[Dict[str, Any]] = None,
                    line_form: Any = None) -> Dict[str, Dict[str, Any]]:
    """Replace each Mastcam-Z camera by one camera per focus bin (see ``SfmProject.create``).  ``model``: the
    focal-length model (:func:`zcam_focus_model`); bins then start from it, and bins of at most ``hold_f_images``
    images hold f there.  The model's temperature and sol terms are evaluated at the bin's median focal-plane
    temperature (``camera_temperature_degC`` of Mastcam-Z images, HEAD_FPA) and sol, and its principal-point slope
    moves the bin's cx, cy from the eye's median label principal point (:func:`zcam_model_pp_shift`)."""
    out = {k: c for k, c in cameras.items() if families.get(k) != "Z"}
    for group in sorted(k for k in cameras if families.get(k) == "Z"):
        base = cameras[group]
        rows = [r for r in images if r["camera_group"] == group]
        bins = focus_bins([r["focus_count"] for r in rows], width)
        gm = (model or {}).get("cameras", {}).get(group)
        all_counts = [r["focus_count"] for r in rows if r["focus_count"] is not None]
        focus_ref = float(np.median(all_counts)) if all_counts else None
        # images of the known low-backlash (regular) focus state get their own bins
        bkeys = [(bb, r.get("focus_state") == "regular") for r, bb in zip(rows, bins)]
        for b in sorted(set(bkeys)):
            members = [r for r, bb in zip(rows, bkeys) if bb == b]
            counts = [r["focus_count"] for r in members if r["focus_count"] is not None]
            med = float(np.median(counts)) if counts else None
            regular = bool(b[1])
            key = f"{group}_{_focus_tag(med)}" + ("_reg" if regular else "")
            params = list(map(float, base["params"]))
            fl = [r["label_f_px"] for r in members if r.get("label_f_px") is not None]
            Ts = [float(r["camera_temperature_degC"]) for r in members if r.get("camera_temperature_degC") is not None]
            Tb = float(np.median(Ts)) if Ts else None
            sols = [float(r["sol"]) for r in members if r.get("sol") is not None]
            Sb = float(np.median(sols)) if sols else None
            lp = None
            if line_params:                                  # the parameters this eye puts on focus lines
                lp = line_params.get(group) or line_params.get(str(group)[:2]) or line_params.get(str(group)[1:2])
            # a line parameter whose form pins both its level and its slope (ZR cx) is never freed in a bin
            pinned = set()
            if lp:
                if isinstance(line_form, dict):
                    fm = line_form.get(group) or line_form.get(str(group)[:2]) or line_form.get(str(group)[1:2]) or {}
                else:
                    fm = forms.ZCAM_LINE_FORM.get(str(group)[:2], {})
                pinned = {pn for pn in lp if fm.get(pn) == "pinned"}
            held_here = (ZCAM_BIN_HELD if refine in ("focal", "model") else
                         (tuple(n for n in ZCAM_BIN_HELD if not (
                             (n in set(lp) - pinned) if lp else (str(group)[1:2] == "R" and n in ("cx", "cy"))))
                          if refine == "focal_pp" else ()))                  # ZR's cx, cy free, and k1, k2
            fixed = sorted(set(held_here) | set(base.get("fixed_params") or ()),
                           key=FULL_OPENCV_NAMES.index)                      # + the eye's (k3 = 0)
            lo, hi = (gm or {}).get("focus_range", [-np.inf, np.inf])
            extra = {}
            if gm and med is not None and not (lo <= med <= hi) and fl and gm.get("label_f0_px"):
                # outside the fitted focus range: the bin's label f, scaled by the model's refined/label ratio
                f_ref, terms = zcam_model_focal(gm, float(gm["reference_focus"]), Tb, Sb)
                ratio = f_ref / float(gm["label_f0_px"])
                f = float(np.median(fl)) * ratio
                a = float(gm.get("aspect", 1.0))
                params[0], params[1] = f / np.sqrt(a), f * np.sqrt(a)
                src = (f"{base['source']}; focus bin {key} ({len(members)} images, focus outside the model's "
                       f"range {lo:g}..{hi:g}: label f x {ratio:.4f})")
                extra = {"focus_model_terms_px": {k: v for k, v in terms.items() if k != "focus"}}
            elif gm and med is not None:
                f, terms = zcam_model_focal(gm, med, Tb, Sb)
                a = float(gm.get("aspect", 1.0))
                params[0], params[1] = f / np.sqrt(a), f * np.sqrt(a)
                src = f"{base['source']}; focus bin {key} ({len(members)} images, f from the focus model"
                src += "".join(f", {k} {v:+.1f} px" for k, v in terms.items() if k != "focus" and v) + ")"
                extra = {"focus_model_terms_px": terms}
                if refine == "model":                 # f linear in focus - held at the model
                    fixed = sorted(set(fixed) | {"fx", "fy"}, key=FULL_OPENCV_NAMES.index)
                    src += "; f held at the focus model (ZCAM_BIN_REFINE = 'model')"
                elif len(members) <= hold_f_images:
                    fixed = sorted(set(fixed) | {"fx", "fy"}, key=FULL_OPENCV_NAMES.index)
                    src += f"; f held (<= {hold_f_images} images)"
            elif fl and "label" in base.get("source", ""):
                f = float(np.median(fl))
                params[0], params[1] = f, f * params[1] / params[0]
                src = f"{base['source']}; focus bin {key} ({len(members)} images, f from the bin's labels)"
            else:
                src = f"{base['source']}; focus bin {key}"
            # principal point against focus (the model's right-minus-left slope), about the model's reference focus
            # when the eye starts at the model's absolute pp0, else about the eye's median focus
            pp_ref = base.get("pp_at_focus", focus_ref)
            dcx, dcy = zcam_model_pp_shift(gm, med, pp_ref)
            if dcx or dcy:
                params[2] += dcx
                params[3] += dcy
                src += f"; cx, cy {dcx:+.2f}, {dcy:+.2f} px (focus model pp slope about focus {pp_ref:.0f})"
                extra["pp_shift_px"] = [dcx, dcy]
            # a calibrated focus model carries one line per focus parameter and focus state,
            # p = a + b (focus - ref): the bin starts on its state's lines (mppp.calib.zcam_line_value)
            lines_used = False
            lmodel = (gm or {}).get("lines") or {}
            lst = lmodel.get("regular" if regular else "backlash") or {}
            if lst and med is not None:
                ref_l = float(lmodel.get("reference_focus", (gm or {}).get("reference_focus", med)))
                To = [float(r["zcam_optics_temperature_degC"]) for r in members
                      if r.get("zcam_optics_temperature_degC") is not None]
                Tl = float(np.median(To)) if To else None
                done = []
                for pn, e in lst.items():
                    if pn not in FULL_OPENCV_NAMES or "a" not in e:
                        continue
                    params[FULL_OPENCV_NAMES.index(pn)] = calib.zcam_line_value(e, med, ref_l, Tl)
                    done.append(pn)
                if done:
                    lines_used = "fx" in done
                    src += f"; {', '.join(done)} on the calibrated focus lines ({'regular' if regular else 'backlash'} state)"
                    extra["focus_lines"] = done
            if regular and lines_used:
                if refine != "focal_pp" or len(members) <= hold_f_images:
                    fixed = sorted(set(fixed) | {"fx", "fy"}, key=FULL_OPENCV_NAMES.index)
                extra = dict(extra, backlash_state="regular", state_source="list")
            elif regular:
                # the regular state sits at the label: the backlash model's f divided by its f0 / label f0 ratio
                ratio = (float(gm["f0_px"]) / float(gm["label_f0_px"])) if (gm and gm.get("label_f0_px")) else None
                if ratio:
                    params[0], params[1] = params[0] / ratio, params[1] / ratio
                    src += f"; regular focus state (ZCAM_LOW_BACKLASH): f / {ratio:.4f}"
                elif fl:
                    a = params[1] / params[0]
                    params[0] = float(np.median(fl))
                    params[1] = params[0] * a
                    src += "; regular focus state (ZCAM_LOW_BACKLASH): f of the bin's labels"
                if refine != "focal_pp" or len(members) <= hold_f_images:   # free, then set on its own line
                    fixed = sorted(set(fixed) | {"fx", "fy"}, key=FULL_OPENCV_NAMES.index)
                extra = dict(extra, backlash_state="regular", state_source="list")
            cam = dict(base, params=params, group=group, focus_count_median=med,
                       focus_count_range=[min(counts), max(counts)] if counts else None, n_images=len(members),
                       fixed_params=fixed, source=src, temperature_median_degC=Tb, sol_median=Sb, **extra)
            cam.pop("label_f_spread_px", None)
            if len(fl) > 1:
                cam["label_f_spread_px"] = float(np.ptp(fl))
            out[key] = cam
            for r in members:
                r["instrument"] = key
    return out


def _rotation_onto_axis(d: np.ndarray) -> np.ndarray:
    """Rotation taking direction ``d`` onto the optical axis (0, 0, 1)."""
    from scipy.spatial.transform import Rotation
    d = np.asarray(d, float) / np.linalg.norm(d)
    v = np.cross(d, [0.0, 0.0, 1.0])
    sn = float(np.linalg.norm(v))
    if sn < 1e-15:
        return np.eye(3)
    return Rotation.from_rotvec(v / sn * np.arctan2(sn, float(d[2]))).as_matrix()


def prior_rotation_correction(label_c: Sequence[float], label_f: float, camera_c: Sequence[float]) -> np.ndarray:
    """
    Rotation M (world-to-camera: R_new = M R_label) that makes a label attitude
    fit a camera whose principal point is ``camera_c`` instead of the label's
    ``label_c`` (both full-frame px, same focal length ``label_f``): the pixel
    at ``camera_c`` sees the ray the label model sees there.
    """
    d = (np.asarray(camera_c, float) - np.asarray(label_c, float)) / float(label_f)
    return _rotation_onto_axis(np.array([d[0], d[1], 1.0]))


def _align_priors_to_cameras(images: List[Dict[str, Any]], cameras: Dict[str, Dict[str, Any]]) -> None:
    """Rotate the prior attitude of every image with a label principal point to its camera's (in place)."""
    for r in images:
        if r.get("label_c_px") is None or r.get("label_f_px") is None:
            continue
        p = cameras[r["instrument"]]["params"]
        M = prior_rotation_correction(r["label_c_px"], r["label_f_px"], (p[2], p[3]))
        r["prior_R_w2c"] = (M @ np.asarray(r["prior_R_w2c"], float)).tolist()
        r["prior_R_correction_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1.0, 1.0))))


def navcam_rig_epochs(images: List[Dict[str, Any]], rig: Dict[str, Any], gap: float,
                      nav_info: Optional[Dict[str, Any]] = None,
                      cameras: Optional[Dict[str, Dict[str, Any]]] = None,
                      breaks: Sequence[float] = (), max_span: Optional[float] = None,
                      min_images: int = NAVCAM_RIG_EPOCH_MIN_IMAGES, min_stations: int = NAVCAM_RIG_EPOCH_MIN_STATIONS) -> Optional[Dict[str, Any]]:
    """
    Splits the Navcam rig of a block into one rig per *epoch* - the clusters of the Navcam images' sols
    separated by more than ``gap`` sols (the visits of a site the rover came back to).  With one cluster nothing
    changes.  With several, the first cluster (in sol order) keeps ``rig["N"]`` and the cameras NL, NR; every later
    cluster k gets its own copies ``rig["N#k"]`` with cameras ``NL#k``, ``NR#k`` (the same start values; a COLMAP
    database puts a camera in one rig only, so an epoch rig needs its own cameras - like the temperature bins), and
    its images' ``instrument`` becomes ``NL#k`` / ``NR#k`` (``rig_epoch`` k; the eye is :func:`mppp.sfm.thermal.
    eye_key`).  When the start rig came from a rig file with a drift (``nav_info["rig"]``, the consensus), each
    epoch's start rotation is the drift model's at that epoch's own median sol (and median Navcam temperature),
    else every epoch starts at the same rotation.  The bundle adjustment then refines the rigs (and the eye cameras'
    free terms) on their own.  Returns the epoch table {"gap_sols", "epochs": [{"index", "rig_key", "cameras",
    "sols", "sol_median", "T_median_degC", "images"}]} (None for one epoch); kept in ``settings["navcam_rig_epochs"]``.
    ``breaks`` (sols, default the pitch knot 300) cut an epoch whose images straddle them; the copies NL#k, NR#k are
    tied to NL, NR in the bundle adjustment (one parameter array), so an epoch differs from another only by its rig
    rotation.
    """
    nav = [r for r in images if str(r.get("instrument", "")).startswith("N") and r.get("sol") is not None]
    if "N" not in rig or not nav:
        return None
    eps = calib.sol_epochs([int(r["sol"]) for r in nav], gap, breaks, max_span)
    eps, merged = merge_small_epochs(eps, nav, min_images=min_images, min_stations=min_stations)
    for r in nav:
        r["rig_epoch"] = 0
    if len(eps) < 2:
        return None
    shipped = None
    rf = ((nav_info or {}).get("rig") or {}).get("file")
    if rf and Path(rf).is_file():
        try:
            shipped = json.loads(Path(rf).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            shipped = None
    base = dict(rig["N"])
    ref_key, sen_key = str(base["ref"]), str(base["sensor"])
    table = []
    for k, (a, b) in enumerate(eps):
        key = "N" if k == 0 else f"N#{k}"
        members = [r for r in nav if a <= int(r["sol"]) <= b]
        sols = [int(r["sol"]) for r in members]
        Ts = [float(r["camera_temperature_degC"]) for r in members if r.get("camera_temperature_degC") is not None]
        ep = {"index": k, "sols": [int(a), int(b)], "sol_median": float(np.median(sols)),
              "T_median_degC": float(np.median(Ts)) if Ts else None, "images": len(members)}
        ent = dict(base, epoch=ep) if k else rig["N"]
        if k:
            ent["ref"], ent["sensor"] = f"{ref_key}#{k}", f"{sen_key}#{k}"
            if cameras is not None:
                for eye in (ref_key, sen_key):
                    if eye in cameras:
                        cameras[f"{eye}#{k}"] = dict(cameras[eye], epoch=k, group=eye,
                                                     source=f"{cameras[eye].get('source', eye)}; rig epoch {k} (tied to {eye}) "
                                                            f"(sols {int(a)}-{int(b)}): the eye's camera again")
            for r in members:
                r["instrument"] = f"{r['instrument']}#{k}"
        else:
            rig["N"]["epoch"] = ep
        if shipped and shipped.get("drift"):
            R2, _, applied = calib.rig_at(shipped, float(np.median(sols)))
            ent["R_sensor_from_ref"] = R2.tolist()
            ent["epoch"]["drift_applied"] = applied
        ent["rotation_source"] = (base.get("rotation_source", "CAHV")
                                  + f"; rig epoch {k} (sols {int(a)}-{int(b)}, {len(members)} Navcam images)")
        for r in members:
            r["rig_epoch"] = k
        rig[key] = ent
        table.append(dict(ep, rig_key=key, cameras=[ent["ref"], ent["sensor"]]))
    return {"gap_sols": float(gap), "breaks_sols": [float(x) for x in breaks],
            "max_span_sols": None if max_span is None else float(max_span), "epochs": table,
            "merged_small_epochs": merged, "min_images": int(min_images), "min_stations": int(min_stations)}


def merge_small_epochs(eps: List[Tuple[int, int]], nav: List[Dict[str, Any]], min_images: int = NAVCAM_RIG_EPOCH_MIN_IMAGES,
                       min_stations: int = NAVCAM_RIG_EPOCH_MIN_STATIONS) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    """0.91: an epoch with fewer than ``min_images`` Navcam images or fewer than ``min_stations`` stations cannot
    determine a rig rotation of its own (Rochette's sol-341 single station of 20 Oct: 1.5 deg off): it is joined to
    the neighbouring epoch nearest in sol, repeatedly, until every epoch is large enough or one is left.  Returns
    (epochs, the merges made)."""
    eps = [tuple(e) for e in eps]
    merged: List[Dict[str, Any]] = []

    def size(e):
        m = [r for r in nav if e[0] <= int(r["sol"]) <= e[1]]
        return len(m), len({str(r.get("station") or r.get("site_drive") or r.get("name", "")[:9]) for r in m})

    while len(eps) > 1:
        small = [i for i, e in enumerate(eps) if size(e)[0] < min_images or size(e)[1] < min_stations]
        if not small:
            break
        i = small[0]
        e = eps[i]
        cands = []
        if i > 0:
            cands.append((e[0] - eps[i - 1][1], i - 1))
        if i + 1 < len(eps):
            cands.append((eps[i + 1][0] - e[1], i + 1))
        j = min(cands)[1]
        a, b = min(e[0], eps[j][0]), max(e[1], eps[j][1])
        merged.append({"epoch": [int(e[0]), int(e[1])], "images": size(e)[0], "stations": size(e)[1],
                       "joined_to": [int(eps[j][0]), int(eps[j][1])]})
        lo, hi = min(i, j), max(i, j)
        eps = eps[:lo] + [(a, b)] + eps[hi + 1:]
    return eps, merged


def _rig_family(key: str) -> str:
    """Camera key without the eye letter: NL -> N, ZL034 -> Z034."""
    return key[0] + key[2:]


def _rig_from_pairs(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Left-referenced rig per camera pair: the median right-from-left pose over
    all CAHV stereo pairs, i.e. left and right exposures with the same SCLK
    (Navcam).  Cameras without such pairs (Mastcam-Z) get no rig.
    """
    from scipy.spatial.transform import Rotation
    by = {}
    for r in images:
        by.setdefault((_rig_family(r["instrument"]), r["sclk_key"]), {})[r["eye"]] = r
    rigs: Dict[str, Any] = {}
    for fam in sorted({k[0] for k in by}):
        rv, tt, keys = [], [], None
        for (f, _), d in by.items():
            if f != fam or not ("L" in d and "R" in d):
                continue
            keys = (d["L"]["instrument"], d["R"]["instrument"])
            RL, RR = np.array(d["L"]["prior_R_w2c"]), np.array(d["R"]["prior_R_w2c"])
            CL, CR = np.array(d["L"]["prior_C"]), np.array(d["R"]["prior_C"])
            rv.append(Rotation.from_matrix(RR @ RL.T).as_rotvec())
            tt.append(RR @ (CL - CR))                       # x_R = R_rel x_L + t
        if not rv:
            continue
        rv, tt = np.array(rv), np.array(tt)
        R_med = Rotation.from_rotvec(np.median(rv, axis=0)).as_matrix()
        rigs[fam] = {"ref": keys[0], "sensor": keys[1], "n_pairs": len(rv),
                     "R_sensor_from_ref": R_med.tolist(), "t_sensor_from_ref": np.median(tt, axis=0).tolist(),
                     "baseline_m": float(np.linalg.norm(np.median(tt, axis=0))),
                     "rot_spread_deg": float(np.degrees(np.max(np.linalg.norm(rv - np.median(rv, 0), axis=1)))),
                     "t_spread_m": float(np.max(np.linalg.norm(tt - np.median(tt, 0), axis=1)))}
    return rigs


def _link_or_copy(src: Path, dst: Path, link: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists():
        # a hard link follows src; a copy is refreshed when src has been written again
        if os.path.samefile(src, dst):
            return
        a, b = src.stat(), dst.stat()
        if a.st_size == b.st_size and a.st_mtime_ns <= b.st_mtime_ns:
            return
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def zcam_focus_model_fingerprint(path: Optional[PathLike] = None, folder: Optional[PathLike] = None) -> str:
    """SHA-256 of the Mastcam-Z focus model files in use (every zoom's file, and ``path``), so that a project built
    with other models is told apart (:func:`mppp.calib.zcam_fingerprint`)."""
    return calib.zcam_fingerprint(folder, path)


def navcam_cameras_fingerprint(folder) -> Optional[str]:
    """SHA-256 over the Navcam camera and rig files of ``folder`` (:func:`mppp.calib.navcam_fingerprint`), so that a
    project is rebuilt when the models are rewritten in the same folder."""
    return calib.navcam_fingerprint(folder) if folder else None
