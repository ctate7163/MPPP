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
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..filenames import parse_filename
from ..paths import data_dir

PathLike = Union[str, Path]

# full detector frame (width, height) at downsample scale 1, by camera family
FULL_FRAME = {"N": (5120, 3840), "F": (5120, 3840), "R": (5120, 3840), "Z": (1648, 1200)}
XML_PATTERN = "M2020_{instrument}0_frame.xml"      # full-resolution calibrations (engineering cameras)
ZCAM_XML_PATTERN = "{camera}_frame.xml"            # Mastcam-Z, per eye and zoom, e.g. ZL034_frame.xml
NAVCAM_RATIONAL_PATTERN = "M2020_{instrument}_rational.json"   # v0p20: COLMAP rational Navcam cameras (full frame)
NAVCAM_FISHEYE_PATTERN = "M2020_{instrument}_fisheye_tangential.json"   # v0p35: THIN_PRISM_FISHEYE (sx1 = sy1 = 0)
NAVCAM_DISTORTION = "rational"                     # default: "rational" (full frame) or "polynomial" (Metashape K1-K3)
SCOPE = "Navcam (NLF/NRF) and Mastcam-Z at 34 mm (ZL0/ZR0 _034)"
SCOPE_CAMERA_CODES = ("NLF", "NRF", "ZL0", "ZR0")
# v0p50: parameter names per COLMAP model (reconstruction._PARAM_NAMES is this table)
PARAM_NAMES = {"FULL_OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"),
               "OPENCV": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"),
               "THIN_PRISM_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"),
               "OPENCV_FISHEYE": ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")}
INTRINSIC_CORE = ("fx", "fy", "cx", "cy")          # focal length and principal point; everything else is distortion
# v0p50: the Navcam distortion (k1-k4, p1, p2, ...) is one set per eye for every sol and temperature: "hold" keeps
# it at the start (consensus) camera in every block and temperature bin; "refine" fits it per block (before v0p50)
NAVCAM_DISTORTION_FIT = "hold"
NAVCAM_TERM_SETTINGS = ("consensus", "zero")       # v0p50: NAVCAM_K4 / NAVCAM_P1: the start camera's value or 0 (held)
ZEROED_TERMS = ("b1", "b2")          # v0p20: p1, p2 kept from the calibration (was also zeroed), as notebook 03
ZCAM_FOCUS_BIN = 30.0                               # focus motor counts per Mastcam-Z camera bin (v0p14.4)
ZCAM_FOCUS_MODEL = "M2020_ZCAM034_focus_model.json"  # v0p22: Mastcam-Z 34 mm focal length against focus count
ZCAM_HOLD_F_IMAGES = 2                              # v0p22: focus bins with <= this many images hold f at the model
NAVCAM_RIG = "consensus"                            # v0p22: Navcam rig rotation starts from the refined consensus
NAVCAM_RIG_FILE = "M2020_N_rig.json"
# v0p43.2: the Navcam consensus cameras shipped with MPPP (the v0p41 joint of 23 blocks: fisheye + tangential at
# -20 degC, f +38.1 ppm/degC, NL cx +0.0517 px/degC, rig with the mission drift; = camera_analysis/navcal_v0p41/
# navcam_joint).  v0p50: all camera models are in mppp/data/cmods (paths.cmods_dir; MPPP_CMODS overrides).
NAVCAM_PACKAGE_CONSENSUS_DIR = Path(__file__).resolve().parents[1] / "data" / "cmods"


def navcam_consensus_dir() -> Path:
    """Where notebook 03 finds the consensus Navcam cameras and rig by default: ``MPPP_CMODS`` when it holds them,
    else ``mppp/data/cmods`` (v0p50)."""
    from ..paths import cmods_dir
    d = cmods_dir()
    if d is not None and (d / NAVCAM_RIG_FILE).is_file():
        return d
    return NAVCAM_PACKAGE_CONSENSUS_DIR


NAVCAM_CONSENSUS_DIR = navcam_consensus_dir()
FULL_OPENCV_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
# parameters a focus-bin camera holds with zcam_bin_refine="focal": all but the focal length
ZCAM_BIN_HELD = ("cx", "cy", "k1", "k2", "p1", "p2", "k3")


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
NAVCAM_MIN_FRAME_FRACTION = 0.25    # select_best_products drops Navcam sub-frames / tiles below this (v0p44; 0.5 in v0p43.3)


def frame_fraction(path: PathLike, size: Optional[int] = None) -> Optional[float]:
    """v0p43.3: the fraction of the detector frame a Navcam product covers (image lines x samples against the full
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
    ``min_frame_fraction`` (v0p43.3; default 0.25 since v0p44, was 0.5; None: off): products of ``frame_fraction_families`` (Navcam)
    covering less of the detector frame than this are left out - the single full-resolution tiles (1/16 of the
    frame; a quarter-frame tile, exactly 1/4, is kept) of a full-resolution acquisition, which also comes as one full frame.  Returns (paths, report).
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
    A COLMAP camera stored as JSON (``model``, ``width``, ``height``,
    ``params``, optional ``free_params`` = parameters the bundle adjustment
    refines beyond the defaults, e.g. ``["k4"]`` for the rational Navcam
    model).  ``p1``/``p2`` in ``zero_terms`` are set to 0.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    params = [float(v) for v in d["params"]]
    names = PARAM_NAMES.get(d["model"], ())            # v0p50: any model (THIN_PRISM_FISHEYE p1, p2 were not zeroed)
    for n in ("p1", "p2"):
        if n in zero_terms and n in names:
            params[names.index(n)] = 0.0
    zeroed = [n for n in ("p1", "p2") if n in zero_terms]
    return {"model": d["model"], "width": int(d["width"]), "height": int(d["height"]), "params": params,
            "free_params": list(d.get("free_params") or []),
            "distortion": "rational" if "rational" in str(d.get("distortion", "")) else d.get("distortion"),
            "source": f"{Path(path).name}" + (f" ({', '.join(zeroed)} set to 0)" if zeroed else "")}


def read_metashape_calibration(path: PathLike) -> Dict[str, Any]:
    """Alias of :func:`mppp.camera.read_metashape_xml` (v0p13: one XML reader)."""
    from ..camera import read_metashape_xml
    return read_metashape_xml(path)


def camera_from_metashape_xml(path: PathLike, zero_terms: Sequence[str] = ZEROED_TERMS) -> Dict[str, Any]:
    """
    Metashape frame calibration -> COLMAP FULL_OPENCV at the XML's resolution.

    Metashape (corner pixel origin, as COLMAP):
        u = w/2 + cx + x' (f + b1) + y' b2,   v = h/2 + cy + y' f
        radial 1 + K1 r^2 + K2 r^4 + K3 r^6 (+ K4 r^8), tangential P1, P2.
    COLMAP FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 with the
    radial factor (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6),
    so Metashape's K1-K3 map exactly with k4 = k5 = k6 = 0; Metashape's K4
    (an r^8 numerator term) has no FULL_OPENCV equivalent and raises.
    OpenCV p1 = Metashape P2 and p2 = P1.  b2 (skew) cannot be represented:
    it must be in ``zero_terms`` (default zeroes p1, p2, b1, b2).
    """
    c = read_metashape_calibration(path)
    g = lambda k: 0.0 if k in zero_terms else float(c.get(k) or 0.0)       # noqa: E731
    if abs(float(c.get("k4") or 0.0)) > 0:
        raise ValueError(f"{Path(path).name}: K4 != 0 cannot be represented in COLMAP FULL_OPENCV")
    if abs(g("b2")) > 0:
        raise ValueError(f"{Path(path).name}: b2 (skew) cannot be represented in COLMAP; zero it")
    w, h = float(c["width"]), float(c["height"])
    f = float(c["f"])
    params = [f + g("b1"), f, w / 2 + float(c.get("cx") or 0.0), h / 2 + float(c.get("cy") or 0.0),
              float(c.get("k1") or 0.0), float(c.get("k2") or 0.0), g("p2"), g("p1"),
              float(c.get("k3") or 0.0), 0.0, 0.0, 0.0]
    return {"model": "FULL_OPENCV", "width": int(round(w)), "height": int(round(h)), "params": params,
            "source": f"{Path(path).name} ({', '.join(zero_terms)} set to 0)"}


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
        v0p20: bring the project's ``images/`` and ``masks/`` up to date with the
        processed files (a hard link follows them already; a copy is replaced
        when the processed file is newer).  For a project that is reused rather
        than created again.  Returns the number of project images checked.
        """
        processed = Path(self.settings.get("processed_dir", ""))
        if not processed.is_dir() and (self.root.parent / "processed").is_dir():    # v0p50: a renamed WORK folder
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
    @classmethod
    def create(cls, metas: Sequence[Dict[str, Any]], processed_dir: PathLike, root: PathLike,
               image_format: str = "PNG8", xml_dir: Optional[PathLike] = None,
               zero_terms: Sequence[str] = ZEROED_TERMS, link: bool = True,
               prior_sigma_m: Sequence[float] = (1.0, 1.0, 1.0),
               zcam_intrinsics: str = "focus_model", zcam_focus_bin: Optional[float] = ZCAM_FOCUS_BIN,
               zcam_bin_refine: str = "focal", zcam_rig: bool = False,
               navcam_distortion: str = NAVCAM_DISTORTION, zcam_hold_f_images: int = ZCAM_HOLD_F_IMAGES,
               navcam_rig: str = NAVCAM_RIG, navcam_cameras: Optional[PathLike] = None,
               zcam_zero_terms: Optional[Sequence[str]] = None,
               zcam_focus_model_file: Optional[PathLike] = None,
               navcam_distortion_fit: str = NAVCAM_DISTORTION_FIT, navcam_k4: str = "consensus",
               navcam_p1: str = "consensus") -> "SfmProject":
        """
        ``metas``: ``MPPPImage.meta`` dicts with ``outputs`` (as in the MPPP
        manifest; paths relative to ``processed_dir``), padded to the detector
        frame.  Engineering cameras (one COLMAP camera per eye, from the
        Metashape XML) and, since v0p13, Mastcam-Z (one camera per eye and
        zoom, ``ZL034``/``ZR034``).
        ``prior_sigma_m``: 1-sigma of the CAHV/waypoint camera positions
        (east, north, up) used as position priors.
        ``zcam_intrinsics``: initial Mastcam-Z camera - ``"label"`` (default):
        the median of the per-image label CAHVOR intrinsics of that eye and
        zoom (k1, k2; focus changes f by a few pixels between images, reported
        as ``label_f_spread_px``); ``"xml"``: ``mppp/data/cmods/ZL034_frame.xml``
        etc. (rounded values; a rough start); ``"focus_model"`` (v0p22,
        default): as ``"label"``, but each focus bin's focal length (and fy/fx)
        starts from the focal length against focus count fitted to earlier
        refined solutions (``cmods/M2020_ZCAM034_focus_model.json``, notebook
        05) instead of the label value, which is about 1 % short.
        ``zcam_zero_terms`` (v0p40): the terms set to 0 in the Mastcam-Z
        cameras (e.g. ``("p1", "p2", "b1", "b2")``) when they differ from the
        Navcam ones (``zero_terms``); None: ``zero_terms`` for both.
        ``navcam_cameras`` (v0p22.2): a folder with verified consensus Navcam
        cameras written by notebook 04 (``calibration.write_navcam_consensus``:
        ``M2020_NL_rational.json``, ``M2020_NR_rational.json`` and optionally
        ``M2020_N_rig.json``).  The rational Navcam cameras and, when present,
        the consensus rig rotation are read from there instead of the shipped
        ``cmods`` files; the files' ``verification`` blocks are kept in
        ``settings["navcam_cameras"]``.
        ``navcam_rig`` (v0p22): ``"consensus"`` (default, with the rational
        model) starts the right camera's rotation in the rig from the refined
        consensus of earlier solutions (``cmods/M2020_N_rig.json``, which
        matches the shipped rational cameras); the translation (the stereo
        baseline, which sets the scale) stays the project's CAHV value.
        ``"cahv"``: both from the label CAHV pairs (before v0p22).
        ``zcam_focus_model_file`` (v0p42): a focus-model JSON to use instead of the shipped
        ``cmods/M2020_ZCAM034_focus_model.json`` (e.g. the candidate notebook 04 §5b writes); the
        project records its path and SHA-256 (``settings["zcam_focus_model"]``).
        ``zcam_hold_f_images`` (v0p22, default 2): with ``"focus_model"``, bins of
        at most this many images hold their focal length at the model value
        (a bin of one or two images constrains it poorly: it scattered by
        about +-100 px).  Either way f, c, k1-k3 are refined.
        Mastcam-Z left and right exposures have different spacecraft clocks, so
        they are separate frames (no rig constraint); Navcam pairs share the
        clock and form the left-referenced rig.
        ``zcam_focus_bin`` (v0p14.4, default 30): Mastcam-Z focus breathing -
        each eye and zoom is split into cameras by focus motor count, bins at
        most this many counts wide (:func:`focus_bins`), named e.g.
        ``ZL034_F02312`` (the bin's median count); each bin starts from the
        median label focal length of its images.  None or 0: one camera per
        eye and zoom (<= 0.14.3).
        ``zcam_bin_refine``: ``"focal"`` (default) - a bin camera refines fx, fy
        only; principal point, distortion and p1, p2 are held at the median of
        the whole eye and zoom (label) or the XML (for a ~25 deg field the
        principal point of a few images is nearly degenerate with their
        attitude).  ``"all"``: every bin refines the same parameters as a
        Navcam camera.  Each image records ``camera_group`` (``ZL034``),
        ``focus_count`` and ``label_f_px``.
        Mastcam-Z priors (v0p14.5): the label CAHVOR models move the principal
        point with focus (by up to ~100 px or more) and
        rotate the pointing to compensate, so a label's attitude only fits its
        own principal point.  Each Mastcam-Z prior attitude is therefore rotated
        to fit the principal point of its COLMAP camera (the ray through the
        camera's principal point is kept); the correction is recorded as
        ``prior_R_correction_deg``.
        ``zcam_rig`` (v0p14.5, default False): no stereo rig for Mastcam-Z - each
        Mastcam-Z image is its own frame.  Even after the correction the pairs
        disagree by up to 0.08 deg (~7 px at f = 4700 px), too much for a rigid
        constraint; True restores the rig when left and right share a camera
        name pattern (only without focus bins).
        ``navcam_distortion`` (v0p20, default ``"rational"``): the Navcam
        cameras start from ``M2020_NL_rational.json`` / ``M2020_NR_rational.json``,
        COLMAP FULL_OPENCV with a rational radial term (k4 in the denominator,
        refined in the bundle adjustment).  The three-term polynomial of the
        Metashape calibration (``"polynomial"``) cannot be inverted beyond
        ~0.88 of the corner radius (~53 deg off-axis), so the frame corners
        (~9 % of each image) were never triangulated.
        Scope: Navcam and Mastcam-Z at 34 mm; anything else raises.
        """
        if navcam_rig not in ("consensus", "cahv"):
            raise ValueError("navcam_rig must be 'consensus' or 'cahv'")
        if navcam_distortion not in ("rational", "polynomial", "fisheye_tangential"):
            raise ValueError("navcam_distortion must be 'rational', 'polynomial' or 'fisheye_tangential'")
        if navcam_distortion == "fisheye_tangential" and not navcam_cameras:
            raise ValueError("navcam_distortion='fisheye_tangential' needs navcam_cameras: a folder with "
                             "M2020_NL/NR_fisheye_tangential.json (notebook 04, section 2e)")
        if zcam_intrinsics not in ("label", "xml", "focus_model"):
            raise ValueError("zcam_intrinsics must be 'label', 'xml' or 'focus_model'")
        if zcam_bin_refine not in ("focal", "all"):
            raise ValueError("zcam_bin_refine must be 'focal' or 'all'")
        if navcam_distortion_fit not in ("hold", "refine"):
            raise ValueError("navcam_distortion_fit must be 'hold' or 'refine'")
        for _n, _v in (("navcam_k4", navcam_k4), ("navcam_p1", navcam_p1)):
            if _v not in NAVCAM_TERM_SETTINGS:
                raise ValueError(f"{_n} must be one of {NAVCAM_TERM_SETTINGS}")
        processed_dir, root = Path(processed_dir), Path(root)
        xml_arg = xml_dir                       # None -> the focus model comes from cmods_dir() (MPPP_CMODS) first
        xml_dir = Path(xml_dir) if xml_dir else data_dir() / "cmods"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
        metas = [m for m in metas if "failed" not in m]
        if not metas:
            raise ValueError("no processed images")
        out_of_scope = [m["source_product"] for m in metas
                        if m["filename"]["camera_code"] not in SCOPE_CAMERA_CODES
                        or (m["filename"]["family"] == "Z" and m["filename"].get("zoom_mm") != 34)]
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
                "camera_temperature_degC": m.get("camera_temperature_degC"),       # v0p31 (manifest, MPPP >= 0.30)
            })

        cameras = {}
        nav_dir = Path(navcam_cameras) if navcam_cameras else xml_dir
        nav_info: Dict[str, Any] = {}
        for instr, fam in sorted(instruments.items()):
            zt = tuple(zcam_zero_terms) if (fam == "Z" and zcam_zero_terms is not None) else tuple(zero_terms)
            if fam == "Z" and zcam_intrinsics in ("label", "focus_model"):
                cameras[instr] = _camera_from_label_median(instr, label_params[instr], FULL_FRAME[fam], zt)
                continue
            if fam == "N" and navcam_distortion in ("rational", "fisheye_tangential"):
                xml = nav_dir / (NAVCAM_RATIONAL_PATTERN if navcam_distortion == "rational"
                                 else NAVCAM_FISHEYE_PATTERN).format(instrument=instr)
                cam = camera_from_colmap_json(xml, zt)
                if navcam_cameras:
                    js = json.loads(xml.read_text(encoding="utf-8"))
                    nav_info[instr] = {"file": str(xml), "verification": js.get("verification")}
                    # v0p31: a consensus at a reference temperature is scaled to this block's camera temperature
                    th = js.get("thermal")
                    Ts = [float(r["camera_temperature_degC"]) for r in images
                          if r["instrument"] == instr and r.get("camera_temperature_degC") is not None]
                    if th and Ts:
                        T = float(np.median(Ts))
                        sc = 1.0 + 1e-6 * float(th["ppm_per_degC"]) * (T - float(th["T0_degC"]))
                        cam["params"] = list(map(float, cam["params"]))
                        cam["params"][0] *= sc
                        cam["params"][1] *= sc
                        # v0p40: a principal point that moves with temperature (cx, cy px/degC about T0)
                        dcx = float(th.get("cx_px_per_degC") or 0.0) * (T - float(th["T0_degC"]))
                        dcy = float(th.get("cy_px_per_degC") or 0.0) * (T - float(th["T0_degC"]))
                        cam["params"][2] += dcx
                        cam["params"][3] += dcy
                        cam["source"] = (f"{cam.get('source', xml.name)}; fx, fy x {sc:.6f} for the median camera "
                                         f"temperature {T:.1f} degC (consensus at {float(th['T0_degC']):.1f} degC, "
                                         f"{float(th['ppm_per_degC']):+.1f} ppm/degC)"
                                         + (f"; cx, cy {dcx:+.3f}, {dcy:+.3f} px (principal point thermal)"
                                            if dcx or dcy else ""))
                        nav_info[instr]["thermal"] = {"T_median_degC": T, "scale": sc, **th}
                    tr = js.get("trend")                    # v0p41: principal point against sol (about sol0)
                    Ss = [float(r["sol"]) for r in images if r["instrument"] == instr and r.get("sol") is not None]
                    if tr and Ss:
                        S = float(np.median(Ss))
                        dcx_s = float(tr.get("cx_px_per_sol") or 0.0) * (S - float(tr["sol0"]))
                        dcy_s = float(tr.get("cy_px_per_sol") or 0.0) * (S - float(tr["sol0"]))
                        cam["params"] = list(map(float, cam["params"]))
                        cam["params"][2] += dcx_s
                        cam["params"][3] += dcy_s
                        cam["source"] = (f"{cam.get('source', xml.name)}; cx, cy {dcx_s:+.3f}, {dcy_s:+.3f} px for the "
                                         f"median sol {S:.0f} (trend about sol {float(tr['sol0']):.0f})")
                        nav_info[instr]["trend"] = {"sol_median": S, "dcx_px": dcx_s, "dcy_px": dcy_s, **tr}
            else:
                xml = xml_dir / (ZCAM_XML_PATTERN.format(camera=instr) if fam == "Z"
                                 else XML_PATTERN.format(instrument=instr))
                cam = camera_from_metashape_xml(xml, zt)
            if (cam["width"], cam["height"]) != FULL_FRAME[fam]:
                raise ValueError(f"{xml.name} is {cam['width']}x{cam['height']}, expected the full frame {FULL_FRAME[fam]}")
            if fam == "N":
                cam = navcam_distortion_terms(cam, navcam_distortion_fit, navcam_k4, navcam_p1)
            cameras[instr] = cam
        focus_info = None
        if zcam_focus_bin:
            fmodel = None
            if zcam_intrinsics == "focus_model":
                fmodel = (json.loads(Path(zcam_focus_model_file).read_text(encoding="utf-8")) if zcam_focus_model_file
                          else zcam_focus_model(xml_arg))
                focus_info = {"file": str(zcam_focus_model_file or zcam_focus_model_path(xml_arg)),
                              "fingerprint": zcam_focus_model_fingerprint(zcam_focus_model_file, xml_arg)}
            cameras = _split_by_focus(images, cameras, instruments, float(zcam_focus_bin), zcam_bin_refine,
                                      fmodel, int(zcam_hold_f_images))

        _align_priors_to_cameras(images, cameras)
        rig = _rig_from_pairs([r for r in images if zcam_rig or instruments.get(r["camera_group"]) != "Z"])
        if navcam_rig == "consensus" and "N" in rig and navcam_distortion in ("rational", "fisheye_tangential"):
            rig_file = data_dir() / "cmods" / NAVCAM_RIG_FILE
            if navcam_cameras and (nav_dir / NAVCAM_RIG_FILE).is_file():
                rig_file = nav_dir / NAVCAM_RIG_FILE
                nav_info["rig"] = {"file": str(rig_file)}
            shipped = json.loads(rig_file.read_text(encoding="utf-8"))
            rig["N"]["R_sensor_from_ref_cahv"] = rig["N"]["R_sensor_from_ref"]
            rig["N"]["R_sensor_from_ref"] = shipped["R_sensor_from_ref"]
            # v0p35: a rig with a temperature model and a drift starts at the block's median Navcam temperature and sol
            Ts = [float(r["camera_temperature_degC"]) for r in images
                  if str(r["instrument"]).startswith("N") and r.get("camera_temperature_degC") is not None]
            sols = [int(r["sol"]) for r in images if str(r["instrument"]).startswith("N") and r.get("sol") is not None]
            if "rig" in nav_info and (shipped.get("thermal") or shipped.get("drift")):
                R2, applied = start_rig_rotation(shipped, float(np.median(Ts)) if Ts else None,
                                                 float(np.median(sols)) if sols else None)
                rig["N"]["R_sensor_from_ref"] = R2.tolist()
                nav_info["rig"].update(applied)
            rig["N"]["rotation_source"] = f"{rig_file.name} (refined consensus; translation from CAHV)" + \
                (f" from {rig_file.parent}" if navcam_cameras else "")
        proj = cls(root, images, cameras, rig, offset,
                   {"world_frame": frames.pop(), "image_format": image_format,
                    "prior_sigma_m": list(map(float, prior_sigma_m)), "zero_terms": list(zero_terms),
                    "zcam_zero_terms": None if zcam_zero_terms is None else list(zcam_zero_terms),
                    "processed_dir": str(processed_dir), "zcam_intrinsics": zcam_intrinsics,
                    "zcam_hold_f_images": int(zcam_hold_f_images) if zcam_intrinsics == "focus_model" else None,
                    "zcam_focus_model": focus_info,
                    "zcam_focus_bin": float(zcam_focus_bin) if zcam_focus_bin else None,
                    "zcam_bin_refine": zcam_bin_refine, "zcam_rig": bool(zcam_rig),
                    "navcam_distortion": navcam_distortion, "navcam_rig": navcam_rig, "prior_R_corrected": True,
                    "navcam_distortion_fit": navcam_distortion_fit, "navcam_k4": navcam_k4, "navcam_p1": navcam_p1,
                    "navcam_cameras": {"dir": str(nav_dir), "fingerprint": navcam_cameras_fingerprint(nav_dir),
                                       **nav_info} if navcam_cameras else None})
        proj.save()
        return proj


def navcam_distortion_terms(cam: Dict[str, Any], fit: str = NAVCAM_DISTORTION_FIT, k4: str = "consensus",
                            p1: str = "consensus") -> Dict[str, Any]:
    """
    v0p50: the Navcam distortion settings of one start camera (a ``project.cameras`` dict, changed and returned).

    ``fit="hold"``: every distortion parameter (all but fx, fy, cx, cy) goes into ``fixed_params`` and out of
    ``free_params`` - the distortion is the consensus camera's for every block, sol and temperature bin (thermal
    bins inherit ``fixed_params``); ``"refine"``: the block refines it as before v0p50 (k1-k4 always, p1, p2 with
    TANGENTIAL = "refine").  ``k4="zero"`` / ``p1="zero"``: that term is set to 0 and held in either case.
    """
    names = PARAM_NAMES.get(cam["model"], FULL_OPENCV_NAMES)
    params = list(map(float, cam["params"]))
    fixed = list(cam.get("fixed_params") or [])
    free = list(cam.get("free_params") or [])
    zeroed = []
    for n, how in (("k4", k4), ("p1", p1)):
        if how == "zero" and n in names:
            params[names.index(n)] = 0.0
            zeroed.append(n)
            if n not in fixed:
                fixed.append(n)
    held = [n for n in names if n not in INTRINSIC_CORE] if fit == "hold" else []
    for n in held:
        if n not in fixed:
            fixed.append(n)
    free = [n for n in free if n not in fixed]
    cam = dict(cam, params=params, fixed_params=fixed, free_params=free, distortion_fit=fit,
               zeroed_terms=zeroed)
    note = ([f"{', '.join(zeroed)} = 0 (held)"] if zeroed else []) + (["distortion held"] if fit == "hold" else [])
    if note:
        cam["source"] = f"{cam.get('source', '')}; " + "; ".join(note)
    return cam


def station_labels(images: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """
    Readable station names, sol first (v0p14.4): ``S032D1184`` -> ``Sol0686 S032D1184``
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


def zcam_focus_model(xml_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    """The shipped Mastcam-Z focal-length model: per eye and zoom, f = f0 + slope (focus - reference) and fy/fx."""
    return json.loads(zcam_focus_model_path(xml_dir).read_text(encoding="utf-8"))


def zcam_focus_model_path(xml_dir: Optional[PathLike] = None) -> Path:
    """The focus-model file in use: ``xml_dir``'s, else ``cmods_dir()``'s (``MPPP_CMODS``), else the package's
    ``mppp/data/cmods`` (v0p50)."""
    from ..paths import cmods_dir
    cands = [Path(xml_dir) / ZCAM_FOCUS_MODEL] if xml_dir else []
    cands.append(cmods_dir() / ZCAM_FOCUS_MODEL)
    cands.append(data_dir() / "cmods" / ZCAM_FOCUS_MODEL)
    return next((f for f in cands if f.is_file()), cands[-1])


def zcam_model_focal(gm: Dict[str, Any], focus: float, T: Optional[float] = None,
                     sol: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
    """v0p42: focal length [px] of one eye's focus model (``zcam_focus_model()["cameras"][group]``) at ``focus``
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
    """v0p42: the principal-point shift (dcx, dcy) [px] of a focus bin at ``focus`` relative to ``focus_ref`` (the
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
                    hold_f_images: int = 0) -> Dict[str, Dict[str, Any]]:
    """Replace each Mastcam-Z camera by one camera per focus bin (see ``SfmProject.create``).  ``model``: the
    focal-length model (:func:`zcam_focus_model`); bins then start from it, and bins of at most ``hold_f_images``
    images hold f there.  v0p42: the model's temperature and sol terms are evaluated at the bin's median focal-plane
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
        for b in sorted(set(bins)):
            members = [r for r, bb in zip(rows, bins) if bb == b]
            counts = [r["focus_count"] for r in members if r["focus_count"] is not None]
            med = float(np.median(counts)) if counts else None
            key = f"{group}_{_focus_tag(med)}"
            params = list(map(float, base["params"]))
            fl = [r["label_f_px"] for r in members if r.get("label_f_px") is not None]
            Ts = [float(r["camera_temperature_degC"]) for r in members if r.get("camera_temperature_degC") is not None]
            Tb = float(np.median(Ts)) if Ts else None
            sols = [float(r["sol"]) for r in members if r.get("sol") is not None]
            Sb = float(np.median(sols)) if sols else None
            fixed = list(ZCAM_BIN_HELD) if refine == "focal" else []
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
                if len(members) <= hold_f_images:
                    fixed = sorted(set(fixed) | {"fx", "fy"}, key=FULL_OPENCV_NAMES.index)
                    src += f"; f held (<= {hold_f_images} images)"
            elif fl and "label" in base.get("source", ""):
                f = float(np.median(fl))
                params[0], params[1] = f, f * params[1] / params[0]
                src = f"{base['source']}; focus bin {key} ({len(members)} images, f from the bin's labels)"
            else:
                src = f"{base['source']}; focus bin {key}"
            # v0p42: principal point against focus (the model's right-minus-left slope, about the eye's median focus)
            dcx, dcy = zcam_model_pp_shift(gm, med, focus_ref)
            if dcx or dcy:
                params[2] += dcx
                params[3] += dcy
                src += f"; cx, cy {dcx:+.2f}, {dcy:+.2f} px (focus model pp slope about focus {focus_ref:.0f})"
                extra["pp_shift_px"] = [dcx, dcy]
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


def _camera_from_label_median(key: str, params: List[tuple], size: Tuple[int, int],
                              zero_terms: Sequence[str]) -> Dict[str, Any]:
    """FULL_OPENCV camera at full resolution from the median of per-image label intrinsics (COLMAP params)."""
    full = []
    for model, p in params:
        q = np.zeros(12)
        q[:4] = p[:4]
        if model in ("OPENCV", "FULL_OPENCV"):
            q[4:8] = p[4:8]
        if model == "FULL_OPENCV":
            q[8:] = p[8:12]
        full.append(q)
    P = np.array(full)
    med = np.median(P, axis=0)
    if "p1" in zero_terms:
        med[6] = 0.0
    if "p2" in zero_terms:
        med[7] = 0.0
    return {"model": "FULL_OPENCV", "width": int(size[0]), "height": int(size[1]), "params": med.tolist(),
            "source": f"median of {len(P)} label CAHVOR models ({key}; {', '.join(zero_terms)} set to 0)",
            "label_f_spread_px": float(np.ptp(P[:, 0])) if len(P) else 0.0}


def _link_or_copy(src: Path, dst: Path, link: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists():
        # a hard link follows src; a copy is refreshed when src has been written again (v0p14.7)
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


def start_rig_rotation(shipped: Dict[str, Any], T_median: Optional[float], sol_median: Optional[float]):
    """v0p35: the start rig rotation of a block from a rig file (``M2020_N_rig.json``) with a temperature model
    (``thermal``: yaw/pitch mdeg per degC about T0) and a drift (``drift``: pitch/yaw/roll mdeg per sol about sol0),
    at the block's median camera temperature and sol.  Returns (R, {"thermal": ..., "drift": ...} as applied)."""
    from scipy.spatial.transform import Rotation
    R = np.asarray(shipped["R_sensor_from_ref"], float)
    applied: Dict[str, Any] = {}
    th = shipped.get("thermal")
    if th and th.get("yaw_mdeg_per_degC") is not None and T_median is not None:
        dT = float(T_median) - float(th["T0_degC"])
        rv = np.radians(1e-3 * dT * np.array([float(th.get("pitch_mdeg_per_degC") or 0.0), float(th["yaw_mdeg_per_degC"]), 0.0]))
        R = Rotation.from_rotvec(rv).as_matrix() @ R
        applied["thermal"] = {**th, "T_median_degC": float(T_median)}
    dr = shipped.get("drift")
    if dr and sol_median is not None:
        # v0p35: rates x (sol - sol0); v0p35.1: an angle may have an early rate before ``knot_sol`` (a hinge:
        # + (early - rate) x (min(sol, knot_sol) - knot_ref)), see mppp.sfm.navcal.rig_drift_model
        ds = float(sol_median) - float(dr["sol0"])
        off = []
        for a in ("pitch", "yaw", "roll"):
            rate = float(dr.get(f"{a}_mdeg_per_sol") or 0.0)
            v = rate * ds
            if dr.get(f"{a}_early_mdeg_per_sol") is not None and dr.get("knot_sol"):
                v += (float(dr[f"{a}_early_mdeg_per_sol"]) - rate) * (min(float(sol_median), float(dr["knot_sol"]))
                                                                     - float(dr.get("knot_ref") or 0.0))
            off.append(float(v))
        rv = np.radians(1e-3 * np.array(off))
        R = Rotation.from_rotvec(rv).as_matrix() @ R
        applied["drift"] = {**dr, "sol_median": float(sol_median),
                            "offset_mdeg": {"pitch": off[0], "yaw": off[1], "roll": off[2]}}
    return R, applied


def zcam_focus_model_fingerprint(path: Optional[PathLike] = None, xml_dir: Optional[PathLike] = None) -> str:
    """v0p42: SHA-256 of the Mastcam-Z focus model file in use (``path``, else the one :func:`zcam_focus_model`
    reads), so that notebook 03 can tell when a project was built with another model."""
    import hashlib
    if path:
        f = Path(path)
    else:
        f = zcam_focus_model_path(xml_dir)
    return hashlib.sha256(f.read_bytes()).hexdigest()


def navcam_cameras_fingerprint(folder) -> Optional[str]:
    """v0p31: SHA-256 over the consensus Navcam camera files of ``folder`` (M2020_NL/NR_rational.json,
    M2020_N_rig.json), so that notebook 03 rebuilds a project when a consensus is rewritten in the same folder."""
    import hashlib
    if not folder:
        return None
    h = hashlib.sha256()
    n = 0
    for name in ("M2020_NL_rational.json", "M2020_NR_rational.json", "M2020_NL_fisheye_tangential.json",
                 "M2020_NR_fisheye_tangential.json", NAVCAM_RIG_FILE):
        f = Path(folder) / name
        if f.is_file():
            h.update(name.encode())
            h.update(f.read_bytes())
            n += 1
    return h.hexdigest() if n else None
