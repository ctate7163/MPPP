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
ZEROED_TERMS = ("p1", "p2", "b1", "b2")
ZCAM_FOCUS_BIN = 30.0                               # focus motor counts per Mastcam-Z camera bin (v0p14.4)
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
def select_best_products(paths: Iterable[PathLike], sizes: Optional[Dict[str, int]] = None,
                         sequence_prefix: Union[None, str, Sequence[str]] = "NCAM") -> Tuple[List[Path], Dict[str, Any]]:
    """
    One product per exposure (instrument + SCLK): the largest file (highest
    resolution / largest sub-frame), then the highest version.  ``sizes``
    maps file name -> bytes (default: stat the files).  ``sequence_prefix``
    keeps only e.g. NCAM sequences (drops SAPP sun images, SCAM support images);
    a tuple keeps several, e.g. ``("NCAM", "ZCAM")`` for Navcam + Mastcam-Z.
    Returns (paths, report).
    """
    prefixes = None if not sequence_prefix else tuple(
        x.upper() for x in ([sequence_prefix] if isinstance(sequence_prefix, str) else sequence_prefix))
    best: Dict[Tuple[str, str], Tuple[int, int, Path]] = {}
    n_in, dropped_seq = 0, 0
    for p in map(Path, paths):
        n_in += 1
        fn = parse_filename(p)
        if prefixes and not fn.sequence.upper().startswith(prefixes):
            dropped_seq += 1
            continue
        size = (sizes or {}).get(p.name)
        if size is None:
            size = p.stat().st_size
        key = (fn.instrument, fn.sclk_key)
        cand = (int(size), fn.version, p)
        if key not in best or cand[:2] > best[key][:2]:
            best[key] = cand
    kept = sorted((v[2] for v in best.values()), key=lambda q: (parse_filename(q).sclk, q.name))
    return kept, {"n_input": n_in, "n_dropped_sequence": dropped_seq, "n_exposure_products": len(kept),
                  "n_superseded": n_in - dropped_seq - len(kept),
                  "sequence_prefix": list(prefixes) if prefixes else None}


# -------------------------------------------------------------------- cameras
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

    # --- creation
    @classmethod
    def create(cls, metas: Sequence[Dict[str, Any]], processed_dir: PathLike, root: PathLike,
               image_format: str = "PNG8", xml_dir: Optional[PathLike] = None,
               zero_terms: Sequence[str] = ZEROED_TERMS, link: bool = True,
               prior_sigma_m: Sequence[float] = (1.0, 1.0, 1.0),
               zcam_intrinsics: str = "label", zcam_focus_bin: Optional[float] = ZCAM_FOCUS_BIN,
               zcam_bin_refine: str = "focal", zcam_rig: bool = False) -> "SfmProject":
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
        as ``label_f_spread_px``); ``"xml"``: ``mppp/data/m20_cmods/ZL034_frame.xml``
        etc. (rounded values; a rough start).  Either way f, c, k1-k3 are refined.
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
        point with focus (ZR034 at Three Forks: ~160 px in x, ~120 px in y) and
        rotate the pointing to compensate, so a label's attitude only fits its
        own principal point.  Each Mastcam-Z prior attitude is therefore rotated
        to fit the principal point of its COLMAP camera (the ray through the
        camera's principal point is kept); the correction is recorded as
        ``prior_R_correction_deg``.  At Three Forks this brings the spread of the
        left/right relative rotations from 1.29 deg to 0.08 deg.
        ``zcam_rig`` (v0p14.5, default False): no stereo rig for Mastcam-Z - each
        Mastcam-Z image is its own frame.  Even after the correction the pairs
        disagree by up to 0.08 deg (~7 px at f = 4700 px), too much for a rigid
        constraint; True restores the rig when left and right share a camera
        name pattern (only without focus bins).
        """
        if zcam_intrinsics not in ("label", "xml"):
            raise ValueError("zcam_intrinsics must be 'label' or 'xml'")
        if zcam_bin_refine not in ("focal", "all"):
            raise ValueError("zcam_bin_refine must be 'focal' or 'all'")
        processed_dir, root = Path(processed_dir), Path(root)
        xml_dir = Path(xml_dir) if xml_dir else data_dir() / "m20_cmods"
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
        metas = [m for m in metas if "failed" not in m]
        if not metas:
            raise ValueError("no processed images")

        C = np.array([m["pose"]["C_enu_m"] for m in metas], float)
        offset = (np.floor(C.mean(axis=0) / 10.0) * 10.0).tolist()
        frames = {m["pose"]["frame"] for m in metas}
        if len(frames) != 1:
            raise ValueError(f"images are in different world frames {frames}; priors would be inconsistent")

        images, instruments, label_params = [], {}, {}
        for m in metas:
            fn = m["filename"]
            fam = fn["family"]
            if fam not in FULL_FRAME:
                raise ValueError(f"{m['source_product']}: family {fam!r} not supported (Navcam/Hazcam/Mastcam-Z)")
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
                "prior_C": (np.asarray(m["pose"]["C_enu_m"], float) - offset).tolist(),
                "prior_R_w2c": R.tolist(), "position_source": m["pose"].get("position_source"),
                "has_mask": "mask" in m["outputs"],
                "camera_group": key, "focus_count": _num(m.get("focus_position_count")), "label_f_px": label_f,
                "label_c_px": label_c,
            })

        cameras = {}
        for instr, fam in sorted(instruments.items()):
            if fam == "Z" and zcam_intrinsics == "label":
                cameras[instr] = _camera_from_label_median(instr, label_params[instr], FULL_FRAME[fam], zero_terms)
                continue
            xml = xml_dir / (ZCAM_XML_PATTERN.format(camera=instr) if fam == "Z"
                             else XML_PATTERN.format(instrument=instr))
            cam = camera_from_metashape_xml(xml, zero_terms)
            if (cam["width"], cam["height"]) != FULL_FRAME[fam]:
                raise ValueError(f"{xml.name} is {cam['width']}x{cam['height']}, expected the full frame {FULL_FRAME[fam]}")
            cameras[instr] = cam
        if zcam_focus_bin:
            cameras = _split_by_focus(images, cameras, instruments, float(zcam_focus_bin), zcam_bin_refine)

        _align_priors_to_cameras(images, cameras)
        rig = _rig_from_pairs([r for r in images if zcam_rig or instruments.get(r["camera_group"]) != "Z"])
        proj = cls(root, images, cameras, rig, offset,
                   {"world_frame": frames.pop(), "image_format": image_format,
                    "prior_sigma_m": list(map(float, prior_sigma_m)), "zero_terms": list(zero_terms),
                    "processed_dir": str(processed_dir), "zcam_intrinsics": zcam_intrinsics,
                    "zcam_focus_bin": float(zcam_focus_bin) if zcam_focus_bin else None,
                    "zcam_bin_refine": zcam_bin_refine, "zcam_rig": bool(zcam_rig),
                    "prior_R_corrected": True})
        proj.save()
        return proj


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


def _split_by_focus(images: List[Dict[str, Any]], cameras: Dict[str, Dict[str, Any]], families: Dict[str, str],
                    width: float, refine: str) -> Dict[str, Dict[str, Any]]:
    """Replace each Mastcam-Z camera by one camera per focus bin (see ``SfmProject.create``)."""
    out = {k: c for k, c in cameras.items() if families.get(k) != "Z"}
    for group in sorted(k for k in cameras if families.get(k) == "Z"):
        base = cameras[group]
        rows = [r for r in images if r["camera_group"] == group]
        bins = focus_bins([r["focus_count"] for r in rows], width)
        for b in sorted(set(bins)):
            members = [r for r, bb in zip(rows, bins) if bb == b]
            counts = [r["focus_count"] for r in members if r["focus_count"] is not None]
            med = float(np.median(counts)) if counts else None
            key = f"{group}_{_focus_tag(med)}"
            params = list(map(float, base["params"]))
            fl = [r["label_f_px"] for r in members if r.get("label_f_px") is not None]
            if fl and "label" in base.get("source", ""):
                f = float(np.median(fl))
                params[0], params[1] = f, f * params[1] / params[0]
            cam = dict(base, params=params, group=group, focus_count_median=med,
                       focus_count_range=[min(counts), max(counts)] if counts else None, n_images=len(members),
                       fixed_params=list(ZCAM_BIN_HELD) if refine == "focal" else [],
                       source=f"{base['source']}; focus bin {key} ({len(members)} images, f from the bin's labels)"
                       if fl and "label" in base.get("source", "") else f"{base['source']}; focus bin {key}")
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
