"""
mppp.ibr.scales - what the colours of the per-image pictures in ``photometry_jpg/`` mean, and the way back to numbers
(0.97.7; Christian, 13 Nov 2026: "Do the photometry_jpg products have a constant scale so that we can take the jpg and
get the actual range, normal vector, emission angle ... We would need to know the float with units to color mapping.
Please make sure those mapping parameters are attached to the images and easy to read back into floats").

The scales are constants - the same for every image, station and run:

    <image>_depth_linear.jpg   range from the camera centre (m), 0 .. 75 linear, OpenCV colour map TURBO
    <image>_depth_log.jpg      the same range, 0.5 .. 200 m, log10, TURBO
    <image>_normal.jpg         the surface normal in the camera frame (x right, y down, z forward): n = 2 RGB / 255 - 1
    <image>_incidence.jpg      angle between the normal and the Sun (deg), 0 .. 90 linear, VIRIDIS
    <image>_emission.jpg       angle between the normal and the camera (deg), 0 .. 90 linear, VIRIDIS
    <image>_phase.jpg          angle Sun - surface - camera (deg), 0 .. 180 linear, VIRIDIS
    <image>_shadow.jpg         lit / shadow / unknown, three fixed colours
    (the sky is white in every picture but the shadow one)
    <image>_rad.jpg            the block's 8-bit picture of the frame - a view, not radiance

Every picture carries its scale as JSON in its XMP packet (``mppp:Scale``), and ``photometry_jpg/scales.json`` holds
all of them.  :func:`decode` turns a picture back into floats with units.

**The pictures are for looking; the numbers are in the .npz files.**  A JPEG holds 8 bits through a colour map and
then loses some to compression: read back on a smooth surface, an angle is good to about 0.2 deg (0 .. 90) or 0.4 deg
(phase), the linear range to 0.15 m, the log range to 1 %, a normal to 0.5 deg (medians; 2-5 times that at the 99.9 %
point, and worse at edges).  ``maps/<image>.npz``
(:func:`mppp.ibr.raycast.load_maps`: depth float32 m, range float16 m, normal int8 / 127, label) and
``photometry/<image>.npz`` (:func:`mppp.ibr.photometry.load_photometry`: incidence, emission, phase float16 deg,
shadow 0 / 1 / 2) are the record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

PathLike = Union[str, Path]
SCALES_FILE = "scales.json"
NPZ_NOTE = "exact values: maps/<image>.npz (depth, range, normal) and photometry/<image>.npz (incidence, emission, phase, shadow)"
NO_DATA_NOTE = "black [0, 0, 0] is the no-data flag: no surface, or the frame itself has no data there (its black padding)"


def scales() -> Dict[str, Dict[str, Any]]:
    """The scale of each picture, by its file-name suffix."""
    from .raycast import DEPTH_LINEAR_MAX_M, DEPTH_LOG_RANGE_M
    special = {"ground_plane": "the same colours x 0.55", "sky": [255, 255, 255], "masked_rgb": [0, 0, 0], "none": [0, 0, 0]}
    return {
        "depth_linear": {"quantity": "range from the camera centre", "units": "m", "map": "linear", "min": 0.0,
                         "max": float(DEPTH_LINEAR_MAX_M), "colormap": "TURBO", "value": "min + (max - min) * index / 255",
                         "special": special},
        "depth_log": {"quantity": "range from the camera centre", "units": "m", "map": "log10", "min": float(DEPTH_LOG_RANGE_M[0]),
                      "max": float(DEPTH_LOG_RANGE_M[1]), "colormap": "TURBO",
                      "value": "10 ** (log10(min) + (log10(max) - log10(min)) * index / 255)", "special": special},
        "normal": {"quantity": "unit surface normal in the camera frame (x right, y down, z forward)", "units": "1",
                   "map": "rgb", "value": "n = 2 * [R, G, B] / 255 - 1", "special": {"sky": [255, 255, 255], "no surface": [0, 0, 0]}},
        "incidence": {"quantity": "angle between the surface normal and the Sun", "units": "deg", "map": "linear",
                      "min": 0.0, "max": 90.0, "colormap": "VIRIDIS", "value": "max * index / 255", "special": {"sky": [255, 255, 255], "none": [0, 0, 0]}},
        "emission": {"quantity": "angle between the surface normal and the direction to the camera", "units": "deg",
                     "map": "linear", "min": 0.0, "max": 90.0, "colormap": "VIRIDIS", "value": "max * index / 255",
                     "special": {"sky": [255, 255, 255], "none": [0, 0, 0]}},
        "phase": {"quantity": "angle Sun - surface - camera", "units": "deg", "map": "linear", "min": 0.0, "max": 180.0,
                  "colormap": "VIRIDIS", "value": "max * index / 255", "special": {"sky": [255, 255, 255], "none": [0, 0, 0]}},
        "shadow": {"quantity": "cast shadow", "units": "class", "map": "classes",
                   "classes_rgb": {"0 unknown": [0, 0, 0], "1 lit": [255, 200, 90], "2 shadow": [20, 40, 120]}},
        "rad": {"quantity": "the block's 8-bit picture of the frame (white-balanced, tone-scaled)", "units": "none",
                "map": "view", "note": "not radiance: use the 16-bit linear frames (images16/) or the PDS product"},
    }


def kind_of(path: PathLike) -> Optional[str]:
    stem = Path(path).stem
    for k in sorted(scales(), key=len, reverse=True):
        if stem.endswith("_" + k):
            return k
    return None


def tag(path: PathLike, kind: Optional[str] = None) -> bool:
    """Attach the scale of a picture to it (XMP ``mppp:Scale``, JSON); False for a file that is none of them."""
    from .. import __version__
    from .xmp import tag_fields
    kind = kind or kind_of(path)
    sc = scales().get(kind or "")
    if sc is None:
        return False
    try:
        tag_fields(path, {"Product": kind, "Scale": json.dumps(sc), "ExactValues": NPZ_NOTE, "NoData": NO_DATA_NOTE, "Software": f"MPPP {__version__}"})
    except (OSError, ValueError):
        return False
    return True


def write_scales(folder: PathLike) -> Path:
    """``<folder>/scales.json``: every picture's scale, and how to read one back."""
    f = Path(folder) / SCALES_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"about": "the value scales of the pictures in this folder, by file-name suffix; each picture "
                                      "also carries its own in its XMP packet (mppp:Scale)",
                             "read_back": "from mppp.ibr.scales import decode; d = decode('<image>_emission.jpg'); d['value'] is "
                                          "float32 in d['units'], NaN where there is no value",
                             "exact_values": NPZ_NOTE, "no_data": NO_DATA_NOTE, "colormaps": "OpenCV COLORMAP_TURBO / COLORMAP_VIRIDIS, 256 entries; "
                                                                    "index = the entry nearest to the pixel's colour",
                             "scales": scales()}, indent=1), encoding="utf-8")
    return f


def _lut(name: str) -> np.ndarray:
    import cv2
    cm = {"TURBO": cv2.COLORMAP_TURBO, "VIRIDIS": cv2.COLORMAP_VIRIDIS}[name]
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cm).reshape(256, 3).astype(np.float32)   # BGR


def decode(path: PathLike, scale: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A picture of ``photometry_jpg/`` back as numbers: {"kind", "units", "value" float32 (H, W) - (H, W, 3) for the
    normal - NaN where there is none, "scale"}; the depth pictures also give "ground_plane" (bool).  The scale is
    read from the picture's own XMP, else taken from the file name."""
    import cv2
    from scipy.spatial import cKDTree
    from .xmp import read_fields
    f = read_fields(path)
    kind = f.get("Product") or kind_of(path)
    sc = scale or (json.loads(f["Scale"]) if f.get("Scale") else scales().get(kind or ""))
    if sc is None:
        raise ValueError(f"{path}: no scale in the file and none known for its name")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(str(path))
    out: Dict[str, Any] = {"kind": kind, "units": sc.get("units"), "scale": sc}
    px = img.reshape(-1, 3).astype(np.float32)
    dark = px.max(axis=1) < 24                                      # black: no value
    if sc["map"] == "rgb":
        white = (px.min(axis=1) > 235).reshape(img.shape[:2])            # the sky
        out["sky"] = white
        n = 2.0 * img[..., ::-1].astype(np.float32) / 255.0 - 1.0
        ln = np.linalg.norm(n, axis=2, keepdims=True)
        n = n / np.maximum(ln, 1e-6)
        n[dark.reshape(img.shape[:2]) | white] = np.nan
        out["value"] = n
        return out
    if sc["map"] == "classes":
        keys = list(sc["classes_rgb"])
        cols = np.array([sc["classes_rgb"][k][::-1] for k in keys], np.float32)
        _, i = cKDTree(cols).query(px)
        out["value"] = np.array([int(k.split()[0]) for k in keys], np.float32)[i].reshape(img.shape[:2])
        out["classes"] = keys
        return out
    if sc["map"] not in ("linear", "log10"):
        raise ValueError(f"{path}: a {sc['map']!r} picture holds no values to read back ({sc.get('note', '')})")
    lut = _lut(sc["colormap"])
    cols, plane_from = lut, None
    sp = sc.get("special") or {}
    extra = []
    if "ground_plane" in sp:                                         # the plane: the same colours darker
        plane_from = len(cols)
        cols = np.vstack([cols, np.floor(lut * 0.55)])
    for k in ("sky", "masked_rgb"):
        if k in sp:
            c = sp[k][::-1] if k.endswith("_rgb") else sp[k]
            extra.append(len(cols))
            cols = np.vstack([cols, np.asarray(c, np.float32)[None]])
    _, i = cKDTree(cols).query(px)
    idx = (i % 256).astype(np.float32) if plane_from is not None else i.astype(np.float32)
    lo, hi = float(sc["min"]), float(sc["max"])
    if sc["map"] == "linear":
        v = lo + (hi - lo) * idx / 255.0
    else:
        v = 10.0 ** (np.log10(lo) + (np.log10(hi) - np.log10(lo)) * idx / 255.0)
    bad = dark | np.isin(i, extra)
    if "sky" in sp:
        out["sky"] = (i == extra[0]).reshape(img.shape[:2]) if extra else np.zeros(img.shape[:2], bool)
    v[bad] = np.nan
    out["value"] = v.reshape(img.shape[:2]).astype(np.float32)
    if plane_from is not None:
        out["ground_plane"] = ((i >= plane_from) & (i < plane_from + 256) & ~bad).reshape(img.shape[:2])
    return out
