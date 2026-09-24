"""
Configuration: defaults, loading, validation, and the snapshot written with
every output set.

A config is a plain nested ``dict``.  User files are deep-merged over
``default_config()`` so that a config written for an older version keeps
working; unknown keys raise a warning (not an error) and are preserved.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import platform
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]

_DEFAULTS: Dict[str, Any] = {
    "camera_model": {
        # World frame of the pose priors: local East-North-Up, metres, origin
        # at the M2020 landing site frame (site 3, drive 0).
        "extrinsics_from_waypoints": True,
        # Replace the label CAHVOR intrinsics by a calibrated Metashape XML.
        "intrinsics_from_xml": True,
        # camera family letter -> XML file pattern in mppp/data/m20_cmods (package data).
        # {eye} is L/R.  Only listed families are replaced.
        "xml_by_family": {"N": "M2020_N{eye}1_frame.xml"},
        # Resolution (relative to the full-resolution detector) at which the
        # XML calibration was made: 0.5 = half-scale 2560x1920 Navcam.
        "xml_scale_by_family": {"N": 0.5},
    },
    "radiometry": {
        "apply_tau_correction": True,
        "tau_reference": 0.3,
        "tau_table": "M2020_taus_versus_L_s.csv",
        # Floor on mu = sin(solar elevation) in the correction (avoids blow-up
        # near the horizon).
        "zenith_min": 0.2,
    },
    "color": {
        "enhance": True,
        "white_balance_ecam": [1.1, 1.4, 1.8],
        "white_balance_zcam": [1.0, 1.3, 2.0],
        "white_balance_vce": [1.1, 1.0, 0.9],
        "scale_rad_to_int16": 2e5,
        "scale_int8_to_int16": 64,
        "offset_int8_to_int16": 0.0,
        "apply_gamma": False,     # applies to the 8-bit product only
        "gamma": 2.0,
    },
    "masking": {
        "mask_invalid": True,
        "infer_mask": True,
        # v0p13: a registry name (the released model, downloaded once into the user cache),
        # a checkpoint path, or a file name in checkpoints_dir() - see mppp.mask.hub
        "checkpoint": "mppp_mask_v1",
        "device": "auto",                             # auto | cpu | cuda
        # null -> values from the checkpoint's model card
        "threshold": None,
        "dilate_kernel": None,
        "static_masks_dir": None,
    },
    "resize": {
        "apply_padding": True,        # pad sub-frames/tiles to the full detector frame
        "extra_padding": False,
        "extra_fraction": 0.1,
        "undistort": False,
        "recenter_principal_point": False,
    },
    "export": {
        # any of: PNG16, PNG8, TIFF16.  One sub-directory per format.
        "formats": ["PNG16"],
        "embed_metadata": True,
        "embed_gps": True,
        "store_mask_in_alpha": True,
        "write_mask_files": False,
        "write_references": True,
        "write_manifest": True,
        "write_colmap": True,
    },
    "verbose": False,
}

_LEGACY_FORMATS = {"PNG8A": (["PNG8"], True), "PNG16A": (["PNG16"], True),
                   "PNG8": (["PNG8"], None), "PNG16": (["PNG16"], None),
                   "TIFF16": (["TIFF16"], None), "TIFF16A": (["TIFF16"], True)}
VALID_FORMATS = ("PNG16", "PNG8", "TIFF16")


def default_config() -> Dict[str, Any]:
    return copy.deepcopy(_DEFAULTS)


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any], path: str = "") -> Dict[str, Any]:
    for k, v in over.items():
        here = f"{path}.{k}" if path else k
        if k not in base:
            warnings.warn(f"MPPP config: unknown key '{here}' (kept, but unused by this version).")
            base[k] = v
        elif isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v, here)
        else:
            base[k] = v
    return base


def _migrate_legacy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Accept keys used by the pre-package ``config.json``."""
    cfg = copy.deepcopy(cfg)
    exp = cfg.get("export", {})
    if "format" in exp:
        fmts, alpha = _LEGACY_FORMATS.get(str(exp.pop("format")).upper(), (None, None))
        if fmts is None:
            raise ValueError("export.format not recognised")
        exp.setdefault("formats", fmts)
        if alpha is not None:
            exp.setdefault("store_mask_in_alpha", alpha)
    exp.pop("colmap_model", None)          # chosen automatically from the coefficients
    m = cfg.get("masking", {})
    if "checkpoint_trained" in m:
        m.setdefault("checkpoint", m.pop("checkpoint_trained"))
    for dead in ("checkpoint_template", "checkpoint_config", "max_length"):
        m.pop(dead, None)
    r = cfg.get("resize", {})
    for dead in ("standard_ratio", "standard_width", "standard_height"):
        r.pop(dead, None)
    cm = cfg.get("camera_model", {})
    cm.pop("coordinate_frame", None)
    for dead in ("output_dir", "cameras", "anchor_sol", "preview", "json_logs"):
        cfg.pop(dead, None)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    fmts = cfg["export"]["formats"]
    if not fmts or any(f not in VALID_FORMATS for f in fmts):
        raise ValueError(f"export.formats must be a non-empty subset of {VALID_FORMATS}; got {fmts}")
    for key in ("white_balance_ecam", "white_balance_zcam", "white_balance_vce"):
        if len(cfg["color"][key]) != 3:
            raise ValueError(f"color.{key} must have three gains (R, G, B)")
    if cfg["color"]["scale_rad_to_int16"] <= 0 or cfg["color"]["scale_int8_to_int16"] <= 0:
        raise ValueError("color scales must be positive")
    if not 0 < cfg["radiometry"]["zenith_min"] <= 1:
        raise ValueError("radiometry.zenith_min must be in (0, 1]")


def load_config(config: Optional[Union[PathLike, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Defaults <- (optional) JSON file or dict.  Returns a validated dict."""
    cfg = default_config()
    if config is not None:
        if isinstance(config, dict):
            user = config
        else:
            p = Path(config)
            if not p.is_file():
                raise FileNotFoundError(f"Config file not found: {p}")
            user = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(user, dict):
                raise ValueError("Config JSON must be a top-level object.")
        _deep_merge(cfg, _migrate_legacy(user))
    validate_config(cfg)
    return cfg


def save_config_snapshot(cfg: Dict[str, Any], out_dir: PathLike,
                         extra: Optional[Dict[str, Any]] = None) -> Path:
    """
    Write ``mppp_config_<version>.json``: the exact configuration (colour,
    atmospheric, models) plus code version and run provenance.
    """
    from . import VERSION_TAG, __version__
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = {
        "mppp_version": __version__,
        "mppp_version_tag": VERSION_TAG,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": cfg,
    }
    if extra:
        snap.update(extra)
    path = out_dir / f"mppp_config_{VERSION_TAG}.json"
    path.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    return path
