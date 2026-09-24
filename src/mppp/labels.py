"""
PDS label access through the Planetary Data Reader (``pdr``).

``read_pds`` returns the image array and the label as a *plain* nested dict in
which PVL quantities (``{'value': v, 'units': u}``) are reduced to their value.
This isolates the rest of the package from differences between ``pdr``/``pvl``
versions (the pre-package code indexed ``label[...][0]`` and broke when ``pdr``
started returning unit-tagged quantities).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def _plain(obj: Any) -> Any:
    """Recursively convert pdr/pvl metadata into dict / list / scalars."""
    if hasattr(obj, "keys") and hasattr(obj, "__getitem__"):
        keys = list(dict.fromkeys(obj.keys()))           # unique, ordered (MultiDict safe)
        if set(keys) == {"value", "units"}:
            return _plain(obj["value"])
        return {k: _plain(obj[k]) for k in keys}         # MultiDict[k] -> first value
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def read_pds(path: PathLike, load_image: bool = True) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Read a PDS3/PDS4 product. Returns ``(label_dict, image or None)``."""
    try:
        import pdr
    except ImportError as e:                              # pragma: no cover
        raise ImportError("MPPP needs the Planetary Data Reader: pip install pdr") from e

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # DuplicateKeyWarning etc.
        data = pdr.read(str(path))
        label = _plain(data.metadata)
        image = None
        if load_image:
            image = np.asarray(data["IMAGE"])
    if image is not None:
        image = ensure_hwc(image)
    return label, image


def ensure_hwc(arr: np.ndarray) -> np.ndarray:
    """(H,W) stays; band-sequential (C,H,W) becomes (H,W,C)."""
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[0] < min(arr.shape[1:]):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unexpected image array shape {arr.shape}")
    return arr


def label_get(label: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    First present key wins.  Keys are dot paths ('GROUP.KEYWORD'),
    matched case-insensitively.
    """
    for key in keys:
        cur: Any = label
        for part in key.split("."):
            if not isinstance(cur, dict):
                cur = None
                break
            if part in cur:
                cur = cur[part]
                continue
            match = [k for k in cur if isinstance(k, str) and k.lower() == part.lower()]
            cur = cur[match[0]] if match else None
            if cur is None:
                break
        if cur is not None:
            return cur
    return default


def first(value: Any) -> Any:
    """Scalar, or first element of a sequence."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def label_float(label: Dict[str, Any], *keys: str, default: Optional[float] = None) -> Optional[float]:
    v = first(label_get(label, *keys))
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def as_vector(value: Iterable[Any], n: Optional[int] = None) -> np.ndarray:
    v = np.asarray(list(value), dtype=np.float64)
    if n is not None and v.shape != (n,):
        raise ValueError(f"Expected a vector of length {n}, got shape {v.shape}")
    return v
