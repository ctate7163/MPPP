"""
The rover state of a PDS product from its label (study 3A, 0.94.4): the rover attitude and position in the site frame,
the rover motion counter, the RSM (mast) angles and head frame, the Navcam plate temperatures, the chassis
(suspension, steering) and arm joint angles, and the camera's CAHV pose in the rover frame.

Keyword names verified on M2020 RDR labels (Navcam, Hazcam, Mastcam-Z; sols 709-711):

- ``ROVER_COORDINATE_SYSTEM``: ``ORIGIN_ROTATION_QUATERNION`` (w, x, y, z; rover nav frame -> site frame),
  ``ORIGIN_OFFSET_VECTOR`` (rover origin in the site frame, m, north-east-down), ``COORDINATE_SYSTEM_INDEX``
  (= ``ROVER_MOTION_COUNTER``: SITE, DRIVE, POSE, ARM, SHA, DRILL, RSM, HGA, BITCAR, SEAL)
- ``RSM_COORDINATE_SYSTEM``: the RSM head frame in the rover nav frame (quaternion head -> rover, offset)
- ``RSM_ARTICULATION_STATE``: ``ARTICULATION_DEVICE_ANGLE`` with ``..._ANGLE_NAME`` ("AZIMUTH FINAL-RESOLVER", ...,
  "ELEVATION INITIAL-ENCODER"; rad), ``ARTICULATION_DEVICE_TEMP`` ("NAVCAM LEFT / RIGHT CAMERA PLATE", degC)
- ``CHASSIS_ARTICULATION_STATE``: steering, bogie and differential angles (rad)
- ``ARM_ARTICULATION_STATE``: the five joints (Hall sensors and resolvers, rad)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

RMC_NAMES = ("SITE", "DRIVE", "POSE", "ARM", "SHA", "DRILL", "RSM", "HGA", "BITCAR", "SEAL")
RSM_KEYS = {"AZIMUTH FINAL-RESOLVER": "az_resolver", "ELEVATION FINAL-RESOLVER": "el_resolver",
            "AZIMUTH FINAL-ENCODER": "az_encoder", "ELEVATION FINAL-ENCODER": "el_encoder",
            "AZIMUTH REQUESTED": "az_requested", "ELEVATION REQUESTED": "el_requested",
            "AZIMUTH INITIAL-RESOLVER": "az_initial", "ELEVATION INITIAL-RESOLVER": "el_initial"}
CHASSIS_KEYS = {"LEFT BOGIE": "left_bogie", "RIGHT BOGIE": "right_bogie", "LEFT DIFFERENTIAL": "left_differential",
                "RIGHT DIFFERENTIAL": "right_differential"}


def _floats(v: Any) -> List[Optional[float]]:
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        v = [v]
    out = []
    for x in v:
        if isinstance(x, dict):                          # {"value": .., "units": ..} from some readers
            x = x.get("value")
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _named(group: Optional[Dict[str, Any]], names_key: str, values_key: str) -> Dict[str, Optional[float]]:
    if not group:
        return {}
    names = group.get(names_key) or []
    if isinstance(names, str):
        names = [names]
    vals = _floats(group.get(values_key))
    return {str(n).upper(): v for n, v in zip(names, vals)}


def _frame(group: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not group:
        return None
    q = _floats(group.get("ORIGIN_ROTATION_QUATERNION"))
    o = _floats(group.get("ORIGIN_OFFSET_VECTOR"))
    if len(q) != 4 or len(o) != 3 or any(x is None for x in q + o):
        return None
    idx = [int(x) for x in _floats(group.get("COORDINATE_SYSTEM_INDEX")) if x is not None]
    return {"q_wxyz": q, "offset_m": o, "index": idx, "solution": group.get("SOLUTION_ID"),
            "reference": group.get("REFERENCE_COORD_SYSTEM_NAME")}


def rover_state(label: Dict[str, Any], width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
    """The rover state of one label (a plain dict: JSON-safe).  ``width`` / ``height`` (the image size) give the
    camera's CAHV pose in the rover frame (``R_cam_rnav``, ``C_rnav``)."""
    from mppp.labels import label_get
    st: Dict[str, Any] = {}
    rmc = [int(x) for x in _floats(label_get(label, "ROVER_MOTION_COUNTER")) if x is not None]
    st["rmc"] = dict(zip(RMC_NAMES, rmc))
    st["rover"] = _frame(label_get(label, "ROVER_COORDINATE_SYSTEM"))
    st["head"] = _frame(label_get(label, "RSM_COORDINATE_SYSTEM"))
    rsm = label_get(label, "RSM_ARTICULATION_STATE") or {}
    ang = _named(rsm, "ARTICULATION_DEVICE_ANGLE_NAME", "ARTICULATION_DEVICE_ANGLE")
    st["rsm"] = {short: ang.get(long) for long, short in RSM_KEYS.items()}
    temps = _named(rsm, "ARTICULATION_DEVICE_TEMP_NAME", "ARTICULATION_DEVICE_TEMP")
    st["navcam_plate_T"] = {"L": temps.get("NAVCAM LEFT CAMERA PLATE"), "R": temps.get("NAVCAM RIGHT CAMERA PLATE")}
    ch = _named(label_get(label, "CHASSIS_ARTICULATION_STATE"), "ARTICULATION_DEVICE_ANGLE_NAME",
                "ARTICULATION_DEVICE_ANGLE")
    st["chassis"] = {short: ch.get(long) for long, short in CHASSIS_KEYS.items()}
    arm = _named(label_get(label, "ARM_ARTICULATION_STATE"), "ARTICULATION_DEVICE_ANGLE_NAME",
                 "ARTICULATION_DEVICE_ANGLE")
    st["arm"] = {f"j{i}": arm.get(n) for i, n in enumerate(
        ("JOINT 1 AZIMUTH-RESOLVER", "JOINT 2 ELEVATION-RESOLVER", "JOINT 3 ELBOW-RESOLVER",
         "JOINT 4 WRIST-RESOLVER", "JOINT 5 TURRET-RESOLVER"), start=1)}
    st["sclk"] = _floats(label_get(label, "SPACECRAFT_CLOCK_START_COUNT"))[:1] or [None]
    st["sclk"] = st["sclk"][0]
    gcm = label_get(label, "GEOMETRIC_CAMERA_MODEL")
    if gcm is not None:
        try:
            from mppp.camera import CAHVOR
            w = width or int(label_get(label, "IMAGE.LINE_SAMPLES"))
            h = height or int(label_get(label, "IMAGE.LINES"))
            cv = CAHVOR.from_label(gcm)
            _, R = cv.decompose(w, h)
            st["R_cam_rnav"] = np.asarray(R, float).tolist()
            st["C_rnav"] = np.asarray(cv.C, float).tolist()
            st["camera_frame"] = str(gcm.get("REFERENCE_COORD_SYSTEM_NAME", "ROVER_NAV_FRAME"))
        except Exception as e:                           # noqa: BLE001 - recorded, the image is left out later
            st["camera_error"] = f"{type(e).__name__}: {e}"
    return st


def read_state(path: str) -> Dict[str, Any]:
    """:func:`rover_state` of a product file (its label only)."""
    from mppp.labels import read_pds
    label, _ = read_pds(path, load_image=False)
    return rover_state(label)


def locate(pds_root: Optional[Path], stems: Sequence[str]) -> Dict[str, Path]:
    """{stem: path} of the products named ``stems`` (without extension) below ``pds_root``: one walk of the tree."""
    import os
    want = {s.upper() for s in stems}
    out: Dict[str, Path] = {}
    if not pds_root or not Path(pds_root).is_dir() or not want:
        return out
    for dirpath, _, files in os.walk(pds_root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() == ".img" and stem.upper() in want:
                out[stem.upper()] = Path(dirpath) / f
        if len(out) == len(want):
            break
    return out


def states(stems: Sequence[str], pds_root: Optional[Path], cache_file: Optional[Path] = None,
           workers: int = 8, log=print) -> Dict[str, Dict[str, Any]]:
    """{stem: rover state} for ``stems``, from ``cache_file`` (JSON, updated) and else the labels below ``pds_root``
    (``workers`` processes)."""
    import json
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_file and Path(cache_file).is_file():
        try:
            cache = json.loads(Path(cache_file).read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    need = [s for s in stems if s.upper() not in cache]
    if need:
        paths = locate(pds_root, need)
        log(f"rover states: {len(stems) - len(need)} cached, {len(paths)} of {len(need)} labels found below {pds_root}")
        items = sorted(paths.items())
        if items:
            if workers > 1 and len(items) > 50:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    res = list(ex.map(_safe_read, [str(p) for _, p in items], chunksize=32))
            else:
                res = [_safe_read(str(p)) for _, p in items]
            for (s, _), r in zip(items, res):
                cache[s] = r
            if cache_file:
                Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(cache_file).with_suffix(".tmp")
                tmp.write_text(json.dumps(cache), encoding="utf-8")
                tmp.replace(cache_file)
    return {s: cache[s.upper()] for s in stems if s.upper() in cache}


def _safe_read(path: str) -> Dict[str, Any]:
    try:
        return read_state(path)
    except Exception as e:                               # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
