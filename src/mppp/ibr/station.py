"""
mppp.ibr.station - one station of a finished block and the pivot of its mast 360.

A **station** is one (site, drive) waypoint of the block (``project.json`` rows carry ``site``, ``drive`` and
``station`` = ``S<site>D<drive>``).  Its images are the frames taken there; nothing is re-aligned - the poses and
cameras stay those of the block's model, so everything made from the station sits in the block frame.

The **pivot** of a mast 360: each camera centre is ``C_i = P + R_i^T d_g``, with ``R_i`` the world-to-camera
rotation, ``P`` the pivot in the world and ``d_g`` the camera's fixed offset from the pivot in its own frame, one
``d_g`` per camera group (NL, NR, ZL034 ...).  ``P`` and the ``d_g`` are solved by linear least squares; the
residual RMS should be millimetres and the NL-NR offset about the Navcam baseline.  Fewer than six frames or no
azimuth spread: ``P`` falls back to the mean of the centres and the fit is flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

MIN_PIVOT_FRAMES = 6            # fewer: the pivot is the mean of the centres (flagged)
MIN_AZIMUTH_SPREAD_DEG = 60.0   # less: the pivot is poorly determined along the view direction (flagged)


@dataclass
class Station:
    """One station: its label, the names of its images, the pivot fit and the ENU axes in the block frame."""
    label: str                                  # S003D1234
    site: int
    drive: int
    sols: List[int]
    images: List[str]                           # image names of the frames at this station (in the model)
    pivot: np.ndarray                           # (3,) block frame, metres
    pivot_fit: Dict[str, Any] = field(default_factory=dict)
    R_block_from_enu: Optional[np.ndarray] = None   # (3, 3) or None: the block frame against label ENU
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))      # ENU up in the block frame
    north: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    east: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def to_json(self) -> Dict[str, Any]:
        return {"station": self.label, "site": self.site, "drive": self.drive, "sols": self.sols, "n_images": len(self.images),
                "images": self.images, "pivot_block_m": [float(v) for v in self.pivot], "pivot_fit": self.pivot_fit,
                "R_block_from_enu": None if self.R_block_from_enu is None else np.asarray(self.R_block_from_enu).tolist(),
                "up_block": [float(v) for v in self.up], "north_block": [float(v) for v in self.north],
                "east_block": [float(v) for v in self.east]}


# 0.97.0 (Christian, 12 Nov 2026: "The vertical height on the axis of azimuth rotations should be at the Mastcam-Z height
# even when rendering navcam.  This will give consistency."): the cameras on the mast head, in the mast frame of the
# 8 Nov note (origin: the elevation axis on the azimuth axis; forward along the level boresight, up) - metres
HEAD_FORWARD_UP_M = {"N": (0.1269, 0.05745), "Z": (0.0238, 0.06125)}     # Navcam (chain), Mastcam-Z 34 mm (5 stations)
ZCAM_HEIGHT_M = HEAD_FORWARD_UP_M["Z"][1]                                 # the Mastcam-Z centres above the elevation axis, level
PIVOT_HEIGHTS = ("zcam", "fit")


def mastcamz_height_pivot(pivot: np.ndarray, centres: np.ndarray, rotations: np.ndarray, groups: Sequence[str],
                          up: Sequence[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The pivot moved along ``up`` to the height of the Mastcam-Z centres with the head level: each frame's centre
    gives the height of the elevation axis (its own height less the camera's rise above the axis at its elevation,
    ``up_c cos(el) + forward_c sin(el)``); the pivot goes ZCAM_HEIGHT_M above their median.  The same point whether
    the station is rendered from Navcam or Mastcam-Z frames.  Returns (pivot, record)."""
    u = np.asarray(up, float)
    u = u / np.linalg.norm(u)
    C = np.asarray(centres, float).reshape(-1, 3)
    R = np.asarray(rotations, float).reshape(-1, 3, 3)
    h = []
    for Ci, Ri, g in zip(C, R, groups):
        fam = str(g)[:1].upper()
        if fam not in HEAD_FORWARD_UP_M:
            continue
        fwd, upc = HEAD_FORWARD_UP_M[fam]
        s_ = float(np.clip((Ri.T @ np.array([0.0, 0.0, 1.0])) @ u, -1.0, 1.0))
        h.append(float(Ci @ u) - (upc * float(np.sqrt(1.0 - s_ * s_)) + fwd * s_))
    if not h:
        return np.asarray(pivot, float), {"rule": "fit", "note": "no Navcam or Mastcam-Z frame: the fitted height kept"}
    axis_h = float(np.median(h))
    P = np.asarray(pivot, float)
    new = P + u * (axis_h + ZCAM_HEIGHT_M - float(P @ u))
    return new, {"rule": "zcam", "elevation_axis_height_spread_m": float(np.std(h)), "frames": len(h),
                 "moved_up_m": float((new - P) @ u), "zcam_above_elevation_axis_m": ZCAM_HEIGHT_M}


def parse_station(spec: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """``S003D1234`` / ``3_1234`` / ``3/1234`` -> (site, drive, None); ``sol:91`` -> (None, None, 91)."""
    s = str(spec).strip()
    m = re.fullmatch(r"sol:(\d+)", s, re.I)
    if m:
        return None, None, int(m.group(1))
    m = re.fullmatch(r"S(\d+)D(\d+)", s, re.I) or re.fullmatch(r"(\d+)[_/](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    raise ValueError(f"station {spec!r}: use S<site>D<drive>, <site>_<drive> or sol:<sol>")


def station_images(rows: Sequence[Dict[str, Any]], spec: str, families: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """The project rows at the station ``spec`` (:func:`parse_station`); ``sol:N`` takes the station of that sol's
    last frame.  ``families``: letters of the camera families to keep (e.g. ``"N"``, ``"NZ"``).  Returns
    (station label, rows)."""
    site, drive, sol = parse_station(spec)
    rows = [r for r in rows if r.get("site") is not None and r.get("drive") is not None]
    if sol is not None:
        on_sol = [r for r in rows if int(r.get("sol") or -1) == sol]
        if not on_sol:
            raise ValueError(f"no frame of sol {sol} in the block")
        last = max(on_sol, key=lambda r: (int(r["site"]), int(r["drive"])))
        site, drive = int(last["site"]), int(last["drive"])
    label = f"S{site:03d}D{drive:04d}"
    out = [r for r in rows if int(r["site"]) == site and int(r["drive"]) == drive]
    if families:
        fams = set(str(families).upper())
        out = [r for r in out if str(r.get("name", ""))[:1].upper() in fams]
    if not out:
        raise ValueError(f"no frame at station {label} in the block")
    return label, out


def _group_of(name: str, row: Optional[Dict[str, Any]] = None) -> str:
    if row and row.get("camera_group"):
        return str(row["camera_group"])
    n = Path(name).name.upper()
    return n[:3] if n.startswith("Z") else n[:2]


def fit_pivot(centres: np.ndarray, rotations: np.ndarray, groups: Sequence[str]) -> Dict[str, Any]:
    """Least squares for the pivot ``P`` and one offset ``d_g`` per group from ``C_i = P + R_i^T d_g``.
    ``centres`` (n, 3), ``rotations`` (n, 3, 3) world-to-camera, ``groups`` (n,).  Returns {"pivot", "offsets"
    {group: d_g}, "rms_m", "n", "azimuth_spread_deg", "fallback" (None or the reason), "baselines_m" {pair: |d_a - d_b|}}."""
    C = np.asarray(centres, float).reshape(-1, 3)
    R = np.asarray(rotations, float).reshape(-1, 3, 3)
    groups = [str(g) for g in groups]
    n = len(C)
    gs = sorted(set(groups))
    # azimuth spread of the viewing directions about the vertical of the first frame's rows (z of the world here)
    view = np.array([Ri.T @ np.array([0.0, 0.0, 1.0]) for Ri in R])
    az = np.degrees(np.arctan2(view[:, 0], view[:, 1]))
    spread = float(np.ptp(np.sort(az))) if n else 0.0
    if n > 1:
        gaps = np.diff(np.sort(az))
        wrap = 360.0 - (np.max(az) - np.min(az))
        spread = 360.0 - max(float(np.max(gaps)) if len(gaps) else 0.0, wrap)
    out: Dict[str, Any] = {"n": int(n), "groups": gs, "azimuth_spread_deg": round(spread, 1), "fallback": None}
    if n < MIN_PIVOT_FRAMES or spread < MIN_AZIMUTH_SPREAD_DEG:
        out["fallback"] = ("fewer than %d frames" % MIN_PIVOT_FRAMES) if n < MIN_PIVOT_FRAMES else "too little azimuth spread"
        out["pivot"] = C.mean(axis=0) if n else np.zeros(3)
        out["offsets"] = {g: np.zeros(3) for g in gs}
        out["rms_m"] = float(np.sqrt(np.mean(np.sum((C - out["pivot"]) ** 2, axis=1)))) if n else 0.0
        out["baselines_m"] = {}
        return out
    # unknowns: P (3), d_g (3 per group); row i: C_i = P + R_i^T d_{g(i)}
    nu = 3 + 3 * len(gs)
    A = np.zeros((3 * n + 1, nu)); b = np.concatenate([C.reshape(-1), [0.0]])
    gi = {g: k for k, g in enumerate(gs)}
    for i in range(n):
        A[3 * i:3 * i + 3, 0:3] = np.eye(3)
        k = gi[groups[i]]
        A[3 * i:3 * i + 3, 3 + 3 * k:6 + 3 * k] = R[i].T
    # a pan at one elevation leaves the pivot's height against the offsets along the pan axis undetermined (every
    # frame is turned about the same axis): a weak row P_z = mean camera height (eye level) picks that one; with
    # two tiers the row changes the answer by ~w^2 (nothing)
    w = 1e-3
    A[3 * n, 2] = w
    b[3 * n] = w * float(C[:, 2].mean())
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = (A[:3 * n] @ x - b[:3 * n]).reshape(-1, 3)
    sv = np.linalg.svd(A[:3 * n], compute_uv=False)
    out["height_determined"] = bool(sv[-1] > 1e-6 * sv[0])
    out["pivot"] = x[:3]
    out["offsets"] = {g: x[3 + 3 * gi[g]:6 + 3 * gi[g]] for g in gs}
    out["rms_m"] = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    out["baselines_m"] = {f"{a}-{b_}": float(np.linalg.norm(out["offsets"][a] - out["offsets"][b_]))
                          for i, a in enumerate(gs) for b_ in gs[i + 1:]}
    return out


def select_station(project, rec, spec: str, families: str = "N", pivot_height: str = "zcam") -> Station:
    """The :class:`Station` ``spec`` of ``project`` (an :class:`mppp.sfm.project.SfmProject`) in the model ``rec``
    (a ``pycolmap.Reconstruction``, the block's native-pixel model): its registered images of ``families``, the
    pivot fit over them, and the ENU axes of the block frame (``error_input/block_frame.json`` when the export
    wrote it, else :func:`mppp.sun.block_rotation`)."""
    import json
    label, rows = station_images(project.images, spec, families)
    by_name = {im.name: im for im in rec.images.values()}
    kept, C, R, G = [], [], [], []
    for r in rows:
        im = by_name.get(r["name"])
        if im is None or not im.has_pose:
            continue
        kept.append(r["name"])
        C.append(np.asarray(im.projection_center(), float))
        R.append(np.asarray(im.cam_from_world().rotation.matrix(), float))
        G.append(_group_of(r["name"], r))
    if not kept:
        raise ValueError(f"station {label}: none of its {len(rows)} frames is registered in the model")
    fit = fit_pivot(np.array(C), np.array(R), G)
    rot = None
    bf = Path(project.root) / "error_input" / "block_frame.json"
    if bf.is_file():
        d = json.loads(bf.read_text(encoding="utf-8"))
        rot = d.get("R_block_from_enu")
    if rot is None:
        try:
            from ..sun import block_rotation
            blk = block_rotation(rec, project)
            rot = blk["R_block_from_enu"] if blk else None
        except Exception:                                      # noqa: BLE001 - a synthetic block has no priors
            rot = None
    Rb = None if rot is None else np.asarray(rot, float)
    st = Station(label=label, site=int(rows[0]["site"]), drive=int(rows[0]["drive"]),
                 sols=sorted({int(r["sol"]) for r in rows if r.get("sol") is not None}), images=kept,
                 pivot=np.asarray(fit["pivot"], float), R_block_from_enu=Rb)
    if Rb is not None:
        st.east, st.north, st.up = Rb @ np.array([1.0, 0, 0]), Rb @ np.array([0, 1.0, 0]), Rb @ np.array([0, 0, 1.0])
    if pivot_height not in PIVOT_HEIGHTS:
        raise ValueError(f"pivot_height {pivot_height!r}: one of {PIVOT_HEIGHTS}")
    if pivot_height == "zcam":                       # 0.97.0: on the azimuth axis, at the Mastcam-Z height
        st.pivot, hrec = mastcamz_height_pivot(st.pivot, np.array(C), np.array(R), G, st.up)
        fit["pivot_fitted"] = np.asarray(fit["pivot"], float)
        fit["pivot"] = st.pivot
        fit["height"] = hrec
    st.pivot_fit = {k: (v.tolist() if isinstance(v, np.ndarray) else ({kk: vv.tolist() for kk, vv in v.items()} if isinstance(v, dict) and v and isinstance(next(iter(v.values())), np.ndarray) else v))
                    for k, v in fit.items()}
    return st
