"""
mppp.ibr.rover_model - the Perseverance model in the station mesh (0.96.1, Christian 10 Nov 2026: "take this model
of perseverance's, simplify it, add it to the mesh, and project the images onto it. The arm is stowed").

``mppp/data/rover/perseverance_rmech.npz`` is NASA's glTF model (credit NASA/JPL-Caltech,
https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/) simplified to 59,000 triangles in the rover
mechanical frame RMECH (x forward, y right, z down, z 0 = the deck top; see :mod:`mppp.ibr.rover_deck`), built by
``scripts/build_rover_model.py``: body, suspension and wheels as modelled, the arm turned to its stowed pose (the
five joint angles fitted to the Van Zyl S003D2046 cloud), without what moves from image to image - the mast and
head, the high-gain antenna.  Against the Van Zyl rover cloud (placed in RMECH by the label CAHVORE poses) 88 % of
the cloud is within 3 cm of the model and 94 % within 5 cm.

With the rover shown, ``mppp mesh`` then
1. drops the fused points within ``NEAR_M`` of the model (the model stands for them) - what is farther stays and is
   meshed as before (the antenna, cables, anything the model lacks);
2. adds the model's triangles, placed in the block by the station's rover pose (:func:`mppp.ibr.rover_deck.rover_pose`),
   to ``mesh.ply``.
The images are then projected onto it like onto any other surface of the mesh.  The suspension and the wheels are in
the model's level-ground pose: on rough ground they are off by the rocker-bogie's articulation (``parts`` to leave
them out).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

FILE = Path(__file__).resolve().parents[1] / "data" / "rover" / "perseverance_rmech.npz"
NEAR_M = 0.04                     # fused points this close to the model are dropped
SAMPLE_SPACING_M = 0.012          # the model's surface sampled about this densely for the distance test
_cache: Dict[str, Any] = {}


def available() -> bool:
    return FILE.is_file()


def load(parts: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(vertices (n, 3) in RMECH metres, faces (m, 3), meta) of the model, or of the named ``parts`` of it (body,
    arm, arm_base, suspension, wheels)."""
    if "raw" not in _cache:
        d = np.load(FILE)
        _cache["raw"] = (d["vertices"].astype(float), d["faces"].astype(np.int64), d["part"], json.loads(str(d["meta"])))
    V, F, P, meta = _cache["raw"]
    if parts is not None:
        ids = [meta["parts"].index(p) for p in parts]
        F = F[np.isin(P, ids)]
    return V, F, meta


def surface_samples(parts: Optional[Sequence[str]] = None, spacing_m: float = SAMPLE_SPACING_M) -> np.ndarray:
    """Points on the model's surface about ``spacing_m`` apart (RMECH)."""
    key = ("samples", tuple(parts) if parts else None, spacing_m)
    if key not in _cache:
        V, F, _ = load(parts)
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        n = int(max(1000, area.sum() / spacing_m ** 2))
        rng = np.random.default_rng(0)
        f = rng.choice(len(F), size=n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        _cache[key] = a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f])
    return _cache[key]


def near_model(points: np.ndarray, pose: Dict[str, Any], near_m: float = NEAR_M, parts: Optional[Sequence[str]] = None) -> np.ndarray:
    """The block-frame points within ``near_m`` of the model's surface."""
    from scipy.spatial import cKDTree
    from .rover_deck import block_to_mech
    M = block_to_mech(points, pose)
    S = surface_samples(parts)
    lo, hi = S.min(axis=0) - near_m, S.max(axis=0) + near_m
    box = np.all((M >= lo) & (M <= hi), axis=1)
    out = np.zeros(len(M), bool)
    if box.any():
        d, _ = cKDTree(S).query(M[box], distance_upper_bound=near_m + SAMPLE_SPACING_M)
        out[np.flatnonzero(box)] = d <= near_m + 0.5 * SAMPLE_SPACING_M
    return out


def model_in_block(pose: Dict[str, Any], parts: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices in the block frame, faces) of the model at the station's rover pose (only the vertices in use)."""
    from .rover_deck import mech_to_block
    V, F, _ = load(parts)
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return mech_to_block(V[used], pose), remap[F]
