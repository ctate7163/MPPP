"""
Pose-guided matching (MPPP v0p64): more tie points between images whose poses are already known, above all between a
Mastcam-Z frame and the Navcam (or other Mastcam-Z) frames that see the same ground.

Exhaustive SIFT matching compares every descriptor of one image with every descriptor of the other and keeps a match
only when the best candidate is clearly better than the second best (the ratio test).  Across a 3-10x difference in
pixel scale most true correspondences fail that test against the tens of thousands of unrelated features.  Once the
block is aligned (stage 2), each keypoint of image A can only match keypoints of image B that lie on its epipolar
curve, between the images of the nearest and the farthest ground A sees - a few candidates instead of thousands - so
the ratio test runs among those only.  The new matches are added (as verified two-view geometry inliers) to a copy of
the database, the block is triangulated from it and adjusted once more.

For a keypoint ray ``a`` of A (world frame, from centre ``C_A``) and a ray ``b`` of B (from ``C_B``): ``b`` is on the
epipolar plane when ``|n . b| < sin(tol)`` with ``n = a x (C_B - C_A)`` (normalised), and the two rays meet at range
``t`` along ``a`` and ``s > 0`` along ``b`` (closest approach), with ``d_min <= t <= d_max`` (A's depth range from its
triangulated points).  by default the depth window is local - for each keypoint of A, the depths of its
``local_k`` nearest triangulated keypoints of A (within ``local_radius`` of the frame diagonal), widened by
``local_factor`` either way; keypoints with fewer than ``local_min`` such neighbours are not matched
(``local_fallback`` "skip") or use the image's whole range ("global").  A window spanning the whole scene lets a
repeated texture match along the epipolar line at the wrong range (COLMAP's guided matching at Three Forks South put a
quarter of the points 10-15 x beyond the ground).  ``tol`` is ``max_px`` native pixels of either image as an angle.  Camera models of any kind
(fisheye included) enter only through their rays (``Camera.cam_ray_from_img``).
"""
from __future__ import annotations

import logging

import shutil
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.sfm")

PathLike = Union[str, Path]

GUIDED_DEFAULTS = {"families": ("Z",), "max_px": 3.0, "max_ratio": 0.85, "max_distance": 0.7, "max_pairs": 8,
                   "min_points": 20, "max_baseline_m": 50.0, "depth_margin": (0.5, 3.0),
                   # the local depth window
                   "depth": "local", "local_k": 8, "local_min": 3, "local_radius": 0.08, "local_factor": 1.4,
                   "local_fallback": "skip",
                   # 0.94.2: the pairs matched by this many threads (numpy releases the GIL in its array work); the
                   # database is written by the main thread in pair order, so the result does not depend on it.
                   # 0.94.4: 1 by default - on the PC (more cores than OpenBLAS was built for) a run stopped for good
                   # at this step (Chal Rocks tiny, 23 Oct 15:11), and the threads saved nothing on the tiny blocks
                   # (120 s either way): the time is in the pair selection and the database, not the matching
                   "workers": 1}                     # 0: min(8, CPU count)


def _pose(im) -> Tuple[np.ndarray, np.ndarray]:
    T = im.cam_from_world()
    R = np.asarray(T.rotation.matrix(), float)
    return R, -R.T @ np.asarray(T.translation, float)


def _image_points(rec, iid: int) -> np.ndarray:
    im = rec.images[iid]
    return np.array([rec.points3D[p.point3D_id].xyz for p in im.points2D if p.has_point3D()], float).reshape(-1, 3)


def _depth_range(rec, iid: int, margin=(0.5, 3.0), default=(0.5, 1000.0)) -> Tuple[float, float]:
    X = _image_points(rec, iid)
    if len(X) < 20:
        return default
    _, C = _pose(rec.images[iid])
    d = np.linalg.norm(X - C, axis=1)
    return max(0.1, margin[0] * float(np.percentile(d, 2))), min(5000.0, margin[1] * float(np.percentile(d, 98)))


def local_depths(rec, iid: int, xy: np.ndarray, k: int = 8, min_neighbours: int = 3, radius: float = 0.08,
                 factor: float = 1.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    (dmin, dmax) for each keypoint ``xy`` of image ``iid`` from the depths (distance from the camera centre)
    of its ``k`` nearest triangulated keypoints within ``radius`` x the frame diagonal: [min / factor, max x factor];
    NaN with fewer than ``min_neighbours``.
    """
    from scipy.spatial import cKDTree
    im = rec.images[iid]
    cam = rec.cameras[im.camera_id]
    n = len(xy)
    lo, hi = np.full(n, np.nan), np.full(n, np.nan)
    pts = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in im.points2D if p.has_point3D()]
    if len(pts) < min_neighbours or not n:
        return lo, hi
    uv = np.array([q[0] for q in pts], float)
    _, C = _pose(im)
    d = np.linalg.norm(np.array([q[1] for q in pts], float) - C, axis=1)
    kk = int(min(k, len(pts)))
    dist, idx = cKDTree(uv).query(np.asarray(xy, float), k=kk, distance_upper_bound=radius * float(np.hypot(cam.width, cam.height)))
    dist, idx = dist.reshape(n, kk), idx.reshape(n, kk)
    ok = np.isfinite(dist)
    cnt = ok.sum(1)
    dd = np.where(ok, d[np.minimum(idx, len(d) - 1)], np.nan)
    good = cnt >= min_neighbours
    if good.any():
        lo[good] = np.nanmin(dd[good], axis=1) / factor
        hi[good] = np.nanmax(dd[good], axis=1) * factor
    return lo, hi


def _in_frame(cam, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
    mx, my = margin * cam.width, margin * cam.height
    return (np.all(np.isfinite(uv), axis=1) & (uv[:, 0] >= -mx) & (uv[:, 0] < cam.width + mx)
            & (uv[:, 1] >= -my) & (uv[:, 1] < cam.height + my))


def _project(rec, iid: int, X: np.ndarray) -> np.ndarray:
    im = rec.images[iid]
    R, C = _pose(im)
    Xc = (X - C) @ R.T
    uv = np.full((len(X), 2), np.nan)
    ok = Xc[:, 2] > 1e-6
    if ok.any():
        uv[ok] = np.asarray(rec.cameras[im.camera_id].img_from_cam(Xc[ok]), float)
    return uv


def select_pairs(rec, project, families: Sequence[str] = ("Z",), max_pairs: int = 8, min_points: int = 20,
                 max_baseline_m: Optional[float] = 50.0, sample: int = 300, seed: int = 0) -> List[Tuple[int, int, int]]:
    """
    (A, B, overlap) for every registered image A of ``families`` and the ``max_pairs`` registered images B whose
    frame sees the most of A's triangulated points (at least ``min_points`` of a sample of ``sample``), centres at
    most ``max_baseline_m`` apart.  A pair of two such images is listed once.
    """
    rng = np.random.default_rng(seed)
    fam = {r["image_id"]: str(r["instrument"])[:1] for r in project.images if "image_id" in r}
    reg = [int(i) for i in rec.reg_image_ids()]
    cen = {i: _pose(rec.images[i])[1] for i in reg}
    out, seen = [], set()
    for a in reg:
        if fam.get(a) not in families:
            continue
        X = _image_points(rec, a)
        if len(X) < min_points:
            continue
        if len(X) > sample:
            X = X[rng.choice(len(X), sample, replace=False)]
        scores = []
        for b in reg:
            if b == a or (max_baseline_m and np.linalg.norm(cen[b] - cen[a]) > max_baseline_m):
                continue
            n = int(_in_frame(rec.cameras[rec.images[b].camera_id], _project(rec, b, X)).sum())
            if n >= min_points:
                scores.append((n, b))
        for n, b in sorted(scores, reverse=True)[:max_pairs]:
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b, n))
    return out


def _unit(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, np.float32)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.where(n > 0, n, 1.0)


def epipolar_dense(A: np.ndarray, B: np.ndarray, n: np.ndarray, tol: float, chunk: int = 2048,
                   rows: Optional[np.ndarray] = None, cols: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(i, j), sorted, with ``|n_i . B_j| < tol`` by testing every pair (of ``rows`` x ``cols`` when given).
    Memory is bounded: at most about 2**21 products at a time."""
    rows = np.arange(len(A)) if rows is None else np.asarray(rows, int)
    cols = np.arange(len(B)) if cols is None else np.asarray(cols, int)
    if not len(rows) or not len(cols):
        return np.zeros(0, int), np.zeros(0, int)
    Bc = B[cols].T
    step = max(1, min(int(chunk), (1 << 21) // max(1, len(cols))))
    I, J = [], []
    for s in range(0, len(rows), step):
        r = rows[s:s + step]
        m = n[r] @ Bc
        np.abs(m, out=m)
        ii, jj = np.nonzero(m < tol)
        if len(ii):
            I.append(r[ii])
            J.append(cols[jj])
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    i, j = np.concatenate(I), np.concatenate(J)
    o = np.lexsort((j, i))
    return i[o], j[o]


def epipolar_candidates(A: np.ndarray, B: np.ndarray, base: np.ndarray, n: np.ndarray, tol: float,
                        chunk: int = 2048, max_window: float = 0.5,
                        max_candidates: int = 1 << 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    The same (i, j) as :func:`epipolar_dense` (rays ``B_j`` within ``tol`` of the epipolar plane of ``A_i``, normal
    ``n_i``), sorted by (i, j), without testing every pair (0.94.3).  Every epipolar plane contains the baseline, so
    a ray's plane is an angle ``phi`` about it: ``|n_i . B_j| = s_j |sin(phi_j - phi_i)|`` with ``s_j`` the size of
    ``B_j`` across the baseline.  The rays of B are grouped by ``s`` (halving from 1) and sorted by ``phi``; a group
    with ``s >= s_lo`` can only hold candidates within ``asin(tol / s_lo)`` of ``phi_i`` or ``phi_i + pi``, found by
    binary search, then tested exactly.  Rays almost along the baseline (either side; window above ``max_window``)
    are tested against every ray.  At most about ``max_candidates`` pairs are gathered at once (more only when one
    row alone has more), so a thread needs a few hundred MB at most whatever the geometry.
    """
    if not len(A) or not len(B):
        return np.zeros(0, int), np.zeros(0, int)
    e = np.asarray(base, float) / np.linalg.norm(base)
    p = np.cross(e, [1.0, 0.0, 0.0] if abs(e[0]) < 0.9 else [0.0, 1.0, 0.0])
    p /= np.linalg.norm(p)
    q = np.cross(e, p)
    two_pi = 2.0 * np.pi

    def plane(X):
        x, y = X @ p, X @ q
        return np.hypot(x, y), np.mod(np.arctan2(y, x), two_pi)

    sa, fa = plane(A)
    sb, fb = plane(B)
    good_a = np.isfinite(n).all(axis=1)
    near_a = np.where(good_a & (sa < 1e-3))[0]                 # rays of A almost along the baseline
    reg_a = np.where(good_a & (sa >= 1e-3))[0]
    I, J = [], []
    if len(near_a):
        i, j = epipolar_dense(A, B, n, tol, chunk, rows=near_a)
        I.append(i)
        J.append(j)
    hi = np.inf
    s_lo = 1.0
    while True:
        last = tol / s_lo > np.sin(max_window)
        grp = np.where((sb >= (0.0 if last else s_lo)) & (sb < hi))[0]
        if len(grp) and len(reg_a):
            if last:
                i, j = epipolar_dense(A, B, n, tol, chunk, rows=reg_a, cols=grp)
            else:
                d = float(np.arcsin(min(1.0, tol / s_lo))) * 1.001 + 1e-9    # the window, with a margin
                o = np.argsort(fb[grp], kind="stable")
                S = fb[grp][o]
                S2 = np.concatenate([S, S + two_pi])
                idx2 = np.concatenate([grp[o], grp[o]])
                step = max(256, int(chunk) * 4)
                for s in range(0, len(reg_a), step):
                    r = reg_a[s:s + step]
                    win = []
                    for c in (fa[r], fa[r] + np.pi):
                        lo = np.mod(c - d, two_pi)
                        st = np.searchsorted(S2, lo, "left")
                        win.append((st, np.searchsorted(S2, lo + 2.0 * d, "right") - st))
                    per_row = win[0][1] + win[1][1]
                    # at most about max_candidates gathered at once (memory per thread; 0.94.3)
                    cut = np.searchsorted(np.cumsum(per_row), np.arange(1, 1 + int(per_row.sum()) // max_candidates)
                                          * max_candidates, "left")
                    bounds = np.unique(np.r_[0, np.minimum(cut + 1, len(r)), len(r)])
                    for r0, r1 in zip(bounds[:-1], bounds[1:]):
                        cand_i, cand_j = [], []
                        for st, cnt in win:
                            st, cnt = st[r0:r1], cnt[r0:r1]
                            tot = int(cnt.sum())
                            if not tot:
                                continue
                            rep = np.repeat(np.arange(r1 - r0), cnt)
                            off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                            cand_i.append(r[r0:r1][rep])
                            cand_j.append(idx2[st[rep] + off])
                        if not cand_i:
                            continue
                        ci, cj = np.concatenate(cand_i), np.concatenate(cand_j)
                        keep = np.abs(np.einsum("ij,ij->i", n[ci], B[cj])) < tol
                        I.append(ci[keep])
                        J.append(cj[keep])
                i = j = None
            if i is not None:
                I.append(i)
                J.append(j)
        if last:
            break
        hi, s_lo = s_lo, s_lo / 2.0
    I = [x for x in I if len(x)]
    J = [x for x in J if len(x)]
    if not I:
        return np.zeros(0, int), np.zeros(0, int)
    i, j = np.concatenate(I), np.concatenate(J)
    o = np.lexsort((j, i))
    return i[o], j[o]


def guided_pair_matches(rec, a: int, b: int, desc_a: np.ndarray, desc_b: np.ndarray, scale_a: float = 1.0,
                        scale_b: float = 1.0, depth: Optional[Tuple[float, float]] = None, max_px: float = 3.0,
                        max_ratio: float = 0.85, max_distance: float = 0.7, chunk: int = 2048,
                        local: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Keypoint index pairs (k in A, l in B) of one image pair with known poses (see the module docstring): epipolar
    and depth-range candidates, then the ratio test (descriptor angles, best / second best < ``max_ratio``; a single
    candidate must be closer than ``max_distance`` rad) and a cross check, both among the candidates only.
    ``scale_*``: the images' downsample scale (native pixels = full-frame pixels x scale).  ``local``:
    {"k", "min", "radius", "factor", "fallback"} - a depth window per keypoint of A (:func:`local_depths`), inside
    ``depth``; None uses ``depth`` for every keypoint.
    """
    ia, ib = rec.images[a], rec.images[b]
    ca, cb = rec.cameras[ia.camera_id], rec.cameras[ib.camera_id]
    Ra, Ca = _pose(ia)
    Rb, Cb = _pose(ib)
    base = Cb - Ca
    if np.linalg.norm(base) < 0.01:
        return np.zeros((0, 2), int)
    ka = np.array([p.xy for p in ia.points2D], float).reshape(-1, 2)
    kb = np.array([p.xy for p in ib.points2D], float).reshape(-1, 2)
    if not len(ka) or not len(kb):
        return np.zeros((0, 2), int)
    dmin, dmax = depth or _depth_range(rec, a)
    ra = np.asarray(ca.cam_ray_from_img(ka), float) @ Ra          # world rays (rows): R^T r
    rb = np.asarray(cb.cam_ray_from_img(kb), float) @ Rb
    lo_a, hi_a = np.full(len(ka), float(dmin)), np.full(len(ka), float(dmax))
    if local is not None:                                          # the depth window of each keypoint
        lo, hi = local_depths(rec, a, ka, int(local.get("k", 8)), int(local.get("min", 3)),
                              float(local.get("radius", 0.08)), float(local.get("factor", 1.4)))
        has = np.isfinite(lo)
        lo_a[has] = np.clip(lo[has], dmin, dmax)
        hi_a[has] = np.clip(hi[has], dmin, dmax)
        if str(local.get("fallback", "skip")) == "skip":
            lo_a[~has], hi_a[~has] = np.inf, -np.inf               # no candidates
    # only keypoints whose rays can reach the other frame (at the near, middle and far depth)
    sel_a = np.zeros(len(ka), bool)
    sel_b = np.zeros(len(kb), bool)
    live = hi_a >= lo_a
    dm_a = np.sqrt(np.where(live, lo_a * hi_a, 1.0))
    for da_ in (lo_a, dm_a, hi_a):
        sel_a |= live & _in_frame(cb, _project(rec, b, Ca + np.where(live, da_, 1.0)[:, None] * ra), 0.02)
    dm = float(np.sqrt(dmin * dmax))
    for d in (dmin, dm, dmax):
        sel_b |= _in_frame(ca, _project(rec, a, Cb + d * rb), 0.02)
    ia_idx, ib_idx = np.where(sel_a)[0], np.where(sel_b)[0]
    if not len(ia_idx) or not len(ib_idx):
        return np.zeros((0, 2), int)
    fa = float(np.sqrt(ca.params[0] * ca.params[1])) if len(ca.params) > 1 else float(ca.params[0])
    fb = float(np.sqrt(cb.params[0] * cb.params[1])) if len(cb.params) > 1 else float(cb.params[0])
    tol = np.sin(max_px / max(scale_a, 1e-6) / fa + max_px / max(scale_b, 1e-6) / fb)
    A, B = ra[ia_idx], rb[ib_idx]
    LO, HI = lo_a[ia_idx], hi_a[ia_idx]
    n = np.cross(A, base)
    with np.errstate(invalid="ignore", divide="ignore"):
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    da, db = _unit(desc_a[ia_idx]), _unit(desc_b[ib_idx])
    w = Ca - Cb
    i, j = epipolar_candidates(A, B, base, n, tol, chunk=chunk)        # 0.94.3: sorted windows, not every pair
    if len(i):
        ok = np.zeros(len(i), bool)
        for s in range(0, len(i), 1 << 20):                              # bounded memory per thread (0.94.3)
            e = slice(s, s + (1 << 20))
            a_, b_ = A[i[e]], B[j[e]]
            c = np.einsum("ij,ij->i", a_, b_)
            d1, e1 = a_ @ w, b_ @ w
            den = np.maximum(1.0 - c * c, 1e-12)
            t = (c * e1 - d1) / den                                      # range along A's ray
            u = (e1 - c * d1) / den                                      # along B's ray
            ok[e] = (t >= LO[i[e]]) & (t <= HI[i[e]]) & (u > 0)
        i, j = i[ok], j[ok]
    if not len(i):
        return np.zeros((0, 2), int)
    g = np.empty(len(i))
    for s in range(0, len(i), 1 << 16):                                  # descriptor angles, bounded memory
        e = slice(s, s + (1 << 16))
        g[e] = np.arccos(np.clip(np.einsum("ij,ij->i", da[i[e]], db[j[e]]), -1.0, 1.0))

    def best_two(key, other, g):
        o = np.lexsort((g, key))
        k, v, gg = key[o], other[o], g[o]
        first = np.r_[True, k[1:] != k[:-1]]
        idx = np.where(first)[0]
        best_v, best_g = v[idx], gg[idx]
        nxt = idx + 1
        has2 = (nxt < len(k)) & (np.r_[k[1:], -1][idx] == k[idx])
        second = np.where(has2, gg[np.minimum(nxt, len(k) - 1)], np.inf)
        return k[idx], best_v, best_g, second

    ka_u, kb_best, g_best, g_second = best_two(i, j, g)
    good = (g_best < max_distance) & (g_best < max_ratio * g_second)
    kb_u, ka_best, _, _ = best_two(j, i, g)
    back = dict(zip(kb_u.tolist(), ka_best.tolist()))
    m = [(int(ia_idx[x]), int(ib_idx[y])) for x, y, ok in zip(ka_u, kb_best, good) if ok and back.get(int(y)) == int(x)]
    return np.array(m, int).reshape(-1, 2)


def _family_ties(rec, project) -> Dict[str, int]:
    fam = {r["image_id"]: str(r["instrument"])[:1] for r in project.images if "image_id" in r}
    nz = 0
    for pt in rec.points3D.values():
        f = {fam.get(el.image_id) for el in pt.track.elements}
        if "N" in f and "Z" in f:
            nz += 1
    return {"points": int(len(rec.points3D)), "navcam_zcam_points": nz}


def pose_guided_matching(rec, project, database: PathLike, out_database: Optional[PathLike] = None,
                         verbose: bool = True, **opts) -> Tuple[Path, Dict[str, Any]]:
    """
    Pose-guided matches for the pairs of :func:`select_pairs`, added to a copy of ``database`` (default
    ``<project>/database_guided.db``) as two-view geometry inliers (merged with the pair's existing inliers).  ``opts``
    override :data:`GUIDED_DEFAULTS`.  Returns (the copy, a report: pairs, new matches, per-family counts, seconds).
    """
    import pycolmap
    o = dict(GUIDED_DEFAULTS, **opts)
    t0 = time.time()
    out = Path(out_database) if out_database else project.root / "database_guided.db"
    if Path(database).resolve() != out.resolve():
        if out.exists():
            try:
                out.unlink()
            except PermissionError:                    # 0.94.4: held by another process (a run that hung)
                alt = out.with_name(f"{out.stem}_{os.getpid()}{out.suffix}")
                log.warning("pose-guided matching: %s is in use by another process; writing %s instead", out.name,
                            alt.name)
                out = alt
        from .database import copy_database
        copy_database(database, out, descriptors=False)        # 0.95.0: the descriptors are read from the original
    pairs = select_pairs(rec, project, tuple(o["families"]), int(o["max_pairs"]), int(o["min_points"]),
                         o.get("max_baseline_m"))
    scale = {r["image_id"]: float(r.get("downsample_scale", 1.0)) for r in project.images if "image_id" in r}
    fam = {r["image_id"]: str(r["instrument"])[:1] for r in project.images if "image_id" in r}
    db = pycolmap.Database.open(str(out))
    # 0.95.0: the descriptors from the project database (the copies carry none); the one of ``database`` when it is
    # the project's own, else the original it was copied from
    ddb_path = Path(getattr(project, "database", database))
    ddb = db if ddb_path.resolve() == out.resolve() else pycolmap.Database.open(str(ddb_path))
    if o.get("depth", "local") not in ("local", "global"):
        raise ValueError("pose-guided matching: depth must be 'local' or 'global'")
    local = ({"k": o["local_k"], "min": o["local_min"], "radius": o["local_radius"], "factor": o["local_factor"],
              "fallback": o["local_fallback"]} if o.get("depth", "local") == "local" else None)
    desc: Dict[int, np.ndarray] = {}
    depth: Dict[int, Tuple[float, float]] = {}
    n_new, by_fam, n_pairs_used, n_nodesc = 0, {}, 0, 0
    workers = int(o.get("workers") or 0) or min(8, os.cpu_count() or 1)
    try:
        todo = []
        for a, b, _ in pairs:                                   # descriptors and depth ranges in the main thread
            for i in (a, b):
                if i not in desc:
                    d = ddb.read_descriptors(int(i))
                    if not len(getattr(d, "data", d)) and ddb is not db:        # (a copy that still has them)
                        d = db.read_descriptors(int(i))
                    desc[i] = np.asarray(getattr(d, "data", d), np.float32)        # pycolmap 4.2: FeatureDescriptors
            if (len(desc[a]) != rec.images[a].num_points2D() or len(desc[b]) != rec.images[b].num_points2D()):
                n_nodesc += 1                                          # descriptors missing (or another feature set)
                continue
            if a not in depth:
                depth[a] = _depth_range(rec, a, tuple(o["depth_margin"]))
            todo.append((a, b))

        def one(ab):
            a_, b_ = ab
            return guided_pair_matches(rec, a_, b_, desc[a_], desc[b_], scale.get(a_, 1.0), scale.get(b_, 1.0),
                                       depth[a_], float(o["max_px"]), float(o["max_ratio"]), float(o["max_distance"]),
                                       local=local)
        if workers > 1 and len(todo) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = ex.map(one, todo)                      # in pair order
                found = list(zip(todo, results))
        else:
            found = [(ab, one(ab)) for ab in todo]
        for (a, b), m in found:
            if not len(m):
                continue
            i1, i2 = (a, b) if a < b else (b, a)
            mm = m if a < b else m[:, ::-1]
            if db.exists_two_view_geometry(int(i1), int(i2)):
                tvg = db.read_two_view_geometry(int(i1), int(i2))
                old = np.asarray(tvg.inlier_matches, int).reshape(-1, 2)
                db.delete_two_view_geometry(int(i1), int(i2))
            else:
                tvg = pycolmap.TwoViewGeometry()
                tvg.config = 2                                  # calibrated
                old = np.zeros((0, 2), int)
            # one match per keypoint of either image: the existing inliers win
            used1, used2 = set(old[:, 0].tolist()), set(old[:, 1].tolist())
            add = np.array([r for r in mm.tolist() if r[0] not in used1 and r[1] not in used2], int).reshape(-1, 2)
            tvg.inlier_matches = np.vstack([old, add]).astype(np.uint32)
            db.write_two_view_geometry(int(i1), int(i2), tvg)
            n_new += len(add)
            n_pairs_used += int(len(add) > 0)
            k = "".join(sorted(fam.get(a, "?") + fam.get(b, "?")))
            by_fam[k] = by_fam.get(k, 0) + int(len(add))
    finally:
        db.close()
        if ddb is not db:
            ddb.close()
    rep = {"database": str(out), "pairs": len(pairs), "pairs_with_new_matches": n_pairs_used, "new_matches": int(n_new),
           "pairs_without_descriptors": n_nodesc,
           "new_matches_by_family": by_fam, "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()},
           "seconds": round(time.time() - t0, 1), "workers": workers}
    if verbose:
        log.info(f"pose-guided matching: {len(pairs)} pairs, {n_new} new matches ({by_fam}) in {n_pairs_used} pairs, "
              f"{rep['seconds']:.0f} s -> {out.name}")
    return out, rep
