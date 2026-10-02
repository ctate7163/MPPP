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
triangulated points).  ``tol`` is ``max_px`` native pixels of either image as an angle.  Camera models of any kind
(fisheye included) enter only through their rays (``Camera.cam_ray_from_img``).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

GUIDED_DEFAULTS = {"families": ("Z",), "max_px": 3.0, "max_ratio": 0.85, "max_distance": 0.7, "max_pairs": 8,
                   "min_points": 20, "max_baseline_m": 50.0, "depth_margin": (0.5, 3.0)}


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


def guided_pair_matches(rec, a: int, b: int, desc_a: np.ndarray, desc_b: np.ndarray, scale_a: float = 1.0,
                        scale_b: float = 1.0, depth: Optional[Tuple[float, float]] = None, max_px: float = 3.0,
                        max_ratio: float = 0.85, max_distance: float = 0.7, chunk: int = 2048) -> np.ndarray:
    """
    Keypoint index pairs (k in A, l in B) of one image pair with known poses (see the module docstring): epipolar
    and depth-range candidates, then the ratio test (descriptor angles, best / second best < ``max_ratio``; a single
    candidate must be closer than ``max_distance`` rad) and a cross check, both among the candidates only.
    ``scale_*``: the images' downsample scale (native pixels = full-frame pixels x scale).
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
    # only keypoints whose rays can reach the other frame (at the near, middle and far depth)
    dm = float(np.sqrt(dmin * dmax))
    sel_a = np.zeros(len(ka), bool)
    sel_b = np.zeros(len(kb), bool)
    for d in (dmin, dm, dmax):
        sel_a |= _in_frame(cb, _project(rec, b, Ca + d * ra), 0.02)
        sel_b |= _in_frame(ca, _project(rec, a, Cb + d * rb), 0.02)
    ia_idx, ib_idx = np.where(sel_a)[0], np.where(sel_b)[0]
    if not len(ia_idx) or not len(ib_idx):
        return np.zeros((0, 2), int)
    fa = float(np.sqrt(ca.params[0] * ca.params[1])) if len(ca.params) > 1 else float(ca.params[0])
    fb = float(np.sqrt(cb.params[0] * cb.params[1])) if len(cb.params) > 1 else float(cb.params[0])
    tol = np.sin(max_px / max(scale_a, 1e-6) / fa + max_px / max(scale_b, 1e-6) / fb)
    A, B = ra[ia_idx], rb[ib_idx]
    n = np.cross(A, base)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    da, db = _unit(desc_a[ia_idx]), _unit(desc_b[ib_idx])
    w = Ca - Cb
    rows, cols, ang = [], [], []
    for s in range(0, len(A), chunk):
        e = slice(s, s + chunk)
        on = np.abs(n[e] @ B.T) < tol                                   # on the epipolar plane
        i, j = np.nonzero(on)
        if not len(i):
            continue
        a_, b_ = A[e][i], B[j]
        c = np.einsum("ij,ij->i", a_, b_)
        d1, e1 = a_ @ w, b_ @ w
        den = np.maximum(1.0 - c * c, 1e-12)
        t = (c * e1 - d1) / den                                          # range along A's ray
        u = (e1 - c * d1) / den                                          # along B's ray
        ok = (t >= dmin) & (t <= dmax) & (u > 0)
        i, j = i[ok], j[ok]
        if not len(i):
            continue
        cosd = np.clip(np.einsum("ij,ij->i", da[e][i], db[j]), -1.0, 1.0)
        rows.append(i + s)
        cols.append(j)
        ang.append(np.arccos(cosd))
    if not rows:
        return np.zeros((0, 2), int)
    i, j, g = np.concatenate(rows), np.concatenate(cols), np.concatenate(ang)

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
            out.unlink()
        shutil.copyfile(database, out)
    pairs = select_pairs(rec, project, tuple(o["families"]), int(o["max_pairs"]), int(o["min_points"]),
                         o.get("max_baseline_m"))
    scale = {r["image_id"]: float(r.get("downsample_scale", 1.0)) for r in project.images if "image_id" in r}
    fam = {r["image_id"]: str(r["instrument"])[:1] for r in project.images if "image_id" in r}
    db = pycolmap.Database.open(str(out))
    desc: Dict[int, np.ndarray] = {}
    depth: Dict[int, Tuple[float, float]] = {}
    n_new, by_fam, n_pairs_used, n_nodesc = 0, {}, 0, 0
    try:
        for a, b, _ in pairs:
            for i in (a, b):
                if i not in desc:
                    d = db.read_descriptors(int(i))
                    desc[i] = np.asarray(getattr(d, "data", d), np.float32)        # pycolmap 4.2: FeatureDescriptors
            if (len(desc[a]) != rec.images[a].num_points2D() or len(desc[b]) != rec.images[b].num_points2D()):
                n_nodesc += 1                                          # descriptors missing (or another feature set)
                continue
            if a not in depth:
                depth[a] = _depth_range(rec, a, tuple(o["depth_margin"]))
            m = guided_pair_matches(rec, a, b, desc[a], desc[b], scale.get(a, 1.0), scale.get(b, 1.0), depth[a],
                                    float(o["max_px"]), float(o["max_ratio"]), float(o["max_distance"]))
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
    rep = {"database": str(out), "pairs": len(pairs), "pairs_with_new_matches": n_pairs_used, "new_matches": int(n_new),
           "pairs_without_descriptors": n_nodesc,
           "new_matches_by_family": by_fam, "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()},
           "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"[sfm] pose-guided matching: {len(pairs)} pairs, {n_new} new matches ({by_fam}) in {n_pairs_used} pairs, "
              f"{rep['seconds']:.0f} s -> {out.name}", flush=True)
    return out, rep
