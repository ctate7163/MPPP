"""
Feature matching on ``database.db`` (full-resolution keypoints).

``mode="lightglue"`` (optional; the pipeline's default is COLMAP's exhaustive SIFT matcher since 0.91) matches COLMAP's SIFT features with LightGlue
(:mod:`mppp.sfm.lightglue`), all pairs, or only ``pairs`` when given.  ``mode="exhaustive"`` is COLMAP's own
nearest-neighbour SIFT matching of all pairs (ratio test, cross check; use a GPU); ``mode="prior_pairs"`` matches
only the pairs from :func:`mppp.sfm.prior_overlap_pairs` with it - for CPU runs and quick tests; it biases
completeness statistics towards pairs the priors expected to overlap, so do not use it for survival / theta_max
analysis.

Geometric verification runs in full-resolution pixels: ``max_error_px`` = 6
is 1.5 px at quarter resolution, 3 px at half and 6 px at full resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from .project import SfmProject

import logging

log = logging.getLogger("mppp.sfm")
PathLike = Union[str, Path]


def match(project: SfmProject, mode: str = "exhaustive", pairs: Optional[Sequence[Tuple[str, str, Any]]] = None,
          use_gpu: Optional[bool] = None, max_num_matches: int = 32768, max_error_px: float = 6.0,
          min_num_inliers: int = 15, num_threads: int = -1, guided_matching: bool = False, block_size: int = 100,
          python: Optional[PathLike] = None, max_ratio: float = 0.8, max_distance: float = 0.7,
          cross_check: bool = True, max_keypoints: int = 8192, filter_threshold: float = 0.1,
          device: Optional[str] = None, max_hours: float = 3.0,
          keypoint_selection: str = "largest", uncalibrated_families: Sequence[str] = ()) -> Dict[str, Any]:
    """Match and verify; results go into ``project.database``.  Returns a summary.
    ``python``: run in another Python environment, e.g. a conda one with a CUDA
    pycolmap (v0p11, see :mod:`mppp.sfm.gpu`).  ``max_ratio``, ``max_distance``,
    ``cross_check``: SIFT matching (Lowe ratio test, descriptor distance,
    mutual best match); ``guided_matching``: a second pass guided by the
    verified two-view geometry; ``max_error_px``: RANSAC threshold of the
    geometric verification, full-resolution pixels.  ``max_keypoints``, ``filter_threshold``, ``device``: LightGlue
    (the keypoints per image it attends over, its match confidence threshold, "cuda"/"cpu"; see
    :mod:`mppp.sfm.lightglue`); ``max_hours``: the LightGlue time budget per block (3 h since 0.94.4, Christian; was 2 h), projected from the first pairs
    - beyond it, or without kornia/torch, the block is matched with COLMAP's SIFT matcher and the record says so
    (``matcher`` against ``matcher_requested``), so that a later run with LightGlue available does not reuse
    SIFT matches as LightGlue ones.

    ``mode="sift+lightglue"`` (0.94.4): COLMAP's SIFT matcher (every pair, or ``pairs``) as "exhaustive", then
    LightGlue over the same pairs, verified on their own, its inliers added to each pair's SIFT inliers (the SIFT
    ones win where both use a keypoint) - SIFT's many precise matches (above all Navcam - Mastcam-Z) and LightGlue's
    wide-baseline (cross-station) ones.  ``keypoint_selection``: which ``max_keypoints`` keypoints LightGlue sees
    ("largest" scales, or "octaves": a share of every octave; :func:`mppp.sfm.lightglue.select_keypoints`).

    ``uncalibrated_families`` (0.95.0, Christian, 28 Oct): the camera families (``"Z"``: Mastcam-Z) whose pairs are
    verified uncalibrated from the start - their database cameras lose ``has_prior_focal_length`` for the matching
    (:func:`set_prior_focal`), so COLMAP verifies with a fundamental matrix instead of the calibrated (ray)
    estimators, which abort on a Mastcam-Z pair whose keypoints fail to unproject (the retry below) after a pass
    that is lost.  Recorded as ``uncalibrated`` in the matches record (a rematch when it changes)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    fams = tuple(str(f) for f in (uncalibrated_families or ()))
    n_unc = set_prior_focal(project, families=fams, value=False) if fams else 0
    try:
        out = _match(project, mode, pairs, use_gpu, max_num_matches, max_error_px, min_num_inliers, num_threads,
                     guided_matching, block_size, python, max_ratio, max_distance, cross_check, max_keypoints,
                     filter_threshold, device, max_hours, keypoint_selection, fams)
    finally:
        if fams:
            set_prior_focal(project, families=fams, value=True)
    if fams:
        out["uncalibrated"] = list(fams)
        out["uncalibrated_cameras"] = int(n_unc)
        project.settings["matching"] = dict(project.settings.get("matching") or {}, uncalibrated=list(fams),
                                            uncalibrated_cameras=int(n_unc))
        project.save()
    return out


def _match(project: SfmProject, mode: str, pairs, use_gpu, max_num_matches: int, max_error_px: float,
           min_num_inliers: int, num_threads: int, guided_matching: bool, block_size: int, python, max_ratio: float,
           max_distance: float, cross_check: bool, max_keypoints: int, filter_threshold: float, device,
           max_hours: float, keypoint_selection: str, fams: Tuple[str, ...] = ()) -> Dict[str, Any]:
    sel = {"keypoint_selection": keypoint_selection} if keypoint_selection != "largest" else {}
    unc = {"uncalibrated": list(fams)} if fams else {}
    if mode == "sift+lightglue":
        sift_mode = "exhaustive" if pairs is None else "prior_pairs"
        sift = _match(project, sift_mode, pairs, use_gpu, max_num_matches, max_error_px, min_num_inliers, num_threads,
                      guided_matching, block_size, python, max_ratio, max_distance, cross_check, max_keypoints,
                      filter_threshold, device, max_hours, keypoint_selection, fams)
        add = add_lightglue(project, pairs, max_keypoints, filter_threshold, device, max_error_px, min_num_inliers,
                            max_hours, keypoint_selection)
        return _finish(project, mode, max_error_px, min_num_inliers, max_num_matches, guided_matching, max_ratio,
                       max_distance, cross_check,
                       extra={"sift": {k: sift.get(k) for k in ("matcher", "verified_pairs", "inlier_matches")},
                              "lightglue": add, "max_keypoints": int(max_keypoints),
                              "filter_threshold": float(filter_threshold), "matcher_requested": "sift+lightglue",
                              "matcher": "sift+lightglue" if add else "sift", **sel, **unc})
    if mode == "lightglue":
        lg = _try_lightglue(project, pairs, max_keypoints, filter_threshold, device, max_error_px, min_num_inliers,
                            max_hours, keypoint_selection)
        if lg is not None:
            return _finish(project, mode, max_error_px, min_num_inliers, max_num_matches, guided_matching, max_ratio,
                           max_distance, cross_check, extra={"lightglue": lg, "max_keypoints": int(max_keypoints),
                                                             "filter_threshold": float(filter_threshold),
                                                             "matcher_requested": "lightglue", "matcher": "lightglue",
                                                             **sel, **unc})
        # LightGlue unavailable or too slow here: COLMAP's SIFT matcher of the same pairs, recorded as such
        requested, mode = "lightglue", ("exhaustive" if pairs is None else "prior_pairs")
    else:
        requested = mode
    if python is not None:
        from .gpu import run_step
        if mode == "prior_pairs" and pairs is not None:
            (project.root / "pairs_prior.txt").write_text("".join(f"{p[0]} {p[1]}\n" for p in pairs), encoding="utf-8")
        kw = dict(mode=mode, use_gpu=use_gpu, max_num_matches=max_num_matches,
                  max_error_px=max_error_px, min_num_inliers=min_num_inliers, num_threads=num_threads,
                  guided_matching=guided_matching, max_ratio=max_ratio, max_distance=max_distance,
                  cross_check=cross_check, block_size=block_size, uncalibrated_families=list(fams))
        try:
            return run_step(python, "match", project, **kw)
        except RuntimeError as e:
            if not any(k in str(e) for k in CALIBRATED_ABORT_SIGNS):
                raise
            # COLMAP 4.2 aborted in the calibrated two-view verification (Rockytop N+Z34, 3 Oct: "Check
            # failed: cam_rays1_with_jac.size() >= kMinNumSamples (4 vs. 5)" - a pair whose keypoints mostly fail to
            # unproject).  The Mastcam-Z cameras then lose their prior focal length in the database for one more run,
            # so their pairs are verified uncalibrated (fundamental matrix, no rays); the pairs already matched are kept
            n = set_prior_focal(project, families=("Z",), value=False)
            log.warning("matching aborted in COLMAP's calibrated verification; again with the %d Mastcam-Z cameras "
                        "verified uncalibrated (prior focal length off)", n)
            try:
                out = run_step(python, "match", project, **kw)
            finally:
                set_prior_focal(project, families=("Z",), value=True)
            out = dict(out) if isinstance(out, dict) else {"result": out}
            out["retried_uncalibrated_zcam"] = n
            return out
    import pycolmap
    mo = pycolmap.FeatureMatchingOptions()
    mo.max_num_matches = int(max_num_matches)
    from .reconstruction import default_threads
    mo.num_threads = int(num_threads) if int(num_threads) > 0 else default_threads()     # 0.95.0: MPPP_THREADS
    mo.guided_matching = bool(guided_matching)
    mo.sift.max_ratio = float(max_ratio)
    mo.sift.max_distance = float(max_distance)
    mo.sift.cross_check = bool(cross_check)
    vo = pycolmap.TwoViewGeometryOptions()
    vo.ransac.max_error = float(max_error_px)
    vo.min_num_inliers = int(min_num_inliers)
    if use_gpu and not getattr(pycolmap, "has_cuda", False):
        import warnings
        warnings.warn("this pycolmap build has no CUDA; running on the CPU (or use the COLMAP GUI/CLI with a GPU)")
        use_gpu = False
    device = pycolmap.Device.auto if use_gpu is None else (pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu)
    db = str(project.database)
    if mode == "exhaustive":
        # block_size images per block (COLMAP default 50): a block's descriptors are loaded once and every
        # pair inside it matched, so larger blocks mean fewer reloads and fewer GPU pipeline drains on a block
        # boundary; the verification of a block runs on num_threads CPU threads while the GPU matches the next
        po = pycolmap.ExhaustivePairingOptions()
        po.block_size = int(block_size)
        pycolmap.match_exhaustive(db, matching_options=mo, pairing_options=po, verification_options=vo, device=device)
    else:
        if pairs is None:
            f = project.root / "pairs_prior.txt"
            if not f.is_file():
                raise FileNotFoundError("no pairs: run prior_overlap_pairs(project) first")
        else:
            f = project.root / "pairs_prior.txt"
            f.write_text("".join(f"{p[0]} {p[1]}\n" for p in pairs), encoding="utf-8")
        po = pycolmap.ImportedPairingOptions()
        po.match_list_path = str(f)
        pycolmap.match_image_pairs(db, matching_options=mo, pairing_options=po, verification_options=vo,
                                   device=device)
    return _finish(project, mode, max_error_px, min_num_inliers, max_num_matches, guided_matching, max_ratio,
                   max_distance, cross_check, extra={"matcher_requested": requested, "matcher": "sift", **unc})


MODES = ("lightglue", "exhaustive", "prior_pairs", "sift+lightglue")


def add_lightglue(project: SfmProject, pairs, max_keypoints: int, filter_threshold: float, device: Optional[str],
                  max_error_px: float, min_num_inliers: int, max_hours: float,
                  keypoint_selection: str = "largest") -> Optional[Dict[str, Any]]:
    """LightGlue's verified inliers added to the SIFT inliers already in ``project.database`` (0.94.4): LightGlue
    matches and verifies in a copy of the database without matches, then :func:`merge_inliers`.  None (the SIFT
    matches alone, logged) when LightGlue cannot run."""
    import shutil
    import pycolmap
    from types import SimpleNamespace
    tmp = Path(project.root) / "database_lightglue_add.db"
    if tmp.exists():
        tmp.unlink()
    shutil.copyfile(project.database, tmp)
    try:
        d = pycolmap.Database.open(str(tmp))
        try:
            d.clear_two_view_geometries()
            d.clear_matches()
        finally:
            d.close()
        lg = _try_lightglue(SimpleNamespace(root=Path(project.root), database=tmp), pairs, max_keypoints,
                            filter_threshold, device, max_error_px, min_num_inliers, max_hours, keypoint_selection)
        if lg is None:
            log.warning("sift+lightglue: the SIFT matches alone")
            return None
        lg = dict(lg, **merge_inliers(tmp, project.database))
        log.info("sift+lightglue: %d LightGlue inliers added (%d pairs, %d of them new), %d left out for keypoints "
                 "the SIFT inliers already use", lg["added_inliers"], lg["pairs_added_to"], lg["new_pairs"],
                 lg["conflicts"])
        return lg
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def merge_inliers(src: PathLike, dst: PathLike) -> Dict[str, int]:
    """The two-view inliers of database ``src`` added to those of ``dst`` pair by pair: a match whose keypoint (in
    either image) an inlier of ``dst`` already uses is left out; a pair ``dst`` lacks gets ``src``'s geometry."""
    import pycolmap
    s = pycolmap.Database.open(str(src))
    try:
        ids, geoms = s.read_two_view_geometries()
        found = [(pycolmap.pair_id_to_image_pair(int(pid)), g) for pid, g in zip(ids, geoms)]
    finally:
        s.close()
    d = pycolmap.Database.open(str(dst))
    added = pairs = new = conflicts = 0
    try:
        for (i1, i2), g in found:
            i1, i2 = int(i1), int(i2)
            m = np.asarray(g.inlier_matches, np.int64).reshape(-1, 2)
            if not len(m):
                continue
            if d.exists_two_view_geometry(i1, i2):
                tvg = d.read_two_view_geometry(i1, i2)
                old = np.asarray(tvg.inlier_matches, np.int64).reshape(-1, 2)
                used1, used2 = set(old[:, 0].tolist()), set(old[:, 1].tolist())
                ok = np.array([a not in used1 and b not in used2 for a, b in m.tolist()], bool)
                if not ok.any():
                    conflicts += len(m)
                    continue
                d.delete_two_view_geometry(i1, i2)
                tvg.inlier_matches = np.vstack([old, m[ok]]).astype(np.uint32)
                d.write_two_view_geometry(i1, i2, tvg)
                conflicts += int((~ok).sum())
                added += int(ok.sum())
            else:
                d.write_two_view_geometry(i1, i2, g)
                added += len(m)
                new += 1
            pairs += 1
    finally:
        d.close()
    return {"added_inliers": added, "pairs_added_to": pairs, "new_pairs": new, "conflicts": conflicts}


def _try_lightglue(project: SfmProject, pairs, max_keypoints: int, filter_threshold: float, device: Optional[str],
                   max_error_px: float, min_num_inliers: int, max_hours: float,
                   keypoint_selection: str = "largest") -> Optional[Dict[str, Any]]:
    """LightGlue matching of the pairs, or None (with a logged reason) when it cannot or should not run here."""
    try:
        import kornia  # noqa: F401
        import torch   # noqa: F401
    except ImportError as e:
        log.warning("SIFT + LightGlue not available (%s); COLMAP's SIFT matcher instead", e)
        return None
    from . import lightglue as LG
    from .gpu import gpu_lock
    pair_names = [(p[0], p[1]) for p in pairs] if pairs is not None else None
    dev = LG.pick_device(device)
    try:
        with gpu_lock(f"LightGlue ({project.root.parent.name})", enabled=dev.startswith("cuda")):
            lg = LG.match_pairs_lightglue(project, pair_names, max_keypoints=max_keypoints,
                                          filter_threshold=filter_threshold, device=dev, max_hours=max_hours,
                                          keypoint_selection=keypoint_selection)
    except LG.TooSlow as e:
        log.warning("%s; COLMAP's SIFT matcher instead", e)
        return None
    except RuntimeError as e:                       # 0.94.3: e.g. CUDA out of memory - the SIFT matcher finishes
        if "out of memory" not in str(e).lower() and "cuda" not in str(e).lower():
            raise
        log.warning("LightGlue stopped (%s); COLMAP's SIFT matcher for the pairs not matched yet",
                    str(e).splitlines()[0][:200])
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                           # noqa: BLE001
            pass
        return None
    LG.verify(project, lg["pairs_file"], max_error_px=max_error_px, min_num_inliers=min_num_inliers)
    return lg


def _finish(project: SfmProject, mode: str, max_error_px: float, min_num_inliers: int, max_num_matches: int,
            guided_matching: bool, max_ratio: float, max_distance: float, cross_check: bool,
            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The matching summary into the project settings and the matches record (what the matches were made from, for
    their reuse by a later run)."""
    import pycolmap
    d = pycolmap.Database.open(str(project.database))
    summary = {"mode": mode, "matched_pairs": int(d.num_matched_image_pairs()),
               "verified_pairs": int(d.num_verified_image_pairs()),
               "inlier_matches": int(d.num_inlier_matches()), "max_error_px_full_res": max_error_px,
               "max_ratio": max_ratio, "max_distance": max_distance, "cross_check": cross_check,
               "guided_matching": bool(guided_matching), **(extra or {})}
    d.close()
    project.settings["matching"] = summary
    project.save()
    from .database import write_matches_record
    try:
        write_matches_record(project, match_settings(mode, max_num_matches=max_num_matches, max_error_px=max_error_px,
                                                     min_num_inliers=min_num_inliers, guided_matching=guided_matching,
                                                     max_ratio=max_ratio, max_distance=max_distance,
                                                     cross_check=cross_check,
                                                     **{k: v for k, v in (extra or {}).items()
                                                        if k in ("max_keypoints", "filter_threshold", "matcher", "keypoint_selection",
                                                                 "uncalibrated")}))
    except (OSError, KeyError, ValueError) as e:        # a missing record only means no reuse next time
        log.warning("matches record not written: %s", e)
    return summary


# + the homography check (Three Forks Large N+Z34, 6 Oct: "homography_matrix.cc:167] Check failed:
# cam_rays1.size() >= 4 (3 vs. 4)" - the same calibrated-verification abort, not caught by the retry)
CALIBRATED_ABORT_SIGNS = ("kMinNumSamples", "essential_matrix.cc", "cam_rays1_with_jac", "homography_matrix.cc",
                          "cam_rays1.size()")


def set_prior_focal(project: SfmProject, families: Sequence[str] = ("Z",), value: bool = False) -> int:
    """set ``has_prior_focal_length`` of the database cameras of these camera families (``"Z"``: Mastcam-Z) -
    without it COLMAP verifies their pairs with a fundamental matrix instead of the calibrated (ray) estimators.
    Returns the number of cameras changed."""
    import pycolmap
    db = pycolmap.Database.open(str(project.database))
    ids = {int(v) for k, v in ((project.settings.get("database") or {}).get("cameras") or {}).items()
           if str(k)[:1] in tuple(families)}
    n = 0
    for cam in db.read_all_cameras():
        if int(cam.camera_id) in ids and bool(cam.has_prior_focal_length) != bool(value):
            cam.has_prior_focal_length = bool(value)
            db.update_camera(cam)
            n += 1
    db.close()
    return n


def match_settings(mode: str = "exhaustive", **kwargs: Any) -> Dict[str, Any]:
    """the settings that decide the matches of :func:`match` (``kwargs`` as passed to it, the rest at its
    defaults), for :func:`mppp.sfm.database.build_database` ``matching=`` (match reuse)."""
    import inspect
    from .database import MATCH_KEYS
    sig = inspect.signature(match).parameters
    out = {"mode": mode}
    for k in MATCH_KEYS:
        if k == "mode":
            continue
        v = kwargs.get(k, sig[k].default if k in sig else None)
        if k == "keypoint_selection" and v in (None, "largest"):
            continue                                 # 0.94.4: the default is not recorded (old records match)
        if k == "uncalibrated":
            v = kwargs.get("uncalibrated", kwargs.get("uncalibrated_families"))
            if not v:
                continue                             # 0.95.0: none is not recorded (old records match)
            v = [str(f) for f in v]
        out[k] = float(v) if isinstance(v, float) or k in ("max_ratio", "max_distance", "max_error_px") else v
    return out
