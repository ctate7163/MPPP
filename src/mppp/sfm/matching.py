"""
Feature matching on ``database.db`` (full-resolution keypoints).

``mode="exhaustive"`` is the baseline of the COLMAP pipeline spec (all pairs,
SIFT brute force; use a GPU).  ``mode="prior_pairs"`` matches only the pairs
from :func:`mppp.sfm.prior_overlap_pairs` — for CPU runs and quick tests; it
biases completeness statistics towards pairs the priors expected to overlap,
so use exhaustive matching for survival / theta_max analysis.

Geometric verification runs in full-resolution pixels: ``max_error_px`` = 6
is 1.5 px at quarter resolution, 3 px at half and 6 px at full resolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from .project import SfmProject

PathLike = Union[str, Path]


def match(project: SfmProject, mode: str = "exhaustive", pairs: Optional[Sequence[Tuple[str, str, Any]]] = None,
          use_gpu: Optional[bool] = None, max_num_matches: int = 32768, max_error_px: float = 6.0,
          min_num_inliers: int = 15, num_threads: int = -1, guided_matching: bool = True, block_size: int = 100,
          python: Optional[PathLike] = None, max_ratio: float = 0.8, max_distance: float = 0.7,
          cross_check: bool = True) -> Dict[str, Any]:
    """Match and verify; results go into ``project.database``.  Returns a summary.
    ``python``: run in another Python environment, e.g. a conda one with a CUDA
    pycolmap (v0p11, see :mod:`mppp.sfm.gpu`).  ``max_ratio``, ``max_distance``,
    ``cross_check`` (v0p22): SIFT matching (Lowe ratio test, descriptor distance,
    mutual best match); ``guided_matching``: a second pass guided by the
    verified two-view geometry; ``max_error_px``: RANSAC threshold of the
    geometric verification, full-resolution pixels."""
    if mode not in ("exhaustive", "prior_pairs"):
        raise ValueError("mode must be 'exhaustive' or 'prior_pairs'")
    if python is not None:
        from .gpu import run_step
        if mode == "prior_pairs" and pairs is not None:
            (project.root / "pairs_prior.txt").write_text("".join(f"{p[0]} {p[1]}\n" for p in pairs), encoding="utf-8")
        return run_step(python, "match", project, mode=mode, use_gpu=use_gpu, max_num_matches=max_num_matches,
                        max_error_px=max_error_px, min_num_inliers=min_num_inliers, num_threads=num_threads,
                        guided_matching=guided_matching, max_ratio=max_ratio, max_distance=max_distance,
                        cross_check=cross_check, block_size=block_size)
    import pycolmap
    mo = pycolmap.FeatureMatchingOptions()
    mo.max_num_matches = int(max_num_matches)
    mo.num_threads = int(num_threads)
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
        # v0p30: block_size images per block (COLMAP default 50): a block's descriptors are loaded once and every
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
    d = pycolmap.Database.open(db)
    summary = {"mode": mode, "matched_pairs": int(d.num_matched_image_pairs()),
               "verified_pairs": int(d.num_verified_image_pairs()),
               "inlier_matches": int(d.num_inlier_matches()), "max_error_px_full_res": max_error_px,
               "max_ratio": max_ratio, "max_distance": max_distance, "cross_check": cross_check,
               "guided_matching": bool(guided_matching)}
    d.close()
    project.settings["matching"] = summary
    project.save()
    from .database import write_matches_record          # v0p43: what these matches were made from (match reuse)
    try:
        write_matches_record(project, match_settings(mode, max_num_matches=max_num_matches, max_error_px=max_error_px,
                                                     min_num_inliers=min_num_inliers, guided_matching=guided_matching,
                                                     max_ratio=max_ratio, max_distance=max_distance,
                                                     cross_check=cross_check))
    except (OSError, KeyError, ValueError) as e:        # a missing record only means no reuse next time
        print(f"[sfm] matches record not written: {e}", flush=True)
    return summary


def match_settings(mode: str = "exhaustive", **kwargs: Any) -> Dict[str, Any]:
    """v0p43: the settings that decide the matches of :func:`match` (``kwargs`` as passed to it, the rest at its
    defaults), for :func:`mppp.sfm.database.build_database` ``matching=`` (match reuse)."""
    import inspect
    from .database import MATCH_KEYS
    sig = inspect.signature(match).parameters
    out = {"mode": mode}
    for k in MATCH_KEYS:
        if k == "mode":
            continue
        v = kwargs.get(k, sig[k].default if k in sig else None)
        out[k] = float(v) if isinstance(v, float) or k in ("max_ratio", "max_distance", "max_error_px") else v
    return out
