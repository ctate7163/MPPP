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
          min_num_inliers: int = 15, num_threads: int = -1, guided_matching: bool = False,
          python: Optional[PathLike] = None) -> Dict[str, Any]:
    """Match and verify; results go into ``project.database``.  Returns a summary.
    ``python``: run in another Python environment, e.g. a conda one with a CUDA
    pycolmap (v0p11, see :mod:`mppp.sfm.gpu`)."""
    if mode not in ("exhaustive", "prior_pairs"):
        raise ValueError("mode must be 'exhaustive' or 'prior_pairs'")
    if python is not None:
        from .gpu import run_step
        if mode == "prior_pairs" and pairs is not None:
            (project.root / "pairs_prior.txt").write_text("".join(f"{p[0]} {p[1]}\n" for p in pairs), encoding="utf-8")
        return run_step(python, "match", project, mode=mode, use_gpu=use_gpu, max_num_matches=max_num_matches,
                        max_error_px=max_error_px, min_num_inliers=min_num_inliers, num_threads=num_threads,
                        guided_matching=guided_matching)
    import pycolmap
    mo = pycolmap.FeatureMatchingOptions()
    mo.max_num_matches = int(max_num_matches)
    mo.num_threads = int(num_threads)
    mo.guided_matching = bool(guided_matching)
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
        pycolmap.match_exhaustive(db, matching_options=mo, verification_options=vo, device=device)
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
               "inlier_matches": int(d.num_inlier_matches()), "max_error_px_full_res": max_error_px}
    d.close()
    project.settings["matching"] = summary
    project.save()
    return summary
