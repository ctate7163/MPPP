"""
Feature matching on ``database.db`` (full-resolution keypoints).

``mode="exhaustive"`` is the baseline of the COLMAP pipeline spec (all pairs,
SIFT brute force; use a GPU).  ``mode="prior_pairs"`` matches only the pairs
from :func:`mppp.sfm.prior_overlap_pairs` — for CPU runs and quick tests; it
biases completeness statistics towards pairs the priors expected to overlap,
so use exhaustive matching for survival / theta_max analysis.

Geometric verification runs in full-resolution pixels: ``max_error_px`` = 6
is 1.5 px at quarter resolution, 3 px at half and 6 px at full resolution.

v0p86 (Christian, 19 Oct): ``matcher="sift_lightglue"`` (default) matches the SIFT features with LightGlue, a
learned matcher (COLMAP 4.2 ``FeatureMatcherType.SIFT_LIGHTGLUE``) that copes better with viewpoint, scale and
illumination differences (Navcam - Mastcam-Z, other LMST).  It needs a pycolmap built with ONNX support and the
LightGlue model (downloaded by COLMAP on first use); without it COLMAP aborts the whole process, so
:func:`lightglue_probe` first tries it in a separate process on synthetic images and times it.  Unavailable, or
slower than :data:`LIGHTGLUE_MAX_MS_PER_PAIR` (a CPU-only build), the run uses SIFT brute force and says so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from .project import SfmProject

PathLike = Union[str, Path]

MATCHERS = ("sift_lightglue", "sift")
DEFAULT_MATCHER = "sift_lightglue"            # v0p86
LIGHTGLUE_MAX_MS_PER_PAIR = 80.0             # slower than this (CPU ONNX, slow GPU): SIFT brute force instead


def match(project: SfmProject, mode: str = "exhaustive", pairs: Optional[Sequence[Tuple[str, str, Any]]] = None,
          use_gpu: Optional[bool] = None, max_num_matches: int = 32768, max_error_px: float = 6.0,
          min_num_inliers: int = 15, num_threads: int = -1, guided_matching: bool = False, block_size: int = 100,
          python: Optional[PathLike] = None, max_ratio: float = 0.8, max_distance: float = 0.7,
          cross_check: bool = True, matcher: str = DEFAULT_MATCHER) -> Dict[str, Any]:
    """Match and verify; results go into ``project.database``.  Returns a summary.
    ``python``: run in another Python environment, e.g. a conda one with a CUDA
    pycolmap (v0p11, see :mod:`mppp.sfm.gpu`).  ``max_ratio``, ``max_distance``,
    ``cross_check`` (v0p22): SIFT matching (Lowe ratio test, descriptor distance,
    mutual best match); ``guided_matching``: a second pass guided by the
    verified two-view geometry; ``max_error_px``: RANSAC threshold of the
    geometric verification, full-resolution pixels."""
    if mode not in ("exhaustive", "prior_pairs"):
        raise ValueError("mode must be 'exhaustive' or 'prior_pairs'")
    if matcher not in MATCHERS:
        raise ValueError(f"matcher must be one of {MATCHERS}")
    if python is not None:
        from .gpu import run_step
        if mode == "prior_pairs" and pairs is not None:
            (project.root / "pairs_prior.txt").write_text("".join(f"{p[0]} {p[1]}\n" for p in pairs), encoding="utf-8")
        kw = dict(mode=mode, use_gpu=use_gpu, max_num_matches=max_num_matches,
                  max_error_px=max_error_px, min_num_inliers=min_num_inliers, num_threads=num_threads,
                  guided_matching=guided_matching, max_ratio=max_ratio, max_distance=max_distance,
                  cross_check=cross_check, block_size=block_size, matcher=matcher)
        try:
            return run_step(python, "match", project, **kw)
        except RuntimeError as e:
            if not any(k in str(e) for k in CALIBRATED_ABORT_SIGNS):
                raise
            # v0p73: COLMAP 4.2 aborted in the calibrated two-view verification (Rockytop N+Z34, 3 Oct: "Check
            # failed: cam_rays1_with_jac.size() >= kMinNumSamples (4 vs. 5)" - a pair whose keypoints mostly fail to
            # unproject).  The Mastcam-Z cameras then lose their prior focal length in the database for one more run,
            # so their pairs are verified uncalibrated (fundamental matrix, no rays); the pairs already matched are kept
            n = set_prior_focal(project, families=("Z",), value=False)
            print(f"[mppp] matching aborted in COLMAP's calibrated verification; again with the {n} Mastcam-Z cameras "
                  f"verified uncalibrated (prior focal length off)", flush=True)
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
    mo.num_threads = int(num_threads)
    mo.guided_matching = bool(guided_matching)
    mo.sift.max_ratio = float(max_ratio)
    mo.sift.max_distance = float(max_distance)
    mo.sift.cross_check = bool(cross_check)
    probe = None
    used = "sift"
    if matcher == "sift_lightglue":                       # v0p86
        probe = lightglue_probe(use_gpu=use_gpu)
        if probe.get("ok") and float(probe.get("ms_per_pair") or 1e9) <= LIGHTGLUE_MAX_MS_PER_PAIR:
            mo.type = pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE
            used = "sift_lightglue"
            print(f"[mppp] matcher: SIFT + LightGlue ({probe.get('ms_per_pair'):.0f} ms per pair in the probe)", flush=True)
        else:
            why = probe.get("error") or f"{probe.get('ms_per_pair'):.0f} ms per pair (> {LIGHTGLUE_MAX_MS_PER_PAIR:g})"
            print(f"[mppp] matcher: SIFT + LightGlue not used here ({why}); SIFT brute force", flush=True)
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
               "guided_matching": bool(guided_matching), "matcher_requested": matcher, "matcher": used,
               "lightglue_probe": probe}
    d.close()
    project.settings["matching"] = summary
    project.save()
    from .database import write_matches_record          # v0p43: what these matches were made from (match reuse)
    try:
        write_matches_record(project, match_settings(mode, max_num_matches=max_num_matches, max_error_px=max_error_px,
                                                     min_num_inliers=min_num_inliers, guided_matching=guided_matching,
                                                     max_ratio=max_ratio, max_distance=max_distance,
                                                     cross_check=cross_check, matcher=matcher))
    except (OSError, KeyError, ValueError) as e:        # a missing record only means no reuse next time
        print(f"[sfm] matches record not written: {e}", flush=True)
    return summary


# v0p74.9: + the homography check (Three Forks Large N+Z34, 6 Oct: "homography_matrix.cc:167] Check failed:
# cam_rays1.size() >= 4 (3 vs. 4)" - the same calibrated-verification abort, not caught by the retry)
CALIBRATED_ABORT_SIGNS = ("kMinNumSamples", "essential_matrix.cc", "cam_rays1_with_jac", "homography_matrix.cc",
                          "cam_rays1.size()")


def set_prior_focal(project: SfmProject, families: Sequence[str] = ("Z",), value: bool = False) -> int:
    """v0p73: set ``has_prior_focal_length`` of the database cameras of these camera families (``"Z"``: Mastcam-Z) -
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


# ------------------------------------------------------------------ v0p86: SIFT + LightGlue availability
def _probe_cache() -> Path:
    import os
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "mppp"
    return base / "lightglue_probe.json"


def lightglue_probe(use_gpu: Optional[bool] = None, refresh: bool = False, timeout_s: float = 600.0) -> Dict[str, Any]:
    """Whether this Python's pycolmap can match SIFT features with LightGlue, and how fast: runs
    ``python -m mppp.sfm.matching probe`` (synthetic 1648 x 1200 images, SIFT, then LightGlue on 1 and on 15 pairs;
    the difference gives ms per pair without the model loading).  A missing ONNX build aborts that process, not
    this one.  The result is cached per Python and pycolmap version (``%LOCALAPPDATA%/mppp/lightglue_probe.json``)."""
    import json
    import subprocess
    import sys
    import pycolmap
    key = f"{sys.executable}|{pycolmap.__version__}|{'gpu' if use_gpu else ('cpu' if use_gpu is False else 'auto')}"
    cache = _probe_cache()
    try:
        have = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    except ValueError:
        have = {}
    if not refresh and key in have:
        return dict(have[key], cached=True)
    try:
        import os
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[2])      # the folder that holds the mppp package
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        r = subprocess.run([sys.executable, "-m", "mppp.sfm.matching", "probe", json.dumps({"use_gpu": use_gpu})],
                           capture_output=True, text=True, timeout=timeout_s, env=env)
        last = [ln for ln in r.stdout.splitlines() if ln.startswith("MPPP_PROBE ")]
        if r.returncode == 0 and last:
            out = json.loads(last[-1][len("MPPP_PROBE "):])
        else:
            err = [ln for ln in (r.stderr or "").splitlines() if "what()" in ln or "Error" in ln or "error" in ln]
            out = {"ok": False, "error": (err[-1].strip() if err else f"exit {r.returncode}")[:300]}
    except subprocess.TimeoutExpired:
        out = {"ok": False, "error": f"probe took more than {timeout_s:.0f} s"}
    out.update(python=sys.executable, pycolmap=pycolmap.__version__)
    have[key] = out
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(have, indent=1), encoding="utf-8")
    except OSError:
        pass
    return out


def _probe_main(opts: Dict[str, Any]) -> None:
    import json
    import tempfile
    import time
    import numpy as np
    import pycolmap
    from PIL import Image
    use_gpu = opts.get("use_gpu")
    device = pycolmap.Device.auto if use_gpu is None else (pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu)
    rng = np.random.default_rng(0)
    H, W = 1200, 1648
    base = rng.random((H // 4 + 40, W // 4 + 40))
    big = np.kron(base, np.ones((4, 4)))                    # blocky texture: thousands of SIFT features
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        times = {}
        for n in (2, 6):
            img = td / f"img{n}"
            img.mkdir()
            for k in range(n):
                y, x = 3 * k, 5 * k
                a = big[y:y + H, x:x + W]
                Image.fromarray((255 * a).astype(np.uint8)).save(img / f"{k}.png")
            db = td / f"d{n}.db"
            eo = pycolmap.FeatureExtractionOptions()
            eo.sift.max_num_features = 8192
            pycolmap.extract_features(str(db), str(img), extraction_options=eo, device=pycolmap.Device.cpu)
            mo = pycolmap.FeatureMatchingOptions()
            mo.type = pycolmap.FeatureMatcherType.SIFT_LIGHTGLUE
            t0 = time.time()
            pycolmap.match_exhaustive(str(db), matching_options=mo, device=device)
            times[n] = time.time() - t0
            d = pycolmap.Database.open(str(db))
            inl = int(d.num_inlier_matches())
            d.close()
        ms = 1000.0 * max(times[6] - times[2], 1e-3) / 14.0
        print("MPPP_PROBE " + json.dumps({"ok": inl > 0, "ms_per_pair": ms, "seconds": times, "inlier_matches": inl,
                                          "cuda": bool(getattr(pycolmap, "has_cuda", False))}), flush=True)


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    if len(_sys.argv) >= 2 and _sys.argv[1] == "probe":
        _probe_main(_json.loads(_sys.argv[2]) if len(_sys.argv) > 2 else {})
