"""
Batch driver: process a list of PDS images into an output directory.

Output layout::

    <out>/images_png16/<PDS stem>.png     16-bit linear RGBA (default)
    <out>/images_png8/ ...                if "PNG8" in export.formats
    <out>/images_tiff16/ ...              if "TIFF16" in export.formats
    <out>/masks/<PDS stem>.png            if export.write_mask_files
    <out>/references.txt                  Metashape reference import (offset removed)
    <out>/references_absolute.txt         same, landing-frame ENU metres
    <out>/colmap/sparse_prior/...         COLMAP text model with pose priors
    <out>/mppp_manifest_<version>.json     per-image metadata
    <out>/mppp_config_<version>.json       config + code version snapshot
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from . import VERSION_TAG
from . import colmap as _colmap
from . import writers
from .config import load_config, save_config_snapshot
from .image import MPPPImage

PathLike = Union[str, Path]
_FORMAT_DIR = {"PNG16": "images_png16", "PNG8": "images_png8", "TIFF16": "images_tiff16"}


def write_image_products(im: MPPPImage, out_dir: PathLike) -> Dict[str, str]:
    out_dir, exp = Path(out_dir), im.config["export"]
    meta = im.meta if exp["embed_metadata"] else None
    alpha = exp["store_mask_in_alpha"]
    written = {}
    for fmt in exp["formats"]:
        bits = 8 if fmt == "PNG8" else 16
        arr = im.rgba(bits) if alpha else (im.image_int8 if bits == 8 else im.image_int16)
        target = out_dir / _FORMAT_DIR[fmt] / im.fn.stem
        if fmt == "TIFF16":
            written[fmt] = str(writers.save_tiff16(arr, target, meta))
        else:
            written[fmt] = str(writers.save_png(arr, target, meta, embed_gps=exp["embed_gps"]))
    if exp["write_mask_files"]:
        written["mask"] = str(writers.save_mask(im.mask, out_dir / "masks" / im.fn.stem))
    return written


def check_mask_checkpoint(cfg: Dict[str, Any]) -> Optional[Path]:
    """
    With ``masking.infer_mask`` on, resolve the mask checkpoint once, before any
    image (v0p10; v0p13: registry names are downloaded here on first use).
    Raises FileNotFoundError with instructions.  Returns the path.
    """
    m = cfg.get("masking", {})
    if not m.get("infer_mask"):
        return None
    from .mask.hub import resolve_checkpoint
    return resolve_checkpoint(m.get("checkpoint"))


def filter_to_existing(paths: Sequence[PathLike], out_dir: PathLike, fmt: str) -> tuple:
    """
    Keep the products whose output image is still in ``out_dir/<folder>``
    (``fmt``: an export format such as ``"PNG8"``, or a folder name).  Images
    are matched by PDS stem (outputs keep the PDS file name).  Returns
    (kept paths, report).
    """
    out_dir = Path(out_dir)
    folder = out_dir / _FORMAT_DIR.get(str(fmt).upper(), str(fmt))
    if not folder.is_dir():
        raise FileNotFoundError(f"only_existing: {folder} does not exist (process once without only_existing first)")
    stems = {p.stem for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in (".png", ".tif", ".tiff")}
    paths = [Path(p) for p in paths]
    kept = [p for p in paths if p.stem in stems]
    if not kept:
        raise ValueError(f"only_existing: none of the {len(paths)} selected products has an image in {folder}")
    selected = {p.stem for p in paths}
    report = {"folder": str(folder), "n_selected": len(paths), "n_kept": len(kept), "n_removed": len(paths) - len(kept),
              "removed": sorted(p.name for p in paths if p.stem not in stems),
              "not_in_selection": sorted(stems - selected)}
    return kept, report


def _canon(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _config_diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Dotted keys whose values differ between two (JSON-canonical) configs."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for k in sorted(set(a) | set(b)):
            out += _config_diff(a.get(k), b.get(k), f"{prefix}{k}.")
        return out
    return [] if a == b else [prefix.rstrip(".")]


def _reference_from_meta(m: Dict[str, Any]) -> list:
    """The ``MPPPImage.reference`` row, rebuilt from a manifest entry."""
    return [Path(m["source_product"]).stem, *m["pose"]["C_enu_m"], *m["pose"]["metashape_ypr_deg"]]


def reusable_images(out_dir: PathLike, cfg: Dict[str, Any]) -> tuple:
    """
    Entries of the existing manifest in ``out_dir`` that can be reused as they
    are: processed with the same configuration (``mppp_config_<tag>.json``)
    and with every output file still present.  Returns ({PDS stem: meta},
    report) — empty when there is no manifest or the configuration changed.
    """
    out_dir = Path(out_dir)
    man_p, cfg_p = out_dir / f"mppp_manifest_{VERSION_TAG}.json", out_dir / f"mppp_config_{VERSION_TAG}.json"
    if not man_p.is_file():
        # v0p15: after a version change, the newest manifest of an earlier version (same configuration required)
        older = sorted(out_dir.glob("mppp_manifest_v*.json"), key=lambda q: q.stat().st_mtime)
        if older:
            man_p = older[-1]
            cfg_p = out_dir / man_p.name.replace("mppp_manifest_", "mppp_config_")
    rep: Dict[str, Any] = {"manifest": str(man_p), "reusable": 0, "outputs_missing": [], "config_changed": []}
    if not man_p.is_file():
        rep["reason"] = "no manifest"
        return {}, rep
    try:
        old_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))["config"]
    except Exception:                                            # noqa: BLE001
        rep["reason"] = f"no readable {cfg_p.name}"
        return {}, rep
    # skip_inference_at only affects the images of the listed stations: checked per image below
    diff = [k for k in _config_diff(_canon(cfg), _canon(old_cfg)) if k != "masking.skip_inference_at"]
    if diff:
        rep.update(reason="configuration changed", config_changed=diff)
        return {}, rep
    from .config import parse_stations
    skip = parse_stations(cfg["masking"].get("skip_inference_at"))
    infer = bool(cfg["masking"].get("infer_mask"))
    rep["mask_inference_changed"] = []
    have: Dict[str, Dict[str, Any]] = {}
    for m in json.loads(man_p.read_text(encoding="utf-8")).get("images", []):
        outs = [out_dir / v for v in (m.get("outputs") or {}).values()]
        stem = Path(m["source_product"]).stem
        want_inferred = infer and (m.get("site"), m.get("drive")) not in skip
        if bool((m.get("mask") or {}).get("inferred")) != want_inferred:
            rep["mask_inference_changed"].append(stem)                   # station added to / removed from the list
        elif outs and all(o.is_file() for o in outs):
            have[stem] = m
        else:
            rep["outputs_missing"].append(stem)
    rep["reusable"] = len(have)
    return have, rep


def process_images(paths: Iterable[PathLike], out_dir: PathLike,
                   config: Optional[Union[PathLike, Dict[str, Any]]] = None,
                   waypoints: Optional[Dict[str, Any]] = None,
                   stop_on_error: bool = False, progress: bool = True,
                   provenance: Optional[Dict[str, Any]] = None,
                   only_existing: Optional[str] = None,
                   reuse_existing: bool = False) -> Dict[str, Any]:
    """
    Returns the manifest (also written to disk).  ``provenance`` (e.g. the
    report from ``mppp.scapes.select_scape``) is stored in both the manifest
    and the config snapshot.

    ``only_existing`` (v0p14): an output format (``"PNG8"``, ``"PNG16"``,
    ``"TIFF16"``) or a sub-folder name of ``out_dir``.  Only the selected
    products whose image is still in that folder are processed, and the
    manifest, references and COLMAP priors are built from those alone.  Use it
    after deleting unsuitable images from e.g. ``images_png8/``: rerun with the
    same selection and ``only_existing="PNG8"``.  Nothing is deleted; outputs
    of the removed images in other folders (masks, other formats) are left
    as they are, but they are no longer in the manifest.

    ``reuse_existing`` (v0p14.7): images already in ``out_dir``'s manifest,
    processed with the same configuration and with all their outputs present,
    are taken from the manifest instead of being processed again; only the
    rest of the selection is processed.  The manifest, references and priors
    always describe exactly this selection (after ``only_existing``).  A
    changed configuration (e.g. another mask model) processes everything.
    """
    cfg = load_config(config)
    if not reuse_existing:
        check_mask_checkpoint(cfg)      # one clear error up front, not one FAILED line per image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in paths]
    kept_filter = None
    if only_existing:
        paths, kept_filter = filter_to_existing(paths, out_dir, only_existing)
        if progress:
            print(f"[mppp] only_existing={only_existing!r}: {kept_filter['n_kept']} of {kept_filter['n_selected']} "
                  f"selected products still have an image in {kept_filter['folder']}; "
                  f"{kept_filter['n_removed']} removed" + (f"; {len(kept_filter['not_in_selection'])} images in the "
                  f"folder are not in this selection and are ignored" if kept_filter["not_in_selection"] else ""))
    reused: Dict[str, Dict[str, Any]] = {}
    reuse_rep = None
    if reuse_existing:
        have, reuse_rep = reusable_images(out_dir, cfg)
        reused = {p.stem: have[p.stem] for p in paths if p.stem in have}
        todo = [p for p in paths if p.stem not in reused]
        reuse_rep.update(reused=len(reused), to_process=len(todo),
                         dropped_from_manifest=sorted(set(have) - set(reused)))
        if progress:
            why = (f" ({reuse_rep['reason']}" + (f": {', '.join(reuse_rep['config_changed'][:6])}"
                   if reuse_rep["config_changed"] else "") + ")") if reuse_rep.get("reason") else ""
            print(f"[mppp] reuse_existing: {len(reused)} of {len(paths)} images reused from the manifest, "
                  f"{len(todo)} to process{why}"
                  + (f"; {len(reuse_rep['dropped_from_manifest'])} manifest images not in this selection are left out"
                     if reuse_rep["dropped_from_manifest"] else "")
                  + (f"; {len(reuse_rep['outputs_missing'])} had missing output files"
                     if reuse_rep["outputs_missing"] else "")
                  + (f"; {len(reuse_rep['mask_inference_changed'])} processed again for masking.skip_inference_at"
                     if reuse_rep.get("mask_inference_changed") else ""))
    else:
        todo = paths
    if todo and reuse_existing:
        check_mask_checkpoint(cfg)      # only when something is processed (reused images need no model)
    new_meta: Dict[str, Dict[str, Any]] = {}
    new_ref: Dict[str, list] = {}
    failed: List[Dict[str, str]] = []
    t0 = time.time()

    for i, p in enumerate(todo):
        try:
            im = MPPPImage(p, cfg, waypoints)
            files = write_image_products(im, out_dir)
            m = im.meta
            m["outputs"] = {k: str(Path(v).relative_to(out_dir)) for k, v in files.items()}
            new_meta[p.stem] = m
            new_ref[p.stem] = im.reference
            if progress:
                print(f"[{i + 1}/{len(todo)}] {p.name}  ok  ({time.time() - t0:.0f} s)")
        except Exception as e:                                   # noqa: BLE001
            if stop_on_error:
                raise
            failed.append({"file": str(p), "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()})
            if progress:
                print(f"[{i + 1}/{len(todo)}] {p.name}  FAILED: {type(e).__name__}: {e}")

    metas: List[Dict[str, Any]] = []
    refs: List[list] = []
    for p in paths:                                              # manifest in selection order
        if p.stem in new_meta:
            metas.append(new_meta[p.stem])
            refs.append(new_ref[p.stem])
        elif p.stem in reused:
            metas.append(reused[p.stem])
            refs.append(_reference_from_meta(reused[p.stem]))

    manifest: Dict[str, Any] = {"mppp_version": VERSION_TAG, "n_requested": len(paths),
                                "n_processed": len(metas), "failed": failed, "provenance": provenance,
                                "only_existing": kept_filter, "reuse_existing": reuse_rep, "images": metas}
    frames = sorted({m["pose"]["frame"] for m in metas})
    manifest["world_frames"] = frames
    if len(frames) > 1:
        manifest["warning"] = ("Images are in different world frames (some lack waypoints); "
                               "references / COLMAP priors are NOT mutually consistent.")
    exp = cfg["export"]
    if metas:
        first_fmt = exp["formats"][0]
        ext = ".tif" if first_fmt == "TIFF16" else ".png"
        offset = writers.reference_offset(refs)
        manifest["reference_offset_enu_m"] = offset.tolist()
        if exp["write_references"]:
            writers.save_references(refs, out_dir / "references.txt", offset, ext)
            writers.save_references(refs, out_dir / "references_absolute.txt", None, ext)
        if exp["write_colmap"]:
            manifest["colmap"] = _colmap.write_text_model(metas, out_dir / "colmap" / "sparse_prior", offset, ext)
    wp_src = (waypoints or {}).get("_mppp_source")
    save_config_snapshot(cfg, out_dir, extra={"waypoints": wp_src, "n_images": len(metas), "provenance": provenance,
                                              "only_existing": kept_filter})
    if exp["write_manifest"]:
        (out_dir / f"mppp_manifest_{VERSION_TAG}.json").write_text(
            json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return manifest
