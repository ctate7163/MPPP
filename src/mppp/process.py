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


def process_images(paths: Iterable[PathLike], out_dir: PathLike,
                   config: Optional[Union[PathLike, Dict[str, Any]]] = None,
                   waypoints: Optional[Dict[str, Any]] = None,
                   stop_on_error: bool = False, progress: bool = True,
                   provenance: Optional[Dict[str, Any]] = None,
                   only_existing: Optional[str] = None) -> Dict[str, Any]:
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
    """
    cfg = load_config(config)
    check_mask_checkpoint(cfg)          # one clear error up front, not one FAILED line per image
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
    metas: List[Dict[str, Any]] = []
    refs: List[list] = []
    failed: List[Dict[str, str]] = []
    t0 = time.time()

    for i, p in enumerate(paths):
        try:
            im = MPPPImage(p, cfg, waypoints)
            files = write_image_products(im, out_dir)
            m = im.meta
            m["outputs"] = {k: str(Path(v).relative_to(out_dir)) for k, v in files.items()}
            metas.append(m)
            refs.append(im.reference)
            if progress:
                print(f"[{i + 1}/{len(paths)}] {p.name}  ok  ({time.time() - t0:.0f} s)")
        except Exception as e:                                   # noqa: BLE001
            if stop_on_error:
                raise
            failed.append({"file": str(p), "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()})
            if progress:
                print(f"[{i + 1}/{len(paths)}] {p.name}  FAILED: {type(e).__name__}: {e}")

    manifest: Dict[str, Any] = {"mppp_version": VERSION_TAG, "n_requested": len(paths),
                                "n_processed": len(metas), "failed": failed, "provenance": provenance,
                                "only_existing": kept_filter, "images": metas}
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
