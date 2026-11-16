"""
mppp.ibr.rover_masks - rover masks from the mask model for the panoramas, for frames the block made without it
(0.98.25, Christian 16 Nov 2026: "the z34 without the rover still has a lot of hardware. you may need to rerun those
images with the mask inference enabled").

A block may process some frames with the mask model off (``no_mask_inference_at`` in sites.json: Three Forks north's
ZCAM08692 and the sol 688 Navcam 360 keep the rover in for the deck-fiducial work).  Their ``masks/`` hold only the
invalid pixels, and the rover model's silhouette (0.98.22, :func:`mppp.ibr.ods.model_rover_masks`) is too coarse a
stand-in: its edges miss by a few pixels to tens of pixels (pose, model detail, the wheels' turn), and every miss is
hardware on the ground of the terrain-only panorama.

The panorama therefore runs the mask model on those frames itself and keeps the result beside the block's masks,

    <project>/masks_ibr/<image>.png      0 = rover / no data, 255 = terrain (the block's mask convention)
    <project>/masks_ibr/masks_ibr.json   per frame: the model, the share kept, the time

without touching ``masks/`` - the block's alignment, its run key and the deck-fiducial work see nothing of it.  The
readers of the IBR (:func:`mask_file`) take ``masks_ibr`` first.  The input is the block's 8-bit image: the 16-bit
linear product / 64, clipped at 255 - the model's own input during processing (``MPPPImage._infer_mask``), the same
clipping included; a block made with ``color.apply_gamma`` is linearised first.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

import numpy as np

log = logging.getLogger("mppp.ibr.rover_masks")
PathLike = Union[str, Path]

DIR = "masks_ibr"
RECORD = "masks_ibr.json"


def mask_file(root: PathLike, name: str) -> Path:
    """The rover mask the IBR uses for ``name``: ``masks_ibr/<name>.png`` where the panorama made one, else the
    block's ``masks/<name>.png`` (either may be missing)."""
    m = Path(root) / DIR / (name + ".png")
    return m if m.is_file() else Path(root) / "masks" / (name + ".png")


def _processed_dir(project) -> Optional[Path]:
    for d in (Path((getattr(project, "settings", None) or {}).get("processed_dir", "") or "."),
              Path(project.root).parent / "processed"):
        if d.is_dir() and any(d.glob("mppp_manifest_v*.json")):
            return d
    return None


def skipped_frames(project) -> Set[str]:
    """The block's images (``<stem>.png``) processed with the mask model off (``mask.inference_skipped`` in the newest
    ``processed/mppp_manifest_v*.json``)."""
    d = _processed_dir(project)
    if d is None:
        return set()
    man = sorted(d.glob("mppp_manifest_v*.json"), key=lambda f: f.stat().st_mtime)[-1]
    try:
        images = json.loads(man.read_text(encoding="utf-8")).get("images", [])
    except (OSError, ValueError) as e:
        log.warning("rover masks: cannot read %s (%s)", man, e)
        return set()
    return {Path(m["source_product"]).stem + ".png" for m in images
            if (m.get("mask") or {}).get("inference_skipped") and m.get("source_product")}


def model_input(img8_rgb: np.ndarray, color: Dict[str, Any]) -> np.ndarray:
    """The block's 8-bit image -> the mask model's input on its 0-255 scale (the gamma of ``color.apply_gamma`` undone)."""
    x = np.asarray(img8_rgb, np.float32)
    if color.get("apply_gamma"):
        g = float(color.get("gamma", 2.0))
        x = (x / 256.0 ** (1.0 - 1.0 / g)) ** g
    return x


def _infer_default(img: np.ndarray, masking: Dict[str, Any]) -> np.ndarray:
    from ..mask import infer_mask                               # lazy: torch / timm
    mask, _, card = infer_mask(img, masking.get("checkpoint"), masking.get("device", "auto"),
                               masking.get("threshold"), masking.get("dilate_kernel"))
    _infer_default.card = card                                  # type: ignore[attr-defined]
    return mask


def ensure_rover_masks(project, names: Iterable[str], force: bool = False,
                       infer: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
                       skipped: Optional[Set[str]] = None) -> Dict[str, Any]:
    """The mask model's rover mask in ``<project>/masks_ibr/`` for every frame of ``names`` the block processed with
    the model off (made where missing or older than the image).  ``infer(image_0_255_rgb, masking_settings)`` -> uint8
    0/255 (default: :func:`mppp.mask.infer_mask` with the block's checkpoint and threshold).  Returns
    {"dir", "skipped": n, "have": [names], "made": n, "failed": {name: why}}."""
    import cv2
    root = Path(project.root)
    names = list(dict.fromkeys(names))
    skip = skipped_frames(project) if skipped is None else set(skipped)
    want = [n for n in names if n in skip]
    rep: Dict[str, Any] = {"dir": str(root / DIR), "skipped": len(want), "have": [], "made": 0, "failed": {}}
    if not want:
        return rep
    out = root / DIR
    out.mkdir(parents=True, exist_ok=True)
    recf = out / RECORD
    record = json.loads(recf.read_text(encoding="utf-8")) if recf.is_file() else {}
    todo = []
    for n in want:
        f, img = out / (n + ".png"), root / "images" / n
        if not force and f.is_file() and (not img.is_file() or f.stat().st_mtime >= img.stat().st_mtime):
            rep["have"].append(n)
        else:
            todo.append(n)
    if todo:
        try:
            from .linear16 import block_config
            cfg = block_config(project)
        except FileNotFoundError:
            from ..config import load_config
            cfg = load_config(None)
        masking, color = dict(cfg.get("masking") or {}), dict(cfg.get("color") or {})
        fn = infer or _infer_default
        t0 = time.time()
        for i, n in enumerate(todo):
            bgr = cv2.imread(str(root / "images" / n), cv2.IMREAD_COLOR)
            if bgr is None:
                rep["failed"][n] = "the block's image cannot be read"
                continue
            try:
                m = np.asarray(fn(model_input(bgr[..., ::-1], color), masking))
            except ImportError as e:                            # no torch: the panorama goes on with what it had
                log.warning("rover masks: the mask model is not available here (%s) - %d frame(s) keep the block's "
                            "mask and the rover model's silhouette", e, len(todo) - i)
                for k in todo[i:]:
                    rep["failed"][k] = f"ImportError: {e}"
                break
            except Exception as e:                              # noqa: BLE001 - reported per frame
                rep["failed"][n] = f"{type(e).__name__}: {e}"
                continue
            m = (m > 0) & (bgr.max(axis=2) > 0)                 # no data stays out (0 = invalid in the 8-bit image)
            old = cv2.imread(str(root / "masks" / (n + ".png")), cv2.IMREAD_GRAYSCALE)
            if old is not None and old.shape == m.shape:
                m &= old > 0                                    # the block's own exclusions (invalid, static mask) kept
            cv2.imwrite(str(out / (n + ".png")), m.astype(np.uint8) * 255)
            card = getattr(fn, "card", None) or {}
            record[n] = {"kept": round(float(m.mean()), 4), "model": card.get("release_name") or card.get("name")
                         or masking.get("checkpoint"), "made": time.strftime("%Y-%m-%d %H:%M:%S")}
            rep["have"].append(n)
            rep["made"] += 1
            if rep["made"] % 25 == 0:
                log.info("rover masks: %d of %d made (%.0f s)", rep["made"], len(todo), time.time() - t0)
        if rep["made"]:
            recf.write_text(json.dumps(record, indent=1), encoding="utf-8")
            log.info("rover masks: the mask model on %d frame(s) the block made without it, in %.0f s -> %s",
                     rep["made"], time.time() - t0, out)
    for n, why in list(rep["failed"].items())[:5]:
        log.warning("rover masks: %s: %s", n, why)
    return rep
