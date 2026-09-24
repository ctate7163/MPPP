"""
Building a mask training set (``images/`` from PDS, ``masks/`` and ``images_variable/``).

v0p6: :func:`regenerate_images_from_pds` rebuilds ``images/`` from the original
PDS products for every mask in ``masks/`` (MPPP 8-bit, linear, padded to the
detector frame — the exact input the network gets at inference).

Layout of a mask training set, e.g. ``masks_training_set_vN`` (reconstructed from v5-v7 on
22 Sep 2026; the original generating code was not found):

* ``images/<stem>.png``        MPPP "standard" 8-bit RGBA (fixed radiometric
                               scale; alpha = valid pixels).  Source of truth
                               for the RGB; never modified here.
* ``masks/<stem>.png``         training mask, 255 = terrain.  Drawn/edited in
                               Metashape (``masks_v*.psx``, cameras pointing at
                               ``images/``) and exported.
* ``images_variable/<stem>.png``  the same scene with a DIFFERENT tone curve
                               (photometric augmentation), alpha = the mask.
                               v6's copies are bit-identical to v5's
                               ``images/`` RGB — an older processing with a
                               per-image contrast stretch — with v6's masks
                               written into alpha.

What was measured (15 v6 standard/variable pairs, all six cameras): variable
≈ clip((x − lo)/(hi − lo), 0, 1)^g applied per image, near-identical across
channels, with lo 0.00–0.11, hi 0.86–1.76, g 0.47–0.85 (typical 0.7); fit rms
0.4–8 DN (the old radiometry differed slightly, so it is not an exact function
of today's images).  Frames that already had an original variable version keep
its RGB; new frames get a synthesized one from that measured parameter range,
seeded by the file name so reruns are identical.  Training reads only the RGB
of the image files and the separate ``masks/`` files.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

# measured range of the v5/v6 "variable" tone curve (see module docstring)
@dataclass(frozen=True)
class VariableRecipe:
    lo: Tuple[float, float] = (0.0, 0.11)
    hi: Tuple[float, float] = (0.90, 1.50)
    gamma: Tuple[float, float] = (0.50, 0.85)
    channel_gain: Tuple[float, float] = (0.95, 1.05)   # small per-channel white-balance jitter


def _seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def make_variable(rgb: np.ndarray, valid: np.ndarray, name: str,
                  recipe: VariableRecipe = VariableRecipe()) -> Tuple[np.ndarray, Dict[str, float]]:
    """uint8 RGB -> uint8 RGB with a per-image tone curve; invalid pixels stay 0, valid never 0."""
    rng = np.random.default_rng(_seed(name))
    lo = rng.uniform(*recipe.lo)
    hi = rng.uniform(*recipe.hi)
    g = rng.uniform(*recipe.gamma)
    gains = rng.uniform(*recipe.channel_gain, size=3)
    x = rgb.astype(np.float32) / 255.0 * gains.reshape(1, 1, 3).astype(np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0) ** g
    out = np.clip(np.rint(y * 255.0), 1, 255).astype(np.uint8)
    out[~valid] = 0
    return out, {"lo": float(lo), "hi": float(hi), "gamma": float(g),
                 "gain_r": float(gains[0]), "gain_g": float(gains[1]), "gain_b": float(gains[2])}


# ------------------------------------------------------------ Metashape masks
@dataclass
class PsxMasks:
    """Masks stored inside a Metashape project (``<name>.files/0/0/masks/masks.zip``)."""
    psx: Path
    frame_dir: Path
    cameras: Dict[str, str]          # camera_id -> photo path (relative to frame_dir)
    masks: Dict[str, str]            # camera_id -> member name in masks.zip

    @classmethod
    def open(cls, psx: PathLike, chunk: int = 0, frame: int = 0) -> "PsxMasks":
        psx = Path(psx)
        files = psx.with_suffix("").with_name(psx.stem + ".files")
        frame_dir = files / str(chunk) / str(frame)
        frame_zip = frame_dir / "frame.zip"
        mask_zip = frame_dir / "masks" / "masks.zip"
        for p in (frame_zip, mask_zip):
            if not p.is_file():
                raise FileNotFoundError(f"{p} not found — is {psx.name} a Metashape project with masks?")
        with zipfile.ZipFile(frame_zip) as z:
            root = ET.fromstring(z.read("doc.xml"))
        cams = {c.get("camera_id"): c.find("photo").get("path")
                for c in root.iter("camera") if c.find("photo") is not None}
        with zipfile.ZipFile(mask_zip) as z:
            mroot = ET.fromstring(z.read("doc.xml"))
        masks = {m.get("camera_id"): m.get("path") for m in mroot.iter("mask")}
        return cls(psx, frame_dir, cams, masks)

    def image_path(self, camera_id: str) -> Path:
        return (self.frame_dir / self.cameras[camera_id]).resolve()

    def iter_masks(self) -> Iterable[Tuple[str, Path, np.ndarray]]:
        """(camera_id, image path, mask uint8 0/255) for every camera that has a mask."""
        with zipfile.ZipFile(self.frame_dir / "masks" / "masks.zip") as z:
            for cid, member in self.masks.items():
                if cid not in self.cameras:
                    continue
                m = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_GRAYSCALE)
                yield cid, self.image_path(cid), np.where(m > 127, 255, 0).astype(np.uint8)


def _archive_dir(d: Path) -> Optional[Path]:
    if not d.is_dir() or not any(d.iterdir()):
        return None
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = d.with_name(f"{d.name}_prev_{stamp}")
    d.rename(dest)
    return dest


def export_masks_from_psx(psx: PathLike, images_dir: PathLike, masks_dir: PathLike,
                          archive_existing: bool = True) -> Dict[str, object]:
    """
    Write ``masks_dir/<image name>`` for every camera in the project whose image
    lies in ``images_dir``.  An existing masks folder is renamed to
    ``<masks>_prev_<time>`` first (nothing is deleted).
    """
    pm = PsxMasks.open(psx)
    images_dir, masks_dir = Path(images_dir).resolve(), Path(masks_dir)
    archived = _archive_dir(masks_dir) if archive_existing else None
    masks_dir.mkdir(parents=True, exist_ok=True)
    written, outside, size_mismatch, no_image = 0, [], [], []
    for cid, img_path, m in pm.iter_masks():
        if img_path.parent != images_dir:
            outside.append(str(img_path))
            continue
        img = images_dir / img_path.name
        if not img.is_file():
            no_image.append(img.name)
            continue
        if _png_size(img) != m.shape:
            size_mismatch.append(img.name)
            continue
        cv2.imwrite(str(masks_dir / img_path.name), m)
        written += 1
    return {"psx": str(psx), "n_cameras": len(pm.cameras), "n_masks_in_project": len(pm.masks),
            "written": written, "cameras_outside_images_dir": outside, "size_mismatch": size_mismatch,
            "camera_image_missing": no_image,
            "archived_previous_masks": str(archived) if archived else None}


def _png_size(path: Path) -> Tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(head[20:24], "big"), int.from_bytes(head[16:20], "big")
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return im.shape[:2]


# ------------------------------------------------------------ images_variable
def _image_names(d: Path) -> List[str]:
    """PNG names in ``d`` (none if it does not exist), ignoring interrupted ``*.tmp.png`` writes."""
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.png") if not p.name.endswith(".tmp.png"))


def build_variable_images(root: PathLike, reuse_from: Sequence[PathLike] = (),
                          recipe: VariableRecipe = VariableRecipe(),
                          archive_existing: bool = True, progress: bool = True) -> Dict[str, object]:
    """
    ``root/images_variable/<name>`` for every ``root/images/<name>`` that has a
    mask: RGB = an original variable version from ``reuse_from`` (same name and
    size) if one exists, else ``make_variable``; alpha = ``root/masks/<name>``.
    Writes ``images_variable_manifest.csv`` (source and tone parameters per file).
    """
    root = Path(root)
    img_dir, msk_dir, out_dir = root / "images", root / "masks", root / "images_variable"
    # Check the inputs BEFORE touching images_variable/ (v0p7 archived it and then wrote
    # nothing when images/ was still empty).
    names = _image_names(img_dir)
    if not names:
        raise FileNotFoundError(
            f"{img_dir} has no images, so images_variable/ was left untouched.  Regenerate images/ first "
            f"(notebook 02, step 1: mppp.mask.dataset.regenerate_images_from_pds).")
    with_mask = [n for n in names if (msk_dir / n).is_file()]
    if not with_mask:
        raise FileNotFoundError(f"none of the {len(names)} images in {img_dir} has a mask in {msk_dir}; "
                                f"images_variable/ was left untouched.")
    n_masks = len(_image_names(msk_dir))
    if n_masks and len(with_mask) < n_masks:
        import warnings
        warnings.warn(f"only {len(with_mask)} of {n_masks} masks have an image in {img_dir.name}/ — is the "
                      f"regeneration finished?  images_variable/ will cover only those.")
    archived = _archive_dir(out_dir) if archive_existing else None
    out_dir.mkdir(parents=True, exist_ok=True)
    reuse_dirs = [Path(p) for p in reuse_from]
    rows, missing_mask, bad = [], [], []
    for k, name in enumerate(names, 1):
        mpath = msk_dir / name
        if not mpath.is_file():
            missing_mask.append(name)
            continue
        im = cv2.imread(str(img_dir / name), cv2.IMREAD_UNCHANGED)
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if im is None or m is None or im.shape[:2] != m.shape:
            bad.append(name)
            continue
        bgr = im[..., :3]
        # Valid = non-zero RGB (MPPP writes exactly 0 for invalid pixels).  NOT the alpha
        # channel: in v6 alpha was the valid mask, but in v7 images/ it is the training
        # mask, and using it would black out the rover in every synthesized image.
        valid = bgr.max(axis=2) > 0
        src, params = None, {}
        for d in reuse_dirs:
            if not (d / name).is_file():
                continue
            old = cv2.imread(str(d / name), cv2.IMREAD_UNCHANGED)
            if old is not None and old.shape[:2] == m.shape:
                out_bgr, src = old[..., :3], str(d)
                break
        if src is None:
            rgb, params = make_variable(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), valid, name, recipe)
            out_bgr, src = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "synthesized"
        alpha = np.where(m > 127, 255, 0).astype(np.uint8)
        cv2.imwrite(str(out_dir / name), np.dstack([out_bgr, alpha]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        rows.append({"name": name, "source": src, **{k2: f"{v:.5f}" for k2, v in params.items()}})
        if progress and k % 250 == 0:
            print(f"  images_variable {k}/{len(names)}")
    fields = ["name", "source", "lo", "hi", "gamma", "gain_r", "gain_g", "gain_b"]
    with (root / "images_variable_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_reused = sum(r["source"] != "synthesized" for r in rows)
    return {"images": len(names), "written": len(rows), "reused_original": n_reused,
            "synthesized": len(rows) - n_reused, "images_without_mask": missing_mask,
            "unreadable_or_size_mismatch": bad, "recipe": asdict(recipe),
            "archived_previous": str(archived) if archived else None}


def build_training_set(root: PathLike, psx: Optional[PathLike] = None, reuse_variable_from: Sequence[PathLike] = (),
                       recipe: VariableRecipe = VariableRecipe()) -> Dict[str, object]:
    """
    1. (optional) export ``masks/`` from the Metashape project ``psx``;
    2. rebuild ``images_variable/``;
    3. write ``build_report.json`` next to them.
    ``images/`` is never modified.
    """
    root = Path(root)
    report: Dict[str, object] = {"root": str(root), "created": _dt.datetime.now().isoformat(timespec="seconds")}
    if psx is not None:
        report["masks"] = export_masks_from_psx(psx, root / "images", root / "masks")
        print(f"[dataset] masks: {report['masks']['written']} written from {Path(psx).name}")
    report["images_variable"] = build_variable_images(root, reuse_variable_from, recipe)
    iv = report["images_variable"]
    print(f"[dataset] images_variable: {iv['written']} written ({iv['reused_original']} original, "
          f"{iv['synthesized']} synthesized); {len(iv['images_without_mask'])} images have no mask")
    (root / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def render_check(root: PathLike, names: Sequence[str], tile_h: int = 260) -> np.ndarray:
    """BGR panel per frame: images | images_variable | mask outline (green) on images."""
    root = Path(root)
    rows = []
    for n in names:
        std = cv2.imread(str(root / "images" / n), cv2.IMREAD_COLOR)
        var = cv2.imread(str(root / "images_variable" / n), cv2.IMREAD_COLOR)
        m = cv2.imread(str(root / "masks" / n), cv2.IMREAD_GRAYSCALE)
        ov = std.copy()
        cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cs, -1, (60, 200, 60), max(2, std.shape[1] // 400))
        ov[m <= 127] = (ov[m <= 127] * 0.45).astype(np.uint8)          # darken excluded pixels
        w = int(std.shape[1] * tile_h / std.shape[0])
        tiles = [cv2.resize(t, (w, tile_h), interpolation=cv2.INTER_AREA) for t in (std, var, ov)]
        sep = np.full((tile_h, 6, 3), 255, np.uint8)
        body = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]])
        lab = np.full((20, body.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, f"{n}   images | images_variable | mask (dark = excluded)", (2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (78, 81, 82), 1, cv2.LINE_AA)
        rows.append(np.vstack([lab, body]))
    width = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    return np.vstack(sum([[r, np.full((6, width, 3), 255, np.uint8)] for r in rows], []))


# ------------------------------------------------------- images/ from PDS
REGEN_MARKER = "_mppp_regen.json"


def regen_config(config: Optional[Union[PathLike, Dict]] = None) -> Dict:
    """
    MPPP config used to regenerate training images: the standard pipeline (the
    same radiance -> 8-bit mapping that ``MPPPImage._infer_mask`` feeds the
    network) with the geometry of the Metashape masks — padded to the full
    detector frame, not undistorted — and no mask inference (no circularity).
    """
    from ..config import load_config
    cfg = load_config(config)
    cfg["masking"]["infer_mask"] = False
    cfg["masking"]["static_masks_dir"] = None
    cfg["color"]["apply_gamma"] = False             # the network's input is linear 8-bit
    cfg["resize"].update(apply_padding=True, extra_padding=False, undistort=False)
    return cfg


def index_pds(pds_dir: PathLike) -> Dict[str, Path]:
    """``{STEM (upper case): path}`` for every ``*.IMG`` below ``pds_dir`` (first found wins)."""
    import os
    idx: Dict[str, Path] = {}
    for dirpath, _dirs, files in os.walk(pds_dir):
        for f in files:
            if f[-4:].upper() == ".IMG":
                idx.setdefault(f[:-4].upper(), Path(dirpath) / f)
    return idx


def find_pds(stem: str, index: Dict[str, Path]) -> Tuple[Optional[Path], str]:
    """(path, status): the exact product, else the highest version of the same product."""
    key = stem.upper()
    if key in index:
        return index[key], "ok"
    if len(key) >= 54:
        base = key[:52]
        cands = sorted(k for k in index if k.startswith(base) and len(k) == len(key))
        if cands:
            return index[cands[-1]], f"version_substituted:{cands[-1][52:]}"
    return None, "missing"


_REGEN_STATE: Dict[str, object] = {}


def _regen_init(cfg: Dict, waypoints: Optional[Dict]) -> None:
    _REGEN_STATE["cfg"], _REGEN_STATE["waypoints"] = cfg, waypoints


def _regen_one(task: Tuple[str, str, str, str, Optional[str]]) -> Dict[str, object]:
    """Worker: one mask -> one 8-bit image.  Never raises (the error goes into the log)."""
    import time
    import warnings
    from ..image import MPPPImage
    mask_path, img_path, out_path, status, alpha = task
    t = time.time()
    rec = {"name": Path(mask_path).name, "status": status, "source": img_path, "width": "", "height": "",
           "seconds": "", "note": ""}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = MPPPImage(img_path, _REGEN_STATE["cfg"], _REGEN_STATE["waypoints"])
        rgb = im.image_int8
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        rec["width"], rec["height"] = rgb.shape[1], rgb.shape[0]
        if m is None or m.shape != rgb.shape[:2]:
            rec["status"] = "size_mismatch"
            rec["note"] = f"mask {None if m is None else (m.shape[1], m.shape[0])} vs image {(rgb.shape[1], rgb.shape[0])}"
            return rec
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if alpha == "mask":
            bgr = np.dstack([bgr, np.where(m > 127, 255, 0).astype(np.uint8)])
        elif alpha == "valid":
            bgr = np.dstack([bgr, im.mask_valid])
        tmp = Path(out_path).with_suffix(".tmp.png")
        if not cv2.imwrite(str(tmp), bgr):
            raise OSError(f"cv2.imwrite failed for {tmp}")
        tmp.replace(out_path)                            # atomic: an interrupted run leaves no half file
    except Exception as e:                               # noqa: BLE001
        rec["status"] = "error"
        rec["note"] = f"{type(e).__name__}: {e}"[:300]
    rec["seconds"] = round(time.time() - t, 2)
    return rec


def regenerate_images_from_pds(root: PathLike, pds_dir: PathLike, masks_dir: str = "masks",
                               images_dir: str = "images", waypoints: Optional[Dict] = None,
                               config: Optional[Union[PathLike, Dict]] = None, alpha: Optional[str] = "mask",
                               workers: int = 1, limit: Optional[int] = None,
                               index: Optional[Dict[str, Path]] = None, print_every: int = 100
                               ) -> Dict[str, object]:
    """
    For every ``root/masks_dir/<stem>.png`` find ``<stem>.IMG`` under ``pds_dir``
    and write ``root/images_dir/<stem>.png``: the MPPP 8-bit image (see
    :func:`regen_config`), RGB + ``alpha`` ("mask" = the training mask, the v7
    convention; "valid" = valid pixels; None = RGB only).

    * A same-product file with a different version is used when the exact one
      is missing (logged as ``version_substituted``).
    * The result must have the mask's size, else it is not written
      (``size_mismatch``).
    * Resumable: ``images_dir`` gets a marker file; rerunning skips images
      already written.  An existing ``images_dir`` WITHOUT the marker (not
      made by this function) is renamed to ``<images_dir>_prev_<time>`` first.
    * ``workers`` > 1 uses separate processes (~1.5 GB RAM each for 5120 px
      frames).  ``limit``: only the first N masks (for a trial run).

    Writes ``<images_dir>/_mppp_regen_log.csv`` (one row per mask) and returns
    counts by status.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    from .. import VERSION_TAG
    from ..waypoints import load_waypoints

    root = Path(root)
    mdir, odir = root / masks_dir, root / images_dir
    masks = sorted(p for p in mdir.iterdir() if p.suffix.lower() == ".png")
    if limit:
        masks = masks[:int(limit)]
    archived = None
    if odir.is_dir() and not (odir / REGEN_MARKER).is_file():
        archived = _archive_dir(odir)
    odir.mkdir(parents=True, exist_ok=True)
    cfg = regen_config(config)
    marker = odir / REGEN_MARKER
    if not marker.is_file():
        marker.write_text(json.dumps({"mppp_version": VERSION_TAG, "pds_dir": str(pds_dir), "alpha": alpha,
                                      "created": _dt.datetime.now().isoformat(timespec="seconds"),
                                      "config": cfg}, indent=1, default=str), encoding="utf-8")
    if waypoints is None:
        waypoints = load_waypoints()

    t0 = time.time()
    if index is None:
        print(f"[regen] indexing {pds_dir} ...")
        index = index_pds(pds_dir)
        print(f"[regen] {len(index)} .IMG files indexed in {time.time() - t0:.0f} s")

    log_path = odir / "_mppp_regen_log.csv"
    fields = ["name", "status", "source", "width", "height", "seconds", "note"]
    new_log = not log_path.is_file()
    counts: Dict[str, int] = {}
    tasks, done_before = [], 0
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new_log:
            w.writeheader()
        for m in masks:
            out = odir / m.name
            if out.is_file():
                done_before += 1
                continue
            src, status = find_pds(m.stem, index)
            if src is None:
                w.writerow({"name": m.name, "status": "missing", "source": "", "width": "", "height": "",
                            "seconds": "", "note": f"no {m.stem}.IMG under {pds_dir}"})
                counts["missing"] = counts.get("missing", 0) + 1
                continue
            tasks.append((str(m), str(src), str(out), status, alpha))
        fh.flush()
        print(f"[regen] {len(masks)} masks: {done_before} already done, {len(tasks)} to process, "
              f"{counts.get('missing', 0)} without a PDS product; workers={workers}")

        def consume(results):
            t1 = time.time()
            for k, rec in enumerate(results, 1):
                w.writerow(rec)
                s = rec["status"].split(":")[0]
                counts[s] = counts.get(s, 0) + 1
                if k % print_every == 0 or k == len(tasks):
                    fh.flush()
                    rate = (time.time() - t1) / k
                    print(f"[regen] {k}/{len(tasks)}  {rate:.1f} s/image  ETA {rate * (len(tasks) - k) / 60:.0f} min  "
                          f"{ {c: n for c, n in sorted(counts.items())} }")

        if workers and workers > 1 and tasks:
            with ProcessPoolExecutor(int(workers), initializer=_regen_init, initargs=(cfg, waypoints)) as ex:
                consume(ex.map(_regen_one, tasks, chunksize=4))
        else:
            _regen_init(cfg, waypoints)
            consume(map(_regen_one, tasks))
    counts["already_done"] = done_before
    res = {"root": str(root), "images_dir": str(odir), "archived_previous_images": str(archived) if archived else None,
           "log": str(log_path), "counts": counts, "minutes": round((time.time() - t0) / 60, 1)}
    print(f"[regen] finished in {res['minutes']} min: {counts}  (details: {log_path.name})")
    return res
