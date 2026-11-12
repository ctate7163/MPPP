"""
mppp.ibr.photometry - per-image photometric angle maps from the station's surface (plan of 29 Oct 2026).

For every pixel with a surface: the **incidence** i = acos(n . s) (the surface and the Sun), the **emission**
e = acos(n . -r) (the surface and the camera; geometry only, the same at any time of day) and the **phase**
g = acos(-r . s) (the view ray and the Sun; it needs no normal), all in degrees in the image's camera frame, with
r the unit pixel ray, n the unit normal facing the camera and s the unit vector to the Sun.  **Shadow**: a ray from
the surface point towards the Sun that meets the mesh - i < 90 degrees alone does not mean lit.

The Sun per image comes from the project rows (``sun_enu`` from the label's SOLAR_AZIMUTH / ELEVATION), turned into
the block frame with the block's rotation away from label ENU (``R_block_from_enu`` of the station, from
``error_input/block_frame.json`` or :func:`mppp.sun.block_rotation`).

Output per image: ``photometry/<image>.npz`` (incidence, emission, phase as float16 degrees, shadow uint8:
0 unknown, 1 lit, 2 shadow) - the **photometric cube** of the station (Christian, 9 Nov: "the per image depth maps
and photometric cube from the cleaned up 3d mesh") - and, for COLMAP-aware tools, the same as
``photometry_maps/<image>.{incidence,emission,phase}.bin`` in COLMAP's dense-map format.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from .raycast import LABEL_MESH, Scene

log = logging.getLogger("mppp.ibr.photometry")
PathLike = Union[str, Path]
SHADOW_UNKNOWN, SHADOW_LIT, SHADOW_SHADOW = 0, 1, 2


def angles(dirs_cam: np.ndarray, normal_cam: np.ndarray, sun_cam: Sequence[float], valid: np.ndarray) -> Dict[str, np.ndarray]:
    """i, e, g in degrees (NaN where ``valid`` is False) from the unit pixel rays ``dirs_cam`` (..., 3), the unit
    normals ``normal_cam`` (..., 3, facing the camera) and the unit Sun vector ``sun_cam`` (3,), all camera frame."""
    s = np.asarray(sun_cam, float)
    s = s / np.linalg.norm(s)
    r = np.asarray(dirs_cam, float)
    n = np.asarray(normal_cam, float)
    cos_i = np.clip(np.sum(n * s, axis=-1), -1, 1)
    cos_e = np.clip(np.sum(n * -r, axis=-1), -1, 1)
    cos_g = np.clip(np.sum(-r * s, axis=-1), -1, 1)
    i, e, g = (np.degrees(np.arccos(c)) for c in (cos_i, cos_e, cos_g))
    bad = ~np.asarray(valid, bool)
    for a in (i, e):
        a[bad] = np.nan
    g = g.astype(float)
    g[bad] = np.nan
    return {"incidence": i, "emission": e, "phase": g}


def shadow_map(scene: Scene, X_world: np.ndarray, sun_world: Sequence[float], valid: np.ndarray, lift_m: float = 0.02) -> np.ndarray:
    """0 unknown / 1 lit / 2 shadow per pixel: a ray from the surface point (lifted ``lift_m`` along the Sun
    direction, off its own triangle) towards the Sun that hits the mesh means shadow."""
    s = np.asarray(sun_world, float)
    s = s / np.linalg.norm(s)
    out = np.zeros(np.asarray(valid).shape, np.uint8)
    idx = np.flatnonzero(np.asarray(valid).reshape(-1))
    if not len(idx):
        return out
    X = np.asarray(X_world, float).reshape(-1, 3)[idx] + lift_m * s
    hit = scene.cast(X, np.broadcast_to(s, (len(idx), 3)))
    flat = out.reshape(-1)
    flat[idx] = np.where(np.isfinite(hit["t"]) & (hit["label"] == LABEL_MESH), SHADOW_SHADOW, SHADOW_LIT)
    return out


def photometric_maps(scene: Scene, maps: Dict[str, np.ndarray], R: np.ndarray, C: np.ndarray, sun_world: Sequence[float],
                     with_shadow: bool = True) -> Dict[str, np.ndarray]:
    """The i / e / g (and shadow) maps of one image from its cast ``maps`` (:func:`mppp.ibr.raycast.cast_image`:
    ``dirs_cam``, ``normal``, ``range``, ``label``), the world-to-camera rotation ``R``, the centre ``C`` and the
    Sun in the block frame."""
    R = np.asarray(R, float)
    valid = maps["label"] == LABEL_MESH
    sun_cam = R @ (np.asarray(sun_world, float) / np.linalg.norm(sun_world))
    out = angles(maps["dirs_cam"], maps["normal"], sun_cam, valid)
    if with_shadow:
        d_world = maps["dirs_cam"].reshape(-1, 3) @ R                       # R^T d
        X = np.asarray(C, float) + maps["range"].reshape(-1, 1) * d_world
        out["shadow"] = shadow_map(scene, X, sun_world, valid)
    return out


def sun_for_image(row: Dict[str, Any], R_block_from_enu: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """The Sun in the block frame for a project row (``sun_enu``, or ``solar_azimuth_deg`` / ``solar_elevation_deg``)."""
    from ..sun import sun_in_block, sun_vector_enu
    s = row.get("sun_enu")
    if s is None:
        s = sun_vector_enu(row.get("solar_azimuth_deg"), row.get("solar_elevation_deg"))
    if s is None:
        return None
    Rb = np.eye(3) if R_block_from_enu is None else np.asarray(R_block_from_enu, float)
    return sun_in_block(s, Rb)


def write_colmap_array(path: PathLike, a: np.ndarray) -> Path:
    """COLMAP's dense-map format (``width&height&channels&`` then float32, column-major); NaN written as 0."""
    a = np.asarray(a, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    h, w, c = a.shape
    a = np.nan_to_num(a, nan=0.0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(f"{w}&{h}&{c}&".encode("ascii"))
        f.write(np.transpose(a, (1, 0, 2)).astype(np.float32).tobytes(order="F"))
    return p


def save_photometry(path: PathLike, pm: Dict[str, np.ndarray], **meta) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, incidence=np.nan_to_num(pm["incidence"], nan=-1).astype(np.float16),
                        emission=np.nan_to_num(pm["emission"], nan=-1).astype(np.float16),
                        phase=np.nan_to_num(pm["phase"], nan=-1).astype(np.float16),
                        shadow=pm.get("shadow", np.zeros(pm["phase"].shape, np.uint8)).astype(np.uint8),
                        meta=np.array(json.dumps(meta, default=float)))
    return p


def load_photometry(path: PathLike) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k].astype(np.float32) for k in ("incidence", "emission", "phase")}
    for k in out:
        out[k][out[k] < 0] = np.nan
    out["shadow"] = z["shadow"]
    out["meta"] = json.loads(str(z["meta"])) if "meta" in z else {}
    return out


def preview_png(pm: Dict[str, np.ndarray], out_prefix: PathLike) -> list:
    """``<prefix>_incidence.png`` (0-90 degrees), ``_emission.png`` (0-90), ``_phase.png`` (0-180), ``_shadow.png``."""
    import cv2
    outs = []
    for k, vmax in (("incidence", 90.0), ("emission", 90.0), ("phase", 180.0)):
        a = pm[k]
        v = np.clip(np.nan_to_num(a, nan=0.0) / vmax, 0, 1)
        img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        img[~np.isfinite(a)] = 0
        p = Path(str(out_prefix) + f"_{k}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        outs.append(p)
    if "shadow" in pm:
        sh = pm["shadow"]
        img = np.zeros(sh.shape + (3,), np.uint8)
        img[sh == SHADOW_LIT] = (90, 200, 255)
        img[sh == SHADOW_SHADOW] = (120, 40, 20)
        p = Path(str(out_prefix) + "_shadow.png")
        cv2.imwrite(str(p), img)
        outs.append(p)
    return outs


def preview_jpg(pm: Dict[str, np.ndarray], out_prefix: PathLike, rad_image: Optional[PathLike] = None) -> list:
    """0.97.0 (Christian, 12 Nov 2026: "In the photometry outputs, change png to jpg to save space and include an 8bit
    rad version of the other products for easy comparison"): ``<prefix>_incidence.jpg`` (0-90 degrees),
    ``_emission.jpg`` (0-90), ``_phase.jpg`` (0-180), ``_shadow.jpg`` and ``<prefix>_rad.jpg`` - the block's 8-bit RAD
    frame (``rad_image``) on the maps' pixel grid."""
    import cv2
    from .raycast import JPEG_QUALITY
    q = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    Path(str(out_prefix)).parent.mkdir(parents=True, exist_ok=True)
    outs = []
    shape = pm["incidence"].shape[:2]
    if rad_image is not None and Path(rad_image).is_file():
        im = cv2.imread(str(rad_image), cv2.IMREAD_COLOR)
        if im is not None:
            if im.shape[:2] != shape:
                im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            p = Path(str(out_prefix) + "_rad.jpg")
            cv2.imwrite(str(p), im, q)
            outs.append(p)
    for k, vmax in (("incidence", 90.0), ("emission", 90.0), ("phase", 180.0)):
        a = pm[k]
        v = np.clip(np.nan_to_num(a, nan=0.0) / vmax, 0, 1)
        img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        img[~np.isfinite(a)] = 0
        p = Path(str(out_prefix) + f"_{k}.jpg")
        cv2.imwrite(str(p), img, q)
        outs.append(p)
    if "shadow" in pm:
        sh = pm["shadow"]
        img = np.zeros(sh.shape + (3,), np.uint8)
        img[sh == SHADOW_LIT] = (90, 200, 255)
        img[sh == SHADOW_SHADOW] = (120, 40, 20)
        p = Path(str(out_prefix) + "_shadow.jpg")
        cv2.imwrite(str(p), img, q)
        outs.append(p)
    return outs


def photometric_cube(project, rec, station, scene: Scene, maps_dir: PathLike, out_dir: PathLike, colmap_bins: bool = True,
                     previews: bool = True, with_shadow: bool = True) -> Dict[str, Any]:
    """The cube of the station: i / e / g / shadow for every image with maps in ``maps_dir`` (``<image>.npz`` from
    :func:`mppp.ibr.raycast.cast_station`), the Sun per image from the project rows.  Writes ``photometry/<image>.npz``,
    the COLMAP ``.bin`` maps, previews and ``photometry_report.json``."""
    from .raycast import load_maps, pixel_rays
    out = Path(out_dir)
    rows_by_name = {r["name"]: r for r in project.images}
    by_name = {im.name: im for im in rec.images.values()}
    rows = []
    for f in sorted(Path(maps_dir).glob("*.npz")):
        m = load_maps(f)
        name = m["meta"].get("image") or f.stem
        im = by_name.get(name)
        row = rows_by_name.get(name)
        if im is None or row is None:
            rows.append({"image": name, "error": "not in the model / project"})
            continue
        sun = sun_for_image(row, station.R_block_from_enu)
        if sun is None:
            rows.append({"image": name, "error": "no Sun vector in the project row (sun_enu)"})
            continue
        cam = rec.cameras[im.camera_id]
        cw = im.cam_from_world()
        R = np.asarray(cw.rotation.matrix(), float)
        C = np.asarray(im.projection_center(), float)
        scale = float(m["meta"].get("scale", 1.0))
        d_cam, _, _ = pixel_rays(cam, R, scale)
        m["dirs_cam"] = d_cam
        pm = photometric_maps(scene, m, R, C, sun, with_shadow=with_shadow)
        stem = Path(name).stem
        save_photometry(out / "photometry" / f"{stem}.npz", pm, image=name, scale=scale, sun_block=[float(v) for v in sun],
                        sun_enu=row.get("sun_enu"), lmst=row.get("lmst"), sol=row.get("sol"))
        if colmap_bins:
            for k in ("incidence", "emission", "phase"):
                write_colmap_array(out / "photometry_maps" / f"{name}.{k}.bin", pm[k])
        if previews:                                 # 0.97.0: jpg, beside the depth and normal pictures
            from .raycast import PREVIEW_DIR
            preview_jpg(pm, out / PREVIEW_DIR / stem, rad_image=Path(project.root) / "images" / name)
        ok = np.isfinite(pm["emission"])
        rows.append({"image": name, "sol": row.get("sol"), "lmst": row.get("lmst"), "sun_block": [round(float(v), 4) for v in sun],
                     "pixels": int(ok.sum()),
                     "incidence_deg": _stats(pm["incidence"][ok]), "emission_deg": _stats(pm["emission"][ok]),
                     "phase_deg": _stats(pm["phase"][ok]),
                     "shadow_fraction": float((pm["shadow"] == SHADOW_SHADOW).sum() / max(1, ok.sum())) if "shadow" in pm else None})
        log.info("photometry: %s i %s e %s g %s", name, rows[-1]["incidence_deg"]["median"], rows[-1]["emission_deg"]["median"],
                 rows[-1]["phase_deg"]["median"])
    rep = {"station": station.label, "n_images": len(rows), "images": rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "photometry_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    return rep


def _stats(a: np.ndarray) -> Dict[str, Optional[float]]:
    if not len(a):
        return {"median": None, "p10": None, "p90": None}
    return {"median": round(float(np.median(a)), 2), "p10": round(float(np.quantile(a, 0.1)), 2), "p90": round(float(np.quantile(a, 0.9)), 2)}
