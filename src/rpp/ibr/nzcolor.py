"""
rpp.ibr.nzcolor - the color conversion between the Navcams and Mastcam-Z, from frames taken close in time (0.99.13).

Christian, 17 Nov 2026: "what is the approximate color conversion between zcam and ncam? estimate the white balances and
apply it to ncam and hazcams so that the ecams and zcams look like they are in more consistent color spaces."

In an N+Z block (``mars2020_sol_*_zcam``), a 3D point seen by a Navcam frame and a Mastcam-Z frame taken within
``--max-dt-min`` of each other gives one sample of the radiance ratio per channel: both frames are read as the 16-bit
linear images of the panorama (``<block>/colmap/images16``, made from the PDS products where missing), each divided
by its own factors (``linear16.json``: white balance x brightness / zenith scale x scale_rad_to_int16), so

    r_c = L_Z,c / L_N,c        c = R, G, B         (L: the PDS radiance, the same sun for both)

at the point (a 5 x 5 median at each frame's keypoint; saturated, dark or empty pixels left out).  The median of r_c
over the samples is the conversion; the display ratio with the white balances in use is r_c x wb_Z,c / wb_N,c.  The
Navcam balance that makes the Navcams look like Mastcam-Z is

    wb_N,c' = wb_N,c x f_c,     f_c = (r_c wb_Z,c / wb_N,c) / (r_G wb_Z,G / wb_N,G)

(the color only: f_G = 1, the brightness kept; ``brightness_factor`` = r_G wb_Z,G / wb_N,G is reported).  The Hazcams
take the same f (their balance is the Navcam's times what matched them to the Navcams, 0.97.15).

    python -m rpp.ibr.nzcolor D:\\rover_blocks\\mars2020_sol_0413_threeforks_north_zcam [more blocks] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("rpp.ibr.nzcolor")
PathLike = Union[str, Path]
MAX_DT_MIN = 45.0
HALF = 2                                  # the 5 x 5 window
MIN_VALUE = 300                           # 16-bit units: darker samples are noise
MAX_VALUE = 60000
MAX_SAMPLES_PER_PAIR = 2000


def _sclk(r: Dict[str, Any]) -> Optional[int]:
    try:
        return int(str(r.get("sclk_key") or "")[:10])
    except ValueError:
        return None


def _family(r: Dict[str, Any]) -> str:
    g = str(r.get("camera_group") or r.get("instrument") or "")[:2].upper()
    return "N" if g in ("NL", "NR") else ("Z" if g in ("ZL", "ZR") else g[:1])


def pairs_of_block(rec, images: Dict[str, Dict[str, Any]], max_dt_s: float) -> List[Tuple[int, Tuple, Tuple]]:
    """[(point id, (N image id, x, y), (Z image id, x, y))] for the points a Navcam and a Mastcam-Z frame taken within
    ``max_dt_s`` of each other both see."""
    info = {}
    for iid, im in rec.images.items():
        r = images.get(im.name)
        if r is not None and _family(r) in ("N", "Z") and _sclk(r) is not None:
            info[iid] = (_family(r), _sclk(r))
    obs: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for iid, im in rec.images.items():
        if iid not in info:
            continue
        for p in im.points2D:
            if p.has_point3D():
                obs[p.point3D_id].append((iid, float(p.xy[0]), float(p.xy[1])))
    out, per_pair = [], defaultdict(int)
    for pid, o in obs.items():
        ns = [x for x in o if info[x[0]][0] == "N"]
        zs = [x for x in o if info[x[0]][0] == "Z"]
        for n in ns:
            for z in zs:
                if abs(info[n[0]][1] - info[z[0]][1]) <= max_dt_s and per_pair[(n[0], z[0])] < MAX_SAMPLES_PER_PAIR:
                    per_pair[(n[0], z[0])] += 1
                    out.append((pid, n, z))
    return out


def _sample(img: np.ndarray, sat: Optional[np.ndarray], x: float, y: float) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    c, r = int(round(x - 0.5)), int(round(y - 0.5))              # COLMAP: pixel centers at +0.5
    if c - HALF < 0 or r - HALF < 0 or c + HALF >= w or r + HALF >= h:
        return None
    win = img[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1].reshape(-1, 3).astype(np.float64)
    if sat is not None and sat[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1].any():
        return None
    if (win <= 0).any() or (win >= MAX_VALUE).any():
        return None
    v = np.median(win, axis=0)
    return None if v.min() < MIN_VALUE else v


def measure_block(block: PathLike, max_dt_min: float = MAX_DT_MIN, pds: Optional[PathLike] = None,
                  sparse: Optional[str] = None) -> Dict[str, Any]:
    """The samples of one N+Z block: {"block", "pairs", "samples": [[rN, gN, bN, rZ, gZ, bZ, sol_N, sol_Z, dt_s], ...]
    (radiance, the frames' factors divided out), "wb": {"N": ..., "Z": ...}, "missing": {...}}."""
    import cv2
    import pycolmap
    from ..sfm.project import SfmProject
    from . import linear16 as L
    root = Path(block)
    proj = SfmProject.load(root / "colmap" if (root / "colmap" / "project.json").is_file() else root)
    images = {r["name"]: r for r in proj.images}
    sp = Path(proj.root) / "sparse"
    cands = [sp / sparse] if sparse else [sp / "cahv_ba", sp / "0"] + sorted(p for p in sp.glob("*") if p.is_dir())
    model = next((c for c in cands if (c / "images.bin").is_file() and (c / "points3D.bin").is_file()
                  and (c / "points3D.bin").stat().st_size > 1000), None)
    if model is None:
        raise FileNotFoundError(f"{sp}: no model with points")
    rec = pycolmap.Reconstruction(str(model))
    prs = pairs_of_block(rec, images, 60.0 * max_dt_min)
    need = sorted({rec.images[n[0]].name for _, n, _ in prs} | {rec.images[z[0]].name for _, _, z in prs})
    got = L.ensure_linear16(proj, need, pds=pds)
    frames = got.get("frames") or {}
    cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]] = {}

    def load(name: str):
        if name not in cache:
            f, s = L.files(proj, name)
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            sat = cv2.imread(str(s), cv2.IMREAD_GRAYSCALE) if s.is_file() else None
            fac = (frames.get(name) or {}).get("to_radiance_divide_by")
            cache[name] = (None if img is None else img[..., ::-1], None if sat is None else sat > 0,
                           None if fac is None else np.asarray(fac, float))
        return cache[name]

    rows, wb = [], {}
    for _, n, z in prs:
        nn, zn = rec.images[n[0]].name, rec.images[z[0]].name
        (iN, sN, fN), (iZ, sZ, fZ) = load(nn), load(zn)
        if iN is None or iZ is None or fN is None or fZ is None:
            continue
        a, b = _sample(iN, sN, n[1], n[2]), _sample(iZ, sZ, z[1], z[2])
        if a is None or b is None:
            continue
        wb.setdefault("N", (frames.get(nn) or {}).get("white_balance"))
        wb.setdefault("Z", (frames.get(zn) or {}).get("white_balance"))
        # 0.99.22: the ratio in the processed images too (columns 9-11: Z / N of the 16-bit values, which carry each
        # frame's balance, brightness and sun-angle scale) - what the eye compares in a panorama
        rows.append(list(a / fN) + list(b / fZ) + [images[nn].get("sol"), images[zn].get("sol"),
                                                   abs(_sclk(images[nn]) - _sclk(images[zn]))] + list(b / a))
    return {"block": str(root), "model": str(model), "pairs": len(prs), "samples": rows, "wb": wb,
            "frames": len(need), "missing": got.get("missing") or {}}


def summarize(blocks: Sequence[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The conversion over the samples of every block, and the balances it gives (the module's text)."""
    from ..config import default_config
    col = (config or default_config())["color"]
    wbN = np.asarray(col["white_balance_ecam"], float) * float((col.get("brightness_by_family") or {}).get("N", 1.0))
    wbZ = np.asarray(col["white_balance_zcam"], float) * float((col.get("brightness_by_family") or {}).get("Z", 1.0))
    wbH = np.asarray(col.get("white_balance_hazcam") or col["white_balance_ecam"], float)
    ncol = min((len(r) for b in blocks for r in b["samples"]), default=9)
    S = np.asarray([r[:ncol] for b in blocks for r in b["samples"]], float)
    out: Dict[str, Any] = {"samples": int(len(S)), "blocks": [{"block": b["block"], "samples": len(b["samples"]),
                                                               "pairs": b["pairs"], "frames": b["frames"],
                                                               "missing": len(b["missing"])} for b in blocks]}
    if not len(S):
        out["message"] = "no Navcam / Mastcam-Z samples within the time limit"
        return out
    ratio = S[:, 3:6] / S[:, 0:3]
    lr = np.log(ratio)
    med = np.exp(np.median(lr, axis=0))
    sd = 1.4826 * np.median(np.abs(lr - np.median(lr, axis=0)), axis=0)
    grp = np.asarray([f"{i}:{int(r[6])}:{int(r[7])}" for i, b in enumerate(blocks) for r in b["samples"]])
    keys = sorted(set(grp))
    members = {k: np.nonzero(grp == k)[0] for k in keys}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(200):                                        # by block and sol pair: one scene does not decide
        sel = np.concatenate([members[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        boot.append(np.exp(np.median(lr[sel], axis=0)))
    boot = np.asarray(boot)
    disp = med * wbZ / wbN
    f = disp / disp[1]
    if S.shape[1] >= 12:
        # 0.99.22: as processed - with the sun-angle (zenith) scale of each frame.  The radiance ratio above assumes
        # the same sun for both cameras: at Three Forks South the Z34 frames had the sun 8 deg higher, and 9 of the
        # 19 % by which their radiance exceeded the Navcam's was the sun (17 Nov: reported as a camera difference)
        proc = np.exp(np.median(np.log(S[:, 9:12]), axis=0))
        out["processed_ratio_Z_over_N"] = [round(float(x), 4) for x in proc]
        out["processed_note"] = ("Z / N of the processed 16-bit values (each frame's balance, brightness and sun-angle "
                                 "scale as it was processed): the difference a panorama shows")
    out.update({
        "radiance_ratio_Z_over_N": [round(float(x), 4) for x in med],
        "radiance_ratio_sd_between_scenes": [round(float(x), 4) for x in boot.std(axis=0)],
        "scatter_sd_log": [round(float(x), 3) for x in sd],
        "chromaticity_Z_over_N": [round(float(x), 4) for x in med / med[1]],
        # the radiance ratio x the balances now in the configuration: no sun-angle scale in it (see processed_ratio)
        "display_ratio_Z_over_N_now": [round(float(x), 4) for x in disp],
        "navcam_factor": [round(float(x), 4) for x in f],
        "brightness_factor": round(float(disp[1]), 4),
        "white_balance_ecam_now": [float(x) for x in col["white_balance_ecam"]],
        "white_balance_ecam_matching_zcam": [round(float(x), 4) for x in np.asarray(col["white_balance_ecam"], float) * f],
        "white_balance_hazcam_now": [float(x) for x in wbH],
        "white_balance_hazcam_matching_zcam": [round(float(x), 4) for x in wbH * f],
        "white_balance_zcam": [float(x) for x in col["white_balance_zcam"]],
        "by_sol_pair": {}})
    pairs = defaultdict(list)
    for i, r in enumerate(S):
        pairs[f"N{int(r[6])}-Z{int(r[7])}"].append(i)
    for k, idx in sorted(pairs.items()):
        m = np.exp(np.median(lr[idx], axis=0))
        out["by_sol_pair"][k] = {"samples": len(idx), "chromaticity": [round(float(x), 4) for x in m / m[1]],
                                 "ratio": [round(float(x), 4) for x in m]}
    return out


def lines(s: Dict[str, Any]) -> List[str]:
    if not s.get("samples"):
        return [s.get("message", "no samples")]
    L = [f"Navcam / Mastcam-Z color, {s['samples']} samples (points both saw within the time limit) in "
         f"{len(s['blocks'])} block(s)",
         f"  radiance ratio Z / N (R, G, B): {s['radiance_ratio_Z_over_N']} (sd between scenes "
         f"{s['radiance_ratio_sd_between_scenes']}); chromaticity {s['chromaticity_Z_over_N']}",
         f"  radiance ratio x the balances in use (no sun-angle scale) Z / N: {s['display_ratio_Z_over_N_now']}",
         f"  Navcam color factor to look like Mastcam-Z: {s['navcam_factor']} (brightness Z / N {s['brightness_factor']}, "
         "not applied)",
         f"  color.white_balance_ecam:   {s['white_balance_ecam_now']} -> {s['white_balance_ecam_matching_zcam']}",
         f"  color.white_balance_hazcam: {s['white_balance_hazcam_now']} -> {s['white_balance_hazcam_matching_zcam']}"]
    if s.get("processed_ratio_Z_over_N"):
        L.insert(3, f"  in the processed images (balance, brightness and sun-angle scale as processed) Z / N: "
                    f"{s['processed_ratio_Z_over_N']} - the difference a panorama shows")
    for k, v in s["by_sol_pair"].items():
        L.append(f"    {k}: {v['samples']} samples, chromaticity {v['chromaticity']}")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("blocks", nargs="+", help="N+Z block folders (mars2020_sol_*_zcam)")
    ap.add_argument("--max-dt-min", type=float, default=MAX_DT_MIN, help=f"the frames' time apart (default {MAX_DT_MIN:g} min)")
    ap.add_argument("--pds", default=None, help="the PDS archive for the 16-bit frames that are missing (default RPP_PDS)")
    ap.add_argument("--out", default=None, help="the result as JSON (default <first block>/nz_color.json)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    res = []
    for b in a.blocks:
        try:
            r = measure_block(b, a.max_dt_min, a.pds)
            print(f"{b}: {len(r['samples'])} samples from {r['pairs']} point pairs, {r['frames']} frames"
                  + (f", {len(r['missing'])} without a 16-bit image" if r["missing"] else ""), flush=True)
            res.append(r)
        except Exception as e:                                   # noqa: BLE001 - the other blocks still count
            print(f"{b}: not used ({type(e).__name__}: {e})", flush=True)
    s = summarize(res)
    out = Path(a.out) if a.out else Path(a.blocks[0]) / "nz_color.json"
    out.write_text(json.dumps(dict(s, raw=[{k: v for k, v in r.items() if k != "samples"} for r in res]), indent=1),
                   encoding="utf-8")
    print("\n".join(lines(s)))
    print(f"written: {out}")
    return 0 if s.get("samples") else 1


if __name__ == "__main__":
    raise SystemExit(main())
