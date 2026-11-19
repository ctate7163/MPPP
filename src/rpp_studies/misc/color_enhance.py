"""Three automatic display transforms of a 16-bit linear panorama (0.99.33; Christian, 18 Nov 2026: "Absent hiring a
graphic artist ... what kind of automatic processes can we try? ... I need to explain the process to scientists in a way
that doesn't sound like making up data"; "Prepare examples of each three color maps made from the highest res 16 bit
images"; "The decorrelation stretch should exclude the sky and rover").

Every method is a fixed, deterministic function of the calibrated linear blend (``*_ods_left_16.png``: value =
radiance x white balance x scale, see ``ods.json`` "linear16"), with one parameter set per station taken from the
station's terrain band (elevation -30 ... -2 deg) and written to ``<name>_parameters.json``.  Nothing is learned and
nothing is painted.

* reference: gamma 2.2 of linear / white (white: the band's 99.9 % luminance).
* A base/detail (Durand and Dorsey 2002): log luminance split by an edge-preserving guided filter (He et al. 2010)
  into a large-scale base and the detail; the base compressed (never expanded) so the band's 0.5-99.5 % span fits a
  contrast of ``A_CONTRAST``, the detail multiplied by ``A_DETAIL``; each pixel's R:G:B ratios kept (to the power
  ``A_RATIO_POWER``).
* B Mantiuk et al. 2006 (OpenCV ``TonemapMantiuk``): the image's contrasts mapped through a model of visible contrast
  and the picture solved for; then the band's 99.5 % luminance set to 0.95.
* C decorrelation stretch (Gillespie et al. 1986): the principal components of the three bands (in the gamma view,
  over the terrain only) given equal spread ``C_SIGMA``; the sky and the rover are left out of the statistics and
  shown as in the reference view -
    sky:   beyond the mesh (label "far" of the station's center depth), brighter than halfway (log luminance)
           between the terrain just below the horizon and the sky above 15 deg elevation, connected to the zenith
           (mesh pixels within 1 deg of the far surface may be sky too: the mesh's silhouette is a little fat); in each
           column it runs from the zenith down to its first non-sky pixel, and its rim is grown 2 px (at a quarter of
           the width) so no colored fringe is left.  The far hills are as dark as the near terrain: they stay terrain;
    rover: the rover model's pixels of the rover run's center depth ("rover" layer), grown by ``ROVER_GROW_DEG`` (an
           eye of the stereo pair sees a point 0.65 m off up to 2.8 deg from where the center sees it).

usage::

    python -m rpp_studies.misc.color_enhance PANORAMA_16.png OUT_DIR NAME [--depth CENTER_DEPTH.npz]
                                            [--rover-depth ROVER_CENTER_DEPTH.npz]

writes ``NAME_pano_<method>.jpg`` (a quarter of the width), ``NAME_crop_<method>.jpg`` (4096 columns at full
resolution, where the band has the most fine contrast), ``NAME_sheet.jpg``, ``NAME_C_mask_sky_blue_rover_green.png``
and ``NAME_parameters.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

BAND_DEG = (-30.0, -2.0)
A_CONTRAST, A_DETAIL, A_RADIUS_DEG, A_EPS, A_RATIO_POWER, A_MID = 8.0, 1.8, 1.0, 0.04, 1.15, 0.30
B_SCALE, B_SAT = 1.6, 1.3
C_SIGMA = 0.075
SKY_REF_EL_DEG, ROVER_GROW_DEG, SKY_GROW_PX = 15.0, 3.0, 2          # (the last at a quarter of the width)
SKY_RIM_DEG = 1.0                    # mesh pixels this close to the far surface may be sky (the mesh's fat silhouette)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)               # RGB


def guided_filter(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. 2010, the image as its own guide (box filters; no opencv-contrib needed)."""
    import cv2
    k = (2 * int(r) + 1, 2 * int(r) + 1)
    m = cv2.blur(I, k)
    v = cv2.blur(I * I, k) - m * m
    a = v / (v + eps)
    b = m - a * m
    return cv2.blur(a, k) * I + cv2.blur(b, k)


def lum(rgb: np.ndarray) -> np.ndarray:
    return np.maximum(rgb @ LUMA, 1e-6)


def masks(small: np.ndarray, full_hw: Tuple[int, int], depth: Optional[str], rover_depth: Optional[str],
          info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """(sky, rover) at full resolution (bool) - see the module.  ``small``: the panorama at a quarter of the width
    (RGB float)."""
    import cv2
    H, W = full_hw
    h4, w4 = small.shape[:2]
    sky = np.zeros((H, W), bool)
    rover = np.zeros((H, W), bool)
    if depth:
        with np.load(depth) as z:
            far = cv2.resize((z["label"] == 2).astype(np.uint8), (w4, h4), interpolation=cv2.INTER_NEAREST) > 0
            mesh = cv2.resize((z["label"] == 1).astype(np.uint8), (w4, h4), interpolation=cv2.INTER_NEAREST) > 0
        L = np.log(np.maximum(small @ LUMA, 1.0))
        el4 = 90.0 - (np.arange(h4) + 0.5) * 180.0 / h4
        ok4 = small.min(axis=2) > 0
        high = far & (el4 > SKY_REF_EL_DEG)[:, None]
        L_sky = float(np.median(L[high])) if high.sum() > 1000 else float(np.percentile(L[far], 90))
        near = mesh & ((el4 > -10.0) & (el4 < 0.0))[:, None] & ok4
        L_ter = float(np.median(L[near])) if near.sum() > 1000 else float(np.percentile(L[ok4], 50))
        cut = 0.5 * (L_sky + L_ter)
        kr = 2 * max(1, int(round(SKY_RIM_DEG * w4 / 360.0))) + 1        # 0.99.34: 1 deg at any width (23 px at 4096)
        rim = mesh & (cv2.dilate(far.astype(np.uint8), np.ones((kr, kr), np.uint8)) > 0)
        like = ((far | rim) & (L > cut)).astype(np.uint8)
        like = cv2.morphologyEx(like, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, lab = cv2.connectedComponents(like, connectivity=4)
        top = np.unique(lab[:3][like[:3] > 0])
        s4 = np.isin(lab, top[top > 0])
        s4 = cv2.morphologyEx(s4.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8)) > 0     # thin seams
        s4 = np.cumprod(s4, axis=0).astype(np.uint8)            # each column: from the zenith down to its first non-sky
        s4 = cv2.dilate(s4, np.ones((2 * SKY_GROW_PX + 1, 2 * SKY_GROW_PX + 1), np.uint8)) & (far | rim).astype(np.uint8)
        sky = cv2.resize(s4, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        info.update(log_luminance_sky=L_sky, log_luminance_terrain=L_ter, log_luminance_cut=cut)
    if rover_depth:
        with np.load(rover_depth) as z:
            rv = (z["rover"] > 0).astype(np.uint8)
        if rv.shape != (H, W):
            rv = cv2.resize(rv, (W, H), interpolation=cv2.INTER_NEAREST)
        k = int(2 * round(ROVER_GROW_DEG * W / 360.0) + 1)
        rover = cv2.dilate(rv, np.ones((max(3, k // 3), k), np.uint8)) > 0
    return sky, rover


def run(src: str, out: str, name: str, depth: Optional[str] = None, rover_depth: Optional[str] = None) -> Dict[str, Any]:
    import cv2
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    im16 = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)          # BGR uint16, linear
    if im16 is None or im16.dtype != np.uint16:
        raise ValueError(f"{src}: not a 16-bit image")
    H, W = im16.shape[:2]
    ppd = H / 180.0
    el = 90.0 - (np.arange(H) + 0.5) / ppd
    band = (el > BAND_DEG[0]) & (el < BAND_DEG[1])
    sub = im16[band][::6, ::6, ::-1].astype(np.float32)
    ref = float(np.percentile(lum(sub)[sub.min(axis=2) > 0], 99.9))       # the linear value shown as white
    small = cv2.resize(im16, (W // 4, H // 4), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32)
    el_s = 90.0 - (np.arange(H // 4) + 0.5) / (ppd / 4)
    band_s = (el_s > BAND_DEG[0]) & (el_s < BAND_DEG[1])
    valid_s = small.min(axis=2) > 0
    reference = lambda rgb: np.clip(rgb / ref, 0, 1) ** (1 / 2.2)      # noqa: E731

    # A
    def base_of(rgb, radius_px):
        L = np.log(lum(rgb) / ref).astype(np.float32)
        return L, guided_filter(L, int(max(2, radius_px)), A_EPS)
    _, base_s = base_of(small, A_RADIUS_DEG * ppd / 4)
    sel = base_s[band_s][valid_s[band_s]]
    b_lo, b_hi = np.percentile(sel, [0.5, 99.5])
    a_c = float(min(1.0, np.log(A_CONTRAST) / max(b_hi - b_lo, 1e-6)))
    b_mid = float(np.median(sel))

    def method_a(rgb, radius_px):
        L, base = base_of(rgb, radius_px)
        Lo = np.exp((base - b_mid) * a_c + (L - base) * A_DETAIL) * A_MID
        ratio = (rgb / lum(rgb)[..., None]) ** A_RATIO_POWER
        return np.clip(ratio * Lo[..., None], 0, 1) ** (1 / 2.2)

    # B
    b_gain = [1.0]

    def method_b(rgb):
        tm = cv2.createTonemapMantiuk(gamma=2.2, scale=B_SCALE, saturation=B_SAT)
        v = np.nan_to_num(tm.process(np.ascontiguousarray(rgb / ref, np.float32)))
        return np.clip(v * b_gain[0], 0, 1)
    vb = method_b(small)
    b_gain[0] = float(0.95 / max(np.percentile(lum(vb[band_s][valid_s[band_s]]), 99.5), 1e-6))
    del vb

    # C
    info: Dict[str, Any] = {}
    sky, rover = masks(small, (H, W), depth, rover_depth, info)
    keep = ~(sky | rover)
    keep_s = cv2.resize(keep.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST) > 0
    px = reference(small)[valid_s & keep_s].astype(np.float64)
    mu = px.mean(axis=0)
    ev, E = np.linalg.eigh(np.cov((px - mu).T))
    T = ((E * (C_SIGMA / np.sqrt(np.maximum(ev, 1e-12)))) @ E.T).astype(np.float32)   # rotate, equalize, rotate back
    mu_out = np.full(3, float(mu.mean()), np.float32)

    def method_c(rgb, k):
        g = reference(rgb)
        return np.where(k[..., None], np.clip((g - mu.astype(np.float32)) @ T.T + mu_out, 0, 1), g)

    def save(n, v):
        cv2.imwrite(str(out_dir / f"{name}_{n}.jpg"), (np.clip(v, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    for k_, v in (("reference_gamma", reference(small)), ("A_base_detail", method_a(small, A_RADIUS_DEG * ppd / 4)),
                  ("B_mantiuk", method_b(small)), ("C_decorrelation", method_c(small, keep_s))):
        save("pano_" + k_, v)
    mk = (np.dstack([sky, rover, np.zeros_like(sky)])[::4, ::4]).astype(np.uint8) * 255
    cv2.imwrite(str(out_dir / f"{name}_C_mask_sky_blue_rover_green.png"), mk)
    # the full-resolution crop: the 4096 columns with the most fine contrast, from 8 deg above the horizon down
    r0 = int((90 - 8) * ppd)
    r1 = r0 + min(2048, int(45 * ppd))
    g4 = lum(small)[r0 // 4:r1 // 4]
    score = np.abs(cv2.Laplacian(np.log(g4).astype(np.float32), cv2.CV_32F)).mean(axis=0)
    kw = 4096 // 4
    c0 = int(np.argmax(np.convolve(np.r_[score, score[:kw]], np.ones(kw), "valid")[:len(score)])) * 4
    pad = int(3 * A_RADIUS_DEG * ppd)
    cols = np.arange(c0 - pad, c0 + 4096 + pad) % W
    rr = slice(max(0, r0 - pad), r1 + pad)
    crop = im16[rr][:, cols][..., ::-1].astype(np.float32)
    top = r0 - max(0, r0 - pad)
    cut = lambda v: v[top:top + (r1 - r0), pad:pad + 4096]             # noqa: E731
    crops = {"reference_gamma": cut(reference(crop)), "A_base_detail": cut(method_a(crop, A_RADIUS_DEG * ppd)),
             "B_mantiuk": cut(method_b(crop)), "C_decorrelation": cut(method_c(crop, keep[rr][:, cols]))}
    tiles = []
    for k_, v in crops.items():
        save("crop_" + k_, v)
        hh, ww = v.shape[:2]
        y0, x0 = max(0, (hh - 700) // 2), max(0, (ww - 1400) // 2)         # the middle 1400 x 700 of the crop
        t = np.ascontiguousarray((np.clip(v[y0:y0 + 700, x0:x0 + 1400], 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1])
        for th, colr in ((6, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(t, k_.replace("_", " "), (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colr, th, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(str(out_dir / f"{name}_sheet.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    rec = {"source": str(src), "size": [W, H], "band_elevation_deg": BAND_DEG, "white_linear_value": ref,
           "A": {"base_filter": "guided filter (He et al. 2010) on log luminance", "radius_deg": A_RADIUS_DEG, "eps": A_EPS,
                 "base_log_range": [float(b_lo), float(b_hi)], "base_median_log": b_mid, "base_compression": a_c,
                 "base_contrast_out_max": A_CONTRAST, "detail_gain": A_DETAIL, "ratio_power": A_RATIO_POWER, "median_out_linear": A_MID},
           "B": {"operator": "OpenCV TonemapMantiuk (Mantiuk et al. 2006)", "gamma": 2.2, "scale": B_SCALE, "saturation": B_SAT,
                 "gain_after": b_gain[0]},
           "C": {"space": "gamma 2.2 of linear / white", "pixels": "terrain only: no sky, no rover", "mean": mu.tolist(),
                 "matrix": T.tolist(), "eigenvalues": ev.tolist(), "sigma_out": C_SIGMA, "sky_share": float(sky.mean()),
                 "rover_share": float(rover.mean()), "sky_reference_elevation_deg": SKY_REF_EL_DEG,
                 "sky_grow_px_quarter": SKY_GROW_PX, "rover_grow_deg": ROVER_GROW_DEG, **info},
           "crop": {"rows": [r0, r1], "first_column": c0, "columns": 4096}}
    (out_dir / f"{name}_parameters.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="three display transforms of a 16-bit linear panorama (see the module)")
    ap.add_argument("panorama", help="a *_ods_left_16.png (or _right_16)")
    ap.add_argument("out", help="the output folder")
    ap.add_argument("name", help="the files' prefix")
    ap.add_argument("--depth", help="the station's *_ods_centre_depth.npz (the sky)")
    ap.add_argument("--rover-depth", help="the rover run's *_rover_ods_centre_depth.npz (the rover)")
    a = ap.parse_args(argv)
    rec = run(a.panorama, a.out, a.name, a.depth, a.rover_depth)
    print(f"{a.name}: white {rec['white_linear_value']:.0f}, sky {rec['C']['sky_share']:.3f}, rover {rec['C']['rover_share']:.3f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
