"""
Site appearance: texture and contrast measures of processed Navcam images inside their terrain masks (v0p31;
working notes §14).  The measures are chosen to describe what the matcher sees, at the scales it works on:

  rms_contrast        std / mean of the linear (16-bit) intensity over the mask: albedo and shading at all scales
  band_contrast_sN    std of a difference of Gaussians (sigma N and 1.6 N native px) / local mean: texture contrast
                      in the octave SIFT detects at scale N (N = 1, 2, 4, 8)
  spectral_slope      beta of the radially averaged power spectrum P(k) ~ k^-beta over 4-64 px wavelengths, on
                      masked tiles: how fast texture fades towards fine scales (roughness / blur)
  sift_per_mpix       SIFT keypoints (OpenCV defaults) per megapixel of mask, on the 8-bit image the pipeline uses
  sift_response_med   median DoG response of those keypoints
  repeat_frac         fraction of keypoints that have a look-alike in the same image: the nearest other keypoint's
                      descriptor is clearly closer than the next (Lowe ratio < 0.8) - repetitive texture, which the
                      ratio test rejects when matching
  shadow_frac         fraction of mask pixels darker than half the median: cast shadows
  dyn_range           p99 / p1 of the linear intensity over the mask
  coherence           mean structure-tensor coherence ((l1 - l2) / (l1 + l2), sigma 2 px): 0 isotropic, 1 layered
  mask_frac           fraction of the frame that is terrain (mask and below the elevation limit)

Only terrain below ``--max-elevation`` (default -3 deg, from the manifest pose and intrinsics) is measured: the ground
within roughly 35 m of the mast, where the tie points are, without sky, horizon and distant hills.

Usage:
  python scripts/site_appearance.py OUT.json PROCESSED_DIR --sites sites.json
where PROCESSED_DIR holds images_png16/, images_png8/ and masks/ (mppp.process_images with write_mask_files) and
sites.json maps a site name to a list of image names.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _read(p, flags=None):
    import cv2
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    if im.ndim == 3:
        im = im[..., :3].mean(-1) if im.shape[2] >= 3 else im[..., 0]
    return im.astype(np.float64)


def _mask_path(d: Path, name: str):
    for cand in (d / "masks" / name, d / "masks" / (name + ".png"), d / "masks" / (Path(name).stem + "_mask.png")):
        if cand.is_file():
            return cand
    hits = sorted((d / "masks").glob(Path(name).stem + "*"))
    return hits[0] if hits else None


def elevation_map(meta, shape, step=8):
    """Elevation [deg] of every pixel's ray in the site frame (East-North-Up) from the manifest intrinsics and pose."""
    import cv2
    H, W = shape
    K = np.array(meta["intrinsics"]["K"], float)
    dd = meta["intrinsics"].get("dist_opencv") or {}
    dist = np.array([dd.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")], float)
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    pts = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float64).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    rc = np.column_stack([und, np.ones(len(und))])
    R = np.array(meta["pose"]["R_world_to_cam"], float)
    dw = rc @ R                                            # R^T r for each row
    el = np.degrees(np.arcsin(dw[:, 2] / np.linalg.norm(dw, axis=1))).reshape(xs.shape)
    from scipy.ndimage import map_coordinates
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return map_coordinates(el.astype(np.float32), [yy / step, xx / step], order=1, mode="nearest")


def measures(img16, img8, mask):
    import cv2
    from scipy.ndimage import gaussian_filter, binary_erosion
    out = {}
    m = mask > 0
    out["mask_frac"] = float(m.mean())
    core = binary_erosion(m, iterations=24)          # away from the mask edge, so filters do not see the rover/sky
    if core.sum() < 5000:
        return None
    I = img16 / max(1.0, np.percentile(img16[m], 99.9))
    v = I[core]
    mu = float(v.mean())
    out["rms_contrast"] = float(v.std() / mu)
    out["shadow_frac"] = float((v < 0.5 * np.median(v)).mean())
    p1, p99 = np.percentile(v, [1, 99])
    out["dyn_range"] = float(p99 / max(p1, 1e-6))
    for s in (1, 2, 4, 8):
        g1, g2 = gaussian_filter(I, s), gaussian_filter(I, 1.6 * s)
        loc = gaussian_filter(I, 4 * s) + 1e-6
        out[f"band_contrast_s{s}"] = float(np.std(((g1 - g2) / loc)[core]))
    # structure tensor coherence
    gx, gy = np.gradient(gaussian_filter(I, 1.0))
    Jxx, Jyy, Jxy = (gaussian_filter(a, 2.0) for a in (gx * gx, gy * gy, gx * gy))
    tr = Jxx + Jyy
    det_term = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    coh = det_term / (tr + 1e-12)
    out["coherence"] = float(np.mean(coh[core]))
    # power spectrum slope on 128 px tiles fully inside the mask
    T = 128
    slopes = []
    H, W = I.shape
    win = np.outer(np.hanning(T), np.hanning(T))
    fy, fx = np.meshgrid(np.fft.fftfreq(T), np.fft.fftfreq(T), indexing="ij")
    kr = np.hypot(fx, fy)
    sel = (kr >= 1 / 64) & (kr <= 1 / 4)
    for y in range(0, H - T, T):
        for x in range(0, W - T, T):
            if core[y:y + T, x:x + T].all():
                t = I[y:y + T, x:x + T]
                t = (t - t.mean()) * win
                P = np.abs(np.fft.fft2(t)) ** 2
                c = np.polyfit(np.log(kr[sel]), np.log(P[sel] + 1e-30), 1)
                slopes.append(-c[0])
    out["spectral_slope"] = float(np.median(slopes)) if slopes else None
    out["spectral_tiles"] = len(slopes)
    # SIFT on the 8-bit image inside the mask
    u8 = np.clip(img8, 0, 255).astype(np.uint8)
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(u8, (core * 255).astype(np.uint8))
    area = core.sum() / 1e6
    out["sift_per_mpix"] = float(len(kps) / area)
    out["sift_response_med"] = float(np.median([k.response for k in kps])) if kps else None
    out["sift_size_med_px"] = float(np.median([k.size for k in kps])) if kps else None
    if des is not None and len(des) > 10:
        idx = np.random.default_rng(0).choice(len(des), min(3000, len(des)), replace=False)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        mm = bf.knnMatch(des[idx], des, k=3)                 # the first neighbour is the keypoint itself
        ratio = [a[1].distance / max(a[2].distance, 1e-6) for a in mm if len(a) == 3]
        out["repeat_frac"] = float(np.mean(np.array(ratio) < 0.8))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("processed")
    ap.add_argument("--sites", required=True)
    ap.add_argument("--max-elevation", type=float, default=-3.0)
    a = ap.parse_args(argv)
    d = Path(a.processed)
    man = json.loads(next(iter(sorted(d.glob("mppp_manifest*.json"), reverse=True)), d / "manifest.json").read_text())
    meta = {Path(m["source_product"]).stem: m for m in man["images"]}
    sites = json.loads(Path(a.sites).read_text())
    res = {}
    for site, names in sites.items():
        rows = []
        for n in names:
            p16, p8 = d / "images_png16" / n, d / "images_png8" / n
            mp = _mask_path(d, n)
            if not (p16.is_file() and p8.is_file() and mp is not None):
                print(f"{site}: {n} missing", flush=True)
                continue
            mk = _read(mp)
            m = meta.get(Path(n).stem)
            if m is not None and a.max_elevation is not None:
                mk = mk * (elevation_map(m, mk.shape) < a.max_elevation)
            r = measures(_read(p16), _read(p8), mk)
            if r:
                rows.append(dict(r, name=n, lmst_h=(m or {}).get("lmst_h"),
                                 boresight_elevation_deg=((m or {}).get("pose") or {}).get("boresight_elevation_deg")))
        keys = [k for k in (rows[0] if rows else {}) if k != "name"]
        med = {k: float(np.median([r[k] for r in rows if r.get(k) is not None])) for k in keys} if rows else {}
        res[site] = {"images": len(rows), "median": med, "rows": rows}
        print(site, len(rows), {k: round(v, 4) for k, v in med.items()}, flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
