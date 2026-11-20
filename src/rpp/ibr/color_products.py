"""
rpp.ibr.color_products - three more color views of a station from its 16-bit eye panoramas (0.99.37; Christian, 19 Nov
2026: "regenerate the color products from the 16-bit images ... The color channels should be: standard color, raw
color (with the standard white balance reversed to show what the raw images look like), enhanced color (enhancement
with local brightness correction), enhanced tone (tone enhanced color with local brightness correction), dcs color
(decorrelation stretch on just the sunlit terrain with the sky white and the rover/shadow black), normal map, distance
map").

The standard and the enhanced color views (``<prefix>_360_3DTD.jpg``, ``<prefix>_enhlocal_360_3DTD.jpg``) are made by
:mod:`rpp.ibr.linear16`; this module adds, from the same ``<prefix>_ods_<eye>_16.png`` (linear: radiance x white
balance x brightness / zenith scale x a fixed scale) and with nothing rendered again:

  <prefix>_raw_360_3DTD.jpg       raw color: the standard view of the linear values divided by the white balance the
                                  frames were given (``ods.json`` linear16.white_balance, red = 1) - the color of the
                                  calibrated RAD products before the processing balanced them.  One balance for the
                                  panorama: the station's first frame's (its main camera); fill frames of another
                                  camera are divided by it too.
  <prefix>_enhtone_360_3DTD.jpg   enhanced tone: a base / detail tone mapping (Durand and Dorsey 2002) - the log
                                  luminance split by an edge-preserving guided filter (He et al. 2010; the fast form of
                                  He and Sun 2015: its coefficients at TONE_WORK_WIDTH, applied at full resolution) into
                                  the local brightness (the base, radius TONE_RADIUS_DEG) and the detail; the base
                                  compressed only (never expanded) so the terrain band's 0.5-99.5 % span fits
                                  TONE_CONTRAST, its median set to TONE_MID, the detail x TONE_DETAIL; each pixel's
                                  R:G:B ratios kept (to the power TONE_RATIO_POWER); gamma 2.2; then (0.99.38) the
                                  values stretched x TONE_OUT_CONTRAST about the band's median and the colors x
                                  TONE_OUT_SATURATION.
  <prefix>_dcs_360_3DTD.jpg       dcs color: a decorrelation stretch (Gillespie et al. 1986) - in the gamma view
                                  (linear / white, gamma 2.2) the principal components of the three bands given equal
                                  spread DCS_SIGMA about mid gray (the terrain's own color not kept), the statistics
                                  from the sunlit terrain only; the sky white, the rover and its shadow black.

Every parameter is a fixed function of the station's terrain band (elevation -30 to -2 deg) of the first eye made
(the left), and is used for both eyes and for the other variant (the station's shared levels file); nothing is learned
and nothing is painted.  ``<prefix>_color_products.json`` records them.  The light blue disc at the nadir is copied from
the standard view (it is a label, not terrain).

The masks of the dcs view, per eye at TONE_WORK_WIDTH (then grown and scaled up):
  sky     brighter than halfway (log luminance) between the terrain just below the horizon and the sky above
          DCS_SKY_REF_EL_DEG, inside the far field (the center depth's label "far" where the station has one, else
          the render's sky weight, else above the horizon), connected to the zenith; each column from the zenith down
          to its first non-sky pixel;
  rover   the rover model's pixels of the center depth (``rover`` layer; the with-rover variant's, used for the
          terrain-only variant too), grown by DCS_ROVER_GROW_DEG (an eye sees the deck up to 2.8 deg from where the
          center sees it).  Without a center depth there is no rover mask (recorded);
  shadow  pixels below the horizon darker than DCS_SHADOW_FRAC of the terrain band's median luminance, connected to the
          rover (grown 1 deg): the rover's shadow.  Other shadows stay terrain, but no pixel darker than DCS_SUNLIT_FRAC
          of that median enters the statistics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

log = logging.getLogger("rpp.ibr.color_products")
PathLike = Union[str, Path]

TAGS = {"raw": "raw color", "enhtone": "enhanced tone", "dcs": "dcs color"}      # file tag -> the viewer's name
BAND_DEG = (-30.0, -2.0)
GAMMA = 2.2
WHITE_PCT = 99.9                     # the band's luminance shown as white (enhanced tone, dcs)
TONE_WORK_WIDTH = 4096
TONE_RADIUS_DEG, TONE_EPS = 1.0, 0.04
TONE_CONTRAST, TONE_DETAIL, TONE_RATIO_POWER, TONE_MID = 8.0, 2.5, 1.2, 0.22   # 0.99.38 (Christian: "The enhanced tone
#   looks too bland"): 1.8, 1.15, 0.30 in 0.99.37 (the 18 Nov study's) - more detail, a little more color, darker
TONE_OUT_CONTRAST = 1.35        # 0.99.38: then the display values stretched about the terrain band's median by this
TONE_OUT_SATURATION = 1.2       # ... and the colors' distance from their gray (luma) x this
DCS_SIGMA = 0.20                     # 0.075 in the 18 Nov study; Christian, 19 Nov 2026: "the dcs does not need to show the
#                                      terrain's original colors underneath. It can be a colorful splotchy mess for now"
DCS_MID = 0.5                        # every component centered on mid gray
DCS_SKY_REF_EL_DEG = 15.0
DCS_SKY_GROW_PX = 2                  # at TONE_WORK_WIDTH
DCS_ROVER_GROW_DEG = 3.0
DCS_SHADOW_FRAC = 0.45
DCS_SUNLIT_FRAC = 0.5
LUMA_BGR = np.array([0.0722, 0.7152, 0.2126], np.float32)
STRIP_ROWS = 512
RULE = (f"raw color: the standard view of linear / white balance; enhanced tone: base/detail tone mapping (guided filter "
        f"radius {TONE_RADIUS_DEG} deg, eps {TONE_EPS}; base compressed to a contrast of {TONE_CONTRAST:g} at most, median "
        f"-> {TONE_MID}; detail x {TONE_DETAIL}; R:G:B ratios ^ {TONE_RATIO_POWER}; gamma {GAMMA}; then x {TONE_OUT_CONTRAST} about the "
        f"band's median and saturation x {TONE_OUT_SATURATION}); dcs color: "
        f"decorrelation stretch in the gamma view, sigma {DCS_SIGMA} per principal component about {DCS_MID}, statistics from the sunlit "
        f"terrain (not sky, rover, its shadow, or darker than {DCS_SUNLIT_FRAC} x the band median), sky white, rover and "
        f"shadow black; white = the terrain band's ({BAND_DEG} deg) {WHITE_PCT} % luminance")


def _elevation(H: int) -> np.ndarray:
    return 90.0 - (np.arange(H) + 0.5) * 180.0 / H


def _lum(bgr: np.ndarray) -> np.ndarray:
    return np.maximum(bgr @ LUMA_BGR, 1e-6)


def _wrap_blur(a: np.ndarray, r: int) -> np.ndarray:
    """A box filter (2r+1) continuous across the 360-degree seam."""
    import cv2
    r = max(1, int(r))
    pad = np.concatenate([a[:, -r:], a, a[:, :r]], axis=1)
    return cv2.blur(pad, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REPLICATE)[:, r:r + a.shape[1]]


def guided_coefficients(I: np.ndarray, r: int, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """The averaged coefficients (a, b) of the guided filter with ``I`` as its own guide (He et al. 2010): the filtered
    image is a * I + b; computed small and scaled up, applied to the full image (He and Sun 2015)."""
    m = _wrap_blur(I, r)
    v = _wrap_blur(I * I, r) - m * m
    a = v / (v + eps)
    b = m - a * m
    return _wrap_blur(a, r), _wrap_blur(b, r)


def center_layers(folder: PathLike) -> Dict[str, Any]:
    """{"label", "rover", "file"} of the station's center depth: the label from this folder's (else its with_rover
    folder's), the rover from the first that has rover pixels (a terrain-only run's layer is empty: the with_rover
    folder's is used)."""
    folder = Path(folder)
    found = []
    for d in (folder, folder / "with_rover"):
        if d.is_dir():
            found += sorted(d.glob("*ods_centre_depth.npz")) + sorted(d.glob("*ods_center_depth.npz"))
    out: Dict[str, Any] = {}
    for f in found:
        try:
            with np.load(f) as z:
                if "label" not in out and "label" in z.files:
                    out["label"] = np.asarray(z["label"], np.uint8)
                    out["file"] = str(f)
                if "rover" not in out and "rover" in z.files:
                    rv = np.asarray(z["rover"], np.uint8)
                    if rv.any():
                        out["rover"], out["rover_file"] = rv, str(f)
        except (OSError, ValueError, KeyError):
            continue
    return out


def _resize(a: np.ndarray, w: int, h: int, nearest: bool = True) -> np.ndarray:
    import cv2
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def masks(small: np.ndarray, layers: Dict[str, np.ndarray], sky_weight: Optional[np.ndarray],
          info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """{"sky", "rover", "shadow"} (bool) at the size of ``small`` (BGR float, linear) - see the module."""
    import cv2
    h, w = small.shape[:2]
    el = _elevation(h)[:, None]
    ok = small.min(axis=2) > 0
    L = np.log(_lum(small))
    if "label" in layers:
        lab = _resize(layers["label"], w, h)
        far, mesh = lab == 2, lab == 1
        info["far_from"] = "center depth"
    elif sky_weight is not None:
        far = _resize((np.asarray(sky_weight) > 0).astype(np.uint8), w, h) > 0
        far = cv2.dilate(far.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        mesh = ~far
        info["far_from"] = "sky weight"
    else:
        far = np.broadcast_to(el > 0, (h, w)).copy()
        mesh = ~far
        info["far_from"] = "above the horizon"
    high = far & (el > DCS_SKY_REF_EL_DEG) & ok
    L_sky = float(np.median(L[high])) if high.sum() > 100 else float(np.percentile(L[far & ok], 90)) if (far & ok).any() else float(L.max())
    near = mesh & (el > -10.0) & (el < 0.0) & ok
    L_ter = float(np.median(L[near])) if near.sum() > 100 else float(np.median(L[ok])) if ok.any() else 0.0
    cut = 0.5 * (L_sky + L_ter)
    kr = 2 * max(1, int(round(1.0 * w / 360.0))) + 1                 # mesh pixels within 1 deg of the far field
    rim = mesh & (cv2.dilate(far.astype(np.uint8), np.ones((kr, kr), np.uint8)) > 0)
    like = ((far | rim) & (L > cut) & ok).astype(np.uint8)
    like = cv2.morphologyEx(like, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, comp = cv2.connectedComponents(like, connectivity=4)
    top = np.unique(comp[:3][like[:3] > 0])
    s = np.isin(comp, top[top > 0]).astype(np.uint8)
    s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    s = np.cumprod(s, axis=0).astype(np.uint8)
    s = cv2.dilate(s, np.ones((2 * DCS_SKY_GROW_PX + 1, 2 * DCS_SKY_GROW_PX + 1), np.uint8)) & (far | rim).astype(np.uint8)
    sky = s > 0
    rover = np.zeros((h, w), bool)
    if "rover" in layers:
        rv = _resize(layers["rover"], w, h)
        k = int(2 * round(DCS_ROVER_GROW_DEG * w / 360.0) + 1)
        rover = cv2.dilate(rv, np.ones((max(3, k // 3), k), np.uint8)) > 0
        info["rover_from"] = "center depth"
    else:
        info["rover_from"] = None
    band = (el >= BAND_DEG[0]) & (el <= BAND_DEG[1])
    terr = band & ok & ~sky & ~rover
    L_med = float(np.median(L[terr])) if terr.sum() > 100 else L_ter
    shadow = np.zeros((h, w), bool)
    if rover.any():
        dark = ((L < L_med + np.log(DCS_SHADOW_FRAC)) & (el < 0) & ~rover & ~sky & ok).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        _, comp = cv2.connectedComponents(dark, connectivity=8)
        k1 = int(2 * round(1.0 * w / 360.0) + 1)
        touch = cv2.dilate(rover.astype(np.uint8), np.ones((k1, k1), np.uint8)) > 0
        ids = np.unique(comp[touch & (dark > 0)])
        shadow = np.isin(comp, ids[ids > 0])
        shadow = cv2.dilate(shadow.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        shadow &= ~sky
    info.update(log_luminance_sky=L_sky, log_luminance_terrain_below_horizon=L_ter, log_luminance_cut=cut,
                log_luminance_band_median=L_med)
    return {"sky": sky, "rover": rover, "shadow": shadow, "L_med": np.float32(L_med)}


def _parameters(small: np.ndarray, m: Dict[str, np.ndarray], mark_row_small: Optional[int]) -> Dict[str, Any]:
    """The station's parameters from the first eye at TONE_WORK_WIDTH."""
    h, w = small.shape[:2]
    el = _elevation(h)[:, None]
    ok = small.min(axis=2) > 0
    band = np.broadcast_to((el >= BAND_DEG[0]) & (el <= BAND_DEG[1]), (h, w)) & ok
    if mark_row_small is not None:
        band[mark_row_small:] = False
    lum = _lum(small)
    ref = float(np.percentile(lum[band], WHITE_PCT)) if band.sum() > 100 else float(np.percentile(lum[ok], WHITE_PCT))
    L = np.log(lum / ref).astype(np.float32)
    r = max(2, int(round(TONE_RADIUS_DEG * w / 360.0)))
    ga, gb = guided_coefficients(L, r, TONE_EPS)
    base = ga * L + gb
    sel = base[band]
    b_lo, b_hi = (np.percentile(sel, [0.5, 99.5]) if len(sel) else (0.0, 1.0))
    a_c = float(min(1.0, np.log(TONE_CONTRAST) / max(float(b_hi - b_lo), 1e-6)))
    b_mid = float(np.median(sel)) if len(sel) else 0.0
    y0 = _tone_display(small, L, base, b_mid, a_c)              # 0.99.38: the band's median in the display values
    out_mid = float(np.median(y0[band].mean(axis=1))) if band.any() else 0.5
    keep = band & ~m["sky"] & ~m["rover"] & ~m["shadow"] & (np.log(lum) >= float(m["L_med"]) + np.log(DCS_SUNLIT_FRAC))
    g = np.clip(small / ref, 0.0, 1.0) ** (1.0 / GAMMA)
    px = g[keep].astype(np.float64) if keep.sum() > 100 else g[band].astype(np.float64)
    mu = px.mean(axis=0)
    ev, E = np.linalg.eigh(np.cov((px - mu).T) + 1e-12 * np.eye(3))
    T = (E * (DCS_SIGMA / np.sqrt(np.maximum(ev, 1e-12)))) @ E.T
    return {"white_linear_8bit_units": ref, "tone": {"base_log_range": [float(b_lo), float(b_hi)], "base_median_log": b_mid,
            "base_compression": a_c, "display_median": out_mid}, "dcs": {"mean_bgr": mu.tolist(), "matrix_bgr": T.tolist(), "eigenvalues": ev.tolist(),
            "pixels": int(keep.sum())}}


def _tone_display(x: np.ndarray, Ls: np.ndarray, base: np.ndarray, b_mid: float, a_c: float) -> np.ndarray:
    """The enhanced tone's display values (0-1, BGR) before its output stretch: the base compressed, the detail x
    TONE_DETAIL, the color ratios kept (^ TONE_RATIO_POWER), gamma."""
    lum = _lum(x)
    Lo = np.exp((base - b_mid) * a_c + (Ls - base) * TONE_DETAIL) * TONE_MID
    ratio = (x / lum[..., None]) ** TONE_RATIO_POWER
    return np.clip(ratio * Lo[..., None], 0.0, 1.0) ** (1.0 / GAMMA)


def _tone_stretch(y: np.ndarray, mid: float) -> np.ndarray:
    """0.99.38: the display values stretched about ``mid`` (TONE_OUT_CONTRAST) and their colors (TONE_OUT_SATURATION)."""
    y = np.clip(mid + (y - mid) * TONE_OUT_CONTRAST, 0.0, 1.0)
    g = (y @ LUMA_BGR)[..., None]
    return np.clip(g + TONE_OUT_SATURATION * (y - g), 0.0, 1.0)


def _mark_row(out: Path, H: int) -> Optional[int]:
    """The first row of the nadir disc (``ods.json`` nadir_mark.first_row, the render's size: the 16-bit eye's)."""
    try:
        nm = (json.loads((out / "ods.json").read_text(encoding="utf-8")).get("nadir_mark") or {})
    except (OSError, ValueError):
        return None
    r = nm.get("first_row")
    return int(r) if r is not None and 0 < int(r) < H else None


RAW_KEEP_LUMA = True                 # 0.99.51 (Christian, 20 Nov 2026: "raw will look darker red and may need a slight
#                                      brightness boost"): the raw view x luma(1) / luma(1 / white balance) - a gray keeps
#                                      its brightness, the colors their raw ratios (x 1.19 for [1, 1.224, 1.569])


def raw_divisor(wb: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """The BGR divisor of the raw color view: the white balance, over the brightness factor of RAW_KEEP_LUMA."""
    if wb is None:
        return None
    w = np.asarray(wb, np.float32)
    k = float((LUMA_BGR[::-1] / w).sum() / LUMA_BGR.sum()) if RAW_KEEP_LUMA else 1.0     # RGB weights vs 1/wb
    return (w * np.float32(k))[::-1].astype(np.float32)


def white_balance(out: Path) -> Optional[np.ndarray]:
    """The frames' white balance x brightness (RGB) of the panorama (``ods.json`` linear16)."""
    try:
        l16 = json.loads((out / "ods.json").read_text(encoding="utf-8")).get("linear16") or {}
    except (OSError, ValueError):
        return None
    wb = l16.get("white_balance")
    if not wb or len(wb) != 3:
        return None
    return np.asarray(wb, np.float32)


def write_color_products(out_dir: PathLike, prefix: str, top: str = "left", jpeg_quality: int = 95,
                         levels_file: Optional[PathLike] = None) -> Dict[str, Any]:
    """The raw color, enhanced tone and dcs color squares of ``out_dir`` from its 16-bit eye panoramas (see the
    module).  Returns the record (also written to ``<prefix>_color_products.json``)."""
    import cv2
    from .linear16 import UNITS_PER_8BIT, _exposure16, shared_value, tone_pipeline
    from .ods import SKY_WEIGHT_NAME, STACK_NAME
    out = Path(out_dir)
    pre = f"{prefix}_" if prefix else ""
    k16 = UNITS_PER_8BIT * _exposure16(out)
    wb = white_balance(out)
    layers = center_layers(out)
    sky_w = cv2.imread(str(out / f"{pre}{SKY_WEIGHT_NAME}"), cv2.IMREAD_GRAYSCALE)
    rec: Dict[str, Any] = {"rule": RULE, "eyes": {},
                           "white_balance_divided_rgb": None if wb is None else wb.tolist(),
                           "raw_brightness_x": None if wb is None else round(float(wb[0] / raw_divisor(wb)[2]), 4),
                           "center_depth": layers.get("file"), "rover_layer": layers.get("rover_file")}
    params = None
    squares: Dict[str, np.ndarray] = {}
    order = ("left", "right") if top == "left" else ("right", "left")
    for half, eye in enumerate(order):
        f = out / f"{pre}ods_{eye}_16.png"
        a16 = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if a16 is None or a16.dtype != np.uint16:
            raise FileNotFoundError(f"{f}: no 16-bit panorama")
        a16 = a16[..., :3]
        H, W = a16.shape[:2]
        if not squares:
            squares = {t: np.empty((2 * H, W, 3), np.uint8) for t in TAGS}
        w4 = min(TONE_WORK_WIDTH, W)
        h4 = max(2, int(round(H * w4 / W)))
        small = cv2.resize(a16, (w4, h4), interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(k16)
        mrow = _mark_row(out, H)
        mrow4 = None if mrow is None else int(mrow * h4 / H)
        info: Dict[str, Any] = {}
        m = masks(small, layers, None if sky_w is None else sky_w.astype(np.float32) / 255.0, info)
        if params is None:
            params, src = shared_value(levels_file, "color_products_parameters_v2", lambda: _parameters(small, m, mrow4))
            rec["parameters"], rec["parameters_from"] = params, src
        ref = float(params["white_linear_8bit_units"])
        tp = params["tone"]
        mu = np.asarray(params["dcs"]["mean_bgr"], np.float32)
        T = np.asarray(params["dcs"]["matrix_bgr"], np.float32)
        mu_out = np.full(3, DCS_MID, np.float32)
        r = max(2, int(round(TONE_RADIUS_DEG * w4 / 360.0)))
        L4 = np.log(_lum(small) / ref).astype(np.float32)
        ga, gb = guided_coefficients(L4, r, TONE_EPS)
        del L4
        black4 = (m["rover"] | m["shadow"]).astype(np.uint8)
        white4 = m["sky"].astype(np.uint8)
        std8 = cv2.imread(str(out / f"{pre}ods_{eye}.png"))
        disc = std8[mrow:].copy() if (std8 is not None and std8.shape[:2] == (H, W) and mrow is not None) else None
        del std8
        fy = H / h4
        wb_bgr = raw_divisor(wb)
        n_white = 0
        for r0 in range(0, H, STRIP_ROWS):
            r1 = min(H, r0 + STRIP_ROWS)
            xs = a16[r0:r1].astype(np.float32) / np.float32(k16)          # BGR, 8-bit linear units
            s0, s1 = max(0, int(r0 / fy) - 2), min(h4, int(np.ceil(r1 / fy)) + 2)   # the small rows about the strip
            n = max(1, int(round((s1 - s0) * fy)))
            off = min(max(0, int(round(r0 - s0 * fy))), max(0, n - (r1 - r0)))
            up = lambda q, nearest=False: cv2.resize(q[s0:s1], (W, n), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)[off:off + (r1 - r0)]   # noqa: E731
            dst = slice(half * H + r0, half * H + r1)
            squares["raw"][dst] = np.rint(tone_pipeline(xs / wb_bgr if wb_bgr is not None else xs)).astype(np.uint8)
            Ls = np.log(_lum(xs) / ref)                                  # enhanced tone
            base = up(ga) * Ls + up(gb)
            yt = _tone_display(xs, Ls, base, tp["base_median_log"], tp["base_compression"])
            squares["enhtone"][dst] = np.rint(255.0 * _tone_stretch(yt, tp.get("display_median", 0.5))).astype(np.uint8)
            del base, Ls, yt
            g = np.clip(xs / ref, 0.0, 1.0) ** (1.0 / GAMMA)            # dcs color
            d = np.clip((g - mu) @ T.T + mu_out, 0.0, 1.0)
            wt = up(white4, True) > 0
            n_white += int(wt.sum())
            d[wt] = 1.0
            d[up(black4, True) > 0] = 0.0
            d[xs.max(axis=2) <= 0] = 0.0
            squares["dcs"][dst] = np.rint(255.0 * d).astype(np.uint8)
            del g, d, xs
        if disc is not None:                                             # the nadir disc as in the standard view
            for t in TAGS:
                squares[t][half * H + mrow:half * H + H] = disc
        rec["eyes"][eye] = {"size": [W, H], "sky_share": n_white / float(H * W), "rover_share": float(m["rover"].mean()),
                            "shadow_share": float(m["shadow"].mean()), **info}
        del a16, ga, gb, small, disc
    files = {}
    for t in TAGS:
        files[t] = str(out / f"{pre}{t}_{STACK_NAME}.jpg")
        cv2.imwrite(files[t], squares[t], [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        del squares[t]
    rec["files"] = files
    rec["notes"] = ([] if wb is not None else ["no white balance in ods.json: the raw color view is the standard one"]) + \
                   ([] if "rover" in layers else ["no center depth with the rover: no rover or shadow mask in the dcs view"])
    (out / f"{pre}color_products.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    log.info("color products: %s (%s)", ", ".join(Path(v).name for v in files.values()), "; ".join(rec["notes"]) or "all masks")
    return rec


SKIN_TAGS = {"enhlocal": "enhanced local", "enhtone": "enhanced tone", "raw": "raw color", "dcs": "dcs color"}   # 0.99.41


def texture_products(a16: np.ndarray, k16: float, folder: PathLike, levels_file: Optional[PathLike],
                     texels_per_deg: float) -> Dict[str, np.ndarray]:
    """0.99.41 (Christian, 19 Nov 2026: "The different color skins for the terrain should be visible in both modes.
    this means they should be rendered as in the full 3d projection too"): the color views of a 16-bit 3D texture
    (BGR uint16 in the panoramas' encoding, :func:`rpp.ibr.station_mesh.write_part`) - {tag: BGR uint8} for the tags
    of :data:`SKIN_TAGS` - with the station's own parameters from ``levels_file`` (the panoramas' shared values: the
    same enhanced-color levels, white balance, enhanced-tone and dcs parameters, so a skin matches its panorama view);
    the filters' radii are set in texels by ``texels_per_deg`` (the texture's texels a degree from its center) for
    the panoramas' angles.  White texels (65535: the sky, unused) stay white; a view whose parameters the file does not
    hold (no color products made yet) is left out."""
    import cv2
    from .linear16 import (ENH_LOCAL_SIGMA_FRAC, desaturate, local_contrast, shared_value, tone_enhanced, tone_pipeline)
    white = (a16 == 65535).all(axis=-1)
    empty = (a16 == 0).all(axis=-1)
    H, W = a16.shape[:2]
    out: Dict[str, np.ndarray] = {}
    wb = white_balance(Path(folder))
    wb_bgr = raw_divisor(wb)
    stored: Dict[str, Any] = {}
    if levels_file is not None and Path(levels_file).is_file():
        try:
            stored = json.loads(Path(levels_file).read_text(encoding="utf-8"))
        except ValueError:
            stored = {}
    lv = stored.get("enhanced_gamma_stretch_s_levels")
    params = stored.get("color_products_parameters_v2")
    tpd = max(float(texels_per_deg), 1e-3)
    raw = np.empty((H, W, 3), np.uint8)
    enh = np.empty((H, W, 3), np.uint8) if lv else None
    enht = dcs = None
    if params:
        ref = float(params["white_linear_8bit_units"])
        tp = params["tone"]
        mu = np.asarray(params["dcs"]["mean_bgr"], np.float32)
        T = np.asarray(params["dcs"]["matrix_bgr"], np.float32)
        mu_out = np.full(3, DCS_MID, np.float32)
        w4 = min(TONE_WORK_WIDTH, W)
        h4 = max(2, int(round(H * w4 / W)))
        small = cv2.resize(a16, (w4, h4), interpolation=cv2.INTER_AREA).astype(np.float32) / np.float32(k16)
        r = max(2, int(round(TONE_RADIUS_DEG * tpd * w4 / W)))
        ga, gb = guided_coefficients(np.log(np.maximum(_lum(small), 1e-6) / ref).astype(np.float32), r, TONE_EPS)
        del small
        enht = np.empty((H, W, 3), np.uint8)
        dcs = np.empty((H, W, 3), np.uint8)
    for r0 in range(0, H, STRIP_ROWS):
        r1 = min(H, r0 + STRIP_ROWS)
        xs = a16[r0:r1].astype(np.float32) / np.float32(k16)             # BGR, 8-bit linear units
        raw[r0:r1] = np.rint(tone_pipeline(xs / wb_bgr if wb_bgr is not None else xs)).astype(np.uint8)
        if enh is not None:
            enh[r0:r1] = np.rint(desaturate(tone_enhanced(xs, *lv))).astype(np.uint8)
        if params:
            fy = H / ga.shape[0]
            s0, s1 = max(0, int(r0 / fy) - 2), min(ga.shape[0], int(np.ceil(r1 / fy)) + 2)
            n = max(1, int(round((s1 - s0) * fy)))
            off = min(max(0, int(round(r0 - s0 * fy))), max(0, n - (r1 - r0)))
            up = lambda q: cv2.resize(q[s0:s1], (W, n), interpolation=cv2.INTER_LINEAR)[off:off + (r1 - r0)]   # noqa: E731
            Ls = np.log(np.maximum(_lum(xs), 1e-6) / ref)
            yt = _tone_display(xs, Ls, up(ga) * Ls + up(gb), tp["base_median_log"], tp["base_compression"])
            enht[r0:r1] = np.rint(255.0 * _tone_stretch(yt, tp.get("display_median", 0.5))).astype(np.uint8)
            del Ls, yt
            g = np.clip(xs / ref, 0.0, 1.0) ** (1.0 / GAMMA)
            d = np.clip((g - mu) @ T.T + mu_out, 0.0, 1.0)
            d[xs.max(axis=2) <= 0] = 0.0
            dcs[r0:r1] = np.rint(255.0 * d).astype(np.uint8)
            del g, d
        del xs
    out["raw"] = raw
    if enh is not None:
        out["enhlocal"] = local_contrast(enh, sigma_px=ENH_LOCAL_SIGMA_FRAC * 360.0 * tpd, wrap=False)
        del enh
    if params:
        out["enhtone"], out["dcs"] = enht, dcs
    for v in out.values():
        v[white] = 255
        v[empty & ~white] = 0
    return out

