"""
The Hazcam joint (0.94.4, Christian 24 Oct 2026): the Hazcam law fitted directly on the image observations of every
block at once, each frame at its own camera temperature, instead of a line through one refined camera per block.

"Bin the hazcams by temp and fit the slope directly in the joint instead of just fitting the scattered points ...
assume the rear and front hazcams have the same thermal slopes; for now, give all four FL's slope ... set the hazcam
early mission offsets to zero until there is significant evidence they are different."

The adjustment, per eye e (FL, FR, RL, RR) and image i at temperature T_i (``camera_temperature_degC``):

    fx = Fx_e (1 + s (T_i - T0)) + early_i Ex_e      fy = Fy_e (1 + s (T_i - T0)) + early_i Ey_e
    k1 = K1_e + k1s_e (T_i - T0)                      k2 = K2_e
    cx, cy, p1, p2, k3, k4: the median of the blocks' start cameras (the blocks hold them)

* one relative focal slope ``s`` (ppm/degC) for all four eyes, informed by the eyes of ``slope_eyes`` (default FL:
  the other eyes use it without pulling on it - :data:`SLOPE_EYES`); ``slope_eyes="all"`` shares it, and
  ``per_eye`` gives each eye its own slope (the comparison in the report);
* k1's slope per front eye; the rear eyes' k1 and k2 are constant (they have a sixteenth of the observations);
* the early-mission offsets Ex, Ey (images before :data:`EARLY_SOL`) are fitted and then kept only where they are
  significant (``|z| >= EARLY_Z_MIN``); the others are set to 0 and the law fitted again;
* each image's pose is free; the 3D points are held at the block's solution and only those seen from at least two
  other frames enter (Navcam frames, the other Hazcam frames) - so a point is not placed by the image it calibrates;
* weights: 0.5 px / the image's downsample scale (the rear eyes at a quarter resolution), Huber at 2 sigma.

The poses are eliminated per image (Schur complement), so every Gauss-Newton step is a solve of the ~30 global terms.

Diagnostic bins (the scatter drawn about the law): per block and eye, the images grouped by temperature in bins of
:data:`BIN_DEG` (wider than the Navcam's 2 degC: the Hazcams have far fewer frames), merged until each holds
:data:`BIN_MIN_IMAGES` images, at most :data:`BIN_MAX_DEG` wide; each bin's focal scale fitted with the poses and
every other term held at the law.

    res = fit_hazcam_joint(blocks)       # {name: WORK}: the result in hazfit's layout (write_hazcam, check_hazcam)
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]

EYES = ("FL", "FR", "RL", "RR")
T0 = -20.0
EARLY_SOL = 380.0
EARLY_Z_MIN = 3.0                 # an early-mission offset is kept when |offset / sd| reaches this
MIN_EARLY_IMAGES = 20             # images on each side of the early-mission sol (and 2 blocks) for an offset
SLOPE_EYES = ("FL",)              # Christian 24 Oct: "for now, give all four FL's slope"
BIN_DEG = 5.0                     # diagnostic bins: the Navcam's are 2 degC with >= 8 images, <= 10 degC wide
BIN_MIN_IMAGES = 3
BIN_MAX_DEG = 15.0
SIGMA_PX = 0.5                    # at full resolution; divided by the image's downsample scale
HUBER = 2.0                       # in sigma
MAX_OBS_PER_IMAGE = 300
MIN_OBS_PER_IMAGE = 8
MIN_OTHER_FRAMES = 2              # a held point needs this many observations from other frames
MIN_EYE_IMAGES = 6                # an eye with fewer images in the joint is left to the block fit
ITERATIONS = 12
MIN_T_SPAN_DEGC = 5.0             # a temperature slope needs this span (as hazfit)
_IDX = {"fx": 0, "fy": 1, "cx": 2, "cy": 3, "k1": 4, "k2": 5}


# ====================================================================================== data
def collect(blocks: Dict[str, PathLike], model: str = "cahv_ba", max_obs: int = MAX_OBS_PER_IMAGE,
            seed: int = 0) -> Dict[str, Any]:
    """The Hazcam images of the finished blocks with their held observations: {"images": [...], "start": {eye:
    params (the median of the blocks' start cameras)}, "blocks": [...]}.  Each image: block, eye, name, T, sol, scale,
    R (cam_from_world), t, X (n, 3) world points, xy (n, 2) keypoints."""
    import pycolmap
    rng = np.random.default_rng(seed)
    images: List[Dict[str, Any]] = []
    starts: Dict[str, List[List[float]]] = {e: [] for e in EYES}
    used_blocks = []
    for name, work in blocks.items():
        root = Path(work)
        if (root / "colmap" / "project.json").is_file():
            root = root / "colmap"
        try:
            pj = json.loads((root / "project.json").read_text(encoding="utf-8"))
            rec = pycolmap.Reconstruction(str(root / "sparse" / model))
        except Exception as e:                              # noqa: BLE001 - a block not read is left out
            log.info(f"hazcam joint: {name} left out ({type(e).__name__}: {e})")
            continue
        db = ((pj.get("settings") or {}).get("database") or {}).get("cameras") or {}
        by_name = {r["name"]: r for r in pj.get("images") or []}
        n0 = len(images)
        for eye in EYES:
            cam = (pj.get("cameras") or {}).get(eye)
            if cam is not None and db.get(eye) is not None:
                starts[eye].append(list(map(float, cam["params"])))
        for iid in rec.reg_image_ids():
            im = rec.images[iid]
            r = by_name.get(im.name)
            if not r or r.get("instrument") not in EYES or r.get("camera_temperature_degC") is None:
                continue
            fr = im.frame_id
            X, xy = [], []
            for p in im.points2D:
                if not p.has_point3D():
                    continue
                pt = rec.points3D[p.point3D_id]
                other = sum(1 for e in pt.track.elements if rec.images[e.image_id].frame_id != fr)
                if other >= MIN_OTHER_FRAMES:
                    X.append(pt.xyz)
                    xy.append(p.xy)
            if len(X) < MIN_OBS_PER_IMAGE:
                continue
            X, xy = np.asarray(X, float), np.asarray(xy, float)
            if len(X) > max_obs:
                k = rng.choice(len(X), max_obs, replace=False)
                X, xy = X[k], xy[k]
            T = im.cam_from_world()
            images.append({"block": name, "eye": r["instrument"], "name": im.name, "T": float(r["camera_temperature_degC"]),
                           "sol": float(r.get("sol") or 0.0), "scale": float(r.get("downsample_scale") or 1.0),
                           "R": np.asarray(T.rotation.matrix(), float), "t": np.asarray(T.translation, float),
                           "X": X, "xy": xy, "model": rec.cameras[im.camera_id].model,
                           "width": rec.cameras[im.camera_id].width, "height": rec.cameras[im.camera_id].height})
        if len(images) > n0:
            used_blocks.append(name)
    start = {e: np.median(np.asarray(v, float), axis=0) for e, v in starts.items() if v}
    return {"images": images, "start": start, "blocks": used_blocks}


# ====================================================================================== the law
class Law:
    """The global terms and their layout in the parameter vector."""

    def __init__(self, eyes: Sequence[str], start: Dict[str, np.ndarray], slope_ppm: float = 0.0,
                 slope_mode: str = "FL", early_eyes: Sequence[str] = (), k1_slope_eyes: Sequence[str] = ("FL", "FR"),
                 k2_eyes: Sequence[str] = ("FL", "FR")):
        self.eyes = list(eyes)
        self.base = {e: np.asarray(start[e], float).copy() for e in eyes}
        self.mode = slope_mode
        names: List[str] = []
        for e in self.eyes:
            names += [f"Fx_{e}", f"Fy_{e}", f"K1_{e}"]
            if e in k2_eyes:
                names.append(f"K2_{e}")
            if e in k1_slope_eyes:
                names.append(f"k1s_{e}")
            if e in early_eyes:
                names += [f"Ex_{e}", f"Ey_{e}"]
        if slope_mode == "per_eye":
            names += [f"s_{e}" for e in self.eyes]
        else:
            names.append("s")
        self.names = names
        self.ix = {n: i for i, n in enumerate(names)}
        x = np.zeros(len(names))
        for e in self.eyes:
            b = self.base[e]
            x[self.ix[f"Fx_{e}"]], x[self.ix[f"Fy_{e}"]], x[self.ix[f"K1_{e}"]] = b[0], b[1], b[4]
            if f"K2_{e}" in self.ix:
                x[self.ix[f"K2_{e}"]] = b[5]
        for n in names:
            if n == "s" or n.startswith("s_"):
                x[self.ix[n]] = 1e-6 * slope_ppm
        self.x = x
        self.held: set = set()

    def slope_name(self, eye: str) -> str:
        return f"s_{eye}" if self.mode == "per_eye" else "s"

    def informs_slope(self, eye: str) -> bool:
        return self.mode in ("per_eye", "all") or eye in (SLOPE_EYES if self.mode == "FL" else ())

    def params(self, eye: str, T: float, early: bool, x: Optional[np.ndarray] = None) -> np.ndarray:
        x = self.x if x is None else x
        p = self.base[eye].copy()
        dT = T - T0
        s = x[self.ix[self.slope_name(eye)]]
        p[0] = x[self.ix[f"Fx_{eye}"]] * (1 + s * dT)
        p[1] = x[self.ix[f"Fy_{eye}"]] * (1 + s * dT)
        p[4] = x[self.ix[f"K1_{eye}"]] + (x[self.ix[f"k1s_{eye}"]] * dT if f"k1s_{eye}" in self.ix else 0.0)
        if f"K2_{eye}" in self.ix:
            p[5] = x[self.ix[f"K2_{eye}"]]
        if early and f"Ex_{eye}" in self.ix:
            p[0] += x[self.ix[f"Ex_{eye}"]]
            p[1] += x[self.ix[f"Ey_{eye}"]]
        return p

    def chain(self, eye: str, T: float, early: bool) -> Dict[str, Dict[int, float]]:
        """d(camera parameter)/d(global term) for fx, fy, k1, k2: {param: {global index: derivative}}."""
        x = self.x
        dT = T - T0
        sn = self.slope_name(eye)
        s = x[self.ix[sn]]
        out: Dict[str, Dict[int, float]] = {"fx": {self.ix[f"Fx_{eye}"]: 1 + s * dT},
                                           "fy": {self.ix[f"Fy_{eye}"]: 1 + s * dT},
                                           "k1": {self.ix[f"K1_{eye}"]: 1.0}, "k2": {}}
        if self.informs_slope(eye):
            out["fx"][self.ix[sn]] = x[self.ix[f"Fx_{eye}"]] * dT
            out["fy"][self.ix[sn]] = x[self.ix[f"Fy_{eye}"]] * dT
        if f"k1s_{eye}" in self.ix:
            out["k1"][self.ix[f"k1s_{eye}"]] = dT
        if f"K2_{eye}" in self.ix:
            out["k2"][self.ix[f"K2_{eye}"]] = 1.0
        if early and f"Ex_{eye}" in self.ix:
            out["fx"][self.ix[f"Ex_{eye}"]] = 1.0
            out["fy"][self.ix[f"Ey_{eye}"]] = 1.0
        return out


# ====================================================================================== projection
def _exp(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def _project(im: Dict[str, Any], p: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    import pycolmap
    cam = pycolmap.Camera(model=im["model"], width=im["width"], height=im["height"], params=p)
    uv = cam.img_from_cam(Xc)
    return np.asarray(uv, float).reshape(-1, 2)


def image_system(im: Dict[str, Any], law: Law) -> Dict[str, Any]:
    """Residuals (normalised), the pose Jacobian (2n, 6) and the global Jacobian (2n, G) of one image."""
    early = im["sol"] < EARLY_SOL
    p = law.params(im["eye"], im["T"], early)
    Xc = im["X"] @ im["R"].T + im["t"]
    ok = Xc[:, 2] > 1e-3
    uv = _project(im, p, Xc)
    sig = SIGMA_PX / max(im["scale"], 1e-6)
    r = ((uv - im["xy"]) / sig).ravel()
    Jp = np.zeros((r.size, 6))
    eps_w, eps_t = 1e-6, 1e-6
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps_w
        Jp[:, k] = ((_project(im, p, Xc @ _exp(d).T) - uv) / (sig * eps_w)).ravel()
        dt = np.zeros(3)
        dt[k] = eps_t
        Jp[:, 3 + k] = ((_project(im, p, Xc + dt) - uv) / (sig * eps_t)).ravel()
    Jg = np.zeros((r.size, len(law.x)))
    ch = law.chain(im["eye"], im["T"], early)
    for name, eps in (("fx", 1e-3), ("fy", 1e-3), ("k1", 1e-7), ("k2", 1e-7)):
        if not ch[name]:
            continue
        q = p.copy()
        q[_IDX[name]] += eps
        col = ((_project(im, q, Xc) - uv) / (sig * eps)).ravel()
        for gi, dv in ch[name].items():
            Jg[:, gi] += col * dv
    mask = np.repeat(ok & np.all(np.isfinite(uv), axis=1), 2)
    r[~mask] = 0.0
    Jp[~mask] = 0.0
    Jg[~mask] = 0.0
    return {"r": r, "Jp": Jp, "Jg": Jg}


def _huber_w(r: np.ndarray) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-12))


def solve(images: List[Dict[str, Any]], law: Law, iterations: int = ITERATIONS, verbose: bool = True) -> Dict[str, Any]:
    """Gauss-Newton with the poses eliminated per image; Huber weights.  Updates ``law.x`` and the images' poses.
    Returns {"cov" (G, G), "sigma0", "rms_px", "median_px", "iterations"}."""
    G = len(law.x)
    free = np.array([n not in law.held for n in law.names])
    lam = 1e-6
    last = None
    for it in range(iterations):
        S = np.zeros((G, G))
        b = np.zeros(G)
        cost = 0.0
        per = []
        for im in images:
            s = image_system(im, law)
            w = _huber_w(s["r"])
            cost += float(np.sum(w * s["r"] ** 2))
            Jp, Jg, r = s["Jp"], s["Jg"], s["r"]
            A = (Jp * w[:, None]).T @ Jp + 1e-9 * np.eye(6)
            B = (Jp * w[:, None]).T @ Jg
            C = (Jg * w[:, None]).T @ Jg
            bp = -(Jp * w[:, None]).T @ r
            bg = -(Jg * w[:, None]).T @ r
            Ai = np.linalg.inv(A)
            S += C - B.T @ Ai @ B
            b += bg - B.T @ Ai @ bp
            per.append((Ai, B, bp))
        Sf = S[np.ix_(free, free)]
        dg = np.zeros(G)
        dg[free] = np.linalg.solve(Sf + lam * np.diag(np.diag(Sf) + 1e-12), b[free])
        law.x = law.x + dg
        for im, (Ai, B, bp) in zip(images, per):
            dp = Ai @ (bp - B @ dg)
            Rw = _exp(dp[:3])
            im["R"] = Rw @ im["R"]
            im["t"] = Rw @ im["t"] + dp[3:]
        if verbose:
            log.info(f"  hazcam joint it {it + 1}: robust cost {cost:.1f}, step {np.max(np.abs(dg[free] / (np.abs(law.x[free]) + 1e-9))):.2e}")
        if last is not None and abs(last - cost) < 1e-6 * max(cost, 1.0):
            break
        last = cost
    # statistics and the covariance at the solution
    S = np.zeros((G, G))
    rr, ww = [], []
    for im in images:
        s = image_system(im, law)
        w = _huber_w(s["r"])
        Jp, Jg = s["Jp"], s["Jg"]
        A = (Jp * w[:, None]).T @ Jp + 1e-9 * np.eye(6)
        B = (Jp * w[:, None]).T @ Jg
        S += (Jg * w[:, None]).T @ Jg - B.T @ np.linalg.inv(A) @ B
        sig = SIGMA_PX / max(im["scale"], 1e-6)
        rr.append(s["r"] * sig)
        ww.append(w * s["r"] ** 2)
        im["residual_px"] = float(np.median(np.hypot(*(s["r"] * sig).reshape(-1, 2).T)))
    r_all = np.concatenate(rr) if rr else np.zeros(0)
    n_obs = r_all.size
    dof = max(n_obs - int(free.sum()) - 6 * len(images), 1)
    sigma0 = float(np.sqrt(np.sum(np.concatenate(ww)) / dof)) if ww else float("nan")
    cov = np.full((G, G), np.nan)
    try:
        Sf = S[np.ix_(free, free)]
        cov_f = np.linalg.inv(Sf) * sigma0 ** 2
        cov[np.ix_(free, free)] = cov_f
    except np.linalg.LinAlgError:
        pass
    pt = np.hypot(*r_all.reshape(-1, 2).T) if n_obs else np.zeros(0)
    return {"cov": cov, "sigma0": sigma0, "rms_px": float(np.sqrt(np.mean(pt ** 2))) if pt.size else None,
            "median_px": float(np.median(pt)) if pt.size else None, "iterations": it + 1, "observations": int(n_obs // 2)}


# ====================================================================================== bins
def temperature_bins(images: List[Dict[str, Any]], bin_deg: float = BIN_DEG, min_images: int = BIN_MIN_IMAGES,
                     max_deg: float = BIN_MAX_DEG) -> Dict[Tuple[str, str], List[List[int]]]:
    """Per (block, eye) the image indices grouped by temperature: fine bins of ``bin_deg`` from T0, adjacent ones
    merged until a group has ``min_images``, at most ``max_deg`` wide (a short remainder joins the group below when
    that stays within the width)."""
    out: Dict[Tuple[str, str], List[List[int]]] = {}
    keys = sorted({(im["block"], im["eye"]) for im in images})
    for k in keys:
        idx = sorted([i for i, im in enumerate(images) if (im["block"], im["eye"]) == k], key=lambda i: images[i]["T"])
        fine: Dict[int, List[int]] = {}
        for i in idx:
            fine.setdefault(int(np.floor((images[i]["T"] - T0) / bin_deg)), []).append(i)
        groups: List[List[int]] = []
        cur: List[int] = []
        lo = None
        for b in sorted(fine):
            if cur and (b - lo + 1) * bin_deg > max_deg:
                groups.append(cur)
                cur, lo = [], None
            if lo is None:
                lo = b
            cur += fine[b]
            if len(cur) >= min_images:
                groups.append(cur)
                cur, lo = [], None
        if cur:
            if groups and (max(images[i]["T"] for i in cur) - min(images[i]["T"] for i in groups[-1])) <= max_deg:
                groups[-1] += cur
            else:
                groups.append(cur)
        out[k] = groups
    return out


def bin_rows(images: List[Dict[str, Any]], law: Law, bins: Dict[Tuple[str, str], List[List[int]]]) -> List[Dict[str, Any]]:
    """Per bin the focal scale about the law (one parameter for fx and fy; poses and every other term held): ppm."""
    rows = []
    for (block, eye), groups in bins.items():
        for g in groups:
            num = den = 0.0
            nobs = 0
            for i in g:
                im = images[i]
                early = im["sol"] < EARLY_SOL
                p = law.params(eye, im["T"], early)
                Xc = im["X"] @ im["R"].T + im["t"]
                uv = _project(im, p, Xc)
                q = p.copy()
                q[0] *= 1 + 1e-6
                q[1] *= 1 + 1e-6
                d = (_project(im, q, Xc) - uv).ravel() / 1e-6          # px per unit of relative scale
                r = (im["xy"] - uv).ravel()
                w = (max(im["scale"], 1e-6) / SIGMA_PX) ** 2 * _huber_w(r * max(im["scale"], 1e-6) / SIGMA_PX)
                ok = np.isfinite(d) & np.isfinite(r)
                num += float(np.sum((w * d * r)[ok]))
                den += float(np.sum((w * d * d)[ok]))
                nobs += int(ok.sum() // 2)
            if den <= 0:
                continue
            m = num / den
            Ts = [images[i]["T"] for i in g]
            Tm = float(np.median(Ts))
            p_law = law.params(eye, Tm, images[g[0]]["sol"] < EARLY_SOL)
            rows.append({"block": block, "eye": eye, "T": Tm, "T_range": [float(min(Ts)), float(max(Ts))],
                         "sol": float(np.median([images[i]["sol"] for i in g])), "images": len(g),
                         "observations": nobs, "ppm": 1e6 * m, "sd_ppm": 1e6 / np.sqrt(den),
                         "fx": float(p_law[0] * (1 + m)), "fy": float(p_law[1] * (1 + m)), "k1": float(p_law[4])})
    return rows


# ====================================================================================== the fit
def _early_eyes(images: List[Dict[str, Any]], eyes: Sequence[str]) -> List[str]:
    out = []
    for e in eyes:
        ims = [im for im in images if im["eye"] == e]
        ea = [im for im in ims if im["sol"] < EARLY_SOL]
        la = [im for im in ims if im["sol"] >= EARLY_SOL]
        if (len(ea) >= MIN_EARLY_IMAGES and len(la) >= MIN_EARLY_IMAGES and len({im["block"] for im in ea}) >= 2
                and len({im["block"] for im in la}) >= 2):
            out.append(e)
    return out


def run_joint(data: Dict[str, Any], slope_mode: str = "FL", slope_ppm: float = 20.0, early: bool = True,
              iterations: int = ITERATIONS, verbose: bool = True) -> Dict[str, Any]:
    """The joint on collected data (:func:`collect`): the law, its covariance and statistics; the early offsets
    kept only where significant (then the law fitted again with the others at 0)."""
    import copy
    images0 = data["images"]
    eyes = [e for e in EYES if sum(im["eye"] == e for im in images0) >= MIN_EYE_IMAGES and e in data["start"]]
    images = [dict(im) for im in images0 if im["eye"] in eyes]
    for im in images:
        im["R"], im["t"] = im["R"].copy(), im["t"].copy()
    ee = _early_eyes(images, eyes) if early else []
    law = Law(eyes, data["start"], slope_ppm, slope_mode, early_eyes=ee)

    def span(es):
        T = [im["T"] for im in images if im["eye"] in es]
        return float(np.ptp(T)) if T else 0.0

    # a slope needs a temperature span (as the block fit: hazfit.MIN_T_SPAN_DEGC); else it is held at 0
    if slope_mode == "per_eye":
        for e in eyes:
            if span([e]) < MIN_T_SPAN_DEGC:
                law.x[law.ix[f"s_{e}"]] = 0.0
                law.held.add(f"s_{e}")
    elif span(SLOPE_EYES if slope_mode == "FL" else eyes) < MIN_T_SPAN_DEGC:
        law.x[law.ix["s"]] = 0.0
        law.held.add("s")
    for e in eyes:
        if f"k1s_{e}" in law.ix and span([e]) < MIN_T_SPAN_DEGC:
            law.held.add(f"k1s_{e}")
    st = solve(images, law, iterations, verbose)
    z: Dict[str, float] = {}
    for e in ee:
        for a in ("x", "y"):
            n = f"E{a}_{e}"
            sd = float(np.sqrt(st["cov"][law.ix[n], law.ix[n]]))
            z[n] = float(law.x[law.ix[n]] / sd) if sd > 0 and np.isfinite(sd) else 0.0
    insignificant = [n for n, v in z.items() if abs(v) < EARLY_Z_MIN]
    if insignificant:
        for n in insignificant:
            law.x[law.ix[n]] = 0.0
            law.held.add(n)
        if verbose:
            log.info("hazcam joint: early-mission offsets not significant (|z| < %g), set to 0: %s"
                     % (EARLY_Z_MIN, ", ".join(f"{n} z {z[n]:+.1f}" for n in insignificant)))
        st = solve(images, law, iterations, verbose)
    return {"law": law, "stats": st, "images": images, "eyes": eyes, "early_z": z, "early_eyes": ee,
            "early_held": insignificant}


def _sd(law: Law, cov: np.ndarray, name: str) -> Optional[float]:
    if name not in law.ix:
        return None
    v = cov[law.ix[name], law.ix[name]]
    return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else None


def eye_result(j: Dict[str, Any], eye: str, rows: List[Dict[str, Any]], start: np.ndarray,
               comparison: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One eye of the joint in hazfit's layout (camera_file, eye_checks)."""
    law, cov = j["law"], j["stats"]["cov"]
    x = law.x
    sn = law.slope_name(eye)
    s = float(x[law.ix[sn]])
    p = law.params(eye, T0, False)
    ims = [im for im in j["images"] if im["eye"] == eye]
    T = np.array([im["T"] for im in ims], float)
    er = [r for r in rows if r["eye"] == eye]
    w = np.array([r["observations"] for r in er], float)
    ppm = np.array([r["ppm"] for r in er], float)
    scatter = float(np.sqrt(np.average(ppm ** 2, weights=w))) if len(er) and w.sum() > 0 else float("nan")
    ex, ey = (f"Ex_{eye}", f"Ey_{eye}")
    applied = [t for t, n in (("fx", ex), ("fy", ey)) if n in law.ix and n not in law.held]
    sd_s = _sd(law, cov, sn)
    src_note = ("own slope" if law.mode == "per_eye" else "slope shared by all four eyes" if law.mode == "all"
                else "FL's slope (Christian 24 Oct: the same thermal slope front and rear)")
    out = {"eye": eye, "method": "joint", "blocks": sorted({im["block"] for im in ims}), "weak_blocks": [],
           "outlier_blocks": [], "params": [float(v) for v in p], "ppm_per_degC": 1e6 * s,
           "sd_ppm_per_degC": None if sd_s is None else 1e6 * sd_s,
           "slope_source": "FL" if law.mode == "FL" else law.mode, "slope_note": src_note,
           "k1_per_degC": float(x[law.ix[f"k1s_{eye}"]]) if f"k1s_{eye}" in law.ix else 0.0,
           "sd_k1_per_degC": _sd(law, cov, f"k1s_{eye}"),
           "early_mission": {"dfx_px": float(x[law.ix[ex]]) if ex in law.ix else 0.0,
                             "dfy_px": float(x[law.ix[ey]]) if ey in law.ix else 0.0,
                             "fitted": bool(applied), "applied_terms": applied,
                             "z": {t: j["early_z"].get(n) for t, n in (("fx", ex), ("fy", ey)) if n in j["early_z"]},
                             "rule": f"kept only where |z| >= {EARLY_Z_MIN:g} (else 0)",
                             "blocks_early": len({im["block"] for im in ims if im["sol"] < EARLY_SOL}),
                             "blocks_late": len({im["block"] for im in ims if im["sol"] >= EARLY_SOL})},
           "sd": {"fx": _sd(law, cov, f"Fx_{eye}"), "fy": _sd(law, cov, f"Fy_{eye}")},
           "scatter_ppm": scatter, "slope_fitted": sn not in law.held, "T_span_degC": float(np.ptp(T)) if T.size else 0.0,
           "athermal": False, "label_start": [float(v) for v in start[:6]],
           "images": len(ims), "observations": int(sum(len(im["X"]) for im in ims)),
           "median_residual_px": float(np.median([im["residual_px"] for im in ims])) if ims else None,
           "rows": er, "bins": len(er)}
    if comparison:
        out["own_slope_ppm_per_degC"] = comparison.get(eye)
    return out


def fit_hazcam_joint(blocks: Dict[str, PathLike], model: str = "cahv_ba", slope_mode: str = "FL",
                     limits: Optional[Dict[str, Any]] = None, compare: bool = True, rigs: bool = True,
                     verbose: bool = True) -> Dict[str, Any]:
    """The Hazcam law from the joint (with the per-eye slopes for comparison when ``compare``) in hazfit's result
    layout, the rigs from the blocks (:func:`mppp.calib.hazfit.fit_rig`).  An eye with too few images in the joint
    comes from the block fit (:func:`mppp.calib.hazfit.fit_eye_with_fallback`)."""
    from . import hazfit as H
    data = collect(blocks, model)
    counts = ", ".join(f"{e} {sum(im['eye'] == e for im in data['images'])}" for e in EYES)
    log.info(f"hazcam joint: {len(data['images'])} Hazcam images with held observations in {len(data['blocks'])} "
             f"blocks ({counts})")
    j = run_joint(data, slope_mode=slope_mode, verbose=verbose)
    bins = temperature_bins(j["images"])
    rows = bin_rows(j["images"], j["law"], bins)
    comp: Dict[str, Any] = {}
    if compare and slope_mode != "per_eye":
        try:
            jp = run_joint(data, slope_mode="per_eye", verbose=False)
            for e in jp["eyes"]:
                n = f"s_{e}"
                comp[e] = {"ppm_per_degC": 1e6 * float(jp["law"].x[jp["law"].ix[n]]),
                           "sd": None if _sd(jp["law"], jp["stats"]["cov"], n) is None
                           else 1e6 * _sd(jp["law"], jp["stats"]["cov"], n)}
        except Exception as e:                              # noqa: BLE001 - a comparison only
            comp = {"error": f"{type(e).__name__}: {e}"}
    block_res = H.fit_hazcam(blocks, model, limits) if rigs else {"eyes": {}, "rigs": {}}
    eyes = {}
    for e in EYES:
        if e in j["eyes"]:
            eyes[e] = eye_result(j, e, rows, data["start"][e], comp)
        elif e in (block_res.get("eyes") or {}):
            eyes[e] = dict(block_res["eyes"][e], method="block fit (too few images in the joint)")
    st = j["stats"]
    return {"when": datetime.date.today().isoformat(), "method": "joint", "blocks": data["blocks"], "T0_degC": T0,
            "early_mission_sol": EARLY_SOL, "eyes": eyes, "limits": limits or {}, "rigs": block_res.get("rigs") or {},
            "joint": {"slope_mode": slope_mode, "slope_eyes": list(SLOPE_EYES) if slope_mode == "FL" else slope_mode,
                      "images": len(j["images"]), "observations": st["observations"], "median_px": st["median_px"],
                      "rms_px": st["rms_px"], "sigma0": st["sigma0"], "iterations": st["iterations"],
                      "early_z": j["early_z"], "early_held": j["early_held"], "per_eye_slopes": comp,
                      "bins": {"deg": BIN_DEG, "min_images": BIN_MIN_IMAGES, "max_deg": BIN_MAX_DEG, "n": len(rows)}}}


__all__ = ["collect", "run_joint", "fit_hazcam_joint", "temperature_bins", "Law"]
