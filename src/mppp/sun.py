"""
The sun as the blocks see it (0.95.0, the first piece of the paper-3 sun module - Christian, 27 Oct 2026: the
freeze reruns are the one moment the per-image sun facts are free).

* :func:`sun_vector_enu` - the label sun (``SITE_DERIVED_GEOMETRY_PARMS`` solar azimuth, clockwise from north, and
  elevation) as a unit vector in the site's East-North-Up frame.
* :func:`mars_sun_distance_au` - the Sun-Mars distance from the areocentric solar longitude L_s (Mars' orbit as an
  ellipse: a = 1.52368 au, e = 0.09340, perihelion at L_s = 251 deg; good to ~0.1 %), or from SPICE when
  ``spiceypy`` and the planetary kernels are at hand (:func:`mars_sun_distance_spice`; ``MPPP_KERNELS``); the
  ``source`` says which.  The d^2 factor of every I/F conversion and of the ECAM -> Mastcam-Z colour transform.
* :func:`block_rotation` - a solved block's rotation away from the label ENU frame (the median over its registered
  images of the world-frame rotation taking the prior attitude to the refined one, as the health check
  ``block_rotation_deg``) and :func:`sun_in_block` - the sun vector in that frame (at i = 60 deg a 1 deg block tilt
  changes mu_0 by 3 %).

Recorded per image in ``manifest.json`` (:attr:`mppp.image.MPPPImage.meta`), on the project rows and in
``error_input/stations.csv`` (``d_mars_au``, ``sun_e/n/u``, ``sun_bx/by/bz``) with ``error_input/block_frame.json``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

MARS_A_AU = 1.523679           # semi-major axis of Mars' orbit (au)
MARS_E = 0.09340               # eccentricity
MARS_LS_PERIHELION_DEG = 251.0  # L_s at perihelion
AU_KM = 149597870.7


def sun_vector_enu(solar_azimuth_deg: Optional[float], solar_elevation_deg: Optional[float]) -> Optional[np.ndarray]:
    """The unit vector towards the sun in East-North-Up from the label's azimuth (clockwise from north) and
    elevation; None without them."""
    if solar_azimuth_deg is None or solar_elevation_deg is None:
        return None
    az, el = math.radians(float(solar_azimuth_deg)), math.radians(float(solar_elevation_deg))
    return np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)], float)


def mars_sun_distance_ls(L_s_deg: float) -> float:
    """The Sun-Mars distance (au) from the solar longitude: the orbit as an ellipse, r = a (1 - e^2) / (1 + e cos nu)
    with the true anomaly nu = L_s - L_s(perihelion)."""
    nu = math.radians(float(L_s_deg) - MARS_LS_PERIHELION_DEG)
    return MARS_A_AU * (1.0 - MARS_E ** 2) / (1.0 + MARS_E * math.cos(nu))


def kernels_dir() -> Optional[Path]:
    """The folder of the planetary kernels (``MPPP_KERNELS``: a leapseconds .tls and a planetary .bsp), else None."""
    env = os.environ.get("MPPP_KERNELS")
    return Path(env) if env else None


def mars_sun_distance_spice(utc: str, kernels: Optional[Path] = None) -> Optional[float]:
    """The Sun-Mars distance (au) at ``utc`` (an ISO time) from SPICE (``spiceypy``; the kernels of ``kernels`` or
    :func:`kernels_dir`: every ``*.tls`` and ``*.bsp`` there); None when spiceypy or the kernels are missing or the
    time is not covered."""
    kd = Path(kernels) if kernels else kernels_dir()
    if kd is None or not kd.is_dir() or not utc:
        return None
    try:
        import spiceypy as sp
    except ImportError:
        return None
    files = sorted(kd.glob("*.tls")) + sorted(kd.glob("*.bsp"))
    if not any(f.suffix == ".tls" for f in files) or not any(f.suffix == ".bsp" for f in files):
        return None
    try:
        for f in files:
            sp.furnsh(str(f))
        et = sp.str2et(str(utc).replace("Z", ""))
        pos, _ = sp.spkpos("MARS BARYCENTER", et, "J2000", "NONE", "SUN")
        return float(np.linalg.norm(pos) / AU_KM)
    except Exception:                                      # noqa: BLE001 - kernels not covering, a bad time
        return None
    finally:
        try:
            for f in files:
                sp.unload(str(f))
        except Exception:                                  # noqa: BLE001
            pass


def mars_sun_distance_au(L_s_deg: Optional[float], utc: Optional[str] = None,
                         spice: bool = True) -> Tuple[Optional[float], Optional[str]]:
    """(the Sun-Mars distance in au, its source): ``"spice"`` when :func:`mars_sun_distance_spice` answers (``spice``
    True and ``utc`` given), else ``"ls_ellipse"`` from ``L_s_deg``; (None, None) without either."""
    if spice and utc:
        d = mars_sun_distance_spice(utc)
        if d is not None:
            return d, "spice"
    if L_s_deg is not None:
        return mars_sun_distance_ls(L_s_deg), "ls_ellipse"
    return None, None


def sun_record(meta: Dict[str, Any], spice: bool = True) -> Dict[str, Any]:
    """The per-image sun facts from a manifest / project record (``solar_azimuth_deg``, ``solar_elevation_deg``,
    ``L_s_deg``, ``start_time_utc``): {"d_mars_au", "d_mars_source", "sun_enu" [e, n, u] or None}."""
    d, src = mars_sun_distance_au(meta.get("L_s_deg"), meta.get("start_time_utc") if spice else None, spice=spice)
    v = sun_vector_enu(meta.get("solar_azimuth_deg"), meta.get("solar_elevation_deg"))
    return {"d_mars_au": d, "d_mars_source": src, "sun_enu": None if v is None else [float(x) for x in v]}


def block_rotation(rec, project, min_observations: int = 30) -> Optional[Dict[str, Any]]:
    """A solved block's rotation away from the label ENU frame: the median over its registered images (at least
    ``min_observations`` observations) of the world-frame rotation W = R1^T R0 taking the prior attitude (``R0``,
    ``prior_R_w2c``) to the refined one (``R1``).  A sun vector known in label ENU (s) appears in the refined block as
    W s.  Returns {"R_block_from_enu" (3x3 list), "rotvec_deg" [about E, N, U], "angle_deg", "images"} or None."""
    from scipy.spatial.transform import Rotation
    by_name = {r["name"]: r for r in project.images}
    rv = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        r = by_name.get(im.name)
        n3 = im.num_points3D if isinstance(im.num_points3D, int) else im.num_points3D()
        if r is None or r.get("prior_R_w2c") is None or n3 < int(min_observations):
            continue
        R1 = im.cam_from_world().rotation.matrix()
        R0 = np.asarray(r["prior_R_w2c"], float)
        rv.append(Rotation.from_matrix(R1.T @ R0).as_rotvec())
    if not rv:
        return None
    med = np.median(np.asarray(rv), axis=0)
    W = Rotation.from_rotvec(med).as_matrix()
    return {"R_block_from_enu": W.tolist(), "rotvec_deg": np.degrees(med).tolist(),
            "angle_deg": float(np.degrees(np.linalg.norm(med))), "images": len(rv)}


def sun_in_block(sun_enu: Optional[Sequence[float]], R_block_from_enu: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    """The sun vector in the block's frame: R_block_from_enu @ sun_enu (None without either)."""
    if sun_enu is None or R_block_from_enu is None:
        return None
    return np.asarray(R_block_from_enu, float) @ np.asarray(sun_enu, float)
