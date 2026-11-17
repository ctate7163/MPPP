"""
rpp_studies.msl.first_block - the first Curiosity (MSL) block: fetch, prepare, align, report (0.99.4).

Christian, 16 Nov 2026: "Prepare a test msl block".  The v0p99 plan (section 3.3) picks site 87 (sols ~3047-3082,
the B-side cameras), where the sample in ``D:\\code\\MPPP\\data\\msl`` is from.

    python -m rpp_studies.msl.first_block fetch   --sols 3047 3082 --site 87 [--list]      (on the PC: PDS)
    python -m rpp_studies.msl.first_block prepare --work D:\\rover_blocks\\msl_sol_3047_mont_mercou
    python -m rpp_studies.msl.first_block align   --work D:\\rover_blocks\\msl_sol_3047_mont_mercou
    python -m rpp_studies.msl.first_block all     --sols 3047 3082 --site 87                (the three in turn)

**fetch.**  The Navcam RDRs (``MSLNAV_1XXX``) of the sols from the PDS Imaging Node, one folder per sol under
``--data`` (``D:\\data\\msl\\navcam\\SOLnnnnn``): every full-frame Navcam radiance product (``RAD``) of the site and
side, both eyes, with its detached label, and MIPL's rover mask (``MXY``) of the same frame where there is one.
Nothing is downloaded twice; ``--list`` shows what would come and from which address (the archive's layout is
looked up, not assumed: several layouts are tried and the one that answers is reported).

**prepare.**  From the downloaded products, into ``--work``:

- the stations (site, drive) with a stereo 360 (at least ``--min-pairs`` full-frame stereo pairs); the block is the
  station with the most pairs and the stations within ``--radius-m`` of it (label positions, one site frame);
- ``images/<product>.png``: the radiance as 8 bits, stretched per image (0.5 to 99.8 percent);
- ``masks/<product>.png.png``: COLMAP's mask, 0 where MIPL's rover mask (MXY, 255 = the rover and its arm) says
  rover - the rover's shadow is not in it; a frame without an MXY product is used unmasked (the report says so);
- ``cameras.json``: one camera per eye and computer side (NLB, NRB, ...), the COLMAP ``THIN_PRISM_FISHEYE`` fitted to
  the label CAHVOR (:func:`rpp.calib.labelfit.fisheye_from_cahvore`; the median over the group's frames);
- ``manifest.json``: per frame its sol, LMST, station, side, CCD temperature (the side's sensor) and the camera's
  center in the site frame (east, north, up; from the label's CAHVOR in the rover frame and the rover's pose in the
  site frame, as the M2020 pipeline does: :func:`rpp.camera.pose_from_label`).

**align.**  COLMAP (pycolmap): SIFT on the masked frames (one camera per group, the label intrinsics held), all
pairs matched (a first block is small), the incremental mapper; the model is then put into the site frame by a
similarity onto the label camera centers (RANSAC).  ``report.json`` / ``report.txt``: frames registered, points,
reprojection error, the similarity's scale and residuals, and the stereo baseline of each registered pair against
the label's.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("rpp_studies.msl")

DEFAULT_DATA = r"D:\data\msl"
DEFAULT_ROOT = r"D:\rover_blocks"            # 0.99.12: msl_sol_<first sol>_<block> with the Mars 2020 blocks
DEFAULT_BLOCK = "mont_mercou"
DEFAULT_NAME = "msl_sol_3047_mont_mercou"
BASES = ("https://planetarydata.jpl.nasa.gov/img/data/msl/MSLNAV_1XXX",
         "https://pds-imaging.jpl.nasa.gov/data/msl/MSLNAV_1XXX")
# 0.99.16 (Christian, 17 Nov 2026: "please include front hazcam images in the msl block"): the Hazcams' volume, laid out
# as the Navcams'; its products go to <data>\hazcam\SOLnnnnn
HAZ_BASES = tuple(b.replace("MSLNAV_1XXX", "MSLHAZ_1XXX") for b in BASES)
VOLUMES = (("N", "navcam", BASES), ("FR", "hazcam", HAZ_BASES))
# DATA holds the newest version of each product (V2 from 2021); the volume's V1 products of releases 1-17 moved to
# DATA_V1 (MSL_NAVCAM_RDR_V1_DS.CAT)
LAYOUTS = ("DATA/SOL{sol:05d}/", "DATA_V1/SOL{sol:05d}/", "DATA/SOL{sol:04d}/", "data/sol{sol:05d}/", "data/SOL{sol:05d}/")
EST_MB = {"RAD": 2.17, "MXY": 1.12}          # a 1024 x 1024 Navcam RAD (int16) / MXY (uint8) with its label
UA = {"User-Agent": "rpp-msl-first-block (research; github.com/ctate7163/rpp)"}
# MSL engineering-camera product names (MSL Camera SIS): NLB_668441715RAD_F0870792NCAM00252M1
#   [0] N/F/R Navcam / front / rear Hazcam   [1] L/R eye   [2] A/B computer side   [4:13] SCLK seconds
#   [13:16] product type (RAD, MXY, XYZ, EDR ...)   [17] F full frame / S subframe / ...   [18:21] site [21:25] drive
#   [25:34] sequence   'M'   version.  No sol: it is PLANET_DAY_NUMBER in the label.
NAME = re.compile(r"^(?P<cam>(?P<fam>[NFR])(?P<eye>[LR])(?P<side>[AB]))_(?P<sclk>\d{9})(?P<prod>[A-Z0-9]{3})_"
                  r"(?P<samp>[A-Z])(?P<site>\d{3})(?P<drive>\d{4})(?P<seq>[A-Z]{4}\d{5})M(?P<ver>\d)$", re.I)
MASK_ROVER = 255                  # MXY: 255 where the pixel's XYZ falls on the rover (MSL_rover_filter.xml), else 0
STRETCH = (0.5, 99.8)


# ---------------------------------------------------------------------------------------------------- names, labels
def parse_name(name: str) -> Optional[Dict[str, Any]]:
    """The fields of an MSL engineering-camera product name (stem or file name), or None."""
    m = NAME.match(Path(name).stem.upper())
    if not m:
        return None
    d = m.groupdict()
    return {"stem": Path(name).stem.upper(), "camera": d["cam"], "family": d["fam"], "eye": d["eye"], "side": d["side"],
            "sclk": int(d["sclk"]), "product_type": d["prod"], "sampling": d["samp"], "site": int(d["site"]),
            "drive": int(d["drive"]), "sequence": d["seq"], "version": int(d["ver"]),
            "group": d["cam"],                                    # one camera per eye and side: NLB, NRB, NLA ...
            "pair_key": f"{d['fam']}{d['side']}_{d['sclk']}_{d['samp']}{d['site']}{d['drive']}{d['seq']}"}


def label_value(L: Dict[str, Any], key: str, group: Optional[str] = None):
    """A keyword of an MSL label, with or without its ``MSL:`` prefix, at the top or in ``group``."""
    from rpp.labels import label_get
    for k in (key, "MSL:" + key):
        v = label_get(L, f"{group}.{k}" if group else k)
        if v is not None:
            return v
    return None


def lmst_hours(value) -> Optional[float]:
    """``"Sol-03052M15:04:27.460"`` (MSL) or ``"15:04:27"`` -> 15.0743."""
    m = re.search(r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\s*$", str(value or ""))
    if not m:
        return None
    return int(m.group(1)) + int(m.group(2)) / 60.0 + float(m.group(3)) / 3600.0


def ccd_temperature(L: Dict[str, Any], side: str, family: str, eye: str) -> Optional[float]:
    """The frame's own CCD temperature among the 24 sensors of both sides (``B_LEFT_NAV_CCD`` ...)."""
    st = label_value(L, "INSTRUMENT_STATE_PARMS") or {}
    names, vals = st.get("INSTRUMENT_TEMPERATURE_NAME"), st.get("INSTRUMENT_TEMPERATURE")
    if not names or not vals:
        return None
    want = f"{side}_{'LEFT' if eye == 'L' else 'RIGHT'}_{'NAV' if family == 'N' else 'HAZ'}_CCD"
    if family in ("F", "R"):
        want = f"{side}_{'FRONT' if family == 'F' else 'REAR'}_{'LEFT' if eye == 'L' else 'RIGHT'}_HAZ_CCD"
    for n, v in zip(names, vals):
        if str(n).upper() == want:
            return float(v)
    return None


def product_info(img: Path) -> Dict[str, Any]:
    """Everything a frame needs from its name and label (not its pixels)."""
    from rpp.camera import CAHVOR, pose_from_label
    from rpp.labels import read_pds
    fn = parse_name(img.name)
    if fn is None:
        raise ValueError(f"not an MSL engineering-camera product: {img.name}")
    L, _ = read_pds(img, load_image=False)
    gk = next((k for k in L if str(k).upper().startswith("GEOMETRIC_CAMERA_MODEL")), None)
    rk = next((k for k in L if str(k).upper() in ("ROVER_COORD_SYSTEM_PARMS", "ROVER_COORDINATE_SYSTEM_PARMS")), None)
    if gk is None or rk is None:
        raise ValueError(f"{img.name}: no camera model or rover frame in the label")
    gcm, rcs = L[gk], L[rk]
    if str(gcm.get("REFERENCE_COORD_SYSTEM_NAME", "")).upper() != "ROVER_NAV_FRAME":
        raise ValueError(f"{img.name}: camera model in {gcm.get('REFERENCE_COORD_SYSTEM_NAME')}, not ROVER_NAV_FRAME")
    if str(rcs.get("REFERENCE_COORD_SYSTEM_NAME", "")).upper() != "SITE_FRAME":
        raise ValueError(f"{img.name}: rover frame given in {rcs.get('REFERENCE_COORD_SYSTEM_NAME')}, not SITE_FRAME")
    im = label_value(L, "IMAGE") or {}
    w, h = int(im.get("LINE_SAMPLES")), int(im.get("LINES"))
    cah = CAHVOR.from_label(gcm)
    _, R_cam_rnav = cah.decompose(w, h)
    pose = pose_from_label(R_cam_rnav, cah.C, rcs["ORIGIN_ROTATION_QUATERNION"], np.asarray(rcs["ORIGIN_OFFSET_VECTOR"], float),
                           f"site{int(rcs.get('REFERENCE_COORD_SYSTEM_INDEX', fn['site']))}_enu", "label (site frame)")
    sol = label_value(L, "PLANET_DAY_NUMBER")
    site_link = None                       # 0.99.5: this site's origin in the previous site's frame (NED, onboard)
    sk = next((k for k in L if str(k).upper() in ("SITE_COORD_SYSTEM_PARMS", "SITE_COORDINATE_SYSTEM_PARMS")), None)
    if sk is not None:
        sg = L[sk]
        try:
            site_link = {"site": int(sg.get("COORDINATE_SYSTEM_INDEX")), "parent": int(sg.get("REFERENCE_COORD_SYSTEM_INDEX")),
                         "offset_ned_m": [float(x) for x in sg.get("ORIGIN_OFFSET_VECTOR")],
                         "quaternion": [float(x) for x in sg.get("ORIGIN_ROTATION_QUATERNION")]}
        except (TypeError, ValueError):
            site_link = None
    return dict(fn, sol=int(sol) if sol is not None else None, site_link=site_link, lmst_h=lmst_hours(label_value(L, "LOCAL_MEAN_SOLAR_TIME")),
                width=w, height=h, model_type=str(gcm.get("MODEL_TYPE")), gcm=gcm,
                site_frame=int(rcs.get("REFERENCE_COORD_SYSTEM_INDEX", fn["site"])),
                C_enu_m=[float(x) for x in pose.C], R_w2c=np.asarray(pose.R_w2c, float).tolist(),
                C_rnav_m=[float(x) for x in cah.C], ccd_degC=ccd_temperature(L, fn["side"], fn["family"], fn["eye"]),
                radiance_scale=label_value(L, "RADIANCE_SCALING_FACTOR", "DERIVED_IMAGE_PARMS"),
                mode=str((label_value(L, "INSTRUMENT_STATE_PARMS") or {}).get("INSTRUMENT_MODE_ID") or ""),
                derived_type=str(label_value(L, "DERIVED_IMAGE_TYPE", "DERIVED_IMAGE_PARMS") or ""))


# ---------------------------------------------------------------------------------------------------- fetch
def _get(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def listing(url: str) -> List[str]:
    """The file names an archive directory page links to."""
    text = _get(url).decode("utf-8", "replace")
    names = (html.unescape(h).rstrip().rsplit("/", 1)[-1] for h in re.findall(r'href\s*=\s*["\']([^"\'?#]+)["\']', text, re.I))
    return sorted({n for n in names if n})                     # folders (a trailing /) and the parent link drop out


def find_sol_dir(sol: int, bases: Sequence[str] = BASES, layouts: Sequence[str] = LAYOUTS) -> Tuple[Optional[str], List[str], List[str]]:
    """(the address of the sol's folder, its file names, the addresses tried)."""
    tried = []
    for b in bases:
        for lay in layouts:
            url = f"{b.rstrip('/')}/{lay.format(sol=sol)}"
            tried.append(url)
            try:
                names = listing(url)
            except Exception:                                        # noqa: BLE001 - the next layout
                continue
            if any(parse_name(n) for n in names):
                return url, names, tried
    return None, [], tried


def side_ok(f: Dict[str, Any], side: Optional[str]) -> bool:
    """``side``: "A", "B", or None / "AB" for both (0.99.5)."""
    return not side or str(side).upper() in ("AB", "BOTH") or f["side"] == str(side).upper()


def wanted(names: Iterable[str], site: Optional[int], side: Optional[str], drives: Optional[Sequence[int]] = None,
           families: str = "N", mxy_too: bool = True) -> List[str]:
    """The stems to fetch from a sol folder: full-frame RAD of the site (None: every site), side ("A", "B", None / "AB":
    both) and families (both eyes), with the MXY of the same frames."""
    rad, mxy = set(), set()
    for n in names:
        if not n.upper().endswith(".IMG"):
            continue
        f = parse_name(n)
        if not f or (site is not None and f["site"] != site) or f["family"] not in families or f["sampling"] != "F":
            continue
        if not side_ok(f, side):
            continue
        if drives and f["drive"] not in set(drives):
            continue
        (rad if f["product_type"] == "RAD" else mxy if f["product_type"] == "MXY" else set()).add(f["stem"])
    keep_mxy = {s for s in mxy if s.replace("MXY_", "RAD_", 1) in rad} if mxy_too else set()
    return sorted(rad) + sorted(keep_mxy)


def fetch(data: Path, sols: Sequence[int], site: int, side: Optional[str] = "B", drives=None, list_only: bool = False,
          families: str = "N", mxy: bool = True) -> Dict[str, Any]:
    """Download the sols' products into ``data/navcam/SOLnnnnn`` (Navcams) and ``data/hazcam/SOLnnnnn`` (Hazcams, when
    ``families`` has F or R; 0.99.16).  ``mxy=False``: the RAD frames only, without MIPL's rover masks.  Returns a
    report; ``sols`` is keyed by the sol (Navcam volume) and by ``"<sol> hazcam"``."""
    rep: Dict[str, Any] = {"sols": {}, "downloaded": 0, "kept": 0, "bytes": 0, "failed": []}
    for fams, folder, bases in VOLUMES:
        fam = "".join(f for f in families if f in fams)
        if not fam:
            continue
        for sol in sols:
            key = sol if folder == "navcam" else f"{sol} {folder}"
            url, names, tried = find_sol_dir(int(sol), bases)
            if url is None:
                rep["sols"][key] = {"found": False, "tried": tried}
                log.warning("sol %d (%s): no archive folder answered (%s)", sol, folder, ", ".join(tried))
                continue
            stems = wanted(names, site, side, drives, fam, mxy)
            lower = {n.upper(): n for n in names}
            n_mxy = sum(1 for x in stems if "MXY_" in x)
            rep["sols"][key] = {"found": True, "url": url, "products": len(stems), "rad": len(stems) - n_mxy, "mxy": n_mxy}
            rep["est_mb"] = round(rep.get("est_mb", 0.0) + EST_MB["RAD"] * (len(stems) - n_mxy) + EST_MB["MXY"] * n_mxy, 1)
            log.info("sol %d (%s): %d products at %s", sol, folder, len(stems), url)
            if list_only:
                rep["sols"][key]["stems"] = stems
                continue
            out = Path(data) / folder / f"SOL{int(sol):05d}"
            out.mkdir(parents=True, exist_ok=True)
            for st in stems:
                for ext in (".IMG", ".LBL"):
                    remote = lower.get((st + ext).upper())
                    if remote is None:
                        continue
                    dst = out / remote
                    if dst.is_file() and dst.stat().st_size > 0:
                        rep["kept"] += 1
                        continue
                    for k in range(3):
                        try:
                            b = _get(url + remote, timeout=300)
                            tmp = dst.with_suffix(dst.suffix + ".part")
                            tmp.write_bytes(b)
                            tmp.replace(dst)
                            rep["downloaded"] += 1
                            rep["bytes"] += len(b)
                            break
                        except Exception as e:                           # noqa: BLE001
                            if k == 2:
                                rep["failed"].append(f"{url + remote}: {e}")
                            time.sleep(5 * (k + 1))
    return rep


# ---------------------------------------------------------------------------------------------------- prepare
def find_products(data: Path, site: Optional[int], side: Optional[str], families: str = "N",
                  sols: Optional[Sequence[int]] = None) -> Tuple[List[Path], Dict[str, Path]]:
    """(the full-frame RAD products of the site (None: all) and side under ``data``, {RAD stem: its MXY file});
    ``sols``: only the ``SOLnnnnn`` folders of these sols (0.99.5)."""
    rad, mxy = [], {}
    want = None if sols is None else {int(x) for x in sols}
    for p in sorted(Path(data).rglob("*")):
        if p.suffix.upper() != ".IMG" or not p.is_file():
            continue
        if want is not None:
            m = re.fullmatch(r"SOL(\d{4,5})", p.parent.name.upper())
            if not m or int(m.group(1)) not in want:
                continue
        f = parse_name(p.name)
        if not f or (site is not None and f["site"] != site) or f["family"] not in families or f["sampling"] != "F":
            continue
        if not side_ok(f, side):
            continue
        if f["product_type"] == "RAD":
            rad.append(p)
        elif f["product_type"] == "MXY":
            mxy[f["stem"].replace("MXY_", "RAD_", 1)] = p
    return rad, mxy


def choose_block(infos: List[Dict[str, Any]], min_pairs: int = 8, radius_m: float = 50.0,
                 station: Optional[Tuple[int, int]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The block's frames: the stations with at least ``min_pairs`` stereo pairs; the anchor station (``station``, else
    the one with the most pairs) and every such station within ``radius_m`` of it."""
    by_st: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in infos:
        by_st.setdefault((r["site"], r["drive"]), []).append(r)
    pairs = {k: len({r["pair_key"] for r in v if r["eye"] == "L"} & {r["pair_key"] for r in v if r["eye"] == "R"})
             for k, v in by_st.items()}
    center = {k: np.mean([r["C_enu_m"] for r in v], axis=0) for k, v in by_st.items()}
    ok = [k for k, n in pairs.items() if n >= min_pairs]
    if not ok:
        raise ValueError(f"no station with {min_pairs} stereo pairs (pairs per station: {pairs})")
    anchor = tuple(station) if station else max(ok, key=lambda k: (pairs[k], -k[1]))
    if anchor not in by_st:
        raise ValueError(f"station {anchor}: no frames")
    keep = [k for k in ok if np.linalg.norm((center[k] - center[anchor])[:2]) <= radius_m]
    if anchor not in keep:
        keep.append(anchor)
    frames = [r for k in sorted(keep) for r in by_st[k]]
    rep = {"anchor": f"S{anchor[0]:03d}D{anchor[1]:04d}", "stations": [
        {"station": f"S{k[0]:03d}D{k[1]:04d}", "frames": len(by_st[k]), "pairs": pairs[k],
         "distance_m": round(float(np.linalg.norm((center[k] - center[anchor])[:2])), 1),
         "sols": sorted({r["sol"] for r in by_st[k]}), "in_block": k in keep} for k in sorted(by_st)]}
    return frames, rep


def stretch8(arr: np.ndarray, lo_hi=STRETCH) -> np.ndarray:
    a = np.asarray(arr, np.float32)
    v = a[np.isfinite(a) & (a > 0)]
    if v.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(v, lo_hi)
    return np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def group_camera(infos: List[Dict[str, Any]], step: float = 16.0) -> Dict[str, Any]:
    """The COLMAP camera of a group: the THIN_PRISM_FISHEYE fitted to each frame's label CAHVOR, median parameters."""
    from rpp.calib.labelfit import fisheye_from_cahvore
    from rpp.cmod import CameraModel
    ps, rms = [], []
    seen = set()
    for r in infos:
        key = json.dumps(r["gcm"], sort_keys=True, default=str)
        if key in seen:                                          # the same model (interpolated by temperature): once
            continue
        seen.add(key)
        cm = CameraModel.from_label(r["gcm"], r["width"], r["height"])
        mc, _ = cm.camera_frame()
        p, rep = fisheye_from_cahvore(mc, r["width"], r["height"], step=step)
        ps.append(p)
        rms.append(rep["rms_px"])
    P = np.median(np.asarray(ps, float), axis=0)
    return {"model": "THIN_PRISM_FISHEYE", "params": [float(x) for x in P], "width": infos[0]["width"],
            "height": infos[0]["height"], "fit_rms_px_max": float(max(rms)), "label_models": len(ps),
            "frames": len(infos), "source": "label CAHVOR fit (rpp.calib.labelfit.fisheye_from_cahvore), median"}


def prepare(work: Path, data: Path, site: int, side: Optional[str] = "B", min_pairs: int = 8, radius_m: float = 50.0,
            station: Optional[Tuple[int, int]] = None, families: str = "N") -> Dict[str, Any]:
    """The block's images, masks, cameras and manifest in ``work`` (see the module's text)."""
    import cv2
    from rpp.labels import read_pds
    work = Path(work)
    rad, mxy = find_products(data, site, side, families)
    if not rad:
        raise ValueError(f"no full-frame RAD products of site {site} (side {side}) under {data}")
    infos = []
    for p in rad:
        try:
            r = product_info(p)
        except Exception as e:                                       # noqa: BLE001
            log.warning("%s: left out (%s)", p.name, e)
            continue
        r["path"] = str(p)
        infos.append(r)
    frames, rep = choose_block(infos, min_pairs, radius_m, station)
    for d in ("images", "masks"):
        (work / d).mkdir(parents=True, exist_ok=True)
    unmasked = []
    for r in frames:
        _, a = read_pds(r["path"])
        a = np.squeeze(a)
        name = r["stem"] + ".png"
        cv2.imwrite(str(work / "images" / name), stretch8(a))
        m = mxy.get(r["stem"])
        if m is not None:
            _, mk = read_pds(m)
            mk = np.squeeze(mk)
            cv2.imwrite(str(work / "masks" / (name + ".png")), np.where(mk == MASK_ROVER, 0, 255).astype(np.uint8))
            r["mask"] = m.name
            r["rover_share"] = float(np.mean(mk == MASK_ROVER))
        else:
            cv2.imwrite(str(work / "masks" / (name + ".png")), np.full(a.shape[:2], 255, np.uint8))
            r["mask"] = None
            unmasked.append(r["stem"])
        r["image"] = name
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in frames:
        groups.setdefault(r["group"], []).append(r)
    cameras = {g: group_camera(v) for g, v in sorted(groups.items())}
    keep = ("stem", "image", "camera", "group", "family", "eye", "side", "sol", "lmst_h", "sclk", "site", "drive",
            "sequence", "pair_key", "C_enu_m", "R_w2c", "C_rnav_m", "ccd_degC", "radiance_scale", "site_frame", "mask",
            "rover_share", "width", "height", "path")
    man = {"mission": "MSL", "created": datetime.datetime.now().isoformat(timespec="seconds"), "site": site, "side": side,
           "data": str(data), "block": rep, "cameras": cameras, "unmasked": unmasked,
           "images": [{k: r.get(k) for k in keep} for r in frames]}
    (work / "cameras.json").write_text(json.dumps(cameras, indent=1), encoding="utf-8")
    (work / "manifest.json").write_text(json.dumps(man, indent=1, default=float), encoding="utf-8")
    write_open_in_colmap(work)                                   # 0.99.15: there from the start
    log.info("prepare: %d frames of %d stations (anchor %s), %d without a rover mask; cameras %s", len(frames),
             sum(s["in_block"] for s in rep["stations"]), rep["anchor"], len(unmasked),
             ", ".join(f"{g} ({c['frames']}, fit {c['fit_rms_px_max']:.3f} px)" for g, c in cameras.items()))
    return man


# ---------------------------------------------------------------------------------------------------- align
def stereo_baselines(images: List[Dict[str, Any]], centers: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Per stereo pair with both eyes registered: |C_L - C_R| of the model (site frame, meters) and of the labels."""
    by = {}
    for r in images:
        by.setdefault(r["pair_key"], {})[r["eye"]] = r
    out = []
    for k, d in sorted(by.items()):
        if "L" in d and "R" in d and d["L"]["image"] in centers and d["R"]["image"] in centers:
            m = float(np.linalg.norm(centers[d["L"]["image"]] - centers[d["R"]["image"]]))
            lab = float(np.linalg.norm(np.asarray(d["L"]["C_enu_m"]) - np.asarray(d["R"]["C_enu_m"])))
            out.append({"pair": k, "model_m": m, "label_m": lab, "ratio": m / lab if lab > 0 else None})
    return out


def align(work: Path, max_features: int = 8192, threads: int = -1, refine_intrinsics: bool = False) -> Dict[str, Any]:
    """COLMAP on the prepared block (see the module's text).  Returns the report (also written)."""
    import pycolmap
    work = Path(work)
    man = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    db = work / "database.db"
    if db.exists():
        db.unlink()
    t0 = time.time()
    for g, cam in man["cameras"].items():
        names = [r["image"] for r in man["images"] if r["group"] == g]
        ro = pycolmap.ImageReaderOptions(camera_model=cam["model"], camera_params=",".join(f"{x:.10g}" for x in cam["params"]),
                                         mask_path=str(work / "masks"))
        eo = pycolmap.FeatureExtractionOptions()
        eo.sift.max_num_features = int(max_features)
        eo.num_threads = int(threads)
        pycolmap.extract_features(db, work / "images", image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=ro, extraction_options=eo)
    pycolmap.match_exhaustive(db)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = bool(refine_intrinsics)
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = bool(refine_intrinsics)
    opts.num_threads = int(threads)
    out = work / "sparse"
    out.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db, work / "images", out, options=opts)
    if not recs:
        rep = {"ok": False, "message": "COLMAP registered no model", "frames": len(man["images"])}
        (work / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        return rep
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    names = [im.name for im in rec.images.values()]
    lab = {r["image"]: np.asarray(r["C_enu_m"], float) for r in man["images"]}
    tgt = [n for n in names if n in lab]
    ro = pycolmap.RANSACOptions()
    ro.max_error = 0.5                                           # meters: the label poses' error at one site
    sim = pycolmap.align_reconstruction_to_locations(rec, tgt, np.array([lab[n] for n in tgt]), 3, ro)
    rep: Dict[str, Any] = {"ok": True, "frames": len(man["images"]), "registered": int(rec.num_reg_images()),
                           "points": int(rec.num_points3D()), "models": len(recs),
                           "reprojection_px": float(rec.compute_mean_reprojection_error()),
                           "minutes": round((time.time() - t0) / 60.0, 1)}
    if sim is not None:
        rec.transform(sim)
        rep["similarity_scale"] = float(sim.scale)
        res = []
        centers = {}
        for im in rec.images.values():
            C = np.asarray(im.projection_center(), float)
            centers[im.name] = C
            if im.name in lab:
                res.append(np.linalg.norm(C - lab[im.name]))
        rep["label_center_residual_m"] = {"median": float(np.median(res)), "p95": float(np.percentile(res, 95)), "max": float(np.max(res))}
        sb = stereo_baselines(man["images"], centers)
        if sb:
            rat = np.array([s["ratio"] for s in sb if s["ratio"]])
            rep["stereo_baseline"] = {"pairs": len(sb), "model_m_median": float(np.median([s["model_m"] for s in sb])),
                                      "label_m_median": float(np.median([s["label_m"] for s in sb])),
                                      "ratio_median": float(np.median(rat)), "ratio_sd": float(np.std(rat))}
    else:
        rep["similarity_scale"] = None
        rep["message"] = "the model could not be put into the site frame (too few frames with label positions agreeing)"
    (out / "site_frame").mkdir(exist_ok=True)
    rec.write(out / "site_frame")
    write_open_in_colmap(work)
    rec.export_PLY(str(work / "points_site_frame.ply"))
    (work / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (work / "report.txt").write_text(report_text(man, rep), encoding="utf-8")
    log.info("\n%s", report_text(man, rep))
    return rep


def write_open_in_colmap(work: "str | Path", models: Sequence[str] = ("block_frame", "site_frame", "0")) -> Path:
    """``open_in_colmap.bat`` (and ``colmap_gui.ini``) in an MSL block's folder (0.99.15; Christian, 17 Nov 2026: "Make
    sure there is always an open_in_colmap present for these alignments"): double-click opens the COLMAP GUI with the
    block's database, images and model - the first of ``sparse\\<models>`` that exists when it is started (the block
    frame, else the mapper's model), so the file can be written before the alignment and still open its result.  COLMAP.bat
    is looked for as in the Mars 2020 blocks (``rpp.sfm.database.COLMAP_BAT_CANDIDATES``, ``COLMAP_BAT``, the PATH)."""
    from rpp import __version__
    from rpp.sfm.database import COLMAP_BAT_CANDIDATES
    work = Path(work)
    lines = ["@echo off", "setlocal",
             f"rem RPP {__version__}: open this MSL block in the COLMAP GUI (4.x).  Double-click.",
             "rem COLMAP.bat is looked for at the paths below, then in COLMAP_BAT, then on the PATH.",
             'set "MODEL="']
    for m in models:
        lines.append(f'if not defined MODEL if exist "%~dp0sparse\\{m}\\cameras.bin" set "MODEL=%~dp0sparse\\{m}"')
    lines += ['if not defined MODEL (echo No model in %~dp0sparse yet: run the align step first. & pause & exit /b 1)',
              'set "CB="']
    for c in COLMAP_BAT_CANDIDATES:
        lines.append(f'if not defined CB if exist "{c}" set "CB={c}"')
    lines += ['if not defined CB if defined COLMAP_BAT if exist "%COLMAP_BAT%" set "CB=%COLMAP_BAT%"',
              'if not defined CB for %%I in (COLMAP.bat) do if not "%%~$PATH:I"=="" set "CB=%%~$PATH:I"',
              'if not defined CB (echo COLMAP.bat not found: setx COLMAP_BAT ^<path^>\\COLMAP.bat & pause & exit /b 1)',
              'echo Starting %CB% with %MODEL%',
              'call "%CB%" gui --database_path "%~dp0database.db" --image_path "%~dp0images" --import_path "%MODEL%"',
              "if errorlevel 1 (echo COLMAP returned an error. & pause)"]
    bat = work / "open_in_colmap.bat"
    bat.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    ini = [f"# RPP {__version__}: COLMAP GUI project. File > Open project (this file), then File > Import model > "
           f"{work / 'sparse' / models[0]}", f"database_path={(work / 'database.db').resolve()}",
           f"image_path={(work / 'images').resolve()}"]
    if (work / "masks").is_dir():
        ini += ["", "[ImageReader]", f"mask_path={(work / 'masks').resolve()}"]
    (work / "colmap_gui.ini").write_text("\n".join(ini) + "\n", encoding="utf-8")
    return bat


def report_text(man: Dict[str, Any], rep: Dict[str, Any]) -> str:
    b = man.get("block") or {}
    lines = [f"MSL first block - site {man.get('site')}, side {man.get('side')}, anchor {b.get('anchor')}"]
    for s in b.get("stations", []):
        if s["in_block"]:
            lines.append(f"  {s['station']}: {s['frames']} frames, {s['pairs']} pairs, sols {s['sols']}, {s['distance_m']} m")
    lines.append(f"frames {rep.get('frames')}, registered {rep.get('registered')}, points {rep.get('points')}, "
                 f"reprojection {rep.get('reprojection_px', float('nan')):.3f} px, models {rep.get('models')}")
    if rep.get("similarity_scale") is not None:
        r = rep["label_center_residual_m"]
        lines.append(f"site frame: scale {rep['similarity_scale']:.4f}, label centers off by median {r['median']:.3f} m, "
                     f"95 % {r['p95']:.3f} m, max {r['max']:.3f} m")
    if rep.get("stereo_baseline"):
        s = rep["stereo_baseline"]
        lines.append(f"stereo baseline: {s['pairs']} pairs, model {1000 * s['model_m_median']:.1f} mm vs label "
                     f"{1000 * s['label_m_median']:.1f} mm (ratio {s['ratio_median']:.4f} +- {s['ratio_sd']:.4f})")
    if man.get("unmasked"):
        lines.append(f"without a rover mask (no MXY): {len(man['unmasked'])} frames")
    if rep.get("message"):
        lines.append(rep["message"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- command line
def _sols(v: Sequence[int]) -> List[int]:
    v = [int(x) for x in v]
    return list(range(v[0], v[1] + 1)) if len(v) == 2 and v[1] > v[0] + 1 else v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("fetch", "prepare", "align", "all"))
    ap.add_argument("--data", default=DEFAULT_DATA, help=f"the MSL archive on disk (default {DEFAULT_DATA})")
    ap.add_argument("--work", default=None, help=f"the block's folder (default {DEFAULT_ROOT}\\{DEFAULT_NAME})")
    ap.add_argument("--sols", nargs="+", type=int, default=[3047, 3082], help="FIRST LAST, or a list (default 3047 3082)")
    ap.add_argument("--site", type=int, default=87)
    ap.add_argument("--block", default=DEFAULT_BLOCK, help=f"the block's name in its folder (default {DEFAULT_BLOCK})")
    ap.add_argument("--side", default="B", help="the computer side of the cameras (A or B; '' for both)")
    ap.add_argument("--drives", nargs="*", type=int, default=None, help="fetch: only these drives")
    ap.add_argument("--list", action="store_true", help="fetch: list what would come, download nothing")
    ap.add_argument("--min-pairs", type=int, default=8, help="prepare: a station needs this many stereo pairs")
    ap.add_argument("--radius-m", type=float, default=50.0, help="prepare: the stations within this of the anchor")
    ap.add_argument("--station", default=None, help="prepare: the anchor station (S087D0792), default the one with most pairs")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--refine-intrinsics", action="store_true", help="align: refine the focal length and distortion")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if a.work:
        work = Path(a.work)
    else:                                                  # 0.99.12: D:\rover_blocks\msl_sol_<first sol>_<block>
        from .block import block_folder_name, blocks_root
        work = blocks_root() / block_folder_name(min(_sols(a.sols)), a.block)
    side = a.side.upper() or None
    rc = 0
    if a.step in ("fetch", "all"):
        rep = fetch(Path(a.data), _sols(a.sols), a.site, side, a.drives, a.list)
        work.mkdir(parents=True, exist_ok=True)
        (work / f"fetch{'_list' if a.list else ''}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        found = sum(1 for v in rep["sols"].values() if v.get("found"))
        print(f"fetch: {found} of {len(rep['sols'])} sols found, {rep['downloaded']} files downloaded "
              f"({rep['bytes'] / 1e6:.0f} MB), {rep['kept']} already there, {len(rep['failed'])} failed")
        if not found:
            print(f"no sol folder answered - the addresses tried are in {work / ('fetch_list.json' if a.list else 'fetch.json')}")
            return 1
    if a.step in ("prepare", "all") and not a.list:
        st = None
        if a.station:
            m = re.fullmatch(r"S?(\d{1,3})D(\d{1,4})", a.station.upper())
            st = (int(m.group(1)), int(m.group(2))) if m else None
        man = prepare(work, Path(a.data), a.site, side, a.min_pairs, a.radius_m, st)
        print(f"prepare: {len(man['images'])} frames in {work}")
    if a.step in ("align", "all") and not a.list:
        rep = align(work, a.max_features, refine_intrinsics=a.refine_intrinsics)
        rc = 0 if rep.get("ok") else 1
        print(report_text(json.loads((work / "manifest.json").read_text(encoding="utf-8")), rep))
    return rc


if __name__ == "__main__":
    sys.exit(main())
