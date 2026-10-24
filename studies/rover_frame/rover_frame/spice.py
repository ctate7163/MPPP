"""
The rover kernels of study 3A in SPICE form, written with spiceypy (0.94.5, Christian 24 Oct 2026):

- ``m2020_surf_rover_mppp_<block>_v01.bc`` (CK type 3): ``M2020_ROVER`` (-168000) relative to ``M2020_LOCAL_LEVEL``
  (-168910), as NAIF's telemetry CKs.  One segment per interval of :func:`rover_frame.estimate.intervals`, the
  attitude constant within it (two equal records, one interpolation interval), so nothing is interpolated across a
  rover pose change; the segment ID names the source (``F+R``, ``front``, ``rear``, ``propagated``, ``telemetry``).
- ``m2020_surf_rover_mppp_<block>_v01.bsp`` (SPK type 9, degree 1): the rover (-168) relative to the landing site
  (-168900) in ``M2020_LOCAL_LEVEL``, constant per interval.  Needs the NAIF clock and leapseconds kernels (ET).
- ``m2020_mppp_hazcams_v01.tf`` (FK): TK frames of the four Hazcams relative to ``M2020_ROVER`` (NAIF's
  ``m2020_v04.tf`` has none); IDs -168401 ... -168404 (an unused range of the M2020 FK; to be confirmed with NAIF).
  The camera frames are COLMAP's (+X right / sample, +Y down / line, +Z boresight).
- ``rover_intervals.csv`` and ``rover_kernels.json``: the sigma, source and rover motion counters of every interval
  (a CK holds no covariance), the global terms, the read-back check; the same summary in each kernel's comment area.
- ``mppp_rover_<date>.tm``: a meta-kernel with the NAIF base kernels and these.

Times: the label's ``SPACECRAFT_CLOCK_START_COUNT`` (seconds with a decimal millisecond fraction, e.g.
``729889968.787``) is the encoded SCLK ``round(count * 65536)`` of NAIF's M2020 clock (two fields, moduli
4294967296 and 65536) - the CK needs no clock kernel; the SPK's ET comes from ``scs2e`` with NAIF's
``m2020_168_sclkscet`` kernel.

Frame caveat: the block frame is the landing site's ENU of MPPP's positions (site frames taken as north-aligned) and
its heading is set by the position priors; north at a later site is rotated by about 5-6 mdeg per km of east-west
travel, so the kernels are right to that level within a block, not mission-wide.

The NAIF base kernels are fetched once into ``<cache>/naif`` (``MPPP_NAIF`` overrides); without them the frame
definitions this writer needs come from a stub with NAIF's IDs and angles and the SPK is left out.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROVER_ID = -168
ROVER_FRAME_ID = -168000
LANDING_SITE_ID = -168900
REF = "M2020_LOCAL_LEVEL"
TICKS = 65536.0
P_NED_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
HAZCAM_IDS = {"FL": -168401, "FR": -168402, "RL": -168403, "RR": -168404}
HAZCAM_NAMES = {"FL": "MPPP_M2020_FHAZ_LEFT", "FR": "MPPP_M2020_FHAZ_RIGHT", "RL": "MPPP_M2020_RHAZ_LEFT",
                "RR": "MPPP_M2020_RHAZ_RIGHT"}
NAIF = "https://naif.jpl.nasa.gov/pub/naif"
BASE = {"lsk": (f"{NAIF}/generic_kernels/lsk/", r"naif(\d{4})\.tls"),
        "pck": (f"{NAIF}/generic_kernels/pck/", r"pck(\d{5})\.tpc"),
        "fk": (f"{NAIF}/MARS2020/kernels/fk/", r"m2020_v(\d{2})\.tf"),
        "sclk": (f"{NAIF}/MARS2020/kernels/sclk/", r"m2020_168_sclkscet_(?:refit_v)?(\d{5})\.tsc")}

# NAIF m2020_v04.tf: M2020_TOPO (-168900 frame, IAU_MARS, 3-2-3 angles), M2020_LOCAL_LEVEL (-168910: +X north,
# +Y east, +Z down), M2020_ROVER (-168000, CK frame) - only for writing when the real FK is not at hand
STUB_FK = r"""KPL/FK
MPPP stub of three frames of NAIF's m2020_v04.tf (IDs and angles as there), used only when that kernel is not
available; load NAIF's m2020_v04.tf for any other use.
\begindata
FRAME_M2020_TOPO               = -168900
FRAME_-168900_NAME             = 'M2020_TOPO'
FRAME_-168900_CLASS            = 4
FRAME_-168900_CLASS_ID         = -168900
FRAME_-168900_CENTER           = -168900
TKFRAME_-168900_RELATIVE       = 'IAU_MARS'
TKFRAME_-168900_SPEC           = 'ANGLES'
TKFRAME_-168900_UNITS          = 'DEGREES'
TKFRAME_-168900_AXES           = ( 3, 2, 3 )
TKFRAME_-168900_ANGLES         = ( -77.45088572, -71.35124184, 180.000 )
FRAME_M2020_LOCAL_LEVEL        = -168910
FRAME_-168910_NAME             = 'M2020_LOCAL_LEVEL'
FRAME_-168910_CLASS            = 4
FRAME_-168910_CLASS_ID         = -168910
FRAME_-168910_CENTER           = -168900
TKFRAME_-168910_RELATIVE       = 'M2020_TOPO'
TKFRAME_-168910_SPEC           = 'ANGLES'
TKFRAME_-168910_UNITS          = 'DEGREES'
TKFRAME_-168910_AXES           = ( 1, 2, 3 )
TKFRAME_-168910_ANGLES         = ( 180.000, 0.000, 0.000 )
FRAME_M2020_ROVER              = -168000
FRAME_-168000_NAME             = 'M2020_ROVER'
FRAME_-168000_CLASS            = 3
FRAME_-168000_CLASS_ID         = -168000
FRAME_-168000_CENTER           = -168
CK_-168000_SCLK                = -168
CK_-168000_SPK                 = -168
\begintext
"""


def spiceypy():
    try:
        import spiceypy as sp
    except ImportError as e:                             # pragma: no cover - reported by write()
        raise RuntimeError("spiceypy is not installed (pip install spiceypy)") from e
    return sp


def encoded_sclk(count: float) -> float:
    """The encoded SCLK (ticks of 1/65536 s) of a label's ``SPACECRAFT_CLOCK_START_COUNT`` (decimal seconds)."""
    return float(round(float(count) * TICKS))


# --- the NAIF base kernels -------------------------------------------------------------------------------------------

def naif_dir() -> Path:
    env = os.environ.get("MPPP_NAIF")
    if env:
        return Path(env)
    try:
        from mppp.paths import cache_dir
        return cache_dir() / "naif"
    except Exception:                                    # noqa: BLE001 - outside the package
        return Path.home() / ".cache" / "mppp" / "naif"


def _get(url: str, timeout: float = 60.0) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "MPPP rover kernels"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def newest(listing: str, pattern: str) -> Optional[str]:
    """The highest-numbered file of a directory listing matching ``pattern`` (one number group; a refit kernel
    before the plain one of the same number)."""
    names = set(re.findall(r'href="([^"/?]+)"', listing))
    best = None
    for n in names:
        m = re.fullmatch(pattern, n, re.I)
        if m:
            k = (int(m.group(1)), "refit" in n.lower())
            if best is None or k > best[0]:
                best = (k, n)
    return best[1] if best else None


def naif_kernels(folder: Optional[Path] = None, fetch: bool = True, get=_get, log=print) -> Dict[str, Optional[Path]]:
    """{lsk, pck, fk, sclk: the newest kernel of each in ``folder`` (fetched from NAIF when missing and ``fetch``)}."""
    folder = Path(folder) if folder else naif_dir()
    folder.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Optional[Path]] = {}
    for kind, (url, pat) in BASE.items():
        have = sorted([p for p in folder.iterdir() if re.fullmatch(pat, p.name, re.I)],
                      key=lambda p: (int(re.fullmatch(pat, p.name, re.I).group(1)), "refit" in p.name.lower()))
        name = None
        if fetch:
            try:
                name = newest(get(url).decode("utf-8", "replace"), pat)
            except Exception as e:                       # noqa: BLE001 - offline: what is there
                log(f"  NAIF {kind}: listing not read ({type(e).__name__}: {e})")
        if name and not (folder / name).is_file():
            try:
                tmp = folder / (name + ".part")
                tmp.write_bytes(get(url + name))
                tmp.replace(folder / name)
                log(f"  NAIF {kind}: {name} fetched")
            except Exception as e:                       # noqa: BLE001
                log(f"  NAIF {kind}: {name} not fetched ({type(e).__name__}: {e})")
                name = None
        out[kind] = folder / name if name and (folder / name).is_file() else (have[-1] if have else None)
    return out


# --- writers ---------------------------------------------------------------------------------------------------------

def _comments(lines: Sequence[str], width: int = 78) -> List[str]:
    import textwrap
    out = []
    for ln in lines:
        out += textwrap.wrap(ln, width) or [""]
    return out


def hazcam_fk(mounts: Dict[str, Dict[str, Any]], path: Path, summary: Sequence[str] = ()) -> Path:
    """The TK frames of the Hazcams relative to M2020_ROVER (``TKFRAME_MATRIX`` = R_c_rover row by row: SPICE's
    ``pxform('M2020_ROVER', frame)``)."""
    L = ["KPL/FK", "", "MPPP Hazcam frames relative to M2020_ROVER (study 3A, rover_frame; "
         f"{datetime.date.today():%Y-%m-%d}).", "Camera axes: +X right (sample), +Y down (line), +Z boresight.",
         "IDs -168401..-168404 are in a range NAIF's m2020_v04.tf leaves unused; to be confirmed with NAIF.",
         "Load after m2020_v04.tf.", *summary, "", "\\begindata", ""]
    for grp in ("FL", "FR", "RL", "RR"):
        if grp not in mounts:
            continue
        i, n = HAZCAM_IDS[grp], HAZCAM_NAMES[grp]
        M = np.asarray(mounts[grp]["R_c_rover"], float)
        L += [f"FRAME_{n:<28}= {i}", f"FRAME_{i}_NAME               = '{n}'", f"FRAME_{i}_CLASS              = 4",
              f"FRAME_{i}_CLASS_ID           = {i}", f"FRAME_{i}_CENTER             = {ROVER_ID}",
              f"TKFRAME_{i}_RELATIVE         = 'M2020_ROVER'", f"TKFRAME_{i}_SPEC             = 'MATRIX'",
              f"TKFRAME_{i}_MATRIX           = ( " + "\n                                     ".join(
                  " ".join(f"{v: .15e}" for v in row) for row in M) + " )", ""]
    L += ["\\begintext", ""]
    for grp in ("FL", "FR", "RL", "RR"):
        if grp in mounts:
            c = mounts[grp]["C_rover_m"]
            L.append(f"{HAZCAM_NAMES[grp]} centre in M2020_ROVER (m): {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def c_matrix(iv: Dict[str, Any]) -> np.ndarray:
    """The C-matrix of an interval: M2020_LOCAL_LEVEL (north-east-down) -> M2020_ROVER."""
    return np.asarray(iv["R_rover_w"], float) @ P_NED_ENU


def position_ll(iv: Dict[str, Any]) -> np.ndarray:
    """The rover origin relative to the landing site in M2020_LOCAL_LEVEL (m -> km for the SPK)."""
    enu = np.asarray(iv["O_w"], float) + np.asarray(iv.get("offset") or [0.0, 0.0, 0.0], float)
    return P_NED_ENU @ enu


def kstr(text: str, width: int = 70) -> List[str]:
    """A text-kernel string value in pieces of at most ``width`` characters, continued with '+' (SPICE's limit
    is 80 characters per string)."""
    t = str(text).replace("\\", "/")
    parts = [t[i:i + width] for i in range(0, len(t), width)] or [""]
    return [f"'{x}+'" for x in parts[:-1]] + [f"'{parts[-1]}'"]


def _segid(block: str, src: str) -> str:
    return f"MPPP {src} {block}"[:40]


def write_ck(path: Path, ivs: Sequence[Dict[str, Any]], comments: Sequence[str] = ()) -> Path:
    sp = spiceypy()
    if path.exists():
        path.unlink()
    h = sp.ckopn(str(path), "MPPP rover CK", 5000)
    try:
        for iv in ivs:
            a, b = encoded_sclk(iv["sclk_start"]), encoded_sclk(iv["sclk_end"])
            if b <= a:
                b = a + 1.0
            q = sp.m2q(c_matrix(iv))
            sp.ckw03(h, a, b, ROVER_FRAME_ID, REF, False, _segid(iv["block"], iv["source"]), 2,
                     [a, b], [list(q), list(q)], [[0.0, 0.0, 0.0]] * 2, 1, [a])
        if comments:
            sp.dafac(h, _comments(comments))
    finally:
        sp.ckcls(h)
    return path


def write_spk(path: Path, ivs: Sequence[Dict[str, Any]], comments: Sequence[str] = ()) -> Path:
    """Needs the NAIF clock and leapseconds kernels loaded (``scs2e``)."""
    sp = spiceypy()
    if path.exists():
        path.unlink()
    h = sp.spkopn(str(path), "MPPP rover SPK", 5000)
    try:
        for iv in ivs:
            e0 = sp.sct2e(ROVER_ID, encoded_sclk(iv["sclk_start"]))
            e1 = sp.sct2e(ROVER_ID, encoded_sclk(iv["sclk_end"]))
            if e1 <= e0:
                e1 = e0 + 1e-3
            p = (position_ll(iv) / 1000.0).tolist()
            st = p + [0.0, 0.0, 0.0]
            sp.spkw09(h, ROVER_ID, LANDING_SITE_ID, REF, e0, e1, _segid(iv["block"], iv["source"]), 1, 2, [st, st],
                      [e0, e1])
        if comments:
            sp.dafac(h, _comments(comments))
    finally:
        sp.spkcls(h)
    return path


def readback(ck: Path, ivs: Sequence[Dict[str, Any]]) -> float:
    """The largest angle (rad) between the written and the read-back attitude at every interval's middle."""
    sp = spiceypy()
    sp.furnsh(str(ck))
    worst = 0.0
    try:
        for iv in ivs:
            t = (encoded_sclk(iv["sclk_start"]) + encoded_sclk(iv["sclk_end"])) / 2.0
            try:
                got = sp.ckgp(ROVER_FRAME_ID, t, 0.0, REF)      # spiceypy raises NotFoundError when not found
            except Exception:                                   # noqa: BLE001
                return float("inf")
            C = got[0]
            if len(got) > 2 and not got[2]:
                return float("inf")
            d = np.asarray(C) @ c_matrix(iv).T
            # the angle from the skew part (arccos of the trace loses half the digits near zero)
            worst = max(worst, float(np.arctan2(np.linalg.norm([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]),
                                                np.trace(d) - 1.0)))
    finally:
        sp.unload(str(ck))
    return worst


INTERVAL_FIELDS = ("block", "station", "SITE", "DRIVE", "POSE", "ARM", "sol", "source", "sclk_start", "sclk_end",
                   "measured_sclk", "sigma_mdeg", "tel_residual_x_mdeg", "tel_residual_y_mdeg", "tel_residual_z_mdeg",
                   "images", "hazcam_frames", "groups")


def write(intervals: Sequence[Dict[str, Any]], gt: Dict[str, Any], mounts: Dict[str, Dict[str, Any]], out: Path,
          naif: Optional[Dict[str, Optional[Path]]] = None, version: int = 1, log=print) -> Dict[str, Any]:
    """Every kernel, the sidecar and the meta-kernel into ``out``; returns the summary (also ``rover_kernels.json``)."""
    sp = spiceypy()
    out.mkdir(parents=True, exist_ok=True)
    naif = naif or {}
    sp.kclear()
    loaded = []
    if naif.get("fk"):
        sp.furnsh(str(naif["fk"]))
        loaded.append(Path(naif["fk"]).name)
    else:
        stub = out / "m2020_frames_stub_mppp.tf"
        stub.write_text(STUB_FK, encoding="utf-8")
        sp.furnsh(str(stub))
        loaded.append(stub.name + " (stub: NAIF's m2020_v04.tf not at hand)")
    for k in ("lsk", "sclk"):
        if naif.get(k):
            sp.furnsh(str(naif[k]))
            loaded.append(Path(naif[k]).name)
    summary = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "kernels_loaded": loaded,
               "global": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gt.items()},
               "hazcams": {g: {"frame": HAZCAM_NAMES[g], "id": HAZCAM_IDS[g],
                               "R_c_rover": np.asarray(m["R_c_rover"]).tolist(), "C_rover_m": list(map(float, m["C_rover_m"])),
                               "rig_images": m["rig_images"]} for g, m in mounts.items() if g in HAZCAM_IDS},
               "blocks": {}}
    g = summary["global"]
    head = ["MPPP rover kernel (study 3A, rover_frame), " + summary["created"] + ".",
            "Attitude constant per interval (rover pose; a later Hazcam measurement starts a new one). Sources: F+R = "
            "end of drive, front and rear Hazcams; front; rear; propagated = the station's measured body error carried "
            "over by the telemetry change; telemetry.",
            "M2020_ROVER = the IMU frame on average (zero mean tilt residual of the end-of-drive bodies); "
            f"alignment applied (x, y) mdeg: {g['alignment_mdeg'][0]:.1f}, {g['alignment_mdeg'][1]:.1f} over "
            f"{g['n_alignment']} stations.",
            f"Rear-to-front Hazcam correction (mdeg): {', '.join(f'{v:.1f}' for v in g['Q_mdeg'])} over {g['n_eod']} "
            "end-of-drive stations.",
            f"Telemetry scatter after alignment: tilt {g['sigma_tilt_mdeg']} mdeg, heading {g['sigma_heading_mdeg']} mdeg "
            "(the sigma of propagated and telemetry intervals).",
            "Per-interval sigma, source and rover motion counters: rover_intervals.csv / rover_kernels.json "
            "(a CK holds no covariance).",
            "Frame caveat: block frame = landing-site ENU of MPPP's positions (site frames taken as north-aligned).",
            "Times: encoded SCLK = round(SPACECRAFT_CLOCK_START_COUNT * 65536)."]
    files = []
    blocks = sorted({iv["block"] for iv in intervals})
    for b in blocks:
        ivs = sorted([iv for iv in intervals if iv["block"] == b], key=lambda iv: iv["sclk_start"])
        src = {s: sum(iv["source"] == s for iv in ivs) for s in sorted({iv["source"] for iv in ivs})}
        com = head + [f"Block {b}: {len(ivs)} intervals ({', '.join(f'{k} {v}' for k, v in src.items())})."]
        ck = write_ck(out / f"m2020_surf_rover_mppp_{b}_v{version:02d}.bc", ivs, com)
        err = readback(ck, ivs)
        files.append(ck.name)
        info = {"ck": ck.name, "intervals": len(ivs), "sources": src, "readback_max_rad": err}
        if naif.get("lsk") and naif.get("sclk"):
            spk = write_spk(out / f"m2020_surf_rover_mppp_{b}_v{version:02d}.bsp", ivs, com)
            files.append(spk.name)
            info["spk"] = spk.name
        else:
            info["spk"] = None
            info["spk_note"] = "left out: the NAIF clock and leapseconds kernels are not at hand"
        summary["blocks"][b] = info
        log(f"  {b}: {len(ivs)} intervals {src}; CK read back within {err:.1e} rad"
            + ("" if info["spk"] else "; no SPK (no NAIF clock kernel)"))
    if mounts:
        fk = hazcam_fk(mounts, out / f"m2020_mppp_hazcams_v{version:02d}.tf",
                       [f"Rear-to-front correction and rover frame alignment as in rover_kernels.json."])
        files.append(fk.name)
        summary["fk"] = fk.name
    sp.kclear()
    with open(out / "rover_intervals.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(INTERVAL_FIELDS) + ["quat_w", "quat_x", "quat_y", "quat_z",
                                                                   "north_m", "east_m", "down_m"], extrasaction="ignore")
        w.writeheader()
        for iv in sorted(intervals, key=lambda iv: (iv["block"], iv["sclk_start"])):
            q = sp.m2q(c_matrix(iv))
            p = position_ll(iv)
            w.writerow(dict(iv, quat_w=q[0], quat_x=q[1], quat_y=q[2], quat_z=q[3],
                            north_m=p[0], east_m=p[1], down_m=p[2]))
    # the meta-kernel: NAIF's base kernels (where they are) and these
    tm = out / f"mppp_rover_{datetime.date.today():%Y%m%d}.tm"
    base = [Path(naif[k]) for k in ("lsk", "pck", "sclk", "fk") if naif.get(k)]
    ndir = str(base[0].parent if base else naif_dir()).replace("\\", "/")
    L = ["KPL/MK", "", "MPPP rover kernels (study 3A). The NAIF kernels: " + (", ".join(x.name for x in base)
                                                                          or "not at hand - add naif0012.tls, pck00010.tpc, "
                                                                             "m2020_168_sclkscet_*.tsc and m2020_v04.tf"),
         "", "\\begindata", "", "PATH_VALUES  = ("]
    L += [f"    {x}" for x in kstr(str(out))] + [f"    {x}" for x in kstr(ndir)] + ["    )"]
    L += ["PATH_SYMBOLS = ( 'MPPP', 'NAIF' )", "KERNELS_TO_LOAD = ("]
    L += [f"    {y}" for x in base for y in kstr(f"$NAIF/{x.name}")]
    if not naif.get("fk"):
        L.append("    '$MPPP/m2020_frames_stub_mppp.tf'")
    L += [f"    '$MPPP/{f}'" for f in files if f.endswith(".tf")]
    L += [f"    '$MPPP/{f}'" for f in files if not f.endswith(".tf")]
    L += ["    )", "", "\\begintext", ""]
    tm.write_text("\n".join(L), encoding="utf-8")
    summary["meta_kernel"] = tm.name
    summary["files"] = files
    (out / "rover_kernels.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary
