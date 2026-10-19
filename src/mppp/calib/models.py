"""
The promoted camera models: loading them from the models folder (:func:`mppp.paths.cmods_dir`), their fingerprint,
the Hazcam start models, a printable description, and the promotion of a new model (replace the file, append to
``CHANGES.md``; the git history keeps the replaced models).

Files (one folder, ``src/mppp/data/cmods`` unless ``MPPP_CMODS`` points elsewhere)::

    M2020_NL_fisheye_tangential.json   Navcam left camera (the rig reference eye)
    M2020_NR_fisheye_tangential.json   Navcam right camera
    M2020_N_rig.json                   Navcam stereo rig (NR from NL) with its drift in sol
    M2020_FL.json, M2020_FR.json       front Hazcam start models (not calibrated)
    M2020_RL.json, M2020_RR.json       rear Hazcam start models (not calibrated)
    M2020_ZCAM<zoom>_focus_model.json  Mastcam-Z calibration of one zoom (034; a zoom without a file starts
                                       from the labels)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ..paths import cmods_dir
from . import forms

log = logging.getLogger("mppp.calib")
PathLike = Union[str, Path]

NAVCAM_FILES = ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json")
CAMERA_FILE = {("N", "L"): "M2020_NL_fisheye_tangential.json", ("N", "R"): "M2020_NR_fisheye_tangential.json",
               ("F", "L"): "M2020_FL.json", ("F", "R"): "M2020_FR.json",
               ("R", "L"): "M2020_RL.json", ("R", "R"): "M2020_RR.json"}
RIG_FILE = {"N": "M2020_N_rig.json", "F": "M2020_F_rig.json", "R": "M2020_R_rig.json"}
ZCAM_FILE_PATTERN = "M2020_ZCAM{zoom:03d}_focus_model.json"
_ZCAM_RE = re.compile(r"M2020_ZCAM(\d{3})_focus_model.*\.json")
CHANGES = "CHANGES.md"
# the Hazcam start model: f = 2130 px, principal point at the detector centre, k1 = -0.25, k2 = +0.08
HAZCAM_START = {"f_px": 2130.0, "k1": -0.25, "k2": 0.08}


def sha12(f: PathLike) -> str:
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()[:12]


def _folder(folder: Optional[PathLike]) -> Path:
    return Path(folder) if folder else cmods_dir()


def _read(f: Path) -> Dict[str, Any]:
    return json.loads(f.read_text(encoding="utf-8"))


def camera_file(family: str, eye: str, folder: Optional[PathLike] = None) -> Path:
    """The camera file of a family (N, F, R) and eye (L, R) in ``folder`` (default: the models in use)."""
    key = (str(family)[:1].upper(), str(eye)[-1:].upper())
    if key not in CAMERA_FILE:
        raise ValueError(f"no camera file for family {family!r} eye {eye!r}")
    return _folder(folder) / CAMERA_FILE[key]


def hazcam_start_model(family: str, eye: str) -> Dict[str, Any]:
    """The start model of a Hazcam eye in the Navcam form: f = 2130 px (fx = fy), the principal point at the
    centre of the full frame, k1 = -0.25, k2 = +0.08, every other distortion term 0, no thermal or early-mission
    term.  Not calibrated: processing a Hazcam product starts from it with a warning."""
    fam = str(family)[:1].upper()
    if fam not in ("F", "R"):
        raise ValueError(f"{family!r} is not a Hazcam family (F or R)")
    w, h = forms.HAZCAM.frame
    f = HAZCAM_START["f_px"]
    params = [f, f, w / 2.0, h / 2.0, HAZCAM_START["k1"], HAZCAM_START["k2"], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return {"model": forms.HAZCAM.colmap_model, "width": w, "height": h, "params": params,
            "param_names": list(forms.HAZCAM.param_names), "family": fam, "eye": str(eye)[-1:].upper(),
            "form": "navcam-v1", "convention": f"{fam}L reference",
            "pixel_origin": "corner of the first pixel (COLMAP)",
            "distortion": "fisheye (theta polynomial k1-k4) + tangential p1, p2; sx1 = sy1 = 0",
            "source": "start model (not calibrated)",
            "thermal": {"T0_degC": -20.0, "fx_ppm_per_degC": 0.0, "fy_ppm_per_degC": 0.0, "ppm_per_degC": 0.0,
                        "cx_px_per_degC": 0.0, "cy_px_per_degC": 0.0, "k1_per_degC": 0.0,
                        "note": "no temperature term (start model)"},
            "fixed_params": [], "free_params": []}


def load_camera(family: str, eye: str, folder: Optional[PathLike] = None) -> Dict[str, Any]:
    """The promoted camera of a family (N, F, R) and eye (L, R): the file's dict with ``file`` and ``family`` added.
    A Hazcam eye without a file gets :func:`hazcam_start_model` (logged as a warning)."""
    fam = str(family)[:1].upper()
    f = camera_file(fam, eye, folder)
    if f.is_file():
        d = _read(f)
        d.setdefault("family", fam)
        d["file"] = str(f)
        return d
    if fam in ("F", "R"):
        log.warning("%s: no calibrated Hazcam model (%s); the start model (f = %g px, k1 = %g) is used",
                    f.name, f.parent, HAZCAM_START["f_px"], HAZCAM_START["k1"])
        d = hazcam_start_model(fam, eye)
        d["file"] = None
        return d
    raise FileNotFoundError(f"camera model {f} missing")


def rig_file(family: str, folder: Optional[PathLike] = None) -> Path:
    fam = str(family)[:1].upper()
    if fam not in RIG_FILE:
        raise ValueError(f"no stereo rig file for family {family!r}")
    return _folder(folder) / RIG_FILE[fam]


def load_rig(family: str = "N", folder: Optional[PathLike] = None) -> Optional[Dict[str, Any]]:
    """The promoted stereo rig of a family (``R_sensor_from_ref``, ``t_sensor_from_ref_m``, ``drift``); None when
    the family has no rig file (the Hazcams: the rig comes from the labels' CAHV pairs)."""
    f = rig_file(family, folder)
    if not f.is_file():
        return None
    d = _read(f)
    d["file"] = str(f)
    return d


def zcam_file(zoom: int, folder: Optional[PathLike] = None) -> Path:
    return _folder(folder) / ZCAM_FILE_PATTERN.format(zoom=int(zoom))


def zcam_files(folder: Optional[PathLike] = None) -> Dict[int, Path]:
    """{zoom mm: file} of the Mastcam-Z focus models: ``folder``'s files first, then the models in use for the
    zooms it does not hold (the first folder holding a zoom's file wins)."""
    from ..paths import package_cmods_dir
    dirs = ([Path(folder)] if folder else []) + [cmods_dir(), package_cmods_dir()]
    out: Dict[int, Path] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("M2020_ZCAM*_focus_model.json")):
            m = re.fullmatch(r"M2020_ZCAM(\d{3})_focus_model\.json", f.name)
            if m and int(m.group(1)) not in out:
                out[int(m.group(1))] = f
    return out


def load_zcam(zoom: int, folder: Optional[PathLike] = None) -> Optional[Dict[str, Any]]:
    """The promoted Mastcam-Z calibration of ``zoom`` mm (``cameras`` ZL<zoom>, ZR<zoom> with ``lines`` per state,
    ``pp``, ``pp0_px``, ``distortion``, ``regular_ratio``, ``aspect``; ``rig``); None for a zoom without a file
    (its cameras start from the labels)."""
    f = zcam_file(zoom, folder)
    if not f.is_file():
        return None
    d = _read(f)
    d["file"] = str(f)
    d["zoom_mm"] = int(zoom)
    return d


def load_zcam_all(folder: Optional[PathLike] = None, model_file: Optional[PathLike] = None) -> Dict[str, Any]:
    """Every zoom's focus model merged: ``cameras`` holds ZL034, ZR034, ZL048 ...; ``zooms[zoom]`` the rest of each
    file (``rig``, ``stereo``, ``state`` ...); ``sources[zoom]`` the file.  The 34 mm file's top-level entries are
    kept at the top level too.  ``model_file``: a file (e.g. a candidate) whose cameras replace those of its zoom."""
    files = zcam_files(folder)
    out: Dict[str, Any] = {"cameras": {}, "zooms": {}, "sources": {}}
    items = [(z, f) for z, f in sorted(files.items())] + ([(None, Path(model_file))] if model_file else [])
    for z, f in items:
        d = _read(Path(f))
        cams = d.get("cameras") or {}
        zooms = {_zoom_of_group(g) for g in cams} - {None} or ({z} if z else set())
        rest = {k: v for k, v in d.items() if k != "cameras"}
        for zz in zooms:
            out["zooms"][zz] = rest
            out["sources"][zz] = str(f)
            if model_file and z is None:
                out["cameras"] = {g: c for g, c in out["cameras"].items() if _zoom_of_group(g) != zz}
        out["cameras"].update(cams)
        if 34 in zooms:
            out.update({k: v for k, v in rest.items() if k not in ("zooms", "sources")})
    return out


def _zoom_of_group(g: str) -> Optional[int]:
    try:
        return int(str(g)[2:5]) if str(g).startswith("Z") else None
    except ValueError:
        return None


def models_fingerprint(folder: Optional[PathLike] = None) -> Optional[str]:
    """SHA-256 (16 hex) over the Navcam model files, the Hazcam files and the Mastcam-Z focus models of ``folder``
    (default: in use): part of the run key, so that a promotion makes a batch align every site again."""
    d = _folder(folder)
    h = hashlib.sha256()
    n = 0
    names = list(NAVCAM_FILES) + sorted(set(CAMERA_FILE.values()) - set(NAVCAM_FILES)) + sorted(RIG_FILE.values())
    files = [d / x for x in names] + sorted(d.glob("M2020_ZCAM*_focus_model.json"))
    for f in files:
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
            n += 1
    return h.hexdigest()[:16] if n else None


def zcam_fingerprint(folder: Optional[PathLike] = None, model_file: Optional[PathLike] = None) -> str:
    """SHA-256 of the Mastcam-Z focus-model files in use (every zoom's file, and ``model_file``)."""
    h = hashlib.sha256()
    files = [f for _, f in sorted(zcam_files(folder).items())] + ([Path(model_file)] if model_file else [])
    for f in files:
        h.update(Path(f).name.encode())
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def navcam_fingerprint(folder: Optional[PathLike] = None) -> Optional[str]:
    """SHA-256 over the Navcam camera and rig files of ``folder``."""
    d = _folder(folder)
    h = hashlib.sha256()
    n = 0
    for name in NAVCAM_FILES:
        f = d / name
        if f.is_file():
            h.update(name.encode())
            h.update(f.read_bytes())
            n += 1
    return h.hexdigest() if n else None


def check_form(folder: Optional[PathLike] = None) -> Dict[str, Any]:
    """Whether the models in ``folder`` (default: in use) follow the frozen forms: {"ok", "errors", "warnings",
    "folder", "files": {name: sha12}}.  ``errors`` would put wrong cameras into the alignments."""
    d = _folder(folder)
    errors: List[str] = []
    warnings: List[str] = []
    files = {n: sha12(d / n) for n in NAVCAM_FILES if (d / n).is_file()}
    missing = [n for n in NAVCAM_FILES if n not in files]
    if missing:
        errors.append(f"missing {', '.join(missing)} in {d}")
        return {"ok": False, "errors": errors, "warnings": warnings, "folder": str(d), "files": files}
    for eye in ("L", "R"):
        for p in forms.validate_camera(_read(d / CAMERA_FILE[("N", eye)]), forms.NAVCAM):
            errors.append(f"N{eye}: {p}")
    for p in forms.validate_rig(_read(d / RIG_FILE["N"])):
        errors.append(f"rig: {p}")
    for fam in ("F", "R"):
        for eye in ("L", "R"):
            f = d / CAMERA_FILE[(fam, eye)]
            if f.is_file():
                files[f.name] = sha12(f)
                for p in forms.validate_camera(_read(f), forms.HAZCAM):
                    errors.append(f"{fam}{eye}: {p}")
            else:
                warnings.append(f"{fam}{eye}: no file; the start model is used")
    for z, f in zcam_files(d).items():
        files[f.name] = sha12(f)
        for p in forms.validate_zcam(_read(f)):
            errors.append(f"ZCAM{z:03d}: {p}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "folder": str(d), "files": files}


def describe(folder: Optional[PathLike] = None) -> List[str]:
    """Printable lines: the models in ``folder`` (default: in use) - fingerprints, thermal terms, rig, zooms."""
    from .constrain import thermal_terms
    d = _folder(folder)
    out = [f"camera models in use: {d} (fingerprint {models_fingerprint(d)})"]
    for fam in ("N", "F", "R"):
        for eye in ("L", "R"):
            f = d / CAMERA_FILE[(fam, eye)]
            if not f.is_file():
                if fam == "N":
                    out.append(f"  {f.name}: MISSING")
                continue
            j = _read(f)
            p = j.get("params") or []
            t = thermal_terms(j.get("thermal"))
            em = j.get("early_mission") or {}
            out.append(f"  {f.name} {sha12(f)}: fx {p[0]:.2f} fy {p[1]:.2f} cx {p[2]:.2f} cy {p[3]:.2f} k4 {p[9]:+.5f}; "
                       f"thermal fx {t['fx_ppm']:+.1f} fy {t['fy_ppm']:+.1f} ppm/degC, cx {t['cx_px']:+.4f} cy "
                       f"{t['cy_px']:+.4f} px/degC, k1 {t['k1']:+.2e}/degC about {t['T0']:g} degC"
                       + (f"; early offsets before sol {float(em['before_sol']):g}" if em else "")
                       + f"; {j.get('source', '')[:60]}")
    for fam in ("N", "F", "R"):
        f = d / RIG_FILE[fam]
        if not f.is_file():
            continue
        j = _read(f)
        dr = j.get("drift") or {}
        out.append(f"  {f.name} {sha12(f)}: baseline {float(j.get('baseline_m', 0)):.5f} m, translation "
                   f"{j.get('translation', 'cahv')}, drift mdeg/1000 sol "
                   + ", ".join(f"{a} {1000 * float(dr.get(f'{a}_mdeg_per_sol') or 0):+.2f}" for a in ("pitch", "yaw", "roll")))
    for z, f in zcam_files(d).items():
        j = _read(f)
        cams = j.get("cameras") or {}
        out.append(f"  {f.name} {sha12(f)}: " + "; ".join(
            f"{g} f0 {float(c.get('f0_px', 0)):.1f} px, slope {float(c.get('slope_px_per_count', 0)):.4f} px/count, "
            f"regular ratio {float(c.get('regular_ratio', 1.0)):.4f}" for g, c in cams.items())
            + ("; rig" if j.get("rig") else ""))
    return out


# ------------------------------------------------------------------------------------------------- promotion
def _target_name(fname: str) -> Optional[str]:
    """The name a candidate file is promoted under (``M2020_ZCAM034_focus_model_candidate.json`` ->
    ``M2020_ZCAM034_focus_model.json``); None for a file that is not a model this package knows."""
    if fname in NAVCAM_FILES or fname in CAMERA_FILE.values() or fname in RIG_FILE.values():
        return fname
    m = _ZCAM_RE.fullmatch(fname)
    return f"M2020_ZCAM{m.group(1)}_focus_model.json" if m else None


def collect(sources: Sequence[PathLike]) -> Dict[str, Path]:
    """{target name: source file} from folders and files; raises for a file this package does not know."""
    out: Dict[str, Path] = {}
    for s in map(Path, sources):
        if not s.exists():
            raise ValueError(f"{s} does not exist")
        files = sorted(s.glob("*.json")) if s.is_dir() else [s]
        for f in files:
            t = _target_name(f.name)
            if t:
                out[t] = f
        if s.is_file() and _target_name(s.name) is None:
            raise ValueError(f"{s.name} is not a camera model file this package knows ({', '.join(NAVCAM_FILES)}, "
                             f"M2020_<F|R><L|R>.json, M2020_ZCAM<zoom>_focus_model*.json)")
    return out


def check_file(name: str, f: PathLike) -> str:
    """Raise if ``f`` does not load as what ``name`` says (and follow its form); return a one-line description."""
    d = _read(Path(f))
    if name in RIG_FILE.values():
        bad = forms.validate_rig(d)
        if bad:
            raise ValueError("; ".join(bad))
        return f"rig, baseline {d.get('baseline_m', '?')} m"
    if _ZCAM_RE.fullmatch(name):
        bad = forms.validate_zcam(d)
        if bad:
            raise ValueError("; ".join(bad))
        zoom = name[10:13]
        wrong = [g for g in d["cameras"] if str(g)[2:5] != zoom]
        if wrong:
            raise ValueError(f"cameras {wrong} are not of the {int(zoom)} mm zoom")
        return f"focus model for {sorted(d['cameras'])}" + (" and rig" if d.get("rig") else "")
    form = forms.HAZCAM if name[6] in ("F", "R") else forms.NAVCAM
    bad = forms.validate_camera(d, form)
    if bad:
        raise ValueError("; ".join(bad))
    th = d.get("thermal") or {}
    return f"{d['model']} {d['width']}x{d['height']}, fx {float(d['params'][0]):.2f} at {th.get('T0_degC')} degC"


def write_model(path: PathLike, model: Dict[str, Any]) -> Path:
    """Write a model dict as JSON (``file`` / ``zoom_mm`` bookkeeping keys dropped)."""
    p = Path(path)
    d = {k: v for k, v in model.items() if k not in ("file", "zoom_mm")}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")
    return p


def promote(sources: Sequence[PathLike], note: str = "", folder: Optional[PathLike] = None,
            dry_run: bool = False) -> Dict[str, Any]:
    """
    Make the model files in ``sources`` (folders or files) the models in use: each is checked against its form,
    then copied over the file in ``folder`` (default: the models in use) and a line is appended to ``CHANGES.md``
    with the source, the note and the SHA-256 of each file.  No history folder: the replaced files are in the git
    history (commit the promotion).  Returns {"folder", "files": {name: sha12}, "stamp"}; ``dry_run`` checks only.
    """
    dst = _folder(folder)
    todo = collect(sources)
    if not todo:
        raise ValueError(f"no camera model files found in {list(map(str, sources))}")
    desc = {name: check_file(name, f) for name, f in todo.items()}
    if dry_run:
        return {"folder": str(dst), "files": {n: sha12(f) for n, f in todo.items()}, "checked": desc, "stamp": None}
    dst.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    lines, shas = [], {}
    for name, f in todo.items():
        shutil.copy2(f, dst / name)
        shas[name] = sha12(dst / name)
        lines.append(f"  - `{name}` {shas[name]} from `{f}` ({desc[name]})")
    with (dst / CHANGES).open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {stamp}" + (f" - {note}" if note else "") + "\n" + "\n".join(lines) + "\n")
    log.info("%d model file(s) promoted into %s", len(todo), dst)
    return {"folder": str(dst), "files": shas, "checked": desc, "stamp": stamp}
