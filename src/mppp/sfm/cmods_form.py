"""
The frozen camera-model form (MPPP v0p73) and the checks that the models in use follow it.

Navcam (``M2020_NL/NR_fisheye_tangential.json``, ``M2020_N_rig.json``), "NL reference" convention (Christian, 3 Oct
2026): NL keeps the label CAHVORE's pointing and position (a constant offset from it); everything that moves goes on NR.

- fisheye + tangential, **k4 = 0** in both eyes (notebook 03 holds the distortion and sets k4 = 0, so a model fitted
  with k4 free would leave its k3 wrong at the corners: the Zcam34 runs of 2-3 Oct);
- **fx and fy against temperature, each eye on its own** (``thermal.fx_ppm_per_degC``, ``fy_ppm_per_degC`` about
  ``T0_degC``; ``ppm_per_degC`` the mean of the two, for older readers);
- **NR cx against temperature** (``thermal.cx_px_per_degC`` of NR) - the stereo yaw change with temperature; NL's
  principal point does not move with temperature (``cx_px_per_degC = cy_px_per_degC = 0``);
- the rig: one yaw for every temperature (``yaw_constant``), pitch, roll **and yaw** with sol
  (``drift.<angle>_mdeg_per_sol``; the yaw rate is measured, 0 until it is), translation of NR from NL.

Mastcam-Z: the focus model per zoom (``M2020_ZCAM<zoom>_focus_model.json``); in a block every focus parameter is a
line in focus count per eye and focus (backlash) state (:mod:`mppp.sfm.zcam`), ZL keeping its principal point (the
Mastcam-Z reference eye, like NL).

:func:`check_form` says whether a folder follows the form; :func:`convert_to_nl_reference` turns a v0p60-v0p72
consensus (the stereo temperature change on NL's cx) into it; :func:`describe` prints what is in use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PathLike = Union[str, Path]

FORM_VERSION = "v0p73"
FORMS = ("v0p73", "v0p80")          # v0p80: as v0p73, but k4 fitted with k3 in the joint (not 0) and k1 against temperature
CONVENTION = "NL reference"
NAVCAM_FILES = ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json")
# the candidate of the v0p72 consensus run of 2 Oct 2026 (16 blocks) - promoted in v0p73 in the NL-reference form
V0P72_CANDIDATE = r"D:\scapes\colmap\camera_analysis\navcal_v0p72_20261002_2\navcam_joint"


def thermal_terms(th: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """{"fx_ppm", "fy_ppm", "cx_px", "cy_px", "T0"} of a camera file's ``thermal`` entry (v0p73: fx and fy each their
    own ppm/degC, else the shared ``ppm_per_degC``); zeros without one."""
    th = th or {}
    ppm = float(th.get("ppm_per_degC") or 0.0)
    return {"fx_ppm": float(th.get("fx_ppm_per_degC", ppm) or 0.0), "fy_ppm": float(th.get("fy_ppm_per_degC", ppm) or 0.0),
            "cx_px": float(th.get("cx_px_per_degC") or 0.0), "cy_px": float(th.get("cy_px_per_degC") or 0.0),
            "k1": float(th.get("k1_per_degC") or 0.0),                       # v0p80
            "T0": float(th.get("T0_degC", -20.0))}


def thermal_scale(th: Optional[Dict[str, Any]], T: float) -> Tuple[float, float]:
    """(sx, sy): fx and fy at temperature ``T`` over their values at T0 (1 + ppm 1e-6 (T - T0) per axis)."""
    t = thermal_terms(th)
    d = float(T) - t["T0"]
    return 1.0 + 1e-6 * t["fx_ppm"] * d, 1.0 + 1e-6 * t["fy_ppm"] * d


def sha12(f: PathLike) -> str:
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()[:12]


def _load(d: Path, name: str) -> Optional[Dict[str, Any]]:
    f = d / name
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def check_form(folder: Optional[PathLike] = None, require_convention: bool = True) -> Dict[str, Any]:
    """
    Whether the Navcam models in ``folder`` (default: the ones in use, :func:`mppp.paths.cmods_dir`) follow the v0p73
    form (module docstring).  v0p80: a file with ``"form": "v0p80"`` may have k4 != 0 (k3 and k4 fitted together in the
    joint; notebook 03 then holds both, NAVCAM_K4 = "consensus"), and its ``thermal.k1_per_degC`` moves k1 with
    temperature; ``"form"`` of the result is the newest form both eyes declare.  Returns {"ok", "errors": [...], "warnings": [...], "folder", "files": {name: sha12}}.
    ``errors`` would put wrong cameras into the alignments (k4 not 0, NL principal point moving with temperature,
    a rig temperature term); ``require_convention`` also makes a missing ``"convention": "NL reference"`` an error.
    """
    from ..paths import cmods_dir
    d = Path(folder) if folder else cmods_dir()
    errors: List[str] = []
    warnings: List[str] = []
    files = {n: sha12(d / n) for n in NAVCAM_FILES if (d / n).is_file()}
    missing = [n for n in NAVCAM_FILES if n not in files]
    if missing:
        errors.append(f"missing {', '.join(missing)} in {d}")
        return {"ok": False, "errors": errors, "warnings": warnings, "folder": str(d), "files": files}
    nl, nr, rig = (_load(d, n) for n in NAVCAM_FILES)
    forms = {str(j.get("form") or "") for j in (nl, nr)}
    form = "v0p80" if forms == {"v0p80"} else "v0p73"
    if len(forms) > 1:
        errors.append(f"the two eyes are in different forms {sorted(forms)}")
    for eye, j in (("NL", nl), ("NR", nr)):
        names = j.get("param_names") or []
        p = j.get("params") or []
        k4 = float(p[names.index("k4")]) if "k4" in names and len(p) > names.index("k4") else None
        if k4 is None:
            errors.append(f"{eye}: no k4 parameter ({j.get('model')})")
        elif abs(k4) > 1e-12 and form != "v0p80":
            errors.append(f"{eye}: k4 = {k4:+.5f}, the form holds k4 = 0 (notebook 03 sets k4 = 0 and would keep a k3 "
                          f"fitted with k4 free: ~3-4 px at the corners)")
        if not j.get("thermal"):
            errors.append(f"{eye}: no thermal model")
        else:
            th = j["thermal"]
            if "fx_ppm_per_degC" not in th or "fy_ppm_per_degC" not in th:
                warnings.append(f"{eye}: one ppm/degC for fx and fy ({th.get('ppm_per_degC')}); the form fits each axis")
        if require_convention and j.get("convention") != CONVENTION:
            errors.append(f"{eye}: convention {j.get('convention')!r}, expected {CONVENTION!r}")
    t = thermal_terms(nl.get("thermal"))
    if abs(t["cx_px"]) > 1e-12 or abs(t["cy_px"]) > 1e-12:
        errors.append(f"NL principal point moves with temperature (cx {t['cx_px']:+.4f}, cy {t['cy_px']:+.4f} px/degC): "
                      f"in the NL-reference form that term is NR's (convert_to_nl_reference)")
    if abs(thermal_terms(nr.get("thermal"))["cx_px"]) == 0.0 and abs(t["cx_px"]) == 0.0:
        warnings.append("no stereo temperature term (NR cx px/degC = 0)")
    th_rig = rig.get("thermal") or {}
    if th_rig.get("yaw_mdeg_per_degC") or th_rig.get("pitch_mdeg_per_degC"):
        errors.append("the rig has a temperature term: the form carries the stereo temperature change on NR cx")
    if not rig.get("yaw_constant"):
        warnings.append("rig: yaw_constant not set")
    if require_convention and rig.get("convention") != CONVENTION:
        errors.append(f"rig: convention {rig.get('convention')!r}, expected {CONVENTION!r}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "folder": str(d), "files": files, "form": form}


def describe(folder: Optional[PathLike] = None) -> List[str]:
    """Printable lines: the Navcam models in ``folder`` (default: in use) - fingerprints, source, thermal terms, rig."""
    from ..paths import cmods_dir
    d = Path(folder) if folder else cmods_dir()
    out = [f"camera models in use: {d}"]
    for n in NAVCAM_FILES:
        j = _load(d, n)
        if j is None:
            out.append(f"  {n}: MISSING")
            continue
        if n == "M2020_N_rig.json":
            dr = j.get("drift") or {}
            out.append(f"  {n} {sha12(d / n)}: baseline {float(j.get('baseline_m', 0)):.5f} m, translation "
                       f"{j.get('translation', 'cahv')}, yaw constant {bool(j.get('yaw_constant'))}, drift mdeg/1000 sol "
                       + ", ".join(f"{a} {1000 * float(dr.get(f'{a}_mdeg_per_sol') or 0):+.2f}" for a in ("pitch", "yaw", "roll"))
                       + f"; {j.get('convention') or 'no convention'}")
            continue
        names = j.get("param_names") or []
        p = j.get("params") or []
        k4 = p[names.index("k4")] if "k4" in names else None
        t = thermal_terms(j.get("thermal"))
        out.append(f"  {n} {sha12(d / n)}: fx {p[0]:.2f} fy {p[1]:.2f} cx {p[2]:.2f} cy {p[3]:.2f} k4 {k4}; thermal fx "
                   f"{t['fx_ppm']:+.1f} fy {t['fy_ppm']:+.1f} ppm/degC, cx {t['cx_px']:+.4f} cy {t['cy_px']:+.4f} px/degC"
                   + (f", k1 {t['k1']:+.2e}/degC" if t["k1"] else "")
                   + f" about {t['T0']:g} degC; {j.get('form') or ''} {j.get('convention') or 'no convention'}")
    return out


def convert_to_nl_reference(src: PathLike, dst: PathLike, note: str = "") -> Dict[str, Any]:
    """
    Write the Navcam models of ``src`` (a consensus folder) into ``dst`` in the NL-reference form:

    - NL's principal-point temperature slopes move to NR: NR cx/cy slope -= NL slope x fx_NR / fx_NL (the stereo
      geometry, i.e. the difference of the two eyes' pointing, stays the same; the whole-head pointing that NL
      carried is absorbed by the frame poses), NL's become 0;
    - fx and fy get their own ppm/degC (both the shared value until a v0p73 consensus fits them);
    - ``"convention": "NL reference"`` and ``"form": "v0p73"`` in all three files; the rig keeps one yaw for every
      temperature (``yaw_constant``) and its drift, the yaw rate as it is (0 in v0p60-v0p72).

    The early-mission offsets (common to both eyes) are kept as they are.  Returns {"files": {name: sha12}, "moved"}.
    """
    import datetime
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    nl, nr, rig = (_load(s, n) for n in NAVCAM_FILES)
    if nl is None or nr is None or rig is None:
        raise FileNotFoundError(f"{s} needs {', '.join(NAVCAM_FILES)}")
    tl = dict(nl.get("thermal") or {})
    tr = dict(nr.get("thermal") or {})
    ratio = float(nr["params"][0]) / float(nl["params"][0])
    moved = {}
    for k in ("cx_px_per_degC", "cy_px_per_degC"):
        a = float(tl.get(k) or 0.0)
        if a:
            tr[k] = float(tr.get(k) or 0.0) - a * ratio
            moved[k] = {"NL_was": a, "NR_now": tr[k]}
        tl[k] = 0.0
    sd_cx = tl.pop("sd_cx_px_per_degC", None)
    if sd_cx is not None:
        tr["sd_cx_px_per_degC"] = float(sd_cx) * ratio
    stamp = datetime.date.today().isoformat()
    for th in (tl, tr):
        ppm = float(th.get("ppm_per_degC") or 0.0)
        th.setdefault("fx_ppm_per_degC", ppm)
        th.setdefault("fy_ppm_per_degC", ppm)
        th["form"] = FORM_VERSION
    tr["note"] = (f"{tr.get('note', '')}; {FORM_VERSION} ({stamp}): NL-reference form - the stereo temperature change "
                  f"moved from NL cx ({moved.get('cx_px_per_degC', {}).get('NL_was', 0.0):+.4f} px/degC) to NR cx").strip("; ")
    tl["note"] = (f"{tl.get('note', '')}; {FORM_VERSION} ({stamp}): NL-reference form - NL's principal point does not "
                  f"move with temperature").strip("; ")
    out_files = {}
    for name, j, th in (("M2020_NL_fisheye_tangential.json", nl, tl), ("M2020_NR_fisheye_tangential.json", nr, tr)):
        j = dict(j, thermal=th, convention=CONVENTION, form=FORM_VERSION)
        j["source"] = f"{j.get('source', '')} [{FORM_VERSION} NL-reference form from {s}{'; ' + note if note else ''}]"
        (d / name).write_text(json.dumps(j, indent=1), encoding="utf-8")
        out_files[name] = sha12(d / name)
    r = dict(rig, convention=CONVENTION, form=FORM_VERSION, yaw_constant=True)
    r["yaw_model"] = "one yaw for every temperature (the stereo temperature change is NR cx's); yaw rate with sol"
    (d / "M2020_N_rig.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    out_files["M2020_N_rig.json"] = sha12(d / "M2020_N_rig.json")
    for extra in ("consensus.json",):
        if (s / extra).is_file():
            (d / extra).write_bytes((s / extra).read_bytes())
    (d / "CONVERTED.md").write_text(
        f"# {FORM_VERSION} NL-reference form ({stamp})\n\nFrom `{s}`.\n\n"
        + "".join(f"- {k}: NL {v['NL_was']:+.5f} -> NR {v['NR_now']:+.5f} px/degC (x fx_NR/fx_NL = {ratio:.6f})\n"
                  for k, v in moved.items())
        + "- fx, fy each their own ppm/degC (both the shared value)\n- convention: NL reference\n", encoding="utf-8")
    return {"files": out_files, "moved": moved, "ratio": ratio, "dst": str(d)}


def same_models(a: PathLike, b: PathLike) -> bool:
    """Whether two folders hold byte-identical Navcam model files."""
    a, b = Path(a), Path(b)
    return all((a / n).is_file() and (b / n).is_file() and (a / n).read_bytes() == (b / n).read_bytes()
               for n in NAVCAM_FILES)


def models_fingerprint(folder: Optional[PathLike] = None) -> Optional[str]:
    """SHA-256 (16 hex) over the Navcam model files and the Mastcam-Z focus models of ``folder`` (default: in use),
    part of the run key (v0p73): a promotion makes the batch run the finished sites again."""
    from ..paths import cmods_dir
    d = Path(folder) if folder else cmods_dir()
    h = hashlib.sha256()
    n = 0
    for f in [d / x for x in NAVCAM_FILES] + sorted(d.glob("M2020_ZCAM*_focus_model.json")):
        if f.is_file():
            h.update(f.name.encode())
            h.update(f.read_bytes())
            n += 1
    return h.hexdigest()[:16] if n else None
