"""
Make a consensus the current best camera models: copy it into ``src/mppp/data/cmods`` (v0p50; ``MPPP_CMODS`` overrides), where
notebook 03 looks for its start cameras by default (``NAVCAM_CAMERAS``, ``ZCAM_FOCUS_MODEL``).

    python scripts\\promote_cmods.py D:\\scapes\\colmap\\camera_analysis\\navcal_v0p44\\navcam_joint --note "v0p44 joint, 27 blocks"
    python scripts\\promote_cmods.py D:\\...\\M2020_ZCAM034_focus_model_candidate.json --note "7-block refit"
    python scripts\\promote_cmods.py --list

Accepted files (a folder is searched for them):

- ``M2020_NL_fisheye_tangential.json``, ``M2020_NR_fisheye_tangential.json`` (and the ``_rational`` pair): the
  Navcam cameras;  ``M2020_N_rig.json``: the Navcam stereo rig;
- ``M2020_ZCAM<zoom>_focus_model*.json``: a Mastcam-Z focus model, one per zoom (v0p53: 034, 048, 063, 110; saved as
  ``M2020_ZCAM<zoom>_focus_model.json``), with the zoom's distortion, principal point and rig when notebook 04 wrote them.

Each file is checked (it must load as a camera / rig / focus model) before anything is copied.  Files it replaces
go to ``cmods/history/<date-time>/``, and ``cmods/CHANGES.md`` gets a line with the source, the note
and the SHA-256 of each file.  Existing projects are rebuilt from the new models on their next notebook 03 run
(the models' fingerprints are part of the project check); features and matches are reused.
The folder is part of the git repository: commit the promotion (git add src/mppp/data/cmods; git commit).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAVCAM = ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_NL_rational.json",
          "M2020_NR_rational.json", "M2020_N_rig.json")
ZCAM = "M2020_ZCAM034_focus_model.json"
import re as _re
_ZCAM_RE = _re.compile(r"M2020_ZCAM(\d{3})_focus_model.*\.json")


def _zcam_name(fname: str):
    """``M2020_ZCAM048_focus_model_candidate.json`` -> ``M2020_ZCAM048_focus_model.json`` (v0p53); None otherwise."""
    m = _ZCAM_RE.fullmatch(fname)
    return f"M2020_ZCAM{m.group(1)}_focus_model.json" if m else None


def target_dir() -> Path:
    from mppp.paths import cmods_dir
    return cmods_dir()


def collect(sources):
    """{target name: source file} from folders and files."""
    out = {}
    for s in map(Path, sources):
        files = sorted(s.glob("*.json")) if s.is_dir() else [s]
        for f in files:
            if f.name in NAVCAM:
                out[f.name] = f
            elif _zcam_name(f.name):
                out[_zcam_name(f.name)] = f
        if s.is_file() and s.name not in NAVCAM and not _zcam_name(s.name):
            raise SystemExit(f"ERROR: {s.name} is not a camera model file this script knows ({', '.join(NAVCAM)}, "
                             f"M2020_ZCAM<zoom>_focus_model*.json)")
    return out


def check(name: str, f: Path) -> str:
    """Raise if ``f`` does not load as what ``name`` says; return a one-line description."""
    from mppp.sfm.project import camera_from_colmap_json
    d = json.loads(f.read_text(encoding="utf-8"))
    if name == "M2020_N_rig.json":
        if "R_sensor_from_ref" not in d and "rotation" not in json.dumps(d).lower():
            raise ValueError("no rig rotation in it")
        return f"rig, baseline {d.get('baseline_m', '?')} m"
    if _zcam_name(name):
        if not isinstance(d.get("cameras"), dict) or not d["cameras"]:
            raise ValueError("no 'cameras' in it")
        zoom = name[10:13]
        bad = [g for g in d["cameras"] if str(g)[2:5] != zoom]
        if bad:
            raise ValueError(f"cameras {bad} are not of the {int(zoom)} mm zoom")
        extra = [k for k in ("distortion", "pp0_px") if any(k in c for c in d["cameras"].values())]
        return (f"focus model for {sorted(d['cameras'])}" + (f" with {', '.join(extra)}" if extra else "")
                + (" and rig" if d.get("rig") else ""))
    cam = camera_from_colmap_json(f)
    th = d.get("thermal") or {}
    return (f"{cam['model']} {cam['width']}x{cam['height']}, fx {cam['params'][0]:.2f}"
            + (f" at {th.get('T0_degC')} degC" if th else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="folders or files with the new models")
    ap.add_argument("--note", default="", help="why these are the best models (goes into CHANGES.md)")
    ap.add_argument("--list", action="store_true", help="show the current models")
    ap.add_argument("--dry-run", action="store_true", help="check and show, copy nothing")
    a = ap.parse_args(argv)
    dst = target_dir()
    if a.list or not a.sources:
        print(f"current camera models: {dst}")
        for f in sorted(dst.glob("*.json")) if dst.is_dir() else []:
            try:
                what = check(f.name, f)
            except Exception as e:                                     # noqa: BLE001
                what = f"UNREADABLE ({e})"
            print(f"  {f.name:40s} {hashlib.sha256(f.read_bytes()).hexdigest()[:12]}  {what}")
        return 0
    todo = collect(a.sources)
    if not todo:
        print("ERROR: no camera model files found in", a.sources)
        return 1
    for name, f in todo.items():
        try:
            print(f"  {name:40s} <- {f}  ({check(name, f)})")
        except Exception as e:                                         # noqa: BLE001
            print(f"ERROR: {f} does not load as {name}: {e}; nothing copied")
            return 1
    if a.dry_run:
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    hist = dst / "history" / stamp
    lines = []
    for name, f in todo.items():
        old = dst / name
        if old.is_file():
            hist.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, hist / name)
        shutil.copy2(f, old)
        lines.append(f"  - `{name}` {hashlib.sha256(old.read_bytes()).hexdigest()[:12]} from `{f}`")
    with (dst / "CHANGES.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {stamp}" + (f" - {a.note}" if a.note else "") + "\n" + "\n".join(lines) + "\n"
                 + (f"  - replaced files kept in `history/{stamp}/`\n" if hist.is_dir() else ""))
    print(f"{len(todo)} file(s) now in {dst}" + (f"; the replaced ones are in {hist}" if hist.is_dir() else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
