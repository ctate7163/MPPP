"""
mppp.ibr.single_sites - the single-station test sites, from a config file (``mppp/data/single_sites.json``).

Christian, 10 Nov 2026: "We'll use these three as the test sites. organize the outputs in
D:\\rover_blocks\\single_sites\\Sol0052_S003D2046 and so forth" - "the single sites should come from a config file".

    python -m mppp single-sites                        every station of the config
    python -m mppp single-sites Sol0052_S003D2046 Sol0686_S032D1184    those
    python -m mppp single-sites --steps ods            only the panoramas (mesh and maps already there)
    python -m mppp single-sites --squares-only         only stack the eye images already there into the squares
    python -m mppp single-sites --list | --dry-run     what would run
    python -m mppp single-sites --install DIR          write run_single_site_ibr.bat into DIR

Per station and variant (masked: the rover masked out; rover: the rover shown) the steps run as their own processes,
``python -m mppp dense | mesh | depthmaps | ods`` (the GPU step through the pipeline's GPU lock), into
``<root>/<out>/<label>/`` and ``.../<label>/with_rover/``; the dense workspace stays in the block
(``colmap/dense_<label>``) and its records are copied next to the products.  Each station's console output goes to
``single_site_log.txt`` in its folder; ``single_sites_summary.md`` in ``<root>/<out>`` has one line per station and
variant (the steps' exit codes, how far the fused points reach, the plane, the panorama's coverage).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]
CONFIG = Path(__file__).resolve().parents[1] / "data" / "single_sites.json"
VARIANTS = ("masked", "rover")
from .station import station_key
STEPS = ("dense", "mesh", "depthmaps", "ods")
DEPTHMAPS_VARIANTS = ("masked",)     # 0.97.14: the per-image maps for the terrain only (config: "depthmaps_variants")
BAT_NAME = "run_single_site_ibr.bat"          # 0.97.0 (Christian, 12 Nov): was update_and_run_single_sites.bat
OLD_BAT_NAME = "update_and_run_single_sites.bat"
TONE_FILE = "tone_levels.json"                 # one per station, shared by its rover-shown and rover-masked panoramas

BAT = r"""@echo off
REM run_single_site_ibr.bat [LABEL ...] [--steps ...]  (written by python -m mppp single-sites --install)
REM 1. D:\code\MPPP fast-forwarded to the newest git bundle in %TRANSFER% - if that bundle is ahead of it
REM    (a bundle behind it, e.g. after a pull from GitHub, is left alone; nothing is overwritten)
REM 2. python -m mppp single-sites with the arguments given: the stations of src\mppp\data\single_sites.json
REM    0.97.0: the Perseverance model in the rover mesh, the panorama blended in 16 bits, ONE 16-to-8-bit scaling per
REM    station for the rover-shown and rover-masked versions (<station>\tone_levels.json), the per-image pictures as
REM    jpg in photometry_jpg (rad, depth linear / log, normal, incidence, emission, phase, shadow), a centre panorama
REM    with its depth, the viewpoint on the azimuth axis at the Mastcam-Z height
REM    examples:  run_single_site_ibr.bat Sol0052_S003D2046            one station
REM               run_single_site_ibr.bat --steps ods          the panoramas again (mesh and maps kept)
REM               run_single_site_ibr.bat --dry-run            the commands, not run
REM set MPPP_NO_PAUSE=1 before it: no pause at the end, and the exit code is the run's (for unattended chains)
REM Nothing in this file names a version, so it never needs replacing.
REM It also lives in scripts\windows of the repository (0.97.0): there it works on the tree it is in.  Everything
REM after the settings is one parenthesised block, read whole before it runs - the merge may replace this file.
REM A copy put into scripts\windows by hand (untracked) is removed before the merge, which brings the tracked one.
setlocal EnableDelayedExpansion
set "REPO=%MPPP_HOME%"
if not defined REPO if exist "%~dp0mppp_env.bat" for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
if not defined REPO set "REPO=D:\code\MPPP"
set "TRANSFER=D:\rover_blocks\_transfer"
cd /d "%REPO%" || ( echo no %REPO% & pause & exit /b 1 )
set "B="
(
  for /f "delims=" %%f in ('dir /b /o-d "%TRANSFER%\mppp_*.bundle" 2^>nul') do if not defined B set "B=%TRANSFER%\%%f"
  for /f %%h in ('git rev-parse HEAD') do set "H0=%%h"
  if defined B (
    git fetch -q "!B!" +main:refs/remotes/bundle/main
    if errorlevel 1 ( echo fetch from !B! failed - send Claude the lines above & pause & exit /b 1 )
    git merge-base --is-ancestor HEAD refs/remotes/bundle/main
    if errorlevel 1 (
      echo the newest bundle !B! is not ahead of %REPO% - the code is kept as it is
    ) else (
      git ls-files --error-unmatch scripts/windows/run_single_site_ibr.bat >nul 2>nul
      if errorlevel 1 if exist scripts\windows\run_single_site_ibr.bat del /q scripts\windows\run_single_site_ibr.bat
      git merge -q --ff-only refs/remotes/bundle/main
      if errorlevel 1 ( echo fast-forward failed - send Claude the lines above & pause & exit /b 1 )
    )
  )
  for /f %%h in ('git rev-parse HEAD') do set "H1=%%h"
  echo %REPO%: !H0:~0,7! -^> !H1:~0,7!
  call scripts\windows\mppp_env.bat || ( pause & exit /b 1 )
  python -m mppp single-sites %*
  set "RC=!errorlevel!"
  if not defined MPPP_NO_PAUSE pause
  exit /b !RC!
)
"""


SQUARES_BAT_NAME = "make_square.bat"
SQUARES_BAT = BAT.replace("REM run_single_site_ibr.bat [LABEL ...] [--steps ...]  (written by python -m mppp single-sites --install)",
                          "REM make_square.bat [LABEL ...]  (written by python -m mppp single-sites --install)")
SQUARES_BAT = SQUARES_BAT.replace("REM 2. python -m mppp single-sites with the arguments given: the stations of src\\mppp\\data\\single_sites.json",
                                  "REM 2. python -m mppp single-sites --squares-only: every station's two eye images already there stacked into\n"
                                  "REM    <station>[_rover]_360_3DTD.jpg / .png, the left eye on top (Christian, 10 Nov); nothing rendered")
SQUARES_BAT = SQUARES_BAT.replace("python -m mppp single-sites %*", "python -m mppp single-sites --squares-only %*")
assert "--squares-only %*" in SQUARES_BAT and "make_square.bat [LABEL" in SQUARES_BAT


def load_config(path: Optional[PathLike] = None) -> Dict[str, Any]:
    cfg = json.loads(Path(path or CONFIG).read_text(encoding="utf-8"))
    if not isinstance(cfg.get("stations"), dict) or not cfg["stations"]:
        raise ValueError(f"{path or CONFIG}: no 'stations'")
    for lab, st in cfg["stations"].items():
        if "station" not in st or not ("site" in st or "work" in st):
            raise ValueError(f"{lab}: needs 'station' and 'site' or 'work'")
        for v in st.get("variants", cfg.get("variants", VARIANTS)):
            if v not in VARIANTS:
                raise ValueError(f"{lab}: variant {v!r} is not one of {VARIANTS}")
    return cfg


def block_folder(st: Dict[str, Any], root: PathLike) -> Path:
    """The block of a station: ``work``, else the Navcam WORK folder of its ``site`` in the site definitions."""
    if st.get("work"):
        return Path(st["work"])
    from ..waypoints.sites import load_site_table, work_folder
    table = load_site_table()
    site = table["sites"].get(st["site"])
    if site is None:
        raise KeyError(f"site {st['site']!r} is not in the site definitions")
    return Path(work_folder(Path(root), st["site"], False, site["sols"]))


def frame_args(frames: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if frames.get("images"):
        out += ["--images", *frames["images"]]
    if frames.get("scale") is not None:
        out += ["--scale", str(frames["scale"])]
    if frames.get("sequences"):
        out += ["--sequence", *frames["sequences"]]
    return out


def plan(cfg: Dict[str, Any], root: PathLike, out_root: Optional[PathLike] = None, labels: Optional[Sequence[str]] = None,
         steps: Sequence[str] = STEPS, variants: Optional[Sequence[str]] = None, ods_width: Optional[int] = None) -> List[Dict[str, Any]]:
    """The runs, in order: one entry per station with its block, its output folder and the commands (argv lists of
    ``python -m mppp``), grouped by variant."""
    root = Path(root)
    out_root = Path(out_root) if out_root else root / cfg.get("out", "single_sites")
    from .station import station_key
    by_key: Dict[str, List[str]] = {}                           # 0.97.14: Sol0052_S003D2046 still names Sol0052_Sol0052_S003D2046
    for lab in cfg["stations"]:
        by_key.setdefault(station_key(lab).upper(), []).append(lab)
    todo = []
    for lab in (list(labels) if labels else list(cfg["stations"])):
        todo += [lab] if lab in cfg["stations"] else by_key.get(str(lab).upper(), [lab])
    unknown = [lab for lab in todo if lab not in cfg["stations"]]
    if unknown:
        raise KeyError(f"not in the config: {unknown} (there: {sorted(cfg['stations'])})")
    runs = []
    for lab in todo:
        st = cfg["stations"][lab]
        work = block_folder(st, root)
        vs = [v for v in (variants or st.get("variants") or cfg.get("variants") or VARIANTS)]
        fa = frame_args(st.get("frames") or {}) + [str(x) for x in (cfg.get("dense_args") or []) + (st.get("dense_args") or [])]
        near = st.get("nearby_m", cfg.get("nearby_m"))          # 0.97.0: the neighbours' frames constrain the ground
        if near:
            fa += ["--nearby-m", f"{float(near):g}"]
        base = Path(out_root) / lab
        cmds: List[Dict[str, Any]] = []
        if "dense" in steps:
            cmds.append({"variant": "masked", "step": "dense",
                         "argv": ["dense", str(work), "--station", st["station"], *fa, "--prepare", "--stereo", "--fusion"]})
            # 0.97.0: no second fusion with the rover kept - the rover-shown mesh is the terrain of the rover-masked
            # cloud with the Perseverance model on it (mppp mesh --rover; --rover-cloud is the 0.96 way)
        for v in vs:
            out = base if v == "masked" else base / "with_rover"
            extra = ["--rover"] if v == "rover" else ["--mask-rover"]    # 0.96.1: the rover shown by default
            for step in ("mesh", "depthmaps", "ods"):
                if step not in steps:
                    continue
                argv = [step, str(work), "--station", st["station"], *extra, "--out", str(out)]
                if step == "mesh" and vs.index(v) > 0 and "mesh" in steps:     # 0.97.0: the terrain mesh is built once,
                    first = vs[0]                                               # by the variant that runs first
                    argv += ["--terrain-from", str(base if first == "masked" else base / "with_rover")]
                if step == "depthmaps":
                    # 0.97.14 (Christian, 13 Nov 2026: "Maps for one variant only. Only run for the terrain. render
                    # depth and normals without photometry products by default"): the per-image maps took a fifth
                    # (Van Zyl) to a half (Taylorfjellet, full-resolution frames) of a station's run
                    if v not in (st.get("depthmaps_variants") or cfg.get("depthmaps_variants") or DEPTHMAPS_VARIANTS):
                        continue
                    if st.get("photometry", cfg.get("photometry", False)):
                        argv.append("--photometry")
                if step == "ods" and st.get("ods_exclude"):   # 10 Nov: pairs left out of the panorama
                    argv += ["--exclude", *[str(x) for x in st["ods_exclude"]]]
                if step == "ods":                    # 0.97.0: one 16-to-8-bit scaling for both variants of a station
                    argv += ["--tone-levels", str(base / TONE_FILE)]
                    from .ods import ods_prefix          # 0.97.14: the files carry the station's code of the list
                    argv += ["--prefix", ods_prefix(lab, rover=v == "rover")]
                    from .nadir import site_name         # 0.97.7: the site's name for the mark at the nadir
                    nm = st.get("name") or site_name(st.get("site"))
                    if nm:
                        argv += ["--site-name", str(nm)]
                if step == "mesh" and near:
                    argv += ["--keep-under-rover", f"{float(st.get('keep_under_rover_m', cfg.get('keep_under_rover_m', 0.25))):g}"]
                argv += [str(x) for x in (cfg.get(f"{step}_args") or [])]      # 0.96.1: for every station
                argv += [str(x) for x in st.get(f"{step}_args", [])]
                if step == "ods" and ods_width:      # after the config's arguments: the last --width counts
                    argv += ["--width", str(int(ods_width))]
                cmds.append({"variant": v, "step": step, "argv": argv})
        runs.append({"label": lab, "work": str(work), "out": str(base), "note": st.get("note", ""), "variants": vs,
                     "commands": cmds})
    return runs


STEP_OUTPUT = {"mesh": "mesh.ply", "depthmaps": "maps_report.md", "ods": "ods.json"}
STATUS_FILE = "single_sites_status.json"


def code_key() -> str:
    """A hash of the code the steps run (the ``mppp.ibr`` sources and the dense step's): a step is made again when it
    changes, not on every version number."""
    import hashlib
    h = hashlib.sha1()
    here = Path(__file__).resolve().parent
    for f in sorted(here.glob("*.py")) + [here.parent / "sfm" / "dense.py"]:
        if f.is_file() and f.name != "single_sites.py":
            h.update(f.name.encode() + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def step_key(argv: Sequence[str], before: str, code: str, inputs: Sequence[PathLike] = ()) -> str:
    """The identity of one step's run: its command, the code, the step before it (``before``: that step's key) and
    the size and time of the files it reads (the fused cloud for the mesh)."""
    import hashlib
    parts = [code, before, *map(str, argv)]
    for f in inputs:
        f = Path(f)
        parts.append(f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" if f.is_file() else f"{f.name}:-")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def step_record(out: PathLike, step: str) -> Path:
    return Path(out) / f"step_{step}.json"


def step_done(out: PathLike, step: str, key: str) -> bool:
    """True when ``step`` already ran in ``out`` with this key, ended with exit 0 and its product is still there."""
    f = step_record(out, step)
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return rec.get("key") == key and rec.get("exit") == 0 and (Path(out) / STEP_OUTPUT[step]).is_file()


def _status(out_root: Path, **kw) -> None:
    """``single_sites_status.json`` in the output root: what is running now (station, variant, step, since when) and
    what is done - so the run can be read without asking (written whole each time)."""
    try:
        (out_root / STATUS_FILE).write_text(json.dumps(dict(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1),
                                            encoding="utf-8")
    except OSError:
        pass


def _run(argv: List[str], log_fh) -> int:
    """``python -m mppp <argv>`` with its output to the console and the log."""
    cmd = [sys.executable, "-m", "mppp", *argv]
    log_fh.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')}  {' '.join(cmd)}\n")
    log_fh.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        log_fh.write(line)
        log_fh.flush()                               # 0.97.0: the log shows where a long step is
    p.wait()
    log_fh.write(f"===== exit {p.returncode}\n")
    log_fh.flush()
    return int(p.returncode)


def _read(f: PathLike) -> Dict[str, Any]:
    try:
        return json.loads(Path(f).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _summary_line(run: Dict[str, Any], variant: str, codes: Dict[str, int], mins: Optional[Dict[str, float]] = None) -> str:
    out = Path(run["out"]) / ("with_rover" if variant == "rover" else "")
    mins = mins or {}
    cells = [run["label"], variant, " ".join(f"{k} {v}" + (" (kept)" if k in mins and mins[k] is None else f" ({mins[k]:.0f} min)" if (mins.get(k) or 0) >= 0.5 else "")
                                             for k, v in codes.items()) or "-"]
    try:
        m = json.loads((out / "mesh_report.json").read_text(encoding="utf-8"))
        rings = [r for r in m.get("points_by_ring") or [] if r["points"]]
        cells.append(f"{m.get('points', 0):,} pts to {rings[-1]['ring_m'][1]:g} m" if rings else "-")
        pl = (m.get("scene") or {}).get("plane") or {}
        cells.append(f"{pl.get('tilt_deg', float('nan')):.1f} deg, {pl.get('rms_m', float('nan')):.2f} m rms" if pl else "-")
    except (OSError, ValueError, KeyError):
        cells += ["-", "-"]
    try:
        o = json.loads((out / "ods.json").read_text(encoding="utf-8"))
        c = o["eyes"]["left"]["coverage"]
        cells.append(f"mesh {100 * c['mesh']:.0f} %, plane {100 * c.get('plane', 0):.0f} %, far {100 * c['far']:.0f} %, "
                     f"coloured {100 * c['coloured']:.0f} %")
        cells.append("yes" if any(out.glob("*360_3DTD.jpg")) else "no")
    except (OSError, ValueError, KeyError):
        cells += ["-", "-"]
    return "| " + " | ".join(cells) + " |"


def run_all(runs: List[Dict[str, Any]], out_root: PathLike, squares_only: bool = False, retone: bool = False,
            force: bool = False) -> int:
    """Runs the plan.  0.97.2: a step whose command, code, inputs and upstream steps are those of its last successful
    run in that folder is skipped (``step_<name>.json`` next to its products; ``force`` runs everything); the summary
    has each step's minutes; ``single_sites_status.json`` says what is running."""
    out_root = Path(out_root)
    code = code_key()
    done: List[str] = []
    t_run = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_root.mkdir(parents=True, exist_ok=True)
    prev = out_root / "single_sites_summary.md"
    if prev.is_file():                                         # the run before, for comparison
        shutil.copy2(prev, out_root / "single_sites_summary_previous.md")
    lines = ["# Single-site runs", "", f"{time.strftime('%Y-%m-%d %H:%M')}", "",
             "| station | variant | exit codes (minutes) | fused points | ground plane | panorama (left eye) | square |",
             "|---|---|---|---|---|---|---|"]
    bad = 0
    for run in runs:
        base = Path(run["out"])
        base.mkdir(parents=True, exist_ok=True)
        print(f"\n######## {run['label']}: {run['note']}\n######## block {run['work']} -> {base}", flush=True)
        codes: Dict[str, Dict[str, int]] = {v: {} for v in run["variants"]}
        mins: Dict[str, Dict[str, Optional[float]]] = {}
        with open(base / "single_site_log.txt", "a", encoding="utf-8") as fh:
            if retone:
                if (base / TONE_FILE).is_file():
                    (base / TONE_FILE).unlink()          # found again on the first variant, used for both
                for v in run["variants"]:
                    out = base / ("with_rover" if v == "rover" else "")
                    if any(out.glob("*ods_right_16.png")):
                        codes[v]["retone"] = _run(["ods", "-", "--station", "x", "--retone", "--out", str(out),
                                                   "--tone-levels", str(base / TONE_FILE)], fh)
            elif squares_only:
                for v in run["variants"]:
                    out = base / ("with_rover" if v == "rover" else "")
                    if any(out.glob("*ods_right.png")):
                        from .ods import ods_prefix
                        codes[v]["square"] = _run(["ods", "-", "--station", "x", "--stack-only", "--stack-top", "left",
                                                   "--prefix", ods_prefix(run["label"], rover=v == "rover"),
                                                   "--out", str(out)], fh)
            else:
                failed = set()
                ods_ran = False
                ods_keys: List[str] = []                   # the station's ods steps so far: one scaling for its variants
                work_ = Path(run["work"])
                keys: Dict[str, str] = {}                  # the key of the last step of each variant ("" for dense)
                for c in run["commands"]:
                    v, step = c["variant"], c["step"]
                    if v in failed:
                        continue
                    vout = base / ("with_rover" if v == "rover" else "")
                    _status(out_root, started=t_run, station=run["label"], variant=v, step=step,
                            since=time.strftime("%Y-%m-%dT%H:%M:%S"), done=done, stations=[r["label"] for r in runs])
                    t0 = time.time()
                    if step == "dense":                    # its own cache (mppp dense: stereo and fusion records)
                        rc = _run(c["argv"], fh)
                    else:
                        clouds = sorted((work_ / "colmap").glob(f"dense_{station_key(run['label'])}*/fused.ply")) if step == "mesh" else []
                        tf = c["argv"][c["argv"].index("--terrain-from") + 1] if "--terrain-from" in c["argv"] else None
                        before = keys.get(v, "") + (json.dumps(_read(step_record(tf, "mesh")).get("key")) if tf else "")
                        if step == "ods":                  # the second variant uses the first one's levels
                            before += "".join(ods_keys)
                        key = step_key(c["argv"], before, code, clouds)
                        keys[v] = key
                        if step == "ods":
                            ods_keys.append(key)
                        if not force and step_done(vout, step, key) and not (step == "ods" and ods_ran):
                            print(f"######## {run['label']} {v}: {step} is up to date - skipped (--force runs it again)", flush=True)
                            codes[v][step] = 0
                            mins.setdefault(v, {})[step] = None
                            continue
                        step_record(vout, step).unlink(missing_ok=True)
                        ods_ran = ods_ran or step == "ods"     # the levels are new: the station's next panorama follows
                        if step == "ods" and len(ods_keys) == 1 and (base / TONE_FILE).is_file():
                            (base / TONE_FILE).unlink()    # the levels are found again by the station's first panorama
                        rc = _run(c["argv"], fh)
                        if rc == 0:
                            vout.mkdir(parents=True, exist_ok=True)
                            step_record(vout, step).write_text(json.dumps(
                                {"key": key, "exit": 0, "argv": c["argv"], "code": code, "minutes": round((time.time() - t0) / 60, 2),
                                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1), encoding="utf-8")
                    mins.setdefault(v, {})[step] = (time.time() - t0) / 60
                    codes[v][step] = rc
                    done.append(f"{run['label']} {v} {step}: exit {rc}, {(time.time() - t0) / 60:.1f} min")
                    if rc:
                        failed.add(v)
                        if c["step"] == "dense" and v == "masked":
                            failed.update(run["variants"])          # no cloud: nothing downstream can run
                        print(f"######## {run['label']} {v}: {c['step']} failed (exit {rc}) - the rest of this variant skipped")
        work = Path(run["work"])
        for name, dst in (("mppp_dense.json", "dense_workspace.json"), ("mppp_dense_run.json", "dense_run.json")):
            for f in sorted((work / "colmap").glob(f"dense_{station_key(run['label'])}*/{name}")):
                shutil.copy2(f, base / dst)
        for v in run["variants"]:
            bad += any(codes[v].values())
            lines.append(_summary_line(run, v, codes[v], mins.get(v)))
    _status(out_root, started=t_run, station=None, step=None, finished=time.strftime("%Y-%m-%dT%H:%M:%S"), done=done,
            stations=[r["label"] for r in runs], failed=int(bad))
    (out_root / "single_sites_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[4:]) + f"\n\nsummary: {out_root / 'single_sites_summary.md'}")
    return 1 if bad else 0


def install(dest: PathLike) -> Path:
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    f = d / BAT_NAME
    f.write_bytes(BAT.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))
    if (d / OLD_BAT_NAME).is_file():                  # 0.97.0: the bat of before, replaced by BAT_NAME
        (d / OLD_BAT_NAME).unlink()
    g = d / SQUARES_BAT_NAME                          # 10 Nov: the squares alone, the left eye on top
    g.write_bytes(SQUARES_BAT.replace("\r\n", "\n").replace("\n", "\r\n").encode("ascii"))
    return f


def add_parser(sub) -> None:
    p = sub.add_parser("single-sites", help="0.96: the single-station test sites of mppp/data/single_sites.json - dense, "
                       "mesh, depthmaps, ods, rover masked and shown (mppp.ibr.single_sites)")
    p.add_argument("labels", nargs="*", help="stations of the config (default: all)")
    p.add_argument("--config", default=None, help="the config file (default mppp/data/single_sites.json)")
    p.add_argument("--root", default=None, help="the blocks' root (default MPPP_SCAPES or D:\\rover_blocks)")
    p.add_argument("--out", default=None, help="the output root (default <root>/<config 'out'>)")
    p.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    p.add_argument("--variants", nargs="+", default=None, choices=VARIANTS)
    p.add_argument("--width", type=int, default=None, metavar="PX",
                   help="0.97.0: the panorama's width for this run (default: the config's, else mppp ods' own, 8192; 16384 for 16K) - give "
                        "--out as well to keep the panoramas already made")
    p.add_argument("--squares-only", action="store_true", help="only stack the eye images already there into "
                   "360_3DTD.jpg / .png")
    p.add_argument("--retone", action="store_true", help="only make the 8-bit panoramas and squares again from the "
                   "16-bit ones already there (no rendering)")
    p.add_argument("--force", action="store_true", help="0.97.2: run every step again (default: a mesh, depthmaps or ods "
                   "step whose command, code, cloud and upstream steps are unchanged since its last good run is skipped)")
    p.add_argument("--publish", metavar="DIR", default=None,
                   help="0.97.3: after the run, the stations' stereo JPEGs into DIR/panos and DIR/panos.json for the web "
                        "viewer (docs/ods of the repository; the page is copied into DIR when it is another folder)")
    p.add_argument("--publish-only", action="store_true", help="with --publish: no steps, publish what --out holds")
    p.add_argument("--publish-width", type=int, default=8192, metavar="PX", help="wider images are reduced to this (default 8192)")
    p.add_argument("--publish-views", nargs="+", default=["standard", "enhanced"], metavar="VIEW",
                   help="standard, enhanced, 'enhanced local' (default: standard enhanced)")
    p.add_argument("--list", action="store_true", help="the stations of the config")
    p.add_argument("--dry-run", action="store_true", help="the commands, not run")
    p.add_argument("--install", metavar="DIR", default=None, help=f"write {BAT_NAME} (and make_square.bat) into DIR and stop")


def cmd_single_sites(a: argparse.Namespace) -> int:
    import os
    if a.install:
        print(f"written: {install(a.install)}")
        return 0
    cfg = load_config(a.config)
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or r"D:\rover_blocks")
    out_root = Path(a.out) if a.out else root / cfg.get("out", "single_sites")
    if a.list:
        for lab, st in cfg["stations"].items():
            print(f"{lab}  {st['station']:>9}  {st.get('site') or st.get('work')}  {st.get('note', '')}")
        return 0
    runs = plan(cfg, root, out_root, a.labels, a.steps, a.variants, ods_width=a.width)
    if a.dry_run:
        for r in runs:
            print(f"{r['label']}: {r['work']} -> {r['out']}")
            for c in r["commands"]:
                print(f"   [{c['variant']}] python -m mppp " + " ".join(c["argv"]))
        return 0
    rc = 0 if (a.publish and a.publish_only) else run_all(runs, out_root, squares_only=a.squares_only, retone=a.retone,
                                                          force=getattr(a, "force", False))
    if a.publish:
        from . import web
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        st = cfg.get("stations") or {}
        titles = {k: {"title": v.get("title") or f"{str(v.get('site', '')).replace('_', ' ').title()} {k}".strip(),
                      "note": v.get("web_note"), "yaw_deg": v.get("yaw_deg")} for k, v in st.items()}
        doc = web.publish(out_root, a.publish, stations=[r["label"] for r in runs], views=a.publish_views,
                          width=a.publish_width, titles=titles)
        if not web.install_viewer(a.publish):
            print("the viewer page (docs/ods/index.html, vendor/) is not in this installation: copy it from the repository")
        print(f"published: {len(doc['panos'])} stations in {Path(a.publish) / 'panos.json'}")
    return rc
