"""
mppp.ibr.single_sites - the single-station test sites, from a config file (``mppp/data/single_sites.json``).

Christian, 10 Nov 2026: "We'll use these three as the test sites. organize the outputs in
D:\\rover_blocks\\single_sites\\S003D2046 and so forth" - "the single sites should come from a config file".

    python -m mppp single-sites                        every station of the config
    python -m mppp single-sites S003D2046 S032D1184    those
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
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]
CONFIG = Path(__file__).resolve().parents[1] / "data" / "single_sites.json"
VARIANTS = ("masked", "rover")
STEPS = ("dense", "mesh", "depthmaps", "ods")
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
REM    examples:  run_single_site_ibr.bat S003D2046            one station
REM               run_single_site_ibr.bat --steps ods          the panoramas again (mesh and maps kept)
REM               run_single_site_ibr.bat --dry-run            the commands, not run
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
  pause
  exit /b
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
    todo = list(labels) if labels else list(cfg["stations"])
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
                    argv.append("--photometry")
                if step == "ods" and st.get("ods_exclude"):   # 10 Nov: pairs left out of the panorama
                    argv += ["--exclude", *[str(x) for x in st["ods_exclude"]]]
                if step == "ods":                    # 0.97.0: one 16-to-8-bit scaling for both variants of a station
                    argv += ["--tone-levels", str(base / TONE_FILE)]
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


def _summary_line(run: Dict[str, Any], variant: str, codes: Dict[str, int]) -> str:
    out = Path(run["out"]) / ("with_rover" if variant == "rover" else "")
    cells = [run["label"], variant, " ".join(f"{k} {v}" for k, v in codes.items()) or "-"]
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


def run_all(runs: List[Dict[str, Any]], out_root: PathLike, squares_only: bool = False, retone: bool = False) -> int:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prev = out_root / "single_sites_summary.md"
    if prev.is_file():                                         # the run before, for comparison
        shutil.copy2(prev, out_root / "single_sites_summary_previous.md")
    lines = ["# Single-site runs", "", f"{time.strftime('%Y-%m-%d %H:%M')}", "",
             "| station | variant | exit codes | fused points | ground plane | panorama (left eye) | square |",
             "|---|---|---|---|---|---|---|"]
    bad = 0
    for run in runs:
        base = Path(run["out"])
        base.mkdir(parents=True, exist_ok=True)
        print(f"\n######## {run['label']}: {run['note']}\n######## block {run['work']} -> {base}", flush=True)
        codes: Dict[str, Dict[str, int]] = {v: {} for v in run["variants"]}
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
                if any(c["step"] == "ods" for c in run["commands"]) and (base / TONE_FILE).is_file():
                    (base / TONE_FILE).unlink()          # 0.97.0: the levels of the run before are not carried over
                for c in run["commands"]:
                    v = c["variant"]
                    if v in failed:
                        continue
                    rc = _run(c["argv"], fh)
                    codes[v][c["step"]] = rc
                    if rc:
                        failed.add(v)
                        if c["step"] == "dense" and v == "masked":
                            failed.update(run["variants"])          # no cloud: nothing downstream can run
                        print(f"######## {run['label']} {v}: {c['step']} failed (exit {rc}) - the rest of this variant skipped")
        work = Path(run["work"])
        for name, dst in (("mppp_dense.json", "dense_workspace.json"), ("mppp_dense_run.json", "dense_run.json")):
            for f in sorted((work / "colmap").glob(f"dense_{run['label']}*/{name}")):
                shutil.copy2(f, base / dst)
        for v in run["variants"]:
            bad += any(codes[v].values())
            lines.append(_summary_line(run, v, codes[v]))
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
                   help="0.97.0: the panorama's width for this run (e.g. 16384; the config's or 8192 otherwise) - give "
                        "--out as well to keep the panoramas already made")
    p.add_argument("--squares-only", action="store_true", help="only stack the eye images already there into "
                   "360_3DTD.jpg / .png")
    p.add_argument("--retone", action="store_true", help="only make the 8-bit panoramas and squares again from the "
                   "16-bit ones already there (no rendering)")
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
    return run_all(runs, out_root, squares_only=a.squares_only, retone=a.retone)
