"""
rpp.ibr.single_sites - the single-station test sites, from a config file (``rpp/data/single_sites.json``).

Christian, 10 Nov 2026: "We'll use these three as the test sites. organize the outputs in
D:\\rover_blocks\\single_sites\\Sol0052_S003D2046 and so forth" - "the single sites should come from a config file".

    python -m rpp single-sites                        every station of the config
    python -m rpp single-sites Sol0052_S003D2046 Sol0686_S032D1184    those
    python -m rpp single-sites --steps ods            only the panoramas (mesh and maps already there)
    python -m rpp single-sites --squares-only         only stack the eye images already there into the squares
    python -m rpp single-sites --list | --dry-run     what would run
    python -m rpp single-sites --install DIR          write run_single_site_ibr.bat into DIR

Per station and variant (masked: the rover masked out; rover: the rover shown) the steps run as their own processes,
``python -m rpp dense | mesh | depthmaps | ods`` (the GPU step through the pipeline's GPU lock), into
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
from .. import env as rpp_env

PathLike = Union[str, Path]
CONFIG = Path(__file__).resolve().parents[1] / "data" / "single_sites.json"
VARIANTS = ("masked", "rover")
from .station import station_key
STEPS = ("dense", "mesh", "depthmaps", "ods")
BLOCK_WORKSPACE = "dense_block_N"       # rpp dense --whole-block --cameras N: the block's Navcam cloud ("terrain": "block")


def block_terrain(st: Dict[str, Any], cfg: Dict[str, Any]):
    """The block terrain of a station: None (its own frames), ("", None) for ``"terrain": "block"`` (every Navcam frame
    of the block), or (name, [frames]) for ``"terrain": "block:<name>"`` - 0.98.19, a set of the config's
    ``terrain_sets`` (Christian, 16 Nov 2026: the Rockytop panoramas "rendered from one large 3d model that includes
    all the medium res eod images and the lowres mid drive imaging frames")."""
    t = st.get("terrain", cfg.get("terrain"))
    if t in ("own", "station"):                                 # 0.99.20: the block terrain is the config's default
        return None
    if not t or not str(t).startswith("block"):
        return None
    name = str(t).split(":", 1)[1].strip() if ":" in str(t) else ""
    if not name:
        return "", None
    sets = cfg.get("terrain_sets") or {}
    if name not in sets:
        raise KeyError(f"terrain {t!r}: no terrain set {name!r} in the config (there: {sorted(sets)})")
    return name, [n if str(n).lower().endswith(".png") else f"{n}.png" for n in sets[name]["images"]]


def mark_name(st: Dict[str, Any]) -> Optional[str]:
    """The block's name in a station's nadir mark: ``"mark_name"``, else its ``"name"`` without a trailing camera tag such
    as ``(Z34)`` (0.99.13, Christian, 17 Nov 2026: "The Zcam blue circles should have the same titles as the block name
    as the ncam versions" - "Three Forks North (Z34)" is "Three Forks North" on the disc), else the block's name."""
    import re
    if st.get("mark_name"):
        return str(st["mark_name"])
    nm = st.get("name")
    if nm:
        return re.sub(r"\s*\((?:Z|Mastcam-Z)\s*\d*(?:\s*mm)?\)\s*$", "", str(nm), flags=re.I) or str(nm)
    from .nadir import block_name
    return block_name(st.get("site"))


ZCAM_TAG = "(zcam)"      # 0.99.36 (Christian, 19 Nov 2026: "The z34 sites should have a (zcam) after the site name to
#                          differentiate them")


def is_zcam_station(st: Dict[str, Any]) -> bool:
    """Whether a station's panorama is made from Mastcam-Z frames (``--cameras Z...`` in its ``ods_args``, or a
    camera tag such as ``(Z34)`` in its name)."""
    import re
    args = [str(x) for x in (st.get("ods_args") or [])]
    for i, x in enumerate(args[:-1]):
        if x == "--cameras" and args[i + 1].upper().startswith("Z"):
            return True
    return bool(re.search(r"\((?:Z|Mastcam-Z)\s*\d*(?:\s*mm)?\)\s*$", str(st.get("name") or ""), flags=re.I))


def viewer_name(st: Dict[str, Any], label: str = "") -> str:
    """The station's name in the viewer's menu and info (0.99.36): the name on its disc (:func:`mark_name`), with
    ``(zcam)`` after it for a Mastcam-Z station - "Three Forks North (zcam)" beside the Navcam "Three Forks North".
    The disc itself keeps the block's name alone (0.99.13)."""
    nm = mark_name(st) or label
    return f"{nm} {ZCAM_TAG}" if is_zcam_station(st) and ZCAM_TAG not in nm.lower() else nm


def viewer_sols(st: Dict[str, Any], label: str = "") -> Optional[List[int]]:
    """For a Mastcam-Z station the one sol the viewer gives (as on its disc, :func:`rpp.ibr.nadir.mark_sols`): the
    first of its ``--sols`` in ``ods_args``, else the sol in its label; None for the others (the disc's sols)."""
    import re
    if not is_zcam_station(st):
        return None
    args = [str(x) for x in (st.get("ods_args") or [])]
    for i, x in enumerate(args[:-1]):
        if x == "--sols" and args[i + 1].isdigit():
            return [int(args[i + 1])]
    m = re.match(r"^Sol(\d+)", str(label))
    return [int(m.group(1))] if m else None


def block_workspace(name: str = "", cameras: Optional[str] = None) -> str:
    """The block cloud's workspace folder: dense_block_N (0.99.43: dense_block_<families> for ``cameras``, sorted, as
    rpp dense --block-cameras names it), or dense_block_<name> for a terrain set."""
    if name:
        return f"dense_block_{name}"
    return f"dense_block_{''.join(sorted(set(cameras.upper())))}" if cameras else BLOCK_WORKSPACE


ZCAM_360_MIN_FRAMES = 36          # 0.99.43: left-eye 34 mm frames of one sol at a station that make it a "Z34 360"
_Z34 = __import__("re").compile(r"^ZL\d_(\d{4})_\d+_\w+?_N(\d{3})(\d{4})ZCAM\d{5}_034")


def zcam_companions(cfg: Dict[str, Any], root: PathLike, names_of=None) -> Dict[str, str]:
    """0.99.43 (Christian, 19 Nov 2026: "for stations with Z34 present like Butler Landing wildcat and skinner ridge,
    render those z34 versions right after the ncam versions" - "do this for every site from which we have a z34 360"):
    for every Navcam station of the config whose site's Mastcam-Z block holds at least ZCAM_360_MIN_FRAMES left-eye
    34 mm frames of one sol taken at the station (the site and drive in the frames' names), a Mastcam-Z station is
    added to ``cfg["stations"]`` - ``Sol<that sol>_<station>_Z34``, as the Three Forks ones: the block is the site's
    Mastcam-Z block, the panorama from the 34 mm frames of the sol with the most (half size, no gains), the Navcam
    and front Hazcam frames of the station fill what Mastcam-Z did not see (its own, by pattern, and the Navcam
    station's context frames), the terrain the block's.  A station that already has a Mastcam-Z entry in the config
    gets none.  Returns {Navcam label: Mastcam-Z label} for every pair, the config's own included: the order of the
    runs (:func:`plan` puts a companion right after its Navcam station).  ``names_of(block folder)`` -> the block's
    image names (default: its project.json)."""
    import re
    from .station import station_key
    from ..waypoints.sites import load_site_table, work_folder

    def default_names(work):
        from ..sfm.project import SfmProject
        d = Path(work) / "colmap"
        return [r["name"] for r in SfmProject.load(d).images] if (d / "project.json").is_file() else []
    names_of = names_of or default_names
    table = load_site_table()["sites"]
    have = {}                                                    # station key -> the config's own Mastcam-Z label
    for lab, st in cfg["stations"].items():
        if is_zcam_station(st):
            have[re.sub(r"_Z\d+$", "", station_key(lab).upper())] = lab
    pairs: Dict[str, str] = {}
    cache: Dict[str, List[str]] = {}
    for lab, st in list(cfg["stations"].items()):
        if is_zcam_station(st):
            continue
        key = station_key(lab).upper()
        if key in have:
            pairs[lab] = have[key]
            continue
        m = re.match(r"^S(\d+)D(\d+)$", key)
        if not m:
            continue
        # 0.99.44: the station's own block when that is the Mastcam-Z block ("work": ..._zcam - most stations since the
        # freeze blocks), else its site's Mastcam-Z block, else <its block>_zcam
        own = block_folder(st, root) if (st.get("work") or st.get("site") in table) else None
        if own is not None and own.name.endswith("_zcam"):
            zwork = own
        elif st.get("site") in table:
            zwork = Path(work_folder(Path(root), st["site"], True, table[st["site"]]["sols"]))
        elif own is not None:
            zwork = own.with_name(own.name + "_zcam")
        else:
            continue
        if not block_aligned(zwork):                             # not there or not aligned yet: looked at again later
            cfg.setdefault("_zcam_pending", {})[lab] = str(zwork)
            continue
        if str(zwork) not in cache:
            try:
                cache[str(zwork)] = list(names_of(zwork))
            except Exception:                                    # noqa: BLE001 - a block that cannot be read has none
                cache[str(zwork)] = []
        sd = (int(m.group(1)), int(m.group(2)))
        per_sol: Dict[int, int] = {}
        for n in cache[str(zwork)]:
            z = _Z34.match(n)
            if z and (int(z.group(2)), int(z.group(3))) == sd:
                per_sol[int(z.group(1))] = per_sol.get(int(z.group(1)), 0) + 1
        if not per_sol or max(per_sol.values()) < int(cfg.get("zcam_360_min_frames", ZCAM_360_MIN_FRAMES)):
            continue
        sol = max(per_sol, key=lambda k: (per_sol[k], -k))
        tok = f"N{sd[0]:03d}{sd[1]:04d}"
        zlab = f"Sol{sol:04d}_{key}_Z34"
        cfg["stations"][zlab] = {
            "name": f"{mark_name(st) or lab} (Z34)", "work": zwork.name, "station": st["station"], "derived_from": lab,
            "frames": dict(st.get("frames") or {}), "variants": ["masked", "rover"], "terrain": "block",
            "ods_args": ["--cameras", "Z", "--sols", str(sol), "--zcam-scale", "0.5", "--fill-cameras", "NF", "--no-gains"],
            "context": [f"N?F_*_{tok}NCAM*.png", f"F?F_*_{tok}FHAZ*.png", *(st.get("context") or [])],
            "note": f"0.99.43, made by zcam_companions: the {per_sol[sol]} left-eye 34 mm frames of sol {sol} at this station "
                    f"(the Mastcam-Z block {zwork.name}), after the Navcam station {lab}"}
        pairs[lab] = zlab
    return pairs
DEPTHMAPS_VARIANTS = ("masked",)     # 0.97.14: the per-image maps for the terrain only (config: "depthmaps_variants")
BAT_NAME = "run_single_site_ibr.bat"          # 0.97.0 (Christian, 12 Nov): was update_and_run_single_sites.bat
OLD_BAT_NAME = "update_and_run_single_sites.bat"
TONE_FILE = "tone_levels.json"                 # one per station, shared by its rover-shown and rover-masked panoramas

BAT = r"""@echo off
REM run_single_site_ibr.bat [LABEL ...] [--steps ...]  (written by python -m rpp single-sites --install)
REM 1. the tree fast-forwarded to the newest git bundle in %TRANSFER% - if that bundle is ahead of it; merged
REM    when both have commits of their own (0.97.21); left alone when it already holds the bundle; nothing overwritten.
REM    0.99.3: the tree's own line - on branch rpp (D:\code\RPP) the newest rpp_v*.bundle's rpp, on main
REM    (D:\code\MPPP) the newest mppp_v*.bundle's main (0.99.0-0.99.2 fetched main from an rpp bundle, which has none)
REM 2. python -m rpp single-sites with the arguments given: the stations of src\rpp\data\single_sites.json
REM    0.97.0: the Perseverance model in the rover mesh, the panorama blended in 16 bits, ONE 16-to-8-bit scaling per
REM    station for the rover-shown and rover-masked versions (<station>\tone_levels.json), the per-image pictures as
REM    jpg in photometry_jpg (rad, depth linear / log, normal, incidence, emission, phase, shadow), a center panorama
REM    with its depth, the viewpoint on the azimuth axis at the Mastcam-Z height
REM    examples:  run_single_site_ibr.bat Sol0052_S003D2046            one station
REM               run_single_site_ibr.bat --steps ods          the panoramas again (mesh and maps kept)
REM               run_single_site_ibr.bat --dry-run            the commands, not run
REM               run_single_site_ibr.bat Sol0052_S003D2046 --no-upload  the run without the upload at its end
REM               run_single_site_ibr.bat --publish-only                 no rendering: publish and push what is there
REM    0.97.24: when the run is done its panoramas are published and uploaded (--upload is given here; --no-upload
REM    after it wins).  0.99.3: the pictures go to the Hugging Face dataset ctate7163/rpp-rpv, station by station,
REM    and panos.json to the viewer's GitHub checkout (D:\code\RPV), committed and pushed; --host github puts the
REM    pictures into that checkout as before.  A push that fails is tried three times and goes up with the next run:
REM    see "NOT (all) uploaded" at the end and upload_pending.json in the viewer's folder.
REM set RPP_NO_PAUSE=1 before it: no pause at the end, and the exit code is the run's (for unattended chains)
REM Nothing in this file names a version, so it never needs replacing.
REM It also lives in scripts\windows of the repository (0.97.0): there it works on the tree it is in.  Everything
REM after the settings is one parenthesized block, read whole before it runs - the merge may replace this file.
REM A copy put into scripts\windows by hand (untracked) is removed before the merge, which brings the tracked one.
setlocal EnableDelayedExpansion
set "REPO="
if exist "%~dp0rpp_env.bat" for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
if not defined REPO set "REPO=%RPP_HOME%"
if not defined REPO set "REPO=D:\code\RPP"
set "TRANSFER=D:\rover_blocks\_transfer"
cd /d "%REPO%" || ( echo no %REPO% & pause & exit /b 1 )
set "B="
set "BR=main"
(
  for /f %%b in ('git rev-parse --abbrev-ref HEAD') do set "BR=%%b"
  set "PAT=mppp_v*.bundle"
  set "SRC=!BR!"
  if exist "%REPO%\src\rpp\__init__.py" ( set "PAT=rpp_v*.bundle" & set "SRC=rpp" )
  for /f "delims=" %%f in ('dir /b /o-d "%TRANSFER%\!PAT!" 2^>nul') do if not defined B set "B=%TRANSFER%\%%f"
  for /f %%h in ('git rev-parse HEAD') do set "H0=%%h"
  if defined B (
    git fetch -q "!B!" +!SRC!:refs/remotes/bundle/!SRC!
    if errorlevel 1 ( echo fetch of !SRC! from !B! failed - send Claude the lines above & pause & exit /b 1 )
    git merge-base --is-ancestor HEAD refs/remotes/bundle/!SRC!
    if errorlevel 1 (
      git merge-base --is-ancestor refs/remotes/bundle/!SRC! HEAD
      if errorlevel 1 (
        git merge -q --no-edit refs/remotes/bundle/!SRC!
        if errorlevel 1 ( git merge --abort & echo the newest bundle !B! does not merge cleanly into %REPO% - send Claude the lines above & pause & exit /b 1 )
        echo the newest bundle !B! merged into %REPO%
      ) else (
        echo %REPO% already holds the newest bundle !B!
      )
    ) else (
      git ls-files --error-unmatch scripts/windows/run_single_site_ibr.bat >nul 2>nul
      if errorlevel 1 if exist scripts\windows\run_single_site_ibr.bat del /q scripts\windows\run_single_site_ibr.bat
      git merge -q --ff-only refs/remotes/bundle/!SRC!
      if errorlevel 1 ( echo fast-forward failed - send Claude the lines above & pause & exit /b 1 )
    )
  )
  for /f %%h in ('git rev-parse HEAD') do set "H1=%%h"
  echo %REPO%: !H0:~0,7! -^> !H1:~0,7!
  call scripts\windows\rpp_env.bat || ( pause & exit /b 1 )
  python -m rpp single-sites --upload %*
  set "RC=!errorlevel!"
  if not defined RPP_NO_PAUSE pause
  exit /b !RC!
)
"""


SQUARES_BAT_NAME = "make_square.bat"
SQUARES_BAT = BAT.replace("REM run_single_site_ibr.bat [LABEL ...] [--steps ...]  (written by python -m rpp single-sites --install)",
                          "REM make_square.bat [LABEL ...]  (written by python -m rpp single-sites --install)")
SQUARES_BAT = SQUARES_BAT.replace("REM 2. python -m rpp single-sites with the arguments given: the stations of src\\rpp\\data\\single_sites.json",
                                  "REM 2. python -m rpp single-sites --squares-only: every station's two eye images already there stacked into\n"
                                  "REM    <station>[_rover]_360_3DTD.jpg / .png, the left eye on top (Christian, 10 Nov); nothing rendered")
SQUARES_BAT = SQUARES_BAT.replace("python -m rpp single-sites --upload %*", "python -m rpp single-sites --squares-only %*")
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
    if st.get("work"):                                   # 0.98.4: a folder name alone is a block under the root (0.98.2's
        w = Path(st["work"])                             # Mastcam-Z station was looked for in the current folder)
        return w if w.is_absolute() or len(w.parts) > 1 else Path(root) / w
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


def block_aligned(work: PathLike) -> bool:
    """0.99.28: the block has finished its alignment (``colmap/run_done.json``)."""
    return (Path(work) / "colmap" / "run_done.json").is_file()


def plan(cfg: Dict[str, Any], root: PathLike, out_root: Optional[PathLike] = None, labels: Optional[Sequence[str]] = None,
         steps: Sequence[str] = STEPS, variants: Optional[Sequence[str]] = None, ods_width: Optional[int] = None,
         rover_surface: bool = False, station_mesh: Union[bool, str] = False,
         companions: Optional[Dict[str, str]] = None, block_cameras: Optional[str] = None,
         terrain: Optional[str] = None) -> List[Dict[str, Any]]:
    """The runs, in order: one entry per station with its block, its output folder and the commands (argv lists of
    ``python -m rpp``), grouped by variant."""
    root = Path(root)
    out_root = Path(out_root) if out_root else root / cfg.get("out", "single_sites")
    from .station import station_key
    by_key: Dict[str, List[str]] = {}                           # 0.97.14: Sol0052_S003D2046 still names Sol0052_Sol0052_S003D2046
    for lab in cfg["stations"]:
        by_key.setdefault(station_key(lab).upper(), []).append(lab)
    todo = []
    for lab in (list(labels) if labels else [k for k, v in cfg["stations"].items() if not v.get("experimental")][::-1]):   # 0.98.3 (Christian, 14 Nov 2026: "run in reverse order by default"): the config's last station first
        todo += [lab] if lab in cfg["stations"] else by_key.get(str(lab).upper(), [lab])      # 0.98.2: an "experimental"
        #                                                                                       station runs only when named
    if companions:                                              # 0.99.43: a Mastcam-Z version right after its Navcam station
        mine = {companions[t] for t in todo if t in companions}     # the versions whose Navcam station runs too
        out_: List[str] = []
        for lab in todo:
            if lab in mine:
                continue
            out_.append(lab)
            if lab in companions and companions[lab] in cfg["stations"]:
                out_.append(companions[lab])
        todo = list(dict.fromkeys(out_))
    bc = block_cameras or cfg.get("block_cameras")              # 0.99.43: the block's cloud from these families (NFR)
    unknown = [lab for lab in todo if lab not in cfg["stations"]]
    if unknown:
        raise KeyError(f"not in the config: {unknown} (there: {sorted(cfg['stations'])})")
    runs = []
    for lab in todo:
        st = cfg["stations"][lab]
        work = block_folder(st, root)
        vs = [v for v in (variants or st.get("variants") or cfg.get("variants") or VARIANTS)]
        frames = dict(st.get("frames") or {})
        ctx = [n if str(n).lower().endswith(".png") else f"{n}.png" for n in (st.get("context") or [])]
        fa = frame_args(frames)
        if ctx:                                     # 0.97.15: context frames (another place or sol) - in the dense step
            fa += ["--add-images", *ctx]            # for the terrain (0.97.19: rpp dense --add-images, so a station
            #                                         chosen by --scale or --sequence can have context frames too)
        fa = fa + [str(x) for x in (cfg.get("dense_args") or []) + (st.get("dense_args") or [])]
        near = st.get("nearby_m", cfg.get("nearby_m"))          # 0.97.0: the neighbors' frames constrain the ground
        bt = block_terrain(dict(st, terrain=terrain) if terrain else st, cfg)     # 0.99.43: --terrain for the run
        block = bt is not None                                       # 0.98.14 (Christian, 15 Nov 2026: "for rockytop use a
        #   model made from the full ncam block alignment"): the terrain from every Navcam frame of the block - one
        #   cloud (colmap/dense_block_N) shared by the block's stations; the panorama still from the station's frames
        if block:
            fa = frame_args(frames) + [str(x) for x in (cfg.get("dense_args") or []) + (st.get("dense_args") or [])] + ["--whole-block"]
            if bt[0]:
                fa += ["--block-name", bt[0], "--block-images", *bt[1]]
            elif bc:
                fa += ["--block-cameras", str(bc)]
            near = None
        elif near:
            fa += ["--nearby-m", f"{float(near):g}"]
        base = Path(out_root) / lab
        cmds: List[Dict[str, Any]] = []
        if "dense" in steps:
            cmds.append({"variant": "masked", "step": "dense",
                         "argv": ["dense", str(work), "--station", st["station"], *fa, "--prepare", "--stereo", "--fusion"]})
            # 0.97.0: no second fusion with the rover kept - the rover-shown mesh is the terrain of the rover-masked
            # cloud with the Perseverance model on it (rpp mesh --rover; --rover-cloud is the 0.96 way)
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
                if step == "ods" and ctx:
                    argv += ["--context-images", *ctx]
                if step == "ods" and st.get("ods_exclude"):   # 10 Nov: pairs left out of the panorama
                    argv += ["--exclude", *[str(x) for x in st["ods_exclude"]]]
                if step == "ods":                    # 0.97.0: one 16-to-8-bit scaling for both variants of a station
                    argv += ["--tone-levels", str(base / TONE_FILE)]
                    from .ods import ods_prefix          # 0.97.14: the files carry the station's code of the list
                    argv += ["--prefix", ods_prefix(lab, rover=v == "rover")]
                    from .nadir import block_name        # 0.97.7: the name for the mark at the nadir; 0.97.17: the
                    nm = mark_name(st)                   # block's name without its size word
                    if nm:
                        argv += ["--site-name", str(nm)]
                if step == "mesh" and block:                 # the block's cloud, not an older one of the station alone
                    argv += ["--points", str(Path(work) / "colmap" / block_workspace(bt[0], None if bt[0] else bc) / "fused.ply")]
                if step == "mesh" and (near or block):
                    argv += ["--keep-under-rover", f"{float(st.get('keep_under_rover_m', cfg.get('keep_under_rover_m', 0.25))):g}"]
                argv += [str(x) for x in (cfg.get(f"{step}_args") or [])]      # 0.96.1: for every station
                argv += [str(x) for x in st.get(f"{step}_args", [])]
                if step == "ods" and ods_width:      # after the config's arguments: the last --width counts
                    argv += ["--width", str(int(ods_width))]
                if step == "ods" and rover_surface and v == "rover":     # 0.98.2 (experimental): the panorama from the
                    argv += ["--eyes", "left", "right", "center"]        # pivot too, for rpp.ibr.rover_surface
                if step == "ods" and station_mesh:       # 0.99.26: each variant's part of the station mesh
                    argv += ["--station-mesh-only" if station_mesh == "only" else "--station-mesh"]
                cmds.append({"variant": v, "step": step, "argv": argv})
        runs.append({"label": lab, "work": str(work), "out": str(base), "note": st.get("note", ""), "variants": vs,
                     "commands": cmds})
    return runs


STEP_OUTPUT = {"mesh": "mesh.ply", "depthmaps": "maps_report.md", "ods": "ods.json"}
STATUS_FILE = "single_sites_status.json"


NOT_IN_CODE_KEY = ("single_sites.py", "web.py", "hf.py")     # the runner and the publishing (panos.json, Hugging Face)


def code_key() -> str:
    """A hash of the code the steps run (the ``rpp.ibr`` sources and the dense step's): a step is made again when it
    changes, not on every version number.  Not in it: this module and the publishing ones (:data:`NOT_IN_CODE_KEY`,
    0.99.36: a change to the viewer's list no longer makes the meshes and panoramas again)."""
    import hashlib
    h = hashlib.sha1()
    here = Path(__file__).resolve().parent
    for f in sorted(here.glob("*.py")) + [here.parent / "sfm" / "dense.py"]:
        if f.is_file() and f.name not in NOT_IN_CODE_KEY:
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
    """``python -m rpp <argv>`` with its output to the console and the log."""
    cmd = [sys.executable, "-m", "rpp", *argv]
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
                     f"colored {100 * c['coloured']:.0f} %")
        cells.append("yes" if any(out.glob("*360_3DTD.jpg")) else "no")
    except (OSError, ValueError, KeyError):
        cells += ["-", "-"]
    return "| " + " | ".join(cells) + " |"


def run_all(runs: List[Dict[str, Any]], out_root: PathLike, squares_only: bool = False, retone: bool = False,
            force: bool = False, after_station=None) -> int:
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
                    if step == "dense":                    # its own cache (rpp dense: stereo and fusion records)
                        rc = _run(c["argv"], fh)
                    else:
                        clouds = sorted((work_ / "colmap").glob(f"dense_{station_key(run['label'])}*/fused.ply")) if step == "mesh" else []
                        if step == "mesh" and "--points" in c["argv"]:      # 0.98.14: the block's cloud
                            clouds = [Path(c["argv"][c["argv"].index("--points") + 1])]
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
        if after_station is not None:                    # 0.98.5 (Christian, 14 Nov 2026: "Upload the ods sites as they
            try:                                         # finish processing"): published and uploaded station by station
                _status(out_root, started=t_run, station=run["label"], step="upload", since=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        done=done, stations=[r["label"] for r in runs])
                done.append(f"{run['label']} upload: {after_station(run)}")
            except Exception as e:                       # noqa: BLE001 - the next station still runs
                done.append(f"{run['label']} upload: failed ({e})")
                print(f"######## {run['label']}: publishing failed ({e})", flush=True)
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
    p = sub.add_parser("single-sites", help="0.96: the single-station test sites of rpp/data/single_sites.json - dense, "
                       "mesh, depthmaps, ods, rover masked and shown (rpp.ibr.single_sites)")
    p.add_argument("labels", nargs="*", help="stations of the config (default: all but the experimental ones)")
    p.add_argument("--all", action="store_true", help="0.99.3: every station of the config, the experimental ones too "
                   "(the Mastcam-Z 34 mm station), the last one first")
    p.add_argument("--config", default=None, help="the config file (default rpp/data/single_sites.json)")
    p.add_argument("--root", default=None, help="the blocks' root (default RPP_SCAPES or D:\\rover_blocks)")
    p.add_argument("--out", default=None, help="the output root (default <root>/<config 'out'>)")
    p.add_argument("--steps", nargs="+", default=list(STEPS), choices=STEPS)
    p.add_argument("--variants", nargs="+", default=None, choices=VARIANTS)
    p.add_argument("--width", type=int, default=None, metavar="PX",
                   help="0.97.0: the panorama's width for this run (default: the config's, else rpp ods' own - 0.99.40: 16384 for "
                        "full-resolution Navcam and Mastcam-Z stations, 8192 for binned Navcam; 8192 for 8K) - give "
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
    p.add_argument("--publish-only", action="store_true", help="with --publish or --upload: no steps, publish what --out holds")
    p.add_argument("--no-upload", action="store_true",
                   help="0.97.24: no upload, also when --upload is given (run_single_site_ibr.bat gives it)")
    p.add_argument("--upload", action="store_true",
                   help="0.97.21: after the run, publish (into docs/ods of this checkout unless --publish names a folder), "
                        "commit the panoramas there and push the branch to origin - the code's commits go with it")
    p.add_argument("--host", choices=("hf", "github"), default="hf",
                   help="0.99.3: where --upload puts the pictures - hf (default; Christian, 16 Nov 2026: \"v0p99 should upload "
                        "rpv products to hugging face\"): the dataset RPP_HF_DATASET (ctate7163/rpp-rpv), one commit per "
                        "station, panos.json on GitHub pointing at it (rpp.ibr.hf); github: the pictures committed into the "
                        "viewer's checkout as before 0.99.3")
    p.add_argument("--no-hf-record", dest="hf_record", action="store_false", default=True,
                   help="0.99.3: with --host hf, the viewer's pictures only (default: also each station's record - the 16-bit "
                        "stacked panoramas, the pivot's 16-bit depth, ods.json, station.json; about 0.35 GB a station)")
    p.add_argument("--publish-width", type=int, default=8192, metavar="PX", help="wider images are reduced to this (default 8192)")
    p.add_argument("--publish-views", nargs="+", default=None, metavar="VIEW",
                   help="standard, 'raw color', 'enhanced local', 'enhanced tone', 'dcs color', 'normal map', 'distance "
                        "map', enhanced, ... (default: rpp.ibr.web.DEFAULT_VIEWS - 0.99.37: all of these but enhanced; the "
                        "page's names 'standard color' and 'enhanced color' and the old normal and 'range log' are read too)")
    p.add_argument("--no-rover-surface", dest="rover_surface", action="store_false",
                   help="0.98.8: without the rover's 3D surface (and the panorama from the pivot it is made from)")
    p.add_argument("--rover-surface", action="store_true", default=True,
                   help="0.98.2, experimental: also render the with-rover panorama from the pivot and make the rover a "
                        "textured surface from it (<prefix>_rover_surface.glb; the viewer lays it on the terrain-only view)")
    p.add_argument("--no-station-mesh", dest="station_mesh", action="store_false", default=True,
                   help="0.99.26: without the station mesh (the textured terrain within 10 m and the rover: the viewer's "
                        "fourth mode, 3D terrain and rover; rpp.ibr.station_mesh)")
    p.add_argument("--station-mesh-only", action="store_true",
                   help="0.99.26: only the station meshes of stations whose panoramas exist (rpp ods --station-mesh-only "
                        "for each variant, then the .glb; with --publish / --upload the fourth mode is published)")
    p.add_argument("--block-cameras", default=None, metavar="FAMILIES",
                   help="0.99.43: the block terrain from the frames of these camera families (NFR: every engineering "
                        "camera of the block, every size, fused into one cloud - rpp dense --block-cameras; default: the "
                        "config's 'block_cameras', else Navcam alone)")
    p.add_argument("--terrain", default=None, choices=("block", "own"),
                   help="0.99.43: the terrain of every station of this run, whatever the config says - block: the whole "
                        "block's cloud (also for the stations with a named terrain set); own: the station's own frames")
    p.add_argument("--no-zcam", dest="zcam_companions", action="store_false", default=True,
                   help="0.99.43: without the Mastcam-Z 34 mm versions of the stations that have a Z34 360 (default, in a "
                        "run of the whole list - no station named, or --all: each is made right after its Navcam station)")
    p.add_argument("--wait-for-blocks", type=float, nargs="?", const=36.0, default=None, metavar="HOURS",
                   help="0.99.44: after the stations whose block is aligned, wait for the others' blocks and run them as "
                        "they come in (looked at every --poll-min minutes); stops when no station is left, or when no "
                        "block has come in for HOURS (default 36)")
    p.add_argument("--poll-min", type=float, default=15.0, help="--wait-for-blocks: minutes between looks (default 15)")
    p.add_argument("--any-block", action="store_true",
                   help="0.99.28: run stations whose block has not finished its alignment too (default: skipped)")
    p.add_argument("--list", action="store_true", help="the stations of the config")
    p.add_argument("--dry-run", action="store_true", help="the commands, not run")
    p.add_argument("--install", metavar="DIR", default=None, help=f"write {BAT_NAME} (and make_square.bat) into DIR and stop")


def cmd_single_sites(a: argparse.Namespace) -> int:
    """0.99.44 (Christian, 19 Nov 2026: "run only on aligned blocks. Skip a bock if it doesn't exist and come back to it
    later. When there are no more stations to run, then stop"): with ``--wait-for-blocks`` the run is repeated for the
    stations whose block was not aligned yet (and the Mastcam-Z versions whose block was not), every ``--poll-min``
    minutes, until none is left - or no block has come in for ``--wait-for-blocks`` hours."""
    wait_h = getattr(a, "wait_for_blocks", None)
    if wait_h is None or a.dry_run or getattr(a, "any_block", False):
        return _single_sites_pass(a)
    import time
    done: set = set()
    rc, last = 0, time.time()
    while True:
        a._skip = set(done)
        r = _single_sites_pass(a)
        rc = rc or r
        ran, late = list(getattr(a, "_ran", [])), list(getattr(a, "_late", []))
        done |= set(ran)
        if ran:
            last = time.time()
        if not late:
            print(f"single-sites: no station left to run ({len(done)} done)", flush=True)
            return rc
        if time.time() - last > float(wait_h) * 3600.0:
            print(f"single-sites: no block came in for {float(wait_h):g} h - stopped with {len(late)} left: {', '.join(late)}", flush=True)
            return rc
        print(f"single-sites: {len(done)} done, waiting for the blocks of {len(late)}: {', '.join(late)}", flush=True)
        time.sleep(max(1.0, float(getattr(a, "poll_min", 15.0))) * 60.0)


def _single_sites_pass(a: argparse.Namespace) -> int:
    if a.install:
        print(f"written: {install(a.install)}")
        return 0
    cfg = load_config(a.config)
    root = Path(a.root or rpp_env.get("RPP_SCAPES") or r"D:\rover_blocks")
    out_root = Path(a.out) if a.out else root / cfg.get("out", "single_sites")
    if a.list:
        for lab, st in cfg["stations"].items():
            print(f"{lab}  {st['station']:>9}  {st.get('site') or st.get('work')}  {st.get('note', '')}")
        return 0
    given = list(a.labels) if a.labels else None
    if getattr(a, "all", False) and not a.labels:                # 0.99.3: the experimental stations too
        a.labels = list(cfg["stations"])[::-1]
    if getattr(a, "station_mesh_only", False):                   # 0.99.26
        a.steps, a.station_mesh = ["ods"], True
    pairs = None
    # only in a run of the whole list (no station named, or --all): a run of named stations is those stations
    if getattr(a, "zcam_companions", True) and (given is None or getattr(a, "all", False)) \
            and not (a.squares_only or a.retone or a.publish_only):
        try:                                                     # 0.99.43: the Z34 versions, where a block has a Z34 360
            pairs = zcam_companions(cfg, root)
            new = [z for z in pairs.values() if cfg["stations"][z].get("derived_from")]
            if new:
                print(f"single-sites: {len(new)} Mastcam-Z 34 mm station(s) from the blocks: {', '.join(new)}")
        except Exception as e:                                   # noqa: BLE001 - the Navcam stations run without them
            print(f"single-sites: no Mastcam-Z companions ({type(e).__name__}: {e})")
    runs = plan(cfg, root, out_root, a.labels, a.steps, a.variants, ods_width=a.width,
                rover_surface=getattr(a, "rover_surface", False),
                station_mesh="only" if getattr(a, "station_mesh_only", False) else getattr(a, "station_mesh", False),
                companions=pairs, block_cameras=getattr(a, "block_cameras", None), terrain=getattr(a, "terrain", None))
    if not getattr(a, "any_block", False):                      # 0.99.28 (Christian, 18 Nov 2026: "The next ods should use
        late = [r for r in runs if not block_aligned(r["work"])]   # the post freeze blocks. Skip an ods site if the block
        for r in late:                                          # has not aligned yet")
            print(f"single-sites: {r['label']} skipped - its block {r['work']} has not finished its alignment (no colmap\\run_done.json)")
        runs = [r for r in runs if r not in late]
        if late:
            print(f"single-sites: {len(runs)} station(s) to run, {len(late)} skipped for blocks not aligned yet")
        a._late = [r["label"] for r in late] + [f"{k} (Z34?)" for k in (cfg.get("_zcam_pending") or {}) if pairs is not None]
    skip = getattr(a, "_skip", None)
    if skip is not None:                                         # 0.99.44: a later pass of --wait-for-blocks
        runs = [r for r in runs if r["label"] not in skip]
        a._ran = [r["label"] for r in runs]
        if not runs:
            return 0
    if a.dry_run:
        for r in runs:
            print(f"{r['label']}: {r['work']} -> {r['out']}")
            for c in r["commands"]:
                print(f"   [{c['variant']}] python -m rpp " + " ".join(c["argv"]))
        return 0
    if getattr(a, "no_upload", False) or a.dry_run or a.squares_only:   # 0.97.24: the batch file asks for the upload by default
        a.upload = False
    if getattr(a, "upload", False) and not a.publish:    # 0.97.21: the checkout's own viewer folder;
        from .web import viewer_dir                      # 0.98.29: the RPV checkout (RPP_VIEWER_DIR, or ..\\RPV)
        a.publish = str(viewer_dir())
        print(f"single-sites: publishing to {a.publish}")
    st_cfg = cfg.get("stations") or {}

    def finish(labels, quiet=False):
        """The rover surfaces (--rover-surface), the publishing and the upload of these stations; a line saying what happened."""
        if getattr(a, "rover_surface", False) and not a.squares_only:      # experimental: never fails the run
            from . import rover_surface as _rs
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            for r in runs:
                if r["label"] in labels:
                    try:
                        _rs.from_folder(Path(r["out"]) / "with_rover")
                    except Exception as e:               # noqa: BLE001
                        print(f"rover surface of {r['label']}: failed ({e})")
        if getattr(a, "station_mesh", False) and not a.squares_only:      # 0.99.26: never fails the run
            from . import station_mesh as _sm
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            for r in runs:
                if r["label"] in labels:
                    try:
                        _sm.assemble(Path(r["out"]))
                        _sm.assemble_scape(Path(r["out"]))       # 0.99.32: the fifth mode
                    except Exception as e:               # noqa: BLE001
                        print(f"station mesh of {r['label']}: failed ({e})")
        if not a.publish:
            return "not published"
        from . import web
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        titles = {k: {"title": v.get("title") or viewer_name(v, k),   # 0.98.19: the page puts "Sol 0xxx" before it
                      "site_name": viewer_name(v, k),     # 0.99.36: before the mark's name (made with the panorama)
                      "sols": viewer_sols(v, k),          # 0.99.36: a Mastcam-Z station's first sol only
                      "note": v.get("web_note"), "yaw_deg": v.get("yaw_deg")} for k, v in st_cfg.items()}
        use_hf = bool(getattr(a, "upload", False)) and getattr(a, "host", "hf") == "hf"
        before = None
        if use_hf:                                       # 0.99.3: the pictures to Hugging Face, panos.json to GitHub
            from . import hf
            try:
                before = json.loads((Path(a.publish) / "panos.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                before = {}
        doc = web.publish(out_root, a.publish, stations=list(labels), views=[web.OLD_VIEW_NAMES.get(v, v) for v in (a.publish_views or web.DEFAULT_VIEWS)], width=a.publish_width, titles=titles,
                          render_map=True, files_dir=hf.stage_dir(out_root) if use_hf else None)
        if not web.install_viewer(a.publish):
            print("the viewer page (docs/ods/index.html, vendor/) is not in this installation: copy it from the repository")
        msg = f"published: {len(doc['panos'])} stations in {Path(a.publish) / 'panos.json'}"
        ok = True
        if getattr(a, "upload", False):
            parts = []
            if use_hf:
                res = hf.publish(a.publish, doc, list(labels), out_root, previous=before, record=getattr(a, "hf_record", True))
                ok = bool(res["ok"])
                parts.append(res["message"])
            up = web.upload(a.publish, list(labels))
            ok = ok and bool(up["ok"])
            parts.append(up["message"])
            msg = ("uploaded: " if ok else "NOT (all) uploaded: ") + " | ".join(parts)
        if not quiet:
            print(msg, flush=True)
        finish.failed = getattr(finish, "failed", False) or not ok
        return msg

    each = (lambda run: finish([run["label"]])) if (a.publish and not a.publish_only and not a.squares_only) else None
    rc = 0 if (a.publish and a.publish_only) else run_all(runs, out_root, squares_only=a.squares_only, retone=a.retone,
                                                          force=getattr(a, "force", False), after_station=each)
    if each is None or getattr(finish, "failed", False):         # all at once, or again what an upload left behind
        finish([r["label"] for r in runs])
    rc = rc or (1 if getattr(finish, "failed", False) and getattr(a, "upload", False) else 0)
    return rc
