"""
The stereo 360 products of the station runs as a folder a static web page can show (0.97.3, prototype).

``publish(out_root, web_dir)`` copies each station's top-down stereo JPEGs (``<name>_360_3DTD.jpg``: the left eye on
top of the right, :mod:`mppp.ibr.ods`) into ``web_dir/panos/`` - reduced to ``width`` when wider - and writes
``web_dir/panos.json``, the list the viewer page reads (``docs/ods/index.html`` of the repository: three.js, WebXR).
``web_dir`` holding ``index.html``, ``vendor/``, ``panos.json`` and ``panos/`` is all a host needs (GitHub Pages, any
static server).  ``test_pano`` draws a labelled pair for checking the viewer's orientation and eye order.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
log = logging.getLogger(__name__)

STACK = "_360_3DTD.jpg"
VIEWS = {"standard": "", "enhanced": "_enh", "enhanced local": "_enhlocal",      # view name -> file name part
         "range log": "_range_log", "range linear": "_range_linear", "normal": "_normal"}   # 0.97.20: the model's
VARIANTS = (("rover", "with_rover", "with the rover"), ("masked", "", "terrain only"))   # key, sub-folder, label
DEFAULT_VIEWS = ("standard",)        # 0.97.18 (Christian, 13 Nov 2026: "don't upload the enhanced version to github");
#                                      ("standard", "enhanced") before; mppp single-sites --publish-views brings it back
DEFAULT_WIDTH = 8192               # the square's side: 8192 x 8192 holds two 8192 x 4096 eyes (Quest 3: fine)


def station_files(folder: PathLike) -> Dict[str, Path]:
    """{view name: file} of the top-down stereo JPEGs in one variant's folder (``<prefix>[_enh|_enhlocal]_360_3DTD.jpg``)."""
    out: Dict[str, Path] = {}
    for f in sorted(Path(folder).glob("*" + STACK)):
        stem = f.name[:-len(STACK)]
        view = next((v for v, part in sorted(VIEWS.items(), key=lambda kv: -len(kv[1])) if part and stem.endswith(part)), "standard")
        out.setdefault(view, f)
    return out


MARK_SCALE = 1.30                  # the viewer's ground disc is this much wider than the mark in the panorama.  1.15
#                                    until 0.98.3 (Christian, 14 Nov 2026: "The blue disk should be a little larger")


def ground_mark(folder: Path, web: Path) -> Optional[Dict[str, Any]]:
    """0.97.24 (Christian, 14 Nov 2026: "in the VR viewer, make that a 3d surface on the ground ... a circle a little
    larger than how it appears in the ods at the level of the ground"): the nadir mark of one variant's folder for the
    page - its picture from above copied into ``web/panos`` and {"file", "depth_m" below the eyes, "radius_m" of the
    disc the page draws (MARK_SCALE x the mark's own, the picture stretched over it), "mark_radius_m"}.  None when the
    panorama run wrote no texture (before 0.97.24, or --no-nadir-mark)."""
    for j in sorted(folder.glob("*ods.json")):
        try:
            m = json.loads(j.read_text(encoding="utf-8")).get("nadir_mark") or {}
        except (OSError, ValueError):
            continue
        tex = folder / str(m.get("texture") or "")
        if m.get("texture") and tex.is_file() and m.get("radius_m") and m.get("depth_below_pivot_m"):
            dst = web / "panos" / tex.name
            if not (dst.is_file() and dst.stat().st_mtime >= tex.stat().st_mtime):
                shutil.copyfile(tex, dst)
            return {"file": "panos/" + tex.name, "depth_m": round(float(m["depth_below_pivot_m"]), 4),
                    "radius_m": round(MARK_SCALE * float(m["radius_m"]), 4), "mark_radius_m": round(float(m["radius_m"]), 4),
                    "radius_deg": float(m.get("radius_deg") or 10.0),
                    "text_centre_azimuth_deg": round(float(m.get("text_centre_azimuth_deg") or 0.0), 2)}
    return None


def rover_surface(folder: Path, web: Path) -> Optional[Dict[str, Any]]:
    """0.98.2 (experimental): the rover's textured surface of a ``with_rover`` folder (``*_rover_surface.glb``,
    :mod:`mppp.ibr.rover_surface`) copied into ``web/panos`` - {"file", "triangles"} for the page, which shows it on
    the terrain-only panorama.  None when the folder has none."""
    glb = next(iter(sorted(folder.glob("*_rover_surface.glb"))), None) if folder.is_dir() else None
    if glb is None:
        return None
    dst = web / "panos" / glb.name
    if not (dst.is_file() and dst.stat().st_mtime >= glb.stat().st_mtime):
        shutil.copy2(glb, dst)
    try:
        n = json.loads(glb.with_suffix(".json").read_text(encoding="utf-8")).get("triangles")
    except (OSError, ValueError):
        n = None
    return {"file": "panos/" + glb.name, "triangles": n, "experimental": True}


WELCOME_ID = "welcome"
WELCOME_STATION = "Sol0052_S003D2046"      # Van Zyl Overlook
WELCOME_WIDTH = 4096
WELCOME_GREY = (0.55, 0.20)                # the welcome picture: this x the grey of the station + this x white (washed out)
OLD_TEST_ID = "orientation_test"           # the labelled test pair of before 0.98.4: taken off the list


def welcome_pano(web: Path, station_entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """0.98.4 (Christian, 14 Nov 2026: "Remove the fake ods. In its stead put van zyl grayed out with north south and
    east in the sky in white letters ... MPPP stereo viewer for Mars 2020 Perseverance Rover in bold welcoming
    letters a few meters from the viewer"): the list's first entry - Van Zyl Overlook's standard panorama in washed-out
    grey, reduced to WELCOME_WIDTH.  The page draws the words over it (an entry with "welcome": true).  None when
    Van Zyl is not published."""
    import cv2
    if not station_entry:
        return None
    var = next((v for v in station_entry.get("variants", []) if v.get("views", {}).get("standard")), None)
    if var is None:
        return None
    src = web / var["views"]["standard"]
    if not src.is_file():
        return None
    dst = web / "panos" / f"{WELCOME_ID}{STACK}"
    if not (dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime):
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.shape[1] > WELCOME_WIDTH:
            img = cv2.resize(img, (WELCOME_WIDTH, WELCOME_WIDTH), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = np.clip(WELCOME_GREY[0] * g + WELCOME_GREY[1] * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dst), cv2.merge([g, g, g]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        try:
            from .xmp import tag_stereo
            tag_stereo(dst, extra={"Station": WELCOME_STATION})
        except Exception:                                    # noqa: BLE001
            pass
    return {"id": WELCOME_ID, "title": "Welcome", "welcome": True, "site_name": "MPPP stereo viewer",
            "yaw_deg": station_entry.get("yaw_deg"),
            "variants": [{"key": "welcome", "label": "welcome", "views": {"standard": "panos/" + dst.name}}]}


def station_info(folder: Path) -> Dict[str, Any]:
    """{"site_name", "sols"} of a variant's folder, from its panorama record's mark (the page's A button shows them)."""
    for j in sorted(folder.glob("*ods.json")):
        try:
            m = json.loads(j.read_text(encoding="utf-8")).get("nadir_mark") or {}
        except (OSError, ValueError):
            continue
        out: Dict[str, Any] = {}
        for line in (m.get("lines") or []):
            g = re.match(r"^Sols? ([0-9][0-9,\s\-]*?)(?: - (.+))?$", str(line).strip())
            if not g:
                continue
            nums = [int(x) for x in re.findall(r"\d+", g.group(1))]
            if "-" in g.group(1) and len(nums) == 2:
                nums = list(range(nums[0], nums[1] + 1))
            out["sols"] = sorted(set(nums))
            if g.group(2):
                out["site_name"] = g.group(2).strip()
            break
        return out
    return {}


def _copy(src: Path, dst: Path, width: int, quality: int) -> Dict[str, Any]:
    """``src`` to ``dst``, reduced to ``width`` (area filter) when wider; copied byte for byte otherwise.  Skipped when
    ``dst`` is newer than ``src``."""
    import cv2
    if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime:
        return {"file": dst.name, "kept": True}
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"{src}: not readable")
    if im.shape[1] > width:
        im = cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    else:
        shutil.copyfile(src, dst)
    return {"file": dst.name, "width": int(min(im.shape[1], width))}


def publish(out_root: PathLike, web_dir: PathLike, stations: Optional[Sequence[str]] = None,
            views: Sequence[str] = DEFAULT_VIEWS, width: int = DEFAULT_WIDTH, quality: int = 88,
            titles: Optional[Dict[str, Dict[str, Any]]] = None, title: str = "MPPP stereo panoramas") -> Dict[str, Any]:
    """Every station folder of ``out_root`` (the output of ``mppp single-sites``) that has a stereo JPEG, into
    ``web_dir``.  ``titles``: {station label: {"title", "note", "yaw_deg"}} (``yaw_deg``: the azimuth, degrees east
    of north, straight ahead when the page opens).  Stations already in ``panos.json`` and not in ``out_root`` stay
    listed.  Returns the list written."""
    from .. import __version__
    out_root, web = Path(out_root), Path(web_dir)
    (web / "panos").mkdir(parents=True, exist_ok=True)
    index = web / "panos.json"
    try:
        old = {p["id"]: p for p in json.loads(index.read_text(encoding="utf-8")).get("panos", [])}
    except (OSError, ValueError):
        old = {}
    bad = [v for v in views if v not in VIEWS]
    if bad:
        raise ValueError(f"views {bad}: one of {list(VIEWS)}")
    n = 0
    for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
        if stations and d.name not in stations:
            continue
        entry: Dict[str, Any] = {"id": d.name, "title": d.name, "variants": [], "mppp": __version__,
                                 "published": time.strftime("%Y-%m-%d")}
        entry.update({k: v for k, v in ((titles or {}).get(d.name) or {}).items() if v is not None})
        for key, sub, label in VARIANTS:
            files = station_files(d / sub) if (d / sub).is_dir() else {}
            got = {}
            for v in views:
                if v in files:
                    got[v] = "panos/" + _copy(files[v], web / "panos" / files[v].name, width, quality)["file"]
                    try:                                         # a reduced copy was encoded again: its XMP with it
                        from .xmp import read_xmp, tag_stereo
                        if read_xmp(web / "panos" / files[v].name) is None:
                            tag_stereo(web / "panos" / files[v].name, extra={"Station": d.name})
                    except Exception:                            # noqa: BLE001
                        pass
                    n += 1
            if got:
                var = {"key": key, "label": label, "views": got}
                gm = ground_mark(d / sub, web)
                if gm:
                    var["ground_mark"] = gm
                if key == "masked":                      # 0.98.2 (experimental): the rover as a surface on the terrain
                    rs = rover_surface(d / "with_rover", web)
                    if rs:
                        var["rover_surface"] = rs
                entry["variants"].append(var)
        # 0.98.4 (Christian, 14 Nov 2026: "always at ground level even when the rover is displayed"): with the rover in
        # the picture the mark was put at the deck's depth under the mast; the page's disc lies on the terrain in both
        # variants - the terrain-only variant's depth, the disc as wide there as it looks in the panorama
        ground = next((v["ground_mark"] for v in entry["variants"] if v["key"] == "masked" and v.get("ground_mark")), None)
        for v in entry["variants"]:
            gm = v.get("ground_mark")
            if ground and gm and v["key"] != "masked" and ground["depth_m"] > gm["depth_m"]:
                r = float(ground["depth_m"]) * float(np.tan(np.radians(gm.get("radius_deg", 10.0))))
                gm.update(depth_m=ground["depth_m"], mark_radius_m=round(r, 4), radius_m=round(MARK_SCALE * r, 4), at="the terrain's level")
        for key, sub, label in VARIANTS:                         # the A button's words
            info = station_info(d / sub) if (d / sub).is_dir() else {}
            if info:
                entry.update({k: v for k, v in info.items() if k not in entry})
                break
        if entry["variants"]:
            old[d.name] = entry
            log.info("publish: %s (%s)", d.name, ", ".join(f"{v['key']}: {len(v['views'])}" for v in entry["variants"]))
    if old.pop(OLD_TEST_ID, None) is not None:                   # 0.98.4: "Remove the fake ods"
        for f in (web / "panos").glob(OLD_TEST_ID + "*"):
            f.unlink()
    old.pop(WELCOME_ID, None)
    first = welcome_pano(web, old.get(WELCOME_STATION))
    doc = {"title": title, "panos": ([first] if first else []) + [old[k] for k in sorted(old)]}
    index.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log.info("publish: %d stations listed in %s (%d images from this run)", len(doc["panos"]), index, n)
    return doc


def install_viewer(web_dir: PathLike, source: Optional[PathLike] = None) -> bool:
    """The page (``index.html`` and ``vendor/``) from the checkout's ``docs/ods`` into ``web_dir`` when that is another
    folder.  False when the checkout has no ``docs/ods`` (an installed package: take the page from the repository)."""
    src = Path(source) if source else Path(__file__).resolve().parents[3] / "docs" / "ods"
    web = Path(web_dir)
    if not (src / "index.html").is_file():
        return False
    if src.resolve() == web.resolve():
        return True
    web.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "index.html", web / "index.html")
    shutil.copytree(src / "vendor", web / "vendor", dirs_exist_ok=True)
    return True


def label_pano(path: PathLike, width: int = 2048) -> Path:
    """A labelled stereo pair for checking a viewer: compass letters every 45 degrees on the horizon (N at the left
    edge, azimuth growing to the right), elevation lines every 30 degrees, "UP" / "DOWN" at the poles, and the eye's
    name ("LEFT EYE" in blue on top, "RIGHT EYE" in red below).  The letters are 3 px apart between the eyes, so
    they float in front of the lines when the eyes are the right way round and sink behind them when swapped."""
    import cv2
    h = width // 2
    eyes = []
    for eye, colour, shift in (("LEFT EYE", (255, 170, 90), 3), ("RIGHT EYE", (90, 140, 255), -3)):
        im = np.full((h, width, 3), 28, np.uint8)
        for el in range(-60, 61, 30):
            y = int(round((90 - el) / 180 * h))
            cv2.line(im, (0, y), (width, y), (200, 200, 200) if el == 0 else (90, 90, 90), 2 if el == 0 else 1)
            cv2.putText(im, f"{el:+d}", (8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        for k, name in enumerate(("N", "NE", "E", "SE", "S", "SW", "W", "NW")):
            x = int(round(k / 8 * width))
            cv2.line(im, (x, 0), (x, h), (90, 90, 90), 1)
            for xx in (x, x + width):                                   # N again at the right edge (the seam)
                cv2.putText(im, name, (xx - 14 * len(name) + shift, h // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(im, eye, (xx + 10 + shift, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        for x in range(width // 16, width, width // 4):
            cv2.putText(im, "UP", (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(im, "DOWN", (x, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        eyes.append(im)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(eyes), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


GIT_TIMEOUT_S = 900                # one git command (a push of several 10 MB panoramas on a slow line)
PUSH_TRIES = 3                     # a push that fails for the network is tried again
PUSH_WAIT_S = 30
PENDING_NAME = "upload_pending.json"


def _git(repo: Path, *args: str, timeout: Optional[int] = None) -> Tuple[int, str]:
    """git in ``repo``: (exit code, output).  It never waits for a password or a merge message at a prompt (a run
    with nobody at the keyboard would hang there): GIT_TERMINAL_PROMPT=0, and a time limit (exit 124)."""
    import os
    import subprocess
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", GIT_MERGE_AUTOEDIT="no")
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env,
                           timeout=timeout or GIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return 124, f"git {args[0]}: no end after {timeout or GIT_TIMEOUT_S} s"
    except OSError as e:
        return 127, f"git: {e}"
    return r.returncode, (r.stdout + r.stderr).strip()


def upload(web_dir: PathLike, stations: Sequence[str] = (), remote: str = "origin", tries: int = PUSH_TRIES,
           wait_s: float = PUSH_WAIT_S) -> Dict[str, Any]:
    """0.97.21 (Christian, 13 Nov 2026: "I'd also like the option to upload the ods after the one bat file
    completes"); 0.97.24 (14 Nov: "The ibr bat script should by default upload the latest ods and other files to
    Github when it is done. how could this be made reliable?").

    ``web_dir/panos`` and ``web_dir/panos.json`` are committed in the repository that holds ``web_dir`` (those paths
    only; nothing else that is changed or staged goes into the commit), then the branch is pushed to ``remote`` -
    with it every commit not yet there, so the code goes up too.  What makes it dependable:
    - the panoramas are committed before the network is touched, so a push that fails loses nothing: the next
      upload (any later run, or ``mppp single-sites --publish-only --upload``) sends that commit as well;
    - a failed push is tried ``tries`` times ``wait_s`` apart; one the remote refuses because it has commits this
      checkout lacks is followed by a ``git pull --no-rebase`` (a merge that conflicts is undone) and pushed again;
    - git never waits at a prompt and every command has a time limit, so an unattended run cannot hang here;
    - after the push the remote's branch is read back and compared with the local commit;
    - what did not go through is written to ``web_dir/upload_pending.json`` (removed by the next upload that works)
      and returned, so the run's summary and exit code show it.
    Nothing is forced and no history is rewritten.  Returns {"ok", "committed", "pushed", "verified", "message"}."""
    from .. import __version__
    web = Path(web_dir).resolve()
    pending = web / PENDING_NAME

    def done(ok: bool, committed: bool, pushed: bool, message: str, verified: bool = False) -> Dict[str, Any]:
        out = {"ok": ok, "committed": committed, "pushed": pushed, "verified": verified, "message": message}
        try:
            if ok:
                pending.unlink(missing_ok=True)
            else:
                pending.write_text(json.dumps(dict(out, when=time.strftime("%Y-%m-%dT%H:%M:%S"), stations=list(stations)), indent=1),
                                   encoding="utf-8")
        except OSError:
            pass
        return out
    rc, top = _git(web, "rev-parse", "--show-toplevel")
    if rc:
        return done(False, False, False, f"{web} is not in a git repository: {top}")
    repo = Path(top)
    if not _git(repo, "rev-parse", "-q", "--verify", "MERGE_HEAD")[0]:      # 0.97.22: an unfinished merge - git refuses the commit
        return done(False, False, False, f"{repo} has an unfinished merge: finish it (git status, git add the listed files, git commit "
                    "--no-edit) or undo it (git merge --abort), then run this again")
    paths = [str(web / "panos"), str(web / "panos.json")]
    rc, out = _git(repo, "add", "--", *paths)
    if rc:
        return done(False, False, False, f"git add failed: {out}")
    committed = bool(_git(repo, "diff", "--cached", "--quiet", "--", *paths)[0])
    if committed:
        what = ", ".join(stations) if 0 < len(stations) <= 4 else f"{len(stations)} stations" if stations else "update"
        rc, out = _git(repo, "commit", "-m", f"Viewer: panoramas {what} (MPPP {__version__})", "--", *paths)
        if rc:
            return done(False, False, False, f"git commit failed: {out}")
    rc, branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if rc or branch == "HEAD":
        return done(False, committed, False, f"{repo} is not on a branch (detached HEAD): nothing pushed; the commit is kept locally")
    out = ""
    pulled = False
    for k in range(max(1, int(tries))):
        rc, out = _git(repo, "push", remote, f"HEAD:{branch}")
        if not rc:
            break
        refused = any(w in out for w in ("[rejected]", "non-fast-forward", "fetch first"))
        if refused and not pulled:                       # the remote is ahead: merge it in once, then again
            pulled = True
            rc2, out2 = _git(repo, "pull", "--no-rebase", "--no-edit", remote, branch)
            if rc2:
                _git(repo, "merge", "--abort")
                return done(False, committed, False, "push refused and the pull did not merge cleanly (undone; the commit is kept "
                            f"locally):\n{out}\n{out2}")
            continue
        if k + 1 < max(1, int(tries)):
            log.warning("upload: push failed (try %d of %d), again in %.0f s: %s", k + 1, tries, wait_s, out.splitlines()[-1] if out else "")
            time.sleep(max(0.0, float(wait_s)))
    else:
        rc = rc or 1
    if rc:
        return done(False, committed, False, f"git push failed after {tries} tries (the commit is kept locally and goes up with the "
                    f"next upload):\n{out}")
    local = _git(repo, "rev-parse", "HEAD")[1]
    rc, ls = _git(repo, "ls-remote", remote, f"refs/heads/{branch}", timeout=120)
    verified = (not rc) and ls.split()[:1] == [local]
    return done(True, committed, True, f"{'panoramas committed and ' if committed else 'no new panorama; '}{branch} pushed to {remote}"
                + (f" (read back: {local[:8]})" if verified else " (the remote could not be read back to check)"), verified)
