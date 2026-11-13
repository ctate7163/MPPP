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
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]
log = logging.getLogger(__name__)

STACK = "_360_3DTD.jpg"
VIEWS = {"standard": "", "enhanced": "_enh", "enhanced local": "_enhlocal"}      # view name -> file name part
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
                entry["variants"].append({"key": key, "label": label, "views": got})
        if entry["variants"]:
            old[d.name] = entry
            log.info("publish: %s (%s)", d.name, ", ".join(f"{v['key']}: {len(v['views'])}" for v in entry["variants"]))
    doc = {"title": title, "panos": [old[k] for k in sorted(old)]}
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
