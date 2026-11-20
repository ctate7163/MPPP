"""
The stereo 360 products of the station runs as a folder a static web page can show (0.97.3, prototype).

``publish(out_root, web_dir)`` copies each station's top-down stereo JPEGs (``<name>_360_3DTD.jpg``: the left eye on
top of the right, :mod:`rpp.ibr.ods`) into ``web_dir/panos/`` - reduced to ``width`` when wider - and writes
``web_dir/panos.json``, the list the viewer page reads (``docs/ods/index.html`` of the repository: three.js, WebXR).
``web_dir`` holding ``index.html``, ``vendor/``, ``panos.json`` and ``panos/`` is all a host needs (GitHub Pages, any
static server).  ``test_pano`` draws a labeled pair for checking the viewer's orientation and eye order.
"""
from __future__ import annotations

import json
import os
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from .. import env as rpp_env

PathLike = Union[str, Path]
log = logging.getLogger(__name__)

STACK = "_360_3DTD.jpg"
VIEWS = {"standard": "", "enhanced": "_enh", "enhanced local": "_enhlocal",      # view name -> file name part
         "raw color": "_raw", "enhanced tone": "_enhtone", "dcs color": "_dcs",        # 0.99.37 (rpp.ibr.color_products)
         "distance map": "_range_log", "distance map linear": "_range_linear", "normal map": "_normal"}   # 0.97.20: the model's
# 0.98.30 (Christian, 16 Nov 2026: 'Change "normal" to "normal map" and "range log" to "distance map"'): the names the
# viewer shows; panos.json entries published before keep their old names, which the page shows by the new ones
OLD_VIEW_NAMES = {"normal": "normal map", "range log": "distance map", "range linear": "distance map linear",
                  "standard color": "standard", "enhanced color": "enhanced local"}     # 0.99.37: the page's names
# 0.99.37 (Christian, 19 Nov 2026: "Replace the enhanced local with just enhanced"; the views "standard color, raw color,
# enhanced color, enhanced tone, dcs color, normal map, distance map"): the page shows "standard" as "standard color"
# and "enhanced local" as "enhanced color" (VIEW_LABEL in docs/ods/index.html); the keys stay, so links and the
# entries published before keep working
GEOMETRY_PARTS = ("_range_log", "_range_linear", "_normal")
VARIANTS = (("masked", "", "terrain only"), ("rover", "with_rover", "terrain with rover (projected)"))   # key, sub-folder, label.  0.98.5
#   (Christian, 14 Nov 2026: "a third mode after the terrain only, terrain with rover, and terrain with 3D rover"): this
#   order, and a third variant "rover3d" - the terrain-only pictures with the rover's surface - where a surface was made
ROVER_3D = ("rover3d", "terrain with rover (3D model)")
# 0.99.26 (Christian, 18 Nov 2026: "Add this as a fourth mode in rpv as 3D terrain and rover. It should include the 3D
# rover"): the terrain-only panorama as the backdrop, with the station mesh (rpp.ibr.station_mesh) in front of it
TERRAIN_3D = ("terrain3d", "3D terrain and rover")     # 0.98.8 (Christian: "terrain only, terrain with rover proj, terrain with rover model")
DEFAULT_VIEWS = ("standard", "raw color", "enhanced local", "enhanced tone", "dcs color", "normal map", "distance map")   # 0.99.37; 0.98.30 (Christian, 16 Nov 2026: "Include the
#   enhlocal image versions in the upload and let it be viewable as one of the skins that button B shows"); 0.98.12 (Christian, 15 Nov 2026: "Uploaded the range map too in log scale"); ("standard", "normal") from 0.98.5 (Christian: "Uploaded the normal maps too"); ("standard",) since 0.97.18 (Christian, 13 Nov 2026: "don't upload the enhanced version to github");
#                                      ("standard", "enhanced") before; rpp single-sites --publish-views brings it back
GEOMETRY_WIDTH = 16384             # 0.98.12: the range and normal views are published at most this wide (the color ones at
#                                    width); 0.99.51: 4096 until now - they are made at half the color width
DEFAULT_WIDTH = 8192               # the square's side: 8192 x 8192 holds two 8192 x 4096 eyes (Quest 3: fine).  0.98.30:
#                                    the panoramas are rendered at 16K (rpp ods); publishing reduces them to this (area filter)


def station_files(folder: PathLike) -> Dict[str, Path]:
    """{view name: file} of the top-down stereo JPEGs in one variant's folder (``<prefix>[_enh|_enhlocal|_raw|_enhtone|_dcs|
    _normal|_range_log|_range_linear]_360_3DTD.jpg``)."""
    out: Dict[str, Path] = {}
    for f in sorted(Path(folder).glob("*" + STACK)):
        stem = f.name[:-len(STACK)]
        view = next((v for v, part in sorted(VIEWS.items(), key=lambda kv: -len(kv[1])) if part and stem.endswith(part)), "standard")
        out.setdefault(view, f)
    return out


MARK_SCALE = 1.30                  # the viewer's ground disc is this much wider than the mark in the panorama.  1.15
#                                    until 0.98.3 (Christian, 14 Nov 2026: "The blue disk should be a little larger")


def ground_mark(folder: Path, web: Path, pano_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
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
            dst = (Path(pano_dir) if pano_dir else web / "panos") / tex.name     # 0.99.3: pano_dir, the Hugging Face stage
            if not (dst.is_file() and dst.stat().st_mtime >= tex.stat().st_mtime):
                shutil.copyfile(tex, dst)
            return {"file": "panos/" + tex.name, "depth_m": round(float(m["depth_below_pivot_m"]), 4),
                    "radius_m": round(MARK_SCALE * float(m["radius_m"]), 4), "mark_radius_m": round(float(m["radius_m"]), 4),
                    "radius_deg": float(m.get("radius_deg") or 10.0),
                    # 0.98.7: where the page cuts the picture into the compass and the text ring - between the two
                    "split": round(0.5 * (float((m.get("compass") or {}).get("radius_in_disc") or 0.5)
                                          + float(m.get("text_inner_edge_in_disc") or 0.56)), 3),
                    "text_centre_azimuth_deg": round(float(m.get("text_centre_azimuth_deg") or 0.0), 2),
                    "rover_forward_azimuth_deg": rover_forward_azimuth(m)}
    return None


def rover_forward_azimuth(mark: Dict[str, Any]) -> Optional[float]:
    """0.98.28 (Christian, 16 Nov 2026: "O should take you to the looking straight ahead of where the rover is (the +x
    direction in rover frame)"): the azimuth (deg east of north) of the rover's +x from a station's ``nadir_mark``
    record - its own field, or, for a station rendered before 0.98.28, the text's middle less the turn the text has
    had from the front since 0.98.7 (``nadir.TEXT_AZIMUTH_FROM_FORWARD_DEG``, 135 deg; until 0.98.27 the page took the
    text's middle itself for the front - 135 deg off).  None when the record has neither."""
    if mark.get("rover_forward_azimuth_deg") is not None:
        return round(float(mark["rover_forward_azimuth_deg"]) % 360.0, 2)
    if mark.get("text_centre_azimuth_deg") is None:
        return None
    from .nadir import TEXT_AZIMUTH_FROM_FORWARD_DEG
    off = float(mark.get("text_from_forward_deg", TEXT_AZIMUTH_FROM_FORWARD_DEG))
    return round((float(mark["text_centre_azimuth_deg"]) - off) % 360.0, 2)


def rover_surface(folder: Path, web: Path, pano_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """0.98.2 (experimental): the rover's textured surface of a ``with_rover`` folder (``*_rover_surface.glb``,
    :mod:`rpp.ibr.rover_surface`) copied into ``web/panos`` - {"file", "triangles"} for the page, which shows it on
    the terrain-only panorama.  None when the folder has none."""
    glb = next(iter(sorted(folder.glob("*_rover_surface.glb"))), None) if folder.is_dir() else None
    if glb is None:
        return None
    dst = (Path(pano_dir) if pano_dir else web / "panos") / glb.name
    if not (dst.is_file() and dst.stat().st_mtime >= glb.stat().st_mtime):
        shutil.copy2(glb, dst)
    try:
        n = json.loads(glb.with_suffix(".json").read_text(encoding="utf-8")).get("triangles")
    except (OSError, ValueError):
        n = None
    return {"file": "panos/" + glb.name, "triangles": n, "experimental": True}


# 0.99.32 (Christian, 18 Nov 2026: "Build this as a fifth viewing mode"): one surface from under the rover to the sky
# (rpp.ibr.station_mesh.assemble_scape), no panorama behind it
SCAPE_3D = ("scape3d", "3D terrain to the horizon and rover")


# 0.99.33 (Christian, 18 Nov 2026: "Build a simple skid steering option. Call it a sixth mode. It will be ride along"):
# the fifth mode's file with the rover moved over the terrain by the page - a simulation, not a drive on Mars
DRIVE_3D = ("drive3d", "Drive the rover (simulation)")
DRIVE = {"track_m": 2.3,                    # between the left and the right wheels (about)
         "wheels_fwd_m": [1.10, 0.0, -1.07],    # the three axles ahead of the rover's center (about; ROVER_NAV x)
         "speed_m_s": 0.25,                 # both sticks full ahead (the rover's own top speed is about 0.04 m/s)
         "limit_m": 15.0}                   # the pivot stays within this of where it stood (0.99.39: the near terrain ends at
#                                             20 m, 10 m before - 6 m then)


def drive_settings() -> Dict[str, Any]:
    """The sixth mode's numbers for the page: :data:`DRIVE` and the pivot in the rover's frame (forward, right; m) -
    the mast axis of the fitted mast chain (:data:`rpp.ibr.rover_model.MAST_P_XY_RNAV_M`)."""
    from .rover_model import MAST_P_XY_RNAV_M
    return dict(DRIVE, pivot_fwd_right_m=[round(float(MAST_P_XY_RNAV_M[0]), 4), round(float(MAST_P_XY_RNAV_M[1]), 4)])


def station_scape(folder: Path, web: Path, pano_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """0.99.32: the fifth mode's file of a terrain-only folder (``*_station_scape.glb``) copied into ``web/panos`` -
    {"file", "triangles", "move_limit_m", "fade_m", "dome_m", "whole": True} for the page.  None when it has none."""
    glb = next(iter(sorted(folder.glob("*_" + "station_scape.glb"))), None) if folder.is_dir() else None
    if glb is None:
        return None
    dst = (Path(pano_dir) if pano_dir else web / "panos") / glb.name
    if not (dst.is_file() and dst.stat().st_mtime >= glb.stat().st_mtime):
        shutil.copy2(glb, dst)
    try:
        rec = json.loads(glb.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {}
    lim = ((rec.get("parts") or {}).get("rover") or {}).get("view_radius_m", 1.0)
    skins = {}
    from .station_mesh import skin_files                         # 0.99.41: the color skins of its texture
    for key, f in skin_files(folder).items():
        dst_s = (Path(pano_dir) if pano_dir else web / "panos") / f.name
        if not (dst_s.is_file() and dst_s.stat().st_mtime >= f.stat().st_mtime):
            shutil.copy2(f, dst_s)
        skins[key] = "panos/" + f.name
    out = {"file": "panos/" + glb.name, "triangles": rec.get("triangles"), "move_limit_m": lim, "fade_m": rec.get("fade_m") or [8.0, 10.0],
           "dome_m": rec.get("dome_m") or 3000.0, "whole": True, "experimental": True, "parts": list(rec.get("parts") or [])}
    if skins:
        out["skins"] = skins
    return out


def station_mesh(folder: Path, web: Path, pano_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """0.99.26: the station mesh of a terrain-only folder (``*_station_mesh.glb``, :mod:`rpp.ibr.station_mesh`) copied
    into ``web/panos`` - {"file", "triangles", "move_limit_m"} for the page.  None when the folder has none."""
    glb = next(iter(sorted(folder.glob("*_" + "station_mesh.glb"))), None) if folder.is_dir() else None
    if glb is None:
        return None
    dst = (Path(pano_dir) if pano_dir else web / "panos") / glb.name
    if not (dst.is_file() and dst.stat().st_mtime >= glb.stat().st_mtime):
        shutil.copy2(glb, dst)
    try:
        rec = json.loads(glb.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {}
    lim = ((rec.get("parts") or {}).get("rover") or {}).get("view_radius_m", 1.0)
    return {"file": "panos/" + glb.name, "triangles": rec.get("triangles"), "move_limit_m": lim,
            "fade_m": rec.get("fade_m") or [8.0, 10.0], "experimental": True}      # 0.99.39: the file's own (8-10 m before)


WELCOME_ID = "welcome"
WELCOME_STATION = "Sol0052_S003D2046"      # Van Zyl Overlook
WELCOME_WIDTH = 4096
WELCOME_MUTE = (0.60, 0.85, 0.08)          # the welcome picture (0.98.14; Christian, 15 Nov 2026: "the welcome screen should
#                                            have color, just be a little more muted than the true van zyl ods"): the colors'
#                                            distance from gray x the first, then x the second + the third x white
#                                            (0.98.4-0.98.13: gray, 0.55 x gray + 0.20 x white)
OLD_TEST_ID = "orientation_test"           # the labeled test pair of before 0.98.4: taken off the list


WELCOME_YAW_DEG = 270.0      # 0.98.19: the welcome view opens looking west, where its title is


def welcome_pano(web: Path, station_entry: Optional[Dict[str, Any]], pano_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """0.98.4 (Christian, 14 Nov 2026: "Remove the fake ods. In its stead put van zyl grayed out with north south and
    east in the sky in white letters ... Rover Photogrammetry Viewer (0.98.19; was MPPP stereo viewer) for Mars 2020 Perseverance Rover in bold welcoming
    letters a few meters from the viewer"): the list's first entry - Van Zyl Overlook's panoramas, their colors muted a
    little (WELCOME_MUTE; gray until 0.98.13), reduced to WELCOME_WIDTH.  0.99.38 (Christian, 19 Nov 2026: "the welcome
    screen should now have the same color products as Van zyl"): every view of its terrain-only variant (the standard
    one alone before).  A view whose picture is not here (Van Zyl's pictures are on Hugging Face) keeps the welcome
    picture made before.  The page draws the words over it (an entry with "welcome": true).  None when Van Zyl is not
    published."""
    import cv2
    if not station_entry:
        return None
    vs = station_entry.get("variants", [])
    var = next((v for v in vs if v.get("key") == "masked" and v.get("views", {}).get("standard")), None) or \
        next((v for v in vs if v.get("views", {}).get("standard")), None)
    if var is None:
        return None
    old = web / "panos" / f"{WELCOME_ID}{STACK}"                 # the gray one of before 0.98.14
    if old.is_file():
        old.unlink()
    views: Dict[str, str] = {}
    for name, rel in var["views"].items():
        rel = str(rel)
        tag = "" if name == "standard" else "_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        dst = web / "panos" / f"{WELCOME_ID}{tag}_muted{STACK}"
        src = web / rel
        if pano_dir is not None and rel.startswith("panos/") and (Path(pano_dir) / rel[6:]).is_file():
            src = Path(pano_dir) / rel[6:]                       # 0.99.3: this run's picture, in the Hugging Face stage
        if not src.is_file():                                    # 0.99.3: Van Zyl's pictures are on Hugging Face (a URL):
            if dst.is_file():                                    # the welcome picture made before stays
                views[name] = "panos/" + dst.name
            continue
        if not (dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime):
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[1] > WELCOME_WIDTH:
                img = cv2.resize(img, (WELCOME_WIDTH, WELCOME_WIDTH), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
            sat, gain, lift = WELCOME_MUTE
            m = np.clip(gain * (g + sat * (img.astype(np.float32) - g)) + lift * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(dst), m, [cv2.IMWRITE_JPEG_QUALITY, 85])
            try:
                from .xmp import tag_stereo
                tag_stereo(dst, extra={"Station": WELCOME_STATION})
            except Exception:                                    # noqa: BLE001
                pass
        views[name] = "panos/" + dst.name
    if "standard" not in views:
        return None
    return _welcome_entry(views)


def _welcome_entry(views) -> Dict[str, Any]:
    """The welcome entry: ``views`` {view: "panos/<file>"} (0.99.38), or the one picture (a Path) of before."""
    if isinstance(views, Path):
        views = {"standard": "panos/" + views.name}
    return {"id": WELCOME_ID, "title": "Welcome", "welcome": True, "site_name": "Rover Photogrammetry Viewer (RPV)",
            "yaw_deg": WELCOME_YAW_DEG,                       # 0.98.19 (Christian, 16 Nov 2026): it opens looking west
            "variants": [{"key": "welcome", "label": "welcome", "views": dict(views)}]}


TRAVERSE_POINTS = 400


def traverse_map() -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """0.98.5 (Christian, 14 Nov 2026: "a little page attached to the left hand that has a little x on the map"): the
    rover's traverse for the page's small map - (the waypoint table, {"origin_en": [E, N] meters, "xy": [[x, y], ...]
    meters east and north of the origin, about TRAVERSE_POINTS of them in drive order}).  (None, None) without the table."""
    try:
        from ..waypoints.table import load_waypoints, _props, _en
        wp = load_waypoints()
        pts = []
        for f in wp.get("features", []):
            pr = _props(f)
            en = _en(pr)
            if en is not None and pr.get("sol") is not None:
                pts.append((int(pr["sol"]), int(pr.get("site") or 0), int(pr.get("drive") or 0), float(en[0]), float(en[1])))
        if len(pts) < 2:
            return None, None
        pts.sort()
        e0, n0 = min(p[3] for p in pts), min(p[4] for p in pts)
        k = max(1, len(pts) // TRAVERSE_POINTS)
        keep = pts[::k] + ([pts[-1]] if (len(pts) - 1) % k else [])
        return wp, {"origin_en": [round(e0, 1), round(n0, 1)], "xy": [[round(p[3] - e0, 1), round(p[4] - n0, 1)] for p in keep],
                    "note": "meters east and north of origin_en (the mission's map frame); the rover's waypoints in order"}
    except Exception as e:                                   # noqa: BLE001 - the map is an extra
        log.info("publish: no traverse map (%s)", e)
        return None, None


ORTHO_FILE = "traverse_ortho.jpg"
ORTHO_PX = 1024                # the orthomosaic under the info page's map: this many pixels a side (low resolution)
ORTHO_MARGIN = 0.04            # ... a square this share wider on each side than the traverse's longer extent


def ortho_sources(root: Optional[PathLike] = None) -> List[str]:
    """The orthomosaic GeoTIFFs for the map: ``M20_ORTHOS`` (an os.pathsep list, as make_maps.bat sets it), else the
    files under ``<root>/dem`` - the Mars 2020 TRN mosaic (``m2020_trn``: the whole traverse) before the MSR ones."""
    env = [x for x in os.environ.get("M20_ORTHOS", "").split(os.pathsep) if x and Path(x).is_file()]
    if env or not root:
        return env
    out: List[str] = []
    for sub in ("m2020_trn", "msr_trn"):
        out += sorted(str(f) for f in (Path(root) / "dem" / sub).glob("*rtho*.tif"))
    return out


def traverse_ortho(wp: Dict[str, Any], traverse: Dict[str, Any], web: Path, root: Optional[PathLike] = None,
                   size: int = ORTHO_PX, old: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """0.98.14 (Christian, 15 Nov 2026: "Add low res ortho to the info map"): the orbital orthomosaic under the
    traverse, ``size`` pixels a side, as ``web/panos/traverse_ortho.jpg`` - {"file", "extent": [x0, y0, x1, y1] in
    the traverse map's meters (north up), "m_per_px", "source"}.  ``old``: the record of the list there; when it has
    the same extent and its file is there, nothing is read again (the mosaic is 7 GB).  None without an orthomosaic,
    rasterio or pyproj - the map is then the line alone."""
    xy = np.asarray(traverse["xy"], float)
    c = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    h = 0.5 * float(np.ptp(xy, axis=0).max()) * (1.0 + 2.0 * ORTHO_MARGIN)
    extent = [round(float(v), 1) for v in (c[0] - h, c[1] - h, c[0] + h, c[1] + h)]
    dst = web / "panos" / ORTHO_FILE
    if old and old.get("extent") == extent and old.get("px") == int(size) and dst.is_file():
        return old
    srcs = ortho_sources(root)
    if not srcs:
        log.info("publish: no orthomosaic for the map (M20_ORTHOS, or <root>/dem/m2020_trn/*rtho*.tif)")
        return None
    try:
        import cv2
        import rasterio
        from pyproj import Transformer
        from scipy.ndimage import map_coordinates
        from ..waypoints.table import _props
        P = [_props(f) for f in wp.get("features", [])]
        P = [q for q in P if all(q.get(k) is not None for k in ("lon", "lat", "easting", "northing"))]
        ce = np.polyfit([q["lon"] for q in P], [q["easting"] for q in P], 1)        # the map frame is linear in lon and lat
        cn = np.polyfit([q["lat"] for q in P], [q["northing"] for q in P], 1)
        e0, n0 = traverse["origin_en"]
        gx = extent[0] + (np.arange(size) + 0.5) / size * 2 * h + e0
        gy = extent[3] - (np.arange(size) + 0.5) / size * 2 * h + n0               # row 0 is the north edge
        E, N = np.meshgrid(gx, gy)
        lon, lat = (E - ce[1]) / ce[0], (N - cn[1]) / cn[0]
        img = np.full((size, size), np.nan)
        used = []
        for path in srcs:
            todo = np.isnan(img)
            if not todo.any():
                break
            with rasterio.open(path) as src:
                x, y = Transformer.from_crs("+proj=longlat +R=3396190 +no_defs", src.crs, always_xy=True).transform(lon[todo], lat[todo])
                r, cc = rasterio.transform.rowcol(src.transform, x, y, op=lambda v: v)
                r, cc = np.asarray(r, float) - 0.5, np.asarray(cc, float) - 0.5
                ok = (r >= 0) & (cc >= 0) & (r <= src.height - 1) & (cc <= src.width - 1)
                if not ok.any():
                    continue
                r0, r1 = int(np.floor(r[ok].min())), min(int(np.ceil(r[ok].max())) + 1, src.height - 1)
                c0, c1 = int(np.floor(cc[ok].min())), min(int(np.ceil(cc[ok].max())) + 1, src.width - 1)
                win = rasterio.windows.Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
                k = max(1, int(max(win.height, win.width) / (2 * size)))             # read at about twice the output
                arr = src.read(1, window=win, out_shape=(max(1, int(win.height // k)), max(1, int(win.width // k))), masked=True)
                arr = arr.astype(float).filled(np.nan)
                arr[arr == 0] = np.nan                                               # 0 = no data in the USGS mosaics
                v = map_coordinates(np.nan_to_num(arr), [(r[ok] - r0) / k, (cc[ok] - c0) / k], order=1, mode="nearest")
                v[map_coordinates(np.isnan(arr).astype(float), [(r[ok] - r0) / k, (cc[ok] - c0) / k], order=0, mode="nearest") > 0] = np.nan
                sub = img[todo]
                sub[ok] = v
                img[todo] = sub
                used.append(Path(path).name)
        if not np.isfinite(img).any():
            log.info("publish: the orthomosaics do not cover the traverse")
            return None
        lo, hi = np.nanpercentile(img, [0.5, 99.5])
        g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 215 + 20
        g[~np.isfinite(img)] = 0
        cv2.imwrite(str(dst), g.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 82])
        log.info("publish: the map's orthomosaic %s (%d px, %.1f m a pixel, from %s)", dst.name, size, 2 * h / size, ", ".join(used))
        return {"file": "panos/" + ORTHO_FILE, "extent": extent, "px": int(size), "m_per_px": round(2 * h / size, 2),
                "source": used, "credit": "HiRISE orthomosaic (NASA/JPL-Caltech/UArizona/USGS)"}
    except Exception as e:                                   # noqa: BLE001 - the picture is an extra
        log.info("publish: no orthomosaic under the map (%s: %s)", type(e).__name__, e)
        return None


DTM_MAP_FILE = "traverse_dtm.jpg"
DTM_MAP_NAME = "m20_ods_traverse_map_dtm"     # rpp maps figures 8 --tag dtm (mapplot3.fig8): figure 5's panel D, blank
DTM_MAP_RENDER_S = 1800


def _lonlat_fit(wp: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """The mission map frame as linear functions of longitude and latitude (easting = ce[0] lon + ce[1], ...)."""
    from ..waypoints.table import _props
    P = [_props(f) for f in wp.get("features", [])]
    P = [q for q in P if all(q.get(k) is not None for k in ("lon", "lat", "easting", "northing"))]
    return (np.polyfit([q["lon"] for q in P], [q["easting"] for q in P], 1),
            np.polyfit([q["lat"] for q in P], [q["northing"] for q in P], 1))


def dtm_files(root: Optional[PathLike]) -> List[str]:
    """The HiRISE DTMs of ``<root>/dem`` as m20_dem_env.bat finds them (msr_trn, then m2020_trn), else ``M20_DEMS``."""
    found = [str(f) for sub in ("msr_trn", "m2020_trn") for f in sorted((Path(root) / "dem" / sub).glob("*DTM*.tif"))] if root else []
    return found or [x for x in os.environ.get("M20_DEMS", "").split(os.pathsep) if x and Path(x).is_file()]


def dtm_map_dir(root: Optional[PathLike]) -> Optional[Path]:
    """Where the info page's map is kept between runs: ``<root>/dem/ods_info_map`` (beside the DTMs)."""
    return Path(root) / "dem" / "ods_info_map" if root else None


def dtm_map_source(web: Path, root: Optional[PathLike] = None) -> Optional[Path]:
    """The blank panel-D map: ``RPP_ODS_MAP`` (a PNG with its .json), else the one kept in ``dtm_map_dir(root)``."""
    env = rpp_env.get("RPP_ODS_MAP", "").strip()
    if env:
        return Path(env) if Path(env).is_file() and Path(env).with_suffix(".json").is_file() else None
    d = dtm_map_dir(root)
    found = [f for f in d.glob(f"{DTM_MAP_NAME}*.png") if f.with_suffix(".json").is_file()] if d and d.is_dir() else []
    return max(found, key=lambda f: f.stat().st_mtime) if found else None


def dtm_map_key(wp: Optional[Dict[str, Any]], root: Optional[PathLike]) -> Dict[str, Any]:
    """What the kept map was drawn from: the DTM files and the last sol of the traverse - it is drawn again when either
    changes."""
    sols = []
    try:
        from ..waypoints.table import _props
        sols = [int(_props(f)["sol"]) for f in (wp or {}).get("features", []) if _props(f).get("sol") is not None]
    except Exception:                                        # noqa: BLE001
        pass
    return {"dtm_files": sorted(Path(x).name for x in dtm_files(root)), "last_sol": max(sols) if sols else None}


def render_dtm_map(root: Optional[PathLike], key: Dict[str, Any]) -> Optional[Path]:
    """``rpp maps figures 8 --background dtm --tag dtm`` (figure 5's panel D, blank) into ``dtm_map_dir(root)``,
    drawn on the high-resolution HiRISE DTMs of ``<root>/dem``; ``key`` goes into its .json."""
    import subprocess
    import sys
    out = dtm_map_dir(root)
    if out is None:
        return None
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    dems = dtm_files(root)
    if dems:
        env["M20_DEMS"] = os.pathsep.join(dems)
    env["M20_BACKGROUND"] = "dtm"
    log.info("publish: drawing the info page's map (figure 5 panel D, blank; %s) into %s",
             f"{len(dems)} HiRISE DTMs" if dems else "no HiRISE DTM found: ~70 m terrain", out)
    try:
        r = subprocess.run([sys.executable, "-m", "rpp", "maps", "figures", "8", "--background", "dtm", "--tag", "dtm",
                            "--out", str(out)], env=env, timeout=DTM_MAP_RENDER_S)
        if r.returncode != 0:
            log.info("publish: the map was not drawn (rpp maps figures 8: exit %d)", r.returncode)
            return None
    except Exception as e:                                   # noqa: BLE001 - the map is an extra
        log.info("publish: the map was not drawn (%s: %s)", type(e).__name__, e)
        return None
    png = out / f"{DTM_MAP_NAME}.png"
    if not png.is_file():
        return None
    meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
    meta["key"] = key
    png.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return png


def traverse_dtm(wp: Dict[str, Any], traverse: Dict[str, Any], web: Path, root: Optional[PathLike] = None,
                 old: Optional[Dict[str, Any]] = None, render: bool = True) -> Optional[Dict[str, Any]]:
    """0.98.19 (Christian, 16 Nov 2026: "The map in the info tab should be the same background as
    m20_mars_to_rover_sites_dtm figure D. Render a blank version of that with the traverse and 2 km scale bar. Put an
    orange x over the ods' particular location"): the blank panel D as ``web/panos/traverse_dtm.jpg`` - {"file",
    "extent": [x0, y0, x1, y1] in the traverse map's meters (north up), "px", "source", "background"}.  The traverse and
    the scale bar are in the picture; the page draws the x.  (Christian, 16 Nov 2026: "render it from the high res dtm on the
    computer. If it isn't there then, have the ibr script render it once and save it for reuse until it changes"):
    drawn once on the HiRISE DTMs of ``<root>/dem`` and kept in ``<root>/dem/ods_info_map``; drawn again
    (``render``) when the DTM files or the traverse's last sol differ from the kept one's.  None without one - the page then draws the orthomosaic map of 0.98.14."""
    src = dtm_map_source(web, root)
    if render and not rpp_env.get("RPP_ODS_MAP", "").strip():
        key = dtm_map_key(wp, root)
        try:
            kept = json.loads(src.with_suffix(".json").read_text(encoding="utf-8")).get("key") if src else None
        except (OSError, ValueError):
            kept = None
        if kept != key:                                      # none yet, other DTMs, or the rover has driven since
            src = render_dtm_map(root, key) or src
    if src is None:
        return None
    try:
        meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
        dst = web / "panos" / DTM_MAP_FILE
        stamp = f"{src.name} {int(src.stat().st_mtime)}"
        if old and old.get("stamp") == stamp and dst.is_file():
            return old
        lo0, lo1, la0, la1 = meta["extent_lonlat"]
        ce, cn = _lonlat_fit(wp)
        e0, n0 = traverse["origin_en"]
        extent = [round(float(np.polyval(ce, lo0) - e0), 1), round(float(np.polyval(cn, la0) - n0), 1),
                  round(float(np.polyval(ce, lo1) - e0), 1), round(float(np.polyval(cn, la1) - n0), 1)]
        import cv2
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return None
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info("publish: the info page's map %s from %s (%d x %d px)", dst.name, src, img.shape[1], img.shape[0])
        return {"file": "panos/" + DTM_MAP_FILE, "extent": extent, "px": [int(img.shape[1]), int(img.shape[0])],
                "stamp": stamp, "background": meta.get("background", ""),
                "credit": "HiRISE DTM and terrain (NASA/JPL-Caltech/UArizona/USGS)"}
    except Exception as e:                                   # noqa: BLE001 - the map is an extra
        log.info("publish: no DTM map (%s: %s)", type(e).__name__, e)
        return None


def station_on_map(label: str, wp: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    """[x, y] of a station ("...S003D2046...") in the traverse map's meters, or None."""
    m = re.search(r"S(\d{3})D(\d{4})", label)
    if not m or not wp:
        return None
    try:
        from ..waypoints.table import waypoint_for_site_drive, _props, _en
        f = waypoint_for_site_drive(wp, int(m.group(1)), int(m.group(2)))
        en = _en(_props(f)) if f else None
        ens = [_en(_props(x)) for x in wp.get("features", [])]
        ens = [x for x in ens if x is not None]
        if en is None or not ens:
            return None
        return [round(float(en[0]) - min(x[0] for x in ens), 1), round(float(en[1]) - min(x[1] for x in ens), 1)]
    except Exception:                                        # noqa: BLE001
        return None


def station_info(folder: Path) -> Dict[str, Any]:
    """{"site_name", "sols"} of a variant's folder, from its panorama record's mark (the page's X button (A before 0.98.13) shows them)."""
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


CENTER_JPG = "_360_center.jpg"     # 0.99.36: the panorama from the pivot (one eye, 2:1), for a screen
CENTER_FULL_JPG = "_360_center_16k.jpg"
CENTER_FULL_WIDTH = 16384          # 0.99.36 (Christian: "the 2d rpv should show the 16k center panorama if the viewer's
#                                    platform can handle it"): the full-width copy, for a page whose GPU takes it


def center_view(folder: PathLike, pano_dir: PathLike, width: int, quality: int = 88,
                full_width: int = CENTER_FULL_WIDTH) -> Optional[Dict[str, str]]:
    """0.99.36 (Christian, 19 Nov 2026: "for the rpv browser layer, include center with the left and right eye if
    center is available"): the 8-bit center panorama of one variant's folder (``<prefix>_ods_center.png``, written by
    ``rpp ods --eyes ... center``: the with-rover variant has it when the rover surface is made) as JPEGs in
    ``pano_dir``: ``<prefix>_360_center.jpg`` at most ``width`` wide, and - when the panorama is wider than that -
    ``<prefix>_360_center_16k.jpg`` at most ``full_width`` wide (the page takes it where the GPU can hold it).
    Returns {"standard": "panos/<name>", "full": "panos/<name>" or absent}, or None without a center panorama."""
    import cv2
    cands = sorted(f for f in Path(folder).glob("*ods_cent*.png")
                   if f.name.endswith(("ods_center.png", "ods_centre.png")))
    if not cands:
        return None
    src = max(cands, key=lambda f: f.stat().st_mtime)
    stem = src.name.rsplit("ods_cent", 1)[0].rstrip("_")
    im = None
    out: Dict[str, str] = {}
    for key, name, w in (("standard", CENTER_JPG, width), ("full", CENTER_FULL_JPG, full_width)):
        dst = Path(pano_dir) / f"{stem}{name}"
        if not (dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime):
            if im is None:
                im = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if im is None:
                    return None
            if key == "full" and im.shape[1] <= width:           # no wider than the standard one: no second copy
                break
            j = im if im.shape[1] <= w else cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))),
                                                      interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), j, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        out[key] = "panos/" + dst.name
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
            titles: Optional[Dict[str, Dict[str, Any]]] = None, title: str = "Rover Photogrammetry Viewer (RPV)",
            ortho_root: Optional[PathLike] = None, render_map: bool = False,
            files_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    """Every station folder of ``out_root`` (the output of ``rpp single-sites``) that has a stereo JPEG, into
    ``web_dir``.  ``titles``: {station label: {"title", "site_name", "sols", "note", "yaw_deg"}} (``site_name``,
    ``sols``: the name and sols in the page's menu and info, before those in the panorama's mark; ``yaw_deg``: the azimuth, degrees east of north,
    straight ahead when the page opens).  Stations already in ``panos.json`` and not in ``out_root`` stay listed, with
    the names in ``titles``.  ``ortho_root``: the folder whose ``dem/`` holds the orthomosaics for the map (default: ``out_root``'s
    parent, the blocks' root).  ``render_map``: render the info page's map (rpp maps figures 8) when there is none.
    ``files_dir`` (0.99.3): the stations' pictures go there instead of ``web_dir/panos`` (the entries still say
    ``panos/<file>``) - the stage :mod:`rpp.ibr.hf` uploads to Hugging Face, outside the viewer's git checkout.
    Returns the list written."""
    from .. import __version__
    out_root, web = Path(out_root), Path(web_dir)
    (web / "panos").mkdir(parents=True, exist_ok=True)
    pano_dir = Path(files_dir) if files_dir else web / "panos"
    pano_dir.mkdir(parents=True, exist_ok=True)
    index = web / "panos.json"
    try:
        old = {p["id"]: p for p in json.loads(index.read_text(encoding="utf-8")).get("panos", [])}
    except (OSError, ValueError):
        old = {}
    NAME_KEYS = ("title", "site_name", "sols", "note", "yaw_deg")         # 0.99.36: the names of stations listed before this
    for k, e in old.items():                                     # run follow the station list too (a station renamed
        t = (titles or {}).get(k) or {}                          # since its panoramas were made: Three Forks North)
        e.update({n: t[n] for n in NAME_KEYS if t.get(n) is not None})
    wp, traverse = traverse_map()
    if traverse:                                                 # 0.98.14: the orbital picture under the map's line
        try:
            was = (json.loads(index.read_text(encoding="utf-8")).get("traverse") or {}).get("ortho")
        except (OSError, ValueError):
            was = None
        ortho = traverse_ortho(wp, traverse, web, root=ortho_root if ortho_root is not None else out_root.parent, old=was)
        if ortho:
            traverse["ortho"] = ortho
        try:
            was = (json.loads(index.read_text(encoding="utf-8")).get("traverse") or {}).get("dtm")
        except (OSError, ValueError):
            was = None
        dtm = traverse_dtm(wp, traverse, web, root=ortho_root if ortho_root is not None else out_root.parent, old=was,
                           render=render_map)
        if dtm:
            traverse["dtm"] = dtm
    bad = [v for v in views if v not in VIEWS]
    if bad:
        raise ValueError(f"views {bad}: one of {list(VIEWS)}")
    n = 0
    for d in sorted(p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith("_")):   # 0.99.3: not _hf_stage
        if stations and d.name not in stations:
            continue
        third = fourth = fifth = sixth = None
        entry: Dict[str, Any] = {"id": d.name, "title": d.name, "variants": [], "mppp": __version__, "rpp": __version__, "author": "Christian Tate",
                                 "published": time.strftime("%Y-%m-%d")}
        entry.update({k: v for k, v in ((titles or {}).get(d.name) or {}).items() if v is not None})
        for key, sub, label in VARIANTS:
            files = station_files(d / sub) if (d / sub).is_dir() else {}
            got = {}
            for v in views:
                if v in files:
                    got[v] = "panos/" + _copy(files[v], pano_dir / files[v].name,
                                              min(width, GEOMETRY_WIDTH) if VIEWS[v] in GEOMETRY_PARTS else width, quality)["file"]
                    try:                                         # a reduced copy was encoded again: its XMP with it
                        from .xmp import read_xmp, tag_stereo
                        if read_xmp(pano_dir / files[v].name) is None:
                            tag_stereo(pano_dir / files[v].name, extra={"Station": d.name})
                    except Exception:                            # noqa: BLE001
                        pass
                    n += 1
            if got:
                var = {"key": key, "label": label, "views": got}
                cv = center_view(d / sub, pano_dir, width, quality) if got.get("standard") else None
                if cv:                                   # 0.99.36: the center eye, a screen's third choice
                    var["center"] = {"standard": cv["standard"]}
                    if cv.get("full"):                   # its 16K copy, for a GPU that can hold it
                        var["center_full"] = {"standard": cv["full"]}
                    n += len(cv)
                gm = ground_mark(d / sub, web, pano_dir)
                if gm:
                    var["ground_mark"] = gm
                entry["variants"].append(var)
                if key == "masked":                      # 0.98.2 (experimental): the rover as a surface on the terrain;
                    rs = rover_surface(d / "with_rover", web, pano_dir)        # 0.98.5: a variant of its own, listed last
                    if rs:
                        third = dict(var, key=ROVER_3D[0], label=ROVER_3D[1], rover_surface=rs)
                    # 0.99.38: the 3D modes carry the rover surface too - the page draws the rover from it (colored
                    # from the pivot's panorama at full resolution) in place of the station mesh's baked rover
                    extra = {"rover_surface": rs} if rs else {}
                    sm = station_mesh(d / sub if sub else d, web, pano_dir)        # 0.99.26: the fourth mode
                    if sm:
                        fourth = dict(var, key=TERRAIN_3D[0], label=TERRAIN_3D[1], station_mesh=sm, **extra)
                    sc = station_scape(d / sub if sub else d, web, pano_dir)       # 0.99.32: the fifth mode
                    if sc:
                        fifth = dict(var, key=SCAPE_3D[0], label=SCAPE_3D[1], station_mesh=sc, **extra)
                        if "rover" in (sc.get("parts") or []):    # 0.99.33: the sixth mode needs the rover in the file
                            sixth = dict(var, key=DRIVE_3D[0], label=DRIVE_3D[1], station_mesh=dict(sc, drive=drive_settings()), **extra)
        # 0.98.4 (Christian, 14 Nov 2026: "always at ground level even when the rover is displayed"): with the rover in
        # the picture the mark was put at the deck's depth under the mast; the page's disc lies on the terrain in both
        # variants - the terrain-only variant's depth, the disc as wide there as it looks in the panorama
        ground = next((v["ground_mark"] for v in entry["variants"] if v["key"] == "masked" and v.get("ground_mark")), None)
        for v in entry["variants"]:
            if ground and not v.get("ground_mark") and v["key"] != "masked":
                # 0.98.19 (Christian, 16 Nov 2026: "the blue disk in the projected rover ods does not spin"): a rover
                # panorama of 0.98.4-0.98.18 wrote no mark texture (its last eye, the pivot's, carries none) - the
                # terrain-only variant's disc, at the terrain's level, covers the mark baked into the picture
                v["ground_mark"] = dict(ground)
            gm = v.get("ground_mark")
            if ground and gm and v["key"] != "masked" and ground["depth_m"] > gm["depth_m"]:
                r = float(ground["depth_m"]) * float(np.tan(np.radians(gm.get("radius_deg", 10.0))))
                gm.update(depth_m=ground["depth_m"], mark_radius_m=round(r, 4), radius_m=round(MARK_SCALE * r, 4), at="the terrain's level")
        if third is not None:
            entry["variants"].append(third)
        if fourth is not None:
            entry["variants"].append(fourth)
        if fifth is not None:
            entry["variants"].append(fifth)
        if sixth is not None:
            entry["variants"].append(sixth)
        xy = traverse and station_on_map(d.name, wp)
        if xy:
            entry["map"] = xy
        for key, sub, label in VARIANTS:                         # the X button's words (info)
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
    first = welcome_pano(web, old.get(WELCOME_STATION), pano_dir if files_dir else None)
    doc = {"title": title, "panos": ([first] if first else []) + [old[k] for k in sorted(old)]}
    if traverse:
        doc["traverse"] = traverse
    index.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    log.info("publish: %d stations listed in %s (%d images from this run)", len(doc["panos"]), index, n)
    return doc


VIEWER_URL = "https://ctate7163.github.io/rpv/"        # 0.98.29: the viewer's own repository, ctate7163/rpv (lower case, 0.99.13)


def viewer_dir() -> Path:
    """Where ``single-sites --upload`` publishes when no ``--publish`` folder is given (0.98.29; Christian, 16 Nov 2026:
    "change https://ctate7163.github.io/rpp/ods/ to https://ctate7163.github.io/RPV/"): ``RPP_VIEWER_DIR``; else a git
    checkout named ``RPV`` beside this one (``D:\\code\\RPV`` beside ``D:\\code\\MPPP``), whose pushes publish
    VIEWER_URL; else this checkout's ``docs/ods`` (as before 0.98.29 - on GitHub that page now sends visitors to
    VIEWER_URL, so panoramas published there are not seen)."""
    env = rpp_env.get("RPP_VIEWER_DIR")
    if env:
        return Path(env)
    home = Path(__file__).resolve().parents[3]
    for name in ("rpv", "RPV"):                      # 0.99.13: the checkout may be renamed with the repository
        sib = home.parent / name
        if (sib / ".git").exists():
            return sib
    return home / "docs" / "ods"


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
    """A labeled stereo pair for checking a viewer: compass letters every 45 degrees on the horizon (N at the left
    edge, azimuth growing to the right), elevation lines every 30 degrees, "UP" / "DOWN" at the poles, and the eye's
    name ("LEFT EYE" in blue on top, "RIGHT EYE" in red below).  The letters are 3 px apart between the eyes, so
    they float in front of the lines when the eyes are the right way round and sink behind them when swapped."""
    import cv2
    h = width // 2
    eyes = []
    for eye, color, shift in (("LEFT EYE", (255, 170, 90), 3), ("RIGHT EYE", (90, 140, 255), -3)):
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
                cv2.putText(im, eye, (xx + 10 + shift, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
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
PAGE_FILES = ("index.html", "vendor", ".nojekyll")   # 0.99.36: committed by upload() with the panoramas


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
      upload (any later run, or ``rpp single-sites --publish-only --upload``) sends that commit as well;
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
    # 0.99.36 (Christian, 19 Nov 2026: 'the rpv still has "Mars rover panorama viewer (RPV)" at the upper left'): the
    # page itself too - since the viewer moved to its own repository (ctate7163/rpv, 0.99.13) publish copied the new
    # index.html into it on every run, but only panos/ and panos.json were committed, so the live page stayed at the
    # one of 17 Nov.  The page, its libraries and .nojekyll are committed with the panoramas when they changed.
    paths += [str(web / n) for n in PAGE_FILES if (web / n).exists()]
    rc, out = _git(repo, "add", "--", *paths)
    if rc:
        return done(False, False, False, f"git add failed: {out}")
    committed = bool(_git(repo, "diff", "--cached", "--quiet", "--", *paths)[0])
    if committed:
        what = ", ".join(stations) if 0 < len(stations) <= 4 else f"{len(stations)} stations" if stations else "update"
        page = bool(_git(repo, "diff", "--cached", "--quiet", "--", *[pp for pp in paths if Path(pp).name in PAGE_FILES])[0])
        rc, out = _git(repo, "commit", "-m", f"Viewer: {'page and ' if page else ''}panoramas {what} (RPP {__version__})", "--", *paths)
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
