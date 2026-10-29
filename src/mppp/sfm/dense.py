"""
Dense reconstruction (0.95.0, Christian, 28 Oct 2026): the COLMAP dense workspace of a finished block, preloaded
from the alignment so that the GUI's Reconstruction > Dense reconstruction dialog - or ``dense_reconstruct.bat``
next to ``open_in_colmap.bat`` - runs the stereo and the fusion straight away.

* :func:`prepare_dense` - ``<project>/dense/``: the images undistorted to pinhole cameras from the NATIVE-pixel
  model ``gui_native/sparse`` (the image files are at their native resolution; the full-resolution model's cameras
  would not fit them), ``sparse/`` (the pinhole model), ``stereo/patch-match.cfg`` and ``fusion.cfg`` (COLMAP's
  ``image_undistorter``, through pycolmap), and ``masks/`` - the rover masks undistorted the same way, for the
  fusion (``StereoFusionOptions.mask_path``: a pixel at 0 is not fused; the patch-match stereo itself takes no
  masks).  ``mppp_dense.json`` records what was done.
* :func:`write_dense_bat` - ``dense_reconstruct.bat``: double-click on the PC: the workspace (if not there yet),
  ``patch_match_stereo`` (needs a CUDA COLMAP; the batch file looks for one first), ``stereo_fusion`` with the
  masks into ``dense/fused.ply``, and a Poisson mesh on request.  ``set MPPP_DENSE_STEP=prepare|stereo|fusion|mesh``
  runs one step, ``MPPP_DENSE_MAX_SIZE`` (default 2000 px on the long side) the working resolution.
* :func:`run_dense` - the same steps from Python (a CUDA pycolmap for the stereo: the ``gpu_python`` environment).

The masks and the undistortion are per image: a block of 1000 images at 2000 px is about 1-2 GB and a few minutes
to prepare; the stereo takes hours on the GPU.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .project import SfmProject

log = logging.getLogger("mppp.sfm")
PathLike = Union[str, Path]
DENSE_DIR = "dense"
DENSE_MAX_IMAGE_SIZE = 2000        # the long side of the undistorted (working) images; the GUI's default too


def native_model_dir(project: SfmProject) -> Path:
    """``gui_native/sparse`` - written at the end of a run (``gui_native``), or now from the saved model."""
    nat = Path(project.root) / "gui_native" / "sparse"
    if not (nat / "cameras.bin").is_file():
        import pycolmap
        from .export import write_gui_native
        rs = project.settings.get("reconstruction") or {}
        model = Path(project.root) / (rs.get("path") or "sparse/cahv_ba")
        if not (model / "cameras.bin").is_file():
            raise FileNotFoundError(f"no model at {model}: align the block first")
        log.info("dense: gui_native missing - writing the native-pixel model from %s", model)
        write_gui_native(project, pycolmap.Reconstruction(str(model)), colors=False)
    return nat


def prepare_dense(project: SfmProject, out_dir: Optional[PathLike] = None,
                  max_image_size: int = DENSE_MAX_IMAGE_SIZE, masks: bool = True, overwrite: bool = False,
                  num_threads: int = -1) -> Dict[str, Any]:
    """The dense workspace ``out_dir`` (default ``<project>/dense``); returns its record (also ``mppp_dense.json``)."""
    import numpy as np
    import pycolmap
    out = Path(out_dir) if out_dir else Path(project.root) / DENSE_DIR
    rec_file = out / "mppp_dense.json"
    if (out / "sparse" / "cameras.bin").is_file() and rec_file.is_file() and not overwrite:
        prev = json.loads(rec_file.read_text(encoding="utf-8"))
        if int(prev.get("max_image_size", -1)) == int(max_image_size):
            log.info("dense: workspace %s is there (max_image_size %d); overwrite=True to redo it", out, max_image_size)
            return prev
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = native_model_dir(project)
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = int(max_image_size)
    from .reconstruction import default_threads
    nt = int(num_threads) if int(num_threads) > 0 else default_threads()
    pycolmap.undistort_images(str(out), str(model), str(project.images_dir), output_type="COLMAP",
                              undistort_options=opts, num_threads=nt)
    und = pycolmap.Reconstruction(str(out / "sparse"))
    res: Dict[str, Any] = {"workspace": str(out), "source_model": str(model), "images": und.num_images(),
                           "cameras": und.num_cameras(), "max_image_size": int(max_image_size), "masks": None,
                           "seconds": None}
    if masks and project.masks_dir.is_dir():
        names = [r["name"] for r in project.images if r.get("has_mask") and (project.masks_dir / (r["name"] + ".png")).is_file()]
        in_model = {im.name for im in und.images.values()}
        names = [n for n in names if n in in_model]
        if names:
            # the masks undistorted with the same cameras: a folder in which each mask carries its image's name,
            # undistorted into a workspace of its own, then moved to masks/<image name>.png (what the fusion reads),
            # thresholded back to 0 / 255 (the resampling makes grey edges)
            tmp_in, tmp_out = out / "_masks_in", out / "_masks_out"
            for d in (tmp_in, tmp_out):
                if d.exists():
                    shutil.rmtree(d)
            tmp_in.mkdir()
            for n in names:
                src = project.masks_dir / (n + ".png")
                dst = tmp_in / n
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
            pycolmap.undistort_images(str(tmp_out), str(model), str(tmp_in), image_names=names, output_type="COLMAP",
                                      undistort_options=opts, num_threads=nt)
            mdir = out / "masks"
            mdir.mkdir(exist_ok=True)
            import cv2
            n_ok = 0
            for n in names:
                f = tmp_out / "images" / n
                if not f.is_file():
                    continue
                m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                m = np.where(m >= 128, 255, 0).astype(np.uint8)
                target = mdir / (n + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), m)
                n_ok += 1
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
            res["masks"] = {"dir": str(mdir), "images": n_ok}
    res["seconds"] = round(time.time() - t0, 1)
    rec_file.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log.info("dense: workspace %s - %d images undistorted at <= %d px%s in %.0f s", out, res["images"], max_image_size,
             f", {res['masks']['images']} masks" if res.get("masks") else "", res["seconds"])
    return res


def run_dense(project: SfmProject, workspace: Optional[PathLike] = None, steps=("prepare", "stereo", "fusion"),
              max_image_size: int = DENSE_MAX_IMAGE_SIZE, geom_consistency: bool = True,
              gpu_index: str = "-1") -> Dict[str, Any]:
    """The dense steps from Python: ``prepare`` (:func:`prepare_dense`), ``stereo`` (patch-match stereo; needs a CUDA
    pycolmap), ``fusion`` (``dense/fused.ply``, the masks applied), ``mesh`` (``dense/meshed-poisson.ply``)."""
    import pycolmap
    ws = Path(workspace) if workspace else Path(project.root) / DENSE_DIR
    out: Dict[str, Any] = {"workspace": str(ws), "steps": {}}
    if "prepare" in steps or not (ws / "sparse" / "cameras.bin").is_file():
        out["steps"]["prepare"] = prepare_dense(project, ws, max_image_size=max_image_size)
    if "stereo" in steps:
        if not getattr(pycolmap, "has_cuda", False):
            raise RuntimeError("patch-match stereo needs a CUDA pycolmap (the gpu_python environment) or "
                               "dense_reconstruct.bat with a CUDA COLMAP")
        t0 = time.time()
        po = pycolmap.PatchMatchOptions()
        po.geom_consistency = bool(geom_consistency)
        po.max_image_size = int(max_image_size)
        po.gpu_index = str(gpu_index)
        pycolmap.patch_match_stereo(str(ws), "COLMAP", options=po)
        out["steps"]["stereo"] = {"seconds": round(time.time() - t0, 1), "geom_consistency": bool(geom_consistency)}
    if "fusion" in steps:
        t0 = time.time()
        fo = pycolmap.StereoFusionOptions()
        if (ws / "masks").is_dir():
            fo.mask_path = str(ws / "masks")
        fo.max_image_size = int(max_image_size)
        fused = pycolmap.stereo_fusion(str(ws / "fused.ply"), str(ws), "COLMAP", input_type="geometric" if geom_consistency else "photometric",
                                       options=fo)
        out["steps"]["fusion"] = {"seconds": round(time.time() - t0, 1), "points": fused.num_points3D(),
                                  "ply": str(ws / "fused.ply"), "masks": fo.mask_path or None}
    if "mesh" in steps:
        t0 = time.time()
        pycolmap.poisson_meshing(str(ws / "fused.ply"), str(ws / "meshed-poisson.ply"))
        out["steps"]["mesh"] = {"seconds": round(time.time() - t0, 1), "ply": str(ws / "meshed-poisson.ply")}
    (ws / "mppp_dense_run.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def read_colmap_array(path: PathLike):
    """A COLMAP dense map (``stereo/depth_maps/<image>.{photometric,geometric}.bin``, ``normal_maps``): the text header
    ``width&height&channels&`` then float32 in column-major order; returns (height, width) or (height, width, 3)."""
    import numpy as np
    raw = Path(path).read_bytes()
    k, n = 0, 0
    while n < 3:
        if raw[k:k + 1] == b"&":
            n += 1
        k += 1
    w, h, c = (int(x) for x in raw[:k].decode("ascii").strip("&").split("&"))
    a = np.frombuffer(raw[k:], dtype=np.float32, count=w * h * c).reshape((w, h, c), order="F")
    return np.transpose(a, (1, 0, 2)).squeeze()


MAPS_MAX_DEPTH_M = 100.0           # 0.95.1 (Christian, 29 Oct): depth beyond this is no depth in the map images


def _map_mask(ws: Path, image_name: str, shape) -> Optional[Any]:
    """The rover mask of an image in the workspace (``masks/<image>.png``, 0 = rover: not fused) at ``shape``
    (nearest neighbour when the map's resolution differs); True where the terrain is.  None without a mask."""
    import numpy as np
    from PIL import Image
    f = ws / "masks" / (image_name + ".png")
    if not f.is_file():
        return None
    m = Image.open(f).convert("L")
    if m.size != (shape[1], shape[0]):
        m = m.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(m) > 0


def export_dense_maps(workspace: PathLike, kind: str = "geometric", out_dir: Optional[PathLike] = None,
                      fmt: str = "png", lo_hi: tuple = (2.0, 98.0), max_depth_m: float = MAPS_MAX_DEPTH_M,
                      masks: bool = True) -> Dict[str, Any]:
    """0.95.1 (Christian, 29 Oct: "how can I view those in jpeg or png?"): the depth and normal maps of a dense
    workspace as images in ``<workspace>/stereo/maps_png/`` - a depth map in a colour scale over its own 2-98th
    percentile (near blue, far red) with the range in the file name, a normal map as RGB = (n + 1) / 2 in the camera
    frame (x right, y down, z forward).  No value ("NaN": transparent in a PNG, black in a JPEG) where there is no
    depth, where the depth is beyond ``max_depth_m`` (100 m) and inside the rover mask (``masks/``, when ``masks``);
    the normal map gets the same cut.  ``kind``: "geometric" (after the consistency check, what the fusion uses) or
    "photometric"."""
    import numpy as np
    from PIL import Image
    ws = Path(workspace)
    out = Path(out_dir) if out_dir else ws / "stereo" / "maps_png"
    out.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib import colormaps
        cmap = colormaps["turbo"]
    except Exception:                                      # noqa: BLE001
        cmap = None
    jpeg = fmt.lower() in ("jpg", "jpeg")

    def save(rgb, ok, path):
        if jpeg:
            rgb = rgb.copy()
            rgb[~ok] = 0
            Image.fromarray(rgb).save(path, quality=92)
        else:
            Image.fromarray(np.dstack([rgb, np.where(ok, 255, 0).astype(np.uint8)]), "RGBA").save(path)

    n = {"depth": 0, "normal": 0, "masked": 0}
    for f in sorted((ws / "stereo" / "depth_maps").glob(f"*.{kind}.bin")):
        name = f.name[: -len(f".{kind}.bin")]                     # the image's name (NLF_....png)
        stem = name.rsplit(".", 1)[0] if name.lower().endswith((".png", ".jpg", ".tif")) else name
        d = read_colmap_array(f)
        ok = (d > 0) & (d <= float(max_depth_m))
        m = _map_mask(ws, name, d.shape) if masks else None
        if m is not None:
            ok &= m
            n["masked"] += 1
        if ok.any():
            d0, d1 = np.percentile(d[ok], lo_hi)
            t = np.clip((d - d0) / max(d1 - d0, 1e-9), 0, 1)
            rgb = (cmap(t)[..., :3] * 255).astype(np.uint8) if cmap else np.repeat((t * 255).astype(np.uint8)[..., None], 3, 2)
            save(rgb, ok, out / f"{stem}.depth_{d0:.2f}-{d1:.2f}m.{fmt}")
            n["depth"] += 1
        fn = ws / "stereo" / "normal_maps" / f.name
        if fn.is_file():
            a = read_colmap_array(fn)
            okn = ok & (np.linalg.norm(a, axis=-1) > 0) if a.shape[:2] == ok.shape else (np.linalg.norm(a, axis=-1) > 0)
            rgb = ((np.clip(a, -1, 1) + 1) * 127.5).astype(np.uint8)
            save(rgb, okn, out / f"{stem}.normal.{fmt}")
            n["normal"] += 1
    return {"folder": str(out), "depth_maps": n["depth"], "normal_maps": n["normal"], "with_masks": n["masked"],
            "kind": kind, "max_depth_m": float(max_depth_m)}


def write_dense_bat(project: SfmProject, colmap_bat: Optional[str] = None, max_image_size: int = DENSE_MAX_IMAGE_SIZE) -> Path:
    """``<project>/dense_reconstruct.bat`` (next to ``open_in_colmap.bat``): the dense reconstruction with the COLMAP
    executable - the workspace from ``gui_native/sparse`` and ``images/`` unless it is there, the stereo, the fusion
    with the masks, a Poisson mesh with ``MPPP_DENSE_MESH=1``.  A CUDA COLMAP is looked for first (the stereo
    needs one); the masks need ``mppp dense --prepare`` (pycolmap) or are left out."""
    from .. import __version__
    from .database import COLMAP_BAT_CANDIDATES
    root = Path(project.root).resolve()
    env_bat = Path(__file__).resolve().parents[3] / "scripts" / "windows" / "mppp_env.bat"   # 0.95.1: MPPP's Python
    cands = ([str(colmap_bat or project.settings.get("colmap_bat") or "")] if (colmap_bat or project.settings.get("colmap_bat")) else [])
    cuda_first = [c for c in COLMAP_BAT_CANDIDATES if "cuda" in c.lower() and "nocuda" not in c.lower()]
    rest = [c for c in COLMAP_BAT_CANDIDATES if c not in cuda_first]
    lines = ["@echo off", "setlocal EnableExtensions",
             f"rem MPPP {__version__}: dense reconstruction of this block with COLMAP (4.x) - the workspace dense\\ from the",
             "rem native-pixel model gui_native\\sparse and images\\, patch-match stereo (CUDA), fusion with the rover masks.",
             "rem Double-click.  set MPPP_DENSE_STEP=prepare|stereo|fusion|mesh for one step; MPPP_DENSE_MAX_SIZE (long side,",
             f"rem default {int(max_image_size)} px); MPPP_DENSE_MESH=1 for a Poisson mesh at the end.  The GUI's Reconstruction >",
             "rem Dense reconstruction dialog takes the same dense\\ folder as its workspace (open_in_colmap.bat first).",
             f'set "WS=%~dp0dense"',
             f'if not defined MPPP_DENSE_MAX_SIZE set "MPPP_DENSE_MAX_SIZE={int(max_image_size)}"',
             'if not defined MPPP_DENSE_STEP set "MPPP_DENSE_STEP=all"',
             'set "CB="',
             'if defined COLMAP_BAT if exist "%COLMAP_BAT%" set "CB=%COLMAP_BAT%"']
    for c in cands + cuda_first:
        lines.append(f'if not defined CB if exist "{c}" set "CB={c}"')
    for c in rest:
        lines.append(f'if not defined CB if exist "{c}" set "CB={c}"')
    lines += ['if not defined CB for %%I in (COLMAP.bat) do if not "%%~$PATH:I"=="" set "CB=%%~$PATH:I"',
              'if not defined CB (echo COLMAP.bat not found: setx COLMAP_BAT ^<path^>\\COLMAP.bat  ^(a CUDA build for the stereo^) & pause & exit /b 1)',
              'echo %CB%',
              'echo "%CB%" | find /i "nocuda" >nul && echo WARNING: a COLMAP without CUDA - patch_match_stereo will not run; the workspace can still be prepared.',
              'if not exist "%~dp0gui_native\\sparse\\cameras.bin" (echo No gui_native\\sparse yet: run  mppp dense --prepare  ^(it writes it^) or open_in_colmap.bat after a run with the full export. & pause & exit /b 1)',
              'if /i "%MPPP_DENSE_STEP%"=="stereo" goto :stereo',
              'if /i "%MPPP_DENSE_STEP%"=="fusion" goto :fusion',
              'if /i "%MPPP_DENSE_STEP%"=="mesh" goto :mesh',
              ':prepare',
              'if exist "%WS%\\sparse\\cameras.bin" goto :prepared',
              # 0.95.1 (Christian, 29 Oct: "does dense_reconstruct.bat also undistort the masks?"): MPPP's Python
              # prepares the workspace with the rover masks (mppp dense --prepare --no-bat: this file is not rewritten
              # while it runs); without it, COLMAP's undistorter and no masks
              f'if exist "{env_bat}" call "{env_bat}" >nul 2>nul',
              'python -m mppp version >nul 2>nul && (',
              '  echo %DATE% %TIME%  undistorting images\\ and the rover masks with gui_native\\sparse into %WS% ^(max %MPPP_DENSE_MAX_SIZE% px^)',
              '  python -m mppp dense "%~dp0." --prepare --no-bat --max-image-size %MPPP_DENSE_MAX_SIZE%',
              '  if not errorlevel 1 goto :prepared',
              '  echo mppp dense --prepare failed - COLMAP\'s undistorter instead ^(no masks^)',
              ')',
              '(',
              '  echo %DATE% %TIME%  undistorting images\\ with gui_native\\sparse into %WS% ^(max %MPPP_DENSE_MAX_SIZE% px^)',
              '  call "%CB%" image_undistorter --image_path "%~dp0images" --input_path "%~dp0gui_native\\sparse" --output_path "%WS%" --output_type COLMAP --max_image_size %MPPP_DENSE_MAX_SIZE%',
              '  if errorlevel 1 (echo image_undistorter failed & pause & exit /b 1)',
              '  echo NOTE: the rover masks are not undistorted by this step - run  mppp dense --prepare  for a workspace with masks ^(the fusion then leaves the rover out^).',
              ')',
              ':prepared',
              'if exist "%WS%\\masks" (echo rover masks: %WS%\\masks ^(the fusion leaves the rover out^)) else (echo no masks in %WS%: the fusion keeps the rover)',
              'if /i "%MPPP_DENSE_STEP%"=="prepare" goto :done',
              ':stereo',
              'echo %DATE% %TIME%  patch-match stereo ^(geometric consistency^) - hours on the GPU',
              'call "%CB%" patch_match_stereo --workspace_path "%WS%" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size %MPPP_DENSE_MAX_SIZE%',
              'if errorlevel 1 (echo patch_match_stereo failed ^(a CUDA COLMAP is needed^) & pause & exit /b 1)',
              'if /i "%MPPP_DENSE_STEP%"=="stereo" goto :done',
              ':fusion',
              'set "MASKS="',
              'if exist "%WS%\\masks" set MASKS=--StereoFusion.mask_path "%WS%\\masks"',
              'echo %DATE% %TIME%  stereo fusion -^> %WS%\\fused.ply %MASKS%',
              'call "%CB%" stereo_fusion --workspace_path "%WS%" --workspace_format COLMAP --input_type geometric --output_path "%WS%\\fused.ply" --StereoFusion.max_image_size %MPPP_DENSE_MAX_SIZE% %MASKS%',
              'if errorlevel 1 (echo stereo_fusion failed & pause & exit /b 1)',
              'if /i "%MPPP_DENSE_STEP%"=="fusion" goto :done',
              'if not defined MPPP_DENSE_MESH goto :done',
              ':mesh',
              'echo %DATE% %TIME%  Poisson mesh -^> %WS%\\meshed-poisson.ply',
              'call "%CB%" poisson_mesher --input_path "%WS%\\fused.ply" --output_path "%WS%\\meshed-poisson.ply"',
              'if errorlevel 1 (echo poisson_mesher failed & pause & exit /b 1)',
              ':done',
              'echo %DATE% %TIME%  done: %WS%  ^(fused.ply opens in CloudCompare / MeshLab; File ^> Import model in the COLMAP GUI shows dense\\sparse^)',
              'pause']
    path = root / "dense_reconstruct.bat"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path
