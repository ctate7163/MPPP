"""
Features and the COLMAP database.

Features are extracted at each image's native resolution into ``features.db``
(pycolmap here, or the COLMAP GUI/CLI with GPU; any camera settings — only the
keypoints and descriptors are used).  ``build_database`` then writes the final
``database.db``: the two full-resolution cameras, the rig, one frame per
exposure, the images, the keypoints scaled to full-resolution pixels, the
descriptors, and one position prior per image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json

import numpy as np

from .project import SfmProject

PathLike = Union[str, Path]


DEFAULT_MAX_NUM_FEATURES = 16380          # v0p14.3 (was 8192)
DEFAULT_MAX_IMAGE_SIZE = 5120             # v0p20 (was 3200): full-resolution Navcam frames are 5120 px wide, so SIFT
                                          # really runs at native resolution (3200 shrank them to 0.625x)


def _features_record(project: SfmProject) -> Path:
    return project.features_db.with_suffix(".json")


def _feature_settings(max_num_features: int, max_image_size: int, domain_size_pooling: bool,
                      estimate_affine_shape: bool = False) -> Dict[str, Any]:
    out = {"max_num_features": int(max_num_features), "max_image_size": int(max_image_size),
           "domain_size_pooling": bool(domain_size_pooling)}
    if estimate_affine_shape:                            # recorded only when on: older features.json stay valid
        out["estimate_affine_shape"] = True
    return out


def image_fingerprints(project: SfmProject) -> Dict[str, list]:
    """name -> [image size, image mtime_ns, mask size, mask mtime_ns] of the project's image and mask files."""
    out: Dict[str, list] = {}
    for r in project.images:
        f = []
        for path in (project.images_dir / r["name"], project.masks_dir / (r["name"] + ".png")):
            try:
                st = path.stat()
                f += [st.st_size, st.st_mtime_ns]
            except OSError:
                f += [None, None]
        out[r["name"]] = f
    return out


def features_up_to_date(project: SfmProject, max_num_features: int = DEFAULT_MAX_NUM_FEATURES,
                        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE, domain_size_pooling: bool = False,
                        estimate_affine_shape: bool = False) -> bool:
    """
    True if ``features.db`` exists and was extracted with these settings from
    the same image and mask files (``features.json`` beside it; v0p14.7: file
    sizes and modification times, so images processed again - e.g. with
    another mask model - are extracted again).
    """
    rec = _features_record(project)
    if not project.features_db.exists() or not rec.is_file():
        return False
    try:
        done = json.loads(rec.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    want = _feature_settings(max_num_features, max_image_size, domain_size_pooling, estimate_affine_shape)
    names = {r["name"] for r in project.images}
    if bool(done.get("estimate_affine_shape", False)) != bool(estimate_affine_shape):
        return False
    if not (all(done.get(k) == v for k, v in want.items()) and names <= set(done.get("images", []))):
        return False
    files = done.get("files")
    if not isinstance(files, dict):
        return False                                   # recorded before v0p14.7: file state unknown
    now = image_fingerprints(project)
    return all(files.get(n) == now[n] for n in names)


def extract_features(project: SfmProject, max_num_features: int = DEFAULT_MAX_NUM_FEATURES, max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
                     use_gpu: Optional[bool] = None, num_threads: int = -1, overwrite: bool = False,
                     domain_size_pooling: bool = False, python: Optional[PathLike] = None,
                     estimate_affine_shape: bool = False) -> Path:
    """
    SIFT at native resolution, with the MPPP masks (keypoints in masked pixels
    are dropped) -> ``features.db``.  Equivalent COLMAP command::

        colmap feature_extractor --database_path features.db --image_path images
            --ImageReader.mask_path masks --ImageReader.camera_model SIMPLE_RADIAL
            --SiftExtraction.max_num_features 16380 --SiftExtraction.max_image_size 5120

    An existing ``features.db`` is reused only if ``features.json`` beside it
    records the same settings and covers every project image (v0p14.3);
    otherwise - e.g. after changing ``max_num_features`` - it is extracted
    again.  ``overwrite=True`` always extracts again.

    ``python``: run this step in another Python environment, e.g. a conda one
    with a CUDA pycolmap (v0p11, see :mod:`mppp.sfm.gpu`).

    ``domain_size_pooling`` (DSP-SIFT) and ``estimate_affine_shape`` (v0p22):
    descriptors pooled over several scales, and affine-adapted keypoint
    regions - both make SIFT more tolerant of the perspective change between
    strongly converging views (notebook 03, "high convergence").  COLMAP runs
    either on the CPU only (several times slower than the GPU).
    """
    db = project.features_db
    if not overwrite and features_up_to_date(project, max_num_features, max_image_size, domain_size_pooling,
                                             estimate_affine_shape):
        return db
    if db.exists() and not overwrite:
        print(f"[sfm] {db.name}: extracted with other settings or from other image/mask files - extracting again "
              f"with max_num_features={max_num_features}", flush=True)
        overwrite = True
    if python is not None:
        from .gpu import run_step
        return Path(run_step(python, "extract_features", project, max_num_features=max_num_features,
                             max_image_size=max_image_size, use_gpu=use_gpu, num_threads=num_threads,
                             overwrite=overwrite, domain_size_pooling=domain_size_pooling,
                             **({"estimate_affine_shape": True} if estimate_affine_shape else {})))
    import pycolmap
    if db.exists():
        db.unlink()
    names = [r["name"] for r in project.images]
    reader = pycolmap.ImageReaderOptions()
    if any(r.get("has_mask") for r in project.images):
        reader.mask_path = str(project.masks_dir)
    opts = pycolmap.FeatureExtractionOptions()
    opts.max_image_size = int(max_image_size)
    opts.num_threads = int(num_threads)
    opts.sift.max_num_features = int(max_num_features)
    opts.sift.domain_size_pooling = bool(domain_size_pooling)
    opts.sift.estimate_affine_shape = bool(estimate_affine_shape)
    if (domain_size_pooling or estimate_affine_shape) and use_gpu is not False:
        if use_gpu:
            print("[sfm] DSP / affine shape: COLMAP extracts these on the CPU", flush=True)
        use_gpu = False
    if use_gpu and not getattr(pycolmap, "has_cuda", False):
        import warnings
        warnings.warn("this pycolmap build has no CUDA; running on the CPU (or use the COLMAP GUI/CLI with a GPU)")
        use_gpu = False
    device = pycolmap.Device.auto if use_gpu is None else (pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu)
    pycolmap.extract_features(str(db), str(project.images_dir), image_names=names,
                              camera_mode=pycolmap.CameraMode.PER_IMAGE, reader_options=reader,
                              extraction_options=opts, device=device)
    project.settings["features"] = {"max_num_features": max_num_features, "max_image_size": max_image_size,
                                    "domain_size_pooling": domain_size_pooling,
                                    "estimate_affine_shape": bool(estimate_affine_shape), "masks": bool(reader.mask_path)}
    record = dict(_feature_settings(max_num_features, max_image_size, domain_size_pooling, estimate_affine_shape),
                  images=sorted(names),
                  masks=bool(reader.mask_path), files=image_fingerprints(project))
    _features_record(project).write_text(json.dumps(record, indent=1), encoding="utf-8")
    project.save()
    return db


def scale_keypoints(kp: np.ndarray, factor: float) -> np.ndarray:
    """
    Native -> full-resolution pixels (``factor`` = 1 / downsample scale).  COLMAP
    keypoints use a corner pixel origin, so a pure scale is exact for binned
    images.  Columns: x, y [, a11, a12, a21, a22] (affine shape, scales too) or
    x, y, scale, orientation (scale scales).
    """
    kp = np.array(kp, dtype=np.float32, copy=True)
    if kp.shape[1] == 4:
        kp[:, :3] *= factor
    else:
        kp[:, :] *= factor
    return kp


def build_database(project: SfmProject, features_db: Optional[PathLike] = None,
                   database: Optional[PathLike] = None, overwrite: bool = True) -> Dict[str, Any]:
    """
    Write ``database.db`` (see module docstring).  Returns a summary.
    ``overwrite`` (default True since v0p13, so that rerunning the notebook
    does not stop): an existing ``database.db`` - and the matches in it - is
    replaced; matching has to be run again afterwards.
    """
    import pycolmap
    fdb_path = Path(features_db) if features_db else project.features_db
    out = Path(database) if database else project.database
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists (overwrite=True replaces it; matches in it are lost)")
        try:
            out.unlink()
        except PermissionError as e:                           # Windows: still open in COLMAP or this kernel
            raise PermissionError(f"cannot replace {out}: it is open elsewhere (COLMAP GUI, or a pycolmap "
                                  f"Database in this kernel - close it or restart the kernel)") from e
    fdb = pycolmap.Database.open(str(fdb_path))
    by_name = {im.name: im.image_id for im in fdb.read_all_images()}
    missing = [r["name"] for r in project.images if r["name"] not in by_name]
    if missing:
        raise KeyError(f"{len(missing)} project images have no features in {fdb_path.name}, e.g. {missing[0]}")

    db = pycolmap.Database.open(str(out))
    sensor = lambda cid: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid)       # noqa: E731

    # cameras: one per instrument at full resolution
    cam_ids: Dict[str, int] = {}
    for instr, c in sorted(project.cameras.items()):
        cam = pycolmap.Camera(model=c["model"], width=c["width"], height=c["height"], params=c["params"])
        cam.has_prior_focal_length = True
        cam_ids[instr] = db.write_camera(cam)

    # rigs: stereo families -> left reference + right sensor; any other camera -> trivial rig
    rig_of: Dict[str, int] = {}
    ref_of_rig: Dict[int, str] = {}
    for fam, r in sorted(project.rig.items()):
        if r["ref"] not in cam_ids:
            continue
        rig = pycolmap.Rig()
        rig.add_ref_sensor(sensor(cam_ids[r["ref"]]))
        if r["sensor"] in cam_ids:
            T = pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(r["R_sensor_from_ref"], float)),
                                 np.asarray(r["t_sensor_from_ref"], float))
            rig.add_sensor(sensor(cam_ids[r["sensor"]]), T)
        rid = db.write_rig(rig)
        rig_of[r["ref"]] = rid
        ref_of_rig[rid] = r["ref"]
        if r["sensor"] in cam_ids:
            rig_of[r["sensor"]] = rid
    for instr, cid in cam_ids.items():
        if instr not in rig_of:
            rig = pycolmap.Rig()
            rig.add_ref_sensor(sensor(cid))
            rig_of[instr] = db.write_rig(rig)
            ref_of_rig[rig_of[instr]] = instr

    # frames = exposures; a frame must contain its rig's reference sensor, so an
    # exposure without it (e.g. a right image whose left is missing) is left out
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in project.images:
        groups.setdefault((rig_of[r["instrument"]], r["sclk_key"]), []).append(r)
    skipped = [r["name"] for (rid, _), rs in groups.items() if ref_of_rig[rid] not in {x["instrument"] for x in rs}
               for r in rs]

    # images, keypoints, descriptors, priors
    sig = np.asarray(project.settings.get("prior_sigma_m", [1.0, 1.0, 1.0]), float)
    cov = np.diag(sig ** 2)
    n_kp = 0
    for r in project.images:
        if r["name"] in skipped:
            r.pop("image_id", None)
            continue
        fid = by_name[r["name"]]
        cid = cam_ids[r["instrument"]]
        iid = db.write_image(pycolmap.Image(name=r["name"], camera_id=cid))
        kp = scale_keypoints(fdb.read_keypoints(fid), 1.0 / float(r["downsample_scale"]))
        db.write_keypoints(iid, kp)
        db.write_descriptors(iid, fdb.read_descriptors(fid))
        n_kp += kp.shape[0]
        db.write_pose_prior(pycolmap.PosePrior(
            position=np.asarray(r["prior_C"], float), position_covariance=cov,
            coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
            corr_data_id=pycolmap.data_t(sensor_id=sensor(cid), id=iid)))
        r["image_id"], r["camera_id"] = int(iid), int(cid)

    for (rid, _), rs in sorted(groups.items(), key=lambda kv: kv[0][1]):
        if rs[0]["name"] in skipped:
            continue
        fr = pycolmap.Frame()
        fr.rig_id = rid
        for r in rs:
            fr.add_data_id(pycolmap.data_t(sensor_id=sensor(r["camera_id"]), id=r["image_id"]))
        fid = db.write_frame(fr)
        for r in rs:
            r["frame_id"] = int(fid)
    db.close()
    fdb.close()
    summary = {"database": str(out), "cameras": cam_ids, "rigs": len(set(rig_of.values())),
               "frames": len({r.get('frame_id') for r in project.images if r.get('frame_id')}),
               "images": len(project.images) - len(skipped), "keypoints": int(n_kp), "images_without_frame": skipped}
    project.settings["database"] = summary
    project.save()
    try:
        summary["gui"] = write_gui_project(project, model=None if not (project.root / "sparse" / "cahv_ba").is_dir()
                                           else "cahv_ba")
    except OSError:                                        # read-only or odd paths: the files are a convenience
        pass
    return summary


def write_gui_project(project: SfmProject, model: Optional[str] = "cahv_ba") -> Dict[str, str]:
    """
    Files to open the project in the COLMAP GUI (v0p14.4, v0p14.5), in the project folder:

    * ``open_in_colmap.bat`` - Windows, double-click: opens the GUI with the
      NATIVE-pixel viewing copy ``gui_native/`` (database + refined model, see
      :func:`mppp.sfm.export.write_gui_native`), in which keypoints, tie points
      and matches line up with the images; before a reconstruction exists it
      opens the full-resolution project.  It calls ``COLMAP.bat`` from the
      environment variable ``COLMAP_BAT`` (set it once, e.g.
      ``setx COLMAP_BAT D:\\tools\\COLMAP\\COLMAP.bat``), else from the PATH;
    * ``open_in_colmap_fullres.bat`` - the full-resolution project and
      ``sparse/<model>`` (what the bundle adjustment works on: keypoints are in
      full-resolution pixels, so on half/quarter-resolution images the GUI's 2-D
      overlay is off by 2x/4x; the 3-D view is right);
    * ``colmap_gui.ini`` / ``gui_native/colmap_gui.ini`` - COLMAP project files
      for File > Open project (then File > Import model).

    Rewritten by ``build_database`` and ``reconstruct``.  Returns the paths.
    """
    from .. import __version__
    root = Path(project.root).resolve()
    native = root / "gui_native"
    has_native = (native / "sparse" / "cameras.bin").is_file() and (native / "database.db").is_file()

    def ini(path: Path, db: Path, model_dir: Optional[Path], note: str) -> None:
        lines = [f"# MPPP {__version__}: COLMAP GUI project ({note}). File > Open project (this file)"
                 + (f", then File > Import model > {model_dir}" if model_dir else ""),
                 f"database_path={db}", f"image_path={project.images_dir.resolve()}"]
        if project.masks_dir.is_dir() and any(r.get("has_mask") for r in project.images):
            lines += ["", "[ImageReader]", f"mask_path={project.masks_dir.resolve()}"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def bat(path: Path, db: str, model_rel: str, note: str) -> None:
        lines = ["@echo off",
                 f"rem MPPP {__version__}: open this project in the COLMAP GUI (4.x): {note}.",
                 "rem Set COLMAP_BAT once to your COLMAP.bat, e.g.  setx COLMAP_BAT D:\\tools\\COLMAP\\COLMAP.bat",
                 'if "%COLMAP_BAT%"=="" set "COLMAP_BAT=COLMAP.bat"',
                 f'set "MODEL=%~dp0{model_rel}"',
                 'if not exist "%MODEL%\\cameras.bin" (echo No model in %MODEL% yet: run the reconstruction first. & pause & exit /b 1)',
                 f'call "%COLMAP_BAT%" gui --database_path "%~dp0{db}" --image_path "%~dp0images" --import_path "%MODEL%"',
                 "if errorlevel 1 (echo Could not start COLMAP: set COLMAP_BAT to your COLMAP.bat. & pause)"]
        path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))

    m = model or "cahv_ba"
    ini(root / "colmap_gui.ini", project.database.resolve(), root / "sparse" / m if model else None,
        "full-resolution keypoints")
    bat(root / "open_in_colmap_fullres.bat", "database.db", f"sparse\\{m}",
        f"full-resolution project and sparse\\{m}")
    out = {"ini": str(root / "colmap_gui.ini"), "bat_fullres": str(root / "open_in_colmap_fullres.bat")}
    if has_native:
        ini(native / "colmap_gui.ini", (native / "database.db").resolve(), (native / "sparse").resolve(),
            "native pixels, for viewing")
        bat(root / "open_in_colmap.bat", "gui_native\\database.db", "gui_native\\sparse",
            "native-pixel viewing copy gui_native (keypoints line up with the images)")
        out["ini_native"] = str(native / "colmap_gui.ini")
    else:
        bat(root / "open_in_colmap.bat", "database.db", f"sparse\\{m}",
            f"full-resolution project and sparse\\{m} (no gui_native yet)")
    out["bat"] = str(root / "open_in_colmap.bat")
    return out
