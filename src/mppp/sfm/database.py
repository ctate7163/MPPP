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

import numpy as np

from .project import SfmProject

PathLike = Union[str, Path]


def extract_features(project: SfmProject, max_num_features: int = 8192, max_image_size: int = 3200,
                     use_gpu: Optional[bool] = None, num_threads: int = -1, overwrite: bool = False,
                     domain_size_pooling: bool = False, python: Optional[PathLike] = None) -> Path:
    """
    SIFT at native resolution, with the MPPP masks (keypoints in masked pixels
    are dropped) -> ``features.db``.  Equivalent COLMAP command::

        colmap feature_extractor --database_path features.db --image_path images
            --ImageReader.mask_path masks --ImageReader.camera_model SIMPLE_RADIAL
            --SiftExtraction.max_num_features 8192 --SiftExtraction.max_image_size 3200

    ``python``: run this step in another Python environment, e.g. a conda one
    with a CUDA pycolmap (v0p11, see :mod:`mppp.sfm.gpu`).
    """
    db = project.features_db
    if db.exists() and not overwrite:
        return db
    if python is not None:
        from .gpu import run_step
        return Path(run_step(python, "extract_features", project, max_num_features=max_num_features,
                             max_image_size=max_image_size, use_gpu=use_gpu, num_threads=num_threads,
                             overwrite=overwrite, domain_size_pooling=domain_size_pooling))
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
    if use_gpu and not getattr(pycolmap, "has_cuda", False):
        import warnings
        warnings.warn("this pycolmap build has no CUDA; running on the CPU (or use the COLMAP GUI/CLI with a GPU)")
        use_gpu = False
    device = pycolmap.Device.auto if use_gpu is None else (pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu)
    pycolmap.extract_features(str(db), str(project.images_dir), image_names=names,
                              camera_mode=pycolmap.CameraMode.PER_IMAGE, reader_options=reader,
                              extraction_options=opts, device=device)
    project.settings["features"] = {"max_num_features": max_num_features, "max_image_size": max_image_size,
                                    "domain_size_pooling": domain_size_pooling, "masks": bool(reader.mask_path)}
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
    return summary
