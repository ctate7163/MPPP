"""Shared test helpers (v0p50): synthetic blocks and reconstructions, tiny mask checkpoints and datasets."""
from __future__ import annotations

import numpy as np
import pytest
from mppp.paths import data_dir  # noqa: E402
import cv2


try:                                   # only the SfM helpers need it
    import pycolmap
except ImportError:                    # pragma: no cover
    pycolmap = None


DATA_DIR = data_dir()


def _metashape_project(xyz, c):
    """Metashape frame model, corner pixel origin."""
    x, y = xyz[0] / xyz[2], xyz[1] / xyz[2]
    r2 = x * x + y * y
    rad = 1 + c["k1"] * r2 + c["k2"] * r2 ** 2 + c["k3"] * r2 ** 3
    xp, yp = x * rad, y * rad
    return np.array([c["width"] * 0.5 + c["cx"] + xp * c["f"], c["height"] * 0.5 + c["cy"] + yp * c["f"]])


def _synthetic(tmp_path, noise_native_px=0.3, seed=0):
    """Two stations, a mast pan of 6 stereo exposures each (half and quarter resolution), terrain points."""
    from mppp.sfm.project import SfmProject, camera_from_metashape_xml
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    cams = {f"N{e}": camera_from_metashape_xml(DATA_DIR / f"cmods/M2020_N{e}0_frame.xml") for e in "LR"}
    R_rel = Rotation.from_rotvec(np.radians([-0.09, 0.015, -0.08])).as_matrix()
    t_rel = np.array([-0.4244, 0.0, 0.0])
    images, truth = [], {}
    stations = {"S001D0000": np.array([0.0, 0.0, 1.9]), "S001D0100": np.array([4.0, 1.0, 1.9])}
    k = 0
    for sname, C0 in stations.items():
        for az in np.linspace(0, 150, 6):
            k += 1
            # camera: z forward along azimuth, pitched 35 deg down; x right, y down
            a = np.radians(az)
            fwd = np.array([np.sin(a), np.cos(a), 0.0]) * np.cos(np.radians(35)) + np.array([0, 0, -np.sin(np.radians(35))])
            right = np.array([np.cos(a), -np.sin(a), 0.0])
            down = np.cross(fwd, right)
            RL = np.stack([right, down, fwd])
            CL = C0 + rng.normal(0, 0.02, 3)
            RR = R_rel @ RL
            CR = CL - RR.T @ t_rel
            s = 0.5 if k % 3 else 0.25
            for eye, R, C in (("L", RL, CL), ("R", RR, CR)):
                name = f"N{eye}F_{k:04d}.png"
                images.append({"name": name, "stem": name[:-4], "instrument": f"N{eye}", "eye": eye,
                               "sclk_key": f"{k:010d}_000", "sol": 1, "site": 1, "drive": int(sname[-4:]),
                               "station": sname, "sequence": "NCAM00000", "downsample_scale": s,
                               "native_size": [int(5120 * s), int(3840 * s)], "prior_C": C.tolist(),
                               "prior_R_w2c": R.tolist(), "has_mask": False})
                truth[name] = (R, C)
    rig = {"N": {"ref": "NL", "sensor": "NR", "n_pairs": k, "R_sensor_from_ref": R_rel.tolist(),
                 "t_sensor_from_ref": t_rel.tolist(), "baseline_m": 0.4244}}
    proj = SfmProject(tmp_path, images, cams, rig, [0, 0, 0], {"prior_sigma_m": [0.05, 0.05, 0.05]})
    proj.root.mkdir(parents=True, exist_ok=True)
    # terrain points around both stations
    P = np.column_stack([rng.uniform(-12, 16, 4000), rng.uniform(-10, 14, 4000), rng.normal(0, 0.3, 4000)])
    return proj, truth, P, cams, (R_rel, t_rel), noise_native_px, rng


def _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=True):
    """Reconstruction with true tracks, noisy native-resolution keypoints, perturbed start."""
    R_rel, t_rel = rigT
    rec = pycolmap.Reconstruction()
    cam_id = {"NL": 1, "NR": 2}
    true_params = {}
    for instr, cid in cam_id.items():
        p = np.array(cams_d[instr]["params"], float)
        true_params[cid] = p.copy()
        q = p.copy()
        if perturb:
            q[0] *= 1.003; q[1] *= 1.003; q[2] += 3; q[3] -= 2; q[4] += 0.004
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="FULL_OPENCV", width=5120, height=3840, params=q))
    rig = pycolmap.Rig(rig_id=1)
    sens = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)        # noqa: E731
    rig.add_ref_sensor(sens(1))
    rig.add_sensor(sens(2), pycolmap.Rigid3d(pycolmap.Rotation3d(R_rel), t_rel))
    rec.add_rig(rig)
    by_frame = {}
    for i, r in enumerate(proj.images, start=1):
        r["image_id"] = i
        by_frame.setdefault(r["sclk_key"], []).append(r)
    true_cams = {cid: pycolmap.Camera(camera_id=cid, model="FULL_OPENCV", width=5120, height=3840,
                                      params=true_params[cid]) for cid in (1, 2)}
    obs = {r["image_id"]: [] for r in proj.images}
    for fid, (key, rs) in enumerate(sorted(by_frame.items()), start=1):
        fr = pycolmap.Frame(frame_id=fid, rig_id=1)
        L = [r for r in rs if r["eye"] == "L"][0]
        R0, C0 = truth[L["name"]]
        if perturb:
            from scipy.spatial.transform import Rotation
            dR = Rotation.from_rotvec(rng.normal(0, 2e-3, 3)).as_matrix()
            R0, C0 = dR @ R0, C0 + rng.normal(0, 0.03, 3)
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R0), -R0 @ C0)
        for r in rs:
            fr.add_data_id(pycolmap.data_t(sensor_id=sens(cam_id[r["instrument"]]), id=r["image_id"]))
        rec.add_frame(fr)
        for r in rs:
            r["frame_id"] = fid
    pts = []
    for r in proj.images:
        R, C = truth[r["name"]]
        cam = true_cams[cam_id[r["instrument"]]]
        Xc = (P - C) @ R.T
        ok = Xc[:, 2] > 0.5
        uv = np.full((len(P), 2), np.nan)
        uv[ok] = cam.img_from_cam(Xc[ok])
        inside = ok & (uv[:, 0] > 50) & (uv[:, 0] < 5070) & (uv[:, 1] > 50) & (uv[:, 1] < 3790)
        s = r["downsample_scale"]
        kp = []
        for j in np.where(inside)[0]:
            native = uv[j] * s + rng.normal(0, noise, 2)
            kp.append(native / s)
            obs[r["image_id"]].append(j)
        im = pycolmap.Image(name=r["name"], keypoints=np.array(kp, float).reshape(-1, 2),
                            camera_id=cam_id[r["instrument"]], image_id=r["image_id"])
        im.frame_id = r["frame_id"]
        rec.add_image(im)
    for fid in list(rec.frames):
        rec.register_frame(fid)
    track_of = {}
    for iid, js in obs.items():
        for k2, j in enumerate(js):
            track_of.setdefault(j, []).append((iid, k2))
    for j, els in track_of.items():
        if len(els) < 2:
            continue
        tr = pycolmap.Track()
        for iid, k2 in els:
            tr.add_element(iid, k2)
        X = P[j] + (rng.normal(0, 0.05, 3) if perturb else 0)
        pid = rec.add_point3D(X, tr)
        for iid, k2 in els:
            rec.images[iid].set_point3D_for_point2D(k2, pid)
    return rec, true_params


def _look_at(C, target):
    """World-to-camera rotation for a camera at C looking at target (y down)."""
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _quat(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _synthetic_model(noise_px=0.0, seed=0, full_keypoint_lists=True):
    """Two stations 6 m apart, three cameras each, 200 terrain points seen by all six."""
    from mppp.colmap import project_camera
    from mppp.error.colmap import ColmapCamera, ColmapImage, ColmapModel, ColmapPoint
    rng = np.random.default_rng(seed)
    cam = ColmapCamera(1, "FULL_OPENCV", 1280, 960, np.array([1000, 1000, 640, 480, -0.2, 0.05, 1e-4, -1e-4,
                                                               0, 0, 0, 0], float))
    X = np.c_[rng.uniform(-3, 3, 200), rng.uniform(8, 12, 200), rng.uniform(-0.3, 0.3, 200)]
    centres = [np.array([sx + dx, 0, 2.0]) for sx in (-3.0, 3.0) for dx in (-0.2, 0.0, 0.2)]
    images, obs = {}, {}
    for k, C in enumerate(centres, start=1):
        R = _look_at(C, [0, 10, 0])
        t = -R @ C
        uv = project_camera(cam.model, cam.params, X @ R.T + t) + rng.normal(0, noise_px, (len(X), 2))
        images[k] = ColmapImage(k, _quat(R), t, 1, f"img{k}.png", uv, np.arange(1, len(X) + 1),
                                station="A" if k <= 3 else "B")
    points = {j + 1: ColmapPoint(j + 1, X[j], np.zeros(3), 0.0, np.arange(1, 7), np.full(6, j)) for j in range(len(X))}
    if not full_keypoint_lists:                     # pre-0.14.5 exports: track indices do not match the lists
        for p in points.values():
            p.point2D_idxs = np.zeros(6, int)
    return ColmapModel({1: cam}, images, points), X


def _synthetic_block(tmp_path):
    pytest.importorskip("pyceres")
    pass  # (helpers in this module)
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    for i, r in enumerate(proj.images):
        r["camera_temperature_degC"] = -32.0 if r["station"].endswith("0000") else -14.0
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    return proj, rec, noise


def _tiny_checkpoint(tmp_path):
    import torch
    from mppp.mask.model import ConvNeXtSeg, write_card
    m = ConvNeXtSeg("convnext_tiny", pretrained=False, fpn_width=32, stride4=True)
    ck = tmp_path / "tiny_s4_seg_test.pt"
    torch.save({"model": m.state_dict(), "val_iou": 0.5, "epoch": 1}, ck)
    write_card(ck, backbone="convnext_tiny", fpn_width=32, stride4=True, threshold=0.5, canvas=[64, 64],
               input_size=64, val_iou=0.5, name=ck.stem)
    return ck


def _make_dataset(root, n_scenes=6, size=64):
    rng = np.random.default_rng(0)
    for d in ("images", "images_variable", "masks"):
        (root / d).mkdir(parents=True)
    for k in range(n_scenes):
        img = rng.integers(0, 255, (size, int(size * 1.3), 3), dtype=np.uint8)
        m = np.zeros(img.shape[:2], np.uint8)
        m[size // 3:, :] = 255                                        # "terrain" below a horizon
        img[m > 0] = (img[m > 0] * 0.3 + 120).astype(np.uint8)
        name = f"scene{k}.png"
        cv2.imwrite(str(root / "images" / name), img)
        cv2.imwrite(str(root / "images_variable" / name), np.clip(img * 1.5, 0, 255).astype(np.uint8))
        cv2.imwrite(str(root / "masks" / name), m)


def _zcam_rig_rec(tmp_path):
    """v0p62: the synthetic block with its second half of frames on a two-camera Mastcam-Z rig (cameras 3, 4,
    rig 2), as in test_staged_reconstruct_with_a_zcam_stereo_rig.  Returns (project, rec, z frame ids)."""
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    fids = sorted(rec0.frames)
    zf = set(fids[len(fids) // 2:])
    new = pycolmap.Reconstruction()
    for cid, c in rec0.cameras.items():
        new.add_camera(c)
        new.add_camera(pycolmap.Camera(camera_id=cid + 2, model=c.model, width=c.width, height=c.height, params=c.params))
    new.add_rig(rec0.rigs[1])
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    r2 = pycolmap.Rig(rig_id=2)
    r2.add_ref_sensor(S(3))
    r2.add_sensor(S(4), rec0.rigs[1].sensor_from_rig(S(2)))
    new.add_rig(r2)
    for fid in fids:
        fr = rec0.frames[fid]
        z = fid in zf
        nf = pycolmap.Frame(frame_id=fid, rig_id=2 if z else 1)
        for d in fr.data_ids:
            nf.add_data_id(pycolmap.data_t(sensor_id=S(d.sensor_id.id + (2 if z else 0)), id=d.id))
        nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
        for d in fr.data_ids:
            im = rec0.images[d.id]
            ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]),
                                camera_id=im.camera_id + (2 if z else 0), image_id=d.id)
            ni.frame_id = fid
            new.add_image(ni)
        new.register_frame(fid)
    for pid, p in rec0.points3D.items():
        new.add_point3D(p.xyz, p.track, np.zeros(3, np.uint8))
    zimg = {d.id for f in zf for d in rec0.frames[f].data_ids}
    for r in proj.images:
        if r["image_id"] in zimg:
            r["instrument"] = "Z" + r["instrument"][1:]
            r["camera_group"] = r["instrument"]
    for k in ("NL", "NR"):
        proj.cameras["Z" + k[1:]] = dict(proj.cameras[k])
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2, "ZL": 3, "ZR": 4}}
    return proj, new, sorted(zf)


def _write_database(rec, path, descriptors=False, keep_pair=None, seed=0):
    """v0p62: a COLMAP database with the cameras, rigs, frames, images and keypoints of ``rec`` and, as verified
    two-view geometries, the matches its tracks imply - enough for ``pycolmap.triangulate_points``.  v0p64:
    ``descriptors`` writes SIFT-like descriptors (one random vector per 3-D point plus noise per observation, random
    for keypoints without a point); ``keep_pair(a, b)`` False leaves that pair's matches out."""
    import itertools
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = pycolmap.Database.open(str(path))
    for c in rec.cameras.values():
        db.write_camera(c, use_camera_id=True)
    for r in rec.rigs.values():
        db.write_rig(r, use_rig_id=True)
    for f in rec.frames.values():
        nf = pycolmap.Frame(frame_id=f.frame_id, rig_id=f.rig_id)
        for d in f.data_ids:
            nf.add_data_id(d)
        db.write_frame(nf, use_frame_id=True)
    rng = np.random.default_rng(seed)
    desc = {i: rng.normal(size=(im.num_points2D(), 128)) for i, im in rec.images.items()} if descriptors else {}
    if descriptors:
        for pt in rec.points3D.values():
            base = rng.normal(size=128) * 3.0
            for e in pt.track.elements:
                desc[e.image_id][e.point2D_idx] = base + rng.normal(size=128) * 0.6
    for i, im in rec.images.items():
        ni = pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=i)
        ni.frame_id = im.frame_id
        db.write_image(ni, use_image_id=True)
        db.write_keypoints(i, np.array([q.xy for q in im.points2D], np.float32).reshape(-1, 2))
        if descriptors:
            d = np.abs(desc[i])
            d = d / np.linalg.norm(d, axis=1, keepdims=True) * 512
            db.write_descriptors(i, pycolmap.FeatureDescriptors(pycolmap.FeatureExtractorType.SIFT,
                                                                np.clip(d, 0, 255).astype(np.uint8)))
    pairs = {}
    for pt in rec.points3D.values():
        els = [(e.image_id, e.point2D_idx) for e in pt.track.elements]
        for (a, ia), (b, ib) in itertools.combinations(els, 2):
            if a > b:
                a, ia, b, ib = b, ib, a, ia
            pairs.setdefault((a, b), []).append((ia, ib))
    for (a, b), m in pairs.items():
        if keep_pair is not None and not keep_pair(a, b):
            continue
        tvg = pycolmap.TwoViewGeometry()
        tvg.config = 2
        tvg.inlier_matches = np.array(m, np.uint32)
        db.write_matches(a, b, np.array(m, np.uint32))
        db.write_two_view_geometry(a, b, tvg)
    db.close()
