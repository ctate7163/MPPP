"""MPPP -> Nerfstudio (0.95.2, Christian 30 Oct): one pinhole size (2048 x 1536), the frames resampled through
COLMAP's exact model, Mastcam-Z clipped 22 / 16 columns, OpenGL camera-to-world matrices that reproject COLMAP's
points."""
import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
cv2 = pytest.importorskip("cv2")

from mppp.sfm import nerfstudio as NS  # noqa: E402

NL = [2948.943055, 2949.208628, 2573.477811, 1950.413699, 0.047527, -0.016316, 3e-05, -0.0002, 0.008977, -0.008065,
      0.0, 0.0]
Z63 = [8627.12, 8636.17, 900.98, 664.32, 0.469, 0.6, 0.0028, 0.0015, 0.0, 0.0, 0.0, 0.0]


def test_target_cameras():
    t = NS.target_camera("THIN_PRISM_FISHEYE", NL, 5120, 3840, "N")
    assert (t["w"], t["h"]) == (2048, 1536) and t["scale"] == pytest.approx(0.4)
    assert t["fl_x"] == pytest.approx(NL[0] * 0.4) and t["cx"] == pytest.approx(1024 + (NL[2] - 2560) * 0.4)
    q = NS.target_camera("THIN_PRISM_FISHEYE", [v / 4 if i < 4 else v for i, v in enumerate(NL)], 1280, 960, "N")
    assert q["fl_x"] == pytest.approx(t["fl_x"]) and q["cx"] == pytest.approx(t["cx"])      # same camera, same frame
    z = NS.target_camera("FULL_OPENCV", Z63, 1648, 1200, "Z")
    assert z["scale"] == pytest.approx(1.28) and z["window"] == [22.0, 0.0, 1632.0, 1200.0]
    assert z["cx"] == pytest.approx(1024 + (Z63[2] - 827) * 1.28) and z["fl_y"] == pytest.approx(Z63[1] * 1.28)


def _scene(tmp_path):
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import SfmProject
    rng = np.random.default_rng(5)
    rec = pycolmap.Reconstruction()
    cams = [("THIN_PRISM_FISHEYE", [v / 4 if i < 4 else v for i, v in enumerate(NL)], 1280, 960),
            ("FULL_OPENCV", Z63, 1648, 1200)]
    for cid, (m, p, w, h) in enumerate(cams, start=1):
        rec.add_camera(pycolmap.Camera(camera_id=cid, model=m, width=w, height=h, params=np.asarray(p, float)))
        rig = pycolmap.Rig(rig_id=cid)
        rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid))
        rec.add_rig(rig)
    pts = np.c_[rng.uniform(-2, 2, (300, 2)) * [1, 0.3] + [0, 8], rng.uniform(-0.2, 0.2, 300)]
    root = tmp_path / "mars2020_sol_0091_t" / "colmap"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rows = []
    for iid, (cid, C) in enumerate([(1, [0, 0, 1.5]), (1, [0.3, 0, 1.5]), (2, [0, 0.1, 1.6])], 1):
        R = Rotation.from_euler("x", 15, degrees=True).as_matrix() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        t = -R @ np.asarray(C, float)
        fr = pycolmap.Frame(frame_id=iid, rig_id=cid)
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid), id=iid))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        rec.add_frame(fr)
        cam = rec.cameras[cid]
        uv = np.asarray(cam.img_from_cam(pts @ R.T + t), float)
        name = f"{'NLF' if cid == 1 else 'ZL0'}_{iid}.png"
        im = pycolmap.Image(name=name, keypoints=uv, camera_id=cid, image_id=iid)
        im.frame_id = iid
        rec.add_image(im)
        yy, xx = np.mgrid[0:cam.height, 0:cam.width]
        img = ((xx * 0.17 + yy * 0.11) % 251).astype(np.uint8)            # a smooth ramp: value tells the position
        cv2.imwrite(str(root / "images" / name), img)
        mask = np.full((cam.height, cam.width), 255, np.uint8)
        mask[: cam.height // 10] = 0                                         # the top rows masked
        cv2.imwrite(str(root / "masks" / (name + ".png")), mask)
        rows.append({"name": name, "camera_group": "NL" if cid == 1 else "ZL063", "has_mask": True})
    for fid in list(rec.frames):
        rec.register_frame(fid)
    for k in range(len(pts)):
        tr = pycolmap.Track()
        for iid in (1, 2, 3):
            tr.add_element(iid, k)
        rec.add_point3D(pts[k], tr, np.array([10, 20, 30], np.uint8))
    model = root / "gui_native" / "sparse"
    model.mkdir(parents=True)
    rec.write(str(model))
    proj = SfmProject(root, rows, {}, {}, [70.0, -50.0, 0.0], {"world_frame": "site3_enu"})
    proj.save()
    return proj, rec


def test_the_export_resamples_exactly_and_the_poses_reproject(tmp_path):
    proj, rec = _scene(tmp_path)
    res = NS.write_nerfstudio(proj, threads=2)
    out = proj.root / "nerfstudio"
    assert res["frames"] == 3 and not res["errors"] and res["points"] == 300
    tj = json.loads((out / "transforms.json").read_text())
    assert tj["camera_model"] == "OPENCV" and tj["ply_file_path"] == "sparse_pc.ply" and tj["mppp"]["offset_m"] == [70, -50, 0]
    assert (out / "sparse_pc.ply").read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 300")
    for fr in tj["frames"]:
        img = cv2.imread(str(out / fr["file_path"]), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(out / fr["mask_path"]), cv2.IMREAD_GRAYSCALE)
        assert img.shape[:2] == (1536, 2048) and mask.shape == (1536, 2048) and set(np.unique(mask)) <= {0, 255}
        im = rec.images[fr["colmap_image_id"]]
        cam = rec.cameras[im.camera_id]
        c2w = np.array(fr["transform_matrix"])
        for X in [rec.points3D[p].xyz for p in list(rec.points3D)[:40]]:
            # Nerfstudio's pinhole (OpenGL axes) against COLMAP's own projection of the same point
            x, y, z = (np.linalg.inv(c2w) @ np.r_[X, 1.0])[:3]
            u, v = fr["cx"] + fr["fl_x"] * x / -z, fr["cy"] - fr["fl_y"] * y / -z
            if not (2 <= u < 2046 and 2 <= v < 1534):
                continue
            us, vs = np.asarray(cam.img_from_cam((im.cam_from_world() * X).reshape(1, 3)), float)[0]
            if mask[int(v), int(u)] == 0:
                continue
            # the resampled value at (u, v) is the source ramp at COLMAP's projection (bilinear: within 1 grey level)
            want = (np.floor(us - 0.5) * 0.17 + np.floor(vs - 0.5) * 0.11) % 251
            got = float(img[int(v), int(u)])
            assert abs(got - want) < 2.0 or abs(abs(got - want) - 251) < 2.0, (fr["file_path"], got, want)
        if fr["file_path"].startswith("images/ZL0"):
            t = NS.target_camera("FULL_OPENCV", Z63, 1648, 1200, "Z")
            mx, my, ok = NS.remap_tables(cam, t)                             # every pixel from columns 22 to 1631
            assert ok.mean() > 0.99 and mx[ok].min() + 0.5 >= 22 and mx[ok].max() + 0.5 < 1632   # corners: distortion
            assert (mask[:150] == 0).all() and mask[160:].any()             # the masked top rows came along
        else:                                    # the fisheye: the pinhole window is inside the frame, a band masked
            assert mask[:, 1024].min() == 0 and mask[1400, 1024] == 255
