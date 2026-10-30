"""MPPP -> Metashape (0.95.2, Christian 30 Oct): the camera models translate exactly, the frames and the Bundler tie
points line up with COLMAP's observations."""
import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from mppp.sfm import metashape as MS  # noqa: E402
from mppp.sfm import metashape_build as MB  # noqa: E402

NL = [2948.943055, 2949.208628, 2573.477811, 1950.413699, 0.047527, -0.016316, 3e-05, -0.0002, 0.008977, -0.008065,
      0.0, 0.0]                                                   # a Navcam camera of Chal Rocks (gui_native)
Z63 = [8627.12, 8636.17, 900.98, 664.32, 0.469, 0.6, 0.0028, 0.0015, 0.0, 0.0, 0.0, 0.0]   # Mastcam-Z 63 mm


def _rays(w, h, half_fov):
    rng = np.random.default_rng(0)
    t = np.tan(half_fov)
    return np.c_[rng.uniform(-t, t, (3000, 2)) * [1, h / w], np.ones(3000)]


@pytest.mark.parametrize("model,params,w,h,fov", [("THIN_PRISM_FISHEYE", NL, 5120, 3840, 0.9),
                                                  ("FULL_OPENCV", Z63, 1648, 1200, 0.1),
                                                  ("OPENCV", Z63[:8], 1648, 1200, 0.1),
                                                  ("OPENCV_FISHEYE", NL[:4] + [0.05, -0.01, 0.009, -0.008], 5120, 3840, 0.9),
                                                  ("PINHOLE", NL[:4], 5120, 3840, 0.6)])
def test_the_camera_models_translate_exactly(model, params, w, h, fov):
    cal = MS.metashape_calibration(model, params, w, h)
    assert cal["exact"] and not cal["lost"] and cal["max_error_px"] < 1e-8
    cam = pycolmap.Camera(model=model, width=w, height=h, params=np.asarray(params, float))
    X = _rays(w, h, fov)
    uv = np.asarray(cam.img_from_cam(X), float)
    assert np.abs(MS.metashape_project_px(cal, X) - uv).max() < 1e-8
    if model in ("THIN_PRISM_FISHEYE", "FULL_OPENCV"):            # the conventions, spelled out
        assert cal["type"] == ("fisheye" if model == "THIN_PRISM_FISHEYE" else "frame")
        assert cal["f"] == params[1] and cal["b1"] == pytest.approx(params[0] - params[1])
        assert cal["cx"] == pytest.approx(params[2] - w / 2) and cal["cy"] == pytest.approx(params[3] - h / 2)
        assert (cal["p1"], cal["p2"]) == (params[7], params[6])   # Metashape P1 = COLMAP p2


def test_terms_metashape_has_no_place_for_are_fitted_and_reported():
    p = list(Z63)
    p[9] = 0.02                                                   # k4: the rational denominator
    cal = MS.metashape_calibration("FULL_OPENCV", p, 1648, 1200)
    assert not cal["exact"] and cal["lost"] == ["k4"] and cal["max_error_px"] < 0.05
    q = list(NL)
    q[10] = 2e-4                                                  # sx1: thin prism
    cal = MS.metashape_calibration("THIN_PRISM_FISHEYE", q, 5120, 3840)
    assert cal["lost"] == ["sx1"] and np.isfinite(cal["max_error_px"])


def _block(tmp_path):
    """A small block: two Navcam-like fisheye frames at two sizes and two Mastcam-Z frames, ground points."""
    from mppp.sfm.project import SfmProject
    rng = np.random.default_rng(3)
    rec = pycolmap.Reconstruction()
    cams = [("THIN_PRISM_FISHEYE", NL, 5120, 3840), ("FULL_OPENCV", Z63, 1648, 1200)]
    for cid, (m, p, w, h) in enumerate(cams, start=1):
        rec.add_camera(pycolmap.Camera(camera_id=cid, model=m, width=w, height=h, params=np.asarray(p, float)))
        rig = pycolmap.Rig(rig_id=cid)
        rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid))
        rec.add_rig(rig)
    pts = np.c_[rng.uniform(-3, 3, (400, 2)) + [0, 6], rng.uniform(-0.3, 0.3, 400)]   # ground ahead (y north)
    from scipy.spatial.transform import Rotation
    rows = []
    for iid, (cid, C) in enumerate([(1, [0, 0, 2]), (1, [0.4, 0, 2]), (2, [0.1, 0.1, 2.1]), (2, [0.2, 0.1, 2.1])], 1):
        # looking north and 30 deg down: x_cam east, z_cam north tilted down
        R_cw = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # world (E, N, U) -> camera (x, y, z)
        R_cw = Rotation.from_euler("x", 30, degrees=True).as_matrix() @ R_cw
        t = -R_cw @ np.asarray(C, float)
        fr = pycolmap.Frame(frame_id=iid, rig_id=cid)
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid), id=iid))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R_cw), t)
        rec.add_frame(fr)
        cam = rec.cameras[cid]
        Xc = pts @ R_cw.T + t
        uv = np.asarray(cam.img_from_cam(Xc), float)
        ok = (Xc[:, 2] > 0.1) & (uv[:, 0] > 0) & (uv[:, 0] < cam.width) & (uv[:, 1] > 0) & (uv[:, 1] < cam.height)
        im = pycolmap.Image(name=f"IMG{iid}.png", keypoints=uv, camera_id=cid, image_id=iid)
        im.frame_id = iid
        rec.add_image(im)
        rows.append({"name": f"IMG{iid}.png", "camera_group": "NL" if cid == 1 else "ZL063", "has_mask": True,
                     "ok": ok})
    for fid in list(rec.frames):
        rec.register_frame(fid)
    for k in range(len(pts)):
        tr = pycolmap.Track()
        for iid, r in enumerate(rows, 1):
            if r["ok"][k]:
                tr.add_element(iid, k)
        if tr.length() >= 2:
            rec.add_point3D(pts[k], tr, np.array([120, 100, 80], np.uint8))
    root = tmp_path / "mars2020_sol_0091_test" / "colmap"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for r in rows:
        (root / "images" / r["name"]).write_bytes(b"x")
        (root / "masks" / (r["name"] + ".png")).write_bytes(b"x")
        r.pop("ok")
    model = root / "gui_native" / "sparse"
    model.mkdir(parents=True)
    rec.write(str(model))
    proj = SfmProject(root, rows, {}, {}, [70.0, -50.0, 0.0], {"world_frame": "site3_enu"})
    proj.save()
    return proj, rec


def test_the_job_and_the_bundler_tie_points_line_up_with_colmap(tmp_path):
    proj, rec = _block(tmp_path)
    res = MS.write_metashape_job(proj)
    out = proj.root / "metashape"
    assert res["psx"] == str(out / "mars2020_sol_0091_test.psx") and res["masks"] == 4 and not res["inexact_sensors"]
    assert (out / "build_metashape_project.py").is_file()
    bat = (out / "make_metashape_project.bat").read_bytes()
    assert b"\r\n" in bat and b"build_metashape_project.py" in bat and b"metashape_job.json" in bat
    job = json.loads((out / "metashape_job.json").read_text())
    assert job["offset"] == [70.0, -50.0, 0.0] and "site3_enu" in job["crs_wkt"] and job["check"]
    sens = {s["id"]: s for s in job["sensors"]}
    assert sens[1]["type"] == "fisheye" and sens[2]["type"] == "frame" and sens[2]["label"].startswith("ZL063_1648x1200")
    # the job's poses and sensors reproject COLMAP's observations exactly (Metashape's formulas)
    ims = {i["name"]: i for i in job["images"]}
    assert all(i["path"].endswith(n) and i["mask"].endswith(n + ".png") for n, i in ims.items())
    for name, X, Y, Z, u, v in job["check"]:
        T = np.array(ims[name]["world_from_camera"])
        Xc = (np.linalg.inv(T) @ np.r_[X, Y, Z, 1.0])[:3]
        assert np.allclose(MS.metashape_project_px(sens[ims[name]["sensor_id"]], Xc[None])[0], [u, v], atol=1e-6)
    # Bundler: the cameras in the image list's order, y and z flipped; keypoints from the centre, y up
    lines = (out / "mars2020_sol_0091_test.bundler.out").read_text().splitlines()
    names = (out / "mars2020_sol_0091_test.bundler_images.txt").read_text().split()
    nc, npt = map(int, lines[1].split())
    assert nc == 4 and npt == len(rec.points3D) and names == sorted(ims)
    cams = []
    for k in range(nc):
        b = lines[2 + 5 * k: 7 + 5 * k]
        R = np.array([list(map(float, b[i].split())) for i in (1, 2, 3)])
        t = np.array(list(map(float, b[4].split())))
        cams.append((R, t))
    i = 2 + 5 * nc
    checked = 0
    while i < len(lines):
        X = np.array(list(map(float, lines[i].split())))
        obs = lines[i + 2].split()
        for c, key, x, y in zip(obs[1::4], obs[2::4], obs[3::4], obs[4::4]):
            R, t = cams[int(c)]
            im = rec.images[[iid for iid in rec.images if rec.images[iid].name == names[int(c)]][0]]
            cam = rec.cameras[im.camera_id]
            Xc = np.diag([1.0, -1.0, -1.0]) @ (R @ X + t)                  # Bundler camera -> COLMAP camera
            u, v = float(x) + cam.width / 2, cam.height / 2 - float(y)
            assert np.allclose([u, v], im.points2D[int(key)].xy, atol=1e-3)
            assert np.allclose(np.asarray(cam.img_from_cam(Xc[None]), float)[0], [u, v], atol=1e-3)
            checked += 1
        i += 3
    assert checked == sum(p.track.length() for p in rec.points3D.values())
    reg = job["region"]
    assert reg["size"][2] >= 0.1 * max(reg["size"][:2]) and abs(reg["center"][1] - 6) < 1


def test_the_metashape_side_script_needs_no_mppp():
    src = open(MB.__file__, encoding="utf-8").read()
    assert "import mppp" not in src and "from mppp" not in src and "numpy" not in src.split('"""', 2)[2]
    assert MB.M is None or hasattr(MB.M, "app")
    s = MB.stats([0.3, -0.1], [0.4, 0.1])
    assert s["n"] == 2 and s["max_px"] == pytest.approx(0.5) and s["mean_du_px"] == pytest.approx(0.1)
    assert MB.median([3, 1, 2]) == 2 and MB.median([]) is None
