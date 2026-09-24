"""mppp.sfm: cameras from Metashape XML, keypoint scaling, rig, weighted BA (synthetic ground truth)."""
import json
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
pytest.importorskip("pyceres")

from conftest import ROOT  # noqa: E402
from mppp.paths import data_dir  # noqa: E402

DATA_DIR = data_dir()


def _metashape_project(xyz, c):
    """Metashape frame model, corner pixel origin."""
    x, y = xyz[0] / xyz[2], xyz[1] / xyz[2]
    r2 = x * x + y * y
    rad = 1 + c["k1"] * r2 + c["k2"] * r2 ** 2 + c["k3"] * r2 ** 3
    xp, yp = x * rad, y * rad
    return np.array([c["width"] * 0.5 + c["cx"] + xp * c["f"], c["height"] * 0.5 + c["cy"] + yp * c["f"]])


def test_camera_from_metashape_xml_matches_metashape_projection():
    from mppp.sfm.project import camera_from_metashape_xml, read_metashape_calibration
    xml = DATA_DIR / "m20_cmods/M2020_NL0_frame.xml"
    cam_d = camera_from_metashape_xml(xml)
    assert cam_d["model"] == "FULL_OPENCV" and (cam_d["width"], cam_d["height"]) == (5120, 3840)
    p = cam_d["params"]
    assert p[0] == p[1] and p[6] == p[7] == 0.0 and p[9:] == [0.0, 0.0, 0.0]        # b1, p1, p2 zeroed; k4-k6 = 0
    c = read_metashape_calibration(xml)
    cam = pycolmap.Camera(model=cam_d["model"], width=5120, height=3840, params=p)
    for X in ([0.3, -0.2, 1.0], [-0.6, 0.4, 1.0], [0.0, 0.0, 1.0]):
        X = np.array(X)
        assert np.allclose(cam.img_from_cam(X), _metashape_project(X, c), atol=1e-6)


def test_keypoint_scaling_is_exact_for_binned_padded_frames():
    from mppp.sfm.database import scale_keypoints
    kp = np.array([[0.5, 0.5, 2, 0, 0, 2], [1279.5, 959.5, 1, 0, 0, 1]], np.float32)   # quarter-res pixel centres
    full = scale_keypoints(kp, 4.0)
    assert np.allclose(full[:, :2], [[2.0, 2.0], [5118.0, 3838.0]])                  # centre of the 4x4 bin
    assert np.allclose(full[0, 2:], [8, 0, 0, 8])
    kp4 = np.array([[10.0, 20.0, 3.0, 0.7]], np.float32)
    assert np.allclose(scale_keypoints(kp4, 2.0), [[20, 40, 6, 0.7]])


def test_error_projection_matches_pycolmap_with_distortion():
    from mppp.error.colmap import project_camera
    rng = np.random.default_rng(3)
    for model, params in [("FULL_OPENCV", [2950, 2951, 2594, 1942, -0.27, 0.1, 1e-4, -2e-4, -0.02, 0.01, 0.002, 0.001]),
                          ("OPENCV", [1000, 1001, 640, 480, -0.1, 0.02, 1e-3, -1e-3]),
                          ("RADIAL", [800, 400, 300, -0.2, 0.05]), ("PINHOLE", [700, 710, 320, 240])]:
        cam = pycolmap.Camera(model=model, width=1280, height=960, params=params)
        for _ in range(5):
            X = np.array([*rng.uniform(-0.5, 0.5, 2), 1.0]) * rng.uniform(1, 10)
            assert np.allclose(project_camera(model, np.array(params, float), X), cam.img_from_cam(X), atol=1e-6)


# ---------------------------------------------------------------- synthetic BA
def _synthetic(tmp_path, noise_native_px=0.3, seed=0):
    """Two stations, a mast pan of 6 stereo exposures each (half and quarter resolution), terrain points."""
    from mppp.sfm.project import SfmProject, camera_from_metashape_xml
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    cams = {f"N{e}": camera_from_metashape_xml(DATA_DIR / f"m20_cmods/M2020_N{e}0_frame.xml") for e in "LR"}
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


def test_weighted_ba_recovers_intrinsics_and_poses(tmp_path):
    from mppp.sfm.reconstruction import bundle_adjust, native_residuals
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, true_params = _build_rec(proj, truth, P, cams_d, rigT, noise, rng)
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=200)
    res = native_residuals(rec, proj)["residual_native_px"]
    assert np.sqrt(np.mean(res ** 2)) < 1.6 * noise * np.sqrt(2) and out["observations"] > 5000
    for cid in (1, 2):
        p, t = np.asarray(rec.cameras[cid].params), true_params[cid]
        assert abs(p[0] - t[0]) < 1.0 and abs(p[2] - t[2]) < 1.0 and abs(p[4] - t[4]) < 1e-3
        assert p[6] == p[7] == 0.0 and p[9] == p[10] == p[11] == 0.0                   # held at zero
    T = rec.rigs[1].sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=2))
    assert np.allclose(T.translation, rigT[1], atol=1e-9)                              # baseline held
    for r in proj.images:
        R, C = truth[r["name"]]
        Tcw = rec.images[r["image_id"]].cam_from_world()
        Cest = -Tcw.rotation.matrix().T @ Tcw.translation
        assert np.linalg.norm(Cest - C) < 0.02


def test_weights_follow_native_resolution(tmp_path):
    """Quarter-resolution observations get 1/16 the weight of half-resolution ones in full-res pixels."""
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=1e6, refine_intrinsics=False, refine_rig=False,
                        use_priors=False, max_iterations=0)
    # at the truth, each whitened residual is ~N(0,1) per coordinate: cost ~ n_obs (0.5 * 2 per obs)
    assert 0.8 < out["initial_cost"] / out["observations"] < 1.2


def test_prior_pairs_see_overlap_and_reject_opposite_views(tmp_path):
    from mppp.sfm.pairs import footprints, _visible_fraction
    proj, truth, *_ = _synthetic(tmp_path)
    for r in proj.images:
        R, C = truth[r["name"]]
        r["prior_R_w2c"], r["prior_C"] = R.tolist(), C.tolist()
    import cv2
    (tmp_path / "images").mkdir(exist_ok=True)
    for r in proj.images:
        w, h = r["native_size"]
        cv2.imwrite(str(tmp_path / "images" / r["name"]), np.full((h, w), 100, np.uint8))
    fps = footprints(proj, grid=(16, 12))
    a = fps["NLF_0001.png"]
    assert _visible_fraction(a, fps["NRF_0001.png"], (16, 12)) > 0.8          # stereo partner
    assert _visible_fraction(a, fps["NLF_0006.png"], (16, 12)) < 0.05         # 150 deg away


def test_filter_observations_removes_outliers(tmp_path):
    """v0p9 bug found on Belva: numpy ids were not recognised by pycolmap maps, so nothing was filtered."""
    from mppp.sfm.reconstruction import filter_observations, native_residuals
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    pid = next(iter(rec.points3D))
    rec.points3D[pid].xyz[:] += np.array([0.5, 0.0, 0.0])            # one gross outlier track
    before = native_residuals(rec, proj)["residual_native_px"]
    n = filter_observations(rec, proj, 2.0)
    after = native_residuals(rec, proj)["residual_native_px"]
    assert 0 < n <= int((before > 2.0).sum()) and (after <= 2.0).all()      # a 2-view track goes whole


def test_gpu_steps_run_in_another_python(tmp_path):
    """v0p11: extraction and matching can run as a subprocess in a second environment (a CUDA pycolmap on
    Windows, whose ceres types are incompatible with pip pyceres); settings come back through project.json."""
    import shutil, sys
    import cv2
    from mppp.sfm import SfmProject, check_gpu_python, extract_features, match
    rng = np.random.default_rng(3)
    big = cv2.GaussianBlur((rng.random((700, 900)) * 255).astype(np.uint8), (0, 0), 2)
    big = cv2.equalizeHist(big)
    (tmp_path / "images").mkdir()
    names = []
    for k, x0 in enumerate((0, 120, 240)):
        n = f"im{k}.png"
        cv2.imwrite(str(tmp_path / "images" / n), cv2.cvtColor(big[50:650, x0:x0 + 600], cv2.COLOR_GRAY2BGR))
        names.append(n)
    proj = SfmProject(tmp_path, [{"name": n, "has_mask": False} for n in names], {}, {}, [0, 0, 0], {"keep": 1})
    from mppp.sfm import check_ba_environment
    assert check_ba_environment()["pyceres"]                                  # pip pair: BA works here
    with pytest.warns(UserWarning, match="own python"):
        info = check_gpu_python(sys.executable, require_cuda=False)
    assert info["pycolmap"] == info["pycolmap_here"]
    extract_features(proj, max_num_features=2000, use_gpu=False, python=sys.executable)
    assert proj.features_db.is_file() and proj.settings["features"]["max_num_features"] == 2000
    assert proj.settings["keep"] == 1                                           # parent's settings survive
    shutil.copy(proj.features_db, proj.database)                               # matchable as it is
    res = match(proj, mode="exhaustive", use_gpu=False, max_error_px=4.0, python=sys.executable)
    assert res["verified_pairs"] >= 2 and proj.settings["matching"]["verified_pairs"] == res["verified_pairs"]
    if not pycolmap.has_cuda:
        with pytest.raises(RuntimeError, match="no CUDA"):
            check_gpu_python(sys.executable, require_cuda=True)
    with pytest.raises(FileNotFoundError):
        check_gpu_python(tmp_path / "nope" / "python.exe")
