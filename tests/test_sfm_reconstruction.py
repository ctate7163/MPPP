"""Triangulation, weighted bundle adjustment, outliers, staging (mppp.sfm.reconstruction). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import numpy as np
import pytest
from mppp.paths import data_dir  # noqa: E402
from pathlib import Path
import copy
from types import SimpleNamespace
from mppp.sfm.reconstruction import camera_changes, print_camera_changes  # noqa: E402


pycolmap = pytest.importorskip("pycolmap")


DATA_DIR = data_dir()


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
        assert abs(p[6] - t[6]) < 5e-5 and abs(p[7] - t[7]) < 5e-5                    # p1, p2 refined (v0p20 default)
        assert p[9] == p[10] == p[11] == 0.0                                          # k4-k6 held at zero
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


def test_two_view_tracks_kept_and_counted():
    import pycolmap
    from mppp.sfm.reconstruction import drop_short_tracks, reconstruct
    import inspect
    assert inspect.signature(reconstruct).parameters["min_track_length"].default == 2
    rec = pycolmap.Reconstruction()
    rec.add_camera(pycolmap.Camera(camera_id=1, model="PINHOLE", width=100, height=100, params=[100, 100, 50, 50]))
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    for i in (1, 2, 3):
        fr = pycolmap.Frame(frame_id=i, rig_id=1)
        fr.rig_from_world = pycolmap.Rigid3d()
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=i))
        rec.add_frame(fr)
        im = pycolmap.Image(name=f"{i}.png", keypoints=np.zeros((2, 2)), camera_id=1, image_id=i)
        im.frame_id = i
        rec.add_image(im)
        rec.register_frame(i)
    for ids, k in (((1, 2), 0), ((1, 2, 3), 1)):
        tr = pycolmap.Track()
        for i in ids:
            tr.add_element(i, k)
        rec.add_point3D(np.array([0, 0, 5.0]), tr)
    assert drop_short_tracks(rec, 2) == 0 and len(rec.points3D) == 2          # the two-view point stays
    assert drop_short_tracks(rec, 3) == 1 and len(rec.points3D) == 1


def test_fixed_camera_params_tangential_switch():
    from mppp.sfm.reconstruction import fixed_camera_params
    assert fixed_camera_params("FULL_OPENCV") == [6, 7, 9, 10, 11]
    assert fixed_camera_params("FULL_OPENCV", refine_tangential=True) == [9, 10, 11]
    assert fixed_camera_params("FULL_OPENCV", refine_principal_point=False, refine_tangential=True) == [2, 3, 9, 10, 11]
    assert fixed_camera_params("OPENCV", refine_tangential=True) == []


def test_reconstruct_defaults_four_rounds_and_half_degree():
    import inspect
    from mppp.sfm.reconstruction import DEFAULT_SCHEDULE, reconstruct
    sig = inspect.signature(reconstruct).parameters
    assert sig["schedule"].default == DEFAULT_SCHEDULE and len(DEFAULT_SCHEDULE) == 4
    assert DEFAULT_SCHEDULE[-1] == DEFAULT_SCHEDULE[-2] == (8.0, 2.0, 2.0)   # v0p14.5: round 4 repeats round 3
    assert all(a[2] >= b[2] for a, b in zip(DEFAULT_SCHEDULE, DEFAULT_SCHEDULE[1:]))   # cut-offs only tighten
    assert sig["min_tri_angle_deg"].default == 0.25 and sig["sigma_px"].default == 0.5     # v0p20: 0.25 (was 0.5)
    assert sig["refine_tangential"].default is True                                       # v0p20 (was False)


def test_ba_recovers_tangential_only_when_asked(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    import numpy as np
    from helpers import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.health import _camera_change
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    for k in cams_d:                                                    # truth has tangential distortion
        cams_d[k]["params"][6], cams_d[k]["params"][7] = 3e-4, -2e-4
    for refine in (True, False):
        rec, true_params = _build_rec(proj, truth, P, cams_d, rigT, noise, np.random.default_rng(1))
        for cid in rec.cameras:                                         # start at p1 = p2 = 0
            q = np.array(rec.cameras[cid].params)
            q[6:8] = 0.0
            rec.cameras[cid].params = q
        out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation",
                            refine_tangential=refine, max_iterations=200)
        assert out["refine_tangential"] is refine
        for cid in (1, 2):
            p = np.asarray(rec.cameras[cid].params)
            if refine:
                assert abs(p[6] - 3e-4) < 5e-5 and abs(p[7] + 2e-4) < 5e-5
            else:
                assert p[6] == p[7] == 0.0
            assert p[9] == p[10] == p[11] == 0.0
    ch = _camera_change(dict(cams_d["NL"], params=list(cams_d["NL"]["params"])), rec.cameras[1])
    assert ch["p1_initial"] == pytest.approx(3e-4) and ch["p1_refined"] == 0.0


def test_ba_holds_camera_specific_parameters(tmp_path):
    pytest.importorskip("pyceres")
    import numpy as np
    from helpers import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, true_params = _build_rec(proj, truth, P, cams_d, rigT, noise, rng)
    start = {cid: np.array(rec.cameras[cid].params) for cid in (1, 2)}
    proj.cameras["NR"]["fixed_params"] = ["cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=100)
    p1, p2 = np.asarray(rec.cameras[1].params), np.asarray(rec.cameras[2].params)
    assert np.array_equal(p2[2:], start[2][2:])                         # NR: all but fx, fy held
    assert abs(p2[0] - start[2][0]) > 1.0                               # its focal length moved (+0.3 % start)
    assert abs(p1[2] - start[1][2]) > 0.5 and abs(p1[4] - start[1][4]) > 1e-4     # NL refines as before
    # NR dressed as a Mastcam-Z focus bin: table, plot and health run on it
    from mppp.sfm.health import assess_alignment
    from mppp.sfm.zcam import write_focus_breathing
    proj.cameras["NR"].update(group="NR", focus_count_median=1000.0, focus_count_range=[995.0, 1004.0], n_images=12)
    for r in proj.images:
        r["camera_group"] = r["instrument"]
        if r["instrument"] == "NR":
            r["focus_count"], r["label_f_px"] = 1000.0, float(start[2][0])
    fb = write_focus_breathing(proj, rec, tmp_path / "fb", min_observations=10)
    row = fb["table"][0]
    assert len(fb["table"]) == 1 and row["camera"] == "NR" and row["observations"] > 1000
    assert abs(row["f_refined_px"] - 0.5 * (p2[0] + p2[1])) < 1e-9 and row["held_params"].startswith("cx,cy")
    assert fb["fits"]["NR"]["refined"] is None and Path(fb["png"]).is_file()          # one bin: no slope
    rep = assess_alignment(proj, rec)
    assert "NR" in rep["cameras"] and any(c["check"] == "residual_eye_ratio" for c in rep["checks"])


def _rational(eye="L", zero=()):
    from mppp.paths import data_dir
    from mppp.sfm.project import camera_from_colmap_json
    return camera_from_colmap_json(data_dir() / f"cmods/M2020_N{eye}_rational.json", zero)


def test_bundle_adjustment_refines_k4_of_the_rational_camera(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    for with_map in (True, False):
        proj, truth, P, _, rigT, noise, rng = _synthetic(tmp_path / str(with_map))
        cams = {f"N{e}": _rational(e) for e in "LR"}
        proj.cameras = cams
        if with_map:
            proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
        rec, true_params = _build_rec(proj, truth, P, cams, rigT, noise, rng)
        for cid in (1, 2):
            q = np.array(rec.cameras[cid].params)
            q[9] += 0.02                                                    # start k4 off
            rec.cameras[cid].params = q
        if not with_map:                                                    # free_params would be ignored
            with pytest.raises(RuntimeError, match="build_database"):
                bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=5)
            continue
        bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=200)
        for cid in (1, 2):
            p, t = np.asarray(rec.cameras[cid].params), true_params[cid]
            # k4 started 0.02 off; it trades off against k1-k3 on this small block (0.005-0.008 left with the
            # v0p22 consensus cameras), so the test asks for most of the offset to be recovered
            assert abs(p[9] - t[9]) < 0.5 * 0.02 and abs(p[0] - t[0]) < 1.0
            assert p[10] == p[11] == 0.0                                    # k5, k6 held


def test_attitude_prior_holds_the_block_orientation(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from helpers import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    rot = {}
    for sigma in (None, 0.01):
        proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path / str(sigma))
        proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
        rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, np.random.default_rng(1), perturb=False)
        # priors: every attitude rotated by 0.5 deg about the line through the two stations (a gauge direction)
        axis = np.array([4.0, 1.0, 0.0]) / np.linalg.norm([4.0, 1.0, 0.0])
        dR = Rotation.from_rotvec(np.radians(0.5) * axis).as_matrix()
        for r in proj.images:
            r["prior_R_w2c"] = (np.asarray(r["prior_R_w2c"]) @ dR.T).tolist()
        out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=100, attitude_prior_deg=sigma)
        assert out["attitude_priors"] == (0 if sigma is None else len(rec.frames))
        ang = []
        for im in rec.images.values():
            R1 = np.asarray(im.cam_from_world().rotation.matrix())
            R0, _ = truth[im.name]
            ang.append(np.degrees(np.linalg.norm(Rotation.from_matrix(R0.T @ R1).as_rotvec())))
        rot[sigma] = float(np.median(ang))
    assert rot[None] < 0.05                  # no attitude prior: the block stays where the tie points put it
    assert 0.4 < rot[0.01] < 0.6             # a strong attitude prior pulls it onto the rotated priors


def _solved(tmp_path, seed=0):
    """The synthetic two-station rig block of test_sfm at its true poses, with a database camera map."""
    pytest.importorskip("pycolmap")
    from helpers import _build_rec, _synthetic
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path, seed=seed)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, np.random.default_rng(seed + 1), perturb=False)
    return proj, rec, truth


def test_find_outlier_frames_flags_a_frame_that_left_its_station(tmp_path):
    from mppp.sfm.reconstruction import OUTLIER_DEFAULTS, exclude_frames, find_outlier_frames
    proj, rec, _ = _solved(tmp_path)
    assert find_outlier_frames(rec, proj) == []                       # a consistent block: nothing flagged
    # frame 3's priors are 1.5 m off (its solved pose did not follow them: a frame that "did not align")
    bad = next(fid for fid in rec.frames if fid == 3)
    names = {rec.images[d.id].name for d in rec.frames[bad].data_ids}
    for r in proj.images:
        if r["name"] in names:
            r["prior_C"] = (np.asarray(r["prior_C"]) + [1.5, 0.0, 0.0]).tolist()
    found = find_outlier_frames(rec, proj)
    assert [f["frame_id"] for f in found] == [bad]
    assert any("shift" in s for s in found[0]["reasons"]) and found[0]["family"] == "N"
    assert set(found[0]["images"]) == names
    # a tilt common to a whole station (a rover attitude error) is not an outlier; one frame's is
    from scipy.spatial.transform import Rotation
    for r in proj.images:
        if r["name"] in names:
            r["prior_C"] = (np.asarray(r["prior_C"]) - [1.5, 0.0, 0.0]).tolist()
    dR = Rotation.from_rotvec(np.radians([0.0, 0.0, 2.0])).as_matrix()
    for r in proj.images:
        if r["station"] == "S001D0000":
            r["prior_R_w2c"] = (np.asarray(r["prior_R_w2c"]) @ dR).tolist()
    assert find_outlier_frames(rec, proj) == []
    for r in proj.images:
        if r["name"] in names:
            r["prior_R_w2c"] = (np.asarray(r["prior_R_w2c"]) @ dR).tolist()
    found = find_outlier_frames(rec, proj)
    assert [f["frame_id"] for f in found] == [bad] and "attitude" in found[0]["reasons"][0]
    # thresholds can be relaxed; min_observations flags weakly tied frames
    assert find_outlier_frames(rec, proj, min_attitude_deg=5.0) == []
    many = find_outlier_frames(rec, proj, min_attitude_deg=5.0, min_observations=10 ** 6)
    assert len(many) == len(rec.frames)
    assert set(OUTLIER_DEFAULTS) >= {"residual_factor", "min_observations", "shift_mad_factor"}
    # exclusion deregisters the frame and its images
    n_img = rec.num_reg_images()
    assert exclude_frames(rec, [bad]) == 1 and exclude_frames(rec, [bad]) == 0
    assert rec.num_reg_images() == n_img - 2 and bad not in set(rec.reg_frame_ids())


def test_find_outlier_frames_flags_large_residuals(tmp_path):
    from scipy.spatial.transform import Rotation
    import pycolmap
    from mppp.sfm.reconstruction import find_outlier_frames
    proj, rec, _ = _solved(tmp_path)
    fr = rec.frames[5]
    T = fr.rig_from_world
    dR = Rotation.from_rotvec(np.radians([0.3, 0.0, 0.0])).as_matrix()
    fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(dR @ T.rotation.matrix()), dR @ np.asarray(T.translation))
    found = find_outlier_frames(rec, proj)
    assert 5 in [f["frame_id"] for f in found]
    f5 = next(f for f in found if f["frame_id"] == 5)
    assert any("median residual" in s for s in f5["reasons"]) and f5["median_residual_px"] > 5


def test_convergence_statistics_counts_ray_angles(tmp_path):
    from mppp.sfm.reconstruction import CONVERGENCE_BINS_DEG, convergence_statistics
    proj, rec, _ = _solved(tmp_path)
    cs = convergence_statistics(rec, proj)
    n2 = sum(1 for p in rec.points3D.values() if p.track.length() >= 2)
    assert sum(cs["points"]) == n2 and cs["bins_deg"] == list(CONVERGENCE_BINS_DEG)
    assert all(a <= b for a, b in zip(cs["cross_station_points"], cs["points"]))
    assert cs["points_over_10deg"] == sum(c for c, lo in zip(cs["points"], cs["bins_deg"]) if lo >= 10)
    assert cs["observations_over_10deg"] >= 2 * cs["points_over_10deg"]
    # the check by hand on one cross-station point
    C = {i: np.asarray(im.projection_center()) for i, im in rec.images.items()}
    pid, pt = next((k, p) for k, p in rec.points3D.items()
                   if len({rec.images[e.image_id].name.split("_")[1] for e in p.track.elements}) > 4)
    d = np.array([pt.xyz - C[e.image_id] for e in pt.track.elements])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    assert np.degrees(np.arccos(np.clip(d @ d.T, -1, 1).min())) > 0
    s = convergence_statistics(rec, proj, max_points=500)
    assert s["sampled"] and abs(sum(s["points"]) - n2) < 0.05 * n2       # sample scaled to all points


def test_triangulate_passes_the_track_options(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm import reconstruction as R
    seen = {}

    def fake(rec, db, images, out, clear_points, options, refine_intrinsics):
        seen["t"] = options.triangulation
        return rec
    monkeypatch.setattr(pycolmap, "triangulate_points", fake)
    proj, rec, _ = _solved(tmp_path)
    R.triangulate(rec, proj, max_reproj_px=6.0, min_angle_deg=0.25, max_transitivity=3, create_max_angle_error_deg=4.0,
                  continue_max_angle_error_deg=3.0, complete_max_transitivity=8)
    t = seen["t"]
    assert (t.max_transitivity, t.complete_max_transitivity) == (3, 8)
    assert (t.create_max_angle_error, t.continue_max_angle_error) == (4.0, 3.0)
    assert t.merge_max_reproj_error == 6.0 and t.min_angle == 0.25 and not t.ignore_two_view_tracks


def test_rig_translation_prior_keeps_the_baseline(tmp_path):
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig=True, max_iterations=100,
                        rig_translation_sigma_m=0.002)
    assert out["rig_translation_priors"] == 1
    assert list(out["rig"]) == ["1:2"]
    rig = out["rig"]["1:2"]
    assert abs(rig["baseline_m"] - 0.4244) < 0.002 and abs(rig["baseline_start_m"] - 0.4244) < 1e-6
    assert np.linalg.norm(rig["centre_change_m"]) < 0.003
    out2 = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=20)
    assert out2["rig_translation_priors"] == 0 and abs(out2["rig"]["1:2"]["baseline_m"] - rig["baseline_m"]) < 1e-9


def test_navcam_network_and_hold(tmp_path):
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust, navcam_network
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    net = navcam_network(proj)
    assert net["stations"] == 2 and 3.5 < net["span_m"] < 4.5 and net["verdict"] == "weak"
    assert navcam_network(proj, min_stations=2, min_span_m=1.0)["verdict"] == "strong"
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)           # perturbed start cameras
    start = {cid: np.array(c.params) for cid, c in rec.cameras.items()}
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_intrinsics=True, max_iterations=10,
                        hold_cameras=["NL", "NR"])
    for cid, c in rec.cameras.items():
        assert np.allclose(np.array(c.params), start[cid])
    assert out["brief"]


def test_reconstruct_holds_navcam_intrinsics_on_a_weak_network(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec0))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)    # no database: keep the synthetic points
    start = {cid: np.array(c.params) for cid, c in rec0.cameras.items()}
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="auto")
    rs = proj.settings["reconstruction"]
    assert rs["navcam_network"]["verdict"] == "weak" and rs["navcam_intrinsics"] == "hold"
    assert set(rs["hold_cameras"]) == {"NL", "NR"} and rs["staged"] is False and rs["navcam_stage"] is None
    for cid, c in rec.cameras.items():
        assert np.allclose(np.array(c.params), start[cid])
    with pytest.raises(ValueError):
        R.reconstruct(proj, navcam_intrinsics="maybe")


def test_reconstruct_staged_solves_navcam_first(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec0))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)
    # pretend the second station's frames are Mastcam-Z
    z = R._frames_of_family(rec0, proj, "N")[len(rec0.frames) // 2:]
    real = R._frames_of_family
    monkeypatch.setattr(R, "_frames_of_family", lambda rec, project, fam: z if fam == "Z" else
                        [f for f in real(rec, project, "N") if f not in z])
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="refine", staged=True, out_name="cahv_ba")
    rs = proj.settings["reconstruction"]
    assert rs["staged"] is True and rs["navcam_stage"]["path"] == "sparse/cahv_ba_navcam"
    assert (proj.root / "sparse" / "cahv_ba_navcam").is_dir()
    import pycolmap
    nav = pycolmap.Reconstruction(str(proj.root / "sparse" / "cahv_ba_navcam"))
    assert nav.num_reg_images() == rec.num_reg_images() - 2 * len(z)   # stage 1 without the "Z" frames
    assert set(rec.reg_frame_ids()) >= set(z)                          # stage 2 registered them again
    assert any(e.get("stage") == 2 for e in rs["log"] if isinstance(e, dict))
    for cid, c in rec.cameras.items():                                 # stage 2 held the Navcam cameras
        assert np.allclose(np.array(c.params), np.array(nav.cameras[cid].params))


def test_bundle_adjust_solver_choice_and_block_rotation(tmp_path):
    pytest.importorskip("pyceres")
    import pycolmap
    from helpers import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.health import assess_alignment
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=10)
    assert out["linear_solver"] == "dense_schur" and out["num_threads"] >= 1 and out["seconds"] > 0
    out2 = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=3, linear_solver="sparse_schur")
    assert out2["linear_solver"] == "sparse_schur"
    with pytest.raises(KeyError):
        bundle_adjust(rec, proj, sigma_px=noise, max_iterations=1, linear_solver="magic")
    rep = assess_alignment(proj, rec)
    c = next(x for x in rep["checks"] if x["check"] == "block_rotation_deg")
    assert c["value"] < 0.3 and c["status"] == "pass" and len(rep["world_frame"]["block_rotation_about_E_N_U_deg"]) == 3
    assert rep["world_frame"]["offset_enu_m"] == [0, 0, 0]
    assert any(x["check"] == "attitude_residual_p95_deg" and x["status"] == "pass" for x in rep["checks"])
    # turn the whole solution by +0.8 deg about Up: reported as +0.8 about U (sign: prior -> refined)
    from scipy.spatial.transform import Rotation
    W = Rotation.from_rotvec(np.radians([0, 0, 0.8])).as_matrix()
    base = np.array(rep["world_frame"]["block_rotation_about_E_N_U_deg"])
    for fr in rec.frames.values():
        T = fr.rig_from_world
        R, C = T.rotation.matrix(), -T.rotation.matrix().T @ T.translation
        R2 = R @ W.T                                      # camera axes in the world: W @ R^T
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R2), -R2 @ (W @ C))
    rep2 = assess_alignment(proj, rec)
    d = np.array(rep2["world_frame"]["block_rotation_about_E_N_U_deg"]) - base
    assert abs(d[2] - 0.8) < 0.05 and abs(d[0]) < 0.05 and abs(d[1]) < 0.05
    c2 = next(x for x in rep2["checks"] if x["check"] == "block_rotation_deg")
    assert c2["status"] == "warn"


def test_outlier_residual_floor():
    from mppp.sfm.reconstruction import OUTLIER_DEFAULTS
    assert OUTLIER_DEFAULTS["min_residual_px"] == 1.2


def _rec():
    rec = pycolmap.Reconstruction()
    p = [2956.0, 2956.0, 2591.0, 1944.0, 0.3, -0.02, 0.0, 0.0, 0.002, 0.59, 0.0, 0.0]
    for cid in (1, 2, 3, 4):
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="THIN_PRISM_FISHEYE", width=5120, height=3840,
                                       params=[x * (1 + 0.001 * cid) for x in p]))
    return rec, p


def _short_stop(tmp_path, offset=(1.5, -1.0, 0.0)):
    """The synthetic block with one stereo frame of the second station relabelled as a two-image station whose
    waypoint prior is ``offset`` metres off."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import _synthetic_block
    proj, rec, noise = _synthetic_block(tmp_path)
    fid = next(rec.images[int(r["image_id"])].frame_id for r in proj.images if r["station"] == "S001D0100")
    ims = {rec.images[d.id].name for d in rec.frames[fid].data_ids}
    for r in proj.images:
        if r["name"] in ims:
            r["station"] = "S001D0150"
            r["prior_C"] = (np.asarray(r["prior_C"], float) + np.asarray(offset)).tolist()
    return proj, rec, noise, fid


def _centre(rec, fid):
    fr = rec.frames[fid]
    return np.asarray(fr.rig_from_world.inverse().translation)


def test_eye_camera_without_images_after_thermal_split(capsys):
    rec, p = _rec()
    proj = SimpleNamespace(
        cameras={"NL": {"params": p}, "NR": {"params": p},
                 "NL_T10": {"params": p, "group": "NL", "thermal_bin": True},
                 "ZL_f1": {"params": p, "group": "ZL"}},
        # every Navcam image moved to a bin camera: the old notebook cell raised IndexError here
        images=[{"name": "a", "instrument": "NL_T10", "base_instrument": "NL", "camera_id": 1}],
        settings={"database": {"cameras": {"NL": 1, "NR": 2, "NL_T10": 3, "ZL_f1": 4}}})
    rows = camera_changes(rec, proj)
    assert [r["camera"] for r in rows] == ["NL", "NR", "NL_T10"]          # Mastcam-Z focus bin left out
    assert rows[0]["names"][-2:] == ["sx1", "sy1"] and rows[0]["images"] == 0
    assert rows[2]["thermal_bin"] and rows[2]["refined"][0] == pytest.approx(p[0] * 1.003)
    print_camera_changes(rec, proj)
    out = capsys.readouterr().out
    assert "no images after the thermal split" in out and "[temperature bin]" in out


def test_camera_missing_from_reconstruction():
    rec, p = _rec()
    proj = SimpleNamespace(cameras={"NL": {"params": p}, "NX": {"params": p}}, images=[],
                           settings={"database": {"cameras": {"NL": 1}}})
    rows = camera_changes(rec, proj)
    assert rows[1] == {"camera": "NX", "missing": True}


def test_unlocalized_stations_and_dropped_priors(tmp_path):
    pytest.importorskip("pyceres")
    from mppp.sfm.reconstruction import bundle_adjust, unlocalized_stations
    proj, rec, noise, fid = _short_stop(tmp_path)
    truth = _centre(rec, fid)
    u = unlocalized_stations(proj, 4, rec=rec)
    assert list(u) == ["S001D0150"] and u["S001D0150"]["images"] == 2 and u["S001D0150"]["prior"] == "dropped"
    assert u["S001D0150"]["cross_points"] > 20
    assert all(v["prior"].startswith("kept: every") for v in unlocalized_stations(proj, 50).values())
    import copy
    r0, p0 = copy.deepcopy(rec), copy.deepcopy(proj)
    ba0 = bundle_adjust(r0, p0, sigma_px=noise, loss_scale=10.0, max_iterations=30)
    r1, p1 = copy.deepcopy(rec), copy.deepcopy(proj)
    p1.settings["localize_min_images"] = 4
    ba1 = bundle_adjust(r1, p1, sigma_px=noise, loss_scale=10.0, max_iterations=30)
    assert ba1["frames_without_position_prior"] == 1 and ba1["priors"] == ba0["priors"] - 1
    # with its prior the 1.8 m error drags the short stop and the whole block (~12-14 cm); without it the block
    # moves only by the noise (~4-5 cm)
    e0 = np.linalg.norm(_centre(r0, fid) - truth)
    e1 = np.linalg.norm(_centre(r1, fid) - truth)
    assert e1 < 0.6 * e0
    other = [f for f in rec.frames if f != fid]
    m0 = np.mean([np.linalg.norm(_centre(r0, f) - _centre(rec, f)) for f in other])
    m1 = np.mean([np.linalg.norm(_centre(r1, f) - _centre(rec, f)) for f in other])
    assert m1 < 0.6 * m0


def test_register_stations_only_moves_the_short_stop(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    import pycolmap
    from mppp.sfm import reconstruction as RC
    proj, rec, noise, fid = _short_stop(tmp_path)
    truth = _centre(rec, fid)
    # matches from the tracks; then the short stop's observations leave the tracks (as after triangulating with
    # its wrong prior pose), so that its tie points are only correspondences to the other station's points
    pairs = {}
    for pt in rec.points3D.values():
        els = [(el.image_id, el.point2D_idx) for el in pt.track.elements]
        for a in range(len(els)):
            for b in range(a + 1, len(els)):
                (i1, k1), (i2, k2) = sorted((els[a], els[b]))
                if i1 != i2:
                    pairs.setdefault((i1, i2), []).append((k1, k2))
    monkeypatch.setattr(RC, "_verified_matches", lambda project: [(i1, i2, np.array(m)) for (i1, i2), m in pairs.items()])
    mine = {d.id for d in rec.frames[fid].data_ids}
    for pid in list(rec.points3D):
        for el in list(rec.points3D[pid].track.elements):
            if el.image_id in mine and pid in rec.points3D:
                rec.delete_observation(el.image_id, el.point2D_idx)
    register_stations = RC.register_stations
    others = {f: _centre(rec, f) for f in rec.frames if f != fid}
    # the frame starts at its (wrong) prior
    fr = rec.frames[fid]
    fr.rig_from_world = fr.rig_from_world * pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), -np.array([1.5, -1.0, 0.0]))
    assert np.linalg.norm(_centre(rec, fid) - truth) > 1.0
    rep = register_stations(rec, proj, only=["S001D0150"], max_error_px=40.0, min_inliers=10, verbose=False)
    assert rep["stations"]["S001D0150"]["registered"]
    assert np.linalg.norm(_centre(rec, fid) - truth) < 0.2          # from 1.8 m: within reach of triangulation
    assert all(np.allclose(_centre(rec, f), c) for f, c in others.items())


def test_reconstruct_with_localize_min_images(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    import copy
    from mppp.sfm import reconstruction as R
    from mppp.sfm import health as H
    proj, rec0, noise, fid = _short_stop(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec0))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)          # no database: keep the points
    calls = []
    monkeypatch.setattr(R, "register_stations", lambda rec, project, **kw: calls.append(kw) or
                        {"stations": {s: {"registered": True} for s in kw.get("only", [])}, "only": kw.get("only")})
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        localize_min_images=4, gui_native=False)
    rs = proj.settings["reconstruction"]
    assert calls and calls[0]["only"] == ["S001D0150"]
    assert rs["localize_min_images"] == 4 and rs["unlocalized_stations"] == ["S001D0150"]
    assert proj.settings["localize_min_images"] == 4 and rs["unlocalized"]["S001D0150"]["prior"] == "dropped"
    rep = H.assess_alignment(proj, rec)
    names = {c["check"]: c for c in rep["checks"]}
    assert "unlocalized_station_shift_max_m" in names and names["unlocalized_station_shift_max_m"]["value"] > 1.0
    assert rep["stations"]["S001D0150"]["position_prior"] is False
    assert all("S001D0150" not in s["stations"] for s in rep["prior_similarity"])
    # 0 keeps every prior and moves no station
    calls.clear()
    R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False, gui_native=False)
    assert not calls and proj.settings["reconstruction"]["unlocalized_stations"] == []


def _zcam_block(tmp_path, seed=0, noise=0.2):
    """A Mastcam-Z pan at one station: one focus-bin camera (backlash state, f 1 % above the label) holding two
    backlash groups (sequences zcam1, zcam2) and one regular group (zcam3, f = label) of the same focus count."""
    pycolmap = pytest.importorskip("pycolmap")
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import SfmProject
    rng = np.random.default_rng(seed)
    W, H, lab = 1648, 1200, 4680.0
    fb = lab * 1.0095
    base = [fb, fb, 824.0, 600.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    C = np.array([0.0, 0.0, 2.0])
    P = np.column_stack([rng.uniform(-40, 40, 6000), rng.uniform(8, 60, 6000), rng.normal(0, 1.5, 6000)])
    images, rec = [], pycolmap.Reconstruction()
    rec.add_camera(pycolmap.Camera(camera_id=1, model="FULL_OPENCV", width=W, height=H, params=base))
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    true_cam = {}
    plan = [("zcam1", fb, 6), ("zcam2", fb, 5), ("zcam3", lab, 3)]
    k, az = 0, -20.0
    tracks = {}
    for seq, f_true, n in plan:
        cam_t = pycolmap.Camera(camera_id=99, model="FULL_OPENCV", width=W, height=H, params=[f_true, f_true] + base[2:])
        for _ in range(n):
            k += 1
            az += 3.0
            a = np.radians(az)
            fwd = np.array([np.sin(a), np.cos(a), 0.0]) * np.cos(0.05) + np.array([0, 0, -np.sin(0.05)])
            right = np.array([np.cos(a), -np.sin(a), 0.0])
            R = np.stack([right, np.cross(fwd, right), fwd])
            xc = (P - C) @ R.T
            ok = xc[:, 2] > 1
            uv = np.full((len(P), 2), np.nan)
            uv[ok] = np.asarray(cam_t.img_from_cam(xc[ok]), float)
            vis = ok & (uv[:, 0] > 5) & (uv[:, 0] < W - 5) & (uv[:, 1] > 5) & (uv[:, 1] < H - 5)
            idx = np.nonzero(vis)[0]
            kps = uv[idx] + rng.normal(0, noise, (len(idx), 2))
            name = f"ZL0_{k:04d}.png"
            im = pycolmap.Image(name=name, keypoints=kps, camera_id=1, image_id=k)
            fr = pycolmap.Frame(frame_id=k, rig_id=1)
            fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=k))
            fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ C)
            rec.add_frame(fr)
            im.frame_id = k
            rec.add_image(im)
            rec.register_frame(k)
            for j, p in enumerate(idx):
                tracks.setdefault(int(p), []).append((k, j))
            true_cam[name] = f_true
            images.append({"name": name, "stem": name[:-4], "instrument": "ZL034_F00600", "camera_group": "ZL034",
                           "eye": "L", "sclk_key": f"{k:010d}_000", "sol": 700, "site": 1, "drive": 0,
                           "station": "S001D0000", "sequence": seq, "downsample_scale": 1.0, "native_size": [W, H],
                           "focus_count": 600.0, "label_f_px": lab * (1 + rng.normal(0, 2e-4)),
                           "prior_C": C.tolist(), "prior_R_w2c": R.tolist(), "has_mask": False, "image_id": k})
    for p, els in tracks.items():
        if len(els) >= 2:
            tr = pycolmap.Track()
            for iid, j in els:
                tr.add_element(iid, j)
            rec.add_point3D(P[p], tr, np.zeros(3, np.uint8))
    cams = {"ZL034_F00600": {"model": "FULL_OPENCV", "width": W, "height": H, "params": base, "group": "ZL034",
                             "focus_count_median": 600.0, "fixed_params": ["cx", "cy", "k1", "k2", "p1", "p2", "k3"],
                             "n_images": k, "source": "test"}}
    proj = SfmProject(tmp_path, images, cams, {}, [0, 0, 0], {"prior_sigma_m": [0.05, 0.05, 0.05],
                                                               "database": {"cameras": {"ZL034_F00600": 1}}})
    proj.root.mkdir(parents=True, exist_ok=True)
    model = {"cameras": {"ZL034": {"f0_px": fb, "label_f0_px": lab}}}
    return proj, rec, model, true_cam


def test_tangential_per_family():
    from mppp.sfm.reconstruction import fixed_camera_params, tangential_for, tangential_setting
    rt = {"N": True, "Z": False}
    assert tangential_for(rt, "NL") and tangential_for(rt, "NR_T-040-030")
    assert not tangential_for(rt, "ZL034") and not tangential_for(rt, "ZR034_f1200")
    assert tangential_for(rt, "") and not tangential_for({"N": True, "default": False}, "")
    assert tangential_for(True, "ZL034") and not tangential_for(False, "NL")
    assert tangential_setting(rt) == rt and tangential_setting(1) is True
    assert 6 not in fixed_camera_params("FULL_OPENCV", True, tangential_for(rt, "NL"))
    assert {6, 7} <= set(fixed_camera_params("FULL_OPENCV", True, tangential_for(rt, "ZL034")))


def test_bundle_adjust_holds_zcam_tangential_only(tmp_path):
    import sys
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    sys.path.insert(0, str(Path(__file__).parent))
    from helpers import _synthetic_block
    from mppp.sfm.reconstruction import bundle_adjust
    proj, rec, noise = _synthetic_block(tmp_path)
    # pretend the right eye is a Mastcam-Z camera: its p1, p2 must stay, the left eye's move
    cid = proj.settings["database"]["cameras"]
    proj.settings["database"]["cameras"] = {"NL": cid["NL"], "ZR034": cid["NR"]}
    proj.cameras["ZR034"] = proj.cameras.pop("NR")
    for c in rec.cameras.values():
        if c.model.name == "FULL_OPENCV":
            q = np.array(c.params, float)
            q[6:8] = [2e-4, -1e-4]
            c.params = q
    p0 = {k: np.array(rec.cameras[v].params, float) for k, v in proj.settings["database"]["cameras"].items()}
    ba = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, max_iterations=10,
                       refine_tangential={"N": True, "Z": False})
    p1 = {k: np.array(rec.cameras[v].params, float) for k, v in proj.settings["database"]["cameras"].items()}
    assert ba["refine_tangential"] == {"N": True, "Z": False}
    if rec.cameras[cid["NR"]].model.name == "FULL_OPENCV":
        assert np.allclose(p1["ZR034"][6:8], p0["ZR034"][6:8])
        assert not np.allclose(p1["NL"][6:8], p0["NL"][6:8])


def test_restore_frames_after_write_read(tmp_path):
    """pycolmap 4 drops deregistered frames on write/read (triangulate_points): stage 2 must bring them back."""
    pycolmap = pytest.importorskip("pycolmap")
    from mppp.sfm.reconstruction import restore_frames
    proj, rec, model, truth = _zcam_block(tmp_path)
    init = pycolmap.Reconstruction(rec)
    rec.deregister_frame(12)
    rec.deregister_frame(13)
    d = tmp_path / "m"
    d.mkdir()
    rec.write(str(d))
    r2 = pycolmap.Reconstruction()
    r2.read(str(d))
    assert 12 not in r2.frames and 13 not in r2.frames                # what broke the staged run
    assert restore_frames(r2, init, [12, 13, 1]) == 2
    for fid in (12, 13):
        r2.frames[fid].rig_from_world = init.frames[fid].rig_from_world
        r2.register_frame(fid)
    assert r2.num_reg_images() == init.num_reg_images()
    assert len(r2.images[12].points2D) == len(init.images[12].points2D)


def _block():
    """Navcam rig 1 (camera 1) with two registered frames; Mastcam-Z stereo rig 2 (cameras 2 = ZL, 3 = ZR) with one
    frame of two images, as ZCAM_RIG builds it."""
    pycolmap = pytest.importorskip("pycolmap")
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    rng = np.random.default_rng(0)
    rec = pycolmap.Reconstruction()
    for c in (1, 2, 3):
        rec.add_camera(pycolmap.Camera(camera_id=c, model="PINHOLE", width=100, height=100,
                                       params=[100.0 + c, 100.0 + c, 50, 50]))
    r1 = pycolmap.Rig(rig_id=1)
    r1.add_ref_sensor(S(1))
    rec.add_rig(r1)
    r2 = pycolmap.Rig(rig_id=2)
    r2.add_ref_sensor(S(2))
    r2.add_sensor(S(3), pycolmap.Rigid3d(pycolmap.Rotation3d(), [0.24, 0, 0]))
    rec.add_rig(r2)

    def frame(fid, rid, cams, x):
        fr = pycolmap.Frame(frame_id=fid, rig_id=rid)
        for k, c in enumerate(cams):
            fr.add_data_id(pycolmap.data_t(sensor_id=S(c), id=10 * fid + k))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(), [x, 0, 0])
        rec.add_frame(fr)
        for k, c in enumerate(cams):
            im = pycolmap.Image(name=f"i{fid}_{k}", keypoints=rng.uniform(0, 100, (40, 2)), camera_id=c,
                                image_id=10 * fid + k)
            im.frame_id = fid
            rec.add_image(im)
        rec.register_frame(fid)

    frame(1, 1, [1], 0.0)
    frame(2, 1, [1], 0.5)
    frame(3, 2, [2, 3], 0.2)
    P = rng.uniform(-1, 1, (40, 3)) + [0, 0, 10]
    for p in range(40):
        t = pycolmap.Track()
        t.add_element(10, p)
        t.add_element(20, p)
        rec.add_point3D(P[p], t, np.zeros(3, np.uint8))
    return pycolmap, rec


def test_restore_frames_brings_back_a_torn_down_stereo_rig():
    """The 30 Sep threeforks_south failure: ``Camera 28 from rig 27 not found`` at the start of stage 2."""
    pycolmap, rec = _block()
    from mppp.sfm.reconstruction import restore_frames
    init = pycolmap.Reconstruction(rec)
    rec.deregister_frame(3)
    rec.tear_down()                                   # what triangulate_points does to stage 1
    assert 2 not in rec.rigs and 2 not in rec.cameras and 3 not in rec.cameras
    assert restore_frames(rec, init, [3, 1]) == 1
    assert sorted(rec.rigs) == [1, 2] and sorted(rec.cameras) == [1, 2, 3]
    assert rec.images[30].camera_id == 2 and rec.images[31].camera_id == 3
    assert np.allclose(rec.cameras[3].params, init.cameras[3].params)
    rec.frames[3].rig_from_world = init.frames[3].rig_from_world
    rec.register_frame(3)
    assert rec.num_reg_images() == init.num_reg_images()
    assert np.allclose(rec.rigs[2].sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=3))
                       .translation, [0.24, 0, 0])


def test_restore_frames_keeps_refined_cameras_that_survived():
    """A camera still in the block (refined in stage 1) is not overwritten by its start value."""
    pycolmap, rec = _block()
    from mppp.sfm.reconstruction import restore_frames
    init = pycolmap.Reconstruction(rec)
    rec.deregister_frame(3)
    rec.tear_down()
    rec.cameras[1].params = [150.0, 150.0, 50, 50]
    restore_frames(rec, init, [3])
    assert rec.cameras[1].params[0] == 150.0


def test_staged_reconstruct_with_a_zcam_stereo_rig(tmp_path, monkeypatch):
    """Staged reconstruct end to end with the second station's frames on their own two-camera rig ("ZL"/"ZR"), and a
    triangulate that tears the block down as COLMAP's triangulate_points does."""
    pytest.importorskip("pyceres")
    pycolmap = pytest.importorskip("pycolmap")
    import copy
    from helpers import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    fids = sorted(rec0.frames)
    zf = set(fids[len(fids) // 2:])
    # move those frames onto rig 2 with cameras 3 (ZL) and 4 (ZR), copies of the Navcam ones
    new = pycolmap.Reconstruction()
    for cid, c in rec0.cameras.items():
        new.add_camera(c)
        cz = pycolmap.Camera(camera_id=cid + 2, model=c.model, width=c.width, height=c.height, params=c.params)
        new.add_camera(cz)
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

    def tearing_triangulate(rec, project, **kw):
        rec.tear_down()
        return rec

    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    monkeypatch.setattr(R, "triangulate", tearing_triangulate)
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="refine", staged=True, out_name="cahv_ba")
    assert proj.settings["reconstruction"]["staged"] is True
    assert set(rec.reg_frame_ids()) >= zf and sorted(rec.rigs)[:2] == [1, 2]
    assert {3, 4} <= set(rec.cameras)


def test_min_frame_observations_default_20_and_one_setting(tmp_path, monkeypatch):
    import inspect
    from mppp.sfm import reconstruction as R
    assert R.MIN_FRAME_OBSERVATIONS == 20
    assert R.OUTLIER_DEFAULTS["min_observations"] is None                  # follows MIN_FRAME_OBSERVATIONS
    assert inspect.signature(R.bundle_adjust).parameters["min_frame_observations"].default is None
    assert "min_frame_observations" in inspect.signature(R.reconstruct).parameters
    seen = {}

    def fake(project, *a, **k):
        seen["during"] = R.MIN_FRAME_OBSERVATIONS
        raise RuntimeError("stop")
    monkeypatch.setattr(R, "_reconstruct", fake)
    with pytest.raises(RuntimeError):
        R.reconstruct(None, min_frame_observations=12)
    assert seen["during"] == 12 and R.MIN_FRAME_OBSERVATIONS == 20        # restored after the run


def test_outlier_test_uses_the_frame_limit(tmp_path):
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    obs = min(sum(1 for q in rec.images[d.id].points2D if q.has_point3D()) for f in rec.frames.values()
              for d in f.data_ids)
    few = lambda th: [f for f in R.find_outlier_frames(rec, proj, **th)                          # noqa: E731
                      if any("observations <" in r for r in f["reasons"])]
    assert few({"min_observations": 10 ** 7})                              # everything is "too few"
    assert not few({"min_observations": 1})
    assert not few({}) or obs < R.MIN_FRAME_OBSERVATIONS                   # default: the module limit
