"""Mastcam-Z focus breathing and backlash states (mppp.sfm.zcam, backlash). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import numpy as np
import pytest


def test_focus_slope_fit_and_plot(tmp_path):
    from mppp.sfm.project import SfmProject
    from mppp.sfm.zcam import fit_focus_slopes, plot_focus_breathing
    imgs, table = [], []
    for g, a in (("ZL034", 0.05), ("ZR034", 0.04)):
        for c in (1000, 1030, 1070, 1100):
            for d in (0, 5):
                imgs.append({"name": f"{g}{c}{d}", "instrument": f"{g}_F{c:05d}", "camera_group": g,
                             "focus_count": c + d, "label_f_px": 4600 + a * (c + d - 1050) + 0.3})
            table.append({"camera": f"{g}_F{c:05d}", "group": g, "focus_count_median": c + 2.5, "focus_count_min": c,
                          "focus_count_max": c + 5, "images": 2, "observations": 500 if c != 1100 else 50,
                          "f_initial_px": 4600 + a * (c - 1047.5), "f_refined_px": 4610 + 2 * a * (c + 2.5 - 1052.5),
                          "refined": True})
    proj = SfmProject(tmp_path, imgs, {}, {}, [0, 0, 0], {})
    fits = fit_focus_slopes(proj, table, min_observations=100)
    assert fits["ZL034"]["bins"] == 4 and fits["ZL034"]["bins_fitted"] == 3          # the 50-observation bin is out
    assert abs(fits["ZL034"]["refined"]["slope_px_per_count"] - 0.10) < 1e-9
    assert abs(fits["ZL034"]["label"]["slope_px_per_count"] - 0.05) < 1e-9
    assert abs(fits["ZR034"]["label"]["slope_px_per_count"] - 0.04) < 1e-9 and fits["ZR034"]["refined"]["n"] == 3
    png = plot_focus_breathing(proj, table, fits, tmp_path / "fb.png")
    assert png.is_file() and png.stat().st_size > 10000


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


def test_focus_groups_and_image_fits(tmp_path):
    pytest.importorskip("scipy")
    from mppp.sfm.backlash import focus_groups, image_focal_fits
    proj, rec, model, truth = _zcam_block(tmp_path)
    g = focus_groups(proj.images)
    assert len(g) == 3 and sorted(len(v) for v in g.values()) == [3, 5, 6]
    fits = image_focal_fits(rec, proj)
    assert len(fits) == 14
    for n, f in fits.items():
        want = truth[n] / f["label_f_px"]
        assert abs(f["ratio_label"] - want) < 1e-3, (n, f["ratio_label"], want)
        assert f["sd_ratio"] < 5e-4


def test_classify_and_split_regular_group(tmp_path):
    pytest.importorskip("scipy")
    from mppp.sfm.backlash import classify_groups, image_focal_fits, image_states, split_by_state
    from mppp.sfm.thermal import strip_thermal_bins
    proj, rec, model, truth = _zcam_block(tmp_path)
    groups = classify_groups(proj, image_focal_fits(rec, proj), model)
    st = {g["sequence"]: g["state"] for g in groups}
    assert st == {"zcam1": "backlash", "zcam2": "backlash", "zcam3": "regular"}
    states = image_states(groups)
    reg = [n for n, s in states.items() if s == "regular"]
    new, rows = split_by_state(rec, proj, reg, hold_f_images=3)
    assert len(rows) == 1 and rows[0]["camera"] == "ZL034_F00600_reg" and rows[0]["images"] == 3 and rows[0]["f_held"]
    assert abs(rows[0]["start_f_px"] - 4680.0) < 5.0
    cid = rows[0]["camera_id"]
    assert sum(1 for im in new.images.values() if im.camera_id == cid) == 3
    assert new.num_reg_images() == rec.num_reg_images() and len(new.points3D) == len(rec.points3D)
    assert proj.settings["database"]["cameras"]["ZL034_F00600_reg"] == cid
    assert {r["instrument"] for r in proj.images if r["sequence"] == "zcam3"} == {"ZL034_F00600_reg"}
    assert "fx" in proj.cameras["ZL034_F00600_reg"]["fixed_params"]
    assert strip_thermal_bins(proj) == 1                                # a rerun starts from the focus bins again
    assert {r["instrument"] for r in proj.images} == {"ZL034_F00600"} and "ZL034_F00600_reg" not in proj.cameras


def test_backlash_stage_refines_both_states(tmp_path):
    pytest.importorskip("pyceres")
    from mppp.sfm.backlash import backlash_stage
    proj, rec, model, truth = _zcam_block(tmp_path)
    rec2, rep = backlash_stage(rec, proj, model=model, mode="split", sigma_px=0.3, loss_scale=2.0,
                               max_iterations=30, verbose=False)
    assert rep["group_counts"] == {"backlash": 2, "regular": 1, "undecided": 0}
    ids = proj.settings["database"]["cameras"]
    fb = 0.5 * sum(rec2.cameras[ids["ZL034_F00600"]].params[:2])
    fr = 0.5 * sum(rec2.cameras[ids["ZL034_F00600_reg"]].params[:2])
    # a pan from one centre holds the block's scale only weakly (no Navcam here): the two states' focal lengths
    # move together, their difference is what the split recovers
    assert abs((fb - fr) - 4680 * 0.0095) < 3.0, (fb, fr)
    assert abs(fb - 4680 * 1.0095) < 15.0 and abs(fr - 4680) < 15.0
    assert abs(rep["after"]["regular"]["ratio_median"] / rep["after"]["backlash"]["ratio_median"] - 1 / 1.0095) < 5e-4
    rec3, rep3 = backlash_stage(rec, proj, model=model, mode="report", verbose=False)
    assert rep3["cameras"] == [] and rec3 is rec
