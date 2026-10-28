"""Mastcam-Z focus breathing and backlash states (mppp.sfm.zcam, backlash). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import numpy as np
import pytest
from types import SimpleNamespace


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
    assert rep["group_counts"] == {"backlash": 2, "regular": 1, "undecided": 0, "single": 0}
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


def test_classify_with_navcam_scale_and_single_state_zoom(tmp_path):
    """v0p53: both focus states are scaled by the block's Navcam focal scale; 110 mm groups are 'single'."""
    pytest.importorskip("scipy")
    from mppp.sfm.backlash import classify_groups, group_zoom, image_focal_fits, image_states
    proj, rec, model, truth = _zcam_block(tmp_path)
    fits = image_focal_fits(rec, proj)
    g0 = {g["sequence"]: g for g in classify_groups(proj, fits, model)}
    g1 = {g["sequence"]: g for g in classify_groups(proj, fits, model, nav_scale=1.004)}
    assert abs(g1["zcam1"]["threshold_ratio"] - 1.004 * g0["zcam1"]["threshold_ratio"]) < 1e-12
    assert g1["zcam1"]["nav_scale"] == 1.004
    assert group_zoom("ZR110_F01234") == 110 and group_zoom("ZL034") == 34 and group_zoom("NL") is None
    for r in proj.images:
        r["camera_group"] = "ZL110"
    g2 = classify_groups(proj, fits, model)
    assert {g["state"] for g in g2} == {"single"}
    assert set(image_states(g2).values()) == {"single"}


def _zcam_pair_block(tmp_path, R_rig, t_rig, seed=1, noise=0.2):
    """Mastcam-Z stereo pairs at one station: ZL034 (camera 1) and ZR034 (camera 2), every image its own frame
    (focus bins: no COLMAP rig); the right images start 2 cm and 0.2 deg off."""
    pycolmap = pytest.importorskip("pycolmap")
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import SfmProject
    rng = np.random.default_rng(seed)
    W, H, f = 1648, 1200, 4720.0
    base = [f, f, 824.0, 600.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    C = np.array([0.0, 0.0, 2.0])
    P = np.column_stack([rng.uniform(-30, 30, 5000), rng.uniform(6, 40, 5000), rng.normal(0, 1.0, 5000)])
    rec = pycolmap.Reconstruction()
    images = []
    for cid in (1, 2):
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="FULL_OPENCV", width=W, height=H, params=base))
        rig = pycolmap.Rig(rig_id=cid)
        rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid))
        rec.add_rig(rig)
    cam_t = rec.cameras[1]
    tracks, k = {}, 0
    for e in range(8):
        a = np.radians(-12 + 3.5 * e)
        fwd = np.array([np.sin(a), np.cos(a), 0.0]) * np.cos(0.05) + np.array([0, 0, -np.sin(0.05)])
        right = np.array([np.cos(a), -np.sin(a), 0.0])
        RL = np.stack([right, np.cross(fwd, right), fwd])
        tL = -RL @ C
        RR, tR = R_rig @ RL, R_rig @ tL + t_rig
        for eye, cid, R, t in (("L", 1, RL, tL), ("R", 2, RR, tR)):
            k += 1
            xc = P @ R.T + t
            ok = xc[:, 2] > 1
            uv = np.full((len(P), 2), np.nan)
            uv[ok] = np.asarray(cam_t.img_from_cam(xc[ok]), float)
            vis = ok & (uv[:, 0] > 5) & (uv[:, 0] < W - 5) & (uv[:, 1] > 5) & (uv[:, 1] < H - 5)
            idx = np.nonzero(vis)[0]
            kps = uv[idx] + rng.normal(0, noise, (len(idx), 2))
            name = f"Z{eye}0_{k:04d}.png"
            im = pycolmap.Image(name=name, keypoints=kps, camera_id=cid, image_id=k)
            fr = pycolmap.Frame(frame_id=k, rig_id=cid)
            fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid), id=k))
            if eye == "R":                                          # a poor start for the right image
                dR = Rotation.from_rotvec(np.radians([0.2, -0.1, 0.1])).as_matrix()
                fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(dR @ R), dR @ t + [0.02, 0.0, 0.0])
            else:
                fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
            rec.add_frame(fr)
            im.frame_id = k
            rec.add_image(im)
            rec.register_frame(k)
            for j, p in enumerate(idx):
                tracks.setdefault(int(p), []).append((k, j))
            Cw = -R.T @ t
            images.append({"name": name, "stem": name[:-4], "instrument": f"Z{eye}034_F00600", "camera_group": f"Z{eye}034",
                           "eye": eye, "sclk_key": f"{e:010d}_000", "sol": 700, "station": "S001D0000",
                           "sequence": "zcam1", "downsample_scale": 1.0, "focus_count": 600.0,
                           "prior_C": Cw.tolist(), "prior_R_w2c": R.tolist(), "has_mask": False, "image_id": k})
    for p, els in tracks.items():
        if len(els) >= 2:
            tr = pycolmap.Track()
            for iid, j in els:
                tr.add_element(iid, j)
            rec.add_point3D(P[p], tr, np.zeros(3, np.uint8))
    cams = {f"Z{e}034_F00600": {"model": "FULL_OPENCV", "width": W, "height": H, "params": base, "group": f"Z{e}034",
                                "fixed_params": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]} for e in "LR"}
    proj = SfmProject(tmp_path, images, cams, {}, [0, 0, 0],
                      {"prior_sigma_m": [0.05, 0.05, 0.05],
                       "database": {"cameras": {"ZL034_F00600": 1, "ZR034_F00600": 2}}})
    proj.root.mkdir(parents=True, exist_ok=True)
    return proj, rec


@pytest.mark.parametrize("hold", [False, True])
def test_zcam_virtual_rig_poses_right_images(tmp_path, hold):
    """v0p53: one Mastcam-Z rig per zoom for every focus bin - the right image is posed from its left partner."""
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from mppp.sfm.reconstruction import bundle_adjust
    R_rig = Rotation.from_rotvec(np.radians([0.05, -1.2, 0.3])).as_matrix()
    t_rig = np.array([-0.2425, 0.001, 0.002])
    proj, rec = _zcam_pair_block(tmp_path, R_rig, t_rig)
    R_start = R_rig if hold else Rotation.from_rotvec(np.radians([0.0, -1.1, 0.25])).as_matrix()
    proj.settings["zcam_virtual_rig"] = {"Z034": {"R_sensor_from_ref": R_start.tolist(), "t_sensor_from_ref": t_rig.tolist(),
                                                  "hold": hold}}
    ba = bundle_adjust(rec, proj, sigma_px=0.2, loss_scale=2.0, refine_rig="rotation", max_iterations=50,
                       attitude_prior_deg=None)
    v = proj.settings["zcam_virtual_rig"]["Z034"]
    R1 = np.asarray(v["R_refined"])
    err_mdeg = 1e3 * np.degrees(np.linalg.norm(Rotation.from_matrix(R1 @ R_rig.T).as_rotvec()))
    assert err_mdeg < 5.0, err_mdeg
    if hold:
        assert np.allclose(R1, R_start)
    names = {im.name: im for im in rec.images.values()}
    for iid, im in rec.images.items():
        if im.name.startswith("ZR"):
            k = int(im.name[4:8])
            L = names[f"ZL0_{k - 1:04d}.png"]
            T = rec.frames[L.frame_id].rig_from_world
            want = R1 @ np.asarray(T.rotation.matrix())
            got = np.asarray(rec.frames[im.frame_id].rig_from_world.rotation.matrix())
            assert np.allclose(got, want, atol=1e-9)
    assert ba["final_cost"] < ba["initial_cost"]


# ---- v0p74.8: measured backlash line, one focus state per sequence ------------------------------------------------

def _rows(groups):
    """Group rows as classify_groups makes them, for the pooling tests: (key, sol, seq, eye, F, ratio, sd, state)."""
    out = []
    for key, sol, seq, eye, F, r, sd, st in groups:
        out.append({"group_key": key, "camera_group": f"Z{eye}034", "eye": eye, "sol": sol, "sequence": seq,
                    "focus_count": F, "ratio_fit": r, "sd_fit": sd, "fitted_unlisted": 3, "state": st,
                    "names": [key], "threshold_ratio": None})
    return out


def test_measured_backlash_line_follows_the_focus_count():
    from mppp.sfm.backlash import backlash_ratio_at, measured_backlash_lines
    rows = _rows([(f"g{i}", 700, "zcam1", "R", F, 1.0095 + 4.5e-6 * (F - 600), 0.001, "x")
                  for i, F in enumerate(range(100, 1300, 100))]
                 + [("reg", 700, "zcam2", "R", 300, 0.9990, 0.001, "x")])        # a regular group stays out of the fit
    ln = measured_backlash_lines(rows, 1.0, {"ZR034": 1.0151})["ZR034"]
    assert ln["source"] == "measured line" and ln["n"] == 12
    assert abs(ln["a"] - 1.0095) < 2e-5 and abs(ln["b"] - 4.5e-6) < 2e-7
    assert abs(backlash_ratio_at(ln, 200) - (1.0095 - 0.0018)) < 3e-5
    few = measured_backlash_lines(rows[:2], 1.0, {"ZR034": 1.0151})["ZR034"]
    assert few["source"].startswith("default") and abs(few["a"] - 1.0095) < 1e-12      # the model's 1.0151 is off
    few = measured_backlash_lines(rows[:2], 1.0, {"ZR034": 1.0100})["ZR034"]
    assert few["source"] == "focus model" and few["a"] == 1.0100


def test_sequence_state_overrides_the_low_focus_groups():
    """Rio Chiquito sol 1337 ZCAM09408: the low-focus groups lie between the states on their own; the sequence is
    backlash and so are they (reported as conflicts); a sequence without a decision keeps the groups' verdicts."""
    from mppp.sfm.backlash import apply_sequence_states
    rows = _rows([("a", 1337, "S1", "L", 180, 1.0057, 0.0010, "regular"),
                  ("b", 1337, "S1", "L", 300, 1.0066, 0.0010, "undecided"),
                  ("c", 1337, "S1", "L", 930, 1.0102, 0.0005, "backlash"),
                  ("d", 1337, "S1", "R", 972, 1.0128, 0.0004, "backlash"),
                  ("e", 1337, "S1", "R", 1170, None, None, "undecided"),
                  ("f", 483, "S2", "R", 1278, 1.0015, 0.0010, "regular")])
    for r in rows:
        r["threshold_ratio"] = 1.0048
    seq = apply_sequence_states(rows)
    assert seq["1337|S1"]["state"] == "backlash" and seq["1337|S1"]["conflicts"] == ["a"]
    st = {r["group_key"]: r["state"] for r in rows}
    assert st == {"a": "backlash", "b": "backlash", "c": "backlash", "d": "backlash", "e": "backlash", "f": "regular"}
    assert rows[0]["own_state"] == "regular" and rows[0]["conflict"]
    assert seq["483|S2"]["state"] == "regular"


def test_listed_images_stay_regular_in_a_backlash_sequence(tmp_path):
    pytest.importorskip("scipy")
    from mppp.sfm.backlash import classify_groups, image_focal_fits, image_states
    proj, rec, model, truth = _zcam_block(tmp_path)
    for r in proj.images:                                    # zcam3 (f = label) becomes part of zcam1's sequence
        if r["sequence"] == "zcam3":
            r["sequence"], r["focus_count"] = "zcam1", 300.0
    fits = image_focal_fits(rec, proj)
    rep = {}
    groups = classify_groups(proj, fits, model, known=False, report=rep)
    st = image_states(groups)
    reg = [n for n, s in st.items() if s == "regular"]
    assert reg == [] and len(rep["sequence_conflicts"]) == 1           # not on the list: the sequence wins
    names3 = [r["name"] for r in proj.images if r["focus_count"] == 300.0]
    known = {"entries": {n[:-4].upper() for n in names3}, "sol_sequences": set(), "clocks": set(), "uncertain": [],
             "file": None}
    groups = classify_groups(proj, fits, model, known=known)
    st = image_states(groups)
    assert sorted(n for n, s in st.items() if s == "regular") == sorted(names3)   # the list's exceptions stay regular
    g3 = [g for g in groups if g["focus_count"] == 300.0][0]
    assert g3["all_listed"] and g3["ratio_fit"] is None and not g3.get("conflict")


# the low-backlash list (mppp.sfm.backlash)


def test_low_backlash_list_shipped_and_matching(tmp_path):
    from mppp.sfm.backlash import is_low_backlash, load_low_backlash, low_backlash_fingerprint, low_backlash_path
    lst = load_low_backlash()
    assert low_backlash_path().is_file() and len(lst["entries"]) >= 60 and lst["clocks"]
    assert low_backlash_fingerprint() and len(lst["uncertain"]) == 1
    row = lambda stem, g: {"stem": stem, "camera_group": g}                     # noqa: E731
    # a sequence at one zoom (ZCAM08045_034), any product type / version
    assert is_low_backlash(row("ZLF_0091_0675032123_223EBY_N0040136ZCAM08045_034085J01", "ZL034"), lst)
    # a full product name: same eye and clock, other product type
    assert is_low_backlash(row("ZRF_0094_0675283792_738EBY_N0040136ZCAM03144_034085J01", "ZR034"), lst)
    # (sol, clock) fragments at 48 mm; never at 110 mm (one focus state)
    assert is_low_backlash(row("ZLF_0121_0677678747_053EBY_N0041250ZCAM08114_048085J01", "ZL048"), lst)
    assert not is_low_backlash(row("ZLF_0121_0677678823_053EBY_N0041250ZCAM08114_110085J01", "ZL110"), lst)
    assert not is_low_backlash(row("ZLF_0363_0700000000_000EBY_N0040136ZCAM08394_034085J01", "ZL034"), lst)
    # own list: comments, separators, sol:sequence, extensions, too-short fragments ignored
    f = tmp_path / "low.csv"
    f.write_text("# list\nZCAM01234_063, 12:ZCAM00001\nABC\nZL0_0005_0600000000_000RAD_N0000000ZCAM00002_034085A01.IMG?\n")
    own = load_low_backlash(f)
    assert "ABC" not in own["entries"] and (12, "ZCAM00001") in own["sol_sequences"] and own["uncertain"]
    assert is_low_backlash({"stem": "x", "sequence": "ZCAM00001", "sol": 12, "camera_group": "ZL034"}, own)
    assert is_low_backlash(row("ZLF_0005_0600000000_000EBY_N0000000ZCAM00002_034085J01", "ZL034"), own)
    assert is_low_backlash(row("ZRF_0010_0610000000_000EBY_N0000000ZCAM01234_063085J01", "ZR063"), own)
    assert low_backlash_fingerprint(tmp_path / "none.csv") is None
    # 0.95.0 (Christian, 28 Oct): sols at one zoom - Overlook Mountain's 48 mm frames (sols 1150-1155) are low-state
    assert (1150, 1155, 48) in lst["sol_zooms"]
    assert is_low_backlash(row("ZLF_1152_0769000000_000EBY_N0520000ZCAM03014_048085J01", "ZL048"), lst)
    assert is_low_backlash({"name": "ZR0_1150_0768900000_000RAD_N0520000ZCAM05065_048085A03", "sol": 1150,
                            "sequence": "ZCAM05065", "camera_group": "ZR048_F02000"}, lst)
    assert not is_low_backlash(row("ZLF_1152_0769000000_000EBY_N0520000ZCAM03014_034085J01", "ZL034"), lst)
    assert not is_low_backlash(row("ZLF_1156_0769400000_000EBY_N0520000ZCAM03014_048085J01", "ZL048"), lst)
    f.write_text("1150-1155:048\n-1151:063\n200:034\n")
    own = load_low_backlash(f)
    assert own["sol_zooms"] == [(1150, 1155, 48), (200, 200, 34)] and own["excluded"]["sol_zooms"] == [(1151, 1151, 63)]
    assert is_low_backlash(row("ZRF_0200_0700000000_000EBY_N0000000ZCAM00009_034085J01", "ZR034"), own)
    assert not is_low_backlash(row("ZRF_1151_0768950000_000EBY_N0000000ZCAM00009_063085J01", "ZR063"), own)


# the fx, fy, cx, cy lines per focus state and the low-backlash found list (v0p73)


def _zcam_setup(monkeypatch, with_regular=True):
    import pycolmap
    from mppp.sfm import zcam
    rec = pycolmap.Reconstruction()
    cams, db, obs = {}, {}, {}
    rng = np.random.default_rng(0)
    cid = 1
    truth = {"fx": (4700.0, 0.05), "fy": (4702.0, 0.05), "cx": (820.0, 0.002), "cy": (600.0, -0.001)}
    for eye in ("L", "R"):
        g = f"Z{eye}034"
        for st, foci in (("backlash", [100, 300, 500, 700, 900]), ("regular", [200, 600, 800] if with_regular else [])):
            for f in foci:
                k = 1.0 / 1.009 if st == "regular" else 1.0
                p = [truth["fx"][0] * k + truth["fx"][1] * (f - 600) + rng.normal(0, 0.3),
                     truth["fy"][0] * k + truth["fy"][1] * (f - 600) + rng.normal(0, 0.3),
                     truth["cx"][0] + truth["cx"][1] * (f - 600) + (3.0 if eye == "R" else 0.0) + rng.normal(0, 0.05),
                     truth["cy"][0] + truth["cy"][1] * (f - 600) + rng.normal(0, 0.05), 0, 0, 0, 0]
                rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200,
                                               params=np.array(p, float)))
                key = f"{g}_F{f:05d}" + ("_reg" if st == "regular" else "")
                cams[key] = {"group": g, "focus_count_median": float(f), "backlash_state": st}
                db[key] = cid
                obs[cid] = 1000
                cid += 1
    proj = SimpleNamespace(cameras=cams, images=[], settings={
        "database": {"cameras": db}, "zcam_line_params": {"ZL": ["fx", "fy"], "ZR": ["fx", "fy", "cx", "cy"]},
        "zcam_line_form": {"ZL": {"fx": "free", "fy": "free"},
                           "ZR": {"fx": "free", "fy": "free", "cx": "free", "cy": "free"}}})   # free lines (a study)
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    model = {"cameras": {g: {"reference_focus": 600.0, "f0_px": 4701.0, "label_f0_px": 4659.0,
                             "slope_px_per_count": 0.05, "aspect": 1.0} for g in ("ZL034", "ZR034")}}
    return rec, proj, model, db


def test_param_lines_per_state(monkeypatch):
    from mppp.sfm.zcam import focus_line_lines, refit_focus_lines
    rec, proj, model, db = _zcam_setup(monkeypatch)
    out = refit_focus_lines(rec, proj, model=model)
    zr = out["ZR034"]["lines"]
    assert set(zr) == {"backlash", "regular"}
    assert zr["backlash"]["fx"]["a"] == pytest.approx(4700.0, abs=0.5)
    assert zr["backlash"]["fx"]["b"] == pytest.approx(0.05, abs=0.003)
    assert zr["backlash"]["cx"]["a"] == pytest.approx(823.0, abs=0.1)
    assert zr["regular"]["fx"]["a"] == pytest.approx(4700.0 / 1.009, abs=0.6)
    assert set(out["ZL034"]["lines"]["backlash"]) == {"fx", "fy"}            # ZL: the reference principal point
    p = rec.cameras[db["ZR034_F00300"]].params                                # every bin set on its state's line
    ln = zr["backlash"]
    assert p[0] == pytest.approx(ln["fx"]["a"] + ln["fx"]["b"] * (300 - 600))
    assert p[2] == pytest.approx(ln["cx"]["a"] + ln["cx"]["b"] * (300 - 600))
    assert any("regular" in s for s in focus_line_lines(out))


def test_param_lines_regular_without_bins_follows_backlash(monkeypatch):
    from mppp.sfm.zcam import refit_focus_lines
    rec, proj, model, db = _zcam_setup(monkeypatch, with_regular=False)
    out = refit_focus_lines(rec, proj, model=model)
    assert "skipped" in out["ZL034"]["lines"]["regular"]


def test_found_list_append_and_load(tmp_path):
    from mppp.sfm.backlash import append_found, is_low_backlash, load_low_backlash, low_backlash_fingerprint
    f = tmp_path / "zcam_low_backlash_found.csv"
    g = {"names": ["ZL0_0500_0711000000_111RAD_N0261154ZCAM07999_0340LMA01.png"], "sol": 500, "sequence": "ZCAM07999",
         "focus_count": 400, "ratio": 1.0004, "z": -9.0}
    fp0 = low_backlash_fingerprint(None, f)
    assert append_found(f, [g], site="test") == 1
    assert append_found(f, [g], site="test") == 0                            # not twice
    lb = load_low_backlash(None, found=f)
    assert lb["found"] == 1 and lb["found_file"] == str(f)
    row = {"stem": "ZL0_0500_0711000000_111RAD_N0261154ZCAM07999_0340LMA01", "camera_group": "ZL034", "sol": 500,
           "sequence": "ZCAM07999"}
    assert is_low_backlash(row, lb)
    assert not is_low_backlash(row, load_low_backlash(None))
    assert low_backlash_fingerprint(None, f) != fp0


def test_backlash_mode_list_split_accepted():
    from mppp.sfm.backlash import backlash_stage
    with pytest.raises(ValueError, match="list_split"):
        backlash_stage(None, None, mode="bogus")


# the virtual rig in the BA, k1/k2 lines, the staged N+Z reconstruct (v0p74)


def test_zcam_rig_full_refines_rotation_and_baseline(tmp_path):
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from mppp.sfm.reconstruction import bundle_adjust
    R_rig = Rotation.from_rotvec(np.radians([0.05, -1.2, 0.3])).as_matrix()
    t_rig = np.array([-0.2425, 0.001, 0.002])
    proj, rec = _zcam_pair_block(tmp_path, R_rig, t_rig)
    R0 = Rotation.from_rotvec(np.radians([0.0, -1.1, 0.25])).as_matrix()
    t0 = t_rig + np.array([0.006, 0.0, 0.0])
    proj.settings["zcam_virtual_rig"] = {"Z034": {"R_sensor_from_ref": R0.tolist(), "t_sensor_from_ref": t0.tolist(),
                                                  "hold": True}}            # held - "full" overrides it
    bundle_adjust(rec, proj, sigma_px=0.2, loss_scale=2.0, refine_rig=False, max_iterations=80, attitude_prior_deg=None,
                  refine_zcam_rig="full", zcam_rig_translation_sigma_m=0.05)
    v = proj.settings["zcam_virtual_rig"]["Z034"]
    R1, t1 = np.asarray(v["R_refined"]), np.asarray(v["t_refined"])
    err_mdeg = 1e3 * np.degrees(np.linalg.norm(Rotation.from_matrix(R1 @ R_rig.T).as_rotvec()))
    assert err_mdeg < 5.0, err_mdeg
    assert abs(np.linalg.norm(t1) - np.linalg.norm(t_rig)) < abs(np.linalg.norm(t0) - np.linalg.norm(t_rig))


def test_zcam_rig_hold_stays_without_fit(tmp_path):
    pytest.importorskip("pyceres")
    from scipy.spatial.transform import Rotation
    from mppp.sfm.reconstruction import bundle_adjust
    R_rig = Rotation.from_rotvec(np.radians([0.05, -1.2, 0.3])).as_matrix()
    t_rig = np.array([-0.2425, 0.001, 0.002])
    proj, rec = _zcam_pair_block(tmp_path, R_rig, t_rig)
    R0 = Rotation.from_rotvec(np.radians([0.0, -1.1, 0.25])).as_matrix()
    proj.settings["zcam_virtual_rig"] = {"Z034": {"R_sensor_from_ref": R0.tolist(), "t_sensor_from_ref": t_rig.tolist(),
                                                  "hold": True}}
    bundle_adjust(rec, proj, sigma_px=0.2, refine_rig=False, max_iterations=20, attitude_prior_deg=None)
    assert np.allclose(proj.settings["zcam_virtual_rig"]["Z034"]["R_refined"], R0)


def test_k1_k2_lines(monkeypatch):
    import pycolmap
    from mppp.sfm import zcam
    rec = pycolmap.Reconstruction()
    cams, db, obs = {}, {}, {}
    for cid, f in enumerate([100, 300, 500, 700, 900], 1):
        p = [4700 + 0.05 * (f - 600), 4702 + 0.05 * (f - 600), 820.0, 600.0, -0.47 + 2e-5 * (f - 600), 0.7, 0, 0]
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200, params=np.array(p)))
        cams[f"ZL034_F{f:05d}"] = {"group": "ZL034", "focus_count_median": float(f), "backlash_state": "backlash"}
        db[f"ZL034_F{f:05d}"] = cid
        obs[cid] = 500
    proj = SimpleNamespace(cameras=cams, images=[], settings={"database": {"cameras": db},
                                                              "zcam_line_params": {"ZL": ["fx", "fy", "k1", "k2"]},
                                                              "zcam_line_form": {"ZL": {p: "free" for p in ("fx", "fy", "k1", "k2")}}})   # free lines (a study)
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    out = zcam.refit_focus_lines(rec, proj, model={"cameras": {"ZL034": {"reference_focus": 600.0, "f0_px": 4700.0,
                                                                          "slope_px_per_count": 0.05}}})
    k1 = out["ZL034"]["lines"]["backlash"]["k1"]
    assert k1["a"] == pytest.approx(-0.47, abs=1e-6) and k1["b"] == pytest.approx(2e-5, rel=1e-3)


def _nz_block(tmp_path):
    """The synthetic block: the first station's exposures as Navcam rig frames, the second station's as Mastcam-Z
    images (ZL, ZR each its own frame, posed through the virtual rig), in focus bins, with one focus group whose labels
    say low-backlash (f / label f = 1.000) but that is not on the list."""
    import pycolmap
    from helpers import _build_rec, _synthetic
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path, noise_native_px=0.1)
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    fids = sorted(rec0.frames)
    zf = set(fids[len(fids) // 2:])
    by_id = {r["image_id"]: r for r in proj.images}
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    new = pycolmap.Reconstruction()
    for c in rec0.cameras.values():
        new.add_camera(c)
    new.add_rig(rec0.rigs[1])
    base = {e: np.asarray(rec0.cameras[cid].params, float) for e, cid in (("L", 1), ("R", 2))}
    focus = {}
    for n, fid in enumerate(sorted(zf)):
        focus[fid] = (200.0, 200.0, 500.0, 500.0, 800.0, 800.0)[n % 6]
    db = {"NL": 1, "NR": 2}
    next_cid, next_rid, next_fid = 10, 10, 1000
    bins = {}
    for fid in fids:
        fr = rec0.frames[fid]
        if fid not in zf:
            nf = pycolmap.Frame(frame_id=fid, rig_id=1)
            for d in fr.data_ids:
                nf.add_data_id(d)
            nf.rig_from_world = fr.rig_from_world
            new.add_frame(nf)
            for d in fr.data_ids:
                im = rec0.images[d.id]
                ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]), camera_id=im.camera_id,
                                    image_id=d.id)
                ni.frame_id = fid
                new.add_image(ni)
            new.register_frame(fid)
            continue
        for d in fr.data_ids:
            im = rec0.images[d.id]
            r = by_id[d.id]
            eye = r["eye"]
            key = f"Z{eye}034_F{int(focus[fid]):05d}"
            if key not in bins:
                cid = next_cid
                next_cid += 1
                new.add_camera(pycolmap.Camera(camera_id=cid, model="FULL_OPENCV", width=5120, height=3840,
                                               params=base[eye].copy()))
                rig = pycolmap.Rig(rig_id=next_rid)
                rig.add_ref_sensor(S(cid))
                new.add_rig(rig)
                bins[key] = (cid, next_rid)
                next_rid += 1
                db[key] = cid
                held = ["p1", "p2", "k3"] + (["cx", "cy"] if eye == "L" else [])
                proj.cameras[key] = {"model": "FULL_OPENCV", "width": 5120, "height": 3840,
                                     "params": list(map(float, base[eye])), "group": f"Z{eye}034",
                                     "focus_count_median": focus[fid], "backlash_state": "backlash",
                                     "fixed_params": held, "n_images": 0}
            cid, rid = bins[key]
            proj.cameras[key]["n_images"] += 1
            T = fr.rig_from_world if eye == "L" else (rec0.rigs[1].sensor_from_rig(S(2)) * fr.rig_from_world)
            nf = pycolmap.Frame(frame_id=next_fid, rig_id=rid)
            nf.add_data_id(pycolmap.data_t(sensor_id=S(cid), id=d.id))
            nf.rig_from_world = T
            new.add_frame(nf)
            ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]), camera_id=cid, image_id=d.id)
            ni.frame_id = next_fid
            new.add_image(ni)
            new.register_frame(next_fid)
            next_fid += 1
            regular = focus[fid] == 800.0 and fid == max(zf)            # the last exposure: a low-backlash group
            f_true = float(base[eye][0])
            r.update({"instrument": key, "camera_group": f"Z{eye}034", "focus_count": focus[fid],
                      "label_f_px": f_true if regular else f_true / 1.0086, "focus_state": "backlash",
                      "sequence": "ZCAM09001" if regular else "ZCAM08000", "sol": 600,
                      "label_c_px": [float(base[eye][2]), float(base[eye][3])]})
    for pid, p in rec0.points3D.items():
        new.add_point3D(p.xyz, p.track, np.zeros(3, np.uint8))
    R_rel, t_rel = rigT
    proj.settings.update({
        "database": {"cameras": db}, "zcam_rig_fit": "full",
        # 0.91: the pipeline's default (zcam-v2, p1 and p2 constant lines) - Rochette Z34 failed with KeyError 'p1'
        "zcam_line_params": {"ZL": ["fx", "fy", "k1", "k2", "p1", "p2"],
                             "ZR": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]},
        "zcam_virtual_rig": {"Z034": {"ref": "ZL034", "sensor": "ZR034", "R_sensor_from_ref": np.asarray(R_rel).tolist(),
                                      "t_sensor_from_ref": (np.asarray(t_rel) + [0.003, 0, 0]).tolist(), "hold": True}},
        "zcam_low_backlash": {"found_file": str(tmp_path / "zcam_low_backlash_found.csv")},
        "zcam_line_min_observations": 20})
    return proj, new


def test_staged_nz_reconstruct_v0p74(tmp_path, monkeypatch):
    """Staged reconstruct with every v0p74 Mastcam-Z path: focus bins with fx, fy, (cx, cy), k1, k2 free, per-state
    lines, the virtual rig refined in rotation and baseline while the bins are held, list_split classification with the
    found list."""
    pytest.importorskip("pyceres")
    import copy
    from mppp.sfm import reconstruction as R
    proj, rec = _nz_block(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec))
    tracks = {pid: [(e.image_id, e.point2D_idx) for e in p.track.elements] for pid, p in rec.points3D.items()}

    def retriangulate(r, project, **kw):
        """the true tracks back on every registered image (what COLMAP's triangulation would find)"""
        import pycolmap
        reg = {int(i) for i in r.reg_image_ids()} if hasattr(r, "reg_image_ids") else \
            {d.id for f in r.reg_frame_ids() for d in r.frames[f].data_ids}
        for pid, els in tracks.items():
            if pid not in r.points3D:
                continue
            have = {(e.image_id, e.point2D_idx) for e in r.points3D[pid].track.elements}
            for iid, j in els:
                if iid in reg and (iid, j) not in have and not r.images[iid].points2D[j].has_point3D():
                    r.add_observation(pid, pycolmap.TrackElement(iid, j))
        return r
    monkeypatch.setattr(R, "triangulate", retriangulate)
    out = R.reconstruct(proj, sigma_px=0.1, schedule=((24.0, 10.0, 8.0), (12.0, 4.0, 4.0), (8.0, 3.0, 3.0)),
                        max_iterations=15, verbose=False, navcam_intrinsics="refine", staged=True, out_name="cahv_ba",
                        zcam_focus_line=True, zcam_line_cycles=1, zcam_backlash="list_split", pose_guided=False,
                        gui_native=False, attitude_prior_deg=None)
    st = proj.settings
    lines = st["zcam_focus_lines"]
    assert set(lines) == {"ZL034", "ZR034"}
    assert set(lines["ZR034"]["lines"]["backlash"]) == {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"}
    assert set(lines["ZL034"]["lines"]["backlash"]) == {"fx", "fy", "k1", "k2", "p1", "p2"}
    v = st["zcam_virtual_rig"]["Z034"]
    assert v.get("R_refined") is not None and v.get("t_refined") is not None
    assert not np.allclose(v["t_refined"], v["t_sensor_from_ref"])           # the baseline moved (held flag overridden)
    zb = st["zcam_backlash"]
    assert zb["mode"] == "list_split"
    assert any(g.get("sequence") == "ZCAM09001" for g in zb.get("new_regular_groups") or []), zb.get("group_counts")
    assert "ZCAM09001" in (tmp_path / "zcam_low_backlash_found.csv").read_text()
    assert out.num_reg_images() > 0
    # 0.95.0: every refit records how far the lines moved the bins from where the round started (here from the
    # model they were held at in round 1); below zcam_line_settle_px (0.1 px) the remaining round is skipped
    log = st["reconstruction"]["log"]
    rounds = [e for e in log if e.get("stage") == 2 and "zcam_focus_lines" in e]
    assert rounds and rounds[0]["line_change_px"]["max_px"] is not None and set(rounds[0]["line_change_px"]["groups"]) == {"ZL034", "ZR034"}
    settled = [e for e in log if e.get("lines_settled")]
    assert (len(settled) == 1 and settled[0]["lines_settled"]["rounds_skipped"] == 1) == (rounds[0]["line_change_px"]["max_px"] < 0.1)


def test_stage_2_skips_its_last_round_when_the_focus_lines_settle(tmp_path, monkeypatch):
    """0.95.0 (Christian, 28 Oct): with zcam_line_settle_px the remaining rounds are skipped once a refit moves the
    bins by less than it (two rounds instead of three); 0 never skips."""
    pytest.importorskip("pyceres")
    import copy
    from mppp.sfm import reconstruction as R
    from mppp.sfm.reconstruction import ZCAM_LINE_SETTLE_PX
    assert ZCAM_LINE_SETTLE_PX == 0.1
    from mppp.pipeline import Settings
    assert Settings().zcam_line_settle_px == 0.1 and "zcam_line_settle_px" not in Settings().key_dict()
    assert Settings(include_zcam=True).key_dict()["zcam_line_settle_px"] == 0.1
    proj, rec = _nz_block(tmp_path)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec))
    monkeypatch.setattr(R, "triangulate", lambda r, project, **kw: r)
    sched = ((24.0, 10.0, 8.0), (12.0, 4.0, 4.0), (8.0, 3.0, 3.0))
    kw = dict(sigma_px=0.1, schedule=sched, max_iterations=10, verbose=False, navcam_intrinsics="hold", staged=True,
              out_name="cahv_ba", zcam_focus_line=True, zcam_line_cycles=2, zcam_backlash=None, pose_guided=False,
              gui_native=False, attitude_prior_deg=None, exclude_outliers=False)
    R.reconstruct(proj, zcam_line_settle_px=1e9, **kw)                   # any move counts as settled
    log = _stage2_log(proj)
    settled = [e for e in log if e.get("lines_settled")]
    assert len(settled) == 1 and settled[0]["lines_settled"]["round"] == 2 and settled[0]["lines_settled"]["rounds_skipped"] == 1
    assert max(e["round"] for e in log if "round" in e) == 2            # round 3 skipped
    assert settled[0]["lines_settled"]["max_px"] == [e for e in log if "zcam_focus_lines" in e][-1]["line_change_px"]["max_px"]
    proj2, rec2 = _nz_block(tmp_path / "b")
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec2))
    R.reconstruct(proj2, zcam_line_settle_px=0, **kw)                    # never
    log2 = _stage2_log(proj2)
    assert not [e for e in log2 if e.get("lines_settled")] and max(e["round"] for e in log2 if "round" in e) == 3


def _stage2_log(proj):
    return [e for e in proj.settings["reconstruction"]["log"] if e.get("stage") == 2]


# the Mastcam-Z line form, the bin-camera fallback of the focus-state classification (v0p75)


def _bins(monkeypatch, pp0=(820.0, 600.0), cx_bins=820.0, form=None):
    import pycolmap
    from mppp.sfm import zcam
    rec = pycolmap.Reconstruction()
    cams, db, obs = {}, {}, {}
    rng = np.random.default_rng(1)
    cid = 1
    for eye in ("L", "R"):
        g = f"Z{eye}034"
        for st, foci in (("backlash", [100, 300, 500, 700, 900, 1100]), ("regular", [200, 600, 800])):
            for f in foci:
                k = 1.0 / 1.0095 if st == "regular" else 1.0
                p = [4700.0 * k + 0.050 * (f - 600) + rng.normal(0, 0.3),          # fx slope 0.050
                     4710.0 * k + 0.060 * (f - 600) + rng.normal(0, 0.3),          # fy slope 0.060: the common one is between
                     cx_bins + 0.0005 * (f - 600) + rng.normal(0, 0.05),
                     600.0 + 0.0017 * (f - 600) + rng.normal(0, 0.05),
                     -0.47 + 3e-5 * (f - 600), 0.7, 0, 0]
                rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200, params=np.array(p, float)))
                key = f"{g}_F{f:05d}" + ("_reg" if st == "regular" else "")
                cams[key] = {"group": g, "focus_count_median": float(f), "backlash_state": st}
                db[key] = cid
                obs[cid] = 1000
                cid += 1
    settings = {"database": {"cameras": db},
                "zcam_line_params": {"ZL": ["fx", "fy", "k1", "k2"], "ZR": ["fx", "fy", "cx", "cy", "k1", "k2"]}}
    if form is not None:
        settings["zcam_line_form"] = form
    proj = SimpleNamespace(cameras=cams, images=[], settings=settings)
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    model = {"cameras": {g: {"reference_focus": 600.0, "f0_px": 4705.0, "label_f0_px": 4659.0, "slope_px_per_count": 0.05,
                             "aspect": 1.002, "pp0_px": list(pp0) if g == "ZR034" else [876.7, 559.1]}
                         for g in ("ZL034", "ZR034")}}
    return rec, proj, model, db


def test_line_form_lookup():
    from mppp.sfm.zcam import LINE_FORM, line_form_for
    p = SimpleNamespace(settings={"zcam_line_params": {"ZR": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]}})
    assert line_form_for(p, "ZR034") == LINE_FORM["ZR"]
    assert LINE_FORM["ZR"]["p1"] == "constant"                      # zcam-v2: p1, p2 constant over focus
    p.settings["zcam_line_form"] = {"ZR": {k: "free" for k in ("fx", "fy", "cx", "cy", "k1", "k2")}}
    assert set(line_form_for(p, "ZR034").values()) == {"free"}
    p.settings["zcam_line_form"] = {"ZR": {"cy": "constant"}}
    assert line_form_for(p, "ZR034")["cy"] == "constant" and line_form_for(p, "ZR034")["fx"] == "free"
    p.settings["zcam_line_form"] = {"ZR": {"cy": "odd"}}
    with pytest.raises(ValueError, match="unknown forms"):
        line_form_for(p, "ZR034")


def test_refit_lines_on_the_v0p75_form(monkeypatch):
    from mppp.sfm.zcam import refit_focus_lines
    rec, proj, model, db = _bins(monkeypatch)
    out = refit_focus_lines(rec, proj, model=model)
    zr, zl = out["ZR034"]["lines"], out["ZL034"]["lines"]
    assert out["ZR034"]["form"]["cx"] == "pinned" and out["ZR034"]["form"]["fx"] == "common"
    # one slope for fx and fy per eye, between the two true slopes; the regular state follows it
    for ln in (zr, zl):
        assert ln["backlash"]["fx"]["b"] == ln["backlash"]["fy"]["b"]
        assert 0.050 < ln["backlash"]["fx"]["b"] < 0.060 and ln["backlash"]["fx"]["slope_source"].startswith("common slope")
        assert ln["regular"]["fx"]["b"] == ln["backlash"]["fx"]["b"] and ln["regular"]["fx"]["a"] == pytest.approx(4700 / 1.0095, abs=0.6)
        assert ln["backlash"]["k1"]["b"] == 0.0 and ln["backlash"]["k1"]["a"] == pytest.approx(-0.47, abs=1e-6)   # constant
        assert ln["backlash"]["k2"]["b"] == 0.0
    assert zr["backlash"]["cx"] == {**zr["backlash"]["cx"], "a": 820.0, "b": 0.0}         # pinned, no slope
    assert zr["backlash"]["cx"]["a_source"].startswith("pinned")
    assert zr["backlash"]["cy"]["a"] == 600.0 and zr["backlash"]["cy"]["b"] == pytest.approx(0.0017, abs=2e-4)   # pinned level, slope
    assert "cx" not in zl["backlash"]                                                     # ZL: the reference pp
    p = rec.cameras[db["ZR034_F00300"]].params                                           # every bin set on the lines
    assert p[2] == pytest.approx(820.0) and p[3] == pytest.approx(600.0 + zr["backlash"]["cy"]["b"] * (300 - 600))
    assert p[4] == pytest.approx(zr["backlash"]["k1"]["a"]) and p[0] == pytest.approx(zr["backlash"]["fx"]["a"] + zr["backlash"]["fx"]["b"] * (300 - 600))


def test_refit_lines_with_sequence_offsets(monkeypatch):
    """0.95.0 (Christian, 27 Oct: different sequences sit at different linear offsets with the same slope): on sequence
    bins the backlash-state fx, fy lines get one intercept per sequence - two sequences +12 / -12 px off the block's
    line are recovered, a thin one (4 frames) is held at 0, the slope is the common one, the bins are set on their
    sequence's line; the k1 line and the pinned centre are untouched."""
    import pycolmap
    from mppp.sfm import zcam
    from mppp.sfm.zcam import refit_focus_lines
    rec = pycolmap.Reconstruction()
    cams, db, obs = {}, {}, {}
    rng = np.random.default_rng(3)
    cid = 1
    seqs = {"ZCAM00001": (12.0, 300, 8), "ZCAM00002": (-12.0, 500, 8), "ZCAM00003": (30.0, 700, 1)}   # offset, sol, frames per bin
    for seq, (off, sol, nim) in seqs.items():
        for f in (100, 400, 700, 1000):
            p = [4700.0 + off + 0.050 * (f - 600) + rng.normal(0, 0.2), 4710.0 + off + 0.050 * (f - 600) + rng.normal(0, 0.2),
                 820.0, 600.0 + 0.0017 * (f - 600), -0.47, 0.7, 0, 0]
            rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200, params=np.array(p, float)))
            key = f"ZR034_F{f:05d}_s{sol:04d}_{seq}"
            cams[key] = {"group": "ZR034", "focus_count_median": float(f), "backlash_state": "backlash",
                         "sequence": seq, "sequence_sol": sol, "n_images": nim}
            db[key] = cid
            obs[cid] = 1000
            cid += 1
    proj = SimpleNamespace(cameras=cams, images=[], settings={"database": {"cameras": db},
                                                            "zcam_line_params": {"ZR": ["fx", "fy", "cx", "cy", "k1", "k2"]}})
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    model = {"cameras": {"ZR034": {"reference_focus": 600.0, "f0_px": 4705.0, "label_f0_px": 4659.0, "slope_px_per_count": 0.05,
                                   "aspect": 1.002, "pp0_px": [820.0, 600.0]}}}
    out = refit_focus_lines(rec, proj, model=model)
    ln = out["ZR034"]["lines"]["backlash"]
    sq = ln["fx"]["sequences"]
    assert set(sq) == {"s300|ZCAM00001", "s500|ZCAM00002", "s700|ZCAM00003"}
    assert sq["s300|ZCAM00001"]["offset_px"] == pytest.approx(12.0, abs=0.5) and not sq["s300|ZCAM00001"]["held"]
    assert sq["s500|ZCAM00002"]["offset_px"] == pytest.approx(-12.0, abs=0.5) and sq["s500|ZCAM00002"]["sd_px"] > 0
    assert sq["s700|ZCAM00003"]["held"].startswith("held at 0: 4 frames") and sq["s700|ZCAM00003"]["offset_px"] == 0.0
    assert ln["fx"]["b"] == pytest.approx(0.050, abs=0.002) and ln["fy"]["b"] == ln["fx"]["b"]
    assert ln["fx"]["a"] == pytest.approx(4700.0, abs=1.0)               # the weighted mean of the two free ones
    assert 10.0 < ln["fx"]["rms"] < 20.0 and ln["k1"]["a"] == pytest.approx(-0.47) and ln["cx"]["a"] == 820.0
    assert "sequences" not in ln["k1"] and "sequences" not in ln["cx"]
    p1 = rec.cameras[db["ZR034_F00400_s0300_ZCAM00001"]].params           # on its sequence's line
    assert p1[0] == pytest.approx(ln["fx"]["a"] + 12.0 + ln["fx"]["b"] * (400 - 600), abs=0.6)
    p3 = rec.cameras[db["ZR034_F00400_s0700_ZCAM00003"]].params           # the thin one: on the block's line
    assert p3[0] == pytest.approx(ln["fx"]["a"] + ln["fx"]["b"] * (400 - 600), abs=0.01)
    assert out["ZR034"]["f0_px"] == pytest.approx(np.sqrt(ln["fx"]["a"] * ln["fy"]["a"]))


def test_refit_lines_only_for_the_zooms_with_real_models(monkeypatch):
    """0.95.0 (Christian, 28 Oct): the bins of 79 and 110 mm stay off the focus lines (their refined values kept) until
    those zooms have real models; zcam_line_zooms in the project settings says which zooms go on lines."""
    import pycolmap
    from mppp.sfm import zcam
    from mppp.sfm.zcam import LINE_ZOOMS, refit_focus_lines
    assert LINE_ZOOMS == (34, 48, 63)
    rec, proj, model, db = _bins(monkeypatch)
    # a 79 mm bin beside the 34 mm ones
    cid = max(db.values()) + 1
    rec.add_camera(pycolmap.Camera(camera_id=cid, model="OPENCV", width=1648, height=1200,
                                   params=np.array([11000.0, 11000.0, 824.0, 600.0, -0.3, 0, 0, 0], float)))
    proj.cameras["ZR079_F00600"] = {"group": "ZR079", "focus_count_median": 600.0, "backlash_state": "backlash"}
    db["ZR079_F00600"] = cid
    obs = zcam._observations_per_camera(rec)
    obs[cid] = 1000
    monkeypatch.setattr(zcam, "_observations_per_camera", lambda rec: obs)
    out = refit_focus_lines(rec, proj, model=model)
    assert "skipped" in out["ZR079"] and "not on focus lines" in out["ZR079"]["skipped"]
    assert rec.cameras[cid].params[0] == 11000.0                            # untouched
    assert "lines" in out["ZR034"] and "skipped" not in out["ZR034"]
    proj.settings["zcam_line_zooms"] = [34, 79]
    out = refit_focus_lines(rec, proj, model=model)
    assert "skipped" not in out["ZR079"] or "not on focus lines" not in out["ZR079"].get("skipped", "")


def test_refit_lines_pin_the_regular_state_at_its_own_principal_point(monkeypatch):
    """0.95.0 (Christian, 27 Oct: the right eye's regular principal-point offset): a ZR regular-state line with its own
    bins is pinned at pp0 plus the model's regular_pp_offset_px, as the no-own-line path already added it (0.93.5) -
    before, the two paths disagreed and the low-state bins were set back to the high state's centre."""
    from mppp.sfm.zcam import refit_focus_lines
    rec, proj, model, db = _bins(monkeypatch)
    model["cameras"]["ZR034"]["regular_pp_offset_px"] = [-5.0, 3.0]
    for key, cid in db.items():                       # the low-state bins' centres carry the offset
        if key.startswith("ZR") and key.endswith("_reg"):
            p = np.array(rec.cameras[cid].params, float)
            p[2] -= 5.0
            p[3] += 3.0
            rec.cameras[cid].params = p
    out = refit_focus_lines(rec, proj, model=model)
    zr = out["ZR034"]["lines"]
    assert zr["backlash"]["cx"]["a"] == 820.0 and zr["backlash"]["cy"]["a"] == 600.0
    assert zr["regular"]["cx"]["a"] == 815.0 and zr["regular"]["cy"]["a"] == 603.0 and zr["regular"]["cx"]["b"] == 0.0
    assert zr["regular"]["cy"]["b"] == pytest.approx(zr["backlash"]["cy"]["b"], abs=5e-4)   # the same slope, through the pin
    assert zr["regular"]["cy"]["rms"] < 0.2 and zr["regular"]["cx"]["rms"] < 0.2        # (the old pin: 3 and 5 px off)
    p = rec.cameras[db["ZR034_F00600_reg"]].params
    assert p[2] == pytest.approx(815.0) and p[3] == pytest.approx(603.0)
    assert rec.cameras[db["ZR034_F00500"]].params[2] == pytest.approx(820.0)
    assert "cx" not in out["ZL034"]["lines"]["regular"]                                  # ZL: the reference eye


def test_refit_lines_free_form_and_foreign_pp0(monkeypatch):
    from mppp.sfm.zcam import PIN_MAX_OFFSET_PX, refit_focus_lines
    rec, proj, model, _ = _bins(monkeypatch, form={"ZL": {k: "free" for k in ("fx", "fy", "k1", "k2")},
                                                   "ZR": {k: "free" for k in ("fx", "fy", "cx", "cy", "k1", "k2")}})
    out = refit_focus_lines(rec, proj, model=model)
    zr = out["ZR034"]["lines"]["backlash"]
    assert zr["fx"]["b"] != zr["fy"]["b"] and zr["fx"]["b"] == pytest.approx(0.050, abs=0.003) and zr["fy"]["b"] == pytest.approx(0.060, abs=0.003)
    assert zr["cx"]["b"] == pytest.approx(0.0005, abs=2e-4) and zr["k1"]["b"] == pytest.approx(3e-5, rel=0.05)
    # a model whose pp0 does not belong to these bins (a synthetic block against the shipped model): not pinned
    rec, proj, model, _ = _bins(monkeypatch, pp0=(820.0 + PIN_MAX_OFFSET_PX + 50, 600.0))
    out = refit_focus_lines(rec, proj, model=model)
    zr = out["ZR034"]["lines"]["backlash"]
    assert zr["cx"]["a"] == pytest.approx(820.0, abs=0.5) and zr["cx"]["b"] == 0.0 and "not pinned" in zr["cx"]["a_source"]


def test_bin_fallback_classifies_groups_without_image_fits():
    from mppp.sfm.backlash import BIN_FALLBACK_SD, classify_groups
    ims = []
    for sol, seq, foc, cam in ((1624, "ZCAM09705", 792.0, "ZR034_F00792"), (1618, "ZCAM09697", 1002.0, "ZR034_F01002")):
        ims += [{"name": f"ZR0_{sol}_{k}_{seq}.png", "instrument": cam, "camera_group": "ZR034", "eye": "R", "sol": sol,
                 "sequence": seq, "focus_count": foc, "label_f_px": 4650.0} for k in range(4)]
    proj = SimpleNamespace(images=ims, settings={})
    fits = {ims[4]["name"]: {"ratio_label": 1.0100, "sd_ratio": 0.0002, "observations": 2000}}   # one image of the second
    fb = {"ZR034_F00792": {"f_px": 4650.0 * 1.0005, "observations": 15000}}      # the first: its bin sits at the label
    rows = classify_groups(proj, fits, model=None, nav_scale=1.0, known=False, sequence_rule=False, measured_ratio=False,
                           bin_fallback=fb)
    by = {r["sequence"]: r for r in rows}
    assert by["ZCAM09705"]["fit_source"] == "bin camera" and by["ZCAM09705"]["sd"] == BIN_FALLBACK_SD
    assert by["ZCAM09705"]["state"] == "regular" and by["ZCAM09705"]["fitted"] == 0
    assert by["ZCAM09697"]["state"] == "backlash" and "fit_source" not in by["ZCAM09697"]
    rows = classify_groups(proj, fits, model=None, nav_scale=1.0, known=False, sequence_rule=False, measured_ratio=False)
    assert {r["sequence"]: r["state"] for r in rows}["ZCAM09705"] == "undecided"           # without the fallback
    # a bin shared with another sequence at the same focus count does not stand for one group
    ims.append(dict(ims[0], name="other.png", sequence="ZCAM09800"))
    rows = classify_groups(proj, fits, model=None, nav_scale=1.0, known=False, sequence_rule=False, measured_ratio=False,
                           bin_fallback=fb)
    assert {r["sequence"]: r["state"] for r in rows}["ZCAM09705"] == "undecided"


def test_append_found_survives_a_group_without_a_fit(tmp_path):
    from mppp.sfm.backlash import append_found
    f = tmp_path / "found.csv"
    n = append_found(f, [{"names": ["a.png"], "sol": 1, "sequence": "ZCAM0", "focus_count": 5.0, "ratio": None, "z": None,
                          "sequence_z": -4.0, "sequence_state": "regular", "own_state": "undecided"}], site="s")
    txt = f.read_text()
    assert n == 1 and "by its sequence" in txt and "z -4.0" in txt


def test_split_starts_the_regular_camera_at_its_own_principal_point(tmp_path):
    """0.93.2: each backlash state its own camera - the split regular camera starts at the backlash camera's
    principal point plus the model's regular-state offset (regular_pp_offset_px); a model without one: unchanged."""
    pytest.importorskip("scipy")
    from mppp.sfm.backlash import classify_groups, image_focal_fits, image_states, regular_pp_offsets, split_by_state
    proj, rec, model, truth = _zcam_block(tmp_path)
    reg = [n for n, s in image_states(classify_groups(proj, image_focal_fits(rec, proj), model)).items() if s == "regular"]
    base = np.array(rec.cameras[proj.settings["database"]["cameras"]["ZL034_F00600"]].params, float)
    assert regular_pp_offsets(model) == {}
    model2 = {"cameras": {"ZL034": dict(model["cameras"]["ZL034"], regular_pp_offset_px=[2.5, -1.5]),
                          "ZR034": {"regular_pp_offset_px": [0.0, 0.0]}}}
    assert regular_pp_offsets(model2) == {"ZL034": [2.5, -1.5]}
    new, rows = split_by_state(rec, proj, reg, hold_f_images=3, pp_offsets=regular_pp_offsets(model2))
    p = np.array(new.cameras[rows[0]["camera_id"]].params, float)
    assert p[2] - base[2] == pytest.approx(2.5) and p[3] - base[3] == pytest.approx(-1.5)
    assert "principal point +2.50, -1.50 px" in proj.cameras["ZL034_F00600_reg"]["source"]


def test_zcam_camera_at_regular_principal_point_from_its_lines():
    """0.93.2: a model whose regular lines carry cx, cy (pp0 + the offset) gives the regular state its own principal
    point; the backlash state keeps pp0."""
    import copy
    from mppp import calib
    zm = calib.load_zcam(34)
    if zm is None:
        pytest.skip("no Z34 focus model")
    m = copy.deepcopy(zm)
    gm = m["cameras"]["ZL034"]
    pp0 = gm["pp0_px"]
    gm["lines"]["regular"]["cx"] = {"a": pp0[0] + 3.0, "b": 0.0, "derived": "pp0 + the regular-state offset"}
    gm["lines"]["regular"]["cy"] = {"a": pp0[1] - 2.0, "b": 0.0, "derived": "pp0 + the regular-state offset"}
    _, pr, _ = calib.zcam_camera_at(m, "ZL034", 700.0, "regular")
    _, pb, _ = calib.zcam_camera_at(m, "ZL034", 700.0, "backlash")
    assert pr[2] == pytest.approx(pp0[0] + 3.0) and pr[3] == pytest.approx(pp0[1] - 2.0)
    assert pb[2] == pytest.approx(pp0[0]) and pb[3] == pytest.approx(pp0[1])


def test_mixed_focus_group_unlisted_images_are_backlash():
    """0.93.3 (Seitah Overlook ZCAM08114): one focus count, five pointings, 1/3/5 listed regular (f/label 0.994),
    2/4 not (1.000) - the absolute threshold (1.0047) called the group regular; the listed images now set the
    reference, so the unlisted ones are backlash.  Unlisted images that fit like the listed ones stay regular."""
    from mppp.sfm.backlash import image_states, mixed_group_state
    base = {"names": ["p1", "p2", "p3", "p4", "p5"], "listed_names": ["p1", "p3", "p5"], "all_listed": False,
            "nav_scale": 1.0, "backlash_ratio": 1.0095, "state": "regular", "decided": True}
    row = dict(base, ratio_listed=0.994, sd_listed=0.0004, ratio_fit=1.000, sd_fit=0.0005)
    assert mixed_group_state(row) == "backlash" and row["state_before_mixed_rule"] == "regular"
    assert image_states([row]) == {"p1": "regular", "p2": "backlash", "p3": "regular", "p4": "backlash",
                                   "p5": "regular"}
    same = dict(base, ratio_listed=0.994, sd_listed=0.0004, ratio_fit=0.9942, sd_fit=0.0005)
    assert mixed_group_state(same) == "regular"
    assert mixed_group_state(dict(same, state="backlash")) == "backlash"          # never made regular by the rule
    nofit = dict(base, ratio_listed=None, ratio_fit=None)
    assert mixed_group_state(nofit) == "backlash"
    for r in (dict(base, all_listed=True), dict(base, listed_names=[]), dict(base, state="single"),
              dict(base, implausible=True, state="backlash")):
        st = r["state"]
        assert mixed_group_state(r) is None and r["state"] == st


def test_split_never_overwrites_the_listed_regular_bin(tmp_path):
    """0.93.4: a <bin>_reg camera that already exists (the list's own regular bin) is kept; the split's new camera
    gets a unique name that still ends in _reg."""
    pytest.importorskip("scipy")
    from mppp.sfm.backlash import classify_groups, image_focal_fits, image_states, split_by_state
    proj, rec, model, truth = _zcam_block(tmp_path)
    reg = [n for n, s in image_states(classify_groups(proj, image_focal_fits(rec, proj), model)).items() if s == "regular"]
    db = proj.settings["database"]["cameras"]
    proj.cameras["ZL034_F00600_reg"] = {"group": "ZL034", "backlash_state": "regular", "n_images": 2, "source": "list"}
    db["ZL034_F00600_reg"] = 7
    new, rows = split_by_state(rec, proj, reg, hold_f_images=3)
    assert rows[0]["camera"] == "ZL034_F00600_2_reg" and rows[0]["camera_id"] == 8
    assert proj.cameras["ZL034_F00600_reg"]["source"] == "list" and db["ZL034_F00600_reg"] == 7
    assert db["ZL034_F00600_2_reg"] == 8 and proj.cameras["ZL034_F00600_2_reg"]["backlash_state"] == "regular"
    assert {r["instrument"] for r in proj.images if r["sequence"] == "zcam3"} == {"ZL034_F00600_2_reg"}


def test_single_state_zoom_and_one_state_rule():
    """0.93.5 (Christian): 110 mm has no state - "single" everywhere; one rule for a camera's state."""
    from mppp import calib
    assert calib.zcam_state("ZL110_F01234") == "single" and calib.zcam_state("ZR110_F00900_reg") == "single"
    assert calib.zcam_state("ZL034_F01254_2_reg") == "regular" and calib.zcam_state("ZL034_F01254_reg@chal") == "regular"
    assert calib.zcam_state("ZL034_F01254_reg", {"backlash_state": "backlash"}) == "backlash"
    assert calib.zcam_state("ZL034_F01254") == "backlash"
    assert calib.focus_state(True, "ZL110") == "single" and calib.focus_state(True, "ZL034") == "regular"
    assert calib.focus_state(False) == "backlash"
    m = {"cameras": {"ZL110": {"f0_px": 15000.0, "reference_focus": 600.0, "regular_ratio": 1.0095,
                               "pp0_px": [824.0, 600.0], "aspect": 1.0,
                               "lines": {"reference_focus": 600.0, "backlash": {"fx": {"a": 15000.0, "b": 0.0},
                                                                               "fy": {"a": 15001.0, "b": 0.0}},
                                         "regular": {"fx": {"a": 14000.0, "b": 0.0}, "fy": {"a": 14001.0, "b": 0.0}}}}}}
    _, p, _ = calib.zcam_camera_at(m, "ZL110", 600.0, "regular")
    assert p[0] == pytest.approx(15000.0) and p[1] == pytest.approx(15001.0)        # the one state's lines
    from mppp.calib.fit import regular_ratio
    assert regular_ratio([], "ZL110")["ratio"] is None
    from mppp.sfm.reconstruction import HAZCAM_MIN_REFINE_OBSERVATIONS
    assert HAZCAM_MIN_REFINE_OBSERVATIONS == 500                                     # Christian, 22 Oct


def test_check_zcam_regular_state_limits():
    """0.93.5: a fitted regular ratio outside 1.004..1.016 or a regular aspect apart from the backlash one fails; a
    110 mm model with a regular state fails."""
    from mppp.calib.fit import check_zcam
    ln = lambda fx, fy: {"fx": {"a": fx, "b": 0.0}, "fy": {"a": fy, "b": 0.0}}           # noqa: E731
    new = {"cameras": {"ZL034": {"f0_px": 4733.0, "aspect": 1.0008, "regular_ratio": 1.0238,
                                 "lines": {"backlash": ln(4731.7, 4735.3), "regular": ln(4659.9, 4587.0)}},
                       "ZL110": {"f0_px": 15000.0, "aspect": 1.0, "regular_ratio": 1.0095,
                                 "lines": {"backlash": ln(15000.0, 15000.0), "regular": ln(14860.0, 14860.0)}}}}
    c = {x["check"]: x for x in check_zcam(new, {"cameras": {}})["checks"]}
    assert not c["ZL034 regular ratio"]["ok"] and not c["ZL034 regular aspect"]["ok"]
    assert not c["ZL110 single state"]["ok"]
    new["cameras"]["ZL034"].update(regular_ratio=1.009, lines={"backlash": ln(4731.7, 4735.3),
                                                               "regular": ln(4731.7 / 1.009, 4735.3 / 1.009)})
    new["cameras"]["ZL110"] = {"f0_px": 15000.0, "aspect": 1.0, "lines": {"backlash": ln(15000.0, 15000.0)}}
    c = {x["check"]: x for x in check_zcam(new, {"cameras": {}})["checks"]}
    assert c["ZL034 regular ratio"]["ok"] and c["ZL034 regular aspect"]["ok"] and c["ZL110 single state"]["ok"]


def test_shipped_z34_regular_state_is_a_focus_state():
    """0.93.5: the shipped Z34 model's regular ratio is a focus state's (the 22 Oct joint wrote 1.024)."""
    from mppp import calib
    zm = calib.load_zcam(34)
    if zm is None:
        pytest.skip("no Z34 focus model")
    for g, gm in zm["cameras"].items():
        assert 1.004 <= float(gm["regular_ratio"]) <= 1.016, (g, gm["regular_ratio"])


def test_split_leaves_regular_cameras_and_strip_restores_the_base(tmp_path):
    """0.93.5: images already in a regular camera (the list's _reg bins) are not split again; strip_thermal_bins
    gives the base camera back its image count and state."""
    pytest.importorskip("scipy")
    from mppp.sfm.backlash import classify_groups, image_focal_fits, image_states, split_by_state
    from mppp.sfm.thermal import strip_thermal_bins
    proj, rec, model, truth = _zcam_block(tmp_path)
    states = image_states(classify_groups(proj, image_focal_fits(rec, proj), model))
    allreg = list(states)                                     # every image asked to be regular
    cam = proj.cameras["ZL034_F00600"]
    cam["backlash_state"] = "regular"                         # as a list _reg bin
    new, rows = split_by_state(rec, proj, allreg)
    assert rows == [] and cam["backlash_state"] == "regular"
    cam["backlash_state"] = "backlash"
    n0 = cam["n_images"]
    reg = [n for n, s in states.items() if s == "regular"]
    new, rows = split_by_state(rec, proj, reg, hold_f_images=0)
    assert rows and "fx" not in proj.cameras[rows[0]["camera"]]["fixed_params"]   # its f free (more than 0 images)
    assert cam["n_images"] == n0 - len(reg)
    assert {r["backlash_state"] for r in proj.images if r["name"] not in reg} == {"backlash"}
    strip_thermal_bins(proj)
    assert cam["n_images"] == n0 and cam["backlash_state"] == "backlash" and "pre_split" not in cam


def test_sequence_rule_skips_implausible_and_disabled_list(tmp_path):
    """0.93.5: a failed bin (implausible) keeps its own state under the sequence rule; a project built without a
    list keeps none when it is classified."""
    from mppp.sfm.backlash import apply_sequence_states, classify_groups, image_focal_fits
    rows = [{"sol": 1, "sequence": "ZCAM1", "group_key": f"g{i}", "state": "regular", "z": -9.0, "ratio_fit": 0.999,
             "sd_fit": 0.0005, "threshold_ratio": 1.0047} for i in range(3)]
    rows.append({"sol": 1, "sequence": "ZCAM1", "group_key": "bad", "state": "undecided", "z": 30.0,
                 "ratio_fit": 1.022, "sd_fit": 0.0005, "threshold_ratio": 1.0047, "implausible": True})
    apply_sequence_states(rows)
    assert [r["state"] for r in rows] == ["regular"] * 3 + ["undecided"]
    pytest.importorskip("scipy")
    proj, rec, model, truth = _zcam_block(tmp_path)
    lst = tmp_path / "list.csv"
    lst.write_text("zcam3\n", encoding="utf-8")                # not a real token: use the image names
    names = [r["name"] for r in proj.images if r["sequence"] == "zcam3"]
    lst.write_text("\n".join(n[:-4] for n in names) + "\n", encoding="utf-8")
    fits = image_focal_fits(rec, proj)
    proj.settings["zcam_low_backlash"] = {"file": str(lst)}
    g = [r for r in classify_groups(proj, fits, model) if r["sequence"] == "zcam3"][0]
    assert sorted(g["listed_names"]) == sorted(names)
    proj.settings["zcam_low_backlash"] = {"file": str(lst), "disabled": True}
    g = [r for r in classify_groups(proj, fits, model) if r["sequence"] == "zcam3"][0]
    assert g["listed_names"] == []
