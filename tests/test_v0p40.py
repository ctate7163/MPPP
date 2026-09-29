"""v0p40: site discovery for notebooks 04 and 05, notebook 03 defaults."""
import json
from pathlib import Path

import numpy as np
import pytest

from mppp.sfm.sites import SITES, check_sites, discover_scapes, scan_scapes, site_label


def _site(root, name, sols, reconstruction=True, error_input=True, model=True, n=None):
    col = root / name / "colmap"
    col.mkdir(parents=True)
    ims = [{"name": f"i{k}.png", "sol": s, "instrument": "NL" if k % 2 == 0 else "NR", "station": f"S{k // 4}"}
           for k, s in enumerate(sols)]
    settings = {"reconstruction": {"path": "sparse\\cahv_ba"}} if reconstruction else {}
    (col / "project.json").write_text(json.dumps({"images": ims, "settings": settings}))
    if model:
        (col / "sparse" / "cahv_ba").mkdir(parents=True)
        (col / "sparse" / "cahv_ba" / "images.bin").write_bytes(b"x")
    if error_input:
        (col / "error_input").mkdir()
        (col / "error_input" / "summary.json").write_text("{}")
    (col / "health").mkdir()
    (col / "health" / "health.json").write_text(json.dumps({"verdict": "pass", "mppp_version": "0.35.2"}))


def test_labels_and_site_checks():
    assert site_label("threeforks_south") == "Three Forks South" and site_label("van_zyl") == "Van Zyl"
    assert site_label("rockytop") == "Rockytop"
    probs = check_sites(SITES)
    assert probs == []
    assert SITES["hippo_pools"] == (1947, 1955) and SITES["origny"] == (1781, 1813) and SITES["origny_large"] == (1765, 1813) and SITES["olifants"] == (1880, 1889)


def test_discover_scapes(tmp_path):
    _site(tmp_path, "rockytop_colmap", [470] * 12)
    _site(tmp_path, "van_zyl_colmap", [60] * 12)
    _site(tmp_path, "olifants_colmap", [1780] * 12)                 # the renamed site's block: left out
    _site(tmp_path, "pearce_canyon_colmap", [1190] * 12, reconstruction=False)   # run in progress
    _site(tmp_path, "sid_colmap", [365] * 12, error_input=False)
    _site(tmp_path, "rockytop_colmap_nav_zcam34", [470] * 12)
    _site(tmp_path, "tiny_colmap", [5] * 4)
    (tmp_path / "camera_analysis").mkdir()
    d = discover_scapes(tmp_path, verbose=False)
    assert list(d) == ["Van Zyl", "Sid", "Rockytop", "Rockytop N+Z34"]      # sol order
    assert d["Rockytop"] == tmp_path / "rockytop_colmap"
    rows = {r["label"]: r for r in scan_scapes(tmp_path)}
    assert "no image in the site's sols 1880-1889" in rows["Olifants"]["reason"] and "origny" in rows["Olifants"]["reason"]
    assert "in progress" in rows["Pearce Canyon"]["reason"] and "< 10" in rows["Tiny"]["reason"]
    e = discover_scapes(tmp_path, require_error_input=True, include_zcam=False, exclude=["van_zyl"], verbose=False)
    assert list(e) == ["Rockytop"]


def test_notebooks_v0p40():
    nbformat = pytest.importorskip("nbformat")
    root = Path(__file__).resolve().parents[1] / "notebooks"
    nb3 = nbformat.read(str(root / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb3.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)
    assert ns["SITES"]["hippo_pools"] == (1947, 1955) and ns["SITES"]["threeforks"] == (652, 692)
    assert ns["SITE"] == "rockytop" and ns["LOCALIZE_MIN_IMAGES"] == 3 and ns["ATTITUDE_PRIOR_DEG"] == 2.0
    assert ns["NAVCAM_RIG_REFINE"] == "refine" and ns["SCHEDULE"][-1] == (8.0, 2.0, 2.0) and len(ns["SCHEDULE"]) == 3
    full3 = "\n".join(c.source for c in nb3.cells)
    assert "refine_rig=RIG_REFINE" in full3 and '"refine": "rotation"' in full3 and "check_sites(SITES)" in full3
    for name, var in (("04_camera_models", "SCAPES"), ("05_error_analysis", "ALIGNMENTS")):
        nb = nbformat.read(str(root / f"{name}.ipynb"), as_version=4)
        p = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
        assert f"{var} = None" in p
        assert "discover_scapes(SCAPES_ROOT" in "\n".join(c.source for c in nb.cells)


# ------------------------------------------------------------------ v0p40 rig epochs (Sid: sols 91-101 and 360-371)
def test_sol_epochs():
    from mppp.sfm.navcal import sol_epochs
    assert sol_epochs([91, 95, 101, 360, 371, 365]) == [(91, 101), (360, 371)]
    assert sol_epochs([652, 660, 674]) == [(652, 674)]
    assert sol_epochs([]) == []
    assert sol_epochs([1, 20, 45], gap=30) == [(1, 45)]


def test_split_rig_by_epoch_gives_each_epoch_its_rig(tmp_path):
    import sys
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_v0p31 import _synthetic_block
    from mppp.sfm.navcal import split_rig_by_epoch
    from mppp.sfm.reconstruction import bundle_adjust
    proj, rec, noise = _synthetic_block(tmp_path)
    for r in proj.images:
        r["sol"] = 95 if r["station"].endswith("0000") else 365
    new, proj2, rows = split_rig_by_epoch(rec, proj, {r["name"]: {"T": r["camera_temperature_degC"]} for r in proj.images})
    assert [(r["sol_min"], r["sol_max"]) for r in rows] == [(95, 95), (365, 365)]
    assert rows[0]["T_median_degC"] == -32.0 and rows[1]["T_median_degC"] == -14.0
    assert sum(r["frames"] for r in rows) == len(rec.frames)
    assert rows[0]["rig_id"] != rows[1]["rig_id"] and len(new.rigs) > len(rec.rigs)
    assert set(new.cameras) == set(rec.cameras)                       # the cameras stay shared
    assert new.num_reg_images() == rec.num_reg_images() and len(new.points3D) == len(rec.points3D)
    for r in rows:                                                     # each epoch's frames use its rig
        fr = [f for f in new.frames.values() if f.rig_id == r["rig_id"]]
        assert len(fr) == r["frames"]
    a, b = (new.rigs[r["rig_id"]] for r in rows)
    sa = next(iter(a.non_ref_sensors))
    assert np.allclose(a.sensor_from_rig(sa).matrix(), b.sensor_from_rig(sa).matrix())
    ba = bundle_adjust(new, proj2, sigma_px=noise, loss_scale=10.0, max_iterations=10, refine_rig="rotation")
    assert np.isfinite(ba["final_cost"])


def test_expand_epochs_places_each_epoch_at_its_sol():
    from mppp.sfm.navcal import expand_epochs
    base = {"scape": "Sid", "network": {"sol_median": 360.0, "T_median_degC": -15.7},
            "rotation_pp": {"rigs": [{"pitch_mdeg": 0.9}]}}
    ep = [{"epoch": 0, "sol_min": 91, "sol_max": 101, "sol_median": 98.0, "T_median_degC": -22.2, "frames": 28,
           "pitch_mdeg": -3.8},
          {"epoch": 1, "sol_min": 360, "sol_max": 371, "sol_median": 361.0, "T_median_degC": -8.6, "frames": 21,
           "pitch_mdeg": 1.2}]
    one = {"scape": "Van Zyl", "network": {"sol_median": 60.0, "T_median_degC": -20.0},
           "rotation_pp": {"rigs": [{"pitch_mdeg": -5.0}]}, "epochs": {"skipped": "one sol epoch"}}
    out = expand_epochs([dict(base, epochs={"rigs": ep}), one])
    assert [s["scape"] for s in out] == ["Sid (sols 91-101)", "Sid (sols 360-371)", "Van Zyl"]
    assert out[0]["network"]["sol_median"] == 98.0 and out[1]["network"]["T_median_degC"] == -8.6
    assert out[0]["rotation_pp"]["rigs"][0]["pitch_mdeg"] == -3.8 and out[0]["parent"] == "Sid"
    assert out[2] is one
    few = [dict(r, frames=3) if r["epoch"] == 0 else r for r in ep]      # too few frames in one epoch: not split
    assert [s["scape"] for s in expand_epochs([dict(base, epochs={"rigs": few})])] == ["Sid"]


# ------------------------------------------------------------------ v0p40 ZCAM_TANGENTIAL
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
    from test_v0p31 import _synthetic_block
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


def test_notebook03_zcam_tangential():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(Path(__file__).parents[1] / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)
    assert ns["TANGENTIAL"] == "refine" and ns["ZCAM_TANGENTIAL"] == "zero"
    derived = next(c.source for c in nb.cells if c.source.startswith("# derived settings"))
    block = derived.split("_ZT = ")[1].split("if GPU_PY")[0]
    for T, Z, zt, rt in (("refine", "zero", ("p1", "p2", "b1", "b2"), {"N": True, "Z": False}),
                         ("refine", None, None, True), ("zero", "refine", ("b1", "b2"), {"N": False, "Z": True}),
                         ("xml", "xml", None, False)):
        g = {"TANGENTIAL": T, "ZCAM_TANGENTIAL": Z}
        exec("_ZT = " + block, g)
        assert g["ZCAM_ZERO_TERMS"] == zt and g["REFINE_TANGENTIAL"] == rt, (T, Z)
    full = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "zcam_zero_terms=ZCAM_ZERO_TERMS" in full and "refine_tangential=REFINE_TANGENTIAL" in full


# ------------------------------------------------------------------ v0p40 Mastcam-Z focus backlash states
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


def test_write_joint_cameras_at_reference_temperature(tmp_path):
    from mppp.sfm.navcal import NAVCAM_T_REF_DEGC, write_joint_cameras
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    p = [2956.0, 2956.0, 2591.0, 1944.0, 0.3, -0.02, 0.0, 0.0, 0.002, 0.59, 0.0, 0.0]
    cam = {"model": "FULL_OPENCV", "width": 5120, "height": 3840, "params": dict(zip(names, p)),
           "free_params": list(names[:10]), "sd": {n: 0.1 for n in names[:10]}, "covariance": np.eye(10).tolist()}
    joint = {"lens": "rational", "NL": cam, "NR": cam, "T0_degC": -18.0, "ppm_per_degC": 40.0,
             "final": {"observations": 1000}, "blocks": {"A": 10}, "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0],
             "rig_thermal": {"yaw_mdeg_per_degC": -1.0, "pitch_mdeg_per_degC": 0.0}}
    assert NAVCAM_T_REF_DEGC == -20.0
    w = write_joint_cameras(joint, tmp_path)
    d = json.loads(w["NL"].read_text())
    assert d["thermal"]["T0_degC"] == -20.0 and abs(d["params"][0] - 2956.0 * (1 - 80e-6)) < 1e-6
    assert d["params"][2] == 2591.0 and joint["T0_degC"] == -18.0            # the input is not changed
    r = json.loads(w["rig"].read_text())
    assert r["thermal"]["T0_degC"] == -20.0
    from scipy.spatial.transform import Rotation
    yaw = np.degrees(Rotation.from_matrix(np.array(r["R_sensor_from_ref"])).as_rotvec()[1]) * 1e3
    assert abs(yaw - 2.0) < 1e-6                                             # -1 mdeg/degC x (-2 degC)


def test_label_temperature_fallback_for_uninterpolated_models():
    from mppp.image import MPPPImage
    from types import SimpleNamespace
    im = MPPPImage.__new__(MPPPImage)
    im.__dict__["camera_model_label"] = SimpleNamespace(meta={"interpolation": "NONE"})
    im.fn = SimpleNamespace(stem="NLF_0054_0671740217_053RAD_N0032046NCAM00745_0A0195J01")
    im.label = {"INSTRUMENT_STATE_PARMS": {"INSTRUMENT_TEMPERATURE_NAME": ["NAVCAM_LEFT_1", "NAVCAM_LEFT_CAL",
                                                                           "NAVCAM_RIGHT_CAL"],
                                           "INSTRUMENT_TEMPERATURE": [-20.24, -20.253, -19.2028]}}
    assert MPPPImage.camera_temperature_degC.fget(im) == -20.253
    im.fn = SimpleNamespace(stem="NRF_0054_0671740217_053RAD_N0032046NCAM00745_0A0195J01")
    assert MPPPImage.camera_temperature_degC.fget(im) == -19.2028
    im.__dict__["camera_model_label"] = SimpleNamespace(meta={"interpolation": "TEMPERATURE", "interpolation_value": -7.5})
    assert MPPPImage.camera_temperature_degC.fget(im) == -7.5
