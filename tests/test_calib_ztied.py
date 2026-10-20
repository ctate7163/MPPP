"""The Mastcam-Z joint (mppp.calib.ztied) on two synthetic Navcam + Mastcam-Z 34 mm blocks: the Navcam held at its
law, the Mastcam-Z focus lines, distortion and rig recovered from a perturbed start; the diagnostic pass, the files."""
import copy
import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
pytest.importorskip("pyceres")

ZW, ZH = 1648, 1200


def _look(C, fwd_az_deg, pitch_deg):
    a, p = np.radians(fwd_az_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(a) * np.cos(p), np.cos(a) * np.cos(p), -np.sin(p)])
    right = np.array([np.cos(a), -np.sin(a), 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd])


def _law(model, g, F, state="backlash"):
    gm = model["cameras"][g]
    ln = gm["lines"]["backlash"]
    ref = gm["reference_focus"]
    ratio = gm["regular_ratio"] if state == "regular" else 1.0
    fx = (ln["fx"]["a"] + ln["fx"]["b"] * (F - ref)) / ratio
    fy = (ln["fy"]["a"] + ln["fy"]["b"] * (F - ref)) / ratio
    cx, cy = gm["pp0_px"]
    if g.startswith("ZR"):
        cy = cy + gm["pp"]["cy_px_per_count"] * (F - ref)
    d = dict(zip(gm["distortion"]["names"], gm["distortion"]["params"]))
    return np.array([fx, fy, cx, cy, d.get("k1", 0), d.get("k2", 0), d.get("p1", 0), d.get("p2", 0), 0, 0, 0, 0], float)


def _truth_model():
    """The focus model in use, made the zcam-v2 truth of the synthetic blocks (p1, p2 small, a ZR cy slope)."""
    from mppp.sfm.project import zcam_focus_model_path
    m = json.loads(zcam_focus_model_path().read_text())
    for g in ("ZL034", "ZR034"):
        gm = m["cameras"][g]
        b = gm["lines"]["backlash"]
        a_fx, a_fy, s = b["fx"]["a"], b["fy"]["a"], 0.04
        gm["lines"]["backlash"]["fx"] = {"a": a_fx, "b": s}
        gm["lines"]["backlash"]["fy"] = {"a": a_fy, "b": s}
        gm["reference_focus"] = 600.0
        gm["regular_ratio"] = 1.009
        gm["distortion"] = {"names": ["k1", "k2", "p1", "p2"], "params": [-0.30, 0.40, 2e-4, -1e-4]}
        gm["pp"] = dict(gm.get("pp") or {}, cy_px_per_count=0.002 if g.startswith("ZR") else 0.0)
    from scipy.spatial.transform import Rotation
    m["rig"]["R_sensor_from_ref"] = Rotation.from_rotvec(np.radians([0.05, 4.0, -0.6])).as_matrix().tolist()
    m["rig"]["t_sensor_from_ref"] = [-0.2435, 0.0, 0.0]
    return m


def _block(tmp_path, name, shift, model, rng, sol):
    """One block: the synthetic Navcam stations (helpers._synthetic) plus Mastcam-Z 34 mm stereo frames at each
    station (three pointings x three focus counts), every image seeing dense terrain patches."""
    from helpers import _synthetic
    from mppp.calib.joint import Scape
    from scipy.spatial.transform import Rotation
    proj, truth, P, cams, rigT, noise, rng_ = _synthetic(tmp_path / name, noise_native_px=0.05, seed=int(sol))
    Rz = np.asarray(model["rig"]["R_sensor_from_ref"], float)
    tz = np.asarray(model["rig"]["t_sensor_from_ref"], float)
    zimgs, k = [], 0
    patches = []
    for st, C0 in (("S001D0000", np.array([0.0, 0.0, 1.9])), ("S001D0100", np.array([4.0, 1.0, 1.9]))):
        for az in (20.0, 60.0, 100.0):
            RL = _look(C0, az, 28.0)
            CL = C0 + np.array([0.0, 0.0, 0.15])
            tgt = CL + RL[2] * (1.9 / np.sin(np.radians(28.0)))
            patches.append(np.column_stack([tgt[0] + rng.uniform(-1.4, 1.4, 500), tgt[1] + rng.uniform(-1.4, 1.4, 500),
                                            rng.normal(0, 0.15, 500)]))
            for F in (300.0, 600.0, 950.0):
                k += 1
                RR = Rz @ RL
                CR = CL - RR.T @ tz
                sclk = f"{7000 + k:010d}_000"
                for eye, R, C in (("L", RL, CL), ("R", RR, CR)):
                    nm = f"Z{eye}0_{sol:04d}_{k:04d}.png"
                    zimgs.append({"name": nm, "stem": nm[:-4], "instrument": f"Z{eye}", "eye": eye, "sclk_key": sclk,
                                  "sol": sol, "site": 1, "drive": int(st[-4:]), "station": st, "sequence": "ZCAM00000",
                                  "downsample_scale": 1.0, "native_size": [ZW, ZH], "prior_C": C.tolist(),
                                  "prior_R_w2c": R.tolist(), "has_mask": False, "camera_group": f"Z{eye}034",
                                  "focus_count": F})
                    truth[nm] = (R, C)
    P = np.vstack([P] + patches)
    for r in proj.images:
        r["sol"] = sol
        r["camera_group"] = r["instrument"]
    proj.images += zimgs
    # cameras: NL 1, NR 2 (the synthetic Navcam), one Mastcam-Z camera per (eye, focus)
    rec = pycolmap.Reconstruction()
    for cid, key in ((1, "NL"), (2, "NR")):
        rec.add_camera(pycolmap.Camera(camera_id=cid, model="FULL_OPENCV", width=5120, height=3840,
                                       params=np.array(cams[key]["params"], float)))
    db = {"NL": 1, "NR": 2}
    zcid, nid = {}, 3
    for r in zimgs:
        key = f"{r['camera_group']}_f{int(r['focus_count'])}"
        if key not in zcid:
            p = _law(model, r["camera_group"], r["focus_count"])
            rec.add_camera(pycolmap.Camera(camera_id=nid, model="FULL_OPENCV", width=ZW, height=ZH, params=p))
            proj.cameras[key] = {"model": "FULL_OPENCV", "params": p.tolist(), "width": ZW, "height": ZH,
                                 "group": r["camera_group"], "focus_count_median": float(r["focus_count"]),
                                 "backlash_state": "backlash", "fixed_params": ["cx", "cy"]}
            zcid[key] = nid
            db[key] = nid
            nid += 1
        r["_cid"] = zcid[key]
    proj.settings["database"] = {"cameras": db}
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=int(c))       # noqa: E731
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(S(1))
    rig.add_sensor(S(2), pycolmap.Rigid3d(pycolmap.Rotation3d(rigT[0]), rigT[1]))
    rec.add_rig(rig)
    for cid in range(3, nid):
        r_ = pycolmap.Rig(rig_id=cid)
        r_.add_ref_sensor(S(cid))
        rec.add_rig(r_)
    fid = 0
    nav = [r for r in proj.images if r["instrument"][0] == "N"]
    for i, r in enumerate(proj.images, start=1):
        r["image_id"] = i
    by_sclk = {}
    for r in nav:
        by_sclk.setdefault(r["sclk_key"], []).append(r)
    obs = {}
    true_cam = {1: rec.cameras[1], 2: rec.cameras[2]}
    for cid in range(3, nid):
        true_cam[cid] = rec.cameras[cid]

    def add_image(r, cid, fid_):
        R, C = truth[r["name"]]
        cam = true_cam[cid]
        Xc = (P - C) @ R.T
        ok = Xc[:, 2] > 0.5
        uv = np.full((len(P), 2), np.nan)
        uv[ok] = cam.img_from_cam(Xc[ok])
        inside = ok & (uv[:, 0] > 20) & (uv[:, 0] < cam.width - 20) & (uv[:, 1] > 20) & (uv[:, 1] < cam.height - 20)
        s = r["downsample_scale"]
        js = np.where(inside)[0]
        kp = uv[js] * s + rng.normal(0, 0.05, (len(js), 2))
        im = pycolmap.Image(name=r["name"], keypoints=(kp / s).reshape(-1, 2), camera_id=cid, image_id=r["image_id"])
        im.frame_id = fid_
        rec.add_image(im)
        obs[r["image_id"]] = list(js)
        r["camera_id"], r["frame_id"] = cid, fid_

    for key, rs in sorted(by_sclk.items()):
        fid += 1
        fr = pycolmap.Frame(frame_id=fid, rig_id=1)
        L = [r for r in rs if r["eye"] == "L"][0]
        R0, C0 = truth[L["name"]]
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R0), -R0 @ C0)
        for r in rs:
            fr.add_data_id(pycolmap.data_t(sensor_id=S(1 if r["eye"] == "L" else 2), id=r["image_id"]))
        rec.add_frame(fr)
        for r in rs:
            add_image(r, 1 if r["eye"] == "L" else 2, fid)
    for r in zimgs:
        fid += 1
        fr = pycolmap.Frame(frame_id=fid, rig_id=r["_cid"])
        R0, C0 = truth[r["name"]]
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R0), -R0 @ C0)
        fr.add_data_id(pycolmap.data_t(sensor_id=S(r["_cid"]), id=r["image_id"]))
        rec.add_frame(fr)
        add_image(r, r["_cid"], fid)
    for f in list(rec.frames):
        rec.register_frame(f)
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
        pid = rec.add_point3D(P[j], tr)
        for iid, k2 in els:
            rec.images[iid].set_point3D_for_point2D(k2, pid)
    temps = {r["name"]: {"T": -20.0, "T_NL": -20.0, "T_NR": -20.0} for r in nav}
    return Scape(name, tmp_path / name, proj, rec, temps), cams, rigT


def _navcam_law(cams, rigT):
    from mppp.calib import tied as T
    st = T.start_values()
    for eye in ("NL", "NR"):
        st["G"][eye] = np.asarray(cams[eye]["params"], float)[:10].copy()
        st["S"][eye] = np.zeros(5)
        st["E"][eye] = np.zeros(4)
        st["cameras"][eye] = dict(st["cameras"][eye], model="FULL_OPENCV", params=list(cams[eye]["params"]))
    st["D"], st["H"] = np.zeros(3), np.zeros(3)
    st["Q0"] = T.q_xyz(rigT[0])
    st["R"], st["t"] = rigT
    return st


def test_zcam_joint_recovers_the_focus_law_and_the_rig(tmp_path):
    from mppp.calib import ztied as Z
    rng = np.random.default_rng(3)
    truth = _truth_model()
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900)
    st = _navcam_law(cams, rigT)
    start = copy.deepcopy(truth)                     # the start: lines, distortion and rig off
    from scipy.spatial.transform import Rotation
    for g in ("ZL034", "ZR034"):
        ln = start["cameras"][g]["lines"]["backlash"]
        ln["fx"]["a"] += 6.0
        ln["fy"]["a"] -= 4.0
        ln["fx"]["b"] = ln["fy"]["b"] = 0.02
        start["cameras"][g]["distortion"] = {"names": ["k1", "k2"], "params": [-0.28, 0.35]}
        start["cameras"][g]["pp"]["cy_px_per_count"] = 0.0
    start["rig"]["R_sensor_from_ref"] = (Rotation.from_rotvec(np.radians([0.0, 0.02, 0.0])).as_matrix()
                                         @ np.asarray(truth["rig"]["R_sensor_from_ref"])).tolist()
    # the blocks' bins start where the blocks would have them: near the truth (as solved), the law is the start
    res = Z.fit_zjoint({}, model=start, scapes=[a, b], st=st, nav_points=None, z_points=None, iterations=60, passes=2,
                       verbose=False)
    for g in ("ZL034", "ZR034"):
        tm = res["eyes"][g]["terms"]
        ln = truth["cameras"][g]["lines"]["backlash"]
        assert tm["a_fx_backlash"] == pytest.approx(ln["fx"]["a"], abs=0.5)
        assert tm["a_fy_backlash"] == pytest.approx(ln["fy"]["a"], abs=0.5)
        assert tm["b"] == pytest.approx(0.04, abs=2e-3)
        assert tm["k1"] == pytest.approx(-0.30, abs=0.01)
        assert "a_fx_regular" in res["eyes"][g]["held"]          # no regular-state bin
    assert res["eyes"]["ZR034"]["terms"]["b_cy"] == pytest.approx(0.002, abs=5e-4)
    assert res["rig"]["change_mdeg"] == pytest.approx(20.0, abs=4.0)        # back to the truth from 20 mdeg off
    assert res["zcam"]["zcam_median_px"] < 0.2 and res["zcam"]["navcam_zcam_points"] > 100
    assert res["covariance_error"] is None and res["eyes"]["ZL034"]["sd"]["b"] > 0
    d = res["diagnostics"]
    assert {r["block"] for r in d["blocks"]} == {"blk_a", "blk_b"}
    assert all(abs(r["yaw_mdeg"]) < 5 for r in d["blocks"])
    assert d["blocks"][0]["lines"]["ZL034"]["backlash"]["fx"]["bins"] == 3
    out = Z.write_zjoint(res, tmp_path / "cand", model=start)
    new = json.loads((tmp_path / "cand" / "M2020_ZCAM034_focus_model.json").read_text())
    from mppp.calib import forms
    assert not forms.validate_zcam(new) and new["form"] == forms.ZCAM_FORM_ID
    assert new["cameras"]["ZR034"]["lines"]["backlash"]["cy"]["form"] == "pinned_slope"
    assert out["check"]["checks"] and (tmp_path / "cand" / "report.md").is_file()
    assert Z.block_line_rows(res)
