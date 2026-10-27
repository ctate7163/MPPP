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


def _block(tmp_path, name, shift, model, rng, sol, reg=None, fb=None):
    """One block: the synthetic Navcam stations (helpers._synthetic) plus Mastcam-Z 34 mm stereo frames at each
    station (three pointings x three focus counts), every image seeing dense terrain patches.  ``reg``
    {"eye": "L", "F": 950.0, "offset": (dcx, dcy)}: that eye's images at that focus are in the regular state (f /
    regular ratio, the principal point moved by the offset; the other eye in backlash)."""
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
                    is_reg = bool(reg) and eye == reg["eye"] and F == reg["F"]
                    is_fb = bool(fb) and eye == fb["eye"] and F == fb["F"]       # 0.94.6: focus from below
                    zimgs.append({"_reg": is_reg, "_fb": is_fb, "name": nm, "stem": nm[:-4], "instrument": f"Z{eye}", "eye": eye, "sclk_key": sclk,
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
        key = (f"{r['camera_group']}_f{int(r['focus_count'])}" + ("_fb" if r["_fb"] else "")
               + ("_reg" if r["_reg"] else ""))
        if key not in zcid:
            p = _law(model, r["camera_group"], r["focus_count"], "regular" if r["_reg"] else "backlash")
            if r["_fb"]:                            # 0.94.6: the focus-from-below offset (px on fx, fy x aspect)
                asp = p[1] / p[0]
                p[0] += fb["offset_px"]
                p[1] += fb["offset_px"] * asp
            if r["_reg"]:
                p[2] += reg["offset"][0]
                p[3] += reg["offset"][1]
                if reg.get("ratio"):                # 0.94.6: this block's own high / low ratio
                    p[:2] *= float(model["cameras"][r["camera_group"]]["regular_ratio"]) / float(reg["ratio"])
            rec.add_camera(pycolmap.Camera(camera_id=nid, model="FULL_OPENCV", width=ZW, height=ZH, params=p))
            proj.cameras[key] = {"model": "FULL_OPENCV", "params": p.tolist(), "width": ZW, "height": ZH,
                                 "group": r["camera_group"], "focus_count_median": float(r["focus_count"]),
                                 "backlash_state": "regular" if r["_reg"] else "backlash", "fixed_params": ["cx", "cy"]}
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


def test_zcam_joint_recovers_the_focus_law_and_the_rig(tmp_path, monkeypatch):
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})       # the truth has k2 = 0.40: no fixed terms (0.93.1) here
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
    # 21 Oct: the distortion as curves and the likelihood-ratio tests (truth: the same k1, k2 = -0.30, 0.40 in both eyes)
    dist = res["distortion"]
    assert len(dist["eyes"]["ZL034"]["displacement_px"]) == 4 and dist["eyes"]["ZL034"]["r3_r5_correlation"] > 0.95
    assert abs(dist["difference_px"][-1]) < 0.5                       # the eyes' curves agree at the corner
    tests = {t["test"]: t for t in dist["tests"]}
    assert tests["k2 = 0 (both eyes)"]["significant"]                 # k2 = 0.40 is real
    assert not tests["ZR034 k1, k2 = ZL034 k1, k2"]["significant"]    # the eyes do not differ
    assert max(abs(z) for z in dist["difference_z"]) < 3               # nor do their curves
    assert any("radial displacement" in x for x in Z.describe(res))
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


def test_joint_gate_residuals_and_block_models():
    """20 Oct, first real joint: NaN Mastcam-Z residuals and blocks aligned with older models were promoted."""
    from mppp.calib import ztied as Z
    base = {"ok": True, "checks": [{"check": "form", "ok": True, "note": ""}]}
    good = {"zcam": {"zcam_median_px": 0.3, "zcam_nonfinite_share": 0.0, "zcam_observations": 10},
            "models_fingerprint": "abc", "block_runs": {"a": {"camera_models": {"fingerprint": "abc"}}}}
    assert Z.joint_checks(good, base)["ok"]
    nan = dict(good, zcam={"zcam_median_px": None, "zcam_nonfinite_share": 0.3, "zcam_observations": 10})
    assert not Z.joint_checks(nan, base)["ok"]
    old = dict(good, block_runs={"a": {"camera_models": {"fingerprint": "abc"}},
                                 "b": {"camera_models": {"fingerprint": "old"}}})
    c = Z.joint_checks(old, base)
    assert not c["ok"] and "b aligned with other camera models" in c["checks"][-1]["note"]


def test_nz_joint_agrees_with_the_navcam_law_and_sees_a_wrong_one(tmp_path):
    """21 Oct (Christian): the combined Navcam + Mastcam-Z joint - both laws free - compared with the Navcam-only law.
    On blocks made with the law it agrees; with the law's NL fx 3 px off it pulls fx back and says so."""
    from mppp.calib import nzjoint as NZ
    rng = np.random.default_rng(3)
    truth = _truth_model()
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900)
    st = _navcam_law(cams, rigT)
    res = NZ.fit_nzjoint({}, model=truth, scapes=[a, b], st=copy.deepcopy(st), nav_points=None, z_points=None,
                         iterations=40, passes=1, verbose=False)
    c = res["comparison"]
    assert c["agree"], NZ.describe(res)
    assert abs(c["navcam"]["NL"]["fx"]["change"]) < 0.5
    assert set(c["residuals"]["both_free"]) == {"navcam", "zcam"}
    bad = copy.deepcopy(st)
    bad["G"]["NL"][0] += 3.0
    res2 = NZ.fit_nzjoint({}, model=truth, scapes=[a, b], st=bad, nav_points=None, z_points=None, iterations=40,
                          passes=1, verbose=False)
    assert res2["comparison"]["navcam"]["NL"]["fx"]["change"] == pytest.approx(-3.0, abs=1.0)
    assert not res2["comparison"]["agree"] and "NL fx" in res2["comparison"]["beyond_limits"][0]
    out = NZ.write_nzjoint(res2, tmp_path / "nz")
    assert (tmp_path / "nz" / "report.md").is_file() and "Navcam NL" in (tmp_path / "nz" / "report.md").read_text()


def test_zcam_joint_gives_the_regular_state_its_own_principal_point(tmp_path, monkeypatch):
    """0.93.2 (Christian): each backlash state its own camera with the pose shared - the regular ZL state's principal
    point (truth +6, -4 px from the backlash state's) is fitted from the exposures with ZR in backlash, and the rig
    stays at the truth; with the offset held at 0 (the old form) the rig takes it up."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    from scipy.spatial.transform import Rotation
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    monkeypatch.setattr(Z, "LOW_LIST_CHECK", False)          # synthetic low-state frames: not on the shipped list
    rng = np.random.default_rng(5)
    truth = _truth_model()
    reg = {"eye": "L", "F": 950.0, "offset": (6.0, -4.0)}
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, reg=reg)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, reg=reg)
    st = _navcam_law(cams, rigT)
    Rt = np.asarray(truth["rig"]["R_sensor_from_ref"])

    def run(model):
        res = Z.fit_zjoint({}, model=copy.deepcopy(model), scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st,
                           nav_points=None, z_points=None, iterations=60, passes=2, verbose=False, diagnostics=False,
                           k_tests=False)
        R = np.asarray(res["rig"]["R_sensor_from_ref"])
        return res, float(np.degrees(np.linalg.norm(Rotation.from_matrix(R @ Rt.T).as_rotvec())) * 1e3)

    res, rig_err = run(truth)
    e = res["eyes"]["ZL034"]
    assert e["mixed_exposures"] == 12 and res["eyes"]["ZR034"]["mixed_exposures"] == 0
    assert "d_cx_regular" not in e["held"] and "d_cx_regular" in res["eyes"]["ZR034"]["held"]
    assert e["terms"]["d_cx_regular"] == pytest.approx(6.0, abs=0.3)
    assert e["terms"]["d_cy_regular"] == pytest.approx(-4.0, abs=0.3)
    assert rig_err < 3.0                                                     # mdeg: the rig not moved by the state
    cand = Z.candidate_model(res, truth)
    gm = cand["cameras"]["ZL034"]
    assert gm["regular_pp_offset_px"] == pytest.approx([6.0, -4.0], abs=0.3) and not gm["regular_pp_offset"]["held"]
    # 0.93.5: the regular state scales f only - its aspect is the backlash one, its ratio the truth's
    assert "a_fy_regular" in e["held"]
    t = e["terms"]
    assert t["a_fy_regular"] / t["a_fx_regular"] == pytest.approx(t["a_fy_backlash"] / t["a_fx_backlash"], abs=2e-4)
    assert gm["regular_ratio"] == pytest.approx(float(truth["cameras"]["ZL034"]["regular_ratio"]), abs=1e-3)
    assert "regular_ratio_note" not in gm
    from mppp import calib
    _, pr, _ = calib.zcam_camera_at(cand, "ZL034", 950.0, "regular")
    _, pb, _ = calib.zcam_camera_at(cand, "ZL034", 950.0, "backlash")
    assert pr[2] - pb[2] == pytest.approx(6.0, abs=0.3) and pr[3] - pb[3] == pytest.approx(-4.0, abs=0.3)
    assert any("low-state principal point" in x for x in Z.describe(res))
    # 0.93.7: with the shipped-list check the synthetic low-state cameras (not on the list) are left out
    monkeypatch.setattr(Z, "LOW_LIST_CHECK", True)
    rl, _ = run(truth)
    assert rl["eyes"]["ZL034"]["regular_bins"] == 0 and rl["low_cameras_not_on_list"]
    assert any("not on the shipped" in x for x in Z.describe(rl))
    monkeypatch.setattr(Z, "LOW_LIST_CHECK", False)
    # the old form: the regular state shares the backlash principal point - the rig takes the offset up
    monkeypatch.setattr(Z, "MIN_MIXED_EXPOSURES", 10 ** 6)
    res0, rig_err0 = run(truth)
    assert "d_cx_regular" in res0["eyes"]["ZL034"]["held"] and rig_err0 > 3 * max(rig_err, 1.0)


def test_zcam_joint_high_low_ratio_by_block(tmp_path, monkeypatch):
    """0.94.6 (Christian, 24 Oct): the high / low ratio can vary by block - two blocks with ZL ratios 1.007 and
    1.011 (truth): with ratio_by "block" each block's ratio is recovered and the eye's is their mean; "global" fits
    one ratio between them."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    monkeypatch.setattr(Z, "LOW_LIST_CHECK", False)
    rng = np.random.default_rng(7)
    truth = _truth_model()
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, reg={"eye": "L", "F": 950.0, "offset": (0, 0),
                                                                         "ratio": 1.007})
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, reg={"eye": "L", "F": 950.0, "offset": (0, 0),
                                                                   "ratio": 1.011})
    st = _navcam_law(cams, rigT)

    def run(by):
        return Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st,
                            nav_points=None, z_points=None, iterations=60, passes=2, verbose=False,
                            diagnostics=False, k_tests=False, ratio_by=by)

    res = run("block")
    g = {r["group"]: r for r in res["regular_ratio_groups"]}
    assert set(g) == {"blk_a", "blk_b"} and res["regular_ratio_by"] == "block"
    assert res["covariance_method"].startswith("exact ties"), res.get("covariance_exact")    # 0.94.6
    assert g["blk_a"]["ratio"] == pytest.approx(1.007, abs=3e-4) and g["blk_b"]["ratio"] == pytest.approx(1.011, abs=3e-4)
    assert g["blk_a"]["sd_ratio"] and g["blk_a"]["sd_ratio"] < 1e-3 and g["blk_a"]["sols"] == [500]
    t = res["eyes"]["ZL034"]["terms"]
    assert t["a_fx_backlash"] / t["a_fx_regular"] == pytest.approx(1.009, abs=4e-4)        # the groups' mean
    assert any("high / low ratio of blk_a" in x for x in Z.describe(res))
    cand = Z.candidate_model(res, truth)
    assert set(cand["cameras"]["ZL034"]["regular_ratio_groups"]["groups"]) == {"blk_a", "blk_b"}
    assert "regular_ratio_groups" not in cand["cameras"]["ZR034"]        # ZR: no low-state bins
    r0 = run("global")
    assert r0["regular_ratio_groups"] == []
    t0 = r0["eyes"]["ZL034"]["terms"]
    assert 1.007 < t0["a_fx_backlash"] / t0["a_fx_regular"] < 1.011
    assert Z.ratio_group({"scape": "x", "sol": 91}, "sol") == "x|sol91" and Z.ratio_group({"scape": "x"}, "global") is None


def test_zcam_joint_focus_from_below_offset(tmp_path, monkeypatch):
    """0.94.6 (Christian, 26 Oct: two backlash effects superimposed): bins whose focus came into place from below
    ("_fb") share the focus slope with an intercept offset of their own; the joint recovers it (-3 px on ZL at
    focus 600 in one block), leaves the other eye without one, and writes it into the candidate model."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    monkeypatch.setattr(Z, "LOW_LIST_CHECK", False)
    assert forms.zcam_focus_below("ZL034_F01254_fb_reg@blk") and not forms.zcam_focus_below("ZL034_F01254_reg")
    assert forms.zcam_state("ZL034_F01254_fb_reg@blk") == "regular"
    rng = np.random.default_rng(11)
    truth = _truth_model()
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, fb={"eye": "L", "F": 600.0, "offset_px": -3.0})
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900)
    st = _navcam_law(cams, rigT)
    res = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[a, b], st=st, nav_points=None, z_points=None,
                       iterations=60, passes=2, verbose=False, diagnostics=False, k_tests=False)
    rows = {r["eye"]: r for r in res["focus_below"]}
    assert set(rows) == {"ZL034"} and rows["ZL034"]["bins"] == 1 and rows["ZL034"]["blocks"] == ["blk_a"]
    assert rows["ZL034"]["offset_px"] == pytest.approx(-3.0, abs=0.3)
    assert rows["ZL034"]["sd_px"] and rows["ZL034"]["sd_px"] < 0.5
    t = res["eyes"]["ZL034"]["terms"]
    assert t["a_fx_backlash"] == pytest.approx(_law(truth, "ZL034", 600.0)[0], abs=0.5)    # the line itself unmoved
    cand = Z.candidate_model(res, truth)
    assert cand["cameras"]["ZL034"]["focus_below_offset_px"] == pytest.approx(rows["ZL034"]["offset_px"])
    assert "focus_below_offset_px" not in cand["cameras"]["ZR034"]


def test_joint_gate_rms_eyes_aspect_ratio_27oct():
    """27 Oct: Z48 was promoted with ZR fy / fx 0.9928, ZL / ZR f0 1.8 % apart, fitted high / low ratios 0.9939 and
    1.0231 (start kept, 'ok') and a Mastcam-Z rms of 3.3 px."""
    from mppp.calib import ztied as Z
    from mppp.calib.fit import check_zcam, ZCAM_MODEL_LIMITS
    base = {"ok": True, "checks": []}
    z = {"zcam_median_px": 0.19, "zcam_nonfinite_share": 0.0, "zcam_observations": 10}
    assert Z.joint_checks({"zcam": dict(z, zcam_rms_px=0.8)}, base)["ok"]
    bad = Z.joint_checks({"zcam": dict(z, zcam_rms_px=3.3)}, base)
    assert not bad["ok"] and any(c["check"] == "Mastcam-Z residual rms" and not c["ok"] for c in bad["checks"])
    assert ZCAM_MODEL_LIMITS["max_aspect_dev"] == 0.005          # Christian, 27 Oct: 0.5 %
    assert {"max_k1", "baseline_m", "max_f0_change_pct"} <= set(ZCAM_MODEL_LIMITS)   # (a comment once swallowed them)
    new = {"cameras": {"ZL048": {"f0_px": 6671.1, "aspect": 1.0006, "regular_ratio": 1.0095,
                                 "regular_ratio_note": "fitted regular ratio 0.9939 outside 1.004..1.016: the start ratio 1.0095 kept"},
                       "ZR048": {"f0_px": 6789.0, "aspect": 0.99284, "regular_ratio": 1.0095}}}
    old = {"cameras": {"ZL048": {"f0_px": 6700.1}, "ZR048": {"f0_px": 6703.7}}, "state": "calibrated"}
    ch = {c["check"]: c["ok"] for c in check_zcam(new, old)["checks"]}
    assert ch["ZL048 regular ratio"] is False and ch["ZR048 regular ratio"] is True
    assert ch["ZR048 aspect"] is False and ch["ZL048 aspect"] is True
    assert ch["eyes f0"] is False
    new["cameras"]["ZR048"]["f0_px"] = 6680.0
    assert {c["check"]: c["ok"] for c in check_zcam(new, old)["checks"]}["eyes f0"] is True
