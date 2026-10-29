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


def _block(tmp_path, name, shift, model, rng, sol, reg=None, fb=None, seq_offsets=None, dist_by_focus=None,
           pp_shift=None):
    """One block: the synthetic Navcam stations (helpers._synthetic) plus Mastcam-Z 34 mm stereo frames at each
    station (three pointings x three focus counts), every image seeing dense terrain patches.  ``reg``
    {"eye": "L", "F": 950.0, "offset": (dcx, dcy)}: that eye's images at that focus are in the regular state (f /
    regular ratio, the principal point moved by the offset; the other eye in backlash).  ``seq_offsets`` (0.95.0)
    {"ZCAM00001": px, ...}: each pointing is a sequence of its own (ZCAM00001..3) with sequence bins (the camera key
    carries sol and sequence, the spec ``sequence`` / ``sequence_sol``) and that focus offset on fx (fy x aspect)."""
    from helpers import _synthetic
    from mppp.calib.joint import Scape
    from scipy.spatial.transform import Rotation
    proj, truth, P, cams, rigT, noise, rng_ = _synthetic(tmp_path / name, noise_native_px=0.05, seed=int(sol))
    Rz = np.asarray(model["rig"]["R_sensor_from_ref"], float)
    tz = np.asarray(model["rig"]["t_sensor_from_ref"], float)
    zimgs, k = [], 0
    patches = []
    for st, C0 in (("S001D0000", np.array([0.0, 0.0, 1.9])), ("S001D0100", np.array([4.0, 1.0, 1.9]))):
        for iaz, az in enumerate((20.0, 60.0, 100.0)):
            seqname = f"ZCAM{iaz + 1:05d}" if seq_offsets is not None else "ZCAM00000"
            RL = _look(C0, az, 28.0)
            CL = C0 + np.array([0.0, 0.0, 0.15])
            tgt = CL + RL[2] * (1.9 / np.sin(np.radians(28.0)))
            if dist_by_focus is None:
                patches.append(np.column_stack([tgt[0] + rng.uniform(-1.4, 1.4, 500), tgt[1] + rng.uniform(-1.4, 1.4, 500),
                                                rng.normal(0, 0.15, 500)]))
            for jF, F in enumerate((300.0, 600.0, 950.0)):
                k += 1
                if dist_by_focus is not None:
                    # 0.95.0: each focus its own pointing and a scene at that focus's distance (the scene in focus)
                    RL = _look(C0, az + 12.0 * jF, 10.0)
                    D = float(dist_by_focus[F])
                    tgt = CL + RL[2] * D
                    w = 0.18 * D
                    patches.append(tgt + np.column_stack([rng.uniform(-w, w, 400), rng.uniform(-w, w, 400),
                                                          rng.normal(0, 0.12 * D, 400)]))
                RR = Rz @ RL
                CR = CL - RR.T @ tz
                sclk = f"{7000 + k:010d}_000"
                for eye, R, C in (("L", RL, CL), ("R", RR, CR)):
                    nm = f"Z{eye}0_{sol:04d}_{k:04d}.png"
                    regs = reg if isinstance(reg, list) else ([reg] if reg else [])   # 0.95.0: one spec per eye
                    rg = next((x for x in regs if eye == x["eye"] and F == x["F"]), None)
                    is_reg = rg is not None
                    is_fb = bool(fb) and eye == fb["eye"] and F == fb["F"]       # 0.94.4: focus from below
                    zimgs.append({"_reg": is_reg, "_regspec": rg, "_fb": is_fb, "name": nm, "stem": nm[:-4], "instrument": f"Z{eye}", "eye": eye, "sclk_key": sclk,
                                  "sol": sol, "site": 1, "drive": int(st[-4:]), "station": st, "sequence": seqname,
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
        key = (f"{r['camera_group']}_f{int(r['focus_count'])}" + (f"_s{sol:04d}_{r['sequence']}" if seq_offsets is not None else "")
               + ("_fb" if r["_fb"] else "") + ("_reg" if r["_reg"] else ""))
        if key not in zcid:
            p = _law(model, r["camera_group"], r["focus_count"], "regular" if r["_reg"] else "backlash")
            if seq_offsets is not None:             # 0.95.0: the sequence's focus offset
                asp = p[1] / p[0]
                p[0] += float(seq_offsets.get(r["sequence"], 0.0))
                p[1] += float(seq_offsets.get(r["sequence"], 0.0)) * asp
            if pp_shift and r["eye"] in pp_shift:   # 0.95.1: this eye's principal point moved in this block
                p[2] += float(pp_shift[r["eye"]][0])
                p[3] += float(pp_shift[r["eye"]][1])
            if r["_fb"]:                            # 0.94.4: the focus-from-below offset (px on fx, fy x aspect)
                asp = p[1] / p[0]
                p[0] += fb["offset_px"]
                p[1] += fb["offset_px"] * asp
            if r["_reg"]:
                rg = r["_regspec"]
                p[2] += rg["offset"][0]
                p[3] += rg["offset"][1]
                if rg.get("ratio"):                 # 0.94.4: this block's own high / low ratio
                    p[:2] *= float(model["cameras"][r["camera_group"]]["regular_ratio"]) / float(rg["ratio"])
            rec.add_camera(pycolmap.Camera(camera_id=nid, model="FULL_OPENCV", width=ZW, height=ZH, params=p))
            proj.cameras[key] = {"model": "FULL_OPENCV", "params": p.tolist(), "width": ZW, "height": ZH,
                                 "group": r["camera_group"], "focus_count_median": float(r["focus_count"]),
                                 "backlash_state": "regular" if r["_reg"] else "backlash", "fixed_params": ["cx", "cy"],
                                 **({"sequence": r["sequence"], "sequence_sol": sol, "n_images": 2} if seq_offsets is not None else {})}
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
    assert set(tests) == {"k2 = 0 (both eyes)"}                       # 0.95.1: no eyes-equal k1 test
    assert max(abs(z) for z in dist["difference_z"]) < 3               # nor do their curves
    assert any("radial displacement" in x for x in Z.describe(res))
    d = res["diagnostics"]
    assert {r["block"] for r in d["blocks"]} == {"blk_a", "blk_b"}
    ba = res["block_aspect"]                                          # 0.94.5: the Navcam's per-block aspect
    assert "error" not in ba, ba
    assert {r["block"] for r in ba["rows"]} == {"blk_a", "blk_b"} and all(r["navcam_ppm"] is not None for r in ba["rows"])
    assert all(abs(r["navcam_ppm"]) < 300 and abs(r["diff_ppm"]) < 1000 for r in ba["rows"]), ba["rows"]
    assert all(abs(r["yaw_mdeg"]) < 5 for r in d["blocks"])
    assert d["blocks"][0]["lines"]["ZL034"]["backlash"]["fx"]["bins"] == 3
    z = res["zcam"]                                                   # 0.95.0: per-image residuals, the worst frames
    assert z["zcam_images"] == 72 and len(z["worst_images"]) == 20 and z["images_over_1px"] == 0
    w = z["worst_images"][0]
    assert w["block"] in ("blk_a", "blk_b") and w["bin"].startswith("Z") and w["state"] == "backlash" and w["observations"] > 10
    assert w["rms_px"] >= z["worst_images"][-1]["rms_px"] and w["sequence"] == "ZCAM00000" and w["sol"] in (500, 900)
    out = Z.write_zjoint(res, tmp_path / "cand", model=start)
    assert "## Worst frames (0 of 72 over 1 px rms)" in (tmp_path / "cand" / "report.md").read_text()
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
    # 0.95.0 (Christian, 29 Oct): the combined Navcam law as model files, for calib lawcheck
    import json
    law = tmp_path / "nz" / NZ.LAW_FOLDER
    assert out["navcam_law"] == str(law) and "calib lawcheck --law in_use= --law combined=" in (tmp_path / "nz" / "report.md").read_text()
    nl = json.loads((law / "M2020_NL_fisheye_tangential.json").read_text())
    p = dict(zip(nl["param_names"], nl["params"]))
    assert p["fx"] == pytest.approx(res2["navcam"]["NL"]["params_T0"]["fx"]) and "combined joint" in nl["source"]
    assert nl["thermal"]["fx_ppm_per_degC"] == pytest.approx(res2["navcam"]["NL"]["fx_ppm_per_degC"])
    assert (law / "M2020_NR_fisheye_tangential.json").is_file() and (law / "M2020_N_rig.json").is_file()
    from mppp.calib import tied as T
    st2 = T.start_values(law)                                         # what calib lawcheck --law reads
    assert st2["G"]["NL"][0] == pytest.approx(res2["navcam"]["NL"]["params_T0"]["fx"], abs=1e-6)


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
    """0.94.4 (Christian, 24 Oct): the high / low ratio can vary by block - two blocks with ZL ratios 1.007 and
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
    assert res["covariance_method"].startswith("exact ties"), res.get("covariance_exact")    # 0.94.4
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
    """0.94.4 (Christian, 26 Oct: two backlash effects superimposed): bins whose focus came into place from below
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
    # 0.95.0 (Christian, 27 Oct): every fb bin of the eye also in the low state - the offset is one with the high /
    # low ratio: held at 0, the reason in the rows and the candidate, no offset written
    reg = {"eye": "L", "F": 600.0, "offset": (0.0, 0.0)}
    a2, _, _ = _block(tmp_path, "blk_a2", 0.0, truth, rng, 500, reg=reg, fb={"eye": "L", "F": 600.0, "offset_px": -3.0})
    b2, _, _ = _block(tmp_path, "blk_b2", 0.0, truth, rng, 900, reg=reg)
    r2 = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[a2, b2], st=st, nav_points=None, z_points=None,
                      iterations=40, passes=1, verbose=False, diagnostics=False, k_tests=False)
    rows2 = {r["eye"]: r for r in r2["focus_below"]}
    assert set(rows2) == {"ZL034"} and rows2["ZL034"]["held"] and rows2["ZL034"]["offset_px"] == 0.0
    assert rows2["ZL034"]["bins"] == 1 and rows2["ZL034"]["bins_also_low"] == 1
    c2 = Z.candidate_model(r2, truth)
    assert "focus_below_offset_px" not in c2["cameras"]["ZL034"] and "held" in c2["cameras"]["ZL034"]["focus_below"]
    assert any("focus from below: held at 0" in x for x in Z.describe(r2))


def test_joint_gate_rms_eyes_aspect_ratio_27oct():
    """27 Oct: Z48 was promoted with ZR fy / fx 0.9928, ZL / ZR f0 1.8 % apart, fitted high / low ratios 0.9939 and
    1.0231 (start kept, 'ok') and a Mastcam-Z rms of 3.3 px."""
    from mppp.calib import ztied as Z
    from mppp.calib.fit import check_zcam, ZCAM_MODEL_LIMITS
    base = {"ok": True, "checks": []}
    z = {"zcam_median_px": 0.19, "zcam_nonfinite_share": 0.0, "zcam_observations": 10}
    assert Z.joint_checks({"zcam": dict(z, zcam_rms_px=0.8)}, base)["ok"]
    bad = Z.joint_checks({"zcam": dict(z, zcam_rms_px=3.3, zcam_images=40, images_over_1px=2, worst_images=[
        {"image": "ZR0_0121_x.png", "block": "seitah_overlook", "sequence": "ZCAM08114", "rms_px": 4.1},
        {"image": "ZR0_0121_y.png", "block": "seitah_overlook", "sequence": "ZCAM08114", "rms_px": 2.0}])}, base)
    assert not bad["ok"] and any(c["check"] == "Mastcam-Z residual rms" and not c["ok"] for c in bad["checks"])
    note = next(c["note"] for c in bad["checks"] if c["check"] == "Mastcam-Z residual rms")      # 0.95.0: the frames
    assert "2 of 40 frames over 1 px, worst ZR0_0121_x.png (seitah_overlook, ZCAM08114, 4.10 px)" in note
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


def test_block_aspect_against_the_navcam():
    """0.94.5 (Christian, 27 Oct): per block the Mastcam-Z fy - fx (the line with the most observations) against the
    Navcam's per-block dfy - dfx; a block line more than 3000 ppm off fails the joint's check, a thin one is not judged."""
    from mppp.calib import ztied as Z
    nav = Z.navcam_aspect_by_block([
        {"scape": "a", "eye": "NL", "images": 20, "dfx_ppm": 100.0, "dfy_ppm": 300.0},
        {"scape": "a", "eye": "NR", "images": 20, "dfx_ppm": 0.0, "dfy_ppm": 400.0},
        {"scape": "a", "eye": "NR", "images": 3, "dfx_ppm": 0.0, "dfy_ppm": 9e5},       # too few images
        {"scape": "b", "eye": "NL", "images": 10, "dfx_ppm": 0.0, "dfy_ppm": -100.0}])
    assert nav["a"]["NL"] == pytest.approx(200.0) and nav["a"]["NR"] == pytest.approx(400.0)
    assert nav["a"]["mean"] == pytest.approx(300.0) and nav["b"]["mean"] == pytest.approx(-100.0)
    asp = 0.9995
    res = {"eyes": {"ZL048": {"terms": {"a_fx_backlash": 6700.0, "a_fy_backlash": 6700.0 * asp}}},
           "diagnostics": {"blocks": [
               {"block": "a", "lines": {"ZL048": {
                   "backlash": {"fx": {"a": 6700.0, "observations": 3000}, "fy": {"a": 6700.0 * (asp + 3e-4), "observations": 3000}},
                   "regular": {"fx": {"a": 6640.0, "observations": 100}, "fy": {"a": 6700.0, "observations": 100}}}}},
               {"block": "b", "lines": {"ZL048": {
                   "regular": {"fx": {"a": 6650.0, "observations": 2500}, "fy": {"a": 6650.0 * (asp + 5e-3), "observations": 2500}},
                   "backlash": {"fx": {"a": 6733.0, "observations": 150}, "fy": {"a": 6846.0, "observations": 150}}}}},
               {"block": "c", "lines": {"ZL048": {
                   "backlash": {"fx": {"a": 6700.0, "observations": 200}, "fy": {"a": 6800.0, "observations": 200}}}}}]}}
    ba = Z.block_aspect_rows(res, nav)
    r = {x["block"]: x for x in ba["rows"]}
    assert r["a"]["state"] == "backlash" and r["a"]["zcam_ppm"] == pytest.approx(300.0, abs=1)
    assert r["a"]["diff_ppm"] == pytest.approx(0.0, abs=1) and r["a"]["judged"]
    assert r["b"]["state"] == "regular" and r["b"]["diff_ppm"] == pytest.approx(5100.0, abs=2)
    assert not r["c"]["judged"]                                   # 200 observations, no Navcam
    chk = Z.joint_checks(dict(res, block_aspect=ba), {"checks": []}, limits={"min_block_aspect_check_observations": 1000})
    c = next(x for x in chk["checks"] if x["check"] == "per-block aspect vs the Navcam")
    assert not c["ok"] and "b ZL048 +5100" in c["note"]
    # 0.95.0 (Christian, 29 Oct): below the check's observation floor (20 k) a line is listed, not failed
    assert Z.ZJOINT_LIMITS["min_block_aspect_check_observations"] == 20000
    chk = Z.joint_checks(dict(res, block_aspect=ba), {"checks": []})
    c = next(x for x in chk["checks"] if x["check"] == "per-block aspect vs the Navcam")
    assert c["ok"] and "no line checked" in c["note"] and "(3 k), b ZL048 +5100 ppm (2 k)" in c["note"]
    big = dict(ba, rows=[dict(x, observations=x["observations"] * 10) for x in ba["rows"]])
    c = next(x for x in Z.joint_checks(dict(res, block_aspect=big), {"checks": []})["checks"]
             if x["check"] == "per-block aspect vs the Navcam")
    assert not c["ok"] and "b ZL048 +5100" in c["note"] and "2 block lines checked" in c["note"]
    lines = Z.describe(dict(res, block_aspect=ba, eyes={}))
    assert any(x.startswith("a ZL048 (backlash state, 3000 obs)") and "Navcam +300 ppm" in x for x in lines)


def test_ratio_held_with_few_low_state_images():
    """0.94.5 (Z48 of 27 Oct: 6 low-state images per eye, two blocks 1.2 % apart): with fewer than MIN_LOW_IMAGES the
    eye's high / low ratio is held at the 34 mm model's; no low-state image at all: held too; 34 mm
    itself and the single-state 110 mm are not."""
    from mppp.calib import ztied as Z
    from mppp.calib.fit import REGULAR_RATIO_DEFAULT
    sg = {"eyes": {"ZL048": {}, "ZR048": {}}, "low_images": {"ZL048": 5, "ZR048": 6}}
    held = Z.set_ratio_holds(sg, 48, verbose=False)
    assert set(held) == {"ZL048"} and 1.004 <= held["ZL048"] <= 1.016
    assert sg["eyes"]["ZL048"]["ratio_hold_source"] in ("the 34 mm model (ZL034)", "the default")
    assert Z.set_ratio_holds({"eyes": {"ZL034": {}}, "low_images": {"ZL034": 3}}, 34, verbose=False) == {}
    h = Z.set_ratio_holds({"eyes": {"ZL063": {}}, "low_images": {"ZL063": 0}}, 63, verbose=False)
    assert set(h) == {"ZL063"}                                    # no low-state image: not measured, held too
    assert Z.set_ratio_holds({"eyes": {"ZL110": {}}, "low_images": {"ZL110": 0}}, 110, verbose=False) == {}
    assert REGULAR_RATIO_DEFAULT and Z.REGULAR_SAME_SLOPE and Z.REGULAR_RATIO_BY == "global"


def test_focus_below_offset_needs_a_high_state_bin():
    """0.94.5 (Z48 of 27 Oct: -1.5 % / -1.0 % from two focus-from-below bins, both in the low state): the offset is
    fitted only for an eye with a focus-from-below bin in the high state."""
    from mppp.calib import ztied as Z
    sg = {"eyes": {"ZL048": {"G": np.zeros(Z.NG)}, "ZR048": {"G": np.zeros(Z.NG)}}}
    zunits = {1: {"group": "ZL048", "state": "regular", "fb": True}, 2: {"group": "ZL048", "state": "backlash", "fb": False},
              3: {"group": "ZR048", "state": "backlash", "fb": True}}
    hook, _g = Z.make_zties(sg, zunits)
    assert set(hook.focus_below) == {"ZR048"}


def test_thin_bins_left_out_of_the_lines():
    """0.94.5 (Christian, 27 Oct): a bin camera with fewer than MIN_BIN_OBSERVATIONS observations in the joint is left
    out of the lines (fx, fy free, the rest held); the low-state image count (for the ratio hold) skips it."""
    from types import SimpleNamespace as NS
    from mppp.calib import ztied as Z
    imgs = {1: NS(camera_id=11), 2: NS(camera_id=12), 3: NS(camera_id=13)}
    pts = {i: NS(track=NS(elements=[NS(image_id=1), NS(image_id=2)] + ([NS(image_id=3)] if i < 100 else [])))
           for i in range(600)}
    rec = NS(points3D=pts, images=imgs)
    proj = NS(cameras={"ZL048_F1@a": {"fixed_params": ["k3"]}, "ZL048_F2_reg@a": {"fixed_params": ["k3"]},
                       "ZL048_F3_reg@b": {"fixed_params": ["k3"]}},
              images=[{"camera_id": 11}, {"camera_id": 12}, {"camera_id": 12}, {"camera_id": 13}])
    zunits = {11: {"key": "ZL048_F1@a", "group": "ZL048", "state": "backlash"},
              12: {"key": "ZL048_F2_reg@a", "group": "ZL048", "state": "regular"},
              13: {"key": "ZL048_F3_reg@b", "group": "ZL048", "state": "regular"}}
    sg = {}
    thin = Z.mark_thin_bins(rec, proj, zunits, sg, min_obs=500)
    assert thin == ["ZL048_F3_reg@b (100)"] and zunits[13]["thin"] and not zunits[12]["thin"]
    assert proj.cameras["ZL048_F3_reg@b"]["free_params"] == ["fx", "fy"]
    assert {"k1", "k2", "cx", "cy"} <= set(proj.cameras["ZL048_F3_reg@b"]["fixed_params"])
    assert sg["low_images"] == {"ZL048": 2}                          # the thin bin's image not counted


def test_right_eye_low_state_principal_point_without_mixed_exposures(tmp_path, monkeypatch):
    """0.95.0 (Christian, 27 Oct: "the right eye's regular principal-point offset - the state shows in the centre"):
    both eyes low at the same focus in every exposure (no mixed exposures at all), ZR's low-state principal point
    -5, +3 px from its high state's.  0.93.2's mixed rule held both offsets and the rig took the shift up; now ZR's
    offset is fitted relative to ZL's (held, the reference eye) from the paired exposures, and the rig stays."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    from scipy.spatial.transform import Rotation
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    monkeypatch.setattr(Z, "LOW_LIST_CHECK", False)
    rng = np.random.default_rng(17)
    truth = _truth_model()
    regs = [{"eye": "L", "F": 950.0, "offset": (0.0, 0.0)}, {"eye": "R", "F": 950.0, "offset": (-5.0, 3.0)}]
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, reg=regs)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, reg=regs)
    st = _navcam_law(cams, rigT)
    Rt = np.asarray(truth["rig"]["R_sensor_from_ref"])

    def run():
        res = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st,
                           nav_points=None, z_points=None, iterations=60, passes=2, verbose=False, diagnostics=False,
                           k_tests=False)
        R = np.asarray(res["rig"]["R_sensor_from_ref"])
        return res, float(np.degrees(np.linalg.norm(Rotation.from_matrix(R @ Rt.T).as_rotvec())) * 1e3)

    res, rig_err = run()
    zl, zr = res["eyes"]["ZL034"], res["eyes"]["ZR034"]
    assert zl["mixed_exposures"] == 0 and zr["mixed_exposures"] == 0
    assert zl["paired_low_exposures"] == 12 and zr["paired_low_exposures"] == 12
    assert "d_cx_regular" in zl["held"] and "d_cx_regular" not in zr["held"]
    assert zr["terms"]["d_cx_regular"] == pytest.approx(-5.0, abs=0.4)
    assert zr["terms"]["d_cy_regular"] == pytest.approx(3.0, abs=0.4)
    assert rig_err < 3.0
    cand = Z.candidate_model(res, truth)
    rp = cand["cameras"]["ZR034"]["regular_pp_offset"]
    assert not rp["held"] and rp["paired_low_exposures"] == 12 and "relative to ZL" in rp["note"]
    assert cand["cameras"]["ZL034"]["regular_pp_offset"]["held"]
    from mppp.calib.fit import check_zcam
    note = next(c["note"] for c in check_zcam(cand, truth)["checks"] if c["check"] == "ZR034 regular principal point")
    assert "12 low-state exposures paired with ZL" in note
    assert any("low-state exposures paired with ZL" in x for x in Z.describe(res))
    # the old rule (0.93.2): both held, the rig takes the shift up
    monkeypatch.setattr(Z, "MIN_PAIRED_LOW_EXPOSURES", 10 ** 6)
    res0, rig_err0 = run()
    assert "d_cx_regular" in res0["eyes"]["ZR034"]["held"] and rig_err0 > 3 * max(rig_err, 1.0)


def test_zcam_joint_sequence_offsets(tmp_path, monkeypatch):
    """0.95.0 (Christian, 27 Oct: different sequences sit at different linear offsets with the same slope): on
    sequence bins each (block, sol, sequence) of an eye gets its own intercept offset, drawn to 0; the truth has
    sequences +10 / 0 / -10 px off the law in both blocks - the offsets are recovered, the eye's line and slope stay
    at the truth, the candidate records them and the check passes; the diagnostic pass puts the bins at the law plus
    their offsets.  Without sequence bins (0.94.x projects) nothing changes."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    monkeypatch.setattr(Z, "MIN_BIN_OBSERVATIONS", 100)  # (the synthetic sequence bins: two frames, ~440 observations)
    rng = np.random.default_rng(19)
    truth = _truth_model()
    so = {"ZCAM00001": 10.0, "ZCAM00002": 0.0, "ZCAM00003": -10.0}
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, seq_offsets=so)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, seq_offsets=so)
    st = _navcam_law(cams, rigT)
    assert a.project.cameras["ZL034_f300_s0500_ZCAM00001"]["sequence"] == "ZCAM00001"
    res = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st,
                       nav_points=None, z_points=None, iterations=60, passes=2, verbose=False, k_tests=False)
    rows = {(r["eye"], r["block"], r["sequence"]): r for r in res["sequence_offsets"]}
    assert len(rows) == 12 and all(not r["held"] for r in rows.values())        # 2 eyes x 2 blocks x 3 sequences
    for eye in ("ZL034", "ZR034"):
        for blk in ("blk_a", "blk_b"):
            for seq, off in so.items():
                r = rows[(eye, blk, seq)]
                assert r["offset_px"] == pytest.approx(off, abs=0.8), (eye, blk, seq, r)
                assert r["sol"] == (500 if blk == "blk_a" else 900) and r["images"] == 6 and r["bins"] == 3
                assert r["focus_span"] == pytest.approx(650.0) and r["sd_px"] and r["sd_px"] < 1.0
        e = res["eyes"][eye]
        tm = e["terms"]
        ln = truth["cameras"][eye]["lines"]["backlash"]
        assert tm["a_fx_backlash"] == pytest.approx(ln["fx"]["a"], abs=0.8)       # the line: the sequences' mean
        assert tm["b"] == pytest.approx(0.04, abs=2e-3)
        assert e["sequences"]["free"] == 6 and e["sequences"]["held"] == 0 and e["sequences"]["with_focus_span"] == 6
        assert 1500 < e["sequences"]["max_abs_ppm"] < 2600
    assert any("sequence offsets: 6 fitted (6 with a focus span" in x for x in Z.describe(res))
    cand = Z.candidate_model(res, truth)
    gm = cand["cameras"]["ZL034"]
    assert gm["sequence_offsets"]["free"] == 6 and len(gm["sequence_offsets"]["rows"]) == 6
    from mppp.calib.fit import check_zcam
    chk = {c["check"]: c for c in check_zcam(cand, truth)["checks"]}
    assert chk["ZL034 sequence offsets"]["ok"] and "6 fitted, 0 held at 0" in chk["ZL034 sequence offsets"]["note"]
    assert not forms.validate_zcam(cand)
    out = Z.write_zjoint(res, tmp_path / "cand", model=truth)
    rep = (tmp_path / "cand" / "report.md").read_text()
    assert "## Sequence offsets" in rep and "| ZL034 | blk_a | 500 | ZCAM00001 | +" in rep
    assert any(c["check"] == "ZR034 sequence offsets" and c["ok"] for c in out["check"]["checks"])
    # the diagnostic pass: the per-block lines sit at the law (the offsets taken out), not 10 px apart
    d = res["diagnostics"]["blocks"]
    a_zl = truth["cameras"]["ZL034"]["lines"]["backlash"]["fx"]["a"]
    for blk in d:
        assert blk["lines"]["ZL034"]["backlash"]["fx"]["a"] == pytest.approx(a_zl, abs=1.5)
    # a sequence offset of 2 % is a mis-tie: the check fails and names it
    gm["sequence_offsets"]["rows"][0].update(ppm=20000.0)
    gm["sequence_offsets"]["max_abs_ppm"] = 20000.0
    c2 = {c["check"]: c for c in check_zcam(cand, truth)["checks"]}["ZL034 sequence offsets"]
    assert not c2["ok"] and "blk_a ZCAM00001 +20000" in c2["note"]
    # focus bins (no sequence on the specs): no offsets at all
    a0, _, _ = _block(tmp_path, "blk_a0", 0.0, truth, rng, 500)
    b0, _, _ = _block(tmp_path, "blk_b0", 0.0, truth, rng, 900)
    r0 = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[a0, b0], st=st, nav_points=None, z_points=None,
                      iterations=30, passes=1, verbose=False, diagnostics=False, k_tests=False)
    assert r0["sequence_offsets"] == [] and r0["eyes"]["ZL034"]["sequences"]["free"] == 0
    assert "sequence_offsets" not in Z.candidate_model(r0, truth)["cameras"]["ZL034"]


def test_eye_slope_tests_find_a_rig_translation_error_along_the_boresight(tmp_path, monkeypatch):
    """0.95.0 (Christian, 29 Oct: "the efl vs focus slopes of both should be about equal"; Z34 28 Oct: ZL 0.0245, ZR
    0.0351 px/count in every block).  Blocks made with the right camera 6 mm further along the left boresight than
    the rig the joint holds, each focus looking at a scene at its own distance (8, 4, 2 m): the joint gives ZR a
    steeper focus slope than ZL although both are 0.04 px/count; the eye-slope tests free the rig translation (the
    right centre moves forward and the slopes come together), and one slope for both eyes then costs little against
    the free rig but much against the held one."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    rng = np.random.default_rng(11)
    model = _truth_model()
    truth = copy.deepcopy(model)
    R = np.asarray(model["rig"]["R_sensor_from_ref"], float)
    C = -R.T @ np.asarray(model["rig"]["t_sensor_from_ref"], float)
    truth["rig"]["t_sensor_from_ref"] = (-R @ (C + np.array([0.0, 0.0, 0.006]))).tolist()   # 6 mm along ZL's z
    dist = {300.0: 8.0, 600.0: 4.0, 950.0: 2.0}
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, dist_by_focus=dist)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, dist_by_focus=dist)
    st = _navcam_law(cams, rigT)
    res = Z.fit_zjoint({}, model=model, scapes=[a, b], st=st, nav_points=None, z_points=None, iterations=40,
                       passes=1, diagnostics=False, verbose=False, k_tests=False, slope_tests=True, rig_fit="rotation")
    bl, br = res["eyes"]["ZL034"]["terms"]["b"], res["eyes"]["ZR034"]["terms"]["b"]
    est = res["eye_slope_tests"]
    t = {x["test"]: x for x in est["tests"]}
    assert set(t) == {"rig translation free", "one slope for both eyes", "one slope, rig translation free"}, est
    free, one, both = t["rig translation free"], t["one slope for both eyes"], t["one slope, rig translation free"]
    gap0 = abs(br - bl)
    gap1 = abs(free["b"]["ZR034"] - free["b"]["ZL034"])
    assert gap0 > 0.002, (bl, br)                                     # the held rig puts the error in ZR's slope
    assert gap1 < 0.5 * gap0, (gap0, gap1)                            # the free rig takes it back
    dz = free["rig_centre_change_mm"][2]
    assert 2.0 < dz < 10.0, free["rig_centre_change_mm"]              # forward, of the order of the 6 mm
    assert free["delta_chi2"] < 0 and one["delta_chi2"] > 0          # the free rig fits better, one slope worse
    assert both["against"] == "rig translation free" and both["delta_chi2"] < 0.25 * one["delta_chi2"]
    assert both["b"]["ZL034"] == pytest.approx(both["b"]["ZR034"], abs=1e-5)
    lines = Z.eye_slope_lines(est)
    assert lines[0].startswith("eye slopes in the joint: ZL034") and any("z (along the boresight)" in x for x in lines)


def test_zjoint_refines_the_rig_translation_and_reports_its_significance(tmp_path, monkeypatch):
    """0.95.1 (Christian, 29 Oct: "the z34 rig translation should not be held ... let these rig parameters change and
    optimize while keeping a close eye on their significance").  The blocks of the eye-slope test (the right camera
    6 mm along the left boresight from the start rig): the joint's default refines the translation - the centre moves
    forward by about 6 mm with a z well beyond 3, the eye slopes agree; the eye-slope tests are then "rig
    translation held" (much worse) and "one slope for both eyes" (costs little)."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    rng = np.random.default_rng(11)
    model = _truth_model()
    truth = copy.deepcopy(model)
    R = np.asarray(model["rig"]["R_sensor_from_ref"], float)
    C = -R.T @ np.asarray(model["rig"]["t_sensor_from_ref"], float)
    truth["rig"]["t_sensor_from_ref"] = (-R @ (C + np.array([0.0, 0.0, 0.006]))).tolist()
    dist = {300.0: 8.0, 600.0: 4.0, 950.0: 2.0}
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, dist_by_focus=dist)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, dist_by_focus=dist)
    st = _navcam_law(cams, rigT)
    res = Z.fit_zjoint({}, model=model, scapes=[a, b], st=st, nav_points=None, z_points=None, iterations=40,
                       passes=1, diagnostics=False, verbose=False, k_tests=False, slope_tests=True)
    r = res["rig"]
    assert r["fit"] == "full" and r["translation_source"].startswith("refined")
    assert 4.0 < r["translation_change_mm"][2] < 8.0, r
    assert r["translation_sd_mm"] and r["translation_z"][2] > 3.0 and r["translation_significant"], r
    bl, br = res["eyes"]["ZL034"]["terms"]["b"], res["eyes"]["ZR034"]["terms"]["b"]
    assert abs(br - bl) < 0.002, (bl, br)
    t = {x["test"]: x for x in res["eye_slope_tests"]["tests"]}
    assert set(t) == {"rig translation held", "one slope for both eyes"}, t
    held, one = t["rig translation held"], t["one slope for both eyes"]
    assert held["delta_chi2"] > 0 and held["significant"] and held["dof"] == 3
    assert one["delta_chi2"] < 0.1 * held["delta_chi2"]
    lines = Z.describe(res)
    assert any(x.startswith("rig translation refined: the right centre moved") for x in lines)
    assert any("(z +" in x and "significant" in x for x in lines)
    new = Z.candidate_model(res, model)
    assert new["rig"]["translation_change_mm"] == r["translation_change_mm"] and "translation refined" in new["rig"]["note"]


def test_zjoint_early_mission_principal_point_offset(tmp_path, monkeypatch):
    """0.95.1 (Christian, 29 Oct: "fix the Zcam rig drift for all zooms.  Put this into an early mission principal
    point offset if it is significant"; Z34: the rig pitch -28 / -36 mdeg at Chal Rocks (sol 102) and Seitah Overlook
    (119), ~0 from sol 182).  An early block (sol 100) whose ZR principal point sits (+0.8, -3.0) px off the later
    block's (sol 900): the joint fits ZR's early offset for its bins before sol 150, significant, into the candidate
    model (pp.early, applied), which the block runs then apply to early ZR bins."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    from mppp.sfm.project import _split_by_focus
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    rng = np.random.default_rng(5)
    model = _truth_model()
    a, cams, rigT = _block(tmp_path, "blk_early", 0.0, model, rng, 100, pp_shift={"R": (0.8, -3.0)})
    b, _, _ = _block(tmp_path, "blk_late", 0.0, model, rng, 900)
    st = _navcam_law(cams, rigT)
    res = Z.fit_zjoint({}, model=model, scapes=[a, b], st=st, nav_points=None, z_points=None, iterations=40,
                       passes=1, diagnostics=False, verbose=False, k_tests=False)
    e = res["early_pp"]
    assert e["before_sol"] == Z.ZCAM_EARLY_SOL and set(e["eyes"]) == {"ZR034"}, e
    r = e["eyes"]["ZR034"]
    assert r["fitted"] and r["blocks"] == ["blk_early"] and r["bins"] == 3, r
    assert r["offset_px"][0] == pytest.approx(0.8, abs=0.25) and r["offset_px"][1] == pytest.approx(-3.0, abs=0.25), r
    assert r["sd_px"] and abs(r["z"][1]) > 3.0 and r["applied"], r
    assert any("early-mission principal point of ZR034" in x and "significant: in the candidate model" in x
               for x in Z.describe(res))
    new = Z.candidate_model(res, model)
    pe = new["cameras"]["ZR034"]["pp"]["early"]
    assert pe["applied"] and pe["before_sol"] == Z.ZCAM_EARLY_SOL and pe["dcy_px"] == pytest.approx(-3.0, abs=0.25)
    assert "early" not in (new["cameras"]["ZL034"].get("pp") or {})
    # the block runs: an early ZR bin starts with the offset, a later one without
    base = {"model": "FULL_OPENCV", "params": [4700.0, 4700.0, 800.0, 600.0] + [0.0] * 8, "width": 1648, "height": 1200,
            "source": "test"}
    imgs = [{"camera_group": "ZR034", "focus_count": 600.0, "sol": s_, "name": f"x{s_}"} for s_ in (100, 100)]
    out_e = _split_by_focus(imgs, {"ZR034": dict(base)}, {"ZR034": "Z"}, 50.0, "free", model=new)
    imgs = [{"camera_group": "ZR034", "focus_count": 600.0, "sol": s_, "name": f"y{s_}"} for s_ in (900, 900)]
    out_l = _split_by_focus(imgs, {"ZR034": dict(base)}, {"ZR034": "Z"}, 50.0, "free", model=new)
    ce = next(v for k, v in out_e.items() if k.startswith("ZR034_"))
    cl = next(v for k, v in out_l.items() if k.startswith("ZR034_"))
    assert ce["params"][3] - cl["params"][3] == pytest.approx(pe["dcy_px"], abs=1e-6)
    assert ce["params"][2] - cl["params"][2] == pytest.approx(pe["dcx_px"], abs=1e-6)
    assert "early-mission pp offset" in ce["source"]


def test_freed_fixed_term_test_moves_the_term(tmp_path, monkeypatch):
    """0.95.1: the "free (vs the fixed values)" test of a held k2 (truth 0.40, held at 0.3): the freed law moves to
    the truth and the fixed value costs a positive delta chi2 (the 28 Oct runs: k2 0.5 -> 0.5000013, the delta chi2
    convergence noise taken as abs())."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {"ZL": {"k2": 0.3}, "ZR": {"k2": 0.3}})
    rng = np.random.default_rng(3)
    truth = _truth_model()
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900)
    st = _navcam_law(cams, rigT)
    res = Z.fit_zjoint({}, model=truth, scapes=[a, b], st=st, nav_points=None, z_points=None, iterations=40,
                       passes=1, verbose=False, diagnostics=False)
    t = {x["test"]: x for x in res["distortion"]["tests"]}["k2 free (vs the fixed values)"]
    assert t["delta_chi2"] > 10 and t["significant"] and t["converged"], t
    assert all(v["k2"] == pytest.approx(0.40, abs=0.05) for v in t["refit_terms"].values()), t
