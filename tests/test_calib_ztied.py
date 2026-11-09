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


@pytest.fixture(autouse=True)
def _flat_synthetic_blocks_are_not_weak(monkeypatch):
    """0.96 (item F): the synthetic blocks point every frame at pitch 28 deg, so every sequence covers one vertical
    field of view and would count as weak; the weak-sequence fy offsets are turned off for these tests except where
    a test turns them on."""
    from mppp.calib import ztied as Z
    monkeypatch.setattr(Z, "WEAK_SEQUENCE_FOV", 0.0)


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
           pp_shift=None, seq_aspect=None):
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
            if seq_aspect:                          # 0.96: this sequence's fy off the law by so many ppm (fx on it)
                p[1] *= 1.0 + 1e-6 * float(seq_aspect.get(r["sequence"], 0.0))
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
    # 0.95.1: every bin's own fx in the diagnostic pass, and the efl against focus figure
    b0 = res["diagnostics"]["bins"]
    assert len(b0) == 12 and {x["eye"] for x in b0} == {"ZL034", "ZR034"} and all(x["observations"] > 0 for x in b0)
    assert (tmp_path / "cand" / "zcam_efl_vs_focus.png").is_file()
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
    hook, _g = Z.make_zties(sg, zunits)                           # 0.96: not fitted unless asked
    assert set(hook.focus_below) == set() and all("not fitted" in v for v in hook.focus_below_held.values())
    assert set(hook.focus_below_held) == {"ZL048", "ZR048"}
    hook, _g = Z.make_zties(dict(sg, focus_below_fit=True), zunits)
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



def test_z110_k1_held_at_zero_and_tested_alone(monkeypatch):
    from mppp.calib import ztied as Z
    """1 Nov (Christian): 110 mm k1 held at 0 with k2, p1, p2; the distortion tests free all of them, then k1 alone."""
    import copy as _copy
    from mppp.calib import forms
    assert forms.zcam_fixed_terms("ZL110") == {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    assert "k1" not in forms.zcam_fixed_terms("ZL079") and forms.zcam_fixed_terms("ZR034") == {"k2": 0.6}
    calls = []

    def fake_adjust(rec, proj, idx, st, units, zunits, sg, **kw):
        calls.append(kw["hold"])
        g = {e: np.asarray(v["G"], float) for e, v in sg["eyes"].items()}
        return {"final_cost": 99.0}, g, None
    monkeypatch.setattr(Z, "adjust_z", fake_adjust)
    G = np.zeros(Z.NG)
    G[Z.GLOBAL_TERMS.index("a_fx_backlash")] = G[Z.GLOBAL_TERMS.index("a_fy_backlash")] = 14700.0
    g = {"ZL110": G.copy(), "ZR110": G.copy()}
    sg = {"eyes": {e: {"G": G.copy()} for e in g}}
    fixed = {e: forms.zcam_fixed_terms(e) for e in g}

    class P:
        cameras = {}
    out = Z.distortion_tests(None, P(), {}, {}, {}, {}, sg, g, {"final_cost": 100.0, "covariance": {"variance_factor": 1.0}},
                             verbose=False, fixed=fixed)
    names = [o["test"] for o in out]
    assert names == ["k1, k2, p1, p2 free (vs the fixed values)", "k1 free (vs the fixed value)",
                     "k2 free (vs the fixed value)", "p1, p2 free (vs the fixed values)"]
    assert calls[0] == {"ZL110": [], "ZR110": []}
    assert calls[1] == {"ZL110": ["k2", "p1", "p2"], "ZR110": ["k2", "p1", "p2"]}
    assert calls[3] == {"ZL110": ["k1", "k2"], "ZR110": ["k1", "k2"]}
    assert out[1]["dof"] == 2 and out[1]["delta_chi2"] == pytest.approx(2.0) and out[3]["dof"] == 4
    assert set(out[0]["refit_terms"]["ZL110"]) == {"k1", "k2", "p1", "p2"} and out[0]["max_tangential_px"] >= 0
    # 63 mm (k2, p1, p2 held): k2 alone and the tangential pair alone; 34 mm (k2 only): no split
    calls.clear()
    g63 = {"ZL063": G.copy(), "ZR063": G.copy()}
    sg63 = {"eyes": {e: {"G": G.copy()} for e in g63}}
    out = Z.distortion_tests(None, P(), {}, {}, {}, {}, sg63, g63,
                             {"final_cost": 100.0, "covariance": {"variance_factor": 1.0}}, verbose=False,
                             fixed={e: forms.zcam_fixed_terms(e) for e in g63})
    assert [o["test"] for o in out] == ["k2, p1, p2 free (vs the fixed values)", "k2 free (vs the fixed value)",
                                        "p1, p2 free (vs the fixed values)"]
    G2 = G.copy()
    G2[Z.GLOBAL_TERMS.index("p1")] = 1e-3
    assert Z._tangential_corner_px(G2) == pytest.approx(14700 * 1e-3 * ((824 / 14700) ** 2 + 3 * (600 / 14700) ** 2),
                                                        rel=0.2)


def test_zcam_joint_aspect_held(tmp_path, monkeypatch):
    """6 Nov 2026: ``aspect_hold`` ties each eye's backlash fy line to its fx line (fy / fx = the held value)."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    rng = np.random.default_rng(5)
    truth = _truth_model()
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900)
    st = _navcam_law(cams, rigT)
    res = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[a, b], st=st, nav_points=None, z_points=None, iterations=40,
                       passes=1, verbose=False, diagnostics=False, k_tests=False, aspect_hold=1.0)
    for g in ("ZL034", "ZR034"):
        tm = res["eyes"][g]["terms"]
        assert tm["a_fy_backlash"] / tm["a_fx_backlash"] == pytest.approx(1.0, abs=1e-6)


# ====================================================================================== 0.96: items A-D of 5 Nov
def test_pooled_diagnostic_pass_recovers_a_sequence_aspect(tmp_path, monkeypatch):
    """0.96, item A (Christian, 5 Nov: "yes to pooling on the block or sequence level"): the diagnostic pass pooled
    by sequence - one fx scale, one fy scale and cx, cy per (eye, sequence), the frames on the line at their own
    focus.  The truth has ZCAM00002's fy +3000 ppm off the law in both blocks (fx on it).  The joint's fy line is the
    sequences' mean (+1000), so about the law ZCAM00002's pools sit at +2000 ppm and the other sequences' at -1000;
    the per-bin bins beside them agree, the block aspect gate reads the pooled block mean (about 0) with the per-bin
    line beside it, and the sequence table leads with ZCAM00002.  pool=None: the per-bin gate
    as before 0.96; pool="block": one pool per eye, block and state."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    monkeypatch.setattr(Z, "MIN_BIN_OBSERVATIONS", 100)
    rng = np.random.default_rng(23)
    truth = _truth_model()
    so = {"ZCAM00001": 0.0, "ZCAM00002": 0.0, "ZCAM00003": 0.0}
    sa = {"ZCAM00002": 3000.0}
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, seq_offsets=so, seq_aspect=sa)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, seq_offsets=so, seq_aspect=sa)
    st = _navcam_law(cams, rigT)
    res = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st,
                       nav_points=None, z_points=None, iterations=60, passes=2, verbose=False, k_tests=False)
    d = res["diagnostics"]
    assert d["pool"] == "sequence" and len(d["pools"]) == 12 and d["frames_at_own_focus"] == 0   # (frames at the bin's focus)
    pools = {(p["eye"], p["block"], p["sequence"]): p for p in d["pools"]}
    for eye in ("ZL034", "ZR034"):
        for blk in ("blk_a", "blk_b"):
            p2 = pools[(eye, blk, "ZCAM00002")]
            assert p2["aspect_ppm"] == pytest.approx(2000.0, abs=500), p2
            assert abs(p2["dfx_ppm"]) < 500 and p2["bins"] == 3 and p2["observations"] > 300 and p2["sd_aspect_ppm"] < 400
            assert p2["sol"] == (500 if blk == "blk_a" else 900) and p2["states"] == ["backlash"]
            for sq in ("ZCAM00001", "ZCAM00003"):
                assert pools[(eye, blk, sq)]["aspect_ppm"] == pytest.approx(-1000.0, abs=500), pools[(eye, blk, sq)]
    # the per-bin pass beside it: every ZCAM00002 bin about +2000 ppm in dfy - dfx as well
    b2 = [x for x in d["bins"] if x["sequence"] == "ZCAM00002"]
    assert len(b2) == 12 and all(abs((x["dfy_ppm"] - x["dfx_ppm"]) - 2000.0) < 600 for x in b2), b2
    # the gate reads the pooled value (the observation-weighted mean over the block's pools: about 0)
    ba = res["block_aspect"]
    assert ba["source"] == "pooled" and ba["pool"] == "sequence"
    rows = [r for r in ba["rows"] if r["visit"] is None]
    assert len(rows) == 4 and all(r["source"] == "pooled" and r["pools"] == 3 and r["judged"] for r in rows)
    for r in rows:
        assert abs(r["zcam_ppm"]) < 400 and r["per_bin_ppm"] is not None and r["per_bin_state"] == "backlash"
        assert abs(r["per_bin_ppm"] - r["zcam_ppm"]) < 200 and r["per_bin_diff_ppm"] is not None
    assert not any(r["visit"] for r in ba["rows"])                      # one visit per block
    assert ba["sequences"][0]["sequence"] == "ZCAM00002" and all(s["sequence"] == "ZCAM00002" for s in ba["sequences"][:4])
    lines = Z.describe(res)
    assert any("per-block aspect from the pooled diagnostic pass" in x for x in lines)
    assert any(x.startswith("blk_a ZL034 (3 pools,") and "pooled d(fy - fx)" in x and "per-bin line" in x for x in lines)
    assert any(x.startswith("pooled sequences with the largest |dfy - dfx|") and "ZCAM00002" in x for x in lines)
    # the sequence scales (for the state flags) come from the pools
    sc = {(s["eye"], s["block"], s["sequence"]): s for s in Z.sequence_scales(d)}
    assert sc[("ZL034", "blk_a", "ZCAM00002")]["source"] == "pooled" and len(sc) == 12
    # item C ran: nothing flagged (no eye sits at its low-state step), the check a passing one without a warning
    os_ = res["opposite_states"]
    assert os_["sequences_checked"] == 6 and os_["flags"] == [] and os_["steps_ppm"]["ZL034"] < -5000
    # item B ran on the final pass: 12 (eye, sequence) rows, small and directionless residuals
    rd = res["residual_directions"]
    assert len(rd["sequences"]) == 12 and all(s["rms_px"] < 0.3 for s in rd["sequences"])
    assert all(abs(s["fy_scale_left_ppm"]) < 1500 for s in rd["sequences"]) and rd["sequences"][0]["rows"] == []
    out = Z.write_zjoint(res, tmp_path / "cand", model=truth)
    rep = (tmp_path / "cand" / "report.md").read_text()
    assert "## Pooled diagnostic pass" in rep and "| ZCAM00002 |" in rep and "## Residual direction per sequence" in rep
    c = {x["check"]: x for x in out["check"]["checks"]}["opposite backlash states between the eyes"]
    assert c["ok"] and not c["warning"] and "none flagged" in c["note"]
    # pool=None: the per-bin gate alone, as before 0.96
    r0 = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st,
                      nav_points=None, z_points=None, iterations=30, passes=1, verbose=False, k_tests=False, pool=None)
    assert r0["diagnostics"]["pool"] is None and "pools" not in r0["diagnostics"]
    assert r0["block_aspect"]["source"] == "per-bin" and r0["block_aspect"]["sequences"] == []
    assert all(r["source"] == "per-bin" and r["state"] == "backlash" for r in r0["block_aspect"]["rows"])
    assert Z.sequence_scales(r0["diagnostics"])[0]["source"] == "per-bin mean"
    assert r0["opposite_states"]["sequences_checked"] == 6
    # pool="block": one pool per eye, block and state
    rec_, proj_, idx_, st_, units_, zunits_, sg_ = Z.build({}, copy.deepcopy(truth), None, 34, None, None,
                                                           scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st)
    assert {Z.pool_key(u, "block") for u in zunits_.values()} == {"blk_a|backlash", "blk_b|backlash"}
    assert len({Z.pool_key(u, "sequence") for u in zunits_.values()}) == 6
    with pytest.raises(ValueError):
        Z.diagnostic_pass(rec_, proj_, idx_, st_, units_, zunits_, sg_, {}, pool="eye")


def test_block_aspect_pooled_rows_visits_and_sequences():
    """0.96, item A: the gate row is the observation-weighted mean of the block's pools' dfy - dfx, the per-bin line
    beside it; a block visited twice (sols 80 and 370) gets a row per visit, report only; the sequence table holds
    the pools with the largest |dfy - dfx|."""
    from mppp.calib import ztied as Z
    nav = {"a": {"mean": 100.0, "NL": 50.0, "NR": 150.0, "images": 40}}
    asp = 0.9995
    res = {"eyes": {"ZL034": {"terms": {"a_fx_backlash": 4700.0, "a_fy_backlash": 4700.0 * asp}}},
           "diagnostics": {"pool": "sequence",
                           "blocks": [{"block": "a", "lines": {"ZL034": {"backlash": {
                               "fx": {"a": 4700.0, "observations": 30000}, "fy": {"a": 4700.0 * (asp + 2e-3), "observations": 30000}}}}}],
                           "pools": [
                               {"eye": "ZL034", "key": "a|s80|ZCAM00010", "block": "a", "sequence": "ZCAM00010", "sol": 80, "states": ["backlash"],
                                "bins": 4, "observations": 10000, "dfx_ppm": 100.0, "dfy_ppm": 600.0, "aspect_ppm": 500.0},
                               {"eye": "ZL034", "key": "a|s370|ZCAM08401", "block": "a", "sequence": "ZCAM08401", "sol": 370, "states": ["backlash"],
                                "bins": 11, "observations": 30000, "dfx_ppm": 0.0, "dfy_ppm": -4500.0, "aspect_ppm": -4500.0, "sd_aspect_ppm": 120.0}]}}
    ba = Z.block_aspect_rows(res, nav)
    assert ba["source"] == "pooled" and ba["pool"] == "sequence"
    main = [r for r in ba["rows"] if r["visit"] is None]
    assert len(main) == 1 and main[0]["source"] == "pooled" and main[0]["pools"] == 2 and main[0]["observations"] == 40000
    assert main[0]["zcam_ppm"] == pytest.approx((500.0 * 10000 - 4500.0 * 30000) / 40000) and main[0]["judged"]
    assert main[0]["diff_ppm"] == pytest.approx(main[0]["zcam_ppm"] - 100.0)
    assert main[0]["per_bin_ppm"] == pytest.approx(2000.0, abs=1) and main[0]["per_bin_diff_ppm"] == pytest.approx(1900.0, abs=1)
    vis = [r for r in ba["rows"] if r["visit"]]
    assert [r["visit"] for r in vis] == ["sol 80", "sol 370"] and not any(r["judged"] for r in vis)
    assert vis[1]["zcam_ppm"] == pytest.approx(-4500.0) and vis[1]["pools"] == 1 and vis[1]["per_bin_ppm"] is None
    assert ba["sequences"][0]["sequence"] == "ZCAM08401" and ba["sequences"][0]["sd_aspect_ppm"] == 120.0
    assert Z.visits_of([72, 80, 100, 360, 379]) == [(72, 100), (360, 379)] and Z.visits_of([None]) == []
    # the gate: the pooled row fails at -3250 ppm against the Navcam (one row checked, the visits not counted)
    chk = Z.joint_checks(dict(res, block_aspect=ba), {"checks": []})
    c = next(x for x in chk["checks"] if x["check"] == "per-block aspect vs the Navcam")
    assert not c["ok"] and "1 block lines checked" in c["note"] and "a ZL034 -3350" in c["note"]
    lines = Z.describe(dict(res, block_aspect=ba, eyes={}))
    assert any(x.startswith("a ZL034 (2 pools, 40000 obs): Mastcam-Z pooled d(fy - fx) -3250 ppm, Navcam +100 ppm, difference -3350 ppm; "
                            "per-bin line +2000 ppm (backlash state, difference +1900)") for x in lines), lines
    assert any(x.startswith("a ZL034 sols") is False and x.startswith("a ZL034 sol 370 (1 pools, 30000 obs)") and x.endswith("(visit: report only)")
               for x in lines), lines


def test_opposite_state_flags_and_the_warning():
    """0.96, item C (Z63 ZCAM08038, sol 78: ZL's bins -3,450 / -3,760 ppm against a low-state step of -4,170, ZR
    within +900 of 0 - ZL in the low state, labelled high): the flag fires on that case, not when the eyes move
    together, nor when fx and fy part; the mirror case (an eye labelled low at +step) flags a possible high state;
    a warning in the checks, never a failure; without pools the per-bin means serve."""
    from mppp.calib import ztied as Z
    eyes = {"ZL063": {"terms": {"a_fx_backlash": 8615.5, "a_fx_regular": 8579.6, "a_fy_backlash": 8613.0}, "held": []},
            "ZR063": {"terms": {"a_fx_backlash": 8612.6, "a_fx_regular": 8577.0, "a_fy_backlash": 8610.0}, "held": []}}
    step = 1e6 * (8579.6 / 8615.5 - 1.0)                                  # about -4170 ppm
    assert Z.low_state_step_ppm({"eyes": eyes}, "ZL063") == pytest.approx(step)
    assert Z.low_state_step_ppm({"eyes": {"ZL110": {"terms": {}}}}, "ZL110") is None

    def pool(eye, seq, dfx, dfy, state="backlash", sol=78):
        return {"eye": eye, "key": f"chal|s{sol}|{seq}", "block": "chal", "sequence": seq, "sol": sol, "states": [state],
                "bins": 2, "observations": 12000, "dfx_ppm": dfx, "dfy_ppm": dfy, "aspect_ppm": dfy - dfx}
    pools = [pool("ZL063", "ZCAM08038", -3450.0, -3760.0), pool("ZR063", "ZCAM08038", 410.0, 870.0),      # the case
             pool("ZL063", "ZCAM08040", -3800.0, -3900.0), pool("ZR063", "ZCAM08040", -3700.0, -3800.0),  # both low: not opposite
             pool("ZL063", "ZCAM08041", -3800.0, -1200.0), pool("ZR063", "ZCAM08041", 100.0, 50.0),       # fx, fy apart
             pool("ZL063", "ZCAM08042", 4300.0, 4200.0, state="regular"), pool("ZR063", "ZCAM08042", -200.0, 0.0)]   # the mirror
    res = {"eyes": eyes, "diagnostics": {"pool": "sequence", "pools": pools}}
    os_ = Z.opposite_state_flags(res)
    assert os_["sequences_checked"] == 4 and os_["source"] == "pooled"
    fl = {(f["sequence"], f["eye"]): f for f in os_["flags"]}
    assert set(fl) == {("ZCAM08038", "ZL063"), ("ZCAM08042", "ZL063")}
    f = fl[("ZCAM08038", "ZL063")]
    assert f["flag"] == "possible low state, eye ZL063" and f["other_eye"] == "ZR063" and f["other_dfx_ppm"] == 410.0
    assert f["step_ppm"] == pytest.approx(step) and f["labelled"] == ["backlash"] and f["sol"] == 78
    assert fl[("ZCAM08042", "ZL063")]["flag"] == "possible high state, eye ZL063"
    lines = Z.opposite_state_lines(os_)
    assert "2 flagged (WARNING" in lines[0] and lines[1].startswith("  chal ZCAM08038 sol 78: possible low state, eye ZL063 (labelled backlash)")
    chk = Z.joint_checks(dict(res, opposite_states=os_, zcam={"zcam_median_px": 0.3, "zcam_nonfinite_share": 0.0}), {"checks": []})
    c = next(x for x in chk["checks"] if x["check"] == "opposite backlash states between the eyes")
    assert c["ok"] and c["warning"] and "ZCAM08038 possible low state, eye ZL063" in c["note"] and Z.check_label(c) == "WARN"
    assert chk["ok"]                                                      # a warning does not fail the joint
    assert Z.check_label({"ok": True}) == "ok" and Z.check_label({"ok": False, "warning": True}) == "FAIL"
    # no pools (pool None): the per-bin means per sequence, weighted by observations
    bins = [{"block": "chal", "eye": "ZL063", "state": "backlash", "sequence": "ZCAM08038", "seq": "chal|s78|ZCAM08038", "sol": 78,
             "observations": 3000, "dfx_ppm": -3000.0, "dfy_ppm": -3300.0},
            {"block": "chal", "eye": "ZL063", "state": "backlash", "sequence": "ZCAM08038", "seq": "chal|s78|ZCAM08038", "sol": 78,
             "observations": 9000, "dfx_ppm": -3600.0, "dfy_ppm": -3900.0},
            {"block": "chal", "eye": "ZR063", "state": "backlash", "sequence": "ZCAM08038", "seq": "chal|s78|ZCAM08038", "sol": 78,
             "observations": 12000, "dfx_ppm": 410.0, "dfy_ppm": 870.0}]
    sc = Z.sequence_scales({"pool": None, "bins": bins})
    zl = next(s for s in sc if s["eye"] == "ZL063")
    assert zl["source"] == "per-bin mean" and zl["dfx_ppm"] == pytest.approx(-3450.0) and zl["bins"] == 2 and zl["observations"] == 12000
    os0 = Z.opposite_state_flags({"eyes": eyes, "diagnostics": {"pool": None, "bins": bins}})
    assert len(os0["flags"]) == 1 and os0["flags"][0]["source"] == "per-bin mean"
    assert Z.opposite_state_lines({"sequences_checked": 0, "flags": [], "steps_ppm": {}, "tolerance": 0.35})[0].endswith("none flagged")


def test_residual_directions_see_an_fy_scale_error(tmp_path):
    """0.96, item B (ZCAM08401 at 1.9-2.5 px rms in the 4 Nov joint, 0.17-0.25 px alone): per sequence and eye,
    from the final pass, the rms of x and y apart and the scale left in fx, fy.  On a synthetic block at its truth
    nothing is left; with one sequence's ZL fy made 2000 ppm too large the y residuals grow away from the centre
    and the row says the frames want fy 2000 ppm smaller, x untouched."""
    from mppp.calib import ztied as Z
    rng = np.random.default_rng(29)
    truth = _truth_model()
    so = {"ZCAM00001": 0.0, "ZCAM00002": 0.0, "ZCAM00003": 0.0}
    a, _, _ = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, seq_offsets=so)
    db = a.project.settings["database"]["cameras"]
    zunits = {}
    for key, spec in a.project.cameras.items():
        if not key.startswith("Z"):
            continue
        zunits[int(db[key])] = {"key": key, "group": spec["group"], "scape": "blk_a", "state": "backlash",
                                "sequence": spec["sequence"], "sequence_sol": 500, "seq": f"blk_a|s500|{spec['sequence']}"}
    rd = Z.residual_directions(a.rec, zunits, a.project.images)
    rows = {(r["eye"], r["sequence"]): r for r in rd["sequences"]}
    assert len(rows) == 6 and all(r["rms_px"] < 0.1 and r["rms_x_px"] < 0.1 and r["rms_y_px"] < 0.1 for r in rows.values())
    assert all(abs(r["fx_scale_left_ppm"]) < 300 and abs(r["fy_scale_left_ppm"]) < 300 for r in rows.values())
    r = rows[("ZL034", "ZCAM00002")]
    assert r["block"] == "blk_a" and r["sol"] == 500 and r["images"] == 6 and r["rows"] == []      # one elevation: no tiers
    assert r["elevation_span_deg"] < 1.0 and 12 < r["vertical_fov_deg"] < 16 and 6 < rd["row_gap_deg"]["ZL034"] < 8
    # ZL's ZCAM00002 cameras with fy 2000 ppm too large
    for key, spec in a.project.cameras.items():
        if key.startswith("ZL") and spec["sequence"] == "ZCAM00002":
            cam = a.rec.cameras[int(db[key])]
            p = np.asarray(cam.params, float).copy()
            p[1] *= 1.002
            cam.params = p
    rd2 = Z.residual_directions(a.rec, zunits, a.project.images)
    assert rd2["sequences"][0]["eye"] == "ZL034" and rd2["sequences"][0]["sequence"] == "ZCAM00002"   # the largest rms first
    r2 = rd2["sequences"][0]
    assert r2["rms_y_px"] > 3 * r2["rms_x_px"] and r2["rms_y_px"] > 0.4
    assert r2["fy_scale_left_ppm"] == pytest.approx(-2000.0, abs=150) and abs(r2["fx_scale_left_ppm"]) < 300
    assert r2["fy_scale_left_sd_ppm"] < 100
    lines = Z.residual_direction_lines({"residual_directions": rd2}, n=2)
    assert lines[0].startswith("residual direction per sequence") and len(lines) == 3
    assert lines[1].startswith("  blk_a ZCAM00002 ZL034 (6 frames): rms") and "scale left fx" in lines[1]
    assert int(lines[1].split("fy ")[-1].split(" ppm")[0]) == pytest.approx(-2000, abs=150)
    # a flagged sequence outside the top n is listed too
    flagged = {"block_aspect": {"sequences": [{"block": "blk_a", "sequence": "ZCAM00003"}]}}
    assert len(Z.residual_direction_lines(dict(flagged, residual_directions=rd2), n=1)) >= 3
    assert Z.residual_direction_lines({"residual_directions": {"error": "x"}}) == ["residual directions: not run (x)"]
    assert Z.residual_direction_lines({}) == []


def test_f0_uncertainty_from_the_sequence_scatter():
    """0.96, item D (Z48 of 4 Nov: f0 +-13.39 px, the same as each sequence offset's sd - the prior / sqrt(sequences),
    not the observed scatter): the scatter of the fitted offsets gives f0 an sd of its own; the larger is adopted."""
    from mppp.calib import ztied as Z
    f0 = 6696.0
    prior = Z.SEQ_OFFSET_SD * f0                                           # 33.5 px per offset
    e = {"terms": {"a_fx_backlash": f0}, "sd": {"a_fx_backlash": prior / np.sqrt(4)}}
    rows = [{"offset_px": o, "sd_px": prior * 0.98} for o in (-40.0, 10.0, 25.0, 45.0)]     # scatter about 36 px
    fs = Z.f0_uncertainty(e, rows)
    assert fs["free_sequences"] == 4 and fs["prior_sd_px"] == pytest.approx(prior / 2) and fs["reported_px"] == pytest.approx(prior / 2)
    sc = float(np.std([-40.0, 10.0, 25.0, 45.0], ddof=1))
    assert fs["sequence_scatter_px"] == pytest.approx(sc) and fs["from_scatter_px"] == pytest.approx(sc / 2)
    assert fs["adopted_px"] == pytest.approx(sc / 2) and "scatter" in fs["adopted_source"] and fs["prior_bound"]
    assert fs["adopted_ppm"] == pytest.approx(1e6 * sc / 2 / f0) and fs["sequence_scatter_ppm"] == pytest.approx(1e6 * sc / f0)
    # a tight scatter (Z34: 0.1-0.3 % within blocks) smaller than the covariance's: the covariance stays
    tight = [{"offset_px": o, "sd_px": 4.0} for o in (-6.0, 2.0, 3.0, 5.0)]
    ft = Z.f0_uncertainty(e, tight)
    assert ft["adopted_px"] == pytest.approx(prior / 2) and ft["adopted_source"].startswith("the covariance") and not ft["prior_bound"]
    one = Z.f0_uncertainty(e, tight[:1])
    assert one["sequence_scatter_px"] is None and one["adopted_px"] == pytest.approx(prior / 2) and one["free_sequences"] == 1
    none = Z.f0_uncertainty({"terms": {"a_fx_backlash": f0}, "sd": {}}, rows)
    assert none["reported_px"] is None and none["adopted_px"] == pytest.approx(sc / 2)
    # the report line and the candidate model carry it
    res = {"eyes": {"ZL048": dict(e, f0_sd=fs, held=[], reference_focus=600.0, bins=4, regular_bins=0, thin_bins=0,
                                  terms={"a_fx_backlash": f0, "a_fy_backlash": f0, "a_fx_regular": f0 / 1.0095, "a_fy_regular": f0 / 1.0095,
                                         "b": 0.05, "k1": -0.3, "k2": 0.4, "p1": 0.0, "p2": 0.0, "b_cy": 0.0, "d_cx_regular": 0.0,
                                         "d_cy_regular": 0.0})},
           "rig": {"change_mdeg": 0.0, "baseline_m": 0.2435}, "zcam": {}}
    line = next(x for x in Z.describe(res) if x.startswith("ZL048 f0 sd:"))
    assert "4 sequences, the offsets' sd at the prior" in line and "adopted" in line


def test_weak_sequences_fy_offsets_and_the_gate(tmp_path, monkeypatch):
    """0.96, item F (Christian, 9 Nov: the Z110 slope pulled shallow by fy of single-row horizon pans; ZCAM08038's
    2-row strip): sequences covering under 2 vertical fields of view are weak; in the joint their fy gets an offset
    of its own, so a +3000 ppm fy departure of one sequence lands in its offset (about the eye's mean) instead of
    pulling the line, and the gate leaves their pools out.  The flat synthetic blocks are all weak at factor 2 and
    none at factor 0.5."""
    from mppp.calib import forms
    from mppp.calib import ztied as Z
    monkeypatch.setitem(forms.ZCAM_FIXED_TERMS, 34, {})
    monkeypatch.setattr(Z, "MIN_BIN_OBSERVATIONS", 100)
    monkeypatch.setattr(Z, "WEAK_SEQUENCE_FOV", 2.0)
    rng = np.random.default_rng(31)
    truth = _truth_model()
    so = {"ZCAM00001": 0.0, "ZCAM00002": 0.0, "ZCAM00003": 0.0}
    sa = {"ZCAM00002": 3000.0}
    a, cams, rigT = _block(tmp_path, "blk_a", 0.0, truth, rng, 500, seq_offsets=so, seq_aspect=sa)
    b, _, _ = _block(tmp_path, "blk_b", 0.0, truth, rng, 900, seq_offsets=so, seq_aspect=sa)
    st = _navcam_law(cams, rigT)
    rec_, proj_, idx_, st_, units_, zunits_, sg_ = Z.build({}, copy.deepcopy(truth), None, 34, None, None,
                                                           scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st)
    w = sg_["weak_sequences"]
    assert len(w) == 12 and all(v["weak"] and v["rows"] == 1 and 0.9 < v["covered_fov"] < 1.2 and v["elevation_span_deg"] < 1.0
                                for v in w.values())
    assert not any(v["weak"] for v in Z.weak_sequences(rec_, zunits_, fov_factor=0.5).values())
    assert not any(v["weak"] for v in Z.weak_sequences(rec_, zunits_, fov_factor=0.0).values())
    res = Z.fit_zjoint({}, model=copy.deepcopy(truth), scapes=[copy.deepcopy(a), copy.deepcopy(b)], st=st,
                       nav_points=None, z_points=None, iterations=60, passes=2, verbose=False, k_tests=False)
    assert len(res["weak_sequences"]) == 12
    wf = {(r["eye"], r["block"], r["sequence"]): r for r in res["weak_fy"]}
    assert len(wf) == 12
    for eye in ("ZL034", "ZR034"):
        for blk in ("blk_a", "blk_b"):
            assert wf[(eye, blk, "ZCAM00002")]["ppm"] == pytest.approx(2000.0, abs=500), wf[(eye, blk, "ZCAM00002")]
            assert wf[(eye, blk, "ZCAM00001")]["ppm"] == pytest.approx(-1000.0, abs=500)
            assert wf[(eye, blk, "ZCAM00002")]["sd_px"] is not None and wf[(eye, blk, "ZCAM00002")]["z"] > 3
            assert wf[(eye, blk, "ZCAM00002")]["prior_sd_px"] == pytest.approx(0.01 * res["eyes"][eye]["terms"]["a_fy_backlash"], rel=0.01)
    # the law (with the offsets) now fits every sequence: the pools' aspect about it is small, and the gate leaves
    # the weak pools out (all are weak here, so it falls back to all of them and says so)
    assert all(p["weak"] and abs(p["aspect_ppm"]) < 700 for p in res["diagnostics"]["pools"])
    ba = res["block_aspect"]
    assert ba["weak_pools"] == 12 and all(r["weak_pools"] == 3 and r["pools"] == 0 for r in ba["rows"])
    assert all(b_["weak"] for b_ in res["diagnostics"]["bins"])
    ln = res["diagnostics"]["blocks"][0]["lines"]["ZL034"]["backlash"]
    assert ln["fy"]["weak_bins_left_out"] == 0 and ln["fx"]["weak_bins_left_out"] == 0    # (no strong bin: all kept)
    lines = Z.describe(res)
    assert any(x.startswith("weak-geometry sequences (under 2 vertical fields of view): 12 in the problem; 12 with a free fy offset") for x in lines)
    assert any("ZCAM00002" in x and "fy +" in x and "1 row(s)" in x for x in lines), lines
    # the focus-approach offset is not fitted (9 Nov)
    assert res["focus_below"] == [] or all(r.get("held") for r in res["focus_below"])
    out = Z.write_zjoint(res, tmp_path / "cand", model=truth)
    assert "weak-geometry sequences" in (tmp_path / "cand" / "report.md").read_text()
    assert not any("focus" in c["check"] and not c["ok"] for c in out["check"]["checks"])
