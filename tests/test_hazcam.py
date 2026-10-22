"""Front and rear Hazcam support (0.93): the label CAHVORE fit, the start cameras, the settings and selection, and the
staged alignment that brings the Hazcam frames in after the Navcam block."""
import copy
import json

import numpy as np
import pytest

from mppp.calib import labelfit

# the label models of FLF / RLF_0461 (sol 461, S026D0630): FLF is a full-resolution sub-frame from detector line 961,
# RLF a quarter-resolution full frame with entrance-pupil terms of several mm (a non-central CAHVORE)
FLF_0461 = {"MODEL_TYPE": "CAHVORE",
            "MODEL_COMPONENT_1": [1.10428, -0.00917128, -0.728844], "MODEL_COMPONENT_2": [0.884826, -0.171816, 0.433084],
            "MODEL_COMPONENT_3": [2605.54, 1674.2, 1273.77], "MODEL_COMPONENT_4": [-101.773, -161.86, 2340.33],
            "MODEL_COMPONENT_5": [0.885207, -0.171185, 0.432556], "MODEL_COMPONENT_6": [8.1e-07, 0.00934156, -0.00640682],
            "MODEL_COMPONENT_7": [1.88e-07, 3.14e-07, 4.02e-07], "MODEL_COMPONENT_8": 2.0, "MODEL_COMPONENT_9": 0.0}
RLF_0461 = {"MODEL_TYPE": "CAHVORE",
            "MODEL_COMPONENT_1": [-1.04711, 0.466977, -0.730608], "MODEL_COMPONENT_2": [-0.703458, 0.00398837, 0.710725],
            "MODEL_COMPONENT_3": [-454.574, -533.591, 456.502], "MODEL_COMPONENT_4": [43.2422, 0.681172, 723.081],
            "MODEL_COMPONENT_5": [-0.700545, 0.00407406, 0.713597], "MODEL_COMPONENT_6": [1.668e-05, 0.0220681, -0.0108751],
            "MODEL_COMPONENT_7": [0.0032911, 0.00694844, 0.00616767], "MODEL_COMPONENT_8": 2.0, "MODEL_COMPONENT_9": 0.0}


def test_numpy_thin_prism_fisheye_is_colmaps():
    pytest.importorskip("pycolmap")
    from mppp.colmap import project_camera
    rng = np.random.default_rng(0)
    X = np.c_[rng.uniform(-3, 3, (300, 2)), rng.uniform(0.05, 2, 300)]
    p = [2147, 2146, 2569, 1911, 0.009, -0.006, 1e-4, -2e-4, 1e-3, -1e-4, 0, 0]
    assert np.nanmax(np.abs(labelfit.project_thin_prism_fisheye(p, X) - project_camera("THIN_PRISM_FISHEYE", p, X))) < 1e-8


def test_front_hazcam_label_fit_beats_the_closed_form():
    p, rep = labelfit.label_fisheye(FLF_0461, 1.0, 0.0, 960.0, "F")
    assert rep["central"] and rep["depths_m"] == [labelfit.CENTRAL_DEPTH_M]
    assert rep["rms_px"] < 0.3 and rep["max_px"] < 1.0                     # 0.13 / 0.42 px
    assert rep["closed_form_rms_px"] > 1.0                                  # 1.37 px: the O tilt
    assert len(p) == 12 and p[10] == p[11] == 0.0                           # sx1 = sy1 = 0 (the form)
    assert abs(p[0] - 2147.6) < 1 and abs(p[3] - 1911.8) < 1               # cy in full-frame pixels (sub-frame + 960)
    assert 0.005 < p[4] < 0.015                                            # k1 on the fisheye theta polynomial


def test_rear_hazcam_fit_is_non_central_over_the_terrain_ranges():
    p, rep = labelfit.label_fisheye(RLF_0461, 0.25, 0.0, 0.0, "R")
    assert not rep["central"] and rep["depths_m"] == list(labelfit.NONCENTRAL_DEPTHS_M)
    assert rep["E_max_m"] > 1e-3
    assert rep["rms_px"] < 4.0 and rep["rms_px"] < rep["closed_form_rms_px"]
    assert abs(p[0] - 2144.6) < 3 and abs(p[2] - 2570.4) < 3              # quarter-resolution label -> full frame
    # cached: the same model and placement gives the same answer at once
    assert labelfit.label_fisheye(RLF_0461, 0.25, 0.0, 0.0, "R")[0] == p


def test_generic_hazcam_start_model_is_a_fisheye():
    """0.92's start model (f 2130, k1 -0.25, k2 +0.08) was 330 px rms from the label; the 0.93 one is close."""
    pytest.importorskip("pycolmap")
    from mppp import calib
    from mppp.cmod import PixelCamera, compare_cameras
    m = calib.hazcam_start_model("F", "L")
    assert m["params"][4] > 0 and m["params"][5] < 0 and "generic" in m["source"]
    cm = labelfit.full_frame_model(FLF_0461, 1.0, 0.0, 960.0, "F")
    r = compare_cameras(PixelCamera.cahv(cm, 5120, 3840), PixelCamera.colmap(m["model"], m["params"], 5120, 3840),
                        step=256, fit_rotation=True)
    assert r["rms_px"] < 40


def test_label_camera_uses_the_fit():
    from mppp import calib
    from mppp.cmod import CameraModel
    cm = CameraModel.from_label(FLF_0461)
    R, lin = cm.decompose()
    K = [[lin["hs"], 0, lin["hc"]], [0, lin["vs"], lin["vc"] + 960.0], [0, 0, 1]]     # padded to the full frame
    meta = {"intrinsics": {"K": K, "width": 5120, "height": 3840, "dist_opencv": {}},
            "filename": {"family": "F"}, "camera_model_label": FLF_0461}
    name, p, src = calib.label_camera(meta)
    want, _ = labelfit.label_fisheye(FLF_0461, 1.0, 0.0, 960.0, "F")
    assert name == "THIN_PRISM_FISHEYE" and "fit" in src
    assert np.allclose(p[:4], want[:4], atol=0.05) and np.allclose(p[4:], want[4:], atol=2e-4)


def _haz_meta(eye="L", fam="F", label=FLF_0461, scale=1.0, top=960):
    return {"source_product": f"{fam}{eye}F_0461_0707870635_633RAD_N0260630{fam}HAZ02008_0A0085J02.IMG",
            "filename": {"family": fam, "instrument": f"{fam}{eye}", "camera_group": f"{fam}{eye}0", "eye": eye,
                         "downsample_scale": scale, "camera_code": f"{fam}{eye}F", "stem": "x", "sol": 461},
            "padding": {"left": 0, "top": top}, "camera_model_label": label}


def test_block_camera_and_hazcam_start_camera():
    from mppp.sfm.start_cameras import HAZCAM_INTRINSICS, hazcam_start_camera
    metas = [_haz_meta(), _haz_meta()]
    cam, info = labelfit.block_camera("FL", metas)
    assert cam["model"] == "THIN_PRISM_FISHEYE" and (cam["width"], cam["height"]) == (5120, 3840)
    assert info["label_fit"]["images"] == 2 and info["calibrated"] is False and "label CAHVORE fit" in cam["source"]
    for mode, free in HAZCAM_INTRINSICS.items():
        c, i = hazcam_start_camera("FL", [], metas, intrinsics=mode)
        assert set(c["fixed_params"]) == {n for n in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1",
                                                       "sy1") if n not in free}
        assert c["hazcam_intrinsics"] == mode and i["refined"] == list(free)
    with pytest.raises(ValueError):
        hazcam_start_camera("FL", [], metas, intrinsics="guess")
    with pytest.raises(ValueError):
        hazcam_start_camera("NL", [], metas)
    with pytest.raises(ValueError):
        labelfit.block_camera("FL", [{"source_product": "FLF_x", "filename": {}}])


def test_hazcam_settings_keep_the_other_run_keys():
    from mppp.pipeline import HAZCAM_FIELDS, Settings, hazcam_families, run_key
    s = Settings(pds_dir=None, scapes_root=None, gpu_python=None)
    assert s.hazcams == () and s.variant == ""
    assert not set(HAZCAM_FIELDS) & set(s.key_dict())                      # the keys of every other run stay
    h = s.replace(hazcams="both")
    assert h.hazcams == ("F", "R") and h.variant == "hazcam" and h.replace(variant="").variant == "hazcam"
    assert set(HAZCAM_FIELDS) <= set(h.key_dict()) and run_key(h) != run_key(s)
    assert s.replace(hazcams="rear", variant="x").variant == "x"
    for v, want in (("F", ("F",)), ("front", ("F",)), ("FR", ("F", "R")), (["rear", "front"], ("F", "R")),
                    ("F,R", ("F", "R")), (None, ()), ("none", ()), ([], ())):
        assert hazcam_families(v) == want
    with pytest.raises(ValueError):
        hazcam_families("navcam")


def test_load_images_leaves_out_hazcams_not_asked_for(tmp_path):
    from mppp.pipeline import Settings, load_images

    def meta(code, k):
        return {"source_product": f"{code}_0461_07078706{k:02d}_633RAD_N0260630FHAZ02008_0A0085J02.IMG",
                "filename": {"family": code[0], "camera_code": code, "sol": 461, "site": 26, "drive": 630,
                             "sequence": "FHAZ02008", "zoom_mm": None, "stem": f"{code}_{k}"},
                "sol": 461, "site": 26, "drive": 630, "mask": {"included_fraction": 0.7}}
    ims = [meta(c, k) for k, c in enumerate(("NLF", "NRF", "FLF", "FRF", "RLF", "RRF"))]
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "mppp_manifest_v0p93.json").write_text(json.dumps({"images": ims}))
    base = Settings(exclude_list=False)
    fam = lambda s: sorted(m["filename"]["family"] for m in load_images(tmp_path, s))     # noqa: E731
    assert fam(base) == ["N", "N"]
    assert fam(base.replace(hazcams="front")) == ["F", "F", "N", "N"]
    assert fam(base.replace(hazcams="both")) == ["F", "F", "N", "N", "R", "R"]


def test_hazcams_in_scope_and_cli():
    from mppp.cli import _settings, build_parser
    from mppp.process import SCOPE_CAMERA_CODES as P
    from mppp.sfm.pairs import CAMERA_HEIGHT_BY_FAMILY_M
    from mppp.sfm.project import SCOPE_CAMERA_CODES as S
    assert {"FLF", "FRF", "RLF", "RRF"} <= set(P) and {"FLF", "FRF", "RLF", "RRF"} <= set(S)
    assert CAMERA_HEIGHT_BY_FAMILY_M["F"] < 1.0 and CAMERA_HEIGHT_BY_FAMILY_M["R"] < 1.0
    a = build_parser().parse_args(["run", "rockytop", "--hazcam", "rear"])
    s = _settings(a, "rockytop")
    assert s.hazcams == ("R",) and s.variant == "hazcam"


def _hazcam_block(tmp_path, monkeypatch, rig_refine="hold", intrinsics="radial"):
    """The synthetic two-station block with the second station's frames moved to a "front Hazcam" rig (cameras FL,
    FR, rig 2), perturbed cameras, and a triangulate that tears the block down as COLMAP's does."""
    pytest.importorskip("pyceres")
    pycolmap = pytest.importorskip("pycolmap")
    from helpers import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    fids = sorted(rec0.frames)
    hf = set(fids[len(fids) // 2:])
    new = pycolmap.Reconstruction()
    for cid, c in rec0.cameras.items():
        new.add_camera(c)
        q = np.array(c.params, float)
        q[0] *= 1.002
        q[1] *= 1.002
        new.add_camera(pycolmap.Camera(camera_id=cid + 2, model=c.model, width=c.width, height=c.height, params=q))
    new.add_rig(rec0.rigs[1])
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    r2 = pycolmap.Rig(rig_id=2)
    r2.add_ref_sensor(S(3))
    r2.add_sensor(S(4), rec0.rigs[1].sensor_from_rig(S(2)))
    new.add_rig(r2)
    for fid in fids:
        fr = rec0.frames[fid]
        h = fid in hf
        nf = pycolmap.Frame(frame_id=fid, rig_id=2 if h else 1)
        for d in fr.data_ids:
            nf.add_data_id(pycolmap.data_t(sensor_id=S(d.sensor_id.id + (2 if h else 0)), id=d.id))
        nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
        for d in fr.data_ids:
            im = rec0.images[d.id]
            ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]),
                                camera_id=im.camera_id + (2 if h else 0), image_id=d.id)
            ni.frame_id = fid
            new.add_image(ni)
        new.register_frame(fid)
    for pid, p in rec0.points3D.items():
        new.add_point3D(p.xyz, p.track, np.zeros(3, np.uint8))
    himg = {d.id for f in hf for d in rec0.frames[f].data_ids}
    for r in proj.images:
        if r["image_id"] in himg:
            r["instrument"] = "F" + r["instrument"][1:]
            r["camera_group"] = r["instrument"]
    from mppp.sfm.start_cameras import HAZCAM_INTRINSICS
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    for k in ("NL", "NR"):
        proj.cameras["F" + k[1:]] = dict(proj.cameras[k], hazcam_intrinsics=intrinsics,
                                         fixed_params=[n for n in names if n not in HAZCAM_INTRINSICS[intrinsics]])
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2, "FL": 3, "FR": 4}}
    proj.settings["hazcam"] = {"intrinsics": intrinsics, "rig_refine": rig_refine, "cameras": ["FL", "FR"]}

    def triangulate_again(rec, project, **kw):
        # COLMAP's triangulate_points tears the block down and builds the tracks of the registered images again
        rec.tear_down()
        reg = set(int(i) for i in rec.reg_image_ids())
        for pid in list(rec.points3D):
            rec.delete_point3D(pid)
        for p in new.points3D.values():
            els = [(int(el.image_id), int(el.point2D_idx)) for el in p.track.elements if int(el.image_id) in reg]
            if len(els) < 2:
                continue
            tr = pycolmap.Track()
            for i, k in els:
                tr.add_element(i, k)
            pid = rec.add_point3D(p.xyz, tr, np.zeros(3, np.uint8))
            for i, k in els:
                if not rec.images[i].points2D[k].has_point3D():
                    rec.images[i].set_point3D_for_point2D(k, pid)
        return rec

    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    monkeypatch.setattr(R, "triangulate", triangulate_again)
    return proj, new, hf, R


@pytest.mark.parametrize("rig_refine", ["hold", "rotation"])
def test_staged_reconstruct_brings_the_hazcams_in_after_the_navcam(tmp_path, monkeypatch, rig_refine):
    pycolmap = pytest.importorskip("pycolmap")
    proj, start, hf, R = _hazcam_block(tmp_path, monkeypatch, rig_refine)
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    rec = R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0), (8.0, 2.0, 4.0)), max_iterations=8,
                        verbose=False, navcam_intrinsics="refine", staged=True, out_name="cahv_ba")
    rs = proj.settings["reconstruction"]
    assert rs["staged"] is True
    nav = pycolmap.Reconstruction(str(proj.root / "sparse" / "cahv_ba_navcam"))
    assert not ({d.id for f in hf for d in start.frames[f].data_ids} & set(nav.reg_image_ids()))   # stage 1: Navcam
    assert set(rec.reg_frame_ids()) >= hf                                                         # stage 2: back
    for cid in (1, 2):                                                     # stage 2 held the Navcam cameras
        assert np.allclose(np.array(rec.cameras[cid].params), np.array(nav.cameras[cid].params))
    for cid in (3, 4):                                                     # "radial": cx, cy held, f refined
        p0, p1 = np.array(start.cameras[cid].params), np.array(rec.cameras[cid].params)
        assert np.allclose(p1[2:4], p0[2:4]) and not np.isclose(p1[0], p0[0])
    T0 = start.rigs[2].sensor_from_rig(S(4))
    T1 = rec.rigs[2].sensor_from_rig(S(4))
    moved = not np.allclose(T0.rotation.quat, T1.rotation.quat, atol=1e-12)
    assert moved == (rig_refine == "rotation")
    assert np.allclose(T0.translation, T1.translation)                     # the baseline is the CAHV one
    cov = rs["hazcam"]
    assert set(cov["eyes"]) == {"FL", "FR"} and cov["eyes"]["FL"]["with_points"] == len(hf)
    assert cov["navcam_hazcam_points"] > 0
    assert any(e.get("hazcam_coverage") for e in rs["log"] if isinstance(e, dict))


def test_hazcam_intrinsics_hold_holds_the_cameras(tmp_path, monkeypatch):
    proj, start, hf, R = _hazcam_block(tmp_path, monkeypatch, intrinsics="hold")
    rec = R.reconstruct(proj, sigma_px=0.3, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="refine", staged=True, out_name="cahv_ba")
    assert {"FL", "FR"} <= set(proj.settings["reconstruction"]["hold_cameras"])
    for cid in (3, 4):
        assert np.allclose(np.array(rec.cameras[cid].params), np.array(start.cameras[cid].params))


def test_hazcam_station_priors_follow_the_navcam_solution():
    from mppp.sfm import reconstruction as R

    class Im:
        def __init__(self, name, C):
            self.name, self._C = name, np.asarray(C, float)

        def projection_center(self):
            return self._C

    class Rec:
        def __init__(self, ims):
            self.images = dict(enumerate(ims, start=1))

        def reg_image_ids(self):
            return list(self.images)

    class P:
        pass
    proj = P()
    proj.images = [{"name": "n1", "instrument": "NL", "station": "A", "prior_C": [0, 0, 0]},
                   {"name": "n2", "instrument": "NR", "station": "A", "prior_C": [0.4, 0, 0]},
                   {"name": "f1", "instrument": "FL", "station": "A", "prior_C": [1, 0, -1]},
                   {"name": "r1", "instrument": "RL", "station": "B", "prior_C": [5, 5, -1]}]
    rec = Rec([Im("n1", [0.1, 0.2, 0]), Im("n2", [0.5, 0.2, 0])])
    out = R.hazcam_station_priors(rec, proj)
    assert set(out) == {"A"} and np.allclose(out["A"], [0.1, 0.2, 0])
    f1, r1 = proj.images[2], proj.images[3]
    assert np.allclose(f1["prior_C"], [1.1, 0.2, -1]) and f1["prior_C_label"] == [1, 0, -1]
    assert f1["prior_sigma_m"] == R.HAZCAM_STATION_PRIOR_SIGMA_M
    assert "prior_sigma_m" not in r1 and r1["prior_C"] == [5, 5, -1]          # no Navcam at station B
    R.hazcam_station_priors(rec, proj)                                       # again: from the label prior
    assert np.allclose(f1["prior_C"], [1.1, 0.2, -1])
    cov = {"eyes": {"FL": {"observations": 5000}, "RL": {"observations": 200}}}
    assert R.hazcam_weak_eyes(cov) == ["RL"] and R.hazcam_weak_eyes({}) == []
