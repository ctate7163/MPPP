"""v0p70: fx, fy, cx, cy offsets of the early-mission Navcam images (before sol 380) in the consensus, with their
significance, written into the camera files and applied by the projects."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = np.array([2956.7, 2956.75, 2591.6, 1944.3, 0.0451, -0.0100, 2.5e-4, 3.6e-4, 0.0030, 0.0, 0.0, 0.0])


def test_early_keypoint_map_is_the_offset_camera():
    import pycolmap
    from mppp.sfm.navcal_consensus import early_keypoint_map
    off = np.array([0.4, -0.3, -2.8, 1.1])
    shared = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840, params=P)
    early_cam = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=5120, height=3840,
                                params=np.r_[P[:4] + off, P[4:]])
    X = np.random.default_rng(0).uniform([-1.2, -0.9, 1.0], [1.2, 0.9, 1.0], (50, 3))
    u_early = np.asarray(early_cam.img_from_cam(X))                     # what an early image records
    proj = SimpleNamespace(images=[{"image_id": 1, "sol": 120}, {"image_id": 2, "sol": 500}, {"image_id": 3}])
    m = early_keypoint_map(proj, 380, off)
    assert m.early_images == {1}
    assert np.allclose(m(1, u_early, shared), np.asarray(shared.img_from_cam(X)), atol=1e-9)
    assert np.array_equal(m(2, u_early, shared), u_early) and np.array_equal(m(3, u_early, shared), u_early)


def test_fit_quadratic_recovers_the_surface():
    from mppp.sfm.navcal_consensus import _fit_quadratic, _quadratic_design
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    H = A @ A.T + 4 * np.eye(4)
    g = rng.normal(size=4)
    D = np.array(_quadratic_design(4, 0.5))
    assert len(D) == 15
    c = 7.0 + D @ g + 0.5 * np.einsum("ij,jk,ik->i", D, H, D)
    a, g1, H1, rms = _fit_quadratic(D, c)
    assert a == pytest.approx(7.0) and np.allclose(g1, g) and np.allclose(H1, H) and rms < 1e-9


def test_fit_early_offsets_finds_the_minimum_and_significance(monkeypatch):
    from mppp.sfm import navcal as NC
    from mppp.sfm import navcal_consensus as NCC
    truth = np.array([0.02, -0.05, -2.7, 0.4])
    Hc = np.diag([40.0, 40.0, 900.0, 900.0]) + 5.0               # cost curvature (cost = chi2 / 2)
    nobs = 1_000_000
    calls = []

    def fake_joint(rec, proj, temps, ppm, T0, max_iterations=50, rig_slopes=None, extra_map=None, **kw):
        x = extra_map.offsets
        calls.append(x.copy())
        d = x - truth
        cost = 0.5 * nobs + 0.5 * d @ Hc @ d                       # variance factor ~ 0.5
        return {"final_cost": float(cost), "observations": nobs, "iterations": 3}
    monkeypatch.setattr(NC, "joint_adjust", fake_joint)
    proj = SimpleNamespace(images=[{"image_id": i, "sol": s} for i, s in enumerate([60, 200, 370, 500, 900], 1)])
    out = NCC.fit_early_offsets(object(), proj, {}, 38.0, -20.0, {}, 380, verbose=False)
    assert out["early_images"] == 3
    assert np.allclose([out["offsets_px"][k] for k in NCC.EARLY_TERMS], truth, atol=1e-6)
    vf = 2 * (0.5 * nobs) / (2 * nobs - 1)
    sd_expected = np.sqrt(np.diag(vf * np.linalg.inv(Hc)))
    assert np.allclose([out["sd_px"][k] for k in NCC.EARLY_TERMS], sd_expected, rtol=1e-3)
    assert abs(out["z"]["cx"]) > 50 and abs(out["z"]["fx"]) < 3
    assert "cx" in out["applied_terms"] and "fx" not in out["applied_terms"]
    assert out["p_value_all"] < 1e-10 and out["cost_at_zero"] > out["cost_at_best"]
    assert len(calls) <= 4 * 16 and out["converged"]
    # nothing before the sol: nothing fitted
    none = NCC.fit_early_offsets(object(), proj, {}, 38.0, -20.0, {}, 10, verbose=False)
    assert none["early_images"] == 0 and "offsets_px" not in none


def test_joint_adjust_composes_the_extra_map(monkeypatch):
    from mppp.sfm import navcal as NC
    import mppp.sfm.reconstruction as R
    seen = {}

    def fake_ba(rec, proj, **kw):
        seen["map"] = kw.get("keypoint_map")
        return {"final_cost": 1.0, "observations": 1, "iterations": 1}
    monkeypatch.setattr(R, "bundle_adjust", fake_ba)
    proj = SimpleNamespace(images=[{"image_id": 1, "name": "a", "sol": 100}])
    extra = lambda iid, k, cam: k + 1.0                                # noqa: E731
    NC.joint_adjust(None, proj, {}, 0.0, -20.0, pp_slopes={1: (0.0, 0.0)}, extra_map=extra)
    cam = SimpleNamespace(camera_id=1, params=P)
    assert np.allclose(seen["map"](1, np.zeros((2, 2)), cam), 1.0)
    NC.joint_adjust(None, proj, {}, 0.0, -20.0, extra_map=extra)
    assert seen["map"] is extra


def test_write_consensus_with_early_mission(tmp_path):
    from test_v0p62 import _consensus_result
    from mppp.sfm.navcal_consensus import write_consensus
    res = _consensus_result()
    res["early_mission"] = {"before_sol": 380.0, "terms": ["fx", "fy", "cx", "cy"], "early_images": 900,
                            "offsets_px": {"fx": 0.05, "fy": -0.02, "cx": -2.7, "cy": 0.3},
                            "sd_px": {"fx": 0.1, "fy": 0.1, "cx": 0.05, "cy": 0.05},
                            "z": {"fx": 0.5, "fy": -0.2, "cx": -54.0, "cy": 6.0},
                            "p_value": {"fx": 0.6, "fy": 0.8, "cx": 0.0, "cy": 2e-9}, "p_value_all": 0.0,
                            "applied_terms": ["cx", "cy"]}
    d = write_consensus(res, tmp_path / "nj")
    em = json.loads((d / "M2020_NR_fisheye_tangential.json").read_text())["early_mission"]
    assert em["dcx_px"] == -2.7 and em["applied_terms"] == ["cx", "cy"] and em["before_sol"] == 380.0


def test_project_applies_the_early_mission_offsets(tmp_path):
    from conftest import NLF
    if not NLF.is_file():
        pytest.skip("example IMGs not present")
    import mppp
    from mppp.sfm.navcal import write_joint_cameras
    from mppp.sfm.project import SfmProject
    out = tmp_path / "proc"
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": False}})
    man = mppp.process_images([NLF], out, cfg, mppp.load_waypoints(), progress=False)
    names = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
    cam = {"model": "THIN_PRISM_FISHEYE", "width": 5120, "height": 3840, "params": dict(zip(names, P))}
    joint = {"NL": cam, "NR": cam, "T0_degC": -19.1, "ppm_per_degC": 0.0, "final": {"observations": 1}, "blocks": {},
             "rig_R": np.eye(3).tolist(), "rig_t": [-0.424, 0, 0]}
    write_joint_cameras(joint, tmp_path / "joint")
    files = sorted((tmp_path / "joint").glob("M2020_N?_fisheye_tangential.json"))
    assert len(files) == 2
    for f in files:
        js = json.loads(f.read_text())
        js["early_mission"] = {"before_sol": 1e6, "dfx_px": 0.3, "dfy_px": 0.0, "dcx_px": -2.7, "dcy_px": 0.5,
                               "applied_terms": ["cx", "cy"]}
        f.write_text(json.dumps(js))
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, navcam_distortion="fisheye_tangential",
                             navcam_cameras=tmp_path / "joint")
    p = proj.cameras["NL"]["params"]
    assert p[0] == pytest.approx(P[0], abs=0.05) and p[2] == pytest.approx(P[2] - 2.7, abs=0.05)
    assert p[3] == pytest.approx(P[3] + 0.5, abs=0.05) and "early-mission" in proj.cameras["NL"]["source"]


def test_notebook_and_cli_v0p70():
    nb4 = json.loads((ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8"))
    src4 = "".join("".join(c["source"]) for c in nb4["cells"] if c["cell_type"] == "code")
    assert "EARLY_MISSION_SOL = 380.0" in src4 and "--early-mission-sol" in src4
    s = (ROOT / "scripts" / "navcam_calibration_study.py").read_text(encoding="utf-8")
    assert "--early-mission-sol" in s and "early_mission_sol=" in s


def test_early_start_from_blocks(tmp_path):
    import struct
    from mppp.sfm.navcal_consensus import _read_cameras_bin, early_start_from_blocks
    from mppp.sfm.project import SfmProject

    def block(name, sols, d):
        root = tmp_path / name / "colmap"
        (root / "sparse" / "cahv_ba").mkdir(parents=True)
        cams = {e: {"model": "THIN_PRISM_FISHEYE", "width": 5120, "height": 3840, "params": P.tolist()} for e in ("NL", "NR")}
        imgs = [{"name": f"{name}_{i}.png", "instrument": "NL", "sol": s, "station": "S", "image_id": i + 1}
                for i, s in enumerate(sols)]
        p = SfmProject(root, imgs, cams, {}, [0, 0, 0], {"database": {"cameras": {"NL": 1, "NR": 2}}})
        p.save()
        with open(root / "sparse" / "cahv_ba" / "cameras.bin", "wb") as fh:
            fh.write(struct.pack("<Q", 2))
            for cid in (1, 2):
                fh.write(struct.pack("<Ii", cid, 10) + struct.pack("<QQ", 5120, 3840))
                fh.write(struct.pack("<12d", *(np.r_[P[:4] + d, P[4:]])))
        return tmp_path / name
    cfg = {"a": block("a", [50, 70], np.array([0.1, 0.0, -3.0, 0.5])),
           "b": block("b", [200, 300], np.array([0.0, 0.0, -2.6, 0.3])),
           "c": block("c", [500, 600], np.array([0.0, 0.1, -0.4, 0.0])),
           "d": block("d", [300, 500], np.array([9.0, 9.0, 9.0, 9.0]))}            # spans the sol: left out
    assert _read_cameras_bin(cfg["a"] / "colmap" / "sparse" / "cahv_ba" / "cameras.bin")[2][2] == pytest.approx(P[2] - 3.0)
    out = early_start_from_blocks(cfg, 380)
    assert out["early_blocks"] == 2 and out["late_blocks"] == 1
    assert np.allclose(out["start"], [0.05, -0.1, -2.4, 0.4])


def test_fit_early_offsets_recovers_a_synthetic_offset(tmp_path):
    """The real joint adjustment: a block whose first station's images were taken with cx -3, cy +1 px."""
    pytest.importorskip("pyceres")
    from helpers import _build_rec, _synthetic
    from mppp.sfm import navcal_consensus as NCC
    proj, truth, Pts, cams, rigT, noise, rng = _synthetic(tmp_path, noise_native_px=0.2)
    rec, _ = _build_rec(proj, truth, Pts, cams, rigT, noise, rng, perturb=False)
    off = np.array([0.0, 0.0, -3.0, 1.0])
    for r in proj.images:
        r["sol"] = 100 if r["station"] == "S001D0000" else 500
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    for r in proj.images:
        if r["sol"] < 380:
            im = rec.images[r["image_id"]]
            p = np.asarray(rec.cameras[im.camera_id].params)
            f, c = p[:2], p[2:4]
            for k in range(im.num_points2D()):
                u = np.asarray(im.points2D[k].xy)
                im.points2D[k].xy = c + (f + off[:2]) * (u - c) / f + off[2:4]
    out = NCC.fit_early_offsets(rec, proj, {}, 0.0, -20.0, {}, 380, start=(0.0, 0.0, -2.5, 0.8), verbose=False)
    x = np.array([out["offsets_px"][k] for k in NCC.EARLY_TERMS])
    sd = np.array([out["sd_px"][k] for k in NCC.EARLY_TERMS])
    assert out["converged"] and np.all(np.abs(x - off) < np.maximum(4 * sd, 0.05))
    assert set(out["applied_terms"]) >= {"cx", "cy"}


def test_notebooks_carry_the_version_guard():
    import glob
    import mppp
    for p in sorted(glob.glob(str(ROOT / "notebooks" / "0*.ipynb"))):
        nb = json.loads(Path(p).read_text(encoding="utf-8"))
        src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        if "import mppp\n" in src:
            assert f'NOTEBOOK_FOR_MPPP = "{mppp.VERSION_TAG}"' in src, p
