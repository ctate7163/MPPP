"""v0p13: package data and cache paths, one version source, safetensors models and the registry,
merged COLMAP helpers, database overwrite, two-view tie points, alignment health."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT


# ------------------------------------------------------------------ packaging
def test_single_version_source():
    import mppp
    assert mppp.VERSION_TAG == "v%sp%s" % tuple(mppp.__version__.split(".")[:2])
    txt = (ROOT / "pyproject.toml").read_text()
    import re
    project = txt.split("[project]")[1].split("\n[")[0]
    assert 'dynamic = ["version"]' in project and not re.search(r"^version\s*=", project, re.M)
    assert 'version = { attr = "mppp.__version__" }' in txt


def test_package_data_and_cache(tmp_path, monkeypatch):
    from mppp import paths
    d = paths.data_dir()
    for f in ("M20_waypoints.json", "M2020_taus_versus_L_s.csv", "M2020_occlusion_profiles.csv", "models.json",
              "m20_cmods/M2020_NL0_frame.xml", "m20_cmods/ZL034_frame.xml"):
        assert (d / f).is_file(), f
    assert paths.params_dir() == d and paths.source_root() == ROOT
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "c"))
    assert paths.cache_dir() == tmp_path / "c" and (tmp_path / "c").is_dir()
    monkeypatch.delenv("MPPP_CHECKPOINTS", raising=False)
    assert paths.checkpoints_dir() == ROOT / "checkpoints"                   # source checkout
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(tmp_path / "k"))
    assert paths.checkpoints_dir() == tmp_path / "k"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"data/*.json", "data/*.csv", "data/m20_cmods/*.xml"' in pyproject


def test_waypoints_snapshot_cache_and_refresh(tmp_path, monkeypatch):
    from mppp import waypoints as W
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path))
    wp = W.load_waypoints()
    assert wp["_mppp_source"]["packaged_snapshot"] and wp["_mppp_source"]["path"] == str(W.snapshot_path())
    src = tmp_path / "remote.json"
    src.write_text(json.dumps({"type": "FeatureCollection", "features": [{"properties": {"site": 1, "drive": 0}}]}))
    wp = W.load_waypoints(refresh=True, url=src.as_uri())
    assert wp["_mppp_source"]["path"] == str(tmp_path / "M20_waypoints.json") and wp["_mppp_source"]["n_features"] == 1
    assert W.load_waypoints()["_mppp_source"]["n_features"] == 1                # the cache now wins
    with pytest.warns(UserWarning, match="could not download"):
        assert W.load_waypoints(refresh=True, url=(tmp_path / "missing.json").as_uri())["_mppp_source"]["n_features"] == 1
    from mppp.error.waypoints import load_featurecollection                     # one loader for both
    assert len(load_featurecollection()["features"]) == 1


def test_legacy_and_studies_are_outside_the_package():
    pkg = ROOT / "src" / "mppp"
    assert not (pkg / "error" / "compat.py").exists() and (ROOT / "src/legacy/error_compat.py").is_file()
    assert not (pkg / "error" / "study_navcam.py").exists() and (ROOT / "studies/error/study_navcam.py").is_file()
    assert not (pkg / "error" / "data").exists()
    for f in ("image.py", "readers.py", "writers.py", "config.json"):
        assert (ROOT / "src/legacy" / f).is_file() and not (ROOT / "src" / f).exists()


# -------------------------------------------------------------- models / hub
class NotATensor:                                                              # any pickled Python object
    pass


def _tiny_checkpoint(tmp_path):
    import torch
    from mppp.mask.model import ConvNeXtSeg, write_card
    m = ConvNeXtSeg("convnext_tiny", pretrained=False, fpn_width=32, stride4=True)
    ck = tmp_path / "tiny_s4_seg_test.pt"
    torch.save({"model": m.state_dict(), "val_iou": 0.5, "epoch": 1}, ck)
    write_card(ck, backbone="convnext_tiny", fpn_width=32, stride4=True, threshold=0.5, canvas=[64, 64],
               input_size=64, val_iou=0.5, name=ck.stem)
    return ck


def test_safetensors_export_is_deterministic_and_equivalent(tmp_path):
    import torch
    from mppp.mask.hub import export_safetensors
    from mppp.mask.model import load_model, read_card
    ck = _tiny_checkpoint(tmp_path)
    a = export_safetensors(ck, out=tmp_path / "a.safetensors")
    b = export_safetensors(ck, out=tmp_path / "b.safetensors")
    assert a["sha256"] == b["sha256"] and a["bytes"] > 0
    card = read_card(tmp_path / "a.safetensors")                             # embedded, no sidecar needed
    assert card["fpn_width"] == 32 and card["exported_from"] == ck.name and not (tmp_path / "a.json").exists()
    m1, _ = load_model(ck)
    m2, _ = load_model(tmp_path / "a.safetensors")
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x))


def test_pt_is_loaded_without_pickle_code(tmp_path):
    import torch
    from mppp.mask.model import read_state_dict_checkpoint

    bad = tmp_path / "bad.pt"
    torch.save({"model": {}, "x": NotATensor()}, bad)
    with pytest.raises(RuntimeError, match="not a plain-tensor checkpoint"):
        read_state_dict_checkpoint(bad)


def test_registry_fetch_install_resolve(tmp_path, monkeypatch):
    from mppp.mask import hub
    ck = _tiny_checkpoint(tmp_path)
    st = hub.export_safetensors(ck, out=tmp_path / "remote" / "m.safetensors")
    reg = {"default": "m1", "models": {"m1": {"file": "m1.safetensors", "sha256": st["sha256"],
                                               "urls": [(tmp_path / "nowhere.safetensors").as_uri(),
                                                        Path(st["path"]).as_uri()]}}}
    (tmp_path / "models.json").write_text(json.dumps(reg))
    monkeypatch.setattr(hub, "registry_path", lambda: tmp_path / "models.json")
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        hub.resolve_checkpoint("m1", download=False).stat()
    p = hub.resolve_checkpoint()                                               # default -> download (2nd URL)
    assert p == tmp_path / "cache" / "models" / "m1.safetensors" and hub.sha256_file(p) == st["sha256"]
    assert hub.resolve_checkpoint(ck.name) == ck                               # a file in checkpoints_dir()
    assert hub.resolve_checkpoint(str(ck)) == ck
    with pytest.raises(FileNotFoundError, match="Mask checkpoint not found"):
        hub.resolve_checkpoint("nope.pt")
    # a corrupted download is rejected and kept aside
    reg["models"]["m1"]["sha256"] = "0" * 64
    (tmp_path / "models.json").write_text(json.dumps(reg))
    with pytest.raises(FileNotFoundError, match="SHA-256"):
        hub.fetch_model("m1", force=True)
    # install a local model under the registry name (different SHA -> warning)
    with pytest.warns(UserWarning, match="differs from the registry"):
        q = hub.install_model(ck, "m1")
    assert q.is_file()
    from mppp.mask import get_model
    model, card = get_model("m1", "cpu")
    assert card["fpn_width"] == 32


def test_export_updates_registry(tmp_path, monkeypatch):
    from mppp.mask import hub
    ck = _tiny_checkpoint(tmp_path)
    (tmp_path / "models.json").write_text(json.dumps({"default": "m1", "models": {"m1": {"file": "m1.safetensors",
                                                                                           "urls": ["u"]}}}))
    monkeypatch.setattr(hub, "registry_path", lambda: tmp_path / "models.json")
    info = hub.export_safetensors(ck, name="m1", update_registry=True)
    reg = json.loads((tmp_path / "models.json").read_text())
    assert Path(info["path"]).name == "m1.safetensors" and reg["models"]["m1"]["sha256"] == info["sha256"]
    assert reg["models"]["m1"]["urls"] == ["u"] and reg["models"]["m1"]["source_checkpoint"] == ck.name


def test_released_registry_entry():
    from mppp.mask.hub import load_registry
    reg = load_registry()
    e = reg["models"][reg["default"]]
    assert e["file"].endswith(".safetensors") and len(e.get("sha256", "")) == 64 and len(e["urls"]) == 2


# ------------------------------------------------------------ COLMAP helpers
def test_one_projection_for_everyone():
    import pycolmap
    from mppp.colmap import project_camera, scale_camera_params
    from mppp.error.colmap import project_camera as pe
    assert pe is project_camera
    p = [2950, 2951, 2560, 1920, -0.27, 0.1, 0, 0, -0.02, 0, 0, 0]
    x = np.array([[0.3, -0.2, 1.0], [0.01, 0.02, 2.0]])
    cam = pycolmap.Camera(model="FULL_OPENCV", width=5120, height=3840, params=p)
    assert np.allclose(project_camera("FULL_OPENCV", p, x), cam.img_from_cam(x), atol=1e-6)
    assert project_camera("FULL_OPENCV", p, x[0]).shape == (2,)
    q = scale_camera_params("FULL_OPENCV", p, 0.25)
    assert np.allclose(q[:4], np.array(p[:4]) / 4) and np.allclose(q[4:], p[4:])


def test_build_database_overwrites_by_default():
    import inspect
    from mppp.sfm.database import build_database
    assert inspect.signature(build_database).parameters["overwrite"].default is True


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


# ---------------------------------------------------------------- health
def test_alignment_health_on_synthetic_block(tmp_path):
    import pycolmap
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.health import assess_alignment, health_table, write_health
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=200)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rep = assess_alignment(proj, rec)
    by = {c["check"]: c for c in rep["checks"]}
    assert rep["verdict"] in ("pass", "warn")
    assert by["registered_fraction"]["value"] == 1.0 and by["registered_fraction"]["status"] == "pass"
    assert by["residual_median_px"]["value"] < 0.6 and by["residual_median_px"]["status"] == "pass"
    assert by["ray_displacement_max_px"]["value"] < 3 and by["rig_rotation_change_deg"]["value"] < 0.02
    assert by["station_shift_median_m"]["value"] < 0.1
    assert sum(rep["track_length_histogram"].values()) == len(rec.points3D)
    assert set(rep["cameras"]) == {"NL", "NR"} and "N" in rep["rig"]
    files = write_health(rep, tmp_path / "health", rec, proj)
    assert Path(files["png"]).is_file() and "alignment health" in Path(files["md"]).read_text()
    assert "residual_median_px" in health_table(rep)
    # a broken rig is caught
    from scipy.spatial.transform import Rotation
    sid = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=2)
    T = rec.rigs[1].sensor_from_rig(sid)
    bad = pycolmap.Rigid3d(pycolmap.Rotation3d(Rotation.from_rotvec(np.radians([0.3, 0, 0])).as_matrix()
                                               @ T.rotation.matrix()), T.translation)
    rec.rigs[1].set_sensor_from_rig(sid, bad)
    by = {c["check"]: c for c in assess_alignment(proj, rec)["checks"]}
    assert by["rig_rotation_change_deg"]["status"] == "fail"
    # thresholds can be overridden
    rep2 = assess_alignment(proj, rec, thresholds={"rig_rotation_change_deg": (1.0, 2.0, "above")})
    assert {c["check"]: c for c in rep2["checks"]}["rig_rotation_change_deg"]["status"] == "pass"


# ------------------------------------------------------- Mastcam-Z 34 mm in mppp.sfm
@pytest.fixture(scope="module")
def processed_pair(tmp_path_factory):
    import mppp
    from conftest import NLF, ZL0, needs_data
    if not (NLF.is_file() and ZL0.is_file()):
        pytest.skip("example IMGs not present")
    out = tmp_path_factory.mktemp("proc")
    cfg = mppp.load_config({"masking": {"infer_mask": False},
                            "export": {"formats": ["PNG8"], "write_colmap": False}})
    man = mppp.process_images([NLF, ZL0], out, cfg, mppp.load_waypoints(), progress=False)
    assert man["n_processed"] == 2
    return man, out


def test_project_with_navcam_and_mastcamz34(processed_pair, tmp_path):
    from mppp.sfm.project import SfmProject, camera_key
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, zcam_focus_bin=None)   # <= 0.14.3
    assert set(proj.cameras) == {"NL", "ZL034"}
    z = proj.cameras["ZL034"]
    assert (z["model"], z["width"], z["height"]) == ("FULL_OPENCV", 1648, 1200) and "label" in z["source"]
    meta = [m for m in man["images"] if m["filename"]["family"] == "Z"][0]
    K = np.asarray(meta["intrinsics"]["K"])
    assert np.isclose(z["params"][0], K[0, 0]) and np.isclose(z["params"][2], K[0, 2] + 0.5)      # corner origin
    assert np.isclose(z["params"][4], meta["intrinsics"]["dist_opencv"]["k1"])
    assert z["params"][6] == z["params"][7] == 0.0
    assert {r["instrument"] for r in proj.images} == {"NL", "ZL034"} and proj.rig == {}
    assert camera_key({"family": "Z", "camera_group": "ZR034", "instrument": "ZR"}) == "ZR034"
    px = SfmProject.create(man["images"], out, tmp_path / "q", link=False, zcam_intrinsics="xml", zcam_focus_bin=None)
    assert "ZL034_frame.xml" in px.cameras["ZL034"]["source"] and px.cameras["ZL034"]["params"][0] == 4720
    with pytest.raises(ValueError):
        SfmProject.create(man["images"], out, tmp_path / "r", zcam_intrinsics="guess")


def test_rig_family_for_mastcamz_pairs_with_shared_clock():
    from mppp.sfm.project import _rig_from_pairs
    R = np.eye(3).tolist()
    imgs = [{"instrument": k, "eye": k[1], "sclk_key": "1", "prior_R_w2c": R, "prior_C": c}
            for k, c in (("ZL034", [0, 0, 0]), ("ZR034", [0.242, 0, 0]), ("NL", [0, 0, 0]))]
    rig = _rig_from_pairs(imgs)
    assert list(rig) == ["Z034"] and (rig["Z034"]["ref"], rig["Z034"]["sensor"]) == ("ZL034", "ZR034")
    assert abs(rig["Z034"]["baseline_m"] - 0.242) < 1e-9


def test_select_best_products_several_sequences():
    from mppp.sfm.project import select_best_products
    names = ["NLF_0770_0736000000_000RAD_N0390000NCAM00500_0A0195J01.IMG",
             "ZL0_0770_0736000100_000RAD_N0390000ZCAM08000_0340LMA01.IMG",
             "NLF_0770_0736000200_000RAD_N0390000SAPP00500_0A0195J01.IMG"]
    kept, rep = select_best_products([Path(n) for n in names], sizes={n: 1 for n in names},
                                     sequence_prefix=("NCAM", "ZCAM"))
    assert [p.name for p in kept] == names[:2] and rep["n_dropped_sequence"] == 1
    kept, _ = select_best_products([Path(n) for n in names], sizes={n: 1 for n in names})
    assert [p.name for p in kept] == names[:1]                                           # default: NCAM only
