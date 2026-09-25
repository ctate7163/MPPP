"""v0p14: reprocess only the images left in the output folder; version-free model export."""
import json

import pytest

from conftest import NLF, ZL0, needs_data


@needs_data
def test_only_existing_rebuilds_manifest_from_remaining_images(tmp_path):
    import mppp
    from mppp.process import filter_to_existing
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": True}})
    wp = mppp.load_waypoints()
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False)
    assert man["n_processed"] == 2 and man["only_existing"] is None
    (tmp_path / "images_png8" / (ZL0.stem + ".png")).unlink()                 # the user removes one frame
    (tmp_path / "images_png8" / "NOT_SELECTED_0001.png").write_bytes(b"x")    # and an unrelated file is there
    kept, rep = filter_to_existing([NLF, ZL0], tmp_path, "PNG8")
    assert kept == [NLF] and rep["removed"] == [ZL0.name] and rep["not_in_selection"] == ["NOT_SELECTED_0001"]
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, only_existing="PNG8")
    assert man["n_requested"] == man["n_processed"] == 1
    assert [m["source_product"].split("/")[-1].split("\\")[-1] for m in man["images"]] == [NLF.name]
    assert man["only_existing"]["n_removed"] == 1
    on_disk = json.loads((tmp_path / f"mppp_manifest_{mppp.VERSION_TAG}.json").read_text())
    assert on_disk["n_processed"] == 1 and on_disk["only_existing"]["folder"].endswith("images_png8")
    refs = [l for l in (tmp_path / "references.txt").read_text().splitlines()
            if l and not l.startswith("#") and not l.startswith("filename")]
    assert len(refs) == 1 and NLF.stem in refs[0]
    assert (tmp_path / "images_png8" / (NLF.stem + ".png")).is_file()          # regenerated, nothing deleted
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mppp.process_images([NLF], tmp_path, cfg, wp, progress=False, only_existing="TIFF16")
    with pytest.raises(ValueError, match="none of the"):
        mppp.process_images([ZL0], tmp_path, cfg, wp, progress=False, only_existing="PNG8")


def test_export_does_not_depend_on_the_mppp_version(tmp_path, monkeypatch):
    from test_v0p13 import _tiny_checkpoint
    import mppp
    from mppp.mask.hub import export_safetensors
    from mppp.mask.model import read_card
    ck = _tiny_checkpoint(tmp_path)
    a = export_safetensors(ck, out=tmp_path / "a.safetensors")
    monkeypatch.setattr(mppp, "__version__", "9.9.9")
    b = export_safetensors(ck, out=tmp_path / "b.safetensors")
    assert a["sha256"] == b["sha256"] and "exported_with_mppp" not in read_card(tmp_path / "a.safetensors")


def test_download_failure_falls_back_to_local_checkpoint(tmp_path, monkeypatch):
    """v0p14.1: the model is not published yet (HF 401, GitHub 404) -> use the local best checkpoint."""
    import shutil
    from test_v0p13 import _tiny_checkpoint
    from mppp.mask import hub
    (tmp_path / "src").mkdir()
    ck = _tiny_checkpoint(tmp_path / "src")
    ckdir = tmp_path / "checkpoints"
    ckdir.mkdir()
    shutil.copy2(ck, ckdir / "convnext_tiny_s4_seg_best.pt")
    shutil.copy2(ck.with_suffix(".json"), ckdir / "convnext_tiny_s4_seg_best.json")
    ref = hub.export_safetensors(ckdir / "convnext_tiny_s4_seg_best.pt", out=tmp_path / "ref.safetensors", name="m1")
    reg = {"default": "m1", "models": {"m1": {"file": "m1.safetensors", "sha256": ref["sha256"],
                                               "source_checkpoint": "convnext_tiny_s4_seg_best.pt",
                                               "backbone": "convnext_tiny", "stride4": True,
                                               "urls": [(tmp_path / "missing.safetensors").as_uri()]}}}
    (tmp_path / "models.json").write_text(json.dumps(reg))
    monkeypatch.setattr(hub, "registry_path", lambda: tmp_path / "models.json")
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(ckdir))
    assert hub.local_candidates("m1") == [ckdir / "convnext_tiny_s4_seg_best.pt"]
    p = hub.resolve_checkpoint("m1")                                      # download fails -> local install
    assert p == tmp_path / "cache" / "models" / "m1.safetensors" and hub.sha256_file(p) == ref["sha256"]
    with pytest.raises(FileNotFoundError, match="No local fallback"):
        hub.fetch_model("m1", force=True, local_fallback=False)


def test_weak_image_causes():
    from mppp.sfm.health import classify_weak_image as c
    assert c(100, 0, 0, 0) == "few_keypoints"
    assert c(5000, None, None, 0) == "no_database"
    assert c(5000, 3, 10, 0) == "unmatched"
    assert c(5000, 5, 400, 2) == "stereo_only_far"
    assert c(5000, 900, 400, 10) == "lost_in_triangulation"
    assert c(5000, 6600, 0, 9, inliers_other_station=0) == "same_station_only"        # a left-only mast pan
    assert c(5000, 6600, 0, 9, inliers_other_station=500) == "lost_in_triangulation"


def test_health_lists_weak_images_with_advice(tmp_path):
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.health import DEFAULT_THRESHOLDS, assess_alignment, health_table
    assert DEFAULT_THRESHOLDS["rig_rotation_change_deg"][:2] == (0.06, 0.2)            # doubled (v0p14.2)
    assert DEFAULT_THRESHOLDS["outlier_image_fraction"][:2] == (0.04, 0.20)
    assert DEFAULT_THRESHOLDS["ray_displacement_max_px"][:2] == (20.0, 60.0)
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    victim = 3
    for pid in [pid for pid, pt in rec.points3D.items() if any(e.image_id == victim for e in pt.track.elements)]:
        if rec.points3D[pid].track.length() <= 2:
            rec.delete_point3D(pid)
        else:
            idx = [e.point2D_idx for e in rec.points3D[pid].track.elements if e.image_id == victim][0]
            rec.delete_observation(victim, idx)
    rep = assess_alignment(proj, rec)
    weak = rep["weak_images"]
    assert [w["name"] for w in weak] == [proj.images[victim - 1]["name"]] and weak[0]["observations"] == 0
    assert weak[0]["cause"] == "no_database"                                           # no database.db here
    txt = health_table(rep)
    assert "images with few observations (1)" in txt and proj.images[victim - 1]["name"] in txt


# ---------------------------------------------------------------- v0p14.3
def test_fixed_camera_params_tangential_switch():
    from mppp.sfm.reconstruction import fixed_camera_params
    assert fixed_camera_params("FULL_OPENCV") == [6, 7, 9, 10, 11]
    assert fixed_camera_params("FULL_OPENCV", refine_tangential=True) == [9, 10, 11]
    assert fixed_camera_params("FULL_OPENCV", refine_principal_point=False, refine_tangential=True) == [2, 3, 9, 10, 11]
    assert fixed_camera_params("OPENCV", refine_tangential=True) == []


def test_reconstruct_defaults_four_rounds_and_half_degree():
    import inspect
    from mppp.sfm.reconstruction import DEFAULT_SCHEDULE, reconstruct
    sig = inspect.signature(reconstruct).parameters
    assert sig["schedule"].default == DEFAULT_SCHEDULE and len(DEFAULT_SCHEDULE) == 4
    assert DEFAULT_SCHEDULE[-1] == (6.0, 1.5, 1.5)                        # 3 sigma at sigma_px = 0.5
    assert all(a[2] >= b[2] for a, b in zip(DEFAULT_SCHEDULE, DEFAULT_SCHEDULE[1:]))   # cut-offs only tighten
    assert sig["min_tri_angle_deg"].default == 0.5 and sig["sigma_px"].default == 0.5
    assert sig["refine_tangential"].default is False


def test_xml_tangential_terms_kept_when_not_zeroed():
    from mppp.paths import data_dir
    from mppp.sfm.project import camera_from_metashape_xml, read_metashape_calibration
    xml = data_dir() / "m20_cmods/M2020_NL0_frame.xml"
    c = read_metashape_calibration(xml)
    p = camera_from_metashape_xml(xml, zero_terms=("b1", "b2"))["params"]
    assert p[6] == pytest.approx(c["p2"]) and p[7] == pytest.approx(c["p1"])     # OpenCV p1 = Metashape P2
    assert abs(p[6]) > 1e-4 and p[0] == p[1]                                      # b1 still zeroed


def test_ba_recovers_tangential_only_when_asked(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("pyceres")
    import numpy as np
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    from mppp.sfm.health import _camera_change
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    for k in cams_d:                                                    # truth has tangential distortion
        cams_d[k]["params"][6], cams_d[k]["params"][7] = 3e-4, -2e-4
    for refine in (True, False):
        rec, true_params = _build_rec(proj, truth, P, cams_d, rigT, noise, np.random.default_rng(1))
        for cid in rec.cameras:                                         # start at p1 = p2 = 0
            q = np.array(rec.cameras[cid].params)
            q[6:8] = 0.0
            rec.cameras[cid].params = q
        out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation",
                            refine_tangential=refine, max_iterations=200)
        assert out["refine_tangential"] is refine
        for cid in (1, 2):
            p = np.asarray(rec.cameras[cid].params)
            if refine:
                assert abs(p[6] - 3e-4) < 5e-5 and abs(p[7] + 2e-4) < 5e-5
            else:
                assert p[6] == p[7] == 0.0
            assert p[9] == p[10] == p[11] == 0.0
    ch = _camera_change(dict(cams_d["NL"], params=list(cams_d["NL"]["params"])), rec.cameras[1])
    assert ch["p1_initial"] == pytest.approx(3e-4) and ch["p1_refined"] == 0.0


def test_features_reused_only_with_same_settings(tmp_path):
    from mppp.sfm.database import features_up_to_date, _features_record
    from mppp.sfm.project import SfmProject
    proj = SfmProject(tmp_path, [{"name": "a.png"}, {"name": "b.png"}], {}, {}, [0, 0, 0], {})
    assert not features_up_to_date(proj)                                # nothing yet
    proj.features_db.write_bytes(b"")
    assert not features_up_to_date(proj)                                # no record (e.g. extracted by 0.14.2)
    rec = {"max_num_features": 8192, "max_image_size": 3200, "domain_size_pooling": False, "images": ["a.png", "b.png"]}
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj, max_num_features=8192)
    assert not features_up_to_date(proj, max_num_features=16380)        # new setting -> extract again
    rec["max_num_features"] = 16380
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj)                                    # default is 16380
    proj.images.append({"name": "c.png"})
    assert not features_up_to_date(proj)                                # an image without features
