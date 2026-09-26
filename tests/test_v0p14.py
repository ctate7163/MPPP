"""v0p14: reprocess only the images left in the output folder; version-free model export."""
import json
from pathlib import Path

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
    assert DEFAULT_SCHEDULE[-1] == DEFAULT_SCHEDULE[-2] == (8.0, 2.0, 2.0)   # v0p14.5: round 4 repeats round 3
    assert all(a[2] >= b[2] for a, b in zip(DEFAULT_SCHEDULE, DEFAULT_SCHEDULE[1:]))   # cut-offs only tighten
    assert sig["min_tri_angle_deg"].default == 0.25 and sig["sigma_px"].default == 0.5     # v0p20: 0.25 (was 0.5)
    assert sig["refine_tangential"].default is True                                       # v0p20 (was False)


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
    from mppp.sfm.database import features_up_to_date, _features_record, image_fingerprints
    from mppp.sfm.project import SfmProject
    proj = SfmProject(tmp_path, [{"name": "a.png"}, {"name": "b.png"}], {}, {}, [0, 0, 0], {})
    assert not features_up_to_date(proj)                                # nothing yet
    proj.features_db.write_bytes(b"")
    assert not features_up_to_date(proj)                                # no record (e.g. extracted by 0.14.2)
    rec = {"max_num_features": 8192, "max_image_size": 5120, "domain_size_pooling": False, "images": ["a.png", "b.png"]}
    _features_record(proj).write_text(json.dumps(rec))
    assert not features_up_to_date(proj, max_num_features=8192)         # no file record (before 0.14.7)
    rec["files"] = image_fingerprints(proj)
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj, max_num_features=8192)
    assert not features_up_to_date(proj, max_num_features=16380)        # new setting -> extract again
    rec["max_num_features"] = 16380
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj)                                    # default is 16380
    proj.images.append({"name": "c.png"})
    assert not features_up_to_date(proj)                                # an image without features


def test_features_extracted_again_when_an_image_or_mask_changes(tmp_path):
    import os
    from mppp.sfm.database import features_up_to_date, _features_record, image_fingerprints
    from mppp.sfm.project import SfmProject
    proj = SfmProject(tmp_path, [{"name": "a.png"}], {}, {}, [0, 0, 0], {})
    proj.images_dir.mkdir(parents=True, exist_ok=True)
    proj.masks_dir.mkdir(parents=True, exist_ok=True)
    (proj.images_dir / "a.png").write_bytes(b"img")
    (proj.masks_dir / "a.png.png").write_bytes(b"mask")
    proj.features_db.write_bytes(b"")
    rec = {"max_num_features": 16380, "max_image_size": 5120, "domain_size_pooling": False, "images": ["a.png"],
           "files": image_fingerprints(proj)}
    _features_record(proj).write_text(json.dumps(rec))
    assert features_up_to_date(proj)
    m = proj.masks_dir / "a.png.png"
    m.write_bytes(b"new mask")                                          # the image was processed again
    os.utime(m, ns=(m.stat().st_atime_ns, m.stat().st_mtime_ns + 10**9))
    assert not features_up_to_date(proj)


def test_project_copy_refreshed_when_the_processed_image_changes(tmp_path):
    import os
    from mppp.sfm.project import _link_or_copy
    src, dst = tmp_path / "src.png", tmp_path / "dst.png"
    src.write_bytes(b"v1")
    _link_or_copy(src, dst, link=False)                                 # a copy (e.g. another drive)
    assert dst.read_bytes() == b"v1"
    _link_or_copy(src, dst, link=False)                                 # unchanged: kept
    src.write_bytes(b"version 2")
    os.utime(src, ns=(src.stat().st_atime_ns, dst.stat().st_mtime_ns + 10**9))
    _link_or_copy(src, dst, link=False)
    assert dst.read_bytes() == b"version 2"
    l = tmp_path / "link.png"
    _link_or_copy(src, l, link=True)
    src.write_bytes(b"version 3")                                       # a hard link follows in place
    _link_or_copy(src, l, link=True)
    assert l.read_bytes() == b"version 3"


# ---------------------------------------------------------------- v0p14.4
def test_focus_bins_greedy_and_tags():
    from mppp.sfm.project import _focus_tag, focus_bins
    counts = [100, 110, 129, 131, 200, None, 205, 250]
    assert focus_bins(counts, 30) == [0, 0, 0, 1, 2, 4, 2, 3]            # no grid line splits 129 / 131 from 100?
    assert focus_bins([5, 6, 7], 30) == [0, 0, 0] and focus_bins([], 30) == []
    assert (_focus_tag(2312.4), _focus_tag(-150), _focus_tag(None)) == ("F02312", "Fm00150", "Fna")


def test_project_bins_mastcamz_by_focus(processed_pair, tmp_path):
    from mppp.sfm.project import ZCAM_BIN_HELD, SfmProject
    man, out = processed_pair
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False)
    zmeta = [m for m in man["images"] if m["filename"]["family"] == "Z"][0]
    fc = float(zmeta["focus_position_count"])
    key = f"ZL034_F{int(round(fc)):05d}"
    assert set(proj.cameras) == {"NL", key}
    z = proj.cameras[key]
    assert z["group"] == "ZL034" and z["focus_count_median"] == fc and z["n_images"] == 1
    assert z["fixed_params"] == list(ZCAM_BIN_HELD) and "focus bin" in z["source"]
    r = [r for r in proj.images if r["camera_group"] == "ZL034"][0]
    assert r["instrument"] == key and r["focus_count"] == fc and abs(r["label_f_px"] - z["params"][0]) < 1e-9
    assert proj.settings["zcam_focus_bin"] == 30.0 and proj.settings["zcam_bin_refine"] == "focal"
    pa = SfmProject.create(man["images"], out, tmp_path / "q", link=False, zcam_bin_refine="all")
    assert pa.cameras[key]["fixed_params"] == []
    with pytest.raises(ValueError):
        SfmProject.create(man["images"], out, tmp_path / "r", zcam_bin_refine="some")


@pytest.fixture(scope="module")
def processed_pair(tmp_path_factory):
    from test_v0p13 import processed_pair as fx
    return fx.__wrapped__(tmp_path_factory)


def test_ba_holds_camera_specific_parameters(tmp_path):
    pytest.importorskip("pyceres")
    import numpy as np
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, true_params = _build_rec(proj, truth, P, cams_d, rigT, noise, rng)
    start = {cid: np.array(rec.cameras[cid].params) for cid in (1, 2)}
    proj.cameras["NR"]["fixed_params"] = ["cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=100)
    p1, p2 = np.asarray(rec.cameras[1].params), np.asarray(rec.cameras[2].params)
    assert np.array_equal(p2[2:], start[2][2:])                         # NR: all but fx, fy held
    assert abs(p2[0] - start[2][0]) > 1.0                               # its focal length moved (+0.3 % start)
    assert abs(p1[2] - start[1][2]) > 0.5 and abs(p1[4] - start[1][4]) > 1e-4     # NL refines as before
    # NR dressed as a Mastcam-Z focus bin: table, plot and health run on it
    from mppp.sfm.health import assess_alignment
    from mppp.sfm.zcam import write_focus_breathing
    proj.cameras["NR"].update(group="NR", focus_count_median=1000.0, focus_count_range=[995.0, 1004.0], n_images=12)
    for r in proj.images:
        r["camera_group"] = r["instrument"]
        if r["instrument"] == "NR":
            r["focus_count"], r["label_f_px"] = 1000.0, float(start[2][0])
    fb = write_focus_breathing(proj, rec, tmp_path / "fb", min_observations=10)
    row = fb["table"][0]
    assert len(fb["table"]) == 1 and row["camera"] == "NR" and row["observations"] > 1000
    assert abs(row["f_refined_px"] - 0.5 * (p2[0] + p2[1])) < 1e-9 and row["held_params"].startswith("cx,cy")
    assert fb["fits"]["NR"]["refined"] is None and Path(fb["png"]).is_file()          # one bin: no slope
    rep = assess_alignment(proj, rec)
    assert "NR" in rep["cameras"] and any(c["check"] == "residual_eye_ratio" for c in rep["checks"])


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


def test_camera_shift_plot(tmp_path):
    pytest.importorskip("pyceres")
    import matplotlib
    matplotlib.use("Agg")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.export import _nice, plot_camera_shifts, pose_residual_table
    from mppp.sfm.reconstruction import bundle_adjust
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    for r in proj.images:                                          # priors = truth, start perturbed
        r["prior_R_w2c"], r["prior_C"] = truth[r["name"]][0].tolist(), truth[r["name"]][1].tolist()
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng)
    bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_rig="rotation", max_iterations=50)
    rows = pose_residual_table(rec, proj)
    fig = plot_camera_shifts(proj, rows=rows, out_png=tmp_path / "shifts.png")
    assert (tmp_path / "shifts.png").stat().st_size > 20000 and len(fig.axes) >= 2
    assert (_nice(37), _nice(0.23), _nice(1)) == (20, 0.2, 1)
    import csv
    with open(tmp_path / "p.csv", "w", newline="") as f:              # also from poses.csv (strings)
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot_camera_shifts(proj, rows=list(csv.DictReader(open(tmp_path / "p.csv"))), exaggeration=100)


def test_colmap_gui_project_files(tmp_path):
    from mppp.sfm.database import write_gui_project
    from mppp.sfm.project import SfmProject
    (tmp_path / "masks").mkdir()
    proj = SfmProject(tmp_path, [{"name": "a.png", "has_mask": True}], {}, {}, [0, 0, 0], {})
    out = write_gui_project(proj, model="cahv_ba")
    ini = Path(out["ini"]).read_text().splitlines()
    assert ini[0].startswith("# MPPP") and f"database_path={tmp_path.resolve() / 'database.db'}" in ini
    assert f"image_path={tmp_path.resolve() / 'images'}" in ini and "[ImageReader]" in ini
    bat = Path(out["bat"]).read_bytes()
    assert b"\r\n" in bat and b"--import_path \"%MODEL%\"" in bat and b"sparse\\cahv_ba" in bat
    assert b"%COLMAP_BAT%" in bat


def test_station_labels_start_with_the_sol():
    from mppp.sfm.project import SfmProject, station_labels
    imgs = [{"station": "S032D1184", "sol": 686}, {"station": "S032D1184", "sol": 686},
            {"station": "S032D1174", "sol": 684}, {"station": "S032D1174", "sol": 685}, {"station": "S001D0000"}]
    lab = station_labels(imgs)
    assert lab == {"S032D1184": "Sol0686 S032D1184", "S032D1174": "Sol0684-0685 S032D1174", "S001D0000": "S001D0000"}
    assert sorted(lab.values())[1] == "Sol0684-0685 S032D1174"
    proj = SfmProject(Path("."), imgs, {}, {}, [0, 0, 0], {})
    assert proj.station_label("S032D1184") == "Sol0686 S032D1184" and proj.station_label("X") == "X"



# ---------------------------------------------------------------- v0p14.5
def test_prior_rotation_correction_keeps_the_ray_at_the_camera_principal_point():
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mppp.sfm.project import prior_rotation_correction
    rng = np.random.default_rng(4)
    R = Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix()
    f, cl, cc = 4690.0, np.array([770.0, 650.0]), np.array([824.0, 600.0])
    M = prior_rotation_correction(cl, f, cc)
    Rn = M @ R
    ray = lambda Rw2c, c, u: Rw2c.T @ np.r_[(u - c) / f, 1.0]                     # noqa: E731
    u = cc
    a, b = ray(R, cl, u), ray(Rn, cc, u)
    assert np.allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), atol=1e-12)    # same ray at the camera pp
    ang = np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1)))
    assert abs(ang - np.degrees(np.arctan(np.linalg.norm(cc - cl) / f))) < 1e-9     # ~0.89 deg here
    u = cc + [300.0, -200.0]                                                        # elsewhere: second order only
    a, b = ray(R, cl, u), ray(Rn, cc, u)
    err_px = f * np.linalg.norm(a / a @ (R.T[:, 2]) - b / b @ (R.T[:, 2]))
    assert err_px < 2.0
    assert np.allclose(prior_rotation_correction(cl, f, cl), np.eye(3))


def test_native_model_is_a_valid_colmap_model(tmp_path):
    pytest.importorskip("pyceres")
    import numpy as np
    import pycolmap
    from test_sfm import _build_rec, _synthetic
    from mppp.error.colmap import read_colmap
    from mppp.sfm.export import native_reconstruction, write_native_text_model
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    info = write_native_text_model(rec, proj, tmp_path / "native")
    back = pycolmap.Reconstruction(str(tmp_path / "native"))       # <= 0.14.4: 'Check failed: point2D.point3D_id'
    assert len(back.points3D) == info["points"] == len(rec.points3D) and len(back.images) == len(rec.images)
    assert {(c.width, c.height) for c in back.cameras.values()} == {(2560, 1920), (1280, 960)}
    res = []
    for pt in list(back.points3D.values())[:300]:
        for el in pt.track.elements:
            im = back.images[el.image_id]
            uv = back.cameras[im.camera_id].img_from_cam(im.cam_from_world() * pt.xyz)
            res.append(np.linalg.norm(uv - im.points2D[el.point2D_idx].xy))
    assert np.median(res) < 3 * noise                                  # native keypoints and native cameras agree
    assert len(read_colmap(str(tmp_path / "native")).points) == info["points"]      # mppp.error still reads it
    full = native_reconstruction(rec, proj, observed_only=False)
    assert sum(len(im.points2D) for im in full.images.values()) == sum(len(im.points2D) for im in rec.images.values())


# ---------------------------------------------------------------- mask training labels
def test_training_labels_come_from_masks_folder_not_alpha(tmp_path):
    import cv2
    import numpy as np
    from mppp.mask.train import MaskDataset, read_pair, scan_dataset
    for d in ("images", "images_variable", "masks"):
        (tmp_path / d).mkdir()
    h, w = 64, 96
    rgb = np.full((h, w, 3), 90, np.uint8)
    rgb[:, :, 1] = 140
    stale_alpha = np.zeros((h, w), np.uint8)              # alpha says "all excluded" ...
    stale_alpha[:, : w // 4] = 255
    mask = np.zeros((h, w), np.uint8)                      # ... masks/ says the right half is terrain
    mask[:, w // 2:] = 255
    for d in ("images", "images_variable"):
        cv2.imwrite(str(tmp_path / d / "a.png"), np.dstack([rgb, stale_alpha]))
    cv2.imwrite(str(tmp_path / "masks" / "a.png"), mask)

    items = scan_dataset(tmp_path)
    assert len(items) == 2 and all(Path(it.mask) == tmp_path / "masks" / "a.png" for it in items)
    for it in items:
        im, ms = read_pair(it)
        assert im.shape == (h, w, 3) and np.array_equal(im, rgb)       # alpha dropped, RGB not composited
        assert np.array_equal(ms, mask)
        x, y, _ = MaskDataset([it], size=w, canvas=(w, h))[0]
        assert np.array_equal(y, (mask > 127).astype(np.uint8))
        assert np.array_equal(x, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    import pytest
    with pytest.raises(ValueError):
        scan_dataset(tmp_path, image_dirs=("images", "masks"), mask_dir="masks")


# ---------------------------------------------------------------- v0p14.7
@needs_data
def test_reuse_existing_processes_only_what_is_missing_or_changed(tmp_path, capsys):
    import mppp
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": True}})
    wp = mppp.load_waypoints()
    png = lambda p: tmp_path / "images_png8" / (p.stem + ".png")                  # noqa: E731
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True)
    assert man["reuse_existing"]["reused"] == 0 and man["reuse_existing"]["to_process"] == 2
    refs_full = (tmp_path / "references.txt").read_text()
    t_nlf = png(NLF).stat().st_mtime_ns

    # False (all selected), nothing changed: nothing is processed, same manifest and references
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True)
    assert man["reuse_existing"]["reused"] == 2 and man["reuse_existing"]["to_process"] == 0
    assert png(NLF).stat().st_mtime_ns == t_nlf and man["n_processed"] == 2
    assert (tmp_path / "references.txt").read_text() == refs_full

    # True (only_existing) after deleting a frame: no reprocessing, manifest and references without it
    png(ZL0).unlink()
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True, only_existing="PNG8")
    assert man["reuse_existing"]["to_process"] == 0 and man["n_processed"] == 1
    assert png(NLF).stat().st_mtime_ns == t_nlf and not png(ZL0).exists()
    refs = (tmp_path / "references.txt").read_text()
    assert NLF.stem in refs and ZL0.stem not in refs

    # False again: the full selection; only the missing frame is processed
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True)
    assert man["reuse_existing"]["reused"] == 1 and man["reuse_existing"]["to_process"] == 1
    assert png(ZL0).is_file() and png(NLF).stat().st_mtime_ns == t_nlf and man["n_processed"] == 2
    assert (tmp_path / "references.txt").read_text() == refs_full                 # rebuilt rows = processed rows

    # another configuration (e.g. another mask model): everything is processed again, with the reason
    cfg2 = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": True,
                                                                          "store_mask_in_alpha": False}})
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg2, wp, progress=True, reuse_existing=True)
    assert man["reuse_existing"]["to_process"] == 2
    assert "export.store_mask_in_alpha" in man["reuse_existing"]["config_changed"]
    assert "configuration changed" in capsys.readouterr().out


def test_mask_model_v2_is_the_default():
    import mppp
    from mppp.mask.hub import load_registry
    reg = load_registry()
    assert reg["default"] == "mppp_mask_v2" == mppp.default_config()["masking"]["checkpoint"]
    v2 = reg["models"]["mppp_mask_v2"]
    assert v2["source_checkpoint"] == "convnext_tiny_s4_seg_20260925.pt" and v2["val_iou"] > 0.977
    assert v2["sha256"] == "227e483369e11d6e36ce3517cf9f6d6ac60a9a50c1e301f9e7ae53d0cef569b9"
    assert all(u.endswith(v2["file"]) and "mask-v2" in u or "huggingface" in u for u in v2["urls"])
    assert "mppp_mask_v1" in reg["models"]                                         # the previous model stays


def test_parse_stations():
    from mppp.config import parse_stations
    assert parse_stations([[32, 1184], "S032D1208", "Sol0686-0688 S032D1062", "32/1394", (5, 7)]) == \
        {(32, 1184), (32, 1208), (32, 1062), (32, 1394), (5, 7)}
    assert parse_stations(None) == set()
    with pytest.raises(ValueError):
        parse_stations(["Sol 686"])
    import mppp
    with pytest.raises(ValueError):
        mppp.load_config({"masking": {"skip_inference_at": ["nonsense"]}})


@needs_data
def test_skip_inference_at_one_station_keeps_the_rover(tmp_path, monkeypatch):
    import numpy as np
    import mppp
    from mppp.image import MPPPImage

    def fake_infer(self, rad):                     # a "model" that excludes the left half of every frame
        m = self.mask_valid.copy()
        m[:, : m.shape[1] // 2] = 0
        self.mask, self.mask_card = m, {"name": "fake", "release_name": "fake"}
    monkeypatch.setattr(MPPPImage, "_infer_mask", fake_infer)
    monkeypatch.setattr("mppp.process.check_mask_checkpoint", lambda cfg: None)
    wp = mppp.load_waypoints()
    base = {"export": {"formats": ["PNG8"], "write_mask_files": True}}
    man = mppp.process_images([NLF, ZL0], tmp_path, mppp.load_config(base), wp, progress=False, reuse_existing=True)
    by = {m["source_product"]: m for m in man["images"]}
    nlf = by[NLF.name]
    assert nlf["mask"]["inferred"] and nlf["mask"]["included_fraction"] < nlf["mask"]["valid_fraction"]
    station = f"S{nlf['site']:03d}D{nlf['drive']:04d}"
    # both example products are from one station: move ZL0 to another drive in the manifest
    mp = tmp_path / f"mppp_manifest_{mppp.VERSION_TAG}.json"
    on_disk = json.loads(mp.read_text())
    for m in on_disk["images"]:
        if m["source_product"] == ZL0.name:
            m["drive"] = nlf["drive"] + 1
    mp.write_text(json.dumps(on_disk))

    cfg = mppp.load_config({**base, "masking": {"skip_inference_at": [station]}})
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True)
    rep = man["reuse_existing"]
    assert rep["mask_inference_changed"] == [NLF.stem] and rep["to_process"] == 1 and rep["reused"] == 1
    by = {m["source_product"]: m for m in man["images"]}
    nlf, zl0 = by[NLF.name], by[ZL0.name]
    assert zl0["drive"] == nlf["drive"] + 1                                   # reused from the manifest
    assert nlf["mask"]["inference_skipped"] and not nlf["mask"]["inferred"]
    assert nlf["mask"]["included_fraction"] == pytest.approx(nlf["mask"]["valid_fraction"])   # only black masked
    assert zl0["mask"]["inferred"] and not zl0["mask"]["inference_skipped"]
    import cv2
    mk = cv2.imread(str(tmp_path / "masks" / (NLF.stem + ".png")), cv2.IMREAD_GRAYSCALE)
    assert (mk > 0).mean() == pytest.approx(nlf["mask"]["valid_fraction"])
    # same list again: nothing to do; list emptied: the station's images get the model mask back
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True)
    assert man["reuse_existing"]["to_process"] == 0
    man = mppp.process_images([NLF, ZL0], tmp_path, mppp.load_config(base), wp, progress=False, reuse_existing=True)
    assert man["reuse_existing"]["mask_inference_changed"] == [NLF.stem]
