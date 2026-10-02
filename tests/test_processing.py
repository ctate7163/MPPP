"""Image processing (mppp.process, image, radiometry, writers, config). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
import warnings
import numpy as np
import pytest
from conftest import NLF, ZL0, needs_data
from pathlib import Path
from conftest import NLF, ZL0, needs_data, synthetic_waypoints
from types import SimpleNamespace
import types


def _meta():
    return {"mppp_version": "v0p1", "source_product": "X.IMG", "camera_group": "NL0",
            "start_time_utc": "2023-02-17T05:32:48.193", "zoom_mm": None, "undistorted": False,
            "LMST": "a", "LTST": "b",
            "intrinsics": {"K": [[1000, 0, 50], [0, 1000, 40], [0, 0, 1]], "pixel_origin": "centre",
                           "dist_opencv": {"k1": -0.1}},
            "geo": {"lon_east_deg": 77.45089, "lat_deg": 18.44463, "elev_geoid_m": -2567.25}}


def test_config_defaults_and_legacy_migration(tmp_path):
    from mppp import load_config
    cfg = load_config()
    assert cfg["export"]["formats"] == ["PNG16"]
    legacy = {"output_dir": "/", "cameras": ["N"], "export": {"format": "PNG8A", "colmap_model": "OPENCV"},
              "masking": {"checkpoint_trained": "x.pt", "checkpoint_template": "", "max_length": 1024},
              "resize": {"standard_width": 2560}, "camera_model": {"coordinate_frame": "NED"}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                           # legacy keys must migrate silently
        cfg = load_config(legacy)
    assert cfg["export"]["formats"] == ["PNG8"] and cfg["export"]["store_mask_in_alpha"] is True
    assert cfg["masking"]["checkpoint"] == "x.pt"
    with pytest.warns(UserWarning, match="unknown key"):
        load_config({"color": {"typo_key": 1}})
    with pytest.raises(ValueError):
        load_config({"export": {"formats": ["JPEG"]}})


def test_config_snapshot_has_version(tmp_path):
    from mppp import load_config, save_config_snapshot, VERSION_TAG
    p = save_config_snapshot(load_config(), tmp_path)
    snap = json.loads(p.read_text())
    assert p.name == f"mppp_config_{VERSION_TAG}.json"
    assert snap["mppp_version_tag"] == VERSION_TAG and "color" in snap["config"]


def test_tau_table_and_zenith_scale(tmp_path):
    from mppp.radiometry import interpolate_table, zenith_scale
    t = tmp_path / "tau.csv"
    t.write_text("L_s,tau\n0,0.7\n25,0.4\n80,0.4\n360,0.7\n")
    assert interpolate_table(t, 12.5) == pytest.approx(0.55)
    assert interpolate_table(t, 50) == pytest.approx(0.4)
    assert interpolate_table(t, 372.5) == pytest.approx(0.55)        # periodic in L_s
    s, mu = zenith_scale(90.0, 0.3, 0.3, 0.2)
    assert (s, mu) == (pytest.approx(1.0), pytest.approx(1.0))
    s, mu = zenith_scale(30.0, 0.9, 0.3, 0.2)
    assert s == pytest.approx(0.5 * np.exp(-0.6 / 6 / 0.5))
    assert zenith_scale(2.0, 0.3, 0.3, 0.2)[1] == 0.2                # floor near the horizon


def test_quantise_reserves_zero_for_invalid():
    from mppp import default_config
    from mppp.radiometry import quantise
    rad = np.zeros((2, 2, 3))
    rad[0, 0] = 0.05
    rad[0, 1] = 1e-9          # valid but darker than one count
    rad[1, 0] = 10.0          # saturates
    valid = np.array([[True, True], [True, False]])
    i16, i8 = quantise(rad, valid, default_config()["color"])
    assert i16.dtype == np.uint16 and i8.dtype == np.uint8
    assert i16[0, 0, 0] == 10000 and i8[0, 0, 0] == round(10000 / 64)
    assert i16[0, 1, 0] == 1 and i8[0, 1, 0] == 1                    # valid never 0
    assert i16[1, 0, 0] == 65535 and i8[1, 0, 0] == 255
    assert not i16[1, 1].any() and not i8[1, 1].any()


def test_png16_rgba_is_lossless_and_carries_metadata(tmp_path):
    import cv2
    import piexif
    from mppp.writers import save_png, read_png_chunks
    rng = np.random.default_rng(0)
    img = rng.integers(0, 65536, (40, 60, 4), dtype=np.uint16)
    p = save_png(img, tmp_path / "a", _meta())
    back = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    assert back.dtype == np.uint16 and np.array_equal(back, img)
    ch = read_png_chunks(p)
    assert list(ch)[0] == "IHDR" and "eXIf" in ch and len(ch["iTXt"]) > 5
    xmp = [c for c in ch["iTXt"] if c.startswith(b"XML:com.adobe.xmp")][0]
    assert b"mppp:payload" in xmp and b"X.IMG" in xmp
    gps = piexif.load(ch["eXIf"][0])["GPS"]
    d, m, s = [n / q for n, q in gps[piexif.GPSIFD.GPSLatitude]]
    assert d + m / 60 + s / 3600 == pytest.approx(18.44463, abs=1e-7)
    d, m, s = [n / q for n, q in gps[piexif.GPSIFD.GPSLongitude]]
    assert d + m / 60 + s / 3600 == pytest.approx(77.45089, abs=1e-7)
    assert gps[piexif.GPSIFD.GPSLongitudeRef] == b"E" and gps[piexif.GPSIFD.GPSAltitudeRef] == 1
    n, q = gps[piexif.GPSIFD.GPSAltitude]
    assert n / q == pytest.approx(2567.25)
    from PIL import Image                                         # an independent decoder accepts the file
    with Image.open(p) as im:
        im.verify()


def test_png8_and_tiff16(tmp_path):
    import tifffile
    from mppp.writers import save_png, save_tiff16
    rng = np.random.default_rng(1)
    i8 = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    from PIL import Image
    assert np.array_equal(np.asarray(Image.open(save_png(i8, tmp_path / "b", _meta()))), i8)
    i16 = rng.integers(0, 65536, (20, 30, 4), dtype=np.uint16)
    p = save_tiff16(i16, tmp_path / "c", _meta())
    assert np.array_equal(tifffile.imread(p), i16)
    with pytest.raises(ValueError):
        save_png(i16.astype(np.float32), tmp_path / "d")


def test_references_offset(tmp_path):
    from mppp.writers import reference_offset, save_references
    refs = [["a", 1234.5, -987.6, 44.0, 10, 20, 30], ["b", 1236.5, -985.6, 46.0, 11, 21, 31]]
    off = reference_offset(refs)
    assert np.allclose(off, [1230, -990, 40])
    rows = save_references(refs, tmp_path / "r.txt", off).read_text().splitlines()
    assert rows[0].split("\t")[0] == "filename" and rows[1].split("\t")[:2] == ["a.png", "4.500000"]


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
    # v0p22.4: a missing folder or a selection with nothing in the folder no longer raises - the filter is
    # ignored for that run and everything is processed (the manifest says so)
    m = mppp.process_images([NLF], tmp_path, cfg, wp, progress=False, only_existing="TIFF16")
    assert m["n_processed"] == 1 and "does not exist" in m["only_existing"]["skipped"]
    m = mppp.process_images([ZL0], tmp_path, cfg, wp, progress=False, only_existing="PNG8")
    assert m["n_processed"] == 1 and "no selected product" in m["only_existing"]["skipped"]


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
    man = mppp.process_images([NLF, ZL0], tmp_path, mppp.load_config(base), wp, progress=False, reuse_existing=True, workers=1)
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
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True, workers=1)
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
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, reuse_existing=True, workers=1)
    assert man["reuse_existing"]["to_process"] == 0
    man = mppp.process_images([NLF, ZL0], tmp_path, mppp.load_config(base), wp, progress=False, reuse_existing=True, workers=1)
    assert man["reuse_existing"]["mask_inference_changed"] == [NLF.stem]


def test_reuse_existing_finds_the_manifest_of_the_previous_version(tmp_path):
    import json
    import mppp
    from mppp.process import reusable_images
    cfg = mppp.load_config({"masking": {"infer_mask": False}})
    (tmp_path / "images_png8").mkdir()
    (tmp_path / "images_png8" / "a.png").write_bytes(b"x")
    meta = {"source_product": "a.IMG", "site": 1, "drive": 2, "outputs": {"PNG8": "images_png8/a.png"},
            "mask": {"inferred": False}}
    (tmp_path / "mppp_manifest_v0p14.json").write_text(json.dumps({"images": [meta]}))
    (tmp_path / "mppp_config_v0p14.json").write_text(json.dumps({"config": cfg}, default=str))
    have, rep = reusable_images(tmp_path, cfg)
    assert list(have) == ["a"] and rep["manifest"].endswith("mppp_manifest_v0p14.json")


def test_reuse_existing_notices_a_new_waypoint_table(tmp_path):
    import mppp
    from mppp.process import reusable_images
    cfg = mppp.load_config({"masking": {"infer_mask": False}})
    (tmp_path / "images_png8").mkdir()
    (tmp_path / "images_png8" / "a.png").write_bytes(b"x")
    meta = {"source_product": "a.IMG", "site": 1, "drive": 2, "outputs": {"PNG8": "images_png8/a.png"},
            "mask": {"inferred": False}}
    (tmp_path / f"mppp_manifest_{mppp.VERSION_TAG}.json").write_text(json.dumps({"images": [meta]}))
    (tmp_path / f"mppp_config_{mppp.VERSION_TAG}.json").write_text(
        json.dumps({"config": cfg, "waypoints": {"sha256": "old"}}, default=str))
    have, rep = reusable_images(tmp_path, cfg, {"_mppp_source": {"sha256": "old"}})
    assert list(have) == ["a"]
    have, rep = reusable_images(tmp_path, cfg, {"_mppp_source": {"sha256": "new"}})
    assert have == {} and rep["reason"] == "waypoint table changed"


DATA = Path(__file__).parent / "data" / "m20"


NLF = DATA / "NLF_0709_0729883381_848RAD_N0332864SAPP00601_0A00LLJ01.IMG"


ZL0 = DATA / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"


def test_manifest_keeps_the_full_label_model(cfg_nomask):
    from mppp.cmod import PixelCamera, compare_cameras
    from mppp.image import MPPPImage
    from mppp.sfm.calibration import label_model_from_meta
    for path, tol in ((ZL0, 0.6), (NLF, 5.0)):
        m = json.loads(json.dumps(MPPPImage(path, cfg_nomask, None).meta, default=float))
        assert m["camera_model_label"]["MODEL_TYPE"] in ("CAHVOR", "CAHVORE")
        exact = label_model_from_meta(m)
        m.pop("camera_model_label")
        approx = label_model_from_meta(m)                       # older manifests: O along A
        d = compare_cameras(PixelCamera.cahv(exact, as_is=True), PixelCamera.cahv(approx, as_is=True), 96, False)
        assert d["max_px"] < tol
        _, li = exact.decompose()
        W, H = (5120, 3840) if path == NLF else (1648, 1200)
        assert (exact.width, exact.height) == (W, H) and 0.4 * W < li["hc"] < 0.6 * W


@needs_data
def test_sky_pointing_navcam_is_skipped(tmp_path):
    import mppp
    from mppp.image import MPPPImage, SkyImage
    from mppp.process import process_images
    wp = synthetic_waypoints()
    on = mppp.load_config({"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": 45.0}})
    off = mppp.load_config({"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": None}})
    with pytest.raises(SkyImage):
        MPPPImage(NLF, on, wp)                                    # the Sun-pointing tile looks up at +78 deg
    assert MPPPImage(NLF, off, wp).pose.boresight_az_el_deg[1] > 45
    MPPPImage(ZL0, on, wp)                                        # Mastcam-Z is not filtered (looks down anyway)
    high = mppp.load_config({"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": 80.0}})
    MPPPImage(NLF, high, wp)
    man = process_images([NLF, ZL0], tmp_path / "out", on, wp, progress=False)
    assert man["n_processed"] == 1 and man["failed"] == [] and len(man["skipped"]) == 1
    assert "sky-pointing" in man["skipped"][0]["reason"] and man["skipped"][0]["file"].endswith(NLF.name)


@needs_data
def test_only_existing_is_ignored_on_a_first_run(tmp_path):
    import mppp
    from mppp.process import process_images, filter_to_existing
    wp = synthetic_waypoints()
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"]}})
    kept, rep = filter_to_existing([ZL0], tmp_path / "nothing", "PNG8")
    assert kept == [ZL0] and rep["skipped"] and rep["n_kept"] == 1
    man = process_images([ZL0], tmp_path / "out", cfg, wp, only_existing="PNG8", progress=False)
    assert man["n_processed"] == 1 and man["only_existing"]["skipped"]
    # second run: the folder exists and the image is in it -> the normal filter applies
    man2 = process_images([ZL0], tmp_path / "out", cfg, wp, only_existing="PNG8", progress=False)
    assert man2["only_existing"]["n_kept"] == 1 and "skipped" not in man2["only_existing"]
    # a selection none of which is in the folder -> processed, not an error
    (tmp_path / "out2" / "images_png8").mkdir(parents=True)
    kept3, rep3 = filter_to_existing([ZL0], tmp_path / "out2", "PNG8")
    assert kept3 == [ZL0] and rep3["skipped"]


ROOT = Path(__file__).resolve().parents[1]


def test_defaults():
    import mppp
    cfg = mppp.load_config({})
    assert cfg["color"]["brightness_by_family"] == {"N": 0.9}
    # the tests run with the selection rules off (conftest); the shipped defaults are these
    import mppp.config as c
    src = (ROOT / "src" / "mppp" / "config.py").read_text()
    assert '"lmst_window_h": [6.0, 18.0]' in src and '"max_saturated_fraction": 0.2,' in src     # v0p65
    assert '"max_boresight_elevation_deg": 45.0' in src
    import inspect
    from mppp.process import process_images
    assert inspect.signature(process_images).parameters["workers"].default == 4


@needs_data
def test_brightness_lmst_and_saturation_rules(tmp_path):
    import mppp
    from mppp.image import MPPPImage, LmstOutOfWindow, SaturatedImage, lmst_hours
    from mppp.process import process_images
    wp = synthetic_waypoints()
    assert lmst_hours("Sol-01451M13:00:25.313") == pytest.approx(13.007, abs=1e-3) and lmst_hours(None) is None
    base = {"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": None}}
    im = MPPPImage(NLF, mppp.load_config(base), wp)
    assert im.brightness == 0.9 and im.meta["brightness"] == 0.9 and im.meta["saturated_fraction"] > 0.99
    assert MPPPImage(ZL0, mppp.load_config(base), wp).brightness == 1.0
    dark = mppp.load_config(dict(base, color={"brightness_by_family": {"N": 0.5}}))
    a = MPPPImage(ZL0, mppp.load_config(base), wp).image_int8.astype(int)
    b = MPPPImage(ZL0, mppp.load_config(dict(base, color={"brightness_by_family": {"Z": 0.5}})), wp).image_int8.astype(int)
    v = a > 10
    assert np.median(b[v] / a[v]) == pytest.approx(0.5, abs=0.05)          # brightness scales the product
    with pytest.raises(SaturatedImage):
        MPPPImage(NLF, mppp.load_config(dict(base, selection={"max_boresight_elevation_deg": None, "max_saturated_fraction": 0.05})), wp)
    with pytest.raises(LmstOutOfWindow):
        MPPPImage(ZL0, mppp.load_config(dict(base, selection={"lmst_window_h": [9.0, 12.0]})), wp)   # taken at 14.5 h
    MPPPImage(ZL0, mppp.load_config(dict(base, selection={"lmst_window_h": [9.0, 17.0]})), wp)
    man = process_images([NLF, ZL0], tmp_path / "out", mppp.load_config(dict(base, selection={
        "max_boresight_elevation_deg": None, "max_saturated_fraction": 0.05, "lmst_window_h": [9.0, 17.0]})), wp,
        progress=False, workers=1)
    assert man["n_processed"] == 1 and len(man["skipped"]) == 1 and "saturated" in man["skipped"][0]["reason"]


@needs_data
def test_worker_pool_matches_sequential(tmp_path):
    import mppp
    from mppp.process import process_images
    wp = synthetic_waypoints()
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"]}})
    seq = process_images([ZL0, NLF], tmp_path / "seq", cfg, wp, progress=False, workers=1)
    par = process_images([ZL0, NLF], tmp_path / "par", cfg, wp, progress=False, workers=2)
    assert seq["n_processed"] == par["n_processed"] == 2 and not par["failed"]
    assert [m["source_product"] for m in seq["images"]] == [m["source_product"] for m in par["images"]]   # selection order
    for a, b in zip(seq["images"], par["images"]):
        assert a["pose"]["C_enu_m"] == b["pose"]["C_enu_m"] and a["saturated_fraction"] == b["saturated_fraction"]


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


def test_exposure_rules():
    from mppp.image import ExposureImage, MPPPImage
    from mppp.process import _exposure_rule
    cfg = {"selection": {"max_exposure_ms": 40.0, "max_centre_tint": 1.2, "exposure_filter_families": ["N"]}}
    assert _exposure_rule(cfg, {"source_product": "NLF_x.IMG", "exposure_duration_ms": 54.06}).startswith("exposure 54.1")
    assert _exposure_rule(cfg, {"source_product": "NLF_x.IMG", "exposure_duration_ms": 18.3}) is None
    assert _exposure_rule(cfg, {"source_product": "ZLF_x.IMG", "exposure_duration_ms": 80.0}) is None
    assert _exposure_rule(cfg, {"source_product": "NRF_x.IMG", "exposure_duration_ms": 10.0, "centre_tint": 1.7})
    im = MPPPImage.__new__(MPPPImage)
    im.fn = SimpleNamespace(stem="NLF_0658_0725353794_270RAD_N0320274NCAM08111_0A0095J01")
    im.config = cfg
    im.label = {"INSTRUMENT_STATE_PARMS": {"EXPOSURE_DURATION": 54.061}}
    with pytest.raises(ExposureImage):
        im._check_exposure()
    # the blue disk: the centre bluer than the edge
    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
    rad = np.ones((h, w, 3), np.float32)
    rad[..., 2] = np.where(r < 0.5, 1.8, 1.0)
    with pytest.raises(ExposureImage):
        im._check_centre_tint(rad, np.ones((h, w), np.uint8))
    rad[..., 2] = 1.0
    im._check_centre_tint(rad, np.ones((h, w), np.uint8))
    assert abs(im.centre_tint - 1.0) < 1e-6


def test_zcam_camera_temperature_is_head_fpa(monkeypatch):
    from mppp import image as I
    ns = types.SimpleNamespace(camera_model_label=types.SimpleNamespace(meta={"interpolation": "ZOOM"}),
                               fn=types.SimpleNamespace(stem="ZL0_0684_0727673848_348RAD_N0321174ZCAM07114_0340LMA01"),
                               label={"INSTRUMENT_STATE_PARMS": {
                                   "INSTRUMENT_TEMPERATURE_NAME": ["DEA", "HEAD_FPA", "HEAD_HTR_1", "HEAD_HTR_2"],
                                   "INSTRUMENT_TEMPERATURE": [30.76, -13.445, -14.77, -14.62]}})
    assert I.MPPPImage.camera_temperature_degC.fget(ns) == pytest.approx(-13.445)
    ns.label["INSTRUMENT_STATE_PARMS"]["INSTRUMENT_TEMPERATURE_NAME"] = ["DEA"]
    assert I.MPPPImage.camera_temperature_degC.fget(ns) is None


def _paths(tmp_path, n):
    return [tmp_path / f"NLF_0100_07000000{i:02d}_000RAD_N0040136NCAM00500_0A0195J01.IMG" for i in range(n)]


def test_interrupted_first_run_is_completed_not_trimmed(tmp_path):
    from mppp.process import filter_to_existing
    out = tmp_path / "processed"
    (out / "images_png8").mkdir(parents=True)
    paths = _paths(tmp_path, 10)
    for p in paths[:3]:                                  # a first run stopped after 3 images: no manifest yet
        (out / "images_png8" / f"{p.stem}.png").write_bytes(b"x")
    kept, rep = filter_to_existing(paths, out, "PNG8")
    assert len(kept) == 10 and rep["n_removed"] == 0 and rep["never_processed"] == 7


def test_deleted_images_stay_out_and_new_ones_come_in(tmp_path):
    from mppp.process import filter_to_existing
    out = tmp_path / "processed"
    (out / "images_png8").mkdir(parents=True)
    paths = _paths(tmp_path, 10)
    done = paths[:6]
    (out / "mppp_manifest_v0p43.json").write_text(json.dumps({"images": [{"source_product": str(p)} for p in done]}))
    for p in done[:4]:                                   # the user deleted 2 of the 6 processed images
        (out / "images_png8" / f"{p.stem}.png").write_bytes(b"x")
    kept, rep = filter_to_existing(paths, out, "PNG8")
    assert {p.stem for p in kept} == {p.stem for p in paths[:4] + paths[6:]}
    assert rep["n_removed"] == 2 and rep["never_processed"] == 4
    assert rep["removed"] == sorted(p.name for p in paths[4:6])


def test_mask_alpha_transparency():
    """v0p51: STORE_MASK_IN_ALPHA = 0.5 -> masked pixels half transparent (alpha 128 of 255), included opaque."""
    from types import SimpleNamespace
    from mppp.image import MPPPImage
    from mppp.process import alpha_transparency
    assert alpha_transparency(True) == 1.0 and alpha_transparency(False) == 0.0 and alpha_transparency(0.5) == 0.5
    with pytest.raises(ValueError):
        alpha_transparency(1.5)
    im = SimpleNamespace(image_int8=np.zeros((2, 2, 3), np.uint8), image_int16=np.zeros((2, 2, 3), np.uint16),
                         mask=np.array([[1, 0], [0, 1]], np.uint8))
    a = MPPPImage.rgba(im, 8, 0.5)[..., 3]
    assert a[0, 0] == 255 and a[0, 1] == 128
    assert MPPPImage.rgba(im, 8)[..., 3][0, 1] == 0 and MPPPImage.rgba(im, 16, 0.5)[..., 3][0, 1] == 32768
