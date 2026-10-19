"""The pipeline as functions (mppp.pipeline): settings, run keys, selection and processing on the two test IMGs."""
import json
import shutil
from pathlib import Path

import pytest

from conftest import NLF, ZL0

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("MPPP_PDS", "MPPP_SCAPES", "MPPP_GPU_PYTHON", "COLMAP_BAT"):
        monkeypatch.delenv(k, raising=False)


def test_settings_defaults(clean_env):
    from mppp.pipeline import Settings, setting_names
    s = Settings()
    assert s.pds_dir is None and s.scapes_root is None and s.gpu_python is None and s.colmap_bat is None
    assert s.source == "pds" and s.variant == "" and s.add_nearby_waypoints == 10 and s.include_zcam is False
    assert s.navcam_intrinsics == "auto" and s.navcam_rig_refine == "refine" and s.navcam_pp == "hold"
    assert s.schedule == ((24.0, 10.0, 8.0), (12.0, 4.0, 4.0), (8.0, 2.0, 2.0)) and s.attitude_prior_deg == 2.0
    assert s.match_mode == "lightglue" and s.match["max_ratio"] == 0.9 and s.match["max_keypoints"] == 8192 and s.match_block_size == 100
    assert s.thermal_bins_deg == 5 and s.thermal_fine_range == (-25.0, -5.0)
    assert len(setting_names()) > 80 and "ATTITUDE_PRIOR_DEG" in setting_names(upper=True)
    for gone in ("navcam_distortion", "navcam_k4", "allow_old_form", "include_zcam34", "run_key", "work_dir"):
        assert gone not in setting_names()
    d = s.to_dict()
    assert d["schedule"] == [[24.0, 10.0, 8.0], [12.0, 4.0, 4.0], [8.0, 2.0, 2.0]] and json.dumps(d)
    assert "pds_dir" not in s.key_dict() and "attitude_prior_deg" in s.key_dict()


def test_settings_from_environment(monkeypatch):
    from mppp.pipeline import Settings
    monkeypatch.setenv("MPPP_PDS", "/pds")
    monkeypatch.setenv("MPPP_SCAPES", "/scapes")
    monkeypatch.setenv("MPPP_GPU_PYTHON", "/envs/gpu/python")
    monkeypatch.setenv("COLMAP_BAT", "/tools/COLMAP.bat")
    s = Settings()
    assert s.pds_dir == Path("/pds") and s.scapes_root == Path("/scapes")
    assert s.gpu_python == "/envs/gpu/python" and s.colmap_bat == "/tools/COLMAP.bat"
    assert Settings(pds_dir="/other").pds_dir == Path("/other")


def test_settings_merge_json_and_set(tmp_path, clean_env):
    from mppp.pipeline import Settings, load_settings, normalize
    from mppp.runner import parse_set
    work = tmp_path / "mars2020_sol_0460_rockytop"
    work.mkdir()
    (work / "mppp_settings.json").write_text(json.dumps({"ATTITUDE_PRIOR_DEG": 1.0, "thermal_bins_deg": None,
                                                         "_comment": "ignored"}))
    f = tmp_path / "more.json"
    f.write_text(json.dumps({"thermal_bins_deg": 10, "exclude": ["S032D1184"]}))
    sets = parse_set(["exclude=[\"sol:658\"]", "schedule=((24,10,8),(8,2,2))", "match=dict(max_ratio=0.85)",
                      "navcam_intrinsics=hold"])
    s = load_settings(work, "rockytop", f, sets)
    assert s.attitude_prior_deg == 1.0 and s.thermal_bins_deg == 10 and s.exclude == ["sol:658"]
    assert s.schedule == [(24, 10, 8), (8, 2, 2)] and s.match == {"max_ratio": 0.85} and s.navcam_intrinsics == "hold"
    assert load_settings(work, "rockytop").thermal_bins_deg is None
    assert load_settings(None, "rochette").nearby_from_first_sol is True       # the site's settings in sites.json
    assert load_settings(None, "rockytop").no_mask_inference_at == ["S032D1184"]
    with pytest.raises(KeyError, match="unknown setting"):
        normalize({"NAVCAM_DISTORTION": "rational"})
    assert Settings().replace(INCLUDE_ZCAM34=True).include_zcam is True
    assert Settings(hold_cameras=["ZL034_f0"]).hold_cameras == ("ZL034_f0",)


def test_run_key_stability(tmp_path, clean_env, monkeypatch):
    from mppp import pipeline
    from mppp.pipeline import Settings, process_key, run_key
    a, b = Settings(), Settings()
    assert run_key(a) == run_key(b) and len(run_key(a)) == 16
    assert run_key(a.replace(pds_dir=tmp_path, workers=1, gpu_python="x")) == run_key(a)     # machine settings: no
    assert run_key(a.replace(attitude_prior_deg=1.0)) != run_key(a)
    assert run_key(a.replace(variant="v")) != run_key(a) and run_key(a.replace(source="processed")) != run_key(a)
    assert run_key(a, "processed", "v") == run_key(a.replace(source="processed", variant="v"))
    assert process_key(a) == process_key(b) and process_key(a) != run_key(a)
    assert process_key(a.replace(variant="v")) == process_key(a)
    import mppp.calib
    import mppp.calib.models
    k = run_key(a)
    for mod in (mppp.calib, mppp.calib.models):
        monkeypatch.setattr(mod, "models_fingerprint", lambda folder=None: "other")
    assert run_key(a) != k and process_key(a) == process_key(b)      # new camera models rerun the alignments only
    monkeypatch.setattr(pipeline, "ALIGN_RULES", pipeline.ALIGN_RULES + 1)
    assert run_key(a) != k


def test_resolve_site(tmp_path, clean_env):
    from mppp.pipeline import Settings, resolve_site
    s = Settings(scapes_root=tmp_path)
    st, s2 = resolve_site("rockytop", s)
    assert st.name == "rockytop" and st.sol_ranges == [(460, 530)] and st.zooms == [] and not st.zcam
    assert st.work == tmp_path / "mars2020_sol_0460_rockytop" and s2.sift["num_octaves"] == 4
    st, s2 = resolve_site("rockytop", s.replace(include_zcam=True))
    assert st.work.name == "mars2020_sol_0460_rockytop_zcam" and st.zooms == [34] and s2.sift["num_octaves"] == 6
    st, s2 = resolve_site(tmp_path / "mars2020_sol_0460_rockytop_zcam", s.replace(zcam_zooms="all"))
    assert st.zooms == [34, 48, 63, 79, 110] and s2.include_zcam is True
    with pytest.raises(KeyError, match="not in"):
        resolve_site("nowhere", s)
    with pytest.raises(ValueError, match="scapes root"):
        resolve_site("rockytop", Settings())
    with pytest.raises(ValueError, match="zcam_zooms"):
        resolve_site("rockytop", s.replace(include_zcam=True, zcam_zooms=[12]))
    extra = s.replace(sites_extra={"mine": (700, 712)})
    assert resolve_site("mine", extra)[0].work.name == "mars2020_sol_0700_mine"


@pytest.fixture
def pds(tmp_path):
    d = tmp_path / "pds" / "sol0709"
    d.mkdir(parents=True)
    for f in (NLF, ZL0):
        shutil.copy(f, d / f.name)
    return tmp_path / "pds"


def _test_settings(tmp_path, pds):
    from mppp.pipeline import Settings
    # the Navcam test product is a Sun-pointing SAPP tile: the selection rules are off for it
    return Settings(pds_dir=pds, scapes_root=tmp_path / "scapes", sites_extra={"sol709": (709, 709)},
                    add_nearby_waypoints=0, include_zcam=True, zcam_zooms=[34], sequence_prefixes=("NCAM", "SAPP"),
                    navcam_min_frame_fraction=None, sky_elevation_deg=None, lmst_window_h=None,
                    max_saturated_fraction=None, max_exposure_ms=None, max_centre_tint=None, infer_mask=False,
                    workers=1)


def test_select_images_on_a_pds_folder(tmp_path, pds, clean_env):
    from mppp.pipeline import select_images, select_products
    s = _test_settings(tmp_path, pds)
    paths = select_images("sol709", s)
    assert sorted(p.name for p in paths) == sorted([NLF.name, ZL0.name])
    assert select_images("sol709", s.replace(include_zcam=False)) == [pds / "sol0709" / NLF.name]
    assert select_images("sol709", s.replace(sequence_prefixes=("NCAM",), include_zcam=False)) == []
    paths, sel = select_products("sol709", s)
    assert sel["site"] == "sol709" and sel["sol_ranges"] == [[709, 709]] and sel["n_exposure_products"] == 2
    with pytest.raises(ValueError, match="PDS archive"):
        select_images("sol709", s.replace(pds_dir=None))


def test_process_site_end_to_end(tmp_path, pds, clean_env):
    from mppp.pipeline import PROCESS_DONE, load_images, process_key, process_site, processed_done, resolve_site
    from mppp.sfm.workdir import load_manifest
    s = _test_settings(tmp_path, pds)
    processed = process_site("sol709", s)
    work = tmp_path / "scapes" / "mars2020_sol_0709_sol709_zcam"
    assert processed == work / "processed" and (processed / "images_png8").is_dir()
    man, mf = load_manifest(processed)
    assert man["n_processed"] == 2 and not man["failed"] and mf.name.startswith("mppp_manifest_v")
    assert {m["outputs"]["PNG8"].split("/")[0] for m in man["images"]} == {"images_png8"}
    assert man["provenance"]["site"] == "sol709"
    done = json.loads((processed / PROCESS_DONE).read_text())
    key = process_key(resolve_site("sol709", s)[1])             # the key is of the resolved settings (zooms, octaves)
    assert done["images"] == 2 and done["run_key"] == key and processed_done(work, key)
    assert process_key(resolve_site(work, s)[1]) == key
    assert processed_done(work, "other") is None
    # the same folder as a WORK path, reusing what is processed
    assert process_site(work, s) == processed
    # the manifest images that go into an alignment: exclusions and zoom filters
    ims = load_images(work, s.replace(exclude=["ZL0_0709_*"]), zooms=[34])
    assert [m["source_product"] for m in ims] == [NLF.name]
    assert len(load_images(work, s, zooms=[])) == 1 and len(load_images(work, s, zooms=[34])) == 2


def test_align_refuses_a_folder_without_processed(tmp_path, clean_env):
    from mppp.pipeline import Settings, align, run_site
    work = tmp_path / "mars2020_sol_0460_rockytop"
    work.mkdir()
    with pytest.raises(FileNotFoundError, match="processed/"):
        align(work, Settings())
    with pytest.raises(FileNotFoundError, match="processed/"):
        run_site(work, Settings(source="processed"))
    with pytest.raises(ValueError, match="stop_after"):
        align(work, Settings(), stop_after="nowhere")


def test_processing_config_and_project_settings(tmp_path, clean_env):
    from mppp.pipeline import Settings, processing_config, wanted_project_settings, _zero_terms, _refine_tangential
    cfg = processing_config(Settings())
    assert cfg["export"]["formats"] == ["PNG8"] and cfg["export"]["store_mask_in_alpha"] == 0.5
    assert cfg["masking"]["skip_inference_at"] == ["S032D1184"] and cfg["masking"]["infer_mask"] is True
    assert cfg["selection"]["lmst_window_h"] == [6.0, 18.0] and cfg["color"]["brightness_by_family"] == {"N": 0.9}
    assert processing_config(Settings(infer_mask=False, mask_checkpoint="x.pt"))["masking"]["checkpoint"] == "x.pt"
    s = Settings()
    assert _zero_terms(s) == (("b1", "b2"), ("p1", "p2", "b1", "b2")) and _refine_tangential(s) == {"N": True, "Z": False}
    assert _zero_terms(s.replace(zcam_k3="zero"))[1] == ("p1", "p2", "b1", "b2", "k3")
    w = wanted_project_settings(s, tmp_path)
    assert w["zcam_zero_terms"] is None and w["navcam_distortion"] == "fisheye_tangential"
    wz = wanted_project_settings(s.replace(include_zcam=True), tmp_path)
    assert wz["zcam_zero_terms"] == ["p1", "p2", "b1", "b2"] and wz["zcam_low_backlash_sha256"]
    from mppp.pipeline import _check_settings
    with pytest.raises(ValueError, match="navcam_rig_refine"):
        _check_settings(s.replace(navcam_rig_refine="sometimes"))
    with pytest.raises(ValueError, match="navcam_pp"):
        _check_settings(s.replace(navcam_pp="zero"))
