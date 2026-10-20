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
    assert Settings().nearby_from_first_sol is False
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
    assert _zero_terms(s) == (("b1", "b2"), None) and _refine_tangential(s) is True   # zcam-v2: p1, p2 refined like the Navcam
    assert _zero_terms(s.replace(zcam_tangential="zero")) == (("b1", "b2"), ("p1", "p2", "b1", "b2"))
    assert _zero_terms(s.replace(zcam_k3="zero"))[1] == ("b1", "b2", "k3")
    w = wanted_project_settings(s, tmp_path)
    assert w["zcam_zero_terms"] is None and w["navcam_distortion"] == "fisheye_tangential"
    wz = wanted_project_settings(s.replace(include_zcam=True), tmp_path)
    assert wz["zcam_zero_terms"] == ["p1", "p2", "b1", "b2"] and wz["zcam_low_backlash_sha256"]
    from mppp.pipeline import _check_settings
    with pytest.raises(ValueError, match="navcam_rig_refine"):
        _check_settings(s.replace(navcam_rig_refine="sometimes"))
    with pytest.raises(ValueError, match="navcam_pp"):
        _check_settings(s.replace(navcam_pp="zero"))


def test_pipeline_defaults_v0p61():
    from mppp.pipeline import Settings
    s = Settings()
    assert s.store_mask_in_alpha == 0.5 and s.zcam_k3 == "label" and s.zcam_backlash == "list_split"
    assert s.zcam_low_backlash is None and s.zcam_focus_line is True and s.thermal_after_stage1 is True
    assert s.no_mask_inference_at == ["S032D1184"]


def test_pipeline_and_script_defaults_v0p64(tmp_path):
    from mppp.pipeline import Settings, resolve_site
    s = Settings(scapes_root=tmp_path)
    assert s.sift["num_octaves"] is None and s.pose_guided is True
    assert resolve_site("rockytop", s)[1].sift["num_octaves"] == 4          # 6 with Mastcam-Z, else COLMAP's 4
    assert resolve_site("rockytop", s.replace(include_zcam=True))[1].sift["num_octaves"] == 6


def test_run_key_includes_the_alignment_rules(monkeypatch):
    from mppp import pipeline
    s = pipeline.Settings()
    k1 = pipeline.run_key(s)
    p1 = pipeline.process_key(s)
    monkeypatch.setattr(pipeline, "ALIGN_RULES", pipeline.ALIGN_RULES + 1)
    assert pipeline.run_key(s) != k1                          # a new rules version reruns the alignments
    assert pipeline.process_key(s) == p1                      # ... but not the processing
    assert pipeline.ALIGN_RULES - 1 >= 65


def test_pipeline_defaults_v0p65():
    from mppp.pipeline import Settings
    assert Settings().navcam_aspect == "hold"


def test_selection_defaults_v0p65p1():
    """0.65.1: Navcam tiles below 1/5 of the frame left out (was 1/4), LMST window 6-18 h, saturated fraction 0.2."""
    from mppp.pipeline import PROCESS_RULES, Settings
    from mppp.sfm.project import NAVCAM_MIN_FRAME_FRACTION
    from mppp.config import default_config
    sel = default_config()["selection"]
    assert sel["lmst_window_h"] == [6.0, 18.0] and sel["max_saturated_fraction"] == 0.2
    assert NAVCAM_MIN_FRAME_FRACTION == pytest.approx(1 / 3) and PROCESS_RULES == 4     # v0p85: 1/3
    s = Settings()
    assert s.lmst_window_h == (6.0, 18.0) and s.max_saturated_fraction == 0.2
    assert s.navcam_min_frame_fraction == pytest.approx(1 / 3)


def test_run_key_holds_the_camera_models(monkeypatch):
    import mppp.calib
    import mppp.calib.models
    from mppp import pipeline
    assert pipeline.ALIGN_RULES >= 73
    s = pipeline.Settings()
    k1 = pipeline.run_key(s)
    for mod in (mppp.calib, mppp.calib.models):
        monkeypatch.setattr(mod, "models_fingerprint", lambda folder=None: "other")
    assert pipeline.run_key(s) != k1
    assert pipeline.process_key(s) == pipeline.process_key(pipeline.Settings())


def test_pipeline_settings_v0p73():
    from mppp.pipeline import Settings
    s = Settings()
    assert s.zcam_bin_refine == "focal_pp" and s.zcam_backlash == "list_split"
    assert s.zcam_line_params == {"ZL": ["fx", "fy", "k1", "k2", "p1", "p2"], "ZR": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]}
    assert s.zcam_low_backlash_found == "auto"


def test_resolved_site_carries_the_visits(tmp_path):
    from mppp.pipeline import Settings, resolve_site
    extra = {"sid2": {"sols": [[361, 378], [91, 101]], "zcam": [34]}, "one": {"sols": [700, 712], "zcam": [34]}}
    s = Settings(scapes_root=tmp_path, sites_extra=extra)
    st, _ = resolve_site("sid2", s)
    assert st.sol_ranges == [(361, 378), (91, 101)] and st.work.name == "mars2020_sol_0361_sid2"
    assert resolve_site("one", s)[0].sol_ranges == [(700, 712)]
    assert resolve_site(tmp_path / "mars2020_sol_0361_sid2_zcam", s)[0].zooms == [34]


def test_pipeline_v0p80_settings(tmp_path):
    from mppp.pipeline import Settings, _rig_refine, wanted_project_settings
    s = Settings()
    assert s.navcam_pp == "hold" and s.round1_hold is True and s.rig_translation_sigma_m == 0.005
    assert _rig_refine(s) == "rotation" and _rig_refine(s.replace(navcam_rig_refine="full")) is True
    w = wanted_project_settings(s, tmp_path)
    assert w["navcam_k4"] == "consensus" and w["navcam_pp"] == "hold"


def test_pipeline_carries_the_v0p82_settings(tmp_path):
    from mppp.pipeline import ALIGN_RULES, Settings, wanted_project_settings
    s = Settings()
    assert s.min_included_fraction is None and s.no_position_prior_at == [] and s.prior_gauge_recentre is True
    assert s.prior_sigma_m == 1.0 and wanted_project_settings(s, tmp_path)["prior_sigma_m"] == [1.0, 1.0, 1.0]
    assert ALIGN_RULES >= 82


def test_one_third_rule_is_about_subframes():
    """Christian (19 Oct): the 1/3 rule leaves out sub-frames covering less than 1/3 of the frame - not images whose
    mask keeps less than 1/3 (v0p82's reading, now off)."""
    from mppp.pipeline import ALIGN_RULES, Settings
    from mppp.sfm.project import NAVCAM_MIN_FRAME_FRACTION
    from mppp.sfm.workdir import MIN_INCLUDED_FRACTION
    assert NAVCAM_MIN_FRAME_FRACTION == pytest.approx(1 / 3) and MIN_INCLUDED_FRACTION is None and ALIGN_RULES >= 85
    s = Settings()
    assert s.min_included_fraction is None and s.subframe_families == ("N",)
