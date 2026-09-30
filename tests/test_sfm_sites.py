"""Site definitions and WORK folders (mppp.sfm.sites, workdir). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
from mppp.sfm.sites import SITES, check_sites, discover_scapes, scan_scapes, site_label
import pytest
from pathlib import Path


def _site(root, name, sols, reconstruction=True, error_input=True, model=True, n=None):
    col = root / name / "colmap"
    col.mkdir(parents=True)
    ims = [{"name": f"i{k}.png", "sol": s, "instrument": "NL" if k % 2 == 0 else "NR", "station": f"S{k // 4}"}
           for k, s in enumerate(sols)]
    settings = {"reconstruction": {"path": "sparse\\cahv_ba"}} if reconstruction else {}
    (col / "project.json").write_text(json.dumps({"images": ims, "settings": settings}))
    if model:
        (col / "sparse" / "cahv_ba").mkdir(parents=True)
        (col / "sparse" / "cahv_ba" / "images.bin").write_bytes(b"x")
    if error_input:
        (col / "error_input").mkdir()
        (col / "error_input" / "summary.json").write_text("{}")
    (col / "health").mkdir()
    (col / "health" / "health.json").write_text(json.dumps({"verdict": "pass", "mppp_version": "0.35.2"}))


def test_labels_and_site_checks():
    assert site_label("threeforks_south") == "Three Forks South" and site_label("van_zyl") == "Van Zyl"
    assert site_label("rockytop") == "Rockytop"
    probs = check_sites(SITES)
    assert probs == []
    assert SITES["hippo_pools"] == (1947, 1955) and SITES["origny"] == (1781, 1813) and SITES["origny_large"] == (1765, 1813) and SITES["olifants"] == (1880, 1889)


def test_discover_scapes(tmp_path):
    _site(tmp_path, "rockytop_colmap", [470] * 12)
    _site(tmp_path, "van_zyl_colmap", [60] * 12)
    _site(tmp_path, "olifants_colmap", [1780] * 12)                 # the renamed site's block: left out
    _site(tmp_path, "pearce_canyon_colmap", [1190] * 12, reconstruction=False)   # run in progress
    _site(tmp_path, "sid_colmap", [365] * 12, error_input=False)
    _site(tmp_path, "rockytop_colmap_nav_zcam34", [470] * 12)
    _site(tmp_path, "tiny_colmap", [5] * 4)
    (tmp_path / "camera_analysis").mkdir()
    d = discover_scapes(tmp_path, verbose=False)
    assert list(d) == ["Van Zyl", "Sid", "Rockytop", "Rockytop N+Z34"]      # sol order
    assert d["Rockytop"] == tmp_path / "rockytop_colmap"
    rows = {r["label"]: r for r in scan_scapes(tmp_path)}
    assert "no image in the site's sols 1880-1889" in rows["Olifants"]["reason"] and "origny" in rows["Olifants"]["reason"]
    assert "in progress" in rows["Pearce Canyon"]["reason"] and "< 10" in rows["Tiny"]["reason"]
    e = discover_scapes(tmp_path, require_error_input=True, include_zcam=False, exclude=["van_zyl"], verbose=False)
    assert list(e) == ["Rockytop"]


def _meta(stem, sol, site, drive, seq):
    return {"source_product": stem + ".IMG", "outputs": {"PNG8": f"images_png8/{stem}.png"},
            "filename": {"stem": stem, "sol": sol, "site": site, "drive": drive, "sequence": seq}}


def test_sites_json_is_the_site_list():
    from mppp.sfm import sites as S
    t = S.load_site_table()
    assert len(t["sites"]) >= 32 and S.SITES["rockytop"] == (461, 530) and S.SITES["south_arm"] == (1408, 1412)
    for g, members in t["groups"].items():
        assert members and all(m in t["sites"] for m in members), g
    for k, v in t["sites"].items():
        assert v["sols"][0] <= v["sols"][1] and isinstance(v.get("settings", {}), dict), k
    assert {"rockytop", "south_arm", "taylorfjellet"} <= set(S.zcam34_sites(t)) and "van_zyl" not in S.zcam34_sites(t)
    assert all(isinstance(v.get("zcam34", False), bool) for v in t["sites"].values())
    assert S.parse_work_folder("D:/x/south_arm_colmap_nav_zcam34") == ("south_arm", True)     # before v0p50
    assert S.parse_work_folder("D:/x/south_arm_colmap_zcam34") == ("south_arm", True)
    assert S.parse_work_folder("D:/x/sid_colmap") == ("sid", False)
    assert S.work_folder("D:/r", "sid", True).name == "sid_colmap_zcam34"


def test_sites_file_override(tmp_path):
    from mppp.sfm.sites import load_sites
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"sites": {"a": {"sols": [5, 9]}, "b": [1, 2]}}))
    assert load_sites(f) == {"a": (5, 9), "b": (1, 2)}
    f.write_text(json.dumps({"sites": {"a": {"sol": [5, 9]}}}))
    with pytest.raises(ValueError):
        load_sites(f)


def test_exclusions_and_existing(tmp_path):
    from mppp.sfm.workdir import apply_exclusions, keep_existing, read_exclusions
    ms = [_meta("NLF_0658_0725353794_270RAD_N0320274NCAM08111_0A0095J01", 658, 32, 274, "NCAM08111"),
          _meta("NRF_0658_0725353794_270RAD_N0320274NCAM08111_0A0095J01", 658, 32, 274, "NCAM08111"),
          _meta("ZR0_0690_0728197573_035RAD_N0321184ZCAM08692_0340LMA01", 690, 32, 1184, "ZCAM08692"),
          _meta("NLF_0700_0729000000_000RAD_N0330000NCAM00200_0A0195J01", 700, 33, 0, "NCAM00200")]
    f = tmp_path / "exclude_images.txt"
    f.write_text("# comment\nS032D1184   # a station\n\nseq:ncam08111\nsol:999\n")
    kept, removed = apply_exclusions(ms, read_exclusions(f))
    assert [m["filename"]["sol"] for m in kept] == [700]
    assert {"image": None, "entry": "sol:999"} in removed
    assert sum(r["image"] is not None for r in removed) == 3
    kept, _ = apply_exclusions(ms, ["sol:650-660", "ZR0_0690_*"])
    assert [m["filename"]["sol"] for m in kept] == [700]
    kept, _ = apply_exclusions(ms, ["NLF_0700_0729000000_000RAD_N0330000NCAM00200_0A0195J01.IMG"])
    assert len(kept) == 3
    (tmp_path / "images_png8").mkdir()
    (tmp_path / "images_png8" / (ms[0]["filename"]["stem"] + ".png")).write_bytes(b"x")
    kept, missing = keep_existing(ms, tmp_path)
    assert len(kept) == 1 and len(missing) == 3


def test_project_dir_seed_and_settings(tmp_path):
    from mppp.sfm.workdir import load_settings, project_dir, seed_variant
    assert project_dir(tmp_path).name == "colmap" and project_dir(tmp_path, "rational").name == "colmap_rational"
    with pytest.raises(ValueError):
        project_dir(tmp_path, "a b")
    base = tmp_path / "colmap"
    base.mkdir()
    for n in ("features.db", "features.json", "database.db", "database_matches.json"):
        (base / n).write_text(n)
    out = seed_variant(project_dir(tmp_path, "v"), base)
    assert out["copied"] == ["features.db", "features.json"] and out["reuse_matches_from"] == [str(base / "database.db")]
    assert (tmp_path / "colmap_v" / "features.json").read_text() == "features.json"
    s = tmp_path / "mppp_settings.json"
    s.write_text(json.dumps({"_comment": "x", "ATTITUDE_PRIOR_DEG": 1.0}))
    assert load_settings(s) == {"ATTITUDE_PRIOR_DEG": 1.0}


ROOT = Path(__file__).resolve().parents[1]


def test_check_sites(tmp_path):
    import importlib.util
    from mppp.sfm.sites import SITES_FILE, validate_site_table
    err, warn = validate_site_table()
    assert err == [] and not any("overlap" in w for w in warn)           # nested blocks are not flagged
    bad = tmp_path / "s.json"
    txt = SITES_FILE.read_text(encoding="utf-8").replace('"settings": {}},', '"settings": {}}', 1)
    bad.write_text(txt, encoding="utf-8")
    err, _ = validate_site_table(bad)
    assert len(err) == 1 and "comma missing" in err[0] and "line" in err[0]
    d = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    d["sites"]["my_site"] = {"sols": [712, 700], "settings": {"ATTITUDE_PRIOR_DEGG": 1}, "no_mask_inference_at": ["x"]}
    d["groups"]["mine"] = ["my_site", "nowhere"]
    bad.write_text(json.dumps(d), encoding="utf-8")
    err, warn = validate_site_table(bad, known_settings={"ATTITUDE_PRIOR_DEG"})
    txt = "\n".join(err + warn)
    assert "end before they start" in txt and "'nowhere'" in txt and "ATTITUDE_PRIOR_DEGG" in txt and "x" in txt
    spec = importlib.util.spec_from_file_location("check_sites", ROOT / "scripts" / "check_sites.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    assert cs.main([]) == 0 and cs.main(["--sites-file", str(bad)]) == 1
    assert "ATTITUDE_PRIOR_DEG" in cs.notebook03_settings()


def test_zcam34_field_and_folder_names(tmp_path):
    """v0p50: "zcam34": true/false per site; folders <site>_colmap_zcam34 (the old _colmap_nav_zcam34 still found)."""
    import json
    from mppp.sfm import sites as S
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"sites": {"a": {"sols": [1, 2], "zcam34": True}, "b": {"sols": [3, 4], "zcam34": "False"},
                                       "c": {"sols": [3, 4], "z34": "True"}}}))
    err, warn = S.validate_site_table(f)
    assert any("'b'" in e and "true or false" in e for e in err) and any("'z34' is now" in e for e in err)
    assert any("same sols" in w for w in warn)
    assert S.zcam34_sites(S.load_site_table(f)) == ["a"]
    assert S.work_folder(tmp_path, "a", True).name == "a_colmap_zcam34"
    (tmp_path / "a_colmap_nav_zcam34").mkdir()
    assert S.work_folder(tmp_path, "a", True).name == "a_colmap_nav_zcam34"      # the old folder while no new one
    (tmp_path / "a_colmap_zcam34").mkdir()
    assert S.work_folder(tmp_path, "a", True).name == "a_colmap_zcam34"
    err, _ = S.validate_site_table()
    assert err == []
