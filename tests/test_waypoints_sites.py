"""Site definitions and WORK folders (mppp.waypoints.sites, workdir). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
from mppp.waypoints.sites import check_sites, discover_scapes, load_sites, scan_scapes, site_label

SITES = load_sites()
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
    assert SITES["hippo_pools"] == (1948, 2009) and SITES["origny"] == (1781, 1813) and SITES["origny_large"] == (1765, 1813) and SITES["olifants"] == (1880, 1889)


def test_discover_scapes(tmp_path):
    _site(tmp_path, "mars2020_sol_0461_rockytop_colmap", [470] * 12)
    _site(tmp_path, "mars2020_sol_0049_van_zyl_colmap", [60] * 12)
    _site(tmp_path, "mars2020_sol_1880_olifants_colmap", [1780] * 12)      # the renamed site's block: left out
    _site(tmp_path, "mars2020_sol_1183_pearce_canyon_colmap", [1190] * 12, reconstruction=False)   # run in progress
    _site(tmp_path, "mars2020_sol_0361_sid_colmap", [365] * 12, error_input=False)
    _site(tmp_path, "mars2020_sol_0461_rockytop_colmap_zcam", [470] * 12)
    _site(tmp_path, "mars2020_sol_0005_tiny_colmap", [5] * 4)
    _site(tmp_path, "rockytop_colmap_zcam34", [470] * 12)              # v0p53: names before v0p53 are not used
    (tmp_path / "camera_analysis").mkdir()
    d = discover_scapes(tmp_path, verbose=False)
    assert list(d) == ["Van Zyl", "Sid", "Rockytop", "Rockytop N+Z"]      # sol order
    assert d["Rockytop"] == tmp_path / "mars2020_sol_0461_rockytop_colmap"
    rows = {r["label"]: r for r in scan_scapes(tmp_path)}
    assert "no image in the site's sols 1880-1889" in rows["Olifants"]["reason"] and "origny" in rows["Olifants"]["reason"]
    assert "in progress" in rows["Pearce Canyon"]["reason"] and "< 10" in rows["Tiny"]["reason"]
    e = discover_scapes(tmp_path, require_error_input=True, include_zcam=False, exclude=["van_zyl"], verbose=False)
    assert list(e) == ["Rockytop"]


def _meta(stem, sol, site, drive, seq):
    return {"source_product": stem + ".IMG", "outputs": {"PNG8": f"images_png8/{stem}.png"},
            "filename": {"stem": stem, "sol": sol, "site": site, "drive": drive, "sequence": seq}}


def test_sites_json_is_the_site_list():
    from mppp.waypoints import sites as S
    t = S.load_site_table()
    assert len(t["sites"]) >= 32 and load_sites()["rockytop"][1] == 530 and load_sites()["south_arm"] == (1408, 1412)
    for g, members in t["groups"].items():
        assert (members or g in ("zcam79_consensus", "zcam110_consensus")) and all(m in t["sites"] for m in members), g
    for k, v in t["sites"].items():
        assert all(a <= b for a, b in S.sol_ranges(v["sols"])) and isinstance(v.get("settings", {}), dict), k
    assert {"rockytop", "south_arm", "taylorfjellet"} <= set(S.zcam34_sites(t)) and "van_zyl_overlook" not in S.zcam34_sites(t)
    assert all(isinstance(v.get("zcam"), list) and set(v["zcam"]) <= {34, 48, 63, 79, 110} for v in t["sites"].values())
    assert 34 in S.site_zooms(t["sites"]["chal_rocks"]) and "bright_angle" in S.zcam_sites(t, zoom=63)
    zc = set(t["groups"]["zcam_consensus"])
    assert all(S.site_zooms(t["sites"][m]) for m in zc)
    for g in ("ecam_consensus", "zcam34_consensus", "zcam48_consensus", "zcam63_consensus"):
        assert g in t["groups"]
    assert len(S.site_group("ecam_consensus")) == len(set(t["groups"]["ecam_consensus"]))
    assert S.parse_work_folder("D:/x/mars2020_sol_1408_south_arm_colmap_zcam") == ("south_arm", True)
    assert S.parse_work_folder("D:/x/mars2020_sol_0361_sid_chal_rocks_colmap") == ("sid_chal_rocks", False)
    assert S.parse_work_folder("D:/x/south_arm_colmap_zcam34") == ("south_arm", True)     # before v0p53 (by hand)
    assert S.parse_work_folder("D:/x/sid_colmap") == ("sid", False)
    assert S.work_folder("D:/r", "chal_rocks", True).name == "mars2020_sol_0091_chal_rocks_zcam"   # 0.93.3: sol 91 first (was 0361)
    assert S.work_folder("D:/r", "x", False, (7, 9)).name == "mars2020_sol_0007_x"


def test_sites_file_override(tmp_path):
    from mppp.waypoints.sites import load_sites
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
    from mppp.waypoints.sites import SITES_FILE, validate_site_table
    err, warn = validate_site_table()
    assert err == [] and not any("overlap" in w and "partly" not in w for w in warn)   # nested blocks are not flagged
    bad = tmp_path / "s.json"
    txt = SITES_FILE.read_text(encoding="utf-8").replace('"},\n  "', '"}\n  "', 1)     # v0p85: lines end with "notes"
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
    assert "ATTITUDE_PRIOR_DEG" in cs.pipeline_settings() and "attitude_prior_deg" in cs.pipeline_settings()


def test_zcam_field_and_folder_names(tmp_path):
    """v0p53: "zcam": [34, 48, 63, 110] per site (the older "zcam34": true is read, with a warning); folders
    mars2020_sol_<first sol>_<site>_colmap[_zcam]."""
    import json
    from mppp.waypoints import sites as S
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"sites": {"a": {"sols": [1, 2], "zcam": [34, 110]}, "b": {"sols": [3, 4], "zcam34": "False"},
                                       "c": {"sols": [3, 4], "z34": "True"}, "d": {"sols": [5, 6], "zcam": [35]},
                                       "e": {"sols": [7, 8], "zcam48": True}},
                             "groups": {"zcam63_consensus": ["a", "a"]}}))
    err, warn = S.validate_site_table(f)
    assert any("'b'" in e and "true or false" in e for e in err) and any("'z34' is now" in e for e in err)
    assert any("'d'" in e and "34, 48, 63, 79, 110" in e for e in err)
    assert any("'e'" in w and "replaced by 'zcam'" in w for w in warn)
    assert any("more than once" in w for w in warn) and any("no 63" in w for w in warn)
    t = S.load_site_table(f)
    assert S.zcam_sites(t) == ["a", "d", "e"] and S.zcam_sites(t, zoom=48) == ["e"] and S.zcam34_sites(t) == ["a"]
    assert S.work_folder(tmp_path, "a", True, (1, 2)).name == "mars2020_sol_0001_a_zcam"
    err, _ = S.validate_site_table()
    assert err == []


def test_discover_scapes_keeps_sites_with_the_same_label(tmp_path, monkeypatch):
    from mppp.waypoints import sites as S
    rows = [{"label": "Bell Island", "site": "bell_island", "folder": str(tmp_path / "a"), "ok": True},
            {"label": "Bell Island", "site": "bell_island_large", "folder": str(tmp_path / "b"), "ok": True},
            {"label": "Van Zyl", "site": "van_zyl", "folder": str(tmp_path / "c"), "ok": True},
            {"label": "Bell Island", "site": "x", "folder": str(tmp_path / "d"), "ok": False}]
    monkeypatch.setattr(S, "scan_scapes", lambda *a, **k: [dict(r) for r in rows])
    out = S.discover_scapes(tmp_path, verbose=False)
    assert set(out) == {"Bell Island", "Bell Island (bell_island_large)", "Van Zyl"}
    assert out["Bell Island (bell_island_large)"] == tmp_path / "b"


# sites with several visits (v0p75)


def test_sol_ranges_span_and_membership():
    from mppp.waypoints import sites as S
    assert S.sol_ranges([1, 2]) == [(1, 2)]
    assert S.sol_ranges([[361, 378], [91, 101]]) == [(361, 378), (91, 101)]
    assert S.sol_span([[361, 378], [91, 101]]) == (361, 378)          # the main visit names the folder; last of all
    assert S.sol_span([[91, 101], [361, 378]]) == (91, 378)
    assert S.in_sol_ranges(95, [[361, 378], [91, 101]]) and not S.in_sol_ranges(200, [[361, 378], [91, 101]])
    for bad in ([1], [1, 2, 3], [[1, 2], 3], "x", [[1, "a"]]):
        with pytest.raises(ValueError):
            S.sol_ranges(bad)
    assert S.check_sites({"a": [[5, 9], [20, 10]]}) and not S.check_sites({"a": [[5, 9], [10, 20]]})


def test_site_table_with_visits(tmp_path):
    from mppp.waypoints import sites as S
    f = tmp_path / "sites.json"
    f.write_text(json.dumps({"sites": {
        "sid": {"sols": [[361, 378], [91, 101]], "zcam": [34], "settings": {}},
        "one": {"sols": [700, 712], "zcam": [], "settings": {}}},
        "groups": {"ecam_consensus": ["sid", "one"]}}))
    assert S.load_sites(f) == {"sid": (361, 378), "one": (700, 712)}
    assert S.load_site_ranges(f) == {"sid": [(361, 378), (91, 101)], "one": [(700, 712)]}
    assert S.work_folder(tmp_path, "sid", True, [[361, 378], [91, 101]]).name == "mars2020_sol_0361_sid_zcam"
    err, warn = S.validate_site_table(f)
    assert not err and not warn
    f.write_text(json.dumps({"sites": {"sid": {"sols": [[361, 378], [370, 400]], "zcam": [34]}}, "groups": {}}))
    err, _ = S.validate_site_table(f)
    assert err and "overlap" in err[0]
    f.write_text(json.dumps({"sites": {"sid": {"sols": [[361, 378], [91]], "zcam": [34]}}, "groups": {}}))
    err, _ = S.validate_site_table(f)
    assert err and "list of such ranges" in err[0]


def test_scan_accepts_a_folder_of_the_second_visit(tmp_path):
    from mppp.waypoints import sites as S
    w = tmp_path / "mars2020_sol_0361_sid_colmap"
    (w / "colmap" / "sparse" / "m").mkdir(parents=True)
    (w / "colmap" / "sparse" / "m" / "images.bin").write_bytes(b"")
    ims = [{"sol": 95, "instrument": "NL", "station": "S003D0100"}] * 12
    (w / "colmap" / "project.json").write_text(json.dumps({"images": ims, "settings": {"reconstruction": {"path": "sparse/m"}}}))
    rows = S.scan_scapes(tmp_path, {"sid": [[361, 378], [91, 101]]})
    assert rows[0]["ok"] and rows[0]["site_sol_ranges"] == [[361, 378], [91, 101]] and rows[0]["site_sols"] == [361, 378]
    rows = S.scan_scapes(tmp_path, {"sid": [361, 378]})
    assert not rows[0]["ok"] and "no image in the site's sols 361-378" in rows[0]["reason"]


def test_shipped_sites_v0p75():
    from mppp.waypoints import sites as S
    t = S.load_site_table()
    r = S.load_site_ranges()
    assert r["chal_rocks"] == [(91, 101), (361, 378)] and r["threeforks_large"][0] == (652, 693)
    assert r["enchanted_lake"] == [(556, 590), (421, 423)] and S.load_sites()["chal_rocks"] == (91, 378)
    g = t["groups"]
    assert "olifants" not in g["ecam_consensus"] and "cheyava_falls" in g["ecam_consensus"]
    assert "threeforks_large" in g["ecam_consensus"] and "threeforks_north" not in g["ecam_consensus"]   # the 7 large blocks back (0.90, 20 Oct)
    assert "zcam34_consensus_v0p82" not in g and "threeforks_large" not in g["zcam34_consensus"]
    assert len(g["ecam_consensus"]) == 29 and "seitah_overlook" in g["ecam_consensus"] and {"plateau_point", "vingeanne", "hans_amundsen_workspace", "pico_turquino", "hippo_pools"} <= set(g["ecam_consensus"]) and "navcam_consensus_v0p82" not in g
    assert len(g["zcam34_consensus"]) == 12 and "olifants" in g["zcam34_consensus"] and "issole_seitah_south" in g["zcam34_consensus"] and g["zcam34_consensus"][:2] == ["chal_rocks_large", "rochette"]   # 22 Oct: chal_rocks_large back; 21 Oct (Christian): Issole back without its video sequence ZCAM05093, olifants back (12 blocks)


def test_zcam34_dev_group_sites_have_the_zoom():
    from mppp.waypoints.sites import site_group
    dev = site_group("zcam34_dev")
    from mppp.waypoints.sites import load_site_table, site_zooms
    t = load_site_table()                     # v0p82: Christian's dev group may use a smaller site of a consensus block
    assert len(dev) == 6 and all(34 in site_zooms(t["sites"][s]) for s in dev)


def test_sites_block_record_and_groups():
    from mppp.waypoints.sites import block_sites, check_block_record, load_site_table, site_block, site_group, validate_site_table
    assert check_block_record({"number": 3, "half_size_m": 30, "centre": {"lon": 77.4, "lat": 18.4}}) == []
    assert check_block_record({"number": 0}) and check_block_record([1])
    t = load_site_table()
    assert t["version"] == 5                                              # v0p81: the official_blocks table
    b = block_sites(t)
    assert len(b) == 40 and b[0] == "oeb_landing_site" and b[-1] == "marble_mountain"   # 0.91: + Rose River Falls, Olifants
    assert site_block("chal_rocks_large", t)["number"] == 3 and site_block("chal_rocks", t)["number"] == 3
    assert len(site_group("blocks_all")) == 40
    assert "chal_rocks_large" in site_group("navcam_baseline")
    err, _ = validate_site_table()
    assert not err


# the unmasked-area filter of the WORK folder, the site order (v0p82)


def test_included_fraction_filter():
    from mppp.sfm.workdir import MIN_INCLUDED_FRACTION, filter_included_fraction
    assert MIN_INCLUDED_FRACTION is None                     # v0p85: off (1/3 in v0p82)
    ims = [{"source_product": "a", "mask": {"included_fraction": 0.30}},
           {"source_product": "b", "mask": {"included_fraction": 0.34}}, {"source_product": "c"}]
    assert filter_included_fraction(ims)[1] == []
    kept, removed = filter_included_fraction(ims, 1 / 3)
    assert [m["source_product"] for m in kept] == ["b", "c"] and removed == [{"image": "a", "included_fraction": 0.3}]
    assert filter_included_fraction(ims, None)[1] == []


def test_sites_in_order_of_first_visit():
    from mppp.waypoints.sites import load_site_table
    t = load_site_table()
    first = [min(a for a, _ in v["sols"]) for v in t["sites"].values()]
    assert first == sorted(first)


# block folders without _colmap, the 19 Oct site fixes (v0p85)


def test_folder_names_and_old_names():
    from mppp.waypoints import sites as S
    assert S.folder_name("south_arm", 1408) == "mars2020_sol_1408_south_arm"
    assert S.folder_name("south_arm", 1408, True) == "mars2020_sol_1408_south_arm_zcam"
    for n, want in [("mars2020_sol_1408_south_arm", ("south_arm", False)),
                    ("mars2020_sol_1408_south_arm_zcam", ("south_arm", True)),
                    ("mars2020_sol_1408_south_arm_colmap", ("south_arm", False)),          # v0p53-v0p82
                    ("mars2020_sol_1408_south_arm_colmap_zcam", ("south_arm", True)),
                    ("mars2020_sol_0072_chal_rocks_large", ("chal_rocks_large", False)),
                    ("camera_analysis", (None, False)), ("mppp_pipeline", (None, False))]:
        assert S.parse_work_folder(n) == want, n


def test_scan_scapes_reads_new_and_old_names(tmp_path):
    from mppp.waypoints import sites as S
    for n in ("mars2020_sol_1408_south_arm", "mars2020_sol_1408_south_arm_colmap_zcam", "camera_analysis"):
        (tmp_path / n).mkdir()
    rows = S.scan_scapes(tmp_path, {"south_arm": [[1408, 1412]]})
    assert sorted((r["site"], r["zcam"]) for r in rows) == [("south_arm", False), ("south_arm", True)]


def test_site_fixes_19_oct():
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))["sites"]
    assert t["chal_rocks_large"]["sols"][0] == [72, 100]
    assert t["chal_rocks_large"]["settings"]["NEARBY_FROM_FIRST_SOL"] is True
    assert [421, 423] in t["enchanted_lake"]["sols"]
    assert t["bell_island_large"]["sols"][0][0] == 1431
    assert t["rochette"]["sols"][0][0] == 178 and t["rochette"]["settings"]["NEARBY_FROM_FIRST_SOL"] is True
    assert t["hippo_pools"]["sols"][0][0] == 1948 and t["hippo_pools"]["settings"]["NEARBY_FROM_FIRST_SOL"] is True


def test_sites_19_oct_afternoon():
    """Christian (19 Oct): notes last; Issole 201 + 286-339; Seitah North Large in the Navcam consensus and the
    baseline; seven more consensus blocks; Seitah Overlook 113-125; Butler Landing 1-14 + 32-44; Butler Landing
    Large gone; Van Zyl Overlook Huge 15-99 + 361-378 replaces Van Zyl Overlook Large."""
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))
    S, G = t["sites"], t["groups"]
    assert all(list(v)[-1] == "notes" for v in S.values() if "notes" in v)
    assert S["issole_seitah_south"]["sols"] == [[201, 201], [286, 339]]
    assert S["seitah_north_large"]["sols"] == [[202, 284]]
    assert S["artuby_ridge"]["sols"] == [[342, 349], [174, 176]]
    assert S["seitah_overlook"]["sols"] == [[113, 125]] and S["butler_landing"]["sols"] == [[1, 14], [32, 44]]
    assert "butler_landing_large" not in S and "van_zyl_overlook_large" not in S
    assert S["van_zyl_overlook_huge"]["sols"] == [[15, 99], [361, 378]]
    nc = set(G["ecam_consensus"])
    assert {"seitah_north_large", "marble_mountain", "hippo_pools", "cheyava_falls", "bunsen_peak", "tuxedo_park",
            "belva_crater_large", "oeb_landing_site", "van_zyl_overlook_huge", "origny_large"} <= nc and len(nc) == 29 and "knob_mountain_large" in nc
    assert "seitah_north_large" in G["navcam_baseline"] and "van_zyl_overlook_huge" in G["navcam_baseline"]
    assert not any(s in ("butler_landing_large", "van_zyl_overlook_large") for g in G.values() for s in g)


def test_zcam34_joint_group_is_the_joint_default(tmp_path):
    """22 Oct: the joints take zcam34_joint (the good blocks) by default and leave out FAIL blocks."""
    import argparse, json as _json
    from mppp.cli import _navcam_blocks
    from mppp.waypoints.sites import load_site_table, work_folder
    t = load_site_table()
    g = t["groups"]["zcam34_joint"]
    assert "rockytop" not in g and "bunsen_peak" not in g and "chal_rocks_large" in g      # 23 Oct: large
    for n, verdict in (("chal_rocks_large", "warn"), ("rochette", "fail")):
        w = work_folder(tmp_path, n, True, t["sites"][n]["sols"]) / "colmap"
        w.mkdir(parents=True)
        (w / "run_done.json").write_text(_json.dumps({"verdict": verdict, "site": n}))
    a = argparse.Namespace(blocks=None, root=str(tmp_path), sites_file=None, which="zcam", zoom=34, group=None)
    assert a and set(_navcam_blocks(a)) == {"chal_rocks_large"} and a.group == "zcam34_joint"
    a2 = argparse.Namespace(blocks=None, root=str(tmp_path), sites_file=None, which="zcam", zoom=34, group=None,
                            include_failed=True)
    assert set(_navcam_blocks(a2)) == {"chal_rocks_large", "rochette"}


def test_ecam_consensus_and_the_old_group_name():
    """0.94.4 (Christian): navcam_consensus is ecam_consensus (Navcam + both Hazcam pairs); the old name still works."""
    from mppp.waypoints import sites as S
    t = S.load_site_table()
    assert "ecam_consensus" in t["groups"] and S.GROUP_ALIASES == {"navcam_consensus": "ecam_consensus"}
    assert S.site_group("navcam_consensus") == S.site_group("ecam_consensus") and "seitah_overlook" in S.site_group("ecam_consensus")
    raw = json.loads(S.SITES_FILE.read_text(encoding="utf-8"))
    assert "navcam_consensus" not in raw["groups"]                    # only the new name in the file
