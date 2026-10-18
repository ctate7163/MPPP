"""v0p82 (17 Oct 2026): the 1/3 unmasked-area filter, the prior gauge, the waypoint check, the baseline stage."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UP = Path("/mnt/user-data/uploads/colmap/mars2020_sol_0074_chal_rocks_large_colmap/colmap_v0p80")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_included_fraction_filter():
    from mppp.sfm.workdir import MIN_INCLUDED_FRACTION, filter_included_fraction
    assert MIN_INCLUDED_FRACTION == pytest.approx(1 / 3)
    ims = [{"source_product": "a", "mask": {"included_fraction": 0.30}},
           {"source_product": "b", "mask": {"included_fraction": 0.34}}, {"source_product": "c"}]
    kept, removed = filter_included_fraction(ims)
    assert [m["source_product"] for m in kept] == ["b", "c"] and removed == [{"image": "a", "included_fraction": 0.3}]
    assert filter_included_fraction(ims, None)[1] == []


def test_notebook_carries_the_v0p82_settings():
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for k in ("MIN_INCLUDED_FRACTION  = 1/3", "NO_POSITION_PRIOR_AT = []", "PRIOR_GAUGE_RECENTRE = True",
              "PRIOR_SIGMA_M = 1.0", "filter_included_fraction(manifest[\"images\"], MIN_INCLUDED_FRACTION)",
              "prior_sigma_m=(PRIOR_SIGMA_M,) * 3", 'proj.settings["no_position_prior_at"]'):
        assert k in src, k
    from mppp.runner import ALIGN_RULES
    assert ALIGN_RULES == 82


class _Frame:
    def __init__(self, C):
        import pycolmap
        self.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), -np.asarray(C, float))


class _Rec:
    def __init__(self, centres):
        self.frames = {i: _Frame(c) for i, c in enumerate(centres)}
        self.moved = None

    def transform(self, sim):
        self.moved = np.asarray(sim.translation, float)
        for f in self.frames.values():
            T = f.rig_from_world
            import pycolmap
            f.rig_from_world = pycolmap.Rigid3d(T.rotation, np.asarray(T.translation) - np.asarray(T.rotation.matrix()) @ self.moved)


def test_prior_gauge_moves_the_block_onto_its_priors():
    from mppp.sfm.reconstruction import PRIOR_GAUGE_RECENTRE, prior_gauge
    assert PRIOR_GAUGE_RECENTRE is True
    priors = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 1.0]])
    rec = _Rec(priors + [-0.19, -0.07, -0.10])
    g = prior_gauge(rec, list(enumerate(priors)))
    assert np.allclose(g["mean_residual_m"], [0.19, 0.07, 0.10]) and g["recentred"]
    g2 = prior_gauge(rec, list(enumerate(priors)), recentre=False)
    assert np.allclose(g2["mean_residual_m"], 0, atol=1e-12) and not g2["recentred"]
    assert prior_gauge(rec, []) is None


@pytest.mark.skipif(not (UP / "error_input" / "native").is_dir(), reason="no Chal Rocks Large run here")
def test_prior_gauge_on_chal_rocks_large():
    import pycolmap
    from mppp.sfm.reconstruction import prior_gauge
    P = json.loads((UP / "project.json").read_text())
    meta = {r["name"]: r for r in P["images"]}
    rec = pycolmap.Reconstruction(str(UP / "error_input" / "native"))
    pf = []
    for fid, fr in rec.frames.items():
        nl = [d.id for d in fr.data_ids if d.id in rec.images and rec.images[d.id].name.startswith("NL")]
        if nl and meta[rec.images[nl[0]].name]["station"] not in ("S004D0012", "S004D0182", "S010D1080", "S010D1178",
                                                                  "S010D1224", "S011D0064"):
            pf.append((fid, np.array(meta[rec.images[nl[0]].name]["prior_C"])))
    g = prior_gauge(rec, pf)
    assert g["frames"] == 246 and g["mean_residual_m"][0] == pytest.approx(0.19, abs=0.01)
    assert np.allclose(prior_gauge(rec, pf, recentre=False)["mean_residual_m"], 0, atol=1e-9)


def test_similarity_fit_is_robust():
    from mppp.sfm.waypoint_check import similarity_fit
    rng = np.random.default_rng(1)
    src = rng.uniform(-50, 50, (12, 3))
    src[:, 2] *= 0.05
    a = np.radians(0.3)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    dst = 1.0025 * src @ R.T + [0.2, -0.1, 0.05] + rng.normal(0, 0.03, src.shape)
    dst[3] += [2.0, -1.5, 0]                                   # one bad localisation
    f = similarity_fit(src, dst)
    assert f["scale"] == pytest.approx(1.0025, abs=4e-4) and f["yaw_deg"] == pytest.approx(0.3, abs=0.05)


def _fake_block(tmp, stations, bad=None, stretch_e=0.0):
    """A project folder with project.json and error_input/poses.csv: refined = prior (+ the bad offset, + an east
    stretch) per station, 10 images each."""
    p = tmp / "mars2020_sol_0001_fake_colmap" / "colmap"
    (p / "error_input").mkdir(parents=True)
    (p / "project.json").write_text(json.dumps({"settings": {"reconstruction": {"unlocalized": {}}}, "images": []}))
    with open(p / "error_input" / "poses.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "station", "station_label", "sol", "instrument", "downsample_scale", "dE_m", "dN_m", "dU_m",
                    "dC_m", "dAttitude_deg", "E_m", "N_m", "U_m", "observations"])
        for k, (st, e, n) in enumerate(stations):
            d = np.array([stretch_e * e, 0.0, 0.0]) + (np.array(bad[1]) if bad and bad[0] == st else 0.0)
            for i in range(10):
                w.writerow([f"NLF_{k}_{i}.png", st, st, 100 + k, "NL", 0.25, *d, float(np.linalg.norm(d)), 0.1,
                            e + d[0], n + d[1], d[2], 500])
    return p


def test_check_block_flags_a_bad_waypoint(tmp_path):
    from mppp.sfm import waypoint_check as WC
    st = [(f"S010D{i:04d}", float(x), float(y)) for i, (x, y) in
          enumerate([(0, 0), (20, 5), (40, -3), (10, 30), (35, 25), (-5, 45)])]
    p = _fake_block(tmp_path, st, bad=("S010D0002", (1.4, -0.6, 0.0)))
    r = WC.check_block(p)
    assert r["flagged"] == ["S010D0002"] and not r["similarity"]["translation_only"]
    m = WC.mission_flags([r])
    assert m["flagged"] == ["S010D0002"]
    out = WC.write_flags(m, tmp_path / "flags")
    assert "S010D0002 **flag**" in (out / "waypoint_flags.md").read_text()
    assert len(list(csv.reader(open(out / "waypoint_flags.csv")))) == 7


def test_check_block_east_west_stretch_is_anisotropic(tmp_path):
    from mppp.sfm import waypoint_check as WC
    rng = np.random.default_rng(3)
    st = [(f"S020D{i:04d}", float(x), float(y)) for i, (x, y) in enumerate(rng.uniform(-60, 60, (10, 2)))]
    r = WC.check_block(_fake_block(tmp_path, st, stretch_e=0.004))
    an = r["anisotropy"]
    assert an["scale_E_ppm"] - an["scale_N_ppm"] == pytest.approx(4000, rel=0.05)


def test_small_block_translation_only(tmp_path):
    from mppp.sfm import waypoint_check as WC
    st = [(f"S030D{i:04d}", float(i * 3), 0.0) for i in range(6)]          # 15 m
    r = WC.check_block(_fake_block(tmp_path, st))
    assert r["similarity"]["translation_only"] and r["anisotropy"] is None


def test_baseline_rule_needs_better_scale_agreement_and_isotropy():
    m = _script("navcam_baseline_study")
    b_use = 0.42401
    rows = [{"site": f"s{i}", "baseline_m": b_use * (1 - 2.5e-3), "baseline_sd_m": 0.0002, "scale_held_ppm": 2500.0,
             "anisotropy_held": {"scale_E_ppm": 0.0, "se_E_ppm": 300.0, "scale_N_ppm": 0.0, "se_N_ppm": 300.0}}
            for i in range(5)]
    res = m.combine_rows(rows, b_use)
    assert res["adopt"] and res["scale_agreement"]["rms_predicted_ppm"] < 50
    # the same baseline change, but the blocks are already in scale: the agreement would get worse
    rows2 = [dict(r, scale_held_ppm=0.0) for r in rows]
    res2 = m.combine_rows(rows2, b_use)
    assert not res2["adopt"] and "would not improve" in res2["verdict"]
    # east-west only: a map-projection error, not the baseline
    rows3 = [dict(r, anisotropy_held={"scale_E_ppm": 5000.0, "se_E_ppm": 300.0, "scale_N_ppm": 0.0, "se_N_ppm": 300.0})
             for r in rows]
    res3 = m.combine_rows(rows3, b_use)
    assert not res3["adopt"] and "map-projection" in res3["verdict"]


def test_baseline_sites_qualify_from_the_waypoint_fields():
    m = _script("navcam_baseline_study")
    keep, left = m.qualifying_sites()
    assert keep and all(s in ("van_zyl_overlook_large", "chal_rocks_large", "threeforks_large", "belva_crater_large",
                              "bell_island_large", "taylorfjellet_huge", "origny_large") for s in keep)
    t = {"sites": {"a": {"n_waypoints": 5, "max_waypoint_separation_m": 31.0},
                   "b": {"n_waypoints": 4, "max_waypoint_separation_m": 90.0},
                   "c": {"n_waypoints": 9, "max_waypoint_separation_m": 30.0}}, "groups": {}}
    keep, left = m.qualifying_sites(["a", "b", "c"], t)
    assert keep == ["a"] and [x for x, _ in left] == ["b", "c"]


def test_pipeline_order_and_translation_held(tmp_path):
    m = _script("run_pipeline")
    S = list(m.STAGES)
    assert S.index("navcam_consensus_2") < S.index("waypoint_check") < S.index("baseline_study") \
        < S.index("baseline_confirm") < S.index("navcam_consensus_3") < S.index("zcam_start")
    src = (ROOT / "scripts" / "run_pipeline.py").read_text()
    assert '"--apply-translation", "--joint-translation"' not in src
    assert m.main(["--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "mppp_pipeline" / "dry_run_log.txt").read_text()
    assert "waypoint_check.py" in log and "navcam_baseline_study.py all" in log


def test_sites_in_order_of_first_visit():
    from mppp.sfm.sites import load_site_table
    t = load_site_table()
    first = [min(a for a, _ in v["sols"]) for v in t["sites"].values()]
    assert first == sorted(first)
    from mppp.maps.official import order_by_first_sol
    assert list(order_by_first_sol({"b": {"sols": [[5, 6], [1, 2]]}, "a": {"sols": [[3, 4]]}})) == ["b", "a"]


def test_v0p83_both_principal_points_pass_the_form_check(tmp_path):
    """v0p83 option (Christian, 18 Oct): the stereo temperature change on both eyes' principal points, never a rig
    temperature term - NL cx(T) is an error unless the cameras say pp_thermal = both."""
    import inspect
    import shutil
    from mppp.paths import cmods_dir
    from mppp.sfm.cmods_form import NAVCAM_FILES, check_form
    from mppp.sfm.navcal_consensus import fit_consensus
    for n in NAVCAM_FILES:
        shutil.copy2(cmods_dir() / n, tmp_path / n)
    f = tmp_path / "M2020_NL_fisheye_tangential.json"
    j = json.loads(f.read_text())
    j["thermal"]["cx_px_per_degC"] = 0.02
    f.write_text(json.dumps(j))
    r = check_form(tmp_path)
    assert not r["ok"] and any("NL principal point" in e for e in r["errors"])
    j["thermal"]["pp_thermal"] = "both"
    f.write_text(json.dumps(j))
    r = check_form(tmp_path)
    assert r["ok"] and any("v0p83" in w for w in r["warnings"])
    sig = inspect.signature(fit_consensus).parameters
    assert sig["nl_pp_thermal"].default is None and sig["early_hold"].default is False
    src = (ROOT / "scripts" / "navcam_calibration_study.py").read_text()
    assert "--nl-pp-thermal" in src and "--early-hold" in src


def test_v0p83_athermal_rig_is_the_v0p80_default():
    from mppp.sfm import navcal_consensus as N
    assert N.NL_PP_THERMAL_FORMS == ("v0p80",) and "NL_cx" in N.THERMAL_TERMS
    src = (ROOT / "src" / "mppp" / "sfm" / "navcal_consensus.py").read_text()
    assert '("NL_cx",) + V0P73_TERMS if nl_pp_thermal' in src
