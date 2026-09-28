"""v0p22.2: fitted gate forms and illumination covariates, rho re-estimate, eps vs convergence angle, solar geometry,
network strength / held Navcam intrinsics / staged Navcam-then-Mastcam-Z, verified consensus cameras, sites table."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------- gate fitting
def _synthetic_pairs(rng, n=40000, A=0.8, theta_bar=5.0, cv=0.6, tau_h=3.0, s_intra=0.7):
    """Cross-station trials drawn from the power gate x exp(-dLMST/tau), plus same-station trials at s_intra."""
    from mppp.error.alignment import gate_curve
    th = rng.uniform(0, 30, n)
    dl = rng.uniform(0, 8, n)
    p = s_intra * gate_curve(th, A, theta_bar, cv) * np.exp(-dl / tau_h)
    obs_c = rng.uniform(size=n) < p
    n_i = 8000
    obs_i = rng.uniform(size=n_i) < s_intra
    az = rng.uniform(0, 360, n + n_i)
    return {"theta_deg": np.r_[th, np.zeros(n_i)].astype(np.float32), "dlmst_h": np.r_[dl, np.zeros(n_i)].astype(np.float32),
            "dsun_deg": np.r_[dl * 10, np.zeros(n_i)].astype(np.float32), "dshadow": np.r_[dl / 4, np.zeros(n_i)].astype(np.float32),
            "observed": np.r_[obs_c, obs_i], "cross": np.r_[np.ones(n, bool), np.zeros(n_i, bool)],
            "families": np.array(["Navcam-Navcam"] * (n + n_i)), "weight": np.ones(n + n_i, np.float32),
            "point": rng.integers(0, 500, n + n_i), "n_points": 500, "image_a": np.zeros(n + n_i, int),
            "image_b": np.zeros(n + n_i, int)}


def test_fit_gate_recovers_all_parameters_and_ranks_the_forms():
    from mppp.error import alignment as A
    rng = np.random.default_rng(3)
    pairs = _synthetic_pairs(rng)
    g = A.fit_gate(pairs, form="power", illumination="dlmst", fit_tau=True, n_boot=0)
    assert g["form"] == "power" and g["illumination"] == "dlmst" and g["constrained"] and not g["at_limit"]
    assert abs(g["A"] - 0.8) < 0.1 and abs(g["theta_bar_deg"] - 5.0) < 1.0 and abs(g["cv"] - 0.6) < 0.15
    assert abs(g["tau_h"] - 3.0) < 0.4 and g["n_params"] == 4 and np.isfinite(g["aic"]) and g["chi2_dof"] < 3
    # holding tau gives 3 parameters; the other covariates and "none" are accepted
    g2 = A.fit_gate(pairs, form="power", illumination="dlmst", fit_tau=False, tau_h=3.0, n_boot=0)
    assert g2["n_params"] == 3 and g2["tau_h"] == 3.0
    for ill, key in (("sunangle", "sun0_deg"), ("shadow", "s0"), ("none", None)):
        gi = A.fit_gate(pairs, form="power", illumination=ill, n_boot=0)
        assert (key in gi) == (key is not None)
    # the other forms fit and carry their own parameter names
    names = {"exp": {"theta_bar_deg"}, "stretch": {"theta_bar_deg", "beta"}, "logistic": {"theta_0_deg", "width_deg"}}
    for form, keys in names.items():
        gf = A.fit_gate(pairs, form=form, illumination="dlmst", n_boot=0)
        assert keys <= set(gf) and gf["n_params"] == 1 + len(keys) + 1
    with pytest.raises(ValueError):
        A.fit_gate(pairs, form="cosine")
    # the comparison ranks the generating form first
    rows = A.compare_gate_forms(pairs, forms=("power", "exp", "logistic"), illuminations=("none", "dlmst"))
    assert rows[0]["form"] == "power" and rows[0]["illumination"] == "dlmst" and rows[0]["dAIC"] == 0.0
    assert all(r["dAIC"] >= 0 for r in rows) and len(rows) == 6
    # gate_expected takes the fit dict and reproduces the binned rate
    edges = np.arange(0, 31, 5.0)
    pred = A.gate_expected(pairs, g, edges)
    sc = A.survival_curve(pairs, edges, cross=True)
    assert pred.shape == sc["rate"].shape and np.nanmax(np.abs(pred - sc["rate"])) < 0.08
    # bootstrap ranges are reported when asked
    gb = A.fit_gate(pairs, form="power", illumination="dlmst", n_boot=4, seed=1)
    assert len(gb["theta_bar_deg_16_84"]) == 2 and len(gb["tau_h_16_84"]) == 2


def test_fit_gate_flags_a_flat_gate_as_unconstrained():
    from mppp.error import alignment as A
    rng = np.random.default_rng(4)
    pairs = _synthetic_pairs(rng, theta_bar=5000.0, cv=0.6)          # no decline with angle inside 30 deg
    g = A.fit_gate(pairs, form="power", illumination="dlmst", n_boot=0)
    assert not g["constrained"]
    ge = A.fit_gate(pairs, form="exp", illumination="dlmst", n_boot=0)
    assert not ge["constrained"] and (ge["at_limit"] or ge["theta_bar_deg"] >= 30)


# ----------------------------------------------------------------------------------------------- rho
def test_fit_rho_and_first_bin():
    from mppp.error.alignment import RHO_EDGES_DEG, fit_rho
    e = np.array(RHO_EDGES_DEG)
    c = 0.5 * (e[:-1] + e[1:])
    rho = 0.01 + 0.15 * np.exp(-c ** 2 / (2 * 0.3 ** 2))
    dc = {"theta_deg": c, "edges_deg": e.tolist(), "rho": rho, "se": np.full(c.size, 0.004), "n": np.full(c.size, 1000)}
    f = fit_rho(dc)
    g = f["gaussian"]
    assert abs(g["rho_0"] - 0.15) < 0.02 and abs(g["theta_c_deg"] - 0.3) < 0.05 and abs(g["rho_inf"] - 0.01) < 0.005
    assert f["first_bin"]["rho"] == pytest.approx(rho[0]) and not f["first_bin"]["consistent_with_zero"]
    assert "chi2_dof" in f["exponential"]
    dc["rho"] = np.zeros(c.size) + 0.001
    assert fit_rho(dc)["first_bin"]["consistent_with_zero"]


def test_decorrelation_reports_disjoint_pairs_with_errors(tmp_path, monkeypatch):
    from test_v0p15 import _synthetic_model
    from mppp.error import alignment as A
    model, X = _synthetic_model(noise_px=0.2)
    e = tmp_path / "work" / "colmap" / "error_input"
    (e / "native").mkdir(parents=True)
    (e / "stations.csv").write_text("name,station,instrument\n" + "".join(
        f"img{k}.png,{'A' if k <= 3 else 'B'},NL\n" for k in range(1, 7)))
    monkeypatch.setattr(A, "read_colmap", lambda path: copy.deepcopy(model))
    al = A.load_alignment(tmp_path / "work", min_track_length=3)
    dc = A.decorrelation(al, n_points=150, seed=0, n_boot=20)
    assert {"theta_deg", "edges_deg", "rho", "se", "n", "rho_shared_image", "n_shared", "n_tracks"} <= set(dc)
    assert dc["n_tracks"] > 50 and len(dc["rho"]) == len(A.RHO_EDGES_DEG) - 1
    ok = np.isfinite(dc["rho"])
    assert ok.any() and np.all(dc["se"][ok] > 0)
    assert np.nanmax(np.abs(dc["rho"][ok])) < 0.5             # independent noise: no strong correlation
    f = A.fit_rho(dc)
    assert "first_bin" in f


# ----------------------------------------------------------------------------------------------- eps vs angle, solar
def _alignment(tmp_path, monkeypatch, stations_csv):
    from test_v0p15 import _synthetic_model
    from mppp.error import alignment as A
    model, X = _synthetic_model(noise_px=0.2)
    e = tmp_path / "work" / "colmap" / "error_input"
    (e / "native").mkdir(parents=True)
    (e / "stations.csv").write_text(stations_csv)
    monkeypatch.setattr(A, "read_colmap", lambda path: copy.deepcopy(model))
    return A, A.load_alignment(tmp_path / "work", min_track_length=3)


def test_eps_by_angle_rows(tmp_path, monkeypatch):
    A, al = _alignment(tmp_path, monkeypatch, "name,station,instrument\n" + "".join(
        f"img{k}.png,{'A' if k <= 3 else 'B'},NL\n" for k in range(1, 7)))
    rows = A.eps_by_angle(al, min_obs=20)
    assert rows and {"alignment", "tracks", "angle", "theta_lo", "theta_hi", "n", "eps_px", "median_px"} <= set(rows[0])
    assert {r["angle"] for r in rows} <= {"theta_max", "theta_nn"} and {r["tracks"] for r in rows} <= {"intra", "cross"}
    n_obs = al.residuals["residual_px"].size if "residual_px" in al.residuals else None
    tm = [r for r in rows if r["angle"] == "theta_max"]
    assert all(r["theta_lo"] < r["theta_hi"] and r["n"] >= 20 and r["eps_px"] > 0 for r in tm)


def test_solar_geometry_from_stations_csv(tmp_path, monkeypatch):
    csv = "name,station,instrument,lmst,solar_elevation_deg,solar_azimuth_deg\n" + "".join(
        f"img{k}.png,{'A' if k <= 3 else 'B'},NL,{10 + k}:00:00,{45 if k <= 3 else 45},{90 if k <= 3 else 180}\n"
        for k in range(1, 7))
    A, al = _alignment(tmp_path, monkeypatch, csv)
    im = al.images
    a, b = im["img1.png"], im["img4.png"]
    assert "sun_vector" in a and "shadow_tip" in a
    # sun at 45 deg elevation, azimuth east vs south: the vectors are 60 deg apart, the shadow tips 2 sqrt(2) cot 45 apart
    assert abs(A.sun_angle_deg(a, b) - 60.0) < 1e-6
    assert np.linalg.norm(np.asarray(a["shadow_tip"]) - np.asarray(b["shadow_tip"])) == pytest.approx(np.sqrt(2), abs=1e-6)
    assert A.sun_angle_deg(a, im["img2.png"]) == pytest.approx(0.0, abs=1e-9)
    pairs = A.pair_survival(al, n_points=100, seed=0)
    assert "dsun_deg" in pairs and "dshadow" in pairs
    cr = pairs["cross"]
    assert np.all(np.abs(pairs["dsun_deg"][cr] - 60.0) < 1e-3) and np.all(np.abs(pairs["dsun_deg"][~cr]) < 1e-3)
    assert A.parameters.__doc__ is not None


# ----------------------------------------------------------------------------------------------- reconstruction
def test_navcam_network_and_hold(tmp_path):
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm.reconstruction import bundle_adjust, navcam_network
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    net = navcam_network(proj)
    assert net["stations"] == 2 and 3.5 < net["span_m"] < 4.5 and net["verdict"] == "weak"
    assert navcam_network(proj, min_stations=2, min_span_m=1.0)["verdict"] == "strong"
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)           # perturbed start cameras
    start = {cid: np.array(c.params) for cid, c in rec.cameras.items()}
    out = bundle_adjust(rec, proj, sigma_px=noise, loss_scale=10.0, refine_intrinsics=True, max_iterations=10,
                        hold_cameras=["NL", "NR"])
    for cid, c in rec.cameras.items():
        assert np.allclose(np.array(c.params), start[cid])
    assert out["brief"]


def test_reconstruct_holds_navcam_intrinsics_on_a_weak_network(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec0))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)    # no database: keep the synthetic points
    start = {cid: np.array(c.params) for cid, c in rec0.cameras.items()}
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="auto")
    rs = proj.settings["reconstruction"]
    assert rs["navcam_network"]["verdict"] == "weak" and rs["navcam_intrinsics"] == "hold"
    assert set(rs["hold_cameras"]) == {"NL", "NR"} and rs["staged"] is False and rs["navcam_stage"] is None
    for cid, c in rec.cameras.items():
        assert np.allclose(np.array(c.params), start[cid])
    with pytest.raises(ValueError):
        R.reconstruct(proj, navcam_intrinsics="maybe")


def test_reconstruct_staged_solves_navcam_first(tmp_path, monkeypatch):
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2}}
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(rec0))
    monkeypatch.setattr(R, "triangulate", lambda rec, project, **kw: rec)
    # pretend the second station's frames are Mastcam-Z
    z = R._frames_of_family(rec0, proj, "N")[len(rec0.frames) // 2:]
    real = R._frames_of_family
    monkeypatch.setattr(R, "_frames_of_family", lambda rec, project, fam: z if fam == "Z" else
                        [f for f in real(rec, project, "N") if f not in z])
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="refine", staged=True, out_name="cahv_ba")
    rs = proj.settings["reconstruction"]
    assert rs["staged"] is True and rs["navcam_stage"]["path"] == "sparse/cahv_ba_navcam"
    assert (proj.root / "sparse" / "cahv_ba_navcam").is_dir()
    import pycolmap
    nav = pycolmap.Reconstruction(str(proj.root / "sparse" / "cahv_ba_navcam"))
    assert nav.num_reg_images() == rec.num_reg_images() - 2 * len(z)   # stage 1 without the "Z" frames
    assert set(rec.reg_frame_ids()) >= set(z)                          # stage 2 registered them again
    assert any(e.get("stage") == 2 for e in rs["log"] if isinstance(e, dict))
    for cid, c in rec.cameras.items():                                 # stage 2 held the Navcam cameras
        assert np.allclose(np.array(c.params), np.array(nav.cameras[cid].params))


# ----------------------------------------------------------------------------------------------- consensus cameras
def test_write_navcam_consensus_round_trip(tmp_path):
    from mppp.sfm import calibration as CAL
    from mppp.sfm.project import NAVCAM_RATIONAL_PATTERN, NAVCAM_RIG_FILE, camera_from_colmap_json
    cams = {g: CAL.reference_camera(g, "rational") for g in ("NL", "NR")}
    R = np.eye(3); t = np.array([-0.4244, 0.0, 0.0])
    written = CAL.write_navcam_consensus(cams, (R, t), tmp_path / "consensus", repeatability={"eps_ref_px": 0.25})
    assert set(written) == {"NL", "NR", "rig"}
    for g in ("NL", "NR"):
        c = camera_from_colmap_json(tmp_path / "consensus" / NAVCAM_RATIONAL_PATTERN.format(instrument=g))
        assert c["model"] == "FULL_OPENCV" and np.allclose(c["params"], cams[g].params) and c["free_params"] == ["k4"]
        d = json.loads(written[g].read_text())
        assert d["verification"]["repeatability"]["eps_ref_px"] == 0.25 and d["verification"]["per_scape"] == []
    rig = json.loads((tmp_path / "consensus" / NAVCAM_RIG_FILE).read_text())
    assert rig["ref"] == "NL" and np.allclose(rig["R_sensor_from_ref"], R) and abs(rig["baseline_m"] - 0.4244) < 1e-9


def test_project_create_accepts_navcam_cameras(tmp_path, monkeypatch):
    from mppp.sfm import project as P
    import inspect
    sig = inspect.signature(P.SfmProject.create)
    assert "navcam_cameras" in sig.parameters
    src = inspect.getsource(P.SfmProject.create)
    assert "nav_dir / NAVCAM_RATIONAL_PATTERN" in src and '"navcam_cameras"' in src


# ----------------------------------------------------------------------------------------------- docs and notebooks
def test_notebooks_carry_the_new_settings():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)
    assert ns["NAVCAM_INTRINSICS"] == "refine" and ns["STAGED"] is True and ns["NAVCAM_RIG_REFINE"] == "rotation"
    assert ns["NAVCAM_CAMERAS"] is None and ns["HOLD_CAMERAS"] == ()
    full = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for k in ("navcam_intrinsics=NAVCAM_INTRINSICS", "staged=STAGED", "hold_cameras=HOLD_CAMERAS", "error_input_navcam",
              "navcam_cameras=NAVCAM_CAMERAS", "navcam_network(proj)"):
        assert k in full, k
    nb4 = nbformat.read(str(ROOT / "notebooks" / "04_error_analysis.ipynb"), as_version=4)
    src4 = next(c.source for c in nb4.cells if "parameters" in c.metadata.get("tags", []))
    assert "GATE_FORM" in src4 and "ILLUMINATION" in src4 and "FIT_TAU" in src4 and "L0" not in src4
    full4 = "\n".join(c.source for c in nb4.cells)
    for k in ("eps_by_angle", "compare_gate_forms", "fit_rho", "decorrelation("):
        assert k in full4, k
    assert "archive" not in full4.lower() and "0.169" not in full4
    nb5 = nbformat.read(str(ROOT / "notebooks" / "05_camera_models.ipynb"), as_version=4)
    assert "write_navcam_consensus" in "\n".join(c.source for c in nb5.cells)


def test_tau_replaces_L0_in_the_error_package():
    import mppp.error.alignment as A
    import mppp.error.core as C
    assert not any(n.startswith("L0") or "L0" in n for n in dir(A) + dir(C) if n.isupper() or "_" in n and "L0_" in n)
    assert "tau_h" in A.ILLUMINATION["dlmst"]
    import inspect
    assert "fit_tau" in inspect.signature(A.fit_gate).parameters


def test_methods_doc_holds_no_site_results():
    txt = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    for bad in ("0.1–0.55 px", "8.5 px", "| Three Forks (52 images)", "59,143", "71.8 %", "0.284 vs 0.288"):
        assert bad not in txt, bad
    notes = (ROOT / "docs" / "results" / "working_notes.md").read_text(encoding="utf-8")
    assert "Withdrawn" in notes and "0.157" in notes
    assert (ROOT / "docs" / "results" / "sites.md").is_file() and (ROOT / "scripts" / "sites_table.py").is_file()


# ----------------------------------------------------------------------------------------------- other visits
def _wp(site, drive, sol, e, n):
    return {"type": "Feature", "properties": {"site": site, "drive": drive, "sol": sol, "easting": e, "northing": n}}


def test_stations_near_and_find_imgs_near(tmp_path):
    from mppp.waypoints import stations_near
    from mppp.select import find_imgs_near
    wps = {"features": [_wp(26, 500, 461, 1000.0, 2000.0), _wp(26, 600, 470, 1010.0, 2000.0),
                        _wp(30, 100, 600, 1003.0, 2004.0),       # 5.0 m from S026D0500: a later visit
                        _wp(30, 200, 610, 1016.0, 2000.0),       # 6 m from S026D0600: outside 5 m
                        _wp(31, 0, 700, 2000.0, 2000.0)]}
    rows = stations_near(wps, [(26, 500), (26, 600)], 5.0)
    keys = {(r["site"], r["drive"]): r for r in rows}
    assert set(keys) == {(26, 500), (26, 600), (30, 100)}
    assert keys[(30, 100)]["distance_m"] == pytest.approx(5.0) and keys[(30, 100)]["nearest"] == [26, 500]
    assert not keys[(30, 100)]["anchor"] and keys[(26, 500)]["anchor"]
    assert {(r["site"], r["drive"]) for r in stations_near(wps, [(26, 500), (26, 600)], 6.5)} >= {(30, 200)}
    # an archive of empty .IMG files named like PDS products
    def name(cam, sol, site, drive):
        return f"{cam}_{sol:04d}_0700000000_000RAD_N{site:03d}{drive:04d}NCAM00100_0A0095J01.IMG"
    for cam, sol, site, drive in [("NLF", 461, 26, 500), ("NRF", 461, 26, 500), ("NLF", 470, 26, 600),
                                  ("NLF", 600, 30, 100), ("NLF", 610, 30, 200), ("NLF", 700, 31, 0)]:
        (tmp_path / name(cam, sol, site, drive)).write_bytes(b"")
    paths, rep = find_imgs_near(tmp_path, ["NLF", "NRF"], (455, 480), wps, radius_m=5.0)
    names = sorted(p.name for p in paths)
    assert len(names) == 4 and any("_0600_" in n for n in names) and not any("_0610_" in n for n in names)
    assert rep["n_in_range"] == 3 and rep["n_added"] == 1
    assert rep["stations_added"] == [{"station": "S030D0100", "distance_m": 5.0, "nearest": "S026D0500", "images": 1,
                                      "sols": [600, 600]}]
    paths0, rep0 = find_imgs_near(tmp_path, ["NLF", "NRF"], (455, 480), wps, radius_m=None)
    assert len(paths0) == 3 and rep0["n_added"] == 0


def test_notebook_03_offers_other_visits():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)
    assert ns["ADD_NEARBY_WAYPOINTS"] == 5
    full = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "find_imgs_near(PDS_DIR" in full and "radius_m=ADD_NEARBY_WAYPOINTS" in full
