"""The reconstruction-error package (mppp.error): error model self-test, stations, gate, decorrelation, projections. (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
from pathlib import Path
import pytest
import numpy as np
import copy


DATA = Path(__file__).resolve().parents[1] / "src/mppp/data/M20_waypoints.json"   # packaged snapshot (v0p13)


@pytest.mark.slow
def test_error_model_selftest_186():
    import matplotlib
    matplotlib.use("Agg")
    from mppp.error.selftest import run_all
    assert run_all(verbose=False)


def test_build_stations_no_longer_shadowed():
    """v0p15: a second `_last_per_sol` shadowed the first -> build_stations raised TypeError."""
    from mppp.error.waypoints import build_stations, load_featurecollection
    stations, info = build_stations(load_featurecollection(str(DATA)), anchor_site=3, anchor_drive=0)
    assert len(stations) >= 2


def test_sol_1842_mid_drive_rule_is_documented_behaviour():
    """
    OPEN ISSUE (docs/mppp_error_review_v0p2.md #7), pinned, NOT fixed: on sol 1842
    the 'highest drive' rule keeps 87_5286 (final='m', mid-drive) over 88_0
    (final='y').  If this changes, the frozen site-87 prediction must be re-checked.
    """
    from mppp.error.waypoints import _last_per_sol
    feats = json.loads(DATA.read_text())["features"]
    by = {f["properties"]["RMC"]: f["properties"] for f in feats}
    assert by["87_5286"]["final"] == "m" and by["88_0"]["final"] == "y"
    assert _last_per_sol(by, ["87_5286", "88_0"]) == ["87_5286"]


pycolmap = pytest.importorskip("pycolmap")


def test_error_projection_matches_pycolmap_with_distortion():
    from mppp.error.colmap import project_camera
    rng = np.random.default_rng(3)
    for model, params in [("FULL_OPENCV", [2950, 2951, 2594, 1942, -0.27, 0.1, 1e-4, -2e-4, -0.02, 0.01, 0.002, 0.001]),
                          ("OPENCV", [1000, 1001, 640, 480, -0.1, 0.02, 1e-3, -1e-3]),
                          ("RADIAL", [800, 400, 300, -0.2, 0.05]), ("PINHOLE", [700, 710, 320, 240])]:
        cam = pycolmap.Camera(model=model, width=1280, height=960, params=params)
        for _ in range(5):
            X = np.array([*rng.uniform(-0.5, 0.5, 2), 1.0]) * rng.uniform(1, 10)
            assert np.allclose(project_camera(model, np.array(params, float), X), cam.img_from_cam(X), atol=1e-6)


def _look_at(C, target):
    """World-to-camera rotation for a camera at C looking at target (y down)."""
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _quat(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _synthetic_model(noise_px=0.0, seed=0, full_keypoint_lists=True):
    """Two stations 6 m apart, three cameras each, 200 terrain points seen by all six."""
    from mppp.colmap import project_camera
    from mppp.error.colmap import ColmapCamera, ColmapImage, ColmapModel, ColmapPoint
    rng = np.random.default_rng(seed)
    cam = ColmapCamera(1, "FULL_OPENCV", 1280, 960, np.array([1000, 1000, 640, 480, -0.2, 0.05, 1e-4, -1e-4,
                                                               0, 0, 0, 0], float))
    X = np.c_[rng.uniform(-3, 3, 200), rng.uniform(8, 12, 200), rng.uniform(-0.3, 0.3, 200)]
    centres = [np.array([sx + dx, 0, 2.0]) for sx in (-3.0, 3.0) for dx in (-0.2, 0.0, 0.2)]
    images, obs = {}, {}
    for k, C in enumerate(centres, start=1):
        R = _look_at(C, [0, 10, 0])
        t = -R @ C
        uv = project_camera(cam.model, cam.params, X @ R.T + t) + rng.normal(0, noise_px, (len(X), 2))
        images[k] = ColmapImage(k, _quat(R), t, 1, f"img{k}.png", uv, np.arange(1, len(X) + 1),
                                station="A" if k <= 3 else "B")
    points = {j + 1: ColmapPoint(j + 1, X[j], np.zeros(3), 0.0, np.arange(1, 7), np.full(6, j)) for j in range(len(X))}
    if not full_keypoint_lists:                     # pre-0.14.5 exports: track indices do not match the lists
        for p in points.values():
            p.point2D_idxs = np.zeros(6, int)
    return ColmapModel({1: cam}, images, points), X


def _synthetic_pairs(A=0.8, thb=5.0, cv=0.4, L0=2.3, s_intra=0.5, n=200_000, seed=0):
    from mppp.error.alignment import gate_curve
    rng = np.random.default_rng(seed)
    cross = rng.random(n) < 0.7
    th = np.where(cross, rng.uniform(0, 30, n), rng.uniform(0, 3, n))
    dl = np.where(cross, rng.choice([0.0, 0.5, 1.5, 3.0], n), 0.0)
    p = np.where(cross, s_intra * gate_curve(th, A, thb, cv) * np.exp(-dl / L0), s_intra)
    return {"point": rng.integers(0, 20_000, n).astype(np.int32), "theta_deg": th.astype(np.float32),
            "cross": cross, "dlmst_h": dl.astype(np.float32), "families": np.full(n, "Navcam-Navcam"),
            "observed": rng.random(n) < p, "weight": np.ones(n, np.int8), "image_a": np.zeros(n, np.int32),
            "image_b": np.ones(n, np.int32), "n_points": 20_000, "label": "synthetic", "conditional": True}


@pytest.mark.parametrize("full_lists", [True, False])
def test_decorrelation_uses_the_observed_rays(full_lists):
    from mppp.error.colmap import measure_theta_c, observed_rays
    m, X = _synthetic_model(noise_px=0.0, full_keypoint_lists=full_lists)
    ids, C, U = observed_rays(m, m.points[5])
    assert len(ids) == 6
    d = X[4] - C
    assert np.allclose(U, d / np.linalg.norm(d, axis=1, keepdims=True), atol=1e-8)
    exact = measure_theta_c(m, n_bins=5, theta_max_deg=40)
    assert exact["residual_scale_m"] < 1e-6                           # perfect keypoints: rays meet at the point
    noisy = measure_theta_c(_synthetic_model(noise_px=0.5, full_keypoint_lists=full_lists)[0], n_bins=5,
                            theta_max_deg=40)
    # before v0p14.7 the rays were aimed at the fitted point, so this was ~0 whatever the keypoint noise
    # 0.5 px at 10 m over 0.2-6 m baselines: of order r^2 sigma / (b f) ~ 0.25 m for the shortest pairs
    assert 0.01 < noisy["residual_scale_m"] < 1.0 and np.isfinite(noisy["rho"]).any()


def test_fit_gate_recovers_a_known_gate():
    from mppp.error.alignment import combine_pairs, fit_gate, survival_curve
    pairs = _synthetic_pairs()
    g = fit_gate(pairs, n_boot=5)
    assert g["constrained"] and g["chi2_dof"] < 3
    assert g["s_intra"] == pytest.approx(0.5, abs=0.01)
    assert g["A"] == pytest.approx(0.8, rel=0.1) and g["theta_bar_deg"] == pytest.approx(5.0, rel=0.2)
    assert g["cv"] == pytest.approx(0.4, abs=0.12) and g["A_16_84"][0] < g["A_16_84"][1]
    assert 2 < g["theta_half_deg"] < 8
    sc = survival_curve(pairs, np.arange(0, 31, 1.0), cross=True)
    assert sc["rate"][0] > sc["rate"][20] and np.all(sc["lo"] <= sc["hi"])
    both = combine_pairs([pairs, _synthetic_pairs(seed=1)])
    assert both["n_points"] == 40_000 and both["point"].max() >= 20_000
    assert fit_gate(both)["theta_bar_deg"] == pytest.approx(5.0, rel=0.2)


def test_fit_gate_flags_a_sharp_cutoff():
    from mppp.error.alignment import fit_gate
    pairs = _synthetic_pairs()
    cut = pairs["cross"] & (pairs["theta_deg"] > 20)          # a cliff at 20 deg, flat before: not a power law
    pairs["observed"] = np.where(pairs["cross"], np.where(cut, False, np.random.default_rng(3).random(cut.size) < 0.4),
                                 pairs["observed"])
    g = fit_gate(pairs)
    assert g["chi2_dof"] > 10 and 15 < g["theta_half_deg"] < 22


def test_lmst_family_and_find_error_input(tmp_path):
    from mppp.error.alignment import _family, _lmst_hours, find_error_input
    assert _lmst_hours("Sol-00684M15:13:15.006") == pytest.approx(15 + 13 / 60 + 15.006 / 3600)
    assert _lmst_hours("") is None
    assert _family("NL") == "Navcam" and _family("ZR034") == "Mastcam-Z"
    e = tmp_path / "work" / "colmap" / "error_input"
    (e / "native").mkdir(parents=True)
    (e / "stations.csv").write_text("name,station\n")
    assert find_error_input(tmp_path / "work") == e == find_error_input(tmp_path / "work" / "colmap") == find_error_input(e)
    with pytest.raises(FileNotFoundError, match="notebook 03"):
        find_error_input(tmp_path)


def test_gate_curve_cv_zero_limit_has_no_warnings():
    import warnings
    from mppp.error.alignment import gate_curve
    th = np.linspace(0, 30, 31)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = gate_curve(th, 1.0, 5.0, 0.0)
        b = gate_curve(th, 1.0, 5.0, 2e-3)
    assert np.allclose(a, np.exp(-th / 5.0)) and np.allclose(a, b, rtol=1e-3)


def test_native_point_ids_follow_the_shortened_keypoint_lists():
    """residuals.npz carries reconstruction keypoint indices; the native model keeps observed keypoints only."""
    from types import SimpleNamespace
    from mppp.error.alignment import _native_point_ids
    # image 1: keypoints 0..9, observed at 2, 5, 9 -> native list [p10, p11, p12]
    # image 2: full list of 4 keypoints, observed at 1 and 3
    m = SimpleNamespace(images={1: SimpleNamespace(point3D_ids=np.array([10, 11, 12])),
                                2: SimpleNamespace(point3D_ids=np.array([-1, 20, -1, 21, -1, -1]))})
    iid = np.array([1, 2, 1, 1, 2, 7])
    p2d = np.array([9, 1, 2, 5, 3, 0])
    assert _native_point_ids(m, iid, p2d).tolist() == [12, 20, 10, 11, 21, -1]


def test_error_analysis_keeps_tracks_of_three_or_more(tmp_path, monkeypatch):
    from helpers import _synthetic_model
    from mppp.error import alignment as A
    model, X = _synthetic_model(noise_px=0.2)
    for pid in list(model.points)[:80]:                      # 80 points seen by two images only
        p = model.points[pid]
        p.image_ids, p.point2D_idxs = p.image_ids[:2], p.point2D_idxs[:2]
    e = tmp_path / "work" / "colmap" / "error_input"
    (e / "native").mkdir(parents=True)
    (e / "stations.csv").write_text("name,station,instrument\n" + "".join(
        f"img{k}.png,{'A' if k <= 3 else 'B'},NL\n" for k in range(1, 7)))
    import copy
    monkeypatch.setattr(A, "read_colmap", lambda path: copy.deepcopy(model))
    a2 = A.load_alignment(tmp_path / "work", min_track_length=2)
    a3 = A.load_alignment(tmp_path / "work", min_track_length=3)
    assert len(a2.model.points) == 200 and len(a3.model.points) == 120
    assert a3.points_before_track_filter == 200 and a3.min_track_length == 3
    rows = A.eps_table(a3)
    top = rows[0]
    assert {"n_points", "eps_dof_px"} <= set(top) and top["n_points"] == 120
    # 120 points x 6 observations: eps_dof = eps sqrt(2N / (2N - 3P)) with N = 720, P = 120
    assert abs(top["eps_dof_px"] / top["eps_px"] - np.sqrt(1440 / (1440 - 360))) < 1e-9


def _synthetic_pairs_v0p22_2(rng, n=40000, A=0.8, theta_bar=5.0, cv=0.6, tau_h=3.0, s_intra=0.7):
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


def _alignment(tmp_path, monkeypatch, stations_csv):
    from helpers import _synthetic_model
    from mppp.error import alignment as A
    model, X = _synthetic_model(noise_px=0.2)
    e = tmp_path / "work" / "colmap" / "error_input"
    (e / "native").mkdir(parents=True)
    (e / "stations.csv").write_text(stations_csv)
    monkeypatch.setattr(A, "read_colmap", lambda path: copy.deepcopy(model))
    return A, A.load_alignment(tmp_path / "work", min_track_length=3)


def test_fit_gate_recovers_all_parameters_and_ranks_the_forms():
    from mppp.error import alignment as A
    rng = np.random.default_rng(3)
    pairs = _synthetic_pairs_v0p22_2(rng)
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
    pairs = _synthetic_pairs_v0p22_2(rng, theta_bar=5000.0, cv=0.6)          # no decline with angle inside 30 deg
    g = A.fit_gate(pairs, form="power", illumination="dlmst", n_boot=0)
    assert not g["constrained"]
    ge = A.fit_gate(pairs, form="exp", illumination="dlmst", n_boot=0)
    assert not ge["constrained"] and (ge["at_limit"] or ge["theta_bar_deg"] >= 30)


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
    from helpers import _synthetic_model
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


def test_tau_replaces_L0_in_the_error_package():
    import mppp.error.alignment as A
    import mppp.error.core as C
    assert not any(n.startswith("L0") or "L0" in n for n in dir(A) + dir(C) if n.isupper() or "_" in n and "L0_" in n)
    assert "tau_h" in A.ILLUMINATION["dlmst"]
    import inspect
    assert "fit_tau" in inspect.signature(A.fit_gate).parameters
