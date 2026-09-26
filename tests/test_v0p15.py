"""v0p15: error analysis of COLMAP alignments (mppp.error.alignment), observed-ray decorrelation."""
import numpy as np
import pytest


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


def test_unproject_inverts_project():
    from mppp.colmap import project_camera, unproject_camera
    p = [2951, 2951, 2594, 1942, -0.27, 0.10, 1.7e-4, 1.7e-4, -0.02, 0, 0, 0]
    xy = np.random.default_rng(1).uniform(-0.8, 0.8, (500, 2))
    uv = project_camera("FULL_OPENCV", p, np.c_[xy, np.ones(len(xy))])
    assert np.abs(unproject_camera("FULL_OPENCV", p, uv) - xy).max() < 1e-8
    assert np.allclose(unproject_camera("SIMPLE_RADIAL", [1000, 500, 400, 0.0], [[600, 400]]), [[0.1, 0.0]])


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
