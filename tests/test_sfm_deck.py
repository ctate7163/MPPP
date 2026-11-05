"""The rover deck targets inside the bundle adjustment (mppp.sfm.deck, 4 Nov 2026)."""
import numpy as np
import pycolmap
import pytest

from helpers import _build_rec, _synthetic


def test_survey_tie_jacobian():
    from mppp.sfm.deck import survey_tie_class
    from scipy.spatial.transform import Rotation
    Tie = survey_tie_class()
    P = np.array([0.4, -0.2, -1.1])
    for sf in (False, True):
        c = Tie(P, 1e-4, scale_free=sf)
        T = np.r_[0.01, -0.02, 0.3, 1.0, 2.0, 3.0, 0.002 if sf else 0.0]
        X = c.pred(T) + np.array([1e-5, -2e-5, 3e-5])
        r = np.zeros(3)
        J0, J1 = np.zeros(9), np.zeros(21)
        assert c.Evaluate([X, T], r, [J0, J1])
        assert np.allclose(r, [0.1, -0.2, 0.3], atol=1e-6)
        num = np.zeros((3, 7))
        for i in range(7):
            Tp = T.copy(); Tp[i] += 1e-6
            rp = np.zeros(3); c.Evaluate([X, Tp], rp, None)
            num[:, i] = (rp - r) / 1e-6
        if not sf:
            num[:, 6] = 0.0
        assert np.allclose(J1.reshape(3, 7), num, atol=2e-2 * np.abs(num).max())
        s = np.exp(T[6]) if sf else 1.0
        assert np.allclose(c.pred(T), s * Rotation.from_rotvec(T[:3]).apply(P) + T[3:6])


def test_interval_of():
    from mppp.sfm.deck import interval_of
    r = {"station": "S032D1184", "sol": 686}
    assert interval_of(r) == "S032D1184" and interval_of(r, "sol") == "S032D1184_sol0686"
    assert interval_of({"site": 32, "drive": 1174, "sol": 684}) == "S032D1174"


def _deck_block(tmp_path, baseline_error=0.01):
    """The synthetic two-station block with a rover deck in view of station 1's pan; the rig's baseline started
    ``baseline_error`` long.  -> rec, proj, obs, T_true, true baseline (m)."""
    from mppp.calib import fiducials as F
    from scipy.spatial.transform import Rotation
    proj, truth, P, cams_d, rigT, noise, rng = _synthetic(tmp_path, noise_native_px=0.1)
    rec, _ = _build_rec(proj, truth, P, cams_d, rigT, noise, rng, perturb=False)
    rig = rec.rigs[1]
    s2 = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=2)
    T0 = rig.sensor_from_rig(s2)
    rig.set_sensor_from_rig(s2, pycolmap.Rigid3d(T0.rotation, np.asarray(T0.translation) * (1 + baseline_error)))
    # the deck: the survey (RNAV) rotated and placed 1.4 m below and ahead of station 1's first exposures
    names = ("TDF1", "TDF2", "TDF3", "AFT1", "RPFA1", "RPFA2")
    S = F.survey_nav_m(names)
    cen = np.mean([S[k] for k in names], 0)
    R = Rotation.from_rotvec([0.02, -0.01, 0.6]).as_matrix()
    t = np.array([0.0, 1.4, 0.6]) - R @ cen
    T_true = np.r_[Rotation.from_matrix(R).as_rotvec(), t, 0.0]
    cam = {cid: pycolmap.Camera(camera_id=cid, model="FULL_OPENCV", width=5120, height=3840,
                                params=np.array(cams_d[k]["params"], float)) for k, cid in (("NL", 1), ("NR", 2))}
    obs = []
    for r in proj.images:
        if r["station"] != "S001D0000":
            continue
        Rc, C = truth[r["name"]]
        cid = 1 if r["eye"] == "L" else 2
        for k in names:
            Xc = Rc @ (R @ S[k] + t - C)
            if Xc[2] < 0.3:
                continue
            uv = cam[cid].img_from_cam(Xc[None])[0]
            if 50 < uv[0] < 5070 and 50 < uv[1] < 3790:
                obs.append({"image_id": r["image_id"], "stem": r["stem"], "target": k, "interval": "S001D0000",
                            "uv": uv + rng.normal(0, 0.1, 2) / r["downsample_scale"]})
    return rec, proj, obs, T_true, float(np.linalg.norm(rigT[1]))


def test_deck_sets_the_baseline(tmp_path):
    """With the rig translation refined, the deck survey (rigid) brings a 1 % long baseline back to the truth."""
    from mppp.sfm.deck import make_hook, deck_residuals
    from mppp.sfm.reconstruction import bundle_adjust, _scales
    rec, proj, obs, T_true, b_true = _deck_block(tmp_path)
    assert len(obs) >= 8 and len({o["image_id"] for o in obs}) >= 3
    T = {"S001D0000": np.ascontiguousarray(T_true + np.r_[0.003, -0.002, 0.004, 0.01, -0.01, 0.005, 0.0])}
    hook, state = make_hook(obs, T, sigma_px=0.1, survey_sigma_mm=0.01, scales=_scales(rec, proj))
    out = bundle_adjust(rec, proj, sigma_px=0.1, loss_scale=10.0, refine_intrinsics=False, refine_rig=True,
                        rig_translation_sigma_m=0.05, use_priors=False, attitude_prior_deg=None,
                        extra_residuals=hook, max_iterations=200)
    assert out["extra"]["deck"]["observations"] == len(obs)
    b = np.linalg.norm(rec.rigs[1].sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=2)).translation)
    assert abs(b / b_true - 1) < 1e-3                        # from +1 % to < 0.1 %
    e = deck_residuals(obs, rec, T)
    assert np.sqrt(np.mean(e ** 2) * 2) < 1.0


def test_deck_hold_poses_moves_only_T(tmp_path):
    from mppp.sfm.deck import make_hook
    from mppp.sfm.reconstruction import bundle_adjust, _scales
    rec, proj, obs, T_true, b_true = _deck_block(tmp_path, baseline_error=0.0)
    before = {i: np.array(rec.images[i].cam_from_world().translation) for i in rec.reg_image_ids()}
    T = {"S001D0000": np.ascontiguousarray(T_true + np.r_[0.002, 0.0, -0.002, 0.004, 0.0, 0.0, 0.0])}
    hook, _ = make_hook(obs, T, sigma_px=0.1, survey_sigma_mm=0.01, scales=_scales(rec, proj), hold_poses=True)
    bundle_adjust(rec, proj, sigma_px=0.1, refine_intrinsics=False, refine_rig=False, use_priors=False,
                  attitude_prior_deg=None, extra_residuals=hook, max_iterations=100)
    for i, t in before.items():
        assert np.allclose(rec.images[i].cam_from_world().translation, t, atol=1e-9)
    assert np.allclose(T["S001D0000"][:3], T_true[:3], atol=5e-4)              # rad (the deck seen from one side)
    assert np.allclose(T["S001D0000"][3:6], T_true[3:6], atol=1e-3)            # m
