"""0.99.34 (Christian, 19 Nov 2026: "Z34 needs to be anchor to the fiducials in order to fix its position in the mast
frame. Prepare that study."): rpp_studies.calibration.zcam_fiducials on synthetic deck frames."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _frames(rng, ch, zm, truth, pointing_mrad=0.5, noise_px=0.25):
    from rpp.calib.constrain import zcam_camera_at
    from rpp_studies.calibration import fiducials as FD
    from rpp_studies.calibration import zcam_fiducials as ZF
    P = FD.survey_nav_m(list(ZF.TARGETS))
    frames, obs = [], []
    for eye, (c, q) in truth.items():
        for az in np.radians(np.arange(0, 120, 1.5)):
            for el in np.radians(np.arange(30, 70, 1.5)):
                focus = float(rng.uniform(-600, 400))
                name, par, _ = zcam_camera_at(zm, f"Z{eye}034", focus)
                pr = FD.Projector(name, par, 1648, 1200)
                d = rng.normal(0, 1e-3 * pointing_mrad, 3)
                R, C = ZF.pose(ch, c, q, az, el, d)
                seen = {}
                for k, X in P.items():
                    uv = pr((R @ (X - C))[None])[0]
                    if np.all(np.isfinite(uv)) and 30 < uv[0] < 1618 and 30 < uv[1] < 1170:
                        seen[k] = uv
                if len(seen) < 2:
                    continue
                stem = f"Z{eye}_{len(frames):04d}"
                R0, C0 = ZF.pose(ch, c + rng.normal(0, 0.004, 3), q + rng.normal(0, 0.003, 3), az, el)
                frames.append({"stem": stem, "eye": eye, "group": f"Z{eye}034", "focus": focus, "az": float(az), "el": float(el),
                               "label_R": R0.tolist(), "label_C": C0.tolist()})
                for k, uv in seen.items():
                    obs.append({"stem": stem, "target": k, "x": float(uv[0] + rng.normal(0, noise_px)),
                                "y": float(uv[1] + rng.normal(0, noise_px))})
    return frames, obs


def test_the_deck_fixes_the_mastcam_z_centers_along_the_boresight():
    from rpp_studies.calibration import zcam_fiducials as ZF
    rng = np.random.default_rng(5)
    ch = ZF.load_chain()
    zm = json.loads((ROOT / "src" / "rpp" / "data" / "cmods" / "M2020_ZCAM034_focus_model.json").read_text())
    p = ch["params"]
    truth = {"L": (np.asarray(p["c_L_m"]) + [0.004, -0.088, 0.002], np.asarray(p["Q_L_rotvec_rad"])),
             "R": (np.asarray(p["c_R_m"]) + [0.006, 0.092, 0.002], np.asarray(p["Q_R_rotvec_rad"]))}
    frames, obs = _frames(rng, ch, zm, truth)
    assert len(frames) > 30 and len(obs) > 80
    for how in ("law", "free"):
        f = ZF.fit(frames, obs, ch, zm, focal=how)
        assert f["rms_px"] < 0.6
        for e, (c, q) in truth.items():
            v = f["eyes"]["Z" + e]
            bore = ZF._RV(q).T @ np.array([0.0, 0.0, 1.0])
            err = 1e3 * (np.asarray(v["c_m"]) - c)
            assert abs(err @ bore) < max(1.0, 4 * v["c_along_boresight_sd_mm"]), (how, e, err, v)
            assert np.all(np.abs(err) < 2.0), (how, e, err)
            if how == "free":
                assert abs(v["s_ppm"]) < max(800.0, 4 * v["s_sd_ppm"])
        true_b = 1e3 * np.linalg.norm(truth["R"][0] - truth["L"][0])
        assert f["pair"]["baseline_mm"] == pytest.approx(true_b, abs=1.0)
    lines = ZF.describe({"law": f}, 34)
    assert any("ZR center in the ZL frame" in ln for ln in lines) and any("N+Z34 joint" in ln for ln in lines)


def test_chain_pose_and_its_inverse_agree():
    from rpp_studies.calibration import zcam_fiducials as ZF
    ch = ZF.load_chain()
    c, q = np.array([0.01, -0.12, 0.13]), np.array([0.02, -0.03, 1.5])
    R, C = ZF.pose(ch, c, q, 0.7, 1.1)
    c2, q2 = ZF.mount_from_pose(ch, R, C, 0.7, 1.1)
    assert np.allclose(c2, c, atol=1e-9) and np.allclose(ZF._RV(q2), ZF._RV(q), atol=1e-9)
