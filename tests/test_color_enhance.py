"""0.99.33: the three display transforms of a 16-bit linear panorama (rpp_studies.misc.color_enhance) - Christian,
18 Nov 2026: "Prepare examples of each three color maps made from the highest res 16 bit images"; "The decorrelation
stretch should exclude the sky and rover"."""
import json

import cv2
import numpy as np


def _pano(tmp_path, H=720, W=1440):
    rng = np.random.default_rng(0)
    el = 90.0 - (np.arange(H) + 0.5) * 180.0 / H
    img = np.zeros((H, W, 3), np.float32)
    sky = el[:, None] > 3.0 + 2.0 * np.sin(np.linspace(0, 6.28, W))[None, :]
    terr = ~sky
    base = np.array([0.45, 0.30, 0.22], np.float32) * 20000
    img[terr] = base * (0.6 + 0.4 * rng.random((int(terr.sum()), 1)))
    img[sky] = np.array([0.75, 0.62, 0.52], np.float32) * 50000
    rover = np.zeros((H, W), bool)
    rover[int(0.75 * H):, 600:800] = True
    img[rover] = np.array([0.5, 0.5, 0.5], np.float32) * 30000
    f = tmp_path / "S_ods_left_16.png"
    cv2.imwrite(str(f), np.clip(img[..., ::-1], 1, 65535).astype(np.uint16))
    lab = np.where(sky, 2, 1).astype(np.uint8)
    np.savez(tmp_path / "d.npz", range=np.ones((H, W), np.float32), label=lab, rover=np.zeros((H, W), np.uint8))
    np.savez(tmp_path / "dr.npz", range=np.ones((H, W), np.float32), label=lab, rover=rover.astype(np.uint8))
    return f, sky, rover


def test_sky_and_rover_left_out_of_the_decorrelation_stretch(tmp_path, monkeypatch):
    from rpp_studies.misc import color_enhance as CE
    f, sky, rover = _pano(tmp_path)
    monkeypatch.setattr(CE, "A_RADIUS_DEG", 2.0)
    out = tmp_path / "out"
    rec = CE.run(str(f), str(out), "S", depth=str(tmp_path / "d.npz"), rover_depth=str(tmp_path / "dr.npz"))
    m = cv2.imread(str(out / "S_C_mask_sky_blue_rover_green.png"))
    sky_m, rov_m = m[..., 0] > 0, m[..., 1] > 0
    s4, r4 = sky[::4, ::4], rover[::4, ::4]
    assert (sky_m & s4).sum() / s4.sum() > 0.97 and (sky_m & ~s4).sum() / (~s4).sum() < 0.02   # the sky found, no terrain
    assert (rov_m & r4).sum() == r4.sum()                                                     # the rover (and a margin)
    assert rec["C"]["pixels"] == "terrain only: no sky, no rover" and len(rec["C"]["matrix"]) == 3
    # outside the terrain the stretch is the reference view
    c = cv2.imread(str(out / "S_pano_C_decorrelation.jpg")).astype(int)
    g = cv2.imread(str(out / "S_pano_reference_gamma.jpg")).astype(int)
    keep_out = (sky_m | rov_m)
    assert np.abs(c - g)[keep_out].mean() < 2 and np.abs(c - g)[~keep_out].mean() > 5
    for k in ("A_base_detail", "B_mantiuk", "reference_gamma"):
        assert (out / f"S_pano_{k}.jpg").is_file() and (out / f"S_crop_{k}.jpg").is_file()
    assert json.loads((out / "S_parameters.json").read_text())["A"]["base_compression"] <= 1.0


def test_guided_filter_keeps_edges():
    from rpp_studies.misc.color_enhance import guided_filter
    I = np.zeros((64, 64), np.float32)
    I[:, 32:] = 1.0
    I += np.random.default_rng(1).normal(0, 0.02, I.shape).astype(np.float32)
    q = guided_filter(I, 4, 0.01)
    assert q[:, 36:].mean() > 0.95 and q[:, :28].mean() < 0.05 and q[:, 36:].std() < I[:, 36:].std()
