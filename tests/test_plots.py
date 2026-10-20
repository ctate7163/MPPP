"""The 0.90 plotting functions, each in its module: camera model vs temperature and rig vs sol (mppp.calib.constrain),
the block table (mppp.calib.fit), site map and LMST histograms (mppp.waypoints.blocks), undistortion (mppp.colmap)."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _fake_block(tmp, name="mars2020_sol_0100_fake", T=-18.0, sol=100):
    """A block folder with project.json, cameras.bin and rigs.bin: two eyes, one bin each, one rig."""
    from mppp.calib import models as M
    from mppp.calib.constrain import camera_at, rig_at
    from scipy.spatial.transform import Rotation
    nl, nr, rig = M.load_camera("N", "L"), M.load_camera("N", "R"), M.load_rig("N")
    cams = {}
    p = tmp / name / "colmap" / "sparse" / "cahv_ba"
    p.mkdir(parents=True)
    keys = {"NL": 1, "NR": 2, "NL_T-020-015": 3, "NR_T-020-015": 4}
    for key, cid in keys.items():
        model, params, _ = camera_at(nl if key.startswith("NL") else nr, T_degC=T, sol=sol)
        params = list(params)
        if "_T" in key:
            params[0] *= 1 + 50e-6                                # the bin sits 50 ppm above the joint
        cams[cid] = (model, params)
    with open(p / "cameras.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(cams)))
        for cid, (model, params) in cams.items():
            fh.write(struct.pack("<Ii", cid, 10) + struct.pack("<QQ", 5120, 3840) + struct.pack("<12d", *params))
    R, _t, _ = rig_at(rig, sol=sol)
    q = Rotation.from_matrix(R).as_quat()                          # x y z w
    with open(p / "rigs.bin", "wb") as fh:
        fh.write(struct.pack("<Q", 1) + struct.pack("<II", 1, 2) + struct.pack("<iI", 0, 1) + struct.pack("<iI", 0, 2)
                 + struct.pack("<B", 1) + struct.pack("<4d", q[3], q[0], q[1], q[2]) + struct.pack("<3d", -0.424, 0, 0))
    images = [{"name": f"N{e}F_{i}.png", "instrument": f"N{e}_T-020-015", "eye": e, "sol": sol, "rig_epoch": 0,
               "camera_temperature_degC": T, "camera_id": keys[f"N{e}_T-020-015"]}
              for e in ("L", "R") for i in range(6)]
    (p.parents[1] / "project.json").write_text(json.dumps({
        "images": images, "cameras": {k: {"params": cams[c][1]} for k, c in keys.items()}, "rig": {},
        "settings": {"database": {"cameras": keys, "rig_ids": {"N": 1}}}}))
    return p.parents[1].parent


def test_block_results_and_table(tmp_path):
    from mppp.calib import fit as F, models as M
    b = F.block_navcam_results(_fake_block(tmp_path))
    assert b["site"] == "fake" and set(b["cameras"]) == {"NL_T-020-015", "NR_T-020-015"} and "N" in b["rigs"]
    rows = F.navcam_block_table([b], M.load_camera("N", "L"), M.load_camera("N", "R"), M.load_rig("N"))
    r = rows[0]
    assert r["block"] == "fake" and r["bins"] == 2
    assert r["NL_fx_ppm"] == pytest.approx(50, abs=3) and abs(r["NL_fy_ppm"]) < 3 and abs(r["NL_cx_px"]) < 1e-6
    assert abs(r["rig_yaw_mdeg"]) < 1e-6 and abs(r["rig_pitch_mdeg"]) < 1e-6
    md = F.block_table_markdown(rows)
    assert "| fake |" in md and "| yes |" in md
    assert F.block_table_markdown(rows, focal_limit_ppm=10).count("**no**") == 1


def test_camera_and_rig_plots(tmp_path):
    from mppp.calib import constrain as C, fit as F, models as M
    b = F.block_navcam_results(_fake_block(tmp_path))
    f1 = C.plot_camera_vs_temperature(M.load_camera("N", "L"), M.load_camera("N", "R"), tmp_path / "cmod.png", blocks=[b])
    f2 = C.plot_rig_vs_sol(M.load_rig("N"), tmp_path / "rig.png", blocks=[b])
    assert f1.stat().st_size > 10000 and f2.stat().st_size > 10000


def test_site_map_and_lmst(tmp_path):
    from mppp.waypoints import blocks as B
    from mppp.waypoints.sites import load_site_table
    from mppp.waypoints.table import snapshot_path
    tab = load_site_table()
    sw = B.site_waypoints(tab["sites"], snapshot_path())
    assert "rochette" in sw and sw["rochette"]["first_sol"] >= 178
    f = B.plot_site_map(tab["sites"], snapshot_path(), tmp_path / "map.png", highlight=["rochette"])   # no terrain
    assert f.stat().st_size > 10000
    rng = np.random.default_rng(0)
    data = {"A": {"lmst_h": rng.uniform(8, 17, 50), "T": rng.uniform(-30, -10, 50), "sol": np.arange(50) + 100},
            "B": {"lmst_h": rng.uniform(12, 16, 20), "T": rng.uniform(-20, -5, 20), "sol": np.arange(20) + 500}}
    assert B.plot_lmst_histograms(data, tmp_path / "lmst.png").stat().st_size > 10000


def test_manifest_lmst(tmp_path):
    from mppp.waypoints.blocks import manifest_lmst_temperature
    m = {"images": [{"camera_group": "NL0", "lmst_h": 13.5, "camera_temperature_degC": -18.0, "sol": 178},
                    {"camera_group": "ZL034", "lmst_h": 14.0, "camera_temperature_degC": -10.0, "sol": 178},
                    {"camera_group": "NR0", "lmst_h": None, "camera_temperature_degC": -18.0, "sol": 178}]}
    (tmp_path / "m.json").write_text(json.dumps(m))
    d = manifest_lmst_temperature(tmp_path / "m.json")
    assert d["lmst_h"].tolist() == [13.5] and d["T"].tolist() == [-18.0]


def test_undistort_and_example(tmp_path):
    from mppp.calib import models as M
    from mppp.calib.constrain import camera_at
    from mppp.colmap import plot_undistorted_example, undistort_image
    model, p, _ = camera_at(M.load_camera("N", "L"), T_degC=-20, sol=1000)
    im = np.zeros((240, 320, 3), np.uint8)
    im[::20, :] = 255
    im[:, ::20] = 255
    u = undistort_image(im, model, p, 5120, mask=np.full((240, 320), 255, np.uint8))
    assert u["image"].shape == im.shape and u["mask"].shape == (240, 320) and 60 < u["field_deg"] < 110
    assert u["f_px"] < u["f_in_px"]                                   # fit="width" of a fisheye: a shorter pinhole
    f = plot_undistorted_example(im, model, p, 5120, tmp_path / "u.png", title="t")
    assert f.stat().st_size > 5000
