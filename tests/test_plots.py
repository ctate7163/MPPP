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


def test_refined_rig_and_intrinsics_figures(tmp_path):
    """The study's rig-trends and intrinsics-vs-temperature formats draw from a small rig_residuals / consensus
    fixture and the block bins."""
    from mppp.calib import constrain as C, fit as F, models as M
    b = F.block_navcam_results(_fake_block(tmp_path))
    rr = {"blocks": [{"block": "Fake [sols 90-110]", "site": "Fake", "sol_median": 100.0 + 10 * k, "T_median_degC": -18.0 - k,
                      "images": 12, "yaw_mdeg": 1.0, "pitch_mdeg": -0.5, "roll_mdeg": 2.0} for k in range(3)],
          "fits": {a: {"vs_sol": {"slope": 0.1, "se": 0.5, "z": 0.2}, "vs_T": {"slope": 0.05, "se": 0.1, "z": 0.5}}
                   for a in ("pitch", "yaw", "roll")}}
    nl, nr = M.load_camera("N", "L"), M.load_camera("N", "R")
    cons = {"NL": dict(zip(nl["param_names"], nl["params"])), "NR": dict(zip(nr["param_names"], nr["params"])),
            "thermal": {"axis_ppm": {"NL": [40.5, 46.9], "NR": [38.9, 46.2]}, "pp_slopes": {"NL": [0.013, 0.0], "NR": [-0.041, 0.0]}}}
    f1 = C.plot_rig_trends(rr, cons, M.load_rig("N")["drift"], tmp_path / "rig.png", cx_inuse_px_per_degC=-0.05, label="test")
    f2 = C.plot_intrinsics_vs_temperature(cons, {"NL": nl, "NR": nr}, F.thermal_bin_rows([b]), rr, tmp_path / "intr.png", label="test")
    assert f1.stat().st_size > 10000 and f2.stat().st_size > 10000


def test_site_map_and_lmst(tmp_path):
    from mppp.waypoints import blocks as B
    from mppp.waypoints.table import snapshot_path
    from mppp.waypoints.sites import load_site_table
    tab = load_site_table()
    sw = B.site_waypoints(tab["sites"], snapshot_path())
    assert "rochette" in sw and sw["rochette"]["first_sol"] >= 178
    f = B.plot_site_map(None, snapshot_path(), tmp_path / "map.png", highlight=["rochette"], dpi=80)   # no terrain
    assert f.stat().st_size > 10000
    work = tmp_path / "scapes"
    for name, sol, hours in (("mars2020_sol_0178_rochette", 178, (13.5, 14.0, 15.2)), ("mars2020_sol_0460_rockytop", 460, (9.0, 12.5))):
        d = work / name / "processed"
        d.mkdir(parents=True)
        recs = [{"source_product": f"NLF_{sol}_{i}.IMG", "filename": {"family": "N", "downsample_scale": 1.0}, "native_size": [5120, 3840],
                 "LMST": f"Sol-{sol:05d}M{int(h):02d}:{int((h % 1) * 60):02d}:00.000", "sol": sol, "site": 6, "drive": 100 + i,
                 "pose": {"C_enu_m": [i * 3.0, 0.0, 0.0]}} for i, h in enumerate(hours)]
        (d / "mppp_manifest_v0p90.json").write_text(json.dumps({"images": recs}))
    images, used = B.lmst_images([work])
    assert set(images) == {"rochette", "rockytop"} and len(used) == 2
    rows = sorted((B.lmst_site_row(k, list(v.values())) for k, v in images.items()), key=lambda r: r["first_sol"])
    assert rows[0]["site"] == "rochette" and rows[0]["navcam"] == 3 and rows[0]["stations"] == 3
    nN, nZ = B.plot_lmst_histograms(images, rows, tmp_path / "lmst.png", window=(6, 18))
    assert (nN, nZ) == (5, 0) and (tmp_path / "lmst.png").stat().st_size > 10000
    assert B.write_lmst_table(rows, tmp_path / "lmst.csv").read_text().startswith("site,")


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


def test_rig_drift_fit_free_knot(tmp_path):
    """fit_rig_drift recovers a hinge drift (knot, rates) from synthetic epochs; sol0 = 0 convention; the candidate
    rig file reproduces the fit through rig_at."""
    from mppp.calib import constrain as C, fit as F, models as M
    from mppp.calib.joint import rig_angles
    rng = np.random.default_rng(2)
    rows = []
    for sol in np.linspace(20, 2000, 30):
        pitch = -100 + 0.006 * sol + (0.02 - 0.006) * min(sol, 400)
        rows.append({"block": f"b{int(sol)}", "epoch": "N", "sol": float(sol), "T": float(rng.uniform(-25, -12)), "images": 100,
                     "pitch_mdeg": pitch + rng.normal(0, 0.3), "yaw_mdeg": 38 + 0.0025 * sol, "roll_mdeg": -58 - 0.004 * sol + rng.normal(0, 2),
                     "sd_pitch_mdeg": 0.3, "sd_yaw_mdeg": 0.3, "sd_roll_mdeg": 0.3})
    m = F.fit_rig_drift(rows, knot_sol="free", knot_grid=range(100, 1001, 20))
    assert m["sol0"] == 0.0 and m["knot_ref"] == 0.0 and abs(m["knot_sol"] - 400) <= 40
    assert m["pitch_mdeg_per_sol"] == pytest.approx(0.006, abs=0.0005) and m["pitch_early_mdeg_per_sol"] == pytest.approx(0.02, abs=0.002)
    assert m["pitch_at_sol0_mdeg"] == pytest.approx(-100, abs=0.5) and m["roll_tau_mdeg"] > 1.0
    rig = F.rig_with_drift(M.load_rig("N"), m, keep_from_file=())
    a0 = rig_angles(C.rig_at(rig, sol=0.0)[0]); a1 = rig_angles(C.rig_at(rig, sol=1000.0)[0])
    assert a0["pitch_mdeg"] == pytest.approx(m["pitch_at_sol0_mdeg"], abs=0.05)
    assert a1["pitch_mdeg"] - a0["pitch_mdeg"] == pytest.approx(0.006 * 1000 + 0.014 * 400, abs=1.0)


def test_panel_letters_limits_and_unlabelled_versions(tmp_path, monkeypatch):
    """22 Oct (Christian): every Navcam and Mastcam-Z subplot lettered A, B ...; rig trends pitch -10..30, yaw and
    roll -20..20 mdeg, the stereo cx panel -20..-17 px; make_figures also writes the plots without point labels."""
    from mppp.calib import constrain as C, fit as F, models as M
    seen = []
    orig = C.panel_letters
    def spy(axs, **kw):
        orig(axs, **kw)
        seen.append([(t.get_text(), ax.get_ylim(), len(ax.texts)) for ax in np.ravel(axs) if ax.axison
                     for t in ax.texts if t.get_fontweight() == "bold"])
    monkeypatch.setattr(C, "panel_letters", spy)
    rr = {"blocks": [{"block": f"Fake{k} [sols 90-110]", "site": f"Fake{k}", "sol_median": 100.0 + 300 * k,
                      "T_median_degC": -18.0 - 3 * k, "images": 12, "yaw_mdeg": 1.0 + k, "pitch_mdeg": -0.5, "roll_mdeg": 2.0}
                     for k in range(4)],
          "fits": {a: {"vs_sol": {"slope": 0.1, "se": 0.5, "z": 0.2}, "vs_T": {"slope": 0.05, "se": 0.1, "z": 0.5}}
                   for a in ("pitch", "yaw", "roll")}}
    nl, nr = M.load_camera("N", "L"), M.load_camera("N", "R")
    cons = {"NL": dict(zip(nl["param_names"], nl["params"])), "NR": dict(zip(nr["param_names"], nr["params"])),
            "thermal": {"axis_ppm": {"NL": [40.5, 46.9], "NR": [38.9, 46.2]}, "pp_slopes": {"NL": [0.013, 0.0], "NR": [-0.041, 0.0]}}}
    drift = M.load_rig("N")["drift"]
    a = C.plot_rig_trends(rr, cons, drift, tmp_path / "rig.png")
    b = C.plot_rig_trends(rr, cons, drift, tmp_path / "rig_nolabels.png", point_labels=False)
    assert a.stat().st_size != b.stat().st_size
    lab, nolab = seen
    assert [x[0] for x in lab] == list("ABCD")
    assert [x[1] for x in lab] == [(-10.0, 30.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, -17.0)]   # pitch, roll, yaw, stereo cx
    assert all(n1 > n2 for (_, _, n1), (_, _, n2) in zip(lab, nolab))      # the block names are gone
    rows = [{"block": f"mars2020_blk{k}", "sol": 100.0 + 200 * k, "T_optics_degC": -10.0 + 4 * k, "yaw_mdeg": 0.3 * k,
             "pitch_mdeg": -0.2 * k, "roll_mdeg": 0.1, "dbaseline_mm": 0.05 * k} for k in range(4)]
    zm = {"rig": {"residuals": {"per_block": rows}, "per_block": [], "baseline_m": 0.2428}}
    C.plot_zcam_rig_trends(zm, tmp_path / "z.png")
    C.plot_zcam_rig_trends(zm, tmp_path / "z_nolabels.png", point_labels=False)
    assert [x[0] for x in seen[2]] == list("ABCDEFGH") and len(seen[3]) == 8
