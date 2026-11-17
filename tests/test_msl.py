"""0.99.4 (Christian, 16 Nov 2026: "Prepare a test msl block"): the MSL first-block study's names, archive listing and
block choice (rpp_studies.msl.first_block); the products themselves are tested on the PC."""
from __future__ import annotations

import json

import numpy as np
import pytest
from pathlib import Path

from rpp_studies.msl import first_block as F


def test_msl_names_lmst_and_what_is_fetched():
    f = F.parse_name("NLB_668441715RAD_F0870792NCAM00252M1.IMG")
    assert f["camera"] == "NLB" and f["family"] == "N" and f["eye"] == "L" and f["side"] == "B" and f["sclk"] == 668441715
    assert f["product_type"] == "RAD" and f["sampling"] == "F" and (f["site"], f["drive"]) == (87, 792) and f["sequence"] == "NCAM00252"
    assert f["group"] == "NLB" and F.parse_name("NRB_668441715RAD_F0870792NCAM00252M1")["pair_key"] == f["pair_key"]
    assert F.parse_name("FLB_668441382RAD_F0870792FHAZ00302M1")["family"] == "F"
    assert F.parse_name("NLF_0052_0671954445_000RAD_N0032046NCAM00500_0A00LLJ01") is None        # an M2020 name
    assert F.parse_name("1408ML0068980090601986C00_DRCL") is None                                 # Mastcam
    assert abs(F.lmst_hours("Sol-03052M15:04:27.460") - (15 + 4 / 60 + 27.46 / 3600)) < 1e-9 and F.lmst_hours(None) is None
    names = ["NLB_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715RAD_F0870792NCAM00252M1.LBL",
             "NRB_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715MXY_F0870792NCAM00252M1.IMG",
             "NLB_668441715XYZ_F0870792NCAM00252M1.IMG", "NLB_670399104RAD_S0871078NCAM00263M1.IMG",     # XYZ, subframe: no
             "NLA_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715RAD_F0880792NCAM00252M1.IMG",     # side A, site 88: no
             "NLB_668441999MXY_F0870792NCAM00252M1.IMG"]                                                 # MXY of no RAD: no
    assert F.wanted(names, 87, "B") == ["NLB_668441715RAD_F0870792NCAM00252M1", "NRB_668441715RAD_F0870792NCAM00252M1",
                                         "NLB_668441715MXY_F0870792NCAM00252M1"]
    assert F.wanted(names, 87, "B", drives=[1078]) == []


def test_the_block_is_the_richest_station_and_its_neighbors():
    rows = []
    for (site, drive, pairs, x) in ((87, 0, 10, 0.0), (87, 792, 12, 30.0), (87, 1078, 9, 200.0), (87, 1200, 3, 10.0)):
        for k in range(pairs):
            for eye in "LR":
                rows.append({"site": site, "drive": drive, "eye": eye, "sol": 3047 + drive // 100, "pair_key": f"{drive}_{k}",
                             "C_enu_m": [x + (0.2 if eye == "R" else 0.0), 0.0, 1.9]})
    frames, rep = F.choose_block(rows, min_pairs=8, radius_m=50.0)
    assert rep["anchor"] == "S087D0792" and {(r["site"], r["drive"]) for r in frames} == {(87, 0), (87, 792)}
    st = {s["station"]: s for s in rep["stations"]}
    assert st["S087D1078"]["in_block"] is False and st["S087D1200"]["in_block"] is False and st["S087D0000"]["pairs"] == 10
    frames, rep = F.choose_block(rows, min_pairs=8, radius_m=500.0, station=(87, 1078))
    assert rep["anchor"] == "S087D1078" and len({r["drive"] for r in frames}) == 3
    a = np.linspace(100, 6000, 1000).reshape(10, 100)
    s = F.stretch8(a)
    assert s.dtype == np.uint8 and s.min() == 0 and s.max() == 255
    pairs = F.stereo_baselines([{"pair_key": "p", "eye": "L", "image": "l.png", "C_enu_m": [0, 0, 0]},
                                {"pair_key": "p", "eye": "R", "image": "r.png", "C_enu_m": [0.424, 0, 0]}],
                               {"l.png": np.zeros(3), "r.png": np.array([0.4, 0.0, 0.0])})
    assert len(pairs) == 1 and abs(pairs[0]["ratio"] - 0.4 / 0.424) < 1e-12


def test_archive_listings_with_relative_and_absolute_links(monkeypatch):
    page = ('<a href="../">Parent</a> <a href="NLB_668441715RAD_F0870792NCAM00252M1.IMG">x</a> '
            "<a href='/data/msl/MSLNAV_1XXX/DATA/SOL03052/NLB_668441715RAD_F0870792NCAM00252M1.LBL'>y</a> "
            '<a href="?C=M;O=A">Last modified</a> <a href="SUB/">sub</a>')
    monkeypatch.setattr(F, "_get", lambda url, timeout=120.0: page.encode())
    assert F.listing("https://example/") == ["NLB_668441715RAD_F0870792NCAM00252M1.IMG", "NLB_668441715RAD_F0870792NCAM00252M1.LBL"]
    url, names, tried = F.find_sol_dir(3052, bases=("https://a/MSLNAV_1XXX",))
    assert url == "https://a/MSLNAV_1XXX/DATA/SOL03052/" and len(names) == 2 and len(tried) == 1


def _info(site, drive, C, yaw_deg=0.0, eye="L", side="A", pair="p", link=None):
    # camera looking horizontally to azimuth yaw (from north, clockwise), 10 deg down; block frame east-north-up
    az, el = np.radians(yaw_deg), np.radians(-10.0)
    z = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return {"site": site, "drive": drive, "C_enu_m": list(C), "R_w2c": R.tolist(), "eye": eye, "side": side,
            "group": f"NL{side}" if eye == "L" else f"NR{side}", "pair_key": pair, "sol": 150, "sclk": 1, "site_link": link,
            "image": f"{site}_{drive}_{yaw_deg}_{eye}.png"}


def test_msl_block_sites_chain_selection_and_pairs():
    from rpp_studies.msl import block as B
    # site 6 is 100 m north and 20 m east of site 5, per the label link (NED offset of site 6's origin in site 5)
    link = {"site": 6, "parent": 5, "offset_ned_m": [100.0, 20.0, -1.0], "quaternion": [1, 0, 0, 0]}
    infos = [_info(5, 0, [0, 0, 1.9]), _info(5, 10, [3, 0, 1.9]), _info(6, 0, [0, 0, 1.9], link=link),
             _info(6, 0, [0.4, 0, 1.9], link=link), _info(7, 0, [0, 0, 1.9])]
    origin, rep = B.site_origins(infos)
    assert rep["anchor"] in (5, 6) and rep["left_out"] == [7]
    d = origin[6] - origin[5]
    assert np.allclose(d, [20.0, 100.0, 1.0])                     # ENU: east 20, north 100, up 1
    frames = B.place(infos, origin)
    assert len(frames) == 4 and all("C_block_m" in r for r in frames)
    kept, sel = B.select(frames, 30.0, rep["anchor"])
    assert sel["dropped"] == 2 and {r["site"] for r in kept} == {rep["anchor"]}   # the anchor site's frames, 100 m away
    cam = {"model": "FULL_OPENCV", "params": [1225.0, 1225.0, 512.0, 512.0] + [0.0] * 8, "width": 1024, "height": 1024}
    a = dict(_info(5, 0, [0, 0, 1.9], 0.0, pair="p"), C_block_m=[0, 0, 1.9], image="a.png")       # looking north
    b = dict(_info(5, 1, [6, 0, 1.9], 0.0, pair="q"), C_block_m=[6, 0, 1.9], image="b.png")       # 6 m east, also north
    c = dict(_info(5, 2, [0, 3, 1.9], 180.0, pair="r"), C_block_m=[0, 3, 1.9], image="c.png")    # looking south, away
    s = dict(_info(5, 0, [0.42, 0, 1.9], 0.0, eye="R", pair="p"), C_block_m=[0.42, 0, 1.9], image="s.png")  # a's partner
    pairs = set(B.image_pairs([a, b, c, s], {"NLA": cam, "NRA": cam}, max_distance_m=60.0, per_image=10))
    assert ("a.png", "b.png") in pairs and ("a.png", "s.png") in pairs
    assert ("a.png", "c.png") not in pairs and ("b.png", "c.png") not in pairs


def test_msl_waypoint_table_and_opencv_fit(tmp_path):
    import json
    from rpp.waypoints.table import MISSIONS, load_waypoints, local_en, mission_url
    assert mission_url("msl").endswith("/MSL/Layers/json/MSL_waypoints.json") and "m2020" in MISSIONS
    feats = [{"type": "Feature", "properties": {"sol": s, "site": 6, "drive": 10 * s, "elev_geoid": -4500.0},
              "geometry": {"type": "Point", "coordinates": [137.44 + 1e-4 * s, -4.59]}} for s in (124, 125, 126)]
    p = tmp_path / "MSL_waypoints.json"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    w = load_waypoints(path=p, mission="msl")
    assert w["_rpp_source"]["mission"] == "msl" and w["_rpp_source"]["n_features"] == 3
    rows = local_en(w)
    assert rows[0]["E"] == 0 and abs(rows[1]["E"] - np.radians(1e-4) * 3396190.0 * np.cos(np.radians(-4.59))) < 1e-6
    assert rows[2]["site"] == 6 and rows[2]["drive"] == 1260 and rows[0]["elev"] == -4500.0
    # the OpenCV k1 k2 p1 p2 k3 camera reproduces a CAHVOR (a synthetic Navcam-like one)
    from rpp.calib.labelfit import opencv5_from_cahvore
    from rpp.cmod import CameraModel
    gcm = {"MODEL_TYPE": "CAHVOR", "MODEL_COMPONENT_1": [0.0, 0.0, 0.0], "MODEL_COMPONENT_2": [1.0, 0.0, 0.0],
           "MODEL_COMPONENT_3": [512.0, 0.0, 1225.0], "MODEL_COMPONENT_4": [512.0, -1225.0, 0.0],
           "MODEL_COMPONENT_5": [1.0, 0.0, 0.0], "MODEL_COMPONENT_6": [0.0, -0.002, 0.003]}
    cm = CameraModel.from_label(gcm, 1024, 1024)
    mc, _ = cm.camera_frame()
    p12, rep = opencv5_from_cahvore(mc, 1024, 1024, step=32)
    assert rep["rms_px"] < 0.05 and p12[9:] == [0.0, 0.0, 0.0] and abs(p12[0] - 1225) < 5


def _synthetic_stereo_block(seed=2, pairs_per_side=8):
    """A block of NLA/NRA and NLB/NRB pairs with one true right-from-left pose per side, cameras started 10 px short."""
    import pycolmap
    rng = np.random.default_rng(seed)
    rec = pycolmap.Reconstruction()
    true = np.array([1225., 1226., 507., 503., -0.02, 0.01, 1e-4, -2e-4, 0.005, 0, 0, 0])
    groups = {1: "NLA", 2: "NRA", 3: "NLB", 4: "NRB"}
    for cid in groups:
        cam = pycolmap.Camera.create_from_model_name(cid, "FULL_OPENCV", 1000., 1024, 1024); cam.params = true.copy()
        rec.add_camera_with_trivial_rig(cam)
    X = rng.uniform([-8, -6, 6], [8, 6, 16], (600, 3))
    ids = [rec.add_point3D(x, pycolmap.Track()) for x in X]
    rel = {"A": (0.004, np.array([-0.42, 0.001, 0.002])), "B": (-0.003, np.array([-0.424, -0.001, 0.0]))}   # yaw rad, t
    images, iid = [], 0
    for side, (lc, rc) in (("A", (1, 2)), ("B", (3, 4))):
        for k in range(pairs_per_side):
            ang = 0.03 * (k - pairs_per_side / 2)
            Rl = pycolmap.Rotation3d(np.array([0, np.sin(ang / 2), 0, np.cos(ang / 2)]))
            left = pycolmap.Rigid3d(Rl, np.array([-0.5 * k + 1.5, 0.1 * (k % 3), 0.02 * (k % 2)]))
            yaw, t = rel[side]
            right = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0, np.sin(yaw / 2), 0, np.cos(yaw / 2)])), t) * left
            for eye, cid, pose in (("L", lc, left), ("R", rc, right)):
                iid += 1
                name = f"N{eye}{side}_{k:02d}.png"
                im = pycolmap.Image(image_id=iid, name=name, camera_id=cid)
                uv = rec.camera(cid).img_from_cam(np.array([pose * x for x in X]))
                im.points2D = pycolmap.Point2DList([pycolmap.Point2D(u + rng.normal(0, 0.2, 2), pid) for u, pid in zip(uv, ids)])
                rec.add_image_with_trivial_frame(im, pose)
                R = pose.rotation.matrix()
                C = -(R.T @ pose.translation)
                images.append({"image": name, "group": groups[cid], "family": "N", "eye": eye, "side": side,
                               "pair_key": f"N{side}_{k}", "site": 6 if k < 5 else 7, "drive": k, "sol": 160 + k,
                               "ccd_degC": -10.0 - k, "R_w2c": R.tolist(), "C_block_m": list(C + rng.normal(0, 0.03, 3))})
    for i in range(1, iid + 1):
        for k, p2 in enumerate(rec.image(i).points2D):
            rec.point3D(p2.point3D_id).track.add_element(i, k)
    start = [1215., 1216., 507., 503., 0, 0, 0, 0, 0, 0, 0, 0]
    for cid in groups:
        rec.camera(cid).params = np.array(start)
    cams = {g: {"model": "FULL_OPENCV", "params": start, "width": 1024, "height": 1024, "frames": pairs_per_side} for g in groups.values()}
    man = {"preset": {"name": "msl_sol_0160_test", "note": "synthetic"}, "selection": {}, "sites": {}, "masks": {}, "images": images,
           "mask_mode": "cnn", "mask_model": "x", "cameras": cams}
    return rec, man, true, rel


def test_msl_block_stages_rigs_scale_verdict_and_promotion(tmp_path, monkeypatch):
    """0.99.16 (Christian, 17 Nov 2026: "hold the principal point on the first stage. Solve A pair rig and B pair rig
    separately. If the camera models look good, then promote the MSL navcams the same way we do for Mars 2020"): the
    principal points stay where they started, each side gets its own right-from-left pose (the two differ, and are
    found), the model's scale is the label baseline's, the verdict blocks a block that is too small, and a promotion
    goes through rpp.calib.models (CHANGES.md) without touching the Mars 2020 fingerprint."""
    from rpp_studies.msl import block as B
    from rpp.calib import models as M
    rec, man, true, rel = _synthetic_stereo_block()
    work = tmp_path / "msl_sol_0160_test"
    work.mkdir()
    rep = B.finish(rec, man, work, 1, [], max_error_m=0.5)
    assert rep["ok"] and list(rep["stages"]) == ["free", "rigs", "pp_check", "thermal"]
    for g, c in rep["cameras"].items():
        assert c["refined"][2:4] == [507.0, 503.0] and c["refined"][9:] == [0.0, 0.0, 0.0]         # held
        assert abs(c["refined"][0] - true[0]) < 1.5 and abs(c["refined"][4] - true[4]) < 0.003
    assert set(rep["rigs"]) == {"NA", "NB"} and rep["rigs"]["NA"]["pairs"] == 8 and rep["rigs"]["NA"]["singles"] == 0
    for side in "AB":
        v = rep["rigs"]["N" + side]
        yaw = np.arcsin(np.asarray(v["R_sensor_from_ref"])[0, 2])                       # rotation about y
        assert abs(yaw - rel[side][0]) < 3e-4
        assert abs(v["baseline_m"] - v["label_baseline_m"]) < 0.004 and abs(v["baseline_ratio"] - 1) < 0.01
    assert abs(rep["scale"]["factor"] - 1) < 0.06 and rep["scale"]["family"] == "N"
    assert rep["residuals"]["NLA"]["radial_spread"] is not None and rep["residuals"]["NLA"]["median_px"] < 0.5
    txt = (work / "report.txt").read_text()
    assert "rig NA (NRA from NLA): 8 pairs" in txt and "Navcam verdict: NOT promoted" in txt and (work / "open_in_colmap.bat").is_file()
    bad = {c["what"] for c in rep["verdict"]["checks"] if not c["ok"]}
    assert not rep["verdict"]["ok"] and "NA rig pairs" in bad and "NLA registered frames" in bad
    files = sorted(f.name for f in (work / "cmods_candidate").iterdir())
    assert files == ["MSL_NA_rig.json", "MSL_NB_rig.json", "MSL_NLA.json", "MSL_NLB.json", "MSL_NRA.json", "MSL_NRB.json"]
    d = json.loads((work / "cmods_candidate" / "MSL_NLA.json").read_text())
    assert d["held_params"] == ["cx", "cy", "k4", "k5", "k6"] and d["form"] == B.MODEL_FORM and "Brown" in d["distortion"]
    r = json.loads((work / "cmods_candidate" / "MSL_NA_rig.json").read_text())
    assert abs(np.linalg.norm(r["t_sensor_from_ref_m"]) - r["baseline_m"]) < 1e-9 and (r["ref"], r["sensor"]) == ("NLA", "NRA")
    cm = tmp_path / "cmods"
    monkeypatch.setenv("RPP_CMODS", str(cm))
    M.init_folder(cm)
    fp = M.models_fingerprint(cm)
    res = B.promote(work)
    assert not res["promoted"] and "NA rig pairs" in res["reason"] and not (cm / "MSL_NLA.json").exists()
    res = B.promote(work, force=True)
    assert res["promoted"] and sorted(res["files"]) == files and (cm / "MSL_NB_rig.json").is_file()
    assert "FORCED past the verdict" in (cm / "CHANGES.md").read_text() and M.models_fingerprint(cm) == fp
    cam = B.promoted_camera("NLA", dict(man["cameras"]["NLA"]))                             # the next block starts there
    assert cam["params"] == d["params"] and cam["label_params"] == man["cameras"]["NLA"]["params"] and "promoted MSL_NLA.json" in cam["source"]
    with pytest.raises(ValueError):
        M.check_msl_file("MSL_NRA.json", d)                                                # the eye does not match the name


def test_msl_front_hazcams_join_the_block_and_masks_go_into_alpha():
    """0.99.16 (Christian, 17 Nov 2026: "please include front hazcam images in the msl block" and "use the current mask
    cnn model and save them as the alpha channel. no need to download the official hardware masks"): the preset takes
    N and F, the Hazcams come from their own volume into hazcam\\SOLnnnnn and keep the fisheye model, and the fetch
    leaves the MXY products out when the mask is the CNN's."""
    from rpp_studies.msl import block as B
    assert B.PRESETS["yellowknife"]["families"] == "NF" and B.CAMERA_HEIGHT_M["F"] < 1.0 < B.CAMERA_HEIGHT_M["N"]
    assert [v[1] for v in F.VOLUMES] == ["navcam", "hazcam"] and all("MSLHAZ_1XXX" in b for b in F.HAZ_BASES)
    names = ["FLB_417301006RAD_F0060000FHAZ00302M1.IMG", "FLB_417301006MXY_F0060000FHAZ00302M1.IMG",
             "NLB_417301006RAD_F0060000NCAM05794M1.IMG", "RLB_417301006RAD_F0060000RHAZ00311M1.IMG"]
    assert F.wanted(names, None, "AB", None, "F", mxy_too=False) == ["FLB_417301006RAD_F0060000FHAZ00302M1"]
    assert len(F.wanted(names, None, "AB", None, "F")) == 2 and F.parse_name(names[0])["group"] == "FLB"
    assert B.rig_sides({"cameras": {"NLA": 1, "NRA": 1, "NLB": 1, "FLB": 1, "FRB": 1}}) == [("F", "B"), ("N", "A")]
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert 'mask_mode: str = "cnn"' in src and "ALPHA_MASKED, 255" in src and 'mxy=a.mask != "cnn"' in src
    assert B.group_camera.__doc__ and 'infos[0].get("family", "N") != "N"' in src


def test_pds_fetch_into_the_archive_layout(tmp_path):
    """0.99.5: rpp.pds - the PDS folder of a product, the local layout, a listing, the example set from the PDS."""
    from rpp import pds
    from rpp.example import EXAMPLE_FILES, fetch_example
    n = "NLF_0461_0707870987_044RAD_N0260630NCAM03461_0A1195J01.IMG"
    assert pds.camera_of(n) == "ncam" and pds.sol_of(n) == 461 and pds.camera_of("ZL0_0464_0708130229_159RAD_x") == "zcam"
    assert pds.folder_url("ncam", 461) == ("https://planetarydata.jpl.nasa.gov/img/data/mars2020/"
                                           "mars2020_navcam_ops_calibrated/data/sol/00461/ids/rdr/ncam/")
    assert pds.local_folder(tmp_path, "fcam", 464) == tmp_path / "datadrive" / "00464" / "ids" / "rdr" / "fcam"
    assert pds.local_folder(tmp_path, "zcam", 464) == tmp_path / "zcam"
    asked = []

    def get(url):
        asked.append(url)
        if url.endswith("/ncam/"):
            return (f'<a href="../">up</a><a href="{n}">x</a><a href="{n[:-4]}.xml">l</a>'
                    '<a href="NRF_0461_0707870987_044EDR_N0260630NCAM03461_0A1195J01.IMG">e</a>').encode()
        return b"data:" + url.rsplit("/", 1)[-1].encode()
    rep = pds.fetch_sols(tmp_path, [461], ["ncam"], get=get)
    assert rep["names"] == [n] and rep["downloaded"] == 2 and (pds.local_folder(tmp_path, "ncam", 461) / n).is_file()
    rep = pds.fetch_sols(tmp_path, [461], ["ncam"], get=get)
    assert rep["kept"] == 2 and rep["downloaded"] == 0                       # nothing twice
    d = fetch_example(tmp_path / "ex", get=get)
    files = sorted(p.name for p in (tmp_path / "ex").rglob("*.IMG"))
    assert len(files) == len(EXAMPLE_FILES) and (d / "zcam" / EXAMPLE_FILES[-1]).is_file()


def test_example_falls_back_to_the_zip_at_the_first_pds_failure(tmp_path, monkeypatch):
    """0.99.5: with the PDS out of reach, the example asks it once (two tries) and takes the release zip at once."""
    import zipfile
    from rpp import example as E, pds
    monkeypatch.setattr(pds.time, "sleep", lambda s: None)
    asked = []

    def get(url):
        asked.append(url)
        raise OSError("unreachable")
    zp = tmp_path / "z" / "example.zip"
    zp.parent.mkdir()
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("datadrive/00461/ids/rdr/ncam/x.IMG", b"x")
    d = E.fetch_example(tmp_path / "ex", url=zp.as_uri(), get=get)
    assert len(asked) == 2 and (d / "datadrive" / "00461" / "ids" / "rdr" / "ncam" / "x.IMG").is_file()


def test_msl_blocks_sit_with_the_mars2020_blocks(monkeypatch, capsys):
    """0.99.12 (Christian, 17 Nov 2026: "put the msl blocks in the rover_blocks folder. Use the naming convention
    D:/rover_blocks/msl_sol_0{sol start}_{block name}"): the folder from the first sol and the block's name; the Mars
    2020 folder parser does not take it for one of its blocks."""
    from rpp_studies.msl import block as B
    from rpp.waypoints.sites import parse_work_folder
    assert B.block_folder_name(124, "yellowknife_bay") == "msl_sol_0124_yellowknife_bay"
    assert B.block_folder_name(3047, "Mont Mercou") == "msl_sol_3047_mont_mercou"
    monkeypatch.setenv("RPP_SCAPES", "/blocks")
    assert B.main(["where", "--sols", "162", "296"]) == 0
    assert capsys.readouterr().out.strip().replace("\\", "/") == "/blocks/msl_sol_0162_yellowknife_bay"
    assert B.main(["where", "--preset", "site87"]) == 0
    assert capsys.readouterr().out.strip().replace("\\", "/") == "/blocks/msl_sol_3047_mont_mercou"
    assert parse_work_folder("/blocks/msl_sol_0162_yellowknife_bay") == (None, False)


def test_msl_blocks_get_open_in_colmap(tmp_path, monkeypatch, capsys):
    """0.99.15 (Christian, 17 Nov 2026: "Make sure there is always an open_in_colmap present for these alignments"): the
    batch file opens the block-frame model (else the site frame, else the mapper's), path-free (%~dp0), CRLF; the gui
    step writes it into a block made before."""
    from rpp_studies.msl import block as B, first_block as F
    (tmp_path / "masks").mkdir()
    bat = F.write_open_in_colmap(tmp_path)
    raw = bat.read_bytes()
    assert b"\r\n" in raw and raw.count(b"\n") == raw.count(b"\r\n")
    t = raw.decode()
    assert t.index("sparse\\block_frame") < t.index("sparse\\site_frame") < t.index("sparse\\0\\")
    assert '--database_path "%~dp0database.db" --image_path "%~dp0images" --import_path "%MODEL%"' in t
    assert "mask_path=" in (tmp_path / "colmap_gui.ini").read_text()
    bat.unlink()
    monkeypatch.setenv("RPP_SCAPES", str(tmp_path))
    assert B.main(["gui", "--sols", "162", "296"]) == 1                    # no such block: nothing made
    (tmp_path / "msl_sol_0162_yellowknife_bay").mkdir()
    assert B.main(["gui", "--sols", "162", "296"]) == 0
    assert (tmp_path / "msl_sol_0162_yellowknife_bay" / "open_in_colmap.bat").is_file()
    src = (Path(B.__file__)).read_text(encoding="utf-8")
    assert src.count("    F.write_open_in_colmap(work)  ") == 2               # at prepare and at align


def test_msl_temperature_stage_and_significance(tmp_path):
    """0.99.17 (Christian, 17 Nov 2026: "model the focal length and rig rotation as a function of temperature. could the
    ncam p1, p2, or k3 be held or set to zero? do a significance study after the next alignment"): a block whose focal
    lengths and B-side rig yaw change with temperature gives the slopes back; the significance step finds the true
    tangential terms needed where they are large and k3 = 0 usable where it is zero."""
    import pycolmap
    from rpp_studies.msl import block as B
    rec, man, true, rel = _synthetic_stereo_block(pairs_per_side=24)
    T = {}
    for r in man["images"]:                                     # temperatures -55 ... -5, the same for a pair
        k = int(r["pair_key"].split("_")[1])
        r["ccd_degC"] = -55.0 + 50.0 * k / 23
        T[r["image"]] = r["ccd_degC"]
    bins, Tm = B.temperature_bins(man, list(T), min_frames=4)
    assert len(set(bins.values())) >= 4 and all(bins[a["image"]] == bins[b["image"]] for a in man["images"] for b in man["images"]
                                                if a["pair_key"] == b["pair_key"])
    ppm = 40.0
    for im in rec.images.values():                              # the image scaled about the principal point: f(T)
        k = 1 + 1e-6 * ppm * (T[im.name] - B.T0_DEGC)
        for p2 in im.points2D:
            p2.xy = np.array([507., 503.]) + k * (np.asarray(p2.xy) - [507., 503.])
    work = tmp_path / "msl_sol_0160_test"
    work.mkdir()
    rep = B.finish(rec, man, work, 1, [], max_error_m=0.5)
    th = rep["thermal"]
    for g, c in th["cameras"].items():
        assert 5 < c["f_ppm_per_degC"] < 75 and len(c["bins"]) >= 4, (g, c["f_ppm_per_degC"])    # the sign and size (a small block)
    assert set(th["rigs"]) == {"NA", "NB"} and all(abs(x) < 3 for x in th["rigs"]["NA"]["rate_mdeg_per_degC"])
    d = json.loads((work / "cmods_candidate" / "MSL_NLA.json").read_text())
    assert d["thermal"]["T0_degC"] == B.T0_DEGC and d["thermal"]["f_ppm_per_degC"] > 5
    assert json.loads((work / "cmods_candidate" / "MSL_NB_rig.json").read_text())["thermal"]["yaw_mdeg_per_degC"] is not None
    assert (work / "sparse" / "free" / "images.bin").is_file() and (work / "sparse" / "thermal" / "images.bin").is_file()
    assert "temperature (bins of 10 degC" in (work / "report.txt").read_text()
    (work / "manifest.json").write_text(json.dumps(man, default=float))
    sig = B.significance(work, folds=3)
    assert set(sig["variants"]) == {v[0] for v in B.SIG_VARIANTS} and set(sig["terms"]) == {"NLA", "NRA", "NLB", "NRB"}
    assert not sig["variants"]["k3 = 0"]["ok"] or sig["variants"]["k3 = 0"]["by_camera"]["NLA"]["image_max_px"] <= 0.15
    assert sig["terms"]["NLA"]["p1"]["sd"] > 0 and "USABLE" in (work / "significance.txt").read_text().upper()
    a = B.camera_difference(true, true, 1024, 1024)
    b = B.camera_difference(true, list(true[:6]) + [0, 0] + list(true[8:]), 1024, 1024)
    assert a["max"] < 1e-6 and b["raw_max"] > b["max"] > 0.01           # the tangential terms of the test camera: visible
