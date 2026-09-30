"""MPPP v0p43: stable site definitions, WORK-folder helpers, the headless runner and match reuse."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_sites_json_is_the_site_list():
    from mppp.sfm import sites as S
    t = S.load_site_table()
    assert len(t["sites"]) >= 32 and S.SITES["rockytop"] == (461, 530) and S.SITES["south_arm"] == (1408, 1412)
    for g, members in t["groups"].items():
        assert members and all(m in t["sites"] for m in members), g
    for k, v in t["sites"].items():
        assert v["sols"][0] <= v["sols"][1] and isinstance(v.get("settings", {}), dict), k
    assert S.site_group("nav_zcam34")[:2] == ["rockytop", "threeforks"]
    assert S.parse_work_folder("D:/x/south_arm_colmap_nav_zcam34") == ("south_arm", True)
    assert S.parse_work_folder("D:/x/sid_colmap") == ("sid", False)
    assert S.work_folder("D:/r", "sid", True).name == "sid_colmap_nav_zcam34"


def test_sites_file_override(tmp_path):
    from mppp.sfm.sites import load_sites
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"sites": {"a": {"sols": [5, 9]}, "b": [1, 2]}}))
    assert load_sites(f) == {"a": (5, 9), "b": (1, 2)}
    f.write_text(json.dumps({"sites": {"a": {"sol": [5, 9]}}}))
    with pytest.raises(ValueError):
        load_sites(f)


def _meta(stem, sol, site, drive, seq):
    return {"source_product": stem + ".IMG", "outputs": {"PNG8": f"images_png8/{stem}.png"},
            "filename": {"stem": stem, "sol": sol, "site": site, "drive": drive, "sequence": seq}}


def test_exclusions_and_existing(tmp_path):
    from mppp.sfm.workdir import apply_exclusions, keep_existing, read_exclusions
    ms = [_meta("NLF_0658_0725353794_270RAD_N0320274NCAM08111_0A0095J01", 658, 32, 274, "NCAM08111"),
          _meta("NRF_0658_0725353794_270RAD_N0320274NCAM08111_0A0095J01", 658, 32, 274, "NCAM08111"),
          _meta("ZR0_0690_0728197573_035RAD_N0321184ZCAM08692_0340LMA01", 690, 32, 1184, "ZCAM08692"),
          _meta("NLF_0700_0729000000_000RAD_N0330000NCAM00200_0A0195J01", 700, 33, 0, "NCAM00200")]
    f = tmp_path / "exclude_images.txt"
    f.write_text("# comment\nS032D1184   # a station\n\nseq:ncam08111\nsol:999\n")
    kept, removed = apply_exclusions(ms, read_exclusions(f))
    assert [m["filename"]["sol"] for m in kept] == [700]
    assert {"image": None, "entry": "sol:999"} in removed
    assert sum(r["image"] is not None for r in removed) == 3
    kept, _ = apply_exclusions(ms, ["sol:650-660", "ZR0_0690_*"])
    assert [m["filename"]["sol"] for m in kept] == [700]
    kept, _ = apply_exclusions(ms, ["NLF_0700_0729000000_000RAD_N0330000NCAM00200_0A0195J01.IMG"])
    assert len(kept) == 3
    (tmp_path / "images_png8").mkdir()
    (tmp_path / "images_png8" / (ms[0]["filename"]["stem"] + ".png")).write_bytes(b"x")
    kept, missing = keep_existing(ms, tmp_path)
    assert len(kept) == 1 and len(missing) == 3


def test_project_dir_seed_and_settings(tmp_path):
    from mppp.sfm.workdir import load_settings, project_dir, seed_variant
    assert project_dir(tmp_path).name == "colmap" and project_dir(tmp_path, "rational").name == "colmap_rational"
    with pytest.raises(ValueError):
        project_dir(tmp_path, "a b")
    base = tmp_path / "colmap"
    base.mkdir()
    for n in ("features.db", "features.json", "database.db", "database_matches.json"):
        (base / n).write_text(n)
    out = seed_variant(project_dir(tmp_path, "v"), base)
    assert out["copied"] == ["features.db", "features.json"] and out["reuse_matches_from"] == [str(base / "database.db")]
    assert (tmp_path / "colmap_v" / "features.json").read_text() == "features.json"
    s = tmp_path / "mppp_settings.json"
    s.write_text(json.dumps({"_comment": "x", "ATTITUDE_PRIOR_DEG": 1.0}))
    assert load_settings(s) == {"ATTITUDE_PRIOR_DEG": 1.0}


def test_runner_settings_and_keys(tmp_path):
    from mppp import runner as R
    v = R.parse_set(["A=3", "B=dict(x=1)", "C=rational", "D=[1, 2]", "E=((24,10,8),(8,2,2))", r"F=D:\scapes\x", "G=True"])
    assert v == {"A": 3, "B": "py:dict(x=1)", "C": "rational", "D": [1, 2], "E": [(24, 10, 8), (8, 2, 2)],
                 "F": "D:\\scapes\\x", "G": True}
    assert R.python_literal("NAVCAM_CAMERAS", "D:/x") == "Path(r'D:/x')"
    assert R.python_literal("MATCH", "py:dict(max_ratio=0.8)") == "dict(max_ratio=0.8)"
    assert R.python_literal("ATTITUDE_PRIOR_DEG", 1.0) == "1.0"
    k = R.run_key({"A": 1}, "processed", "")
    assert k == R.run_key({"A": 1}, "processed", "") and k != R.run_key({"A": 2}, "processed", "")
    assert k != R.run_key({"A": 1}, "pds", "") and k != R.run_key({"A": 1}, "processed", "v")
    work = tmp_path / "sid_colmap"
    work.mkdir()
    (work / "mppp_settings.json").write_text(json.dumps({"X": 1, "Y": 1}))
    f = tmp_path / "more.json"
    f.write_text(json.dumps({"Y": 2, "Z": 2}))
    assert R.work_settings(work, "sid", f, {"Z": 3}) == {"X": 1, "Y": 2, "Z": 3}


def test_status_heartbeat(tmp_path):
    from mppp import runner as R
    st = R.RunStatus(tmp_path / "s.json", work="w")
    st.update(state="running", cell=4)
    d = R.read_status(tmp_path / "s.json")
    assert d["alive"] and d["cell"] == 4
    d = json.loads((tmp_path / "s.json").read_text())
    d["heartbeat"] = "2020-01-01T00:00:00"
    (tmp_path / "s.json").write_text(json.dumps(d))
    d = R.read_status(tmp_path / "s.json")
    assert not d["alive"] and d["state"] == "stopped (no heartbeat)"
    st.stop("finished")
    assert R.read_status(tmp_path / "s.json")["state"] == "finished"
    assert R.read_status(tmp_path / "missing.json") is None


def test_notebook_choice_ignores_copies(tmp_path):
    from mppp.runner import notebook
    for n in ("03_colmap_alignment_v0p40.ipynb", "03_colmap_alignment_v0p43.ipynb",
              "03_colmap_alignment_v0p40 - Copy (2).ipynb", "03_colmap_alignment_v0p9.ipynb"):
        (tmp_path / n).write_text("{}")
    assert notebook("03_colmap_alignment", tmp_path).name == "03_colmap_alignment_v0p43.ipynb"
    (tmp_path / "03_colmap_alignment.ipynb").write_text("{}")
    assert notebook("03_colmap_alignment", tmp_path).name == "03_colmap_alignment.ipynb"


def test_inject_and_notebook03_parameters():
    import nbformat
    from mppp.runner import inject
    nb = nbformat.read(str(ROOT / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    src = "".join(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    for name in ("SOURCE", "WORK_DIR", "VARIANT", "EXCLUDE", "REUSE_MATCHES", "RUN_KEY", "SITES_FILE", "SITES_EXTRA"):
        assert f"\n{name}" in "\n" + src, name
    assert "SITES = {" not in src                       # the site list lives in mppp/data/sites.json
    inject(nb, {"SOURCE": "'processed'"})
    i = next(i for i, c in enumerate(nb.cells) if "parameters" in c.metadata.get("tags", []))
    assert nb.cells[i + 1].metadata["tags"] == ["injected-parameters"] and "SOURCE = 'processed'" in nb.cells[i + 1].source
    import ast
    for c in nb.cells:
        if c.cell_type == "code":
            ast.parse(c.source)


class _Stub:
    def __init__(self, root, db, order, params):
        self.features_db = root / "features.db"
        self.database = db
        self.images = [{"name": n, "instrument": "C"} for n in order]
        self.cameras = {"C": {"model": "SIMPLE_RADIAL", "width": 400, "height": 300, "params": params}}


def test_match_reuse(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    from PIL import Image
    from scipy.ndimage import affine_transform, gaussian_filter
    from mppp.sfm import database as D
    (tmp_path / "images").mkdir()
    rng = np.random.default_rng(3)
    base = gaussian_filter(rng.random((500, 650)), 1.5)
    base = (255 * (base - base.min()) / np.ptp(base)).astype(np.uint8)
    names = []
    for k in range(4):
        im = affine_transform(base.astype(float), np.eye(2), offset=(15 * k, 20 * k), output_shape=(300, 400), order=1)
        names.append(f"i{k}.png")
        Image.fromarray(im.astype(np.uint8)).save(tmp_path / "images" / names[-1])
    fdb = tmp_path / "features.db"
    pycolmap.extract_features(str(fdb), str(tmp_path / "images"), camera_mode=pycolmap.CameraMode.PER_IMAGE)
    (tmp_path / "features.json").write_text('{"k": 1}')

    def build(db, order, params):
        f = pycolmap.Database.open(str(fdb))
        byn = {im.name: im.image_id for im in f.read_all_images()}
        d = pycolmap.Database.open(str(db))
        cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=400, height=300, params=params)
        cam.has_prior_focal_length = True
        cid = d.write_camera(cam)
        sensor = pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=cid)
        rig = pycolmap.Rig()
        rig.add_ref_sensor(sensor)
        rid = d.write_rig(rig)
        for n in order:
            iid = d.write_image(pycolmap.Image(name=n, camera_id=cid))
            d.write_keypoints(iid, f.read_keypoints(byn[n]))
            d.write_descriptors(iid, f.read_descriptors(byn[n]))
            fr = pycolmap.Frame()
            fr.rig_id = rid
            fr.add_data_id(pycolmap.data_t(sensor_id=sensor, id=iid))
            d.write_frame(fr)
        d.close()
        f.close()
        return _Stub(tmp_path, db, order, params)

    cam = [350.0, 200.0, 150.0, 0.0]
    p1 = build(tmp_path / "a.db", names, cam)
    pycolmap.match_exhaustive(str(p1.database))
    settings = {"mode": "exhaustive", "max_ratio": 0.8}
    D.write_matches_record(p1, settings)
    rec = json.loads(D.matches_record_path(p1.database).read_text())
    assert rec["features_key"] and set(rec["cameras"]) == set(names)

    def pairs(db):
        import sqlite3
        con = sqlite3.connect(str(db))
        nm = {i: n for i, n in con.execute("SELECT image_id, name FROM images")}
        out = {}
        for pid, rows, cols, data in con.execute("SELECT pair_id, rows, cols, data FROM matches"):
            a, b = D._pair(pid)
            arr = np.frombuffer(data, np.uint32).reshape(rows, cols) if rows else np.zeros((0, 2), np.uint32)
            if nm[a] > nm[b]:
                arr, key = arr[:, ::-1], (nm[b], nm[a])
            else:
                key = (nm[a], nm[b])
            out[key] = sorted(map(tuple, arr.tolist()))
        con.close()
        return out

    ref = pairs(p1.database)
    # one image left out, order reversed: raw matches carried over (columns swapped), geometries verified again
    order = [n for n in names[::-1] if n != "i1.png"]
    p2 = build(tmp_path / "b.db", order, cam)
    r = D.reuse_matches(p2, [p1.database], matching=settings)
    assert r["matches"] == 3 and r["geometries"] == 0
    assert all(pairs(p2.database)[k] == ref[k] for k in pairs(p2.database))
    # same order and camera: geometries kept too
    p3 = build(tmp_path / "c.db", [n for n in names if n != "i1.png"], cam)
    r = D.reuse_matches(p3, [p1.database], matching=settings)
    assert r["matches"] == 3 and r["geometries"] == 3
    # another camera: raw matches only; other matching settings or features: nothing
    p4 = build(tmp_path / "d.db", names, [360.0, 200.0, 150.0, 0.0])
    assert D.reuse_matches(p4, [p1.database], matching=settings)["geometries"] == 0
    p5 = build(tmp_path / "e.db", names, cam)
    assert D.reuse_matches(p5, [p1.database], matching={"mode": "exhaustive", "max_ratio": 0.9})["matches"] == 0
    (tmp_path / "features.json").write_text('{"k": 2}')
    p6 = build(tmp_path / "f.db", names, cam)
    assert D.reuse_matches(p6, [p1.database], matching=settings)["sources"][0]["skipped"] == "other features"


def test_match_settings_defaults():
    from mppp.sfm.matching import match_settings
    s = match_settings("exhaustive", max_ratio=0.9, max_distance=1.0, cross_check=True, guided_matching=False,
                       max_error_px=6.0)
    assert s["min_num_inliers"] == 15 and s["max_num_matches"] == 32768 and s["max_error_px"] == 6.0
    assert match_settings("exhaustive", max_ratio=0.9) != match_settings("exhaustive", max_ratio=0.8)


def test_windows_bat_files():
    d = ROOT / "scripts" / "windows"
    for n in ("align_here.bat", "run_all_sites.bat", "sites_status.bat", "mppp_env.bat", "_run_align.bat",
              "_run_sites.bat"):
        raw = (d / n).read_bytes()
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), n     # CRLF line ends for cmd.exe
        assert raw.decode("ascii")
    assert b"align_scape.py" in (d / "_run_align.bat").read_bytes()
    assert b"run_sites.py" in (d / "_run_sites.bat").read_bytes()
