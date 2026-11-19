import json

import cv2
import numpy as np

from test_ibr import _ring_station


def _grid(n=120, half=12.0):
    x = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, x)
    V = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    i = np.arange(n - 1)
    a = (i[:, None] * n + i[None, :]).ravel()
    return V, np.r_[np.c_[a, a + 1, a + n], np.c_[a + 1, a + n + 1, a + n]]


def _box(c, s):
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * np.asarray(s, float) / 2 + np.asarray(c, float)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return v, np.array(f)


def _scene(tmp_path):
    """The ring station's frames over a grid terrain with a rock 4 m north and a 'rover' box under the head."""
    from rpp.ibr import raycast as RC
    proj, rec, st, _ = _ring_station(tmp_path)
    P = np.asarray(st.pivot)
    Vg, Fg = _grid()
    Vr, Fr = _box(P + [0, 4, -P[2] + 0.4], [1.0, 1.0, 0.8])
    Vv, Fv = _box(P + [0.6, 0, -0.9], [0.8, 0.6, 0.3])
    V = np.vstack([Vg, Vr, Vv])
    F = np.vstack([Fg, Fr + len(Vg), Fv + len(Vg) + len(Vr)])
    sc = RC.Scene((V, F), plane={"normal": [0, 0, 1], "offset": 0.0, "centre": P, "up": [0, 0, 1], "radius_m": 100.0}, sky=True)
    sc.rover_triangles = (len(Fg) + len(Fr), len(F))
    return proj, rec, st, sc


def test_terrain_texture_is_seen_from_above_the_pivot_and_hidden_faces_get_cells(tmp_path):
    """0.99.26 (Christian, 18 Nov 2026: "a simplified medium poly mesh of the 10m radius area to be textured ... projected
    onto the terrain largely from the perspective of the viewer so that the near terrain gets higher res"; "the texture
    projection center might be better a little above the pivot point without the rover"): the disc's texels are equal
    angles from 0.5 m above the pivot, the triangles behind the rock get cells outside the disc, every texel has a
    color, and a vertex's texture coordinates are its direction."""
    from rpp.ibr import ods as O, station_mesh as SM
    proj, rec, st, sc = _scene(tmp_path)
    src = O.station_sources(proj, rec, st, scale=1.0)
    tp = SM.terrain_part(sc, src, st, size=1024, target=8000)
    r = tp["record"]
    assert r["triangles"] <= 8000 and r["center_above_pivot_m"] == 0.5 and 70 < r["theta_max_deg"] < 85
    assert r["triangles_in_cells"] > 10 and r["hidden_from_center"] > 10 and r["disc_texels_colored"] > 0.6
    uv = tp["uv"]
    assert uv.min() >= 0 and uv.max() <= 1 and len(tp["V"]) == len(uv) and tp["F"].max() == len(uv) - 1
    C = np.asarray(st.pivot) + [0, 0, 0.5]
    north = tp["V"][:, 1] - C[1] > 3                                  # the terrain north of the pivot is in the top half
    disc = np.hypot(uv[:, 0] - 0.5, uv[:, 1] - 0.5) < 0.5
    assert (uv[north & disc, 1] < 0.5).all()
    tex = tp["texture"]
    assert tex.shape == (1024, 1024, 3) and np.isfinite(tex).all()
    # equal angle per texel: a texel near the pivot covers less ground than one 9 m out
    d = np.array([[0.0, 1.0, -C[2]], [0.0, 9.0, -C[2]]]) + C
    uvd = SM.disc_uv(d, C, 1024, 1024 / 2 - SM.GUTTER_PX - 2, np.radians(r["theta_max_deg"]), st.up, st.north, st.east)
    d2 = d + [0, 0.1, 0]
    uvd2 = SM.disc_uv(d2, C, 1024, 1024 / 2 - SM.GUTTER_PX - 2, np.radians(r["theta_max_deg"]), st.up, st.north, st.east)
    px_per_10cm = np.abs(uvd2[:, 1] - uvd[:, 1]) * 1024
    assert px_per_10cm[0] > 5 * px_per_10cm[1]


def test_rover_textured_where_seen_within_1m_and_the_glb(tmp_path):
    """The rover's triangles seen from within 1 m of the pivot get cells, the rest the gray strip; the two parts make one
    binary glTF the viewer reads (positions, texture coordinates, an unlit textured material each)."""
    from rpp.ibr import ods as O, station_mesh as SM, rover_surface as RS
    proj, rec, st, sc = _scene(tmp_path)
    src = O.station_sources(proj, rec, st, scale=1.0)
    rp = SM.rover_part(sc, src, st, size=512)
    r = rp["record"]
    assert r["triangles"] == 12 and 0 < r["textured_triangles"] < 12 and r["viewpoints"] >= 60 and r["families"] == "N"
    for s in src[:2]:                                   # 0.99.27: a Mastcam-Z run - its frames first, the Navcams fill
        s.name = "Z" + s.name[1:]
    src[2].fill_only = True
    assert SM.rover_families(src) == "NZ"
    to8 = lambda a: np.clip(np.rint(a), 0, 255).astype(np.uint8)
    tp = SM.terrain_part(sc, src, st, size=512, target=5000)
    (tmp_path / "st" / "with_rover").mkdir(parents=True)
    SM.write_part(tmp_path / "st", "S1", "terrain", tp, st, to8)
    SM.write_part(tmp_path / "st" / "with_rover", "S1", "rover", rp, st, to8)
    out = SM.assemble(tmp_path / "st")
    assert out["file"] == "S1_station_mesh.glb" and set(out["parts"]) == {"terrain", "rover"}
    doc, binary = RS.read_glb((tmp_path / "st" / out["file"]).read_bytes())
    assert [m["name"] for m in doc["meshes"]] == ["terrain", "rover"] and len(doc["images"]) == 2
    for m in doc["meshes"]:
        prim = m["primitives"][0]
        assert "TEXCOORD_0" in prim["attributes"]
        mat = doc["materials"][prim["material"]]
        assert "KHR_materials_unlit" in mat["extensions"] and doc["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"] in (0, 1)
    assert "projection" not in doc["asset"]["extras"] and doc["asset"]["extras"]["station_mesh"]["view_radius_m"] == 1.0
    v = doc["bufferViews"][doc["images"][0]["bufferView"]]
    img = cv2.imdecode(np.frombuffer(binary[v["byteOffset"]:v["byteOffset"] + v["byteLength"]], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (512, 512, 3)
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert acc["max"][1] < 0 and -20.5 < acc["min"][0] and acc["max"][0] < 20.5     # y up: the ground below the eyes (20 m: 0.99.39)


def test_texture_on_render_processes_is_the_same(tmp_path):
    from rpp.ibr import ods as O, station_mesh as SM
    from rpp.ibr.parallel import RenderPool
    proj, rec, st, sc = _scene(tmp_path)
    a = SM.terrain_part(sc, O.station_sources(proj, rec, st, scale=1.0), st, size=512, target=5000, rows_per_chunk=64)
    src = O.station_sources(proj, rec, st, scale=1.0)
    with RenderPool(sc, src, 2) as pool:
        b = SM.terrain_part(sc, src, st, pool=pool, size=512, target=5000, rows_per_chunk=64)
        assert pool.failed is None
    assert np.array_equal(a["texture"], b["texture"]) and np.array_equal(a["uv"], b["uv"])


def test_render_station_part_web_entry_and_plan(tmp_path, monkeypatch):
    """``rpp ods --station-mesh-only`` makes the part without a panorama; the published entry gets the fourth mode;
    single-sites asks every ods run for its part."""
    from rpp.ibr import ods as O, web, single_sites as SS
    proj, rec, st, sc = _scene(tmp_path)
    sc.rover_triangles = None
    from rpp.ibr import station_mesh as SM
    monkeypatch.setattr(SM, "TEXTURE_PX", 512)
    monkeypatch.setattr(SM, "TARGET_TRIANGLES", 5000)
    out = tmp_path / "Sol0091_S003D0010"
    out.mkdir()
    r = O.render_station(proj, rec, st, sc, out, width=256, scale=1.0, rows_per_chunk=64, station_mesh="only", workers=1)
    assert r["kind"] == "terrain" and (out / "Sol0091_S003D0010_station_mesh_terrain.jpg").is_file()
    assert not list(out.glob("*_ods_left.png"))
    from rpp.ibr import station_mesh as SM
    SM.assemble(out)
    pano = tmp_path / "panos"
    pano.mkdir()
    sm = web.station_mesh(out, tmp_path, pano)
    assert sm["file"] == "panos/Sol0091_S003D0010_station_mesh.glb" and sm["move_limit_m"] == 1.0 and (pano / "Sol0091_S003D0010_station_mesh.glb").is_file()
    # 0.99.39 (Christian, 19 Nov 2026: "the 3d terrain should go out to 20 m instead of 10 m"): the file's own fade
    assert SM.RADIUS_M == 20.0 and SM.FADE_M == (18.0, 20.0) and SM.SHELL_FROM_M == 17.0 and sm["fade_m"] == [18.0, 20.0]
    import json as _json
    rec_ = _json.loads((out / "Sol0091_S003D0010_station_mesh.json").read_text())
    rec_.pop("fade_m")
    (out / "Sol0091_S003D0010_station_mesh.json").write_text(_json.dumps(rec_))
    assert web.station_mesh(out, tmp_path, pano)["fade_m"] == [8.0, 10.0]      # a file of before 0.99.39: its 10 m mesh
    assert web.TERRAIN_3D == ("terrain3d", "3D terrain and rover")
    src = (SS.__file__)
    page = (__import__("pathlib").Path(src).resolve().parents[3] / "docs" / "ods" / "index.html").read_text(encoding="utf-8")
    assert "showRover(sm, { normals: nmView, distance: dView," in page and "holdNearPivot" in page


def test_single_sites_skips_stations_whose_block_is_not_aligned(tmp_path, capsys):
    """0.99.28 (Christian, 18 Nov 2026: "The next ods should use the post freeze blocks. Skip an ods site if the block
    has not aligned yet")."""
    import argparse
    from rpp.ibr import single_sites as SS
    from rpp.waypoints.sites import load_site_table
    cfg = {"stations": {"A": {"work": "m_a", "station": "3_10"}, "B": {"work": "m_b", "station": "3_11"}}}
    cfgf = tmp_path / "ss.json"
    cfgf.write_text(json.dumps(cfg))
    (tmp_path / "m_a" / "colmap").mkdir(parents=True)
    (tmp_path / "m_a" / "colmap" / "run_done.json").write_text("{}")
    (tmp_path / "m_b" / "colmap").mkdir(parents=True)
    assert SS.block_aligned(tmp_path / "m_a") and not SS.block_aligned(tmp_path / "m_b")
    p = argparse.ArgumentParser()
    SS.add_parser(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["single-sites", "--config", str(cfgf), "--root", str(tmp_path), "--dry-run", "A", "B"])
    assert SS.cmd_single_sites(a) == 0
    out = capsys.readouterr().out
    assert "B skipped - its block" in out and "1 station(s) to run, 1 skipped" in out and "A: " in out
    a = p.parse_args(["single-sites", "--config", str(cfgf), "--root", str(tmp_path), "--dry-run", "--any-block", "A", "B"])
    SS.cmd_single_sites(a)
    assert "skipped - its block" not in capsys.readouterr().out


def test_sol121_seitah_station_in_the_list():
    """0.99.30 (Christian, 18 Nov 2026: "Add the sol 121 location to the ods list. Use the eod images with the one mid
    drive imaging for context")."""
    import json
    from pathlib import Path
    from rpp.ibr import single_sites as SS
    cfg = json.loads((Path(SS.__file__).parents[1] / "data" / "single_sites.json").read_text(encoding="utf-8"))
    st = cfg["stations"]["Sol0116_S004D1250"]
    assert st["station"] == "4_1250" and len(st["frames"]["images"]) == 11 and all("N0041250" in n for n in st["frames"]["images"])
    assert len(st["context"]) == 2 and all("N0041206NCAM00310" in n for n in st["context"])
    run = SS.plan(cfg, "/tmp/root", labels=["Sol0116_S004D1250"])[0]
    ods = [c for c in run["commands"] if c["step"] == "ods"]
    assert ods and all("--context-images" in c["argv"] for c in ods) and run["work"].endswith("seitah_overlook_zcam")


def test_fifth_mode_surface_to_the_horizon_one_texture(tmp_path):
    """0.99.32 (Christian, 18 Nov 2026: "expanding the mesh to the horizon with a low poly mesh ... continuous from below
    the rover to infinity ... the horizon which lies on a circle about 90% of the max size of circle that will fit in the
    square. Beyond that horizon circle, the uv ... goes to the sky"; "The 3d model rover should be a separate mesh and
    texture from the terrain, but both can be bundled in the same glb file"; "Build this as a fifth viewing mode")."""
    from rpp.ibr import ods as O, station_mesh as SM, rover_surface as RS, web
    proj, rec, st, sc = _scene(tmp_path)
    src = O.station_sources(proj, rec, st, scale=1.0)
    C = np.asarray(st.pivot) + [0, 0, 0.5]
    sh = SM.horizon_shell(sc, C, st.pivot, st.up, st.north, st.east, az_step_deg=2.0, step_deg=1.0)
    r = np.linalg.norm(sh["V"] - C, axis=1)
    assert np.isclose(r.max(), SM.DOME_M) and (r > 0.99 * SM.DOME_M).sum() > 100           # the sky, at the dome
    hd = np.hypot(*(sh["V"][:, :2] - np.asarray(st.pivot)[:2]).T)
    assert hd[sh["F"]].max(axis=1).min() >= SM.SHELL_FROM_M - 1e-3                           # nothing of it under the rover
    assert 94.9 < np.degrees(sh["knee"]) < 97.0                                              # flat ground: 5 deg above the horizontal at least (0.99.41; 10 before)
    first, last = sh["V"][sh["F"][:10]].reshape(-1, 3), sh["V"][sh["F"][-10:]].reshape(-1, 3)
    assert np.linalg.norm(first - C, axis=1).min() > 0.99 * SM.DOME_M and np.linalg.norm(last - C, axis=1).max() < 20   # the far rings first
    tp = SM.terrain_part(sc, src, st, size=1024, target=8000, full=True)
    rr = tp["record"]
    assert rr["full"] and rr["horizon_frac"] == 0.97 and rr["shell"]["triangles"] == len(tp["shell"]["F"]) and rr["fade_m"] == [18.0, 20.0]
    rpx = 1024 / 2 - SM.GUTTER_PX - 2
    rho = np.hypot(tp["shell"]["uv"][:, 0] - 0.5, tp["shell"]["uv"][:, 1] - 0.5) * 1024
    assert rho.max() <= rpx + 1e-3 and (rho > 0.97 * rpx + 1).any()                          # the sky beyond the 97 % circle
    ground = np.linalg.norm(tp["shell"]["V"] - C, axis=1) < 0.5 * SM.DOME_M
    assert rho[ground].max() <= 0.97 * rpx + 1e-3                                            # the terrain inside it
    rho_in = np.hypot(tp["uv"][:, 0] - 0.5, tp["uv"][:, 1] - 0.5) * 1024
    assert np.isfinite(tp["texture"]).all() and tp["texture"].shape == (1024, 1024, 3)
    # 0.99.41: the sky and the unused texels white; the ground not
    w = tp["white"]
    assert w.shape == (1024, 1024) and w[512, 1023] and not w[512, 512] and rr["white_texels"] == int(w.sum()) > 0
    # the map and its inverse agree across the knee
    knee = (np.radians(rr["skyline_deg"] + 90.0), 0.97)
    rows, cols = np.array([100, 512, 900, 30]), np.array([512, 300, 700, 512])
    d, _ = SM.disc_dirs(rows, cols, 1024, rpx, np.pi, st.up, st.north, st.east, knee=knee)
    uv = SM.disc_uv(C + 5 * d, C, 1024, rpx, np.pi, st.up, st.north, st.east, knee=knee)
    assert np.allclose(uv[:, 0] * 1024, cols + 0.5, atol=1e-6) and np.allclose(uv[:, 1] * 1024, rows + 0.5, atol=1e-6)
    to8 = lambda a: np.clip(np.rint(a), 0, 255).astype(np.uint8)
    rp = SM.rover_part(sc, src, st, size=256)
    (tmp_path / "st" / "with_rover").mkdir(parents=True)
    SM.write_part(tmp_path / "st", "S1", "scape", tp, st, to8)
    SM.write_part(tmp_path / "st" / "with_rover", "S1", "rover", rp, st, to8)
    out = SM.assemble_scape(tmp_path / "st")
    assert out["file"] == "S1_station_scape.glb" and SM.assemble(tmp_path / "st") is None
    doc, _ = RS.read_glb((tmp_path / "st" / out["file"]).read_bytes())
    assert [m["name"] for m in doc["meshes"]] == ["terrain", "horizon", "rover"] and len(doc["images"]) == 2
    tex = [doc["materials"][m["primitives"][0]["material"]]["pbrMetallicRoughness"]["baseColorTexture"]["index"] for m in doc["meshes"]]
    assert tex[0] == tex[1] != tex[2]                                 # terrain and horizon on one texture, the rover on its own
    assert doc["asset"]["extras"]["station_scape"]["fade_m"] == [18.0, 20.0]   # 0.99.39 (8-10 m before)
    pano = tmp_path / "panos"
    pano.mkdir()
    sc5 = web.station_scape(tmp_path / "st", tmp_path, pano)
    assert sc5["whole"] and sc5["file"] == "panos/S1_station_scape.glb" and sc5["dome_m"] == SM.DOME_M and web.SCAPE_3D[0] == "scape3d"
    # 0.99.33 (Christian, 18 Nov 2026: "Build a simple skid steering option. Call it a sixth mode. It will be ride along")
    assert sc5["parts"] == ["terrain", "rover"] and web.DRIVE_3D == ("drive3d", "Drive the rover (simulation)")
    dv = web.drive_settings()
    assert dv["track_m"] == 2.3 and dv["speed_m_s"] == 0.25 and dv["limit_m"] == 15.0 and len(dv["wheels_fwd_m"]) == 3
    assert abs(dv["pivot_fwd_right_m"][0] - 0.80) < 0.01 and abs(dv["pivot_fwd_right_m"][1] - 0.56) < 0.01      # the mast: ahead and right of the center
    import pathlib
    page = (pathlib.Path(web.__file__).resolve().parents[3] / "docs" / "ods" / "index.html").read_text(encoding="utf-8")
    for s in ("function driveStep", "(vr - vl) * DRIVE.speed / DRIVE.track", "length() <= 1.0", "rs.drive && rs.whole", "driveKeys"):
        assert s in page, s


def test_viewer_has_the_screen_eye_movement_the_fade_and_every_mesh():
    """0.99.32 (Christian: "Add the off pivot motion, swaying, and the cursor directed motion. Add fading at the end of
    the terrain mesh")."""
    import pathlib
    from rpp.ibr import single_sites as SS
    page = (pathlib.Path(SS.__file__).resolve().parents[3] / "docs" / "ods" / "index.html").read_text(encoding="utf-8")
    # 0.99.38 (Christian: "remove the swaying"): no sway; 0.99.39 (Christian: "the curser should move 5 cm")
    assert "SWAY_M" not in page and "swayGain" not in page
    for s in ("NECK_M = 0.15", "HOVER_M = 0.05", "half = 20.5", "limit: 15.0,", "'dcs color': 'decorrelation stretch'", "'raw color': 'original color'", "smoothstep(fade.x, fade.y, vR)", "NORMAL_FRAG", "skipRover", "O_PITCH_DEG = -30",
              "for (const me of doc.meshes)", "wholeScene", "m.userData.kind === 'rover'"):
        assert s in page, s
    assert "doc.meshes[0].primitives" not in page


def test_viewer_terrain_modes_rover_button_and_hidden_drive():
    """0.99.40 (Christian, 19 Nov 2026: "change the mode banner to be just for the terrain ... the rover is just visible
    or not ... Move drive mode to a steering wheel icon above the rover show/hide button. That is experimental and
    should be hidden ... The full 3d terrain model does not need the ods")."""
    import pathlib
    from rpp.ibr import single_sites as SS
    page = (pathlib.Path(SS.__file__).resolve().parents[3] / "docs" / "ods" / "index.html").read_text(encoding="utf-8")
    for s in ("['stereo', 'terrain stereo'], ['local', 'terrain local 3D'], ['full', 'terrain full 3D']",
              "get('experimental') === '1'", 'id="driveBtn"', "$('driveBtn').onclick = driveToggle;",
              "if (isWhole(vr)) return null;", "if (!whole) {", "rk !== 'no-rover'", "roverOn = false, roverId = 0",
              "if (e.data && e.data.handedness === 'left') roverToggle(); else step('v', 1);",
              "right grip: terrain   |   left grip: rover", 'id="driveBeta"', "if (pressed(src, 3) &&",
              "outlineWidth: 0.14, spacing: 0.12", "az, 24, 40, 10.5", "const on = welcome.visible || infoPanel.visible;",
              "#title { font-weight: 600; margin-right: 6px; color: #fff; }", "#bar button#home, #compass button#faceO { color: #ff9500; }",
              "const MAP_ORANGE = '#ff9500';", "#nav button#roverBtn { color: #ff9500; }", "#nav button#roverBtn.on { color: #fff; border-color: #555; }",
              "q.key === 'scape3d' && q.station_mesh", "noHorizon: true", "skipHorizon: isMain && !!rs.noHorizon",
              "function stereoOrFull()", "const DIST_FRAG", "canSwap(sm, skinUrl) ? swapSkin(skinUrl)", "sk[k] || /^normal/.test(k) || k === 'range log' || k === 'distance map'", "if (id !== swapId || rid !== roverId) { bmp.close(); return; }",
              "B: stereo / full 3D"):
        assert s in page, s
    nav = page[page.index('<div id="nav">'):page.index('<div id="msg">')]
    assert nav.index('id="driveBtn"') < nav.index('id="roverBtn"') and "#nav button#driveBtn { position: absolute;" in page
    assert "lastRoverKey" not in page and "grip: rover on, off, 3D, 3D terrain" not in page
    drive = page[page.index("if (pressed(src, 3) &&"):page.index("const x = gp.axes.length >= 4")]
    assert "continue;" not in drive                               # in the drive the sticks still turn the view


def test_station_mesh_textures_kept_in_16_bits_and_retoned_as_the_panoramas(tmp_path):
    """0.99.40 (Christian, 19 Nov 2026: "the ...mesh_terrain.jpg and full 3D texture should be made projected as 16-bit
    images and converted to 8-bit the same way as the ods"): the 16-bit texture beside the .jpg, and the .jpg made
    again from it by the panoramas' conversion."""
    from types import SimpleNamespace as NS
    import cv2
    from rpp.ibr import station_mesh as SM
    from rpp.ibr.linear16 import tone_pipeline
    rng = np.random.default_rng(0)
    tex = (rng.random((64, 64, 3)) * 300.0).astype(np.float32)          # linear, 8-bit units
    k16 = 64.0 * 0.8
    to16 = lambda rgb: np.clip(np.rint(rgb * k16), 0, 65535).astype(np.uint16)                       # noqa: E731
    to8 = lambda rgb: np.clip(np.rint(tone_pipeline(np.clip(np.rint(rgb * k16), 0, 65535) / k16)), 0, 255).astype(np.uint8)   # noqa: E731
    part = {"texture": tex, "V": np.zeros((3, 3)), "F": np.array([[0, 1, 2]]), "uv": np.zeros((3, 2)), "record": {}}
    st = NS(pivot=(0, 0, 0), east=(1, 0, 0), up=(0, 0, 1), north=(0, 1, 0))
    rec = SM.write_part(tmp_path, "S", "terrain", part, st, to8, to16)
    f16 = tmp_path / "S_station_mesh_terrain_16.png"
    assert f16.is_file() and "S_station_mesh_terrain_16.png" in rec["files"]
    a = cv2.imread(str(f16), cv2.IMREAD_UNCHANGED)
    assert a.dtype == np.uint16 and np.array_equal(a[..., ::-1], to16(tex))
    before = cv2.imread(str(tmp_path / "S_station_mesh_terrain.jpg"))
    (tmp_path / "S_station_mesh_terrain.jpg").unlink()
    done = SM.retone_textures(tmp_path, lambda b: np.rint(tone_pipeline(b / k16)).astype(np.uint8))
    after = cv2.imread(str(tmp_path / "S_station_mesh_terrain.jpg"))
    assert done == ["S_station_mesh_terrain.jpg"] and np.abs(after.astype(int) - before.astype(int)).mean() < 1.0
    assert SM.write_part(tmp_path, "T", "scape", dict(part, shell=None), st, to8)["files"] == ["T_station_scape_terrain.npz", "T_station_scape_terrain.jpg"]
    import inspect
    from rpp.ibr import linear16
    assert "retone_textures(out" in inspect.getsource(linear16.retone)


def test_full_3d_texture_color_skins_and_their_listing(tmp_path):
    """0.99.41 (Christian, 19 Nov 2026: "The different color skins for the terrain should be visible in both modes.
    this means they should be rendered as in the full 3d projection too"): the skins of the 16-bit full 3D texture
    with the station's parameters, white kept white; web.station_scape lists them."""
    import cv2
    import json as _json
    from rpp.ibr import station_mesh as SM, web
    rng = np.random.default_rng(1)
    a = (rng.random((256, 256, 3)) * 9000 + 1000).astype(np.uint16)
    a[:20] = 65535                                                  # the white sky
    stem = "S_station_scape_terrain"
    cv2.imwrite(str(tmp_path / (stem + "_16.png")), a)
    cv2.imwrite(str(tmp_path / (stem + ".jpg")), (a // 257).astype(np.uint8))
    (tmp_path / (stem + ".json")).write_text(_json.dumps({"texels_per_deg": 10.0}))
    (tmp_path / "ods.json").write_text(_json.dumps({"linear16": {"exposure16": 1.0, "white_balance": [1.1, 1.0, 1.4]}}))
    lv = tmp_path / "tone_levels.json"
    lv.write_text(_json.dumps({"enhanced_gamma_stretch_s_levels": [150.0, 20.0, 200.0, 100.0],
                               "color_products_parameters_v2": {"white_linear_8bit_units": 255.0,
                                   "tone": {"base_median_log": -1.0, "base_compression": 0.5, "display_median": 0.3},
                                   "dcs": {"mean_bgr": [0.5, 0.5, 0.5], "matrix_bgr": np.eye(3).tolist()}}}))
    got = SM.write_skins(tmp_path, lv)
    assert set(got) == {"enhanced local", "enhanced tone", "raw color", "dcs color"}
    for f in got.values():
        v = cv2.imread(str(tmp_path / f))
        assert v.shape == (256, 256, 3) and v[:12].min() >= 245 and v[40:].std() > 1      # white stays white (JPEG)
    (tmp_path / "S_station_scape.glb").write_bytes(b"glTF")
    (tmp_path / "S_station_scape.json").write_text(_json.dumps({"triangles": 3, "fade_m": [18, 20]}))
    pano = tmp_path / "panos"
    pano.mkdir()
    sc = web.station_scape(tmp_path, tmp_path, pano)
    assert sc["skins"]["dcs color"] == "panos/S_station_scape_terrain_dcs.jpg" and (pano / "S_station_scape_terrain_dcs.jpg").is_file()
    lv.write_text("{}")                                             # no parameters yet: the original color only
    assert set(SM.write_skins(tmp_path, lv)) == {"raw color"}


def test_sky_texels_keep_the_far_hills():
    """0.99.42: the far texels are the sky and the hills beyond the mesh - only the bright ones joined to the zenith
    are sky."""
    from rpp.ibr import station_mesh as SM
    S, rpx, knee = 512, 250.0, (np.radians(95.0), 0.97)
    ax = np.arange(S) + 0.5 - S / 2
    rho = np.hypot(ax[None, :], ax[:, None])
    in_disc = rho <= rpx
    far = in_disc & (rho > 200)
    rgb = np.full((S, S, 3), 60.0, np.float32)                     # the ground
    rgb[far] = 240.0                                               # the sky ...
    hill = far & (rho < 215) & (ax[None, :] > 0)                   # ... and a dark hill on the east half, beyond the mesh
    rgb[hill] = 70.0
    have = in_disc.copy()
    have[far & ~hill & (ax[:, None] > 150)] = False                # a part of the sky no frame colored
    sky = SM.sky_texels(rgb, far, have, in_disc, S, rpx, knee)
    assert not sky[hill].any() and not sky[in_disc & ~far].any()
    assert sky[far & ~hill].mean() > 0.999
    assert "surfState.drive !== !!(sm.drive && sm.whole)" in (__import__("pathlib").Path(SM.__file__).resolve().parents[3] / "docs" / "ods" / "index.html").read_text(encoding="utf-8")


def test_zcam_companions_follow_their_navcam_station_and_block_cameras(tmp_path, monkeypatch):
    """0.99.43 (Christian, 19 Nov 2026: "for stations with Z34 present ... render those z34 versions right after the ncam
    versions" - "for every site from which we have a z34 360"; "the whole block depth should use all the ecams, low to
    high res and fuse them together in one mesh")."""
    import copy
    from rpp.ibr import single_sites as SS
    cfg = copy.deepcopy(SS.load_config())
    lab = "Sol0461_S026D0630"
    z = [f"ZL0_0466_07083{i:05d}_000RAD_N0260630ZCAM08500_0340LMJ01.png" for i in range(40)]
    z += [f"ZL0_0470_07090{i:05d}_000RAD_N0260630ZCAM08501_0340LMJ01.png" for i in range(10)]      # too few: no 360
    z += [f"ZL0_0466_07083{i:05d}_000RAD_N0260850ZCAM08500_1100LMJ01.png" for i in range(50)]      # 110 mm: not 34
    monkeypatch.setattr(SS, "block_aligned", lambda w: "pico" not in str(w))
    pairs = SS.zcam_companions(cfg, tmp_path, names_of=lambda work: z if "rockytop" in str(work) else [])
    assert "Sol1308_S061D1678" in cfg["_zcam_pending"]            # 0.99.44: its Mastcam-Z block is not aligned yet
    zlab = "Sol0466_S026D0630_Z34"
    assert pairs[lab] == zlab and pairs["Sol0686_S032D1184"] == "Sol0690_S032D1184_Z34"           # the config's own pair
    assert "Sol0474_S026D0850" not in pairs
    st = cfg["stations"][zlab]
    assert SS.is_zcam_station(st) and st["work"].endswith("rockytop_zcam") and st["ods_args"][:4] == ["--cameras", "Z", "--sols", "466"]
    assert "N?F_*_N0260630NCAM*.png" in st["context"] and SS.viewer_name(st, zlab).endswith("(zcam)")
    oa = st["ods_args"]                                            # 0.99.46: the 34 mm frames of that sol named, both eyes
    assert oa[oa.index("--images") + 1:] == sorted(n for n in z if n.startswith("ZL0_0466") and "_0340" in n)
    runs = SS.plan(cfg, tmp_path, labels=[lab, "Sol0474_S026D0850"], companions=pairs, block_cameras="NF", terrain="block")
    assert [r["label"] for r in runs] == [lab, zlab, "Sol0474_S026D0850"]
    assert not any(c["step"] == "depthmaps" for c in runs[1]["commands"])
    zods = next(c["argv"] for c in runs[1]["commands"] if c["step"] == "ods")
    assert "--images" in zods and "--sols" in zods
    dense = next(c["argv"] for c in runs[0]["commands"] if c["step"] == "dense")
    mesh = next(c["argv"] for c in runs[0]["commands"] if c["step"] == "mesh")
    assert dense[dense.index("--block-cameras") + 1] == "NF" and "--block-images" not in dense      # --terrain block: not the named set
    assert mesh[mesh.index("--points") + 1].endswith(str(__import__("pathlib").Path("colmap") / "dense_block_FN" / "fused.ply"))
    plain = SS.plan(cfg, tmp_path, labels=[lab])
    d0 = next(c["argv"] for c in plain[0]["commands"] if c["step"] == "dense")
    assert "--block-cameras" not in d0 and "--block-name" in d0                                     # the config's own, unchanged


def test_wait_for_blocks_runs_what_comes_in_and_stops(tmp_path, monkeypatch):
    """0.99.44 (Christian, 19 Nov 2026: "run only on aligned blocks. Skip a bock if it doesn't exist and come back to it
    later. When there are no more stations to run, then stop")."""
    from types import SimpleNamespace as NS
    from rpp.ibr import single_sites as SS
    calls = []

    def one(a):
        calls.append(set(a._skip))
        n = len(calls)
        a._ran = ["A"] if n == 1 else ["B"] if n == 3 else []
        a._late = ["B"] if n < 3 else []
        return 0
    monkeypatch.setattr(SS, "_single_sites_pass", one)
    monkeypatch.setattr("time.sleep", lambda s: None)
    out = SS.cmd_single_sites(NS(wait_for_blocks=36.0, poll_min=15.0, dry_run=False, any_block=False))
    assert out == 0 and calls == [set(), {"A"}, {"A"}]
    calls.clear()
    monkeypatch.setattr(SS, "_single_sites_pass", lambda a: (calls.append(1), setattr(a, "_ran", []), setattr(a, "_late", ["B"]), 0)[-1])
    assert SS.cmd_single_sites(NS(wait_for_blocks=0.0, poll_min=15.0, dry_run=False, any_block=False)) == 0 and len(calls) == 1


def test_stations_in_sol_order_butler_z34_and_rose_river_falls():
    """0.99.46 (Christian, 19 Nov 2026: "put the single sites in sol order. add Sol0011_S003D0000 and a _Z34 for every
    z34 360"; "use rose river falls instead")."""
    from rpp.ibr import single_sites as SS
    cfg = SS.load_config()
    runs = SS.plan(cfg, "/r", labels=None)
    labs = [r["label"] for r in runs]
    sols = [int(l[3:7]) for l in labs]
    assert sols == sorted(sols) and labs[0] == "Sol0009_S003D0000"
    runs = SS.plan(cfg, "/r", labels=SS.sol_order(list(cfg["stations"])), companions={"Sol0009_S003D0000": "Sol0011_S003D0000_Z34"})
    labs = [r["label"] for r in runs]
    assert labs[:2] == ["Sol0009_S003D0000", "Sol0011_S003D0000_Z34"]
    z = next(c["argv"] for c in runs[1]["commands"] if c["step"] == "ods")
    assert "Z?0_0011_*_N0030000ZCAM*_034*" in z and not any(c["step"] == "depthmaps" for c in runs[1]["commands"])
    assert cfg["stations"]["Sol0449_S026D0000"]["work"] == "mars2020_sol_0449_rose_river_falls_zcam"
    import json as _json
    from rpp.waypoints.sites import load_site_table
    g = load_site_table().get("groups") or _json.loads((__import__("pathlib").Path(SS.__file__).resolve().parents[1] / "data" / "sites.json").read_text())["groups"]
    assert "rose_river_falls" in g["zcam_consensus"] and "knob_mountain_large" not in g["zcam_consensus"]
