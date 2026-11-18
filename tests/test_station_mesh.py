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
    assert r["triangles"] == 12 and 0 < r["textured_triangles"] < 12 and r["viewpoints"] >= 60
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
    assert acc["max"][1] < 0 and -10.5 < acc["min"][0] and acc["max"][0] < 10.5     # y up: the ground below the eyes


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
    assert web.TERRAIN_3D == ("terrain3d", "3D terrain and rover")
    src = (SS.__file__)
    page = (__import__("pathlib").Path(src).resolve().parents[3] / "docs" / "ods" / "index.html").read_text(encoding="utf-8")
    assert "vr.station_mesh || vr.rover_surface" in page and "holdNearPivot" in page
