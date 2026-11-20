"""1.0: the JOSS figures 1-4 in the standard figure set (rpp_studies.paper.joss_figures)."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rpp_studies")
pycolmap = pytest.importorskip("pycolmap")
pytest.importorskip("open3d")
cv2 = pytest.importorskip("cv2")


def _block(tmp_path):
    """The synthetic station of test_ibr as an aligned block: its model in colmap/sparse/cahv_ba, with tie points."""
    import test_ibr as T
    proj, rec, st, scene, P, colors = T._station_scene(tmp_path)
    rng = np.random.default_rng(0)
    for x, y in rng.uniform(-8, 8, (400, 2)):
        rec.add_point3D(np.array([5 + x, 5 + y, 0.0]), pycolmap.Track(), np.array([200, 120, 60], np.uint8))
    model = proj.root / "sparse" / "cahv_ba"
    model.mkdir(parents=True)
    rec.write(str(model))
    return proj.root.parent, rec


def test_the_figure_list_and_its_names():
    from rpp_studies.paper.joss_figures import entries
    names = [n for _, n, _ in entries({})]
    assert names == ["joss_figure1_pipeline_diagram.png", "joss_figure2_navcam_undistorted_sol0009.png",
                     "joss_figure2b_navcam_undistorted_sol0052.png", "joss_figure3_block_butler_landing.png",
                     "joss_figure3b_block_van_zyl_overlook.png", "joss_figure3c_block_rockytop.png",
                     "joss_figure4_station_sol0009.png", "joss_figure4b_station_sol0461.png",
                     "joss_figure4c_station_sol0052.png"]


def test_make_all_skips_each_missing_input_with_one_line(tmp_path, capsys):
    from rpp_studies.paper.joss_figures import make_all
    res = make_all(tmp_path / "out", root=tmp_path / "none", freeze_root=None)
    assert res["figure1"] == "made"
    assert (tmp_path / "out" / "joss_figure1_pipeline_diagram.png").is_file()
    assert (tmp_path / "out" / "joss_figure1_pipeline_diagram.svg").is_file()
    for k, v in res.items():
        if k != "figure1":
            assert v.startswith("skipped: "), (k, v)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("JOSS ")]
    assert len(lines) == 9 and all("\n" not in ln for ln in lines)
    assert any("station_scape.glb" in ln for ln in lines if ln.startswith("JOSS figure4c"))


def test_the_model_readers_agree_with_pycolmap(tmp_path):
    from rpp_studies.paper.joss_figures import MODEL_DIR, read_cameras_bin, read_images_bin, read_points3d_bin
    block, rec = _block(tmp_path)
    ims = read_images_bin(block / MODEL_DIR)
    for im in rec.images.values():
        assert np.allclose(ims[im.name]["C"], im.projection_center(), atol=1e-9)
        assert ims[im.name]["camera_id"] == im.camera_id
    assert set(read_images_bin(block / MODEL_DIR, {"NLF_2.png"})) == {"NLF_2.png"}
    cams = read_cameras_bin(block / MODEL_DIR)
    assert cams[1][1:3] == (400, 300) and np.allclose(cams[1][3], [200, 200, 150])
    tp = read_points3d_bin(block / MODEL_DIR)
    assert len(tp["xyz"]) == 400 and np.all(tp["rgb"] == [200, 120, 60]) and np.allclose(tp["xyz"][:, 2], 0)


def test_figure2_and_figure3_on_a_small_block(tmp_path, monkeypatch):
    from rpp_studies.paper import joss_figures as J
    block, rec = _block(tmp_path)
    info = J.figure2_navcam(block, ["NLF_1", "NLF_2"], tmp_path / "f2.png", dpi=60)
    assert (tmp_path / "f2.png").is_file() and len(info["cameras"]) == 2
    monkeypatch.setitem(J.FIGURE3, "rockytop", dict(J.FIGURE3["rockytop"], block="mars2020_sol_0091_x*", half=8.0, res=0.1,
                                                     clean=False, eye=(0.0, -12.0, 4.0), tgt=(0.0, 2.0, 0.0), focal=800.0))
    monkeypatch.setattr(J, "FIG3_W", 400)
    monkeypatch.setattr(J, "FIG3_H", 143)
    info = J.figure3_block("rockytop", {"root": block.parent, "freeze": None}, tmp_path / "f3.png", dpi=60)
    assert (tmp_path / "f3.png").is_file()
    assert info["tie_points"]["points"] == 400 and info["images"] == 4 and info["rovers"] == []
    assert "panel D left out" in info["mesh"]                                  # no station mesh: no panel D


def test_figure4_from_a_station_folder(tmp_path):
    """A full 3D model written by the station run's own writer (station_mesh.glb_bytes) read and rendered: a ground
    quad whose texture is red in its top-left quarter (glTF uv (0, 0)) seen from above."""
    from rpp.ibr.station_mesh import glb_bytes
    from rpp_studies.paper import joss_figures as J
    lab = "Sol0009_S003D0000"
    d = tmp_path / "f3root" / "single_sites" / lab
    (d / "with_rover").mkdir(parents=True)
    pano = np.full((400, 1200, 3), 90, np.uint8)
    cv2.imwrite(str(d / f"{lab}_360_3DTD.jpg"), pano)
    cv2.imwrite(str(d / "with_rover" / f"{lab}_rover_360_3DTD.jpg"), pano)
    cv2.imwrite(str(d / f"{lab}_station_scape_terrain.jpg"), np.full((300, 300, 3), 150, np.uint8))
    tex = np.full((64, 64, 3), 255, np.uint8)
    tex[:32, :32] = (0, 0, 255)                                  # BGR: red at the top left
    ok, jpg = cv2.imencode(".jpg", tex)
    V = np.array([[-5, -1.5, -5], [5, -1.5, -5], [5, -1.5, 5], [-5, -1.5, 5]], float)   # x east, y up, z south
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    rover = (np.array([[0, -1, 1], [0.5, -1, 1], [0.5, -0.5, 1]], float), np.array([[0, 1, 2]]), np.array([[0, 0], [1, 0], [1, 1]], float))
    (d / f"{lab}_station_scape.glb").write_bytes(glb_bytes([("terrain", V, F, uv, jpg.tobytes()), ("rover", *rover, "terrain")]))
    parts = J.read_scape_glb(d / f"{lab}_station_scape.glb")
    assert set(parts) == {"terrain", "rover"} and parts["terrain"][3].shape == (64, 64, 3)
    view = J.View([0, 5, 0], [0, -1, 0.0001], (0, 0, -1), 100.0, 200, 200)    # from above, north (-z) up the picture
    img = J.render_scape({"terrain": parts["terrain"]}, view)
    tl, br = img[60, 60], img[140, 140]                          # the north-west quarter is the texture's top left
    assert tl[0] > 200 and tl[1] < 80 and br.min() > 200
    info = J.figure4_station("sol0009", {"freeze": tmp_path / "f3root", "root": None}, tmp_path / "f4.png", dpi=60)
    assert (tmp_path / "f4.png").is_file() and info["station"] == lab
