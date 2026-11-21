"""1.0: the depth filter before the fusion and the variant comparison (rpp.sfm.depth_filter, rpp.sfm.fusion_compare):
made apart from the station products, so nothing of them can get worse."""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pycolmap = pytest.importorskip("pycolmap")


def _ground_with_rock(h=200, w=300):
    """A depth map of ground rising away from the camera, a rock in front of it, one flying pixel row on the rock's
    outline, and a 5 x 5 island of sky depth; normals toward the camera except a back-facing patch."""
    v = np.arange(h, dtype=np.float32)[:, None]
    D = np.repeat(4.0 + 6.0 * (h - v) / h, w, axis=1).astype(np.float32)        # 4 m (bottom) to 10 m (top), smooth
    D[80:140, 100:160] = 3.0                                                     # the rock
    D[79, 100:160] = 6.0                                                         # flying pixels above it
    D[20:25, 250:255] = 50.0                                                     # an island far behind
    D[0:20, :] = 0.0                                                             # no depth (sky)
    D[:20, 250:255] = 0.0
    N = np.zeros((h, w, 3), np.float32)
    N[..., 2] = -1.0                                                             # toward the camera
    N[150:160, 10:20, 2] = 1.0                                                   # back-facing
    return D, N


def test_filter_depth_map_drops_edges_speckles_and_back_faces():
    from rpp.sfm.depth_filter import camera_rays, filter_depth_map
    D, N = _ground_with_rock()
    rays = camera_rays(D.shape[1], D.shape[0], 300.0, 300.0, 150.0, 100.0)
    keep, rep = filter_depth_map(D, N, rays, speckle_px=64)
    assert not keep[79, 100:160].any()                                           # the flying pixels
    assert not keep[80, 100:160].any() and not keep[139, 100:160].any()           # the rock's outline, both sides
    assert keep[110, 130] and keep[110, 50] and keep[180, 200]                   # the rock's and the ground's inside
    assert not keep[22, 252]                                                     # the island
    assert not keep[155, 15] and rep["back_facing"] == 100
    smooth = keep[150:200, 200:300]                                              # the ground (0.3 % a row) is not cut
    assert smooth.all()
    assert rep["kept"] == int(keep.sum()) and rep["valid"] > rep["kept"] > 0.9 * rep["valid"]
    keep2, rep2 = filter_depth_map(D, N, rays, max_range_m=5.0, speckle_px=0, edge_rel=0, back_facing=False)
    assert rep2["beyond_range"] > 0 and not keep2[30, 10] and keep2[199, 150]    # range = depth x |ray|


def test_filter_workspace_leaves_the_workspace_maps_alone(tmp_path, monkeypatch):
    import test_ibr as T
    from rpp.sfm.dense import read_colmap_array
    from rpp.sfm.depth_filter import filter_workspace
    from rpp.sfm.fusion_compare import fuse, score_mesh, variant_workspace, verdict
    T.test_dense_subset_prepare_and_fusion_write_fused_ply(tmp_path, monkeypatch)
    ws = next(tmp_path.rglob("dense_S003D0010"))
    maps = sorted((ws / "stereo" / "depth_maps").glob("*.geometric.bin"))
    before = {f.name: f.read_bytes() for f in maps}
    out = filter_workspace(ws, tmp_path / "cmp" / "filtered")
    assert out["images"] == len(maps) and 0 < out["kept_fraction"] <= 1
    assert {f.name: f.read_bytes() for f in maps} == before                       # the workspace's own maps unchanged
    for f in maps:
        A = read_colmap_array(f)
        B = read_colmap_array(tmp_path / "cmp" / "filtered" / "stereo" / "depth_maps" / f.name)
        assert A.shape == B.shape and np.all((B == 0) | (B == A)) and (B > 0).sum() <= (A > 0).sum()
    settings = {"fusion_options": {"min_num_pixels": 1}, "max_image_size": 400, "mask_path": None}
    for name, src, two in (("filtered", tmp_path / "cmp" / "filtered", False), ("two_pass", ws, True)):
        vws = variant_workspace(ws, tmp_path / "cmp" / name, src)
        r = fuse(vws, settings, two_pass=two)
        assert r["fused.ply"]["points"] > 1000 and (vws / "fused.ply.vis").is_file()
        if two:
            assert r["merge"]["points"] == r["fused_strict.ply"]["points"] + r["merge"]["loose_kept"]
    assert (ws / "fused.ply").read_bytes()                                        # the product cloud is still there
    # the score: a plane mesh z = 0 out to 30 m (the scene's ground), tie points on it agree, points 1 m up do not
    md = tmp_path / "mesh_plane"
    md.mkdir()
    import open3d as o3d
    from rpp.ibr.mesh import write_mesh
    g = np.linspace(-40, 40, 41)
    X, Y = np.meshgrid(g, g)
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    F = [[i * 41 + j, i * 41 + j + 1, (i + 1) * 41 + j] for i in range(40) for j in range(40)] + \
        [[i * 41 + j + 1, (i + 1) * 41 + j + 1, (i + 1) * 41 + j] for i in range(40) for j in range(40)]
    write_mesh(md / "mesh.ply", V, np.asarray(F))
    (md / "mesh_report.json").write_text(json.dumps({"pivot_block_m": [0, 0, 2.0], "terrain_triangles": [0, len(F)]}))
    rng = np.random.default_rng(1)
    P = np.c_[rng.uniform(-10, 10, (2000, 2)), np.zeros(2000)]          # within 15 m: tau under 0.6 m
    good = score_mesh(md, P)
    assert good["agree_all"] == 1.0 and all(r.get("agree", 1.0) == 1.0 for r in good["rings"])
    Q = P.copy()
    Q[:1000, 2] = 1.0
    bad = score_mesh(md, Q)
    assert bad["agree_all"] < 0.6
    assert verdict(good, bad)["verdict"] == "worse" and verdict(good, good)["verdict"] == "not worse"
    assert verdict(bad, good)["verdict"] == "better"
    assert o3d is not None


def test_replace_and_the_mesh_argv_of_a_station():
    from rpp.sfm.fusion_compare import _replace
    a = ["mesh", "W", "--station", "S", "--points", "old.ply", "--out", "o"]
    assert _replace(_replace(a, "--points", "new.ply"), "--out", "x") == ["mesh", "W", "--station", "S", "--points",
                                                                           "new.ply", "--out", "x"]
    assert _replace(a, "--colmap", None) == a


def test_the_comparison_runs_on_a_listed_station(tmp_path, monkeypatch):
    """compare_station end to end, the meshing step replaced: the variants' clouds in compare/, the verdicts, the
    table; the block workspace's fused.ply and maps are not touched."""
    import subprocess
    import test_ibr as T
    from rpp.sfm import fusion_compare as FC
    T.test_dense_subset_prepare_and_fusion_write_fused_ply(tmp_path, monkeypatch)
    ws = next(tmp_path.rglob("dense_S003D0010"))
    (ws / "rpp_fusion.json").write_text(json.dumps({"fusion_options": {"min_num_pixels": 1}, "max_image_size": 400}))
    monkeypatch.setattr(FC, "_mesh_argv", lambda *a, **k: ["mesh", "W", "--station", "S", "--points", str(ws / "fused.ply")])
    fused_before = (ws / "fused.ply").read_bytes()
    maps_before = {f: f.stat().st_size for f in (ws / "stereo").rglob("*.bin")}

    def fake_mesh(cmd, **kw):                                  # a plane mesh where --out says, z = 0 as the scene's ground
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        from rpp.ibr.mesh import write_mesh
        V = np.array([[-50, -50, 0], [50, -50, 0], [50, 50, 0], [-50, 50, 0]], float)
        write_mesh(out / "mesh.ply", V, np.array([[0, 1, 2], [0, 2, 3]]))
        (out / "mesh_report.json").write_text(json.dumps({"pivot_block_m": [0, 0, 2.0], "terrain_triangles": [0, 2]}))
        assert cmd[cmd.index("--points") + 1].endswith("fused.ply")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(FC.subprocess, "run", fake_mesh)
    monkeypatch.setattr(FC, "tie_points", lambda sparse: np.c_[np.random.default_rng(0).uniform(-20, 20, (500, 2)), np.zeros(500)])
    rep = FC.compare_station("S", tmp_path, cfg={"stations": {}}, keep_maps=False)
    assert set(rep["variants"]) == {"current", "two_pass", "filtered", "filtered_two_pass"}
    assert all(rep["variants"][v]["verdict"] == "not worse" for v in ("two_pass", "filtered", "filtered_two_pass"))
    assert (ws / "compare" / "S_comparison.md").read_text().count("|") > 10
    assert (ws / "fused.ply").read_bytes() == fused_before
    assert {f: f.stat().st_size for f in (ws / "stereo").rglob("*.bin")} == maps_before
    assert not list((ws / "compare").glob("*/stereo/*/*.bin"))                  # the variants' maps deleted
    for v in ("two_pass", "filtered", "filtered_two_pass"):
        assert (ws / "compare" / v / "fused.ply").is_file()
    again = FC.compare_station("S", tmp_path, cfg={"stations": {}})              # cached: nothing fused again
    assert again["variants"]["filtered"]["verdict"] == "not worse"
