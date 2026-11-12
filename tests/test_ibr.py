"""mppp.ibr (0.96.0 prototypes): the pivot fit, the mesh cleanup, ray-cast depth against the analytic plane, the
photometric angles, and the ODS render on a synthetic station."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
o3d = pytest.importorskip("open3d")
from scipy.spatial.transform import Rotation  # noqa: E402

from mppp.ibr import mesh as M  # noqa: E402
from mppp.ibr import ods as O  # noqa: E402
from mppp.ibr import photometry as PH  # noqa: E402
from mppp.ibr import raycast as RC  # noqa: E402
from mppp.ibr.station import fit_pivot, parse_station, select_station, station_images  # noqa: E402

NL = [2948.943055, 2949.208628, 2573.477811, 1950.413699, 0.047527, -0.016316, 3e-05, -0.0002, 0.008977, -0.008065, 0.0, 0.0]
Z63 = [8627.12, 8636.17, 900.98, 664.32, 0.469, 0.6, 0.0028, 0.0015, 0.0, 0.0, 0.0, 0.0]


def _look(R_yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """World-to-camera rotation of a camera looking along azimuth ``yaw`` (from +y north towards +x east), tilted
    ``pitch`` degrees down; camera z forward, x right, y down; world z up."""
    base = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # looks along +y, z up -> camera y down
    tilt = Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    yaw = Rotation.from_euler("z", -R_yaw_deg, degrees=True).as_matrix()
    return tilt @ base @ yaw.T


# ----------------------------------------------------------------------------- station and pivot
def test_parse_station_and_pivot_fit():
    assert parse_station("S003D1234") == (3, 1234, None) and parse_station("3_1234") == (3, 1234, None)
    assert parse_station("sol:91") == (None, None, 91)
    with pytest.raises(ValueError):
        parse_station("abc")
    P = np.array([10.0, -3.0, 1.9])
    d = {"NL": np.array([0.2, -0.212, 0.05]), "NR": np.array([0.2, 0.212, 0.05])}
    C, R, G = [], [], []
    for yaw in range(0, 360, 30):
        Ri = _look(yaw, 10)
        for g in ("NL", "NR"):
            C.append(P + Ri.T @ d[g])
            R.append(Ri)
            G.append(g)
    fit = fit_pivot(np.array(C), np.array(R), G)
    assert fit["fallback"] is None and fit["rms_m"] < 1e-9
    # one tier: the pivot's height is not determined (every frame turns about the same axis) - it is put at the
    # mean camera height (eye level); the horizontal position and the baseline are exact
    assert not fit["height_determined"]
    assert np.allclose(fit["pivot"][:2], P[:2], atol=1e-8) and abs(fit["pivot"][2] - np.mean(np.array(C)[:, 2])) < 1e-6
    assert abs(fit["baselines_m"]["NL-NR"] - 0.424) < 1e-9
    # two tiers: everything determined
    for yaw in range(15, 360, 45):
        Ri = _look(yaw, -5)
        for g in ("NL", "NR"):
            C.append(P + Ri.T @ d[g]); R.append(Ri); G.append(g)
    fit2 = fit_pivot(np.array(C), np.array(R), G)
    assert fit2["height_determined"] and np.allclose(fit2["pivot"], P, atol=1e-6) and fit2["rms_m"] < 1e-9
    few = fit_pivot(np.array(C[:4]), np.array(R[:4]), G[:4])
    assert few["fallback"] and np.allclose(few["pivot"], np.mean(C[:4], axis=0))
    rows = [{"name": "NLF_1.png", "site": 3, "drive": 1234, "sol": 91}, {"name": "ZL0_2.png", "site": 3, "drive": 1234, "sol": 91},
            {"name": "NLF_3.png", "site": 3, "drive": 1300, "sol": 92}]
    label, got = station_images(rows, "sol:91", "N")
    assert label == "S003D1234" and [r["name"] for r in got] == ["NLF_1.png"]
    assert station_images(rows, "3_1300")[0] == "S003D1300"


# ----------------------------------------------------------------------------- mesh cleanup
def _terrain(n=20, size=10.0):
    xs = np.linspace(-size, size, n)
    X, Y = np.meshgrid(xs, xs)
    V = np.stack([X.ravel(), Y.ravel(), 0.02 * np.sin(X.ravel()) * np.cos(Y.ravel())], axis=1)
    F = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            F.append([a, a + 1, a + n]); F.append([a + 1, a + n + 1, a + n])
    return V, np.asarray(F, np.int64)


def test_cleanup_removes_floaters_rubber_sheets_and_unsupported_surface():
    V, F = _terrain()
    n0 = len(F)
    pivot = np.array([0.0, 0.0, 1.5])
    # a floater: a tiny triangle far away, not connected
    V = np.vstack([V, [[30, 30, 5], [30.1, 30, 5], [30, 30.1, 5]]])
    F = np.vstack([F, [[len(V) - 3, len(V) - 2, len(V) - 1]]])
    # a rubber sheet: a long, grazing triangle stretched from the terrain edge upward along the view ray
    a = int(np.argmin(np.linalg.norm(V[:, :2] - [8, 0], axis=1)))
    V = np.vstack([V, [[8.0, 0.05, 2.0], [8.0, -0.05, 2.0]]])
    F = np.vstack([F, [[a, len(V) - 2, len(V) - 1]]])
    pts = _terrain()[0]                                             # the fused points: the terrain only
    V2, F2, rep = M.clean_arrays(V, F, pivot, points=pts, radius_m=60)
    steps = {s["step"]: s for s in rep["steps"]}
    assert steps["components"]["removed"] == 1                      # the floater
    assert steps["rubber_sheets"]["removed"] >= 1                   # the sheet
    assert len(F2) <= n0 and rep["triangles_out"] == len(F2) and F2.max() < len(V2)
    # the radius clip
    V3, F3, rep3 = M.clean_arrays(*_terrain(), pivot, points=None, radius_m=5.0)
    cen, _, _ = M.triangle_geometry(V3, F3)
    assert np.all(np.linalg.norm(cen - pivot, axis=1) <= 5.0) and len(F3) < n0
    cov = M.coverage_by_range(V3, F3, pivot)
    assert sum(c["triangles"] for c in cov) == len(F3)


def test_far_sphere_and_tau():
    V, F = M.sphere_arrays([1, 2, 3], radius=100.0, step_deg=10.0)
    assert np.allclose(np.linalg.norm(V - [1, 2, 3], axis=1), 100.0)
    assert F.min() == 0 and F.max() == len(V) - 1
    assert len(F) == 2 * 36 * 16 + 2 * 36                            # 17 rings of vertices: 16 quad bands + two caps
    assert M.tau_m(2) == pytest.approx(0.05) and M.tau_m(10) == pytest.approx(0.24, abs=0.02)
    assert float(M.tau_m(50)) > 5.0


def test_poisson_decimate_and_write(tmp_path):
    V, F = M.sphere_arrays([0, 0, 0], radius=2.0, step_deg=6.0)
    V2, F2, rep = M.poisson_arrays(V, normals=V / 2.0, depth=5, density_quantile=0.0)
    assert len(F2) > 100 and "poisson" in rep["method"]
    V3, F3, drep = M.decimate_and_smooth(V2, F2, target_triangles=300, taubin_iterations=2)
    assert 250 <= len(F3) <= 300 and drep["triangles_before"] == len(F2)
    p = M.write_mesh(tmp_path / "m.ply", V3, F3)
    V4, F4 = M.read_mesh(p)
    assert len(F4) == len(F3)


# ----------------------------------------------------------------------------- ray casting
def _plane(z=0.0, size=200.0):
    V = np.array([[-size, -size, z], [size, -size, z], [size, size, z], [-size, size, z]])
    return V, np.array([[0, 1, 2], [0, 2, 3]], np.int64)


@pytest.mark.parametrize("model,params,w,h,pitch", [("THIN_PRISM_FISHEYE", [v / 4 if i < 4 else v for i, v in enumerate(NL)], 1280, 960, 35.0),
                                                     ("FULL_OPENCV", Z63, 1648, 1200, 20.0)])
def test_cast_depth_equals_the_analytic_plane(model, params, w, h, pitch):
    cam = pycolmap.Camera(camera_id=1, model=model, width=w, height=h, params=np.asarray(params, float))
    R = _look(30.0, pitch)
    C = np.array([2.0, -1.0, 1.8])
    scene = RC.Scene(_plane(0.0), M.sphere_arrays(C, 100.0, 5.0))
    maps = RC.cast_image(scene, cam, R, C, scale=0.25)
    d_cam, d_world, valid = RC.pixel_rays(cam, R, 0.25)
    down = d_world[..., 2] < -1e-3
    t = -C[2] / d_world[..., 2]
    z_analytic = t * d_cam[..., 2]
    on_mesh = maps["label"] == RC.LABEL_MESH
    assert on_mesh.sum() > 0.3 * on_mesh.size
    assert np.all(on_mesh <= (down & valid))                                 # a mesh hit only below the horizon
    near = on_mesh & (maps["range"] < 60)                                   # inside the sphere; beyond it the sphere is hit first
    assert np.abs(maps["depth"][near] - z_analytic[near]).max() < 1e-3
    assert np.abs(maps["range"][near] - t[near]).max() < 1e-3
    up = valid & (d_world[..., 2] > 0.05)
    assert np.all(maps["label"][up] == RC.LABEL_FAR)                       # rays above the horizon end on the sphere
    n = maps["normal"][near]
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    assert np.all(np.sum(n * d_cam[near], axis=1) < 0)                     # facing the camera
    assert np.allclose(n @ R, [0, 0, 1], atol=1e-6)                         # the plane's normal, back in the world (R^T n)
    assert maps["tri"][near].min() >= 0 and np.all(maps["tri"][maps["label"] == RC.LABEL_FAR] == -1)


def test_mask_label_save_load_preview_and_coverage(tmp_path):
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=200, height=100, params=np.array([150.0, 100.0, 50.0]))
    R, C = _look(0.0, 45.0), np.array([0.0, 0.0, 2.0])
    scene = RC.Scene(_plane(0.0), M.sphere_arrays(C, 100.0, 10.0))
    mask = np.ones((100, 200), bool)
    mask[:20] = False
    maps = RC.cast_image(scene, cam, R, C, scale=1.0, mask=mask)
    assert np.all(maps["label"][:20] == RC.LABEL_MASKED) and np.all(maps["depth"][:20] == 0)
    cov = RC.coverage(maps)
    assert abs(sum(cov.values()) - 1.0) < 1e-9 and cov["masked"] == pytest.approx(0.2)
    f = RC.save_maps(tmp_path / "maps" / "a.npz", maps, image="a.png", scale=1.0)
    back = RC.load_maps(f)
    assert back["meta"]["image"] == "a.png" and back["depth"].shape == (100, 200)
    assert np.abs(back["normal"][maps["label"] == RC.LABEL_MESH] - maps["normal"][maps["label"] == RC.LABEL_MESH]).max() < 0.01
    p1, p2 = RC.preview_png(maps, tmp_path / "png" / "a")
    assert p1.is_file() and p2.is_file()
    pan = RC.panorama_range(scene, C, width=72)
    assert pan["range"].shape == (36, 72) and np.all(pan["label"][:18] == RC.LABEL_FAR) and np.all(pan["label"][-10:] == RC.LABEL_MESH)


# ----------------------------------------------------------------------------- photometry
def test_photometric_angles_on_a_plane_and_a_shadow():
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=120, height=80, params=np.array([100.0, 60.0, 40.0]))
    R, C = _look(0.0, 40.0), np.array([0.0, 0.0, 2.0])
    V, F = _plane(0.0, 50.0)
    # a wall north of the camera casting a shadow towards it when the Sun is low in the north
    wall = np.array([[-3.0, 6.0, 0.0], [3.0, 6.0, 0.0], [3.0, 6.0, 3.0], [-3.0, 6.0, 3.0]])
    V = np.vstack([V, wall]); F = np.vstack([F, [[4, 5, 6], [4, 6, 7]]])
    scene = RC.Scene((V, F))
    maps = RC.cast_image(scene, cam, R, C, scale=1.0)
    sun_elev, sun_az = 25.0, 0.0                                              # from the north, 25 degrees up
    sun = np.array([np.cos(np.radians(sun_elev)) * np.sin(np.radians(sun_az)), np.cos(np.radians(sun_elev)) * np.cos(np.radians(sun_az)), np.sin(np.radians(sun_elev))])
    pm = PH.photometric_maps(scene, maps, R, C, sun, with_shadow=True)
    ground = (maps["label"] == RC.LABEL_MESH) & (np.abs(maps["normal"] @ R - [0, 0, 1]).sum(axis=-1) < 1e-6)   # normal (0,0,1) in the world
    assert ground.sum() > 1000
    assert np.allclose(pm["incidence"][ground], 90.0 - sun_elev, atol=1e-4)  # i = 90 - Sun elevation on level ground
    e = pm["emission"][ground]
    assert np.all((e >= 0) & (e <= 90.0))
    i, g = pm["incidence"][ground], pm["phase"][ground]
    assert np.all(g >= np.abs(i - e) - 1e-6) and np.all(g <= i + e + 1e-6)   # the spherical triangle inequality
    # the phase angle needs no normal: it is the angle between the pixel ray and the Sun
    d_cam = maps["dirs_cam"][ground]
    assert np.allclose(pm["phase"][ground], np.degrees(np.arccos(np.clip(-d_cam @ (R @ sun), -1, 1))), atol=1e-4)
    sh = pm["shadow"]
    assert (sh[ground] == PH.SHADOW_SHADOW).sum() > 50 and (sh[ground] == PH.SHADOW_LIT).sum() > 50
    # the shadow lies on the near side of the wall: closer to the camera than the wall and at most 3/tan(25) m from it
    d_world = maps["dirs_cam"].reshape(-1, 3) @ R
    X = C + maps["range"].reshape(-1, 1) * d_world
    shadowed = X[(sh.reshape(-1) == PH.SHADOW_SHADOW) & ground.reshape(-1)]
    assert np.all(shadowed[:, 1] < 6.0) and np.all(shadowed[:, 1] > 6.0 - 3.0 / np.tan(np.radians(sun_elev)) - 0.1)


def test_colmap_array_round_trip_and_photometry_files(tmp_path):
    from mppp.sfm.dense import read_colmap_array
    a = np.random.default_rng(1).uniform(0, 90, (7, 11)).astype(np.float32)
    a[0, 0] = np.nan
    f = PH.write_colmap_array(tmp_path / "x.incidence.bin", a)
    b = read_colmap_array(f)
    assert b.shape == (7, 11) and b[0, 0] == 0 and np.allclose(b[1:], a[1:])
    pm = {"incidence": a, "emission": a / 2, "phase": a * 2, "shadow": np.ones((7, 11), np.uint8)}
    p = PH.save_photometry(tmp_path / "p.npz", pm, image="x")
    back = PH.load_photometry(p)
    assert np.isnan(back["incidence"][0, 0]) and abs(back["phase"][3, 3] - a[3, 3] * 2) < 0.1 and back["meta"]["image"] == "x"
    assert len(PH.preview_png(pm, tmp_path / "png" / "x")) == 4
    assert PH.sun_for_image({"sun_enu": [0, 0, 1]}, None).tolist() == [0, 0, 1]
    assert PH.sun_for_image({"solar_azimuth_deg": 90.0, "solar_elevation_deg": 0.0}, np.eye(3)) == pytest.approx([1, 0, 0], abs=1e-9)
    assert PH.sun_for_image({}, None) is None


# ----------------------------------------------------------------------------- ODS
class _Proj:
    def __init__(self, root, rows):
        self.root, self.images = Path(root), rows


def _station_scene(tmp_path, colour_by_yaw=True):
    """A synthetic station: a pinhole camera at the pivot looking north, east, south, west (four frames, red,
    green, blue, white), the ground plane and the far sphere."""
    import cv2
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=400, height=300, params=np.array([200.0, 200.0, 150.0]))
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    root = tmp_path / "mars2020_sol_0091_x" / "colmap"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    P = np.array([5.0, 5.0, 1.8])
    rows = []
    colours = {0: (255, 0, 0), 90: (0, 255, 0), 180: (0, 0, 255), 270: (255, 255, 255)}
    for k, yaw in enumerate(colours, 1):
        R = _look(yaw, 15.0)
        Ck = P + R.T @ np.array([0.0, 0.0, 0.3])           # the camera 30 cm ahead of the pivot along its own axis
        fr = pycolmap.Frame(frame_id=k, rig_id=1)
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=k))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ Ck)
        rec.add_frame(fr)
        name = f"NLF_{k}.png"
        im = pycolmap.Image(name=name, keypoints=np.zeros((0, 2)), camera_id=1, image_id=k)
        im.frame_id = k
        rec.add_image(im)
        img = np.zeros((300, 400, 3), np.uint8)
        img[:] = colours[yaw][::-1]
        cv2.imwrite(str(root / "images" / name), img)
        rows.append({"name": name, "site": 3, "drive": 10, "sol": 91, "camera_group": "NL", "sun_enu": [0.0, 0.5, 0.8]})
    for fid in list(rec.frames):
        rec.register_frame(fid)
    proj = _Proj(root, rows)
    (root / "project.json").write_text(json.dumps({"images": rows, "cameras": {}, "rig": {}, "offset": [0, 0, 0]}), encoding="utf-8")
    st = select_station(proj, rec, "3_10", families="N")
    scene = RC.Scene(_plane(0.0, 300.0), M.sphere_arrays(st.pivot, 100.0, 5.0))
    return proj, rec, st, scene, P, colours


def test_ods_rays_geometry():
    P = np.array([1.0, 2.0, 3.0])
    oL, dL = O.ods_rays(360, 180, (90, 91), P, 0.064, "left")
    oR, dR = O.ods_rays(360, 180, (90, 91), P, 0.064, "right")
    assert np.allclose(np.linalg.norm(dL, axis=1), 1.0) and np.allclose(dL, dR)
    sep = oR - oL
    assert np.allclose(np.linalg.norm(sep, axis=1), 0.064)                 # the eyes an IPD apart
    assert np.allclose(np.sum(sep * dL, axis=1), 0.0, atol=1e-12)          # across the view direction
    assert np.allclose(sep[:, 2], 0.0)                                      # in the horizontal plane
    assert np.allclose(np.linalg.norm((oL + oR) / 2 - P, axis=1), 0.0)
    assert np.allclose(dL[0], [0, 1, 0], atol=0.02) and np.allclose(dL[90], [1, 0, 0], atol=0.02)   # N at column 0, E a quarter on
    assert np.allclose(sep[0], [0.064, 0, 0], atol=1e-3)                    # looking north, the right eye is east


def test_ods_render_colours_each_direction_from_its_frame(tmp_path):
    import cv2
    proj, rec, st, scene, P, colours = _station_scene(tmp_path)
    assert st.label == "S003D0010" and len(st.images) == 4 and st.pivot_fit["fallback"]   # four frames: the mean centre, flagged
    out = tmp_path / "ods"
    O.render_station(proj, rec, st, scene, out, width=256, ipd_m=0.064, scale=0.5, rows_per_chunk=32)
    L = cv2.imread(str(out / "S003D0010_ods_left.png"))[..., ::-1]
    Rr = cv2.imread(str(out / "S003D0010_ods_right.png"))[..., ::-1]
    assert L.shape == (128, 256, 3) and Rr.shape == L.shape and (out / "provenance.png").is_file()
    # looking north along the horizon (row 64 ~ elevation 0, column 0 azimuth 0): the north frame (red) colours it
    def at(img, az_deg, el_deg):
        u = int(az_deg / 360 * 256) % 256
        v = int((90 - el_deg) / 180 * 128)
        return img[v, u]
    for yaw, col in colours.items():
        px = at(L, yaw + 1.0, -20.0)                                         # 20 degrees below the horizon: the ground
        assert np.all(np.abs(px.astype(int) - np.array(col)) < 40), (yaw, px)
        pr = at(Rr, yaw + 1.0, -20.0)
        assert np.all(np.abs(pr.astype(int) - np.array(col)) < 40), (yaw, pr)
    # 45 degrees between two frames, their fields of view (+-45 deg horizontally) both end there: a blend or one of them
    px = at(L, 45.0, -20.0).astype(int)
    assert px.sum() > 100
    # straight up: no frame looks there (they tilt 15 deg down): the sky filled from the seen sky below (10 Nov)
    assert at(L, 10.0, 85.0).sum() > 100
    O.render_station(proj, rec, st, scene, tmp_path / "nofill", width=256, ipd_m=0.064, scale=0.5, rows_per_chunk=32,
                     sky_fill=False)
    assert np.all(at(cv2.imread(str(tmp_path / "nofill" / "S003D0010_ods_left.png"))[..., ::-1], 10.0, 85.0) == 0)
    rec_json = json.loads((out / "ods.json").read_text(encoding="utf-8"))
    cov = rec_json["eyes"]["left"]["coverage"]
    assert 0.2 < cov["coloured"] < 0.9 and cov["no_surface"] == 0.0 and rec_json["eyes"]["left"]["sources_used"] == 4
    assert rec_json["sources"] == [f"NLF_{k}.png" for k in range(1, 5)]


# ----------------------------------------------------------------------------- the CLI end to end
def test_cli_mesh_depthmaps_ods_on_a_saved_block(tmp_path):
    """``mppp mesh`` -> ``mppp depthmaps --photometry`` -> ``mppp ods`` on a saved block: the station's folder under
    the project, the fused cloud meshed (Poisson), four maps, the photometric cube, two eye images."""
    import cv2
    from mppp.cli import main as mppp_main
    from mppp.sfm.project import SfmProject
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=400, height=300, params=np.array([200.0, 200.0, 150.0]))
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    root = tmp_path / "mars2020_sol_0091_cli" / "colmap"
    (root / "images").mkdir(parents=True)
    P = np.array([5.0, 5.0, 1.8])
    rows = []
    for k, yaw in enumerate((0, 90, 180, 270), 1):
        R = _look(yaw, 15.0)
        Ck = P + R.T @ np.array([0.0, 0.0, 0.3])
        fr = pycolmap.Frame(frame_id=k, rig_id=1)
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=k))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ Ck)
        rec.add_frame(fr)
        name = f"NLF_{k}.png"
        im = pycolmap.Image(name=name, keypoints=np.zeros((0, 2)), camera_id=1, image_id=k)
        im.frame_id = k
        rec.add_image(im)
        img = np.zeros((300, 400, 3), np.uint8)
        img[:] = (40 * k, 200 - 30 * k, 90)
        cv2.imwrite(str(root / "images" / name), img)
        rows.append({"name": name, "site": 3, "drive": 10, "sol": 91, "camera_group": "NL", "sun_enu": [0.0, 0.5, 0.8]})
    for fid in list(rec.frames):
        rec.register_frame(fid)
    model = root / "gui_native" / "sparse"
    model.mkdir(parents=True)
    rec.write(str(model))
    SfmProject(root, rows, {}, {}, [0.0, 0.0, 0.0], {"world_frame": "site3_enu"}).save()
    # the fused cloud: the ground plane z = 0 within 30 m of the pivot, with normals up (a dense run's fused.ply)
    rng = np.random.default_rng(3)
    xy = rng.uniform(-30, 30, (40_000, 2)) + P[:2]
    xy = xy[np.linalg.norm(xy - P[:2], axis=1) < 30]
    pts = np.c_[xy, 0.02 * rng.standard_normal(len(xy))]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(pts), 1)))
    (root / "dense").mkdir()
    o3d.io.write_point_cloud(str(root / "dense" / "fused.ply"), pcd)

    work = str(root.parent)
    assert mppp_main(["mesh", work, "--station", "3_10", "--mask-rover", "--poisson-depth", "7", "--target-triangles", "20000", "--no-preview"]) == 0
    out = root / "ods" / "S003D0010"
    rep = json.loads((out / "mesh_report.json").read_text(encoding="utf-8"))
    assert (out / "mesh.ply").is_file() and (out / "ground_plane.json").is_file() and rep["points"] == len(pts)
    assert not (out / "far_sphere.ply").exists() and rep["scene"]["beyond"] == "the sky at infinity"
    gp = json.loads((out / "ground_plane.json").read_text(encoding="utf-8"))
    assert gp["method"] == "fit" and gp["tilt_deg"] < 0.5 and abs(gp["offset"]) < 0.05          # the plane z = 0
    assert rep["scene"]["mesh_radius_m"] == 50.0                     # 0.97.0: the mesh to 50 m
    assert rep["station"]["station"] == "S003D0010" and rep["points_file"].endswith("fused.ply")
    V, F = M.read_mesh(out / "mesh.ply")
    assert len(F) > 1000 and abs(float(np.median(V[:, 2]))) < 0.1                     # the ground plane, near z = 0
    cen, _, _ = M.triangle_geometry(V, F)
    assert M.horizontal_distance(cen, rep["pivot_block_m"]).max() <= 50.05            # 50 m from the axis (the cloud here ends at 30 m)

    assert mppp_main(["depthmaps", work, "--station", "3_10", "--mask-rover", "--scale", "0.25", "--photometry", "--no-preview"]) == 0
    maps = sorted((out / "maps").glob("*.npz"))
    assert len(maps) == 4
    m = RC.load_maps(maps[0])
    d = m["depth"]
    assert d.shape == (75, 100) and np.isfinite(d[-1, 50]) and 0.5 < d[-1, 50] < 20       # the ground at the bottom row
    assert (out / "maps_report.md").is_file() and (out / "photometry_report.json").is_file()
    pm = PH.load_photometry(sorted((out / "photometry").glob("*.npz"))[0])
    inc = pm["incidence"]
    ok = np.isfinite(inc)
    assert ok.any() and np.all(inc[ok] >= 0) and np.all(inc[ok] <= 90)

    assert mppp_main(["ods", work, "--station", "3_10", "--mask-rover", "--width", "256", "--scale", "0.25", "--rows-per-chunk", "32"]) == 0
    L = cv2.imread(str(out / "S003D0010_ods_left.png"))
    Rr = cv2.imread(str(out / "S003D0010_ods_right.png"))
    assert L.shape == (128, 256, 3) and Rr.shape == L.shape
    oj = json.loads((out / "ods.json").read_text(encoding="utf-8"))
    assert oj["eyes"]["left"]["coverage"]["coloured"] > 0.2 and oj["eyes"]["left"]["sources_used"] == 4
    assert (out / "station.json").is_file()

    # --rover (0.97.0): the terrain of the rover-masked cloud (fused.ply), then the rover model; --rover-cloud is the
    # 0.96 way - a 'deck' (points 1 m up, within 1.5 m of the axis) in fused_rover.ply stays in the mesh, the plane
    # still fitted to the terrain cloud; the products go to ods/<station>_rover
    deck = np.c_[rng.uniform(-1.5, 1.5, (6000, 2)) + P[:2], np.full(6000, 1.0)]
    pcd2 = o3d.geometry.PointCloud()
    allp = np.vstack([pts, deck])
    pcd2.points = o3d.utility.Vector3dVector(allp)
    pcd2.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(allp), 1)))
    o3d.io.write_point_cloud(str(root / "dense" / "fused_rover.ply"), pcd2)
    outr = root / "ods" / "S003D0010_rover"
    assert mppp_main(["mesh", work, "--station", "3_10", "--rover", "--rover-cloud", "--poisson-depth", "8",
                      "--target-triangles", "40000", "--no-preview"]) == 0
    rr = json.loads((outr / "mesh_report.json").read_text(encoding="utf-8"))
    assert rr["rover"] == "shown" and rr["points_file"].endswith("fused_rover.ply") and rr["points"] == len(allp)
    gpr = json.loads((outr / "ground_plane.json").read_text(encoding="utf-8"))
    assert abs(gpr["offset"]) < 0.05 and gpr["sources"]["fused_points"] == len(pts)          # the terrain, not the deck
    Vr, Fr = M.read_mesh(outr / "mesh.ply")
    assert (np.abs(Vr[:, 2] - 1.0) < 0.15).sum() > 50                                        # the deck is in the mesh
    assert mppp_main(["mesh", work, "--station", "3_10", "--rover", "--poisson-depth", "8", "--target-triangles", "40000",
                      "--no-preview"]) == 0
    rr = json.loads((outr / "mesh_report.json").read_text(encoding="utf-8"))
    assert rr["rover"] == "shown" and rr["points_file"].endswith("fused.ply") and rr["points"] == len(pts)   # terrain alone
    assert rr["fill"]["triangles"] >= 0 and "filled_by_ring" in rr["fill"]
    assert mppp_main(["ods", work, "--station", "3_10", "--rover", "--width", "128", "--scale", "0.25"]) == 0
    assert json.loads((outr / "ods.json").read_text(encoding="utf-8"))["rover"] == "shown"
    assert (out / "S003D0010_ods_left.png").is_file() and (outr / "S003D0010_rover_ods_left.png").is_file()     # both variants side by side
    assert not (outr / "S003D0010_ods_left.png").is_file()                                    # 10 Nov: the station prefix
    sq = cv2.imread(str(out / "S003D0010_360_3DTD.jpg"))                             # left eye on top: a square
    assert sq.shape == (256, 256, 3) and (out / "S003D0010_360_3DTD.png").is_file()
    Lp, Rp = cv2.imread(str(out / "S003D0010_ods_left.png")), cv2.imread(str(out / "S003D0010_ods_right.png"))
    sqp = cv2.imread(str(out / "S003D0010_360_3DTD.png"))
    assert np.array_equal(sqp[:128], Lp) and np.array_equal(sqp[128:], Rp)                   # 10 Nov: left on top
    (out / "S003D0010_360_3DTD.png").unlink()
    assert mppp_main(["ods", work, "--station", "3_10", "--mask-rover", "--stack-only", "--stack-top", "right"]) == 0
    assert np.array_equal(cv2.imread(str(out / "S003D0010_360_3DTD.png"))[:128], Rp)


# ----------------------------------------------------------------------------- 9 Nov evening: what the Rockytop run found
def _world_texture(X):
    """A texture fixed to the ground (random grey per 10 cm cell): every source frame shows the same pattern where it
    sees the same ground, so the two eyes' images differ only by their parallax."""
    i = np.floor(X[..., 0] / 0.10).astype(np.int64)
    j = np.floor(X[..., 1] / 0.10).astype(np.int64)
    return (40 + (((i * 73856093) ^ (j * 19349663)) % 176)).astype(np.uint8)


def test_ods_eyes_differ_by_the_parallax_of_the_baseline(tmp_path):
    """The stereo itself (the Rockytop sandbox run, 9 Nov: a test with one colour per frame cannot see it - the
    seams between frames are fixed in direction): with a texture fixed to the ground, the left eye's image is the
    right eye's shifted by 2 asin(IPD / 2D) at horizontal distance D, and not at all vertically."""
    import cv2
    proj, rec, st, scene, P, colours = _station_scene(tmp_path)
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        R = np.asarray(im.cam_from_world().rotation.matrix())
        C = np.asarray(im.projection_center())
        m = RC.cast_image(scene, cam, R, C, scale=1.0)
        _, dw, _ = RC.pixel_rays(cam, R, 1.0)
        X = C + m["range"][..., None] * dw
        img = np.where(m["label"] == RC.LABEL_MESH, _world_texture(X), 0).astype(np.uint8)
        cv2.imwrite(str(proj.root / "images" / im.name), img)
    W = 2048
    out = tmp_path / "ods_tex"
    O.render_station(proj, rec, st, scene, out, width=W, ipd_m=0.064, scale=1.0, rows_per_chunk=128)
    L = cv2.imread(str(out / "S003D0010_ods_left.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    Rr = cv2.imread(str(out / "S003D0010_ods_right.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    H = L.shape[0]
    h_cam = P[2]                                                        # the plane is z = 0
    for el in (-35.0, -25.0):
        v = int((90 - el) / 180 * H)
        D = h_cam / np.tan(np.radians(-el))
        expected = 2 * np.arcsin(0.032 / D) * W / (2 * np.pi)
        got = []
        for u0 in range(16, W - 96, 128):                               # windows inside the frames, off the seams
            a, b = L[v - 16:v + 16, u0:u0 + 64], Rr[v - 16:v + 16, u0:u0 + 64]
            if (a == 0).mean() > 0.01 or (b == 0).mean() > 0.01 or a.std() < 10:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(b, a, cv2.createHanningWindow((64, 32), cv2.CV_32F))
            if resp > 0.2:
                got.append((dx, dy))
        got = np.array(got)
        assert len(got) >= 4, el
        assert abs(np.median(got[:, 0]) - expected) < 0.08 * expected + 0.3, (el, expected, np.median(got[:, 0]))
        assert abs(np.median(got[:, 1])) < 0.3


def test_dense_subset_prepare_and_fusion_write_fused_ply(tmp_path, monkeypatch):
    """mppp dense --station (0.96): a sub-model of the station's frames, its own workspace; and the fusion writes
    fused.ply (pycolmap 4.2's stereo_fusion writes a model folder unless output_type="ply" - the Rockytop sandbox
    run found run_dense raising after the fusion with no fused.ply).  Patch-match needs CUDA: the depth maps are
    cast from the plane here, in COLMAP's format."""
    import cv2
    from mppp.sfm.dense import prepare_dense, run_dense, subset_images, subset_label
    from mppp.sfm.project import SfmProject
    from mppp.ibr.photometry import write_colmap_array
    proj0, rec, st, scene, P, colours = _station_scene(tmp_path)
    root = proj0.root
    for im in rec.images.values():                                      # textured, so the fusion has colours
        cam = rec.cameras[im.camera_id]
        R = np.asarray(im.cam_from_world().rotation.matrix())
        m = RC.cast_image(scene, cam, R, np.asarray(im.projection_center()), scale=1.0)
        _, dw, _ = RC.pixel_rays(cam, R, 1.0)
        X = np.asarray(im.projection_center()) + m["range"][..., None] * dw
        cv2.imwrite(str(root / "images" / im.name), np.where(m["label"] == 1, _world_texture(X), 0).astype(np.uint8))
    model = root / "gui_native" / "sparse"
    model.mkdir(parents=True)
    rec.write(str(model))
    rows = json.loads((root / "project.json").read_text(encoding="utf-8"))["images"]
    for r in rows:
        r.update(station="S003D0010", sequence="NCAM00001" if r["name"] < "NLF_3" else "NCAM00002", downsample_scale=1.0)
    proj = SfmProject(root, rows, {}, {}, [0.0, 0.0, 0.0], {"world_frame": "site3_enu"})
    proj.save()
    names = subset_images(proj, station="3_10", families="N", sequences=["NCAM00001"])
    assert names == ["NLF_1.png", "NLF_2.png"]
    assert subset_label("3_10", ["NCAM00001"]) == "dense_S003D0010_NCAM00001"
    assert len(subset_images(proj, station="3_10")) == 4 and subset_images(proj, station="3_11") == []
    ws = root / subset_label("3_10")
    res = prepare_dense(proj, ws, max_image_size=400, masks=False, images=subset_images(proj, station="3_10"))
    assert res["images"] == 4 and res["subset"] == sorted(r["name"] for r in rows)
    und = pycolmap.Reconstruction(str(ws / "sparse"))
    for im in und.images.values():                                      # the plane through each undistorted camera
        cam = und.cameras[im.camera_id]
        R = np.asarray(im.cam_from_world().rotation.matrix())
        m = RC.cast_image(scene, cam, R, np.asarray(im.projection_center()), scale=1.0)
        ground = m["label"] == RC.LABEL_MESH                            # the far sphere is no stereo depth
        write_colmap_array(ws / "stereo" / "depth_maps" / f"{im.name}.geometric.bin", np.where(ground, m["depth"], 0))
        write_colmap_array(ws / "stereo" / "normal_maps" / f"{im.name}.geometric.bin", m["normal"] * ground[..., None])
    real_opts = pycolmap.StereoFusionOptions

    def one_pixel_opts():                     # the four frames look N / E / S / W: a point from one image is enough
        o = real_opts()
        o.min_num_pixels = 1
        return o
    monkeypatch.setattr(pycolmap, "StereoFusionOptions", one_pixel_opts)
    out = run_dense(proj, workspace=ws, steps=("fusion",), max_image_size=400, images=res["subset"])
    assert (ws / "fused.ply").is_file() and out["steps"]["fusion"]["points"] > 1000
    pts = np.asarray(o3d.io.read_point_cloud(str(ws / "fused.ply")).points)
    assert np.median(np.abs(pts[:, 2])) < 0.01                           # on the plane z = 0


def test_dense_stereo_goes_to_the_pipelines_gpu_python(tmp_path, monkeypatch):
    """9 Nov: mppp dense --stereo looked only at GPU_PYTHON; on the PC the CUDA python is gpu_python.txt (MPPP_GPU_PY),
    so the stereo would have run in the CPU pycolmap and stopped.  Now: --python, else GPU_PYTHON, else the alignment's
    own lookup (pipeline._default_gpu_python: MPPP_GPU_PYTHON, MPPP_GPU_PY, gpu_python.txt, the mppp_gpu env)."""
    import mppp.sfm.gpu as G
    from mppp.cli import main as mppp_main
    from mppp.sfm.project import SfmProject
    proj0, rec, st, scene, P, colours = _station_scene(tmp_path)
    root = proj0.root
    (root / "gui_native" / "sparse").mkdir(parents=True)
    rec.write(str(root / "gui_native" / "sparse"))
    rows = json.loads((root / "project.json").read_text(encoding="utf-8"))["images"]
    for r in rows:
        r.update(station="S003D0010", sequence="NCAM00001", downsample_scale=1.0)
    SfmProject(root, rows, {}, {}, [0.0, 0.0, 0.0], {"world_frame": "site3_enu"}).save()
    calls = []
    monkeypatch.delenv("GPU_PYTHON", raising=False)
    monkeypatch.setattr(pycolmap, "has_cuda", False, raising=False)
    import mppp.pipeline as PL
    monkeypatch.setattr(PL, "_default_gpu_python", lambda: "/envs/mppp_gpu/python.exe")
    monkeypatch.setattr(G, "run_step", lambda py, step, proj, **kw: calls.append((py, step, kw)) or {"steps": {}})
    assert mppp_main(["dense", str(root.parent), "--station", "3_10", "--stereo", "--fusion"]) == 0
    py, step, kw = calls[0]
    assert py == "/envs/mppp_gpu/python.exe" and step == "dense" and kw["steps"] == ["stereo", "fusion"]
    assert kw["workspace"].endswith("dense_S003D0010") and len(kw["images"]) == 4
    from mppp.sfm.dense import STATION_FUSION_OPTIONS, STATION_STEREO_OPTIONS
    assert kw["stereo_options"] == STATION_STEREO_OPTIONS and kw["fusion_options"] == STATION_FUSION_OPTIONS
    calls.clear()
    assert mppp_main(["dense", str(root.parent), "--station", "3_10", "--stereo", "--min-angle", "1.5", "--min-views", "3"]) == 0
    kw = calls[0][2]
    assert kw["stereo_options"] == {"min_triangulation_angle": 1.0, "filter_min_triangulation_angle": 1.5,
                                    "filter_min_num_consistent": 2}
    assert kw["fusion_options"] == {"min_num_pixels": 3}


def test_station_dense_options_reach_the_mesh_radius_and_are_set_on_pycolmap():
    """10 Nov, Rockytop S026D0630 on the PC: COLMAP's 3 deg filter angle stopped the fused cloud at 7.7 m (Navcam
    0.424 m baseline: 8.1 m).  The station options reach the mesh radius, and set_options puts them on pycolmap's
    option objects (a misspelt field raises instead of being ignored)."""
    import math
    import pytest
    from mppp.ibr.mesh import DEFAULT_RADIUS_M
    from mppp.sfm.dense import STATION_FUSION_OPTIONS, STATION_STEREO_OPTIONS, set_options
    assert 0.424 / math.tan(math.radians(3.0)) < 8.2                     # the default: the 8 m cut-off seen
    assert 0.424 / math.tan(math.radians(STATION_STEREO_OPTIONS["filter_min_triangulation_angle"])) >= DEFAULT_RADIUS_M
    po, fo = pycolmap.PatchMatchOptions(), pycolmap.StereoFusionOptions()
    assert po.filter_min_triangulation_angle == 3.0 and fo.min_num_pixels == 5   # COLMAP's defaults, still
    got = set_options(po, STATION_STEREO_OPTIONS)
    assert po.filter_min_num_consistent == 1 and abs(po.filter_min_triangulation_angle - 0.4) < 1e-6
    assert set(got) == set(STATION_STEREO_OPTIONS)
    set_options(fo, STATION_FUSION_OPTIONS)
    assert fo.min_num_pixels == 2 and isinstance(fo.min_num_pixels, int)
    with pytest.raises(ValueError):
        set_options(fo, {"min_num_pixel": 2})


# ----------------------------------------------------------------------------- 10 Nov: mesh to 20 m, ground plane to 100 m, sky at infinity
def test_ground_plane_follows_the_regional_slope_and_ignores_outliers():
    """The plane through the (sector, ring) medians: an 8 deg slope with rocks and a few wild points is recovered to
    a fraction of a degree; too few bins, or asked, gives the horizontal plane at the median height."""
    rng = np.random.default_rng(5)
    P = np.array([0.0, 0.0, 2.0])
    xy = rng.uniform(-100, 100, (60_000, 2))
    xy = xy[np.hypot(*xy.T) <= 100]
    slope = np.tan(np.radians(8.0))
    z = slope * xy[:, 0] + 0.15 * rng.standard_normal(len(xy))                 # uphill towards +x
    rocks = rng.random(len(xy)) < 0.05
    z[rocks] += rng.uniform(0.2, 1.5, rocks.sum())
    pts = np.c_[xy, z]
    wild = np.c_[rng.uniform(-50, 50, (300, 2)), rng.uniform(-40, 60, 300)]   # mismatched tie points
    gp = M.fit_ground_plane(np.vstack([pts, wild]), P)
    assert gp["method"] == "fit" and abs(gp["tilt_deg"] - 8.0) < 0.3
    assert np.allclose(gp["downhill_direction"], [-1, 0, 0], atol=0.02)
    n = np.asarray(gp["normal"])
    assert abs(gp["offset"] - n @ np.array([0.0, 0.0, 0.1])) < 0.2                # through the median ground at the axis
    assert gp["bins"] > 200 and gp["by_ring"][-1]["ring_m"] == [65.0, 100.0]
    flat = M.fit_ground_plane(pts, P, horizontal=True)
    assert flat["tilt_deg"] == 0.0 and flat["method"].startswith("horizontal")
    few = M.fit_ground_plane(pts[np.hypot(*pts[:, :2].T) < 4], P)               # nothing beyond 4 m: 1 ring
    assert few["method"].startswith("horizontal")


def test_scene_mesh_then_plane_then_sky():
    """Priority, not nearest: a ray on the mesh stays on it even where the plane is nearer; a ray through a hole
    in the mesh or beyond its 20 m reach ends on the plane; above the horizon, or beyond the plane's 100 m, the sky
    at infinity (label far, no depth in the maps). Only the mesh occludes."""
    P = np.array([0.0, 0.0, 2.0])
    V, F = _terrain(n=41, size=20.0)                                            # terrain z ~ 0 to |x|,|y| <= 20 m
    V[:, 2] -= 0.5                                                               # the mesh 0.5 m below the plane
    cen, _, _ = M.triangle_geometry(V, F)
    keep = ~((np.abs(cen[:, 0] - 6) < 1.5) & (np.abs(cen[:, 1]) < 1.5))          # a hole 6 m east
    V, F = M.keep_triangles(V, F, keep)
    plane = {"normal": [0, 0, 1], "offset": 0.0, "centre": P, "up": [0, 0, 1], "radius_m": 100.0}
    sc = RC.Scene((V, F), plane=plane, sky=True)

    def ray(x, y, z):
        d = np.array([[x, y, z]], float) - P
        return sc.cast(P, d / np.linalg.norm(d))
    h = ray(3.0, 3.0, -0.5)
    assert h["label"][0] == RC.LABEL_MESH and abs(h["t"][0] - np.linalg.norm([3, 3, -2.5])) < 0.05
    h = ray(6.0, 0.0, 0.0)                                                       # through the hole: the plane z = 0
    assert h["label"][0] == RC.LABEL_PLANE and abs(h["t"][0] - np.linalg.norm([6, 0, -2])) < 1e-6
    assert np.allclose(h["normal"][0], [0, 0, 1])
    h = ray(60.0, 0.0, 0.0)                                                      # beyond the mesh
    assert h["label"][0] == RC.LABEL_PLANE
    h = ray(300.0, 0.0, 0.0)                                                     # beyond the plane's reach
    assert h["label"][0] == RC.LABEL_FAR and h["t"][0] == RC.SKY_RANGE_M
    h = ray(10.0, 0.0, 5.0)                                                      # up
    assert h["label"][0] == RC.LABEL_FAR
    # the plane does not occlude: a mesh point below it is visible from above
    tgt = np.array([[3.0, 3.0, -0.5]])
    assert not sc.occluded(P[None], tgt, np.array([0.05]))[0]
    # cast_image: sky pixels have label far and no depth; plane pixels a depth
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=160, height=120, params=np.array([80.0, 80.0, 60.0]))
    m = RC.cast_image(sc, cam, _look(90.0, 0.0), P, scale=1.0)
    far = m["label"] == RC.LABEL_FAR
    pl = m["label"] == RC.LABEL_PLANE
    assert far[:50].all() and np.all(m["depth"][far] == 0) and np.all(m["range"][far] == 0)
    assert pl.any() and np.all(m["depth"][pl] > 0) and RC.coverage(m)["plane"] > 0
    pan = RC.panorama_range(sc, P, width=72)
    assert np.isinf(pan["range"][0]).all()


def test_horizontal_radius_clip_is_from_the_mast_axis():
    V, F = _terrain(n=41, size=30.0)
    V[:, 2] = -10.0 + 0.3 * V[:, 0]                                             # sloped ground well below the head
    P = np.array([0.0, 0.0, 2.0])
    V2, F2, rep = M.clean_arrays(V, F, P, points=None, radius_m=20.0, up=(0, 0, 1))
    cen, _, _ = M.triangle_geometry(V2, F2)
    h = M.horizontal_distance(cen, P)
    assert h.max() <= 20.0 and h.max() > 18.0                                   # 3-D range would have cut nearer
    assert rep["steps"][-1]["measured"].startswith("from the mast axis")


def test_ods_on_the_plane_and_sky_scene_the_sky_has_no_disparity(tmp_path):
    """With the plane and the sky: every pixel has a surface; the rays above the horizon that a frame sees are
    coloured, and identically in both eyes (at infinity); below the horizon beyond the mesh the plane gives the
    eyes their parallax."""
    import cv2
    proj, rec, st, _, P, colours = _station_scene(tmp_path)
    Vm, Fm = _plane(0.0, 14.0)                                                  # a 'mesh' only to 14 m
    plane = M.fit_ground_plane(np.c_[np.random.default_rng(0).uniform(-90, 90, (20000, 2)) + P[:2], np.zeros(20000)], st.pivot)
    scene = RC.Scene((Vm, Fm), plane=plane, sky=True)
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        R = np.asarray(im.cam_from_world().rotation.matrix())
        C = np.asarray(im.projection_center())
        m = RC.cast_image(scene, cam, R, C, scale=1.0)
        _, dw, _ = RC.pixel_rays(cam, R, 1.0)
        X = C + m["range"][..., None] * dw
        sky = m["label"] == RC.LABEL_FAR
        img = np.where(sky, 200, _world_texture(X)).astype(np.uint8)
        cv2.imwrite(str(proj.root / "images" / im.name), img)
    out = tmp_path / "ods_ps"
    rep = O.render_station(proj, rec, st, scene, out, width=512, ipd_m=0.064, scale=1.0, rows_per_chunk=64)
    cov = rep["eyes"]["left"]["coverage"]
    assert cov["no_surface"] == 0.0 and cov["plane"] > 0.02 and cov["far"] > 0.3 and cov["mesh"] > 0.05
    L = cv2.imread(str(out / "S003D0010_ods_left.png"), cv2.IMREAD_GRAYSCALE)
    Rr = cv2.imread(str(out / "S003D0010_ods_right.png"), cv2.IMREAD_GRAYSCALE)
    v = int((90 - 10.0) / 180 * 256)                                            # 10 deg above the horizon: sky, seen
    row_l, row_r = L[v], Rr[v]
    seen = (row_l > 0) & (row_r > 0)
    assert seen.mean() > 0.8 and np.all(np.abs(row_l[seen].astype(int) - 200) <= 2) and np.abs(row_l[seen].astype(int) - row_r[seen]).max() <= 1
    # the plane band (14-100 m, elevation -7.3 to -1 deg): the eyes differ there (the ground texture's parallax)
    lab = O.render_eye(scene, O.station_sources(proj, rec, st, scale=1.0), st.pivot, "left", 512, 256, 0.064,
                       up=st.up, north=st.north, east=st.east)["label"]
    assert (lab == RC.LABEL_PLANE).any() and (lab == RC.LABEL_MESH).any()


def test_two_view_terrain_is_fused_with_the_station_options(tmp_path):
    """Christian, 10 Nov: "Many parts of the terrain will only be seen by two images, so make sure those get good
    depth coverage."  A Navcam-like stereo pair (0.424 m apart, both looking north, 20 deg down) over textured
    ground: every ground point is seen by exactly two images.  COLMAP's default fusion (5 images a point) fuses
    nothing; the station options (2) fuse the pair's ground out to 20 m and beyond, on the ground to a few mm."""
    import cv2
    from mppp.sfm.dense import STATION_FUSION_OPTIONS, prepare_dense, run_dense
    from mppp.sfm.project import SfmProject
    from mppp.ibr.photometry import write_colmap_array
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=640, height=480, params=np.array([500.0, 320.0, 240.0]))
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    root = tmp_path / "mars2020_sol_0461_pair" / "colmap"
    (root / "images").mkdir(parents=True)
    P = np.array([0.0, 0.0, 2.0])
    R = _look(0.0, 20.0)
    rows = []
    # tie points on the ground seen by both eyes: COLMAP's fusion takes which images overlap from them
    g = np.random.default_rng(2)
    T = np.c_[g.uniform(-6, 6, 400), g.uniform(4, 30, 400), np.zeros(400)]
    centres = {1: P + np.array([-0.212, 0.0, 0.0]), 2: P + np.array([0.212, 0.0, 0.0])}
    kps = {}
    for k, Ck in centres.items():
        Xc = (T - Ck) @ R.T
        kps[k] = np.asarray(cam.img_from_cam(Xc), float)
    vis = np.all([(kp[:, 0] > 0) & (kp[:, 0] < 640) & (kp[:, 1] > 0) & (kp[:, 1] < 480) for kp in kps.values()], axis=0)
    T, kps = T[vis], {k: v[vis] for k, v in kps.items()}
    for k, Ck in centres.items():
        fr = pycolmap.Frame(frame_id=k, rig_id=1)
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=k))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ Ck)
        rec.add_frame(fr)
        name = ("NLF" if k == 1 else "NRF") + "_0461_pair.png"
        im = pycolmap.Image(name=name, keypoints=kps[k], camera_id=1, image_id=k)
        im.frame_id = k
        rec.add_image(im)
        rows.append({"name": name, "site": 26, "drive": 630, "sol": 461, "camera_group": "NL" if k == 1 else "NR",
                     "station": "S026D0630", "sequence": "NCAM00001", "downsample_scale": 1.0})
    for fid in list(rec.frames):
        rec.register_frame(fid)
    for j, X in enumerate(T):
        tr = pycolmap.Track()
        tr.add_element(1, j)
        tr.add_element(2, j)
        rec.add_point3D(X, tr, np.array([128, 128, 128], np.uint8))
    assert rec.num_points3D() > 100
    scene = RC.Scene(_plane(0.0, 500.0))
    for im in rec.images.values():
        Rm = np.asarray(im.cam_from_world().rotation.matrix())
        C = np.asarray(im.projection_center())
        m = RC.cast_image(scene, cam, Rm, C, scale=1.0)
        _, dw, _ = RC.pixel_rays(cam, Rm, 1.0)
        X = C + m["range"][..., None] * dw
        cv2.imwrite(str(root / "images" / im.name), np.where(m["label"] == 1, _world_texture(X), 0).astype(np.uint8))
    model = root / "gui_native" / "sparse"
    model.mkdir(parents=True)
    rec.write(str(model))
    # a 'rover' in the bottom fifth of both frames (mask 0 = rover): the fusion leaves it out unless the rover is kept
    (root / "masks").mkdir()
    for r in rows:
        mk = np.full((480, 640), 255, np.uint8)
        mk[384:] = 0
        cv2.imwrite(str(root / "masks" / (r["name"] + ".png")), mk)
        r["has_mask"] = True
    proj = SfmProject(root, rows, {}, {}, [0.0, 0.0, 0.0], {"world_frame": "site26_enu"})
    proj.save()
    ws = root / "dense_S026D0630"
    prep = prepare_dense(proj, ws, max_image_size=640, masks=True, images=[r["name"] for r in rows])
    assert prep["masks"]["images"] == 2
    und = pycolmap.Reconstruction(str(ws / "sparse"))
    for im in und.images.values():
        c2 = und.cameras[im.camera_id]
        Rm = np.asarray(im.cam_from_world().rotation.matrix())
        m = RC.cast_image(scene, c2, Rm, np.asarray(im.projection_center()), scale=1.0)
        g = (m["label"] == RC.LABEL_MESH) & (m["range"] < 60)
        write_colmap_array(ws / "stereo" / "depth_maps" / f"{im.name}.geometric.bin", np.where(g, m["depth"], 0))
        write_colmap_array(ws / "stereo" / "normal_maps" / f"{im.name}.geometric.bin", m["normal"] * g[..., None])
    out = run_dense(proj, workspace=ws, steps=("fusion",), max_image_size=640)
    assert out["steps"]["fusion"]["points"] < 50                                  # COLMAP's 5 images: the pair is lost
    out = run_dense(proj, workspace=ws, steps=("fusion",), max_image_size=640, fusion_options=STATION_FUSION_OPTIONS)
    assert out["steps"]["fusion"]["options"] == {"min_num_pixels": 2}
    pts = np.asarray(o3d.io.read_point_cloud(str(ws / "fused.ply")).points)
    h = M.horizontal_distance(pts, P)
    assert len(pts) > 20_000 and np.median(np.abs(pts[:, 2])) < 0.01
    assert (h > 20).sum() > 500 and h.max() > 40                                 # the pair's far ground is there
    rings = M.points_by_ring(pts, P)
    assert all(r["points"] > 0 for r in rings if 10 <= r["ring_m"][0] < 30)
    # mppp dense --keep-rover: the same depth maps fused without the masks, into fused_rover.ply
    kept = run_dense(proj, workspace=ws, steps=("fusion",), max_image_size=640, fusion_options=STATION_FUSION_OPTIONS,
                     keep_rover=True)["steps"]["fusion"]
    assert kept["ply"].endswith("fused_rover.ply") and kept["masks"] is None and kept["rover_kept"]
    pr = np.asarray(o3d.io.read_point_cloud(str(ws / "fused_rover.ply")).points)
    hr = M.horizontal_distance(pr, P)
    assert len(pr) > len(pts) and (hr < 3.0).sum() > 3 * max(1, (h < 3.0).sum())   # the masked near field is back
    assert (ws / "fused.ply").is_file() and len(np.asarray(o3d.io.read_point_cloud(str(ws / "fused.ply")).points)) == len(pts)


# ----------------------------------------------------------------------------- 10 Nov: stale patch-match maps, the blend
def test_old_patch_match_maps_are_cleared_when_the_options_change(tmp_path):
    """COLMAP's patch-match skips every image whose maps exist (the PC, 10 Nov: --stereo again took 0.3 s and the
    8 m cut-off stayed).  Maps made with other options - or unrecorded ones when this run sets options - go first."""
    from mppp.sfm.dense import STATION_STEREO_OPTIONS, stereo_outputs_ready
    ws = tmp_path / "ws"
    for d in ("depth_maps", "normal_maps"):
        (ws / "stereo" / d).mkdir(parents=True)
        for k in ("photometric", "geometric"):
            (ws / "stereo" / d / f"a.png.{k}.bin").write_bytes(b"x")
    rec = {"options": dict(STATION_STEREO_OPTIONS), "geom_consistency": True, "max_image_size": 2000}
    r = stereo_outputs_ready(ws, rec)                                        # the 9 Nov maps: no record, defaults
    assert r["cleared"] == 4 and "unrecorded" in r["why"] and not list((ws / "stereo" / "depth_maps").glob("*.bin"))
    (ws / "stereo" / "depth_maps" / "a.png.geometric.bin").write_bytes(b"x")
    r = stereo_outputs_ready(ws, rec)                                        # same options: kept (COLMAP resumes)
    assert r["cleared"] == 0 and r["kept"] == 1
    r = stereo_outputs_ready(ws, dict(rec, options={**rec["options"], "filter_min_triangulation_angle": 1.0}))
    assert r["cleared"] == 1 and "filter_min_triangulation_angle" in r["why"]
    (ws / "stereo" / "depth_maps" / "a.png.geometric.bin").write_bytes(b"x")
    assert stereo_outputs_ready(ws, dict(rec, options={**rec["options"], "filter_min_triangulation_angle": 1.0}), force=True)["cleared"] == 1
    ws2 = tmp_path / "ws2"                                                  # a whole-block run with defaults keeps old maps
    (ws2 / "stereo" / "depth_maps").mkdir(parents=True)
    (ws2 / "stereo" / "depth_maps" / "b.bin").write_bytes(b"x")
    assert stereo_outputs_ready(ws2, {"options": {}, "geom_consistency": True, "max_image_size": 2000})["cleared"] == 0


def _ring_station(tmp_path, n=8, darken=0.7):
    """n frames every 360/n deg (45 deg overlaps at n = 8), a ground texture fixed to the ground; every other frame
    ``darken`` times as bright (another exposure)."""
    import cv2
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=400, height=300, params=np.array([200.0, 200.0, 150.0]))
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    root = tmp_path / "mars2020_sol_0091_ring" / "colmap"
    (root / "images").mkdir(parents=True)
    P = np.array([5.0, 5.0, 1.8])
    rows = []
    for k in range(1, n + 1):
        R = _look((k - 1) * 360.0 / n, 25.0)
        Ck = P + R.T @ np.array([0.0, 0.0, 0.3])
        fr = pycolmap.Frame(frame_id=k, rig_id=1)
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=k))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ Ck)
        rec.add_frame(fr)
        name = f"NLF_{k}.png"
        im = pycolmap.Image(name=name, keypoints=np.zeros((0, 2)), camera_id=1, image_id=k)
        im.frame_id = k
        rec.add_image(im)
        rows.append({"name": name, "site": 3, "drive": 10, "sol": 91, "camera_group": "NL"})
    for fid in list(rec.frames):
        rec.register_frame(fid)
    proj = _Proj(root, rows)
    st = select_station(proj, rec, "3_10", families="N")
    scene = RC.Scene(_plane(0.0, 300.0), plane={"normal": [0, 0, 1], "offset": 0.0, "centre": st.pivot, "up": [0, 0, 1],
                                                 "radius_m": 100.0}, sky=True)
    for im in rec.images.values():
        Rm = np.asarray(im.cam_from_world().rotation.matrix())
        C = np.asarray(im.projection_center())
        m = RC.cast_image(scene, cam, Rm, C, scale=1.0)
        _, dw, _ = RC.pixel_rays(cam, Rm, 1.0)
        X = C + m["range"][..., None] * dw
        tex = 120.0 + 0.3 * (_world_texture(X).astype(float) - 128.0)
        f = darken if im.image_id % 2 == 0 else 1.0
        cv2.imwrite(str(root / "images" / im.name), np.clip(np.where(m["label"] == RC.LABEL_FAR, 200, tex) * f, 0, 255).astype(np.uint8))
    return proj, rec, st, scene


def test_blend_has_no_seams_between_exposures(tmp_path):
    """Christian, 10 Nov: "the mosaic should be blended together to avoid jagged borders".  Six frames, every other
    one 30 % darker: the gain pass finds the ratio, and around the horizon band the brightness has no steps; without
    the gains the frames' levels show; the detail band keeps the texture sharp (not averaged away)."""
    import cv2
    proj, rec, st, scene = _ring_station(tmp_path, n=6)                     # 30 deg overlaps, the frames' middles alone
    out_g, out_n = tmp_path / "with_gains", tmp_path / "no_gains"
    rg = O.render_station(proj, rec, st, scene, out_g, width=720, ipd_m=0.064, scale=1.0, rows_per_chunk=60, eyes=("left",))
    O.render_station(proj, rec, st, scene, out_n, width=720, ipd_m=0.064, scale=1.0, rows_per_chunk=60, eyes=("left",), gains=False)
    g = rg["gains"]["gains"]
    ratio = np.median([g[f"NLF_{k}.png"] for k in (2, 4, 6)]) / np.median([g[f"NLF_{k}.png"] for k in (1, 3, 5)])
    assert rg["gains"]["pairs"] >= 6 and abs(ratio - 1 / 0.7) < 0.05
    def band_profile(f):
        L = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(float)
        v0, v1 = int((90 + 30) / 180 * 360), int((90 + 40) / 180 * 360)          # 30-40 deg below the horizon
        prof = L[v0:v1].mean(axis=0)
        k = 9                                                                   # average out the 10 cm cells
        return np.convolve(prof, np.ones(k) / k, mode="valid"), L[v0:v1]
    pg, Lg = band_profile(out_g / "S003D0010_ods_left.png")
    pn, _ = band_profile(out_n / "S003D0010_ods_left.png")
    # the truth: the same frames all at one exposure; the blends against it, column by column (the texture cancels)
    proj1, rec1, st1, scene1 = _ring_station(tmp_path / "same", n=6, darken=1.0)
    O.render_station(proj1, rec1, st1, scene1, tmp_path / "truth", width=720, ipd_m=0.064, scale=1.0, rows_per_chunk=60,
                     eyes=("left",), gains=False)
    pt, _ = band_profile(tmp_path / "truth" / "S003D0010_ods_left.png")
    qg, qn = pg / pt, pn / pt
    assert qg.min() / qg.max() > 0.95                                           # one level all round
    assert qn.min() / qn.max() < 0.8                                            # the exposures show without gains
    assert np.abs(np.diff(qg)).max() < 0.01                                     # no step at a seam
    assert Lg.std() > 10                                                        # the texture is still there


def test_edge_distance_counts_from_the_mask_and_the_border(tmp_path):
    import cv2
    img = np.full((60, 80, 3), 100, np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    m = np.full((60, 80), 255, np.uint8)
    m[40:] = 0                                                                   # the rover in the bottom third
    cv2.imwrite(str(tmp_path / "a.png.mask.png"), m)
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=80, height=60, params=np.array([50.0, 40.0, 30.0]))
    s_ = O.SourceImage("a.png", cam, np.eye(3), np.zeros(3), tmp_path / "a.png", tmp_path / "a.png.mask.png", 1.0)
    d = s_.edge_distance()
    assert d[45, 40] == 0 and d[0, 40] <= 1.0 and 9 <= d[30, 40] <= 11 and d[20, 40] > d[35, 40]
    nm = O.SourceImage("a.png", cam, np.eye(3), np.zeros(3), tmp_path / "a.png", None, 1.0).edge_distance()
    assert nm[45, 40] > 10                                                       # the rover shown: only the border counts


# ----------------------------------------------------------------------------- 10 Nov: zones, normals, cracks
def test_zoned_poisson_is_finer_near_the_mast_and_uses_the_given_normals():
    """Christian, 10 Nov: "The near and mid terrain needs a little more detail and higher polygon count".  Each zone
    has its own octree box, so the near field's triangles are several times smaller than the outer zone's; the
    fused normals are used as given (COLMAP orients them), else estimated and turned to the mast head."""
    rng = np.random.default_rng(7)
    P = np.array([0.0, 0.0, 2.0])
    r = np.sqrt(rng.uniform(0, 20.5 ** 2, 400_000))
    a = rng.uniform(0, 2 * np.pi, len(r))
    xy = np.c_[r * np.cos(a), r * np.sin(a)]
    keep = rng.random(len(r)) < np.clip(4.0 / np.maximum(r, 1.0), 0.05, 1.0)       # dense near, sparse far (a station)
    xy = xy[keep]
    z = 0.05 * np.sin(3 * xy[:, 0]) * np.cos(3 * xy[:, 1])
    pts = np.c_[xy, z]
    nrm = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    V, F, how, crep = M.zoned_poisson(pts, nrm, P, (0, 0, 1), 20.0, zones=((0, 4, 8), (4, 12, 8), (12, None, 8)))
    assert [z_["zone_m"] for z_ in how["zones"]] == [[0, 4.0], [4, 12.0], [12, 20.0]]
    assert all(z_["normals"] == "given" and z_["triangles"] > 100 for z_ in how["zones"])
    cen, _, longest = M.triangle_geometry(V, F)
    h = M.horizontal_distance(cen, P)
    assert h.max() <= 20.0 + 1e-6 and h.min() < 1.5
    assert np.median(longest[h < 3.5]) * 3 < np.median(longest[h > 13])            # the near field is finer
    for ring in (4.0, 12.0):                                                         # no gap at the zone edges
        band = (h > ring - 0.15) & (h < ring + 0.15)
        az = np.degrees(np.arctan2(cen[band, 1], cen[band, 0])) % 360
        assert len(np.unique((az // 10).astype(int))) == 36
    _, _, how2 = M.poisson_arrays(pts[:5000], None, depth=6, viewpoint=P)
    assert how2["normals"] == "estimated, towards the mast head"


def test_a_point_hidden_from_every_source_takes_the_best_hidden_view(tmp_path):
    """The black cracks of the Van Zyl run: the eye's ray through a gap in the mesh ends on the ground, which the mesh
    hides from the camera 30 cm ahead.  Now the hidden view colours it (marked), instead of black."""
    proj, rec, st, scene0, P, colours = _station_scene(tmp_path)
    Vg, Fg = _plane(0.0, 300.0)
    # a plate at z = 1 m north of the station with a 6 cm hole where the eye's ray (north, 45 deg down) crosses it
    Q = np.asarray(st.pivot, float)
    xs = np.linspace(-1.0, 1.0, 81) + Q[0]
    ys = np.linspace(0.3, 1.6, 53) + Q[1]
    X, Y = np.meshgrid(xs, ys)
    Vp = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
    Fp = []
    nx = len(xs)
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            q = j * nx + i
            cx, cy = (xs[i] + xs[i + 1]) / 2 - Q[0], (ys[j] + ys[j + 1]) / 2 - Q[1]
            if np.hypot(cx, cy - (Q[2] - 1.0)) < 0.06:
                continue
            Fp += [[q, q + 1, q + nx], [q + 1, q + nx + 1, q + nx]]
    V = np.vstack([Vg, Vp])
    F = np.vstack([Fg, np.asarray(Fp) + len(Vg)])
    scene = RC.Scene((V, F))
    sources = O.station_sources(proj, rec, st, scale=1.0)
    d = np.array([[0.0, np.cos(np.radians(45)), -np.sin(np.radians(45))]])
    c = O.candidates(scene, sources, st.pivot[None], d)
    assert c["label"][0] == RC.LABEL_MESH and abs(c["t"][0] - Q[2] / np.sin(np.radians(45))) < 0.1   # through the hole
    assert c["hidden_fallback"][0] and c["k"][0, 0] == 0                          # the north frame, hidden: used


def test_thin_cracks_are_filled_and_large_holes_stay_black():
    rgb = np.full((200, 400, 3), 120, np.uint8)
    rgb[:, :200] = 60
    source = np.zeros((200, 400), np.int16)
    label = np.ones((200, 400), np.uint8)
    source[100, 20:380] = -1                                                       # a crack, 1 px wide
    rgb[100, 20:380] = 0
    source[20:80, 250:390] = -1                                                    # a large hole
    rgb[20:80, 250:390] = 0
    label[150:190, 10:60] = 0                                                      # no surface at all: not ours
    source[150:190, 10:60] = -1
    rgb[150:190, 10:60] = 0
    filled = O.fill_cracks(rgb, source, label)
    assert filled[100, 20:380].all() and np.all(source[100, 20:380] == -2)
    assert abs(int(rgb[100, 50, 0]) - 60) < 8 and abs(int(rgb[100, 300, 0]) - 120) < 8
    assert not filled[20:80, 250:390].any() and np.all(rgb[50, 300] == 0)
    assert not filled[150:190, 10:60].any()


# ----------------------------------------------------------------------------- 10 Nov: padding, sky fill, mast-head hardware
def test_black_padding_is_not_image_content(tmp_path):
    import cv2
    img = np.full((100, 160, 3), 90, np.uint8)
    img[:25] = 0                                                                    # padding along the top
    img[60:63, 80:83] = 0                                                           # a real black speck inside
    cv2.imwrite(str(tmp_path / "a.png"), img)
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=160, height=100, params=np.array([80.0, 80.0, 50.0]))
    s_ = O.SourceImage("a.png", cam, np.eye(3), np.zeros(3), tmp_path / "a.png", None, 1.0)
    pad = s_.padding()
    assert pad[:25].all() and not pad[25:].any()
    d = s_.edge_distance()
    assert d[10, 80] == 0 and 9 <= d[35, 80] <= 11                                  # feathers count from the padding


def test_sky_fill_carries_the_sky_up_without_black_or_sawtooth():
    """Christian, 10 Nov: "extrapolate and blur the light blue sky upwards so that there isn't a shape boundary
    there" - "make sure not to extrapolate black pixel areas"."""
    from mppp.ibr.raycast import LABEL_FAR, LABEL_MESH
    H, W = 400, 800
    rgb = np.zeros((H, W, 3), np.uint8)
    source = np.full((H, W), -1, np.int16)
    label = np.full((H, W), LABEL_FAR, np.uint8)
    label[200:] = LABEL_MESH
    cols = np.arange(W)
    top = 120 + (25 * np.abs(((cols % 100) / 100.0) - 0.5) * 2).astype(int)        # the frames' sawtooth tops
    for c in cols:
        rgb[top[c]:200, c] = (180, 170, 200)
        source[top[c]:200, c] = 0
        rgb[top[c]:top[c] + 4, c] = (5, 5, 5)                                       # the dark top edge of the frames
    rgb[200:] = (110, 80, 60)
    source[200:] = 0
    changed = O.fill_sky(rgb, source, label)
    up = rgb[:150].astype(int)
    assert changed[:100].all() and up.min() > 120                                  # no black carried up
    assert np.abs(up - np.array([180, 170, 200])).max() < 12                       # the sky's colour
    assert np.abs(np.diff(rgb[100].astype(int), axis=0)).max() <= 2                # no sawtooth left in a row
    assert np.all(rgb[300] == (110, 80, 60))                                       # below the horizon: untouched


def test_mast_head_hardware_is_what_stays_put_across_the_tier():
    """A boom fixed to the mast head sits at the same pixels in every frame of one eye and elevation; the deck moves
    across the frames as the mast turns.  Only the boom is masked with the rover shown."""
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=100, height=80, params=np.array([60.0, 50.0, 40.0]))
    srcs, masks = [], []
    for k, yaw in enumerate((0, 60, 120, 180, 240)):
        s_ = O.SourceImage(f"NLF_{k}.png", cam, _look(yaw, 30.0), np.zeros(3), Path("x.png"), None, 1.0)
        m = np.ones((80, 100), bool)
        m[60:70, 70:90] = False                                                     # the boom: always there
        m[40:80, 10 * k:10 * k + 25] = False                                        # the deck: moves
        srcs.append(s_)
        masks.append(m)
    hw = O.head_hardware_masks(srcs, masks, (0, 0, 1))
    for h in hw:
        assert h is not None and h[60:70, 70:90].all() and h.sum() == 200            # the boom, nothing of the deck
    few = O.head_hardware_masks(srcs[:2], masks[:2], (0, 0, 1))
    assert all(h is None for h in few)                                               # two frames: no verdict


# ----------------------------------------------------------------------------- 10 Nov: the single sites from a config file
def test_single_sites_config_plan_run_and_installed_bat(tmp_path, monkeypatch):
    """Christian, 10 Nov: "the single sites should come from a config file" and one update_and_run_single_sites.bat.
    The shipped config has the three test stations; the plan resolves each block from sites.json and builds the
    commands of both variants; a failed dense step stops that station; the summary has a line per variant; the
    installed batch file names no version."""
    from mppp.ibr import single_sites as SS
    cfg = SS.load_config()
    assert set(cfg["stations"]) == {"S003D2046", "S026D0630", "S032D1184"}
    assert len(cfg["stations"]["S003D2046"]["frames"]["images"]) == 12
    root = tmp_path / "rb"
    for f in ("mars2020_sol_0015_van_zyl_overlook_huge", "mars2020_sol_0460_rockytop", "mars2020_sol_0413_threeforks_large"):
        (root / f).mkdir(parents=True)
    runs = SS.plan(cfg, root)
    assert [Path(r["work"]).name for r in runs] == ["mars2020_sol_0015_van_zyl_overlook_huge", "mars2020_sol_0460_rockytop",
                                                    "mars2020_sol_0413_threeforks_large"]
    r0 = runs[0]
    steps = [(c["variant"], c["step"]) for c in r0["commands"]]
    assert steps == [("masked", "dense"), ("rover", "mesh"), ("rover", "depthmaps"), ("rover", "ods"),
                     ("masked", "mesh"), ("masked", "depthmaps"), ("masked", "ods")]          # 0.97.0: one fusion, the terrain's
    dense = r0["commands"][0]["argv"]
    assert dense[:4] == ["dense", r0["work"], "--station", "3_2046"] and dense.count("--images") == 1 and "--stereo" in dense
    rover_ods = r0["commands"][3]["argv"]
    assert "--mask-rover" in r0["commands"][-1]["argv"]
    assert "--rover" in rover_ods and rover_ods[rover_ods.index("--out") + 1].endswith(str(Path("S003D2046") / "with_rover"))
    assert rover_ods[-6:] == ["--bits", "16", "--eyes", "left", "right", "centre"]         # 0.97.0: 16 bits, the centre eye
    tl = str(root / "single_sites" / "S003D2046" / "tone_levels.json")                     # 0.97.0: one scaling per station
    assert rover_ods[rover_ods.index("--tone-levels") + 1] == tl
    assert r0["commands"][-1]["argv"][r0["commands"][-1]["argv"].index("--tone-levels") + 1] == tl
    assert SS.plan(cfg, root, labels=["S026D0630"], steps=["ods"], variants=["masked"])[0]["commands"] == [
        {"variant": "masked", "step": "ods", "argv": ["ods", str(root / "mars2020_sol_0460_rockytop"), "--station", "26_630",
                                                      "--mask-rover", "--out", str(root / "single_sites" / "S026D0630"),
                                                      "--exclude", "NLF_0461_0707875039_745RAD_N0260630NCAM13461_0A0195J01",
                                                      "--tone-levels", str(root / "single_sites" / "S026D0630" / "tone_levels.json"),
                                                      "--bits", "16", "--eyes", "left", "right", "centre"]}]
    with pytest.raises(KeyError):
        SS.plan(cfg, root, labels=["S999D9999"])
    # the run: the steps stood in for; Rockytop's dense step fails, so nothing else of it runs
    calls = []

    def fake_run(argv, fh):
        calls.append(argv)
        if argv[0] == "dense" and "26_630" in argv:
            return 1
        if argv[0] == "ods":
            out = Path(argv[argv.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "ods.json").write_text(json.dumps({"eyes": {"left": {"coverage": {"mesh": 0.2, "plane": 0.3, "far": 0.5,
                                                                                      "coloured": 0.9}}}}), encoding="utf-8")
            (out / "S003D0010_360_3DTD.jpg").write_bytes(b"x")
        return 0
    monkeypatch.setattr(SS, "_run", fake_run)
    rc = SS.run_all(runs, root / "single_sites")
    assert rc == 1                                                                     # one station failed
    rock = [c for c in calls if "26_630" in c]
    assert rock == [runs[1]["commands"][0]["argv"]]                                    # its dense step only
    assert sum(1 for c in calls if "3_2046" in c) == 7 and sum(1 for c in calls if "32_1184" in c) == 7   # 0.97.0: one fusion
    summary = (root / "single_sites" / "single_sites_summary.md").read_text(encoding="utf-8")
    assert summary.count("| S003D2046 |") == 2 and "| S026D0630 | masked | dense 1 |" in summary
    assert "mesh 20 %, plane 30 %, far 50 %" in summary
    assert (root / "single_sites" / "S003D2046" / "single_site_log.txt").exists()
    f = SS.install(tmp_path / "inst")
    b = f.read_bytes()
    assert f.name == "run_single_site_ibr.bat" and b"\r\n" in b and b"python -m mppp single-sites %*" in b
    assert b"7dbd5da" not in b and b"_transfer" in b and b"--ff-only" in b and b"merge-base --is-ancestor" in b


def test_ods_first_column_azimuth():
    """Christian, 10 Nov: "provide an option to rotate the sphere so that the first column is North" - north is the
    first column by default; --azimuth0 / --north-center turn the panorama."""
    P = np.zeros(3)
    _, d0 = O.ods_rays(360, 180, (90, 91), P, 0.064, "centre")
    assert np.allclose(d0[0], [0, 1, 0], atol=0.02) and np.allclose(d0[90], [1, 0, 0], atol=0.02)       # N, then E
    _, d1 = O.ods_rays(360, 180, (90, 91), P, 0.064, "centre", azimuth0_deg=90.0)
    assert np.allclose(d1[0], [1, 0, 0], atol=0.02)                                                    # east first
    _, d2 = O.ods_rays(360, 180, (90, 91), P, 0.064, "centre", azimuth0_deg=-180.0)
    assert np.allclose(d2[180], [0, 1, 0], atol=0.02) and np.allclose(d2[0], [0, -1, 0], atol=0.02)    # north centred


def test_full_fov_undistortion_keeps_the_fisheye_corners(tmp_path):
    """Christian, 10 Nov: "the undistorted images throw away the edges of the fov leading to less deck coverage".
    The default undistortion crops a Navcam fisheye to the rectangle of valid pixels (the corners go); --full-fov
    keeps them, at a size scaled so the centre keeps the default's focal length; a change of the setting redoes
    the workspace, and the stereo record carries it (old maps cleared)."""
    import cv2
    from mppp.sfm.dense import prepare_dense, stereo_outputs_ready
    from mppp.sfm.project import SfmProject
    params = [v / 4 if i < 4 else v for i, v in enumerate(NL)]                       # a quarter-size Navcam: 1280 x 960
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(camera_id=1, model="THIN_PRISM_FISHEYE", width=1280, height=960, params=np.asarray(params, float))
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    root = tmp_path / "mars2020_sol_0052_fov" / "colmap"
    (root / "images").mkdir(parents=True)
    fr = pycolmap.Frame(frame_id=1, rig_id=1)
    fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=1))
    fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(_look(0.0, 30.0)), np.zeros(3))
    rec.add_frame(fr)
    im = pycolmap.Image(name="NLF_1.png", keypoints=np.zeros((0, 2)), camera_id=1, image_id=1)
    im.frame_id = 1
    rec.add_image(im)
    rec.register_frame(1)
    cv2.imwrite(str(root / "images" / "NLF_1.png"), np.full((960, 1280, 3), 120, np.uint8))
    (root / "gui_native" / "sparse").mkdir(parents=True)
    rec.write(str(root / "gui_native" / "sparse"))
    proj = SfmProject(root, [{"name": "NLF_1.png", "site": 3, "drive": 2046, "sol": 52}], {}, {}, [0, 0, 0], {})
    proj.save()
    ws = root / "dense_test"
    r0 = prepare_dense(proj, ws, max_image_size=1000, masks=False)
    c0 = list(pycolmap.Reconstruction(str(ws / "sparse")).cameras.values())[0]
    r1 = prepare_dense(proj, ws, max_image_size=1000, masks=False, full_fov=True)
    c1 = list(pycolmap.Reconstruction(str(ws / "sparse")).cameras.values())[0]
    assert not r0["full_fov"] and r1["full_fov"] and r1["max_image_size_used"] > 1000
    assert abs(c1.params[0] - c0.params[0]) / c0.params[0] < 0.01                    # the centre keeps its focal length

    def corner_deg(c):
        xy = np.asarray(c.cam_from_img(np.array([[0.5, 0.5]])))[0]
        return np.degrees(np.arctan(np.hypot(*xy)))
    native = corner_deg(cam)
    assert corner_deg(c0) < native - 3.0 and corner_deg(c1) > native - 0.5               # the corners: cut / kept
    assert prepare_dense(proj, ws, max_image_size=1000, masks=False, full_fov=True)["max_image_size_used"] == r1["max_image_size_used"]
    (ws / "stereo" / "depth_maps").mkdir(parents=True, exist_ok=True)
    (ws / "stereo" / "depth_maps" / "NLF_1.png.geometric.bin").write_bytes(b"x")
    stereo_outputs_ready(ws, {"options": {}, "geom_consistency": True, "max_image_size": 1000, "full_fov": False})
    (ws / "stereo" / "depth_maps" / "NLF_1.png.geometric.bin").write_bytes(b"x")
    r = stereo_outputs_ready(ws, {"options": {}, "geom_consistency": True, "max_image_size": r1["max_image_size_used"],
                                  "full_fov": True})
    assert r["cleared"] == 1 and "full_fov" in r["why"]


# ----------------------------------------------------------------------------- 10 Nov: the rover deck floor, left-out pairs, prefixes
def test_rover_deck_pose_floor_and_clamp():
    from scipy.spatial.transform import Rotation as Rot
    from mppp.ibr import rover_deck as RD
    rng = np.random.default_rng(3)
    R_true = Rot.from_euler("zyx", [73.0, 1.5, -2.0], degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])  # z down -> up
    t_true = np.array([30.4, 7.7, 0.8])
    Rcb, Cb, Rcr, Cr = [], [], [], []
    for k in range(8):                                         # mast frames: camera in RNAV, the same camera in the block
        R_cr = Rot.from_euler("xyz", rng.normal(0, 40, 3), degrees=True).as_matrix()
        C_r = np.array([0.8, 0.5, -1.95]) + rng.normal(0, 0.1, 3)
        Rcr.append(R_cr); Cr.append(C_r)
        Rcb.append(R_cr @ R_true.T); Cb.append(R_true @ C_r + t_true)
    pose = RD.pose_from_pairs(Rcb, Cb, Rcr, Cr)
    assert np.allclose(pose["R_block_from_rnav"], R_true, atol=1e-9) and np.allclose(pose["t_block_from_rnav"], t_true)
    assert pose["rot_spread_deg"] < 1e-6 and pose["n"] == 8
    up = -R_true[:, 2]
    # the deck top at RMECH z = 0 is 1.133 m above the RNAV origin
    top = RD.mech_to_block(np.array([[0.0, 0.0, 0.0]]), pose)[0]
    assert abs((top - t_true) @ up - 1.13338) < 1e-6
    # points: on the deck, 0.2 m below it (dropped), above it, and below it but outside the footprint (kept)
    M = np.array([[0.0, 0.0, -0.01], [0.2, 0.1, 0.2], [-0.5, 0.3, -0.3], [1.3, 0.0, 0.4], [0.0, 0.9, 0.6]])
    X = RD.mech_to_block(M, pose)
    assert RD.below_deck(X, pose).tolist() == [False, True, False, False, False]
    V2, n = RD.clamp_to_deck(X, pose)
    assert n == 1 and np.allclose(RD.block_to_mech(V2, pose)[1], [0.2, 0.1, 0.0]) and np.allclose(V2[[0, 2, 3, 4]], X[[0, 2, 3, 4]])
    Vd, Fd = RD.deck_floor_arrays(pose)
    Md = RD.block_to_mech(Vd, pose)
    assert np.allclose(Md[:, 2], 0) and Md[:, 0].min() == pytest.approx(-1.10) and Md[:, 1].max() == pytest.approx(0.60)
    nrm = np.cross(Vd[Fd[:, 1]] - Vd[Fd[:, 0]], Vd[Fd[:, 2]] - Vd[Fd[:, 0]])
    assert (nrm @ up > 0).all()                                # facing up
    area = 0.5 * np.linalg.norm(nrm, axis=1).sum()
    assert area == pytest.approx(2.05 * 1.20, rel=1e-6)
    rec = RD.deck_record(pose, up)
    assert rec["size_m"] == pytest.approx([2.05, 1.20]) and rec["rover_tilt_deg"] < 1e-4


def test_mesh_with_the_deck_floor(tmp_path):
    from mppp.ibr import rover_deck as RD
    R = np.diag([1.0, -1.0, -1.0])                              # RNAV x = block x, z down
    pose = {"R_block_from_rnav": R, "t_block_from_rnav": np.zeros(3)}
    # terrain at z = 0 around the rover, a partial deck at its top (z = 1.133 m), and a sag below it
    g = np.random.default_rng(0)
    terr = np.c_[g.uniform(-6, 6, (40000, 2)), np.zeros(40000)]
    deck = RD.mech_to_block(np.c_[g.uniform(-1.0, 0.0, (4000, 1)), g.uniform(-0.5, 0.5, (4000, 1)), np.zeros(4000)], pose)
    sag = RD.mech_to_block(np.c_[g.uniform(0.2, 0.8, (500, 1)), g.uniform(-0.3, 0.3, (500, 1)), np.full(500, 0.5)], pose)
    pts = np.vstack([terr, deck, sag])
    pivot = RD.mech_to_block(np.array([[0.72, 0.56, -0.78]]), pose)[0]
    rep = M.build_station_mesh(pts, pivot, tmp_path, radius_m=6.0, plane_radius_m=10.0, poisson_depth=7, deck=pose,
                               rover_model_parts=None)
    d = rep["rover_deck"]
    assert d["points_below_dropped"] >= 500 and d["floor_triangles"] > 1000
    V, F = M.read_mesh(tmp_path / "mesh.ply")
    Mv = RD.block_to_mech(V[np.unique(F)], pose)                # the vertices the faces use
    inside = RD.in_footprint(Mv)
    assert (Mv[inside, 2] <= 1e-6).all()                        # nothing below the deck top inside the footprint


def test_exclude_pairs_and_prefix(tmp_path):
    from types import SimpleNamespace
    from mppp.ibr.cli import _frames, _pair_key
    from mppp.ibr.ods import ods_file, ods_prefix, find_prefix, stack_file
    a, b = "NLF_0461_0707875039_745RAD_N0260630NCAM13461_0A0195J01", "NRF_0461_0707875039_745RAD_N0260630NCAM13461_0A0195J01"
    assert _pair_key(a) == _pair_key(b + ".png") and _pair_key(a) != _pair_key(a.replace("0707875039", "0707875040"))
    st = SimpleNamespace(label="S026D0630", images=[a + ".png", b + ".png", "NLF_0461_1_x.png", "NRF_0461_1_x.png"])
    proj = SimpleNamespace(root=tmp_path)
    got = _frames(proj, st, SimpleNamespace(images=None, sols=None, exclude=[a]))
    assert got == ["NLF_0461_1_x.png", "NRF_0461_1_x.png"]
    assert ods_prefix("S003D2046") == "S003D2046" and ods_prefix("S003D2046", rover=True) == "S003D2046_rover"
    assert ods_file(tmp_path, "left", "S003D2046").name == "S003D2046_ods_left.png"
    assert stack_file(tmp_path, "", ".jpg").name == "360_3DTD.jpg"
    (tmp_path / "S003D2046_rover_ods_left.png").write_bytes(b"x")
    assert find_prefix(tmp_path) == "S003D2046_rover"


def test_squares_left_on_top_named_by_station(tmp_path):
    import cv2
    from mppp.ibr.ods import stack_eyes
    from mppp.ibr import single_sites as SS
    L = np.full((8, 16, 3), 40, np.uint8)
    R = np.full((8, 16, 3), 200, np.uint8)
    cv2.imwrite(str(tmp_path / "ods_left.png"), L)              # eye images named before 10 Nov
    cv2.imwrite(str(tmp_path / "ods_right.png"), R)
    r = stack_eyes(tmp_path, square_prefix="S003D2046")
    sq = cv2.imread(str(tmp_path / "S003D2046_360_3DTD.png"))
    assert r["top"] == "left" and sq.shape == (16, 16, 3) and np.array_equal(sq[:8], L) and np.array_equal(sq[8:], R)
    f = SS.install(tmp_path / "bats")
    b = (tmp_path / "bats" / "make_square.bat").read_text()
    assert f.name == "run_single_site_ibr.bat" and "single-sites --squares-only %*" in b and "+main:" in b
    assert "+main:refs/remotes/bundle/main" in f.read_text()


def test_floating_surface_removed_and_deep_faces_cut(tmp_path):
    from mppp.ibr import rover_deck as RD
    P = np.array([0.0, 0.0, 1.9])
    plane = {"normal": [0.0, 0.0, 1.0], "offset": 0.0}
    X = np.array([[1.0, 0, 2.5],        # above the cameras, near: goes
                  [1.0, 0, 1.5],        # within 3 m, below the cameras: the rover, stays
                  [5.0, 0, 1.2],        # beyond 3 m, 0.7 m below the cameras: goes
                  [5.0, 0, 0.5],        # beyond 3 m, a rock 1.4 m below: stays
                  [20.0, 0, 0.0]])
    assert M.floating(X, P, (0, 0, 1), plane).tolist() == [True, False, True, False, False]
    tilt = np.radians(8.0)              # a slope: the ground 14 m uphill is at the cameras' level, and stays
    n = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
    up_hill = np.array([[14.0, 0.0, 14.0 * np.tan(tilt)], [14.0, 0.0, 14.0 * np.tan(tilt) + 1.2]])
    assert M.floating(up_hill, P, (0, 0, 1), {"normal": n.tolist(), "offset": 0.0}).tolist() == [False, True]
    # a mesh: level ground and a floating sheet 1.5 m up at 6-8 m
    g = np.random.default_rng(1)
    terr = np.c_[g.uniform(-12, 12, (60000, 2)), np.zeros(60000)]
    sheet = np.c_[g.uniform(6, 8, (3000, 1)), g.uniform(-1, 1, (3000, 1)), np.full(3000, 1.5)]
    rep = M.build_station_mesh(np.vstack([terr, sheet]), P, tmp_path / "a", radius_m=10.0, plane_radius_m=20.0, poisson_depth=7)
    V, F = M.read_mesh(tmp_path / "a" / "mesh.ply")
    assert rep["floating"]["points_dropped"] == 3000 and V[np.unique(F), 2].max() < 0.9
    rep2 = M.build_station_mesh(np.vstack([terr, sheet]), P, tmp_path / "b", radius_m=10.0, plane_radius_m=20.0, poisson_depth=7,
                                remove_floating=False)
    V2, F2 = M.read_mesh(tmp_path / "b" / "mesh.ply")
    assert rep2["floating"] is None and V2[np.unique(F2), 2].max() > 1.3
    # the deck: a shallow vertex is lifted, a face with a deep vertex is dropped
    pose = {"R_block_from_rnav": np.diag([1.0, -1.0, -1.0]), "t_block_from_rnav": np.zeros(3)}
    Vm = RD.mech_to_block(np.array([[0.0, 0.0, 0.05], [0.1, 0.0, -0.01], [0.0, 0.1, 0.0], [0.2, 0.2, 0.8]]), pose)
    Vc, Fc, lifted, cut = RD.clamp_to_deck(Vm, pose, np.array([[0, 1, 2], [1, 2, 3]]))
    assert (lifted, cut) == (1, 1) and Fc.tolist() == [[0, 1, 2]] and abs(RD.block_to_mech(Vc, pose)[0, 2]) < 1e-12


# ----------------------------------------------------------------------------- 10 Nov: the 16-bit linear panorama
def test_linear16_of_a_real_product_and_the_archive_lookup(tmp_path):
    import cv2
    import shutil
    from types import SimpleNamespace
    import mppp
    from conftest import NLF
    from mppp.ibr import linear16 as L16
    cfg = mppp.load_config()
    i16, sat, facts = L16.linear16_of_product(NLF, cfg)
    from mppp.image import MPPPImage
    c8 = mppp.load_config({"masking": {"infer_mask": False}, "camera_model": {"extrinsics_from_waypoints": False},
                           "selection": {"max_boresight_elevation_deg": None, "max_exposure_ms": None, "max_centre_tint": None,
                                         "max_saturated_fraction": None, "lmst_window_h": None}})
    i8 = MPPPImage(NLF, c8, None).image_int8
    assert i16.dtype == np.uint16 and i16.shape == i8.shape
    free = (i8 < 255).all(axis=2)
    assert np.abs(i16[free] / 64.0 - i8[free]).max() <= 0.51                   # the 8-bit image is the 16-bit one / 64 ...
    assert 0.005 < facts["clipped_in_8bit_share"] < 0.05 and facts["clipped_in_16bit_share"] == 0     # ... clipped at 255
    assert i16.max() > 255 * 64 and facts["white_balance"] == [1.1, 1.4, 1.8] and facts["saturated_share"] == sat.mean()
    # a plateau is saturation; a smooth highlight is not
    g = np.random.default_rng(0)
    raw = g.uniform(0.1, 0.8, (200, 300, 3))
    assert not L16.saturation_mask(raw).any()
    raw[50:90, 100:160, 1] = 1.0
    m = L16.saturation_mask(raw)
    assert m[50:90, 100:160].all() and m.sum() == 40 * 60
    # the block: its 8-bit image at half the size, the product in <archive>/00461/ids/rdr/ncam/
    root = tmp_path / "blk" / "colmap"
    (root / "images").mkdir(parents=True)
    (tmp_path / "blk" / "processed").mkdir()
    (tmp_path / "blk" / "processed" / "mppp_config_v0p95.json").write_text(json.dumps(mppp.load_config()), encoding="utf-8")
    name = NLF.stem + ".png"
    cv2.imwrite(str(root / "images" / name), i8[: i8.shape[0] // 2, : i8.shape[1] // 2])
    arch = tmp_path / "pds" / "00461" / "ids" / "rdr" / "ncam"
    arch.mkdir(parents=True)
    shutil.copy2(NLF, arch / NLF.name)
    proj = SimpleNamespace(root=root, settings={"processed_dir": str(tmp_path / "blk" / "processed")})
    rep = L16.ensure_linear16(proj, [name, "NLF_0461_0000000000_000RAD_N0260630NCAM00000_0A0195J01.png"], pds=tmp_path / "pds")
    assert rep["have"] == [name] and rep["made"] == 1 and len(rep["missing"]) == 1
    f, sf = L16.files(proj, name)
    back = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16 and back.shape[:2] == (i8.shape[0] // 2, i8.shape[1] // 2) and sf.is_file()
    assert json.loads((root / "images16" / "linear16.json").read_text())[name]["cropped"]       # a half-resolution frame
    assert L16.ensure_linear16(proj, [name], pds=tmp_path / "nowhere")["made"] == 0          # kept, not made again
    # the view's tone curve: unchanged below the knee, monotone, the 16-bit ceiling at 255, the hue kept
    x = np.array([[100.0, 50.0, 25.0], [190.0, 95.0, 0.0], [600.0, 300.0, 150.0], [1023.98, 1023.98, 1023.98]], np.float32)
    y = L16.tone_soft(x)
    assert np.allclose(y[:2], x[:2]) and 190 < y[2, 0] < 255 and abs(y[2, 1] / y[2, 0] - 0.5) < 1e-4 and y[3, 0] > 254.9


def test_ods_16bit_keeps_what_8bit_clips(tmp_path):
    import cv2
    proj, rec, st, scene, P, colours = _station_scene(tmp_path)
    d16 = Path(proj.root) / "images16"
    d16.mkdir()
    for n in st.images:                                           # the frames 3x brighter than the 8-bit range holds
        img8 = cv2.imread(str(Path(proj.root) / "images" / n), cv2.IMREAD_COLOR)
        cv2.imwrite(str(d16 / n), (img8.astype(np.uint16) * 64 * 3))
        sat = np.zeros(img8.shape[:2], np.uint8)
        if n == st.images[0]:
            sat[:, :200] = 255                                    # the left half of the red frame "saturated"
        cv2.imwrite(str(d16 / (Path(n).stem + ".sat.png")), sat)
    out = tmp_path / "o16"
    r = O.render_station(proj, rec, st, scene, out, width=256, ipd_m=0.064, scale=0.5, rows_per_chunk=32, bits=16, gains=False)
    assert r["bits"] == 16 and r["linear16"]["rendered"] and not r["linear16"]["missing"]
    L16 = cv2.imread(str(out / "S003D0010_ods_left_16.png"), cv2.IMREAD_UNCHANGED)[..., ::-1]
    L8 = cv2.imread(str(out / "S003D0010_ods_left.png"))[..., ::-1]
    assert L16.dtype == np.uint16 and L16.shape == (128, 256, 3) and L8.dtype == np.uint8
    east = L16[70, 64]                                            # az 90: the green frame, 255 x 3 x 64, at exposure 0.5
    assert abs(int(east[1]) - 255 * 3 * 32) < 300 and east[0] < 300 and r["eyes"]["left"]["above_8bit_range"] > 0.2
    assert r["eyes"]["left"]["clipped_in_16bit"] == 0 and r["linear16"]["exposure16"] == 0.5
    assert 200 < L8[70, 64, 1] <= 255 and L8[70, 64, 0] < L8[70, 64, 1] - 50      # the view: bright, faded, still green
    lo, hi = r["linear16"]["view_levels_8bit_units"]
    assert r["linear16"]["view_tone"] == "auto" and hi > lo >= 0
    north = L16[70, 2:126:8]                                      # the red frame: neutral where it was saturated
    assert (np.abs(north[:, 0].astype(int) - north[:, 1].astype(int)) < 300).any() and (north[:, 1] < 300).any()
    r2 = O.render_station(proj, rec, st, scene, tmp_path / "o16b", width=128, scale=0.5, rows_per_chunk=32, bits=16, gains=False,
                          exposure16=1.0, tone="clip", eyes=("left",))
    e2 = cv2.imread(str(tmp_path / "o16b" / "S003D0010_ods_left_16.png"), cv2.IMREAD_UNCHANGED)[35, 32]
    assert abs(int(e2[1]) - 255 * 3 * 64) < 300 and r2["linear16"]["view_levels_8bit_units"] is None
    sq = cv2.imread(str(out / "S003D0010_360_3DTD_16.png"), cv2.IMREAD_UNCHANGED)
    assert sq.dtype == np.uint16 and sq.shape == (256, 256, 3) and np.array_equal(sq[:128], L16[..., ::-1])
    assert (out / "S003D0010_360_3DTD.jpg").is_file()
    # a frame without its 16-bit image: the 8-bit panorama, and the record says so
    (d16 / st.images[1]).unlink()
    r8 = O.render_station(proj, rec, st, scene, tmp_path / "o8", width=128, scale=0.5, rows_per_chunk=32, bits=16, gains=False)
    assert r8["bits"] == 8 and r8["linear16"]["missing"] and not (tmp_path / "o8" / "S003D0010_ods_left_16.png").is_file()


def test_ground_fill_and_levels():
    from mppp.ibr.linear16 import auto_levels, tone_levels
    H, W = 64, 128
    rgb = np.zeros((H, W, 3), np.float32)
    source = np.full((H, W), -1, np.int16)
    donors = np.zeros((H, W), bool)
    rgb[32:48] = (120.0, 80.0, 60.0)                              # terrain below the horizon, rows 32-47
    source[32:48] = 0
    donors[32:48] = True
    rgb[44:48, 40:60] = (900.0, 900.0, 900.0)                     # the white deck: coloured, not a donor
    donors[44:48, 40:60] = False
    f = O.fill_ground(rgb, source, donors)
    assert f[48:].all() and not f[:48].any() and (source[48:] == -4).all()
    assert np.allclose(rgb[60, 100], (120, 80, 60), atol=2) and np.allclose(rgb[50, 50], (120, 80, 60), atol=25)   # terrain, not deck
    assert (rgb[:32] == 0).all()                                  # nothing above the horizon
    pan = np.zeros((100, 200, 3), np.float32)
    pan[:50] = 300.0                                              # sky
    pan[50:] = (60.0, 40.0, 30.0)                                 # ground: mean 43.3
    pan[80:, :40] = 900.0                                         # the white deck, a fifth of the lower half
    lo, mid = auto_levels(pan)
    assert abs(mid - 130.0 / 3) < 1 and lo <= 30
    y = tone_levels(np.array([[lo, lo, lo], [mid, mid, mid], [900.0, 880.0, 990.0], [1e5, 1e5, 1e5]], np.float32), lo, mid)
    assert np.allclose(y[0], 0) and abs(y[1, 0] - 100) < 0.01                      # the ground's median at 100
    assert 200 < y[2].min() and y[2].max() <= 255 and y[2].max() - y[2].min() < 8   # the deck: near white, not clipped, its tint faded
    assert y[3].max() <= 255


def test_retone_from_the_16bit_panoramas(tmp_path):
    import cv2
    from mppp.ibr.linear16 import retone
    g = np.random.default_rng(0)
    for eye in ("left", "right"):
        a = np.zeros((64, 128, 3), np.uint16)
        a[32:] = g.integers(2000, 3000, (32, 128, 3))
        a[:32] = 9000
        cv2.imwrite(str(tmp_path / f"S1_rover_ods_{eye}_16.png"), a)
    r = retone(tmp_path, "S1_rover")
    sq = cv2.imread(str(tmp_path / "S1_rover_360_3DTD.jpg"))
    L8 = cv2.imread(str(tmp_path / "S1_rover_ods_left.png"))
    assert sq.shape == (128, 128, 3) and 80 < np.median(L8[32:]) < 120 and L8[:32].min() > 180 and r["top"] == "left"


def test_rover_model_in_the_mesh(tmp_path):
    from mppp.ibr import rover_deck as RD, rover_model as RM
    assert RM.available()
    V, F, meta = RM.load()
    assert meta["credit"] == "NASA/JPL-Caltech" and 40000 < len(F) < 80000 and meta["arm_stow_deg"] == [96.0, 20.4, 71.9, -93.3, 85.0] and "antenna" in meta["parts"]   # 0.97.0
    assert V[:, 0].min() < -1.6 and V[:, 0].max() < 1.5            # the RTG behind; the arm stowed, not reaching forward
    assert abs(V[:, 1]).max() < 1.4 and 1.05 < V[:, 2].max() < 1.2  # the wheels on the ground, 1.13 m below the deck top
    Vb, Fb, _ = RM.load(["body"])
    assert len(Fb) < len(F)
    top = Vb[np.unique(Fb)]
    deck = top[(np.abs(top[:, 0]) < 0.5) & (np.abs(top[:, 1]) < 0.4) & (np.abs(top[:, 2]) < 0.02)]
    assert len(deck) > 20 and (np.abs(deck[:, 2]) < 0.012).mean() > 0.3     # the deck top at z 0
    pose = {"R_block_from_rnav": np.diag([1.0, -1.0, -1.0]), "t_block_from_rnav": np.array([3.0, 2.0, 0.0])}
    S = RD.mech_to_block(RM.surface_samples()[::50], pose)
    assert RM.near_model(S, pose).all() and not RM.near_model(S + np.array([0.0, 0.0, 3.0]), pose).any()
    Vm, Fm = RM.model_in_block(pose)
    assert Fm.max() == len(Vm) - 1 and abs(Vm[:, 2].min()) < 0.02   # block z up: the wheels at z 0 (RNAV origin on the ground)
    g = np.random.default_rng(0)
    terr = np.c_[g.uniform(-4, 10, (40000, 2)), np.zeros(40000)]
    on_deck = RD.mech_to_block(np.c_[g.uniform(-0.5, 0.5, (3000, 1)), g.uniform(-0.3, 0.3, (3000, 1)), g.normal(0, 0.005, 3000)], pose)
    pivot = RD.mech_to_block(np.array([[0.72, 0.56, -0.78]]), pose)[0]
    rep = M.build_station_mesh(np.vstack([terr, on_deck]), pivot, tmp_path, radius_m=6.0, plane_radius_m=10.0, poisson_depth=7, deck=pose)
    d = rep["rover_deck"]
    assert d["model_triangles"] == len(RM.load(M.ROVER_MODEL_PARTS)[1]) and d["points_on_model_dropped"] > 2500 and "rear_floor_triangles" not in d
    rep2 = M.build_station_mesh(np.vstack([terr, on_deck]), pivot, tmp_path / "b", radius_m=6.0, plane_radius_m=10.0, poisson_depth=7,
                                deck=pose, rover_model_parts=None)
    assert "model_triangles" not in rep2["rover_deck"] and rep2["rover_deck"]["rear_floor_triangles"] > 0


def test_mast_mask_from_the_rover_model():
    from mppp.ibr import rover_deck as RD, rover_model as RM
    V, F, meta = RM.load(["mast"])
    mv = V[np.unique(F)]
    assert meta["mast_axis_xy_m"] == [0.716, 0.559] and len(F) > 2000
    boom = mv[(mv[:, 1] > 0.85)]                                  # the boom to the right rear, 0.46 m above the deck
    assert len(boom) and abs(boom[:, 2].mean() + 0.46) < 0.05 and "mast" not in M.ROVER_MODEL_PARTS
    pose = {"R_block_from_rnav": np.diag([1.0, -1.0, -1.0]), "t_block_from_rnav": np.zeros(3)}

    class Src:                                                    # a pinhole at the head, looking at the boom / away from it
        W, H = 400, 300

        def __init__(self, target):
            self.C = RD.mech_to_block(np.array([[0.75, 0.6, -0.80]]), pose)[0]
            z = target - self.C
            z /= np.linalg.norm(z)
            x = np.cross(z, [0.0, 0.0, 1.0]); x /= np.linalg.norm(x)
            self.R = np.stack([x, np.cross(z, x), z])

        def project(self, X):
            Xc = (X - self.C) @ self.R.T
            uv = np.full((len(X), 2), np.nan)
            f = Xc[:, 2] > 1e-3
            uv[f] = 200.0 * Xc[f, :2] / Xc[f, 2:3] + (200.0, 150.0)
            return uv, Xc[:, 2], None
    tip = RD.mech_to_block(boom.mean(axis=0)[None], pose)[0]
    far = Src(tip)
    far.C = far.C + np.array([20.0, 0.0, 0.0])                    # 20 m off, looking the same way: the mast is not in view
    at, away = RM.mast_masks([Src(tip), far], pose)
    assert at is not None and at[150, 200] and 0.01 < at.mean() < 0.9 and away is None
    cloud = RD.mech_to_block(np.vstack([boom, boom + np.array([0.0, 0.0, 1.0])]), pose)
    near = RM.near_model(cloud, pose, near_m=RM.MAST_NEAR_M, parts=("mast",))
    assert near[: len(boom)].all() and not near[len(boom):].any()


def test_pivot_at_mastcamz_height_and_shared_tone_levels(tmp_path):
    """0.97.0 (Christian, 12 Nov 2026): the pivot on the azimuth axis at the Mastcam-Z height whatever cameras are
    rendered; one 16-to-8-bit scaling for the rover-shown and the rover-masked panorama; the centre depth files; the
    jpg pictures with linear and log depth."""
    from scipy.spatial.transform import Rotation
    from mppp.ibr import linear16 as L16
    from mppp.ibr import station as S
    up = np.array([0.0, 0.0, 1.0])
    axis = np.array([2.0, -1.0, 5.0])                       # the elevation axis on the azimuth axis
    C, R, G = [], [], []
    for fam, el in (("N", -30.0), ("N", -5.0), ("Z", -30.0), ("Z", 10.0)):
        fwd, upc = S.HEAD_FORWARD_UP_M[fam]
        for az in range(0, 360, 45):
            a, e = np.radians(az), np.radians(el)
            view = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
            right = np.cross(view, up); right /= np.linalg.norm(right)
            down = np.cross(view, right)
            R.append(np.stack([right, down, view]))                           # world -> camera rows
            hz = np.array([np.sin(a), np.cos(a), 0.0])
            C.append(axis + hz * (fwd * np.cos(e) - upc * np.sin(e)) + up * (upc * np.cos(e) + fwd * np.sin(e)))
            G.append(fam + "L")
    C, R = np.array(C), np.array(R)
    for sel in ([g[0] == "N" for g in G], [g[0] == "Z" for g in G]):          # Navcam frames alone, Mastcam-Z alone
        sel = np.array(sel)
        P, rec = S.mastcamz_height_pivot(axis + [0, 0, 0.3], C[sel], R[sel], [g for g, k in zip(G, sel) if k], up)
        assert np.allclose(P, axis + up * S.ZCAM_HEIGHT_M, atol=1e-9) and rec["rule"] == "zcam"
    # the levels: found once, written, then read - in the caller's units
    f = tmp_path / "S1" / "tone_levels.json"
    lv, src = L16.shared_levels(f, lambda: (2.0, 50.0))
    assert src == "computed" and lv == (2.0, 50.0) and f.is_file()
    lv2, src2 = L16.shared_levels(f, lambda: (9.0, 99.0))                      # the other variant: not found again
    assert src2 == "file" and lv2 == (2.0, 50.0)
    lv3, _ = L16.shared_levels(f, lambda: (0.0, 0.0), scale=32.0)              # the 16-bit files at exposure 0.5
    assert lv3 == (64.0, 1600.0)
    # the centre depth and the jpg pictures
    rng = np.full((8, 16), 4.0, np.float32); lab = np.full((8, 16), RC.LABEL_MESH, np.uint8)
    rng[0], lab[0] = RC.SKY_RANGE_M, RC.LABEL_FAR
    rng[7], lab[7] = 0.0, RC.LABEL_NONE
    d = O.write_centre_depth(tmp_path, "S1_rover", rng, lab)
    z = np.load(d["files"]["npz"])
    assert np.isinf(z["range"][0]).all() and (z["range"][7] == 0).all() and np.allclose(z["range"][3], 4.0)
    import cv2
    d16 = cv2.imread(d["files"]["png16"], cv2.IMREAD_UNCHANGED)
    assert d16.dtype == np.uint16 and d16[3, 0] == 2000 and d16[0, 0] == 65535 and d16[7, 0] == 0
    assert Path(d["files"]["linear"]).name == "S1_rover_ods_centre_depth_linear.jpg" and Path(d["files"]["log"]).is_file()
    lin, lg = RC.depth_rgb(rng, lab, "linear"), RC.depth_rgb(rng, lab, "log")
    assert lin.shape == (8, 16, 3) and not np.array_equal(lin[3], lg[3])
    maps = {"range": rng, "label": lab, "normal": np.zeros((8, 16, 3), np.float32)}
    names = sorted(p.name for p in RC.preview_jpg(maps, tmp_path / "photometry_jpg" / "IMG"))
    assert names == ["IMG_depth_linear.jpg", "IMG_depth_log.jpg", "IMG_normal.jpg"]


def test_floating_rule_keeps_far_terrain_above_the_plane():
    """0.97.0 (Rockytop S026D0630: the mesh ended at 15 m where the ground rose above the fitted plane): with the
    local ground, a hill 3 m high 25-40 m away is terrain; a blob 2 m above the ground 8 m away is still floating."""
    from mppp.ibr import mesh as M
    rng = np.random.default_rng(3)
    xy = rng.uniform(-45, 45, (60_000, 2))
    z = 3.0 * np.exp(-((xy[:, 0] - 30) ** 2 + xy[:, 1] ** 2) / (2 * 8.0 ** 2))          # a hill to the east
    ground_pts = np.c_[xy, z]
    blob = np.c_[rng.normal(8, 0.2, 300), rng.normal(0, 0.2, 300), rng.normal(2.0, 0.1, 300)]
    pts = np.vstack([ground_pts, blob])
    P, up = np.array([0.0, 0.0, 1.5]), np.array([0.0, 0.0, 1.0])
    plane = {"normal": [0.0, 0.0, 1.0], "offset": 0.0}
    hill = (np.hypot(xy[:, 0] - 30, xy[:, 1]) < 6)
    old = M.floating(pts, P, up, plane)
    assert old[:len(xy)][hill].mean() > 0.9 and old[len(xy):].all()                 # the 0.96 rule took the hill too
    new = M.floating(pts, P, up, plane, ground=M.local_ground(pts, P, up, plane))
    assert new[:len(xy)][hill].mean() < 0.02 and new[:len(xy)].mean() < 0.01
    assert new[len(xy):].mean() > 0.95


def test_fill_terrain_closes_holes_and_flattens_under_the_rover():
    """0.97.0 (Christian, 12 Nov 2026: "interpolate terrain under the rover so that it is flattish terrain without
    holes. The far field should interpolate over hole in the terrain"): a sloped terrain with a far hole and junk
    under the rover - after the fill every downward ray from the pivot within the radius meets a surface, the ground
    under the rover follows the slope, and the far hole is bridged at the terrain's height."""
    from mppp.ibr import mesh as M
    n = 161
    g = np.linspace(-20, 20, n)
    X, Y = np.meshgrid(g, g)
    Z = 0.1 * X                                              # a 5.7 deg slope
    V = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    F = np.r_[np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()],
              np.c_[idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]]
    cen = V[F].mean(1)
    far_hole = (np.hypot(cen[:, 0] + 12, cen[:, 1]) < 3)     # a hole 12 m behind
    F = F[~far_hole]
    V = V.copy()
    near = np.hypot(V[:, 0], V[:, 1]) < 1.2
    V[near, 2] += 0.6                                        # junk under the rover
    P, up = np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0])
    plane = {"normal": [0.0, 0.0, 1.0], "offset": 0.0}
    pose = {"R_block_from_rnav": np.diag([1.0, -1.0, -1.0]).tolist(), "t_block_from_rnav": [0.0, 0.0, 1.0]}
    Vf, Ff, rep = M.fill_terrain(V, F, P, up, plane, 18.0, rover=pose)
    assert rep["under_rover_triangles_removed"] > 100 and rep["filled"] > 0 and rep["triangles"] > 0
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(Vf.astype(np.float32)), o3d.core.Tensor(Ff.astype(np.uint32)))
    rng = np.random.default_rng(0)
    tgt = np.c_[rng.uniform(-12, 12, (4000, 2)), np.zeros(4000)]
    tgt[:, 2] = 0.1 * tgt[:, 0]
    d = tgt - P
    d /= np.linalg.norm(d, axis=1)[:, None]
    t = sc.cast_rays(o3d.core.Tensor(np.c_[np.tile(P, (len(d), 1)), d].astype(np.float32)))["t_hit"].numpy()
    assert np.isfinite(t).all()                                                    # no ray falls through
    hit = P + d * t[:, None]
    under = np.hypot(hit[:, 0], hit[:, 1]) < 1.0
    assert under.sum() > 5 and np.abs(hit[under, 2] - 0.1 * hit[under, 0]).max() < 0.15    # flattish: the slope, not the junk
    hole = np.hypot(hit[:, 0] + 12, hit[:, 1]) < 2.5
    assert hole.sum() > 20 and np.abs(hit[hole, 2] - 0.1 * hit[hole, 0]).max() < 0.2


def test_surface_check_keeps_rover_pixels_on_the_rover_and_terrain_on_terrain(tmp_path):
    """0.97.0 (Christian, 12 Nov 2026: "nothing from the rover mask lands on the terrain and nothing from the terrain
    mask lands on the rover ... where the disagreement is too much ... leave that a brown fill color"): a box (the
    'rover model') on a ground mesh, one camera looking down; its mask calls a patch larger than the box 'rover'."""
    import cv2
    import pycolmap
    from mppp.ibr import ods as O
    from mppp.ibr.raycast import Scene
    # the ground z = 0 (two triangles) and a box top at z = 0.5 over |x|, |y| < 0.5: the model, last in the mesh
    Vg = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], float)
    Fg = np.array([[0, 1, 2], [0, 2, 3]])
    Vb = np.array([[-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]], float)
    Fb = np.array([[0, 1, 2], [0, 2, 3]])
    sc = Scene((np.vstack([Vg, Vb]), np.vstack([Fg, Fb + 4])))
    sc.rover_triangles = (2, 4)
    W = H = 200
    cam = pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[100.0, 100.0, 100.0, 100.0])
    R = np.diag([1.0, -1.0, -1.0])                           # looking straight down from z = 4
    C = np.array([0.0, 0.0, 4.0])
    img = np.full((H, W, 3), 200, np.uint8)
    cv2.imwrite(str(tmp_path / "im.png"), img)
    mask = np.full((H, W), 255, np.uint8)                    # > 0: terrain
    mask[70:130, 70:130] = 0                                 # 'rover': +-30 px = +-1.2 m on the ground, +-1.05 m at the box top
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    src = O.SourceImage("im.png", cam, R, C, tmp_path / "im.png", None, 1.0)
    src.rover_mask_file = tmp_path / "mask.png"
    # rays straight down onto: the box top (rover pixel, rover surface), the ground inside the mask patch (rover
    # pixel, terrain: rejected), the ground outside (terrain pixel, terrain)
    X = np.array([[0.0, 0.0], [0.9, 0.0], [2.0, 0.0]])
    o = np.c_[X, np.full(3, 3.0)]
    d = np.tile([0.0, 0.0, -1.0], (3, 1))
    c = O.candidates(sc, [src], o, d)
    assert list(c["on_rover"]) == [True, False, False]
    assert list(c["k"][:, 0] >= 0) == [True, False, True] and c["rejected"] == [1, 0]
    mask[:] = 255                                            # the mask says terrain everywhere: nothing lands on the box
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    src._rover = None
    c = O.candidates(sc, [src], o, d)
    assert list(c["k"][:, 0] >= 0) == [False, True, True] and c["rejected"] == [0, 1]
    c = O.candidates(sc, [src], o, d, surface_check=False)   # the check off: as before
    assert (c["k"][:, 0] >= 0).all()
