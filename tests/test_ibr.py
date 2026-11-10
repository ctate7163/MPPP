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
    L = cv2.imread(str(out / "ods_left.png"))[..., ::-1]
    Rr = cv2.imread(str(out / "ods_right.png"))[..., ::-1]
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
    # straight up: no frame looks there (they tilt 15 deg down): black, provenance none
    assert np.all(at(L, 10.0, 85.0) == 0)
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
    assert mppp_main(["mesh", work, "--station", "3_10", "--poisson-depth", "7", "--target-triangles", "20000", "--no-preview"]) == 0
    out = root / "ods" / "S003D0010"
    rep = json.loads((out / "mesh_report.json").read_text(encoding="utf-8"))
    assert (out / "mesh.ply").is_file() and (out / "far_sphere.ply").is_file() and rep["points"] == len(pts)
    assert rep["station"]["station"] == "S003D0010" and rep["points_file"].endswith("fused.ply")
    V, F = M.read_mesh(out / "mesh.ply")
    assert len(F) > 1000 and abs(float(np.median(V[:, 2]))) < 0.1                     # the ground plane, near z = 0

    assert mppp_main(["depthmaps", work, "--station", "3_10", "--scale", "0.25", "--photometry", "--no-preview"]) == 0
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

    assert mppp_main(["ods", work, "--station", "3_10", "--width", "256", "--scale", "0.25", "--rows-per-chunk", "32"]) == 0
    L = cv2.imread(str(out / "ods_left.png"))
    Rr = cv2.imread(str(out / "ods_right.png"))
    assert L.shape == (128, 256, 3) and Rr.shape == L.shape
    oj = json.loads((out / "ods.json").read_text(encoding="utf-8"))
    assert oj["eyes"]["left"]["coverage"]["coloured"] > 0.2 and oj["eyes"]["left"]["sources_used"] == 4
    assert (out / "station.json").is_file()


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
    L = cv2.imread(str(out / "ods_left.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    Rr = cv2.imread(str(out / "ods_right.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
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
