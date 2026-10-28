"""mppp.sun (0.95.0): the label sun vector, the Sun-Mars distance, a block's tilt and the sun in the block frame -
recorded at freeze time for the paper-3 work (Christian, 27 Oct 2026)."""
import json

import numpy as np
import pytest

from mppp import sun as S


def test_sun_vector_and_mars_distance():
    assert np.allclose(S.sun_vector_enu(90.0, 0.0), [1, 0, 0])            # east, on the horizon
    assert np.allclose(S.sun_vector_enu(0.0, 90.0), [0, 0, 1])            # zenith
    v = S.sun_vector_enu(180.0, 30.0)
    assert v[1] < 0 and v[2] == pytest.approx(0.5) and np.linalg.norm(v) == pytest.approx(1.0)
    assert S.sun_vector_enu(None, 10.0) is None
    # the orbit: perihelion 1.381 au at L_s 251, aphelion 1.666 au at L_s 71; 1.4-1.7 au over the year
    assert S.mars_sun_distance_ls(251.0) == pytest.approx(S.MARS_A_AU * (1 - S.MARS_E), abs=1e-6)
    assert S.mars_sun_distance_ls(71.0) == pytest.approx(S.MARS_A_AU * (1 + S.MARS_E), abs=1e-6)
    assert 1.38 < S.mars_sun_distance_ls(0.0) < 1.67
    d, src = S.mars_sun_distance_au(100.0, utc=None)
    assert src == "ls_ellipse" and 1.38 < d < 1.67
    assert S.mars_sun_distance_au(None) == (None, None)
    rec = S.sun_record({"L_s_deg": 300.0, "solar_azimuth_deg": 45.0, "solar_elevation_deg": 60.0}, spice=False)
    assert rec["d_mars_source"] == "ls_ellipse" and rec["sun_enu"][2] == pytest.approx(np.sin(np.radians(60)))
    assert S.sun_record({}, spice=False) == {"d_mars_au": None, "d_mars_source": None, "sun_enu": None}


def test_spice_without_kernels_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("MPPP_KERNELS", str(tmp_path))                        # an empty folder: no kernels
    assert S.mars_sun_distance_spice("2021-02-18T20:55:00") is None
    d, src = S.mars_sun_distance_au(5.0, utc="2021-02-18T20:55:00")
    assert src == "ls_ellipse"
    monkeypatch.delenv("MPPP_KERNELS")
    assert S.kernels_dir() is None and S.mars_sun_distance_spice("2021-02-18T20:55:00") is None


def test_block_rotation_and_sun_in_block():
    """A block whose refined attitudes are the label attitudes turned by W (about E by 1 deg): W is recovered and the
    sun vector rotated with it; an image with too few points or no prior is left out."""
    pycolmap = pytest.importorskip("pycolmap")
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace
    W = Rotation.from_rotvec(np.radians([1.0, 0.0, 0.0])).as_matrix()
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(camera_id=1, model="SIMPLE_PINHOLE", width=100, height=100, params=[100.0, 50.0, 50.0])
    rec.add_camera(cam)
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1))
    rec.add_rig(rig)
    rows = []
    rng = np.random.default_rng(0)
    pts = [rec.add_point3D(rng.normal(0, 1, 3) + [0, 5, 0], pycolmap.Track()) for _ in range(40)]
    for i in range(1, 6):
        R0 = Rotation.from_euler("xyz", [90 + 2 * i, 0, 10 * i], degrees=True).as_matrix()
        R1 = R0 @ W.T                                                         # R1^T R0 = W
        fr = pycolmap.Frame(frame_id=i, rig_id=1)
        fr.add_data_id(pycolmap.data_t(sensor_id=pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=1), id=i))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R1), np.zeros(3))
        rec.add_frame(fr)
        im = pycolmap.Image(name=f"im{i}.png", keypoints=np.zeros((40, 2)), camera_id=1, image_id=i)
        im.frame_id = i
        rec.add_image(im)
        rec.register_frame(i)
        n = 40 if i < 5 else 5                                                # the last: too few points
        for j in range(n):
            rec.images[i].set_point3D_for_point2D(j, pts[j])
        rows.append({"name": f"im{i}.png", "prior_R_w2c": R0.tolist() if i < 4 else None})
    proj = SimpleNamespace(images=rows)
    b = S.block_rotation(rec, proj)
    assert b["images"] == 3 and b["angle_deg"] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(b["R_block_from_enu"], W, atol=1e-9) and b["rotvec_deg"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    s = S.sun_in_block([0.0, 0.0, 1.0], b["R_block_from_enu"])
    assert np.allclose(s, W @ [0, 0, 1]) and s[2] == pytest.approx(np.cos(np.radians(1.0)))
    assert S.sun_in_block(None, W) is None and S.block_rotation(rec, SimpleNamespace(images=[])) is None
