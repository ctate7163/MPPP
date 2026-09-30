"""MPPP v0p43.1: stage 2 of a staged Nav+Zcam run restores a Mastcam-Z stereo rig whose cameras COLMAP dropped."""
import sys

import numpy as np
import pytest


def _block():
    """Navcam rig 1 (camera 1) with two registered frames; Mastcam-Z stereo rig 2 (cameras 2 = ZL, 3 = ZR) with one
    frame of two images, as ZCAM_RIG builds it."""
    pycolmap = pytest.importorskip("pycolmap")
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    rng = np.random.default_rng(0)
    rec = pycolmap.Reconstruction()
    for c in (1, 2, 3):
        rec.add_camera(pycolmap.Camera(camera_id=c, model="PINHOLE", width=100, height=100,
                                       params=[100.0 + c, 100.0 + c, 50, 50]))
    r1 = pycolmap.Rig(rig_id=1)
    r1.add_ref_sensor(S(1))
    rec.add_rig(r1)
    r2 = pycolmap.Rig(rig_id=2)
    r2.add_ref_sensor(S(2))
    r2.add_sensor(S(3), pycolmap.Rigid3d(pycolmap.Rotation3d(), [0.24, 0, 0]))
    rec.add_rig(r2)

    def frame(fid, rid, cams, x):
        fr = pycolmap.Frame(frame_id=fid, rig_id=rid)
        for k, c in enumerate(cams):
            fr.add_data_id(pycolmap.data_t(sensor_id=S(c), id=10 * fid + k))
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(), [x, 0, 0])
        rec.add_frame(fr)
        for k, c in enumerate(cams):
            im = pycolmap.Image(name=f"i{fid}_{k}", keypoints=rng.uniform(0, 100, (40, 2)), camera_id=c,
                                image_id=10 * fid + k)
            im.frame_id = fid
            rec.add_image(im)
        rec.register_frame(fid)

    frame(1, 1, [1], 0.0)
    frame(2, 1, [1], 0.5)
    frame(3, 2, [2, 3], 0.2)
    P = rng.uniform(-1, 1, (40, 3)) + [0, 0, 10]
    for p in range(40):
        t = pycolmap.Track()
        t.add_element(10, p)
        t.add_element(20, p)
        rec.add_point3D(P[p], t, np.zeros(3, np.uint8))
    return pycolmap, rec


def test_restore_frames_brings_back_a_torn_down_stereo_rig():
    """The 30 Sep threeforks_south failure: ``Camera 28 from rig 27 not found`` at the start of stage 2."""
    pycolmap, rec = _block()
    from mppp.sfm.reconstruction import restore_frames
    init = pycolmap.Reconstruction(rec)
    rec.deregister_frame(3)
    rec.tear_down()                                   # what triangulate_points does to stage 1
    assert 2 not in rec.rigs and 2 not in rec.cameras and 3 not in rec.cameras
    assert restore_frames(rec, init, [3, 1]) == 1
    assert sorted(rec.rigs) == [1, 2] and sorted(rec.cameras) == [1, 2, 3]
    assert rec.images[30].camera_id == 2 and rec.images[31].camera_id == 3
    assert np.allclose(rec.cameras[3].params, init.cameras[3].params)
    rec.frames[3].rig_from_world = init.frames[3].rig_from_world
    rec.register_frame(3)
    assert rec.num_reg_images() == init.num_reg_images()
    assert np.allclose(rec.rigs[2].sensor_from_rig(pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=3))
                       .translation, [0.24, 0, 0])


def test_restore_frames_keeps_refined_cameras_that_survived():
    """A camera still in the block (refined in stage 1) is not overwritten by its start value."""
    pycolmap, rec = _block()
    from mppp.sfm.reconstruction import restore_frames
    init = pycolmap.Reconstruction(rec)
    rec.deregister_frame(3)
    rec.tear_down()
    rec.cameras[1].params = [150.0, 150.0, 50, 50]
    restore_frames(rec, init, [3])
    assert rec.cameras[1].params[0] == 150.0


def test_version():
    import mppp
    assert tuple(int(x) for x in mppp.__version__.split(".")[:3]) >= (0, 43, 1)


def test_staged_reconstruct_with_a_zcam_stereo_rig(tmp_path, monkeypatch):
    """Staged reconstruct end to end with the second station's frames on their own two-camera rig ("ZL"/"ZR"), and a
    triangulate that tears the block down as COLMAP's triangulate_points does."""
    pytest.importorskip("pyceres")
    pycolmap = pytest.importorskip("pycolmap")
    import copy
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    rec0, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng)
    fids = sorted(rec0.frames)
    zf = set(fids[len(fids) // 2:])
    # move those frames onto rig 2 with cameras 3 (ZL) and 4 (ZR), copies of the Navcam ones
    new = pycolmap.Reconstruction()
    for cid, c in rec0.cameras.items():
        new.add_camera(c)
        cz = pycolmap.Camera(camera_id=cid + 2, model=c.model, width=c.width, height=c.height, params=c.params)
        new.add_camera(cz)
    new.add_rig(rec0.rigs[1])
    S = lambda c: pycolmap.sensor_t(type=pycolmap.SensorType.CAMERA, id=c)      # noqa: E731
    r2 = pycolmap.Rig(rig_id=2)
    r2.add_ref_sensor(S(3))
    r2.add_sensor(S(4), rec0.rigs[1].sensor_from_rig(S(2)))
    new.add_rig(r2)
    for fid in fids:
        fr = rec0.frames[fid]
        z = fid in zf
        nf = pycolmap.Frame(frame_id=fid, rig_id=2 if z else 1)
        for d in fr.data_ids:
            nf.add_data_id(pycolmap.data_t(sensor_id=S(d.sensor_id.id + (2 if z else 0)), id=d.id))
        nf.rig_from_world = fr.rig_from_world
        new.add_frame(nf)
        for d in fr.data_ids:
            im = rec0.images[d.id]
            ni = pycolmap.Image(name=im.name, keypoints=np.array([q.xy for q in im.points2D]),
                                camera_id=im.camera_id + (2 if z else 0), image_id=d.id)
            ni.frame_id = fid
            new.add_image(ni)
        new.register_frame(fid)
    for pid, p in rec0.points3D.items():
        new.add_point3D(p.xyz, p.track, np.zeros(3, np.uint8))
    zimg = {d.id for f in zf for d in rec0.frames[f].data_ids}
    for r in proj.images:
        if r["image_id"] in zimg:
            r["instrument"] = "Z" + r["instrument"][1:]
            r["camera_group"] = r["instrument"]
    for k in ("NL", "NR"):
        proj.cameras["Z" + k[1:]] = dict(proj.cameras[k])
    proj.settings["database"] = {"cameras": {"NL": 1, "NR": 2, "ZL": 3, "ZR": 4}}

    def tearing_triangulate(rec, project, **kw):
        rec.tear_down()
        return rec

    monkeypatch.setattr(R, "initial_reconstruction", lambda project: copy.deepcopy(new))
    monkeypatch.setattr(R, "triangulate", tearing_triangulate)
    rec = R.reconstruct(proj, sigma_px=noise, schedule=((24.0, 10.0, 8.0),), max_iterations=5, verbose=False,
                        navcam_intrinsics="refine", staged=True, out_name="cahv_ba")
    assert proj.settings["reconstruction"]["staged"] is True
    assert set(rec.reg_frame_ids()) >= zf and sorted(rec.rigs)[:2] == [1, 2]
    assert {3, 4} <= set(rec.cameras)


# ------------------------------------------------------------------------------ processing only (process_sites.py)
def test_notebook03_processing_sections_come_before_section_3():
    import nbformat
    from mppp.runner import PROCESS_STOP, notebook, truncate_before
    nb = nbformat.read(str(notebook("03_colmap_alignment")), as_version=4)
    n = truncate_before(nb, PROCESS_STOP)
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert n > 5 and "process_images(" in src and "select_best_products(" in src
    assert "extract_features(" not in src and "reconstruct(" not in src and "SfmProject.create" not in src
    with pytest.raises(ValueError):
        truncate_before(nb, "## 99")


def test_run_notebook_stops_before_a_heading(tmp_path):
    pytest.importorskip("nbclient")
    import nbformat
    from mppp.runner import run_notebook
    nb = nbformat.v4.new_notebook()
    p = nbformat.v4.new_code_cell("X = 1")
    p.metadata["tags"] = ["parameters"]
    nb.cells = [p, nbformat.v4.new_code_cell("open(OUT, 'w').write(str(X))"),
                nbformat.v4.new_markdown_cell("## 3. Align"), nbformat.v4.new_code_cell("raise RuntimeError('ran on')")]
    src = tmp_path / "t.ipynb"
    nbformat.write(nb, str(src))
    out = tmp_path / "x.txt"
    lines = []
    assert run_notebook(src, tmp_path / "t_out.ipynb", {"X": "7", "OUT": repr(str(out))}, lines.append,
                        stop_before="## 3")
    assert out.read_text() == "7"
    assert not run_notebook(src, tmp_path / "t_out2.ipynb", {"X": "7", "OUT": repr(str(out))}, lines.append)


def test_processed_done(tmp_path):
    import json
    from mppp.runner import PROCESS_DONE, processed_done
    (tmp_path / "processed").mkdir()
    assert processed_done(tmp_path, "k") is None
    (tmp_path / "processed" / PROCESS_DONE).write_text(json.dumps({"run_key": "k", "images": 3}))
    assert processed_done(tmp_path, "k")["images"] == 3 and processed_done(tmp_path, "other") is None


def test_process_sites_dry_run(tmp_path, capsys):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("process_sites", root / "scripts" / "process_sites.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    assert ps.main(["--sites", "sid", "rockytop", "--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "process_sites_log.txt").read_text()
    assert "sid: would run" in log and "rockytop_colmap" in log
    assert ps.main(["--group", "nav_zcam34", "--zcam", "--root", str(tmp_path), "--dry-run"]) == 0
    assert "_colmap_nav_zcam34" in (tmp_path / "process_sites_log.txt").read_text()
    assert ps.main(["--sites", "nowhere", "--root", str(tmp_path), "--dry-run"]) == 2


def test_windows_bat_files_are_unversioned_and_crlf():
    from pathlib import Path
    win = Path(__file__).resolve().parents[1] / "scripts" / "windows"
    for name in ("process_sites.bat", "_run_process.bat", "align_here.bat"):
        b = (win / name).read_bytes()
        assert b"\r\n" in b and b.count(b"\n") == b.count(b"\r\n"), name
        assert b"v0p4" not in b, name
    assert b"process_sites.py" in (win / "_run_process.bat").read_bytes()


def test_bat_files_work_from_any_folder_and_find_the_environment():
    """0.43.1: top-level .bat files find MPPP through MPPP_HOME when copied elsewhere; mppp_env.bat searches the
    conda environments and uses check_env.py."""
    import subprocess
    from pathlib import Path
    win = Path(__file__).resolve().parents[1] / "scripts" / "windows"
    for name in ("process_sites.bat", "run_all_sites.bat", "sites_status.bat"):
        s = (win / name).read_text()
        assert 'call "%MPPP_WIN%\\mppp_env.bat"' in s and '"%~dp0mppp_env.bat" ||' not in s, name
        assert "%~dp0_run" not in s, name
    env = (win / "mppp_env.bat").read_text()
    for part in ("check_env.py", "environments.txt", "mppp_python.txt", "MPPP_PYTHON", ":try_prefix", ":try_root",
                 "KMP_DUPLICATE_LIB_OK"):
        assert part in env, part
    body = env.split(":try_prefix\n")[1]
    assert "%~dp0" not in body                     # inside a call :label, %0 is the label
    r = subprocess.run([sys.executable, str(win / "check_env.py")], capture_output=True, text=True)
    assert r.returncode in (0, 1) and sys.executable in r.stdout
