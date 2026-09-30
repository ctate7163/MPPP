"""Command-line scripts and Windows .bat files (scripts/). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
from pathlib import Path
import numpy as np
import pytest
import sys
import json


ROOT = Path(__file__).resolve().parents[1]


def test_site_appearance_measures_on_synthetic_texture():
    pytest.importorskip("cv2")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from site_appearance import measures, elevation_map
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    base = 20000 + 4000 * gaussian_filter(rng.normal(size=(600, 800)), 2.0)
    smooth = 20000 + 4000 * gaussian_filter(rng.normal(size=(600, 800)), 8.0)
    mask = np.zeros((600, 800)); mask[50:550, 50:750] = 255
    u8 = lambda x: 255 * (x - x.min()) / np.ptp(x)                               # noqa: E731
    a = measures(base, u8(base), mask)
    b = measures(smooth, u8(smooth), mask)
    assert a["band_contrast_s1"] > b["band_contrast_s1"] and a["sift_per_mpix"] > b["sift_per_mpix"]
    assert 0 < a["mask_frac"] <= 1 and a["spectral_tiles"] > 0 and b["spectral_slope"] > a["spectral_slope"]
    meta = {"intrinsics": {"K": [[500, 0, 400], [0, 500, 300], [0, 0, 1]], "dist_opencv": {}},
            "pose": {"R_world_to_cam": [[1, 0, 0], [0, 0, -1], [0, 1, 0]]}}      # looking north, level
    el = elevation_map(meta, (600, 800))
    assert abs(el[300, 400]) < 0.2 and el[0, 400] > 25 and el[-1, 400] < -25


def test_windows_bat_files():
    d = ROOT / "scripts" / "windows"
    for n in ("align_here.bat", "run_all_sites.bat", "sites_status.bat", "mppp_env.bat", "_run_align.bat",
              "_run_sites.bat"):
        raw = (d / n).read_bytes()
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), n     # CRLF line ends for cmd.exe
        assert raw.decode("ascii")
    assert b"align_scape.py" in (d / "_run_align.bat").read_bytes()
    assert b"run_sites.py" in (d / "_run_sites.bat").read_bytes()


def test_process_sites_dry_run(tmp_path, capsys):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("process_sites", root / "scripts" / "process_sites.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    assert ps.main(["--sites", "sid_chal_rocks", "rockytop", "--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "process_sites_log.txt").read_text()
    assert "sid_chal_rocks: would run" in log and "rockytop_colmap" in log
    assert ps.main(["--all", "--zcam", "--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "process_sites_log.txt").read_text()
    assert "rockytop_colmap_zcam34" in log and "van_zyl_colmap_zcam34" not in log      # v0p50: zcam34 sites only
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


def test_process_sites_refuses_while_a_batch_runs(tmp_path):
    import importlib.util
    from mppp.runner import batch_lock
    spec = importlib.util.spec_from_file_location("process_sites", ROOT / "scripts" / "process_sites.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    b = batch_lock(tmp_path, "run_sites")
    try:
        assert ps.main(["--sites", "sid_chal_rocks", "--root", str(tmp_path)]) == 3
        assert "not started" in (tmp_path / "process_sites_log.txt").read_text()
    finally:
        b.stop("finished")


def test_stop_runs_lists_nothing_here(capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stop_runs", ROOT / "scripts" / "stop_runs.py")
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    assert sr.main(["--list", "--root", "/nonexistent"]) == 0
    assert "MPPP run" in capsys.readouterr().out


def test_stop_mppp_bat():
    b = (ROOT / "scripts" / "windows" / "stop_mppp.bat").read_bytes()
    assert b.count(b"\n") == b.count(b"\r\n") and b"stop_runs.py" in b and b"mppp_env.bat" in b
    for name in ("process_sites.py", "run_sites.py"):
        assert "batch_lock(" in (ROOT / "scripts" / name).read_text()


def test_promote_cmods(monkeypatch, tmp_path, capsys):
    import importlib.util
    import shutil
    from mppp.sfm.project import NAVCAM_PACKAGE_CONSENSUS_DIR, navcam_consensus_dir
    dst = tmp_path / "cmods"
    monkeypatch.setenv("MPPP_CMODS", str(dst))
    spec = importlib.util.spec_from_file_location("promote_cmods", ROOT / "scripts" / "promote_cmods.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    assert pc.main([str(NAVCAM_PACKAGE_CONSENSUS_DIR), "--note", "first"]) == 0
    assert navcam_consensus_dir() == dst and (dst / "CHANGES.md").read_text().count("first") == 1
    # a new NL camera: the old one goes to history/
    new = tmp_path / "new"
    new.mkdir()
    d = json.loads((NAVCAM_PACKAGE_CONSENSUS_DIR / "M2020_NL_fisheye_tangential.json").read_text())
    d["params"][0] += 1.0
    (new / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(d))
    assert pc.main([str(new), "--note", "better"]) == 0
    assert json.loads((dst / "M2020_NL_fisheye_tangential.json").read_text())["params"][0] == d["params"][0]
    hist = list((dst / "history").glob("*/M2020_NL_fisheye_tangential.json"))
    assert len(hist) == 1
    # a broken file is refused and nothing changes
    (new / "M2020_N_rig.json").write_text("{}")
    before = (dst / "M2020_N_rig.json").read_bytes()
    assert pc.main([str(new / "M2020_N_rig.json")]) == 1
    assert (dst / "M2020_N_rig.json").read_bytes() == before
    # the focus model under any candidate name
    z = tmp_path / "M2020_ZCAM034_focus_model_candidate.json"
    shutil.copy(NAVCAM_PACKAGE_CONSENSUS_DIR / "M2020_ZCAM034_focus_model.json", z)
    assert pc.main([str(z)]) == 0 and (dst / "M2020_ZCAM034_focus_model.json").is_file()
    assert pc.main(["--list"]) == 0 and "focus model" in capsys.readouterr().out


def test_lmst_histogram(tmp_path):
    pytest.importorskip("matplotlib")
    import importlib.util
    spec = importlib.util.spec_from_file_location("lmst_histogram", ROOT / "scripts" / "lmst_histogram.py")
    lh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lh)

    def rec(stem, lmst, fam="N", ds=0.5, size=(2560, 1920), site=1, drive=10, sol=100):
        return {"source_product": f"/x/{stem}.IMG", "LMST": f"Sol-00100M{lmst}", "sol": sol, "site": site,
                "drive": drive, "native_size": list(size), "pose": {"C_enu_m": [drive * 1.0, 0.0, 0.0]},
                "filename": {"family": fam, "downsample_scale": ds, "stem": stem}}
    for root, folder, ver, recs in (
            ("old", "sid_colmap", "v0p35", [rec("A", "10:00:00"), rec("B", "11:00:00"), rec("OLDONLY", "12:00:00")]),
            ("new", "sid_colmap", "v0p43", [rec("A", "10:00:00"), rec("B", "11:00:00"), rec("C", "13:30:00", drive=11),
                                            rec("TILE", "14:00:00", ds=1.0, size=(1280, 960))]),
            ("new", "sid_colmap_nav_zcam34", "v0p43", [rec("Z1", "14:30:00", fam="Z", size=(1648, 1200)),
                                                       rec("A", "10:00:00")])):
        d = tmp_path / root / folder / "processed"
        d.mkdir(parents=True)
        (d / f"mppp_manifest_{ver}.json").write_text(json.dumps({"images": recs}))
    out = tmp_path / "out"
    assert lh.main(["--roots", str(tmp_path / "new"), str(tmp_path / "old"), "--out", str(out), "--name", "t"]) == 0
    assert (out / "t.png").stat().st_size > 10000
    import csv
    rows = list(csv.DictReader((out / "t_by_site.csv").open()))
    assert len(rows) == 1 and rows[0]["site"] == "sid"
    assert rows[0]["navcam"] == "3" and rows[0]["mastcam_z34"] == "1"      # A, B, C; OLDONLY and TILE left out
    assert rows[0]["stations"] == "2"
    assert lh.lmst_hours("Sol-01451M13:00:25.313") == pytest.approx(13.007, abs=1e-3)


def test_run_all_sites_bat_files_navcam_and_zcam34():
    """v0p50: run_all_sites.bat runs the Navcam blocks only, run_all_sites_zcam34.bat the zcam34 sites."""
    w = ROOT / "scripts" / "windows"
    nav = (w / "run_all_sites.bat").read_bytes()
    z = (w / "run_all_sites_zcam34.bat").read_bytes()
    assert b"\r\n" in nav and b"\r\n" in z
    assert b"--zcam" not in nav.split(b"start ")[-1] and b"%WHICH% --zcam" in z


def test_filter_zcam34_and_rename_script(tmp_path):
    import importlib.util
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sites
    table = {"sites": {"a": {"sols": [1, 2], "zcam34": True}, "b": {"sols": [3, 4], "zcam34": False}}}
    assert run_sites.filter_zcam34(["a", "b"], table, True, False) == (["a"], ["b"])
    assert run_sites.filter_zcam34(["a", "b"], table, False, False) == (["a", "b"], [])
    assert run_sites.filter_zcam34(["b"], table, True, True) == (["b"], [])
    old = tmp_path / "x_colmap_nav_zcam34"
    (old / "colmap").mkdir(parents=True)
    (old / "colmap" / "project.json").write_text(json.dumps({"processed_dir": str(old / "processed")}))
    spec = importlib.util.spec_from_file_location("rz", ROOT / "scripts" / "rename_zcam34_folders.py")
    rz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rz)
    assert rz.main(["--root", str(tmp_path)]) == 0 and old.is_dir()                  # dry run
    assert rz.main(["--root", str(tmp_path), "--apply"]) == 0 and not old.exists()
    pj = json.loads((tmp_path / "x_colmap_zcam34" / "colmap" / "project.json").read_text())
    assert pj["processed_dir"].endswith("x_colmap_zcam34" + ("\\" if "\\" in pj["processed_dir"] else "/") + "processed")


def test_experiment_scripts_live_in_studies():
    for name in ("lens_model_experiment.py", "lens_terms_experiment.py", "pair_experiment.py",
                 "temperature_bins_experiment.py", "error_sources.py", "run_v0p22_batch.bat"):
        assert not (ROOT / "scripts" / name).exists() and (ROOT / "studies" / "experiments" / name).is_file(), name
