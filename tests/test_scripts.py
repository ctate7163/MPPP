"""Command-line scripts and Windows .bat files (scripts/). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
from pathlib import Path
import numpy as np
import pytest
import sys
import json


ROOT = Path(__file__).resolve().parents[1]




def test_windows_bat_files():
    d = ROOT / "scripts" / "windows"
    for n in ("align_here.bat", "run_sites.bat", "sites_status.bat", "mppp_env.bat", "_run_align.bat",
              "_run_sites.bat"):
        raw = (d / n).read_bytes()
        assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), n     # CRLF line ends for cmd.exe
        assert raw.decode("ascii")
    assert b"align_scape.py" in (d / "_run_align.bat").read_bytes()
    assert b"run_sites.py" in (d / "_run_sites.bat").read_bytes()


def _rocky(zcam=False):
    from mppp.waypoints.sites import folder_name, load_sites
    return folder_name("rockytop", load_sites()["rockytop"][0], zcam=zcam)


def test_process_sites_dry_run(tmp_path, capsys):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("process_sites", root / "scripts" / "process_sites.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    assert ps.main(["--sites", "chal_rocks", "rockytop", "--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "process_sites_log.txt").read_text()
    assert "chal_rocks: would run" in log and _rocky() in log
    assert ps.main(["--all", "--zcam", "--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "process_sites_log.txt").read_text()
    assert _rocky(True) in log and "van_zyl_colmap_zcam" not in log   # v0p53: "zcam" sites only
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
    for name in ("process_sites.bat", "run_sites.bat", "sites_status.bat"):
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


def _batch():
    sys.path.insert(0, str(ROOT / "scripts"))
    import _batch
    return _batch


def test_process_sites_refuses_while_a_batch_runs(tmp_path):
    import importlib.util
    batch_lock = _batch().batch_lock
    spec = importlib.util.spec_from_file_location("process_sites", ROOT / "scripts" / "process_sites.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    b = batch_lock(tmp_path, "run_sites")
    try:
        assert ps.main(["--sites", "chal_rocks", "--root", str(tmp_path)]) == 3
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
    for name in ("align_scape.py", "process_sites.py", "run_sites.py", "stop_runs.py", "check_sites.py", "_batch.py"):
        src = (ROOT / "scripts" / name).read_text()
        assert "nbclient" not in src and "run_notebook" not in src and "ipynb" not in src, name


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
    # a new NL camera replaces the old one; the change is logged (git history keeps the replaced file)
    new = tmp_path / "new"
    new.mkdir()
    d = json.loads((NAVCAM_PACKAGE_CONSENSUS_DIR / "M2020_NL_fisheye_tangential.json").read_text())
    d["params"][0] += 1.0
    (new / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(d))
    assert pc.main([str(new), "--note", "better"]) == 0
    assert json.loads((dst / "M2020_NL_fisheye_tangential.json").read_text())["params"][0] == d["params"][0]
    assert not (dst / "history").exists() and "better" in (dst / "CHANGES.md").read_text()
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




def test_run_sites_bat_files_navcam_and_zcam34():
    """v0p50: run_sites.bat runs the Navcam blocks only, run_sites_zcam34.bat the zcam34 sites (v0p61: no "all")."""
    w = ROOT / "scripts" / "windows"
    assert not list(w.glob("run_all_sites*.bat"))
    nav = (w / "run_sites.bat").read_bytes()
    z = (w / "run_sites_zcam34.bat").read_bytes()
    assert b"\r\n" in nav and b"\r\n" in z
    assert b"--zcam" not in nav.split(b"start ")[-1] and b"%WHICH% --zcam" in z
    # 20 Oct: the Mastcam-Z joint after the Z34 blocks (MPPP_THEN, run by _run_sites.bat when the batch ran)
    assert b'set "MPPP_THEN=zcam_tied_joint.bat"' in z and b"MPPP_THEN" not in nav
    j = (w / "zcam_tied_joint.bat").read_bytes()                       # 21 Oct: the combined N+Z check last
    assert j.index(b"calib zcam") < j.index(b"calib nz") and b"--promote" not in j.split(b"calib nz")[1]
    r = (w / "_run_sites.bat").read_bytes()
    assert b"if defined MPPP_THEN if %RC% LEQ 1" in r and b"%MPPP_THEN_ARGS%" in r
    # the scapes root is checked after mppp_env.bat (mppp_scapes.txt), with D:\scapes as the fallback
    for f in (nav, z, (w / "stop_mppp.bat").read_bytes()):
        assert f.index(b'mppp_env.bat"') < f.index(b"if not defined MPPP_SCAPES if exist")


def test_filter_zcam_and_consensus_bats(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sites
    table = {"sites": {"a": {"sols": [1, 2], "zcam": [34]}, "b": {"sols": [3, 4], "zcam": []},
                       "c": {"sols": [5, 6], "zcam34": True}}}
    assert run_sites.filter_zcam(["a", "b", "c"], table, True, False) == (["a", "c"], ["b"])
    assert run_sites.filter_zcam(["a", "b"], table, False, False) == (["a", "b"], [])
    assert run_sites.filter_zcam(["b"], table, True, True) == (["b"], [])
    assert not (ROOT / "scripts" / "rename_zcam34_folders.py").exists()                 # v0p53: no renaming
    win = ROOT / "scripts" / "windows"
    assert 'set "WHICH=--group navcam_consensus"' in (win / "run_sites.bat").read_text()
    t = (win / "run_sites_zcam.bat").read_text()           # v0p53: every Mastcam-Z consensus block
    assert 'set "WHICH=--group zcam_consensus"' in t and "--zcam" in t
    assert 'set "EXTRA="' in t and "ZCAM_ZOOMS" not in t               # v0p53.1: the sites' own zoom lists
    for z in (34,):                     # v0p53: each runs its consensus group (v0p85: 48, 63, 79, 110 removed)
        t = (win / f"run_sites_zcam{z}.bat").read_text()
        assert f'set "WHICH=--group zcam{z}_consensus"' in t and "--zcam" in t, z
        assert f'set "EXTRA=--set ZCAM_ZOOMS=[{z}]"' in t, z              # v0p61: only that zoom
        assert "--all" not in t, z


def test_batch_lock_refuses_a_second_batch(tmp_path):
    from mppp.runner import read_status
    B = _batch()
    b = B.batch_lock(tmp_path, "process_sites")
    try:
        with pytest.raises(B.BatchRunning, match="already running"):
            B.batch_lock(tmp_path, "run_sites")
    finally:
        b.stop("finished")
    assert read_status(tmp_path / B.BATCH_FILE)["alive"] is False
    b2 = B.batch_lock(tmp_path, "run_sites")                  # free again
    b2.stop("finished")


def test_mppp_processes_picks_the_tops_of_run_trees():
    mppp_processes = _batch().mppp_processes
    rows = [
        {"pid": 10, "ppid": 1, "cmd": r'cmd /c call "D:\code\MPPP\scripts\windows\_run_process.bat" --root D:\x --all'},
        {"pid": 11, "ppid": 10, "cmd": r'python "D:\code\MPPP\scripts\process_sites.py" --root D:\x --all'},
        {"pid": 12, "ppid": 11, "cmd": r"python -m mppp.process_worker"},
        {"pid": 20, "ppid": 1, "cmd": r"python D:\code\MPPP\scripts\run_sites.py --all"},
        {"pid": 21, "ppid": 20, "cmd": r"python D:\code\MPPP\scripts\align_scape.py D:\x\sid_colmap"},
        {"pid": 30, "ppid": 1, "cmd": r"python -m ipykernel_launcher -f kernel-2.json"},        # Jupyter: untouched
        {"pid": 31, "ppid": 1, "cmd": r"jupyter-lab"},
        {"pid": 40, "ppid": 1, "cmd": r'cmd /c "D:\code\MPPP\scripts\windows\stop_mppp.bat"'},
        {"pid": 50, "ppid": 1, "cmd": r"mppp align D:\x\mars2020_sol_0460_rockytop"},
    ]
    assert [r["pid"] for r in mppp_processes(rows)] == [10, 20, 50]


def test_mppp_processes_never_includes_its_own_ancestors():
    import os
    mppp_processes = _batch().mppp_processes
    me, parent = os.getpid(), os.getppid()
    rows = [{"pid": parent, "ppid": 1, "cmd": "bash -c 'python scripts/stop_runs.py; python scripts/process_sites.py'"},
            {"pid": me, "ppid": parent, "cmd": "python scripts/stop_runs.py"},
            {"pid": 999999, "ppid": 1, "cmd": "python scripts/process_sites.py --all"}]
    assert [r["pid"] for r in mppp_processes(rows)] == [999999]


def test_mark_stopped(tmp_path):
    import os
    from mppp.runner import STATUS_FILE, RunStatus, read_status
    B = _batch()
    (tmp_path / "sid_colmap").mkdir()
    a = RunStatus(tmp_path / B.BATCH_FILE, pid=4242)
    a.update(state="running")
    b = RunStatus(tmp_path / "sid_colmap" / STATUS_FILE, pid=os.getpid())
    b.update(state="running")
    changed = B.mark_stopped(tmp_path, [4242])
    assert changed == [str(tmp_path / B.BATCH_FILE)]
    assert read_status(tmp_path / B.BATCH_FILE)["state"] == "stopped"
    assert read_status(tmp_path / "sid_colmap" / STATUS_FILE)["alive"] is True


def test_run_sites_dry_run_and_finished(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("run_sites", ROOT / "scripts" / "run_sites.py")
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    assert rs.main(["--sites", "rockytop", "--root", str(tmp_path), "--dry-run"]) == 0
    log = (tmp_path / "run_sites_log.txt").read_text()
    assert "rockytop: would run (not finished)" in log and "camera models fingerprint" in log
    assert rs.main(["--sites", "rockytop", "--root", str(tmp_path), "--source", "processed", "--dry-run"]) == 0
    assert "has no processed/" in (tmp_path / "run_sites_log.txt").read_text()
    # a finished folder (run_done.json with the run key of these settings) is skipped
    from mppp.pipeline import Settings, load_settings, resolve_site, run_key
    st, s = resolve_site("rockytop", Settings(scapes_root=tmp_path))
    (st.work / "processed").mkdir(parents=True)
    (st.work / "colmap").mkdir()
    s = load_settings(st.work, "rockytop", base=Settings(scapes_root=tmp_path)).replace(source="processed")
    key = run_key(resolve_site(st.work, s)[1])                  # the key is of the resolved settings
    (st.work / "colmap" / "run_done.json").write_text(json.dumps({"run_key": key, "verdict": "pass", "finished": "t"}))
    assert rs.main(["--sites", "rockytop", "--root", str(tmp_path), "--source", "processed", "--dry-run"]) == 0
    assert "rockytop: skipped - finished t with these settings" in (tmp_path / "run_sites_log.txt").read_text()
    assert rs.main(["--sites", "rockytop", "--root", str(tmp_path), "--source", "processed", "--dry-run",
                    "--set", "attitude_prior_deg=1.0"]) == 0
    assert "finished with other settings" in (tmp_path / "run_sites_log.txt").read_text()


def test_experiment_scripts_live_in_studies():
    for name in ("lens_model_experiment.py", "lens_terms_experiment.py", "pair_experiment.py",
                 "temperature_bins_experiment.py", "error_sources.py"):
        assert not (ROOT / "scripts" / name).exists() and (ROOT / "studies" / "experiments" / name).is_file(), name


def test_windows_run_sites_zcam_passes_extra_through():
    z = (ROOT / "scripts" / "windows" / "run_sites_zcam.bat").read_text()
    assert 'set "EXTRA="' in z and "ZCAM_ZOOMS" not in z.split("set \"EXTRA=")[1].splitlines()[0]


def test_scripts_and_bat_exist():
    assert (ROOT / "studies" / "calibration" / "prepare_next_run.py").is_file()      # 0.90: a calibration study
    assert not (ROOT / "scripts" / "prepare_next_run.py").exists()
    assert not (ROOT / "scripts" / "windows" / "prepare_next_run.bat").exists()


def test_stop_runs_knows_the_pipeline(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from _batch import mark_stopped, mppp_processes
    rows = [{"pid": 10, "ppid": 1, "cmd": "python scripts/run_pipeline.py --root D:/x"},
            {"pid": 11, "ppid": 10, "cmd": "python scripts/run_sites.py --root D:/x --group zcam34_consensus"},
            {"pid": 12, "ppid": 11, "cmd": "python scripts/align_scape.py D:/x/site"}]
    top = mppp_processes(rows)
    assert [r["pid"] for r in top] == [10]                                  # the pipeline is the top of the tree
    d = tmp_path / "mppp_pipeline"
    d.mkdir()
    (d / "pipeline_status.json").write_text(json.dumps({"state": "running", "pid": 10, "host": "x"}))
    assert mark_stopped(tmp_path, [10]) == [str(d / "pipeline_status.json")]
    assert json.loads((d / "pipeline_status.json").read_text())["state"] == "stopped"


def test_skip_file_leaves_a_block_out(tmp_path):
    """21 Oct: a SKIP file in a WORK folder takes the block out of a batch, also of one already running (its
    align_scape child exits 5 before doing anything)."""
    import subprocess
    import sys
    work = tmp_path / "mars2020_sol_1880_olifants_zcam"
    work.mkdir()
    (work / "SKIP").write_text("taken out 21 Oct")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "align_scape.py"), str(work)], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 5 and "skipped" in r.stdout
    src = (ROOT / "scripts" / "run_sites.py").read_text()
    assert '(work / "SKIP").is_file()' in src and "p.returncode == 5" in src


def test_multizoom_test_bat():
    """22 Oct: the multi-zoom test block (Taylorfjellet, all five zooms) in its own variant."""
    b = (ROOT / "scripts" / "windows" / "run_multizoom_test.bat").read_bytes()
    assert b"\r\n" in b and b"--variant multizoom" in b and b"ZCAM_ZOOMS=[34,48,63,79,110]" in b
