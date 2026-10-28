"""Command-line scripts and Windows .bat files (scripts/). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
from pathlib import Path
import os
import numpy as np
import pytest
import sys
import json


ROOT = Path(__file__).resolve().parents[1]




def test_windows_bat_files():
    d = ROOT / "scripts" / "windows"
    for n in ("align_here.bat", "run_sites.bat", "check_sites.bat", "mppp_env.bat", "_run_align.bat",
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
    for name in ("process_sites.bat", "run_sites.bat", "check_sites.bat"):
        s = (win / name).read_text()
        assert 'call "%MPPP_WIN%\\mppp_env.bat"' in s and '"%~dp0mppp_env.bat" ||' not in s, name
        assert "%~dp0_run" not in s, name
    env = (win / "mppp_env.bat").read_text()
    for part in ("check_env.py", "environments.txt", "mppp_python.txt", "MPPP_PYTHON", ":try_prefix", ":try_root",
                 "KMP_DUPLICATE_LIB_OK"):
        assert part in env, part
    body = env.split(":try_prefix\n")[1]
    assert "%~dp0" not in body                     # inside a call :label, %0 is the label
    r = subprocess.run([sys.executable, str(win.parent / "check_env.py")], capture_output=True, text=True)   # 0.94.4: scripts\
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
    assert navcam_consensus_dir() == dst and (dst / "CHANGES.md").read_text().count(" - first\n") == 1
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




def test_run_sites_bat_files_navcam_and_zcam():
    """v0p50: run_sites.bat runs the Navcam blocks only, run_sites_zcam.bat the Mastcam-Z blocks (v0p61: no "all";
    0.94.4: run_sites_zcam34.bat and zcam_tied_joint.bat removed)."""
    w = ROOT / "scripts" / "windows"
    assert not list(w.glob("run_all_sites*.bat"))
    nav = (w / "run_sites.bat").read_bytes()
    z = (w / "run_sites_zcam.bat").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")   # crlf on checkout
    assert b"\r\n" in nav
    assert b"--zcam" not in nav.split(b"start ")[-1] and b"%WHICH% --zcam" in z
    # the Mastcam-Z joints after the Mastcam-Z blocks (MPPP_THEN, run by _run_sites.bat when the batch ran)
    assert b'set "MPPP_THEN=zcam_joints.bat"' in z and b"MPPP_THEN" not in nav
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
    assert 'set "WHICH=--group ecam_consensus"' in (win / "run_sites.bat").read_text()
    t = (win / "run_sites_zcam.bat").read_text()           # v0p53: every Mastcam-Z consensus block
    assert 'set "WHICH=--group zcam_consensus"' in t and "--zcam" in t
    assert 'set "EXTRA="' in t and "ZCAM_ZOOMS" not in t               # v0p53.1: the sites' own zoom lists
    assert not list(win.glob("run_sites_zcam[0-9]*.bat"))     # 0.94.4: the single-zoom batches removed (v0p85: 48-110)


def test_batch_lock_refuses_a_second_batch(tmp_path):
    from mppp.runner import read_status
    B = _batch()
    b = B.batch_lock(tmp_path, "process_sites")
    try:
        with pytest.raises(B.BatchRunning, match="already running"):
            B.batch_lock(tmp_path, "run_sites")
    finally:
        b.stop("finished")
    assert read_status(tmp_path / B.BATCH_DIR / f"process_sites_{os.getpid()}.json")["alive"] is False    # 0.95.0: its own file
    assert not (tmp_path / B.BATCH_FILE).exists()
    b2 = B.batch_lock(tmp_path, "run_sites")                  # free again
    b2.stop("finished")


def test_batches_side_by_side_on_other_sites(tmp_path):
    """0.95.0 (Christian, 27 Oct): a one-site run beside the pipeline - each batch has its own status file; a
    second batch is refused only when it shares a site with a running one (or either site set is unknown)."""
    B = _batch()
    a = B.batch_lock(tmp_path, "run_sites", sites=["chal_rocks", "rochette"])
    try:
        assert len(B.alive_batches(tmp_path)) == 1 and B.alive_batches(tmp_path)[0]["sites"] == ["chal_rocks", "rochette"]
        b = B.batch_lock(tmp_path, "run_sites", sites=["belva_crater"])             # other sites: fine
        try:
            assert len(B.alive_batches(tmp_path)) == 2
            with pytest.raises(B.BatchRunning, match="shares sites with this one \\(rochette\\)"):
                B.batch_lock(tmp_path, "run_sites", sites=["rochette", "south_arm"])
            with pytest.raises(B.BatchRunning):                                    # an unknown site set: as before
                B.batch_lock(tmp_path, "run_sites")
            c = B.batch_lock(tmp_path, "run_sites", sites=["rochette"], force=True)
            c.stop("finished")
        finally:
            b.stop("finished")
    finally:
        a.stop("finished")
    assert B.alive_batches(tmp_path) == []
    # the old one-file batch still counts while alive
    from mppp.runner import RunStatus
    old = RunStatus(tmp_path / B.BATCH_FILE, pid=os.getpid())
    old.update(state="running")
    assert [x["file"] for x in B.alive_batches(tmp_path)] == [str(tmp_path / B.BATCH_FILE)]


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
    (tmp_path / B.BATCH_DIR).mkdir()
    a2 = RunStatus(tmp_path / B.BATCH_DIR / "run_sites_4243.json", pid=4243)       # 0.95.0: the per-batch files too
    a2.update(state="running")
    b = RunStatus(tmp_path / "sid_colmap" / STATUS_FILE, pid=os.getpid())
    b.update(state="running")
    changed = B.mark_stopped(tmp_path, [4242, 4243])
    assert changed == [str(tmp_path / B.BATCH_FILE), str(tmp_path / B.BATCH_DIR / "run_sites_4243.json")]
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


def test_windows_folder_is_short():
    """0.94.4 (Christian, 23 Oct): at most 20 files in scripts\\windows, the removed ones gone."""
    w = ROOT / "scripts" / "windows"
    files = [f for f in w.iterdir() if f.is_file() and not f.name.endswith(".txt")]
    assert len(files) <= 20, sorted(f.name for f in files)
    for gone in ("run_weekend.bat", "build_example_data.bat", "run_multizoom_test.bat", "run_hazcam.bat",
                 "test_lightglue.bat", "sites_status.bat", "run_sites_zcam34.bat", "zcam_tied_joint.bat"):
        assert not (w / gone).exists(), gone
    # every batch file a kept one calls is still there
    import re
    names = {f.name.lower() for f in files}
    for f in files:
        if f.suffix == ".bat":
            for ref in re.findall(r"%MPPP_WIN%\\(\w+\.bat)", f.read_text(errors="replace")):
                assert ref.lower() in names, (f.name, ref)


def test_check_sites_shows_the_last_runs(tmp_path, capsys):
    """0.94.4 (Christian): check_sites lists when each block last finished and with which MPPP version."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("check_sites", ROOT / "scripts" / "check_sites.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    from mppp.waypoints.sites import load_site_table, work_folder
    t = load_site_table()["sites"]
    n = work_folder(tmp_path, "rockytop", False, t["rockytop"]["sols"])
    z = work_folder(tmp_path, "rockytop", True, t["rockytop"]["sols"])
    for col, when, ver in (("colmap", "2026-10-21T09:00:00", "0.93.4"), ("colmap_lightglue", "2026-10-23T11:55:55", "0.94.0")):
        (z / col).mkdir(parents=True)
        (z / col / "run_done.json").write_text(json.dumps({"finished": when, "mppp": ver, "verdict": "warn"}))
    (n / "colmap").mkdir(parents=True)
    assert cs.last_run(n) == "not finished"
    assert cs.last_run(z) == "23 Oct 11:55 0.94.0 lightglue warn"          # the newest of the variants
    assert cs.last_run(tmp_path / "nothing") == "-"
    (n / "mppp_status.json").write_text(json.dumps({"state": "running", "pid": os.getpid(),
                                                    "started": "2026-10-23T12:10:16", "mppp": "0.94.4"}))
    assert cs.last_run(n) == "running since 10-23 12:10 (0.94.4)"
    assert cs.main(["--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.strip().startswith("rockytop ")][0]
    assert "N: running since" in line and "Z: 23 Oct 11:55 0.94.0 lightglue warn" in line


def test_archive_gaps_finds_and_fetches_the_missing_eye(tmp_path):
    """0.94.4: a right Hazcam (J01) whose left image is not in the archive; the PDS has both eyes as J02."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("archive_gaps", ROOT / "scripts" / "archive_gaps.py")
    ag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ag)
    d = tmp_path / "00122" / "ids" / "rdr" / "fcam"
    d.mkdir(parents=True)
    (d / "FRF_0122_0677784010_631RAD_N0041422FHAZ02420_0A0295J01.IMG").write_bytes(b"r")
    e = tmp_path / "00116" / "ids" / "rdr" / "fcam"
    e.mkdir(parents=True)
    for n in ("FLF_0116_0677245019_226RAD_N0041250FHAZ02420_0A0295J01.IMG",
              "FRF_0116_0677245019_226RAD_N0041250FHAZ02420_0A0295J01.IMG"):
        (e / n).write_bytes(b"x")
    rows = ag.gaps(ag.rad_products(tmp_path))
    assert len(rows) == 1 and rows[0]["missing"] == "FL" and rows[0]["prefix"] == "FLF_0122_0677784010_631RAD_"
    assert ag.listing_url("F", 122).endswith("/mars2020_hazcam_ops_calibrated/data/sol/00122/ids/rdr/fcam/")
    html = ('<a href="FLF_0122_0677784010_631RAD_N0041422FHAZ02420_0A0295J02.IMG">x</a>'
            '<a href="FLF_0122_0677784010_631RAD_N0041422FHAZ02420_0A0295J02.xml">x</a>'
            '<a href="FLF_0122_0677784010_631RASLN_N0041422FHAZ02420_0A0295J02.IMG">x</a>'
            '<a href="FRF_0122_0677784010_631RAD_N0041422FHAZ02420_0A0295J02.IMG">x</a>')
    calls = []

    def get(url):
        calls.append(url)
        return html.encode() if url.endswith("/") else b"IMG"
    assert ag.download(rows, get=get, log=lambda *_: None) == 1
    got = d / "FLF_0122_0677784010_631RAD_N0041422FHAZ02420_0A0295J02.IMG"
    assert got.read_bytes() == b"IMG" and len(calls) == 2
    assert ag.download(rows, get=get, log=lambda *_: None) == 0          # never twice


def test_tie_point_studies_scaffold(tmp_path, capsys):
    """0.95.0 (Christian, 28 Oct): the studies S1-S6 as variant sets run through ``mppp run --variant``, measured
    the same way (completeness, quality, runtime per step from the log) and compared (S2 against the all-points
    solution, S6 the per-bin focal residuals by observation band)."""
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("tps", ROOT / "scripts" / "tie_point_studies.py")
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    assert set(T.STUDIES) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    for st, vs in T.STUDIES.items():
        assert next(iter(vs.values())) == {}                                  # the first variant: today's default
    from mppp.pipeline import SETTING_NAMES, normalize
    from mppp.runner import parse_set
    for st, vs in T.STUDIES.items():
        for v, settings in vs.items():
            sets = T._sets(settings)
            assert len(sets) == 2 * len(settings)
            parsed = normalize(parse_set(sets[1::2]))                          # every setting is a real one
            assert set(parsed) == {k.lower() for k in settings} <= set(SETTING_NAMES)
    assert T.STUDIES["S2"]["cap1500"] == {"ba_points_cap": 1500} and T.variant_name("S2", "cap1500") == "s2_cap1500"
    assert T.main(["--list"]) == 0 and "S2" in capsys.readouterr().out
    log = tmp_path / "log.txt"
    log.write_text("2026-10-28T10:00:00 | database: x\n2026-10-28T10:10:00 | matching: y\n"
                   "2026-10-28T10:12:00 stage 2: 5 frames\n2026-10-28T10:40:00 stage 2, pose-guided: z\n"
                   "2026-10-28T10:50:00 stage 2: 3 isolated two-view points deleted\n"
                   "2026-10-28T11:20:00 | export: done\n", encoding="utf-8")
    s = T.step_seconds(log)
    assert s["matching"] == 600 and s["stage 1"] == 120 and s["stage 2 rounds"] == 28 * 60
    assert s["final adjustment"] == 30 * 60 and s["database to export"] == 80 * 60
    ref = {"model": {"centres": {"a": [0, 0, 0], "b": [1, 0, 0]}, "attitudes": {"a": np.eye(3).tolist(), "b": np.eye(3).tolist()},
                     "cameras": {"ZL034_F1": [4700, 4700, 800, 600], "NL": [1000, 1000, 0, 0]}}}
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(np.radians([0, 0, 0.002])).as_matrix().tolist()
    oth = {"model": {"centres": {"a": [0.005, 0, 0], "b": [1, 0.02, 0]}, "attitudes": {"a": R, "b": np.eye(3).tolist()},
                     "cameras": {"ZL034_F1": [4702.35, 4702.35, 800, 600], "NL": [1000, 1000, 0, 0]}}}
    c = T.compare_solutions(ref, oth)
    assert c["centre_cm"]["max"] == pytest.approx(2.0) and c["centre_cm"]["median"] == pytest.approx(1.25)
    assert c["attitude_mdeg"]["max"] == pytest.approx(2.0, abs=1e-6) and c["focal_pct"]["max"] == pytest.approx(0.05, abs=1e-6)
    bins = [{"param": "fx", "residual": 2.0, "observations": 300}, {"param": "fy", "residual": -0.5, "observations": 3000},
            {"param": "k1", "residual": 9.0, "observations": 10}]
    tab = T.s6_table([{"bins": bins}])
    assert any(ln.startswith("| 250-500 | 1 | 2.00") for ln in tab) and any(ln.startswith("| 2000-5000 | 1 | 0.50") for ln in tab)
    assert T.bin_rms(bins) == pytest.approx(np.sqrt((4 + 0.25) / 2))
    vs = {"all": {"model": ref["model"], "checks": {"cross_station_fraction": {"value": 0.03, "status": "OK"}},
                  "steps_s": {"matching": 600}, "verdict": "OK", "bins": bins, "run_done": {"registered_images": 2, "images": 2}},
          "cap1500": {"model": oth["model"], "checks": {}, "steps_s": {}, "verdict": "WARN", "bins": [], "run_done": {}}}
    md = T.report_site(tmp_path, "S2", {"when": "now", "variants": vs})
    assert "| all | cap1500 |" in md and "Against the all-points solution" in md and "2 / 2" in md
    md6 = T.report_site(tmp_path, "S6", {"when": "now", "variants": {"default": vs["all"]}})
    assert "per-bin focal residual about the lines vs observations" in md6
    assert "| S2 (" in T.report_all(tmp_path, "S2", {"x": {"variants": vs, "report": "r.md"}}) or "# S2" in T.report_all(tmp_path, "S2", {"x": {"variants": vs, "report": "r.md"}})


def test_trial_joints_bat_runs_every_0p95_joint_without_promoting():
    """0.95.0 (Christian, 28 Oct): the trial of the 0.95.0 joints over the blocks aligned on 0.94 - the Navcam joint
    twice (with and without --tie-f-slope), the law check of the candidate, the Hazcam fit, the Mastcam-Z joints of
    34, 48 and 63 mm, the N+Z check on request; nothing promoted, no waiting for running blocks."""
    b = (ROOT / "scripts" / "windows" / "trial_joints_0p95.bat").read_bytes()
    assert b"\r\n" in b and b.count(b"\n") == b.count(b"\r\n")
    t = b.decode("ascii")
    assert t.count("calib navcam") == 2 and "--tie-f-slope" in t and "calib lawcheck" in t and "calib hazcam" in t
    assert "calib zcam --zoom %%Z" in t and 'MPPP_ZOOMS=34 48 63' in t and "calib nz" in t
    assert "--promote-if-ok" not in t and t.count("--no-promote") == 4 and t.count("--wait-h 0") == 7   # nz never promotes
    assert "mppp_env.bat" in t and "trial_0p95_" in t and "plot_lawcheck.py" in t
