"""Running notebooks without Jupyter: status, batches, stopping runs (mppp.runner). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
import pytest
import os


def test_runner_settings_and_keys(tmp_path):
    from mppp import runner as R
    v = R.parse_set(["A=3", "B=dict(x=1)", "C=rational", "D=[1, 2]", "E=((24,10,8),(8,2,2))", r"F=D:\scapes\x", "G=True"])
    assert v == {"A": 3, "B": "py:dict(x=1)", "C": "rational", "D": [1, 2], "E": [(24, 10, 8), (8, 2, 2)],
                 "F": "D:\\scapes\\x", "G": True}
    assert R.python_literal("NAVCAM_CAMERAS", "D:/x") == "Path(r'D:/x')"
    assert R.python_literal("MATCH", "py:dict(max_ratio=0.8)") == "dict(max_ratio=0.8)"
    assert R.python_literal("ATTITUDE_PRIOR_DEG", 1.0) == "1.0"
    k = R.run_key({"A": 1}, "processed", "")
    assert k == R.run_key({"A": 1}, "processed", "") and k != R.run_key({"A": 2}, "processed", "")
    assert k != R.run_key({"A": 1}, "pds", "") and k != R.run_key({"A": 1}, "processed", "v")
    work = tmp_path / "sid_colmap"
    work.mkdir()
    (work / "mppp_settings.json").write_text(json.dumps({"X": 1, "Y": 1}))
    f = tmp_path / "more.json"
    f.write_text(json.dumps({"Y": 2, "Z": 2}))
    assert R.work_settings(work, "sid", f, {"Z": 3}) == {"X": 1, "Y": 2, "Z": 3}


def test_status_heartbeat(tmp_path):
    from mppp import runner as R
    st = R.RunStatus(tmp_path / "s.json", work="w")
    st.update(state="running", cell=4)
    d = R.read_status(tmp_path / "s.json")
    assert d["alive"] and d["cell"] == 4
    d = json.loads((tmp_path / "s.json").read_text())
    d["heartbeat"] = "2020-01-01T00:00:00"
    (tmp_path / "s.json").write_text(json.dumps(d))
    d = R.read_status(tmp_path / "s.json")
    assert not d["alive"] and d["state"] == "stopped (no heartbeat)"
    st.stop("finished")
    assert R.read_status(tmp_path / "s.json")["state"] == "finished"
    assert R.read_status(tmp_path / "missing.json") is None


def test_notebook_choice_ignores_copies(tmp_path):
    from mppp.runner import notebook
    for n in ("03_colmap_alignment_v0p40.ipynb", "03_colmap_alignment_v0p43.ipynb",
              "03_colmap_alignment_v0p40 - Copy (2).ipynb", "03_colmap_alignment_v0p9.ipynb"):
        (tmp_path / n).write_text("{}")
    assert notebook("03_colmap_alignment", tmp_path).name == "03_colmap_alignment_v0p43.ipynb"
    (tmp_path / "03_colmap_alignment.ipynb").write_text("{}")
    assert notebook("03_colmap_alignment", tmp_path).name == "03_colmap_alignment.ipynb"


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


def test_pid_alive_and_dead_status(tmp_path):
    from mppp.runner import RunStatus, pid_alive, read_status
    assert pid_alive(os.getpid())
    assert not pid_alive(2 ** 22 + 12345)
    st = RunStatus(tmp_path / "s.json", pid=2 ** 22 + 12345)
    st.update(state="running")
    d = read_status(tmp_path / "s.json")
    assert d["alive"] is False and d["state"].startswith("stopped")


def test_batch_lock_refuses_a_second_batch(tmp_path):
    from mppp.runner import BATCH_FILE, BatchRunning, batch_lock, read_status
    b = batch_lock(tmp_path, "process_sites")
    try:
        with pytest.raises(BatchRunning, match="already running"):
            batch_lock(tmp_path, "run_sites")
    finally:
        b.stop("finished")
    assert read_status(tmp_path / BATCH_FILE)["alive"] is False
    b2 = batch_lock(tmp_path, "run_sites")                  # free again
    b2.stop("finished")


def test_mppp_processes_picks_the_tops_of_run_trees():
    from mppp.runner import mppp_processes
    rows = [
        {"pid": 10, "ppid": 1, "cmd": r'cmd /c call "D:\code\MPPP\scripts\windows\_run_process.bat" --root D:\x --all'},
        {"pid": 11, "ppid": 10, "cmd": r'python "D:\code\MPPP\scripts\process_sites.py" --root D:\x --all'},
        {"pid": 12, "ppid": 11, "cmd": r"python -m ipykernel_launcher -f kernel-1.json"},
        {"pid": 20, "ppid": 1, "cmd": r"python D:\code\MPPP\scripts\run_sites.py --all"},
        {"pid": 21, "ppid": 20, "cmd": r"python D:\code\MPPP\scripts\align_scape.py D:\x\sid_colmap"},
        {"pid": 30, "ppid": 1, "cmd": r"python -m ipykernel_launcher -f kernel-2.json"},        # Jupyter: untouched
        {"pid": 31, "ppid": 1, "cmd": r"jupyter-lab"},
        {"pid": 40, "ppid": 1, "cmd": r'cmd /c "D:\code\MPPP\scripts\windows\stop_mppp.bat"'},
    ]
    assert [r["pid"] for r in mppp_processes(rows)] == [10, 20]


def test_mark_stopped(tmp_path):
    from mppp.runner import BATCH_FILE, STATUS_FILE, RunStatus, mark_stopped, read_status
    (tmp_path / "sid_colmap").mkdir()
    a = RunStatus(tmp_path / BATCH_FILE, pid=4242)
    a.update(state="running")
    b = RunStatus(tmp_path / "sid_colmap" / STATUS_FILE, pid=os.getpid())
    b.update(state="running")
    changed = mark_stopped(tmp_path, [4242])
    assert changed == [str(tmp_path / BATCH_FILE)]
    assert read_status(tmp_path / BATCH_FILE)["state"] == "stopped"
    assert read_status(tmp_path / "sid_colmap" / STATUS_FILE)["alive"] is True


def test_mppp_processes_never_includes_its_own_ancestors():
    from mppp.runner import mppp_processes
    me, parent = os.getpid(), os.getppid()
    rows = [{"pid": parent, "ppid": 1, "cmd": "bash -c 'python scripts/stop_runs.py; python scripts/process_sites.py'"},
            {"pid": me, "ppid": parent, "cmd": "python scripts/stop_runs.py"},
            {"pid": 999999, "ppid": 1, "cmd": "python scripts/process_sites.py --all"}]
    assert [r["pid"] for r in mppp_processes(rows)] == [999999]


def test_log_writes_the_file_before_the_console(tmp_path, monkeypatch):
    """v0p50: a blocked console (Windows QuickEdit selection) must not hold the log file back."""
    import builtins
    from mppp.runner import Log, disable_quickedit
    order = []
    real = builtins.print
    monkeypatch.setattr(builtins, "print", lambda *a, **k: order.append(("print", (tmp_path / "l.txt").is_file())))
    Log(tmp_path / "l.txt")("hello")
    monkeypatch.setattr(builtins, "print", real)
    assert order == [("print", True)] and "hello" in (tmp_path / "l.txt").read_text()
    assert disable_quickedit() in (True, False)
