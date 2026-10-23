"""Logged runs of the pipeline functions: status files, heartbeats, the log file, --set parsing (mppp.runner)."""
import json
import logging
import os
from pathlib import Path

import pytest


def test_parse_set_and_python_literal():
    from mppp import runner as R
    v = R.parse_set(["A=3", "B=dict(x=1)", "C=rational", "D=[1, 2]", "E=((24,10,8),(8,2,2))", r"F=D:\scapes\x", "G=True"])
    assert v == {"A": 3, "B": "py:dict(x=1)", "C": "rational", "D": [1, 2], "E": [(24, 10, 8), (8, 2, 2)],
                 "F": "D:\\scapes\\x", "G": True}
    with pytest.raises(ValueError, match="NAME=VALUE"):
        R.parse_set(["novalue"])
    assert R.python_literal("navcam_cameras", "D:/x") == "Path(r'D:/x')"
    assert R.python_literal("MATCH", "py:dict(max_ratio=0.8)") == "dict(max_ratio=0.8)"
    assert R.python_literal("ATTITUDE_PRIOR_DEG", 1.0) == "1.0"


def test_status_heartbeat(tmp_path):
    from mppp import runner as R
    st = R.RunStatus(tmp_path / "s.json", work="w")
    st.update(state="running", step="features")
    d = R.read_status(tmp_path / "s.json")
    assert d["alive"] and d["step"] == "features"
    d = json.loads((tmp_path / "s.json").read_text())
    d["heartbeat"] = "2020-01-01T00:00:00"
    (tmp_path / "s.json").write_text(json.dumps(d))
    d = R.read_status(tmp_path / "s.json")
    assert not d["alive"] and d["state"] == "stopped (no heartbeat)"
    st.stop("finished")
    assert R.read_status(tmp_path / "s.json")["state"] == "finished"
    assert R.read_status(tmp_path / "missing.json") is None


def test_pid_alive_and_dead_status(tmp_path):
    from mppp.runner import RunStatus, pid_alive, read_status
    assert pid_alive(os.getpid())
    assert not pid_alive(2 ** 22 + 12345)
    st = RunStatus(tmp_path / "s.json", pid=2 ** 22 + 12345)
    st.update(state="running")
    d = read_status(tmp_path / "s.json")
    assert d["alive"] is False and d["state"].startswith("stopped")


def test_log_writes_the_file_before_the_console(tmp_path, monkeypatch):
    """A blocked console (Windows QuickEdit selection) must not hold the log file back."""
    import builtins
    from mppp.runner import Log, disable_quickedit
    order = []
    real = builtins.print
    monkeypatch.setattr(builtins, "print", lambda *a, **k: order.append(("print", (tmp_path / "l.txt").is_file())))
    Log(tmp_path / "l.txt")("hello")
    monkeypatch.setattr(builtins, "print", real)
    assert order == [("print", True)] and "hello" in (tmp_path / "l.txt").read_text()
    assert disable_quickedit() in (True, False)


def test_run_logged_writes_log_status_and_runs_txt(tmp_path):
    from mppp.runner import STATUS_FILE, RunAlive, read_status, run_logged

    def step(work, settings, stop_after=None):
        logging.getLogger("mppp.pipeline").info("working on %s", Path(work).name)
        return {"verdict": "pass", "run_done": {"verdict": "pass", "images": 3, "run_key": "k"}, "tables": [1, 2]}

    work = tmp_path / "mars2020_sol_0460_rockytop"
    st = run_logged(step, work, {"a": 1}, stage="align", key="k", variant="v", echo=False, stop_after=None)
    assert st["state"] == "finished" and st["result"] == {"verdict": "pass", "images": 3, "run_key": "k"}
    assert st["last_error"] is None and st["variant"] == "v"
    log = Path(st["log"]).read_text(encoding="utf-8")
    assert "working on mars2020_sol_0460_rockytop" in log and "finished" in log and "_v" in Path(st["log"]).parent.name
    assert read_status(work / STATUS_FILE)["state"] == "finished"
    runs = (work / "runs" / "runs.txt").read_text()
    assert "finished" in runs and "variant=v" in runs and "verdict=pass" in runs

    def boom(work, settings):
        raise ValueError("no images")

    st = run_logged(boom, work, None, stage="process", key="p", echo=False)
    assert st["state"] == "failed" and st["last_error"] == "ValueError: no images"
    assert "_process" in Path(st["log"]).parent.name and "FAILED: ValueError: no images" in Path(st["log"]).read_text()
    # a live run of the folder refuses a second one
    alive = json.loads((work / STATUS_FILE).read_text())
    alive.update(state="running", pid=os.getpid())
    from mppp.runner import now
    alive["heartbeat"] = now()
    (work / STATUS_FILE).write_text(json.dumps(alive))
    with pytest.raises(RunAlive, match="still alive"):
        run_logged(step, work, None, echo=False)
    assert run_logged(step, work, None, echo=False, force=True)["state"] == "finished"


def test_set_value_with_a_plus_is_a_word():
    """0.94.4: --set MATCH_MODE=sift+lightglue is the string, not Python source (the A/B variant failed at once)."""
    from mppp.runner import parse_set
    d = parse_set(["MATCH_MODE=sift+lightglue", "PG=dict(max_ratio=0.8)"])
    assert d["MATCH_MODE"] == "sift+lightglue" and d["PG"] == "py:dict(max_ratio=0.8)"
    from mppp.pipeline import Settings, normalize
    assert Settings().replace(**normalize({"match_mode": d["MATCH_MODE"]})).match_mode == "sift+lightglue"
