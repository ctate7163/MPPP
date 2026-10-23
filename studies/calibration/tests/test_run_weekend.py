"""0.94.4: run_weekend - several days unattended, restartable."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RW = ROOT / "studies" / "calibration" / "run_weekend.py"


def _mod():
    spec = importlib.util.spec_from_file_location("run_weekend", RW)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _args(m, root, **kw):
    a = dict(root=str(root), set=list(m.DEFAULT_SET), jobs=2, ab_variants=list(m.AB_VARIANTS), wait_h=0.001,
             ab_timeout_h=1.0, pipeline_timeout_h=1.0, redo=None, status=False, dry_run=False, reset=False)
    a.update(kw)
    return argparse.Namespace(**a)


def test_weekend_plan_restart_and_resume(tmp_path, monkeypatch):
    m = _mod()
    w = m.Weekend(_args(m, tmp_path))
    calls = []
    monkeypatch.setattr(m.Weekend, "run", lambda self, step, cmd, timeout_h=None: calls.append((step, [str(c) for c in cmd])) or 0)
    assert w.go() == 0
    steps = [c[0] for c in calls]
    assert steps == ["ab_chal", "ab_seitah", "pipeline"]
    chal, seitah, pipe = (c[1] for c in calls)
    assert "--rerun" in chal and "--rerun" not in seitah and "--no-joint" in chal          # Chal Rocks: masks changed
    assert "sift_iso_off" in chal and "sift_pg_off" in chal
    assert "--reset" in pipe and pipe[pipe.index("--set") + 1] == "MATCH_MODE=exhaustive" and "--jobs" in pipe
    st = json.loads((tmp_path / "mppp_pipeline" / "weekend_state.json").read_text())
    assert all(st[s]["status"] == "done" for s in m.STEPS) and (tmp_path / "mppp_pipeline" / "WEEKEND.md").is_file()
    # started again: nothing is repeated
    calls.clear()
    assert m.Weekend(_args(m, tmp_path)).go() == 0 and calls == []
    # a crash during the pipeline (state "running"): it goes on without --reset; the A/B steps are not repeated
    st["pipeline"]["status"] = "running"
    (tmp_path / "mppp_pipeline" / "weekend_state.json").write_text(json.dumps(st))
    assert m.Weekend(_args(m, tmp_path)).go() == 0
    assert [c[0] for c in calls] == ["pipeline"] and "--reset" not in calls[0][1]
    # a crash during the Chal Rocks A/B: again without --rerun (the variants done since stay)
    calls.clear()
    st = json.loads((tmp_path / "mppp_pipeline" / "weekend_state.json").read_text())
    st["ab_chal"]["status"] = "running"
    (tmp_path / "mppp_pipeline" / "weekend_state.json").write_text(json.dumps(st))
    m.Weekend(_args(m, tmp_path)).go()
    assert calls[0][0] == "ab_chal" and "--rerun" not in calls[0][1]


def test_weekend_waits_for_a_live_alignment(tmp_path, monkeypatch):
    m = _mod()
    from mppp.runner import STATUS_FILE
    work = tmp_path / "mars2020_sol_0116_seitah_overlook_tiny_zcam"
    work.mkdir()
    import datetime
    import socket
    (work / STATUS_FILE).write_text(json.dumps({"state": "running", "pid": os.getppid(), "host": socket.gethostname(),
                                                "heartbeat": datetime.datetime.now().isoformat(timespec="seconds"),
                                                "variant": "lightglue_octaves"}))
    w = m.Weekend(_args(m, tmp_path))
    assert "seitah_overlook_tiny" in (w.busy() or "")
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    status, note = w.s_wait()
    assert status == "failed" and "still running" in note                         # wait_h is tiny here
    (work / STATUS_FILE).write_text(json.dumps({"state": "finished", "pid": 1}))
    assert w.busy() is None and w.s_wait()[0] == "done"


def test_weekend_bats_and_stop(tmp_path):
    w = ROOT / "scripts" / "windows"
    for name in ("run_weekend.bat", "_run_weekend_loop.bat"):
        b = (w / name).read_bytes()
        assert b.count(b"\r\n") == b.count(b"\n") > 10, name
    loop = (w / "_run_weekend_loop.bat").read_text()
    assert "run_weekend.py" in loop and "weekend_status.json" in loop and "'stopped'" in loop and "gtr 20" in loop
    assert "_run_weekend_loop.bat" in (w / "run_weekend.bat").read_text()
    sys.path.insert(0, str(ROOT / "scripts"))
    from _batch import mark_stopped
    d = tmp_path / "mppp_pipeline"
    d.mkdir()
    (d / "weekend_status.json").write_text(json.dumps({"state": "running", "pid": 4242, "host": "x"}))
    assert mark_stopped(tmp_path, [4242]) == [str(d / "weekend_status.json")]
    assert json.loads((d / "weekend_status.json").read_text())["state"] == "stopped"


def test_weekend_dry_run_cli(tmp_path):
    import subprocess
    r = subprocess.run([sys.executable, str(RW), "--root", str(tmp_path), "--dry-run"], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    log = (tmp_path / "mppp_pipeline" / "weekend_dry_run_log.txt").read_text()
    assert "lightglue_ab.py" in log and "run_pipeline.py" in log and "--reset" in log
    assert not (tmp_path / "mppp_pipeline" / "weekend_state.json").exists()


def test_stop_mppp_stops_the_weekend_loop():
    sys.path.insert(0, str(ROOT / "scripts"))
    from _batch import mppp_processes
    rows = [{"pid": 10, "ppid": 1, "cmd": 'cmd /c call "D:\\code\\MPPP\\scripts\\windows\\_run_weekend_loop.bat" "D:\\scapes"'},
            {"pid": 11, "ppid": 10, "cmd": "python studies/calibration/run_weekend.py --root D:/scapes"},
            {"pid": 12, "ppid": 11, "cmd": "python studies/calibration/run_pipeline.py --root D:/scapes"}]
    assert [r["pid"] for r in mppp_processes(rows)] == [10]           # the loop window: /T takes the rest
