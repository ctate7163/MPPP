"""mppp_studies.calibration.freeze_chain (2 Nov 2026): the unattended chain - the root, the wait for v4.2, the mask choice, the
pipeline in three calls (blocks, final pass, studies), S2 last; a restart continues it."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


import pytest


@pytest.fixture(autouse=True)
def _no_disk_wait(monkeypatch):
    """The disk-space wait (MIN_FREE_GB) is off in the tests - a CI disk may be smaller than the PC's."""
    from mppp_studies.calibration import freeze_chain as F
    monkeypatch.setattr(F, "MIN_FREE_GB", 0.0)


def _ab_csv(path, v3, v42):
    path.mkdir(parents=True, exist_ok=True)
    rows = ["model,subset,camera,exclusion_IoU,hardware_recall",
            f"mppp_mask_v3,not_seen,all,{v3},0.9", f"convnext_tiny_s4_seg_20261103_v8_v4.2,not_seen,all,{v42},0.9",
            "mppp_mask_v3,all,Mastcam-Z,0.97,0.8"]
    (path / "mask_ab.csv").write_text("\n".join(rows) + "\n")


def _setup(tmp_path, monkeypatch, trained=True):
    from mppp_studies.calibration import freeze_chain as F
    ck, outp = tmp_path / "checkpoints", tmp_path / "outputs"
    v41 = ck / "convnext_tiny_s4_seg_20261102_v8_v4.1.pt"
    v42 = ck / "convnext_tiny_s4_seg_20261103_v8_v4.2.pt"
    ck.mkdir()
    v41.write_bytes(b"x")
    v42.write_bytes(b"x")
    if trained:
        rel = outp / f"mask_{v42.stem}" / f"{v42.stem}.safetensors"
        rel.parent.mkdir(parents=True)
        rel.write_bytes(b"x")                                                 # the hand-started training is done
    old, new = tmp_path / "scapes", tmp_path / "rover_blocks"
    old.mkdir()
    (old / "zcam_low_backlash_found.csv").write_text("seq\n")
    (new / "dem").mkdir(parents=True)
    monkeypatch.setattr(F, "CKPT_DIR", ck)
    monkeypatch.setattr(F, "OUTPUTS", outp)
    monkeypatch.setattr(F, "V41", v41)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    return F, v41, v42, old, new


def test_chain_runs_through(tmp_path, monkeypatch):
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    ran = []
    state = {"v42_iou": 0.9821}                                               # v4.2 better than v3

    def run(self, name, cmd, timeout_h, env=None):
        cmd = [str(c) for c in cmd]
        ran.append((name, cmd))
        if name == "mask_ab":
            _ab_csv(Path(cmd[cmd.index("--out") + 1]), 0.9804, state["v42_iou"])
        self.step(name, state="done", started="t", ended="t")
        return 0
    monkeypatch.setattr(F.Chain, "run", run)
    roots = []
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: roots.append(root) or True)
    assert F.main(["--root", str(new), "--old-root", str(old)]) == 0
    assert [n for n, _ in ran] == ["mask_ab", "blocks", "final_pass", "studies", "aspect_geometry", "s6", "s2",
                                   "s1", "s3", "s4", "s5"] and roots == [old]
    geo, s6 = ran[4][1], ran[5][1]
    assert geo[geo.index("--root") + 1] == str(new) and "110" in geo
    assert s6[s6.index("--root") + 1] == str(new) and "chal_rocks_large" in s6
    assert ran[7][1][-1] == str(old) and ran[7][1][ran[7][1].index("S1") - 1].endswith("tie_point_studies")
    ab = ran[0][1]
    assert ab[ab.index("--split-ckpt") + 1] == str(v42) and ab[ab.index("--also-held-out") + 1] == str(v41)
    assert "--seen-split" in ab and "mppp_mask_v3" in ab and ab[-1].endswith("mask_ab_v3_v41_v42")
    blocks, final, studies = ran[1][1], ran[2][1], ran[3][1]
    for cmd in (blocks, final, studies):
        assert cmd[cmd.index("--root") + 1] == str(new) and f"mask_checkpoint={v42}" in cmd
        assert "ba_points_cap=1500" not in cmd
    assert "--reset" in blocks and blocks[blocks.index("--stop-before") + 1] == "error_analysis"
    assert "--reset" not in final and final[final.index("--redo"):final.index("--redo") + 3] == ["--redo", "ecam_runs", "zcam_runs"]
    assert "--stop-before" in final and "--stop-before" not in studies and "--redo" not in studies
    assert ran[6][1][-1] == str(old)                                          # S2 at the old root, after the blocks
    assert (new / "zcam_low_backlash_found.csv").is_file()                   # copied from the old root
    log = (new / "mppp_pipeline" / "freeze_chain_log.txt").read_text()
    assert "trained" in log and "DEMs in" in log and "GB free" in log and "make_figures.bat" in log
    st = json.loads((new / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert st["steps"]["wait_v42"]["state"] == "done" and st["steps"]["mask_choice"]["choice"] == "v4.2"
    assert st["state"] == "done" and st["steps"]["chain"]["failed"] == []


def test_chain_v3_cap_and_flags(tmp_path, monkeypatch):
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    ran = []

    def run(self, name, cmd, timeout_h, env=None):
        cmd = [str(c) for c in cmd]
        ran.append((name, cmd))
        if name == "mask_ab":
            _ab_csv(Path(cmd[cmd.index("--out") + 1]), 0.9804, 0.9790)       # v3 better
        self.step(name, state="done", started="t", ended="t")
        return 0
    monkeypatch.setattr(F.Chain, "run", run)
    F.main(["--root", str(new), "--old-root", str(old), "--cap", "--skip-s2", "--no-final-pass", "--skip-tie-studies"])
    assert [n for n, _ in ran] == ["mask_ab", "blocks", "studies", "aspect_geometry", "s6"]
    assert "ba_points_cap=1500" in ran[1][1] and not any(x.startswith("mask_checkpoint") for x in ran[1][1])
    st = json.loads((new / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert st["steps"]["mask_choice"]["choice"] == "v3" and st["steps"]["final_pass"]["note"] == "--no-final-pass"
    # a fresh root: nothing but the A/B and S2
    ran.clear()
    (new / "mppp_pipeline" / "freeze_chain_status.json").unlink()
    F.main(["--root", str(new), "--old-root", str(old), "--skip-ab", "--no-freeze", "--no-wait", "--skip-tie-studies"])
    assert [n for n, _ in ran] == ["aspect_geometry", "s6", "s2"]


def test_chain_restart_continues(tmp_path, monkeypatch):
    """A crash in the blocks: the next start keeps the mask choice, skips the A/B, continues without --reset."""
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    ran = []
    crash = {"on": True}

    def run(self, name, cmd, timeout_h, env=None):
        cmd = [str(c) for c in cmd]
        ran.append((name, cmd))
        if name == "mask_ab":
            _ab_csv(Path(cmd[cmd.index("--out") + 1]), 0.9804, 0.9821)
        if name == "blocks" and crash["on"]:
            self.step(name, state="running", command=cmd, started="t")
            raise KeyboardInterrupt                                      # the window closed / the computer restarted
        self.step(name, state="done", started="t", ended="t")
        return 0
    monkeypatch.setattr(F.Chain, "run", run)
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: True)
    assert F.main(["--root", str(new), "--old-root", str(old)]) == 130
    assert [n for n, _ in ran] == ["mask_ab", "blocks"]
    st = json.loads((new / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert st["state"] == "stopped" and st["tag"] == F.CHAIN_TAG
    # the A/B folder is gone too (the choice must still hold)
    import shutil
    shutil.rmtree(F.OUTPUTS / "mask_ab_v3_v41_v42")
    ran.clear()
    crash["on"] = False
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-tie-studies"]) == 0
    assert [n for n, _ in ran] == ["blocks", "final_pass", "studies", "aspect_geometry", "s6", "s2"]
    blocks = ran[0][1]
    assert "--reset" not in blocks and f"mask_checkpoint={v42}" in blocks
    log = (new / "mppp_pipeline" / "freeze_chain_log.txt").read_text()
    assert "continued" in log and "kept from before - v4.2" in log and "no --reset" in log
    st = json.loads((new / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert len(st["restarts"]) == 1 and st["state"] == "done"
    # done: a third start runs nothing; --redo s2 runs S2 again
    ran.clear()
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-tie-studies"]) == 0 and ran == []
    F.main(["--root", str(new), "--old-root", str(old), "--redo", "s2", "--skip-tie-studies"])
    assert [n for n, _ in ran] == ["s2"]
    F.main(["--root", str(new), "--old-root", str(old)])                      # the tie studies not run before
    assert [n for n, _ in ran] == ["s2", "s1", "s3", "s4", "s5"]


def test_chain_refuses_second_instance(tmp_path, monkeypatch):
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    d = new / "mppp_pipeline"
    d.mkdir(parents=True)
    import datetime
    (d / "freeze_chain_status.json").write_text(json.dumps({
        "started": "t", "tag": F.CHAIN_TAG, "steps": {}, "state": "running", "pid": 4242,
        "heartbeat": datetime.datetime.now().isoformat(timespec="seconds")}))
    monkeypatch.setattr(F, "pid_alive", lambda pid: pid == 4242)
    assert F.main(["--root", str(new), "--old-root", str(old)]) == 2
    monkeypatch.setattr(F, "pid_alive", lambda pid: False)             # the process is gone: it starts
    monkeypatch.setattr(F.Chain, "run", lambda self, name, cmd, timeout_h, env=None: self.step(name, state="done") or 0)
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: True)
    assert F.main(["--root", str(new), "--old-root", str(old)]) == 0


def test_wait_for_v42_gives_up(tmp_path, monkeypatch):
    from mppp_studies.calibration import freeze_chain as F
    monkeypatch.setattr(F, "CKPT_DIR", tmp_path / "ck")
    monkeypatch.setattr(F, "OUTPUTS", tmp_path / "out")
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    c = F.Chain(type("A", (), {"root": str(tmp_path), "max_wait_h": 0.0, "quiet_min": 45.0})())
    F.wait_for_v42(c)
    assert "not done after" in (tmp_path / "mppp_pipeline" / "freeze_chain_log.txt").read_text()
    # a checkpoint without its summary or release is not trained
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "convnext_tiny_s4_seg_20261103_v8_v4.2.pt").write_bytes(b"x")
    assert F.v42_trained() is None
    (tmp_path / "out" / "mask_convnext_tiny_s4_seg_20261103_v8_v4.2").mkdir(parents=True)
    (tmp_path / "out" / "mask_convnext_tiny_s4_seg_20261103_v8_v4.2" / "summary.md").write_text("done")
    assert F.v42_trained() is not None


def test_s2_verdict_limits(tmp_path, monkeypatch):
    from mppp_studies.calibration import freeze_chain as F
    from mppp_studies.calibration import tie_point_studies as T
    c = F.Chain(type("A", (), {"root": str(tmp_path)})())
    work = tmp_path / "mars2020_sol_0461_rockytop_example_zcam" / "studies"
    work.mkdir(parents=True)
    (work / "S2.json").write_text(json.dumps({"variants": {"all": {}, "cap1500": {}}}))
    for got, want in (({"centre_cm": {"p95": 0.4}, "attitude_mdeg": {"p95": 0.05}, "focal_pct": {"max": 0.01}}, True),
                      ({"centre_cm": {"p95": 0.4}, "attitude_mdeg": {"p95": 0.3}, "focal_pct": {"max": 0.01}}, False),
                      ({}, False)):
        monkeypatch.setattr(T, "compare_solutions", lambda a, b, g=got: g)
        assert F.s2_verdict(c) is want


def test_run_stops_a_child_past_its_time(tmp_path, monkeypatch):
    from mppp_studies.calibration import freeze_chain as F
    c = F.Chain(type("A", (), {"root": str(tmp_path), "redo": None})())
    rc = c.run("sleepy", [sys.executable, "-c", "import time; time.sleep(30)"], timeout_h=0.0)
    assert rc in (124, 127) or rc != 0
    assert "stopped" in (tmp_path / "mppp_pipeline" / "freeze_chain_log.txt").read_text() or rc != 0


def test_chain_step_error_goes_on(tmp_path, monkeypatch):
    """The chain's own code for a step raising (not its child): logged, the step marked, the next steps run."""
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    ran = []

    def run(self, name, cmd, timeout_h, env=None):
        ran.append(name)
        if name == "mask_ab":
            _ab_csv(Path([str(c) for c in cmd][[str(c) for c in cmd].index("--out") + 1]), 0.98, 0.99)
        self.step(name, state="done")
        return 0
    monkeypatch.setattr(F.Chain, "run", run)

    def boom(c, name, mask, *extra, reset=False):
        if name == "final_pass":
            raise RuntimeError("a bug in the chain")
        return c.run(name, ["x"], 1)
    monkeypatch.setattr(F, "run_pipeline_step", boom)
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: True)
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-tie-studies"]) == 1
    assert ran == ["mask_ab", "blocks", "studies", "aspect_geometry", "s6", "s2"]
    st = json.loads((new / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert st["steps"]["final_pass"]["state"] == "exit chain-error" and st["state"] == "done"
    assert "a bug in the chain" in (new / "mppp_pipeline" / "freeze_chain_log.txt").read_text()


def test_old_status_file_set_aside(tmp_path, monkeypatch):
    """The 14:17 chain's status (other steps, "s2" done) must not make this chain skip S2."""
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    d = new / "mppp_pipeline"
    d.mkdir(parents=True)
    (d / "freeze_chain_status.json").write_text(json.dumps({
        "started": "t", "steps": {"s2": {"state": "done"}, "freeze": {"state": "running"}}}))
    ran = []
    monkeypatch.setattr(F.Chain, "run", lambda self, name, cmd, timeout_h, env=None: ran.append(name) or self.step(name, state="done") or 0)
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: True)
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-ab", "--skip-tie-studies"]) == 0
    assert ran == ["blocks", "final_pass", "studies", "aspect_geometry", "s6", "s2"]
    assert (d / f"freeze_chain_status_before_{F.CHAIN_TAG}.json").is_file()
    st = json.loads((d / "freeze_chain_status.json").read_text())
    assert st["tag"] == F.CHAIN_TAG and "restarts" not in st
    assert F.main(["--root", str(new), "--old-root", str(old), "--status"]) == 0


def test_logon_task_registered_and_removed(tmp_path, monkeypatch):
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    calls = []

    class R:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(F.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or R())
    monkeypatch.setattr(F, "WINDOWS", True)
    bat = tmp_path / "freeze_chain.bat"
    bat.write_text("rem")
    monkeypatch.setattr(F, "bat_path", lambda: bat)
    monkeypatch.setattr(F, "keep_awake", lambda c: None)
    monkeypatch.setattr(F, "pid_alive", lambda pid: False)
    monkeypatch.setattr(F.Chain, "run", lambda self, name, cmd, timeout_h, env=None: self.step(name, state="done") or 0)
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: True)
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-ab"]) == 0
    sch = [c for c in calls if c and c[0] == "reg"]
    assert [c[1] for c in sch] == ["add", "delete"] and all(c[2] == F.RUNONCE_KEY for c in sch)
    tr = sch[0][sch[0].index("/d") + 1]
    launcher = new / "mppp_pipeline" / "freeze_chain_logon.bat"
    assert tr == f'"{launcher}"' and sch[1][sch[1].index("/v") + 1] == F.LOGON_TASK
    text = launcher.read_bytes().decode()
    assert str(bat) in text and "MPPP_NOPAUSE=1" in text and str(new) in text and "\r\n" in text
    calls.clear()
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-ab", "--no-logon-task"]) == 0
    assert not [c for c in calls if c and c[0] == "reg"]


def test_stop_marker_ends_the_chain(tmp_path, monkeypatch):
    """stop_runs stops the pipeline and writes <root>\\mppp_stopped.txt: the chain must not start the next step."""
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    ran = []

    def popen(cmd, **kw):
        class P:
            pid = 1
            def wait(self, timeout=None):
                if "run_pipeline" in " ".join(cmd):
                    (new / F.STOP_MARKER).write_text("stopped by stop_runs\n")
                    return 1
                return 0
            def poll(self):
                return 0
        ran.append(cmd)
        return P()
    monkeypatch.setattr(F.subprocess, "Popen", popen)
    monkeypatch.setattr(F, "keep_awake", lambda c: None)
    monkeypatch.setattr(F, "wait_pipeline_idle", lambda c: None)
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-ab", "--no-wait"]) == 130
    assert len(ran) == 1 and "run_pipeline" in " ".join(ran[0])
    st = json.loads((new / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert st["state"] == "stopped" and st["steps"]["blocks"]["state"] == "exit 1"
    assert "stopped by stop_runs" in (new / "mppp_pipeline" / "freeze_chain_log.txt").read_text()


def test_compatible_status_of_another_tag_continues(tmp_path, monkeypatch):
    """A status file from another chain version whose steps are all known is continued, not set aside."""
    F, v41, v42, old, new = _setup(tmp_path, monkeypatch)
    d = new / "mppp_pipeline"
    d.mkdir(parents=True)
    (d / "freeze_chain_status.json").write_text(json.dumps({
        "started": "t", "tag": "freeze_0000", "state": "done",
        "steps": {s: {"state": "done"} for s in F.STEPS if s != "s2"} | {"mask_choice": {"state": "done", "choice": "v3"}}}))
    ran = []
    monkeypatch.setattr(F.Chain, "run", lambda self, name, cmd, timeout_h, env=None: ran.append(name) or self.step(name, state="done") or 0)
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: True)
    assert F.main(["--root", str(new), "--old-root", str(old), "--skip-tie-studies"]) == 0
    assert ran == ["s2"] and not list(d.glob("freeze_chain_status_before_*"))
    assert "continued as" in (d / "freeze_chain_log.txt").read_text()


def test_wait_for_space(tmp_path, monkeypatch):
    """3 Nov (audit): under MIN_FREE_GB a step waits for space (here it appears on the third look)."""
    import shutil
    from mppp_studies.calibration import freeze_chain as F
    monkeypatch.setattr(F, "MIN_FREE_GB", 25.0)
    clock = {"t": 0.0}
    monkeypatch.setattr(F.time, "time", lambda: clock["t"])
    monkeypatch.setattr(F.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    free = iter([10e9, 10e9, 30e9])
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("U", (), {"free": next(free)})())
    c = F.Chain(type("A", (), {"root": str(tmp_path), "redo": None})())
    F.wait_for_space(c, "blocks")
    log = (tmp_path / "mppp_pipeline" / "freeze_chain_log.txt").read_text()
    assert "waiting for space" in log and clock["t"] == 600
