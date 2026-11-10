"""The 0.96 freeze driver (mppp_studies.calibration.freeze): its steps in a dry run, the mask gate, the status."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from mppp_studies.calibration import freeze as fz


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "rover_blocks"
    (root / "mppp_pipeline").mkdir(parents=True)
    return root


def _run(root: Path, *args: str, code: Path | None = None) -> int:
    argv = ["--root", str(root), "--code", str(code or Path(fz.REPO)), "--dry-run", "--no-logon-task",
            "--cmods", str(root / "cmods"), "--cmods-from", str(root / "cmods_from"), *args]
    return fz.main(argv)


def _status(root: Path) -> dict:
    return json.loads((root / "mppp_pipeline" / fz.STATUS_FILE).read_text(encoding="utf-8"))


def test_dry_run_walks_every_step_and_writes_the_report(tmp_path):
    root = _root(tmp_path)
    (root / "cmods_from").mkdir()
    (root / "cmods_from" / "M2020_N_rig.json").write_text(json.dumps({"source": "3 Nov"}), encoding="utf-8")
    assert _run(root) == 0
    st = _status(root)
    assert st["state"] == "done" and st["tag"] == fz.TAG and st["failed"] == []
    assert [s for s in fz.STEPS if st["steps"][s]["state"] == "done"] == list(fz.STEPS)
    assert st["mask"]["choice"].startswith("keep (mppp_mask_v3") and st["mask"]["set"] is None
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "zjoints (dry run)" in log and "-m mppp_studies.calibration.run_pipeline" in log
    assert "--zcam-waive-aspect 63 79 110" in log and "mask_checkpoint=" not in log     # keep: no --set
    assert "-m mppp_studies.calibration.deck" in log and "calib nz" in log and "ztest_zcam110_without_marble" in log
    rep = (root / "mppp_pipeline" / fz.REPORT_FILE).read_text(encoding="utf-8")
    assert "# Freeze run report" in rep and "| zjoints | done |" in rep and "mask model: keep" in rep
    # a second start skips the done steps and ends at once
    assert _run(root) == 0
    assert len(_status(root)["restarts"]) == 1


def test_explicit_mask_model_sets_the_checkpoint_on_every_pipeline_call(tmp_path):
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze-2 (8 Nov)"}), encoding="utf-8")
    ck = tmp_path / "convnext_tiny_s4_seg_20261109_v9A.pt"
    ck.write_bytes(b"x")
    assert _run(root, "--mask-model", str(ck)) == 0
    st = _status(root)
    assert st["mask"]["set"] == str(ck)
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert log.count(f"--set mask_checkpoint={ck}") == 4                    # zjoints, zfinal, ecam, studies
    assert "every image is processed again" in log
    assert st["steps"]["cmods"]["note"].startswith(str(root / "cmods")) and "already in it" in st["steps"]["cmods"]["note"]


def test_blocks_mask_reads_the_finished_blocks(tmp_path):
    root = _root(tmp_path)
    for i, m in enumerate([None, None, "D:\\ck\\v42.pt"]):
        d = root / f"block{i}" / "colmap"
        d.mkdir(parents=True)
        (d / "run_settings.json").write_text(json.dumps({"settings": {"mask_checkpoint": m}}), encoding="utf-8")
    assert fz.blocks_mask(root) is None                                    # the majority: the registry default
    assert fz.blocks_mask(tmp_path / "nothing") is None


def test_mask_auto_takes_the_verdict_and_the_margin(tmp_path, monkeypatch):
    home = tmp_path / "MPPP"
    ck_dir = home / "checkpoints"
    ck_dir.mkdir(parents=True)
    ck = ck_dir / "convnext_tiny_s4_seg_20261109_v9A.pt"
    ck.write_bytes(b"x")
    dbg = ck_dir / "convnext_tiny_s4_seg_20261109_v9A_debug"
    dbg.mkdir()
    (dbg / "log.csv").write_text("iter,loss\n1,1\n", encoding="utf-8")
    out = home / "Claude outputs" / f"mask_{ck.stem}"
    out.mkdir(parents=True)
    (out / "summary.md").write_text("done", encoding="utf-8")
    assert fz.newest_training_tag(home) == "v9A"
    assert fz.trained_checkpoint(home, "v9A") == ck
    assert fz.trained_checkpoint(home, "v9B") is None

    def verdict(new, base):
        (out / "mask_verdict.json").write_text(json.dumps({"models": {ck.stem: new, "mppp_mask_v3": base},
                                                           "subset": "all"}), encoding="utf-8")
    # a clear win: the new model is used
    verdict(0.985, 0.970)
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")
    assert _run(root, "--mask-model", "auto", "--mask-home", str(home)) == 0
    st = _status(root)
    assert st["mask"]["set"] == str(ck) and st["mask"]["choice"] == f"new {ck.name}" and st["mask"]["tag"] == "v9A"
    assert abs(st["mask"]["margin"] - 0.015) < 1e-9
    # a marginal win: the blocks' model stays - and a restart keeps the first choice regardless
    verdict(0.972, 0.970)
    root2 = _root(tmp_path / "second")
    (root2 / "cmods").mkdir()
    (root2 / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")
    assert _run(root2, "--mask-model", "auto", "--mask-home", str(home)) == 0
    st2 = _status(root2)
    assert st2["mask"]["set"] is None and "margin +0.0020 < 0.01" in st2["mask"]["choice"]
    assert _run(root2, "--mask-model", str(ck), "--redo", "mask") == 0
    assert _status(root2)["mask"]["set"] is None and "kept from before" in _status(root2)["steps"]["mask"]["note"]


def test_mask_auto_without_a_training_keeps_the_blocks_model(tmp_path):
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")
    assert _run(root, "--mask-model", "auto", "--mask-home", str(tmp_path / "empty")) == 0
    assert "no training found" in _status(root)["mask"]["choice"]


def test_training_set_helpers(tmp_path):
    for v in (7, 8, 9):
        (tmp_path / f"masks_training_set_v{v}" / "masks").mkdir(parents=True)
    (tmp_path / "masks_training_set_v10").mkdir()                           # no masks/: not a set
    assert fz.newest_training_set(str(tmp_path)) == str(tmp_path / "masks_training_set_v9")
    assert fz.previous_training_set(str(tmp_path / "masks_training_set_v9")) == str(tmp_path / "masks_training_set_v8")
    assert fz.previous_training_set(str(tmp_path / "masks_training_set_v7")) is None


def test_status_tells_running_stopped_and_not_running_apart(tmp_path, capsys):
    root = _root(tmp_path)
    args = fz.parser().parse_args(["--root", str(root), "--code", str(fz.REPO)])
    assert "not started" in fz.describe(fz.Run(args, read_only=True))
    f = root / "mppp_pipeline" / fz.STATUS_FILE
    now = fz.now()
    f.write_text(json.dumps({"state": "running", "heartbeat": now, "pid": os.getppid(), "tag": fz.TAG,
                             "steps": {"zjoints": {"state": "running", "started": now}}}), encoding="utf-8")
    d = fz.describe(fz.Run(args, read_only=True))
    assert d.startswith(f"freeze on {root}: running - step zjoints")
    f.write_text(json.dumps({"state": "running", "heartbeat": "2026-11-01T00:00:00", "pid": 999999, "tag": fz.TAG,
                             "steps": {"zjoints": {"state": "running"}}}), encoding="utf-8")
    assert "NOT running, and it did not end on its own" in fz.describe(fz.Run(args, read_only=True))
    f.write_text(json.dumps({"state": "stopped", "heartbeat": now, "ended": now, "stopped_by": "Ctrl-C", "pid": 1,
                             "tag": fz.TAG, "steps": {}}), encoding="utf-8")
    assert "stopped by the user at" in fz.describe(fz.Run(args, read_only=True)) and "Ctrl-C" in fz.describe(fz.Run(args, read_only=True))
    assert fz.main(["--root", str(root), "--code", str(fz.REPO), "--status"]) == 0
    out = capsys.readouterr().out
    assert "stopped by the user" in out and "zjoints" in out and "pipeline:" in out


def test_a_second_run_does_not_start_beside_a_live_one(tmp_path):
    root = _root(tmp_path)
    f = root / "mppp_pipeline" / fz.STATUS_FILE
    f.write_text(json.dumps({"state": "running", "heartbeat": fz.now(), "pid": os.getppid(), "tag": fz.TAG, "steps": {}}),
                 encoding="utf-8")                                          # another live process: this one's parent
    assert _run(root) == 2
    assert _status(root)["state"] == "running"                             # untouched


def test_ctrl_c_and_stop_marker_end_as_stopped_by_the_user(tmp_path, monkeypatch):
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")

    def boom(self):
        raise KeyboardInterrupt
    monkeypatch.setattr(fz.Run, "s_zjoints", boom)
    assert _run(root) == 130
    st = _status(root)
    assert st["state"] == "stopped" and st["stopped_by"] == "Ctrl-C" and st["steps"]["zjoints"]["state"] == "stopped"
    assert st["steps"]["cmods"]["state"] == "done" and "zfinal" not in st["steps"]
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "stopped by the user (Ctrl-C)" in log

    def stopped(self):
        raise fz.ChainStopped("zjoints")
    monkeypatch.setattr(fz.Run, "s_zjoints", stopped)
    assert _run(root) == 130
    assert _status(root)["stopped_by"].startswith("stop_mppp.bat")
    # and a start afterwards continues (the step runs again, the done ones are skipped)
    monkeypatch.setattr(fz.Run, "s_zjoints", lambda self: "exit 0")
    assert _run(root) == 0
    assert _status(root)["state"] == "done" and _status(root)["steps"]["zjoints"]["state"] == "done"


def test_a_failed_step_is_logged_and_the_run_goes_on(tmp_path, monkeypatch):
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")

    def bad(self):
        raise RuntimeError("no joints")
    monkeypatch.setattr(fz.Run, "s_zjoints", bad)
    assert _run(root) == 1
    st = _status(root)
    assert st["steps"]["zjoints"]["state"].startswith("failed: RuntimeError") and st["failed"] == ["zjoints"]
    assert st["steps"]["report"]["state"] == "done"


def test_update_code_refuses_under_a_live_run(tmp_path, capsys):
    root = _root(tmp_path)
    f = root / "mppp_pipeline" / fz.STATUS_FILE
    f.write_text(json.dumps({"state": "running", "heartbeat": fz.now(), "pid": os.getppid(), "tag": fz.TAG, "steps": {}}),
                 encoding="utf-8")
    assert fz.main(["--root", str(root), "--code", str(tmp_path / "clone"), "--update-code"]) == 2
    assert "not updated" in capsys.readouterr().out
    f.unlink()
    assert fz.main(["--root", str(root), "--code", str(tmp_path / "clone"), "--update-code"]) == 1   # not a checkout
    assert "not a git checkout" in capsys.readouterr().out


def test_version_gate(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(fz, "code_version", lambda r: "0.95.5 D:\\code\\MPPP_freeze2\\src\\mppp\\__init__.py")
    assert _run(root) == 1
    assert _status(root)["state"] == "failed"
    assert "needs MPPP 0.96.x" in (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")
    assert _run(root, "--any-version") == 0


def test_tag_regex():
    assert fz.re.match(r"convnext_\w+?_seg_\d{8}(?:_(\d+))?(?:_(.+))?_debug$", "convnext_tiny_s4_seg_20261109_v9A_debug").group(2) == "v9A"


def test_train_mask_verdict_from_the_ab_table(tmp_path):
    """train_masks.bat --compare (Christian, 9 Nov: "will the mask comparison run after the training so that we will
    know which is better?"): the verdict the freeze reads, from compare_mask_models' table - the fair subset when
    there is one, the winner and the margin new minus baseline."""
    from mppp_studies.mask_training.train_mask import DEFAULT_BASELINE, read_verdict_from_ab, parser_for_test
    ab = tmp_path / "mask_ab"
    ab.mkdir()
    ck = tmp_path / "convnext_tiny_s4_seg_20261109_v9A.pt"
    rows = ["subset,camera,frames,model,exclusion_IoU,hardware_recall,hardware_precision",
            "all,all,40,mppp_mask_v3,0.9700,0.98,0.98", f"all,all,40,{ck.stem},0.9800,0.99,0.98",
            "all,Mastcam-Z,10,mppp_mask_v3,0.9990,0.99,0.99",
            "not_seen,all,12,mppp_mask_v3,0.9600,0.97,0.98", f"not_seen,all,12,{ck.stem},0.9750,0.98,0.98"]
    (ab / "mask_ab.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    v = read_verdict_from_ab(ab, new=ck, baseline="mppp_mask_v3")
    assert v["subset"] == "not_seen" and v["winner"] == ck.stem and abs(v["margin"] - 0.015) < 1e-9
    assert v["models"] == {"mppp_mask_v3": 0.96, ck.stem: 0.975} and v["baseline_label"] == DEFAULT_BASELINE
    assert read_verdict_from_ab(tmp_path / "none", new=ck, baseline="mppp_mask_v3")["winner"] is None
    a = parser_for_test().parse_args(["--compare", "--smoke"])
    assert a.compare == [] and a.compare_device == "auto"
    assert parser_for_test().parse_args(["--smoke"]).compare is None
