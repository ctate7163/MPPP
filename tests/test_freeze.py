"""The 0.96 freeze driver (mppp_studies.calibration.freeze): its steps in a dry run, the mask gate, the status."""
from __future__ import annotations

import json
import os
from pathlib import Path


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


# ----------------------------------------------------------------------------- 9 Nov evening: the PC as it stands
def _pc_like_root(tmp_path: Path) -> Path:
    """D:\\rover_blocks on the evening of 9 Nov: the first chain's status says 'studies running' with a heartbeat
    hours old, the pipeline lock stale (freeze-2 stopped at 14:53), cmods_freeze2 with the freeze-2 rig, finished
    blocks processed with the registry default."""
    root = _root(tmp_path)
    d = root / "mppp_pipeline"
    (d / "freeze_chain_status.json").write_text(json.dumps({
        "state": "running", "heartbeat": "2026-11-09T09:26:53", "pid": 31060,
        "steps": {"studies": {"state": "running"}, "figures": {}, "maps": {}}}), encoding="utf-8")
    (d / "pipeline_status.json").write_text(json.dumps({
        "state": "running", "pid": 43512, "host": "scipc25-04", "heartbeat": "2026-11-09T14:53:16",
        "kind": "pipeline", "stage": "zcam_runs"}), encoding="utf-8")
    (root / "cmods_freeze2").mkdir()
    (root / "cmods_freeze2" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze-2 (8 Nov): deck rig"}), encoding="utf-8")
    for name in ("mars2020_sol_0072_chal_rocks_large_zcam", "mars2020_sol_0413_threeforks_north_zcam", "mars2020_sol_0001_butler_landing"):
        c = root / name / "colmap"
        c.mkdir(parents=True)
        (c / "run_settings.json").write_text(json.dumps({"settings": {"mask_checkpoint": None, "infer_mask": True}}), encoding="utf-8")
        (c / "run_done.json").write_text("{}", encoding="utf-8")
    return root


def _pc_like_training(tmp_path: Path, last_epoch: int, last_iter: int, epochs: int = 30, improved_at: int = 17) -> Path:
    """D:\\code\\MPPP\\checkpoints on 9 Nov: the v9A start that died at 14:48 and the live v9_v4.4 run - its card
    written at its best epoch only, its log at (last_epoch, last_iter)."""
    home = tmp_path / "MPPP"
    ck_dir = home / "checkpoints"
    dead = ck_dir / "convnext_tiny_s4_seg_20261109_v9A_debug"
    dead.mkdir(parents=True)
    (dead / "log.csv").write_text("epoch,iter,n_iter,loss\n1,150,1080,0.93\n", encoding="utf-8")
    os.utime(dead / "log.csv", (1794260894, 1794260894))                                  # 14:48
    live = ck_dir / "convnext_tiny_s4_seg_20261109_v9_v4.4_debug"
    live.mkdir()
    iters = list(range(50, 1051, 50)) + [1080]                                          # every 50, then the epoch's end
    rows = ["epoch,iter,n_iter,loss"] + [f"{e},{i},1080,0.05" for e in range(1, last_epoch + 1) for i in iters
                                         if (e < last_epoch or i <= last_iter)]
    (live / "log.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    ck = ck_dir / "convnext_tiny_s4_seg_20261109_v9_v4.4.pt"
    ck.write_bytes(b"x")
    ck.with_suffix(".json").write_text(json.dumps({"epoch": improved_at, "val_iou": 0.9754, "training": {"epochs": epochs},
                                                   "history": [{"epoch": e} for e in range(1, improved_at + 1)]}), encoding="utf-8")
    return home


def test_the_training_end_is_read_from_the_log_not_the_card(tmp_path):
    home = _pc_like_training(tmp_path, last_epoch=18, last_iter=900)
    assert fz.newest_training_tag(home) == "v9_v4.4"                        # the live run, not the dead v9A start
    assert fz.training_log_epoch(home, "v9_v4.4") == (18, 900, 1080)
    assert fz.trained_checkpoint(home, "v9_v4.4") is None                   # epoch 18 of 30: not through
    # the last epoch's last iteration logged, the card still at epoch 17 (no later epoch improved): through
    home2 = _pc_like_training(tmp_path / "b", last_epoch=30, last_iter=1080)
    ck = fz.trained_checkpoint(home2, "v9_v4.4")
    assert ck is not None and ck.name == "convnext_tiny_s4_seg_20261109_v9_v4.4.pt"
    # quiet: the log stopped mid-way (a crash) - with quiet=True the checkpoint there is taken, else not
    assert fz.trained_checkpoint(home, "v9_v4.4", quiet=True) is not None
    assert fz.trained_checkpoint(home, "v9_v4.4", quiet=False) is None


def test_keep_passes_the_blocks_own_checkpoint_when_it_is_a_path(tmp_path):
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")
    for i in range(2):
        c = root / f"block{i}" / "colmap"
        c.mkdir(parents=True)
        (c / "run_settings.json").write_text(json.dumps({"settings": {"mask_checkpoint": "D:\\ck\\v42.pt"}}), encoding="utf-8")
    assert _run(root) == 0
    st = _status(root)
    assert st["mask"]["set"] == "D:\\ck\\v42.pt" and st["mask"]["choice"] == "keep (v42)"
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert log.count("--set mask_checkpoint=D:\\ck\\v42.pt") == 4           # the same configuration: nothing processed again
    assert "every image is processed again" not in log


def test_pc_like_root_auto_with_the_live_training_in_a_dry_run(tmp_path):
    """The evening's state in a dry run: wait passes (both status files stale), cmods already the freeze rig, the
    mask step finds the live training and - in a dry run - goes on without waiting; the chain runs on v3."""
    root = _pc_like_root(tmp_path)
    home = _pc_like_training(tmp_path, last_epoch=18, last_iter=900)
    assert _run(root, "--mask-model", "auto", "--mask-home", str(home), "--cmods", str(root / "cmods_freeze2")) == 0
    st = _status(root)
    assert st["steps"]["wait"]["note"].startswith("pipeline idle")
    assert "already in it" in st["steps"]["cmods"]["note"]
    assert st["mask"]["tag"] == "v9_v4.4" and st["mask"]["set"] is None and "not finished" in st["mask"]["choice"]
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "mask auto: waiting for the training 'v9_v4.4'" in log and "mask_checkpoint=" not in log
    assert "deck_chal_rocks_large (dry run)" in log and "deck_threeforks_north (dry run)" in log
    assert "deck_airey_hill (dry run)" in log


def test_pc_like_root_auto_with_a_finished_training_and_a_verdict(tmp_path):
    root = _pc_like_root(tmp_path)
    home = _pc_like_training(tmp_path, last_epoch=30, last_iter=1080)
    ck = home / "checkpoints" / "convnext_tiny_s4_seg_20261109_v9_v4.4.pt"
    out = home / "Claude outputs" / f"mask_{ck.stem}"
    out.mkdir(parents=True)
    (out / "mask_verdict.json").write_text(json.dumps({"models": {ck.stem: 0.9842, "mppp_mask_v3": 0.9712}, "subset": "not_seen"}), encoding="utf-8")
    assert _run(root, "--mask-model", "auto", "--mask-home", str(home), "--cmods", str(root / "cmods_freeze2")) == 0
    st = _status(root)
    assert st["mask"]["set"] == str(ck) and st["mask"]["choice"] == f"new {ck.name}"
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert log.count(f"--set mask_checkpoint={ck}") == 4 and "every image is processed again" in log
    # the dry-run comparison command when no verdict is there: the baseline's own training set as --seen
    root2 = _pc_like_root(tmp_path / "two")
    (out / "mask_verdict.json").unlink()
    masks = tmp_path / "masks"
    for v in (7, 8, 9):
        (masks / f"masks_training_set_v{v}" / "masks").mkdir(parents=True)
    assert fz.baseline_training_set(str(masks), "mppp_mask_v3") == str(masks / "masks_training_set_v7")
    assert fz.baseline_training_set(str(masks), "D:\\ck\\x.pt") is None
    assert _run(root2, "--mask-model", "auto", "--mask-home", str(home), "--mask-root", str(masks),
                "--cmods", str(root2 / "cmods_freeze2")) == 0
    log2 = (root2 / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert ("mask_ab (dry run)" in log2 and "-m mppp_studies.mask_training.compare_mask_models" in log2
            and f"--data {masks / 'masks_training_set_v9'}" in log2 and "--models mppp_mask_v3" in log2
            and f"--seen {masks / 'masks_training_set_v7'} --seen-split mppp_mask_v3 {masks / 'masks_training_set_v7'}" in log2)


def test_logon_launcher_carries_the_mask_options(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(fz, "WINDOWS", True)
    calls = []
    monkeypatch.setattr(fz.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    a = fz.parser().parse_args(["--root", str(root), "--code", str(fz.REPO), "--mask-model", "auto", "--mask-margin", "0.005"])
    r = fz.Run(a)
    fz.logon_task_register(r)
    launcher = (root / "mppp_pipeline" / "freeze_logon.bat").read_text(encoding="ascii")
    assert '--mask-model "auto" --mask-margin 0.005' in launcher and "freeze.bat" in launcher
    assert calls and calls[0][:2] == ["reg", "add"]


# ----------------------------------------------------------------------------- 0.96.1 (11 Nov): the PC restarted under the run
def test_a_child_that_fails_fails_its_step_and_a_restart_takes_it_up(tmp_path, monkeypatch):
    root = _root(tmp_path)
    argv = ["--root", str(root), "--code", str(fz.REPO), "--no-logon-task", "--cmods", str(root / "cmods")]
    rcs = {"zfinal": [1, 0]}
    monkeypatch.setattr(fz.Run, "pipeline", lambda self, name, *a: rcs.get(name, [0]).pop(0) if rcs.get(name) else 0)
    r = fz.Run(fz.parser().parse_args(argv))
    r.guarded("zfinal", r.s_zfinal)
    assert r.status["steps"]["zfinal"]["state"].startswith("failed: RuntimeError: zfinal exit 1") and not r.done("zfinal")
    r.guarded("ecam", r.s_ecam)                                             # the run goes on
    assert r.done("ecam")
    r2 = fz.Run(fz.parser().parse_args(argv))                               # started again: the failed step runs
    r2.guarded("zfinal", r2.s_zfinal)
    assert r2.done("zfinal") and r2.status["steps"]["zfinal"]["note"] == "exit 0"


def test_a_restart_after_a_reboot_is_said_and_keeps_the_runs_day(tmp_path, monkeypatch):
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")
    f = root / "mppp_pipeline" / fz.STATUS_FILE
    f.write_text(json.dumps({"state": "running", "started": "2026-11-10T08:29:36", "heartbeat": "2026-11-11T03:10:00", "pid": 4,
                             "tag": fz.TAG, "steps": {"wait": {"state": "done"}, "mask": {"state": "done"},
                                                      "cmods": {"state": "done"}, "zjoints": {"state": "running"}},
                             "mask": {"state": "done", "choice": "keep (mppp_mask_v3)", "set": None}}), encoding="utf-8")
    seen = {}

    def pipeline(self, name, *args):
        seen.setdefault("day", self.day())
        return 0
    monkeypatch.setattr(fz.Run, "pipeline", pipeline)
    monkeypatch.setattr(fz.Run, "mppp", lambda self, name, *a, **k: seen.setdefault(name, a) and 0)
    monkeypatch.setattr(fz.Run, "s_deck", lambda self: "no deck here")
    assert _run(root) == 0
    st = _status(root)
    assert st["interruptions"][0]["last_heartbeat"] == "2026-11-11T03:10:00" and st["interruptions"][0]["step"] == "zjoints"
    assert seen["day"] == "20261110" and any("nz34_joint_freeze_20261110" in str(x) for x in seen["nz"])
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "ended without a stop (last heartbeat 2026-11-11T03:10:00, in zjoints)" in log


def test_the_block_passes_get_the_long_time_limit(tmp_path, monkeypatch):
    root = _root(tmp_path)
    cmds = []
    monkeypatch.setattr(fz.Run, "run", lambda self, name, cmd, **k: cmds.append([str(c) for c in cmd]) or 0)
    r = fz.Run(fz.parser().parse_args(["--root", str(root), "--code", str(fz.REPO), "--cmods", str(root / "cmods")]))
    r.pipeline("zjoints", "--from", "zcam_runs")
    assert cmds[0][cmds[0].index("--sites-timeout-h") + 1] == "168"
