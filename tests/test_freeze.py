"""The 0.96 freeze driver (rpp_studies.calibration.freeze): its steps in a dry run, the mask gate, the status."""
from __future__ import annotations

import json
import os
from pathlib import Path


from rpp_studies.calibration import freeze as fz


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
    assert st["mask"]["choice"].startswith("keep (rpp_mask_v5") and st["mask"]["set"] is None
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "zjoints (dry run)" in log and "-m rpp_studies.calibration.run_pipeline" in log
    assert "--zcam-waive-aspect 63 79 110" in log and "mask_checkpoint=" not in log     # keep: no --set
    assert "-m rpp_studies.calibration.deck" in log and "calib nz" in log and "ztest_zcam110_without_marble" in log
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
    assert st["steps"]["cmods"]["note"].startswith(str(root / "cmods")) and "no --candidate" in st["steps"]["cmods"]["note"]


def test_blocks_mask_reads_the_finished_blocks(tmp_path):
    root = _root(tmp_path)
    for i, m in enumerate([None, None, "D:\\ck\\v42.pt"]):
        d = root / f"block{i}" / "colmap"
        d.mkdir(parents=True)
        (d / "run_settings.json").write_text(json.dumps({"settings": {"mask_checkpoint": m}}), encoding="utf-8")
    assert fz.blocks_mask(root) is None                                    # the majority: the registry default
    assert fz.blocks_mask(tmp_path / "nothing") is None


def test_mask_auto_takes_the_verdict_and_the_margin(tmp_path, monkeypatch):
    home = tmp_path / "RPP"
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
        (out / "mask_verdict.json").write_text(json.dumps({"models": {ck.stem: new, "rpp_mask_v5": base},
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
    assert _status(root)["stopped_by"].startswith("stop_rpp.bat")
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


def test_update_code_refuses_under_a_live_run_on_another_root(tmp_path, capsys, monkeypatch):
    """0.99.49: sync_code.bat (root D:\\rover_blocks) updated D:\\code\\RPP_freeze under freeze 3 on D:\\rover_blocks_f3."""
    monkeypatch.delenv("RPP_FREEZE_ROOT", raising=False)
    root, f3 = tmp_path / "rover_blocks", tmp_path / "rover_blocks_f3"
    root.mkdir()
    (f3 / "mppp_pipeline").mkdir(parents=True)
    clone = tmp_path / "RPP_freeze"
    f = f3 / "mppp_pipeline" / fz.STATUS_FILE
    alive = {"state": "running", "heartbeat": fz.now(), "pid": os.getppid(), "tag": fz.TAG, "steps": {}}
    f.write_text(json.dumps(dict(alive, code=f"0.99.40 {clone / 'src' / 'rpp' / '__init__.py'} [1310abf]")), encoding="utf-8")
    assert fz.main(["--root", str(root), "--code", str(clone), "--update-code"]) == 2
    out = capsys.readouterr().out
    assert "not updated" in out and "rover_blocks_f3" in out
    f.write_text(json.dumps(dict(alive, code=f"0.99.40 {tmp_path / 'other' / 'src' / 'rpp' / '__init__.py'}")), encoding="utf-8")
    assert fz.main(["--root", str(root), "--code", str(clone), "--update-code"]) == 1      # another checkout: not a git one
    f.write_text(json.dumps(dict(alive, code=f"0.99.40 {clone}", heartbeat="2026-01-01T00:00:00")), encoding="utf-8")
    assert fz.main(["--root", str(root), "--code", str(clone), "--update-code"]) == 1      # a stale run does not hold it


def test_version_gate(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(fz, "code_version", lambda r: "0.95.5 D:\\code\\MPPP_freeze2\\src\\rpp\\__init__.py")
    assert _run(root) == 1
    assert _status(root)["state"] == "failed"
    assert f"needs RPP {fz.REQUIRED_VERSION}.x" in (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze"}), encoding="utf-8")
    assert _run(root, "--any-version") == 0


def test_tag_regex():
    assert fz.re.match(r"convnext_\w+?_seg_\d{8}(?:_(\d+))?(?:_(.+))?_debug$", "convnext_tiny_s4_seg_20261109_v9A_debug").group(2) == "v9A"


def test_train_mask_verdict_from_the_ab_table(tmp_path):
    """train_masks.bat --compare (Christian, 9 Nov: "will the mask comparison run after the training so that we will
    know which is better?"): the verdict the freeze reads, from compare_mask_models' table - the fair subset when
    there is one, the winner and the margin new minus baseline."""
    from rpp_studies.mask_training.train_mask import DEFAULT_BASELINE, read_verdict_from_ab, parser_for_test
    ab = tmp_path / "mask_ab"
    ab.mkdir()
    ck = tmp_path / "convnext_tiny_s4_seg_20261109_v9A.pt"
    rows = ["subset,camera,frames,model,exclusion_IoU,hardware_recall,hardware_precision",
            "all,all,40,rpp_mask_v5,0.9700,0.98,0.98", f"all,all,40,{ck.stem},0.9800,0.99,0.98",
            "all,Mastcam-Z,10,rpp_mask_v5,0.9990,0.99,0.99",
            "not_seen,all,12,rpp_mask_v5,0.9600,0.97,0.98", f"not_seen,all,12,{ck.stem},0.9750,0.98,0.98"]
    (ab / "mask_ab.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    v = read_verdict_from_ab(ab, new=ck, baseline="rpp_mask_v5")
    assert v["subset"] == "not_seen" and v["winner"] == ck.stem and abs(v["margin"] - 0.015) < 1e-9
    assert v["models"] == {"rpp_mask_v5": 0.96, ck.stem: 0.975} and v["baseline_label"] == DEFAULT_BASELINE
    assert read_verdict_from_ab(tmp_path / "none", new=ck, baseline="rpp_mask_v5")["winner"] is None
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
    home = tmp_path / "RPP"
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
    assert "no --candidate" in st["steps"]["cmods"]["note"]          # 0.98.15: no candidate, the folder as it is
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
    (out / "mask_verdict.json").write_text(json.dumps({"models": {ck.stem: 0.9842, "rpp_mask_v5": 0.9712}, "subset": "not_seen"}), encoding="utf-8")
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
    assert fz.baseline_training_set(str(masks), "rpp_mask_v3") == str(masks / "masks_training_set_v7")
    assert fz.baseline_training_set(str(masks), "rpp_mask_v5") == str(masks / "masks_training_set_v8")   # 1.0: the default
    assert fz.baseline_training_set(str(masks), "D:\\ck\\x.pt") is None
    assert _run(root2, "--mask-model", "auto", "--mask-home", str(home), "--mask-root", str(masks),
                "--cmods", str(root2 / "cmods_freeze2")) == 0
    log2 = (root2 / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert ("mask_ab (dry run)" in log2 and "-m rpp_studies.mask_training.compare_mask_models" in log2
            and f"--data {masks / 'masks_training_set_v9'}" in log2 and "--models rpp_mask_v5" in log2
            and f"--seen {masks / 'masks_training_set_v8'} --seen-split rpp_mask_v5 {masks / 'masks_training_set_v8'}" in log2)


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
                             "mask": {"state": "done", "choice": "keep (rpp_mask_v5)", "set": None}}), encoding="utf-8")
    seen = {}

    def pipeline(self, name, *args):
        seen.setdefault("day", self.day())
        return 0
    monkeypatch.setattr(fz.Run, "pipeline", pipeline)
    monkeypatch.setattr(fz.Run, "rpp", lambda self, name, *a, **k: seen.setdefault(name, a) and 0)
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


def test_fresh_sets_the_old_status_aside(tmp_path, monkeypatch):
    """0.97.0 (12 Nov): --fresh starts from the beginning; the status of the run before is kept under another name."""
    root = _root(tmp_path)
    d = root / "mppp_pipeline"
    d.mkdir(parents=True, exist_ok=True)
    (d / fz.STATUS_FILE).write_text(json.dumps({"started": "2026-11-09T00:00:00", "tag": "freeze_096", "state": "stopped",
                                                "steps": {"zjoints": {"state": "done"}}}), encoding="utf-8")
    monkeypatch.setattr(fz, "code_version", lambda r: fz.REQUIRED_VERSION + ".0 x")
    monkeypatch.setattr(fz.Run, "guarded", lambda self, s, f: None)
    for n in ("keep_awake", "logon_task_register", "logon_task_delete", "free_gb"):
        monkeypatch.setattr(fz, n, lambda *a, **k: None)
    fz.main(["--root", str(root), "--code", str(tmp_path), "--fresh", "--now", "--no-logon-task"])
    old = list(d.glob("freeze_status_before_*.json"))
    assert len(old) == 1 and json.loads(old[0].read_text(encoding="utf-8"))["tag"] == "freeze_096"
    new = json.loads((d / fz.STATUS_FILE).read_text(encoding="utf-8"))
    assert new["tag"] == fz.TAG and "zjoints" not in new["steps"]


def test_seitah_overlook_parallax():
    from rpp.pipeline import load_settings
    assert load_settings(site="seitah_overlook").min_tri_angle_deg == 0.05      # 0.98.23 (was 0.10)
    assert load_settings(site="chal_rocks_large").min_tri_angle_deg == 0.25


def test_window_shows_the_block_being_aligned(tmp_path):
    """0.97.0 (12 Nov): a line when a block starts and when it ends."""
    import time
    root = _root(tmp_path)
    r = fz.Run(fz.parser().parse_args(["--root", str(root), "--code", str(tmp_path)]))
    w = root / "mars2020_sol_0500_wildcat_ridge_zcam"
    w.mkdir(parents=True)
    st = {"state": "running", "site": "wildcat_ridge", "started": "2026-11-12T14:40:46", "log": str(w / "log.txt")}
    (w / "log.txt").write_text("2026-11-12T14:43:41      | stage 1: Navcam only\n", encoding="utf-8")
    (w / "mppp_status.json").write_text(json.dumps(st), encoding="utf-8")
    t0 = time.time() - 60
    r.show_blocks("zjoints", t0)
    r.show_blocks("zjoints", t0)                                          # no second "aligning" line
    st.update(state="finished", ended="2026-11-12T18:10:46",
              result={"verdict": "warn", "images": 697, "registered_images": 683, "residual_rms_native_px": 0.44})
    (w / "mppp_status.json").write_text(json.dumps(st), encoding="utf-8")
    r.show_blocks("zjoints", t0)
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert log.count("aligning wildcat_ridge") == 1 and "stage 1: Navcam only" in log
    assert "wildcat_ridge finished after 3.5 h: warn, 683 of 697 images, 0.44 px" in log


def test_freeze3_candidate_over_a_freeze_rig_the_zooms_and_the_folder_kept(tmp_path, monkeypatch):
    """0.98.15 (freeze-3): a candidate rig that differs from the folder's freeze rig is promoted (before: the freeze
    rig in the folder stopped it); the pipeline gets the freeze's zooms with 100 mm; a restart without --cmods or
    --candidate goes on with the run's own."""
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze-2 (8 Nov)"}), encoding="utf-8")
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "M2020_N_rig.json").write_text(json.dumps({"source": "freeze-3 rig (deck joint)"}), encoding="utf-8")
    assert _run(root, "--candidate", str(cand)) == 0
    st = _status(root)
    assert "promote_cmods" in (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert st["models"] == {"cmods": str(root / "cmods"), "candidate": str(cand)}
    assert 100 in fz.ZOOMS
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "--zooms 26 34 48 63 79 100 110" in log
    # the candidate's files already in the folder: nothing promoted
    (root / "cmods" / "M2020_N_rig.json").write_bytes((cand / "M2020_N_rig.json").read_bytes())
    r = fz.Run(fz.parser().parse_args(["--root", str(root), "--dry-run", "--candidate", str(cand),
                                       "--cmods", str(root / "cmods")]))
    assert "already in it" in r.s_cmods()
    # a restart with no folder arguments (the logon restart) keeps the run's folder and candidate
    a = fz.parser().parse_args(["--root", str(root), "--dry-run"])
    r = fz.Run(a)
    assert r.cmods == root / "cmods" and a.candidate == str(cand)


def test_freeze3_ladder_step_runs_seitahs_states_with_100mm(tmp_path):
    """0.98.21 (Christian, 16 Nov 2026: "The zcam_zoom_ladder_seitah_sol121_zcam_states plot should also include 100"):
    a step after the studies runs Seitah Overlook's states variant (each Mastcam-Z image its own camera) and its states
    report into camera_analysis/zcam_states_<day>_seitah - the newest states study, which the figures draw the ladder
    from; Seitah Overlook's block lists 100 mm."""
    import inspect
    from rpp_studies.calibration import freeze as fz
    from rpp.paths import data_file
    assert fz.STEPS[-2:] == ("ladder", "report") and fz.LADDER_SITES == ("seitah_overlook",)
    src = inspect.getsource(fz.Run.s_ladder)
    assert "--variant\", \"states\"" in src and "zcam_states_{self.day()}_seitah" in src and "ZCAM_STATES_SETS" in src
    sites = json.loads(Path(data_file("sites.json")).read_text(encoding="utf-8"))["sites"]
    assert 100 in sites["seitah_overlook"]["zcam"]
    for k in ("seitah_overlook", "skinner_ridge", "taylorfjellet", "chal_rocks_large"):     # 0.98.21: position priors
        assert sites[k]["settings"]["zcam_station_prior_sigma_m"] == 0.02


def test_freeze_runs_on_this_version_and_keeps_its_checkout_on_its_line(tmp_path):
    """0.99.3 (Christian, 16 Nov 2026: "Is there anything preventing freeze3 from running on v0p99?"): the freeze needs the
    package's own minor version (0.99 on branch rpp); --update-code pulls the tree's branch, and refuses a checkout on
    another branch (a v0p98 checkout is never moved to v0p99); freeze.bat runs from main or rpp, each with its checkout."""
    import subprocess
    import rpp
    from rpp_studies.calibration import freeze as F
    assert F.REQUIRED_VERSION == ".".join(rpp.__version__.split(".")[:2])
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True)          # noqa: E731
    tree = tmp_path / "tree"
    git("init", "-q", "-b", "rpp", str(tree))
    git("-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "one")
    git("clone", "-q", "-b", "rpp", str(tree), str(tmp_path / "freeze_rpp"))
    git("-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "two")
    git("-C", str(tree), "branch", "-q", "main")
    git("clone", "-q", "-b", "main", str(tree), str(tmp_path / "freeze_main"))
    root = tmp_path / "root"
    root.mkdir()
    base = ["--root", str(root), "--from-tree", str(tree), "--update-code"]
    assert F.tree_branch(tree) == "rpp"
    assert F.main(base + ["--code", str(tmp_path / "freeze_rpp")]) == 0
    assert "two" in F.git_head(tmp_path / "freeze_rpp")
    assert F.main(base + ["--code", str(tmp_path / "freeze_main")]) == 1           # main stays main
    bat = (Path(F.REPO) / "scripts" / "windows" / "freeze.bat").read_text(encoding="ascii")
    assert r'"FREEZE_DEFAULT_CODE=D:\code\RPP_freeze"' in bat and r'"FREEZE_DEFAULT_CODE=D:\code\MPPP_freeze"' in bat
    assert "git clone -q -b !FREEZE_BRANCH!" in bat
    assert r'if exist "%RPP_MAIN%\src\rpp\__init__.py" set "FREEZE_DEFAULT_CODE=D:\code\RPP_freeze"' in bat   # 0.99.5
    assert "knee" in F.STEPS and F.STEPS.index("knee") == F.STEPS.index("deck") + 1


def test_knee_step_adds_extreme_temperature_stations_and_the_scale_tests(tmp_path):
    """0.99.7 (Christian, 16 Nov: "for the next rover deck joint, add 20 stations at extreme temperatures"; and the
    elevation-scale question): the knee step passes --extreme-temps 20 to the drive package when its fill knows the
    option, and sets EOD_JOINT_OPTS=--scale-tests for that call only."""
    root = _root(tmp_path)
    (root / "cmods_from").mkdir()
    (root / "cmods_from" / "M2020_N_rig.json").write_text(json.dumps({"source": "3 Nov"}), encoding="utf-8")
    pkgs = tmp_path / "outputs"
    pkg = pkgs / "deck_joint_drive_2026-11-14"
    pkg.mkdir(parents=True)
    (pkg / "run_eod_drive.bat").write_text("@echo off\n", encoding="ascii")
    (pkg / "fill_from_drive.py").write_text('ap.add_argument("--extreme-temps", type=int)\n', encoding="utf-8")
    assert _run(root, "--eod-packages", str(pkgs)) == 0
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    line = next(ln for ln in log.splitlines() if "knee_deck (dry run)" in ln)
    assert "--first 250 --last 400 --bins 30" in line and line.rstrip().endswith("--extreme-temps 20")
    note = _status(root)["steps"]["knee"]["note"]
    assert "20 at extreme temperatures, scale tests" in note
    # an older package without the option: the stations of 250-400 only, and a note
    (pkg / "fill_from_drive.py").write_text("old\n", encoding="utf-8")
    root2 = _root(tmp_path / "two")
    (root2 / "cmods_from").mkdir()
    (root2 / "cmods_from" / "M2020_N_rig.json").write_text("{}", encoding="utf-8")
    assert _run(root2, "--eod-packages", str(pkgs), "--no-knee-scale-tests") == 0
    log2 = (root2 / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "--extreme-temps" not in next(ln for ln in log2.splitlines() if "knee_deck (dry run)" in ln)
    note2 = _status(root2)["steps"]["knee"]["note"]
    assert "has no --extreme-temps" in note2 and "scale tests" not in note2
    r = fz.Run(fz.parser().parse_args(["--root", str(root), "--dry-run"]))
    assert "EOD_JOINT_OPTS" not in r.env or r.env["EOD_JOINT_OPTS"] != "--scale-tests"


def test_new_root_is_seeded_with_the_backlash_lists_and_z26_promotes_from_one_block(tmp_path):
    """0.99.22 (freeze-3 in a new, empty root; Christian, 18 Nov: "Promote 26 ... look for other roadblocks")."""
    old = tmp_path / "old_root"
    old.mkdir()
    (old / "zcam_low_backlash_found.csv").write_text("sequence,sol\nZCAM08114,201\n", encoding="utf-8")
    (old / "zcam_focus_from_below.csv").write_text("sequence\n", encoding="utf-8")
    root = _root(tmp_path)
    (root / "cmods").mkdir()
    (root / "cmods" / "M2020_N_rig.json").write_text("{}", encoding="utf-8")
    # a dry run copies nothing and says what it would
    assert _run(root, "--seed-from", str(old)) == 0
    assert not (root / "zcam_low_backlash_found.csv").exists()
    log = (root / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    assert "zcam_low_backlash_found.csv from" in log and "dry run: not copied" in log
    assert "--zcam-one-block 26 100" in log and fz.ONE_BLOCK == (26, 100)
    # the real copy; a second call leaves the files alone
    a = fz.parser().parse_args(["--root", str(root), "--cmods", str(root / "cmods"), "--seed-from", str(old),
                                "--no-logon-task"])
    r = fz.Run(a)
    first = r.seed_root()
    assert "zcam_low_backlash_found.csv" in first and "zcam_focus_from_below.csv" in first
    assert (root / "zcam_low_backlash_found.csv").read_text(encoding="utf-8").startswith("sequence,sol")
    assert (root / "zcam_focus_from_below.csv").is_file() and r.seed_root() == ""
    # a root without the list and nowhere to take it from: said in the log
    bare = tmp_path / "bare" / "rover_blocks"
    (bare / "mppp_pipeline").mkdir(parents=True)
    rb = fz.Run(fz.parser().parse_args(["--root", str(bare), "--cmods", str(bare / "cmods"), "--no-logon-task"]))
    assert "no zcam_low_backlash_found.csv" in rb.seed_root()
    assert "WARNING - no zcam_low_backlash_found.csv" in (bare / "mppp_pipeline" / fz.LOG_FILE).read_text(encoding="utf-8")
    # the zooms of a single block reach the joints
    from rpp_studies.calibration import run_pipeline as rp
    src = Path(rp.__file__).read_text(encoding="utf-8")
    assert '"--min-blocks", "1"' in src and "--zcam-one-block" in src
