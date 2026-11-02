"""scripts/freeze_chain.py (2 Nov 2026): the unattended chain - the new root, the wait for v4.2, the mask choice, the S2 verdict, the commands."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _ab_csv(path, v3, v42):
    path.mkdir(parents=True, exist_ok=True)
    rows = ["model,subset,camera,exclusion_IoU,hardware_recall",
            f"mppp_mask_v3,not_seen,all,{v3},0.9", f"convnext_tiny_s4_seg_20261103_v8_v4.2,not_seen,all,{v42},0.9",
            "mppp_mask_v3,all,Mastcam-Z,0.97,0.8"]
    (path / "mask_ab.csv").write_text("\n".join(rows) + "\n")


def test_chain_waits_decides_and_runs(tmp_path, monkeypatch):
    import freeze_chain as F
    ck, outp = tmp_path / "checkpoints", tmp_path / "outputs"
    v41 = ck / "convnext_tiny_s4_seg_20261102_v8_v4.1.pt"
    v42 = ck / "convnext_tiny_s4_seg_20261103_v8_v4.2.pt"
    rel = outp / f"mask_{v42.stem}" / f"{v42.stem}.safetensors"
    rel.parent.mkdir(parents=True)
    rel.write_bytes(b"x")                                                     # the hand-started training is done
    ck.mkdir()
    v41.write_bytes(b"x")
    v42.write_bytes(b"x")
    old, new = tmp_path / "scapes", tmp_path / "rover_blocks"
    old.mkdir()
    (old / "zcam_low_backlash_found.csv").write_text("seq\n")
    (new / "dem").mkdir(parents=True)
    monkeypatch.setattr(F, "CKPT_DIR", ck)
    monkeypatch.setattr(F, "OUTPUTS", outp)
    monkeypatch.setattr(F, "V41", v41)
    ran = []

    def run(self, name, cmd, timeout_h):
        ran.append((name, [str(c) for c in cmd]))
        if name == "mask_ab":
            _ab_csv(Path(cmd[cmd.index("--out") + 1]), 0.9804, v42_iou)
        return 0
    monkeypatch.setattr(F.Chain, "run", run)
    roots = []
    monkeypatch.setattr(F, "s2_verdict", lambda c, root=None: roots.append(root) or True)
    v42_iou = 0.9821                                                          # v4.2 better than v3
    assert F.main(["--root", str(new), "--old-root", str(old)]) == 0
    assert [n for n, _ in ran] == ["mask_ab", "freeze", "s2"] and roots == [old]          # S2 last, at the old root
    assert ran[2][1][-1] == str(old)
    ab = ran[0][1]
    assert ab[ab.index("--split-ckpt") + 1] == str(v42) and ab[ab.index("--also-held-out") + 1] == str(v41)
    assert "--seen-split" in ab and "mppp_mask_v3" in ab and ab[-1].endswith("mask_ab_v3_v41_v42")
    fr = ran[1][1]
    assert fr[fr.index("--root") + 1] == str(new) and "--reset" in fr and "ba_points_cap=1500" not in fr
    assert f"mask_checkpoint={rel}" in fr
    assert (new / "zcam_low_backlash_found.csv").is_file()                   # copied from the old root
    log = (new / "mppp_pipeline" / "freeze_chain_log.txt").read_text()
    assert "the training is done" in log and "DEMs in" in log and "paper_studies" in log
    st = json.loads((new / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert st["steps"]["wait_v42"]["state"] == "done" and st["steps"]["mask_choice"]["choice"] == "v4.2"
    # v3 better: no mask_checkpoint; --cap: the cap; --skip-s2: no study
    ran.clear()
    v42_iou = 0.9790
    F.main(["--root", str(new), "--old-root", str(old), "--cap", "--skip-s2"])
    assert [n for n, _ in ran] == ["mask_ab", "freeze"]
    assert "ba_points_cap=1500" in ran[1][1] and not any(x.startswith("mask_checkpoint") for x in ran[1][1])
    ran.clear()
    F.main(["--root", str(new), "--old-root", str(old), "--skip-ab", "--no-freeze", "--no-wait", "--skip-s2"])
    assert ran == []


def test_wait_for_v42_gives_up(tmp_path, monkeypatch):
    import freeze_chain as F
    monkeypatch.setattr(F, "CKPT_DIR", tmp_path / "ck")
    monkeypatch.setattr(F, "OUTPUTS", tmp_path / "out")
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    c = F.Chain(type("A", (), {"root": str(tmp_path), "max_wait_h": 0.0, "quiet_min": 45.0})())
    F.wait_for_v42(c)
    assert "not done after" in (tmp_path / "mppp_pipeline" / "freeze_chain_log.txt").read_text()


def test_s2_verdict_limits(tmp_path, monkeypatch):
    import freeze_chain as F
    import tie_point_studies as T
    c = F.Chain(type("A", (), {"root": str(tmp_path)})())
    work = tmp_path / "mars2020_sol_0461_rockytop_example_zcam" / "studies"
    work.mkdir(parents=True)
    (work / "S2.json").write_text(json.dumps({"variants": {"all": {}, "cap1500": {}}}))
    for got, want in (({"centre_cm": {"p95": 0.4}, "attitude_mdeg": {"p95": 0.05}, "focal_pct": {"max": 0.01}}, True),
                      ({"centre_cm": {"p95": 0.4}, "attitude_mdeg": {"p95": 0.3}, "focal_pct": {"max": 0.01}}, False),
                      ({}, False)):
        monkeypatch.setattr(T, "compare_solutions", lambda a, b, g=got: g)
        assert F.s2_verdict(c) is want
