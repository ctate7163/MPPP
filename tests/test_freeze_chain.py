"""scripts/freeze_chain.py (2 Nov 2026): the unattended chain - waiting for the training, the S2 verdict, the commands."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_chain_waits_decides_and_runs(tmp_path, monkeypatch):
    import freeze_chain as F
    ck, outp = tmp_path / "checkpoints", tmp_path / "outputs"
    v41 = ck / "convnext_tiny_s4_seg_20261102_v8_v4.1.pt"
    (outp / f"mask_{v41.stem}").mkdir(parents=True)
    (outp / f"mask_{v41.stem}" / f"{v41.stem}.safetensors").write_bytes(b"x")       # the training is done
    ck.mkdir()
    v41.write_bytes(b"x")
    monkeypatch.setattr(F, "CKPT_DIR", ck)
    monkeypatch.setattr(F, "OUTPUTS", outp)
    monkeypatch.setattr(F, "V41", v41)
    ran = []
    monkeypatch.setattr(F.Chain, "run", lambda self, name, cmd, timeout_h: ran.append((name, [str(c) for c in cmd])) or 0)
    # S2 passes: every difference under its limit
    monkeypatch.setattr(F, "s2_verdict", lambda c: True)
    assert F.main(["--root", str(tmp_path / "scapes")]) == 0
    names = [n for n, _ in ran]
    assert names == ["mask_ab", "s2", "freeze"]
    ab = ran[0][1]
    assert "--seen-split" in ab and str(v41) in ab and "mppp_mask_v3" in ab
    assert ab[ab.index("--split-ckpt") + 1] == str(v41) and "--also-held-out" in ab
    fr = ran[2][1]
    assert "--reset" in fr and "ba_points_cap=1500" in fr
    log = (tmp_path / "scapes" / "mppp_pipeline" / "freeze_chain_log.txt").read_text()
    assert "the training is done" in log and "paper_studies" in log
    st = json.loads((tmp_path / "scapes" / "mppp_pipeline" / "freeze_chain_status.json").read_text())
    assert st["steps"]["wait_training"]["state"] == "done"
    # no cap when S2 fails; --skip-s2 runs no study
    ran.clear()
    monkeypatch.setattr(F, "s2_verdict", lambda c: False)
    F.main(["--root", str(tmp_path / "scapes"), "--skip-ab"])
    assert [n for n, _ in ran] == ["s2", "freeze"] and "ba_points_cap=1500" not in ran[1][1]
    ran.clear()
    F.main(["--root", str(tmp_path / "scapes"), "--skip-ab", "--skip-s2", "--no-freeze"])
    assert ran == []


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
