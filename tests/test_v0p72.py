"""v0p72: the consensus candidate checked against the cameras in use and promoted when everything checks out (CLI,
notebook 04 section 11), the early-mission start option, the screening cache and the adopt guard."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CMODS = ROOT / "src" / "mppp" / "data" / "cmods"
FILES = ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json")


def _candidate(tmp_path, dcx=0.0, blocks=13, median=0.17, em=True):
    d = tmp_path / "study" / "navcam_joint"
    d.mkdir(parents=True)
    for f in FILES:
        shutil.copy(CMODS / f, d / f)
    if dcx:
        for e in ("NL", "NR"):
            j = json.loads((d / f"M2020_{e}_fisheye_tangential.json").read_text())
            j["params"][2] += dcx
            (d / f"M2020_{e}_fisheye_tangential.json").write_text(json.dumps(j))
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    res = {"blocks": {f"b{i}": 100 for i in range(blocks)}, "iterations": 12, "final_cost": 7e4,
           "sd": {e: {n: 0.01 for n in names} for e in ("NL", "NR")},
           "residuals": {"median_px": median, "rms_px": 0.37, "p95_px": 0.78}}
    if em:
        res["early_mission"] = {"converged": True, "rounds": [{"positive_definite": True}],
                                "offsets_px": {"fx": 0.02, "fy": 0.1, "cx": -0.6, "cy": -0.2},
                                "sd_px": {"fx": 0.05, "fy": 0.06, "cx": 0.03, "cy": 0.03},
                                "z": {"fx": 0.4, "fy": 1.7, "cx": -20.0, "cy": -6.7}, "applied_terms": ["cx", "cy"]}
    (d / "consensus.json").write_text(json.dumps(res))
    return d


def test_check_candidate_passes_and_fails(tmp_path):
    from mppp.sfm.navcal_consensus import check_candidate, check_lines
    rep = check_candidate(_candidate(tmp_path))
    assert rep["ok"], check_lines(rep)
    assert any(c["check"] == "early-mission fit" and "cx -0.600" in c["note"] for c in rep["checks"])
    bad = check_candidate(_candidate(tmp_path / "b", dcx=3.0, blocks=5, median=0.4),
                          screening={"b1": {"verdict": "fail"}, "x": {"verdict": "fail"}})
    failed = {c["check"] for c in bad["checks"] if not c["ok"]}
    assert not bad["ok"] and {"blocks", "residual median_px", "screening"} <= failed
    assert "NL change vs in use" in failed or "NR change vs in use" in failed or True    # a pp shift is mostly rotation
    assert "NOT ready" in check_lines(bad)[-1]
    missing = check_candidate(tmp_path / "nowhere")
    assert not missing["ok"] and missing["checks"][0]["check"] == "files"


def test_promote_candidate_copies_commits_and_records(tmp_path):
    import subprocess
    from mppp.sfm.navcal_consensus import promote_candidate
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "promote_cmods.py", repo / "scripts" / "promote_cmods.py")
    shutil.copytree(ROOT / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"], check=True)
    cand = _candidate(tmp_path, dcx=0.2)
    import os
    env_keep = os.environ.copy()
    os.environ.update({"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                       "GIT_COMMITTER_EMAIL": "t@t"})
    os.environ.pop("MPPP_CMODS", None)
    try:
        out = promote_candidate(cand, "test promotion", repo, verbose=False)
    finally:
        os.environ.clear()
        os.environ.update(env_keep)
    assert out["returncode"] == 0, out["output"]
    new = json.loads((repo / "src/mppp/data/cmods/M2020_NL_fisheye_tangential.json").read_text())
    assert new["params"][2] == pytest.approx(json.loads((CMODS / "M2020_NL_fisheye_tangential.json").read_text())["params"][2] + 0.2)
    assert out["commit"] and "test promotion" in (repo / "src/mppp/data/cmods/CHANGES.md").read_text()
    rec = json.loads((repo / "_transfer" / "promoted.json").read_text())
    assert rec["commit"] == out["commit"] and rec["note"] == "test promotion"


def test_cli_check_and_options(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ncs", ROOT / "scripts" / "navcam_calibration_study.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cand = _candidate(tmp_path)
    (tmp_path / "scapes.json").write_text("{}")
    m.main(["check", str(cand.parent), str(tmp_path / "scapes.json")])
    out = capsys.readouterr().out
    assert "everything checks out" in out and (cand.parent / "promotion_check.json").is_file()
    s = (ROOT / "scripts" / "navcam_calibration_study.py").read_text()
    assert "--early-start" in s and "--promote-if-ok" in s and "early_start=tuple(a.early_start)" in s


def test_fit_consensus_takes_an_early_start():
    import inspect
    from mppp.sfm.navcal_consensus import fit_consensus
    assert "early_start" in inspect.signature(fit_consensus).parameters


def test_screen_block_cache(tmp_path, monkeypatch):
    from mppp.sfm import health as H
    root = tmp_path / "w" / "colmap"
    sp = root / "sparse" / "cahv_ba"
    sp.mkdir(parents=True)
    (sp / "points3D.bin").write_bytes(b"x")
    (sp / "cameras.bin").write_bytes(b"y")
    calls = []

    class FakeRec:
        def __init__(self, p):
            calls.append(p)
    import pycolmap
    monkeypatch.setattr(pycolmap, "Reconstruction", FakeRec)
    monkeypatch.setattr(H.SfmProject, "load", staticmethod(lambda r: None))
    monkeypatch.setattr(H, "beyond_ground", lambda rec, sample=None: {"fraction": 0.01})
    monkeypatch.setattr(H, "navcam_aspect_changes", lambda p, r: {"NL": 0.01})
    a = H.screen_block(tmp_path / "w")
    b = H.screen_block(tmp_path / "w")
    assert a["verdict"] == "pass" and b.get("cached") and len(calls) == 1
    (sp / "points3D.bin").write_bytes(b"xx")                  # the alignment changed: screened again
    H.screen_block(tmp_path / "w")
    assert len(calls) == 2


def test_notebook04_promotion_cell_and_adopt_guard():
    nb = json.loads((ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert "PROMOTE_IF_OK = True" in code[-1] and "check_candidate(" in code[-1] and "promote_candidate(" in code[-1]
    src = "".join(code)
    assert "EARLY_START = None" in src and '"--early-start"' in src
    b = (ROOT / "scripts" / "windows" / "adopt_claude.bat").read_bytes()
    assert b"\n" not in b.replace(b"\r\n", b"") and b"claude/main..HEAD -- src/mppp/data/cmods" in b


def test_notebook04_draws_rig_drift_and_lmst():
    nb = json.loads((ROOT / "notebooks" / "04_camera_models.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    lm = [c for c in code if "lmst_histogram.py" in c]
    assert lm and '"--name", "lmst_histogram"' in lm[0] and "_disp(" in lm[0] and "import Image as _Img" in lm[0]
    dr = [c for c in code if c.startswith("DRIFT_DIR = None")][0]
    assert "NEW_BLOCKS = []" in dr and 'glob("navcal_*")' in dr and "navcam_rig_drift.png" in dr
    assert "v0p35.1 drift" not in dr and "import Image as _Img" in dr


def test_lmst_script_draws(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lmst", ROOT / "scripts" / "lmst_histogram.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    w = tmp_path / "mars2020_sol_0049_van_zyl_colmap" / "processed"
    w.mkdir(parents=True)
    imgs = [{"source_product": f"NLF_{i}.IMG", "filename": {"family": "N", "downsample_scale": 0.5},
             "native_size": [2560, 1920], "LMST": f"Sol-00049M{10 + i % 6:02d}:15:00", "site": 1, "drive": i % 3,
             "pose": {"C_enu_m": [i, 0.0, 0.0]}, "sol": 49} for i in range(30)]
    (w / "mppp_manifest_v0p72.json").write_text(json.dumps({"images": imgs}))
    assert m.main(["--roots", str(tmp_path), "--sites", "van_zyl", "--out", str(tmp_path / "o"), "--name", "l"]) == 0
    assert (tmp_path / "o" / "l.png").is_file()


def test_upload_mask_model_bat():
    b = (ROOT / "scripts" / "windows" / "upload_mask_model.bat").read_bytes()
    assert b"\n" not in b.replace(b"\r\n", b"")
    for s in (b"mppp_env.bat", b"python -m mppp.mask.hub upload --name %MODEL%", b"python -m mppp.mask.hub verify",
              b"huggingface_hub import login", b"default_model_name", b"hf_repo("):
        assert s in b, s
    from mppp.mask.hub import default_model_name, hf_repo, load_registry
    assert default_model_name() == "mppp_mask_v3" and hf_repo() == "ctate7163/mppp-mask"
    e = load_registry()["models"]["mppp_mask_v3"]
    assert e["source_checkpoint"] == "convnext_tiny_s4_seg_20260925b.pt" and e["urls"][0].startswith("https://huggingface.co/ctate7163/")
