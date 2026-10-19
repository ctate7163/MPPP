"""v0p72: the screening cache (the candidate check and promotion live in mppp.calib since 0.90, tested in
test_calib.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    # v0p85: upload_mask_model.bat removed (one-off, 2 Oct); the registry stays
    from mppp.mask.hub import default_model_name, hf_repo, load_registry
    assert default_model_name() == "mppp_mask_v3" and hf_repo() == "ctate7163/mppp-mask"
    e = load_registry()["models"]["mppp_mask_v3"]
    assert e["source_checkpoint"] == "convnext_tiny_s4_seg_20260925b.pt" and e["urls"][0].startswith("https://huggingface.co/ctate7163/")
