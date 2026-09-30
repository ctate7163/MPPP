"""Mask model, inference, registry and Hugging Face release (mppp.mask.model, infer, hub). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import pytest
import torch
from conftest import CKPT, LEGACY_CKPT, needs_ckpt, needs_legacy_ckpt
import numpy as np
import json
from pathlib import Path
import sys
import types
from helpers import _tiny_checkpoint


@needs_ckpt
def test_split_forward_equals_forward():
    from mppp.mask.model import load_model
    model, _ = load_model(CKPT, "cpu")
    x = torch.randn(1, 3, 128, 160)
    with torch.no_grad():
        assert torch.allclose(model(x), model.decode(model.features(x), x.shape[-2:]))


def test_offline_fails_fast_with_instructions(monkeypatch, tmp_path):
    import time
    import mppp.mask.model as M
    monkeypatch.setattr(M, "_hub_reachable", lambda *a, **k: False)
    monkeypatch.setattr(M, "_cached_weights", lambda name: None)
    t = time.time()
    with pytest.raises(RuntimeError, match="init_from"):
        M.ConvNeXtSeg("convnext_base", pretrained=True)
    assert time.time() - t < 10


def test_stride4_decoder_shapes_and_old_cards_load(tmp_path):
    from mppp.mask.model import ConvNeXtSeg, load_model, write_card
    m4 = ConvNeXtSeg("convnext_tiny", stride4=True).eval()
    m8 = ConvNeXtSeg("convnext_tiny", stride4=False).eval()
    x = torch.randn(1, 3, 96, 128)
    with torch.no_grad():
        assert m4(x).shape == m8(x).shape == (1, 1, 96, 128)
        assert len(m4.features(x)) == 4 and len(m8.features(x)) == 3
    assert any(k.startswith("s4.") for k in m4.state_dict()) and not any(k.startswith("s4.") for k in m8.state_dict())
    # a pre-v0p6 checkpoint + card (no stride4 key) still loads strictly
    torch.save({"model": m8.state_dict()}, tmp_path / "old.pt")
    write_card(tmp_path / "old.pt", backbone="convnext_tiny")
    import json
    card = json.loads((tmp_path / "old.json").read_text())
    card.pop("stride4"); card.pop("quad_split_above"); card.pop("quad_overlap")
    (tmp_path / "old.json").write_text(json.dumps(card))
    model, c = load_model(tmp_path / "old.pt")
    assert model.stride4 is False and c["quad_split_above"] is None


@pytest.mark.parametrize("h,w,o", [(1920, 2560, 64), (3840, 5120, 64), (961, 2561, 64), (100, 3000, 500)])
def test_quad_cores_tile_frame_exactly(h, w, o):
    from mppp.mask.model import quad_boxes
    cover = np.zeros((h, w), int)
    for (y0, y1, x0, x1), (cy0, cy1, cx0, cx1) in quad_boxes(h, w, o):
        assert y0 <= cy0 < cy1 <= y1 and x0 <= cx0 < cx1 <= x1                  # core inside crop
        cover[cy0:cy1, cx0:cx1] += 1
        assert (y1 - y0) - (cy1 - cy0) <= o and (x1 - x0) - (cx1 - cx0) <= o
    assert (cover == 1).all()


def test_quad_inference_stitches_cores(monkeypatch):
    """With a pixel-wise stand-in for the network, the stitched result equals the whole frame."""
    import mppp.mask.infer as I
    calls = []
    def fake(model, card, img):
        calls.append(img.shape[:2])
        return img[..., 0].astype(np.float32) / 255.0
    monkeypatch.setattr(I, "_predict_frame", fake)
    rng = np.random.default_rng(1)
    img = rng.integers(1, 255, (1920, 2560, 3)).astype(np.uint8)
    img[1000:] = 0                                          # sub-frame padding: bottom quads not run
    card = {"quad_split_above": 2500, "quad_overlap": 64}
    p = I.predict_probability(None, card, img)
    assert np.allclose(p, img[..., 0] / 255.0)
    assert calls == [(960 + 64, 1280 + 64)] * 4        # bottom crops start 64 px above the centre line
    calls.clear()
    I.predict_probability(None, card, img[:, :1648])        # long side 1648: one pass
    assert calls == [(1920, 1648)]
    calls.clear()
    I.predict_probability(None, {"input_size": 1648}, img)  # pre-v0p6 card: never split
    assert calls == [(1920, 2560)]
    calls.clear()
    img2 = img.copy(); img2[960:] = 0                       # half-height sub-frame padded to the full frame
    p2 = I.predict_probability(None, card, img2)
    assert len(calls) == 2 and np.allclose(p2, img2[..., 0] / 255.0)   # padding-only quadrants skipped


def test_pretrained_file_auto(tmp_path, monkeypatch):
    from mppp.mask import train as T
    monkeypatch.setattr("mppp.paths.checkpoints_dir", lambda: tmp_path)
    assert T.resolve_pretrained_file("auto", "convnext_tiny") is None             # -> hub / HF cache
    (tmp_path / "convnext_tiny.fb_in22k.safetensors").write_bytes(b"x")
    assert T.resolve_pretrained_file("auto", "convnext_tiny") == str(tmp_path / "convnext_tiny.fb_in22k.safetensors")
    assert T.resolve_pretrained_file("auto", "convnext_base") is None
    assert T.resolve_pretrained_file(None, "convnext_tiny") is None
    assert T.resolve_pretrained_file("D:/w.safetensors", "convnext_tiny") == "D:/w.safetensors"


class NotATensor:                                                              # any pickled Python object
    pass


def _tiny_checkpoint(tmp_path):
    import torch
    from mppp.mask.model import ConvNeXtSeg, write_card
    m = ConvNeXtSeg("convnext_tiny", pretrained=False, fpn_width=32, stride4=True)
    ck = tmp_path / "tiny_s4_seg_test.pt"
    torch.save({"model": m.state_dict(), "val_iou": 0.5, "epoch": 1}, ck)
    write_card(ck, backbone="convnext_tiny", fpn_width=32, stride4=True, threshold=0.5, canvas=[64, 64],
               input_size=64, val_iou=0.5, name=ck.stem)
    return ck


def test_safetensors_export_is_deterministic_and_equivalent(tmp_path):
    import torch
    from mppp.mask.hub import export_safetensors
    from mppp.mask.model import load_model, read_card
    ck = _tiny_checkpoint(tmp_path)
    a = export_safetensors(ck, out=tmp_path / "a.safetensors")
    b = export_safetensors(ck, out=tmp_path / "b.safetensors")
    assert a["sha256"] == b["sha256"] and a["bytes"] > 0
    card = read_card(tmp_path / "a.safetensors")                             # embedded, no sidecar needed
    assert card["fpn_width"] == 32 and card["exported_from"] == ck.name and not (tmp_path / "a.json").exists()
    m1, _ = load_model(ck)
    m2, _ = load_model(tmp_path / "a.safetensors")
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x))


def test_pt_is_loaded_without_pickle_code(tmp_path):
    import torch
    from mppp.mask.model import read_state_dict_checkpoint

    bad = tmp_path / "bad.pt"
    torch.save({"model": {}, "x": NotATensor()}, bad)
    with pytest.raises(RuntimeError, match="not a plain-tensor checkpoint"):
        read_state_dict_checkpoint(bad)


def test_registry_fetch_install_resolve(tmp_path, monkeypatch):
    from mppp.mask import hub
    ck = _tiny_checkpoint(tmp_path)
    st = hub.export_safetensors(ck, out=tmp_path / "remote" / "m.safetensors")
    reg = {"default": "m1", "models": {"m1": {"file": "m1.safetensors", "sha256": st["sha256"],
                                               "urls": [(tmp_path / "nowhere.safetensors").as_uri(),
                                                        Path(st["path"]).as_uri()]}}}
    (tmp_path / "models.json").write_text(json.dumps(reg))
    monkeypatch.setattr(hub, "registry_path", lambda: tmp_path / "models.json")
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        hub.resolve_checkpoint("m1", download=False).stat()
    p = hub.resolve_checkpoint()                                               # default -> download (2nd URL)
    assert p == tmp_path / "cache" / "models" / "m1.safetensors" and hub.sha256_file(p) == st["sha256"]
    assert hub.resolve_checkpoint(ck.name) == ck                               # a file in checkpoints_dir()
    assert hub.resolve_checkpoint(str(ck)) == ck
    with pytest.raises(FileNotFoundError, match="Mask checkpoint not found"):
        hub.resolve_checkpoint("nope.pt")
    # a corrupted download is rejected and kept aside
    reg["models"]["m1"]["sha256"] = "0" * 64
    (tmp_path / "models.json").write_text(json.dumps(reg))
    with pytest.raises(FileNotFoundError, match="SHA-256"):
        hub.fetch_model("m1", force=True)
    # install a local model under the registry name (different SHA -> warning)
    with pytest.warns(UserWarning, match="differs from the registry"):
        q = hub.install_model(ck, "m1")
    assert q.is_file()
    from mppp.mask import get_model
    model, card = get_model(str(q), "cpu")          # 0.31.2: the registry name would fetch the released file again
    assert card["fpn_width"] == 32


def test_export_updates_registry(tmp_path, monkeypatch):
    from mppp.mask import hub
    ck = _tiny_checkpoint(tmp_path)
    (tmp_path / "models.json").write_text(json.dumps({"default": "m1", "models": {"m1": {"file": "m1.safetensors",
                                                                                           "urls": ["u"]}}}))
    monkeypatch.setattr(hub, "registry_path", lambda: tmp_path / "models.json")
    info = hub.export_safetensors(ck, name="m1", update_registry=True)
    reg = json.loads((tmp_path / "models.json").read_text())
    assert Path(info["path"]).name == "m1.safetensors" and reg["models"]["m1"]["sha256"] == info["sha256"]
    assert reg["models"]["m1"]["urls"] == ["u"] and reg["models"]["m1"]["source_checkpoint"] == ck.name


def test_released_registry_entry():
    from mppp.mask.hub import load_registry
    reg = load_registry()
    e = reg["models"][reg["default"]]
    assert e["file"].endswith(".safetensors") and len(e.get("sha256", "")) == 64
    assert e["urls"][0].startswith("https://huggingface.co/")        # 0.31.2: the default comes from Hugging Face


def test_export_does_not_depend_on_the_mppp_version(tmp_path, monkeypatch):
    from helpers import _tiny_checkpoint
    import mppp
    from mppp.mask.hub import export_safetensors
    from mppp.mask.model import read_card
    ck = _tiny_checkpoint(tmp_path)
    a = export_safetensors(ck, out=tmp_path / "a.safetensors")
    monkeypatch.setattr(mppp, "__version__", "9.9.9")
    b = export_safetensors(ck, out=tmp_path / "b.safetensors")
    assert a["sha256"] == b["sha256"] and "exported_with_mppp" not in read_card(tmp_path / "a.safetensors")


def test_download_failure_falls_back_to_local_checkpoint(tmp_path, monkeypatch):
    """v0p14.1: the model is not published yet (HF 401, GitHub 404) -> use the local best checkpoint."""
    import shutil
    from helpers import _tiny_checkpoint
    from mppp.mask import hub
    (tmp_path / "src").mkdir()
    ck = _tiny_checkpoint(tmp_path / "src")
    ckdir = tmp_path / "checkpoints"
    ckdir.mkdir()
    shutil.copy2(ck, ckdir / "convnext_tiny_s4_seg_best.pt")
    shutil.copy2(ck.with_suffix(".json"), ckdir / "convnext_tiny_s4_seg_best.json")
    ref = hub.export_safetensors(ckdir / "convnext_tiny_s4_seg_best.pt", out=tmp_path / "ref.safetensors", name="m1")
    reg = {"default": "m1", "models": {"m1": {"file": "m1.safetensors", "sha256": ref["sha256"],
                                               "source_checkpoint": "convnext_tiny_s4_seg_best.pt",
                                               "backbone": "convnext_tiny", "stride4": True,
                                               "urls": [(tmp_path / "missing.safetensors").as_uri()]}}}
    (tmp_path / "models.json").write_text(json.dumps(reg))
    monkeypatch.setattr(hub, "registry_path", lambda: tmp_path / "models.json")
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(ckdir))
    assert hub.local_candidates("m1") == [ckdir / "convnext_tiny_s4_seg_best.pt"]
    with pytest.raises(FileNotFoundError, match="MPPP_MASK_LOCAL_FALLBACK"):
        hub.resolve_checkpoint("m1")                                      # 0.31.2: no local stand-in by default
    monkeypatch.setenv("MPPP_MASK_LOCAL_FALLBACK", "1")
    p = hub.resolve_checkpoint("m1")                                      # download fails -> local install
    assert p == tmp_path / "cache" / "models" / "m1.safetensors" and hub.sha256_file(p) == ref["sha256"]
    with pytest.raises(FileNotFoundError, match="stands in only with"):
        hub.fetch_model("m1", force=True, local_fallback=False)


def test_mask_model_v2_is_registered():
    import mppp
    from mppp.mask.hub import load_registry
    reg = load_registry()
    assert reg["default"] == "mppp_mask_v3" == mppp.default_config()["masking"]["checkpoint"]   # v0p22.4
    v2 = reg["models"]["mppp_mask_v2"]
    assert v2["source_checkpoint"] == "convnext_tiny_s4_seg_20260925.pt" and v2["val_iou"] > 0.977
    assert v2["sha256"] == "227e483369e11d6e36ce3517cf9f6d6ac60a9a50c1e301f9e7ae53d0cef569b9"
    assert all(u.endswith(v2["file"]) and "mask-v2" in u or "huggingface" in u for u in v2["urls"])
    assert "mppp_mask_v1" in reg["models"]                                         # the previous model stays


def test_default_mask_model_is_v3():
    from mppp.mask.hub import load_registry, default_model_name, local_candidates
    import mppp
    reg = load_registry()
    assert default_model_name() == "mppp_mask_v3" and mppp.load_config({})["masking"]["checkpoint"] == "mppp_mask_v3"
    e = reg["models"]["mppp_mask_v3"]
    assert e["source_checkpoint"] == "convnext_tiny_s4_seg_20260925b.pt" and e["file"].endswith("_v3.safetensors")
    assert len(e["sha256"]) == 64 and e["bytes"] > 10 ** 8 and e["val_iou"] > 0.979
    assert isinstance(local_candidates("mppp_mask_v3"), list)     # the .pt in checkpoints/ stands in until the release is published


def _registry(tmp_path, monkeypatch, sha, urls):
    from mppp.mask import hub
    reg = {"default": "m1", "models": {"m1": {"file": "m1.safetensors", "sha256": sha, "urls": urls,
                                               "source_checkpoint": "tiny_s4_seg_test.pt", "hf_repo": "me/mask"}}}
    (tmp_path / "models.json").write_text(json.dumps(reg))
    monkeypatch.setattr(hub, "registry_path", lambda: tmp_path / "models.json")
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(tmp_path))
    monkeypatch.delenv("MPPP_MASK_LOCAL_FALLBACK", raising=False)
    return hub


def test_default_v3_is_on_hugging_face():
    from mppp.mask.hub import load_registry, hf_repo
    reg = load_registry()
    e = reg["models"][reg["default"]]
    assert reg["default"] == "mppp_mask_v3" and hf_repo() == "ctate7163/mppp-mask"
    assert e["urls"] == ["https://huggingface.co/ctate7163/mppp-mask/resolve/main/" + e["file"]]
    # the released file carries the real card (stride-4 decoder); 0.31.1's registry hash was a card-less export
    assert e["stride4"] is True and e["sha256"].startswith("46830126") and e["bytes"] == 124826276


def test_cached_file_that_is_not_the_release_is_replaced(tmp_path, monkeypatch):
    ck = _tiny_checkpoint(tmp_path)
    from mppp.mask import hub as h
    rel = h.export_safetensors(ck, out=tmp_path / "remote" / "m1.safetensors")
    hub = _registry(tmp_path, monkeypatch, rel["sha256"], [Path(rel["path"]).as_uri()])
    cached = hub.model_path("m1")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not the released model")
    p = hub.resolve_checkpoint("m1")
    assert p == cached and hub.sha256_file(p) == rel["sha256"]
    assert cached.with_suffix(".sha256-mismatch").read_bytes() == b"not the released model"
    assert hub.resolve_checkpoint("m1") == cached                        # verified copy is reused


def test_no_local_stand_in_by_default(tmp_path, monkeypatch):
    ck = _tiny_checkpoint(tmp_path)
    hub = _registry(tmp_path, monkeypatch, "0" * 64, [(tmp_path / "missing.safetensors").as_uri()])
    with pytest.raises(FileNotFoundError, match="MPPP_MASK_LOCAL_FALLBACK"):
        hub.fetch_model("m1")
    assert not hub.model_path("m1").exists()
    monkeypatch.setenv("MPPP_MASK_LOCAL_FALLBACK", "1")
    with pytest.warns(UserWarning, match="differs from the registry"):
        assert hub.fetch_model("m1").is_file()


def test_export_refuses_a_checkpoint_without_its_card(tmp_path):
    from mppp.mask import hub
    ck = _tiny_checkpoint(tmp_path)
    ck.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError, match="no model card"):
        hub.export_safetensors(ck, out=tmp_path / "x.safetensors")


def test_hf_token_goes_only_to_hugging_face(monkeypatch):
    from mppp.mask import hub
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert hub._headers("https://huggingface.co/a/b/resolve/main/f")["Authorization"] == "Bearer hf_test"
    assert "Authorization" not in hub._headers("https://github.com/a/b/releases/download/t/f")


def test_upload_checks_sha_and_uploads_file_and_card(tmp_path, monkeypatch):
    ck = _tiny_checkpoint(tmp_path)
    from mppp.mask import hub as h
    ref = h.export_safetensors(ck, out=tmp_path / "ref.safetensors", name="m1")      # release_name is in the card
    url = "https://huggingface.co/me/mask/resolve/main/m1.safetensors"
    hub = _registry(tmp_path, monkeypatch, ref["sha256"], [url])
    calls = []

    class FakeApi:
        def create_repo(self, repo_id, **kw):
            calls.append(("create", repo_id, kw.get("private")))

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, **kw):
            calls.append(("upload", Path(path_or_fileobj).name, path_in_repo, repo_id))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    card = tmp_path / "card.md"
    card.write_text("# card")
    assert hub.upload_model("m1", card=card) == url                   # exported from source_checkpoint first
    assert (tmp_path / "m1.safetensors").is_file()
    assert calls == [("create", "me/mask", False), ("upload", "m1.safetensors", "m1.safetensors", "me/mask"),
                     ("upload", "card.md", "README.md", "me/mask")]
    (tmp_path / "m1.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="not uploading"):
        hub.upload_model("m1", card=card)


def test_verify_reports_each_url(tmp_path, monkeypatch):
    ck = _tiny_checkpoint(tmp_path)
    from mppp.mask import hub as h
    rel = h.export_safetensors(ck, out=tmp_path / "remote" / "m1.safetensors")
    hub = _registry(tmp_path, monkeypatch, rel["sha256"],
                    [Path(rel["path"]).as_uri(), (tmp_path / "missing.safetensors").as_uri()])
    res = hub.verify_model("m1")
    assert [r["ok"] for r in res] == [True, False]
