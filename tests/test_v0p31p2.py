"""0.31.2: the default mask model always comes from Hugging Face."""
import json
import sys
import types
from pathlib import Path

import pytest

from test_v0p13 import _tiny_checkpoint


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
