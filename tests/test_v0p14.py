"""v0p14: reprocess only the images left in the output folder; version-free model export."""
import json

import pytest

from conftest import NLF, ZL0, needs_data


@needs_data
def test_only_existing_rebuilds_manifest_from_remaining_images(tmp_path):
    import mppp
    from mppp.process import filter_to_existing
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": True}})
    wp = mppp.load_waypoints()
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False)
    assert man["n_processed"] == 2 and man["only_existing"] is None
    (tmp_path / "images_png8" / (ZL0.stem + ".png")).unlink()                 # the user removes one frame
    (tmp_path / "images_png8" / "NOT_SELECTED_0001.png").write_bytes(b"x")    # and an unrelated file is there
    kept, rep = filter_to_existing([NLF, ZL0], tmp_path, "PNG8")
    assert kept == [NLF] and rep["removed"] == [ZL0.name] and rep["not_in_selection"] == ["NOT_SELECTED_0001"]
    man = mppp.process_images([NLF, ZL0], tmp_path, cfg, wp, progress=False, only_existing="PNG8")
    assert man["n_requested"] == man["n_processed"] == 1
    assert [m["source_product"].split("/")[-1].split("\\")[-1] for m in man["images"]] == [NLF.name]
    assert man["only_existing"]["n_removed"] == 1
    on_disk = json.loads((tmp_path / f"mppp_manifest_{mppp.VERSION_TAG}.json").read_text())
    assert on_disk["n_processed"] == 1 and on_disk["only_existing"]["folder"].endswith("images_png8")
    refs = [l for l in (tmp_path / "references.txt").read_text().splitlines()
            if l and not l.startswith("#") and not l.startswith("filename")]
    assert len(refs) == 1 and NLF.stem in refs[0]
    assert (tmp_path / "images_png8" / (NLF.stem + ".png")).is_file()          # regenerated, nothing deleted
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mppp.process_images([NLF], tmp_path, cfg, wp, progress=False, only_existing="TIFF16")
    with pytest.raises(ValueError, match="none of the"):
        mppp.process_images([ZL0], tmp_path, cfg, wp, progress=False, only_existing="PNG8")


def test_export_does_not_depend_on_the_mppp_version(tmp_path, monkeypatch):
    from test_v0p13 import _tiny_checkpoint
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
    from test_v0p13 import _tiny_checkpoint
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
    p = hub.resolve_checkpoint("m1")                                      # download fails -> local install
    assert p == tmp_path / "cache" / "models" / "m1.safetensors" and hub.sha256_file(p) == ref["sha256"]
    with pytest.raises(FileNotFoundError, match="No local fallback"):
        hub.fetch_model("m1", force=True, local_fallback=False)
