"""v0p10: frame audit, checkpoint promotion, default-checkpoint preflight, pretrained_file="auto"."""
import json

import cv2
import numpy as np
import pytest

from test_mask_train import _make_dataset


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from mppp.mask.train import TrainConfig, scan_dataset, train
    root = tmp_path_factory.mktemp("v0p10")
    _make_dataset(root / "ds", n_scenes=8)
    m = cv2.imread(str(root / "ds" / "masks" / "scene3.png"), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(root / "ds" / "masks" / "scene3.png"), 255 - m)          # a deliberately wrong label
    cfg = TrainConfig(input_size=64, canvas=(64, 64), epochs=2, batch_size=2, pretrained=False, lr=1e-3,
                      val_ratio=0.25, val_preview_every=0, num_workers=0)
    items = scan_dataset(root / "ds")
    ck = train(items, root / "ck" / "convnext_tiny_s4_seg_20260101.pt", cfg, device="cpu", debug_dir=None)
    return root, items, ck


def test_audit_ranks_the_wrong_label_first_and_writes_csv(trained):
    from mppp.mask.audit import audit_frames, read_audit_csv, render_worst, summarize, worst_table
    root, items, ck = trained
    rows = audit_frames(items, ck, device="cpu", progress_every=0)
    assert len(rows) == 8 and all(r["variant"] == "images" for r in rows)      # images/ only by default
    assert rows[0]["name"] == "scene3.png" and rows[0]["iou"] < 0.5 < rows[-1]["iou"]
    assert [r["rank"] for r in rows] == list(range(1, 9))
    assert {r["split"] for r in rows} == {"train", "val"}
    back = read_audit_csv(ck.parent / f"{ck.stem}_debug" / "frame_audit.csv")
    assert [r["name"] for r in back] == [r["name"] for r in rows] and abs(back[0]["iou"] - rows[0]["iou"]) < 1e-4
    assert "mean IoU" in summarize(rows) and "scene3.png" in worst_table(rows, 3)
    both = audit_frames(items, ck, device="cpu", image_dirs=None, which="val", out_csv=None, progress_every=0)
    assert all(r["split"] == "val" for r in both) and {r["variant"] for r in both} == {"images", "images_variable"}
    by_err = audit_frames(items, ck, device="cpu", sort_by="error_pct", out_csv=None, progress_every=0)
    assert by_err[0]["name"] == "scene3.png"
    assert all(a["error_pct"] >= b["error_pct"] for a, b in zip(by_err, by_err[1:]))
    pages = render_worst(rows, ck, n=3, per_page=2, device="cpu")
    assert [p.name for p in pages] == ["worst_frames_01.png", "worst_frames_02.png"]
    assert cv2.imread(str(pages[0])) is not None
    assert (pages[0].parent / "worst_frames.txt").read_text().splitlines()[0] == "scene3.png"


def test_audit_reports_size_mismatch(trained, tmp_path):
    import shutil
    from mppp.mask.audit import audit_frames
    from mppp.mask.train import scan_dataset
    root, _, ck = trained
    ds = tmp_path / "ds"
    shutil.copytree(root / "ds", ds)
    cv2.imwrite(str(ds / "masks" / "scene5.png"), np.zeros((10, 10), np.uint8))
    rows = audit_frames(scan_dataset(ds), ck, device="cpu", out_csv=None, progress_every=0)
    assert rows[0]["name"] == "scene5.png" and rows[0]["status"].startswith("size mismatch")


def test_promote_checkpoint(trained, tmp_path):
    import shutil
    from mppp.mask.model import load_model
    from mppp.mask.train import promote_checkpoint
    _, _, ck0 = trained
    ck = tmp_path / ck0.name
    shutil.copy2(ck0, ck)
    shutil.copy2(ck0.with_suffix(".json"), ck.with_suffix(".json"))
    r = promote_checkpoint(ck)
    best = tmp_path / "convnext_tiny_s4_seg_best.pt"
    assert r["promoted"] and best.is_file() and json.loads(best.with_suffix(".json").read_text())["promoted_from"] == ck.name
    load_model(best, "cpu")
    # same dataset, lower val IoU -> refused unless forced; the old best is archived, never deleted
    card = json.loads(ck.with_suffix(".json").read_text())
    worse = tmp_path / "convnext_tiny_s4_seg_20260102.pt"
    shutil.copy2(ck, worse)
    worse.with_suffix(".json").write_text(json.dumps(dict(card, val_iou=card["val_iou"] - 0.1)))
    r = promote_checkpoint(worse)
    assert not r["promoted"] and "higher val IoU" in r["reason"]
    r = promote_checkpoint(worse, force=True)
    assert r["promoted"] and len(r["archived"]) == 2
    assert all((tmp_path / "archive").joinpath(p.split("/")[-1].split("\\")[-1]).is_file() for p in r["archived"])
    assert json.loads(best.with_suffix(".json").read_text())["promoted_from"] == worse.name
    # a different dataset: val IoUs are not comparable, the new one wins
    other = tmp_path / "convnext_tiny_s4_seg_20260103.pt"
    shutil.copy2(ck, other)
    c2 = dict(card, val_iou=0.1, training=dict(card["training"], dataset_fingerprint="different"))
    other.with_suffix(".json").write_text(json.dumps(c2))
    assert promote_checkpoint(other)["promoted"]
    assert promote_checkpoint(best)["reason"] == "already the default"
    assert not promote_checkpoint(other)["promoted"]                            # same file again: no-op
    n = len(list((tmp_path / "archive").iterdir()))
    for _ in range(2):                                                         # same second: unique names
        promote_checkpoint(ck, force=True); promote_checkpoint(other, force=True)
    assert len(list((tmp_path / "archive").iterdir())) == n + 8


def test_default_checkpoint_name_and_preflight(tmp_path, monkeypatch):
    import mppp
    from mppp.process import check_mask_checkpoint, process_images
    from mppp.mask.train import best_checkpoint_name
    cfg = mppp.default_config()
    assert cfg["masking"]["checkpoint"] == "mppp_mask_v2"                         # v0p14.7: the released default
    assert best_checkpoint_name({"backbone": "convnext_tiny", "stride4": True}) == "convnext_tiny_s4_seg_best.pt"
    assert best_checkpoint_name({"backbone": "convnext_base"}) == "convnext_base_seg_best.pt"
    cfg["masking"]["checkpoint"] = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="Mask checkpoint not found"):
        process_images(["a.IMG", "b.IMG"], tmp_path / "out", cfg, progress=False)      # once, before any image
    assert not (tmp_path / "out").exists()
    cfg["masking"]["infer_mask"] = False
    assert check_mask_checkpoint(cfg) is None


def test_pretrained_file_auto(tmp_path, monkeypatch):
    from mppp.mask import train as T
    monkeypatch.setattr("mppp.paths.checkpoints_dir", lambda: tmp_path)
    assert T.resolve_pretrained_file("auto", "convnext_tiny") is None             # -> hub / HF cache
    (tmp_path / "convnext_tiny.fb_in22k.safetensors").write_bytes(b"x")
    assert T.resolve_pretrained_file("auto", "convnext_tiny") == str(tmp_path / "convnext_tiny.fb_in22k.safetensors")
    assert T.resolve_pretrained_file("auto", "convnext_base") is None
    assert T.resolve_pretrained_file(None, "convnext_tiny") is None
    assert T.resolve_pretrained_file("D:/w.safetensors", "convnext_tiny") == "D:/w.safetensors"
