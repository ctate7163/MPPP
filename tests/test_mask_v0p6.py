"""v0p6: stride-4 decoder, quad split (training + inference), scan_dataset(missing='skip')."""
import cv2
import numpy as np
import pytest
import torch


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


def test_init_from_pre_v0p6_checkpoint_keeps_old_decoder(tmp_path):
    from mppp.mask.model import ConvNeXtSeg
    from mppp.mask.train import init_from_checkpoint
    src = ConvNeXtSeg("convnext_tiny", stride4=False)
    torch.save({"model": src.state_dict()}, tmp_path / "old.pt")
    dst = ConvNeXtSeg("convnext_tiny", stride4=True)
    init_from_checkpoint(dst, tmp_path / "old.pt")
    sd = dst.state_dict()
    assert torch.equal(sd["aspp.proj.weight"], src.state_dict()["aspp.proj.weight"])     # fpn/aspp/head copied
    assert torch.equal(sd["head.3.weight"], src.state_dict()["head.3.weight"])
    assert not torch.equal(sd["fuse.0.weight"], torch.zeros_like(sd["fuse.0.weight"]))  # fresh init, not zeros


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


def _big_dataset(root, h=120, w=200, n=4):
    for d in ("images", "masks"):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for k in range(n):
        img = rng.integers(1, 255, (h, w, 3), dtype=np.uint8)
        m = np.zeros((h, w), np.uint8); m[h // 3:, :] = 255
        if k == 0:                                           # sub-frame padded to the detector frame
            img[h // 2 - 10:] = 0; m[h // 2 - 10:] = 0      # below the bottom crops' top edge (overlap 8)
        cv2.imwrite(str(root / "images" / f"s{k}.png"), img)
        cv2.imwrite(str(root / "masks" / f"s{k}.png"), m)


def test_expand_quads_and_dataset_crop(tmp_path):
    from mppp.mask.train import MaskDataset, expand_quads, scan_dataset
    _big_dataset(tmp_path)
    items = scan_dataset(tmp_path, image_dirs=("images",))
    q, n = expand_quads(items, 150, overlap=8)
    assert n["split"] == 4 and n["empty_skipped"] == 2 and len(q) == 14
    assert {i.tile for i in q} == {0, 1, 2, 3} and q[0].label.endswith("#q0")
    same, n0 = expand_quads(items, None)
    assert same == items and n0["split"] == 0
    ds = MaskDataset([i for i in q if i.tile == 3], size=64, canvas=(64, 64), quad_overlap=8)
    im, ms, _ = ds[0]
    assert im.shape == (64, 64, 3) and ms.shape == (64, 64)


def test_train_with_quads_writes_card_and_infers(tmp_path):
    from mppp.mask.infer import predict_probability
    from mppp.mask.model import load_model
    from mppp.mask.train import TrainConfig, scan_dataset, train
    _big_dataset(tmp_path / "ds", n=6)
    cfg = TrainConfig(input_size=64, canvas=(64, 64), epochs=1, batch_size=2, pretrained=False,
                      val_ratio=0.3, val_preview_every=0, quad_split_above=150, quad_overlap=8)
    out = train(scan_dataset(tmp_path / "ds", image_dirs=("images",)), tmp_path / "q.pt", cfg,
                device="cpu", debug_dir=None)
    model, card = load_model(out, "cpu")
    assert card["stride4"] is True and card["quad_split_above"] == 150 and card["quad_overlap"] == 8
    assert card["backbone"] == "convnext_tiny" and model.stride4
    assert card["training"]["n_train"] > card["training"]["n_train_frames"]
    img = np.full((120, 200, 3), 100, np.uint8)
    assert predict_probability(model, card, img).shape == (120, 200)


def test_scan_dataset_missing_skip(tmp_path):
    from mppp.mask.train import scan_dataset
    _big_dataset(tmp_path)
    (tmp_path / "masks" / "s1.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing='skip'"):
        scan_dataset(tmp_path, image_dirs=("images",))
    with pytest.warns(UserWarning, match="skipped 1"):
        assert len(scan_dataset(tmp_path, image_dirs=("images",), missing="skip")) == 3


def test_curves_axes_start_at_zero(tmp_path):
    import csv, subprocess, sys
    from conftest import ROOT
    d = tmp_path / "dbg"; d.mkdir()
    cols = ["epoch", "iter", "n_iter", "loss", "iou", "loss_interval", "iou_interval", "grad_norm", "lr",
            "aspp_pre_bn_peak", "elapsed_s", "step"]
    with (d / "log.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for k in range(1, 6):
            w.writerow([1, k * 10, 50, 0.5 / k, 0.9, 0.5 / k, 0.9 + k / 100, 1, 1e-4, 3, k, k * 10])
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r});"
            "import mppp.mask.monitor as M, matplotlib.figure as F;"
            "lims=[];orig=F.Figure.savefig\n"
            "def sv(self,*a,**k):\n lims.extend(float(ax.get_ylim()[0]) for ax in self.axes[:2]); return orig(self,*a,**k)\n"
            f"F.Figure.savefig=sv; M.plot_curves_file({str(d)!r}); print(lims)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[0.0, 0.0]"


def test_imagenet_safetensors_via_init_from_or_pretrained_file(tmp_path):
    """22 Sep: init_from=<timm model.safetensors> failed with 'UnpicklingError: invalid load key'."""
    import timm
    from safetensors.torch import save_file
    from mppp.mask.model import ConvNeXtSeg
    from mppp.mask.train import init_from_checkpoint
    ref = timm.create_model("convnext_tiny", pretrained=False, num_classes=21841)      # in22k-shaped head
    f = tmp_path / "convnext_tiny_model.safetensors"
    save_file({k: v.contiguous() for k, v in ref.state_dict().items()}, str(f))
    k_seg, k_ref = "bb.stages_3.blocks.2.mlp.fc2.weight", "stages.3.blocks.2.mlp.fc2.weight"
    a = ConvNeXtSeg("convnext_tiny", stride4=True)
    rep = init_from_checkpoint(a, f)
    assert rep["backbone_tensors"] == "178/178" and rep["decoder_tensors"].startswith("0/")
    assert torch.equal(a.state_dict()[k_seg], ref.state_dict()[k_ref])
    b = ConvNeXtSeg("convnext_tiny", pretrained=True, pretrained_file=str(f))
    assert torch.equal(b.state_dict()[k_seg], ref.state_dict()[k_ref])
    with pytest.raises(ValueError, match="wrong model size"):
        ConvNeXtSeg("convnext_base", pretrained=True, pretrained_file=str(f))
    torch.save({"model": a.state_dict()}, tmp_path / "seg.pt")
    with pytest.raises(ValueError, match="use init_from"):
        ConvNeXtSeg("convnext_tiny", pretrained=True, pretrained_file=str(tmp_path / "seg.pt"))
