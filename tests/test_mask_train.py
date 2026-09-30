"""Mask training and its monitor (mppp.mask.train, monitor). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import json
import cv2
import numpy as np
import pytest
import torch
from conftest import CKPT, LEGACY_CKPT, needs_ckpt, needs_legacy_ckpt
from pathlib import Path


def _make_dataset(root, n_scenes=6, size=64):
    rng = np.random.default_rng(0)
    for d in ("images", "images_variable", "masks"):
        (root / d).mkdir(parents=True)
    for k in range(n_scenes):
        img = rng.integers(0, 255, (size, int(size * 1.3), 3), dtype=np.uint8)
        m = np.zeros(img.shape[:2], np.uint8)
        m[size // 3:, :] = 255                                        # "terrain" below a horizon
        img[m > 0] = (img[m > 0] * 0.3 + 120).astype(np.uint8)
        name = f"scene{k}.png"
        cv2.imwrite(str(root / "images" / name), img)
        cv2.imwrite(str(root / "images_variable" / name), np.clip(img * 1.5, 0, 255).astype(np.uint8))
        cv2.imwrite(str(root / "masks" / name), m)


def test_grouped_split_keeps_variants_together(tmp_path):
    from mppp.mask.train import grouped_split, scan_dataset
    _make_dataset(tmp_path, n_scenes=20)
    items = scan_dataset(tmp_path)
    assert len(items) == 40
    tr, va = grouped_split(items, 0.2, seed=1)
    assert len(tr) + len(va) == 40 and len(va) == 8
    assert not ({i.mask for i in tr} & {i.mask for i in va})           # no scene on both sides


def test_scan_requires_masks(tmp_path):
    from mppp.mask.train import scan_dataset
    _make_dataset(tmp_path, 2)
    (tmp_path / "masks" / "scene0.png").unlink()
    with pytest.raises(FileNotFoundError, match="no mask"):
        scan_dataset(tmp_path)


def test_letterbox_matches_inference_geometry():
    from mppp.mask.train import letterbox
    im = np.full((120, 160, 3), 200, np.uint8)
    ms = np.full((120, 160), 255, np.uint8)
    a, m = letterbox(im, ms, 80)
    assert a.shape == (80, 80, 3) and m.shape == (80, 80)
    assert m[:60].all() and not m[60:].any() and not a[60:].any()     # pad bottom only, like infer


def test_precision_policy():
    from mppp.mask.train import resolve_precision
    assert resolve_precision("auto", "cpu") == "fp32"
    with pytest.raises(ValueError):
        resolve_precision("fp16", "cuda")                             # plain fp16 is no longer offered


def test_loss_is_fp32_even_for_half_logits():
    from mppp.mask.train import bce_tversky_loss
    logits = torch.full((1, 1, 8, 8), 30.0, dtype=torch.float16)
    gt = torch.ones(1, 8, 8)
    loss, _, _ = bce_tversky_loss(logits, gt)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_training_process_never_imports_matplotlib(tmp_path):
    """v0p3 kernel crash: plotting runs in a child process; the trainer stays matplotlib-free."""
    import subprocess, sys, textwrap
    from conftest import ROOT
    _make_dataset(tmp_path / "ds", n_scenes=4)
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {str(ROOT / 'src')!r})
        from mppp.mask.train import TrainConfig, scan_dataset, train
        cfg = TrainConfig(backbone="convnext_tiny", canvas=None, input_size=64, epochs=1, batch_size=2, print_every=1, val_preview_every=1,
                          n_val_preview=1, pretrained=False, val_ratio=0.25)
        train(scan_dataset({str(tmp_path / 'ds')!r}), {str(tmp_path / 'm.pt')!r}, cfg, device="cpu")
        assert "matplotlib" not in sys.modules, "matplotlib imported in the training process"
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
    dbg = tmp_path / "m_debug"
    assert (dbg / "curves.png").is_file(), (dbg / "curves_render.log").read_text()
    assert not (dbg / "curves.tmp.png").exists()


def test_smoke_train_writes_checkpoint_and_card_and_reloads(tmp_path):
    from mppp.mask.model import load_model
    from mppp.mask.train import TrainConfig, scan_dataset, train
    _make_dataset(tmp_path / "ds", n_scenes=6)
    cfg = TrainConfig(backbone="convnext_tiny", canvas=None, input_size=64, epochs=1, batch_size=4, print_every=1, val_preview_every=1,
                      n_val_preview=2, pretrained=False, val_ratio=0.34)
    out = train(scan_dataset(tmp_path / "ds"), tmp_path / "m.pt", cfg, device="cpu")
    dbg = tmp_path / "m_debug"                                        # monitor output, default location
    assert (dbg / "curves.png").is_file() and (dbg / "log.csv").is_file()
    assert len(list((dbg / "batch").glob("*.png"))) == 2 and len(list((dbg / "val").glob("*.png"))) == 2
    rows = (dbg / "log.csv").read_text().splitlines()
    assert rows[0].startswith("epoch,iter") and len(rows) == 3
    card = json.loads(out.with_suffix(".json").read_text())
    assert card["training"]["split"] == "grouped by mask file" and card["training"]["n_val"] == 4
    assert card["input_size"] == 64 and card["training"]["precision_resolved"] == "fp32"
    assert np.isfinite(card["val_iou"]) and card["history"][0]["aspp_pre_bn_peak"] > 0
    model, c2 = load_model(out, "cpu")
    assert c2["val_iou"] == card["val_iou"]
    with pytest.raises(FileExistsError):
        train(scan_dataset(tmp_path / "ds"), out, cfg, device="cpu")  # never overwrite silently


def test_nonfinite_loss_stops_training(tmp_path, monkeypatch):
    from mppp.mask import train as T
    _make_dataset(tmp_path / "ds", n_scenes=4)
    real = T.forward
    monkeypatch.setattr(T, "forward", lambda *a: real(*a) * float("inf"))
    cfg = T.TrainConfig(backbone="convnext_tiny", canvas=None, input_size=64, epochs=1, batch_size=2, pretrained=False, val_ratio=0.25)
    with pytest.raises(T.NonFiniteLoss, match="scene"):
        T.train(T.scan_dataset(tmp_path / "ds"), tmp_path / "x.pt", cfg, device="cpu")
    assert not (tmp_path / "x.pt").exists()


@needs_legacy_ckpt
def test_reproduces_fp16_overflow_and_fix():
    """
    Known-truth reproduction of the Sept-2026 failure.  In train mode the output
    is invariant to the scale of aspp.proj (BatchNorm follows), which is why
    that scale can drift.  Scaled x4 — the pre-BN peak of the 2025 checkpoint
    (2.1e4 on this input, 2.3e4 on a real frame) times 4 exceeds fp16's 6.55e4 — full fp16 overflows;
    the fp16-backbone / fp32-decoder path reproduces the fp32 result.
    """
    from mppp.mask.model import load_model
    from mppp.mask.train import forward
    model, _ = load_model(LEGACY_CKPT, "cpu")   # a v0p10 default (peak ~44) never overflows
    x = torch.randn(2, 3, 256, 320) * 2
    model.train()
    with torch.no_grad():
        ref = forward(model, x, "fp32", "cpu")
        mixed_before = forward(model, x, "fp16-backbone", "cpu")
        model.aspp.proj.weight.mul_(4.0)
        scaled = forward(model, x, "fp32", "cpu")
        assert torch.allclose(ref, scaled, atol=1e-3, rtol=1e-3)            # BN scale invariance
        with torch.autocast("cpu", dtype=torch.float16):
            full16 = model(x)
        assert not torch.isfinite(full16).all()                            # the failure
        mixed = forward(model, x, "fp16-backbone", "cpu")
        assert torch.isfinite(mixed).all()
        # the fix: the mixed path is as scale-invariant as fp32 ...
        assert torch.allclose(mixed, mixed_before, atol=1e-3, rtol=1e-3)
        # ... and its masks agree with fp32 up to fp16 backbone rounding
        agree = ((torch.sigmoid(mixed) > 0.4) == (torch.sigmoid(ref) > 0.4)).float().mean()
        assert float(agree) > 0.995


def test_new_run_archives_previous_debug_folder(tmp_path):
    """Restarted runs appended to one log.csv (seen on the 22 Sep GPU run); now each run starts clean."""
    from mppp.mask.monitor import TrainingMonitor, latest_images
    d = tmp_path / "x_debug"
    (d / "val").mkdir(parents=True)
    (d / "log.csv").write_text("epoch,iter\n1,50\n")
    (d / "val" / "ep1_it00050.png").write_bytes(b"old")
    m = TrainingMonitor(d)
    assert m.archived is not None and (m.archived / "log.csv").read_text() == "epoch,iter\n1,50\n"
    assert not (d / "log.csv").exists() and (d / "val").is_dir() and not list((d / "val").glob("*.png"))
    assert latest_images(d) == {"curves": None, "val": None, "batch": None}
    assert TrainingMonitor(d).archived is None                         # empty folder: nothing to archive


def test_latest_images_ignores_partial_writes(tmp_path):
    from mppp.mask.monitor import latest_images
    (tmp_path / "val").mkdir()
    (tmp_path / "batch").mkdir()
    for n in ("ep1_it00250.png", "ep1_it01000.png", "ep2_it00250.png", "ep2_it00500.tmp.png"):
        (tmp_path / "val" / n).write_bytes(b"x")
    assert latest_images(tmp_path)["val"].name == "ep2_it00250.png"


def test_viewer_process_runs(tmp_path):
    """`monitor.py watch` (the live window) starts and keeps polling without error (Agg backend here)."""
    import os, subprocess, sys, time
    import cv2
    from conftest import ROOT
    for sub in ("val", "batch"):
        (tmp_path / sub).mkdir()
        cv2.imwrite(str(tmp_path / sub / "ep1_it00050.png"), np.full((40, 60, 3), 200, np.uint8))
    env = dict(os.environ, MPLBACKEND="Agg")
    p = subprocess.Popen([sys.executable, str(ROOT / "src/mppp/mask/monitor.py"), "watch", str(tmp_path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    time.sleep(6)
    alive = p.poll() is None
    p.kill()
    err = p.communicate()[1].decode()
    assert alive, err


def test_canvas_geometry():
    from mppp.mask.train import letterbox
    from mppp.mask.model import canvas_of, fit_to_canvas
    z = np.full((1200, 1648, 3), 100, np.uint8)
    a, m = letterbox(z, np.full((1200, 1648), 255, np.uint8), 1648, (1664, 1248))
    assert a.shape == (1248, 1664, 3) and m[:1200, :1648].all() and not m[1200:].any() and not m[:, 1648:].any()
    assert fit_to_canvas(3840, 5120, 1648, (1664, 1248)) == (1236, 1648)          # Navcam full frame
    assert fit_to_canvas(960, 1280, 1648, (1664, 1248)) == (1236, 1648)           # quarter frame, upsampled
    assert fit_to_canvas(1200, 1648, 1648, (1648, 1648)) == (1200, 1648)          # pre-v0p5 square
    assert canvas_of({"input_size": 1648}) == (1648, 1648)                       # old cards unchanged
    assert canvas_of({"input_size": 1648, "canvas": [1664, 1248]}) == (1664, 1248)
    with pytest.raises(ValueError, match="multiples of 32"):
        from mppp.mask.train import TrainConfig, train
        train([], "x.pt", TrainConfig(canvas=(1648, 1200)), device="cpu")


def test_defaults_requested_22sep_to_24sep():
    from mppp.mask.train import TrainConfig
    c = TrainConfig()
    assert (c.backbone, c.epochs, c.batch_size, c.lr, c.weight_decay, c.threshold, c.input_size) == \
           ("convnext_tiny", 6, 4, 5e-5, 5e-3, 0.5, 1648)                     # v0p10 (24 Sep)
    assert (c.stride4, c.quad_split_above, c.quad_overlap, c.skip_empty_quads) == (True, None, 64, True)
    assert (c.precision, c.print_every, c.val_preview_every, c.n_val_preview) == ("auto", 50, 200, 10)
    assert (c.hflip, c.num_workers, c.pretrained, c.pretrained_file, c.init_from) == (True, 2, True, "auto", None)
    assert c.canvas == (1664, 1248)


def test_rectangular_canvas_train_and_infer(tmp_path):
    from mppp.mask.model import load_model
    from mppp.mask.infer import predict_probability
    from mppp.mask.train import TrainConfig, scan_dataset, train
    _make_dataset(tmp_path / "ds", n_scenes=4)
    cfg = TrainConfig(backbone="convnext_tiny", input_size=96, canvas=(96, 96 - 32), epochs=1, batch_size=2,
                      pretrained=False, val_ratio=0.25, val_preview_every=0)
    out = train(scan_dataset(tmp_path / "ds"), tmp_path / "r.pt", cfg, device="cpu", debug_dir=None)
    model, card = load_model(out, "cpu")
    assert card["canvas"] == [96, 64] and card["threshold"] == 0.5
    img = np.full((50, 70, 3), 120, np.uint8)
    assert predict_probability(model, card, img).shape == (50, 70)


def test_init_from_checkpoint_backbone_only_when_decoder_differs(tmp_path):
    from mppp.mask.model import ConvNeXtSeg
    from mppp.mask.train import init_from_checkpoint
    src = ConvNeXtSeg("convnext_tiny", fpn_width=192)
    torch.save({"model": src.state_dict()}, tmp_path / "old.pt")
    dst = ConvNeXtSeg("convnext_tiny", fpn_width=256)
    rep = init_from_checkpoint(dst, tmp_path / "old.pt")
    assert rep["decoder_tensors"].startswith("0/")                              # all-or-nothing
    k = "bb.stem_0.weight"
    assert torch.equal(dst.state_dict()[k], src.state_dict()[k])
    same = ConvNeXtSeg("convnext_tiny", fpn_width=192)
    rep2 = init_from_checkpoint(same, tmp_path / "old.pt")
    assert rep2["decoder_tensors"].split("/")[0] == rep2["decoder_tensors"].split("/")[1]
    with pytest.raises(ValueError, match="backbone"):
        init_from_checkpoint(ConvNeXtSeg("convnext_base"), tmp_path / "old.pt")


def test_dataloader_workers_with_spawn_like_windows(tmp_path):
    """v0p6 on Windows, num_workers=2: "Can't get local object 'make_collate.<locals>.collate'".
    Windows starts DataLoader workers by spawn, which pickles the collate function; run
    train() under spawn in a child process to reproduce that here."""
    import pickle, subprocess, sys, textwrap
    from conftest import ROOT
    from mppp.mask.train import make_collate
    from mppp.mask.model import DEFAULT_CARD
    pickle.loads(pickle.dumps(make_collate(DEFAULT_CARD)))
    _make_dataset(tmp_path / "ds", n_scenes=6)
    code = textwrap.dedent(f"""
        import multiprocessing as mp, sys
        sys.path.insert(0, {str(ROOT / 'src')!r})
        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            from mppp.mask.train import TrainConfig, scan_dataset, train
            cfg = TrainConfig(input_size=64, canvas=(64, 64), epochs=2, batch_size=2, pretrained=False,
                              val_ratio=0.34, val_preview_every=0, num_workers=2)
            out = train(scan_dataset({str(tmp_path / 'ds')!r}), {str(tmp_path / 'w.pt')!r}, cfg,
                        device="cpu", debug_dir=None)
            print("OK", out)
    """)
    script = tmp_path / "run_spawn.py"
    script.write_text(code)
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OK" in r.stdout


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
    assert cfg["masking"]["checkpoint"] == "mppp_mask_v3"                         # v0p22.4: the released default
    assert best_checkpoint_name({"backbone": "convnext_tiny", "stride4": True}) == "convnext_tiny_s4_seg_best.pt"
    assert best_checkpoint_name({"backbone": "convnext_base"}) == "convnext_base_seg_best.pt"
    cfg["masking"]["checkpoint"] = str(tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError, match="Mask checkpoint not found"):
        process_images(["a.IMG", "b.IMG"], tmp_path / "out", cfg, progress=False)      # once, before any image
    assert not (tmp_path / "out").exists()
    cfg["masking"]["infer_mask"] = False
    assert check_mask_checkpoint(cfg) is None


def test_training_labels_come_from_masks_folder_not_alpha(tmp_path):
    import cv2
    import numpy as np
    from mppp.mask.train import MaskDataset, read_pair, scan_dataset
    for d in ("images", "images_variable", "masks"):
        (tmp_path / d).mkdir()
    h, w = 64, 96
    rgb = np.full((h, w, 3), 90, np.uint8)
    rgb[:, :, 1] = 140
    stale_alpha = np.zeros((h, w), np.uint8)              # alpha says "all excluded" ...
    stale_alpha[:, : w // 4] = 255
    mask = np.zeros((h, w), np.uint8)                      # ... masks/ says the right half is terrain
    mask[:, w // 2:] = 255
    for d in ("images", "images_variable"):
        cv2.imwrite(str(tmp_path / d / "a.png"), np.dstack([rgb, stale_alpha]))
    cv2.imwrite(str(tmp_path / "masks" / "a.png"), mask)

    items = scan_dataset(tmp_path)
    assert len(items) == 2 and all(Path(it.mask) == tmp_path / "masks" / "a.png" for it in items)
    for it in items:
        im, ms = read_pair(it)
        assert im.shape == (h, w, 3) and np.array_equal(im, rgb)       # alpha dropped, RGB not composited
        assert np.array_equal(ms, mask)
        x, y, _ = MaskDataset([it], size=w, canvas=(w, h))[0]
        assert np.array_equal(y, (mask > 127).astype(np.uint8))
        assert np.array_equal(x, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    import pytest
    with pytest.raises(ValueError):
        scan_dataset(tmp_path, image_dirs=("images", "masks"), mask_dir="masks")
