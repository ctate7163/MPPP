"""Mask training: split, precision policy, non-finite handling, and a CPU smoke run."""
import json

import cv2
import numpy as np
import pytest
import torch

from conftest import CKPT, LEGACY_CKPT, needs_ckpt, needs_legacy_ckpt


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


@needs_ckpt
def test_split_forward_equals_forward():
    from mppp.mask.model import load_model
    model, _ = load_model(CKPT, "cpu")
    x = torch.randn(1, 3, 128, 160)
    with torch.no_grad():
        assert torch.allclose(model(x), model.decode(model.features(x), x.shape[-2:]))


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


# ------------------------------------------------------------------ v0p5
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


def test_offline_fails_fast_with_instructions(monkeypatch, tmp_path):
    import time
    import mppp.mask.model as M
    monkeypatch.setattr(M, "_hub_reachable", lambda *a, **k: False)
    monkeypatch.setattr(M, "_cached_weights", lambda name: None)
    t = time.time()
    with pytest.raises(RuntimeError, match="init_from"):
        M.ConvNeXtSeg("convnext_base", pretrained=True)
    assert time.time() - t < 10


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
