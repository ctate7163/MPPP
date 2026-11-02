"""scripts/train_mask.py (headless retraining) and scripts/make_mask_figure.py (the paper's mask figure) end to end on
a tiny synthetic set (CPU)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_train_smoke_then_figure(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_mask.py"), "--smoke"], capture_output=True,
                       text=True, timeout=900, env={**__import__("os").environ, "TMPDIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr[-2000:]
    smoke = sorted(tmp_path.glob("mppp_mask_smoke_*"))[-1]
    out = tmp_path / "fig"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_mask_figure.py"), "--ckpt",
                        str(smoke / "ckpt" / "smoke.pt"), "--data", str(smoke / "set"), "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "figure1_masks.png").is_file() and (out / "mask_metrics_by_camera.md").is_file()
    # 31 Oct: the A/B of two models on the same held-out frames, with a "seen" set
    seen = tmp_path / "older" / "masks"
    seen.mkdir(parents=True)
    first = sorted((smoke / "set" / "masks").glob("*.png"))[0]
    (seen / first.name).write_bytes(first.read_bytes())
    ab = tmp_path / "ab"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_mask_models.py"), "--data", str(smoke / "set"),
                        "--split-ckpt", str(smoke / "ckpt" / "smoke.pt"), "--models", str(smoke / "ckpt" / "smoke.pt"),
                        str(smoke / "ckpt" / "smoke.pt"), "--seen", str(seen.parent), "--seen-split",
                        str(smoke / "ckpt" / "smoke.pt"), str(smoke / "set"), "--out", str(ab)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    md = (ab / "mask_ab.md").read_text()
    assert "All held-out frames" in md and "| all |" in md and (ab / "mask_ab_frames.csv").is_file()


@pytest.mark.slow
def test_train_smoke_with_the_paper_outputs(tmp_path):
    """0.95.2 (train_masks.bat): training, then the JOSS outputs in one run."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    out = tmp_path / "paper"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_mask.py"), "--smoke", "--paper", str(out)],
                       capture_output=True, text=True, timeout=900, env={**__import__("os").environ, "TMPDIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr[-2000:]
    for f in ("figure1_masks.png", "mask_metrics_by_camera.md", "summary.md", "summary.json", "smoke.card.json",
              "training_curves.png", "smoke.safetensors"):
        assert (out / f).is_file(), f
    s = (out / "summary.md").read_text()
    assert "image-mask pairs" in s and "horizontal flip" in s and "zoom" in s


def test_zoom_crop_of_large_ecam_frames():
    """0.95.2 (Christian, 30 Oct): large Navcam/Hazcam frames zoomed in by up to 2 somewhere in the frame."""
    pytest.importorskip("torch")
    import random
    import numpy as np
    from mppp.mask.training.train import zoom_crop
    rng = random.Random(1)
    im = np.full((3840, 5120, 3), 100, np.uint8)
    ms = np.zeros((3840, 5120), np.uint8)
    zs = []
    for _ in range(50):
        ci, cm, z = zoom_crop(im, ms, 2.0, rng)
        assert ci.shape[:2] == cm.shape and 1.0 <= z <= 2.0
        assert abs(ci.shape[1] - 5120 / z) <= 1 and abs(ci.shape[0] - 3840 / z) <= 1
        zs.append(z)
    assert min(zs) < 1.3 and max(zs) > 1.7
    pad = im.copy()
    pad[:, 2560:] = 0                                                # a half frame padded to the full detector
    for _ in range(20):
        ci, cm, z = zoom_crop(pad, ms, 2.0, rng)
        assert z == 1.0 or np.mean(ci[::16, ::16].max(axis=2) > 0) >= 0.5


def test_training_uses_the_pipeline_images_only_by_default():
    """30 Oct (Christian: "only trains on the images that went through the standard mppp pipeline")."""
    src = (ROOT / "scripts" / "train_mask.py").read_text()
    assert 'image_dirs = ("images", "images_variable") if (a.variable or a.smoke) else ("images",)' in src
    assert "if a.variable:" in src                     # images_variable/ is only built on request


def test_random_tone_augmentation(tmp_path):
    """0.95.3 (mask v4.1): a random tone curve per training sample (v3's images_variable recipe), invalid pixels
    kept at 0; make_variable gives the same images as before (same draws)."""
    import numpy as np
    from mppp.mask.training.dataset import VariableRecipe, make_variable, random_tone
    rgb = np.random.default_rng(1).integers(1, 256, (40, 60, 3)).astype(np.uint8)
    rgb[:, :5] = 0
    valid = rgb.max(axis=2) > 0
    a, pa = make_variable(rgb, valid, "NLF_x.png")
    b, pb = random_tone(rgb, valid, np.random.default_rng(__import__("mppp.mask.training.dataset",
                                                                       fromlist=["_seed"])._seed("NLF_x.png")))
    assert (a == b).all() and pa == pb
    r = VariableRecipe()
    assert r.lo[0] <= pa["lo"] <= r.lo[1] and r.gamma[0] <= pa["gamma"] <= r.gamma[1]
    c, _ = random_tone(rgb, valid, np.random.default_rng(7))
    assert (c[:, :5] == 0).all() and c[valid].min() >= 1 and not (c == a).all()
    tm = (ROOT / "scripts" / "windows" / "train_masks.bat").read_bytes()
    assert b"--epochs 20" in tm and b"--tone-prob 0.5" in tm
    src = (ROOT / "scripts" / "train_mask.py").read_text()
    assert "tone_prob=a.tone_prob" in src


def test_mask_dataset_applies_the_tone_curve(tmp_path):
    pytest.importorskip("torch")
    import cv2
    import numpy as np
    from mppp.mask.training.train import Item, MaskDataset
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    img = np.full((60, 80, 3), 120, np.uint8)
    img[:, :4] = 0
    cv2.imwrite(str(tmp_path / "images" / "NLF_a.png"), img)
    cv2.imwrite(str(tmp_path / "masks" / "NLF_a.png"), np.full((60, 80), 255, np.uint8))
    it = Item(image=str(tmp_path / "images" / "NLF_a.png"), mask=str(tmp_path / "masks" / "NLF_a.png"))
    plain = MaskDataset([it], size=80, canvas=(80, 60))[0][0]
    toned = MaskDataset([it], size=80, canvas=(80, 60), tone_prob=1.0)[0][0]
    assert (plain[:, 10:] == 120).all() and not (toned[:, 10:] == 120).all()
    assert (toned[:, :4] == 0).all()


def test_items_as_trained_drops_the_newest_masks(tmp_path):
    """2 Nov 2026: a model trained on fewer masks gets its split rebuilt without the newest ones."""
    import os
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from compare_mask_models import items_as_trained
    items = []
    for i in range(6):
        m = tmp_path / f"m{i}.png"
        m.write_bytes(b"x")
        os.utime(m, (1000 + i, 1000 + i))
        items.append(SimpleNamespace(mask=str(m), image=str(m)))
    card = {"name": "old", "training": {"n_train": 3, "n_val": 1}}
    kept = items_as_trained(items, card)
    assert [Path(it.mask).name for it in kept] == ["m0.png", "m1.png", "m2.png", "m3.png"]
    assert items_as_trained(items, {"training": {"n_train": 5, "n_val": 1}}) == items
    assert items_as_trained(items, {}) == items
