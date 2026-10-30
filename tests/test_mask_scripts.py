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
