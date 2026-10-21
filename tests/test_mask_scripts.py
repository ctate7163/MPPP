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
