"""0.98.25: the mask model's rover masks for the panoramas where the block processed a frame without it."""
import json
from types import SimpleNamespace

import cv2
import numpy as np

from mppp.ibr.rover_masks import DIR, ensure_rover_masks, mask_file, model_input, skipped_frames


def _block(tmp_path):
    root = tmp_path / "colmap"
    for d in ("images", "masks"):
        (root / d).mkdir(parents=True)
    proc = tmp_path / "processed"
    proc.mkdir()
    img = np.full((40, 60, 3), 120, np.uint8)
    img[:, :5] = 0                                                  # padding: no data
    for n in ("ZL0_A.png", "ZL0_B.png", "NLF_C.png"):
        cv2.imwrite(str(root / "images" / n), img)
        valid = np.full((40, 60), 255, np.uint8)
        valid[:, :5] = 0
        valid[0, :] = 0                                             # a block exclusion the new mask must keep
        cv2.imwrite(str(root / "masks" / (n + ".png")), valid)
    man = {"images": [
        {"source_product": "X/ZL0_A.IMG", "mask": {"inferred": False, "inference_skipped": True}},
        {"source_product": "X/ZL0_B.IMG", "mask": {"inferred": True, "inference_skipped": False}},
        {"source_product": "X/NLF_C.IMG", "mask": {"inferred": False, "inference_skipped": True}}]}
    (proc / "mppp_manifest_v0p98p24.json").write_text(json.dumps(man), encoding="utf-8")
    return SimpleNamespace(root=root, settings={"processed_dir": str(proc)})


def _rover_bottom(img, masking):
    m = np.full(img.shape[:2], 255, np.uint8)
    m[30:, :] = 0                                                   # the "rover": the bottom rows
    return m


def test_skipped_frames_come_from_the_manifest(tmp_path):
    assert skipped_frames(_block(tmp_path)) == {"ZL0_A.png", "NLF_C.png"}


def test_only_the_frames_made_without_the_model_get_a_mask_and_the_block_masks_stay(tmp_path):
    p = _block(tmp_path)
    before = cv2.imread(str(p.root / "masks" / "ZL0_A.png.png"), cv2.IMREAD_GRAYSCALE).copy()
    rep = ensure_rover_masks(p, ["ZL0_A.png", "ZL0_B.png"], infer=_rover_bottom)
    assert rep["skipped"] == 1 and rep["made"] == 1 and rep["have"] == ["ZL0_A.png"]
    m = cv2.imread(str(p.root / DIR / "ZL0_A.png.png"), cv2.IMREAD_GRAYSCALE)
    assert (m[30:] == 0).all()                                      # the rover out
    assert (m[:, :5] == 0).all() and (m[0] == 0).all()             # no data and the block's own exclusions kept
    assert (m[5:30, 10:] == 255).all()
    assert not (p.root / DIR / "ZL0_B.png.png").exists()
    assert (cv2.imread(str(p.root / "masks" / "ZL0_A.png.png"), cv2.IMREAD_GRAYSCALE) == before).all()
    assert mask_file(p.root, "ZL0_A.png") == p.root / DIR / "ZL0_A.png.png"
    assert mask_file(p.root, "ZL0_B.png") == p.root / "masks" / "ZL0_B.png.png"
    rec = json.loads((p.root / DIR / "masks_ibr.json").read_text(encoding="utf-8"))
    assert abs(rec["ZL0_A.png"]["kept"] - float((m > 0).mean())) < 1e-3


def test_masks_are_reused_and_made_again_when_forced(tmp_path):
    p = _block(tmp_path)
    ensure_rover_masks(p, ["ZL0_A.png"], infer=_rover_bottom)
    calls = []
    rep = ensure_rover_masks(p, ["ZL0_A.png"], infer=lambda i, m: calls.append(1) or _rover_bottom(i, m))
    assert rep["made"] == 0 and rep["have"] == ["ZL0_A.png"] and not calls
    rep = ensure_rover_masks(p, ["ZL0_A.png"], infer=lambda i, m: calls.append(1) or _rover_bottom(i, m), force=True)
    assert rep["made"] == 1 and calls == [1]


def test_without_torch_the_panorama_goes_on_with_the_block_masks(tmp_path):
    p = _block(tmp_path)

    def no_torch(img, masking):
        raise ImportError("No module named 'torch'")
    rep = ensure_rover_masks(p, ["ZL0_A.png", "NLF_C.png"], infer=no_torch)
    assert rep["made"] == 0 and set(rep["failed"]) == {"ZL0_A.png", "NLF_C.png"}
    assert mask_file(p.root, "ZL0_A.png") == p.root / "masks" / "ZL0_A.png.png"


def test_model_input_undoes_the_gamma_of_the_8bit_product():
    lin = np.array([[[16.0, 64.0, 200.0]]])
    g = 2.0
    enc = np.clip(lin, 0, None) ** (1 / g) * 256.0 ** (1 - 1 / g)
    assert np.allclose(model_input(enc, {"apply_gamma": True, "gamma": g}), lin, rtol=1e-4)
    assert np.allclose(model_input(lin, {"apply_gamma": False}), lin)
