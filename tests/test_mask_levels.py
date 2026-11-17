"""0.99.9 (Christian, 16 Nov 2026, for the next mask run): no dilation, small islands removed by connected components,
the label-edge band out of the loss (or a Lovasz term), and the probability map on disk so that each step takes its own
threshold (rpp.mask_levels)."""
import json
import os

import cv2
import numpy as np
import pytest

from conftest import NLF, config_selection_off, needs_data


def test_small_islands_are_removed_both_ways_and_there_is_no_dilation():
    from rpp import mask_levels as L
    inc = np.ones((200, 300), bool)
    inc[10:13, 10:13] = False                     # a 9 px rover speck in the terrain
    inc[100:180, 100:200] = False                 # the rover
    inc[120:123, 120:123] = True                  # a 9 px terrain speck in the rover
    out = L.remove_small_islands(inc, 20)
    assert out[11, 11] and not out[121, 121] and not out[150, 150] and out.sum() == inc.size - 80 * 100
    assert np.array_equal(L.remove_small_islands(inc, 0), inc)
    p = np.where(inc, 0.95, 0.05).astype(np.float32)
    m = L.probability_to_mask(p, 0.5, 0, min_island_frac=20 / inc.size)
    assert set(np.unique(m)) == {0, 255} and np.array_equal(m > 0, out)
    # no dilation unless asked (rpp_mask_v3's card asks for 3: its masks stay as they were)
    assert (L.probability_to_mask(p, 0.5) > 0).sum() == inc.sum()
    assert (L.probability_to_mask(p, 0.5, 3) > 0).sum() > inc.sum()


def test_step_thresholds_from_the_probability():
    from rpp import mask_levels as L
    p = np.zeros((50, 100), np.float32)
    p[:, :60] = 0.97                               # terrain
    p[:, 60:64] = np.linspace(0.85, 0.55, 4)       # the soft edge
    valid = np.ones_like(p, np.uint8)
    valid[:, :5] = 0                               # no data
    p8 = L.prob_to_uint8(p, valid)
    assert p8[:, :5].max() == 0 and p8[0, 10] == round(0.97 * 255)
    sfm, tight = (L.mask_from_probability(p8, t) > 0 for t in (0.9, 0.5))
    assert sfm.sum() == 50 * 55 and tight.sum() == 50 * 59 and not tight[:, :5].any()     # the margin and the edge
    assert L.step_thresholds({"step_thresholds": {"sfm": 0.9, "tight": 0.5}}) == {"sfm": 0.9, "tight": 0.5}
    assert L.step_thresholds({"name": "rpp_mask_v3"}) is None                              # v3: the binary mask
    assert L.step_thresholds({"step_thresholds": {"sfm": 0.9}}, {"sfm": 0.8, "tight": 0.4}) == {"sfm": 0.8, "tight": 0.4}
    assert L.step_record({"step_thresholds": {"sfm": 0.9, "tight": 0.5}, "min_island_frac": 1e-4}) == \
        {"thresholds": {"sfm": 0.9, "tight": 0.5}, "min_island_frac": 1e-4}


def test_project_masks_from_the_probability(tmp_path):
    from rpp import mask_levels as L
    from rpp.sfm.project import place_masks
    proc = tmp_path / "processed"
    (proc / "masks").mkdir(parents=True)
    (proc / "masks_prob").mkdir()
    p = np.zeros((40, 80), np.float32)
    p[:, :40] = 0.95
    p[:, 40:44] = 0.7
    cv2.imwrite(str(proc / "masks" / "A.png"), ((p > 0.4) * 255).astype(np.uint8))
    cv2.imwrite(str(proc / "masks_prob" / "A.png"), L.prob_to_uint8(p))
    rec = {"outputs": {"mask": "masks/A.png", "mask_prob": "masks_prob/A.png"},
           "mask": {"steps": {"thresholds": {"sfm": 0.9, "tight": 0.5}, "min_island_frac": 0.0}}}
    root = tmp_path / "colmap"
    (root / "masks").mkdir(parents=True)
    os.link(proc / "masks" / "A.png", root / "masks" / "A.png.png")         # a project made before: a hard link
    before = (proc / "masks" / "A.png").read_bytes()
    place_masks(proc, rec, root, "A.png")
    sfm = cv2.imread(str(root / "masks" / "A.png.png"), cv2.IMREAD_GRAYSCALE)
    tight = cv2.imread(str(root / L.TIGHT_DIR / "A.png.png"), cv2.IMREAD_GRAYSCALE)
    assert (sfm > 0).sum() == 40 * 40 and (tight > 0).sum() == 40 * 44
    assert (proc / "masks" / "A.png").read_bytes() == before                  # the processed mask is not touched
    assert (root / L.PROB_DIR / "A.png.png").is_file()
    t0 = (root / "masks" / "A.png.png").stat().st_mtime_ns
    rep = place_masks(proc, rec, root, "A.png")                              # again: nothing rewritten
    assert rep["kept"]["sfm"] == 1 and (root / "masks" / "A.png.png").stat().st_mtime_ns == t0
    # a mask model without step thresholds (rpp_mask_v3): the binary mask, as before
    root2 = tmp_path / "colmap2"
    place_masks(proc, {"outputs": rec["outputs"], "mask": {"steps": None}}, root2, "A.png")
    assert (root2 / "masks" / "A.png.png").read_bytes() == before and not (root2 / L.TIGHT_DIR).exists()
    # the steps' readers: the panorama and dense fusion take the tight mask
    from rpp.ibr.rover_masks import mask_file
    assert mask_file(root, "A.png") == root / L.TIGHT_DIR / "A.png.png"
    assert mask_file(root2, "A.png") == root2 / "masks" / "A.png.png"
    # another threshold later, without the network
    rep = L.write_step_masks(root, ["A.png"], {"tight": 0.8}, which=("tight",))
    assert rep["written"]["tight"] == 1
    assert (cv2.imread(str(root / L.TIGHT_DIR / "A.png.png"), cv2.IMREAD_GRAYSCALE) > 0).sum() == 40 * 40
    assert json.loads((root / L.PROB_DIR / "step_masks.json").read_text())["thresholds"] == {"tight": 0.8}


@needs_data
def test_processing_writes_the_probability_beside_the_mask(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import rpp
    import rpp.mask as M
    card = {"name": "next", "release_name": "next", "threshold": 0.5, "dilate_kernel": 0, "min_island_frac": 1e-4,
            "step_thresholds": {"sfm": 0.9, "tight": 0.5}}

    def fake_infer(img, checkpoint, device, threshold, dilate_kernel, min_island_frac=None, **kw):
        h, w = img.shape[:2]
        prob = np.full((h, w), 0.95, np.float32)
        prob[:, : w // 3] = 0.1                    # "the rover" on the left third
        from rpp.mask_levels import probability_to_mask
        return probability_to_mask(prob, 0.5, 0, 1e-4), prob, card
    monkeypatch.setattr(M, "infer_mask", fake_infer)
    monkeypatch.setattr("rpp.process.check_mask_checkpoint", lambda cfg: None)
    cfg = config_selection_off({"export": {"formats": ["PNG8"], "write_mask_files": True}})
    man = rpp.process_images([NLF], tmp_path, cfg, rpp.load_waypoints(), progress=False, workers=1)
    m = man["images"][0]
    assert m["mask"]["probability"] and m["mask"]["steps"] == {"thresholds": {"sfm": 0.9, "tight": 0.5}, "min_island_frac": 1e-4}
    assert m["mask"]["model"]["dilate_kernel"] == 0 and "mask_prob" in m["outputs"]
    mk = cv2.imread(str(tmp_path / m["outputs"]["mask"]), cv2.IMREAD_GRAYSCALE)
    p8 = cv2.imread(str(tmp_path / m["outputs"]["mask_prob"]), cv2.IMREAD_GRAYSCALE)
    assert p8.shape == mk.shape                                               # the same padded frame
    assert np.array_equal(p8 > 127, mk > 0)                                   # the binary mask is the probability at 0.5
    assert p8[mk == 0].max() < 128 and p8.max() == round(0.95 * 255)


def test_mask_levels_command(tmp_path, capsys):
    from rpp import mask_levels as L
    from rpp.cli import main
    root = tmp_path / "colmap"
    (root / L.PROB_DIR).mkdir(parents=True)
    p = np.zeros((20, 40), np.float32)
    p[:, :20] = 0.95
    p[:, 20:24] = 0.7
    for n in ("A.png", "B.png"):
        cv2.imwrite(str(root / L.PROB_DIR / (n + ".png")), L.prob_to_uint8(p))
    assert main(["mask", "levels", str(root), "--sfm", "0.9", "--tight", "0.5"]) == 0
    assert (cv2.imread(str(root / "masks" / "A.png.png"), 0) > 0).sum() == 20 * 20
    assert (cv2.imread(str(root / L.TIGHT_DIR / "B.png.png"), 0) > 0).sum() == 20 * 24
    assert main(["mask", "levels", str(root), "--tight", "0.8"]) == 0     # one step: the other one is kept
    assert json.loads((root / L.PROB_DIR / "step_masks.json").read_text())["thresholds"] == {"sfm": 0.9, "tight": 0.8}
    assert (cv2.imread(str(root / L.TIGHT_DIR / "B.png.png"), 0) > 0).sum() == 20 * 20
    out = capsys.readouterr().out
    assert "takes effect when the block's features are extracted again" in out
    assert main(["mask", "levels", str(tmp_path)]) == 2                  # no masks_prob/
