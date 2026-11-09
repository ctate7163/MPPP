"""0.96: the online mask augmentations (mppp.mask.training.augment) - numpy only, no torch."""
import json
import sys

import numpy as np
import pytest

try:
    from mppp.mask.training import augment as A
except ModuleNotFoundError:                      # no torch here: mppp.mask's __init__ needs it, augment.py does not
    import importlib.util
    from pathlib import Path

    import mppp
    _p = Path(mppp.__file__).parent / "mask" / "training" / "augment.py"
    _spec = importlib.util.spec_from_file_location("mppp_mask_augment_standalone", _p)
    A = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = A                  # dataclasses look the module up in sys.modules
    _spec.loader.exec_module(A)

NAV = "NLF_0474_0708800000_000FDR_N0260000NCAM02474_01_095J01.png"
Z34 = "ZL0_0181_0683000000_000FDR_N0060000ZCAM08199_0340LMJ01.png"
Z110 = "ZR0_0483_0709900000_000FDR_N0260000ZCAM08508_1100LMJ01.png"


def _scene(h=240, w=320, seed=0):
    """A terrain image with a dark hardware block bottom-left and an invalid strip at the right."""
    rng = np.random.default_rng(seed)
    im = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    ms = np.full((h, w), 255, np.uint8)
    ms[h // 2:, : w // 3] = 0                        # hardware
    im[ms == 0] = (70, 70, 75)
    im[:, -20:] = 0                                  # invalid pixels
    ms[:, -20:] = 0
    return im, ms


def test_lmst_from_the_clock():
    assert A.lmst_hours_from_stem("ZL0_0000_0666893526_000FDR_N0000000ZCAM00000_0340LMJ01.png") == pytest.approx(0.0, abs=1e-6)
    half = int(A.SCLK_SOL0_MIDNIGHT + A.SOL_SECONDS / 2)
    assert A.lmst_hours_from_stem(f"NLF_0000_{half:010d}_000FDR_N0000000NCAM00000_01_095J01.png") == pytest.approx(12.0, abs=0.01)
    assert A.lmst_hours_from_stem("not_a_product.png") is None
    # a midday Navcam frame at 1.0, an early-morning Mastcam-Z frame at the upweight, an unknown name at 1.0
    morning = int(A.SCLK_SOL0_MIDNIGHT + 400 * A.SOL_SECONDS + 7.5 * 3600)
    noon = int(A.SCLK_SOL0_MIDNIGHT + 400 * A.SOL_SECONDS + 13.0 * 3600)
    w = A.lmst_weights([f"ZL0_0400_{morning:010d}_000FDR_N0000000ZCAM00001_0340LMJ01.png",
                        f"NLF_0400_{noon:010d}_000FDR_N0000000NCAM00001_01_095J01.png", "x.png"], upweight=4.5)
    assert w.tolist() == [4.5, 1.0, 1.0]


def test_native_angular_scale():
    assert A.family_of(NAV) == "N" and A.family_of(Z34) == "Z" and A.family_of("x.png") == ""
    assert A.native_mrad_per_px(NAV, 5120) == pytest.approx(1000 / 2956, rel=1e-6)
    assert A.native_mrad_per_px(NAV, 1280) == pytest.approx(4 * 1000 / 2956, rel=1e-6)      # a downsampled product
    assert A.native_mrad_per_px(Z34, 1648) == pytest.approx(1000 / 4720, rel=1e-6)
    assert A.native_mrad_per_px(Z110, 1648) == pytest.approx(1000 / 14758, rel=1e-6)
    assert A.native_mrad_per_px("SIF_0682_0727000000_000FDR_N0290000SRLC08040_0000LMJ01.png", 1600) is None
    # the gap the plan names: the hardware is ~5x / 16x larger in Z34 / Z110 than in Navcam at the native scale
    assert A.native_mrad_per_px(NAV, 5120) / A.native_mrad_per_px(Z34, 1648) == pytest.approx(4720 / 2956, rel=1e-6)


def test_angular_scale_resamples_image_and_mask_together():
    im, ms = _scene()
    rng = np.random.default_rng(1)
    native = 1.0                                                     # mrad / px of the synthetic frame
    out_i, out_m, info = A.angular_scale(im, ms, native, (160, 120), rng, target=2.0)   # half size
    assert out_i.shape == (120, 160, 3) and out_m.shape == (120, 160) and info["factor"] == pytest.approx(0.5)
    assert set(np.unique(out_m)) <= {0, 255}
    # hardware (mask 0, valid) covers the same share of the valid area as before
    valid0, valid1 = im.max(axis=2) > 0, out_i.max(axis=2) > 0
    share0 = ((ms == 0) & valid0).sum() / valid0.sum(); share1 = ((out_m == 0) & valid1).sum() / valid1.sum()
    assert share1 == pytest.approx(share0, abs=0.03)
    # zoom in: a crop fills the canvas
    out_i, out_m, info = A.angular_scale(im, ms, native, (160, 120), rng, target=0.5)
    assert info["factor"] == pytest.approx(2.0) and info["crop"] < 0.3 and (out_i.max(axis=2) > 0).mean() > 0.9
    # the clamps: an upscale beyond max_upscale or a frame below min_long_side is refused
    cfg = A.AngularScaleConfig(target_range=(0.05, 5.0), max_upscale=2.0, min_long_side=100)
    for _ in range(20):
        _, _, info = A.angular_scale(im, ms, native, (160, 120), rng, cfg)
        assert info["factor"] <= 2.0 + 1e-9 and max(im.shape[:2]) * info["factor"] >= 100 - 1e-6


def test_photometric_keeps_validity_and_is_seeded():
    im, ms = _scene()
    valid = im.max(axis=2) > 0
    out1, p1 = A.photometric(im, valid, np.random.default_rng(7))
    out2, _ = A.photometric(im, valid, np.random.default_rng(7))
    out3, _ = A.photometric(im, valid, np.random.default_rng(8))
    assert out1.dtype == np.uint8 and out1.shape == im.shape
    assert np.array_equal(out1, out2) and not np.array_equal(out1, out3)
    assert not out1[~valid].any() and out1[valid].max(axis=-1).min() >= 1       # invalid stays 0, valid never 0
    assert "gamma" in p1 and "gains" in p1
    grey, p = A.photometric(im, valid, np.random.default_rng(0), A.PhotometricConfig(grey_prob=1.0, noise_prob=0.0))
    assert p.get("grey") and np.array_equal(grey[..., 0], grey[..., 1]) and np.array_equal(grey[..., 1], grey[..., 2])
    # exposure only: a brighter copy
    cfg = A.PhotometricConfig(exposure_range=(2.0, 2.0), exposure_prob=1.0, gamma_range=(1.0, 1.0), gain_range=(1.0, 1.0),
                              black_level=0.0, noise_prob=0.0, shadow_prob=0.0, haze_prob=0.0, grey_prob=0.0)
    bright, _ = A.photometric(im, valid, np.random.default_rng(0), cfg)
    assert np.all(bright[valid].astype(int) >= im[valid].astype(int))


def test_defocus_blurs_hardware_only():
    im, ms = _scene()
    valid = im.max(axis=2) > 0
    out, p = A.defocus_hardware(im, ms, valid, np.random.default_rng(3), A.DefocusConfig(radius_px=(4.0, 4.0), rescale_prob=0.0))
    assert p["sigma"] == 4.0 and out.shape == im.shape
    far = np.zeros_like(ms, bool); far[:40, 200:280] = True                  # terrain far from the hardware
    assert np.array_equal(out[far], im[far])
    assert not out[~valid].any()
    # a frame without hardware is returned as it is
    same, p0 = A.defocus_hardware(im, np.where(valid, 255, 0).astype(np.uint8), valid, np.random.default_rng(0))
    assert p0 == {} and same is im


def test_copy_paste_moves_hardware_and_its_label():
    im, ms = _scene()
    cut = A.hardware_cutout(im, ms)
    assert cut is not None and cut[0].shape[:2] == cut[1].shape == (120, 106)
    assert A.hardware_cutout(im, np.where(im.max(axis=2) > 0, 255, 0).astype(np.uint8)) is None
    target = np.full((240, 320), 255, np.uint8); target[:, -20:] = 0
    tim = im.copy(); tim[ms == 0] = (150, 160, 170)                            # terrain everywhere
    out, om, info = A.paste_hardware(tim, target, cut, 0.5, np.random.default_rng(5), A.CopyPasteConfig(max_fill=0.5))
    y0, x0, ch, cw = info["box"]
    assert (ch, cw) == (60, 53) and info["scale"] == pytest.approx(0.5)
    assert (om == 0).sum() - (target == 0).sum() == pytest.approx(60 * 53, rel=0.02)   # the cutout's hardware landed
    inside = np.zeros_like(om, bool); inside[y0:y0 + ch, x0:x0 + cw] = True
    assert np.array_equal(om[~inside], target[~inside]) and np.array_equal(out[~inside & (target > 0)], tim[~inside & (target > 0)])
    # max_fill caps a huge paste
    _, om2, info2 = A.paste_hardware(tim, target, cut, 10.0, np.random.default_rng(5), A.CopyPasteConfig(max_fill=0.25))
    assert info2["box"][2] * info2["box"][3] <= 0.25 * om2.size + 1


def test_color_matrix_round_trip(tmp_path):
    cm = A.ColorMatrix()
    assert cm.is_identity()
    M = [[1.1, 0.05, 0.0], [0.0, 0.95, 0.02], [0.03, 0.0, 0.9]]
    (tmp_path / "cm.json").write_text(json.dumps({"M": M, "b": [0.01, 0.0, -0.01], "from": "N", "to": "Z", "note": "test"}))
    cm = A.ColorMatrix.load(tmp_path / "cm.json")
    assert not cm.is_identity() and cm.src == "N" and cm.dst == "Z"
    im, _ = _scene()
    valid = im.max(axis=2) > 0
    fwd = cm.apply(im, valid)
    back = cm.apply(fwd, valid, inverse=True)
    assert not fwd[~valid].any() and np.abs(back[valid].astype(int) - im[valid].astype(int)).max() <= 3
    # the transfer goes N -> Z forwards, Z -> N inverted, other families untouched; identity = off
    out, p = A.color_transfer(im, valid, "N", cm, np.random.default_rng(0), prob=1.0)
    assert p["color_transfer"] == "N->Z" and np.array_equal(out, fwd)
    out, p = A.color_transfer(im, valid, "Z", cm, np.random.default_rng(0), prob=1.0)
    assert p["color_transfer"] == "Z->N"
    out, p = A.color_transfer(im, valid, "F", cm, np.random.default_rng(0), prob=1.0)
    assert p == {} and out is im
    out, p = A.color_transfer(im, valid, "N", A.ColorMatrix(), np.random.default_rng(0), prob=1.0)
    assert p == {}


def test_recipes_follow_the_plan():
    assert set(A.RECIPES) == {"A", "B", "C"}
    assert A.RECIPES["A"] == {"photometric": True, "lmst_sampler": True}
    assert A.RECIPES["B"]["angular_scale"] and A.RECIPES["B"]["defocus"] and A.RECIPES["B"]["quad_split_above"] == 4000
    assert A.RECIPES["C"]["copy_paste"] and A.RECIPES["C"]["family_batches"] and A.RECIPES["C"]["color_matrix"] == "auto"
    assert A.AugmentConfig().summary()["photometric"] is False


def _letterbox(size, canvas):
    """The inference resize without torch (train.letterbox's layout): long side -> size, fit the canvas, pad bottom/right."""
    import cv2
    cw, ch = canvas

    def fn(im, ms):
        h, w = ms.shape[:2]
        r = min(size / max(h, w), cw / w, ch / h)
        nh, nw = min(ch, int(round(h * r))), min(cw, int(round(w * r)))
        oi = np.zeros((ch, cw, 3), np.uint8); oi[:nh, :nw] = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        om = np.zeros((ch, cw), np.uint8); om[:nh, :nw] = cv2.resize(ms, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return oi, om
    fn.factor = lambda h, w: min(size / max(h, w), cw / w, ch / h)
    return fn


def test_augment_pipeline_stages_follow_the_config():
    """0.96: Augment.from_config builds the stages the config turns on, in the plan's order; add_stage extends it
    without touching the dataset; the geometry step is the letterbox unless the angular scale applies."""
    im, ms = _scene()
    lb = _letterbox(160, (160, 120))
    off = A.Augment.from_config(A.AugmentConfig(), lb, (160, 120))
    assert off.names() == []
    oi, om = off(im, ms, NAV, 1.0, np.random.default_rng(0))
    ri, rm = lb(im, ms)
    assert np.array_equal(oi, ri) and np.array_equal(om, rm)             # nothing on: exactly the letterbox
    cfg = A.AugmentConfig(photometric=True, defocus=True, copy_paste=True, angular_scale=True)
    cut = A.hardware_cutout(im, ms)
    donor = lambda rng: (cut, 1.0)                                          # noqa: E731
    aug = A.Augment.from_config(cfg, lb, (160, 120), donor=donor)
    assert aug.names() == ["copy_paste", "defocus", "photometric"]        # colour transfer off: identity matrix
    calls = []
    aug.add_stage("my_new_one", lambda s, rng: (calls.append(s.canvas_mrad), s)[1], before="photometric")
    assert aug.names() == ["copy_paste", "defocus", "my_new_one", "photometric"]
    for seed in range(4):
        oi, om = aug(im, ms, NAV, 1.0, np.random.default_rng(seed))
        assert oi.shape == (120, 160, 3) and om.shape == (120, 160) and set(np.unique(om)) <= {0, 255}
    assert len(calls) == 4 and all(c is not None for c in calls)         # the sample carries its canvas scale
    # an unknown camera keeps the letterbox and no canvas scale (copy-paste then skips itself)
    s = aug.geometry(im, ms, "SIF_0682_0727000000_000FDR_N0290000SRLC08040_0000LMJ01.png", None, np.random.default_rng(0))
    assert s.canvas_mrad is None and s.info["geometry"] == {"letterbox": True}
    s2 = aug.stage_copy_paste(s, np.random.default_rng(0))
    assert "copy_paste" not in s2.info
    # the letterbox's canvas scale: the frame drawn 160 px wide from 320 native -> 2x the native mrad
    plain = A.Augment.from_config(A.AugmentConfig(), lb, (160, 120)).geometry(im, ms, NAV, 1.0, np.random.default_rng(0))
    assert plain.canvas_mrad == pytest.approx(2.0, rel=0.02)


def test_frozen_test_list_round_trip(tmp_path):
    """0.96: read_test_list / write_test_list and grouped_split(test_list=...) - the listed frames are the held-out
    side whatever the seed; names may be given as stems or with another extension."""
    import importlib.util
    import mppp
    _p = Path(mppp.__file__).parent / "mask" / "training" / "train.py"
    try:
        from mppp.mask.training import train as T
    except ModuleNotFoundError:
        pytest.skip("torch is not installed: train.py is tested on the GPU machine")
    names = [f"ZL0_{k:04d}_0700000000_000FDR_N0000000ZCAM0{k:04d}_0340LMJ01.png" for k in range(10)]
    items = [T.Item(image=f"/d/images/{n}", mask=f"/d/masks/{n}") for n in names]
    lst = T.write_test_list(tmp_path / "frozen.txt", [names[1], names[4], Path(names[7]).stem, names[4]], note="t")
    got = T.read_test_list(lst)
    assert got == sorted({names[1], names[4], Path(names[7]).stem})
    for seed in (1, 2, 3):
        tr, va = T.grouped_split(items, 0.1, seed, got)
        assert sorted(Path(it.mask).name for it in va) == sorted([names[1], names[4], names[7]])
        assert len(tr) == 7
    with pytest.raises(ValueError):
        T.grouped_split(items, 0.1, 1, ["not_there.png"])
