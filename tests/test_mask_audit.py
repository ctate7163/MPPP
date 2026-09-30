"""Mask audit (mppp.mask.audit). (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
import cv2
import numpy as np
import pytest
from helpers import _make_dataset


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


def _frame(h=400, w=600):
    t = np.zeros((h, w), bool)
    t[150:, :] = True                                       # terrain below a horizon
    return t


def test_empty_truth_and_empty_prediction_is_perfect():
    from mppp.mask.audit import frame_stats
    from mppp.mask.monitor import panel_stats
    t = np.zeros((100, 120), bool)
    assert panel_stats(t, np.zeros(t.shape, np.float32), 0.5)[0] == 1.0
    s = frame_stats(t, np.zeros(t.shape, np.float32), 0.5)
    assert s["iou"] == 1.0 and s["error_pct"] == 0.0 and s["terrain_truth_pct"] == 0.0
    p = np.zeros(t.shape, np.float32); p[0, 0] = 1.0           # one false pixel: IoU 0, but a tiny error
    s = frame_stats(t, p, 0.5)
    assert s["iou"] == 0.0 and s["error_pct"] == pytest.approx(100 / t.size)


def test_boundary_jitter_is_not_interior_error_but_a_missed_region_is():
    from mppp.mask.audit import frame_stats
    t = _frame()
    p = np.full(t.shape, 0.05, np.float32); p[153:, :] = 0.8   # horizon 3 px low: boundary jitter,
    p[150:153, :] = 0.3                                        # and the model is unsure there
    s = frame_stats(t, p, 0.5)
    assert s["error_pct"] > 0.5 and s["interior_error_pct"] == 0.0 and s["confident_error_pct"] == 0.0
    p = t.astype(np.float32) * 0.99
    p[300:360, 100:200] = 0.01                                 # model is sure: a rock the label calls terrain
    s = frame_stats(t, p, 0.5)
    assert s["band_px"] == 3 and s["interior_error_pct"] > 1.5 and s["confident_error_pct"] == pytest.approx(
        100 * 60 * 100 / t.size)


def test_rank_rows():
    from mppp.mask.audit import rank_rows, worst_table
    rows = [dict(name=f"f{k}", split=sp, status="ok", iou=iou, error_pct=e, interior_error_pct=i,
                 confident_error_pct=c, missed_pct=e, false_pct=0.0, terrain_truth_pct=tt)
            for k, (sp, iou, e, i, c, tt) in enumerate([
                ("train", 0.0, 0.2, 0.0, 0.0, 0.0),         # no terrain, a few false pixels
                ("train", 0.9, 5.0, 0.5, 0.1, 60.0),        # boundary-heavy
                ("val", 0.7, 3.0, 2.5, 2.0, 40.0),          # a wrong region
                ("train", 0.99, 0.1, 0.0, 0.0, 80.0)])]
    rows.append(dict(name="bad", split="train", status="size mismatch", iou=-1.0))
    assert [r["name"] for r in rank_rows(rows)] == ["bad", "f1", "f2", "f0", "f3"]       # error_pct default
    assert [r["name"] for r in rank_rows(rows, "iou")][:3] == ["bad", "f2", "f1"]         # no-terrain frame dropped
    assert [r["name"] for r in rank_rows(rows, "interior_error_pct", include_unscored=False)][:2] == ["f2", "f1"]
    v = rank_rows(rows, "confident_error_pct", split="train", include_unscored=False)
    assert [r["name"] for r in v] == ["f1", "f0", "f3"] and [r["rank"] for r in v] == [1, 2, 3]
    assert rows[1].get("rank") is None                                                  # copies, input unchanged
    assert "f2" in worst_table(rank_rows(rows, "interior_error_pct"), 3)
    old = [dict(name="x", split="train", status="ok", iou=0.5, error_pct=1.0)]          # a v0p10 CSV row
    assert rank_rows(old)[0]["name"] == "x"
    with pytest.raises(KeyError, match="rerun"):
        rank_rows(old, "interior_error_pct")
    with pytest.raises(ValueError):
        rank_rows(rows, "loss")
