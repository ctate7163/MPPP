"""v0p12: audit statistics that are meaningful for frames without terrain; re-ranking."""
import numpy as np
import pytest


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
