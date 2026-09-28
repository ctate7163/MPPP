"""v0p22.4: mask model v3 default, sky-pointing rule, tolerant only_existing, ADD_NEARBY_WAYPOINTS."""
import json
import sys
from pathlib import Path

import pytest

from conftest import NLF, ZL0, needs_data, synthetic_waypoints

ROOT = Path(__file__).resolve().parents[1]


def test_default_mask_model_is_v3():
    from mppp.mask.hub import load_registry, default_model_name, local_candidates
    import mppp
    reg = load_registry()
    assert default_model_name() == "mppp_mask_v3" and mppp.load_config({})["masking"]["checkpoint"] == "mppp_mask_v3"
    e = reg["models"]["mppp_mask_v3"]
    assert e["source_checkpoint"] == "convnext_tiny_s4_seg_20260925b.pt" and e["file"].endswith("_v3.safetensors")
    assert len(e["sha256"]) == 64 and e["bytes"] > 10 ** 8 and e["val_iou"] > 0.979
    assert isinstance(local_candidates("mppp_mask_v3"), list)     # the .pt in checkpoints/ stands in until the release is published


@needs_data
def test_sky_pointing_navcam_is_skipped(tmp_path):
    import mppp
    from mppp.image import MPPPImage, SkyImage
    from mppp.process import process_images
    wp = synthetic_waypoints()
    on = mppp.load_config({"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": 45.0}})
    off = mppp.load_config({"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": None}})
    with pytest.raises(SkyImage):
        MPPPImage(NLF, on, wp)                                    # the Sun-pointing tile looks up at +78 deg
    assert MPPPImage(NLF, off, wp).pose.boresight_az_el_deg[1] > 45
    MPPPImage(ZL0, on, wp)                                        # Mastcam-Z is not filtered (looks down anyway)
    high = mppp.load_config({"masking": {"infer_mask": False}, "selection": {"max_boresight_elevation_deg": 80.0}})
    MPPPImage(NLF, high, wp)
    man = process_images([NLF, ZL0], tmp_path / "out", on, wp, progress=False)
    assert man["n_processed"] == 1 and man["failed"] == [] and len(man["skipped"]) == 1
    assert "sky-pointing" in man["skipped"][0]["reason"] and man["skipped"][0]["file"].endswith(NLF.name)


@needs_data
def test_only_existing_is_ignored_on_a_first_run(tmp_path):
    import mppp
    from mppp.process import process_images, filter_to_existing
    wp = synthetic_waypoints()
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"]}})
    kept, rep = filter_to_existing([ZL0], tmp_path / "nothing", "PNG8")
    assert kept == [ZL0] and rep["skipped"] and rep["n_kept"] == 1
    man = process_images([ZL0], tmp_path / "out", cfg, wp, only_existing="PNG8", progress=False)
    assert man["n_processed"] == 1 and man["only_existing"]["skipped"]
    # second run: the folder exists and the image is in it -> the normal filter applies
    man2 = process_images([ZL0], tmp_path / "out", cfg, wp, only_existing="PNG8", progress=False)
    assert man2["only_existing"]["n_kept"] == 1 and "skipped" not in man2["only_existing"]
    # a selection none of which is in the folder -> processed, not an error
    (tmp_path / "out2" / "images_png8").mkdir(parents=True)
    kept3, rep3 = filter_to_existing([ZL0], tmp_path / "out2", "PNG8")
    assert kept3 == [ZL0] and rep3["skipped"]


def test_notebook_03_new_defaults():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "notebooks" / "03_colmap_alignment.ipynb"), as_version=4)
    src = next(c.source for c in nb.cells if "parameters" in c.metadata.get("tags", []))
    ns = {}
    exec(compile("from pathlib import Path\n" + src, "settings", "exec"), ns)
    assert ns["ADD_NEARBY_WAYPOINTS"] == 5 and "NEARBY_M" not in ns
    assert ns["SITES"]["threeforks_large"] == (652, 693) and ns["SITES"]["rockytop"] == (461, 530)
    assert {"taylorfjellet", "rockytop", "belva_crater", "butler_landing", "pearce_canyon", "olifants"} <= set(ns["SITES"])
    assert all(" " not in k for k in ns["SITES"])
    assert ns["KEEP_ONLY_REMAINING"] is True and ns["STORE_MASK_IN_ALPHA"] is True and ns["ZCAM_RIG"] is True
    assert ns["MAX_NUM_FEATURES"] == 12000 and ns["SCHEDULE"][1] == (12.0, 4.0, 4.0)
    assert ns["NO_MASK_INFERENCE_AT"] == ["S032D1184"] and ns["SKY_ELEVATION_DEG"] == 45.0
    full = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "radius_m=ADD_NEARBY_WAYPOINTS" in full and "max_boresight_elevation_deg" in full
