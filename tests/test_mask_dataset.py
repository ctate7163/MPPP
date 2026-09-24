"""Training-set builder: masks from a Metashape project, images_variable rebuild."""
import io
import json
import zipfile

import cv2
import numpy as np
import pytest


def _psx(root, cams, masks):
    """Minimal Metashape project: <root>/p.psx + p.files/0/0/{frame.zip, masks/masks.zip}."""
    (root / "p.psx").write_text('<document version="1.2.0" path="{projectname}.files/project.zip"/>')
    fd = root / "p.files" / "0" / "0"
    (fd / "masks").mkdir(parents=True)
    xml = "".join(f'<camera camera_id="{i}"><photo path="{p}"/></camera>' for i, p in cams.items())
    with zipfile.ZipFile(fd / "frame.zip", "w") as z:
        z.writestr("doc.xml", f'<frame version="1.2.0"><cameras>{xml}</cameras></frame>')
    with zipfile.ZipFile(fd / "masks" / "masks.zip", "w") as z:
        z.writestr("doc.xml", "<masks>" + "".join(f'<mask camera_id="{i}" path="c{i}.png"/>' for i in masks) + "</masks>")
        for i, m in masks.items():
            z.writestr(f"c{i}.png", cv2.imencode(".png", m)[1].tobytes())
    return root / "p.psx"


@pytest.fixture
def tset(tmp_path):
    r = tmp_path / "set"
    (r / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    masks = {}
    for i, name in enumerate(["A.png", "B.png", "C.png"]):
        img = np.dstack([rng.integers(10, 250, (40, 60, 3), dtype=np.uint8), np.full((40, 60), 255, np.uint8)])
        img[:5, :, :3] = 0                                               # invalid rows: RGB exactly 0
        img[30:, :, 3] = 0                                               # alpha = a mask (as in v7), not validity
        cv2.imwrite(str(r / "images" / name), img)
        m = np.zeros((40, 60), np.uint8)
        m[20:] = 255
        masks[str(i)] = m
    cams = {"0": "../../../images/A.png", "1": "../../../images/B.png", "2": "../../../images/C.png",
            "3": "../../../elsewhere.png"}
    masks["3"] = np.zeros((40, 60), np.uint8)
    del masks["2"]                                                       # C has no mask in the project
    return r, _psx(r, cams, masks)


def test_export_masks_from_psx(tset):
    from mppp.mask.dataset import export_masks_from_psx
    r, psx = tset
    (r / "masks").mkdir()
    (r / "masks" / "old.png").write_bytes(b"x")
    rep = export_masks_from_psx(psx, r / "images", r / "masks")
    assert rep["written"] == 2 and len(rep["cameras_outside_images_dir"]) == 1
    assert rep["archived_previous_masks"]
    assert not (r / "masks" / "old.png").exists()                      # moved aside, not deleted:
    assert list(r.glob("masks_prev_*/old.png"))
    m = cv2.imread(str(r / "masks" / "A.png"), 0)
    assert m.shape == (40, 60) and set(np.unique(m)) == {0, 255} and m[25].all() and not m[5].any()


def test_build_training_set_reuses_originals_and_synthesizes_the_rest(tset, tmp_path):
    from mppp.mask.dataset import build_training_set
    r, psx = tset
    old = tmp_path / "v6_images_variable"
    old.mkdir()
    orig = np.dstack([np.full((40, 60, 3), 77, np.uint8), np.zeros((40, 60), np.uint8)])
    cv2.imwrite(str(old / "A.png"), orig)
    rep = build_training_set(r, psx=psx, reuse_variable_from=[old])
    iv = rep["images_variable"]
    assert iv["written"] == 2 and iv["reused_original"] == 1 and iv["synthesized"] == 1
    assert iv["images_without_mask"] == ["C.png"]
    a = cv2.imread(str(r / "images_variable" / "A.png"), -1)
    assert (a[..., :3] == 77).all()                                     # original RGB kept bit-exact
    assert np.array_equal(a[..., 3] > 0, cv2.imread(str(r / "masks" / "A.png"), 0) > 0)   # alpha = mask
    b = cv2.imread(str(r / "images_variable" / "B.png"), -1)
    src = cv2.imread(str(r / "images" / "B.png"), -1)
    assert not b[:5, :, :3].any() and b[5:, :, :3].min() >= 1          # invalid stays 0, valid never 0
    assert b[30:, :, :3].min() >= 1                                     # alpha-masked rows are NOT blacked out
    assert not np.array_equal(b[..., :3], src[..., :3])                 # a different tone curve
    assert json.loads((r / "build_report.json").read_text())["images_variable"]["synthesized"] == 1
    rows = (r / "images_variable_manifest.csv").read_text().splitlines()
    assert rows[0].startswith("name,source,lo") and len(rows) == 3


def test_make_variable_is_deterministic_and_in_measured_range():
    from mppp.mask.dataset import VariableRecipe, make_variable
    rgb = np.random.default_rng(1).integers(0, 256, (30, 30, 3), dtype=np.uint8)
    valid = np.ones((30, 30), bool)
    a, pa = make_variable(rgb, valid, "X.png")
    b, pb = make_variable(rgb, valid, "X.png")
    c, pc = make_variable(rgb, valid, "Y.png")
    assert np.array_equal(a, b) and pa == pb and pa != pc
    rc = VariableRecipe()
    assert rc.lo[0] <= pa["lo"] <= rc.lo[1] and rc.gamma[0] <= pa["gamma"] <= rc.gamma[1]


# ------------------------------------------------------ v0p6: images/ from PDS
from pathlib import Path  # noqa: E402

from conftest import NLF, ZL0, needs_data  # noqa: E402


@needs_data
def test_regenerate_images_from_pds(tmp_path, waypoints):
    import shutil
    from mppp import MPPPImage
    from mppp.mask.dataset import regen_config, regenerate_images_from_pds
    root, pds = tmp_path / "set", tmp_path / "pds" / "00709" / "ids"
    (root / "masks").mkdir(parents=True)
    pds.mkdir(parents=True)
    shutil.copy(ZL0, pds / ZL0.name)
    ref = MPPPImage(ZL0, regen_config(), waypoints)
    h, w = ref.image_int8.shape[:2]
    m = np.zeros((h, w), np.uint8); m[h // 3:] = 255
    cv2.imwrite(str(root / "masks" / f"{ZL0.stem}.png"), m)
    cv2.imwrite(str(root / "masks" / f"{NLF.stem}.png"), np.zeros((10, 10), np.uint8))   # no PDS product
    wrong = ZL0.stem[:-2] + "07"                                                             # other version
    (root / "images").mkdir(); (root / "images" / "foreign.png").write_bytes(b"x")          # not ours -> archived
    r = regenerate_images_from_pds(root, tmp_path / "pds", waypoints=waypoints)
    assert r["counts"] == {"ok": 1, "missing": 1, "already_done": 0}
    assert r["archived_previous_images"] and (Path(r["archived_previous_images"]) / "foreign.png").is_file()
    out = cv2.imread(str(root / "images" / f"{ZL0.stem}.png"), cv2.IMREAD_UNCHANGED)
    assert out.shape == (h, w, 4)
    assert np.array_equal(out[..., 2::-1], ref.image_int8)                                  # BGR on disk
    assert np.array_equal(out[..., 3], m)                                                    # alpha = mask
    # resumable: a second run does nothing new
    r2 = regenerate_images_from_pds(root, tmp_path / "pds", waypoints=waypoints)
    assert r2["counts"]["already_done"] == 1 and "ok" not in r2["counts"]
    # version substitution and size check
    shutil.copy(root / "masks" / f"{ZL0.stem}.png", root / "masks" / f"{wrong}.png")
    cv2.imwrite(str(root / "masks" / f"{ZL0.stem[:5]}8{ZL0.stem[6:]}.png"), m)            # absent product
    r3 = regenerate_images_from_pds(root, tmp_path / "pds", waypoints=waypoints)
    assert r3["counts"].get("version_substituted") == 1


def test_find_pds_versions():
    from mppp.mask.dataset import find_pds
    idx = {"NLF_1408_0791940746_034RAD_N0680000NCAM13408_0A0195J02": Path("a"),
           "NLF_1408_0791940746_034RAD_N0680000NCAM13408_0A0195J03": Path("b")}
    assert find_pds("NLF_1408_0791940746_034RAD_N0680000NCAM13408_0A0195J03", idx) == (Path("b"), "ok")
    assert find_pds("nlf_1408_0791940746_034RAD_N0680000NCAM13408_0A0195J01", idx) == (Path("b"), "version_substituted:03")
    assert find_pds("NLF_1408_0791940746_034RAD_N0680000NCAM13408_0A1195J01", idx) == (None, "missing")


@needs_data
def test_regenerate_with_spawned_workers(tmp_path):
    """Windows runs workers>1 by spawn: everything handed to the pool must pickle."""
    import shutil, subprocess, sys, textwrap
    from conftest import ROOT
    root, pds = tmp_path / "set", tmp_path / "pds"
    (root / "masks").mkdir(parents=True); pds.mkdir()
    shutil.copy(ZL0, pds / ZL0.name)
    cv2.imwrite(str(root / "masks" / f"{ZL0.stem}.png"), np.full((1200, 1648), 255, np.uint8))
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(f"""
        import multiprocessing as mp, sys
        sys.path.insert(0, {str(ROOT / 'src')!r})
        if __name__ == "__main__":
            mp.set_start_method("spawn", force=True)
            from mppp.mask.dataset import regenerate_images_from_pds
            r = regenerate_images_from_pds({str(root)!r}, {str(pds)!r}, workers=2)
            print("COUNTS", r["counts"])
    """))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "'ok': 1" in r.stdout and (root / "images" / f"{ZL0.stem}.png").is_file()


def test_variable_refuses_empty_images_and_leaves_existing_untouched(tmp_path):
    """23 Sep: run before images/ was regenerated, v0p7 archived images_variable/ and wrote nothing."""
    from mppp.mask.dataset import build_variable_images
    r = tmp_path / "set"
    for d in ("images", "masks", "images_variable"):
        (r / d).mkdir(parents=True)
    cv2.imwrite(str(r / "masks" / "A.png"), np.zeros((8, 8), np.uint8))
    cv2.imwrite(str(r / "images_variable" / "A.png"), np.full((8, 8, 3), 9, np.uint8))
    (r / "images" / "A.tmp.png").write_bytes(b"half-written")          # an interrupted write does not count
    with pytest.raises(FileNotFoundError, match="notebook 02, step 1"):
        build_variable_images(r)
    assert (r / "images_variable" / "A.png").is_file() and not list(r.glob("images_variable_prev_*"))
    cv2.imwrite(str(r / "images" / "Z.png"), np.full((8, 8, 3), 9, np.uint8))    # an image, but no mask for it
    with pytest.raises(FileNotFoundError, match="none of the 1 images"):
        build_variable_images(r)
    assert not list(r.glob("images_variable_prev_*"))
