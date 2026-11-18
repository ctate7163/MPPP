"""0.99.21: the mask validation set (rpp_studies.mask_training.validation_set) - selection and the guess; no torch."""
import json

import numpy as np
import pytest

from rpp_studies.mask_training import validation_set as V

NAV = "NLF_0686_0727849538_450RAD_N0321184NCAM03686_0A0185J01"
Z34 = "ZL0_0965_0752605363_081RAD_N0470000ZCAM05086_0340LMA01"
MSL = "NLB_422633248RAD_F0060082NCAM00207M1"


def _nav(sol, sclk, eye="L", seq=3000, site=10, drive=0, lmst=12.0, cam="N", look=0.0):
    stem = f"{cam}{eye}F_{sol:04d}_{sclk:010d}_000RAD_N{site:03d}{drive:04d}NCAM{seq:05d}_0A0195J01"
    return dict(V.frame_key(stem), lmst_h=lmst, sun_el_deg=40.0, look_el_deg=look, block="b", source=None)


def test_frame_key_names_the_camera_pair_and_scene():
    k = V.frame_key(NAV + ".png")
    assert (k["mission"], k["stratum"], k["eye"], k["sol"]) == ("M2020", "N", "L", 686)
    assert V.frame_key("NRF" + NAV[3:])["pair"] == k["pair"]              # the other eye: the same pair
    assert k["group"] == "M20:32_1184:686:NCAM03686" and k["station"] == "M20:32_1184"
    assert V.frame_key(NAV[:-2] + "02")["base"] == k["base"]              # another version: the same product
    assert V.frame_key(Z34)["stratum"] == "Z034"
    m = V.frame_key(MSL)
    assert (m["mission"], m["stratum"], m["eye"], m["sol"]) == ("MSL", "MSL_N", "L", None)
    assert V.frame_key("NRB" + MSL[3:])["pair"] == m["pair"] and V.frame_key("NLA" + MSL[3:])["pair"] != m["pair"]
    assert V.frame_key("SIF_0100_0675000000_000RAD_N0040000SRLC00000_0000LMJ01") is None
    assert V.frame_key("notes") is None


def test_lmst_and_look_elevation():
    assert abs(V.lmst_hours("Sol-01608M13:37:59.331") - 13.6331) < 1e-3
    assert V.lmst_hours(9.5) == 9.5 and V.lmst_hours(None) is None and V.lmst_hours("no time") is None
    R = [[1, 0, 0], [0, 0, -1], [0, 0.5, -np.sqrt(0.75)]]
    assert abs(V.look_elevation_deg(R) + 60.0) < 1e-6 and V.look_elevation_deg(None) is None


def test_independence_levels():
    train = V.training_keys([NAV + ".png", MSL])
    assert V.independence(V.frame_key(NAV), train) == 0
    assert V.independence(V.frame_key(NAV[:-2] + "03"), train) == 0                     # a later version of it
    assert V.independence(V.frame_key("NRF" + NAV[3:]), train) == 1                     # its stereo partner
    same_seq = NAV.replace("0727849538", "0727849700")
    assert V.independence(V.frame_key(same_seq), train) == 1                            # the same sequence there
    other = NAV.replace("NCAM03686", "NCAM00500").replace("0727849538", "0727859999")
    assert V.independence(V.frame_key(other), train) == 2
    assert V.independence(V.frame_key("NRB" + MSL[3:]), train) == 1
    assert V.independence(V.frame_key(MSL.replace("0060082", "0070000").replace("422633248", "423000000")), train) == 2


def test_scaled_quotas():
    assert V.scaled_quotas(100) == V.QUOTAS and sum(V.QUOTAS.values()) == 100
    for n in (10, 37, 250):
        q = V.scaled_quotas(n)
        assert sum(q.values()) == n and all(v >= 0 for v in q.values())


def test_select_spreads_over_sol_time_and_pointing():
    rng = np.random.default_rng(1)
    cands = []
    for i in range(300):                                   # most candidates crowd sols 1400-1500 around noon
        crowd = i < 240
        sol = int(rng.integers(1400, 1500)) if crowd else int(rng.integers(10, 2000))
        lm = float(rng.normal(12.5, 0.4)) if crowd else float(rng.uniform(7, 18))
        cands.append(_nav(sol, 700000000 + i * 500, "LR"[i % 2], seq=3000 + i, site=i // 3, lmst=lm,
                          look=float(rng.uniform(-60, 10))))
    frames, rep = V.select(cands, V.training_keys([]), n=20, quotas={"N": 1})
    assert len(frames) == 20 and rep["selected"] == 20 and len({f["stem"] for f in frames}) == 20
    sols = np.array([f["sol"] for f in frames]); lm = np.array([f["lmst_h"] for f in frames])
    assert sols.min() < 400 and sols.max() > 1700 and (sols < 1400).sum() >= 8          # not the crowd's share (80 %)
    assert lm.min() < 9.5 and lm.max() > 15.5
    assert len({f["group"] for f in frames}) == 20                                      # one frame per sequence
    eyes = [f["eye"] for f in frames]
    assert 7 <= eyes.count("L") <= 13
    again, _ = V.select(cands, V.training_keys([]), n=20, quotas={"N": 1})
    assert [f["stem"] for f in again] == [f["stem"] for f in frames]                    # the same every time


def test_select_leaves_training_out_and_marks_near_frames():
    indep = [_nav(100 + 10 * i, 600000000 + i * 900, seq=100 + i, site=i) for i in range(4)]
    seen = _nav(900, 650000000, seq=777, site=50)
    near = [_nav(900, 650000000 + 60 * i, seq=777, site=50) for i in range(1, 4)]       # its sequence, other frames
    train = V.training_keys([seen["stem"]])
    frames, rep = V.select(indep + [seen] + near, train, n=6, quotas={"N": 1})
    stems = [f["stem"] for f in frames]
    assert seen["stem"] not in stems and rep["in_training"] == 1
    assert set(f["stem"] for f in indep) <= set(stems)                                  # the independent ones first
    assert sum(f["near_training"] for f in frames) == 2 and len(frames) == 6
    assert rep["by_camera"]["N"] == {"quota": 6, "selected": 6, "near_training": 2, "independent_candidates": 4,
                                     "near_candidates": 3}


def test_select_gives_a_short_camera_s_quota_to_the_others():
    nav = [_nav(100 + i, 600000000 + i * 900, seq=100 + i, site=i) for i in range(30)]
    haz = [_nav(200 + i, 610000000 + i * 900, seq=500 + i, site=i, cam="F") for i in range(2)]
    frames, rep = V.select(nav + haz, V.training_keys([]), n=10, quotas={"N": 1, "F": 1})
    assert len(frames) == 10
    assert rep["by_camera"]["F"]["selected"] == 2 and rep["by_camera"]["N"]["selected"] == 8
    assert len({f["stem"] for f in frames}) == 10


def test_block_candidates_reads_both_missions_once(tmp_path):
    b = tmp_path / "mars2020_sol_0686_x" / "colmap"
    b.mkdir(parents=True)
    img = {"stem": NAV, "lmst": "Sol-00686M14:46:10.000", "solar_elevation_deg": 41.2,
           "prior_R_w2c": [[1, 0, 0], [0, 0, -1], [0, 0.5, -0.8660254]]}
    (b / "project.json").write_text(json.dumps({"images": [img, {"stem": Z34}, {"stem": "odd"}]}))
    b2 = tmp_path / "mars2020_sol_0686_y" / "colmap"
    b2.mkdir(parents=True)
    (b2 / "project.json").write_text(json.dumps({"images": [img]}))                     # the same frame again
    m = tmp_path / "msl_sol_0124_y"
    (m / "images").mkdir(parents=True)
    (m / "manifest.json").write_text(json.dumps({"images": [{"stem": MSL, "image": MSL + ".png", "sol": 150,
                                                             "lmst_h": 11.5}]}))
    c = V.block_candidates(tmp_path)
    assert [x["stem"] for x in c] == [NAV, Z34, MSL]
    assert abs(c[0]["lmst_h"] - 14.7694) < 1e-3 and abs(c[0]["look_el_deg"] + 60) < 1e-4 and c[0]["source"] is None
    assert c[0]["block"] == "mars2020_sol_0686_x"
    assert c[2]["sol"] == 150 and c[2]["source"].endswith(MSL + ".png") and c[2]["block"] == "msl_sol_0124_y"
    assert V.block_candidates([tmp_path, tmp_path]) == c
    assert [x["stem"] for x in V.available(c, None)] == [NAV, Z34]                      # no Curiosity image on disk
    (m / "images" / (MSL + ".png")).write_bytes(b"x")
    assert [x["stem"] for x in V.available(c, {NAV.upper(): tmp_path})] == [NAV, MSL]   # Z34 not in the archive


def test_guess_mask_and_where_the_models_disagree():
    a = np.full((4, 6), 0.9, np.float32)
    b = a.copy()
    a[:, :2] = 0.1; b[:, :2] = 0.2                       # both: rover on the left
    a[0, 5] = 0.2                                        # one model alone excludes a pixel
    b[1, 5] = 0.45; a[1, 5] = 0.62                       # they differ, the mean says terrain
    valid = np.ones((4, 6), bool); valid[3] = False
    g = V.guess_mask([a, b], [0.5, 0.5], valid)
    assert g["mask"].dtype == np.uint8 and set(np.unique(g["mask"])) == {0, 255}
    assert (g["mask"][:3, :2] == 0).all() and (g["mask"][:3, 2:5] == 255).all() and (g["mask"][3] == 0).all()
    assert g["mask"][0, 5] == 255 and g["mask"][1, 5] == 255                            # mean 0.55 and 0.535
    assert g["disagree"][0, 5] and g["disagree"][1, 5] and g["disagree"].sum() == 2 and not g["disagree"][3].any()
    assert g["unsure"][0, 5] and not g["unsure"][0, 0]
    assert abs(g["rover_pct"] - 100 * 6 / 18) < 1e-6 and abs(g["disagree_pct"] - 100 * 2 / 18) < 1e-6
    one = V.guess_mask([a], [0.5])
    assert not one["disagree"].any() and one["mask"][0, 5] == 0


def test_review_panel_marks_the_guess():
    rgb = np.full((60, 80, 3), 90, np.uint8)
    mask = np.full((60, 80), 255, np.uint8); mask[30:, :40] = 0
    dis = np.zeros((60, 80), bool); dis[5:10, 60:70] = True
    p = V.review_panel(rgb, mask, dis)
    assert p.shape == (60, 80 * 2 + 12, 3) and p.dtype == np.uint8
    right = p[:, 92:]
    assert right[45, 10, 0] > right[45, 10, 2] + 40                                     # orange where excluded
    assert tuple(right[7, 65]) == (255, 0, 255)                                         # magenta where they disagree
    assert abs(int(right[10, 30, 0]) - int(p[10, 30, 0])) <= 1                          # terrain as in the image


def test_selection_file_and_report(tmp_path):
    frames, rep = V.select([_nav(100 + i, 600000000 + i * 900, seq=100 + i, site=i) for i in range(5)],
                           V.training_keys([]), n=3, quotas={"N": 1})
    V.write_selection(frames, tmp_path / "selection.csv")
    rows = (tmp_path / "selection.csv").read_text().splitlines()
    assert rows[0].startswith("stem,camera,stratum,eye,sol,lmst_h") and len(rows) == 4 and ",Navcam,N," in rows[1]
    text = "\n".join(V.report_text(rep))
    assert "3 of 3 frames from 5 block images" in text and "Navcam" in text
    assert V.camera_name("Z110") == "Mastcam-Z 110 mm" and V.camera_name("MSL_N") == "Curiosity Navcam"


@pytest.mark.slow
def test_build_and_score_end_to_end_with_a_smoke_model(tmp_path):
    """The guess, the files and the review panels from a real (tiny) model on Curiosity-style block images, then the
    set scored with compare_mask_models --all-frames."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import os
    import subprocess
    import sys
    import cv2
    r = subprocess.run([sys.executable, "-m", "rpp_studies.mask_training.train_mask", "--smoke"], capture_output=True,
                       text=True, timeout=900, env={**os.environ, "TMPDIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr[-2000:]
    smoke = sorted(tmp_path.glob("rpp_mask_smoke_*"))[-1]
    ckpt = str(smoke / "ckpt" / "smoke.pt")
    block = tmp_path / "blocks" / "msl_sol_0124_test"
    (block / "images").mkdir(parents=True)
    images = []
    for i, src in enumerate(sorted((smoke / "set" / "images").glob("*.png"))[:5]):
        stem = f"N{'LR'[i % 2]}B_{422000000 + 1000 * i:09d}RAD_F{6 + i:03d}0082NCAM{200 + i:05d}M1"
        a = cv2.imread(str(src), cv2.IMREAD_COLOR)
        cv2.imwrite(str(block / "images" / f"{stem}.png"), np.dstack([a, np.full(a.shape[:2], 128, np.uint8)]))
        images.append({"stem": stem, "image": f"{stem}.png", "sol": 124 + 10 * i, "lmst_h": 9.0 + i,
                       "R_w2c": [[1, 0, 0], [0, 0, -1], [0, 0.5, -0.866]]})
    (block / "manifest.json").write_text(json.dumps({"images": images}))
    train = tmp_path / "train" / "masks"
    train.mkdir(parents=True)
    (train / (images[0]["stem"] + ".png")).write_bytes(b"x")                 # one of them is a training frame
    out = tmp_path / "val"
    args = ["--blocks", str(tmp_path / "blocks"), "--training", str(tmp_path / "train"), "--out", str(out),
            "--models", ckpt, ckpt, "--n", "3", "--pds", str(tmp_path / "no_archive"), "--device", "cpu"]
    with pytest.raises(SystemExit):
        V.main(args[:5] + [str(tmp_path / "train" / "inside")] + args[6:])    # never inside the training set
    assert V.main(args + ["--select-only"]) == 0 and not (out / "masks").exists()
    with pytest.raises(SystemExit):
        V.main(args)                                                          # no archive: said, not guessed
    (tmp_path / "no_archive").mkdir()
    assert V.main(args) == 0
    masks = sorted((out / "masks").glob("*.png"))
    assert len(masks) == 3 and images[0]["stem"] not in {m.stem for m in masks}
    for m in masks:
        g = cv2.imread(str(m), cv2.IMREAD_UNCHANGED)
        im = cv2.imread(str(out / "images" / m.name), cv2.IMREAD_UNCHANGED)
        assert g.ndim == 2 and set(np.unique(g)) <= {0, 255} and im.shape == g.shape + (4,)
        assert (im[..., 3] == g).all() and (out / "masks_auto" / m.name).is_file()
    assert len(list((out / "review").glob("*.jpg"))) == 3 and (out / "README.txt").is_file()
    rows = (out / "manifest.csv").read_text().splitlines()
    assert rows[0].startswith("review_rank,stem,camera") and len(rows) == 4 and "Curiosity Navcam" in rows[1]
    import csv
    man = list(csv.DictReader(open(out / "manifest.csv", newline="")))
    assert all(float(m["disagree_pct"]) == 0.0 for m in man)                  # one model twice: no disagreement
    assert all(m["rover_pct smoke"] == m["rover_pct"] for m in man) and man[0]["width"] and man[0]["look_el_deg"]
    cleaned = cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)
    cleaned[:4] = 0
    cv2.imwrite(str(masks[0]), cleaned)
    assert V.main(args) == 0                                                  # a rerun keeps the cleaned mask
    assert (cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)[:4] == 0).all()
    ab = tmp_path / "ab"
    r = subprocess.run([sys.executable, "-m", "rpp_studies.mask_training.compare_mask_models", "--data", str(out),
                        "--all-frames", "--frames-like", "^[NFR][LR][AB]_", "--models", ckpt, "--out", str(ab),
                        "--device", "cpu"], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    md = (ab / "mask_ab.md").read_text()
    assert "All test frames" in md and "Curiosity Navcam" in md and "3 frames of val" in md
