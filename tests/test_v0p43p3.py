"""MPPP v0p43.3: interrupted processing runs are completed, Navcam tiles are left out, one batch per scapes folder,
stop_runs.py."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_version():
    import mppp
    assert mppp.__version__ == "0.43.3"


# ------------------------------------------------------------------------------------ interrupted runs completed
def _paths(tmp_path, n):
    return [tmp_path / f"NLF_0100_07000000{i:02d}_000RAD_N0040136NCAM00500_0A0195J01.IMG" for i in range(n)]


def test_interrupted_first_run_is_completed_not_trimmed(tmp_path):
    from mppp.process import filter_to_existing
    out = tmp_path / "processed"
    (out / "images_png8").mkdir(parents=True)
    paths = _paths(tmp_path, 10)
    for p in paths[:3]:                                  # a first run stopped after 3 images: no manifest yet
        (out / "images_png8" / f"{p.stem}.png").write_bytes(b"x")
    kept, rep = filter_to_existing(paths, out, "PNG8")
    assert len(kept) == 10 and rep["n_removed"] == 0 and rep["never_processed"] == 7


def test_deleted_images_stay_out_and_new_ones_come_in(tmp_path):
    from mppp.process import filter_to_existing
    out = tmp_path / "processed"
    (out / "images_png8").mkdir(parents=True)
    paths = _paths(tmp_path, 10)
    done = paths[:6]
    (out / "mppp_manifest_v0p43.json").write_text(json.dumps({"images": [{"source_product": str(p)} for p in done]}))
    for p in done[:4]:                                   # the user deleted 2 of the 6 processed images
        (out / "images_png8" / f"{p.stem}.png").write_bytes(b"x")
    kept, rep = filter_to_existing(paths, out, "PNG8")
    assert {p.stem for p in kept} == {p.stem for p in paths[:4] + paths[6:]}
    assert rep["n_removed"] == 2 and rep["never_processed"] == 4
    assert rep["removed"] == sorted(p.name for p in paths[4:6])


# ------------------------------------------------------------------------------------------------ Navcam tiles
def test_select_best_products_leaves_out_navcam_tiles(tmp_path, monkeypatch):
    from mppp.sfm import project as P
    tile = tmp_path / "NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    full = tmp_path / "NRF_0092_0675115592_000RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    half = tmp_path / "NLF_0092_0675110134_691RAD_N0040136NCAM00507_0A0295J01.IMG"
    for f in (tile, full, half):
        f.write_bytes(b"")
    sizes = {tile.name: 7426560, full.name: 118046720, half.name: 1774080}
    fractions = {tile.name: 1 / 16, half.name: 1.0}
    calls = []

    def fake(path, size=None):
        calls.append(Path(path).name)
        return fractions.get(Path(path).name)
    monkeypatch.setattr(P, "frame_fraction", fake)
    kept, rep = P.select_best_products([tile, full, half], sizes=sizes)
    assert {p.name for p in kept} == {full.name, half.name}
    assert rep["n_dropped_subframes"] == 1 and rep["dropped_subframes"][0]["file"] == tile.name
    assert rep["n_superseded"] == 0
    kept, rep = P.select_best_products([tile, full, half], sizes=sizes, min_frame_fraction=None)
    assert len(kept) == 3 and rep["n_dropped_subframes"] == 0


def test_frame_fraction_reads_labels_only_for_small_files(tmp_path, monkeypatch):
    from mppp.sfm import project as P
    import mppp.labels as L
    big = tmp_path / "NRF_0092_0675115592_000RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    big.write_bytes(b"")
    monkeypatch.setattr(L, "read_pds", lambda *a, **k: pytest.fail("label read for a full-size file"))
    assert P.frame_fraction(big, size=118046720) is None
    tile = tmp_path / "NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01.IMG"
    tile.write_bytes(b"")
    monkeypatch.setattr(L, "read_pds", lambda *a, **k: ({"IMAGE": {"LINES": 960, "LINE_SAMPLES": 1280}}, None))
    assert P.frame_fraction(tile, size=7426560) == pytest.approx(1 / 16)
    half = tmp_path / "NLF_0092_0675110134_691RAD_N0040136NCAM00507_0A0195J01.IMG"   # downsample 1: 2560 x 1920
    half.write_bytes(b"")
    monkeypatch.setattr(L, "read_pds", lambda *a, **k: ({"IMAGE": {"LINES": 1920, "LINE_SAMPLES": 2560}}, None))
    assert P.frame_fraction(half, size=1774080) == pytest.approx(1.0)
    zcam = tmp_path / "ZL0_0709_0729888971_069RAD_N0332864ZCAM07114_0340LMA01.IMG"
    zcam.write_bytes(b"")
    kept, rep = P.select_best_products([zcam], sizes={zcam.name: 100}, sequence_prefix=("ZCAM",))
    assert len(kept) == 1 and rep["n_dropped_subframes"] == 0            # Mastcam-Z is not filtered


def test_real_tile_label_if_staged():
    f = Path("/mnt/user-data/uploads/m2020/datadrive/00092/ids/rdr/ncam/"
             "NRF_0092_0675115592_573RAD_N0040136NCAM00698_0A00LLJ01.IMG")
    if not f.is_file():
        pytest.skip("tile product not available here")
    from mppp.sfm.project import frame_fraction
    assert frame_fraction(f) == pytest.approx(1 / 16)


def test_notebook03_reports_dropped_tiles():
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "".join(nb["cells"][6]["source"])
    assert "n_dropped_subframes" in src and 'k != "dropped_subframes"' in src


# ------------------------------------------------------------------------------------------ batch lock and stop
def test_pid_alive_and_dead_status(tmp_path):
    from mppp.runner import RunStatus, pid_alive, read_status
    assert pid_alive(os.getpid())
    assert not pid_alive(2 ** 22 + 12345)
    st = RunStatus(tmp_path / "s.json", pid=2 ** 22 + 12345)
    st.update(state="running")
    d = read_status(tmp_path / "s.json")
    assert d["alive"] is False and d["state"].startswith("stopped")


def test_batch_lock_refuses_a_second_batch(tmp_path):
    from mppp.runner import BATCH_FILE, BatchRunning, batch_lock, read_status
    b = batch_lock(tmp_path, "process_sites")
    try:
        with pytest.raises(BatchRunning, match="already running"):
            batch_lock(tmp_path, "run_sites")
    finally:
        b.stop("finished")
    assert read_status(tmp_path / BATCH_FILE)["alive"] is False
    b2 = batch_lock(tmp_path, "run_sites")                  # free again
    b2.stop("finished")


def test_process_sites_refuses_while_a_batch_runs(tmp_path):
    import importlib.util
    from mppp.runner import batch_lock
    spec = importlib.util.spec_from_file_location("process_sites", ROOT / "scripts" / "process_sites.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    b = batch_lock(tmp_path, "run_sites")
    try:
        assert ps.main(["--sites", "sid", "--root", str(tmp_path)]) == 3
        assert "not started" in (tmp_path / "process_sites_log.txt").read_text()
    finally:
        b.stop("finished")


def test_mppp_processes_picks_the_tops_of_run_trees():
    from mppp.runner import mppp_processes
    rows = [
        {"pid": 10, "ppid": 1, "cmd": r'cmd /c call "D:\code\MPPP\scripts\windows\_run_process.bat" --root D:\x --all'},
        {"pid": 11, "ppid": 10, "cmd": r'python "D:\code\MPPP\scripts\process_sites.py" --root D:\x --all'},
        {"pid": 12, "ppid": 11, "cmd": r"python -m ipykernel_launcher -f kernel-1.json"},
        {"pid": 20, "ppid": 1, "cmd": r"python D:\code\MPPP\scripts\run_sites.py --all"},
        {"pid": 21, "ppid": 20, "cmd": r"python D:\code\MPPP\scripts\align_scape.py D:\x\sid_colmap"},
        {"pid": 30, "ppid": 1, "cmd": r"python -m ipykernel_launcher -f kernel-2.json"},        # Jupyter: untouched
        {"pid": 31, "ppid": 1, "cmd": r"jupyter-lab"},
        {"pid": 40, "ppid": 1, "cmd": r'cmd /c "D:\code\MPPP\scripts\windows\stop_mppp.bat"'},
    ]
    assert [r["pid"] for r in mppp_processes(rows)] == [10, 20]


def test_mark_stopped(tmp_path):
    from mppp.runner import BATCH_FILE, STATUS_FILE, RunStatus, mark_stopped, read_status
    (tmp_path / "sid_colmap").mkdir()
    a = RunStatus(tmp_path / BATCH_FILE, pid=4242)
    a.update(state="running")
    b = RunStatus(tmp_path / "sid_colmap" / STATUS_FILE, pid=os.getpid())
    b.update(state="running")
    changed = mark_stopped(tmp_path, [4242])
    assert changed == [str(tmp_path / BATCH_FILE)]
    assert read_status(tmp_path / BATCH_FILE)["state"] == "stopped"
    assert read_status(tmp_path / "sid_colmap" / STATUS_FILE)["alive"] is True


def test_stop_runs_lists_nothing_here(capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stop_runs", ROOT / "scripts" / "stop_runs.py")
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    assert sr.main(["--list", "--root", "/nonexistent"]) == 0
    assert "MPPP run" in capsys.readouterr().out


def test_stop_mppp_bat():
    b = (ROOT / "scripts" / "windows" / "stop_mppp.bat").read_bytes()
    assert b.count(b"\n") == b.count(b"\r\n") and b"stop_runs.py" in b and b"mppp_env.bat" in b
    for name in ("process_sites.py", "run_sites.py"):
        assert "batch_lock(" in (ROOT / "scripts" / name).read_text()


def test_mppp_processes_never_includes_its_own_ancestors():
    from mppp.runner import mppp_processes
    me, parent = os.getpid(), os.getppid()
    rows = [{"pid": parent, "ppid": 1, "cmd": "bash -c 'python scripts/stop_runs.py; python scripts/process_sites.py'"},
            {"pid": me, "ppid": parent, "cmd": "python scripts/stop_runs.py"},
            {"pid": 999999, "ppid": 1, "cmd": "python scripts/process_sites.py --all"}]
    assert [r["pid"] for r in mppp_processes(rows)] == [999999]


def test_check_sites(tmp_path):
    import importlib.util
    from mppp.sfm.sites import SITES_FILE, validate_site_table
    err, warn = validate_site_table()
    assert err == [] and not any("overlap" in w for w in warn)           # nested blocks are not flagged
    bad = tmp_path / "s.json"
    txt = SITES_FILE.read_text(encoding="utf-8").replace('"settings": {}},', '"settings": {}}', 1)
    bad.write_text(txt, encoding="utf-8")
    err, _ = validate_site_table(bad)
    assert len(err) == 1 and "comma missing" in err[0] and "line" in err[0]
    d = json.loads(SITES_FILE.read_text(encoding="utf-8"))
    d["sites"]["my_site"] = {"sols": [712, 700], "settings": {"ATTITUDE_PRIOR_DEGG": 1}, "no_mask_inference_at": ["x"]}
    d["groups"]["mine"] = ["my_site", "nowhere"]
    bad.write_text(json.dumps(d), encoding="utf-8")
    err, warn = validate_site_table(bad, known_settings={"ATTITUDE_PRIOR_DEG"})
    txt = "\n".join(err + warn)
    assert "end before they start" in txt and "'nowhere'" in txt and "ATTITUDE_PRIOR_DEGG" in txt and "x" in txt
    spec = importlib.util.spec_from_file_location("check_sites", ROOT / "scripts" / "check_sites.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    assert cs.main([]) == 0 and cs.main(["--sites-file", str(bad)]) == 1
    assert "ATTITUDE_PRIOR_DEG" in cs.notebook03_settings()


# ---------------------------------------------------------------------------------------- params/cmods (in use)
def test_cmods_dir_is_the_default_camera_folder(monkeypatch, tmp_path):
    from mppp.paths import REPO_ROOT, cmods_dir
    from mppp.sfm.project import (NAVCAM_PACKAGE_CONSENSUS_DIR, navcam_consensus_dir, navcam_cameras_fingerprint,
                                  zcam_focus_model_path)
    monkeypatch.delenv("MPPP_CMODS", raising=False)
    assert cmods_dir() == REPO_ROOT / "params" / "cmods"
    assert navcam_consensus_dir() == REPO_ROOT / "params" / "cmods"
    assert zcam_focus_model_path() == REPO_ROOT / "params" / "cmods" / "M2020_ZCAM034_focus_model.json"
    # the working copy starts as the shipped models, byte for byte
    assert navcam_cameras_fingerprint(cmods_dir()) == navcam_cameras_fingerprint(NAVCAM_PACKAGE_CONSENSUS_DIR)
    # MPPP_CMODS elsewhere; an empty folder falls back to the package copies
    monkeypatch.setenv("MPPP_CMODS", str(tmp_path))
    assert cmods_dir() == tmp_path
    assert navcam_consensus_dir() == NAVCAM_PACKAGE_CONSENSUS_DIR
    assert zcam_focus_model_path().parent.name == "m20_cmods"


def test_promote_cmods(monkeypatch, tmp_path, capsys):
    import importlib.util
    import shutil
    from mppp.sfm.project import NAVCAM_PACKAGE_CONSENSUS_DIR, navcam_consensus_dir
    dst = tmp_path / "cmods"
    monkeypatch.setenv("MPPP_CMODS", str(dst))
    spec = importlib.util.spec_from_file_location("promote_cmods", ROOT / "scripts" / "promote_cmods.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    assert pc.main([str(NAVCAM_PACKAGE_CONSENSUS_DIR), "--note", "first"]) == 0
    assert navcam_consensus_dir() == dst and (dst / "CHANGES.md").read_text().count("first") == 1
    # a new NL camera: the old one goes to history/
    new = tmp_path / "new"
    new.mkdir()
    d = json.loads((NAVCAM_PACKAGE_CONSENSUS_DIR / "M2020_NL_fisheye_tangential.json").read_text())
    d["params"][0] += 1.0
    (new / "M2020_NL_fisheye_tangential.json").write_text(json.dumps(d))
    assert pc.main([str(new), "--note", "better"]) == 0
    assert json.loads((dst / "M2020_NL_fisheye_tangential.json").read_text())["params"][0] == d["params"][0]
    hist = list((dst / "history").glob("*/M2020_NL_fisheye_tangential.json"))
    assert len(hist) == 1
    # a broken file is refused and nothing changes
    (new / "M2020_N_rig.json").write_text("{}")
    before = (dst / "M2020_N_rig.json").read_bytes()
    assert pc.main([str(new / "M2020_N_rig.json")]) == 1
    assert (dst / "M2020_N_rig.json").read_bytes() == before
    # the focus model under any candidate name
    z = tmp_path / "M2020_ZCAM034_focus_model_candidate.json"
    shutil.copy(ROOT / "params" / "cmods" / "M2020_ZCAM034_focus_model.json", z)
    assert pc.main([str(z)]) == 0 and (dst / "M2020_ZCAM034_focus_model.json").is_file()
    assert pc.main(["--list"]) == 0 and "focus model" in capsys.readouterr().out


def test_notebook03_points_to_params_cmods():
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    params = "".join(nb["cells"][3]["source"])
    assert "params/cmods" in params and "NAVCAM_CAMERAS    = NAVCAM_CONSENSUS_DIR" in params
    assert (ROOT / "params" / "cmods" / "README.md").is_file()


# ------------------------------------------------------------------------------ frame observation limit (20)
def test_min_frame_observations_default_20_and_one_setting(tmp_path, monkeypatch):
    import inspect
    from mppp.sfm import reconstruction as R
    assert R.MIN_FRAME_OBSERVATIONS == 20
    assert R.OUTLIER_DEFAULTS["min_observations"] is None                  # follows MIN_FRAME_OBSERVATIONS
    assert inspect.signature(R.bundle_adjust).parameters["min_frame_observations"].default is None
    assert "min_frame_observations" in inspect.signature(R.reconstruct).parameters
    seen = {}

    def fake(project, *a, **k):
        seen["during"] = R.MIN_FRAME_OBSERVATIONS
        raise RuntimeError("stop")
    monkeypatch.setattr(R, "_reconstruct", fake)
    with pytest.raises(RuntimeError):
        R.reconstruct(None, min_frame_observations=12)
    assert seen["during"] == 12 and R.MIN_FRAME_OBSERVATIONS == 20        # restored after the run


def test_outlier_test_uses_the_frame_limit(tmp_path):
    pytest.importorskip("pyceres")
    from test_sfm import _build_rec, _synthetic
    from mppp.sfm import reconstruction as R
    proj, truth, P, cams, rigT, noise, rng = _synthetic(tmp_path)
    rec, _ = _build_rec(proj, truth, P, cams, rigT, noise, rng, perturb=False)
    obs = min(sum(1 for q in rec.images[d.id].points2D if q.has_point3D()) for f in rec.frames.values()
              for d in f.data_ids)
    few = lambda th: [f for f in R.find_outlier_frames(rec, proj, **th)                          # noqa: E731
                      if any("observations <" in r for r in f["reasons"])]
    assert few({"min_observations": 10 ** 7})                              # everything is "too few"
    assert not few({"min_observations": 1})
    assert not few({}) or obs < R.MIN_FRAME_OBSERVATIONS                   # default: the module limit


def test_notebook03_min_frame_observations():
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "MIN_FRAME_OBSERVATIONS = 20" in src and "min_frame_observations=MIN_FRAME_OBSERVATIONS" in src
