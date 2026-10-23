"""Package layout, version, paths and package data. (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
from conftest import ROOT
from pathlib import Path


def test_single_version_source():
    import mppp
    assert mppp.VERSION_TAG == "v%sp%s" % tuple(mppp.__version__.split(".")[:2])
    txt = (ROOT / "pyproject.toml").read_text()
    import re
    project = txt.split("[project]")[1].split("\n[")[0]
    assert 'dynamic = ["version"]' in project and not re.search(r"^version\s*=", project, re.M)
    assert 'version = { attr = "mppp.__version__" }' in txt


def test_package_data_and_cache(tmp_path, monkeypatch):
    from mppp import paths
    d = paths.data_dir()
    for f in ("M20_waypoints.json", "M2020_taus_versus_L_s.csv", "M2020_occlusion_profiles.csv", "models.json",
              "sites.json", "cmods/M2020_NL_fisheye_tangential.json", "cmods/M2020_ZCAM034_focus_model.json"):
        assert (d / f).is_file(), f
    assert paths.source_root() == ROOT
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "c"))
    assert paths.cache_dir() == tmp_path / "c" and (tmp_path / "c").is_dir()
    monkeypatch.delenv("MPPP_CHECKPOINTS", raising=False)
    assert paths.checkpoints_dir() == ROOT / "checkpoints"                   # source checkout
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(tmp_path / "k"))
    assert paths.checkpoints_dir() == tmp_path / "k"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"data/*.json", "data/*.csv", "data/*.txt", "data/cmods/*.json", "data/cmods/*.md"' in pyproject
    assert "history" not in pyproject and "m20_sites" not in pyproject


def test_studies_are_outside_the_package():
    pkg = ROOT / "src" / "mppp"
    assert not (ROOT / "src" / "legacy").exists() and not (pkg / "maps").exists()
    for gone in ("calibration.py", "navcal.py", "navcal_consensus.py", "navcal_report.py", "navcal_tied.py",
                 "zcam_calib.py", "zcam_lines.py", "diagnose.py", "sites.py", "cmods_form.py"):
        assert not (pkg / "sfm" / gone).exists(), gone
    for f in ("calibration/navcal_consensus.py", "calibration/zcam_calib.py", "m20_sites/prototype.py"):
        assert (ROOT / "studies" / f).is_file(), f
    assert not (pkg / "scapes.py").exists()
    for f in ("tests/data/legacy_expected_ZL0.npz", "tests/data/legacy_expected_NLF.npz"):
        assert (ROOT / f).is_file()


ROOT = Path(__file__).resolve().parents[1]


def test_methods_doc_holds_no_site_results():
    txt = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    for bad in ("0.1–0.55 px", "8.5 px", "| Three Forks (52 images)", "59,143", "71.8 %", "0.284 vs 0.288"):
        assert bad not in txt, bad


def test_cmods_dir_is_the_default_camera_folder(monkeypatch, tmp_path):
    from mppp.paths import REPO_ROOT, cmods_dir, data_dir, package_cmods_dir
    from mppp.sfm.project import NAVCAM_PACKAGE_CONSENSUS_DIR, navcam_consensus_dir, zcam_focus_model_path
    monkeypatch.delenv("MPPP_CMODS", raising=False)
    # v0p50: one camera-model folder in the package data; params/ is gone
    assert cmods_dir() == package_cmods_dir() == data_dir() / "cmods" == NAVCAM_PACKAGE_CONSENSUS_DIR
    assert navcam_consensus_dir() == cmods_dir()
    assert zcam_focus_model_path() == cmods_dir() / "M2020_ZCAM034_focus_model.json"
    assert not (REPO_ROOT / "params").exists()
    assert not (data_dir() / "m20_cmods").exists() and not (data_dir() / "navcam_consensus").exists()
    for f in ("M2020_NL_fisheye_tangential.json", "M2020_NR_fisheye_tangential.json", "M2020_N_rig.json",
              "M2020_ZCAM034_focus_model.json", "README.md", "CHANGES.md"):
        assert (cmods_dir() / f).is_file(), f
    assert not list(cmods_dir().glob("*.xml")) and not list(cmods_dir().glob("*rational*"))
    assert not (cmods_dir() / "history").exists()
    for f in ("M20_waypoints.json", "M2020_occlusion_profiles.csv", "M2020_taus_versus_L_s.csv"):
        assert (data_dir() / f).is_file(), f
    # MPPP_CMODS elsewhere: a model missing there falls back to the package copy
    from mppp.calib.models import camera_file, zcam_file
    monkeypatch.setenv("MPPP_CMODS", str(tmp_path))
    assert cmods_dir() == tmp_path and navcam_consensus_dir() == tmp_path
    assert camera_file("N", "L") == package_cmods_dir() / "M2020_NL_fisheye_tangential.json"
    assert zcam_file(34).parent == package_cmods_dir()
    assert zcam_focus_model_path().parent.name == "cmods"


def test_version():
    import mppp
    assert mppp.__version__ == "0.93.7" and mppp.VERSION_TAG == "v0p93"
    cff = (ROOT / "CITATION.cff").read_text()
    assert f"version: {mppp.__version__}" in cff
    assert f"## {mppp.__version__}" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_notebooks_compile_and_have_a_parameters_cell():
    """Not a check of the notebooks' text: every code cell must be valid Python, each notebook has a tagged
    parameters cell, prints the package version, and carries no author paths or version-sync cells."""
    import ast
    import json
    import re
    files = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted((ROOT / "notebooks" / "training").glob("*.ipynb"))
    assert [f.name for f in files] == ["01_process_images.ipynb", "02_colmap_alignment.ipynb",
                                       "03_error_analysis.ipynb", "train_mask.ipynb"]
    for f in files:
        nb = json.loads(f.read_text(encoding="utf-8"))
        cells = nb["cells"]
        assert any("parameters" in c.get("metadata", {}).get("tags", []) for c in cells), f.name
        code = [c for c in cells if c["cell_type"] == "code"]
        assert "mppp.__version__" in "".join(code[0]["source"]), f.name
        for c in code:
            src = "".join(c["source"])
            ast.parse("\n".join(ln for ln in src.splitlines() if not re.match(r"\s*(%|!(?!=))", ln)))    # magics
            for bad in ("NOTEBOOK_FOR_MPPP", "sync_from_claude", "adopt_claude", "D:/", "D:\\", "C:\\"):
                assert bad not in src, (f.name, bad)
    nb2 = json.loads((ROOT / "notebooks" / "02_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src2 = "".join("".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code")
    assert "from mppp.pipeline import align" in src2 and "health_table" in src2 and "plot_camera_shifts" in src2


def test_no_logger_call_with_flush():
    """print -> logging leftovers: Logger.info(..., flush=True) raises TypeError at run time (Chal Rocks, 20 Oct,
    1.6 h into the block, after the pose-guided matching)."""
    import re
    src = ROOT / "src" / "mppp"
    bad = []
    for f in src.rglob("*.py"):
        t = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\b(?:log|logger)\.(?:info|warning|debug|error)\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*flush=", t, re.S):
            bad.append(f"{f.relative_to(ROOT)}:{t[:m.start()].count(chr(10)) + 1}")
    assert not bad, bad
