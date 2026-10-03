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
              "cmods/M2020_NL0_frame.xml", "cmods/ZL034_frame.xml"):
        assert (d / f).is_file(), f
    assert paths.params_dir() == d and paths.source_root() == ROOT
    monkeypatch.setenv("MPPP_CACHE", str(tmp_path / "c"))
    assert paths.cache_dir() == tmp_path / "c" and (tmp_path / "c").is_dir()
    monkeypatch.delenv("MPPP_CHECKPOINTS", raising=False)
    assert paths.checkpoints_dir() == ROOT / "checkpoints"                   # source checkout
    monkeypatch.setenv("MPPP_CHECKPOINTS", str(tmp_path / "k"))
    assert paths.checkpoints_dir() == tmp_path / "k"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"data/*.json", "data/*.csv", "data/cmods/*.xml"' in pyproject


def test_legacy_and_studies_are_outside_the_package():
    pkg = ROOT / "src" / "mppp"
    assert not (pkg / "error" / "compat.py").exists() and (ROOT / "src/legacy/error_compat.py").is_file()
    assert not (pkg / "error" / "study_navcam.py").exists() and (ROOT / "studies/error/study_navcam.py").is_file()
    assert not (pkg / "error" / "data").exists()
    for f in ("image.py", "readers.py", "writers.py", "config.json"):
        assert (ROOT / "src/legacy" / f).is_file() and not (ROOT / "src" / f).exists()


ROOT = Path(__file__).resolve().parents[1]


def test_methods_doc_holds_no_site_results():
    txt = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    for bad in ("0.1–0.55 px", "8.5 px", "| Three Forks (52 images)", "59,143", "71.8 %", "0.284 vs 0.288"):
        assert bad not in txt, bad
    notes = (ROOT / "docs" / "results" / "working_notes.md").read_text(encoding="utf-8")
    assert "Withdrawn" in notes and "0.157" in notes
    assert (ROOT / "docs" / "results" / "sites.md").is_file() and (ROOT / "scripts" / "sites_table.py").is_file()


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
              "M2020_ZCAM034_focus_model.json", "M2020_NL_rational.json", "M2020_NL0_frame.xml", "ZL034_frame.xml",
              "README.md", "CHANGES.md"):
        assert (cmods_dir() / f).is_file(), f
    for f in ("M20_waypoints.json", "M2020_occlusion_profiles.csv", "M2020_taus_versus_L_s.csv"):
        assert (data_dir() / f).is_file(), f
    # MPPP_CMODS elsewhere; an empty folder falls back to the package copies
    monkeypatch.setenv("MPPP_CMODS", str(tmp_path))
    assert cmods_dir() == tmp_path
    assert navcam_consensus_dir() == NAVCAM_PACKAGE_CONSENSUS_DIR
    assert zcam_focus_model_path().parent.name == "cmods"


def test_version():
    import mppp
    assert mppp.__version__ == "0.71.0" and mppp.VERSION_TAG == "v0p71"


def test_notebooks_compile_and_have_a_parameters_cell():
    """Not a check of the notebooks' text: every code cell must be valid Python and the runner needs the tagged
    parameters cell (and, in notebook 03, the section-3 heading where processing-only runs stop)."""
    import ast
    import json
    import re
    from mppp.runner import PROCESS_STOP
    for f in sorted((ROOT / "notebooks").glob("0[345]_*.ipynb")):
        nb = json.loads(f.read_text(encoding="utf-8"))
        cells = nb["cells"]
        assert any("parameters" in c.get("metadata", {}).get("tags", []) for c in cells), f.name
        for c in cells:
            if c["cell_type"] == "code":
                src = "".join(c["source"])
                ast.parse("\n".join(ln for ln in src.splitlines() if not re.match(r"\s*(%|!(?!=))", ln)))    # magics
        if f.name.startswith("03_"):
            assert any(c["cell_type"] == "markdown" and "".join(c["source"]).startswith(PROCESS_STOP) for c in cells)
