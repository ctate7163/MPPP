"""Package layout, version, paths and package data. (v0p50: the tests of the earlier test_v0pNN.py files, by module)."""
from conftest import ROOT
from pathlib import Path


def test_single_version_source():
    import rpp
    assert rpp.VERSION_TAG == "v%sp%s" % tuple(rpp.__version__.split(".")[:2])
    txt = (ROOT / "pyproject.toml").read_text()
    import re
    project = txt.split("[project]")[1].split("\n[")[0]
    assert 'dynamic = ["version"]' in project and not re.search(r"^version\s*=", project, re.M)
    assert 'version = { attr = "rpp.__version__" }' in txt


def test_package_data_and_cache(tmp_path, monkeypatch):
    from rpp import paths
    d = paths.data_dir()
    for f in ("M20_waypoints.json", "M2020_taus_versus_L_s.csv", "M2020_occlusion_profiles.csv", "models.json",
              "sites.json", "cmods/M2020_NL_fisheye_tangential.json", "cmods/M2020_ZCAM034_focus_model.json"):
        assert (d / f).is_file(), f
    assert paths.source_root() == ROOT
    monkeypatch.setenv("RPP_CACHE", str(tmp_path / "c"))
    assert paths.cache_dir() == tmp_path / "c" and (tmp_path / "c").is_dir()
    monkeypatch.delenv("RPP_CHECKPOINTS", raising=False)
    assert paths.checkpoints_dir() == ROOT / "checkpoints"                   # source checkout
    monkeypatch.setenv("RPP_CHECKPOINTS", str(tmp_path / "k"))
    assert paths.checkpoints_dir() == tmp_path / "k"
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"data/*.json", "data/*.csv", "data/*.txt", "data/cmods/*.json", "data/cmods/*.md"' in pyproject
    assert "history" not in pyproject and "m20_sites" not in pyproject


def test_studies_are_outside_the_package():
    pkg = ROOT / "src" / "rpp"
    assert not (ROOT / "src" / "legacy").exists() and not (pkg / "maps").exists()
    for gone in ("calibration.py", "navcal.py", "navcal_consensus.py", "navcal_report.py", "navcal_tied.py",
                 "zcam_calib.py", "zcam_lines.py", "diagnose.py", "sites.py", "cmods_form.py"):
        assert not (pkg / "sfm" / gone).exists(), gone
    # 0.96.0: the studies are the package rpp_studies; the old calibration scripts sit in its misc part
    for f in ("misc/calibration_old/navcal_consensus.py", "misc/calibration_old/zcam_calib.py", "sites/prototype.py"):
        assert (ROOT / "src" / "rpp_studies" / f).is_file(), f
    assert not (ROOT / "studies").exists()
    assert not (pkg / "scapes.py").exists()
    for f in ("tests/data/legacy_expected_ZL0.npz", "tests/data/legacy_expected_NLF.npz"):
        assert (ROOT / f).is_file()


def test_methods_doc_holds_no_site_results():
    txt = (ROOT / "docs" / "methods.md").read_text(encoding="utf-8")
    for bad in ("0.1–0.55 px", "8.5 px", "| Three Forks (52 images)", "59,143", "71.8 %", "0.284 vs 0.288"):
        assert bad not in txt, bad


def test_cmods_dir_is_the_default_camera_folder(monkeypatch, tmp_path):
    from rpp.paths import REPO_ROOT, cmods_dir, data_dir, package_cmods_dir
    from rpp.sfm.project import NAVCAM_PACKAGE_CONSENSUS_DIR, navcam_consensus_dir, zcam_focus_model_path
    monkeypatch.delenv("RPP_CMODS", raising=False)
    monkeypatch.delenv("MPPP_CMODS", raising=False)          # 0.99.14: the old name is read too (the PC has it set)
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
    # RPP_CMODS elsewhere: a model missing there falls back to the package copy
    from rpp.calib.models import camera_file, zcam_file
    monkeypatch.setenv("RPP_CMODS", str(tmp_path))
    assert cmods_dir() == tmp_path and navcam_consensus_dir() == tmp_path
    assert camera_file("N", "L") == package_cmods_dir() / "M2020_NL_fisheye_tangential.json"
    assert zcam_file(34).parent == package_cmods_dir()
    assert zcam_focus_model_path().parent.name == "cmods"


def test_version():
    import rpp
    assert rpp.__version__ == "0.99.35" and rpp.VERSION_TAG == "v0p99"
    cff = (ROOT / "CITATION.cff").read_text()
    assert f"version: {rpp.__version__}" in cff
    assert f"## {rpp.__version__}" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_notebooks_compile_and_have_a_parameters_cell():
    """Not a check of the notebooks' text: every code cell must be valid Python, each notebook has a tagged
    parameters cell, prints the package version, and carries no author paths or version-sync cells."""
    import ast
    import json
    import re
    files = sorted((ROOT / "notebooks").glob("*.ipynb")) + sorted((ROOT / "notebooks" / "studies").glob("*.ipynb"))
    # 0.96.0 (Christian, 9 Nov: "clean up the notebooks to have one example of the pipeline and other notebooks
    # for each study"): the pipeline example, then one per study
    assert [f.name for f in files] == ["pipeline_example.ipynb", "van_zyl_ods_example.ipynb", "camera_models.ipynb",
                                       "error_analysis.ipynb", "mask_training.ipynb", "sites_maps.ipynb"]   # 0.99.33: Van Zyl
    for f in files:
        nb = json.loads(f.read_text(encoding="utf-8"))
        cells = nb["cells"]
        assert any("parameters" in c.get("metadata", {}).get("tags", []) for c in cells), f.name
        code = [c for c in cells if c["cell_type"] == "code"]
        assert "rpp.__version__" in "".join(code[0]["source"]), f.name
        for c in code:
            src = "".join(c["source"])
            ast.parse("\n".join(ln for ln in src.splitlines() if not re.match(r"\s*(%|!(?!=))", ln)))    # magics
            for bad in ("NOTEBOOK_FOR_RPP", "sync_from_claude", "adopt_claude", "D:/", "D:\\", "C:\\"):
                assert bad not in src, (f.name, bad)
    # 0.99.33 (Christian, 18 Nov 2026: "Make a notebook that downloads van zyl ecam frames and build the ods and models
    # through the whole pipeline"): from the PDS to the viewer; written by scripts/make_van_zyl_notebook.py
    nb3 = json.loads((ROOT / "notebooks" / "van_zyl_ods_example.ipynb").read_text(encoding="utf-8"))
    src3 = "".join("".join(c["source"]) for c in nb3["cells"] if c["cell_type"] == "code")
    for need in ("pds.fetch_products", "process_site", "align(WORK", "health_table", '"single-sites"', "read_glb",
                 "color_enhance", "write_nerfstudio", "http.server"):
        assert need in src3, need
    nb2 = json.loads((ROOT / "notebooks" / "pipeline_example.ipynb").read_text(encoding="utf-8"))
    src2 = "".join("".join(c["source"]) for c in nb2["cells"] if c["cell_type"] == "code")
    for need in ("process_site", "from rpp.pipeline import align", "health_table", "plot_camera_shifts", "run_dense",
                 "build_station_mesh", "cast_station", "photometric_cube", "render_station", "write_nerfstudio"):
        assert need in src2, need
    assert "rpp_studies" not in src2                   # the pipeline example is the core alone; the studies have their own


def test_no_logger_call_with_flush():
    """print -> logging leftovers: Logger.info(..., flush=True) raises TypeError at run time (Chal Rocks, 20 Oct,
    1.6 h into the block, after the pose-guided matching)."""
    import re
    src = ROOT / "src" / "rpp"
    bad = []
    for f in src.rglob("*.py"):
        t = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\b(?:log|logger)\.(?:info|warning|debug|error)\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*flush=", t, re.S):
            bad.append(f"{f.relative_to(ROOT)}:{t[:m.start()].count(chr(10)) + 1}")
    assert not bad, bad


def test_pypi_packaging():
    """0.94.4 (Christian: https://pypi.org/project/mppp/ for the JOSS submission): the metadata PyPI shows, the
    sdist contents, the trusted-publishing workflow and `rpp --version`.  0.99.0: the distribution is
    rover-photogrammetry-pipeline (Christian, 16 Nov: "I like rover-photogrammetry-pipeline name for PyPI")."""
    import subprocess
    import sys
    try:
        import tomllib
    except ImportError:                                    # Python 3.10
        import tomli as tomllib
    root = Path(__file__).resolve().parents[1]
    pp = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pr = pp["project"]
    assert pr["name"] == "rover-photogrammetry-pipeline" and pr["license"] == "Apache-2.0" and set(pr["license-files"]) == {"LICENSE", "NOTICE"}
    assert pr["authors"][0]["name"] == "Christian Tate" and "Programming Language :: Python :: 3" in pr["classifiers"]
    assert {"Homepage", "Documentation", "Repository", "Issues", "Changelog"} <= set(pr["urls"])
    assert pr["scripts"] == {"rpp": "rpp.cli:main", "mppp": "mppp:cli_main"}      # 0.99: the mppp command too
    assert pp["tool"]["setuptools"]["packages"]["find"]["include"] == ["rpp*", "mppp*"]
    # every data file is in the wheel's package data
    import fnmatch
    pats = pp["tool"]["setuptools"]["package-data"]["rpp"]
    data = root / "src" / "rpp" / "data"
    for f in data.rglob("*"):
        if f.is_file() and "__pycache__" not in f.parts:
            rel = "data/" + f.relative_to(data).as_posix()
            assert any(fnmatch.fnmatch(rel, p) for p in pats), rel
    man = (root / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft tests/data" in man and "prune studies" in man and "prune paper" in man
    wf = (root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert "pypa/gh-action-pypi-publish@release/v1" in wf and "id-token: write" in wf and "environment:" in wf
    assert "release:" in wf and "twine check --strict" in wf
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "pip install \"rover-photogrammetry-pipeline[mask,sfm,plots]\"" in readme
    import re
    assert not re.findall(r"\]\((?!https?://|#|mailto:)[^)]+\)", readme)       # relative links break on PyPI
    r = subprocess.run([sys.executable, "-m", "rpp", "--version"], capture_output=True, text=True, timeout=120)
    import rpp
    assert r.returncode == 0 and r.stdout.strip() == f"rpp {rpp.__version__}"


def test_cmods_dir_reads_rpp_cmods_txt_like_the_batch_files(monkeypatch, tmp_path):
    """0.95.1 (29 Oct): without RPP_CMODS a prompt took the package copy while the batch files took
    scripts/windows/rpp_cmods.txt - a promotion from the prompt went where no run looked.  Now both read the file."""
    from rpp import paths as P
    root = tmp_path / "RPP"
    (root / "scripts" / "windows").mkdir(parents=True)
    use = tmp_path / "cmods"
    use.mkdir()
    monkeypatch.delenv("RPP_CMODS", raising=False)
    monkeypatch.delenv("MPPP_CMODS", raising=False)          # 0.99.14: the old name is read too (the PC has it set)
    monkeypatch.setattr(P, "source_root", lambda: root)
    assert P.cmods_dir() == P.package_cmods_dir()                      # no file: the package copy
    (root / "scripts" / "windows" / "rpp_cmods.txt").write_text(f"{use}\n", encoding="utf-8")
    assert P.cmods_dir() == use
    monkeypatch.setenv("RPP_CMODS", str(tmp_path))                    # the environment still wins
    assert P.cmods_dir() == tmp_path
    # 0.99.0: a tree set up before the rename has mppp_cmods.txt, and its MPPP_CMODS still counts
    monkeypatch.delenv("RPP_CMODS")
    (root / "scripts" / "windows" / "rpp_cmods.txt").unlink()
    (root / "scripts" / "windows" / "mppp_cmods.txt").write_text(f"{use}\n", encoding="utf-8")
    assert P.cmods_dir() == use
    monkeypatch.setenv("MPPP_CMODS", str(tmp_path))
    assert P.cmods_dir() == tmp_path
