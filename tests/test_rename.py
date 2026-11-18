"""0.99.0, the rename MPPP -> RPP (Christian, 16 Nov 2026: "replace every mppp with rpp throughout the code and
documentation. make sure it doesn't break the code anywhere").

Everything new says RPP; what a PC, a block or a script already has keeps working: the old imports and pickles, the
old environment variables, the old mask names, the WORK-folder file names and the run keys.
"""
import json
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# --- where the old spelling may stay ---------------------------------------------------------------------------
WHOLE_FILES = ("docs/history/", "paper/", "CHANGELOG.md", "src/rpp/data/cmods/", "src/mppp/", "src/mppp_studies/",
               "src/rpp/ondisk.py", "src/rpp/env.py", "src/rpp/mask_names.py", "src/rpp/data/models.json",
               "scripts/windows/mppp_env.bat", "tests/test_rename.py")
KEPT = [
    # names in WORK folders, roots and records (rpp.ondisk)
    r"_?mppp_(?:manifest|config|settings|status|dense|sfm_log|stereo|stopped|batch|pipeline|regen)\w*",
    r"mppp_version(?:_tag)?", r"mppp_card", r"mppp-mask-safetensors-1", r"mppp_example_m2020_rockytop",
    r"[\"']mppp[\"'](?=\s*:)", r"\.get\([\"']mppp[\"']", r"\[[\"']mppp[\"']\]", r"\bp\.mppp\b", r"k == \"mppp\"",
    # the user cache, the PC's folders and conda environments
    r"(?i)(?:LOCALAPPDATA%{1,2}|AppData[\\/]+Local|\.cache|Library[\\/]+Caches|XDG_CACHE_HOME\}?)[\\/]+mppp\b",
    r"/ \"mppp\"", r"(?i)code\\{1,2}MPPP\w*", r"(?i)code/MPPP\w*", r"mppp_gpu(?:_lg)?",
    # the compatibility layer, and text about the rename
    r"MPPP_(?:\*|X\b|<NAME>|HOME|SCAPES|PDS|CMODS|THREADS)", r"set MPPP_", r"\"MPPP_\"", r"\(rpp mppp\)",
    r"mppp_(?:python|scapes|pds|cmods|\*)\.txt", r"mppp_mask_(?:v\d|convnext_tiny_s4_v[\d*]\S*)", r"ctate7163/mppp-mask",
    r"MPPPImage", r"mppp\.image\.MPPPImage", r"import mppp", r"python -m mppp", r"mppp\*", r"mppp:cli_main",
    r"mppp, mppp_studies", r"the mppp command", r"\bmppp = ", r"formerly mppp", r"``mppp:``",
    r"mppp_env\.bat", r"install of mppp",
    r"mppp_v\*\.bundle", r"\"MPPP freeze\"",            # 0.99.3: the v0p98 line's bundles and its freeze restart entry
    r"MPPP 0\.98", r"rename_mppp_to_rpp", r"From MPPP to RPP",   # 0.99.5: the old line, and the PC's rename batch
    r"MPPP -> RPP", r"\(RPP\|MPPP\)", r"/MPPP/ods", r"old name, MPPP", r"[Ff]ormerly MPPP", r"called MPPP",
    r"was MPPP stereo viewer", r"MPPP \(Mars Photogrammetry Preprocessing Pipeline\)",
    r"MPPP, Mars Photogrammetry Preprocessing Pipeline", r"made by MPPP", r"pypi\.org/project/mppp/",
    # Christian's own words
    r"other MPPP users", r"the MPPP directory", r"I will stop MPPP", r"the standard mppp pipeline",
    r"the MPPP/scripts/windows/ folder", r"the mppp metashape and nerfstudio commands",
]
_KEPT = [re.compile(p) for p in KEPT]


def _tracked():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
        return [f for f in out.decode().split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        return [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]


def stray_old_names(files=None):
    """``file:line: text`` of every MPPP spelling that is not one of the kept names."""
    bad = []
    for rel in files or _tracked():
        if rel.startswith(WHOLE_FILES):
            continue
        p = ROOT / rel
        try:
            raw = p.read_bytes()
        except OSError:
            continue
        if b"\0" in raw or not re.search(rb"(?i)mppp", raw):
            continue
        text = raw.decode("utf-8", "replace")
        for n, line in enumerate(text.splitlines(), 1):
            hits = [m.span() for m in re.finditer(r"(?i)mppp", line)]
            if not hits:
                continue
            ok = [m.span() for rx in _KEPT for m in rx.finditer(line)]
            for a, b in hits:
                if not any(x <= a and b <= y for x, y in ok):
                    bad.append(f"{rel}:{n}: {line.strip()[:140]}")
                    break
    return bad


def test_the_old_name_is_left_only_where_it_is_kept():
    bad = stray_old_names()
    assert not bad, "\n".join(bad[:40])


def test_no_file_name_says_mppp_outside_the_kept_ones():
    names = [f for f in _tracked() if re.search(r"(?i)mppp", f)]
    allowed = ("docs/history/", "src/mppp/", "src/mppp_studies/", "scripts/windows/mppp_env.bat")
    assert names and all(f.startswith(allowed) for f in names), [f for f in names if not f.startswith(allowed)]


# --- imports, pickles, commands -----------------------------------------------------------------------------------
def test_import_mppp_gives_the_rpp_modules():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import mppp
        import mppp.sfm.project as old
        import mppp_studies.calibration.freeze as old_freeze
    import rpp
    import rpp.sfm.project as new
    import rpp_studies.calibration.freeze as new_freeze
    assert old is new and old_freeze is new_freeze and mppp.__version__ == rpp.__version__
    from mppp.image import MPPPImage
    from rpp import RPPImage
    assert MPPPImage is RPPImage
    from mppp import pipeline                                       # a submodule through the package's attributes
    assert pipeline is sys.modules["rpp.pipeline"]


def test_a_pickle_that_names_mppp_classes_loads():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from rpp.sfm.project import SfmProject
        old = pickle.dumps(SfmProject, protocol=0).replace(b"rpp.sfm.project", b"mppp.sfm.project")
        assert b"mppp.sfm.project" in old and pickle.loads(old) is SfmProject


def test_python_m_mppp_runs_rpp_with_a_note():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-m", "mppp", "--version"], capture_output=True, text=True, timeout=120,
                       env=env)
    import rpp
    assert r.returncode == 0 and r.stdout.strip() == f"rpp {rpp.__version__}" and "called rpp" in r.stderr


# --- environment variables ----------------------------------------------------------------------------------------
def test_an_old_variable_alone_still_works(monkeypatch, tmp_path):
    from rpp import env as rpp_env
    from rpp import pipeline as P
    monkeypatch.delenv("RPP_SCAPES", raising=False)
    monkeypatch.setenv("MPPP_SCAPES", str(tmp_path))
    assert rpp_env.get("RPP_SCAPES") == str(tmp_path) and rpp_env.get("SCAPES") == str(tmp_path)
    assert P.Settings().scapes_root == tmp_path
    monkeypatch.setenv("RPP_SCAPES", str(tmp_path / "new"))         # the new name wins
    assert P.Settings().scapes_root == tmp_path / "new" and rpp_env.which("SCAPES") == "RPP_SCAPES"
    child = {"MPPP_HOME": "D:/code/MPPP", "MPPP_JOBS": "3", "RPP_PDS": "x", "MPPP_PDS": "y"}
    assert rpp_env.adopt_legacy(child) == ["RPP_JOBS"] and child["RPP_JOBS"] == "3" and child["RPP_PDS"] == "x"
    assert "RPP_HOME" not in child                                  # a tree's home is its own folder


def test_no_python_reads_an_rpp_variable_past_the_fallback():
    """Every ``RPP_*`` read goes through rpp.env (or names it for a child environment)."""
    bad = []
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        t = f.read_text(encoding="utf-8")
        for m in re.finditer(r"os\.(?:environ\.get|getenv)\(\s*[\"']RPP_", t):
            bad.append(f"{f.relative_to(ROOT)}:{t[:m.start()].count(chr(10)) + 1}")
    assert not bad, bad


def test_rpp_env_bat_reads_the_old_names():
    t = (ROOT / "scripts" / "windows" / "rpp_env.bat").read_text(encoding="utf-8")
    assert "('set MPPP_ 2^>nul') do call :_rpp_legacy %%A" in t and ':_rpp_legacy' in t
    assert 'if /i "%~1"=="MPPP_HOME" exit /b 0' in t and 'if /i "%~1"=="--legacy-only" exit /b 0' in t
    for what in ("python", "scapes", "pds", "cmods"):
        assert f'for %%N in (rpp mppp) do if not defined RPP_{what.upper()} if exist "%~dp0%%N_{what}.txt"' in t
    stub = (ROOT / "scripts" / "windows" / "mppp_env.bat").read_text(encoding="utf-8")
    assert 'call "%~dp0rpp_env.bat" %*' in stub
    for bat in ("freeze.bat", "sync_code.bat", "run_zcam_states.bat"):
        assert "rpp_env.bat\" --legacy-only" in (ROOT / "scripts" / "windows" / bat).read_text(encoding="utf-8"), bat


def test_sync_code_and_freeze_stay_on_main():
    t = (ROOT / "scripts" / "windows" / "sync_code.bat").read_text(encoding="utf-8")
    assert "rev-parse --abbrev-ref HEAD" in t and '"main"' in t
    # 0.99.5: the freeze clones the tree's own branch (main after rename_mppp_to_rpp.bat) and refuses a detached HEAD;
    # its checkout follows the tree's code (src\rpp: RPP_freeze), not the branch name
    t = (ROOT / "scripts" / "windows" / "freeze.bat").read_text(encoding="utf-8")
    assert "rev-parse --abbrev-ref HEAD" in t and '"!FREEZE_BRANCH!"=="HEAD"' in t
    assert 'if exist "%RPP_MAIN%\\src\\rpp\\__init__.py" set "FREEZE_DEFAULT_CODE=D:\\code\\RPP_freeze"' in t
    assert '"%RPP_TRANSFER%\\rpp_v*.bundle"' in (ROOT / "scripts" / "windows" / "sync_code.bat").read_text()


# --- the data on disk -----------------------------------------------------------------------------------------------
def test_the_kept_names_are_the_ones_the_code_writes():
    from rpp import ondisk
    from rpp.batch import BATCH_DIR, BATCH_FILE, STOP_MARKER
    from rpp.runner import STATUS_FILE
    from rpp.sfm.workdir import SETTINGS_FILE
    from rpp.paths import cache_dir
    assert (STATUS_FILE, SETTINGS_FILE, STOP_MARKER, BATCH_FILE, BATCH_DIR) == \
        ("mppp_status.json", "mppp_settings.json", "mppp_stopped.txt", "mppp_batch.json", "mppp_batches")
    assert ondisk.CACHE_DIR == "mppp" and ondisk.MASK_FORMAT == "mppp-mask-safetensors-1"
    assert os.environ.get("RPP_CACHE") or os.environ.get("MPPP_CACHE") or cache_dir(create=False).name == "mppp"


def test_old_mask_names_are_the_same_models():
    from rpp.mask_names import canonical, key_name
    reg = json.loads((ROOT / "src" / "rpp" / "data" / "models.json").read_text(encoding="utf-8"))
    assert reg["default"] == "rpp_mask_v3" and sorted(reg["models"]) == [f"rpp_mask_v{i}" for i in (1, 2, 3, 4)]
    for i in (1, 2, 3, 4):
        e = reg["models"][f"rpp_mask_v{i}"]
        assert e["aliases"] == [f"mppp_mask_v{i}"] and e["old_files"] == [f"mppp_mask_convnext_tiny_s4_v{i}.safetensors"]
        assert e["file"] == f"rpp_mask_convnext_tiny_s4_v{i}.safetensors"
        assert e["urls"][0] == f"https://huggingface.co/ctate7163/rpp-mask/resolve/main/{e['file']}"
        assert e["urls"][1] == f"https://huggingface.co/ctate7163/mppp-mask/resolve/main/{e['old_files'][0]}"
        assert canonical(f"mppp_mask_v{i}") == f"rpp_mask_v{i}" == canonical(f"rpp_mask_v{i}")
        assert key_name(f"rpp_mask_v{i}") == f"mppp_mask_v{i}" == key_name(f"mppp_mask_v{i}")
    assert reg["models"]["rpp_mask_v4"]["released"] is False        # 16 Nov: not on Hugging Face (404)
    assert canonical("checkpoints/mine.pt") == "checkpoints/mine.pt" and canonical(None) is None


def test_a_v0p98_configuration_is_not_changed_by_the_mask_name():
    from rpp.config import default_config
    from rpp.process import _canon, _config_diff
    new = default_config()
    old = json.loads(json.dumps(new))
    old["masking"]["checkpoint"] = "mppp_mask_v3"
    assert new["masking"]["checkpoint"] == "rpp_mask_v3"
    assert _config_diff(_canon(new), _canon(old)) == []
    old["masking"]["checkpoint"] = "mppp_mask_v4"
    assert _config_diff(_canon(new), _canon(old)) == ["masking.checkpoint"]


def test_a_v0p98_work_folder_is_reused(tmp_path):
    """A WORK folder processed by 0.98 (mppp_manifest_v0p98.json, mppp_config_v0p98.json naming mppp_mask_v3):
    every image is reused, none processed again."""
    from rpp.config import default_config
    from rpp.process import reusable_images
    out = tmp_path / "processed"
    (out / "images").mkdir(parents=True)
    cfg = default_config()
    old_cfg = json.loads(json.dumps(cfg))
    old_cfg["masking"]["checkpoint"] = "mppp_mask_v3"
    (out / "images" / "a.png").write_bytes(b"x")
    meta = {"source_product": "NLF_0461_X.IMG", "outputs": {"png": "images/a.png"}, "mask": {"inferred": True},
            "site": 26, "drive": 630, "filename": {"sequence": "NCAM03461"}}
    (out / "mppp_manifest_v0p98.json").write_text(json.dumps({"mppp_version": "v0p98", "images": [meta]}))
    wp = {"sha256": "abc"}
    (out / "mppp_config_v0p98.json").write_text(json.dumps({"mppp_version": "0.98.29", "mppp_version_tag": "v0p98",
                                                            "config": old_cfg, "waypoints": wp}))
    have, rep = reusable_images(out, cfg, {"_rpp_source": wp})
    assert rep.get("reason") is None and rep["config_changed"] == [] and list(have) == ["NLF_0461_X"], rep


def test_run_keys_did_not_change_with_the_rename():
    """The run and processing keys of a few blocks, as 0.98.29 computed them (all 122 site/zoom keys of sites.json
    were compared on 16 Nov 2026): the rename makes no block align or process again."""
    from rpp import pipeline as P
    s = P.Settings(scapes_root=Path("/nowhere"), pds_dir=Path("/nowhere"))
    if os.environ.get("RPP_CMODS") or os.environ.get("MPPP_CMODS"):
        pytest.skip("RPP_CMODS points at other camera models: the keys differ by design")
    # 0.99.22: the colors changed (the Navcam brightness comes from the configuration, navcam_brightness None), so
    # every block processes and aligns again - by design.  With the old setting the keys are 0.98.29's still.
    old = P.Settings(scapes_root=Path("/nowhere"), pds_dir=Path("/nowhere"), navcam_brightness=0.9)
    assert (P.run_key(old), P.process_key(old)) == ("078faba160de6bc4", "873a5aef563fc306")
    assert (P.run_key(s), P.process_key(s)) == ("bca04ab04c5df34d", "c583d0b67eada6ad")
    for ck, want in (("mppp_mask_v4", ("24426225031637e2", "c91cfb74c3e594c5")),
                     ("rpp_mask_v4", ("24426225031637e2", "c91cfb74c3e594c5"))):
        s = P.Settings(scapes_root=Path("/nowhere"), pds_dir=Path("/nowhere"), mask_checkpoint=ck)
        assert (P.run_key(s), P.process_key(s)) == want, ck


def test_xmp_reader_reads_old_products(tmp_path):
    from rpp.ibr import xmp
    import numpy as np
    import cv2
    p = tmp_path / "old_360_3DTD.png"
    cv2.imwrite(str(p), np.zeros((8, 4, 3), np.uint8))
    old = xmp.fields_xmp({"Scale": "1.5", "Unit": "m"}).decode().replace("rpp:", "mppp:").replace(
        "https://github.com/ctate7163/RPP/ns/1.0/", "https://github.com/ctate7163/MPPP/ns/1.0/")
    xmp.inject_png(p, old.encode())
    assert xmp.read_fields(p) == {"Scale": "1.5", "Unit": "m"}
    xmp.tag_fields(p, {"Scale": "2"})
    assert xmp.read_fields(p) == {"Scale": "2"} and b"xmlns:rpp=" in xmp.read_xmp(p)
