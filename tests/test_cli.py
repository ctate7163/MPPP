"""The mppp command (mppp.cli): argument parsing, the bookkeeping subcommands, a dry processing run, refusals."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import NLF, ZL0

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("MPPP_PDS", "MPPP_SCAPES", "MPPP_GPU_PYTHON", "COLMAP_BAT"):
        monkeypatch.delenv(k, raising=False)


def test_parser_and_entry_point():
    from mppp.cli import build_parser, main
    ap = build_parser()
    a = ap.parse_args(["-v", "--log", "x.txt", "align", "/w", "--variant", "v", "--set", "A=1", "--set", "B=2",
                       "--gui", "--stop-after", "match", "--force"])
    assert a.cmd == "align" and a.work == "/w" and a.variant == "v" and a.set == ["A=1", "B=2"]
    assert a.gui and a.stop_after == "match" and a.force and a.verbose and a.log == "x.txt"
    a = ap.parse_args(["process", "rockytop", "--root", "/r", "--pds", "/p", "--zcam", "--dry-run"])
    assert a.cmd == "process" and a.target == "rockytop" and a.root == "/r" and a.pds == "/p" and a.zcam and a.dry_run
    a = ap.parse_args(["run", "rockytop", "--settings", "s.json"])
    assert a.cmd == "run" and a.settings == "s.json" and a.variant == ""
    assert ap.parse_args(["sites"]).what == "list" and ap.parse_args(["sites", "folder", "sid"]).site == "sid"
    assert ap.parse_args(["cmods", "promote", "a", "b", "--note", "n"]).paths == ["a", "b"]
    assert ap.parse_args(["mask", "install", "f.pt", "--name", "m"]).arg == "f.pt"
    with pytest.raises(SystemExit):
        ap.parse_args(["align"])
    assert main([]) == 2
    assert 'mppp = "mppp.cli:main"' in (ROOT / "pyproject.toml").read_text()


def test_version(capsys):
    import mppp
    from mppp.cli import main
    assert main(["version"]) == 0
    assert f"mppp {mppp.__version__}" in capsys.readouterr().out
    r = subprocess.run([sys.executable, "-m", "mppp.cli", "version"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and mppp.__version__ in r.stdout


def test_sites(capsys, tmp_path):
    from mppp.cli import main
    assert main(["sites", "list"]) == 0
    out = capsys.readouterr().out
    assert "rockytop" in out and "sols 460-530" in out and "groups:" in out
    assert main(["sites", "check"]) == 0
    assert "0 error(s)" in capsys.readouterr().out
    assert main(["sites", "folder", "rockytop", "--root", str(tmp_path), "--zcam"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "mars2020_sol_0460_rockytop_zcam")
    assert main(["sites", "folder", "nowhere"]) == 2
    assert main(["sites", "folder"]) == 2


def test_cmods_list(capsys):
    from mppp.cli import main
    assert main(["cmods", "list"]) == 0
    out = capsys.readouterr().out
    assert "camera models in use" in out and "M2020_NL_fisheye_tangential.json" in out and "ERROR" not in out
    assert main(["cmods", "promote", "/nonexistent/folder"]) == 1


def test_mask_list(capsys):
    from mppp.cli import main
    assert main(["mask", "list"]) == 0
    assert "(default)" in capsys.readouterr().out
    assert main(["mask", "install"]) == 2


def test_process_dry_run_and_run(tmp_path, capsys, clean_env):
    from mppp.cli import main
    pds = tmp_path / "pds" / "sol0709"
    pds.mkdir(parents=True)
    for f in (NLF, ZL0):
        shutil.copy(f, pds / f.name)
    extra = json.dumps({"sol709": [709, 709]})
    settings = tmp_path / "settings.json"
    # the Navcam test product is a Sun-pointing SAPP tile: the selection rules are off for it
    settings.write_text(json.dumps({"sequence_prefixes": ["NCAM", "SAPP"], "navcam_min_frame_fraction": None,
                                    "sky_elevation_deg": None, "lmst_window_h": None, "max_saturated_fraction": None,
                                    "max_exposure_ms": None, "max_centre_tint": None, "infer_mask": False,
                                    "workers": 1, "add_nearby_waypoints": 0}))
    args = ["process", "sol709", "--root", str(tmp_path / "scapes"), "--pds", str(tmp_path / "pds"), "--zcam",
            "--settings", str(settings), "--set", f"sites_extra={extra}"]
    assert main(args + ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 products would be processed" in out and NLF.name in out and ZL0.name in out
    work = tmp_path / "scapes" / "mars2020_sol_0709_sol709_zcam"
    assert not work.exists()
    assert main(args + ["--no-echo"]) == 0
    assert (work / "processed" / "process_done.json").is_file() and (work / "mppp_status.json").is_file()
    st = json.loads((work / "mppp_status.json").read_text())
    assert st["state"] == "finished" and st["stage"] == "process" and st["site"] == "sol709"
    assert list((work / "runs").glob("*_process/log.txt"))
    assert main(["process", "nowhere", "--root", str(tmp_path / "scapes"), "--pds", str(tmp_path / "pds")]) == 2
    assert "not in" in capsys.readouterr().err


def test_align_refuses_a_folder_without_processed(tmp_path, capsys, clean_env):
    from mppp.cli import main
    work = tmp_path / "mars2020_sol_0460_rockytop"
    work.mkdir()
    assert main(["align", str(work)]) == 2
    assert "no processed/ folder" in capsys.readouterr().err
    assert main(["align", str(work), "--status"]) == 1
    (work / "processed").mkdir()
    assert main(["align", str(work), "--stop-after", "nowhere"]) == 2
    assert main(["run", str(work), "--stop-after", "nowhere"]) == 2
    # a folder with processed/ but no manifest: the run starts, is logged and fails cleanly
    assert main(["align", str(work), "--no-echo"]) == 1
    err = capsys.readouterr().err
    assert "FAILED" in err and "mppp_manifest" in err
    st = json.loads((work / "mppp_status.json").read_text())
    assert st["state"] == "failed" and st["stage"] == "align" and "mppp_manifest" in st["last_error"]
    assert main(["align", str(work), "--status"]) == 0
    assert '"state": "failed"' in capsys.readouterr().out


def test_align_scape_script_wraps_the_cli(tmp_path, capsys, clean_env):
    import importlib.util
    spec = importlib.util.spec_from_file_location("align_scape", ROOT / "scripts" / "align_scape.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    work = tmp_path / "mars2020_sol_0460_rockytop"
    work.mkdir()
    assert m.main([str(work), "--is-running"]) == 1
    assert m.main([str(work), "--status"]) == 1
    assert m.main([str(work)]) == 2                       # no processed/
