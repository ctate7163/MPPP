"""0.91: the batch runs find the GPU environment like the notebook does (Chal Rocks Z34, 20 Oct: SIFT on the CPU in the
run's own process, killed by a COLMAP abort)."""
from mppp.pipeline import Settings


def test_gpu_python_from_the_pointer_file(tmp_path, monkeypatch):
    py = tmp_path / "envs" / "mppp_gpu" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_text("")
    (tmp_path / "mppp").mkdir()
    (tmp_path / "mppp" / "gpu_python.txt").write_text(f'"{py}"\n', encoding="utf-8")
    for k in ("MPPP_GPU_PYTHON", "MPPP_GPU_PY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert Settings().gpu_python == str(py)
    monkeypatch.setenv("MPPP_GPU_PYTHON", "C:/other/python.exe")         # the explicit setting still wins
    assert Settings().gpu_python == "C:/other/python.exe"


def test_no_gpu_python(tmp_path, monkeypatch):
    import sys
    (tmp_path / "python.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    for k in ("MPPP_GPU_PYTHON", "MPPP_GPU_PY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert Settings().gpu_python is None



def test_gpu_python_from_the_conda_envs_folder(tmp_path, monkeypatch):
    """The old notebook's fallback: <conda>/envs/mppp_gpu next to the running python."""
    import sys
    from mppp.sfm import gpu
    base = tmp_path / "miniconda3"
    other = base / "envs" / "mppp_gpu" / "python.exe"
    other.parent.mkdir(parents=True)
    other.write_text("")
    (base / "python.exe").write_text("")
    monkeypatch.setattr(sys, "executable", str(base / "python.exe"))
    for k in ("MPPP_GPU_PYTHON", "MPPP_GPU_PY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert gpu.conda_env_python() == str(other)
    assert Settings().gpu_python == str(other)


def test_big_blocks_match_prior_pairs_and_keep_their_run_key():
    """21 Oct: exhaustive matching over more than max_exhaustive_images images matches the prior-overlap pairs; the
    threshold is not in the run key (finished blocks stay finished)."""
    from mppp.pipeline import effective_match_mode, run_key
    s = Settings()
    assert effective_match_mode(s, 347) == "exhaustive" and effective_match_mode(s, 2713) == "prior_pairs"
    assert effective_match_mode(Settings(max_exhaustive_images=0), 2713) == "exhaustive"
    assert effective_match_mode(Settings(match_mode="lightglue"), 2713) == "lightglue"
    assert run_key(Settings(max_exhaustive_images=5)) == run_key(Settings())
