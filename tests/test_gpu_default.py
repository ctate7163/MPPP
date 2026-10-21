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
