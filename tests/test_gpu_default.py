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
    s = Settings(match_mode="exhaustive")                # 0.93.9: the default is "lightglue"
    assert effective_match_mode(s, 299) == "exhaustive" and effective_match_mode(s, 347) == "prior_pairs"
    assert effective_match_mode(Settings(match_mode="exhaustive", max_exhaustive_images=0), 2713) == "exhaustive"
    assert effective_match_mode(Settings(match_mode="lightglue"), 2713) == "lightglue"
    assert run_key(Settings(max_exhaustive_images=5)) == run_key(Settings())


def test_shipped_exclusion_list():
    """22 Oct (Christian): one list of observations left out of every alignment (Mastcam-Z movies, ZCAM01*, four
    Navcam stereo pairs) - on by default, EXCLUDE_LIST=false keeps them; not in the run key."""
    from mppp.pipeline import run_key, shipped_exclusions
    from mppp.sfm.workdir import apply_exclusions
    e = shipped_exclusions()
    assert {"seq:ZCAM05179", "seq:ZCAM05180", "seq:ZCAM05114", "seq:ZCAM05093", "seq:ZCAM05062", "seq:ZCAM01*", "seq:ZCAM00*",
            "seq:ZCAM08512"} <= set(e)
    def im(stem, seq):
        return {"filename": {"stem": stem, "sequence": seq, "sol": int(stem[4:8])}}
    ims = [im("ZL0_1065_0761484524_183RAD_N0501422ZCAM05179_034050A01", "ZCAM05179"),
           im("ZR0_1058_0760874769_848RAD_N0501254ZCAM01539_034050A01", "ZCAM01539"),
           im("ZL0_1058_0760874769_848RAD_N0501254ZCAM09097_034050A01", "ZCAM09097"),
           im("NLF_0693_0728460284_145RAD_N0321184NCAM00500_0A0095J01", "NCAM00500"),
           im("NRG_0693_0728460284_145EBY_N0321184NCAM00500_0A0095J01", "NCAM00500"),
           im("NLF_0693_0728460300_145RAD_N0321184NCAM00500_0A0095J01", "NCAM00500"),
           im("ZR0_0092_0675116815_348RAD_N0040136ZCAM00033_0340LUA03", "ZCAM00033"),      # Chal Rocks: kept
           im("ZL0_0093_0675176651_519RAD_N0040136ZCAM00034_0340LUA03", "ZCAM00034"),      # kept
           im("ZL0_0100_0675800000_000RAD_N0040136ZCAM00040_0340LUA03", "ZCAM00040")]      # ZCAM00*: out
    keep, removed = apply_exclusions(ims, e)
    assert [m["filename"]["stem"][:8] for m in keep] == ["ZL0_1058", "NLF_0693", "ZR0_0092", "ZL0_0093"]
    assert keep[1]["filename"]["stem"].startswith("NLF_0693_0728460300")
    assert Settings().exclude_list and run_key(Settings(exclude_list=False)) == run_key(Settings())
    assert Settings().max_exhaustive_images == 300
