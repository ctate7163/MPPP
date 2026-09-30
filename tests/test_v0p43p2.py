"""MPPP v0p43.2: the Windows .bat files find the notebooks' Python and work from any folder (tests in test_v0p43p1)."""


def test_version():
    import mppp
    assert tuple(int(x) for x in mppp.__version__.split(".")[:3]) >= (0, 43, 2)


def test_navcam_consensus_shipped_with_mppp():
    import json
    from mppp.sfm.project import (NAVCAM_PACKAGE_CONSENSUS_DIR as NAVCAM_CONSENSUS_DIR, NAVCAM_FISHEYE_PATTERN, NAVCAM_RIG_FILE,
                                  camera_from_colmap_json, navcam_cameras_fingerprint)
    for eye in ("NL", "NR"):
        f = NAVCAM_CONSENSUS_DIR / NAVCAM_FISHEYE_PATTERN.format(instrument=eye)
        cam = camera_from_colmap_json(f)
        assert cam["model"] == "THIN_PRISM_FISHEYE" and len(cam["params"]) == 12
        th = json.loads(f.read_text())["thermal"]
        assert abs(th["ppm_per_degC"] - 38.1) < 0.5 and th["T0_degC"] == -20
    assert (NAVCAM_CONSENSUS_DIR / NAVCAM_RIG_FILE).is_file()
    assert navcam_cameras_fingerprint(NAVCAM_CONSENSUS_DIR)
    # the v0p41 joint (camera_analysis/navcal_v0p41/navcam_joint), byte for byte
    import hashlib
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in NAVCAM_CONSENSUS_DIR.glob("*.json")}
    assert sha == {"M2020_NL_fisheye_tangential.json": "15389f18ab97", "M2020_NR_fisheye_tangential.json": "8c114d80b388",
                   "M2020_N_rig.json": "949b9c26aca7"}


def test_notebook03_defaults_to_the_shipped_consensus():
    import json
    from pathlib import Path
    nb = json.loads((Path(__file__).resolve().parents[1] / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    params = "".join(nb["cells"][3]["source"])
    assert "NAVCAM_CAMERAS    = NAVCAM_CONSENSUS_DIR" in params and "camera_analysis" not in params
    derived = "".join(nb["cells"][4]["source"])
    assert "no consensus Navcam cameras there" in derived
