"""v0p70: fx, fy, cx, cy offsets of the early-mission Navcam images (before sol 380) in the consensus, with their
significance, written into the camera files and applied by the projects."""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
P = np.array([2956.7, 2956.75, 2591.6, 1944.3, 0.0451, -0.0100, 2.5e-4, 3.6e-4, 0.0030, 0.0, 0.0, 0.0])


def test_project_applies_the_early_mission_offsets(tmp_path):
    from conftest import NLF
    if not NLF.is_file():
        pytest.skip("example IMGs not present")
    import mppp
    from mppp.calib import load_camera
    from mppp.calib.models import CAMERA_FILE
    from mppp.sfm.project import SfmProject
    out = tmp_path / "proc"
    cfg = mppp.load_config({"masking": {"infer_mask": False}, "export": {"formats": ["PNG8"], "write_colmap": False}})
    man = mppp.process_images([NLF], out, cfg, mppp.load_waypoints(), progress=False)
    (tmp_path / "joint").mkdir()
    for eye in "LR":
        js = load_camera("N", eye)
        js.pop("file", None)
        js["params"] = [float(v) for v in P]
        js["thermal"] = {k: 0.0 for k in ("fx_ppm_per_degC", "fy_ppm_per_degC", "cx_px_per_degC", "cy_px_per_degC",
                                          "k1_per_degC")} | {"T0_degC": -19.1}
        js["early_mission"] = {"before_sol": 1e6, "dfx_px": 0.3, "dfy_px": 0.0, "dcx_px": -2.7, "dcy_px": 0.5,
                               "applied_terms": ["cx", "cy"]}
        (tmp_path / "joint" / CAMERA_FILE[("N", eye)]).write_text(json.dumps(js))
    proj = SfmProject.create(man["images"], out, tmp_path / "p", link=False, navcam_distortion="fisheye_tangential",
                             navcam_cameras=tmp_path / "joint")
    p = proj.cameras["NL"]["params"]
    assert p[0] == pytest.approx(P[0], abs=0.05) and p[2] == pytest.approx(P[2] - 2.7, abs=0.05)
    assert p[3] == pytest.approx(P[3] + 0.5, abs=0.05) and "early-mission" in proj.cameras["NL"]["source"]


def test_notebooks_carry_the_version_guard():
    import glob
    import mppp
    for p in sorted(glob.glob(str(ROOT / "notebooks" / "0*.ipynb"))):
        nb = json.loads(Path(p).read_text(encoding="utf-8"))
        src = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        if "import mppp\n" in src:
            assert f'NOTEBOOK_FOR_MPPP = "{mppp.VERSION_TAG}"' in src, p
