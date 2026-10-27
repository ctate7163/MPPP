"""The one-off folder move of v0p85 (studies/calibration/move_scapes.py): block folders without "_colmap", renamed to
their first sol; moved from the versioned tests when the script left scripts/."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MOVE = HERE / "move_scapes.py"
def test_move_scapes_plan_and_apply(tmp_path):
    new = tmp_path / "scapes"
    old = new / "colmap"
    for n in ("mars2020_sol_0072_chal_rocks_large_colmap", "mars2020_sol_1435_bell_island_large_colmap",
              "mars2020_sol_1947_hippo_pools_colmap", "mars2020_sol_1408_south_arm_colmap_zcam",
              "mars2020_sol_0020_van_zyl_overlook_large_colmap", "mars2020_sol_0047_van_zyl_overlook_large_colmap",
              "camera_analysis/navcal_x", "mppp_pipeline", "error_analysis/2026-10-05", "error_analysis/2026-10-01"):
        (old / n).mkdir(parents=True)
    (new / "error_analysis" / "2026-10-01").mkdir(parents=True)
    pj = old / "mars2020_sol_1435_bell_island_large_colmap" / "colmap"
    pj.mkdir()
    (pj / "project.json").write_text(json.dumps({"settings": {"processed_dir": str(pj.parent / "processed")}}))
    cmd = [sys.executable, str(MOVE), "--old-root", str(old), "--new-root", str(new)]
    dry = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert "dry run" in dry and (old / "mppp_pipeline").is_dir()                        # nothing moved
    out = subprocess.run(cmd + ["--apply"], capture_output=True, text=True, check=True).stdout
    names = {p.name for p in new.iterdir()}
    assert {"mars2020_sol_0072_chal_rocks_large", "mars2020_sol_1431_bell_island_large",     # new first sols
            "mars2020_sol_1948_hippo_pools", "mars2020_sol_1359_south_arm_zcam",     # 27 Oct: the first sol of all
            "mars2020_sol_0020_van_zyl_overlook_large", "mars2020_sol_0047_van_zyl_overlook_large",
            "camera_analysis", "mppp_pipeline", "error_analysis"} <= names, names
    assert (new / "error_analysis" / "2026-10-05").is_dir()                              # merged
    assert (old / "error_analysis" / "2026-10-01").is_dir() and "CONFLICT" in out        # not overwritten
    p = json.loads((new / "mars2020_sol_1431_bell_island_large" / "colmap" / "project.json").read_text())
    assert p["settings"]["processed_dir"] == str(new / "mars2020_sol_1431_bell_island_large" / "processed")
    assert (new / "mppp_pipeline" / "move_scapes_v0p85.json").is_file()
    again = subprocess.run(cmd + ["--apply"], capture_output=True, text=True, check=True).stdout
    assert "0 moved" in again


def test_move_scapes_renames_in_the_new_root(tmp_path):
    new = tmp_path / "scapes"
    (new / "mars2020_sol_0285_issole_seitah_south" / "colmap").mkdir(parents=True)
    (new / "mars2020_sol_0178_rochette_zcam").mkdir()
    cmd = [sys.executable, str(MOVE), "--old-root", str(new / "colmap"),
           "--new-root", str(new), "--apply"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    names = {p.name for p in new.iterdir()}
    assert "mars2020_sol_0201_issole_seitah_south" in names and "mars2020_sol_0178_rochette_zcam" in names
