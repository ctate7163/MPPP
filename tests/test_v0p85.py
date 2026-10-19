"""v0p85 (19 Oct 2026): block folders in D:/scapes without "_colmap", the folder move, nearby visits from the
block's first sol, and the site fixes Christian asked for."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_folder_names_and_old_names():
    from mppp.sfm import sites as S
    assert S.DEFAULT_SCAPES_ROOT == "D:/scapes"
    assert S.folder_name("south_arm", 1408) == "mars2020_sol_1408_south_arm"
    assert S.folder_name("south_arm", 1408, True) == "mars2020_sol_1408_south_arm_zcam"
    for n, want in [("mars2020_sol_1408_south_arm", ("south_arm", False)),
                    ("mars2020_sol_1408_south_arm_zcam", ("south_arm", True)),
                    ("mars2020_sol_1408_south_arm_colmap", ("south_arm", False)),          # v0p53-v0p82
                    ("mars2020_sol_1408_south_arm_colmap_zcam", ("south_arm", True)),
                    ("mars2020_sol_0072_chal_rocks_large", ("chal_rocks_large", False)),
                    ("camera_analysis", (None, False)), ("mppp_pipeline", (None, False))]:
        assert S.parse_work_folder(n) == want, n


def test_scan_scapes_reads_new_and_old_names(tmp_path):
    from mppp.sfm import sites as S
    for n in ("mars2020_sol_1408_south_arm", "mars2020_sol_1408_south_arm_colmap_zcam", "camera_analysis"):
        (tmp_path / n).mkdir()
    rows = S.scan_scapes(tmp_path, {"south_arm": [[1408, 1412]]})
    assert sorted((r["site"], r["zcam"]) for r in rows) == [("south_arm", False), ("south_arm", True)]


def test_no_default_root_left_at_scapes_colmap():
    hits = []
    for p in list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "scripts").rglob("*.bat")) + \
            list((ROOT / "src").rglob("*.py")) + list((ROOT / "notebooks").glob("*.ipynb")):
        if p.name in ("move_scapes.py", "sites_table.py", "move_scapes.bat") or "__pycache__" in p.parts:
            continue
        s = p.read_text(encoding="utf-8")
        for pat in ('default="D:/scapes/colmap"', 'ROOT=D:\\scapes\\colmap"', 'Path("D:/scapes/colmap")',
                    'Path(r\\"D:\\\\scapes\\\\colmap\\")'):
            if pat in s:
                hits.append(f"{p.relative_to(ROOT)}: {pat}")
    assert not hits, hits


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
    cmd = [sys.executable, str(ROOT / "scripts" / "move_scapes.py"), "--old-root", str(old), "--new-root", str(new)]
    dry = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert "dry run" in dry and (old / "mppp_pipeline").is_dir()                        # nothing moved
    out = subprocess.run(cmd + ["--apply"], capture_output=True, text=True, check=True).stdout
    names = {p.name for p in new.iterdir()}
    assert {"mars2020_sol_0072_chal_rocks_large", "mars2020_sol_1431_bell_island_large",     # new first sols
            "mars2020_sol_1948_hippo_pools", "mars2020_sol_1408_south_arm_zcam",
            "mars2020_sol_0020_van_zyl_overlook_large", "mars2020_sol_0047_van_zyl_overlook_large",
            "camera_analysis", "mppp_pipeline", "error_analysis"} <= names, names
    assert (new / "error_analysis" / "2026-10-05").is_dir()                              # merged
    assert (old / "error_analysis" / "2026-10-01").is_dir() and "CONFLICT" in out        # not overwritten
    p = json.loads((new / "mars2020_sol_1431_bell_island_large" / "colmap" / "project.json").read_text())
    assert p["settings"]["processed_dir"] == str(new / "mars2020_sol_1431_bell_island_large" / "processed")
    assert (new / "mppp_pipeline" / "move_scapes_v0p85.json").is_file()
    again = subprocess.run(cmd + ["--apply"], capture_output=True, text=True, check=True).stdout
    assert "0 moved" in again


def _wp(site, drive, sol, e, n):
    return {"type": "Feature", "properties": {"site": site, "drive": drive, "sol": sol, "easting": e, "northing": n}}


def test_find_imgs_near_min_sol(tmp_path):
    from mppp.select import find_imgs_near
    wps = {"features": [_wp(9, 1000, 178, 0.0, 0.0), _wp(9, 900, 175, 5.0, 0.0), _wp(9, 1914, 341, 0.0, 8.0)]}
    for sol, site, drive in [(178, 9, 1000), (175, 9, 900), (177, 9, 900), (341, 9, 1914)]:
        (tmp_path / f"NLF_{sol:04d}_0700000000_000RAD_N{site:03d}{drive:04d}NCAM00100_0A0095J01.IMG").write_bytes(b"")
    p, r = find_imgs_near(tmp_path, ["NLF"], [[178, 190]], wps, radius_m=10.0)
    assert sorted(int(x.name[4:8]) for x in p) == [175, 177, 178, 341]
    p, r = find_imgs_near(tmp_path, ["NLF"], [[178, 190]], wps, radius_m=10.0, min_sol=178)
    assert sorted(int(x.name[4:8]) for x in p) == [178, 341] and r["min_sol"] == 178


def test_site_fixes_19_oct():
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))["sites"]
    assert t["chal_rocks_large"]["sols"][0] == [72, 100]
    assert t["chal_rocks_large"]["settings"]["NEARBY_FROM_FIRST_SOL"] is True
    assert [421, 423] in t["enchanted_lake"]["sols"]
    assert t["bell_island_large"]["sols"][0][0] == 1431
    assert t["rochette"]["sols"][0][0] == 178 and t["rochette"]["settings"]["NEARBY_FROM_FIRST_SOL"] is True
    assert t["hippo_pools"]["sols"][0][0] == 1948 and t["hippo_pools"]["settings"]["NEARBY_FROM_FIRST_SOL"] is True
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "NEARBY_FROM_FIRST_SOL = False" in src and "min_sol=_NEAR_MIN" in src
    bat = (ROOT / "scripts" / "windows" / "run_v0p85.bat").read_text()
    assert "move_scapes.py\" --apply" in bat
