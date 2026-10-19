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


def test_one_third_rule_is_about_subframes():
    """Christian (19 Oct): the 1/3 rule leaves out sub-frames covering less than 1/3 of the frame - not images whose
    mask keeps less than 1/3 (v0p82's reading, now off)."""
    import pytest
    from mppp.runner import ALIGN_RULES
    from mppp.sfm.project import NAVCAM_MIN_FRAME_FRACTION
    from mppp.sfm.workdir import MIN_INCLUDED_FRACTION
    assert NAVCAM_MIN_FRAME_FRACTION == pytest.approx(1 / 3) and MIN_INCLUDED_FRACTION is None and ALIGN_RULES >= 85
    nb = json.loads((ROOT / "notebooks" / "03_colmap_alignment.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "MIN_INCLUDED_FRACTION  = None" in src and 'SUBFRAME_FAMILIES = ("N",)' in src
    assert "frame_fraction_families=SUBFRAME_FAMILIES" in src


def test_pipeline_stop_before(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--root", str(tmp_path),
                          "--dry-run", "--from", "navcam_runs", "--stop-before", "zcam_start", "--no-baseline-study"],
                         capture_output=True, text=True, timeout=120)
    log = out.stdout + out.stderr
    assert "stopped before zcam_start" in log, log[-2000:]
    bat = (ROOT / "scripts" / "windows" / "run_v0p85.bat").read_bytes()
    assert b"--stop-before zcam_start" in bat and b"\r\n" in bat


class _Im:
    def __init__(self, n, posed=True):
        self.num_points3D, self.has_pose = n, posed


class _Cam:
    def __init__(self, p):
        self.params = p


class _Rec:
    def __init__(self, ims, cams):
        self.images, self.cameras = ims, cams


class _Proj:
    def __init__(self, images, cams):
        self.images = images
        self.settings = {"database": {"cameras": cams}}


def test_zcam_guard_coverage_collapse_and_restore():
    import numpy as np
    from mppp.sfm import zcam_guard as ZG
    imgs = [{"image_id": i, "instrument": "ZL034_F01092", "camera_group": "ZL034"} for i in range(1, 7)] + \
           [{"image_id": i, "instrument": "ZR034_F01128", "camera_group": "ZR034"} for i in range(7, 13)] + \
           [{"image_id": 13, "instrument": "NL"}]
    proj = _Proj(imgs, {"ZL034_F01092": 5, "ZR034_F01128": 6, "NL": 1})
    cams = {5: _Cam([4700.0, 4710.0, 824.0, 600.0]), 6: _Cam([4729.0, 4742.0, 743.15, 663.96]), 1: _Cam([1.0])}
    rec = _Rec({i: _Im(500) for i in range(1, 14)}, cams)
    before = ZG.group_coverage(rec, proj)
    assert before == {"ZL034": {"frames": 6, "with_points": 6, "observations": 3000},
                      "ZR034": {"frames": 6, "with_points": 6, "observations": 3000}}
    snap = ZG.snapshot(rec, proj)
    for i in range(7, 13):
        rec.images[i] = _Im(3)                     # the right eye loses its tie points
    cams[6].params = np.array([4729.0, 4742.0, 743.15, 680.0])
    after = ZG.group_coverage(rec, proj)
    assert ZG.collapsed(before, after) == ["ZR034"] and ZG.brief(after) == "ZL034 6/6, ZR034 0/6"
    assert ZG.camera_ids_of(proj, ["ZR034"]) == [6]
    assert ZG.restore(rec, snap, [6]) == 1 and cams[6].params[3] == 663.96


def test_stage2_uses_the_zcam_guard():
    src = (ROOT / "src" / "mppp" / "sfm" / "reconstruction.py").read_text(encoding="utf-8")
    for k in ("ZG.collapsed(cov_prev, cov)", "line_snap = ZG.snapshot(rec, project)", "zbad = ZG.collapsed(zcov0, zcov1)",
              '"zcam_coverage": ZG.group_coverage(rec, project)'):
        assert k in src, k


def test_zcam_guard_db_pairs(tmp_path):
    import sqlite3
    from mppp.sfm import zcam_guard as ZG
    db = tmp_path / "d.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER)")
    K = 2147483647
    con.executemany("INSERT INTO two_view_geometries VALUES (?, ?)", [(1 * K + 7, 120), (7 * K + 9, 5), (2 * K + 9, 40)])
    con.commit()
    con.close()
    assert ZG._db_pairs(db, [7, 9, 11]) == {7: 1, 9: 1, 11: 0}


def test_zr_trace_and_report(tmp_path):
    """zr_report finds the step where an eye loses its frames, in the last run of the trace."""
    def head(step, zr):
        return {"head": {"step": step, "coverage": {"ZL034": {"frames": 8, "with_points": 8, "observations": 1},
                                                     "ZR034": {"frames": 8, "with_points": zr, "observations": 1}},
                         "points": 100}}

    def row(step, n, obs):
        return {"step": step, "name": n, "group": n[:2] + "034", "registered": True, "obs": obs, "keypoints": 50,
                "frame_id": 1, "rig_id": 1, "camera_id": 1, "cam": [1, 1, 1, 1, 0, 0], "C": [0, 0, 0],
                "q": [0, 0, 0, 1]}
    proj = tmp_path / "mars2020_sol_1408_south_arm_zcam" / "colmap_zrdiag"
    proj.mkdir(parents=True)
    lines = []
    for run in range(2):
        for step, zr in (("stage2_start", 0), ("r1_tri", 8), ("r2_ba", 8), ("r3_tri", 0 if run else 8)):
            lines.append(head(step, zr))
            lines += [row(step, f"ZL{i}", 100) for i in range(8)]
            lines += [row(step, f"ZR{i}", 100 if zr else 0) for i in range(8)]
    (proj / "zr_trace.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    out = tmp_path / "r.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "zr_report.py"), str(proj), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    t = out.read_text()
    assert "**ZR lost its frames at `r3_tri`** (after `r2_ba`: 8 -> 0)" in t and t.count("| r3_tri |") == 1
    bat = (ROOT / "scripts" / "windows" / "zr_diagnose_z34.bat").read_bytes()
    assert b"MPPP_ZCAM_TRACE=2" in bat and b"--sites rochette south_arm --zcam" in bat and b"\r\n" in bat


def test_sites_19_oct_afternoon():
    """Christian (19 Oct): notes last; Issole 201 + 286-339; Seitah North Large in the Navcam consensus and the
    baseline; seven more consensus blocks; Seitah Overlook 113-125; Butler Landing 1-14 + 32-44; Butler Landing
    Large gone; Van Zyl Overlook Huge 15-99 + 361-378 replaces Van Zyl Overlook Large."""
    t = json.loads((ROOT / "src" / "mppp" / "data" / "sites.json").read_text(encoding="utf-8"))
    S, G = t["sites"], t["groups"]
    assert all(list(v)[-1] == "notes" for v in S.values() if "notes" in v)
    assert S["issole_seitah_south"]["sols"] == [[201, 201], [286, 339]]
    assert S["seitah_north_large"]["sols"] == [[202, 284]]
    assert S["artuby_ridge"]["sols"] == [[342, 349], [174, 176]]
    assert S["seitah_overlook"]["sols"] == [[113, 125]] and S["butler_landing"]["sols"] == [[1, 14], [32, 44]]
    assert "butler_landing_large" not in S and "van_zyl_overlook_large" not in S
    assert S["van_zyl_overlook_huge"]["sols"] == [[15, 99], [361, 378]]
    nc = set(G["navcam_consensus"])
    assert {"seitah_north_large", "marble_mountain", "hippo_pools", "cheyava_falls", "bunsen_peak", "tuxedo_park",
            "belva_crater_large", "oeb_landing_site"} <= nc and len(nc) == 17
    assert "seitah_north_large" in G["navcam_baseline"] and "van_zyl_overlook_huge" in G["navcam_baseline"]
    assert not any(s in ("butler_landing_large", "van_zyl_overlook_large") for g in G.values() for s in g)


def test_move_scapes_renames_in_the_new_root(tmp_path):
    new = tmp_path / "scapes"
    (new / "mars2020_sol_0285_issole_seitah_south" / "colmap").mkdir(parents=True)
    (new / "mars2020_sol_0178_rochette_zcam").mkdir()
    cmd = [sys.executable, str(ROOT / "scripts" / "move_scapes.py"), "--old-root", str(new / "colmap"),
           "--new-root", str(new), "--apply"]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    names = {p.name for p in new.iterdir()}
    assert "mars2020_sol_0201_issole_seitah_south" in names and "mars2020_sol_0178_rochette_zcam" in names
