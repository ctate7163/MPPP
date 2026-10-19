"""The Mastcam-Z coverage guard of stage 2 (mppp.sfm.zcam_guard): group coverage, collapse, restore, database pairs."""
import json
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
