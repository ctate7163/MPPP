"""Mastcam-Z frames against Navcam (20 Oct): the valid Navcam tie points per Mastcam-Z image (a third look on each,
at least 10 per frame) and the Mastcam-Z points above / below the Navcam terrain (an alignment proxy)."""
from types import SimpleNamespace as NS

import numpy as np


class _Track:
    def __init__(self, ids):
        self.elements = [NS(image_id=i) for i in ids]

    def length(self):
        return len(self.elements)


class _Image:
    def __init__(self, name, C):
        self.name, self._C, self.has_pose = name, np.asarray(C, float), True

    def projection_center(self):
        return self._C


class _Rec:
    def __init__(self, images, points):
        self.images = images
        self.points3D = {k: NS(xyz=np.asarray(x, float), track=_Track(ids)) for k, (x, ids) in enumerate(points)}

    def reg_image_ids(self):
        return list(self.images)


def _rec(z_offset=0.0, n_zz=0):
    rng = np.random.default_rng(0)
    images = {1: _Image("NLF_a", [0, 0, 1.9]), 2: _Image("NRF_a", [0.42, 0, 1.9]), 3: _Image("NLF_b", [3, 0, 1.9]),
              10: _Image("ZL0_a", [0.2, 0, 2.0]), 11: _Image("ZR0_a", [0.4, 0, 2.0]), 12: _Image("ZL0_b", [0.2, 0.1, 2.0])}
    pts = []
    for _ in range(3000):                                   # Navcam terrain: flat ground at z = 0 over 10 x 10 m
        x, y = rng.uniform(-5, 5, 2)
        pts.append(([x, y, rng.normal(0, 0.01)], [1, 2, 3]))
    for _ in range(40):                                      # ZL a: valid ties (a third look), on the ground
        x, y = rng.uniform(-4, 4, 2)
        pts.append(([x, y, z_offset], [10, 11, 1]))
    for _ in range(5):                                       # ZL b: only five ties with Navcam ...
        x, y = rng.uniform(-4, 4, 2)
        pts.append(([x, y, 0.0], [12, 10, 2]))
    for _ in range(30):                                      # ... and two-view N-Z points (no third look: not valid)
        x, y = rng.uniform(-4, 4, 2)
        pts.append(([x, y, 0.0], [12, 3]))
    for _ in range(n_zz):
        pts.append(([0, 0, 0.0], [10, 11]))
    return _Rec(images, pts)


def test_navcam_tie_counts_need_a_third_look():
    from mppp.sfm.reconstruction import ZCAM_MIN_NAVCAM_TIES, navcam_tie_counts
    c = navcam_tie_counts(_rec())
    assert ZCAM_MIN_NAVCAM_TIES == 10
    assert c[10] == 45 and c[11] == 40 and c[12] == 5          # the two-view N-Z points do not count
    assert 1 not in c                                           # Navcam images are not counted


def test_zcam_points_above_and_below_the_navcam_terrain():
    from mppp.sfm.health import zcam_terrain, zcam_terrain_line
    on = zcam_terrain(_rec())
    assert on["zcam"]["below_fraction"] == 0 and on["zcam"]["above_fraction"] == 0 and on["zcam"]["on_grid"] > 0
    low = zcam_terrain(_rec(z_offset=-0.6))                    # a frame aligned 0.6 m low: its ties below the ground
    assert low["navcam_zcam"]["below"] >= 40 and low["navcam_zcam"]["above"] == 0
    assert low["zcam"]["median_dz_m"] < 0
    assert "below" in zcam_terrain_line(low)
    assert zcam_terrain(_Rec({1: _Image("NLF_a", [0, 0, 1.9])}, [([0, 0, 0], [1])])).get("note")
