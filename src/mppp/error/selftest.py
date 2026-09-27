"""
mppp.selftest -- analytic validation of the physics engine.

Every test compares the vectorised implementation against either a closed-form
result or an independently written naive reference.  Run with:

    python -m mppp_error selftest

A failure here means the physics is wrong, not that a plot looks odd.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

from .core import (Grid, FlatPlane, Instrument, Station, ModelConfig, PoseModel,
                   OcclusionMask, solve_precision_field, skew, outer, sym_inv3)
from .metrics import compute_metrics

__all__ = ["run_all"]

_RESULTS: List[Tuple[str, bool, str]] = []


def _check(name, ok, msg=""):
    _RESULTS.append((name, bool(ok), msg))
    return bool(ok)


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def _plain_instrument(ifov=2.17e-4, b=0.244, eps=0.5, eyes=(-0.5, 0.5)):
    return Instrument(name="test", ifov_rad=ifov, baseline_m=b,
                      eps_intra_px=eps, eye_offsets=eyes)


def _bare_cfg(**kw):
    d = dict(apply_occlusion=False, cos_e_min=0.0, use_correlation=False)
    d.update(kw)
    return ModelConfig(**d)


# --------------------------------------------------------------------------
# 1. two-ray analytic: transverse = eps psi r / sqrt2, range = sqrt2 eps psi r^2/b
# --------------------------------------------------------------------------

def test_two_ray_analytic():
    ifov, b, eps = 2.0e-4, 0.25, 0.5
    inst = _plain_instrument(ifov, b, eps)
    # station directly above the origin so the LOS is vertical (nadir)
    h = 50.0
    st = Station(xyz=[0, 0, h], az_deg=0.0, instrument=inst, mask=None, is_anchor=True)
    g = Grid(x=np.array([0.0]), y=np.array([0.0]), surface=FlatPlane(0.0))
    f = solve_precision_field(g, [st], _bare_cfg())

    S = f.Sigma[0, 0]
    ev, V = np.linalg.eigh(S)
    sig_small = np.sqrt(ev[0])          # transverse (twice degenerate)
    sig_mid = np.sqrt(ev[1])
    sig_big = np.sqrt(ev[2])            # along the line of sight

    r = h
    exp_t = eps * ifov * r / np.sqrt(2.0)
    exp_r = np.sqrt(2.0) * eps * ifov * r * (r / b)

    ok_t = _rel(sig_small, exp_t) < 2e-3 and _rel(sig_mid, exp_t) < 2e-3
    ok_r = _rel(sig_big, exp_r) < 2e-3
    _check("two-ray transverse = eps*psi*r/sqrt2", ok_t,
           f"got {sig_small:.6g},{sig_mid:.6g} expected {exp_t:.6g}")
    _check("two-ray range = sqrt2*eps*psi*r^2/b", ok_r,
           f"got {sig_big:.6g} expected {exp_r:.6g}")

    # the long axis must lie along the line of sight (vertical here)
    los = abs(V[:, -1] @ np.array([0, 0, 1.0]))
    _check("long axis parallel to LOS", los > 0.999, f"|cos| = {los:.6f}")

    # equivalence with the textbook disparity form: sigma_disp = sqrt2 * eps
    text_book = (r ** 2) * ifov * (np.sqrt(2.0) * eps) / b
    _check("agrees with sigma_Z = r^2 psi sigma_disp / b",
           _rel(sig_big, text_book) < 2e-3,
           f"{sig_big:.6g} vs {text_book:.6g}")


# --------------------------------------------------------------------------
# 2. monocular station is rank-2 (no intra-station range)
# --------------------------------------------------------------------------

def test_monocular_rank_deficient():
    inst = _plain_instrument(eyes=(0.0,))
    st = Station(xyz=[0, 0, 30.0], az_deg=0.0, instrument=inst, mask=None, is_anchor=True)
    g = Grid(x=np.array([0.0]), y=np.array([0.0]))
    cfg = _bare_cfg(prior_sigma_m=1.0e4)
    f = solve_precision_field(g, [st], cfg)
    ev = np.linalg.eigvalsh(f.Sigma[0, 0])
    sig_big = np.sqrt(ev[2])
    # the range direction should be limited only by the prior
    _check("monocular station unconstrained along ray",
           sig_big > 0.5 * cfg.prior_sigma_m,
           f"sigma_max = {sig_big:.4g}, prior = {cfg.prior_sigma_m:.4g}")
    _check("monocular transverse still constrained",
           np.sqrt(ev[0]) < 1e-2, f"sigma_min = {np.sqrt(ev[0]):.4g}")


# --------------------------------------------------------------------------
# 3. precision addition: N co-located stations -> 1/sqrt(N)
# --------------------------------------------------------------------------

def test_precision_addition():
    inst = _plain_instrument()
    g = Grid(x=np.array([20.0]), y=np.array([0.0]))
    cfg = _bare_cfg()
    sig = []
    for n in (1, 4, 9):
        sts = [Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst,
                       mask=None, is_anchor=True) for _ in range(n)]
        f = solve_precision_field(g, sts, cfg)
        m = compute_metrics(f)
        sig.append(float(m["sigma_n"][0, 0]))
    ok = (_rel(sig[0] / sig[1], 2.0) < 1e-6) and (_rel(sig[0] / sig[2], 3.0) < 1e-6)
    _check("N co-located stations give 1/sqrt(N)", ok,
           f"ratios {sig[0]/sig[1]:.6f}, {sig[0]/sig[2]:.6f} (want 2, 3)")


# --------------------------------------------------------------------------
# 4. correlation floor: sigma_fused -> sqrt(rho_inf) * sigma_single
# --------------------------------------------------------------------------

def test_correlation_floor():
    """
    The rho_inf floor is the asymptote at LARGE angular separation.  At
    theta = 0 the model correctly gives rho = 1 (identical observations, zero
    information gain), so the floor must be probed with well-separated views.
    """
    from .core import _correlation_inflation

    # -- unit test the participation ratio against k_eff = N/(1+(N-1)rho) ----
    for rho_inf in (0.0, 0.05, 0.25):
        for n in (2, 8, 64):
            # n directions spread over a hemisphere so all pairwise angles are
            # large compared with theta_c -> rho_ij -> rho_inf
            ang = np.linspace(0, np.pi, n, endpoint=False)
            u = np.zeros((n, 1, 1, 3))
            u[:, 0, 0, 0] = np.cos(ang)
            u[:, 0, 0, 1] = np.sin(ang)
            vis = np.ones((n, 1, 1), dtype=bool)
            cfg = _bare_cfg(use_correlation=True, correlation_kernel="cosine", rho_inf=rho_inf,
                            theta_c_deg=0.05, store_station_dirs=True)
            infl = float(_correlation_inflation(u, vis, cfg)[0, 0])
            k_eff_expected = n / (1.0 + (n - 1) * rho_inf)
            _check(f"N_eff formula n={n} rho_inf={rho_inf}",
                   _rel(n / infl, k_eff_expected) < 1e-9,
                   f"N_eff {n/infl:.6f} vs {k_eff_expected:.6f}")

    # -- the hard floor: sigma_fused / sigma_single -> sqrt(rho_inf) ---------
    rho_inf = 0.25
    n = 512
    ang = np.linspace(0, np.pi, n, endpoint=False)
    u = np.zeros((n, 1, 1, 3)); u[:, 0, 0, 0] = np.cos(ang); u[:, 0, 0, 1] = np.sin(ang)
    vis = np.ones((n, 1, 1), dtype=bool)
    cfg = _bare_cfg(use_correlation=True, correlation_kernel="cosine", rho_inf=rho_inf,
                    theta_c_deg=0.05, store_station_dirs=True)
    infl = float(_correlation_inflation(u, vis, cfg)[0, 0])
    # sigma_fused = sigma_single / sqrt(N_eff);  N_eff -> 1/rho_inf
    ratio = np.sqrt(infl / n)
    _check("sigma_fused/sigma_single -> sqrt(rho_inf) at large N",
           _rel(ratio, np.sqrt(rho_inf)) < 0.01,
           f"ratio {ratio:.5f}, sqrt(rho_inf) = {np.sqrt(rho_inf):.5f}")

    # -- rho_inf = 0 with wide separation recovers full 1/sqrt(N) gain ------
    cfg0 = _bare_cfg(use_correlation=True, correlation_kernel="cosine", rho_inf=0.0,
                     theta_c_deg=0.05, store_station_dirs=True)
    infl0 = float(_correlation_inflation(u, vis, cfg0)[0, 0])
    _check("rho_inf=0, wide separation -> no inflation",
           _rel(infl0, 1.0) < 1e-9, f"inflation {infl0:.9f}")

    # -- co-located stations must completeness ZERO information gain ---------------
    inst = _plain_instrument()
    g = Grid(x=np.array([20.0]), y=np.array([0.0]))
    sts = [Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst,
                   mask=None, is_anchor=True) for _ in range(16)]
    # NOTE: the Tikhonov prior is an ABSOLUTE precision floor and does not
    # scale with the number of stations, so it perturbs exact identities at
    # the 1/prior_sigma^2 level.  Weakening it demonstrates that this is the
    # entire source of the residual.
    # perfect-correlation limit: rho_0 = 1 (the measured 0.22 would give a
    # 2.1x cap, not zero gain)
    cfgc = _bare_cfg(use_correlation=True, correlation_kernel="cosine", rho_inf=0.0, rho_0=1.0,
                     theta_c_deg=15.0, store_station_dirs=True, prior_sigma_m=1.0e8)
    s_many = float(compute_metrics(solve_precision_field(g, sts, cfgc))["sigma_n"][0, 0])
    s_one = float(compute_metrics(solve_precision_field(g, sts[:1], cfgc))["sigma_n"][0, 0])
    _check("co-located stations give zero gain when correlation is on",
           _rel(s_many / s_one, 1.0) < 1e-12,
           f"ratio {s_many/s_one:.9f} (want 1.0)")




# --------------------------------------------------------------------------
# 4b. the simple 'cluster' N_eff model (default): count distinct looks
# --------------------------------------------------------------------------

def test_cluster_neff_model():
    inst = _plain_instrument()
    g = Grid(x=np.array([20.0]), y=np.array([0.0]))
    base = dict(apply_occlusion=False, cos_e_min=0.0, prior_sigma_m=1.0e8)
    cfg_c = ModelConfig(use_correlation=True, correlation_kernel="cluster",
                        theta_c_deg=5.0, **base)
    cfg_0 = ModelConfig(use_correlation=False, **base)

    def sig(sts, cfg):
        return float(compute_metrics(solve_precision_field(g, sts, cfg))["sigma_n"][0, 0])

    # k co-located stations: exactly ONE look under the cluster model,
    # 1/sqrt(k) under independent rays
    one = [Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst, mask=None, is_anchor=True)]
    many = [Station(xyz=[0, 0, 2.0], az_deg=40.0*i, instrument=inst, mask=None,
                    is_anchor=(i == 0)) for i in range(6)]
    _check("cluster model: 6 co-located stations == 1 look",
           _rel(sig(many, cfg_c), sig(one, cfg_c)) < 1e-9,
           f"ratio {sig(many, cfg_c)/sig(one, cfg_c):.9f}")
    _check("independent model: 6 co-located stations == 1/sqrt(6)",
           _rel(sig(one, cfg_0) / sig(many, cfg_0), np.sqrt(6)) < 1e-6)

    # stations separated by more than theta_c: cluster model == independent
    far = [Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst, mask=None, is_anchor=True),
           Station(xyz=[0, 8, 2.0], az_deg=0.0, instrument=inst, mask=None),   # ~22 deg apart at 20 m
           Station(xyz=[0, -8, 2.0], az_deg=0.0, instrument=inst, mask=None)]
    _check("cluster model: well-separated stations untouched",
           _rel(sig(far, cfg_c), sig(far, cfg_0)) < 1e-12)

    # a bunch of 3 within theta_c plus one separated: 2 looks, not 4
    bunch = [Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst, mask=None, is_anchor=True),
             Station(xyz=[0.3, 0, 2.0], az_deg=0.0, instrument=inst, mask=None),
             Station(xyz=[0, 0.3, 2.0], az_deg=0.0, instrument=inst, mask=None),
             Station(xyz=[0, 10, 2.0], az_deg=0.0, instrument=inst, mask=None)]
    two = [bunch[1], bunch[3]]      # (0.3,0) is the closest, hence best, member
    _check("cluster model: bunch of 3 + 1 separated == 2 looks",
           _rel(sig(bunch, cfg_c), sig(two, cfg_c)) < 1e-9,
           f"ratio {sig(bunch, cfg_c)/sig(two, cfg_c):.5f}")


def test_cluster_keeps_best_member():
    """Co-located Z34-like (fine IFOV) and Navcam-like (coarse IFOV) stations
    must give the Z34's precision, not an average of the two."""
    fine = Instrument(name="fine", ifov_rad=2.0e-4, baseline_m=0.25, eps_intra_px=0.5)
    coarse = Instrument(name="coarse", ifov_rad=3.3e-4, baseline_m=0.42, eps_intra_px=0.5)
    g = Grid(x=np.array([20.0]), y=np.array([0.0]))
    cfg_c = ModelConfig(apply_occlusion=False, cos_e_min=0.0, prior_sigma_m=1e8,
                        use_correlation=True, correlation_kernel="cluster", theta_c_deg=5.0)
    def sig(sts):
        return float(compute_metrics(solve_precision_field(g, sts, cfg_c))["sigma_n"][0, 0])
    both = [Station(xyz=[0, 0, 2.0], az_deg=0, instrument=fine, mask=None, is_anchor=True),
            Station(xyz=[0, 0, 2.0], az_deg=0, instrument=coarse, mask=None)]
    _check("co-located fine+coarse == fine alone (best member kept)",
           _rel(sig(both), sig(both[:1])) < 1e-9, f"{sig(both):.4g} vs {sig(both[:1]):.4g}")
    _check("order independent", _rel(sig(both[::-1]), sig(both[:1])) < 1e-9)


def test_cluster_neff_is_not_a_bandpass_eps():
    """
    Folding redundancy into eps_cross (a weight that vanishes at small angles)
    is NOT equivalent to the cluster model.  The station tie weight is
    w_i = 1 - prod_j(1 - w_ij): a bunched station with a good link to a
    separated anchor gets w ~ 1 and contributes in full, so 3 bunched stations
    count as 3 looks, not 1.  Demonstrated by comparing the cluster model with
    full fusion (which is what the band-pass weight reduces to here).
    """
    inst = _plain_instrument()
    g = Grid(x=np.array([20.0]), y=np.array([0.0]))
    base = dict(apply_occlusion=False, cos_e_min=0.0, prior_sigma_m=1.0e8)
    bunch = [Station(xyz=[0, 10, 2.0], az_deg=0.0, instrument=inst, mask=None, is_anchor=True),
             Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst, mask=None),
             Station(xyz=[0.3, 0, 2.0], az_deg=0.0, instrument=inst, mask=None),
             Station(xyz=[0, 0.3, 2.0], az_deg=0.0, instrument=inst, mask=None)]
    # band-pass eps: every bunched station links to the anchor at ~27 deg
    # with full weight, so its result equals plain full fusion
    f_band = solve_precision_field(g, bunch, ModelConfig(link_mode="full", use_correlation=False, **base))
    f_clu = solve_precision_field(g, bunch, ModelConfig(use_correlation=True,
                                                         correlation_kernel="cluster",
                                                         theta_c_deg=5.0, **base))
    f_two = solve_precision_field(g, [bunch[0], bunch[2]],      # anchor + best (closest) member
                                  ModelConfig(use_correlation=False, **base))
    s_band = float(compute_metrics(f_band)["sigma_n"][0, 0])
    s_clu = float(compute_metrics(f_clu)["sigma_n"][0, 0])
    s_two = float(compute_metrics(f_two)["sigma_n"][0, 0])
    _check("band-pass-eps result counts the bunch as 3 looks (clearly below cluster)",
           s_band < 0.8 * s_clu, f"band {s_band:.4g} vs cluster {s_clu:.4g}")
    _check("cluster model counts the bunch as 1 look (matches anchor + one station)",
           _rel(s_clu, s_two) < 1e-9, f"cluster {s_clu:.4g} vs two-station {s_two:.4g}")

# --------------------------------------------------------------------------
# 5. two stations at 90 degrees -> near-isotropic in the shared plane
# --------------------------------------------------------------------------

def test_convergent_pair_isotropy():
    inst = _plain_instrument()
    d = 20.0
    sts = [Station(xyz=[-d, 0, 2.0], az_deg=90.0, instrument=inst, mask=None, is_anchor=True),
           Station(xyz=[0, -d, 2.0], az_deg=0.0, instrument=inst, mask=None)]
    g = Grid(x=np.array([0.0]), y=np.array([0.0]))
    f2 = solve_precision_field(g, sts, _bare_cfg())
    f1 = solve_precision_field(g, sts[:1], _bare_cfg())
    k2 = float(compute_metrics(f2)["kappa"][0, 0])
    k1 = float(compute_metrics(f1)["kappa"][0, 0])
    _check("orthogonal pair reduces anisotropy", k2 < k1 / 10.0,
           f"kappa single {k1:.1f} -> pair {k2:.1f}")


# --------------------------------------------------------------------------
# 6. eps scaling: sigma ~ eps, G_n invariant
# --------------------------------------------------------------------------

def test_eps_scaling():
    g = Grid.square(20.0, 15)
    out = {}
    for eps in (0.25, 1.0):
        inst = _plain_instrument(eps=eps)
        sts = [Station(xyz=[-8, 0, 2.0], az_deg=90.0, instrument=inst, mask=None, is_anchor=True),
               Station(xyz=[8, 3, 2.0], az_deg=270.0, instrument=inst, mask=None)]
        # weak prior: see note in test_correlation_floor -- the prior is an
        # absolute floor and is the only thing breaking exact eps invariance.
        m = compute_metrics(solve_precision_field(
            g, sts, _bare_cfg(prior_sigma_m=1.0e8)))
        out[eps] = (m["sigma_n"], m["G_n"])
    ratio = np.nanmedian(out[1.0][0] / out[0.25][0])
    gdiff = np.nanmax(np.abs(out[1.0][1] - out[0.25][1])
                      / np.maximum(np.abs(out[0.25][1]), 1e-30))
    _check("sigma_n scales linearly with eps", _rel(ratio, 4.0) < 1e-9,
           f"ratio {ratio:.9f}")
    _check("G_n invariant to eps", gdiff < 1e-12,
           f"max relative |dG_n| = {gdiff:.3g}")


# --------------------------------------------------------------------------
# 7. rotation invariance of the whole scene
# --------------------------------------------------------------------------

def test_rotation_invariance():
    inst = _plain_instrument()
    base = [([-10.0, 2.0, 2.0], 80.0), ([6.0, -7.0, 2.0], 300.0), ([3.0, 9.0, 2.0], 190.0)]
    pts = np.array([[7.0, -3.0], [-4.0, 11.0], [15.0, 15.0]])
    phi = np.radians(37.0)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    def sigma_at(stations, P2):
        vals = []
        for p in P2:
            g = Grid(x=np.array([p[0]]), y=np.array([p[1]]))
            f = solve_precision_field(g, stations, ModelConfig(apply_occlusion=True))
            vals.append(float(compute_metrics(f)["sigma_n"][0, 0]))
        return np.array(vals)

    s0 = sigma_at([Station(xyz=x, az_deg=a, instrument=inst, is_anchor=(i == 0))
                   for i, (x, a) in enumerate(base)], pts)
    rot = []
    for i, (x, a) in enumerate(base):
        xy = R @ np.array(x[:2])
        rot.append(Station(xyz=[xy[0], xy[1], x[2]],
                           az_deg=(a + np.degrees(phi)) % 360.0,
                           instrument=inst, is_anchor=(i == 0)))
    s1 = sigma_at(rot, (R @ pts.T).T)
    err = np.nanmax(np.abs(s1 - s0) / np.abs(s0))
    _check("rotation invariance (with occlusion mask)", err < 1e-8,
           f"max relative difference {err:.3g}")


# --------------------------------------------------------------------------
# 8. occlusion mask actually masks
# --------------------------------------------------------------------------

def test_occlusion():
    mask = OcclusionMask(az_deg=[0, 180, 359], el_deg=[-10, -10, -10])
    inst = _plain_instrument()
    st = Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst, mask=mask, is_anchor=True)
    # near point: steep look-down (el ~ -63 deg) -> occluded
    g_near = Grid(x=np.array([1.0]), y=np.array([0.0]))
    # far point: shallow look-down (el ~ -3.8 deg) -> visible
    g_far = Grid(x=np.array([30.0]), y=np.array([0.0]))
    # this near-field cell is deliberately fully occluded, so the
    # no-visibility guard must be told that it is intentional
    f_near = solve_precision_field(g_near, [st], ModelConfig(
        apply_occlusion=True, on_no_visibility="ignore"))
    f_far = solve_precision_field(g_far, [st], ModelConfig(apply_occlusion=True))
    _check("steep near-field view is occluded", f_near.n_vis[0, 0] == 0,
           f"n_vis = {f_near.n_vis[0,0]}")
    _check("shallow far-field view is visible", f_far.n_vis[0, 0] == 1,
           f"n_vis = {f_far.n_vis[0,0]}")
    off = solve_precision_field(g_near, [st], ModelConfig(apply_occlusion=False))
    _check("occlusion can be disabled", off.n_vis[0, 0] == 1)


# --------------------------------------------------------------------------
# 9. pose error dominates the far field: sigma_n -> r * sigma_omega
# --------------------------------------------------------------------------

def test_pose_far_field():
    inst = _plain_instrument()
    sw = 2.0e-3
    r = 200.0
    anchor = Station(xyz=[0, 0, 2.0], az_deg=0.0, instrument=inst, mask=None, is_anchor=True)
    far = Station(xyz=[0.0, -5.0, 2.0], az_deg=0.0, instrument=inst, mask=None,
                  path_m=5.0)
    g = Grid(x=np.array([r]), y=np.array([0.0]))
    pm = PoseModel(mode="ba", ba_pos_sigma_m=0.0, ba_att_sigma_rad=sw)
    f_pose = solve_precision_field(g, [anchor, far], _bare_cfg(), pm)
    f_geom = solve_precision_field(g, [anchor, far], _bare_cfg(), PoseModel(mode="none"))
    s_pose = float(compute_metrics(f_pose)["sigma_n"][0, 0])
    s_geom = float(compute_metrics(f_geom)["sigma_n"][0, 0])
    _check("pose error degrades far-field precision", s_pose > 5 * s_geom,
           f"geom {s_geom*100:.3f} cm -> with pose {s_pose*100:.3f} cm")
    # order-of-magnitude: the attitude term contributes ~ r * sigma_omega
    expect = r * sw
    _check("far-field sigma is within an order of r*sigma_omega",
           0.1 * expect < s_pose < 10 * expect,
           f"s_pose = {s_pose:.4g}, r*sigma_w = {expect:.4g}")
    _check("anchor carries zero pose error",
           np.allclose(pm.station_sigmas(anchor)[1], 0.0))


# --------------------------------------------------------------------------
# 10. vectorised solver vs. an independent naive reference
# --------------------------------------------------------------------------

def _naive_reference(grid, stations, cfg):
    """Deliberately slow, obviously-correct implementation. Loops everything."""
    P, N = grid.points()
    Ny, Nx = grid.shape
    Sig = np.zeros((Ny, Nx, 3, 3))
    for iy in range(Ny):
        for ix in range(Nx):
            p = P[iy, ix]
            n = N[iy, ix]
            Lam = np.eye(3) / cfg.prior_sigma_m ** 2
            for st in stations:
                inst = st.instrument
                d0 = p - st.xyz
                hn = np.hypot(d0[0], d0[1])
                eb = (np.array([-d0[1], d0[0], 0.0]) / hn if hn > 1e-12
                      else np.array([1.0, 0.0, 0.0]))
                for k, frac in enumerate(inst.eye_offsets):
                    c = st.xyz + frac * inst.baseline_m * eb
                    d = p - c
                    r = np.linalg.norm(d)
                    u = d / r
                    cos_e = -float(u @ n)
                    if cos_e <= max(cfg.cos_e_min, 0.0):
                        continue
                    if cfg.apply_occlusion and st.mask is not None:
                        az = np.degrees(np.arctan2(u[0], u[1]))
                        el = np.degrees(np.arcsin(np.clip(u[2], -1, 1)))
                        if st.mask.occluded((az - st.az_deg) % 360.0, el):
                            continue
                    s = inst.eps_intra_px * inst.ifov_rad * r
                    Lam += (np.eye(3) - np.outer(u, u)) / s ** 2
            Sig[iy, ix] = np.linalg.inv(Lam)
    return Sig


def test_vectorised_matches_reference():
    inst = _plain_instrument()
    sts = [Station(xyz=[-9, 1, 2.0], az_deg=75.0, instrument=inst, is_anchor=True),
           Station(xyz=[5, -6, 1.8], az_deg=310.0, instrument=inst),
           Station(xyz=[2, 8, 2.1], az_deg=185.0, instrument=inst)]
    g = Grid.square(18.0, 11)
    cfg = ModelConfig(apply_occlusion=True, cos_e_min=0.0, use_correlation=False)
    fast = solve_precision_field(g, sts, cfg).Sigma
    slow = _naive_reference(g, sts, cfg)
    num = np.abs(fast - slow)
    den = np.maximum(np.abs(slow), 1e-12)
    err = np.max(num / den)
    _check("vectorised == naive reference", err < 1e-9,
           f"max relative difference {err:.3e}")


# --------------------------------------------------------------------------
# 10b. cross-station link weighting
# --------------------------------------------------------------------------

def test_link_modes():
    from .core import ModelConfig
    inst = _plain_instrument()
    sts = [Station(xyz=[-11, 2, 2.0], az_deg=70.0, instrument=inst, is_anchor=True),
           Station(xyz=[7, -8, 1.9], az_deg=300.0, instrument=inst),
           Station(xyz=[1, 10, 2.1], az_deg=190.0, instrument=inst)]
    g = Grid.square(22.0, 21)

    f_full = solve_precision_field(g, sts, ModelConfig(link_mode="full", use_correlation=False))
    m_full = compute_metrics(f_full)

    # eps_cross == eps_intra and a huge theta_max -> gating must reproduce 'full'
    # eps_cross=None ties it to eps_intra, which is now the default assumption
    cfg_open = ModelConfig(link_mode="gated", use_theta_gate=True, use_correlation=False,
                           gate_form="gaussian", eps_cross_px=None, theta_max_deg=1.0e6)
    m_open = compute_metrics(solve_precision_field(g, sts, cfg_open))
    d = np.nanmax(np.abs(m_open["sigma_n"] - m_full["sigma_n"])
                  / np.maximum(m_full["sigma_n"], 1e-30))
    # Tolerance is 1e-7, not 1e-9: with the gate exponent m=2 the weight at
    # theta_max=1e6 deg is 1 - ~7e-8 rather than underflowing to exactly 1 as
    # it did at m=4. Expected, and far below any physical effect.
    _check("gated with perfect cross-matching == full fusion", d < 1e-7,
           f"max relative difference {d:.3e}")

    # eps_cross enormous -> non-anchor stations drop out
    cfg_shut = ModelConfig(link_mode="gated", use_theta_gate=True, use_correlation=False,
                           eps_cross_px=1.0e6, theta_max_deg=20.0)
    f_shut = solve_precision_field(g, sts, cfg_shut)
    w = f_shut.link_weights
    non_anchor = np.array([not s.is_anchor for s in sts])
    _check("eps_cross -> inf shuts non-anchor links",
           np.nanmax(w[non_anchor]) < 1e-9,
           f"max non-anchor weight {np.nanmax(w[non_anchor]):.3e}")
    _check("anchor weight forced to 1", np.allclose(w[0][f_shut.station_vis[0]], 1.0))

    # 'ops' mode must equal the best single station
    f_ops = solve_precision_field(g, sts, ModelConfig(link_mode="ops", use_correlation=False))
    m_ops = compute_metrics(f_ops)
    best = np.min(f_ops.sigma_n_single, axis=0)
    ok = np.isfinite(m_ops["sigma_n"]) & np.isfinite(best)
    d = np.max(np.abs(m_ops["sigma_n"][ok] - best[ok]) / best[ok])
    _check("ops mode == best single station", d < 1e-9,
           f"max relative difference {d:.3e}")

    # fusion must never be worse than the ops baseline
    _check("full fusion beats ops everywhere",
           np.all(m_full["sigma_n"][ok] <= best[ok] * (1 + 1e-9)))

    # tighter theta_max must not improve anything
    m_tight = compute_metrics(solve_precision_field(g, sts, ModelConfig(
        link_mode="gated", use_theta_gate=True, eps_cross_px=0.5,
        theta_max_deg=8.0, use_correlation=False)))
    ok2 = np.isfinite(m_tight["sigma_n"]) & np.isfinite(m_open["sigma_n"])
    _check("tighter theta_max monotonically degrades precision",
           np.all(m_tight["sigma_n"][ok2] >= m_open["sigma_n"][ok2] * (1 - 1e-9)))


# --------------------------------------------------------------------------
# 10c. COLMAP round trip: write a synthetic model with KNOWN eps and theta_max,
#      read it back, and check the calibration recovers them.
# --------------------------------------------------------------------------

def _write_synthetic_colmap(d, eps_intra=0.4, eps_cross=1.1,
                            theta_max_deg=30.0, seed=11):
    import os
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(seed)
    th_max = np.radians(theta_max_deg)

    def q_from_R(R):                                   # Shepperd, branch-safe
        t = np.trace(R)
        if t > 0:
            S = np.sqrt(t + 1.0) * 2
            q = [0.25*S, (R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S]
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2]) * 2
            q = [(R[2,1]-R[1,2])/S, 0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S]
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2]) * 2
            q = [(R[0,2]-R[2,0])/S, (R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S]
        else:
            S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1]) * 2
            q = [(R[1,0]-R[0,1])/S, (R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S]
        q = np.array(q)
        return q / np.linalg.norm(q)

    cen = [np.array([0., 0., 1.9])]
    for _ in range(4):
        cen.append(cen[-1] + np.array([rng.uniform(6, 14), rng.uniform(-8, 8),
                                       rng.normal(0, .1)]))
    images, iid = {}, 1
    for si, C in enumerate(cen):
        for m in range(10):
            Cm = C + rng.normal(0, 0.12, 3)
            az = 2 * np.pi * m / 10
            f = np.array([np.sin(az), np.cos(az), -0.35]); f /= np.linalg.norm(f)
            r = np.cross(np.array([0, 0, 1.]), f); r /= np.linalg.norm(r)
            images[iid] = dict(R=np.vstack([r, np.cross(f, r), f]), C=Cm,
                               name=f"ST{si:02d}_F{m}.png")
            iid += 1

    FOC, W, H = 1400.0, 1280, 960
    P = np.column_stack([rng.uniform(-25, 45, 5000), rng.uniform(-30, 30, 5000),
                         np.zeros(5000)])
    pts, obs = [], {i: [] for i in images}
    for pid, X in enumerate(P, start=1):
        seen = []
        for i, im in images.items():
            dd = X - im["C"]; cam = im["R"] @ dd
            if cam[2] <= 0.2:
                continue
            u = FOC*cam[0]/cam[2] + W/2; v = FOC*cam[1]/cam[2] + H/2
            if 0 <= u < W and 0 <= v < H:
                seen.append((i, u, v, dd/np.linalg.norm(dd), im["name"][:4]))
        if len(seen) < 2:
            continue
        keep = [seen[0]]
        for o in seen[1:]:
            th = np.arccos(np.clip(o[3] @ seen[0][3], -1, 1))
            if rng.random() < np.exp(-(th/th_max) ** 4):
                keep.append(o)
        if len(keep) < 2:
            continue
        e = eps_cross if len({o[4] for o in keep}) > 1 else eps_intra
        trk = []
        for i, u, v, _, _ in keep:
            obs[i].append((u + rng.normal(0, e), v + rng.normal(0, e), pid))
            trk.append((i, len(obs[i]) - 1))
        pts.append((pid, X, e*np.sqrt(2), trk))

    open(f"{d}/cameras.txt", "w").write(
        f"# CAMERA_ID MODEL WIDTH HEIGHT PARAMS\n"
        f"1 SIMPLE_PINHOLE {W} {H} {FOC} {W/2} {H/2}\n")
    with open(f"{d}/images.txt", "w") as fh:
        fh.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        for i, im in images.items():
            q = q_from_R(im["R"]); t = -im["R"] @ im["C"]
            fh.write(f"{i} {' '.join(f'{x:.10f}' for x in q)} "
                     f"{' '.join(f'{x:.6f}' for x in t)} 1 {im['name']}\n")
            fh.write(" ".join(f"{u:.3f} {v:.3f} {p}" for u, v, p in obs[i]) + "\n")
    with open(f"{d}/points3D.txt", "w") as fh:
        fh.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for pid, X, err, trk in pts:
            fh.write(f"{pid} {X[0]:.6f} {X[1]:.6f} {X[2]:.6f} 128 128 128 "
                     f"{err:.4f} " + " ".join(f"{i} {k}" for i, k in trk) + "\n")
    return len(images), len(pts)


def test_colmap_roundtrip():
    import tempfile
    from .colmap import (read_colmap, calibrate_eps_from_residuals,
                             match_survival, measured_view_graph)
    EI, EC, TM = 0.4, 1.1, 30.0
    with tempfile.TemporaryDirectory() as d:
        n_im, n_pt = _write_synthetic_colmap(d, EI, EC, TM)
        m = read_colmap(d)
        _check("colmap reader parses all images", len(m.images) == n_im,
               f"{len(m.images)} of {n_im}")
        _check("colmap reader parses all points", len(m.points) == n_pt,
               f"{len(m.points)} of {n_pt}")

        g = m.assign_stations(cluster_radius_m=1.0)
        _check("station clustering recovers 5 stations", len(g) == 5,
               f"got {len(g)}: {sorted((k, len(v)) for k, v in g.items())}")

        c = calibrate_eps_from_residuals(m)
        _check("eps_intra recovered", _rel(c["eps_intra_px"], EI) < 0.10,
               f"{c['eps_intra_px']:.4f} vs {EI}")
        _check("eps_cross recovered", _rel(c["eps_cross_px"], EC) < 0.10,
               f"{c['eps_cross_px']:.4f} vs {EC}")
        _check("eps ratio recovered", _rel(c["eps_ratio"], EC/EI) < 0.10,
               f"{c['eps_ratio']:.4f} vs {EC/EI:.4f}")

        s = match_survival(m, n_bins=15, theta_max_deg=75)
        # the generator uses exp(-(th/TM)^4), whose half-point is TM*ln(2)^(1/4)
        half_true = TM * np.log(2.0) ** 0.25
        _check("theta_max recovered from match survival",
               _rel(s["theta_half_deg"], half_true) < 0.25,
               f"theta_half {s['theta_half_deg']:.1f} deg vs {half_true:.1f} deg")
        _check("survival decreases with angle",
               np.nanmax(np.diff(s["survival"][:8])) < 0.05,
               "survival curve is not monotone")

        W, names, _ = measured_view_graph(m)
        _check("measured view graph is symmetric and hollow",
               np.allclose(W, W.T) and np.allclose(np.diag(W), 0))
        _check("measured view graph connects all stations",
               len(names) == 5 and np.all(W.sum(axis=1) > 0))


def test_colmap_rejects_bad_quaternion():
    import tempfile, os
    from .colmap import read_colmap
    with tempfile.TemporaryDirectory() as d:
        _write_synthetic_colmap(d)
        lines = open(f"{d}/images.txt").read().split("\n")
        for k, ln in enumerate(lines):
            if ln and not ln.startswith("#") and len(ln.split()) == 10:
                t = ln.split(); t[1] = "3.0"          # break the quaternion
                lines[k] = " ".join(t)
                break
        open(f"{d}/images.txt", "w").write("\n".join(lines))
        try:
            read_colmap(d)
            _check("non-unit quaternion rejected", False, "no error raised")
        except ValueError as e:
            _check("non-unit quaternion rejected", "quaternion" in str(e).lower())


def test_colmap_handles_empty_observation_lines():
    """COLMAP writes a BLANK second line for images with no observations.
    Filtering blank lines desynchronises the header/observation pairing."""
    import tempfile
    from .colmap import read_colmap
    with tempfile.TemporaryDirectory() as d:
        _write_synthetic_colmap(d)
        lines = open(f"{d}/images.txt").read().split("\n")
        blanked = 0
        for k in range(len(lines) - 1):
            if lines[k] and not lines[k].startswith("#") \
                    and len(lines[k].split()) == 10 and blanked < 3:
                lines[k + 1] = ""
                blanked += 1
        open(f"{d}/images.txt", "w").write("\n".join(lines))
        m = read_colmap(d)
        _check("empty observation lines do not desynchronise the parser",
               len(m.images) == 50 and all(
                   im.name.startswith("ST") for im in m.images.values()),
               f"{len(m.images)} images, names "
               f"{sorted(im.name for im in m.images.values())[:2]}")


# --------------------------------------------------------------------------
# 10d. precision convention, correlation kernels, four cases, network pose
# --------------------------------------------------------------------------

def test_precision_convention():
    """eps='disparity' must equal eps='per_image' scaled by sqrt(2)."""
    from .core import Instrument
    g = Grid(x=np.array([18.0]), y=np.array([0.0]))
    a = Instrument(name="a", ifov_rad=2e-4, baseline_m=0.25, eps_intra_px=0.5,
                   precision_convention="per_image")
    b = Instrument(name="b", ifov_rad=2e-4, baseline_m=0.25,
                   eps_intra_px=0.5*np.sqrt(2.0),
                   precision_convention="disparity")
    sa = compute_metrics(solve_precision_field(
        g, [Station(xyz=[0,0,2.0], az_deg=0, instrument=a, mask=None,
                    is_anchor=True)], _bare_cfg()))["sigma_n"][0, 0]
    sb = compute_metrics(solve_precision_field(
        g, [Station(xyz=[0,0,2.0], az_deg=0, instrument=b, mask=None,
                    is_anchor=True)], _bare_cfg()))["sigma_n"][0, 0]
    _check("sigma_d = sqrt(2) sigma_x' conventions agree", _rel(sa, sb) < 1e-12,
           f"{sa:.6g} vs {sb:.6g}")


def test_correlation_kernels():
    from .core import correlation_kernel
    tc = np.radians(15.0)
    for kind in ("gaussian", "exponential", "cosine", "tilt"):
        _check(f"kernel {kind}: rho(0) = 1",
               _rel(float(correlation_kernel(np.array(0.0), tc, 0.0, kind)), 1.0) < 1e-12)
        th = np.linspace(0, np.pi/2, 60)
        r = correlation_kernel(th, tc, 0.0, kind)
        _check(f"kernel {kind}: monotone non-increasing",
               np.all(np.diff(r) <= 1e-12))
        _check(f"kernel {kind}: rho_inf is the floor",
               np.all(correlation_kernel(th, tc, 0.3, kind) >= 0.3 - 1e-12))
    # compact support is the reason 'cosine' is the default
    far = float(correlation_kernel(np.array(2.0*tc), tc, 0.0, "cosine"))
    _check("cosine kernel has compact support (exactly 0 beyond theta_c)",
           far < 1e-15, f"rho(2 theta_c) = {far:.3e}")
    _check("gaussian kernel does NOT reach zero",
           float(correlation_kernel(np.array(2.0*tc), tc, 0.0, "gaussian")) > 1e-3)


def test_four_cases_ordering():
    from .cases import run_four_cases
    from .core import Instrument, NAVCAM
    inst = Instrument(name="N", ifov_rad=NAVCAM.ifov_rad,
                      baseline_m=NAVCAM.baseline_m, eps_intra_px=0.5)
    sts = [Station(xyz=[0, 0, 1.9], az_deg=30, name="W1", instrument=inst),
           Station(xyz=[3.2, 1.1, 1.9], az_deg=120, name="W2", instrument=inst),
           Station(xyz=[1.0, 4.0, 1.9], az_deg=240, name="W3", instrument=inst)]
    from .core import choose_anchor
    choose_anchor(sts, "center")
    g = Grid.covering(sts, margin_m=30.0, n=61)
    res = run_four_cases(sts, g, use_correlation=False)
    for k in ("fixed", "pessimistic", "optimistic", "lbs", "ideal", "full_fusion"):
        _check(f"case '{k}' present", k in res)
    sf = res["fixed"][1]["sigma_n"]
    so = res["optimistic"][1]["sigma_n"]
    sp = res["pessimistic"][1]["sigma_n"]
    ok = np.isfinite(sf) & np.isfinite(so) & np.isfinite(sp)
    _check("optimistic beats fixed", np.all(so[ok] <= sf[ok]*(1+1e-9)))
    _check("optimistic beats pessimistic", np.all(so[ok] <= sp[ok]*(1+1e-9)))
    _check("pessimistic beats or equals fixed", np.all(sp[ok] <= sf[ok]*(1+1e-9)))
    lbs = res["lbs"][0]
    _check("lbs reports an effective baseline",
           hasattr(lbs, "effective_baseline")
           and np.nanmax(lbs.effective_baseline) > 1.0,
           f"max b_eff = {np.nanmax(lbs.effective_baseline):.2f} m")


def test_trim_border_fraction():
    from .sitemap import trim_border_fraction
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[100:300, 100:300] = 255           # a known centre-50% block
    out = trim_border_fraction(img, 0.25)
    _check("trim_border_fraction keeps the centre 50% in each dim",
           out.shape == (200, 200, 3), f"{out.shape}")
    _check("trimmed content matches the known centre block",
           np.all(out == 255), f"min={out.min()}, max={out.max()}")
    _check("frac=0 is a no-op", trim_border_fraction(img, 0.0).shape == img.shape)
    try:
        trim_border_fraction(img, 0.5)
        _check("frac=0.5 rejected (would leave nothing)", False)
    except ValueError:
        _check("frac=0.5 rejected (would leave nothing)", True)


def test_plot_map_colorbar_and_rover_icon():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import plot_map
    inst = _plain_instrument()
    sts = [Station(xyz=[-6, 1, 2.0], az_deg=75.0, instrument=inst, is_anchor=True),
           Station(xyz=[4, -3, 1.9], az_deg=310.0, instrument=inst)]
    g = Grid.square(15.0, 21)
    f = solve_precision_field(g, sts, ModelConfig())
    m = compute_metrics(f)

    fig, axes = plt.subplots(1, 2)
    ax_with = plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=axes[0],
                       show_colorbar=True)
    ax_without = plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=axes[1],
                          show_colorbar=False)
    _check("show_colorbar=True attaches a colorbar",
           len(fig.axes) > 2)  # 2 data axes + at least 1 colorbar axis
    n_axes_with_cbar = len(fig.axes)
    fig2, ax2 = plt.subplots()
    plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=ax2, show_colorbar=False)
    _check("show_colorbar=False attaches no extra axis",
           len(fig2.axes) == 1, f"{len(fig2.axes)} axes")
    plt.close(fig); plt.close(fig2)

    fig3, ax3 = plt.subplots()
    plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=ax3, show_stations=True,
             rover_icon=True)
    n_patches = len(ax3.patches)
    _check("rover_icon=True draws patches (chassis+wheels+mast)",
           n_patches >= len(sts) * 8,   # 1 body + 6 wheels + 1 mast circle, per station
           f"{n_patches} patches for {len(sts)} stations")
    plt.close(fig3)

    fig4, ax4 = plt.subplots()
    plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=ax4, show_stations=True,
             rover_icon=False)
    _check("rover_icon=False draws no patches (plain markers only)",
           len(ax4.patches) == 0, f"{len(ax4.patches)} patches")
    plt.close(fig4)


def test_study_navcam_metric_figures_and_filenames():
    """
    End-to-end: --metrics with multiple keys must produce one figure per
    metric, named Mars2020_recon_error_Sol_{sol}_{metric}.png, with the
    correct sol baked in from the real anchor waypoint.
    """
    import subprocess, sys, glob, tempfile
    if _study_script("study_navcam") is None:
        _check("study_navcam figures (skipped: studies/ not present, not a source checkout)", True)
        return
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, _study_script("study_navcam"), "--out", d,
             "--grid", "41", "--metrics", "sigma_n,kappa", "--no-sweep"],
            capture_output=True, text=True, timeout=180, env=_subprocess_env())
        _check("study_navcam --metrics run completes",
               r.returncode == 0, r.stderr[-500:] if r.returncode else "")
        for mk in ("sigma_n", "kappa"):
            hits = glob.glob(f"{d}/Mars2020_recon_error_Sol_13_{mk}.png")
            _check(f"filename convention correct for metric '{mk}'",
                   len(hits) == 1, f"found {hits}")
        # a bogus metric key should be skipped, not crash the run
        r2 = subprocess.run(
            [sys.executable, _study_script("study_navcam"), "--out", d,
             "--grid", "41", "--metrics", "sigma_n,not_a_real_metric",
             "--no-sweep"],
            capture_output=True, text=True, timeout=180, env=_subprocess_env())
        _check("unknown metric key is skipped, not fatal",
               r2.returncode == 0
               and "unknown metric key" in r2.stdout.lower(),
               r2.stdout[-300:])


def test_ideal_ordering_holds_on_real_data():
    """
    Same three-tier invariant (full_fusion <= ideal <= lbs) but on the BUNDLED
    REAL waypoint geometry, which is exactly what exposed the visibility
    inconsistency this suite's other tests fixed (real occlusion-mask
    boundary interactions don't reliably show up in hand-placed synthetic
    layouts). Skips gracefully if the data file is not present in this build.
    """
    import os
    from .core import Instrument, NAVCAM
    from .waypoints import stations_from_rmcs
    from .cases import run_four_cases
    from ..waypoints import snapshot_path
    data = str(snapshot_path())                       # the frozen packaged snapshot
    if not os.path.exists(data):
        _check("real-data ordering test (skipped, data file absent)", True)
        return
    inst = Instrument(name="Navcam", ifov_rad=NAVCAM.ifov_rad,
                      baseline_m=NAVCAM.baseline_m, eps_intra_px=0.5)
    sts, _ = stations_from_rmcs(data, ["3_0", "3_110", "3_1266", "3_1398"],
                                instrument=inst)
    g = Grid.square(25.0, 91)
    res = run_four_cases(sts, g, use_correlation=False, pose=PoseModel())
    s_lbs = res["lbs"][1]["sigma_n"]
    s_ideal = res["ideal"][1]["sigma_n"]
    s_full = res["full_fusion"][1]["sigma_n"]
    ok = np.isfinite(s_lbs) & np.isfinite(s_ideal) & np.isfinite(s_full)
    _check("real data (equal pose): ideal never worse than lbs",
           np.all(s_ideal[ok] <= s_lbs[ok] * (1 + 1e-6)),
           f"{int(np.sum(s_ideal[ok] > s_lbs[ok]*(1+1e-6)))} violations "
           f"of {int(ok.sum())} cells")
    _check("real data (equal pose): full_fusion never worse than ideal",
           np.all(s_full[ok] <= s_ideal[ok] * (1 + 1e-6)),
           f"{int(np.sum(s_full[ok] > s_ideal[ok]*(1+1e-6)))} violations "
           f"of {int(ok.sum())} cells")


def test_stations_from_rmcs():
    from .waypoints import stations_from_rmcs
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0, 0]},
         "properties": {"RMC": "1_0", "site": 1, "drive": 0, "sol": 1,
                        "easting": 1000.0, "northing": 2000.0,
                        "elev_geoid": -100.0, "yaw": 90.0, "dist_total_m": 0.0}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0, 0]},
         "properties": {"RMC": "1_50", "site": 1, "drive": 50, "sol": 3,
                        "easting": 1010.0, "northing": 2005.0,
                        "elev_geoid": -100.2, "yaw": 200.0,
                        "dist_total_m": 47.3}},   # winding path >> straight-line
    ]}
    sts, info = stations_from_rmcs(fc, ["1_0", "1_50"])
    _check("stations_from_rmcs returns the right count", len(sts) == 2)
    _check("anchor defaults to the first RMC and sits at the origin",
           sts[0].is_anchor and np.allclose(sts[0].xyz[:2], [0, 0]))
    straight = float(np.linalg.norm(sts[1].xyz[:2]))
    _check("second station offset matches easting/northing delta",
           abs(straight - np.hypot(10.0, 5.0)) < 1e-6, f"{straight:.4f}")
    _check("path_m uses REAL ODOMETRY (dist_total_m), not straight-line",
           abs(sts[1].path_m - 47.3) < 1e-6 and sts[1].path_m > straight * 3,
           f"path_m={sts[1].path_m}, straight-line={straight:.2f}")
    _check("camera height applied on top of terrain elevation",
           abs(sts[0].xyz[2] - 1.9) < 1e-9)

    try:
        stations_from_rmcs(fc, ["1_0", "9_999"])
        _check("missing RMC raises", False, "no exception")
    except ValueError as e:
        _check("missing RMC raises", "9_999" in str(e))


def test_ideal_dominates_lbs_and_is_dominated_by_full_fusion():
    """
    Ordering that must hold everywhere AT EQUAL POSE UNCERTAINTY, by
    construction (more independent ray information cannot increase variance):

        full_fusion <= ideal <= lbs

    The equal-pose condition is essential and is why these calls pass an
    explicit PoseModel(). In normal use run_four_cases applies a DIFFERENT pose
    model per case (CASE_POSE): LBS/IDEAL are zero-pose ceilings while
    full_fusion carries SfM pose error, so full_fusion can legitimately exceed
    IDEAL there. That is physics, not a violation -- the invariant is about
    ray information alone.

    full_fusion sums EVERY ray from every station (both eyes each) at perfect
    matching; ideal sums ONE ray per station (centre only) at perfect
    matching -- a strict subset of full_fusion's terms, so full_fusion is at
    least as informative.  lbs uses only the single best PAIR's two rays -- a
    subset of ideal's terms (which include every station), so ideal is at
    least as informative as lbs.  This is the corrected replacement for the
    old "ideal never exceeds full_fusion" check, extended to include the new
    all-pairs IDEAL sitting strictly between the two.
    """
    from .cases import run_four_cases
    from .core import Instrument, NAVCAM, choose_anchor
    inst = Instrument(name="N", ifov_rad=NAVCAM.ifov_rad,
                      baseline_m=NAVCAM.baseline_m, eps_intra_px=0.5)
    sts = [Station(xyz=[0, 0, 1.9], az_deg=30, name="W1", instrument=inst),
           Station(xyz=[4.1, 1.1, 1.9], az_deg=120, name="W2", instrument=inst),
           Station(xyz=[1.0, 4.4, 1.9], az_deg=240, name="W3", instrument=inst),
           Station(xyz=[-2.3, 2.6, 1.9], az_deg=300, name="W4", instrument=inst)]
    choose_anchor(sts, "first")
    g = Grid.square(25.0, 71)
    res = run_four_cases(sts, g, use_correlation=False, include_full=True,
                         pose=PoseModel())
    s_lbs = res["lbs"][1]["sigma_n"]
    s_ideal = res["ideal"][1]["sigma_n"]
    s_full = res["full_fusion"][1]["sigma_n"]
    ok = np.isfinite(s_lbs) & np.isfinite(s_ideal) & np.isfinite(s_full)
    _check("ideal (all pairs) never worse than lbs (single best pair)",
           np.all(s_ideal[ok] <= s_lbs[ok] * (1 + 1e-6)),
           f"max violation {np.nanmax(s_ideal[ok]/s_lbs[ok]):.4f}")
    _check("full fusion never worse than ideal (all pairs, centre rays only)",
           np.all(s_full[ok] <= s_ideal[ok] * (1 + 1e-6)),
           f"max violation {np.nanmax(s_full[ok]/s_ideal[ok]):.4f}")
    # sanity: ideal should differ from both endpoints where multiple stations
    # contribute (otherwise the three-case ordering isn't actually exercised)
    n_vis = res["full_fusion"][0].n_vis
    many = ok & (n_vis >= 3)
    if many.any():
        _check("ideal is strictly between lbs and full_fusion somewhere "
               "(the three-tier ordering is actually exercised)",
               np.any(s_full[many] < s_ideal[many] - 1e-9)
               and np.any(s_ideal[many] < s_lbs[many] - 1e-9))


def test_ideal_field_uses_single_ray_per_station():
    """ideal_field must ignore each station's own intra-pair baseline -- it
    should give the SAME result as full_fusion on stations whose instrument
    already has eye_offsets=(0.0,) (i.e. no intra-pair to lose)."""
    from .core import Instrument, choose_anchor
    from .cases import ideal_field
    mono = Instrument(name="mono", ifov_rad=2e-4, baseline_m=0.25,
                      eps_intra_px=0.5, eye_offsets=(0.0,))
    sts = [Station(xyz=[0, 0, 1.9], az_deg=30, name="W1", instrument=mono),
           Station(xyz=[4.1, 1.1, 1.9], az_deg=120, name="W2", instrument=mono)]
    choose_anchor(sts, "first")
    g = Grid.square(15.0, 41)
    f_ideal = ideal_field(g, sts, ModelConfig(store_station_dirs=True, use_correlation=False))
    from .core import ModelConfig as MC
    f_full = solve_precision_field(g, sts, MC(link_mode="full", use_correlation=False,
                                              store_station_dirs=True))
    m1 = compute_metrics(f_ideal)["sigma_n"]
    m2 = compute_metrics(f_full)["sigma_n"]
    ok = np.isfinite(m1) & np.isfinite(m2)
    _check("ideal_field == full fusion when stations are already monocular",
           np.allclose(m1[ok], m2[ok], rtol=1e-9),
           f"max relative diff {np.nanmax(np.abs(m1[ok]-m2[ok])/m2[ok]):.2e}")


def test_network_pose():
    from .network import (pose_covariance_from_network, datum_modes,
                              rigidity_score)
    from .core import Instrument, NAVCAM, choose_anchor
    inst = Instrument(name="N", ifov_rad=NAVCAM.ifov_rad,
                      baseline_m=NAVCAM.baseline_m, eps_intra_px=0.5)
    sts = [Station(xyz=[0, 0, 1.9], az_deg=30, name="W1", instrument=inst),
           Station(xyz=[3.2, 1.1, 1.95], az_deg=120, name="W2", instrument=inst),
           Station(xyz=[1.0, 4.0, 1.85], az_deg=240, name="W3", instrument=inst),
           Station(xyz=[-2.5, 2.2, 1.9], az_deg=310, name="W4", instrument=inst)]
    choose_anchor(sts, "center")
    g = Grid.covering(sts, margin_m=40.0, n=101)
    net = pose_covariance_from_network(sts, g, n_tie=40)

    _check("reduced normal matrix has the expected rank",
           net.rank == 6*len(sts) - net.datum_defect,
           f"rank {net.rank}, expected {6*len(sts)-net.datum_defect}")
    _check("network is well determined", net.well_determined)

    ev = np.sort(net.eigenvalues)
    _check("three exact translation null modes present",
           np.all(np.abs(ev[:3]) / ev[-1] < 1e-12),
           f"first three relative: {[f'{v/ev[-1]:.1e}' for v in ev[:3]]}")
    _check("null-space leak is small",
           getattr(net, "null_leak", 1.0) < 1e-3,
           f"leak = {getattr(net,'null_leak',float('nan')):.2e}")

    G = datum_modes(sts, fix_scale=True)
    _check("datum modes have the right shape", G.shape == (6*len(sts), 6),
           f"{G.shape}")
    G7 = datum_modes(sts, fix_scale=False)
    _check("scale mode added when scale is free", G7.shape[1] == 7)

    rel = net.relative_to("W1")
    _check("relative_to gives the reference station exactly zero",
           np.allclose(rel.Sigma[0], 0.0, atol=1e-18),
           f"max |Sigma_ref| = {np.abs(rel.Sigma[0]).max():.3e}")
    _check("relative_to leaves other stations positive",
           np.all(rel.pos_sigma_m[1:] > 0))
    _check("covariance blocks are symmetric PSD",
           all(np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > -1e-15
               for S in net.Sigma))

    r = rigidity_score(net)
    _check("conditioning is reported and positive",
           0 < r["conditioning"] < 1, f"{r['conditioning']:.3e}")

    # more tie points must not degrade pose precision
    net2 = pose_covariance_from_network(sts, g, n_tie=80)
    _check("more tie points do not degrade pose precision",
           np.mean(net2.pos_sigma_m) <= np.mean(net.pos_sigma_m) * 1.05,
           f"{np.mean(net2.pos_sigma_m):.5f} vs {np.mean(net.pos_sigma_m):.5f}")

    # tie correlation must inflate
    net3 = pose_covariance_from_network(sts, g, n_tie=40, tie_correlation=0.01)
    _check("tie correlation inflates the covariance",
           np.mean(net3.pos_sigma_m) > np.mean(net.pos_sigma_m))


def test_plot_stats_box_and_axis_label_control():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import plot_map
    inst = _plain_instrument()
    sts = [Station(xyz=[-6, 1, 2.0], az_deg=75.0, instrument=inst, is_anchor=True),
           Station(xyz=[4, -3, 1.9], az_deg=310.0, instrument=inst)]
    g = Grid.square(15.0, 21)
    f = solve_precision_field(g, sts, ModelConfig())
    m = compute_metrics(f)
    fig, axes = plt.subplots(1, 3)

    ax0 = plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=axes[0],
                   show_stations=False, stats_box=True, stats_unit=" cm")
    v = m["sigma_n"][np.isfinite(m["sigma_n"])] * 100
    expect_mean, expect_med = float(np.mean(v)), float(np.median(v))
    expect_rms = float(np.sqrt(np.mean(v ** 2)))
    _check("stats box is drawn when requested", len(ax0.texts) == 1,
           f"{len(ax0.texts)} text artists")
    if ax0.texts:
        txt = ax0.texts[0].get_text()
        _check("stats box mean matches computed mean",
               f"{expect_mean:.2f}" in txt, txt)
        _check("stats box rms matches computed rms",
               f"{expect_rms:.2f}" in txt, txt)
        _check("stats box median matches computed median",
               f"{expect_med:.2f}" in txt, txt)

    ax1 = plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=axes[1],
                   show_stations=False, stats_box=False)
    _check("no stats box when not requested", len(ax1.texts) == 0)

    ax2 = plot_map(f, m["sigma_n"], scale=100.0, log=True, ax=axes[2],
                   show_stations=False, show_xlabel=False, show_ylabel=False,
                   show_xticklabels=False, show_yticklabels=False)
    _check("xlabel suppressed", ax2.get_xlabel() == "")
    _check("ylabel suppressed", ax2.get_ylabel() == "")
    _check("x tick labels suppressed",
           not any(t.get_text() for t in ax2.get_xticklabels()
                   if t.get_position()[0] >= ax2.get_xlim()[0])
           or not ax2.xaxis.get_tick_params()["labelbottom"],
           "labelbottom still True")
    _check("y tick labels suppressed",
           not ax2.yaxis.get_tick_params()["labelleft"],
           "labelleft still True")
    plt.close(fig)


def test_sitemap_crop():
    from .sitemap import crop_to_extent, load_site_image, SiteImageError
    img = np.full((500, 500, 3), 200, dtype=np.uint8)
    W = 50.0
    px_per_m = 500 / W
    col = int(250 + 10*px_per_m); row = int(250 - 10*px_per_m)
    img[row-3:row+3, col-3:col+3] = [255, 0, 0]

    crop = crop_to_extent(img, W, (-20, 20, -20, 20))
    _check("crop has the expected shape", crop.shape == (400, 400, 3),
           f"{crop.shape}")
    _check("fully-contained crop is not flagged partial",
           crop.partial_coverage is False)
    mask = (crop[..., 0] > 200) & (crop[..., 1] < 50)
    rr, cc = np.where(mask)
    ppm = crop.shape[1] / 40.0
    x_est = cc.mean()/ppm - 20
    y_est = 20 - rr.mean()/ppm
    _check("known feature recovered at the right E/N position",
           abs(x_est - 10) < 0.5 and abs(y_est - 10) < 0.5,
           f"x={x_est:.2f} (want 10), y={y_est:.2f} (want 10)")

    crop2 = crop_to_extent(img, W, (-40, 40, -40, 40))
    _check("out-of-bounds extent is padded and flagged partial",
           crop2.shape == (800, 800, 3) and crop2.partial_coverage,
           f"shape {crop2.shape}, partial {crop2.partial_coverage}")
    _check("padding uses white fill",
           np.mean(crop2 == 255) > 0.3, f"{np.mean(crop2==255):.3f}")

    try:
        crop_to_extent(img, W, (100, 140, 100, 140))
        _check("fully-outside extent raises", False, "no exception raised")
    except SiteImageError:
        _check("fully-outside extent raises", True)

    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        p_img = f"{d}/s.jpg"
        Image.fromarray(img).save(p_img)
        loaded = load_site_image(p_img)
        _check("local image loads with the right shape",
               loaded.shape == (500, 500, 3), f"{loaded.shape}")
    try:
        load_site_image("https://mcz-images.sese.asu.edu/nope.jpg")
        _check("blocked-host fetch raises with an actionable message", False)
    except SiteImageError as e:
        _check("blocked-host fetch raises with an actionable message",
               "network" in str(e).lower() or "allowlist" in str(e).lower()
               or "add" in str(e).lower(), str(e)[:80])


def test_spacing_sweep_baseline_marker():
    """
    Regression test for two bugs found by visual inspection that automated
    testing should have caught:
      1. a silent str.replace() no-op left the baseline-marker code entirely
         absent despite the script printing "patched" and exiting 0;
      2. set_xlim() called before set_xscale('log') was silently discarded
         when the scale changed, collapsing the whole plot to a sliver.
    Both produced a working, crash-free script with a visibly wrong figure --
    exactly the failure mode plain execution checks miss.
    """
    import matplotlib
    matplotlib.use("Agg")
    import subprocess, sys, glob, tempfile
    if _study_script("study_navcam") is None:
        _check("study_navcam sweep (skipped: studies/ not present, not a source checkout)", True)
        return
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, _study_script("study_navcam"), "--out", d,
             "--stations", "3", "--grid", "41"],
            capture_output=True, text=True, timeout=180, env=_subprocess_env())
        _check("study_navcam runs to completion", r.returncode == 0,
               r.stderr[-500:] if r.returncode else "")
        png = glob.glob(f"{d}/navcam_spacing_sweep.png")
        _check("spacing sweep figure was written", len(png) == 1)
        if not png:
            return
        from PIL import Image
        import numpy as np
        im = np.asarray(Image.open(png[0]).convert("RGB"))
        # The specific green used for the baseline marker (tab:green ~ (44,160,44))
        green = ((np.abs(im[..., 0].astype(int) - 44) < 25) &
                 (np.abs(im[..., 1].astype(int) - 160) < 25) &
                 (np.abs(im[..., 2].astype(int) - 44) < 25))
        _check("baseline marker (tab:green) is actually drawn in the figure",
               green.sum() > 50, f"{green.sum()} matching px")
        # Collapsed-axis failure mode: nearly all plotted (non-white,
        # non-black-text) content crammed into a narrow vertical strip.
        content = np.any(im < 250, axis=-1)                # non-white pixels
        col_has_content = content.any(axis=0)
        cols = np.where(col_has_content)[0]
        if cols.size:
            span_frac = (cols.max() - cols.min()) / im.shape[1]
            _check("plot content spans a reasonable fraction of the figure "
                   "width (not collapsed to a sliver)",
                   span_frac > 0.5, f"content spans {span_frac:.2f} of width")


def test_plot_shares_colour_limits():
    """Explicit vmin/vmax must be honoured EXACTLY, or shared colour bars across
    panels silently desynchronise and the panels stop being comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import plot_map
    inst = _plain_instrument()
    sts = [Station(xyz=[-6, 1, 2.0], az_deg=75.0, instrument=inst, is_anchor=True),
           Station(xyz=[4, -3, 1.9], az_deg=310.0, instrument=inst)]
    g = Grid.square(15.0, 21)
    f = solve_precision_field(g, sts, ModelConfig())
    m = compute_metrics(f)
    fig, axes = plt.subplots(1, 2)
    for log, data in ((True, m["sigma_n"]), (False, m["n_vis"])):
        ax = plot_map(f, data, scale=100.0, log=log, vmin=1.0, vmax=50.0,
                      ax=axes[0 if log else 1], show_stations=False)
        im = ax.images[-1]
        _check(f"explicit vmin honoured (log={log})",
               abs(im.norm.vmin - 1.0) < 1e-12, f"vmin = {im.norm.vmin}")
        _check(f"explicit vmax honoured (log={log})",
               abs(im.norm.vmax - 50.0) < 1e-12, f"vmax = {im.norm.vmax}")
    plt.close(fig)


def test_pose_modes_path_vs_baseline():
    """
    The distinction that matters for the four cases:
      telemetry/deadreckon scale with PATH DRIVEN,
      sfm scales with STRAIGHT-LINE BASELINE to the anchor.
    A station 18 m away but 167 m driven must get a large telemetry sigma and
    a small sfm sigma -- that asymmetry IS the argument for cross-station ties.
    """
    inst = _plain_instrument()
    anchor = Station(xyz=[0, 0, 1.9], az_deg=0, instrument=inst, path_m=0.0, is_anchor=True)
    far = Station(xyz=[18.0, 0, 1.9], az_deg=0, instrument=inst, path_m=167.0)
    tele = PoseModel(mode="telemetry").bind_anchor([anchor, far])
    sfm = PoseModel(mode="sfm").bind_anchor([anchor, far])
    st_t, sa_t = tele.station_sigmas(far)
    st_s, sa_s = sfm.station_sigmas(far)
    _check("telemetry pose scales with PATH (167 m)",
           _rel(st_t[0], tele.vo_drift_frac * 167.0) < 1e-9, f"{st_t[0]:.4f} m")
    _check("sfm pose scales with BASELINE (18 m), not path",
           _rel(st_s[0], sfm.ba_drift_frac * 18.0) < 1e-9, f"{st_s[0]:.4f} m")
    _check("sfm pose is far smaller than telemetry for a winding drive",
           st_s[0] < st_t[0] / 20, f"sfm {st_s[0]:.4f} vs telemetry {st_t[0]:.4f}")
    _check("attitude follows sigma_pos / tie_range",
           _rel(sa_s[0], max(st_s[0]/sfm.tie_range_m, sfm.ba_att_sigma_rad)) < 1e-9)
    _check("anchor itself has zero pose error",
           np.allclose(sfm.station_sigmas(anchor)[0], 0.0))

    # the per-case mapping must use these
    from .cases import CASE_POSE
    _check("all cases are geometry-only (pose none); registration reported separately",
           CASE_POSE["fixed"] == "none"
           and CASE_POSE["optimistic"] == "none"
           and CASE_POSE["lbs"] == "none" and CASE_POSE["ideal"] == "none")
    reg = PoseModel(mode="registered", reg_sigma_m=0.3)
    _check("registered mode is a constant, independent of path",
           np.allclose(reg.station_sigmas(far)[0], 0.3))


def test_occlusion_profiles_from_csv():
    from .core import load_occlusion_profiles, eye_masks_for
    import os
    from ..paths import data_dir
    if not os.path.exists(os.path.join(str(data_dir()), "M2020_occlusion_profiles.csv")):
        _check("occlusion profile CSV (skipped, absent)", True)
        return
    prof = load_occlusion_profiles()
    _check("four per-eye profiles loaded",
           set(prof) == {"zcam_left", "zcam_right", "ncam_left", "ncam_right"},
           f"{sorted(prof)}")
    _check("left and right eyes differ (they are separate measurements)",
           not np.allclose(prof["zcam_left"].el_deg, prof["zcam_right"].el_deg))
    _check("zcam and ncam are identical as delivered (documented duplication)",
           np.allclose(prof["zcam_left"].el_deg, prof["ncam_left"].el_deg))
    # rear of the deck (az~180) blocks steep looks; the side does not
    _check("occlusion blocks steep looks over the rear deck",
           bool(prof["zcam_left"].occluded(np.array([180.0]), np.array([-30.0]))[0]))
    _check("side view at the same elevation is clear",
           not bool(prof["zcam_left"].occluded(np.array([90.0]), np.array([-30.0]))[0]))
    _check("eye_masks_for returns [left, right]", len(eye_masks_for("ncam")) == 2)


def test_measured_gate():
    from .core import gate_powerlaw, illumination_gate
    cfg = ModelConfig(gate_A=0.4, theta_bar_deg=4.3, gate_cv=0.36)
    _check("gate at theta=0 equals A", abs(float(gate_powerlaw(0.0, cfg)) - 0.4) < 1e-12)
    th = np.radians(np.array([1, 2, 4.3, 10, 20, 45]))
    g = gate_powerlaw(th, cfg) / 0.4
    _check("gate is monotone decreasing", np.all(np.diff(g) < 0))
    # CV -> 0 must tend to the plain exponential exp(-theta/theta_bar)
    cfg0 = ModelConfig(gate_A=1.0, theta_bar_deg=4.3, gate_cv=0.02)
    e = np.exp(-th / np.radians(4.3))
    _check("CV->0 limit is the plain exponential",
           np.max(np.abs(gate_powerlaw(th, cfg0) - e) / e) < 0.05)
    # heavier tail than exponential for CV > 0 (the terrain-heterogeneity signature)
    cfg1 = ModelConfig(gate_A=1.0, theta_bar_deg=4.3, gate_cv=0.5)
    _check("CV>0 gives a heavier tail than the exponential at wide angle",
           float(gate_powerlaw(np.radians(30.0), cfg1)) > float(np.exp(-30/4.3)))
    _check("illumination gate: dL=0 -> 1", illumination_gate(12.0, 12.0, 2.2) == 1.0)
    _check("illumination gate: e-fold at tau",
           abs(illumination_gate(12.0, 14.2, 2.2) - np.exp(-1.0)) < 1e-9)
    _check("illumination gate: unknown time -> 1", illumination_gate(None, 12.0, 2.2) == 1.0)
    _check("illumination gate: wraps midnight",
           abs(illumination_gate(23.5, 0.5, 2.2) - np.exp(-1.0/2.2)) < 1e-9)
    # a network with 3 h of LMST spread must fuse worse than a co-temporal one
    inst = _plain_instrument()
    g = Grid.square(15.0, 21)
    base = dict(apply_occlusion=False, cos_e_min=0.0, link_mode="gated", use_theta_gate=True)
    A = [Station(xyz=[0,0,2.0], az_deg=0, instrument=inst, mask=None, is_anchor=True, lmst_h=12.0),
         Station(xyz=[3,0,2.0], az_deg=0, instrument=inst, mask=None, lmst_h=12.0)]
    B = [Station(xyz=[0,0,2.0], az_deg=0, instrument=inst, mask=None, is_anchor=True, lmst_h=12.0),
         Station(xyz=[3,0,2.0], az_deg=0, instrument=inst, mask=None, lmst_h=15.0)]
    sa = np.nanmedian(compute_metrics(solve_precision_field(g, A, ModelConfig(**base)))["sigma_n"])
    sb = np.nanmedian(compute_metrics(solve_precision_field(g, B, ModelConfig(**base)))["sigma_n"])
    _check("3 h of LMST offset degrades the fused precision", sb > sa * 1.05, f"{sa:.4g} vs {sb:.4g}")

    # clumping: a cell with mu = k ln2 expected pairs has P_cross = 0.5
    inst2 = _plain_instrument()
    g2 = Grid(x=np.array([15.0]), y=np.array([0.0]))
    cfgc = ModelConfig(apply_occlusion=False, cos_e_min=0.0, link_mode="cross",
                       use_theta_gate=True, gate_form="powerlaw", clump_k=20.0)
    two = [Station(xyz=[0,0,2.0], az_deg=0, instrument=inst2, mask=None, is_anchor=True),
           Station(xyz=[1.5,0,2.0], az_deg=0, instrument=inst2, mask=None)]
    fc = solve_precision_field(g2, two, cfgc)
    mu = float(fc.completeness[0,0]); pc = float(fc.p_cross[0,0])
    _check("p_cross = 1 - exp(-mu/k)", abs(pc - (1-np.exp(-mu/20.0))) < 1e-12, f"mu={mu:.3f} p={pc:.3f}")
    _check("p_cross is a probability", 0.0 <= pc <= 1.0)
    # cross geometry cannot make a cell worse than intra-only, and improves it
    cfg_intra = ModelConfig(apply_occlusion=False, cos_e_min=0.0, link_mode="nearest")
    s_intra = float(compute_metrics(solve_precision_field(g2, two, cfg_intra))["sigma_n"][0,0])
    s_cross = float(compute_metrics(fc)["sigma_n"][0,0])
    _check("cross mode never worse than nearest-station intra", s_cross <= s_intra*(1+1e-9), f"{s_cross:.4g} vs {s_intra:.4g}")
    _check("completeness uses tau (counts), not L_eff",
           ModelConfig().tau_h > ModelConfig().L_eff_h)

    from .cases import CASE_PARAMS
    _check("three matching cases present", set(CASE_PARAMS) == {"pessimistic","measured","optimistic"})
    _check("cases are ordered pessimistic < measured < optimistic in theta_bar",
           CASE_PARAMS["pessimistic"][1] < CASE_PARAMS["measured"][1] < CASE_PARAMS["optimistic"][1])


def test_redundancy_cap():
    """N repeat looks from one place must saturate at 1/sqrt(rho_0) = 2.1x."""
    inst = _plain_instrument()
    g = Grid(x=np.array([20.0]), y=np.array([0.0]))
    base = dict(apply_occlusion=False, cos_e_min=0.0, prior_sigma_m=1e8, link_mode="full",
                use_correlation=True, correlation_kernel="exponential", rho_0=0.22, rho_inf=0.0,
                theta_c_deg=0.4)
    def sig(n):
        sts = [Station(xyz=[0,0,2.0], az_deg=0, instrument=inst, mask=None, is_anchor=(i==0)) for i in range(n)]
        return float(compute_metrics(solve_precision_field(g, sts, ModelConfig(**base)))["sigma_n"][0,0])
    s1 = sig(1)
    for n, want in ((2, 1.28), (4, 1.55), (8, 1.77)):
        got = s1/sig(n)
        _check(f"{n} repeat looks improve by ~{want}x (cap 2.1x)", abs(got-want)/want < 0.03, f"{got:.3f}")
    # two stations 1.5 m apart at 20 m (4.3 deg) are independent looks: ~sqrt(2)
    two = [Station(xyz=[0,0,2.0], az_deg=0, instrument=inst, mask=None, is_anchor=True),
           Station(xyz=[1.5,0,2.0], az_deg=0, instrument=inst, mask=None)]
    cfg2 = ModelConfig(**{**base, "use_correlation": True})
    cfg0 = ModelConfig(**{**base, "use_correlation": False})
    r = float(compute_metrics(solve_precision_field(g, two, cfg0))["sigma_n"][0,0]) / \
        float(compute_metrics(solve_precision_field(g, two, cfg2))["sigma_n"][0,0])
    _check("separated stations are not penalised by the zero-angle correlation (<5%; residual is the intra L/R pair at 1.2 deg)", abs(r-1.0) < 0.05, f"{r:.4f}")


def test_anchor_modes():
    from .core import choose_anchor
    sts = [Station(xyz=[0, 0, 1.9], az_deg=0, name="A"),
           Station(xyz=[10, 0, 1.9], az_deg=0, name="B"),
           Station(xyz=[20, 0, 1.9], az_deg=0, name="C")]
    _check("anchor 'first' picks index 0", choose_anchor(sts, "first") == 0)
    _check("anchor 'center' picks the middle station",
           choose_anchor(sts, "center") == 1)
    _check("anchor flag is exclusive",
           sum(s.is_anchor for s in sts) == 1)
    _check("anchor 'site' honours an existing flag",
           choose_anchor(sts, "site") == 1)


# --------------------------------------------------------------------------
# 11. numerical hygiene
# --------------------------------------------------------------------------

def test_numerics():
    inst = _plain_instrument()
    sts = [Station(xyz=[-9, 1, 2.0], az_deg=75.0, instrument=inst, is_anchor=True),
           Station(xyz=[5, -6, 1.8], az_deg=310.0, instrument=inst)]
    g = Grid.square(40.0, 41)
    f = solve_precision_field(g, sts, ModelConfig())
    S = f.Sigma
    _check("Sigma finite everywhere", np.all(np.isfinite(S)))
    _check("Sigma symmetric", np.max(np.abs(S - np.swapaxes(S, -1, -2))) < 1e-18)
    ev = np.linalg.eigvalsh(S)
    _check("Sigma positive definite", np.all(ev > 0), f"min eig {ev.min():.3g}")
    # station directly overhead: hn -> 0 degenerate baseline direction
    st_over = Station(xyz=[0.0, 0.0, 3.0], az_deg=0.0, instrument=inst,
                      mask=None, is_anchor=True)
    g0 = Grid(x=np.array([0.0]), y=np.array([0.0]))
    f0 = solve_precision_field(g0, [st_over], ModelConfig(apply_occlusion=False))
    _check("degenerate overhead geometry handled",
           np.all(np.isfinite(f0.Sigma)) and f0.n_vis[0, 0] == 1)


# --------------------------------------------------------------------------
# 12. pixel-locking-free sanity: metrics run end to end
# --------------------------------------------------------------------------

def test_metrics_end_to_end():
    inst = _plain_instrument()
    sts = [Station(xyz=[-9, 1, 2.0], az_deg=75.0, instrument=inst, is_anchor=True),
           Station(xyz=[5, -6, 1.8], az_deg=310.0, instrument=inst),
           Station(xyz=[2, 8, 2.1], az_deg=185.0, instrument=inst)]
    g = Grid.square(25.0, 31)
    f = solve_precision_field(g, sts, ModelConfig())
    m = compute_metrics(f, rho_adjacent=0.0, coverage_threshold_m=0.05)
    need = ["sigma_n", "G_n", "gsd", "kappa", "n_vis", "n_eff", "improvement",
            "sigma_slope", "sigma_t_min", "sigma_t_max", "_coverage_fraction"]
    missing = [k for k in need if k not in m]
    _check("all metrics present", not missing, f"missing {missing}")
    good = np.isfinite(m["sigma_n"])
    _check("some cells are constrained", good.sum() > 0.5 * good.size,
           f"{good.sum()}/{good.size}")
    _check("gain >= 1 where fused", np.all(m["improvement"][np.isfinite(m["improvement"])] >= 1 - 1e-9))
    _check("coverage fraction in [0,1]", 0.0 <= float(m["_coverage_fraction"]) <= 1.0)


# --------------------------------------------------------------------------

def _study_script(name: str):
    """Path of a research script in ``studies/error/`` (v0p13: moved out of the package), or None."""
    from ..paths import source_root
    root = source_root()
    f = root / "studies" / "error" / f"{name}.py" if root else None
    return str(f) if f is not None and f.is_file() else None


def _subprocess_env():
    """Child interpreters must find the ``mppp`` package (merged layout, v0p2)."""
    import os
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_all(verbose: bool = True) -> bool:
    _RESULTS.clear()
    tests = [
        test_two_ray_analytic,
        test_monocular_rank_deficient,
        test_precision_addition,
        test_correlation_floor,
        test_cluster_neff_model,
        test_cluster_keeps_best_member,
        test_cluster_neff_is_not_a_bandpass_eps,
        test_convergent_pair_isotropy,
        test_eps_scaling,
        test_rotation_invariance,
        test_occlusion,
        test_pose_far_field,
        test_vectorised_matches_reference,
        test_link_modes,
        test_colmap_roundtrip,
        test_colmap_rejects_bad_quaternion,
        test_colmap_handles_empty_observation_lines,
        test_precision_convention,
        test_correlation_kernels,
        test_four_cases_ordering,
        test_network_pose,
        test_anchor_modes,
        test_redundancy_cap,
        test_measured_gate,
        test_pose_modes_path_vs_baseline,
        test_occlusion_profiles_from_csv,
        test_plot_shares_colour_limits,
        test_plot_stats_box_and_axis_label_control,
        test_sitemap_crop,
        test_ideal_dominates_lbs_and_is_dominated_by_full_fusion,
        test_ideal_field_uses_single_ray_per_station,
        test_stations_from_rmcs,
        test_trim_border_fraction,
        test_plot_map_colorbar_and_rover_icon,
        test_study_navcam_metric_figures_and_filenames,
        test_ideal_ordering_holds_on_real_data,
        test_spacing_sweep_baseline_marker,
        test_numerics,
        test_metrics_end_to_end,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:                       # noqa: BLE001
            _check(t.__name__, False, f"EXCEPTION {type(exc).__name__}: {exc}")

    n_ok = sum(1 for _, ok, _ in _RESULTS if ok)
    if verbose:
        print("=" * 74)
        print("mppp self-test")
        print("=" * 74)
        for name, ok, msg in _RESULTS:
            flag = "PASS" if ok else "FAIL"
            print(f"[{flag}] {name}")
            if msg and not ok:
                print(f"       {msg}")
            elif msg and ok:
                print(f"       {msg}")
        print("-" * 74)
        print(f"{n_ok}/{len(_RESULTS)} checks passed")
    return n_ok == len(_RESULTS)
