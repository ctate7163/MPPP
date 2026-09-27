"""
mppp_error.cases -- the four reconstruction regimes.

FIXED       Fixed-baseline stereo only.  Each ground cell falls back to whichever
            single station gives the best normal precision from its own 24.4 cm
            (MCZ) or 42.4 cm (Navcam) baseline.  No cross-station correspondence.
            This is what operational Mars stereo delivers, and it is the honest
            baseline: without cross-station matching you cannot combine stations
            at all, because you have no correspondence to combine.

PESSIMISTIC SfM+MVS with poor cross-station matching.  eps_cross ~2 px,
            theta_max ~12 deg: dense NCC/SGM on low-texture, differently
            illuminated regolith.

OPTIMISTIC  SfM+MVS with good cross-station matching.  eps_cross ~0.6 px,
            theta_max ~45 deg: a learned dense matcher (LoFTR / DKM / RoMa
            class) coping with wide baselines.

LBS         Ideal SINGLE-PAIR long-baseline stereo ("long baseline stereo").
            The basic range-stereo equation evaluated with b = the LARGEST
            available station separation among stations that see the cell, at
            PERFECT (eps_intra) matching precision -- see `lbs_field`.
            Renamed from an earlier "IDEAL" label: it answers "what would one
            well-chosen long-baseline pair deliver," not "what is the best
            possible result."  LEFT OFF the main four-panel figure for now
            (still computed, still in case_table) and replaced there by:

IDEAL       Best of every possible long-baseline constraint, combined.  Not
            one pair -- ALL of them, properly fused.  See `ideal_field`.
            Since Fisher information adds linearly across independent ray
            observations, summing every visible station's own single ray
            precision ONCE (no double counting) automatically incorporates the
            joint triangulation constraint from every possible pair of
            stations simultaneously; there is no better combination achievable
            under this ray model.  It uses PERFECT (eps_intra) matching, no
            angular/tilt gating, and -- to isolate "how many independent
            baselines exist" from "how good is each station's own stereo" --
            ONE ray per station (its centre), not each station's own two eyes.
            Provably ideal >= lbs is false -- ideal <= lbs always (more
            information cannot increase variance); see
            `test_ideal_dominates_lbs_and_is_dominated_by_full_fusion`.

Why the FIXED baseline is best-single-station and not a fusion
-------------------------------------------------------------
Precision addition is associative, so "each station triangulates independently,
then average the results" is mathematically IDENTICAL to "put all rays in one
bundle".  The two regimes cannot differ that way.  What actually distinguishes
SfM+MVS from fixed-baseline stereo is that without cross-station correspondence
there is nothing to average: you get separate point clouds whose relative
registration is only as good as rover localisation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Tuple, List

from dataclasses import replace as _dc_replace

from .core import (Grid, Station, Instrument, ModelConfig, PoseModel,
                   PrecisionField, solve_precision_field, sym_inv3,
                   outer, _EPS)
from .metrics import compute_metrics

def run_three_cases(stations, grid, pose=None, **cfg_kw):
    """
    The three-case study used for the deck.

      fixed  every cell takes the CLOSEST visible station's own stereo.
             completeness = 1 where any station sees the cell (its own pair).
      sfm    link_mode='cross' with the MEASURED gate (A .4, theta_bar 4.3 deg,
             CV .36, L_eff 1.5 h) and measured redundancy (rho_inf .22,
             theta_rho .4 deg).  completeness = expected matched cross pairs.
      best   the ceiling: every station weighted equally, N_eff = N (no
             correlation), dLMST = 0, epsilon and completeness flat to 90 deg.
             Equivalent to full fusion with no gate; completeness = every
             possible cross pair.
    """
    from copy import deepcopy
    out = {}
    fx = solve_precision_field(grid, stations, ModelConfig(
        link_mode="nearest", store_station_dirs=True, use_correlation=False, **cfg_kw),
        _pose_for("fixed", pose, stations))
    fx.completeness = np.where(fx.n_vis > 0, 1.0, 0.0)
    fx.p_cross = np.zeros(grid.shape)
    out["fixed"] = (fx, compute_metrics(fx))

    A_, thb, cv_, tau_, _ = CASE_PARAMS["measured"]
    sf = solve_precision_field(grid, stations, ModelConfig(
        link_mode="cross", use_theta_gate=True, use_tau_gate=False, gate_form="powerlaw",
        gate_A=A_, theta_bar_deg=thb, gate_cv=cv_, tau_h=tau_,
        store_station_dirs=True, **cfg_kw), _pose_for("measured", pose, stations))
    out["sfm"] = (sf, compute_metrics(sf))

    # best: strip LMST so dL=0, no correlation, no gate
    st0 = deepcopy(list(stations))
    for st in st0:
        st.lmst_h = None
    bf = solve_precision_field(grid, st0, ModelConfig(
        link_mode="full", use_correlation=False, store_station_dirs=True, **cfg_kw),
        _pose_for("ideal", pose, st0))
    # every possible cross pair matches
    npair = np.zeros(grid.shape)
    for i in range(len(st0)):
        for j in range(i + 1, len(st0)):
            both = bf.station_vis[i] & bf.station_vis[j]
            npair += np.where(both, len(st0[i].instrument.eye_offsets)
                              * len(st0[j].instrument.eye_offsets), 0.0)
    bf.completeness = npair
    bf.p_cross = np.where(npair > 0, 1.0, 0.0)
    out["best"] = (bf, compute_metrics(bf))
    return out


__all__ = ["CASE_PARAMS", "CASE_POSE", "run_four_cases", "run_three_cases", "lbs_field", "ideal_field",
           "case_table"]


#: name -> (eps_cross_px, theta_max_deg, tau_max, description)
#: name -> (A, theta_bar_deg, CV, tau_h, description)
#: All three share the measured functional form; only the numbers differ.
#:   pessimistic  classical matcher at the worst archive site, dL uncontrolled
#:   measured     pooled five-site Navcam archive (Metashape/COLMAP SIFT)
#:   optimistic   learned matcher with dL < 1 h -- UNMEASURED placeholder
CASE_PARAMS: Dict[str, Tuple[float, float, float, float, str]] = {
    "pessimistic": (0.4, 2.4, 0.50, 2.3, "classical matcher, worst site (Rockytop), dL uncontrolled"),
    "measured":    (0.4, 4.3, 0.36, 2.3, "pooled five-site Navcam archive"),
    "optimistic":  (0.7, 8.6, 0.36, 2.3, "learned matcher, dL<1 h (placeholder)"),
}


def _center_ray_clones(stations: Sequence[Station]) -> List[Station]:
    """
    Clone every station with its eyes collapsed to a single centre ray
    (eye_offsets=(0.0,)).

    Used by BOTH `lbs_field` and `ideal_field` so their visibility tests and
    geometry are self-consistent by construction.  Originally they were not:
    `lbs_field` picked its best pair using two-eye visibility (from the real
    stereo geometry) but then computed precision from the station-CENTRE ray.
    At an occlusion boundary one eye can see past the mask while the exact
    centre ray cannot, so a station could be credited as visible by the
    eligibility test and then contribute geometry that test would have
    rejected -- `test_ideal_dominates_lbs_and_is_dominated_by_full_fusion`
    caught this as a real (if rare, ~0.1% of cells) violation of the "more
    information cannot increase variance" invariant. Building both functions
    from the same clones removes the inconsistency at its source rather than
    patching each symptom.
    """
    out = []
    for st in stations:
        inst = st.instrument
        center_inst = Instrument(
            name=inst.name + "-ctr", ifov_rad=inst.ifov_rad,
            baseline_m=inst.baseline_m, eps_intra_px=inst.eps_intra_px,
            eye_offsets=(0.0,), precision_convention=inst.precision_convention)
        out.append(_dc_replace(st, instrument=center_inst))
    return out


def lbs_field(grid: Grid, stations: Sequence[Station],
                              config: Optional[ModelConfig] = None,
                              pose: Optional[PoseModel] = None) -> PrecisionField:
    """
    Ideal single-pair long-baseline stereo reference.

    For every cell, find the widest-separated pair of stations that both see
    it -- both eligibility and geometry now come from `_center_ray_clones`, so
    "widest pair that sees it" and "the ray used to compute its precision"
    agree by construction (see that function's docstring for the bug this
    fixes) -- and treat each as ONE camera (its own centre, its own range),
    i.e. classic two-camera long-baseline photogrammetry, ignoring each
    station's own intra-pair stereo baseline.  The precision is the EXACT
    rank-2 sum of the two ray precisions:

        Lambda = (I - u_i u_i^T)/sigma_i^2 + (I - u_j u_j^T)/sigma_j^2,
        sigma_k = eps * psi * r_k   (each station's OWN range)

    NOT the small-angle disparity-formula shortcut (average range, symmetric
    bisector assumption).  That shortcut is exact only when b << r and the two
    ranges are nearly equal (verified against it in `test_two_ray_analytic`,
    which uses exactly that symmetric regime).  Applied outside it -- large
    parallactic angle, or the two stations at substantially different ranges
    from the point, both common for rover geometry at moderate range -- the
    shortcut UNDERESTIMATES sigma_n, sometimes enough to beat full network
    fusion, which is unphysical (fusion cannot destroy information).  Measured
    case: beta=25 deg, r1=4.74 m, r2=7.11 m gave the shortcut 0.104 cm against
    an exact 0.134 cm, a fictitious 23% improvement.

    The exact form reduces to the small-angle shortcut in its valid regime (see
    `test_two_ray_analytic`) and is otherwise a strict improvement, so this is
    not a behaviour change within the regime the shortcut was designed for --
    only outside it.
    """
    cfg = config or ModelConfig(store_station_dirs=True)
    cfg = ModelConfig(**{**cfg.__dict__, "store_station_dirs": True,
                         "link_mode": "full"})
    center_stations = _center_ray_clones(stations)
    base = solve_precision_field(grid, center_stations, cfg, pose or PoseModel())

    u = base.station_dirs
    vis = base.station_vis
    r = base.station_range
    Nst, Ny, Nx = vis.shape
    P, N = grid.points()

    best_b = np.zeros((Ny, Nx))
    best_i = np.full((Ny, Nx), -1, dtype=int)
    best_j = np.full((Ny, Nx), -1, dtype=int)
    for i in range(Nst):
        for j in range(i + 1, Nst):
            both = vis[i] & vis[j]
            if not both.any():
                continue
            b = float(np.linalg.norm(center_stations[i].xyz - center_stations[j].xyz))
            upd = both & (b > best_b)
            best_b = np.where(upd, b, best_b)
            best_i = np.where(upd, i, best_i)
            best_j = np.where(upd, j, best_j)

    Lam = np.zeros((Ny, Nx, 3, 3))
    ok = best_i >= 0
    if ok.any():
        ii, jj = best_i[ok], best_j[ok]
        cell = np.nonzero(ok)
        ui = u[ii, cell[0], cell[1]]
        uj = u[jj, cell[0], cell[1]]
        ri = r[ii, cell[0], cell[1]]
        rj = r[jj, cell[0], cell[1]]

        psi_i = np.array([center_stations[k].instrument.ifov_rad for k in ii])
        psi_j = np.array([center_stations[k].instrument.ifov_rad for k in jj])
        eps_i = np.array([center_stations[k].instrument.sigma_x_px for k in ii])
        eps_j = np.array([center_stations[k].instrument.sigma_x_px for k in jj])
        sig_i = eps_i * psi_i * ri
        sig_j = eps_j * psi_j * rj

        I3 = np.eye(3)
        L = ((I3 - outer(ui)) / (sig_i ** 2)[:, None, None]
             + (I3 - outer(uj)) / (sig_j ** 2)[:, None, None])
        Lam[ok] = L

    prior = np.eye(3) / (cfg.prior_sigma_m ** 2)
    Sigma = sym_inv3(Lam + prior)
    unconstrained = ~ok

    out = PrecisionField(
        grid=grid, Sigma=Sigma, normals=N, n_vis=(ok * 2).astype(int),
        n_rays=(ok * 2).astype(int), range_min=base.range_min,
        cos_e_ref=base.cos_e_ref, sigma_n_single=base.sigma_n_single,
        station_vis=vis, station_dirs=u, link_weights=base.link_weights,
        station_cos_e=base.station_cos_e, station_range=r,
        stations=list(stations), config=cfg, pose=pose or PoseModel(),
        unconstrained=unconstrained)
    out.effective_baseline = np.where(ok, best_b, np.nan)
    return out


def ideal_field(grid: Grid, stations: Sequence[Station],
                config: Optional[ModelConfig] = None,
                pose: Optional[PoseModel] = None) -> PrecisionField:
    """
    Best-of-every-possible-pair reference: properly fuse (no double counting)
    the single station-centre-ray precision from EVERY visible station
    simultaneously, at perfect matching, no gating.

    Implementation note: rather than looping over pairs by hand (which risks
    the double-counting bug this docstring warns against -- summing raw
    Fisher information over all C(N,2) pairs counts each station's own ray
    (N-1) times, wildly over-stating precision), this clones each station with
    its eyes collapsed to a single centre ray (eye_offsets=(0.0,)) and hands
    the clones to the SAME exact-geometry solver (`solve_precision_field`,
    link_mode='full') already validated against a naive independent reference
    in `test_vectorised_matches_reference`.  Reusing tested machinery here
    rather than a fresh derivation is deliberate: it is the safer path to a
    provably-correct "sum every ray once."

    A station contributes only if `cfg.cos_e_min`/occlusion allow it, same as
    every other case -- IDEAL still respects real visibility, just not real
    matching degradation.
    """
    cfg = config or ModelConfig()
    center_stations = _center_ray_clones(stations)

    ideal_cfg = ModelConfig(
        apply_occlusion=cfg.apply_occlusion, cos_e_min=cfg.cos_e_min,
        link_mode="full", store_station_dirs=True,
        prior_sigma_m=cfg.prior_sigma_m, on_no_visibility=cfg.on_no_visibility,
        use_correlation=cfg.use_correlation, correlation_kernel=cfg.correlation_kernel,
        theta_c_deg=cfg.theta_c_deg, rho_inf=cfg.rho_inf)
    return solve_precision_field(grid, center_stations, ideal_cfg,
                                 pose or PoseModel())


def _apply_fixed_fallback(gated: PrecisionField, fixed: PrecisionField,
                          m_gated: dict, m_fixed: dict):
    """
    A station's OWN intra-station stereo needs no cross-station tie.

    The link weight w_i scales a station's whole precision contribution, which
    is right for the cross-station part but wrong for its own fixed-baseline
    stereo.  Where the tie is weak this drove the gated result BELOW the
    fixed-baseline result, which no real pipeline would ever deliver: it would
    simply fall back to per-station stereo.

    So the delivered product is the better of (gated fusion, best single
    station), chosen per cell.

    CAVEAT worth surfacing rather than hiding: where the fallback is taken, the
    terrain is well MEASURED but poorly REGISTERED -- that station's cloud
    floats relative to the rest of the network, because there is no tie to place
    it.  `fallback_taken` flags exactly those cells.
    """
    sg, sf = m_gated["sigma_n"], m_fixed["sigma_n"]
    take = np.isfinite(sf) & (~np.isfinite(sg) | (sf < sg))
    S = np.where(take[..., None, None], fixed.Sigma, gated.Sigma)
    out = PrecisionField(**{**gated.__dict__, "Sigma": S,
                            "unconstrained": gated.unconstrained & fixed.unconstrained})
    out.fallback_taken = take
    return out


#: Per-case pose model. Each regime is registered differently, so each gets
#: the pose error its own registration path actually incurs:
#:   FIXED       telemetry/VO, ~3% of PATH DRIVEN -- products are placed in the
#:               common frame by rover localisation, and the path is what
#:               drifts (18 m apart can be 167 m driven).
#:   SfM+MVS     ~0.2% of straight-line BASELINE to the anchor -- relative pose
#:               comes from bundle adjustment on the cross-station ties.
#:   LBS/IDEAL   zero -- the ceiling, geometry only.
# H2 dry run on the archive (2026-09-09): with bundle-adjusted poses the ray
# model alone reproduces sigma_z to 1-4 %, so intra-cluster pose error after BA
# is negligible -> SfM cases carry no pose term.  FIXED carries the MEASURED
# telemetry registration error (0.15-0.5 m) as a constant.
# FIXED is the INTRINSIC single-station stereo precision (pose none): that is
# the product's own error map.  Telemetry registration (0.15-0.5 m measured) is
# a separate, spatially uniform number and is reported alongside, not folded in
# -- folding it in swamped the map (41 cm everywhere) and made "improvement"
# a registration ratio (~100x, uniform) instead of a geometry one.
CASE_POSE = {"fixed": "none", "pessimistic": "none", "measured": "none", "optimistic": "none",
             "lbs": "none", "ideal": "none", "full_fusion": "none"}


def _pose_for(case: str, pose, stations):
    """Resolve the per-case pose model unless the caller supplied one."""
    if pose is not None:
        return pose
    pm = PoseModel(mode=CASE_POSE.get(case, "none"))
    return pm.bind_anchor(stations)


def run_four_cases(stations: Sequence[Station], grid: Grid,
                   pose: Optional[PoseModel] = None,
                   eps_cross_override: Optional[Dict[str, float]] = None,
                   include_full: bool = True,
                   fallback_to_fixed: bool = True,
                   **cfg_kw) -> Dict[str, Tuple[PrecisionField, dict]]:
    """
    Solve all four regimes on identical geometry.

    eps_cross_override lets you replace the default case parameters, e.g.
    {"optimistic": 0.0} for perfect cross-station matching or
    {"optimistic": eps_intra} for cross-matching as good as intra-matching.

    fallback_to_fixed : deliver max(SfM+MVS, fixed-baseline stereo) per cell,
    which is what a real pipeline does.  See _apply_fixed_fallback.
    """
    out: Dict[str, Tuple[PrecisionField, dict]] = {}

    f = solve_precision_field(grid, stations, ModelConfig(
        link_mode="ops", store_station_dirs=True, **cfg_kw),
        _pose_for("fixed", pose, stations))
    out["fixed"] = (f, compute_metrics(f))

    for name, (A_, thb, cv_, tau_, _) in CASE_PARAMS.items():
        # tau gate dropped: on terrain flat over ~5 m viewed from one mast
        # height it is redundant with theta, and it was one parameter nobody
        # could interpret.
        cfg = ModelConfig(link_mode="cross", use_theta_gate=True,
                          use_tau_gate=False, gate_form="powerlaw",
                          gate_A=A_, theta_bar_deg=thb, gate_cv=cv_, tau_h=tau_,
                          store_station_dirs=True, **cfg_kw)
        f = solve_precision_field(grid, stations, cfg, _pose_for(name, pose, stations))
        m = compute_metrics(f)
        # NOTE: the fixed-fallback is no longer applied.  It existed to paper
        # over link_mode='gated' scaling a station's own stereo by its tie
        # strength; under 'cross' the fused result can never be worse than the
        # best single station, so the rule is unreachable by construction and
        # keeping it would only hide a regression.
        out[name] = (f, m)

    f = lbs_field(grid, stations,
                 ModelConfig(store_station_dirs=True, **cfg_kw),
                 _pose_for("lbs", pose, stations))
    out["lbs"] = (f, compute_metrics(f))

    f = ideal_field(grid, stations,
                    ModelConfig(store_station_dirs=True, **cfg_kw),
                    _pose_for("ideal", pose, stations))
    out["ideal"] = (f, compute_metrics(f))

    if include_full:
        f = solve_precision_field(grid, stations, ModelConfig(
            link_mode="full", store_station_dirs=True, **cfg_kw),
            _pose_for("full_fusion", pose, stations))
        out["full_fusion"] = (f, compute_metrics(f))
    return out


def case_table(res: Dict[str, Tuple[PrecisionField, dict]],
               reference: str = "fixed") -> str:
    """Formatted comparison table, all cases relative to `reference`."""
    ref = res[reference][1]["sigma_n"]
    order = [k for k in ("fixed", "pessimistic", "measured", "optimistic", "lbs",
                         "ideal", "full_fusion") if k in res]
    lines = [f"{'case':<14} {'gate':>6} {'thb/CV':>9} {'sig_n[cm]':>10} "
             f"{'G_n':>7} {'x vs FIXED':>11} {'kappa':>8} {'cov<2cm':>8}",
             "-" * 76]
    for k in order:
        f, m = res[k]
        s = m["sigma_n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            imp = np.where(np.isfinite(s) & (s > 0), ref / s, np.nan)
        ec, th = "--", "--"
        if k in CASE_PARAMS:
            ec = f"A={f.config.gate_A:.1f}"
            th = f"{f.config.theta_bar_deg:.1f}/{f.config.gate_cv:.2f}"
        cov = np.nanmean((s < 0.02).astype(float))
        lines.append(
            f"{k.upper():<14} {ec:>6} {th:>7} {np.nanmedian(s)*100:10.3f} "
            f"{np.nanmedian(m['G_n']):7.3f} {np.nanmedian(imp):11.2f} "
            f"{np.nanmedian(m['kappa']):8.2f} {cov:8.3f}")
    return "\n".join(lines)
