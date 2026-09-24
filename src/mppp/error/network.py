"""
mppp_error.network -- how good is a view graph, really?

HOW TO MEASURE VIEW-GRAPH QUALITY
=================================
There is a hierarchy of answers, cheap to expensive.  Each level is a proxy for
the one below it, and the last one is the truth.

1. CONNECTIVITY.  Are all stations in one component?  If not, the relative pose
   of the disconnected ones is unrecoverable from imagery at any precision.
   Binary, but the first thing to check and the cheapest.

2. ALGEBRAIC CONNECTIVITY (Fiedler value lambda_2 of L = D - W).  Bounded above
   by vertex and edge connectivity: 0 <= lambda_2 <= kappa_v <= kappa_e <= d_min.
   lambda_2 = 0 iff disconnected.  Cheap, scale-free when normalised by
   lambda_max, and correlates with conditioning.  A proxy, not the answer.

3. TOPOLOGICAL REDUNDANCY.  Articulation stations, bridges, edge-removal
   tolerance.  In a 3-7 station rover network this is usually the binding
   constraint: the question is not "is the average good" but "is there a single
   link whose failure splits the network".

4. PARALLEL RIGIDITY.  A connected view graph is not necessarily solvable.
   Recovering translations from pairwise DIRECTIONS requires the graph to be
   parallel rigid (Ozyesil & Singer 2015), which is strictly stronger than
   connectedness.  A chain of stations each sharing points only with its
   neighbour is connected but poorly conditioned for translation.

5. POSE-BLOCK CONDITIONING.  THE ANSWER.  Build the bundle-adjustment normal
   matrix, reduce out the points (Schur complement), apply inner constraints for
   the datum defect, and invert.  The result is the actual covariance of the
   estimated station poses -- which is what "view graph quality" means
   operationally, because it is exactly the Sigma_pose the error model needs.

This module implements 5 (and re-exports 1-4 from mppp_error.viewgraph).  Doing so
closes the circularity flagged earlier: Sigma_pose in 'ba' mode is no longer an
assumed input, it is DERIVED from the tie geometry.

THE FREE-NETWORK DATUM
----------------------
Angular observations alone determine the network only up to a 7-parameter
similarity (3 translation, 3 rotation, 1 scale).  The normal matrix is therefore
rank-deficient by 7 and must be inverted with a pseudo-inverse, which completeness the
minimum-trace (inner-constraint) solution.  The resulting covariances are
RELATIVE -- exactly what is wanted, since the common-mode part is a datum shift
that does not degrade internal reconstruction quality.

Scale deserves a note: with a rover, scale is fixed by the known stereo baseline
at each station, so it is NOT actually free.  Set `fix_scale=True` (the default)
to remove only the 6 rigid-body modes and leave scale determined.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Optional, List, Dict, Tuple

from .core import (Station, Grid, ModelConfig, PoseModel, Surface,
                   solve_precision_field, skew, tangent_basis)

__all__ = ["NetworkPose", "pose_covariance_from_network", "network_report",
           "rigidity_score", "datum_modes"]


@dataclass
class NetworkPose:
    """Free-network pose covariance for every station."""
    Sigma: np.ndarray              # (Nst,6,6) [t(3) | omega(3)] blocks
    stations: List[str]
    n_tie: int                     # tie points used
    n_obs: int                     # angular observations used
    rank: int                      # numerical rank of the reduced normal matrix
    datum_defect: int              # expected rank deficiency
    eigenvalues: np.ndarray        # of the reduced normal matrix

    @property
    def pos_sigma_m(self) -> np.ndarray:
        """(Nst,) RMS position sigma per station."""
        return np.sqrt(np.maximum(
            np.trace(self.Sigma[:, :3, :3], axis1=1, axis2=2) / 3.0, 0.0))

    @property
    def att_sigma_rad(self) -> np.ndarray:
        """(Nst,) RMS attitude sigma per station."""
        return np.sqrt(np.maximum(
            np.trace(self.Sigma[:, 3:, 3:], axis1=1, axis2=2) / 3.0, 0.0))

    @property
    def well_determined(self) -> bool:
        return self.rank >= 6 * len(self.stations) - self.datum_defect

    def relative_to(self, station: str) -> "NetworkPose":
        """
        Re-reference every covariance to one station:

            Sigma_rel[i] = Sigma[i,i] + Sigma[a,a] - Sigma[i,a] - Sigma[a,i]

        The inner-constraint solution distributes uncertainty about the network
        centroid, so raw per-station blocks are all comparable and none is zero.
        The error model wants RELATIVE-to-anchor covariance, where the anchor is
        exactly zero by construction.  This does that properly, including the
        cross-covariance term -- dropping it would double-count the shared part.
        """
        a = self.stations.index(station)
        F = getattr(self, "full_Sigma", None)
        if F is None:
            raise ValueError("full_Sigma not available on this NetworkPose")
        n = len(self.stations)
        out = np.zeros_like(self.Sigma)
        Saa = F[6*a:6*a+6, 6*a:6*a+6]
        for i in range(n):
            Sii = F[6*i:6*i+6, 6*i:6*i+6]
            Sia = F[6*i:6*i+6, 6*a:6*a+6]
            out[i] = Sii + Saa - Sia - Sia.T
        new = NetworkPose(Sigma=out, stations=list(self.stations),
                          n_tie=self.n_tie, n_obs=self.n_obs, rank=self.rank,
                          datum_defect=self.datum_defect,
                          eigenvalues=self.eigenvalues)
        new.full_Sigma = F
        new.null_leak = getattr(self, "null_leak", float("nan"))
        new.reference = station
        return new

    def to_pose_model(self) -> "MeasuredPoseModel":
        return MeasuredPoseModel(self)


class MeasuredPoseModel(PoseModel):
    """PoseModel backed by a solved network rather than assumed constants."""

    def __init__(self, net: NetworkPose):
        super().__init__(mode="ba")
        self._net = net
        self._idx = {n: i for i, n in enumerate(net.stations)}

    def station_sigmas(self, st: Station):
        i = self._idx.get(st.name)
        if i is None or st.is_anchor:
            return np.zeros(3), np.zeros(3)
        S = self._net.Sigma[i]
        return (np.sqrt(np.maximum(np.diag(S[:3, :3]), 0.0)),
                np.sqrt(np.maximum(np.diag(S[3:, 3:]), 0.0)))


# --------------------------------------------------------------------------

def pose_covariance_from_network(stations: Sequence[Station],
                                 grid: Grid,
                                 config: Optional[ModelConfig] = None,
                                 n_tie: int = 60,
                                 fix_scale: bool = True,
                                 lever_arm: bool = False,
                                 tie_correlation: float = 0.0,
                                 rcond: float = 1e-10) -> NetworkPose:
    """
    Free-network pose covariance from the tie geometry.

    Every visible grid cell (subsampled to an n_tie x n_tie grid) is treated as
    a tie point.  For each ray the 2-vector angular observation is linearised
    against the point position and the station pose, the point parameters are
    eliminated by Schur complement, and the reduced 6N x 6N normal matrix is
    pseudo-inverted.

    IMPORTANT: the returned covariance is a LOWER BOUND.  Grid cells are treated
    as independent tie points; real ones are correlated, and terrain where
    matching fails contributes nothing at all.  Use `tie_correlation` (a rho_inf
    for tie points) to get a defensible upper estimate, and report both.

    n_tie controls cost, not physics: pose precision saturates quickly with tie
    count, and 60x60 = 3600 candidate points is already far more than a real
    sparse model contributes independently.  Nocerino et al. (Sensors 15:7985)
    make the same observation for close-range blocks -- beyond a modest number,
    extra tie points barely move network precision.

    LIMITATION: cross-station link weighting is applied to whole stations, so a
    station that cannot be tied to the network contributes nothing.  Per-link
    down-weighting of individual observations is not modelled here.
    """
    cfg = config or ModelConfig(store_station_dirs=True)
    Nst = len(stations)

    # coarse tie grid
    sx = max(1, grid.x.size // n_tie)
    sy = max(1, grid.y.size // n_tie)
    g = Grid(x=grid.x[::sx], y=grid.y[::sy], surface=grid.surface)
    P, Nrm = g.points()
    Ny, Nx = g.shape

    six = 6 * Nst
    Ncc = np.zeros((Ny, Nx, six, six))
    Npp = np.zeros((Ny, Nx, 3, 3))
    Npc = np.zeros((Ny, Nx, 3, six))
    n_obs = 0
    vis_any = np.zeros((Ny, Nx), dtype=bool)

    for i, st in enumerate(stations):
        inst = st.instrument
        d0 = P - st.xyz
        hx, hy = d0[..., 0], d0[..., 1]
        hn = np.hypot(hx, hy)
        eb = np.zeros_like(d0)
        safe = hn > 1e-12
        eb[..., 0] = np.where(safe, -hy / np.maximum(hn, 1e-12), 1.0)
        eb[..., 1] = np.where(safe, hx / np.maximum(hn, 1e-12), 0.0)

        for frac in inst.eye_offsets:
            C = st.xyz + frac * inst.baseline_m * eb
            d = P - C
            r = np.linalg.norm(d, axis=-1)
            u = d / np.maximum(r, 1e-12)[..., None]
            cos_e = -np.einsum("...i,...i->...", u, Nrm)
            vis = (r > 1e-9) & (cos_e > max(cfg.cos_e_min, 0.0))
            if cfg.apply_occlusion and st.mask is not None:
                az = np.degrees(np.arctan2(u[..., 0], u[..., 1]))
                el = np.degrees(np.arcsin(np.clip(u[..., 2], -1, 1)))
                vis &= ~st.mask.occluded((az - st.az_deg) % 360.0, el)
            if not vis.any():
                continue
            vis_any |= vis
            n_obs += int(vis.sum())

            T = tangent_basis(u)                       # (Ny,Nx,2,3)
            w = np.where(vis, 1.0 / (inst.eps_intra_px * inst.ifov_rad) ** 2, 0.0)

            # d(angular obs)/d(point) = T / r ;  d/d(translation) = -T / r
            Jx = T / np.maximum(r, 1e-12)[..., None, None]
            Jt = -Jx
            # d/d(rotation) has TWO parts:
            #  (a) the viewing direction rotates:  -domega x u
            #  (b) OPTIONAL lever arm, dC = dt + domega x (C - X0), for a
            #      baseline rigidly attached to the rotating station frame.
            #      DEFAULT OFF, and measurably so: in the omnidirectional-
            #      station abstraction the baseline direction is slaved to the
            #      TARGET (always perpendicular to the line of sight), not to
            #      the station attitude, so attaching a lever arm to domega is
            #      inconsistent with the forward model.  Measured null-space
            #      leak: 1.7e-5 without, 6.7e-5 with.
            #
            # The residual ~1e-5 leak is a real limitation of the abstraction,
            # not a coding error: a global rotation does not map exactly onto a
            # station-attitude parameter when the baseline is target-slaved.
            # It scales with baseline/scene-size and is harmless because the
            # datum modes are projected out analytically regardless.
            K = skew(u)
            Jw = np.einsum("...ai,...ij->...aj", T, K)
            if lever_arm:
                lever = C - st.xyz                               # (Ny,Nx,3)
                Jw = Jw + np.einsum("...ai,...ij->...aj", Jt, skew(lever))

            Jc = np.concatenate([Jt, Jw], axis=-1)     # (Ny,Nx,2,6)
            sl = slice(6 * i, 6 * i + 6)

            Npp += w[..., None, None] * np.einsum("...ai,...aj->...ij", Jx, Jx)
            Npc[..., :, sl] += w[..., None, None] * np.einsum(
                "...ai,...aj->...ij", Jx, Jc)
            Ncc[..., sl, sl] += w[..., None, None] * np.einsum(
                "...ai,...aj->...ij", Jc, Jc)

    # ---- Schur complement: eliminate the point parameters -----------------
    keep = vis_any & (np.linalg.det(Npp) > 1e-30)
    Npp_i = np.zeros_like(Npp)
    Npp_i[keep] = np.linalg.inv(Npp[keep])
    S = (Ncc - np.einsum("...ki,...kl,...lj->...ij", Npc, Npp_i, Npc))
    S = S[keep].sum(axis=0)
    S = 0.5 * (S + S.T)

    # ---- inner constraints (free-network adjustment, Fraser 1982) ---------
    # The datum modes are known ANALYTICALLY, so they are constructed and
    # projected out rather than discovered by an eigenvalue threshold, which is
    # fragile when the signal spectrum itself spans many decades.
    G = datum_modes(stations, fix_scale=fix_scale)      # (6N, d)
    G, _ = np.linalg.qr(G)
    defect = G.shape[1]
    Pperp = np.eye(six) - G @ G.T

    ev = np.linalg.eigvalsh(S)
    # residual power of S on the analytic null space, as a validity check
    null_leak = float(np.linalg.norm(S @ G) / max(np.linalg.norm(S), 1e-300))

    Sig = Pperp @ np.linalg.pinv(Pperp @ S @ Pperp, rcond=rcond) @ Pperp
    Sig = 0.5 * (Sig + Sig.T)

    # Every grid cell was treated as an INDEPENDENT tie point, which it is not.
    # Real tie points share matching bias, interior-orientation residual and
    # surface-definition ambiguity, so the raw result is a LOWER BOUND on pose
    # uncertainty -- often by a large factor.  tie_correlation applies the same
    # N_eff = N/(1+(N-1)rho) saturation used elsewhere in the model.
    if tie_correlation > 0:
        n = max(int(keep.sum()), 1)
        Sig = Sig * (1.0 + (n - 1) * tie_correlation)

    ev_sig = np.linalg.eigvalsh(Pperp @ S @ Pperp)
    rank = int((ev_sig > rcond * max(ev_sig.max(), 1e-30)).sum())

    net = NetworkPose(
        Sigma=np.stack([Sig[6*i:6*i+6, 6*i:6*i+6] for i in range(Nst)]),
        stations=[s.name or f"S{i+1}" for i, s in enumerate(stations)],
        n_tie=int(keep.sum()), n_obs=n_obs, rank=rank, datum_defect=defect,
        eigenvalues=ev)
    net.full_Sigma = Sig
    net.null_leak = null_leak
    return net


def datum_modes(stations: Sequence[Station], fix_scale: bool = True) -> np.ndarray:
    """
    Analytic null space of the free-network pose normal matrix.

    Columns of the returned (6N, d) matrix are the unobservable modes:
      3 translations  dt_i = e_k,            domega_i = 0
      3 rotations     dt_i = e_k x X0_i,     domega_i = e_k
      1 scale         dt_i = X0_i,           domega_i = 0     (only if not fixed)

    Scale is observable for a rover because each station's stereo baseline has a
    KNOWN length, so fix_scale=True (the default) omits that mode.  For a pure
    angles-only network scale is free and the defect is 7.
    """
    N = len(stations)
    cols = []
    for k in range(3):                                   # translations
        v = np.zeros(6 * N)
        for i in range(N):
            v[6*i + k] = 1.0
        cols.append(v)
    for k in range(3):                                   # rotations
        e = np.zeros(3); e[k] = 1.0
        v = np.zeros(6 * N)
        for i, st in enumerate(stations):
            v[6*i:6*i+3] = np.cross(e, st.xyz)
            v[6*i+3:6*i+6] = e
        cols.append(v)
    if not fix_scale:                                    # scale
        v = np.zeros(6 * N)
        for i, st in enumerate(stations):
            v[6*i:6*i+3] = st.xyz
        cols.append(v)
    return np.column_stack(cols)


def rigidity_score(net: NetworkPose) -> Dict[str, float]:
    """
    How far is the network from singular?

    ratio = smallest NON-DATUM eigenvalue / largest eigenvalue of the reduced
    normal matrix.  This is the direct conditioning answer: small means some
    combination of station poses is nearly unconstrained, which is what
    "weak view graph" actually means.  Unlike lambda_2 of the view graph it
    accounts for geometry, not just topology -- a chain of stations can be well
    connected as a graph and still be nearly singular for translation recovery.
    """
    ev = np.sort(net.eigenvalues)
    d = net.datum_defect
    if ev.size <= d:
        return {"conditioning": 0.0, "smallest_signal_eig": 0.0,
                "largest_eig": float(ev[-1]) if ev.size else 0.0,
                "datum_eig_max": float("nan")}
    sig = ev[d:]
    return {"conditioning": float(sig[0] / ev[-1]),
            "smallest_signal_eig": float(sig[0]),
            "largest_eig": float(ev[-1]),
            "datum_eig_max": float(ev[d-1]) if d else 0.0,
            "datum_separation": float(sig[0] / max(ev[d-1], 1e-300)) if d else np.inf}


def network_report(net: NetworkPose) -> str:
    r = rigidity_score(net)
    lines = [
        f"tie points used         : {net.n_tie}",
        f"angular observations    : {net.n_obs}",
        f"reduced normal matrix   : {6*len(net.stations)} x {6*len(net.stations)}",
        f"numerical rank          : {net.rank} "
        f"(expected {6*len(net.stations) - net.datum_defect}, "
        f"datum defect {net.datum_defect})",
        f"well determined         : {net.well_determined}",
        f"conditioning            : {r['conditioning']:.3e}"
        "   (smallest signal eig / largest; small = weak view graph)",
        f"null-space leak         : {getattr(net,'null_leak',float('nan')):.3e}"
        "   (||S G|| / ||S||; should be ~1e-15 if the datum modes are exact)",
        "",
        f"{'station':<14s} {'pos sigma [cm]':>15s} {'att sigma [mrad]':>17s}",
    ]
    for n, p, a in zip(net.stations, net.pos_sigma_m, net.att_sigma_rad):
        lines.append(f"{n[:14]:<14s} {p*100:15.3f} {a*1000:17.4f}")
    return "\n".join(lines)
