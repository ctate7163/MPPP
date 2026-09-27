"""
mppp_error.viewgraph -- station view graph and its strength.

The view graph asks a different question from the precision field.  The
precision field asks "how well is each ground point determined?"  The view graph
asks "is the NETWORK rigid?" -- can the relative pose of every station be
recovered at all, and how robustly.  A network can have excellent per-point
geometry and still be one weak link away from splitting into two unconnected
components whose relative pose is unknown.

Edge weight
-----------
    W_ij = (fraction of the grid visible from BOTH i and j)
           x (mean cross-station link weight w_ij over that common footprint)

The first factor is shared coverage; the second is matchability.  Their product
is "effective shared, matchable coverage", which is what a feature matcher
actually has to work with.

Strength metrics
----------------
algebraic_connectivity (Fiedler value, lambda_2 of the Laplacian L = D - W)
    The standard spectral measure of graph robustness, bounded above by node and
    edge connectivity: 0 <= lambda_2 <= kappa_v <= kappa_e <= d_min.
    lambda_2 = 0 means the graph is DISCONNECTED -- some stations cannot be tied
    to the others at all, and their relative pose is unrecoverable from imagery.
    Larger is more rigid.  Used in recent SfM work to characterise view-graph
    difficulty.

normalised_fiedler = lambda_2 / lambda_max
    Scale-free version, comparable between sites with different station counts
    and different absolute overlap.

min_degree / weakest_station
    The weakest-tied station.  In a small rover network (3-7 stations) this is
    usually the binding constraint, not the average.

articulation_stations
    Stations whose removal would disconnect the graph.  A single-point-of-
    failure list: if matching fails at one of these, the network splits.

redundancy
    Fraction of edges that could be removed while keeping the graph connected.
    Low redundancy means no margin for a failed match.

CAVEAT
------
This is a PREDICTED graph, built from geometry and a matchability model.  The
measured graph from an actual SfM run (see mppp_error.colmap.measured_view_graph)
will differ, and the difference is exactly the calibration signal for
eps_cross and theta_max.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field as _dcfield
from typing import List, Optional, Dict, Sequence

from .core import (PrecisionField, ModelConfig, _link_weights, gate_powerlaw,
                   illumination_gate)

__all__ = ["ViewGraph", "build_view_graph", "graph_report"]


@dataclass
class ViewGraph:
    W: np.ndarray                 # (Nst,Nst) symmetric edge weights, zero diagonal
    names: List[str]
    shared_fraction: np.ndarray   # (Nst,Nst) common visible grid fraction
    mean_theta_deg: np.ndarray    # (Nst,Nst) mean convergence angle over common cells
    mean_w: np.ndarray            # (Nst,Nst) mean matchability weight


    # ---- spectral ---------------------------------------------------------
    @property
    def laplacian(self) -> np.ndarray:
        return np.diag(self.W.sum(axis=1)) - self.W

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.laplacian)

    @property
    def algebraic_connectivity(self) -> float:
        ev = self.eigenvalues
        return float(ev[1]) if ev.size > 1 else 0.0

    @property
    def normalised_fiedler(self) -> float:
        ev = self.eigenvalues
        return float(ev[1] / ev[-1]) if ev.size > 1 and ev[-1] > 0 else 0.0

    @property
    def degrees(self) -> np.ndarray:
        return self.W.sum(axis=1)

    @property
    def connected(self) -> bool:
        return self.algebraic_connectivity > 1e-9

    @property
    def weakest_station(self) -> str:
        return self.names[int(np.argmin(self.degrees))]

    #: Topology metrics need a MEANINGFUL edge threshold, not 1e-9.  An edge
    #: whose weight is 0.1% of the strongest edge is not a usable tie, and
    #: counting it makes every dense graph look perfectly redundant.  The
    #: default is 5% of the strongest edge; set absolute values via `thresh`.
    rel_threshold: float = 0.05

    def _thr(self, thresh: Optional[float] = None) -> float:
        if thresh is not None:
            return thresh
        mx = float(self.W.max()) if self.W.size else 0.0
        return self.rel_threshold * mx

    def components(self, thresh: Optional[float] = None) -> List[List[int]]:
        thresh = self._thr(thresh)
        n = len(self.names)
        seen = set()
        out = []
        for s in range(n):
            if s in seen:
                continue
            stack, comp = [s], []
            while stack:
                v = stack.pop()
                if v in seen:
                    continue
                seen.add(v); comp.append(v)
                stack.extend(int(u) for u in np.where(self.W[v] > thresh)[0]
                             if u not in seen)
            out.append(sorted(comp))
        return out

    def articulation_stations(self, thresh: Optional[float] = None) -> List[str]:
        thresh = self._thr(thresh)
        base = len(self.components(thresh))
        out = []
        for i in range(len(self.names)):
            keep = [j for j in range(len(self.names)) if j != i]
            sub = ViewGraph(self.W[np.ix_(keep, keep)], [self.names[j] for j in keep],
                            self.shared_fraction[np.ix_(keep, keep)],
                            self.mean_theta_deg[np.ix_(keep, keep)],
                            self.mean_w[np.ix_(keep, keep)])
            if len(sub.components(thresh)) > max(base, 1):
                out.append(self.names[i])
        return out

    def redundancy(self, thresh: Optional[float] = None) -> float:
        thresh = self._thr(thresh)
        n = len(self.names)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if self.W[i, j] > thresh]
        if not edges:
            return 0.0
        removable = 0
        for (i, j) in edges:
            W2 = self.W.copy(); W2[i, j] = W2[j, i] = 0.0
            g2 = ViewGraph(W2, self.names, self.shared_fraction,
                           self.mean_theta_deg, self.mean_w)
            if g2.connected and len(g2.components(thresh)) == 1:
                removable += 1
        return removable / len(edges)


def build_view_graph(field: PrecisionField,
                     config: Optional[ModelConfig] = None) -> ViewGraph:
    """
    Build the predicted view graph from a solved precision field.

    Requires station directions, so the field must have been solved with a
    config for which needs_dirs() is True (link_mode='gated', any pairwise
    option, or store_station_dirs=True).
    """
    cfg = config or field.config
    if field.station_dirs is None:
        raise ValueError(
            "field has no station directions; re-solve with "
            "ModelConfig(store_station_dirs=True) or link_mode='gated'")

    u = field.station_dirs
    vis = field.station_vis
    cos_e = field.station_cos_e
    n = len(field.stations)
    ncell = float(vis.shape[1] * vis.shape[2])

    shared = np.zeros((n, n))
    theta = np.full((n, n), np.nan)
    meanw = np.zeros((n, n))
    W = np.zeros((n, n))

    m = cfg.theta_gate_exponent
    th_max = np.radians(cfg.theta_max_deg)
    s_tau = max(np.log(max(cfg.tau_max, 1.0000001)) / 2.0, 1e-6)

    for i in range(n):
        eps_i = field.stations[i].instrument.eps_intra_px
        eps_x = eps_i if cfg.eps_cross_px is None else cfg.eps_cross_px
        base = min((eps_i / max(eps_x, 1e-9)) ** 2, 1.0)
        for j in range(i + 1, n):
            both = vis[i] & vis[j]
            k = int(both.sum())
            if k == 0:
                continue
            c = np.clip(np.einsum("...k,...k->...", u[i], u[j]), -1.0, 1.0)
            th = np.arccos(c)
            w = np.full(th.shape, base)
            if cfg.use_theta_gate:
                if cfg.gate_form == "powerlaw":
                    w = w * gate_powerlaw(th, cfg) * illumination_gate(
                        field.stations[i].lmst_h, field.stations[j].lmst_h, cfg.tau_h)
                else:
                    w = w * np.exp(-2.0 * (th / th_max) ** m)
            if cfg.use_tau_gate:
                hi = np.maximum(cos_e[i], cos_e[j])
                lo = np.maximum(np.minimum(cos_e[i], cos_e[j]), 1e-9)
                w = w * np.exp(-np.log(hi / lo) ** 2 / (2 * s_tau ** 2))
            w = np.clip(w, 0.0, 1.0)

            shared[i, j] = shared[j, i] = k / ncell
            theta[i, j] = theta[j, i] = float(np.degrees(th[both].mean()))
            meanw[i, j] = meanw[j, i] = float(w[both].mean())
            W[i, j] = W[j, i] = shared[i, j] * meanw[i, j]

    names = [s.name or f"S{i+1}" for i, s in enumerate(field.stations)]
    return ViewGraph(W=W, names=names, shared_fraction=shared,
                     mean_theta_deg=theta, mean_w=meanw)


def graph_report(g: ViewGraph) -> str:
    """Human-readable strength report."""
    lines = [
        f"stations                : {len(g.names)}",
        f"connected               : {g.connected}",
        f"components              : {len(g.components())}",
        f"algebraic connectivity  : {g.algebraic_connectivity:.5f}"
        "   (lambda_2; 0 = disconnected)",
        f"edge threshold          : {g._thr():.5f}"
        f"   ({g.rel_threshold:.0%} of strongest edge)",
        f"normalised Fiedler      : {g.normalised_fiedler:.5f}"
        "   (lambda_2 / lambda_max)",
        f"weakest station         : {g.weakest_station} "
        f"(degree {g.degrees.min():.4f})",
        f"edge redundancy         : {g.redundancy():.3f}"
        "   (fraction of edges removable without splitting)",
    ]
    art = g.articulation_stations()
    lines.append(f"articulation stations   : {art if art else 'none'}")
    lines.append("")
    lines.append("edge matrix (shared coverage x matchability):")
    hdr = "         " + " ".join(f"{n[:8]:>9s}" for n in g.names)
    lines.append(hdr)
    for i, nm in enumerate(g.names):
        row = " ".join(f"{g.W[i,j]:9.4f}" for j in range(len(g.names)))
        lines.append(f"{nm[:8]:>8s} {row}")
    lines.append("")
    lines.append("mean convergence angle between stations [deg]:")
    lines.append(hdr)
    for i, nm in enumerate(g.names):
        row = " ".join(("      nan" if not np.isfinite(g.mean_theta_deg[i, j])
                        else f"{g.mean_theta_deg[i,j]:9.2f}")
                       for j in range(len(g.names)))
        lines.append(f"{nm[:8]:>8s} {row}")
    return "\n".join(lines)
