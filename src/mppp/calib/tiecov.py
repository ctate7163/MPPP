"""
The covariance of the global terms of a tied joint, with the ties eliminated exactly (0.94.6, 25 Oct 2026).

The joints (:mod:`mppp.calib.tied`, :mod:`mppp.calib.ztied`) tie each unit - a bin camera, a rig unit - to the
global terms with linear residuals of sigma 1e-3 px .. 1e-8 rad (``LinearTie``): effectively exact, weights up to
1e16 next to data weights of 1e7.  The solve is fine with that, but the covariance is not: COLMAP's estimator
eliminates the points and inverts the reduced normal matrix as it is, and on the 22-29 block Navcam joints (20,
24 and 25 Oct) the global blocks came out infinite (the three-block subsets are fine) - the information of a global
term is the small difference of two numbers of 1e16 x the number of units, below the precision of a double.

Here the ties are constraints, not residuals: after the solve the tie residual blocks are taken out of the problem,
the Jacobian of the rest (observations and priors) is evaluated, the points are eliminated (3 x 3 blocks), and the
tied unit parameters are substituted by the global terms they equal (``t_u = -C_uu^+ C_ug g``, from the ties' own
Jacobian; tie rows that hold a unit parameter which is itself held constrain the globals alone and reduce them to
their null space).  The globals' covariance is then the inverse of their marginal information - every number in it
of the size of the data.  For exact ties this is the same covariance the full problem has, in exact arithmetic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("mppp.calib")


def _ptr(a: np.ndarray) -> int:
    return int(a.__array_interface__["data"][0])


def _csr(J):
    """A pyceres CRSMatrix as scipy CSR (``to_tuple`` gives the COO triplets: row, column, value)."""
    import scipy.sparse as sp
    r, c, v = (np.asarray(x) for x in J.to_tuple())
    return sp.coo_matrix((v.astype(float), (r.astype(np.int64), c.astype(np.int64))),
                         shape=(int(J.num_rows), int(J.num_cols))).tocsr()


def _point_block_inverse(Hpp, n_points: int, rel_damp: float = 1e-12, lm: float = 0.0):
    """The inverse of the block-diagonal 3 x 3 point part of the normal matrix, as a sparse block-diagonal matrix
    (``lm``: Levenberg-Marquardt damping, lm x its diagonal added)."""
    import scipy.sparse as sp
    B = np.zeros((n_points, 3, 3))
    for a in range(3):
        for b in range(3):
            k = b - a
            d = Hpp.diagonal(k)
            idx = 3 * np.arange(n_points) + min(a, b)
            B[:, a, b] = d[idx]
    if lm:
        B = B + lm * (np.diagonal(B, axis1=1, axis2=2)[:, :, None] * np.eye(3))
    tr = np.trace(B, axis1=1, axis2=2)
    B += (rel_damp * np.maximum(tr, 1e-300))[:, None, None] * np.eye(3)
    Binv = np.linalg.inv(B)
    return sp.bsr_matrix((Binv, np.arange(n_points), np.arange(n_points + 1)), shape=(3 * n_points, 3 * n_points)).tocsr()


def exact_tie_covariance(prob, tie_ids: Sequence[Any], point_blocks: Sequence[np.ndarray],
                         free_blocks: Sequence[np.ndarray], global_blocks: Dict[str, np.ndarray],
                         apply_loss: bool = True, rel_damp: float = 1e-12) -> Dict[str, Any]:
    """Covariances (tangent space, unit variance factor) of the ``global_blocks`` {name: array} of a solved problem
    whose ``tie_ids`` residual blocks tie unit blocks (among ``free_blocks``) to them.  The tie residual blocks are
    REMOVED from ``prob`` (call it after the solve, last).  Returns {"blocks": {"extra:<name>": cov}, "variance_factor",
    "method", "dims", "error"}."""
    import pyceres
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    out: Dict[str, Any] = {"blocks": {}, "method": "exact ties (mppp.calib.tiecov)", "error": None}
    try:
        seen, plist, kinds = set(), [], []
        for kind, arrs in (("point", point_blocks), ("free", free_blocks), ("global", list(global_blocks.values()))):
            for a in arrs:
                if a is None or not prob.has_parameter_block(a) or prob.is_parameter_block_constant(a):
                    continue
                p = _ptr(a)
                if p in seen:
                    continue
                seen.add(p)
                plist.append(a)
                kinds.append(kind)
        sizes = [int(prob.parameter_block_tangent_size(a)) for a in plist]
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        n = int(offs[-1])
        npts = sum(1 for k in kinds if k == "point")
        P = int(offs[npts])                                    # the point columns come first
        gname_of = {_ptr(a): nm for nm, a in global_blocks.items()}
        gcols: Dict[str, np.ndarray] = {}
        for a, k, o, s in zip(plist, kinds, offs[:-1], sizes):
            if k == "global":
                gcols[gname_of[_ptr(a)]] = np.arange(o, o + s)
        if not gcols:
            out["error"] = "no free global block"
            return out
        # the ties' Jacobian (tangent space): which unit columns each tie fixes, and by which globals
        eo = pyceres.EvaluateOptions()
        eo.set_parameter_blocks(plist)
        eo.set_residual_blocks(list(tie_ids))
        eo.apply_loss_function = False
        C = _csr(prob.evaluate_jacobian(eo))
        for rid in tie_ids:                                   # the ties leave the problem: constraints from here on
            prob.remove_residual_block(rid)
        eo = pyceres.EvaluateOptions()
        eo.set_parameter_blocks(plist)
        eo.apply_loss_function = bool(apply_loss)
        J = _csr(prob.evaluate_jacobian(eo))
        res = np.asarray(prob.evaluate_residuals(eo), float)
        m = J.shape[0]
        # --- the points eliminated (each observation row touches one point)
        Jc = J.tocsc()
        Jp, Jr = Jc[:, :P].tocsr(), Jc[:, P:].tocsr()
        Hpp = (Jp.T @ Jp).tocsr()
        Hpp_inv = _point_block_inverse(Hpp, npts, rel_damp)
        Hrp = (Jr.T @ Jp).tocsr()
        S = (Jr.T @ Jr - Hrp @ Hpp_inv @ Hrp.T).tocsr()       # (n - P) x (n - P)
        r = n - P
        # --- the constraints in the remaining columns
        Cr = C[:, P:].tocsr()
        gidx = np.concatenate([gcols[k] for k in gcols]) - P
        gmask = np.zeros(r, bool)
        gmask[gidx] = True
        Cg, Cu = Cr[:, np.where(gmask)[0]].toarray(), Cr[:, np.where(~gmask)[0]].tocsr()
        ucols_all = np.where(~gmask)[0]
        # group the tie rows by the unit columns they touch (one unit block per tie residual)
        rows_of: Dict[tuple, List[int]] = {}
        for i in range(Cu.shape[0]):
            cs = tuple(sorted(Cu.indices[Cu.indptr[i]:Cu.indptr[i + 1]][np.abs(Cu.data[Cu.indptr[i]:Cu.indptr[i + 1]]) > 0]))
            rows_of.setdefault(cs, []).append(i)
        # merge rows sharing columns into units (connected components over columns)
        parent: Dict[int, int] = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for cs in rows_of:
            for c in cs[1:]:
                parent[find(c)] = find(cs[0])
            if cs:
                find(cs[0])
        units: Dict[int, Dict[str, List[int]]] = {}
        gcons: List[np.ndarray] = []
        for cs, rws in rows_of.items():
            if not cs:                                        # a tie on a held unit parameter: globals only
                gcons += [Cg[i] for i in rws]
                continue
            u = units.setdefault(find(cs[0]), {"rows": [], "cols": set()})
            u["rows"] += rws
            u["cols"].update(cs)
        tied_map: Dict[int, np.ndarray] = {}                  # unit column (in Cu numbering) -> row of -M (|g|)
        for u in units.values():
            cols = sorted(u["cols"])
            A = Cu[u["rows"]][:, cols].toarray()
            Bg = Cg[u["rows"]]
            Q, R = np.linalg.qr(A, mode="complete")
            k = len(cols)
            Rk = R[:k, :k]
            if np.linalg.matrix_rank(Rk) < k:
                raise ValueError("a unit's tie Jacobian is rank-deficient")
            M = np.linalg.solve(Rk, Q[:, :k].T @ Bg)          # t_u = -M g
            for j, c in enumerate(cols):
                tied_map[c] = -M[j]
            if A.shape[0] > k:                                # extra rows: constraints on the globals
                gcons += list(Q[:, k:].T @ Bg)
        ng = len(gidx)
        if gcons:
            G = np.array(gcons)
            _, sv, Vt = np.linalg.svd(G, full_matrices=True)
            rank = int(np.sum(sv > 1e-10 * max(sv[0], 1e-300)))
            Ng = Vt[rank:].T                                   # globals = Ng y
        else:
            rank, Ng = 0, np.eye(ng)
        ky = Ng.shape[1]
        # --- the substitution x = T z, z = [untied columns, y]
        untied = np.array([c for i, c in enumerate(ucols_all) if i not in tied_map], int)
        nf = len(untied)
        Trows, Tcols, Tvals = [], [], []
        for j, c in enumerate(untied):
            Trows.append(int(c)); Tcols.append(j); Tvals.append(1.0)
        Tg = Ng
        for i, c in enumerate(gidx):
            for jy in range(ky):
                if Tg[i, jy] != 0.0:
                    Trows.append(int(c)); Tcols.append(nf + jy); Tvals.append(float(Tg[i, jy]))
        for cu, mrow in tied_map.items():
            coef = mrow @ Tg                                   # |y|
            c = int(ucols_all[cu])
            for jy in range(ky):
                if coef[jy] != 0.0:
                    Trows.append(c); Tcols.append(nf + jy); Tvals.append(float(coef[jy]))
        T = sp.csr_matrix((Tvals, (Trows, Tcols)), shape=(r, nf + ky))
        Sz = (T.T @ S @ T).tocsc()
        Sff, Sfy, Syy = Sz[:nf, :nf].tocsc(), Sz[:nf, nf:].toarray(), Sz[nf:, nf:].toarray()
        diag = np.abs(Sff.diagonal())
        damp = rel_damp * np.where(diag > 0, diag, 1.0)
        lu = spla.splu((Sff + sp.diags(damp)).tocsc())
        X = lu.solve(Sfy)
        Mh = Syy - Sfy.T @ X
        Mh = 0.5 * (Mh + Mh.T)
        dm = np.sqrt(np.clip(np.diag(Mh), 1e-300, None))      # the globals' units differ by 1e15: scaled first
        Ms = Mh / np.outer(dm, dm)
        w, V = np.linalg.eigh(Ms)
        if w[0] <= 1e-12 * max(w[-1], 1e-300):
            raise ValueError(f"the globals' marginal information is singular (scaled eigenvalues {w[0]:.3g} .. "
                             f"{w[-1]:.3g})")
        cov_y = ((V / w) @ V.T) / np.outer(dm, dm)
        cov_g = Ng @ cov_y @ Ng.T
        dg = np.diag(cov_g)
        if not np.all(np.isfinite(cov_g)) or np.any(dg < -1e-12 * np.max(np.abs(dg))):
            raise ValueError("the globals' covariance is not finite / not positive")
        start = 0
        columns = []
        for name, cols in gcols.items():
            k = len(cols)
            out["blocks"][f"extra:{name}"] = cov_g[start:start + k, start:start + k]
            columns += [(name, j) for j in range(k)]
            start += k
        out["full"] = {"columns": columns, "cov": cov_g}         # every global term with every other (tangent)
        n_eff = P + nf + ky
        out["variance_factor"] = float(np.sum(res ** 2) / max(m - n_eff, 1))
        out["dims"] = {"rows": int(m), "points": npts, "free_columns": int(nf), "global_columns": int(ng),
                       "global_constraints": int(rank), "tied_columns": len(tied_map),
                       "condition": float(w[-1] / w[0])}
    except Exception as e:                                     # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out
