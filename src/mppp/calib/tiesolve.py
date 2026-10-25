"""
The tied joints solved with the ties as exact constraints (0.94.6, Christian 25 Oct 2026: "write each unit's camera as
a function of the globals, so there are no tie residuals at all").

The joints tie each unit (bin camera, rig unit) to the global terms with stiff linear residuals (``LinearTie``, sigma
1e-3 px .. 1e-8 rad).  Ceres solves that, but the problem is ill-conditioned: on the synthetic joint the converged
pitch drift rate moved with the tie sigma (0.24 at 1e-6, 0.22 at 1e-8, 0.19 at 1e-10, 0.01 at 1e-12; sd 0.29) at the
same cost - Levenberg-Marquardt stops in a flat valley.  :func:`exact_tie_refine` continues from Ceres' solution with
the ties eliminated: the tie residuals leave the problem, every tied unit parameter is SET from the globals (the
law), and Levenberg-Marquardt steps are taken in the reduced variables - points (eliminated by the Schur complement),
poses and the untied unit parameters, and the globals' null space - with the Jacobian of the remaining residuals
(observations, priors) evaluated by Ceres.  Each trial is accepted on Ceres' own cost (robust loss included).  At the
end the ties are added back (zero residuals: the units are on the law), so the covariance and the rest of the
pipeline see the problem as before.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .tiecov import _csr, _point_block_inverse, _ptr

log = logging.getLogger("mppp.calib")


def _probe_plus_jacobians(prob, blocks: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Each block's PlusJacobian (ambient x tangent) at its current value: an identity residual over the blocks,
    evaluated once and removed."""
    import pyceres

    class Identity(pyceres.CostFunction):
        def __init__(self, sizes):
            super().__init__()
            self.sizes = list(map(int, sizes))
            self.set_num_residuals(int(sum(self.sizes)))
            self.set_parameter_block_sizes(self.sizes)

        def Evaluate(self, parameters, residuals, jacobians):
            o = 0
            for j, p in enumerate(parameters):
                n = self.sizes[j]
                residuals[o:o + n] = np.asarray(p, float)
                if jacobians is not None and jacobians[j] is not None:
                    jac = np.zeros((int(sum(self.sizes)), n))
                    jac[o:o + n] = np.eye(n)
                    jacobians[j][:] = jac.ravel()
                o += n
            return True

    sizes = [int(prob.parameter_block_size(b)) for b in blocks]
    cf = Identity(sizes)
    rid = prob.add_residual_block(cf, None, list(blocks))
    eo = pyceres.EvaluateOptions()
    eo.set_parameter_blocks(list(blocks))
    eo.set_residual_blocks([rid])
    eo.apply_loss_function = False
    Jp = _csr(prob.evaluate_jacobian(eo)).toarray()
    prob.remove_residual_block(rid)
    out, ro, co = [], 0, 0
    for b, n in zip(blocks, sizes):
        k = int(prob.parameter_block_tangent_size(b))
        out.append(Jp[ro:ro + n, co:co + k].copy())
        ro += n
        co += k
    return out


def _cost(prob) -> float:
    import pyceres
    o = pyceres.SolverOptions()
    o.max_num_iterations = 0
    o.minimizer_progress_to_stdout = False
    o.logging_type = pyceres.LoggingType.SILENT
    s = pyceres.SolverSummary()
    pyceres.solve(o, prob, s)
    return float(s.initial_cost)


def _set_units_from_law(ties: Sequence[Dict[str, Any]]) -> None:
    """Every tied unit parameter set to what its tie says (A x = 0 with a coefficient 1 on the unit parameter); a
    rig unit's quaternion w renormalised."""
    for t in ties:
        A, blocks = t["A"], t["blocks"]
        n_u = len(blocks[0])
        rest = np.concatenate([np.asarray(b, float).ravel() for b in blocks[1:]]) if len(blocks) > 1 else np.zeros(0)
        u = blocks[0]
        for i in range(A.shape[0]):
            a = A[i, :n_u]
            nz = np.nonzero(a)[0]
            if len(nz) != 1 or abs(a[nz[0]] - 1.0) > 1e-12:
                continue
            u[nz[0]] = -float(A[i, n_u:] @ rest)
        if n_u == 7 and t.get("quaternion"):
            xyz = np.asarray(u[:3], float)
            w = np.sqrt(max(0.0, 1.0 - float(xyz @ xyz)))
            u[3] = w if u[3] >= 0 else -w


def exact_tie_refine(prob, ties: Sequence[Dict[str, Any]], point_blocks: Sequence[np.ndarray],
                     free_blocks: Sequence[np.ndarray], global_blocks: Dict[str, np.ndarray],
                     max_iterations: int = 30, function_tolerance: float = 1e-10, lam0: float = 1e-4,
                     verbose: bool = True) -> Dict[str, Any]:
    """Levenberg-Marquardt with the ties ``ties`` [{"id", "A" (unscaled, ambient), "sigma", "sizes", "blocks"
    (unit first), "quaternion"}] as exact constraints, from the problem's current values.  The tie residuals are
    taken out and added back at the end (new ids in each tie's "id").  Returns {"iterations", "cost_start",
    "cost_end", "global_change" {name: tangent step}, "lambda", "seconds", "error"}."""
    import pyceres
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from .tied import _linear_tie_class
    t0 = time.time()
    out: Dict[str, Any] = {"error": None, "iterations": 0, "history": []}
    LinearTie = _linear_tie_class()
    removed = False
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
        npts = sum(1 for k in kinds if k == "point")
        nonpt = plist[npts:]
        Pj = _probe_plus_jacobians(prob, nonpt)            # tangent -> ambient of every non-point block
        sizes = [3] * npts + [P.shape[1] for P in Pj]
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        NP = int(offs[npts])
        gname_of = {_ptr(a): nm for nm, a in global_blocks.items()}
        gcols: Dict[str, np.ndarray] = {}
        for a, k, o, s in zip(plist, kinds, offs[:-1], sizes):
            if k == "global":
                gcols[gname_of[_ptr(a)]] = np.arange(o, o + s)
        # the constraints in tangent space, from the ties' Jacobian at the start
        eo = pyceres.EvaluateOptions()
        eo.set_parameter_blocks(plist)
        eo.set_residual_blocks([t["id"] for t in ties])
        eo.apply_loss_function = False
        C = _csr(prob.evaluate_jacobian(eo))
        for t in ties:
            prob.remove_residual_block(t["id"])
        removed = True
        _set_units_from_law(ties)
        r_dim = int(offs[-1]) - NP
        Cr = C[:, NP:].tocsr()
        gidx = np.concatenate([gcols[k] for k in gcols]) - NP
        gmask = np.zeros(r_dim, bool)
        gmask[gidx] = True
        ucols_all = np.where(~gmask)[0]
        Cg = Cr[:, np.where(gmask)[0]].toarray()
        Cu = Cr[:, ucols_all].tocsr()
        # unit columns grouped by tie rows (as in tiecov)
        from .tiecov import exact_tie_covariance as _unused  # noqa: F401  (same structure)
        rows_by_cols: Dict[tuple, List[int]] = {}
        for i in range(Cu.shape[0]):
            sl = slice(Cu.indptr[i], Cu.indptr[i + 1])
            cs = tuple(sorted(Cu.indices[sl][np.abs(Cu.data[sl]) > 0]))
            rows_by_cols.setdefault(cs, []).append(i)
        parent: Dict[int, int] = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for cs in rows_by_cols:
            for c in cs[1:]:
                parent[find(c)] = find(cs[0])
            if cs:
                find(cs[0])
        units: Dict[int, Dict[str, Any]] = {}
        gcons: List[np.ndarray] = []
        for cs, rws in rows_by_cols.items():
            if not cs:
                gcons += [Cg[i] for i in rws]
                continue
            u = units.setdefault(find(cs[0]), {"rows": [], "cols": set()})
            u["rows"] += rws
            u["cols"].update(cs)
        tied_map: Dict[int, np.ndarray] = {}
        for u in units.values():
            cols = sorted(u["cols"])
            A = Cu[u["rows"]][:, cols].toarray()
            Q, R = np.linalg.qr(A, mode="complete")
            k = len(cols)
            M = np.linalg.solve(R[:k, :k], Q[:, :k].T @ Cg[u["rows"]])
            for j, c in enumerate(cols):
                tied_map[c] = -M[j]
            if A.shape[0] > k:
                gcons += list(Q[:, k:].T @ Cg[u["rows"]])
        ng = len(gidx)
        if gcons:
            _, sv, Vt = np.linalg.svd(np.array(gcons), full_matrices=True)
            rank = int(np.sum(sv > 1e-10 * max(sv[0], 1e-300)))
            Ng = Vt[rank:].T
        else:
            Ng = np.eye(ng)
        ky = Ng.shape[1]
        untied = np.array([c for i, c in enumerate(ucols_all) if i not in tied_map], int)
        nf = len(untied)
        Tr, Tc, Tv = list(map(int, untied)), list(range(nf)), [1.0] * nf
        for i, c in enumerate(gidx):
            for jy in np.nonzero(Ng[i])[0]:
                Tr.append(int(c)); Tc.append(nf + int(jy)); Tv.append(float(Ng[i, jy]))
        for cu, mrow in tied_map.items():
            coef = mrow @ Ng
            for jy in np.nonzero(coef)[0]:
                Tr.append(int(ucols_all[cu])); Tc.append(nf + int(jy)); Tv.append(float(coef[jy]))
        T = sp.csr_matrix((Tv, (Tr, Tc)), shape=(r_dim, nf + ky))
        Tg = T[np.sort(gidx)] if False else None                # (globals updated through Ng below)

        def snapshot():
            return [np.array(a, copy=True) for a in plist]

        def restore(snap):
            for a, s in zip(plist, snap):
                a[:] = s

        lam = float(lam0)
        cost = _cost(prob)
        out["cost_start"] = cost
        g_start = {k: np.array(v, copy=True) for k, v in global_blocks.items()}
        for it in range(int(max_iterations)):
            eo = pyceres.EvaluateOptions()
            eo.set_parameter_blocks(plist)
            eo.apply_loss_function = True
            J = _csr(prob.evaluate_jacobian(eo))
            res = np.asarray(prob.evaluate_residuals(eo), float)
            Jc = J.tocsc()
            Jp, Jr = Jc[:, :NP].tocsr(), Jc[:, NP:].tocsr()
            Hpp = (Jp.T @ Jp).tocsr()
            Hrp = (Jr.T @ Jp).tocsr()
            bp, br = Jp.T @ res, Jr.T @ res
            Arr = (Jr.T @ Jr).tocsr()
            ATz = (T.T @ Arr @ T).diagonal()
            dz = np.abs(ATz)
            dz = np.where(dz > 0, dz, 1.0)                     # a column nothing measures: plain damping
            accepted = False
            for _try in range(20):               # 1e-4 x 5^19: down to a gradient step
                # 25 Oct: the points damped too (as Ceres does) - undamped, a poorly seen far point stepped by a
                # metre and an observation behind its camera came back (+34 in cost at any step size)
                Hpp_inv = _point_block_inverse(Hpp, npts, 1e-12, lm=lam)
                S = (Arr - Hrp @ Hpp_inv @ Hrp.T).tocsr()
                b = br - Hrp @ (Hpp_inv @ bp)
                Sz = (T.T @ S @ T).tocsc()
                bz = T.T @ b
                try:
                    lu = spla.splu((Sz + sp.diags(lam * dz)).tocsc())
                    step_z = -lu.solve(bz)
                except RuntimeError:
                    lam *= 10.0
                    continue
                if not np.all(np.isfinite(step_z)):
                    lam *= 10.0
                    continue
                step_r = T @ step_z
                step_p = -(Hpp_inv @ (bp + Hrp.T @ step_r))
                snap = snapshot()
                # apply: points additive, the rest through their PlusJacobian (quaternions renormalised)
                full = np.concatenate([step_p, step_r])
                for j, a in enumerate(plist):
                    d = full[offs[j]:offs[j + 1]]
                    if j < npts:
                        a += d
                        continue
                    P = Pj[j - npts]
                    a += P @ d
                    if len(a) == 7 and P.shape[1] in (3, 6) and abs(P[3]).sum() > 0:
                        q = np.asarray(a[:4], float)
                        a[:4] = q / max(np.linalg.norm(q), 1e-300)
                _set_units_from_law(ties)
                c1 = _cost(prob)
                if log.isEnabledFor(logging.DEBUG):
                    pred = -float(step_z @ bz) - 0.5 * float(step_z @ (Sz @ step_z))
                    log.debug(f"  exact-tie trial {it + 1}.{_try + 1}: lambda {lam:.3g}, cost {cost:.8g} -> {c1:.8g} "
                              f"(predicted decrease {pred:.4g}), |step_z| {np.linalg.norm(step_z):.3g}")
                    if c1 > cost + 1.0 and not out.get("_dumped"):
                        out["_dumped"] = True
                        r1 = np.asarray(prob.evaluate_residuals(eo), float)
                        dd = np.abs(r1 - res)
                        top = np.argsort(dd)[::-1][:8]
                        log.debug("    residual jumps (index, before, after): " + "; ".join(
                            f"{int(i)}/{len(res)} {res[i]:.4g} -> {r1[i]:.4g}" for i in top))
                        log.debug(f"    0.5 sum r^2 before {0.5 * float(res @ res):.8g}, after {0.5 * float(r1 @ r1):.8g}")
                        big = []
                        for j, a in enumerate(plist):
                            d_ = full[offs[j]:offs[j + 1]]
                            if len(d_) and np.max(np.abs(d_)) > 0:
                                big.append((float(np.max(np.abs(d_))), j, kinds[j], len(a)))
                        big.sort(reverse=True)
                        log.debug("    largest block steps: " + "; ".join(f"{k} #{j} (size {n}) {v:.3g}" for v, j, k, n in big[:6]))
                # 25 Oct: a failed evaluation reports -1 (the 12-block run accepted it and left the problem invalid)
                if np.isfinite(c1) and 0.0 <= c1 < cost and all(np.all(np.isfinite(a)) for a in plist):
                    rel = (cost - c1) / max(cost, 1e-300)
                    out["history"].append({"iteration": it + 1, "cost": c1, "lambda": lam, "rel_decrease": rel})
                    cost = c1
                    lam = max(lam / 3.0, 1e-12)
                    accepted = True
                    break
                restore(snap)
                lam *= 5.0
            out["iterations"] = it + 1
            if not accepted:
                out["stop"] = "no decrease"
                break
            if out["history"][-1]["rel_decrease"] < function_tolerance:
                out["stop"] = "function tolerance"
                break
        out["cost_end"] = cost
        out["lambda"] = lam
        out["global_change"] = {k: (np.asarray(global_blocks[k], float) - g_start[k]).tolist() for k in global_blocks}
    except Exception as e:                                      # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        if removed:                                             # the ties back (the units are on the law)
            for t in ties:
                t["id"] = prob.add_residual_block(LinearTie(t["A"], t["sigma"], t["sizes"]), None, list(t["blocks"]))
    out.pop("_dumped", None)
    out["seconds"] = time.time() - t0
    if verbose:
        log.info("exact-tie refinement: %s" % (
            f"{out['iterations']} iterations, cost {out.get('cost_start', float('nan')):.6g} -> "
            f"{out.get('cost_end', float('nan')):.6g} ({out.get('stop', 'max iterations')}), {out['seconds']:.0f} s"
            if not out["error"] else f"FAILED ({out['error']})"))
    return out
