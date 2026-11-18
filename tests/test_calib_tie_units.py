"""0.99.22: a unit with fewer tie rows than columns (one angle of a rig unit tied) in the exact-tie step and covariance
(rpp.calib.tiecov.eliminate_unit) - the rig_T Navcam joint's diagnostic pass of 18 Nov stopped on it ("LinAlgError:
Last 2 dimensions of the array must be square")."""
import numpy as np
import pytest

from rpp.calib import tiecov as TC


def test_eliminate_unit_fully_tied_is_the_old_solution():
    rng = np.random.default_rng(0)
    A, Bg = rng.normal(size=(5, 3)), rng.normal(size=(5, 4))
    M, N, extra = TC.eliminate_unit(A, Bg)
    Q, R = np.linalg.qr(A, mode="complete")
    assert np.allclose(M, np.linalg.solve(R[:3, :3], Q[:, :3].T @ Bg)) and N.shape == (3, 0)
    assert extra.shape == (2, 4) and np.allclose(extra, Q[:, 3:].T @ Bg)
    M, N, extra = TC.eliminate_unit(A[:3], Bg[:3])                       # square: nothing left for the globals
    assert N.shape == (3, 0) and extra.shape == (0, 4) and np.allclose(A[:3] @ M, Bg[:3])


def test_eliminate_unit_under_tied_keeps_the_free_directions():
    rng = np.random.default_rng(1)
    A, Bg = rng.normal(size=(1, 3)), rng.normal(size=(1, 2))             # one angle tied: one row over three columns
    M, N, extra = TC.eliminate_unit(A, Bg)
    assert M.shape == (3, 2) and N.shape == (3, 2) and extra.shape == (0, 2)
    assert np.allclose(A @ N, 0) and np.allclose(N.T @ N, np.eye(2))
    g, w = rng.normal(size=2), rng.normal(size=2)
    assert np.allclose(A @ (-M @ g + N @ w) + Bg @ g, 0)                 # the tie holds for any globals and any w
    A2 = np.vstack([A, 2 * A, A])                                        # three rows of rank one
    B2 = np.vstack([Bg, 2 * Bg, Bg + [1.0, 0.0]])                        # the third asks for something of g alone
    M, N, extra = TC.eliminate_unit(A2, B2)
    assert N.shape == (3, 2) and extra.shape == (2, 2)
    assert np.linalg.matrix_rank(extra) == 1                             # one constraint left on the globals
    g = np.array([0.0, 0.7])                                             # ... which this g meets
    assert np.allclose(extra @ g, 0, atol=1e-12) and np.allclose(A2 @ (-M @ g) + B2 @ g, 0)


def test_exact_tie_step_and_covariance_with_one_component_of_a_quaternion_tied():
    pyceres = pytest.importorskip("pyceres")
    from rpp.calib import tiesolve as TS

    class Pull(pyceres.CostFunction):
        def __init__(self, target, w):
            super().__init__()
            self.t, self.w = np.asarray(target, float), float(w)
            self.set_num_residuals(len(self.t))
            self.set_parameter_block_sizes([len(self.t)])

        def Evaluate(self, parameters, residuals, jacobians):
            residuals[:] = self.w * (np.asarray(parameters[0]) - self.t)
            if jacobians is not None and jacobians[0] is not None:
                jacobians[0][:] = (self.w * np.eye(len(self.t))).ravel()
            return True

    prob = pyceres.Problem()
    q = np.array([0.02, -0.01, 0.03, 0.0]); q[3] = np.sqrt(1 - q[:3] @ q[:3])
    g, pt = np.array([0.05]), np.array([1.0, 2.0, 3.0])
    keep = []
    for arr, t, w in ((q, [0.04, 0.01, -0.02, 0.999], 10.0), (g, [0.01], 3.0), (pt, [1.1, 2.1, 2.9], 1.0)):
        keep.append(Pull(t, w))
        prob.add_residual_block(keep[-1], None, [arr])
    prob.set_manifold(q, pyceres.EigenQuaternionManifold())
    A, sigma = np.array([[1.0, 0, 0, 0, -1.0]]), np.array([1e-6])        # the quaternion's x tied to the global
    keep.append(TS.linear_tie_class()(A, sigma, [4, 1]))
    tie = {"id": prob.add_residual_block(keep[-1], None, [q, g]), "A": A, "sigma": sigma, "sizes": [4, 1],
           "blocks": [q, g], "quaternion": False}
    out = TS.exact_tie_refine(prob, [tie], [pt], [q], {"g": g}, verbose=False)
    assert out["error"] is None and out["iterations"] >= 1 and out["cost_end"] < out["cost_start"]
    assert abs(q[0] - g[0]) < 1e-12                                      # the tie is exact
    assert abs(q[1] - 0.01) < 0.05 and abs(q[2] + 0.02) < 0.08           # the untied directions followed their data
    cov = TC.exact_tie_covariance(prob, [tie["id"]], [pt], [q], {"g": g})
    assert cov["error"] is None and cov["dims"]["untied_directions"] == 2
    c = float(cov["blocks"]["extra:g"][0, 0])
    assert np.isfinite(c) and 0 < c < 1.0 / 9.0 + 1e-9                   # tighter than the global's own prior alone
