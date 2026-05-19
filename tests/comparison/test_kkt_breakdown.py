"""Unit tests for ``evaluate_problem``'s KKT-residual breakdown.

The tests construct a tiny analytical problem with known optimum and
verify each component of ``KKTBreakdown`` matches what we'd compute by
hand.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from tests.comparison.problem_spec import (
    KKTBreakdown,
    ProblemSpec,
    attach_kkt,
    evaluate_problem,
    histories_from_iterates,
    SolverResult,
)


def _make_tiny_problem() -> ProblemSpec:
    """Tiny 1-D problem: minimize 0.5 sum_t x_t^2 + 0.5 sum_t u_t^2.

    T = 2, n = m = 1, theta_dim = 0.
    Dynamics: x_{t+1} = x_t + u_t.
    Initial state: x0 = 1.0.
    No user equality / inequality.

    Optimum is x = [1, ...], u driving x toward 0; closed-form via
    Riccati. We mostly use this to verify the *evaluator's* arithmetic
    rather than the optimum itself.
    """
    T = 2
    n = m = 1

    def cost(x, u, theta, t):  # noqa: ARG001
        return 0.5 * (x[0] ** 2) + 0.5 * (u[0] ** 2)

    def dyn(x, u, theta, t):  # noqa: ARG001
        return x + u

    return ProblemSpec(
        name="tiny",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        x0=jnp.array([1.0]),
        cost=cost,
        dynamics=dyn,
        equalities=None,
        inequalities=None,
        eq_dim=0,
        ineq_dim=0,
        X_init=jnp.zeros((T + 1, n)),
        U_init=jnp.zeros((T, m)),
        Theta_init=jnp.zeros((0,)),
    )


def _make_tiny_inequality_problem() -> ProblemSpec:
    """Same as ``_make_tiny_problem`` but with a dummy ``ineq = u <= 0`` row."""
    T = 2
    n = m = 1

    def cost(x, u, theta, t):  # noqa: ARG001
        return 0.5 * (x[0] ** 2) + 0.5 * (u[0] ** 2)

    def dyn(x, u, theta, t):  # noqa: ARG001
        return x + u

    def ineq(x, u, theta, t):  # noqa: ARG001
        return jnp.array([u[0]])  # u <= 0

    return ProblemSpec(
        name="tiny_ineq",
        T=T,
        n=n,
        m=m,
        theta_dim=0,
        x0=jnp.array([1.0]),
        cost=cost,
        dynamics=dyn,
        equalities=None,
        inequalities=ineq,
        eq_dim=0,
        ineq_dim=1,
        X_init=jnp.zeros((T + 1, n)),
        U_init=jnp.zeros((T, m)),
        Theta_init=jnp.zeros((0,)),
    )


class EvaluateProblemKKTTests(unittest.TestCase):
    def test_legacy_3tuple_no_multipliers(self):
        problem = _make_tiny_problem()
        X = np.array([[1.0], [0.5], [0.25]])
        U = np.array([[-0.5], [-0.25]])
        Theta = np.zeros((0,))
        out = evaluate_problem(problem, X, U, Theta)
        self.assertEqual(len(out), 3)
        cost, eq_v, ineq_v = out
        # cost: 0.5*(1+.25+.0625) + 0.5*(.25+.0625) = 0.65625 + 0.15625 = 0.8125
        self.assertAlmostEqual(cost, 0.8125, places=10)
        self.assertAlmostEqual(eq_v, 0.0, places=12)
        self.assertEqual(ineq_v, 0.0)

    def test_kkt_breakdown_no_constraints(self):
        problem = _make_tiny_problem()
        X = np.array([[1.0], [0.5], [0.25]])
        U = np.array([[-0.5], [-0.25]])
        Theta = np.zeros((0,))
        # Equality residual stack = [init_defect (1,); dyn_defects (T*n=2,)] = (3,)
        lam = np.zeros(3)
        ineq_mults = np.zeros(0)
        out = evaluate_problem(
            problem,
            X,
            U,
            Theta,
            multipliers_eq=lam,
            multipliers_ineq=ineq_mults,
        )
        self.assertEqual(len(out), 4)
        cost, eq_v, ineq_v, kkt = out
        self.assertAlmostEqual(kkt.init, 0.0, places=12)
        self.assertAlmostEqual(kkt.dyn, 0.0, places=12)
        self.assertEqual(kkt.eq, 0.0)
        self.assertEqual(kkt.ineq, 0.0)
        self.assertEqual(kkt.dual, 0.0)
        self.assertEqual(kkt.complementarity, 0.0)
        # Stationarity at this iterate: ∇_x f at z = (X, U) — multipliers
        # are zero so L = f. ∇L wrt the flat z is just (X, U) packed back
        # together (since f = .5 sum_x x^2 + .5 sum_u u^2 → ∇ = z elementwise).
        # max |∇| = max|X[t]| + max|U[t]| = 1.0 (X[0]=1.0).
        self.assertAlmostEqual(kkt.stationarity, 1.0, places=10)
        self.assertAlmostEqual(kkt.joint, 1.0, places=10)

    def test_kkt_init_violation(self):
        """If X[0] doesn't match x0 the init-violation should fire."""
        problem = _make_tiny_problem()
        X = np.array([[2.0], [0.5], [0.25]])  # X[0]=2.0 vs x0=1.0
        U = np.array([[-1.5], [-0.25]])  # so dyn defect 0 again
        Theta = np.zeros((0,))
        lam = np.zeros(3)
        out = evaluate_problem(
            problem,
            X,
            U,
            Theta,
            multipliers_eq=lam,
            multipliers_ineq=np.zeros(0),
        )
        cost, eq_v, ineq_v, kkt = out
        self.assertAlmostEqual(kkt.init, 1.0, places=12)
        self.assertAlmostEqual(kkt.dyn, 0.0, places=12)
        self.assertEqual(kkt.eq, 0.0)

    def test_kkt_dual_and_complementarity_with_inequality(self):
        """Verify dual + complementarity components on a problem with an active ineq."""
        problem = _make_tiny_inequality_problem()
        # Dynamics-feasible iterate.
        X = np.array([[1.0], [0.5], [0.25]])
        U = np.array([[-0.5], [-0.25]])  # u <= 0 is satisfied tightly nowhere
        Theta = np.zeros((0,))
        # Equality stack = [init (1); dyn (2)] = (3,)
        lam = np.array([0.0, 0.0, 0.0])
        # Inequality stack = (T+1)*ineq_dim = 3*1 = 3.
        # Test 1: positive multipliers (ok dual), product = z * g.
        z_mults = np.array([0.1, 0.0, 0.0])
        out = evaluate_problem(
            problem,
            X,
            U,
            Theta,
            multipliers_eq=lam,
            multipliers_ineq=z_mults,
        )
        _, _, _, kkt = out
        self.assertEqual(kkt.dual, 0.0)
        # comp = max |z * g| where g = [u_0, u_1, 0] (terminal u padded as 0)
        #     = max |0.1 * (-0.5)| = 0.05
        self.assertAlmostEqual(kkt.complementarity, 0.05, places=10)

        # Test 2: negative multiplier should fire dual violation.
        z_mults_bad = np.array([-0.2, 0.0, 0.0])
        out2 = evaluate_problem(
            problem,
            X,
            U,
            Theta,
            multipliers_eq=lam,
            multipliers_ineq=z_mults_bad,
        )
        _, _, _, kkt2 = out2
        self.assertAlmostEqual(kkt2.dual, 0.2, places=10)


class HistoriesTests(unittest.TestCase):
    def test_histories_from_iterates_basic(self):
        problem = _make_tiny_problem()
        Theta = np.zeros((0,))
        # Iter 0: X = [1, 0, 0], U = [0, 0] — fails dynamics from stage 0
        # (next_x=1+0=1 vs X[1]=0 → defect -1 etc). Iter 1: dyn-feasible.
        iters = [
            (np.array([[1.0], [0.0], [0.0]]), np.zeros((2, 1)), Theta),
            (np.array([[1.0], [0.5], [0.25]]), np.array([[-0.5], [-0.25]]), Theta),
        ]
        c_h, eq_h, ineq_h = histories_from_iterates(problem, iters)
        self.assertEqual(len(c_h), 2)
        # Iter 0: cost = 0.5 * (1+0+0) + 0.5 * (0+0) = 0.5
        # Defects: X[0]-x0=0; dyn[0]= (1+0)-0 = 1; dyn[1]=(0+0)-0=0. eq_v = 1.
        self.assertAlmostEqual(c_h[0], 0.5, places=10)
        self.assertAlmostEqual(eq_h[0], 1.0, places=10)
        self.assertEqual(ineq_h[0], 0.0)


class AttachKKTTests(unittest.TestCase):
    def test_attach_kkt_populates_optional_fields(self):
        result = SolverResult(
            solver_name="dummy",
            problem_name="dummy",
            iterations=0,
            solve_time_ms=0.0,
            final_cost=0.0,
            eq_violation_inf=0.0,
            ineq_violation_inf=0.0,
            success=True,
        )
        kkt = KKTBreakdown(
            init=0.1,
            dyn=0.2,
            eq=0.3,
            ineq=0.4,
            dual=0.5,
            complementarity=0.6,
            stationarity=0.7,
            joint=0.7,
        )
        attach_kkt(result, kkt)
        self.assertEqual(result.kkt_init_violation_inf, 0.1)
        self.assertEqual(result.kkt_dyn_violation_inf, 0.2)
        self.assertEqual(result.kkt_residual_inf, 0.7)
        self.assertEqual(result.kkt_dual_violation_inf, 0.5)

    def test_attach_kkt_propagates_nan_as_none(self):
        result = SolverResult(
            solver_name="dummy",
            problem_name="dummy",
            iterations=0,
            solve_time_ms=0.0,
            final_cost=0.0,
            eq_violation_inf=0.0,
            ineq_violation_inf=0.0,
            success=True,
        )
        kkt = KKTBreakdown(
            init=0.1,
            dyn=0.2,
            eq=0.3,
            ineq=0.4,
            dual=float("nan"),
            complementarity=float("nan"),
            stationarity=float("nan"),
            joint=0.4,
        )
        attach_kkt(result, kkt)
        self.assertIsNone(result.kkt_dual_violation_inf)
        self.assertIsNone(result.kkt_complementarity_inf)
        self.assertIsNone(result.kkt_stationarity_inf)
        self.assertEqual(result.kkt_residual_inf, 0.4)


if __name__ == "__main__":
    unittest.main()
