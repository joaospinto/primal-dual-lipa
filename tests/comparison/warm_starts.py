"""Shared warm-start helpers used by multiple comparison adapters.

These helpers were previously duplicated across the adapter modules; this
module consolidates them so the canonical implementations live in one
place. Every helper returns numpy arrays so the (CasADi / Crocoddyl /
acados) consumers don't need a JAX -> numpy step on their side.

Helpers
-------
``rollout_warm_start(problem, *, fallback_to_x_init=False)``
    Forward-propagate ``problem.U_init`` through ``problem.dynamics``
    from ``problem.x0`` to produce a dynamically-consistent
    ``(X, U)`` warm start. With ``fallback_to_x_init=True``, if the
    rollout produces non-finite values (or the call raises), falls back
    to ``(problem.X_init, problem.U_init)``. With the default
    ``False``, propagates non-finite values to the caller (matching the
    historical behaviour of the SQP/IP analytical adapters).

``linspace_to_extracted_goal_warm_start(problem)``
    JAX-only inference path. If ``problem`` looks like ``X_init =
    tile(x0)`` AND has a terminal equality of dimension ``problem.n``
    that almost certainly encodes ``x_T - goal == 0``, infer ``goal``
    by evaluating the equality at ``(x0, 0, theta, T)`` and return
    ``(linspace(x0, goal, T+1), U_init)``. Otherwise return ``(X_init,
    U_init)`` unchanged. Used by ``ipopt_casadi`` for its ``"auto"``
    warm-start strategy.

``linspace_to_casadi_extracted_goal_warm_start(problem)``
    CasADi-builder inference path. If the problem has a
    ``casadi_builder`` whose terminal equality has shape ``(n,)`` and
    is exactly affine with identity Jacobian (i.e. ``x_T - goal == 0``
    in closed form), infer ``goal`` from the constant offset and return
    ``(np.linspace(x0, goal, T+1), U_init)``. Otherwise return
    ``(X_init, U_init)`` unchanged. Used by the analytical ``fatrop``
    adapter.
"""

from __future__ import annotations

import numpy as np

from tests.comparison.problem_spec import ProblemSpec


def rollout_warm_start(
    problem: ProblemSpec,
    *,
    fallback_to_x_init: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-roll ``U_init`` through ``problem.dynamics`` from ``x0``.

    Why: a problem may ship with a dynamically-infeasible ``X_init``.
    For SQP / interior-point NLP solvers (acados, CSQP, IPOPT, fatrop,
    SIP) the first linearization at such a warm start has a large
    per-stage defect residual, which can make the QP/Newton system
    infeasible (SQP) or route filter-IPMs into restoration. Rolling
    ``U_init`` forward through the actual dynamics produces an
    ``(X, U)`` pair with zero per-stage defects, so the solver only
    has to drive any terminal equality and inequality residuals.

    LIPA tolerates the bad warm start natively because its primal-dual
    interior-point updates ``X`` and ``U`` jointly with slack-variable
    backpressure on the dynamics defect.

    Parameters
    ----------
    problem
        The ``ProblemSpec``. Reads ``x0``, ``T``, ``m``, ``Theta_init``,
        ``dynamics``, ``U_init`` (and ``X_init`` if ``fallback_to_x_init``).
    fallback_to_x_init
        If True, when the rollout produces a non-finite value or
        ``problem.dynamics`` raises, fall back to
        ``(problem.X_init, problem.U_init)``. This preserves the
        defensive behaviour of the MJX adapters whose dynamics can
        diverge on first call. Default ``False`` preserves the
        historical no-fallback behaviour of the analytical adapters.
    """
    # JAX is imported lazily so the helper stays importable in
    # environments that load this file before JAX is available.
    import jax.numpy as jnp

    T, m = problem.T, problem.m
    x = np.asarray(problem.x0, dtype=np.float64)
    theta_j = jnp.asarray(problem.Theta_init)
    U_init = np.asarray(problem.U_init, dtype=np.float64)

    xs = [x.copy()]
    us = [np.asarray(U_init[t], dtype=np.float64) for t in range(T)]
    try:
        for t in range(T):
            x = np.asarray(
                problem.dynamics(
                    jnp.asarray(x),
                    jnp.asarray(us[t]),
                    theta_j,
                    jnp.int32(t),
                ),
                dtype=np.float64,
            )
            if fallback_to_x_init and not np.all(np.isfinite(x)):
                X_init = np.asarray(
                    problem.X_init,
                    dtype=np.float64,
                ).reshape(T + 1, problem.n)
                return X_init, np.asarray(U_init).reshape(T, m)
            xs.append(x.copy())
    except (ValueError, RuntimeError, ArithmeticError, TypeError):
        # MJX integrator divergence / numeric overflow / shape mismatch.
        # MemoryError / NameError / AttributeError still propagate so
        # genuine bugs are visible instead of being silently masked.
        if fallback_to_x_init:
            X_init = np.asarray(
                problem.X_init,
                dtype=np.float64,
            ).reshape(T + 1, problem.n)
            return X_init, np.asarray(U_init).reshape(T, m)
        raise
    return np.asarray(xs), np.asarray(us)


def linspace_to_extracted_goal_warm_start(
    problem: ProblemSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Linspace warm-start that targets an extracted terminal goal.

    Strategy: if the LIPA-shipped ``X_init`` is the degenerate
    ``tile(x0)`` ("stay put forever"), AND the problem has a terminal
    user equality of dimension ``n`` whose value at ``(x0, 0, theta, T)``
    is non-zero (so it almost certainly encodes ``x_T - goal == 0``),
    infer ``goal = x0 - eq(x0, 0, theta, T)`` and return
    ``(linspace(x0, goal, T+1), U_init)``.

    Otherwise return ``(X_init, U_init)`` unchanged.

    Why: canonical ``tile(x0)`` warm starts can make filter-based
    IPMs immediately enter restoration trying to satisfy a terminal
    equality from a stationary trajectory; a linspace warm start
    respects the terminal equality at iteration 0 and tends to be
    much more robust for those solvers. Problems whose shipped
    ``X_init`` already encodes a non-degenerate trajectory, or that
    do not have a terminal equality of dimension ``n``, fall through
    to ``X_init`` unchanged.
    """
    import jax.numpy as jnp

    X_init = np.asarray(problem.X_init, dtype=np.float64)
    U_init = np.asarray(problem.U_init, dtype=np.float64)
    x0 = np.asarray(problem.x0, dtype=np.float64)

    # Heuristic 1: is X_init the degenerate "stay-put" tile of x0?
    is_tile = np.max(np.abs(X_init - x0)) < 1e-10
    # Heuristic 2: does the problem have a terminal equality of dim n
    # (i.e. plausibly ``x_T - goal == 0``)?
    has_terminal_state_eq = (
        problem.equalities is not None and problem.eq_dim == problem.n
    )
    if not (is_tile and has_terminal_state_eq):
        return X_init, U_init

    # Infer goal by querying the user equality at t=T evaluated at
    # ``(x0, 0, theta)``. If equalities ~= ``x_T - goal``, the result
    # is ``x0 - goal``, so ``goal = x0 - eq(x0, 0, theta, T)``.
    theta_j = jnp.asarray(problem.Theta_init)
    u_zero = jnp.zeros(problem.m)
    eq_T = np.asarray(
        problem.equalities(
            jnp.asarray(x0),
            u_zero,
            theta_j,
            jnp.int32(problem.T),
        ),
        dtype=np.float64,
    )
    if not np.any(np.abs(eq_T) > 1e-9):
        # Terminal equality already satisfied at x0 — nothing to interp.
        return X_init, U_init
    inferred_goal = x0 - eq_T
    X_ws = np.linspace(x0, inferred_goal, problem.T + 1)
    return X_ws, U_init


def _extract_terminal_goal_via_casadi(problem: ProblemSpec):
    """Recover the implicit terminal goal ``g`` from a CasADi-builder
    terminal equality of the form ``x_T - g = 0``.

    Returns the goal as an ``np.ndarray`` of length ``problem.n`` if the
    problem has a ``casadi_builder`` whose terminal equality is affine
    in ``x`` with identity Jacobian; ``None`` otherwise.
    """
    import casadi as ca

    builder = problem.metadata.get("casadi_builder")
    if builder is None:
        return None
    n, m, T = problem.n, problem.m, problem.T
    x_sx = ca.SX.sym("x", n)
    u_sx = ca.SX.sym("u", m)
    theta_sx = ca.SX.zeros(0)
    try:
        stage_T = builder(x_sx, u_sx, theta_sx, T)
    except (ValueError, RuntimeError, TypeError, AttributeError, KeyError):
        # The builder either rejects an SX symbol or returns an
        # unexpected shape. Genuine bugs (MemoryError, NameError, ...)
        # propagate so they don't get silently masked.
        return None
    eq_T = stage_T.get("eq")
    if eq_T is None or eq_T.numel() != n:
        # We need the terminal eq to be exactly n rows for ``x_T = goal``
        # to be a clean substitution.
        return None
    eq_fn = ca.Function("ftgoal_eq", [x_sx, u_sx], [eq_T])
    eq_at_zero = np.asarray(eq_fn(np.zeros(n), np.zeros(m))).reshape(-1)
    jac = ca.jacobian(eq_T, x_sx)
    jac_fn = ca.Function("ftgoal_jac", [x_sx, u_sx], [jac])
    jac_at_zero = np.asarray(jac_fn(np.zeros(n), np.zeros(m)))
    # If jac is identity and eq is affine, then eq(x) = x - goal so
    # goal = -eq(0). Verify by checking eq at a second point.
    if not np.allclose(jac_at_zero, np.eye(n), atol=1e-12):
        return None
    rng = np.random.default_rng(0)
    x_test = rng.standard_normal(n)
    eq_at_test = np.asarray(eq_fn(x_test, np.zeros(m))).reshape(-1)
    expected = x_test + eq_at_zero  # x_test - (-eq_at_zero) = x_test - goal
    if not np.allclose(eq_at_test, expected, atol=1e-10):
        return None
    return -eq_at_zero


def linspace_to_casadi_extracted_goal_warm_start(
    problem: ProblemSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a ``(X_init, U_init)`` warm start via CasADi-builder goal extraction.

    For problems whose terminal equality pins ``x_T = goal`` (and the
    extraction succeeds — see ``_extract_terminal_goal_via_casadi``),
    return ``(np.linspace(x0, goal, T+1), problem.U_init)``. For
    problems with no extractable terminal goal, return ``(problem.X_init,
    problem.U_init)`` unchanged.

    Used by the analytical fatrop adapter; see ``fatrop.py``'s module
    docstring for the rationale.
    """
    X_lipa = np.asarray(problem.X_init, dtype=np.float64).reshape(
        problem.T + 1,
        problem.n,
    )
    U_lipa = np.asarray(problem.U_init, dtype=np.float64).reshape(
        problem.T,
        problem.m,
    )
    goal = _extract_terminal_goal_via_casadi(problem)
    if goal is None:
        return X_lipa, U_lipa
    x0 = np.asarray(problem.x0, dtype=np.float64).reshape(problem.n)
    X_lin = np.linspace(x0, goal, problem.T + 1)
    return X_lin, U_lipa
