"""mim_solvers / Crocoddyl CSQP adapter.

Wraps a ``ProblemSpec`` as a sequence of Crocoddyl ``ActionModelAbstract``
instances backed by either JAX (``backend="jax"``, default) or CasADi
(``backend="casadi"``, when the problem ships
``metadata["casadi_builder"]``). The CasADi path runs compiled C++ per
``calc`` / ``calcDiff`` call; iter counts should be identical between
the two backends since the algorithm is unchanged. Solved with
``mim_solvers.SolverCSQP`` in both cases.

The JAX backend uses the "runtime ``t``" trick: every ``jax.jit``'d
value / gradient / Hessian function is built ONCE per problem with
``t`` as a runtime ``jnp.int32`` argument, and each per-stage
``ActionModelAbstract`` stores its own ``self._t`` and dispatches to
the shared compiled functions. Without this, the JAX backend would
trace + compile each function ~T times.

The Crocoddyl 3.x C++ ActionModel supports inline ``ng`` (inequality)
and ``nh`` (equality) constraints with bounds on a ``g`` vector and a
zero target on ``h``. We use:

* ``g(x, u) = ineq(x, u, theta, t)`` with bounds ``[-inf, 0]`` so the
  LIPA convention ``ineq <= 0`` is preserved.
* ``h(x, u) = eq(x, u, theta, t)`` (must equal zero).

CSQP from mim_solvers expects ``ng_T`` / ``nh_T`` for the terminal
model. We pad the same vectors at t=T.
"""

from __future__ import annotations

from timeit import default_timer as timer
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from tests.comparison.adapters import register
from tests.comparison.adapters.base import SolverAdapter
from tests.comparison.problem_spec import (
    ProblemSpec,
    SolverResult,
    make_failure_result,
    pack_solver_result,
)


def _import_crocoddyl():
    import crocoddyl  # noqa: F401

    return crocoddyl


def _import_mim_solvers():
    import mim_solvers  # noqa: F401

    return mim_solvers


def _build_shared_jax_jits(problem: ProblemSpec) -> dict:
    """Build the set of ``jax.jit``'d cost/dyn/eq/ineq value+grad+hessian
    functions ONCE per problem, parameterised by ``t`` as a runtime
    ``jnp.int32`` argument.

    Why: the previous code built ~15 jitted functions per *stage*, each
    with ``t`` captured as a Python constant in the closure. For T=100+
    stage MJX problems this meant the first solve triggered 100 × 15 =
    1500 separate JIT traces of the MJX dynamics graph, which is the
    pathology the IPOPT-MJX and fatrop-MJX adapters already solve via
    the "runtime t" trick. We adopt the same trick here.

    The returned dict has two sub-dicts:
    * ``shared["nt"]``: 9 functions used by non-terminal stages, all
      signatures ``(x, u, t)`` -> tensor:
        - ``cost``, ``cost_grad_x``, ``cost_grad_u``, ``cost_hess_xx``,
          ``cost_hess_uu``, ``cost_hess_xu``,
        - ``dyn``, ``dyn_jac_x``, ``dyn_jac_u``,
      plus 2 each (value + jac_x + jac_u) for ``eq`` and ``ineq`` if
      present (so up to 15 in the no-empty-constraint case).
    * ``shared["term"]``: terminal-stage functions, all signatures
      ``(x, t)`` -> tensor (``u`` is baked to zero internally so the
      LIPA pad convention is preserved):
        - ``cost``, ``cost_grad_x``, ``cost_hess_xx``,
        - ``eq``, ``eq_jac_x``, ``ineq``, ``ineq_jac_x``.

    Plus the per-problem shape constants ``ng_ineq``, ``ng_eq``, ``ng``
    (uniform across stages because we always pad to ``problem.eq_dim`` /
    ``problem.ineq_dim``).

    Note we still bake ``theta`` as a captured constant (CSQP doesn't
    support cross-stage Theta — the caller's ``solve()`` already rejects
    such problems with a clear notes= message).
    """
    theta = jnp.asarray(problem.Theta_init)
    m = problem.m

    cost_fn = problem.cost
    dyn_fn = problem.dynamics
    eq_fn = problem.equalities
    ineq_fn = problem.inequalities

    # ---- Non-terminal: (x, u, t) signatures -------------------------------
    def _cost_xu(x, u, t):
        return cost_fn(x, u, theta, t)

    def _dyn_xu(x, u, t):
        return dyn_fn(x, u, theta, t)

    nt: dict = {
        "cost": jax.jit(_cost_xu),
        "cost_grad_x": jax.jit(jax.grad(_cost_xu, argnums=0)),
        "cost_grad_u": jax.jit(jax.grad(_cost_xu, argnums=1)),
        "cost_hess_xx": jax.jit(jax.hessian(_cost_xu, argnums=0)),
        "cost_hess_uu": jax.jit(jax.hessian(_cost_xu, argnums=1)),
        "cost_hess_xu": jax.jit(jax.jacobian(jax.grad(_cost_xu, argnums=0), argnums=1)),
        "dyn": jax.jit(_dyn_xu),
        "dyn_jac_x": jax.jit(jax.jacobian(_dyn_xu, argnums=0)),
        "dyn_jac_u": jax.jit(jax.jacobian(_dyn_xu, argnums=1)),
    }

    if eq_fn is not None:

        def _eq_xu(x, u, t):
            return eq_fn(x, u, theta, t)

        nt["eq"] = jax.jit(_eq_xu)
        nt["eq_jac_x"] = jax.jit(jax.jacobian(_eq_xu, argnums=0))
        nt["eq_jac_u"] = jax.jit(jax.jacobian(_eq_xu, argnums=1))

    if ineq_fn is not None:

        def _ineq_xu(x, u, t):
            return ineq_fn(x, u, theta, t)

        nt["ineq"] = jax.jit(_ineq_xu)
        nt["ineq_jac_x"] = jax.jit(jax.jacobian(_ineq_xu, argnums=0))
        nt["ineq_jac_u"] = jax.jit(jax.jacobian(_ineq_xu, argnums=1))

    # ---- Terminal: (x, t) signatures, u baked to zero ---------------------
    def _cost_x(x, t):
        return cost_fn(x, jnp.zeros(m, dtype=x.dtype), theta, t)

    term: dict = {
        "cost": jax.jit(_cost_x),
        "cost_grad_x": jax.jit(jax.grad(_cost_x, argnums=0)),
        "cost_hess_xx": jax.jit(jax.hessian(_cost_x, argnums=0)),
    }

    if eq_fn is not None:

        def _eq_x(x, t):
            return eq_fn(x, jnp.zeros(m, dtype=x.dtype), theta, t)

        term["eq"] = jax.jit(_eq_x)
        term["eq_jac_x"] = jax.jit(jax.jacobian(_eq_x, argnums=0))

    if ineq_fn is not None:

        def _ineq_x(x, t):
            return ineq_fn(x, jnp.zeros(m, dtype=x.dtype), theta, t)

        term["ineq"] = jax.jit(_ineq_x)
        term["ineq_jac_x"] = jax.jit(jax.jacobian(_ineq_x, argnums=0))

    ng_ineq = problem.ineq_dim if ineq_fn is not None else 0
    ng_eq = problem.eq_dim if eq_fn is not None else 0

    return {
        "nt": nt,
        "term": term,
        "ng_ineq": ng_ineq,
        "ng_eq": ng_eq,
        "ng": ng_ineq + ng_eq,
    }


def _make_jax_action_model(problem: ProblemSpec, t: int, terminal: bool, shared: dict):
    """Build a Crocoddyl ActionModel that delegates to *shared* JAX jits.

    A "terminal" model has nu = 0 (no controls). The per-stage subclass
    holds ``self._t = jnp.int32(t)`` and dispatches to the shared
    pre-jitted functions, so a single JAX trace + compile is reused across
    all stages. This is the same "runtime t" trick the IPOPT-MJX and
    fatrop-MJX adapters use to keep JIT compile cost from scaling with T.

    CSQP / Crocoddyl call ``calcDiff`` after ``calc`` so we can be lazy
    about caching across the two.
    """
    crocoddyl = _import_crocoddyl()

    n = problem.n
    m = 0 if terminal else problem.m
    nh = 0  # we always go through g, not h

    ng_ineq = shared["ng_ineq"]
    ng_eq = shared["ng_eq"]
    ng = shared["ng"]
    # Pick the appropriate function bundle for this stage's terminal-ness.
    fns = shared["term"] if terminal else shared["nt"]
    t_j = jnp.int32(t)

    state = crocoddyl.StateVector(n)

    class _JaxModel(crocoddyl.ActionModelAbstract):
        def __init__(self_inner):  # noqa: N804
            crocoddyl.ActionModelAbstract.__init__(
                self_inner,
                state,
                m,
                max(n, 1),
                ng,
                nh,
            )
            self_inner._t = t_j
            if ng > 0:
                lb = np.empty(ng)
                ub = np.empty(ng)
                # Inequality block first: -inf <= g <= 0
                lb[:ng_ineq] = -np.inf
                ub[:ng_ineq] = 0.0
                # Equality block as two-sided ineq: 0 <= g <= 0
                lb[ng_ineq:] = 0.0
                ub[ng_ineq:] = 0.0
                self_inner.g_lb = lb
                self_inner.g_ub = ub

        def calc(self_inner, data, x, u=None):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            tau = self_inner._t
            if terminal:
                data.cost = float(fns["cost"](x_np, tau))
                data.xnext = x_np
                if ng > 0:
                    pieces = []
                    if ng_ineq > 0:
                        pieces.append(
                            np.asarray(
                                fns["ineq"](x_np, tau), dtype=np.float64
                            ).reshape(-1)
                        )
                    if ng_eq > 0:
                        pieces.append(
                            np.asarray(fns["eq"](x_np, tau), dtype=np.float64).reshape(
                                -1
                            )
                        )
                    data.g = np.concatenate(pieces) if pieces else np.empty(0)
            else:
                if u is None:
                    u_np = np.zeros(problem.m, dtype=np.float64)
                else:
                    u_np = np.asarray(u, dtype=np.float64).reshape(-1)
                data.cost = float(fns["cost"](x_np, u_np, tau))
                data.xnext = np.asarray(
                    fns["dyn"](x_np, u_np, tau), dtype=np.float64
                ).reshape(-1)
                if ng > 0:
                    pieces = []
                    if ng_ineq > 0:
                        pieces.append(
                            np.asarray(
                                fns["ineq"](x_np, u_np, tau), dtype=np.float64
                            ).reshape(-1)
                        )
                    if ng_eq > 0:
                        pieces.append(
                            np.asarray(
                                fns["eq"](x_np, u_np, tau), dtype=np.float64
                            ).reshape(-1)
                        )
                    data.g = np.concatenate(pieces) if pieces else np.empty(0)

        def calcDiff(self_inner, data, x, u=None):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            tau = self_inner._t
            if terminal:
                data.Lx = np.asarray(
                    fns["cost_grad_x"](x_np, tau), dtype=np.float64
                ).reshape(-1)
                data.Lxx = np.asarray(
                    fns["cost_hess_xx"](x_np, tau), dtype=np.float64
                ).reshape(n, n)
                if ng > 0:
                    gx_pieces = []
                    if ng_ineq > 0:
                        gx_pieces.append(
                            np.asarray(
                                fns["ineq_jac_x"](x_np, tau), dtype=np.float64
                            ).reshape(ng_ineq, n)
                        )
                    if ng_eq > 0:
                        gx_pieces.append(
                            np.asarray(
                                fns["eq_jac_x"](x_np, tau), dtype=np.float64
                            ).reshape(ng_eq, n)
                        )
                    data.Gx = np.vstack(gx_pieces) if gx_pieces else np.empty((0, n))
            else:
                if u is None:
                    u_np = np.zeros(problem.m, dtype=np.float64)
                else:
                    u_np = np.asarray(u, dtype=np.float64).reshape(-1)
                data.Lx = np.asarray(
                    fns["cost_grad_x"](x_np, u_np, tau), dtype=np.float64
                ).reshape(-1)
                data.Lxx = np.asarray(
                    fns["cost_hess_xx"](x_np, u_np, tau), dtype=np.float64
                ).reshape(n, n)
                data.Lu = np.asarray(
                    fns["cost_grad_u"](x_np, u_np, tau), dtype=np.float64
                ).reshape(-1)
                data.Luu = np.asarray(
                    fns["cost_hess_uu"](x_np, u_np, tau), dtype=np.float64
                ).reshape(m, m)
                data.Lxu = np.asarray(
                    fns["cost_hess_xu"](x_np, u_np, tau), dtype=np.float64
                ).reshape(n, m)
                data.Fx = np.asarray(
                    fns["dyn_jac_x"](x_np, u_np, tau), dtype=np.float64
                ).reshape(n, n)
                data.Fu = np.asarray(
                    fns["dyn_jac_u"](x_np, u_np, tau), dtype=np.float64
                ).reshape(n, m)
                if ng > 0:
                    gx_pieces = []
                    gu_pieces = []
                    if ng_ineq > 0:
                        gx_pieces.append(
                            np.asarray(
                                fns["ineq_jac_x"](x_np, u_np, tau), dtype=np.float64
                            ).reshape(ng_ineq, n)
                        )
                        gu_pieces.append(
                            np.asarray(
                                fns["ineq_jac_u"](x_np, u_np, tau), dtype=np.float64
                            ).reshape(ng_ineq, m)
                        )
                    if ng_eq > 0:
                        gx_pieces.append(
                            np.asarray(
                                fns["eq_jac_x"](x_np, u_np, tau), dtype=np.float64
                            ).reshape(ng_eq, n)
                        )
                        gu_pieces.append(
                            np.asarray(
                                fns["eq_jac_u"](x_np, u_np, tau), dtype=np.float64
                            ).reshape(ng_eq, m)
                        )
                    data.Gx = np.vstack(gx_pieces) if gx_pieces else np.empty((0, n))
                    data.Gu = np.vstack(gu_pieces) if gu_pieces else np.empty((0, m))

    return _JaxModel()


def _import_casadi():
    import casadi as ca  # local import so a missing CasADi only fails this backend

    return ca


def _make_casadi_action_model(problem: ProblemSpec, t: int, terminal: bool):
    """Build a Crocoddyl ActionModel that delegates to CasADi Functions.

    Same algorithm / layout as ``_make_jax_action_model`` (Lx / Lxx / Fx /
    Gx etc. populated identically), but the per-call ``calc`` / ``calcDiff``
    cost goes through compiled CasADi Functions instead of JIT-traced JAX
    ones. This avoids the ~200 ms Python-to-XLA dispatch tax per outer
    iter that dominates the JAX backend on the analytical problems.

    The CasADi Functions are built from ``problem.metadata["casadi_builder"]``
    SX expressions. Hessian sliceing follows the convention:

    * ``L_xx_xu`` = full ``hessian(L, [x; u])`` is (n+m)x(n+m) symmetric.
      Crocoddyl wants ``Lxx`` = upper-left n x n block, ``Luu`` = lower-right
      m x m block, ``Lxu`` = upper-right n x m block.

    The ``calc`` / ``calcDiff`` body is otherwise identical to the JAX
    path (same g concatenation order: ineq rows first, then eq rows).
    The terminal stage carries ``m_model = 0`` (no controls), and we feed
    a zero ``u`` SX into the builder to mirror the LIPA pad convention.
    """
    crocoddyl = _import_crocoddyl()
    ca = _import_casadi()

    builder = problem.metadata["casadi_builder"]
    n = problem.n
    m = 0 if terminal else problem.m  # Crocoddyl's view of nu
    m_full = problem.m  # The dim that the casadi builder always expects

    # Build per-stage SX expressions through the same builder used by
    # IPOPT / acados / fatrop. We always evaluate at ``t`` (the stage
    # index; T for terminal — matching the JAX path's ``jnp.int32(t)``
    # argument so the builder's ``t == T`` branch fires correctly).
    x_sx = ca.SX.sym("x", n)
    u_sx = ca.SX.sym("u", m_full)
    theta_sx = ca.SX.sym("theta", max(problem.theta_dim, 0))

    if terminal:
        # Mirror JAX's "u=0" convention: pass a zero SX into the builder
        # so any (x, u)-coupled terms collapse cleanly.
        u_for_builder = ca.SX.zeros(m_full, 1)
    else:
        u_for_builder = u_sx

    stage = builder(x_sx, u_for_builder, theta_sx, t)
    f_expr = stage["f"]
    next_x_expr = stage.get("next_x")
    eq_expr = stage.get("eq")
    ineq_expr = stage.get("ineq")

    # Pad eq / ineq to the JAX-path sizes so the per-stage ``ng`` layout
    # is identical to the JAX backend (CSQP / mim_solvers don't care about
    # per-stage uniformity, but matching exactly means iter counts should
    # be bit-for-bit identical between backends — that's the whole point
    # of this backend).
    ng_ineq = problem.ineq_dim if problem.inequalities is not None else 0
    ng_eq = problem.eq_dim if problem.equalities is not None else 0
    ng = ng_ineq + ng_eq

    # Concrete theta value carried as a CasADi DM constant.
    theta_dm = ca.DM(np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1))

    # If theta is empty, ca.substitute happily turns theta_sx into a 0x1
    # constant. Substitute it out so the resulting Functions only take
    # (x, u) and we don't have to feed theta on every call.
    def _sub(expr):
        if expr is None:
            return None
        # SX.substitute requires SX inputs; theta_sx might be 0x1.
        if problem.theta_dim > 0:
            return ca.substitute(expr, theta_sx, theta_dm)
        return expr

    f_expr = _sub(f_expr)
    next_x_expr = _sub(next_x_expr)
    eq_expr = _sub(eq_expr)
    ineq_expr = _sub(ineq_expr)

    # Pad eq / ineq to match the JAX-path sizing. The JAX path returns
    # ``jnp.zeros_like(...)`` at non-terminal stages where the CasADi
    # builder returns None — this preserves ng across stages, which
    # is what mim_solvers' inner ProxQP expects (its block layout
    # assumes a uniform per-stage constraint vector length).
    if ineq_expr is None or (hasattr(ineq_expr, "numel") and ineq_expr.numel() == 0):
        ineq_padded = ca.SX.zeros(ng_ineq, 1) if ng_ineq > 0 else ca.SX.zeros(0, 1)
    else:
        # Builder might return a row vector or a column; normalize to column.
        ineq_padded = ca.reshape(ineq_expr, ineq_expr.numel(), 1)
        if ineq_padded.numel() != ng_ineq:
            raise ValueError(
                f"casadi backend: at t={t}, ineq numel={ineq_padded.numel()} "
                f"!= problem.ineq_dim={ng_ineq}",
            )

    if eq_expr is None or (hasattr(eq_expr, "numel") and eq_expr.numel() == 0):
        eq_padded = ca.SX.zeros(ng_eq, 1) if ng_eq > 0 else ca.SX.zeros(0, 1)
    else:
        eq_padded = ca.reshape(eq_expr, eq_expr.numel(), 1)
        if eq_padded.numel() != ng_eq:
            raise ValueError(
                f"casadi backend: at t={t}, eq numel={eq_padded.numel()} "
                f"!= problem.eq_dim={ng_eq}",
            )

    # Combined (x; u) vector for stacked-Hessian sliceing.
    xu_sx = ca.vertcat(x_sx, u_sx) if m_full > 0 else x_sx

    # ---- Cost: value, gradient, Hessian ------------------------------------
    # We need:
    #   Lx = df/dx        (n,)
    #   Lu = df/du        (m_full,)
    #   Lxx = d2f/dx2     (n, n)
    #   Luu = d2f/du2     (m_full, m_full)
    #   Lxu = d2f/dx du   (n, m_full)  -- Crocoddyl convention
    #
    # We compute the full Hessian wrt [x; u] (which is symmetric and
    # square (n+m, n+m)) once, then slice it.
    grad_xu = ca.gradient(f_expr, xu_sx)
    hess_xu, _ = ca.hessian(f_expr, xu_sx)  # ca.hessian returns (H, grad)

    Lx_expr = grad_xu[:n]
    Lxx_expr = hess_xu[:n, :n]
    if m_full > 0:
        Lu_expr = grad_xu[n:]
        Luu_expr = hess_xu[n:, n:]
        Lxu_expr = hess_xu[:n, n:]
    else:
        Lu_expr = ca.SX.zeros(0, 1)
        Luu_expr = ca.SX.zeros(0, 0)
        Lxu_expr = ca.SX.zeros(n, 0)

    # ---- Dynamics: value + Jacobians ---------------------------------------
    # At terminal stages next_x is unused (mirroring the JAX path), but
    # we still need to give Crocoddyl *something* for ``xnext``; we use
    # ``x`` itself (identity dynamics) and zero Jacobians, same as JAX path.
    if terminal or next_x_expr is None:
        next_x_expr_used = x_sx
    else:
        next_x_expr_used = next_x_expr
    Fx_expr = ca.jacobian(next_x_expr_used, x_sx)
    Fu_expr = ca.jacobian(next_x_expr_used, u_sx) if m_full > 0 else ca.SX.zeros(n, 0)

    # ---- Combined g (ineq first, then eq) Jacobians ------------------------
    g_expr = (
        ca.vertcat(ineq_padded, eq_padded)
        if (ng_ineq + ng_eq) > 0
        else ca.SX.zeros(0, 1)
    )
    Gx_expr = ca.jacobian(g_expr, x_sx) if g_expr.numel() > 0 else ca.SX.zeros(0, n)
    Gu_expr = (
        ca.jacobian(g_expr, u_sx)
        if (g_expr.numel() > 0 and m_full > 0)
        else ca.SX.zeros(0, m_full)
    )

    # Build Functions. We use a single combined value Function and a
    # single combined derivative Function to minimize per-call CasADi
    # dispatch overhead (one call rather than 8).
    inputs = [x_sx, u_sx]
    val_fn = ca.Function(
        f"stage_val_{t}_{int(terminal)}",
        inputs,
        [f_expr, next_x_expr_used, g_expr],
        ["x", "u"],
        ["f", "xnext", "g"],
    )
    der_fn = ca.Function(
        f"stage_der_{t}_{int(terminal)}",
        inputs,
        [
            Lx_expr,
            Lu_expr,
            Lxx_expr,
            Luu_expr,
            Lxu_expr,
            Fx_expr,
            Fu_expr,
            Gx_expr,
            Gu_expr,
        ],
        ["x", "u"],
        ["Lx", "Lu", "Lxx", "Luu", "Lxu", "Fx", "Fu", "Gx", "Gu"],
    )

    state = crocoddyl.StateVector(n)

    class _CasadiModel(crocoddyl.ActionModelAbstract):
        def __init__(self_inner):  # noqa: N804
            crocoddyl.ActionModelAbstract.__init__(
                self_inner,
                state,
                m,
                max(n, 1),
                ng,
                0,
            )
            if ng > 0:
                lb = np.empty(ng)
                ub = np.empty(ng)
                lb[:ng_ineq] = -np.inf
                ub[:ng_ineq] = 0.0
                lb[ng_ineq:] = 0.0
                ub[ng_ineq:] = 0.0
                self_inner.g_lb = lb
                self_inner.g_ub = ub

        def calc(self_inner, data, x, u=None):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            if u is None:
                u_np = np.zeros(m_full, dtype=np.float64)
            else:
                u_np = np.asarray(u, dtype=np.float64).reshape(-1)
                if u_np.size < m_full:
                    # Crocoddyl can hand us a zero-length u in terminal calls;
                    # pad to m_full so the CasADi Function shape matches.
                    pad = np.zeros(m_full - u_np.size, dtype=np.float64)
                    u_np = np.concatenate([u_np, pad])
            f_val, next_x_val, g_val = val_fn(x_np, u_np)
            data.cost = float(f_val)
            if not terminal:
                data.xnext = np.asarray(next_x_val, dtype=np.float64).reshape(-1)
            else:
                data.xnext = x_np
            if ng > 0:
                data.g = np.asarray(g_val, dtype=np.float64).reshape(-1)

        def calcDiff(self_inner, data, x, u=None):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            if u is None:
                u_np = np.zeros(m_full, dtype=np.float64)
            else:
                u_np = np.asarray(u, dtype=np.float64).reshape(-1)
                if u_np.size < m_full:
                    pad = np.zeros(m_full - u_np.size, dtype=np.float64)
                    u_np = np.concatenate([u_np, pad])
            Lx_v, Lu_v, Lxx_v, Luu_v, Lxu_v, Fx_v, Fu_v, Gx_v, Gu_v = der_fn(x_np, u_np)
            data.Lx = np.asarray(Lx_v, dtype=np.float64).reshape(-1)
            data.Lxx = np.asarray(Lxx_v, dtype=np.float64).reshape(n, n)
            if not terminal:
                data.Lu = np.asarray(Lu_v, dtype=np.float64).reshape(-1)
                data.Luu = np.asarray(Luu_v, dtype=np.float64).reshape(m, m)
                data.Lxu = np.asarray(Lxu_v, dtype=np.float64).reshape(n, m)
                data.Fx = np.asarray(Fx_v, dtype=np.float64).reshape(n, n)
                data.Fu = np.asarray(Fu_v, dtype=np.float64).reshape(n, m)
            if ng > 0:
                data.Gx = np.asarray(Gx_v, dtype=np.float64).reshape(ng, n)
                if not terminal:
                    data.Gu = np.asarray(Gu_v, dtype=np.float64).reshape(ng, m)

    return _CasadiModel()


# Forward-rollout warm-start lives in tests.comparison.warm_starts;
# see that module's docstring for the rationale.
from tests.comparison.warm_starts import rollout_warm_start as _rollout_warm_start  # noqa: E402


class CsqpAdapter(SolverAdapter):
    """Crocoddyl ShootingProblem solved by mim_solvers' SolverCSQP.

    See the module docstring for what each knob does. All knobs are
    overrideable via constructor kwargs.
    """

    name = "csqp"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        with_callbacks: bool = False,
        # Backend for the per-stage callbacks. "jax" (default) keeps the
        # existing JAX-traced functions; "casadi" routes through the same
        # ``casadi_builder`` IPOPT/acados/fatrop use, dropping the
        # per-iter Python<->JAX dispatch cost from ~200 ms to ~ms on the
        # analytical problems. Iter counts should be identical between
        # backends since the algorithm is unchanged.
        backend: str = "jax",
        # Forward-roll U_init through problem.dynamics to get a feasible
        # starting trajectory before CSQP starts. ``None`` (the default)
        # picks per-problem: True for analytical problems (the shipped
        # X_init is dynamically-inconsistent and needs rolling), False
        # for MJX problems (the shipped reference X_init is a better
        # warm start than rolling u_ref through MJX dynamics).
        rollout_warm_start: Optional[bool] = None,
        # Inner-ProxQP defaults: keep mim_solvers' own tolerances rather
        # than the previous behaviour of mirroring `tol` into them.
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        # Smaller proximal regularizer (CSQP default 1e-6); see docstring.
        sigma: float = 1e-9,
        # Reset ProxQP state per outer iter — pairs with the smaller sigma.
        reset_rho: bool = True,
        reset_y: bool = True,
        # AL-style penalties on dynamics / path constraints (CSQP's
        # default value).
        mu_dynamic: float = 10.0,
        mu_constraint: float = 10.0,
        use_filter_line_search: bool = True,
        max_qp_iters: int = 1000,
        # Override the per-problem ``max_iter_overrides["csqp"]`` cap
        # that ships in some problems' metadata (MJX problems cap CSQP
        # at 200 to keep the benchmark bounded). Tuning passes need to
        # be able to lift this; the metadata cap is applied in
        # ``run_benchmark._run_one_in_process`` BEFORE this adapter is
        # constructed, so we can't see the original CLI ``--max-iter``
        # anymore. When set to a positive int, replaces ``max_iter``
        # outright. When ``None`` (default) the constructor-passed
        # ``max_iter`` is honored as-is.
        force_max_iter: Optional[int] = None,
        # Free-form pass-through for SolverCSQP attributes not promoted
        # to first-class kwargs (e.g. ``rho_sparse``,
        # ``adaptive_rho_tolerance``, ``filter_size``,
        # ``extra_iteration_for_last_kkt``,
        # ``lag_mul_inf_norm_coef``, ``equality_qp_initial_guess``).
        # Each ``{name: value}`` pair is set on the SolverCSQP instance
        # via ``setattr`` right before the timed ``solve`` call. Unknown
        # attributes silently no-op (mim_solvers' bindings are
        # tolerant). Use sparingly — preferred path is to promote a
        # frequently-tuned knob to a real kwarg.
        csqp_extra_options: Optional[dict] = None,
    ) -> None:
        self.max_iter = max_iter if force_max_iter is None else int(force_max_iter)
        self.tol = tol
        self.with_callbacks = with_callbacks
        if backend not in {"jax", "casadi"}:
            raise ValueError(f"backend must be 'jax' or 'casadi', got {backend!r}")
        self.backend = backend
        self.rollout_warm_start = rollout_warm_start
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.sigma = sigma
        self.reset_rho = reset_rho
        self.reset_y = reset_y
        self.mu_dynamic = mu_dynamic
        self.mu_constraint = mu_constraint
        self.use_filter_line_search = use_filter_line_search
        self.max_qp_iters = max_qp_iters
        self.csqp_extra_options = dict(csqp_extra_options) if csqp_extra_options else {}

    def is_available(self) -> tuple[bool, str]:
        try:
            _import_crocoddyl()
            _import_mim_solvers()
        except ImportError as e:
            return False, f"{e}"
        if self.backend == "casadi":
            try:
                _import_casadi()
            except ImportError as e:
                return False, f"casadi backend requested but casadi import failed: {e}"
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:
        from tests.comparison.problem_spec import effective_solver_tol

        tol = effective_solver_tol(problem, self.tol)
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name,
                problem.name,
                f"unavailable: {reason}",
            )

        if problem.theta_dim > 0:
            return make_failure_result(
                self.name,
                problem.name,
                f"CSQP/Crocoddyl does not support cross-stage Theta "
                f"(theta_dim={problem.theta_dim})",
            )

        crocoddyl = _import_crocoddyl()
        mim_solvers = _import_mim_solvers()

        if self.backend == "casadi":
            if "casadi_builder" not in problem.metadata:
                return make_failure_result(
                    self.name,
                    problem.name,
                    f"csqp[backend=casadi] needs problem.metadata['casadi_builder']; "
                    f"problem={problem.name} doesn't ship one (re-run with backend='jax').",
                )
            running_models = [
                _make_casadi_action_model(problem, t, terminal=False)
                for t in range(problem.T)
            ]
            terminal_model = _make_casadi_action_model(
                problem, problem.T, terminal=True
            )
        else:
            # JAX backend: build the shared jit'd function bundle ONCE for
            # the whole problem, then have every per-stage ActionModel
            # close over it and dispatch with its own ``t`` as a runtime
            # ``jnp.int32``. This keeps JIT compile cost from scaling
            # with T (critical for MJX problems with T=100+).
            shared = _build_shared_jax_jits(problem)
            running_models = [
                _make_jax_action_model(problem, t, terminal=False, shared=shared)
                for t in range(problem.T)
            ]
            terminal_model = _make_jax_action_model(
                problem,
                problem.T,
                terminal=True,
                shared=shared,
            )

        x0 = np.asarray(problem.x0, dtype=np.float64)
        shooting = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

        # Auto-pick rollout policy if the caller didn't override. MJX
        # problems ship a reference X_init that is the better warm
        # start; the analytical problems benefit from a forward
        # rollout of U_init.
        if self.rollout_warm_start is None:
            do_rollout = not problem.metadata.get("is_mjx", False)
        else:
            do_rollout = bool(self.rollout_warm_start)
        if do_rollout:
            xs_arr, us_arr = _rollout_warm_start(problem)
        else:
            xs_arr = np.asarray(problem.X_init, dtype=np.float64)
            us_arr = np.asarray(problem.U_init, dtype=np.float64)
        xs_init = [np.asarray(x, dtype=np.float64) for x in xs_arr]
        us_init = [np.asarray(u, dtype=np.float64) for u in us_arr]

        def _make_solver():
            s = mim_solvers.SolverCSQP(shooting)
            s.max_qp_iters = self.max_qp_iters
            s.with_callbacks = self.with_callbacks
            s.eps_abs = self.eps_abs
            s.eps_rel = self.eps_rel
            s.use_filter_line_search = self.use_filter_line_search
            s.sigma = self.sigma
            s.reset_rho = self.reset_rho
            s.reset_y = self.reset_y
            s.mu_dynamic = self.mu_dynamic
            s.mu_constraint = self.mu_constraint
            s.termination_tolerance = tol

            # Per-problem layers, applied in order so later layers
            # shadow earlier ones, ending with the CLI override:
            #   csqp_settings (shared, both backends)
            #   csqp_<backend>_settings (backend-specific)
            #   self.csqp_extra_options (CLI override)
            def _apply(d):
                for k, v in d.items():
                    try:
                        setattr(s, k, v)
                    except Exception:  # noqa: BLE001
                        # mim_solvers bindings are usually tolerant, but
                        # type-mismatched attributes (e.g. int vs float
                        # on a Boost.Python property) can raise. Skip
                        # silently — the user-visible report still tells
                        # the truth about how the solver behaved.
                        pass

            _apply(problem.metadata.get("csqp_settings", {}))
            _apply(problem.metadata.get(f"csqp_{self.backend}_settings", {}))
            _apply(self.csqp_extra_options)
            return s

        # Warm-up call to amortize JAX JIT compile so the timed run reflects solve cost.
        try:
            _make_solver().solve(xs_init, us_init, 1)
        except Exception:  # noqa: BLE001
            pass

        solver = _make_solver()
        # Per-iter recorder: mim_solvers' CallbackLogger collects per-iter
        # xs / us / cost / gap_norm / constraint_norm into a dict on
        # ``cb_log.convergence_data``. Cheap (Python-side bookkeeping)
        # and runs inside the timed window — the cost / eq / ineq
        # *history* arrays are then computed via histories_from_iterates
        # AFTER the timed solve, so the per-iter recording overhead is
        # excluded from solve_time_ms.
        try:
            mim_solvers_mod = _import_mim_solvers()
            cb_log = mim_solvers_mod.CallbackLogger()
            existing_cbs = (
                list(solver.getCallbacks()) if hasattr(solver, "getCallbacks") else []
            )
            solver.setCallbacks([*existing_cbs, cb_log])
        except Exception:  # noqa: BLE001
            cb_log = None

        start = timer()
        try:
            ok = bool(solver.solve(xs_init, us_init, self.max_iter))
            err = ""
        except Exception as e:  # noqa: BLE001
            ok = False
            err = f"{type(e).__name__}: {e}"
        solve_time_ms = 1e3 * (timer() - start)

        try:
            xs_out = np.asarray([np.asarray(x) for x in solver.xs])
            us_out = np.asarray([np.asarray(u) for u in solver.us])
            Theta = np.asarray(problem.Theta_init)
            iters = int(solver.iter)
        except Exception as e:  # noqa: BLE001
            xs_out = np.asarray(problem.X_init)
            us_out = np.asarray(problem.U_init)
            Theta = np.asarray(problem.Theta_init)
            iters = 0
            err = err or f"{type(e).__name__}: {e}"
            ok = False

        # Multiplier extraction.
        # * solver.lag_mul has shape (T+1, n): per-stage co-state for the
        #   dynamics defects. lag_mul[0] is unused (no dynamics into
        #   stage 0); lag_mul[1..T] multiplies the dynamics-defect
        #   constraint between stage k-1 and k. CSQP's convention
        #   (mirroring crocoddyl) is L = f + lag_mul^T (X[k] - dyn(...)),
        #   the same sign as IPOPT/fatrop and the OPPOSITE of
        #   evaluate_problem's defect = dyn(...) - X[k]. Hence sign-flip.
        # * solver.y has shape (T+1, ng) per stage where ng = ng_ineq +
        #   ng_eq (mim_solvers stacks ineq first, then eq — same order
        #   as our _make_jax_action_model and _make_casadi_action_model
        #   build above). Sign convention matches evaluate_problem
        #   (g <= 0 with multiplier >= 0; the equality block enters
        #   without sign massaging).
        # * Init defect: not exposed as a separate multiplier by CSQP
        #   (the initial state is constraint-pinned via
        #   ShootingProblem(x0=...)). We leave that slot at 0; the
        #   stationarity contribution from a satisfied init constraint
        #   is benign when the cost has no x0 dependence.
        try:
            T_problem = problem.T
            n = problem.n
            eq_dim = problem.eq_dim if problem.equalities is not None else 0
            ineq_dim = problem.ineq_dim if problem.inequalities is not None else 0
            lag_mul_arr = np.asarray(solver.lag_mul, dtype=np.float64)
            # Build eq stack [init_defect (n,); dyn_defects (T*n,);
            # user_eqs ((T+1)*eq_dim,)]
            eq_full_size = n + T_problem * n + (T_problem + 1) * eq_dim
            multipliers_eq = np.zeros(eq_full_size, dtype=np.float64)
            if lag_mul_arr.shape[0] == T_problem + 1 and lag_mul_arr.shape[1] == n:
                # dyn defects from stages 1..T (lag_mul[0] is the init
                # multiplier slot in CSQP, but CSQP doesn't actually
                # populate it because the init state is constraint-
                # pinned externally — keep at 0).
                # CSQP's lag_mul[k] for k >= 1 multiplies the dyn
                # defect between stages k-1 and k. Sign convention
                # mirrors crocoddyl's (L = f + lag_mul^T defect with
                # defect = X[k] - dyn(...)). To bring it into our
                # evaluate_problem convention (defect = dyn(...) -
                # X[k]), negate.
                multipliers_eq[n : n + T_problem * n] = -lag_mul_arr[1:].reshape(-1)
            # path-constraint multipliers (per-stage [ineq (ng_ineq); eq (ng_eq)])
            ineq_full_size = (T_problem + 1) * ineq_dim
            multipliers_ineq = (
                np.zeros(ineq_full_size, dtype=np.float64)
                if ineq_dim > 0
                else np.zeros(0)
            )
            if hasattr(solver, "y"):
                for t in range(T_problem + 1):
                    y_t = np.asarray(solver.y[t], dtype=np.float64).reshape(-1)
                    if y_t.size != ineq_dim + eq_dim:
                        continue  # shape mismatch; skip
                    if ineq_dim > 0:
                        multipliers_ineq[t * ineq_dim : (t + 1) * ineq_dim] = y_t[
                            :ineq_dim
                        ]
                    if eq_dim > 0:
                        base = n + T_problem * n
                        multipliers_eq[base + t * eq_dim : base + (t + 1) * eq_dim] = (
                            y_t[ineq_dim : ineq_dim + eq_dim]
                        )
        except Exception:  # noqa: BLE001
            multipliers_eq = None
            multipliers_ineq = None

        # Per-iter histories from the CallbackLogger's convergence_data.
        iterates_xut = None
        if cb_log is not None:
            try:
                cd = cb_log.convergence_data
                xs_hist = list(cd.get("xs", [])) if hasattr(cd, "get") else []
                us_hist = list(cd.get("us", [])) if hasattr(cd, "get") else []
                if xs_hist and us_hist and len(xs_hist) == len(us_hist):
                    iterates_xut = []
                    for X_iter, U_iter in zip(xs_hist, us_hist, strict=False):
                        Xi = np.asarray(
                            [np.asarray(x, dtype=np.float64) for x in X_iter],
                            dtype=np.float64,
                        )
                        Ui = np.asarray(
                            [np.asarray(u, dtype=np.float64) for u in U_iter],
                            dtype=np.float64,
                        )
                        iterates_xut.append((Xi, Ui, np.asarray(Theta)))
            except Exception:  # noqa: BLE001
                iterates_xut = None

        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=xs_out,
            U=us_out,
            Theta=Theta,
            iterations=iters,
            solve_time_ms=solve_time_ms,
            success=ok,
            notes=err,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq,
            iterates_xut=iterates_xut,
        )


@register("csqp")
def _factory(**kwargs) -> SolverAdapter:
    return CsqpAdapter(**kwargs)
