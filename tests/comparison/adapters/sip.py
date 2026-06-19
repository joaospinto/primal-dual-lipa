"""sip_python adapter for the analytical comparison suite.

Builds the multi-shooting NLP from a ``ProblemSpec`` and solves it via
``sip_python.Solver``. Decision vector::

    z = (vec(X), vec(U), Theta)

``theta_dim`` may be zero or positive; unlike the OCP-structured
solvers, sip_python is a generic NLP solver and doesn't care whether a
component of ``z`` is stage-local or cross-stage.

Two model-evaluation backends:

* ``backend="jax"`` (default): cost / equality / inequality come from
  the ``ProblemSpec`` JAX functions; gradients, Jacobians and the
  Hessian are built via ``jax.grad`` / ``jax.jacrev`` / ``jax.hessian``
  and JIT-compiled once per solve. Sparsity is detected from a small
  number of dense probe evaluations at perturbed warm starts.
* ``backend="casadi"``: the model is built from the problem's
  ``metadata["casadi_builder"]``. Sparsity comes from CasADi's
  symbolic graph. Wall-clock is typically several times faster than
  the JAX backend on the analytical suite.

The adapter always supplies SIP with the Lagrangian Hessian. SIP's
solver-side regularization handles indefiniteness; this adapter does
not apply Schur/eigenvalue projection to the Hessian blocks.

Conventions (from sip_python's tests/test_simple_constrained_lqr.py):

* Jacobians as ``scipy.sparse.csr_matrix`` of shape ``(c_dim, x_dim)``;
  reinterpreting CSR data as CSC yields the transpose, hence
  ``ProblemDimensions.is_jacobian_c_transposed = True``.
* Hessian as ``scipy.sparse.csc_matrix`` of shape ``(x_dim, x_dim)``
  containing the upper triangle of a PSD approximation.

The sparsity pattern is determined once per solve from dense JAX
evaluations at perturbed warm starts. At each callback we re-evaluate
the dense Jacobian/Hessian and overwrite ``.data`` of the pre-built
sparse matrices.
"""

from __future__ import annotations

from timeit import default_timer as timer
from typing import Optional

import numpy as np

from tests.comparison.adapters import register
from tests.comparison.adapters.base import SolverAdapter
from tests.comparison.problem_spec import (
    ProblemSpec,
    SolverResult,
    make_failure_result,
    pack_solver_result,
)


def _import_sip():
    import sip_python  # noqa: F401

    return sip_python


def _import_jax():
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401

    # SIP's QDLDL backend does the linear solves in float64; if JAX runs in
    # float32 we'd silently degrade convergence. Match the test files.
    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def _slice_iterate(
    z_val: np.ndarray, problem: ProblemSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m, T, td = problem.n, problem.m, problem.T, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    X = z_val[:nx].reshape(T + 1, n)
    U = z_val[nx : nx + nu].reshape(T, m)
    Theta = z_val[nx + nu : nx + nu + td] if td > 0 else np.zeros(0, dtype=z_val.dtype)
    return X, U, Theta


def _initial_decision_vector(problem: ProblemSpec) -> np.ndarray:
    X_init = np.asarray(problem.X_init, dtype=np.float64).copy()
    return np.concatenate(
        [
            X_init.reshape(-1),
            np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
            np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1),
        ]
    )


def _import_casadi():
    import casadi as ca  # local import so a missing CasADi only fails this backend

    return ca


_ADAPTER_ONLY_SETTING_KEYS = frozenset({"time_limit_s"})


def _merge_setting_dicts(*settings_dicts: dict) -> dict:
    merged: dict = {}
    for settings_dict in settings_dicts:
        if not settings_dict:
            continue
        for key, value in settings_dict.items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
                and key not in _ADAPTER_ONLY_SETTING_KEYS
            ):
                merged[key] = _merge_setting_dicts(merged[key], value)
            else:
                merged[key] = value
    return merged


def _apply_sip_settings(settings, settings_dict: dict) -> None:
    for key, value in settings_dict.items():
        if key in _ADAPTER_ONLY_SETTING_KEYS:
            continue
        target = getattr(settings, key)
        if isinstance(value, dict):
            for attr, attr_value in value.items():
                setattr(target, attr, attr_value)
        else:
            setattr(settings, key, value)


def _custom_time_limit_s(*settings_dicts: dict) -> float:
    time_limit_s = float("inf")
    for settings_dict in settings_dicts:
        if not settings_dict or "time_limit_s" not in settings_dict:
            continue
        value = settings_dict["time_limit_s"]
        if value is None:
            time_limit_s = float("inf")
        else:
            time_limit_s = float(value)
        if time_limit_s < 0.0:
            raise ValueError(f"time_limit_s must be nonnegative, got {value!r}")
    return time_limit_s


def _initialize_slacks_and_duals(
    g_init: np.ndarray,
    initial_mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interior slack/dual guesses consistent with ``g(x) + s = 0``.

    SIP models inequalities as ``g(x) + s = 0, s > 0, z > 0``. Starting
    every slack at one is harmless for small toy constraints, but it can
    create huge artificial infeasibility for MJX warm starts where inactive
    constraints often have large negative margins. Match the primal residual
    whenever possible and choose ``z`` on the initial central path.
    """
    if g_init.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    mu = max(float(initial_mu), 1e-16)
    floor = max(mu, 1e-8)
    s_init = np.maximum(-np.asarray(g_init, dtype=np.float64), floor)
    z_init = mu / s_init
    return s_init, z_init


def _strictly_positive_warm_start(
    warm_start: np.ndarray | None,
    *,
    floor: float,
) -> np.ndarray | None:
    if warm_start is None:
        return None
    arr = np.asarray(warm_start, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return None
    return np.maximum(arr, floor)


def _warm_start_vector(warm_start: dict, key: str, size: int) -> np.ndarray | None:
    value = warm_start.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size or not np.all(np.isfinite(arr)):
        return None
    return arr


def _initialize_eq_multipliers_from_warm_start(
    warm_start: dict,
    *,
    n: int,
    T: int,
    eq_dim: int,
) -> np.ndarray:
    """Return SIP-internal equality multipliers from a warm-start dict.

    ``Y_init`` and flattened ``Y_dyn`` use the canonical benchmark
    convention: init residual ``X[0] - x0`` and dynamics residual
    ``dyn(x, u) - X[1:]``. SIP's internal dynamics equality is the
    opposite sign, ``X[1:] - dyn(x, u)``, hence the sign flip.

    For compatibility with LIPA caches, also accept ``Y_dyn`` with
    ``(T + 1) * n`` entries. In that layout row 0 is LIPA's internal
    initial multiplier, so the canonical init multiplier is ``-row0``.
    """
    y_full = np.zeros(n + T * n + (T + 1) * eq_dim, dtype=np.float64)

    y_init_init = _warm_start_vector(warm_start, "Y_init", n)
    y_dyn_value = warm_start.get("Y_dyn")
    if y_dyn_value is not None:
        y_dyn_arr = np.asarray(y_dyn_value, dtype=np.float64).reshape(-1)
        if y_dyn_arr.size == T * n and np.all(np.isfinite(y_dyn_arr)):
            y_full[n : n + T * n] = -y_dyn_arr
        elif y_dyn_arr.size == (T + 1) * n and np.all(np.isfinite(y_dyn_arr)):
            if y_init_init is None:
                y_full[:n] = -y_dyn_arr[:n]
            y_full[n : n + T * n] = -y_dyn_arr[n:]
    if y_init_init is not None:
        y_full[:n] = y_init_init

    if eq_dim > 0:
        y_eq_init = _warm_start_vector(
            warm_start,
            "Y_eq",
            (T + 1) * eq_dim,
        )
        if y_eq_init is not None:
            y_full[n + T * n :] = y_eq_init
    return y_full


def _make_warm_start_out(
    *,
    multipliers_eq_full: np.ndarray | None,
    multipliers_ineq_full: np.ndarray | None,
    slacks_full: np.ndarray | None,
    n: int,
    T: int,
    eq_dim: int,
) -> dict | None:
    warm_start_out: dict[str, np.ndarray] = {}
    dyn_size = T * n
    dyn_start = n
    dyn_stop = dyn_start + dyn_size
    if multipliers_eq_full is not None and multipliers_eq_full.size >= n:
        warm_start_out["Y_init"] = np.asarray(
            multipliers_eq_full[:n],
            dtype=np.float64,
        ).copy()
    if multipliers_eq_full is not None and multipliers_eq_full.size >= dyn_stop:
        warm_start_out["Y_dyn"] = np.asarray(
            multipliers_eq_full[dyn_start:dyn_stop],
            dtype=np.float64,
        ).copy()
        eq_size = (T + 1) * eq_dim
        if eq_size > 0 and multipliers_eq_full.size - dyn_stop == eq_size:
            warm_start_out["Y_eq"] = np.asarray(
                multipliers_eq_full[dyn_stop:],
                dtype=np.float64,
            ).copy()
    if slacks_full is not None:
        warm_start_out["S"] = np.asarray(slacks_full, dtype=np.float64).copy()
    if multipliers_ineq_full is not None:
        warm_start_out["Z"] = np.asarray(
            multipliers_ineq_full,
            dtype=np.float64,
        ).copy()
    return warm_start_out or None


def _sip_status_notes(output) -> str:
    try:
        return (
            f"{output.exit_status}; "
            f"sip_primal={float(output.max_primal_violation):.3e}; "
            f"sip_dual={float(output.max_dual_violation):.3e}"
        )
    except Exception:  # noqa: BLE001
        return str(output.exit_status)


def _build_casadi_nlp(problem: ProblemSpec):
    """Build the flat NLP via the problem's ``casadi_builder``.

    Mirrors ``_build_jax_nlp`` but everything happens symbolically in
    CasADi SX. Returns a dict bundling:

    * ``f_fn(z)``       — scalar cost ``casadi.Function``
    * ``c_fn(z)``       — equality residual ``casadi.Function``
                          (init defect + dynamics defects + non-None eq rows).
    * ``g_fn(z)``       — inequality residual ``casadi.Function``,
                          convention ``g <= 0`` (only stages where
                          ``casadi_builder`` returns a non-None
                          ``ineq``).
    * ``grad_f_fn(z)``  — gradient of f, dense (n_z,) output.
    * ``jac_c_fn(z)``   — Jacobian of c, CasADi-sparse.
    * ``jac_g_fn(z)``   — Jacobian of g, CasADi-sparse.
    * ``hess_lag_fn(z, y, z_mult)`` — Lagrangian Hessian function with
                          CasADi-sparse output.
    * ``y_dim``, ``s_dim``, ``x_dim`` — dimensions.
    * ``jac_c_sparsity`` / ``jac_g_sparsity`` — symbolic CasADi
                          sparsity patterns (used to build the
                          ``scipy.sparse`` template buffers).

    The decision vector layout matches the JAX path exactly:
    ``z = [vec(X) (n*(T+1)), vec(U) (m*T), Theta (td)]``.
    """
    ca = _import_casadi()
    casadi_builder = problem.metadata["casadi_builder"]

    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    x_dim = nx + nu + td

    X_sx = ca.SX.sym("X", n, T + 1)
    U_sx = ca.SX.sym("U", m, T)
    Theta_sx = ca.SX.sym(
        "Theta", max(td, 1)
    )  # CasADi can't build a 0-length sym; slice it off below

    z_pieces = [ca.reshape(X_sx, -1, 1), ca.reshape(U_sx, -1, 1)]
    if td > 0:
        z_pieces.append(Theta_sx[:td])
    z = ca.vertcat(*z_pieces)

    theta_arg = Theta_sx[:td] if td > 0 else ca.SX.zeros(0)

    # Substitute the Theta_sx symbol with a 0-length placeholder when
    # td == 0 so the resulting Functions only take z as input.
    # Otherwise theta_arg is just a slice of Theta_sx and is part of z.
    f_total = ca.SX(0.0)
    c_pieces = []  # init defect + dyn defects + non-None eq rows
    g_pieces = []  # non-None ineq rows

    # Initial-state defect (always present).
    c_pieces.append(X_sx[:, 0] - ca.DM(np.asarray(problem.x0)))

    for t in range(T + 1):
        u_t = U_sx[:, t] if t < T else ca.SX.zeros(m, 1)
        stage = casadi_builder(X_sx[:, t], u_t, theta_arg, t)
        f_total = f_total + stage["f"]
        if t < T:
            # Dynamics defect: x_{t+1} - f(x_t, u_t, theta, t) = 0
            c_pieces.append(X_sx[:, t + 1] - stage["next_x"])
        eq_t = stage.get("eq")
        if eq_t is not None and (not hasattr(eq_t, "numel") or eq_t.numel() > 0):
            c_pieces.append(ca.reshape(eq_t, eq_t.numel(), 1))
        ineq_t = stage.get("ineq")
        if ineq_t is not None and (not hasattr(ineq_t, "numel") or ineq_t.numel() > 0):
            g_pieces.append(ca.reshape(ineq_t, ineq_t.numel(), 1))

    c_expr = ca.vertcat(*c_pieces) if c_pieces else ca.SX.zeros(0, 1)
    g_expr = ca.vertcat(*g_pieces) if g_pieces else ca.SX.zeros(0, 1)
    y_dim = c_expr.numel()
    s_dim = g_expr.numel()

    # Autodiff for the global-graph pieces (cost, jacobians).
    grad_f = ca.gradient(f_total, z)
    jac_c = ca.jacobian(c_expr, z) if y_dim > 0 else ca.SX.zeros(0, x_dim)
    jac_g = ca.jacobian(g_expr, z) if s_dim > 0 else ca.SX.zeros(0, x_dim)

    y_sym = ca.SX.sym("Y", y_dim) if y_dim > 0 else ca.SX.zeros(0, 1)
    z_sym = ca.SX.sym("Z", s_dim) if s_dim > 0 else ca.SX.zeros(0, 1)
    lag_expr = f_total
    if y_dim > 0:
        lag_expr = lag_expr + ca.dot(y_sym, c_expr)
    if s_dim > 0:
        lag_expr = lag_expr + ca.dot(z_sym, g_expr)
    H_lag, _ = ca.hessian(lag_expr, z)

    f_fn = ca.Function("sip_cost_val", [z], [f_total], ["z"], ["f"])
    c_fn = ca.Function("sip_eq_val", [z], [c_expr], ["z"], ["c"])
    g_fn = ca.Function("sip_ineq_val", [z], [g_expr], ["z"], ["g"])
    grad_f_fn = ca.Function("sip_cost_grad", [z], [grad_f], ["z"], ["grad_f"])
    jac_c_fn = ca.Function("sip_eq_jac", [z], [jac_c], ["z"], ["jac_c"])
    jac_g_fn = ca.Function("sip_ineq_jac", [z], [jac_g], ["z"], ["jac_g"])
    hess_lag_inputs = [z]
    hess_lag_input_names = ["z"]
    if y_dim > 0:
        hess_lag_inputs.append(y_sym)
        hess_lag_input_names.append("y")
    if s_dim > 0:
        hess_lag_inputs.append(z_sym)
        hess_lag_input_names.append("z_mult")
    hess_lag_fn = ca.Function(
        "sip_lagrangian_hess",
        hess_lag_inputs,
        [H_lag],
        hess_lag_input_names,
        ["H"],
    )

    return {
        "f_fn": f_fn,
        "c_fn": c_fn,
        "g_fn": g_fn,
        "grad_f_fn": grad_f_fn,
        "jac_c_fn": jac_c_fn,
        "jac_g_fn": jac_g_fn,
        "hess_lag_fn": hess_lag_fn,
        "hess_lag_sparsity": H_lag.sparsity(),
        "x_dim": x_dim,
        "y_dim": y_dim,
        "s_dim": s_dim,
        "jac_c_sparsity": jac_c.sparsity() if y_dim > 0 else None,
        "jac_g_sparsity": jac_g.sparsity() if s_dim > 0 else None,
    }


class SipAdapter(SolverAdapter):
    """sip_python solver adapter."""

    name = "sip"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        # Default penalty / barrier ramp. Per-problem overrides via
        # problem.metadata["sip_settings"] dial in different schedules
        # for problems whose defaults don't converge.
        penalty_parameter_increase_factor: float = 2.0,
        mu_update_factor: float = 0.9,
        initial_mu: float = 1e-1,
        initial_penalty_parameter: float = 1.0,
        print_logs: bool = False,
        timeout_s: float | None = None,
        sip_extra_settings: Optional[dict] = None,
        # Backend for per-iter model evaluations. "casadi" reuses the
        # problem's casadi_builder (same one IPOPT / acados / fatrop /
        # CSQP use) and is several times faster wall-clock on
        # analytical problems while producing the same iter counts.
        # MJX problems have no casadi_builder, so requesting
        # "unavailable" SolverResult.
        backend: str = "jax",
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.penalty_parameter_increase_factor = float(
            penalty_parameter_increase_factor
        )
        self.mu_update_factor = float(mu_update_factor)
        self.initial_mu = float(initial_mu)
        self.initial_penalty_parameter = float(initial_penalty_parameter)
        self.print_logs = bool(print_logs)
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.sip_extra_settings = sip_extra_settings or {}
        if backend not in {"jax", "casadi"}:
            raise ValueError(f"backend must be 'jax' or 'casadi', got {backend!r}")
        self.backend = backend

    def is_available(self) -> tuple[bool, str]:
        try:
            _import_sip()
        except ImportError as e:
            return False, f"{e}"
        if self.backend == "casadi":
            try:
                _import_casadi()
            except ImportError as e:
                return False, f"casadi backend requested but casadi import failed: {e}"
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0915
        from tests.comparison.problem_spec import effective_solver_tol

        # Stored on self so the inner ``_solve_jax_per_stage`` /
        # ``_solve_casadi`` dispatch targets can read it.
        self._effective_tol = effective_solver_tol(problem, self.tol)
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name,
                problem.name,
                f"unavailable: {reason}",
            )

        # NOTE: sip_python is a generic NLP solver — unlike CSQP /
        # Aligator / acados / fatrop it has no native OCP layout and
        # doesn't care whether a component of ``z`` is stage-local or
        # cross-stage, so theta_dim > 0 is handled by extending ``z``
        # with the Theta block (see ``_build_jax_nlp`` /
        # ``_build_casadi_nlp``).

        # Backend dispatch. The CasADi backend reuses all of the SIP
        # solver wiring (Settings / QDLDLSettings / ProblemDimensions /
        # solver.solve / Variables) but builds the model callback from
        # compiled CasADi Functions instead of jit'd JAX functions; see
        # ``_solve_casadi`` for the alternate model-construction path.
        if self.backend == "casadi":
            if "casadi_builder" not in problem.metadata:
                return make_failure_result(
                    self.name,
                    problem.name,
                    "backend='casadi' requires casadi_builder in problem.metadata "
                    f"(problem={problem.name!r} doesn't ship one — re-run with backend='jax')",
                )
            return self._solve_casadi(problem)

        return self._solve_jax_per_stage(problem)

    def _solve_jax_per_stage(
        self,
        problem: ProblemSpec,
    ) -> SolverResult:  # noqa: PLR0915, PLR0912
        """JAX backend with per-stage Jacobian / Hessian assembly.

        Replaces the original ``_solve`` body that did global
        ``jax.jacrev(c)`` / ``jax.jacrev(g)`` and produced dense per-iter
        Jacobians (e.g. 162 MB on quadpendulum_theta). The per-stage
        pattern mirrors what ``sip_mjx`` does — vmap of stage-local
        ``jax.jacrev`` returning compact ``(T, out_dim, n+m+td)`` blocks
        — but adds theta-coupling support and a user-equality vmap path
        so it works on every problem class (analytical + MJX, with or
        without theta and user equalities).

        Hessian callbacks return the symmetrized Lagrangian Hessian blocks.
        SIP's solver-side inertia correction handles any regularization.
        """
        sip = _import_sip()
        jax, jnp = _import_jax()
        from scipy import sparse as sp

        T, n, m = problem.T, problem.n, problem.m
        td = problem.theta_dim
        nx = n * (T + 1)
        nu = m * T
        x_dim = nx + nu + td

        cost_fn = problem.cost
        dyn_fn = problem.dynamics
        eq_fn = problem.equalities
        ineq_fn = problem.inequalities
        has_user_eq = eq_fn is not None and problem.eq_dim > 0
        has_ineq = ineq_fn is not None and problem.ineq_dim > 0
        eq_dim_user = problem.eq_dim if has_user_eq else 0
        ineq_dim = problem.ineq_dim if has_ineq else 0

        # Equality stack: init(n) + dyn(T*n) + user_eq((T+1)*eq_dim_user).
        c_full_dim = n + T * n + (T + 1) * eq_dim_user
        g_full_dim = (T + 1) * ineq_dim

        x0_const = jnp.asarray(problem.x0)

        def _slice_xut(z):
            X = z[:nx].reshape(T + 1, n)
            U = z[nx : nx + nu].reshape(T, m)
            theta = z[nx + nu : nx + nu + td] if td > 0 else jnp.zeros(0, dtype=z.dtype)
            return X, U, theta

        # ===== Per-stage JAX primitives =====
        @jax.jit
        def stage_cost(x, u, theta, t):
            return cost_fn(x, u, theta, t)

        @jax.jit
        def stage_dyn(x, u, theta, t):
            return dyn_fn(x, u, theta, t)

        @jax.jit
        def terminal_cost(x_T, theta):
            u_zero = jnp.zeros(m, dtype=x_T.dtype)
            return cost_fn(x_T, u_zero, theta, jnp.int32(T))

        if has_user_eq:

            @jax.jit
            def stage_user_eq(x, u, theta, t):
                return eq_fn(x, u, theta, t)

            @jax.jit
            def terminal_user_eq(x_T, theta):
                u_zero = jnp.zeros(m, dtype=x_T.dtype)
                return eq_fn(x_T, u_zero, theta, jnp.int32(T))

        if has_ineq:

            @jax.jit
            def stage_ineq(x, u, theta, t):
                return ineq_fn(x, u, theta, t)

            @jax.jit
            def terminal_ineq(x_T, theta):
                u_zero = jnp.zeros(m, dtype=x_T.dtype)
                return ineq_fn(x_T, u_zero, theta, jnp.int32(T))

        # ===== Flat-NLP scalar / vector functions =====
        @jax.jit
        def f_fn(z):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)
            inner = jax.vmap(stage_cost, in_axes=(0, 0, None, 0))(
                X[:T],
                U,
                theta,
                ts,
            )
            return jnp.sum(inner) + terminal_cost(X[T], theta)

        @jax.jit
        def grad_f_fn(z):
            return jax.grad(f_fn)(z)

        @jax.jit
        def c_fn(z):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)
            init_defect = X[0] - x0_const
            dyn_next = jax.vmap(stage_dyn, in_axes=(0, 0, None, 0))(
                X[:T],
                U,
                theta,
                ts,
            )
            # SIGN convention: c() returns X[t+1] - dyn(...) (opposite from
            # evaluate_problem's dyn(...) - X[t+1]); sign-flipped in the
            # multiplier extraction post-solve.
            dyn_defects = (X[1:] - dyn_next).reshape(-1)
            pieces = [init_defect, dyn_defects]
            if has_user_eq:
                inner_eq = jax.vmap(stage_user_eq, in_axes=(0, 0, None, 0))(
                    X[:T],
                    U,
                    theta,
                    ts,
                )
                term_eq = terminal_user_eq(X[T], theta)
                pieces.append(
                    jnp.concatenate(
                        [inner_eq.reshape(-1), term_eq.reshape(-1)],
                    )
                )
            return jnp.concatenate(pieces)

        if has_ineq:

            @jax.jit
            def g_fn(z):
                X, U, theta = _slice_xut(z)
                ts = jnp.arange(T)
                inner_ineq = jax.vmap(stage_ineq, in_axes=(0, 0, None, 0))(
                    X[:T],
                    U,
                    theta,
                    ts,
                )
                term_ineq = terminal_ineq(X[T], theta)
                return jnp.concatenate(
                    [inner_ineq.reshape(-1), term_ineq.reshape(-1)],
                )

        y_dim = c_full_dim
        s_dim = g_full_dim if has_ineq else 0

        # ===== Per-stage Jacobian builders =====
        # All produce per-stage blocks of shape (..., out_dim, n+m+td) for
        # inner stages and (out_dim, n+td) for terminal stage.

        def _stage_input(x, u, theta):
            # Concatenate (x, u, theta) for the per-stage Jacobian/Hessian
            # input. theta may be empty; concatenation handles that.
            return jnp.concatenate([x, u, theta])

        def _terminal_input(x_T, theta):
            return jnp.concatenate([x_T, theta])

        @jax.jit
        def dyn_jac_blocks(z):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)

            def jac_one(x_in, u_in, t_in):
                def _f(xut):
                    xx = xut[:n]
                    uu = xut[n : n + m]
                    thth = (
                        xut[n + m : n + m + td]
                        if td > 0
                        else jnp.zeros(0, dtype=xut.dtype)
                    )
                    return stage_dyn(xx, uu, thth, t_in)

                return jax.jacrev(_f)(_stage_input(x_in, u_in, theta))

            return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

        if has_user_eq:

            @jax.jit
            def user_eq_jac_inner_blocks(z):
                X, U, theta = _slice_xut(z)
                ts = jnp.arange(T)

                def jac_one(x_in, u_in, t_in):
                    def _f(xut):
                        xx = xut[:n]
                        uu = xut[n : n + m]
                        thth = (
                            xut[n + m : n + m + td]
                            if td > 0
                            else jnp.zeros(0, dtype=xut.dtype)
                        )
                        return stage_user_eq(xx, uu, thth, t_in)

                    return jax.jacrev(_f)(_stage_input(x_in, u_in, theta))

                return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

            @jax.jit
            def user_eq_jac_terminal(z):
                X, _U, theta = _slice_xut(z)

                def _f(xt):
                    xx = xt[:n]
                    thth = xt[n : n + td] if td > 0 else jnp.zeros(0, dtype=xt.dtype)
                    return terminal_user_eq(xx, thth)

                return jax.jacrev(_f)(_terminal_input(X[T], theta))

        if has_ineq:

            @jax.jit
            def ineq_jac_inner_blocks(z):
                X, U, theta = _slice_xut(z)
                ts = jnp.arange(T)

                def jac_one(x_in, u_in, t_in):
                    def _f(xut):
                        xx = xut[:n]
                        uu = xut[n : n + m]
                        thth = (
                            xut[n + m : n + m + td]
                            if td > 0
                            else jnp.zeros(0, dtype=xut.dtype)
                        )
                        return stage_ineq(xx, uu, thth, t_in)

                    return jax.jacrev(_f)(_stage_input(x_in, u_in, theta))

                return jax.vmap(jac_one, in_axes=(0, 0, 0))(X[:T], U, ts)

            @jax.jit
            def ineq_jac_terminal(z):
                X, _U, theta = _slice_xut(z)

                def _f(xt):
                    xx = xt[:n]
                    thth = xt[n : n + td] if td > 0 else jnp.zeros(0, dtype=xt.dtype)
                    return terminal_ineq(xx, thth)

                return jax.jacrev(_f)(_terminal_input(X[T], theta))

        # ===== Per-stage Hessian builders =====
        def _symmetrize(H):
            return 0.5 * (H + H.T)

        @jax.jit
        def inner_hess_blocks(z, y, zd):
            X, U, theta = _slice_xut(z)
            ts = jnp.arange(T)
            # Multiplier slices for the Lagrangian Hessian.
            Y_dyn = y[n : n + T * n].reshape(T, n)
            if has_user_eq:
                Y_eq_inner = y[n + T * n : n + T * n + T * eq_dim_user].reshape(
                    T, eq_dim_user
                )
            else:
                Y_eq_inner = jnp.zeros((T, 0), dtype=z.dtype)
            if has_ineq:
                Z_inner = zd[: T * ineq_dim].reshape(T, ineq_dim)
            else:
                Z_inner = jnp.zeros((T, 0), dtype=z.dtype)

            def stage_hess(x, u, t, y_dyn_t, y_eq_t, z_ineq_t):
                def L_xut(xut):
                    xx = xut[:n]
                    uu = xut[n : n + m]
                    thth = (
                        xut[n + m : n + m + td]
                        if td > 0
                        else jnp.zeros(0, dtype=xut.dtype)
                    )
                    # Lagrangian = cost - y_dyn·dyn (sign matches c's
                    # X[t+1] - dyn convention) + y_eq·eq + z_ineq·ineq.
                    terms = stage_cost(xx, uu, thth, t) - jnp.dot(
                        y_dyn_t,
                        stage_dyn(xx, uu, thth, t),
                    )
                    if has_user_eq:
                        terms = terms + jnp.dot(
                            y_eq_t,
                            stage_user_eq(xx, uu, thth, t),
                        )
                    if has_ineq:
                        terms = terms + jnp.dot(
                            z_ineq_t,
                            stage_ineq(xx, uu, thth, t),
                        )
                    return terms

                return jax.hessian(L_xut)(_stage_input(x, u, theta))

            blocks = jax.vmap(stage_hess, in_axes=(0, 0, 0, 0, 0, 0))(
                X[:T],
                U,
                ts,
                Y_dyn,
                Y_eq_inner,
                Z_inner,
            )
            return jax.vmap(_symmetrize)(blocks)

        @jax.jit
        def terminal_hess_block(z, y, zd):
            X, _U, theta = _slice_xut(z)
            if has_user_eq:
                y_eq_T = y[n + T * n + T * eq_dim_user :].reshape(eq_dim_user)
            else:
                y_eq_T = jnp.zeros(0, dtype=z.dtype)
            if has_ineq:
                z_ineq_T = zd[T * ineq_dim :].reshape(ineq_dim)
            else:
                z_ineq_T = jnp.zeros(0, dtype=z.dtype)

            def L_xt(xt):
                xx = xt[:n]
                thth = xt[n : n + td] if td > 0 else jnp.zeros(0, dtype=xt.dtype)
                terms = terminal_cost(xx, thth)
                if has_user_eq:
                    terms = terms + jnp.dot(
                        y_eq_T,
                        terminal_user_eq(xx, thth),
                    )
                if has_ineq:
                    terms = terms + jnp.dot(
                        z_ineq_T,
                        terminal_ineq(xx, thth),
                    )
                return terms

            H = jax.hessian(L_xt)(_terminal_input(X[T], theta))
            return _symmetrize(H)

        # ===== Sparsity templates (hand-assembled) =====
        # Column layout of z = [X (n*(T+1)), U (m*T), Theta (td)].
        def _x_col(t, i):
            return t * n + i

        def _u_col(t, i):
            return nx + t * m + i

        def _theta_col(i):
            return nx + nu + i

        # --- c-Jacobian rows: init(n) + dyn(T*n) + user_eq((T+1)*eq_dim_user) ---
        c_rows: list[int] = []
        c_cols: list[int] = []
        # init: identity on x_0.
        for i in range(n):
            c_rows.append(i)
            c_cols.append(_x_col(0, i))
        # dyn defects: per stage t, identity on x_{t+1}, then dense block
        # on (x_t, u_t, theta) with sign -1 (we don't track signs in the
        # sparsity pattern, just nonzero positions).
        for t in range(T):
            row_off = n + t * n
            for i in range(n):
                c_rows.append(row_off + i)
                c_cols.append(_x_col(t + 1, i))
            for i in range(n):
                for j in range(n):
                    c_rows.append(row_off + i)
                    c_cols.append(_x_col(t, j))
            for i in range(n):
                for j in range(m):
                    c_rows.append(row_off + i)
                    c_cols.append(_u_col(t, j))
            if td > 0:
                for i in range(n):
                    for j in range(td):
                        c_rows.append(row_off + i)
                        c_cols.append(_theta_col(j))
        # user equalities: per stage t in 0..T-1 (inner) — dense block on
        # (x_t, u_t, theta); terminal — dense block on (x_T, theta).
        # Block-major ordering (all i's, x-cols block; then all i's, u-cols
        # block; then theta-cols) so the per-iter writeback can use
        # ueq_J[t, :, :n].reshape(-1) etc. directly.
        if has_user_eq:
            ueq_off_inner = n + T * n
            for t in range(T):
                row_off = ueq_off_inner + t * eq_dim_user
                # x-cols block
                for i in range(eq_dim_user):
                    for j in range(n):
                        c_rows.append(row_off + i)
                        c_cols.append(_x_col(t, j))
                # u-cols block
                for i in range(eq_dim_user):
                    for j in range(m):
                        c_rows.append(row_off + i)
                        c_cols.append(_u_col(t, j))
                # theta-cols block
                if td > 0:
                    for i in range(eq_dim_user):
                        for j in range(td):
                            c_rows.append(row_off + i)
                            c_cols.append(_theta_col(j))
            row_off = ueq_off_inner + T * eq_dim_user
            # Terminal: x-cols block, then theta-cols block.
            for i in range(eq_dim_user):
                for j in range(n):
                    c_rows.append(row_off + i)
                    c_cols.append(_x_col(T, j))
            if td > 0:
                for i in range(eq_dim_user):
                    for j in range(td):
                        c_rows.append(row_off + i)
                        c_cols.append(_theta_col(j))
        c_rows_arr = np.asarray(c_rows, dtype=np.int32)
        c_cols_arr = np.asarray(c_cols, dtype=np.int32)
        jac_c_template = sp.csr_matrix(
            (
                np.ones(c_rows_arr.shape[0], dtype=np.float64),
                (c_rows_arr, c_cols_arr),
            ),
            shape=(y_dim, x_dim),
        )

        # --- g-Jacobian rows: (T+1) * ineq_dim ---
        # Block-major ordering (all i's, x-cols block; then all i's, u-cols
        # block; then theta-cols) so the per-iter writeback can use
        # ineq_J[t, :, :n].reshape(-1) etc. directly.
        if has_ineq:
            g_rows: list[int] = []
            g_cols: list[int] = []
            for t in range(T):
                row_off = t * ineq_dim
                # x-cols block
                for i in range(ineq_dim):
                    for j in range(n):
                        g_rows.append(row_off + i)
                        g_cols.append(_x_col(t, j))
                # u-cols block
                for i in range(ineq_dim):
                    for j in range(m):
                        g_rows.append(row_off + i)
                        g_cols.append(_u_col(t, j))
                # theta-cols block
                if td > 0:
                    for i in range(ineq_dim):
                        for j in range(td):
                            g_rows.append(row_off + i)
                            g_cols.append(_theta_col(j))
            row_off = T * ineq_dim
            # Terminal: x-cols block, then theta-cols block.
            for i in range(ineq_dim):
                for j in range(n):
                    g_rows.append(row_off + i)
                    g_cols.append(_x_col(T, j))
            if td > 0:
                for i in range(ineq_dim):
                    for j in range(td):
                        g_rows.append(row_off + i)
                        g_cols.append(_theta_col(j))
            g_rows_arr = np.asarray(g_rows, dtype=np.int32)
            g_cols_arr = np.asarray(g_cols, dtype=np.int32)
            jac_g_template = sp.csr_matrix(
                (
                    np.ones(g_rows_arr.shape[0], dtype=np.float64),
                    (g_rows_arr, g_cols_arr),
                ),
                shape=(s_dim, x_dim),
            )
        else:
            g_rows_arr = np.zeros(0, dtype=np.int32)
            g_cols_arr = np.zeros(0, dtype=np.int32)
            jac_g_template = sp.csr_matrix((s_dim, x_dim))

        # --- Lagrangian Hessian upper triangle (block-diagonal in stages,
        # theta-coupling shared across stages) ---
        h_rows: list[int] = []
        h_cols: list[int] = []
        h_writebacks: list[dict] = []  # bookkeeping for theta scatter-add
        for t in range(T):
            x_idx = [_x_col(t, i) for i in range(n)]
            u_idx = [_u_col(t, i) for i in range(m)]
            th_idx = [_theta_col(i) for i in range(td)]
            block_idx = x_idx + u_idx + th_idx  # block layout = (x, u, theta)
            # Track the natural-order index range for this stage's block.
            start_idx = len(h_rows)
            for ii, ri in enumerate(block_idx):
                for jj, cj in enumerate(block_idx):
                    if ri <= cj:
                        h_rows.append(ri)
                        h_cols.append(cj)
            end_idx = len(h_rows)
            h_writebacks.append(
                {
                    "kind": "inner",
                    "t": t,
                    "start": start_idx,
                    "end": end_idx,
                    "block_size": n + m + td,
                }
            )
        x_idx_T = [_x_col(T, i) for i in range(n)]
        th_idx = [_theta_col(i) for i in range(td)]
        block_idx_T = x_idx_T + th_idx
        start_idx = len(h_rows)
        for ii, ri in enumerate(block_idx_T):
            for jj, cj in enumerate(block_idx_T):
                if ri <= cj:
                    h_rows.append(ri)
                    h_cols.append(cj)
        end_idx = len(h_rows)
        h_writebacks.append(
            {
                "kind": "terminal",
                "start": start_idx,
                "end": end_idx,
                "block_size": n + td,
            }
        )
        h_rows_arr = np.asarray(h_rows, dtype=np.int32)
        h_cols_arr = np.asarray(h_cols, dtype=np.int32)
        upp_hess_template = sp.csc_matrix(
            (np.ones(h_rows_arr.shape[0], dtype=np.float64), (h_rows_arr, h_cols_arr)),
            shape=(x_dim, x_dim),
        )

        # Detect duplicate (row, col) entries — this happens when theta
        # is present, because every stage's block has the theta-theta
        # entries in the same global positions. scipy's COO->CSC sums
        # duplicates by default, but our natural-order writeback needs
        # to account for it. We compute per-natural-index targets in
        # the canonical CSC data array; entries that collide get the
        # same target and we use np.add.at for the writeback.
        # For the non-theta case there are no collisions and the simple
        # permutation suffices.
        def _csr_perm_with_duplicates(
            rows: np.ndarray, cols: np.ndarray, template
        ) -> np.ndarray:
            """Map natural-order indices to canonical CSR/CSC .data slots.

            ``template.data[target[k]] += values[k]`` reconstructs the
            sparse matrix value-correctly even when (rows[k], cols[k])
            has duplicates across k.
            """
            # CSR sorts by (row, col within row). For each natural-order
            # entry, find which (sorted) data slot it belongs to.
            # Use scipy's own coo->csr machinery to get the sorted order.
            # The template was built by COO->CSR/CSC, which sums dupes.
            # We re-derive the canonical (row, col) per data slot from
            # the template, then for each (rows[k], cols[k]) find that
            # slot via a hash lookup.
            from scipy.sparse import csr_matrix, csc_matrix

            if isinstance(template, csr_matrix):
                ax_major, ax_minor = template.indptr, template.indices
                n_major = template.shape[0]

                def _slot(r, c):
                    lo, hi = ax_major[r], ax_major[r + 1]
                    # binary search for c in ax_minor[lo:hi]
                    sub = ax_minor[lo:hi]
                    pos = np.searchsorted(sub, c)
                    return lo + pos
            elif isinstance(template, csc_matrix):
                ax_major, ax_minor = template.indptr, template.indices

                def _slot(r, c):
                    lo, hi = ax_major[c], ax_major[c + 1]
                    sub = ax_minor[lo:hi]
                    pos = np.searchsorted(sub, r)
                    return lo + pos
            else:
                raise TypeError(f"unsupported sparse type {type(template).__name__}")
            target = np.empty(rows.shape[0], dtype=np.int64)
            for k in range(rows.shape[0]):
                target[k] = _slot(int(rows[k]), int(cols[k]))
            return target

        c_target = _csr_perm_with_duplicates(
            c_rows_arr,
            c_cols_arr,
            jac_c_template,
        )
        h_target = _csr_perm_with_duplicates(h_rows_arr, h_cols_arr, upp_hess_template)
        if has_ineq:
            g_target = _csr_perm_with_duplicates(
                g_rows_arr,
                g_cols_arr,
                jac_g_template,
            )
        else:
            g_target = np.zeros(0, dtype=np.int64)

        # Per-stage writeback index lookups (for the Hessian, we need to
        # know which .data slots each stage's natural-order block maps to).
        # For c-Jacobian and g-Jacobian, the targets are already non-
        # duplicated (each entry has a unique (row, col)) except in the
        # presence of theta-coupling, where dyn / user_eq / ineq rows have
        # theta cols shared across stages — but those are SEPARATE
        # (row, col) pairs since the rows differ, so no duplicates.
        # For the Hessian, theta-theta entries DO have the same (row, col)
        # across stages, so we use np.add.at to accumulate.

        # Reusable buffers — same scipy sparse object reused per call.
        jac_c_buf = jac_c_template.copy()
        jac_g_buf = jac_g_template.copy()
        upp_hess_buf = upp_hess_template.copy()

        # ===== ProblemDimensions / QDLDLSettings =====
        pd = sip.ProblemDimensions()
        pd.x_dim = x_dim
        pd.s_dim = s_dim
        pd.y_dim = y_dim
        pd.upper_hessian_lagrangian_nnz = upp_hess_template.nnz
        pd.jacobian_c_nnz = jac_c_template.nnz
        pd.jacobian_g_nnz = jac_g_template.nnz
        pd.is_jacobian_c_transposed = True
        pd.is_jacobian_g_transposed = True

        qs = sip.QDLDLSettings()
        qs.permute_kkt_system = True
        from tests.comparison.sip_kkt_perm import compute_kkt_perm_inv_and_nnzs

        _perm_result = compute_kkt_perm_inv_and_nnzs(
            upp_hess_template,
            jac_c_template,
            jac_g_template,
        )
        qs.kkt_pinv = _perm_result.perm_inv
        pd.kkt_nnz = _perm_result.kkt_nnz
        pd.kkt_L_nnz = _perm_result.L_nnz

        # ===== Settings =====
        # Layering: constructor defaults <- problem.metadata['sip_settings']
        # <- problem.metadata['sip_<backend>_settings'] <- CLI
        # self.sip_extra_settings. The backend-specific key lets a
        # problem ship per-backend tuning where the JAX and CasADi
        # numerics paths need different schedules.
        problem_overrides = dict(problem.metadata.get("sip_settings", {}))
        backend_key = f"sip_{self.backend}_settings"
        problem_overrides = _merge_setting_dicts(
            problem_overrides,
            problem.metadata.get(backend_key, {}),
        )
        time_limit_s = _custom_time_limit_s(
            {"time_limit_s": self.timeout_s},
            problem_overrides,
            self.sip_extra_settings,
        )
        ss = sip.Settings()
        ss.max_iterations = int(self.max_iter)
        ss.termination.max_dual_residual = float(self._effective_tol)
        ss.termination.max_constraint_violation = float(self._effective_tol)
        ss.termination.max_complementarity_gap = float(self._effective_tol)
        ss.penalty.penalty_parameter_increase_factor = (
            self.penalty_parameter_increase_factor
        )
        ss.barrier.mu_update_factor = self.mu_update_factor
        ss.barrier.initial_mu = self.initial_mu
        ss.penalty.initial_penalty_parameter = self.initial_penalty_parameter
        ss.logging.print_logs = self.print_logs
        if not self.print_logs:
            ss.logging.print_line_search_logs = False
            ss.logging.print_search_direction_logs = False
            ss.logging.print_derivative_check_logs = False
        ss.assert_checks_pass = False
        _apply_sip_settings(ss, problem_overrides)
        _apply_sip_settings(ss, self.sip_extra_settings)

        # ===== Warm start =====
        z_init = _initial_decision_vector(problem)

        # ===== Per-iter recording =====
        iter_xut: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # ===== Model callback =====
        def mc(mci):
            mco = sip.ModelCallbackOutput()
            z_jax = jnp.asarray(mci.x, dtype=jnp.float64)
            z_np_iter = np.asarray(mci.x, dtype=np.float64).copy()
            Xi, Ui, Ti = _slice_iterate(z_np_iter, problem)

            # Scalar / vector primitives.
            mco.f = float(np.asarray(f_fn(z_jax)))
            mco.c = np.asarray(c_fn(z_jax), dtype=np.float64).copy()
            if has_ineq:
                mco.g = np.asarray(g_fn(z_jax), dtype=np.float64).copy()
            else:
                mco.g = np.zeros(0, dtype=np.float64)
            mco.gradient_f = np.asarray(
                grad_f_fn(z_jax),
                dtype=np.float64,
            ).copy()

            # ----- c-Jacobian (init + dyn + user_eq) -----
            dyn_J = np.asarray(dyn_jac_blocks(z_jax), dtype=np.float64)
            c_vals = np.empty(c_rows_arr.shape[0], dtype=np.float64)
            idx = 0
            # init: identity (n entries).
            c_vals[idx : idx + n] = 1.0
            idx += n
            # dyn defects: per stage, identity on x_{t+1} + (-dyn_J[t]).
            for t in range(T):
                c_vals[idx : idx + n] = 1.0
                idx += n  # x_{t+1} identity
                # (-) dyn / dx (n*n entries)
                c_vals[idx : idx + n * n] = -dyn_J[t, :, :n].reshape(-1)
                idx += n * n
                # (-) dyn / du (n*m entries)
                c_vals[idx : idx + n * m] = -dyn_J[t, :, n : n + m].reshape(-1)
                idx += n * m
                if td > 0:
                    c_vals[idx : idx + n * td] = -dyn_J[
                        t, :, n + m : n + m + td
                    ].reshape(-1)
                    idx += n * td
            # user equalities.
            if has_user_eq:
                ueq_inner_J = np.asarray(
                    user_eq_jac_inner_blocks(z_jax),
                    dtype=np.float64,
                )
                ueq_term_J = np.asarray(
                    user_eq_jac_terminal(z_jax),
                    dtype=np.float64,
                )
                for t in range(T):
                    c_vals[idx : idx + eq_dim_user * n] = ueq_inner_J[t, :, :n].reshape(
                        -1
                    )
                    idx += eq_dim_user * n
                    c_vals[idx : idx + eq_dim_user * m] = ueq_inner_J[
                        t, :, n : n + m
                    ].reshape(-1)
                    idx += eq_dim_user * m
                    if td > 0:
                        c_vals[idx : idx + eq_dim_user * td] = ueq_inner_J[
                            t, :, n + m : n + m + td
                        ].reshape(-1)
                        idx += eq_dim_user * td
                # terminal user eq.
                c_vals[idx : idx + eq_dim_user * n] = ueq_term_J[:, :n].reshape(-1)
                idx += eq_dim_user * n
                if td > 0:
                    c_vals[idx : idx + eq_dim_user * td] = ueq_term_J[
                        :, n : n + td
                    ].reshape(-1)
                    idx += eq_dim_user * td
            jac_c_buf.data[:] = 0.0
            np.add.at(jac_c_buf.data, c_target, c_vals)
            mco.jacobian_c = jac_c_buf

            # ----- g-Jacobian (per-stage ineq + terminal) -----
            if has_ineq:
                ineq_J = np.asarray(
                    ineq_jac_inner_blocks(z_jax),
                    dtype=np.float64,
                )
                ineq_term_J = np.asarray(
                    ineq_jac_terminal(z_jax),
                    dtype=np.float64,
                )
                g_vals = np.empty(g_rows_arr.shape[0], dtype=np.float64)
                idx = 0
                for t in range(T):
                    g_vals[idx : idx + ineq_dim * n] = ineq_J[t, :, :n].reshape(-1)
                    idx += ineq_dim * n
                    g_vals[idx : idx + ineq_dim * m] = ineq_J[t, :, n : n + m].reshape(
                        -1
                    )
                    idx += ineq_dim * m
                    if td > 0:
                        g_vals[idx : idx + ineq_dim * td] = ineq_J[
                            t, :, n + m : n + m + td
                        ].reshape(-1)
                        idx += ineq_dim * td
                g_vals[idx : idx + ineq_dim * n] = ineq_term_J[:, :n].reshape(-1)
                idx += ineq_dim * n
                if td > 0:
                    g_vals[idx : idx + ineq_dim * td] = ineq_term_J[
                        :, n : n + td
                    ].reshape(-1)
                    idx += ineq_dim * td
                jac_g_buf.data[:] = 0.0
                np.add.at(jac_g_buf.data, g_target, g_vals)
                mco.jacobian_g = jac_g_buf
            else:
                mco.jacobian_g = jac_g_buf

            # ----- Lagrangian Hessian (per-stage PSD-projected blocks) -----
            y_full = np.asarray(mci.y, dtype=np.float64)
            y_jax = jnp.asarray(y_full, dtype=jnp.float64)
            if has_ineq:
                zd_full = np.asarray(mci.z, dtype=np.float64)
                zd_jax = jnp.asarray(zd_full, dtype=jnp.float64)
            else:
                zd_full = np.zeros(0, dtype=np.float64)
                zd_jax = jnp.zeros(0, dtype=jnp.float64)
            iter_xut.append((Xi, Ui, Ti))
            inner_blocks = np.asarray(
                inner_hess_blocks(z_jax, y_jax, zd_jax),
                dtype=np.float64,
            )
            terminal_block = np.asarray(
                terminal_hess_block(z_jax, y_jax, zd_jax),
                dtype=np.float64,
            )
            h_vals = np.empty(h_rows_arr.shape[0], dtype=np.float64)
            for wb in h_writebacks:
                if wb["kind"] == "inner":
                    blk = inner_blocks[wb["t"]]
                else:
                    blk = terminal_block
                bsz = wb["block_size"]
                vidx = wb["start"]
                for ii in range(bsz):
                    for jj in range(bsz):
                        ri = (
                            blk_row_for_inner(ii, n, m, T, td, wb)
                            if wb["kind"] == "inner"
                            else blk_row_for_terminal(ii, n, T, td)
                        )
                        cj = (
                            blk_row_for_inner(jj, n, m, T, td, wb)
                            if wb["kind"] == "inner"
                            else blk_row_for_terminal(jj, n, T, td)
                        )
                        if ri <= cj:
                            h_vals[vidx] = blk[ii, jj]
                            vidx += 1
            upp_hess_buf.data[:] = 0.0
            np.add.at(upp_hess_buf.data, h_target, h_vals)
            mco.upper_hessian_lagrangian = upp_hess_buf

            return mco

        # Helper closures used inside mc for Hessian writeback row/col mapping.
        # Defined OUTSIDE mc to avoid per-call closure overhead.
        def blk_row_for_inner(
            local_idx: int, n: int, m: int, T_: int, td: int, wb: dict
        ) -> int:
            t = wb["t"]
            if local_idx < n:
                return t * n + local_idx
            if local_idx < n + m:
                return nx + t * m + (local_idx - n)
            return nx + nu + (local_idx - n - m)

        def blk_row_for_terminal(local_idx: int, n: int, T_: int, td: int) -> int:
            if local_idx < n:
                return T_ * n + local_idx
            return nx + nu + (local_idx - n)

        # ===== Construct solver =====
        solver = sip.Solver(ss, qs, pd, mc, time_limit_s)

        vars_in = sip.Variables(pd)
        vars_in.x[:] = z_init
        if has_ineq:
            g_init = np.asarray(
                g_fn(jnp.asarray(z_init, dtype=jnp.float64)),
                dtype=np.float64,
            )
        else:
            g_init = np.zeros(0, dtype=np.float64)
        s_init, zd_init = _initialize_slacks_and_duals(
            g_init,
            ss.barrier.initial_mu,
        )
        vars_in.s[:] = s_init
        vars_in.z[:] = zd_init
        vars_in.y[:] = 0.0
        warm_start = problem.warm_start or {}
        y_full_init = _initialize_eq_multipliers_from_warm_start(
            warm_start,
            n=n,
            T=T,
            eq_dim=eq_dim_user,
        )
        if np.any(y_full_init):
            vars_in.y[:] = y_full_init
        if has_ineq:
            s_warm = _warm_start_vector(warm_start, "S", g_full_dim)
            z_dual_warm = _warm_start_vector(warm_start, "Z", g_full_dim)
            warm_floor = max(
                float(ss.barrier.initial_mu),
                1e-8,
            )
            s_warm = _strictly_positive_warm_start(
                s_warm,
                floor=warm_floor,
            )
            z_dual_warm = _strictly_positive_warm_start(
                z_dual_warm,
                floor=warm_floor,
            )
            if s_warm is not None:
                vars_in.s[:] = s_warm
            if z_dual_warm is not None:
                vars_in.z[:] = z_dual_warm

        # ===== Warm up JIT caches on z_init =====
        try:
            z_flat_warm = jnp.asarray(z_init, dtype=jnp.float64)
            y_warm = jnp.asarray(y_full_init, dtype=jnp.float64)
            if has_ineq:
                zd_warm = jnp.asarray(vars_in.z, dtype=jnp.float64)
            else:
                zd_warm = jnp.zeros(0, dtype=jnp.float64)
            jax.block_until_ready(f_fn(z_flat_warm))
            jax.block_until_ready(grad_f_fn(z_flat_warm))
            jax.block_until_ready(c_fn(z_flat_warm))
            jax.block_until_ready(dyn_jac_blocks(z_flat_warm))
            if has_user_eq:
                jax.block_until_ready(user_eq_jac_inner_blocks(z_flat_warm))
                jax.block_until_ready(user_eq_jac_terminal(z_flat_warm))
            if has_ineq:
                jax.block_until_ready(g_fn(z_flat_warm))
                jax.block_until_ready(ineq_jac_inner_blocks(z_flat_warm))
                jax.block_until_ready(ineq_jac_terminal(z_flat_warm))
            jax.block_until_ready(inner_hess_blocks(z_flat_warm, y_warm, zd_warm))
            jax.block_until_ready(terminal_hess_block(z_flat_warm, y_warm, zd_warm))
        except Exception:  # noqa: BLE001
            pass

        # ===== Solve =====
        start = timer()
        try:
            output = solver.solve(vars_in)
            err = ""
        except Exception as e:  # noqa: BLE001
            output = None
            err = f"{type(e).__name__}: {e}"
        solve_time_ms = 1e3 * (timer() - start)

        z_val = np.asarray(vars_in.x, dtype=np.float64).copy()
        X, U, Theta = _slice_iterate(z_val, problem)

        # ===== Multiplier extraction =====
        # Sign-flip dyn rows because c() encodes X[t+1] - dyn(...) while
        # evaluate_problem measures dyn(...) - X[t+1].
        try:
            multipliers_eq_full = np.asarray(vars_in.y, dtype=np.float64).reshape(-1)
            multipliers_eq_full[n : n + T * n] = -multipliers_eq_full[n : n + T * n]
        except Exception:  # noqa: BLE001
            multipliers_eq_full = None
        try:
            if has_ineq:
                multipliers_ineq_full = np.asarray(
                    vars_in.z,
                    dtype=np.float64,
                ).reshape(-1)
                slacks_full = np.asarray(vars_in.s, dtype=np.float64).reshape(-1)
            else:
                multipliers_ineq_full = None
                slacks_full = None
        except Exception:  # noqa: BLE001
            multipliers_ineq_full = None
            slacks_full = None
        if output is None:
            warm_start_out = _make_warm_start_out(
                multipliers_eq_full=multipliers_eq_full,
                multipliers_ineq_full=multipliers_ineq_full,
                slacks_full=slacks_full,
                n=n,
                T=T,
                eq_dim=eq_dim_user,
            )
            return pack_solver_result(
                solver_name=self.name,
                problem_name=problem.name,
                problem=problem,
                X=X,
                U=U,
                Theta=Theta,
                iterations=0,
                solve_time_ms=solve_time_ms,
                success=False,
                notes=err,
                multipliers_eq=multipliers_eq_full,
                multipliers_ineq=multipliers_ineq_full,
                warm_start_out=warm_start_out,
                iterates_xut=iter_xut or None,
            )

        status = output.exit_status
        success = status == sip.Status.SOLVED
        notes = _sip_status_notes(output)
        warm_start_out = _make_warm_start_out(
            multipliers_eq_full=multipliers_eq_full,
            multipliers_ineq_full=multipliers_ineq_full,
            slacks_full=slacks_full,
            n=n,
            T=T,
            eq_dim=eq_dim_user,
        )
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X,
            U=U,
            Theta=Theta,
            iterations=int(output.num_iterations),
            solve_time_ms=solve_time_ms,
            success=success,
            notes=notes,
            multipliers_eq=multipliers_eq_full,
            multipliers_ineq=multipliers_ineq_full,
            warm_start_out=warm_start_out,
            iterates_xut=iter_xut or None,
        )

    def _solve_casadi(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0915
        """CasADi-backend variant of ``solve``.

        Same SIP wiring as the JAX path (Settings / QDLDLSettings /
        ProblemDimensions / Solver / Variables), but the per-iter model
        callback dispatches to compiled CasADi Functions instead of
        jit'd JAX functions. The decision-vector layout is identical
        (``z = vec(X) ++ vec(U) ++ Theta``) so iter counts should match
        the JAX backend to within FP noise on analytical problems.

        Sparsity is taken directly from CasADi's symbolic-graph analysis
        (no mock-eval probe), and the Jacobian / Lagrangian-Hessian
        templates are built from the CasADi sparsity patterns.
        """
        sip = _import_sip()
        ca = _import_casadi()
        from scipy import sparse as sp

        # --- Build cost + constraint callables --------------------------------
        nlp = _build_casadi_nlp(problem)
        f_fn = nlp["f_fn"]
        c_fn = nlp["c_fn"]
        g_fn = nlp["g_fn"]
        grad_f_fn = nlp["grad_f_fn"]
        jac_c_fn = nlp["jac_c_fn"]
        jac_g_fn = nlp["jac_g_fn"]
        hess_lag_fn = nlp["hess_lag_fn"]
        hess_lag_sparsity = nlp["hess_lag_sparsity"]
        x_dim = nlp["x_dim"]
        y_dim = nlp["y_dim"]
        s_dim = nlp["s_dim"]
        jac_c_sparsity = nlp["jac_c_sparsity"]
        jac_g_sparsity = nlp["jac_g_sparsity"]

        # --- Build initial guess ---------------------------------------------
        z_init = _initial_decision_vector(problem)
        assert z_init.size == x_dim, (
            f"CasADi flat NLP got x_dim={x_dim} but z_init.size={z_init.size}; "
            "this indicates a mismatch between ProblemSpec dimensions and the "
            "CasADi builder output."
        )

        # --- Build sparse-matrix templates from CasADi sparsity --------------
        # SIP's convention: jacobians as CSR of shape (out_dim, x_dim) — the
        # C++ side reinterprets CSR row-major data as CSC of the transpose,
        # so ProblemDimensions.is_jacobian_*_transposed = True. CasADi
        # gives us the Jacobian in CSC; we need to permute the per-call
        # nonzero values (CSC order) into the CSR template's .data slots
        # (row-major order). The permutation is computed once via the
        # idx-CSC -> idx-CSR trick: build a CSC whose data is the
        # 0..nnz-1 enumeration of CasADi's nz, convert to CSR, and the
        # resulting .data is exactly the permutation we need.
        def _build_jac_template(sparsity, out_dim: int):
            """Return (csr_template, casadi_nz_to_csr_perm).

            ``csr_template`` is a scipy.sparse.csr_matrix with the right
            structural nnz pattern (zero values).
            ``casadi_nz_to_csr_perm`` is an ``np.ndarray`` of indices such
            that per-call we can do
                csr_template.data[:] = casadi_dm.nonzeros()[perm]
            to fill in the new values without going through dense.
            """
            if sparsity is None or out_dim == 0:
                empty = sp.csr_matrix((out_dim, x_dim), dtype=np.float64)
                return empty, np.zeros(0, dtype=np.intp)
            nnz = sparsity.nnz()
            csc_pat = sp.csc_matrix(
                (
                    np.ones(nnz, dtype=np.float64),
                    np.asarray(sparsity.row(), dtype=np.intp),
                    np.asarray(sparsity.colind(), dtype=np.intp),
                ),
                shape=(out_dim, x_dim),
            )
            csr_template = csc_pat.tocsr()
            # Build the permutation: idx_csc has 0..nnz-1 as its data, in
            # CasADi's CSC order; idx_csr.data is then perm[i] = which
            # CasADi nz index ends up at CSR position i.
            idx_csc = sp.csc_matrix(
                (
                    np.arange(nnz, dtype=np.intp),
                    np.asarray(sparsity.row(), dtype=np.intp),
                    np.asarray(sparsity.colind(), dtype=np.intp),
                ),
                shape=(out_dim, x_dim),
            )
            perm = np.asarray(idx_csc.tocsr().data, dtype=np.intp)
            return csr_template, perm

        jac_c_template, jac_c_perm = _build_jac_template(jac_c_sparsity, y_dim)
        jac_g_template, jac_g_perm = _build_jac_template(jac_g_sparsity, s_dim)

        def _build_upper_hess_template(sparsity):
            col_ind = np.asarray(sparsity.colind(), dtype=np.intp)
            row_ind = np.asarray(sparsity.row(), dtype=np.intp)
            col_by_nz = np.repeat(np.arange(sparsity.size2()), np.diff(col_ind))
            source_by_upper: dict[tuple[int, int], int] = {}
            for src, (row, col) in enumerate(zip(row_ind, col_by_nz)):
                row_i = int(row)
                col_i = int(col)
                upper_key = (
                    (row_i, col_i) if row_i <= col_i else (col_i, row_i)
                )
                if upper_key not in source_by_upper or row_i <= col_i:
                    source_by_upper[upper_key] = src

            for i in range(x_dim):
                source_by_upper.setdefault((i, i), -1)

            upper_entries = sorted(source_by_upper, key=lambda rc: (rc[1], rc[0]))
            rows, cols = zip(*upper_entries)
            source_nz = [source_by_upper[(row, col)] for row, col in zip(rows, cols)]

            template = sp.csc_matrix(
                (
                    np.zeros(len(rows), dtype=np.float64),
                    (np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)),
                ),
                shape=(x_dim, x_dim),
            )
            return template, np.asarray(source_nz, dtype=np.intp)

        upp_hess_template, hess_lag_perm = _build_upper_hess_template(
            hess_lag_sparsity
        )

        # --- Build ProblemDimensions / QDLDLSettings -------------------------
        pd = sip.ProblemDimensions()
        pd.x_dim = x_dim
        pd.s_dim = s_dim
        pd.y_dim = y_dim
        pd.upper_hessian_lagrangian_nnz = upp_hess_template.nnz
        pd.jacobian_c_nnz = jac_c_template.nnz
        pd.jacobian_g_nnz = jac_g_template.nnz
        pd.is_jacobian_c_transposed = True
        pd.is_jacobian_g_transposed = True

        qs = sip.QDLDLSettings()
        qs.permute_kkt_system = True
        # AMD-first KKT permutation (see comment + import in the JAX
        # backend block above; same module).
        from tests.comparison.sip_kkt_perm import compute_kkt_perm_inv_and_nnzs

        _perm_result = compute_kkt_perm_inv_and_nnzs(
            upp_hess_template,
            jac_c_template,
            jac_g_template,
        )
        qs.kkt_pinv = _perm_result.perm_inv
        pd.kkt_nnz = _perm_result.kkt_nnz
        pd.kkt_L_nnz = _perm_result.L_nnz

        # --- Settings --------------------------------------------------------
        problem_overrides = dict(problem.metadata.get("sip_settings", {}))
        time_limit_s = _custom_time_limit_s(
            {"time_limit_s": self.timeout_s},
            problem_overrides,
            self.sip_extra_settings,
        )
        ss = sip.Settings()
        ss.max_iterations = int(self.max_iter)
        ss.termination.max_dual_residual = float(self._effective_tol)
        ss.termination.max_constraint_violation = float(self._effective_tol)
        ss.termination.max_complementarity_gap = float(self._effective_tol)
        ss.penalty.penalty_parameter_increase_factor = (
            self.penalty_parameter_increase_factor
        )
        ss.barrier.mu_update_factor = self.mu_update_factor
        ss.barrier.initial_mu = self.initial_mu
        ss.penalty.initial_penalty_parameter = self.initial_penalty_parameter
        ss.logging.print_logs = self.print_logs
        if not self.print_logs:
            ss.logging.print_line_search_logs = False
            ss.logging.print_search_direction_logs = False
            ss.logging.print_derivative_check_logs = False
        ss.assert_checks_pass = False
        _apply_sip_settings(ss, problem_overrides)
        _apply_sip_settings(ss, self.sip_extra_settings)

        # --- Build the model callback ---------------------------------------
        # Reusable buffers, written in-place each callback. We bind them to
        # the closure so SIP sees the same memory each call.
        jac_c_buf = jac_c_template.copy()
        jac_g_buf = jac_g_template.copy()
        upp_hess_buf = upp_hess_template.copy()

        # Per-iter recorder (CasADi backend variant).
        iter_xut: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def mc(mci):
            mco = sip.ModelCallbackOutput()
            z_np = np.asarray(mci.x, dtype=np.float64)
            # Per-iter snapshot for post-process history extraction.
            Xi, Ui, Ti = _slice_iterate(z_np.copy(), problem)
            iter_xut.append((Xi, Ui, Ti))

            # CasADi Function calls accept numpy arrays directly; returns
            # casadi.DM (sparse) for sparse outputs, dense for dense.
            mco.f = float(f_fn(z_np))
            mco.c = np.asarray(c_fn(z_np), dtype=np.float64).reshape(-1).copy()
            if s_dim > 0:
                mco.g = np.asarray(g_fn(z_np), dtype=np.float64).reshape(-1).copy()
            else:
                mco.g = np.zeros(0, dtype=np.float64)
            mco.gradient_f = (
                np.asarray(grad_f_fn(z_np), dtype=np.float64).reshape(-1).copy()
            )

            # Jacobians: extract values directly from the casadi.DM's
            # nonzeros() (CSC-ordered list, ~40x faster than
            # np.asarray(DM) which densifies). The pre-computed permutation
            # ``jac_*_perm`` maps from CSC nz index -> CSR template .data
            # slot.
            if y_dim > 0:
                jc_nz = np.asarray(jac_c_fn(z_np).nonzeros(), dtype=np.float64)
                jac_c_buf.data[:] = jc_nz[jac_c_perm]
            mco.jacobian_c = jac_c_buf

            if s_dim > 0:
                jg_nz = np.asarray(jac_g_fn(z_np).nonzeros(), dtype=np.float64)
                jac_g_buf.data[:] = jg_nz[jac_g_perm]
            mco.jacobian_g = jac_g_buf

            h_args = [z_np]
            if y_dim > 0:
                h_args.append(np.asarray(mci.y, dtype=np.float64))
            if s_dim > 0:
                h_args.append(np.asarray(mci.z, dtype=np.float64))
            h_nz = np.asarray(hess_lag_fn(*h_args).nonzeros(), dtype=np.float64)
            upp_hess_buf.data[:] = 0.0
            present = hess_lag_perm >= 0
            upp_hess_buf.data[present] = h_nz[hess_lag_perm[present]]
            mco.upper_hessian_lagrangian = upp_hess_buf

            return mco

        # --- Construct the solver --------------------------------------------
        solver = sip.Solver(ss, qs, pd, mc, time_limit_s)

        # --- Initial Variables ----------------------------------------------
        vars_in = sip.Variables(pd)
        vars_in.x[:] = z_init
        if s_dim > 0:
            g_init = np.asarray(g_fn(z_init), dtype=np.float64).reshape(-1)
        else:
            g_init = np.zeros(0, dtype=np.float64)
        s_init, zd_init = _initialize_slacks_and_duals(
            g_init,
            ss.barrier.initial_mu,
        )
        vars_in.s[:] = s_init
        vars_in.z[:] = zd_init
        vars_in.y[:] = 0.0
        warm_start = problem.warm_start or {}
        y_full_init = _initialize_eq_multipliers_from_warm_start(
            warm_start,
            n=problem.n,
            T=problem.T,
            eq_dim=problem.eq_dim,
        )
        if y_full_init.size != y_dim:
            y_compact = np.zeros(y_dim, dtype=np.float64)
            common = min(y_dim, problem.n + problem.T * problem.n)
            y_compact[:common] = y_full_init[:common]
            y_full_init = y_compact
        if np.any(y_full_init):
            vars_in.y[:] = y_full_init
        if s_dim > 0:
            s_warm = _warm_start_vector(warm_start, "S", s_dim)
            z_dual_warm = _warm_start_vector(warm_start, "Z", s_dim)
            warm_floor = max(
                float(ss.barrier.initial_mu),
                1e-8,
            )
            s_warm = _strictly_positive_warm_start(s_warm, floor=warm_floor)
            z_dual_warm = _strictly_positive_warm_start(
                z_dual_warm,
                floor=warm_floor,
            )
            if s_warm is not None:
                vars_in.s[:] = s_warm
            if z_dual_warm is not None:
                vars_in.z[:] = z_dual_warm

        # --- Warm-up call -----------------------------------------------------
        # CasADi Functions have no JIT to amortise but we still call each
        # once to mirror the JAX path and to catch any structural issues
        # before timing begins.
        try:
            _ = float(f_fn(z_init))
            _ = np.asarray(c_fn(z_init))
            if s_dim > 0:
                _ = np.asarray(g_fn(z_init))
            _ = np.asarray(grad_f_fn(z_init))
            if y_dim > 0:
                _ = np.asarray(jac_c_fn(z_init))
            if s_dim > 0:
                _ = np.asarray(jac_g_fn(z_init))
            # Warm up each per-stage Hessian Function.
            for wb in stage_writebacks:
                _ = np.asarray(wb["fn"](z_init[wb["z_idx"]]))
        except Exception:  # noqa: BLE001
            pass

        # --- Solve -----------------------------------------------------------
        start = timer()
        try:
            output = solver.solve(vars_in)
            err = ""
        except Exception as e:  # noqa: BLE001
            output = None
            err = f"{type(e).__name__}: {e}"
        solve_time_ms = 1e3 * (timer() - start)

        z_val = np.asarray(vars_in.x, dtype=np.float64).copy()
        X, U, Theta = _slice_iterate(z_val, problem)

        # Multipliers are one-to-one with evaluate_problem's constraint
        # stacks. The CasADi NLP encodes the dyn defect as
        # X[t+1] - dyn(...) (same as the JAX backend, opposite from
        # evaluate_problem), so we sign-flip just the dyn-defect rows.
        try:
            multipliers_eq_full = np.asarray(
                vars_in.y,
                dtype=np.float64,
            ).reshape(-1)
            n_p = problem.n
            T_p = problem.T
            if multipliers_eq_full.size >= n_p + T_p * n_p:
                multipliers_eq_full[n_p : n_p + T_p * n_p] = -multipliers_eq_full[
                    n_p : n_p + T_p * n_p
                ]
        except Exception:  # noqa: BLE001
            multipliers_eq_full = None
        try:
            multipliers_ineq_full = np.asarray(
                vars_in.z,
                dtype=np.float64,
            ).reshape(-1)
            slacks_full = np.asarray(
                vars_in.s,
                dtype=np.float64,
            ).reshape(-1)
        except Exception:  # noqa: BLE001
            multipliers_ineq_full = None
            slacks_full = None
        if output is None:
            warm_start_out = _make_warm_start_out(
                multipliers_eq_full=multipliers_eq_full,
                multipliers_ineq_full=multipliers_ineq_full,
                slacks_full=slacks_full,
                n=problem.n,
                T=problem.T,
                eq_dim=problem.eq_dim,
            )
            return pack_solver_result(
                solver_name=self.name,
                problem_name=problem.name,
                problem=problem,
                X=X,
                U=U,
                Theta=Theta,
                iterations=0,
                solve_time_ms=solve_time_ms,
                success=False,
                notes=err,
                multipliers_eq=multipliers_eq_full,
                multipliers_ineq=multipliers_ineq_full,
                warm_start_out=warm_start_out,
                iterates_xut=iter_xut or None,
            )

        status = output.exit_status
        success = status == sip.Status.SOLVED
        notes = _sip_status_notes(output)
        warm_start_out = _make_warm_start_out(
            multipliers_eq_full=multipliers_eq_full,
            multipliers_ineq_full=multipliers_ineq_full,
            slacks_full=slacks_full,
            n=problem.n,
            T=problem.T,
            eq_dim=problem.eq_dim,
        )
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X,
            U=U,
            Theta=Theta,
            iterations=int(output.num_iterations),
            solve_time_ms=solve_time_ms,
            success=success,
            notes=notes,
            multipliers_eq=multipliers_eq_full,
            multipliers_ineq=multipliers_ineq_full,
            warm_start_out=warm_start_out,
            iterates_xut=iter_xut or None,
        )


@register("sip")
def _factory(**kwargs) -> SolverAdapter:
    return SipAdapter(**kwargs)
