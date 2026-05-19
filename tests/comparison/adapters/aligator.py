"""Aligator (Inria/LAAS) ProxDDP adapter.

Aligator is a Boost.Python / CMake project; it has no working PyPI sdist.
We install it via conda-forge into a side env (see
``tests/comparison/aligator_install.md``) and load it from this venv by
prepending the conda env's ``site-packages`` to ``sys.path``. The
adapter checks a couple of standard locations and a
``ALIGATOR_SITE_PACKAGES`` env override before declaring itself
unavailable.

Translation of a ``ProblemSpec`` into aligator:

* State manifold: ``aligator.manifolds.VectorSpace(n)`` (all our
  problems live on R^n).
* Dynamics: a Python subclass of ``aligator.dynamics.ExplicitDynamicsModel``
  that delegates ``forward`` / ``dForward`` to the JAX
  ``dynamics(x, u, theta, t)`` callback (and its JAX jacobian).
* Stage cost: a Python subclass of ``aligator.CostAbstract`` whose
  ``evaluate`` / ``computeGradients`` / ``computeHessians`` delegate to
  ``cost(x, u, theta, t)`` and its JAX grad/hessian.
* Terminal cost: same, with ``u`` pinned to zero and ``t`` to ``T``
  (matching the LIPA pad convention).
* Per-stage constraints:
  - ``equalities`` -> ``StageFunction`` + ``EqualityConstraintSet``
  - ``inequalities`` -> ``StageFunction`` + ``NegativeOrthant`` (LIPA
    convention is ``g(x,u) <= 0``).
* Terminal-stage constraints: same, added via
  ``problem.addTerminalConstraint(fn, set)``.

theta: aligator has no analogue of LIPA's cross-stage decision
variable. We refuse problems with ``theta_dim > 0`` (matching the CSQP
adapter).

Solver: ``aligator.SolverProxDDP`` (proximal augmented Lagrangian DDP).
``num_iters`` is the total outer ProxDDP iteration count (each one
runs an inner backward/forward pass and an AL multiplier update); we
report it as the ``iterations`` field. Convergence is judged by the
solver's ``conv`` flag.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from timeit import default_timer as timer

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


def _candidate_aligator_paths() -> list[Path]:
    """Search order for the conda-installed aligator.

    1. ``ALIGATOR_SITE_PACKAGES`` env var (used as-is if set).
    2. The default ``~/.conda/envs/aligator-side/lib/pythonX.Y/site-packages``
       layout produced by ``mamba create -n aligator-side aligator``.
    """
    out: list[Path] = []
    env = os.environ.get("ALIGATOR_SITE_PACKAGES")
    if env:
        out.append(Path(env))
    home = Path.home()
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    out.append(home / ".conda/envs/aligator-side/lib" / pyver / "site-packages")
    return out


def _ensure_aligator_on_path() -> bool:
    """Mutate ``sys.path`` so ``import aligator`` works.

    Returns ``True`` if aligator can subsequently be imported.
    """
    try:
        import aligator  # noqa: F401

        return True
    except ImportError:
        pass
    except Exception:
        # Boost.Python ABI clash (e.g. crocoddyl loaded from a different
        # source first). Aligator can't be loaded here.
        return False

    # If a competing pinocchio / crocoddyl / eigenpy is already loaded
    # in this process (e.g. csqp's adapter has run is_available), the
    # boost-python global type registry is already locked to that
    # ecosystem. Aligator (which ships its own pinocchio + eigenpy
    # under its conda env) cannot coexist with that, so refuse cleanly.
    if any(
        m in sys.modules for m in ("pinocchio", "eigenpy", "crocoddyl", "mim_solvers")
    ):
        return False

    for path in _candidate_aligator_paths():
        if path.is_dir() and (path / "aligator").is_dir() and str(path) not in sys.path:
            # Insert at the FRONT so aligator's bundled pinocchio /
            # eigenpy / proxsuite_nlp wins over the project venv's
            # versions. (See top-level note: aligator and csqp can't
            # share a single Python process due to boost-python ABI.)
            sys.path.insert(0, str(path))
            try:
                import aligator  # noqa: F401

                return True
            except ImportError:
                sys.path.remove(str(path))
            except Exception:
                # Path was correct but the import failed for another
                # reason (boost-python ABI clash, missing libstdc++, ...).
                # Leave the path on sys.path but report unavailable.
                return False
    return False


def _import_aligator():
    if not _ensure_aligator_on_path():
        raise ImportError(
            "Could not import aligator; see tests/comparison/aligator_install.md."
        )
    import aligator  # noqa: F401

    return aligator


# Eager import: aligator's Boost.Python bindings register a global type
# registry shared with proxsuite_nlp, pinocchio, eigenpy, etc. If
# crocoddyl (linked against a *different* libpinocchio + libeigenpy from
# the project venv) is imported first, aligator's `from .pyaligator
# import *` later raises "AttributeError: 'NoneType' object has no
# attribute '__dict__'" — the symbol-resolution graph ends up with a
# Boost type registered from one shared lib and dereferenced from
# another. Loading aligator at module-import time (i.e. before csqp.py's
# `_import_crocoddyl()` is ever called) lets aligator claim the global
# type slots first; crocoddyl is then willing to coexist (we only use
# crocoddyl from a *separate* benchmarking adapter, never in the same
# stage models).
#
# To run a CSQP-only pass without aligator hijacking sys.path, set
# ``LIPA_DISABLE_ALIGATOR=1`` in the environment. The adapter then
# reports unavailable rather than mutating the import path.
if os.environ.get("LIPA_DISABLE_ALIGATOR") == "1":
    _ALIGATOR_PRELOAD_OK = False
else:
    _ALIGATOR_PRELOAD_OK = _ensure_aligator_on_path()


def _build_casadi_callbacks(problem: ProblemSpec, t: int, terminal: bool):
    """Build per-stage CasADi-Function callbacks mirroring ``_build_jax_callbacks``.

    Pre-condition: ``problem.metadata["casadi_builder"]`` exists. The
    builder returns a dict with ``f`` (scalar cost), ``next_x`` (n-vector
    dynamics), ``eq`` (eq_dim vector or None), ``ineq`` (ineq_dim vector
    or None) — same protocol used by the IPOPT and fatrop adapters.

    Returned dict has the same keys as ``_build_jax_callbacks`` but each
    value is a ``ca.Function`` of two arguments ``(x, u)``. The Aligator
    subclasses below feed numpy arrays through these and read back numpy
    arrays — the per-call overhead is C++-level, no JAX dispatch.

    ``terminal`` controls whether ``u`` is symbolically pinned to zero in
    the gradients / hessians (matching the JAX terminal-cost convention).
    The cost / eq / ineq still take ``u`` formally so the same write-back
    layout in the Aligator subclasses works for both stages.
    """
    import casadi as ca

    builder = problem.metadata["casadi_builder"]
    n = problem.n
    m = problem.m
    td = problem.theta_dim

    x_sx = ca.SX.sym("x", n)
    u_sx = ca.SX.sym("u", m)
    theta_arg = (
        ca.DM(np.asarray(problem.Theta_init).reshape(-1)) if td > 0 else ca.SX.zeros(0)
    )

    stage = builder(x_sx, u_sx, theta_arg, t)
    f_sx = stage["f"]
    next_x_sx = stage.get("next_x")
    eq_sx = stage.get("eq")
    ineq_sx = stage.get("ineq")

    # For the terminal stage, the JAX path calls cost(x, 0, theta, T) and
    # routes its grad-wrt-u through a literal zero pad. Mirror that by
    # substituting u_sx -> 0 in every SX expression we build a derivative
    # of. (We keep ``u`` as a formal input to the ca.Function so the
    # callback signature is uniform across stages.)
    if terminal:
        zero_u = ca.SX.zeros(m)
        f_sx = ca.substitute(f_sx, u_sx, zero_u)
        if next_x_sx is not None:
            next_x_sx = ca.substitute(next_x_sx, u_sx, zero_u)
        if eq_sx is not None:
            eq_sx = ca.substitute(eq_sx, u_sx, zero_u)
        if ineq_sx is not None:
            ineq_sx = ca.substitute(ineq_sx, u_sx, zero_u)

    # --- Cost ---
    cost_fn = ca.Function("cost", [x_sx, u_sx], [f_sx])
    grad_x_sx = ca.gradient(f_sx, x_sx)
    grad_u_sx = ca.gradient(f_sx, u_sx)
    cost_grad_x_fn = ca.Function("cost_gx", [x_sx, u_sx], [grad_x_sx])
    cost_grad_u_fn = ca.Function("cost_gu", [x_sx, u_sx], [grad_u_sx])
    hess_xx_sx = ca.jacobian(grad_x_sx, x_sx)
    hess_uu_sx = ca.jacobian(grad_u_sx, u_sx)
    hess_xu_sx = ca.jacobian(grad_x_sx, u_sx)
    cost_hess_xx_fn = ca.Function("cost_hxx", [x_sx, u_sx], [hess_xx_sx])
    cost_hess_uu_fn = ca.Function("cost_huu", [x_sx, u_sx], [hess_uu_sx])
    cost_hess_xu_fn = ca.Function("cost_hxu", [x_sx, u_sx], [hess_xu_sx])

    # --- Dynamics ---
    if next_x_sx is None:
        # Terminal stage in our convention has no dynamics; supply zero
        # placeholders so the wrappers don't have to special-case.
        dyn_fn = ca.Function("dyn", [x_sx, u_sx], [ca.SX.zeros(n)])
        dyn_jac_x_fn = ca.Function("dyn_jx", [x_sx, u_sx], [ca.SX.zeros(n, n)])
        dyn_jac_u_fn = ca.Function("dyn_ju", [x_sx, u_sx], [ca.SX.zeros(n, m)])
    else:
        dyn_fn = ca.Function("dyn", [x_sx, u_sx], [next_x_sx])
        dyn_jac_x_fn = ca.Function(
            "dyn_jx", [x_sx, u_sx], [ca.jacobian(next_x_sx, x_sx)]
        )
        dyn_jac_u_fn = ca.Function(
            "dyn_ju", [x_sx, u_sx], [ca.jacobian(next_x_sx, u_sx)]
        )

    # --- Equalities ---
    if eq_sx is None or eq_sx.numel() == 0:
        # The "None / empty" case maps to nr == 0 -> the helper that wraps
        # this will skip building the StageFunction, so the values returned
        # here for eq don't actually get called. Still, define them with
        # the right zero shapes to make debugging less surprising.
        eq_fn = ca.Function("eq", [x_sx, u_sx], [ca.SX.zeros(0)])
        eq_jac_x_fn = ca.Function("eq_jx", [x_sx, u_sx], [ca.SX.zeros(0, n)])
        eq_jac_u_fn = ca.Function("eq_ju", [x_sx, u_sx], [ca.SX.zeros(0, m)])
    else:
        eq_fn = ca.Function("eq", [x_sx, u_sx], [eq_sx])
        eq_jac_x_fn = ca.Function("eq_jx", [x_sx, u_sx], [ca.jacobian(eq_sx, x_sx)])
        eq_jac_u_fn = ca.Function("eq_ju", [x_sx, u_sx], [ca.jacobian(eq_sx, u_sx)])

    # --- Inequalities ---
    if ineq_sx is None or ineq_sx.numel() == 0:
        ineq_fn = ca.Function("ineq", [x_sx, u_sx], [ca.SX.zeros(0)])
        ineq_jac_x_fn = ca.Function("ineq_jx", [x_sx, u_sx], [ca.SX.zeros(0, n)])
        ineq_jac_u_fn = ca.Function("ineq_ju", [x_sx, u_sx], [ca.SX.zeros(0, m)])
    else:
        ineq_fn = ca.Function("ineq", [x_sx, u_sx], [ineq_sx])
        ineq_jac_x_fn = ca.Function(
            "ineq_jx", [x_sx, u_sx], [ca.jacobian(ineq_sx, x_sx)]
        )
        ineq_jac_u_fn = ca.Function(
            "ineq_ju", [x_sx, u_sx], [ca.jacobian(ineq_sx, u_sx)]
        )

    # Track per-stage availability so the constraint-helper below can
    # short-circuit the StageFunction construction (matching the JAX
    # path's "always trivial" probe-and-skip, but driven by the builder
    # output instead of a numerical probe).
    has_eq = eq_sx is not None and eq_sx.numel() > 0
    has_ineq = ineq_sx is not None and ineq_sx.numel() > 0

    return {
        "cost": cost_fn,
        "cost_grad_x": cost_grad_x_fn,
        "cost_grad_u": cost_grad_u_fn,
        "cost_hess_xx": cost_hess_xx_fn,
        "cost_hess_uu": cost_hess_uu_fn,
        "cost_hess_xu": cost_hess_xu_fn,
        "dyn": dyn_fn,
        "dyn_jac_x": dyn_jac_x_fn,
        "dyn_jac_u": dyn_jac_u_fn,
        "eq": eq_fn,
        "eq_jac_x": eq_jac_x_fn,
        "eq_jac_u": eq_jac_u_fn,
        "ineq": ineq_fn,
        "ineq_jac_x": ineq_jac_x_fn,
        "ineq_jac_u": ineq_jac_u_fn,
        "_backend": "casadi",
        "_has_eq": has_eq,
        "_has_ineq": has_ineq,
    }


def _build_jax_callbacks(problem: ProblemSpec, t: int, terminal: bool):
    """JIT the per-stage cost / dynamics / equality / inequality callbacks.

    ``terminal`` toggles the ``u``-padded form needed for the terminal
    cost and any terminal equality / inequality.
    """
    n = problem.n
    m = problem.m
    theta = jnp.asarray(problem.Theta_init)
    cost_fn = problem.cost
    dyn_fn = problem.dynamics
    eq_fn = problem.equalities
    ineq_fn = problem.inequalities
    t_jax = jnp.int32(t)

    if terminal:
        # Terminal: u is empty / zero-padded; dynamics is unused. We
        # still need cost(x) and any equality/inequality at t=T.
        def _cost_xu(x, _u_dummy):
            return cost_fn(x, jnp.zeros(m, dtype=x.dtype), theta, t_jax)

        def _eq_xu(x, _u_dummy):
            if eq_fn is None:
                return jnp.empty(0, dtype=x.dtype)
            return eq_fn(x, jnp.zeros(m, dtype=x.dtype), theta, t_jax)

        def _ineq_xu(x, _u_dummy):
            if ineq_fn is None:
                return jnp.empty(0, dtype=x.dtype)
            return ineq_fn(x, jnp.zeros(m, dtype=x.dtype), theta, t_jax)

        def _dyn_xu(x, _u_dummy):
            return jnp.zeros_like(x)
    else:

        def _cost_xu(x, u):
            return cost_fn(x, u, theta, t_jax)

        def _eq_xu(x, u):
            if eq_fn is None:
                return jnp.empty(0, dtype=x.dtype)
            return eq_fn(x, u, theta, t_jax)

        def _ineq_xu(x, u):
            if ineq_fn is None:
                return jnp.empty(0, dtype=x.dtype)
            return ineq_fn(x, u, theta, t_jax)

        def _dyn_xu(x, u):
            return dyn_fn(x, u, theta, t_jax)

    return {
        "cost": jax.jit(_cost_xu),
        "cost_grad_x": jax.jit(jax.grad(_cost_xu, argnums=0)),
        "cost_grad_u": jax.jit(jax.grad(_cost_xu, argnums=1)),
        "cost_hess_xx": jax.jit(jax.hessian(_cost_xu, argnums=0)),
        "cost_hess_uu": jax.jit(jax.hessian(_cost_xu, argnums=1)),
        "cost_hess_xu": jax.jit(jax.jacobian(jax.grad(_cost_xu, argnums=0), argnums=1)),
        "dyn": jax.jit(_dyn_xu),
        "dyn_jac_x": jax.jit(jax.jacobian(_dyn_xu, argnums=0)),
        "dyn_jac_u": jax.jit(jax.jacobian(_dyn_xu, argnums=1)),
        "eq": jax.jit(_eq_xu),
        "eq_jac_x": jax.jit(jax.jacobian(_eq_xu, argnums=0)),
        "eq_jac_u": jax.jit(jax.jacobian(_eq_xu, argnums=1)),
        "ineq": jax.jit(_ineq_xu),
        "ineq_jac_x": jax.jit(jax.jacobian(_ineq_xu, argnums=0)),
        "ineq_jac_u": jax.jit(jax.jacobian(_ineq_xu, argnums=1)),
        "_backend": "jax",
        # Triviality flags only used by the casadi path (the jax path
        # has its own runtime probe-and-skip in _build_constraint_function).
        "_has_eq": True,
        "_has_ineq": True,
    }


def _build_callbacks(problem: ProblemSpec, t: int, terminal: bool, backend: str):
    """Dispatch to the JAX or CasADi callback builder by ``backend`` name."""
    if backend == "jax":
        return _build_jax_callbacks(problem, t, terminal)
    if backend == "casadi":
        return _build_casadi_callbacks(problem, t, terminal)
    raise ValueError(f"unknown backend={backend!r}; expected 'jax' or 'casadi'")


def _build_stage_dynamics(aligator, space, problem, t: int, backend: str = "jax"):
    """A Python ``ExplicitDynamicsModel`` calling the chosen-backend dynamics."""
    n = problem.n
    m = problem.m
    cb = _build_callbacks(problem, t, terminal=False, backend=backend)

    class JaxDynamics(aligator.dynamics.ExplicitDynamicsModel):
        def __init__(self_inner):  # noqa: N805
            super().__init__(space, m)

        def __getinitargs__(self_inner):  # noqa: N805
            return ()

        def forward(self_inner, x, u, data):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            u_np = np.asarray(u, dtype=np.float64).reshape(-1)
            xnext = np.asarray(cb["dyn"](x_np, u_np), dtype=np.float64).reshape(-1)
            np.asarray(data.xnext).reshape(-1)[:] = xnext

        def dForward(self_inner, x, u, data):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            u_np = np.asarray(u, dtype=np.float64).reshape(-1)
            jx = np.asarray(cb["dyn_jac_x"](x_np, u_np), dtype=np.float64).reshape(n, n)
            ju = np.asarray(cb["dyn_jac_u"](x_np, u_np), dtype=np.float64).reshape(n, m)
            np.asarray(data.Jx).reshape(n, n)[:] = jx
            # data.Ju collapses to (n,) when m == 1; reshape lets the
            # write-through still work on the same backing buffer.
            np.asarray(data.Ju).reshape(n, m)[:] = ju

    return JaxDynamics()


def _build_stage_cost(
    aligator, space, problem, t: int, terminal: bool, backend: str = "jax"
):
    """A Python ``CostAbstract`` calling the chosen-backend cost + autodiff."""
    n = problem.n
    m = problem.m
    cb = _build_callbacks(problem, t, terminal=terminal, backend=backend)

    class JaxCost(aligator.CostAbstract):
        def __init__(self_inner):  # noqa: N805
            super().__init__(space, m)

        def __getinitargs__(self_inner):  # noqa: N805
            return ()

        def evaluate(self_inner, x, u, data):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            u_np = np.asarray(u, dtype=np.float64).reshape(-1)
            data.value = float(cb["cost"](x_np, u_np))

        def computeGradients(self_inner, x, u, data):  # noqa: N805,N802
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            u_np = np.asarray(u, dtype=np.float64).reshape(-1)
            np.asarray(data.Lx).reshape(-1)[:] = np.asarray(
                cb["cost_grad_x"](x_np, u_np), dtype=np.float64
            ).reshape(-1)
            np.asarray(data.Lu).reshape(-1)[:] = np.asarray(
                cb["cost_grad_u"](x_np, u_np), dtype=np.float64
            ).reshape(-1)

        def computeHessians(self_inner, x, u, data):  # noqa: N805,N802
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            u_np = np.asarray(u, dtype=np.float64).reshape(-1)
            np.asarray(data.Lxx).reshape(n, n)[:] = np.asarray(
                cb["cost_hess_xx"](x_np, u_np), dtype=np.float64
            ).reshape(n, n)
            np.asarray(data.Luu).reshape(m, m)[:] = np.asarray(
                cb["cost_hess_uu"](x_np, u_np), dtype=np.float64
            ).reshape(m, m)
            np.asarray(data.Lxu).reshape(n, m)[:] = np.asarray(
                cb["cost_hess_xu"](x_np, u_np), dtype=np.float64
            ).reshape(n, m)

    return JaxCost()


def _build_constraint_function(
    aligator, space, problem, t: int, terminal: bool, kind: str, backend: str = "jax"
):
    """A Python ``StageFunction`` for either equalities or inequalities.

    ``kind`` is ``"eq"`` or ``"ineq"``. Returns ``None`` if the problem
    doesn't have a constraint of that kind, or if the per-stage output
    is structurally a zero function (otherwise we'd feed aligator a
    redundant rank-deficient constraint that NaNs the inner Newton
    step).
    """
    n = problem.n
    m = problem.m
    cb = _build_callbacks(problem, t, terminal=terminal, backend=backend)

    if kind == "eq":
        nr = problem.eq_dim
        eval_fn = cb["eq"]
        jx_fn = cb["eq_jac_x"]
        ju_fn = cb["eq_jac_u"]
        has_kind = cb["_has_eq"]
    elif kind == "ineq":
        nr = problem.ineq_dim
        eval_fn = cb["ineq"]
        jx_fn = cb["ineq_jac_x"]
        ju_fn = cb["ineq_jac_u"]
        has_kind = cb["_has_ineq"]
    else:  # pragma: no cover
        raise ValueError(f"unknown constraint kind: {kind}")

    if nr == 0:
        return None

    # CasADi short-circuit: the casadi_builder already returns ``None`` /
    # zero-numel for stages where this constraint is absent. Trust that
    # signal directly (no need for the JAX-path's runtime probe).
    if backend == "casadi" and not has_kind:
        return None

    # Probe-and-skip: if the constraint is a constant-zero function (no
    # dependence on x or u and value identically 0) for inequalities,
    # OR identically zero for equalities (value AND Jacobian zero), skip
    # it. This matches the LIPA pad convention. We only skip for
    # equalities, since a constant-zero inequality is automatically
    # satisfied and harmless.
    if kind == "eq" and backend == "jax":
        # Probe at three different (x, u) samples drawn from the warm
        # start; if all values and jacobians are exactly zero, we treat
        # this as a no-op constraint at this stage.
        rng = np.random.default_rng(0xA11_6A_70 ^ (t << 1) ^ int(terminal))
        sample_x = [np.asarray(problem.X_init[min(t, problem.T)], dtype=np.float64)]
        if problem.X_init.shape[0] > 1:
            sample_x.append(np.asarray(problem.X_init[0], dtype=np.float64))
        sample_x.append(rng.standard_normal(n))
        sample_u = [np.zeros(m, dtype=np.float64), rng.standard_normal(m)]

        always_trivial = True
        for x_s in sample_x:
            for u_s in sample_u:
                v = np.asarray(eval_fn(x_s, u_s))
                jx = np.asarray(jx_fn(x_s, u_s))
                ju = np.asarray(ju_fn(x_s, u_s))
                if (
                    not np.all(v == 0.0)
                    or not np.all(jx == 0.0)
                    or not np.all(ju == 0.0)
                ):
                    always_trivial = False
                    break
            if not always_trivial:
                break
        if always_trivial:
            return None

    class JaxConstraint(aligator.StageFunction):
        def __init__(self_inner):  # noqa: N805
            super().__init__(n, m, nr)

        def __getinitargs__(self_inner):  # noqa: N805
            return ()

        def evaluate(self_inner, x, u, data):  # noqa: N805
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            u_np = np.asarray(u, dtype=np.float64).reshape(-1)
            v = np.asarray(eval_fn(x_np, u_np), dtype=np.float64).reshape(-1)
            np.asarray(data.value).reshape(-1)[:] = v

        def computeJacobians(self_inner, x, u, data):  # noqa: N805,N802
            x_np = np.asarray(x, dtype=np.float64).reshape(-1)
            u_np = np.asarray(u, dtype=np.float64).reshape(-1)
            jx = np.asarray(jx_fn(x_np, u_np), dtype=np.float64).reshape(nr, n)
            ju = np.asarray(ju_fn(x_np, u_np), dtype=np.float64).reshape(nr, m)
            np.asarray(data.Jx).reshape(nr, n)[:] = jx
            np.asarray(data.Ju).reshape(nr, m)[:] = ju

    return JaxConstraint()


class AligatorAdapter(SolverAdapter):
    """Build an ``aligator.TrajOptProblem`` from a ``ProblemSpec`` and run ProxDDP."""

    name = "aligator"

    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-6,
        mu_init: float = 1e-1,
        max_al_iters: int = 100,
        multiplier_update_mode: str = "PRIMAL",
        verbose: bool = False,
        backend: str = "jax",
        aligator_extra_settings: dict | None = None,
    ) -> None:
        # ``backend`` selects the per-stage callback runtime that backs
        # Aligator's Python ExplicitDynamicsModel / CostAbstract /
        # StageFunction subclasses:
        #
        # * ``"jax"`` (default): each cost / dynamics / jacobian /
        #   hessian call is a JAX-jitted host-callback. Works for any
        #   ProblemSpec, including MJX problems with no casadi_builder.
        # * ``"casadi"``: build per-stage ``ca.Function`` objects from
        #   ``problem.metadata["casadi_builder"]``. ProxDDP is
        #   unchanged; only the callback runtime changes, so iter
        #   counts should match the JAX backend bit-for-bit with much
        #   lower per-iter wall time. Requires a casadi_builder; MJX
        #   problems don't have one and report ``unavailable``.
        # ``mu_init`` is the initial AL penalty weight. The documented
        # default (1e-2) is too small for problems with large
        # stage-vs-terminal Hessian disparity; 0.1 lets the AL outer
        # loop ramp adaptively from a stabler starting point.
        #
        # ``multiplier_update_mode``: the "NEWTON" default can stall the
        # backward pass on tightly-constrained problems (alpha collapses,
        # proximal regularizer climbs, eventual NaN). "PRIMAL" performs
        # an Uzawa-style update from the primal residual which defers
        # / softens that failure mode.
        #
        # ``aligator_extra_settings``: a dict of free-form knobs applied
        # to the SolverProxDDP instance AFTER the standard constructor
        # args. Keys come in two flavors:
        #   * Top-level attributes of SolverProxDDP, e.g. ``ls_mode``
        #     ("PRIMAL" / "PRIMAL_DUAL"), ``sa_strategy``
        #     ("SA_LINESEARCH_NONMONOTONE" / "SA_FILTER" /
        #     "SA_LINESEARCH_ARMIJO"), ``linear_solver_choice``
        #     ("LQ_SOLVER_SERIAL" / "LQ_SOLVER_PARALLEL" /
        #     "LQ_SOLVER_STAGEDENSE"), ``rollout_type``
        #     ("ROLLOUT_NONLINEAR" / "ROLLOUT_LINEAR"), ``preg``,
        #     ``reg_init``, ``reg_min``, ``reg_max``, ``dual_weight``,
        #     ``rollout_max_iters``, ``num_threads``,
        #     ``max_refinement_steps``, ``refinement_threshold``.
        #     String values for the four enum attributes above are
        #     auto-resolved via getattr on the corresponding aligator
        #     enum class.
        #   * Nested dicts ``bcl_params`` and ``ls_params``: each entry
        #     is setattr'd onto solver.bcl_params / solver.ls_params.
        #     bcl_params knobs: ``mu_lower_bound``, ``mu_update_factor``,
        #     ``prim_alpha``, ``prim_beta``, ``dual_alpha``, ``dual_beta``,
        #     ``dyn_al_scale``. ls_params knobs: ``alpha_min``, ``armijo_c1``,
        #     ``contraction_min``, ``contraction_max``, ``dphi_thresh``,
        #     ``max_num_steps``, ``wolfe_c2``.
        # Unknown keys raise AttributeError (loud failure so a typo'd
        # tuning sweep doesn't silently no-op).
        self.max_iter = max_iter
        self.tol = tol
        self.mu_init = mu_init
        self.max_al_iters = max_al_iters
        self.multiplier_update_mode = multiplier_update_mode
        self.verbose = verbose
        if backend not in {"jax", "casadi"}:
            raise ValueError(
                f"unknown aligator backend={backend!r}; expected 'jax' or 'casadi'"
            )
        self.backend = backend
        self.aligator_extra_settings = aligator_extra_settings or {}

    def is_available(self) -> tuple[bool, str]:
        if not _ALIGATOR_PRELOAD_OK:
            return False, (
                "aligator could not be imported at adapter-module load "
                "time; see tests/comparison/aligator_install.md for the "
                "conda-forge install path (and Boost.Python ABI notes)"
            )
        try:
            _import_aligator()
        except ImportError as e:
            return False, (
                f"{e}; see tests/comparison/aligator_install.md for the "
                f"conda-forge install path"
            )
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
                f"aligator does not support cross-stage Theta "
                f"(theta_dim={problem.theta_dim})",
            )

        if problem.metadata.get("is_mjx") and not problem.metadata.get(
            "aligator_two_phase",
            False,
        ):
            # Aligator's MJX integration is structurally impractical with
            # the current Python API. Per ProxDDP iter, Aligator's C++
            # inner loop calls back into Python ~16 times per stage (one
            # call each for cost, cost_grad_x, cost_grad_u, cost_hess_xx,
            # cost_hess_uu, cost_hess_xu, dynamics, dyn_jac_x, dyn_jac_u,
            # eq, eq_jac_x, eq_jac_u, ineq, ineq_jac_x, ineq_jac_u, ...).
            # That's many boundary crossings per outer iter, each ~1ms
            # minimum even fully JIT-cached. The shared-jit trick used
            # by ipopt-mjx / fatrop-mjx / csqp addresses COMPILE cost,
            # not CALL COUNT, so it doesn't help here.
            # The underlying problem IS batchable in principle (per-stage
            # function evaluations are independent — only the Riccati
            # backward recursion is sequential), but Aligator's Python
            # API is per-stage by design and exposes no batch entry
            # point. A real Aligator-on-MJX adapter would need either an
            # upstream patch to Aligator's C++ side (batch callbacks) or
            # a fragile cache-multiple-stages-per-call hack in this
            # adapter. Skipping cleanly is the right call until one of
            # those is done. See tests/comparison/README.md for the
            # full rationale.
            #
            # Per-problem opt-in via ``metadata['aligator_two_phase'] = True``
            # lets a problem ask for the Phase-1 soft-penalty / Phase-2
            # main solve orchestration (matching the LIPA-shaped split
            # used elsewhere in the benchmark). Even with that, the
            # per-call overhead is large; the opt-in exists so problems
            # whose Phase-1 warmup is cheap enough can still be attempted.
            return make_failure_result(
                self.name,
                problem.name,
                "aligator MJX integration is impractical due to call "
                "overhead; see tests/comparison/README.md "
                '("Solver coverage notes"). Opt in per-problem via '
                "metadata['aligator_two_phase'] = True if the "
                "Phase-1 warmup is cheap enough to make it tractable.",
            )

        if self.backend == "casadi" and "casadi_builder" not in problem.metadata:
            return make_failure_result(
                self.name,
                problem.name,
                (
                    "aligator(casadi) requires problem.metadata['casadi_builder']; "
                    "this problem has no SX builder (MJX problems take the "
                    "jax backend only)"
                ),
            )

        aligator = _import_aligator()
        from aligator import constraints, manifolds

        n = problem.n
        m = problem.m
        T = problem.T
        space = manifolds.VectorSpace(n)

        # Build per-stage dynamics, cost, equality, and inequality.
        be = self.backend
        dyn_models = [
            _build_stage_dynamics(aligator, space, problem, t, backend=be)
            for t in range(T)
        ]
        cost_models = [
            _build_stage_cost(aligator, space, problem, t, terminal=False, backend=be)
            for t in range(T)
        ]
        eq_models = [
            _build_constraint_function(
                aligator, space, problem, t, terminal=False, kind="eq", backend=be
            )
            for t in range(T)
        ]
        ineq_models = [
            _build_constraint_function(
                aligator, space, problem, t, terminal=False, kind="ineq", backend=be
            )
            for t in range(T)
        ]

        term_cost = _build_stage_cost(
            aligator, space, problem, T, terminal=True, backend=be
        )
        term_eq = _build_constraint_function(
            aligator, space, problem, T, terminal=True, kind="eq", backend=be
        )
        term_ineq = _build_constraint_function(
            aligator, space, problem, T, terminal=True, kind="ineq", backend=be
        )

        x0 = np.asarray(problem.x0, dtype=np.float64)
        traj_problem = aligator.TrajOptProblem(x0, m, space, term_cost=term_cost)
        for t in range(T):
            stage = aligator.StageModel(cost_models[t], dyn_models[t])
            if eq_models[t] is not None:
                stage.addConstraint(eq_models[t], constraints.EqualityConstraintSet())
            if ineq_models[t] is not None:
                stage.addConstraint(ineq_models[t], constraints.NegativeOrthant())
            traj_problem.addStage(stage)
        if term_eq is not None:
            traj_problem.addTerminalConstraint(
                term_eq, constraints.EqualityConstraintSet()
            )
        if term_ineq is not None:
            traj_problem.addTerminalConstraint(term_ineq, constraints.NegativeOrthant())

        # Initial guess.
        xs_init = [np.asarray(x, dtype=np.float64) for x in np.asarray(problem.X_init)]
        us_init = [np.asarray(u, dtype=np.float64) for u in np.asarray(problem.U_init)]

        verbose_level = (
            aligator.VerboseLevel.VERBOSE
            if self.verbose
            else aligator.VerboseLevel.QUIET
        )

        # Map of attribute names that take an aligator enum (so a JSON
        # string value can be auto-resolved). Centralised here so adding
        # a new enum-valued knob is a one-line change.
        enum_attr_map = {
            "ls_mode": aligator.LinesearchMode,
            "sa_strategy": aligator.StepAcceptanceStrategy,
            "linear_solver_choice": aligator.LQSolverChoice,
            "rollout_type": aligator.RolloutType,
            "multiplier_update_mode": aligator.MultiplierUpdateMode,
        }

        def _resolve(attr_name, value):
            """Resolve an enum string -> enum value; pass through scalars."""
            if isinstance(value, str) and attr_name in enum_attr_map:
                try:
                    return getattr(enum_attr_map[attr_name], value)
                except AttributeError as e:
                    raise AttributeError(
                        f"aligator: enum {attr_name!r} has no member {value!r} "
                        f"(available: {list(enum_attr_map[attr_name].__members__)})"
                    ) from e
            return value

        def _apply_nested(holder, nested_attr, sub_settings, holder_label):
            """setattr each (k, v) of sub_settings on holder.<nested_attr>.

            Unknown attributes warn-and-skip (not a hard failure):
            aligator's BCLParams / LSParams surface drifts across
            versions, but a config that asks for a non-existent attr
            is still a bug that should be loud.
            """
            import sys

            sub = getattr(holder, nested_attr)
            for k, v in sub_settings.items():
                if not hasattr(sub, k):
                    print(
                        f"  WARN: aligator {holder_label}.{nested_attr} "
                        f"has no attribute {k!r}; skipping",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                setattr(sub, k, v)

        def _apply_extra_settings(s, extras):
            """Apply ``aligator_extra_settings`` to a SolverProxDDP instance.

            Handles the two nested dicts ``bcl_params`` and ``ls_params``
            specially; everything else is a top-level setattr (with
            enum-string auto-resolution).
            """
            for k, v in extras.items():
                if k == "bcl_params":
                    if not isinstance(v, dict):
                        raise TypeError(
                            "aligator: 'bcl_params' must be a dict, got "
                            f"{type(v).__name__}"
                        )
                    _apply_nested(s, "bcl_params", v, "solver")
                elif k == "ls_params":
                    if not isinstance(v, dict):
                        raise TypeError(
                            "aligator: 'ls_params' must be a dict, got "
                            f"{type(v).__name__}"
                        )
                    _apply_nested(s, "ls_params", v, "solver")
                else:
                    if not hasattr(s, k):
                        raise AttributeError(
                            f"aligator: SolverProxDDP has no attribute {k!r}"
                        )
                    setattr(s, k, _resolve(k, v))

        def _make_solver():
            s = aligator.SolverProxDDP(
                tol,
                self.mu_init,
                max_iters=self.max_iter,
                verbose=verbose_level,
            )
            s.max_al_iters = self.max_al_iters
            s.force_initial_condition = True
            try:
                s.multiplier_update_mode = getattr(
                    aligator.MultiplierUpdateMode, self.multiplier_update_mode
                )
            except AttributeError:
                # Older aligator versions may not export every enum value.
                # Fall back silently (default is NEWTON).
                pass
            # Per-problem layers, applied in order so later layers
            # shadow earlier ones, ending with the CLI override:
            #   aligator_settings (shared, both backends)
            #   aligator_<backend>_settings (backend-specific)
            #   self.aligator_extra_settings (CLI override)
            problem_settings = problem.metadata.get("aligator_settings", {})
            if problem_settings:
                _apply_extra_settings(s, problem_settings)
            backend_settings = problem.metadata.get(
                f"aligator_{self.backend}_settings",
                {},
            )
            if backend_settings:
                _apply_extra_settings(s, backend_settings)
            if self.aligator_extra_settings:
                _apply_extra_settings(s, self.aligator_extra_settings)
            return s

        # Warm up: amortize JAX JIT compile out of the timed solve.
        try:
            warmup = _make_solver()
            warmup.max_iters = 1
            warmup.setup(traj_problem)
            warmup.run(traj_problem, xs_init, us_init)
        except Exception:  # noqa: BLE001
            pass

        solver = _make_solver()
        solver.setup(traj_problem)
        start = timer()
        try:
            ok = bool(solver.run(traj_problem, xs_init, us_init))
            err = ""
        except Exception as e:  # noqa: BLE001
            ok = False
            err = f"{type(e).__name__}: {e}"
        solve_time_ms = 1e3 * (timer() - start)

        # Multiplier extraction. aligator's results expose a `lams`
        # vector of per-stage Lagrange multipliers (dynamics + path
        # constraint multipliers stacked) and `vs` co-states. The
        # exact split depends on the StageModel API. We attempt to
        # extract `results.lams` if available; otherwise leave as
        # None. Per-iter history would require addCallback with a
        # SolverProxDDP-specific signature; given Aligator's
        # boost-python ABI fragility (it can't share a process with
        # crocoddyl/mim_solvers, see _ALIGATOR_PRELOAD_OK), we keep
        # the instrumentation minimal: extract final-iterate
        # multipliers when available, leave per-iter history as
        # None (the convergence plot falls back to an endpoint marker
        # for aligator, matching Trajax).
        multipliers_eq = None
        multipliers_ineq = None
        try:
            res = solver.results
            xs_out = np.asarray([np.asarray(x, dtype=np.float64) for x in res.xs])
            us_out = np.asarray([np.asarray(u, dtype=np.float64) for u in res.us])
            iters = int(res.num_iters)
            # Attempt to pull lams. Each entry in `res.lams` is the
            # full per-stage stacked multiplier vector (size depends
            # on what constraints are attached at that stage). The
            # mapping back to evaluate_problem's eq / ineq stacks is
            # non-trivial without per-stage probing of the StageModel
            # constraint dimensions, so we don't expose a coherent
            # eq / ineq split here. Storing the raw lams for inspection.
            if hasattr(res, "lams"):
                try:
                    lams_raw = [
                        np.asarray(L, dtype=np.float64).reshape(-1) for L in res.lams
                    ]
                    multipliers_eq = np.concatenate(lams_raw) if lams_raw else None
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            xs_out = np.asarray(problem.X_init)
            us_out = np.asarray(problem.U_init)
            iters = 0
            err = err or f"{type(e).__name__}: {e}"
            ok = False

        Theta = np.asarray(problem.Theta_init)
        # Multipliers' shapes don't match evaluate_problem's stacks
        # (see comment above); skip the KKT computation but keep the
        # raw multipliers on the SolverResult for downstream consumers.
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
            compute_kkt=False,
        )


@register("aligator")
def _factory(**kwargs) -> SolverAdapter:
    return AligatorAdapter(**kwargs)
