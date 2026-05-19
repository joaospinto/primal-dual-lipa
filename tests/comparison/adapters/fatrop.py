"""fatrop adapter — structured interior-point solver for OCPs.

fatrop (https://github.com/meco-group/fatrop) is the closest direct
algorithmic competitor to LIPA in this comparison: it's a primal-dual
interior-point solver that exploits the multi-shooting OCP block
structure via a generalized Riccati recursion (BLASFEO-backed). The
acados team benchmarks against it in their papers, so a head-to-head
LIPA / fatrop number on the same problems is informative.

## Why CasADi-Opti?

fatrop ships a CasADi ``nlpsol`` plugin and the CasADi pip wheel
(>= 3.7) bundles a prebuilt copy. So the natural Python surface here
is CasADi's ``Opti`` builder configured with ``solver('fatrop', ...)``
and ``structure_detection='manual'`` — there is no separate ``fatropy``
PyPI package. (fatrop's own "low-level" C++ ``OcpAbstract`` interface
exists but has no Python binding upstream; the CasADi route is what
the official examples use.)

## OCP layout

We mirror the ``ProblemSpec`` stage-by-stage:

* For ``k = 0..T-1``: the variables are ``(x_k, u_k)`` and the
  per-stage path constraints are the user equalities (``g == 0``) plus
  inequalities (``g <= 0``); the dynamics defect ``x_{k+1} -
  next_x(x_k, u_k) = 0`` is added separately and is what fatrop's
  Riccati recursion exploits.
* For ``k = T``: only ``x_T`` is a variable (``nu[T] = 0``) and the
  terminal cost / equality / inequality slots come from the same
  ``casadi_builder`` evaluated at ``t=T``. The dummy ``u_T`` is fed in
  as a zero ``SX`` so the builder doesn't see ``None``.

The initial-state defect ``x_0 - x0 = 0`` goes into the ``k=0`` path
constraints (so it counts toward ``ng[0]``).

## Encoding constraints

CasADi-fatrop expects path constraints as ``lb <= g_k(x_k, u_k) <= ub``;
we mirror the LIPA convention by setting:

* equality rows: ``lb = ub = 0``
* inequality rows: ``lb = -inf``, ``ub = 0``  (LIPA's ``ineq <= 0``)

## Warm start

We hand fatrop a per-problem ``(X_init, U_init)`` warm start through
``opti.set_initial``:

* If the problem has an *affine* terminal equality constraint
  ``x_T - goal = 0``, build X as ``linspace(x0, goal)`` regardless
  of what ``problem.X_init`` contains. The ``goal`` is recovered by
  reading the constant offset of the terminal equality (jacobian
  must be the identity for this to work; otherwise we fall back to
  the shipped ``X_init``).
* Otherwise, use the shipped ``problem.X_init`` directly.
* ``U_init`` is always the shipped one.

From a constant ``tile(x0)`` warm start fatrop can drop into the
restoration phase on problems with a terminal-equality goal (the
filter line search has trouble simultaneously improving the barrier
objective and the terminal residual). The ``linspace(x0, GOAL)``
substitution lets the filter make progress.

The dynamics-rollout warm start (``U_init = 0`` rolled forward,
which acados uses) also tends to land fatrop in restoration on these
problems. fatrop, like LIPA, is a primal-dual interior-point method
and absorbs constraint violation through slacks, so it tolerates a
dynamics-infeasible iterate as long as the iterate "points toward"
the goal.

acados-SQP needs the dynamics-feasible warm start because its outer
loop solves a QP with linearized constraints — at a far-from-feasible
warm start that QP is itself infeasible. fatrop has no such issue.

## Skipped scenarios

* ``theta_dim > 0``: fatrop's OCP layout has no notion of a
  cross-stage decision variable. We could add it as a per-stage
  variable replicated everywhere with equality glue, but it would
  defeat fatrop's Riccati structure. We skip cleanly with
  ``success=False, notes='...'`` (same policy as acados / Aligator /
  CSQP for this case).
* No ``casadi_builder`` in metadata: fatrop has no JAX-callback path
  in this adapter — it would require the same per-stage CasADi
  callback machinery the IPOPT JAX adapter uses, and none of the
  problems that hit this branch (the MJX ones) are realistic targets
  for fatrop anyway. We skip with a clear message.

## Timing

The reported `solve_time_ms` covers `opti.solve()` only — the CasADi
expand + per-OCP BLASFEO setup happens inside `_build_opti` (i.e.
`opti.solver('fatrop', ...)`), which runs *before* the timer starts.
Same convention as the acados and CSQP adapters. fatrop's own
internal BLASFEO buffer initialization is part of the first solver
call's CPU time and is included in the reported number.
"""

from __future__ import annotations

from timeit import default_timer as timer

import numpy as np

from tests.comparison.adapters import register
from tests.comparison.adapters.base import SolverAdapter
from tests.comparison.problem_spec import (
    ProblemSpec,
    SolverResult,
    make_failure_result,
    pack_solver_result,
)


def _import_casadi():
    import casadi as ca  # local import so a missing CasADi only fails this adapter

    return ca


def _fatrop_plugin_available(ca) -> tuple[bool, str]:
    """Return (True, '') if ``nlpsol(..., 'fatrop', ...)`` works, else (False, reason)."""
    try:
        nlp = {"x": ca.SX.sym("x"), "f": ca.SX(1.0), "g": ca.SX(0.0)}
        ca.nlpsol("probe", "fatrop", nlp, {
            "structure_detection": "manual",
            "nx": [1, 0],
            "nu": [0, 0],
            "ng": [0, 0],
            "N": 1,
            "fatrop": {"print_level": 0},
        })
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "Plugin" in msg and ("could not be loaded" in msg or "not found" in msg):
            return False, f"CasADi fatrop plugin missing: {msg.splitlines()[0]}"
        # Construction with a degenerate 1-var problem can fail for other
        # reasons; if the error mentions fatrop at all the plugin is
        # at least loaded.
        return True, ""
    return True, ""


# Linspace-toward-extracted-goal warm start (CasADi-builder route) lives
# in tests.comparison.warm_starts. See this module's docstring for the
# rationale.
from tests.comparison.warm_starts import (  # noqa: E402
    linspace_to_casadi_extracted_goal_warm_start as _initial_iterate,
)


def _build_opti(problem: ProblemSpec, *, max_iter: int, tol: float):
    """Translate ``problem`` into a CasADi ``Opti`` configured for fatrop.

    Returns ``(opti, x_vars, u_vars, nx, nu, ng)`` where ``x_vars`` and
    ``u_vars`` are length-(T+1) and length-(T+1) lists of Opti symbols
    (``u_vars[T]`` is an empty 0x1 placeholder so the indexing matches
    the conventions of CasADi-fatrop's ``nu`` argument, which expects
    length N+1).
    """
    ca = _import_casadi()

    if problem.theta_dim > 0:
        raise ValueError(
            f"fatrop adapter cannot handle cross-stage Theta "
            f"(theta_dim={problem.theta_dim})",
        )
    if "casadi_builder" not in problem.metadata:
        raise ValueError(
            "fatrop adapter requires problem.metadata['casadi_builder']; "
            "this is the same builder the IPOPT and acados adapters use.",
        )

    casadi_builder = problem.metadata["casadi_builder"]
    n, m, T = problem.n, problem.m, problem.T
    K = T + 1  # total number of stages

    opti = ca.Opti()

    # Per-stage symbolic variables. CasADi-fatrop's manual structure
    # detection expects ``opti.x`` to be packed in stage-interleaved
    # order ``[x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T]`` — that
    # block-diagonal layout is what its Riccati linear solver exploits.
    # We have to allocate the variables in that exact order.
    # ``nu[T] = 0`` means no control variable at the terminal stage; we
    # still keep a length-(T+1) ``u_vars`` list with a zero ``MX`` at
    # index T so the per-stage code below can index uniformly.
    x_vars = []
    u_vars: list = []
    for k in range(K):
        x_vars.append(opti.variable(n))
        if k < T:
            u_vars.append(opti.variable(m))
    u_vars.append(ca.MX.zeros(m, 1))  # dummy zero "control" at t=T (not an Opti variable)

    nx_list = [n] * K
    nu_list = [m] * T + [0]
    ng_list: list[int] = []
    # Per-stage layout metadata for the multiplier remapping below.
    # eq_per_stage[k] = number of *user* equality rows added at stage k
    # (excludes the init defect and dyn defect, which fatrop tracks
    # separately). ineq_per_stage[k] same for inequalities.
    eq_per_stage = np.zeros(K, dtype=np.intp)
    ineq_per_stage = np.zeros(K, dtype=np.intp)
    init_eq_at_stage_0 = True  # always true (we always add it)

    # Initial-state defect goes into the k=0 path constraints (so it
    # counts toward ng[0]); it's an equality lb = ub = 0.
    init_eq = x_vars[0] - ca.DM(np.asarray(problem.x0, dtype=np.float64))

    # Cost accumulator.
    f_total = ca.MX(0.0)

    # Track per-stage constraint blocks added to opti so we can map the
    # solution-time lam_g back to the per-stage init/eq/ineq splits.
    constraint_blocks = []  # list of {kind, stage, size}
    for k in range(K):
        is_terminal = (k == T)
        x_k = x_vars[k]
        u_k = u_vars[k]
        # Build the stage expressions through the same builder the IPOPT
        # / acados adapters use. theta_arg is empty (theta_dim == 0 here).
        theta_arg = ca.SX.zeros(0)
        # The builder takes plain SX symbols; we have MX Opti vars, so
        # we go through ca.Function. (The casadi_builder branches on
        # ``t == T``, so we need to call it with the right ``t`` value.)
        x_sx = ca.SX.sym("x", n)
        u_sx = ca.SX.sym("u", m)
        stage = casadi_builder(x_sx, u_sx, theta_arg, k)

        # Cost term.
        f_fn = ca.Function("stage_f", [x_sx, u_sx], [stage["f"]])
        f_total = f_total + f_fn(x_k, u_k)

        # Dynamics defect (only for non-terminal stages).
        if not is_terminal:
            if stage.get("next_x") is None:
                raise ValueError(f"casadi_builder(t={k}) must return 'next_x'")
            dyn_fn = ca.Function("stage_dyn", [x_sx, u_sx], [stage["next_x"]])
            opti.subject_to(x_vars[k + 1] == dyn_fn(x_k, u_k))
            constraint_blocks.append({"kind": "dyn", "stage": k, "size": n})

        # Path constraints: equality + inequality glue.
        # Each stage's "ng" counts the rows of *path* constraints (the
        # dynamics defect above does NOT count toward ng — fatrop tracks
        # it separately as the OCP gap).
        eq_rows: list = []
        ineq_rows: list = []
        if k == 0:
            eq_rows.append(init_eq)

        stage_eq = stage.get("eq")
        if stage_eq is not None and stage_eq.numel() > 0:
            eq_fn = ca.Function("stage_eq", [x_sx, u_sx], [stage_eq])
            eq_rows.append(eq_fn(x_k, u_k))
            eq_per_stage[k] = int(stage_eq.numel())
        stage_ineq = stage.get("ineq")
        if stage_ineq is not None and stage_ineq.numel() > 0:
            ineq_fn = ca.Function("stage_ineq", [x_sx, u_sx], [stage_ineq])
            ineq_rows.append(ineq_fn(x_k, u_k))
            ineq_per_stage[k] = int(stage_ineq.numel())

        n_eq = sum(int(r.numel()) for r in eq_rows)
        n_ineq = sum(int(r.numel()) for r in ineq_rows)

        if eq_rows:
            opti.subject_to(ca.vertcat(*eq_rows) == ca.MX.zeros(n_eq, 1))
            # Record sub-blocks with kinds 'init' (k==0 only) and 'eq'.
            if k == 0:
                constraint_blocks.append({"kind": "init", "stage": 0, "size": n})
            if eq_per_stage[k] > 0:
                constraint_blocks.append({
                    "kind": "eq", "stage": k, "size": int(eq_per_stage[k]),
                })
        if ineq_rows:
            opti.subject_to(ca.vertcat(*ineq_rows) <= ca.MX.zeros(n_ineq, 1))
            constraint_blocks.append({
                "kind": "ineq", "stage": k, "size": int(ineq_per_stage[k]),
            })

        ng_list.append(n_eq + n_ineq)

    opti.minimize(f_total)

    fatrop_opts = {
        "tol": float(tol),
        "max_iter": int(max_iter),
        "print_level": 0,
        # mu_init: starting barrier parameter (fatrop's default).
        "mu_init": 1e-1,
    }
    plugin_opts = {
        "structure_detection": "manual",
        "nx": nx_list,
        "nu": nu_list,
        "ng": ng_list,
        "N": T,
        "expand": True,
        "fatrop": fatrop_opts,
        "print_time": False,
    }
    opti.solver("fatrop", plugin_opts)
    return opti, x_vars, u_vars, nx_list, nu_list, ng_list, constraint_blocks


def _remap_fatrop_lam_g(
    lam_g: np.ndarray,
    constraint_blocks: list,
    problem: ProblemSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Splice fatrop's lam_g (ordered per the constraint_blocks list) into
    the (eq, ineq) stacks expected by evaluate_problem.

    Returns ``(multipliers_eq, multipliers_ineq)`` of shapes
    ``(n + T*n + (T+1)*eq_dim,)`` and ``((T+1)*ineq_dim,)`` respectively.

    Sign conventions: CasADi-Opti's ``opti.lam_g`` follows IPOPT's
    convention (``L = f + λ^T g`` with the constraint AS WRITTEN).
    Our constraint forms are ``X[t+1] - dyn(...) = 0`` (sign-flipped vs
    evaluator's dyn_defect = dyn(...) - X[t+1]) and
    ``init_eq = X[0] - x0 = 0`` (matches evaluator). User-eq matches
    evaluator. Inequality is ``user_ineq <= 0`` (matches evaluator).
    """
    n = problem.n
    T = problem.T
    eq_dim = problem.eq_dim
    ineq_dim = problem.ineq_dim
    out_eq = np.zeros(n + T * n + (T + 1) * eq_dim, dtype=np.float64)
    out_ineq = np.zeros((T + 1) * ineq_dim, dtype=np.float64) if ineq_dim > 0 else np.zeros(0)

    cursor = 0
    for blk in constraint_blocks:
        sz = blk["size"]
        slice_lam = lam_g[cursor:cursor + sz]
        cursor += sz
        kind = blk["kind"]
        stage = blk["stage"]
        if kind == "init":
            out_eq[:n] = slice_lam
        elif kind == "dyn":
            # dyn defect at stage k: occupies rows
            # ``n + k*n : n + (k+1)*n`` of the eq stack. Sign flip.
            out_eq[n + stage * n: n + (stage + 1) * n] = -slice_lam
        elif kind == "eq":
            base = n + T * n
            out_eq[base + stage * eq_dim: base + stage * eq_dim + sz] = slice_lam
        elif kind == "ineq":
            out_ineq[stage * ineq_dim: stage * ineq_dim + sz] = slice_lam

    return out_eq, out_ineq


class FatropAdapter(SolverAdapter):
    """fatrop (CasADi-Opti backend) on a ``ProblemSpec``."""

    name = "fatrop"

    def __init__(self, max_iter: int = 1000, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def is_available(self) -> tuple[bool, str]:
        try:
            ca = _import_casadi()
        except ImportError as e:
            return False, f"casadi not importable: {e}"
        ok, reason = _fatrop_plugin_available(ca)
        if not ok:
            return False, reason
        return True, ""

    def solve(self, problem: ProblemSpec) -> SolverResult:  # noqa: PLR0912, PLR0915
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name, problem.name, f"unavailable: {reason}",
            )

        if problem.theta_dim > 0:
            return make_failure_result(
                self.name, problem.name,
                f"fatrop does not natively support cross-stage Theta "
                f"(theta_dim={problem.theta_dim})",
            )

        if "casadi_builder" not in problem.metadata:
            return make_failure_result(
                self.name, problem.name,
                "problem has no metadata['casadi_builder']",
            )

        T, n, m = problem.T, problem.n, problem.m

        try:
            opti, x_vars, u_vars, _, _, _, constraint_blocks = _build_opti(
                problem, max_iter=self.max_iter, tol=self.tol,
            )
        except Exception as e:  # noqa: BLE001
            return make_failure_result(
                self.name, problem.name,
                f"build failed: {type(e).__name__}: {e}",
            )

        # Per-problem warm start: linspace(x0, goal) when the problem
        # has an affine terminal-equality goal, the shipped X_init
        # otherwise. U is always the shipped one. See `_initial_iterate`
        # and the module docstring.
        xs_ws, us_ws = _initial_iterate(problem)
        for k in range(T + 1):
            opti.set_initial(x_vars[k], xs_ws[k])
        for k in range(T):
            opti.set_initial(u_vars[k], us_ws[k])

        # Per-iter recorder: opti.callback gets called every fatrop
        # outer iter and we can read the live (x_vars, u_vars) via
        # opti.debug.value. We append the (X, U) snapshots for
        # post-process history extraction (outside the timed window).
        iter_xu = []
        def _iter_cb(_i):
            try:
                Xi = np.zeros((T + 1, n), dtype=np.float64)
                Ui = np.zeros((T, m), dtype=np.float64)
                for k in range(T + 1):
                    Xi[k] = np.asarray(
                        opti.debug.value(x_vars[k]), dtype=np.float64,
                    ).reshape(-1)
                for k in range(T):
                    Ui[k] = np.asarray(
                        opti.debug.value(u_vars[k]), dtype=np.float64,
                    ).reshape(-1)
                iter_xu.append((Xi, Ui, np.zeros(0)))
            except Exception:  # noqa: BLE001
                # CasADi can throw if the iterate hasn't been initialized
                # yet; just skip that record.
                pass
        opti.callback(_iter_cb)

        notes_pieces: list[str] = []
        xs = np.zeros((T + 1, n), dtype=np.float64)
        us = np.zeros((T, m), dtype=np.float64)
        lam_g_final = np.zeros(0)
        start = timer()
        try:
            sol = opti.solve()
            solve_time_ms = 1e3 * (timer() - start)
            stats = opti.stats()
            for k in range(T + 1):
                xs[k] = np.asarray(sol.value(x_vars[k]), dtype=np.float64).reshape(-1)
            for k in range(T):
                us[k] = np.asarray(sol.value(u_vars[k]), dtype=np.float64).reshape(-1)
            try:
                lam_g_final = np.asarray(sol.value(opti.lam_g), dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001
                lam_g_final = np.zeros(0)
        except Exception as e:  # noqa: BLE001
            solve_time_ms = 1e3 * (timer() - start)
            # opti.stats() is only valid after a successful solve; fall
            # back to opti.debug for the iterate values and to a
            # synthetic stats dict (success=False).
            try:
                stats = opti.stats()
            except Exception:  # noqa: BLE001
                stats = {"iter_count": -1, "success": False, "return_status": "error"}
            try:
                for k in range(T + 1):
                    xs[k] = np.asarray(
                        opti.debug.value(x_vars[k]), dtype=np.float64,
                    ).reshape(-1)
                for k in range(T):
                    us[k] = np.asarray(
                        opti.debug.value(u_vars[k]), dtype=np.float64,
                    ).reshape(-1)
            except Exception:  # noqa: BLE001
                pass
            try:
                lam_g_final = np.asarray(opti.debug.value(opti.lam_g), dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001
                lam_g_final = np.zeros(0)
            notes_pieces.append(f"{type(e).__name__}: {str(e).splitlines()[0]}")

        Theta = np.asarray(problem.Theta_init, dtype=np.float64)

        # Multiplier remap.
        if lam_g_final.size > 0:
            multipliers_eq, multipliers_ineq = _remap_fatrop_lam_g(
                lam_g_final, constraint_blocks, problem,
            )
        else:
            multipliers_eq = np.zeros(n + T * n + (T + 1) * problem.eq_dim)
            multipliers_ineq = (
                np.zeros((T + 1) * problem.ineq_dim) if problem.ineq_dim > 0 else np.zeros(0)
            )

        iters = int(stats.get("iter_count", -1))
        success = bool(stats.get("success", False))
        return_status = stats.get("return_status", "")
        notes = "; ".join([*notes_pieces, str(return_status)]).strip("; ")

        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=xs, U=us, Theta=Theta,
            iterations=iters,
            solve_time_ms=solve_time_ms,
            success=success,
            notes=notes,
            multipliers_eq=multipliers_eq,
            multipliers_ineq=multipliers_ineq if problem.ineq_dim > 0 else None,
            iterates_xut=iter_xu or None,
        )


@register("fatrop")
def _factory(**kwargs) -> SolverAdapter:
    return FatropAdapter(**kwargs)
