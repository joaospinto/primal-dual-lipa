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

The CasADi backend supports ``hessian_mode="cost"`` only.

Hessian modes:

* ``"cost"`` (default): eigen-clamp ``hessian(f)``, take ``triu``.
  Drops second-order constraint terms (Gauss-Newton-flavoured).
* ``"lagrangian"``: include ``y^T hess(c) + z^T hess(g)``, symmetrize,
  PSD-clamp, ``triu``.

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


def _build_jax_nlp(problem: ProblemSpec):
    """Build flat-NLP cost + constraint callables from a ``ProblemSpec``.

    Returns ``(f, c, g, x_dim, y_dim, s_dim, T, n, m, td)``:

    * ``f(z)``  — scalar total cost
    * ``c(z)``  — equality residual vector of length ``y_dim``
    * ``g(z)``  — inequality residual vector of length ``s_dim``,
                   convention ``g <= 0``

    ``z`` is laid out as ``[vec(X) (n*(T+1)), vec(U) (m*T), Theta (td)]``;
    when ``problem.theta_dim == 0`` the Theta block is empty and ``z``
    reduces to ``[vec(X), vec(U)]``.

    All three are JAX functions (un-jitted; the caller jits them after
    composition with PSD-projection / autodiff).
    """
    jax, jnp = _import_jax()

    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    x_dim = nx + nu + td

    cost_fn = problem.cost
    dyn_fn = problem.dynamics
    eq_fn = problem.equalities
    ineq_fn = problem.inequalities

    def split(z):
        X = z[:nx].reshape(T + 1, n)
        U = z[nx:nx + nu].reshape(T, m)
        Theta = z[nx + nu:nx + nu + td]  # length td (possibly 0)
        return X, U, Theta

    def f(z):
        X, U, Theta = split(z)
        # Pad U with a zero row so the per-stage cost can be evaluated at t=T.
        U_pad = jnp.concatenate([U, jnp.zeros((1, m), dtype=U.dtype)], axis=0)
        ts = jnp.arange(T + 1)
        costs = jax.vmap(cost_fn, in_axes=(0, 0, None, 0))(X, U_pad, Theta, ts)
        return jnp.sum(costs)

    x0_const = jnp.asarray(problem.x0)

    def c(z):
        X, U, Theta = split(z)
        # Initial-state defect
        init_defect = X[0] - x0_const  # (n,)
        # Dynamics defects: x_{t+1} - dyn(x_t, u_t, theta, t)
        dyn_next = jax.vmap(dyn_fn, in_axes=(0, 0, None, 0))(
            X[:-1], U, Theta, jnp.arange(T),
        )
        dyn_defects = (X[1:] - dyn_next).reshape(-1)  # (T*n,)
        pieces = [init_defect, dyn_defects]
        if eq_fn is not None and problem.eq_dim > 0:
            U_pad = jnp.concatenate([U, jnp.zeros((1, m), dtype=U.dtype)], axis=0)
            ts = jnp.arange(T + 1)
            eq_full = jax.vmap(eq_fn, in_axes=(0, 0, None, 0))(
                X, U_pad, Theta, ts,
            ).reshape(-1)  # ((T+1)*eq_dim,)
            pieces.append(eq_full)
        return jnp.concatenate(pieces)

    def g(z):
        if ineq_fn is None or problem.ineq_dim == 0:
            return jnp.zeros((0,), dtype=z.dtype)
        X, U, Theta = split(z)
        U_pad = jnp.concatenate([U, jnp.zeros((1, m), dtype=U.dtype)], axis=0)
        ts = jnp.arange(T + 1)
        ineq_full = jax.vmap(ineq_fn, in_axes=(0, 0, None, 0))(
            X, U_pad, Theta, ts,
        ).reshape(-1)
        return ineq_full

    y_dim = n + T * n + ((T + 1) * problem.eq_dim if eq_fn is not None else 0)
    s_dim = (T + 1) * problem.ineq_dim if ineq_fn is not None else 0

    # Per-stage cost callable (with explicit theta+t args) so the caller
    # can build a stage-decomposed Hessian without going through the
    # global f(z). Stage cost shape: (n,) state, (m,) control (a zero
    # padding at t=T for problems where the terminal cost has no u
    # dependency), (td,) theta, scalar t.
    return f, c, g, x_dim, y_dim, s_dim, T, n, m, td, cost_fn


def _filter_zero_rows(c_fn, c_dim: int, x_dim: int, mock_zs: list):
    """Detect rows of c that are identically zero across all probe inputs.

    Used to drop trivially-zero equality rows (some problems pad
    inactive stages with zeros). Such rows generate spurious dual
    variables and inflate the KKT system; filtering them out leaves
    only the constraints with actual content. Returns
    ``(keep_mask, c_filtered)`` where ``c_filtered(z) = c(z)[keep_mask]``.

    Probe inputs should be a list of dense ``z`` vectors; rows are
    kept if any probe produces a non-zero entry.
    """
    import jax  # noqa: F401
    import jax.numpy as jnp

    if c_dim == 0:
        return np.ones(0, dtype=bool), c_fn

    keep = np.zeros(c_dim, dtype=bool)
    c_jit = jax.jit(c_fn)
    for z in mock_zs:
        cz = np.asarray(c_jit(jnp.asarray(z)), dtype=np.float64)
        keep = keep | (np.abs(cz) > 0.0)
    # Also probe the Jacobian at the warm start — a row could happen
    # to evaluate to 0 at a probe but have a non-zero derivative
    # elsewhere (e.g. nonlinear constraint that only "lights up" at
    # certain x). Take the union with rows that have non-zero
    # gradient at any probe.
    jac_jit = jax.jit(jax.jacrev(c_fn))
    for z in mock_zs:
        Jz = np.asarray(jac_jit(jnp.asarray(z)), dtype=np.float64)
        keep = keep | (np.abs(Jz).sum(axis=1) > 0.0)

    keep_indices = np.where(keep)[0]

    def c_filtered(z):
        return c_fn(z)[jnp.asarray(keep_indices)]

    return keep, c_filtered


def _proj_psd_factory(reg_delta: float):
    """Return a JAX-traceable function that projects a Hessian to PSD.

    ``reg_delta`` controls the floor on the smallest eigenvalue; the
    sip_python tests use ``1e-6`` and we mirror that as the default.

    Used by the JAX backend's ``hessian_mode='lagrangian'`` path (which
    still does a single global ``eigh`` because the Lagrangian Hessian
    is dense in the multipliers and not naturally stage-decomposable).
    The default ``hessian_mode='cost'`` path uses the per-stage block
    PSD-projection helpers below — same algorithm as the CasADi backend
    in this file and the ``sip-mjx`` adapter, much cheaper than a
    global eigh on the full Hessian.
    """
    _, jnp = _import_jax()

    def proj_psd(Q):
        S, _V = jnp.linalg.eigh(Q)
        k = -jnp.minimum(jnp.min(S), 0.0) + reg_delta
        return Q + k * jnp.eye(Q.shape[0])

    return proj_psd


def _build_per_stage_cost_hess(problem: ProblemSpec, cost_fn, x_dim: int,
                               psd_reg_delta: float):
    """Build per-stage cost-Hessian machinery for the JAX backend's cost mode.

    The OCP cost decomposes as ``f(z) = sum_t cost_t(x_t, u_t, theta, t)``,
    so the cost-only Hessian ``∇²f(z)`` is block-diagonal in stages on
    ``(x_t, u_t, theta)`` (the theta block sums across stages when ``td > 0``).

    This helper mirrors the CasADi backend's per-stage Hessian path:

    * Compute ``(n+m+td)`` per-stage Hessian blocks for ``t in [0, T)``
      and a ``(n+td)`` terminal block for ``t = T`` (no ``u_T`` in z).
    * PSD-clamp each block independently via ``eigh`` + eigenvalue floor.
    * Scatter-add upper-triangle entries into a global sparse buffer.

    Returns a dict with:

    * ``inner_hess_fn(z)``      — jit'd JAX function returning a
                                  ``(T, n+m+td, n+m+td)`` array of
                                  PSD-clamped per-stage Hessians.
    * ``terminal_hess_fn(z)``   — jit'd JAX function returning the
                                  ``(n+td, n+td)`` PSD-clamped terminal
                                  Hessian.
    * ``upp_hess_pat``          — ``scipy.sparse.csc_matrix`` (boolean
                                  cast to float64) covering the union of
                                  per-stage block upper-triangle entries
                                  ∪ the diagonal of every block's z-index
                                  scope (the eigen-clamp may add a k*I
                                  bump to every diagonal entry).
    * ``stage_writebacks``      — list of dicts (one per stage) for
                                  per-call scatter-add into
                                  ``upp_hess_buf.data``. Each entry has
                                  ``src_i, src_j, data_idx`` arrays.
                                  Stage 0..T-1 are inner; stage T is
                                  terminal. Index conventions mirror
                                  ``_build_casadi_nlp``.

    The vmap-over-T-stages path keeps the per-iter cost low even on
    long horizons: each inner per-stage block is (n+m+td)x(n+m+td)
    and the terminal block is (n+td)x(n+td) — a single fused eigh
    kernel call instead of T+1 separate ones.
    """
    jax, jnp = _import_jax()

    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    inner_block = n + m + td
    terminal_block = n + td

    # --- Per-stage z-index helpers (global flat-z layout: vec(X), vec(U), Theta) ---
    def _inner_idx(t: int) -> np.ndarray:
        idx = list(range(t * n, (t + 1) * n))
        idx.extend(range(nx + t * m, nx + (t + 1) * m))
        if td > 0:
            idx.extend(range(nx + nu, nx + nu + td))
        return np.asarray(idx, dtype=np.intp)

    def _terminal_idx() -> np.ndarray:
        idx = list(range(T * n, (T + 1) * n))
        if td > 0:
            idx.extend(range(nx + nu, nx + nu + td))
        return np.asarray(idx, dtype=np.intp)

    inner_z_idx = np.stack([_inner_idx(t) for t in range(T)], axis=0) if T > 0 \
        else np.zeros((0, inner_block), dtype=np.intp)
    terminal_z_idx = _terminal_idx()

    # --- Per-stage Hessian functions ---
    # cost_fn signature: cost_fn(x, u, theta, t) -> scalar.
    # For the per-stage Hessian we differentiate wrt the concatenated
    # block input. ``theta`` is shared across stages but each block has
    # its own slice of the global Hessian; the scatter-add at the end
    # sums theta-theta contributions across stages.

    def _inner_stage_cost(z_slice, t):
        # z_slice = [x_t (n,), u_t (m,), theta (td,)]
        x_t = z_slice[:n]
        u_t = z_slice[n:n + m]
        theta = z_slice[n + m:n + m + td]
        return cost_fn(x_t, u_t, theta, t)

    def _terminal_stage_cost(z_slice):
        # z_slice = [x_T (n,), theta (td,)]
        x_T = z_slice[:n]
        theta = z_slice[n:n + td]
        # Terminal stage: cost_fn(x_T, u_zero, theta, T). u is unused at
        # t=T (the cost branches via jnp.where(t == T, ...) in every
        # analytical problem here), so feed zeros.
        u_zero = jnp.zeros(m, dtype=z_slice.dtype)
        return cost_fn(x_T, u_zero, theta, T)

    # PSD-clamp helper (per-block).
    def _proj_block(H):
        H = 0.5 * (H + H.T)
        S, _V = jnp.linalg.eigh(H)
        k = -jnp.minimum(jnp.min(S), 0.0) + psd_reg_delta
        return H + k * jnp.eye(H.shape[0])

    # Per-inner-stage Hessian (vmap'd across t in [0, T)).
    def _inner_hess_one(z_slice, t):
        H = jax.hessian(_inner_stage_cost, argnums=0)(z_slice, t)
        return _proj_block(H)

    def inner_hess_fn(z):
        # Gather per-stage z-slices via fancy indexing on the flat z.
        # inner_z_idx is (T, n+m+td); z[idx] -> (T, n+m+td).
        z_slices = z[jnp.asarray(inner_z_idx)]
        ts = jnp.arange(T)
        return jax.vmap(_inner_hess_one, in_axes=(0, 0))(z_slices, ts)

    def terminal_hess_fn(z):
        z_slice = z[jnp.asarray(terminal_z_idx)]
        H = jax.hessian(_terminal_stage_cost)(z_slice)
        return _proj_block(H)

    inner_hess_jit = jax.jit(inner_hess_fn)
    terminal_hess_jit = jax.jit(terminal_hess_fn)

    # --- Probe per-block structural sparsity ---
    # Conservatively-tightest per-block pattern: for each stage probe
    # the dense Hessian at a few perturbed z-slices (warm start +
    # randomized) and take the union of nonzero entries. Analogous to
    # CasADi's symbolic structural-sparsity analysis: capture every
    # entry that could light up during the solve, but don't include
    # block entries that are mathematically zero everywhere.
    #
    # We also OR in the block diagonal — the PSD-clamp can add a k*I
    # bump that touches every diagonal entry within a block's z-index
    # scope, even if some diagonal positions are zero in the raw
    # Hessian.
    rng = np.random.default_rng(0)
    z_init_probe = np.concatenate([
        np.asarray(problem.X_init, dtype=np.float64).reshape(-1),
        np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
        np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1),
    ])
    # Perturbed probes; bias the theta slot to a nonzero value (Theta_init
    # may default to 0, masking entries that depend nonlinearly on theta).
    probe_zs = [z_init_probe]
    for noise in (0.01, 0.1):
        zp = z_init_probe + noise * rng.standard_normal(z_init_probe.shape)
        if td > 0:
            zp[-td:] = 0.1
        probe_zs.append(zp)
    if td > 0:
        z_theta = z_init_probe.copy()
        z_theta[-td:] = 0.1
        probe_zs.append(z_theta)

    # Per-stage block pattern, indexed by t (or "T" for terminal).
    inner_unproj_jit = jax.jit(jax.vmap(
        lambda zs, t: jax.hessian(_inner_stage_cost, argnums=0)(zs, t),
        in_axes=(0, 0),
    ))
    terminal_unproj_jit = jax.jit(
        jax.hessian(_terminal_stage_cost),
    )

    inner_block_pat = np.zeros((T, inner_block, inner_block), dtype=bool)
    terminal_block_pat = np.zeros((terminal_block, terminal_block), dtype=bool)
    ts_arr = jnp.arange(T)
    for zp in probe_zs:
        zp_j = jnp.asarray(zp)
        inner_z_idx_j = jnp.asarray(inner_z_idx)
        z_slices_inner = zp_j[inner_z_idx_j]
        inner_H = np.asarray(inner_unproj_jit(z_slices_inner, ts_arr))
        inner_block_pat |= (np.abs(inner_H) > 0.0)
        z_slice_terminal = zp_j[jnp.asarray(terminal_z_idx)]
        terminal_H = np.asarray(terminal_unproj_jit(z_slice_terminal))
        terminal_block_pat |= (np.abs(terminal_H) > 0.0)
    # OR with the block diagonal (PSD-clamp may add k*I).
    for t in range(T):
        np.fill_diagonal(inner_block_pat[t], True)
    np.fill_diagonal(terminal_block_pat, True)

    # --- Build upper-triangle sparsity pattern ---
    # Scatter per-stage block patterns into the global upper triangle.
    # Theta-theta entries get OR'd across stages (when td > 0).
    from scipy import sparse as sp
    upp_hess_pat = sp.lil_matrix((x_dim, x_dim), dtype=bool)
    for t in range(T):
        idx = inner_z_idx[t]
        bs = inner_block
        blk = inner_block_pat[t]
        for i in range(bs):
            gi = int(idx[i])
            for j in range(bs):
                if not blk[i, j]:
                    continue
                gj = int(idx[j])
                if gi <= gj:
                    upp_hess_pat[gi, gj] = True
                else:
                    upp_hess_pat[gj, gi] = True
    # Terminal stage.
    idx = terminal_z_idx
    bs = terminal_block
    blk = terminal_block_pat
    for i in range(bs):
        gi = int(idx[i])
        for j in range(bs):
            if not blk[i, j]:
                continue
            gj = int(idx[j])
            if gi <= gj:
                upp_hess_pat[gi, gj] = True
            else:
                upp_hess_pat[gj, gi] = True
    upp_hess_pat = upp_hess_pat.tocsc().astype(np.float64)
    # Force structural nnz to be != 0 (CSC stores symbolic zeros if we
    # built from a bool matrix that already had False entries) — cast
    # back through pat != 0 to drop any spurious False stores.
    upp_hess_pat.eliminate_zeros()
    upp_hess_pat = (upp_hess_pat != 0).astype(np.float64).tocsc()

    # --- Per-stage write-back plan into upp_hess_buf.data ---
    # Build the (gi, gj) -> data-slot dictionary for the global buffer
    # once; then for each stage record (src_i, src_j, data_idx) arrays.
    buf_indptr = upp_hess_pat.indptr
    buf_indices = upp_hess_pat.indices
    gij_to_data_idx = {}
    for col in range(upp_hess_pat.shape[1]):
        for k in range(buf_indptr[col], buf_indptr[col + 1]):
            row = buf_indices[k]
            gij_to_data_idx[(row, col)] = k

    stage_writebacks = []
    for t in range(T):
        idx = inner_z_idx[t]
        bs = inner_block
        src_i, src_j, data_idx = [], [], []
        for i in range(bs):
            gi = int(idx[i])
            for j in range(bs):
                gj = int(idx[j])
                if gi > gj:
                    continue
                pos = gij_to_data_idx.get((gi, gj))
                if pos is None:
                    continue
                src_i.append(i); src_j.append(j); data_idx.append(pos)
        stage_writebacks.append({
            "kind": "inner",
            "t": t,
            "src_i": np.asarray(src_i, dtype=np.intp),
            "src_j": np.asarray(src_j, dtype=np.intp),
            "data_idx": np.asarray(data_idx, dtype=np.intp),
        })
    # Terminal stage.
    idx = terminal_z_idx
    bs = terminal_block
    src_i, src_j, data_idx = [], [], []
    for i in range(bs):
        gi = int(idx[i])
        for j in range(bs):
            gj = int(idx[j])
            if gi > gj:
                continue
            pos = gij_to_data_idx.get((gi, gj))
            if pos is None:
                continue
            src_i.append(i); src_j.append(j); data_idx.append(pos)
    stage_writebacks.append({
        "kind": "terminal",
        "t": T,
        "src_i": np.asarray(src_i, dtype=np.intp),
        "src_j": np.asarray(src_j, dtype=np.intp),
        "data_idx": np.asarray(data_idx, dtype=np.intp),
    })

    return {
        "inner_hess_fn": inner_hess_jit,
        "terminal_hess_fn": terminal_hess_jit,
        "upp_hess_pat": upp_hess_pat,
        "stage_writebacks": stage_writebacks,
    }


def _slice_iterate(z_val: np.ndarray, problem: ProblemSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m, T, td = problem.n, problem.m, problem.T, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    X = z_val[:nx].reshape(T + 1, n)
    U = z_val[nx:nx + nu].reshape(T, m)
    Theta = z_val[nx + nu:nx + nu + td] if td > 0 else np.zeros(0, dtype=z_val.dtype)
    return X, U, Theta


def _import_casadi():
    import casadi as ca  # local import so a missing CasADi only fails this backend

    return ca


def _build_casadi_nlp(problem: ProblemSpec):
    """Build the flat NLP via the problem's ``casadi_builder``.

    Mirrors ``_build_jax_nlp`` but everything happens symbolically in
    CasADi SX. Returns a dict bundling:

    * ``f_fn(z)``       — scalar cost ``casadi.Function``
    * ``c_fn(z)``       — equality residual ``casadi.Function``
                          (init defect + dynamics defects + non-None
                          eq rows). Already filtered: stages where
                          ``casadi_builder`` returns ``eq=None`` are
                          omitted entirely (analogous to the JAX
                          backend's ``_filter_zero_rows`` step).
    * ``g_fn(z)``       — inequality residual ``casadi.Function``,
                          convention ``g <= 0`` (only stages where
                          ``casadi_builder`` returns a non-None
                          ``ineq``).
    * ``grad_f_fn(z)``  — gradient of f, dense (n_z,) output.
    * ``jac_c_fn(z)``   — Jacobian of c, CasADi-sparse.
    * ``jac_g_fn(z)``   — Jacobian of g, CasADi-sparse.
    * ``stage_hess_meta`` — list of per-stage cost-Hessian descriptors.
                          Each entry is a dict ``{"fn", "z_idx",
                          "block_size", "sparsity"}`` where ``fn`` is a
                          ``casadi.Function`` taking the block's
                          flattened input ``[x_t; (u_t if t < T);
                          theta]`` and returning the dense (small)
                          per-stage Hessian wrt that vector. ``z_idx``
                          gives the indices in the global ``z`` for
                          writing the PSD-projected block back to the
                          upper-triangle sparse buffer; ``sparsity`` is
                          the CasADi symbolic sparsity of the block
                          (used to size the global Hessian template).
    * ``y_dim``, ``s_dim``, ``x_dim`` — dimensions.
    * ``jac_c_sparsity`` / ``jac_g_sparsity`` — symbolic CasADi
                          sparsity patterns (used to build the
                          ``scipy.sparse`` template buffers).
    * ``upp_hess_pat`` — (x_dim, x_dim) sparse boolean upper-triangle
                          pattern: union of per-stage block symbolic
                          nonzeros ∪ the diagonal of every block's
                          z-index scope. Used to size the sparse
                          Hessian template handed to sip_python.

    The decision vector layout matches the JAX path exactly:
    ``z = [vec(X) (n*(T+1)), vec(U) (m*T), Theta (td)]``.

    Per-stage cost Hessian extraction is the key optimisation that
    lets the CasADi backend scale to large x_dim: the global Hessian
    is block-diagonal (plus an arrow from the cross-stage ``theta``
    if any), so the eigen-clamp factorises across stage blocks of
    size at most (n+m+theta_dim) x (n+m+theta_dim) instead of doing
    a single O(x_dim^3) eigh.
    """
    from scipy import sparse as sp

    ca = _import_casadi()
    casadi_builder = problem.metadata["casadi_builder"]

    T, n, m, td = problem.T, problem.n, problem.m, problem.theta_dim
    nx = n * (T + 1)
    nu = m * T
    x_dim = nx + nu + td

    X_sx = ca.SX.sym("X", n, T + 1)
    U_sx = ca.SX.sym("U", m, T)
    Theta_sx = ca.SX.sym("Theta", max(td, 1))  # CasADi can't build a 0-length sym; slice it off below

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
    # Per-stage Hessian metadata: list of (stage_hess_fn, idx_array)
    # where idx_array is the array of z-indices for the stage's inputs
    # (X[:, t] then U[:, t] then optional Theta). At terminal stages
    # U is omitted (no u arg).
    stage_hess_meta = []

    # Initial-state defect (always present).
    c_pieces.append(X_sx[:, 0] - ca.DM(np.asarray(problem.x0)))

    # Helper to compute z-indices for a stage's inputs.
    def _stage_indices(t: int, include_u: bool) -> np.ndarray:
        # X[:, t] occupies z[t*n : (t+1)*n]
        idx = list(range(t * n, (t + 1) * n))
        if include_u and t < T:
            # U[:, t] occupies z[nx + t*m : nx + (t+1)*m]
            idx.extend(range(nx + t * m, nx + (t + 1) * m))
        if td > 0:
            idx.extend(range(nx + nu, nx + nu + td))
        return np.asarray(idx, dtype=np.intp)

    for t in range(T + 1):
        # Per-stage cost: re-build the stage in isolated SX symbols so we
        # can take its Hessian w.r.t. just (x_t, u_t, theta) without the
        # global graph. This is the "split symbolic" trick that keeps the
        # per-stage Hessian Function small.
        include_u = (t < T)
        x_local = ca.SX.sym(f"xs_{t}", n)
        u_local = ca.SX.sym(f"us_{t}", m) if include_u else ca.SX.zeros(m, 1)
        theta_local = ca.SX.sym(f"th_{t}", max(td, 1))
        theta_local_arg = theta_local[:td] if td > 0 else ca.SX.zeros(0)
        stage_local = casadi_builder(x_local, u_local, theta_local_arg, t)
        f_local = stage_local["f"]
        # Hessian variables for this stage block. Order: x_t, [u_t,] theta.
        # Matches _stage_indices() ordering above.
        hess_vars = [x_local]
        if include_u:
            hess_vars.append(u_local)
        if td > 0:
            hess_vars.append(theta_local_arg)
        xu_local = ca.vertcat(*hess_vars)
        H_local, _ = ca.hessian(f_local, xu_local)
        # Build the per-stage Hessian Function. Input is the same
        # variables; output is the (block_size, block_size) dense matrix.
        # We want the Function to take a single flat input matching the
        # global z slice — this lets per-call dispatch be a single
        # ``np.ndarray`` slice + Function call rather than a build-up.
        fn_name = f"sip_stage_hess_{t}"
        # Use the same xu_local as the Function input.
        stage_hess_fn = ca.Function(fn_name, [xu_local], [H_local])
        stage_hess_meta.append({
            "fn": stage_hess_fn,
            "z_idx": _stage_indices(t, include_u),
            "block_size": xu_local.numel(),
            "sparsity": H_local.sparsity(),
        })
        # ---- Global-graph pieces (cost, c, g, jacobians) ---------------------
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

    f_fn = ca.Function("sip_cost_val", [z], [f_total], ["z"], ["f"])
    c_fn = ca.Function("sip_eq_val", [z], [c_expr], ["z"], ["c"])
    g_fn = ca.Function("sip_ineq_val", [z], [g_expr], ["z"], ["g"])
    grad_f_fn = ca.Function("sip_cost_grad", [z], [grad_f], ["z"], ["grad_f"])
    jac_c_fn = ca.Function("sip_eq_jac", [z], [jac_c], ["z"], ["jac_c"])
    jac_g_fn = ca.Function("sip_ineq_jac", [z], [jac_g], ["z"], ["jac_g"])

    # Build the (x_dim, x_dim) upper-triangle sparsity pattern as the
    # union of (a) each per-stage block's symbolic sparsity, mapped from
    # block-local (i, j) to global (z_idx[i], z_idx[j]), and (b) the
    # diagonal (the eigen-clamp may add k*I to every block, which lights
    # up every diagonal entry within a block's z_idx scope — strictly
    # speaking we only need diagonal entries that some stage covers, but
    # adding the full diagonal is cheap and avoids edge cases).
    #
    # Using the symbolic per-stage nnz (vs the full block) keeps the
    # global Hessian template close to truly minimal.
    upp_hess_pat = sp.lil_matrix((x_dim, x_dim), dtype=bool)
    for meta in stage_hess_meta:
        idx = meta["z_idx"]
        sp_block = meta["sparsity"]
        # CasADi sparsity: iterate (row, col) of structurally non-zero
        # entries. Use sparsity.row() / sparsity.colind() (CSC).
        col_ind = np.asarray(sp_block.colind(), dtype=np.intp)
        row_ind = np.asarray(sp_block.row(), dtype=np.intp)
        for j in range(sp_block.size2()):  # local cols
            for k in range(col_ind[j], col_ind[j + 1]):
                i = row_ind[k]  # local row
                gi = int(idx[i])
                gj = int(idx[j])
                if gi <= gj:  # upper triangle
                    upp_hess_pat[gi, gj] = True
                else:
                    upp_hess_pat[gj, gi] = True  # mirror to upper triangle
        # Also mark the diagonal entries the block can populate via
        # k*I clamp (every variable in z_idx may get a k*I bump).
        for i in idx:
            upp_hess_pat[i, i] = True
    upp_hess_pat = upp_hess_pat.tocsc()
    upp_hess_pat = (upp_hess_pat.astype(np.float64) != 0)

    return {
        "f_fn": f_fn,
        "c_fn": c_fn,
        "g_fn": g_fn,
        "grad_f_fn": grad_f_fn,
        "jac_c_fn": jac_c_fn,
        "jac_g_fn": jac_g_fn,
        "stage_hess_meta": stage_hess_meta,
        "upp_hess_pat": upp_hess_pat,
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
        # PSD-projection floor for the cost Hessian. Matches sip_python's
        # test default. Larger values trade convergence speed for
        # robustness on hard nonconvex sub-problems.
        psd_reg_delta: float = 1e-6,
        # Elastic mode lets the solver soften inequalities by absorbing
        # infeasibility into an L1-penalised slack. Off by default: the
        # dynamics defects in our OCPs are stiff equalities that elastic
        # relaxation would silently corrupt.
        enable_elastics: bool = False,
        elastic_var_cost_coeff: float = 1e6,
        # Default penalty / barrier ramp. Per-problem overrides via
        # problem.metadata["sip_settings"] dial in different schedules
        # for problems whose defaults don't converge.
        penalty_parameter_increase_factor: float = 2.0,
        mu_update_factor: float = 0.9,
        initial_mu: float = 1e-1,
        initial_penalty_parameter: float = 1.0,
        # Hessian mode: "lagrangian" (full y^T hess(c) + z^T hess(g) +
        # hess(f), PSD-projected) or "cost" (just hess(f) PSD-projected,
        # matching the sip_python test convention).
        hessian_mode: str = "cost",
        print_logs: bool = False,
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
        self.psd_reg_delta = float(psd_reg_delta)
        self.enable_elastics = bool(enable_elastics)
        self.elastic_var_cost_coeff = float(elastic_var_cost_coeff)
        self.penalty_parameter_increase_factor = float(penalty_parameter_increase_factor)
        self.mu_update_factor = float(mu_update_factor)
        self.initial_mu = float(initial_mu)
        self.initial_penalty_parameter = float(initial_penalty_parameter)
        self.hessian_mode = str(hessian_mode)
        if self.hessian_mode not in ("lagrangian", "cost"):
            raise ValueError(
                f"hessian_mode must be 'lagrangian' or 'cost', got {hessian_mode!r}",
            )
        self.print_logs = bool(print_logs)
        self.sip_extra_settings = sip_extra_settings or {}
        if backend not in {"jax", "casadi"}:
            raise ValueError(f"backend must be 'jax' or 'casadi', got {backend!r}")
        self.backend = backend
        if backend == "casadi" and self.hessian_mode != "cost":
            # The CasADi backend currently only implements the cost-only
            # Hessian PSD-projection path.
            raise ValueError(
                "backend='casadi' currently only supports hessian_mode='cost'; "
                f"got hessian_mode={self.hessian_mode!r}",
            )

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
        avail, reason = self.is_available()
        if not avail:
            return make_failure_result(
                self.name, problem.name, f"unavailable: {reason}",
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
                    self.name, problem.name,
                    "backend='casadi' requires casadi_builder in problem.metadata "
                    f"(problem={problem.name!r} doesn't ship one — re-run with backend='jax')",
                )
            return self._solve_casadi(problem)

        sip = _import_sip()
        jax, jnp = _import_jax()
        from scipy import sparse as sp

        # --- Build cost + constraint callables --------------------------------
        f, c_full, g_full, x_dim, y_dim_full, s_dim_full, T, n, m, td, cost_fn = (
            _build_jax_nlp(problem)
        )
        proj_psd = _proj_psd_factory(self.psd_reg_delta)

        # Filter out structurally-zero equality rows. Some problems use
        # ``jnp.where(t == T, ..., zeros)`` to zero-pad inactive stages;
        # sip would otherwise allocate dual variables and KKT-rows for
        # them, inflating the linear system and confusing AMD reordering.
        # Probe at the warm-start plus perturbed mock-z to catch rows
        # that are zero at one iterate but non-zero elsewhere.
        z_init = np.concatenate([
            np.asarray(problem.X_init, dtype=np.float64).reshape(-1),
            np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
            np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1),
        ])
        rng = np.random.default_rng(0)
        probe_zs = [
            z_init,
            z_init + 0.01 * rng.standard_normal(z_init.shape),
            z_init + 0.1 * rng.standard_normal(z_init.shape),
        ]
        # For problems with theta_dim > 0, also probe at a z where the
        # Theta block is explicitly non-zero so we don't miss Hessian
        # entries that depend nonlinearly on theta when ``Theta_init``
        # happens to be zero.
        if td > 0:
            z_theta_probe = z_init.copy()
            z_theta_probe[-td:] = 0.1
            probe_zs.append(z_theta_probe)
        c_keep_mask, c = _filter_zero_rows(c_full, y_dim_full, x_dim, probe_zs)
        g_keep_mask, g = _filter_zero_rows(g_full, s_dim_full, x_dim, probe_zs)
        y_dim = int(c_keep_mask.sum())
        s_dim = int(g_keep_mask.sum())

        # --- Compose autodiff for gradient / Jacobians / Hessian -------------
        f_jit = jax.jit(f)
        c_jit = jax.jit(c)
        g_jit = jax.jit(g)
        grad_f_jit = jax.jit(jax.grad(f))
        # reverse-mode Jacobians: c_dim, s_dim are O(n*T), x_dim is also
        # O(n*T), so neither direction dominates — pick reverse for code
        # simplicity (matches the sip_python tests).
        jac_c_jit = jax.jit(jax.jacrev(c))
        if s_dim > 0:
            jac_g_jit = jax.jit(jax.jacrev(g))
        else:
            jac_g_jit = None

        # Lagrangian-Hessian builder. Two modes:
        #
        #   * "cost"       — H = ∇²f(z), PSD-clamped per stage block,
        #                    scatter-add'd into a sparse upper-triangle
        #                    buffer. The OCP cost is stage-separable
        #                    (f = sum_t cost_t), so the cost Hessian is
        #                    block-diagonal on ``(x_t, u_t, theta)`` and
        #                    each (n+m+td)x(n+m+td) inner block plus the
        #                    (n+td)x(n+td) terminal block can be
        #                    PSD-clamped independently — much cheaper
        #                    than a global eigh.
        #
        #   * "lagrangian" — H = ∇²[f + y·c + z·g], PSD-clamped, triu.
        #                    Global eigh because the Lagrangian Hessian
        #                    is dense in the multipliers and not
        #                    naturally stage-decomposable.
        if self.hessian_mode == "cost":
            per_stage = _build_per_stage_cost_hess(
                problem, cost_fn, x_dim, self.psd_reg_delta,
            )
            inner_hess_jit = per_stage["inner_hess_fn"]
            terminal_hess_jit = per_stage["terminal_hess_fn"]
            upp_hess_pat_per_stage = per_stage["upp_hess_pat"]
            stage_writebacks = per_stage["stage_writebacks"]
            upp_hess_jit = None  # not used; stage-block path takes over
        else:  # "lagrangian"
            def lagrangian(z, y, zd):
                terms = [f(z), jnp.dot(y, c(z))]
                if s_dim > 0:
                    terms.append(jnp.dot(zd, g(z)))
                return sum(terms[1:], terms[0])
            hess_lag = jax.hessian(lagrangian, argnums=0)
            def upp_hess_with_dual(z, y, zd):
                # Symmetrize before PSD clamp — autodiff Hessian is
                # mathematically symmetric but float64 can introduce tiny
                # asymmetries that the eigh routine doesn't tolerate.
                H = hess_lag(z, y, zd)
                H = 0.5 * (H + H.T)
                return jnp.triu(proj_psd(H))
            upp_hess_jit = jax.jit(upp_hess_with_dual)
            inner_hess_jit = None
            terminal_hess_jit = None
            upp_hess_pat_per_stage = None
            stage_writebacks = None

        # --- Determine sparsity pattern at a mock z ---------------------------
        # An all-ones probe is too symmetric for quadratic-around-goal
        # cost: diagonal entries dominate and off-diagonal coupling
        # drops out unless we perturb. Mix in a small per-component
        # perturbation so generic-zero cancellations don't artificially
        # shrink the pattern.
        # Bias mock_z toward the warm-start so the sparsity pattern of
        # ``c(mock_z)`` (which depends on dynamics linearisations) and
        # ``hessian_lag(mock_z, mock_y, mock_zd)`` resemble what the
        # solver will see at later iterates. ``z_init`` and ``rng`` are
        # already defined above where we filtered zero constraint rows.
        mock_z = z_init + 0.01 * rng.standard_normal(z_init.shape)
        # For problems with theta_dim > 0, force the Theta block of
        # mock_z to a non-zero value. ``Theta_init`` may default to
        # zero, and small Gaussian perturbations may leave it small
        # enough that Hessian entries whose coupling to (X, U) is
        # multiplied by theta would collapse and be missed
        # by the sparsity probe.
        if td > 0:
            mock_z[-td:] = 0.1
        # Mock multipliers — non-zero so every constraint contributes
        # to the Lagrangian-Hessian sparsity pattern (otherwise we'd
        # under-allocate when the solver activates a constraint that
        # was zero at construction).
        mock_y = rng.standard_normal(y_dim) * 0.1
        mock_zd = np.abs(rng.standard_normal(s_dim)) + 0.1

        jac_c_dense = np.asarray(jac_c_jit(jnp.asarray(mock_z)), dtype=np.float64)
        # Hessian-pattern probe: only probe the dense Hessian when in
        # global lagrangian mode (cost mode uses the per-stage symbolic
        # block-diagonal pattern instead — same approach as the CasADi
        # backend / sip-mjx adapter).
        if self.hessian_mode != "cost":
            upp_hess_dense = np.asarray(
                upp_hess_jit(jnp.asarray(mock_z), jnp.asarray(mock_y), jnp.asarray(mock_zd)),
                dtype=np.float64,
            )

        # Also OR with the warm-start z's pattern to be safe.
        z_init_for_pat = z_init.copy()
        if td > 0:
            # Likewise probe with a non-trivial Theta value, for the
            # same reason as above.
            z_init_for_pat[-td:] = 0.1
        jac_c_dense_ws = np.asarray(jac_c_jit(jnp.asarray(z_init_for_pat)), dtype=np.float64)
        if self.hessian_mode != "cost":
            upp_hess_dense_ws = np.asarray(
                upp_hess_jit(jnp.asarray(z_init_for_pat), jnp.asarray(mock_y), jnp.asarray(mock_zd)),
                dtype=np.float64,
            )
        jac_c_pat = (np.abs(jac_c_dense) > 0.0) | (np.abs(jac_c_dense_ws) > 0.0)
        if self.hessian_mode == "cost":
            # Use the per-stage block-diagonal symbolic pattern built
            # earlier; no dense probe needed.
            upp_hess_template = upp_hess_pat_per_stage
            upp_hess_pat = (upp_hess_template.toarray() != 0)
        else:
            upp_hess_pat = (np.abs(upp_hess_dense) > 0.0) | (np.abs(upp_hess_dense_ws) > 0.0)

        if s_dim > 0:
            jac_g_dense = np.asarray(jac_g_jit(jnp.asarray(mock_z)), dtype=np.float64)
            jac_g_dense_ws = np.asarray(jac_g_jit(jnp.asarray(z_init_for_pat)), dtype=np.float64)
            jac_g_pat = (np.abs(jac_g_dense) > 0.0) | (np.abs(jac_g_dense_ws) > 0.0)
            # Additional large-perturbation probes. The two small probes
            # above land on measure-zero hyperplanes of the user's
            # inequality function for entries that pass through clip /
            # where / min / max primitives (their gradients are exactly
            # zero on the saturated branch). Without these probes, such
            # entries are dropped from the sparsity pattern and sip then
            # silently drops their values during the solve, yielding a
            # structurally-wrong Newton system.
            rng_probe = np.random.default_rng(12345)
            for _ in range(15):
                zp = z_init + 3.0 * rng_probe.standard_normal(z_init.shape)
                if td > 0:
                    zp[-td:] = 1.0 * rng_probe.standard_normal(td)
                jg_probe = np.asarray(
                    jac_g_jit(jnp.asarray(zp)), dtype=np.float64,
                )
                jac_g_pat = jac_g_pat | (np.abs(jg_probe) > 0.0)
        else:
            jac_g_dense = np.zeros((0, x_dim), dtype=np.float64)
            jac_g_pat = np.zeros((0, x_dim), dtype=bool)

        # Build the sparse-matrix templates. Per the sip_python convention
        # (see test_simple_constrained_lqr.py): jacobians as CSR of shape
        # (out_dim, x_dim) — the C++ side reinterprets CSR row-major data
        # as CSC of the transpose, so we set is_jacobian_*_transposed=True
        # in ProblemDimensions.
        jac_c_template = sp.csr_matrix(jac_c_pat.astype(np.float64))
        jac_g_template = sp.csr_matrix(jac_g_pat.astype(np.float64))
        # Hessian: CSC of upper triangle, shape (x_dim, x_dim). In cost
        # mode this was already built above from the per-stage symbolic
        # pattern; in lagrangian mode build it from the dense probe.
        if self.hessian_mode != "cost":
            upp_hess_template = sp.csc_matrix(upp_hess_pat.astype(np.float64))

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
        # AMD-first KKT permutation (falls back to RCM when cvxopt is
        # unavailable). On Linux sip_python's bundled helper otherwise
        # uses RCM, which gives orders-of-magnitude more L-factor fill
        # on OCP-shaped KKT matrices.
        from tests.comparison.sip_kkt_perm import compute_kkt_perm_inv_and_nnzs
        _perm_result = compute_kkt_perm_inv_and_nnzs(
            upp_hess_template, jac_c_template, jac_g_template,
        )
        qs.kkt_pinv = _perm_result.perm_inv
        pd.kkt_nnz = _perm_result.kkt_nnz
        pd.kkt_L_nnz = _perm_result.L_nnz

        # --- Settings --------------------------------------------------------
        # Per-problem override hook: `problem.metadata["sip_settings"]`
        # may be a dict of Settings field overrides. Mirrors the
        # `lipa_settings` hook in the LIPA adapter and lets a problem
        # ship its own tuned parameters when the adapter's defaults
        # don't suit its constraint structure.
        problem_overrides = problem.metadata.get("sip_settings", {})
        ss = sip.Settings()
        ss.max_iterations = int(self.max_iter)
        ss.max_kkt_violation = float(self.tol)
        ss.enable_elastics = self.enable_elastics
        if self.enable_elastics:
            ss.elastic_var_cost_coeff = self.elastic_var_cost_coeff
        ss.penalty_parameter_increase_factor = self.penalty_parameter_increase_factor
        ss.mu_update_factor = self.mu_update_factor
        ss.initial_mu = self.initial_mu
        ss.initial_penalty_parameter = self.initial_penalty_parameter
        ss.print_logs = self.print_logs
        # When print_logs is off, ALL the sub-loggers must be off too —
        # the solver's check_settings refuses any combination where a
        # sub-logger is on while the master switch is off (it returns
        # Status.FAILED_CHECK with 0 iterations). Their defaults are all
        # True, so we have to explicitly clear them. When print_logs is
        # on, leave the sub-flags alone so the user's settings overrides
        # in sip_extra_settings can selectively re-enable them.
        if not self.print_logs:
            ss.print_line_search_logs = False
            ss.print_search_direction_logs = False
            ss.print_derivative_check_logs = False
        # Don't fail-fast on internal asserts — let the solver iterate.
        ss.assert_checks_pass = False
        # Apply per-problem overrides BEFORE caller-provided extras so
        # the latter always wins.
        for k, v in problem_overrides.items():
            setattr(ss, k, v)
        for k, v in self.sip_extra_settings.items():
            setattr(ss, k, v)

        # --- Per-iter recording ---------------------------------------------
        # SIP's model callback ``mc`` is invoked once per outer iter
        # with the live x-vector (decision variables). We append
        # (X, U, Theta) snapshots into ``iter_xut`` and convert them
        # to (cost, eq, ineq) histories via histories_from_iterates
        # AFTER the timed solve closes — the per-iter recording itself
        # is a single np.copy + slice, negligible inside the timed
        # window.
        iter_xut: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # --- Build the model callback ---------------------------------------
        # Buffers we reuse and mutate in-place each callback (CSR/CSC
        # objects whose .data we overwrite). We bind them to the closure.
        jac_c_buf = jac_c_template.copy()
        jac_g_buf = jac_g_template.copy()
        upp_hess_buf = upp_hess_template.copy()

        # Pre-compute (row, col) indices for each nonzero so we can
        # extract ``.data`` from the dense Jacobian/Hessian via fancy
        # indexing without allocating. The order returned by
        # ``.nonzero()`` matches the format's internal ordering (CSR
        # row-major, CSC column-major), so fancy-indexed values land in
        # ``.data`` in the right slots.
        jac_c_rows, jac_c_cols = jac_c_buf.nonzero()
        jac_g_rows, jac_g_cols = jac_g_buf.nonzero()
        if self.hessian_mode != "cost":
            upp_hess_rows, upp_hess_cols = upp_hess_buf.nonzero()
        else:
            upp_hess_rows = None
            upp_hess_cols = None

        is_cost_mode = self.hessian_mode == "cost"

        def mc(mci):
            mco = sip.ModelCallbackOutput()
            z_jax = jnp.asarray(mci.x, dtype=jnp.float64)
            # Per-iter snapshot — copy out the iterate as (X, U, Theta)
            # for post-process history. This is the *only* extra work
            # the recorder does inside the timed window; it's a single
            # contiguous np.copy of the decision vector.
            z_np_iter = np.asarray(mci.x, dtype=np.float64).copy()
            Xi, Ui, Ti = _slice_iterate(z_np_iter, problem)
            iter_xut.append((Xi, Ui, Ti))
            mco.f = float(np.asarray(f_jit(z_jax)))
            mco.c = np.asarray(c_jit(z_jax), dtype=np.float64).copy()
            if s_dim > 0:
                mco.g = np.asarray(g_jit(z_jax), dtype=np.float64).copy()
            else:
                mco.g = np.zeros(0, dtype=np.float64)
            mco.gradient_f = np.asarray(grad_f_jit(z_jax), dtype=np.float64).copy()

            jc = np.asarray(jac_c_jit(z_jax), dtype=np.float64)
            jac_c_buf.data[:] = jc[jac_c_rows, jac_c_cols]
            mco.jacobian_c = jac_c_buf

            if s_dim > 0:
                jg = np.asarray(jac_g_jit(z_jax), dtype=np.float64)
                jac_g_buf.data[:] = jg[jac_g_rows, jac_g_cols]
                mco.jacobian_g = jac_g_buf
            else:
                mco.jacobian_g = jac_g_buf

            if is_cost_mode:
                # Per-stage cost Hessian + per-block PSD-projection,
                # scatter-added into the upper-triangle sparse buffer.
                # Same algorithm as the CasADi backend / sip-mjx adapter.
                # For problems with td > 0 (cross-stage theta), the
                # theta-rows/cols of every stage's block overlap on the
                # global theta indices — np.add.at sums them, matching
                # what a global cost Hessian would produce before
                # PSD-projection (per-block PSD-projection vs global is
                # the standard SQP-with-stage-decomposed-Hessian
                # trade-off; harmless because the analytical OCPs here
                # have purely stage-local cost dependencies on theta).
                inner_blocks = np.asarray(
                    inner_hess_jit(z_jax), dtype=np.float64,
                )
                terminal_block = np.asarray(
                    terminal_hess_jit(z_jax), dtype=np.float64,
                )
                upp_hess_buf.data[:] = 0.0
                for wb in stage_writebacks:
                    if wb["kind"] == "inner":
                        H_block = inner_blocks[wb["t"]]
                    else:
                        H_block = terminal_block
                    if H_block.size == 0:
                        continue
                    vals = H_block[wb["src_i"], wb["src_j"]]
                    np.add.at(upp_hess_buf.data, wb["data_idx"], vals)
                mco.upper_hessian_lagrangian = upp_hess_buf
            else:
                y_jax = jnp.asarray(mci.y, dtype=jnp.float64)
                zd_jax = jnp.asarray(mci.z, dtype=jnp.float64) if s_dim > 0 \
                    else jnp.empty(0, dtype=jnp.float64)
                uh = np.asarray(upp_hess_jit(z_jax, y_jax, zd_jax), dtype=np.float64)
                upp_hess_buf.data[:] = uh[upp_hess_rows, upp_hess_cols]
                mco.upper_hessian_lagrangian = upp_hess_buf

            return mco

        # --- Construct the solver --------------------------------------------
        solver = sip.Solver(ss, qs, pd, mc)

        # --- Initial Variables ----------------------------------------------
        vars_in = sip.Variables(pd)
        vars_in.x[:] = z_init
        vars_in.s[:] = 1.0
        vars_in.y[:] = 0.0
        vars_in.e[:] = 0.0
        vars_in.z[:] = 1.0

        # --- Warm-up call so JIT compile doesn't pollute the timing ---------
        # We call each JIT'd function once on z_init. Cheap relative to a
        # full solve and matches the convention used by acados / aligator
        # / csqp adapters.
        try:
            z_warm = jnp.asarray(z_init, dtype=jnp.float64)
            jax.block_until_ready(f_jit(z_warm))
            jax.block_until_ready(c_jit(z_warm))
            if s_dim > 0:
                jax.block_until_ready(g_jit(z_warm))
            jax.block_until_ready(grad_f_jit(z_warm))
            jax.block_until_ready(jac_c_jit(z_warm))
            if jac_g_jit is not None:
                jax.block_until_ready(jac_g_jit(z_warm))
            if is_cost_mode:
                jax.block_until_ready(inner_hess_jit(z_warm))
                jax.block_until_ready(terminal_hess_jit(z_warm))
            else:
                y_warm = jnp.zeros(y_dim, dtype=jnp.float64)
                zd_warm = jnp.ones(s_dim, dtype=jnp.float64) if s_dim > 0 \
                    else jnp.empty(0, dtype=jnp.float64)
                jax.block_until_ready(upp_hess_jit(z_warm, y_warm, zd_warm))
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

        # Multiplier extraction. SIP's vars_in.y has shape (y_dim,)
        # which is the FILTERED equality stack (rows where
        # c_keep_mask was True). vars_in.z has shape (s_dim,) which
        # is the FILTERED inequality stack. Splice them back into the
        # full per-stage stacks expected by evaluate_problem so the
        # filtered-out rows (always zero in the JAX evaluator) get 0
        # multipliers — the corresponding stationarity contribution
        # then vanishes as it should.
        eq_full_size = (
            problem.n + problem.T * problem.n
            + (problem.T + 1) * problem.eq_dim
        )
        ineq_full_size = (
            (problem.T + 1) * problem.ineq_dim
            if problem.inequalities is not None else 0
        )
        multipliers_eq_full = np.zeros(eq_full_size, dtype=np.float64)
        multipliers_ineq_full = (
            np.zeros(ineq_full_size, dtype=np.float64)
            if ineq_full_size > 0 else np.zeros(0)
        )
        try:
            y_filtered = np.asarray(vars_in.y, dtype=np.float64).reshape(-1)
            z_filtered = np.asarray(vars_in.z, dtype=np.float64).reshape(-1)
            # c_keep_mask was built earlier in this function. Splice
            # y_filtered into the full eq stack at positions where
            # c_keep_mask is True. Same for ineq.
            keep_idx_eq = np.where(c_keep_mask)[0]
            keep_idx_ineq = np.where(g_keep_mask)[0]
            # SIP's _build_jax_nlp.c() encodes the dyn defect as
            # ``X[t+1] - dyn(x_t, u_t)`` (opposite sign from
            # evaluate_problem's ``dyn(...) - X[t+1]``); init defect
            # and user equalities match. We splice y_filtered into
            # the full eq stack, then sign-flip just the dyn-defect
            # rows. Inequality block matches (g <= 0 with z >= 0 in
            # both conventions).
            if y_filtered.size == keep_idx_eq.size:
                multipliers_eq_full[keep_idx_eq] = y_filtered
                # Sign-flip the dyn-defect rows (positions
                # [problem.n : problem.n + problem.T * problem.n]).
                n_p = problem.n
                T_p = problem.T
                multipliers_eq_full[n_p:n_p + T_p * n_p] = (
                    -multipliers_eq_full[n_p:n_p + T_p * n_p]
                )
            if z_filtered.size == keep_idx_ineq.size and ineq_full_size > 0:
                multipliers_ineq_full[keep_idx_ineq] = z_filtered
        except Exception:  # noqa: BLE001
            pass

        multipliers_ineq_out = (
            multipliers_ineq_full if ineq_full_size > 0 else None
        )

        if output is None:
            return pack_solver_result(
                solver_name=self.name,
                problem_name=problem.name,
                problem=problem,
                X=X, U=U, Theta=Theta,
                iterations=0,
                solve_time_ms=solve_time_ms,
                success=False,
                notes=err,
                multipliers_eq=multipliers_eq_full,
                multipliers_ineq=multipliers_ineq_out,
                iterates_xut=iter_xut or None,
            )

        status = output.exit_status
        success = status == sip.Status.SOLVED
        # SUBOPTIMAL = converged constraint-wise but couldn't tighten KKT
        # below the tolerance; treat as a soft failure for the runner.
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X, U=U, Theta=Theta,
            iterations=int(output.num_iterations),
            solve_time_ms=solve_time_ms,
            success=success,
            notes=str(status),
            multipliers_eq=multipliers_eq_full,
            multipliers_ineq=multipliers_ineq_out,
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

        Sparsity is taken directly from CasADi's symbolic-graph
        analysis (no mock-eval probe), and the Jacobian / Hessian
        templates are built from the CasADi sparsity patterns. The
        cost-only Hessian is materialised dense per-call for the
        eigen-clamp PSD projection — same as the JAX path.

        Caller is responsible for verifying
        ``problem.metadata["casadi_builder"]`` is present and the
        Hessian mode is "cost"; the constructor + ``solve`` enforce both.
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
        stage_hess_meta = nlp["stage_hess_meta"]
        upp_hess_pat_dense = nlp["upp_hess_pat"]
        x_dim = nlp["x_dim"]
        y_dim = nlp["y_dim"]
        s_dim = nlp["s_dim"]
        jac_c_sparsity = nlp["jac_c_sparsity"]
        jac_g_sparsity = nlp["jac_g_sparsity"]

        # --- Build initial guess ---------------------------------------------
        z_init = np.concatenate([
            np.asarray(problem.X_init, dtype=np.float64).reshape(-1),
            np.asarray(problem.U_init, dtype=np.float64).reshape(-1),
            np.asarray(problem.Theta_init, dtype=np.float64).reshape(-1),
        ])
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
                (np.ones(nnz, dtype=np.float64),
                 np.asarray(sparsity.row(), dtype=np.intp),
                 np.asarray(sparsity.colind(), dtype=np.intp)),
                shape=(out_dim, x_dim),
            )
            csr_template = csc_pat.tocsr()
            # Build the permutation: idx_csc has 0..nnz-1 as its data, in
            # CasADi's CSC order; idx_csr.data is then perm[i] = which
            # CasADi nz index ends up at CSR position i.
            idx_csc = sp.csc_matrix(
                (np.arange(nnz, dtype=np.intp),
                 np.asarray(sparsity.row(), dtype=np.intp),
                 np.asarray(sparsity.colind(), dtype=np.intp)),
                shape=(out_dim, x_dim),
            )
            perm = np.asarray(idx_csc.tocsr().data, dtype=np.intp)
            return csr_template, perm

        jac_c_template, jac_c_perm = _build_jac_template(jac_c_sparsity, y_dim)
        jac_g_template, jac_g_perm = _build_jac_template(jac_g_sparsity, s_dim)

        # Hessian: CSC of upper triangle, shape (x_dim, x_dim). The sparse
        # template comes from the union of per-stage block symbolic
        # nonzeros (built in ``_build_casadi_nlp`` as ``upp_hess_pat``).
        # Each per-call PSD-projected stage block is written into the
        # corresponding slot via fancy indexing.
        upp_hess_template = upp_hess_pat_dense.astype(np.float64).tocsc()

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
            upp_hess_template, jac_c_template, jac_g_template,
        )
        qs.kkt_pinv = _perm_result.perm_inv
        pd.kkt_nnz = _perm_result.kkt_nnz
        pd.kkt_L_nnz = _perm_result.L_nnz

        # --- Settings --------------------------------------------------------
        problem_overrides = problem.metadata.get("sip_settings", {})
        ss = sip.Settings()
        ss.max_iterations = int(self.max_iter)
        ss.max_kkt_violation = float(self.tol)
        ss.enable_elastics = self.enable_elastics
        if self.enable_elastics:
            ss.elastic_var_cost_coeff = self.elastic_var_cost_coeff
        ss.penalty_parameter_increase_factor = self.penalty_parameter_increase_factor
        ss.mu_update_factor = self.mu_update_factor
        ss.initial_mu = self.initial_mu
        ss.initial_penalty_parameter = self.initial_penalty_parameter
        ss.print_logs = self.print_logs
        if not self.print_logs:
            ss.print_line_search_logs = False
            ss.print_search_direction_logs = False
            ss.print_derivative_check_logs = False
        ss.assert_checks_pass = False
        for k, v in problem_overrides.items():
            setattr(ss, k, v)
        for k, v in self.sip_extra_settings.items():
            setattr(ss, k, v)

        # --- Build the model callback ---------------------------------------
        # Reusable buffers, written in-place each callback. We bind them to
        # the closure so SIP sees the same memory each call.
        jac_c_buf = jac_c_template.copy()
        jac_g_buf = jac_g_template.copy()
        upp_hess_buf = upp_hess_template.copy()

        psd_reg_delta = self.psd_reg_delta

        # Pre-compute a write-back plan for each per-stage Hessian block:
        # for each stage, determine which entries of the dense block (i, j
        # ranging over [0, block_size)) land in which slot of
        # ``upp_hess_buf.data``. Computed once; per-call we just iterate
        # the stages and do an in-place assignment.
        #
        # The buffer is CSC; CSC stores .data in column-major order, and
        # column k's entries are at indices [indptr[k], indptr[k+1]) with
        # row indices ``indices[indptr[k]:indptr[k+1]]``. We build, for
        # each stage block, two flattened arrays:
        #
        #   ``block_src_i[k], block_src_j[k]`` — block-local (row, col)
        #     of the k-th nonzero we want to write back (block-local in
        #     [0, block_size)). We always write the upper-triangle entries
        #     (row <= col), summing contributions across stages where
        #     blocks overlap (theta cross-stage entries).
        #   ``buf_data_idx[k]`` — index into ``upp_hess_buf.data`` that
        #     this entry writes to.
        #
        # Since multiple stages can hit the same global (i, j) entry (the
        # theta-theta block in particular), we sum contributions by
        # `np.add.at` per call. Build the index dictionary now to map
        # (global_i, global_j) -> upp_hess_buf.data slot.
        buf_indptr = upp_hess_buf.indptr
        buf_indices = upp_hess_buf.indices
        gij_to_data_idx = {}
        for col in range(upp_hess_buf.shape[1]):
            for k in range(buf_indptr[col], buf_indptr[col + 1]):
                row = buf_indices[k]
                gij_to_data_idx[(row, col)] = k

        # For each stage, build the index arrays.
        stage_writebacks = []
        for meta in stage_hess_meta:
            idx = meta["z_idx"]
            bs = meta["block_size"]
            src_i = []
            src_j = []
            data_idx = []
            for i in range(bs):
                gi = int(idx[i])
                for j in range(bs):
                    gj = int(idx[j])
                    if gi > gj:
                        continue  # only upper triangle
                    pos = gij_to_data_idx.get((gi, gj))
                    if pos is None:
                        # Shouldn't happen — upp_hess_pat covers every
                        # stage-block entry by construction.
                        continue
                    src_i.append(i)
                    src_j.append(j)
                    data_idx.append(pos)
            stage_writebacks.append({
                "fn": meta["fn"],
                "z_idx": idx,
                "block_size": bs,
                "src_i": np.asarray(src_i, dtype=np.intp),
                "src_j": np.asarray(src_j, dtype=np.intp),
                "data_idx": np.asarray(data_idx, dtype=np.intp),
            })

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
            mco.gradient_f = np.asarray(grad_f_fn(z_np), dtype=np.float64).reshape(-1).copy()

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

            # Per-stage cost Hessian + eigen-clamp PSD projection. Each
            # block is small (the per-stage dimension), so the eigh is
            # essentially free; the key benefit vs a single global eigh on
            # the full Hessian is avoiding cubic scaling in the horizon.
            #
            # Subtle: for problems with cross-stage theta (td > 0), the
            # per-stage blocks overlap on the theta dimension(s). We
            # ZERO the .data slots first and then ADD each stage's
            # contribution — this correctly sums the theta-theta blocks
            # across stages (matching what the global Hessian would give
            # before PSD-projection). After per-stage PSD-projection the
            # resulting Hessian differs from a global eigen-clamp; this
            # is the standard local-block-regularization trade-off that
            # SQP-with-stage-decomposed-Hessians takes, and is acceptable
            # because the cost-only mode is already an SQP approximation.
            upp_hess_buf.data[:] = 0.0
            for wb in stage_writebacks:
                z_slice = z_np[wb["z_idx"]]
                H_block = np.asarray(wb["fn"](z_slice), dtype=np.float64)
                if H_block.size == 0:
                    continue
                # PSD-project this block.
                if H_block.shape[0] > 1:
                    S, _V = np.linalg.eigh(H_block)
                    s_min = float(S[0])
                else:
                    s_min = float(H_block[0, 0])
                k = max(-s_min, 0.0) + psd_reg_delta
                if k > 0.0:
                    # Add k*I to the block.
                    bs = wb["block_size"]
                    H_block.flat[::bs + 1] += k
                # Write back upper-triangle entries (sum across stages
                # for overlapping entries — np.add.at, scatter-add).
                vals = H_block[wb["src_i"], wb["src_j"]]
                np.add.at(upp_hess_buf.data, wb["data_idx"], vals)
            mco.upper_hessian_lagrangian = upp_hess_buf

            return mco

        # --- Construct the solver --------------------------------------------
        solver = sip.Solver(ss, qs, pd, mc)

        # --- Initial Variables ----------------------------------------------
        vars_in = sip.Variables(pd)
        vars_in.x[:] = z_init
        vars_in.s[:] = 1.0
        vars_in.y[:] = 0.0
        vars_in.e[:] = 0.0
        vars_in.z[:] = 1.0

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

        # Multipliers — the CasADi backend doesn't filter rows so the
        # mapping is direct (one-to-one with evaluate_problem's
        # constraint stacks). The CasADi NLP encodes the dyn defect
        # as X[t+1] - dyn(...) (same as the JAX backend, opposite
        # from evaluate_problem) so we sign-flip just the dyn-defect
        # rows.
        try:
            multipliers_eq_full = np.asarray(
                vars_in.y, dtype=np.float64,
            ).reshape(-1)
            n_p = problem.n
            T_p = problem.T
            if multipliers_eq_full.size >= n_p + T_p * n_p:
                multipliers_eq_full[n_p:n_p + T_p * n_p] = (
                    -multipliers_eq_full[n_p:n_p + T_p * n_p]
                )
        except Exception:  # noqa: BLE001
            multipliers_eq_full = None
        try:
            multipliers_ineq_full = np.asarray(
                vars_in.z, dtype=np.float64,
            ).reshape(-1)
        except Exception:  # noqa: BLE001
            multipliers_ineq_full = None

        if output is None:
            return pack_solver_result(
                solver_name=self.name,
                problem_name=problem.name,
                problem=problem,
                X=X, U=U, Theta=Theta,
                iterations=0,
                solve_time_ms=solve_time_ms,
                success=False,
                notes=err,
                multipliers_eq=multipliers_eq_full,
                multipliers_ineq=multipliers_ineq_full,
                iterates_xut=iter_xut or None,
            )

        status = output.exit_status
        success = status == sip.Status.SOLVED
        return pack_solver_result(
            solver_name=self.name,
            problem_name=problem.name,
            problem=problem,
            X=X, U=U, Theta=Theta,
            iterations=int(output.num_iterations),
            solve_time_ms=solve_time_ms,
            success=success,
            notes=str(status),
            multipliers_eq=multipliers_eq_full,
            multipliers_ineq=multipliers_ineq_full,
            iterates_xut=iter_xut or None,
        )


@register("sip")
def _factory(**kwargs) -> SolverAdapter:
    return SipAdapter(**kwargs)
