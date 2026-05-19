"""Shared CasADi ``Callback`` wrapper that delegates to a single ``jax.jit``'d
function across stages.

Background
----------
The MJX adapters (``ipopt_mjx_sparse.py``, ``fatrop_mjx.py``, and any
future MJX-capable CasADi adapter) need per-stage CasADi callbacks
because CasADi's NLP infrastructure walks the symbolic graph stage by
stage. A naive implementation closes over a *different* ``jax.jit``'d
function per stage; JAX then has to trace and compile O(T) functions,
which on longer-horizon MJX problems dominates the warm-up time.

The trick this module encapsulates: build ONE ``jax.jit``'d function
for the whole shape and pass the stage index ``t`` as a runtime
``jnp.int32`` argument. JAX traces and compiles ``(x_dtype, u_dtype,
int32)`` exactly once and reuses the compiled code for every stage.
Total JIT cost is O(1) rather than O(T).

Two callback layouts
--------------------
``PerStageJaxCallback`` exposes a single forward callback. Depending on
whether the caller supplies ``adj_hess_jit_fn``, the wrapper builds one
of two layouts:

* **Forward-only** (``adj_hess_jit_fn=None``): just ``_Cb`` + a child
  ``_JacCb``. Used by IPOPT in ``hessian_approximation='limited-memory'``
  mode, which never asks for a Lagrangian Hessian. JacCb is built
  WITHOUT ``enable_fd`` because no second-derivative path is exercised.

* **With reverse** (``adj_hess_jit_fn`` supplied): ``_Cb`` exposes
  ``has_reverse`` / ``get_reverse`` so CasADi can construct the
  Lagrangian Hessian via reverse-of-Jacobian. Adds a ``_RevCb`` with
  its own ``_RevJacCb`` (the third-order callback CasADi may ask for).
  Required by fatrop, whose IPM step uses an exact Lagrangian Hessian.
  JacCb is built WITH ``enable_fd`` as a safety net for the case where
  CasADi falls back to needing the JacCb's own derivative.

The two layouts are built by ``_build_forward_only`` and
``_build_with_reverse``; ``__init__`` dispatches between them.

Lifetime note: CasADi keeps only weak references to its ``Callback``
objects. Every nested callback (forward, Jacobian, reverse, reverse-
Jacobian) is pinned in ``self._held`` so the Python wrappers outlive
the surrounding solve.
"""

from __future__ import annotations

import numpy as np


def _import_casadi():
    import casadi as ca

    return ca


class PerStageJaxCallback:
    """One CasADi ``Callback`` that delegates to a shared JIT'd JAX function.

    Parameters
    ----------
    name
        Unique CasADi-callback name (passed to ``construct``).
    t
        Stage index. Passed to ``eval_jit_fn`` / ``jac_jit_fn`` /
        ``adj_hess_jit_fn`` as a runtime ``jnp.int32`` argument so the
        same compiled JAX function is reused across stages.
    in_dim
        Length of the (single, vectorized) input ``z``.
    out_dim
        Length of the (single, vectorized) output ``y``.
    eval_jit_fn
        ``eval_jit_fn(z, t_int32) -> y`` (length-``out_dim``).
    jac_jit_fn
        ``jac_jit_fn(z, t_int32) -> J`` of shape ``(out_dim, in_dim)``.
    adj_hess_jit_fn
        ``adj_hess_jit_fn(z, seed, t_int32) -> H_adj`` of shape
        ``(in_dim, in_dim)``. Only required when the calling solver
        needs a second derivative through CasADi's reverse-mode path
        (fatrop). Pass ``None`` for solvers that don't (IPOPT in
        L-BFGS mode); the wrapper picks the forward-only layout in
        that case.

    Attributes
    ----------
    The constructed callback is exposed via ``self.cb()``; the wrapper
    instance also pins every nested callback in ``self._held`` so the
    Python objects outlive CasADi's weak references.
    """

    def __init__(
        self,
        name: str,
        t: int,
        in_dim: int,
        out_dim: int,
        eval_jit_fn,
        jac_jit_fn,
        adj_hess_jit_fn=None,
    ):
        import jax.numpy as jnp

        self._name = name
        self._t = t
        self._t_jnp = jnp.int32(t)
        self._eval_jit = eval_jit_fn
        self._jac_jit = jac_jit_fn
        self._adj_hess_jit = adj_hess_jit_fn
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._held: list = []
        self._jac_cb = None
        self._rev_cb = None

        if adj_hess_jit_fn is None:
            self._cb = self._build_forward_only(name, in_dim, out_dim)
        else:
            self._cb = self._build_with_reverse(name, in_dim, out_dim)
        self._held.append(self._cb)

    def cb(self):
        return self._cb

    # ------------------------------------------------------------------
    # Layout 1: forward-only (no Lagrangian Hessian via this callback).
    # ------------------------------------------------------------------
    def _build_forward_only(self, name: str, in_dim: int, out_dim: int):
        ca = _import_casadi()
        wrapper = self

        # JacCb is built WITHOUT enable_fd: no second-derivative path is
        # exercised in this layout, so CasADi never needs a fallback
        # derivative of the JacCb.
        jac_opts: dict = {}

        class _Cb(ca.Callback):
            def __init__(cb_self):  # noqa: N804
                ca.Callback.__init__(cb_self)
                cb_self.construct(name, {})

            def get_n_in(cb_self): return 1  # noqa: N805
            def get_n_out(cb_self): return 1  # noqa: N805
            def get_sparsity_in(cb_self, _i): return ca.Sparsity.dense(in_dim, 1)  # noqa: N805
            def get_sparsity_out(cb_self, _i): return ca.Sparsity.dense(out_dim, 1)  # noqa: N805

            def eval(cb_self, args):  # noqa: N805
                z = np.asarray(args[0]).reshape(-1)
                y = np.asarray(
                    wrapper._eval_jit(z, wrapper._t_jnp),
                ).reshape(out_dim, 1)
                return [y]

            def has_jacobian(cb_self): return True  # noqa: N805

            def get_jacobian(cb_self, name_unused, inames, onames, opts):  # noqa: N805, ARG002
                if wrapper._jac_cb is not None:
                    return wrapper._jac_cb
                jc = _build_jac_cb(wrapper, name, in_dim, out_dim, jac_opts)
                wrapper._jac_cb = jc
                wrapper._held.append(jc)
                return jc

        return _Cb()

    # ------------------------------------------------------------------
    # Layout 2: forward + reverse + reverse-Jacobian (fatrop's path).
    # ------------------------------------------------------------------
    def _build_with_reverse(self, name: str, in_dim: int, out_dim: int):
        ca = _import_casadi()
        wrapper = self

        # JacCb is built WITH enable_fd as a safety net for the case
        # where CasADi falls back to needing the JacCb's own derivative
        # (which the reverse-mode path does not directly request, but
        # has been observed during certain Lagrangian-Hessian builds).
        jac_opts = {"enable_fd": True}

        class _Cb(ca.Callback):
            def __init__(cb_self):  # noqa: N804
                ca.Callback.__init__(cb_self)
                cb_self.construct(name, {})

            def get_n_in(cb_self): return 1  # noqa: N805
            def get_n_out(cb_self): return 1  # noqa: N805
            def get_sparsity_in(cb_self, _i): return ca.Sparsity.dense(in_dim, 1)  # noqa: N805
            def get_sparsity_out(cb_self, _i): return ca.Sparsity.dense(out_dim, 1)  # noqa: N805

            def eval(cb_self, args):  # noqa: N805
                z = np.asarray(args[0]).reshape(-1)
                y = np.asarray(
                    wrapper._eval_jit(z, wrapper._t_jnp),
                ).reshape(out_dim, 1)
                return [y]

            def has_jacobian(cb_self): return True  # noqa: N805

            def get_jacobian(cb_self, name_unused, inames, onames, opts):  # noqa: N805, ARG002
                if wrapper._jac_cb is not None:
                    return wrapper._jac_cb
                jc = _build_jac_cb(wrapper, name, in_dim, out_dim, jac_opts)
                wrapper._jac_cb = jc
                wrapper._held.append(jc)
                return jc

            def has_reverse(cb_self, nadj): return nadj == 1  # noqa: N805, ARG002

            def get_reverse(cb_self, nadj, name_unused, inames, onames, opts):  # noqa: N805, ARG002
                if wrapper._rev_cb is not None and nadj == 1:
                    return wrapper._rev_cb
                rc = _build_rev_cb(wrapper, name, in_dim, out_dim)
                if nadj == 1:
                    wrapper._rev_cb = rc
                wrapper._held.append(rc)
                return rc

        return _Cb()


def _build_jac_cb(wrapper, name: str, in_dim: int, out_dim: int, jac_opts: dict):
    """Forward Jacobian callback: J(z) = ∂y/∂z."""
    ca = _import_casadi()
    jac_jit = wrapper._jac_jit
    t_jnp = wrapper._t_jnp

    class _JacCb(ca.Callback):
        def __init__(jac_self):  # noqa: N804
            ca.Callback.__init__(jac_self)
            # The Jacobian-callback itself does NOT need a
            # ``has_jacobian`` because CasADi prefers the
            # ``get_reverse`` route (which is what fatrop's
            # Lagrangian-Hessian construction triggers).
            jac_self.construct(name + "_jac", jac_opts)

        def get_n_in(jac_self): return 2  # noqa: N805
        def get_n_out(jac_self): return 1  # noqa: N805
        def get_sparsity_in(jac_self, i):  # noqa: N805
            if i == 0: return ca.Sparsity.dense(in_dim, 1)
            return ca.Sparsity.dense(out_dim, 1)
        def get_sparsity_out(jac_self, _i):  # noqa: N805
            return ca.Sparsity.dense(out_dim, in_dim)

        def eval(jac_self, args):  # noqa: N805
            z = np.asarray(args[0]).reshape(-1)
            J = np.asarray(jac_jit(z, t_jnp)).reshape(out_dim, in_dim)
            return [J]

    return _JacCb()


def _build_rev_cb(wrapper, name: str, in_dim: int, out_dim: int):
    """Reverse-mode (adjoint) callback for the parent _Cb.

    Inputs (CasADi reverse-mode convention):
      0: z (the original input, length in_dim)
      1: y_unused (the original output, length out_dim) — we don't
         need it because our adjoint Hessian is computed directly
         from z.
      2: seed (output adjoint, length out_dim)

    Output:
      0: input adjoint = seed^T * J(z), length in_dim.

    Crucially, this callback ALSO provides its Jacobian (via
    ``get_jacobian``) so that CasADi can ask for the second derivative
    — which is what fatrop needs for the Lagrangian Hessian. The
    Jacobian wrt z of the input-adjoint is the adjoint Hessian
    ``Σ_i seed_i * ∂²f_i/∂z²``, computed directly via
    ``adj_hess_jit_fn`` (a JAX double-jacrev).
    """
    ca = _import_casadi()
    jac_jit = wrapper._jac_jit
    t_jnp = wrapper._t_jnp

    class _RevCb(ca.Callback):
        def __init__(rev_self):  # noqa: N804
            ca.Callback.__init__(rev_self)
            rev_self.construct(name + "_rev", {})

        def get_n_in(rev_self): return 3  # noqa: N805
        def get_n_out(rev_self): return 1  # noqa: N805

        def get_sparsity_in(rev_self, i):  # noqa: N805
            if i == 0: return ca.Sparsity.dense(in_dim, 1)
            if i == 1: return ca.Sparsity.dense(out_dim, 1)
            return ca.Sparsity.dense(out_dim, 1)

        def get_sparsity_out(rev_self, _i):  # noqa: N805
            return ca.Sparsity.dense(in_dim, 1)

        def eval(rev_self, args):  # noqa: N805
            z = np.asarray(args[0]).reshape(-1)
            seed = np.asarray(args[2]).reshape(-1)
            # adjoint = seed^T J(z) -- a length-in_dim row vector.
            J = np.asarray(jac_jit(z, t_jnp)).reshape(out_dim, in_dim)
            adj = (seed @ J).reshape(in_dim, 1)
            return [adj]

        def has_jacobian(rev_self): return True  # noqa: N805

        def get_jacobian(rev_self, name_unused, inames, onames, opts):  # noqa: N805, ARG002
            jc = _build_rev_jac_cb(wrapper, name, in_dim, out_dim)
            wrapper._held.append(jc)
            return jc

    return _RevCb()


def _build_rev_jac_cb(wrapper, name: str, in_dim: int, out_dim: int):
    """Jacobian of the reverse-mode callback.

    CasADi's Jacobian-of-Function convention: for a function with
    ``n_in`` inputs and ``n_out`` outputs, its Jacobian is itself a
    function with ``n_in + n_out`` inputs (the originals + the
    "nominal output" placeholders) and ``n_in * n_out`` outputs (one
    Jacobian block per (output, input) pair), named
    ``jac_o<out_idx>_i<in_idx>``. Since our parent (_RevCb) has
    n_in=3 and n_out=1, this Jacobian has 4 inputs and 3 outputs.
    """
    ca = _import_casadi()
    adj_hess_jit = wrapper._adj_hess_jit
    jac_jit = wrapper._jac_jit
    t_jnp = wrapper._t_jnp

    class _RevJacCb(ca.Callback):
        def __init__(jac_self):  # noqa: N804
            ca.Callback.__init__(jac_self)
            # Third-order derivatives are not asked by fatrop.
            # FD is a safety net.
            jac_self.construct(name + "_revjac", {"enable_fd": True})

        def get_n_in(jac_self): return 4  # noqa: N805
        def get_n_out(jac_self): return 3  # noqa: N805
        def get_sparsity_in(jac_self, i):  # noqa: N805
            if i == 0: return ca.Sparsity.dense(in_dim, 1)    # z
            if i == 1: return ca.Sparsity.dense(out_dim, 1)   # y
            if i == 2: return ca.Sparsity.dense(out_dim, 1)   # seed
            return ca.Sparsity.dense(in_dim, 1)               # adj (rev's nominal out)
        def get_sparsity_out(jac_self, i):  # noqa: N805
            if i == 0: return ca.Sparsity.dense(in_dim, in_dim)    # d adj / d z
            if i == 1: return ca.Sparsity.dense(in_dim, out_dim)   # d adj / d y
            return ca.Sparsity.dense(in_dim, out_dim)              # d adj / d seed

        def eval(jac_self, args):  # noqa: N805
            z = np.asarray(args[0]).reshape(-1)
            seed = np.asarray(args[2]).reshape(-1)
            # d adj / d z = adj Hessian (in_dim, in_dim).
            H = np.asarray(
                adj_hess_jit(z, seed, t_jnp),
            ).reshape(in_dim, in_dim)
            # d adj / d y = 0  (in_dim, out_dim).
            zero_y = np.zeros((in_dim, out_dim), dtype=np.float64)
            # d adj / d seed = J(z)^T  (in_dim, out_dim).
            Jz = np.asarray(jac_jit(z, t_jnp)).reshape(out_dim, in_dim)
            Jt = Jz.T
            return [H, zero_y, Jt]

    return _RevJacCb()
