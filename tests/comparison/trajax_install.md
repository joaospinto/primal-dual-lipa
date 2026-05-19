# Trajax (Google) install for the LIPA comparison driver

[Trajax](https://github.com/google/trajax) is Google's JAX-native
trajectory-optimization library. It exposes:

* `trajax.optimizers.ilqr` — unconstrained iLQR (single-shooting).
* `trajax.optimizers.constrained_ilqr` — Augmented-Lagrangian iLQR
  with explicit equality + inequality constraint support
  (single-shooting).
* `trajax.optimizers.cem` / `random_shooting` / `scipy_minimize` —
  sampling-based baselines, not used by this adapter.

Trajax is the closest non-LIPA algorithmic competitor in the comparison
because it is **JAX-native** (no Python ↔ CasADi callback overhead like
ipopt-mjx / fatrop-mjx) and uses **augmented-Lagrangian** constraint
handling (mechanistically similar to LIPA's `η` penalty, and SHOULD
tolerate multi-basin non-convex problems where SQP-style merit
functions stall).

> Warning: the PyPI package named `trajax` is a *different*, unrelated
> MPPI/MPCC library. Always install from the Google git repo
> (`git+https://github.com/google/trajax`), never `pip install trajax`.

## TL;DR — one-line install

```bash
bash tests/comparison/install/install_trajax.sh
```

The script runs

```bash
UV_NO_BUILD_ISOLATION=0 uv pip install \
    --python /home/joapinto/github/primal-dual-lipa/.venv/bin/python3 \
    --no-build-isolation \
    git+https://github.com/google/trajax
```

into the project venv. Trajax is pure-Python (just imports JAX) so the
install is fast (~1 second after dependencies are cached) and there is
no shared library to plumb (no `LD_LIBRARY_PATH`, no env vars).

Override the editable-install fallback path with
`TRAJAX_DIR=/some/other/path` (default `$HOME/github/trajax`); the
fallback only kicks in if the git-direct install fails.

## Sanity check

```bash
UV_NO_BUILD_ISOLATION=0 uv run --no-sync python -c \
    "from trajax.optimizers import ilqr, constrained_ilqr; print('ok')"
```

Expected output: `ok`.

## Run-time environment

No env vars needed. Trajax is a pure-Python JAX package and uses the
project's JAX install; the only dependency it pulls in beyond the
stdlib is `ml-collections` (and `pyyaml` transitively), both
permissively licensed pure-Python packages.

## API discoveries

Trajax's two main entry points:

```python
ilqr(cost, dynamics, x0, U, maxiter=100, grad_norm_threshold=1e-4,
     make_psd=False, psd_delta=0.0, alpha_0=1.0, alpha_min=5e-5,
     vjp_method='tvlqr', ...)

constrained_ilqr(cost, dynamics, x0, U,
                 equality_constraint=lambda x, u, t: jnp.zeros(0),
                 inequality_constraint=lambda x, u, t: jnp.zeros(0),
                 maxiter_al=5, maxiter_ilqr=100,
                 constraints_threshold=1e-2,
                 penalty_init=1.0, penalty_update_rate=10.0,
                 make_psd=True, psd_delta=0.0,
                 alpha_0=1.0, alpha_min=5e-5)
```

Notes:

* **Single-shooting.** Only `U` (shape `(T, m)`) is a decision
  variable; the state trajectory is recomputed by rolling out
  `dynamics(x, u, t)` from `x0`. There is no Crocoddyl-style
  multi-shooting / FDDP variant in trajax — for our multi-shooting
  `ProblemSpec` the adapter discards `problem.X_init` and feeds only
  `problem.U_init`.
* **Stage signature is `(x, u, t)`** — no `theta` argument. The
  adapter captures `problem.Theta_init` via closure and skips cleanly
  with `success=False, notes="trajax does not natively support
  cross-stage Theta (theta_dim=N)"` when `theta_dim > 0`.
* **Constraint dimension must be constant across `t`**. Trajax
  internally `vmap`s the equality / inequality callable along the
  time axis, so the output shape has to be the same at every stage.
  Our `ProblemSpec` already uses the `jnp.where(t == T, terminal,
  stage)` pattern to satisfy this — same convention as the trajax
  test suite.
* **Equality / inequality cannot be zero-dimensional**. Trajax's
  `constrained_ilqr` runs `np.max(np.abs(equality_constraints))`
  inside its termination check, and a zero-sized reduction has no
  identity. When a problem has no equalities we supply a
  constant-zero `(1,)` stub equality; same for missing inequalities,
  with a constant `-1.0` (`g <= 0`, trivially satisfied). The
  corresponding duals stay at zero throughout the AL solve.
* **`constrained_ilqr` returns a 12-tuple**:
  `(X, U, dual_eq, dual_ineq, penalty, eq_constraints,
    ineq_constraints, max_violation, obj, gradient, iter_ilqr,
    iter_al)`. The adapter reports `iter_ilqr` (cumulative inner
  iLQR steps) as the `iterations` field and `max_violation` as the
  primary success criterion.

## Smoke test

### Analytical problems

```bash
LIPA_DISABLE_ALIGATOR=1 UV_NO_BUILD_ISOLATION=0 uv run --no-sync \
    python -m tests.comparison.run_benchmark \
    --problems cartpole,acrobot,quadpendulum \
    --solvers trajax \
    --max-iter 500 \
    --out-dir tmp/trajax_analytical
```

### MJX problems

Trajax also runs on MJX problems — the adapter's JAX-native pipeline
composes with the MJX cost / dynamics / inequality functions:

```bash
LIPA_DISABLE_ALIGATOR=1 UV_NO_BUILD_ISOLATION=0 uv run --no-sync \
    python -m tests.comparison.run_benchmark \
    --problems barrel_roll --solvers trajax \
    --max-iter 5 --out-dir tmp/trajax_mjx_smoke
```

## What the adapter does

`tests/comparison/adapters/trajax.py`:

* Detects `problem.theta_dim > 0` and refuses cleanly (same policy as
  CSQP / Aligator / acados / fatrop).
* Reformulates the multi-shooting `ProblemSpec` as single-shooting:
  feeds only `problem.U_init` as the warm start; state is recomputed
  by trajax's internal rollout.
* Adapts the `(x, u, theta, t)` LIPA stage signature to trajax's
  `(x, u, t)` by closing over `problem.Theta_init`.
* Dispatches to `constrained_ilqr` whenever the problem has any
  equality or inequality constraint, and to plain `ilqr` for
  fully-unconstrained cost-only problems.
* Stubs out missing equality / inequality callables with constant
  `(1,)`-shape benign outputs so `constrained_ilqr`'s internal
  `np.max` doesn't trip over zero-sized arrays.
* Does one warm-up call (`maxiter=1`) before the timed solve so the
  JAX trace + compile cost is excluded from `solve_time_ms`. Same
  convention as the sip-mjx / fatrop-mjx adapters.
* Re-evaluates the final iterate through the canonical
  `evaluate_problem` helper so the reported `final_cost`,
  `eq_violation_inf`, and `ineq_violation_inf` numbers are
  computed identically to every other solver.

## Caveats

1. **Theta is not supported**: `quadpendulum_theta` is refused with
   `success=False, notes="trajax does not natively support
   cross-stage Theta (theta_dim=1)"`.
2. **Single-shooting forces dynamics-feasible iterates**: trajax
   cannot start from a `tile(x0)` warm start in the LIPA sense —
   `problem.X_init` is silently discarded and `X` is whatever the
   rollout of `U_init` from `x0` produces.
3. **`iter_ilqr` is cumulative**, not per-AL-iter — it sums the
   inner iLQR sweeps across all outer AL updates.
4. **Per-iter wall clock**: Trajax's inner tvlqr factorization runs
   in JAX (CPU by default) and is not as fast as fatrop's
   BLASFEO-backed Riccati.
5. **Trajax `0.0.1` is from October 2022** — the upstream repo
   hasn't seen a release in ~3 years. The git HEAD still builds
   against modern JAX (`0.7.x`) without code changes, but expect
   no upstream support if something breaks against a future JAX.
