# fatrop install for the LIPA comparison driver

[fatrop](https://github.com/meco-group/fatrop) is a structured nonlinear
optimal-control solver — a primal-dual interior-point method that
exploits the OCP block structure with a generalized Riccati recursion
(BLASFEO-backed). It's the closest direct algorithmic competitor to
LIPA in this comparison set; the head-to-head numbers on the same
problems are the most informative single data point in the report.

## TL;DR — nothing extra to install

fatrop ships a CasADi `nlpsol` plugin and the **CasADi pip wheel
(>= 3.7) bundles a prebuilt copy of that plugin**. The project venv
already has CasADi (the `comparisons` group pulls it in for the IPOPT
and acados adapters), so `nlpsol(..., 'fatrop', ...)` works out of the
box — no separate `fatropy` PyPI package, no source build, no
`LD_LIBRARY_PATH` plumbing.

The install script in this repo therefore does almost nothing: it just
verifies the bundled plugin loads and (optionally) clones the upstream
repo for examples / docs reference.

```bash
bash tests/comparison/install/install_fatrop.sh
```

Override the clone location with `FATROP_DIR=/some/other/path`
(default `$HOME/github/fatrop`). The clone is purely informational —
nothing in this repo reads from it.

## Sanity-check

```bash
UV_NO_BUILD_ISOLATION=0 uv run --no-sync python -c \
    "import casadi as ca; \
     ca.nlpsol('p','fatrop',{'x':ca.SX.sym('x'),'f':ca.SX(1.),'g':ca.SX(0.)}, \
       {'structure_detection':'manual','nx':[1,0],'nu':[0,0],'ng':[0,0],'N':1, \
        'fatrop':{'print_level':0}}); \
     print('ok')"
```

Expected output: a single `ok`.

## Source build (only if the bundled plugin is missing)

If you need to build fatrop from source — e.g. you're on an exotic
platform where the CasADi wheel didn't bundle the fatrop plugin, or
you want to track upstream `meco-group/fatrop` master — the upstream
build sequence is:

```bash
git clone --recursive https://github.com/meco-group/fatrop.git ~/github/fatrop
mkdir -p ~/github/fatrop/build
cd ~/github/fatrop/build
cmake -DBUILD_WITH_C_INTERFACE=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install   # or set CMAKE_INSTALL_PREFIX
```

Then point CasADi at the plugin you just built via `CASADIPATH` (or
copy `libcasadi_nlpsol_fatrop.so` into the CasADi pip wheel's lib
dir). In our case this was unnecessary: the pip-bundled plugin works.

There is **no upstream `fatropy` Python binding** — the Python entry
points for fatrop are all via the CasADi `nlpsol` plugin (and the C
interface in `c_interface/`). The `fatrop_demo` repo cited from
upstream just uses the CasADi route, same as our adapter.

## Run-time environment

No special env vars are needed for the bundled-plugin route. Compare
that with acados, which needs `ACADOS_SOURCE_DIR` and a
`LD_LIBRARY_PATH` append; fatrop's plugin is fully self-contained
inside the CasADi wheel.

One gotcha: if a *different* CasADi (e.g. the conda-forge `casadi`
that ships inside the `aligator-side` env) gets prepended to
`sys.path`, you'll suddenly hit
`Plugin 'fatrop' is not found` because the conda-forge CasADi build
generally does NOT bundle the fatrop plugin. The aligator adapter
prepends its conda env's site-packages eagerly at import time; pass
`LIPA_DISABLE_ALIGATOR=1` (the same flag the README documents for the
CSQP pass) to suppress that and keep the venv's CasADi (with its
fatrop plugin) on top.

## What the adapter does

`tests/comparison/adapters/fatrop.py`:

* Builds a CasADi `Opti` problem stage by stage (variables allocated
  in the interleaved `[x_0, u_0, x_1, u_1, ..., x_T]` order required
  by fatrop's manual structure detection).
* Reuses each `ProblemSpec`'s `metadata["casadi_builder"]` (the same
  builder the IPOPT and acados adapters use), so dynamics, cost, and
  stage / terminal constraints come from one source of truth.
* Encodes equalities as `lb = ub = 0` rows of the per-stage path
  constraint vector `g_k`, inequalities as `lb = -inf, ub = 0` rows.
  The dynamics defect is added via `opti.subject_to(x_{k+1} ==
  next_x(x_k, u_k))` and is what fatrop's Riccati step exploits.
* Initializes the X warm start as `linspace(x0, goal)` when the
  problem has an affine terminal-equality goal, and uses the shipped
  `X_init` directly otherwise. The `goal` is recovered by reading
  the constant offset of the terminal-equality constraint from the
  casadi builder (jacobian must be identity; otherwise we silently
  fall back). The controls always come from the shipped `U_init`.
  See the module docstring for why a linspace warm start is robust
  for filter-IPMs when the shipped warm start is a degenerate
  `tile(x0)`.
* Skips cleanly with `success=False, notes='...'` when
  `theta_dim > 0` (fatrop's OCP layout has no cross-stage decision
  variable); same policy as the CSQP / Aligator / acados adapters.
* Does one warm-up solve before the timed solve so the per-OCP CasADi
  expand + fatrop BLASFEO setup is excluded from the reported wall
  clock (same convention as the other adapters).

## Smoke test

```bash
LIPA_DISABLE_ALIGATOR=1 UV_NO_BUILD_ISOLATION=0 uv run --no-sync \
    python -m tests.comparison.run_benchmark \
    --problems cartpole,acrobot,quadpendulum \
    --solvers fatrop \
    --max-iter 1000 \
    --out-dir /tmp/fatrop_smoke
```

## Caveats

1. `quadpendulum_theta` is refused cleanly with `notes="fatrop does
   not natively support cross-stage Theta (theta_dim=1)"`. To compare
   on this problem you have to fall back to LIPA or IPOPT.
2. The `iter_count` reported in `SolverResult` is fatrop's main IPM
   loop count. The restoration phase has its own inner iterations
   (`R`-prefixed lines under `fatrop.print_level`) which are NOT
   counted. Same convention as IPOPT.
3. The bundled `libfatrop.so` enforces an internal upper bound on
   `max_iter` (~5000); requesting more emits a warning but doesn't
   raise.
4. On non-convex problems, fatrop and LIPA may converge to different
   local minima depending on warm start.
