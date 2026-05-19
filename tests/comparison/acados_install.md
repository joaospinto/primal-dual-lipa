# acados install for the LIPA comparison driver

acados can't be `pip install`-ed: the Python bindings (`acados_template`)
are a thin wrapper around a C library that you have to build yourself
with CMake. The official install guide is at
https://docs.acados.org/installation/index.html. The recipe below
follows that guide on Linux x86_64.

## 1. Clone the repository

```bash
git clone --recursive https://github.com/acados/acados.git ~/github/acados
```

(The `--recursive` flag is important: acados depends on submodules for
HPIPM, BLASFEO, qpOASES, DAQP, and the Tera renderer interface.)

You can also use `--depth 1` if you don't need the full git history;
that's what I used and the build still works.

## 2. Build the C library

```bash
mkdir -p ~/github/acados/build
cd ~/github/acados/build
cmake \
    -DACADOS_INSTALL_DIR=$HOME/github/acados \
    -DACADOS_PYTHON=ON \
    -DACADOS_WITH_QPOASES=ON \
    -DACADOS_WITH_DAQP=ON \
    ..
make -j$(nproc) install
```

`ACADOS_INSTALL_DIR` is where the resulting `lib/libacados.so`,
`include/`, and CMake config files end up. We point it at the source
tree itself (matching the convention in acados's README) so that
`ACADOS_SOURCE_DIR` and the installed headers/libs share a path.

The single-core build takes ~10–20 minutes on this machine; with
`-j$(nproc)` it's a few minutes. CMake will emit a harmless
"Manually-specified variables were not used by the project: ACADOS_PYTHON"
warning — that flag is only needed by older acados versions to pull in
the Python interface; recent versions install the templating files
unconditionally.

## 3. Download the Tera renderer binary

acados generates C code from `.in.c` templates via
[Tera](https://github.com/acados/tera_renderer). The template binary is
*not* built by CMake; you have to download a prebuilt one. The
`acados_template` package will offer to do this for you on first solve,
but the download often fails behind a corporate proxy, so it's safer to
do it up front:

```bash
mkdir -p ~/github/acados/bin
cd ~/github/acados/bin
wget -O t_renderer https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64
chmod +x t_renderer
```

(For non-Linux / non-x86_64 hosts grab the appropriate binary from the
[releases page](https://github.com/acados/tera_renderer/releases).)

## 4. Install the Python bindings

The Python interface package lives at `interfaces/acados_template/`. We
install it editable into the project's uv venv:

```bash
cd ~/github/primal-dual-lipa
UV_NO_BUILD_ISOLATION=0 \
VIRTUAL_ENV=$PWD/.venv \
    uv pip install -e ~/github/acados/interfaces/acados_template
```

Notes:
- `UV_NO_BUILD_ISOLATION=0` is the project-wide quirk for uv 0.5.11
  (see the project README / `pyproject.toml`); without it the install
  silently picks up the wrong Python.
- `VIRTUAL_ENV=$PWD/.venv` forces uv to install into the project's
  local venv. Without it, uv 0.5.11 falls back to whatever Python is on
  `PATH` (typically a system conda env you don't have write access to)
  and the install fails with "Permission denied".
- The install is editable so future `git pull`s in `~/github/acados`
  pick up Python-side changes for free; for C-side changes you also
  have to re-run `make install`.

### Important: `uv run` strips ad-hoc installs

`uv run --group comparisons` re-syncs the venv against `pyproject.toml`
on every invocation, which removes the editable `acados_template`
install (it isn't listed in `pyproject.toml` because acados can't be
installed from PyPI). Two workarounds:

1. **Recommended**: pass `--no-sync` to `uv run` so the editable
   install survives:

   ```bash
   ACADOS_SOURCE_DIR=$HOME/github/acados \
   LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/github/acados/lib \
   UV_NO_BUILD_ISOLATION=0 \
       uv run --no-sync \
       python -m tests.comparison.run_benchmark \
       --solvers acados --problems cartpole,acrobot --max-iter 500
   ```

2. **Fallback**: re-run the `uv pip install -e ...` from step 4 each
   time uv has just synced (e.g. after switching branches or running
   `uv sync`).

## 5. Set the runtime env vars

Every Python process that uses acados needs two env vars:

```bash
export ACADOS_SOURCE_DIR=$HOME/github/acados
# IMPORTANT: append, don't prepend. acados ships its own libblasfeo.so,
# and prepending it to the search path makes JAX (which loads its own
# CUDA / CPU runtime through the same dynamic linker) segfault inside
# jnp.linspace and other ops on this machine — see "Caveats" below.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HOME/github/acados/lib
```

`ACADOS_SOURCE_DIR` tells `acados_template` where to find
`include/`, `bin/t_renderer`, and `c_templates_tera/`.
`LD_LIBRARY_PATH` is what the dynamically-loaded
`libacados_ocp_solver_<name>.so` (built per-OCP at solve time) uses to
find `libacados.so`, `libhpipm.so`, `libblasfeo.so`, and friends at
import time.

For convenience you can stick these in your shell profile, or wrap
the comparison driver call:

```bash
ACADOS_SOURCE_DIR=$HOME/github/acados \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/github/acados/lib \
UV_NO_BUILD_ISOLATION=0 \
    uv run --no-sync \
    python -m tests.comparison.run_benchmark \
    --solvers acados --problems cartpole,acrobot
```

## 6. Sanity-check

```bash
ACADOS_SOURCE_DIR=$HOME/github/acados \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/github/acados/lib \
UV_NO_BUILD_ISOLATION=0 \
    uv run --no-sync python -c \
    "from acados_template import AcadosOcp, AcadosOcpSolver; print('ok')"
```

Expected output: a single `ok` (the `SyntaxWarning: invalid escape
sequence '\s'` from acados's own docstring is harmless).

## What the adapter does

`tests/comparison/adapters/acados.py`:

* Reuses each `ProblemSpec`'s `metadata["casadi_builder"]` (the same
  CasADi builder used by the IPOPT adapter), so dynamics, cost, and
  stage/terminal constraints come from one source of truth.
* Encodes the problem as an `AcadosOcp` with
  `nlp_solver_type='SQP_WITH_FEASIBLE_QP'` (NOT RTI — we want fully
  converged iterations; the FEASIBLE_QP variant runs Byrd-Omojokun
  feasibility restoration whenever the nominal QP is infeasible).
* `hessian_approx='EXACT'` (Gauss-Newton is undefined for arbitrary
  external costs), `regularize_method='PROJECT'` (eigenvalue-clip to
  keep the indefinite Lagrangian Hessian PSD), and
  `globalization='FUNNEL_L1PEN_LINESEARCH'` (the only globalization
  consistent with the FEASIBLE_QP restoration phase).
* Forward-rolls the shipped `U_init` through the dynamics to get a
  *dynamics-feasible* warm start before handing to acados — SQP
  needs a feasible-enough start because each outer iteration solves
  a QP with linearized constraints. See `_rollout_warm_start` in the
  adapter for details.
* Skips cleanly with `success=False, notes='...'` when
  `theta_dim > 0` (acados has no notion of a cross-stage decision
  variable); that's the same policy the CSQP and Aligator adapters
  apply.
* Generates C code into a per-solve `tempfile.TemporaryDirectory`, so
  successive solves don't trample each other and the project tree
  stays clean.

## Smoke test

```
ACADOS_SOURCE_DIR=$HOME/github/acados \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/github/acados/lib \
UV_NO_BUILD_ISOLATION=0 \
    uv run --no-sync \
    python -m tests.comparison.run_benchmark \
    --problems cartpole,acrobot \
    --solvers acados \
    --max-iter 500 \
    --out-dir /tmp/acados_smoke
```

## Caveats

1. `LD_LIBRARY_PATH` ordering matters. acados's `libblasfeo.so`
   conflicts with whatever JAX/numpy/CUDA stack lives earlier on the
   search path: prepending `$HOME/github/acados/lib` makes
   `jnp.linspace(...)` segfault inside the JAX runtime. Append, don't
   prepend.
2. Tera renderer can't parse `Infinity` in the generated JSON. The
   adapter uses `ACADOS_INFTY = 1e10` as the "no lower bound" sentinel
   for `lh` / `lh_e` instead of `-np.inf`. Don't switch back.
3. Problems whose feasible set has multiple disconnected components
   (e.g. obstacle-avoidance plus a terminal equality) can stall in
   the funnel line search; SQP-based methods get trapped inside the
   first locally-feasible component the line search reaches.
   Reformulation (slacked obstacles, terminal cost instead of terminal
   constraint, relaxation-then-tighten) is the usual workaround; we
   leave the problem statement unchanged across the comparison.
