#!/usr/bin/env bash
# Orchestrate the full multi-pass comparison benchmark. Produces
# comparison_results/full/{report.md, results.csv, plots/}.
#
# Why multiple passes:
#   1. Aligator's bundled pinocchio (3.4 from conda-forge) and CSQP's
#      pinocchio (3.8 from cmeel via PyPI) cannot coexist in one Python
#      process — Boost.Python's global type registry is single-tenant.
#      So Aligator runs in its own pass.
#   2. LIPA-gpu vs LIPA-cpu: separate JAX_PLATFORMS for each.
#   3. csqp / aligator / sip have both a casadi backend and a jax
#      backend. We run them twice (once per backend) and report both
#      rows so the comparison covers each backend.
#   4. MJX problems use a different set of solvers (ipopt-mjx,
#      fatrop-mjx, sip-mjx, trajax, lipa, csqp) than analytical
#      problems (ipopt, fatrop, sip, acados, aligator, csqp, trajax,
#      lipa). Splitting by problem class keeps each pass focused.
#
# Each pass emits a results.csv into its own subdir. The merge step
# combines them into one canonical report.

# NOTE: `-e` is intentionally OFF: the runner returns nonzero when any
# (solver, problem) pair didn't converge, but we want the orchestrator
# to keep going so the merged report aggregates all passes' results.
# `-u` (undefined-var trap) and `pipefail` stay on for safety.
set -uo pipefail

cd "$(dirname "$0")/../.."  # cd to repo root

OUT_ROOT="${OUT_ROOT:-comparison_results}"
MAX_ITER="${MAX_ITER:-1000}"
MAX_ITER_MJX="${MAX_ITER_MJX:-500}"

# Env vars consumed by adapters:
#   ACADOS_SOURCE_DIR / LD_LIBRARY_PATH — acados needs both, see acados_install.md
#   ALIGATOR_SITE_PACKAGES — aligator adapter's conda-env path probe
#   LIPA_DISABLE_ALIGATOR — when set, aligator skips its sys.path injection so
#     csqp's pinocchio can load without ABI conflict.
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/github/acados}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ACADOS_SOURCE_DIR/lib"
export ALIGATOR_SITE_PACKAGES="${ALIGATOR_SITE_PACKAGES:-$HOME/.conda/envs/aligator-side/lib/python3.13/site-packages}"

# Bit-stable XLA flags: deterministic GPU reductions and single-threaded
# XLA-CPU Eigen path (otherwise XLA-CPU's Eigen threadpool produces
# different reduction orders under varying load). These only affect
# JAX-using adapters; the C/C++ solvers' BLAS threading is unchanged.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_cpu_multi_thread_eigen=false"

mkdir -p "$OUT_ROOT"

uv_run() {
    UV_NO_BUILD_ISOLATION=0 uv run --no-sync "$@"
}

echo "==> Pass 1a/6: analytical on GPU — lipa + trajax (both JAX)"
# LIPA uses regularized_lqr_jax with associative scans (parallel
# Riccati), so GPU helps. Trajax does sequential Riccati recursion,
# so GPU is mostly wasted on it, but we run it on GPU for parity.
LIPA_DISABLE_ALIGATOR=1 \
    uv_run python -m tests.comparison.run_benchmark \
        --problems cartpole,acrobot,quadpendulum,quadpendulum_theta \
        --solvers lipa,trajax \
        --max-iter "$MAX_ITER" \
        --label-suffix=-gpu \
        --out-dir "$OUT_ROOT/jax_gpu"

echo
echo "==> Pass 1b/6: analytical on CPU — ipopt + acados + fatrop (no GPU code paths)"
LIPA_DISABLE_ALIGATOR=1 \
    uv_run python -m tests.comparison.run_benchmark \
        --problems cartpole,acrobot,quadpendulum,quadpendulum_theta \
        --solvers ipopt,acados,fatrop \
        --max-iter "$MAX_ITER" \
        --out-dir "$OUT_ROOT/cpu_baselines"

echo
echo "==> Pass 2/6: analytical, csqp + sip (casadi backend)"
LIPA_DISABLE_ALIGATOR=1 \
    uv_run python -m tests.comparison.run_benchmark \
        --problems cartpole,acrobot,quadpendulum,quadpendulum_theta \
        --solvers csqp,sip \
        --max-iter "$MAX_ITER" \
        --backend casadi \
        --label-suffix=-casadi \
        --out-dir "$OUT_ROOT/csqp_sip_casadi"

echo
echo "==> Pass 3/6: analytical, csqp + sip (jax backend — for comparison)"
LIPA_DISABLE_ALIGATOR=1 \
    uv_run python -m tests.comparison.run_benchmark \
        --problems cartpole,acrobot,quadpendulum,quadpendulum_theta \
        --solvers csqp,sip \
        --max-iter "$MAX_ITER" \
        --backend jax \
        --label-suffix=-jax \
        --out-dir "$OUT_ROOT/csqp_sip_jax"

echo
echo "==> Pass 4/6: LIPA-cpu analytical (forces JAX onto CPU)"
JAX_PLATFORMS=cpu LIPA_DISABLE_ALIGATOR=1 \
    uv_run python -m tests.comparison.run_benchmark \
        --problems cartpole,acrobot,quadpendulum,quadpendulum_theta \
        --solvers lipa \
        --max-iter "$MAX_ITER" \
        --label-suffix=-cpu \
        --out-dir "$OUT_ROOT/lipa_cpu"

echo
echo "==> Pass 5/6: Aligator analytical (separate process to avoid pinocchio ABI clash)"
uv_run python -m tests.comparison.run_benchmark \
    --problems cartpole,acrobot \
    --solvers aligator \
    --max-iter "$MAX_ITER" \
    --backend casadi \
    --label-suffix=-casadi \
    --out-dir "$OUT_ROOT/aligator_casadi"

uv_run python -m tests.comparison.run_benchmark \
    --problems cartpole,acrobot \
    --solvers aligator \
    --max-iter "$MAX_ITER" \
    --backend jax \
    --label-suffix=-jax \
    --out-dir "$OUT_ROOT/aligator_jax"

echo
echo "==> Pass 6/6: MJX problems (lipa + ipopt-mjx + fatrop-mjx + sip-mjx + csqp + trajax)"
LIPA_DISABLE_ALIGATOR=1 \
    uv_run python -m tests.comparison.run_benchmark \
        --problems barrel_roll,h1_backflip,h1_jump_forward,aliengo_trot \
        --solvers lipa,ipopt-mjx,fatrop-mjx,sip-mjx,csqp,trajax \
        --max-iter "$MAX_ITER_MJX" \
        --out-dir "$OUT_ROOT/mjx"

echo
echo "==> Merging all passes into $OUT_ROOT/full/"
uv_run python -m tests.comparison.merge_reports \
    "$OUT_ROOT/jax_gpu/results.csv" \
    "$OUT_ROOT/cpu_baselines/results.csv" \
    "$OUT_ROOT/csqp_sip_casadi/results.csv" \
    "$OUT_ROOT/csqp_sip_jax/results.csv" \
    "$OUT_ROOT/lipa_cpu/results.csv" \
    "$OUT_ROOT/aligator_casadi/results.csv" \
    "$OUT_ROOT/aligator_jax/results.csv" \
    "$OUT_ROOT/mjx/results.csv" \
    --out-dir "$OUT_ROOT/full"

echo
echo "Done. See:"
echo "  $OUT_ROOT/full/report.md       (markdown summary)"
echo "  $OUT_ROOT/full/results.csv     (raw rows)"
echo "  $OUT_ROOT/full/plots/          (per-problem convergence PNGs)"
