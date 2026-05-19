#!/usr/bin/env bash
# Orchestrate the full multi-pass comparison benchmark. Produces
# comparison_results/full/{report.md, results.csv, plots/}.
#
# Layout: ONE pass per (solver, backend, platform) combination. Each
# pass writes one results.csv with exactly one solver_name (after
# --label-suffix). The merge step combines all per-pass CSVs into one
# canonical report.
#
# Why one pass per variant (rather than grouping solvers):
#   * Each (solver, problem) pair already runs in its own subprocess
#     via --hard-timeout-s, so the per-stage pinocchio-ABI / JAX-state
#     isolation that historically motivated lumping is no longer a
#     pass-level concern.
#   * One pass = one report-row variant makes re-running a single
#     solver after a tuning tweak as cheap as re-invoking that one
#     ``run_benchmark`` line.
#   * Per-pass env-var differences (JAX_PLATFORMS, LIPA_DISABLE_ALIGATOR)
#     and CLI-flag differences (--backend, --label-suffix) are minimal
#     and self-documenting at the call site.
#
# LIPA gets explicit -cpu and -gpu variants because the comparison is
# the whole point. Other JAX-using solvers (trajax, sip-jax, csqp-jax,
# aligator-jax) run on whatever platform is current — pick GPU when
# available, CPU when not. If you want both for those, run twice with
# different ``JAX_PLATFORMS``.

# NOTE: ``-e`` is intentionally OFF: the runner returns nonzero when any
# (solver, problem) pair didn't converge, but we want the orchestrator
# to keep going so the merged report aggregates all passes' results.
# ``-u`` (undefined-var trap) and ``pipefail`` stay on for safety.
set -uo pipefail

cd "$(dirname "$0")/../.."  # cd to repo root

OUT_ROOT="${OUT_ROOT:-comparison_results}"
MAX_ITER="${MAX_ITER:-1000}"
MAX_ITER_MJX="${MAX_ITER_MJX:-500}"

# Two-tier wall-time caps so a stuck solver can't burn the whole run:
#   * TIMEOUT_S — soft cap passed to the adapter. IPOPT / IPOPT-MJX
#     honor this natively (max_wall_time); the rest ignore it.
#   * HARD_TIMEOUT_S — hard cap. Each (solver, problem) pair runs in
#     its own subprocess and is SIGKILLed if it exceeds this budget.
#     The gap (HARD - SOFT) must cover JIT compile + problem build +
#     subprocess startup. Killed pairs land in the report as
#     success=False, notes="hard-killed after Ns".
TIMEOUT_S="${TIMEOUT_S:-60}"
HARD_TIMEOUT_S="${HARD_TIMEOUT_S:-180}"
TIMEOUT_S_MJX="${TIMEOUT_S_MJX:-600}"
HARD_TIMEOUT_S_MJX="${HARD_TIMEOUT_S_MJX:-1200}"

ANALYTICAL_PROBLEMS="cartpole,acrobot,quadpendulum,quadpendulum_theta"
MJX_PROBLEMS="barrel_roll,backflip,jump,trot"
ALL_PROBLEMS="${ANALYTICAL_PROBLEMS},${MJX_PROBLEMS}"

# Env vars consumed by adapters:
#   ACADOS_SOURCE_DIR / LD_LIBRARY_PATH — acados needs both, see acados_install.md
#   ALIGATOR_SITE_PACKAGES — aligator adapter's conda-env path probe
#   LIPA_DISABLE_ALIGATOR — when set, aligator skips its sys.path injection so
#     csqp's pinocchio can load without ABI conflict.
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/github/acados}"
# Append acados's BLASFEO/HPIPM build output dir (acados runtime needs
# it). Also append cmeel.prefix/lib: that's where the cmeel pinocchio
# wheels stash their shared libs, and the crocoddyl/mim_solvers /
# fatrop-plugin .so files have rpath that doesn't reach there
# automatically — so without this LD_LIBRARY_PATH entry, csqp and
# fatrop both report "libpinocchio_parsers.so.3 not found" at import
# time. We probe the venv path lazily so this doesn't fail on a
# fresh checkout where .venv hasn't been created yet.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ACADOS_SOURCE_DIR/lib"
_CMEEL_LIB=$(ls -d .venv/lib/python*/site-packages/cmeel.prefix/lib 2>/dev/null | head -1 || true)
if [ -n "$_CMEEL_LIB" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(realpath "$_CMEEL_LIB")"
fi
export ALIGATOR_SITE_PACKAGES="${ALIGATOR_SITE_PACKAGES:-$HOME/.conda/envs/aligator-side/lib/python3.13/site-packages}"

# Bit-stable XLA flags: deterministic GPU reductions and single-threaded
# XLA-CPU Eigen path (otherwise XLA-CPU's Eigen threadpool produces
# different reduction orders under varying load). These only affect
# JAX-using adapters; the C/C++ solvers' BLAS threading is unchanged.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_cpu_multi_thread_eigen=false"

mkdir -p "$OUT_ROOT"

# Detect NVIDIA GPU presence. The NVIDIA kernel module creates
# ``/dev/nvidia*`` device nodes when loaded; their absence means CUDA
# is unusable (Docker on macOS, GPU-less CI box, etc.). When no GPU
# is present, we skip GPU-labelled passes (so we don't write
# misleading -gpu rows that actually ran on CPU) and tell JAX upfront
# that CPU is the only platform (so its CUDA plugin doesn't print a
# noisy traceback on every Python process).
HAS_GPU=0
if compgen -G "/dev/nvidia*" > /dev/null; then
    HAS_GPU=1
else
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
fi

# Whether to run the MJX problems. They are JAX-native and designed
# for GPU acceleration; on CPU each MJX solve takes minutes (LIPA
# 1-5 min, others 10-30 min) and the comparison numbers don't reflect
# what users would see in production. Default to running MJX iff a
# GPU is present; set RUN_MJX=1 to override and force CPU MJX
# (e.g. on a beefy CPU box where 30-min solves are tolerable).
RUN_MJX="${RUN_MJX:-$HAS_GPU}"
if [ "$RUN_MJX" -eq 1 ]; then
    ALL_PROBLEMS_OR_ANALYTICAL="$ALL_PROBLEMS"
else
    ALL_PROBLEMS_OR_ANALYTICAL="$ANALYTICAL_PROBLEMS"
    echo "==> RUN_MJX=0 (default when no GPU). All MJX-only passes will be"
    echo "    SKIPPED and dual-class passes will run only on the 4 analytical"
    echo "    problems. Set RUN_MJX=1 to force CPU MJX runs."
fi

uv_run() {
    UV_NO_BUILD_ISOLATION=0 uv run --no-sync "$@"
}

# ---------------------------------------------------------------------
# Pass helper: invoke run_benchmark for one (solver, backend, platform)
# combo and add its results.csv to the merge list. All passes set
# LIPA_DISABLE_ALIGATOR=1 by default — the two aligator passes below
# call ``uv_run`` directly to opt out.
# ---------------------------------------------------------------------
MERGE_CSVS=()

run_pass() {
    # Usage: run_pass <name> <solver> <problems> <soft_timeout> <hard_timeout> [extra args...]
    local name="$1"; local solver="$2"; local problems="$3"
    local soft="$4"; local hard="$5"
    shift 5
    local out_dir="$OUT_ROOT/$name"
    echo
    echo "==> Pass: $name  ($solver on $(echo "$problems" | tr ',' ' '))"
    LIPA_DISABLE_ALIGATOR=1 \
        uv_run python -m tests.comparison.run_benchmark \
            --problems "$problems" \
            --solvers "$solver" \
            --max-iter "$MAX_ITER" \
            --timeout-s "$soft" \
            --hard-timeout-s "$hard" \
            --out-dir "$out_dir" \
            "$@"
    MERGE_CSVS+=("$out_dir/results.csv")
}

# ---------------------------------------------------------------------
# Passes. Each is one (solver, backend, platform) -> one report row.
# Per-pass timeouts use the MJX values when the pass runs on any MJX
# problem (most stiff MJX solves need minutes; the analytical timeout
# would kill them prematurely).
# ---------------------------------------------------------------------

# ---- LIPA: explicit cpu vs gpu variants (the comparison we care about).
if [ "$HAS_GPU" -eq 1 ]; then
    run_pass "lipa_gpu" "lipa" "$ALL_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
        --label-suffix=-gpu
else
    echo
    echo "==> Pass: lipa_gpu  SKIPPED (no NVIDIA GPU detected)."
    echo "    Running on CPU with a -gpu suffix would mis-label results."
fi
RUN_LIPA_CPU_ON_MJX_PROBLEMS="${RUN_LIPA_CPU_ON_MJX_PROBLEMS:-0}"
if [ "$RUN_LIPA_CPU_ON_MJX_PROBLEMS" -eq 1 ]; then
    LIPA_CPU_PROBLEMS="$ALL_PROBLEMS_OR_ANALYTICAL"
else
    LIPA_CPU_PROBLEMS="$ANALYTICAL_PROBLEMS"
fi
JAX_PLATFORMS=cpu run_pass "lipa_cpu" "lipa" "$LIPA_CPU_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
    --label-suffix=-cpu

# ---- IPOPT: two distinct adapter classes — ipopt-casadi (CasADi
#      symbolic) and ipopt-jax (per-stage JAX via CasADi callback).
#      Both run on every supported problem so we can cross-validate the
#      autodiff backends.
run_pass "ipopt_casadi" "ipopt-casadi" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
run_pass "ipopt_jax_analytical" "ipopt-jax" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
if [ "$RUN_MJX" -eq 1 ]; then
    run_pass "ipopt_jax" "ipopt-jax" "$MJX_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX"
else
    echo; echo "==> Pass: ipopt_jax (MJX) SKIPPED (RUN_MJX=0)."
fi

# ---- acados: CasADi backend, analytical only.
run_pass "acados" "acados" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"

# ---- fatrop: same shape as IPOPT — two distinct adapter classes.
#      Both run on every supported problem (-jax does not support
#      theta_dim>0).
run_pass "fatrop_casadi" "fatrop-casadi" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
run_pass "fatrop_jax_analytical" "fatrop-jax" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
if [ "$RUN_MJX" -eq 1 ]; then
    run_pass "fatrop_jax" "fatrop-jax" "$MJX_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX"
else
    echo; echo "==> Pass: fatrop_jax (MJX) SKIPPED (RUN_MJX=0)."
fi

# ---- sip: ONE adapter, two backends — sip-casadi refuses MJX problems
#      (no casadi_builder); sip-jax handles both analytical and MJX
#      (per-stage JAX vmaps). Because the jax pass has real analytical
#      work to do even without MJX, ALL_PROBLEMS_OR_ANALYTICAL gates it
#      to the analytical subset rather than skipping the whole pass.
run_pass "sip_casadi" "sip" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S"     "$HARD_TIMEOUT_S"     \
    --backend casadi --label-suffix=-casadi
run_pass "sip_jax"    "sip" "$ALL_PROBLEMS_OR_ANALYTICAL" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
    --backend jax    --label-suffix=-jax

# ---- csqp: same shape as sip — ONE adapter, two backends. csqp-casadi
#      handles analytical only; csqp-jax handles both (crocoddyl's JAX
#      action models cover MJX). Like sip_jax, the jax pass keeps its
#      analytical work when RUN_MJX=0 via ALL_PROBLEMS_OR_ANALYTICAL.
run_pass "csqp_casadi" "csqp" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S"     "$HARD_TIMEOUT_S"     \
    --backend casadi --label-suffix=-casadi
run_pass "csqp_jax"    "csqp" "$ALL_PROBLEMS_OR_ANALYTICAL" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
    --backend jax    --label-suffix=-jax

# ---- aligator: one adapter, two backends. Both refuse MJX problems
#      structurally (see aligator.py docstring). Both also refuse
#      theta_dim>0 cleanly via the adapter's own gate (aligator.py:614),
#      so the report row for quadpendulum_theta surfaces the refusal
#      reason rather than disappearing. Quadpendulum stays in the list
#      too: the casadi backend fails fast (Newton diverges to massive
#      ineq/NaN), and the jax backend hits the 180s subprocess hard-kill
#      — both produce informative rows.
#      Aligator is the ONE adapter where LIPA_DISABLE_ALIGATOR must be
#      UNSET (so the aligator adapter can inject its conda-env path),
#      so we bypass the run_pass helper and invoke uv_run directly.
echo
echo "==> Pass: aligator_casadi  (aligator on $(echo "$ANALYTICAL_PROBLEMS" | tr ',' ' '))"
uv_run python -m tests.comparison.run_benchmark \
    --problems "$ANALYTICAL_PROBLEMS" \
    --solvers aligator \
    --max-iter "$MAX_ITER" \
    --timeout-s "$TIMEOUT_S" \
    --hard-timeout-s "$HARD_TIMEOUT_S" \
    --backend casadi \
    --label-suffix=-casadi \
    --out-dir "$OUT_ROOT/aligator_casadi"
MERGE_CSVS+=("$OUT_ROOT/aligator_casadi/results.csv")

echo
echo "==> Pass: aligator_jax  (aligator on $(echo "$ANALYTICAL_PROBLEMS" | tr ',' ' '))"
uv_run python -m tests.comparison.run_benchmark \
    --problems "$ANALYTICAL_PROBLEMS" \
    --solvers aligator \
    --max-iter "$MAX_ITER" \
    --timeout-s "$TIMEOUT_S" \
    --hard-timeout-s "$HARD_TIMEOUT_S" \
    --backend jax \
    --label-suffix=-jax \
    --out-dir "$OUT_ROOT/aligator_jax"
MERGE_CSVS+=("$OUT_ROOT/aligator_jax/results.csv")

# ---- trajax: JAX-only. Runs on whatever platform is current
#      (no --backend, no -cpu/-gpu suffix). If you want both, run
#      twice with different JAX_PLATFORMS.
run_pass "trajax" "trajax" "$ALL_PROBLEMS_OR_ANALYTICAL" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX"

echo
echo "==> Merging ${#MERGE_CSVS[@]} pass(es) into $OUT_ROOT/full/"
uv_run python -m tests.comparison.merge_reports \
    "${MERGE_CSVS[@]}" \
    --out-dir "$OUT_ROOT/full"

echo
echo "Done. See:"
echo "  $OUT_ROOT/full/report.md       (markdown summary)"
echo "  $OUT_ROOT/full/results.csv     (raw rows)"
echo "  $OUT_ROOT/full/plots/          (per-problem convergence PNGs)"
