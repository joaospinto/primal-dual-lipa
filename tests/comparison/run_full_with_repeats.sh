#!/usr/bin/env bash
# Full LIPA-vs-SOTA benchmark with:
# - --ignore-known-timeouts on every invocation (no pair is skipped)
# - doubled default process timeouts (analytical 60→120s soft, 180→360s hard;
#   MJX 600→1200s soft, 1200→2400s hard)
# - LIPA-GPU runs 5× total on the MJX problems (kept separate so the user
#   can combine them after); the analytical LIPA-GPU pass + all other
#   solvers run once.
#
# Outputs layout under $OUT_ROOT (default comparison_results.full_repeats):
#   $OUT_ROOT/main_run/           — single full benchmark (the run_all.sh-equivalent
#                                    pass that already includes 1× LIPA-GPU on MJX)
#   $OUT_ROOT/lipa_gpu_mjx_run{2,3,4,5}/  — additional 4 LIPA-GPU MJX-only repeats
#
# Each subdir contains results.csv + report.md + plots/. The user combines
# the LIPA-MJX runs however they want.

set -uo pipefail

cd "$(dirname "$0")/../.."

OUT_ROOT="${OUT_ROOT:-comparison_results.full_repeats}"
MAX_ITER="${MAX_ITER:-1000}"
MAX_ITER_MJX="${MAX_ITER_MJX:-500}"

# Doubled defaults vs run_all.sh:
TIMEOUT_S="${TIMEOUT_S:-120}"
HARD_TIMEOUT_S="${HARD_TIMEOUT_S:-360}"
TIMEOUT_S_MJX="${TIMEOUT_S_MJX:-1200}"
HARD_TIMEOUT_S_MJX="${HARD_TIMEOUT_S_MJX:-2400}"

ANALYTICAL_PROBLEMS="cartpole,acrobot,quadpendulum,quadpendulum_theta"
MJX_PROBLEMS="barrel_roll,backflip,jump,trot"
ALL_PROBLEMS="${ANALYTICAL_PROBLEMS},${MJX_PROBLEMS}"

# Same env-var preamble as run_all.sh (acados, cmeel, aligator, XLA flags)
export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-$HOME/github/acados}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$ACADOS_SOURCE_DIR/lib"
_CMEEL_LIB=$(ls -d .venv/lib/python*/site-packages/cmeel.prefix/lib 2>/dev/null | head -1 || true)
if [ -n "$_CMEEL_LIB" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(realpath "$_CMEEL_LIB")"
fi
export ALIGATOR_SITE_PACKAGES="${ALIGATOR_SITE_PACKAGES:-$HOME/.conda/envs/aligator-side/lib/python3.13/site-packages}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_cpu_multi_thread_eigen=false"

mkdir -p "$OUT_ROOT"

HAS_GPU=0
if compgen -G "/dev/nvidia*" > /dev/null; then
    HAS_GPU=1
fi

uv_run() { UV_NO_BUILD_ISOLATION=0 uv run --no-sync "$@"; }

# Pass helper — adds --ignore-known-timeouts to every invocation
run_pass() {
    local name="$1"; local solver="$2"; local problems="$3"
    local soft="$4"; local hard="$5"
    shift 5
    local out_dir="$MAIN_OUT/$name"
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
            --ignore-known-timeouts \
            "$@"
    MERGE_CSVS+=("$out_dir/results.csv")
}

# -------- Main run: full set of passes (LIPA on all problems, all SOTA solvers)
MAIN_OUT="$OUT_ROOT/main_run"
mkdir -p "$MAIN_OUT"
MERGE_CSVS=()

if [ "$HAS_GPU" -eq 1 ]; then
    run_pass "lipa_gpu" "lipa" "$ALL_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
        --label-suffix=-gpu
fi

# LIPA CPU on analytical only (matches run_all.sh default behavior; set
# RUN_LIPA_CPU_ON_MJX_PROBLEMS=1 to extend)
RUN_LIPA_CPU_ON_MJX_PROBLEMS="${RUN_LIPA_CPU_ON_MJX_PROBLEMS:-0}"
if [ "$RUN_LIPA_CPU_ON_MJX_PROBLEMS" -eq 1 ]; then
    LIPA_CPU_PROBLEMS="$ALL_PROBLEMS"
else
    LIPA_CPU_PROBLEMS="$ANALYTICAL_PROBLEMS"
fi
JAX_PLATFORMS=cpu run_pass "lipa_cpu" "lipa" "$LIPA_CPU_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
    --label-suffix=-cpu

run_pass "ipopt_casadi" "ipopt-casadi" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
run_pass "ipopt_jax_analytical" "ipopt-jax" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
run_pass "ipopt_jax" "ipopt-jax" "$MJX_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX"

run_pass "acados" "acados" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"

run_pass "fatrop_casadi" "fatrop-casadi" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
run_pass "fatrop_jax_analytical" "fatrop-jax" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S"
run_pass "fatrop_jax" "fatrop-jax" "$MJX_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX"

run_pass "sip_casadi" "sip" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S" \
    --backend casadi --label-suffix=-casadi
run_pass "sip_jax" "sip" "$ALL_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
    --backend jax --label-suffix=-jax

run_pass "csqp_casadi" "csqp" "$ANALYTICAL_PROBLEMS" "$TIMEOUT_S" "$HARD_TIMEOUT_S" \
    --backend casadi --label-suffix=-casadi
run_pass "csqp_jax" "csqp" "$ALL_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX" \
    --backend jax --label-suffix=-jax

# aligator (no LIPA_DISABLE_ALIGATOR)
for variant in casadi jax; do
    name="aligator_${variant}"
    out_dir="$MAIN_OUT/$name"
    echo; echo "==> Pass: $name"
    uv_run python -m tests.comparison.run_benchmark \
        --problems "$ANALYTICAL_PROBLEMS" \
        --solvers aligator \
        --max-iter "$MAX_ITER" \
        --timeout-s "$TIMEOUT_S" --hard-timeout-s "$HARD_TIMEOUT_S" \
        --backend "$variant" --label-suffix="-$variant" \
        --out-dir "$out_dir" \
        --ignore-known-timeouts
    MERGE_CSVS+=("$out_dir/results.csv")
done

run_pass "trajax" "trajax" "$ALL_PROBLEMS" "$TIMEOUT_S_MJX" "$HARD_TIMEOUT_S_MJX"

echo
echo "==> Merging ${#MERGE_CSVS[@]} pass(es) into $MAIN_OUT/merged/"
uv_run python -m tests.comparison.merge_reports \
    "${MERGE_CSVS[@]}" --out-dir "$MAIN_OUT/merged"

# -------- LIPA-GPU MJX repeats (runs 2..5; run 1 is already in main_run/lipa_gpu)
if [ "$HAS_GPU" -eq 1 ]; then
    for run_idx in 2 3 4 5; do
        out_dir="$OUT_ROOT/lipa_gpu_mjx_run${run_idx}"
        echo; echo "==> LIPA-GPU MJX repeat $run_idx → $out_dir"
        LIPA_DISABLE_ALIGATOR=1 uv_run python -m tests.comparison.run_benchmark \
            --problems "$MJX_PROBLEMS" \
            --solvers lipa --max-iter "$MAX_ITER" \
            --timeout-s "$TIMEOUT_S_MJX" --hard-timeout-s "$HARD_TIMEOUT_S_MJX" \
            --label-suffix=-gpu \
            --out-dir "$out_dir" \
            --ignore-known-timeouts
    done
fi

echo
echo "Done."
echo "Main merged report:  $MAIN_OUT/merged/report.md"
echo "Per-pass CSVs:       $MAIN_OUT/<pass_name>/results.csv"
echo "LIPA-GPU MJX runs:   $OUT_ROOT/main_run/lipa_gpu/results.csv (run 1)"
echo "                     $OUT_ROOT/lipa_gpu_mjx_run{2,3,4,5}/results.csv"
