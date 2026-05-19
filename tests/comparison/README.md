# LIPA solver comparison

A standalone CLI that benchmarks LIPA against other NLP-OCP solvers.

Solvers covered (one file per adapter in `adapters/`):

| solver | analytical | MJX |
|---|---|---|
| LIPA | ✓ | ✓ |
| IPOPT (CasADi) | ✓ (`ipopt-casadi`) | ✓ (sparse-callback variant `ipopt-jax`) |
| acados | ✓ | — |
| fatrop | ✓ (`fatrop-casadi`) | ✓ (variant `fatrop-jax`) |
| Aligator | ✓ | — |
| CSQP (Crocoddyl + mim_solvers) | ✓ | ✓ |
| SIP (`sip_python`) | ✓ | ✓ (variant `sip-mjx`) |
| Trajax | ✓ | ✓ |

Problems live in `problems/`: 4 analytical (`cartpole`, `acrobot`,
`quadpendulum`, `quadpendulum_theta`) and 4 MJX (`barrel_roll`,
`backflip`, `jump`, `trot`).

## Install

Each per-solver `*_install.md` (e.g. `acados_install.md`,
`aligator_install.md`) lists the steps for its solver. The
`install/` directory bundles shell scripts that automate the
non-pip-installable ones. For a pinned reproducible environment,
build the `Dockerfile` in this directory.

## Run

```bash
# Full multi-pass benchmark. Output lands in ./comparison_results/.
bash tests/comparison/run_all.sh

# Single (problem, solver) pair:
uv run python -m tests.comparison.run_benchmark \
    --problems cartpole --solvers lipa --out-dir /tmp/single
```

`run_benchmark.py` accepts `--problems`, `--solvers`, `--max-iter`,
`--tol`, `--backend {jax,casadi}` (for adapters that have multiple
model-evaluation backends), and `--solver-kwargs-json` for arbitrary
per-solver overrides.

`run_all.sh` orchestrates 6 passes that have to run in separate
processes (Aligator and CSQP can't coexist in one process due to a
pinocchio ABI conflict; analytical and MJX use disjoint solver sets;
LIPA-GPU vs LIPA-CPU need different `JAX_PLATFORMS`).

## Layout

```
adapters/              # one file per solver, all exposing the SolverAdapter interface
problems/              # one file per problem, all returning ProblemSpec
problem_spec.py        # ProblemSpec / SolverResult / KKT-residual evaluator
warm_starts.py         # shared warm-start helpers
casadi_jax_callback.py # shared per-stage CasADi-callback-wrapping-jax helper
sip_kkt_perm.py        # AMD-first KKT permutation for sip_python
run_benchmark.py       # single-pass CLI
run_all.sh             # 6-pass orchestrator (sets bit-stable XLA flags)
report.py              # markdown table + CSV + loglog plots
merge_reports.py       # combine per-pass CSVs into one merged report
install/               # one shell script per non-pip-installable solver
Dockerfile             # CPU-only pinned environment
```
