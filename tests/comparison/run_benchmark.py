"""Standalone CLI runner for the LIPA-vs-other-solvers comparison.

Examples:

    # Run cartpole with LIPA + IPOPT
    uv run --extra test --extra mpc-examples --group comparisons \\
        python -m tests.comparison.run_benchmark \\
        --problems cartpole --solvers lipa,ipopt \\
        --out-dir comparison_results/

    # Run all MJX-free problems with all available solvers
    uv run ... --problems cartpole,acrobot,quadpendulum

    # Run barrel_roll through LIPA + IPOPT (the only MJX-capable solvers)
    uv run ... --problems barrel_roll --solvers lipa,ipopt --max-iter 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from tests.comparison.adapters import all_adapter_names, get_adapter
from tests.comparison.problem_spec import SolverResult
from tests.comparison.problems import all_problem_names, make_problem
from tests.comparison.report import (
    render_convergence_plots,
    render_markdown,
    write_csv,
)


def _run_one(solver_name: str, problem_name: str, *, max_iter: int, tol: float, timeout_s: float | None, verbose: bool, solver_verbose: bool = False, backend: str | None = None, extra_kwargs: dict | None = None) -> SolverResult:
    if verbose:
        print(f"  -> {solver_name} on {problem_name} ...", flush=True)
    try:
        problem = make_problem(problem_name)
    except Exception as e:  # noqa: BLE001
        return SolverResult(
            solver_name=solver_name, problem_name=problem_name,
            iterations=0, solve_time_ms=0.0,
            final_cost=float("nan"), eq_violation_inf=float("nan"),
            ineq_violation_inf=float("nan"), success=False,
            notes=f"problem build failed: {e}",
        )

    # Per-solver kwargs for the adapter constructor.
    kwargs: dict = {"max_iter": max_iter}
    if solver_name in {"ipopt", "ipopt-mjx", "csqp", "aligator", "acados"}:
        kwargs["tol"] = tol
    if solver_name in {"ipopt", "ipopt-mjx"}:
        kwargs["timeout_s"] = timeout_s
        if solver_verbose:
            kwargs["print_level"] = 5
    if solver_name == "csqp" and solver_verbose:
        kwargs["with_callbacks"] = True
    # Backend selector for adapters that have a backend knob (csqp,
    # aligator, sip). Other adapters either are CasADi-only (ipopt,
    # acados, fatrop, ipopt-mjx, fatrop-mjx) or JAX-only (lipa, sip-mjx,
    # trajax) and don't accept the kwarg — TypeError fallback below
    # handles that gracefully.
    if backend is not None and solver_name in {"csqp", "aligator", "sip"}:
        kwargs["backend"] = backend

    # Free-form per-solver extras from --solver-kwargs-json. Used for
    # tuning sweeps where we want to drive adapter-specific parameters
    # (e.g. ipopt_extra_options, sip_extra_settings, psd_reg_delta)
    # without burning a round-trip on the adapter's signature.
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        adapter = get_adapter(solver_name, **kwargs)
    except TypeError:
        # adapter doesn't accept tol / timeout / backend — strip them and
        # retry with just max_iter so we don't crash for older adapter
        # signatures or unrecognized kwargs.
        # Preserve extra_kwargs across the retry so tuning sweeps don't
        # silently lose their settings just because the adapter doesn't
        # take ``tol``.
        retry_kwargs = {"max_iter": max_iter}
        if extra_kwargs:
            retry_kwargs.update(extra_kwargs)
        adapter = get_adapter(solver_name, **retry_kwargs)

    avail, reason = adapter.is_available()
    if not avail:
        return SolverResult(
            solver_name=solver_name, problem_name=problem_name,
            iterations=0, solve_time_ms=0.0,
            final_cost=float("nan"), eq_violation_inf=float("nan"),
            ineq_violation_inf=float("nan"), success=False,
            notes=f"unavailable: {reason}",
        )

    try:
        result = adapter.solve(problem)
    except Exception as e:  # noqa: BLE001
        return SolverResult(
            solver_name=solver_name, problem_name=problem_name,
            iterations=0, solve_time_ms=0.0,
            final_cost=float("nan"), eq_violation_inf=float("nan"),
            ineq_violation_inf=float("nan"), success=False,
            notes=f"adapter raised: {type(e).__name__}: {e}",
        )

    if verbose:
        print(
            f"     iters={result.iterations} time={result.solve_time_ms:.0f}ms "
            f"cost={result.final_cost:.4g} eq={result.eq_violation_inf:.2e} "
            f"ineq={result.ineq_violation_inf:.2e} ok={result.success}",
            flush=True,
        )
    return result


def _parse_csv(s: str | None, default: list[str]) -> list[str]:
    if s is None:
        return default
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--problems", type=str, default=None,
        help=f"Comma-separated problem names. Default: all known. Choices: {','.join(all_problem_names())}",
    )
    parser.add_argument(
        "--solvers", type=str, default=None,
        help=f"Comma-separated solver names. Default: all registered. Choices: {','.join(all_adapter_names())}",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--timeout-s", type=float, default=None,
                        help="Per-solve wall-clock cap (currently honored only by IPOPT).")
    parser.add_argument(
        "--solver-verbose", action="store_true",
        help="Pass print_level=5 to IPOPT and with_callbacks=True to CSQP "
             "so each solver emits its own per-iter log line.",
    )
    parser.add_argument(
        "--backend", type=str, default=None, choices=["jax", "casadi"],
        help="Per-solver backend selector (only csqp / aligator / sip "
             "accept this knob; ignored for other solvers). 'jax' "
             "preserves each adapter's historical default; 'casadi' "
             "uses casadi.Function-backed callbacks (faster on "
             "analytical problems, unavailable for MJX problems "
             "which have no casadi_builder). If omitted, each "
             "adapter's own default is used.",
    )
    parser.add_argument(
        "--solver-kwargs-json", type=str, default=None,
        help="JSON dict (or path to a JSON file) of extra kwargs to "
             "forward to EVERY solver adapter constructor. Used by "
             "tuning sweeps to inject adapter-specific knobs like "
             "ipopt_extra_options, sip_extra_settings, psd_reg_delta. "
             "Adapters that don't accept the kwarg will silently drop "
             "it via the existing TypeError fallback. Example: "
             "--solver-kwargs-json "
             "'{\"psd_reg_delta\": 0.01, \"sip_extra_settings\": "
             "{\"max_merit_callbacks\": 30}}'.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("comparison_results"))
    parser.add_argument(
        "--label-suffix", type=str, default="",
        help="Append this string to every solver name in the emitted CSV / "
             "report. Lets two passes of the same solver (e.g. lipa on GPU "
             "vs. lipa on CPU under JAX_PLATFORMS=cpu) coexist in a merged "
             "report under distinct labels (e.g. --label-suffix '-cpu').",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verbose = not args.quiet
    problems = _parse_csv(args.problems, all_problem_names())
    solvers = _parse_csv(args.solvers, all_adapter_names())

    # Parse --solver-kwargs-json: accept either an inline JSON string or
    # a path to a JSON file. Empty / unset -> no extras.
    extra_kwargs: dict = {}
    if args.solver_kwargs_json:
        s = args.solver_kwargs_json
        candidate = Path(s)
        try:
            if candidate.is_file():
                extra_kwargs = json.loads(candidate.read_text())
            else:
                extra_kwargs = json.loads(s)
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"ERROR: --solver-kwargs-json failed to parse: {e}",
                file=sys.stderr,
            )
            return 2
        if not isinstance(extra_kwargs, dict):
            print(
                f"ERROR: --solver-kwargs-json must decode to a dict, got "
                f"{type(extra_kwargs).__name__}",
                file=sys.stderr,
            )
            return 2
        if verbose:
            print(f"Extra solver kwargs: {extra_kwargs}")

    if verbose:
        print(f"Problems: {problems}")
        print(f"Solvers: {solvers}")
        print(f"max_iter={args.max_iter} tol={args.tol} timeout_s={args.timeout_s}")

    results: list[SolverResult] = []
    for problem in problems:
        if verbose:
            print(f"\n=== {problem} ===")
        for solver in solvers:
            r = _run_one(
                solver, problem,
                max_iter=args.max_iter,
                tol=args.tol,
                timeout_s=args.timeout_s,
                verbose=verbose,
                solver_verbose=args.solver_verbose,
                backend=args.backend,
                extra_kwargs=extra_kwargs,
            )
            if args.label_suffix:
                r = dataclasses.replace(
                    r, solver_name=r.solver_name + args.label_suffix,
                )
            results.append(r)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(results)
    (args.out_dir / "report.md").write_text(md)
    write_csv(results, args.out_dir / "results.csv")
    render_convergence_plots(results, args.out_dir / "plots")

    if verbose:
        print(f"\nWrote markdown -> {args.out_dir / 'report.md'}")
        print(f"Wrote CSV -> {args.out_dir / 'results.csv'}")
        print(f"Wrote plots -> {args.out_dir / 'plots'}/")

    # Exit nonzero if any solver reported failure (so CI-style consumers can detect).
    if any(not r.success for r in results):
        if verbose:
            print("\nSome solver/problem pairs did NOT succeed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
